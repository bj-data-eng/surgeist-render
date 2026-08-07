use super::{
    BackendErrorCode, Error, ImageBuffer, Result,
    layout::decode_padded_rows,
    lifecycle::{ReadbackOwner, ReadbackStagingMapState},
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[cfg(not(target_arch = "wasm32"))]
use std::{thread, time::Duration};

pub(super) enum ReadbackCompletionResult {
    Map(std::result::Result<(), wgpu::BufferAsyncError>),
    #[cfg(not(target_arch = "wasm32"))]
    PollFailure(wgpu::PollError),
}

enum ReadbackCompletionStatus {
    Pending { waker: Option<Waker> },
    Ready(ReadbackCompletionResult),
    Consumed,
    Canceled,
}

struct ReadbackCompletionState {
    status: ReadbackCompletionStatus,
}

pub(super) struct ReadbackCompletion {
    state: Mutex<ReadbackCompletionState>,
}

impl ReadbackCompletion {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(ReadbackCompletionState {
                status: ReadbackCompletionStatus::Pending { waker: None },
            }),
        }
    }

    pub(super) fn complete(&self, result: ReadbackCompletionResult) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::mem::replace(&mut state.status, ReadbackCompletionStatus::Consumed);
            match previous {
                ReadbackCompletionStatus::Pending { waker } => {
                    state.status = ReadbackCompletionStatus::Ready(result);
                    waker
                }
                terminal => {
                    state.status = terminal;
                    None
                }
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub(super) fn poll(&self, context: &mut Context<'_>) -> Poll<ReadbackCompletionResult> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut state.status {
            ReadbackCompletionStatus::Pending { waker } => {
                *waker = Some(context.waker().clone());
                Poll::Pending
            }
            ReadbackCompletionStatus::Ready(_) => {
                let ReadbackCompletionStatus::Ready(result) =
                    std::mem::replace(&mut state.status, ReadbackCompletionStatus::Consumed)
                else {
                    unreachable!("the ready readback result disappeared while locked")
                };
                Poll::Ready(result)
            }
            ReadbackCompletionStatus::Consumed | ReadbackCompletionStatus::Canceled => {
                panic!("a completed or canceled readback completion cell was polled")
            }
        }
    }

    pub(super) fn cancel(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(
            &state.status,
            ReadbackCompletionStatus::Pending { .. } | ReadbackCompletionStatus::Ready(_)
        ) {
            state.status = ReadbackCompletionStatus::Canceled;
        }
    }

    #[cfg(test)]
    pub(super) fn is_canceled_for_test(&self) -> bool {
        matches!(
            &self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status,
            ReadbackCompletionStatus::Canceled
        )
    }

    #[cfg(test)]
    pub(super) fn result_will_be_accepted_for_test(&self) -> bool {
        matches!(
            &self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status,
            ReadbackCompletionStatus::Pending { .. }
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn is_pending(&self) -> bool {
        matches!(
            &self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status,
            ReadbackCompletionStatus::Pending { .. }
        )
    }
}

pub(super) fn map_completion_callback(
    completion: Arc<ReadbackCompletion>,
    staging_map: Arc<ReadbackStagingMapState>,
) -> impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + 'static {
    move |result| {
        staging_map.map_callback_completed(result.is_ok());
        completion.complete(ReadbackCompletionResult::Map(result));
    }
}

pub(super) fn completion_result(result: ReadbackCompletionResult) -> Result<()> {
    match result {
        ReadbackCompletionResult::Map(Ok(())) => Ok(()),
        ReadbackCompletionResult::Map(Err(source)) => Err(Error::new(
            BackendErrorCode::ReadbackFailed,
            "failed to map the texture readback staging buffer",
        )
        .with_source(source)),
        #[cfg(not(target_arch = "wasm32"))]
        ReadbackCompletionResult::PollFailure(source) => Err(Error::new(
            BackendErrorCode::ReadbackFailed,
            "the readback helper received the wrong submission index",
        )
        .with_source(source)),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NativePollAction {
    Continue,
    Stop,
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn handle_native_poll_result(
    completion: &ReadbackCompletion,
    result: std::result::Result<wgpu::PollStatus, wgpu::PollError>,
) -> NativePollAction {
    match result {
        Ok(_) | Err(wgpu::PollError::Timeout) => {
            if completion.is_pending() {
                NativePollAction::Continue
            } else {
                NativePollAction::Stop
            }
        }
        Err(source @ wgpu::PollError::WrongSubmissionIndex(_, _)) => {
            completion.complete(ReadbackCompletionResult::PollFailure(source));
            NativePollAction::Stop
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_native_poll_helper(
    device: wgpu::Device,
    submission_index: wgpu::SubmissionIndex,
    completion: Arc<ReadbackCompletion>,
) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("surgeist-readback-poll".to_owned())
        .spawn(move || {
            while completion.is_pending() {
                let result = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index.clone()),
                    timeout: Some(Duration::from_millis(50)),
                });
                if handle_native_poll_result(completion.as_ref(), result) == NativePollAction::Stop
                {
                    break;
                }
            }
        })
}

pub(super) struct ReadbackMapFuture {
    completion: Arc<ReadbackCompletion>,
    owner: Option<ReadbackOwner>,
    #[cfg(not(target_arch = "wasm32"))]
    _poll_helper: thread::JoinHandle<()>,
}

impl ReadbackMapFuture {
    pub(super) fn start(
        mut owner: ReadbackOwner,
        device: wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
    ) -> Result<Self> {
        let completion = Arc::new(ReadbackCompletion::new());
        owner.map_pending();
        let staging_map = owner.staging_map();
        owner.staging().slice(owner.mapped_range()).map_async(
            wgpu::MapMode::Read,
            map_completion_callback(Arc::clone(&completion), staging_map),
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let poll_helper =
                match spawn_native_poll_helper(device, submission_index, Arc::clone(&completion)) {
                    Ok(poll_helper) => poll_helper,
                    Err(source) => {
                        completion.cancel();
                        owner.fail();
                        return Err(Error::new(
                            BackendErrorCode::ReadbackFailed,
                            "failed to start the native readback progress helper",
                        )
                        .with_source(source));
                    }
                };
            Ok(Self {
                completion,
                owner: Some(owner),
                _poll_helper: poll_helper,
            })
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (device, submission_index);
            Ok(Self {
                completion,
                owner: Some(owner),
            })
        }
    }
}

impl Future for ReadbackMapFuture {
    type Output = Result<ImageBuffer>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let completion = match this.completion.poll(context) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(completion) => completion,
        };
        let mut owner = this
            .owner
            .take()
            .expect("a readback map future must be polled to completion only once");
        if let Err(error) = completion_result(completion) {
            owner.fail();
            return Poll::Ready(Err(error));
        }

        owner.mapped();
        let decoded = {
            let mapped = owner
                .staging()
                .slice(owner.mapped_range())
                .get_mapped_range();
            decode_padded_rows(&owner.layout, &mapped)
        };
        match decoded {
            Ok(rgba) => Poll::Ready(owner.publish_mapped(rgba)),
            Err(error) => {
                owner.fail();
                Poll::Ready(Err(error))
            }
        }
    }
}

impl Drop for ReadbackMapFuture {
    fn drop(&mut self) {
        if let Some(mut owner) = self.owner.take() {
            self.completion.cancel();
            owner.cancel();
        }
    }
}
