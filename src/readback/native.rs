#[cfg(all(test, not(target_arch = "wasm32")))]
use super::test_support::ReadbackStagingDispositionForTest;
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

#[cfg(all(test, not(target_arch = "wasm32")))]
use std::{
    cell::RefCell,
    sync::{Condvar, Weak},
    time::Instant,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST: RefCell<Option<NativeReadbackObservationForTest>> = const { RefCell::new(None) };
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeReadbackPhaseForTest {
    Allocated,
    CopySubmitted,
    MapPending,
    Mapped,
    PublishedBytes,
    Failed,
    Canceled,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Debug, Default)]
struct NativeReadbackObservationStateForTest {
    phase: Option<NativeReadbackPhaseForTest>,
    phase_history: Vec<NativeReadbackPhaseForTest>,
    submission_index: Option<wgpu::SubmissionIndex>,
    staging_map: Option<Weak<ReadbackStagingMapState>>,
    helper_started: usize,
    helper_exited: usize,
    callback_invoked: usize,
    callback_released: usize,
    callback_succeeded: Option<bool>,
    accepted_results: usize,
    discarded_results: usize,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct NativeReadbackConditionForTest {
    state: Mutex<NativeReadbackObservationStateForTest>,
    changed: Condvar,
    hold_helper_until_canceled: bool,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone)]
pub(crate) struct NativeReadbackObservationForTest {
    condition: Arc<NativeReadbackConditionForTest>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl NativeReadbackObservationForTest {
    fn new(hold_helper_until_canceled: bool) -> Self {
        Self {
            condition: Arc::new(NativeReadbackConditionForTest {
                state: Mutex::new(NativeReadbackObservationStateForTest::default()),
                changed: Condvar::new(),
                hold_helper_until_canceled,
            }),
        }
    }

    pub(super) fn attach_staging(&self, staging_map: &Arc<ReadbackStagingMapState>) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            state.phase.is_none(),
            "one native readback observation may attach to only one real readback"
        );
        state.staging_map = Some(Arc::downgrade(staging_map));
        Self::record_phase_locked(&mut state, NativeReadbackPhaseForTest::Allocated);
        self.condition.changed.notify_all();
    }

    pub(super) fn record_copy_submitted(&self, submission_index: &wgpu::SubmissionIndex) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.submission_index = Some(submission_index.clone());
        Self::record_phase_locked(&mut state, NativeReadbackPhaseForTest::CopySubmitted);
        self.condition.changed.notify_all();
    }

    pub(super) fn record_phase(&self, phase: NativeReadbackPhaseForTest) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::record_phase_locked(&mut state, phase);
        self.condition.changed.notify_all();
    }

    fn record_phase_locked(
        state: &mut NativeReadbackObservationStateForTest,
        phase: NativeReadbackPhaseForTest,
    ) {
        state.phase = Some(phase);
        state.phase_history.push(phase);
    }

    fn helper_lifetime(&self) -> NativeReadbackHelperLifetimeForTest {
        {
            let mut state = self
                .condition
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.helper_started = state.helper_started.saturating_add(1);
            self.condition.changed.notify_all();
        }
        NativeReadbackHelperLifetimeForTest {
            observation: self.clone(),
        }
    }

    fn callback_lifetime(&self) -> NativeReadbackCallbackLifetimeForTest {
        NativeReadbackCallbackLifetimeForTest {
            observation: self.clone(),
        }
    }

    fn record_callback_invoked(&self, succeeded: bool) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.callback_invoked = state.callback_invoked.saturating_add(1);
        state.callback_succeeded = Some(succeeded);
        self.condition.changed.notify_all();
    }

    fn record_completion_result(&self, accepted: bool) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if accepted {
            state.accepted_results = state.accepted_results.saturating_add(1);
        } else {
            state.discarded_results = state.discarded_results.saturating_add(1);
        }
        self.condition.changed.notify_all();
    }

    fn hold_helper_until_canceled(&self) -> bool {
        if !self.condition.hold_helper_until_canceled {
            return false;
        }
        let state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _state = self
            .condition
            .changed
            .wait_while(state, |state| {
                state.phase != Some(NativeReadbackPhaseForTest::Canceled)
            })
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        true
    }

    pub(crate) fn snapshot_for_test(&self) -> NativeReadbackObservationSnapshotForTest {
        let state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let staging_map = state.staging_map.as_ref().and_then(Weak::upgrade);
        NativeReadbackObservationSnapshotForTest {
            phase: state.phase,
            phase_history: state.phase_history.clone(),
            submission_index: state.submission_index.clone(),
            staging_disposition: staging_map
                .as_ref()
                .map(|map| ReadbackStagingDispositionForTest::from(map.disposition_for_test())),
            staging_state_dropped: state.staging_map.is_some() && staging_map.is_none(),
            helper_started: state.helper_started,
            helper_exited: state.helper_exited,
            callback_invoked: state.callback_invoked,
            callback_released: state.callback_released,
            callback_succeeded: state.callback_succeeded,
            accepted_results: state.accepted_results,
            discarded_results: state.discarded_results,
        }
    }

    pub(crate) fn wait_for_published_cleanup_for_test(&self, deadline: Instant) -> bool {
        self.wait_until(deadline, |state| {
            state.phase == Some(NativeReadbackPhaseForTest::PublishedBytes)
                && state.helper_started == 1
                && state.helper_exited == 1
                && state.callback_invoked == 1
                && state.callback_released == 1
                && state.accepted_results == 1
                && state
                    .staging_map
                    .as_ref()
                    .is_some_and(|staging| staging.upgrade().is_none())
        })
    }

    pub(crate) fn wait_for_canceled_helper_cleanup_for_test(&self, deadline: Instant) -> bool {
        self.wait_until(deadline, |state| {
            state.phase == Some(NativeReadbackPhaseForTest::Canceled)
                && state.helper_started == 1
                && state.helper_exited == 1
        })
    }

    pub(crate) fn wait_for_late_callback_cleanup_for_test(&self, deadline: Instant) -> bool {
        self.wait_until(deadline, |state| {
            state.phase == Some(NativeReadbackPhaseForTest::Canceled)
                && state.callback_invoked == 1
                && state.callback_released == 1
                && state.accepted_results == 0
                && state.discarded_results == 1
                && state
                    .staging_map
                    .as_ref()
                    .is_some_and(|staging| staging.upgrade().is_none())
        })
    }

    fn wait_until(
        &self,
        deadline: Instant,
        ready: impl Fn(&NativeReadbackObservationStateForTest) -> bool,
    ) -> bool {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if ready(&state) {
                return true;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, timeout) = self
                .condition
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if timeout.timed_out() && !ready(&state) {
                return false;
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct NativeReadbackHelperLifetimeForTest {
    observation: NativeReadbackObservationForTest,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl Drop for NativeReadbackHelperLifetimeForTest {
    fn drop(&mut self) {
        let mut state = self
            .observation
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.helper_exited = state.helper_exited.saturating_add(1);
        self.observation.condition.changed.notify_all();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct NativeReadbackCallbackLifetimeForTest {
    observation: NativeReadbackObservationForTest,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl Drop for NativeReadbackCallbackLifetimeForTest {
    fn drop(&mut self) {
        let mut state = self
            .observation
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.callback_released = state.callback_released.saturating_add(1);
        self.observation.condition.changed.notify_all();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Debug)]
pub(crate) struct NativeReadbackObservationSnapshotForTest {
    phase: Option<NativeReadbackPhaseForTest>,
    phase_history: Vec<NativeReadbackPhaseForTest>,
    submission_index: Option<wgpu::SubmissionIndex>,
    staging_disposition: Option<ReadbackStagingDispositionForTest>,
    staging_state_dropped: bool,
    helper_started: usize,
    helper_exited: usize,
    callback_invoked: usize,
    callback_released: usize,
    callback_succeeded: Option<bool>,
    accepted_results: usize,
    discarded_results: usize,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl NativeReadbackObservationSnapshotForTest {
    pub(crate) const fn phase_for_test(&self) -> Option<NativeReadbackPhaseForTest> {
        self.phase
    }

    pub(crate) fn phase_history_for_test(&self) -> &[NativeReadbackPhaseForTest] {
        &self.phase_history
    }

    pub(crate) fn submission_index_for_test(&self) -> Option<wgpu::SubmissionIndex> {
        self.submission_index.clone()
    }

    pub(crate) const fn staging_disposition_for_test(
        &self,
    ) -> Option<ReadbackStagingDispositionForTest> {
        self.staging_disposition
    }

    pub(crate) const fn staging_state_dropped_for_test(&self) -> bool {
        self.staging_state_dropped
    }

    pub(crate) const fn helper_counts_for_test(&self) -> (usize, usize) {
        (self.helper_started, self.helper_exited)
    }

    pub(crate) const fn callback_counts_for_test(&self) -> (usize, usize) {
        (self.callback_invoked, self.callback_released)
    }

    pub(crate) const fn callback_succeeded_for_test(&self) -> Option<bool> {
        self.callback_succeeded
    }

    pub(crate) const fn completion_counts_for_test(&self) -> (usize, usize) {
        (self.accepted_results, self.discarded_results)
    }
}

/// Installs one fresh private observation on the next native production readback.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) struct ScopedNativeReadbackObservationForTest {
    observation: NativeReadbackObservationForTest,
    previous: Option<NativeReadbackObservationForTest>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl ScopedNativeReadbackObservationForTest {
    pub(crate) fn begin() -> Self {
        Self::install(false)
    }

    pub(crate) fn hold_helper_until_canceled() -> Self {
        Self::install(true)
    }

    fn install(hold_helper_until_canceled: bool) -> Self {
        let observation = NativeReadbackObservationForTest::new(hold_helper_until_canceled);
        let previous = ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(observation.clone())));
        Self {
            observation,
            previous,
        }
    }

    pub(crate) fn observation_for_test(&self) -> NativeReadbackObservationForTest {
        self.observation.clone()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl Drop for ScopedNativeReadbackObservationForTest {
    fn drop(&mut self) {
        ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn take_native_readback_observation_for_test() -> Option<NativeReadbackObservationForTest>
{
    ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST.with(|active| active.borrow_mut().take())
}

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
    #[cfg(all(test, not(target_arch = "wasm32")))]
    observation: Option<NativeReadbackObservationForTest>,
}

impl ReadbackCompletion {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(ReadbackCompletionState {
                status: ReadbackCompletionStatus::Pending { waker: None },
            }),
            #[cfg(all(test, not(target_arch = "wasm32")))]
            observation: None,
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn with_observation_for_test(
        mut self,
        observation: Option<NativeReadbackObservationForTest>,
    ) -> Self {
        self.observation = observation;
        self
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
                    #[cfg(all(test, not(target_arch = "wasm32")))]
                    if let Some(observation) = &self.observation {
                        observation.record_completion_result(true);
                    }
                    waker
                }
                terminal => {
                    state.status = terminal;
                    #[cfg(all(test, not(target_arch = "wasm32")))]
                    if let Some(observation) = &self.observation {
                        observation.record_completion_result(false);
                    }
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
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let callback_lifetime = completion
        .observation
        .as_ref()
        .map(NativeReadbackObservationForTest::callback_lifetime);
    move |result| {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(callback_lifetime) = &callback_lifetime {
            callback_lifetime
                .observation
                .record_callback_invoked(result.is_ok());
        }
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
    #[cfg(test)]
    let observation = completion.observation.clone();
    thread::Builder::new()
        .name("surgeist-readback-poll".to_owned())
        .spawn(move || {
            #[cfg(test)]
            let _helper_lifetime = observation
                .as_ref()
                .map(NativeReadbackObservationForTest::helper_lifetime);
            #[cfg(test)]
            if observation
                .as_ref()
                .is_some_and(NativeReadbackObservationForTest::hold_helper_until_canceled)
            {
                return;
            }
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
        #[cfg(all(test, not(target_arch = "wasm32")))] observation: Option<
            NativeReadbackObservationForTest,
        >,
    ) -> Result<Self> {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        let completion =
            Arc::new(ReadbackCompletion::new().with_observation_for_test(observation.clone()));
        #[cfg(not(all(test, not(target_arch = "wasm32"))))]
        let completion = Arc::new(ReadbackCompletion::new());
        owner.map_pending();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &observation {
            observation.record_phase(NativeReadbackPhaseForTest::MapPending);
        }
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
                        #[cfg(all(test, not(target_arch = "wasm32")))]
                        if let Some(observation) = &observation {
                            observation.record_phase(NativeReadbackPhaseForTest::Failed);
                        }
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
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &this.completion.observation {
                observation.record_phase(NativeReadbackPhaseForTest::Failed);
            }
            return Poll::Ready(Err(error));
        }

        owner.mapped();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &this.completion.observation {
            observation.record_phase(NativeReadbackPhaseForTest::Mapped);
        }
        let decoded = {
            let mapped = owner
                .staging()
                .slice(owner.mapped_range())
                .get_mapped_range();
            decode_padded_rows(&owner.layout, &mapped)
        };
        match decoded {
            Ok(rgba) => {
                let result = owner.publish_mapped(rgba);
                #[cfg(all(test, not(target_arch = "wasm32")))]
                if let Some(observation) = &this.completion.observation {
                    observation.record_phase(if result.is_ok() {
                        NativeReadbackPhaseForTest::PublishedBytes
                    } else {
                        NativeReadbackPhaseForTest::Failed
                    });
                }
                Poll::Ready(result)
            }
            Err(error) => {
                owner.fail();
                #[cfg(all(test, not(target_arch = "wasm32")))]
                if let Some(observation) = &this.completion.observation {
                    observation.record_phase(NativeReadbackPhaseForTest::Failed);
                }
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
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &self.completion.observation {
                observation.record_phase(NativeReadbackPhaseForTest::Canceled);
            }
        }
    }
}
