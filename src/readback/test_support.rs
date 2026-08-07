#[cfg(not(target_arch = "wasm32"))]
use super::native::{NativePollAction, handle_native_poll_result};
use super::{
    BackendErrorCode, Error, ImageBuffer, PhysicalSize, Result,
    layout::{ReadbackLayout, decode_padded_rows},
    lifecycle::{
        ReadbackLifecycle, ReadbackPhase, ReadbackStagingCleanupAction, ReadbackStagingDisposition,
        ReadbackStagingMapState,
    },
    native::{
        ReadbackCompletion, ReadbackCompletionResult, completion_result, map_completion_callback,
    },
};
use std::{
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackPhaseForTest {
    Allocated,
    CopySubmitted { submission_index: u64 },
    MapPending,
    Mapped,
    PublishedBytes,
    Failed,
    Canceled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackStagingDispositionForTest {
    Idle,
    MapPending,
    MappedActive,
    Released,
}

impl From<ReadbackStagingDisposition> for ReadbackStagingDispositionForTest {
    fn from(disposition: ReadbackStagingDisposition) -> Self {
        match disposition {
            ReadbackStagingDisposition::Idle => Self::Idle,
            ReadbackStagingDisposition::MapPending => Self::MapPending,
            ReadbackStagingDisposition::MappedActive => Self::MappedActive,
            ReadbackStagingDisposition::Released => Self::Released,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackCleanupEventForTest {
    MappedViewDropped,
    StagingUnmapped,
    StagingDropped,
    PublishedBytes,
}

#[derive(Default)]
struct ReadbackStateMachineObservationStateForTest {
    terminal_phase: Option<ReadbackPhaseForTest>,
    cleanup_events: Vec<ReadbackCleanupEventForTest>,
}

#[derive(Clone)]
pub(crate) struct ReadbackStateMachineObservationForTest {
    state: Arc<Mutex<ReadbackStateMachineObservationStateForTest>>,
    staging_map: Arc<ReadbackStagingMapState>,
}

impl Default for ReadbackStateMachineObservationForTest {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(
                ReadbackStateMachineObservationStateForTest::default(),
            )),
            staging_map: Arc::new(ReadbackStagingMapState::idle()),
        }
    }
}

impl ReadbackStateMachineObservationForTest {
    pub(crate) fn terminal_phase_for_test(&self) -> Option<ReadbackPhaseForTest> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .terminal_phase
    }

    pub(crate) fn staging_disposition_for_test(&self) -> ReadbackStagingDispositionForTest {
        self.staging_map.disposition_for_test().into()
    }

    pub(crate) fn cleanup_events_for_test(&self) -> Vec<ReadbackCleanupEventForTest> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cleanup_events
            .clone()
    }
}

pub(crate) struct ReadbackStateMachineForTest {
    lifecycle: ReadbackLifecycle<u64>,
    observation: ReadbackStateMachineObservationForTest,
}

impl ReadbackStateMachineForTest {
    pub(crate) fn allocated() -> Self {
        Self {
            lifecycle: ReadbackLifecycle::allocated(),
            observation: ReadbackStateMachineObservationForTest::default(),
        }
    }

    pub(crate) fn phase_for_test(&self) -> ReadbackPhaseForTest {
        match &self.lifecycle.phase {
            ReadbackPhase::Allocated => ReadbackPhaseForTest::Allocated,
            ReadbackPhase::CopySubmitted { submission_index } => {
                ReadbackPhaseForTest::CopySubmitted {
                    submission_index: *submission_index,
                }
            }
            ReadbackPhase::MapPending => ReadbackPhaseForTest::MapPending,
            ReadbackPhase::Mapped => ReadbackPhaseForTest::Mapped,
            ReadbackPhase::PublishedBytes => ReadbackPhaseForTest::PublishedBytes,
            ReadbackPhase::Failed => ReadbackPhaseForTest::Failed,
            ReadbackPhase::Canceled => ReadbackPhaseForTest::Canceled,
        }
    }

    pub(crate) fn copy_submitted_for_test(&mut self, submission_index: u64) {
        self.lifecycle.copy_submitted(submission_index);
    }

    pub(crate) fn map_pending_for_test(&mut self) {
        self.lifecycle.map_pending();
        self.observation.staging_map.map_pending();
    }

    pub(crate) fn map_callback_succeeded_for_test(&self) {
        self.observation.staging_map.map_callback_completed(true);
    }

    pub(crate) fn map_callback_failed_for_test(&self) {
        self.observation.staging_map.map_callback_completed(false);
    }

    pub(crate) fn mapped_for_test(&mut self) {
        self.observation.staging_map.assert_mapped_active();
        self.lifecycle.mapped();
    }

    pub(crate) fn fail_for_test(&mut self) {
        if self.lifecycle.fail() {
            self.record_terminal_phase_for_test();
            self.release_staging_for_test();
        }
    }

    pub(crate) fn cancel_for_test(&mut self) {
        if self.lifecycle.cancel() {
            self.record_terminal_phase_for_test();
            self.release_staging_for_test();
        }
    }

    pub(crate) fn finish_mapped_for_test(
        &mut self,
        size: PhysicalSize,
        mapped: &[u8],
    ) -> Result<ImageBuffer> {
        assert!(matches!(&self.lifecycle.phase, ReadbackPhase::Mapped));
        let layout = ReadbackLayout::try_new(size)?;
        let decoded = decode_padded_rows(&layout, mapped);
        self.observation
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cleanup_events
            .push(ReadbackCleanupEventForTest::MappedViewDropped);
        let rgba = match decoded {
            Ok(rgba) => rgba,
            Err(error) => {
                self.fail_for_test();
                return Err(error);
            }
        };
        self.release_staging_for_test();
        match ImageBuffer::try_new(size, rgba) {
            Ok(image) => {
                self.lifecycle.published();
                self.record_terminal_phase_for_test();
                self.observation
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .cleanup_events
                    .push(ReadbackCleanupEventForTest::PublishedBytes);
                Ok(image)
            }
            Err(source) => {
                self.lifecycle.fail();
                self.record_terminal_phase_for_test();
                Err(Error::new(
                    BackendErrorCode::ReadbackFailed,
                    "decoded readback bytes did not form a valid RGBA8 image",
                )
                .with_source(source))
            }
        }
    }

    pub(crate) fn staging_disposition_for_test(&self) -> ReadbackStagingDispositionForTest {
        self.observation.staging_disposition_for_test()
    }

    pub(crate) fn cleanup_events_for_test(&self) -> Vec<ReadbackCleanupEventForTest> {
        self.observation.cleanup_events_for_test()
    }

    pub(crate) fn observation_for_test(&self) -> ReadbackStateMachineObservationForTest {
        self.observation.clone()
    }

    fn release_staging_for_test(&mut self) {
        let action = self.observation.staging_map.take_cleanup_action();
        let mut observation = self
            .observation
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match action {
            ReadbackStagingCleanupAction::Drop => {
                observation
                    .cleanup_events
                    .push(ReadbackCleanupEventForTest::StagingDropped);
            }
            ReadbackStagingCleanupAction::UnmapThenDrop => {
                observation
                    .cleanup_events
                    .push(ReadbackCleanupEventForTest::StagingUnmapped);
                observation
                    .cleanup_events
                    .push(ReadbackCleanupEventForTest::StagingDropped);
            }
            ReadbackStagingCleanupAction::None => {}
        }
    }

    fn record_terminal_phase_for_test(&self) {
        self.observation
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .terminal_phase = Some(self.phase_for_test());
    }
}

impl Drop for ReadbackStateMachineForTest {
    fn drop(&mut self) {
        self.cancel_for_test();
    }
}

#[derive(Default)]
struct ReadbackCompletionCountsForTest {
    accepted: usize,
    discarded: usize,
}

pub(crate) struct ReadbackCompletionForTest {
    completion: Arc<ReadbackCompletion>,
    staging_map: Arc<ReadbackStagingMapState>,
    counts: Mutex<ReadbackCompletionCountsForTest>,
}

impl ReadbackCompletionForTest {
    pub(crate) fn new() -> Self {
        let staging_map = Arc::new(ReadbackStagingMapState::idle());
        staging_map.map_pending();
        Self {
            completion: Arc::new(ReadbackCompletion::new()),
            staging_map,
            counts: Mutex::new(ReadbackCompletionCountsForTest::default()),
        }
    }

    pub(crate) fn poll_for_test(&self, context: &mut Context<'_>) -> Poll<Result<()>> {
        self.completion.poll(context).map(completion_result)
    }

    pub(crate) fn invoke_map_callback_for_test(
        &self,
        result: std::result::Result<(), wgpu::BufferAsyncError>,
    ) {
        let accepted = self.result_will_be_accepted_for_test();
        map_completion_callback(Arc::clone(&self.completion), Arc::clone(&self.staging_map))(
            result,
        );
        self.record_result_for_test(accepted);
    }

    pub(crate) fn deliver_late_map_result_for_test(
        &self,
        result: std::result::Result<(), wgpu::BufferAsyncError>,
    ) {
        let accepted = self.result_will_be_accepted_for_test();
        self.completion
            .complete(ReadbackCompletionResult::Map(result));
        self.record_result_for_test(accepted);
    }

    pub(crate) fn cancel_for_test(&self) {
        self.completion.cancel();
    }

    pub(crate) fn is_canceled_for_test(&self) -> bool {
        self.completion.is_canceled_for_test()
    }

    pub(crate) fn accepted_result_count_for_test(&self) -> usize {
        self.counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .accepted
    }

    pub(crate) fn discarded_result_count_for_test(&self) -> usize {
        self.counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .discarded
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn timeout_slice_for_test(&self) -> bool {
        handle_native_poll_result(self.completion.as_ref(), Err(wgpu::PollError::Timeout))
            == NativePollAction::Continue
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wrong_submission_index_for_test(&self, requested: u64, successful: u64) {
        let accepted = self.result_will_be_accepted_for_test();
        let action = handle_native_poll_result(
            self.completion.as_ref(),
            Err(wgpu::PollError::WrongSubmissionIndex(requested, successful)),
        );
        self.record_result_for_test(accepted);
        assert_eq!(action, NativePollAction::Stop);
    }

    fn result_will_be_accepted_for_test(&self) -> bool {
        self.completion.result_will_be_accepted_for_test()
    }

    fn record_result_for_test(&self, accepted: bool) {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if accepted {
            counts.accepted = counts.accepted.saturating_add(1);
        } else {
            counts.discarded = counts.discarded.saturating_add(1);
        }
    }
}
