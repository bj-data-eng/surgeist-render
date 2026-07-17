use super::{
    BackendErrorCode, Error, ImageBuffer, PhysicalSize, Result, RuntimeOperation,
    backend::{Backend, DeviceSlotIdentity},
    gpu_transaction::GpuOperationStage,
};
use std::{
    future::Future,
    ops::Range,
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

const RGBA8_BYTES_PER_PIXEL: u64 = 4;
const COPY_BYTES_PER_ROW_ALIGNMENT: u64 = 256;

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST: RefCell<Option<NativeReadbackObservationForTest>> = const { RefCell::new(None) };
}

struct ReadbackLayout {
    width: u32,
    height: u32,
    row_bytes: usize,
    padded_bytes_per_row: u32,
    buffer_size: u64,
    decoded_len: usize,
    mapped_range: ValidatedMappedRange,
}

impl ReadbackLayout {
    fn try_new(size: PhysicalSize) -> Result<Self> {
        let width = size.width();
        let height = size.height();
        let row_bytes_u64 = u64::from(width)
            .checked_mul(RGBA8_BYTES_PER_PIXEL)
            .ok_or_else(|| readback_failed("readback row byte count overflowed"))?;
        let padded_bytes_per_row_u64 = row_bytes_u64
            .checked_add(COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            .map(|bytes| bytes / COPY_BYTES_PER_ROW_ALIGNMENT * COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| readback_failed("aligned readback row byte count overflowed"))?;
        let padded_bytes_per_row = u32::try_from(padded_bytes_per_row_u64)
            .map_err(|_| readback_failed("aligned readback row byte count exceeds WGPU limits"))?;
        let buffer_size = padded_bytes_per_row_u64
            .checked_mul(u64::from(height))
            .ok_or_else(|| readback_failed("readback staging buffer size overflowed"))?;
        let row_bytes = usize::try_from(row_bytes_u64)
            .map_err(|_| readback_failed("readback row byte count exceeds addressable memory"))?;
        let decoded_len = row_bytes
            .checked_mul(
                usize::try_from(height)
                    .map_err(|_| readback_failed("readback height exceeds addressable memory"))?,
            )
            .ok_or_else(|| readback_failed("decoded readback byte count overflowed"))?;
        let mapped_range = ValidatedMappedRange::try_new(buffer_size)?;
        Ok(Self {
            width,
            height,
            row_bytes,
            padded_bytes_per_row,
            buffer_size,
            decoded_len,
            mapped_range,
        })
    }
}

#[derive(Clone)]
struct ValidatedMappedRange {
    bytes: Range<wgpu::BufferAddress>,
}

impl ValidatedMappedRange {
    fn try_new(buffer_size: u64) -> Result<Self> {
        let bytes = 0..buffer_size;
        let length = bytes
            .end
            .checked_sub(bytes.start)
            .ok_or_else(|| readback_failed("readback mapped range was reversed"))?;
        if length == 0 {
            return Err(readback_failed("readback mapped range must be nonempty"));
        }
        if bytes.start % wgpu::MAP_ALIGNMENT != 0 {
            return Err(readback_failed(
                "readback mapped range offset was not map-aligned",
            ));
        }
        if length % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(readback_failed(
                "readback mapped range length was not four-byte aligned",
            ));
        }
        Ok(Self { bytes })
    }

    fn bytes(&self) -> Range<wgpu::BufferAddress> {
        self.bytes.clone()
    }
}

enum ReadbackPhase<I> {
    Allocated,
    CopySubmitted { submission_index: I },
    MapPending,
    Mapped,
    PublishedBytes,
    Failed,
    Canceled,
}

struct ReadbackLifecycle<I> {
    phase: ReadbackPhase<I>,
}

impl<I> ReadbackLifecycle<I> {
    const fn allocated() -> Self {
        Self {
            phase: ReadbackPhase::Allocated,
        }
    }

    fn copy_submitted(&mut self, submission_index: I) {
        assert!(
            matches!(&self.phase, ReadbackPhase::Allocated),
            "readback copy submission must follow allocation"
        );
        self.phase = ReadbackPhase::CopySubmitted { submission_index };
    }

    fn map_pending(&mut self) {
        let previous = std::mem::replace(&mut self.phase, ReadbackPhase::MapPending);
        let ReadbackPhase::CopySubmitted { submission_index } = previous else {
            self.phase = previous;
            panic!("readback mapping must follow copy submission");
        };
        drop(submission_index);
    }

    fn mapped(&mut self) {
        assert!(
            matches!(&self.phase, ReadbackPhase::MapPending),
            "mapped readback bytes require callback success"
        );
        self.phase = ReadbackPhase::Mapped;
    }

    fn published(&mut self) {
        assert!(
            matches!(&self.phase, ReadbackPhase::Mapped),
            "readback bytes may publish only from the mapped phase"
        );
        self.phase = ReadbackPhase::PublishedBytes;
    }

    fn fail(&mut self) -> bool {
        if self.is_uncertain() {
            self.phase = ReadbackPhase::Failed;
            true
        } else {
            false
        }
    }

    fn cancel(&mut self) -> bool {
        if self.is_uncertain() {
            self.phase = ReadbackPhase::Canceled;
            true
        } else {
            false
        }
    }

    const fn is_uncertain(&self) -> bool {
        matches!(
            &self.phase,
            ReadbackPhase::Allocated
                | ReadbackPhase::CopySubmitted { .. }
                | ReadbackPhase::MapPending
                | ReadbackPhase::Mapped
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadbackStagingDisposition {
    Idle,
    MapPending,
    MappedActive,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadbackStagingCleanupAction {
    Drop,
    UnmapThenDrop,
    None,
}

struct ReadbackStagingMapState {
    disposition: Mutex<ReadbackStagingDisposition>,
}

impl ReadbackStagingMapState {
    fn idle() -> Self {
        Self {
            disposition: Mutex::new(ReadbackStagingDisposition::Idle),
        }
    }

    fn map_pending(&self) {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            *disposition,
            ReadbackStagingDisposition::Idle,
            "a readback map request requires known-idle staging"
        );
        *disposition = ReadbackStagingDisposition::MapPending;
    }

    fn map_callback_completed(&self, succeeded: bool) {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *disposition {
            ReadbackStagingDisposition::MapPending => {
                *disposition = if succeeded {
                    ReadbackStagingDisposition::MappedActive
                } else {
                    ReadbackStagingDisposition::Idle
                };
            }
            ReadbackStagingDisposition::Released => {}
            ReadbackStagingDisposition::Idle | ReadbackStagingDisposition::MappedActive => {
                panic!("a readback map request callback may complete only once")
            }
        }
    }

    fn assert_mapped_active(&self) {
        assert_eq!(
            *self
                .disposition
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ReadbackStagingDisposition::MappedActive,
            "map callback success must make staging actively mapped"
        );
    }

    fn take_cleanup_action(&self) -> ReadbackStagingCleanupAction {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match std::mem::replace(&mut *disposition, ReadbackStagingDisposition::Released) {
            ReadbackStagingDisposition::Idle => ReadbackStagingCleanupAction::Drop,
            ReadbackStagingDisposition::MapPending | ReadbackStagingDisposition::MappedActive => {
                ReadbackStagingCleanupAction::UnmapThenDrop
            }
            ReadbackStagingDisposition::Released => ReadbackStagingCleanupAction::None,
        }
    }

    #[cfg(test)]
    fn disposition_for_test(&self) -> ReadbackStagingDisposition {
        *self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
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

    fn attach_staging(&self, staging_map: &Arc<ReadbackStagingMapState>) {
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

    fn record_copy_submitted(&self, submission_index: &wgpu::SubmissionIndex) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.submission_index = Some(submission_index.clone());
        Self::record_phase_locked(&mut state, NativeReadbackPhaseForTest::CopySubmitted);
        self.condition.changed.notify_all();
    }

    fn record_phase(&self, phase: NativeReadbackPhaseForTest) {
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

    fn record_completion_counts(&self, accepted_results: usize, discarded_results: usize) {
        let mut state = self
            .condition
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.accepted_results = accepted_results;
        state.discarded_results = discarded_results;
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
fn take_native_readback_observation_for_test() -> Option<NativeReadbackObservationForTest> {
    ACTIVE_NATIVE_READBACK_OBSERVATION_FOR_TEST.with(|active| active.borrow_mut().take())
}

struct ReadbackOwner {
    lifecycle: ReadbackLifecycle<wgpu::SubmissionIndex>,
    staging: Option<wgpu::Buffer>,
    staging_map: Arc<ReadbackStagingMapState>,
    layout: ReadbackLayout,
    physical_size: PhysicalSize,
    #[cfg(all(test, not(target_arch = "wasm32")))]
    observation: Option<NativeReadbackObservationForTest>,
}

impl ReadbackOwner {
    fn allocated(
        staging: wgpu::Buffer,
        layout: ReadbackLayout,
        physical_size: PhysicalSize,
    ) -> Self {
        Self {
            lifecycle: ReadbackLifecycle::allocated(),
            staging: Some(staging),
            staging_map: Arc::new(ReadbackStagingMapState::idle()),
            layout,
            physical_size,
            #[cfg(all(test, not(target_arch = "wasm32")))]
            observation: None,
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn attach_observation_for_test(&mut self, observation: NativeReadbackObservationForTest) {
        observation.attach_staging(&self.staging_map);
        self.observation = Some(observation);
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn observation_for_test(&self) -> Option<NativeReadbackObservationForTest> {
        self.observation.clone()
    }

    fn staging(&self) -> &wgpu::Buffer {
        self.staging
            .as_ref()
            .expect("an uncertain readback phase must own its staging buffer")
    }

    fn copy_submitted(&mut self, submission_index: wgpu::SubmissionIndex) {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &self.observation {
            observation.record_copy_submitted(&submission_index);
        }
        self.lifecycle.copy_submitted(submission_index);
    }

    fn map_pending(&mut self) {
        self.lifecycle.map_pending();
        self.staging_map.map_pending();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &self.observation {
            observation.record_phase(NativeReadbackPhaseForTest::MapPending);
        }
    }

    fn mapped(&mut self) {
        self.staging_map.assert_mapped_active();
        self.lifecycle.mapped();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &self.observation {
            observation.record_phase(NativeReadbackPhaseForTest::Mapped);
        }
    }

    fn staging_map(&self) -> Arc<ReadbackStagingMapState> {
        Arc::clone(&self.staging_map)
    }

    fn mapped_range(&self) -> Range<wgpu::BufferAddress> {
        self.layout.mapped_range.bytes()
    }

    fn fail(&mut self) {
        if self.lifecycle.fail() {
            self.release_staging();
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &self.observation {
                observation.record_phase(NativeReadbackPhaseForTest::Failed);
            }
        }
    }

    fn cancel(&mut self) {
        if self.lifecycle.cancel() {
            self.release_staging();
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &self.observation {
                observation.record_phase(NativeReadbackPhaseForTest::Canceled);
            }
        }
    }

    fn publish_mapped(mut self, rgba: Vec<u8>) -> Result<ImageBuffer> {
        self.release_staging();
        match ImageBuffer::try_new(self.physical_size, rgba) {
            Ok(image) => {
                self.lifecycle.published();
                #[cfg(all(test, not(target_arch = "wasm32")))]
                if let Some(observation) = &self.observation {
                    observation.record_phase(NativeReadbackPhaseForTest::PublishedBytes);
                }
                Ok(image)
            }
            Err(source) => {
                self.lifecycle.fail();
                #[cfg(all(test, not(target_arch = "wasm32")))]
                if let Some(observation) = &self.observation {
                    observation.record_phase(NativeReadbackPhaseForTest::Failed);
                }
                Err(Error::new(
                    BackendErrorCode::ReadbackFailed,
                    "decoded readback bytes did not form a valid RGBA8 image",
                )
                .with_source(source))
            }
        }
    }

    fn release_staging(&mut self) {
        let action = self.staging_map.take_cleanup_action();
        let Some(staging) = self.staging.take() else {
            assert_eq!(
                action,
                ReadbackStagingCleanupAction::None,
                "readback staging cleanup action must be consumed with ownership"
            );
            return;
        };
        match action {
            ReadbackStagingCleanupAction::Drop => drop(staging),
            ReadbackStagingCleanupAction::UnmapThenDrop => {
                staging.unmap();
                drop(staging);
            }
            ReadbackStagingCleanupAction::None => {
                unreachable!("owned readback staging cannot already be released")
            }
        }
    }
}

impl Drop for ReadbackOwner {
    fn drop(&mut self) {
        self.cancel();
    }
}

enum ReadbackCompletionResult {
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
    #[cfg(test)]
    accepted_results: usize,
    #[cfg(test)]
    discarded_results: usize,
}

struct ReadbackCompletion {
    state: Mutex<ReadbackCompletionState>,
    #[cfg(all(test, not(target_arch = "wasm32")))]
    observation: Option<NativeReadbackObservationForTest>,
}

impl ReadbackCompletion {
    fn new() -> Self {
        Self {
            state: Mutex::new(ReadbackCompletionState {
                status: ReadbackCompletionStatus::Pending { waker: None },
                #[cfg(test)]
                accepted_results: 0,
                #[cfg(test)]
                discarded_results: 0,
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

    fn complete(&self, result: ReadbackCompletionResult) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::mem::replace(&mut state.status, ReadbackCompletionStatus::Consumed);
            match previous {
                ReadbackCompletionStatus::Pending { waker } => {
                    state.status = ReadbackCompletionStatus::Ready(result);
                    #[cfg(test)]
                    {
                        state.accepted_results = state.accepted_results.saturating_add(1);
                    }
                    waker
                }
                terminal => {
                    state.status = terminal;
                    #[cfg(test)]
                    {
                        state.discarded_results = state.discarded_results.saturating_add(1);
                    }
                    None
                }
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn record_completion_counts_for_test(&self) {
        let Some(observation) = &self.observation else {
            return;
        };
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        observation.record_completion_counts(state.accepted_results, state.discarded_results);
    }

    fn poll(&self, context: &mut Context<'_>) -> Poll<ReadbackCompletionResult> {
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

    fn cancel(&self) {
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

fn map_completion_callback(
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
        #[cfg(all(test, not(target_arch = "wasm32")))]
        completion.record_completion_counts_for_test();
    }
}

fn completion_result(result: ReadbackCompletionResult) -> Result<()> {
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
enum NativePollAction {
    Continue,
    Stop,
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_native_poll_result(
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

struct ReadbackMapFuture {
    completion: Arc<ReadbackCompletion>,
    owner: Option<ReadbackOwner>,
    #[cfg(not(target_arch = "wasm32"))]
    _poll_helper: thread::JoinHandle<()>,
}

impl ReadbackMapFuture {
    fn start(
        mut owner: ReadbackOwner,
        device: wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
    ) -> Result<Self> {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        let completion = Arc::new(
            ReadbackCompletion::new().with_observation_for_test(owner.observation_for_test()),
        );
        #[cfg(not(all(test, not(target_arch = "wasm32"))))]
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

pub(crate) async fn read_texture_rgba(
    backend: &mut Backend,
    device_identity: DeviceSlotIdentity,
    texture: &wgpu::Texture,
    physical_size: PhysicalSize,
    operation: RuntimeOperation,
) -> Result<ImageBuffer> {
    if physical_size.width() == 0 || physical_size.height() == 0 {
        return ImageBuffer::try_new(physical_size, Vec::new());
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    let observation = take_native_readback_observation_for_test();

    let transaction =
        backend.begin_gpu_operation(device_identity, GpuOperationStage::Readback, operation)?;
    let layout = match ReadbackLayout::try_new(physical_size) {
        Ok(layout) => layout,
        Err(error) => {
            let scope_result = transaction.finish(operation).await;
            backend.observe_device_terminal(device_identity);
            scope_result?;
            return Err(error);
        }
    };
    let (mut owner, pending_submission) = {
        let (device, queue) = backend.gpu_operation_device_queue(
            device_identity,
            operation,
            GpuOperationStage::Readback,
        )?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist scoped texture readback"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut owner = ReadbackOwner::allocated(buffer, layout, physical_size);
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = observation {
            owner.attach_observation_for_test(observation);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist scoped texture readback copy"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: owner.staging(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(owner.layout.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: owner.layout.width,
                height: owner.layout.height,
                depth_or_array_layers: 1,
            },
        );
        let pending_submission = transaction.submit_readback(queue, encoder.finish());
        owner.copy_submitted(pending_submission.submission_index());
        (owner, pending_submission)
    };

    let submission_result = pending_submission.finish(operation).await;
    backend.observe_device_terminal(device_identity);
    let submission = match submission_result {
        Ok(submission) => submission,
        Err(error) => {
            owner.fail();
            return Err(error);
        }
    };

    let device = match backend.gpu_operation_device_queue(
        device_identity,
        operation,
        GpuOperationStage::Readback,
    ) {
        Ok((device, _)) => device.clone(),
        Err(error) => {
            owner.fail();
            return Err(error);
        }
    };
    let readback_result =
        match ReadbackMapFuture::start(owner, device, submission.into_submission_index()) {
            Ok(readback) => readback.await,
            Err(error) => Err(error),
        };

    backend.observe_device_terminal(device_identity);
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    readback_result
}

fn decode_padded_rows(layout: &ReadbackLayout, mapped: &[u8]) -> Result<Vec<u8>> {
    let mut rgba = Vec::with_capacity(layout.decoded_len);
    for row in 0..layout.height {
        let start = u64::from(row)
            .checked_mul(u64::from(layout.padded_bytes_per_row))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| readback_failed("mapped readback row offset overflowed"))?;
        let end = start
            .checked_add(layout.row_bytes)
            .ok_or_else(|| readback_failed("mapped readback row end overflowed"))?;
        let row = mapped
            .get(start..end)
            .ok_or_else(|| readback_failed("mapped readback row was incomplete"))?;
        rgba.extend_from_slice(row);
    }
    if rgba.len() != layout.decoded_len {
        return Err(readback_failed(
            "decoded readback byte count did not match the validated layout",
        ));
    }
    Ok(rgba)
}

fn readback_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::ReadbackFailed, message)
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackStagingDispositionForTest {
    Idle,
    MapPending,
    MappedActive,
    Released,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadbackCleanupEventForTest {
    MappedViewDropped,
    StagingUnmapped,
    StagingDropped,
    PublishedBytes,
}

#[cfg(test)]
#[derive(Default)]
struct ReadbackStateMachineObservationStateForTest {
    terminal_phase: Option<ReadbackPhaseForTest>,
    cleanup_events: Vec<ReadbackCleanupEventForTest>,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ReadbackStateMachineObservationForTest {
    state: Arc<Mutex<ReadbackStateMachineObservationStateForTest>>,
    staging_map: Arc<ReadbackStagingMapState>,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) struct ReadbackStateMachineForTest {
    lifecycle: ReadbackLifecycle<u64>,
    observation: ReadbackStateMachineObservationForTest,
}

#[cfg(test)]
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

#[cfg(test)]
impl Drop for ReadbackStateMachineForTest {
    fn drop(&mut self) {
        self.cancel_for_test();
    }
}

#[cfg(test)]
pub(crate) struct ReadbackCompletionForTest {
    completion: Arc<ReadbackCompletion>,
    staging_map: Arc<ReadbackStagingMapState>,
}

#[cfg(test)]
impl ReadbackCompletionForTest {
    pub(crate) fn new() -> Self {
        let staging_map = Arc::new(ReadbackStagingMapState::idle());
        staging_map.map_pending();
        Self {
            completion: Arc::new(ReadbackCompletion::new()),
            staging_map,
        }
    }

    pub(crate) fn poll_for_test(&self, context: &mut Context<'_>) -> Poll<Result<()>> {
        self.completion.poll(context).map(completion_result)
    }

    pub(crate) fn invoke_map_callback_for_test(
        &self,
        result: std::result::Result<(), wgpu::BufferAsyncError>,
    ) {
        map_completion_callback(Arc::clone(&self.completion), Arc::clone(&self.staging_map))(
            result,
        );
    }

    pub(crate) fn deliver_late_map_result_for_test(
        &self,
        result: std::result::Result<(), wgpu::BufferAsyncError>,
    ) {
        self.completion
            .complete(ReadbackCompletionResult::Map(result));
    }

    pub(crate) fn cancel_for_test(&self) {
        self.completion.cancel();
    }

    pub(crate) fn is_canceled_for_test(&self) -> bool {
        matches!(
            &self
                .completion
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status,
            ReadbackCompletionStatus::Canceled
        )
    }

    pub(crate) fn accepted_result_count_for_test(&self) -> usize {
        self.completion
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .accepted_results
    }

    pub(crate) fn discarded_result_count_for_test(&self) -> usize {
        self.completion
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .discarded_results
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn timeout_slice_for_test(&self) -> bool {
        handle_native_poll_result(self.completion.as_ref(), Err(wgpu::PollError::Timeout))
            == NativePollAction::Continue
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wrong_submission_index_for_test(&self, requested: u64, successful: u64) {
        let action = handle_native_poll_result(
            self.completion.as_ref(),
            Err(wgpu::PollError::WrongSubmissionIndex(requested, successful)),
        );
        assert_eq!(action, NativePollAction::Stop);
    }
}
