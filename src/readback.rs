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

const RGBA8_BYTES_PER_PIXEL: u64 = 4;
const COPY_BYTES_PER_ROW_ALIGNMENT: u64 = 256;

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

struct ReadbackOwner {
    lifecycle: ReadbackLifecycle<wgpu::SubmissionIndex>,
    staging: Option<wgpu::Buffer>,
    layout: ReadbackLayout,
    physical_size: PhysicalSize,
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
            layout,
            physical_size,
        }
    }

    fn staging(&self) -> &wgpu::Buffer {
        self.staging
            .as_ref()
            .expect("an uncertain readback phase must own its staging buffer")
    }

    fn copy_submitted(&mut self, submission_index: wgpu::SubmissionIndex) {
        self.lifecycle.copy_submitted(submission_index);
    }

    fn map_pending(&mut self) {
        self.lifecycle.map_pending();
    }

    fn mapped(&mut self) {
        self.lifecycle.mapped();
    }

    fn mapped_range(&self) -> Range<wgpu::BufferAddress> {
        self.layout.mapped_range.bytes()
    }

    fn fail(&mut self) {
        if self.lifecycle.fail() {
            self.release_staging();
        }
    }

    fn cancel(&mut self) {
        if self.lifecycle.cancel() {
            self.release_staging();
        }
    }

    fn publish_mapped(mut self, rgba: Vec<u8>) -> Result<ImageBuffer> {
        self.release_staging();
        match ImageBuffer::try_new(self.physical_size, rgba) {
            Ok(image) => {
                self.lifecycle.published();
                Ok(image)
            }
            Err(source) => {
                self.lifecycle.fail();
                Err(Error::new(
                    BackendErrorCode::ReadbackFailed,
                    "decoded readback bytes did not form a valid RGBA8 image",
                )
                .with_source(source))
            }
        }
    }

    fn release_staging(&mut self) {
        if let Some(staging) = self.staging.take() {
            staging.unmap();
            drop(staging);
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
        }
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
) -> impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + 'static {
    move |result| completion.complete(ReadbackCompletionResult::Map(result))
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
        let completion = Arc::new(ReadbackCompletion::new());
        owner.map_pending();
        owner.staging().slice(owner.mapped_range()).map_async(
            wgpu::MapMode::Read,
            map_completion_callback(Arc::clone(&completion)),
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
    staging_unmapped: bool,
    staging_dropped: bool,
    staging_reused: bool,
    cleanup_events: Vec<ReadbackCleanupEventForTest>,
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct ReadbackStateMachineObservationForTest {
    state: Arc<Mutex<ReadbackStateMachineObservationStateForTest>>,
}

#[cfg(test)]
impl ReadbackStateMachineObservationForTest {
    pub(crate) fn terminal_phase_for_test(&self) -> Option<ReadbackPhaseForTest> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .terminal_phase
    }

    pub(crate) fn staging_was_unmapped_for_test(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .staging_unmapped
    }

    pub(crate) fn staging_was_dropped_for_test(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .staging_dropped
    }

    pub(crate) fn staging_was_reused_for_test(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .staging_reused
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
    }

    pub(crate) fn mapped_for_test(&mut self) {
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

    pub(crate) fn staging_was_unmapped_for_test(&self) -> bool {
        self.observation.staging_was_unmapped_for_test()
    }

    pub(crate) fn staging_was_dropped_for_test(&self) -> bool {
        self.observation.staging_was_dropped_for_test()
    }

    pub(crate) fn staging_was_reused_for_test(&self) -> bool {
        self.observation.staging_was_reused_for_test()
    }

    pub(crate) fn cleanup_events_for_test(&self) -> Vec<ReadbackCleanupEventForTest> {
        self.observation.cleanup_events_for_test()
    }

    pub(crate) fn observation_for_test(&self) -> ReadbackStateMachineObservationForTest {
        self.observation.clone()
    }

    fn release_staging_for_test(&mut self) {
        let mut observation = self
            .observation
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !observation.staging_dropped {
            observation.staging_unmapped = true;
            observation
                .cleanup_events
                .push(ReadbackCleanupEventForTest::StagingUnmapped);
            observation.staging_dropped = true;
            observation
                .cleanup_events
                .push(ReadbackCleanupEventForTest::StagingDropped);
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
}

#[cfg(test)]
impl ReadbackCompletionForTest {
    pub(crate) fn new() -> Self {
        Self {
            completion: Arc::new(ReadbackCompletion::new()),
        }
    }

    pub(crate) fn poll_for_test(&self, context: &mut Context<'_>) -> Poll<Result<()>> {
        self.completion.poll(context).map(completion_result)
    }

    pub(crate) fn invoke_map_callback_for_test(
        &self,
        result: std::result::Result<(), wgpu::BufferAsyncError>,
    ) {
        map_completion_callback(Arc::clone(&self.completion))(result);
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
