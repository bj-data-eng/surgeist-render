use super::{
    BackendErrorCode, Error, GpuFaultKind, Result, RuntimeOperation,
    backend::{DeviceSignal, DeviceTerminalSignal},
    vello_engine::PendingVelloResourceCommit,
};
use std::sync::Arc;

#[cfg(test)]
use std::sync::{
    Mutex,
    mpsc::{Receiver, SyncSender, sync_channel},
};

#[cfg(test)]
use super::vello_engine::VelloResourceAllocationSummaryForTest;

/// Private ownership stage for a render-owned GPU operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuOperationStage {
    SurfaceCreate,
    RendererCreate,
    Render,
    #[cfg(test)]
    Present,
}

impl GpuOperationStage {
    const fn error_code(self) -> BackendErrorCode {
        match self {
            Self::SurfaceCreate => BackendErrorCode::SurfaceCreateFailed,
            Self::RendererCreate => BackendErrorCode::RendererCreateFailed,
            Self::Render => BackendErrorCode::RenderFailed,
            #[cfg(test)]
            Self::Present => BackendErrorCode::PresentFailed,
        }
    }

    fn classify_fault(self, kind: GpuFaultKind, message: &str) -> Error {
        let code = if kind == GpuFaultKind::OutOfMemory {
            BackendErrorCode::SurfaceOutOfMemory
        } else {
            self.error_code()
        };
        Error::new(code, message)
    }

    #[cfg(test)]
    pub(crate) fn classify_fault_for_test(self, kind: GpuFaultKind, message: &str) -> Error {
        self.classify_fault(kind, message)
    }
}

/// A value that can only be made visible by an explicit successful commit.
#[must_use = "GPU draft state must be committed or dropped"]
pub(crate) struct GpuOperationDraft<'a, T> {
    target: &'a mut Option<T>,
    value: Option<T>,
}

impl<'a, T> GpuOperationDraft<'a, T> {
    pub(crate) fn new(target: &'a mut Option<T>, value: T) -> Self {
        Self {
            target,
            value: Some(value),
        }
    }

    pub(crate) fn commit(mut self) {
        *self.target = self.value.take();
    }
}

/// Owns one active generation and clears only that generation on every exit.
#[must_use = "GPU operation leases must remain alive until scopes resolve"]
pub(crate) struct GpuOperationLease {
    signal: Arc<DeviceSignal>,
    generation: u64,
}

impl GpuOperationLease {
    pub(crate) fn begin(signal: Arc<DeviceSignal>, generation: u64) -> Self {
        signal.activate(generation);
        Self { signal, generation }
    }

    #[cfg(test)]
    pub(crate) fn begin_for_test(signal: &Arc<DeviceSignal>) -> Result<Self> {
        let generation = signal.next_test_generation()?;
        Ok(Self::begin(Arc::clone(signal), generation))
    }

    const fn generation(&self) -> u64 {
        self.generation
    }

    fn finish(&self) -> Option<Arc<DeviceTerminalSignal>> {
        self.signal.finish_active_generation(self.generation)
    }

    #[cfg(test)]
    pub(crate) const fn generation_for_test(&self) -> u64 {
        self.generation
    }
}

impl Drop for GpuOperationLease {
    fn drop(&mut self) {
        self.signal.clear_active(self.generation);
    }
}

/// Nested WGPU scopes and the active-generation lease for one operation.
///
/// Scope fields are stored in reverse pop order. The explicit `Drop` below
/// preserves that order when an async operation future is canceled.
#[must_use = "GPU operation transactions must resolve scopes before publishing state"]
pub(crate) struct GpuOperationTransaction {
    validation: Option<wgpu::ErrorScopeGuard>,
    out_of_memory: Option<wgpu::ErrorScopeGuard>,
    internal: Option<wgpu::ErrorScopeGuard>,
    lease: GpuOperationLease,
    stage: GpuOperationStage,
}

#[must_use = "internal Vello command buffers must remain owned by their GPU transaction"]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "T6 keeps internal Vello submission and resource commitment transaction-owned before T7 production cutover."
    )
)]
pub(crate) struct InternalVelloPayload<'resources> {
    command_buffer: wgpu::CommandBuffer,
    resources: PendingVelloResourceCommit<'resources>,
    #[cfg(test)]
    submission_observation: Option<InternalVelloSubmissionObservationForTest>,
    #[cfg(test)]
    after_submit_checkpoint: Option<AfterInternalVelloSubmitCheckpointForTest>,
}

/// Test-only evidence carried by the real single-buffer internal raster payload.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct InternalVelloSubmissionObservationForTest {
    state: Arc<Mutex<InternalVelloSubmissionObservationStateForTest>>,
}

#[cfg(test)]
#[derive(Default)]
struct InternalVelloSubmissionObservationStateForTest {
    queue_submission_count: usize,
    transaction_generation: Option<u64>,
    payload_raster_pass_count: usize,
    allocation_summary: Option<VelloResourceAllocationSummaryForTest>,
}

#[cfg(test)]
impl InternalVelloSubmissionObservationForTest {
    fn record_payload_submission(
        &self,
        transaction_generation: u64,
        allocation_summary: VelloResourceAllocationSummaryForTest,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("internal Vello submission observation must remain available");
        state.queue_submission_count = state.queue_submission_count.saturating_add(1);
        state.transaction_generation = Some(transaction_generation);
        // `InternalVelloPayload` owns exactly the command buffer consumed by this transition.
        state.payload_raster_pass_count = state.payload_raster_pass_count.saturating_add(1);
        state.allocation_summary = Some(allocation_summary);
    }

    pub(crate) fn queue_submission_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("internal Vello submission observation must remain available")
            .queue_submission_count
    }

    pub(crate) fn transaction_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("internal Vello submission observation must remain available")
            .transaction_generation
    }

    pub(crate) fn payload_raster_pass_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("internal Vello submission observation must remain available")
            .payload_raster_pass_count
    }

    pub(crate) fn allocation_summary_for_test(
        &self,
    ) -> Option<VelloResourceAllocationSummaryForTest> {
        self.state
            .lock()
            .expect("internal Vello submission observation must remain available")
            .allocation_summary
            .clone()
    }
}

/// Test-only pause reached after the real queue submission and before transaction completion.
#[cfg(test)]
pub(crate) struct AfterInternalVelloSubmitCheckpointForTest {
    reached: SyncSender<()>,
}

#[cfg(test)]
impl AfterInternalVelloSubmitCheckpointForTest {
    pub(crate) fn paused() -> (Self, Receiver<()>) {
        let (reached, observed) = sync_channel(1);
        (Self { reached }, observed)
    }

    async fn wait(self) {
        self.reached
            .send(())
            .expect("the cancellation adapter must observe the post-submit checkpoint");
        std::future::pending::<()>().await;
    }
}

/// Proof that an internal Vello submission has reached its clean terminal boundary.
pub(crate) struct VelloResourceCommitProof {
    _private: (),
}

impl<'resources> InternalVelloPayload<'resources> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T6 constructs transaction-owned Vello payloads before T7 production cutover."
        )
    )]
    pub(crate) fn new(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit<'resources>,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            #[cfg(test)]
            submission_observation: None,
            #[cfg(test)]
            after_submit_checkpoint: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn observed_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit<'resources>,
        submission_observation: InternalVelloSubmissionObservationForTest,
    ) -> Self {
        let mut payload = Self::new(command_buffer, resources);
        payload.submission_observation = Some(submission_observation);
        payload
    }

    #[cfg(test)]
    pub(crate) fn paused_after_submit_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit<'resources>,
        checkpoint: AfterInternalVelloSubmitCheckpointForTest,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            submission_observation: None,
            after_submit_checkpoint: Some(checkpoint),
        }
    }
}

impl GpuOperationTransaction {
    pub(crate) fn begin(
        device: &wgpu::Device,
        signal: Arc<DeviceSignal>,
        generation: u64,
        stage: GpuOperationStage,
    ) -> Self {
        let lease = GpuOperationLease::begin(signal, generation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            validation: Some(validation),
            out_of_memory: Some(out_of_memory),
            internal: Some(internal),
            lease,
            stage,
        }
    }

    /// Resolves all error scopes before the caller may publish its draft state.
    pub(crate) async fn finish(mut self, operation: RuntimeOperation) -> Result<()> {
        let validation = self
            .validation
            .take()
            .expect("transaction validation scope must be present")
            .pop()
            .await;
        let out_of_memory = self
            .out_of_memory
            .take()
            .expect("transaction out-of-memory scope must be present")
            .pop()
            .await;
        let internal = self
            .internal
            .take()
            .expect("transaction internal scope must be present")
            .pop()
            .await;

        if let Some(terminal) = self.lease.finish() {
            return match terminal.as_ref() {
                DeviceTerminalSignal::Lost { .. } => Err(terminal.error(operation)),
                DeviceTerminalSignal::Faulted {
                    kind,
                    message,
                    operation_generation: Some(generation),
                } if *generation == self.lease.generation() => {
                    Err(self.stage.classify_fault(*kind, message))
                }
                DeviceTerminalSignal::Faulted { .. } => Err(terminal.error(operation)),
            };
        }

        if let Some(error) = [validation, out_of_memory, internal]
            .into_iter()
            .flatten()
            .next()
        {
            return Err(classify_captured_error(self.stage, error));
        }
        Ok(())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T6 is the sole internal Vello submission route before T7 production cutover."
        )
    )]
    pub(crate) async fn submit_internal_vello(
        self,
        queue: &wgpu::Queue,
        payload: InternalVelloPayload<'_>,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let InternalVelloPayload {
            command_buffer,
            resources,
            #[cfg(test)]
            submission_observation,
            #[cfg(test)]
            after_submit_checkpoint,
        } = payload;
        queue.submit([command_buffer]);
        #[cfg(test)]
        if let Some(observation) = submission_observation {
            observation.record_payload_submission(
                self.lease.generation(),
                resources.allocation_summary_for_test(),
            );
        }
        #[cfg(test)]
        if let Some(checkpoint) = after_submit_checkpoint {
            checkpoint.wait().await;
        }
        match self.finish(operation).await {
            Ok(()) => {
                resources.commit(VelloResourceCommitProof { _private: () });
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl Drop for GpuOperationTransaction {
    fn drop(&mut self) {
        drop(self.validation.take());
        drop(self.out_of_memory.take());
        drop(self.internal.take());
    }
}

fn classify_captured_error(stage: GpuOperationStage, error: wgpu::Error) -> Error {
    let kind = match error {
        wgpu::Error::Validation { .. } => GpuFaultKind::Validation,
        wgpu::Error::OutOfMemory { .. } => GpuFaultKind::OutOfMemory,
        wgpu::Error::Internal { .. } => GpuFaultKind::Internal,
    };
    stage
        .classify_fault(kind, &error.to_string())
        .with_source(error)
}
