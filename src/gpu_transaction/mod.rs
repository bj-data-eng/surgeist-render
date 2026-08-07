mod graph;

#[expect(
    unused_imports,
    reason = "preserves the existing crate-visible transaction front-door path"
)]
pub(crate) use graph::GraphSubmissionCommit;
pub(crate) use graph::{GraphOutputCommit, GraphSubmissionPayload};
#[cfg(test)]
pub(crate) use graph::{
    GraphResourceRetentionForTest, GraphSubmissionObservationForTest,
    ScopedGraphPostSubmitControlForTest, ScopedGraphSubmissionObservationForTest,
};

use super::{
    BackendErrorCode, Error, GpuFaultKind, Result, RuntimeOperation,
    backend::{DeviceSignal, DeviceTerminalSignal},
    vello_engine::{DirectVelloLogicalPass, PendingVelloResourceCommit},
};

use std::sync::Arc;

#[cfg(test)]
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, SyncSender, sync_channel},
};

#[cfg(test)]
use std::cell::RefCell;

#[cfg(test)]
use super::vello_engine::VelloResourceAllocationSummaryForTest;

#[cfg(test)]
thread_local! {
    static ACTIVE_GPU_OPERATION_SUBMISSION_OBSERVATION_FOR_TEST: RefCell<Option<GpuOperationSubmissionObservationForTest>> = const { RefCell::new(None) };
    static ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST: RefCell<Option<GpuOperationPostSubmitControlForTest>> = const { RefCell::new(None) };
    static ACTIVE_INTERNAL_VELLO_SUBMISSION_OBSERVATION_FOR_TEST: RefCell<Option<InternalVelloSubmissionObservationForTest>> = const { RefCell::new(None) };
    static ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST: RefCell<Option<InternalVelloPostSubmitControlForTest>> = const { RefCell::new(None) };
}

/// Private ownership stage for a render-owned GPU operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuOperationStage {
    Render,
    Readback,
    #[cfg(any(
        test,
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Configure,
    #[cfg(any(
        test,
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Present,
}

impl GpuOperationStage {
    pub(crate) const fn error_code(self) -> BackendErrorCode {
        match self {
            Self::Render => BackendErrorCode::RenderFailed,
            Self::Readback => BackendErrorCode::ReadbackFailed,
            #[cfg(any(
                test,
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            Self::Configure => BackendErrorCode::SurfaceConfigureFailed,
            #[cfg(any(
                test,
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
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
    fn active_generation_for_test(&self) -> Option<u64> {
        self.signal.active_generation_for_test()
    }

    #[cfg(test)]
    pub(crate) const fn generation_for_test(&self) -> u64 {
        self.generation
    }
}

/// Test-only observation of one generic transaction-owned command-buffer submission.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct GpuOperationSubmissionObservationForTest {
    state: Arc<Mutex<GpuOperationSubmissionObservationStateForTest>>,
}

#[cfg(test)]
#[derive(Default)]
struct GpuOperationSubmissionObservationStateForTest {
    queue_submission_count: usize,
    transaction_generation: Option<u64>,
    active_generation: Option<u64>,
    scopes_resolved: bool,
    readback_queue_submission_count: usize,
    readback_transaction_generation: Option<u64>,
    readback_active_generation: Option<u64>,
    readback_submission_index: Option<wgpu::SubmissionIndex>,
    readback_scopes_resolved: bool,
}

#[cfg(test)]
impl GpuOperationSubmissionObservationForTest {
    fn record_submission(
        &self,
        transaction_generation: u64,
        active_generation: Option<u64>,
        readback_submission_index: Option<wgpu::SubmissionIndex>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("GPU operation submission observation must remain available");
        if let Some(submission_index) = readback_submission_index {
            state.readback_queue_submission_count =
                state.readback_queue_submission_count.saturating_add(1);
            state.readback_transaction_generation = Some(transaction_generation);
            state.readback_active_generation = active_generation;
            state.readback_submission_index = Some(submission_index);
        } else {
            state.queue_submission_count = state.queue_submission_count.saturating_add(1);
            state.transaction_generation = Some(transaction_generation);
            state.active_generation = active_generation;
        }
    }

    fn record_scope_resolution(&self, readback: bool) {
        let mut state = self
            .state
            .lock()
            .expect("GPU operation submission observation must remain available");
        if readback {
            state.readback_scopes_resolved = true;
        } else {
            state.scopes_resolved = true;
        }
    }

    pub(crate) fn queue_submission_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .queue_submission_count
    }

    pub(crate) fn transaction_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .transaction_generation
    }

    pub(crate) fn active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .active_generation
    }

    pub(crate) fn readback_submission_index_for_test(&self) -> Option<wgpu::SubmissionIndex> {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .readback_submission_index
            .clone()
    }

    pub(crate) fn readback_queue_submission_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .readback_queue_submission_count
    }

    pub(crate) fn readback_transaction_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .readback_transaction_generation
    }

    pub(crate) fn readback_active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .readback_active_generation
    }

    pub(crate) fn readback_scopes_resolved_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .readback_scopes_resolved
    }

    pub(crate) fn scopes_resolved_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("GPU operation submission observation must remain available")
            .scopes_resolved
    }
}

/// Installs a private observation for generic transaction submissions on this thread.
#[cfg(test)]
pub(crate) struct ScopedGpuOperationSubmissionObservationForTest {
    observation: GpuOperationSubmissionObservationForTest,
    previous: Option<GpuOperationSubmissionObservationForTest>,
}

#[cfg(test)]
impl ScopedGpuOperationSubmissionObservationForTest {
    pub(crate) fn begin() -> Self {
        let observation = GpuOperationSubmissionObservationForTest::default();
        let previous = ACTIVE_GPU_OPERATION_SUBMISSION_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(observation.clone())));
        Self {
            observation,
            previous,
        }
    }

    pub(crate) fn observation_for_test(&self) -> GpuOperationSubmissionObservationForTest {
        self.observation.clone()
    }
}

#[cfg(test)]
impl Drop for ScopedGpuOperationSubmissionObservationForTest {
    fn drop(&mut self) {
        ACTIVE_GPU_OPERATION_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

/// Private test-only pause immediately after a generic transaction submits work.
#[cfg(test)]
pub(crate) struct ScopedGpuOperationPostSubmitCheckpointForTest {
    observed: Receiver<()>,
    release: Option<Arc<AtomicBool>>,
    previous: Option<GpuOperationPostSubmitControlForTest>,
}

#[cfg(test)]
#[derive(Clone)]
enum GpuOperationPostSubmitControlForTest {
    Pause(SyncSender<()>),
    Yield {
        reached: SyncSender<()>,
        released: Arc<AtomicBool>,
    },
}

#[cfg(test)]
impl ScopedGpuOperationPostSubmitCheckpointForTest {
    pub(crate) fn begin() -> Self {
        let (reached, observed) = sync_channel(1);
        let previous = ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            active.replace(Some(GpuOperationPostSubmitControlForTest::Pause(reached)))
        });
        Self {
            observed,
            release: None,
            previous,
        }
    }

    pub(crate) fn yielding() -> Self {
        let (reached, observed) = sync_channel(1);
        let released = Arc::new(AtomicBool::new(false));
        let previous = ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            active.replace(Some(GpuOperationPostSubmitControlForTest::Yield {
                reached,
                released: Arc::clone(&released),
            }))
        });
        Self {
            observed,
            release: Some(released),
            previous,
        }
    }

    pub(crate) fn wait_for_submission_for_test(&self, deadline: std::time::Duration) {
        self.observed
            .recv_timeout(deadline)
            .expect("the real generic submission did not reach the bounded post-submit checkpoint");
    }

    pub(crate) fn release_for_test(&self) {
        self.release
            .as_ref()
            .expect("only a yielding post-submit checkpoint can resume the submission")
            .store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
impl Drop for ScopedGpuOperationPostSubmitCheckpointForTest {
    fn drop(&mut self) {
        ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
fn record_active_gpu_operation_submission_for_test(
    transaction_generation: u64,
    active_generation: Option<u64>,
    readback_submission_index: Option<wgpu::SubmissionIndex>,
) -> Option<GpuOperationSubmissionObservationForTest> {
    ACTIVE_GPU_OPERATION_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
        let observation = active.borrow().clone();
        if let Some(observation) = &observation {
            observation.record_submission(
                transaction_generation,
                active_generation,
                readback_submission_index,
            );
        }
        observation
    })
}

#[cfg(test)]
async fn wait_at_active_gpu_operation_post_submit_checkpoint_for_test() {
    let checkpoint =
        ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| active.borrow().clone());
    match checkpoint {
        Some(GpuOperationPostSubmitControlForTest::Pause(reached)) => {
            reached
                .send(())
                .expect("the generic submission test must observe the post-submit checkpoint");
            std::future::pending::<()>().await;
        }
        Some(GpuOperationPostSubmitControlForTest::Yield { reached, released }) => {
            reached
                .send(())
                .expect("the generic submission test must observe the post-submit checkpoint");
            std::future::poll_fn(|_| {
                released
                    .load(Ordering::SeqCst)
                    .then_some(())
                    .map_or(std::task::Poll::Pending, std::task::Poll::Ready)
            })
            .await;
        }
        None => {}
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
pub(crate) struct InternalVelloPayload {
    command_buffer: wgpu::CommandBuffer,
    resources: PendingVelloResourceCommit,
    logical_pass: DirectVelloLogicalPass,
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
    active_generation: Option<u64>,
    payload_raster_pass_count: usize,
    allocation_summary: Option<VelloResourceAllocationSummaryForTest>,
}

#[cfg(test)]
impl InternalVelloSubmissionObservationForTest {
    fn record_payload_submission(
        &self,
        transaction_generation: u64,
        active_generation: Option<u64>,
        logical_pass: &DirectVelloLogicalPass,
        allocation_summary: VelloResourceAllocationSummaryForTest,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("internal Vello submission observation must remain available");
        state.queue_submission_count = state.queue_submission_count.saturating_add(1);
        state.transaction_generation = Some(transaction_generation);
        state.active_generation = active_generation;
        state.payload_raster_pass_count = logical_pass.cardinality_for_test();
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

    pub(crate) fn active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("internal Vello submission observation must remain available")
            .active_generation
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

/// Observes the actual transaction-owned internal raster submission for one test scope.
#[cfg(test)]
pub(crate) struct ScopedInternalVelloSubmissionObservationForTest {
    observation: InternalVelloSubmissionObservationForTest,
    previous: Option<InternalVelloSubmissionObservationForTest>,
}

#[cfg(test)]
impl ScopedInternalVelloSubmissionObservationForTest {
    pub(crate) fn begin() -> Self {
        let observation = InternalVelloSubmissionObservationForTest::default();
        let previous = ACTIVE_INTERNAL_VELLO_SUBMISSION_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(observation.clone())));
        Self {
            observation,
            previous,
        }
    }

    pub(crate) fn observation_for_test(&self) -> InternalVelloSubmissionObservationForTest {
        self.observation.clone()
    }
}

#[cfg(test)]
impl Drop for ScopedInternalVelloSubmissionObservationForTest {
    fn drop(&mut self) {
        ACTIVE_INTERNAL_VELLO_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
fn record_active_internal_vello_submission_for_test(
    transaction_generation: u64,
    active_generation: Option<u64>,
    logical_pass: &DirectVelloLogicalPass,
    allocation_summary: VelloResourceAllocationSummaryForTest,
) {
    ACTIVE_INTERNAL_VELLO_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
        if let Some(observation) = active.borrow().as_ref() {
            observation.record_payload_submission(
                transaction_generation,
                active_generation,
                logical_pass,
                allocation_summary,
            );
        }
    });
}

/// Test-only pause reached after the real queue submission and before transaction completion.
#[cfg(test)]
pub(crate) struct AfterInternalVelloSubmitCheckpointForTest {
    reached: SyncSender<()>,
}

/// Private test control applied by the production submission path after `queue.submit`.
#[cfg(test)]
#[derive(Clone)]
enum InternalVelloPostSubmitControlForTest {
    Fail {
        scope_resolution_observed: SyncSender<()>,
    },
    AccountingFault,
    Pause(SyncSender<()>),
}

#[cfg(test)]
pub(crate) struct ScopedInternalVelloPostSubmitControlForTest {
    reached: Option<Receiver<()>>,
    scope_resolution_observed: Option<Receiver<()>>,
    previous: Option<InternalVelloPostSubmitControlForTest>,
}

#[cfg(test)]
impl ScopedInternalVelloPostSubmitControlForTest {
    pub(crate) fn failing() -> Self {
        let (scope_resolution_observed, observed) = sync_channel(1);
        let previous = ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(InternalVelloPostSubmitControlForTest::Fail {
                scope_resolution_observed,
            }))
        });
        Self {
            reached: None,
            scope_resolution_observed: Some(observed),
            previous,
        }
    }

    pub(crate) fn paused() -> Self {
        let (reached, observed) = sync_channel(1);
        let previous = ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(InternalVelloPostSubmitControlForTest::Pause(reached)))
        });
        Self {
            reached: Some(observed),
            scope_resolution_observed: None,
            previous,
        }
    }

    pub(crate) fn accounting_fault() -> Self {
        let previous = ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(InternalVelloPostSubmitControlForTest::AccountingFault))
        });
        Self {
            reached: None,
            scope_resolution_observed: None,
            previous,
        }
    }

    pub(crate) fn wait_for_submission_for_test(&self, deadline: std::time::Duration) {
        self.reached
            .as_ref()
            .expect("only a paused post-submit control has a submission receiver")
            .recv_timeout(deadline)
            .expect(
                "the real production submission did not reach the bounded post-submit checkpoint",
            );
    }

    pub(crate) fn scope_resolution_observed_for_test(&self) -> bool {
        self.scope_resolution_observed
            .as_ref()
            .is_some_and(|observed| observed.try_recv().is_ok())
    }
}

#[cfg(test)]
impl Drop for ScopedInternalVelloPostSubmitControlForTest {
    fn drop(&mut self) {
        ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
impl InternalVelloPostSubmitControlForTest {
    async fn apply(self, device: &wgpu::Device, resources: &PendingVelloResourceCommit) {
        match self {
            Self::Fail { .. } => {
                let _ = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Surgeist test-injected scoped validation failure"),
                    size: wgpu::Extent3d {
                        width: 0,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            }
            Self::AccountingFault => {
                let _ = resources.poison_retained_byte_accounting_for_test();
            }
            Self::Pause(reached) => {
                reached
                    .send(())
                    .expect("the production render test must observe the post-submit checkpoint");
                std::future::pending::<()>().await;
            }
        }
    }
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

/// Transaction-owned result of submitting one texture readback copy.
#[must_use = "the exact readback submission index must drive map completion"]
pub(crate) struct ReadbackSubmission {
    submission_index: wgpu::SubmissionIndex,
}

impl ReadbackSubmission {
    pub(crate) fn into_submission_index(self) -> wgpu::SubmissionIndex {
        self.submission_index
    }
}

/// A submitted readback copy whose WGPU error scopes are still resolving.
#[must_use = "the readback transaction scopes must resolve before mapping"]
pub(crate) struct PendingReadbackSubmission {
    submission_index: wgpu::SubmissionIndex,
    transaction: GpuOperationTransaction,
    #[cfg(test)]
    submission_observation: Option<GpuOperationSubmissionObservationForTest>,
}

impl PendingReadbackSubmission {
    pub(crate) fn submission_index(&self) -> wgpu::SubmissionIndex {
        self.submission_index.clone()
    }

    pub(crate) async fn finish(self, operation: RuntimeOperation) -> Result<ReadbackSubmission> {
        #[cfg(test)]
        wait_at_active_gpu_operation_post_submit_checkpoint_for_test().await;

        let result = self.transaction.finish(operation).await;
        #[cfg(test)]
        if let Some(observation) = self.submission_observation {
            observation.record_scope_resolution(true);
        }
        result.map(|()| ReadbackSubmission {
            submission_index: self.submission_index,
        })
    }
}

impl InternalVelloPayload {
    pub(crate) fn new(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            logical_pass,
            #[cfg(test)]
            submission_observation: None,
            #[cfg(test)]
            after_submit_checkpoint: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn observed_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
        submission_observation: InternalVelloSubmissionObservationForTest,
    ) -> Self {
        let mut payload = Self::new(command_buffer, resources, logical_pass);
        payload.submission_observation = Some(submission_observation);
        payload
    }

    #[cfg(test)]
    pub(crate) fn paused_after_submit_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
        checkpoint: AfterInternalVelloSubmitCheckpointForTest,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            logical_pass,
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

    async fn pop_active_scopes(&mut self) -> Option<wgpu::Error> {
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

        [validation, out_of_memory, internal]
            .into_iter()
            .flatten()
            .next()
    }

    fn terminal_result(
        &self,
        terminal: Option<Arc<DeviceTerminalSignal>>,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(terminal) = terminal else {
            return Ok(());
        };
        match terminal.as_ref() {
            DeviceTerminalSignal::Lost { .. } => Err(terminal.error(operation)),
            DeviceTerminalSignal::Faulted {
                kind,
                message,
                operation_generation: Some(generation),
            } if *generation == self.lease.generation() => {
                Err(self.stage.classify_fault(*kind, message))
            }
            DeviceTerminalSignal::Faulted { .. } => Err(terminal.error(operation)),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn resolve_submission_phase(&mut self, operation: RuntimeOperation) -> Result<()> {
        let captured = self.pop_active_scopes().await;
        self.terminal_result(self.lease.signal.first_terminal(), operation)?;
        if let Some(error) = captured {
            return Err(classify_captured_error(self.stage, error));
        }
        Ok(())
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn begin_present_phase(
        &mut self,
        device: &wgpu::Device,
        operation: RuntimeOperation,
    ) -> Result<()> {
        if self.validation.is_some() || self.out_of_memory.is_some() || self.internal.is_some() {
            return Err(Error::new(
                BackendErrorCode::PresentFailed,
                "the graph presentation phase started before submission scopes resolved",
            ));
        }
        self.terminal_result(self.lease.signal.first_terminal(), operation)?;
        self.stage = GpuOperationStage::Present;
        self.internal = Some(device.push_error_scope(wgpu::ErrorFilter::Internal));
        self.out_of_memory = Some(device.push_error_scope(wgpu::ErrorFilter::OutOfMemory));
        self.validation = Some(device.push_error_scope(wgpu::ErrorFilter::Validation));
        Ok(())
    }

    /// Resolves all error scopes before the caller may publish its draft state.
    pub(crate) async fn finish(mut self, operation: RuntimeOperation) -> Result<()> {
        let captured = self.pop_active_scopes().await;

        #[cfg(test)]
        ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            if let Some(InternalVelloPostSubmitControlForTest::Fail {
                scope_resolution_observed,
            }) = active.borrow().as_ref()
            {
                let _ = scope_resolution_observed.send(());
            }
        });

        self.terminal_result(self.lease.finish(), operation)?;
        if let Some(error) = captured {
            return Err(classify_captured_error(self.stage, error));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) async fn finish_vello_resources_without_submission_for_test(
        self,
        resources: PendingVelloResourceCommit,
        operation: RuntimeOperation,
    ) -> Result<()> {
        self.finish(operation).await?;
        resources
            .into_accounting_ready()?
            .commit(VelloResourceCommitProof { _private: () })?;
        Ok(())
    }

    /// Submits one command buffer while this transaction owns its generation and scopes.
    #[cfg(test)]
    pub(crate) async fn submit_command_buffer(
        self,
        queue: &wgpu::Queue,
        command_buffer: wgpu::CommandBuffer,
        operation: RuntimeOperation,
    ) -> Result<()> {
        self.submit_command_buffer_with_host_effect(queue, command_buffer, || {}, operation)
            .await
    }

    /// Submits output work and applies its non-rollbackable host effect while scoped.
    #[cfg_attr(
        all(
            not(test),
            not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))
        ),
        expect(dead_code, reason = "presented output submission is feature-gated")
    )]
    pub(crate) async fn submit_command_buffer_with_host_effect(
        self,
        queue: &wgpu::Queue,
        command_buffer: wgpu::CommandBuffer,
        host_effect: impl FnOnce(),
        operation: RuntimeOperation,
    ) -> Result<()> {
        queue.submit([command_buffer]);
        #[cfg(test)]
        let submission_observation = record_active_gpu_operation_submission_for_test(
            self.lease.generation(),
            self.lease.active_generation_for_test(),
            None,
        );
        host_effect();
        #[cfg(test)]
        wait_at_active_gpu_operation_post_submit_checkpoint_for_test().await;

        let result = self.finish(operation).await;
        #[cfg(test)]
        if let Some(observation) = submission_observation {
            observation.record_scope_resolution(false);
        }
        result
    }

    /// Submits one texture-copy command and returns its exact queue submission index.
    pub(crate) fn submit_readback(
        self,
        queue: &wgpu::Queue,
        command_buffer: wgpu::CommandBuffer,
    ) -> PendingReadbackSubmission {
        let submission_index = queue.submit([command_buffer]);
        #[cfg(test)]
        let submission_observation = record_active_gpu_operation_submission_for_test(
            self.lease.generation(),
            self.lease.active_generation_for_test(),
            Some(submission_index.clone()),
        );

        PendingReadbackSubmission {
            submission_index,
            transaction: self,
            #[cfg(test)]
            submission_observation,
        }
    }

    pub(crate) async fn submit_internal_vello(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        payload: InternalVelloPayload,
        operation: RuntimeOperation,
    ) -> Result<()> {
        #[cfg(not(test))]
        let _ = device;
        let InternalVelloPayload {
            command_buffer,
            resources,
            logical_pass,
            #[cfg(test)]
            submission_observation,
            #[cfg(test)]
            after_submit_checkpoint,
        } = payload;
        queue.submit([command_buffer]);
        #[cfg(not(test))]
        let _ = logical_pass;
        #[cfg(test)]
        let active_generation = self.lease.active_generation_for_test();
        #[cfg(test)]
        let allocation_summary = resources.allocation_summary_for_test();
        #[cfg(test)]
        if let Some(observation) = submission_observation {
            observation.record_payload_submission(
                self.lease.generation(),
                active_generation,
                &logical_pass,
                allocation_summary.clone(),
            );
        }
        #[cfg(test)]
        record_active_internal_vello_submission_for_test(
            self.lease.generation(),
            active_generation,
            &logical_pass,
            allocation_summary,
        );
        #[cfg(test)]
        if let Some(checkpoint) = after_submit_checkpoint {
            checkpoint.wait().await;
        }
        #[cfg(test)]
        if let Some(control) = ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST
            .with(|active| active.borrow().clone())
        {
            control.apply(device, &resources).await;
        }
        match self.finish(operation).await {
            Ok(()) => {
                resources
                    .into_accounting_ready()?
                    .commit(VelloResourceCommitProof { _private: () })?;
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
