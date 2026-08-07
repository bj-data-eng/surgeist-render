use super::{
    GpuOperationSubmissionObservationForTest,
    graph::{GraphOutputCommit, GraphSubmissionRawFact},
    record_active_gpu_operation_submission_for_test,
};
use crate::{backend::DeviceSignal, pass::PendingPreparedFrameCommit};

#[cfg(feature = "render-window")]
use crate::DeviceLossReason;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
};

thread_local! {
    static ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST: RefCell<Option<GpuOperationPostSubmitControlForTest>> = const { RefCell::new(None) };
    static ACTIVE_GRAPH_SUBMISSION_OBSERVATION_FOR_TEST: RefCell<Option<GraphSubmissionObservationForTest>> = const { RefCell::new(None) };
    static ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST: RefCell<Option<GraphPostSubmitControlForTest>> = const { RefCell::new(None) };
}

static NEXT_GRAPH_SUBMISSION_RAW_FACT_ID_FOR_TEST: AtomicU64 = AtomicU64::new(0);
static GRAPH_SUBMISSION_STATE_FOR_TEST: OnceLock<
    Mutex<BTreeMap<u64, GraphSubmissionStateForTest>>,
> = OnceLock::new();

#[derive(Clone)]
struct GraphSubmissionStateForTest {
    generic_observation: Option<GpuOperationSubmissionObservationForTest>,
    graph_observation: Option<GraphSubmissionObservationForTest>,
    control: Option<GraphPostSubmitControlForTest>,
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
pub(super) async fn wait_at_active_gpu_operation_post_submit_checkpoint_for_test() {
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

#[cfg(test)]
impl GraphOutputCommit {
    fn observation_kind_for_test(&self) -> GraphCommittedOutputForTest {
        match self {
            Self::Headless(_) => GraphCommittedOutputForTest::Headless,
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            Self::Presented => GraphCommittedOutputForTest::Presented,
        }
    }
}

/// Test-only evidence emitted by the real graph transaction payload.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct GraphSubmissionObservationForTest {
    state: Arc<Mutex<GraphSubmissionObservationStateForTest>>,
}

#[cfg(test)]
#[derive(Default)]
struct GraphSubmissionObservationStateForTest {
    queue_submission_count: usize,
    transaction_generation: Option<u64>,
    active_generation: Option<u64>,
    capture_lease_count: usize,
    prepared_frame_resource_identities: Vec<crate::resource::ResourceIdentity>,
    prepared_frame_resource_identity_history: Vec<Vec<crate::resource::ResourceIdentity>>,
    scopes_resolved: bool,
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    presentation_scopes_resolved: bool,
    prepared_frame_committed: bool,
    capture_resources_committed: bool,
    committed_output: Option<GraphCommittedOutputForTest>,
    resource_retention: Option<GraphResourceRetentionForTest>,
    resource_retention_history: Vec<GraphResourceRetentionForTest>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphCommittedOutputForTest {
    Headless,
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphResourceRetentionForTest {
    RetainedReusable,
    ReleasedAllIdle,
}

#[cfg(test)]
impl GraphSubmissionObservationForTest {
    fn record_submission(
        &self,
        transaction_generation: u64,
        active_generation: Option<u64>,
        capture_lease_count: usize,
        prepared_frame_resource_identities: Vec<crate::resource::ResourceIdentity>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("graph submission observation must remain available");
        state.queue_submission_count = state.queue_submission_count.saturating_add(1);
        state.transaction_generation = Some(transaction_generation);
        state.active_generation = active_generation;
        state.capture_lease_count = capture_lease_count;
        state
            .prepared_frame_resource_identity_history
            .push(prepared_frame_resource_identities.clone());
        state.prepared_frame_resource_identities = prepared_frame_resource_identities;
    }

    fn record_scope_resolution(&self) {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .scopes_resolved = true;
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn record_presentation_scope_resolution(&self) {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .presentation_scopes_resolved = true;
    }

    fn record_commit(
        &self,
        output: GraphCommittedOutputForTest,
        retention: crate::resource::ResourceRetentionOutcome,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("graph submission observation must remain available");
        state.prepared_frame_committed = true;
        state.capture_resources_committed = true;
        state.committed_output = Some(output);
        let observed_retention = if retention.retains_reusable_resources() {
            Some(GraphResourceRetentionForTest::RetainedReusable)
        } else if retention.released_all_idle_resources() {
            Some(GraphResourceRetentionForTest::ReleasedAllIdle)
        } else {
            None
        };
        state.resource_retention = observed_retention;
        if let Some(observed_retention) = observed_retention {
            state.resource_retention_history.push(observed_retention);
        }
    }

    pub(crate) fn queue_submission_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .queue_submission_count
    }

    pub(crate) fn transaction_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .transaction_generation
    }

    pub(crate) fn active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .active_generation
    }

    pub(crate) fn capture_lease_count_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .capture_lease_count
    }

    pub(crate) fn prepared_frame_resource_identities_for_test(
        &self,
    ) -> Vec<crate::resource::ResourceIdentity> {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .prepared_frame_resource_identities
            .clone()
    }

    pub(crate) fn prepared_frame_resource_identity_history_for_test(
        &self,
    ) -> Vec<Vec<crate::resource::ResourceIdentity>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prepared_frame_resource_identity_history
            .clone()
    }

    pub(crate) fn scopes_resolved_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .scopes_resolved
    }

    #[cfg(feature = "render-window")]
    pub(crate) fn presentation_scopes_resolved_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .presentation_scopes_resolved
    }

    pub(crate) fn prepared_frame_committed_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .prepared_frame_committed
    }

    pub(crate) fn capture_resources_committed_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .capture_resources_committed
    }

    pub(crate) fn headless_draft_released_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .committed_output
            == Some(GraphCommittedOutputForTest::Headless)
    }

    pub(crate) fn resource_retention_for_test(&self) -> Option<GraphResourceRetentionForTest> {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .resource_retention
    }

    pub(crate) fn resource_retention_history_for_test(&self) -> Vec<GraphResourceRetentionForTest> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .resource_retention_history
            .clone()
    }

    #[cfg(feature = "render-window")]
    pub(crate) fn presented_host_effect_applied_for_test(&self) -> bool {
        self.state
            .lock()
            .expect("graph submission observation must remain available")
            .committed_output
            == Some(GraphCommittedOutputForTest::Presented)
    }
}

#[cfg(test)]
pub(crate) struct ScopedGraphSubmissionObservationForTest {
    observation: GraphSubmissionObservationForTest,
    previous: Option<GraphSubmissionObservationForTest>,
}

#[cfg(test)]
impl ScopedGraphSubmissionObservationForTest {
    pub(crate) fn begin() -> Self {
        let observation = GraphSubmissionObservationForTest::default();
        let previous = ACTIVE_GRAPH_SUBMISSION_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(observation.clone())));
        Self {
            observation,
            previous,
        }
    }

    pub(crate) fn observation_for_test(&self) -> GraphSubmissionObservationForTest {
        self.observation.clone()
    }
}

#[cfg(test)]
impl Drop for ScopedGraphSubmissionObservationForTest {
    fn drop(&mut self) {
        ACTIVE_GRAPH_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
fn record_active_base_graph_submission_for_test(
    transaction_generation: u64,
    active_generation: Option<u64>,
    capture_lease_count: usize,
    prepared_frame_resource_identities: Vec<crate::resource::ResourceIdentity>,
) -> Option<GraphSubmissionObservationForTest> {
    ACTIVE_GRAPH_SUBMISSION_OBSERVATION_FOR_TEST.with(|active| {
        active.borrow().as_ref().map(|observation| {
            observation.record_submission(
                transaction_generation,
                active_generation,
                capture_lease_count,
                prepared_frame_resource_identities,
            );
            observation.clone()
        })
    })
}

#[cfg(test)]
#[derive(Clone)]
enum GraphPostSubmitControlForTest {
    Fail {
        scope_resolution_observed: SyncSender<()>,
    },
    #[cfg(feature = "render-window")]
    TerminalLoss {
        scope_resolution_observed: SyncSender<()>,
    },
    #[cfg(feature = "render-window")]
    PresentFail {
        scope_resolution_observed: SyncSender<()>,
    },
    AccountingFault,
    Pause(SyncSender<()>),
}

/// Private deterministic failure/cancellation control for the production graph executor.
#[cfg(test)]
pub(crate) struct ScopedGraphPostSubmitControlForTest {
    reached: Option<Receiver<()>>,
    scope_resolution_observed: Option<Receiver<()>>,
    previous: Option<GraphPostSubmitControlForTest>,
}

#[cfg(test)]
impl ScopedGraphPostSubmitControlForTest {
    pub(crate) fn failing() -> Self {
        let (scope_resolution_observed, observed) = sync_channel(1);
        let previous = ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(GraphPostSubmitControlForTest::Fail {
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
        let previous = ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST
            .with(|active| active.replace(Some(GraphPostSubmitControlForTest::Pause(reached))));
        Self {
            reached: Some(observed),
            scope_resolution_observed: None,
            previous,
        }
    }

    pub(crate) fn accounting_fault() -> Self {
        let previous = ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST
            .with(|active| active.replace(Some(GraphPostSubmitControlForTest::AccountingFault)));
        Self {
            reached: None,
            scope_resolution_observed: None,
            previous,
        }
    }

    #[cfg(feature = "render-window")]
    pub(crate) fn terminal_loss() -> Self {
        let (scope_resolution_observed, observed) = sync_channel(1);
        let previous = ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(GraphPostSubmitControlForTest::TerminalLoss {
                scope_resolution_observed,
            }))
        });
        Self {
            reached: None,
            scope_resolution_observed: Some(observed),
            previous,
        }
    }

    #[cfg(feature = "render-window")]
    pub(crate) fn present_failing() -> Self {
        let (scope_resolution_observed, observed) = sync_channel(1);
        let previous = ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(GraphPostSubmitControlForTest::PresentFail {
                scope_resolution_observed,
            }))
        });
        Self {
            reached: None,
            scope_resolution_observed: Some(observed),
            previous,
        }
    }

    pub(crate) fn wait_for_submission_for_test(&self, deadline: std::time::Duration) {
        self.reached
            .as_ref()
            .expect("only a paused graph control has a submission receiver")
            .recv_timeout(deadline)
            .expect("the production graph did not reach its post-submit checkpoint");
    }

    pub(crate) fn scope_resolution_observed_for_test(&self) -> bool {
        self.scope_resolution_observed
            .as_ref()
            .is_some_and(|observed| observed.try_recv().is_ok())
    }
}

#[cfg(test)]
impl Drop for ScopedGraphPostSubmitControlForTest {
    fn drop(&mut self) {
        ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
impl GraphPostSubmitControlForTest {
    async fn apply(
        self,
        device: &wgpu::Device,
        _signal: &DeviceSignal,
        prepared_frame: &PendingPreparedFrameCommit,
    ) {
        match self {
            Self::Fail { .. } => {
                let _ = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Surgeist test-injected graph validation failure"),
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
            #[cfg(feature = "render-window")]
            Self::TerminalLoss { .. } => {
                _signal.record_loss_for_test(DeviceLossReason::Destroyed);
            }
            #[cfg(feature = "render-window")]
            Self::PresentFail { .. } => {}
            Self::AccountingFault => {
                let _ = prepared_frame.poison_retained_byte_accounting_for_test();
            }
            Self::Pause(reached) => {
                reached
                    .send(())
                    .expect("the graph test must observe the post-submit checkpoint");
                std::future::pending::<()>().await;
            }
        }
    }

    fn notify_submission_scope_resolution(&self) {
        let observed = match self {
            Self::Fail {
                scope_resolution_observed,
            } => Some(scope_resolution_observed),
            #[cfg(feature = "render-window")]
            Self::TerminalLoss {
                scope_resolution_observed,
            } => Some(scope_resolution_observed),
            _ => None,
        };
        if let Some(observed) = observed {
            let _ = observed.send(());
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn apply_present_failure(&self, device: &wgpu::Device) {
        #[cfg(feature = "render-window")]
        if matches!(self, Self::PresentFail { .. }) {
            let _ = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Surgeist test-injected graph present validation failure"),
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
        #[cfg(not(feature = "render-window"))]
        let _ = device;
    }
}

fn active_graph_post_submit_control_for_test() -> Option<GraphPostSubmitControlForTest> {
    ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| active.borrow().clone())
}

pub(super) fn begin_graph_submission_observation_for_test(
    transaction_generation: u64,
    active_generation: Option<u64>,
    capture_lease_count: usize,
    prepared_frame_resource_identities: Vec<crate::resource::ResourceIdentity>,
) -> GraphSubmissionRawFact {
    let generic_observation = record_active_gpu_operation_submission_for_test(
        transaction_generation,
        active_generation,
        None,
    );
    let graph_observation = record_active_base_graph_submission_for_test(
        transaction_generation,
        active_generation,
        capture_lease_count,
        prepared_frame_resource_identities,
    );
    let control = active_graph_post_submit_control_for_test();
    let id = NEXT_GRAPH_SUBMISSION_RAW_FACT_ID_FOR_TEST
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("graph submission test raw-fact identity must remain available");
    {
        let previous = lock_graph_submission_states_for_test().insert(
            id,
            GraphSubmissionStateForTest {
                generic_observation,
                graph_observation,
                control,
            },
        );
        assert!(
            previous.is_none(),
            "graph submission test raw-fact identity must be unique"
        );
    }
    GraphSubmissionRawFact { id }
}

pub(super) async fn apply_active_graph_post_submit_control_for_test(
    raw_fact: &GraphSubmissionRawFact,
    device: &wgpu::Device,
    signal: &DeviceSignal,
    prepared_frame: &PendingPreparedFrameCommit,
) {
    if let Some(control) = graph_submission_state_for_test(raw_fact).control {
        control.apply(device, signal, prepared_frame).await;
    }
}

pub(super) fn record_active_graph_headless_scope_resolution_for_test(
    raw_fact: &GraphSubmissionRawFact,
) {
    let state = graph_submission_state_for_test(raw_fact);
    if let Some(observation) = state.generic_observation {
        observation.record_scope_resolution(false);
    }
    if let Some(observation) = state.graph_observation {
        observation.record_scope_resolution();
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(super) fn record_active_graph_submission_scope_resolution_for_test(
    raw_fact: &GraphSubmissionRawFact,
) {
    if let Some(observation) = graph_submission_state_for_test(raw_fact).graph_observation {
        observation.record_scope_resolution();
    }
}

pub(super) fn notify_active_graph_submission_scope_resolution_for_test(
    raw_fact: &GraphSubmissionRawFact,
) {
    if let Some(control) = graph_submission_state_for_test(raw_fact).control {
        control.notify_submission_scope_resolution();
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(super) fn apply_active_graph_present_failure_for_test(
    raw_fact: &GraphSubmissionRawFact,
    device: &wgpu::Device,
) {
    if let Some(control) = graph_submission_state_for_test(raw_fact).control {
        control.apply_present_failure(device);
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(super) fn record_active_graph_presentation_scope_resolution_for_test(
    raw_fact: &GraphSubmissionRawFact,
) {
    let state = graph_submission_state_for_test(raw_fact);
    if let Some(observation) = state.generic_observation {
        observation.record_scope_resolution(false);
    }
    if let Some(observation) = state.graph_observation {
        observation.record_presentation_scope_resolution();
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(super) fn notify_active_graph_presentation_scope_resolution_for_test(
    raw_fact: &GraphSubmissionRawFact,
) {
    #[cfg(feature = "render-window")]
    if let Some(GraphPostSubmitControlForTest::PresentFail {
        scope_resolution_observed,
    }) = graph_submission_state_for_test(raw_fact).control
    {
        let _ = scope_resolution_observed.send(());
    }
    #[cfg(not(feature = "render-window"))]
    let _ = raw_fact;
}

pub(super) fn record_active_graph_commit_for_test(
    raw_fact: &GraphSubmissionRawFact,
    output: &GraphOutputCommit,
    retention: crate::resource::ResourceRetentionOutcome,
) {
    if let Some(observation) = graph_submission_state_for_test(raw_fact).graph_observation {
        observation.record_commit(output.observation_kind_for_test(), retention);
    }
}

fn graph_submission_state_for_test(
    raw_fact: &GraphSubmissionRawFact,
) -> GraphSubmissionStateForTest {
    lock_graph_submission_states_for_test()
        .get(&raw_fact.id)
        .cloned()
        .expect("graph submission test raw facts must remain bound until completion")
}

fn lock_graph_submission_states_for_test()
-> std::sync::MutexGuard<'static, BTreeMap<u64, GraphSubmissionStateForTest>> {
    GRAPH_SUBMISSION_STATE_FOR_TEST
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Drop for GraphSubmissionRawFact {
    fn drop(&mut self) {
        let removed = lock_graph_submission_states_for_test().remove(&self.id);
        debug_assert!(
            removed.is_some(),
            "graph submission test raw facts must be removed exactly once"
        );
    }
}
