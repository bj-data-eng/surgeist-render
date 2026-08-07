#[cfg(test)]
use super::{
    GpuOperationSubmissionObservationForTest, record_active_gpu_operation_submission_for_test,
    wait_at_active_gpu_operation_post_submit_checkpoint_for_test,
};
use super::{GpuOperationTransaction, VelloResourceCommitProof};
use crate::{
    Result, RuntimeOperation,
    pass::{
        AccountingReadyPreparedFrameCommit, EncodedGpuGraphActivity, PendingPreparedFrameCommit,
        PreparedGraphSubmission,
    },
    resource::FrameCleanup,
    shader::DevicePassCache,
    surface::HeadlessPublication,
    vello_engine::{AccountingReadyVelloResourceCommit, PendingVelloResourceCommit},
};

#[cfg(all(test, feature = "render-window"))]
use crate::DeviceLossReason;
#[cfg(test)]
use crate::backend::DeviceSignal;
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::AcquiredPresentedSurfaceTexture;

#[cfg(test)]
use std::{
    cell::RefCell,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender, sync_channel},
    },
};

#[cfg(test)]
thread_local! {
    static ACTIVE_GRAPH_SUBMISSION_OBSERVATION_FOR_TEST: RefCell<Option<GraphSubmissionObservationForTest>> = const { RefCell::new(None) };
    static ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST: RefCell<Option<GraphPostSubmitControlForTest>> = const { RefCell::new(None) };
}

/// One complete graph draft whose effects remain private until the owning
/// transaction resolves cleanly.
#[must_use = "graph payloads must be submitted or dropped atomically"]
pub(crate) struct GraphSubmissionPayload {
    command_buffer: wgpu::CommandBuffer,
    capture_resources: PendingVelloResourceCommit,
    prepared_frame: PendingPreparedFrameCommit,
    activity: EncodedGpuGraphActivity,
    output: PendingGraphHostEffect,
}

/// A clean graph result that the renderer may publish with frame stats.
#[must_use = "clean graph results must reach the renderer publication gate"]
pub(crate) struct GraphSubmissionCommit {
    output: GraphOutputCommit,
    frame_cleanup: FrameCleanup,
    activity: EncodedGpuGraphActivity,
}

struct GraphSubmittedCommand {
    #[cfg(test)]
    generic_observation: Option<GpuOperationSubmissionObservationForTest>,
    #[cfg(test)]
    graph_observation: Option<GraphSubmissionObservationForTest>,
    #[cfg(test)]
    control: Option<GraphPostSubmitControlForTest>,
}

struct GraphSubmittedResources {
    capture_resources: PendingVelloResourceCommit,
    prepared_frame: PendingPreparedFrameCommit,
}

/// The only output effects that one pending graph transaction may own.
#[must_use = "pending graph host effects must resolve through their transaction"]
enum PendingGraphHostEffect {
    Headless(HeadlessPublication),
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented(PendingGraphPresentedHostEffect),
}

/// Non-clone ownership of one acquired presented output before clean submission.
#[must_use = "an acquired base graph presented output must be authorized or discarded"]
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
struct PendingGraphPresentedHostEffect {
    acquired: AcquiredPresentedSurfaceTexture,
}

/// Sealed proof that submission scopes and the active terminal signal are clean.
struct CleanGraphSubmissionReceipt {
    _resource_readiness: GraphResourceReadinessReceipt,
}

/// Sealed one-shot evidence that both pending resource owners and the
/// provisional cache passed exact readiness checks before host authorization.
struct GraphResourceReadinessReceipt {
    _private: (),
}

#[must_use = "accounting-ready graph resources must commit or abort on drop"]
struct AccountingReadyGraphResources {
    capture_resources: AccountingReadyVelloResourceCommit,
    prepared_frame: AccountingReadyPreparedFrameCommit,
}

/// A host effect that can exist only after clean graph submission.
#[must_use = "clean graph host effects must be applied exactly once"]
enum CleanGraphHostEffect {
    Headless(HeadlessPublication),
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented(CleanGraphPresentedHostEffect),
}

#[must_use = "clean acquired presented outputs must be presented exactly once"]
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
struct CleanGraphPresentedHostEffect {
    acquired: AcquiredPresentedSurfaceTexture,
}

pub(crate) enum GraphOutputCommit {
    Headless(HeadlessPublication),
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented,
}

impl GraphSubmissionPayload {
    pub(crate) fn new(
        command_buffer: wgpu::CommandBuffer,
        prepared: PreparedGraphSubmission,
        headless_draft: HeadlessPublication,
    ) -> Self {
        let (capture_resources, prepared_frame, activity) = prepared.into_parts();
        Self {
            command_buffer,
            capture_resources,
            prepared_frame,
            activity,
            output: PendingGraphHostEffect::Headless(headless_draft),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(crate) fn presented(
        command_buffer: wgpu::CommandBuffer,
        prepared: PreparedGraphSubmission,
        acquired: AcquiredPresentedSurfaceTexture,
    ) -> Self {
        let (capture_resources, prepared_frame, activity) = prepared.into_parts();
        Self {
            command_buffer,
            capture_resources,
            prepared_frame,
            activity,
            output: PendingGraphHostEffect::Presented(PendingGraphPresentedHostEffect { acquired }),
        }
    }
}

impl AccountingReadyGraphResources {
    fn try_new(
        capture_resources: PendingVelloResourceCommit,
        prepared_frame: PendingPreparedFrameCommit,
        pass_cache: &DevicePassCache,
    ) -> Result<Self> {
        let capture_resources = capture_resources.into_accounting_ready()?;
        let prepared_frame = prepared_frame.into_accounting_ready(pass_cache)?;
        Ok(Self {
            capture_resources,
            prepared_frame,
        })
    }

    fn authorization_receipt(
        &self,
        pass_cache: &DevicePassCache,
    ) -> Result<GraphResourceReadinessReceipt> {
        self.capture_resources.ensure_commit_ready()?;
        self.prepared_frame.ensure_commit_ready(pass_cache)?;
        Ok(GraphResourceReadinessReceipt { _private: () })
    }

    fn commit(self, pass_cache: &mut DevicePassCache) -> Result<FrameCleanup> {
        self.capture_resources.ensure_commit_ready()?;
        self.prepared_frame.ensure_commit_ready(pass_cache)?;
        let capture_cleanup = self
            .capture_resources
            .commit(VelloResourceCommitProof { _private: () })?;
        let prepared_cleanup = self.prepared_frame.commit(pass_cache)?;
        Ok(capture_cleanup.followed_by(prepared_cleanup))
    }
}

impl GraphSubmissionCommit {
    pub(crate) fn into_parts(self) -> (GraphOutputCommit, FrameCleanup, EncodedGpuGraphActivity) {
        (self.output, self.frame_cleanup, self.activity)
    }
}

impl PendingGraphHostEffect {
    fn authorize(self, _receipt: CleanGraphSubmissionReceipt) -> CleanGraphHostEffect {
        match self {
            Self::Headless(publication) => CleanGraphHostEffect::Headless(publication),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            Self::Presented(effect) => {
                CleanGraphHostEffect::Presented(CleanGraphPresentedHostEffect {
                    acquired: effect.acquired,
                })
            }
        }
    }
}

impl CleanGraphHostEffect {
    fn apply(self) -> GraphOutputCommit {
        match self {
            Self::Headless(publication) => GraphOutputCommit::Headless(publication),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            Self::Presented(effect) => {
                effect.acquired.present();
                GraphOutputCommit::Presented
            }
        }
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

impl GpuOperationTransaction {
    /// Submits one complete graph and commits every private graph-owned
    /// resource only after the transaction scopes and device signal are clean.
    pub(crate) async fn submit_base_graph(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass_cache: &mut DevicePassCache,
        payload: GraphSubmissionPayload,
        operation: RuntimeOperation,
    ) -> Result<GraphSubmissionCommit> {
        #[cfg(not(test))]
        let _ = device;
        let GraphSubmissionPayload {
            command_buffer,
            capture_resources,
            prepared_frame,
            activity,
            output,
        } = payload;
        let submitted = self
            .submit_graph_command(
                device,
                queue,
                command_buffer,
                &capture_resources,
                &prepared_frame,
            )
            .await;
        #[cfg(not(test))]
        let _ = submitted;
        let resources = GraphSubmittedResources {
            capture_resources,
            prepared_frame,
        };
        let (output, frame_cleanup) = match output {
            PendingGraphHostEffect::Headless(publication) => {
                self.finish_graph_headless(
                    publication,
                    resources,
                    pass_cache,
                    operation,
                    &submitted,
                )
                .await?
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            PendingGraphHostEffect::Presented(effect) => {
                self.finish_base_graph_presented(
                    device, effect, resources, pass_cache, operation, &submitted,
                )
                .await?
            }
        };
        #[cfg(test)]
        if let Some(observation) = submitted.graph_observation {
            observation.record_commit(
                output.observation_kind_for_test(),
                frame_cleanup.retention(),
            );
        }
        Ok(GraphSubmissionCommit {
            output,
            frame_cleanup,
            activity,
        })
    }

    async fn finish_graph_headless(
        self,
        publication: HeadlessPublication,
        submitted_resources: GraphSubmittedResources,
        pass_cache: &mut DevicePassCache,
        operation: RuntimeOperation,
        _submitted: &GraphSubmittedCommand,
    ) -> Result<(GraphOutputCommit, FrameCleanup)> {
        let result = self.finish(operation).await;
        #[cfg(test)]
        if let Some(observation) = &_submitted.generic_observation {
            observation.record_scope_resolution(false);
        }
        #[cfg(test)]
        if let Some(observation) = &_submitted.graph_observation {
            observation.record_scope_resolution();
        }
        #[cfg(test)]
        if let Some(control) = &_submitted.control {
            control.notify_submission_scope_resolution();
        }
        result?;
        let resources = AccountingReadyGraphResources::try_new(
            submitted_resources.capture_resources,
            submitted_resources.prepared_frame,
            pass_cache,
        )?;
        let receipt = resources.authorization_receipt(pass_cache)?;
        let output =
            PendingGraphHostEffect::Headless(publication).authorize(CleanGraphSubmissionReceipt {
                _resource_readiness: receipt,
            });
        let frame_cleanup = resources.commit(pass_cache)?;
        Ok((output.apply(), frame_cleanup))
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn finish_base_graph_presented(
        self,
        device: &wgpu::Device,
        effect: PendingGraphPresentedHostEffect,
        submitted_resources: GraphSubmittedResources,
        pass_cache: &mut DevicePassCache,
        operation: RuntimeOperation,
        _submitted: &GraphSubmittedCommand,
    ) -> Result<(GraphOutputCommit, FrameCleanup)> {
        let mut transaction = self;
        let result = transaction.resolve_submission_phase(operation).await;
        #[cfg(test)]
        if let Some(observation) = &_submitted.graph_observation {
            observation.record_scope_resolution();
        }
        #[cfg(test)]
        if let Some(control) = &_submitted.control {
            control.notify_submission_scope_resolution();
        }
        result?;
        let resources = AccountingReadyGraphResources::try_new(
            submitted_resources.capture_resources,
            submitted_resources.prepared_frame,
            pass_cache,
        )?;
        let receipt = resources.authorization_receipt(pass_cache)?;
        let clean =
            PendingGraphHostEffect::Presented(effect).authorize(CleanGraphSubmissionReceipt {
                _resource_readiness: receipt,
            });
        transaction.begin_present_phase(device, operation)?;
        let output = clean.apply();
        #[cfg(test)]
        if let Some(control) = &_submitted.control {
            control.apply_present_failure(device);
        }
        let result = transaction.finish(operation).await;
        #[cfg(test)]
        if let Some(observation) = &_submitted.generic_observation {
            observation.record_scope_resolution(false);
        }
        #[cfg(test)]
        if let Some(observation) = &_submitted.graph_observation {
            observation.record_presentation_scope_resolution();
        }
        #[cfg(all(test, feature = "render-window"))]
        if let Some(GraphPostSubmitControlForTest::PresentFail {
            scope_resolution_observed,
        }) = &_submitted.control
        {
            let _ = scope_resolution_observed.send(());
        }
        result?;
        let frame_cleanup = resources.commit(pass_cache)?;
        Ok((output, frame_cleanup))
    }

    async fn submit_graph_command(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_buffer: wgpu::CommandBuffer,
        capture_resources: &PendingVelloResourceCommit,
        prepared_frame: &PendingPreparedFrameCommit,
    ) -> GraphSubmittedCommand {
        #[cfg(not(test))]
        let _ = (device, capture_resources, prepared_frame);
        #[cfg(test)]
        let capture_lease_count = capture_resources.lease_count_for_test();
        #[cfg(test)]
        let prepared_frame_resource_identities = prepared_frame.resource_identities_for_test();
        queue.submit([command_buffer]);
        #[cfg(test)]
        let generic_observation = record_active_gpu_operation_submission_for_test(
            self.lease.generation(),
            self.lease.active_generation_for_test(),
            None,
        );
        #[cfg(test)]
        let graph_observation = record_active_base_graph_submission_for_test(
            self.lease.generation(),
            self.lease.active_generation_for_test(),
            capture_lease_count,
            prepared_frame_resource_identities,
        );
        #[cfg(test)]
        let control =
            ACTIVE_GRAPH_POST_SUBMIT_CONTROL_FOR_TEST.with(|active| active.borrow().clone());
        #[cfg(test)]
        if let Some(control) = control.clone() {
            control
                .apply(device, &self.lease.signal, prepared_frame)
                .await;
        }
        #[cfg(test)]
        wait_at_active_gpu_operation_post_submit_checkpoint_for_test().await;
        GraphSubmittedCommand {
            #[cfg(test)]
            generic_observation,
            #[cfg(test)]
            graph_observation,
            #[cfg(test)]
            control,
        }
    }
}
