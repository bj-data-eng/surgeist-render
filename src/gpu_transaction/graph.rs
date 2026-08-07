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

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::AcquiredPresentedSurfaceTexture;

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
        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = device;
        let GraphSubmissionPayload {
            command_buffer,
            capture_resources,
            prepared_frame,
            activity,
            output,
        } = payload;
        self.submit_graph_command(queue, command_buffer).await;
        let resources = GraphSubmittedResources {
            capture_resources,
            prepared_frame,
        };
        let (output, frame_cleanup) = match output {
            PendingGraphHostEffect::Headless(publication) => {
                self.finish_graph_headless(publication, resources, pass_cache, operation)
                    .await?
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            PendingGraphHostEffect::Presented(effect) => {
                self.finish_base_graph_presented(device, effect, resources, pass_cache, operation)
                    .await?
            }
        };
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
    ) -> Result<(GraphOutputCommit, FrameCleanup)> {
        let result = self.finish(operation).await;
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
    ) -> Result<(GraphOutputCommit, FrameCleanup)> {
        let mut transaction = self;
        let result = transaction.resolve_submission_phase(operation).await;
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
        let result = transaction.finish(operation).await;
        result?;
        let frame_cleanup = resources.commit(pass_cache)?;
        Ok((output, frame_cleanup))
    }

    async fn submit_graph_command(&self, queue: &wgpu::Queue, command_buffer: wgpu::CommandBuffer) {
        queue.submit([command_buffer]);
    }
}
