#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::present;
use super::{Backend, DeviceSlotIdentity, create_headless_texture};

use crate::pass::{
    BasePreparableGraph, CompositionPreparableGraph, EncodedGpuGraphActivity,
    GraphExternalOutputView, PreparedGraph,
};
use crate::resource::{FrameCleanup, WorkingFormat};
use crate::stats::GpuGraphStatsObservation;
use crate::surface::{HeadlessPublication, SurfaceBackend};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::{PresentedSurface, PresentedSurfaceState};
use crate::vello_engine::{
    ActiveVelloEncodingScope, EncodedVelloPass, RasterParameters, TransactionEncodingState,
    TransactionTargetIntent, scene::VelloScene,
};
use crate::{
    Antialiasing, BackendErrorCode, Color, Error, Format, Parameters, PhysicalSize, Result,
    RuntimeCapabilityUnavailableReason, RuntimeOperation, Stats, Surface,
    gpu_transaction::{
        GpuOperationStage, GpuOperationTransaction, GraphOutputCommit, GraphSubmissionPayload,
        InternalVelloPayload,
    },
    shader::ProvisionalDevicePassCacheUpdate,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

/// One validated exact graph selected for atomic surface execution.
#[must_use = "an exact surface graph must enter its GPU transaction"]
pub(crate) enum ExactSurfaceGraph {
    Base(BasePreparableGraph),
    Composition(CompositionPreparableGraph),
    Backdrop(crate::pass::BackdropPreparableGraph),
}

impl ExactSurfaceGraph {
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        match self {
            Self::Base(preparable) => preparable.working_format(),
            Self::Composition(preparable) => preparable.working_format(),
            Self::Backdrop(preparable) => preparable.working_format(),
        }
    }

    pub(crate) const fn output_format(&self) -> Format {
        match self {
            Self::Base(preparable) => preparable.output_format(),
            Self::Composition(preparable) => preparable.output_format(),
            Self::Backdrop(preparable) => preparable.output_format(),
        }
    }

    pub(super) fn known_output_extent(&self) -> Result<Option<PhysicalSize>> {
        match self {
            Self::Base(preparable) => preparable.output_extent().map(Some),
            Self::Composition(_) => Ok(None),
            Self::Backdrop(preparable) => preparable.output_extent().map(Some),
        }
    }
}

/// Exact texture-bound raster stage shared by direct surface and local offscreen execution.
pub(super) struct InternalVelloTextureTarget<'a> {
    view: &'a wgpu::TextureView,
    extent: PhysicalSize,
    usage: wgpu::TextureUsages,
}

impl<'a> InternalVelloTextureTarget<'a> {
    pub(super) const fn new(
        view: &'a wgpu::TextureView,
        extent: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            view,
            extent,
            usage,
        }
    }
}

pub(super) struct InternalVelloTextureStage<'a> {
    identity: DeviceSlotIdentity,
    operation: RuntimeOperation,
    scene: &'a VelloScene,
    target: InternalVelloTextureTarget<'a>,
    base_color: Color,
    antialiasing: Antialiasing,
}

impl<'a> InternalVelloTextureStage<'a> {
    pub(super) fn new(
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
        scene: &'a VelloScene,
        target: InternalVelloTextureTarget<'a>,
        base_color: Color,
        antialiasing: Antialiasing,
    ) -> Self {
        Self {
            identity,
            operation,
            scene,
            target,
            base_color,
            antialiasing,
        }
    }
}

impl Backend {
    fn create_headless_surface_texture(
        &mut self,
        identity: DeviceSlotIdentity,
        physical_size: PhysicalSize,
        format: Format,
    ) -> Result<(wgpu::Texture, wgpu::TextureView)> {
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "headless Vello device resources are unavailable before allocation",
        )?;
        create_headless_texture(&ready.device, physical_size, format)
    }

    pub(super) async fn render_internal_vello_to_texture(
        &mut self,
        transaction: GpuOperationTransaction,
        stage: InternalVelloTextureStage<'_>,
    ) -> Result<()> {
        let prepared = stage.scene.prepare_raster(RasterParameters::try_new(
            stage.target.extent,
            peniko::Color::from(stage.base_color),
            stage.antialiasing,
        )?)?;
        {
            let ready = self.ready_state_mut(
                stage.identity,
                stage.operation,
                BackendErrorCode::RenderFailed,
                "internal Vello device resources are unavailable before rendering",
            )?;
            let mut command_encoder =
                ready
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Surgeist internal Vello frame encoder"),
                    });
            let mut scope = ActiveVelloEncodingScope::begin(&ready.device);
            let encoded: EncodedVelloPass = {
                let mut encoding = TransactionEncodingState::new(
                    &mut scope,
                    &ready.queue,
                    &mut command_encoder,
                    stage.target.view,
                    TransactionTargetIntent::new(
                        stage.target.extent,
                        wgpu::TextureFormat::Rgba8Unorm,
                        stage.target.usage,
                    ),
                );
                match prepared.encode_into(&ready.engine, &ready.resources, &mut encoding) {
                    Ok(encoded) => encoded,
                    Err(failure) => {
                        return Err(failure.into_error_and_aborted_resources().0);
                    }
                }
            };
            let (lease, logical_pass) = encoded.into_resources_and_logical_pass();
            let lease = match scope.finish_with_lease(lease).await {
                Ok(lease) => lease,
                Err(failure) => {
                    return Err(failure.into_error_and_aborted_resources().0);
                }
            };
            let payload = InternalVelloPayload::new(
                command_encoder.finish(),
                crate::vello_engine::PendingVelloResourceCommit::new(lease),
                logical_pass,
            );
            transaction
                .submit_internal_vello(&ready.device, &ready.queue, payload, stage.operation)
                .await?;
        }
        self.commit_checked_pass_cache_update(stage.identity, None, stage.operation)
    }

    pub(super) fn commit_checked_pass_cache_update(
        &mut self,
        identity: DeviceSlotIdentity,
        update: Option<ProvisionalDevicePassCacheUpdate>,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(update) = update else {
            return Ok(());
        };
        let ready = self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::RenderFailed,
            "checked core pass objects lost their persistent device cache",
        )?;
        update.commit(&mut ready.pass_cache)
    }

    pub(crate) fn begin_gpu_operation(
        &mut self,
        identity: DeviceSlotIdentity,
        stage: GpuOperationStage,
        operation: RuntimeOperation,
    ) -> Result<GpuOperationTransaction> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                stage.error_code(),
                "GPU device slot disappeared before transaction setup",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                stage.error_code(),
                "GPU device generation changed before transaction setup",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(operation));
        }
        state.next_operation_generation = state
            .next_operation_generation
            .checked_add(1)
            .ok_or_else(|| {
                Error::invalid_value(
                    "GPU operation generation",
                    state.next_operation_generation,
                    "must have remaining generation space",
                )
            })?;
        let signal = Arc::clone(&state.signal);
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                stage.error_code(),
                "GPU device slot disappeared before transaction scopes",
            )
        })?;
        Ok(GpuOperationTransaction::begin(
            &ready.device,
            signal,
            state.next_operation_generation,
            stage,
        ))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "base graph calls the validated prepared-graph handoff before execution"
        )
    )]
    pub(crate) fn prepare_graph_resources(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: crate::pass::LoweredGraphPlan,
        policy: crate::EffectQualityPolicy,
    ) -> Result<PreparedGraph<'_>> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot is unavailable for graph preparation",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before graph preparation",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(RuntimeOperation::EffectRendering));
        }
        let capabilities = state.capabilities;
        let realize_checked_passes = state.signal.has_active_operation();
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "ready GPU device resources disappeared before graph preparation",
            )
        })?;
        PreparedGraph::try_prepare(
            lowered,
            policy,
            &capabilities,
            &ready.device,
            &ready.queue,
            &ready.resources,
            (&ready.pass_cache, realize_checked_passes),
        )
        .map(|prepared| prepared.with_vello_engine(&ready.engine))
    }

    fn prepare_exact_surface_graph_resources(
        &mut self,
        identity: DeviceSlotIdentity,
        graph: ExactSurfaceGraph,
    ) -> Result<PreparedGraph<'_>> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot is unavailable for exact graph preparation",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before exact graph preparation",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(RuntimeOperation::SurfaceRendering));
        }
        if !state.signal.has_active_operation() {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "exact graph preparation requires one active GPU transaction",
            ));
        }
        let selected_working_format = graph.working_format();
        let capabilities = state
            .capabilities
            .for_selected_working_format(selected_working_format)?;
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "ready GPU device resources disappeared before exact graph preparation",
            )
        })?;
        let prepared = match graph {
            ExactSurfaceGraph::Base(preparable) => {
                PreparedGraph::try_prepare_base_with_working_format(
                    preparable,
                    selected_working_format,
                    &capabilities,
                    &ready.device,
                    &ready.queue,
                    &ready.resources,
                    (&ready.pass_cache, true),
                )
            }
            ExactSurfaceGraph::Composition(preparable) => PreparedGraph::try_prepare_composition(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            ExactSurfaceGraph::Backdrop(preparable) => PreparedGraph::try_prepare_backdrop(
                preparable,
                selected_working_format,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
        }?
        .with_vello_engine(&ready.engine);
        Ok(prepared)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RenderTimings {
    pub(crate) render_time: Duration,
    pub(crate) present_time: Duration,
}

/// Private result of a clean frame transaction, held until the renderer publishes it.
#[must_use = "clean frame results must be committed or dropped"]
pub(crate) struct SurfaceFrameCommit {
    timings: RenderTimings,
    headless_publication: Option<HeadlessPublication>,
    _frame_cleanup: Option<FrameCleanup>,
    stats_observation: Option<GpuGraphStatsObservation>,
}

impl SurfaceFrameCommit {
    pub(super) fn without_headless_publication(timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: None,
            _frame_cleanup: None,
            stats_observation: None,
        }
    }

    fn headless(publication: HeadlessPublication, timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: Some(publication),
            _frame_cleanup: None,
            stats_observation: None,
        }
    }

    pub(super) fn headless_graph(
        publication: HeadlessPublication,
        frame_cleanup: FrameCleanup,
        graph_activity: EncodedGpuGraphActivity,
        working_format: WorkingFormat,
        timings: RenderTimings,
    ) -> Self {
        let stats_observation =
            GpuGraphStatsObservation::after_cleanup(working_format, graph_activity, &frame_cleanup);
        Self {
            timings,
            headless_publication: Some(publication),
            _frame_cleanup: Some(frame_cleanup),
            stats_observation: Some(stats_observation),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(super) fn presented_graph(
        frame_cleanup: FrameCleanup,
        graph_activity: EncodedGpuGraphActivity,
        working_format: WorkingFormat,
        timings: RenderTimings,
    ) -> Self {
        let stats_observation =
            GpuGraphStatsObservation::after_cleanup(working_format, graph_activity, &frame_cleanup);
        Self {
            timings,
            headless_publication: None,
            _frame_cleanup: Some(frame_cleanup),
            stats_observation: Some(stats_observation),
        }
    }

    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    pub(crate) fn apply_stats_observation(&self, stats: &mut Stats) {
        if let Some(observation) = self.stats_observation {
            observation.apply_to(stats);
        }
    }

    pub(crate) fn commit(self, surface: &mut Surface) {
        if let Some(publication) = self.headless_publication {
            surface.commit_headless_publication(publication);
        }
    }
}

pub(crate) async fn render_exact_headless_graph_surface(
    backend: &mut Backend,
    surface: &Surface,
    graph: ExactSurfaceGraph,
) -> Result<SurfaceFrameCommit> {
    let (device_identity, physical_size, selected_working_format) =
        exact_headless_graph_target(surface, &graph)?;
    let capabilities = backend
        .device_capabilities(device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the exact graph executor lost immutable device capabilities",
            )
        })?;
    capabilities.validate_supported_working_format(selected_working_format)?;

    let transaction = backend.begin_gpu_operation(
        device_identity,
        GpuOperationStage::Render,
        RuntimeOperation::SurfaceRendering,
    )?;
    let (device, queue) = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the exact graph executor lost its ready device before draft allocation",
        )?;
        (ready.device.clone(), ready.queue.clone())
    };
    let render_start = Instant::now();
    let (draft_texture, draft_view) =
        create_headless_texture(&device, physical_size, surface.options.format)?;
    let mut prepared = backend.prepare_exact_surface_graph_resources(device_identity, graph)?;
    if prepared.output_extent()? != physical_size
        || prepared.output_format() != surface.options.format
        || prepared.working_format() != selected_working_format
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "prepared exact graph output changed after eligibility validation",
        ));
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist exact headless graph encoder"),
    });
    let pending_encoding = prepared
        .encode_custom_spine(
            &mut encoder,
            GraphExternalOutputView::try_new(&draft_view, surface.options.format, physical_size)?,
        )
        .await;
    let pending_encoding = pending_encoding?;
    let prepared_submission = prepared.finish_graph_submission(pending_encoding)?;
    let payload = GraphSubmissionPayload::new(
        encoder.finish(),
        prepared_submission,
        HeadlessPublication::new(draft_texture),
    );
    let clean = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the exact graph executor lost its ready device before submission",
        )?;
        transaction
            .submit_base_graph(
                &device,
                &queue,
                &mut ready.pass_cache,
                payload,
                RuntimeOperation::SurfaceRendering,
            )
            .await?
    };
    let (output, frame_cleanup, graph_activity) = clean.into_parts();
    #[cfg(not(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )))]
    let GraphOutputCommit::Headless(publication) = output;
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    let publication = match output {
        GraphOutputCommit::Headless(publication) => publication,
        GraphOutputCommit::Presented => {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the headless exact graph transaction returned a presented host effect",
            ));
        }
    };
    Ok(SurfaceFrameCommit::headless_graph(
        publication,
        frame_cleanup,
        graph_activity,
        selected_working_format,
        RenderTimings {
            render_time: render_start.elapsed(),
            present_time: Duration::ZERO,
        },
    ))
}

fn exact_headless_graph_target(
    surface: &Surface,
    graph: &ExactSurfaceGraph,
) -> Result<(DeviceSlotIdentity, PhysicalSize, WorkingFormat)> {
    let selected_working_format = graph.working_format();
    let graph_output_format = graph.output_format();
    let known_output_extent = graph.known_output_extent()?;
    let (device_identity, physical_size) = match &surface.backend {
        SurfaceBackend::Headless {
            device_identity,
            physical_size,
            ..
        } => (*device_identity, *physical_size),
        SurfaceBackend::ContractOnly { .. } => {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the exact graph executor requires a device-backed headless surface",
            ));
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        SurfaceBackend::Presented { .. } => {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "presented exact graph execution requires the presented executor",
            ));
        }
    };
    if physical_size.width() == 0
        || physical_size.height() == 0
        || surface.options.format != Format::Rgba8
        || graph_output_format != surface.options.format
        || known_output_extent.is_some_and(|extent| extent != physical_size)
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "the headless draft differs from the exact eligible graph output",
        ));
    }
    Ok((device_identity, physical_size, selected_working_format))
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) async fn render_exact_presented_graph_surface(
    backend: &mut Backend,
    surface: &mut Surface,
    graph: ExactSurfaceGraph,
) -> Result<SurfaceFrameCommit> {
    let (device_identity, physical_size, output_format, selected_working_format) =
        present::exact_presented_graph_target(
            surface,
            graph.working_format(),
            graph.output_format(),
            graph.known_output_extent()?,
        )?;
    let capabilities = backend
        .device_capabilities(device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the presented exact graph executor lost immutable device capabilities",
            )
        })?;
    capabilities.validate_supported_working_format(selected_working_format)?;

    let transaction = backend.begin_gpu_operation(
        device_identity,
        GpuOperationStage::Render,
        RuntimeOperation::SurfaceRendering,
    )?;
    let (device, queue) = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the presented exact graph lost its ready device before preparation",
        )?;
        (ready.device.clone(), ready.queue.clone())
    };
    let render_start = Instant::now();
    let prepared = backend.prepare_exact_surface_graph_resources(device_identity, graph)?;
    if prepared.output_extent()? != physical_size
        || prepared.output_format() != output_format
        || prepared.working_format() != selected_working_format
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "prepared presented exact graph output changed after eligibility validation",
        ));
    }

    let present_start = Instant::now();
    let acquired =
        present::acquire_exact_presented_graph_texture(surface, &device, prepared, transaction)
            .await?;
    let (acquired, mut prepared, transaction) = acquired;
    let output_view = acquired.create_view();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist exact presented graph encoder"),
    });
    let pending_encoding = prepared
        .encode_custom_spine(
            &mut encoder,
            GraphExternalOutputView::try_new(&output_view, output_format, physical_size)?,
        )
        .await;
    let pending_encoding = pending_encoding?;
    let prepared_submission = prepared.finish_graph_submission(pending_encoding)?;
    drop(output_view);
    let payload =
        GraphSubmissionPayload::presented(encoder.finish(), prepared_submission, acquired);
    let clean = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the presented exact graph lost its ready device before submission",
        )?;
        transaction
            .submit_base_graph(
                &device,
                &queue,
                &mut ready.pass_cache,
                payload,
                RuntimeOperation::SurfaceRendering,
            )
            .await?
    };
    let (output, frame_cleanup, graph_activity) = clean.into_parts();
    if !matches!(output, GraphOutputCommit::Presented) {
        return Err(Error::new(
            BackendErrorCode::PresentFailed,
            "the presented exact graph transaction returned a headless publication",
        ));
    }
    Ok(SurfaceFrameCommit::presented_graph(
        frame_cleanup,
        graph_activity,
        selected_working_format,
        RenderTimings {
            render_time: present_start.duration_since(render_start),
            present_time: present_start.elapsed(),
        },
    ))
}

pub(crate) async fn render_internal_vello_surface(
    backend: &mut Backend,
    transaction: GpuOperationTransaction,
    surface: &mut Surface,
    scene: &VelloScene,
    parameters: Parameters,
    antialiasing: Antialiasing,
) -> Result<SurfaceFrameCommit> {
    let frame = InternalVelloFrameParameters {
        scene,
        parameters,
        antialiasing,
    };
    match &mut surface.backend {
        SurfaceBackend::ContractOnly { .. } => Ok(
            SurfaceFrameCommit::without_headless_publication(RenderTimings::default()),
        ),
        SurfaceBackend::Headless {
            device_identity,
            physical_size,
            ..
        } => {
            render_internal_vello_headless(
                backend,
                transaction,
                *device_identity,
                *physical_size,
                surface.options.format,
                frame,
            )
            .await
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        SurfaceBackend::Presented {
            surface: native,
            device_identity,
            state,
        } => {
            render_internal_vello_presented(
                backend,
                transaction,
                native,
                *device_identity,
                state,
                frame,
            )
            .await
        }
    }
}

#[derive(Clone, Copy)]
struct InternalVelloFrameParameters<'a> {
    scene: &'a VelloScene,
    parameters: Parameters,
    antialiasing: Antialiasing,
}

async fn render_internal_vello_headless(
    backend: &mut Backend,
    transaction: GpuOperationTransaction,
    device_identity: DeviceSlotIdentity,
    physical_size: PhysicalSize,
    format: Format,
    frame: InternalVelloFrameParameters<'_>,
) -> Result<SurfaceFrameCommit> {
    if physical_size.width() == 0 || physical_size.height() == 0 {
        return Ok(SurfaceFrameCommit::without_headless_publication(
            RenderTimings::default(),
        ));
    }
    let (texture, view) =
        backend.create_headless_surface_texture(device_identity, physical_size, format)?;
    let render_start = Instant::now();
    backend
        .render_internal_vello_to_texture(
            transaction,
            InternalVelloTextureStage::new(
                device_identity,
                RuntimeOperation::SurfaceRendering,
                frame.scene,
                InternalVelloTextureTarget::new(
                    &view,
                    physical_size,
                    wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                ),
                frame.parameters.base_color,
                frame.antialiasing,
            ),
        )
        .await?;
    Ok(SurfaceFrameCommit::headless(
        HeadlessPublication::new(texture),
        RenderTimings {
            render_time: render_start.elapsed(),
            present_time: Duration::ZERO,
        },
    ))
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
async fn render_internal_vello_presented(
    backend: &mut Backend,
    transaction: GpuOperationTransaction,
    native: &mut PresentedSurface,
    device_identity: DeviceSlotIdentity,
    state: &mut PresentedSurfaceState,
    frame: InternalVelloFrameParameters<'_>,
) -> Result<SurfaceFrameCommit> {
    let Some(resources) = present::internal_vello_presented_resources(native, state)? else {
        return Ok(SurfaceFrameCommit::without_headless_publication(
            RenderTimings::default(),
        ));
    };
    let _ = &resources.target_texture;
    let render_start = Instant::now();
    backend
        .render_internal_vello_to_texture(
            transaction,
            InternalVelloTextureStage::new(
                device_identity,
                RuntimeOperation::SurfaceRendering,
                frame.scene,
                InternalVelloTextureTarget::new(
                    &resources.target_view,
                    PhysicalSize::new(resources.config.width, resources.config.height),
                    wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
                frame.parameters.base_color,
                frame.antialiasing,
            ),
        )
        .await?;
    let render_time = render_start.elapsed();
    present::present_internal_vello_target(
        backend,
        native,
        device_identity,
        state,
        resources,
        render_time,
    )
    .await
}
