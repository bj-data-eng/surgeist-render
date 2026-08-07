#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::present;
use super::{
    Backend, create_headless_texture,
    device::{
        DeviceCapabilities, DeviceSignal, DeviceSlotIdentity, DeviceState, DeviceTerminalSignal,
        ReadyDeviceState,
    },
    execute,
    offscreen::{self, OffscreenRenderTarget, OffscreenRenderedTextureLease},
};
use crate::{
    Antialiasing, Attachment, BlendMode, EffectQualityPolicy, ErrorCode, Extend, FilterList,
    Format, ImageQuality, Options, Parameters, PhysicalSize, Point, Rect, Surface, SurfaceOptions,
    capability::EffectPrecisionCapabilities,
    command::OffscreenBounds,
    error::{
        BackendErrorCode, DeviceLossReason, Error, GpuFaultKind, Result,
        RuntimeCapabilityUnavailableReason, RuntimeOperation,
    },
    gpu_transaction::{
        GpuOperationStage, GraphOutputCommit, GraphSubmissionPayload, InternalVelloPayload,
        test_support::{
            InternalVelloSubmissionActionForTest, InternalVelloSubmissionObservationForTest,
            InternalVelloSubmissionOutcomeForTest,
            finish_vello_resources_without_submission_for_test,
            hold_internal_vello_after_submit_for_test, submit_internal_vello_observed_for_test,
            vello_accounting_failure_after_submission_for_test,
            vello_scope_failure_after_submission_for_test,
        },
    },
    pass::{
        BackdropPreparableGraph, BasePreparableGraph, ColorFilterPreparableGraph,
        CompositionPreparableGraph, CorePassCacheRequestsForTest, GraphExternalOutputView,
        LayerCompositeCacheRequestsForTest, LoweredGraphPlan, PreparedGraph,
        SpatialFilterPreparableGraph,
    },
    renderer::ResourceCacheBudget,
    resource::{
        ManagerIdentity, ResourceAccountingFault, ResourceManager,
        ResourceManagerObservationForTest, WorkingFormat,
    },
    shader::{
        ColorFilterOperationBufferLimits, DevicePassCache, DevicePassCacheCountsForTest,
        ProvisionalDevicePassCacheUpdate,
    },
    surface::{HeadlessPublication, HeadlessResources, RendererIdentity, SurfaceBackend},
    vello_engine::{
        ActiveVelloEncodingScope, EncodedVelloPass, PreparedVelloPass, TransactionEncodingState,
        TransactionTargetIntent, VelloAtlasOutcome, VelloEngineState, scene::VelloScene,
    },
};

/// Test-owned exact graph ingress for real production and fixture preparation stages.
#[must_use = "an exact test surface graph must enter its GPU transaction"]
pub(crate) enum ExactSurfaceGraph {
    Base(BasePreparableGraph),
    Composition(CompositionPreparableGraph),
    ColorFilter(ColorFilterPreparableGraph),
    SpatialFilter(SpatialFilterPreparableGraph),
    Backdrop(BackdropPreparableGraph),
}

enum ExactSurfaceGraphStage {
    Production(execute::ExactSurfaceGraph),
    TestFilter(ExactSurfaceGraph),
}

impl ExactSurfaceGraph {
    const fn working_format(&self) -> WorkingFormat {
        match self {
            Self::Base(preparable) => preparable.working_format(),
            Self::Composition(preparable) => preparable.working_format(),
            Self::ColorFilter(preparable) => preparable.working_format(),
            Self::SpatialFilter(preparable) => preparable.working_format(),
            Self::Backdrop(preparable) => preparable.working_format(),
        }
    }

    const fn output_format(&self) -> Format {
        match self {
            Self::Base(preparable) => preparable.output_format(),
            Self::Composition(preparable) => preparable.output_format(),
            Self::ColorFilter(preparable) => preparable.output_format(),
            Self::SpatialFilter(preparable) => preparable.output_format(),
            Self::Backdrop(preparable) => preparable.output_format(),
        }
    }

    fn known_output_extent(&self) -> Result<Option<PhysicalSize>> {
        match self {
            Self::Base(preparable) => preparable.output_extent().map(Some),
            Self::Composition(_) | Self::ColorFilter(_) | Self::SpatialFilter(_) => Ok(None),
            Self::Backdrop(preparable) => preparable.output_extent().map(Some),
        }
    }

    fn into_stage(self) -> ExactSurfaceGraphStage {
        match self {
            Self::Base(preparable) => {
                ExactSurfaceGraphStage::Production(execute::ExactSurfaceGraph::Base(preparable))
            }
            Self::Composition(preparable) => ExactSurfaceGraphStage::Production(
                execute::ExactSurfaceGraph::Composition(preparable),
            ),
            Self::Backdrop(preparable) => {
                ExactSurfaceGraphStage::Production(execute::ExactSurfaceGraph::Backdrop(preparable))
            }
            fixture @ (Self::ColorFilter(_) | Self::SpatialFilter(_)) => {
                ExactSurfaceGraphStage::TestFilter(fixture)
            }
        }
    }
}

impl Backend {
    fn prepare_graph_resources_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
    ) -> Result<PreparedGraph<'_>> {
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        prepared.apply_color_filter_shader_failure_for_test();
        Ok(prepared)
    }

    fn prepare_test_filter_surface_graph_resources(
        &mut self,
        identity: DeviceSlotIdentity,
        graph: ExactSurfaceGraph,
    ) -> Result<PreparedGraph<'_>> {
        let selected_working_format = graph.working_format();
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot is unavailable for exact test graph preparation",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before exact test graph preparation",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(RuntimeOperation::SurfaceRendering));
        }
        if !state.signal.has_active_operation() {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "exact test graph preparation requires one active GPU transaction",
            ));
        }
        let capabilities = state
            .capabilities
            .for_selected_working_format(selected_working_format)?;
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "ready GPU device resources disappeared before exact test graph preparation",
            )
        })?;
        let prepared = match graph {
            ExactSurfaceGraph::ColorFilter(preparable) => PreparedGraph::try_prepare_color_filter(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            ExactSurfaceGraph::SpatialFilter(preparable) => {
                PreparedGraph::try_prepare_spatial_filter(
                    preparable,
                    &capabilities,
                    &ready.device,
                    &ready.queue,
                    &ready.resources,
                    (&ready.pass_cache, true),
                )
            }
            ExactSurfaceGraph::Base(_)
            | ExactSurfaceGraph::Composition(_)
            | ExactSurfaceGraph::Backdrop(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "a production exact graph entered the test filter preparation stage",
                ));
            }
        }?
        .with_vello_engine(&ready.engine);
        let mut prepared = prepared;
        prepared.apply_color_filter_shader_failure_for_test();
        Ok(prepared)
    }
}

pub(crate) async fn render_exact_headless_graph_surface(
    backend: &mut Backend,
    surface: &Surface,
    graph: ExactSurfaceGraph,
) -> Result<execute::SurfaceFrameCommit> {
    let graph = match graph.into_stage() {
        ExactSurfaceGraphStage::Production(production) => {
            return execute::render_exact_headless_graph_surface(backend, surface, production)
                .await;
        }
        ExactSurfaceGraphStage::TestFilter(graph) => graph,
    };
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
                "the exact test graph executor requires a device-backed headless surface",
            ));
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        SurfaceBackend::Presented { .. } => {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "presented exact test graph execution requires the presented executor",
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
            "the headless draft differs from the exact eligible test graph output",
        ));
    }
    render_exact_headless_filter_graph_surface(
        backend,
        surface,
        graph,
        device_identity,
        physical_size,
        selected_working_format,
    )
    .await
}

async fn render_exact_headless_filter_graph_surface(
    backend: &mut Backend,
    surface: &Surface,
    graph: ExactSurfaceGraph,
    device_identity: DeviceSlotIdentity,
    physical_size: PhysicalSize,
    selected_working_format: WorkingFormat,
) -> Result<execute::SurfaceFrameCommit> {
    let capabilities = backend
        .device_capabilities(device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the exact test graph executor lost immutable device capabilities",
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
            "the exact test graph executor lost its ready device before draft allocation",
        )?;
        (ready.device.clone(), ready.queue.clone())
    };
    let render_start = Instant::now();
    let (draft_texture, draft_view) =
        create_headless_texture(&device, physical_size, surface.options.format)?;
    let mut prepared =
        backend.prepare_test_filter_surface_graph_resources(device_identity, graph)?;
    if prepared.output_extent()? != physical_size
        || prepared.output_format() != surface.options.format
        || prepared.working_format() != selected_working_format
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "prepared exact test graph output changed after eligibility validation",
        ));
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist exact headless test graph encoder"),
    });
    let pending_encoding = prepared
        .encode_custom_spine(
            &mut encoder,
            GraphExternalOutputView::try_new(&draft_view, surface.options.format, physical_size)?,
        )
        .await
        .map_err(crate::pass::normalize_color_filter_shader_failure_for_test)?;
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
            "the exact test graph executor lost its ready device before submission",
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
                "the headless exact test graph transaction returned a presented host effect",
            ));
        }
    };
    Ok(execute::SurfaceFrameCommit::headless_graph(
        publication,
        frame_cleanup,
        graph_activity,
        selected_working_format,
        execute::RenderTimings {
            render_time: render_start.elapsed(),
            present_time: Duration::ZERO,
        },
    ))
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) async fn render_exact_presented_graph_surface(
    backend: &mut Backend,
    surface: &mut Surface,
    graph: ExactSurfaceGraph,
) -> Result<execute::SurfaceFrameCommit> {
    let graph = match graph.into_stage() {
        ExactSurfaceGraphStage::Production(production) => {
            return execute::render_exact_presented_graph_surface(backend, surface, production)
                .await;
        }
        ExactSurfaceGraphStage::TestFilter(graph) => graph,
    };
    let selected_working_format = graph.working_format();
    let (device_identity, physical_size, output_format, selected_working_format) =
        present::exact_presented_graph_target(
            surface,
            selected_working_format,
            graph.output_format(),
            graph.known_output_extent()?,
        )?;
    let capabilities = backend
        .device_capabilities(device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the presented exact test graph executor lost immutable device capabilities",
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
            "the presented exact test graph lost its ready device before preparation",
        )?;
        (ready.device.clone(), ready.queue.clone())
    };
    let render_start = Instant::now();
    let prepared = backend.prepare_test_filter_surface_graph_resources(device_identity, graph)?;
    if prepared.output_extent()? != physical_size
        || prepared.output_format() != output_format
        || prepared.working_format() != selected_working_format
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "prepared presented exact test graph output changed after eligibility validation",
        ));
    }
    let present_start = Instant::now();
    let acquired =
        present::acquire_exact_presented_graph_texture(surface, &device, prepared, transaction)
            .await?;
    let (acquired, mut prepared, transaction) = acquired;
    let output_view = acquired.create_view();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist exact presented test graph encoder"),
    });
    let pending_encoding = prepared
        .encode_custom_spine(
            &mut encoder,
            GraphExternalOutputView::try_new(&output_view, output_format, physical_size)?,
        )
        .await
        .map_err(crate::pass::normalize_color_filter_shader_failure_for_test)?;
    let prepared_submission = prepared.finish_graph_submission(pending_encoding)?;
    drop(output_view);
    let payload =
        GraphSubmissionPayload::presented(encoder.finish(), prepared_submission, acquired);
    let clean = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the presented exact test graph lost its ready device before submission",
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
            "the presented exact test graph transaction returned a headless publication",
        ));
    }
    Ok(execute::SurfaceFrameCommit::presented_graph(
        frame_cleanup,
        graph_activity,
        selected_working_format,
        execute::RenderTimings {
            render_time: present_start.duration_since(render_start),
            present_time: present_start.elapsed(),
        },
    ))
}
use std::{
    sync::Arc,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

#[cfg(feature = "render-window")]
use crate::{
    Renderer,
    gpu_transaction::GpuOperationTransaction,
    surface::{
        DisplayFreePresentedSurfaceObservationForTest,
        DisplayFreePresentedSurfaceObservationHandleForTest, PresentedAcquireOutcomeForTest,
        PresentedLifecycle, PresentedSurface,
    },
};

#[cfg(feature = "render-window")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DisplayFreePresentedDeviceCompatibilityForTest {
    identity: DeviceSlotIdentity,
    compatible: bool,
}

#[cfg(feature = "render-window")]
impl DisplayFreePresentedDeviceCompatibilityForTest {
    pub(crate) const fn compatible(identity: DeviceSlotIdentity) -> Self {
        Self {
            identity,
            compatible: true,
        }
    }

    pub(crate) const fn incompatible(identity: DeviceSlotIdentity) -> Self {
        Self {
            identity,
            compatible: false,
        }
    }
}

/// Runs the display-free fixture's explicit compatibility stage over real device
/// terminal signals. The selected identity is then supplied to the ordinary
/// presented recreation path; no production selection callback is involved.
#[cfg(feature = "render-window")]
pub(crate) fn select_display_free_presented_device_for_test(
    renderer: &mut Renderer,
    preferred: DeviceSlotIdentity,
    candidates: &[DisplayFreePresentedDeviceCompatibilityForTest],
) -> Option<DeviceSlotIdentity> {
    let is_ready_and_compatible =
        |renderer: &mut Renderer, candidate: DisplayFreePresentedDeviceCompatibilityForTest| {
            candidate.compatible
                && renderer
                    .device_signal_for_test(candidate.identity)
                    .is_some_and(|signal| signal.first_terminal().is_none())
        };
    if let Some(candidate) = candidates
        .iter()
        .copied()
        .find(|candidate| candidate.identity == preferred)
        && is_ready_and_compatible(renderer, candidate)
    {
        return Some(candidate.identity);
    }
    candidates
        .iter()
        .copied()
        .find(|candidate| is_ready_and_compatible(renderer, *candidate))
        .map(|candidate| candidate.identity)
}

/// Executes the real Configure draft and transaction scope resolution with an
/// explicit test-owned invalid WGPU operation. The draft is never returned for
/// publication, so callers can assert failure atomicity at the owning boundary.
#[cfg(feature = "render-window")]
async fn configure_presented_surface_validation_failure_for_test(
    device: &wgpu::Device,
    signal: Arc<DeviceSignal>,
    surface: &PresentedSurface,
    physical_size: PhysicalSize,
    present_mode: wgpu::PresentMode,
    operation: RuntimeOperation,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let transaction =
        GpuOperationTransaction::begin(device, signal, generation, GpuOperationStage::Configure);
    let draft = surface.configure_draft(device, physical_size, present_mode);
    let _invalid_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surgeist explicit Configure validation failure stage"),
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
    let result = transaction.finish(operation).await;
    drop(draft);
    result
}

/// Executes and then explicitly discards a real Configure transaction and its
/// draft before publication.
#[cfg(feature = "render-window")]
fn discard_presented_configuration_draft_for_test(
    device: &wgpu::Device,
    signal: Arc<DeviceSignal>,
    surface: &PresentedSurface,
    physical_size: PhysicalSize,
    present_mode: wgpu::PresentMode,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let transaction =
        GpuOperationTransaction::begin(device, signal, generation, GpuOperationStage::Configure);
    let draft = surface.configure_draft(device, physical_size, present_mode);
    drop(draft);
    drop(transaction);
    Ok(())
}

#[cfg(feature = "render-window")]
pub(crate) async fn presented_configuration_validation_failure_stage_for_test(
    renderer: &mut Renderer,
    surface: &Surface,
    operation: RuntimeOperation,
) -> Result<()> {
    let identity = presented_device_identity_for_test(surface);
    let signal = renderer.device_signal_for_test(identity).ok_or_else(|| {
        Error::new(
            BackendErrorCode::SurfaceConfigureFailed,
            "the explicit Configure failure stage requires a current device signal",
        )
    })?;
    let (native, physical_size) = match &surface.backend {
        SurfaceBackend::Presented { surface, state, .. } => {
            (surface.as_ref(), state.requested_physical_size())
        }
        _ => {
            return Err(Error::new(
                BackendErrorCode::SurfaceConfigureFailed,
                "the explicit Configure failure stage requires a presented surface",
            ));
        }
    };
    let present_mode: wgpu::PresentMode = surface.options.present_mode.into();
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::SurfaceConfigureFailed,
                "the explicit Configure failure stage requires ready device resources",
            )
        })?;
    configure_presented_surface_validation_failure_for_test(
        ready.device_for_test(),
        signal,
        native,
        physical_size,
        present_mode,
        operation,
    )
    .await
}

#[cfg(feature = "render-window")]
pub(crate) fn discard_presented_configuration_stage_for_test(
    renderer: &mut Renderer,
    surface: &Surface,
) -> Result<()> {
    let identity = presented_device_identity_for_test(surface);
    let signal = renderer.device_signal_for_test(identity).ok_or_else(|| {
        Error::new(
            BackendErrorCode::SurfaceConfigureFailed,
            "the explicit Configure discard stage requires a current device signal",
        )
    })?;
    let (native, physical_size) = match &surface.backend {
        SurfaceBackend::Presented { surface, state, .. } => {
            (surface.as_ref(), state.requested_physical_size())
        }
        _ => {
            return Err(Error::new(
                BackendErrorCode::SurfaceConfigureFailed,
                "the explicit Configure discard stage requires a presented surface",
            ));
        }
    };
    let present_mode = surface.options.present_mode.into();
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::SurfaceConfigureFailed,
                "the explicit Configure discard stage requires ready device resources",
            )
        })?;
    discard_presented_configuration_draft_for_test(
        ready.device_for_test(),
        signal,
        native,
        physical_size,
        present_mode,
    )
}

#[cfg(feature = "render-window")]
pub(crate) fn display_free_presented_surface_for_test(
    renderer: &mut Renderer,
    options: SurfaceOptions,
) -> Surface {
    renderer
        .display_free_presented_surface_for_test(options)
        .expect("the display-free fixture must establish a real presented surface backend")
}

#[cfg(feature = "render-window")]
pub(crate) fn configured_display_free_presented_surface_for_test(
    renderer: &mut Renderer,
) -> Surface {
    let mut surface = display_free_presented_surface_for_test(
        renderer,
        SurfaceOptions {
            size: crate::Size::new(2.0, 2.0),
            ..SurfaceOptions::default()
        },
    );
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect("the display-free surface must configure through the real Configure transaction");
    surface
}

#[cfg(feature = "render-window")]
pub(crate) fn display_free_presented_surface_on_device_for_test(
    renderer: &mut Renderer,
    options: SurfaceOptions,
    device_identity: DeviceSlotIdentity,
    attachment: Attachment,
) -> Surface {
    renderer
        .display_free_presented_surface_on_device_for_test(options, device_identity, attachment)
        .expect("the display-free fixture must establish a real presented surface backend")
}

#[cfg(feature = "render-window")]
pub(crate) fn configured_display_free_presented_surface_on_device_for_test(
    renderer: &mut Renderer,
    device_identity: DeviceSlotIdentity,
    attachment: Attachment,
) -> Surface {
    let mut surface = display_free_presented_surface_on_device_for_test(
        renderer,
        SurfaceOptions {
            size: crate::Size::new(2.0, 2.0),
            ..SurfaceOptions::default()
        },
        device_identity,
        attachment,
    );
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect("the display-free surface must configure through the real Configure transaction");
    surface
}

#[cfg(feature = "render-window")]
pub(crate) fn set_presented_acquire_outcome_for_test(
    surface: &mut Surface,
    outcome: PresentedAcquireOutcomeForTest,
) {
    match &mut surface.backend {
        SurfaceBackend::Presented { surface, .. } => {
            surface.set_acquire_outcome_for_test(outcome);
        }
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn take_last_presented_texture_for_test(surface: &mut Surface) -> Option<wgpu::Texture> {
    match &mut surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.take_last_presented_texture_for_test(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_observation_for_test(
    surface: &Surface,
) -> DisplayFreePresentedSurfaceObservationForTest {
    match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.observation_for_test(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_observation_handle_for_test(
    surface: &Surface,
) -> DisplayFreePresentedSurfaceObservationHandleForTest {
    match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.observation_handle_for_test(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_lifecycle_for_test(surface: &Surface) -> PresentedLifecycle {
    match &surface.backend {
        SurfaceBackend::Presented { state, .. } => state.lifecycle(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_resource_id_for_test(surface: &Surface) -> Option<u64> {
    match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface
            .committed()
            .map(|resources| resources.resource_id_for_test()),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_configuration_count_for_test(surface: &Surface) -> usize {
    match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.configuration_count_for_test(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_target_identity_for_test(surface: &Surface) -> u64 {
    match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.target_identity_for_test(),
        _ => panic!("the fixture must retain a presented surface backend"),
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn presented_device_identity_for_test(surface: &Surface) -> DeviceSlotIdentity {
    surface
        .device_identity()
        .expect("the display-free fixture must retain a device slot identity")
}

pub(crate) struct OffscreenRenderGpuContext<'a> {
    backend: &'a mut Backend,
    device_identity: DeviceSlotIdentity,
}

impl<'a> OffscreenRenderGpuContext<'a> {
    #[must_use]
    pub(crate) fn new(backend: &'a mut Backend, device_identity: DeviceSlotIdentity) -> Self {
        Self {
            backend,
            device_identity,
        }
    }
}

/// Test-owned request facts for a Vello scene already encoded in
/// offscreen-local coordinates. Bounds size allocates the real target texture;
/// it is not a scene crop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OffscreenLocalSceneRenderRequest {
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
    parameters: Parameters,
}

impl OffscreenLocalSceneRenderRequest {
    #[must_use]
    pub(crate) const fn new(
        bounds: OffscreenBounds,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Self {
        Self {
            bounds,
            scale,
            format,
            parameters,
        }
    }
}

impl OffscreenRenderTarget {
    #[must_use]
    pub(crate) const fn resource_id(self) -> u64 {
        self.resource_identity.get()
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> OffscreenBounds {
        self.bounds
    }
}

impl OffscreenRenderedTextureLease {
    #[must_use]
    pub(crate) const fn target(&self) -> OffscreenRenderTarget {
        self.target
    }

    #[must_use]
    pub(crate) const fn timings(&self) -> super::RenderTimings {
        self.timings
    }

    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        self.frame_scope
            .as_ref()
            .expect("an unresolved offscreen lease must own its resource frame")
            .poison_retained_byte_accounting_for_test()
    }
}

pub(crate) async fn render_internal_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    scene: &VelloScene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let context = context.map(|context| (context.backend, context.device_identity));
    offscreen::render_internal_vello_local_scene_to_offscreen_texture(
        context,
        options,
        scene,
        request.bounds,
        request.scale,
        request.format,
        request.parameters,
    )
    .await
}

pub(crate) struct ReadyDeviceStateBorrowForTest<'ready> {
    adapter: &'ready wgpu::Adapter,
    device: &'ready wgpu::Device,
    queue: &'ready wgpu::Queue,
    engine: &'ready VelloEngineState,
    resources: &'ready ResourceManager,
    pass_cache: &'ready DevicePassCache,
}

#[derive(Debug)]
pub(crate) struct DeviceTerminalWaitObservationForTest {
    pub(crate) final_terminal: Option<Arc<DeviceTerminalSignal>>,
    pub(crate) active_operation_generation: Option<u64>,
    pub(crate) requested_timeout: Duration,
    pub(crate) elapsed: Duration,
}

impl DeviceTerminalWaitObservationForTest {
    pub(crate) const fn observed_terminal_for_test(&self) -> bool {
        self.final_terminal.is_some()
    }
}

impl ReadyDeviceStateBorrowForTest<'_> {
    pub(crate) fn sole_resource_manager_identity_for_test(&self) -> Option<ManagerIdentity> {
        Some(self.resources.identity_for_test())
    }

    pub(crate) fn adapter_for_test(&self) -> &wgpu::Adapter {
        self.adapter
    }

    pub(crate) fn device_for_test(&self) -> &wgpu::Device {
        self.device
    }

    pub(crate) fn queue_for_test(&self) -> &wgpu::Queue {
        self.queue
    }

    pub(crate) fn checked_pipeline_for_test(&self) -> &wgpu::ComputePipeline {
        self.engine.checked_pipeline_for_test()
    }

    pub(crate) fn internal_resources_empty_for_test(&self) -> bool {
        self.resources.is_empty_for_test()
    }

    pub(crate) fn internal_resource_manager_observation_for_test(
        &self,
    ) -> ResourceManagerObservationForTest {
        self.resources.observation_for_test()
    }

    pub(crate) fn resource_cache_budget_for_test(&self) -> ResourceCacheBudget {
        self.resources.budget_for_test()
    }

    pub(crate) fn device_pass_cache_counts_for_test(&self) -> DevicePassCacheCountsForTest {
        self.pass_cache.counts_for_test()
    }
}

impl ReadyDeviceState {
    fn seed_pass_cache_sampler_for_test(&mut self) -> DevicePassCacheCountsForTest {
        self.pass_cache.seed_sampler_for_test(&self.device)
    }

    fn borrow_for_test(&self) -> ReadyDeviceStateBorrowForTest<'_> {
        ReadyDeviceStateBorrowForTest {
            adapter: &self.adapter,
            device: &self.device,
            queue: &self.queue,
            engine: &self.engine,
            resources: &self.resources,
            pass_cache: &self.pass_cache,
        }
    }
}

impl DeviceCapabilities {
    pub(crate) fn from_test_facts(
        high_precision: bool,
        reduced_precision: bool,
        max_effect_texture_dimension_2d: u32,
    ) -> Self {
        let complete_features = |supported| wgpu::TextureFormatFeatures {
            allowed_usages: if supported {
                WorkingFormat::HighPrecision.required_usages()
            } else {
                wgpu::TextureUsages::empty()
            },
            flags: if supported {
                wgpu::TextureFormatFeatureFlags::FILTERABLE
            } else {
                wgpu::TextureFormatFeatureFlags::empty()
            },
        };
        Self {
            high_precision,
            reduced_precision,
            high_precision_features: complete_features(high_precision),
            reduced_precision_features: complete_features(reduced_precision),
            max_effect_texture_dimension_2d,
        }
    }
}

impl DeviceTerminalSignal {
    pub(crate) const fn operation_generation_for_test(&self) -> Option<u64> {
        match self {
            Self::Lost { .. } => None,
            Self::Faulted {
                operation_generation,
                ..
            } => *operation_generation,
        }
    }
}

impl DeviceSignal {
    pub(crate) fn new_for_test() -> Arc<Self> {
        Arc::new(Self::new())
    }

    pub(crate) fn next_test_generation(&self) -> Result<u64> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state
            .active_operation_generation
            .map_or(Ok(1), |generation| {
                generation.checked_add(1).ok_or_else(|| {
                    Error::invalid_value(
                        "GPU operation generation",
                        generation,
                        "must have remaining generation space",
                    )
                })
            })
    }

    pub(crate) fn active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active_operation_generation
    }

    pub(crate) fn record_uncaptured_fault_for_test(&self, kind: GpuFaultKind, message: &str) {
        self.record_fault(kind, message.into());
    }

    pub(crate) fn record_loss_for_test(&self, reason: DeviceLossReason) {
        self.record(DeviceTerminalSignal::lost(
            reason,
            "test device loss".into(),
        ));
    }

    pub(crate) fn finish_active_generation_for_test(
        &self,
        generation: u64,
    ) -> Option<Arc<DeviceTerminalSignal>> {
        self.finish_active_generation(generation)
    }

    pub(crate) fn wait_for_terminal_for_test(
        &self,
        timeout: Duration,
    ) -> DeviceTerminalWaitObservationForTest {
        let started = Instant::now();
        let deadline = started + timeout;
        loop {
            let current = self.terminal_wait_observation_for_test(timeout, started);
            if current.observed_terminal_for_test() {
                return current;
            }
            if Instant::now() >= deadline {
                return self.terminal_wait_observation_for_test(timeout, started);
            }
            std::thread::yield_now();
        }
    }

    pub(crate) fn terminal_wait_observation_for_test(
        &self,
        requested_timeout: Duration,
        started: Instant,
    ) -> DeviceTerminalWaitObservationForTest {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        DeviceTerminalWaitObservationForTest {
            final_terminal: state.first_terminal.clone(),
            active_operation_generation: state.active_operation_generation,
            requested_timeout,
            elapsed: started.elapsed(),
        }
    }
}

#[cfg(feature = "render-window")]
pub(crate) fn require_presented_device_identity_for_test(
    identity: Option<DeviceSlotIdentity>,
) -> Result<DeviceSlotIdentity> {
    super::present::require_presented_device_identity(identity)
}

impl DeviceSlotIdentity {
    pub(crate) fn mark_stale_for_test(&mut self) {
        self.generation = self.generation.checked_add(1).unwrap();
    }
}

impl Backend {
    #[cfg(feature = "render-window")]
    pub(crate) async fn create_display_free_presented_surface_for_test(
        &mut self,
        preferred: Option<DeviceSlotIdentity>,
        operation: RuntimeOperation,
        format: Format,
    ) -> Result<(PresentedSurface, DeviceSlotIdentity)> {
        let identity = if let Some(identity) = self.compatible_ready_device(preferred, |_| true) {
            Some(identity)
        } else {
            self.new_device(None).await?
        };
        let identity = super::present::require_presented_device_identity(identity)?;
        self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::SurfaceCreateFailed,
            "the selected presentation device is unavailable",
        )?;
        Ok((PresentedSurface::display_free_for_test(format), identity))
    }

    pub(crate) fn device_queue(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
    ) -> Result<(&wgpu::Device, &wgpu::Queue)> {
        let ready = self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::RenderFailed,
            "GPU device resources are unavailable",
        )?;
        Ok((&ready.device, &ready.queue))
    }

    pub(crate) fn override_device_effect_precision_facts_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        effect_precisions: EffectPrecisionCapabilities,
    ) -> bool {
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        if state.generation != identity.generation {
            return false;
        }
        state.observe_terminal();
        if state.terminal().is_some() {
            return false;
        }
        state.capabilities = DeviceCapabilities::from_test_facts(
            effect_precisions.supports_high_precision(),
            effect_precisions.supports_reduced_precision(),
            state.capabilities.max_effect_texture_dimension_2d,
        );
        true
    }

    pub(crate) fn signal_loss_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        reason: DeviceLossReason,
    ) {
        if let Some(state) = self.device_states.get(identity.slot())
            && state.generation == identity.generation
        {
            state.signal.record_loss_for_test(reason);
        }
    }

    pub(crate) fn signal_uncaptured_fault_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        kind: GpuFaultKind,
    ) {
        if let Some(state) = self.device_states.get(identity.slot())
            && state.generation == identity.generation
        {
            state
                .signal
                .record_uncaptured_fault_for_test(kind, "test uncaptured GPU fault");
        }
    }

    pub(crate) fn device_signal_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<Arc<DeviceSignal>> {
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .map(|state| Arc::clone(&state.signal))
    }

    pub(crate) fn wait_for_terminal_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        timeout: Duration,
    ) -> bool {
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .is_some_and(|state| {
                let observation = state.signal.wait_for_terminal_for_test(timeout);
                let observed_terminal = observation.observed_terminal_for_test();
                if !observed_terminal {
                    eprintln!("device terminal wait timed out: {observation:?}");
                }
                observed_terminal
            })
    }

    pub(crate) fn renderer_released_for_test(&mut self, identity: DeviceSlotIdentity) -> bool {
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        state.observe_terminal();
        state.ready().is_none()
    }

    pub(crate) fn ready_device_state_borrow_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state.observe_terminal();
        state.ready().map(ReadyDeviceState::borrow_for_test)
    }

    pub(crate) fn seed_device_pass_cache_sampler_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<DevicePassCacheCountsForTest> {
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state.observe_terminal();
        state
            .ready_mut()
            .map(ReadyDeviceState::seed_pass_cache_sampler_for_test)
    }

    pub(crate) fn active_operation_generation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<u64> {
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .and_then(|state| state.signal.active_generation_for_test())
    }

    pub(crate) async fn add_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        self.new_device(None).await?.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the donor WGPU device could not be created",
            )
        })
    }

    pub(crate) fn destroy_device_for_test(&mut self, identity: DeviceSlotIdentity) -> bool {
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        if state.generation != identity.generation {
            return false;
        }
        let Some(ready) = state.ready() else {
            return false;
        };
        ready.device.destroy();
        let _ = ready.device.poll(wgpu::PollType::Poll);
        true
    }
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CorePassShaderCacheRealizationObservationForTest {
    pub(crate) realizes_all_checked_programs: bool,
    pub(crate) provisional_handles_are_encoding_ready: bool,
    pub(crate) commits_only_after_clean_transaction: bool,
    pub(crate) reuses_exact_committed_entries: bool,
    pub(crate) failed_validation_publishes_none: bool,
    pub(crate) cancellation_publishes_none: bool,
    pub(crate) device_transition_publishes_none: bool,
    pub(crate) specializes_rgba_and_bgra_outputs: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LayerCompositeCacheRealizationObservationForTest {
    pub(crate) realizes_normal_and_destination_programs: bool,
    pub(crate) realizes_all_optional_binding_combinations: bool,
    pub(crate) normal_uses_fixed_premultiplied_source_over: bool,
    pub(crate) destination_uses_replace_blending: bool,
    pub(crate) commits_only_after_clean_transaction: bool,
    pub(crate) reuses_exact_committed_entries: bool,
    pub(crate) failed_validation_publishes_none: bool,
    pub(crate) cancellation_publishes_none: bool,
    pub(crate) device_transition_publishes_none: bool,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompositionMaskSamplingVectorForTest {
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
    pub(crate) layer_point: Point,
    pub(crate) clip_alpha: Option<f32>,
    pub(crate) opacity: f32,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompositionMaskSamplingInputForTest {
    pub(crate) mask_size: PhysicalSize,
    pub(crate) mask_rgba: Vec<u8>,
    pub(crate) mask_bounds: Rect,
    pub(crate) source: [f32; 4],
    pub(crate) vectors: Vec<CompositionMaskSamplingVectorForTest>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompositionBlendVectorForTest {
    pub(crate) blend: BlendMode,
    pub(crate) source: [f32; 4],
    pub(crate) parent: [f32; 4],
    pub(crate) opacity: f32,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompositionGpuVectorResultsForTest {
    pub(crate) working_format: WorkingFormat,
    pub(crate) rgba: Vec<[f32; 4]>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CustomSpineEncodingObservationForTest {
    pub(crate) encodes_custom_passes_in_order: bool,
    pub(crate) clears_full_root_once: bool,
    pub(crate) uses_exact_prepared_spatial_mapping: bool,
    pub(crate) presents_to_exact_external_output: bool,
    pub(crate) exposes_bounded_capture_handoff: bool,
    pub(crate) validates_checked_capture_completion: bool,
    pub(crate) completes_custom_passes_after_encoding: bool,
    pub(crate) parent_and_result_are_distinct: bool,
    pub(crate) copies_full_parent_before_bounded_source_render: bool,
    pub(crate) samples_only_source_with_fixed_premultiplied_blend: bool,
    pub(crate) preserves_signed_source_origin: bool,
    pub(crate) keeps_cache_update_provisional: bool,
    pub(crate) encodes_without_submission_or_sync: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompositionOrderedGraphEncodingObservationForTest {
    pub(crate) encodes_clip_mask_opacity_and_blend_in_authored_order: bool,
    pub(crate) normal_uses_fixed_premultiplied_blend: bool,
    pub(crate) normal_omits_parent_sample: bool,
    pub(crate) destination_copies_full_parent: bool,
    pub(crate) destination_avoids_read_write_alias: bool,
    pub(crate) composite_count: usize,
    pub(crate) one_graph_command_encoder: bool,
    pub(crate) transaction_committed: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OrderedColorFilterGraphEncodingObservationForTest {
    pub(crate) fused_runs_preserve_authored_order: bool,
    pub(crate) color_pass_count: usize,
    pub(crate) binds_exact_source_spatial_and_operations: bool,
    pub(crate) source_and_result_are_distinct: bool,
    pub(crate) uses_validated_viewport_and_scissor: bool,
    pub(crate) releases_every_resource_at_last_use: bool,
    pub(crate) one_graph_command_encoder: bool,
    pub(crate) transaction_committed: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ColorFilterOversizedBufferPreservationObservationForTest {
    pub(crate) returns_exact_limit_error: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SpatialFilterGraphEncodingObservationForTest {
    pub(crate) pass_order: Vec<crate::pass::SpatialFilterPassTagForTest>,
    pub(crate) blur_pass_count: usize,
    pub(crate) drop_shadow_colorize_count: usize,
    pub(crate) drop_shadow_merge_count: usize,
    pub(crate) each_pass_advances_once: bool,
    pub(crate) binds_exact_prepared_resources: bool,
    pub(crate) uses_signed_viewport_and_scissor: bool,
    pub(crate) blur_sources_intermediates_and_results_are_distinct: bool,
    pub(crate) kernels_release_at_validated_last_use: bool,
    pub(crate) textures_release_at_validated_last_use: bool,
    pub(crate) drop_shadow_reads_original_source_twice: bool,
    pub(crate) original_source_releases_after_merge: bool,
    pub(crate) one_graph_command_encoder: bool,
    pub(crate) transaction_committed: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SpatialFilterFailurePreservationObservationForTest {
    pub(crate) encode_failure_is_reported: bool,
    pub(crate) scope_failure_is_reported: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackdropGraphEncodingObservationForTest {
    pub(crate) encodes_copy_filter_clip_foreground_and_group_in_order: bool,
    pub(crate) parent_is_copied_once: bool,
    pub(crate) copy_filter_foreground_and_group_are_distinct: bool,
    pub(crate) later_sibling_reads_completed_group: bool,
    pub(crate) releases_at_validated_last_use: bool,
    pub(crate) one_graph_command_encoder: bool,
    pub(crate) transaction_committed: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BackdropFailurePreservationObservationForTest {
    pub(crate) encode_failure_is_reported: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum SpatialFilterInjectedFailureForTest {
    Encode,
    Scope,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VelloCaptureFailureObservationForTest {
    pub(crate) capture_failure_is_reported: bool,
    pub(crate) complete_pass_is_rejected: bool,
    pub(crate) retry_on_new_encoder_is_rejected: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MultipleVelloCaptureEncodingObservationForTest {
    pub(crate) exact_capture_count: bool,
    pub(crate) one_graph_command_encoder: bool,
    pub(crate) one_gpu_transaction: bool,
    pub(crate) one_active_vello_scope: bool,
    pub(crate) aggregate_pending_commit: bool,
    pub(crate) commits_every_capture_after_transaction_success: bool,
    pub(crate) aborts_every_capture_on_drop: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TwoCaptureFailureForTest {
    LaterCaptureEncoding,
    SharedScopeResolution,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TwoCaptureFailureObservationForTest {
    pub(crate) acquired_capture_lease_count: usize,
    pub(crate) failure_is_reported: bool,
    pub(crate) produces_no_pending_commit: bool,
    pub(crate) retry_is_rejected: bool,
    pub(crate) resource_creation_was_observed: bool,
    pub(crate) remaining_leased_resource_count: usize,
    pub(crate) remaining_resource_count: usize,
    pub(crate) atlas_recovery_outcome: Option<VelloAtlasOutcome>,
    pub(crate) transaction_lease_is_released: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VelloCaptureRasterContractObservationForTest {
    pub(crate) lowers_with_exact_initial_transform: bool,
    pub(crate) uses_transparent_base: bool,
    pub(crate) uses_requested_antialiasing: bool,
    pub(crate) uses_exact_positive_extent: bool,
    pub(crate) uses_exact_rgba8_target_and_view: bool,
    pub(crate) uses_exact_capture_usage: bool,
    pub(crate) has_unforgeable_encoded_capture_proof: bool,
}

#[cfg(test)]
fn provision_core_pass_requests_for_test(
    ready: &ReadyDeviceState,
    requests: &CorePassCacheRequestsForTest,
    invalidate_last_pipeline: bool,
) -> Result<(ProvisionalDevicePassCacheUpdate, bool)> {
    let mut update = ready.pass_cache.provisional_update();
    let last = requests.passes().len().saturating_sub(1);
    let mut encoding_ready = !requests.passes().is_empty();
    for (index, keys) in requests.passes().iter().enumerate() {
        let objects = if invalidate_last_pipeline && index == last {
            update.realize_core_pass_with_invalid_fragment_for_test(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        } else {
            update.realize_core_pass(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        };
        drop(objects);
        encoding_ready &= update.contains_core_pass_for_test(
            &ready.pass_cache,
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        );
    }
    Ok((update, encoding_ready))
}

#[cfg(test)]
fn core_pass_requests_are_cached_for_test(
    cache: &DevicePassCache,
    requests: &CorePassCacheRequestsForTest,
) -> bool {
    !requests.passes().is_empty()
        && requests.passes().iter().all(|keys| {
            cache.contains_core_pass_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
        })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LayerCompositeProvisionObservationForTest {
    encoding_ready: bool,
    has_normal: bool,
    has_destination: bool,
    all_optional_combinations: bool,
    normal_uses_fixed_blend: bool,
    destination_uses_replace_blend: bool,
}

#[cfg(test)]
fn provision_layer_composite_requests_for_test(
    ready: &ReadyDeviceState,
    requests: &LayerCompositeCacheRequestsForTest,
    invalidate_last_pipeline: bool,
) -> Result<(
    ProvisionalDevicePassCacheUpdate,
    LayerCompositeProvisionObservationForTest,
)> {
    let mut update = ready.pass_cache.provisional_update();
    let last = requests.passes().len().saturating_sub(1);
    let mut encoding_ready = !requests.passes().is_empty();
    let mut has_normal = false;
    let mut has_destination = false;
    let mut normal_uses_fixed_blend = true;
    let mut destination_uses_replace_blend = true;
    let mut combinations = [[false; 4]; 2];
    for (index, keys) in requests.passes().iter().enumerate() {
        let objects = if invalidate_last_pipeline && index == last {
            update.realize_composite_pass_with_invalid_fragment_for_test(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        } else {
            update.realize_composite_pass(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        };
        encoding_ready &= objects.require_encoding_ready().is_ok();
        let path_index = match objects.path() {
            crate::shader::ShaderCompositePathKey::Normal => {
                has_normal = true;
                normal_uses_fixed_blend &= objects.uses_fixed_source_over_blend();
                0
            }
            crate::shader::ShaderCompositePathKey::DestinationSampling => {
                has_destination = true;
                destination_uses_replace_blend &= objects.uses_replace_blend();
                1
            }
        };
        let combination_index =
            usize::from(objects.has_clip_coverage()) + 2 * usize::from(objects.has_alpha_mask());
        combinations[path_index][combination_index] = true;
        encoding_ready &= update.contains_composite_pass_for_test(
            &ready.pass_cache,
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        );
    }
    Ok((
        update,
        LayerCompositeProvisionObservationForTest {
            encoding_ready,
            has_normal,
            has_destination,
            all_optional_combinations: combinations.into_iter().flatten().all(|present| present),
            normal_uses_fixed_blend,
            destination_uses_replace_blend,
        },
    ))
}

#[cfg(test)]
fn layer_composite_requests_are_cached_for_test(
    cache: &DevicePassCache,
    requests: &LayerCompositeCacheRequestsForTest,
) -> bool {
    !requests.passes().is_empty()
        && requests.passes().iter().all(|keys| {
            cache.contains_composite_pass_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
        })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CompositionGpuVectorDrawForTest {
    path: crate::shader::ShaderCompositePathKey,
    has_clip_coverage: bool,
    has_alpha_mask: bool,
    source: [f32; 4],
    parent: [f32; 4],
    layer_point: Point,
    clip_alpha: f32,
    opacity: f32,
    blend: BlendMode,
    quality: ImageQuality,
    extend: Extend,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct CompositionGpuMaskTextureForTest<'a> {
    size: PhysicalSize,
    rgba: &'a [u8],
    bounds: Rect,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) struct CompositionPreparedGpuVectorsForTest {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) working_format: WorkingFormat,
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) outputs: Vec<wgpu::Texture>,
    pub(crate) pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_vector_texture(
    device: &wgpu::Device,
    size: PhysicalSize,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &'static str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.width(),
            height: size.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_clear_vector_texture(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    color: [f32; 4],
    label: &'static str,
) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: f64::from(color[0]),
                    g: f64::from(color[1]),
                    b: f64::from(color[2]),
                    a: f64::from(color[3]),
                }),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_vector_uniform_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    label: &'static str,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::try_from(bytes.len()).unwrap(),
        usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_upload_vector_mask(
    ready: &ReadyDeviceState,
    mask: Option<&CompositionGpuMaskTextureForTest<'_>>,
) -> Result<Option<wgpu::Texture>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    let expected_len = usize::try_from(mask.size.width())
        .ok()
        .and_then(|width| {
            usize::try_from(mask.size.height())
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "composition GPU mask vector byte length overflowed",
            )
        })?;
    if mask.rgba.len() != expected_len || mask.size.width() == 0 || mask.size.height() == 0 {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "composition GPU mask vector bytes do not match a positive RGBA8 extent",
        ));
    }
    let texture = composition_vector_texture(
        &ready.device,
        mask.size,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST),
        "Surgeist composition GPU vector mask",
    );
    ready.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        mask.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(mask.size.width() * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: mask.size.width(),
            height: mask.size.height(),
            depth_or_array_layers: 1,
        },
    );
    Ok(Some(texture))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct CompositionVectorDrawTextures {
    source: wgpu::TextureView,
    parent: Option<wgpu::TextureView>,
    clip: Option<wgpu::TextureView>,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct CompositionVectorDrawEncodingContext<'a> {
    ready: &'a ReadyDeviceState,
    requests: &'a LayerCompositeCacheRequestsForTest,
    mask_view: Option<&'a wgpu::TextureView>,
    mask: Option<&'a CompositionGpuMaskTextureForTest<'a>>,
    spatial_bytes: &'a [u8],
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_prepare_vector_draw_textures(
    ready: &ReadyDeviceState,
    encoder: &mut wgpu::CommandEncoder,
    working_format: WorkingFormat,
    source_size: PhysicalSize,
    draw: CompositionGpuVectorDrawForTest,
) -> CompositionVectorDrawTextures {
    let source = composition_vector_texture(
        &ready.device,
        source_size,
        working_format.texture_format(),
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
        "Surgeist composition GPU vector source",
    );
    let source = source.create_view(&wgpu::TextureViewDescriptor::default());
    composition_clear_vector_texture(
        encoder,
        &source,
        draw.source,
        "Surgeist composition GPU vector source clear",
    );
    let parent =
        (draw.path == crate::shader::ShaderCompositePathKey::DestinationSampling).then(|| {
            let texture = composition_vector_texture(
                &ready.device,
                PhysicalSize::new(1, 1),
                working_format.texture_format(),
                wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
                "Surgeist composition GPU vector parent",
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            composition_clear_vector_texture(
                encoder,
                &view,
                draw.parent,
                "Surgeist composition GPU vector parent clear",
            );
            view
        });
    let clip = draw.has_clip_coverage.then(|| {
        let texture = composition_vector_texture(
            &ready.device,
            PhysicalSize::new(1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
            "Surgeist composition GPU vector clip coverage",
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        composition_clear_vector_texture(
            encoder,
            &view,
            [1.0, 0.25, 0.75, draw.clip_alpha],
            "Surgeist composition GPU vector clip clear",
        );
        view
    });
    let output = composition_vector_texture(
        &ready.device,
        PhysicalSize::new(1, 1),
        working_format.texture_format(),
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::COPY_SRC),
        "Surgeist composition GPU vector output",
    );
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let base = if draw.path == crate::shader::ShaderCompositePathKey::Normal {
        draw.parent
    } else {
        [0.125, 0.25, 0.375, 0.5]
    };
    composition_clear_vector_texture(
        encoder,
        &output_view,
        base,
        "Surgeist composition GPU vector output clear",
    );
    CompositionVectorDrawTextures {
        source,
        parent,
        clip,
        output,
        output_view,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_vector_parameter_bytes(
    mask: Option<&CompositionGpuMaskTextureForTest<'_>>,
    draw: CompositionGpuVectorDrawForTest,
) -> Result<[u8; 112]> {
    let mask_bounds = mask.map_or([0.0, 0.0, 1.0, 1.0], |mask| {
        [
            mask.bounds.x(),
            mask.bounds.y(),
            mask.bounds.width(),
            mask.bounds.height(),
        ]
    });
    let mask_dimensions = mask.map_or([1, 1], |mask| [mask.size.width(), mask.size.height()]);
    crate::shader::composite_parameter_bytes_for_gpu_vector_for_test(
        crate::shader::CompositeParameterGpuVectorFactsForTest {
            layer_point: [draw.layer_point.x(), draw.layer_point.y()],
            mask_bounds,
            mask_dimensions,
            quality: draw.quality,
            extend: draw.extend,
            opacity: draw.opacity,
            blend: draw.blend,
            has_clip: draw.has_clip_coverage,
            has_mask: draw.has_alpha_mask,
        },
    )
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn composition_encode_vector_draw(
    context: &CompositionVectorDrawEncodingContext<'_>,
    update: &mut ProvisionalDevicePassCacheUpdate,
    encoder: &mut wgpu::CommandEncoder,
    textures: &CompositionVectorDrawTextures,
    draw: CompositionGpuVectorDrawForTest,
) -> Result<()> {
    let keys = context
        .requests
        .composite_pass(draw.path, draw.has_clip_coverage, draw.has_alpha_mask)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "composition GPU vector draw has no exact composite pipeline keys",
            )
        })?;
    let spatial = composition_vector_uniform_buffer(
        &context.ready.device,
        &context.ready.queue,
        context.spatial_bytes,
        "Surgeist composition GPU vector spatial uniform",
    );
    let parameters = composition_vector_parameter_bytes(context.mask, draw)?;
    let parameters = composition_vector_uniform_buffer(
        &context.ready.device,
        &context.ready.queue,
        &parameters,
        "Surgeist composition GPU vector composite parameters",
    );
    let objects = update.realize_composite_pass(
        &context.ready.device,
        &context.ready.pass_cache,
        keys.samplers(),
        keys.layout(),
        keys.shader(),
        keys.pipeline(),
    )?;
    objects.require_encoding_ready()?;
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&textures.source),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(objects.source_sampler()),
        },
    ];
    for (binding, view) in [(2, textures.parent.as_ref()), (3, textures.clip.as_ref())] {
        if let Some(view) = view {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
    }
    if draw.has_alpha_mask {
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(context.mask_view.ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "composition GPU mask draw has no uploaded mask texture",
                )
            })?),
        });
    }
    entries.extend([
        wgpu::BindGroupEntry {
            binding: 5,
            resource: spatial.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: parameters.as_entire_binding(),
        },
    ]);
    let bindings = context
        .ready
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist composition GPU vector bindings"),
            layout: objects.bind_group_layout(),
            entries: &entries,
        });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Surgeist composition GPU vector composite"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &textures.output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    pass.set_pipeline(objects.render_pipeline());
    pass.set_bind_group(0, &bindings, &[]);
    pass.draw(0..3, 0..1);
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn encode_composition_gpu_vectors_for_test(
    ready: &ReadyDeviceState,
    requests: &LayerCompositeCacheRequestsForTest,
    working_format: WorkingFormat,
    mask: Option<CompositionGpuMaskTextureForTest<'_>>,
    draws: &[CompositionGpuVectorDrawForTest],
) -> Result<CompositionPreparedGpuVectorsForTest> {
    if draws.is_empty() {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "composition GPU vector execution requires at least one draw",
        ));
    }
    let mask_texture = composition_upload_vector_mask(ready, mask.as_ref())?;
    let mask_view = mask_texture
        .as_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
    let vector_source_origin = Point::new(-1.0, -1.0);
    let vector_source_size = PhysicalSize::new(7, 4);
    let spatial_bytes = crate::pass::pass_spatial_uniform_bytes_for_test(
        vector_source_origin,
        1.0,
        vector_source_size,
        Point::new(0.0, 0.0),
        1.0,
        PhysicalSize::new(1, 1),
    )?;
    let mut encoder = ready
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist composition GPU vector encoder"),
        });
    let mut outputs = Vec::with_capacity(draws.len());
    let mut pass_cache_update = ready.pass_cache.provisional_update();
    let context = CompositionVectorDrawEncodingContext {
        ready,
        requests,
        mask_view: mask_view.as_ref(),
        mask: mask.as_ref(),
        spatial_bytes: &spatial_bytes,
    };
    for draw in draws.iter().copied() {
        let draw_textures = composition_prepare_vector_draw_textures(
            ready,
            &mut encoder,
            working_format,
            vector_source_size,
            draw,
        );
        composition_encode_vector_draw(
            &context,
            &mut pass_cache_update,
            &mut encoder,
            &draw_textures,
            draw,
        )?;
        outputs.push(draw_textures.output);
    }
    Ok(CompositionPreparedGpuVectorsForTest {
        device: ready.device.clone(),
        queue: ready.queue.clone(),
        working_format,
        encoder,
        outputs,
        pass_cache_update,
    })
}

#[cfg(test)]
fn color_filter_limit_error_is_exact(rejection: Option<Error>) -> bool {
    rejection.is_some_and(|error| {
        error.code() == ErrorCode::InvalidInput
            && error.invalid_value_diagnostic().is_some_and(|invalid| {
                invalid.field() == "color filter operation buffer byte length"
            })
    })
}

#[cfg(test)]
fn composition_ordered_encoding_observation(
    summary: &crate::pass::CustomSpineEncodingSummary,
) -> CompositionOrderedGraphEncodingObservationForTest {
    CompositionOrderedGraphEncodingObservationForTest {
        encodes_clip_mask_opacity_and_blend_in_authored_order: summary
            .encodes_custom_passes_in_order
            && summary.layer_composites_bind_exact_resources_and_parameters
            && summary.layer_composites_preserve_signed_mapping
            && summary.advances_every_pass_once,
        normal_uses_fixed_premultiplied_blend: summary.normal_composite_count > 0
            && summary.normal_composites_use_fixed_premultiplied_blend,
        normal_omits_parent_sample: summary.normal_composite_count > 0
            && summary.normal_composites_omit_parent_sample,
        destination_copies_full_parent: summary.destination_composites_copy_full_parent
            && summary.destination_composite_count > 0,
        destination_avoids_read_write_alias: summary.destination_composites_avoid_read_write_alias
            && summary.destination_composite_count > 0,
        composite_count: summary.layer_composite_count,
        one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
        transaction_committed: false,
    }
}

#[cfg(test)]
fn spatial_filter_spatial_encoding_observation(
    summary: &crate::pass::CustomSpineEncodingSummary,
) -> SpatialFilterGraphEncodingObservationForTest {
    SpatialFilterGraphEncodingObservationForTest {
        pass_order: summary.spatial_filter_pass_order.clone(),
        blur_pass_count: summary.blur_pass_count,
        drop_shadow_colorize_count: summary.drop_shadow_colorize_count,
        drop_shadow_merge_count: summary.drop_shadow_merge_count,
        each_pass_advances_once: summary.advances_every_pass_once
            && summary.encodes_custom_passes_in_order,
        binds_exact_prepared_resources: summary.spatial_filter_binds_exact_prepared_resources,
        uses_signed_viewport_and_scissor: summary.spatial_filter_uses_signed_viewport_and_scissor,
        blur_sources_intermediates_and_results_are_distinct: summary
            .blur_sources_intermediates_and_results_are_distinct,
        kernels_release_at_validated_last_use: summary
            .spatial_filter_kernels_release_at_validated_last_use,
        textures_release_at_validated_last_use: summary
            .spatial_filter_textures_release_at_validated_last_use,
        drop_shadow_reads_original_source_twice: summary.drop_shadow_reads_original_source_twice,
        original_source_releases_after_merge: summary.original_source_releases_after_merge,
        one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
        transaction_committed: false,
    }
}

#[cfg(test)]
fn backdrop_encoding_observation(
    summary: &crate::pass::CustomSpineEncodingSummary,
) -> BackdropGraphEncodingObservationForTest {
    BackdropGraphEncodingObservationForTest {
        encodes_copy_filter_clip_foreground_and_group_in_order: summary
            .encodes_custom_passes_in_order
            && summary.copy_backdrop_count == 1
            && summary.color_filter_count > 0
            && summary.blur_pass_count > 0
            && summary.drop_shadow_colorize_count > 0
            && summary.drop_shadow_merge_count > 0
            && summary.layer_composite_count >= 2
            && summary.backdrop_group_order_is_exact
            && summary.advances_every_pass_once,
        parent_is_copied_once: summary.copy_backdrop_count == 1
            && summary.copy_backdrop_binds_exact_prepared_resources
            && summary.copy_backdrop_preserves_signed_mapping,
        copy_filter_foreground_and_group_are_distinct: summary
            .copy_backdrop_source_and_result_are_distinct
            && summary.color_filter_sources_and_results_are_distinct
            && summary.blur_sources_intermediates_and_results_are_distinct
            && summary.parent_and_result_are_distinct
            && summary.backdrop_group_resources_are_distinct,
        later_sibling_reads_completed_group: summary.backdrop_later_sibling_transition_is_exact,
        releases_at_validated_last_use: summary.advances_every_pass_once
            && summary.color_filter_operation_buffers_released
            && summary.spatial_filter_kernels_release_at_validated_last_use
            && summary.spatial_filter_textures_release_at_validated_last_use,
        one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
        transaction_committed: false,
    }
}

#[cfg(test)]
fn spatial_filter_resources_preserved(
    before: &ResourceManagerObservationForTest,
    after: &ResourceManagerObservationForTest,
) -> bool {
    after.leased_count == 0
        && after.active_frame_count == 0
        && after.resolved_lease_count == 0
        && after.accounting_fault_for_test().is_none()
        && after
            .entry_identities_for_test()
            .iter()
            .all(|identity| before.entry_identities_for_test().contains(identity))
}

#[cfg(test)]
fn spatial_filter_failure_publication_for_test(
    device: &wgpu::Device,
    identity: DeviceSlotIdentity,
) -> Result<Surface> {
    let extent = PhysicalSize::new(1, 1);
    let (texture, view) = create_headless_texture(device, extent, Format::Rgba8)?;
    drop(view);
    let mut surface = Surface::with_backend(
        Attachment::Headless,
        SurfaceOptions::default(),
        SurfaceBackend::Headless {
            device_identity: identity,
            resources: HeadlessResources::Pending,
            physical_size: extent,
        },
        RendererIdentity::new(),
    );
    surface.commit_headless_publication(HeadlessPublication::new(texture));
    Ok(surface)
}

#[cfg(test)]
fn custom_spine_observation(
    summary: crate::pass::CustomSpineEncodingSummary,
    capture_count: usize,
    captures_are_exact: bool,
    cache_before: DevicePassCacheCountsForTest,
    cache_after: DevicePassCacheCountsForTest,
) -> CustomSpineEncodingObservationForTest {
    CustomSpineEncodingObservationForTest {
        encodes_custom_passes_in_order: summary.encodes_custom_passes_in_order,
        clears_full_root_once: summary.clears_full_root_once,
        uses_exact_prepared_spatial_mapping: summary.uses_exact_prepared_spatial_mapping,
        presents_to_exact_external_output: summary.presents_to_exact_external_output,
        exposes_bounded_capture_handoff: summary.exposes_bounded_capture_handoff
            && capture_count > 0
            && captures_are_exact,
        validates_checked_capture_completion: summary.validates_checked_capture_completion,
        completes_custom_passes_after_encoding: summary.completes_custom_passes_after_encoding,
        parent_and_result_are_distinct: summary.parent_and_result_are_distinct,
        copies_full_parent_before_bounded_source_render: summary
            .copies_full_parent_before_bounded_source_render,
        samples_only_source_with_fixed_premultiplied_blend: summary
            .samples_only_source_with_fixed_premultiplied_blend,
        preserves_signed_source_origin: summary.preserves_signed_source_origin,
        keeps_cache_update_provisional: summary.keeps_cache_update_provisional
            && cache_after == cache_before,
        encodes_without_submission_or_sync: true,
    }
}

#[cfg(test)]
async fn observe_two_capture_encoding_failure(
    prepared: &mut PreparedGraph<'_>,
    device: &wgpu::Device,
    output: &wgpu::TextureView,
    extent: PhysicalSize,
    failure: TwoCaptureFailureForTest,
) -> Result<(usize, bool, bool, bool)> {
    match failure {
        TwoCaptureFailureForTest::LaterCaptureEncoding => {
            prepared.fail_capture_encoding_after_for_test(1);
        }
        TwoCaptureFailureForTest::SharedScopeResolution => {
            prepared.fail_scope_resolution_for_test();
        }
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist base graph two-capture failure encoder"),
    });
    let result = prepared
        .encode_custom_spine(
            &mut encoder,
            GraphExternalOutputView::try_new(output, Format::Rgba8, extent)?,
        )
        .await;
    let acquired = prepared.acquired_capture_lease_count_for_test();
    let (reported, no_commit) = match result {
        Ok(pending) => {
            drop(pending);
            (false, false)
        }
        Err(error) => (
            match failure {
                TwoCaptureFailureForTest::LaterCaptureEncoding => {
                    error.message() == "prepared runtime resource binding is missing"
                }
                TwoCaptureFailureForTest::SharedScopeResolution => {
                    error.message() == "checked internal Vello resource or command encoding failed"
                }
            },
            true,
        ),
    };
    let mut retry = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist base graph forbidden two-capture retry encoder"),
    });
    let retry_rejected = prepared
        .encode_custom_spine(
            &mut retry,
            GraphExternalOutputView::try_new(output, Format::Rgba8, extent)?,
        )
        .await
        .is_err_and(|error| {
            error.message()
                == "the custom-spine encoding is one-shot; discard this prepared graph and its encoder"
        });
    drop(retry.finish());
    drop(encoder.finish());
    Ok((acquired, reported, no_commit, retry_rejected))
}

#[cfg(test)]
fn graph_test_output_texture(
    device: &wgpu::Device,
    output_extent: PhysicalSize,
    output_format: Format,
    label: &'static str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: output_extent.width(),
            height: output_extent.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::from(output_format),
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

#[cfg(test)]
fn internal_vello_test_target(
    device: &wgpu::Device,
    target_extent: PhysicalSize,
    target_usage: wgpu::TextureUsages,
    label: &'static str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: target_extent.width(),
            height: target_extent.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: target_usage,
        view_formats: &[],
    })
}

impl Backend {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn commit_checked_pass_cache_update_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        update: ProvisionalDevicePassCacheUpdate,
    ) -> Result<()> {
        self.commit_checked_pass_cache_update(
            identity,
            Some(update),
            RuntimeOperation::EffectRendering,
        )
    }
    #[cfg(test)]
    async fn submit_prepared_vello_pass_with_action_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
        action: InternalVelloSubmissionActionForTest<'_>,
    ) -> Result<InternalVelloSubmissionOutcomeForTest> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before internal Vello submission",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before internal Vello submission",
            ));
        }
        let ready = state.ready_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before internal Vello encoding",
            )
        })?;
        let ReadyDeviceState {
            device,
            queue,
            engine,
            resources,
            ..
        } = ready;
        let target_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist transaction-owned internal Vello target"),
            size: wgpu::Extent3d {
                width: target_extent.width(),
                height: target_extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: target_usage,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist transaction-owned internal Vello encoder"),
        });
        let mut scope = ActiveVelloEncodingScope::begin(device);
        let encoded: EncodedVelloPass = {
            let mut encoding = TransactionEncodingState::new(
                &mut scope,
                queue,
                &mut command_encoder,
                &target_view,
                TransactionTargetIntent::new(
                    target_extent,
                    wgpu::TextureFormat::Rgba8Unorm,
                    target_usage,
                ),
            );
            match prepared.encode_into(engine, resources, &mut encoding) {
                Ok(lease) => lease,
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
        match action {
            InternalVelloSubmissionActionForTest::Observe => {
                submit_internal_vello_observed_for_test(
                    transaction,
                    device,
                    queue,
                    payload,
                    RuntimeOperation::SurfaceRendering,
                )
                .await
                .map(InternalVelloSubmissionOutcomeForTest::Observed)
            }
            InternalVelloSubmissionActionForTest::ScopeFailure(publication) => {
                vello_scope_failure_after_submission_for_test(
                    transaction,
                    device,
                    queue,
                    payload,
                    RuntimeOperation::SurfaceRendering,
                    publication,
                )
                .await?;
                Ok(InternalVelloSubmissionOutcomeForTest::Completed)
            }
            InternalVelloSubmissionActionForTest::AccountingFailure => {
                vello_accounting_failure_after_submission_for_test(
                    transaction,
                    queue,
                    payload,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                Ok(InternalVelloSubmissionOutcomeForTest::Completed)
            }
        }
    }

    #[cfg(test)]
    pub(crate) async fn submit_prepared_vello_pass_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<InternalVelloSubmissionObservationForTest> {
        match self
            .submit_prepared_vello_pass_with_action_for_test(
                identity,
                prepared,
                target_extent,
                InternalVelloSubmissionActionForTest::Observe,
            )
            .await?
        {
            InternalVelloSubmissionOutcomeForTest::Observed(observation) => Ok(observation),
            InternalVelloSubmissionOutcomeForTest::Completed => {
                unreachable!("the explicit observe action must return its stage facts")
            }
        }
    }

    #[cfg(test)]
    pub(crate) async fn fail_prepared_vello_pass_after_submit_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
        publication: &mut Option<u64>,
    ) -> Result<()> {
        match self
            .submit_prepared_vello_pass_with_action_for_test(
                identity,
                prepared,
                target_extent,
                InternalVelloSubmissionActionForTest::ScopeFailure(publication),
            )
            .await?
        {
            InternalVelloSubmissionOutcomeForTest::Completed => Ok(()),
            InternalVelloSubmissionOutcomeForTest::Observed(_) => {
                unreachable!("the explicit failure action cannot return observation facts")
            }
        }
    }

    #[cfg(test)]
    pub(crate) async fn fault_prepared_vello_accounting_after_submit_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<()> {
        match self
            .submit_prepared_vello_pass_with_action_for_test(
                identity,
                prepared,
                target_extent,
                InternalVelloSubmissionActionForTest::AccountingFailure,
            )
            .await?
        {
            InternalVelloSubmissionOutcomeForTest::Completed => Ok(()),
            InternalVelloSubmissionOutcomeForTest::Observed(_) => {
                unreachable!("the explicit accounting action cannot return observation facts")
            }
        }
    }

    #[cfg(test)]
    pub(crate) async fn cancel_prepared_vello_pass_after_submit_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<ResourceManagerObservationForTest> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before cancellation submission setup",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before cancellation submission setup",
            ));
        }
        let ready = state.ready_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before cancellation encoding",
            )
        })?;
        let ReadyDeviceState {
            device,
            queue,
            engine,
            resources,
            ..
        } = ready;
        let target_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let target = internal_vello_test_target(
            device,
            target_extent,
            target_usage,
            "Surgeist cancellation-owned internal Vello target",
        );
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist cancellation-owned internal Vello encoder"),
        });
        let mut scope = ActiveVelloEncodingScope::begin(device);
        let encoded: EncodedVelloPass = {
            let mut encoding = TransactionEncodingState::new(
                &mut scope,
                queue,
                &mut command_encoder,
                &target_view,
                TransactionTargetIntent::new(
                    target_extent,
                    wgpu::TextureFormat::Rgba8Unorm,
                    target_usage,
                ),
            );
            match prepared.encode_into(engine, resources, &mut encoding) {
                Ok(lease) => lease,
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
        let mut publication = None;
        let mut submission = Box::pin(hold_internal_vello_after_submit_for_test(
            transaction,
            queue,
            payload,
            &mut publication,
        ));
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let poll = submission.as_mut().poll(&mut context);
        assert!(
            matches!(poll, Poll::Pending),
            "the post-submit cancellation checkpoint must pause the real submission future"
        );
        drop(submission);

        Ok(resources.observation_for_test())
    }
    #[cfg(test)]
    fn prepare_color_filter_graph_resources_with_operation_limits_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        operation_limits: ColorFilterOperationBufferLimits,
    ) -> Result<PreparedGraph<'_>> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot is unavailable for color-filter limit preparation",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before color-filter limit preparation",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(RuntimeOperation::EffectRendering));
        }
        if !state.signal.has_active_operation() {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "color-filter limit preparation requires one active GPU transaction",
            ));
        }
        let capabilities = state.capabilities;
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "ready GPU resources disappeared before color-filter limit preparation",
            )
        })?;
        PreparedGraph::try_prepare_color_filter_with_operation_limits_for_test(
            lowered,
            policy,
            &capabilities,
            &ready.device,
            &ready.queue,
            &ready.resources,
            (&ready.pass_cache, operation_limits),
        )
        .map(|prepared| prepared.with_vello_engine(&ready.engine))
    }

    #[cfg(test)]
    pub(crate) async fn layer_composite_cache_realization_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &LayerCompositeCacheRequestsForTest,
    ) -> Result<LayerCompositeCacheRealizationObservationForTest> {
        let initial_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition composite realization requires a ready device",
            )?
            .pass_cache
            .counts_for_test();
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, provision) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition composite realization lost its ready device",
            )?;
            provision_layer_composite_requests_for_test(ready, requests, false)?
        };
        let counts_before_commit = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition composite realization lost its persistent cache",
            )?
            .pass_cache
            .counts_for_test();
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        self.commit_checked_pass_cache_update(
            identity,
            Some(update),
            RuntimeOperation::EffectRendering,
        )?;
        let committed_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition committed compositor cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        let realizes_normal_and_destination_programs = initial_counts.is_empty()
            && counts_before_commit == initial_counts
            && committed_counts != initial_counts
            && provision.encoding_ready
            && provision.has_normal
            && provision.has_destination
            && layer_composite_requests_are_cached_for_test(
                &self
                    .ready_state_mut(
                        identity,
                        RuntimeOperation::EffectRendering,
                        BackendErrorCode::RenderFailed,
                        "composition committed compositor programs disappeared",
                    )?
                    .pass_cache,
                requests,
            );

        let reuses_exact_committed_entries = self
            .composition_reuses_committed_entries_for_test(identity, requests, committed_counts)
            .await?;

        let failed_validation_publishes_none = self
            .composition_validation_publishes_none_for_test(requests)
            .await?;
        let (cancellation_publishes_none, device_transition_publishes_none) = self
            .composition_cancellation_publishes_none_for_test(requests)
            .await?;

        Ok(LayerCompositeCacheRealizationObservationForTest {
            realizes_normal_and_destination_programs,
            realizes_all_optional_binding_combinations: provision.all_optional_combinations,
            normal_uses_fixed_premultiplied_source_over: provision.normal_uses_fixed_blend,
            destination_uses_replace_blending: provision.destination_uses_replace_blend,
            commits_only_after_clean_transaction: counts_before_commit == initial_counts
                && committed_counts != counts_before_commit,
            reuses_exact_committed_entries,
            failed_validation_publishes_none,
            cancellation_publishes_none,
            device_transition_publishes_none,
        })
    }

    #[cfg(test)]
    async fn composition_reuses_committed_entries_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &LayerCompositeCacheRequestsForTest,
        committed: DevicePassCacheCountsForTest,
    ) -> Result<bool> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, provision) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition compositor cache reuse lost its ready device",
            )?;
            provision_layer_composite_requests_for_test(ready, requests, false)?
        };
        let reused_existing = update.is_empty_for_test();
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        self.commit_checked_pass_cache_update(
            identity,
            Some(update),
            RuntimeOperation::EffectRendering,
        )?;
        let counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition reused compositor cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        Ok(reused_existing && provision.encoding_ready && counts == committed)
    }

    #[cfg(test)]
    async fn composition_validation_publishes_none_for_test(
        &mut self,
        requests: &LayerCompositeCacheRequestsForTest,
    ) -> Result<bool> {
        let identity = self.add_device_slot_for_test().await?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let update = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition validation probe lost its ready device",
            )?;
            provision_layer_composite_requests_for_test(ready, requests, true)?.0
        };
        let error = transaction.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        Ok(error
            .as_ref()
            .is_err_and(|error| error.code() == ErrorCode::RenderFailed)
            && self
                .device_states
                .get(identity.slot())
                .and_then(DeviceState::ready)
                .map(|ready| ready.pass_cache.counts_for_test())
                .is_some_and(DevicePassCacheCountsForTest::is_empty))
    }

    #[cfg(test)]
    async fn composition_cancellation_publishes_none_for_test(
        &mut self,
        requests: &LayerCompositeCacheRequestsForTest,
    ) -> Result<(bool, bool)> {
        let identity = self.add_device_slot_for_test().await?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, provision) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition cancellation probe lost its ready device",
            )?;
            provision_layer_composite_requests_for_test(ready, requests, false)?
        };
        let cache_empty = self
            .device_states
            .get(identity.slot())
            .and_then(DeviceState::ready)
            .map(|ready| ready.pass_cache.counts_for_test())
            .is_some_and(DevicePassCacheCountsForTest::is_empty);
        drop(update);
        drop(transaction);
        let canceled = provision.encoding_ready
            && cache_empty
            && self
                .device_states
                .get(identity.slot())
                .and_then(DeviceState::ready)
                .map(|ready| ready.pass_cache.counts_for_test())
                .is_some_and(DevicePassCacheCountsForTest::is_empty)
            && self
                .device_states
                .get(identity.slot())
                .is_some_and(|state| state.signal.active_generation_for_test().is_none());
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, provision) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition transition probe lost its ready device",
            )?;
            provision_layer_composite_requests_for_test(ready, requests, false)?
        };
        self.signal_loss_for_test(identity, DeviceLossReason::Destroyed);
        let error = transaction.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        let transitioned =
            provision.encoding_ready && error.is_err() && self.renderer_released_for_test(identity);
        Ok((canceled, transitioned))
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn composition_shader_mask_sampling_preparation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &LayerCompositeCacheRequestsForTest,
        input: &CompositionMaskSamplingInputForTest,
    ) -> Result<CompositionPreparedGpuVectorsForTest> {
        let working_format = self
            .device_capabilities(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "composition mask vectors require immutable device capabilities",
                )
            })?
            .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)?;
        let draws = input
            .vectors
            .iter()
            .map(|vector| CompositionGpuVectorDrawForTest {
                path: crate::shader::ShaderCompositePathKey::Normal,
                has_clip_coverage: vector.clip_alpha.is_some(),
                has_alpha_mask: true,
                source: input.source,
                parent: [0.0; 4],
                layer_point: vector.layer_point,
                clip_alpha: vector.clip_alpha.unwrap_or(1.0),
                opacity: vector.opacity,
                blend: BlendMode::Normal,
                quality: vector.quality,
                extend: vector.extend,
            })
            .collect::<Vec<_>>();
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::EffectRendering,
            BackendErrorCode::RenderFailed,
            "composition mask vectors lost their ready device",
        )?;
        encode_composition_gpu_vectors_for_test(
            ready,
            requests,
            working_format,
            Some(CompositionGpuMaskTextureForTest {
                size: input.mask_size,
                rgba: &input.mask_rgba,
                bounds: input.mask_bounds,
            }),
            &draws,
        )
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn composition_shader_blend_preparation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &LayerCompositeCacheRequestsForTest,
        vectors: &[CompositionBlendVectorForTest],
    ) -> Result<CompositionPreparedGpuVectorsForTest> {
        let working_format = self
            .device_capabilities(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "composition blend vectors require immutable device capabilities",
                )
            })?
            .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)?;
        let draws = vectors
            .iter()
            .map(|vector| CompositionGpuVectorDrawForTest {
                path: if vector.blend == BlendMode::Normal {
                    crate::shader::ShaderCompositePathKey::Normal
                } else {
                    crate::shader::ShaderCompositePathKey::DestinationSampling
                },
                has_clip_coverage: false,
                has_alpha_mask: false,
                source: vector.source,
                parent: vector.parent,
                layer_point: Point::new(0.5, 0.5),
                clip_alpha: 1.0,
                opacity: vector.opacity,
                blend: vector.blend,
                quality: ImageQuality::Low,
                extend: Extend::Pad,
            })
            .collect::<Vec<_>>();
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::EffectRendering,
            BackendErrorCode::RenderFailed,
            "composition blend vectors lost their ready device",
        )?;
        encode_composition_gpu_vectors_for_test(ready, requests, working_format, None, &draws)
    }

    #[cfg(test)]
    pub(crate) async fn core_pass_shader_cache_realization_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        rgba_requests: &CorePassCacheRequestsForTest,
        bgra_requests: &CorePassCacheRequestsForTest,
    ) -> Result<CorePassShaderCacheRealizationObservationForTest> {
        let initial_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "core-pass shader-cache observation requires a ready device",
            )?
            .pass_cache
            .counts_for_test();
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (rgba_update, provisional_handles_are_encoding_ready) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "core-pass shader realization lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, rgba_requests, false)?
        };
        let counts_before_commit = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "core-pass shader realization lost its persistent cache",
            )?
            .pass_cache
            .counts_for_test();
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        self.commit_checked_pass_cache_update(
            identity,
            Some(rgba_update),
            RuntimeOperation::EffectRendering,
        )?;
        let rgba_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph committed cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        let realizes_all_checked_programs = initial_counts.is_empty()
            && counts_before_commit == initial_counts
            && rgba_counts != initial_counts
            && core_pass_requests_are_cached_for_test(
                &self
                    .ready_state_mut(
                        identity,
                        RuntimeOperation::EffectRendering,
                        BackendErrorCode::RenderFailed,
                        "base graph committed programs disappeared",
                    )?
                    .pass_cache,
                rgba_requests,
            );

        let reuses_exact_committed_entries = self
            .core_pass_reuses_committed_entries_for_test(identity, rgba_requests, rgba_counts)
            .await?;

        let (failed_validation_publishes_none, specializes_rgba_and_bgra_outputs) = self
            .core_pass_validation_and_specialization_for_test(
                identity,
                rgba_requests,
                bgra_requests,
                rgba_counts,
            )
            .await?;
        let (cancellation_publishes_none, device_transition_publishes_none) = self
            .graph_cancellation_publishes_none_for_test(rgba_requests)
            .await?;

        Ok(CorePassShaderCacheRealizationObservationForTest {
            realizes_all_checked_programs,
            provisional_handles_are_encoding_ready,
            commits_only_after_clean_transaction: counts_before_commit == initial_counts
                && rgba_counts != counts_before_commit,
            reuses_exact_committed_entries,
            failed_validation_publishes_none,
            cancellation_publishes_none,
            device_transition_publishes_none,
            specializes_rgba_and_bgra_outputs,
        })
    }

    #[cfg(test)]
    async fn core_pass_reuses_committed_entries_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &CorePassCacheRequestsForTest,
        committed: DevicePassCacheCountsForTest,
    ) -> Result<bool> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, handles_ready) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "core-pass cache reuse lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, requests, false)?
        };
        let exact_existing = update.is_empty_for_test() && handles_ready;
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        self.commit_checked_pass_cache_update(
            identity,
            Some(update),
            RuntimeOperation::EffectRendering,
        )?;
        let counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph reused cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        Ok(exact_existing && counts == committed)
    }

    #[cfg(test)]
    async fn core_pass_validation_and_specialization_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        rgba: &CorePassCacheRequestsForTest,
        bgra: &CorePassCacheRequestsForTest,
        rgba_counts: DevicePassCacheCountsForTest,
    ) -> Result<(bool, bool)> {
        let validation = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let update = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph validation probe lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, bgra, true)?.0
        };
        let error = validation.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        let after_validation = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph validation probe lost its persistent cache",
            )?
            .pass_cache
            .counts_for_test();
        let failed = error
            .as_ref()
            .is_err_and(|error| error.code() == ErrorCode::RenderFailed)
            && after_validation == rgba_counts;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, handles_ready) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph BGRA specialization lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, bgra, false)?
        };
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        self.commit_checked_pass_cache_update(
            identity,
            Some(update),
            RuntimeOperation::EffectRendering,
        )?;
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::EffectRendering,
            BackendErrorCode::RenderFailed,
            "base graph specialized programs disappeared",
        )?;
        let counts = ready.pass_cache.counts_for_test();
        let specialized = handles_ready
            && counts != rgba_counts
            && core_pass_requests_are_cached_for_test(&ready.pass_cache, rgba)
            && core_pass_requests_are_cached_for_test(&ready.pass_cache, bgra);
        Ok((failed, specialized))
    }

    #[cfg(test)]
    async fn graph_cancellation_publishes_none_for_test(
        &mut self,
        requests: &CorePassCacheRequestsForTest,
    ) -> Result<(bool, bool)> {
        let identity = self.add_device_slot_for_test().await?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, handles_ready) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph cancellation probe lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, requests, false)?
        };
        let cache_empty = self
            .device_states
            .get(identity.slot())
            .and_then(DeviceState::ready)
            .map(|ready| ready.pass_cache.counts_for_test())
            .is_some_and(DevicePassCacheCountsForTest::is_empty);
        drop(update);
        drop(transaction);
        let canceled = handles_ready
            && cache_empty
            && self
                .device_states
                .get(identity.slot())
                .and_then(DeviceState::ready)
                .map(|ready| ready.pass_cache.counts_for_test())
                .is_some_and(DevicePassCacheCountsForTest::is_empty)
            && self
                .device_states
                .get(identity.slot())
                .is_some_and(|state| state.signal.active_generation_for_test().is_none());
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (update, handles_ready) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "base graph transition probe lost its ready device",
            )?;
            provision_core_pass_requests_for_test(ready, requests, false)?
        };
        self.signal_loss_for_test(identity, DeviceLossReason::Destroyed);
        let cache_empty = self
            .device_states
            .get(identity.slot())
            .and_then(DeviceState::ready)
            .map(|ready| ready.pass_cache.counts_for_test())
            .is_some_and(DevicePassCacheCountsForTest::is_empty);
        let error = transaction.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        let transitioned = handles_ready
            && cache_empty
            && error.is_err()
            && self.renderer_released_for_test(identity);
        Ok((canceled, transitioned))
    }

    #[cfg(test)]
    pub(crate) async fn custom_spine_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
        output_format: Format,
    ) -> Result<CustomSpineEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "custom-spine observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::forced_base_graph_for_test(commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            output_format,
            &capabilities,
        )?;
        let pass_cache_before = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "custom-spine observation requires a ready pass cache",
            )?
            .pass_cache
            .counts_for_test();
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "custom-spine observation lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = graph_test_output_texture(
            &device,
            output_extent,
            output_format,
            "Surgeist graph external output observation",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output = GraphExternalOutputView::try_new(&output_view, output_format, output_extent)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph caller-owned custom-spine observation encoder"),
        });
        let encoded = prepared.encode_custom_spine(&mut encoder, output).await?;
        let (summary, capture_resources) = encoded.into_summary_and_resources();
        let capture_handoff_count = summary.capture_count;
        let capture_handoffs_are_exact = summary.capture_observations.iter().all(|capture| {
            capture.target_extent.width() > 0
                && capture.target_extent.height() > 0
                && capture.target_and_view_are_exact
                && matches!(
                    capture.antialiasing,
                    Antialiasing::Area | Antialiasing::Msaa8 | Antialiasing::Msaa16
                )
        });
        drop(capture_resources);
        let command_buffer = encoder.finish();
        drop(command_buffer);
        drop(prepared);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        let pass_cache_after = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "custom-spine observation lost its provisional cache boundary",
            )?
            .pass_cache
            .counts_for_test();

        Ok(custom_spine_observation(
            summary,
            capture_handoff_count,
            capture_handoffs_are_exact,
            pass_cache_before,
            pass_cache_after,
        ))
    }

    #[cfg(test)]
    pub(crate) async fn ordered_color_filter_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<OrderedColorFilterGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "color-filter graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::authored_filter_graph_for_test(filters, commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (device, queue) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "color-filter graph encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist color-filter caller-owned graph observation encoder"),
        });
        let pending = match prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await
        {
            Ok(pending) => pending,
            Err(encoding_error) => {
                drop(encoder.finish());
                drop(prepared);
                return match transaction.finish(RuntimeOperation::EffectRendering).await {
                    Ok(()) => Err(encoding_error),
                    Err(scope_error) => Err(scope_error),
                };
            }
        };
        let summary = pending.summary_for_test();
        let mut observed = OrderedColorFilterGraphEncodingObservationForTest {
            fused_runs_preserve_authored_order: summary.color_filters_preserve_authored_order
                && summary.encodes_custom_passes_in_order,
            color_pass_count: summary.color_filter_count,
            binds_exact_source_spatial_and_operations: summary
                .color_filters_bind_exact_source_spatial_and_operations
                && summary.color_filters_preserve_signed_texel_mapping,
            source_and_result_are_distinct: summary.color_filter_sources_and_results_are_distinct,
            uses_validated_viewport_and_scissor: summary
                .color_filters_use_validated_viewport_and_scissor,
            releases_every_resource_at_last_use: summary.color_filter_operation_buffers_released
                && summary.advances_every_pass_once,
            one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
            transaction_committed: false,
        };
        let prepared_submission = prepared.finish_graph_submission(pending)?;
        drop(output_view);
        let payload = GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "color-filter graph encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_base_graph(
                    &device,
                    &queue,
                    &mut ready.pass_cache,
                    payload,
                    RuntimeOperation::EffectRendering,
                )
                .await?
        };
        let _ = committed.into_parts();
        observed.transaction_committed = true;
        Ok(observed)
    }

    #[cfg(test)]
    pub(crate) async fn spatial_filter_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<SpatialFilterGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "spatial-filter encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::authored_filter_graph_for_test(filters, commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (device, queue) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "spatial-filter encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist spatial-filter caller-owned graph observation encoder"),
        });
        let pending = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let summary = pending.summary_for_test();
        let mut observed = spatial_filter_spatial_encoding_observation(summary);
        let prepared_submission = prepared.finish_graph_submission(pending)?;
        drop(output_view);
        let payload = GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "spatial-filter encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_base_graph(
                    &device,
                    &queue,
                    &mut ready.pass_cache,
                    payload,
                    RuntimeOperation::EffectRendering,
                )
                .await?
        };
        let _ = committed.into_parts();
        observed.transaction_committed = true;
        Ok(observed)
    }

    #[cfg(test)]
    pub(crate) async fn backdrop_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<BackdropGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let crate::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop encoding observation requires a validated GPU graph",
            ));
        };
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (device, queue) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "backdrop encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist backdrop caller-owned graph observation encoder"),
        });
        let pending = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let mut observed = backdrop_encoding_observation(pending.summary_for_test());
        let prepared_submission = prepared.finish_graph_submission(pending)?;
        drop(output_view);
        let payload = GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "backdrop encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_base_graph(
                    &device,
                    &queue,
                    &mut ready.pass_cache,
                    payload,
                    RuntimeOperation::EffectRendering,
                )
                .await?
        };
        let _ = committed.into_parts();
        observed.transaction_committed = true;
        Ok(observed)
    }

    #[cfg(test)]
    pub(crate) async fn backdrop_failure_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<BackdropFailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let crate::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop failure observation requires a validated GPU graph",
            ));
        };
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "backdrop failure observation lost its publication device",
            )?
            .device
            .clone();
        let published_surface = spatial_filter_failure_publication_for_test(&device, identity)?;
        let publication_count_before = published_surface.headless_publication_count_for_test();
        let publication_state_before = published_surface.resource_state();
        let (resources_before, cache_before) =
            self.spatial_filter_resource_and_cache_state(identity)?;
        let encode_error = self
            .run_spatial_filter_failed_encoding_attempt(
                identity,
                lowered,
                policy,
                SpatialFilterInjectedFailureForTest::Encode,
            )
            .await?;
        let (resources_after, cache_after) =
            self.spatial_filter_resource_and_cache_state(identity)?;
        Ok(BackdropFailurePreservationObservationForTest {
            encode_failure_is_reported: encode_error
                .message()
                .contains("injected color-filter shader failure"),
            resources_are_unchanged: spatial_filter_resources_preserved(
                &resources_before,
                &resources_after,
            ),
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
        })
    }

    #[cfg(test)]
    pub(crate) async fn spatial_filter_failure_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<SpatialFilterFailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "spatial-filter failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::authored_filter_graph_for_test(filters, commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "spatial-filter failure observation lost its publication device",
            )?
            .device
            .clone();
        let published_surface = spatial_filter_failure_publication_for_test(&device, identity)?;
        let publication_count_before = published_surface.headless_publication_count_for_test();
        let publication_state_before = published_surface.resource_state();
        let (resources_before, cache_before) =
            self.spatial_filter_resource_and_cache_state(identity)?;
        let encode_error = self
            .run_spatial_filter_failed_encoding_attempt(
                identity,
                lowered.clone(),
                policy,
                SpatialFilterInjectedFailureForTest::Encode,
            )
            .await?;
        let scope_error = self
            .run_spatial_filter_failed_encoding_attempt(
                identity,
                lowered,
                policy,
                SpatialFilterInjectedFailureForTest::Scope,
            )
            .await?;
        let (resources_after, cache_after) =
            self.spatial_filter_resource_and_cache_state(identity)?;
        Ok(SpatialFilterFailurePreservationObservationForTest {
            encode_failure_is_reported: encode_error
                .message()
                .contains("injected color-filter shader failure"),
            scope_failure_is_reported: scope_error.message()
                == "checked internal Vello resource or command encoding failed",
            resources_are_unchanged: spatial_filter_resources_preserved(
                &resources_before,
                &resources_after,
            ),
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
        })
    }

    #[cfg(test)]
    fn spatial_filter_resource_and_cache_state(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Result<(
        ResourceManagerObservationForTest,
        DevicePassCacheCountsForTest,
    )> {
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::EffectRendering,
            BackendErrorCode::RenderFailed,
            "spatial-filter failure observation lost its ready state",
        )?;
        Ok((
            ready.resources.observation_for_test(),
            ready.pass_cache.counts_for_test(),
        ))
    }

    #[cfg(test)]
    async fn run_spatial_filter_failed_encoding_attempt(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        failure: SpatialFilterInjectedFailureForTest,
    ) -> Result<Error> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "spatial-filter failure attempt lost its ready device",
            )?
            .device
            .clone();
        let _encode_failure = matches!(failure, SpatialFilterInjectedFailureForTest::Encode)
            .then(crate::pass::ScopedColorFilterShaderFailureForTest::after_checked_realization);
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        if matches!(failure, SpatialFilterInjectedFailureForTest::Scope) {
            prepared.fail_scope_resolution_for_test();
        }
        let extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist spatial-filter injected-failure graph encoder"),
        });
        let result = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, extent)?,
            )
            .await;
        let result = result.map_err(crate::pass::normalize_color_filter_shader_failure_for_test);
        drop(output_view);
        drop(output_texture);
        drop(encoder.finish());
        drop(prepared);
        let scope_result = transaction.finish(RuntimeOperation::EffectRendering).await;
        match failure {
            SpatialFilterInjectedFailureForTest::Encode => {
                scope_result?;
                result.err().ok_or_else(|| {
                    Error::new(
                        BackendErrorCode::RenderFailed,
                        "the injected spatial-filter encoding failure unexpectedly succeeded",
                    )
                })
            }
            SpatialFilterInjectedFailureForTest::Scope => {
                let encoding_failed = result.is_err();
                drop(result);
                scope_result
                    .err()
                    .map(crate::pass::normalize_scope_resolution_failure_for_test)
                    .filter(|_| encoding_failed)
                    .ok_or_else(|| {
                        Error::new(
                            BackendErrorCode::RenderFailed,
                            "the injected spatial-filter scope failure unexpectedly succeeded",
                        )
                    })
            }
        }
    }

    #[cfg(test)]
    pub(crate) async fn color_filter_oversized_buffer_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<ColorFilterOversizedBufferPreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "color-filter limit observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::authored_filter_graph_for_test(filters, commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "color-filter limit observation lost its ready device",
            )?
            .device
            .clone();
        let publication_extent = PhysicalSize::new(1, 1);
        let (published_texture, published_view) =
            create_headless_texture(&device, publication_extent, Format::Rgba8)?;
        drop(published_view);
        let mut published_surface = Surface::with_backend(
            Attachment::Headless,
            SurfaceOptions::default(),
            SurfaceBackend::Headless {
                device_identity: identity,
                resources: HeadlessResources::Pending,
                physical_size: publication_extent,
            },
            RendererIdentity::new(),
        );
        published_surface.commit_headless_publication(HeadlessPublication::new(published_texture));
        let publication_count_before = published_surface.headless_publication_count_for_test();
        let publication_state_before = published_surface.resource_state();
        let (resources_before, cache_before) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "color-filter limit observation lost its preflight state",
            )?;
            (
                ready.resources.observation_for_test(),
                ready.pass_cache.counts_for_test(),
            )
        };

        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let first_run_byte_len = 16_u64 + 3 * 32;
        let rejection = match self
            .prepare_color_filter_graph_resources_with_operation_limits_for_test(
                identity,
                lowered,
                policy,
                ColorFilterOperationBufferLimits::for_test(first_run_byte_len - 1, u64::MAX),
            ) {
            Ok(prepared) => {
                drop(prepared);
                None
            }
            Err(error) => Some(error),
        };
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;

        let (resources_after, cache_after) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "color-filter limit observation lost its post-rejection state",
            )?;
            (
                ready.resources.observation_for_test(),
                ready.pass_cache.counts_for_test(),
            )
        };
        let returns_exact_limit_error = color_filter_limit_error_is_exact(rejection);
        Ok(ColorFilterOversizedBufferPreservationObservationForTest {
            returns_exact_limit_error,
            resources_are_unchanged: resources_after == resources_before,
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
        })
    }

    #[cfg(test)]
    pub(crate) async fn composition_ordered_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<CompositionOrderedGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "composition graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let crate::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "composition graph encoding observation requires a validated GPU graph",
            ));
        };
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let (device, queue) = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition graph encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist composition caller-owned graph observation encoder"),
        });
        let pending = match prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await
        {
            Ok(pending) => pending,
            Err(_) => {
                drop(encoder.finish());
                drop(prepared);
                transaction
                    .finish(RuntimeOperation::EffectRendering)
                    .await?;
                return Ok(CompositionOrderedGraphEncodingObservationForTest::default());
            }
        };
        let summary = pending.summary_for_test();
        let mut observed = composition_ordered_encoding_observation(summary);
        let prepared_submission = prepared.finish_graph_submission(pending)?;
        drop(output_view);
        let payload = GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "composition graph encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_base_graph(
                    &device,
                    &queue,
                    &mut ready.pass_cache,
                    payload,
                    RuntimeOperation::EffectRendering,
                )
                .await?
        };
        let _ = committed.into_parts();
        observed.transaction_committed = true;
        Ok(observed)
    }

    #[cfg(test)]
    pub(crate) async fn multiple_vello_capture_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        donor_commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
    ) -> Result<MultipleVelloCaptureEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "multiple Vello capture coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = crate::pass::two_capture_spine_lowered_for_test(
            commands,
            donor_commands,
            context,
            capabilities,
            policy,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let transaction_generation = self.active_operation_generation_for_test(identity);
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "multiple Vello capture coverage lost its ready device",
            )?
            .device
            .clone();
        let mut prepared =
            self.prepare_graph_resources_for_test(identity, lowered.clone(), policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = graph_test_output_texture(
            &device,
            output_extent,
            Format::Rgba8,
            "Surgeist base graph multiple-capture output",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph multiple-capture graph encoder"),
        });
        let encoded = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let (summary, capture_resources) = encoded.into_summary_and_resources();
        let committed_lease_count = capture_resources.lease_count_for_test();
        drop(encoder.finish());
        drop(prepared);
        let same_transaction = transaction_generation.is_some()
            && transaction_generation == self.active_operation_generation_for_test(identity);
        finish_vello_resources_without_submission_for_test(
            transaction,
            capture_resources,
            RuntimeOperation::EffectRendering,
        )
        .await?;
        let after_commit = self
            .ready_device_state_borrow_for_test(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "multiple Vello capture commit lost its resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();

        let (aborted_lease_count, after_abort) = self
            .multiple_capture_abort_for_test(
                identity,
                lowered,
                policy,
                &device,
                &output_view,
                output_extent,
            )
            .await?;

        Ok(MultipleVelloCaptureEncodingObservationForTest {
            exact_capture_count: summary.capture_count == 2
                && summary.exposes_bounded_capture_handoff,
            one_graph_command_encoder: summary.captures_share_one_command_encoder,
            one_gpu_transaction: same_transaction,
            one_active_vello_scope: summary.captures_share_one_active_vello_scope,
            aggregate_pending_commit: committed_lease_count == 2 && aborted_lease_count == 2,
            commits_every_capture_after_transaction_success: committed_lease_count == 2
                && after_commit.leased_count == 0
                && after_commit.recovery_outcome_for_test().is_none(),
            aborts_every_capture_on_drop: aborted_lease_count == 2
                && after_abort.leased_count == 0
                && after_abort.recovery_outcome_for_test() == Some(VelloAtlasOutcome::Recreate),
        })
    }

    #[cfg(test)]
    async fn multiple_capture_abort_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        device: &wgpu::Device,
        output: &wgpu::TextureView,
        extent: PhysicalSize,
    ) -> Result<(usize, ResourceManagerObservationForTest)> {
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph multiple-capture aggregate-abort encoder"),
        });
        let encoded = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(output, Format::Rgba8, extent)?,
            )
            .await?;
        let (_, resources) = encoded.into_summary_and_resources();
        let count = resources.lease_count_for_test();
        drop(encoder.finish());
        drop(prepared);
        drop(resources);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        let observation = self
            .ready_device_state_borrow_for_test(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "multiple Vello capture abort lost its resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();
        Ok((count, observation))
    }

    #[cfg(test)]
    pub(crate) async fn two_capture_failure_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        donor_commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
        failure: TwoCaptureFailureForTest,
    ) -> Result<TwoCaptureFailureObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "two-capture failure coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = crate::pass::two_capture_spine_lowered_for_test(
            commands,
            donor_commands,
            context,
            capabilities,
            policy,
        )?;
        let resources_before = self
            .ready_device_state_borrow_for_test(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "two-capture failure coverage lost its initial resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "two-capture failure coverage lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = graph_test_output_texture(
            &device,
            output_extent,
            Format::Rgba8,
            "Surgeist base graph two-capture failure output",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (
            acquired_capture_lease_count,
            mut failure_is_reported,
            produces_no_pending_commit,
            retry_is_rejected,
        ) = observe_two_capture_encoding_failure(
            &mut prepared,
            &device,
            &output_view,
            output_extent,
            failure,
        )
        .await?;
        drop(prepared);
        let scope_result = transaction.finish(RuntimeOperation::EffectRendering).await;
        if matches!(failure, TwoCaptureFailureForTest::SharedScopeResolution) {
            failure_is_reported &= scope_result
                .err()
                .map(crate::pass::normalize_scope_resolution_failure_for_test)
                .is_some_and(|error| {
                    error.message() == "checked internal Vello resource or command encoding failed"
                });
        } else {
            scope_result?;
        }
        let transaction_lease_is_released = self
            .active_operation_generation_for_test(identity)
            .is_none();
        let resources_after = self
            .ready_device_state_borrow_for_test(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "two-capture failure coverage lost its cleanup resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();

        Ok(TwoCaptureFailureObservationForTest {
            acquired_capture_lease_count,
            failure_is_reported,
            produces_no_pending_commit,
            retry_is_rejected,
            resource_creation_was_observed: resources_after.payload_creation_attempts
                > resources_before.payload_creation_attempts,
            remaining_leased_resource_count: resources_after.leased_count,
            remaining_resource_count: resources_after.entry_count,
            atlas_recovery_outcome: resources_after.recovery_outcome_for_test(),
            transaction_lease_is_released,
        })
    }

    #[cfg(test)]
    pub(crate) async fn vello_capture_raster_contract_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
        requested_antialiasing: Antialiasing,
    ) -> Result<VelloCaptureRasterContractObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "Vello capture raster coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::forced_base_graph_for_test(commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "Vello capture raster coverage lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist base graph raster-contract output"),
            size: wgpu::Extent3d {
                width: output_extent.width(),
                height: output_extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph raster-contract graph encoder"),
        });
        let encoded = prepared
            .encode_custom_spine(
                &mut encoder,
                GraphExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let (summary, capture_resources) = encoded.into_summary_and_resources();
        let capture = summary.capture_observations.first().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "Vello capture raster coverage produced no encoded capture proof",
            )
        })?;
        let observed = VelloCaptureRasterContractObservationForTest {
            lowers_with_exact_initial_transform: capture.lowers_with_exact_initial_transform,
            uses_transparent_base: capture.uses_transparent_base,
            uses_requested_antialiasing: capture.antialiasing == requested_antialiasing,
            uses_exact_positive_extent: capture.target_extent.width() > 0
                && capture.target_extent.height() > 0,
            uses_exact_rgba8_target_and_view: capture.target_and_view_are_exact
                && capture.target_format == wgpu::TextureFormat::Rgba8Unorm,
            uses_exact_capture_usage: capture.target_usage
                == crate::pass::VELLO_CAPTURE_TEXTURE_USAGES,
            has_unforgeable_encoded_capture_proof: summary.validates_checked_capture_completion,
        };
        drop(capture_resources);
        drop(encoder.finish());
        drop(prepared);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        Ok(observed)
    }

    #[cfg(test)]
    pub(crate) async fn vello_capture_failure_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: crate::command::RenderCommands,
        context: crate::frame::FrameContext,
        output_format: Format,
    ) -> Result<VelloCaptureFailureObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "Vello capture-failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = crate::frame::forced_base_graph_for_test(commands, context)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            output_format,
            &capabilities,
        )?;
        let transaction = self.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let device = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "Vello capture-failure observation lost its ready device",
            )?
            .device
            .clone();

        let mut first = self.prepare_graph_resources_for_test(identity, lowered.clone(), policy)?;
        let output_extent = first.output_extent()?;
        let output_texture = graph_test_output_texture(
            &device,
            output_extent,
            output_format,
            "Surgeist Vello capture-failure external output observation",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut first_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph failed-capture first encoder observation"),
        });
        let failed_pass = first
            .base_execution_facts()
            .and_then(|facts| facts.captures().first())
            .map(crate::pass::ExecutableVelloCaptureFacts::pass);
        first.fail_capture_encoding_for_test();
        let capture_failure_is_reported = first
            .encode_custom_spine(
                &mut first_encoder,
                GraphExternalOutputView::try_new(&output_view, output_format, output_extent)?,
            )
            .await
            .is_err_and(|error| error.message() == "prepared runtime resource binding is missing")
            && failed_pass.is_some();
        let complete_pass_is_rejected =
            failed_pass.is_some_and(|pass| first.complete_pass(pass).is_err());
        drop(first_encoder.finish());
        drop(first);

        let mut retried = self.prepare_graph_resources_for_test(identity, lowered, policy)?;
        let mut failed_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph failed-capture retry source encoder observation"),
        });
        retried.fail_capture_encoding_for_test();
        let initial_failure = retried
            .encode_custom_spine(
                &mut failed_encoder,
                GraphExternalOutputView::try_new(&output_view, output_format, output_extent)?,
            )
            .await
            .is_err_and(|error| error.message() == "prepared runtime resource binding is missing");
        drop(failed_encoder.finish());
        let mut retry_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist base graph forbidden new retry encoder observation"),
        });
        let retry_on_new_encoder_is_rejected = initial_failure
            && retried
                .encode_custom_spine(
                    &mut retry_encoder,
                    GraphExternalOutputView::try_new(&output_view, output_format, output_extent)?,
                )
                .await
                .is_err();
        drop(retry_encoder.finish());
        drop(retried);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;

        Ok(VelloCaptureFailureObservationForTest {
            capture_failure_is_reported,
            complete_pass_is_rejected,
            retry_on_new_encoder_is_rejected,
        })
    }
}
