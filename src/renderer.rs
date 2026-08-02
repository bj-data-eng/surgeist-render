#[cfg(test)]
use super::resource::{ResourceManagerObservationForTest, WorkingFormat};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, PresentedSurfaceState, ResizeState};
use super::{
    backend::*,
    command::RenderCommands,
    encode::encode_vello_scene,
    frame::{
        FrameContext, FramePlan, GpuRenderGraph, GraphLoweringCompositeKind, GraphLoweringPassKind,
    },
    geometry::physical_size,
    gpu_transaction::{GpuOperationDraft, GpuOperationStage},
    pass::{ExecutableGraphDispatchEligibility, ExecutableGraphWorkingFormatRequest},
    readback::read_texture_rgba,
    stats::collect_render_stats,
    surface::{HeadlessResources, RendererIdentity, SurfaceBackend},
    validation::*,
    vello_engine::scene::VelloScene,
    *,
};
#[cfg(test)]
use std::{cell::RefCell, sync::Arc};
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

#[cfg(test)]
thread_local! {
    static ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST: RefCell<bool> = const { RefCell::new(false) };
}

#[cfg(all(test, feature = "render-window"))]
thread_local! {
    static ACTIVE_PRESENTED_CREATION_LOSS_FOR_TEST: RefCell<bool> = const { RefCell::new(false) };
}

pub struct Renderer {
    identity: RendererIdentity,
    options: Options,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    backend: Option<Backend>,
    default_device: Option<DeviceSlotIdentity>,
    #[cfg(test)]
    preexecution_frame_gate_observation: PreexecutionFrameGateObservationForTest,
    #[cfg(test)]
    dispatch_observation: RendererDispatchObservationForTest,
    #[cfg(test)]
    exact_graph_working_format: Option<WorkingFormat>,
}

#[must_use = "the renderer dispatch boundary must resolve to exactly one execution route"]
enum RendererFrameDispatch {
    DirectVello(RenderCommands),
    ExactGraph(Box<ExactSurfaceGraph>),
}

#[must_use = "prepared renderer execution must reach its selected GPU transaction"]
enum PreparedRendererExecution {
    DirectVello(Box<VelloScene>),
    ExactGraph(Box<ExactSurfaceGraph>),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreexecutionFrameGateObservationForTest {
    pub(crate) validated_plan_count: u8,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RendererDispatchObservationForTest {
    pub(crate) boundary_invocations: usize,
    pub(crate) direct_vello_routes: usize,
    pub(crate) exact_c08_graph_routes: usize,
    pub(crate) exact_c09_graph_routes: usize,
    pub(crate) exact_c10_fixture_routes: usize,
    pub(crate) exact_c11_fixture_routes: usize,
    pub(crate) exact_c12_graph_routes: usize,
    pub(crate) exact_c12_fixture_routes: usize,
    pub(crate) future_pass_rejections: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResourcePreparationObservationForTest {
    pub(crate) complete_resource_and_pass_handoff: bool,
    pub(crate) exact_capture_coverage_working_and_mask_allocations: bool,
    pub(crate) typed_bindings_and_last_use_releases: bool,
    pub(crate) spatial_bytes_and_cache_keys_preserved: bool,
    pub(crate) allocation_preflight_is_atomic: bool,
    pub(crate) failure_and_drop_cleanup: bool,
    pub(crate) repeated_reuse_is_exact_and_bounded: bool,
    pub(crate) populated_pass_cache_is_preserved: bool,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct C08ForcedGraphRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) captures: Vec<C08ForcedGraphCaptureForTest>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C08ForcedGraphCaptureForTest {
    pub(crate) antialiasing: Antialiasing,
    pub(crate) capture_transform: Transform,
    pub(crate) parent_to_surface: Transform,
    pub(crate) device_origin: (i32, i32),
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
}

#[cfg(test)]
impl From<super::frame::ForcedC08GraphCaptureObservationForTest> for C08ForcedGraphCaptureForTest {
    fn from(capture: super::frame::ForcedC08GraphCaptureObservationForTest) -> Self {
        Self {
            antialiasing: capture.antialiasing,
            capture_transform: capture.capture_transform,
            parent_to_surface: capture.parent_to_surface,
            device_origin: capture.device_origin,
            texel_origin: capture.texel_origin,
            extent: capture.extent,
            raster_scale: capture.raster_scale,
        }
    }
}

#[cfg(test)]
struct ForcedC08PreparationForTest {
    device_identity: DeviceSlotIdentity,
    normalized: RenderCommands,
    preparable: super::pass::C08PreparableGraph,
    output_extent: PhysicalSize,
    captures: Vec<C08ForcedGraphCaptureForTest>,
}

#[cfg(test)]
fn preparation_resource_observation(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    missing: &'static str,
) -> Result<ResourceManagerObservationForTest> {
    backend
        .ready_device_state_borrow_for_test(identity)
        .ok_or_else(|| Error::new(BackendErrorCode::RenderFailed, missing))
        .map(|ready| ready.internal_resource_manager_observation_for_test())
}

#[cfg(test)]
fn preparation_preflight_is_atomic(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: &super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<bool> {
    let before = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared before preparation preflight",
    )?;
    let rejected = backend
        .prepare_graph_resources(
            identity,
            lowered.with_duplicate_preparation_resource_for_test(),
            policy,
        )
        .is_err();
    let after = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after preparation preflight",
    )?;
    Ok(rejected && before == after)
}

#[cfg(test)]
fn exercise_preparation_reuse(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: &super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<(super::pass::PreparedGraphExerciseObservationForTest, bool)> {
    let (first_exercise, first_identities) = {
        let mut prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let identities = prepared.allocation_identities_for_test();
        let exercise = prepared.exercise_for_test()?;
        let _ = prepared.finish()?;
        (exercise, identities)
    };
    let after_first = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after first complete preparation",
    )?;
    let second_identities = {
        let mut prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let identities = prepared.allocation_identities_for_test();
        let _ = prepared.exercise_for_test()?;
        let _ = prepared.finish()?;
        identities
    };
    let after_second = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after repeated complete preparation",
    )?;
    let reuse = first_identities == second_identities
        && after_second.payload_creation_attempts == after_first.payload_creation_attempts
        && after_second.entry_count == after_first.entry_count
        && after_second.retained_bytes == after_first.retained_bytes;
    Ok((first_exercise, reuse))
}

#[cfg(test)]
fn preparation_failure_cleanup(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<bool> {
    let early_finish_failed = {
        let prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        prepared.finish().is_err()
    };
    let after_finish = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after failed prepared finish",
    )?;
    drop(backend.prepare_graph_resources(identity, lowered, policy)?);
    let after_drop = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after prepared cancellation",
    )?;
    Ok(early_finish_failed && after_finish.leased_count == 0 && after_drop.leased_count == 0)
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct C10ColorFilterRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) source_origin: (i32, i32),
    pub(crate) source_extent: PhysicalSize,
    pub(crate) source_texel_origin: Point,
    pub(crate) source_raster_scale: f64,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct C11SpatialFilterRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) source_spatial: super::pass::C10ColorSpatialObservationForTest,
    pub(crate) result_spatial: super::pass::C10ColorSpatialObservationForTest,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct C12BackdropRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) parent_spatial: super::pass::C10ColorSpatialObservationForTest,
    pub(crate) capture_spatial: super::pass::C10ColorSpatialObservationForTest,
}

#[cfg(test)]
struct C10ColorFilterFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    source_spatial: super::pass::C10ColorSpatialObservationForTest,
}

#[cfg(test)]
struct C11SpatialFilterFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    source_spatial: super::pass::C10ColorSpatialObservationForTest,
    result_spatial: super::pass::C10ColorSpatialObservationForTest,
}

#[cfg(test)]
struct C12BackdropFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    parent_spatial: super::pass::C10ColorSpatialObservationForTest,
    capture_spatial: super::pass::C10ColorSpatialObservationForTest,
}

struct RenderPublication {
    frame: SurfaceFrameCommit,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    parameters: Parameters,
}

impl RenderPublication {
    fn commit(self, renderer: &mut Renderer, surface: &mut Surface) -> Stats {
        self.frame.commit(surface);
        renderer.stats = self.stats;
        renderer.uploaded_images = self.uploaded_images;
        surface.last_parameters = Some(self.parameters);
        self.stats
    }
}

/// Private control that injects loss after a clean transaction and before publication.
#[cfg(test)]
pub(crate) struct ScopedFinalPublicationLossForTest {
    previous: bool,
}

#[cfg(test)]
impl ScopedFinalPublicationLossForTest {
    pub(crate) fn after_transaction_completion() -> Self {
        let previous = ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ScopedFinalPublicationLossForTest {
    fn drop(&mut self) {
        ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous;
        });
    }
}

/// Private control that injects loss after presented creation and before configuration.
#[cfg(all(test, feature = "render-window"))]
pub(crate) struct ScopedPresentedCreationTerminalLossForTest {
    previous: bool,
}

#[cfg(all(test, feature = "render-window"))]
impl ScopedPresentedCreationTerminalLossForTest {
    pub(crate) fn after_device_selection() -> Self {
        let previous = ACTIVE_PRESENTED_CREATION_LOSS_FOR_TEST.with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(all(test, feature = "render-window"))]
impl Drop for ScopedPresentedCreationTerminalLossForTest {
    fn drop(&mut self) {
        ACTIVE_PRESENTED_CREATION_LOSS_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous;
        });
    }
}

#[cfg(test)]
fn inject_final_publication_loss_for_test(signal: &DeviceSignal) {
    if ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| *active.borrow()) {
        signal.record_loss_for_test(DeviceLossReason::Unknown);
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
fn ensure_presented_device_available_after_creation(
    backend: &mut Backend,
    device_identity: DeviceSlotIdentity,
    operation: RuntimeOperation,
) -> Result<()> {
    #[cfg(all(test, feature = "render-window"))]
    if ACTIVE_PRESENTED_CREATION_LOSS_FOR_TEST.with(|active| *active.borrow()) {
        backend.signal_loss_for_test(device_identity, DeviceLossReason::Unknown);
    }
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    Ok(())
}

impl Renderer {
    pub async fn new(options: Options) -> Result<Self> {
        let mut backend = Backend::new(options.resource_cache_budget());
        let default_device = backend.select_device(None).await?;
        let backend = default_device.map(|_| backend);

        Ok(Self {
            identity: RendererIdentity::new(),
            options,
            stats: Stats::default(),
            uploaded_images: HashSet::new(),
            backend,
            default_device,
            #[cfg(test)]
            preexecution_frame_gate_observation: PreexecutionFrameGateObservationForTest::default(),
            #[cfg(test)]
            dispatch_observation: RendererDispatchObservationForTest::default(),
            #[cfg(test)]
            exact_graph_working_format: None,
        })
    }

    /// Creates a surface and awaits any native or WebGPU surface setup.
    ///
    /// The returned surface is ready for its next lifecycle operation when this
    /// future succeeds. Invalid options and unsupported attachments preserve
    /// their existing diagnostics when the future is awaited. This future does
    /// not promise to be `Send`.
    pub async fn create_surface(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        self.create_surface_with_configuration_operation(
            attachment,
            options,
            RuntimeOperation::SurfaceRendering,
            None,
        )
        .await
    }

    async fn create_surface_with_configuration_operation(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        match attachment {
            Attachment::Headless => self.create_headless_surface(options).await,
            Attachment::WebCanvas(canvas) => {
                self.create_web_canvas_surface(
                    canvas,
                    options,
                    configuration_operation,
                    preferred_device,
                )
                .await
            }
            #[cfg(feature = "render-window")]
            Attachment::Window(handle) => {
                let Some(backend) = self.backend.as_mut() else {
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::AdapterSelection,
                        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                        "no compatible wgpu adapter is available",
                    ));
                };
                let physical_size = physical_size(options.size, options.scale)?;
                let (surface, device_identity) = backend
                    .create_presented_surface(
                        handle.clone(),
                        preferred_device,
                        configuration_operation,
                    )
                    .await?;
                ensure_presented_device_available_after_creation(
                    backend,
                    device_identity,
                    configuration_operation,
                )?;
                let mut created = Surface::with_backend(
                    Attachment::Window(handle),
                    options,
                    SurfaceBackend::Presented {
                        surface: Box::new(surface),
                        device_identity,
                        state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
                    },
                    self.identity.clone(),
                );
                self.configure_presented_surface_if_needed(&mut created, configuration_operation)
                    .await?;
                Ok(created)
            }
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    async fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        let Some(html_canvas) = canvas.html_canvas() else {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                format!("web canvas surface '{}' has no canvas handle", canvas.id()),
            ));
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible WebGPU adapter is available",
            ));
        };
        let physical_size = physical_size(options.size, options.scale)?;
        let (surface, device_identity) = backend
            .create_presented_surface(
                wgpu::SurfaceTarget::Canvas(html_canvas),
                preferred_device,
                configuration_operation,
            )
            .await?;
        ensure_presented_device_available_after_creation(
            backend,
            device_identity,
            configuration_operation,
        )?;
        let mut created = Surface::with_backend(
            Attachment::WebCanvas(canvas),
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                device_identity,
                state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
            },
            self.identity.clone(),
        );
        self.configure_presented_surface_if_needed(&mut created, configuration_operation)
            .await?;
        Ok(created)
    }

    #[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
    async fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        _options: SurfaceOptions,
        _configuration_operation: RuntimeOperation,
        _preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        let _ = canvas;
        Capabilities::CURRENT.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))?;
        unreachable!("web canvas support requires the render-web feature on wasm32");
    }

    /// Creates a headless surface for a later asynchronous render operation.
    ///
    /// Await this operation before using the surface. Input and format failures
    /// are reported when the future is awaited; readback is a separate asynchronous operation.
    pub async fn create_headless(&mut self, size: Size, scale: f64) -> Result<Surface> {
        let options = SurfaceOptions {
            size,
            scale,
            ..SurfaceOptions::default()
        };
        self.create_headless_surface(options).await
    }

    async fn create_headless_surface(&mut self, options: SurfaceOptions) -> Result<Surface> {
        validate_surface_options(options)?;
        if options.format != Format::Rgba8 {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "headless surfaces require Rgba8 format for Vello storage rendering",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = if let (Some(backend), Some(device_identity)) =
            (self.backend.as_mut(), self.default_device)
        {
            if let Some(error) =
                backend.terminal_error(device_identity, RuntimeOperation::AdapterSelection)
            {
                return Err(error);
            }
            SurfaceBackend::Headless {
                device_identity,
                resources: HeadlessResources::for_physical_size(physical_size),
                physical_size,
            }
        } else {
            SurfaceBackend::ContractOnly { physical_size }
        };

        Ok(Surface::with_backend(
            Attachment::Headless,
            options,
            backend,
            self.identity.clone(),
        ))
    }

    pub fn set_surface_resizing(&mut self, surface: &mut Surface, resizing: bool) -> Result<()> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;

        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = resizing;

        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { state, .. } = &mut surface.backend {
            let next = if resizing {
                ResizeState::Resizing
            } else {
                ResizeState::Idle
            };
            if state.lifecycle().resize_state() == next {
                return Ok(());
            }
            state.set_resizing(next);
        }

        Ok(())
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn configure_presented_surface_if_needed(
        &mut self,
        surface: &mut Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let (device_identity, native, requested_physical_size, present_mode, needs_configuration) =
            match &surface.backend {
                SurfaceBackend::Presented {
                    surface: native,
                    device_identity,
                    state,
                } => (
                    *device_identity,
                    native.as_ref(),
                    state.requested_physical_size(),
                    surface.options.present_mode.into(),
                    state.needs_configuration(),
                ),
                SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => {
                    return Ok(());
                }
            };
        if !needs_configuration {
            return Ok(());
        }
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                operation,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            )
        })?;
        let draft = backend
            .configure_presented_surface(
                device_identity,
                operation,
                native,
                requested_physical_size,
                present_mode,
            )
            .await?;
        let publication_signal = backend.publication_signal(device_identity, operation)?;
        #[cfg(test)]
        inject_final_publication_loss_for_test(&publication_signal);
        let result = publication_signal.commit_if_no_terminal(operation, || {
            let SurfaceBackend::Presented { surface, state, .. } = &mut surface.backend else {
                unreachable!("presented configuration must commit into the originating surface");
            };
            surface.commit_configuration(draft);
            state.commit_configuration();
        });
        if let Err(error) = result {
            backend.observe_device_terminal(device_identity);
            return Err(error);
        };
        Ok(())
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) async fn configure_presented_surface_for_test(
        &mut self,
        surface: &mut Surface,
    ) -> Result<()> {
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await
    }

    #[cfg(not(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )))]
    async fn configure_presented_surface_if_needed(
        &mut self,
        _surface: &mut Surface,
        _operation: RuntimeOperation,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn display_free_presented_surface_for_test(
        &mut self,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a host adapter",
            )
        })?;
        self.display_free_presented_surface_on_device_for_test(
            options,
            device_identity,
            Attachment::from_web_canvas("display-free-presented-test-target"),
        )
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn display_free_presented_surface_on_device_for_test(
        &mut self,
        options: SurfaceOptions,
        device_identity: DeviceSlotIdentity,
        attachment: Attachment,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        if !matches!(&attachment, Attachment::WebCanvas(_)) {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "the display-free presented fixture requires a web-canvas attachment",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a renderer backend",
            )
        })?;
        if !backend.has_device_slot(device_identity) {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the display-free presented fixture requires a current device slot",
            ));
        }
        if let Some(error) =
            backend.terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
        {
            return Err(error);
        }
        Ok(Surface::with_backend(
            attachment,
            options,
            SurfaceBackend::Presented {
                surface: Box::new(super::surface::PresentedSurface::display_free_for_test(
                    options.format,
                )),
                device_identity,
                state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
            },
            self.identity.clone(),
        ))
    }

    #[cfg(all(test, feature = "render-window"))]
    async fn create_display_free_presented_surface_with_configuration_operation_for_test(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        if !matches!(&attachment, Attachment::WebCanvas(_)) {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "the display-free presented fixture requires a web-canvas attachment",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a renderer backend",
            )
        })?;
        let (surface, device_identity) = backend
            .create_display_free_presented_surface_for_test(
                preferred_device,
                configuration_operation,
                options.format,
            )
            .await?;
        ensure_presented_device_available_after_creation(
            backend,
            device_identity,
            configuration_operation,
        )?;
        let mut created = Surface::with_backend(
            attachment,
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                device_identity,
                state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
            },
            self.identity.clone(),
        );
        self.configure_presented_surface_if_needed(&mut created, configuration_operation)
            .await?;
        Ok(created)
    }

    fn classify_frame_dispatch(
        &mut self,
        plan: FramePlan,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        #[cfg(test)]
        {
            self.dispatch_observation.boundary_invocations = self
                .dispatch_observation
                .boundary_invocations
                .saturating_add(1);
        }
        match plan {
            FramePlan::DirectVello(plan) => {
                #[cfg(test)]
                {
                    self.dispatch_observation.direct_vello_routes = self
                        .dispatch_observation
                        .direct_vello_routes
                        .saturating_add(1);
                }
                Ok(RendererFrameDispatch::DirectVello(plan.into_commands()))
            }
            FramePlan::GpuGraph(graph) => match ExecutableGraphDispatchEligibility::try_classify(
                &graph,
                output_format,
                working_format,
                capabilities,
            )? {
                ExecutableGraphDispatchEligibility::ExactC08(preparable) => {
                    #[cfg(test)]
                    {
                        self.dispatch_observation.exact_c08_graph_routes = self
                            .dispatch_observation
                            .exact_c08_graph_routes
                            .saturating_add(1);
                    }
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::C08(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::ExactC09(preparable) => {
                    #[cfg(test)]
                    {
                        self.dispatch_observation.exact_c09_graph_routes = self
                            .dispatch_observation
                            .exact_c09_graph_routes
                            .saturating_add(1);
                    }
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::C09(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::ExactC12(preparable) => {
                    #[cfg(test)]
                    {
                        self.dispatch_observation.exact_c12_graph_routes = self
                            .dispatch_observation
                            .exact_c12_graph_routes
                            .saturating_add(1);
                    }
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::C12(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::FuturePasses => {
                    #[cfg(test)]
                    {
                        self.dispatch_observation.future_pass_rejections = self
                            .dispatch_observation
                            .future_pass_rejections
                            .saturating_add(1);
                    }
                    reject_future_graph_with_typed_diagnostic(&graph)?;
                    Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "a future GPU graph had no unavailable execution pass",
                    ))
                }
            },
        }
    }

    #[cfg(test)]
    fn classify_c10_fixture_dispatch(
        &mut self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        self.dispatch_observation.boundary_invocations = self
            .dispatch_observation
            .boundary_invocations
            .saturating_add(1);
        let preparable = super::pass::c10_preparable_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        self.dispatch_observation.exact_c10_fixture_routes = self
            .dispatch_observation
            .exact_c10_fixture_routes
            .saturating_add(1);
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::C10(preparable),
        )))
    }

    #[cfg(test)]
    fn classify_c11_fixture_dispatch(
        &mut self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        self.dispatch_observation.boundary_invocations = self
            .dispatch_observation
            .boundary_invocations
            .saturating_add(1);
        let preparable = super::pass::c11_preparable_graph_from_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        self.dispatch_observation.exact_c11_fixture_routes = self
            .dispatch_observation
            .exact_c11_fixture_routes
            .saturating_add(1);
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::C11(preparable),
        )))
    }

    #[cfg(test)]
    fn classify_c12_fixture_dispatch(
        &mut self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        self.dispatch_observation.boundary_invocations = self
            .dispatch_observation
            .boundary_invocations
            .saturating_add(1);
        let preparable = super::pass::c12_preparable_graph_from_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        self.dispatch_observation.exact_c12_fixture_routes = self
            .dispatch_observation
            .exact_c12_fixture_routes
            .saturating_add(1);
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::C12(preparable),
        )))
    }

    /// Submits one render operation for an available surface.
    ///
    /// Awaiting this future returns render statistics after scene validation and
    /// submission, or the existing lifecycle, validation, or backend diagnostic.
    pub async fn render(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<Stats> {
        let device_identity = self.render_device_identity(surface)?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let (normalized, execution) =
            self.prepare_render_execution(surface, scene, parameters, device_identity)?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: Duration::ZERO,
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        if matches!(&execution, PreparedRendererExecution::DirectVello(_)) {
            stats.route = Some(RenderRoute::DirectVello);
            stats.vello_passes = stats.vello_passes.saturating_add(1);
        }
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
        stats.encode_time = encode_start.elapsed();
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        let frame = self
            .execute_render_frame(surface, execution, parameters, device_identity)
            .await?;
        self.publish_clean_render_frame(
            surface,
            device_identity,
            RenderPublication {
                frame,
                stats,
                uploaded_images,
                parameters,
            },
            frame_start,
        )
    }

    fn render_device_identity(&mut self, surface: &Surface) -> Result<DeviceSlotIdentity> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;

        let Some(device_identity) = surface.device_identity() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        if self.backend.is_none() {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        }
        Ok(device_identity)
    }

    fn prepare_render_execution(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        device_identity: DeviceSlotIdentity,
    ) -> Result<(RenderCommands, PreparedRendererExecution)> {
        #[cfg(test)]
        {
            self.preexecution_frame_gate_observation =
                PreexecutionFrameGateObservationForTest::default();
        }
        let normalized = scene.normalize(self.capabilities())?;
        let graph_source = normalized.clone();
        let frame_context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let frame_plan = normalized.plan_for(frame_context)?;
        #[cfg(test)]
        {
            self.preexecution_frame_gate_observation
                .validated_plan_count += 1;
        }
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the renderer dispatch boundary lost immutable device capabilities",
                )
            })?;
        #[cfg(test)]
        let working_format = match self.exact_graph_working_format {
            Some(working_format) => ExecutableGraphWorkingFormatRequest::Exact(working_format),
            None => ExecutableGraphWorkingFormatRequest::ConfiguredPolicy(
                self.options.effect_quality_policy(),
            ),
        };
        #[cfg(not(test))]
        let working_format = ExecutableGraphWorkingFormatRequest::ConfiguredPolicy(
            self.options.effect_quality_policy(),
        );
        let dispatch = self.classify_frame_dispatch(
            frame_plan,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )?;
        Ok(match dispatch {
            RendererFrameDispatch::DirectVello(normalized) => {
                let vello_scene = encode_vello_scene(&normalized, surface.scale())?;
                (
                    normalized,
                    PreparedRendererExecution::DirectVello(Box::new(vello_scene)),
                )
            }
            RendererFrameDispatch::ExactGraph(graph) => {
                (graph_source, PreparedRendererExecution::ExactGraph(graph))
            }
        })
    }

    async fn execute_render_frame(
        &mut self,
        surface: &mut Surface,
        execution: PreparedRendererExecution,
        parameters: Parameters,
        device_identity: DeviceSlotIdentity,
    ) -> Result<SurfaceFrameCommit> {
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("surface preflight confirmed the renderer backend is available");
            match execution {
                PreparedRendererExecution::DirectVello(vello_scene) => {
                    let transaction = backend.begin_gpu_operation(
                        device_identity,
                        GpuOperationStage::Render,
                        RuntimeOperation::SurfaceRendering,
                    )?;
                    render_internal_vello_surface(
                        backend,
                        transaction,
                        surface,
                        &vello_scene,
                        parameters,
                        self.options.antialiasing(),
                    )
                    .await
                }
                PreparedRendererExecution::ExactGraph(graph) => {
                    #[cfg(any(
                        feature = "render-window",
                        all(feature = "render-web", target_arch = "wasm32")
                    ))]
                    {
                        if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                            render_exact_presented_graph_surface(backend, surface, *graph).await
                        } else {
                            render_exact_headless_graph_surface(backend, surface, *graph).await
                        }
                    }
                    #[cfg(not(any(
                        feature = "render-window",
                        all(feature = "render-web", target_arch = "wasm32")
                    )))]
                    {
                        render_exact_headless_graph_surface(backend, surface, *graph).await
                    }
                }
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(device_identity);
        }
        let frame = match frame {
            Err(error) if error.code() == ErrorCode::SurfaceOutdated => {
                self.configure_presented_surface_if_needed(
                    surface,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                return Err(error);
            }
            Err(error) => return Err(error),
            Ok(frame) => frame,
        };
        Ok(frame)
    }

    fn publish_clean_render_frame(
        &mut self,
        surface: &mut Surface,
        device_identity: DeviceSlotIdentity,
        mut publication: RenderPublication,
        frame_start: Instant,
    ) -> Result<Stats> {
        let timings = publication.frame.timings();
        publication.stats.render_time = timings.render_time;
        publication.stats.present_time = timings.present_time;
        publication.stats.frame_time = frame_start.elapsed();
        let mut published = None;
        GpuOperationDraft::new(&mut published, publication).commit();
        let publication =
            published.expect("a clean GPU transaction must commit its staged public state");
        #[cfg(test)]
        {
            let Some(publication_signal) = self
                .backend
                .as_mut()
                .and_then(|backend| backend.device_signal_for_test(device_identity))
            else {
                panic!("a clean frame must retain its device signal until publication");
            };
            inject_final_publication_loss_for_test(&publication_signal);
        }
        let stats = publication.commit(self, surface);
        if let Some(backend) = self.backend.as_mut() {
            backend.observe_device_terminal(device_identity);
        }
        Ok(stats)
    }

    /// Private T5 entry for forcing ordinary commands through the exact
    /// production C08 graph executor without adding a public route or option.
    #[cfg(test)]
    pub(crate) async fn render_forced_c08_graph_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C08ForcedGraphRenderResultForTest> {
        self.render_forced_c08_graph_with_capture_mapping_for_test(
            surface,
            scene,
            parameters,
            working_format,
            super::frame::ForcedC08CaptureMappingForTest::identity(),
        )
        .await
    }

    /// Private T6 entry that keeps capture and parent mappings distinct while
    /// executing the same production C08 graph path.
    #[cfg(test)]
    pub(crate) async fn render_forced_c08_graph_with_capture_mapping_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
        capture_mapping: super::frame::ForcedC08CaptureMappingForTest,
    ) -> Result<C08ForcedGraphRenderResultForTest> {
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let ForcedC08PreparationForTest {
            device_identity,
            normalized,
            preparable,
            output_extent,
            captures,
        } = self.prepare_forced_c08_graph_for_test(
            surface,
            scene,
            parameters,
            working_format,
            capture_mapping,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let (stats, uploaded_images) =
            self.forced_c08_stats_for_test(&normalized, parameters, encode_start);
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("forced C08 preflight confirmed the renderer backend is available");
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(
                        backend,
                        surface,
                        ExactSurfaceGraph::C08(preparable),
                    )
                    .await
                } else {
                    render_exact_headless_graph_surface(
                        backend,
                        surface,
                        ExactSurfaceGraph::C08(preparable),
                    )
                    .await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(
                    backend,
                    surface,
                    ExactSurfaceGraph::C08(preparable),
                )
                .await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(device_identity);
        }
        let frame = match frame {
            Err(error) if error.code() == ErrorCode::SurfaceOutdated => {
                self.configure_presented_surface_if_needed(
                    surface,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                return Err(error);
            }
            Err(error) => return Err(error),
            Ok(frame) => frame,
        };
        let stats = self.publish_clean_render_frame(
            surface,
            device_identity,
            RenderPublication {
                frame,
                stats,
                uploaded_images,
                parameters,
            },
            frame_start,
        )?;
        Ok(C08ForcedGraphRenderResultForTest {
            stats,
            working_format,
            output_extent,
            captures,
        })
    }

    #[cfg(test)]
    fn forced_c08_stats_for_test(
        &self,
        normalized: &RenderCommands,
        parameters: Parameters,
        encode_start: Instant,
    ) -> (Stats, HashSet<ImageId>) {
        let mut stats = Stats {
            encode_time: encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        (stats, uploaded_images)
    }

    #[cfg(test)]
    fn prepare_forced_c08_graph_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
        capture_mapping: super::frame::ForcedC08CaptureMappingForTest,
    ) -> Result<ForcedC08PreparationForTest> {
        let device_identity = self.validate_forced_c08_surface_for_test(surface)?;
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph = super::frame::forced_c08_graph_with_capture_mapping_for_test(
            normalized.clone(),
            context,
            capture_mapping,
        )?;
        let captures = graph.forced_capture_observations_for_test();
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C08 forced route lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_frame_dispatch(
            FramePlan::GpuGraph(graph),
            runtime_surface_format(surface),
            ExecutableGraphWorkingFormatRequest::Exact(working_format),
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::C08(preparable) => preparable,
                ExactSurfaceGraph::C09(_)
                | ExactSurfaceGraph::C10(_)
                | ExactSurfaceGraph::C11(_)
                | ExactSurfaceGraph::C12(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private forced graph is outside the exact executable C08 subset",
                    ));
                }
            },
            _ => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private forced graph is outside the exact executable C08 subset",
                ));
            }
        };
        let output_extent = preparable.output_extent()?;
        let prepared_grids = preparable.capture_grids_for_test();
        if captures.len() != prepared_grids.len()
            || captures
                .iter()
                .zip(&prepared_grids)
                .any(|(capture, prepared)| {
                    capture.texel_origin != prepared.texel_origin
                        || capture.extent != prepared.extent
                        || capture.raster_scale != prepared.raster_scale
                })
        {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the prepared C08 capture grid differs from the validated semantic graph",
            ));
        }
        Ok(ForcedC08PreparationForTest {
            device_identity,
            normalized,
            preparable,
            output_extent,
            captures: captures
                .into_iter()
                .map(C08ForcedGraphCaptureForTest::from)
                .collect(),
        })
    }

    #[cfg(test)]
    fn validate_forced_c08_surface_for_test(
        &mut self,
        surface: &Surface,
    ) -> Result<DeviceSlotIdentity> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;
        surface.device_identity().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the private C08 forced route requires a device-backed surface",
            )
        })
    }

    /// Private C10 authored-filter ingress into the shared exact graph executor.
    #[cfg(test)]
    pub(crate) async fn render_c10_color_filter_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C10ColorFilterRenderResultForTest> {
        let prepared = self.prepare_c10_color_filter_fixture_for_test(
            surface,
            scene,
            filters,
            parameters,
            working_format,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("C10 fixture preflight confirmed the renderer backend is available");
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(backend, surface, prepared.graph).await
                } else {
                    render_exact_headless_graph_surface(backend, surface, prepared.graph).await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(backend, surface, prepared.graph).await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = match frame {
            Err(error) if error.code() == ErrorCode::SurfaceOutdated => {
                self.configure_presented_surface_if_needed(
                    surface,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                return Err(error);
            }
            Err(error) => return Err(error),
            Ok(frame) => frame,
        };
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication {
                frame,
                stats,
                uploaded_images,
                parameters,
            },
            prepared.frame_start,
        )?;
        Ok(C10ColorFilterRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            source_origin: prepared.source_spatial.device_origin,
            source_extent: prepared.source_spatial.device_extent,
            source_texel_origin: prepared.source_spatial.texel_origin,
            source_raster_scale: prepared.source_spatial.raster_scale,
        })
    }

    #[cfg(test)]
    fn prepare_c10_color_filter_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C10ColorFilterFixturePreparationForTest> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;
        let device_identity = surface.device_identity().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the private C10 fixture requires a device-backed surface",
            )
        })?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph =
            super::frame::authored_c10_color_graph_for_test(filters, normalized.clone(), context)?;
        let capabilities = self
            .backend
            .as_mut()
            .ok_or_else(|| {
                Error::runtime_unavailable(
                    RuntimeOperation::SurfaceRendering,
                    RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                    "the private C10 fixture requires a renderer backend",
                )
            })?
            .device_capabilities(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C10 fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_c10_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::C10(preparable) => preparable,
                ExactSurfaceGraph::C08(_)
                | ExactSurfaceGraph::C09(_)
                | ExactSurfaceGraph::C11(_)
                | ExactSurfaceGraph::C12(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private C10 fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C10 fixture left its exact renderer dispatch route",
                ));
            }
        };
        let output_extent = preparable.output_extent()?;
        let source_spatial = preparable.first_color_spatial_for_test().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the private C10 fixture lost its first exact color source",
            )
        })?;
        Ok(C10ColorFilterFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::C10(preparable),
            output_extent,
            source_spatial,
        })
    }

    /// Private C11 authored-filter ingress into the shared exact graph executor.
    #[cfg(test)]
    pub(crate) async fn render_c11_spatial_filter_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C11SpatialFilterRenderResultForTest> {
        let prepared = self.prepare_c11_spatial_filter_fixture_for_test(
            surface,
            scene,
            filters,
            parameters,
            working_format,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("C11 fixture preflight confirmed the renderer backend is available");
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(backend, surface, prepared.graph).await
                } else {
                    render_exact_headless_graph_surface(backend, surface, prepared.graph).await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(backend, surface, prepared.graph).await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = frame?;
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication {
                frame,
                stats,
                uploaded_images,
                parameters,
            },
            prepared.frame_start,
        )?;
        Ok(C11SpatialFilterRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            source_spatial: prepared.source_spatial,
            result_spatial: prepared.result_spatial,
        })
    }

    #[cfg(test)]
    fn prepare_c11_spatial_filter_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C11SpatialFilterFixturePreparationForTest> {
        let device_identity = self.validate_forced_c08_surface_for_test(surface)?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph =
            super::frame::authored_c10_color_graph_for_test(filters, normalized.clone(), context)?;
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C11 fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_c11_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::C11(preparable) => preparable,
                ExactSurfaceGraph::C08(_)
                | ExactSurfaceGraph::C09(_)
                | ExactSurfaceGraph::C10(_)
                | ExactSurfaceGraph::C12(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private C11 fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C11 fixture left its exact renderer dispatch route",
                ));
            }
        };
        let output_extent = preparable.output_extent()?;
        let (source_spatial, result_spatial) =
            preparable.first_filter_spatial_for_test().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C11 fixture lost its first spatial filter mapping",
                )
            })?;
        Ok(C11SpatialFilterFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::C11(preparable),
            output_extent,
            source_spatial,
            result_spatial,
        })
    }

    /// Private C12 bounded-backdrop ingress into the shared exact graph executor.
    #[cfg(test)]
    pub(crate) async fn render_c12_backdrop_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C12BackdropRenderResultForTest> {
        let prepared =
            self.prepare_c12_backdrop_fixture_for_test(surface, scene, parameters, working_format)?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("C12 fixture preflight confirmed the renderer backend is available");
            render_exact_headless_graph_surface(backend, surface, prepared.graph).await
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = frame?;
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication {
                frame,
                stats,
                uploaded_images,
                parameters,
            },
            prepared.frame_start,
        )?;
        Ok(C12BackdropRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            parent_spatial: prepared.parent_spatial,
            capture_spatial: prepared.capture_spatial,
        })
    }

    #[cfg(test)]
    fn prepare_c12_backdrop_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<C12BackdropFixturePreparationForTest> {
        let device_identity = self.validate_forced_c08_surface_for_test(surface)?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let FramePlan::GpuGraph(graph) = normalized.clone().plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the private C12 fixture did not produce a bounded backdrop graph",
            ));
        };
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C12 fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_c12_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::C12(preparable) => preparable,
                _ => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private C12 fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C12 fixture left its exact renderer dispatch route",
                ));
            }
        };
        let output_extent = preparable.output_extent()?;
        let (parent_spatial, capture_spatial) =
            preparable.backdrop_spatial_for_test().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private C12 fixture lost its exact backdrop mapping",
                )
            })?;
        Ok(C12BackdropFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::C12(preparable),
            output_extent,
            parent_spatial,
            capture_spatial,
        })
    }

    /// Resumes a compatible surface, awaiting recreation when it is presented.
    ///
    /// Await this operation before rendering again. Incompatible attachments and
    /// identity failures preserve their existing error ordering.
    pub async fn resume_surface(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
    ) -> Result<()> {
        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = attachment;
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceResume)?;

        match &surface.backend {
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { state, .. } => {
                let action = Surface::presented_resume_action(surface.state, state.lifecycle());
                let resizing = state.lifecycle().resize_state();
                self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceResume)?;
                surface.ensure_attachment_compatible(&attachment)?;
                match action {
                    super::surface::PresentedResumeAction::NoOp => Ok(()),
                    super::surface::PresentedResumeAction::ConfigureExisting => {
                        self.configure_presented_surface_if_needed(
                            surface,
                            RuntimeOperation::SurfaceResume,
                        )
                        .await
                    }
                    super::surface::PresentedResumeAction::Configure => {
                        self.recreate_presented_surface_for_resume(
                            surface, attachment, resizing, true,
                        )
                        .await
                    }
                    super::surface::PresentedResumeAction::Recreate => {
                        self.recreate_presented_surface_for_resume(
                            surface, attachment, resizing, false,
                        )
                        .await
                    }
                }
            }
            SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => unreachable!(),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn recreate_presented_surface_for_resume(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
        resizing: ResizeState,
        preserve_renderer_identity: bool,
    ) -> Result<()> {
        let preferred_device = surface
            .device_identity()
            .expect("a presented surface must retain its device slot identity");
        #[cfg(all(test, feature = "render-window"))]
        if surface.is_display_free_presented_for_test() {
            let mut next = self
                .create_display_free_presented_surface_with_configuration_operation_for_test(
                    attachment,
                    surface.options,
                    RuntimeOperation::SurfaceResume,
                    Some(preferred_device),
                )
                .await?;
            next.last_parameters = surface.last_parameters;
            next.renderer_identity = surface.renderer_identity.clone();
            if let SurfaceBackend::Presented { state, .. } = &mut next.backend {
                state.set_resizing(resizing);
            }
            *surface = next;
            return Ok(());
        }
        let mut next = self
            .create_surface_with_configuration_operation(
                attachment,
                surface.options,
                RuntimeOperation::SurfaceResume,
                Some(preferred_device),
            )
            .await?;
        next.last_parameters = surface.last_parameters;
        if preserve_renderer_identity {
            next.renderer_identity = surface.renderer_identity.clone();
        }
        if let SurfaceBackend::Presented { state, .. } = &mut next.backend {
            state.set_resizing(resizing);
        }
        *surface = next;
        Ok(())
    }

    /// Reads the current complete publication from a headless surface.
    ///
    /// A zero-area available surface returns an empty validated image without GPU work. A
    /// nonzero surface without a published frame returns its typed uninitialized diagnostic.
    /// The returned future is not promised to be `Send`.
    pub async fn read_headless(&mut self, surface: &Surface) -> Result<ImageBuffer> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceReadback)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceReadback)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceReadback)?;
        surface.ensure_available(RuntimeOperation::SurfaceReadback)?;
        let (device_identity, texture, physical_size) = match &surface.backend {
            SurfaceBackend::ContractOnly { physical_size }
                if physical_size.width() == 0 || physical_size.height() == 0 =>
            {
                return ImageBuffer::try_new(*physical_size, Vec::new());
            }
            SurfaceBackend::ContractOnly { .. } => {
                return Err(Error::runtime_unavailable(
                    RuntimeOperation::SurfaceReadback,
                    RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                    "no compatible wgpu adapter is available",
                ));
            }
            SurfaceBackend::Headless {
                physical_size,
                resources: HeadlessResources::Empty,
                ..
            } => {
                return ImageBuffer::try_new(*physical_size, Vec::new());
            }
            SurfaceBackend::Headless {
                resources: HeadlessResources::Pending,
                ..
            } => {
                return Err(Error::runtime_unavailable(
                    RuntimeOperation::SurfaceReadback,
                    RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                        state: RenderSurfaceAvailability::Uninitialized,
                    },
                    "headless surface has no published texture",
                ));
            }
            SurfaceBackend::Headless {
                device_identity,
                resources: HeadlessResources::Ready { texture, .. },
                physical_size,
            } => (*device_identity, texture, *physical_size),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => unreachable!(),
        };
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceReadback)?;
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        read_texture_rgba(
            backend,
            device_identity,
            texture,
            physical_size,
            RuntimeOperation::SurfaceReadback,
        )
        .await
    }

    /// Projects immutable capabilities of the device selected by `surface`.
    ///
    /// This query observes pending terminal device signals but performs no
    /// allocation, submission, mapping, polling, or Vello/WGPU resource call.
    #[must_use]
    pub fn runtime_capabilities(&mut self, surface: &Surface) -> RuntimeCapabilities {
        if !self.identity.matches(&surface.renderer_identity) {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::ForeignRenderer,
                },
            );
        }
        let Some(device_identity) = surface.device_identity() else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            );
        };
        let Some(backend) = self.backend.as_mut() else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        };
        if !backend.has_device_slot(device_identity) {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        }
        if let Some(reason) = backend.terminal_reason(device_identity) {
            return RuntimeCapabilities::Unavailable(reason);
        }
        if let Some(reason) = runtime_surface_unavailable_reason(surface) {
            return RuntimeCapabilities::Unavailable(reason);
        }
        let Some(capabilities) = backend.device_capabilities(device_identity) else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        };
        RuntimeCapabilities::Available(capabilities.runtime_report(runtime_surface_format(surface)))
    }

    #[cfg(test)]
    pub(crate) fn default_wgpu_device_queue(&mut self) -> Option<(&wgpu::Device, &wgpu::Queue)> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        backend
            .device_queue(device_identity, RuntimeOperation::SurfaceRendering)
            .ok()
    }

    #[cfg(test)]
    pub(crate) fn default_offscreen_render_context(
        &mut self,
    ) -> Option<OffscreenRenderGpuContext<'_>> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        if backend
            .terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
            .is_some()
        {
            return None;
        }
        Some(OffscreenRenderGpuContext::new(backend, device_identity))
    }

    #[cfg(test)]
    pub(crate) async fn read_render_texture_for_test(
        &mut self,
        texture: &wgpu::Texture,
        physical_size: PhysicalSize,
    ) -> Result<ImageBuffer> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "required render-texture readback needs an available wgpu device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "required render-texture readback needs an available wgpu backend",
            )
        })?;
        read_texture_rgba(
            backend,
            device_identity,
            texture,
            physical_size,
            RuntimeOperation::SurfaceRendering,
        )
        .await
    }

    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    #[cfg(test)]
    pub(crate) const fn preexecution_frame_gate_observation_for_test(
        &self,
    ) -> PreexecutionFrameGateObservationForTest {
        self.preexecution_frame_gate_observation
    }

    #[cfg(test)]
    pub(crate) const fn dispatch_observation_for_test(&self) -> RendererDispatchObservationForTest {
        self.dispatch_observation
    }

    #[cfg(test)]
    pub(crate) fn select_exact_graph_working_format_for_test(
        &mut self,
        working_format: WorkingFormat,
    ) {
        self.exact_graph_working_format = Some(working_format);
    }

    #[must_use]
    pub const fn options(&self) -> Options {
        self.options
    }

    fn validate_surface_renderer_identity(
        &self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        if self.identity.matches(&surface.renderer_identity) {
            return Ok(());
        }
        Err(surface_identity_mismatch(
            operation,
            SurfaceIdentityMismatchKind::ForeignRenderer,
        ))
    }

    fn validate_surface_device_identity(
        &mut self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(device_identity) = surface.device_identity() else {
            return Ok(());
        };
        if self
            .backend
            .as_mut()
            .is_some_and(|backend| backend.has_device_slot(device_identity))
        {
            return Ok(());
        }
        Err(surface_identity_mismatch(
            operation,
            SurfaceIdentityMismatchKind::StaleDeviceGeneration,
        ))
    }

    fn validate_surface_operation_backend(
        &self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let supported = match operation {
            RuntimeOperation::SurfaceReadback => matches!(
                surface.backend,
                SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. }
            ),
            RuntimeOperation::SurfaceResume => {
                #[cfg(any(
                    feature = "render-window",
                    all(feature = "render-web", target_arch = "wasm32")
                ))]
                {
                    matches!(surface.backend, SurfaceBackend::Presented { .. })
                }
                #[cfg(not(any(
                    feature = "render-window",
                    all(feature = "render-web", target_arch = "wasm32")
                )))]
                {
                    false
                }
            }
            _ => true,
        };
        if supported {
            Ok(())
        } else {
            Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "surface backend does not support this operation",
            ))
        }
    }

    fn validate_surface_device_terminal(
        &mut self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(device_identity) = surface.device_identity() else {
            return Ok(());
        };
        if let Some(error) = self
            .backend
            .as_mut()
            .and_then(|backend| backend.terminal_error(device_identity, operation))
        {
            return Err(error);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn signal_default_device_loss_for_test(&mut self, reason: DeviceLossReason) {
        if let Some(device_identity) = self.default_device {
            self.signal_device_loss_for_test(device_identity, reason);
        }
    }

    #[cfg(test)]
    pub(crate) fn signal_device_loss_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
        reason: DeviceLossReason,
    ) {
        if let Some(backend) = self.backend.as_mut() {
            backend.signal_loss_for_test(device_identity, reason);
        }
    }

    #[cfg(test)]
    pub(crate) fn device_signal_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> Option<Arc<DeviceSignal>> {
        self.backend
            .as_mut()?
            .device_signal_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn default_device_signal_for_test(&mut self) -> Option<Arc<DeviceSignal>> {
        self.device_signal_for_test(self.default_device?)
    }

    #[cfg(test)]
    pub(crate) fn signal_device_uncaptured_fault_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
        kind: GpuFaultKind,
    ) {
        if let Some(backend) = self.backend.as_mut() {
            backend.signal_uncaptured_fault_for_test(device_identity, kind);
        }
    }

    #[cfg(test)]
    pub(crate) fn default_device_renderer_released_for_test(&mut self) -> bool {
        match (self.backend.as_mut(), self.default_device) {
            (Some(backend), Some(device_identity)) => {
                backend.renderer_released_for_test(device_identity)
            }
            (None, None) => true,
            _ => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn device_renderer_released_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> bool {
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.renderer_released_for_test(device_identity))
    }

    #[cfg(test)]
    pub(crate) fn default_ready_device_state_borrow_for_test(
        &mut self,
    ) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .ready_device_state_borrow_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn resource_preparation_observation_for_test(
        &mut self,
        commands: RenderCommands,
        surface_size: Size,
        surface_scale: f64,
        base_color: Color,
        output_format: Format,
    ) -> Result<ResourcePreparationObservationForTest> {
        let context = FrameContext::try_new(
            surface_size,
            surface_scale,
            self.options.antialiasing(),
            base_color,
        )?;
        let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the resource preparation fixture did not produce a GPU graph",
            ));
        };
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::EffectRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "resource preparation coverage requires a ready default device",
            )
        })?;
        let policy = self.options.effect_quality_policy();
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::EffectRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "resource preparation coverage requires a renderer backend",
            )
        })?;
        let capabilities = backend
            .device_capabilities(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "resource preparation coverage requires immutable device capabilities",
                )
            })?;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let lowered = super::pass::LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            output_format,
            &capabilities,
        )?;
        let pass_cache_before = backend
            .seed_device_pass_cache_sampler_for_test(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "ready device disappeared before pass-cache preservation coverage",
                )
            })?;

        let allocation_preflight_is_atomic =
            preparation_preflight_is_atomic(backend, device_identity, &lowered, policy)?;
        let (first_exercise, repeated_reuse_is_exact_and_bounded) =
            exercise_preparation_reuse(backend, device_identity, &lowered, policy)?;
        let failure_and_drop_cleanup =
            preparation_failure_cleanup(backend, device_identity, lowered, policy)?;
        let pass_cache_after = backend
            .ready_device_state_borrow_for_test(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "ready device disappeared after pass-cache preservation coverage",
                )
            })?
            .device_pass_cache_counts_for_test();
        let populated_pass_cache_is_preserved =
            pass_cache_before.has_exactly_one_sampler() && pass_cache_after == pass_cache_before;

        Ok(ResourcePreparationObservationForTest {
            complete_resource_and_pass_handoff: first_exercise.complete_resource_and_pass_handoff,
            exact_capture_coverage_working_and_mask_allocations: first_exercise
                .exact_capture_coverage_working_and_mask_allocations,
            typed_bindings_and_last_use_releases: first_exercise
                .typed_bindings_and_last_use_releases,
            spatial_bytes_and_cache_keys_preserved: first_exercise
                .spatial_bytes_and_cache_keys_preserved,
            allocation_preflight_is_atomic,
            failure_and_drop_cleanup,
            repeated_reuse_is_exact_and_bounded,
            populated_pass_cache_is_preserved,
        })
    }

    #[cfg(test)]
    pub(crate) async fn deliberate_validation_error_for_test(&mut self) -> Result<Result<()>> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (device, _) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
        let _ = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist deliberate scoped validation failure"),
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
        let result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
        backend.observe_device_terminal(device_identity);
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) async fn scoped_clear_fill_probe_for_test(&mut self) -> Result<ImageBuffer> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (result, destination_texture) = {
            use super::texture::{TextureDescriptor, TextureUsageIntent};

            let destination = TextureDescriptor::try_new(
                PhysicalSize::new(2, 2),
                Format::Rgba8,
                TextureUsageIntent::IntermediatePass,
            )?;
            let (device, queue) =
                backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
            let (destination_texture, destination_view) =
                create_texture(device, "Surgeist scoped clear destination", destination);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist scoped test-only clear encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Surgeist scoped test-only clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &destination_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.25,
                                g: 0.5,
                                b: 0.75,
                                a: 1.0,
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
            (
                transaction
                    .submit_command_buffer(
                        queue,
                        encoder.finish(),
                        RuntimeOperation::SurfaceRendering,
                    )
                    .await,
                destination_texture,
            )
        };
        backend.observe_device_terminal(device_identity);
        result?;
        read_texture_rgba(
            backend,
            device_identity,
            &destination_texture,
            PhysicalSize::new(2, 2),
            RuntimeOperation::SurfaceRendering,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn submit_prepared_vello_pass_for_test(
        &mut self,
        prepared: &super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<super::gpu_transaction::InternalVelloSubmissionObservationForTest> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "T6 transaction coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "T6 transaction coverage requires a renderer backend",
            )
        })?;
        backend
            .submit_prepared_vello_pass_for_test(device_identity, prepared, target_extent)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn cancel_prepared_vello_pass_after_submit_for_test(
        &mut self,
        prepared: &super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<super::resource::ResourceManagerObservationForTest> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "T6 cancellation coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "T6 cancellation coverage requires a renderer backend",
            )
        })?;
        backend
            .cancel_prepared_vello_pass_after_submit_for_test(
                device_identity,
                prepared,
                target_extent,
            )
            .await
    }

    #[cfg(test)]
    pub(crate) fn default_device_active_operation_generation_for_test(&mut self) -> Option<u64> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .active_operation_generation_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn uploaded_images_for_test(&self) -> HashSet<ImageId> {
        self.uploaded_images.clone()
    }

    #[cfg(test)]
    pub(crate) fn default_device_has_no_terminal_signal_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return true;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.terminal_reason(device_identity).is_none())
    }

    #[cfg(test)]
    pub(crate) fn default_device_capabilities_for_test(&mut self) -> AvailableRuntimeCapabilities {
        let device_identity = self.default_device.expect("test requires a default device");
        self.backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .expect("test requires a ready default device")
            .runtime_report(Format::Rgba8)
    }

    #[cfg(test)]
    pub(crate) fn override_default_device_effect_precision_facts_for_test(
        &mut self,
        effect_precisions: EffectPrecisionCapabilities,
    ) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        self.backend.as_mut().is_some_and(|backend| {
            backend
                .override_device_effect_precision_facts_for_test(device_identity, effect_precisions)
        })
    }

    #[cfg(test)]
    pub(crate) fn destroy_default_device_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        let Some(backend) = self.backend.as_mut() else {
            return false;
        };
        backend.destroy_device_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn wait_for_default_terminal_signal_for_test(&mut self, timeout: Duration) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.wait_for_terminal_for_test(device_identity, timeout))
    }

    #[cfg(test)]
    pub(crate) async fn add_donor_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the renderer has no backend to receive a donor wgpu device",
            )
        })?;
        backend.add_device_slot_for_test().await
    }

    #[cfg(test)]
    pub(crate) async fn submit_scoped_wgpu_probe_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> Result<()> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real second-slot WGPU coverage requires a renderer backend",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (device, queue) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
        let command_buffer = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Surgeist second-slot terminal test target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist second-slot terminal test encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Surgeist second-slot terminal test pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
            Ok(encoder.finish())
        };
        let scope_result = match command_buffer {
            Ok(command_buffer) => {
                transaction
                    .submit_command_buffer(
                        queue,
                        command_buffer,
                        RuntimeOperation::SurfaceRendering,
                    )
                    .await
            }
            Err(error) => match transaction.finish(RuntimeOperation::SurfaceRendering).await {
                Ok(()) => Err(error),
                Err(scope_error) => Err(scope_error),
            },
        };
        backend.observe_device_terminal(device_identity);
        scope_result
    }

    #[must_use]
    pub const fn capabilities(&self) -> Capabilities {
        Capabilities::CURRENT
    }
}

fn runtime_surface_format(surface: &Surface) -> Format {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let SurfaceBackend::Presented {
        surface: native, ..
    } = &surface.backend
    {
        return match native.format {
            wgpu::TextureFormat::Rgba8Unorm => Format::Rgba8,
            wgpu::TextureFormat::Bgra8Unorm => Format::Bgra8,
            _ => surface.options.format,
        };
    }
    surface.options.format
}

fn runtime_surface_unavailable_reason(
    _surface: &Surface,
) -> Option<RuntimeCapabilityUnavailableReason> {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let SurfaceBackend::Presented { state, .. } = &_surface.backend {
        let state = match state.lifecycle() {
            PresentedLifecycle::NonRenderable { .. } => RenderSurfaceAvailability::NonRenderable,
            PresentedLifecycle::Occluded { .. } => RenderSurfaceAvailability::Occluded,
            PresentedLifecycle::Lost => RenderSurfaceAvailability::Lost,
            PresentedLifecycle::Ready { .. } | PresentedLifecycle::ResizePending { .. } => {
                return None;
            }
        };
        return Some(RuntimeCapabilityUnavailableReason::SurfaceUnavailable { state });
    }
    None
}

fn surface_identity_mismatch(
    operation: RuntimeOperation,
    kind: SurfaceIdentityMismatchKind,
) -> Error {
    let diagnostic = RuntimeCapabilityUnavailable::try_new(
        operation,
        RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { kind },
    )
    .expect("surface identity mismatch is valid for every surface operation");
    Error::runtime_capability_unavailable(diagnostic)
}

fn reject_future_graph_with_typed_diagnostic(graph: &GpuRenderGraph) -> Result<()> {
    let mut has_copy_backdrop = false;
    let mut has_color_filter = false;
    let mut has_blur = false;
    let mut has_drop_shadow = false;

    for pass in graph.lowering_view()?.passes() {
        match pass.kind()? {
            GraphLoweringPassKind::ClearRoot { .. }
            | GraphLoweringPassKind::VelloCapture(Some(_))
            | GraphLoweringPassKind::CanonicalizeCapture
            | GraphLoweringPassKind::Present => {}
            GraphLoweringPassKind::CopyBackdrop => has_copy_backdrop = true,
            GraphLoweringPassKind::ColorFilter(Some(_)) => has_color_filter = true,
            GraphLoweringPassKind::BlurHorizontal(Some(_))
            | GraphLoweringPassKind::BlurVertical(Some(_)) => has_blur = true,
            GraphLoweringPassKind::DropShadowColorize(Some(_)) => has_drop_shadow = true,
            GraphLoweringPassKind::Composite(Some(composite)) => match composite.kind() {
                GraphLoweringCompositeKind::SpanSourceOver => {}
                GraphLoweringCompositeKind::Layer { .. } => {}
                GraphLoweringCompositeKind::DropShadow => has_drop_shadow = true,
            },
            GraphLoweringPassKind::VelloCapture(None)
            | GraphLoweringPassKind::ColorFilter(None)
            | GraphLoweringPassKind::BlurHorizontal(None)
            | GraphLoweringPassKind::BlurVertical(None)
            | GraphLoweringPassKind::DropShadowColorize(None)
            | GraphLoweringPassKind::Composite(None) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "a malformed GPU graph reached production dispatch",
                ));
            }
        }
    }

    let unsupported = if has_copy_backdrop {
        Some((
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    } else if has_drop_shadow {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ))
    } else if has_blur {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ))
    } else if has_color_filter {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ))
    } else {
        None
    };
    if let Some((family, operation)) = unsupported {
        return Err(Error::unsupported_render_primitive(
            UnsupportedPrimitive::new(family, operation),
        ));
    }
    Err(Error::new(
        BackendErrorCode::RenderFailed,
        "a future GPU graph had no unavailable execution pass",
    ))
}

#[cfg(test)]
pub(crate) fn future_graph_diagnostic_for_test(
    graph: &GpuRenderGraph,
    output_format: Format,
    capabilities: &DeviceCapabilities,
) -> Result<Option<UnsupportedPrimitive>> {
    match ExecutableGraphDispatchEligibility::try_classify(
        graph,
        output_format,
        ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
        capabilities,
    )? {
        ExecutableGraphDispatchEligibility::FuturePasses => {
            let error = reject_future_graph_with_typed_diagnostic(graph)
                .expect_err("a future graph diagnostic probe must reject before execution");
            Ok(error.unsupported_primitive())
        }
        ExecutableGraphDispatchEligibility::ExactC08(_)
        | ExecutableGraphDispatchEligibility::ExactC09(_) => Ok(None),
        ExecutableGraphDispatchEligibility::ExactC12(_) => Ok(None),
    }
}

/// Renderer configuration that is fixed when a [`Renderer`] is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    antialiasing: Antialiasing,
    debug: bool,
    effect_quality_policy: EffectQualityPolicy,
    resource_cache_budget: ResourceCacheBudget,
}

impl Options {
    /// Creates the default GPU-only renderer configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            antialiasing: Antialiasing::Area,
            debug: false,
            effect_quality_policy: EffectQualityPolicy::RequireHighPrecision,
            resource_cache_budget: ResourceCacheBudget::DEFAULT,
        }
    }

    /// Returns the configured antialiasing method.
    #[must_use]
    pub const fn antialiasing(self) -> Antialiasing {
        self.antialiasing
    }

    /// Returns this configuration with a different antialiasing method.
    #[must_use]
    pub const fn with_antialiasing(mut self, antialiasing: Antialiasing) -> Self {
        self.antialiasing = antialiasing;
        self
    }

    /// Returns whether renderer diagnostics are enabled.
    #[must_use]
    pub const fn debug(self) -> bool {
        self.debug
    }

    /// Returns this configuration with renderer diagnostics enabled or disabled.
    #[must_use]
    pub const fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Returns the policy for effect precision when high precision is unavailable.
    #[must_use]
    pub const fn effect_quality_policy(self) -> EffectQualityPolicy {
        self.effect_quality_policy
    }

    /// Returns this configuration with a different effect precision policy.
    #[must_use]
    pub const fn with_effect_quality_policy(
        mut self,
        effect_quality_policy: EffectQualityPolicy,
    ) -> Self {
        self.effect_quality_policy = effect_quality_policy;
        self
    }

    /// Returns the maximum retained idle effect-resource cache budget.
    #[must_use]
    pub const fn resource_cache_budget(self) -> ResourceCacheBudget {
        self.resource_cache_budget
    }

    /// Returns this configuration with a different idle effect-resource cache budget.
    #[must_use]
    pub const fn with_resource_cache_budget(
        mut self,
        resource_cache_budget: ResourceCacheBudget,
    ) -> Self {
        self.resource_cache_budget = resource_cache_budget;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy for choosing effect precision on a compatible GPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EffectQualityPolicy {
    /// Require high-precision effect execution.
    #[default]
    RequireHighPrecision,
    /// Prefer high precision and allow reduced precision only when it is unavailable.
    AllowReducedPrecision,
}

/// Byte budget for retaining idle effect resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceCacheBudget(u64);

impl ResourceCacheBudget {
    /// Disables idle effect-resource retention.
    pub const DISABLED: Self = Self(0);

    /// Retains up to 64 MiB of idle effect resources by default.
    pub const DEFAULT: Self = Self(64 * 1024 * 1024);

    /// Creates an idle effect-resource retention budget in bytes.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// Returns this retention budget in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.0
    }
}

impl Default for ResourceCacheBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antialiasing {
    #[default]
    Area,
    Msaa8,
    Msaa16,
}
