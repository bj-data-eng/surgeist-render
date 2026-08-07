use super::{
    Backend,
    device::{
        DeviceCapabilities, DeviceSignal, DeviceSlotIdentity, DeviceTerminalSignal,
        ReadyDeviceState,
    },
    offscreen::{self, OffscreenRenderTarget, OffscreenRenderedTextureLease},
};
use crate::{
    Format, Options, Parameters,
    capability::EffectPrecisionCapabilities,
    command::OffscreenBounds,
    error::{
        BackendErrorCode, DeviceLossReason, Error, GpuFaultKind, Result,
        RuntimeCapabilityUnavailableReason, RuntimeOperation,
    },
    renderer::ResourceCacheBudget,
    resource::{
        ManagerIdentity, ResourceAccountingFault, ResourceManager,
        ResourceManagerObservationForTest, WorkingFormat,
    },
    shader::{DevicePassCache, DevicePassCacheCountsForTest},
    vello_engine::{VelloEngineState, scene::VelloScene},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "render-window")]
use crate::{
    Attachment, Renderer, SurfaceOptions,
    geometry::PhysicalSize,
    gpu_transaction::{GpuOperationStage, GpuOperationTransaction},
    surface::{
        DisplayFreePresentedSurfaceObservationForTest,
        DisplayFreePresentedSurfaceObservationHandleForTest, PresentedAcquireOutcomeForTest,
        PresentedLifecycle, PresentedSurface, Surface, SurfaceBackend,
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
