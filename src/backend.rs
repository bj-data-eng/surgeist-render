#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::PresentedLifecycle;
use super::surface::{HeadlessResources, SurfaceBackend};
use super::*;
use super::{
    command::OffscreenBounds,
    geometry::physical_size,
    texture::{
        OffscreenTextureCache, OffscreenTextureHandle, TextureCacheKey, TextureDescriptor,
        TextureLifecycleStats, TextureUsageIntent, headless_texture_descriptor,
    },
};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::sync::Condvar;

pub(crate) struct Backend {
    pub(crate) context: vello::util::RenderContext,
    pub(crate) device_states: Vec<DeviceState>,
}

pub(crate) struct DeviceState {
    generation: u64,
    renderer: Option<vello::Renderer>,
    capabilities: DeviceCapabilities,
    signal: Arc<DeviceSignal>,
    terminal: Option<DeviceTerminalSignal>,
}

impl DeviceState {
    fn new(device_handle: &vello::util::DeviceHandle) -> Self {
        let signal = Arc::new(DeviceSignal::new());
        register_device_callbacks(&device_handle.device, Arc::clone(&signal));
        Self {
            generation: 0,
            renderer: None,
            capabilities: DeviceCapabilities::from_device(device_handle),
            signal,
            terminal: None,
        }
    }

    fn observe_terminal(&mut self) {
        let Some(signal) = self.signal.first_terminal() else {
            return;
        };
        if self.terminal.is_none() {
            self.terminal = Some(signal);
            self.renderer.take();
        }
    }

    fn terminal(&mut self) -> Option<&DeviceTerminalSignal> {
        self.observe_terminal();
        self.terminal.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn ready_slot_remains_ready_after_other_slot_loss_for_test() -> bool {
        let first_signal = Arc::new(DeviceSignal::new());
        let mut first = Self {
            generation: 0,
            renderer: None,
            capabilities: DeviceCapabilities::empty(),
            signal: Arc::clone(&first_signal),
            terminal: None,
        };
        let mut second = Self {
            generation: 0,
            renderer: None,
            capabilities: DeviceCapabilities::empty(),
            signal: Arc::new(DeviceSignal::new()),
            terminal: None,
        };
        first_signal.record(DeviceTerminalSignal::lost(
            DeviceLossReason::Destroyed,
            "test device loss".into(),
        ));
        first.observe_terminal();
        second.observe_terminal();
        first.terminal.is_some() && second.terminal.is_none()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceCapabilities {
    high_precision: bool,
    reduced_precision: bool,
    max_texture_dimension_2d: u32,
}

impl DeviceCapabilities {
    fn from_device(device_handle: &vello::util::DeviceHandle) -> Self {
        Self {
            high_precision: supports_effect_texture_format(
                device_handle
                    .adapter()
                    .get_texture_format_features(wgpu::TextureFormat::Rgba16Float),
            ),
            reduced_precision: supports_effect_texture_format(
                device_handle
                    .adapter()
                    .get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm),
            ),
            max_texture_dimension_2d: device_handle.device.limits().max_texture_dimension_2d,
        }
    }

    #[cfg(test)]
    const fn empty() -> Self {
        Self {
            high_precision: false,
            reduced_precision: false,
            max_texture_dimension_2d: 0,
        }
    }

    pub(crate) const fn runtime_report(
        self,
        surface_format: Format,
    ) -> AvailableRuntimeCapabilities {
        AvailableRuntimeCapabilities::new(
            surface_format,
            EffectPrecisionCapabilities::new(self.high_precision, self.reduced_precision),
            self.max_texture_dimension_2d,
        )
    }
}

fn supports_effect_texture_format(features: wgpu::TextureFormatFeatures) -> bool {
    features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        && features
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        && features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

#[derive(Clone, Debug)]
enum DeviceTerminalSignal {
    Lost {
        reason: DeviceLossReason,
        message: String,
    },
    Faulted {
        kind: GpuFaultKind,
        message: String,
        operation_generation: Option<u64>,
    },
}

impl DeviceTerminalSignal {
    fn lost(reason: DeviceLossReason, message: String) -> Self {
        Self::Lost { reason, message }
    }

    fn faulted(kind: GpuFaultKind, message: String, operation_generation: Option<u64>) -> Self {
        Self::Faulted {
            kind,
            message,
            operation_generation,
        }
    }

    const fn unavailable_reason(&self) -> RuntimeCapabilityUnavailableReason {
        match self {
            Self::Lost { reason, .. } => {
                RuntimeCapabilityUnavailableReason::DeviceLost { reason: *reason }
            }
            Self::Faulted { kind, .. } => {
                RuntimeCapabilityUnavailableReason::DeviceFaulted { kind: *kind }
            }
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Lost { message, .. } => message,
            Self::Faulted {
                message,
                operation_generation,
                ..
            } => {
                let _ = operation_generation;
                message
            }
        }
    }

    fn error(&self, operation: RuntimeOperation) -> Error {
        let diagnostic =
            RuntimeCapabilityUnavailable::try_new(operation, self.unavailable_reason())
                .expect("terminal-device diagnostics always use a permitted operation/reason pair");
        let mut error = Error::runtime_capability_unavailable(diagnostic);
        error.append_message(format_args!(": {}", self.message()));
        error
    }
}

struct DeviceSignal {
    state: Mutex<DeviceSignalState>,
    #[cfg(test)]
    changed: Condvar,
}

struct DeviceSignalState {
    first_terminal: Option<DeviceTerminalSignal>,
    // Task 4 owns transaction installation. Callbacks observe no active operation for now.
    active_operation_generation: Option<u64>,
}

impl DeviceSignal {
    fn new() -> Self {
        Self {
            state: Mutex::new(DeviceSignalState {
                first_terminal: None,
                active_operation_generation: None,
            }),
            #[cfg(test)]
            changed: Condvar::new(),
        }
    }

    fn record(&self, signal: DeviceTerminalSignal) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.first_terminal.is_none() {
            state.first_terminal = Some(signal);
            #[cfg(test)]
            self.changed.notify_all();
        }
    }

    fn first_terminal(&self) -> Option<DeviceTerminalSignal> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .first_terminal
            .clone()
    }

    fn record_fault(&self, kind: GpuFaultKind, message: String) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.first_terminal.is_none() {
            state.first_terminal = Some(DeviceTerminalSignal::faulted(
                kind,
                message,
                state.active_operation_generation,
            ));
            #[cfg(test)]
            self.changed.notify_all();
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_for_terminal(&self, timeout: Duration) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |state| state.first_terminal.is_none())
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.first_terminal.is_some()
    }
}

fn register_device_callbacks(device: &wgpu::Device, signal: Arc<DeviceSignal>) {
    let loss_signal = Arc::clone(&signal);
    device.set_device_lost_callback(move |reason, message| {
        loss_signal.record(DeviceTerminalSignal::lost(
            map_device_loss_reason(reason),
            message,
        ));
    });
    device.on_uncaptured_error(Arc::new(move |error| {
        let message = error.to_string();
        let kind = match error {
            wgpu::Error::Validation { .. } => GpuFaultKind::Validation,
            wgpu::Error::OutOfMemory { .. } => GpuFaultKind::OutOfMemory,
            wgpu::Error::Internal { .. } => GpuFaultKind::Internal,
        };
        signal.record_fault(kind, message);
    }));
}

const fn map_device_loss_reason(reason: wgpu::DeviceLostReason) -> DeviceLossReason {
    match reason {
        wgpu::DeviceLostReason::Unknown => DeviceLossReason::Unknown,
        wgpu::DeviceLostReason::Destroyed => DeviceLossReason::Destroyed,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceSlotIdentity {
    slot: usize,
    generation: u64,
}

impl DeviceSlotIdentity {
    const fn new(slot: usize, generation: u64) -> Self {
        Self { slot, generation }
    }

    pub(crate) const fn slot(self) -> usize {
        self.slot
    }

    #[cfg(test)]
    pub(crate) fn mark_stale_for_test(&mut self) {
        self.generation = self.generation.checked_add(1).unwrap();
    }
}

impl Backend {
    fn ensure_device_states(&mut self) {
        let known_slots = self.device_states.len();
        self.device_states.extend(
            self.context.devices[known_slots..]
                .iter()
                .map(DeviceState::new),
        );
    }

    pub(crate) fn device_slot_identity(&mut self, slot: usize) -> Result<DeviceSlotIdentity> {
        self.ensure_device_states();
        let Some(state) = self.device_states.get(slot) else {
            return Err(Error::new(
                BackendErrorCode::RendererCreateFailed,
                "Vello selected an unavailable device slot",
            ));
        };
        Ok(DeviceSlotIdentity::new(slot, state.generation))
    }

    pub(crate) fn has_device_slot(&mut self, identity: DeviceSlotIdentity) -> bool {
        self.ensure_device_states();
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        if state.generation != identity.generation
            || self.context.devices.get(identity.slot()).is_none()
        {
            return false;
        }
        state.observe_terminal();
        true
    }

    pub(crate) fn terminal_error(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
    ) -> Option<Error> {
        self.ensure_device_states();
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation
            || self.context.devices.get(identity.slot()).is_none()
        {
            return None;
        }
        state.terminal().map(|terminal| terminal.error(operation))
    }

    pub(crate) fn terminal_reason(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<RuntimeCapabilityUnavailableReason> {
        self.ensure_device_states();
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation
            || self.context.devices.get(identity.slot()).is_none()
        {
            return None;
        }
        state
            .terminal()
            .map(DeviceTerminalSignal::unavailable_reason)
    }

    pub(crate) fn device_capabilities(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<DeviceCapabilities> {
        self.ensure_device_states();
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation
            || self.context.devices.get(identity.slot()).is_none()
        {
            return None;
        }
        state.observe_terminal();
        (state.terminal.is_none()).then_some(state.capabilities)
    }

    #[cfg(test)]
    pub(crate) fn signal_loss_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        reason: DeviceLossReason,
    ) {
        self.ensure_device_states();
        if let Some(state) = self.device_states.get(identity.slot())
            && state.generation == identity.generation
        {
            state.signal.record(DeviceTerminalSignal::lost(
                reason,
                "test device loss".into(),
            ));
        }
    }

    #[cfg(test)]
    pub(crate) fn wait_for_terminal_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        timeout: Duration,
    ) -> bool {
        self.ensure_device_states();
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .is_some_and(|state| state.signal.wait_for_terminal(timeout))
    }

    #[cfg(test)]
    pub(crate) fn renderer_released_for_test(&mut self, identity: DeviceSlotIdentity) -> bool {
        self.ensure_device_states();
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        state.observe_terminal();
        state.renderer.is_none()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenRenderGpuContext<'a> {
    backend: &'a mut Backend,
    device_identity: DeviceSlotIdentity,
}

#[cfg_attr(not(test), allow(dead_code))]
impl<'a> OffscreenRenderGpuContext<'a> {
    #[must_use]
    pub(crate) fn new(backend: &'a mut Backend, device_identity: DeviceSlotIdentity) -> Self {
        Self {
            backend,
            device_identity,
        }
    }
}

/// Describes a Vello scene that has already been encoded in offscreen-local
/// coordinates; bounds size allocates the target texture, not a scene crop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenLocalSceneRenderRequest {
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
    parameters: Parameters,
}

#[cfg_attr(not(test), allow(dead_code))]
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

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenRenderTarget {
    handle: OffscreenTextureHandle,
    resource_id: u64,
    bounds: OffscreenBounds,
    descriptor: TextureDescriptor,
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenRenderTarget {
    fn new(
        handle: OffscreenTextureHandle,
        resource_id: u64,
        bounds: OffscreenBounds,
        descriptor: TextureDescriptor,
    ) -> Self {
        Self {
            handle,
            resource_id,
            bounds,
            descriptor,
        }
    }

    #[must_use]
    pub(crate) const fn handle(self) -> OffscreenTextureHandle {
        self.handle
    }

    #[must_use]
    pub(crate) const fn resource_id(self) -> u64 {
        self.resource_id
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> OffscreenBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn descriptor(self) -> TextureDescriptor {
        self.descriptor
    }
}

struct OffscreenTextureResource {
    resource_id: u64,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenTextureResourceCache {
    lifecycle: OffscreenTextureCache,
    released_by_key: HashMap<TextureCacheKey, VecDeque<OffscreenTextureResource>>,
    next_resource_id: u64,
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenTextureResourceCache {
    pub(crate) fn new() -> Self {
        Self {
            lifecycle: OffscreenTextureCache::new(),
            released_by_key: HashMap::new(),
            next_resource_id: 0,
        }
    }

    fn acquire(
        &mut self,
        device: &wgpu::Device,
        bounds: OffscreenBounds,
        descriptor: TextureDescriptor,
    ) -> Result<OffscreenTextureResourceLease> {
        let key = TextureCacheKey::from_descriptor(descriptor);
        let stats_before = self.lifecycle.stats();
        let handle = self.lifecycle.acquire(descriptor)?;
        let reused = self.lifecycle.stats().hits > stats_before.hits;
        let resource = if reused {
            match self
                .released_by_key
                .get_mut(&key)
                .and_then(VecDeque::pop_front)
            {
                Some(resource) => resource,
                None => {
                    let _ = self.lifecycle.release(handle);
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "offscreen texture resource cache lost a released GPU resource",
                    ));
                }
            }
        } else {
            self.next_resource_id = self.next_resource_id.checked_add(1).ok_or_else(|| {
                Error::invalid_value(
                    "offscreen texture resource id",
                    self.next_resource_id,
                    "must have remaining resource id space",
                )
            })?;
            let (texture, view) =
                create_texture(device, "Surgeist offscreen local scene target", descriptor);
            OffscreenTextureResource {
                resource_id: self.next_resource_id,
                texture,
                view,
            }
        };
        Ok(OffscreenTextureResourceLease {
            target: OffscreenRenderTarget::new(handle, resource.resource_id, bounds, descriptor),
            texture: resource.texture,
            view: resource.view,
        })
    }

    fn release_resource(&mut self, resource: OffscreenTextureResourceLease) -> Result<()> {
        self.lifecycle.release(resource.target.handle())?;
        self.released_by_key
            .entry(TextureCacheKey::from_descriptor(
                resource.target.descriptor(),
            ))
            .or_default()
            .push_back(OffscreenTextureResource {
                resource_id: resource.target.resource_id(),
                texture: resource.texture,
                view: resource.view,
            });
        Ok(())
    }

    #[must_use]
    pub(crate) const fn stats(&self) -> TextureLifecycleStats {
        self.lifecycle.stats()
    }

    #[must_use]
    pub(crate) const fn live_count(&self) -> usize {
        self.lifecycle.live_count()
    }

    #[must_use]
    pub(crate) fn released_resource_count(&self) -> usize {
        self.released_by_key.values().map(VecDeque::len).sum()
    }
}

impl Default for OffscreenTextureResourceCache {
    fn default() -> Self {
        Self::new()
    }
}

struct OffscreenTextureResourceLease {
    target: OffscreenRenderTarget,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use = "offscreen rendered texture leases must be released back to their resource cache"]
pub(crate) struct OffscreenRenderedTextureLease {
    target: OffscreenRenderTarget,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    timings: RenderTimings,
}

impl fmt::Debug for OffscreenRenderedTextureLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OffscreenRenderedTextureLease")
            .field("target", &self.target)
            .field("timings", &self.timings)
            .finish_non_exhaustive()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenRenderedTextureLease {
    #[must_use]
    pub(crate) const fn target(&self) -> OffscreenRenderTarget {
        self.target
    }

    #[must_use]
    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[must_use]
    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[must_use]
    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    pub(crate) fn release(self, cache: &mut OffscreenTextureResourceCache) -> Result<()> {
        cache.release_resource(OffscreenTextureResourceLease {
            target: self.target,
            texture: self.texture,
            view: self.view,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn offscreen_local_scene_texture_descriptor(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<TextureDescriptor> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    TextureDescriptor::try_new(
        physical_size(bounds.rect().size(), scale)?,
        format,
        TextureUsageIntent::OffscreenLayer,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn render_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    cache: &mut OffscreenTextureResourceCache,
    scene: &vello::Scene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let descriptor =
        offscreen_local_scene_texture_descriptor(request.bounds, request.scale, request.format)?;
    let Some(context) = context else {
        return Err(Error::new(
            BackendErrorCode::AdapterUnavailable,
            "offscreen Vello local scene rendering requires an available wgpu device context",
        ));
    };
    ensure_vello_renderer(context.backend, options, context.device_identity)?;
    let slot = context.device_identity.slot();
    let (render_context, device_states) =
        (&context.backend.context, &mut context.backend.device_states);
    let Some(device_handle) = render_context.devices.get(slot) else {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "offscreen Vello device slot disappeared before rendering",
        ));
    };
    let resource = cache.acquire(&device_handle.device, request.bounds, descriptor)?;
    let render_start = Instant::now();
    let Some(renderer) = device_states
        .get_mut(slot)
        .and_then(|state| state.renderer.as_mut())
    else {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "offscreen Vello renderer disappeared before rendering",
        ));
    };
    let result = renderer.render_to_texture(
        &device_handle.device,
        &device_handle.queue,
        scene,
        &resource.view,
        &vello_render_params(
            request.parameters,
            resource.target.descriptor().physical_size(),
            options.antialiasing(),
        ),
    );
    if let Err(source) = result {
        let error =
            Error::new(vello_error_code(&source), vello_error_message(&source)).with_source(source);
        cache.release_resource(resource)?;
        return Err(error);
    }
    Ok(OffscreenRenderedTextureLease {
        target: resource.target,
        texture: resource.texture,
        view: resource.view,
        timings: RenderTimings {
            render_time: render_start.elapsed(),
            present_time: Duration::ZERO,
        },
    })
}

pub(crate) fn render_vello_surface(
    backend: &mut Backend,
    options: Options,
    surface: &mut Surface,
    scene: &vello::Scene,
    parameters: Parameters,
) -> Result<RenderTimings> {
    match &mut surface.backend {
        SurfaceBackend::ContractOnly { .. } => Ok(RenderTimings::default()),
        SurfaceBackend::Headless {
            device_identity,
            resources,
            physical_size,
        } => {
            if physical_size.width() == 0 || physical_size.height() == 0 {
                return Ok(RenderTimings::default());
            }
            ensure_vello_renderer(backend, options, *device_identity)?;
            let slot = device_identity.slot();
            if matches!(resources, HeadlessResources::Pending) {
                let Some(device_handle) = backend.context.devices.get(slot) else {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "headless Vello device slot disappeared before allocation",
                    ));
                };
                let (next_texture, next_view) = create_headless_texture(
                    &device_handle.device,
                    *physical_size,
                    surface.options.format,
                )?;
                *resources = HeadlessResources::Ready {
                    texture: next_texture,
                    view: next_view,
                };
            }
            let HeadlessResources::Ready { view, .. } = resources else {
                unreachable!("headless resources should be ready after allocation");
            };
            let (render_context, device_states) = (&backend.context, &mut backend.device_states);
            let Some(device_handle) = render_context.devices.get(slot) else {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "headless Vello device slot disappeared before rendering",
                ));
            };
            let render_start = Instant::now();
            let Some(renderer) = device_states
                .get_mut(slot)
                .and_then(|state| state.renderer.as_mut())
            else {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "headless Vello renderer disappeared before rendering",
                ));
            };
            renderer
                .render_to_texture(
                    &device_handle.device,
                    &device_handle.queue,
                    scene,
                    view,
                    &vello_render_params(parameters, *physical_size, options.antialiasing()),
                )
                .map_err(|source| {
                    Error::new(vello_error_code(&source), vello_error_message(&source))
                        .with_source(source)
                })?;
            Ok(RenderTimings {
                render_time: render_start.elapsed(),
                present_time: Duration::ZERO,
            })
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        SurfaceBackend::Presented {
            surface: native,
            device_identity,
            lifecycle,
            ..
        } => {
            match lifecycle {
                PresentedLifecycle::ResizePending {
                    physical_size,
                    resizing,
                } => {
                    backend.context.resize_surface(
                        native,
                        physical_size.width(),
                        physical_size.height(),
                    );
                    *lifecycle = PresentedLifecycle::Ready {
                        resizing: *resizing,
                    };
                }
                PresentedLifecycle::NonRenderable { .. } | PresentedLifecycle::Lost => {
                    return Ok(RenderTimings::default());
                }
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::Occluded { .. } => {}
            }
            ensure_vello_renderer(backend, options, *device_identity)?;
            let slot = device_identity.slot();
            let (render_context, device_states) = (&backend.context, &mut backend.device_states);
            let Some(device_handle) = render_context.devices.get(slot) else {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "presented Vello device slot disappeared before rendering",
                ));
            };
            let resizing = lifecycle.resize_state();
            let render_start = Instant::now();
            let Some(renderer) = device_states
                .get_mut(slot)
                .and_then(|state| state.renderer.as_mut())
            else {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "presented Vello renderer disappeared before rendering",
                ));
            };
            renderer
                .render_to_texture(
                    &device_handle.device,
                    &device_handle.queue,
                    scene,
                    &native.target_view,
                    &vello::RenderParams {
                        width: native.config.width,
                        height: native.config.height,
                        base_color: parameters.base_color.into(),
                        antialiasing_method: options.antialiasing().into(),
                    },
                )
                .map_err(|source| {
                    Error::new(vello_error_code(&source), vello_error_message(&source))
                        .with_source(source)
                })?;
            let render_time = render_start.elapsed();

            let present_start = Instant::now();
            let surface_texture = match native.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                    *lifecycle = PresentedLifecycle::Ready { resizing };
                    surface_texture
                }
                wgpu::CurrentSurfaceTexture::Outdated
                | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    backend.context.configure_surface(native);
                    return Err(Error::new(
                        BackendErrorCode::SurfaceOutdated,
                        "surface is outdated and requires reconfiguration",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    *lifecycle = PresentedLifecycle::Occluded { resizing };
                    return Ok(RenderTimings {
                        render_time,
                        present_time: present_start.elapsed(),
                    });
                }
                wgpu::CurrentSurfaceTexture::Timeout => {
                    return Err(Error::new(
                        BackendErrorCode::SurfaceTimeout,
                        "timed out acquiring surface texture",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    *lifecycle = PresentedLifecycle::Lost;
                    return Err(Error::new(
                        BackendErrorCode::SurfaceLost,
                        "surface was lost",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "surface texture validation failed",
                    ));
                }
            };

            let mut encoder =
                device_handle
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Surgeist surface blit"),
                    });
            native.blitter.copy(
                &device_handle.device,
                &mut encoder,
                &native.target_view,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
            device_handle.queue.submit([encoder.finish()]);
            surface_texture.present();
            device_handle
                .device
                .poll(wgpu::PollType::Poll)
                .map_err(|source| {
                    Error::new(
                        BackendErrorCode::PresentFailed,
                        "failed to poll render device",
                    )
                    .with_source(source)
                })?;
            Ok(RenderTimings {
                render_time,
                present_time: present_start.elapsed(),
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RenderTimings {
    pub(crate) render_time: Duration,
    pub(crate) present_time: Duration,
}

pub(crate) fn ensure_vello_renderer(
    backend: &mut Backend,
    options: Options,
    device_identity: DeviceSlotIdentity,
) -> Result<()> {
    if !backend.has_device_slot(device_identity) {
        return Err(Error::new(
            BackendErrorCode::RendererCreateFailed,
            "Vello device slot is unavailable",
        ));
    }
    if let Some(error) = backend.terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
    {
        return Err(error);
    }
    let slot = device_identity.slot();
    let renderer_is_ready = backend
        .device_states
        .get(slot)
        .is_some_and(|state| state.renderer.is_some());
    if !renderer_is_ready {
        let Some(device_handle) = backend.context.devices.get(slot) else {
            return Err(Error::new(
                BackendErrorCode::RendererCreateFailed,
                "Vello device slot disappeared before renderer creation",
            ));
        };
        let renderer = vello::Renderer::new(&device_handle.device, vello_renderer_options(options))
            .map_err(|source| {
                Error::new(
                    BackendErrorCode::RendererCreateFailed,
                    "failed to create Vello renderer",
                )
                .with_source(source)
            })?;
        let Some(state) = backend.device_states.get_mut(slot) else {
            return Err(Error::new(
                BackendErrorCode::RendererCreateFailed,
                "Vello device state disappeared before renderer creation",
            ));
        };
        state.renderer = Some(renderer);
    }
    Ok(())
}

pub(crate) fn vello_renderer_options(options: Options) -> vello::RendererOptions {
    vello::RendererOptions {
        use_cpu: false,
        antialiasing_support: vello_aa_support(options.antialiasing()),
        num_init_threads: NonZeroUsize::new(1),
        ..vello::RendererOptions::default()
    }
}

pub(crate) fn vello_error_code(error: &vello::Error) -> BackendErrorCode {
    match error {
        vello::Error::WgpuErrorFromScope(wgpu::Error::OutOfMemory { .. }) => {
            BackendErrorCode::SurfaceOutOfMemory
        }
        _ => BackendErrorCode::RenderFailed,
    }
}

pub(crate) fn vello_error_message(error: &vello::Error) -> &'static str {
    match vello_error_code(error) {
        BackendErrorCode::SurfaceOutOfMemory => "rendering exhausted GPU memory",
        _ => "failed to render scene",
    }
}

pub(crate) fn vello_render_params(
    parameters: Parameters,
    physical_size: PhysicalSize,
    antialiasing: Antialiasing,
) -> vello::RenderParams {
    vello::RenderParams {
        base_color: parameters.base_color.into(),
        width: physical_size.width(),
        height: physical_size.height(),
        antialiasing_method: antialiasing.into(),
    }
}

pub(crate) fn vello_aa_support(antialiasing: Antialiasing) -> vello::AaSupport {
    [vello::AaConfig::from(antialiasing)].into_iter().collect()
}

pub(crate) fn create_headless_texture(
    device: &wgpu::Device,
    physical_size: PhysicalSize,
    format: Format,
) -> Result<(wgpu::Texture, wgpu::TextureView)> {
    let descriptor = headless_texture_descriptor(physical_size, format)?;
    Ok(create_texture(
        device,
        "Surgeist headless target",
        descriptor,
    ))
}

pub(crate) fn create_texture(
    device: &wgpu::Device,
    label: &'static str,
    descriptor: TextureDescriptor,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: descriptor.physical_size().width(),
            height: descriptor.physical_size().height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: descriptor.format().into(),
        usage: descriptor.wgpu_usage(),
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(crate) fn read_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    physical_size: PhysicalSize,
) -> Result<ImageBuffer> {
    let width = physical_size.width().max(1);
    let height = physical_size.height().max(1);
    let padded_bytes_per_row = (width * 4).next_multiple_of(256);
    let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Surgeist headless readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist headless copy"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|source| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "failed to poll render device",
            )
            .with_source(source)
        })?;
    receiver
        .recv()
        .map_err(|_| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "headless readback callback dropped",
            )
        })?
        .map_err(|source| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "failed to map headless readback",
            )
            .with_source(source)
        })?;

    let mapped = slice.get_mapped_range();
    let row_bytes = (width * 4) as usize;
    let mut rgba = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        rgba.extend_from_slice(&mapped[start..start + row_bytes]);
    }
    drop(mapped);
    buffer.unmap();

    Ok(ImageBuffer {
        size: physical_size,
        rgba,
    })
}

impl From<Antialiasing> for vello::AaConfig {
    fn from(antialiasing: Antialiasing) -> Self {
        match antialiasing {
            Antialiasing::Area => Self::Area,
            Antialiasing::Msaa8 => Self::Msaa8,
            Antialiasing::Msaa16 => Self::Msaa16,
        }
    }
}
