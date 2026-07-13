#[cfg(test)]
use super::gpu_transaction::{
    AfterInternalVelloSubmitCheckpointForTest, InternalVelloPayload,
    InternalVelloSubmissionObservationForTest,
};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::PresentedLifecycle;
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::PresentedSurface;
use super::surface::{HeadlessResources, SurfaceBackend};
#[cfg(test)]
use super::vello_engine::{
    ActiveVelloEncodingScope, PreparedVelloPass, TransactionEncodingState, TransactionTargetIntent,
    VelloResourceManagerObservationForTest,
};
use super::vello_engine::{VelloEngineState, VelloResourceManager};
use super::*;
use super::{
    command::OffscreenBounds,
    geometry::physical_size,
    gpu_transaction::{GpuOperationStage, GpuOperationTransaction},
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
use std::{
    sync::{Condvar, Weak},
    task::{Context, Poll, Waker},
};

pub(crate) struct Backend {
    instance: wgpu::Instance,
    device_states: Vec<DeviceState>,
    #[cfg(test)]
    pub(crate) terminal_signal_after_renderer_creation: Option<DeviceSlotIdentity>,
}

pub(crate) struct DeviceState {
    generation: u64,
    lifecycle: DeviceLifecycle,
    capabilities: DeviceCapabilities,
    signal: Arc<DeviceSignal>,
    next_operation_generation: u64,
}

struct ReadyDeviceState {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T5 establishes the per-device checked engine owner that T6 will encode through."
        )
    )]
    engine: VelloEngineState,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T5 establishes the per-device resource owner that T6 will adopt leases into."
        )
    )]
    resources: VelloResourceManager,
    renderer: Option<vello::Renderer>,
    #[cfg(test)]
    drop_witness: Arc<()>,
}

enum DeviceLifecycle {
    Ready(Box<ReadyDeviceState>),
    Terminal(Arc<DeviceTerminalSignal>),
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ReadyDeviceStateDropWitnessForTest {
    ready_bundle: Weak<()>,
}

#[cfg(test)]
impl ReadyDeviceStateDropWitnessForTest {
    fn from_ready_bundle(ready_bundle: &Arc<()>) -> Self {
        Self {
            ready_bundle: Arc::downgrade(ready_bundle),
        }
    }

    pub(crate) fn was_dropped_for_test(&self) -> bool {
        self.ready_bundle.upgrade().is_none()
    }
}

#[cfg(test)]
pub(crate) struct ReadyDeviceStateBorrowForTest<'ready> {
    adapter: &'ready wgpu::Adapter,
    device: &'ready wgpu::Device,
    queue: &'ready wgpu::Queue,
    engine: &'ready VelloEngineState,
    resources: &'ready VelloResourceManager,
    renderer: Option<&'ready vello::Renderer>,
    drop_witness: ReadyDeviceStateDropWitnessForTest,
}

#[cfg(test)]
impl ReadyDeviceStateBorrowForTest<'_> {
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
    ) -> VelloResourceManagerObservationForTest {
        self.resources.observation_for_test()
    }

    pub(crate) fn external_renderer_for_test(&self) -> Option<&vello::Renderer> {
        self.renderer
    }

    pub(crate) fn drop_witness_for_test(&self) -> ReadyDeviceStateDropWitnessForTest {
        self.drop_witness.clone()
    }
}

#[cfg(test)]
impl ReadyDeviceState {
    fn borrow_for_test(&self) -> ReadyDeviceStateBorrowForTest<'_> {
        ReadyDeviceStateBorrowForTest {
            adapter: &self.adapter,
            device: &self.device,
            queue: &self.queue,
            engine: &self.engine,
            resources: &self.resources,
            renderer: self.renderer.as_ref(),
            drop_witness: ReadyDeviceStateDropWitnessForTest::from_ready_bundle(&self.drop_witness),
        }
    }
}

impl DeviceState {
    async fn new(adapter: wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let signal = Arc::new(DeviceSignal::new());
        register_device_callbacks(&device, Arc::clone(&signal));
        let capabilities = DeviceCapabilities::from_device(&adapter, &device);
        let engine = VelloEngineState::new_for_device_state(&device)
            .await
            .map_err(|source| {
                Error::new(
                    BackendErrorCode::RendererCreateFailed,
                    "failed to create the checked internal Vello engine",
                )
                .with_source(source)
            })?;
        let resources = VelloResourceManager::new();
        debug_assert!(resources.is_empty());
        Ok(Self {
            generation: 0,
            lifecycle: DeviceLifecycle::Ready(Box::new(ReadyDeviceState {
                adapter,
                device,
                queue,
                engine,
                resources,
                renderer: None,
                #[cfg(test)]
                drop_witness: Arc::new(()),
            })),
            capabilities,
            signal,
            next_operation_generation: 0,
        })
    }

    fn observe_terminal(&mut self) {
        let Some(terminal) = self.signal.first_terminal() else {
            return;
        };
        if matches!(&self.lifecycle, DeviceLifecycle::Ready(_)) {
            self.lifecycle = DeviceLifecycle::Terminal(terminal);
        }
    }

    fn terminal(&mut self) -> Option<&DeviceTerminalSignal> {
        self.observe_terminal();
        match &self.lifecycle {
            DeviceLifecycle::Ready(_) => None,
            DeviceLifecycle::Terminal(terminal) => Some(terminal.as_ref()),
        }
    }

    fn ready(&self) -> Option<&ReadyDeviceState> {
        match &self.lifecycle {
            DeviceLifecycle::Ready(ready) => Some(ready),
            DeviceLifecycle::Terminal(_) => None,
        }
    }

    fn ready_mut(&mut self) -> Option<&mut ReadyDeviceState> {
        match &mut self.lifecycle {
            DeviceLifecycle::Ready(ready) => Some(ready),
            DeviceLifecycle::Terminal(_) => None,
        }
    }

    #[cfg(test)]
    fn ready_borrow_for_test(&mut self) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        self.observe_terminal();
        self.ready().map(ReadyDeviceState::borrow_for_test)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceCapabilities {
    high_precision: bool,
    reduced_precision: bool,
    max_texture_dimension_2d: u32,
}

impl DeviceCapabilities {
    fn from_device(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        Self {
            high_precision: supports_effect_texture_format(
                adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float),
            ),
            reduced_precision: supports_effect_texture_format(
                adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm),
            ),
            max_texture_dimension_2d: device.limits().max_texture_dimension_2d,
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

#[derive(Debug)]
pub(crate) enum DeviceTerminalSignal {
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

    pub(crate) fn error(&self, operation: RuntimeOperation) -> Error {
        let diagnostic =
            RuntimeCapabilityUnavailable::try_new(operation, self.unavailable_reason())
                .expect("terminal-device diagnostics always use a permitted operation/reason pair");
        let mut error = Error::runtime_capability_unavailable(diagnostic);
        error.append_message(format_args!(": {}", self.message()));
        error
    }

    #[cfg(test)]
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

pub(crate) struct DeviceSignal {
    state: Mutex<DeviceSignalState>,
    #[cfg(test)]
    changed: Condvar,
}

struct DeviceSignalState {
    first_terminal: Option<Arc<DeviceTerminalSignal>>,
    // The lease clears this only when it still owns the recorded generation.
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
            state.first_terminal = Some(Arc::new(signal));
            #[cfg(test)]
            self.changed.notify_all();
        }
    }

    pub(crate) fn first_terminal(&self) -> Option<Arc<DeviceTerminalSignal>> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .first_terminal
            .clone()
    }

    /// Atomically snapshots terminal state and releases this operation's lease.
    pub(crate) fn finish_active_generation(
        &self,
        generation: u64,
    ) -> Option<Arc<DeviceTerminalSignal>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let terminal = state.first_terminal.clone();
        if state.active_operation_generation == Some(generation) {
            state.active_operation_generation = None;
        }
        terminal
    }

    fn record_fault(&self, kind: GpuFaultKind, message: String) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.first_terminal.is_none() {
            state.first_terminal = Some(Arc::new(DeviceTerminalSignal::faulted(
                kind,
                message,
                state.active_operation_generation,
            )));
            #[cfg(test)]
            self.changed.notify_all();
        }
    }

    pub(crate) fn activate(&self, generation: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.active_operation_generation = Some(generation);
    }

    pub(crate) fn clear_active(&self, generation: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active_operation_generation == Some(generation) {
            state.active_operation_generation = None;
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Arc<Self> {
        Arc::new(Self::new())
    }

    #[cfg(test)]
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

    #[cfg(test)]
    pub(crate) fn active_generation_for_test(&self) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active_operation_generation
    }

    #[cfg(test)]
    pub(crate) fn record_uncaptured_fault_for_test(&self, kind: GpuFaultKind, message: &str) {
        self.record_fault(kind, message.into());
    }

    #[cfg(test)]
    pub(crate) fn finish_active_generation_for_test(
        &self,
        generation: u64,
    ) -> Option<Arc<DeviceTerminalSignal>> {
        self.finish_active_generation(generation)
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
    pub(crate) fn new() -> Self {
        let backends = wgpu::Backends::from_env().unwrap_or_default();
        let flags = wgpu::InstanceFlags::from_build_config().with_env();
        let memory_budget_thresholds = wgpu::MemoryBudgetThresholds::default();
        let backend_options = wgpu::BackendOptions::from_env_or_default();
        Self {
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor {
                display: None,
                backends,
                flags,
                memory_budget_thresholds,
                backend_options,
            }),
            device_states: Vec::new(),
            #[cfg(test)]
            terminal_signal_after_renderer_creation: None,
        }
    }

    pub(crate) async fn select_device(
        &mut self,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Option<DeviceSlotIdentity>> {
        let existing = if let Some(surface) = compatible_surface {
            self.device_states
                .iter()
                .enumerate()
                .find_map(|(slot, state)| {
                    state
                        .ready()
                        .filter(|ready| ready.adapter.is_surface_supported(surface))
                        .map(|_| DeviceSlotIdentity::new(slot, state.generation))
                })
        } else {
            self.device_states
                .first()
                .map(|state| DeviceSlotIdentity::new(0, state.generation))
        };
        if existing.is_some() {
            return Ok(existing);
        }
        self.new_device(compatible_surface).await
    }

    async fn new_device(
        &mut self,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Option<DeviceSlotIdentity>> {
        let adapter = match wgpu::util::initialize_adapter_from_env_or_default(
            &self.instance,
            compatible_surface,
        )
        .await
        {
            Ok(adapter) => adapter,
            Err(_) => return Ok(None),
        };
        let supported_features = adapter.features();
        let requested_features = wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE;
        let (device, queue) = match adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: supported_features & requested_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
        {
            Ok(device) => device,
            Err(_) => return Ok(None),
        };
        let state = DeviceState::new(adapter, device, queue).await?;
        let slot = self.device_states.len();
        let generation = state.generation;
        self.device_states.push(state);
        Ok(Some(DeviceSlotIdentity::new(slot, generation)))
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(crate) async fn create_presented_surface(
        &mut self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        physical_size: PhysicalSize,
        present_mode: wgpu::PresentMode,
    ) -> Result<(PresentedSurface, DeviceSlotIdentity)> {
        let surface = self
            .instance
            .create_surface(target.into())
            .map_err(|source| {
                Error::new(
                    BackendErrorCode::SurfaceCreateFailed,
                    "failed to create a WGPU presentation surface",
                )
                .with_source(source)
            })?;
        let identity = self.select_device(Some(&surface)).await?.ok_or_else(|| {
            Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "no compatible WGPU adapter is available for the presentation surface",
            )
        })?;
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::AdapterSelection,
            BackendErrorCode::SurfaceCreateFailed,
            "the selected presentation device is unavailable",
        )?;
        let presented = PresentedSurface::new(
            surface,
            &ready.adapter,
            &ready.device,
            physical_size,
            present_mode,
        )?;
        Ok((presented, identity))
    }

    fn ready_state_mut(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
        error_code: BackendErrorCode,
        unavailable_message: &'static str,
    ) -> Result<&mut ReadyDeviceState> {
        let state = self
            .device_states
            .get_mut(identity.slot())
            .ok_or_else(|| Error::new(error_code, unavailable_message))?;
        if state.generation != identity.generation {
            return Err(Error::new(error_code, unavailable_message));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(operation));
        }
        state
            .ready_mut()
            .ok_or_else(|| Error::new(error_code, unavailable_message))
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

    fn render_vello_to_texture(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
        scene: &vello::Scene,
        target: &wgpu::TextureView,
        parameters: &vello::RenderParams,
    ) -> Result<()> {
        let ready = self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::RenderFailed,
            "Vello device resources are unavailable before rendering",
        )?;
        let renderer = ready.renderer.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "Vello renderer disappeared before rendering",
            )
        })?;
        renderer
            .render_to_texture(&ready.device, &ready.queue, scene, target, parameters)
            .map_err(|source| {
                Error::new(vello_error_code(&source), vello_error_message(&source))
                    .with_source(source)
            })
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn resize_presented_surface(
        &mut self,
        identity: DeviceSlotIdentity,
        surface: &mut PresentedSurface,
        physical_size: PhysicalSize,
    ) -> Result<()> {
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "presented Vello device resources are unavailable before resize",
        )?;
        surface.resize(&ready.device, physical_size);
        Ok(())
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn configure_presented_surface(
        &mut self,
        identity: DeviceSlotIdentity,
        surface: &PresentedSurface,
    ) -> Result<()> {
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "presented Vello device resources are unavailable before configuration",
        )?;
        surface.configure(&ready.device);
        Ok(())
    }

    pub(crate) fn begin_gpu_operation(
        &mut self,
        identity: DeviceSlotIdentity,
        stage: GpuOperationStage,
        operation: RuntimeOperation,
    ) -> Result<GpuOperationTransaction> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before transaction setup",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
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
                BackendErrorCode::RenderFailed,
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

    #[cfg(test)]
    pub(crate) async fn submit_prepared_vello_pass_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<InternalVelloSubmissionObservationForTest> {
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
            label: Some("T6 transaction-owned internal Vello target"),
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
            label: Some("T6 transaction-owned internal Vello encoder"),
        });
        let mut scope = ActiveVelloEncodingScope::begin(device);
        let lease = {
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
            match prepared.encode_into(engine, &mut encoding) {
                Ok(lease) => lease,
                Err(failure) => {
                    let (error, aborted) = failure.into_error_and_aborted_resources();
                    resources.record_aborted_resources(aborted);
                    return Err(error);
                }
            }
        };
        let lease = match scope.finish_with_lease(lease).await {
            Ok(lease) => lease,
            Err(failure) => {
                let (error, aborted) = failure.into_error_and_aborted_resources();
                resources.record_aborted_resources(aborted);
                return Err(error);
            }
        };
        let observation = InternalVelloSubmissionObservationForTest::default();
        let payload = InternalVelloPayload::observed_for_test(
            command_encoder.finish(),
            resources.pending_commit(lease),
            observation.clone(),
        );
        transaction
            .submit_internal_vello(queue, payload, RuntimeOperation::SurfaceRendering)
            .await?;
        Ok(observation)
    }

    #[cfg(test)]
    pub(crate) async fn cancel_prepared_vello_pass_after_submit_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        prepared: &PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<VelloResourceManagerObservationForTest> {
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
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T6 cancellation-owned internal Vello target"),
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
            label: Some("T6 cancellation-owned internal Vello encoder"),
        });
        let mut scope = ActiveVelloEncodingScope::begin(device);
        let lease = {
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
            match prepared.encode_into(engine, &mut encoding) {
                Ok(lease) => lease,
                Err(failure) => {
                    let (error, aborted) = failure.into_error_and_aborted_resources();
                    resources.record_aborted_resources(aborted);
                    return Err(error);
                }
            }
        };
        let lease = match scope.finish_with_lease(lease).await {
            Ok(lease) => lease,
            Err(failure) => {
                let (error, aborted) = failure.into_error_and_aborted_resources();
                resources.record_aborted_resources(aborted);
                return Err(error);
            }
        };
        let (checkpoint, checkpoint_observed) = AfterInternalVelloSubmitCheckpointForTest::paused();
        let payload = InternalVelloPayload::paused_after_submit_for_test(
            command_encoder.finish(),
            resources.pending_commit(lease),
            checkpoint,
        );
        let mut submission = Box::pin(transaction.submit_internal_vello(
            queue,
            payload,
            RuntimeOperation::SurfaceRendering,
        ));
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let poll = submission.as_mut().poll(&mut context);
        assert!(
            matches!(poll, Poll::Pending),
            "the post-submit cancellation checkpoint must pause the real submission future"
        );
        checkpoint_observed
            .try_recv()
            .expect("the real internal queue submission must reach the post-submit checkpoint");
        drop(submission);

        Ok(resources.observation_for_test())
    }

    pub(crate) fn has_device_slot(&mut self, identity: DeviceSlotIdentity) -> bool {
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        if state.generation != identity.generation {
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
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state.terminal().map(|terminal| terminal.error(operation))
    }

    pub(crate) fn terminal_reason(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<RuntimeCapabilityUnavailableReason> {
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state
            .terminal()
            .map(DeviceTerminalSignal::unavailable_reason)
    }

    pub(crate) fn observe_device_terminal(&mut self, identity: DeviceSlotIdentity) {
        if let Some(state) = self.device_states.get_mut(identity.slot())
            && state.generation == identity.generation
        {
            state.observe_terminal();
        }
    }

    pub(crate) fn device_capabilities(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<DeviceCapabilities> {
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state.observe_terminal();
        state.terminal().is_none().then_some(state.capabilities)
    }

    #[cfg(test)]
    pub(crate) fn signal_loss_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        reason: DeviceLossReason,
    ) {
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
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .is_some_and(|state| state.signal.wait_for_terminal(timeout))
    }

    #[cfg(test)]
    pub(crate) fn renderer_released_for_test(&mut self, identity: DeviceSlotIdentity) -> bool {
        let Some(state) = self.device_states.get_mut(identity.slot()) else {
            return false;
        };
        state.observe_terminal();
        state.ready().is_none_or(|ready| ready.renderer.is_none())
    }

    #[cfg(test)]
    pub(crate) fn ready_device_state_borrow_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        let state = self.device_states.get_mut(identity.slot())?;
        if state.generation != identity.generation {
            return None;
        }
        state.ready_borrow_for_test()
    }

    #[cfg(test)]
    pub(crate) fn active_operation_generation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<u64> {
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .and_then(|state| state.signal.active_generation_for_test())
    }

    #[cfg(test)]
    pub(crate) fn arm_terminal_signal_after_renderer_creation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) {
        self.terminal_signal_after_renderer_creation = Some(identity);
    }

    #[cfg(test)]
    fn inject_terminal_signal_after_renderer_creation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) {
        if self.terminal_signal_after_renderer_creation != Some(identity) {
            return;
        }
        self.terminal_signal_after_renderer_creation = None;
        self.signal_loss_for_test(identity, DeviceLossReason::Destroyed);
    }

    #[cfg(test)]
    pub(crate) async fn add_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        self.new_device(None).await?.ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "the donor WGPU device could not be created",
            )
        })
    }

    #[cfg(test)]
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
    ensure_vello_renderer(
        context.backend,
        options,
        context.device_identity,
        RuntimeOperation::SurfaceRendering,
    )?;
    let resource = {
        let (device, _) = context
            .backend
            .device_queue(context.device_identity, RuntimeOperation::SurfaceRendering)?;
        cache.acquire(device, request.bounds, descriptor)?
    };
    let render_start = Instant::now();
    let result = context.backend.render_vello_to_texture(
        context.device_identity,
        RuntimeOperation::SurfaceRendering,
        scene,
        &resource.view,
        &vello_render_params(
            request.parameters,
            resource.target.descriptor().physical_size(),
            options.antialiasing(),
        ),
    );
    if let Err(error) = result {
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
            ensure_vello_renderer(
                backend,
                options,
                *device_identity,
                RuntimeOperation::SurfaceRendering,
            )?;
            if matches!(resources, HeadlessResources::Pending) {
                let (next_texture, next_view) = backend.create_headless_surface_texture(
                    *device_identity,
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
            let render_start = Instant::now();
            backend.render_vello_to_texture(
                *device_identity,
                RuntimeOperation::SurfaceRendering,
                scene,
                view,
                &vello_render_params(parameters, *physical_size, options.antialiasing()),
            )?;
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
                    backend.resize_presented_surface(*device_identity, native, *physical_size)?;
                    *lifecycle = PresentedLifecycle::Ready {
                        resizing: *resizing,
                    };
                }
                PresentedLifecycle::NonRenderable { .. } | PresentedLifecycle::Lost => {
                    return Ok(RenderTimings::default());
                }
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::Occluded { .. } => {}
            }
            ensure_vello_renderer(
                backend,
                options,
                *device_identity,
                RuntimeOperation::SurfaceRendering,
            )?;
            let resizing = lifecycle.resize_state();
            let render_start = Instant::now();
            backend.render_vello_to_texture(
                *device_identity,
                RuntimeOperation::SurfaceRendering,
                scene,
                &native.target_view,
                &vello::RenderParams {
                    width: native.config.width,
                    height: native.config.height,
                    base_color: parameters.base_color.into(),
                    antialiasing_method: options.antialiasing().into(),
                },
            )?;
            let render_time = render_start.elapsed();

            let present_start = Instant::now();
            let surface_texture = match native.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                    *lifecycle = PresentedLifecycle::Ready { resizing };
                    surface_texture
                }
                wgpu::CurrentSurfaceTexture::Outdated
                | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    backend.configure_presented_surface(*device_identity, native)?;
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

            let (device, queue) =
                backend.device_queue(*device_identity, RuntimeOperation::SurfaceRendering)?;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist surface blit"),
            });
            native.blitter.copy(
                device,
                &mut encoder,
                &native.target_view,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
            queue.submit([encoder.finish()]);
            surface_texture.present();
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
    operation: RuntimeOperation,
) -> Result<()> {
    if !backend.has_device_slot(device_identity) {
        return Err(Error::new(
            BackendErrorCode::RendererCreateFailed,
            "Vello device slot is unavailable",
        ));
    }
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    let renderer_is_ready = backend
        .ready_state_mut(
            device_identity,
            operation,
            BackendErrorCode::RendererCreateFailed,
            "Vello device resources are unavailable before renderer creation",
        )?
        .renderer
        .is_some();
    if !renderer_is_ready {
        let renderer = {
            let ready = backend.ready_state_mut(
                device_identity,
                operation,
                BackendErrorCode::RendererCreateFailed,
                "Vello device resources are unavailable before renderer creation",
            )?;
            vello::Renderer::new(&ready.device, vello_renderer_options(options))
        };
        #[cfg(test)]
        backend.inject_terminal_signal_after_renderer_creation_for_test(device_identity);
        if let Some(error) = backend.terminal_error(device_identity, operation) {
            return Err(error);
        }
        let renderer = renderer.map_err(|source| {
            Error::new(
                BackendErrorCode::RendererCreateFailed,
                "failed to create Vello renderer",
            )
            .with_source(source)
        })?;
        backend
            .ready_state_mut(
                device_identity,
                operation,
                BackendErrorCode::RendererCreateFailed,
                "Vello device resources are unavailable before renderer creation",
            )?
            .renderer = Some(renderer);
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
