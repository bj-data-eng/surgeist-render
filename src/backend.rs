#[cfg(test)]
use super::gpu_transaction::{
    AfterInternalVelloSubmitCheckpointForTest, InternalVelloSubmissionObservationForTest,
};
use super::surface::{HeadlessPublication, SurfaceBackend};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedConfigurationDraft, PresentedLifecycle};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedSurface, PresentedSurfaceAcquire};
use super::vello_engine::{
    ActiveVelloEncodingScope, EncodedVelloPass, RasterParameters, TransactionEncodingState,
    TransactionTargetIntent, VelloEngineState, VelloResourceManager, scene::VelloScene,
};
#[cfg(test)]
use super::vello_engine::{PreparedVelloPass, VelloResourceManagerObservationForTest};
use super::*;
use super::{
    command::OffscreenBounds,
    geometry::physical_size,
    gpu_transaction::{GpuOperationStage, GpuOperationTransaction, InternalVelloPayload},
    texture::{
        OffscreenTextureCache, OffscreenTextureHandle, TextureCacheKey, TextureDescriptor,
        TextureLifecycleStats, TextureUsageIntent, headless_texture_descriptor,
    },
};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Condvar, Weak},
    task::{Context, Poll, Waker},
};

#[cfg(all(test, feature = "render-window"))]
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

#[cfg(test)]
thread_local! {
    static ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST: RefCell<Option<Arc<AtomicUsize>>> = const { RefCell::new(None) };
}

#[cfg(all(test, feature = "render-window"))]
thread_local! {
    static ACTIVE_PRESENTED_CONFIGURE_CONTROL_FOR_TEST: RefCell<Option<PresentedConfigureControlForTest>> = const { RefCell::new(None) };
    static ACTIVE_DISPLAY_FREE_PREFERRED_DEVICE_INCOMPATIBILITY_FOR_TEST: RefCell<bool> = const { RefCell::new(false) };
}

#[cfg(all(test, feature = "render-window"))]
#[derive(Clone)]
enum PresentedConfigureControlForTest {
    Fail {
        scope_resolution_observed: SyncSender<()>,
    },
    Pause {
        reached: SyncSender<()>,
    },
}

/// Private deterministic control for the production Configure transaction path.
#[cfg(all(test, feature = "render-window"))]
pub(crate) struct ScopedPresentedConfigureControlForTest {
    observed: Receiver<()>,
    failure_scope_resolution: Option<Receiver<()>>,
    previous: Option<PresentedConfigureControlForTest>,
}

#[cfg(all(test, feature = "render-window"))]
impl ScopedPresentedConfigureControlForTest {
    pub(crate) fn failing() -> Self {
        let (scope_resolution_observed, observed) = sync_channel(1);
        let previous = ACTIVE_PRESENTED_CONFIGURE_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(PresentedConfigureControlForTest::Fail {
                scope_resolution_observed,
            }))
        });
        let (_unused, reached) = sync_channel(1);
        Self {
            observed: reached,
            failure_scope_resolution: Some(observed),
            previous,
        }
    }

    pub(crate) fn paused() -> Self {
        let (reached, observed) = sync_channel(1);
        let previous = ACTIVE_PRESENTED_CONFIGURE_CONTROL_FOR_TEST.with(|active| {
            active.replace(Some(PresentedConfigureControlForTest::Pause { reached }))
        });
        Self {
            observed,
            failure_scope_resolution: None,
            previous,
        }
    }

    pub(crate) fn wait_for_draft_for_test(&self, deadline: Duration) {
        self.observed
            .recv_timeout(deadline)
            .expect("the Configure transaction did not reach its bounded draft checkpoint");
    }

    pub(crate) fn scope_resolution_observed_for_test(&self) -> bool {
        self.failure_scope_resolution
            .as_ref()
            .is_some_and(|observed| observed.try_recv().is_ok())
    }
}

#[cfg(all(test, feature = "render-window"))]
impl Drop for ScopedPresentedConfigureControlForTest {
    fn drop(&mut self) {
        ACTIVE_PRESENTED_CONFIGURE_CONTROL_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

/// Models a replacement target that the installed device cannot present to.
#[cfg(all(test, feature = "render-window"))]
pub(crate) struct ScopedDisplayFreePreferredDeviceIncompatibilityForTest {
    previous: bool,
}

#[cfg(all(test, feature = "render-window"))]
impl ScopedDisplayFreePreferredDeviceIncompatibilityForTest {
    pub(crate) fn active() -> Self {
        let previous = ACTIVE_DISPLAY_FREE_PREFERRED_DEVICE_INCOMPATIBILITY_FOR_TEST
            .with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(all(test, feature = "render-window"))]
impl Drop for ScopedDisplayFreePreferredDeviceIncompatibilityForTest {
    fn drop(&mut self) {
        ACTIVE_DISPLAY_FREE_PREFERRED_DEVICE_INCOMPATIBILITY_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous;
        });
    }
}

#[cfg(test)]
pub(crate) struct ScopedOffscreenTextureAcquireObservationForTest {
    count: Arc<AtomicUsize>,
    previous: Option<Arc<AtomicUsize>>,
}

#[cfg(test)]
impl ScopedOffscreenTextureAcquireObservationForTest {
    pub(crate) fn begin() -> Self {
        let count = Arc::new(AtomicUsize::new(0));
        let previous = ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(Arc::clone(&count))));
        Self { count, previous }
    }

    pub(crate) fn acquire_count_for_test(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl Drop for ScopedOffscreenTextureAcquireObservationForTest {
    fn drop(&mut self) {
        ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
fn record_offscreen_texture_acquire_for_test() {
    ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST.with(|active| {
        if let Some(count) = active.borrow().as_ref() {
            count.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub(crate) struct Backend {
    instance: wgpu::Instance,
    device_states: Vec<DeviceState>,
}

struct InternalVelloRenderRequest<'a> {
    identity: DeviceSlotIdentity,
    operation: RuntimeOperation,
    scene: &'a VelloScene,
    target: &'a wgpu::TextureView,
    target_extent: PhysicalSize,
    base_color: Color,
    antialiasing: Antialiasing,
    target_usage: wgpu::TextureUsages,
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
    engine: VelloEngineState,
    resources: VelloResourceManager,
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

    fn ready_after_observing_terminal(&mut self) -> Option<&ReadyDeviceState> {
        self.observe_terminal();
        self.ready()
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

    /// Linearizes public state with terminal signal delivery.
    pub(crate) fn commit_if_no_terminal<T>(
        &self,
        operation: RuntimeOperation,
        commit: impl FnOnce() -> T,
    ) -> Result<T> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(terminal) = state.first_terminal.as_ref() {
            return Err(terminal.error(operation));
        }
        Ok(commit())
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
    pub(crate) fn record_loss_for_test(&self, reason: DeviceLossReason) {
        self.record(DeviceTerminalSignal::lost(
            reason,
            "test device loss".into(),
        ));
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

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
fn require_presented_device_identity(
    identity: Option<DeviceSlotIdentity>,
) -> Result<DeviceSlotIdentity> {
    identity.ok_or_else(|| {
        Error::runtime_unavailable(
            RuntimeOperation::AdapterSelection,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            "no compatible WGPU adapter is available for the presentation surface",
        )
    })
}

#[cfg(all(test, feature = "render-window"))]
pub(crate) fn require_presented_device_identity_for_test(
    identity: Option<DeviceSlotIdentity>,
) -> Result<DeviceSlotIdentity> {
    require_presented_device_identity(identity)
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
        }
    }

    fn compatible_ready_device(
        &mut self,
        preferred: Option<DeviceSlotIdentity>,
        mut supports_surface: impl FnMut(&ReadyDeviceState) -> bool,
    ) -> Option<DeviceSlotIdentity> {
        if let Some(identity) = preferred
            && let Some(state) = self.device_states.get_mut(identity.slot())
            && state.generation == identity.generation
            && let Some(ready) = state.ready_after_observing_terminal()
            && supports_surface(ready)
        {
            return Some(identity);
        }
        self.device_states
            .iter_mut()
            .enumerate()
            .find_map(|(slot, state)| {
                let generation = state.generation;
                state
                    .ready_after_observing_terminal()
                    .filter(|ready| supports_surface(ready))
                    .map(|_| DeviceSlotIdentity::new(slot, generation))
            })
    }

    async fn select_presented_device(
        &mut self,
        surface: &wgpu::Surface<'_>,
        preferred: Option<DeviceSlotIdentity>,
    ) -> Result<Option<DeviceSlotIdentity>> {
        if let Some(identity) = self.compatible_ready_device(preferred, |ready| {
            ready.adapter.is_surface_supported(surface)
        }) {
            return Ok(Some(identity));
        }
        self.new_device(Some(surface)).await
    }

    pub(crate) async fn select_device(
        &mut self,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Option<DeviceSlotIdentity>> {
        if let Some(surface) = compatible_surface {
            return self.select_presented_device(surface, None).await;
        }
        let existing = self
            .device_states
            .first()
            .map(|state| DeviceSlotIdentity::new(0, state.generation));
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
        preferred: Option<DeviceSlotIdentity>,
        operation: RuntimeOperation,
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
        let identity = require_presented_device_identity(
            self.select_presented_device(&surface, preferred).await?,
        )?;
        let ready = self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::SurfaceCreateFailed,
            "the selected presentation device is unavailable",
        )?;
        let presented = PresentedSurface::new(surface, &ready.adapter)?;
        Ok((presented, identity))
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) async fn create_display_free_presented_surface_for_test(
        &mut self,
        preferred: Option<DeviceSlotIdentity>,
        operation: RuntimeOperation,
    ) -> Result<(PresentedSurface, DeviceSlotIdentity)> {
        let incompatible_preferred = ACTIVE_DISPLAY_FREE_PREFERRED_DEVICE_INCOMPATIBILITY_FOR_TEST
            .with(|active| {
                if !*active.borrow() {
                    return None;
                }
                let identity = preferred?;
                let state = self.device_states.get(identity.slot())?;
                (state.generation == identity.generation)
                    .then(|| state.ready())
                    .flatten()
                    .map(|ready| Arc::clone(&ready.drop_witness))
            });
        let identity = if let Some(identity) = self.compatible_ready_device(preferred, |ready| {
            !incompatible_preferred
                .as_ref()
                .is_some_and(|incompatible| Arc::ptr_eq(incompatible, &ready.drop_witness))
        }) {
            Some(identity)
        } else {
            self.new_device(None).await?
        };
        let identity = require_presented_device_identity(identity)?;
        self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::SurfaceCreateFailed,
            "the selected presentation device is unavailable",
        )?;
        Ok((PresentedSurface::display_free_for_test(), identity))
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

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn present_device_queue(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Result<(&wgpu::Device, &wgpu::Queue)> {
        let ready = self.ready_state_mut(
            identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::PresentFailed,
            "presented device resources are unavailable before output submission",
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

    async fn render_internal_vello_to_texture(
        &mut self,
        transaction: GpuOperationTransaction,
        request: InternalVelloRenderRequest<'_>,
    ) -> Result<()> {
        let prepared = request.scene.prepare_raster(RasterParameters::try_new(
            request.target_extent,
            peniko::Color::from(request.base_color),
            request.antialiasing,
        )?)?;
        let ready = self.ready_state_mut(
            request.identity,
            request.operation,
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
                request.target,
                TransactionTargetIntent::new(
                    request.target_extent,
                    wgpu::TextureFormat::Rgba8Unorm,
                    request.target_usage,
                ),
            );
            match prepared.encode_into(&ready.engine, &mut encoding) {
                Ok(encoded) => encoded,
                Err(failure) => {
                    let (error, aborted) = failure.into_error_and_aborted_resources();
                    ready.resources.record_aborted_resources(aborted);
                    return Err(error);
                }
            }
        };
        let (lease, logical_pass) = encoded.into_resources_and_logical_pass();
        let lease = match scope.finish_with_lease(lease).await {
            Ok(lease) => lease,
            Err(failure) => {
                let (error, aborted) = failure.into_error_and_aborted_resources();
                ready.resources.record_aborted_resources(aborted);
                return Err(error);
            }
        };
        let payload = InternalVelloPayload::new(
            command_encoder.finish(),
            ready.resources.pending_commit(lease),
            logical_pass,
        );
        transaction
            .submit_internal_vello(&ready.device, &ready.queue, payload, request.operation)
            .await
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(crate) async fn configure_presented_surface(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
        surface: &PresentedSurface,
        physical_size: PhysicalSize,
        present_mode: wgpu::PresentMode,
    ) -> Result<PresentedConfigurationDraft> {
        let transaction =
            self.begin_gpu_operation(identity, GpuOperationStage::Configure, operation)?;
        let ready = self.ready_state_mut(
            identity,
            operation,
            BackendErrorCode::SurfaceConfigureFailed,
            "presented device resources are unavailable before configuration",
        )?;
        let draft = surface.configure_draft(&ready.device, physical_size, present_mode);
        #[cfg(all(test, feature = "render-window"))]
        let control =
            ACTIVE_PRESENTED_CONFIGURE_CONTROL_FOR_TEST.with(|active| active.borrow().clone());
        #[cfg(all(test, feature = "render-window"))]
        if let Some(PresentedConfigureControlForTest::Fail { .. }) = &control {
            let _ = ready.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Surgeist test-injected Configure validation failure"),
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
        #[cfg(all(test, feature = "render-window"))]
        if let Some(PresentedConfigureControlForTest::Pause { reached }) = &control {
            reached
                .send(())
                .expect("the Configure test must observe the draft checkpoint");
            std::future::pending::<()>().await;
        }
        let result = transaction.finish(operation).await;
        #[cfg(all(test, feature = "render-window"))]
        if let Some(PresentedConfigureControlForTest::Fail {
            scope_resolution_observed,
        }) = control
        {
            let _ = scope_resolution_observed.send(());
        }
        result?;
        Ok(draft)
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
            match prepared.encode_into(engine, &mut encoding) {
                Ok(lease) => lease,
                Err(failure) => {
                    let (error, aborted) = failure.into_error_and_aborted_resources();
                    resources.record_aborted_resources(aborted);
                    return Err(error);
                }
            }
        };
        let (lease, logical_pass) = encoded.into_resources_and_logical_pass();
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
            logical_pass,
            observation.clone(),
        );
        transaction
            .submit_internal_vello(device, queue, payload, RuntimeOperation::SurfaceRendering)
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
            match prepared.encode_into(engine, &mut encoding) {
                Ok(lease) => lease,
                Err(failure) => {
                    let (error, aborted) = failure.into_error_and_aborted_resources();
                    resources.record_aborted_resources(aborted);
                    return Err(error);
                }
            }
        };
        let (lease, logical_pass) = encoded.into_resources_and_logical_pass();
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
            logical_pass,
            checkpoint,
        );
        let mut submission = Box::pin(transaction.submit_internal_vello(
            device,
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

    pub(crate) fn publication_signal(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
    ) -> Result<Arc<DeviceSignal>> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot disappeared before frame publication",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before frame publication",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(operation));
        }
        Ok(Arc::clone(&state.signal))
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

    #[cfg(test)]
    pub(crate) fn device_signal_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
    ) -> Option<Arc<DeviceSignal>> {
        self.device_states
            .get(identity.slot())
            .filter(|state| state.generation == identity.generation)
            .map(|state| Arc::clone(&state.signal))
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
        state.ready().is_none()
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
    pub(crate) async fn add_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        self.new_device(None).await?.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
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
        #[cfg(test)]
        record_offscreen_texture_acquire_for_test();
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
pub(crate) async fn render_internal_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    cache: &mut OffscreenTextureResourceCache,
    scene: &VelloScene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let descriptor =
        offscreen_local_scene_texture_descriptor(request.bounds, request.scale, request.format)?;
    let Some(context) = context else {
        return Err(Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            "offscreen Vello local scene rendering requires an available wgpu device context",
        ));
    };
    let resource = {
        let (device, _) = context
            .backend
            .device_queue(context.device_identity, RuntimeOperation::SurfaceRendering)?;
        cache.acquire(device, request.bounds, descriptor)?
    };
    let render_start = Instant::now();
    let transaction = context.backend.begin_gpu_operation(
        context.device_identity,
        GpuOperationStage::Render,
        RuntimeOperation::SurfaceRendering,
    )?;
    let result = context
        .backend
        .render_internal_vello_to_texture(
            transaction,
            InternalVelloRenderRequest {
                identity: context.device_identity,
                operation: RuntimeOperation::SurfaceRendering,
                scene,
                target: &resource.view,
                target_extent: resource.target.descriptor().physical_size(),
                base_color: request.parameters.base_color,
                antialiasing: options.antialiasing(),
                target_usage: resource.target.descriptor().wgpu_usage(),
            },
        )
        .await;
    context
        .backend
        .observe_device_terminal(context.device_identity);
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
}

impl SurfaceFrameCommit {
    fn without_headless_publication(timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: None,
        }
    }

    fn headless(publication: HeadlessPublication, timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: Some(publication),
        }
    }

    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    pub(crate) fn commit(self, surface: &mut Surface) {
        if let Some(publication) = self.headless_publication {
            surface.commit_headless_publication(publication);
        }
    }
}

pub(crate) async fn render_internal_vello_surface(
    backend: &mut Backend,
    transaction: GpuOperationTransaction,
    surface: &mut Surface,
    scene: &VelloScene,
    parameters: Parameters,
    antialiasing: Antialiasing,
) -> Result<SurfaceFrameCommit> {
    match &mut surface.backend {
        SurfaceBackend::ContractOnly { .. } => Ok(
            SurfaceFrameCommit::without_headless_publication(RenderTimings::default()),
        ),
        SurfaceBackend::Headless {
            device_identity,
            physical_size,
            ..
        } => {
            if physical_size.width() == 0 || physical_size.height() == 0 {
                return Ok(SurfaceFrameCommit::without_headless_publication(
                    RenderTimings::default(),
                ));
            }
            let (texture, view) = backend.create_headless_surface_texture(
                *device_identity,
                *physical_size,
                surface.options.format,
            )?;
            let render_start = Instant::now();
            backend
                .render_internal_vello_to_texture(
                    transaction,
                    InternalVelloRenderRequest {
                        identity: *device_identity,
                        operation: RuntimeOperation::SurfaceRendering,
                        scene,
                        target: &view,
                        target_extent: *physical_size,
                        base_color: parameters.base_color,
                        antialiasing,
                        target_usage: wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC,
                    },
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
        SurfaceBackend::Presented {
            surface: native,
            device_identity,
            state,
        } => {
            match state.lifecycle() {
                PresentedLifecycle::NonRenderable { .. } | PresentedLifecycle::Lost => {
                    return Ok(SurfaceFrameCommit::without_headless_publication(
                        RenderTimings::default(),
                    ));
                }
                PresentedLifecycle::ResizePending { .. } => {
                    return Err(Error::new(
                        BackendErrorCode::SurfaceConfigureFailed,
                        "presented rendering started before configuration committed",
                    ));
                }
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::Occluded { .. } => {}
            }

            let resources = native.committed().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::SurfaceConfigureFailed,
                    "ready presented lifecycle has no committed target resources",
                )
            })?;
            let _ = &resources.target_texture;
            let render_start = Instant::now();
            backend
                .render_internal_vello_to_texture(
                    transaction,
                    InternalVelloRenderRequest {
                        identity: *device_identity,
                        operation: RuntimeOperation::SurfaceRendering,
                        scene,
                        target: &resources.target_view,
                        target_extent: PhysicalSize::new(
                            resources.config.width,
                            resources.config.height,
                        ),
                        base_color: parameters.base_color,
                        antialiasing,
                        target_usage: wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                    },
                )
                .await?;
            let render_time = render_start.elapsed();

            let present_start = Instant::now();
            let transaction = backend.begin_gpu_operation(
                *device_identity,
                GpuOperationStage::Present,
                RuntimeOperation::SurfaceRendering,
            )?;
            let (device, queue) = backend.present_device_queue(*device_identity)?;
            let surface_texture = match native.acquire_texture(device) {
                PresentedSurfaceAcquire::Success(surface_texture) => surface_texture,
                PresentedSurfaceAcquire::Suboptimal(surface_texture) => {
                    drop(surface_texture);
                    let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
                    state.mark_configuration_pending();
                    scope_result?;
                    return Err(Error::new(
                        BackendErrorCode::SurfaceOutdated,
                        "surface is suboptimal and requires reconfiguration",
                    ));
                }
                PresentedSurfaceAcquire::Outdated => {
                    let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
                    state.mark_configuration_pending();
                    scope_result?;
                    return Err(Error::new(
                        BackendErrorCode::SurfaceOutdated,
                        "surface is outdated and requires reconfiguration",
                    ));
                }
                PresentedSurfaceAcquire::Occluded => {
                    let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
                    state.mark_occluded();
                    scope_result?;
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::SurfaceRendering,
                        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                            state: RenderSurfaceAvailability::Occluded,
                        },
                        "surface is occluded",
                    ));
                }
                PresentedSurfaceAcquire::Timeout => {
                    transaction
                        .finish(RuntimeOperation::SurfaceRendering)
                        .await?;
                    return Err(Error::new(
                        BackendErrorCode::SurfaceTimeout,
                        "timed out acquiring surface texture",
                    ));
                }
                PresentedSurfaceAcquire::Lost => {
                    let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
                    state.mark_lost();
                    scope_result?;
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::SurfaceRendering,
                        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                            state: RenderSurfaceAvailability::Lost,
                        },
                        "surface was lost",
                    ));
                }
                PresentedSurfaceAcquire::Validation => {
                    transaction
                        .finish(RuntimeOperation::SurfaceRendering)
                        .await?;
                    return Err(Error::new(
                        BackendErrorCode::PresentFailed,
                        "surface texture validation failed",
                    ));
                }
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist surface blit"),
            });
            let surface_view = surface_texture.create_view();
            resources
                .blitter
                .copy(device, &mut encoder, &resources.target_view, &surface_view);
            transaction
                .submit_command_buffer_with_host_effect(
                    queue,
                    encoder.finish(),
                    || surface_texture.present(),
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
            Ok(SurfaceFrameCommit::without_headless_publication(
                RenderTimings {
                    render_time,
                    present_time: present_start.elapsed(),
                },
            ))
        }
    }
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

    ImageBuffer::try_new(physical_size, rgba)
}
