use super::Backend;
use crate::{
    capability::{AvailableRuntimeCapabilities, EffectPrecisionCapabilities},
    error::{
        BackendErrorCode, DeviceLossReason, Error, GpuFaultKind, Result,
        RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    },
    geometry::PhysicalSize,
    gpu_transaction::GpuOperationStage,
    renderer::{EffectQualityPolicy, ResourceCacheBudget},
    resource::{ResourceManager, WorkingFormat},
    shader::DevicePassCache,
    surface::Format,
    vello_engine::VelloEngineState,
};
use std::sync::{Arc, Mutex};

#[cfg(test)]
use crate::{
    resource::{ManagerIdentity, ResourceManagerObservationForTest},
    shader::DevicePassCacheCountsForTest,
};
#[cfg(test)]
use std::{
    sync::{Condvar, Weak},
    time::Duration,
};

pub(crate) struct DeviceState {
    pub(super) generation: u64,
    lifecycle: DeviceLifecycle,
    pub(super) capabilities: DeviceCapabilities,
    pub(super) signal: Arc<DeviceSignal>,
    pub(super) next_operation_generation: u64,
}

pub(super) struct ReadyDeviceState {
    pub(super) adapter: wgpu::Adapter,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) engine: VelloEngineState,
    pub(super) resources: ResourceManager,
    pub(super) pass_cache: DevicePassCache,
    #[cfg(test)]
    pub(super) drop_witness: Arc<()>,
}

impl Drop for ReadyDeviceState {
    fn drop(&mut self) {
        self.pass_cache.clear();
    }
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
    resources: &'ready ResourceManager,
    pass_cache: &'ready DevicePassCache,
    drop_witness: ReadyDeviceStateDropWitnessForTest,
}

#[cfg(test)]
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

    pub(crate) fn drop_witness_for_test(&self) -> ReadyDeviceStateDropWitnessForTest {
        self.drop_witness.clone()
    }
}

#[cfg(test)]
impl ReadyDeviceState {
    fn seed_pass_cache_sampler_for_test(&mut self) -> DevicePassCacheCountsForTest {
        let Self {
            device, pass_cache, ..
        } = self;
        pass_cache.seed_sampler_for_test(device)
    }

    fn borrow_for_test(&self) -> ReadyDeviceStateBorrowForTest<'_> {
        ReadyDeviceStateBorrowForTest {
            adapter: &self.adapter,
            device: &self.device,
            queue: &self.queue,
            engine: &self.engine,
            resources: &self.resources,
            pass_cache: &self.pass_cache,
            drop_witness: ReadyDeviceStateDropWitnessForTest::from_ready_bundle(&self.drop_witness),
        }
    }
}
impl DeviceState {
    async fn new(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        resource_cache_budget: ResourceCacheBudget,
    ) -> Result<Self> {
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
        let resources = ResourceManager::new(resource_cache_budget);
        #[cfg(test)]
        debug_assert!(resources.is_empty_for_test());
        Ok(Self {
            generation: 0,
            lifecycle: DeviceLifecycle::Ready(Box::new(ReadyDeviceState {
                adapter,
                device,
                queue,
                engine,
                resources,
                pass_cache: DevicePassCache::new(),
                #[cfg(test)]
                drop_witness: Arc::new(()),
            })),
            capabilities,
            signal,
            next_operation_generation: 0,
        })
    }

    pub(super) fn observe_terminal(&mut self) {
        let Some(terminal) = self.signal.first_terminal() else {
            return;
        };
        if matches!(&self.lifecycle, DeviceLifecycle::Ready(_)) {
            self.lifecycle = DeviceLifecycle::Terminal(terminal);
        }
    }

    pub(super) fn terminal(&mut self) -> Option<&DeviceTerminalSignal> {
        self.observe_terminal();
        match &self.lifecycle {
            DeviceLifecycle::Ready(_) => None,
            DeviceLifecycle::Terminal(terminal) => Some(terminal.as_ref()),
        }
    }

    pub(super) fn ready(&self) -> Option<&ReadyDeviceState> {
        match &self.lifecycle {
            DeviceLifecycle::Ready(ready) => Some(ready),
            DeviceLifecycle::Terminal(_) => None,
        }
    }

    fn ready_after_observing_terminal(&mut self) -> Option<&ReadyDeviceState> {
        self.observe_terminal();
        self.ready()
    }

    pub(super) fn ready_mut(&mut self) -> Option<&mut ReadyDeviceState> {
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
    high_precision_features: wgpu::TextureFormatFeatures,
    reduced_precision_features: wgpu::TextureFormatFeatures,
    max_effect_texture_dimension_2d: u32,
}

impl DeviceCapabilities {
    pub(crate) fn from_device(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let high_precision = WorkingFormat::HighPrecision;
        let reduced_precision = WorkingFormat::ReducedPrecision;
        let high_precision_features =
            adapter.get_texture_format_features(high_precision.texture_format());
        let reduced_precision_features =
            adapter.get_texture_format_features(reduced_precision.texture_format());
        Self {
            high_precision: supports_effect_texture_format(high_precision, high_precision_features),
            reduced_precision: supports_effect_texture_format(
                reduced_precision,
                reduced_precision_features,
            ),
            high_precision_features,
            reduced_precision_features,
            max_effect_texture_dimension_2d: device.limits().max_texture_dimension_2d,
        }
    }

    pub(crate) const fn runtime_report(
        self,
        surface_format: Format,
    ) -> AvailableRuntimeCapabilities {
        AvailableRuntimeCapabilities::new(
            surface_format,
            EffectPrecisionCapabilities::new(self.high_precision, self.reduced_precision),
            self.max_effect_texture_dimension_2d,
        )
    }

    pub(crate) fn resolve_effect_working_format(
        &self,
        policy: EffectQualityPolicy,
    ) -> Result<WorkingFormat> {
        if self.high_precision {
            return Ok(WorkingFormat::HighPrecision);
        }
        if policy == EffectQualityPolicy::AllowReducedPrecision && self.reduced_precision {
            return Ok(WorkingFormat::ReducedPrecision);
        }

        Err(Error::runtime_unavailable(
            RuntimeOperation::EffectRendering,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { policy },
            "the selected GPU has no effect working format permitted by the configured quality policy",
        ))
    }

    pub(crate) fn validate_supported_working_format(
        &self,
        working_format: WorkingFormat,
    ) -> Result<()> {
        self.validate_effect_texture_allocation(
            PhysicalSize::new(1, 1),
            Some(working_format),
            working_format.texture_format(),
            working_format.required_usages(),
        )
    }

    pub(super) fn for_selected_working_format(
        mut self,
        working_format: WorkingFormat,
    ) -> Result<Self> {
        self.validate_supported_working_format(working_format)?;
        if working_format == WorkingFormat::ReducedPrecision {
            self.high_precision = false;
        }
        Ok(self)
    }

    pub(crate) fn validate_effect_texture_extent(&self, requested: PhysicalSize) -> Result<()> {
        if requested.width() == 0 || requested.height() == 0 {
            return Ok(());
        }
        let maximum = self.max_effect_texture_dimension_2d;
        if requested.width() <= maximum && requested.height() <= maximum {
            return Ok(());
        }

        Err(Error::runtime_unavailable(
            RuntimeOperation::EffectTextureAllocation,
            RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { requested, maximum },
            format!(
                "effect texture extent {}x{} exceeds the selected device limit of {maximum}",
                requested.width(),
                requested.height(),
            ),
        ))
    }

    pub(crate) fn validate_effect_texture_allocation(
        &self,
        requested: PhysicalSize,
        working_format: Option<WorkingFormat>,
        texture_format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Result<()> {
        self.validate_effect_texture_extent(requested)?;
        let (features, policy) = match texture_format {
            wgpu::TextureFormat::Rgba16Float => (
                self.high_precision_features,
                EffectQualityPolicy::RequireHighPrecision,
            ),
            wgpu::TextureFormat::Rgba8Unorm => (
                self.reduced_precision_features,
                EffectQualityPolicy::AllowReducedPrecision,
            ),
            _ => {
                return Err(Error::invalid_value(
                    "effect texture format",
                    format!("{texture_format:?}"),
                    "must be Rgba16Float or Rgba8Unorm",
                ));
            }
        };
        if working_format.is_some_and(|format| format.texture_format() != texture_format) {
            return Err(Error::invalid_value(
                "effect working texture format",
                format!("{working_format:?} as {texture_format:?}"),
                "must use the underlying format selected by WorkingFormat",
            ));
        }
        let complete_working_support = working_format.is_none_or(|format| {
            format.is_supported_by(features)
                && match format {
                    WorkingFormat::HighPrecision => self.high_precision,
                    WorkingFormat::ReducedPrecision => self.reduced_precision,
                }
        });
        let exact_usage_support = features.allowed_usages.contains(usage);
        let filterable_support = !usage.contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE);
        if complete_working_support && exact_usage_support && filterable_support {
            return Ok(());
        }

        Err(Error::runtime_unavailable(
            RuntimeOperation::EffectRendering,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { policy },
            format!(
                "the selected GPU does not support {texture_format:?} with exact usage {usage:?}"
            ),
        ))
    }

    #[cfg(test)]
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

fn supports_effect_texture_format(
    working_format: WorkingFormat,
    features: wgpu::TextureFormatFeatures,
) -> bool {
    working_format.is_supported_by(features)
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

    pub(crate) fn has_active_operation(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active_operation_generation
            .is_some()
    }

    /// Linearizes public state with terminal signal delivery.
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
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
pub(super) fn require_presented_device_identity(
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
    pub(super) generation: u64,
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
    pub(super) fn compatible_ready_device(
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

    pub(super) async fn select_presented_device(
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

    pub(super) async fn new_device(
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
        let state = DeviceState::new(adapter, device, queue, self.resource_cache_budget).await?;
        let slot = self.device_states.len();
        let generation = state.generation;
        self.device_states.push(state);
        Ok(Some(DeviceSlotIdentity::new(slot, generation)))
    }
    pub(super) fn ready_state_mut(
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

    #[cfg(test)]
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

    pub(crate) fn gpu_operation_device_queue(
        &mut self,
        identity: DeviceSlotIdentity,
        operation: RuntimeOperation,
        stage: GpuOperationStage,
    ) -> Result<(&wgpu::Device, &wgpu::Queue)> {
        let ready = self.ready_state_mut(
            identity,
            operation,
            stage.error_code(),
            "GPU device resources are unavailable for the active operation",
        )?;
        Ok((&ready.device, &ready.queue))
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(super) fn present_device_queue(
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

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
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
