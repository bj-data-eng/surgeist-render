use super::{
    Backend,
    device::{
        DeviceCapabilities, DeviceSignal, DeviceSlotIdentity, DeviceTerminalSignal,
        ReadyDeviceState,
    },
};
use crate::{
    capability::EffectPrecisionCapabilities,
    error::{
        BackendErrorCode, DeviceLossReason, Error, GpuFaultKind, Result,
        RuntimeCapabilityUnavailableReason, RuntimeOperation,
    },
    renderer::ResourceCacheBudget,
    resource::{
        ManagerIdentity, ResourceManager, ResourceManagerObservationForTest, WorkingFormat,
    },
    shader::{DevicePassCache, DevicePassCacheCountsForTest},
    vello_engine::VelloEngineState,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

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
    super::device::require_presented_device_identity(identity)
}

impl DeviceSlotIdentity {
    pub(crate) fn mark_stale_for_test(&mut self) {
        self.generation = self.generation.checked_add(1).unwrap();
    }
}

impl Backend {
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
