mod device;
mod execute;
mod offscreen;
mod present;
#[cfg(test)]
mod test_support;
mod texture;

pub(crate) use device::{
    DeviceCapabilities, DeviceSignal, DeviceSlotIdentity, DeviceState, DeviceTerminalSignal,
};
#[cfg(any(
    test,
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) use execute::RenderTimings;
#[cfg(all(
    not(test),
    any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )
))]
pub(crate) use execute::render_exact_presented_graph_surface;
#[cfg(not(test))]
pub(crate) use execute::{ExactSurfaceGraph, render_exact_headless_graph_surface};
pub(crate) use execute::{SurfaceFrameCommit, render_internal_vello_surface};
#[cfg(test)]
pub(crate) use offscreen::offscreen_local_scene_texture_descriptor;
#[cfg(all(
    test,
    any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )
))]
pub(crate) use test_support::render_exact_presented_graph_surface;
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use test_support::{
    CompositionBlendVectorForTest, CompositionGpuVectorResultsForTest,
    CompositionMaskSamplingInputForTest, CompositionMaskSamplingVectorForTest,
    CompositionPreparedGpuVectorsForTest,
};
#[cfg(test)]
pub(crate) use test_support::{
    CustomSpineEncodingObservationForTest, ExactSurfaceGraph, TwoCaptureFailureForTest,
    render_exact_headless_graph_surface,
};
#[cfg(all(test, feature = "render-window"))]
pub(crate) use test_support::{
    DisplayFreePresentedDeviceCompatibilityForTest,
    configured_display_free_presented_surface_for_test,
    configured_display_free_presented_surface_on_device_for_test,
    discard_presented_configuration_stage_for_test, display_free_presented_surface_for_test,
    display_free_presented_surface_on_device_for_test, presented_configuration_count_for_test,
    presented_configuration_validation_failure_stage_for_test, presented_device_identity_for_test,
    presented_lifecycle_for_test, presented_observation_for_test,
    presented_observation_handle_for_test, presented_resource_id_for_test,
    presented_target_identity_for_test, require_presented_device_identity_for_test,
    select_display_free_presented_device_for_test, set_presented_acquire_outcome_for_test,
    take_last_presented_texture_for_test,
};
#[cfg(test)]
pub(crate) use test_support::{
    OffscreenLocalSceneRenderRequest, OffscreenRenderGpuContext, ReadyDeviceStateBorrowForTest,
    render_internal_vello_local_scene_to_offscreen_texture,
};
pub(crate) use texture::create_headless_texture;
#[cfg(test)]
pub(crate) use texture::create_texture;

use crate::{ResourceCacheBudget, Result};

pub(crate) struct Backend {
    instance: wgpu::Instance,
    device_states: Vec<DeviceState>,
    resource_cache_budget: ResourceCacheBudget,
}

impl Backend {
    pub(crate) fn new(resource_cache_budget: ResourceCacheBudget) -> Self {
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
            resource_cache_budget,
        }
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
}
