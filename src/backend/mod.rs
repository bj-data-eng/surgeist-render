mod device;

#[cfg(test)]
use device::ReadyDeviceState;
#[cfg(test)]
pub(crate) use device::ReadyDeviceStateBorrowForTest;
#[cfg(test)]
#[expect(
    unused_imports,
    reason = "preserves the crate-visible device drop-witness path until T02"
)]
pub(crate) use device::ReadyDeviceStateDropWitnessForTest;
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use device::require_presented_device_identity;
#[cfg(all(test, feature = "render-window"))]
pub(crate) use device::require_presented_device_identity_for_test;
pub(crate) use device::{
    DeviceCapabilities, DeviceSignal, DeviceSlotIdentity, DeviceState, DeviceTerminalSignal,
};

#[cfg(test)]
use super::gpu_transaction::test_support::{
    InternalVelloSubmissionActionForTest, InternalVelloSubmissionObservationForTest,
    InternalVelloSubmissionOutcomeForTest, finish_vello_resources_without_submission_for_test,
    hold_internal_vello_after_submit_for_test, submit_internal_vello_observed_for_test,
    vello_accounting_failure_after_submission_for_test,
    vello_scope_failure_after_submission_for_test,
};
use super::pass::{
    BasePreparableGraph, CompositionPreparableGraph, EncodedGpuGraphActivity,
    GraphExternalOutputView, LoweredGraphPlan, PreparedGraph,
};
#[cfg(test)]
use super::pass::{ColorFilterPreparableGraph, SpatialFilterPreparableGraph};
#[cfg(test)]
use super::pass::{CorePassCacheRequestsForTest, LayerCompositeCacheRequestsForTest};
use super::resource::{FrameCleanup, WorkingFormat};
#[cfg(test)]
use super::resource::{
    FrameResourceScope, ResourceIdentity, ResourceLease, ResourceManagerObservationForTest,
};
#[cfg(test)]
use super::shader::{
    ColorFilterOperationBufferLimits, DevicePassCache, DevicePassCacheCountsForTest,
};
use super::stats::GpuGraphStatsObservation;
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{
    AcquiredPresentedSurfaceTexture, PresentedResourceBundle, PresentedSurface,
    PresentedSurfaceAcquire, PresentedSurfaceState,
};
use super::surface::{HeadlessPublication, SurfaceBackend};
#[cfg(test)]
use super::surface::{HeadlessResources, RendererIdentity};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedConfigurationDraft, PresentedLifecycle};
use super::vello_engine::{
    ActiveVelloEncodingScope, EncodedVelloPass, RasterParameters, TransactionEncodingState,
    TransactionTargetIntent, scene::VelloScene,
};
#[cfg(test)]
use super::vello_engine::{PreparedVelloPass, VelloAtlasOutcome};
use super::*;
#[cfg(test)]
use super::{command::OffscreenBounds, geometry::physical_size, texture::EffectTextureDescriptor};
use super::{
    gpu_transaction::{
        GpuOperationStage, GpuOperationTransaction, GraphOutputCommit, GraphSubmissionPayload,
        InternalVelloPayload,
    },
    shader::ProvisionalDevicePassCacheUpdate,
    texture::{TextureDescriptor, headless_texture_descriptor},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(test)]
use std::{
    cell::RefCell,
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
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
    resource_cache_budget: ResourceCacheBudget,
}

/// One validated exact graph selected for atomic surface execution.
#[must_use = "an exact surface graph must enter its GPU transaction"]
pub(crate) enum ExactSurfaceGraph {
    Base(BasePreparableGraph),
    Composition(CompositionPreparableGraph),
    #[cfg(test)]
    ColorFilter(ColorFilterPreparableGraph),
    #[cfg(test)]
    SpatialFilter(SpatialFilterPreparableGraph),
    Backdrop(super::pass::BackdropPreparableGraph),
}

impl ExactSurfaceGraph {
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        match self {
            Self::Base(preparable) => preparable.working_format(),
            Self::Composition(preparable) => preparable.working_format(),
            #[cfg(test)]
            Self::ColorFilter(preparable) => preparable.working_format(),
            #[cfg(test)]
            Self::SpatialFilter(preparable) => preparable.working_format(),
            Self::Backdrop(preparable) => preparable.working_format(),
        }
    }

    pub(crate) const fn output_format(&self) -> Format {
        match self {
            Self::Base(preparable) => preparable.output_format(),
            Self::Composition(preparable) => preparable.output_format(),
            #[cfg(test)]
            Self::ColorFilter(preparable) => preparable.output_format(),
            #[cfg(test)]
            Self::SpatialFilter(preparable) => preparable.output_format(),
            Self::Backdrop(preparable) => preparable.output_format(),
        }
    }

    fn known_output_extent(&self) -> Result<Option<PhysicalSize>> {
        match self {
            Self::Base(preparable) => preparable.output_extent().map(Some),
            Self::Composition(_) => Ok(None),
            #[cfg(test)]
            Self::ColorFilter(_) => Ok(None),
            #[cfg(test)]
            Self::SpatialFilter(_) => Ok(None),
            Self::Backdrop(preparable) => preparable.output_extent().map(Some),
        }
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
    pub(crate) pass_order: Vec<super::pass::SpatialFilterPassTagForTest>,
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
            super::shader::ShaderCompositePathKey::Normal => {
                has_normal = true;
                normal_uses_fixed_blend &= objects.uses_fixed_source_over_blend();
                0
            }
            super::shader::ShaderCompositePathKey::DestinationSampling => {
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
    path: super::shader::ShaderCompositePathKey,
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
        (draw.path == super::shader::ShaderCompositePathKey::DestinationSampling).then(|| {
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
    let base = if draw.path == super::shader::ShaderCompositePathKey::Normal {
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
    super::shader::composite_parameter_bytes_for_gpu_vector_for_test(
        super::shader::CompositeParameterGpuVectorFactsForTest {
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
    let spatial_bytes = super::pass::pass_spatial_uniform_bytes_for_test(
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
    summary: &super::pass::CustomSpineEncodingSummary,
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
    summary: &super::pass::CustomSpineEncodingSummary,
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
    summary: &super::pass::CustomSpineEncodingSummary,
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
    summary: super::pass::CustomSpineEncodingSummary,
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
        format: Format,
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
        Ok((PresentedSurface::display_free_for_test(format), identity))
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
        {
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
                super::vello_engine::PendingVelloResourceCommit::new(lease),
                logical_pass,
            );
            transaction
                .submit_internal_vello(&ready.device, &ready.queue, payload, request.operation)
                .await?;
        }
        self.commit_checked_pass_cache_update(request.identity, None, request.operation)
    }

    fn commit_checked_pass_cache_update(
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
            super::vello_engine::PendingVelloResourceCommit::new(lease),
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
            super::vello_engine::PendingVelloResourceCommit::new(lease),
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
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
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
        let prepared = PreparedGraph::try_prepare(
            lowered,
            policy,
            &capabilities,
            &ready.device,
            &ready.queue,
            &ready.resources,
            (&ready.pass_cache, realize_checked_passes),
        )?
        .with_vello_engine(&ready.engine);
        #[cfg(test)]
        let prepared = {
            let mut prepared = prepared;
            prepared.apply_color_filter_shader_failure_for_test();
            prepared
        };
        Ok(prepared)
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
            #[cfg(test)]
            ExactSurfaceGraph::ColorFilter(preparable) => PreparedGraph::try_prepare_color_filter(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            #[cfg(test)]
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
        #[cfg(test)]
        let prepared = {
            let mut prepared = prepared;
            prepared.apply_color_filter_shader_failure_for_test();
            prepared
        };
        Ok(prepared)
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
                path: super::shader::ShaderCompositePathKey::Normal,
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
                    super::shader::ShaderCompositePathKey::Normal
                } else {
                    super::shader::ShaderCompositePathKey::DestinationSampling
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
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
        let graph = super::frame::forced_base_graph_for_test(commands, context)?;
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<OrderedColorFilterGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "color-filter graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_filter_graph_for_test(filters, commands, context)?;
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<SpatialFilterGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "spatial-filter encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_filter_graph_for_test(filters, commands, context)?;
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<BackdropGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<BackdropFailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "backdrop failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<SpatialFilterFailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "spatial-filter failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_filter_graph_for_test(filters, commands, context)?;
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
            .then(super::pass::ScopedColorFilterShaderFailureForTest::after_checked_realization);
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        let result = result.map_err(super::pass::normalize_color_filter_shader_failure_for_test);
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
                    .map(super::pass::normalize_scope_resolution_failure_for_test)
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<ColorFilterOversizedBufferPreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "color-filter limit observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_filter_graph_for_test(filters, commands, context)?;
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<CompositionOrderedGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "composition graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        donor_commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<MultipleVelloCaptureEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "multiple Vello capture coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = super::pass::two_capture_spine_lowered_for_test(
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
        let mut prepared = self.prepare_graph_resources(identity, lowered.clone(), policy)?;
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
        commands: super::command::RenderCommands,
        donor_commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
        failure: TwoCaptureFailureForTest,
    ) -> Result<TwoCaptureFailureObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "two-capture failure coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = super::pass::two_capture_spine_lowered_for_test(
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
                .map(super::pass::normalize_scope_resolution_failure_for_test)
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
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
        let graph = super::frame::forced_base_graph_for_test(commands, context)?;
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
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
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
                == super::pass::VELLO_CAPTURE_TEXTURE_USAGES,
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
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
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
        let graph = super::frame::forced_base_graph_for_test(commands, context)?;
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

        let mut first = self.prepare_graph_resources(identity, lowered.clone(), policy)?;
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
            .map(super::pass::ExecutableVelloCaptureFacts::pass);
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

        let mut retried = self.prepare_graph_resources(identity, lowered, policy)?;
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

#[cfg(test)]
pub(crate) struct OffscreenRenderGpuContext<'a> {
    backend: &'a mut Backend,
    device_identity: DeviceSlotIdentity,
}

#[cfg(test)]
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
#[cfg(test)]
pub(crate) struct OffscreenLocalSceneRenderRequest {
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
    parameters: Parameters,
}

#[cfg(test)]
impl OffscreenLocalSceneRenderRequest {
    #[must_use]
    #[cfg(test)]
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
#[cfg(test)]
pub(crate) struct OffscreenRenderTarget {
    #[cfg(test)]
    resource_identity: ResourceIdentity,
    #[cfg(test)]
    bounds: OffscreenBounds,
    descriptor: EffectTextureDescriptor,
}

#[cfg(test)]
impl OffscreenRenderTarget {
    fn new(
        _resource_identity: ResourceIdentity,
        _bounds: OffscreenBounds,
        descriptor: EffectTextureDescriptor,
    ) -> Self {
        Self {
            #[cfg(test)]
            resource_identity: _resource_identity,
            #[cfg(test)]
            bounds: _bounds,
            descriptor,
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn resource_id(self) -> u64 {
        self.resource_identity.get()
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn bounds(self) -> OffscreenBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn descriptor(self) -> EffectTextureDescriptor {
        self.descriptor
    }
}

#[must_use = "offscreen rendered texture leases must be resolved by their device resource frame"]
#[cfg(test)]
pub(crate) struct OffscreenRenderedTextureLease {
    target: OffscreenRenderTarget,
    frame_scope: Option<FrameResourceScope>,
    resource: Option<ResourceLease>,
    timings: RenderTimings,
}

#[cfg(test)]
impl fmt::Debug for OffscreenRenderedTextureLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OffscreenRenderedTextureLease")
            .field("target", &self.target)
            .field("timings", &self.timings)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl OffscreenRenderedTextureLease {
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn target(&self) -> OffscreenRenderTarget {
        self.target
    }

    pub(crate) fn texture(&self) -> Result<&wgpu::Texture> {
        self.managed_texture().map(|(texture, _)| texture)
    }

    pub(crate) fn view(&self) -> Result<&wgpu::TextureView> {
        self.managed_texture().map(|(_, view)| view)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    #[cfg(test)]
    pub(crate) fn poison_retained_byte_accounting_for_test(
        &self,
    ) -> super::resource::ResourceAccountingFault {
        self.frame_scope
            .as_ref()
            .expect("an unresolved offscreen lease must own its resource frame")
            .poison_retained_byte_accounting_for_test()
    }

    pub(crate) fn release(mut self) -> Result<()> {
        let mut frame_scope = self
            .frame_scope
            .take()
            .expect("an unresolved offscreen lease must own its resource frame");
        let resource = self
            .resource
            .take()
            .expect("an unresolved offscreen lease must own its resource lease");
        frame_scope.ensure_commit_ready(&[&resource])?;
        frame_scope.release(resource)?;
        let _ = frame_scope.finish_checked()?;
        Ok(())
    }

    fn managed_texture(&self) -> Result<(&wgpu::Texture, &wgpu::TextureView)> {
        let frame_scope = self.frame_scope.as_ref().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen texture resource frame was already resolved",
            )
        })?;
        let resource = self.resource.as_ref().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen texture resource lease was already resolved",
            )
        })?;
        frame_scope.effect_texture(resource)
    }

    fn discard(&mut self) {
        let Some(mut frame_scope) = self.frame_scope.take() else {
            return;
        };
        if let Some(resource) = self.resource.take() {
            let _ = frame_scope.discard(resource);
        }
        let _ = frame_scope.finish();
    }
}

#[cfg(test)]
impl Drop for OffscreenRenderedTextureLease {
    fn drop(&mut self) {
        self.discard();
    }
}

#[cfg(test)]
pub(crate) fn offscreen_local_scene_texture_descriptor(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<EffectTextureDescriptor> {
    let physical_size = offscreen_local_scene_physical_size(bounds, scale, format)?;
    offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, format)
}

#[cfg(test)]
fn offscreen_local_scene_physical_size(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<PhysicalSize> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    physical_size(bounds.rect().size(), scale)
}

#[cfg(test)]
fn offscreen_local_scene_texture_descriptor_for_physical_size(
    physical_size: PhysicalSize,
    format: Format,
) -> Result<EffectTextureDescriptor> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    EffectTextureDescriptor::try_capture(
        physical_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            .union(wgpu::TextureUsages::STORAGE_BINDING)
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST),
    )
}

#[cfg(test)]
pub(crate) async fn render_internal_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    scene: &VelloScene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let physical_size =
        offscreen_local_scene_physical_size(request.bounds, request.scale, request.format)?;
    let Some(context) = context else {
        offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, request.format)?;
        return Err(Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            "offscreen Vello local scene rendering requires an available wgpu device context",
        ));
    };
    let capabilities = context
        .backend
        .device_capabilities(context.device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen device capabilities are unavailable before allocation",
            )
        })?;
    capabilities.validate_effect_texture_extent(physical_size)?;
    let descriptor =
        offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, request.format)?;
    let mut rendered = {
        let ready = context.backend.ready_state_mut(
            context.device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "offscreen device resources are unavailable before allocation",
        )?;
        #[cfg(test)]
        record_offscreen_texture_acquire_for_test();
        let mut frame_scope = ready.resources.begin_frame()?;
        let resource =
            frame_scope.acquire_effect_texture(&ready.device, &capabilities, descriptor)?;
        let target =
            OffscreenRenderTarget::new(resource.resource_identity(), request.bounds, descriptor);
        OffscreenRenderedTextureLease {
            target,
            frame_scope: Some(frame_scope),
            resource: Some(resource),
            timings: RenderTimings::default(),
        }
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
                target: rendered.view()?,
                target_extent: rendered.target.descriptor().physical_size(),
                base_color: request.parameters.base_color,
                antialiasing: options.antialiasing(),
                target_usage: rendered.target.descriptor().usage(),
            },
        )
        .await;
    context
        .backend
        .observe_device_terminal(context.device_identity);
    result?;
    rendered.timings = RenderTimings {
        render_time: render_start.elapsed(),
        present_time: Duration::ZERO,
    };
    Ok(rendered)
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
    fn without_headless_publication(timings: RenderTimings) -> Self {
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

    fn headless_graph(
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
    fn presented_graph(
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
    #[cfg(test)]
    let pending_encoding =
        pending_encoding.map_err(super::pass::normalize_color_filter_shader_failure_for_test);
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
        exact_presented_graph_target(surface, &graph)?;
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
        acquire_exact_presented_graph_texture(surface, &device, prepared, transaction).await?;
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
    #[cfg(test)]
    let pending_encoding =
        pending_encoding.map_err(super::pass::normalize_color_filter_shader_failure_for_test);
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

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
fn exact_presented_graph_target(
    surface: &Surface,
    graph: &ExactSurfaceGraph,
) -> Result<(DeviceSlotIdentity, PhysicalSize, Format, WorkingFormat)> {
    let selected_working_format = graph.working_format();
    let graph_output_format = graph.output_format();
    let known_output_extent = graph.known_output_extent()?;
    let (device_identity, physical_size, output_format) = match &surface.backend {
        SurfaceBackend::Presented {
            surface: native,
            device_identity,
            state,
        } => {
            match state.lifecycle() {
                PresentedLifecycle::Ready { .. } => {}
                PresentedLifecycle::ResizePending { .. } => {
                    return Err(Error::new(
                        BackendErrorCode::SurfaceConfigureFailed,
                        "presented exact graph execution started before configuration committed",
                    ));
                }
                PresentedLifecycle::NonRenderable { .. } => {
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::SurfaceRendering,
                        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                            state: RenderSurfaceAvailability::NonRenderable,
                        },
                        "presented exact graph output is not renderable",
                    ));
                }
                PresentedLifecycle::Occluded { .. } => {
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::SurfaceRendering,
                        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                            state: RenderSurfaceAvailability::Occluded,
                        },
                        "presented exact graph output is occluded",
                    ));
                }
                PresentedLifecycle::Lost => {
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::SurfaceRendering,
                        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                            state: RenderSurfaceAvailability::Lost,
                        },
                        "presented exact graph output is lost",
                    ));
                }
            }
            let resources = native.committed().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::SurfaceConfigureFailed,
                    "ready presented exact graph output has no committed configuration",
                )
            })?;
            let physical_size = PhysicalSize::new(resources.config.width, resources.config.height);
            if resources.config.format != native.format
                || state.requested_physical_size() != physical_size
            {
                return Err(Error::new(
                    BackendErrorCode::SurfaceConfigureFailed,
                    "presented exact graph output differs from its committed configuration",
                ));
            }
            let output_format = match native.format {
                wgpu::TextureFormat::Rgba8Unorm => Format::Rgba8,
                wgpu::TextureFormat::Bgra8Unorm => Format::Bgra8,
                _ => {
                    return Err(Error::new(
                        BackendErrorCode::PresentFailed,
                        "presented exact graph output is not an advertised RGBA8 or BGRA8 format",
                    ));
                }
            };
            (*device_identity, physical_size, output_format)
        }
        SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "presented exact graph execution requires a presented surface",
            ));
        }
    };
    if physical_size.width() == 0
        || physical_size.height() == 0
        || graph_output_format != output_format
        || known_output_extent.is_some_and(|extent| extent != physical_size)
    {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "the presented graph differs from the exact eligible output",
        ));
    }
    Ok((
        device_identity,
        physical_size,
        output_format,
        selected_working_format,
    ))
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
enum PresentedAcquireFailure {
    Suboptimal,
    Outdated,
    Occluded,
    Timeout,
    Lost,
    Validation,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
async fn finish_presented_acquire_failure(
    transaction: GpuOperationTransaction,
    state: &mut PresentedSurfaceState,
    failure: PresentedAcquireFailure,
) -> Result<Error> {
    let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
    match failure {
        PresentedAcquireFailure::Suboptimal | PresentedAcquireFailure::Outdated => {
            state.mark_configuration_pending();
        }
        PresentedAcquireFailure::Occluded => state.mark_occluded(),
        PresentedAcquireFailure::Lost => state.mark_lost(),
        PresentedAcquireFailure::Timeout | PresentedAcquireFailure::Validation => {}
    }
    scope_result?;
    Ok(match failure {
        PresentedAcquireFailure::Suboptimal => Error::new(
            BackendErrorCode::SurfaceOutdated,
            "surface is suboptimal and requires reconfiguration",
        ),
        PresentedAcquireFailure::Outdated => Error::new(
            BackendErrorCode::SurfaceOutdated,
            "surface is outdated and requires reconfiguration",
        ),
        PresentedAcquireFailure::Occluded => Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                state: RenderSurfaceAvailability::Occluded,
            },
            "surface is occluded",
        ),
        PresentedAcquireFailure::Timeout => Error::new(
            BackendErrorCode::SurfaceTimeout,
            "timed out acquiring surface texture",
        ),
        PresentedAcquireFailure::Lost => Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                state: RenderSurfaceAvailability::Lost,
            },
            "surface was lost",
        ),
        PresentedAcquireFailure::Validation => Error::new(
            BackendErrorCode::PresentFailed,
            "surface texture validation failed",
        ),
    })
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
async fn acquire_exact_presented_graph_texture<'a>(
    surface: &mut Surface,
    device: &wgpu::Device,
    prepared: PreparedGraph<'a>,
    transaction: GpuOperationTransaction,
) -> Result<(
    AcquiredPresentedSurfaceTexture,
    PreparedGraph<'a>,
    GpuOperationTransaction,
)> {
    let mut prepared = Some(prepared);
    let mut transaction = Some(transaction);
    let acquired = match &mut surface.backend {
        SurfaceBackend::Presented {
            surface: native,
            state,
            ..
        } => match native.acquire_texture(device) {
            PresentedSurfaceAcquire::Success(acquired) => acquired,
            PresentedSurfaceAcquire::Suboptimal(acquired) => {
                drop(acquired);
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Suboptimal,
                )
                .await?);
            }
            PresentedSurfaceAcquire::Outdated => {
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Outdated,
                )
                .await?);
            }
            PresentedSurfaceAcquire::Occluded => {
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Occluded,
                )
                .await?);
            }
            PresentedSurfaceAcquire::Timeout => {
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Timeout,
                )
                .await?);
            }
            PresentedSurfaceAcquire::Lost => {
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Lost,
                )
                .await?);
            }
            PresentedSurfaceAcquire::Validation => {
                drop(prepared.take());
                return Err(finish_presented_acquire_failure(
                    transaction
                        .take()
                        .expect("presented transaction must remain available"),
                    state,
                    PresentedAcquireFailure::Validation,
                )
                .await?);
            }
        },
        SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => {
            unreachable!("presented exact graph output changed after eligibility validation")
        }
    };
    Ok((
        acquired,
        prepared
            .take()
            .expect("prepared graph must remain available after successful acquire"),
        transaction
            .take()
            .expect("presented transaction must remain available after successful acquire"),
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
            InternalVelloRenderRequest {
                identity: device_identity,
                operation: RuntimeOperation::SurfaceRendering,
                scene: frame.scene,
                target: &view,
                target_extent: physical_size,
                base_color: frame.parameters.base_color,
                antialiasing: frame.antialiasing,
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
async fn render_internal_vello_presented(
    backend: &mut Backend,
    transaction: GpuOperationTransaction,
    native: &mut PresentedSurface,
    device_identity: DeviceSlotIdentity,
    state: &mut PresentedSurfaceState,
    frame: InternalVelloFrameParameters<'_>,
) -> Result<SurfaceFrameCommit> {
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
                identity: device_identity,
                operation: RuntimeOperation::SurfaceRendering,
                scene: frame.scene,
                target: &resources.target_view,
                target_extent: PhysicalSize::new(resources.config.width, resources.config.height),
                base_color: frame.parameters.base_color,
                antialiasing: frame.antialiasing,
                target_usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            },
        )
        .await?;
    let render_time = render_start.elapsed();
    present_internal_vello_target(
        backend,
        native,
        device_identity,
        state,
        resources,
        render_time,
    )
    .await
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
async fn present_internal_vello_target(
    backend: &mut Backend,
    native: &PresentedSurface,
    device_identity: DeviceSlotIdentity,
    state: &mut PresentedSurfaceState,
    resources: &PresentedResourceBundle,
    render_time: Duration,
) -> Result<SurfaceFrameCommit> {
    let present_start = Instant::now();
    let transaction = backend.begin_gpu_operation(
        device_identity,
        GpuOperationStage::Present,
        RuntimeOperation::SurfaceRendering,
    )?;
    let (device, queue) = backend.present_device_queue(device_identity)?;
    let (surface_texture, transaction) =
        acquire_internal_vello_surface_texture(native, device, state, transaction).await?;
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

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
async fn acquire_internal_vello_surface_texture(
    native: &PresentedSurface,
    device: &wgpu::Device,
    state: &mut PresentedSurfaceState,
    transaction: GpuOperationTransaction,
) -> Result<(AcquiredPresentedSurfaceTexture, GpuOperationTransaction)> {
    let mut transaction = Some(transaction);
    let surface_texture = match native.acquire_texture(device) {
        PresentedSurfaceAcquire::Success(surface_texture) => surface_texture,
        PresentedSurfaceAcquire::Suboptimal(surface_texture) => {
            drop(surface_texture);
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Suboptimal,
            )
            .await?);
        }
        PresentedSurfaceAcquire::Outdated => {
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Outdated,
            )
            .await?);
        }
        PresentedSurfaceAcquire::Occluded => {
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Occluded,
            )
            .await?);
        }
        PresentedSurfaceAcquire::Timeout => {
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Timeout,
            )
            .await?);
        }
        PresentedSurfaceAcquire::Lost => {
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Lost,
            )
            .await?);
        }
        PresentedSurfaceAcquire::Validation => {
            return Err(finish_presented_acquire_failure(
                transaction
                    .take()
                    .expect("present transaction must remain available"),
                state,
                PresentedAcquireFailure::Validation,
            )
            .await?);
        }
    };
    Ok((
        surface_texture,
        transaction
            .take()
            .expect("present transaction must remain available after successful acquire"),
    ))
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
