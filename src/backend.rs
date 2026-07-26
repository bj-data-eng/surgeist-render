#[cfg(test)]
use super::gpu_transaction::{
    AfterInternalVelloSubmitCheckpointForTest, InternalVelloSubmissionObservationForTest,
};
use super::pass::{
    C08ExternalOutputView, C08PreparableGraph, C09PreparableGraph, LoweredGraphPlan, PreparedGraph,
};
#[cfg(test)]
use super::pass::{C08PassCacheRequestsForTest, C09CompositeCacheRequestsForTest};
#[cfg(test)]
use super::pass::{C10PreparableGraph, C11PreparableGraph};
use super::resource::{FrameCleanup, ResourceManager, WorkingFormat};
#[cfg(test)]
use super::resource::{
    FrameResourceScope, ManagerIdentity, ResourceIdentity, ResourceLease,
    ResourceManagerObservationForTest,
};
#[cfg(test)]
use super::shader::{ColorFilterOperationBufferLimits, DevicePassCacheCountsForTest};
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
    TransactionTargetIntent, VelloEngineState, scene::VelloScene,
};
#[cfg(test)]
use super::vello_engine::{PreparedVelloPass, VelloAtlasOutcome};
use super::*;
#[cfg(test)]
use super::{command::OffscreenBounds, geometry::physical_size, texture::EffectTextureDescriptor};
use super::{
    gpu_transaction::{
        C08GraphOutputCommit, C08GraphSubmissionPayload, GpuOperationStage,
        GpuOperationTransaction, InternalVelloPayload,
    },
    shader::{DevicePassCache, ProvisionalDevicePassCacheUpdate},
    texture::{TextureDescriptor, headless_texture_descriptor},
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::{
    cell::RefCell,
    fmt,
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
    resource_cache_budget: ResourceCacheBudget,
}

/// One validated exact graph selected for atomic surface execution.
#[must_use = "an exact surface graph must enter its GPU transaction"]
pub(crate) enum ExactSurfaceGraph {
    C08(C08PreparableGraph),
    C09(C09PreparableGraph),
    #[cfg(test)]
    C10(C10PreparableGraph),
    #[cfg(test)]
    C11(C11PreparableGraph),
    C12(super::pass::C12PreparableGraph),
}

impl ExactSurfaceGraph {
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        match self {
            Self::C08(preparable) => preparable.working_format(),
            Self::C09(preparable) => preparable.working_format(),
            #[cfg(test)]
            Self::C10(preparable) => preparable.working_format(),
            #[cfg(test)]
            Self::C11(preparable) => preparable.working_format(),
            Self::C12(preparable) => preparable.working_format(),
        }
    }

    pub(crate) const fn output_format(&self) -> Format {
        match self {
            Self::C08(preparable) => preparable.output_format(),
            Self::C09(preparable) => preparable.output_format(),
            #[cfg(test)]
            Self::C10(preparable) => preparable.output_format(),
            #[cfg(test)]
            Self::C11(preparable) => preparable.output_format(),
            Self::C12(preparable) => preparable.output_format(),
        }
    }

    fn known_output_extent(&self) -> Result<Option<PhysicalSize>> {
        match self {
            Self::C08(preparable) => preparable.output_extent().map(Some),
            Self::C09(_) => Ok(None),
            #[cfg(test)]
            Self::C10(_) => Ok(None),
            #[cfg(test)]
            Self::C11(_) => Ok(None),
            Self::C12(preparable) => preparable.output_extent().map(Some),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08ShaderCacheRealizationObservationForTest {
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
pub(crate) struct C09CompositeCacheRealizationObservationForTest {
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
pub(crate) struct C09MaskSamplingVectorForTest {
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
    pub(crate) layer_point: Point,
    pub(crate) clip_alpha: Option<f32>,
    pub(crate) opacity: f32,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct C09MaskSamplingInputForTest {
    pub(crate) mask_size: PhysicalSize,
    pub(crate) mask_rgba: Vec<u8>,
    pub(crate) mask_bounds: Rect,
    pub(crate) source: [f32; 4],
    pub(crate) vectors: Vec<C09MaskSamplingVectorForTest>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C09BlendVectorForTest {
    pub(crate) blend: BlendMode,
    pub(crate) source: [f32; 4],
    pub(crate) parent: [f32; 4],
    pub(crate) opacity: f32,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct C09GpuVectorResultsForTest {
    pub(crate) working_format: WorkingFormat,
    pub(crate) rgba: Vec<[f32; 4]>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08CustomSpineEncodingObservationForTest {
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
pub(crate) struct C09OrderedGraphEncodingObservationForTest {
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
pub(crate) struct C10OrderedColorGraphEncodingObservationForTest {
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
pub(crate) struct C10OversizedBufferPreservationObservationForTest {
    pub(crate) returns_exact_limit_error: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11SpatialFilterGraphEncodingObservationForTest {
    pub(crate) pass_order: Vec<super::pass::C11FilterPassTagForTest>,
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
pub(crate) struct C11FailurePreservationObservationForTest {
    pub(crate) encode_failure_is_reported: bool,
    pub(crate) scope_failure_is_reported: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
    pub(crate) performs_no_submission_or_retry: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12BackdropGraphEncodingObservationForTest {
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
pub(crate) struct C12FailurePreservationObservationForTest {
    pub(crate) encode_failure_is_reported: bool,
    pub(crate) resources_are_unchanged: bool,
    pub(crate) cache_is_unchanged: bool,
    pub(crate) publication_is_unchanged: bool,
    pub(crate) performs_no_submission_or_retry: bool,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum C11InjectedFailureForTest {
    Encode,
    Scope,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08CaptureFailureObservationForTest {
    pub(crate) capture_failure_is_reported: bool,
    pub(crate) complete_pass_is_rejected: bool,
    pub(crate) retry_on_new_encoder_is_rejected: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08MultipleVelloCaptureEncodingObservationForTest {
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
pub(crate) enum C08TwoCaptureFailureForTest {
    LaterCaptureEncoding,
    SharedScopeResolution,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct C08TwoCaptureFailureObservationForTest {
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
pub(crate) struct C08VelloCaptureRasterContractObservationForTest {
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
    resources: ResourceManager,
    pass_cache: DevicePassCache,
    #[cfg(test)]
    drop_witness: Arc<()>,
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

#[cfg(test)]
fn provision_c08_requests_for_test(
    ready: &ReadyDeviceState,
    requests: &C08PassCacheRequestsForTest,
    invalidate_last_pipeline: bool,
) -> Result<(ProvisionalDevicePassCacheUpdate, bool)> {
    let mut update = ready.pass_cache.provisional_update();
    let last = requests.passes().len().saturating_sub(1);
    let mut encoding_ready = !requests.passes().is_empty();
    for (index, keys) in requests.passes().iter().enumerate() {
        let objects = if invalidate_last_pipeline && index == last {
            update.realize_c08_pass_with_invalid_fragment_for_test(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        } else {
            update.realize_c08_pass(
                &ready.device,
                &ready.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?
        };
        drop(objects);
        encoding_ready &= update.contains_c08_pass_for_test(
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
fn c08_requests_are_cached_for_test(
    cache: &DevicePassCache,
    requests: &C08PassCacheRequestsForTest,
) -> bool {
    !requests.passes().is_empty()
        && requests.passes().iter().all(|keys| {
            cache.contains_c08_pass_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
        })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct C09CompositeProvisionObservationForTest {
    encoding_ready: bool,
    has_normal: bool,
    has_destination: bool,
    all_optional_combinations: bool,
    normal_uses_fixed_blend: bool,
    destination_uses_replace_blend: bool,
}

#[cfg(test)]
fn provision_c09_composite_requests_for_test(
    ready: &ReadyDeviceState,
    requests: &C09CompositeCacheRequestsForTest,
    invalidate_last_pipeline: bool,
) -> Result<(
    ProvisionalDevicePassCacheUpdate,
    C09CompositeProvisionObservationForTest,
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
        C09CompositeProvisionObservationForTest {
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
fn c09_composite_requests_are_cached_for_test(
    cache: &DevicePassCache,
    requests: &C09CompositeCacheRequestsForTest,
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
struct C09GpuVectorDrawForTest {
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
struct C09GpuMaskTextureForTest<'a> {
    size: PhysicalSize,
    rgba: &'a [u8],
    bounds: Rect,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) struct C09PreparedGpuVectorsForTest {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) working_format: WorkingFormat,
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) outputs: Vec<wgpu::Texture>,
    pub(crate) pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn c09_vector_texture(
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
fn c09_clear_vector_texture(
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
fn c09_vector_uniform_buffer(
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
fn c09_upload_vector_mask(
    ready: &ReadyDeviceState,
    mask: Option<&C09GpuMaskTextureForTest<'_>>,
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
                "C09 GPU mask vector byte length overflowed",
            )
        })?;
    if mask.rgba.len() != expected_len || mask.size.width() == 0 || mask.size.height() == 0 {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "C09 GPU mask vector bytes do not match a positive RGBA8 extent",
        ));
    }
    let texture = c09_vector_texture(
        &ready.device,
        mask.size,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST),
        "Surgeist C09 GPU vector mask",
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
struct C09VectorDrawTextures {
    source: wgpu::TextureView,
    parent: Option<wgpu::TextureView>,
    clip: Option<wgpu::TextureView>,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
struct C09VectorDrawEncodingContext<'a> {
    ready: &'a ReadyDeviceState,
    requests: &'a C09CompositeCacheRequestsForTest,
    mask_view: Option<&'a wgpu::TextureView>,
    mask: Option<&'a C09GpuMaskTextureForTest<'a>>,
    spatial_bytes: &'a [u8],
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn c09_prepare_vector_draw_textures(
    ready: &ReadyDeviceState,
    encoder: &mut wgpu::CommandEncoder,
    working_format: WorkingFormat,
    source_size: PhysicalSize,
    draw: C09GpuVectorDrawForTest,
) -> C09VectorDrawTextures {
    let source = c09_vector_texture(
        &ready.device,
        source_size,
        working_format.texture_format(),
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
        "Surgeist C09 GPU vector source",
    );
    let source = source.create_view(&wgpu::TextureViewDescriptor::default());
    c09_clear_vector_texture(
        encoder,
        &source,
        draw.source,
        "Surgeist C09 GPU vector source clear",
    );
    let parent =
        (draw.path == super::shader::ShaderCompositePathKey::DestinationSampling).then(|| {
            let texture = c09_vector_texture(
                &ready.device,
                PhysicalSize::new(1, 1),
                working_format.texture_format(),
                wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
                "Surgeist C09 GPU vector parent",
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            c09_clear_vector_texture(
                encoder,
                &view,
                draw.parent,
                "Surgeist C09 GPU vector parent clear",
            );
            view
        });
    let clip = draw.has_clip_coverage.then(|| {
        let texture = c09_vector_texture(
            &ready.device,
            PhysicalSize::new(1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING),
            "Surgeist C09 GPU vector clip coverage",
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        c09_clear_vector_texture(
            encoder,
            &view,
            [1.0, 0.25, 0.75, draw.clip_alpha],
            "Surgeist C09 GPU vector clip clear",
        );
        view
    });
    let output = c09_vector_texture(
        &ready.device,
        PhysicalSize::new(1, 1),
        working_format.texture_format(),
        wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::COPY_SRC),
        "Surgeist C09 GPU vector output",
    );
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let base = if draw.path == super::shader::ShaderCompositePathKey::Normal {
        draw.parent
    } else {
        [0.125, 0.25, 0.375, 0.5]
    };
    c09_clear_vector_texture(
        encoder,
        &output_view,
        base,
        "Surgeist C09 GPU vector output clear",
    );
    C09VectorDrawTextures {
        source,
        parent,
        clip,
        output,
        output_view,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn c09_vector_parameter_bytes(
    mask: Option<&C09GpuMaskTextureForTest<'_>>,
    draw: C09GpuVectorDrawForTest,
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
fn c09_encode_vector_draw(
    context: &C09VectorDrawEncodingContext<'_>,
    update: &mut ProvisionalDevicePassCacheUpdate,
    encoder: &mut wgpu::CommandEncoder,
    textures: &C09VectorDrawTextures,
    draw: C09GpuVectorDrawForTest,
) -> Result<()> {
    let keys = context
        .requests
        .composite_pass(draw.path, draw.has_clip_coverage, draw.has_alpha_mask)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C09 GPU vector draw has no exact composite pipeline keys",
            )
        })?;
    let spatial = c09_vector_uniform_buffer(
        &context.ready.device,
        &context.ready.queue,
        context.spatial_bytes,
        "Surgeist C09 GPU vector spatial uniform",
    );
    let parameters = c09_vector_parameter_bytes(context.mask, draw)?;
    let parameters = c09_vector_uniform_buffer(
        &context.ready.device,
        &context.ready.queue,
        &parameters,
        "Surgeist C09 GPU vector composite parameters",
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
                    "C09 GPU mask draw has no uploaded mask texture",
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
            label: Some("Surgeist C09 GPU vector bindings"),
            layout: objects.bind_group_layout(),
            entries: &entries,
        });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Surgeist C09 GPU vector composite"),
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
fn encode_c09_gpu_vectors_for_test(
    ready: &ReadyDeviceState,
    requests: &C09CompositeCacheRequestsForTest,
    working_format: WorkingFormat,
    mask: Option<C09GpuMaskTextureForTest<'_>>,
    draws: &[C09GpuVectorDrawForTest],
) -> Result<C09PreparedGpuVectorsForTest> {
    if draws.is_empty() {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            "C09 GPU vector execution requires at least one draw",
        ));
    }
    let mask_texture = c09_upload_vector_mask(ready, mask.as_ref())?;
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
            label: Some("Surgeist C09 GPU vector encoder"),
        });
    let mut outputs = Vec::with_capacity(draws.len());
    let mut pass_cache_update = ready.pass_cache.provisional_update();
    let context = C09VectorDrawEncodingContext {
        ready,
        requests,
        mask_view: mask_view.as_ref(),
        mask: mask.as_ref(),
        spatial_bytes: &spatial_bytes,
    };
    for draw in draws.iter().copied() {
        let draw_textures = c09_prepare_vector_draw_textures(
            ready,
            &mut encoder,
            working_format,
            vector_source_size,
            draw,
        );
        c09_encode_vector_draw(
            &context,
            &mut pass_cache_update,
            &mut encoder,
            &draw_textures,
            draw,
        )?;
        outputs.push(draw_textures.output);
    }
    Ok(C09PreparedGpuVectorsForTest {
        device: ready.device.clone(),
        queue: ready.queue.clone(),
        working_format,
        encoder,
        outputs,
        pass_cache_update,
    })
}

#[cfg(test)]
fn c10_limit_error_is_exact(rejection: Option<Error>) -> bool {
    rejection.is_some_and(|error| {
        error.code() == ErrorCode::InvalidInput
            && error.invalid_value_diagnostic().is_some_and(|invalid| {
                invalid.field() == "color filter operation buffer byte length"
            })
    })
}

#[cfg(test)]
fn c09_ordered_encoding_observation(
    summary: &super::pass::C08CustomSpineEncodingSummary,
) -> C09OrderedGraphEncodingObservationForTest {
    C09OrderedGraphEncodingObservationForTest {
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
fn c11_spatial_encoding_observation(
    summary: &super::pass::C08CustomSpineEncodingSummary,
) -> C11SpatialFilterGraphEncodingObservationForTest {
    C11SpatialFilterGraphEncodingObservationForTest {
        pass_order: summary.c11_pass_order.clone(),
        blur_pass_count: summary.blur_pass_count,
        drop_shadow_colorize_count: summary.drop_shadow_colorize_count,
        drop_shadow_merge_count: summary.drop_shadow_merge_count,
        each_pass_advances_once: summary.advances_every_pass_once
            && summary.encodes_custom_passes_in_order,
        binds_exact_prepared_resources: summary.c11_binds_exact_prepared_resources,
        uses_signed_viewport_and_scissor: summary.c11_uses_signed_viewport_and_scissor,
        blur_sources_intermediates_and_results_are_distinct: summary
            .blur_sources_intermediates_and_results_are_distinct,
        kernels_release_at_validated_last_use: summary.c11_kernels_release_at_validated_last_use,
        textures_release_at_validated_last_use: summary.c11_textures_release_at_validated_last_use,
        drop_shadow_reads_original_source_twice: summary.drop_shadow_reads_original_source_twice,
        original_source_releases_after_merge: summary.original_source_releases_after_merge,
        one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
        transaction_committed: false,
    }
}

#[cfg(test)]
fn c12_backdrop_encoding_observation(
    summary: &super::pass::C08CustomSpineEncodingSummary,
) -> C12BackdropGraphEncodingObservationForTest {
    C12BackdropGraphEncodingObservationForTest {
        encodes_copy_filter_clip_foreground_and_group_in_order: summary
            .encodes_custom_passes_in_order
            && summary.copy_backdrop_count == 1
            && summary.color_filter_count > 0
            && summary.blur_pass_count > 0
            && summary.drop_shadow_colorize_count > 0
            && summary.drop_shadow_merge_count > 0
            && summary.layer_composite_count >= 2
            && summary.c12_group_order_is_exact
            && summary.advances_every_pass_once,
        parent_is_copied_once: summary.copy_backdrop_count == 1
            && summary.copy_backdrop_binds_exact_prepared_resources
            && summary.copy_backdrop_preserves_signed_mapping,
        copy_filter_foreground_and_group_are_distinct: summary
            .copy_backdrop_source_and_result_are_distinct
            && summary.color_filter_sources_and_results_are_distinct
            && summary.blur_sources_intermediates_and_results_are_distinct
            && summary.parent_and_result_are_distinct
            && summary.c12_group_resources_are_distinct,
        later_sibling_reads_completed_group: summary.c12_later_sibling_transition_is_exact,
        releases_at_validated_last_use: summary.advances_every_pass_once
            && summary.color_filter_operation_buffers_released
            && summary.c11_kernels_release_at_validated_last_use
            && summary.c11_textures_release_at_validated_last_use,
        one_graph_command_encoder: summary.graph_work_shares_one_command_encoder,
        transaction_committed: false,
    }
}

#[cfg(test)]
fn c11_resources_preserved(
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
fn c11_failure_publication_for_test(
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
fn c08_custom_spine_observation(
    summary: super::pass::C08CustomSpineEncodingSummary,
    capture_count: usize,
    captures_are_exact: bool,
    cache_before: DevicePassCacheCountsForTest,
    cache_after: DevicePassCacheCountsForTest,
) -> C08CustomSpineEncodingObservationForTest {
    C08CustomSpineEncodingObservationForTest {
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
async fn observe_c08_two_capture_encoding_failure(
    prepared: &mut PreparedGraph<'_>,
    device: &wgpu::Device,
    output: &wgpu::TextureView,
    extent: PhysicalSize,
    failure: C08TwoCaptureFailureForTest,
) -> Result<(usize, bool, bool, bool)> {
    match failure {
        C08TwoCaptureFailureForTest::LaterCaptureEncoding => {
            prepared.fail_capture_encoding_after_for_test(1);
        }
        C08TwoCaptureFailureForTest::SharedScopeResolution => {
            prepared.fail_scope_resolution_for_test();
        }
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist C08 two-capture failure encoder"),
    });
    let result = prepared
        .encode_c08_custom_spine(
            &mut encoder,
            C08ExternalOutputView::try_new(output, Format::Rgba8, extent)?,
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
                C08TwoCaptureFailureForTest::LaterCaptureEncoding => {
                    error.message() == "injected C08 Vello capture encoding failure"
                }
                C08TwoCaptureFailureForTest::SharedScopeResolution => {
                    error.message() == "checked internal Vello resource or command encoding failed"
                }
            },
            true,
        ),
    };
    let mut retry = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist C08 forbidden two-capture retry encoder"),
    });
    let retry_rejected = prepared
        .encode_c08_custom_spine(
            &mut retry,
            C08ExternalOutputView::try_new(output, Format::Rgba8, extent)?,
        )
        .await
        .is_err_and(|error| {
            error.message()
                == "the C08 custom encoding is one-shot; discard this prepared graph and its encoder"
        });
    drop(retry.finish());
    drop(encoder.finish());
    Ok((acquired, reported, no_commit, retry_rejected))
}

#[cfg(test)]
fn c08_test_output_texture(
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

    fn for_selected_working_format(mut self, working_format: WorkingFormat) -> Result<Self> {
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
        let state = DeviceState::new(adapter, device, queue, self.resource_cache_budget).await?;
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
            "checked C08 pass objects lost their persistent device cache",
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
        let observation = InternalVelloSubmissionObservationForTest::default();
        let payload = InternalVelloPayload::observed_for_test(
            command_encoder.finish(),
            super::vello_engine::PendingVelloResourceCommit::new(lease),
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
            "T6 cancellation-owned internal Vello target",
        );
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
        let (checkpoint, checkpoint_observed) = AfterInternalVelloSubmitCheckpointForTest::paused();
        let payload = InternalVelloPayload::paused_after_submit_for_test(
            command_encoder.finish(),
            super::vello_engine::PendingVelloResourceCommit::new(lease),
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

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 calls the validated C07 graph preparation handoff before execution"
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
        PreparedGraph::try_prepare(
            lowered,
            policy,
            &capabilities,
            &ready.device,
            &ready.queue,
            &ready.resources,
            (&ready.pass_cache, realize_checked_passes),
        )
        .map(|prepared| prepared.with_vello_engine(&ready.engine))
    }

    #[cfg(test)]
    fn prepare_c10_graph_resources_with_operation_limits_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        operation_limits: ColorFilterOperationBufferLimits,
    ) -> Result<PreparedGraph<'_>> {
        let state = self.device_states.get_mut(identity.slot()).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device slot is unavailable for C10 limit preparation",
            )
        })?;
        if state.generation != identity.generation {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "GPU device generation changed before C10 limit preparation",
            ));
        }
        if let Some(terminal) = state.terminal() {
            return Err(terminal.error(RuntimeOperation::EffectRendering));
        }
        if !state.signal.has_active_operation() {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "C10 limit preparation requires one active GPU transaction",
            ));
        }
        let capabilities = state.capabilities;
        let ready = state.ready().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "ready GPU resources disappeared before C10 limit preparation",
            )
        })?;
        PreparedGraph::try_prepare_c10_with_operation_limits_for_test(
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
        match graph {
            ExactSurfaceGraph::C08(preparable) => {
                PreparedGraph::try_prepare_c08_with_working_format(
                    preparable,
                    selected_working_format,
                    &capabilities,
                    &ready.device,
                    &ready.queue,
                    &ready.resources,
                    (&ready.pass_cache, true),
                )
            }
            ExactSurfaceGraph::C09(preparable) => PreparedGraph::try_prepare_c09(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            #[cfg(test)]
            ExactSurfaceGraph::C10(preparable) => PreparedGraph::try_prepare_c10(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            #[cfg(test)]
            ExactSurfaceGraph::C11(preparable) => PreparedGraph::try_prepare_c11(
                preparable,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
            ExactSurfaceGraph::C12(preparable) => PreparedGraph::try_prepare_c12(
                preparable,
                selected_working_format,
                &capabilities,
                &ready.device,
                &ready.queue,
                &ready.resources,
                (&ready.pass_cache, true),
            ),
        }
        .map(|prepared| prepared.with_vello_engine(&ready.engine))
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
    pub(crate) async fn c09_composite_cache_realization_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &C09CompositeCacheRequestsForTest,
    ) -> Result<C09CompositeCacheRealizationObservationForTest> {
        let initial_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C09 composite realization requires a ready device",
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
                "C09 composite realization lost its ready device",
            )?;
            provision_c09_composite_requests_for_test(ready, requests, false)?
        };
        let counts_before_commit = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C09 composite realization lost its persistent cache",
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
                "C09 committed compositor cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        let realizes_normal_and_destination_programs = initial_counts.is_empty()
            && counts_before_commit == initial_counts
            && committed_counts != initial_counts
            && provision.encoding_ready
            && provision.has_normal
            && provision.has_destination
            && c09_composite_requests_are_cached_for_test(
                &self
                    .ready_state_mut(
                        identity,
                        RuntimeOperation::EffectRendering,
                        BackendErrorCode::RenderFailed,
                        "C09 committed compositor programs disappeared",
                    )?
                    .pass_cache,
                requests,
            );

        let reuses_exact_committed_entries = self
            .c09_reuses_committed_entries_for_test(identity, requests, committed_counts)
            .await?;

        let failed_validation_publishes_none = self
            .c09_validation_publishes_none_for_test(requests)
            .await?;
        let (cancellation_publishes_none, device_transition_publishes_none) = self
            .c09_cancellation_publishes_none_for_test(requests)
            .await?;

        Ok(C09CompositeCacheRealizationObservationForTest {
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
    async fn c09_reuses_committed_entries_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &C09CompositeCacheRequestsForTest,
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
                "C09 compositor cache reuse lost its ready device",
            )?;
            provision_c09_composite_requests_for_test(ready, requests, false)?
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
                "C09 reused compositor cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        Ok(reused_existing && provision.encoding_ready && counts == committed)
    }

    #[cfg(test)]
    async fn c09_validation_publishes_none_for_test(
        &mut self,
        requests: &C09CompositeCacheRequestsForTest,
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
                "C09 validation probe lost its ready device",
            )?;
            provision_c09_composite_requests_for_test(ready, requests, true)?.0
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
    async fn c09_cancellation_publishes_none_for_test(
        &mut self,
        requests: &C09CompositeCacheRequestsForTest,
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
                "C09 cancellation probe lost its ready device",
            )?;
            provision_c09_composite_requests_for_test(ready, requests, false)?
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
                "C09 transition probe lost its ready device",
            )?;
            provision_c09_composite_requests_for_test(ready, requests, false)?
        };
        self.signal_loss_for_test(identity, DeviceLossReason::Destroyed);
        let error = transaction.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        let transitioned =
            provision.encoding_ready && error.is_err() && self.renderer_released_for_test(identity);
        Ok((canceled, transitioned))
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn c09_shader_mask_sampling_preparation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &C09CompositeCacheRequestsForTest,
        input: &C09MaskSamplingInputForTest,
    ) -> Result<C09PreparedGpuVectorsForTest> {
        let working_format = self
            .device_capabilities(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "C09 mask vectors require immutable device capabilities",
                )
            })?
            .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)?;
        let draws = input
            .vectors
            .iter()
            .map(|vector| C09GpuVectorDrawForTest {
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
            "C09 mask vectors lost their ready device",
        )?;
        encode_c09_gpu_vectors_for_test(
            ready,
            requests,
            working_format,
            Some(C09GpuMaskTextureForTest {
                size: input.mask_size,
                rgba: &input.mask_rgba,
                bounds: input.mask_bounds,
            }),
            &draws,
        )
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn c09_shader_blend_preparation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &C09CompositeCacheRequestsForTest,
        vectors: &[C09BlendVectorForTest],
    ) -> Result<C09PreparedGpuVectorsForTest> {
        let working_format = self
            .device_capabilities(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "C09 blend vectors require immutable device capabilities",
                )
            })?
            .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)?;
        let draws = vectors
            .iter()
            .map(|vector| C09GpuVectorDrawForTest {
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
            "C09 blend vectors lost their ready device",
        )?;
        encode_c09_gpu_vectors_for_test(ready, requests, working_format, None, &draws)
    }

    #[cfg(test)]
    pub(crate) async fn c08_shader_cache_realization_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        rgba_requests: &C08PassCacheRequestsForTest,
        bgra_requests: &C08PassCacheRequestsForTest,
    ) -> Result<C08ShaderCacheRealizationObservationForTest> {
        let initial_counts = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C08 shader-cache observation requires a ready device",
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
                "C08 shader realization lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, rgba_requests, false)?
        };
        let counts_before_commit = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C08 shader realization lost its persistent cache",
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
                "C08 committed cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        let realizes_all_checked_programs = initial_counts.is_empty()
            && counts_before_commit == initial_counts
            && rgba_counts != initial_counts
            && c08_requests_are_cached_for_test(
                &self
                    .ready_state_mut(
                        identity,
                        RuntimeOperation::EffectRendering,
                        BackendErrorCode::RenderFailed,
                        "C08 committed programs disappeared",
                    )?
                    .pass_cache,
                rgba_requests,
            );

        let reuses_exact_committed_entries = self
            .c08_reuses_committed_entries_for_test(identity, rgba_requests, rgba_counts)
            .await?;

        let (failed_validation_publishes_none, specializes_rgba_and_bgra_outputs) = self
            .c08_validation_and_specialization_for_test(
                identity,
                rgba_requests,
                bgra_requests,
                rgba_counts,
            )
            .await?;
        let (cancellation_publishes_none, device_transition_publishes_none) = self
            .c08_cancellation_publishes_none_for_test(rgba_requests)
            .await?;

        Ok(C08ShaderCacheRealizationObservationForTest {
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
    async fn c08_reuses_committed_entries_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        requests: &C08PassCacheRequestsForTest,
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
                "C08 cache reuse lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, requests, false)?
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
                "C08 reused cache disappeared",
            )?
            .pass_cache
            .counts_for_test();
        Ok(exact_existing && counts == committed)
    }

    #[cfg(test)]
    async fn c08_validation_and_specialization_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        rgba: &C08PassCacheRequestsForTest,
        bgra: &C08PassCacheRequestsForTest,
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
                "C08 validation probe lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, bgra, true)?.0
        };
        let error = validation.finish(RuntimeOperation::EffectRendering).await;
        drop(update);
        let after_validation = self
            .ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C08 validation probe lost its persistent cache",
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
                "C08 BGRA specialization lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, bgra, false)?
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
            "C08 specialized programs disappeared",
        )?;
        let counts = ready.pass_cache.counts_for_test();
        let specialized = handles_ready
            && counts != rgba_counts
            && c08_requests_are_cached_for_test(&ready.pass_cache, rgba)
            && c08_requests_are_cached_for_test(&ready.pass_cache, bgra);
        Ok((failed, specialized))
    }

    #[cfg(test)]
    async fn c08_cancellation_publishes_none_for_test(
        &mut self,
        requests: &C08PassCacheRequestsForTest,
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
                "C08 cancellation probe lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, requests, false)?
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
                "C08 transition probe lost its ready device",
            )?;
            provision_c08_requests_for_test(ready, requests, false)?
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
    pub(crate) async fn c08_custom_spine_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
        output_format: Format,
    ) -> Result<C08CustomSpineEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C08 custom-spine observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
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
                "C08 custom-spine observation requires a ready pass cache",
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
                "C08 custom-spine observation lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = c08_test_output_texture(
            &device,
            output_extent,
            output_format,
            "Surgeist C08 external output observation",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output = C08ExternalOutputView::try_new(&output_view, output_format, output_extent)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C08 caller-owned custom-spine observation encoder"),
        });
        let encoded = prepared
            .encode_c08_custom_spine(&mut encoder, output)
            .await?;
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
                "C08 custom-spine observation lost its provisional cache boundary",
            )?
            .pass_cache
            .counts_for_test();

        Ok(c08_custom_spine_observation(
            summary,
            capture_handoff_count,
            capture_handoffs_are_exact,
            pass_cache_before,
            pass_cache_after,
        ))
    }

    #[cfg(test)]
    pub(crate) async fn c10_ordered_color_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C10OrderedColorGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C10 graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_c10_color_graph_for_test(filters, commands, context)?;
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
                "C10 graph encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C10 caller-owned graph observation encoder"),
        });
        let pending = match prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
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
        let mut observed = C10OrderedColorGraphEncodingObservationForTest {
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
        let prepared_submission = prepared.finish_c08_submission(pending)?;
        drop(output_view);
        let payload = C08GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C10 graph encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_c08_graph(
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
    pub(crate) async fn c11_spatial_filter_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C11SpatialFilterGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C11 encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_c10_color_graph_for_test(filters, commands, context)?;
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
                "C11 encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C11 caller-owned graph observation encoder"),
        });
        let pending = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let summary = pending.summary_for_test();
        let mut observed = c11_spatial_encoding_observation(summary);
        let prepared_submission = prepared.finish_c08_submission(pending)?;
        drop(output_view);
        let payload = C08GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C11 encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_c08_graph(
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
    pub(crate) async fn c12_backdrop_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C12BackdropGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C12 encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "C12 encoding observation requires a validated GPU graph",
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
                "C12 encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C12 caller-owned graph observation encoder"),
        });
        let pending = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let mut observed = c12_backdrop_encoding_observation(pending.summary_for_test());
        let prepared_submission = prepared.finish_c08_submission(pending)?;
        drop(output_view);
        let payload = C08GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C12 encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_c08_graph(
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
    pub(crate) async fn c12_failure_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C12FailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C12 failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "C12 failure observation requires a validated GPU graph",
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
                "C12 failure observation lost its publication device",
            )?
            .device
            .clone();
        let published_surface = c11_failure_publication_for_test(&device, identity)?;
        let publication_count_before = published_surface.headless_publication_count_for_test();
        let publication_state_before = published_surface.resource_state();
        let (resources_before, cache_before) = self.c11_resource_and_cache_state(identity)?;
        let submission_scope =
            super::gpu_transaction::ScopedGpuOperationSubmissionObservationForTest::begin();
        let submission = submission_scope.observation_for_test();
        let graph_scope =
            super::gpu_transaction::ScopedC08GraphSubmissionObservationForTest::begin();
        let graph_submission = graph_scope.observation_for_test();
        let direct_scope =
            super::gpu_transaction::ScopedInternalVelloSubmissionObservationForTest::begin();
        let direct_submission = direct_scope.observation_for_test();
        let encode_error = self
            .run_c11_failed_encoding_attempt(
                identity,
                lowered,
                policy,
                C11InjectedFailureForTest::Encode,
            )
            .await?;
        let performs_no_submission_or_retry = submission.queue_submission_count_for_test() == 0
            && graph_submission.queue_submission_count_for_test() == 0
            && direct_submission.queue_submission_count_for_test() == 0;
        drop(direct_scope);
        drop(graph_scope);
        drop(submission_scope);
        let (resources_after, cache_after) = self.c11_resource_and_cache_state(identity)?;
        Ok(C12FailurePreservationObservationForTest {
            encode_failure_is_reported: encode_error
                .message()
                .contains("injected C10 color-filter shader failure"),
            resources_are_unchanged: c11_resources_preserved(&resources_before, &resources_after),
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
            performs_no_submission_or_retry,
        })
    }

    #[cfg(test)]
    pub(crate) async fn c11_failure_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C11FailurePreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C11 failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_c10_color_graph_for_test(filters, commands, context)?;
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
                "C11 failure observation lost its publication device",
            )?
            .device
            .clone();
        let published_surface = c11_failure_publication_for_test(&device, identity)?;
        let publication_count_before = published_surface.headless_publication_count_for_test();
        let publication_state_before = published_surface.resource_state();
        let (resources_before, cache_before) = self.c11_resource_and_cache_state(identity)?;
        let submission_scope =
            super::gpu_transaction::ScopedGpuOperationSubmissionObservationForTest::begin();
        let submission = submission_scope.observation_for_test();
        let graph_scope =
            super::gpu_transaction::ScopedC08GraphSubmissionObservationForTest::begin();
        let graph_submission = graph_scope.observation_for_test();
        let direct_scope =
            super::gpu_transaction::ScopedInternalVelloSubmissionObservationForTest::begin();
        let direct_submission = direct_scope.observation_for_test();
        let encode_error = self
            .run_c11_failed_encoding_attempt(
                identity,
                lowered.clone(),
                policy,
                C11InjectedFailureForTest::Encode,
            )
            .await?;
        let scope_error = self
            .run_c11_failed_encoding_attempt(
                identity,
                lowered,
                policy,
                C11InjectedFailureForTest::Scope,
            )
            .await?;
        let performs_no_submission_or_retry = submission.queue_submission_count_for_test() == 0
            && graph_submission.queue_submission_count_for_test() == 0
            && direct_submission.queue_submission_count_for_test() == 0;
        drop(direct_scope);
        drop(graph_scope);
        drop(submission_scope);
        let (resources_after, cache_after) = self.c11_resource_and_cache_state(identity)?;
        Ok(C11FailurePreservationObservationForTest {
            encode_failure_is_reported: encode_error
                .message()
                .contains("injected C10 color-filter shader failure"),
            scope_failure_is_reported: scope_error.message()
                == "checked internal Vello resource or command encoding failed",
            resources_are_unchanged: c11_resources_preserved(&resources_before, &resources_after),
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
            performs_no_submission_or_retry,
        })
    }

    #[cfg(test)]
    fn c11_resource_and_cache_state(
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
            "C11 failure observation lost its ready state",
        )?;
        Ok((
            ready.resources.observation_for_test(),
            ready.pass_cache.counts_for_test(),
        ))
    }

    #[cfg(test)]
    async fn run_c11_failed_encoding_attempt(
        &mut self,
        identity: DeviceSlotIdentity,
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        failure: C11InjectedFailureForTest,
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
                "C11 failure attempt lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        if matches!(failure, C11InjectedFailureForTest::Scope) {
            prepared.fail_scope_resolution_for_test();
        }
        let extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C11 injected-failure graph encoder"),
        });
        let _encode_failure = matches!(failure, C11InjectedFailureForTest::Encode)
            .then(super::pass::ScopedC10ColorFilterShaderFailureForTest::after_checked_realization);
        let result = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, extent)?,
            )
            .await;
        drop(output_view);
        drop(output_texture);
        drop(encoder.finish());
        drop(prepared);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
        result.err().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the injected C11 encoding failure unexpectedly succeeded",
            )
        })
    }

    #[cfg(test)]
    pub(crate) async fn c10_oversized_buffer_preservation_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        filters: Vec<FilterList>,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C10OversizedBufferPreservationObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C10 limit observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::authored_c10_color_graph_for_test(filters, commands, context)?;
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
                "C10 limit observation lost its ready device",
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
                "C10 limit observation lost its preflight state",
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
        let rejection = match self.prepare_c10_graph_resources_with_operation_limits_for_test(
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
                "C10 limit observation lost its post-rejection state",
            )?;
            (
                ready.resources.observation_for_test(),
                ready.pass_cache.counts_for_test(),
            )
        };
        let returns_exact_limit_error = c10_limit_error_is_exact(rejection);
        Ok(C10OversizedBufferPreservationObservationForTest {
            returns_exact_limit_error,
            resources_are_unchanged: resources_after == resources_before,
            cache_is_unchanged: cache_after == cache_before,
            publication_is_unchanged: published_surface.headless_publication_count_for_test()
                == publication_count_before
                && published_surface.resource_state() == publication_state_before,
        })
    }

    #[cfg(test)]
    pub(crate) async fn c09_ordered_graph_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C09OrderedGraphEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C09 graph encoding observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let super::frame::FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "C09 graph encoding observation requires a validated GPU graph",
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
                "C09 graph encoding observation lost its ready device",
            )?;
            (ready.device.clone(), ready.queue.clone())
        };
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let (output_texture, output_view) =
            create_headless_texture(&device, output_extent, Format::Rgba8)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C09 caller-owned graph observation encoder"),
        });
        let pending = match prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
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
                return Ok(C09OrderedGraphEncodingObservationForTest::default());
            }
        };
        let summary = pending.summary_for_test();
        let mut observed = c09_ordered_encoding_observation(summary);
        let prepared_submission = prepared.finish_c08_submission(pending)?;
        drop(output_view);
        let payload = C08GraphSubmissionPayload::new(
            encoder.finish(),
            prepared_submission,
            HeadlessPublication::new(output_texture),
        );
        let committed = {
            let ready = self.ready_state_mut(
                identity,
                RuntimeOperation::EffectRendering,
                BackendErrorCode::RenderFailed,
                "C09 graph encoding observation lost its pass cache before commit",
            )?;
            transaction
                .submit_c08_graph(
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
    pub(crate) async fn c08_multiple_vello_capture_encoding_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        donor_commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
    ) -> Result<C08MultipleVelloCaptureEncodingObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "multiple C08 capture coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = super::pass::c08_two_capture_spine_lowered_for_test(
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
                "multiple C08 capture coverage lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = c08_test_output_texture(
            &device,
            output_extent,
            Format::Rgba8,
            "Surgeist C08 multiple-capture output",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C08 multiple-capture graph encoder"),
        });
        let encoded = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let (summary, capture_resources) = encoded.into_summary_and_resources();
        let committed_lease_count = capture_resources.lease_count_for_test();
        drop(encoder.finish());
        drop(prepared);
        let same_transaction = transaction_generation.is_some()
            && transaction_generation == self.active_operation_generation_for_test(identity);
        transaction
            .finish_vello_resources_without_submission_for_test(
                capture_resources,
                RuntimeOperation::EffectRendering,
            )
            .await?;
        let after_commit = self
            .ready_device_state_borrow_for_test(identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "multiple C08 capture commit lost its resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();

        let (aborted_lease_count, after_abort) = self
            .c08_multiple_capture_abort_for_test(
                identity,
                lowered,
                policy,
                &device,
                &output_view,
                output_extent,
            )
            .await?;

        Ok(C08MultipleVelloCaptureEncodingObservationForTest {
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
    async fn c08_multiple_capture_abort_for_test(
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
            label: Some("Surgeist C08 multiple-capture aggregate-abort encoder"),
        });
        let encoded = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(output, Format::Rgba8, extent)?,
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
                    "multiple C08 capture abort lost its resource manager",
                )
            })?
            .internal_resource_manager_observation_for_test();
        Ok((count, observation))
    }

    #[cfg(test)]
    pub(crate) async fn c08_two_capture_failure_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        donor_commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
        failure: C08TwoCaptureFailureForTest,
    ) -> Result<C08TwoCaptureFailureObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "two-capture failure coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let lowered = super::pass::c08_two_capture_spine_lowered_for_test(
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
        let output_texture = c08_test_output_texture(
            &device,
            output_extent,
            Format::Rgba8,
            "Surgeist C08 two-capture failure output",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (
            acquired_capture_lease_count,
            failure_is_reported,
            produces_no_pending_commit,
            retry_is_rejected,
        ) = observe_c08_two_capture_encoding_failure(
            &mut prepared,
            &device,
            &output_view,
            output_extent,
            failure,
        )
        .await?;
        drop(prepared);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;
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

        Ok(C08TwoCaptureFailureObservationForTest {
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
    pub(crate) async fn c08_vello_capture_raster_contract_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
        requested_antialiasing: Antialiasing,
    ) -> Result<C08VelloCaptureRasterContractObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C08 capture raster coverage requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
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
                "C08 capture raster coverage lost its ready device",
            )?
            .device
            .clone();
        let mut prepared = self.prepare_graph_resources(identity, lowered, policy)?;
        let output_extent = prepared.output_extent()?;
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist C08 raster-contract output"),
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
            label: Some("Surgeist C08 raster-contract graph encoder"),
        });
        let encoded = prepared
            .encode_c08_custom_spine(
                &mut encoder,
                C08ExternalOutputView::try_new(&output_view, Format::Rgba8, output_extent)?,
            )
            .await?;
        let (summary, capture_resources) = encoded.into_summary_and_resources();
        let capture = summary.capture_observations.first().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C08 capture raster coverage produced no encoded capture proof",
            )
        })?;
        let observed = C08VelloCaptureRasterContractObservationForTest {
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
    pub(crate) async fn c08_capture_failure_observation_for_test(
        &mut self,
        identity: DeviceSlotIdentity,
        commands: super::command::RenderCommands,
        context: super::frame::FrameContext,
        output_format: Format,
    ) -> Result<C08CaptureFailureObservationForTest> {
        let capabilities = self.device_capabilities(identity).ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "C08 capture-failure observation requires immutable device capabilities",
            )
        })?;
        let policy = EffectQualityPolicy::AllowReducedPrecision;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
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
                "C08 capture-failure observation lost its ready device",
            )?
            .device
            .clone();

        let mut first = self.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let output_extent = first.output_extent()?;
        let output_texture = c08_test_output_texture(
            &device,
            output_extent,
            output_format,
            "Surgeist C08 capture-failure external output observation",
        );
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut first_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C08 failed-capture first encoder observation"),
        });
        let failed_pass = first
            .c08_execution_facts()
            .and_then(|facts| facts.captures().first())
            .map(super::pass::ExecutableVelloCaptureFacts::pass);
        first.fail_capture_encoding_for_test();
        let capture_failure_is_reported = first
            .encode_c08_custom_spine(
                &mut first_encoder,
                C08ExternalOutputView::try_new(&output_view, output_format, output_extent)?,
            )
            .await
            .is_err_and(|error| error.message() == "injected C08 Vello capture encoding failure")
            && failed_pass.is_some();
        let complete_pass_is_rejected =
            failed_pass.is_some_and(|pass| first.complete_pass(pass).is_err());
        drop(first_encoder.finish());
        drop(first);

        let mut retried = self.prepare_graph_resources(identity, lowered, policy)?;
        let mut failed_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C08 failed-capture retry source encoder observation"),
        });
        retried.fail_capture_encoding_for_test();
        let initial_failure = retried
            .encode_c08_custom_spine(
                &mut failed_encoder,
                C08ExternalOutputView::try_new(&output_view, output_format, output_extent)?,
            )
            .await
            .is_err_and(|error| error.message() == "injected C08 Vello capture encoding failure");
        drop(failed_encoder.finish());
        let mut retry_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist C08 forbidden new retry encoder observation"),
        });
        let retry_on_new_encoder_is_rejected = initial_failure
            && retried
                .encode_c08_custom_spine(
                    &mut retry_encoder,
                    C08ExternalOutputView::try_new(&output_view, output_format, output_extent)?,
                )
                .await
                .is_err();
        drop(retry_encoder.finish());
        drop(retried);
        transaction
            .finish(RuntimeOperation::EffectRendering)
            .await?;

        Ok(C08CaptureFailureObservationForTest {
            capture_failure_is_reported,
            complete_pass_is_rejected,
            retry_on_new_encoder_is_rejected,
        })
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
}

impl SurfaceFrameCommit {
    fn without_headless_publication(timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: None,
            _frame_cleanup: None,
        }
    }

    fn headless(publication: HeadlessPublication, timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: Some(publication),
            _frame_cleanup: None,
        }
    }

    fn headless_graph(
        publication: HeadlessPublication,
        frame_cleanup: FrameCleanup,
        timings: RenderTimings,
    ) -> Self {
        Self {
            timings,
            headless_publication: Some(publication),
            _frame_cleanup: Some(frame_cleanup),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    fn presented_graph(frame_cleanup: FrameCleanup, timings: RenderTimings) -> Self {
        Self {
            timings,
            headless_publication: None,
            _frame_cleanup: Some(frame_cleanup),
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
        .encode_c08_custom_spine(
            &mut encoder,
            C08ExternalOutputView::try_new(&draft_view, surface.options.format, physical_size)?,
        )
        .await?;
    let prepared_submission = prepared.finish_c08_submission(pending_encoding)?;
    let payload = C08GraphSubmissionPayload::new(
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
            .submit_c08_graph(
                &device,
                &queue,
                &mut ready.pass_cache,
                payload,
                RuntimeOperation::SurfaceRendering,
            )
            .await?
    };
    let (output, frame_cleanup) = clean.into_parts();
    #[cfg(not(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )))]
    let C08GraphOutputCommit::Headless(publication) = output;
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    let publication = match output {
        C08GraphOutputCommit::Headless(publication) => publication,
        C08GraphOutputCommit::Presented => {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the headless exact graph transaction returned a presented host effect",
            ));
        }
    };
    Ok(SurfaceFrameCommit::headless_graph(
        publication,
        frame_cleanup,
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
        .encode_c08_custom_spine(
            &mut encoder,
            C08ExternalOutputView::try_new(&output_view, output_format, physical_size)?,
        )
        .await?;
    let prepared_submission = prepared.finish_c08_submission(pending_encoding)?;
    drop(output_view);
    let payload =
        C08GraphSubmissionPayload::presented(encoder.finish(), prepared_submission, acquired);
    let clean = {
        let ready = backend.ready_state_mut(
            device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "the presented exact graph lost its ready device before submission",
        )?;
        transaction
            .submit_c08_graph(
                &device,
                &queue,
                &mut ready.pass_cache,
                payload,
                RuntimeOperation::SurfaceRendering,
            )
            .await?
    };
    let (output, frame_cleanup) = clean.into_parts();
    if !matches!(output, C08GraphOutputCommit::Presented) {
        return Err(Error::new(
            BackendErrorCode::PresentFailed,
            "the presented exact graph transaction returned a headless publication",
        ));
    }
    Ok(SurfaceFrameCommit::presented_graph(
        frame_cleanup,
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
