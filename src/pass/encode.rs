use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::super::{
    Color, Format, PhysicalSize, Result, Transform,
    encode::{encode_vello_clip_coverage_scene, encode_vello_scene_with_initial_transform},
    layer::BlendMode,
    renderer::Antialiasing,
    resource::{FrameCleanup, FrameResourceScope, GaussianKernelKey, ResourceManager},
    shader::{
        BlurEdgeParameterBytes, ColorFilterOperationBufferLimits, ColorFilterOperationBytes,
        CompositeParameterBytes, DevicePassCache, DropShadowParameterBytes,
        PassSpatialUniformBytes, ProvisionalC08PassObjects, ProvisionalColorFilterPassObjects,
        ProvisionalCompositePassObjects, ProvisionalDevicePassCacheUpdate, ShaderCompositePathKey,
    },
    vello_engine::{
        ActiveVelloEncodingScope, EncodedVelloCaptureProof, PendingVelloResourceCommit,
        RasterParameters, TransactionEncodingState, TransactionTargetIntent, VelloEngineState,
        VelloResourceLeaseAggregate, scene::VelloScene,
    },
};
use super::{
    close::{ExecutableFilterStepFacts, ExecutableVelloCaptureFacts, preparation_error},
    model::{
        RuntimeBlurInput, RuntimeColorFilter, RuntimeComposite, RuntimeCompositeKind,
        RuntimeInitialization, RuntimeLayerCompositeParameters, RuntimePassCacheKeys,
        RuntimePassId, RuntimePassKind, RuntimeReadBinding, RuntimeReadRole, RuntimeResourceFormat,
        RuntimeResourceId, RuntimeResourceRole, RuntimeResultBinding, RuntimeSamplingEdge,
        RuntimeSamplingFilter, RuntimeSpatialDescriptor, RuntimeVelloCapture,
    },
    parameters::c12_blur_edge_uniform_bytes,
    prepare::{
        PreparedC11PassObjects, PreparedGraph, PreparedTextureBinding,
        RuntimePassPreparationRequest, VELLO_CAPTURE_TEXTURE_USAGES,
    },
};

#[cfg(test)]
use super::{
    PREPARED_GRAPH_TEST_SUPPORT, begin_c08_encoding_observations_for_test,
    finish_c08_encoding_observations_for_test, inject_color_filter_shader_failure_for_test,
    record_c08_capture_observation_for_test, record_c08_graph_encoder_for_test,
};

pub(super) fn backdrop_filter_passes(steps: &[ExecutableFilterStepFacts]) -> Vec<RuntimePassId> {
    steps
        .iter()
        .flat_map(|step| match *step {
            ExecutableFilterStepFacts::Color(pass) => vec![pass],
            ExecutableFilterStepFacts::Blur {
                horizontal,
                vertical,
            } => vec![horizontal, vertical],
            ExecutableFilterStepFacts::DropShadow {
                horizontal,
                vertical,
                colorize,
                merge,
            } => vec![horizontal, vertical, colorize, merge],
        })
        .collect()
}

pub(super) fn vello_capture_raster_parameters(
    target_extent: PhysicalSize,
    antialiasing: Antialiasing,
) -> Result<RasterParameters> {
    RasterParameters::try_new(target_extent, peniko::Color::TRANSPARENT, antialiasing)
}

pub(crate) struct C08VelloCaptureEncodingHandoff<'prepared> {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: &'prepared RuntimeVelloCapture,
    initial_transform: Transform,
    antialiasing: Antialiasing,
    target_extent: PhysicalSize,
    raster_scale: f64,
    texture: &'prepared wgpu::Texture,
    view: &'prepared wgpu::TextureView,
    session: Arc<()>,
}

struct C08VelloCaptureCompletionSeal;

/// Opaque proof that one exact capture finished inside the active C08 encoding
/// session. Only a successfully encoded internal Vello capture can seal it.
#[must_use = "a capture completion receipt must return to the owning C08 scheduler"]
pub(crate) struct C08VelloCaptureCompletionReceipt {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    session: Arc<()>,
    _seal: C08VelloCaptureCompletionSeal,
}

impl C08VelloCaptureEncodingHandoff<'_> {
    pub(crate) const fn target(&self) -> RuntimeResourceId {
        self.target
    }

    const fn work(&self) -> &RuntimeVelloCapture {
        self.work
    }

    fn has_bounded_work(&self) -> bool {
        match self.work() {
            RuntimeVelloCapture::Span(span) => !span.commands.commands.is_empty(),
            RuntimeVelloCapture::ClipCoverage(coverage) => !coverage.elements.is_empty(),
        }
    }

    pub(crate) const fn initial_transform(&self) -> Transform {
        self.initial_transform
    }

    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }

    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target_extent
    }

    pub(crate) const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }

    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        self.texture
    }

    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        self.view
    }

    fn complete_after_encoded_capture(
        self,
        proof: EncodedVelloCaptureProof,
    ) -> Result<C08VelloCaptureCompletionReceipt> {
        if !proof.proves_capture_contract(
            self.target_extent,
            wgpu::TextureFormat::Rgba8Unorm,
            VELLO_CAPTURE_TEXTURE_USAGES,
            self.antialiasing,
        ) {
            return Err(preparation_error(
                "encoded C08 Vello capture proof changed its exact raster target contract",
            ));
        }
        Ok(C08VelloCaptureCompletionReceipt {
            pass: self.pass,
            target: self.target,
            session: self.session,
            _seal: C08VelloCaptureCompletionSeal,
        })
    }
}

pub(crate) struct C08ExternalOutputView<'output> {
    view: &'output wgpu::TextureView,
    format: Format,
    extent: PhysicalSize,
}

impl<'output> C08ExternalOutputView<'output> {
    pub(crate) fn try_new(
        view: &'output wgpu::TextureView,
        format: Format,
        extent: PhysicalSize,
    ) -> Result<Self> {
        if extent.width() == 0 || extent.height() == 0 {
            return Err(preparation_error(
                "C08 external output view must have a positive exact extent",
            ));
        }
        Ok(Self {
            view,
            format,
            extent,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum C08ScheduledEncodingKind {
    ClearRoot,
    VelloCapture,
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter,
    BlurHorizontalRgba,
    BlurVerticalRgba,
    BlurHorizontalSourceAlpha,
    BlurVerticalSourceAlpha,
    DropShadowColorize,
    DropShadowMerge,
    SpanSourceOver,
    LayerComposite,
    Present,
}

/// Immutable counts derived only from runtime passes that finished encoding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EncodedGpuGraphActivity {
    vello_passes: usize,
    image_passes: usize,
    composite_passes: usize,
    copy_operations: usize,
    custom_present_passes: usize,
}

impl EncodedGpuGraphActivity {
    fn from_scheduled(
        scheduled: &[C08ScheduledEncodingKind],
        destination_parent_copies: usize,
    ) -> Self {
        let mut activity = Self {
            copy_operations: destination_parent_copies,
            ..Self::default()
        };
        for kind in scheduled {
            match kind {
                C08ScheduledEncodingKind::VelloCapture => {
                    activity.vello_passes = activity.vello_passes.saturating_add(1);
                }
                C08ScheduledEncodingKind::ClearRoot
                | C08ScheduledEncodingKind::CanonicalizeCapture
                | C08ScheduledEncodingKind::ColorFilter
                | C08ScheduledEncodingKind::BlurHorizontalRgba
                | C08ScheduledEncodingKind::BlurVerticalRgba
                | C08ScheduledEncodingKind::BlurHorizontalSourceAlpha
                | C08ScheduledEncodingKind::BlurVerticalSourceAlpha
                | C08ScheduledEncodingKind::DropShadowColorize => {
                    activity.image_passes = activity.image_passes.saturating_add(1);
                }
                C08ScheduledEncodingKind::CopyBackdrop => {
                    activity.copy_operations = activity.copy_operations.saturating_add(1);
                }
                C08ScheduledEncodingKind::DropShadowMerge
                | C08ScheduledEncodingKind::SpanSourceOver
                | C08ScheduledEncodingKind::LayerComposite => {
                    activity.composite_passes = activity.composite_passes.saturating_add(1);
                }
                C08ScheduledEncodingKind::Present => {
                    activity.custom_present_passes =
                        activity.custom_present_passes.saturating_add(1);
                }
            }
        }
        activity
    }

    pub(crate) const fn vello_passes(self) -> usize {
        self.vello_passes
    }

    pub(crate) const fn image_passes(self) -> usize {
        self.image_passes
    }

    pub(crate) const fn composite_passes(self) -> usize {
        self.composite_passes
    }

    pub(crate) const fn copy_operations(self) -> usize {
        self.copy_operations
    }

    pub(crate) const fn custom_present_passes(self) -> usize {
        self.custom_present_passes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum C08CustomSpineEncodingState {
    Ready,
    Encoding,
    Complete,
    AbortOnly,
}

pub(crate) struct C08CustomSpineEncodingSummary {
    activity: EncodedGpuGraphActivity,
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
    pub(crate) layer_composite_count: usize,
    pub(crate) normal_composite_count: usize,
    pub(crate) destination_composite_count: usize,
    pub(crate) normal_composites_use_fixed_premultiplied_blend: bool,
    pub(crate) normal_composites_omit_parent_sample: bool,
    pub(crate) destination_composites_copy_full_parent: bool,
    pub(crate) destination_composites_avoid_read_write_alias: bool,
    pub(crate) layer_composites_bind_exact_resources_and_parameters: bool,
    pub(crate) layer_composites_preserve_signed_mapping: bool,
    pub(crate) color_filter_count: usize,
    pub(crate) color_filters_preserve_authored_order: bool,
    pub(crate) color_filters_bind_exact_source_spatial_and_operations: bool,
    pub(crate) color_filter_sources_and_results_are_distinct: bool,
    pub(crate) color_filters_use_validated_viewport_and_scissor: bool,
    pub(crate) color_filters_preserve_signed_texel_mapping: bool,
    pub(crate) color_filter_operation_buffers_released: bool,
    pub(crate) copy_backdrop_count: usize,
    pub(crate) copy_backdrop_binds_exact_prepared_resources: bool,
    pub(crate) copy_backdrop_source_and_result_are_distinct: bool,
    pub(crate) copy_backdrop_uses_validated_viewport_and_scissor: bool,
    pub(crate) copy_backdrop_preserves_signed_mapping: bool,
    pub(crate) c12_group_order_is_exact: bool,
    pub(crate) c12_group_resources_are_distinct: bool,
    pub(crate) blur_pass_count: usize,
    pub(crate) drop_shadow_colorize_count: usize,
    pub(crate) drop_shadow_merge_count: usize,
    pub(crate) c11_binds_exact_prepared_resources: bool,
    pub(crate) c11_uses_signed_viewport_and_scissor: bool,
    pub(crate) blur_sources_intermediates_and_results_are_distinct: bool,
    pub(crate) c11_kernels_release_at_validated_last_use: bool,
    pub(crate) c11_textures_release_at_validated_last_use: bool,
    pub(crate) drop_shadow_reads_original_source_twice: bool,
    pub(crate) original_source_releases_after_merge: bool,
    pub(crate) advances_every_pass_once: bool,
}

struct C08CustomSpineEncodingProgress {
    scheduled: Vec<C08ScheduledEncodingKind>,
    expected_capture_count: usize,
    expected_pass_count: usize,
    capture_count: usize,
    validated_capture_receipts: usize,
    bounded_capture_handoffs: bool,
    custom_encoded: usize,
    custom_completed: usize,
    completed_pass_count: usize,
    root_clear_count: usize,
    clears_full_root: bool,
    exact_spatial: bool,
    exact_external_output: bool,
    source_over_count: usize,
    layer_composite_count: usize,
    normal_composite_count: usize,
    destination_composite_count: usize,
    parent_and_result_are_distinct: bool,
    full_copy_before_bounded_render: bool,
    samples_source_with_fixed_blend: bool,
    preserves_signed_origin: bool,
    normal_fixed_blend: bool,
    normal_omits_parent_sample: bool,
    destination_copies_full_parent: bool,
    destination_avoids_alias: bool,
    layer_bindings_are_exact: bool,
    layer_signed_mapping_is_exact: bool,
    color_filter_count: usize,
    color_filters_preserve_authored_order: bool,
    color_filter_bindings_are_exact: bool,
    color_filter_sources_and_results_are_distinct: bool,
    color_filter_regions_are_validated: bool,
    color_filter_signed_texel_mapping_is_exact: bool,
    color_filter_operation_buffers_released: bool,
    copy_backdrop_count: usize,
    copy_backdrop_bindings_are_exact: bool,
    copy_backdrop_allocations_are_distinct: bool,
    copy_backdrop_regions_are_validated: bool,
    copy_backdrop_signed_mapping_is_exact: bool,
    blur_pass_count: usize,
    drop_shadow_colorize_count: usize,
    drop_shadow_merge_count: usize,
    c11_bindings_are_exact: bool,
    c11_regions_are_exact: bool,
    blur_allocations_are_distinct: bool,
    c11_kernels_released: bool,
    c11_textures_released: bool,
    shadow_source_read_twice: bool,
    shadow_source_released_after_merge: bool,
}

impl C08CustomSpineEncodingProgress {
    fn new(expected_pass_count: usize, expected_capture_count: usize) -> Self {
        Self {
            scheduled: Vec::with_capacity(expected_pass_count),
            expected_capture_count,
            expected_pass_count,
            capture_count: 0,
            validated_capture_receipts: 0,
            bounded_capture_handoffs: true,
            custom_encoded: 0,
            custom_completed: 0,
            completed_pass_count: 0,
            root_clear_count: 0,
            clears_full_root: true,
            exact_spatial: true,
            exact_external_output: false,
            source_over_count: 0,
            layer_composite_count: 0,
            normal_composite_count: 0,
            destination_composite_count: 0,
            parent_and_result_are_distinct: true,
            full_copy_before_bounded_render: true,
            samples_source_with_fixed_blend: true,
            preserves_signed_origin: true,
            normal_fixed_blend: true,
            normal_omits_parent_sample: true,
            destination_copies_full_parent: true,
            destination_avoids_alias: true,
            layer_bindings_are_exact: true,
            layer_signed_mapping_is_exact: true,
            color_filter_count: 0,
            color_filters_preserve_authored_order: true,
            color_filter_bindings_are_exact: true,
            color_filter_sources_and_results_are_distinct: true,
            color_filter_regions_are_validated: true,
            color_filter_signed_texel_mapping_is_exact: true,
            color_filter_operation_buffers_released: true,
            copy_backdrop_count: 0,
            copy_backdrop_bindings_are_exact: true,
            copy_backdrop_allocations_are_distinct: true,
            copy_backdrop_regions_are_validated: true,
            copy_backdrop_signed_mapping_is_exact: true,
            blur_pass_count: 0,
            drop_shadow_colorize_count: 0,
            drop_shadow_merge_count: 0,
            c11_bindings_are_exact: true,
            c11_regions_are_exact: true,
            blur_allocations_are_distinct: true,
            c11_kernels_released: true,
            c11_textures_released: true,
            shadow_source_read_twice: true,
            shadow_source_released_after_merge: true,
        }
    }

    fn record_custom_completion(&mut self, kind: C08ScheduledEncodingKind) {
        self.scheduled.push(kind);
        self.custom_encoded = self.custom_encoded.saturating_add(1);
        self.custom_completed = self.custom_completed.saturating_add(1);
        self.completed_pass_count = self.completed_pass_count.saturating_add(1);
    }

    fn record_capture_completion(&mut self) {
        self.scheduled.push(C08ScheduledEncodingKind::VelloCapture);
        self.capture_count = self.capture_count.saturating_add(1);
        self.validated_capture_receipts = self.validated_capture_receipts.saturating_add(1);
        self.completed_pass_count = self.completed_pass_count.saturating_add(1);
    }

    fn finish(self, prepared: &PreparedGraph<'_>) -> C08CustomSpineEncodingSummary {
        let total_composites = self
            .source_over_count
            .saturating_add(self.layer_composite_count);
        let c12 = c12_execution_receipt(prepared);
        #[cfg(test)]
        finish_c08_encoding_observations_for_test(
            &self.scheduled,
            self.expected_capture_count,
            self.capture_count,
            self.color_filter_count,
            prepared,
        );
        C08CustomSpineEncodingSummary {
            activity: EncodedGpuGraphActivity::from_scheduled(
                &self.scheduled,
                self.destination_composite_count,
            ),
            encodes_custom_passes_in_order: c08_scheduled_encoding_order_is_exact(
                &self.scheduled,
                &prepared.plan.passes,
            ),
            clears_full_root_once: self.root_clear_count == 1 && self.clears_full_root,
            uses_exact_prepared_spatial_mapping: self.exact_spatial,
            presents_to_exact_external_output: self.exact_external_output,
            exposes_bounded_capture_handoff: self.expected_capture_count > 0
                && self.capture_count == self.expected_capture_count
                && self.bounded_capture_handoffs,
            validates_checked_capture_completion: self.validated_capture_receipts
                == self.expected_capture_count,
            completes_custom_passes_after_encoding: self.custom_encoded > 0
                && self.custom_completed == self.custom_encoded,
            parent_and_result_are_distinct: total_composites > 0
                && self.parent_and_result_are_distinct,
            copies_full_parent_before_bounded_source_render: total_composites > 0
                && self.full_copy_before_bounded_render,
            samples_only_source_with_fixed_premultiplied_blend: (self.source_over_count > 0
                || self.normal_composite_count > 0
                || self.destination_composite_count > 0)
                && self.samples_source_with_fixed_blend,
            preserves_signed_source_origin: total_composites > 0 && self.preserves_signed_origin,
            keeps_cache_update_provisional: prepared.pass_cache_update.is_some(),
            layer_composite_count: self.layer_composite_count,
            normal_composite_count: self.normal_composite_count,
            destination_composite_count: self.destination_composite_count,
            normal_composites_use_fixed_premultiplied_blend: self.normal_fixed_blend,
            normal_composites_omit_parent_sample: self.normal_omits_parent_sample,
            destination_composites_copy_full_parent: self.destination_copies_full_parent,
            destination_composites_avoid_read_write_alias: self.destination_avoids_alias,
            layer_composites_bind_exact_resources_and_parameters: self.layer_bindings_are_exact,
            layer_composites_preserve_signed_mapping: self.layer_signed_mapping_is_exact,
            color_filter_count: self.color_filter_count,
            color_filters_preserve_authored_order: self.color_filter_count > 0
                && self.color_filters_preserve_authored_order,
            color_filters_bind_exact_source_spatial_and_operations: self.color_filter_count > 0
                && self.color_filter_bindings_are_exact,
            color_filter_sources_and_results_are_distinct: self.color_filter_count > 0
                && self.color_filter_sources_and_results_are_distinct,
            color_filters_use_validated_viewport_and_scissor: self.color_filter_count > 0
                && self.color_filter_regions_are_validated,
            color_filters_preserve_signed_texel_mapping: self.color_filter_count > 0
                && self.color_filter_signed_texel_mapping_is_exact,
            color_filter_operation_buffers_released: self.color_filter_count > 0
                && self.color_filter_operation_buffers_released,
            copy_backdrop_count: self.copy_backdrop_count,
            copy_backdrop_binds_exact_prepared_resources: self.copy_backdrop_count > 0
                && self.copy_backdrop_bindings_are_exact,
            copy_backdrop_source_and_result_are_distinct: self.copy_backdrop_count > 0
                && self.copy_backdrop_allocations_are_distinct,
            copy_backdrop_uses_validated_viewport_and_scissor: self.copy_backdrop_count > 0
                && self.copy_backdrop_regions_are_validated,
            copy_backdrop_preserves_signed_mapping: self.copy_backdrop_count > 0
                && self.copy_backdrop_signed_mapping_is_exact,
            c12_group_order_is_exact: c12.group_order_is_exact,
            c12_group_resources_are_distinct: c12.group_resources_are_distinct,
            blur_pass_count: self.blur_pass_count,
            drop_shadow_colorize_count: self.drop_shadow_colorize_count,
            drop_shadow_merge_count: self.drop_shadow_merge_count,
            c11_binds_exact_prepared_resources: self.c11_bindings_are_exact,
            c11_uses_signed_viewport_and_scissor: self.c11_regions_are_exact,
            blur_sources_intermediates_and_results_are_distinct: self.blur_allocations_are_distinct,
            c11_kernels_release_at_validated_last_use: self.c11_kernels_released,
            c11_textures_release_at_validated_last_use: self.c11_textures_released,
            drop_shadow_reads_original_source_twice: self.shadow_source_read_twice,
            original_source_releases_after_merge: self.shadow_source_released_after_merge,
            advances_every_pass_once: self.completed_pass_count == self.expected_pass_count
                && prepared.next_pass == self.expected_pass_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct C12ExecutionReceipt {
    group_order_is_exact: bool,
    group_resources_are_distinct: bool,
}

fn c12_execution_receipt(prepared: &PreparedGraph<'_>) -> C12ExecutionReceipt {
    let Some(execution) = prepared.c12_execution.as_ref() else {
        return C12ExecutionReceipt::default();
    };
    let [backdrop] = execution.backdrops.as_slice() else {
        return C12ExecutionReceipt::default();
    };
    let positions = prepared
        .plan
        .passes
        .iter()
        .enumerate()
        .map(|(position, pass)| (pass.runtime.id, position))
        .collect::<BTreeMap<_, _>>();
    let Some(copy) = positions.get(&backdrop.copy).copied() else {
        return C12ExecutionReceipt::default();
    };
    let Some(clear) = positions.get(&backdrop.group_clear).copied() else {
        return C12ExecutionReceipt::default();
    };
    let Some(backdrop_composite) = positions.get(&backdrop.backdrop_composite).copied() else {
        return C12ExecutionReceipt::default();
    };
    let Some(outer) = positions.get(&backdrop.outer_composite).copied() else {
        return C12ExecutionReceipt::default();
    };
    let filters = backdrop_filter_passes(&backdrop.filter_steps);
    let filters_are_ordered = filters.iter().all(|pass| {
        positions
            .get(pass)
            .is_some_and(|position| copy < *position && *position < clear)
    });
    let foreground_is_ordered = backdrop.foreground_composite.is_none_or(|pass| {
        positions
            .get(&pass)
            .is_some_and(|position| backdrop_composite < *position && *position < outer)
    });
    let mut resources = BTreeSet::from([
        backdrop.completed_parent,
        backdrop.copied,
        backdrop.filtered,
        backdrop.completed_group,
        backdrop.result,
    ]);
    if let Some(foreground) = backdrop.foreground {
        resources.insert(foreground);
    }
    let expected_resource_count = 5usize.saturating_add(usize::from(backdrop.foreground.is_some()));
    C12ExecutionReceipt {
        group_order_is_exact: filters_are_ordered
            && clear < backdrop_composite
            && backdrop_composite < outer
            && foreground_is_ordered,
        group_resources_are_distinct: resources.len() == expected_resource_count,
    }
}

impl C08CustomSpineEncodingSummary {
    pub(crate) const fn activity(&self) -> EncodedGpuGraphActivity {
        self.activity
    }

    fn proves_complete_submission(&self) -> bool {
        let common = self.encodes_custom_passes_in_order
            && self.clears_full_root_once
            && self.uses_exact_prepared_spatial_mapping
            && self.presents_to_exact_external_output
            && self.exposes_bounded_capture_handoff
            && self.validates_checked_capture_completion
            && self.completes_custom_passes_after_encoding
            && self.parent_and_result_are_distinct
            && self.copies_full_parent_before_bounded_source_render
            && self.samples_only_source_with_fixed_premultiplied_blend
            && self.preserves_signed_source_origin
            && self.keeps_cache_update_provisional
            && self.advances_every_pass_once;
        let exact_layers = self
            .normal_composite_count
            .saturating_add(self.destination_composite_count)
            == self.layer_composite_count
            && (self.normal_composite_count == 0
                || (self.normal_composites_use_fixed_premultiplied_blend
                    && self.normal_composites_omit_parent_sample))
            && (self.destination_composite_count == 0
                || (self.destination_composites_copy_full_parent
                    && self.destination_composites_avoid_read_write_alias))
            && (self.layer_composite_count == 0
                || (self.layer_composites_bind_exact_resources_and_parameters
                    && self.layer_composites_preserve_signed_mapping));
        let spatial_pass_count = self
            .blur_pass_count
            .saturating_add(self.drop_shadow_colorize_count)
            .saturating_add(self.drop_shadow_merge_count);
        let exact_c08 = self.layer_composite_count == 0
            && self.color_filter_count == 0
            && self.copy_backdrop_count == 0
            && spatial_pass_count == 0;
        let exact_c09 = self.layer_composite_count > 0
            && self.color_filter_count == 0
            && self.copy_backdrop_count == 0
            && spatial_pass_count == 0
            && self
                .normal_composite_count
                .saturating_add(self.destination_composite_count)
                == self.layer_composite_count
            && (self.normal_composite_count == 0
                || (self.normal_composites_use_fixed_premultiplied_blend
                    && self.normal_composites_omit_parent_sample))
            && (self.destination_composite_count == 0
                || (self.destination_composites_copy_full_parent
                    && self.destination_composites_avoid_read_write_alias))
            && self.layer_composites_bind_exact_resources_and_parameters
            && self.layer_composites_preserve_signed_mapping;
        let exact_c10 = self.color_filter_count > 0
            && self.copy_backdrop_count == 0
            && spatial_pass_count == 0
            && self.color_filters_preserve_authored_order
            && self.color_filters_bind_exact_source_spatial_and_operations
            && self.color_filter_sources_and_results_are_distinct
            && self.color_filters_use_validated_viewport_and_scissor
            && self.color_filters_preserve_signed_texel_mapping
            && self.color_filter_operation_buffers_released
            && exact_layers;
        let exact_drop_shadows = (self.drop_shadow_colorize_count == 0
            && self.drop_shadow_merge_count == 0)
            || (self.drop_shadow_colorize_count > 0
                && self.drop_shadow_colorize_count == self.drop_shadow_merge_count
                && self.drop_shadow_reads_original_source_twice
                && self.original_source_releases_after_merge);
        let exact_color_filters = self.color_filter_count == 0
            || (self.color_filters_preserve_authored_order
                && self.color_filters_bind_exact_source_spatial_and_operations
                && self.color_filter_sources_and_results_are_distinct
                && self.color_filters_use_validated_viewport_and_scissor
                && self.color_filters_preserve_signed_texel_mapping
                && self.color_filter_operation_buffers_released);
        let exact_spatial_passes = (self.blur_pass_count == 0
            && self.drop_shadow_colorize_count == 0
            && self.drop_shadow_merge_count == 0)
            || (self.blur_pass_count > 0
                && exact_drop_shadows
                && self.c11_binds_exact_prepared_resources
                && self.c11_uses_signed_viewport_and_scissor
                && self.blur_sources_intermediates_and_results_are_distinct
                && self.c11_kernels_release_at_validated_last_use
                && self.c11_textures_release_at_validated_last_use);
        let exact_c11 = self.copy_backdrop_count == 0
            && self.blur_pass_count > 0
            && exact_layers
            && exact_color_filters
            && exact_spatial_passes;
        let exact_c12 = self.copy_backdrop_count == 1
            && self.color_filter_count.saturating_add(self.blur_pass_count) > 0
            && self.copy_backdrop_binds_exact_prepared_resources
            && self.copy_backdrop_source_and_result_are_distinct
            && self.copy_backdrop_uses_validated_viewport_and_scissor
            && self.copy_backdrop_preserves_signed_mapping
            && self.c12_group_order_is_exact
            && self.c12_group_resources_are_distinct
            && exact_layers
            && exact_color_filters
            && exact_spatial_passes;
        common && (exact_c08 || exact_c09 || exact_c10 || exact_c11 || exact_c12)
    }
}

struct C08EncodedCaptureResult {
    receipt: C08VelloCaptureCompletionReceipt,
}

struct C08VelloCaptureEncodingContext<'encoding, 'device> {
    engine: &'device VelloEngineState,
    resources: &'device ResourceManager,
    queue: &'device wgpu::Queue,
    scope: &'encoding mut ActiveVelloEncodingScope<'device>,
    leases: &'encoding mut VelloResourceLeaseAggregate,
}

/// Owns the scope-clean capture leases until T5 gives them to the transaction
/// submission payload. Dropping this value aborts every capture lease.
#[must_use = "encoded C08 graph captures must remain pending until transaction resolution"]
pub(crate) struct C08PendingGraphEncoding {
    pub(super) summary: C08CustomSpineEncodingSummary,
    pub(super) resources: PendingVelloResourceCommit,
    session: Arc<()>,
}

#[must_use = "prepared C08 frame state must commit only after graph transaction success"]
pub(crate) struct PendingC08PreparedFrameCommit {
    pub(super) frame_scope: FrameResourceScope,
    pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

/// Sealed one-shot state proving that the prepared frame and provisional cache
/// can still complete without an accounting or cache-identity fault.
#[must_use = "accounting-ready C08 prepared state must be committed or aborted on drop"]
pub(crate) struct AccountingReadyC08PreparedFrameCommit {
    frame_scope: FrameResourceScope,
    pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

impl PendingC08PreparedFrameCommit {
    pub(crate) fn into_accounting_ready(
        self,
        pass_cache: &DevicePassCache,
    ) -> Result<AccountingReadyC08PreparedFrameCommit> {
        self.pass_cache_update.ensure_commit_ready(pass_cache)?;
        self.frame_scope.ensure_commit_ready(&[])?;
        Ok(AccountingReadyC08PreparedFrameCommit {
            frame_scope: self.frame_scope,
            pass_cache_update: self.pass_cache_update,
        })
    }
}

impl AccountingReadyC08PreparedFrameCommit {
    pub(crate) fn ensure_commit_ready(&self, pass_cache: &DevicePassCache) -> Result<()> {
        self.pass_cache_update.ensure_commit_ready(pass_cache)?;
        self.frame_scope.ensure_commit_ready(&[])
    }

    pub(crate) fn commit(self, pass_cache: &mut DevicePassCache) -> Result<FrameCleanup> {
        self.ensure_commit_ready(pass_cache)?;
        let frame_cleanup = self.frame_scope.finish_checked()?;
        self.pass_cache_update.commit(pass_cache)?;
        Ok(frame_cleanup)
    }
}

#[must_use = "C08 graph submission state must remain owned by one transaction payload"]
pub(crate) struct C08PreparedGraphSubmission {
    capture_resources: PendingVelloResourceCommit,
    prepared_frame: PendingC08PreparedFrameCommit,
    activity: EncodedGpuGraphActivity,
}

impl C08PreparedGraphSubmission {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PendingVelloResourceCommit,
        PendingC08PreparedFrameCommit,
        EncodedGpuGraphActivity,
    ) {
        (self.capture_resources, self.prepared_frame, self.activity)
    }
}

#[derive(Clone)]
struct C08PreparedPassEncodingRequest {
    id: RuntimePassId,
    kind: RuntimePassKind,
    reads: Vec<RuntimeReadBinding>,
    result: RuntimeResultBinding,
    spatial_uniform: Option<PassSpatialUniformBytes>,
    blur_edge_parameters: Option<BlurEdgeParameterBytes>,
    color_filter_operations: Option<ColorFilterOperationBytes>,
    drop_shadow_parameters: Option<DropShadowParameterBytes>,
    composite_parameters: Option<CompositeParameterBytes>,
    cache_keys: Option<RuntimePassCacheKeys>,
    releases: Vec<RuntimeResourceId>,
    kernel_releases: Vec<GaussianKernelKey>,
}

impl From<&RuntimePassPreparationRequest> for C08PreparedPassEncodingRequest {
    fn from(request: &RuntimePassPreparationRequest) -> Self {
        Self {
            id: request.runtime.id,
            kind: request.runtime.kind.clone(),
            reads: request.runtime.reads.clone(),
            result: request.runtime.result,
            spatial_uniform: request.spatial_uniform.clone(),
            blur_edge_parameters: request.blur_edge_parameters.clone(),
            color_filter_operations: request.color_filter_operations.clone(),
            drop_shadow_parameters: request.drop_shadow_parameters.clone(),
            composite_parameters: request.composite_parameters.clone(),
            cache_keys: request.cache_keys.clone(),
            releases: request.runtime.releases.clone(),
            kernel_releases: request.kernel_releases.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct C08RenderRegion {
    viewport_x: f32,
    viewport_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    scissor_x: u32,
    scissor_y: u32,
    scissor_width: u32,
    scissor_height: u32,
    unclipped_x: f64,
    unclipped_y: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct C08PassEncodingFacts {
    full_target: bool,
    exact_spatial_uniform: bool,
    external_output_exact: bool,
    parent_and_result_distinct: bool,
    copied_full_parent_before_render: bool,
    sampled_only_source: bool,
    fixed_source_over_blend: bool,
    preserved_signed_source_origin: bool,
}

#[derive(Clone, Copy, Debug)]
struct C09LayerCompositeEncodingFacts {
    normal_path: bool,
    destination_path: bool,
    fixed_premultiplied_blend: bool,
    omits_parent_sample: bool,
    copied_full_parent: bool,
    avoids_read_write_alias: bool,
    exact_resources_and_parameters: bool,
    preserved_signed_mapping: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct C10ColorFilterEncodingFacts {
    exact_operation_bytes: bool,
    exact_source_spatial_and_operations: bool,
    source_and_result_are_distinct: bool,
    validated_viewport_and_scissor: bool,
    preserved_signed_texel_mapping: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct C12CopyBackdropEncodingFacts {
    exact_prepared_bindings: bool,
    source_and_result_are_distinct: bool,
    validated_viewport_and_scissor: bool,
    preserved_signed_mapping: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct C11BlurEncodingFacts {
    exact_prepared_bindings: bool,
    distinct_source_and_result: bool,
    validated_region: bool,
    preserved_signed_mapping: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct C11DropShadowColorizeEncodingFacts {
    exact_prepared_bindings: bool,
    distinct_source_and_result: bool,
    validated_region: bool,
    preserved_signed_mapping: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct C11DropShadowMergeEncodingFacts {
    exact_prepared_bindings: bool,
    distinct_source_shadow_and_result: bool,
    validated_region: bool,
    preserved_signed_mapping: bool,
    fixed_source_over_blend: bool,
    reads_original_source_and_shadow: bool,
}

struct PreparedC10ColorFilterEncoding<'prepared> {
    source_binding: PreparedTextureBinding<'prepared>,
    target_binding: PreparedTextureBinding<'prepared>,
    objects: ProvisionalColorFilterPassObjects<'prepared>,
    spatial: &'prepared PassSpatialUniformBytes,
    operation_buffer: &'prepared wgpu::Buffer,
    region: C08RenderRegion,
    facts: C10ColorFilterEncodingFacts,
}

struct C09CompositeSemantic<'prepared> {
    transform: Transform,
    parameters: &'prepared RuntimeLayerCompositeParameters,
    parent: RuntimeReadBinding,
    source: RuntimeReadBinding,
    clip: Option<RuntimeReadBinding>,
    alpha_mask: Option<RuntimeReadBinding>,
    target: RuntimeResourceId,
    normal_path: bool,
    destination_path: bool,
}

struct C09CompositeBindings<'prepared> {
    semantic: C09CompositeSemantic<'prepared>,
    parent: PreparedTextureBinding<'prepared>,
    source: PreparedTextureBinding<'prepared>,
    target: PreparedTextureBinding<'prepared>,
    clip: Option<PreparedTextureBinding<'prepared>>,
    mask: Option<PreparedTextureBinding<'prepared>>,
    parent_spatial: RuntimeSpatialDescriptor,
    source_spatial: RuntimeSpatialDescriptor,
    target_spatial: RuntimeSpatialDescriptor,
    parent_and_result_are_distinct: bool,
    sampled_allocations_are_distinct: bool,
}

struct C08SampledRenderTarget<'target> {
    view: &'target wgpu::TextureView,
    extent: PhysicalSize,
    region: Option<C08RenderRegion>,
    load: wgpu::LoadOp<wgpu::Color>,
    label: &'static str,
}

impl C08RenderRegion {
    fn full(extent: PhysicalSize) -> Result<Self> {
        if extent.width() == 0 || extent.height() == 0 {
            return Err(preparation_error(
                "the C08 render region requires a positive target extent",
            ));
        }
        let viewport_width = extent.width() as f32;
        let viewport_height = extent.height() as f32;
        if !viewport_width.is_finite() || !viewport_height.is_finite() {
            return Err(preparation_error(
                "the C08 render extent cannot be represented by WGPU viewport coordinates",
            ));
        }
        Ok(Self {
            viewport_x: 0.0,
            viewport_y: 0.0,
            viewport_width,
            viewport_height,
            scissor_x: 0,
            scissor_y: 0,
            scissor_width: extent.width(),
            scissor_height: extent.height(),
            unclipped_x: 0.0,
            unclipped_y: 0.0,
        })
    }

    fn bounded_source(
        source: RuntimeSpatialDescriptor,
        destination: RuntimeSpatialDescriptor,
    ) -> Result<Option<Self>> {
        let source_width = f64::from(source.device_extent.width()) / source.raster_scale;
        let source_height = f64::from(source.device_extent.height()) / source.raster_scale;
        let unclipped_x =
            (source.texel_origin.x() - destination.texel_origin.x()) * destination.raster_scale;
        let unclipped_y =
            (source.texel_origin.y() - destination.texel_origin.y()) * destination.raster_scale;
        let unclipped_end_x = (source.texel_origin.x() + source_width
            - destination.texel_origin.x())
            * destination.raster_scale;
        let unclipped_end_y = (source.texel_origin.y() + source_height
            - destination.texel_origin.y())
            * destination.raster_scale;
        Self::bounded_unclipped(
            [unclipped_x, unclipped_y, unclipped_end_x, unclipped_end_y],
            destination,
            "the C08 signed bounded render mapping is non-finite",
            "the C08 bounded viewport or scissor cannot represent its signed mapping",
        )
    }

    fn bounded_transformed_source(
        source: RuntimeSpatialDescriptor,
        destination: RuntimeSpatialDescriptor,
        source_to_destination: Transform,
    ) -> Result<(Option<Self>, [f64; 4])> {
        let bounds =
            c09_transformed_source_device_bounds(source, destination, source_to_destination)?;
        let region = Self::bounded_unclipped(
            bounds,
            destination,
            "the C09 transformed bounded render mapping is non-finite",
            "the C09 transformed viewport or scissor cannot represent its signed mapping",
        )?;
        Ok((region, bounds))
    }

    fn bounded_unclipped(
        [unclipped_x, unclipped_y, unclipped_end_x, unclipped_end_y]: [f64; 4],
        destination: RuntimeSpatialDescriptor,
        non_finite_message: &'static str,
        invalid_region_message: &'static str,
    ) -> Result<Option<Self>> {
        if [unclipped_x, unclipped_y, unclipped_end_x, unclipped_end_y]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(preparation_error(non_finite_message));
        }

        let destination_width = f64::from(destination.device_extent.width());
        let destination_height = f64::from(destination.device_extent.height());
        let clipped_x = unclipped_x.max(0.0).min(destination_width);
        let clipped_y = unclipped_y.max(0.0).min(destination_height);
        let clipped_end_x = unclipped_end_x.max(0.0).min(destination_width);
        let clipped_end_y = unclipped_end_y.max(0.0).min(destination_height);
        if clipped_end_x <= clipped_x || clipped_end_y <= clipped_y {
            return Ok(None);
        }

        let scissor_x = clipped_x.floor() as u32;
        let scissor_y = clipped_y.floor() as u32;
        let scissor_end_x = clipped_end_x.ceil() as u32;
        let scissor_end_y = clipped_end_y.ceil() as u32;
        let viewport_x = clipped_x as f32;
        let viewport_y = clipped_y as f32;
        let viewport_width = (clipped_end_x - clipped_x) as f32;
        let viewport_height = (clipped_end_y - clipped_y) as f32;
        if !viewport_x.is_finite()
            || !viewport_y.is_finite()
            || !viewport_width.is_finite()
            || !viewport_height.is_finite()
            || viewport_width <= 0.0
            || viewport_height <= 0.0
            || scissor_end_x <= scissor_x
            || scissor_end_y <= scissor_y
        {
            return Err(preparation_error(invalid_region_message));
        }
        Ok(Some(Self {
            viewport_x,
            viewport_y,
            viewport_width,
            viewport_height,
            scissor_x,
            scissor_y,
            scissor_width: scissor_end_x - scissor_x,
            scissor_height: scissor_end_y - scissor_y,
            unclipped_x,
            unclipped_y,
        }))
    }
}

fn c09_transformed_source_device_bounds(
    source: RuntimeSpatialDescriptor,
    destination: RuntimeSpatialDescriptor,
    source_to_destination: Transform,
) -> Result<[f64; 4]> {
    let source_width = f64::from(source.device_extent.width()) / source.raster_scale;
    let source_height = f64::from(source.device_extent.height()) / source.raster_scale;
    let source_end_x = source.texel_origin.x() + source_width;
    let source_end_y = source.texel_origin.y() + source_height;
    let [a, b, c, d, e, f] = source_to_destination.as_array();
    let mut minimum_x = f64::INFINITY;
    let mut minimum_y = f64::INFINITY;
    let mut maximum_x = f64::NEG_INFINITY;
    let mut maximum_y = f64::NEG_INFINITY;
    for (source_x, source_y) in [
        (source.texel_origin.x(), source.texel_origin.y()),
        (source_end_x, source.texel_origin.y()),
        (source.texel_origin.x(), source_end_y),
        (source_end_x, source_end_y),
    ] {
        let destination_x = a * source_x + c * source_y + e;
        let destination_y = b * source_x + d * source_y + f;
        let device_x = (destination_x - destination.texel_origin.x()) * destination.raster_scale;
        let device_y = (destination_y - destination.texel_origin.y()) * destination.raster_scale;
        if !device_x.is_finite() || !device_y.is_finite() {
            return Err(preparation_error(
                "the C09 transformed source bounds are non-finite",
            ));
        }
        minimum_x = minimum_x.min(device_x);
        minimum_y = minimum_y.min(device_y);
        maximum_x = maximum_x.max(device_x);
        maximum_y = maximum_y.max(device_y);
    }
    Ok([minimum_x, minimum_y, maximum_x, maximum_y])
}

fn encode_c11_full_target_pass(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    region: C08RenderRegion,
    label: &'static str,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_viewport(
        region.viewport_x,
        region.viewport_y,
        region.viewport_width,
        region.viewport_height,
        0.0,
        1.0,
    );
    pass.set_scissor_rect(
        region.scissor_x,
        region.scissor_y,
        region.scissor_width,
        region.scissor_height,
    );
    pass.draw(0..3, 0..1);
}

fn c11_full_region_is_exact(region: C08RenderRegion, extent: PhysicalSize) -> bool {
    region.viewport_x == 0.0
        && region.viewport_y == 0.0
        && region.viewport_width == extent.width() as f32
        && region.viewport_height == extent.height() as f32
        && region.scissor_x == 0
        && region.scissor_y == 0
        && region.scissor_width == extent.width()
        && region.scissor_height == extent.height()
}

fn exact_c08_read(
    request: &C08PreparedPassEncodingRequest,
    role: RuntimeReadRole,
) -> Result<&RuntimeReadBinding> {
    let mut matching = request.reads.iter().filter(|read| read.role == role);
    let read = matching
        .next()
        .ok_or_else(|| preparation_error("the C08 prepared source binding is missing"))?;
    if matching.next().is_some() {
        return Err(preparation_error(
            "the C08 prepared source binding is duplicated",
        ));
    }
    Ok(read)
}

fn c08_scheduled_encoding_order_is_exact(
    scheduled: &[C08ScheduledEncodingKind],
    passes: &[RuntimePassPreparationRequest],
) -> bool {
    scheduled.len() == passes.len()
        && scheduled.iter().zip(passes).all(|(scheduled, pass)| {
            let expected = match &pass.runtime.kind {
                RuntimePassKind::ClearRoot { .. } => C08ScheduledEncodingKind::ClearRoot,
                RuntimePassKind::VelloCapture(Some(_)) => C08ScheduledEncodingKind::VelloCapture,
                RuntimePassKind::CanonicalizeCapture => {
                    C08ScheduledEncodingKind::CanonicalizeCapture
                }
                RuntimePassKind::CopyBackdrop => C08ScheduledEncodingKind::CopyBackdrop,
                RuntimePassKind::ColorFilter(Some(_)) => C08ScheduledEncodingKind::ColorFilter,
                RuntimePassKind::BlurHorizontal(Some(blur)) => match blur.input {
                    RuntimeBlurInput::Rgba => C08ScheduledEncodingKind::BlurHorizontalRgba,
                    RuntimeBlurInput::SourceAlpha => {
                        C08ScheduledEncodingKind::BlurHorizontalSourceAlpha
                    }
                },
                RuntimePassKind::BlurVertical(Some(blur)) => match blur.input {
                    RuntimeBlurInput::Rgba => C08ScheduledEncodingKind::BlurVerticalRgba,
                    RuntimeBlurInput::SourceAlpha => {
                        C08ScheduledEncodingKind::BlurVerticalSourceAlpha
                    }
                },
                RuntimePassKind::DropShadowColorize(Some(_)) => {
                    C08ScheduledEncodingKind::DropShadowColorize
                }
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::SpanSourceOver,
                    ..
                })) => C08ScheduledEncodingKind::SpanSourceOver,
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::DropShadow,
                    ..
                })) => C08ScheduledEncodingKind::DropShadowMerge,
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::Layer { .. },
                    ..
                })) => C08ScheduledEncodingKind::LayerComposite,
                RuntimePassKind::Present => C08ScheduledEncodingKind::Present,
                RuntimePassKind::VelloCapture(None)
                | RuntimePassKind::ColorFilter(None)
                | RuntimePassKind::BlurHorizontal(None)
                | RuntimePassKind::BlurVertical(None)
                | RuntimePassKind::DropShadowColorize(None)
                | RuntimePassKind::Composite(None) => return false,
            };
            *scheduled == expected
        })
}

fn c08_capture_handoff_is_bounded(handoff: &C08VelloCaptureEncodingHandoff<'_>) -> bool {
    handoff.has_bounded_work()
        && handoff.target_extent().width() > 0
        && handoff.target_extent().height() > 0
        && handoff.texture().width() == handoff.target_extent().width()
        && handoff.texture().height() == handoff.target_extent().height()
        && handoff.texture().depth_or_array_layers() == 1
        && handoff.texture().mip_level_count() == 1
        && handoff.texture().sample_count() == 1
        && handoff.raster_scale().is_finite()
        && handoff.raster_scale() > 0.0
        && handoff
            .initial_transform()
            .as_array()
            .iter()
            .all(|value| value.is_finite())
}

fn c08_spatial_uniform_preserves_source_origin(
    bytes: &PassSpatialUniformBytes,
    source: RuntimeSpatialDescriptor,
) -> bool {
    let encoded_x = f32::from_le_bytes(bytes.as_bytes()[0..4].try_into().unwrap_or([0; 4]));
    let encoded_y = f32::from_le_bytes(bytes.as_bytes()[4..8].try_into().unwrap_or([0; 4]));
    encoded_x == source.texel_origin.x() as f32 && encoded_y == source.texel_origin.y() as f32
}

fn close_f64(left: f64, right: f64) -> bool {
    let tolerance = f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0;
    (left - right).abs() <= tolerance
}

fn validate_c10_render_region(
    spatial: &PassSpatialUniformBytes,
    source: RuntimeSpatialDescriptor,
    target: RuntimeSpatialDescriptor,
) -> Result<(C08RenderRegion, bool, bool)> {
    let region = C08RenderRegion::bounded_source(source, target)?
        .ok_or_else(|| preparation_error("the C10 color pass has an empty bounded region"))?;
    let viewport_and_scissor = region.scissor_x.saturating_add(region.scissor_width)
        <= target.device_extent.width()
        && region.scissor_y.saturating_add(region.scissor_height) <= target.device_extent.height()
        && region.viewport_width > 0.0
        && region.viewport_height > 0.0;
    let signed_texel_mapping = c08_spatial_uniform_preserves_source_origin(spatial, source)
        && close_f64(region.unclipped_x, 0.0)
        && close_f64(region.unclipped_y, 0.0)
        && source.texel_origin == target.texel_origin
        && source.device_origin == target.device_origin;
    if !viewport_and_scissor || !signed_texel_mapping {
        return Err(preparation_error(
            "the C10 bounded viewport or signed texel mapping changed after validation",
        ));
    }
    Ok((region, viewport_and_scissor, signed_texel_mapping))
}

fn encode_c09_composite_region(
    encoder: &mut wgpu::CommandEncoder,
    target: &PreparedTextureBinding<'_>,
    objects: &ProvisionalCompositePassObjects<'_>,
    bind_group: &wgpu::BindGroup,
    region: Option<C08RenderRegion>,
    target_spatial: RuntimeSpatialDescriptor,
) -> Result<()> {
    let Some(region) = region else {
        return Ok(());
    };
    if region.scissor_x.saturating_add(region.scissor_width) > target_spatial.device_extent.width()
        || region.scissor_y.saturating_add(region.scissor_height)
            > target_spatial.device_extent.height()
    {
        return Err(preparation_error(
            "the C09 bounded composite exceeds its exact parent extent",
        ));
    }
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Surgeist C09 bounded layer composite"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target.view(),
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
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_viewport(
        region.viewport_x,
        region.viewport_y,
        region.viewport_width,
        region.viewport_height,
        0.0,
        1.0,
    );
    pass.set_scissor_rect(
        region.scissor_x,
        region.scissor_y,
        region.scissor_width,
        region.scissor_height,
    );
    pass.draw(0..3, 0..1);
    Ok(())
}

fn copy_c09_composite_parent(
    encoder: &mut wgpu::CommandEncoder,
    parent: &PreparedTextureBinding<'_>,
    target: &PreparedTextureBinding<'_>,
    extent: wgpu::Extent3d,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: parent.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: target.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        extent,
    );
}

fn c09_composite_copy_extent(spatial: RuntimeSpatialDescriptor) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: spatial.device_extent.width(),
        height: spatial.device_extent.height(),
        depth_or_array_layers: 1,
    }
}

fn c09_composite_region_mapping(
    spatial: &PassSpatialUniformBytes,
    source: RuntimeSpatialDescriptor,
    target: RuntimeSpatialDescriptor,
    transform: Transform,
) -> Result<(Option<C08RenderRegion>, bool)> {
    let (region, transformed_source_bounds) =
        C08RenderRegion::bounded_transformed_source(source, target, transform)?;
    let preserved = c08_spatial_uniform_preserves_source_origin(spatial, source)
        && region.is_none_or(|region| {
            close_f64(region.unclipped_x, transformed_source_bounds[0])
                && close_f64(region.unclipped_y, transformed_source_bounds[1])
        });
    Ok((region, preserved))
}

/// One allocation-backed, generation-bound C07 handoff. Its lifetime prevents
/// the ready device bundle from transitioning while C08 owns its frame scope.
fn validate_c08_vello_capture_target(handoff: &C08VelloCaptureEncodingHandoff<'_>) -> Result<()> {
    let target_extent = handoff.target_extent();
    if target_extent.width() == 0
        || target_extent.height() == 0
        || handoff.texture().width() != target_extent.width()
        || handoff.texture().height() != target_extent.height()
        || handoff.texture().depth_or_array_layers() != 1
        || handoff.texture().mip_level_count() != 1
        || handoff.texture().sample_count() != 1
        || handoff.texture().dimension() != wgpu::TextureDimension::D2
        || handoff.texture().format() != wgpu::TextureFormat::Rgba8Unorm
        || handoff.texture().usage() != VELLO_CAPTURE_TEXTURE_USAGES
    {
        return Err(preparation_error(
            "the C08 Vello capture target changed its exact RGBA8 storage contract",
        ));
    }
    Ok(())
}

fn c08_vello_capture_scene(handoff: &C08VelloCaptureEncodingHandoff<'_>) -> Result<VelloScene> {
    let initial_transform = handoff.initial_transform();
    match handoff.work() {
        RuntimeVelloCapture::Span(span) => {
            encode_vello_scene_with_initial_transform(&span.commands, initial_transform)
        }
        RuntimeVelloCapture::ClipCoverage(coverage) => {
            let elements = coverage
                .elements
                .iter()
                .map(|element| (element.clip.clone(), element.transform))
                .collect::<Vec<_>>();
            encode_vello_clip_coverage_scene(&elements, initial_transform, handoff.target_extent())
        }
    }
}

impl<'device> PreparedGraph<'device> {
    pub(crate) fn with_vello_engine(mut self, engine: &'device VelloEngineState) -> Self {
        self.vello_engine = Some(engine);
        self
    }
    pub(crate) async fn encode_c08_custom_spine(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: C08ExternalOutputView<'_>,
    ) -> Result<C08PendingGraphEncoding> {
        let expected_capture_count = self.c08_custom_spine_requirements(&output)?;
        let engine = self.vello_engine.ok_or_else(|| {
            preparation_error("the C08 capture scheduler has no ready internal Vello engine")
        })?;
        let resources = self.resources;
        let queue = self.queue;

        let session = Arc::new(());
        let mut scope = ActiveVelloEncodingScope::begin(self.device);
        let mut leases = VelloResourceLeaseAggregate::new();
        self.c08_encoding_state = Some(C08CustomSpineEncodingState::Encoding);
        let result = {
            let mut capture_encoding = C08VelloCaptureEncodingContext {
                engine,
                resources,
                queue,
                scope: &mut scope,
                leases: &mut leases,
            };
            self.encode_c08_custom_spine_once(
                encoder,
                &output,
                expected_capture_count,
                &session,
                &mut capture_encoding,
            )
        };
        let summary = match result {
            Ok(summary) => summary,
            Err(encoding_error) => {
                let _ = leases.abort();
                let scope_result = scope.finish().await;
                self.c08_encoding_state = Some(C08CustomSpineEncodingState::AbortOnly);
                return match scope_result {
                    Ok(()) => Err(encoding_error),
                    Err(scope_error) => Err(scope_error),
                };
            }
        };
        #[cfg(test)]
        if PREPARED_GRAPH_TEST_SUPPORT.with(|support| {
            let mut state = support.get();
            let inject = state.fail_scope_resolution;
            state.fail_scope_resolution = false;
            support.set(state);
            inject
        }) {
            scope.inject_validation_error_for_test();
        }
        let leases = match scope.finish_with_leases(leases).await {
            Ok(leases) => leases,
            Err(failure) => {
                self.c08_encoding_state = Some(C08CustomSpineEncodingState::AbortOnly);
                return Err(failure.into_error_and_aborted_resources().0);
            }
        };
        self.c08_encoding_state = Some(C08CustomSpineEncodingState::Complete);
        self.c08_completed_session = Some(Arc::clone(&session));
        Ok(C08PendingGraphEncoding {
            summary,
            resources: PendingVelloResourceCommit::from_aggregate(leases),
            session,
        })
    }

    fn c08_custom_spine_requirements(&self, output: &C08ExternalOutputView<'_>) -> Result<usize> {
        match self.c08_encoding_state {
            Some(C08CustomSpineEncodingState::Ready) => {}
            Some(
                C08CustomSpineEncodingState::Encoding
                | C08CustomSpineEncodingState::Complete
                | C08CustomSpineEncodingState::AbortOnly,
            ) => {
                return Err(preparation_error(
                    "the C08 custom encoding is one-shot; discard this prepared graph and its encoder",
                ));
            }
            None => {
                return Err(preparation_error(
                    "the C08 custom scheduler requires validated execution facts",
                ));
            }
        }
        let (execution_working_format, execution_output_format, expected_capture_count) =
            if let Some(execution) = self.c08_execution.as_ref() {
                (
                    execution.working_format(),
                    execution.output_format(),
                    execution.captures().len(),
                )
            } else if let Some(execution) = self.c09_execution.as_ref() {
                (
                    execution.working_format,
                    execution.output_format,
                    execution.captures.len(),
                )
            } else if let Some(execution) = self.c10_execution.as_ref() {
                (
                    execution.working_format,
                    execution.output_format,
                    execution.captures.len(),
                )
            } else if let Some(execution) = self.c11_execution.as_ref() {
                (
                    execution.working_format,
                    execution.output_format,
                    execution.captures.len(),
                )
            } else if let Some(execution) = self.c12_execution.as_ref() {
                (
                    execution.working_format,
                    execution.output_format,
                    execution.captures.len(),
                )
            } else {
                return Err(preparation_error(
                    "the C08 custom scheduler requires validated execution facts",
                ));
            };
        if execution_working_format != self.plan.working_format
            || execution_output_format != self.plan.output_format
            || output.format != self.plan.output_format
            || output.extent != self.output_extent()?
        {
            return Err(preparation_error(
                "the C08 external output differs from the exact prepared format or extent",
            ));
        }
        if self.pass_cache_update.is_none() {
            return Err(preparation_error(
                "the C08 custom scheduler requires transaction-provisional pass objects",
            ));
        }
        if expected_capture_count == 0 || self.next_pass != 0 {
            return Err(preparation_error(
                "the C08 custom scheduler requires one unstarted capture spine",
            ));
        }
        Ok(expected_capture_count)
    }

    fn encode_c08_custom_spine_once(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &C08ExternalOutputView<'_>,
        expected_capture_count: usize,
        session: &Arc<()>,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
    ) -> Result<C08CustomSpineEncodingSummary> {
        #[cfg(test)]
        begin_c08_encoding_observations_for_test(expected_capture_count);
        let mut progress =
            C08CustomSpineEncodingProgress::new(self.plan.passes.len(), expected_capture_count);
        while let Some(request) = self
            .plan
            .passes
            .get(self.next_pass)
            .map(C08PreparedPassEncodingRequest::from)
        {
            self.encode_c08_custom_request(
                encoder,
                output,
                session,
                capture_encoding,
                &request,
                &mut progress,
            )?;
        }
        Ok(progress.finish(self))
    }

    fn encode_c08_custom_request(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &C08ExternalOutputView<'_>,
        session: &Arc<()>,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        match &request.kind {
            RuntimePassKind::ClearRoot { initialization, .. } => {
                self.encode_c08_clear_step(encoder, request, *initialization, progress)
            }
            RuntimePassKind::VelloCapture(Some(_)) => {
                self.encode_c08_capture_step(encoder, request, session, capture_encoding, progress)
            }
            RuntimePassKind::CanonicalizeCapture => {
                self.encode_c08_canonicalize_step(encoder, request, progress)
            }
            RuntimePassKind::CopyBackdrop => {
                self.encode_c12_copy_backdrop_step(encoder, request, progress)
            }
            RuntimePassKind::ColorFilter(Some(_)) => {
                self.encode_c10_color_filter_step(encoder, request, progress)
            }
            RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_)) => {
                self.encode_c11_blur_step(encoder, request, progress)
            }
            RuntimePassKind::DropShadowColorize(Some(_)) => {
                self.encode_c11_drop_shadow_colorize_step(encoder, request, progress)
            }
            RuntimePassKind::Composite(Some(composite))
                if matches!(composite.kind, RuntimeCompositeKind::DropShadow) =>
            {
                self.encode_c11_drop_shadow_merge_step(encoder, request, progress)
            }
            RuntimePassKind::Composite(Some(composite))
                if matches!(composite.kind, RuntimeCompositeKind::SpanSourceOver) =>
            {
                self.encode_c08_source_over_step(encoder, request, progress)
            }
            RuntimePassKind::Composite(Some(composite))
                if matches!(composite.kind, RuntimeCompositeKind::Layer { .. }) =>
            {
                self.encode_c09_layer_step(encoder, request, progress)
            }
            RuntimePassKind::Present => {
                self.encode_c08_present_step(encoder, output, request, progress)
            }
            RuntimePassKind::VelloCapture(None)
            | RuntimePassKind::ColorFilter(None)
            | RuntimePassKind::BlurHorizontal(None)
            | RuntimePassKind::BlurVertical(None)
            | RuntimePassKind::DropShadowColorize(None)
            | RuntimePassKind::Composite(_) => Err(preparation_error(
                "a non-C08 pass reached the custom graph spine scheduler",
            )),
        }
    }

    fn encode_c08_clear_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        initialization: RuntimeInitialization,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c08_clear_root(encoder, request)?;
        if initialization == RuntimeInitialization::SurfaceBaseColor {
            progress.root_clear_count = progress.root_clear_count.saturating_add(1);
            progress.clears_full_root &= facts.full_target;
        }
        self.complete_c08_custom_pass(request.id)?;
        progress.record_custom_completion(C08ScheduledEncodingKind::ClearRoot);
        Ok(())
    }

    fn encode_c08_capture_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        session: &Arc<()>,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        #[cfg(test)]
        if PREPARED_GRAPH_TEST_SUPPORT.with(|support| {
            let mut state = support.get();
            let inject = state.fail_capture_encoding_after == Some(progress.capture_count);
            if inject {
                state.fail_capture_encoding_after = None;
                support.set(state);
            }
            inject
        }) {
            return Err(preparation_error(
                "injected C08 Vello capture encoding failure",
            ));
        }
        let handoff = self.c08_vello_capture_handoff(request, session)?;
        let target = handoff.target();
        progress.bounded_capture_handoffs &= c08_capture_handoff_is_bounded(&handoff);
        let encoded = Self::encode_c08_vello_capture(handoff, encoder, capture_encoding)?;
        self.complete_c08_capture(request.id, target, session, encoded.receipt)?;
        #[cfg(test)]
        PREPARED_GRAPH_TEST_SUPPORT.with(|support| {
            let mut state = support.get();
            state.acquired_capture_lease_count =
                state.acquired_capture_lease_count.saturating_add(1);
            support.set(state);
        });
        progress.record_capture_completion();
        Ok(())
    }

    fn encode_c08_canonicalize_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c08_canonicalize(encoder, request)?;
        progress.exact_spatial &= facts.exact_spatial_uniform;
        self.complete_c08_custom_pass(request.id)?;
        progress.record_custom_completion(C08ScheduledEncodingKind::CanonicalizeCapture);
        Ok(())
    }

    fn encode_c10_color_filter_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c10_color_filter(encoder, request)?;
        progress.color_filter_count = progress.color_filter_count.saturating_add(1);
        progress.color_filters_preserve_authored_order &= facts.exact_operation_bytes;
        progress.color_filter_bindings_are_exact &= facts.exact_source_spatial_and_operations;
        progress.color_filter_sources_and_results_are_distinct &=
            facts.source_and_result_are_distinct;
        progress.color_filter_regions_are_validated &= facts.validated_viewport_and_scissor;
        progress.color_filter_signed_texel_mapping_is_exact &= facts.preserved_signed_texel_mapping;
        self.complete_c08_custom_pass(request.id)?;
        progress.color_filter_operation_buffers_released &= self
            .color_filter_operation_bindings
            .get(&request.id)
            .is_some_and(|binding| binding.buffer.is_none());
        progress.record_custom_completion(C08ScheduledEncodingKind::ColorFilter);
        Ok(())
    }

    fn encode_c12_copy_backdrop_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c12_copy_backdrop(encoder, request)?;
        progress.copy_backdrop_count = progress.copy_backdrop_count.saturating_add(1);
        progress.copy_backdrop_bindings_are_exact &= facts.exact_prepared_bindings;
        progress.copy_backdrop_allocations_are_distinct &= facts.source_and_result_are_distinct;
        progress.copy_backdrop_regions_are_validated &= facts.validated_viewport_and_scissor;
        progress.copy_backdrop_signed_mapping_is_exact &= facts.preserved_signed_mapping;
        #[cfg(test)]
        record_c08_graph_encoder_for_test(encoder);
        self.complete_c08_custom_pass(request.id)?;
        progress.c11_textures_released &= request.releases.iter().all(|resource| {
            self.resource_bindings
                .get(resource)
                .is_some_and(|binding| binding.lease.is_none())
        });
        progress.record_custom_completion(C08ScheduledEncodingKind::CopyBackdrop);
        Ok(())
    }

    fn encode_c11_blur_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c11_blur(encoder, request)?;
        let kind = match &request.kind {
            RuntimePassKind::BlurHorizontal(Some(blur)) => match blur.input {
                RuntimeBlurInput::Rgba => C08ScheduledEncodingKind::BlurHorizontalRgba,
                RuntimeBlurInput::SourceAlpha => {
                    C08ScheduledEncodingKind::BlurHorizontalSourceAlpha
                }
            },
            RuntimePassKind::BlurVertical(Some(blur)) => match blur.input {
                RuntimeBlurInput::Rgba => C08ScheduledEncodingKind::BlurVerticalRgba,
                RuntimeBlurInput::SourceAlpha => C08ScheduledEncodingKind::BlurVerticalSourceAlpha,
            },
            _ => {
                return Err(preparation_error(
                    "the C11 blur scheduler lost its pass kind",
                ));
            }
        };
        #[cfg(test)]
        record_c08_graph_encoder_for_test(encoder);
        self.complete_c08_custom_pass(request.id)?;
        progress.blur_pass_count = progress.blur_pass_count.saturating_add(1);
        progress.c11_bindings_are_exact &= facts.exact_prepared_bindings;
        progress.c11_regions_are_exact &= facts.validated_region && facts.preserved_signed_mapping;
        progress.blur_allocations_are_distinct &= facts.distinct_source_and_result;
        progress.c11_kernels_released &= request.kernel_releases.iter().all(|kernel| {
            self.kernel_bindings
                .get(kernel)
                .is_some_and(|binding| binding.lease.is_none())
        });
        progress.c11_textures_released &= request.releases.iter().all(|resource| {
            self.resource_bindings
                .get(resource)
                .is_some_and(|binding| binding.lease.is_none())
        });
        progress.record_custom_completion(kind);
        Ok(())
    }

    fn encode_c11_drop_shadow_colorize_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c11_drop_shadow_colorize(encoder, request)?;
        #[cfg(test)]
        record_c08_graph_encoder_for_test(encoder);
        self.complete_c08_custom_pass(request.id)?;
        progress.drop_shadow_colorize_count = progress.drop_shadow_colorize_count.saturating_add(1);
        progress.c11_bindings_are_exact &= facts.exact_prepared_bindings;
        progress.c11_regions_are_exact &= facts.validated_region && facts.preserved_signed_mapping;
        progress.blur_allocations_are_distinct &= facts.distinct_source_and_result;
        progress.c11_textures_released &= request.releases.iter().all(|resource| {
            self.resource_bindings
                .get(resource)
                .is_some_and(|binding| binding.lease.is_none())
        });
        progress.record_custom_completion(C08ScheduledEncodingKind::DropShadowColorize);
        Ok(())
    }

    fn encode_c11_drop_shadow_merge_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c11_drop_shadow_merge(encoder, request)?;
        let source = exact_c08_read(request, RuntimeReadRole::CompositeSource)?.resource();
        let source_is_live = self
            .resource_bindings
            .get(&source)
            .is_some_and(|binding| binding.lease.is_some());
        let source_read_twice = [&self.c11_execution, &self.c12_execution]
            .into_iter()
            .flatten()
            .any(|execution| {
                execution.drop_shadows.iter().any(|shadow| {
                    shadow.merge == request.id
                        && shadow.source == source
                        && self
                            .resource_request(source)
                            .is_ok_and(|resource| resource.expected_reads == 2)
                })
            });
        #[cfg(test)]
        record_c08_graph_encoder_for_test(encoder);
        self.complete_c08_custom_pass(request.id)?;
        let source_is_released = self
            .resource_bindings
            .get(&source)
            .is_some_and(|binding| binding.lease.is_none());
        progress.drop_shadow_merge_count = progress.drop_shadow_merge_count.saturating_add(1);
        progress.source_over_count = progress.source_over_count.saturating_add(1);
        progress.c11_bindings_are_exact &= facts.exact_prepared_bindings;
        progress.c11_regions_are_exact &= facts.validated_region && facts.preserved_signed_mapping;
        progress.parent_and_result_are_distinct &= facts.distinct_source_shadow_and_result;
        progress.full_copy_before_bounded_render &= facts.distinct_source_shadow_and_result;
        progress.samples_source_with_fixed_blend &= facts.fixed_source_over_blend;
        progress.preserves_signed_origin &= facts.preserved_signed_mapping;
        progress.shadow_source_read_twice &=
            facts.reads_original_source_and_shadow && source_read_twice;
        progress.shadow_source_released_after_merge &=
            source_is_live && source_is_released && request.releases.contains(&source);
        progress.c11_textures_released &= request.releases.iter().all(|resource| {
            self.resource_bindings
                .get(resource)
                .is_some_and(|binding| binding.lease.is_none())
        });
        progress.record_custom_completion(C08ScheduledEncodingKind::DropShadowMerge);
        Ok(())
    }

    fn encode_c08_source_over_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c08_span_source_over(encoder, request)?;
        progress.source_over_count = progress.source_over_count.saturating_add(1);
        progress.exact_spatial &= facts.exact_spatial_uniform;
        progress.parent_and_result_are_distinct &= facts.parent_and_result_distinct;
        progress.full_copy_before_bounded_render &= facts.copied_full_parent_before_render;
        progress.samples_source_with_fixed_blend &=
            facts.sampled_only_source && facts.fixed_source_over_blend;
        progress.preserves_signed_origin &= facts.preserved_signed_source_origin;
        self.complete_c08_custom_pass(request.id)?;
        progress.record_custom_completion(C08ScheduledEncodingKind::SpanSourceOver);
        Ok(())
    }

    fn encode_c09_layer_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c09_layer_composite(encoder, request)?;
        progress.layer_composite_count = progress.layer_composite_count.saturating_add(1);
        progress.normal_composite_count = progress
            .normal_composite_count
            .saturating_add(usize::from(facts.normal_path));
        progress.destination_composite_count = progress
            .destination_composite_count
            .saturating_add(usize::from(facts.destination_path));
        progress.parent_and_result_are_distinct &= facts.avoids_read_write_alias;
        progress.full_copy_before_bounded_render &= facts.copied_full_parent;
        if facts.normal_path {
            progress.samples_source_with_fixed_blend &=
                facts.fixed_premultiplied_blend && facts.omits_parent_sample;
            progress.normal_fixed_blend &= facts.fixed_premultiplied_blend;
            progress.normal_omits_parent_sample &= facts.omits_parent_sample;
        }
        if facts.destination_path {
            progress.destination_copies_full_parent &= facts.copied_full_parent;
            progress.destination_avoids_alias &= facts.avoids_read_write_alias;
        }
        progress.layer_bindings_are_exact &= facts.exact_resources_and_parameters;
        progress.layer_signed_mapping_is_exact &= facts.preserved_signed_mapping;
        progress.preserves_signed_origin &= facts.preserved_signed_mapping;
        #[cfg(test)]
        record_c08_graph_encoder_for_test(encoder);
        self.complete_c08_custom_pass(request.id)?;
        progress.record_custom_completion(C08ScheduledEncodingKind::LayerComposite);
        Ok(())
    }

    fn encode_c08_present_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &C08ExternalOutputView<'_>,
        request: &C08PreparedPassEncodingRequest,
        progress: &mut C08CustomSpineEncodingProgress,
    ) -> Result<()> {
        let facts = self.encode_c08_present(encoder, request, output)?;
        progress.exact_spatial &= facts.exact_spatial_uniform;
        progress.exact_external_output |= facts.external_output_exact;
        self.complete_c08_custom_pass(request.id)?;
        progress.record_custom_completion(C08ScheduledEncodingKind::Present);
        Ok(())
    }

    fn encode_c08_vello_capture(
        handoff: C08VelloCaptureEncodingHandoff<'_>,
        encoder: &mut wgpu::CommandEncoder,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
    ) -> Result<C08EncodedCaptureResult> {
        let target_extent = handoff.target_extent();
        let antialiasing = handoff.antialiasing();
        validate_c08_vello_capture_target(&handoff)?;
        let scene = c08_vello_capture_scene(&handoff)?;
        let prepared = scene.prepare_raster(vello_capture_raster_parameters(
            target_extent,
            antialiasing,
        )?)?;
        let encoded = {
            let mut encoding = TransactionEncodingState::new_reusable_graph_capture(
                capture_encoding.scope,
                capture_encoding.queue,
                encoder,
                handoff.view(),
                TransactionTargetIntent::new(
                    target_extent,
                    wgpu::TextureFormat::Rgba8Unorm,
                    VELLO_CAPTURE_TEXTURE_USAGES,
                ),
            );
            prepared.encode_capture_into(
                capture_encoding.engine,
                capture_encoding.resources,
                &mut encoding,
            )
        };
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(failure) => return Err(failure.into_error_and_aborted_resources().0),
        };
        let (lease, proof) = encoded.into_resources_and_proof();
        #[cfg(test)]
        record_c08_capture_observation_for_test(
            handoff.work(),
            handoff.initial_transform(),
            &scene,
            (handoff.texture(), handoff.view()),
            &proof,
            encoder,
            capture_encoding.scope,
        );
        let receipt = match handoff.complete_after_encoded_capture(proof) {
            Ok(receipt) => receipt,
            Err(error) => {
                let _ = lease.abort();
                return Err(error);
            }
        };
        capture_encoding.leases.push(lease);
        Ok(C08EncodedCaptureResult { receipt })
    }

    fn validate_texture_binding(
        &self,
        binding: &PreparedTextureBinding<'_>,
        resource: RuntimeResourceId,
    ) -> Result<RuntimeSpatialDescriptor> {
        let request = self.resource_request(resource)?;
        let texture = binding.texture();
        let expected_format = match request.format {
            RuntimeResourceFormat::VelloCaptureRgba8Unorm
            | RuntimeResourceFormat::ClipCoverageRgba8Unorm
            | RuntimeResourceFormat::ResolvedMaskRgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            RuntimeResourceFormat::Working(format) => format.texture_format(),
        };
        if binding.runtime_resource() != resource
            || texture.width() != request.spatial.device_extent.width()
            || texture.height() != request.spatial.device_extent.height()
            || texture.depth_or_array_layers() != 1
            || texture.mip_level_count() != 1
            || texture.sample_count() != 1
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.format() != expected_format
        {
            return Err(preparation_error(
                "the C08 prepared texture differs from its exact runtime binding",
            ));
        }
        Ok(request.spatial)
    }

    fn c08_pass_objects<'prepared>(
        &'prepared self,
        keys: &RuntimePassCacheKeys,
    ) -> Result<ProvisionalC08PassObjects<'prepared>> {
        self.pass_cache_update
            .as_ref()
            .ok_or_else(|| preparation_error("C08 provisional pass objects are unavailable"))?
            .encoding_objects(
                self.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
    }

    fn c09_composite_pass_objects<'prepared>(
        &'prepared self,
        keys: &RuntimePassCacheKeys,
    ) -> Result<ProvisionalCompositePassObjects<'prepared>> {
        self.pass_cache_update
            .as_ref()
            .ok_or_else(|| preparation_error("C09 provisional pass objects are unavailable"))?
            .composite_encoding_objects(
                self.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
    }

    fn c10_color_filter_pass_objects<'prepared>(
        &'prepared self,
        keys: &RuntimePassCacheKeys,
    ) -> Result<ProvisionalColorFilterPassObjects<'prepared>> {
        self.pass_cache_update
            .as_ref()
            .ok_or_else(|| preparation_error("C10 provisional pass objects are unavailable"))?
            .color_filter_encoding_objects(
                self.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
    }

    fn c12_copy_backdrop_pass_objects(
        &self,
        pass: RuntimePassId,
    ) -> Result<(
        &wgpu::Sampler,
        &wgpu::BindGroupLayout,
        &wgpu::RenderPipeline,
    )> {
        match self.c11_pass_objects.get(&pass) {
            Some(PreparedC11PassObjects::CopyBackdrop {
                parent_sampler,
                layout,
                pipeline,
            }) => Ok((parent_sampler, layout, pipeline)),
            Some(
                PreparedC11PassObjects::Blur { .. }
                | PreparedC11PassObjects::DropShadowColorize { .. },
            )
            | None => Err(preparation_error(
                "the C12 backdrop-copy pass lost its exact prepared object handles",
            )),
        }
    }

    fn c11_blur_pass_objects(
        &self,
        pass: RuntimePassId,
    ) -> Result<(
        &wgpu::Sampler,
        &wgpu::BindGroupLayout,
        &wgpu::RenderPipeline,
    )> {
        match self.c11_pass_objects.get(&pass) {
            Some(PreparedC11PassObjects::Blur {
                source_sampler,
                layout,
                pipeline,
            }) => Ok((source_sampler, layout, pipeline)),
            Some(
                PreparedC11PassObjects::CopyBackdrop { .. }
                | PreparedC11PassObjects::DropShadowColorize { .. },
            )
            | None => Err(preparation_error(
                "the C11 blur pass lost its exact prepared object handles",
            )),
        }
    }

    fn c11_drop_shadow_colorize_pass_objects(
        &self,
        pass: RuntimePassId,
    ) -> Result<(
        &wgpu::Sampler,
        &wgpu::BindGroupLayout,
        &wgpu::RenderPipeline,
    )> {
        match self.c11_pass_objects.get(&pass) {
            Some(PreparedC11PassObjects::DropShadowColorize {
                source_sampler,
                layout,
                pipeline,
            }) => Ok((source_sampler, layout, pipeline)),
            Some(
                PreparedC11PassObjects::CopyBackdrop { .. } | PreparedC11PassObjects::Blur { .. },
            )
            | None => Err(preparation_error(
                "the C11 drop-shadow pass lost its exact prepared object handles",
            )),
        }
    }

    fn create_c08_spatial_uniform_buffer(&self, bytes: &PassSpatialUniformBytes) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C08 pass spatial uniform"),
            size: bytes.as_bytes().len() as u64,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes.as_bytes());
        buffer
    }

    fn create_c09_composite_parameter_buffer(
        &self,
        bytes: &CompositeParameterBytes,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C09 composite parameter uniform"),
            size: bytes.as_bytes().len() as u64,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes.as_bytes());
        buffer
    }

    fn create_c11_drop_shadow_parameter_buffer(
        &self,
        bytes: &DropShadowParameterBytes,
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C11 drop-shadow parameter uniform"),
            size: bytes.as_bytes().len() as u64,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes.as_bytes());
        buffer
    }

    fn create_c12_blur_edge_parameter_buffer(&self, bytes: &[u8; 16]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C12 semantic backdrop edge uniform"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    fn c08_vello_capture_handoff<'prepared>(
        &'prepared self,
        request: &C08PreparedPassEncodingRequest,
        session: &Arc<()>,
    ) -> Result<C08VelloCaptureEncodingHandoff<'prepared>> {
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 Vello capture has no exact prepared target",
            ));
        };
        let capture: &ExecutableVelloCaptureFacts =
            self.c08_execution
                .as_ref()
                .and_then(|execution| {
                    execution
                        .captures()
                        .iter()
                        .find(|capture| capture.pass() == request.id && capture.target() == target)
                })
                .or_else(|| {
                    self.c09_execution.as_ref().and_then(|execution| {
                        execution.captures.iter().find(|capture| {
                            capture.pass() == request.id && capture.target() == target
                        })
                    })
                })
                .or_else(|| {
                    self.c10_execution.as_ref().and_then(|execution| {
                        execution.captures.iter().find(|capture| {
                            capture.pass() == request.id && capture.target() == target
                        })
                    })
                })
                .or_else(|| {
                    self.c11_execution.as_ref().and_then(|execution| {
                        execution.captures.iter().find(|capture| {
                            capture.pass() == request.id && capture.target() == target
                        })
                    })
                })
                .or_else(|| {
                    self.c12_execution.as_ref().and_then(|execution| {
                        execution.captures.iter().find(|capture| {
                            capture.pass() == request.id && capture.target() == target
                        })
                    })
                })
                .ok_or_else(|| preparation_error("the bounded C08 capture handoff is missing"))?;
        let binding = self.texture_binding_for_pass(request.id, target)?;
        let spatial = self.validate_texture_binding(&binding, target)?;
        if spatial.device_extent != capture.target_extent()
            || spatial.texel_origin != capture.texel_origin()
            || spatial.raster_scale != capture.raster_scale()
        {
            return Err(preparation_error(
                "the bounded C08 capture target changed after preparation",
            ));
        }
        Ok(C08VelloCaptureEncodingHandoff {
            pass: request.id,
            target,
            work: capture.work(),
            initial_transform: capture.initial_transform(),
            antialiasing: capture.antialiasing(),
            target_extent: capture.target_extent(),
            raster_scale: capture.raster_scale(),
            texture: binding.texture(),
            view: binding.view(),
            session: Arc::clone(session),
        })
    }

    fn encode_c08_clear_root(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let RuntimePassKind::ClearRoot {
            initialization,
            color,
        } = &request.kind
        else {
            return Err(preparation_error(
                "the C08 root clear changed its initialization contract",
            ));
        };
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error("the C08 root clear has no target"));
        };
        let target_request = self.resource_request(target)?;
        let exact_initialization = match initialization {
            RuntimeInitialization::SurfaceBaseColor => {
                target == self.plan.root_working_image
                    && target_request.role == RuntimeResourceRole::RootWorkingImage
            }
            RuntimeInitialization::Transparent => {
                *color == Color::TRANSPARENT
                    && target_request.role == RuntimeResourceRole::IsolationWorkingImage
            }
        };
        if !exact_initialization
            || !request.reads.is_empty()
            || request.spatial_uniform.is_some()
            || request.color_filter_operations.is_some()
            || request.composite_parameters.is_some()
            || request.cache_keys.is_some()
        {
            return Err(preparation_error(
                "the C08 root clear has non-root or sampled bindings",
            ));
        }
        let binding = self.texture_binding_for_pass(request.id, target)?;
        let spatial = self.validate_texture_binding(&binding, target)?;
        let alpha = f64::from(color.a());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Surgeist C08 full root clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: binding.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(color.r()) * alpha,
                            g: f64::from(color.g()) * alpha,
                            b: f64::from(color.b()) * alpha,
                            a: alpha,
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
        Ok(C08PassEncodingFacts {
            full_target: spatial.device_extent == target_request.spatial.device_extent,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_canonicalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let source = exact_c08_read(request, RuntimeReadRole::CaptureSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 canonicalization pass has no prepared result",
            ));
        };
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource)?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        if source_spatial != target_spatial
            || self.resource_request(source.resource)?.format
                != RuntimeResourceFormat::VelloCaptureRgba8Unorm
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
        {
            return Err(preparation_error(
                "C08 canonicalization changed its exact capture-to-working binding",
            ));
        }
        let region = C08RenderRegion::full(target_spatial.device_extent)?;
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: target_binding.view(),
                extent: target_spatial.device_extent,
                region: Some(region),
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                label: "Surgeist C08 canonicalize capture",
            },
        )?;
        Ok(C08PassEncodingFacts {
            full_target: true,
            exact_spatial_uniform: true,
            fixed_source_over_blend: fixed_blend,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_span_source_over(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let parent = exact_c08_read(request, RuntimeReadRole::CompositeParent)?;
        let source = exact_c08_read(request, RuntimeReadRole::CompositeSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 source-over pass has no prepared result",
            ));
        };
        let parent_binding = self.texture_binding_for_pass(request.id, parent.resource)?;
        let parent_spatial = self.validate_texture_binding(&parent_binding, parent.resource)?;
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource)?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let parent_and_result_distinct = parent.resource != target
            && parent_binding.allocation_resource() != target_binding.allocation_resource();
        if !parent_and_result_distinct
            || parent_spatial != target_spatial
            || parent_binding.texture().format() != target_binding.texture().format()
            || self.resource_request(parent.resource)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(source.resource)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
        {
            return Err(preparation_error(
                "C08 source-over parent, source, and distinct result bindings are inconsistent",
            ));
        }

        let copy_extent = c09_composite_copy_extent(parent_spatial);
        copy_c09_composite_parent(encoder, &parent_binding, &target_binding, copy_extent);

        let region = C08RenderRegion::bounded_source(source_spatial, target_spatial)?;
        let preserved_signed_source_origin =
            request.spatial_uniform.as_ref().is_some_and(|bytes| {
                c08_spatial_uniform_preserves_source_origin(bytes, source_spatial)
                    && region.is_none_or(|region| {
                        let expected_x = (source_spatial.texel_origin.x()
                            - target_spatial.texel_origin.x())
                            * target_spatial.raster_scale;
                        let expected_y = (source_spatial.texel_origin.y()
                            - target_spatial.texel_origin.y())
                            * target_spatial.raster_scale;
                        close_f64(region.unclipped_x, expected_x)
                            && close_f64(region.unclipped_y, expected_y)
                    })
            });
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: target_binding.view(),
                extent: target_spatial.device_extent,
                region,
                load: wgpu::LoadOp::Load,
                label: "Surgeist C08 bounded span source-over",
            },
        )?;
        Ok(C08PassEncodingFacts {
            exact_spatial_uniform: true,
            parent_and_result_distinct,
            copied_full_parent_before_render: copy_extent.width == target_binding.texture().width()
                && copy_extent.height == target_binding.texture().height(),
            sampled_only_source: request.reads.len() == 2,
            fixed_source_over_blend: fixed_blend,
            preserved_signed_source_origin,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c12_copy_backdrop(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C12CopyBackdropEncodingFacts> {
        if !matches!(request.kind, RuntimePassKind::CopyBackdrop) {
            return Err(preparation_error(
                "the C12 backdrop-copy payload changed after preparation",
            ));
        }
        let parent = exact_c08_read(request, RuntimeReadRole::CompletedParent)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C12 backdrop-copy pass has no prepared result",
            ));
        };
        let parent_binding = self.texture_binding_for_pass(request.id, parent.resource())?;
        let parent_spatial = self.validate_texture_binding(&parent_binding, parent.resource())?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let distinct = parent.resource() != target
            && parent_binding.allocation_resource() != target_binding.allocation_resource();
        let spatial = request
            .spatial_uniform
            .as_ref()
            .ok_or_else(|| preparation_error("the C12 backdrop-copy spatial bytes are missing"))?;
        let expected_spatial = PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
            parent_spatial,
            target_spatial,
        )?;
        let keys = request
            .cache_keys
            .as_ref()
            .ok_or_else(|| preparation_error("the C12 backdrop-copy cache keys are missing"))?;
        let (sampler, layout, pipeline) = self.c12_copy_backdrop_pass_objects(request.id)?;
        if request.reads.len() != 1
            || !distinct
            || spatial != &expected_spatial
            || keys.samplers() != [parent.sampler_key()]
            || parent.sampling_filter() != RuntimeSamplingFilter::Nearest
            || parent.sampling_edge() != RuntimeSamplingEdge::TransparentBlack
            || self.resource_request(parent.resource())?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(target)?.role != RuntimeResourceRole::BackdropCopy
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || !parent_binding
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !target_binding
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(preparation_error(
                "the C12 backdrop-copy bindings differ from the checked pass",
            ));
        }
        let region = C08RenderRegion::full(target_spatial.device_extent)?;
        let uniform = self.create_c08_spatial_uniform_buffer(spatial);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C12 exact backdrop-copy bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(parent_binding.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        encode_c11_full_target_pass(
            encoder,
            target_binding.view(),
            pipeline,
            &bind_group,
            region,
            "Surgeist C12 bounded completed-parent copy",
        );
        Ok(C12CopyBackdropEncodingFacts {
            exact_prepared_bindings: true,
            source_and_result_are_distinct: distinct,
            validated_viewport_and_scissor: c11_full_region_is_exact(
                region,
                target_spatial.device_extent,
            ),
            preserved_signed_mapping: c08_spatial_uniform_preserves_source_origin(
                spatial,
                parent_spatial,
            ),
        })
    }

    fn prepare_c10_color_filter_encoding<'prepared>(
        &'prepared self,
        request: &'prepared C08PreparedPassEncodingRequest,
    ) -> Result<PreparedC10ColorFilterEncoding<'prepared>> {
        let RuntimePassKind::ColorFilter(Some(filter)) = &request.kind else {
            return Err(preparation_error(
                "the C10 color pass changed its checked semantic payload",
            ));
        };
        let source = exact_c08_read(request, RuntimeReadRole::FilterSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C10 color pass has no prepared working result",
            ));
        };
        if request.reads.len() != 1 || filter.operations().is_empty() {
            return Err(preparation_error(
                "the C10 color pass must have one source and a nonempty ordered program",
            ));
        }
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource())?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let source_and_result_are_distinct = source.resource() != target
            && source_binding.allocation_resource() != target_binding.allocation_resource();
        if !source_and_result_are_distinct
            || source_spatial != target_spatial
            || filter.spatial.source != source_spatial
            || filter.spatial.result != target_spatial
            || filter.edge != RuntimeSamplingEdge::ClampToExtent
            || source.sampling_filter() != RuntimeSamplingFilter::Nearest
            || source.sampling_edge() != RuntimeSamplingEdge::ClampToExtent
            || self.resource_request(source.resource())?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || !source_binding
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || !target_binding
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(preparation_error(
                "the C10 source and distinct working result bindings are inconsistent",
            ));
        }
        let keys = request
            .cache_keys
            .as_ref()
            .ok_or_else(|| preparation_error("the C10 color pass has no provisional cache keys"))?;
        if keys.samplers() != [source.sampler_key()] {
            return Err(preparation_error(
                "the C10 color pass changed its nearest ClampToExtent sampler",
            ));
        }
        let spatial = request
            .spatial_uniform
            .as_ref()
            .ok_or_else(|| preparation_error("the C10 color pass has no prepared spatial bytes"))?;
        let expected_spatial = PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
            source_spatial,
            target_spatial,
        )?;
        if spatial != &expected_spatial {
            return Err(preparation_error(
                "the C10 color spatial bytes changed after immutable preparation",
            ));
        }
        let (operation_buffer, exact_operation_bytes) =
            self.validate_c10_operation_buffer(request, filter)?;
        let objects = self.c10_color_filter_pass_objects(keys)?;
        objects.require_encoding_ready()?;
        let (region, validated_viewport_and_scissor, preserved_signed_texel_mapping) =
            validate_c10_render_region(spatial, source_spatial, target_spatial)?;
        Ok(PreparedC10ColorFilterEncoding {
            source_binding,
            target_binding,
            objects,
            spatial,
            operation_buffer,
            region,
            facts: C10ColorFilterEncodingFacts {
                exact_operation_bytes,
                exact_source_spatial_and_operations: spatial == &expected_spatial,
                source_and_result_are_distinct,
                validated_viewport_and_scissor,
                preserved_signed_texel_mapping,
            },
        })
    }

    fn validate_c10_operation_buffer<'prepared>(
        &'prepared self,
        request: &'prepared C08PreparedPassEncodingRequest,
        filter: &RuntimeColorFilter,
    ) -> Result<(&'prepared wgpu::Buffer, bool)> {
        let operation_bytes = request.color_filter_operations.as_ref().ok_or_else(|| {
            preparation_error("the C10 color pass has no checked operation bytes")
        })?;
        let expected_operation_bytes =
            ColorFilterOperationBytes::try_from_runtime_operations_with_limits(
                filter.operations(),
                ColorFilterOperationBufferLimits::from_device_limits(&self.device.limits()),
            )?;
        let operation_binding = self
            .color_filter_operation_bindings
            .get(&request.id)
            .ok_or_else(|| preparation_error("the C10 operation buffer binding is missing"))?;
        let operation_buffer = operation_binding.buffer.as_ref().ok_or_else(|| {
            preparation_error("the C10 operation buffer is stale or already released")
        })?;
        let exact = operation_bytes == &expected_operation_bytes
            && &operation_binding.bytes == operation_bytes
            && operation_buffer.size()
                == u64::try_from(operation_bytes.as_bytes().len()).map_err(|_| {
                    preparation_error("the C10 operation buffer size does not fit u64")
                })?
            && operation_buffer.usage()
                == wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);
        if !exact {
            return Err(preparation_error(
                "the C10 operation buffer differs from its exact checked bytes",
            ));
        }
        Ok((operation_buffer, exact))
    }

    fn encode_c10_color_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C10ColorFilterEncodingFacts> {
        let prepared = self.prepare_c10_color_filter_encoding(request)?;
        #[cfg(test)]
        inject_color_filter_shader_failure_for_test()?;
        let spatial_buffer = self.create_c08_spatial_uniform_buffer(prepared.spatial);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C10 exact color-filter bindings"),
            layout: prepared.objects.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(prepared.source_binding.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(prepared.objects.source_sampler()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prepared.operation_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Surgeist C10 bounded ordered color filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: prepared.target_binding.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(prepared.objects.render_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_viewport(
            prepared.region.viewport_x,
            prepared.region.viewport_y,
            prepared.region.viewport_width,
            prepared.region.viewport_height,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            prepared.region.scissor_x,
            prepared.region.scissor_y,
            prepared.region.scissor_width,
            prepared.region.scissor_height,
        );
        pass.draw(0..3, 0..1);

        Ok(prepared.facts)
    }

    fn encode_c11_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C11BlurEncodingFacts> {
        let blur = match &request.kind {
            RuntimePassKind::BlurHorizontal(Some(blur))
            | RuntimePassKind::BlurVertical(Some(blur)) => blur,
            _ => return Err(preparation_error("the C11 blur payload is missing")),
        };
        let source = exact_c08_read(request, RuntimeReadRole::FilterSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error("the C11 blur result is missing"));
        };
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource())?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let distinct = source.resource() != target
            && source_binding.allocation_resource() != target_binding.allocation_resource();
        let spatial = request
            .spatial_uniform
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 blur spatial bytes are missing"))?;
        let expected_spatial = PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
            source_spatial,
            target_spatial,
        )?;
        let kernel = self
            .gaussian_kernel_binding_for_pass(request.id)?
            .ok_or_else(|| preparation_error("the C11 blur kernel binding is missing"))?;
        let keys = request
            .cache_keys
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 blur cache keys are missing"))?;
        let (sampler, layout, pipeline) = self.c11_blur_pass_objects(request.id)?;
        let edge_bytes =
            c12_blur_edge_uniform_bytes(blur, source, request.blur_edge_parameters.as_ref())?;
        if request.reads.len() != 1
            || !distinct
            || blur.spatial.source != source_spatial
            || blur.spatial.result != target_spatial
            || blur.kernel != kernel.key()
            || spatial != &expected_spatial
            || keys.samplers() != [source.sampler_key()]
            || source.sampling_filter() != RuntimeSamplingFilter::Linear
            || self.resource_request(source.resource())?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
        {
            return Err(preparation_error(
                "the C11 blur bindings differ from the checked pass",
            ));
        }
        let region = C08RenderRegion::full(target_spatial.device_extent)?;
        let spatial_buffer = self.create_c08_spatial_uniform_buffer(spatial);
        let edge_buffer = edge_bytes
            .as_ref()
            .map(|bytes| self.create_c12_blur_edge_parameter_buffer(bytes));
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_binding.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: spatial_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: kernel.buffer().as_entire_binding(),
            },
        ];
        if let Some(edge_buffer) = &edge_buffer {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: edge_buffer.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C11 exact blur bindings"),
            layout,
            entries: &entries,
        });
        encode_c11_full_target_pass(
            encoder,
            target_binding.view(),
            pipeline,
            &bind_group,
            region,
            "Surgeist C11 Gaussian blur",
        );
        Ok(C11BlurEncodingFacts {
            exact_prepared_bindings: true,
            distinct_source_and_result: distinct,
            validated_region: c11_full_region_is_exact(region, target_spatial.device_extent),
            preserved_signed_mapping: c08_spatial_uniform_preserves_source_origin(
                spatial,
                source_spatial,
            ),
        })
    }

    fn encode_c11_drop_shadow_colorize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C11DropShadowColorizeEncodingFacts> {
        let RuntimePassKind::DropShadowColorize(Some(shadow)) = &request.kind else {
            return Err(preparation_error("the C11 drop-shadow payload is missing"));
        };
        let source = exact_c08_read(request, RuntimeReadRole::BlurredSourceAlpha)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error("the C11 drop-shadow result is missing"));
        };
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource())?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let distinct = source.resource() != target
            && source_binding.allocation_resource() != target_binding.allocation_resource();
        let spatial = request
            .spatial_uniform
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 drop-shadow spatial bytes are missing"))?;
        let expected_spatial = PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
            source_spatial,
            target_spatial,
        )?;
        let parameters = request
            .drop_shadow_parameters
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 drop-shadow parameter bytes are missing"))?;
        let expected_parameters = DropShadowParameterBytes::try_new(shadow.offset, shadow.color)?;
        let keys = request
            .cache_keys
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 drop-shadow cache keys are missing"))?;
        let (sampler, layout, pipeline) = self.c11_drop_shadow_colorize_pass_objects(request.id)?;
        if request.reads.len() != 1
            || !distinct
            || shadow.spatial.source != source_spatial
            || shadow.spatial.result != target_spatial
            || spatial != &expected_spatial
            || parameters != &expected_parameters
            || keys.samplers() != [source.sampler_key()]
            || source.sampling_filter() != RuntimeSamplingFilter::Linear
            || source.sampling_edge() != RuntimeSamplingEdge::TransparentBlack
        {
            return Err(preparation_error(
                "the C11 drop-shadow bindings differ from the checked pass",
            ));
        }
        let region = C08RenderRegion::full(target_spatial.device_extent)?;
        let spatial_buffer = self.create_c08_spatial_uniform_buffer(spatial);
        let parameter_buffer = self.create_c11_drop_shadow_parameter_buffer(parameters);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C11 exact drop-shadow colorize bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_binding.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: parameter_buffer.as_entire_binding(),
                },
            ],
        });
        encode_c11_full_target_pass(
            encoder,
            target_binding.view(),
            pipeline,
            &bind_group,
            region,
            "Surgeist C11 drop-shadow colorize",
        );
        Ok(C11DropShadowColorizeEncodingFacts {
            exact_prepared_bindings: true,
            distinct_source_and_result: distinct,
            validated_region: c11_full_region_is_exact(region, target_spatial.device_extent),
            preserved_signed_mapping: c08_spatial_uniform_preserves_source_origin(
                spatial,
                source_spatial,
            ),
        })
    }

    fn encode_c11_drop_shadow_merge(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C11DropShadowMergeEncodingFacts> {
        let RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::DropShadow,
            source_captured_before_outer_semantics: true,
        })) = &request.kind
        else {
            return Err(preparation_error(
                "the C11 drop-shadow merge payload is missing",
            ));
        };
        let source = exact_c08_read(request, RuntimeReadRole::CompositeSource)?;
        let shadow = exact_c08_read(request, RuntimeReadRole::Shadow)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C11 drop-shadow merge result is missing",
            ));
        };
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource())?;
        let shadow_binding = self.texture_binding_for_pass(request.id, shadow.resource())?;
        let shadow_spatial = self.validate_texture_binding(&shadow_binding, shadow.resource())?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let distinct = source_binding.allocation_resource() != shadow_binding.allocation_resource()
            && source_binding.allocation_resource() != target_binding.allocation_resource()
            && shadow_binding.allocation_resource() != target_binding.allocation_resource();
        if request.reads.len() != 2 || !distinct || shadow_spatial != target_spatial {
            return Err(preparation_error(
                "the C11 drop-shadow merge aliases or changed its shadow grid",
            ));
        }
        copy_c09_composite_parent(
            encoder,
            &shadow_binding,
            &target_binding,
            c09_composite_copy_extent(shadow_spatial),
        );
        let region = C08RenderRegion::bounded_source(source_spatial, target_spatial)?;
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: target_binding.view(),
                extent: target_spatial.device_extent,
                region,
                load: wgpu::LoadOp::Load,
                label: "Surgeist C11 unchanged source over drop shadow",
            },
        )?;
        let spatial = request
            .spatial_uniform
            .as_ref()
            .ok_or_else(|| preparation_error("the C11 merge spatial bytes are missing"))?;
        Ok(C11DropShadowMergeEncodingFacts {
            exact_prepared_bindings: true,
            distinct_source_shadow_and_result: distinct,
            validated_region: region.is_none_or(|region| {
                region.scissor_x.saturating_add(region.scissor_width)
                    <= target_spatial.device_extent.width()
                    && region.scissor_y.saturating_add(region.scissor_height)
                        <= target_spatial.device_extent.height()
            }),
            preserved_signed_mapping: c08_spatial_uniform_preserves_source_origin(
                spatial,
                source_spatial,
            ),
            fixed_source_over_blend: fixed_blend,
            reads_original_source_and_shadow: source.sampling_edge()
                == RuntimeSamplingEdge::TransparentBlack
                && shadow.sampling_edge() == RuntimeSamplingEdge::TransparentBlack,
        })
    }

    fn c09_composite_semantic<'prepared>(
        request: &'prepared C08PreparedPassEncodingRequest,
    ) -> Result<C09CompositeSemantic<'prepared>> {
        let RuntimePassKind::Composite(Some(RuntimeComposite {
            kind:
                RuntimeCompositeKind::Layer {
                    transform,
                    parameters,
                    clip_coverage,
                    ..
                },
            source_captured_before_outer_semantics: true,
        })) = &request.kind
        else {
            return Err(preparation_error(
                "the C09 layer composite changed its checked semantic payload",
            ));
        };
        let parent = *exact_c08_read(request, RuntimeReadRole::CompositeParent)?;
        let source = *exact_c08_read(request, RuntimeReadRole::CompositeSource)?;
        let clip = clip_coverage
            .map(|resource| {
                let read = *exact_c08_read(request, RuntimeReadRole::ClipCoverage)?;
                if read.resource() != resource {
                    return Err(preparation_error(
                        "the C09 clip coverage read changed its exact resource",
                    ));
                }
                Ok(read)
            })
            .transpose()?;
        let alpha_mask = parameters
            .alpha_mask()
            .map(|mask| {
                let read = *exact_c08_read(request, RuntimeReadRole::AlphaMask)?;
                if read.resource() != mask.resource() {
                    return Err(preparation_error(
                        "the C09 alpha-mask read changed its exact retained upload",
                    ));
                }
                Ok(read)
            })
            .transpose()?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C09 layer composite has no prepared result",
            ));
        };
        let expected_read_count = 2usize
            .saturating_add(usize::from(clip.is_some()))
            .saturating_add(usize::from(alpha_mask.is_some()));
        if request.reads.len() != expected_read_count {
            return Err(preparation_error(
                "the C09 layer composite contains an absent or duplicated semantic read",
            ));
        }
        let normal_path = parameters.blend() == BlendMode::Normal;
        Ok(C09CompositeSemantic {
            transform: *transform,
            parameters,
            parent,
            source,
            clip,
            alpha_mask,
            target,
            normal_path,
            destination_path: !normal_path,
        })
    }

    fn c09_composite_bindings<'prepared>(
        &'prepared self,
        request: &'prepared C08PreparedPassEncodingRequest,
        semantic: C09CompositeSemantic<'prepared>,
    ) -> Result<C09CompositeBindings<'prepared>> {
        let parent = self.texture_binding_for_pass(request.id, semantic.parent.resource())?;
        let parent_spatial = self.validate_texture_binding(&parent, semantic.parent.resource())?;
        let source = self.texture_binding_for_pass(request.id, semantic.source.resource())?;
        let source_spatial = self.validate_texture_binding(&source, semantic.source.resource())?;
        let target = self.texture_binding_for_pass(request.id, semantic.target)?;
        let target_spatial = self.validate_texture_binding(&target, semantic.target)?;
        let clip = semantic
            .clip
            .map(|read| {
                let binding = self.texture_binding_for_pass(request.id, read.resource())?;
                if self.validate_texture_binding(&binding, read.resource())? != target_spatial {
                    return Err(preparation_error(
                        "the C09 clip coverage grid changed from its parent mapping",
                    ));
                }
                Ok(binding)
            })
            .transpose()?;
        let mask = semantic
            .alpha_mask
            .map(|read| {
                let binding = self.texture_binding_for_pass(request.id, read.resource())?;
                let spatial = self.validate_texture_binding(&binding, read.resource())?;
                if semantic
                    .parameters
                    .alpha_mask()
                    .is_none_or(|mask| spatial.device_extent != mask.image_dimensions())
                {
                    return Err(preparation_error(
                        "the C09 alpha-mask texture changed from its exact image extent",
                    ));
                }
                Ok(binding)
            })
            .transpose()?;
        let parent_and_result_are_distinct = semantic.parent.resource() != semantic.target
            && parent.allocation_resource() != target.allocation_resource();
        let sampled_allocations_are_distinct = source.allocation_resource()
            != target.allocation_resource()
            && clip.as_ref().is_none_or(|binding| {
                binding.allocation_resource() != target.allocation_resource()
            })
            && mask.as_ref().is_none_or(|binding| {
                binding.allocation_resource() != target.allocation_resource()
            });
        let bindings = C09CompositeBindings {
            semantic,
            parent,
            source,
            target,
            clip,
            mask,
            parent_spatial,
            source_spatial,
            target_spatial,
            parent_and_result_are_distinct,
            sampled_allocations_are_distinct,
        };
        self.validate_c09_composite_bindings(&bindings)?;
        Ok(bindings)
    }

    fn validate_c09_composite_bindings(&self, bindings: &C09CompositeBindings<'_>) -> Result<()> {
        let semantic = &bindings.semantic;
        let clip_format_is_exact = semantic.clip.is_none_or(|read| {
            self.resource_request(read.resource()).is_ok_and(|request| {
                request.format == RuntimeResourceFormat::ClipCoverageRgba8Unorm
            })
        });
        let mask_format_is_exact = semantic.alpha_mask.is_none_or(|read| {
            self.resource_request(read.resource()).is_ok_and(|request| {
                request.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm
            })
        });
        if !bindings.parent_and_result_are_distinct
            || !bindings.sampled_allocations_are_distinct
            || bindings.parent_spatial != bindings.target_spatial
            || bindings.parent.texture().format() != bindings.target.texture().format()
            || self.resource_request(semantic.parent.resource())?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(semantic.source.resource())?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(semantic.target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || !clip_format_is_exact
            || !mask_format_is_exact
            || !bindings
                .parent
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::COPY_SRC)
            || (semantic.destination_path
                && !bindings
                    .parent
                    .texture()
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING))
            || !bindings
                .target
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT)
            || !bindings
                .source
                .texture()
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || bindings.clip.as_ref().is_some_and(|binding| {
                !binding
                    .texture()
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            })
            || bindings.mask.as_ref().is_some_and(|binding| {
                !binding
                    .texture()
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            })
        {
            return Err(preparation_error(
                "C09 parent, source, and distinct composite result bindings are inconsistent",
            ));
        }
        Ok(())
    }

    fn create_c09_composite_bind_group(
        &self,
        bindings: &C09CompositeBindings<'_>,
        objects: &ProvisionalCompositePassObjects<'_>,
        spatial: &PassSpatialUniformBytes,
        parameters: &CompositeParameterBytes,
        expected_read_count: usize,
    ) -> Result<(wgpu::BindGroup, bool)> {
        let spatial_buffer = self.create_c08_spatial_uniform_buffer(spatial);
        let parameter_buffer = self.create_c09_composite_parameter_buffer(parameters);
        let mut entries = Vec::with_capacity(expected_read_count.saturating_add(4));
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(bindings.source.view()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(objects.source_sampler()),
        });
        if bindings.semantic.destination_path {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(bindings.parent.view()),
            });
        }
        if let Some(binding) = &bindings.clip {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(binding.view()),
            });
        }
        if let Some(binding) = &bindings.mask {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(binding.view()),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: spatial_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: parameter_buffer.as_entire_binding(),
        });
        let binds_parent_sample = entries.iter().any(|entry| entry.binding == 2);
        let expected_entry_count = 4usize
            .saturating_add(usize::from(bindings.semantic.destination_path))
            .saturating_add(usize::from(bindings.clip.is_some()))
            .saturating_add(usize::from(bindings.mask.is_some()));
        if entries.len() != expected_entry_count
            || binds_parent_sample != bindings.semantic.destination_path
        {
            return Err(preparation_error(
                "the C09 composite bind group contains a dummy or missing resource",
            ));
        }
        Ok((
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Surgeist C09 exact layer-composite bindings"),
                layout: objects.bind_group_layout(),
                entries: &entries,
            }),
            binds_parent_sample,
        ))
    }

    fn encode_c09_layer_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C09LayerCompositeEncodingFacts> {
        let semantic = Self::c09_composite_semantic(request)?;
        let bindings = self.c09_composite_bindings(request, semantic)?;
        let normal_path = bindings.semantic.normal_path;
        let destination_path = bindings.semantic.destination_path;
        let source = bindings.semantic.source;
        let source_spatial = bindings.source_spatial;
        let target_spatial = bindings.target_spatial;
        let parent_spatial = bindings.parent_spatial;
        let parameters = bindings.semantic.parameters;
        let transform = bindings.semantic.transform;
        let parent_binding = &bindings.parent;
        let target_binding = &bindings.target;
        let parent_and_result_are_distinct = bindings.parent_and_result_are_distinct;
        let sampled_allocations_are_distinct = bindings.sampled_allocations_are_distinct;
        let expected_read_count = request.reads.len();

        let keys = request.cache_keys.as_ref().ok_or_else(|| {
            preparation_error("the C09 layer composite has no provisional cache keys")
        })?;
        if keys.samplers() != [source.sampler_key()] {
            return Err(preparation_error(
                "the C09 layer composite changed its one exact source sampler",
            ));
        }
        let spatial = request.spatial_uniform.as_ref().ok_or_else(|| {
            preparation_error("the C09 layer composite has no prepared spatial bytes")
        })?;
        let expected_spatial = PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
            source_spatial,
            target_spatial,
        )?;
        if spatial != &expected_spatial {
            return Err(preparation_error(
                "the C09 layer composite spatial bytes changed after preparation",
            ));
        }
        let composite_parameters = request.composite_parameters.as_ref().ok_or_else(|| {
            preparation_error("the C09 layer composite has no prepared parameter bytes")
        })?;
        let expected_parameters = CompositeParameterBytes::try_from_runtime_layer(parameters)?;
        if composite_parameters != &expected_parameters {
            return Err(preparation_error(
                "the C09 layer composite parameter bytes changed after preparation",
            ));
        }
        let objects = self.c09_composite_pass_objects(keys)?;
        objects.require_encoding_ready()?;
        if objects.path()
            != if normal_path {
                ShaderCompositePathKey::Normal
            } else {
                ShaderCompositePathKey::DestinationSampling
            }
            || objects.has_clip_coverage() != bindings.clip.is_some()
            || objects.has_alpha_mask() != bindings.mask.is_some()
        {
            return Err(preparation_error(
                "the C09 composite objects changed their checked entry-point interface",
            ));
        }

        let copy_extent = c09_composite_copy_extent(parent_spatial);
        copy_c09_composite_parent(encoder, parent_binding, target_binding, copy_extent);

        let (region, preserved_signed_mapping) =
            c09_composite_region_mapping(spatial, source_spatial, target_spatial, transform)?;
        let (bind_group, binds_parent_sample) = self.create_c09_composite_bind_group(
            &bindings,
            &objects,
            spatial,
            composite_parameters,
            expected_read_count,
        )?;
        encode_c09_composite_region(
            encoder,
            target_binding,
            &objects,
            &bind_group,
            region,
            target_spatial,
        )?;

        Ok(C09LayerCompositeEncodingFacts {
            normal_path,
            destination_path,
            fixed_premultiplied_blend: normal_path && objects.uses_fixed_source_over_blend(),
            omits_parent_sample: normal_path && !binds_parent_sample,
            copied_full_parent: copy_extent.width == target_binding.texture().width()
                && copy_extent.height == target_binding.texture().height(),
            avoids_read_write_alias: parent_and_result_are_distinct
                && sampled_allocations_are_distinct,
            exact_resources_and_parameters: spatial == &expected_spatial
                && &expected_parameters == composite_parameters
                && binds_parent_sample == destination_path
                && objects.uses_fixed_source_over_blend() == normal_path
                && objects.uses_replace_blend() == destination_path,
            preserved_signed_mapping,
        })
    }

    fn encode_c08_present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        output: &C08ExternalOutputView<'_>,
    ) -> Result<C08PassEncodingFacts> {
        let source = exact_c08_read(request, RuntimeReadRole::FinalWorkingImage)?;
        if request.result != RuntimeResultBinding::Output(output.format)
            || output.format != self.plan.output_format
            || output.extent != self.output_extent()?
        {
            return Err(preparation_error(
                "the C08 present pass changed its external output binding",
            ));
        }
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let _ = self.validate_texture_binding(&source_binding, source.resource)?;
        let region = C08RenderRegion::full(output.extent)?;
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: output.view,
                extent: output.extent,
                region: Some(region),
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                label: "Surgeist C08 present external output",
            },
        )?;
        Ok(C08PassEncodingFacts {
            full_target: true,
            exact_spatial_uniform: true,
            external_output_exact: true,
            fixed_source_over_blend: fixed_blend,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_sampled_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        source: &RuntimeReadBinding,
        target: C08SampledRenderTarget<'_>,
    ) -> Result<bool> {
        let spatial = request.spatial_uniform.as_ref().ok_or_else(|| {
            preparation_error("the C08 sampled pass has no exact prepared spatial uniform")
        })?;
        let keys = request.cache_keys.as_ref().ok_or_else(|| {
            preparation_error("the C08 sampled pass has no provisional cache keys")
        })?;
        if keys.samplers() != [source.sampler_key()] {
            return Err(preparation_error(
                "the C08 sampled pass changed its exact source sampler",
            ));
        }
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let _ = self.validate_texture_binding(&source_binding, source.resource())?;
        let objects = self.c08_pass_objects(keys)?;
        objects.require_encoding_ready()?;
        let uniform = self.create_c08_spatial_uniform_buffer(spatial);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C08 sampled pass bindings"),
            layout: objects.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_binding.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(objects.sampler()?),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        if let Some(region) = target.region {
            if region.scissor_x.saturating_add(region.scissor_width) > target.extent.width()
                || region.scissor_y.saturating_add(region.scissor_height) > target.extent.height()
            {
                return Err(preparation_error(
                    "the C08 bounded render region exceeds its exact target extent",
                ));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(target.label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: target.load,
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_viewport(
                region.viewport_x,
                region.viewport_y,
                region.viewport_width,
                region.viewport_height,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                region.scissor_x,
                region.scissor_y,
                region.scissor_width,
                region.scissor_height,
            );
            pass.draw(0..3, 0..1);
        }
        Ok(objects.uses_fixed_source_over_blend())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 resolves prepared leases at each validated runtime last-use point"
        )
    )]
    pub(crate) fn complete_pass(&mut self, pass: RuntimePassId) -> Result<()> {
        if self.c08_encoding_state.is_some() {
            return Err(preparation_error(
                "C08 pass completion belongs to its one-shot scheduler; discard an aborted graph and encoder",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_c08_custom_pass(&mut self, pass: RuntimePassId) -> Result<()> {
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Encoding) {
            return Err(preparation_error(
                "C08 custom-pass progress requires the active one-shot encoding session",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_c08_capture(
        &mut self,
        pass: RuntimePassId,
        target: RuntimeResourceId,
        session: &Arc<()>,
        receipt: C08VelloCaptureCompletionReceipt,
    ) -> Result<()> {
        let request = self.require_current_pass(pass)?;
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Encoding)
            || !matches!(request.runtime.kind, RuntimePassKind::VelloCapture(Some(_)))
            || request.runtime.result != RuntimeResultBinding::Resource(target)
            || receipt.pass != pass
            || receipt.target != target
            || !Arc::ptr_eq(&receipt.session, session)
        {
            return Err(preparation_error(
                "C08 capture completion does not match the exact pass, target, and encoding session",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_pass_inner(&mut self, pass: RuntimePassId) -> Result<()> {
        let request = self.require_current_pass(pass)?;
        let resource_releases = request.runtime.releases.clone();
        let kernel_releases = request.kernel_releases.clone();
        let releases_color_filter_operation = request.color_filter_operations.is_some();

        for resource in &resource_releases {
            let binding = self
                .resource_bindings
                .get(resource)
                .ok_or_else(|| preparation_error("prepared runtime release binding is missing"))?;
            if binding.lease.is_none() {
                return Err(preparation_error(
                    "prepared runtime release is stale or duplicate",
                ));
            }
        }
        for kernel in &kernel_releases {
            let binding = self
                .kernel_bindings
                .get(kernel)
                .ok_or_else(|| preparation_error("prepared Gaussian release binding is missing"))?;
            if binding.lease.is_none() {
                return Err(preparation_error(
                    "prepared Gaussian release is stale or duplicate",
                ));
            }
        }
        if releases_color_filter_operation
            && self
                .color_filter_operation_bindings
                .get(&pass)
                .is_none_or(|binding| binding.buffer.is_none())
        {
            return Err(preparation_error(
                "prepared color-filter operation release is stale or missing",
            ));
        }

        let Self {
            resource_bindings,
            kernel_bindings,
            color_filter_operation_bindings,
            frame_scope,
            ..
        } = self;
        let mut leases = Vec::with_capacity(resource_releases.len() + kernel_releases.len());
        for resource in &resource_releases {
            let lease = resource_bindings
                .get(resource)
                .and_then(|binding| binding.lease.as_ref())
                .expect("prepared resource releases were validated before atomic resolution");
            leases.push(lease);
        }
        for kernel in &kernel_releases {
            let lease = kernel_bindings
                .get(kernel)
                .and_then(|binding| binding.lease.as_ref())
                .expect("prepared kernel releases were validated before atomic resolution");
            leases.push(lease);
        }
        frame_scope
            .as_mut()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?
            .resolve_leases_atomically(&leases)?;
        for resource in resource_releases {
            let _ = resource_bindings
                .get_mut(&resource)
                .and_then(|binding| binding.lease.take())
                .expect("atomically resolved prepared resource must remain bound");
        }
        for kernel in kernel_releases {
            let _ = kernel_bindings
                .get_mut(&kernel)
                .and_then(|binding| binding.lease.take())
                .expect("atomically resolved prepared kernel must remain bound");
        }
        if releases_color_filter_operation {
            let _ = color_filter_operation_bindings
                .get_mut(&pass)
                .and_then(|binding| binding.buffer.take())
                .expect("validated color-filter operation buffer must remain bound");
        }
        self.next_pass = self.next_pass.saturating_add(1);
        Ok(())
    }

    pub(crate) fn finish_c08_submission(
        mut self,
        pending: C08PendingGraphEncoding,
    ) -> Result<C08PreparedGraphSubmission> {
        let completed_session = self.c08_completed_session.take().ok_or_else(|| {
            preparation_error("the prepared C08 graph has no completed encoding session")
        })?;
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Complete)
            || !Arc::ptr_eq(&completed_session, &pending.session)
            || !pending.summary.proves_complete_submission()
            || self.next_pass != self.plan.passes.len()
            || self
                .resource_bindings
                .values()
                .any(|binding| binding.lease.is_some())
            || self
                .kernel_bindings
                .values()
                .any(|binding| binding.lease.is_some())
            || self
                .color_filter_operation_bindings
                .values()
                .any(|binding| binding.buffer.is_some())
        {
            return Err(preparation_error(
                "the C08 graph submission does not own one exact completed prepared frame",
            ));
        }
        let pass_cache_update = self.pass_cache_update.take().ok_or_else(|| {
            preparation_error("the completed C08 graph lost its provisional pass-cache update")
        })?;
        let frame_scope = self.frame_scope.take().ok_or_else(|| {
            preparation_error("the completed C08 graph lost its prepared frame scope")
        })?;
        let C08PendingGraphEncoding {
            summary,
            resources: capture_resources,
            session: _,
        } = pending;
        Ok(C08PreparedGraphSubmission {
            capture_resources,
            prepared_frame: PendingC08PreparedFrameCommit {
                frame_scope,
                pass_cache_update,
            },
            activity: summary.activity(),
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 finishes a completely released prepared frame scope after execution"
        )
    )]
    pub(crate) fn finish(mut self) -> Result<FrameCleanup> {
        if self.next_pass != self.plan.passes.len()
            || self
                .resource_bindings
                .values()
                .any(|binding| binding.lease.is_some())
            || self
                .kernel_bindings
                .values()
                .any(|binding| binding.lease.is_some())
            || self
                .color_filter_operation_bindings
                .values()
                .any(|binding| binding.buffer.is_some())
        {
            return Err(preparation_error(
                "prepared graph cannot finish before every pass and last-use release",
            ));
        }
        let _ = self.pass_cache_update.take();
        self.frame_scope
            .take()
            .ok_or_else(|| preparation_error("prepared frame resource scope is already closed"))?
            .finish_checked()
    }
}
