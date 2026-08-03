use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

#[cfg(test)]
use std::cell::Cell;

use super::{
    BackendErrorCode, Color, Error, Format, PhysicalSize, Point, Rect, Result, Transform,
    backend::DeviceCapabilities,
    command::{RenderClip, RenderCommands},
    encode::{encode_vello_clip_coverage_scene, encode_vello_scene_with_initial_transform},
    filter::{
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS, ColorClampBoundary, RuntimeFilterAmount,
        RuntimeFilterAngle, RuntimeUnitFilterAmount,
    },
    frame::{
        GpuRenderGraph, GraphLoweringBlur, GraphLoweringBlurInput, GraphLoweringClipCoverage,
        GraphLoweringColorFilter, GraphLoweringComposite, GraphLoweringCompositeKind,
        GraphLoweringDropShadow, GraphLoweringEdgePolicy, GraphLoweringGeneration,
        GraphLoweringImportView, GraphLoweringInitialization, GraphLoweringPassId,
        GraphLoweringPassKind, GraphLoweringPassResult, GraphLoweringPassView,
        GraphLoweringReadBinding, GraphLoweringReadRole, GraphLoweringResourceId,
        GraphLoweringResourceProducer, GraphLoweringResourceRole, GraphLoweringResourceView,
        GraphLoweringSamplingEdge, GraphLoweringSamplingFilter, GraphLoweringSpatialDescriptor,
        GraphLoweringVelloCapture, GraphLoweringVelloSpan, GraphLoweringVelloSpanScope,
    },
    image::ResolvedMaskUploadDescriptor,
    layer::BlendMode,
    renderer::{Antialiasing, EffectQualityPolicy},
    resource::{
        FrameCleanup, FrameResourceScope, GaussianKernelBufferLimits, GaussianKernelKey,
        GaussianKernelPlan, GaussianKernelSamplingForm, ResourceAllocationPreflight,
        ResourceIdentity, ResourceLease, ResourceManager, WorkingFormat,
    },
    shader::{
        BindGroupLayoutKey, BlurEdgeParameterBytes, ColorFilterOperationBufferLimits,
        ColorFilterOperationBytes, CompositeParameterBytes, DevicePassCache,
        DropShadowParameterBytes, PassSpatialUniformBytes, ProvisionalC08PassObjects,
        ProvisionalColorFilterPassObjects, ProvisionalCompositePassObjects,
        ProvisionalDevicePassCacheUpdate, RenderPipelineKey, SamplerKey, ShaderBindingRoleKey,
        ShaderCompositeKey, ShaderCompositePathKey, ShaderDataBindingKey, ShaderMaskQualityKey,
        ShaderMaskSamplingKey, ShaderModuleKey, ShaderProgramKey, ShaderSamplingEdgeKey,
        ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    style::ColorFilterOp,
    texture::EffectTextureDescriptor,
    vello_engine::{
        ActiveVelloEncodingScope, EncodedVelloCaptureProof, PendingVelloResourceCommit,
        RasterParameters, TransactionEncodingState, TransactionTargetIntent, VelloEngineState,
        VelloResourceLeaseAggregate, scene::VelloScene,
    },
};

#[cfg(test)]
use super::texture::EffectTextureRole;

#[cfg(test)]
use super::resource::ResourceAccountingFault;

#[cfg(test)]
use super::frame::GraphLoweringView;
#[cfg(test)]
use super::frame::{FrameContext, FramePlan};

#[cfg(test)]
use super::vello_engine::{
    prepared_vello_pass_observation_for_test, scene::VelloPathDrawObservationForTest,
};

#[cfg(test)]
thread_local! {
    static ACTIVE_COLOR_FILTER_SHADER_FAILURE_FOR_TEST: Cell<bool> = const { Cell::new(false) };
}

/// Test-only deterministic failure at the checked color-filter shader boundary.
#[cfg(test)]
pub(crate) struct ScopedColorFilterShaderFailureForTest {
    previous: bool,
}

#[cfg(test)]
impl ScopedColorFilterShaderFailureForTest {
    pub(crate) fn after_checked_realization() -> Self {
        let previous =
            ACTIVE_COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ScopedColorFilterShaderFailureForTest {
    fn drop(&mut self) {
        ACTIVE_COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(|active| active.set(self.previous));
    }
}

#[cfg(test)]
fn inject_color_filter_shader_failure_for_test() -> Result<()> {
    if ACTIVE_COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(Cell::get) {
        return Err(preparation_error(
            "injected color-filter shader failure after checked realization",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08ExecutableSubsetObservationForTest {
    pub(crate) accepts_exact_rgba_and_bgra: bool,
    pub(crate) rejects_every_other_pass_kind_and_composite_payload: bool,
    pub(crate) rejects_missing_or_reordered_spine_passes: bool,
    pub(crate) rejects_malformed_dependencies_reads_results_and_releases: bool,
    pub(crate) rejects_graph_outside_base_subset: bool,
    pub(crate) preserves_direct_and_graph_planner_routes: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct C09ExecutableGraphObservationForTest {
    pub(crate) accepts_spine_and_layer_composition_for_all_formats: bool,
    pub(crate) layer_composition_reads_are_exact: bool,
    pub(crate) rejects_c10_plus_passes_and_payloads: bool,
    pub(crate) rejects_missing_payloads: bool,
    pub(crate) rejects_malformed_graph_facts: bool,
    pub(crate) rejects_unsupported_output_binding: bool,
    pub(crate) preserves_exact_c09_dispatch: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeColorOperationTagForTest {
    Brightness,
    Contrast,
    Grayscale,
    HueRotate,
    Invert,
    Opacity,
    Saturate,
    Sepia,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeFilterAmountObservationForTest {
    pub(crate) zero: bool,
    pub(crate) mantissa: f32,
    pub(crate) exponent: i32,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeColorScalarObservationForTest {
    Unit(f32),
    Amount(RuntimeFilterAmountObservationForTest),
    Angle { sine: f32, cosine: f32 },
}

#[cfg(test)]
impl RuntimeColorScalarObservationForTest {
    pub(crate) fn is_finite_normalized(self) -> bool {
        match self {
            Self::Unit(value) => value.is_finite() && (0.0..=1.0).contains(&value),
            Self::Amount(amount) if amount.zero => amount.mantissa == 0.0 && amount.exponent == 0,
            Self::Amount(amount) => {
                amount.mantissa.is_finite() && (0.5..1.0).contains(&amount.mantissa)
            }
            Self::Angle { sine, cosine } => sine.is_finite() && cosine.is_finite(),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeColorOperationObservationForTest {
    pub(crate) tag: RuntimeColorOperationTagForTest,
    pub(crate) scalar: RuntimeColorScalarObservationForTest,
    pub(crate) clamps_straight_rgba_then_premultiplies: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RuntimeColorFilterObservationForTest {
    pub(crate) operations: Vec<RuntimeColorOperationObservationForTest>,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBytesObservationForTest {
    pub(crate) bytes: Vec<u8>,
    pub(crate) preserves_one_clamp_per_record: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBufferLimitObservationForTest {
    pub(crate) count_overflow_is_exact: bool,
    pub(crate) max_buffer_size_is_exact: bool,
    pub(crate) max_storage_binding_size_is_exact: bool,
    pub(crate) equality_at_both_limits_is_accepted: bool,
    pub(crate) rejects_before_any_allocation_or_cache_action: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C10ColorFilterCacheRealizationObservationForTest {
    pub(crate) realizes_high_precision: bool,
    pub(crate) realizes_reduced_precision: bool,
    pub(crate) checked_scope_is_clean: bool,
    pub(crate) publishes_only_color_filter_entries: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C10ColorFilterLayoutObservationForTest {
    pub(crate) realizes_both_working_formats: bool,
    pub(crate) binds_exact_filter_source: bool,
    pub(crate) binds_exact_nearest_sampler: bool,
    pub(crate) binds_spatial_and_read_only_operations: bool,
    pub(crate) targets_only_the_working_format: bool,
    pub(crate) contains_no_dummy_binding: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11BlurLayoutObservationForTest {
    pub(crate) realizes_all_axis_input_and_precision_keys: bool,
    pub(crate) binds_exact_working_source: bool,
    pub(crate) binds_only_one_linear_sampler: bool,
    pub(crate) binds_spatial_and_read_only_kernel: bool,
    pub(crate) targets_only_the_working_format: bool,
    pub(crate) contains_no_dummy_binding: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11BlurCacheRealizationObservationForTest {
    pub(crate) realizes_all_eight_programs: bool,
    pub(crate) checked_scope_is_clean: bool,
    pub(crate) publishes_only_blur_entries: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11DropShadowLayoutObservationForTest {
    pub(crate) realizes_both_working_formats: bool,
    pub(crate) binds_exact_blurred_source_alpha: bool,
    pub(crate) binds_only_one_linear_transparent_sampler: bool,
    pub(crate) binds_spatial_and_parameters: bool,
    pub(crate) targets_only_the_working_format: bool,
    pub(crate) contains_no_dummy_binding: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11DropShadowCacheRealizationObservationForTest {
    pub(crate) realizes_checked_colorize_and_merge_programs: bool,
    pub(crate) checked_scope_is_clean: bool,
    pub(crate) merge_uses_fixed_premultiplied_source_over: bool,
    pub(crate) merge_omits_destination_sample: bool,
    pub(crate) publishes_only_drop_shadow_entries: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C10ExecutableGraphObservationForTest {
    pub(crate) accepts_spine_composition_and_color_for_all_formats: bool,
    pub(crate) accepts_multiple_ordered_color_runs: bool,
    pub(crate) rejects_empty_missing_and_malformed_color_facts: bool,
    pub(crate) rejects_copy_blur_shadow_and_drop_shadow_composite: bool,
    pub(crate) rejects_unsupported_output: bool,
    pub(crate) preserves_public_c09_dispatch_boundary: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C10ColorSpatialObservationForTest {
    pub(crate) logical_bounds: [f64; 4],
    pub(crate) device_origin: (i32, i32),
    pub(crate) device_extent: PhysicalSize,
    pub(crate) texel_origin: Point,
    pub(crate) raster_scale: f64,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ColorFilterGraphObservationForTest {
    pub(crate) operation_tags_by_run: Vec<Vec<RuntimeColorOperationTagForTest>>,
    pub(crate) first_source_spatial: Option<C10ColorSpatialObservationForTest>,
    pub(crate) every_run_has_one_source_and_distinct_result: bool,
    pub(crate) every_run_preserves_exact_spatial_descriptor: bool,
    pub(crate) every_operation_retains_one_clamp: bool,
    pub(crate) current_resource_advances_after_each_run: bool,
    pub(crate) dependencies_and_last_use_are_exact: bool,
    pub(crate) closed_color_facts_match_runtime_passes: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MixedColorUnsupportedDiagnosticObservationForTest {
    pub(crate) pure_color_retains_gpu_color_diagnostic: bool,
    pub(crate) color_then_blur_reports_gpu_blur_diagnostic: bool,
    pub(crate) mixed_graph_stays_outside_c10_preparation: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11ExecutableGraphObservationForTest {
    pub(crate) accepts_color_blur_and_drop_shadow_for_all_formats: bool,
    pub(crate) preserves_ordered_nonzero_filter_steps: bool,
    pub(crate) rejects_empty_missing_and_malformed_spatial_facts: bool,
    pub(crate) rejects_wrong_axes_inputs_edges_and_aliases: bool,
    pub(crate) rejects_copy_backdrop_stale_forward_and_c12_plus: bool,
    pub(crate) rejects_before_resource_acquisition: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct C11FilterGraphObservationForTest {
    pub(crate) pass_order: Vec<C11FilterPassTagForTest>,
    pub(crate) ordinary_blur_uses_transparent_black: bool,
    pub(crate) drop_shadow_uses_source_alpha_and_continuous_offset: bool,
    pub(crate) spatial_mappings_are_exact: bool,
    pub(crate) sources_and_results_are_distinct: bool,
    pub(crate) source_alpha_fanout_reads_original_twice: bool,
    pub(crate) original_source_releases_only_after_merge: bool,
    pub(crate) dependencies_and_last_use_are_exact: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12ExecutableGraphObservationForTest {
    pub(crate) accepts_bounded_top_level_backdrop: bool,
    pub(crate) rejects_outside_bounded_subset: bool,
    pub(crate) rejects_before_resource_acquisition: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12BackdropGraphObservationForTest {
    pub(crate) closed_subset_receipt: bool,
    pub(crate) reads_completed_parent_once: bool,
    pub(crate) copy_precedes_authored_filters: bool,
    pub(crate) post_filter_clip_precedes_foreground: bool,
    pub(crate) foreground_precedes_outer_composition: bool,
    pub(crate) later_sibling_depends_on_completed_group: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12CopyBackdropLayoutObservationForTest {
    pub(crate) realizes_both_working_formats: bool,
    pub(crate) binds_exact_completed_parent: bool,
    pub(crate) binds_only_one_nearest_transparent_sampler: bool,
    pub(crate) binds_only_spatial_uniform: bool,
    pub(crate) targets_only_the_working_format: bool,
    pub(crate) source_and_result_are_distinct: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12CopyBackdropCacheRealizationObservationForTest {
    pub(crate) realizes_high_precision: bool,
    pub(crate) realizes_reduced_precision: bool,
    pub(crate) checked_scope_is_clean: bool,
    pub(crate) publishes_only_copy_backdrop_entries: bool,
    pub(crate) rejects_unsupported_format_before_publication: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12BlurCacheRealizationObservationForTest {
    pub(crate) realizes_all_transparent_and_mirrored_programs: bool,
    pub(crate) checked_scope_is_clean: bool,
    pub(crate) publishes_exact_edge_programs: bool,
    pub(crate) edge_program_keys_are_distinct: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12BackdropBlurLayoutObservationForTest {
    pub(crate) realizes_all_axis_input_precision_and_edge_keys: bool,
    pub(crate) binds_exact_working_source: bool,
    pub(crate) binds_only_one_linear_mirror_sampler: bool,
    pub(crate) binds_spatial_kernel_and_semantic_bounds: bool,
    pub(crate) targets_only_the_working_format: bool,
    pub(crate) semantic_bounds_match_every_mirrored_read: bool,
    pub(crate) shader_mirrors_logical_bounds_before_texture_mapping: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct C12BackdropFilterChainObservationForTest {
    pub(crate) pass_order: Vec<C11FilterPassTagForTest>,
    pub(crate) every_backdrop_blur_uses_mirror: bool,
    pub(crate) source_alpha_blur_uses_mirror: bool,
    pub(crate) every_color_operation_retains_one_clamp: bool,
    pub(crate) semantic_bounds_are_exact: bool,
    pub(crate) every_mirrored_stage_is_realizable: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum C11FilterPassTagForTest {
    Color,
    BlurHorizontalRgba,
    BlurVerticalRgba,
    BlurHorizontalSourceAlpha,
    BlurVerticalSourceAlpha,
    DropShadowColorize,
    DropShadowMerge,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompositionOuterOperationObservationForTest {
    SourceMapping,
    ClipCoverage,
    AlphaMask,
    Opacity,
    Blend,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompositionReadObservationForTest {
    Parent,
    Source,
    ClipCoverage,
    AlphaMask,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClipCoverageElementObservationForTest {
    pub(crate) clip: RenderClip,
    pub(crate) transform: Transform,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClipCoverageCaptureObservationForTest {
    pub(crate) elements: Vec<ClipCoverageElementObservationForTest>,
    pub(crate) antialiasing: Antialiasing,
    pub(crate) device_origin: (i32, i32),
    pub(crate) target_extent: PhysicalSize,
    pub(crate) texel_origin: Point,
    pub(crate) raster_scale: f64,
    pub(crate) first_texel_center: Point,
    pub(crate) initial_transform: Transform,
    pub(crate) emitted_draws: Vec<VelloPathDrawObservationForTest>,
    pub(crate) uses_coverage_resource_role: bool,
    pub(crate) uses_rgba8_target: bool,
    pub(crate) uses_transparent_base: bool,
    pub(crate) raster_antialiasing: Antialiasing,
    pub(crate) raster_target_extent: PhysicalSize,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GraphClipCoverageObservationForTest {
    pub(crate) captures: Vec<ClipCoverageCaptureObservationForTest>,
    pub(crate) all_vello_capture_count: usize,
    pub(crate) composite_coverage_read_count: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerCompositionObservationForTest {
    pub(crate) transform: Transform,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) has_own_clip: bool,
    pub(crate) inherited_outer_clip_count: usize,
    pub(crate) inherited_outer_clip_transforms: Vec<Transform>,
    pub(crate) reads: Vec<CompositionReadObservationForTest>,
    pub(crate) outer_operations: Vec<CompositionOuterOperationObservationForTest>,
    pub(crate) source_captured_before_outer_semantics: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CompositionGraphObservationForTest {
    pub(crate) layers_inner_to_outer: Vec<LayerCompositionObservationForTest>,
    pub(crate) mask_identity_is_preserved: bool,
    pub(crate) root_surface_base_clears: usize,
    pub(crate) root_surface_base_color: Option<Color>,
    pub(crate) transparent_isolation_clears: usize,
    pub(crate) nontransparent_isolation_clears: usize,
}

#[cfg(test)]
pub(crate) fn c09_executable_graph_observation_for_test(
    c08_commands: RenderCommands,
    c09_commands: RenderCommands,
    c10_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C09ExecutableGraphObservationForTest {
    c09_executable_graph_observation(
        c08_commands,
        c09_commands,
        c10_commands,
        context,
        capabilities,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c10_executable_graph_observation_for_test(
    color_filters: Vec<super::FilterList>,
    blur_filters: Vec<super::FilterList>,
    shadow_filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C10ExecutableGraphObservationForTest {
    c10_executable_graph_observation(
        color_filters,
        blur_filters,
        shadow_filters,
        commands,
        context,
        capabilities,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn color_filter_graph_observation_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> ColorFilterGraphObservationForTest {
    color_filter_graph_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn mixed_color_unsupported_diagnostic_observation_for_test(
    color_filters: Vec<super::FilterList>,
    mixed_filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> MixedColorUnsupportedDiagnosticObservationForTest {
    mixed_color_unsupported_diagnostic_observation(
        color_filters,
        mixed_filters,
        commands,
        context,
        capabilities,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c11_executable_graph_observation_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11ExecutableGraphObservationForTest {
    c11_executable_graph_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c11_filter_graph_observation_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11FilterGraphObservationForTest {
    c11_filter_graph_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c12_executable_graph_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C12ExecutableGraphObservationForTest {
    c12_executable_graph_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c12_backdrop_graph_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C12BackdropGraphObservationForTest {
    c12_backdrop_graph_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c12_copy_backdrop_layout_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C12CopyBackdropLayoutObservationForTest {
    c12_copy_backdrop_layout_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) async fn c12_copy_backdrop_cache_realization_observation_for_test(
    device: &wgpu::Device,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C12CopyBackdropCacheRealizationObservationForTest> {
    capabilities.validate_supported_working_format(WorkingFormat::HighPrecision)?;
    capabilities.validate_supported_working_format(WorkingFormat::ReducedPrecision)?;
    let high = c12_copy_backdrop_cache_request_for_test(
        commands.clone(),
        context,
        capabilities,
        WorkingFormat::HighPrecision,
    )?;
    let reduced = c12_copy_backdrop_cache_request_for_test(
        commands.clone(),
        context,
        capabilities,
        WorkingFormat::ReducedPrecision,
    )?;
    let rejected_cache = DevicePassCache::new();
    let unsupported = DeviceCapabilities::from_test_facts(false, false, 4_096);
    let rejects_unsupported_format_before_publication = c12_copy_backdrop_cache_request_for_test(
        commands,
        context,
        unsupported,
        WorkingFormat::HighPrecision,
    )
    .is_err()
        && rejected_cache.counts_for_test().is_empty();
    realize_c12_copy_backdrop_requests(
        device,
        [high, reduced],
        rejects_unsupported_format_before_publication,
    )
    .await
}

#[cfg(test)]
pub(crate) fn color_filter_operation_bytes_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<ColorFilterOperationBytesObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the C10 operation-byte fixture did not produce a GPU graph",
        ));
    };
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let filters = lowered
        .passes
        .iter()
        .filter_map(|pass| match &pass.kind {
            RuntimePassKind::ColorFilter(Some(filter)) => Some(filter),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [filter] = filters.as_slice() else {
        return Err(lowering_error(
            "the C10 operation-byte fixture must contain one color-filter run",
        ));
    };
    let limits = ColorFilterOperationBufferLimits::for_test(u64::MAX, u64::MAX);
    let bytes = ColorFilterOperationBytes::try_from_runtime_operations_for_test(
        filter.operations(),
        limits,
    )?;
    Ok(ColorFilterOperationBytesObservationForTest {
        bytes: bytes.as_bytes().to_vec(),
        preserves_one_clamp_per_record: filter.operations().iter().all(|operation| {
            operation.clamp_boundary()
                == RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
        }),
    })
}

#[cfg(test)]
pub(crate) fn color_filter_operation_buffer_limit_observation_for_test()
-> ColorFilterOperationBufferLimitObservationForTest {
    let resources = ResourceManager::new(super::ResourceCacheBudget::DISABLED);
    let cache = DevicePassCache::new();
    let resources_before = resources.observation_for_test();
    let cache_before = cache.counts_for_test();
    let exact_error = |result: Result<u64>, field: &'static str| {
        result.is_err_and(|error| {
            error.code() == super::ErrorCode::InvalidInput
                && error
                    .invalid_value_diagnostic()
                    .is_some_and(|invalid| invalid.field() == field)
        })
    };
    let exact_byte_len = 16 + 32;
    let count_overflow_is_exact = exact_error(
        super::shader::color_filter_operation_byte_len_for_test(
            u64::from(u32::MAX) + 1,
            ColorFilterOperationBufferLimits::for_test(u64::MAX, u64::MAX),
        ),
        "color filter operation count",
    );
    let max_buffer_size_is_exact = exact_error(
        super::shader::color_filter_operation_byte_len_for_test(
            1,
            ColorFilterOperationBufferLimits::for_test(exact_byte_len - 1, exact_byte_len),
        ),
        "color filter operation buffer byte length",
    );
    let max_storage_binding_size_is_exact = exact_error(
        super::shader::color_filter_operation_byte_len_for_test(
            1,
            ColorFilterOperationBufferLimits::for_test(exact_byte_len, exact_byte_len - 1),
        ),
        "color filter operation buffer byte length",
    );
    let equality_at_both_limits_is_accepted =
        super::shader::color_filter_operation_byte_len_for_test(
            1,
            ColorFilterOperationBufferLimits::for_test(exact_byte_len, exact_byte_len),
        )
        .is_ok_and(|byte_len| byte_len == exact_byte_len);
    let resources_after = resources.observation_for_test();
    let cache_after = cache.counts_for_test();
    ColorFilterOperationBufferLimitObservationForTest {
        count_overflow_is_exact,
        max_buffer_size_is_exact,
        max_storage_binding_size_is_exact,
        equality_at_both_limits_is_accepted,
        rejects_before_any_allocation_or_cache_action: count_overflow_is_exact
            && max_buffer_size_is_exact
            && max_storage_binding_size_is_exact
            && resources_after == resources_before
            && cache_after == cache_before,
    }
}

#[cfg(test)]
pub(crate) async fn c10_color_filter_cache_realization_observation_for_test(
    device: &wgpu::Device,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C10ColorFilterCacheRealizationObservationForTest> {
    capabilities.validate_supported_working_format(WorkingFormat::HighPrecision)?;
    capabilities.validate_supported_working_format(WorkingFormat::ReducedPrecision)?;
    let high = c10_color_filter_cache_requests_for_test(
        commands.clone(),
        context,
        capabilities,
        WorkingFormat::HighPrecision,
    )?;
    let reduced = c10_color_filter_cache_requests_for_test(
        commands,
        context,
        capabilities,
        WorkingFormat::ReducedPrecision,
    )?;
    let mut cache = DevicePassCache::new();
    let mut update = cache.provisional_update();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let realization = (|| -> Result<(bool, bool)> {
        let mut high_count = 0_usize;
        for keys in high.passes() {
            let objects = update.realize_color_filter_pass(
                device,
                &cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            high_count += 1;
        }
        let mut reduced_count = 0_usize;
        for keys in reduced.passes() {
            let objects = update.realize_color_filter_pass(
                device,
                &cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            reduced_count += 1;
        }
        Ok((high_count == 1, reduced_count == 1))
    })();
    let scope_error = error_scope.pop().await;
    let (realizes_high_precision, realizes_reduced_precision) = realization?;
    if let Some(error) = scope_error {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            format!("C10 checked shader realization failed validation: {error}"),
        ));
    }
    update.commit(&mut cache)?;
    let all_requests_are_cached = high.passes().iter().chain(reduced.passes()).all(|keys| {
        cache.contains_color_filter_pass_for_test(
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        )
    });
    Ok(C10ColorFilterCacheRealizationObservationForTest {
        realizes_high_precision,
        realizes_reduced_precision,
        checked_scope_is_clean: true,
        publishes_only_color_filter_entries: all_requests_are_cached
            && cache.contains_only_two_color_filter_passes_for_test(),
    })
}

#[cfg(test)]
async fn realize_c12_copy_backdrop_requests(
    device: &wgpu::Device,
    requests: [C12CopyBackdropCacheRequestForTest; 2],
    rejects_unsupported_format_before_publication: bool,
) -> Result<C12CopyBackdropCacheRealizationObservationForTest> {
    let mut cache = DevicePassCache::new();
    let mut update = cache.provisional_update();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let realization = (|| -> Result<[bool; 2]> {
        let mut realized = [false; 2];
        for (index, request) in requests.iter().enumerate() {
            update
                .realize_copy_backdrop_pass(
                    device,
                    &cache,
                    request.keys.samplers(),
                    request.keys.layout(),
                    request.keys.shader(),
                    request.keys.pipeline(),
                )?
                .require_encoding_ready()?;
            realized[index] = true;
        }
        Ok(realized)
    })();
    let scope_error = error_scope.pop().await;
    let [realizes_high_precision, realizes_reduced_precision] = realization?;
    if let Some(error) = scope_error {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            format!("C12 checked backdrop-copy realization failed validation: {error}"),
        ));
    }
    update.commit(&mut cache)?;
    let all_requests_are_cached = requests.iter().all(|request| {
        cache.contains_copy_backdrop_pass_for_test(
            request.keys.samplers(),
            request.keys.layout(),
            request.keys.shader(),
            request.keys.pipeline(),
        )
    });
    Ok(C12CopyBackdropCacheRealizationObservationForTest {
        realizes_high_precision,
        realizes_reduced_precision,
        checked_scope_is_clean: true,
        publishes_only_copy_backdrop_entries: all_requests_are_cached
            && cache.contains_only_two_copy_backdrop_passes_for_test(),
        rejects_unsupported_format_before_publication,
    })
}

#[cfg(test)]
pub(crate) fn c10_color_filter_layout_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C10ColorFilterLayoutObservationForTest {
    c10_color_filter_layout_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c11_blur_layout_observation_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11BlurLayoutObservationForTest {
    c11_blur_layout_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) async fn c11_blur_cache_realization_observation_for_test(
    device: &wgpu::Device,
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C11BlurCacheRealizationObservationForTest> {
    capabilities.validate_supported_working_format(WorkingFormat::HighPrecision)?;
    capabilities.validate_supported_working_format(WorkingFormat::ReducedPrecision)?;
    let high = c11_blur_cache_requests_for_test(
        filters.clone(),
        commands.clone(),
        context,
        capabilities,
        WorkingFormat::HighPrecision,
    )?;
    let reduced = c11_blur_cache_requests_for_test(
        filters,
        commands,
        context,
        capabilities,
        WorkingFormat::ReducedPrecision,
    )?;
    let mut cache = DevicePassCache::new();
    let mut update = cache.provisional_update();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let realization = (|| -> Result<usize> {
        let mut count = 0_usize;
        for keys in high.iter().chain(&reduced) {
            update
                .realize_blur_pass(
                    device,
                    &cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            count += 1;
        }
        Ok(count)
    })();
    let scope_error = error_scope.pop().await;
    let realized_count = realization?;
    if let Some(error) = scope_error {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            format!("C11 checked blur realization failed validation: {error}"),
        ));
    }
    update.commit(&mut cache)?;
    let all_requests_are_cached = high.iter().chain(&reduced).all(|keys| {
        cache.contains_blur_pass_for_test(
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        )
    });
    Ok(C11BlurCacheRealizationObservationForTest {
        realizes_all_eight_programs: realized_count == 8,
        checked_scope_is_clean: true,
        publishes_only_blur_entries: all_requests_are_cached
            && cache.contains_only_eight_blur_passes_for_test(),
    })
}

#[cfg(test)]
pub(crate) async fn c12_blur_cache_realization_observation_for_test(
    device: &wgpu::Device,
    ordinary_filters: Vec<super::FilterList>,
    ordinary_commands: RenderCommands,
    backdrop_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C12BlurCacheRealizationObservationForTest> {
    capabilities.validate_supported_working_format(WorkingFormat::HighPrecision)?;
    capabilities.validate_supported_working_format(WorkingFormat::ReducedPrecision)?;
    let mut transparent = Vec::with_capacity(8);
    let mut mirrored = Vec::with_capacity(8);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        transparent.extend(c11_blur_cache_requests_for_test(
            ordinary_filters.clone(),
            ordinary_commands.clone(),
            context,
            capabilities,
            working_format,
        )?);
        mirrored.extend(
            c12_backdrop_blur_cache_requests_for_test(
                backdrop_commands.clone(),
                context,
                capabilities,
                working_format,
            )?
            .into_iter()
            .map(|request| request.keys),
        );
    }
    let edge_program_keys_are_distinct = transparent.iter().all(|ordinary| {
        mirrored
            .iter()
            .all(|backdrop| ordinary.shader() != backdrop.shader())
    });
    let mut cache = DevicePassCache::new();
    let mut update = cache.provisional_update();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let realization = (|| -> Result<usize> {
        let mut count = 0_usize;
        for keys in transparent.iter().chain(&mirrored) {
            update
                .realize_blur_pass(
                    device,
                    &cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            count = count.saturating_add(1);
        }
        Ok(count)
    })();
    let scope_error = error_scope.pop().await;
    let realized_count = realization?;
    if let Some(error) = scope_error {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            format!("C12 checked edge-aware blur realization failed validation: {error}"),
        ));
    }
    update.commit(&mut cache)?;
    let all_requests_are_cached = transparent.iter().chain(&mirrored).all(|keys| {
        cache.contains_blur_pass_for_test(
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        )
    });
    Ok(C12BlurCacheRealizationObservationForTest {
        realizes_all_transparent_and_mirrored_programs: realized_count == 16,
        checked_scope_is_clean: true,
        publishes_exact_edge_programs: all_requests_are_cached
            && cache.contains_only_sixteen_edge_blur_passes_for_test(),
        edge_program_keys_are_distinct,
    })
}

#[cfg(test)]
pub(crate) fn c12_backdrop_blur_layout_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C12BackdropBlurLayoutObservationForTest {
    c12_backdrop_blur_layout_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c12_backdrop_filter_chain_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C12BackdropFilterChainObservationForTest {
    c12_backdrop_filter_chain_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c11_drop_shadow_layout_observation_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11DropShadowLayoutObservationForTest {
    c11_drop_shadow_layout_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) async fn c11_drop_shadow_cache_realization_observation_for_test(
    device: &wgpu::Device,
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C11DropShadowCacheRealizationObservationForTest> {
    capabilities.validate_supported_working_format(WorkingFormat::HighPrecision)?;
    capabilities.validate_supported_working_format(WorkingFormat::ReducedPrecision)?;
    let high = c11_drop_shadow_cache_requests_for_test(
        filters.clone(),
        commands.clone(),
        context,
        capabilities,
        WorkingFormat::HighPrecision,
    )?;
    let reduced = c11_drop_shadow_cache_requests_for_test(
        filters,
        commands,
        context,
        capabilities,
        WorkingFormat::ReducedPrecision,
    )?;
    let mut cache = DevicePassCache::new();
    let mut update = cache.provisional_update();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let realization = (|| -> Result<usize> {
        let mut count = 0_usize;
        for requests in [&high, &reduced] {
            update
                .realize_drop_shadow_colorize_pass(
                    device,
                    &cache,
                    requests.colorize.samplers(),
                    requests.colorize.layout(),
                    requests.colorize.shader(),
                    requests.colorize.pipeline(),
                )?
                .require_encoding_ready()?;
            update
                .realize_c08_pass(
                    device,
                    &cache,
                    requests.merge.samplers(),
                    requests.merge.layout(),
                    requests.merge.shader(),
                    requests.merge.pipeline(),
                )?
                .require_encoding_ready()?;
            count += 2;
        }
        Ok(count)
    })();
    let scope_error = error_scope.pop().await;
    let realized_count = realization?;
    if let Some(error) = scope_error {
        return Err(Error::new(
            BackendErrorCode::RenderFailed,
            format!("C11 checked drop-shadow realization failed validation: {error}"),
        ));
    }
    update.commit(&mut cache)?;
    let all_requests_are_cached = [&high, &reduced].into_iter().all(|requests| {
        cache.contains_drop_shadow_colorize_pass_for_test(
            requests.colorize.samplers(),
            requests.colorize.layout(),
            requests.colorize.shader(),
            requests.colorize.pipeline(),
        ) && cache.contains_c08_pass_for_test(
            requests.merge.samplers(),
            requests.merge.layout(),
            requests.merge.shader(),
            requests.merge.pipeline(),
        )
    });
    let merge_facts = [&high, &reduced]
        .into_iter()
        .filter_map(|requests| {
            super::shader::c08_pass_key_facts_for_test(
                requests.merge.samplers(),
                requests.merge.layout(),
                requests.merge.shader(),
                requests.merge.pipeline(),
            )
        })
        .collect::<Vec<_>>();
    Ok(C11DropShadowCacheRealizationObservationForTest {
        realizes_checked_colorize_and_merge_programs: realized_count == 4,
        checked_scope_is_clean: true,
        merge_uses_fixed_premultiplied_source_over: merge_facts.iter().all(|facts| {
            facts.program == super::shader::C08ProgramForTest::DropShadowMerge
                && facts.has_fixed_source_over_blend
        }),
        merge_omits_destination_sample: merge_facts.iter().all(|facts| {
            facts.source_role == ShaderBindingRoleKey::CompositeSource
                && facts.has_only_spatial_uniform
        }),
        publishes_only_drop_shadow_entries: all_requests_are_cached
            && cache.contains_only_four_drop_shadow_passes_for_test(),
    })
}

#[cfg(test)]
pub(crate) fn runtime_color_filter_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<RuntimeColorFilterObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the color-filter lowering fixture did not produce a GPU graph",
        ));
    };
    let plan = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let operations = plan
        .passes
        .iter()
        .filter_map(|pass| match &pass.kind {
            RuntimePassKind::ColorFilter(Some(filter)) => Some(&filter.operations),
            RuntimePassKind::ClearRoot { .. }
            | RuntimePassKind::VelloCapture(_)
            | RuntimePassKind::CanonicalizeCapture
            | RuntimePassKind::CopyBackdrop
            | RuntimePassKind::ColorFilter(None)
            | RuntimePassKind::BlurHorizontal(_)
            | RuntimePassKind::BlurVertical(_)
            | RuntimePassKind::DropShadowColorize(_)
            | RuntimePassKind::Composite(_)
            | RuntimePassKind::Present => None,
        })
        .flatten()
        .map(runtime_color_operation_observation_for_test)
        .collect::<Vec<_>>();
    if operations.is_empty() {
        return Err(lowering_error(
            "the color-filter lowering fixture produced no runtime operations",
        ));
    }
    Ok(RuntimeColorFilterObservationForTest { operations })
}

#[cfg(test)]
fn runtime_color_operation_observation_for_test(
    operation: &RuntimeColorOperation,
) -> RuntimeColorOperationObservationForTest {
    let (tag, scalar) = match operation.operation {
        RuntimeColorOperationKind::Brightness(amount) => (
            RuntimeColorOperationTagForTest::Brightness,
            observe_runtime_filter_amount_for_test(amount),
        ),
        RuntimeColorOperationKind::Contrast(amount) => (
            RuntimeColorOperationTagForTest::Contrast,
            observe_runtime_filter_amount_for_test(amount),
        ),
        RuntimeColorOperationKind::Grayscale(amount) => (
            RuntimeColorOperationTagForTest::Grayscale,
            RuntimeColorScalarObservationForTest::Unit(amount.value()),
        ),
        RuntimeColorOperationKind::HueRotate(angle) => (
            RuntimeColorOperationTagForTest::HueRotate,
            RuntimeColorScalarObservationForTest::Angle {
                sine: angle.sine(),
                cosine: angle.cosine(),
            },
        ),
        RuntimeColorOperationKind::Invert(amount) => (
            RuntimeColorOperationTagForTest::Invert,
            RuntimeColorScalarObservationForTest::Unit(amount.value()),
        ),
        RuntimeColorOperationKind::Opacity(amount) => (
            RuntimeColorOperationTagForTest::Opacity,
            RuntimeColorScalarObservationForTest::Unit(amount.value()),
        ),
        RuntimeColorOperationKind::Saturate(amount) => (
            RuntimeColorOperationTagForTest::Saturate,
            observe_runtime_filter_amount_for_test(amount),
        ),
        RuntimeColorOperationKind::Sepia(amount) => (
            RuntimeColorOperationTagForTest::Sepia,
            RuntimeColorScalarObservationForTest::Unit(amount.value()),
        ),
    };
    RuntimeColorOperationObservationForTest {
        tag,
        scalar,
        clamps_straight_rgba_then_premultiplies: matches!(
            operation.clamp_boundary,
            RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
        ),
    }
}

#[cfg(test)]
fn observe_runtime_filter_amount_for_test(
    amount: RuntimeFilterAmount,
) -> RuntimeColorScalarObservationForTest {
    RuntimeColorScalarObservationForTest::Amount(RuntimeFilterAmountObservationForTest {
        zero: amount.zero(),
        mantissa: amount.mantissa(),
        exponent: amount.exponent(),
    })
}

#[cfg(test)]
pub(crate) fn composition_graph_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> CompositionGraphObservationForTest {
    composition_graph_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn graph_clip_coverage_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> GraphClipCoverageObservationForTest {
    graph_clip_coverage_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08PassLayoutObservationForTest {
    pub(crate) canonicalize_binds_capture_and_spatial_only: bool,
    pub(crate) span_source_over_binds_source_and_spatial_only: bool,
    pub(crate) present_binds_final_image_and_spatial_only: bool,
    pub(crate) copy_only_parent_is_not_sampled: bool,
    pub(crate) dummy_parameters_are_not_bound: bool,
    pub(crate) c09_typed_vocabulary_is_preserved: bool,
    pub(crate) output_specialization_is_exact: bool,
}

#[cfg(test)]
pub(crate) struct C08PassCacheRequestsForTest {
    passes: Vec<RuntimePassCacheKeys>,
}

#[cfg(test)]
impl C08PassCacheRequestsForTest {
    pub(crate) fn passes(&self) -> &[RuntimePassCacheKeys] {
        &self.passes
    }
}

#[cfg(test)]
pub(crate) struct C09CompositeCacheRequestsForTest {
    passes: Vec<RuntimePassCacheKeys>,
}

#[cfg(test)]
impl C09CompositeCacheRequestsForTest {
    pub(crate) fn passes(&self) -> &[RuntimePassCacheKeys] {
        &self.passes
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn composite_pass(
        &self,
        path: ShaderCompositePathKey,
        has_clip_coverage: bool,
        has_alpha_mask: bool,
    ) -> Option<&RuntimePassCacheKeys> {
        self.passes.iter().find(|keys| {
            super::shader::c09_composite_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
            .is_some_and(|facts| {
                facts.path == path
                    && facts.has_clip_coverage == has_clip_coverage
                    && facts.has_alpha_mask == has_alpha_mask
            })
        })
    }
}

#[cfg(test)]
pub(crate) struct C10ColorFilterCacheRequestsForTest {
    passes: Vec<RuntimePassCacheKeys>,
}

#[cfg(test)]
impl C10ColorFilterCacheRequestsForTest {
    pub(crate) fn passes(&self) -> &[RuntimePassCacheKeys] {
        &self.passes
    }
}

#[cfg(test)]
pub(crate) fn c09_composite_cache_requests_for_test(
    command_sets: &[RenderCommands],
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<C09CompositeCacheRequestsForTest> {
    let mut passes = Vec::with_capacity(command_sets.len());
    for commands in command_sets {
        let FramePlan::GpuGraph(graph) = commands.clone().plan_for(context)? else {
            return Err(lowering_error(
                "the C09 composite cache fixture did not produce a GPU graph",
            ));
        };
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let resource_formats = lowered
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let layer_passes = lowered.passes.iter().filter_map(|pass| {
            matches!(
                &pass.kind,
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::Layer { .. },
                    ..
                }))
            )
            .then_some(pass.cache_keys.as_ref())
            .flatten()
        });
        let mut found_layer = false;
        for keys in layer_passes {
            found_layer = true;
            if !passes.contains(keys) {
                passes.push(keys.clone());
            }
        }
        if !found_layer {
            return Err(lowering_error(
                "the C09 composite cache fixture has no layer-composite keys",
            ));
        }
        for pass in &lowered.passes {
            let RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) = &pass.kind
            else {
                continue;
            };
            if parameters.blend() == BlendMode::Normal {
                continue;
            }
            let mut normal_kind = pass.kind.clone();
            let RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) = &mut normal_kind
            else {
                unreachable!("the cloned C09 layer-composite kind must remain a layer")
            };
            parameters.blend = BlendMode::Normal;
            let Some(keys) = runtime_pass_cache_keys(
                &normal_kind,
                &pass.reads,
                pass.result,
                working_format,
                Format::Rgba8,
                &resource_formats,
            )?
            else {
                return Err(lowering_error(
                    "the C09 normal-path cache fixture lost its exact keys",
                ));
            };
            if !passes.contains(&keys) {
                passes.push(keys);
            }
        }
    }
    if passes.is_empty() {
        return Err(lowering_error(
            "the C09 composite cache fixture contains no composite variants",
        ));
    }
    Ok(C09CompositeCacheRequestsForTest { passes })
}

#[cfg(test)]
fn c10_color_filter_cache_requests_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<C10ColorFilterCacheRequestsForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the C10 color-filter cache fixture did not produce a GPU graph",
        ));
    };
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let passes = lowered
        .passes
        .iter()
        .filter_map(|pass| match &pass.kind {
            RuntimePassKind::ColorFilter(Some(filter)) if !filter.operations().is_empty() => {
                pass.cache_keys.clone()
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if passes.len() != 1 {
        return Err(lowering_error(
            "the C10 color-filter cache fixture must contain one checked program request",
        ));
    }
    Ok(C10ColorFilterCacheRequestsForTest { passes })
}

#[cfg(test)]
struct C12CopyBackdropCacheRequestForTest {
    keys: RuntimePassCacheKeys,
    source_and_result_are_distinct: bool,
}

#[cfg(test)]
fn c12_copy_backdrop_cache_request_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<C12CopyBackdropCacheRequestForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the C12 backdrop-copy cache fixture did not produce a GPU graph",
        ));
    };
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let PrePreparationGraphClassification::ExactC12(preparable) =
        PrePreparationGraphClassification::classify(lowered)
    else {
        return Err(lowering_error(
            "the C12 backdrop-copy fixture is outside the exact bounded graph",
        ));
    };
    let copies = preparable
        .closed
        .lowered
        .passes
        .iter()
        .filter(|pass| matches!(pass.kind, RuntimePassKind::CopyBackdrop))
        .collect::<Vec<_>>();
    let [copy] = copies.as_slice() else {
        return Err(lowering_error(
            "the C12 backdrop-copy fixture must contain one checked copy request",
        ));
    };
    let [parent] = copy.reads.as_slice() else {
        return Err(lowering_error(
            "the C12 backdrop-copy fixture must read one completed parent",
        ));
    };
    let RuntimeResultBinding::Resource(result) = copy.result else {
        return Err(lowering_error(
            "the C12 backdrop-copy fixture must write one working resource",
        ));
    };
    Ok(C12CopyBackdropCacheRequestForTest {
        keys: copy
            .cache_keys
            .clone()
            .ok_or_else(|| lowering_error("the C12 backdrop-copy cache keys are missing"))?,
        source_and_result_are_distinct: parent.resource != result,
    })
}

#[cfg(test)]
fn c12_copy_backdrop_layout_observation(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C12CopyBackdropLayoutObservationForTest> {
    let mut requests = Vec::with_capacity(2);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        requests.push(c12_copy_backdrop_cache_request_for_test(
            commands.clone(),
            context,
            capabilities,
            working_format,
        )?);
    }
    let facts = requests
        .iter()
        .filter_map(|request| {
            super::shader::c12_copy_backdrop_pass_key_facts_for_test(
                request.keys.samplers(),
                request.keys.layout(),
                request.keys.shader(),
                request.keys.pipeline(),
            )
        })
        .collect::<Vec<_>>();
    Ok(c12_copy_backdrop_layout_facts(&requests, &facts))
}

#[cfg(test)]
fn c12_copy_backdrop_layout_facts(
    requests: &[C12CopyBackdropCacheRequestForTest],
    facts: &[super::shader::C12CopyBackdropPassKeyFactsForTest],
) -> C12CopyBackdropLayoutObservationForTest {
    let realizes_both_working_formats = facts.len() == 2
        && [
            ShaderTextureFormatKey::working(WorkingFormat::HighPrecision),
            ShaderTextureFormatKey::working(WorkingFormat::ReducedPrecision),
        ]
        .into_iter()
        .all(|format| {
            facts
                .iter()
                .filter(|facts| facts.working_format == format)
                .count()
                == 1
        });
    C12CopyBackdropLayoutObservationForTest {
        realizes_both_working_formats,
        binds_exact_completed_parent: facts.iter().all(|facts| {
            facts.source_role == ShaderBindingRoleKey::CompletedParent
                && facts.source_format == facts.working_format
        }),
        binds_only_one_nearest_transparent_sampler: facts
            .iter()
            .all(|facts| facts.has_only_nearest_transparent_sampler),
        binds_only_spatial_uniform: facts.iter().all(|facts| facts.has_only_spatial_uniform),
        targets_only_the_working_format: facts
            .iter()
            .all(|facts| facts.target_format == facts.working_format),
        source_and_result_are_distinct: requests
            .iter()
            .all(|request| request.source_and_result_are_distinct),
    }
}

#[cfg(test)]
fn c11_blur_cache_requests_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<Vec<RuntimePassCacheKeys>> {
    let (_, lowered) = lower_authored_c10_graph_for_test(
        filters,
        commands,
        context,
        working_format,
        Format::Rgba8,
        &capabilities,
    )
    .ok_or_else(|| lowering_error("the C11 blur cache fixture did not produce a GPU graph"))?;
    let preparable = c11_preparable_graph_for_test(lowered)?;
    let passes = preparable
        .closed
        .lowered
        .passes
        .iter()
        .filter_map(|pass| {
            matches!(
                pass.kind,
                RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_))
            )
            .then(|| pass.cache_keys.clone())
            .flatten()
        })
        .collect::<Vec<_>>();
    if passes.len() != 4 {
        return Err(lowering_error(
            "the C11 blur cache fixture must contain four axis/input program requests",
        ));
    }
    Ok(passes)
}

#[cfg(test)]
struct C12BackdropBlurCacheRequestForTest {
    keys: RuntimePassCacheKeys,
    horizontal: bool,
    source_alpha: bool,
    semantic_bounds: Rect,
    read_edge: RuntimeSamplingEdge,
    edge_parameters: BlurEdgeParameterBytes,
}

#[cfg(test)]
fn c12_backdrop_blur_cache_requests_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<Vec<C12BackdropBlurCacheRequestForTest>> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the C12 backdrop blur cache fixture did not produce a GPU graph",
        ));
    };
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let PrePreparationGraphClassification::ExactC12(preparable) =
        PrePreparationGraphClassification::classify(lowered)
    else {
        return Err(lowering_error(
            "the C12 backdrop blur fixture is outside the exact bounded graph",
        ));
    };
    let mut requests = Vec::with_capacity(4);
    for pass in &preparable.closed.lowered.passes {
        let blur = match &pass.kind {
            RuntimePassKind::BlurHorizontal(Some(blur))
            | RuntimePassKind::BlurVertical(Some(blur)) => blur,
            _ => continue,
        };
        let RuntimeSamplingEdge::SemanticBorderMirror(bounds) = blur.edge else {
            continue;
        };
        let [read] = pass.reads.as_slice() else {
            return Err(lowering_error(
                "a C12 mirrored blur must read one filter source",
            ));
        };
        requests.push(C12BackdropBlurCacheRequestForTest {
            keys: pass
                .cache_keys
                .clone()
                .ok_or_else(|| lowering_error("a C12 mirrored blur has no checked cache keys"))?,
            horizontal: blur.axis == RuntimeBlurAxis::Horizontal,
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
            semantic_bounds: bounds,
            read_edge: read.sampling_edge,
            edge_parameters: BlurEdgeParameterBytes::try_from_semantic_bounds(bounds)?,
        });
    }
    if requests.len() != 4 {
        return Err(lowering_error(
            "the C12 backdrop blur fixture must contain four mirrored axis/input requests",
        ));
    }
    Ok(requests)
}

#[cfg(test)]
fn c12_backdrop_blur_layout_observation(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C12BackdropBlurLayoutObservationForTest> {
    let mut requests = Vec::with_capacity(8);
    let mut facts = Vec::with_capacity(8);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for request in c12_backdrop_blur_cache_requests_for_test(
            commands.clone(),
            context,
            capabilities,
            working_format,
        )? {
            let Some(observed) = super::shader::c12_backdrop_blur_pass_key_facts_for_test(
                request.keys.samplers(),
                request.keys.layout(),
                request.keys.shader(),
                request.keys.pipeline(),
            ) else {
                return Ok(C12BackdropBlurLayoutObservationForTest::default());
            };
            requests.push(request);
            facts.push(observed);
        }
    }
    let realizes_all_axis_input_precision_and_edge_keys = facts.len() == 8
        && requests.iter().zip(&facts).all(|(request, facts)| {
            request.horizontal == facts.horizontal && request.source_alpha == facts.source_alpha
        })
        && [
            ShaderTextureFormatKey::working(WorkingFormat::HighPrecision),
            ShaderTextureFormatKey::working(WorkingFormat::ReducedPrecision),
        ]
        .into_iter()
        .all(|format| {
            [true, false].into_iter().all(|horizontal| {
                [true, false].into_iter().all(|source_alpha| {
                    facts
                        .iter()
                        .filter(|facts| {
                            facts.working_format == format
                                && facts.horizontal == horizontal
                                && facts.source_alpha == source_alpha
                        })
                        .count()
                        == 1
                })
            })
        });
    let semantic_bounds_match_every_mirrored_read = requests.iter().all(|request| {
        request.read_edge == RuntimeSamplingEdge::SemanticBorderMirror(request.semantic_bounds)
            && BlurEdgeParameterBytes::try_from_semantic_bounds(request.semantic_bounds)
                .is_ok_and(|expected| request.edge_parameters == expected)
    });
    Ok(C12BackdropBlurLayoutObservationForTest {
        realizes_all_axis_input_precision_and_edge_keys,
        binds_exact_working_source: facts.iter().all(|facts| {
            facts.source_role == ShaderBindingRoleKey::FilterSource
                && facts.source_format == facts.working_format
        }),
        binds_only_one_linear_mirror_sampler: facts
            .iter()
            .all(|facts| facts.has_only_linear_mirror_sampler),
        binds_spatial_kernel_and_semantic_bounds: facts
            .iter()
            .all(|facts| facts.has_exact_data_bindings),
        targets_only_the_working_format: facts
            .iter()
            .all(|facts| facts.target_format == facts.working_format),
        semantic_bounds_match_every_mirrored_read,
        shader_mirrors_logical_bounds_before_texture_mapping:
            super::shader::c12_blur_shader_mirrors_semantic_bounds_before_texture_mapping_for_test(),
    })
}

#[cfg(test)]
fn c11_blur_layout_observation(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C11BlurLayoutObservationForTest> {
    let mut facts = Vec::with_capacity(8);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let requests = c11_blur_cache_requests_for_test(
            filters.clone(),
            commands.clone(),
            context,
            capabilities,
            working_format,
        )?;
        for keys in &requests {
            let Some(observed) = super::shader::c11_blur_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            ) else {
                return Ok(C11BlurLayoutObservationForTest::default());
            };
            facts.push(observed);
        }
    }
    let realizes_all_axis_input_and_precision_keys = facts.len() == 8
        && [
            ShaderTextureFormatKey::working(WorkingFormat::HighPrecision),
            ShaderTextureFormatKey::working(WorkingFormat::ReducedPrecision),
        ]
        .into_iter()
        .all(|format| {
            [true, false].into_iter().all(|horizontal| {
                [true, false].into_iter().all(|source_alpha| {
                    facts
                        .iter()
                        .filter(|facts| {
                            facts.working_format == format
                                && facts.horizontal == horizontal
                                && facts.source_alpha == source_alpha
                        })
                        .count()
                        == 1
                })
            })
        });
    let binds_exact_working_source = facts.iter().all(|facts| {
        facts.source_role == ShaderBindingRoleKey::FilterSource
            && facts.source_format == facts.working_format
    });
    let binds_only_one_linear_sampler = facts
        .iter()
        .all(|facts| facts.has_only_linear_source_sampler);
    let binds_spatial_and_read_only_kernel =
        facts.iter().all(|facts| facts.has_exact_data_bindings);
    let targets_only_the_working_format = facts
        .iter()
        .all(|facts| facts.target_format == facts.working_format);
    Ok(C11BlurLayoutObservationForTest {
        realizes_all_axis_input_and_precision_keys,
        binds_exact_working_source,
        binds_only_one_linear_sampler,
        binds_spatial_and_read_only_kernel,
        targets_only_the_working_format,
        contains_no_dummy_binding: realizes_all_axis_input_and_precision_keys
            && binds_exact_working_source
            && binds_only_one_linear_sampler
            && binds_spatial_and_read_only_kernel,
    })
}

#[cfg(test)]
struct C11DropShadowCacheRequestsForTest {
    colorize: RuntimePassCacheKeys,
    merge: RuntimePassCacheKeys,
}

#[cfg(test)]
fn c11_drop_shadow_cache_requests_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<C11DropShadowCacheRequestsForTest> {
    let (_, lowered) = lower_authored_c10_graph_for_test(
        filters,
        commands,
        context,
        working_format,
        Format::Rgba8,
        &capabilities,
    )
    .ok_or_else(|| lowering_error("the C11 drop-shadow fixture did not produce a GPU graph"))?;
    let preparable = c11_preparable_graph_for_test(lowered)?;
    let mut colorize = None;
    let mut merge = None;
    for pass in &preparable.closed.lowered.passes {
        match &pass.kind {
            RuntimePassKind::DropShadowColorize(Some(_)) => {
                set_unique_drop_shadow_cache_keys(
                    &mut colorize,
                    pass.cache_keys.clone(),
                    "the C11 fixture contains more than one drop-shadow colorize request",
                )?;
            }
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::DropShadow,
                ..
            })) => {
                set_unique_drop_shadow_cache_keys(
                    &mut merge,
                    pass.cache_keys.clone(),
                    "the C11 fixture contains more than one drop-shadow merge request",
                )?;
            }
            _ => {}
        }
    }
    let colorize = colorize
        .flatten()
        .ok_or_else(|| lowering_error("the C11 fixture lost its drop-shadow colorize keys"))?;
    let merge = merge
        .flatten()
        .ok_or_else(|| lowering_error("the C11 fixture lost its drop-shadow merge keys"))?;
    Ok(C11DropShadowCacheRequestsForTest { colorize, merge })
}

#[cfg(test)]
fn set_unique_drop_shadow_cache_keys(
    slot: &mut Option<Option<RuntimePassCacheKeys>>,
    keys: Option<RuntimePassCacheKeys>,
    duplicate_message: &'static str,
) -> Result<()> {
    if slot.replace(keys).is_some() {
        return Err(lowering_error(duplicate_message));
    }
    Ok(())
}

#[cfg(test)]
fn c11_drop_shadow_layout_observation(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C11DropShadowLayoutObservationForTest> {
    let mut facts = Vec::with_capacity(2);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let requests = c11_drop_shadow_cache_requests_for_test(
            filters.clone(),
            commands.clone(),
            context,
            capabilities,
            working_format,
        )?;
        let Some(observed) = super::shader::c11_drop_shadow_colorize_key_facts_for_test(
            requests.colorize.samplers(),
            requests.colorize.layout(),
            requests.colorize.shader(),
            requests.colorize.pipeline(),
        ) else {
            return Ok(C11DropShadowLayoutObservationForTest::default());
        };
        facts.push(observed);
    }
    let expected_formats = [
        ShaderTextureFormatKey::working(WorkingFormat::HighPrecision),
        ShaderTextureFormatKey::working(WorkingFormat::ReducedPrecision),
    ];
    let realizes_both_working_formats = facts.len() == 2
        && expected_formats.into_iter().all(|format| {
            facts
                .iter()
                .filter(|facts| facts.working_format == format)
                .count()
                == 1
        });
    let binds_exact_blurred_source_alpha = facts.iter().all(|facts| {
        facts.source_role == ShaderBindingRoleKey::BlurredSourceAlpha
            && facts.source_format == facts.working_format
    });
    let binds_only_one_linear_transparent_sampler = facts
        .iter()
        .all(|facts| facts.has_only_linear_transparent_sampler);
    let binds_spatial_and_parameters = facts.iter().all(|facts| facts.has_exact_data_bindings);
    let targets_only_the_working_format = facts
        .iter()
        .all(|facts| facts.target_format == facts.working_format);
    Ok(C11DropShadowLayoutObservationForTest {
        realizes_both_working_formats,
        binds_exact_blurred_source_alpha,
        binds_only_one_linear_transparent_sampler,
        binds_spatial_and_parameters,
        targets_only_the_working_format,
        contains_no_dummy_binding: realizes_both_working_formats
            && binds_exact_blurred_source_alpha
            && binds_only_one_linear_transparent_sampler
            && binds_spatial_and_parameters,
    })
}

#[cfg(test)]
fn c10_color_filter_layout_observation(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Result<C10ColorFilterLayoutObservationForTest> {
    let mut facts = Vec::with_capacity(2);
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let requests = c10_color_filter_cache_requests_for_test(
            commands.clone(),
            context,
            capabilities,
            working_format,
        )?;
        for keys in requests.passes() {
            let Some(observed) = super::shader::c10_color_filter_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            ) else {
                return Ok(C10ColorFilterLayoutObservationForTest::default());
            };
            facts.push(observed);
        }
    }
    let realizes_both_working_formats = facts.len() == 2
        && [
            ShaderTextureFormatKey::working(WorkingFormat::HighPrecision),
            ShaderTextureFormatKey::working(WorkingFormat::ReducedPrecision),
        ]
        .into_iter()
        .all(|working_format| {
            facts
                .iter()
                .filter(|facts| facts.working_format == working_format)
                .count()
                == 1
        });
    let binds_exact_filter_source = facts.iter().all(|facts| {
        facts.source_role == ShaderBindingRoleKey::FilterSource
            && facts.source_format == facts.working_format
    });
    let binds_exact_nearest_sampler = facts
        .iter()
        .all(|facts| facts.has_only_nearest_source_sampler);
    let binds_spatial_and_read_only_operations =
        facts.iter().all(|facts| facts.has_exact_data_bindings);
    let targets_only_the_working_format = facts
        .iter()
        .all(|facts| facts.target_format == facts.working_format);
    Ok(C10ColorFilterLayoutObservationForTest {
        realizes_both_working_formats,
        binds_exact_filter_source,
        binds_exact_nearest_sampler,
        binds_spatial_and_read_only_operations,
        targets_only_the_working_format,
        contains_no_dummy_binding: facts.len() == 2
            && binds_exact_filter_source
            && binds_exact_nearest_sampler
            && binds_spatial_and_read_only_operations,
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C09CompositeLayoutObservationForTest {
    pub(crate) realizes_all_eight_entry_interfaces: bool,
    pub(crate) normal_omits_parent: bool,
    pub(crate) destination_binds_parent: bool,
    pub(crate) optional_clip_is_exact: bool,
    pub(crate) optional_mask_is_exact: bool,
    pub(crate) binds_only_one_source_sampler: bool,
    pub(crate) binds_only_exact_uniforms: bool,
}

#[cfg(test)]
pub(crate) fn c09_composite_layout_observation_for_test(
    requests: &C09CompositeCacheRequestsForTest,
) -> C09CompositeLayoutObservationForTest {
    use super::shader::c09_composite_pass_key_facts_for_test;

    let facts = requests
        .passes()
        .iter()
        .filter_map(|keys| {
            c09_composite_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
        })
        .collect::<Vec<_>>();
    let realizes_all_eight_entry_interfaces = facts.len() == requests.passes().len()
        && facts.len() == 8
        && [
            ShaderCompositePathKey::Normal,
            ShaderCompositePathKey::DestinationSampling,
        ]
        .into_iter()
        .all(|path| {
            [(false, false), (true, false), (false, true), (true, true)]
                .into_iter()
                .all(|(has_clip, has_mask)| {
                    facts
                        .iter()
                        .filter(|facts| {
                            facts.path == path
                                && facts.has_clip_coverage == has_clip
                                && facts.has_alpha_mask == has_mask
                        })
                        .count()
                        == 1
                })
        });
    let normal_omits_parent = facts.iter().all(|facts| {
        facts.path != ShaderCompositePathKey::Normal
            || (!facts
                .sampled_roles
                .contains(&ShaderBindingRoleKey::CompositeParent)
                && facts.uses_fixed_source_over_blend
                && !facts.uses_replace_blend)
    });
    let destination_binds_parent = facts.iter().all(|facts| {
        facts.path != ShaderCompositePathKey::DestinationSampling
            || (facts
                .sampled_roles
                .iter()
                .filter(|role| **role == ShaderBindingRoleKey::CompositeParent)
                .count()
                == 1
                && facts.uses_replace_blend
                && !facts.uses_fixed_source_over_blend)
    });
    let optional_clip_is_exact = facts.iter().all(|facts| {
        facts
            .sampled_roles
            .iter()
            .filter(|role| **role == ShaderBindingRoleKey::ClipCoverage)
            .count()
            == usize::from(facts.has_clip_coverage)
    });
    let optional_mask_is_exact = facts.iter().all(|facts| {
        facts
            .sampled_roles
            .iter()
            .filter(|role| **role == ShaderBindingRoleKey::AlphaMask)
            .count()
            == usize::from(facts.has_alpha_mask)
    });
    let binds_only_one_source_sampler = facts.iter().all(|facts| {
        facts.has_only_source_sampler
            && facts
                .sampled_roles
                .iter()
                .filter(|role| **role == ShaderBindingRoleKey::CompositeSource)
                .count()
                == 1
    });
    let binds_only_exact_uniforms = facts.iter().all(|facts| {
        facts.has_exact_uniforms
            && facts.working_format == facts.target_format
            && facts.sampled_roles.len()
                == 1 + usize::from(facts.path == ShaderCompositePathKey::DestinationSampling)
                    + usize::from(facts.has_clip_coverage)
                    + usize::from(facts.has_alpha_mask)
    });
    C09CompositeLayoutObservationForTest {
        realizes_all_eight_entry_interfaces,
        normal_omits_parent,
        destination_binds_parent,
        optional_clip_is_exact,
        optional_mask_is_exact,
        binds_only_one_source_sampler,
        binds_only_exact_uniforms,
    }
}

#[cfg(test)]
pub(crate) fn c08_pass_cache_requests_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
    output_format: Format,
) -> Result<C08PassCacheRequestsForTest> {
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        output_format,
        &capabilities,
    )?;
    let preparable = C08PreparableGraph::try_from_lowered(lowered).map_err(|_| {
        lowering_error("the checked C08 cache fixture did not retain exact executable keys")
    })?;
    let passes = preparable
        .lowered
        .passes
        .iter()
        .filter_map(|pass| pass.cache_keys.clone())
        .collect::<Vec<_>>();
    if passes.is_empty() {
        return Err(lowering_error(
            "the checked C08 cache fixture contains no custom pass keys",
        ));
    }
    Ok(C08PassCacheRequestsForTest { passes })
}

#[cfg(test)]
pub(crate) fn c08_pass_layout_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C08PassLayoutObservationForTest {
    use super::shader::{C08ProgramForTest, c08_pass_key_facts_for_test};

    let mut canonicalize_binds_capture_and_spatial_only = true;
    let mut span_source_over_binds_source_and_spatial_only = true;
    let mut present_binds_final_image_and_spatial_only = true;
    let mut copy_only_parent_is_not_sampled = true;
    let mut dummy_parameters_are_not_bound = true;
    let mut output_specializations = Vec::new();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let Ok(requests) = c08_pass_cache_requests_for_test(
                commands.clone(),
                context,
                capabilities,
                working_format,
                output_format,
            ) else {
                return C08PassLayoutObservationForTest::default();
            };
            let facts = requests
                .passes()
                .iter()
                .filter_map(|keys| {
                    c08_pass_key_facts_for_test(
                        keys.samplers(),
                        keys.layout(),
                        keys.shader(),
                        keys.pipeline(),
                    )
                })
                .collect::<Vec<_>>();
            if facts.len() != requests.passes().len() {
                return C08PassLayoutObservationForTest::default();
            }
            let canonicalize = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::CanonicalizeCapture);
            let source_over = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::SpanSourceOver);
            let present = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::Present);
            canonicalize_binds_capture_and_spatial_only &= canonicalize.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::CaptureSource
                    && facts.source_format == ShaderTextureFormatKey::VelloCaptureRgba8Unorm
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format.is_none()
                    && facts.target_format == ShaderTextureFormatKey::working(working_format)
                    && facts.has_only_spatial_uniform
            });
            span_source_over_binds_source_and_spatial_only &= source_over.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::CompositeSource
                    && facts.source_format == ShaderTextureFormatKey::working(working_format)
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format.is_none()
                    && facts.target_format == ShaderTextureFormatKey::working(working_format)
                    && facts.has_only_spatial_uniform
                    && facts.has_fixed_source_over_blend
            });
            copy_only_parent_is_not_sampled &= facts
                .iter()
                .all(|facts| facts.source_role != ShaderBindingRoleKey::CompositeParent);
            dummy_parameters_are_not_bound &=
                facts.iter().all(|facts| facts.has_only_spatial_uniform);
            present_binds_final_image_and_spatial_only &= present.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::FinalWorkingImage
                    && facts.source_format == ShaderTextureFormatKey::working(working_format)
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format == Some(ShaderTextureFormatKey::output(output_format))
                    && facts.target_format == ShaderTextureFormatKey::output(output_format)
                    && facts.has_only_spatial_uniform
            });
            if let Some(facts) = present {
                output_specializations.push((facts.working_format, facts.output_format));
            }
        }
    }
    let output_specialization_is_exact =
        c08_output_specializations_are_exact(&output_specializations);
    let c09_typed_vocabulary_is_preserved = c09_typed_vocabulary_is_preserved_for_test();

    C08PassLayoutObservationForTest {
        canonicalize_binds_capture_and_spatial_only,
        span_source_over_binds_source_and_spatial_only,
        present_binds_final_image_and_spatial_only,
        copy_only_parent_is_not_sampled,
        dummy_parameters_are_not_bound,
        c09_typed_vocabulary_is_preserved,
        output_specialization_is_exact,
    }
}

#[cfg(test)]
fn c08_output_specializations_are_exact(
    output_specializations: &[(ShaderTextureFormatKey, Option<ShaderTextureFormatKey>)],
) -> bool {
    output_specializations.len() == 4
        && output_specializations.iter().all(|specialization| {
            output_specializations
                .iter()
                .filter(|candidate| *candidate == specialization)
                .count()
                == 1
        })
}

#[cfg(test)]
fn c09_typed_vocabulary_is_preserved_for_test() -> bool {
    matches!(
        ShaderBindingRoleKey::CompositeParent,
        ShaderBindingRoleKey::CompositeParent
    ) && matches!(
        ShaderDataBindingKey::CompositeParameters,
        ShaderDataBindingKey::CompositeParameters
    ) && matches!(
        ShaderDataBindingKey::PresentParameters,
        ShaderDataBindingKey::PresentParameters
    )
}

#[cfg(test)]
pub(crate) fn c08_executable_subset_observation_for_test(
    c08_commands: RenderCommands,
    expanded_graph_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C08ExecutableSubsetObservationForTest {
    c08_executable_subset_observation(c08_commands, expanded_graph_commands, context, capabilities)
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c08_zero_capture_spine_lowered_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    policy: EffectQualityPolicy,
) -> Result<LoweredGraphPlan> {
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let working_format = capabilities.resolve_effect_working_format(policy)?;
    let mut lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let clear = lowered
        .passes
        .first()
        .cloned()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no root clear"))?;
    let mut present =
        lowered.passes.last().cloned().ok_or_else(|| {
            lowering_error("the C08 zero-capture fixture has no terminal present")
        })?;
    let mut root = lowered
        .resources
        .iter()
        .find(|resource| resource.id == lowered.root_working_image)
        .cloned()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no root resource"))?;

    root.expected_reads = 1;
    root.last_use = present.id;
    present.dependencies = vec![clear.id];
    let final_read = present
        .reads
        .first_mut()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no final read"))?;
    final_read.resource = root.id;
    present.releases = vec![root.id];
    lowered.resources = vec![root];
    lowered.passes = vec![clear, present];

    Ok(lowered)
}

#[cfg(test)]
pub(crate) fn c08_two_capture_spine_lowered_for_test(
    commands: RenderCommands,
    donor_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    policy: EffectQualityPolicy,
) -> Result<LoweredGraphPlan> {
    let working_format = capabilities.resolve_effect_working_format(policy)?;
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let mut lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let FramePlan::GpuGraph(donor_graph) = donor_commands.plan_for(context)? else {
        return Err(lowering_error(
            "the two-capture C08 fixture requires a graph-shaped identity donor",
        ));
    };
    let donor = LoweredGraphPlan::try_lower_validated_graph(
        &donor_graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;

    let C08DonorIdentitiesForTest {
        second_capture_pass,
        second_canonicalize_pass,
        second_composite_pass,
        second_capture_target,
        second_canonical_target,
        second_composite_target,
    } = c08_donor_identities_for_test(&lowered, &donor)?;

    if lowered.passes.len() != 5 || lowered.resources.len() != 4 {
        return Err(lowering_error(
            "the two-capture C08 fixture source is not the exact one-capture spine",
        ));
    }
    let first_capture = lowered.passes[1].clone();
    let first_canonicalize = lowered.passes[2].clone();
    let first_composite = lowered.passes[3].clone();
    let mut present = lowered
        .passes
        .pop()
        .ok_or_else(|| lowering_error("the two-capture C08 fixture has no terminal present"))?;
    let RuntimeResultBinding::Resource(first_capture_target) = first_capture.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no capture target",
        ));
    };
    let RuntimeResultBinding::Resource(first_canonical_target) = first_canonicalize.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no canonical target",
        ));
    };
    let RuntimeResultBinding::Resource(first_composite_target) = first_composite.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no composite target",
        ));
    };

    let second_resources = c08_second_capture_resources_for_test(
        &mut lowered,
        [
            first_capture_target,
            first_canonical_target,
            first_composite_target,
        ],
        &C08DonorIdentitiesForTest {
            second_capture_pass,
            second_canonicalize_pass,
            second_composite_pass,
            second_capture_target,
            second_canonical_target,
            second_composite_target,
        },
        present.id,
    )?;

    let mut second_capture = first_capture;
    second_capture.id = second_capture_pass;
    second_capture.result = RuntimeResultBinding::Resource(second_capture_target);

    let mut second_canonicalize = first_canonicalize;
    second_canonicalize.id = second_canonicalize_pass;
    second_canonicalize.dependencies = vec![second_capture_pass];
    second_canonicalize.reads[0].resource = second_capture_target;
    second_canonicalize.result = RuntimeResultBinding::Resource(second_canonical_target);
    second_canonicalize.releases = vec![second_capture_target];

    let mut second_composite = first_composite;
    second_composite.id = second_composite_pass;
    second_composite.dependencies = vec![lowered.passes[3].id, second_canonicalize_pass];
    second_composite.reads[0].resource = first_composite_target;
    second_composite.reads[1].resource = second_canonical_target;
    second_composite.result = RuntimeResultBinding::Resource(second_composite_target);
    second_composite.releases = vec![first_composite_target, second_canonical_target];

    present.dependencies = vec![second_composite_pass];
    present.reads[0].resource = second_composite_target;
    present.releases = vec![second_composite_target];
    lowered.resources.extend(second_resources);
    lowered.passes.extend([
        second_capture,
        second_canonicalize,
        second_composite,
        present,
    ]);

    validate_two_capture_fixture(lowered)
}

#[cfg(test)]
fn validate_two_capture_fixture(lowered: LoweredGraphPlan) -> Result<LoweredGraphPlan> {
    if lowered.c08_execution_facts().is_none() {
        return Err(lowering_error(
            "the two-capture C08 fixture did not preserve the validated executable subset",
        ));
    }
    Ok(lowered)
}

#[cfg(test)]
struct C08DonorIdentitiesForTest {
    second_capture_pass: RuntimePassId,
    second_canonicalize_pass: RuntimePassId,
    second_composite_pass: RuntimePassId,
    second_capture_target: RuntimeResourceId,
    second_canonical_target: RuntimeResourceId,
    second_composite_target: RuntimeResourceId,
}

#[cfg(test)]
fn c08_second_capture_resources_for_test(
    lowered: &mut LoweredGraphPlan,
    first_targets: [RuntimeResourceId; 3],
    second: &C08DonorIdentitiesForTest,
    present: RuntimePassId,
) -> Result<[RuntimeResourceRequest; 3]> {
    let clone_resource = |id, missing| {
        lowered
            .resources
            .iter()
            .find(|resource| resource.id == id)
            .cloned()
            .ok_or_else(|| lowering_error(missing))
    };
    let mut capture = clone_resource(
        first_targets[0],
        "the two-capture C08 fixture lost its capture resource",
    )?;
    capture.id = second.second_capture_target;
    capture.producer = RuntimeResourceProducer::Pass(second.second_capture_pass);
    capture.last_use = second.second_canonicalize_pass;
    let mut canonical = clone_resource(
        first_targets[1],
        "the two-capture C08 fixture lost its canonical resource",
    )?;
    canonical.id = second.second_canonical_target;
    canonical.producer = RuntimeResourceProducer::Pass(second.second_canonicalize_pass);
    canonical.last_use = second.second_composite_pass;
    let mut composite = clone_resource(
        first_targets[2],
        "the two-capture C08 fixture lost its composite resource",
    )?;
    composite.id = second.second_composite_target;
    composite.producer = RuntimeResourceProducer::Pass(second.second_composite_pass);
    composite.last_use = present;
    lowered
        .resources
        .iter_mut()
        .find(|resource| resource.id == first_targets[2])
        .ok_or_else(|| lowering_error("the two-capture C08 fixture lost its first result"))?
        .last_use = second.second_composite_pass;
    Ok([capture, canonical, composite])
}

#[cfg(test)]
fn c08_donor_identities_for_test(
    lowered: &LoweredGraphPlan,
    donor: &LoweredGraphPlan,
) -> Result<C08DonorIdentitiesForTest> {
    let existing_passes = lowered
        .passes
        .iter()
        .map(|pass| pass.id)
        .collect::<BTreeSet<_>>();
    let mut donor_passes = donor
        .passes
        .iter()
        .map(|pass| pass.id)
        .filter(|pass| !existing_passes.contains(pass));
    let second_capture_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare capture pass identity")
    })?;
    let second_canonicalize_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare canonical pass identity")
    })?;
    let second_composite_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare composite pass identity")
    })?;
    let existing_resources = lowered
        .resources
        .iter()
        .map(|resource| resource.id)
        .collect::<BTreeSet<_>>();
    let mut donor_resources = donor
        .resources
        .iter()
        .map(|resource| resource.id)
        .filter(|resource| !existing_resources.contains(resource));
    Ok(C08DonorIdentitiesForTest {
        second_capture_pass,
        second_canonicalize_pass,
        second_composite_pass,
        second_capture_target: donor_resources.next().ok_or_else(|| {
            lowering_error("the two-capture C08 fixture has no spare capture resource identity")
        })?,
        second_canonical_target: donor_resources.next().ok_or_else(|| {
            lowering_error("the two-capture C08 fixture has no spare canonical resource identity")
        })?,
        second_composite_target: donor_resources.next().ok_or_else(|| {
            lowering_error("the two-capture C08 fixture has no spare composite resource identity")
        })?,
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BoundedCaptureTransformObservationForTest {
    pub(crate) preserves_application_order_formula: bool,
    pub(crate) preserves_signed_texel_center_mapping: bool,
    pub(crate) covers_required_raster_scales: bool,
    pub(crate) preserves_capture_execution_facts: bool,
    pub(crate) lowers_scene_with_explicit_initial_transform: bool,
}

#[cfg(test)]
pub(crate) fn bounded_capture_transform_observation_for_test(
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
) -> BoundedCaptureTransformObservationForTest {
    bounded_capture_transform_observation(
        commands,
        capture_transform,
        parent_to_surface,
        antialiasing,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn pass_spatial_uniform_bytes_for_test(
    source_origin: Point,
    source_raster_scale: f64,
    source_extent: PhysicalSize,
    destination_origin: Point,
    destination_raster_scale: f64,
    destination_extent: PhysicalSize,
) -> Result<[u8; 48]> {
    let source = RuntimeSpatialDescriptor {
        logical_bounds: Rect::new(0.0, 0.0, 1.0, 1.0),
        device_origin: (0, 0),
        device_extent: source_extent,
        texel_origin: source_origin,
        raster_scale: source_raster_scale,
    };
    let destination = RuntimeSpatialDescriptor {
        logical_bounds: Rect::new(0.0, 0.0, 1.0, 1.0),
        device_origin: (0, 0),
        device_extent: destination_extent,
        texel_origin: destination_origin,
        raster_scale: destination_raster_scale,
    };
    super::shader::PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
        source,
        destination,
    )
    .map(super::shader::PassSpatialUniformBytes::into_bytes_for_test)
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaskUploadAllocationObservationForTest {
    pub(crate) allocation_extents: Vec<PhysicalSize>,
    pub(crate) retained_upload_count: usize,
}

#[cfg(test)]
pub(crate) fn mask_upload_allocation_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> MaskUploadAllocationObservationForTest {
    let Some(lowered) = lowered_c09_mask_plan_for_test(commands, context) else {
        return MaskUploadAllocationObservationForTest::default();
    };
    let allocation_extents = lowered
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.import,
                Some(RuntimeResourceImport::ResolvedAlphaMask(_))
            )
        })
        .map(|resource| resource.spatial.device_extent)
        .collect::<Vec<_>>();
    MaskUploadAllocationObservationForTest {
        retained_upload_count: allocation_extents.len(),
        allocation_extents,
    }
}

#[cfg(test)]
pub(crate) fn composite_parameter_bytes_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> Option<[u8; 112]> {
    let plan = lowered_c09_mask_plan_for_test(commands, context)?;
    plan.passes.iter().find_map(|pass| {
        let RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) = &pass.kind
        else {
            return None;
        };
        parameters
            .alpha_mask()
            .and_then(|_| CompositeParameterBytes::try_from_runtime_layer(parameters).ok())
            .map(CompositeParameterBytes::into_bytes_for_test)
    })
}

#[cfg(test)]
pub(crate) fn mask_pipeline_keys_exclude_image_identity_for_test(
    first: RenderCommands,
    second: RenderCommands,
    context: FrameContext,
) -> bool {
    fn first_mask_keys(plan: &LoweredGraphPlan) -> Option<&RuntimePassCacheKeys> {
        plan.passes.iter().find_map(|pass| match &pass.kind {
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) if parameters.alpha_mask().is_some() => pass.cache_keys.as_ref(),
            _ => None,
        })
    }

    let Some(first) = lowered_c09_mask_plan_for_test(first, context) else {
        return false;
    };
    let Some(second) = lowered_c09_mask_plan_for_test(second, context) else {
        return false;
    };
    first_mask_keys(&first) == first_mask_keys(&second)
}

#[cfg(test)]
fn lowered_c09_mask_plan_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> Option<LoweredGraphPlan> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeGraphGeneration(GraphLoweringGeneration);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeResourceId(GraphLoweringResourceId);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimePassId(GraphLoweringPassId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    ClipCoverage,
    IsolationWorkingImage,
    ImportedImage,
    BackdropCopy,
    FilterIntermediate,
    ShadowImage,
    CompositeResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceFormat {
    VelloCaptureRgba8Unorm,
    ClipCoverageRgba8Unorm,
    Working(WorkingFormat),
    ResolvedMaskRgba8Unorm,
}

impl RuntimeResourceFormat {
    const fn shader_key(self) -> ShaderTextureFormatKey {
        match self {
            Self::VelloCaptureRgba8Unorm => ShaderTextureFormatKey::VelloCaptureRgba8Unorm,
            Self::ClipCoverageRgba8Unorm => ShaderTextureFormatKey::ClipCoverageRgba8Unorm,
            Self::Working(format) => ShaderTextureFormatKey::working(format),
            Self::ResolvedMaskRgba8Unorm => ShaderTextureFormatKey::ResolvedMaskRgba8Unorm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeSpatialDescriptor {
    logical_bounds: Rect,
    device_origin: (i32, i32),
    device_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl RuntimeSpatialDescriptor {
    fn from_graph(spatial: GraphLoweringSpatialDescriptor) -> Self {
        Self {
            logical_bounds: spatial.logical_bounds(),
            device_origin: spatial.device_origin(),
            device_extent: spatial.device_extent(),
            texel_origin: spatial.texel_origin(),
            raster_scale: spatial.raster_scale(),
        }
    }

    #[must_use]
    pub(crate) const fn device_extent(self) -> PhysicalSize {
        self.device_extent
    }

    #[must_use]
    pub(crate) const fn texel_origin(self) -> Point {
        self.texel_origin
    }

    #[must_use]
    pub(crate) const fn raster_scale(self) -> f64 {
        self.raster_scale
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceProducer {
    Imported,
    Pass(RuntimePassId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceImport {
    ResolvedAlphaMask(ResolvedMaskUploadDescriptor),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeResourceRequest {
    id: RuntimeResourceId,
    role: RuntimeResourceRole,
    format: RuntimeResourceFormat,
    spatial: RuntimeSpatialDescriptor,
    producer: RuntimeResourceProducer,
    expected_reads: u32,
    last_use: RuntimePassId,
    import: Option<RuntimeResourceImport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeInitialization {
    SurfaceBaseColor,
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeVelloSpan {
    scope: RuntimeVelloSpanScope,
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
    captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverageElement {
    clip: RenderClip,
    transform: Transform,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverage {
    elements: Vec<RuntimeClipCoverageElement>,
    antialiasing: Antialiasing,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeVelloCapture {
    Span(RuntimeVelloSpan),
    ClipCoverage(RuntimeClipCoverage),
}

impl RuntimeVelloCapture {
    const fn antialiasing(&self) -> Antialiasing {
        match self {
            Self::Span(span) => span.antialiasing,
            Self::ClipCoverage(coverage) => coverage.antialiasing,
        }
    }

    fn span(&self) -> Option<&RuntimeVelloSpan> {
        match self {
            Self::Span(span) => Some(span),
            Self::ClipCoverage(_) => None,
        }
    }

    fn clip_coverage(&self) -> Option<&RuntimeClipCoverage> {
        match self {
            Self::Span(_) => None,
            Self::ClipCoverage(coverage) => Some(coverage),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeColorClampBoundary {
    ClampStraightRgbaToUnitThenPremultiply,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeColorOperationKind {
    Brightness(RuntimeFilterAmount),
    Contrast(RuntimeFilterAmount),
    Grayscale(RuntimeUnitFilterAmount),
    HueRotate(RuntimeFilterAngle),
    Invert(RuntimeUnitFilterAmount),
    Opacity(RuntimeUnitFilterAmount),
    Saturate(RuntimeFilterAmount),
    Sepia(RuntimeUnitFilterAmount),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeColorOperation {
    operation: RuntimeColorOperationKind,
    clamp_boundary: RuntimeColorClampBoundary,
}

impl RuntimeColorOperation {
    #[must_use]
    pub(crate) const fn operation(self) -> RuntimeColorOperationKind {
        self.operation
    }

    #[must_use]
    pub(crate) const fn clamp_boundary(self) -> RuntimeColorClampBoundary {
        self.clamp_boundary
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeSamplingEdge {
    ClampToExtent,
    TransparentBlack,
    SemanticBorderMirror(Rect),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeFilterSpatialMapping {
    source: RuntimeSpatialDescriptor,
    result: RuntimeSpatialDescriptor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeColorFilter {
    operations: Vec<RuntimeColorOperation>,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
}

impl RuntimeColorFilter {
    #[must_use]
    pub(crate) fn operations(&self) -> &[RuntimeColorOperation] {
        &self.operations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeBlur {
    axis: RuntimeBlurAxis,
    input: RuntimeBlurInput,
    standard_deviation: f64,
    support_radius: u32,
    kernel: GaussianKernelKey,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDropShadow {
    offset: Point,
    standard_deviation: f64,
    color: Color,
    support_radius: u32,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
    uses_source_alpha: bool,
    uses_continuous_offset: bool,
    retains_unchanged_source: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeOuterClip {
    clip: RenderClip,
    transform: Transform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDestinationToLayerLocal {
    affine: Transform,
}

impl RuntimeDestinationToLayerLocal {
    fn try_new(affine: Transform) -> Result<Self> {
        if !runtime_affine_is_finite_and_non_singular(affine) {
            return Err(Error::invalid_value(
                "destination-to-layer-local affine mapping",
                format!("{:?}", affine.as_array()),
                "must be finite and non-singular",
            ));
        }
        Ok(Self { affine })
    }

    #[must_use]
    pub(crate) const fn affine(self) -> Transform {
        self.affine
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeMaskTexelCenterFacts {
    half_texel_normalized: [f64; 2],
    texel_size_normalized: [f64; 2],
}

impl RuntimeMaskTexelCenterFacts {
    fn try_new(image_dimensions: PhysicalSize) -> Result<Self> {
        if image_dimensions.width() == 0 || image_dimensions.height() == 0 {
            return Err(Error::invalid_value(
                "composite mask image dimensions",
                format!("{}x{}", image_dimensions.width(), image_dimensions.height()),
                "must be positive before deriving texel-center facts",
            ));
        }
        let texel_size_normalized = [
            1.0 / f64::from(image_dimensions.width()),
            1.0 / f64::from(image_dimensions.height()),
        ];
        let half_texel_normalized = [
            texel_size_normalized[0] * 0.5,
            texel_size_normalized[1] * 0.5,
        ];
        if texel_size_normalized
            .into_iter()
            .chain(half_texel_normalized)
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(lowering_error(
                "composite mask texel-center facts must be finite and positive",
            ));
        }
        Ok(Self {
            half_texel_normalized,
            texel_size_normalized,
        })
    }

    #[must_use]
    pub(crate) const fn half_texel_normalized(self) -> [f64; 2] {
        self.half_texel_normalized
    }

    #[must_use]
    pub(crate) const fn texel_size_normalized(self) -> [f64; 2] {
        self.texel_size_normalized
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeResolvedAlphaMaskComposition {
    resource: RuntimeResourceId,
    bounds: Rect,
    image_dimensions: PhysicalSize,
    texel_center_facts: RuntimeMaskTexelCenterFacts,
    sampling: ShaderMaskSamplingKey,
}

impl RuntimeResolvedAlphaMaskComposition {
    fn try_new(
        resource: RuntimeResourceId,
        bounds: Rect,
        image_dimensions: PhysicalSize,
        sampling: ShaderMaskSamplingKey,
    ) -> Result<Self> {
        let maximum_x = bounds.x() + bounds.width();
        let maximum_y = bounds.y() + bounds.height();
        if !bounds.x().is_finite()
            || !bounds.y().is_finite()
            || !bounds.width().is_finite()
            || !bounds.height().is_finite()
            || bounds.width() <= 0.0
            || bounds.height() <= 0.0
            || !maximum_x.is_finite()
            || !maximum_y.is_finite()
        {
            return Err(Error::invalid_value(
                "composite mask semantic bounds",
                format!(
                    "({}, {}, {}, {})",
                    bounds.x(),
                    bounds.y(),
                    bounds.width(),
                    bounds.height()
                ),
                "must be a finite positive rectangle with a finite maximum",
            ));
        }
        Ok(Self {
            resource,
            bounds,
            image_dimensions,
            texel_center_facts: RuntimeMaskTexelCenterFacts::try_new(image_dimensions)?,
            sampling,
        })
    }

    #[must_use]
    pub(crate) const fn resource(self) -> RuntimeResourceId {
        self.resource
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> Rect {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn image_dimensions(self) -> PhysicalSize {
        self.image_dimensions
    }

    #[must_use]
    pub(crate) const fn texel_center_facts(self) -> RuntimeMaskTexelCenterFacts {
        self.texel_center_facts
    }

    #[must_use]
    pub(crate) const fn sampling(self) -> ShaderMaskSamplingKey {
        self.sampling
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeLayerCompositeParameters {
    destination_to_layer_local: RuntimeDestinationToLayerLocal,
    opacity: f32,
    blend: BlendMode,
    has_clip: bool,
    alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
}

impl RuntimeLayerCompositeParameters {
    fn try_new(
        destination_to_layer_local: Transform,
        opacity: f32,
        blend: BlendMode,
        has_clip: bool,
        alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
    ) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "composite opacity",
                opacity,
                "must be finite before clamping",
            ));
        }
        Ok(Self {
            destination_to_layer_local: RuntimeDestinationToLayerLocal::try_new(
                destination_to_layer_local,
            )?,
            opacity: opacity.clamp(0.0, 1.0),
            blend,
            has_clip,
            alpha_mask,
        })
    }

    #[must_use]
    pub(crate) const fn destination_to_layer_local(self) -> RuntimeDestinationToLayerLocal {
        self.destination_to_layer_local
    }

    #[must_use]
    pub(crate) const fn opacity(self) -> f32 {
        self.opacity
    }

    #[must_use]
    pub(crate) const fn blend(self) -> BlendMode {
        self.blend
    }

    #[must_use]
    pub(crate) const fn has_clip(self) -> bool {
        self.has_clip
    }

    #[must_use]
    pub(crate) const fn alpha_mask(self) -> Option<RuntimeResolvedAlphaMaskComposition> {
        self.alpha_mask
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        parameters: Box<RuntimeLayerCompositeParameters>,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<RuntimeOuterClip>,
        clip_coverage: Option<RuntimeResourceId>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeComposite {
    kind: RuntimeCompositeKind,
    source_captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimePassKind {
    ClearRoot {
        initialization: RuntimeInitialization,
        color: Color,
    },
    VelloCapture(Option<RuntimeVelloCapture>),
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter(Option<RuntimeColorFilter>),
    BlurHorizontal(Option<RuntimeBlur>),
    BlurVertical(Option<RuntimeBlur>),
    DropShadowColorize(Option<RuntimeDropShadow>),
    Composite(Option<RuntimeComposite>),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeReadRole {
    CaptureSource,
    CompletedParent,
    FilterSource,
    BlurredSourceAlpha,
    CompositeParent,
    CompositeSource,
    ClipCoverage,
    AlphaMask,
    Shadow,
    FinalWorkingImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeSamplingFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeReadBinding {
    role: RuntimeReadRole,
    resource: RuntimeResourceId,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
    sampler_key: SamplerKey,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these exact immutable runtime read-binding facts"
    )
)]
impl RuntimeReadBinding {
    pub(crate) const fn role(&self) -> RuntimeReadRole {
        self.role
    }

    pub(crate) const fn resource(&self) -> RuntimeResourceId {
        self.resource
    }

    pub(crate) const fn sampling_filter(&self) -> RuntimeSamplingFilter {
        self.sampling_filter
    }

    pub(crate) const fn sampling_edge(&self) -> RuntimeSamplingEdge {
        self.sampling_edge
    }

    pub(crate) const fn sampler_key(&self) -> SamplerKey {
        self.sampler_key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResultBinding {
    Empty,
    Resource(RuntimeResourceId),
    Output(Format),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimePassCacheKeys {
    samplers: Vec<SamplerKey>,
    layout: BindGroupLayoutKey,
    shader: ShaderModuleKey,
    pipeline: RenderPipelineKey,
}

impl RuntimePassCacheKeys {
    pub(crate) fn samplers(&self) -> &[SamplerKey] {
        &self.samplers
    }

    pub(crate) const fn layout(&self) -> &BindGroupLayoutKey {
        &self.layout
    }

    pub(crate) const fn shader(&self) -> &ShaderModuleKey {
        &self.shader
    }

    pub(crate) const fn pipeline(&self) -> &RenderPipelineKey {
        &self.pipeline
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimePass {
    id: RuntimePassId,
    kind: RuntimePassKind,
    dependencies: Vec<RuntimePassId>,
    reads: Vec<RuntimeReadBinding>,
    result: RuntimeResultBinding,
    releases: Vec<RuntimeResourceId>,
    cache_keys: Option<RuntimePassCacheKeys>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredGraphPlan {
    generation: RuntimeGraphGeneration,
    working_format: WorkingFormat,
    output_format: Format,
    resources: Vec<RuntimeResourceRequest>,
    passes: Vec<RuntimePass>,
    root_working_image: RuntimeResourceId,
    final_present: RuntimePassId,
}

#[derive(Clone)]
struct ExecutableLayerCompositionFacts {
    pass: RuntimePassId,
    parent: RuntimeResourceId,
    source: RuntimeResourceId,
    clip_coverage: Option<RuntimeResourceId>,
    alpha_mask: Option<RuntimeResourceId>,
    result: RuntimeResourceId,
    composite: RuntimeComposite,
}

#[derive(Clone)]
struct ExecutableColorFilterFacts {
    pass: RuntimePassId,
    source: RuntimeResourceId,
    result: RuntimeResourceId,
    filter: RuntimeColorFilter,
}

#[derive(Clone)]
struct ExecutableBlurFacts {
    horizontal: RuntimePassId,
    vertical: RuntimePassId,
    source: RuntimeResourceId,
    intermediate: RuntimeResourceId,
    result: RuntimeResourceId,
    blur: RuntimeBlur,
}

#[derive(Clone)]
struct ExecutableDropShadowFacts {
    horizontal: RuntimePassId,
    vertical: RuntimePassId,
    colorize: RuntimePassId,
    merge: RuntimePassId,
    source: RuntimeResourceId,
    horizontal_result: RuntimeResourceId,
    vertical_result: RuntimeResourceId,
    shadow: RuntimeResourceId,
    result: RuntimeResourceId,
    blur: RuntimeBlur,
    parameters: RuntimeDropShadow,
}

#[derive(Clone)]
struct ExecutableBackdropFacts {
    copy: RuntimePassId,
    completed_parent: RuntimeResourceId,
    copied: RuntimeResourceId,
    foreground: Option<RuntimeResourceId>,
    filter_steps: Vec<ExecutableFilterStepFacts>,
    filtered: RuntimeResourceId,
    group_clear: RuntimePassId,
    backdrop_composite: RuntimePassId,
    foreground_composite: Option<RuntimePassId>,
    outer_composite: RuntimePassId,
    completed_group: RuntimeResourceId,
    result: RuntimeResourceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutableFilterStepFacts {
    Color(RuntimePassId),
    Blur {
        horizontal: RuntimePassId,
        vertical: RuntimePassId,
    },
    DropShadow {
        horizontal: RuntimePassId,
        vertical: RuntimePassId,
        colorize: RuntimePassId,
        merge: RuntimePassId,
    },
}

#[derive(Clone)]
struct ClosedExecutableGraphFacts {
    working_format: WorkingFormat,
    output_format: Format,
    captures: Vec<ExecutableVelloCaptureFacts>,
    layer_compositions: Vec<ExecutableLayerCompositionFacts>,
    color_filters: Vec<ExecutableColorFilterFacts>,
    blurs: Vec<ExecutableBlurFacts>,
    drop_shadows: Vec<ExecutableDropShadowFacts>,
    filter_steps: Vec<ExecutableFilterStepFacts>,
    backdrops: Vec<ExecutableBackdropFacts>,
}

#[derive(Clone, Copy)]
struct ExecutableCompositionContext {
    current: RuntimeResourceId,
    producer: RuntimePassId,
    contains_captured_source: bool,
}

#[must_use = "a closed executable graph must reach dispatch or explicit rejection"]
struct ClosedExecutableGraph {
    lowered: LoweredGraphPlan,
    facts: ClosedExecutableGraphFacts,
}

impl ClosedExecutableGraph {
    fn try_from_lowered(lowered: LoweredGraphPlan) -> std::result::Result<Self, LoweredGraphPlan> {
        let Some(facts) = lowered.closed_executable_graph_facts() else {
            return Err(lowered);
        };
        if !facts.proves_exact_facts_for(&lowered) {
            return Err(lowered);
        }
        Ok(Self { lowered, facts })
    }

    #[cfg(test)]
    fn into_lowered(self) -> LoweredGraphPlan {
        self.lowered
    }

    fn has_layer_composition(&self) -> bool {
        !self.facts.layer_compositions.is_empty()
    }

    fn has_color_filters(&self) -> bool {
        !self.facts.color_filters.is_empty()
    }

    fn has_spatial_filters(&self) -> bool {
        !self.facts.blurs.is_empty() || !self.facts.drop_shadows.is_empty()
    }

    fn has_backdrops(&self) -> bool {
        !self.facts.backdrops.is_empty()
    }
}

impl ClosedExecutableGraphFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        if self.working_format != plan.working_format
            || self.output_format != plan.output_format
            || self.captures.is_empty()
        {
            return false;
        }
        let captures_are_exact = self.captures.iter().all(|capture| {
            plan.passes.iter().any(|pass| {
                pass.id == capture.pass()
                    && matches!(
                        &pass.kind,
                        RuntimePassKind::VelloCapture(Some(work))
                            if work == capture.work()
                                && work.antialiasing() == capture.antialiasing()
                    )
                    && pass.result == RuntimeResultBinding::Resource(capture.target())
            })
        });
        let layers_are_exact = self.layer_compositions.iter().all(|layer| {
            let Some(pass) = plan.passes.iter().find(|pass| pass.id == layer.pass) else {
                return false;
            };
            let RuntimePassKind::Composite(Some(composite)) = &pass.kind else {
                return false;
            };
            let mut expected_reads = vec![layer.parent, layer.source];
            if let Some(coverage) = layer.clip_coverage {
                expected_reads.push(coverage);
            }
            if let Some(mask) = layer.alpha_mask {
                expected_reads.push(mask);
            }
            composite == &layer.composite
                && pass
                    .reads
                    .iter()
                    .map(|read| read.resource)
                    .eq(expected_reads)
                && pass.result == RuntimeResultBinding::Resource(layer.result)
        });
        let color_filters_are_exact = self.color_filters.iter().all(|color| {
            plan.passes.iter().any(|pass| {
                pass.id == color.pass
                    && matches!(
                        &pass.kind,
                        RuntimePassKind::ColorFilter(Some(filter)) if filter == &color.filter
                    )
                    && pass.reads.len() == 1
                    && pass.reads[0].resource == color.source
                    && pass.result == RuntimeResultBinding::Resource(color.result)
            })
        }) && self.color_filters.len()
            == plan
                .passes
                .iter()
                .filter(|pass| matches!(pass.kind, RuntimePassKind::ColorFilter(Some(_))))
                .count();
        let blurs_are_exact = self
            .blurs
            .iter()
            .all(|blur| blur.proves_exact_facts_for(plan))
            && self.blurs.len()
                == plan
                    .passes
                    .iter()
                    .filter(|pass| {
                        matches!(
                            pass.kind,
                            RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                                input: RuntimeBlurInput::Rgba,
                                ..
                            }))
                        )
                    })
                    .count();
        let drop_shadows_are_exact = self
            .drop_shadows
            .iter()
            .all(|shadow| shadow.proves_exact_facts_for(plan))
            && self.drop_shadows.len()
                == plan
                    .passes
                    .iter()
                    .filter(|pass| {
                        matches!(pass.kind, RuntimePassKind::DropShadowColorize(Some(_)))
                    })
                    .count();
        let filter_order_is_exact =
            executable_filter_step_order(plan) == Some(self.filter_steps.clone());
        let backdrops_are_exact = self
            .backdrops
            .iter()
            .all(|backdrop| backdrop.proves_exact_facts_for(plan));
        captures_are_exact
            && layers_are_exact
            && color_filters_are_exact
            && blurs_are_exact
            && drop_shadows_are_exact
            && filter_order_is_exact
            && backdrops_are_exact
    }
}

impl ExecutableBackdropFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let pass = |id| plan.passes.iter().find(|candidate| candidate.id == id);
        let Some(copy) = pass(self.copy) else {
            return false;
        };
        let foreground_is_distinct = self
            .foreground
            .is_none_or(|foreground| foreground != self.copied && foreground != self.filtered);
        matches!(copy.kind, RuntimePassKind::CopyBackdrop)
            && copy.reads.len() == 1
            && copy.reads[0].resource == self.completed_parent
            && copy.result == RuntimeResultBinding::Resource(self.copied)
            && pass(self.group_clear).is_some_and(|clear| {
                matches!(
                    clear.kind,
                    RuntimePassKind::ClearRoot {
                        initialization: RuntimeInitialization::Transparent,
                        color,
                    } if color == Color::TRANSPARENT
                )
            })
            && pass(self.backdrop_composite).is_some()
            && self
                .foreground_composite
                .is_none_or(|id| pass(id).is_some())
            && pass(self.outer_composite).is_some()
            && self.completed_group != self.completed_parent
            && self.result != self.completed_group
            && foreground_is_distinct
    }
}

impl ExecutableBlurFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let Some(horizontal) = plan.passes.iter().find(|pass| pass.id == self.horizontal) else {
            return false;
        };
        let Some(vertical) = plan.passes.iter().find(|pass| pass.id == self.vertical) else {
            return false;
        };
        matches!(
            &horizontal.kind,
            RuntimePassKind::BlurHorizontal(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Horizontal)
        ) && matches!(
            &vertical.kind,
            RuntimePassKind::BlurVertical(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Vertical)
        ) && horizontal
            .reads
            .first()
            .is_some_and(|read| read.resource == self.source)
            && horizontal.result == RuntimeResultBinding::Resource(self.intermediate)
            && vertical
                .reads
                .first()
                .is_some_and(|read| read.resource == self.intermediate)
            && vertical.result == RuntimeResultBinding::Resource(self.result)
    }
}

impl ExecutableDropShadowFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let pass = |id| plan.passes.iter().find(|pass| pass.id == id);
        let (Some(horizontal), Some(vertical), Some(colorize), Some(merge)) = (
            pass(self.horizontal),
            pass(self.vertical),
            pass(self.colorize),
            pass(self.merge),
        ) else {
            return false;
        };
        matches!(
            &horizontal.kind,
            RuntimePassKind::BlurHorizontal(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Horizontal)
        ) && matches!(
            &vertical.kind,
            RuntimePassKind::BlurVertical(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Vertical)
        ) && matches!(
            &colorize.kind,
            RuntimePassKind::DropShadowColorize(Some(parameters))
                if parameters == &self.parameters
        ) && matches!(
            &merge.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::DropShadow,
                ..
            }))
        ) && horizontal
            .reads
            .first()
            .is_some_and(|read| read.resource == self.source)
            && horizontal.result == RuntimeResultBinding::Resource(self.horizontal_result)
            && vertical.result == RuntimeResultBinding::Resource(self.vertical_result)
            && colorize.result == RuntimeResultBinding::Resource(self.shadow)
            && merge.result == RuntimeResultBinding::Resource(self.result)
    }
}

fn runtime_blur_matches_axis(
    candidate: &RuntimeBlur,
    expected: &RuntimeBlur,
    axis: RuntimeBlurAxis,
) -> bool {
    let expected_spatial = match axis {
        RuntimeBlurAxis::Horizontal => expected.spatial,
        RuntimeBlurAxis::Vertical => RuntimeFilterSpatialMapping {
            source: expected.spatial.result,
            result: expected.spatial.result,
        },
    };
    candidate.axis == axis
        && candidate.input == expected.input
        && candidate.standard_deviation == expected.standard_deviation
        && candidate.support_radius == expected.support_radius
        && candidate.kernel == expected.kernel
        && candidate.spatial == expected_spatial
        && candidate.edge == expected.edge
}

fn executable_filter_step_order(plan: &LoweredGraphPlan) -> Option<Vec<ExecutableFilterStepFacts>> {
    let mut steps = Vec::new();
    let mut cursor = 0_usize;
    while cursor < plan.passes.len() {
        let pass = &plan.passes[cursor];
        match &pass.kind {
            RuntimePassKind::ColorFilter(Some(_)) => {
                steps.push(ExecutableFilterStepFacts::Color(pass.id));
                cursor = cursor.checked_add(1)?;
            }
            RuntimePassKind::BlurHorizontal(Some(blur)) if blur.input == RuntimeBlurInput::Rgba => {
                let vertical = plan.passes.get(cursor.checked_add(1)?)?;
                steps.push(ExecutableFilterStepFacts::Blur {
                    horizontal: pass.id,
                    vertical: vertical.id,
                });
                cursor = cursor.checked_add(2)?;
            }
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.input == RuntimeBlurInput::SourceAlpha =>
            {
                let vertical = plan.passes.get(cursor.checked_add(1)?)?;
                let colorize = plan.passes.get(cursor.checked_add(2)?)?;
                let merge = plan.passes.get(cursor.checked_add(3)?)?;
                steps.push(ExecutableFilterStepFacts::DropShadow {
                    horizontal: pass.id,
                    vertical: vertical.id,
                    colorize: colorize.id,
                    merge: merge.id,
                });
                cursor = cursor.checked_add(4)?;
            }
            _ => cursor = cursor.checked_add(1)?,
        }
    }
    Some(steps)
}

#[derive(Clone, Copy)]
enum GraphLoweringCapabilityValidation<'capabilities> {
    Required(&'capabilities DeviceCapabilities),
    ClassificationOnly,
}

#[must_use]
pub(crate) struct C08ExecutionFacts {
    working_format: WorkingFormat,
    output_format: Format,
    captures: Vec<ExecutableVelloCaptureFacts>,
}

impl C08ExecutionFacts {
    #[must_use]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.working_format
    }

    #[must_use]
    pub(crate) const fn output_format(&self) -> Format {
        self.output_format
    }

    #[must_use]
    pub(crate) fn captures(&self) -> &[ExecutableVelloCaptureFacts] {
        &self.captures
    }

    fn proves_exact_execution_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        if self.working_format() != plan.working_format
            || self.output_format() != plan.output_format
        {
            return false;
        }
        let mut passes = BTreeSet::new();
        let mut targets = BTreeSet::new();
        self.captures().iter().all(|capture| {
            let Some(pass) = plan.passes.iter().find(|pass| pass.id == capture.pass()) else {
                return false;
            };
            let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) = &pass.kind
            else {
                return false;
            };
            let Some(capture_span) = capture.span() else {
                return false;
            };
            let Some(target) = plan
                .resources
                .iter()
                .find(|resource| resource.id == capture.target())
            else {
                return false;
            };
            passes.insert(capture.pass())
                && targets.insert(capture.target())
                && capture_span.scope == RuntimeVelloSpanScope::CurrentParent
                && capture_span == span
                && capture
                    .initial_transform()
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite())
                && capture.antialiasing() == span.antialiasing
                && capture.target_extent() == target.spatial.device_extent
                && capture.texel_origin() == target.spatial.texel_origin
                && capture.raster_scale() == target.spatial.raster_scale
        })
    }
}

#[must_use]
pub(crate) struct C08PreparableGraph {
    lowered: LoweredGraphPlan,
    execution: C08ExecutionFacts,
}

impl C08PreparableGraph {
    #[cfg(test)]
    pub(crate) fn try_from_lowered(
        lowered: LoweredGraphPlan,
    ) -> std::result::Result<Self, LoweredGraphPlan> {
        let closed = ClosedExecutableGraph::try_from_lowered(lowered)?;
        Self::try_from_closed(closed).map_err(|closed| (*closed).into_lowered())
    }

    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if closed.has_layer_composition()
            || closed.has_color_filters()
            || closed.has_spatial_filters()
            || closed.has_backdrops()
            || closed.facts.captures.is_empty()
            || closed.facts.captures.iter().any(|capture| {
                capture
                    .span()
                    .is_none_or(|span| span.scope != RuntimeVelloSpanScope::CurrentParent)
            })
        {
            return Err(Box::new(closed));
        }
        let execution = C08ExecutionFacts {
            working_format: closed.facts.working_format,
            output_format: closed.facts.output_format,
            captures: closed.facts.captures.clone(),
        };
        if !execution.proves_exact_execution_facts_for(&closed.lowered) {
            return Err(Box::new(closed));
        }
        Ok(Self {
            lowered: closed.lowered,
            execution,
        })
    }

    fn into_parts(self) -> (LoweredGraphPlan, C08ExecutionFacts) {
        (self.lowered, self.execution)
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.execution.working_format()
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.execution.output_format()
    }

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the C08 root output resource is missing"))
    }

    #[cfg(test)]
    pub(crate) fn capture_grids_for_test(&self) -> Vec<C08CaptureGridForTest> {
        self.execution
            .captures()
            .iter()
            .map(|capture| C08CaptureGridForTest {
                texel_origin: capture.texel_origin(),
                extent: capture.target_extent(),
                raster_scale: capture.raster_scale(),
            })
            .collect()
    }
}

#[must_use]
pub(crate) struct C09PreparableGraph {
    closed: ClosedExecutableGraph,
}

impl C09PreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_layer_composition()
            || closed.has_color_filters()
            || closed.has_spatial_filters()
            || closed.has_backdrops()
        {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }
}

#[must_use]
pub(crate) struct C10PreparableGraph {
    closed: ClosedExecutableGraph,
}

impl C10PreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_color_filters() || closed.has_spatial_filters() || closed.has_backdrops() {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    fn proves_closed_color_facts(&self) -> bool {
        self.closed.has_color_filters()
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    #[cfg(test)]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    #[cfg(test)]
    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }

    #[cfg(test)]
    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.closed
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.closed.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the C10 root output resource is missing"))
    }

    #[cfg(test)]
    pub(crate) fn first_color_spatial_for_test(&self) -> Option<C10ColorSpatialObservationForTest> {
        self.closed
            .facts
            .color_filters
            .first()
            .map(|filter| c10_spatial_observation(filter.filter.spatial.source))
    }

    fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }

    #[cfg(test)]
    fn color_filters(&self) -> &[ExecutableColorFilterFacts] {
        &self.closed.facts.color_filters
    }
}

#[must_use]
pub(crate) struct C11PreparableGraph {
    closed: ClosedExecutableGraph,
}

impl C11PreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_spatial_filters() || closed.has_backdrops() {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    fn proves_closed_filter_facts(&self) -> bool {
        self.closed.has_spatial_filters()
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    #[cfg(test)]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    #[cfg(test)]
    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }

    #[cfg(test)]
    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.closed
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.closed.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the C11 root output resource is missing"))
    }

    #[cfg(test)]
    pub(crate) fn first_filter_spatial_for_test(
        &self,
    ) -> Option<(
        C10ColorSpatialObservationForTest,
        C10ColorSpatialObservationForTest,
    )> {
        self.closed
            .facts
            .blurs
            .first()
            .map(|blur| {
                (
                    c10_spatial_observation(blur.blur.spatial.source),
                    c10_spatial_observation(blur.blur.spatial.result),
                )
            })
            .or_else(|| {
                self.closed.facts.drop_shadows.first().map(|shadow| {
                    (
                        c10_spatial_observation(shadow.blur.spatial.source),
                        c10_spatial_observation(shadow.parameters.spatial.result),
                    )
                })
            })
    }

    fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }
}

#[must_use]
pub(crate) struct C12PreparableGraph {
    closed: ClosedExecutableGraph,
}

impl C12PreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        let [backdrop] = closed.facts.backdrops.as_slice() else {
            return Err(Box::new(closed));
        };
        if backdrop.filter_steps != closed.facts.filter_steps {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    fn proves_closed_backdrop_facts(&self) -> bool {
        let [backdrop] = self.closed.facts.backdrops.as_slice() else {
            return false;
        };
        backdrop.filter_steps == self.closed.facts.filter_steps
            && backdrop.proves_exact_facts_for(&self.closed.lowered)
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.closed
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.closed.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the C12 root output resource is missing"))
    }

    #[cfg(test)]
    pub(crate) fn backdrop_spatial_for_test(
        &self,
    ) -> Option<(
        C10ColorSpatialObservationForTest,
        C10ColorSpatialObservationForTest,
    )> {
        let [backdrop] = self.closed.facts.backdrops.as_slice() else {
            return None;
        };
        let spatial = |id| {
            self.closed
                .lowered
                .resources
                .iter()
                .find(|resource| resource.id == id)
                .map(|resource| c10_spatial_observation(resource.spatial))
        };
        Some((
            spatial(backdrop.completed_parent)?,
            spatial(backdrop.copied)?,
        ))
    }

    fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C08CaptureGridForTest {
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
}

#[derive(Clone)]
pub(crate) struct ExecutableVelloCaptureFacts {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: RuntimeVelloCapture,
    initial_transform: Transform,
    antialiasing: Antialiasing,
    target_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl ExecutableVelloCaptureFacts {
    #[must_use]
    pub(crate) const fn pass(&self) -> RuntimePassId {
        self.pass
    }

    #[must_use]
    pub(crate) const fn target(&self) -> RuntimeResourceId {
        self.target
    }

    #[must_use]
    fn span(&self) -> Option<&RuntimeVelloSpan> {
        self.work.span()
    }

    #[cfg(test)]
    fn commands(&self) -> Option<&RenderCommands> {
        self.span().map(|span| &span.commands)
    }

    #[must_use]
    const fn work(&self) -> &RuntimeVelloCapture {
        &self.work
    }

    #[must_use]
    pub(crate) const fn initial_transform(&self) -> Transform {
        self.initial_transform
    }

    #[must_use]
    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }

    #[must_use]
    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target_extent
    }

    #[must_use]
    pub(crate) const fn texel_origin(&self) -> Point {
        self.texel_origin
    }

    #[must_use]
    pub(crate) const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08PassClass {
    ClearRoot,
    VelloCapture,
    CanonicalizeCapture,
    SpanSourceOver,
    Present,
}

#[cfg(test)]
fn c08_pass_class(kind: &RuntimePassKind) -> Option<C08PassClass> {
    match kind {
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            ..
        } => Some(C08PassClass::ClearRoot),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            ..
        } => None,
        RuntimePassKind::VelloCapture(Some(_)) => Some(C08PassClass::VelloCapture),
        RuntimePassKind::VelloCapture(None) => None,
        RuntimePassKind::CanonicalizeCapture => Some(C08PassClass::CanonicalizeCapture),
        RuntimePassKind::CopyBackdrop
        | RuntimePassKind::ColorFilter(_)
        | RuntimePassKind::BlurHorizontal(_)
        | RuntimePassKind::BlurVertical(_)
        | RuntimePassKind::DropShadowColorize(_) => None,
        RuntimePassKind::Composite(Some(composite)) => match &composite.kind {
            RuntimeCompositeKind::SpanSourceOver
                if composite.source_captured_before_outer_semantics =>
            {
                Some(C08PassClass::SpanSourceOver)
            }
            RuntimeCompositeKind::SpanSourceOver
            | RuntimeCompositeKind::Layer { .. }
            | RuntimeCompositeKind::DropShadow => None,
        },
        RuntimePassKind::Composite(None) => None,
        RuntimePassKind::Present => Some(C08PassClass::Present),
    }
}

struct LoweredResourceSet {
    resources: Vec<RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
}

fn lower_runtime_resources(
    resource_views: &[GraphLoweringResourceView<'_>],
    pass_ids: &BTreeSet<RuntimePassId>,
    working_format: WorkingFormat,
    capability_validation: GraphLoweringCapabilityValidation<'_>,
) -> Result<LoweredResourceSet> {
    let mut resources = Vec::with_capacity(resource_views.len());
    let mut resource_ids = BTreeSet::new();
    let mut resource_formats = BTreeMap::new();
    for resource in resource_views {
        let resource = *resource;
        let id = RuntimeResourceId(resource.id());
        if !resource_ids.insert(id) {
            return Err(lowering_error("duplicate runtime resource binding"));
        }
        let role = runtime_resource_role(resource.role());
        let spatial = RuntimeSpatialDescriptor::from_graph(resource.spatial());
        let format = runtime_resource_format(role, working_format);
        if let GraphLoweringCapabilityValidation::Required(capabilities) = capability_validation {
            capabilities.validate_effect_texture_extent(spatial.device_extent)?;
            if let RuntimeResourceFormat::Working(format) = format {
                capabilities.validate_effect_texture_allocation(
                    spatial.device_extent,
                    Some(format),
                    format.texture_format(),
                    format.required_usages(),
                )?;
            }
        }
        let producer = match resource.producer() {
            GraphLoweringResourceProducer::Imported => RuntimeResourceProducer::Imported,
            GraphLoweringResourceProducer::Pass(pass) => {
                let pass = RuntimePassId(pass);
                if !pass_ids.contains(&pass) {
                    return Err(lowering_error("resource producer binding is missing"));
                }
                RuntimeResourceProducer::Pass(pass)
            }
        };
        let import = resource.import().map(|import| match import {
            GraphLoweringImportView::ResolvedAlphaMask(upload) => {
                RuntimeResourceImport::ResolvedAlphaMask(upload.clone())
            }
        });
        if matches!(producer, RuntimeResourceProducer::Imported) != import.is_some() {
            return Err(lowering_error(
                "imported runtime resource binding is inconsistent",
            ));
        }
        let last_use = RuntimePassId(resource.last_use());
        if !pass_ids.contains(&last_use) {
            return Err(lowering_error("resource last-use binding is missing"));
        }
        resource_formats.insert(id, format);
        resources.push(RuntimeResourceRequest {
            id,
            role,
            format,
            spatial,
            producer,
            expected_reads: resource.expected_reads(),
            last_use,
            import,
        });
    }
    Ok(LoweredResourceSet {
        resources,
        resource_formats,
    })
}

fn lower_runtime_passes(
    pass_views: &[GraphLoweringPassView<'_>],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    releases: &mut BTreeMap<RuntimePassId, Vec<RuntimeResourceId>>,
    working_format: WorkingFormat,
    output_format: Format,
) -> Result<Vec<RuntimePass>> {
    let mut seen_passes = BTreeSet::new();
    let mut passes = Vec::with_capacity(pass_views.len());
    for pass in pass_views {
        let pass = *pass;
        let id = RuntimePassId(pass.id());
        if seen_passes.contains(&id) {
            return Err(lowering_error("duplicate runtime pass binding"));
        }
        let dependencies = pass
            .dependencies()
            .into_iter()
            .map(RuntimePassId)
            .collect::<Vec<_>>();
        if dependencies
            .iter()
            .any(|dependency| !seen_passes.contains(dependency))
        {
            return Err(lowering_error(
                "runtime pass has a missing or forward dependency",
            ));
        }
        let kind = runtime_pass_kind(pass.kind()?, working_format)?;
        let reads = lower_read_bindings(&pass.reads()?, resource_by_id, resource_formats)?;
        let result =
            lower_runtime_result(pass.result(), &kind, &reads, resource_by_id, output_format)?;
        let pass_releases = releases.remove(&id).unwrap_or_default();
        if pass_releases
            .iter()
            .any(|resource| !reads.iter().any(|binding| binding.resource == *resource))
        {
            return Err(lowering_error(
                "runtime release is not the resource's last read",
            ));
        }
        let cache_keys = runtime_pass_cache_keys(
            &kind,
            &reads,
            result,
            working_format,
            output_format,
            resource_formats,
        )?;
        passes.push(RuntimePass {
            id,
            kind,
            dependencies,
            reads,
            result,
            releases: pass_releases,
            cache_keys,
        });
        seen_passes.insert(id);
    }
    Ok(passes)
}

fn lower_runtime_result(
    result: GraphLoweringPassResult,
    kind: &RuntimePassKind,
    reads: &[RuntimeReadBinding],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    output_format: Format,
) -> Result<RuntimeResultBinding> {
    match result {
        GraphLoweringPassResult::Empty if matches!(kind, RuntimePassKind::Present) => {
            Ok(RuntimeResultBinding::Output(output_format))
        }
        GraphLoweringPassResult::Empty => Ok(RuntimeResultBinding::Empty),
        GraphLoweringPassResult::Resource(resource) => {
            let resource = RuntimeResourceId(resource);
            if !resource_by_id.contains_key(&resource)
                || reads.iter().any(|read| read.resource == resource)
            {
                return Err(lowering_error(
                    "runtime pass result binding is inconsistent",
                ));
            }
            Ok(RuntimeResultBinding::Resource(resource))
        }
    }
}

struct ClosedGraphMaps<'plan> {
    resource_by_id: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    pass_positions: BTreeMap<RuntimePassId, usize>,
}

impl<'plan> ClosedGraphMaps<'plan> {
    fn try_new(plan: &'plan LoweredGraphPlan) -> Option<Self> {
        let resource_by_id = plan
            .resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();
        if resource_by_id.len() != plan.resources.len() {
            return None;
        }
        let resource_formats = plan
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let pass_positions = plan
            .passes
            .iter()
            .enumerate()
            .map(|(position, pass)| (pass.id, position))
            .collect::<BTreeMap<_, _>>();
        (pass_positions.len() == plan.passes.len()).then_some(Self {
            resource_by_id,
            resource_formats,
            pass_positions,
        })
    }
}

struct ClosedGraphAccounting {
    actual_reads: BTreeMap<RuntimeResourceId, u32>,
    actual_last_reads: BTreeMap<RuntimeResourceId, RuntimePassId>,
    releases: BTreeMap<RuntimeResourceId, RuntimePassId>,
    results: BTreeMap<RuntimeResourceId, RuntimePassId>,
}

fn validate_closed_graph_accounting(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
) -> Option<()> {
    let accounting = validate_closed_pass_accounting(plan, maps)?;
    for resource in &plan.resources {
        if resource.format != runtime_resource_format(resource.role, plan.working_format)
            || resource.expected_reads == 0
            || accounting.actual_reads.get(&resource.id).copied() != Some(resource.expected_reads)
            || accounting.actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
            || accounting.releases.get(&resource.id).copied() != Some(resource.last_use)
            || resource.spatial.device_extent.width() == 0
            || resource.spatial.device_extent.height() == 0
            || !resource.spatial.texel_origin.x().is_finite()
            || !resource.spatial.texel_origin.y().is_finite()
            || !resource.spatial.raster_scale.is_finite()
            || resource.spatial.raster_scale <= 0.0
        {
            return None;
        }
        match (&resource.producer, &resource.import) {
            (
                RuntimeResourceProducer::Imported,
                Some(RuntimeResourceImport::ResolvedAlphaMask(_)),
            ) if resource.role == RuntimeResourceRole::ImportedImage
                && resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm => {}
            (RuntimeResourceProducer::Pass(pass), None)
                if accounting.results.get(&resource.id).copied() == Some(*pass) => {}
            _ => return None,
        }
    }
    Some(())
}

fn validate_closed_pass_accounting(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
) -> Option<ClosedGraphAccounting> {
    let mut accounting = ClosedGraphAccounting {
        actual_reads: BTreeMap::new(),
        actual_last_reads: BTreeMap::new(),
        releases: BTreeMap::new(),
        results: BTreeMap::new(),
    };
    for (position, pass) in plan.passes.iter().enumerate() {
        let mut dependencies = BTreeSet::new();
        if pass.dependencies.iter().any(|dependency| {
            !dependencies.insert(*dependency)
                || maps
                    .pass_positions
                    .get(dependency)
                    .is_none_or(|dependency_position| *dependency_position >= position)
        }) {
            return None;
        }
        let mut pass_reads = BTreeSet::new();
        for read in &pass.reads {
            if !pass_reads.insert(read.resource)
                || !runtime_read_sampler_is_exact(read, &maps.resource_by_id)
            {
                return None;
            }
            validate_closed_read(plan, maps, &mut accounting, pass, read, position)?;
        }
        validate_closed_result(plan, maps, &mut accounting, pass)?;
        let mut pass_releases = BTreeSet::new();
        if pass.releases.iter().any(|resource| {
            !pass_releases.insert(*resource)
                || !pass_reads.contains(resource)
                || accounting.releases.insert(*resource, pass.id).is_some()
        }) {
            return None;
        }
        let expected_cache_keys = runtime_pass_cache_keys(
            &pass.kind,
            &pass.reads,
            pass.result,
            plan.working_format,
            plan.output_format,
            &maps.resource_formats,
        )
        .ok()?;
        if expected_cache_keys != pass.cache_keys {
            return None;
        }
    }
    Some(accounting)
}

fn validate_closed_read(
    _plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
    accounting: &mut ClosedGraphAccounting,
    pass: &RuntimePass,
    read: &RuntimeReadBinding,
    position: usize,
) -> Option<()> {
    let resource = maps.resource_by_id.get(&read.resource).copied()?;
    if pass.result == RuntimeResultBinding::Resource(read.resource) {
        return None;
    }
    if let RuntimeResourceProducer::Pass(producer) = resource.producer
        && (maps
            .pass_positions
            .get(&producer)
            .is_none_or(|producer_position| *producer_position >= position)
            || !pass.dependencies.contains(&producer))
    {
        return None;
    }
    let reads = accounting.actual_reads.entry(read.resource).or_default();
    *reads = reads.checked_add(1)?;
    accounting.actual_last_reads.insert(read.resource, pass.id);
    Some(())
}

fn validate_closed_result(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
    accounting: &mut ClosedGraphAccounting,
    pass: &RuntimePass,
) -> Option<()> {
    match pass.result {
        RuntimeResultBinding::Resource(resource) => {
            let request = maps.resource_by_id.get(&resource).copied()?;
            if request.producer != RuntimeResourceProducer::Pass(pass.id)
                || accounting.results.insert(resource, pass.id).is_some()
            {
                return None;
            }
        }
        RuntimeResultBinding::Output(format) => {
            if !matches!(pass.kind, RuntimePassKind::Present) || format != plan.output_format {
                return None;
            }
        }
        RuntimeResultBinding::Empty => {}
    }
    Some(())
}

fn closed_graph_root<'plan>(
    plan: &'plan LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
) -> Option<(&'plan RuntimePass, &'plan RuntimeResourceRequest)> {
    let clear = plan.passes.first()?;
    let RuntimePassKind::ClearRoot {
        initialization: RuntimeInitialization::SurfaceBaseColor,
        ..
    } = clear.kind
    else {
        return None;
    };
    if !clear.dependencies.is_empty()
        || !clear.reads.is_empty()
        || !clear.releases.is_empty()
        || clear.cache_keys.is_some()
        || clear.result != RuntimeResultBinding::Resource(plan.root_working_image)
    {
        return None;
    }
    let root = resources.get(&plan.root_working_image).copied()?;
    c08_resource_has_fixed_facts(
        root,
        RuntimeResourceRole::RootWorkingImage,
        RuntimeResourceFormat::Working(plan.working_format),
        RuntimeResourceProducer::Pass(clear.id),
    )
    .then_some((clear, root))
}

struct ClosedGraphTraversal<'plan> {
    plan: &'plan LoweredGraphPlan,
    resources: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    contexts: Vec<ExecutableCompositionContext>,
    captures: Vec<ExecutableVelloCaptureFacts>,
    layer_compositions: Vec<ExecutableLayerCompositionFacts>,
    color_filters: Vec<ExecutableColorFilterFacts>,
    blurs: Vec<ExecutableBlurFacts>,
    drop_shadows: Vec<ExecutableDropShadowFacts>,
    filter_steps: Vec<ExecutableFilterStepFacts>,
    backdrops: Vec<ExecutableBackdropFacts>,
    expected_resources: BTreeSet<RuntimeResourceId>,
    cursor: usize,
}

impl<'plan> ClosedGraphTraversal<'plan> {
    fn new(
        plan: &'plan LoweredGraphPlan,
        resources: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
        clear: &RuntimePass,
        root: &RuntimeResourceRequest,
    ) -> Self {
        Self {
            plan,
            resources,
            contexts: vec![ExecutableCompositionContext {
                current: root.id,
                producer: clear.id,
                contains_captured_source: false,
            }],
            captures: Vec::new(),
            layer_compositions: Vec::new(),
            color_filters: Vec::new(),
            blurs: Vec::new(),
            drop_shadows: Vec::new(),
            filter_steps: Vec::new(),
            backdrops: Vec::new(),
            expected_resources: BTreeSet::from([root.id]),
            cursor: 1,
        }
    }

    fn run(mut self) -> Option<ClosedExecutableGraphFacts> {
        while self.cursor < self.plan.passes.len() {
            self.visit_current_pass()?;
        }
        let clip_coverages_are_exact = self
            .layer_compositions
            .iter()
            .all(|layer| layer_has_exact_clip_coverage_capture(layer, &self.captures))
            && self.captures.iter().all(|capture| {
                capture.work().clip_coverage().is_none()
                    || self
                        .layer_compositions
                        .iter()
                        .any(|layer| layer.clip_coverage == Some(capture.target()))
            });
        if self.captures.is_empty()
            || self.contexts.len() != 1
            || !clip_coverages_are_exact
            || self.expected_resources.len() != self.plan.resources.len()
            || self
                .expected_resources
                .iter()
                .any(|resource| !self.resources.contains_key(resource))
        {
            return None;
        }
        Some(ClosedExecutableGraphFacts {
            working_format: self.plan.working_format,
            output_format: self.plan.output_format,
            captures: self.captures,
            layer_compositions: self.layer_compositions,
            color_filters: self.color_filters,
            blurs: self.blurs,
            drop_shadows: self.drop_shadows,
            filter_steps: self.filter_steps,
            backdrops: self.backdrops,
        })
    }

    fn visit_current_pass(&mut self) -> Option<()> {
        let pass = self.plan.passes.get(self.cursor)?;
        match &pass.kind {
            RuntimePassKind::ClearRoot {
                initialization: RuntimeInitialization::Transparent,
                color,
            } => self.visit_transparent_clear(pass, *color),
            RuntimePassKind::VelloCapture(Some(work)) if work.span().is_some() => {
                self.visit_span_capture(pass, work)
            }
            RuntimePassKind::VelloCapture(Some(work)) if work.clip_coverage().is_some() => {
                self.visit_clip_coverage_capture(pass)
            }
            RuntimePassKind::ColorFilter(Some(filter)) => self.visit_color_filter(pass, filter),
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.axis == RuntimeBlurAxis::Horizontal
                    && blur.input == RuntimeBlurInput::Rgba =>
            {
                self.visit_blur(pass, blur)
            }
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.axis == RuntimeBlurAxis::Horizontal
                    && blur.input == RuntimeBlurInput::SourceAlpha =>
            {
                self.visit_drop_shadow(pass, blur)
            }
            RuntimePassKind::CopyBackdrop => self.visit_backdrop(pass),
            RuntimePassKind::Composite(Some(composite))
                if matches!(composite.kind, RuntimeCompositeKind::Layer { .. }) =>
            {
                self.visit_layer_composite(pass)
            }
            RuntimePassKind::Present => self.visit_present(pass),
            RuntimePassKind::ClearRoot {
                initialization: RuntimeInitialization::SurfaceBaseColor,
                ..
            }
            | RuntimePassKind::VelloCapture(None)
            | RuntimePassKind::VelloCapture(Some(_))
            | RuntimePassKind::CanonicalizeCapture
            | RuntimePassKind::ColorFilter(_)
            | RuntimePassKind::BlurHorizontal(_)
            | RuntimePassKind::BlurVertical(_)
            | RuntimePassKind::DropShadowColorize(_)
            | RuntimePassKind::Composite(_) => None,
        }
    }

    fn visit_transparent_clear(&mut self, pass: &RuntimePass, color: Color) -> Option<()> {
        if color != Color::TRANSPARENT
            || !pass.dependencies.is_empty()
            || !pass.reads.is_empty()
            || !pass.releases.is_empty()
            || pass.cache_keys.is_some()
        {
            return None;
        }
        let RuntimeResultBinding::Resource(resource) = pass.result else {
            return None;
        };
        let request = self.resources.get(&resource).copied()?;
        if !c08_resource_has_fixed_facts(
            request,
            RuntimeResourceRole::IsolationWorkingImage,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(pass.id),
        ) {
            return None;
        }
        self.expected_resources.insert(resource);
        self.contexts.push(ExecutableCompositionContext {
            current: resource,
            producer: pass.id,
            contains_captured_source: false,
        });
        self.advance(1)
    }

    fn visit_span_capture(&mut self, pass: &RuntimePass, work: &RuntimeVelloCapture) -> Option<()> {
        let span = work.span()?;
        let canonicalize = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let after_canonicalize = self.plan.passes.get(self.cursor.checked_add(2)?)?;
        let (coverage_pass, composite, pass_count) = if matches!(
            after_canonicalize.kind,
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(_)))
        ) {
            (
                Some(after_canonicalize),
                self.plan.passes.get(self.cursor.checked_add(3)?)?,
                4,
            )
        } else {
            (None, after_canonicalize, 3)
        };
        let (capture_target, capture_resource) =
            self.validate_span_capture_source(pass, canonicalize, span)?;
        let (canonical_target, canonical_resource) = self.validate_canonical_capture(
            pass,
            canonicalize,
            composite,
            capture_target,
            capture_resource,
        )?;
        let capture_facts = executable_vello_capture_facts(
            pass.id,
            capture_target,
            work,
            capture_resource.spatial,
        )?;
        let coverage_facts = match coverage_pass {
            Some(coverage) => Some(validate_closed_clip_coverage_capture(
                coverage,
                composite.id,
                &self.resources,
            )?),
            None => None,
        };
        let parent = *self.contexts.last()?;
        let layer = validate_closed_composite(
            composite,
            parent,
            canonical_resource,
            &self.resources,
            self.plan.working_format,
            false,
        )?;
        let RuntimeResultBinding::Resource(result) = composite.result else {
            return None;
        };
        self.record_span_capture(
            composite,
            [capture_target, canonical_target, result],
            capture_facts,
            coverage_facts,
            layer,
        )?;
        self.advance(pass_count)
    }

    fn validate_span_capture_source(
        &self,
        pass: &RuntimePass,
        canonicalize: &RuntimePass,
        span: &RuntimeVelloSpan,
    ) -> Option<(RuntimeResourceId, &'plan RuntimeResourceRequest)> {
        let RuntimeResultBinding::Resource(capture_target) = pass.result else {
            return None;
        };
        let expected_scope = if self.contexts.len() == 1 {
            RuntimeVelloSpanScope::CurrentParent
        } else {
            RuntimeVelloSpanScope::LayerSource
        };
        if !pass.dependencies.is_empty()
            || !pass.reads.is_empty()
            || !pass.releases.is_empty()
            || pass.cache_keys.is_some()
            || span.scope != expected_scope
        {
            return None;
        }
        let resource = self.resources.get(&capture_target).copied()?;
        c08_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::CaptureWorkingImage,
            RuntimeResourceFormat::VelloCaptureRgba8Unorm,
            RuntimeResourceProducer::Pass(pass.id),
        )
        .then_some(())
        .filter(|()| resource.expected_reads == 1 && resource.last_use == canonicalize.id)?;
        Some((capture_target, resource))
    }

    fn validate_canonical_capture(
        &self,
        capture: &RuntimePass,
        canonicalize: &RuntimePass,
        composite: &RuntimePass,
        capture_target: RuntimeResourceId,
        capture_resource: &RuntimeResourceRequest,
    ) -> Option<(RuntimeResourceId, &'plan RuntimeResourceRequest)> {
        if !matches!(canonicalize.kind, RuntimePassKind::CanonicalizeCapture)
            || canonicalize.dependencies.as_slice() != [capture.id]
            || canonicalize.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &canonicalize.reads[0],
                RuntimeReadRole::CaptureSource,
                capture_resource,
                RuntimeSamplingFilter::Linear,
                RuntimeSamplingEdge::ClampToExtent,
            )
            || canonicalize.releases.as_slice() != [capture_target]
        {
            return None;
        }
        let RuntimeResultBinding::Resource(canonical_target) = canonicalize.result else {
            return None;
        };
        let resource = self.resources.get(&canonical_target).copied()?;
        c08_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::FilterIntermediate,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(canonicalize.id),
        )
        .then_some(())
        .filter(|()| {
            resource.expected_reads == 1
                && resource.last_use == composite.id
                && resource.spatial == capture_resource.spatial
        })?;
        Some((canonical_target, resource))
    }

    fn record_span_capture(
        &mut self,
        composite: &RuntimePass,
        resources: [RuntimeResourceId; 3],
        capture: ExecutableVelloCaptureFacts,
        coverage: Option<(RuntimeResourceId, ExecutableVelloCaptureFacts)>,
        layer: Option<ExecutableLayerCompositionFacts>,
    ) -> Option<()> {
        let context = self.contexts.last_mut()?;
        context.current = resources[2];
        context.producer = composite.id;
        context.contains_captured_source = true;
        self.expected_resources.extend(resources);
        self.captures.push(capture);
        if let Some((resource, facts)) = coverage {
            self.expected_resources.insert(resource);
            self.captures.push(facts);
        }
        if let Some(layer) = layer {
            self.record_layer_resources(&layer);
            self.layer_compositions.push(layer);
        }
        Some(())
    }

    fn visit_clip_coverage_capture(&mut self, pass: &RuntimePass) -> Option<()> {
        let composite = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let (coverage, facts) =
            validate_closed_clip_coverage_capture(pass, composite.id, &self.resources)?;
        self.expected_resources.insert(coverage);
        self.captures.push(facts);
        self.advance(1)
    }

    fn visit_color_filter(
        &mut self,
        pass: &RuntimePass,
        filter: &RuntimeColorFilter,
    ) -> Option<()> {
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let color = validate_closed_color_filter(
            pass,
            context,
            source,
            filter,
            &self.resources,
            self.plan.working_format,
        )?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = color.result;
        runtime_context.producer = pass.id;
        self.expected_resources.insert(color.result);
        self.filter_steps
            .push(ExecutableFilterStepFacts::Color(pass.id));
        self.color_filters.push(color);
        self.advance(1)
    }

    fn visit_blur(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        self.visit_blur_with_edge(horizontal, blur, RuntimeSamplingEdge::TransparentBlack)
    }

    fn visit_blur_with_edge(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
        edge: RuntimeSamplingEdge,
    ) -> Option<()> {
        let vertical = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let validation = ClosedFilterValidation {
            context,
            source,
            resources: &self.resources,
            working_format: self.plan.working_format,
            edge,
        };
        let facts = validate_closed_blur(horizontal, vertical, blur, validation)?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = facts.result;
        runtime_context.producer = facts.vertical;
        self.expected_resources
            .extend([facts.intermediate, facts.result]);
        self.filter_steps.push(ExecutableFilterStepFacts::Blur {
            horizontal: facts.horizontal,
            vertical: facts.vertical,
        });
        self.blurs.push(facts);
        self.advance(2)
    }

    fn visit_drop_shadow(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        self.visit_drop_shadow_with_edge(horizontal, blur, RuntimeSamplingEdge::TransparentBlack)
    }

    fn visit_drop_shadow_with_edge(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
        edge: RuntimeSamplingEdge,
    ) -> Option<()> {
        let vertical = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let colorize = self.plan.passes.get(self.cursor.checked_add(2)?)?;
        let merge = self.plan.passes.get(self.cursor.checked_add(3)?)?;
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let validation = ClosedFilterValidation {
            context,
            source,
            resources: &self.resources,
            working_format: self.plan.working_format,
            edge,
        };
        let facts =
            validate_closed_drop_shadow([horizontal, vertical, colorize, merge], blur, validation)?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = facts.result;
        runtime_context.producer = facts.merge;
        self.expected_resources.extend([
            facts.horizontal_result,
            facts.vertical_result,
            facts.shadow,
            facts.result,
        ]);
        self.filter_steps
            .push(ExecutableFilterStepFacts::DropShadow {
                horizontal: facts.horizontal,
                vertical: facts.vertical,
                colorize: facts.colorize,
                merge: facts.merge,
            });
        self.drop_shadows.push(facts);
        self.advance(4)
    }

    fn visit_backdrop(&mut self, copy: &RuntimePass) -> Option<()> {
        let foreground = match self.contexts.len() {
            1 => None,
            2 => Some(self.contexts.pop()?),
            _ => return None,
        };
        if foreground.is_some_and(|context| !context.contains_captured_source) {
            return None;
        }
        let parent = *self.contexts.first()?;
        let copied = self.validate_backdrop_copy(copy, parent)?;
        let filter_start = self.filter_steps.len();
        self.contexts.push(ExecutableCompositionContext {
            current: copied,
            producer: copy.id,
            contains_captured_source: true,
        });
        self.advance(1)?;
        self.visit_backdrop_filters()?;
        let filtered = self.contexts.pop()?;
        let filter_steps = self.filter_steps.get(filter_start..)?.to_vec();
        let group_clear = self.plan.passes.get(self.cursor)?;
        self.visit_transparent_clear(group_clear, Color::TRANSPARENT)?;
        let backdrop_composite = self.visit_backdrop_source_composite(filtered)?;
        let foreground_composite = match foreground {
            Some(foreground) => Some(self.visit_backdrop_foreground(foreground)?),
            None => None,
        };
        let completed_group = *self.contexts.last()?;
        let outer_composite = self.visit_backdrop_outer_composite(parent, completed_group)?;
        let result = self.contexts.first()?.current;
        self.backdrops.push(ExecutableBackdropFacts {
            copy: copy.id,
            completed_parent: parent.current,
            copied,
            foreground: foreground.map(|context| context.current),
            filter_steps,
            filtered: filtered.current,
            group_clear: group_clear.id,
            backdrop_composite,
            foreground_composite,
            outer_composite,
            completed_group: completed_group.current,
            result,
        });
        Some(())
    }

    fn validate_backdrop_copy(
        &mut self,
        copy: &RuntimePass,
        parent: ExecutableCompositionContext,
    ) -> Option<RuntimeResourceId> {
        let parent_resource = self.resources.get(&parent.current).copied()?;
        if copy.dependencies.as_slice() != [parent.producer]
            || copy.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &copy.reads[0],
                RuntimeReadRole::CompletedParent,
                parent_resource,
                RuntimeSamplingFilter::Nearest,
                RuntimeSamplingEdge::TransparentBlack,
            )
            || !copy.releases.is_empty()
            || copy.cache_keys.is_none()
        {
            return None;
        }
        let RuntimeResultBinding::Resource(copied) = copy.result else {
            return None;
        };
        let resource = self.resources.get(&copied).copied()?;
        if !c08_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::BackdropCopy,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(copy.id),
        ) || resource.expected_reads != 1
        {
            return None;
        }
        self.expected_resources.insert(copied);
        Some(copied)
    }

    fn visit_backdrop_filters(&mut self) -> Option<()> {
        loop {
            let pass = self.plan.passes.get(self.cursor)?;
            match &pass.kind {
                RuntimePassKind::ColorFilter(Some(filter)) => {
                    self.visit_color_filter(pass, filter)?;
                }
                RuntimePassKind::BlurHorizontal(Some(blur))
                    if blur.axis == RuntimeBlurAxis::Horizontal
                        && blur.input == RuntimeBlurInput::Rgba =>
                {
                    self.visit_backdrop_blur(pass, blur)?;
                }
                RuntimePassKind::BlurHorizontal(Some(blur))
                    if blur.axis == RuntimeBlurAxis::Horizontal
                        && blur.input == RuntimeBlurInput::SourceAlpha =>
                {
                    self.visit_backdrop_drop_shadow(pass, blur)?;
                }
                RuntimePassKind::ClearRoot {
                    initialization: RuntimeInitialization::Transparent,
                    color,
                } if *color == Color::TRANSPARENT => return Some(()),
                _ => return None,
            }
        }
    }

    fn visit_backdrop_blur(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        let RuntimeSamplingEdge::SemanticBorderMirror(_) = blur.edge else {
            return None;
        };
        self.visit_blur_with_edge(horizontal, blur, blur.edge)
    }

    fn visit_backdrop_drop_shadow(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
    ) -> Option<()> {
        let RuntimeSamplingEdge::SemanticBorderMirror(_) = blur.edge else {
            return None;
        };
        self.visit_drop_shadow_with_edge(horizontal, blur, blur.edge)
    }

    fn visit_backdrop_source_composite(
        &mut self,
        filtered: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.next_layer_composite_with_optional_coverage()?;
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&filtered.current).copied()?;
        let layer = validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
        )??;
        if !runtime_composite_is_backdrop_inner(&layer.composite) {
            return None;
        }
        self.record_backdrop_composite_result(pass, layer)?;
        Some(pass.id)
    }

    fn visit_backdrop_foreground(
        &mut self,
        foreground: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.plan.passes.get(self.cursor)?;
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&foreground.current).copied()?;
        if validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            false,
        )?
        .is_some()
        {
            return None;
        }
        self.record_backdrop_span_result(pass)?;
        Some(pass.id)
    }

    fn visit_backdrop_outer_composite(
        &mut self,
        parent: ExecutableCompositionContext,
        group: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.next_layer_composite_with_optional_coverage()?;
        let source = self.resources.get(&group.current).copied()?;
        let layer = validate_closed_composite_with_parent_reads(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
            2,
        )??;
        if !runtime_composite_is_untransformed_outer(&layer.composite) {
            return None;
        }
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        self.contexts.pop()?;
        let root = self.contexts.first_mut()?;
        root.current = result;
        root.producer = pass.id;
        root.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)?;
        Some(pass.id)
    }

    fn next_layer_composite_with_optional_coverage(&mut self) -> Option<&'plan RuntimePass> {
        let pass = self.plan.passes.get(self.cursor)?;
        if matches!(
            pass.kind,
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(_)))
        ) {
            self.visit_clip_coverage_capture(pass)?;
        }
        let composite = self.plan.passes.get(self.cursor)?;
        matches!(
            composite.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { .. },
                ..
            }))
        )
        .then_some(composite)
    }

    fn record_backdrop_composite_result(
        &mut self,
        pass: &RuntimePass,
        layer: ExecutableLayerCompositionFacts,
    ) -> Option<()> {
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)
    }

    fn record_backdrop_span_result(&mut self, pass: &RuntimePass) -> Option<()> {
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.advance(1)
    }

    fn visit_layer_composite(&mut self, pass: &RuntimePass) -> Option<()> {
        if self.contexts.len() < 2 {
            return None;
        }
        let source_context = self.contexts.pop()?;
        if !source_context.contains_captured_source {
            return None;
        }
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&source_context.current).copied()?;
        let layer = validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
        )??;
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)
    }

    fn visit_present(&mut self, pass: &RuntimePass) -> Option<()> {
        if self.cursor.checked_add(1)? != self.plan.passes.len()
            || pass.id != self.plan.final_present
            || self.contexts.len() != 1
        {
            return None;
        }
        let parent = self.contexts[0];
        let resource = self.resources.get(&parent.current).copied()?;
        if pass.dependencies.as_slice() != [parent.producer]
            || pass.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &pass.reads[0],
                RuntimeReadRole::FinalWorkingImage,
                resource,
                RuntimeSamplingFilter::Linear,
                RuntimeSamplingEdge::ClampToExtent,
            )
            || pass.result != RuntimeResultBinding::Output(self.plan.output_format)
            || pass.releases.as_slice() != [parent.current]
            || resource.expected_reads != 1
            || resource.last_use != pass.id
        {
            return None;
        }
        self.advance(1)
    }

    fn record_layer_resources(&mut self, layer: &ExecutableLayerCompositionFacts) {
        if let Some(coverage) = layer.clip_coverage {
            self.expected_resources.insert(coverage);
        }
        if let Some(mask) = layer.alpha_mask {
            self.expected_resources.insert(mask);
        }
    }

    fn advance(&mut self, count: usize) -> Option<()> {
        self.cursor = self.cursor.checked_add(count)?;
        Some(())
    }
}

impl LoweredGraphPlan {
    pub(crate) fn try_lower_validated_graph(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
        capabilities: &DeviceCapabilities,
    ) -> Result<Self> {
        Self::try_lower_validated_graph_inner(
            graph,
            working_format,
            output_format,
            GraphLoweringCapabilityValidation::Required(capabilities),
        )
    }

    fn try_lower_for_dispatch_classification(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
    ) -> Result<Self> {
        Self::try_lower_validated_graph_inner(
            graph,
            working_format,
            output_format,
            GraphLoweringCapabilityValidation::ClassificationOnly,
        )
    }

    fn try_lower_validated_graph_inner(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
        capability_validation: GraphLoweringCapabilityValidation<'_>,
    ) -> Result<Self> {
        let view = graph.lowering_view()?;
        let resource_views = view.resources();
        let pass_views = view.passes();
        let pass_ids = pass_views
            .iter()
            .map(|pass| RuntimePassId(pass.id()))
            .collect::<BTreeSet<_>>();
        let LoweredResourceSet {
            resources,
            resource_formats,
        } = lower_runtime_resources(
            &resource_views,
            &pass_ids,
            working_format,
            capability_validation,
        )?;

        let mut releases = BTreeMap::<RuntimePassId, Vec<RuntimeResourceId>>::new();
        for resource in &resources {
            releases
                .entry(resource.last_use)
                .or_default()
                .push(resource.id);
        }
        let resource_by_id = resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();

        let passes = lower_runtime_passes(
            &pass_views,
            &resource_by_id,
            &resource_formats,
            &mut releases,
            working_format,
            output_format,
        )?;
        if !releases.is_empty() {
            return Err(lowering_error(
                "one or more release bindings have no runtime pass",
            ));
        }

        let root_working_image = RuntimeResourceId(view.root_working_image());
        if resource_formats.get(&root_working_image)
            != Some(&RuntimeResourceFormat::Working(working_format))
        {
            return Err(lowering_error(
                "root working image does not use the graph format",
            ));
        }
        let final_present = RuntimePassId(view.final_present());
        if passes.last().is_none_or(|pass| {
            pass.id != final_present || !matches!(pass.kind, RuntimePassKind::Present)
        }) {
            return Err(lowering_error(
                "the runtime plan has no terminal present pass",
            ));
        }

        Ok(Self {
            generation: RuntimeGraphGeneration(view.generation()),
            working_format,
            output_format,
            resources,
            passes,
            root_working_image,
            final_present,
        })
    }

    fn closed_executable_graph_facts(&self) -> Option<ClosedExecutableGraphFacts> {
        if !matches!(self.output_format, Format::Rgba8 | Format::Bgra8) || self.passes.len() < 5 {
            return None;
        }
        let maps = ClosedGraphMaps::try_new(self)?;
        validate_closed_graph_accounting(self, &maps)?;
        let (clear, root) = closed_graph_root(self, &maps.resource_by_id)?;
        ClosedGraphTraversal::new(self, maps.resource_by_id, clear, root).run()
    }

    #[cfg(test)]
    fn c08_execution_facts(&self) -> Option<C08ExecutionFacts> {
        if ![Format::Rgba8, Format::Bgra8].contains(&self.output_format) {
            return None;
        }
        let resource_by_id = self
            .resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();
        if resource_by_id.len() != self.resources.len() {
            return None;
        }
        let resource_formats = self
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let pass_ids = self
            .passes
            .iter()
            .map(|pass| pass.id)
            .collect::<BTreeSet<_>>();
        if pass_ids.len() != self.passes.len() {
            return None;
        }

        let clear = self.passes.first()?;
        let root = resource_by_id.get(&self.root_working_image).copied()?;
        if !c08_root_is_exact(self, clear, root) {
            return None;
        }
        let sequence = c08_capture_sequence(self, &resource_by_id, root, clear.id)?;
        let present = self.passes.get(sequence.cursor)?;
        if !c08_present_is_exact(self, present, &sequence) {
            return None;
        }
        if sequence.expected_resources.len() != self.resources.len()
            || sequence
                .expected_resources
                .iter()
                .any(|resource| !resource_by_id.contains_key(resource))
            || !c08_cache_keys_are_exact(self, &resource_formats)
        {
            return None;
        }

        let execution = C08ExecutionFacts {
            working_format: self.working_format,
            output_format: self.output_format,
            captures: sequence.captures,
        };
        execution
            .proves_exact_execution_facts_for(self)
            .then_some(execution)
    }

    #[cfg(test)]
    pub(crate) fn with_duplicate_preparation_resource_for_test(&self) -> Self {
        let mut invalid = self.clone();
        if invalid.resources.len() > 1 {
            invalid.resources[1].id = invalid.resources[0].id;
        }
        invalid
    }
}

#[cfg(test)]
struct C08CaptureSequenceFacts<'resource> {
    cursor: usize,
    parent: &'resource RuntimeResourceRequest,
    parent_producer: RuntimePassId,
    captures: Vec<ExecutableVelloCaptureFacts>,
    expected_resources: BTreeSet<RuntimeResourceId>,
}

#[cfg(test)]
struct C08CapturePairFacts {
    capture_target: RuntimeResourceId,
    canonical_target: RuntimeResourceId,
    canonical_pass: RuntimePassId,
    capture: ExecutableVelloCaptureFacts,
}

#[cfg(test)]
fn c08_root_is_exact(
    plan: &LoweredGraphPlan,
    clear: &RuntimePass,
    root: &RuntimeResourceRequest,
) -> bool {
    c08_pass_class(&clear.kind) == Some(C08PassClass::ClearRoot)
        && clear.dependencies.is_empty()
        && clear.reads.is_empty()
        && clear.releases.is_empty()
        && clear.cache_keys.is_none()
        && clear.result == RuntimeResultBinding::Resource(plan.root_working_image)
        && c08_resource_has_fixed_facts(
            root,
            RuntimeResourceRole::RootWorkingImage,
            RuntimeResourceFormat::Working(plan.working_format),
            RuntimeResourceProducer::Pass(clear.id),
        )
}

#[cfg(test)]
fn c08_capture_pair(
    plan: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    cursor: usize,
) -> Option<C08CapturePairFacts> {
    let capture = plan.passes.get(cursor)?;
    let canonicalize = plan.passes.get(cursor.checked_add(1)?)?;
    let RuntimePassKind::VelloCapture(Some(work @ RuntimeVelloCapture::Span(_))) = &capture.kind
    else {
        return None;
    };
    let RuntimeResultBinding::Resource(capture_target) = capture.result else {
        return None;
    };
    if !capture.dependencies.is_empty()
        || !capture.reads.is_empty()
        || !capture.releases.is_empty()
        || capture.cache_keys.is_some()
    {
        return None;
    }
    let capture_resource = resources.get(&capture_target).copied()?;
    if !c08_resource_has_fixed_facts(
        capture_resource,
        RuntimeResourceRole::CaptureWorkingImage,
        RuntimeResourceFormat::VelloCaptureRgba8Unorm,
        RuntimeResourceProducer::Pass(capture.id),
    ) || capture_resource.expected_reads != 1
        || capture_resource.last_use != canonicalize.id
        || c08_pass_class(&canonicalize.kind) != Some(C08PassClass::CanonicalizeCapture)
        || canonicalize.dependencies.as_slice() != [capture.id]
        || canonicalize.reads.len() != 1
        || !c08_read_is_exact(
            &canonicalize.reads[0],
            RuntimeReadRole::CaptureSource,
            capture_target,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::ClampToExtent,
            capture_resource.format,
        )
        || canonicalize.releases.as_slice() != [capture_target]
    {
        return None;
    }
    let RuntimeResultBinding::Resource(canonical_target) = canonicalize.result else {
        return None;
    };
    let canonical_resource = resources.get(&canonical_target).copied()?;
    let composite = plan.passes.get(cursor.checked_add(2)?)?;
    if !c08_resource_has_fixed_facts(
        canonical_resource,
        RuntimeResourceRole::FilterIntermediate,
        RuntimeResourceFormat::Working(plan.working_format),
        RuntimeResourceProducer::Pass(canonicalize.id),
    ) || canonical_resource.expected_reads != 1
        || canonical_resource.last_use != composite.id
        || canonical_resource.spatial != capture_resource.spatial
    {
        return None;
    }
    Some(C08CapturePairFacts {
        capture_target,
        canonical_target,
        canonical_pass: canonicalize.id,
        capture: executable_vello_capture_facts(
            capture.id,
            capture_target,
            work,
            capture_resource.spatial,
        )?,
    })
}

#[cfg(test)]
fn c08_composite_after_capture<'resource>(
    plan: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &'resource RuntimeResourceRequest>,
    cursor: usize,
    parent: &RuntimeResourceRequest,
    parent_producer: RuntimePassId,
    pair: &C08CapturePairFacts,
    root_spatial: RuntimeSpatialDescriptor,
) -> Option<&'resource RuntimeResourceRequest> {
    let composite = plan.passes.get(cursor.checked_add(2)?)?;
    let canonical = resources.get(&pair.canonical_target).copied()?;
    if c08_pass_class(&composite.kind) != Some(C08PassClass::SpanSourceOver)
        || composite.dependencies.as_slice() != [parent_producer, pair.canonical_pass]
        || composite.reads.len() != 2
        || !c08_read_is_exact(
            &composite.reads[0],
            RuntimeReadRole::CompositeParent,
            parent.id,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::ClampToExtent,
            parent.format,
        )
        || !c08_read_is_exact(
            &composite.reads[1],
            RuntimeReadRole::CompositeSource,
            pair.canonical_target,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
            canonical.format,
        )
        || composite.releases.as_slice() != [parent.id, pair.canonical_target]
        || parent.expected_reads != 1
        || parent.last_use != composite.id
    {
        return None;
    }
    let RuntimeResultBinding::Resource(target) = composite.result else {
        return None;
    };
    let result = resources.get(&target).copied()?;
    (c08_resource_has_fixed_facts(
        result,
        RuntimeResourceRole::CompositeResult,
        RuntimeResourceFormat::Working(plan.working_format),
        RuntimeResourceProducer::Pass(composite.id),
    ) && result.spatial == root_spatial)
        .then_some(result)
}

#[cfg(test)]
fn c08_capture_sequence<'resource>(
    plan: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &'resource RuntimeResourceRequest>,
    root: &'resource RuntimeResourceRequest,
    clear: RuntimePassId,
) -> Option<C08CaptureSequenceFacts<'resource>> {
    let mut facts = C08CaptureSequenceFacts {
        cursor: 1,
        parent: root,
        parent_producer: clear,
        captures: Vec::new(),
        expected_resources: BTreeSet::from([plan.root_working_image]),
    };
    while plan
        .passes
        .get(facts.cursor)
        .is_some_and(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::VelloCapture))
    {
        let pair = c08_capture_pair(plan, resources, facts.cursor)?;
        let result = c08_composite_after_capture(
            plan,
            resources,
            facts.cursor,
            facts.parent,
            facts.parent_producer,
            &pair,
            root.spatial,
        )?;
        facts
            .expected_resources
            .extend([pair.capture_target, pair.canonical_target, result.id]);
        facts.captures.push(pair.capture);
        facts.parent = result;
        facts.parent_producer = plan.passes[facts.cursor.checked_add(2)?].id;
        facts.cursor = facts.cursor.checked_add(3)?;
    }
    (!facts.captures.is_empty()).then_some(facts)
}

#[cfg(test)]
fn c08_present_is_exact(
    plan: &LoweredGraphPlan,
    present: &RuntimePass,
    sequence: &C08CaptureSequenceFacts<'_>,
) -> bool {
    sequence.cursor.checked_add(1) == Some(plan.passes.len())
        && present.id == plan.final_present
        && c08_pass_class(&present.kind) == Some(C08PassClass::Present)
        && present.dependencies.as_slice() == [sequence.parent_producer]
        && present.reads.len() == 1
        && c08_read_is_exact(
            &present.reads[0],
            RuntimeReadRole::FinalWorkingImage,
            sequence.parent.id,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::ClampToExtent,
            sequence.parent.format,
        )
        && present.result == RuntimeResultBinding::Output(plan.output_format)
        && present.releases.as_slice() == [sequence.parent.id]
        && sequence.parent.expected_reads == 1
        && sequence.parent.last_use == present.id
}

#[cfg(test)]
fn c08_cache_keys_are_exact(
    plan: &LoweredGraphPlan,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
) -> bool {
    plan.passes.iter().all(|pass| {
        runtime_pass_cache_keys(
            &pass.kind,
            &pass.reads,
            pass.result,
            plan.working_format,
            plan.output_format,
            resource_formats,
        )
        .ok()
        .is_some_and(|expected| pass.cache_keys == expected)
    })
}

fn runtime_read_sampler_is_exact(
    read: &RuntimeReadBinding,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> bool {
    let Some(resource) = resources.get(&read.resource).copied() else {
        return false;
    };
    let resolved_mask = match (&read.role, &resource.import) {
        (RuntimeReadRole::AlphaMask, Some(RuntimeResourceImport::ResolvedAlphaMask(upload))) => {
            Some(ShaderMaskSamplingKey::new(
                upload.quality(),
                upload.extend(),
            ))
        }
        (RuntimeReadRole::AlphaMask, None) => return false,
        (RuntimeReadRole::ClipCoverage, Some(_)) => return false,
        (RuntimeReadRole::ClipCoverage, None) => None,
        (_, _) => None,
    };
    read.sampler_key
        == SamplerKey::new(
            shader_binding_role(read.role),
            resource.format.shader_key(),
            match read.sampling_filter {
                RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
            },
            shader_sampling_edge(read.sampling_edge),
            resolved_mask,
        )
}

fn runtime_read_has_exact_facts(
    read: &RuntimeReadBinding,
    role: RuntimeReadRole,
    resource: &RuntimeResourceRequest,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
) -> bool {
    read.role == role
        && read.resource == resource.id
        && read.sampling_filter == sampling_filter
        && read.sampling_edge == sampling_edge
        && runtime_read_sampler_is_exact(read, &BTreeMap::from([(resource.id, resource)]))
}

fn validate_closed_clip_coverage_capture(
    pass: &RuntimePass,
    composite: RuntimePassId,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> Option<(RuntimeResourceId, ExecutableVelloCaptureFacts)> {
    let RuntimePassKind::VelloCapture(Some(work @ RuntimeVelloCapture::ClipCoverage(_))) =
        &pass.kind
    else {
        return None;
    };
    let RuntimeResultBinding::Resource(target) = pass.result else {
        return None;
    };
    if !pass.dependencies.is_empty()
        || !pass.reads.is_empty()
        || !pass.releases.is_empty()
        || pass.cache_keys.is_some()
    {
        return None;
    }
    let resource = resources.get(&target).copied()?;
    if !c08_resource_has_fixed_facts(
        resource,
        RuntimeResourceRole::ClipCoverage,
        RuntimeResourceFormat::ClipCoverageRgba8Unorm,
        RuntimeResourceProducer::Pass(pass.id),
    ) || resource.expected_reads != 1
        || resource.last_use != composite
    {
        return None;
    }
    Some((
        target,
        executable_vello_capture_facts(pass.id, target, work, resource.spatial)?,
    ))
}

fn layer_has_exact_clip_coverage_capture(
    layer: &ExecutableLayerCompositionFacts,
    captures: &[ExecutableVelloCaptureFacts],
) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        clip,
        outer_clips,
        clip_coverage,
        ..
    } = &layer.composite.kind
    else {
        return false;
    };
    let mut expected = outer_clips
        .iter()
        .map(|outer| RuntimeClipCoverageElement {
            clip: outer.clip.clone(),
            transform: outer.transform,
        })
        .collect::<Vec<_>>();
    if let Some(clip) = clip {
        expected.push(RuntimeClipCoverageElement {
            clip: (**clip).clone(),
            transform: *transform,
        });
    }
    match (expected.is_empty(), clip_coverage) {
        (true, None) => true,
        (false, Some(coverage)) => {
            let mut matching = captures
                .iter()
                .filter(|capture| capture.target() == *coverage);
            let exact = matching.next().is_some_and(|capture| {
                capture
                    .work()
                    .clip_coverage()
                    .is_some_and(|coverage| coverage.elements == expected)
            });
            exact && matching.next().is_none() && layer.clip_coverage == Some(*coverage)
        }
        (true, Some(_)) | (false, None) => false,
    }
}

fn validate_closed_color_filter(
    pass: &RuntimePass,
    context: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    filter: &RuntimeColorFilter,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
) -> Option<ExecutableColorFilterFacts> {
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || pass.dependencies.as_slice() != [source_producer]
        || pass.reads.len() != 1
        || !runtime_read_has_exact_facts(
            &pass.reads[0],
            RuntimeReadRole::FilterSource,
            source,
            RuntimeSamplingFilter::Nearest,
            RuntimeSamplingEdge::ClampToExtent,
        )
        || pass.releases.as_slice() != [source.id]
        || pass.cache_keys.is_none()
        || source.format != RuntimeResourceFormat::Working(working_format)
        || !matches!(
            source.role,
            RuntimeResourceRole::BackdropCopy
                | RuntimeResourceRole::FilterIntermediate
                | RuntimeResourceRole::CompositeResult
        )
        || source.expected_reads != 1
        || source.last_use != pass.id
        || filter.operations.is_empty()
        || filter.edge != RuntimeSamplingEdge::ClampToExtent
        || filter
            .operations
            .iter()
            .any(|operation| !runtime_color_operation_is_closed(operation))
    {
        return None;
    }
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    if result == source.id {
        return None;
    }
    let result_resource = resources.get(&result).copied()?;
    if !c08_resource_has_fixed_facts(
        result_resource,
        RuntimeResourceRole::FilterIntermediate,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    ) || result_resource.expected_reads != 1
        || source.spatial != result_resource.spatial
        || filter.spatial.source != source.spatial
        || filter.spatial.result != result_resource.spatial
    {
        return None;
    }
    Some(ExecutableColorFilterFacts {
        pass: pass.id,
        source: source.id,
        result,
        filter: filter.clone(),
    })
}

fn runtime_color_operation_is_closed(operation: &RuntimeColorOperation) -> bool {
    if operation.clamp_boundary != RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
    {
        return false;
    }
    match operation.operation {
        RuntimeColorOperationKind::Brightness(amount)
        | RuntimeColorOperationKind::Contrast(amount)
        | RuntimeColorOperationKind::Saturate(amount) => {
            if amount.zero() {
                amount.mantissa() == 0.0 && amount.exponent() == 0
            } else {
                amount.mantissa().is_finite() && (0.5..1.0).contains(&amount.mantissa())
            }
        }
        RuntimeColorOperationKind::Grayscale(amount)
        | RuntimeColorOperationKind::Invert(amount)
        | RuntimeColorOperationKind::Opacity(amount)
        | RuntimeColorOperationKind::Sepia(amount) => {
            amount.value().is_finite() && (0.0..=1.0).contains(&amount.value())
        }
        RuntimeColorOperationKind::HueRotate(angle) => {
            angle.sine().is_finite() && angle.cosine().is_finite()
        }
    }
}

#[derive(Clone, Copy)]
struct ClosedFilterValidation<'plan> {
    context: ExecutableCompositionContext,
    source: &'plan RuntimeResourceRequest,
    resources: &'plan BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
    edge: RuntimeSamplingEdge,
}

fn validate_closed_blur(
    horizontal: &RuntimePass,
    vertical: &RuntimePass,
    blur: &RuntimeBlur,
    validation: ClosedFilterValidation<'_>,
) -> Option<ExecutableBlurFacts> {
    let ClosedFilterValidation {
        context,
        source,
        resources,
        working_format,
        edge,
    } = validation;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || blur.axis != RuntimeBlurAxis::Horizontal
        || blur.input != RuntimeBlurInput::Rgba
        || !runtime_blur_is_closed(blur, true, edge)
        || blur.spatial.source != source.spatial
        || !closed_filter_source_is_exact(source, working_format, 1, horizontal.id)
        || !closed_unary_filter_pass_is_exact(
            horizontal,
            source_producer,
            source,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let intermediate = closed_filter_result(
        horizontal,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let RuntimePassKind::BlurVertical(Some(vertical_blur)) = &vertical.kind else {
        return None;
    };
    if !runtime_blur_matches_axis(vertical_blur, blur, RuntimeBlurAxis::Vertical)
        || intermediate.expected_reads != 1
        || intermediate.last_use != vertical.id
        || !closed_unary_filter_pass_is_exact(
            vertical,
            horizontal.id,
            intermediate,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let result = closed_filter_result(
        vertical,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    Some(ExecutableBlurFacts {
        horizontal: horizontal.id,
        vertical: vertical.id,
        source: source.id,
        intermediate: intermediate.id,
        result: result.id,
        blur: blur.clone(),
    })
}

fn validate_closed_drop_shadow(
    passes: [&RuntimePass; 4],
    blur: &RuntimeBlur,
    validation: ClosedFilterValidation<'_>,
) -> Option<ExecutableDropShadowFacts> {
    let ClosedFilterValidation {
        context,
        source,
        resources,
        working_format,
        edge,
    } = validation;
    let [horizontal, vertical, colorize, merge] = passes;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || blur.axis != RuntimeBlurAxis::Horizontal
        || blur.input != RuntimeBlurInput::SourceAlpha
        || !runtime_blur_is_closed(blur, false, edge)
        || blur.spatial.source != source.spatial
        || !closed_filter_source_is_exact(source, working_format, 2, merge.id)
        || !closed_unary_filter_pass_is_exact(
            horizontal,
            source_producer,
            source,
            RuntimeReadRole::FilterSource,
            false,
            edge,
        )
    {
        return None;
    }
    let horizontal_result = closed_filter_result(
        horizontal,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let (vertical_result, parameters, shadow) = validate_closed_shadow_tail(
        [vertical, colorize],
        horizontal,
        horizontal_result,
        blur,
        resources,
        working_format,
        edge,
    )?;
    let result = validate_closed_shadow_merge(
        merge,
        ClosedShadowMergeInputs {
            source_producer,
            source,
            colorize,
            shadow,
            result_spatial: parameters.spatial.result,
        },
        resources,
        working_format,
    )?;
    Some(ExecutableDropShadowFacts {
        horizontal: horizontal.id,
        vertical: vertical.id,
        colorize: colorize.id,
        merge: merge.id,
        source: source.id,
        horizontal_result: horizontal_result.id,
        vertical_result: vertical_result.id,
        shadow: shadow.id,
        result: result.id,
        blur: blur.clone(),
        parameters,
    })
}

fn validate_closed_shadow_tail<'plan>(
    passes: [&RuntimePass; 2],
    horizontal: &RuntimePass,
    horizontal_result: &'plan RuntimeResourceRequest,
    blur: &RuntimeBlur,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
    edge: RuntimeSamplingEdge,
) -> Option<(
    &'plan RuntimeResourceRequest,
    RuntimeDropShadow,
    &'plan RuntimeResourceRequest,
)> {
    let [vertical, colorize] = passes;
    let RuntimePassKind::BlurVertical(Some(vertical_blur)) = &vertical.kind else {
        return None;
    };
    if !runtime_blur_matches_axis(vertical_blur, blur, RuntimeBlurAxis::Vertical)
        || horizontal_result.expected_reads != 1
        || horizontal_result.last_use != vertical.id
        || !closed_unary_filter_pass_is_exact(
            vertical,
            horizontal.id,
            horizontal_result,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let vertical_result = closed_filter_result(
        vertical,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let RuntimePassKind::DropShadowColorize(Some(parameters)) = &colorize.kind else {
        return None;
    };
    if !runtime_drop_shadow_is_closed(parameters, blur)
        || vertical_result.expected_reads != 1
        || vertical_result.last_use != colorize.id
        || !closed_unary_filter_pass_is_exact(
            colorize,
            vertical.id,
            vertical_result,
            RuntimeReadRole::BlurredSourceAlpha,
            true,
            RuntimeSamplingEdge::TransparentBlack,
        )
    {
        return None;
    }
    let shadow = closed_filter_result(
        colorize,
        resources,
        RuntimeResourceRole::ShadowImage,
        parameters.spatial.result,
        working_format,
    )?;
    Some((vertical_result, *parameters, shadow))
}

struct ClosedShadowMergeInputs<'plan> {
    source_producer: RuntimePassId,
    source: &'plan RuntimeResourceRequest,
    colorize: &'plan RuntimePass,
    shadow: &'plan RuntimeResourceRequest,
    result_spatial: RuntimeSpatialDescriptor,
}

fn validate_closed_shadow_merge<'plan>(
    merge: &RuntimePass,
    inputs: ClosedShadowMergeInputs<'plan>,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
) -> Option<&'plan RuntimeResourceRequest> {
    let ClosedShadowMergeInputs {
        source_producer,
        source,
        colorize,
        shadow,
        result_spatial,
    } = inputs;
    let RuntimePassKind::Composite(Some(composite)) = &merge.kind else {
        return None;
    };
    if !matches!(composite.kind, RuntimeCompositeKind::DropShadow)
        || !composite.source_captured_before_outer_semantics
        || merge.dependencies.as_slice() != [source_producer, colorize.id]
        || merge.reads.len() != 2
        || !runtime_read_has_exact_facts(
            &merge.reads[0],
            RuntimeReadRole::CompositeSource,
            source,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || !runtime_read_has_exact_facts(
            &merge.reads[1],
            RuntimeReadRole::Shadow,
            shadow,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || !same_resource_set(&merge.releases, &[source.id, shadow.id])
        || merge.cache_keys.is_none()
        || shadow.expected_reads != 1
        || shadow.last_use != merge.id
    {
        return None;
    }
    closed_filter_result(
        merge,
        resources,
        RuntimeResourceRole::CompositeResult,
        result_spatial,
        working_format,
    )
}

fn closed_unary_filter_pass_is_exact(
    pass: &RuntimePass,
    producer: RuntimePassId,
    source: &RuntimeResourceRequest,
    role: RuntimeReadRole,
    releases_source: bool,
    edge: RuntimeSamplingEdge,
) -> bool {
    pass.dependencies.as_slice() == [producer]
        && pass.reads.len() == 1
        && runtime_read_has_exact_facts(
            &pass.reads[0],
            role,
            source,
            RuntimeSamplingFilter::Linear,
            edge,
        )
        && if releases_source {
            pass.releases.as_slice() == [source.id]
        } else {
            pass.releases.is_empty()
        }
        && pass.cache_keys.is_some()
}

fn closed_filter_source_is_exact(
    source: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    expected_reads: u32,
    last_use: RuntimePassId,
) -> bool {
    matches!(
        source.role,
        RuntimeResourceRole::BackdropCopy
            | RuntimeResourceRole::FilterIntermediate
            | RuntimeResourceRole::CompositeResult
    ) && source.format == RuntimeResourceFormat::Working(working_format)
        && source.expected_reads == expected_reads
        && source.last_use == last_use
}

fn closed_filter_result<'plan>(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    role: RuntimeResourceRole,
    spatial: RuntimeSpatialDescriptor,
    working_format: WorkingFormat,
) -> Option<&'plan RuntimeResourceRequest> {
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    let resource = resources.get(&result).copied()?;
    c08_resource_has_fixed_facts(
        resource,
        role,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    )
    .then_some(())
    .filter(|()| resource.spatial == spatial)?;
    Some(resource)
}

fn runtime_blur_is_closed(
    blur: &RuntimeBlur,
    require_nonzero: bool,
    edge: RuntimeSamplingEdge,
) -> bool {
    blur.standard_deviation.is_finite()
        && if require_nonzero {
            blur.standard_deviation > 0.0 && blur.support_radius > 0
        } else {
            blur.standard_deviation >= 0.0
        }
        && blur.edge == edge
        && blur.spatial.source.raster_scale.is_finite()
        && blur.spatial.result.raster_scale.is_finite()
}

fn runtime_drop_shadow_is_closed(shadow: &RuntimeDropShadow, blur: &RuntimeBlur) -> bool {
    shadow.standard_deviation == blur.standard_deviation
        && shadow.support_radius == blur.support_radius
        && shadow.spatial.source == blur.spatial.result
        && shadow.spatial.result == blur.spatial.result
        && shadow.edge == blur.edge
        && shadow.uses_source_alpha
        && shadow.uses_continuous_offset
        && shadow.retains_unchanged_source
        && shadow.offset.x().is_finite()
        && shadow.offset.y().is_finite()
        && [
            shadow.color.r(),
            shadow.color.g(),
            shadow.color.b(),
            shadow.color.a(),
        ]
        .into_iter()
        .all(f32::is_finite)
}

fn same_resource_set(actual: &[RuntimeResourceId], expected: &[RuntimeResourceId]) -> bool {
    actual.len() == expected.len()
        && actual.iter().copied().collect::<BTreeSet<_>>()
            == expected.iter().copied().collect::<BTreeSet<_>>()
}

fn validate_closed_composite(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    requires_isolated_source: bool,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    validate_closed_composite_with_parent_reads(
        pass,
        parent,
        source,
        resources,
        working_format,
        requires_isolated_source,
        1,
    )
}

fn runtime_composite_is_backdrop_inner(composite: &RuntimeComposite) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        parameters,
        outer_clips,
        ..
    } = &composite.kind
    else {
        return false;
    };
    *transform == Transform::identity()
        && parameters.destination_to_layer_local().affine() == Transform::identity()
        && parameters.opacity() == 1.0
        && parameters.blend() == BlendMode::Normal
        && outer_clips.is_empty()
        && parameters.alpha_mask().is_none()
}

fn runtime_composite_is_untransformed_outer(composite: &RuntimeComposite) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        parameters,
        ..
    } = &composite.kind
    else {
        return false;
    };
    *transform == Transform::identity()
        && parameters.destination_to_layer_local().affine() == Transform::identity()
}

fn validate_closed_composite_with_parent_reads(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    requires_isolated_source: bool,
    parent_expected_reads: u32,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    let RuntimePassKind::Composite(Some(composite)) = &pass.kind else {
        return None;
    };
    let result = validate_closed_composite_base(
        pass,
        parent,
        source,
        resources,
        working_format,
        composite,
        parent_expected_reads,
    )?;

    match &composite.kind {
        RuntimeCompositeKind::SpanSourceOver => {
            if requires_isolated_source
                || pass.reads.len() != 2
                || !matches!(
                    source.role,
                    RuntimeResourceRole::FilterIntermediate | RuntimeResourceRole::CompositeResult
                )
            {
                return None;
            }
            Some(None)
        }
        RuntimeCompositeKind::Layer {
            transform,
            parameters,
            clip,
            outer_clips,
            clip_coverage,
        } => {
            let layer = ClosedLayerCompositeView {
                composite,
                transform,
                parameters,
                clip,
                outer_clips,
                clip_coverage: *clip_coverage,
            };
            validate_closed_layer_composite(
                pass,
                parent,
                source,
                resources,
                requires_isolated_source,
                result,
                layer,
            )
        }
        RuntimeCompositeKind::DropShadow => None,
    }
}

fn validate_closed_composite_base(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    composite: &RuntimeComposite,
    parent_expected_reads: u32,
) -> Option<RuntimeResourceId> {
    if !composite.source_captured_before_outer_semantics || source.id == parent.current {
        return None;
    }
    let parent_resource = resources.get(&parent.current).copied()?;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    let mut expected_dependencies = vec![parent.producer, source_producer];
    if let RuntimeCompositeKind::Layer {
        clip_coverage: Some(coverage),
        ..
    } = &composite.kind
    {
        let coverage_resource = resources.get(coverage).copied()?;
        let RuntimeResourceProducer::Pass(coverage_producer) = coverage_resource.producer else {
            return None;
        };
        expected_dependencies.push(coverage_producer);
    }
    if pass.dependencies != expected_dependencies
        || pass.reads.len() < 2
        || !runtime_read_has_exact_facts(
            &pass.reads[0],
            RuntimeReadRole::CompositeParent,
            parent_resource,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::ClampToExtent,
        )
        || !runtime_read_has_exact_facts(
            &pass.reads[1],
            RuntimeReadRole::CompositeSource,
            source,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || parent_resource.format != RuntimeResourceFormat::Working(working_format)
        || source.format != RuntimeResourceFormat::Working(working_format)
        || !matches!(
            parent_resource.role,
            RuntimeResourceRole::RootWorkingImage
                | RuntimeResourceRole::IsolationWorkingImage
                | RuntimeResourceRole::CompositeResult
        )
        || parent_resource.expected_reads != parent_expected_reads
        || parent_resource.last_use != pass.id
        || source.expected_reads != 1
        || source.last_use != pass.id
    {
        return None;
    }
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    let result_resource = resources.get(&result).copied()?;
    c08_resource_has_fixed_facts(
        result_resource,
        RuntimeResourceRole::CompositeResult,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    )
    .then_some(())
    .filter(|()| result_resource.spatial == parent_resource.spatial)
    .map(|()| result)
}

struct ClosedLayerCompositeView<'a> {
    composite: &'a RuntimeComposite,
    transform: &'a Transform,
    parameters: &'a RuntimeLayerCompositeParameters,
    clip: &'a Option<Box<RenderClip>>,
    outer_clips: &'a [RuntimeOuterClip],
    clip_coverage: Option<RuntimeResourceId>,
}

fn validate_closed_layer_composite(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    requires_isolated_source: bool,
    result: RuntimeResourceId,
    layer: ClosedLayerCompositeView<'_>,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    let opacity = layer.parameters.opacity();
    let alpha_mask = layer.parameters.alpha_mask();
    if layer
        .transform
        .as_array()
        .iter()
        .any(|value| !value.is_finite())
        || !opacity.is_finite()
        || !(0.0..=1.0).contains(&opacity)
        || !runtime_affine_is_finite_and_non_singular(
            layer.parameters.destination_to_layer_local().affine(),
        )
        || layer.outer_clips.iter().any(|clip| {
            clip.transform
                .as_array()
                .iter()
                .any(|value| !value.is_finite())
        })
        || layer.parameters.has_clip() != (layer.clip.is_some() || !layer.outer_clips.is_empty())
        || layer.parameters.has_clip() != layer.clip_coverage.is_some()
        || !closed_composite_source_is_valid(source, requires_isolated_source, &layer)
    {
        return None;
    }
    let expected_read_count = 2usize
        .checked_add(usize::from(layer.clip_coverage.is_some()))?
        .checked_add(usize::from(alpha_mask.is_some()))?;
    if pass.reads.len() != expected_read_count {
        return None;
    }
    let next_read = validate_closed_clip_coverage(pass, resources, layer.clip_coverage, 2)?;
    validate_closed_alpha_mask(pass, resources, alpha_mask, next_read)?;
    Some(Some(ExecutableLayerCompositionFacts {
        pass: pass.id,
        parent: parent.current,
        source: source.id,
        clip_coverage: layer.clip_coverage,
        alpha_mask: alpha_mask.map(RuntimeResolvedAlphaMaskComposition::resource),
        result,
        composite: layer.composite.clone(),
    }))
}

fn closed_composite_source_is_valid(
    source: &RuntimeResourceRequest,
    requires_isolated_source: bool,
    layer: &ClosedLayerCompositeView<'_>,
) -> bool {
    (requires_isolated_source
        && matches!(
            source.role,
            RuntimeResourceRole::CompositeResult | RuntimeResourceRole::FilterIntermediate
        ))
        || (!requires_isolated_source
            && source.role == RuntimeResourceRole::FilterIntermediate
            && *layer.transform == Transform::identity()
            && layer.parameters.opacity() == 1.0
            && layer.parameters.blend() == BlendMode::Normal
            && layer.clip.is_none()
            && !layer.outer_clips.is_empty()
            && layer.parameters.alpha_mask().is_none())
}

fn validate_closed_clip_coverage(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    clip_coverage: Option<RuntimeResourceId>,
    next_read: usize,
) -> Option<usize> {
    let Some(coverage) = clip_coverage else {
        return Some(next_read);
    };
    let coverage_resource = resources.get(&coverage).copied()?;
    if !runtime_read_has_exact_facts(
        &pass.reads[next_read],
        RuntimeReadRole::ClipCoverage,
        coverage_resource,
        RuntimeSamplingFilter::Linear,
        RuntimeSamplingEdge::TransparentBlack,
    ) || coverage_resource.role != RuntimeResourceRole::ClipCoverage
        || coverage_resource.format != RuntimeResourceFormat::ClipCoverageRgba8Unorm
        || !matches!(coverage_resource.producer, RuntimeResourceProducer::Pass(_))
        || coverage_resource.import.is_some()
        || coverage_resource.expected_reads != 1
        || coverage_resource.last_use != pass.id
    {
        return None;
    }
    next_read.checked_add(1)
}

fn validate_closed_alpha_mask(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
    next_read: usize,
) -> Option<()> {
    let Some(mask) = alpha_mask else {
        return Some(());
    };
    let mask_resource = resources.get(&mask.resource()).copied()?;
    let Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) = &mask_resource.import else {
        return None;
    };
    let mask_filter = match mask.sampling().quality() {
        ShaderMaskQualityKey::Low => RuntimeSamplingFilter::Nearest,
        ShaderMaskQualityKey::Medium | ShaderMaskQualityKey::High => RuntimeSamplingFilter::Linear,
    };
    (runtime_read_has_exact_facts(
        &pass.reads[next_read],
        RuntimeReadRole::AlphaMask,
        mask_resource,
        mask_filter,
        RuntimeSamplingEdge::ClampToExtent,
    ) && mask_resource.role == RuntimeResourceRole::ImportedImage
        && mask_resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm
        && mask_resource.producer == RuntimeResourceProducer::Imported
        && upload.physical_size() == mask.image_dimensions()
        && mask_resource.spatial.device_extent == mask.image_dimensions()
        && mask.sampling() == ShaderMaskSamplingKey::new(upload.quality(), upload.extend())
        && RuntimeMaskTexelCenterFacts::try_new(mask.image_dimensions()).ok()
            == Some(mask.texel_center_facts())
        && mask_resource.expected_reads != 0
        && mask_resource.last_use >= pass.id)
        .then_some(())
}

fn c08_resource_has_fixed_facts(
    resource: &RuntimeResourceRequest,
    role: RuntimeResourceRole,
    format: RuntimeResourceFormat,
    producer: RuntimeResourceProducer,
) -> bool {
    resource.role == role
        && resource.format == format
        && resource.producer == producer
        && resource.import.is_none()
}

#[cfg(test)]
fn c08_read_is_exact(
    read: &RuntimeReadBinding,
    role: RuntimeReadRole,
    resource: RuntimeResourceId,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
    source_format: RuntimeResourceFormat,
) -> bool {
    let sampler_key = SamplerKey::new(
        shader_binding_role(role),
        source_format.shader_key(),
        match sampling_filter {
            RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
            RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
        },
        shader_sampling_edge(sampling_edge),
        None,
    );
    read.role == role
        && read.resource == resource
        && read.sampling_filter == sampling_filter
        && read.sampling_edge == sampling_edge
        && read.sampler_key == sampler_key
}

fn executable_vello_capture_facts(
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: &RuntimeVelloCapture,
    spatial: RuntimeSpatialDescriptor,
) -> Option<ExecutableVelloCaptureFacts> {
    let valid_work = match work {
        RuntimeVelloCapture::Span(span) => {
            !span.commands.commands.is_empty() && span.captured_before_outer_semantics
        }
        RuntimeVelloCapture::ClipCoverage(coverage) => {
            !coverage.elements.is_empty()
                && coverage.elements.iter().all(|element| {
                    element
                        .transform
                        .as_array()
                        .iter()
                        .all(|value| value.is_finite())
                })
        }
    };
    if !valid_work
        || spatial.device_extent.width() == 0
        || spatial.device_extent.height() == 0
        || !spatial.texel_origin.x().is_finite()
        || !spatial.texel_origin.y().is_finite()
        || !spatial.raster_scale.is_finite()
        || spatial.raster_scale <= 0.0
    {
        return None;
    }
    let expected_device_x = spatial.texel_origin.x() * spatial.raster_scale;
    let expected_device_y = spatial.texel_origin.y() * spatial.raster_scale;
    let tolerance = f64::EPSILON
        * spatial
            .raster_scale
            .abs()
            .max(expected_device_x.abs())
            .max(expected_device_y.abs())
            .max(1.0)
        * 8.0;
    if (expected_device_x - f64::from(spatial.device_origin.0)).abs() > tolerance
        || (expected_device_y - f64::from(spatial.device_origin.1)).abs() > tolerance
    {
        return None;
    }
    let grid_transform =
        Transform::translation(-spatial.texel_origin.x(), -spatial.texel_origin.y())
            .ok()?
            .then(Transform::scale(spatial.raster_scale, spatial.raster_scale).ok()?)
            .ok()?;
    let initial_transform = match work {
        RuntimeVelloCapture::Span(span) => match span.scope {
            RuntimeVelloSpanScope::CurrentParent => span
                .capture_transform
                .then(span.parent_to_surface)
                .ok()?
                .then(grid_transform)
                .ok()?,
            RuntimeVelloSpanScope::LayerSource => {
                span.capture_transform.then(grid_transform).ok()?
            }
        },
        RuntimeVelloCapture::ClipCoverage(_) => grid_transform,
    };
    Some(ExecutableVelloCaptureFacts {
        pass,
        target,
        work: work.clone(),
        initial_transform,
        antialiasing: work.antialiasing(),
        target_extent: spatial.device_extent,
        texel_origin: spatial.texel_origin,
        raster_scale: spatial.raster_scale,
    })
}

#[cfg(test)]
fn c08_executable_subset_observation(
    c08_commands: RenderCommands,
    expanded_graph_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C08ExecutableSubsetObservationForTest> {
    let direct_route = matches!(
        c08_commands.clone().plan_for(context),
        Ok(FramePlan::DirectVello(_))
    );
    let expanded_graph_plan = expanded_graph_commands.clone().plan_for(context).ok()?;
    let graph_route = matches!(&expanded_graph_plan, FramePlan::GpuGraph(_));
    let FramePlan::GpuGraph(expanded_graph) = expanded_graph_plan else {
        return None;
    };
    let c08_graph = super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
    let rgba = LoweredGraphPlan::try_lower_validated_graph(
        &c08_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )
    .ok()?;
    let bgra = LoweredGraphPlan::try_lower_validated_graph(
        &c08_graph,
        WorkingFormat::HighPrecision,
        Format::Bgra8,
        &capabilities,
    )
    .ok()?;
    let expanded_graph = LoweredGraphPlan::try_lower_validated_graph(
        &expanded_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )
    .ok()?;
    let rgba_preparable = C08PreparableGraph::try_from_lowered(rgba.clone()).ok()?;
    let bgra_preparable = C08PreparableGraph::try_from_lowered(bgra).ok()?;
    let rgba_subset = &rgba_preparable.execution;
    let bgra_subset = &bgra_preparable.execution;
    let accepts_exact_rgba_and_bgra = rgba_subset.working_format() == WorkingFormat::HighPrecision
        && rgba_subset.output_format() == Format::Rgba8
        && bgra_subset.working_format() == WorkingFormat::HighPrecision
        && bgra_subset.output_format() == Format::Bgra8
        && !rgba_subset.captures().is_empty()
        && rgba_subset.captures().len() == bgra_subset.captures().len()
        && rgba_subset.captures().iter().all(|capture| {
            capture.target_extent().width() > 0
                && capture.target_extent().height() > 0
                && capture.raster_scale().is_finite()
                && capture.raster_scale() > 0.0
        });

    Some(C08ExecutableSubsetObservationForTest {
        accepts_exact_rgba_and_bgra,
        rejects_every_other_pass_kind_and_composite_payload:
            c08_rejects_every_other_pass_kind_and_composite_payload(&rgba),
        rejects_missing_or_reordered_spine_passes: c08_rejects_missing_or_reordered_spine_passes(
            &rgba,
        ),
        rejects_malformed_dependencies_reads_results_and_releases: c08_rejects_malformed_bindings(
            &rgba,
        ),
        rejects_graph_outside_base_subset: C08PreparableGraph::try_from_lowered(expanded_graph)
            .is_err(),
        preserves_direct_and_graph_planner_routes: direct_route && graph_route,
    })
}

#[cfg(test)]
fn c09_executable_graph_observation(
    c08_commands: RenderCommands,
    c09_commands: RenderCommands,
    c10_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C09ExecutableGraphObservationForTest> {
    let c08_graph = super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
    let FramePlan::GpuGraph(c09_graph) = c09_commands.plan_for(context).ok()? else {
        return None;
    };
    let FramePlan::GpuGraph(c10_graph) = c10_commands.plan_for(context).ok()? else {
        return None;
    };

    let accepts_spine_and_layer_composition_for_all_formats =
        c09_accepts_all_working_and_output_formats(&c08_graph, &c09_graph);

    let c09_lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &c09_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let c09_closed = ClosedExecutableGraph::try_from_lowered(c09_lowered.clone()).ok()?;
    let layer_composition_reads_are_exact = c09_layer_composition_reads_are_exact(&c09_closed);

    let c10_lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &c10_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let rejects_actual_c10 = !matches!(
        PrePreparationGraphClassification::classify(c10_lowered),
        PrePreparationGraphClassification::ExactC09(_)
    );
    let layer_index = c09_lowered.passes.iter().position(|pass| {
        matches!(
            pass.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { .. },
                ..
            }))
        )
    })?;
    let capture_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::VelloCapture(Some(_))))?;
    let canonicalize_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::CanonicalizeCapture))?;
    let present_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::Present))?;
    let rejects = |invalid| ClosedExecutableGraph::try_from_lowered(invalid).is_err();

    let mut invalid_copy = c09_lowered.clone();
    invalid_copy.passes[layer_index].kind = RuntimePassKind::CopyBackdrop;
    let mut invalid_payload = c09_lowered.clone();
    invalid_payload.passes[layer_index].kind = RuntimePassKind::Composite(Some(RuntimeComposite {
        kind: RuntimeCompositeKind::DropShadow,
        source_captured_before_outer_semantics: true,
    }));
    let rejects_c10_plus_passes_and_payloads =
        rejects_actual_c10 && rejects(invalid_copy) && rejects(invalid_payload);

    let mut missing_capture = c09_lowered.clone();
    missing_capture.passes[capture_index].kind = RuntimePassKind::VelloCapture(None);
    let mut missing_composite = c09_lowered.clone();
    missing_composite.passes[layer_index].kind = RuntimePassKind::Composite(None);
    let rejects_missing_payloads = rejects(missing_capture) && rejects(missing_composite);

    let rejects_malformed_graph_facts = c09_rejects_malformed_graph_facts(
        &c09_lowered,
        capture_index,
        canonicalize_index,
        layer_index,
    );

    let mut unsupported_output = c09_lowered;
    unsupported_output.passes[present_index].result = RuntimeResultBinding::Output(Format::Bgra8);
    let rejects_unsupported_output_binding = rejects(unsupported_output);
    let preserves_exact_c09_dispatch = matches!(
        ExecutableGraphDispatchEligibility::try_classify(
            &c09_graph,
            Format::Rgba8,
            ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
            &capabilities,
        ),
        Ok(ExecutableGraphDispatchEligibility::ExactC09(_))
    );

    Some(C09ExecutableGraphObservationForTest {
        accepts_spine_and_layer_composition_for_all_formats,
        layer_composition_reads_are_exact,
        rejects_c10_plus_passes_and_payloads,
        rejects_missing_payloads,
        rejects_malformed_graph_facts,
        rejects_unsupported_output_binding,
        preserves_exact_c09_dispatch,
    })
}

#[cfg(test)]
fn c09_accepts_all_working_and_output_formats(
    c08_graph: &GpuRenderGraph,
    c09_graph: &GpuRenderGraph,
) -> bool {
    [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ]
    .into_iter()
    .all(|working_format| {
        [Format::Rgba8, Format::Bgra8]
            .into_iter()
            .all(|output_format| {
                let lower = |graph| {
                    LoweredGraphPlan::try_lower_for_dispatch_classification(
                        graph,
                        working_format,
                        output_format,
                    )
                    .ok()
                    .and_then(|lowered| ClosedExecutableGraph::try_from_lowered(lowered).ok())
                };
                lower(c08_graph).is_some_and(|closed| !closed.has_layer_composition())
                    && lower(c09_graph).is_some_and(|closed| closed.has_layer_composition())
            })
    })
}

#[cfg(test)]
fn c09_layer_composition_reads_are_exact(closed: &ClosedExecutableGraph) -> bool {
    !closed.facts.layer_compositions.is_empty()
        && closed.facts.layer_compositions.iter().all(|layer| {
            closed
                .lowered
                .passes
                .iter()
                .find(|pass| pass.id == layer.pass)
                .is_some_and(|pass| {
                    let mut expected = vec![
                        (RuntimeReadRole::CompositeParent, layer.parent),
                        (RuntimeReadRole::CompositeSource, layer.source),
                    ];
                    if let Some(coverage) = layer.clip_coverage {
                        expected.push((RuntimeReadRole::ClipCoverage, coverage));
                    }
                    if let Some(mask) = layer.alpha_mask {
                        expected.push((RuntimeReadRole::AlphaMask, mask));
                    }
                    pass.reads
                        .iter()
                        .map(|read| (read.role, read.resource))
                        .eq(expected)
                })
        })
}

#[cfg(test)]
fn c09_rejects_malformed_graph_facts(
    lowered: &LoweredGraphPlan,
    capture: usize,
    canonicalize: usize,
    layer: usize,
) -> bool {
    let mut malformed = Vec::new();
    let mut invalid = lowered.clone();
    invalid.passes[canonicalize].dependencies.clear();
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[layer].reads.swap(0, 1);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[layer].result =
        RuntimeResultBinding::Resource(invalid.passes[layer].reads[0].resource);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[layer].releases.clear();
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.resources[0].expected_reads = invalid.resources[0].expected_reads.saturating_add(1);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes.swap(capture, canonicalize);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.resources[1].id = invalid.resources[0].id;
    malformed.push(invalid);
    malformed
        .into_iter()
        .all(|invalid| ClosedExecutableGraph::try_from_lowered(invalid).is_err())
}

#[cfg(test)]
fn lower_authored_c10_graph_for_test(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    working_format: WorkingFormat,
    output_format: Format,
    capabilities: &DeviceCapabilities,
) -> Option<(GpuRenderGraph, LoweredGraphPlan)> {
    let graph = super::frame::authored_filter_graph_for_test(filters, commands, context).ok()?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        output_format,
        capabilities,
    )
    .ok()?;
    Some((graph, lowered))
}

#[cfg(test)]
fn c10_executable_graph_observation(
    color_filters: Vec<super::FilterList>,
    blur_filters: Vec<super::FilterList>,
    shadow_filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C10ExecutableGraphObservationForTest> {
    let (color_graph, color_lowered) = lower_authored_c10_graph_for_test(
        color_filters.clone(),
        commands.clone(),
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let (_, blur_lowered) = lower_authored_c10_graph_for_test(
        blur_filters,
        commands.clone(),
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let (_, shadow_lowered) = lower_authored_c10_graph_for_test(
        shadow_filters,
        commands.clone(),
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let mut accepts_spine_composition_and_color_for_all_formats = true;
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let (_, lowered) = lower_authored_c10_graph_for_test(
                color_filters.clone(),
                commands.clone(),
                context,
                working_format,
                output_format,
                &capabilities,
            )?;
            accepts_spine_composition_and_color_for_all_formats &= c10_plan_is_closed(lowered);
        }
    }

    let color_pass_indices = color_lowered
        .passes
        .iter()
        .enumerate()
        .filter_map(|(index, pass)| {
            matches!(pass.kind, RuntimePassKind::ColorFilter(Some(_))).then_some(index)
        })
        .collect::<Vec<_>>();
    let accepts_multiple_ordered_color_runs = color_pass_indices.len() == color_filters.len()
        && color_pass_indices.len() > 1
        && c10_plan_is_closed(color_lowered.clone());
    let first_color = *color_pass_indices.first()?;
    let rejects_empty_missing_and_malformed_color_facts =
        c10_rejects_malformed_color_facts(&color_lowered, &color_pass_indices)?;

    let mut copy = color_lowered.clone();
    copy.passes[first_color].kind = RuntimePassKind::CopyBackdrop;
    let rejects_copy_blur_shadow_and_drop_shadow_composite = !c10_plan_is_closed(copy)
        && !c10_plan_is_closed(blur_lowered)
        && !c10_plan_is_closed(shadow_lowered);

    let mut unsupported_output = color_lowered;
    unsupported_output.output_format = Format::Bgra8;
    let rejects_unsupported_output = !c10_plan_is_closed(unsupported_output);
    let preserves_public_c09_dispatch_boundary = matches!(
        ExecutableGraphDispatchEligibility::try_classify(
            &color_graph,
            Format::Rgba8,
            ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
            &capabilities,
        ),
        Ok(ExecutableGraphDispatchEligibility::FuturePasses)
    );

    Some(C10ExecutableGraphObservationForTest {
        accepts_spine_composition_and_color_for_all_formats,
        accepts_multiple_ordered_color_runs,
        rejects_empty_missing_and_malformed_color_facts,
        rejects_copy_blur_shadow_and_drop_shadow_composite,
        rejects_unsupported_output,
        preserves_public_c09_dispatch_boundary,
    })
}

#[cfg(test)]
fn c10_plan_is_closed(lowered: LoweredGraphPlan) -> bool {
    ClosedExecutableGraph::try_from_lowered(lowered)
        .ok()
        .is_some_and(|closed| C10PreparableGraph::try_from_closed(closed).is_ok())
}

#[cfg(test)]
fn c10_rejects_malformed_color_facts(
    lowered: &LoweredGraphPlan,
    color_passes: &[usize],
) -> Option<bool> {
    let first = *color_passes.first()?;
    let mut malformed = Vec::new();
    let mut invalid = lowered.clone();
    invalid.passes[first].kind = RuntimePassKind::ColorFilter(None);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::ColorFilter(Some(filter)) = &mut invalid.passes[first].kind else {
        return None;
    };
    filter.operations.clear();
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[first].dependencies.clear();
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[first].reads[0].sampling_filter = RuntimeSamplingFilter::Linear;
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    let source = invalid.passes[first].reads[0].resource;
    invalid.passes[first].result = RuntimeResultBinding::Resource(source);
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[first].releases.clear();
    malformed.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimeResultBinding::Resource(result) = invalid.passes[first].result else {
        return None;
    };
    let result_index = invalid
        .resources
        .iter()
        .position(|resource| resource.id == result)?;
    invalid.resources[result_index].spatial.device_origin.0 = invalid.resources[result_index]
        .spatial
        .device_origin
        .0
        .checked_add(1)?;
    malformed.push(invalid);
    if color_passes.len() > 1 {
        let mut invalid = lowered.clone();
        invalid.passes.swap(color_passes[0], color_passes[1]);
        malformed.push(invalid);
    }
    Some(malformed.into_iter().all(|plan| !c10_plan_is_closed(plan)))
}

#[cfg(test)]
fn c10_spatial_observation(spatial: RuntimeSpatialDescriptor) -> C10ColorSpatialObservationForTest {
    C10ColorSpatialObservationForTest {
        logical_bounds: [
            spatial.logical_bounds.x(),
            spatial.logical_bounds.y(),
            spatial.logical_bounds.width(),
            spatial.logical_bounds.height(),
        ],
        device_origin: spatial.device_origin,
        device_extent: spatial.device_extent,
        texel_origin: spatial.texel_origin,
        raster_scale: spatial.raster_scale,
    }
}

#[cfg(test)]
fn color_filter_graph_observation(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<ColorFilterGraphObservationForTest> {
    let (_, lowered) = lower_authored_c10_graph_for_test(
        filters,
        commands,
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let preparable =
        C10PreparableGraph::try_from_closed(ClosedExecutableGraph::try_from_lowered(lowered).ok()?)
            .ok()?;
    let closed = &preparable.closed;
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let color_passes = closed
        .lowered
        .passes
        .iter()
        .filter(|pass| matches!(pass.kind, RuntimePassKind::ColorFilter(Some(_))))
        .collect::<Vec<_>>();
    let mut operation_tags_by_run = Vec::with_capacity(color_passes.len());
    let mut first_source_spatial = None;
    let mut every_run_has_one_source_and_distinct_result = !color_passes.is_empty();
    let mut every_run_preserves_exact_spatial_descriptor = !color_passes.is_empty();
    let mut every_operation_retains_one_clamp = !color_passes.is_empty();
    let mut current_resource_advances_after_each_run = !color_passes.is_empty();
    let mut dependencies_and_last_use_are_exact = !color_passes.is_empty();
    let mut previous_result = None;

    for pass in &color_passes {
        let RuntimePassKind::ColorFilter(Some(filter)) = &pass.kind else {
            return None;
        };
        let read = pass.reads.first()?;
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let source = resources.get(&read.resource).copied()?;
        let result_resource = resources.get(&result).copied()?;
        first_source_spatial.get_or_insert_with(|| c10_spatial_observation(source.spatial));
        every_run_has_one_source_and_distinct_result &= pass.reads.len() == 1
            && read.resource != result
            && read.role == RuntimeReadRole::FilterSource
            && read.sampling_filter == RuntimeSamplingFilter::Nearest
            && read.sampling_edge == RuntimeSamplingEdge::ClampToExtent;
        every_run_preserves_exact_spatial_descriptor &= source.spatial == result_resource.spatial
            && filter.spatial.source == source.spatial
            && filter.spatial.result == result_resource.spatial;
        every_operation_retains_one_clamp &= !filter.operations.is_empty()
            && filter.operations.iter().all(|operation| {
                operation.clamp_boundary
                    == RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
            });
        if let Some(previous_result) = previous_result {
            current_resource_advances_after_each_run &= read.resource == previous_result;
        }
        previous_result = Some(result);
        dependencies_and_last_use_are_exact &= matches!(
            source.producer,
            RuntimeResourceProducer::Pass(producer)
                if pass.dependencies.as_slice() == [producer]
        ) && source.expected_reads == 1
            && source.last_use == pass.id
            && pass.releases.as_slice() == [read.resource];
        operation_tags_by_run.push(
            filter
                .operations
                .iter()
                .map(|operation| runtime_color_operation_observation_for_test(operation).tag)
                .collect(),
        );
    }

    Some(ColorFilterGraphObservationForTest {
        operation_tags_by_run,
        first_source_spatial,
        every_run_has_one_source_and_distinct_result,
        every_run_preserves_exact_spatial_descriptor,
        every_operation_retains_one_clamp,
        current_resource_advances_after_each_run,
        dependencies_and_last_use_are_exact,
        closed_color_facts_match_runtime_passes: preparable.color_filters().len()
            == color_passes.len()
            && !color_passes.is_empty()
            && closed.facts.proves_exact_facts_for(&closed.lowered),
    })
}

#[cfg(test)]
fn mixed_color_unsupported_diagnostic_observation(
    color_filters: Vec<super::FilterList>,
    mixed_filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<MixedColorUnsupportedDiagnosticObservationForTest> {
    let (color_graph, _) = lower_authored_c10_graph_for_test(
        color_filters,
        commands.clone(),
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let (mixed_graph, mixed_lowered) = lower_authored_c10_graph_for_test(
        mixed_filters,
        commands,
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let color_diagnostic = super::renderer::unsupported_graph_diagnostic_for_test(
        &color_graph,
        Format::Rgba8,
        &capabilities,
    )
    .ok()??;
    let mixed_diagnostic = super::renderer::unsupported_graph_diagnostic_for_test(
        &mixed_graph,
        Format::Rgba8,
        &capabilities,
    )
    .ok()??;
    Some(MixedColorUnsupportedDiagnosticObservationForTest {
        pure_color_retains_gpu_color_diagnostic: color_diagnostic
            == super::UnsupportedPrimitive::new(
                super::PrimitiveFamily::Filters,
                super::PrimitiveOperation::GpuColorFilterExecution,
            ),
        color_then_blur_reports_gpu_blur_diagnostic: mixed_diagnostic
            == super::UnsupportedPrimitive::new(
                super::PrimitiveFamily::Filters,
                super::PrimitiveOperation::GpuBlurFilterExecution,
            ),
        mixed_graph_stays_outside_c10_preparation: ClosedExecutableGraph::try_from_lowered(
            mixed_lowered,
        )
        .ok()
        .is_none_or(|closed| C10PreparableGraph::try_from_closed(closed).is_err()),
    })
}

#[cfg(test)]
fn c11_executable_graph_observation(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C11ExecutableGraphObservationForTest> {
    let (_, lowered) = lower_authored_c10_graph_for_test(
        filters.clone(),
        commands.clone(),
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let preparable = c11_preparable_graph_for_test(lowered.clone()).ok()?;
    let accepts_color_blur_and_drop_shadow_for_all_formats =
        c11_accepts_all_formats(&filters, &commands, context, &capabilities)?;
    let preserves_ordered_nonzero_filter_steps =
        c11_filter_step_shape_is_exact(&preparable.closed.facts);
    let stale_dependency = RuntimePassId(
        lowered
            .passes
            .get(C11PassIndices::try_new(&lowered)?.color)?
            .id
            .0
            .stale_generation_for_test()?,
    );
    let malformed = c11_malformed_plan_observation(&lowered, stale_dependency)?;
    let resources = ResourceManager::new(super::ResourceCacheBudget::DISABLED);
    let cache = DevicePassCache::new();
    let resources_before = resources.observation_for_test();
    let cache_before = cache.counts_for_test();
    let valid_classification = c11_preparable_graph_for_test(lowered.clone()).is_ok();
    let invalid_classification = malformed
        .all_invalid
        .iter()
        .cloned()
        .all(|plan| c11_preparable_graph_for_test(plan).is_err());
    Some(C11ExecutableGraphObservationForTest {
        accepts_color_blur_and_drop_shadow_for_all_formats,
        preserves_ordered_nonzero_filter_steps,
        rejects_empty_missing_and_malformed_spatial_facts: malformed.empty_missing_spatial,
        rejects_wrong_axes_inputs_edges_and_aliases: malformed.axes_inputs_edges_aliases,
        rejects_copy_backdrop_stale_forward_and_c12_plus: malformed.copy_stale_forward_c12,
        rejects_before_resource_acquisition: valid_classification
            && invalid_classification
            && resources.observation_for_test() == resources_before
            && cache.counts_for_test() == cache_before,
    })
}

#[cfg(test)]
fn c11_accepts_all_formats(
    filters: &[super::FilterList],
    commands: &RenderCommands,
    context: FrameContext,
    capabilities: &DeviceCapabilities,
) -> Option<bool> {
    let mut accepts = true;
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let (_, lowered) = lower_authored_c10_graph_for_test(
                filters.to_vec(),
                commands.clone(),
                context,
                working_format,
                output_format,
                capabilities,
            )?;
            accepts &= c11_plan_is_closed(lowered);
        }
    }
    Some(accepts)
}

#[cfg(test)]
fn c11_filter_step_shape_is_exact(facts: &ClosedExecutableGraphFacts) -> bool {
    matches!(
        facts.filter_steps.as_slice(),
        [
            ExecutableFilterStepFacts::Color(_),
            ExecutableFilterStepFacts::Blur { .. },
            ExecutableFilterStepFacts::DropShadow { .. },
            ExecutableFilterStepFacts::Color(_),
        ]
    ) && facts.color_filters.len() == 2
        && facts.blurs.len() == 1
        && facts.drop_shadows.len() == 1
        && facts
            .blurs
            .iter()
            .all(|blur| blur.blur.standard_deviation > 0.0)
}

#[cfg(test)]
struct C11MalformedPlanObservation {
    empty_missing_spatial: bool,
    axes_inputs_edges_aliases: bool,
    copy_stale_forward_c12: bool,
    all_invalid: Vec<LoweredGraphPlan>,
}

#[cfg(test)]
fn c11_malformed_plan_observation(
    lowered: &LoweredGraphPlan,
    stale_dependency: RuntimePassId,
) -> Option<C11MalformedPlanObservation> {
    let indices = C11PassIndices::try_new(lowered)?;
    let empty_missing_spatial = c11_empty_missing_spatial_plans(lowered, indices)?;
    let axes_inputs_edges_aliases = c11_axes_inputs_edges_alias_plans(lowered, indices)?;
    let copy_stale_forward_c12 = c11_copy_stale_forward_plans(lowered, indices, stale_dependency)?;
    let invalid =
        |plans: &[LoweredGraphPlan]| plans.iter().cloned().all(|plan| !c11_plan_is_closed(plan));
    let all_invalid = empty_missing_spatial
        .iter()
        .chain(&axes_inputs_edges_aliases)
        .chain(&copy_stale_forward_c12)
        .cloned()
        .collect::<Vec<_>>();
    Some(C11MalformedPlanObservation {
        empty_missing_spatial: invalid(&empty_missing_spatial),
        axes_inputs_edges_aliases: invalid(&axes_inputs_edges_aliases),
        copy_stale_forward_c12: invalid(&copy_stale_forward_c12),
        all_invalid,
    })
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct C11PassIndices {
    color: usize,
    blur_horizontal: usize,
    shadow_horizontal: usize,
    shadow_colorize: usize,
    shadow_merge: usize,
}

#[cfg(test)]
impl C11PassIndices {
    fn try_new(plan: &LoweredGraphPlan) -> Option<Self> {
        let find = |predicate: fn(&RuntimePassKind) -> bool| {
            plan.passes.iter().position(|pass| predicate(&pass.kind))
        };
        Some(Self {
            color: find(|kind| matches!(kind, RuntimePassKind::ColorFilter(Some(_))))?,
            blur_horizontal: find(|kind| {
                matches!(
                    kind,
                    RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                        input: RuntimeBlurInput::Rgba,
                        ..
                    }))
                )
            })?,
            shadow_horizontal: find(|kind| {
                matches!(
                    kind,
                    RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                        input: RuntimeBlurInput::SourceAlpha,
                        ..
                    }))
                )
            })?,
            shadow_colorize: find(|kind| {
                matches!(kind, RuntimePassKind::DropShadowColorize(Some(_)))
            })?,
            shadow_merge: find(|kind| {
                matches!(
                    kind,
                    RuntimePassKind::Composite(Some(RuntimeComposite {
                        kind: RuntimeCompositeKind::DropShadow,
                        ..
                    }))
                )
            })?,
        })
    }
}

#[cfg(test)]
fn c11_empty_missing_spatial_plans(
    lowered: &LoweredGraphPlan,
    indices: C11PassIndices,
) -> Option<Vec<LoweredGraphPlan>> {
    let mut plans = Vec::new();
    let mut invalid = lowered.clone();
    let RuntimePassKind::ColorFilter(Some(filter)) = &mut invalid.passes[indices.color].kind else {
        return None;
    };
    filter.operations.clear();
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[indices.blur_horizontal].kind = RuntimePassKind::BlurHorizontal(None);
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[indices.shadow_colorize].kind = RuntimePassKind::DropShadowColorize(None);
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::BlurHorizontal(Some(blur)) =
        &mut invalid.passes[indices.blur_horizontal].kind
    else {
        return None;
    };
    blur.standard_deviation = 0.0;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::BlurHorizontal(Some(blur)) =
        &mut invalid.passes[indices.blur_horizontal].kind
    else {
        return None;
    };
    blur.spatial.result.device_origin.0 = blur.spatial.result.device_origin.0.checked_add(1)?;
    plans.push(invalid);
    Some(plans)
}

#[cfg(test)]
fn c11_axes_inputs_edges_alias_plans(
    lowered: &LoweredGraphPlan,
    indices: C11PassIndices,
) -> Option<Vec<LoweredGraphPlan>> {
    let mut plans = Vec::new();
    for mutate in [
        |blur: &mut RuntimeBlur| blur.axis = RuntimeBlurAxis::Vertical,
        |blur: &mut RuntimeBlur| blur.input = RuntimeBlurInput::SourceAlpha,
        |blur: &mut RuntimeBlur| {
            blur.edge = RuntimeSamplingEdge::SemanticBorderMirror(Rect::new(0.0, 0.0, 1.0, 1.0));
        },
    ] {
        let mut invalid = lowered.clone();
        let RuntimePassKind::BlurHorizontal(Some(blur)) =
            &mut invalid.passes[indices.blur_horizontal].kind
        else {
            return None;
        };
        mutate(blur);
        plans.push(invalid);
    }
    let mut invalid = lowered.clone();
    let RuntimePassKind::BlurHorizontal(Some(blur)) =
        &mut invalid.passes[indices.shadow_horizontal].kind
    else {
        return None;
    };
    blur.input = RuntimeBlurInput::Rgba;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let source = invalid.passes[indices.blur_horizontal]
        .reads
        .first()?
        .resource;
    invalid.passes[indices.blur_horizontal].result = RuntimeResultBinding::Resource(source);
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::DropShadowColorize(Some(shadow)) =
        &mut invalid.passes[indices.shadow_colorize].kind
    else {
        return None;
    };
    shadow.uses_source_alpha = false;
    plans.push(invalid);
    Some(plans)
}

#[cfg(test)]
fn c11_copy_stale_forward_plans(
    lowered: &LoweredGraphPlan,
    indices: C11PassIndices,
    stale_dependency: RuntimePassId,
) -> Option<Vec<LoweredGraphPlan>> {
    let mut plans = Vec::new();
    let mut invalid = lowered.clone();
    invalid.passes[indices.blur_horizontal].kind = RuntimePassKind::CopyBackdrop;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let forward = invalid
        .passes
        .get(indices.blur_horizontal.checked_add(1)?)?
        .id;
    invalid.passes[indices.blur_horizontal].dependencies = vec![forward];
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[indices.shadow_merge].dependencies =
        vec![invalid.passes[indices.shadow_colorize].id];
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[indices.blur_horizontal].dependencies = vec![stale_dependency];
    plans.push(invalid);
    Some(plans)
}

#[cfg(test)]
fn c11_plan_is_closed(lowered: LoweredGraphPlan) -> bool {
    c11_preparable_graph_for_test(lowered).is_ok()
}

#[cfg(test)]
fn c11_filter_graph_observation(
    filters: Vec<super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C11FilterGraphObservationForTest> {
    let (_, lowered) = lower_authored_c10_graph_for_test(
        filters,
        commands,
        context,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )?;
    let preparable = c11_preparable_graph_for_test(lowered).ok()?;
    let closed = &preparable.closed;
    Some(C11FilterGraphObservationForTest {
        pass_order: c11_filter_pass_tags(&closed.facts),
        ordinary_blur_uses_transparent_black: c11_blur_edges_are_exact(closed),
        drop_shadow_uses_source_alpha_and_continuous_offset: c11_shadow_facts_are_exact(closed),
        spatial_mappings_are_exact: c11_spatial_mappings_are_exact(closed),
        sources_and_results_are_distinct: c11_sources_and_results_are_distinct(closed),
        source_alpha_fanout_reads_original_twice: c11_shadow_fanout_is_exact(closed),
        original_source_releases_only_after_merge: c11_shadow_releases_are_exact(closed),
        dependencies_and_last_use_are_exact: c11_dependencies_and_lifetimes_are_exact(closed),
    })
}

#[cfg(test)]
fn c12_executable_graph_observation(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C12ExecutableGraphObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let mut accepts_bounded_top_level_backdrop = true;
    let mut selected = None;
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
                &graph,
                working_format,
                output_format,
            )
            .ok()?;
            accepts_bounded_top_level_backdrop &= c12_plan_is_exact(lowered.clone());
            if selected.is_none() {
                selected = Some(lowered);
            }
        }
    }
    let lowered = selected?;
    let resources = ResourceManager::new(super::ResourceCacheBudget::DISABLED);
    let cache = DevicePassCache::new();
    let resources_before = resources.observation_for_test();
    let cache_before = cache.counts_for_test();
    accepts_bounded_top_level_backdrop &= matches!(
        ExecutableGraphDispatchEligibility::try_classify(
            &graph,
            Format::Rgba8,
            ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
            &capabilities,
        ),
        Ok(ExecutableGraphDispatchEligibility::ExactC12(_))
    );
    let rejects_outside_bounded_subset = c12_malformed_plans(&lowered)?
        .into_iter()
        .all(|plan| !c12_plan_is_exact(plan));
    Some(C12ExecutableGraphObservationForTest {
        accepts_bounded_top_level_backdrop,
        rejects_outside_bounded_subset,
        rejects_before_resource_acquisition: resources.observation_for_test() == resources_before
            && cache.counts_for_test() == cache_before,
    })
}

#[cfg(test)]
fn c12_plan_is_exact(lowered: LoweredGraphPlan) -> bool {
    matches!(
        PrePreparationGraphClassification::classify(lowered),
        PrePreparationGraphClassification::ExactC12(_)
    )
}

#[cfg(test)]
fn c12_malformed_plans(lowered: &LoweredGraphPlan) -> Option<Vec<LoweredGraphPlan>> {
    let copy = lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::CopyBackdrop))?;
    let mirror = lowered.passes.iter().position(|pass| {
        matches!(
            pass.kind,
            RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                edge: RuntimeSamplingEdge::SemanticBorderMirror(_),
                ..
            }))
        )
    })?;
    let outer = lowered.passes.iter().rposition(|pass| {
        matches!(
            pass.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { .. },
                ..
            }))
        )
    })?;
    let mut plans = Vec::new();
    let mut invalid = lowered.clone();
    invalid.passes[copy].dependencies.clear();
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[copy].reads[0].role = RuntimeReadRole::FilterSource;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    invalid.passes[mirror].kind = RuntimePassKind::CopyBackdrop;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::BlurHorizontal(Some(blur)) = &mut invalid.passes[mirror].kind else {
        return None;
    };
    blur.edge = RuntimeSamplingEdge::TransparentBlack;
    plans.push(invalid);
    let mut invalid = lowered.clone();
    let RuntimePassKind::Composite(Some(RuntimeComposite {
        kind: RuntimeCompositeKind::Layer { transform, .. },
        ..
    })) = &mut invalid.passes[outer].kind
    else {
        return None;
    };
    *transform = Transform::translation(1.0, 0.0).ok()?;
    plans.push(invalid);
    Some(plans)
}

#[cfg(test)]
fn c12_backdrop_graph_observation(
    commands: RenderCommands,
    context: FrameContext,
    _capabilities: DeviceCapabilities,
) -> Option<C12BackdropGraphObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let PrePreparationGraphClassification::ExactC12(preparable) =
        PrePreparationGraphClassification::classify(lowered)
    else {
        return None;
    };
    let [backdrop] = preparable.closed.facts.backdrops.as_slice() else {
        return None;
    };
    let positions = preparable
        .closed
        .lowered
        .passes
        .iter()
        .enumerate()
        .map(|(position, pass)| (pass.id, position))
        .collect::<BTreeMap<_, _>>();
    let copy_position = *positions.get(&backdrop.copy)?;
    let filter_passes = backdrop_filter_passes(&backdrop.filter_steps);
    let backdrop_position = *positions.get(&backdrop.backdrop_composite)?;
    let foreground_position = backdrop
        .foreground_composite
        .and_then(|pass| positions.get(&pass).copied());
    let outer_position = *positions.get(&backdrop.outer_composite)?;
    let copy_pass = preparable.closed.lowered.passes.get(copy_position)?;
    let reads_completed_parent_once = copy_pass.reads.len() == 1
        && copy_pass.reads[0].role == RuntimeReadRole::CompletedParent
        && copy_pass.reads[0].resource == backdrop.completed_parent;
    let backdrop_layer = preparable
        .closed
        .facts
        .layer_compositions
        .iter()
        .find(|layer| layer.pass == backdrop.backdrop_composite)?;
    let post_filter_clip_precedes_foreground = matches!(
        &backdrop_layer.composite.kind,
        RuntimeCompositeKind::Layer {
            clip: Some(_),
            clip_coverage: Some(_),
            ..
        }
    ) && filter_passes.iter().all(|pass| {
        positions
            .get(pass)
            .is_some_and(|position| *position < backdrop_position)
    }) && foreground_position
        .is_some_and(|position| backdrop_position < position);
    let later_sibling_depends_on_completed_group = preparable
        .closed
        .lowered
        .passes
        .iter()
        .skip(outer_position.saturating_add(1))
        .any(|pass| {
            pass.dependencies.contains(&backdrop.outer_composite)
                && pass
                    .reads
                    .iter()
                    .any(|read| read.resource == backdrop.result)
        });
    Some(C12BackdropGraphObservationForTest {
        closed_subset_receipt: preparable.proves_closed_backdrop_facts(),
        reads_completed_parent_once,
        copy_precedes_authored_filters: filter_passes.iter().all(|pass| {
            positions
                .get(pass)
                .is_some_and(|position| copy_position < *position)
        }),
        post_filter_clip_precedes_foreground,
        foreground_precedes_outer_composition: foreground_position
            .is_some_and(|position| position < outer_position),
        later_sibling_depends_on_completed_group,
    })
}

#[cfg(test)]
fn c12_backdrop_filter_chain_observation(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C12BackdropFilterChainObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.clone().plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )
    .ok()?;
    let PrePreparationGraphClassification::ExactC12(preparable) =
        PrePreparationGraphClassification::classify(lowered)
    else {
        return None;
    };
    let facts = &preparable.closed.facts;
    let pass_by_id = preparable
        .closed
        .lowered
        .passes
        .iter()
        .map(|pass| (pass.id, pass))
        .collect::<BTreeMap<_, _>>();
    let blur_edges = facts
        .blurs
        .iter()
        .map(|blur| (blur.horizontal, blur.vertical, blur.blur.edge))
        .chain(
            facts
                .drop_shadows
                .iter()
                .map(|shadow| (shadow.horizontal, shadow.vertical, shadow.blur.edge)),
        )
        .collect::<Vec<_>>();
    let semantic_bounds_are_exact = !blur_edges.is_empty()
        && blur_edges.iter().all(|(horizontal, vertical, edge)| {
            let RuntimeSamplingEdge::SemanticBorderMirror(bounds) = edge else {
                return false;
            };
            BlurEdgeParameterBytes::try_from_semantic_bounds(*bounds).is_ok()
                && [horizontal, vertical].into_iter().all(|pass| {
                    pass_by_id.get(pass).is_some_and(|pass| {
                        matches!(
                            pass.reads.as_slice(),
                            [RuntimeReadBinding {
                                sampling_edge:
                                    RuntimeSamplingEdge::SemanticBorderMirror(read_bounds),
                                ..
                            }] if read_bounds == bounds
                        )
                    })
                })
        });
    let requests = c12_backdrop_blur_cache_requests_for_test(
        commands,
        context,
        capabilities,
        WorkingFormat::HighPrecision,
    )
    .ok()?;
    let every_mirrored_stage_is_realizable = requests.iter().all(|request| {
        super::shader::c12_backdrop_blur_pass_key_facts_for_test(
            request.keys.samplers(),
            request.keys.layout(),
            request.keys.shader(),
            request.keys.pipeline(),
        )
        .is_some_and(|facts| facts.has_only_linear_mirror_sampler && facts.has_exact_data_bindings)
    });
    Some(C12BackdropFilterChainObservationForTest {
        pass_order: c11_filter_pass_tags(facts),
        every_backdrop_blur_uses_mirror: !facts.blurs.is_empty()
            && facts
                .blurs
                .iter()
                .all(|blur| matches!(blur.blur.edge, RuntimeSamplingEdge::SemanticBorderMirror(_))),
        source_alpha_blur_uses_mirror: !facts.drop_shadows.is_empty()
            && facts.drop_shadows.iter().all(|shadow| {
                shadow.blur.input == RuntimeBlurInput::SourceAlpha
                    && matches!(
                        shadow.blur.edge,
                        RuntimeSamplingEdge::SemanticBorderMirror(_)
                    )
            }),
        every_color_operation_retains_one_clamp: !facts.color_filters.is_empty()
            && facts.color_filters.iter().all(|filter| {
                !filter.filter.operations.is_empty()
                    && filter.filter.operations.iter().all(|operation| {
                        operation.clamp_boundary
                            == RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
                    })
            }),
        semantic_bounds_are_exact,
        every_mirrored_stage_is_realizable,
    })
}

fn backdrop_filter_passes(steps: &[ExecutableFilterStepFacts]) -> Vec<RuntimePassId> {
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

#[cfg(test)]
fn c11_filter_pass_tags(facts: &ClosedExecutableGraphFacts) -> Vec<C11FilterPassTagForTest> {
    let mut tags = Vec::new();
    for step in &facts.filter_steps {
        match step {
            ExecutableFilterStepFacts::Color(_) => {
                tags.push(C11FilterPassTagForTest::Color);
            }
            ExecutableFilterStepFacts::Blur { .. } => {
                tags.extend([
                    C11FilterPassTagForTest::BlurHorizontalRgba,
                    C11FilterPassTagForTest::BlurVerticalRgba,
                ]);
            }
            ExecutableFilterStepFacts::DropShadow { .. } => {
                tags.extend([
                    C11FilterPassTagForTest::BlurHorizontalSourceAlpha,
                    C11FilterPassTagForTest::BlurVerticalSourceAlpha,
                    C11FilterPassTagForTest::DropShadowColorize,
                    C11FilterPassTagForTest::DropShadowMerge,
                ]);
            }
        }
    }
    tags
}

#[cfg(test)]
fn c11_blur_edges_are_exact(closed: &ClosedExecutableGraph) -> bool {
    !closed.facts.blurs.is_empty()
        && closed.facts.blurs.iter().all(|facts| {
            facts.blur.edge == RuntimeSamplingEdge::TransparentBlack
                && [facts.horizontal, facts.vertical].into_iter().all(|pass| {
                    closed
                        .lowered
                        .passes
                        .iter()
                        .find(|candidate| candidate.id == pass)
                        .is_some_and(|pass| {
                            pass.reads.len() == 1
                                && pass.reads[0].sampling_edge
                                    == RuntimeSamplingEdge::TransparentBlack
                        })
                })
        })
}

#[cfg(test)]
fn c11_shadow_facts_are_exact(closed: &ClosedExecutableGraph) -> bool {
    !closed.facts.drop_shadows.is_empty()
        && closed.facts.drop_shadows.iter().all(|facts| {
            facts.blur.input == RuntimeBlurInput::SourceAlpha
                && facts.parameters.uses_source_alpha
                && facts.parameters.uses_continuous_offset
                && facts.parameters.retains_unchanged_source
                && facts.parameters.edge == RuntimeSamplingEdge::TransparentBlack
        })
}

#[cfg(test)]
fn c11_spatial_mappings_are_exact(closed: &ClosedExecutableGraph) -> bool {
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let blurs = closed.facts.blurs.iter().all(|facts| {
        let vertical = closed
            .lowered
            .passes
            .iter()
            .find(|pass| pass.id == facts.vertical)
            .and_then(|pass| match &pass.kind {
                RuntimePassKind::BlurVertical(Some(blur)) => Some(blur),
                _ => None,
            });
        resources
            .get(&facts.source)
            .zip(resources.get(&facts.intermediate))
            .zip(resources.get(&facts.result))
            .is_some_and(|((source, intermediate), result)| {
                facts.blur.spatial.source == source.spatial
                    && facts.blur.spatial.result == intermediate.spatial
                    && facts.blur.spatial.result == result.spatial
                    && vertical.is_some_and(|blur| {
                        blur.spatial.source == intermediate.spatial
                            && blur.spatial.result == result.spatial
                    })
            })
    });
    let shadows = closed.facts.drop_shadows.iter().all(|facts| {
        resources
            .get(&facts.source)
            .zip(resources.get(&facts.horizontal_result))
            .zip(resources.get(&facts.vertical_result))
            .zip(resources.get(&facts.shadow))
            .zip(resources.get(&facts.result))
            .is_some_and(|((((source, horizontal), vertical), shadow), result)| {
                facts.blur.spatial.source == source.spatial
                    && facts.blur.spatial.result == horizontal.spatial
                    && facts.blur.spatial.result == vertical.spatial
                    && facts.parameters.spatial.source == vertical.spatial
                    && facts.parameters.spatial.result == shadow.spatial
                    && facts.parameters.spatial.result == result.spatial
            })
    });
    blurs && shadows
}

#[cfg(test)]
fn c11_sources_and_results_are_distinct(closed: &ClosedExecutableGraph) -> bool {
    closed.facts.blurs.iter().all(|facts| {
        [facts.source, facts.intermediate, facts.result]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len()
            == 3
    }) && closed.facts.drop_shadows.iter().all(|facts| {
        [
            facts.source,
            facts.horizontal_result,
            facts.vertical_result,
            facts.shadow,
            facts.result,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len()
            == 5
    })
}

#[cfg(test)]
fn c11_shadow_fanout_is_exact(closed: &ClosedExecutableGraph) -> bool {
    closed.facts.drop_shadows.iter().all(|facts| {
        let readers = closed
            .lowered
            .passes
            .iter()
            .filter(|pass| pass.reads.iter().any(|read| read.resource == facts.source))
            .map(|pass| pass.id)
            .collect::<Vec<_>>();
        readers == [facts.horizontal, facts.merge]
            && closed
                .lowered
                .resources
                .iter()
                .find(|resource| resource.id == facts.source)
                .is_some_and(|resource| resource.expected_reads == 2)
    })
}

#[cfg(test)]
fn c11_shadow_releases_are_exact(closed: &ClosedExecutableGraph) -> bool {
    closed.facts.drop_shadows.iter().all(|facts| {
        let horizontal = closed
            .lowered
            .passes
            .iter()
            .find(|pass| pass.id == facts.horizontal);
        let merge = closed
            .lowered
            .passes
            .iter()
            .find(|pass| pass.id == facts.merge);
        let source = closed
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == facts.source);
        horizontal.is_some_and(|pass| !pass.releases.contains(&facts.source))
            && merge.is_some_and(|pass| pass.releases.contains(&facts.source))
            && source.is_some_and(|resource| resource.last_use == facts.merge)
    })
}

#[cfg(test)]
fn c11_dependencies_and_lifetimes_are_exact(closed: &ClosedExecutableGraph) -> bool {
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let pass_positions = closed
        .lowered
        .passes
        .iter()
        .enumerate()
        .map(|(position, pass)| (pass.id, position))
        .collect::<BTreeMap<_, _>>();
    closed.facts.proves_exact_facts_for(&closed.lowered)
        && closed
            .lowered
            .passes
            .iter()
            .enumerate()
            .all(|(position, pass)| {
                pass.dependencies.iter().all(|dependency| {
                    pass_positions
                        .get(dependency)
                        .is_some_and(|dependency_position| *dependency_position < position)
                }) && pass.releases.iter().all(|released| {
                    resources
                        .get(released)
                        .is_some_and(|resource| resource.last_use == pass.id)
                })
            })
}

fn vello_capture_raster_parameters(
    target_extent: PhysicalSize,
    antialiasing: Antialiasing,
) -> Result<RasterParameters> {
    RasterParameters::try_new(target_extent, peniko::Color::TRANSPARENT, antialiasing)
}

#[cfg(test)]
fn graph_clip_coverage_observation(
    commands: RenderCommands,
    context: FrameContext,
    _capabilities: DeviceCapabilities,
) -> Option<GraphClipCoverageObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let closed = ClosedExecutableGraph::try_from_lowered(lowered).ok()?;
    let all_vello_capture_count = closed
        .lowered
        .passes
        .iter()
        .filter(|pass| matches!(pass.kind, RuntimePassKind::VelloCapture(Some(_))))
        .count();
    let composite_coverage_read_count = closed
        .lowered
        .passes
        .iter()
        .flat_map(|pass| &pass.reads)
        .filter(|read| read.role == RuntimeReadRole::ClipCoverage)
        .count();
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let captures = closed
        .facts
        .captures
        .iter()
        .filter_map(|capture| {
            let coverage = capture.work().clip_coverage()?;
            let resource = resources.get(&capture.target()).copied()?;
            let elements = coverage
                .elements
                .iter()
                .map(|element| (element.clip.clone(), element.transform))
                .collect::<Vec<_>>();
            let scene = encode_vello_clip_coverage_scene(
                &elements,
                capture.initial_transform(),
                capture.target_extent(),
            )
            .ok()?;
            let emitted_draws = scene.observation_for_test().solid_path_draws_for_test()?;
            let prepared = scene
                .prepare_raster(
                    vello_capture_raster_parameters(
                        capture.target_extent(),
                        capture.antialiasing(),
                    )
                    .ok()?,
                )
                .ok()?;
            let raster = prepared_vello_pass_observation_for_test(&prepared);
            let elements = coverage
                .elements
                .iter()
                .map(|element| ClipCoverageElementObservationForTest {
                    clip: element.clip.clone(),
                    transform: element.transform,
                })
                .collect();
            Some(ClipCoverageCaptureObservationForTest {
                elements,
                antialiasing: capture.antialiasing(),
                device_origin: resource.spatial.device_origin,
                target_extent: capture.target_extent(),
                texel_origin: capture.texel_origin(),
                raster_scale: capture.raster_scale(),
                first_texel_center: Point::new(
                    (f64::from(resource.spatial.device_origin.0) + 0.5) / capture.raster_scale(),
                    (f64::from(resource.spatial.device_origin.1) + 0.5) / capture.raster_scale(),
                ),
                initial_transform: capture.initial_transform(),
                emitted_draws,
                uses_coverage_resource_role: resource.role == RuntimeResourceRole::ClipCoverage,
                uses_rgba8_target: resource.format == RuntimeResourceFormat::ClipCoverageRgba8Unorm,
                uses_transparent_base: raster.transparent_base_for_test(),
                raster_antialiasing: raster.antialiasing_for_test(),
                raster_target_extent: raster.target_extent_for_test(),
            })
        })
        .collect();

    Some(GraphClipCoverageObservationForTest {
        captures,
        all_vello_capture_count,
        composite_coverage_read_count,
    })
}

#[cfg(test)]
fn composition_graph_observation(
    commands: RenderCommands,
    context: FrameContext,
    _capabilities: DeviceCapabilities,
) -> Option<CompositionGraphObservationForTest> {
    let authored_mask_ids = resolved_mask_image_ids_inner_to_outer(&commands.commands);
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let closed = ClosedExecutableGraph::try_from_lowered(lowered).ok()?;
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let mut observed_mask_ids = Vec::new();
    let layers_inner_to_outer = closed
        .facts
        .layer_compositions
        .iter()
        .map(|layer| {
            observe_layer_composition(layer, &closed.lowered, &resources, &mut observed_mask_ids)
        })
        .collect::<Option<Vec<_>>>()?;
    let (
        root_surface_base_clears,
        root_surface_base_color,
        transparent_isolation_clears,
        nontransparent_isolation_clears,
    ) = composition_clear_observation(&closed.lowered);

    Some(CompositionGraphObservationForTest {
        layers_inner_to_outer,
        mask_identity_is_preserved: authored_mask_ids == observed_mask_ids,
        root_surface_base_clears,
        root_surface_base_color,
        transparent_isolation_clears,
        nontransparent_isolation_clears,
    })
}

#[cfg(test)]
fn observe_layer_composition(
    layer: &ExecutableLayerCompositionFacts,
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    observed_mask_ids: &mut Vec<super::ImageId>,
) -> Option<LayerCompositionObservationForTest> {
    let RuntimeCompositeKind::Layer {
        transform,
        parameters,
        clip,
        outer_clips,
        clip_coverage,
    } = &layer.composite.kind
    else {
        return None;
    };
    let alpha_mask = parameters
        .alpha_mask()
        .map(RuntimeResolvedAlphaMaskComposition::resource);
    if clip_coverage != &layer.clip_coverage || alpha_mask != layer.alpha_mask {
        return None;
    }
    if let Some(mask) = alpha_mask {
        let resource = resources.get(&mask).copied()?;
        let Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) = &resource.import else {
            return None;
        };
        observed_mask_ids.push(upload.cache_key().image_id());
    }
    let pass = lowered.passes.iter().find(|pass| pass.id == layer.pass)?;
    let reads = pass
        .reads
        .iter()
        .map(|read| match read.role {
            RuntimeReadRole::CompositeParent => Some(CompositionReadObservationForTest::Parent),
            RuntimeReadRole::CompositeSource => Some(CompositionReadObservationForTest::Source),
            RuntimeReadRole::ClipCoverage => Some(CompositionReadObservationForTest::ClipCoverage),
            RuntimeReadRole::AlphaMask => Some(CompositionReadObservationForTest::AlphaMask),
            RuntimeReadRole::CaptureSource
            | RuntimeReadRole::CompletedParent
            | RuntimeReadRole::FilterSource
            | RuntimeReadRole::BlurredSourceAlpha
            | RuntimeReadRole::Shadow
            | RuntimeReadRole::FinalWorkingImage => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let mut outer_operations = vec![CompositionOuterOperationObservationForTest::SourceMapping];
    if clip.is_some() || !outer_clips.is_empty() {
        outer_operations.push(CompositionOuterOperationObservationForTest::ClipCoverage);
    }
    if alpha_mask.is_some() {
        outer_operations.push(CompositionOuterOperationObservationForTest::AlphaMask);
    }
    outer_operations.extend([
        CompositionOuterOperationObservationForTest::Opacity,
        CompositionOuterOperationObservationForTest::Blend,
    ]);
    Some(LayerCompositionObservationForTest {
        transform: *transform,
        opacity: parameters.opacity(),
        blend: parameters.blend(),
        has_own_clip: clip.is_some(),
        inherited_outer_clip_count: outer_clips.len(),
        inherited_outer_clip_transforms: outer_clips.iter().map(|clip| clip.transform).collect(),
        reads,
        outer_operations,
        source_captured_before_outer_semantics: layer
            .composite
            .source_captured_before_outer_semantics,
    })
}

#[cfg(test)]
fn composition_clear_observation(
    lowered: &LoweredGraphPlan,
) -> (usize, Option<Color>, usize, usize) {
    let mut observation = (0usize, None, 0usize, 0usize);
    for pass in &lowered.passes {
        let RuntimePassKind::ClearRoot {
            initialization,
            color,
        } = pass.kind
        else {
            continue;
        };
        match initialization {
            RuntimeInitialization::SurfaceBaseColor => {
                observation.0 = observation.0.saturating_add(1);
                observation.1 = Some(color);
            }
            RuntimeInitialization::Transparent if color == Color::TRANSPARENT => {
                observation.2 = observation.2.saturating_add(1);
            }
            RuntimeInitialization::Transparent => {
                observation.3 = observation.3.saturating_add(1);
            }
        }
    }
    observation
}

#[cfg(test)]
fn resolved_mask_image_ids_inner_to_outer(
    commands: &[super::command::RenderCommand],
) -> Vec<super::ImageId> {
    fn collect(commands: &[super::command::RenderCommand], image_ids: &mut Vec<super::ImageId>) {
        for command in commands {
            let super::command::RenderCommand::Layer { layer, children } = command else {
                continue;
            };
            collect(children, image_ids);
            if let Some(mask) = &layer.mask {
                image_ids.push(mask.upload().cache_key().image_id());
            }
        }
    }

    let mut image_ids = Vec::new();
    collect(commands, &mut image_ids);
    image_ids
}

#[cfg(test)]
fn c08_rejects_every_other_pass_kind_and_composite_payload(plan: &LoweredGraphPlan) -> bool {
    let forbidden_kinds = [
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            color: Color::TRANSPARENT,
        },
        RuntimePassKind::VelloCapture(None),
        RuntimePassKind::CopyBackdrop,
        RuntimePassKind::ColorFilter(None),
        RuntimePassKind::BlurHorizontal(None),
        RuntimePassKind::BlurVertical(None),
        RuntimePassKind::DropShadowColorize(None),
        RuntimePassKind::Composite(None),
    ];
    if forbidden_kinds
        .iter()
        .any(|kind| c08_pass_class(kind).is_some())
    {
        return false;
    }
    let Some(composite_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::SpanSourceOver))
    else {
        return false;
    };
    let Ok(layer_parameters) = RuntimeLayerCompositeParameters::try_new(
        Transform::identity(),
        1.0,
        BlendMode::Normal,
        false,
        None,
    ) else {
        return false;
    };
    let composite_payloads = [
        None,
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver,
            source_captured_before_outer_semantics: false,
        }),
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer {
                transform: Transform::identity(),
                parameters: Box::new(layer_parameters),
                clip: None,
                outer_clips: Vec::new(),
                clip_coverage: None,
            },
            source_captured_before_outer_semantics: true,
        }),
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::DropShadow,
            source_captured_before_outer_semantics: true,
        }),
    ];
    composite_payloads.into_iter().all(|payload| {
        let mut invalid = plan.clone();
        invalid.passes[composite_index].kind = RuntimePassKind::Composite(payload);
        C08PreparableGraph::try_from_lowered(invalid).is_err()
    })
}

#[cfg(test)]
fn c08_rejects_missing_or_reordered_spine_passes(plan: &LoweredGraphPlan) -> bool {
    let Some(capture_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::VelloCapture))
    else {
        return false;
    };
    let Some(canonicalize_index) = capture_index.checked_add(1) else {
        return false;
    };
    let Some(present_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::Present))
    else {
        return false;
    };

    let mut missing_canonicalize = plan.clone();
    missing_canonicalize.passes.remove(canonicalize_index);
    let mut reordered_pair = plan.clone();
    reordered_pair
        .passes
        .swap(capture_index, canonicalize_index);
    let mut repeated_clear = plan.clone();
    repeated_clear
        .passes
        .insert(capture_index, repeated_clear.passes[0].clone());
    let mut missing_present = plan.clone();
    missing_present.passes.remove(present_index);
    let mut nonterminal_present = plan.clone();
    nonterminal_present
        .passes
        .swap(present_index - 1, present_index);

    [
        missing_canonicalize,
        reordered_pair,
        repeated_clear,
        missing_present,
        nonterminal_present,
    ]
    .into_iter()
    .all(|invalid| C08PreparableGraph::try_from_lowered(invalid).is_err())
}

#[cfg(test)]
fn c08_rejects_malformed_bindings(plan: &LoweredGraphPlan) -> bool {
    let Some(capture_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::VelloCapture))
    else {
        return false;
    };
    let canonicalize_index = capture_index + 1;
    let composite_index = capture_index + 2;
    let present_index = plan.passes.len() - 1;
    let RuntimeResultBinding::Resource(capture_target) = plan.passes[capture_index].result else {
        return false;
    };
    let RuntimeResultBinding::Resource(canonical_target) = plan.passes[canonicalize_index].result
    else {
        return false;
    };
    let Some(capture_resource_index) = plan
        .resources
        .iter()
        .position(|resource| resource.id == capture_target)
    else {
        return false;
    };

    let mut invalid_plans = Vec::new();

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].dependencies.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].reads[0].role = RuntimeReadRole::FinalWorkingImage;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].reads[0].resource = plan.root_working_image;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].result = RuntimeResultBinding::Empty;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].releases.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].reads.swap(0, 1);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].result = RuntimeResultBinding::Resource(canonical_target);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].cache_keys = None;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[present_index].result = RuntimeResultBinding::Output(Format::Bgra8);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[present_index].releases.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index].expected_reads = 2;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index].format =
        RuntimeResourceFormat::Working(plan.working_format);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index]
        .spatial
        .texel_origin = Point::new(-0.25, 0.5);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.final_present = invalid.passes[0].id;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    if let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) =
        &mut invalid.passes[capture_index].kind
    {
        span.scope = RuntimeVelloSpanScope::LayerSource;
    }
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    if let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) =
        &mut invalid.passes[capture_index].kind
    {
        span.captured_before_outer_semantics = false;
    }
    invalid_plans.push(invalid);

    invalid_plans
        .into_iter()
        .all(|invalid| C08PreparableGraph::try_from_lowered(invalid).is_err())
}

#[cfg(test)]
fn bounded_capture_transform_observation(
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
) -> Option<BoundedCaptureTransformObservationForTest> {
    let capabilities = DeviceCapabilities::from_test_facts(true, true, 4_096);
    let scales = [1.0, 1.25, 2.0];
    let mut preserves_application_order_formula = true;
    let mut preserves_signed_texel_center_mapping = true;
    let mut preserves_capture_execution_facts = true;
    let mut lowers_scene_with_explicit_initial_transform = true;

    for raster_scale in scales {
        let context = FrameContext::try_new(
            super::Size::new(64.0, 64.0),
            raster_scale,
            antialiasing,
            Color::TRANSPARENT,
        )
        .ok()?;
        let graph = super::frame::forced_c08_graph_for_test(commands.clone(), context).ok()?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            WorkingFormat::HighPrecision,
            Format::Rgba8,
            &capabilities,
        )
        .ok()?;
        let preparable = C08PreparableGraph::try_from_lowered(lowered).ok()?;
        let actual_capture = preparable.execution.captures().first()?;
        let capture_pass = preparable
            .lowered
            .passes
            .iter()
            .find(|pass| pass.id == actual_capture.pass())?;
        let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(actual_span))) =
            &capture_pass.kind
        else {
            return None;
        };
        let mut span = actual_span.clone();
        span.capture_transform = capture_transform;
        span.parent_to_surface = parent_to_surface;
        let target = actual_capture.target();
        let mut spatial = preparable
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == target)?
            .spatial;
        spatial.device_origin = (-3, -2);
        spatial.texel_origin = Point::new(-3.0 / raster_scale, -2.0 / raster_scale);
        spatial.raster_scale = raster_scale;
        let facts = executable_vello_capture_facts(
            capture_pass.id,
            target,
            &RuntimeVelloCapture::Span(span),
            spatial,
        )?;

        let expected_transform =
            expected_bounded_capture_transform(capture_transform, parent_to_surface, spatial)?;
        preserves_application_order_formula &= transforms_are_close(
            facts.initial_transform(),
            expected_transform,
            f64::EPSILON * 32.0,
        );

        let local_to_surface = capture_transform.then(parent_to_surface).ok()?;
        let texel = (2_u32, 3_u32);
        let mapped_center = Point::new(
            spatial.texel_origin.x() + (f64::from(texel.0) + 0.5) / raster_scale,
            spatial.texel_origin.y() + (f64::from(texel.1) + 0.5) / raster_scale,
        );
        let local_center = inverse_transform_point(local_to_surface, mapped_center)?;
        let encoded_center = apply_transform(facts.initial_transform(), local_center);
        preserves_signed_texel_center_mapping &= (encoded_center.x() - 2.5).abs() <= 1.0e-12
            && (encoded_center.y() - 3.5).abs() <= 1.0e-12
            && facts.texel_origin() == spatial.texel_origin;

        preserves_capture_execution_facts &= facts.pass() == capture_pass.id
            && facts.target() == target
            && facts.commands() == Some(&commands)
            && facts.antialiasing() == antialiasing
            && facts.target_extent() == spatial.device_extent
            && facts.raster_scale() == raster_scale;

        let encoded = super::encode::encode_vello_scene_with_initial_transform(
            facts.commands()?,
            facts.initial_transform(),
        )
        .ok()?;
        let encoded_transform = encoded
            .observation_for_test()
            .first_glyph_run_for_test()?
            .transform_components_for_test();
        lowers_scene_with_explicit_initial_transform &= encoded_transform
            .iter()
            .zip(facts.initial_transform().as_array())
            .all(|(actual, expected)| (*actual - expected as f32).abs() <= 1.0e-5);
    }

    Some(BoundedCaptureTransformObservationForTest {
        preserves_application_order_formula,
        preserves_signed_texel_center_mapping,
        covers_required_raster_scales: scales == [1.0, 1.25, 2.0],
        preserves_capture_execution_facts,
        lowers_scene_with_explicit_initial_transform,
    })
}

#[cfg(test)]
fn expected_bounded_capture_transform(
    capture_transform: Transform,
    parent_to_surface: Transform,
    spatial: RuntimeSpatialDescriptor,
) -> Option<Transform> {
    capture_transform
        .then(parent_to_surface)
        .ok()?
        .then(Transform::translation(-spatial.texel_origin.x(), -spatial.texel_origin.y()).ok()?)
        .ok()?
        .then(Transform::scale(spatial.raster_scale, spatial.raster_scale).ok()?)
        .ok()
}

#[cfg(test)]
fn transforms_are_close(left: Transform, right: Transform, tolerance: f64) -> bool {
    left.as_array()
        .into_iter()
        .zip(right.as_array())
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

#[cfg(test)]
fn apply_transform(transform: Transform, point: Point) -> Point {
    let [a, b, c, d, e, f] = transform.as_array();
    Point::new(
        a * point.x() + c * point.y() + e,
        b * point.x() + d * point.y() + f,
    )
}

#[cfg(test)]
fn inverse_transform_point(transform: Transform, point: Point) -> Option<Point> {
    let [a, b, c, d, e, f] = transform.as_array();
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return None;
    }
    let x = point.x() - e;
    let y = point.y() - f;
    Some(Point::new(
        (d * x - c * y) / determinant,
        (-b * x + a * y) / determinant,
    ))
}

pub(crate) const VELLO_CAPTURE_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::STORAGE_BINDING
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST);
const RESOLVED_MASK_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST);

#[derive(Clone, Debug, Eq, PartialEq)]
enum RuntimeAllocationRequest {
    EffectTexture(EffectTextureDescriptor),
    ResolvedMask(ResolvedMaskUploadDescriptor),
}

impl RuntimeAllocationRequest {
    fn preflight(&self) -> Result<ResourceAllocationPreflight> {
        match self {
            Self::EffectTexture(descriptor) => {
                ResourceAllocationPreflight::effect_texture(*descriptor)
            }
            Self::ResolvedMask(descriptor) => {
                ResourceAllocationPreflight::resolved_mask(descriptor)?.ok_or_else(|| {
                    preparation_error(
                        "an explicitly empty resolved mask survived graph contribution pruning",
                    )
                })
            }
        }
    }

    fn acquire(
        &self,
        frame_scope: &mut FrameResourceScope,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capabilities: &DeviceCapabilities,
    ) -> Result<ResourceLease> {
        match self {
            Self::EffectTexture(descriptor) => {
                frame_scope.acquire_effect_texture(device, capabilities, *descriptor)
            }
            Self::ResolvedMask(descriptor) => {
                frame_scope.acquire_resolved_mask_upload(device, queue, capabilities, descriptor)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeResourcePreparationRequest {
    runtime: RuntimeResourceRequest,
    allocation: RuntimeAllocationRequest,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeKernelPreparationRequest {
    key: GaussianKernelKey,
    plan: GaussianKernelPlan,
    last_use: RuntimePassId,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimePassPreparationRequest {
    runtime: RuntimePass,
    spatial_uniform: Option<PassSpatialUniformBytes>,
    blur_edge_parameters: Option<BlurEdgeParameterBytes>,
    color_filter_operations: Option<ColorFilterOperationBytes>,
    drop_shadow_parameters: Option<DropShadowParameterBytes>,
    composite_parameters: Option<CompositeParameterBytes>,
    cache_keys: Option<RuntimePassCacheKeys>,
    kernel: Option<GaussianKernelKey>,
    kernel_releases: Vec<GaussianKernelKey>,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeGraphPreparationPlan {
    generation: RuntimeGraphGeneration,
    working_format: WorkingFormat,
    output_format: Format,
    resources: Vec<RuntimeResourcePreparationRequest>,
    kernels: Vec<RuntimeKernelPreparationRequest>,
    passes: Vec<RuntimePassPreparationRequest>,
    root_working_image: RuntimeResourceId,
    final_present: RuntimePassId,
    allocation_preflights: Vec<ResourceAllocationPreflight>,
}

impl RuntimeGraphPreparationPlan {
    fn try_derive(
        lowered: LoweredGraphPlan,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
    ) -> Result<Self> {
        Self::try_derive_with_color_filter_limits(
            lowered,
            selected_working_format,
            capabilities,
            device,
            ColorFilterOperationBufferLimits::from_device_limits(&device.limits()),
        )
    }

    fn try_derive_with_color_filter_limits(
        lowered: LoweredGraphPlan,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
        color_filter_limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        capabilities.validate_supported_working_format(selected_working_format)?;
        if selected_working_format != lowered.working_format {
            return Err(preparation_error(
                "the lowered graph working format does not match immutable device policy",
            ));
        }
        let mut prepared_resources = prepare_runtime_resources(&lowered, capabilities)?;
        let pass_facts = analyze_runtime_passes(
            &lowered,
            &prepared_resources.resource_by_id,
            &prepared_resources.resource_formats,
            device,
        )?;
        validate_prepared_graph_lifetimes(
            &lowered,
            &prepared_resources.resource_by_id,
            &pass_facts,
        )?;
        let passes = prepare_runtime_pass_requests(
            &lowered,
            &prepared_resources.resource_by_id,
            &pass_facts,
            color_filter_limits,
        )?;
        let kernels = pass_facts.kernels.into_values().collect::<Vec<_>>();
        for kernel in &kernels {
            prepared_resources
                .allocation_preflights
                .push(ResourceAllocationPreflight::gaussian_kernel(&kernel.plan)?);
        }

        Ok(Self {
            generation: lowered.generation,
            working_format: lowered.working_format,
            output_format: lowered.output_format,
            resources: prepared_resources.resources,
            kernels,
            passes,
            root_working_image: lowered.root_working_image,
            final_present: lowered.final_present,
            allocation_preflights: prepared_resources.allocation_preflights,
        })
    }
}

struct PreparedRuntimeResources<'a> {
    resource_by_id: BTreeMap<RuntimeResourceId, &'a RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    resources: Vec<RuntimeResourcePreparationRequest>,
    allocation_preflights: Vec<ResourceAllocationPreflight>,
}

fn prepare_runtime_resources<'a>(
    lowered: &'a LoweredGraphPlan,
    capabilities: &DeviceCapabilities,
) -> Result<PreparedRuntimeResources<'a>> {
    let mut prepared = PreparedRuntimeResources {
        resource_by_id: BTreeMap::new(),
        resource_formats: BTreeMap::new(),
        resources: Vec::with_capacity(lowered.resources.len()),
        allocation_preflights: Vec::with_capacity(lowered.resources.len()),
    };
    for resource in &lowered.resources {
        if prepared
            .resource_by_id
            .insert(resource.id, resource)
            .is_some()
        {
            return Err(preparation_error(
                "duplicate runtime resource reached graph preparation",
            ));
        }
        validate_runtime_resource_request(resource, lowered.working_format, capabilities)?;
        let allocation =
            prepare_runtime_resource_allocation(resource, lowered.working_format, capabilities)?;
        prepared.allocation_preflights.push(allocation.preflight()?);
        prepared
            .resource_formats
            .insert(resource.id, resource.format);
        prepared.resources.push(RuntimeResourcePreparationRequest {
            runtime: resource.clone(),
            allocation,
        });
    }
    Ok(prepared)
}

fn validate_runtime_resource_request(
    resource: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<()> {
    if runtime_resource_format(resource.role, working_format) != resource.format {
        return Err(preparation_error(
            "runtime resource role and working format are inconsistent",
        ));
    }
    if resource.expected_reads == 0 {
        return Err(preparation_error(
            "a prepared runtime resource has no scheduled reader",
        ));
    }
    let extent = resource.spatial.device_extent;
    if extent.width() == 0 || extent.height() == 0 {
        return Err(preparation_error(
            "a concrete runtime resource has an empty allocation extent",
        ));
    }
    capabilities.validate_effect_texture_extent(extent)
}

fn prepare_runtime_resource_allocation(
    resource: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<RuntimeAllocationRequest> {
    let extent = resource.spatial.device_extent;
    match (&resource.format, &resource.import) {
        (RuntimeResourceFormat::VelloCaptureRgba8Unorm, None)
            if resource.role == RuntimeResourceRole::CaptureWorkingImage
                && matches!(resource.producer, RuntimeResourceProducer::Pass(_)) =>
        {
            let descriptor =
                EffectTextureDescriptor::try_capture(extent, VELLO_CAPTURE_TEXTURE_USAGES)?;
            capabilities.validate_effect_texture_allocation(
                extent,
                None,
                descriptor.texture_format(),
                descriptor.usage(),
            )?;
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
        }
        (RuntimeResourceFormat::ClipCoverageRgba8Unorm, None)
            if resource.role == RuntimeResourceRole::ClipCoverage
                && matches!(resource.producer, RuntimeResourceProducer::Pass(_)) =>
        {
            let descriptor =
                EffectTextureDescriptor::try_coverage(extent, VELLO_CAPTURE_TEXTURE_USAGES)?;
            capabilities.validate_effect_texture_allocation(
                extent,
                None,
                descriptor.texture_format(),
                descriptor.usage(),
            )?;
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
        }
        (RuntimeResourceFormat::Working(format), None)
            if *format == working_format
                && resource.role != RuntimeResourceRole::CaptureWorkingImage
                && resource.role != RuntimeResourceRole::ClipCoverage
                && resource.role != RuntimeResourceRole::ImportedImage =>
        {
            let descriptor =
                EffectTextureDescriptor::try_working(*format, extent, format.required_usages())?;
            capabilities.validate_effect_texture_allocation(
                extent,
                Some(*format),
                descriptor.texture_format(),
                descriptor.usage(),
            )?;
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
        }
        (
            RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
            Some(RuntimeResourceImport::ResolvedAlphaMask(descriptor)),
        ) if resource.role == RuntimeResourceRole::ImportedImage
            && matches!(resource.producer, RuntimeResourceProducer::Imported) =>
        {
            if descriptor.physical_size() != extent {
                return Err(preparation_error(
                    "resolved-mask upload extent differs from its runtime resource",
                ));
            }
            descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
            capabilities.validate_effect_texture_allocation(
                extent,
                None,
                wgpu::TextureFormat::Rgba8Unorm,
                RESOLVED_MASK_TEXTURE_USAGES,
            )?;
            Ok(RuntimeAllocationRequest::ResolvedMask(descriptor.clone()))
        }
        _ => Err(preparation_error(
            "runtime resource has no exact concrete preparation request",
        )),
    }
}

struct RuntimePassAnalysis {
    actual_reads: BTreeMap<RuntimeResourceId, u32>,
    actual_last_reads: BTreeMap<RuntimeResourceId, RuntimePassId>,
    release_passes: BTreeMap<RuntimeResourceId, RuntimePassId>,
    produced_results: BTreeMap<RuntimeResourceId, RuntimePassId>,
    kernel_by_pass: BTreeMap<RuntimePassId, GaussianKernelKey>,
    kernels: BTreeMap<GaussianKernelKey, RuntimeKernelPreparationRequest>,
}

fn analyze_runtime_passes(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    device: &wgpu::Device,
) -> Result<RuntimePassAnalysis> {
    let mut pass_positions = BTreeMap::new();
    for (position, pass) in lowered.passes.iter().enumerate() {
        if pass_positions.insert(pass.id, position).is_some() {
            return Err(preparation_error(
                "duplicate runtime pass reached graph preparation",
            ));
        }
    }
    let mut facts = RuntimePassAnalysis {
        actual_reads: BTreeMap::new(),
        actual_last_reads: BTreeMap::new(),
        release_passes: BTreeMap::new(),
        produced_results: BTreeMap::new(),
        kernel_by_pass: BTreeMap::new(),
        kernels: BTreeMap::new(),
    };
    for (position, pass) in lowered.passes.iter().enumerate() {
        let pass_reads = analyze_runtime_pass(
            pass,
            position,
            lowered,
            resources,
            resource_formats,
            &pass_positions,
            &mut facts,
        )?;
        analyze_runtime_kernel(pass, device, &mut facts)?;
        debug_assert!(pass_reads.len() == pass.reads.len());
    }
    Ok(facts)
}

fn analyze_runtime_pass(
    pass: &RuntimePass,
    position: usize,
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    pass_positions: &BTreeMap<RuntimePassId, usize>,
    facts: &mut RuntimePassAnalysis,
) -> Result<BTreeSet<RuntimeResourceId>> {
    if pass.dependencies.iter().any(|dependency| {
        pass_positions
            .get(dependency)
            .is_none_or(|dependency_position| *dependency_position >= position)
    }) {
        return Err(preparation_error(
            "prepared pass has a missing or forward dependency",
        ));
    }
    let pass_reads = analyze_runtime_reads(pass, position, resources, pass_positions, facts)?;
    analyze_runtime_result(pass, lowered, resources, &pass_reads, facts)?;
    let expected_cache_keys = runtime_pass_cache_keys(
        &pass.kind,
        &pass.reads,
        pass.result,
        lowered.working_format,
        lowered.output_format,
        resource_formats,
    )?;
    if expected_cache_keys != pass.cache_keys {
        return Err(preparation_error(
            "prepared pass cache keys differ from exact runtime lowering",
        ));
    }
    for resource in &pass.releases {
        if !pass_reads.contains(resource)
            || facts.release_passes.insert(*resource, pass.id).is_some()
        {
            return Err(preparation_error(
                "prepared pass release is missing, duplicate, or not a last read",
            ));
        }
    }
    Ok(pass_reads)
}

fn analyze_runtime_reads(
    pass: &RuntimePass,
    position: usize,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    pass_positions: &BTreeMap<RuntimePassId, usize>,
    facts: &mut RuntimePassAnalysis,
) -> Result<BTreeSet<RuntimeResourceId>> {
    let mut pass_reads = BTreeSet::new();
    for read in &pass.reads {
        if !pass_reads.insert(read.resource) {
            return Err(preparation_error(
                "prepared pass contains a duplicate runtime read binding",
            ));
        }
        let resource = resources
            .get(&read.resource)
            .ok_or_else(|| preparation_error("prepared pass names a missing runtime resource"))?;
        if let RuntimeResourceProducer::Pass(producer) = resource.producer
            && pass_positions
                .get(&producer)
                .is_none_or(|producer_position| *producer_position >= position)
        {
            return Err(preparation_error(
                "prepared pass reads before its runtime resource producer",
            ));
        }
        let reads = facts.actual_reads.entry(read.resource).or_default();
        *reads = reads
            .checked_add(1)
            .ok_or_else(|| preparation_error("prepared runtime read count overflowed"))?;
        facts.actual_last_reads.insert(read.resource, pass.id);
    }
    Ok(pass_reads)
}

fn analyze_runtime_result(
    pass: &RuntimePass,
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    pass_reads: &BTreeSet<RuntimeResourceId>,
    facts: &mut RuntimePassAnalysis,
) -> Result<()> {
    match pass.result {
        RuntimeResultBinding::Resource(resource_id) => {
            let resource = resources
                .get(&resource_id)
                .ok_or_else(|| preparation_error("prepared pass result resource is missing"))?;
            if resource.producer != RuntimeResourceProducer::Pass(pass.id)
                || pass_reads.contains(&resource_id)
                || facts
                    .produced_results
                    .insert(resource_id, pass.id)
                    .is_some()
            {
                return Err(preparation_error(
                    "prepared pass result binding has no unique matching producer",
                ));
            }
        }
        RuntimeResultBinding::Output(format)
            if !matches!(pass.kind, RuntimePassKind::Present)
                || format != lowered.output_format =>
        {
            return Err(preparation_error(
                "prepared output binding differs from the terminal present target",
            ));
        }
        RuntimeResultBinding::Output(_) | RuntimeResultBinding::Empty => {}
    }
    Ok(())
}

fn analyze_runtime_kernel(
    pass: &RuntimePass,
    device: &wgpu::Device,
    facts: &mut RuntimePassAnalysis,
) -> Result<()> {
    let Some(blur) = runtime_blur_for_kernel(&pass.kind) else {
        return Ok(());
    };
    let kernel_plan = GaussianKernelPlan::try_new(
        blur.standard_deviation,
        blur.spatial.result.raster_scale,
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
        GaussianKernelSamplingForm::PairedLinear,
    )?;
    if kernel_plan.key() != blur.kernel
        || kernel_plan
            .validate_buffer_limits(GaussianKernelBufferLimits::from_device_limits(
                &device.limits(),
            ))
            .is_err()
    {
        return Err(preparation_error(
            "Gaussian kernel preparation differs from the exact runtime blur plan",
        ));
    }
    facts.kernel_by_pass.insert(pass.id, blur.kernel);
    match facts.kernels.entry(blur.kernel) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(RuntimeKernelPreparationRequest {
                key: blur.kernel,
                plan: kernel_plan,
                last_use: pass.id,
            });
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            if entry.get().plan != kernel_plan {
                return Err(preparation_error(
                    "one Gaussian kernel identity names conflicting plans",
                ));
            }
            entry.get_mut().last_use = pass.id;
        }
    }
    Ok(())
}

fn validate_prepared_graph_lifetimes(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    facts: &RuntimePassAnalysis,
) -> Result<()> {
    for resource in &lowered.resources {
        if facts.actual_reads.get(&resource.id).copied().unwrap_or(0) != resource.expected_reads
            || facts.actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
            || facts.release_passes.get(&resource.id).copied() != Some(resource.last_use)
        {
            return Err(preparation_error(
                "prepared runtime resource lifetime differs from exact lowering",
            ));
        }
        match resource.producer {
            RuntimeResourceProducer::Imported if resource.import.is_some() => {}
            RuntimeResourceProducer::Pass(pass)
                if resource.import.is_none()
                    && facts.produced_results.get(&resource.id).copied() == Some(pass) => {}
            RuntimeResourceProducer::Imported | RuntimeResourceProducer::Pass(_) => {
                return Err(preparation_error(
                    "prepared runtime resource producer/import binding is inconsistent",
                ));
            }
        }
    }
    validate_prepared_graph_root(lowered, resources)
}

fn validate_prepared_graph_root(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> Result<()> {
    let root = resources
        .get(&lowered.root_working_image)
        .ok_or_else(|| preparation_error("prepared root working resource is missing"))?;
    if root.format != RuntimeResourceFormat::Working(lowered.working_format) {
        return Err(preparation_error(
            "prepared root resource does not use the selected working format",
        ));
    }
    if lowered.passes.last().is_none_or(|pass| {
        pass.id != lowered.final_present
            || !matches!(pass.kind, RuntimePassKind::Present)
            || pass.result != RuntimeResultBinding::Output(lowered.output_format)
    }) {
        return Err(preparation_error(
            "prepared graph has no exact terminal present binding",
        ));
    }
    Ok(())
}

fn prepare_runtime_pass_requests(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    facts: &RuntimePassAnalysis,
    color_filter_limits: ColorFilterOperationBufferLimits,
) -> Result<Vec<RuntimePassPreparationRequest>> {
    let kernel_releases = facts.kernels.values().fold(
        BTreeMap::<RuntimePassId, Vec<GaussianKernelKey>>::new(),
        |mut releases, kernel| {
            releases
                .entry(kernel.last_use)
                .or_default()
                .push(kernel.key);
            releases
        },
    );
    lowered
        .passes
        .iter()
        .map(|pass| {
            let spatial_uniform =
                prepared_pass_spatial_uniform(pass, resources, lowered.root_working_image)?;
            let blur_edge_parameters = prepare_blur_edge_parameters(pass)?;
            let color_filter_operations =
                prepare_color_filter_operations(pass, color_filter_limits)?;
            let drop_shadow_parameters = prepare_drop_shadow_parameters(pass)?;
            let composite_parameters = prepared_pass_composite_parameters(pass)?;
            if spatial_uniform.is_some() != pass.cache_keys.is_some() {
                return Err(preparation_error(
                    "prepared pass spatial bytes and executable cache keys disagree",
                ));
            }
            Ok(RuntimePassPreparationRequest {
                runtime: pass.clone(),
                spatial_uniform,
                blur_edge_parameters,
                color_filter_operations,
                drop_shadow_parameters,
                composite_parameters,
                cache_keys: pass.cache_keys.clone(),
                kernel: facts.kernel_by_pass.get(&pass.id).copied(),
                kernel_releases: kernel_releases.get(&pass.id).cloned().unwrap_or_default(),
            })
        })
        .collect()
}

fn prepare_blur_edge_parameters(pass: &RuntimePass) -> Result<Option<BlurEdgeParameterBytes>> {
    let blur = match &pass.kind {
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            blur
        }
        _ => return Ok(None),
    };
    match blur.edge {
        RuntimeSamplingEdge::SemanticBorderMirror(bounds) => {
            BlurEdgeParameterBytes::try_from_semantic_bounds(bounds).map(Some)
        }
        RuntimeSamplingEdge::TransparentBlack => Ok(None),
        RuntimeSamplingEdge::ClampToExtent => Err(preparation_error(
            "a Gaussian blur cannot use clamp-to-extent edge semantics",
        )),
    }
}

fn prepare_drop_shadow_parameters(pass: &RuntimePass) -> Result<Option<DropShadowParameterBytes>> {
    let RuntimePassKind::DropShadowColorize(Some(shadow)) = &pass.kind else {
        return Ok(None);
    };
    let bytes = DropShadowParameterBytes::try_new(shadow.offset, shadow.color)?;
    if bytes.as_bytes().len() != 32 {
        return Err(preparation_error(
            "drop-shadow parameter serialization changed its exact WGSL byte length",
        ));
    }
    Ok(Some(bytes))
}

fn prepare_color_filter_operations(
    pass: &RuntimePass,
    limits: ColorFilterOperationBufferLimits,
) -> Result<Option<ColorFilterOperationBytes>> {
    let RuntimePassKind::ColorFilter(Some(filter)) = &pass.kind else {
        return Ok(None);
    };
    let bytes = ColorFilterOperationBytes::try_from_runtime_operations_with_limits(
        filter.operations(),
        limits,
    )?;
    if bytes.as_bytes().is_empty() {
        return Err(preparation_error(
            "prepared color-filter operation bytes are empty",
        ));
    }
    Ok(Some(bytes))
}

fn runtime_blur_for_kernel(kind: &RuntimePassKind) -> Option<&RuntimeBlur> {
    match kind {
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            Some(blur)
        }
        _ => None,
    }
}

fn prepared_pass_spatial_uniform(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    root_working_image: RuntimeResourceId,
) -> Result<Option<PassSpatialUniformBytes>> {
    if pass.cache_keys.is_none() {
        return Ok(None);
    }
    let result_spatial = || -> Result<RuntimeSpatialDescriptor> {
        let RuntimeResultBinding::Resource(resource) = pass.result else {
            return Err(preparation_error(
                "custom pass has no concrete runtime result resource",
            ));
        };
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass result spatial binding is missing"))
    };
    let read_spatial = |role| -> Result<RuntimeSpatialDescriptor> {
        let resource = pass
            .reads
            .iter()
            .find(|read| read.role == role)
            .map(|read| read.resource)
            .ok_or_else(|| preparation_error("custom pass source spatial binding is missing"))?;
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass source resource is missing"))
    };

    let (source, destination) = match &pass.kind {
        RuntimePassKind::CanonicalizeCapture => (
            read_spatial(RuntimeReadRole::CaptureSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::CopyBackdrop => (
            read_spatial(RuntimeReadRole::CompletedParent)?,
            result_spatial()?,
        ),
        RuntimePassKind::ColorFilter(Some(filter)) => {
            (filter.spatial.source, filter.spatial.result)
        }
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            (blur.spatial.source, blur.spatial.result)
        }
        RuntimePassKind::DropShadowColorize(Some(shadow)) => {
            (shadow.spatial.source, shadow.spatial.result)
        }
        RuntimePassKind::Composite(Some(_)) => (
            read_spatial(RuntimeReadRole::CompositeSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::Present => {
            let source = read_spatial(RuntimeReadRole::FinalWorkingImage)?;
            let destination = resources
                .get(&root_working_image)
                .map(|resource| resource.spatial)
                .ok_or_else(|| preparation_error("present destination spatial is missing"))?;
            (source, destination)
        }
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => {
            return Err(preparation_error(
                "non-executable pass unexpectedly requested spatial serialization",
            ));
        }
    };
    PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(source, destination).map(Some)
}

fn prepared_pass_composite_parameters(
    pass: &RuntimePass,
) -> Result<Option<CompositeParameterBytes>> {
    match &pass.kind {
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) => {
            let bytes = CompositeParameterBytes::try_from_runtime_layer(parameters)?;
            if bytes.as_bytes().len() != 112 {
                return Err(preparation_error(
                    "composite parameter serialization changed its exact WGSL byte length",
                ));
            }
            Ok(Some(bytes))
        }
        _ => Ok(None),
    }
}

fn preparation_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

struct PreparedResourceBinding {
    allocation: RuntimeAllocationRequest,
    lease: Option<ResourceLease>,
}

struct PreparedKernelBinding {
    lease: Option<ResourceLease>,
}

struct PreparedColorFilterOperationBinding {
    bytes: ColorFilterOperationBytes,
    buffer: Option<wgpu::Buffer>,
}

enum PreparedC11PassObjects {
    CopyBackdrop {
        parent_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
    Blur {
        source_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
    DropShadowColorize {
        source_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
}

struct PreparedPassRealization {
    update: Option<ProvisionalDevicePassCacheUpdate>,
    c11_objects: BTreeMap<RuntimePassId, PreparedC11PassObjects>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchPassSemantics {
    ClosedExecutable,
    FuturePass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphPreparationIneligibility {
    OutsideClosedExecutableGraph,
}

impl GraphPreparationIneligibility {
    fn into_error(self) -> Error {
        match self {
            Self::OutsideClosedExecutableGraph => preparation_error(
                "a graph outside the closed executable subset cannot enter runtime preparation",
            ),
        }
    }
}

enum PrePreparationGraphClassification {
    ExactC08(C08PreparableGraph),
    ExactC09(ClosedExecutableGraph),
    ExactC10(C10PreparableGraph),
    ExactC11(C11PreparableGraph),
    ExactC12(C12PreparableGraph),
    FuturePasses,
    Ineligible(GraphPreparationIneligibility),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutableGraphWorkingFormatRequest {
    ConfiguredPolicy(EffectQualityPolicy),
    #[cfg(test)]
    Exact(WorkingFormat),
}

impl ExecutableGraphWorkingFormatRequest {
    fn resolve(self, capabilities: &DeviceCapabilities) -> Result<WorkingFormat> {
        match self {
            Self::ConfiguredPolicy(policy) => capabilities.resolve_effect_working_format(policy),
            #[cfg(test)]
            Self::Exact(working_format) => {
                capabilities.validate_supported_working_format(working_format)?;
                Ok(working_format)
            }
        }
    }
}

#[must_use = "the closed graph dispatch result must select exactly one renderer route"]
pub(crate) enum ExecutableGraphDispatchEligibility {
    ExactC08(C08PreparableGraph),
    ExactC09(C09PreparableGraph),
    ExactC12(C12PreparableGraph),
    FuturePasses,
}

impl ExecutableGraphDispatchEligibility {
    fn try_classify_c12(
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
        preparable: C12PreparableGraph,
    ) -> Result<Self> {
        if !preparable.proves_closed_backdrop_facts() {
            return Err(preparation_error(
                "C12 classification lost its closed pre-allocation facts",
            ));
        }
        let working_format = working_format.resolve(capabilities)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            graph,
            working_format,
            output_format,
            capabilities,
        )?;
        match PrePreparationGraphClassification::classify(lowered) {
            PrePreparationGraphClassification::ExactC12(preparable)
                if preparable.proves_closed_backdrop_facts() =>
            {
                Ok(Self::ExactC12(preparable))
            }
            PrePreparationGraphClassification::ExactC08(_)
            | PrePreparationGraphClassification::ExactC09(_)
            | PrePreparationGraphClassification::ExactC10(_)
            | PrePreparationGraphClassification::ExactC11(_)
            | PrePreparationGraphClassification::ExactC12(_)
            | PrePreparationGraphClassification::FuturePasses
            | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                "checked C12 dispatch lowering changed its closed eligibility result",
            )),
        }
    }

    pub(crate) fn try_classify(
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
    ) -> Result<Self> {
        let classification = PrePreparationGraphClassification::classify(
            LoweredGraphPlan::try_lower_for_dispatch_classification(
                graph,
                WorkingFormat::HighPrecision,
                output_format,
            )?,
        );
        match classification {
            PrePreparationGraphClassification::ExactC12(preparable) => Self::try_classify_c12(
                graph,
                output_format,
                working_format,
                capabilities,
                preparable,
            ),
            PrePreparationGraphClassification::ExactC11(preparable) => {
                if !preparable.proves_closed_filter_facts() {
                    return Err(preparation_error(
                        "C11 classification lost its closed pre-allocation facts",
                    ));
                }
                Ok(Self::FuturePasses)
            }
            PrePreparationGraphClassification::ExactC10(preparable) => {
                if !preparable.proves_closed_color_facts() {
                    return Err(preparation_error(
                        "C10 classification lost its closed pre-allocation facts",
                    ));
                }
                Ok(Self::FuturePasses)
            }
            PrePreparationGraphClassification::ExactC09(_) => {
                let working_format = working_format.resolve(capabilities)?;
                let lowered = LoweredGraphPlan::try_lower_validated_graph(
                    graph,
                    working_format,
                    output_format,
                    capabilities,
                )?;
                match PrePreparationGraphClassification::classify(lowered) {
                    PrePreparationGraphClassification::ExactC09(closed) => {
                        C09PreparableGraph::try_from_closed(closed)
                            .map(Self::ExactC09)
                            .map_err(|_| {
                                preparation_error(
                                    "checked C09 dispatch lowering lost its C09-only facts",
                                )
                            })
                    }
                    PrePreparationGraphClassification::ExactC08(_)
                    | PrePreparationGraphClassification::ExactC10(_)
                    | PrePreparationGraphClassification::ExactC11(_)
                    | PrePreparationGraphClassification::ExactC12(_)
                    | PrePreparationGraphClassification::FuturePasses
                    | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                        "checked C09 dispatch lowering changed its closed eligibility result",
                    )),
                }
            }
            PrePreparationGraphClassification::FuturePasses => Ok(Self::FuturePasses),
            PrePreparationGraphClassification::Ineligible(ineligibility) => {
                Err(ineligibility.into_error())
            }
            PrePreparationGraphClassification::ExactC08(_) => {
                let working_format = working_format.resolve(capabilities)?;
                let lowered = LoweredGraphPlan::try_lower_validated_graph(
                    graph,
                    working_format,
                    output_format,
                    capabilities,
                )?;
                match PrePreparationGraphClassification::classify(lowered) {
                    PrePreparationGraphClassification::ExactC08(preparable) => {
                        Ok(Self::ExactC08(preparable))
                    }
                    PrePreparationGraphClassification::ExactC09(_)
                    | PrePreparationGraphClassification::ExactC10(_)
                    | PrePreparationGraphClassification::ExactC11(_)
                    | PrePreparationGraphClassification::ExactC12(_)
                    | PrePreparationGraphClassification::FuturePasses
                    | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                        "checked C08 dispatch lowering changed its closed eligibility result",
                    )),
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn c10_preparable_graph_for_test(
    graph: &GpuRenderGraph,
    output_format: Format,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<C10PreparableGraph> {
    capabilities.validate_supported_working_format(working_format)?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        graph,
        working_format,
        output_format,
        capabilities,
    )?;
    match PrePreparationGraphClassification::classify(lowered) {
        PrePreparationGraphClassification::ExactC10(preparable)
            if preparable.proves_closed_color_facts() =>
        {
            Ok(preparable)
        }
        PrePreparationGraphClassification::ExactC08(_)
        | PrePreparationGraphClassification::ExactC09(_)
        | PrePreparationGraphClassification::ExactC10(_)
        | PrePreparationGraphClassification::ExactC11(_)
        | PrePreparationGraphClassification::ExactC12(_)
        | PrePreparationGraphClassification::FuturePasses
        | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
            "the authored C10 fixture is outside the exact closed color graph",
        )),
    }
}

#[cfg(test)]
fn c11_preparable_graph_for_test(lowered: LoweredGraphPlan) -> Result<C11PreparableGraph> {
    match PrePreparationGraphClassification::classify(lowered) {
        PrePreparationGraphClassification::ExactC11(preparable)
            if preparable.proves_closed_filter_facts() =>
        {
            Ok(preparable)
        }
        PrePreparationGraphClassification::ExactC08(_)
        | PrePreparationGraphClassification::ExactC09(_)
        | PrePreparationGraphClassification::ExactC10(_)
        | PrePreparationGraphClassification::ExactC11(_)
        | PrePreparationGraphClassification::ExactC12(_)
        | PrePreparationGraphClassification::FuturePasses
        | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
            "the authored C11 fixture is outside the exact closed spatial-filter graph",
        )),
    }
}

#[cfg(test)]
pub(crate) fn c11_preparable_graph_from_graph_for_test(
    graph: &GpuRenderGraph,
    output_format: Format,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<C11PreparableGraph> {
    capabilities.validate_supported_working_format(working_format)?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        graph,
        working_format,
        output_format,
        capabilities,
    )?;
    c11_preparable_graph_for_test(lowered)
}

#[cfg(test)]
pub(crate) fn c12_preparable_graph_from_graph_for_test(
    graph: &GpuRenderGraph,
    output_format: Format,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<C12PreparableGraph> {
    capabilities.validate_supported_working_format(working_format)?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        graph,
        working_format,
        output_format,
        capabilities,
    )?;
    match PrePreparationGraphClassification::classify(lowered) {
        PrePreparationGraphClassification::ExactC12(preparable)
            if preparable.proves_closed_backdrop_facts() =>
        {
            Ok(preparable)
        }
        PrePreparationGraphClassification::ExactC08(_)
        | PrePreparationGraphClassification::ExactC09(_)
        | PrePreparationGraphClassification::ExactC10(_)
        | PrePreparationGraphClassification::ExactC11(_)
        | PrePreparationGraphClassification::ExactC12(_)
        | PrePreparationGraphClassification::FuturePasses
        | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
            "the authored C12 fixture is outside the exact bounded backdrop graph",
        )),
    }
}

impl PrePreparationGraphClassification {
    fn classify(lowered: LoweredGraphPlan) -> Self {
        let closed = match ClosedExecutableGraph::try_from_lowered(lowered) {
            Ok(closed) => closed,
            Err(lowered) => {
                let mut contains_future_passes = false;
                for pass in &lowered.passes {
                    match dispatch_pass_semantics(&pass.kind) {
                        Some(DispatchPassSemantics::ClosedExecutable) => {}
                        Some(DispatchPassSemantics::FuturePass) => {
                            contains_future_passes = true;
                        }
                        None => {
                            return Self::Ineligible(
                                GraphPreparationIneligibility::OutsideClosedExecutableGraph,
                            );
                        }
                    }
                }
                return if contains_future_passes {
                    Self::FuturePasses
                } else {
                    Self::Ineligible(GraphPreparationIneligibility::OutsideClosedExecutableGraph)
                };
            }
        };
        match C08PreparableGraph::try_from_closed(closed) {
            Ok(preparable) => Self::ExactC08(preparable),
            Err(closed) => match C12PreparableGraph::try_from_closed(*closed) {
                Ok(preparable) => Self::ExactC12(preparable),
                Err(closed) => match C11PreparableGraph::try_from_closed(*closed) {
                    Ok(preparable) => Self::ExactC11(preparable),
                    Err(closed) => match C10PreparableGraph::try_from_closed(*closed) {
                        Ok(preparable) => Self::ExactC10(preparable),
                        Err(closed) => match C09PreparableGraph::try_from_closed(*closed) {
                            Ok(preparable) => Self::ExactC09(preparable.into_closed()),
                            Err(_) => Self::Ineligible(
                                GraphPreparationIneligibility::OutsideClosedExecutableGraph,
                            ),
                        },
                    },
                },
            },
        }
    }
}

fn dispatch_pass_semantics(kind: &RuntimePassKind) -> Option<DispatchPassSemantics> {
    match kind {
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            ..
        }
        | RuntimePassKind::CanonicalizeCapture
        | RuntimePassKind::Present => Some(DispatchPassSemantics::ClosedExecutable),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            color,
        } if *color == Color::TRANSPARENT => Some(DispatchPassSemantics::ClosedExecutable),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            ..
        } => None,
        RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span)))
            if !span.commands.commands.is_empty()
                && span.captured_before_outer_semantics
                && span
                    .capture_transform
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite())
                && span
                    .parent_to_surface
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite()) =>
        {
            Some(DispatchPassSemantics::ClosedExecutable)
        }
        RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(coverage)))
            if !coverage.elements.is_empty()
                && coverage.elements.iter().all(|element| {
                    element
                        .transform
                        .as_array()
                        .iter()
                        .all(|value| value.is_finite())
                }) =>
        {
            Some(DispatchPassSemantics::ClosedExecutable)
        }
        RuntimePassKind::VelloCapture(_) => None,
        RuntimePassKind::CopyBackdrop
        | RuntimePassKind::ColorFilter(Some(_))
        | RuntimePassKind::DropShadowColorize(Some(_)) => Some(DispatchPassSemantics::FuturePass),
        RuntimePassKind::BlurHorizontal(Some(blur)) if blur.axis == RuntimeBlurAxis::Horizontal => {
            Some(DispatchPassSemantics::FuturePass)
        }
        RuntimePassKind::BlurVertical(Some(blur)) if blur.axis == RuntimeBlurAxis::Vertical => {
            Some(DispatchPassSemantics::FuturePass)
        }
        RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(_)
        | RuntimePassKind::BlurVertical(_)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => None,
        RuntimePassKind::Composite(Some(composite))
            if composite.source_captured_before_outer_semantics =>
        {
            Some(match composite.kind {
                RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::Layer { .. } => {
                    DispatchPassSemantics::ClosedExecutable
                }
                RuntimeCompositeKind::DropShadow => DispatchPassSemantics::FuturePass,
            })
        }
        RuntimePassKind::Composite(Some(_)) => None,
    }
}

enum GraphPreparationSource {
    C08(C08PreparableGraph),
    C09(ClosedExecutableGraph),
    C10 {
        preparable: C10PreparableGraph,
        operation_limits: Option<ColorFilterOperationBufferLimits>,
    },
    C11(C11PreparableGraph),
    C12(C12PreparableGraph),
}

type GraphPreparationParts = (
    LoweredGraphPlan,
    Option<C08ExecutionFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
);

impl GraphPreparationSource {
    const fn color_filter_operation_limits(&self) -> Option<ColorFilterOperationBufferLimits> {
        match self {
            Self::C10 {
                operation_limits, ..
            } => *operation_limits,
            Self::C08(_) | Self::C09(_) | Self::C11(_) | Self::C12(_) => None,
        }
    }

    fn into_parts(self) -> GraphPreparationParts {
        match self {
            Self::C08(preparable) => {
                let (lowered, execution) = preparable.into_parts();
                (lowered, Some(execution), None, None, None, None)
            }
            Self::C09(closed) => (closed.lowered, None, Some(closed.facts), None, None, None),
            Self::C10 { preparable, .. } => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, Some(closed.facts), None, None)
            }
            Self::C11(preparable) => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, None, Some(closed.facts), None)
            }
            Self::C12(preparable) => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, None, None, Some(closed.facts))
            }
        }
    }
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
enum C08ScheduledEncodingKind {
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
enum C08CustomSpineEncodingState {
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
    #[cfg(test)]
    pub(crate) c12_later_sibling_transition_is_exact: bool,
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
    #[cfg(test)]
    pub(crate) c11_pass_order: Vec<C11FilterPassTagForTest>,
    #[cfg(test)]
    pub(crate) capture_count: usize,
    #[cfg(test)]
    pub(crate) captures_share_one_command_encoder: bool,
    #[cfg(test)]
    pub(crate) captures_share_one_active_vello_scope: bool,
    #[cfg(test)]
    pub(crate) graph_work_shares_one_command_encoder: bool,
    #[cfg(test)]
    pub(crate) capture_observations: Vec<C08EncodedCaptureObservationForTest>,
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
    #[cfg(test)]
    capture_observations: Vec<C08EncodedCaptureObservationForTest>,
    #[cfg(test)]
    composite_encoder_identities: Vec<usize>,
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
            #[cfg(test)]
            capture_observations: Vec::with_capacity(expected_capture_count),
            #[cfg(test)]
            composite_encoder_identities: Vec::new(),
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
            #[cfg(test)]
            c12_later_sibling_transition_is_exact: c12.later_sibling_transition_is_exact,
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
            #[cfg(test)]
            c11_pass_order: c11_scheduled_pass_order_for_test(&self.scheduled),
            #[cfg(test)]
            capture_count: self.capture_count,
            #[cfg(test)]
            captures_share_one_command_encoder: self.captures_share_one_command_encoder(),
            #[cfg(test)]
            captures_share_one_active_vello_scope: self.captures_share_one_active_vello_scope(),
            #[cfg(test)]
            graph_work_shares_one_command_encoder: self.graph_work_shares_one_command_encoder(
                prepared.c10_execution.is_some()
                    || prepared.c11_execution.is_some()
                    || prepared.c12_execution.is_some(),
            ),
            #[cfg(test)]
            capture_observations: self.capture_observations,
        }
    }

    #[cfg(test)]
    fn captures_share_one_command_encoder(&self) -> bool {
        self.capture_observations.first().is_some_and(|first| {
            self.capture_observations.len() == self.expected_capture_count
                && self
                    .capture_observations
                    .iter()
                    .all(|capture| capture.encoder_identity == first.encoder_identity)
        })
    }

    #[cfg(test)]
    fn captures_share_one_active_vello_scope(&self) -> bool {
        self.capture_observations.first().is_some_and(|first| {
            self.capture_observations.len() == self.expected_capture_count
                && self
                    .capture_observations
                    .iter()
                    .all(|capture| capture.scope_identity == first.scope_identity)
        })
    }

    #[cfg(test)]
    fn graph_work_shares_one_command_encoder(&self, has_color_filters: bool) -> bool {
        self.capture_observations
            .first()
            .map(|capture| capture.encoder_identity)
            .or_else(|| self.composite_encoder_identities.first().copied())
            .is_some_and(|identity| {
                self.capture_observations
                    .iter()
                    .all(|capture| capture.encoder_identity == identity)
                    && self
                        .composite_encoder_identities
                        .iter()
                        .all(|composite| *composite == identity)
            })
            && (self.color_filter_count == 0 || has_color_filters)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct C12ExecutionReceipt {
    group_order_is_exact: bool,
    group_resources_are_distinct: bool,
    #[cfg(test)]
    later_sibling_transition_is_exact: bool,
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
    #[cfg(test)]
    let later_sibling_transition_is_exact =
        prepared.plan.passes.iter().skip(outer + 1).any(|pass| {
            pass.runtime
                .dependencies
                .contains(&backdrop.outer_composite)
                && pass
                    .runtime
                    .reads
                    .iter()
                    .any(|read| read.resource == backdrop.result)
        });
    C12ExecutionReceipt {
        group_order_is_exact: filters_are_ordered
            && clear < backdrop_composite
            && backdrop_composite < outer
            && foreground_is_ordered,
        group_resources_are_distinct: resources.len() == expected_resource_count,
        #[cfg(test)]
        later_sibling_transition_is_exact,
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct C08EncodedCaptureObservationForTest {
    pub(crate) lowers_with_exact_initial_transform: bool,
    pub(crate) uses_transparent_base: bool,
    pub(crate) antialiasing: Antialiasing,
    pub(crate) target_extent: PhysicalSize,
    pub(crate) target_format: wgpu::TextureFormat,
    pub(crate) target_usage: wgpu::TextureUsages,
    pub(crate) target_and_view_are_exact: bool,
    encoder_identity: usize,
    scope_identity: usize,
}

struct C08EncodedCaptureResult {
    receipt: C08VelloCaptureCompletionReceipt,
    #[cfg(test)]
    observation: C08EncodedCaptureObservationForTest,
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
    summary: C08CustomSpineEncodingSummary,
    resources: PendingVelloResourceCommit,
    session: Arc<()>,
}

#[cfg(test)]
impl C08PendingGraphEncoding {
    pub(crate) const fn summary_for_test(&self) -> &C08CustomSpineEncodingSummary {
        &self.summary
    }

    pub(crate) fn into_summary_and_resources(
        self,
    ) -> (C08CustomSpineEncodingSummary, PendingVelloResourceCommit) {
        (self.summary, self.resources)
    }
}

#[must_use = "prepared C08 frame state must commit only after graph transaction success"]
pub(crate) struct PendingC08PreparedFrameCommit {
    frame_scope: FrameResourceScope,
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

    #[cfg(test)]
    pub(crate) fn resource_identities_for_test(&self) -> Vec<ResourceIdentity> {
        self.frame_scope.leased_resource_identities_for_test()
    }

    #[cfg(test)]
    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        self.frame_scope.poison_retained_byte_accounting_for_test()
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
    #[cfg(test)]
    encoder_identity: usize,
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

fn c12_blur_edge_uniform_bytes(
    blur: &RuntimeBlur,
    source: &RuntimeReadBinding,
    request: &C08PreparedPassEncodingRequest,
) -> Result<Option<[u8; 16]>> {
    let RuntimeSamplingEdge::SemanticBorderMirror(bounds) = blur.edge else {
        if blur.edge != RuntimeSamplingEdge::TransparentBlack
            || source.sampling_edge() != RuntimeSamplingEdge::TransparentBlack
            || request.blur_edge_parameters.is_some()
        {
            return Err(preparation_error(
                "the C11 transparent blur changed its checked edge contract",
            ));
        }
        return Ok(None);
    };
    let expected = BlurEdgeParameterBytes::try_from_semantic_bounds(bounds)?;
    if source.sampling_edge() != blur.edge
        || request.blur_edge_parameters.as_ref() != Some(&expected)
    {
        return Err(preparation_error(
            "the C12 mirrored blur changed its checked semantic edge",
        ));
    }
    let values = [
        bounds.x() as f32,
        bounds.y() as f32,
        (bounds.x() + bounds.width()) as f32,
        (bounds.y() + bounds.height()) as f32,
    ];
    let mut bytes = [0_u8; 16];
    for (index, value) in values.into_iter().enumerate() {
        let offset = index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(Some(bytes))
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

#[cfg(test)]
fn c11_scheduled_pass_order_for_test(
    scheduled: &[C08ScheduledEncodingKind],
) -> Vec<C11FilterPassTagForTest> {
    scheduled
        .iter()
        .filter_map(|kind| match kind {
            C08ScheduledEncodingKind::ColorFilter => Some(C11FilterPassTagForTest::Color),
            C08ScheduledEncodingKind::BlurHorizontalRgba => {
                Some(C11FilterPassTagForTest::BlurHorizontalRgba)
            }
            C08ScheduledEncodingKind::BlurVerticalRgba => {
                Some(C11FilterPassTagForTest::BlurVerticalRgba)
            }
            C08ScheduledEncodingKind::BlurHorizontalSourceAlpha => {
                Some(C11FilterPassTagForTest::BlurHorizontalSourceAlpha)
            }
            C08ScheduledEncodingKind::BlurVerticalSourceAlpha => {
                Some(C11FilterPassTagForTest::BlurVerticalSourceAlpha)
            }
            C08ScheduledEncodingKind::DropShadowColorize => {
                Some(C11FilterPassTagForTest::DropShadowColorize)
            }
            C08ScheduledEncodingKind::DropShadowMerge => {
                Some(C11FilterPassTagForTest::DropShadowMerge)
            }
            C08ScheduledEncodingKind::ClearRoot
            | C08ScheduledEncodingKind::VelloCapture
            | C08ScheduledEncodingKind::CanonicalizeCapture
            | C08ScheduledEncodingKind::CopyBackdrop
            | C08ScheduledEncodingKind::SpanSourceOver
            | C08ScheduledEncodingKind::LayerComposite
            | C08ScheduledEncodingKind::Present => None,
        })
        .collect()
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
pub(crate) struct PreparedGraph<'device> {
    plan: RuntimeGraphPreparationPlan,
    c08_execution: Option<C08ExecutionFacts>,
    c09_execution: Option<ClosedExecutableGraphFacts>,
    c10_execution: Option<ClosedExecutableGraphFacts>,
    c11_execution: Option<ClosedExecutableGraphFacts>,
    c12_execution: Option<ClosedExecutableGraphFacts>,
    resource_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
    color_filter_operation_bindings: BTreeMap<RuntimePassId, PreparedColorFilterOperationBinding>,
    c11_pass_objects: BTreeMap<RuntimePassId, PreparedC11PassObjects>,
    pass_cache_update: Option<ProvisionalDevicePassCacheUpdate>,
    frame_scope: Option<FrameResourceScope>,
    next_pass: usize,
    c08_encoding_state: Option<C08CustomSpineEncodingState>,
    c08_completed_session: Option<Arc<()>>,
    #[cfg(test)]
    fail_capture_encoding_after_for_test: Option<usize>,
    #[cfg(test)]
    fail_scope_resolution_for_test: bool,
    #[cfg(test)]
    acquired_capture_lease_count_for_test: usize,
    device: &'device wgpu::Device,
    queue: &'device wgpu::Queue,
    vello_engine: Option<&'device VelloEngineState>,
    resources: &'device ResourceManager,
    pass_cache: &'device DevicePassCache,
    _ready_device: PhantomData<&'device ResourceManager>,
}

struct AcquiredGraphBindings {
    runtime_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    gaussian_kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
}

fn acquire_prepared_graph_resources(
    plan: &RuntimeGraphPreparationPlan,
    frame_scope: &mut FrameResourceScope,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    capabilities: &DeviceCapabilities,
) -> Result<AcquiredGraphBindings> {
    let mut runtime_bindings = BTreeMap::new();
    for request in &plan.resources {
        let lease = request
            .allocation
            .acquire(frame_scope, device, queue, capabilities)?;
        if runtime_bindings
            .insert(
                request.runtime.id,
                PreparedResourceBinding {
                    allocation: request.allocation.clone(),
                    lease: Some(lease),
                },
            )
            .is_some()
        {
            return Err(preparation_error(
                "one runtime resource acquired more than one concrete binding",
            ));
        }
    }
    let mut gaussian_kernel_bindings = BTreeMap::new();
    for request in &plan.kernels {
        let lease = frame_scope.acquire_gaussian_kernel_buffer(device, &request.plan)?;
        if gaussian_kernel_bindings
            .insert(request.key, PreparedKernelBinding { lease: Some(lease) })
            .is_some()
        {
            return Err(preparation_error(
                "one Gaussian kernel acquired more than one concrete binding",
            ));
        }
    }
    Ok(AcquiredGraphBindings {
        runtime_bindings,
        gaussian_kernel_bindings,
    })
}

fn create_color_filter_operation_bindings(
    plan: &RuntimeGraphPreparationPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<BTreeMap<RuntimePassId, PreparedColorFilterOperationBinding>> {
    let mut bindings = BTreeMap::new();
    for request in &plan.passes {
        let Some(bytes) = request.color_filter_operations.as_ref() else {
            continue;
        };
        if !matches!(request.runtime.kind, RuntimePassKind::ColorFilter(Some(_)))
            || bytes.as_bytes().is_empty()
        {
            return Err(preparation_error(
                "prepared color-filter bytes have no exact runtime pass",
            ));
        }
        let size = u64::try_from(bytes.as_bytes().len()).map_err(|_| {
            preparation_error("prepared color-filter buffer length does not fit u64")
        })?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C10 ordered color-filter operations"),
            size,
            usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytes.as_bytes());
        if bindings
            .insert(
                request.runtime.id,
                PreparedColorFilterOperationBinding {
                    bytes: bytes.clone(),
                    buffer: Some(buffer),
                },
            )
            .is_some()
        {
            return Err(preparation_error(
                "one color-filter pass acquired more than one operation buffer",
            ));
        }
    }
    Ok(bindings)
}

fn realize_prepared_graph_passes(
    plan: &RuntimeGraphPreparationPlan,
    device: &wgpu::Device,
    pass_cache: &DevicePassCache,
    enabled: bool,
) -> Result<PreparedPassRealization> {
    if !enabled {
        return Ok(PreparedPassRealization {
            update: None,
            c11_objects: BTreeMap::new(),
        });
    }
    let mut update = pass_cache.provisional_update();
    let mut realized_pass = false;
    let mut c11_objects = BTreeMap::new();
    for request in &plan.passes {
        let Some(keys) = request.cache_keys.as_ref() else {
            continue;
        };
        let objects = realize_prepared_graph_pass(&mut update, device, pass_cache, request, keys)?;
        if let Some(objects) = objects
            && c11_objects.insert(request.runtime.id, objects).is_some()
        {
            return Err(preparation_error(
                "one C11 pass retained more than one prepared object set",
            ));
        }
        realized_pass = true;
    }
    Ok(PreparedPassRealization {
        update: realized_pass.then_some(update),
        c11_objects,
    })
}

fn realize_prepared_graph_pass(
    update: &mut ProvisionalDevicePassCacheUpdate,
    device: &wgpu::Device,
    pass_cache: &DevicePassCache,
    request: &RuntimePassPreparationRequest,
    keys: &RuntimePassCacheKeys,
) -> Result<Option<PreparedC11PassObjects>> {
    match &request.runtime.kind {
        RuntimePassKind::CopyBackdrop => {
            let objects = realize_copy_backdrop_for_preparation(update, device, pass_cache, keys)?;
            Ok(Some(PreparedC11PassObjects::CopyBackdrop {
                parent_sampler: objects.parent_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::ColorFilter(Some(_)) => {
            update
                .realize_color_filter_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_)) => {
            let objects = update.realize_blur_pass(
                device,
                pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            Ok(Some(PreparedC11PassObjects::Blur {
                source_sampler: objects.source_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::DropShadowColorize(Some(_)) => {
            let objects = update.realize_drop_shadow_colorize_pass(
                device,
                pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            Ok(Some(PreparedC11PassObjects::DropShadowColorize {
                source_sampler: objects.source_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { .. },
            ..
        })) => {
            update
                .realize_composite_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::CanonicalizeCapture
        | RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::DropShadow,
            ..
        }))
        | RuntimePassKind::Present => {
            update
                .realize_c08_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Err(preparation_error(
            "checked pass realization reached an unsupported graph pass",
        )),
    }
}

fn realize_copy_backdrop_for_preparation<'a>(
    update: &'a mut ProvisionalDevicePassCacheUpdate,
    device: &wgpu::Device,
    pass_cache: &'a DevicePassCache,
    keys: &RuntimePassCacheKeys,
) -> Result<super::shader::ProvisionalCopyBackdropPassObjects<'a>> {
    let objects = update.realize_copy_backdrop_pass(
        device,
        pass_cache,
        keys.samplers(),
        keys.layout(),
        keys.shader(),
        keys.pipeline(),
    )?;
    objects.require_encoding_ready()?;
    Ok(objects)
}

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
    pub(crate) fn try_prepare_c08(
        preparable: C08PreparableGraph,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache: &'device DevicePassCache,
    ) -> Result<Self> {
        let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
        Self::try_prepare_c08_with_working_format(
            preparable,
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            (pass_cache, false),
        )
    }

    pub(crate) fn try_prepare_c08_with_working_format(
        preparable: C08PreparableGraph,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C08(preparable),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c08_execution_facts().is_none() {
            return Err(preparation_error(
                "C08 preparation lost its validated execution facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare_c09(
        preparable: C09PreparableGraph,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let selected_working_format = preparable.working_format();
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C09(preparable.into_closed()),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c09_execution.is_none() {
            return Err(preparation_error(
                "C09 preparation lost its validated closed execution facts",
            ));
        }
        Ok(prepared)
    }

    #[cfg(test)]
    pub(crate) fn try_prepare_c10(
        preparable: C10PreparableGraph,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let selected_working_format = preparable.working_format();
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C10 {
                preparable,
                operation_limits: None,
            },
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c10_execution.is_none() {
            return Err(preparation_error(
                "C10 preparation lost its validated closed execution facts",
            ));
        }
        Ok(prepared)
    }

    #[cfg(test)]
    pub(crate) fn try_prepare_c11(
        preparable: C11PreparableGraph,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let selected_working_format = preparable.working_format();
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C11(preparable),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c11_execution.is_none() {
            return Err(preparation_error(
                "C11 preparation lost its validated closed execution facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare_c12(
        preparable: C12PreparableGraph,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C12(preparable),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c12_execution.is_none() {
            return Err(preparation_error(
                "C12 preparation lost its validated closed backdrop facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare(
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        match PrePreparationGraphClassification::classify(lowered) {
            PrePreparationGraphClassification::ExactC08(preparable) if pass_cache_phase.1 => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C08(preparable),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c08_execution_facts().is_none() {
                    return Err(preparation_error(
                        "C08 preparation lost its validated execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC08(preparable) => Self::try_prepare_c08(
                preparable,
                policy,
                capabilities,
                device,
                queue,
                resources,
                pass_cache_phase.0,
            ),
            PrePreparationGraphClassification::ExactC09(closed) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                Self::try_prepare_inner(
                    GraphPreparationSource::C09(closed),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )
            }
            PrePreparationGraphClassification::ExactC10(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C10 {
                        preparable,
                        operation_limits: None,
                    },
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c10_execution.is_none() {
                    return Err(preparation_error(
                        "C10 preparation lost its validated closed execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC11(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C11(preparable),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c11_execution.is_none() {
                    return Err(preparation_error(
                        "C11 preparation lost its validated closed execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC12(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                Self::try_prepare_c12(
                    preparable,
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )
            }
            PrePreparationGraphClassification::FuturePasses => Err(preparation_error(
                "a future GPU pass cannot enter C09 resource preparation",
            )),
            PrePreparationGraphClassification::Ineligible(ineligibility) => {
                Err(ineligibility.into_error())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn try_prepare_c10_with_operation_limits_for_test(
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_and_limits: (&'device DevicePassCache, ColorFilterOperationBufferLimits),
    ) -> Result<Self> {
        let (pass_cache, operation_limits) = pass_cache_and_limits;
        let PrePreparationGraphClassification::ExactC10(preparable) =
            PrePreparationGraphClassification::classify(lowered)
        else {
            return Err(preparation_error(
                "the C10 limit fixture requires one exact closed color graph",
            ));
        };
        let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C10 {
                preparable,
                operation_limits: Some(operation_limits),
            },
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            (pass_cache, true),
        )?;
        if prepared.c10_execution.is_none() {
            return Err(preparation_error(
                "C10 limit preparation lost its validated closed execution facts",
            ));
        }
        Ok(prepared)
    }

    fn try_prepare_inner(
        source: GraphPreparationSource,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let (pass_cache, realize_checked_passes) = pass_cache_phase;
        let color_filter_operation_limits = source.color_filter_operation_limits();
        let (lowered, c08_execution, c09_execution, c10_execution, c11_execution, c12_execution) =
            source.into_parts();
        let plan = match color_filter_operation_limits {
            Some(limits) => RuntimeGraphPreparationPlan::try_derive_with_color_filter_limits(
                lowered,
                selected_working_format,
                capabilities,
                device,
                limits,
            )?,
            None => RuntimeGraphPreparationPlan::try_derive(
                lowered,
                selected_working_format,
                capabilities,
                device,
            )?,
        };
        resources.preflight_graph_acquisitions(&plan.allocation_preflights)?;

        let mut frame_scope = resources.begin_frame()?;
        frame_scope.abort_provisional_on_drop();
        if c08_execution.is_some()
            || c09_execution.is_some()
            || c10_execution.is_some()
            || c11_execution.is_some()
            || c12_execution.is_some()
        {
            frame_scope.discard_on_drop();
        }
        let acquired_resources =
            acquire_prepared_graph_resources(&plan, &mut frame_scope, device, queue, capabilities)?;
        let color_filter_operation_bindings =
            create_color_filter_operation_bindings(&plan, device, queue)?;
        let pass_realization =
            realize_prepared_graph_passes(&plan, device, pass_cache, realize_checked_passes)?;
        let c08_encoding_state = (c08_execution.is_some()
            || c09_execution.is_some()
            || c10_execution.is_some()
            || c11_execution.is_some()
            || c12_execution.is_some())
        .then_some(C08CustomSpineEncodingState::Ready);

        Ok(Self {
            plan,
            c08_execution,
            c09_execution,
            c10_execution,
            c11_execution,
            c12_execution,
            resource_bindings: acquired_resources.runtime_bindings,
            kernel_bindings: acquired_resources.gaussian_kernel_bindings,
            color_filter_operation_bindings,
            c11_pass_objects: pass_realization.c11_objects,
            pass_cache_update: pass_realization.update,
            frame_scope: Some(frame_scope),
            next_pass: 0,
            c08_encoding_state,
            c08_completed_session: None,
            #[cfg(test)]
            fail_capture_encoding_after_for_test: None,
            #[cfg(test)]
            fail_scope_resolution_for_test: false,
            #[cfg(test)]
            acquired_capture_lease_count_for_test: 0,
            device,
            queue,
            vello_engine: None,
            resources,
            pass_cache,
            _ready_device: PhantomData,
        })
    }

    pub(crate) fn with_vello_engine(mut self, engine: &'device VelloEngineState) -> Self {
        self.vello_engine = Some(engine);
        self
    }

    #[cfg(test)]
    pub(crate) fn fail_capture_encoding_for_test(&mut self) {
        self.fail_capture_encoding_after_for_test = Some(0);
    }

    #[cfg(test)]
    pub(crate) fn fail_capture_encoding_after_for_test(&mut self, successful_capture_count: usize) {
        self.fail_capture_encoding_after_for_test = Some(successful_capture_count);
    }

    #[cfg(test)]
    pub(crate) fn fail_scope_resolution_for_test(&mut self) {
        self.fail_scope_resolution_for_test = true;
    }

    #[cfg(test)]
    pub(crate) const fn acquired_capture_lease_count_for_test(&self) -> usize {
        self.acquired_capture_lease_count_for_test
    }

    pub(crate) const fn c08_execution_facts(&self) -> Option<&C08ExecutionFacts> {
        self.c08_execution.as_ref()
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the typed prepared graph generation for stale-binding checks"
        )
    )]
    pub(crate) const fn generation(&self) -> RuntimeGraphGeneration {
        self.plan.generation
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.plan.working_format
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.plan.output_format
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the typed prepared root and terminal identities"
        )
    )]
    pub(crate) const fn root_and_final(&self) -> (RuntimeResourceId, RuntimePassId) {
        (self.plan.root_working_image, self.plan.final_present)
    }

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.resource_request(self.plan.root_working_image)
            .map(|resource| resource.spatial.device_extent)
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
        if self.fail_scope_resolution_for_test {
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
        if self.fail_capture_encoding_after_for_test == Some(progress.capture_count) {
            return Err(preparation_error(
                "injected C08 Vello capture encoding failure",
            ));
        }
        let handoff = self.c08_vello_capture_handoff(request, session)?;
        let target = handoff.target();
        progress.bounded_capture_handoffs &= c08_capture_handoff_is_bounded(&handoff);
        let encoded = Self::encode_c08_vello_capture(handoff, encoder, capture_encoding)?;
        #[cfg(test)]
        progress.capture_observations.push(encoded.observation);
        self.complete_c08_capture(request.id, target, session, encoded.receipt)?;
        #[cfg(test)]
        {
            self.acquired_capture_lease_count_for_test =
                self.acquired_capture_lease_count_for_test.saturating_add(1);
        }
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
        progress
            .composite_encoder_identities
            .push(std::ptr::from_mut(&mut *encoder) as usize);
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
        progress
            .composite_encoder_identities
            .push(std::ptr::from_mut(&mut *encoder) as usize);
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
        progress
            .composite_encoder_identities
            .push(std::ptr::from_mut(&mut *encoder) as usize);
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
        progress
            .composite_encoder_identities
            .push(std::ptr::from_mut(&mut *encoder) as usize);
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
        progress
            .composite_encoder_identities
            .push(facts.encoder_identity);
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
        #[cfg(test)]
        let lowers_with_exact_initial_transform = match handoff.work() {
            RuntimeVelloCapture::Span(_) => scene
                .observation_for_test()
                .first_glyph_run_for_test()
                .is_some_and(|run| {
                    run.transform_components_for_test()
                        .iter()
                        .zip(handoff.initial_transform().as_array())
                        .all(|(actual, expected)| (*actual - expected as f32).abs() <= 1.0e-5)
                }),
            RuntimeVelloCapture::ClipCoverage(_) => true,
        };
        let prepared = scene.prepare_raster(vello_capture_raster_parameters(
            target_extent,
            antialiasing,
        )?)?;
        #[cfg(test)]
        let encoder_identity = std::ptr::from_mut(&mut *encoder) as usize;
        #[cfg(test)]
        let scope_identity = std::ptr::from_ref(&*capture_encoding.scope) as usize;
        #[cfg(test)]
        let target_view_identity = std::ptr::from_ref(handoff.view()) as usize;
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
        let observation = C08EncodedCaptureObservationForTest {
            lowers_with_exact_initial_transform,
            uses_transparent_base: proof.transparent_base_for_test(),
            antialiasing: proof.antialiasing_for_test(),
            target_extent: proof.target_extent_for_test(),
            target_format: proof.target_format_for_test(),
            target_usage: proof.target_usage_for_test(),
            target_and_view_are_exact: handoff.texture().format() == proof.target_format_for_test()
                && handoff.texture().width() == proof.target_extent_for_test().width()
                && handoff.texture().height() == proof.target_extent_for_test().height()
                && proof.target_view_identity_for_test() == target_view_identity,
            encoder_identity,
            scope_identity,
        };
        let receipt = match handoff.complete_after_encoded_capture(proof) {
            Ok(receipt) => receipt,
            Err(error) => {
                let _ = lease.abort();
                return Err(error);
            }
        };
        capture_encoding.leases.push(lease);
        Ok(C08EncodedCaptureResult {
            receipt,
            #[cfg(test)]
            observation,
        })
    }

    fn resource_request(&self, resource: RuntimeResourceId) -> Result<&RuntimeResourceRequest> {
        self.plan
            .resources
            .iter()
            .find(|request| request.runtime.id == resource)
            .map(|request| &request.runtime)
            .ok_or_else(|| preparation_error("the prepared C08 resource request is missing"))
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
        let capture =
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
        let edge_bytes = c12_blur_edge_uniform_bytes(blur, source, request)?;
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
        #[cfg(test)]
        let encoder_identity = std::ptr::from_mut(&mut *encoder) as usize;
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
            #[cfg(test)]
            encoder_identity,
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
            reason = "C08 consumes prepared pass requests through this narrow iterator"
        )
    )]
    pub(crate) fn current_pass(&self) -> Option<PreparedPassView<'_>> {
        self.plan
            .passes
            .get(self.next_pass)
            .map(|request| PreparedPassView { request })
    }

    fn require_current_pass(&self, pass: RuntimePassId) -> Result<&RuntimePassPreparationRequest> {
        let request = self
            .plan
            .passes
            .get(self.next_pass)
            .ok_or_else(|| preparation_error("prepared graph has no remaining pass"))?;
        if request.runtime.id != pass {
            return Err(preparation_error(
                "prepared pass request is missing, stale, duplicate, or out of order",
            ));
        }
        Ok(request)
    }

    pub(crate) fn texture_binding_for_pass(
        &self,
        pass: RuntimePassId,
        resource: RuntimeResourceId,
    ) -> Result<PreparedTextureBinding<'_>> {
        let request = self.require_current_pass(pass)?;
        let bound = request
            .runtime
            .reads
            .iter()
            .any(|read| read.resource == resource)
            || request.runtime.result == RuntimeResultBinding::Resource(resource);
        if !bound {
            return Err(preparation_error(
                "runtime resource is not bound to the requested prepared pass",
            ));
        }
        let binding = self
            .resource_bindings
            .get(&resource)
            .ok_or_else(|| preparation_error("prepared runtime resource binding is missing"))?;
        let lease = binding.lease.as_ref().ok_or_else(|| {
            preparation_error("prepared runtime resource binding is stale or already released")
        })?;
        let frame_scope = self
            .frame_scope
            .as_ref()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?;
        let (texture, view) = match &binding.allocation {
            RuntimeAllocationRequest::EffectTexture(_) => frame_scope.effect_texture(lease)?,
            RuntimeAllocationRequest::ResolvedMask(_) => {
                frame_scope.resolved_mask_texture(lease)?
            }
        };
        Ok(PreparedTextureBinding {
            runtime_resource: resource,
            allocation_resource: lease.resource_identity(),
            texture,
            view,
        })
    }

    pub(crate) fn gaussian_kernel_binding_for_pass(
        &self,
        pass: RuntimePassId,
    ) -> Result<Option<PreparedGaussianKernelBinding<'_>>> {
        let request = self.require_current_pass(pass)?;
        let Some(kernel) = request.kernel else {
            return Ok(None);
        };
        let binding = self
            .kernel_bindings
            .get(&kernel)
            .ok_or_else(|| preparation_error("prepared Gaussian kernel binding is missing"))?;
        let lease = binding.lease.as_ref().ok_or_else(|| {
            preparation_error("prepared Gaussian kernel binding is stale or already released")
        })?;
        let frame_scope = self
            .frame_scope
            .as_ref()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?;
        Ok(Some(PreparedGaussianKernelBinding {
            key: kernel,
            allocation_resource: lease.resource_identity(),
            buffer: frame_scope.gaussian_kernel_buffer(lease)?,
        }))
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

    #[cfg(test)]
    pub(crate) fn allocation_identities_for_test(&self) -> PreparedAllocationIdentitiesForTest {
        PreparedAllocationIdentitiesForTest {
            resources: self
                .resource_bindings
                .iter()
                .map(|(runtime, binding)| {
                    (
                        *runtime,
                        binding
                            .lease
                            .as_ref()
                            .expect("new prepared resources must own live leases")
                            .resource_identity(),
                    )
                })
                .collect(),
            kernels: self
                .kernel_bindings
                .iter()
                .map(|(kernel, binding)| {
                    (
                        *kernel,
                        binding
                            .lease
                            .as_ref()
                            .expect("new prepared kernels must own live leases")
                            .resource_identity(),
                    )
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn exercise_for_test(&mut self) -> Result<PreparedGraphExerciseObservationForTest> {
        if self.c09_execution.is_some() {
            self.c08_encoding_state = None;
        }
        let _ = (
            self.generation(),
            self.working_format(),
            self.output_format(),
            self.root_and_final(),
        );
        let complete_resource_and_pass_handoff = prepared_handoff_is_complete(&self.plan);
        let exact_capture_coverage_working_and_mask_allocations =
            prepared_allocations_are_exact(&self.plan);
        let spatial_bytes_and_cache_keys_preserved = prepared_spatial_keys_are_exact(&self.plan);
        let typed_bindings_and_last_use_releases = exercise_prepared_bindings(self)?;

        Ok(PreparedGraphExerciseObservationForTest {
            complete_resource_and_pass_handoff,
            exact_capture_coverage_working_and_mask_allocations,
            typed_bindings_and_last_use_releases,
            spatial_bytes_and_cache_keys_preserved,
        })
    }

    #[cfg(test)]
    fn outstanding_lease_count_for_test(&self) -> usize {
        self.resource_bindings
            .values()
            .filter(|binding| binding.lease.is_some())
            .count()
            + self
                .kernel_bindings
                .values()
                .filter(|binding| binding.lease.is_some())
                .count()
    }
}

#[cfg(test)]
fn prepared_handoff_is_complete(plan: &RuntimeGraphPreparationPlan) -> bool {
    let mut vocabulary = [false; 10];
    for pass in &plan.passes {
        vocabulary[runtime_pass_kind_index(&pass.runtime.kind)] = true;
    }
    let closed_c09_vocabulary = vocabulary[0]
        && vocabulary[1]
        && vocabulary[2]
        && vocabulary[8]
        && vocabulary[9]
        && vocabulary[3..8].iter().all(|present| !present);
    plan.resources
        .iter()
        .find(|resource| resource.runtime.role == RuntimeResourceRole::RootWorkingImage)
        .map(|resource| resource.runtime.id)
        == Some(plan.root_working_image)
        && plan
            .passes
            .last()
            .is_some_and(|pass| pass.runtime.id == plan.final_present)
        && closed_c09_vocabulary
        && !plan.resources.is_empty()
        && plan.kernels.is_empty()
}

#[cfg(test)]
fn prepared_allocations_are_exact(plan: &RuntimeGraphPreparationPlan) -> bool {
    let mut present = [false; 4];
    let exact_resources =
        plan.resources.iter().all(
            |request| match (&request.runtime.format, &request.allocation) {
                (
                    RuntimeResourceFormat::VelloCaptureRgba8Unorm,
                    RuntimeAllocationRequest::EffectTexture(descriptor),
                ) => {
                    present[0] = true;
                    descriptor.role() == EffectTextureRole::Capture
                        && descriptor.working_format().is_none()
                        && descriptor.texture_format() == wgpu::TextureFormat::Rgba8Unorm
                        && descriptor.usage() == VELLO_CAPTURE_TEXTURE_USAGES
                }
                (
                    RuntimeResourceFormat::ClipCoverageRgba8Unorm,
                    RuntimeAllocationRequest::EffectTexture(descriptor),
                ) => {
                    present[1] = true;
                    descriptor.role() == EffectTextureRole::Coverage
                        && descriptor.working_format().is_none()
                        && descriptor.texture_format() == wgpu::TextureFormat::Rgba8Unorm
                        && descriptor.usage() == VELLO_CAPTURE_TEXTURE_USAGES
                }
                (
                    RuntimeResourceFormat::Working(format),
                    RuntimeAllocationRequest::EffectTexture(descriptor),
                ) => {
                    present[2] = true;
                    descriptor.role() == EffectTextureRole::Working
                        && descriptor.working_format() == Some(*format)
                        && descriptor.texture_format() == format.texture_format()
                        && descriptor.usage() == format.required_usages()
                }
                (
                    RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
                    RuntimeAllocationRequest::ResolvedMask(descriptor),
                ) => {
                    present[3] = true;
                    matches!(
                        &request.runtime.import,
                        Some(RuntimeResourceImport::ResolvedAlphaMask(runtime))
                            if runtime.cache_key() == descriptor.cache_key()
                                && runtime.physical_size() == descriptor.physical_size()
                    )
                }
                _ => false,
            },
        );
    let exact_kernels = plan.kernels.iter().all(|kernel| {
        kernel.key == kernel.plan.key()
            && kernel.plan.byte_len() > 0
            && plan
                .passes
                .iter()
                .any(|pass| pass.kernel == Some(kernel.key))
    });
    present.into_iter().all(|value| value)
        && exact_resources
        && exact_kernels
        && plan.kernels.is_empty()
}

#[cfg(test)]
fn prepared_spatial_keys_are_exact(plan: &RuntimeGraphPreparationPlan) -> bool {
    plan.passes.iter().all(|pass| {
        pass.cache_keys == pass.runtime.cache_keys
            && pass.spatial_uniform.is_some() == pass.cache_keys.is_some()
            && pass
                .spatial_uniform
                .as_ref()
                .is_none_or(|bytes| bytes.as_bytes().len() == 48)
    })
}

#[cfg(test)]
fn prepared_exercise_rejections(graph: &mut PreparedGraph<'_>) -> Result<(bool, bool)> {
    let initial_pass = graph
        .current_pass()
        .ok_or_else(|| preparation_error("prepared test graph has no first pass"))?
        .id();
    let initial_outstanding = graph.outstanding_lease_count_for_test();
    let out_of_order = graph.plan.final_present != initial_pass
        && graph.complete_pass(graph.plan.final_present).is_err()
        && graph.next_pass == 0
        && graph.outstanding_lease_count_for_test() == initial_outstanding;
    let unrelated_resource = graph.plan.resources.iter().find_map(|resource| {
        let bound = graph.plan.passes[0]
            .runtime
            .reads
            .iter()
            .any(|read| read.resource == resource.runtime.id)
            || graph.plan.passes[0].runtime.result
                == RuntimeResultBinding::Resource(resource.runtime.id);
        (!bound).then_some(resource.runtime.id)
    });
    let missing_binding = unrelated_resource.is_some_and(|resource| {
        graph
            .texture_binding_for_pass(initial_pass, resource)
            .is_err()
            && graph.next_pass == 0
            && graph.outstanding_lease_count_for_test() == initial_outstanding
    });
    Ok((out_of_order, missing_binding))
}

#[cfg(test)]
fn exercise_prepared_bindings(graph: &mut PreparedGraph<'_>) -> Result<bool> {
    let (out_of_order, missing_binding) = prepared_exercise_rejections(graph)?;
    let mut bindings_inspected = true;
    let mut releases_exact = true;
    let mut duplicate_release = false;
    let mut completed = 0usize;
    while let Some(pass) = graph.current_pass() {
        let pass_id = pass.id();
        bindings_inspected &= prepared_pass_view_is_consistent(&pass);
        let bound_resources = pass.bound_resources_for_test();
        let resource_releases = pass.resource_releases_for_test().to_vec();
        let kernel_releases = pass.kernel_releases_for_test().to_vec();
        for resource in bound_resources {
            let binding = graph.texture_binding_for_pass(pass_id, resource)?;
            bindings_inspected &= binding.runtime_resource() == resource
                && binding.allocation_resource().get() > 0
                && binding.texture().width() > 0;
            let _ = binding.view();
        }
        if let Some(binding) = graph.gaussian_kernel_binding_for_pass(pass_id)? {
            bindings_inspected &= binding.allocation_resource().get() > 0
                && graph
                    .plan
                    .passes
                    .get(graph.next_pass)
                    .is_some_and(|request| request.kernel == Some(binding.key()));
            let _ = binding.buffer();
        }
        graph.complete_pass(pass_id)?;
        releases_exact &= resource_releases.iter().all(|resource| {
            graph
                .resource_bindings
                .get(resource)
                .is_some_and(|binding| binding.lease.is_none())
        }) && kernel_releases.iter().all(|kernel| {
            graph
                .kernel_bindings
                .get(kernel)
                .is_some_and(|binding| binding.lease.is_none())
        });
        if completed == 0 {
            let after_first = graph.outstanding_lease_count_for_test();
            duplicate_release = graph.complete_pass(pass_id).is_err()
                && graph.outstanding_lease_count_for_test() == after_first
                && graph.next_pass == 1;
        }
        completed = completed.saturating_add(1);
    }
    Ok(out_of_order
        && missing_binding
        && duplicate_release
        && bindings_inspected
        && releases_exact
        && completed == graph.plan.passes.len()
        && graph.outstanding_lease_count_for_test() == 0)
}

#[cfg(test)]
fn prepared_pass_view_is_consistent(pass: &PreparedPassView<'_>) -> bool {
    let _ = (pass.kind(), pass.dependencies(), pass.result());
    let reads_are_accessible = pass.reads().iter().all(|read| {
        let _ = (
            read.role(),
            read.resource(),
            read.sampling_filter(),
            read.sampling_edge(),
            read.sampler_key(),
        );
        true
    });
    if let Some(keys) = pass.cache_keys() {
        let _ = (
            keys.samplers(),
            keys.layout(),
            keys.shader(),
            keys.pipeline(),
        );
    }
    reads_are_accessible
        && pass
            .spatial_uniform()
            .is_some_and(|bytes| bytes.as_bytes().len() == 48)
            == pass.cache_keys().is_some()
        && pass
            .blur_edge_parameters()
            .is_some_and(|bytes| bytes.as_bytes().len() == 16)
            == matches!(
                pass.kind(),
                RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                    edge: RuntimeSamplingEdge::SemanticBorderMirror(_),
                    ..
                })) | RuntimePassKind::BlurVertical(Some(RuntimeBlur {
                    edge: RuntimeSamplingEdge::SemanticBorderMirror(_),
                    ..
                }))
            )
        && pass
            .composite_parameters()
            .is_some_and(|bytes| bytes.as_bytes().len() == 112)
            == matches!(
                pass.kind(),
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::Layer { .. },
                    ..
                }))
            )
        && pass
            .drop_shadow_parameters()
            .is_some_and(|bytes| bytes.as_bytes().len() == 32)
            == matches!(pass.kind(), RuntimePassKind::DropShadowColorize(Some(_)))
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes this immutable view of the current prepared runtime pass"
    )
)]
pub(crate) struct PreparedPassView<'prepared> {
    request: &'prepared RuntimePassPreparationRequest,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these narrow immutable prepared-pass facts"
    )
)]
impl PreparedPassView<'_> {
    pub(crate) const fn id(&self) -> RuntimePassId {
        self.request.runtime.id
    }

    pub(crate) const fn kind(&self) -> &RuntimePassKind {
        &self.request.runtime.kind
    }

    pub(crate) fn dependencies(&self) -> &[RuntimePassId] {
        &self.request.runtime.dependencies
    }

    pub(crate) fn reads(&self) -> &[RuntimeReadBinding] {
        &self.request.runtime.reads
    }

    pub(crate) const fn result(&self) -> RuntimeResultBinding {
        self.request.runtime.result
    }

    pub(crate) const fn spatial_uniform(&self) -> Option<&PassSpatialUniformBytes> {
        self.request.spatial_uniform.as_ref()
    }

    pub(crate) const fn blur_edge_parameters(&self) -> Option<&BlurEdgeParameterBytes> {
        self.request.blur_edge_parameters.as_ref()
    }

    pub(crate) const fn composite_parameters(&self) -> Option<&CompositeParameterBytes> {
        self.request.composite_parameters.as_ref()
    }

    pub(crate) const fn drop_shadow_parameters(&self) -> Option<&DropShadowParameterBytes> {
        self.request.drop_shadow_parameters.as_ref()
    }

    pub(crate) const fn cache_keys(&self) -> Option<&RuntimePassCacheKeys> {
        self.request.cache_keys.as_ref()
    }

    #[cfg(test)]
    fn bound_resources_for_test(&self) -> Vec<RuntimeResourceId> {
        let mut resources = self
            .request
            .runtime
            .reads
            .iter()
            .map(|read| read.resource)
            .collect::<Vec<_>>();
        if let RuntimeResultBinding::Resource(resource) = self.request.runtime.result {
            resources.push(resource);
        }
        resources
    }

    #[cfg(test)]
    fn resource_releases_for_test(&self) -> &[RuntimeResourceId] {
        &self.request.runtime.releases
    }

    #[cfg(test)]
    fn kernel_releases_for_test(&self) -> &[GaussianKernelKey] {
        &self.request.kernel_releases
    }
}

pub(crate) struct PreparedTextureBinding<'prepared> {
    runtime_resource: RuntimeResourceId,
    allocation_resource: ResourceIdentity,
    texture: &'prepared wgpu::Texture,
    view: &'prepared wgpu::TextureView,
}

impl<'prepared> PreparedTextureBinding<'prepared> {
    pub(crate) const fn runtime_resource(&self) -> RuntimeResourceId {
        self.runtime_resource
    }

    pub(crate) const fn allocation_resource(&self) -> ResourceIdentity {
        self.allocation_resource
    }

    pub(crate) const fn texture(&self) -> &'prepared wgpu::Texture {
        self.texture
    }

    pub(crate) const fn view(&self) -> &'prepared wgpu::TextureView {
        self.view
    }
}

pub(crate) struct PreparedGaussianKernelBinding<'prepared> {
    key: GaussianKernelKey,
    allocation_resource: ResourceIdentity,
    buffer: &'prepared wgpu::Buffer,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 reads these exact typed Gaussian binding facts during blur encoding"
    )
)]
impl<'prepared> PreparedGaussianKernelBinding<'prepared> {
    pub(crate) const fn key(&self) -> GaussianKernelKey {
        self.key
    }

    pub(crate) const fn allocation_resource(&self) -> ResourceIdentity {
        self.allocation_resource
    }

    pub(crate) const fn buffer(&self) -> &'prepared wgpu::Buffer {
        self.buffer
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedAllocationIdentitiesForTest {
    resources: Vec<(RuntimeResourceId, ResourceIdentity)>,
    kernels: Vec<(GaussianKernelKey, ResourceIdentity)>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedGraphExerciseObservationForTest {
    pub(crate) complete_resource_and_pass_handoff: bool,
    pub(crate) exact_capture_coverage_working_and_mask_allocations: bool,
    pub(crate) typed_bindings_and_last_use_releases: bool,
    pub(crate) spatial_bytes_and_cache_keys_preserved: bool,
}

fn lowering_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

fn runtime_affine_is_finite_and_non_singular(transform: Transform) -> bool {
    let [a, b, c, d, e, f] = transform.as_array();
    if [a, b, c, d, e, f]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == 0.0 {
        return false;
    }
    let a = a / scale;
    let b = b / scale;
    let c = c / scale;
    let d = d / scale;
    let determinant = a * d - b * c;
    determinant.is_finite() && determinant != 0.0
}

const fn runtime_resource_role(role: GraphLoweringResourceRole) -> RuntimeResourceRole {
    match role {
        GraphLoweringResourceRole::RootWorkingImage => RuntimeResourceRole::RootWorkingImage,
        GraphLoweringResourceRole::CaptureWorkingImage => RuntimeResourceRole::CaptureWorkingImage,
        GraphLoweringResourceRole::ClipCoverage => RuntimeResourceRole::ClipCoverage,
        GraphLoweringResourceRole::IsolationWorkingImage => {
            RuntimeResourceRole::IsolationWorkingImage
        }
        GraphLoweringResourceRole::ImportedImage => RuntimeResourceRole::ImportedImage,
        GraphLoweringResourceRole::BackdropCopy => RuntimeResourceRole::BackdropCopy,
        GraphLoweringResourceRole::FilterIntermediate => RuntimeResourceRole::FilterIntermediate,
        GraphLoweringResourceRole::ShadowImage => RuntimeResourceRole::ShadowImage,
        GraphLoweringResourceRole::CompositeResult => RuntimeResourceRole::CompositeResult,
    }
}

const fn runtime_resource_format(
    role: RuntimeResourceRole,
    working_format: WorkingFormat,
) -> RuntimeResourceFormat {
    match role {
        RuntimeResourceRole::CaptureWorkingImage => RuntimeResourceFormat::VelloCaptureRgba8Unorm,
        RuntimeResourceRole::ClipCoverage => RuntimeResourceFormat::ClipCoverageRgba8Unorm,
        RuntimeResourceRole::ImportedImage => RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
        RuntimeResourceRole::RootWorkingImage
        | RuntimeResourceRole::IsolationWorkingImage
        | RuntimeResourceRole::BackdropCopy
        | RuntimeResourceRole::FilterIntermediate
        | RuntimeResourceRole::ShadowImage
        | RuntimeResourceRole::CompositeResult => RuntimeResourceFormat::Working(working_format),
    }
}

fn runtime_pass_kind(
    kind: GraphLoweringPassKind,
    working_format: WorkingFormat,
) -> Result<RuntimePassKind> {
    Ok(match kind {
        GraphLoweringPassKind::ClearRoot {
            initialization,
            color,
        } => RuntimePassKind::ClearRoot {
            initialization: match initialization {
                GraphLoweringInitialization::SurfaceBaseColor => {
                    RuntimeInitialization::SurfaceBaseColor
                }
                GraphLoweringInitialization::Transparent => RuntimeInitialization::Transparent,
            },
            color,
        },
        GraphLoweringPassKind::VelloCapture(work) => {
            RuntimePassKind::VelloCapture(work.map(runtime_vello_capture))
        }
        GraphLoweringPassKind::CanonicalizeCapture => RuntimePassKind::CanonicalizeCapture,
        GraphLoweringPassKind::CopyBackdrop => RuntimePassKind::CopyBackdrop,
        GraphLoweringPassKind::ColorFilter(filter) => {
            RuntimePassKind::ColorFilter(filter.map(runtime_color_filter).transpose()?)
        }
        GraphLoweringPassKind::BlurHorizontal(blur) => RuntimePassKind::BlurHorizontal(
            blur.map(|blur| runtime_blur(blur, RuntimeBlurAxis::Horizontal, working_format)),
        ),
        GraphLoweringPassKind::BlurVertical(blur) => RuntimePassKind::BlurVertical(
            blur.map(|blur| runtime_blur(blur, RuntimeBlurAxis::Vertical, working_format)),
        ),
        GraphLoweringPassKind::DropShadowColorize(shadow) => {
            RuntimePassKind::DropShadowColorize(shadow.map(runtime_drop_shadow))
        }
        GraphLoweringPassKind::Composite(composite) => {
            RuntimePassKind::Composite(composite.map(runtime_composite).transpose()?)
        }
        GraphLoweringPassKind::Present => RuntimePassKind::Present,
    })
}

fn runtime_vello_capture(capture: GraphLoweringVelloCapture) -> RuntimeVelloCapture {
    match capture {
        GraphLoweringVelloCapture::Span(span) => {
            RuntimeVelloCapture::Span(runtime_vello_span(span))
        }
        GraphLoweringVelloCapture::ClipCoverage(coverage) => {
            RuntimeVelloCapture::ClipCoverage(runtime_clip_coverage(coverage))
        }
    }
}

fn runtime_vello_span(span: GraphLoweringVelloSpan) -> RuntimeVelloSpan {
    RuntimeVelloSpan {
        scope: match span.scope() {
            GraphLoweringVelloSpanScope::CurrentParent => RuntimeVelloSpanScope::CurrentParent,
            GraphLoweringVelloSpanScope::LayerSource => RuntimeVelloSpanScope::LayerSource,
        },
        commands: span.commands().clone(),
        capture_transform: span.capture_transform(),
        parent_to_surface: span.parent_to_surface(),
        antialiasing: span.antialiasing(),
        captured_before_outer_semantics: span.captured_before_outer_semantics(),
    }
}

fn runtime_clip_coverage(coverage: GraphLoweringClipCoverage) -> RuntimeClipCoverage {
    RuntimeClipCoverage {
        elements: coverage
            .elements()
            .iter()
            .map(|element| RuntimeClipCoverageElement {
                clip: element.clip().clone(),
                transform: element.transform(),
            })
            .collect(),
        antialiasing: coverage.antialiasing(),
    }
}

fn runtime_filter_spatial(
    spatial: super::frame::GraphLoweringFilterSpatialMapping,
) -> RuntimeFilterSpatialMapping {
    RuntimeFilterSpatialMapping {
        source: RuntimeSpatialDescriptor::from_graph(spatial.source()),
        result: RuntimeSpatialDescriptor::from_graph(spatial.result()),
    }
}

fn runtime_edge(edge: GraphLoweringEdgePolicy) -> RuntimeSamplingEdge {
    match edge {
        GraphLoweringEdgePolicy::NoSampling => RuntimeSamplingEdge::ClampToExtent,
        GraphLoweringEdgePolicy::TransparentBlack => RuntimeSamplingEdge::TransparentBlack,
        GraphLoweringEdgePolicy::SemanticBorderMirror(bounds) => {
            RuntimeSamplingEdge::SemanticBorderMirror(bounds)
        }
    }
}

fn runtime_color_filter(filter: GraphLoweringColorFilter) -> Result<RuntimeColorFilter> {
    Ok(RuntimeColorFilter {
        operations: filter
            .operations()
            .iter()
            .copied()
            .map(runtime_color_operation)
            .collect::<Result<Vec<_>>>()?,
        spatial: runtime_filter_spatial(filter.spatial()),
        edge: runtime_edge(filter.edge()),
    })
}

fn runtime_color_operation(
    operation: super::frame::GraphLoweringColorOperation,
) -> Result<RuntimeColorOperation> {
    let runtime_operation = match operation.operation() {
        ColorFilterOp::Brightness(amount) => {
            RuntimeColorOperationKind::Brightness(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Contrast(amount) => {
            RuntimeColorOperationKind::Contrast(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Grayscale(amount) => RuntimeColorOperationKind::Grayscale(
            RuntimeUnitFilterAmount::try_from_algorithm(amount)?,
        ),
        ColorFilterOp::HueRotate(angle) => {
            RuntimeColorOperationKind::HueRotate(RuntimeFilterAngle::try_from_algorithm(angle)?)
        }
        ColorFilterOp::Invert(amount) => {
            RuntimeColorOperationKind::Invert(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Opacity(amount) => {
            RuntimeColorOperationKind::Opacity(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Saturate(amount) => {
            RuntimeColorOperationKind::Saturate(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Sepia(amount) => {
            RuntimeColorOperationKind::Sepia(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
    };
    let clamp_boundary = match operation.clamp_boundary() {
        ColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply => {
            RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
        }
    };
    Ok(RuntimeColorOperation {
        operation: runtime_operation,
        clamp_boundary,
    })
}

fn runtime_blur(
    blur: GraphLoweringBlur,
    axis: RuntimeBlurAxis,
    _working_format: WorkingFormat,
) -> RuntimeBlur {
    let spatial = runtime_filter_spatial(blur.spatial());
    let kernel = GaussianKernelKey::from_exact_plan(
        blur.standard_deviation().to_bits(),
        spatial.result.raster_scale.to_bits(),
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS.to_bits(),
        blur.support_radius(),
        GaussianKernelSamplingForm::PairedLinear,
    );
    RuntimeBlur {
        axis,
        input: match blur.input() {
            GraphLoweringBlurInput::Rgba => RuntimeBlurInput::Rgba,
            GraphLoweringBlurInput::SourceAlpha => RuntimeBlurInput::SourceAlpha,
        },
        standard_deviation: blur.standard_deviation(),
        support_radius: blur.support_radius(),
        kernel,
        spatial,
        edge: runtime_edge(blur.edge()),
    }
}

fn runtime_drop_shadow(shadow: GraphLoweringDropShadow) -> RuntimeDropShadow {
    RuntimeDropShadow {
        offset: shadow.offset(),
        standard_deviation: shadow.standard_deviation(),
        color: shadow.color(),
        support_radius: shadow.support_radius(),
        spatial: runtime_filter_spatial(shadow.spatial()),
        edge: runtime_edge(shadow.edge()),
        uses_source_alpha: shadow.uses_source_alpha(),
        uses_continuous_offset: shadow.uses_continuous_offset(),
        retains_unchanged_source: shadow.retains_unchanged_source(),
    }
}

fn runtime_composite(composite: GraphLoweringComposite) -> Result<RuntimeComposite> {
    let kind = match composite.kind() {
        GraphLoweringCompositeKind::SpanSourceOver => RuntimeCompositeKind::SpanSourceOver,
        GraphLoweringCompositeKind::Layer {
            transform,
            destination_to_layer_local,
            opacity,
            blend,
            clip,
            outer_clips,
            clip_coverage,
            alpha_mask,
        } => {
            let alpha_mask = alpha_mask
                .as_deref()
                .map(|mask| {
                    RuntimeResolvedAlphaMaskComposition::try_new(
                        RuntimeResourceId(mask.resource()),
                        mask.bounds(),
                        mask.image_dimensions(),
                        ShaderMaskSamplingKey::new(mask.quality(), mask.extend()),
                    )
                })
                .transpose()?;
            RuntimeCompositeKind::Layer {
                transform: *transform,
                parameters: Box::new(RuntimeLayerCompositeParameters::try_new(
                    destination_to_layer_local.affine(),
                    *opacity,
                    *blend,
                    clip.is_some() || !outer_clips.is_empty(),
                    alpha_mask,
                )?),
                clip: clip.clone(),
                outer_clips: outer_clips
                    .iter()
                    .map(|clip| RuntimeOuterClip {
                        clip: clip.clip().clone(),
                        transform: clip.transform(),
                    })
                    .collect(),
                clip_coverage: clip_coverage.map(RuntimeResourceId),
            }
        }
        GraphLoweringCompositeKind::DropShadow => RuntimeCompositeKind::DropShadow,
    };
    Ok(RuntimeComposite {
        kind,
        source_captured_before_outer_semantics: composite.source_captured_before_outer_semantics(),
    })
}

fn lower_read_bindings(
    reads: &[GraphLoweringReadBinding],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
) -> Result<Vec<RuntimeReadBinding>> {
    let mut seen = BTreeSet::new();
    reads
        .iter()
        .copied()
        .map(|read| {
            let resource = RuntimeResourceId(read.resource());
            if !seen.insert(resource) {
                return Err(lowering_error("duplicate runtime read binding"));
            }
            let resource_request = resource_by_id
                .get(&resource)
                .ok_or_else(|| lowering_error("runtime read resource is missing"))?;
            let source_format = resource_formats
                .get(&resource)
                .copied()
                .ok_or_else(|| lowering_error("runtime read format is missing"))?;
            let role = runtime_read_role(read.role());
            let sampling_edge = runtime_sampling_edge(read.sampling_edge());
            let resolved_mask_sampling = match read.sampling_filter() {
                GraphLoweringSamplingFilter::ImportedMask => match &resource_request.import {
                    Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) => Some(
                        ShaderMaskSamplingKey::new(upload.quality(), upload.extend()),
                    ),
                    None => {
                        return Err(lowering_error(
                            "mask sampler is not bound to an imported resolved mask",
                        ));
                    }
                },
                GraphLoweringSamplingFilter::Nearest
                | GraphLoweringSamplingFilter::Linear
                | GraphLoweringSamplingFilter::GaussianKernel => None,
            };
            let sampling_filter = match (read.sampling_filter(), resolved_mask_sampling) {
                (GraphLoweringSamplingFilter::Nearest, _) => RuntimeSamplingFilter::Nearest,
                (GraphLoweringSamplingFilter::ImportedMask, Some(sampling))
                    if sampling.quality() == ShaderMaskQualityKey::Low =>
                {
                    RuntimeSamplingFilter::Nearest
                }
                (GraphLoweringSamplingFilter::Linear, _)
                | (GraphLoweringSamplingFilter::GaussianKernel, _)
                | (GraphLoweringSamplingFilter::ImportedMask, Some(_)) => {
                    RuntimeSamplingFilter::Linear
                }
                (GraphLoweringSamplingFilter::ImportedMask, None) => {
                    return Err(lowering_error(
                        "mask sampling policy is missing from an imported mask read",
                    ));
                }
            };
            let sampler_key = SamplerKey::new(
                shader_binding_role(role),
                source_format.shader_key(),
                match sampling_filter {
                    RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                    RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
                },
                shader_sampling_edge(sampling_edge),
                resolved_mask_sampling,
            );
            Ok(RuntimeReadBinding {
                role,
                resource,
                sampling_filter,
                sampling_edge,
                sampler_key,
            })
        })
        .collect()
}

const fn runtime_read_role(role: GraphLoweringReadRole) -> RuntimeReadRole {
    match role {
        GraphLoweringReadRole::CaptureSource => RuntimeReadRole::CaptureSource,
        GraphLoweringReadRole::CompletedParent => RuntimeReadRole::CompletedParent,
        GraphLoweringReadRole::FilterSource => RuntimeReadRole::FilterSource,
        GraphLoweringReadRole::BlurredSourceAlpha => RuntimeReadRole::BlurredSourceAlpha,
        GraphLoweringReadRole::CompositeParent => RuntimeReadRole::CompositeParent,
        GraphLoweringReadRole::CompositeSource => RuntimeReadRole::CompositeSource,
        GraphLoweringReadRole::ClipCoverage => RuntimeReadRole::ClipCoverage,
        GraphLoweringReadRole::AlphaMask => RuntimeReadRole::AlphaMask,
        GraphLoweringReadRole::Shadow => RuntimeReadRole::Shadow,
        GraphLoweringReadRole::FinalWorkingImage => RuntimeReadRole::FinalWorkingImage,
    }
}

const fn runtime_sampling_edge(edge: GraphLoweringSamplingEdge) -> RuntimeSamplingEdge {
    match edge {
        GraphLoweringSamplingEdge::ClampToExtent => RuntimeSamplingEdge::ClampToExtent,
        GraphLoweringSamplingEdge::TransparentBlack => RuntimeSamplingEdge::TransparentBlack,
        GraphLoweringSamplingEdge::SemanticBorderMirror(bounds) => {
            RuntimeSamplingEdge::SemanticBorderMirror(bounds)
        }
    }
}

const fn shader_binding_role(role: RuntimeReadRole) -> ShaderBindingRoleKey {
    match role {
        RuntimeReadRole::CaptureSource => ShaderBindingRoleKey::CaptureSource,
        RuntimeReadRole::CompletedParent => ShaderBindingRoleKey::CompletedParent,
        RuntimeReadRole::FilterSource => ShaderBindingRoleKey::FilterSource,
        RuntimeReadRole::BlurredSourceAlpha => ShaderBindingRoleKey::BlurredSourceAlpha,
        RuntimeReadRole::CompositeParent => ShaderBindingRoleKey::CompositeParent,
        RuntimeReadRole::CompositeSource => ShaderBindingRoleKey::CompositeSource,
        RuntimeReadRole::ClipCoverage => ShaderBindingRoleKey::ClipCoverage,
        RuntimeReadRole::AlphaMask => ShaderBindingRoleKey::AlphaMask,
        RuntimeReadRole::Shadow => ShaderBindingRoleKey::Shadow,
        RuntimeReadRole::FinalWorkingImage => ShaderBindingRoleKey::FinalWorkingImage,
    }
}

const fn shader_sampling_edge(edge: RuntimeSamplingEdge) -> ShaderSamplingEdgeKey {
    match edge {
        RuntimeSamplingEdge::ClampToExtent => ShaderSamplingEdgeKey::ClampToExtent,
        RuntimeSamplingEdge::TransparentBlack => ShaderSamplingEdgeKey::TransparentBlack,
        RuntimeSamplingEdge::SemanticBorderMirror(_) => ShaderSamplingEdgeKey::SemanticBorderMirror,
    }
}

const fn runtime_read_uses_shader_sampler(kind: &RuntimePassKind, role: RuntimeReadRole) -> bool {
    match kind {
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { .. },
            ..
        })) => matches!(role, RuntimeReadRole::CompositeSource),
        _ => true,
    }
}

const fn runtime_read_uses_shader_texture(kind: &RuntimePassKind, role: RuntimeReadRole) -> bool {
    !matches!(
        (kind, role),
        (
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::DropShadow,
                ..
            })),
            RuntimeReadRole::Shadow
        )
    )
}

fn runtime_pass_cache_keys(
    kind: &RuntimePassKind,
    reads: &[RuntimeReadBinding],
    result: RuntimeResultBinding,
    working_format: WorkingFormat,
    output_format: Format,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
) -> Result<Option<RuntimePassCacheKeys>> {
    if matches!(
        kind,
        RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
    ) || result == RuntimeResultBinding::Empty
    {
        return Ok(None);
    }
    let program = shader_program(kind)?;
    let composite_samples_parent = matches!(
        kind,
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) if parameters.blend() != BlendMode::Normal
    );
    let sampled_reads = reads
        .iter()
        .filter(|read| {
            runtime_read_uses_shader_texture(kind, read.role)
                && (read.role != RuntimeReadRole::CompositeParent
                    || (!matches!(
                        kind,
                        RuntimePassKind::Composite(Some(RuntimeComposite {
                            kind: RuntimeCompositeKind::SpanSourceOver,
                            ..
                        }))
                    ) && composite_samples_parent))
        })
        .collect::<Vec<_>>();
    let sampled_textures = sampled_reads
        .iter()
        .map(|read| {
            let format = resource_formats
                .get(&read.resource)
                .copied()
                .ok_or_else(|| lowering_error("cache key source format is missing"))?;
            Ok((shader_binding_role(read.role), format.shader_key()))
        })
        .collect::<Result<Vec<_>>>()?;
    let samplers = sampled_reads
        .iter()
        .filter(|read| runtime_read_uses_shader_sampler(kind, read.role))
        .map(|read| read.sampler_key)
        .collect::<Vec<_>>();
    let layout = BindGroupLayoutKey::new(program, &sampled_textures, shader_data_bindings(kind));
    let output_key = matches!(kind, RuntimePassKind::Present)
        .then(|| ShaderTextureFormatKey::output(output_format));
    let shader = ShaderModuleKey::new(
        program,
        layout.clone(),
        samplers.clone(),
        Some(ShaderTextureFormatKey::working(working_format)),
        output_key,
    );
    let target_format = match result {
        RuntimeResultBinding::Resource(resource) => resource_formats
            .get(&resource)
            .copied()
            .map(RuntimeResourceFormat::shader_key)
            .ok_or_else(|| lowering_error("cache key target format is missing"))?,
        RuntimeResultBinding::Output(format) => ShaderTextureFormatKey::output(format),
        RuntimeResultBinding::Empty => {
            return Err(lowering_error("executable pass has no target binding"));
        }
    };
    let pipeline = RenderPipelineKey::new(
        shader.clone(),
        layout.clone(),
        samplers.clone(),
        target_format,
    );
    Ok(Some(RuntimePassCacheKeys {
        samplers,
        layout,
        shader,
        pipeline,
    }))
}

fn shader_program(kind: &RuntimePassKind) -> Result<ShaderProgramKey> {
    match kind {
        RuntimePassKind::CanonicalizeCapture => Ok(ShaderProgramKey::CanonicalizeCapture),
        RuntimePassKind::CopyBackdrop => Ok(ShaderProgramKey::CopyBackdrop),
        RuntimePassKind::ColorFilter(Some(_)) => Ok(ShaderProgramKey::ColorFilter),
        RuntimePassKind::BlurHorizontal(Some(blur)) => Ok(ShaderProgramKey::BlurHorizontal {
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
            edge: shader_sampling_edge(blur.edge),
        }),
        RuntimePassKind::BlurVertical(Some(blur)) => Ok(ShaderProgramKey::BlurVertical {
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
            edge: shader_sampling_edge(blur.edge),
        }),
        RuntimePassKind::DropShadowColorize(Some(_)) => Ok(ShaderProgramKey::DropShadowColorize),
        RuntimePassKind::Composite(Some(composite)) => Ok(ShaderProgramKey::Composite(
            shader_composite_key(&composite.kind),
        )),
        RuntimePassKind::Present => Ok(ShaderProgramKey::Present),
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Err(lowering_error(
            "a specialized or empty pass requested a custom shader key",
        )),
    }
}

fn shader_composite_key(kind: &RuntimeCompositeKind) -> ShaderCompositeKey {
    match kind {
        RuntimeCompositeKind::SpanSourceOver => ShaderCompositeKey::SpanSourceOver,
        RuntimeCompositeKind::Layer {
            parameters,
            clip_coverage,
            ..
        } => ShaderCompositeKey::Layer {
            path: if parameters.blend() == BlendMode::Normal {
                ShaderCompositePathKey::Normal
            } else {
                ShaderCompositePathKey::DestinationSampling
            },
            has_clip_coverage: clip_coverage.is_some(),
            has_alpha_mask: parameters.alpha_mask().is_some(),
        },
        RuntimeCompositeKind::DropShadow => ShaderCompositeKey::DropShadow,
    }
}

fn shader_data_bindings(kind: &RuntimePassKind) -> Vec<ShaderDataBindingKey> {
    match kind {
        RuntimePassKind::CanonicalizeCapture | RuntimePassKind::CopyBackdrop => {
            vec![ShaderDataBindingKey::SpatialUniform]
        }
        RuntimePassKind::ColorFilter(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::ColorFilterOperations,
        ],
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            let mut bindings = vec![
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::GaussianKernel,
            ];
            if matches!(blur.edge, RuntimeSamplingEdge::SemanticBorderMirror(_)) {
                bindings.push(ShaderDataBindingKey::BlurEdgeParameters);
            }
            bindings
        }
        RuntimePassKind::DropShadowColorize(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::DropShadowParameters,
        ],
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::DropShadow,
            ..
        })) => vec![ShaderDataBindingKey::SpatialUniform],
        RuntimePassKind::Composite(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::CompositeParameters,
        ],
        RuntimePassKind::Present => vec![ShaderDataBindingKey::SpatialUniform],
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Vec::new(),
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RuntimeLoweringObservationForTest {
    pub(crate) has_exact_closed_vocabulary: bool,
    pub(crate) preserves_backend_ready_resource_facts: bool,
    pub(crate) preserves_semantic_pass_facts: bool,
    pub(crate) preserves_topological_bindings: bool,
    pub(crate) preserves_exact_last_use_releases: bool,
    pub(crate) rejects_inconsistent_bindings_atomically: bool,
    pub(crate) has_exact_cache_keys: bool,
    pub(crate) keys_separate_program_layout_sampling_and_edge: bool,
    pub(crate) keys_separate_source_working_and_output_formats: bool,
}

#[cfg(test)]
pub(crate) fn runtime_lowering_observation_for_test(
    commands: RenderCommands,
    surface_size: super::Size,
    surface_scale: f64,
    antialiasing: Antialiasing,
    base_color: Color,
    output_format: Format,
    capabilities: DeviceCapabilities,
) -> Result<RuntimeLoweringObservationForTest> {
    let context = FrameContext::try_new(surface_size, surface_scale, antialiasing, base_color)?;
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the lowering fixture did not produce a GPU graph",
        ));
    };
    let plan = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        output_format,
        &capabilities,
    )?;
    let reduced = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::ReducedPrecision,
        output_format,
        &capabilities,
    )?;
    let alternate_output = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Bgra8,
        &capabilities,
    )?;
    let graph_view = graph.lowering_view()?;

    let has_exact_closed_vocabulary = runtime_vocabulary_is_exact(&plan);
    let preserves_backend_ready_resource_facts =
        runtime_resource_facts_are_exact(&graph, graph_view, &plan, output_format);
    let preserves_semantic_pass_facts = runtime_semantic_pass_facts_are_exact(&plan);
    let preserves_topological_bindings = runtime_topology_is_exact(graph_view, &plan);
    let preserves_exact_last_use_releases = runtime_releases_are_exact(graph_view, &plan);
    let rejects_inconsistent_bindings_atomically =
        lowering_faults_are_rejected(&graph, &capabilities);
    let (has_exact_cache_keys, keys_separate_program_layout_sampling_and_edge) =
        runtime_cache_key_facts(&plan);
    let keys_separate_source_working_and_output_formats =
        runtime_format_keys_are_separate(graph_view, &plan, &reduced, &alternate_output);

    Ok(RuntimeLoweringObservationForTest {
        has_exact_closed_vocabulary,
        preserves_backend_ready_resource_facts,
        preserves_semantic_pass_facts,
        preserves_topological_bindings,
        preserves_exact_last_use_releases,
        rejects_inconsistent_bindings_atomically,
        has_exact_cache_keys,
        keys_separate_program_layout_sampling_and_edge,
        keys_separate_source_working_and_output_formats,
    })
}

#[cfg(test)]
fn runtime_vocabulary_is_exact(plan: &LoweredGraphPlan) -> bool {
    let mut vocabulary = [false; 10];
    for pass in &plan.passes {
        vocabulary[runtime_pass_kind_index(&pass.kind)] = true;
    }
    vocabulary.into_iter().all(|present| present)
}

#[cfg(test)]
fn runtime_resource_facts_are_exact(
    graph: &GpuRenderGraph,
    graph_view: GraphLoweringView<'_>,
    plan: &LoweredGraphPlan,
    output_format: Format,
) -> bool {
    let imported_keys = plan
        .resources
        .iter()
        .filter_map(|resource| {
            resource
                .import
                .as_ref()
                .map(|RuntimeResourceImport::ResolvedAlphaMask(upload)| upload.cache_key())
        })
        .collect::<Vec<_>>();
    let graph_imported_keys = graph_view
        .resources()
        .into_iter()
        .filter_map(|resource| {
            resource
                .import()
                .map(|GraphLoweringImportView::ResolvedAlphaMask(upload)| upload.cache_key())
        })
        .collect::<Vec<_>>();
    let has_distinct_formats = plan
        .resources
        .iter()
        .any(|resource| resource.format == RuntimeResourceFormat::VelloCaptureRgba8Unorm)
        && plan.resources.iter().any(|resource| {
            resource.format == RuntimeResourceFormat::Working(WorkingFormat::HighPrecision)
        })
        && plan
            .resources
            .iter()
            .any(|resource| resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm);
    let over_limit = DeviceCapabilities::from_test_facts(true, true, 1);
    let extent_rejected = LoweredGraphPlan::try_lower_validated_graph(
        graph,
        WorkingFormat::HighPrecision,
        output_format,
        &over_limit,
    )
    .is_err();
    plan.working_format == WorkingFormat::HighPrecision
        && plan.output_format == output_format
        && plan.resources.iter().all(|resource| {
            resource.spatial.device_extent.width() > 0
                && resource.spatial.device_extent.height() > 0
                && resource.expected_reads > 0
        })
        && has_distinct_formats
        && imported_keys == graph_imported_keys
        && !imported_keys.is_empty()
        && extent_rejected
}

#[cfg(test)]
fn runtime_semantic_pass_facts_are_exact(plan: &LoweredGraphPlan) -> bool {
    plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span)))
                if !span.commands.commands.is_empty()
                    && span.captured_before_outer_semantics
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::ColorFilter(Some(filter)) if !filter.operations.is_empty()
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::DropShadowColorize(Some(shadow))
                if shadow.uses_source_alpha
                    && shadow.uses_continuous_offset
                    && shadow.retains_unchanged_source
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) if parameters.alpha_mask().is_some()
        )
    })
}

#[cfg(test)]
fn runtime_topology_is_exact(graph_view: GraphLoweringView<'_>, plan: &LoweredGraphPlan) -> bool {
    let graph_passes = graph_view.passes();
    graph_passes.len() == plan.passes.len()
        && graph_passes
            .iter()
            .zip(&plan.passes)
            .all(|(graph_pass, runtime_pass)| {
                RuntimePassId(graph_pass.id()) == runtime_pass.id
                    && graph_pass
                        .dependencies()
                        .into_iter()
                        .map(RuntimePassId)
                        .collect::<Vec<_>>()
                        == runtime_pass.dependencies
                    && graph_pass.reads().is_ok_and(|reads| {
                        reads
                            .into_iter()
                            .map(|read| RuntimeResourceId(read.resource()))
                            .eq(runtime_pass.reads.iter().map(|read| read.resource))
                    })
            })
}

#[cfg(test)]
fn runtime_releases_are_exact(graph_view: GraphLoweringView<'_>, plan: &LoweredGraphPlan) -> bool {
    let expected = graph_view
        .resources()
        .into_iter()
        .map(|resource| {
            (
                RuntimeResourceId(resource.id()),
                RuntimePassId(resource.last_use()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let observed = plan
        .passes
        .iter()
        .flat_map(|pass| {
            pass.releases
                .iter()
                .copied()
                .map(move |resource| (resource, pass.id))
        })
        .collect::<BTreeMap<_, _>>();
    expected == observed
        && plan
            .resources
            .iter()
            .any(|resource| resource.expected_reads > 1)
}

#[cfg(test)]
fn runtime_cache_key_facts(plan: &LoweredGraphPlan) -> (bool, bool) {
    let custom = plan
        .passes
        .iter()
        .filter(|pass| {
            !matches!(
                pass.kind,
                RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
            )
        })
        .collect::<Vec<_>>();
    let exact = !custom.is_empty()
        && custom.iter().all(|pass| pass.cache_keys.is_some())
        && plan.passes.iter().all(|pass| {
            matches!(
                pass.kind,
                RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
            ) == pass.cache_keys.is_none()
        })
        && plan.passes.iter().all(|pass| {
            pass.reads
                .iter()
                .all(|read| runtime_read_key_is_exact(plan, read))
        });
    let unique_layouts = custom
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| &keys.layout))
        .fold(Vec::new(), |mut unique, key| {
            if !unique.contains(&key) {
                unique.push(key);
            }
            unique
        });
    let unique_shaders = custom
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| &keys.shader))
        .fold(Vec::new(), |mut unique, key| {
            if !unique.contains(&key) {
                unique.push(key);
            }
            unique
        });
    let edges = custom.iter().flat_map(|pass| pass.reads.iter());
    let transparent = edges
        .clone()
        .any(|read| matches!(read.sampling_edge, RuntimeSamplingEdge::TransparentBlack));
    let mirror = edges.clone().any(|read| {
        matches!(
            read.sampling_edge,
            RuntimeSamplingEdge::SemanticBorderMirror(_)
        )
    });
    (
        exact,
        unique_layouts.len() > 3 && unique_shaders.len() > 5 && transparent && mirror,
    )
}

#[cfg(test)]
fn runtime_read_key_is_exact(plan: &LoweredGraphPlan, read: &RuntimeReadBinding) -> bool {
    let Some(resource) = plan
        .resources
        .iter()
        .find(|resource| resource.id == read.resource)
    else {
        return false;
    };
    let expected_mask = match &resource.import {
        Some(RuntimeResourceImport::ResolvedAlphaMask(upload))
            if read.role == RuntimeReadRole::AlphaMask =>
        {
            Some(ShaderMaskSamplingKey::new(
                upload.quality(),
                upload.extend(),
            ))
        }
        Some(RuntimeResourceImport::ResolvedAlphaMask(_)) | None => None,
    };
    read.sampler_key.facts_for_test()
        == (
            shader_binding_role(read.role),
            resource.format.shader_key(),
            match read.sampling_filter {
                RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
            },
            shader_sampling_edge(read.sampling_edge),
            expected_mask,
        )
}

#[cfg(test)]
fn runtime_format_keys_are_separate(
    graph_view: GraphLoweringView<'_>,
    plan: &LoweredGraphPlan,
    reduced: &LoweredGraphPlan,
    alternate_output: &LoweredGraphPlan,
) -> bool {
    let main_keys = custom_key_map(plan);
    let reduced_keys = custom_key_map(reduced);
    let alternate_keys = custom_key_map(alternate_output);
    let working_changes = main_keys
        .iter()
        .all(|(id, keys)| reduced_keys.get(id).is_some_and(|other| *other != *keys));
    let output_changes_only_present = plan.passes.iter().all(|pass| {
        let Some(main) = pass.cache_keys.as_ref() else {
            return true;
        };
        alternate_keys
            .get(&pass.id)
            .is_some_and(|other| matches!(pass.kind, RuntimePassKind::Present) == (*other != main))
    });
    working_changes
        && output_changes_only_present
        && plan.root_working_image == RuntimeResourceId(graph_view.root_working_image())
        && plan.final_present == RuntimePassId(graph_view.final_present())
}

#[cfg(test)]
fn runtime_pass_kind_index(kind: &RuntimePassKind) -> usize {
    match kind {
        RuntimePassKind::ClearRoot { .. } => 0,
        RuntimePassKind::VelloCapture(_) => 1,
        RuntimePassKind::CanonicalizeCapture => 2,
        RuntimePassKind::CopyBackdrop => 3,
        RuntimePassKind::ColorFilter(_) => 4,
        RuntimePassKind::BlurHorizontal(_) => 5,
        RuntimePassKind::BlurVertical(_) => 6,
        RuntimePassKind::DropShadowColorize(_) => 7,
        RuntimePassKind::Composite(_) => 8,
        RuntimePassKind::Present => 9,
    }
}

#[cfg(test)]
fn custom_key_map(plan: &LoweredGraphPlan) -> BTreeMap<RuntimePassId, &RuntimePassCacheKeys> {
    plan.passes
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| (pass.id, keys)))
        .collect()
}

#[cfg(test)]
fn lowering_faults_are_rejected(graph: &GpuRenderGraph, capabilities: &DeviceCapabilities) -> bool {
    use super::frame::GraphLoweringFaultForTest;

    [
        GraphLoweringFaultForTest::MissingResourceBinding,
        GraphLoweringFaultForTest::DuplicateReadBinding,
        GraphLoweringFaultForTest::ForwardDependency,
        GraphLoweringFaultForTest::StaleResourceGeneration,
        GraphLoweringFaultForTest::InconsistentLastUse,
    ]
    .into_iter()
    .all(|fault| {
        let invalid = graph.with_lowering_fault_for_test(fault);
        LoweredGraphPlan::try_lower_validated_graph(
            &invalid,
            WorkingFormat::HighPrecision,
            Format::Rgba8,
            capabilities,
        )
        .is_err()
    })
}
