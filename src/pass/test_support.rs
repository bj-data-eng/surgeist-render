use super::ExecutableGraphWorkingFormatRequest;
use super::close::{
    C08ExecutionFacts, C08PreparableGraph, C10PreparableGraph, C11PreparableGraph,
    C12PreparableGraph, ClosedExecutableGraph, ClosedExecutableGraphFacts,
    ExecutableColorFilterFacts, ExecutableLayerCompositionFacts, ExecutableVelloCaptureFacts,
    PrePreparationGraphClassification, c08_resource_has_fixed_facts,
    executable_vello_capture_facts,
};
use super::close::{ExecutableFilterStepFacts, preparation_error};
use super::encode::{
    C08CustomSpineEncodingSummary, C08PendingGraphEncoding, PendingC08PreparedFrameCommit,
    backdrop_filter_passes, vello_capture_raster_parameters,
};
use super::lower::{
    lowering_error, runtime_pass_cache_keys, shader_binding_role, shader_sampling_edge,
};
use super::model::{
    LoweredGraphPlan, RuntimeBlur, RuntimeBlurAxis, RuntimeBlurInput, RuntimeColorClampBoundary,
    RuntimeColorOperation, RuntimeColorOperationKind, RuntimeComposite, RuntimeCompositeKind,
    RuntimeInitialization, RuntimeLayerCompositeParameters, RuntimePass, RuntimePassCacheKeys,
    RuntimePassId, RuntimePassKind, RuntimeReadBinding, RuntimeReadRole,
    RuntimeResolvedAlphaMaskComposition, RuntimeResourceFormat, RuntimeResourceId,
    RuntimeResourceImport, RuntimeResourceProducer, RuntimeResourceRequest, RuntimeResourceRole,
    RuntimeResultBinding, RuntimeSamplingEdge, RuntimeSamplingFilter, RuntimeSpatialDescriptor,
    RuntimeVelloCapture, RuntimeVelloSpanScope,
};
use super::prepare::{
    ExecutableGraphDispatchEligibility, GraphPreparationSource, PreparedGraph, PreparedPassView,
    RuntimeAllocationRequest, RuntimeGraphPreparationPlan, VELLO_CAPTURE_TEXTURE_USAGES,
};

use super::super::{
    Result, backend::DeviceCapabilities, renderer::EffectQualityPolicy, resource::WorkingFormat,
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
};

use super::super::{
    BackendErrorCode, Color, Error, Format, PhysicalSize, Transform,
    encode::encode_vello_clip_coverage_scene,
    filter::RuntimeFilterAmount,
    frame::{GpuRenderGraph, GraphLoweringImportView},
    layer::BlendMode,
    renderer::Antialiasing,
    resource::{GaussianKernelKey, ResourceIdentity, ResourceManager},
    shader::{
        BlurEdgeParameterBytes, ColorFilterOperationBufferLimits, ColorFilterOperationBytes,
        CompositeParameterBytes, DevicePassCache, SamplerKey, ShaderBindingRoleKey,
        ShaderCompositePathKey, ShaderDataBindingKey, ShaderMaskSamplingKey,
        ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    vello_engine::PendingVelloResourceCommit,
};

#[cfg(test)]
use super::super::texture::EffectTextureRole;

#[cfg(test)]
use super::super::{Point, Rect, command::RenderClip, command::RenderCommands};

#[cfg(test)]
use super::super::resource::ResourceAccountingFault;

#[cfg(test)]
use super::super::frame::GraphLoweringView;
#[cfg(test)]
use super::super::frame::{FrameContext, FramePlan};

#[cfg(test)]
use super::super::vello_engine::scene::VelloPathDrawObservationForTest;

/// Test-only deterministic failure at the checked color-filter shader boundary.
#[cfg(test)]
pub(crate) struct ScopedColorFilterShaderFailureForTest {
    previous: bool,
}

thread_local! {
    static COLOR_FILTER_SHADER_FAILURE_FOR_TEST: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
impl ScopedColorFilterShaderFailureForTest {
    pub(crate) fn after_checked_realization() -> Self {
        let previous = COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ScopedColorFilterShaderFailureForTest {
    fn drop(&mut self) {
        COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(|active| active.set(self.previous));
    }
}

pub(crate) fn normalize_color_filter_shader_failure_for_test(mut error: Error) -> Error {
    let active = COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(Cell::get);
    if active && error.message() == "the C10 operation buffer binding is missing" {
        error.replace_message("injected color-filter shader failure after checked realization");
    }
    error
}

pub(crate) fn normalize_scope_resolution_failure_for_test(mut error: Error) -> Error {
    error.replace_message("checked internal Vello resource or command encoding failed");
    error
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

pub(crate) use super::encode::ScheduledFilterRawFact as C11FilterPassTagForTest;

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
    color_filters: Vec<super::super::FilterList>,
    blur_filters: Vec<super::super::FilterList>,
    shadow_filters: Vec<super::super::FilterList>,
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
    filters: Vec<super::super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> ColorFilterGraphObservationForTest {
    color_filter_graph_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn mixed_color_unsupported_diagnostic_observation_for_test(
    color_filters: Vec<super::super::FilterList>,
    mixed_filters: Vec<super::super::FilterList>,
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
    filters: Vec<super::super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11ExecutableGraphObservationForTest {
    c11_executable_graph_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn c11_filter_graph_observation_for_test(
    filters: Vec<super::super::FilterList>,
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
    let resources = ResourceManager::new(super::super::ResourceCacheBudget::DISABLED);
    let cache = DevicePassCache::new();
    let resources_before = resources.observation_for_test();
    let cache_before = cache.counts_for_test();
    let exact_error = |result: Result<u64>, field: &'static str| {
        result.is_err_and(|error| {
            error.code() == super::super::ErrorCode::InvalidInput
                && error
                    .invalid_value_diagnostic()
                    .is_some_and(|invalid| invalid.field() == field)
        })
    };
    let exact_byte_len = 16 + 32;
    let count_overflow_is_exact = exact_error(
        super::super::shader::color_filter_operation_byte_len_for_test(
            u64::from(u32::MAX) + 1,
            ColorFilterOperationBufferLimits::for_test(u64::MAX, u64::MAX),
        ),
        "color filter operation count",
    );
    let max_buffer_size_is_exact = exact_error(
        super::super::shader::color_filter_operation_byte_len_for_test(
            1,
            ColorFilterOperationBufferLimits::for_test(exact_byte_len - 1, exact_byte_len),
        ),
        "color filter operation buffer byte length",
    );
    let max_storage_binding_size_is_exact = exact_error(
        super::super::shader::color_filter_operation_byte_len_for_test(
            1,
            ColorFilterOperationBufferLimits::for_test(exact_byte_len, exact_byte_len - 1),
        ),
        "color filter operation buffer byte length",
    );
    let equality_at_both_limits_is_accepted =
        super::super::shader::color_filter_operation_byte_len_for_test(
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
    filters: Vec<super::super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11BlurLayoutObservationForTest {
    c11_blur_layout_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) async fn c11_blur_cache_realization_observation_for_test(
    device: &wgpu::Device,
    filters: Vec<super::super::FilterList>,
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
    ordinary_filters: Vec<super::super::FilterList>,
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
    filters: Vec<super::super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C11DropShadowLayoutObservationForTest {
    c11_drop_shadow_layout_observation(filters, commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) async fn c11_drop_shadow_cache_realization_observation_for_test(
    device: &wgpu::Device,
    filters: Vec<super::super::FilterList>,
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
            super::super::shader::c08_pass_key_facts_for_test(
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
            facts.program == super::super::shader::C08ProgramForTest::DropShadowMerge
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
            super::super::shader::c09_composite_pass_key_facts_for_test(
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
            super::super::shader::c12_copy_backdrop_pass_key_facts_for_test(
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
    facts: &[super::super::shader::C12CopyBackdropPassKeyFactsForTest],
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
    filters: Vec<super::super::FilterList>,
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
            let Some(observed) = super::super::shader::c12_backdrop_blur_pass_key_facts_for_test(
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
            super::super::shader::c12_blur_shader_mirrors_semantic_bounds_before_texture_mapping_for_test(),
    })
}

#[cfg(test)]
fn c11_blur_layout_observation(
    filters: Vec<super::super::FilterList>,
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
            let Some(observed) = super::super::shader::c11_blur_pass_key_facts_for_test(
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
    filters: Vec<super::super::FilterList>,
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
    filters: Vec<super::super::FilterList>,
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
        let Some(observed) = super::super::shader::c11_drop_shadow_colorize_key_facts_for_test(
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
            let Some(observed) = super::super::shader::c10_color_filter_pass_key_facts_for_test(
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
    use super::super::shader::c09_composite_pass_key_facts_for_test;

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
    let graph = super::super::frame::forced_c08_graph_for_test(commands, context)?;
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
    use super::super::shader::{C08ProgramForTest, c08_pass_key_facts_for_test};

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
    let graph = super::super::frame::forced_c08_graph_for_test(commands, context)?;
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
    let graph = super::super::frame::forced_c08_graph_for_test(commands, context)?;
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
    super::super::shader::PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
        source,
        destination,
    )
    .map(super::super::shader::PassSpatialUniformBytes::into_bytes_for_test)
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

#[cfg(test)]
impl ClosedExecutableGraph {
    fn into_lowered(self) -> LoweredGraphPlan {
        self.lowered
    }
}

#[cfg(test)]
impl C08PreparableGraph {
    pub(crate) fn try_from_lowered(
        lowered: LoweredGraphPlan,
    ) -> std::result::Result<Self, LoweredGraphPlan> {
        let closed = ClosedExecutableGraph::try_from_lowered(lowered)?;
        Self::try_from_closed(closed).map_err(|closed| (*closed).into_lowered())
    }

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

#[cfg(test)]
impl C10PreparableGraph {
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
            .ok_or_else(|| preparation_error("the C10 root output resource is missing"))
    }

    pub(crate) fn first_color_spatial_for_test(&self) -> Option<C10ColorSpatialObservationForTest> {
        self.closed
            .facts
            .color_filters
            .first()
            .map(|filter| c10_spatial_observation(filter.filter.spatial.source))
    }

    fn color_filters(&self) -> &[ExecutableColorFilterFacts] {
        &self.closed.facts.color_filters
    }
}

#[cfg(test)]
impl C11PreparableGraph {
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
            .ok_or_else(|| preparation_error("the C11 root output resource is missing"))
    }

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
}

#[cfg(test)]
impl C12PreparableGraph {
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
}

#[cfg(test)]
impl ExecutableVelloCaptureFacts {
    fn commands(&self) -> Option<&RenderCommands> {
        self.span().map(|span| &span.commands)
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C08CaptureGridForTest {
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
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

impl LoweredGraphPlan {
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
    let c08_graph = super::super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
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
    let c08_graph = super::super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
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
    filters: Vec<super::super::FilterList>,
    commands: RenderCommands,
    context: FrameContext,
    working_format: WorkingFormat,
    output_format: Format,
    capabilities: &DeviceCapabilities,
) -> Option<(GpuRenderGraph, LoweredGraphPlan)> {
    let graph =
        super::super::frame::authored_filter_graph_for_test(filters, commands, context).ok()?;
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
    color_filters: Vec<super::super::FilterList>,
    blur_filters: Vec<super::super::FilterList>,
    shadow_filters: Vec<super::super::FilterList>,
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
    filters: Vec<super::super::FilterList>,
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
    color_filters: Vec<super::super::FilterList>,
    mixed_filters: Vec<super::super::FilterList>,
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
    let color_diagnostic = super::super::renderer::unsupported_graph_diagnostic_for_test(
        &color_graph,
        Format::Rgba8,
        &capabilities,
    )
    .ok()??;
    let mixed_diagnostic = super::super::renderer::unsupported_graph_diagnostic_for_test(
        &mixed_graph,
        Format::Rgba8,
        &capabilities,
    )
    .ok()??;
    Some(MixedColorUnsupportedDiagnosticObservationForTest {
        pure_color_retains_gpu_color_diagnostic: color_diagnostic
            == super::super::UnsupportedPrimitive::new(
                super::super::PrimitiveFamily::Filters,
                super::super::PrimitiveOperation::GpuColorFilterExecution,
            ),
        color_then_blur_reports_gpu_blur_diagnostic: mixed_diagnostic
            == super::super::UnsupportedPrimitive::new(
                super::super::PrimitiveFamily::Filters,
                super::super::PrimitiveOperation::GpuBlurFilterExecution,
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
    filters: Vec<super::super::FilterList>,
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
    let resources = ResourceManager::new(super::super::ResourceCacheBudget::DISABLED);
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
    filters: &[super::super::FilterList],
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
    filters: Vec<super::super::FilterList>,
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
    let resources = ResourceManager::new(super::super::ResourceCacheBudget::DISABLED);
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
        super::super::shader::c12_backdrop_blur_pass_key_facts_for_test(
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
            let raster = prepared.observation_for_test();
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
    observed_mask_ids: &mut Vec<super::super::ImageId>,
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
    commands: &[super::super::command::RenderCommand],
) -> Vec<super::super::ImageId> {
    fn collect(
        commands: &[super::super::command::RenderCommand],
        image_ids: &mut Vec<super::super::ImageId>,
    ) {
        for command in commands {
            let super::super::command::RenderCommand::Layer { layer, children } = command else {
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
            super::super::Size::new(64.0, 64.0),
            raster_scale,
            antialiasing,
            Color::TRANSPARENT,
        )
        .ok()?;
        let graph =
            super::super::frame::forced_c08_graph_for_test(commands.clone(), context).ok()?;
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

        let encoded = super::super::encode::encode_vello_scene_with_initial_transform(
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

pub(crate) use super::encode::EncodedCaptureRawFact as C08EncodedCaptureObservationForTest;

#[cfg(test)]
impl PendingC08PreparedFrameCommit {
    pub(crate) fn resource_identities_for_test(&self) -> Vec<ResourceIdentity> {
        self.frame_scope.leased_resource_identities_for_test()
    }

    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        self.frame_scope.poison_retained_byte_accounting_for_test()
    }
}

#[cfg(test)]
impl<'device> PreparedGraph<'device> {
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
}

#[cfg(test)]
impl<'device> PreparedGraph<'device> {
    pub(crate) fn fail_capture_encoding_for_test(&mut self) {
        self.fail_capture_encoding_after_for_test(0);
    }

    pub(crate) fn fail_capture_encoding_after_for_test(&mut self, successful_capture_count: usize) {
        let target = self
            .plan
            .passes
            .iter()
            .filter_map(|request| match request.runtime.kind {
                RuntimePassKind::VelloCapture(Some(_)) => match request.runtime.result {
                    RuntimeResultBinding::Resource(target) => Some(target),
                    RuntimeResultBinding::Empty | RuntimeResultBinding::Output(_) => None,
                },
                _ => None,
            })
            .nth(successful_capture_count)
            .expect("the capture-failure fixture requires the selected prepared capture");
        let removed = self.resource_bindings.remove(&target);
        assert!(
            removed.is_some(),
            "the capture-failure fixture requires a live prepared target binding"
        );
    }

    pub(crate) fn apply_color_filter_shader_failure_for_test(&mut self) {
        if !COLOR_FILTER_SHADER_FAILURE_FOR_TEST.with(Cell::get) {
            return;
        }
        let pass = self
            .plan
            .passes
            .iter()
            .find_map(|request| {
                matches!(request.runtime.kind, RuntimePassKind::ColorFilter(Some(_)))
                    .then_some(request.runtime.id)
            })
            .expect("the color-filter shader-failure fixture requires a prepared color pass");
        let removed = self.color_filter_operation_bindings.remove(&pass);
        assert!(
            removed.is_some(),
            "the color-filter shader-failure fixture requires a realized operation binding"
        );
    }

    pub(crate) fn fail_scope_resolution_for_test(&mut self) {
        let layout = self
            .plan
            .passes
            .iter()
            .find_map(|request| {
                matches!(request.runtime.kind, RuntimePassKind::Present)
                    .then(|| {
                        request
                            .cache_keys
                            .as_ref()
                            .map(|keys| keys.layout().clone())
                    })
                    .flatten()
            })
            .expect("the scope-failure fixture requires prepared present-pass cache keys");
        self.pass_cache_update
            .as_mut()
            .expect("the scope-failure fixture requires a realized provisional cache update")
            .replace_layout_with_empty_scope_failure_fixture_for_test(self.device, &layout);
    }

    pub(crate) fn acquired_capture_lease_count_for_test(&self) -> usize {
        self.acquired_capture_lease_count_raw_fact
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

#[cfg(test)]
impl PreparedPassView<'_> {
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
    surface_size: super::super::Size,
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
    use super::super::frame::GraphLoweringFaultForTest;

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
