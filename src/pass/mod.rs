mod close;
mod encode;
mod lower;
mod model;
mod parameters;
mod prepare;
#[cfg(test)]
mod test_support;

use encode::CustomSpineEncodingState;

#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "the pass front door preserves its crate-visible closure contract"
    )
)]
pub(crate) use close::{
    BackdropPreparableGraph, BaseExecutionFacts, BasePreparableGraph, ColorFilterPreparableGraph,
    CompositionPreparableGraph, ExecutableVelloCaptureFacts, SpatialFilterPreparableGraph,
};
#[expect(
    unused_imports,
    reason = "the pass front door preserves its crate-visible encoding contract"
)]
pub(crate) use encode::{
    AccountingReadyPreparedFrameCommit, CustomSpineEncodingSummary, EncodedGpuGraphActivity,
    GraphExternalOutputView, PendingGraphEncoding, PendingPreparedFrameCommit,
    PreparedGraphSubmission,
};
#[expect(
    unused_imports,
    reason = "the pass front door preserves its crate-visible capture contract"
)]
pub(crate) use encode::{VelloCaptureCompletionReceipt, VelloCaptureEncodingHandoff};
#[expect(
    unused_imports,
    reason = "the pass front door preserves its crate-visible color-filter contract"
)]
pub(crate) use model::RuntimeColorFilter;
#[expect(
    unused_imports,
    reason = "the pass front door preserves its crate-visible runtime model contract"
)]
pub(crate) use model::{
    LoweredGraphPlan, RuntimeBlur, RuntimeBlurAxis, RuntimeBlurInput, RuntimeClipCoverage,
    RuntimeClipCoverageElement, RuntimeColorClampBoundary, RuntimeColorOperation,
    RuntimeColorOperationKind, RuntimeComposite, RuntimeCompositeKind, RuntimeDropShadow,
    RuntimeFilterSpatialMapping, RuntimeGraphGeneration, RuntimeInitialization,
    RuntimeLayerCompositeParameters, RuntimeOuterClip, RuntimePass, RuntimePassCacheKeys,
    RuntimePassId, RuntimePassKind, RuntimeReadBinding, RuntimeReadRole,
    RuntimeResolvedAlphaMaskComposition, RuntimeResourceFormat, RuntimeResourceId,
    RuntimeResourceImport, RuntimeResourceProducer, RuntimeResourceRequest, RuntimeResourceRole,
    RuntimeResultBinding, RuntimeSamplingEdge, RuntimeSamplingFilter, RuntimeSpatialDescriptor,
    RuntimeVelloCapture, RuntimeVelloSpan, RuntimeVelloSpanScope,
};
#[expect(
    unused_imports,
    reason = "the pass front door preserves its crate-visible texture-binding contract"
)]
pub(crate) use prepare::PreparedTextureBinding;
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "the pass front door preserves its crate-visible prepared-binding contract"
    )
)]
pub(crate) use prepare::{
    ExecutableGraphDispatchEligibility, PreparedGaussianKernelBinding, PreparedGraph,
    PreparedPassView, VELLO_CAPTURE_TEXTURE_USAGES,
};

#[cfg(test)]
#[expect(
    unused_imports,
    reason = "the test-only pass front door preserves every existing crate-visible test contract"
)]
pub(crate) use test_support::{
    BackdropBlurCacheRealizationObservationForTest, BackdropBlurLayoutObservationForTest,
    BackdropExecutableGraphObservationForTest, BackdropFilterChainObservationForTest,
    BackdropGraphObservationForTest, BaseGraphExecutableSubsetObservationForTest,
    BlurCacheRealizationObservationForTest, BlurLayoutObservationForTest,
    BoundedCaptureTransformObservationForTest, ClipCoverageCaptureObservationForTest,
    ClipCoverageElementObservationForTest, ColorFilterCacheRealizationObservationForTest,
    ColorFilterCacheRequestsForTest, ColorFilterExecutableGraphObservationForTest,
    ColorFilterGraphObservationForTest, ColorFilterLayoutObservationForTest,
    ColorFilterOperationBufferLimitObservationForTest, ColorFilterOperationBytesObservationForTest,
    ColorFilterSpatialObservationForTest, CompositionExecutableGraphObservationForTest,
    CompositionGraphObservationForTest, CompositionOuterOperationObservationForTest,
    CompositionReadObservationForTest, CopyBackdropCacheRealizationObservationForTest,
    CopyBackdropLayoutObservationForTest, CorePassCacheRequestsForTest,
    CorePassLayoutObservationForTest, DropShadowCacheRealizationObservationForTest,
    DropShadowLayoutObservationForTest, EncodedVelloCaptureObservationForTest,
    GraphClipCoverageObservationForTest, LayerCompositeCacheRequestsForTest,
    LayerCompositeLayoutObservationForTest, LayerCompositionObservationForTest,
    MaskUploadAllocationObservationForTest, MixedColorUnsupportedDiagnosticObservationForTest,
    PreparedAllocationIdentitiesForTest, PreparedGraphExerciseObservationForTest,
    RuntimeColorFilterObservationForTest, RuntimeColorOperationObservationForTest,
    RuntimeColorOperationTagForTest, RuntimeColorScalarObservationForTest,
    RuntimeFilterAmountObservationForTest, RuntimeLoweringObservationForTest,
    ScopedColorFilterShaderFailureForTest, SpatialFilterExecutableGraphObservationForTest,
    SpatialFilterGraphObservationForTest, SpatialFilterPassTagForTest, VelloCaptureGridForTest,
    backdrop_blur_cache_realization_observation_for_test,
    backdrop_blur_layout_observation_for_test, backdrop_executable_graph_observation_for_test,
    backdrop_filter_chain_observation_for_test, backdrop_graph_observation_for_test,
    backdrop_preparable_graph_from_graph_for_test,
    base_graph_executable_subset_observation_for_test, blur_cache_realization_observation_for_test,
    blur_layout_observation_for_test, bounded_capture_transform_observation_for_test,
    color_filter_cache_realization_observation_for_test,
    color_filter_executable_graph_observation_for_test, color_filter_graph_observation_for_test,
    color_filter_layout_observation_for_test,
    color_filter_operation_buffer_limit_observation_for_test,
    color_filter_operation_bytes_observation_for_test, color_filter_preparable_graph_for_test,
    composite_parameter_bytes_for_test, composition_executable_graph_observation_for_test,
    composition_graph_observation_for_test, copy_backdrop_cache_realization_observation_for_test,
    copy_backdrop_layout_observation_for_test, core_pass_cache_requests_for_test,
    core_pass_layout_observation_for_test, drop_shadow_cache_realization_observation_for_test,
    drop_shadow_layout_observation_for_test, graph_clip_coverage_observation_for_test,
    layer_composite_cache_requests_for_test, layer_composite_layout_observation_for_test,
    mask_pipeline_keys_exclude_image_identity_for_test,
    mask_upload_allocation_observation_for_test,
    mixed_color_unsupported_diagnostic_observation_for_test,
    normalize_color_filter_shader_failure_for_test, normalize_scope_resolution_failure_for_test,
    pass_spatial_uniform_bytes_for_test, runtime_color_filter_observation_for_test,
    runtime_lowering_observation_for_test, spatial_filter_executable_graph_observation_for_test,
    spatial_filter_graph_observation_for_test, spatial_filter_preparable_graph_from_graph_for_test,
    two_capture_spine_lowered_for_test, zero_capture_spine_lowered_for_test,
};

use super::{
    Result, backend::DeviceCapabilities, renderer::EffectQualityPolicy, resource::WorkingFormat,
};

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
