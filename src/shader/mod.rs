mod cache;
mod key;
mod parameters;
mod pipeline;
#[cfg(test)]
mod test_support;
mod validate;

pub(crate) use cache::{
    DevicePassCache, ProvisionalColorFilterPassObjects, ProvisionalCompositePassObjects,
    ProvisionalCopyBackdropPassObjects, ProvisionalCorePassObjects,
    ProvisionalDevicePassCacheUpdate,
};
#[cfg(test)]
pub(crate) use cache::{ProvisionalBlurPassObjects, ProvisionalDropShadowColorizePassObjects};
pub(crate) use key::{
    BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderBindingRoleKey, ShaderCompositeKey,
    ShaderCompositePathKey, ShaderDataBindingKey, ShaderMaskExtendKey, ShaderMaskQualityKey,
    ShaderMaskSamplingKey, ShaderModuleKey, ShaderProgramKey, ShaderSamplingEdgeKey,
    ShaderSamplingFilterKey, ShaderTextureFormatKey,
};
pub(crate) use parameters::{
    BlurEdgeParameterBytes, ColorFilterOperationBufferLimits, ColorFilterOperationBytes,
    CompositeParameterBytes, DropShadowParameterBytes, PassSpatialUniformBytes,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use parameters::{
    CompositeParameterGpuVectorFactsForTest, composite_parameter_bytes_for_gpu_vector_for_test,
};
#[cfg(test)]
pub(crate) use parameters::{
    color_filter_operation_byte_len_for_test, drop_shadow_parameter_bytes_for_test,
};
#[cfg(test)]
pub(crate) use test_support::{
    BackdropBlurPassKeyFactsForTest, BlurPassKeyFactsForTest, ColorFilterPassKeyFactsForTest,
    CopyBackdropPassKeyFactsForTest, CorePassKeyFactsForTest, CorePassProgramForTest,
    DevicePassCacheCountsForTest, DropShadowColorizeKeyFactsForTest,
    LayerCompositePassKeyFactsForTest, backdrop_blur_pass_key_facts_for_test,
    backdrop_blur_shader_mirrors_semantic_bounds_before_texture_mapping_for_test,
    blur_pass_key_facts_for_test, color_filter_pass_key_facts_for_test,
    copy_backdrop_pass_key_facts_for_test, core_pass_key_facts_for_test,
    device_pass_cache_owns_exact_key_spaces_for_test, drop_shadow_colorize_key_facts_for_test,
    layer_composite_pass_key_facts_for_test,
};
