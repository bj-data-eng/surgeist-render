use std::collections::HashMap;

use crate::Result;

use super::{
    cache::{
        DevicePassCache, ProvisionalCompositePassObjects, ProvisionalCorePassObjects,
        ProvisionalDevicePassCacheUpdate,
    },
    key::{
        BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderBindingRoleKey,
        ShaderCompositeKey, ShaderCompositePathKey, ShaderDataBindingKey, ShaderModuleKey,
        ShaderProgramKey, ShaderSamplingEdgeKey, ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    pipeline::{BLUR_WGSL, span_source_over_blend},
    validate::{
        BlurAxis, BlurInput, BlurPassKeyRefs, ColorFilterPassKeyRefs, CompositePassKeyRefs,
        CopyBackdropPassKeyRefs, CorePassKeyRefs, CorePassProgram, DropShadowColorizePassKeyRefs,
        is_blur_program, validate_blur_pass_keys, validate_color_filter_pass_keys,
        validate_composite_pass_keys, validate_copy_backdrop_pass_keys, validate_core_pass_keys,
        validate_drop_shadow_colorize_pass_keys,
    },
};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CorePassProgramForTest {
    CanonicalizeCapture,
    SpanSourceOver,
    DropShadowMerge,
    Present,
}

impl ProvisionalDevicePassCacheUpdate {
    pub(crate) fn replace_layout_with_empty_scope_failure_fixture_for_test(
        &mut self,
        device: &wgpu::Device,
        layout: &BindGroupLayoutKey,
    ) {
        let malformed = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surgeist test-owned empty scope-failure layout"),
            entries: &[],
        });
        let previous = self.layouts.insert(layout.clone(), malformed);
        assert!(
            previous.is_some(),
            "the scope-failure fixture requires a realized bind-group layout"
        );
    }

    #[cfg(test)]
    pub(crate) fn realize_core_pass_with_invalid_fragment_for_test<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalCorePassObjects<'a>> {
        let keys = CorePassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        self.realize_core_pass_with_fragment_entry(
            device,
            cache,
            keys,
            "missing_core_pass_fragment_main",
        )
    }

    #[cfg(test)]
    pub(crate) fn realize_composite_pass_with_invalid_fragment_for_test<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalCompositePassObjects<'a>> {
        self.realize_composite_pass_with_fragment_entry(
            device,
            cache,
            CompositePassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
            Some("missing_layer_composite_fragment"),
        )
    }
    #[cfg(test)]
    pub(crate) fn contains_composite_pass_for_test(
        &self,
        cache: &DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        self.composite_pass_objects(
            cache,
            CompositePassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
        )
        .is_ok_and(|objects| objects.require_encoding_ready().is_ok())
    }
    #[cfg(test)]
    pub(crate) fn contains_core_pass_for_test(
        &self,
        cache: &DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        self.pass_objects(
            cache,
            CorePassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
        )
        .is_ok_and(|objects| objects.is_encoding_ready_for_test())
    }

    #[cfg(test)]
    pub(crate) fn is_empty_for_test(&self) -> bool {
        self.samplers.is_empty()
            && self.layouts.is_empty()
            && self.shaders.is_empty()
            && self.pipelines.is_empty()
    }
}

impl ProvisionalCorePassObjects<'_> {
    #[cfg(test)]
    fn is_encoding_ready_for_test(&self) -> bool {
        self.require_encoding_ready().is_ok()
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CorePassKeyFactsForTest {
    pub(crate) program: CorePassProgramForTest,
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) output_format: Option<ShaderTextureFormatKey>,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_spatial_uniform: bool,
    pub(crate) has_fixed_source_over_blend: bool,
}

#[cfg(test)]
pub(crate) fn core_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::CorePassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_core_pass_keys(CorePassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(CorePassKeyFactsForTest {
        program: match description.program {
            CorePassProgram::CanonicalizeCapture => CorePassProgramForTest::CanonicalizeCapture,
            CorePassProgram::SpanSourceOver => CorePassProgramForTest::SpanSourceOver,
            CorePassProgram::DropShadowMerge => CorePassProgramForTest::DropShadowMerge,
            CorePassProgram::Present => CorePassProgramForTest::Present,
        },
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: shader.working_format?,
        output_format: shader.output_format,
        target_format: pipeline.target_format,
        has_only_spatial_uniform: layout.data_bindings.as_slice()
            == [ShaderDataBindingKey::SpatialUniform],
        has_fixed_source_over_blend: !matches!(
            description.program,
            CorePassProgram::SpanSourceOver | CorePassProgram::DropShadowMerge
        ) || span_source_over_blend()
            == wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
    })
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerCompositePassKeyFactsForTest {
    pub(crate) path: ShaderCompositePathKey,
    pub(crate) has_clip_coverage: bool,
    pub(crate) has_alpha_mask: bool,
    pub(crate) sampled_roles: Vec<ShaderBindingRoleKey>,
    pub(crate) has_only_source_sampler: bool,
    pub(crate) has_exact_uniforms: bool,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) uses_fixed_source_over_blend: bool,
    pub(crate) uses_replace_blend: bool,
}

#[cfg(test)]
pub(crate) fn layer_composite_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::LayerCompositePassKeyFactsForTest> {
    let description = validate_composite_pass_keys(CompositePassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(LayerCompositePassKeyFactsForTest {
        path: description.path,
        has_clip_coverage: description.has_clip_coverage,
        has_alpha_mask: description.has_alpha_mask,
        sampled_roles: layout
            .sampled_textures
            .iter()
            .map(|texture| texture.binding_role)
            .collect(),
        has_only_source_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::CompositeSource,
                filter: ShaderSamplingFilterKey::Linear,
                edge: ShaderSamplingEdgeKey::TransparentBlack,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_exact_uniforms: layout.data_bindings.as_slice()
            == [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::CompositeParameters,
            ],
        working_format: shader.working_format?,
        target_format: pipeline.target_format,
        uses_fixed_source_over_blend: description.path == ShaderCompositePathKey::Normal,
        uses_replace_blend: description.path == ShaderCompositePathKey::DestinationSampling,
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ColorFilterPassKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_nearest_source_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn color_filter_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::ColorFilterPassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_color_filter_pass_keys(ColorFilterPassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(ColorFilterPassKeyFactsForTest {
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: description.working_format,
        target_format: pipeline.target_format,
        has_only_nearest_source_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::FilterSource,
                filter: ShaderSamplingFilterKey::Nearest,
                edge: ShaderSamplingEdgeKey::ClampToExtent,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_exact_data_bindings: layout.data_bindings.as_slice()
            == [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::ColorFilterOperations,
            ],
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CopyBackdropPassKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_nearest_transparent_sampler: bool,
    pub(crate) has_only_spatial_uniform: bool,
}

#[cfg(test)]
pub(crate) fn copy_backdrop_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<CopyBackdropPassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_copy_backdrop_pass_keys(CopyBackdropPassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(CopyBackdropPassKeyFactsForTest {
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: description.working_format,
        target_format: pipeline.target_format,
        has_only_nearest_transparent_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::CompletedParent,
                filter: ShaderSamplingFilterKey::Nearest,
                edge: ShaderSamplingEdgeKey::TransparentBlack,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_only_spatial_uniform: layout.data_bindings.as_slice()
            == [ShaderDataBindingKey::SpatialUniform],
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlurPassKeyFactsForTest {
    pub(crate) horizontal: bool,
    pub(crate) source_alpha: bool,
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_linear_source_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn blur_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::BlurPassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_blur_pass_keys(BlurPassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(BlurPassKeyFactsForTest {
        horizontal: description.axis == BlurAxis::Horizontal,
        source_alpha: description.input == BlurInput::SourceAlpha,
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: description.working_format,
        target_format: pipeline.target_format,
        has_only_linear_source_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::FilterSource,
                filter: ShaderSamplingFilterKey::Linear,
                edge: ShaderSamplingEdgeKey::TransparentBlack,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_exact_data_bindings: layout.data_bindings.as_slice()
            == [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::GaussianKernel,
            ],
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackdropBlurPassKeyFactsForTest {
    pub(crate) horizontal: bool,
    pub(crate) source_alpha: bool,
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_linear_mirror_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn backdrop_blur_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::BackdropBlurPassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_blur_pass_keys(BlurPassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(BackdropBlurPassKeyFactsForTest {
        horizontal: description.axis == BlurAxis::Horizontal,
        source_alpha: description.input == BlurInput::SourceAlpha,
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: description.working_format,
        target_format: pipeline.target_format,
        has_only_linear_mirror_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::FilterSource,
                filter: ShaderSamplingFilterKey::Linear,
                edge: ShaderSamplingEdgeKey::SemanticBorderMirror,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_exact_data_bindings: layout.data_bindings.as_slice()
            == [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::GaussianKernel,
                ShaderDataBindingKey::BlurEdgeParameters,
            ],
    })
}

#[cfg(test)]
pub(crate) fn backdrop_blur_shader_mirrors_semantic_bounds_before_texture_mapping_for_test() -> bool
{
    BLUR_WGSL.contains("let logical_sample = destination_point(destination_position)")
        && BLUR_WGSL.contains("+ axis * offset / spatial.source_origin_scale.z;")
        && BLUR_WGSL.contains("let mirrored_sample = mirror_logical_point(logical_sample);")
        && BLUR_WGSL.contains(
            "return (mirrored_sample - spatial.source_origin_scale.xy)\n        * spatial.source_origin_scale.z;",
        )
        && BLUR_WGSL.contains("let bounds = blur_edge.semantic_minimum_maximum;")
        && BLUR_WGSL.contains("fn fragment_horizontal_rgba_mirror(")
        && BLUR_WGSL.contains("fn fragment_vertical_rgba_mirror(")
        && BLUR_WGSL.contains("fn fragment_horizontal_source_alpha_mirror(")
        && BLUR_WGSL.contains("fn fragment_vertical_source_alpha_mirror(")
        && BLUR_WGSL.contains("sample_transparent_black(center + axis * sample.offset)")
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DropShadowColorizeKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_linear_transparent_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn drop_shadow_colorize_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<super::DropShadowColorizeKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_drop_shadow_colorize_pass_keys(DropShadowColorizePassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(DropShadowColorizeKeyFactsForTest {
        source_role: sampled_texture.binding_role,
        source_format: sampled_texture.source_format,
        working_format: description.working_format,
        target_format: pipeline.target_format,
        has_only_linear_transparent_sampler: matches!(
            samplers,
            [SamplerKey {
                binding_role: ShaderBindingRoleKey::BlurredSourceAlpha,
                filter: ShaderSamplingFilterKey::Linear,
                edge: ShaderSamplingEdgeKey::TransparentBlack,
                resolved_mask_sampling: None,
                ..
            }]
        ),
        has_exact_data_bindings: layout.data_bindings.as_slice()
            == [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::DropShadowParameters,
            ],
    })
}

#[cfg(test)]
pub(crate) fn device_pass_cache_owns_exact_key_spaces_for_test() -> bool {
    fn sampler_space(space: &HashMap<SamplerKey, wgpu::Sampler>) -> bool {
        space.is_empty()
    }
    fn layout_space(space: &HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>) -> bool {
        space.is_empty()
    }
    fn shader_space(space: &HashMap<ShaderModuleKey, wgpu::ShaderModule>) -> bool {
        space.is_empty()
    }
    fn pipeline_space(space: &HashMap<RenderPipelineKey, wgpu::RenderPipeline>) -> bool {
        space.is_empty()
    }

    let cache = DevicePassCache::new();
    sampler_space(&cache.samplers)
        && layout_space(&cache.layouts)
        && shader_space(&cache.shaders)
        && pipeline_space(&cache.pipelines)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DevicePassCacheCountsForTest {
    sampler_count: usize,
    layout_count: usize,
    shader_count: usize,
    pipeline_count: usize,
}

#[cfg(test)]
impl DevicePassCacheCountsForTest {
    #[must_use]
    pub(crate) const fn has_exactly_one_sampler(self) -> bool {
        self.sampler_count == 1
            && self.layout_count == 0
            && self.shader_count == 0
            && self.pipeline_count == 0
    }

    #[must_use]
    pub(crate) const fn is_empty(self) -> bool {
        self.sampler_count == 0
            && self.layout_count == 0
            && self.shader_count == 0
            && self.pipeline_count == 0
    }

    #[must_use]
    pub(crate) const fn has_render_pipelines(self) -> bool {
        self.pipeline_count > 0
    }
}

impl DevicePassCache {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn counts_for_test(&self) -> DevicePassCacheCountsForTest {
        DevicePassCacheCountsForTest {
            sampler_count: self.samplers.len(),
            layout_count: self.layouts.len(),
            shader_count: self.shaders.len(),
            pipeline_count: self.pipelines.len(),
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn seed_sampler_for_test(
        &mut self,
        device: &wgpu::Device,
    ) -> DevicePassCacheCountsForTest {
        let key = SamplerKey::new(
            ShaderBindingRoleKey::CaptureSource,
            ShaderTextureFormatKey::VelloCaptureRgba8Unorm,
            ShaderSamplingFilterKey::Nearest,
            ShaderSamplingEdgeKey::ClampToExtent,
            None,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surgeist-render pass-cache preservation test sampler"),
            ..Default::default()
        });
        assert!(
            self.samplers.insert(key, sampler).is_none(),
            "the fixed pass-cache preservation sampler key must be vacant"
        );
        self.counts_for_test()
    }

    #[cfg(test)]
    pub(crate) fn contains_core_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_composite_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        validate_composite_pass_keys(CompositePassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        })
        .is_ok()
            && samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_color_filter_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        validate_color_filter_pass_keys(ColorFilterPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        })
        .is_ok()
            && samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_only_two_color_filter_passes_for_test(&self) -> bool {
        self.samplers.len() == 2
            && self.layouts.len() == 2
            && self.shaders.len() == 2
            && self.pipelines.len() == 2
            && self.samplers.keys().all(|key| {
                key.binding_role == ShaderBindingRoleKey::FilterSource
                    && key.filter == ShaderSamplingFilterKey::Nearest
                    && key.edge == ShaderSamplingEdgeKey::ClampToExtent
                    && key.resolved_mask_sampling.is_none()
            })
            && self
                .layouts
                .keys()
                .all(|key| key.program == ShaderProgramKey::ColorFilter)
            && self
                .shaders
                .keys()
                .all(|key| key.program == ShaderProgramKey::ColorFilter)
            && self.pipelines.keys().all(|key| {
                key.shader.program == ShaderProgramKey::ColorFilter
                    && key.layout.program == ShaderProgramKey::ColorFilter
            })
    }

    #[cfg(test)]
    pub(crate) fn contains_copy_backdrop_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        validate_copy_backdrop_pass_keys(CopyBackdropPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        })
        .is_ok()
            && samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_only_two_copy_backdrop_passes_for_test(&self) -> bool {
        self.samplers.len() == 2
            && self.layouts.len() == 2
            && self.shaders.len() == 2
            && self.pipelines.len() == 2
            && self.samplers.keys().all(|key| {
                key.binding_role == ShaderBindingRoleKey::CompletedParent
                    && key.filter == ShaderSamplingFilterKey::Nearest
                    && key.edge == ShaderSamplingEdgeKey::TransparentBlack
                    && key.resolved_mask_sampling.is_none()
            })
            && self
                .layouts
                .keys()
                .all(|key| key.program == ShaderProgramKey::CopyBackdrop)
            && self
                .shaders
                .keys()
                .all(|key| key.program == ShaderProgramKey::CopyBackdrop)
            && self.pipelines.keys().all(|key| {
                key.shader.program == ShaderProgramKey::CopyBackdrop
                    && key.layout.program == ShaderProgramKey::CopyBackdrop
            })
    }

    #[cfg(test)]
    pub(crate) fn contains_blur_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        validate_blur_pass_keys(BlurPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        })
        .is_ok()
            && samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_only_eight_blur_passes_for_test(&self) -> bool {
        self.samplers.len() == 2
            && self.layouts.len() == 8
            && self.shaders.len() == 8
            && self.pipelines.len() == 8
            && self.samplers.keys().all(|key| {
                key.binding_role == ShaderBindingRoleKey::FilterSource
                    && key.filter == ShaderSamplingFilterKey::Linear
                    && key.edge == ShaderSamplingEdgeKey::TransparentBlack
                    && key.resolved_mask_sampling.is_none()
            })
            && self.layouts.keys().all(|key| is_blur_program(key.program))
            && self.shaders.keys().all(|key| is_blur_program(key.program))
            && self.pipelines.keys().all(|key| {
                is_blur_program(key.shader.program) && is_blur_program(key.layout.program)
            })
    }

    #[cfg(test)]
    pub(crate) fn contains_only_sixteen_edge_blur_passes_for_test(&self) -> bool {
        self.samplers.len() == 4
            && self.layouts.len() == 16
            && self.shaders.len() == 16
            && self.pipelines.len() == 16
            && self.samplers.keys().all(|key| {
                key.binding_role == ShaderBindingRoleKey::FilterSource
                    && key.filter == ShaderSamplingFilterKey::Linear
                    && matches!(
                        key.edge,
                        ShaderSamplingEdgeKey::TransparentBlack
                            | ShaderSamplingEdgeKey::SemanticBorderMirror
                    )
                    && key.resolved_mask_sampling.is_none()
            })
            && self.layouts.keys().all(|key| is_blur_program(key.program))
            && self.shaders.keys().all(|key| is_blur_program(key.program))
            && self.pipelines.keys().all(|key| {
                is_blur_program(key.shader.program) && is_blur_program(key.layout.program)
            })
    }

    #[cfg(test)]
    pub(crate) fn contains_drop_shadow_colorize_pass_for_test(
        &self,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        validate_drop_shadow_colorize_pass_keys(DropShadowColorizePassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        })
        .is_ok()
            && samplers.iter().all(|key| self.samplers.contains_key(key))
            && self.layouts.contains_key(layout)
            && self.shaders.contains_key(shader)
            && self.pipelines.contains_key(pipeline)
    }

    #[cfg(test)]
    pub(crate) fn contains_only_four_drop_shadow_passes_for_test(&self) -> bool {
        self.samplers.len() == 4
            && self.layouts.len() == 4
            && self.shaders.len() == 4
            && self.pipelines.len() == 4
            && self.samplers.keys().all(|key| {
                matches!(
                    key.binding_role,
                    ShaderBindingRoleKey::BlurredSourceAlpha
                        | ShaderBindingRoleKey::CompositeSource
                ) && key.filter == ShaderSamplingFilterKey::Linear
                    && key.edge == ShaderSamplingEdgeKey::TransparentBlack
                    && key.resolved_mask_sampling.is_none()
            })
            && self.layouts.keys().all(|key| {
                matches!(
                    key.program,
                    ShaderProgramKey::DropShadowColorize
                        | ShaderProgramKey::Composite(ShaderCompositeKey::DropShadow)
                )
            })
            && self.shaders.keys().all(|key| {
                matches!(
                    key.program,
                    ShaderProgramKey::DropShadowColorize
                        | ShaderProgramKey::Composite(ShaderCompositeKey::DropShadow)
                )
            })
            && self.pipelines.keys().all(|key| {
                matches!(
                    key.shader.program,
                    ShaderProgramKey::DropShadowColorize
                        | ShaderProgramKey::Composite(ShaderCompositeKey::DropShadow)
                ) && key.shader.program == key.layout.program
            })
    }
}
