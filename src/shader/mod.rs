use std::{collections::HashMap, sync::Arc};

use super::Result;

mod key;
mod parameters;
mod pipeline;
mod validate;

#[cfg(test)]
use pipeline::{BLUR_WGSL, span_source_over_blend};
use pipeline::{
    create_blur_bind_group_layout, create_blur_render_pipeline, create_blur_shader_module,
    create_c08_bind_group_layout, create_c08_render_pipeline, create_c08_shader_module,
    create_color_filter_bind_group_layout, create_color_filter_render_pipeline,
    create_color_filter_shader_module, create_composite_bind_group_layout,
    create_composite_render_pipeline, create_composite_shader_module,
    create_copy_backdrop_bind_group_layout, create_copy_backdrop_pipeline,
    create_copy_backdrop_shader_module, create_drop_shadow_colorize_bind_group_layout,
    create_drop_shadow_colorize_render_pipeline, create_drop_shadow_colorize_shader_module,
    create_sampler,
};
#[cfg(test)]
use validate::{BlurAxis, BlurInput, is_blur_program};
use validate::{
    BlurPassDescription, BlurPassKeyRefs, C08PassKeyRefs, C08Program, ColorFilterPassDescription,
    ColorFilterPassKeyRefs, CompositePassDescription, CompositePassKeyRefs,
    CopyBackdropPassDescription, CopyBackdropPassKeyRefs, DropShadowColorizePassDescription,
    DropShadowColorizePassKeyRefs, blur_cache_error, c08_cache_error, color_filter_cache_error,
    copy_backdrop_cache_error, drop_shadow_cache_error, validate_blur_pass_keys,
    validate_c08_pass_keys, validate_color_filter_pass_keys, validate_composite_pass_keys,
    validate_copy_backdrop_pass_keys, validate_drop_shadow_colorize_pass_keys,
};

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

/// Non-clone handles created inside one checked GPU-operation scope. New entries
/// remain private to this phase until the caller explicitly commits after the
/// owning transaction resolves cleanly.
#[must_use = "provisional pass-cache handles must be committed only after checked success"]
pub(crate) struct ProvisionalDevicePassCacheUpdate {
    cache_identity: Arc<()>,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    layouts: HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>,
    shaders: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    pipelines: HashMap<RenderPipelineKey, wgpu::RenderPipeline>,
}

/// Borrowed C08 objects that are ready for later bind-group creation and pass
/// encoding even while newly created handles remain provisional.
pub(crate) struct ProvisionalC08PassObjects<'a> {
    program: C08Program,
    samplers: Vec<&'a wgpu::Sampler>,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Borrowed layer-compositor objects that remain provisional until the owning
/// checked transaction resolves successfully.
pub(crate) struct ProvisionalCompositePassObjects<'a> {
    description: CompositePassDescription,
    source_sampler: &'a wgpu::Sampler,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Borrowed C10 color-filter objects that remain provisional until the owning
/// checked GPU operation resolves successfully.
pub(crate) struct ProvisionalColorFilterPassObjects<'a> {
    description: ColorFilterPassDescription,
    source_sampler: &'a wgpu::Sampler,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Borrowed C12 backdrop-copy objects selected entirely by checked cache facts.
pub(crate) struct ProvisionalCopyBackdropPassObjects<'a> {
    description: CopyBackdropPassDescription,
    parent_sampler: &'a wgpu::Sampler,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Borrowed C11 blur objects selected entirely by checked cache-key facts.
pub(crate) struct ProvisionalBlurPassObjects<'a> {
    description: BlurPassDescription,
    source_sampler: &'a wgpu::Sampler,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Borrowed C11 drop-shadow colorize objects selected by checked key facts.
pub(crate) struct ProvisionalDropShadowColorizePassObjects<'a> {
    description: DropShadowColorizePassDescription,
    source_sampler: &'a wgpu::Sampler,
    layout: &'a wgpu::BindGroupLayout,
    shader: &'a wgpu::ShaderModule,
    pipeline: &'a wgpu::RenderPipeline,
}

/// Device-lifetime ownership for the four exact custom-pass WGPU handle spaces.
pub(crate) struct DevicePassCache {
    identity: Arc<()>,
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    layouts: HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>,
    shaders: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    pipelines: HashMap<RenderPipelineKey, wgpu::RenderPipeline>,
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
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            identity: Arc::new(()),
            samplers: HashMap::new(),
            layouts: HashMap::new(),
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.samplers.clear();
        self.layouts.clear();
        self.shaders.clear();
        self.pipelines.clear();
    }

    pub(crate) fn provisional_update(&self) -> ProvisionalDevicePassCacheUpdate {
        ProvisionalDevicePassCacheUpdate {
            cache_identity: Arc::clone(&self.identity),
            samplers: HashMap::new(),
            layouts: HashMap::new(),
            shaders: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }

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
    pub(crate) fn contains_c08_pass_for_test(
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

impl ProvisionalDevicePassCacheUpdate {
    pub(crate) fn realize_c08_pass<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalC08PassObjects<'a>> {
        let keys = C08PassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        self.realize_c08_pass_with_fragment_entry(device, cache, keys, "fragment_main")
    }

    #[cfg(test)]
    pub(crate) fn realize_c08_pass_with_invalid_fragment_for_test<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalC08PassObjects<'a>> {
        let keys = C08PassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        self.realize_c08_pass_with_fragment_entry(device, cache, keys, "missing_c08_fragment_main")
    }

    fn realize_c08_pass_with_fragment_entry<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        keys: C08PassKeyRefs<'_>,
        fragment_entry: &'static str,
    ) -> Result<ProvisionalC08PassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(c08_cache_error(
                "provisional C08 pass objects belong to another device cache",
            ));
        }
        let description = validate_c08_pass_keys(keys)?;

        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_c08_bind_group_layout(device, description),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                create_c08_shader_module(device, description),
            );
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let layout_handle = self
                .layouts
                .get(keys.layout)
                .or_else(|| cache.layouts.get(keys.layout))
                .ok_or_else(|| c08_cache_error("C08 bind-group layout realization was lost"))?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| c08_cache_error("C08 shader-module realization was lost"))?;
            let created = create_c08_render_pipeline(
                device,
                description,
                layout_handle,
                shader_handle,
                fragment_entry,
            )?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }

        self.pass_objects(cache, keys)
    }

    fn pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: C08PassKeyRefs<'_>,
    ) -> Result<ProvisionalC08PassObjects<'a>> {
        let samplers = keys
            .samplers
            .iter()
            .map(|key| {
                self.samplers
                    .get(key)
                    .or_else(|| cache.samplers.get(key))
                    .ok_or_else(|| c08_cache_error("C08 sampler realization was lost"))
            })
            .collect::<Result<Vec<_>>>()?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| c08_cache_error("C08 bind-group layout realization was lost"))?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| c08_cache_error("C08 shader-module realization was lost"))?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| c08_cache_error("C08 render-pipeline realization was lost"))?;
        Ok(ProvisionalC08PassObjects {
            program: validate_c08_pass_keys(keys)?.program,
            samplers,
            layout,
            shader,
            pipeline,
        })
    }

    pub(crate) fn encoding_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalC08PassObjects<'a>> {
        self.pass_objects(
            cache,
            C08PassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
        )
    }

    pub(crate) fn realize_copy_backdrop_pass<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalCopyBackdropPassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(copy_backdrop_cache_error(
                "provisional backdrop-copy objects belong to another device cache",
            ));
        }
        let keys = CopyBackdropPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        let description = validate_copy_backdrop_pass_keys(keys)?;
        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_copy_backdrop_bind_group_layout(device),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                create_copy_backdrop_shader_module(device),
            );
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let created = create_copy_backdrop_pipeline(
                device,
                description,
                self.layouts
                    .get(keys.layout)
                    .or_else(|| cache.layouts.get(keys.layout))
                    .ok_or_else(|| {
                        copy_backdrop_cache_error(
                            "backdrop-copy bind-group layout realization was lost",
                        )
                    })?,
                self.shaders
                    .get(keys.shader)
                    .or_else(|| cache.shaders.get(keys.shader))
                    .ok_or_else(|| {
                        copy_backdrop_cache_error(
                            "backdrop-copy shader-module realization was lost",
                        )
                    })?,
            )?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }
        self.copy_backdrop_pass_objects(cache, keys)
    }

    fn copy_backdrop_pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: CopyBackdropPassKeyRefs<'_>,
    ) -> Result<ProvisionalCopyBackdropPassObjects<'a>> {
        let [parent_sampler_key] = keys.samplers else {
            return Err(copy_backdrop_cache_error(
                "backdrop-copy realization requires one parent sampler",
            ));
        };
        let parent_sampler = self
            .samplers
            .get(parent_sampler_key)
            .or_else(|| cache.samplers.get(parent_sampler_key))
            .ok_or_else(|| {
                copy_backdrop_cache_error("backdrop-copy sampler realization was lost")
            })?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| {
                copy_backdrop_cache_error("backdrop-copy layout realization was lost")
            })?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| {
                copy_backdrop_cache_error("backdrop-copy shader realization was lost")
            })?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| {
                copy_backdrop_cache_error("backdrop-copy pipeline realization was lost")
            })?;
        Ok(ProvisionalCopyBackdropPassObjects {
            description: validate_copy_backdrop_pass_keys(keys)?,
            parent_sampler,
            layout,
            shader,
            pipeline,
        })
    }

    pub(crate) fn realize_color_filter_pass<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalColorFilterPassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(color_filter_cache_error(
                "provisional color-filter objects belong to another device cache",
            ));
        }
        let keys = ColorFilterPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        let description = validate_color_filter_pass_keys(keys)?;
        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_color_filter_bind_group_layout(device),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                create_color_filter_shader_module(device),
            );
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let layout_handle = self
                .layouts
                .get(keys.layout)
                .or_else(|| cache.layouts.get(keys.layout))
                .ok_or_else(|| {
                    color_filter_cache_error("color-filter bind-group layout realization was lost")
                })?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| {
                    color_filter_cache_error("color-filter shader-module realization was lost")
                })?;
            let created = create_color_filter_render_pipeline(
                device,
                description,
                layout_handle,
                shader_handle,
            )?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }
        self.color_filter_pass_objects(cache, keys)
    }

    pub(crate) fn color_filter_encoding_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalColorFilterPassObjects<'a>> {
        self.color_filter_pass_objects(
            cache,
            ColorFilterPassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
        )
    }

    pub(crate) fn realize_blur_pass<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalBlurPassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(blur_cache_error(
                "provisional blur objects belong to another device cache",
            ));
        }
        let keys = BlurPassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        let description = validate_blur_pass_keys(keys)?;
        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_blur_bind_group_layout(device, description),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders
                .insert(keys.shader.clone(), create_blur_shader_module(device));
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let layout_handle = self
                .layouts
                .get(keys.layout)
                .or_else(|| cache.layouts.get(keys.layout))
                .ok_or_else(|| blur_cache_error("blur bind-group layout realization was lost"))?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| blur_cache_error("blur shader-module realization was lost"))?;
            let created =
                create_blur_render_pipeline(device, description, layout_handle, shader_handle)?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }
        self.blur_pass_objects(cache, keys)
    }

    pub(crate) fn realize_drop_shadow_colorize_pass<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalDropShadowColorizePassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(drop_shadow_cache_error(
                "provisional drop-shadow objects belong to another device cache",
            ));
        }
        let keys = DropShadowColorizePassKeyRefs {
            samplers,
            layout,
            shader,
            pipeline,
        };
        let description = validate_drop_shadow_colorize_pass_keys(keys)?;
        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_drop_shadow_colorize_bind_group_layout(device),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                create_drop_shadow_colorize_shader_module(device),
            );
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let layout_handle = self
                .layouts
                .get(keys.layout)
                .or_else(|| cache.layouts.get(keys.layout))
                .ok_or_else(|| {
                    drop_shadow_cache_error(
                        "drop-shadow colorize bind-group layout realization was lost",
                    )
                })?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| {
                    drop_shadow_cache_error(
                        "drop-shadow colorize shader-module realization was lost",
                    )
                })?;
            let created = create_drop_shadow_colorize_render_pipeline(
                device,
                description,
                layout_handle,
                shader_handle,
            )?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }
        self.drop_shadow_colorize_pass_objects(cache, keys)
    }

    fn drop_shadow_colorize_pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: DropShadowColorizePassKeyRefs<'_>,
    ) -> Result<ProvisionalDropShadowColorizePassObjects<'a>> {
        let [source_sampler_key] = keys.samplers else {
            return Err(drop_shadow_cache_error(
                "drop-shadow colorize realization requires one source sampler",
            ));
        };
        let source_sampler = self
            .samplers
            .get(source_sampler_key)
            .or_else(|| cache.samplers.get(source_sampler_key))
            .ok_or_else(|| {
                drop_shadow_cache_error("drop-shadow colorize sampler realization was lost")
            })?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| {
                drop_shadow_cache_error("drop-shadow colorize layout realization was lost")
            })?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| {
                drop_shadow_cache_error("drop-shadow colorize shader realization was lost")
            })?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| {
                drop_shadow_cache_error("drop-shadow colorize pipeline realization was lost")
            })?;
        Ok(ProvisionalDropShadowColorizePassObjects {
            description: validate_drop_shadow_colorize_pass_keys(keys)?,
            source_sampler,
            layout,
            shader,
            pipeline,
        })
    }

    fn blur_pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: BlurPassKeyRefs<'_>,
    ) -> Result<ProvisionalBlurPassObjects<'a>> {
        let [source_sampler_key] = keys.samplers else {
            return Err(blur_cache_error(
                "blur realization requires one source sampler",
            ));
        };
        let source_sampler = self
            .samplers
            .get(source_sampler_key)
            .or_else(|| cache.samplers.get(source_sampler_key))
            .ok_or_else(|| blur_cache_error("blur sampler realization was lost"))?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| blur_cache_error("blur bind-group layout realization was lost"))?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| blur_cache_error("blur shader-module realization was lost"))?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| blur_cache_error("blur render-pipeline realization was lost"))?;
        Ok(ProvisionalBlurPassObjects {
            description: validate_blur_pass_keys(keys)?,
            source_sampler,
            layout,
            shader,
            pipeline,
        })
    }

    fn color_filter_pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: ColorFilterPassKeyRefs<'_>,
    ) -> Result<ProvisionalColorFilterPassObjects<'a>> {
        let [source_sampler_key] = keys.samplers else {
            return Err(color_filter_cache_error(
                "color-filter realization requires one source sampler",
            ));
        };
        let source_sampler = self
            .samplers
            .get(source_sampler_key)
            .or_else(|| cache.samplers.get(source_sampler_key))
            .ok_or_else(|| color_filter_cache_error("color-filter sampler realization was lost"))?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| {
                color_filter_cache_error("color-filter bind-group layout realization was lost")
            })?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| {
                color_filter_cache_error("color-filter shader-module realization was lost")
            })?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| {
                color_filter_cache_error("color-filter render-pipeline realization was lost")
            })?;
        Ok(ProvisionalColorFilterPassObjects {
            description: validate_color_filter_pass_keys(keys)?,
            source_sampler,
            layout,
            shader,
            pipeline,
        })
    }

    pub(crate) fn realize_composite_pass<'a>(
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
            None,
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
            Some("missing_c09_composite_fragment"),
        )
    }

    fn realize_composite_pass_with_fragment_entry<'a>(
        &'a mut self,
        device: &wgpu::Device,
        cache: &'a DevicePassCache,
        keys: CompositePassKeyRefs<'_>,
        fragment_entry_override: Option<&'static str>,
    ) -> Result<ProvisionalCompositePassObjects<'a>> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(c08_cache_error(
                "provisional composite pass objects belong to another device cache",
            ));
        }
        let description = validate_composite_pass_keys(keys)?;
        for sampler_key in keys.samplers {
            if !cache.samplers.contains_key(sampler_key) && !self.samplers.contains_key(sampler_key)
            {
                self.samplers
                    .insert(*sampler_key, create_sampler(device, *sampler_key));
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_composite_bind_group_layout(device, description),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders
                .insert(keys.shader.clone(), create_composite_shader_module(device));
        }
        if !cache.pipelines.contains_key(keys.pipeline)
            && !self.pipelines.contains_key(keys.pipeline)
        {
            let layout_handle = self
                .layouts
                .get(keys.layout)
                .or_else(|| cache.layouts.get(keys.layout))
                .ok_or_else(|| {
                    c08_cache_error("composite bind-group layout realization was lost")
                })?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| c08_cache_error("composite shader-module realization was lost"))?;
            let created = create_composite_render_pipeline(
                device,
                description,
                layout_handle,
                shader_handle,
                fragment_entry_override,
            )?;
            self.pipelines.insert(keys.pipeline.clone(), created);
        }
        self.composite_pass_objects(cache, keys)
    }

    fn composite_pass_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        keys: CompositePassKeyRefs<'_>,
    ) -> Result<ProvisionalCompositePassObjects<'a>> {
        let [source_sampler_key] = keys.samplers else {
            return Err(c08_cache_error(
                "composite pass realization requires one source sampler",
            ));
        };
        let source_sampler = self
            .samplers
            .get(source_sampler_key)
            .or_else(|| cache.samplers.get(source_sampler_key))
            .ok_or_else(|| c08_cache_error("composite source sampler realization was lost"))?;
        let layout = self
            .layouts
            .get(keys.layout)
            .or_else(|| cache.layouts.get(keys.layout))
            .ok_or_else(|| c08_cache_error("composite bind-group layout realization was lost"))?;
        let shader = self
            .shaders
            .get(keys.shader)
            .or_else(|| cache.shaders.get(keys.shader))
            .ok_or_else(|| c08_cache_error("composite shader-module realization was lost"))?;
        let pipeline = self
            .pipelines
            .get(keys.pipeline)
            .or_else(|| cache.pipelines.get(keys.pipeline))
            .ok_or_else(|| c08_cache_error("composite render-pipeline realization was lost"))?;
        Ok(ProvisionalCompositePassObjects {
            description: validate_composite_pass_keys(keys)?,
            source_sampler,
            layout,
            shader,
            pipeline,
        })
    }

    pub(crate) fn composite_encoding_objects<'a>(
        &'a self,
        cache: &'a DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> Result<ProvisionalCompositePassObjects<'a>> {
        self.composite_pass_objects(
            cache,
            CompositePassKeyRefs {
                samplers,
                layout,
                shader,
                pipeline,
            },
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

    pub(crate) fn ensure_commit_ready(&self, cache: &DevicePassCache) -> Result<()> {
        if !Arc::ptr_eq(&self.cache_identity, &cache.identity) {
            return Err(c08_cache_error(
                "provisional pass objects cannot enter another device cache",
            ));
        }
        if self
            .samplers
            .keys()
            .any(|key| cache.samplers.contains_key(key))
            || self
                .layouts
                .keys()
                .any(|key| cache.layouts.contains_key(key))
            || self
                .shaders
                .keys()
                .any(|key| cache.shaders.contains_key(key))
            || self
                .pipelines
                .keys()
                .any(|key| cache.pipelines.contains_key(key))
        {
            return Err(c08_cache_error(
                "persistent pass cache changed during provisional realization",
            ));
        }
        Ok(())
    }

    pub(crate) fn commit(self, cache: &mut DevicePassCache) -> Result<()> {
        self.ensure_commit_ready(cache)?;
        cache.samplers.extend(self.samplers);
        cache.layouts.extend(self.layouts);
        cache.shaders.extend(self.shaders);
        cache.pipelines.extend(self.pipelines);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn contains_c08_pass_for_test(
        &self,
        cache: &DevicePassCache,
        samplers: &[SamplerKey],
        layout: &BindGroupLayoutKey,
        shader: &ShaderModuleKey,
        pipeline: &RenderPipelineKey,
    ) -> bool {
        self.pass_objects(
            cache,
            C08PassKeyRefs {
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

impl ProvisionalC08PassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (self.layout, self.shader, self.pipeline);
        if self.samplers.len() != 1 {
            return Err(c08_cache_error(
                "C08 pass realization did not retain its exact encoding handles",
            ));
        }
        Ok(())
    }

    pub(crate) fn sampler(&self) -> Result<&wgpu::Sampler> {
        let [sampler] = self.samplers.as_slice() else {
            return Err(c08_cache_error(
                "C08 pass encoding requires one exact sampled-image sampler",
            ));
        };
        Ok(sampler)
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }

    pub(crate) const fn uses_fixed_source_over_blend(&self) -> bool {
        matches!(
            self.program,
            C08Program::SpanSourceOver | C08Program::DropShadowMerge
        )
    }

    #[cfg(test)]
    fn is_encoding_ready_for_test(&self) -> bool {
        self.require_encoding_ready().is_ok()
    }
}

impl ProvisionalCompositePassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (
            self.description,
            self.layout,
            self.shader,
            self.pipeline,
            self.source_sampler,
        );
        Ok(())
    }

    pub(crate) const fn source_sampler(&self) -> &wgpu::Sampler {
        self.source_sampler
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }

    pub(crate) const fn path(&self) -> ShaderCompositePathKey {
        self.description.path
    }

    pub(crate) const fn has_clip_coverage(&self) -> bool {
        self.description.has_clip_coverage
    }

    pub(crate) const fn has_alpha_mask(&self) -> bool {
        self.description.has_alpha_mask
    }

    pub(crate) const fn uses_fixed_source_over_blend(&self) -> bool {
        matches!(self.description.path, ShaderCompositePathKey::Normal)
    }

    pub(crate) const fn uses_replace_blend(&self) -> bool {
        matches!(
            self.description.path,
            ShaderCompositePathKey::DestinationSampling
        )
    }
}

impl ProvisionalColorFilterPassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (
            self.description,
            self.source_sampler,
            self.layout,
            self.shader,
            self.pipeline,
        );
        Ok(())
    }

    pub(crate) const fn source_sampler(&self) -> &wgpu::Sampler {
        self.source_sampler
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }
}

impl ProvisionalCopyBackdropPassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (
            self.description,
            self.parent_sampler(),
            self.bind_group_layout(),
            self.shader,
            self.render_pipeline(),
        );
        Ok(())
    }

    pub(crate) const fn parent_sampler(&self) -> &wgpu::Sampler {
        self.parent_sampler
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }
}

impl ProvisionalBlurPassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (
            self.description,
            self.source_sampler(),
            self.bind_group_layout(),
            self.shader,
            self.render_pipeline(),
        );
        Ok(())
    }

    pub(crate) const fn source_sampler(&self) -> &wgpu::Sampler {
        self.source_sampler
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }
}

impl ProvisionalDropShadowColorizePassObjects<'_> {
    pub(crate) fn require_encoding_ready(&self) -> Result<()> {
        let _ = (
            self.description,
            self.source_sampler(),
            self.bind_group_layout(),
            self.shader,
            self.render_pipeline(),
        );
        Ok(())
    }

    pub(crate) const fn source_sampler(&self) -> &wgpu::Sampler {
        self.source_sampler
    }

    pub(crate) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.layout
    }

    pub(crate) const fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        self.pipeline
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum C08ProgramForTest {
    CanonicalizeCapture,
    SpanSourceOver,
    DropShadowMerge,
    Present,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct C08PassKeyFactsForTest {
    pub(crate) program: C08ProgramForTest,
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) output_format: Option<ShaderTextureFormatKey>,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_spatial_uniform: bool,
    pub(crate) has_fixed_source_over_blend: bool,
}

#[cfg(test)]
pub(crate) fn c08_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C08PassKeyFactsForTest> {
    let [sampled_texture] = layout.sampled_textures.as_slice() else {
        return None;
    };
    let description = validate_c08_pass_keys(C08PassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(C08PassKeyFactsForTest {
        program: match description.program {
            C08Program::CanonicalizeCapture => C08ProgramForTest::CanonicalizeCapture,
            C08Program::SpanSourceOver => C08ProgramForTest::SpanSourceOver,
            C08Program::DropShadowMerge => C08ProgramForTest::DropShadowMerge,
            C08Program::Present => C08ProgramForTest::Present,
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
            C08Program::SpanSourceOver | C08Program::DropShadowMerge
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
pub(crate) struct C09CompositePassKeyFactsForTest {
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
pub(crate) fn c09_composite_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C09CompositePassKeyFactsForTest> {
    let description = validate_composite_pass_keys(CompositePassKeyRefs {
        samplers,
        layout,
        shader,
        pipeline,
    })
    .ok()?;
    Some(C09CompositePassKeyFactsForTest {
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
pub(crate) struct C10ColorFilterPassKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_nearest_source_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn c10_color_filter_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C10ColorFilterPassKeyFactsForTest> {
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
    Some(C10ColorFilterPassKeyFactsForTest {
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
pub(crate) struct C12CopyBackdropPassKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_nearest_transparent_sampler: bool,
    pub(crate) has_only_spatial_uniform: bool,
}

#[cfg(test)]
pub(crate) fn c12_copy_backdrop_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C12CopyBackdropPassKeyFactsForTest> {
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
    Some(C12CopyBackdropPassKeyFactsForTest {
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
pub(crate) struct C11BlurPassKeyFactsForTest {
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
pub(crate) fn c11_blur_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C11BlurPassKeyFactsForTest> {
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
    Some(C11BlurPassKeyFactsForTest {
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
pub(crate) struct C12BackdropBlurPassKeyFactsForTest {
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
pub(crate) fn c12_backdrop_blur_pass_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C12BackdropBlurPassKeyFactsForTest> {
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
    Some(C12BackdropBlurPassKeyFactsForTest {
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
pub(crate) fn c12_blur_shader_mirrors_semantic_bounds_before_texture_mapping_for_test() -> bool {
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
pub(crate) struct C11DropShadowColorizeKeyFactsForTest {
    pub(crate) source_role: ShaderBindingRoleKey,
    pub(crate) source_format: ShaderTextureFormatKey,
    pub(crate) working_format: ShaderTextureFormatKey,
    pub(crate) target_format: ShaderTextureFormatKey,
    pub(crate) has_only_linear_transparent_sampler: bool,
    pub(crate) has_exact_data_bindings: bool,
}

#[cfg(test)]
pub(crate) fn c11_drop_shadow_colorize_key_facts_for_test(
    samplers: &[SamplerKey],
    layout: &BindGroupLayoutKey,
    shader: &ShaderModuleKey,
    pipeline: &RenderPipelineKey,
) -> Option<C11DropShadowColorizeKeyFactsForTest> {
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
    Some(C11DropShadowColorizeKeyFactsForTest {
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
