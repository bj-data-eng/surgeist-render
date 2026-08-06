use std::{collections::HashMap, sync::Arc};

use crate::Result;

use super::key::{
    BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderCompositePathKey, ShaderModuleKey,
};
use super::pipeline::{
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
use super::validate::{
    BlurPassDescription, BlurPassKeyRefs, C08PassKeyRefs, C08Program, ColorFilterPassDescription,
    ColorFilterPassKeyRefs, CompositePassDescription, CompositePassKeyRefs,
    CopyBackdropPassDescription, CopyBackdropPassKeyRefs, DropShadowColorizePassDescription,
    DropShadowColorizePassKeyRefs, blur_cache_error, c08_cache_error, color_filter_cache_error,
    copy_backdrop_cache_error, drop_shadow_cache_error, validate_blur_pass_keys,
    validate_c08_pass_keys, validate_color_filter_pass_keys, validate_composite_pass_keys,
    validate_copy_backdrop_pass_keys, validate_drop_shadow_colorize_pass_keys,
};

/// Non-clone handles created inside one checked GPU-operation scope. New entries
/// remain private to this phase until the caller explicitly commits after the
/// owning transaction resolves cleanly.
#[must_use = "provisional pass-cache handles must be committed only after checked success"]
pub(crate) struct ProvisionalDevicePassCacheUpdate {
    pub(super) cache_identity: Arc<()>,
    pub(super) samplers: HashMap<SamplerKey, wgpu::Sampler>,
    pub(super) layouts: HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>,
    pub(super) shaders: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    pub(super) pipelines: HashMap<RenderPipelineKey, wgpu::RenderPipeline>,
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
    pub(super) identity: Arc<()>,
    pub(super) samplers: HashMap<SamplerKey, wgpu::Sampler>,
    pub(super) layouts: HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>,
    pub(super) shaders: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    pub(super) pipelines: HashMap<RenderPipelineKey, wgpu::RenderPipeline>,
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

    pub(super) fn realize_c08_pass_with_fragment_entry<'a>(
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

    pub(super) fn pass_objects<'a>(
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

    pub(super) fn realize_composite_pass_with_fragment_entry<'a>(
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

    pub(super) fn composite_pass_objects<'a>(
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
