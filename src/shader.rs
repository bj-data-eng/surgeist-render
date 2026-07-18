use std::collections::HashMap;

use super::{
    Error, Format, Result, image::ResolvedMaskUploadKey, pass::RuntimeSpatialDescriptor,
    resource::WorkingFormat,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderBlendKey {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Plus,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderCompositeKey {
    SpanSourceOver,
    Layer {
        blend: ShaderBlendKey,
        has_clip: bool,
        has_outer_clips: bool,
        has_alpha_mask: bool,
    },
    DropShadow,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderProgramKey {
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter,
    BlurHorizontal { source_alpha: bool },
    BlurVertical { source_alpha: bool },
    DropShadowColorize,
    Composite(ShaderCompositeKey),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderTextureFormatKey {
    VelloCaptureRgba8Unorm,
    WorkingHighPrecisionRgba16Float,
    WorkingReducedPrecisionRgba8Unorm,
    ResolvedMaskRgba8Unorm,
    OutputRgba8Unorm,
    OutputBgra8Unorm,
}

impl ShaderTextureFormatKey {
    #[must_use]
    pub(crate) const fn working(format: WorkingFormat) -> Self {
        match format {
            WorkingFormat::HighPrecision => Self::WorkingHighPrecisionRgba16Float,
            WorkingFormat::ReducedPrecision => Self::WorkingReducedPrecisionRgba8Unorm,
        }
    }

    #[must_use]
    pub(crate) const fn output(format: Format) -> Self {
        match format {
            Format::Rgba8 => Self::OutputRgba8Unorm,
            Format::Bgra8 => Self::OutputBgra8Unorm,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderBindingRoleKey {
    CaptureSource,
    CompletedParent,
    FilterSource,
    BlurredSourceAlpha,
    CompositeParent,
    CompositeSource,
    AlphaMask,
    Shadow,
    FinalWorkingImage,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderSamplingFilterKey {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderSamplingEdgeKey {
    ClampToExtent,
    TransparentBlack,
    SemanticBorderMirror,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderDataBindingKey {
    SpatialUniform,
    ColorFilterOperations,
    GaussianKernel,
    DropShadowParameters,
    CompositeParameters,
    PresentParameters,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SamplerKey {
    binding_role: ShaderBindingRoleKey,
    source_format: ShaderTextureFormatKey,
    filter: ShaderSamplingFilterKey,
    edge: ShaderSamplingEdgeKey,
    resolved_mask_sampling: Option<ResolvedMaskUploadKey>,
}

impl SamplerKey {
    #[must_use]
    pub(crate) const fn new(
        binding_role: ShaderBindingRoleKey,
        source_format: ShaderTextureFormatKey,
        filter: ShaderSamplingFilterKey,
        edge: ShaderSamplingEdgeKey,
        resolved_mask_sampling: Option<ResolvedMaskUploadKey>,
    ) -> Self {
        Self {
            binding_role,
            source_format,
            filter,
            edge,
            resolved_mask_sampling,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn facts_for_test(
        self,
    ) -> (
        ShaderBindingRoleKey,
        ShaderTextureFormatKey,
        ShaderSamplingFilterKey,
        ShaderSamplingEdgeKey,
        Option<ResolvedMaskUploadKey>,
    ) {
        (
            self.binding_role,
            self.source_format,
            self.filter,
            self.edge,
            self.resolved_mask_sampling,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SampledTextureLayoutKey {
    binding_role: ShaderBindingRoleKey,
    source_format: ShaderTextureFormatKey,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BindGroupLayoutKey {
    program: ShaderProgramKey,
    sampled_textures: Vec<SampledTextureLayoutKey>,
    data_bindings: Vec<ShaderDataBindingKey>,
}

impl BindGroupLayoutKey {
    #[must_use]
    pub(crate) fn new(
        program: ShaderProgramKey,
        sampled_textures: &[(ShaderBindingRoleKey, ShaderTextureFormatKey)],
        data_bindings: Vec<ShaderDataBindingKey>,
    ) -> Self {
        Self {
            program,
            sampled_textures: sampled_textures
                .iter()
                .copied()
                .map(|(binding_role, source_format)| SampledTextureLayoutKey {
                    binding_role,
                    source_format,
                })
                .collect(),
            data_bindings,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShaderModuleKey {
    program: ShaderProgramKey,
    layout: BindGroupLayoutKey,
    samplers: Vec<SamplerKey>,
    working_format: Option<ShaderTextureFormatKey>,
    output_format: Option<ShaderTextureFormatKey>,
}

impl ShaderModuleKey {
    #[must_use]
    pub(crate) fn new(
        program: ShaderProgramKey,
        layout: BindGroupLayoutKey,
        samplers: Vec<SamplerKey>,
        working_format: Option<ShaderTextureFormatKey>,
        output_format: Option<ShaderTextureFormatKey>,
    ) -> Self {
        Self {
            program,
            layout,
            samplers,
            working_format,
            output_format,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RenderPipelineKey {
    shader: ShaderModuleKey,
    layout: BindGroupLayoutKey,
    samplers: Vec<SamplerKey>,
    target_format: ShaderTextureFormatKey,
}

impl RenderPipelineKey {
    #[must_use]
    pub(crate) fn new(
        shader: ShaderModuleKey,
        layout: BindGroupLayoutKey,
        samplers: Vec<SamplerKey>,
        target_format: ShaderTextureFormatKey,
    ) -> Self {
        Self {
            shader,
            layout,
            samplers,
            target_format,
        }
    }
}

/// Device-lifetime ownership for the four exact custom-pass WGPU handle spaces.
pub(crate) struct DevicePassCache {
    samplers: HashMap<SamplerKey, wgpu::Sampler>,
    layouts: HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>,
    shaders: HashMap<ShaderModuleKey, wgpu::ShaderModule>,
    pipelines: HashMap<RenderPipelineKey, wgpu::RenderPipeline>,
}

impl DevicePassCache {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
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

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.samplers.is_empty()
            && self.layouts.is_empty()
            && self.shaders.is_empty()
            && self.pipelines.is_empty()
    }
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

/// Exact 48-byte WGSL spatial uniform with explicit little-endian encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PassSpatialUniformBytes([u8; 48]);

impl PassSpatialUniformBytes {
    pub(crate) fn try_from_runtime_spatial_descriptors(
        source: RuntimeSpatialDescriptor,
        destination: RuntimeSpatialDescriptor,
    ) -> Result<Self> {
        let source_origin = source.texel_origin();
        let source_origin_x =
            narrow_spatial_scalar("pass spatial source origin x", source_origin.x())?;
        let source_origin_y =
            narrow_spatial_scalar("pass spatial source origin y", source_origin.y())?;
        let source_raster_scale =
            narrow_raster_scale("pass spatial source raster scale", source.raster_scale())?;

        let destination_origin = destination.texel_origin();
        let destination_origin_x =
            narrow_spatial_scalar("pass spatial destination origin x", destination_origin.x())?;
        let destination_origin_y =
            narrow_spatial_scalar("pass spatial destination origin y", destination_origin.y())?;
        let destination_raster_scale = narrow_raster_scale(
            "pass spatial destination raster scale",
            destination.raster_scale(),
        )?;

        let source_extent = source.device_extent();
        let destination_extent = destination.device_extent();
        let mut bytes = [0_u8; 48];
        bytes[0..4].copy_from_slice(&source_origin_x.to_le_bytes());
        bytes[4..8].copy_from_slice(&source_origin_y.to_le_bytes());
        bytes[8..12].copy_from_slice(&source_raster_scale.to_le_bytes());
        bytes[16..20].copy_from_slice(&destination_origin_x.to_le_bytes());
        bytes[20..24].copy_from_slice(&destination_origin_y.to_le_bytes());
        bytes[24..28].copy_from_slice(&destination_raster_scale.to_le_bytes());
        bytes[32..36].copy_from_slice(&source_extent.width().to_le_bytes());
        bytes[36..40].copy_from_slice(&source_extent.height().to_le_bytes());
        bytes[40..44].copy_from_slice(&destination_extent.width().to_le_bytes());
        bytes[44..48].copy_from_slice(&destination_extent.height().to_le_bytes());
        Ok(Self(bytes))
    }

    #[must_use]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 writes the validated prepared spatial bytes into uniform buffers"
        )
    )]
    pub(crate) const fn as_bytes(&self) -> &[u8; 48] {
        &self.0
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn into_bytes_for_test(self) -> [u8; 48] {
        self.0
    }
}

fn narrow_spatial_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain finite after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn narrow_raster_scale(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = narrow_spatial_scalar(field, value)?;
    if narrowed <= 0.0 {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain strictly positive after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}
