use crate::{
    Format,
    image::{Extend, ImageQuality},
    resource::WorkingFormat,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderCompositePathKey {
    Normal,
    DestinationSampling,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderCompositeKey {
    SpanSourceOver,
    Layer {
        path: ShaderCompositePathKey,
        has_clip_coverage: bool,
        has_alpha_mask: bool,
    },
    DropShadow,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderProgramKey {
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter,
    BlurHorizontal {
        source_alpha: bool,
        edge: ShaderSamplingEdgeKey,
    },
    BlurVertical {
        source_alpha: bool,
        edge: ShaderSamplingEdgeKey,
    },
    DropShadowColorize,
    Composite(ShaderCompositeKey),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderTextureFormatKey {
    VelloCaptureRgba8Unorm,
    ClipCoverageRgba8Unorm,
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
    ClipCoverage,
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
pub(crate) enum ShaderMaskQualityKey {
    Low,
    Medium,
    High,
}

impl ShaderMaskQualityKey {
    #[must_use]
    pub(crate) const fn from_image_quality(quality: ImageQuality) -> Self {
        match quality {
            ImageQuality::Low => Self::Low,
            ImageQuality::Medium => Self::Medium,
            ImageQuality::High => Self::High,
        }
    }

    pub(super) const fn parameter_code(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderMaskExtendKey {
    Pad,
    Repeat,
    Reflect,
}

impl ShaderMaskExtendKey {
    #[must_use]
    pub(crate) const fn from_extend(extend: Extend) -> Self {
        match extend {
            Extend::Pad => Self::Pad,
            Extend::Repeat => Self::Repeat,
            Extend::Reflect => Self::Reflect,
        }
    }

    pub(super) const fn parameter_code(self) -> u32 {
        match self {
            Self::Pad => 0,
            Self::Repeat => 1,
            Self::Reflect => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShaderMaskSamplingKey {
    pub(super) quality: ShaderMaskQualityKey,
    pub(super) extend: ShaderMaskExtendKey,
}

impl ShaderMaskSamplingKey {
    #[must_use]
    pub(crate) const fn new(quality: ImageQuality, extend: Extend) -> Self {
        Self {
            quality: ShaderMaskQualityKey::from_image_quality(quality),
            extend: ShaderMaskExtendKey::from_extend(extend),
        }
    }

    #[must_use]
    pub(crate) const fn quality(self) -> ShaderMaskQualityKey {
        self.quality
    }

    #[must_use]
    pub(crate) const fn extend(self) -> ShaderMaskExtendKey {
        self.extend
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ShaderDataBindingKey {
    SpatialUniform,
    ColorFilterOperations,
    GaussianKernel,
    BlurEdgeParameters,
    DropShadowParameters,
    CompositeParameters,
    PresentParameters,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SamplerKey {
    pub(super) binding_role: ShaderBindingRoleKey,
    pub(super) source_format: ShaderTextureFormatKey,
    pub(super) filter: ShaderSamplingFilterKey,
    pub(super) edge: ShaderSamplingEdgeKey,
    pub(super) resolved_mask_sampling: Option<ShaderMaskSamplingKey>,
}

impl SamplerKey {
    #[must_use]
    pub(crate) const fn new(
        binding_role: ShaderBindingRoleKey,
        source_format: ShaderTextureFormatKey,
        filter: ShaderSamplingFilterKey,
        edge: ShaderSamplingEdgeKey,
        resolved_mask_sampling: Option<ShaderMaskSamplingKey>,
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
        Option<ShaderMaskSamplingKey>,
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
pub(super) struct SampledTextureLayoutKey {
    pub(super) binding_role: ShaderBindingRoleKey,
    pub(super) source_format: ShaderTextureFormatKey,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BindGroupLayoutKey {
    pub(super) program: ShaderProgramKey,
    pub(super) sampled_textures: Vec<SampledTextureLayoutKey>,
    pub(super) data_bindings: Vec<ShaderDataBindingKey>,
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
    pub(super) program: ShaderProgramKey,
    pub(super) layout: BindGroupLayoutKey,
    pub(super) samplers: Vec<SamplerKey>,
    pub(super) working_format: Option<ShaderTextureFormatKey>,
    pub(super) output_format: Option<ShaderTextureFormatKey>,
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
    pub(super) shader: ShaderModuleKey,
    pub(super) layout: BindGroupLayoutKey,
    pub(super) samplers: Vec<SamplerKey>,
    pub(super) target_format: ShaderTextureFormatKey,
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
