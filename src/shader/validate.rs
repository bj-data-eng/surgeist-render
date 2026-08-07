use crate::{BackendErrorCode, Error, Result};

use super::key::{
    BindGroupLayoutKey, RenderPipelineKey, SampledTextureLayoutKey, SamplerKey,
    ShaderBindingRoleKey, ShaderCompositeKey, ShaderCompositePathKey, ShaderDataBindingKey,
    ShaderModuleKey, ShaderProgramKey, ShaderSamplingEdgeKey, ShaderSamplingFilterKey,
    ShaderTextureFormatKey,
};

const CORE_PASS_EXCLUDED_COMPOSITION_PARAMETER_BINDINGS: [ShaderDataBindingKey; 2] = [
    ShaderDataBindingKey::CompositeParameters,
    ShaderDataBindingKey::PresentParameters,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CorePassProgram {
    CanonicalizeCapture,
    SpanSourceOver,
    DropShadowMerge,
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CorePassDescription {
    pub(super) program: CorePassProgram,
    pub(super) target_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct CorePassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompositePassDescription {
    pub(super) path: ShaderCompositePathKey,
    pub(super) has_clip_coverage: bool,
    pub(super) has_alpha_mask: bool,
    pub(super) target_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct CompositePassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ColorFilterPassDescription {
    pub(super) working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct ColorFilterPassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CopyBackdropPassDescription {
    pub(super) working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct CopyBackdropPassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BlurPassDescription {
    pub(super) axis: BlurAxis,
    pub(super) input: BlurInput,
    pub(super) edge: ShaderSamplingEdgeKey,
    pub(super) working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct BlurPassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DropShadowColorizePassDescription {
    pub(super) working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
pub(super) struct DropShadowColorizePassKeyRefs<'a> {
    pub(super) samplers: &'a [SamplerKey],
    pub(super) layout: &'a BindGroupLayoutKey,
    pub(super) shader: &'a ShaderModuleKey,
    pub(super) pipeline: &'a RenderPipelineKey,
}

pub(super) fn validate_core_pass_keys(keys: CorePassKeyRefs<'_>) -> Result<CorePassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(core_pass_cache_error(
            "a core pass layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(core_pass_cache_error(
            "a core pass must bind exactly one exact sampler",
        ));
    };
    validate_core_pass_key_consistency(keys, sampled_texture, sampler)?;
    let (program, expected_role, expected_filter, expected_edge) =
        core_pass_program_sampling(keys.layout.program)?;
    if sampled_texture.binding_role != expected_role
        || sampler.filter != expected_filter
        || sampler.edge != expected_edge
    {
        return Err(core_pass_cache_error(
            "core pass sampled texture and sampler semantics are not exact",
        ));
    }

    let Some(working_format) = keys.shader.working_format else {
        return Err(core_pass_cache_error(
            "core-pass shader key has no selected working format",
        ));
    };
    if !is_working_format(working_format) {
        return Err(core_pass_cache_error(
            "core-pass shader key selected a non-working intermediate format",
        ));
    }
    let target_format = core_pass_target_format(keys, sampled_texture, program, working_format)?;
    Ok(CorePassDescription {
        program,
        target_format,
    })
}

fn validate_core_pass_key_consistency(
    keys: CorePassKeyRefs<'_>,
    sampled_texture: &SampledTextureLayoutKey,
    sampler: &SamplerKey,
) -> Result<()> {
    if keys.layout.data_bindings.as_slice() != [ShaderDataBindingKey::SpatialUniform]
        || keys
            .layout
            .data_bindings
            .iter()
            .any(|binding| CORE_PASS_EXCLUDED_COMPOSITION_PARAMETER_BINDINGS.contains(binding))
        || keys.shader.program != keys.layout.program
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || sampler.binding_role != sampled_texture.binding_role
        || sampler.source_format != sampled_texture.source_format
        || sampler.resolved_mask_sampling.is_some()
    {
        return Err(core_pass_cache_error(
            "core pass keys disagree across sampler, layout, shader, or pipeline phases",
        ));
    }
    Ok(())
}

fn core_pass_program_sampling(
    program: ShaderProgramKey,
) -> Result<(
    CorePassProgram,
    ShaderBindingRoleKey,
    ShaderSamplingFilterKey,
    ShaderSamplingEdgeKey,
)> {
    let sampling = match program {
        ShaderProgramKey::CanonicalizeCapture => (
            CorePassProgram::CanonicalizeCapture,
            ShaderBindingRoleKey::CaptureSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::ClampToExtent,
        ),
        ShaderProgramKey::Composite(ShaderCompositeKey::SpanSourceOver) => (
            CorePassProgram::SpanSourceOver,
            ShaderBindingRoleKey::CompositeSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::TransparentBlack,
        ),
        ShaderProgramKey::Composite(ShaderCompositeKey::DropShadow) => (
            CorePassProgram::DropShadowMerge,
            ShaderBindingRoleKey::CompositeSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::TransparentBlack,
        ),
        ShaderProgramKey::Present => (
            CorePassProgram::Present,
            ShaderBindingRoleKey::FinalWorkingImage,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::ClampToExtent,
        ),
        ShaderProgramKey::CopyBackdrop
        | ShaderProgramKey::ColorFilter
        | ShaderProgramKey::BlurHorizontal { .. }
        | ShaderProgramKey::BlurVertical { .. }
        | ShaderProgramKey::DropShadowColorize
        | ShaderProgramKey::Composite(ShaderCompositeKey::Layer { .. }) => {
            return Err(core_pass_cache_error(
                "an unsupported shader program reached core pass realization",
            ));
        }
    };
    Ok(sampling)
}

fn core_pass_target_format(
    keys: CorePassKeyRefs<'_>,
    sampled_texture: &SampledTextureLayoutKey,
    program: CorePassProgram,
    working_format: ShaderTextureFormatKey,
) -> Result<ShaderTextureFormatKey> {
    let target_format = match program {
        CorePassProgram::CanonicalizeCapture => {
            if sampled_texture.source_format != ShaderTextureFormatKey::VelloCaptureRgba8Unorm
                || keys.shader.output_format.is_some()
                || keys.pipeline.target_format != working_format
            {
                return Err(core_pass_cache_error(
                    "canonicalization keys changed capture or working formats",
                ));
            }
            working_format
        }
        CorePassProgram::SpanSourceOver | CorePassProgram::DropShadowMerge => {
            if sampled_texture.source_format != working_format
                || keys.shader.output_format.is_some()
                || keys.pipeline.target_format != working_format
            {
                return Err(core_pass_cache_error(
                    "source-over keys changed source or working target formats",
                ));
            }
            working_format
        }
        CorePassProgram::Present => {
            let Some(output_format) = keys.shader.output_format else {
                return Err(core_pass_cache_error(
                    "present key has no exact output format",
                ));
            };
            if sampled_texture.source_format != working_format
                || !is_output_format(output_format)
                || keys.pipeline.target_format != output_format
            {
                return Err(core_pass_cache_error(
                    "present keys changed working source or output specialization",
                ));
            }
            output_format
        }
    };
    Ok(target_format)
}

pub(super) fn validate_color_filter_pass_keys(
    keys: ColorFilterPassKeyRefs<'_>,
) -> Result<ColorFilterPassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(color_filter_cache_error(
            "a color-filter layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(color_filter_cache_error(
            "a color-filter pass must bind exactly one source sampler",
        ));
    };
    let Some(working_format) = keys.shader.working_format else {
        return Err(color_filter_cache_error(
            "a color-filter shader key has no selected working format",
        ));
    };
    if keys.layout.program != ShaderProgramKey::ColorFilter
        || keys.layout.data_bindings.as_slice()
            != [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::ColorFilterOperations,
            ]
        || keys.shader.program != ShaderProgramKey::ColorFilter
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || keys.shader.output_format.is_some()
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || !is_working_format(working_format)
        || keys.pipeline.target_format != working_format
        || sampled_texture.binding_role != ShaderBindingRoleKey::FilterSource
        || sampled_texture.source_format != working_format
        || sampler.binding_role != ShaderBindingRoleKey::FilterSource
        || sampler.source_format != working_format
        || sampler.filter != ShaderSamplingFilterKey::Nearest
        || sampler.edge != ShaderSamplingEdgeKey::ClampToExtent
        || sampler.resolved_mask_sampling.is_some()
    {
        return Err(color_filter_cache_error(
            "color-filter keys disagree across source, layout, shader, or working target",
        ));
    }
    Ok(ColorFilterPassDescription { working_format })
}

pub(super) fn validate_copy_backdrop_pass_keys(
    keys: CopyBackdropPassKeyRefs<'_>,
) -> Result<CopyBackdropPassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(copy_backdrop_cache_error(
            "a backdrop-copy layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(copy_backdrop_cache_error(
            "a backdrop-copy pass must bind exactly one parent sampler",
        ));
    };
    let Some(working_format) = keys.shader.working_format else {
        return Err(copy_backdrop_cache_error(
            "a backdrop-copy shader key has no selected working format",
        ));
    };
    if keys.layout.program != ShaderProgramKey::CopyBackdrop
        || keys.layout.data_bindings.as_slice() != [ShaderDataBindingKey::SpatialUniform]
        || keys.shader.program != ShaderProgramKey::CopyBackdrop
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || keys.shader.output_format.is_some()
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || !is_working_format(working_format)
        || keys.pipeline.target_format != working_format
        || sampled_texture.binding_role != ShaderBindingRoleKey::CompletedParent
        || sampled_texture.source_format != working_format
        || sampler.binding_role != ShaderBindingRoleKey::CompletedParent
        || sampler.source_format != working_format
        || sampler.filter != ShaderSamplingFilterKey::Nearest
        || sampler.edge != ShaderSamplingEdgeKey::TransparentBlack
        || sampler.resolved_mask_sampling.is_some()
    {
        return Err(copy_backdrop_cache_error(
            "backdrop-copy keys disagree across parent, layout, shader, or working target",
        ));
    }
    Ok(CopyBackdropPassDescription { working_format })
}

pub(super) fn validate_blur_pass_keys(keys: BlurPassKeyRefs<'_>) -> Result<BlurPassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(blur_cache_error(
            "a blur layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(blur_cache_error(
            "a blur pass must bind exactly one source sampler",
        ));
    };
    let (axis, input, edge) = blur_program_facts(keys.layout.program).ok_or_else(|| {
        blur_cache_error("a non-blur program reached spatial-filter blur realization")
    })?;
    let Some(working_format) = keys.shader.working_format else {
        return Err(blur_cache_error(
            "a blur shader key has no selected working format",
        ));
    };
    let expected_data_bindings = match edge {
        ShaderSamplingEdgeKey::TransparentBlack => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::GaussianKernel,
        ],
        ShaderSamplingEdgeKey::SemanticBorderMirror => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::GaussianKernel,
            ShaderDataBindingKey::BlurEdgeParameters,
        ],
        ShaderSamplingEdgeKey::ClampToExtent => {
            return Err(blur_cache_error(
                "a Gaussian blur program has no clamp-to-extent edge policy",
            ));
        }
    };
    if keys.layout.data_bindings != expected_data_bindings
        || keys.shader.program != keys.layout.program
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || keys.shader.output_format.is_some()
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || !is_working_format(working_format)
        || keys.pipeline.target_format != working_format
        || sampled_texture.binding_role != ShaderBindingRoleKey::FilterSource
        || sampled_texture.source_format != working_format
        || sampler.binding_role != ShaderBindingRoleKey::FilterSource
        || sampler.source_format != working_format
        || sampler.filter != ShaderSamplingFilterKey::Linear
        || sampler.edge != edge
        || sampler.resolved_mask_sampling.is_some()
    {
        return Err(blur_cache_error(
            "blur keys disagree across source, layout, shader, or working target",
        ));
    }
    Ok(BlurPassDescription {
        axis,
        input,
        edge,
        working_format,
    })
}

pub(super) fn validate_drop_shadow_colorize_pass_keys(
    keys: DropShadowColorizePassKeyRefs<'_>,
) -> Result<DropShadowColorizePassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(drop_shadow_cache_error(
            "a drop-shadow colorize layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(drop_shadow_cache_error(
            "a drop-shadow colorize pass must bind exactly one source sampler",
        ));
    };
    let Some(working_format) = keys.shader.working_format else {
        return Err(drop_shadow_cache_error(
            "a drop-shadow colorize shader key has no selected working format",
        ));
    };
    if keys.layout.program != ShaderProgramKey::DropShadowColorize
        || keys.layout.data_bindings.as_slice()
            != [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::DropShadowParameters,
            ]
        || keys.shader.program != ShaderProgramKey::DropShadowColorize
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || keys.shader.output_format.is_some()
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || !is_working_format(working_format)
        || keys.pipeline.target_format != working_format
        || sampled_texture.binding_role != ShaderBindingRoleKey::BlurredSourceAlpha
        || sampled_texture.source_format != working_format
        || sampler.binding_role != ShaderBindingRoleKey::BlurredSourceAlpha
        || sampler.source_format != working_format
        || sampler.filter != ShaderSamplingFilterKey::Linear
        || sampler.edge != ShaderSamplingEdgeKey::TransparentBlack
        || sampler.resolved_mask_sampling.is_some()
    {
        return Err(drop_shadow_cache_error(
            "drop-shadow colorize keys disagree across source, layout, shader, or target",
        ));
    }
    Ok(DropShadowColorizePassDescription { working_format })
}

const fn blur_program_facts(
    program: ShaderProgramKey,
) -> Option<(BlurAxis, BlurInput, ShaderSamplingEdgeKey)> {
    match program {
        ShaderProgramKey::BlurHorizontal { source_alpha, edge } => Some((
            BlurAxis::Horizontal,
            if source_alpha {
                BlurInput::SourceAlpha
            } else {
                BlurInput::Rgba
            },
            edge,
        )),
        ShaderProgramKey::BlurVertical { source_alpha, edge } => Some((
            BlurAxis::Vertical,
            if source_alpha {
                BlurInput::SourceAlpha
            } else {
                BlurInput::Rgba
            },
            edge,
        )),
        _ => None,
    }
}

#[cfg(test)]
pub(super) const fn is_blur_program(program: ShaderProgramKey) -> bool {
    blur_program_facts(program).is_some()
}

pub(super) fn validate_composite_pass_keys(
    keys: CompositePassKeyRefs<'_>,
) -> Result<CompositePassDescription> {
    let ShaderProgramKey::Composite(ShaderCompositeKey::Layer {
        path,
        has_clip_coverage,
        has_alpha_mask,
    }) = keys.layout.program
    else {
        return Err(core_pass_cache_error(
            "a non-layer program reached composition composite realization",
        ));
    };
    let Some(working_format) = keys.shader.working_format else {
        return Err(core_pass_cache_error(
            "composition composite shader key has no selected working format",
        ));
    };
    if !is_working_format(working_format)
        || keys.layout.data_bindings.as_slice()
            != [
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::CompositeParameters,
            ]
        || keys.shader.program != keys.layout.program
        || &keys.shader.layout != keys.layout
        || keys.shader.samplers.as_slice() != keys.samplers
        || keys.shader.output_format.is_some()
        || &keys.pipeline.shader != keys.shader
        || &keys.pipeline.layout != keys.layout
        || keys.pipeline.samplers.as_slice() != keys.samplers
        || keys.pipeline.target_format != working_format
    {
        return Err(core_pass_cache_error(
            "composition composite keys disagree across layout, shader, format, or pipeline phases",
        ));
    }

    let [source_sampler] = keys.samplers else {
        return Err(core_pass_cache_error(
            "composition composite pass must bind exactly one source sampler",
        ));
    };
    if source_sampler.binding_role != ShaderBindingRoleKey::CompositeSource
        || source_sampler.source_format != working_format
        || source_sampler.filter != ShaderSamplingFilterKey::Linear
        || source_sampler.edge != ShaderSamplingEdgeKey::TransparentBlack
        || source_sampler.resolved_mask_sampling.is_some()
    {
        return Err(core_pass_cache_error(
            "composition composite source sampler semantics are not exact",
        ));
    }

    let mut expected_textures = Vec::with_capacity(4);
    if path == ShaderCompositePathKey::DestinationSampling {
        expected_textures.push(SampledTextureLayoutKey {
            binding_role: ShaderBindingRoleKey::CompositeParent,
            source_format: working_format,
        });
    }
    expected_textures.push(SampledTextureLayoutKey {
        binding_role: ShaderBindingRoleKey::CompositeSource,
        source_format: working_format,
    });
    if has_clip_coverage {
        expected_textures.push(SampledTextureLayoutKey {
            binding_role: ShaderBindingRoleKey::ClipCoverage,
            source_format: ShaderTextureFormatKey::ClipCoverageRgba8Unorm,
        });
    }
    if has_alpha_mask {
        expected_textures.push(SampledTextureLayoutKey {
            binding_role: ShaderBindingRoleKey::AlphaMask,
            source_format: ShaderTextureFormatKey::ResolvedMaskRgba8Unorm,
        });
    }
    if keys.layout.sampled_textures != expected_textures {
        return Err(core_pass_cache_error(
            "composition composite layout contains an absent or missing semantic texture",
        ));
    }

    Ok(CompositePassDescription {
        path,
        has_clip_coverage,
        has_alpha_mask,
        target_format: working_format,
    })
}

const fn is_working_format(format: ShaderTextureFormatKey) -> bool {
    matches!(
        format,
        ShaderTextureFormatKey::WorkingHighPrecisionRgba16Float
            | ShaderTextureFormatKey::WorkingReducedPrecisionRgba8Unorm
    )
}

const fn is_output_format(format: ShaderTextureFormatKey) -> bool {
    matches!(
        format,
        ShaderTextureFormatKey::OutputRgba8Unorm | ShaderTextureFormatKey::OutputBgra8Unorm
    )
}

pub(super) fn core_pass_cache_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

pub(super) fn color_filter_cache_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

pub(super) fn copy_backdrop_cache_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

pub(super) fn blur_cache_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

pub(super) fn drop_shadow_cache_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}
