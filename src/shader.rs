use std::{borrow::Cow, collections::HashMap, sync::Arc};

use super::{
    Error, Format, Result,
    image::{Extend, ImageQuality},
    layer::BlendMode,
    pass::{
        RuntimeColorClampBoundary, RuntimeColorOperation, RuntimeColorOperationKind,
        RuntimeLayerCompositeParameters, RuntimeSpatialDescriptor,
    },
    resource::WorkingFormat,
};

const CANONICALIZE_CAPTURE_WGSL: &str = include_str!("shaders/canonicalize_capture.wgsl");
const SPAN_SOURCE_OVER_WGSL: &str = include_str!("shaders/span_source_over.wgsl");
const PRESENT_WGSL: &str = include_str!("shaders/present.wgsl");
const LAYER_COMPOSITE_WGSL: &str = include_str!("shaders/layer_composite.wgsl");
const COLOR_FILTER_WGSL: &str = include_str!("shaders/color_filter.wgsl");
const BLUR_WGSL: &str = include_str!("shaders/blur.wgsl");
const DROP_SHADOW_WGSL: &str = include_str!("shaders/drop_shadow.wgsl");
const COPY_BACKDROP_WGSL: &str = include_str!("shaders/copy_backdrop.wgsl");

const COLOR_FILTER_OPERATION_HEADER_BYTE_LEN: u64 = 16;
const COLOR_FILTER_OPERATION_RECORD_BYTE_LEN: u64 = 32;

/// Exact 32-byte WGSL drop-shadow parameter block.
///
/// Bytes `0..8` retain the continuous logical offset, bytes `8..16` are zero
/// alignment, and bytes `16..32` contain the finite solid premultiplied color.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DropShadowParameterBytes([u8; 32]);

impl DropShadowParameterBytes {
    pub(crate) fn try_new(offset: super::Point, color: super::Color) -> Result<Self> {
        let offset_x = narrow_drop_shadow_scalar("drop shadow offset x", offset.x())?;
        let offset_y = narrow_drop_shadow_scalar("drop shadow offset y", offset.y())?;
        let channels = [color.r(), color.g(), color.b(), color.a()];
        if channels
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(drop_shadow_parameter_error(
                "drop shadow solid color",
                "must contain finite unit-interval channels",
            ));
        }
        let premultiplied = [
            color.r() * color.a(),
            color.g() * color.a(),
            color.b() * color.a(),
            color.a(),
        ];
        let mut bytes = [0_u8; 32];
        bytes[0..4].copy_from_slice(&offset_x.to_le_bytes());
        bytes[4..8].copy_from_slice(&offset_y.to_le_bytes());
        for (index, value) in premultiplied.into_iter().enumerate() {
            bytes[16 + index * 4..20 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self(bytes))
    }

    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[cfg(test)]
pub(crate) fn drop_shadow_parameter_bytes_for_test(
    offset: super::Point,
    color: super::Color,
) -> Result<[u8; 32]> {
    DropShadowParameterBytes::try_new(offset, color).map(|bytes| bytes.0)
}

fn narrow_drop_shadow_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(drop_shadow_parameter_error(
            field,
            "must remain finite after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn drop_shadow_parameter_error(field: &'static str, invariant: &'static str) -> Error {
    Error::invalid_value(field, "runtime drop shadow", invariant)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBufferLimits {
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
}

impl ColorFilterOperationBufferLimits {
    #[must_use]
    pub(crate) fn from_device_limits(limits: &wgpu::Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn for_test(
        max_buffer_size: u64,
        max_storage_buffer_binding_size: u64,
    ) -> Self {
        Self {
            max_buffer_size,
            max_storage_buffer_binding_size,
        }
    }
}

/// Exact checked storage-buffer bytes for one ordered color-filter run.
///
/// Bytes `0..16` are the operation count followed by three zero `u32` pads.
/// Every following 32-byte record stores tag, zero flag, exponent, one zero
/// alignment word, a four-scalar finite payload, and no other state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBytes {
    bytes: Vec<u8>,
}

impl ColorFilterOperationBytes {
    pub(crate) fn try_from_runtime_operations_with_limits(
        operations: &[RuntimeColorOperation],
        limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        let operation_count = u32::try_from(operations.len())
            .map_err(|_| color_filter_operation_count_error(operations.len()))?;
        let byte_len = checked_color_filter_operation_byte_len(operation_count, limits)?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            color_filter_operation_byte_len_error(
                byte_len,
                "must fit the host address space after device-limit validation",
            )
        })?;

        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|_| {
            color_filter_operation_byte_len_error(
                byte_len,
                "must fit available host serialization memory",
            )
        })?;
        bytes.resize(byte_len, 0);
        bytes[0..4].copy_from_slice(&operation_count.to_le_bytes());

        for (index, operation) in operations.iter().copied().enumerate() {
            let record = serialize_color_filter_operation(operation)?;
            let offset = 16 + index * 32;
            bytes[offset..offset + 32].copy_from_slice(&record);
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[cfg(test)]
    pub(crate) fn try_from_runtime_operations_for_test(
        operations: &[RuntimeColorOperation],
        limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        Self::try_from_runtime_operations_with_limits(operations, limits)
    }
}

fn checked_color_filter_operation_byte_len(
    operation_count: u32,
    limits: ColorFilterOperationBufferLimits,
) -> Result<u64> {
    let records = COLOR_FILTER_OPERATION_RECORD_BYTE_LEN
        .checked_mul(u64::from(operation_count))
        .ok_or_else(|| {
            color_filter_operation_byte_len_error(
                operation_count,
                "must produce a checked u64 record byte length",
            )
        })?;
    let byte_len = COLOR_FILTER_OPERATION_HEADER_BYTE_LEN
        .checked_add(records)
        .ok_or_else(|| {
            color_filter_operation_byte_len_error(
                operation_count,
                "must produce a checked u64 header-plus-record byte length",
            )
        })?;
    if byte_len > limits.max_buffer_size || byte_len > limits.max_storage_buffer_binding_size {
        return Err(color_filter_operation_byte_len_error(
            byte_len,
            "must not exceed max_buffer_size or max_storage_buffer_binding_size",
        ));
    }
    Ok(byte_len)
}

fn serialize_color_filter_operation(operation: RuntimeColorOperation) -> Result<[u8; 32]> {
    if operation.clamp_boundary()
        != RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
    {
        return Err(Error::invalid_value(
            "color filter operation clamp boundary",
            "unsupported runtime boundary",
            "must clamp straight RGBA and premultiply after every operation",
        ));
    }

    let (tag, zero, exponent, payload) = match operation.operation() {
        RuntimeColorOperationKind::Brightness(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (0_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Contrast(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (1_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Grayscale(amount) => (
            2_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::HueRotate(angle) => {
            let sine = checked_color_filter_payload(angle.sine())?;
            let cosine = checked_color_filter_payload(angle.cosine())?;
            (3_u32, 0, 0, [sine, cosine, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Invert(amount) => (
            4_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::Opacity(amount) => (
            5_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::Saturate(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (6_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Sepia(amount) => (
            7_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
    };

    let mut bytes = [0_u8; 32];
    bytes[0..4].copy_from_slice(&tag.to_le_bytes());
    bytes[4..8].copy_from_slice(&zero.to_le_bytes());
    bytes[8..12].copy_from_slice(&exponent.to_le_bytes());
    for (index, value) in payload.into_iter().enumerate() {
        let offset = 16 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn checked_runtime_amount_parts(
    amount: super::filter::RuntimeFilterAmount,
) -> Result<(u32, i32, f32)> {
    if amount.zero() {
        if amount.mantissa() != 0.0 || amount.exponent() != 0 {
            return Err(color_filter_operation_scalar_error(
                "zero amount must carry zero mantissa and exponent",
            ));
        }
        return Ok((1, 0, 0.0));
    }
    let mantissa = checked_color_filter_payload(amount.mantissa())?;
    if !(0.5..1.0).contains(&mantissa) {
        return Err(color_filter_operation_scalar_error(
            "nonzero amount mantissa must be normalized to [0.5, 1)",
        ));
    }
    Ok((0, amount.exponent(), mantissa))
}

fn checked_runtime_unit(value: f32) -> Result<f32> {
    let value = checked_color_filter_payload(value)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(color_filter_operation_scalar_error(
            "unit amount must remain in the inclusive unit interval",
        ));
    }
    Ok(value)
}

fn checked_color_filter_payload(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(color_filter_operation_scalar_error(
            "every scalar payload must be finite",
        ));
    }
    Ok(value)
}

fn color_filter_operation_count_error(value: impl std::fmt::Display) -> Error {
    Error::invalid_value(
        "color filter operation count",
        value,
        "must fit in u32 before operation-buffer byte calculation",
    )
}

fn color_filter_operation_byte_len_error(
    value: impl std::fmt::Display,
    invariant: &'static str,
) -> Error {
    Error::invalid_value(
        "color filter operation buffer byte length",
        value,
        invariant,
    )
}

fn color_filter_operation_scalar_error(invariant: &'static str) -> Error {
    Error::invalid_value(
        "color filter operation scalar payload",
        "runtime operation",
        invariant,
    )
}

#[cfg(test)]
pub(crate) fn color_filter_operation_byte_len_for_test(
    operation_count: u64,
    limits: ColorFilterOperationBufferLimits,
) -> Result<u64> {
    let operation_count = u32::try_from(operation_count)
        .map_err(|_| color_filter_operation_count_error(operation_count))?;
    checked_color_filter_operation_byte_len(operation_count, limits)
}

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

    const fn parameter_code(self) -> u32 {
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

    const fn parameter_code(self) -> u32 {
        match self {
            Self::Pad => 0,
            Self::Repeat => 1,
            Self::Reflect => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShaderMaskSamplingKey {
    quality: ShaderMaskQualityKey,
    extend: ShaderMaskExtendKey,
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

const C08_EXCLUDED_C09_PARAMETER_BINDINGS: [ShaderDataBindingKey; 2] = [
    ShaderDataBindingKey::CompositeParameters,
    ShaderDataBindingKey::PresentParameters,
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SamplerKey {
    binding_role: ShaderBindingRoleKey,
    source_format: ShaderTextureFormatKey,
    filter: ShaderSamplingFilterKey,
    edge: ShaderSamplingEdgeKey,
    resolved_mask_sampling: Option<ShaderMaskSamplingKey>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08Program {
    CanonicalizeCapture,
    SpanSourceOver,
    DropShadowMerge,
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct C08PassDescription {
    program: C08Program,
    target_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct C08PassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompositePassDescription {
    path: ShaderCompositePathKey,
    has_clip_coverage: bool,
    has_alpha_mask: bool,
    target_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct CompositePassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ColorFilterPassDescription {
    working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct ColorFilterPassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CopyBackdropPassDescription {
    working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct CopyBackdropPassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlurPassDescription {
    axis: BlurAxis,
    input: BlurInput,
    edge: ShaderSamplingEdgeKey,
    working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct BlurPassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DropShadowColorizePassDescription {
    working_format: ShaderTextureFormatKey,
}

#[derive(Clone, Copy)]
struct DropShadowColorizePassKeyRefs<'a> {
    samplers: &'a [SamplerKey],
    layout: &'a BindGroupLayoutKey,
    shader: &'a ShaderModuleKey,
    pipeline: &'a RenderPipelineKey,
}

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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_c08_bind_group_layout(device, description.program),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(c08_shader_label(description.program)),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(c08_shader_source(
                        description.program,
                    ))),
                }),
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
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(c08_pipeline_layout_label(description.program)),
                bind_group_layouts: &[Some(layout_handle)],
                immediate_size: 0,
            });
            let blend = matches!(
                description.program,
                C08Program::SpanSourceOver | C08Program::DropShadowMerge
            )
            .then_some(span_source_over_blend());
            let target = wgpu::ColorTargetState {
                format: texture_format(description.target_format)?,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let created = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(c08_pipeline_label(description.program)),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader_handle,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_handle,
                    entry_point: Some(fragment_entry),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
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
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Surgeist C12 backdrop-copy shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COPY_BACKDROP_WGSL)),
                }),
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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
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
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Surgeist C10 color-filter shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COLOR_FILTER_WGSL)),
                }),
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
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surgeist C10 color-filter pipeline layout"),
                bind_group_layouts: &[Some(layout_handle)],
                immediate_size: 0,
            });
            let target = wgpu::ColorTargetState {
                format: texture_format(description.working_format)?,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let created = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Surgeist C10 color-filter pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader_handle,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_handle,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_blur_bind_group_layout(device, description.edge),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Surgeist checked Gaussian blur shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLUR_WGSL)),
                }),
            );
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
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surgeist checked Gaussian blur pipeline layout"),
                bind_group_layouts: &[Some(layout_handle)],
                immediate_size: 0,
            });
            let target = wgpu::ColorTargetState {
                format: texture_format(description.working_format)?,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let created = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Surgeist checked Gaussian blur pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader_handle,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_handle,
                    entry_point: Some(blur_fragment_entry(description)),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
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
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Surgeist C11 drop-shadow colorize shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DROP_SHADOW_WGSL)),
                }),
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
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surgeist C11 drop-shadow colorize pipeline layout"),
                bind_group_layouts: &[Some(layout_handle)],
                immediate_size: 0,
            });
            let target = wgpu::ColorTargetState {
                format: texture_format(description.working_format)?,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let created = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Surgeist C11 drop-shadow colorize pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader_handle,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_handle,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
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
                self.samplers.insert(
                    *sampler_key,
                    device.create_sampler(&sampler_descriptor(*sampler_key)),
                );
            }
        }
        if !cache.layouts.contains_key(keys.layout) && !self.layouts.contains_key(keys.layout) {
            self.layouts.insert(
                keys.layout.clone(),
                create_composite_bind_group_layout(device, description),
            );
        }
        if !cache.shaders.contains_key(keys.shader) && !self.shaders.contains_key(keys.shader) {
            self.shaders.insert(
                keys.shader.clone(),
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Surgeist C09 layer-composite shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LAYER_COMPOSITE_WGSL)),
                }),
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
                    c08_cache_error("composite bind-group layout realization was lost")
                })?;
            let shader_handle = self
                .shaders
                .get(keys.shader)
                .or_else(|| cache.shaders.get(keys.shader))
                .ok_or_else(|| c08_cache_error("composite shader-module realization was lost"))?;
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surgeist C09 layer-composite pipeline layout"),
                bind_group_layouts: &[Some(layout_handle)],
                immediate_size: 0,
            });
            let target = wgpu::ColorTargetState {
                format: texture_format(description.target_format)?,
                blend: match description.path {
                    ShaderCompositePathKey::Normal => Some(span_source_over_blend()),
                    ShaderCompositePathKey::DestinationSampling => None,
                },
                write_mask: wgpu::ColorWrites::ALL,
            };
            let fragment_entry =
                fragment_entry_override.unwrap_or_else(|| composite_fragment_entry(description));
            let created = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Surgeist C09 layer-composite pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader_handle,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_handle,
                    entry_point: Some(fragment_entry),
                    targets: &[Some(target)],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
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

fn validate_c08_pass_keys(keys: C08PassKeyRefs<'_>) -> Result<C08PassDescription> {
    let [sampled_texture] = keys.layout.sampled_textures.as_slice() else {
        return Err(c08_cache_error(
            "a C08 pass layout must bind exactly one sampled texture",
        ));
    };
    let [sampler] = keys.samplers else {
        return Err(c08_cache_error(
            "a C08 pass must bind exactly one exact sampler",
        ));
    };
    validate_c08_key_consistency(keys, sampled_texture, sampler)?;
    let (program, expected_role, expected_filter, expected_edge) =
        c08_program_sampling(keys.layout.program)?;
    if sampled_texture.binding_role != expected_role
        || sampler.filter != expected_filter
        || sampler.edge != expected_edge
    {
        return Err(c08_cache_error(
            "C08 sampled texture and sampler semantics are not exact",
        ));
    }

    let Some(working_format) = keys.shader.working_format else {
        return Err(c08_cache_error(
            "C08 shader key has no selected working format",
        ));
    };
    if !is_working_format(working_format) {
        return Err(c08_cache_error(
            "C08 shader key selected a non-working intermediate format",
        ));
    }
    let target_format = c08_target_format(keys, sampled_texture, program, working_format)?;
    Ok(C08PassDescription {
        program,
        target_format,
    })
}

fn validate_c08_key_consistency(
    keys: C08PassKeyRefs<'_>,
    sampled_texture: &SampledTextureLayoutKey,
    sampler: &SamplerKey,
) -> Result<()> {
    if keys.layout.data_bindings.as_slice() != [ShaderDataBindingKey::SpatialUniform]
        || keys
            .layout
            .data_bindings
            .iter()
            .any(|binding| C08_EXCLUDED_C09_PARAMETER_BINDINGS.contains(binding))
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
        return Err(c08_cache_error(
            "C08 pass keys disagree across sampler, layout, shader, or pipeline phases",
        ));
    }
    Ok(())
}

fn c08_program_sampling(
    program: ShaderProgramKey,
) -> Result<(
    C08Program,
    ShaderBindingRoleKey,
    ShaderSamplingFilterKey,
    ShaderSamplingEdgeKey,
)> {
    let sampling = match program {
        ShaderProgramKey::CanonicalizeCapture => (
            C08Program::CanonicalizeCapture,
            ShaderBindingRoleKey::CaptureSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::ClampToExtent,
        ),
        ShaderProgramKey::Composite(ShaderCompositeKey::SpanSourceOver) => (
            C08Program::SpanSourceOver,
            ShaderBindingRoleKey::CompositeSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::TransparentBlack,
        ),
        ShaderProgramKey::Composite(ShaderCompositeKey::DropShadow) => (
            C08Program::DropShadowMerge,
            ShaderBindingRoleKey::CompositeSource,
            ShaderSamplingFilterKey::Linear,
            ShaderSamplingEdgeKey::TransparentBlack,
        ),
        ShaderProgramKey::Present => (
            C08Program::Present,
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
            return Err(c08_cache_error(
                "a later-cycle shader program reached C08 pass realization",
            ));
        }
    };
    Ok(sampling)
}

fn c08_target_format(
    keys: C08PassKeyRefs<'_>,
    sampled_texture: &SampledTextureLayoutKey,
    program: C08Program,
    working_format: ShaderTextureFormatKey,
) -> Result<ShaderTextureFormatKey> {
    let target_format = match program {
        C08Program::CanonicalizeCapture => {
            if sampled_texture.source_format != ShaderTextureFormatKey::VelloCaptureRgba8Unorm
                || keys.shader.output_format.is_some()
                || keys.pipeline.target_format != working_format
            {
                return Err(c08_cache_error(
                    "canonicalization keys changed capture or working formats",
                ));
            }
            working_format
        }
        C08Program::SpanSourceOver | C08Program::DropShadowMerge => {
            if sampled_texture.source_format != working_format
                || keys.shader.output_format.is_some()
                || keys.pipeline.target_format != working_format
            {
                return Err(c08_cache_error(
                    "source-over keys changed source or working target formats",
                ));
            }
            working_format
        }
        C08Program::Present => {
            let Some(output_format) = keys.shader.output_format else {
                return Err(c08_cache_error("present key has no exact output format"));
            };
            if sampled_texture.source_format != working_format
                || !is_output_format(output_format)
                || keys.pipeline.target_format != output_format
            {
                return Err(c08_cache_error(
                    "present keys changed working source or output specialization",
                ));
            }
            output_format
        }
    };
    Ok(target_format)
}

fn validate_color_filter_pass_keys(
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

fn validate_copy_backdrop_pass_keys(
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

fn validate_blur_pass_keys(keys: BlurPassKeyRefs<'_>) -> Result<BlurPassDescription> {
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
    let (axis, input, edge) = blur_program_facts(keys.layout.program)
        .ok_or_else(|| blur_cache_error("a non-blur program reached C11 blur realization"))?;
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

fn validate_drop_shadow_colorize_pass_keys(
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
const fn is_blur_program(program: ShaderProgramKey) -> bool {
    blur_program_facts(program).is_some()
}

fn validate_composite_pass_keys(
    keys: CompositePassKeyRefs<'_>,
) -> Result<CompositePassDescription> {
    let ShaderProgramKey::Composite(ShaderCompositeKey::Layer {
        path,
        has_clip_coverage,
        has_alpha_mask,
    }) = keys.layout.program
    else {
        return Err(c08_cache_error(
            "a non-layer program reached C09 composite realization",
        ));
    };
    let Some(working_format) = keys.shader.working_format else {
        return Err(c08_cache_error(
            "C09 composite shader key has no selected working format",
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
        return Err(c08_cache_error(
            "C09 composite keys disagree across layout, shader, format, or pipeline phases",
        ));
    }

    let [source_sampler] = keys.samplers else {
        return Err(c08_cache_error(
            "C09 composite pass must bind exactly one source sampler",
        ));
    };
    if source_sampler.binding_role != ShaderBindingRoleKey::CompositeSource
        || source_sampler.source_format != working_format
        || source_sampler.filter != ShaderSamplingFilterKey::Linear
        || source_sampler.edge != ShaderSamplingEdgeKey::TransparentBlack
        || source_sampler.resolved_mask_sampling.is_some()
    {
        return Err(c08_cache_error(
            "C09 composite source sampler semantics are not exact",
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
        return Err(c08_cache_error(
            "C09 composite layout contains an absent or missing semantic texture",
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

fn sampler_descriptor(key: SamplerKey) -> wgpu::SamplerDescriptor<'static> {
    let filter = match key.filter {
        ShaderSamplingFilterKey::Nearest => wgpu::FilterMode::Nearest,
        ShaderSamplingFilterKey::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode = match key
        .resolved_mask_sampling
        .map(ShaderMaskSamplingKey::extend)
    {
        None | Some(ShaderMaskExtendKey::Pad) => wgpu::AddressMode::ClampToEdge,
        Some(ShaderMaskExtendKey::Repeat) => wgpu::AddressMode::Repeat,
        Some(ShaderMaskExtendKey::Reflect) => wgpu::AddressMode::MirrorRepeat,
    };
    wgpu::SamplerDescriptor {
        label: Some("Surgeist sampled-image sampler"),
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    }
}

fn create_c08_bind_group_layout(
    device: &wgpu::Device,
    program: C08Program,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(c08_bind_group_layout_label(program)),
        entries: &entries,
    })
}

fn create_composite_bind_group_layout(
    device: &wgpu::Device,
    description: CompositePassDescription,
) -> wgpu::BindGroupLayout {
    let texture_entry = |binding, filterable| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let mut entries = Vec::with_capacity(7);
    entries.push(texture_entry(0, true));
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    if description.path == ShaderCompositePathKey::DestinationSampling {
        entries.push(texture_entry(2, false));
    }
    if description.has_clip_coverage {
        entries.push(texture_entry(3, false));
    }
    if description.has_alpha_mask {
        entries.push(texture_entry(4, false));
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(48),
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(112),
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist C09 layer-composite bindings"),
        entries: &entries,
    })
}

fn create_copy_backdrop_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist C12 backdrop-copy bindings"),
        entries: &entries,
    })
}

fn create_copy_backdrop_pipeline(
    device: &wgpu::Device,
    description: CopyBackdropPassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist C12 backdrop-copy pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.working_format)?,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist C12 backdrop-copy pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

fn create_color_filter_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist C10 color-filter bindings"),
        entries: &entries,
    })
}

fn create_blur_bind_group_layout(
    device: &wgpu::Device,
    edge: ShaderSamplingEdgeKey,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(8),
            },
            count: None,
        },
    ];
    if edge == ShaderSamplingEdgeKey::SemanticBorderMirror {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist checked Gaussian blur bindings"),
        entries: &entries,
    })
}

fn create_drop_shadow_colorize_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(32),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist C11 drop-shadow colorize bindings"),
        entries: &entries,
    })
}

const fn blur_fragment_entry(description: BlurPassDescription) -> &'static str {
    match (description.axis, description.input, description.edge) {
        (BlurAxis::Horizontal, BlurInput::Rgba, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_horizontal_rgba"
        }
        (BlurAxis::Vertical, BlurInput::Rgba, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_vertical_rgba"
        }
        (BlurAxis::Horizontal, BlurInput::SourceAlpha, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_horizontal_source_alpha"
        }
        (BlurAxis::Vertical, BlurInput::SourceAlpha, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_vertical_source_alpha"
        }
        (BlurAxis::Horizontal, BlurInput::Rgba, ShaderSamplingEdgeKey::SemanticBorderMirror) => {
            "fragment_horizontal_rgba_mirror"
        }
        (BlurAxis::Vertical, BlurInput::Rgba, ShaderSamplingEdgeKey::SemanticBorderMirror) => {
            "fragment_vertical_rgba_mirror"
        }
        (
            BlurAxis::Horizontal,
            BlurInput::SourceAlpha,
            ShaderSamplingEdgeKey::SemanticBorderMirror,
        ) => "fragment_horizontal_source_alpha_mirror",
        (
            BlurAxis::Vertical,
            BlurInput::SourceAlpha,
            ShaderSamplingEdgeKey::SemanticBorderMirror,
        ) => "fragment_vertical_source_alpha_mirror",
        (_, _, ShaderSamplingEdgeKey::ClampToExtent) => "invalid_blur_edge_policy",
    }
}

const fn composite_fragment_entry(description: CompositePassDescription) -> &'static str {
    match (
        description.path,
        description.has_clip_coverage,
        description.has_alpha_mask,
    ) {
        (ShaderCompositePathKey::Normal, false, false) => "fragment_normal",
        (ShaderCompositePathKey::Normal, true, false) => "fragment_normal_clip",
        (ShaderCompositePathKey::Normal, false, true) => "fragment_normal_mask",
        (ShaderCompositePathKey::Normal, true, true) => "fragment_normal_clip_mask",
        (ShaderCompositePathKey::DestinationSampling, false, false) => "fragment_destination",
        (ShaderCompositePathKey::DestinationSampling, true, false) => "fragment_destination_clip",
        (ShaderCompositePathKey::DestinationSampling, false, true) => "fragment_destination_mask",
        (ShaderCompositePathKey::DestinationSampling, true, true) => {
            "fragment_destination_clip_mask"
        }
    }
}

const fn c08_shader_source(program: C08Program) -> &'static str {
    match program {
        C08Program::CanonicalizeCapture => CANONICALIZE_CAPTURE_WGSL,
        C08Program::SpanSourceOver | C08Program::DropShadowMerge => SPAN_SOURCE_OVER_WGSL,
        C08Program::Present => PRESENT_WGSL,
    }
}

const fn c08_shader_label(program: C08Program) -> &'static str {
    match program {
        C08Program::CanonicalizeCapture => "Surgeist C08 canonicalize-capture shader",
        C08Program::SpanSourceOver => "Surgeist C08 span source-over shader",
        C08Program::DropShadowMerge => "Surgeist C11 drop-shadow merge shader",
        C08Program::Present => "Surgeist C08 present shader",
    }
}

const fn c08_bind_group_layout_label(program: C08Program) -> &'static str {
    match program {
        C08Program::CanonicalizeCapture => "Surgeist C08 canonicalize-capture bindings",
        C08Program::SpanSourceOver => "Surgeist C08 span source-over bindings",
        C08Program::DropShadowMerge => "Surgeist C11 drop-shadow merge bindings",
        C08Program::Present => "Surgeist C08 present bindings",
    }
}

const fn c08_pipeline_layout_label(program: C08Program) -> &'static str {
    match program {
        C08Program::CanonicalizeCapture => "Surgeist C08 canonicalize-capture pipeline layout",
        C08Program::SpanSourceOver => "Surgeist C08 span source-over pipeline layout",
        C08Program::DropShadowMerge => "Surgeist C11 drop-shadow merge pipeline layout",
        C08Program::Present => "Surgeist C08 present pipeline layout",
    }
}

const fn c08_pipeline_label(program: C08Program) -> &'static str {
    match program {
        C08Program::CanonicalizeCapture => "Surgeist C08 canonicalize-capture pipeline",
        C08Program::SpanSourceOver => "Surgeist C08 span source-over pipeline",
        C08Program::DropShadowMerge => "Surgeist C11 drop-shadow merge pipeline",
        C08Program::Present => "Surgeist C08 present pipeline",
    }
}

const fn span_source_over_blend() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

const fn texture_format(format: ShaderTextureFormatKey) -> Result<wgpu::TextureFormat> {
    match format {
        ShaderTextureFormatKey::VelloCaptureRgba8Unorm
        | ShaderTextureFormatKey::ClipCoverageRgba8Unorm
        | ShaderTextureFormatKey::WorkingReducedPrecisionRgba8Unorm
        | ShaderTextureFormatKey::ResolvedMaskRgba8Unorm
        | ShaderTextureFormatKey::OutputRgba8Unorm => Ok(wgpu::TextureFormat::Rgba8Unorm),
        ShaderTextureFormatKey::WorkingHighPrecisionRgba16Float => {
            Ok(wgpu::TextureFormat::Rgba16Float)
        }
        ShaderTextureFormatKey::OutputBgra8Unorm => Ok(wgpu::TextureFormat::Bgra8Unorm),
    }
}

fn c08_cache_error(message: &'static str) -> Error {
    Error::new(super::BackendErrorCode::RenderFailed, message)
}

fn color_filter_cache_error(message: &'static str) -> Error {
    Error::new(super::BackendErrorCode::RenderFailed, message)
}

fn copy_backdrop_cache_error(message: &'static str) -> Error {
    Error::new(super::BackendErrorCode::RenderFailed, message)
}

fn blur_cache_error(message: &'static str) -> Error {
    Error::new(super::BackendErrorCode::RenderFailed, message)
}

fn drop_shadow_cache_error(message: &'static str) -> Error {
    Error::new(super::BackendErrorCode::RenderFailed, message)
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
    pub(crate) const fn as_bytes(&self) -> &[u8; 48] {
        &self.0
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn into_bytes_for_test(self) -> [u8; 48] {
        self.0
    }
}

/// Exact semantic backdrop mirror rectangle in logical coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BlurEdgeParameterBytes([u8; 16]);

impl BlurEdgeParameterBytes {
    pub(crate) fn try_from_semantic_bounds(bounds: super::Rect) -> Result<Self> {
        let minimum_x = narrow_spatial_scalar("blur semantic mirror minimum x", bounds.x())?;
        let minimum_y = narrow_spatial_scalar("blur semantic mirror minimum y", bounds.y())?;
        let maximum_x = narrow_spatial_scalar(
            "blur semantic mirror maximum x",
            bounds.x() + bounds.width(),
        )?;
        let maximum_y = narrow_spatial_scalar(
            "blur semantic mirror maximum y",
            bounds.y() + bounds.height(),
        )?;
        if maximum_x <= minimum_x || maximum_y <= minimum_y {
            return Err(Error::invalid_value(
                "blur semantic mirror bounds",
                format!("{bounds:?}"),
                "must narrow to a finite positive logical rectangle",
            ));
        }
        let mut bytes = [0_u8; 16];
        for (index, value) in [minimum_x, minimum_y, maximum_x, maximum_y]
            .into_iter()
            .enumerate()
        {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self(bytes))
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Exact 112-byte WGSL composite parameter block.
///
/// The byte ranges are fixed as follows: affine linear coefficients `0..16`,
/// affine translation plus zero alignment bytes `16..32`, mask rectangle
/// `32..48`, image dimensions plus zero alignment bytes `48..64`, normalized
/// texel-center facts `64..80`, opacity/blend/quality/extend `80..96`, and
/// exact clip/mask presence plus zero alignment bytes `96..112`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompositeParameterBytes([u8; 112]);

#[derive(Clone, Copy, Debug, PartialEq)]
struct CompositeMaskParameterFacts {
    bounds: [f64; 4],
    dimensions: [u32; 2],
    texel_center_facts: [f64; 4],
    sampling: ShaderMaskSamplingKey,
}

impl CompositeParameterBytes {
    pub(crate) fn try_from_runtime_layer(
        parameters: &RuntimeLayerCompositeParameters,
    ) -> Result<Self> {
        let mask = parameters.alpha_mask().map(|mask| {
            let bounds = mask.bounds();
            let dimensions = mask.image_dimensions();
            let texel_centers = mask.texel_center_facts();
            let [half_x, half_y] = texel_centers.half_texel_normalized();
            let [texel_x, texel_y] = texel_centers.texel_size_normalized();
            CompositeMaskParameterFacts {
                bounds: [bounds.x(), bounds.y(), bounds.width(), bounds.height()],
                dimensions: [dimensions.width(), dimensions.height()],
                texel_center_facts: [half_x, half_y, texel_x, texel_y],
                sampling: mask.sampling(),
            }
        });
        Self::try_from_facts(
            parameters.destination_to_layer_local().affine().as_array(),
            parameters.opacity(),
            parameters.blend(),
            parameters.has_clip(),
            mask,
        )
    }

    fn try_from_facts(
        affine: [f64; 6],
        opacity: f32,
        blend: BlendMode,
        has_clip: bool,
        mask: Option<CompositeMaskParameterFacts>,
    ) -> Result<Self> {
        let [a, b, c, d, e, f] = affine;
        let affine = [
            narrow_composite_scalar("composite affine coefficient a", a)?,
            narrow_composite_scalar("composite affine coefficient b", b)?,
            narrow_composite_scalar("composite affine coefficient c", c)?,
            narrow_composite_scalar("composite affine coefficient d", d)?,
            narrow_composite_scalar("composite affine translation x", e)?,
            narrow_composite_scalar("composite affine translation y", f)?,
        ];
        validate_narrowed_composite_affine(affine)?;

        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(Error::invalid_value(
                "composite opacity",
                opacity,
                "must be finite and clamped to the inclusive unit interval",
            ));
        }

        let mut bytes = [0_u8; 112];
        write_f32(&mut bytes, 0, affine[0]);
        write_f32(&mut bytes, 4, affine[1]);
        write_f32(&mut bytes, 8, affine[2]);
        write_f32(&mut bytes, 12, affine[3]);
        write_f32(&mut bytes, 16, affine[4]);
        write_f32(&mut bytes, 20, affine[5]);

        if let Some(mask) = mask {
            let bounds = [
                narrow_composite_scalar("composite mask bounds x", mask.bounds[0])?,
                narrow_composite_scalar("composite mask bounds y", mask.bounds[1])?,
                narrow_positive_composite_scalar("composite mask bounds width", mask.bounds[2])?,
                narrow_positive_composite_scalar("composite mask bounds height", mask.bounds[3])?,
            ];
            for (index, value) in bounds.into_iter().enumerate() {
                write_f32(&mut bytes, 32 + index * 4, value);
            }

            let [width, height] = mask.dimensions;
            if width == 0 || height == 0 {
                return Err(Error::invalid_value(
                    "composite mask image dimensions",
                    format!("{width}x{height}"),
                    "must be positive before parameter serialization",
                ));
            }
            write_u32(&mut bytes, 48, width);
            write_u32(&mut bytes, 52, height);

            for (index, value) in mask.texel_center_facts.into_iter().enumerate() {
                write_f32(
                    &mut bytes,
                    64 + index * 4,
                    narrow_positive_composite_scalar("composite mask texel fact", value)?,
                );
            }

            let sampling = mask.sampling;
            write_u32(&mut bytes, 88, sampling.quality().parameter_code());
            write_u32(&mut bytes, 92, sampling.extend().parameter_code());
            write_u32(&mut bytes, 100, 1);
        }

        write_f32(&mut bytes, 80, opacity);
        write_u32(&mut bytes, 84, blend_parameter_code(blend));
        write_u32(&mut bytes, 96, u32::from(has_clip));
        Ok(Self(bytes))
    }

    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 112] {
        &self.0
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn into_bytes_for_test(self) -> [u8; 112] {
        self.0
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) struct CompositeParameterGpuVectorFactsForTest {
    pub(crate) layer_point: [f64; 2],
    pub(crate) mask_bounds: [f64; 4],
    pub(crate) mask_dimensions: [u32; 2],
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) has_clip: bool,
    pub(crate) has_mask: bool,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn composite_parameter_bytes_for_gpu_vector_for_test(
    facts: CompositeParameterGpuVectorFactsForTest,
) -> Result<[u8; 112]> {
    if !facts.opacity.is_finite() {
        return Err(Error::invalid_value(
            "composite opacity",
            facts.opacity,
            "must be finite before clamping",
        ));
    }
    let mask = facts.has_mask.then(|| {
        let [width, height] = facts.mask_dimensions;
        let texel_x = 1.0 / f64::from(width);
        let texel_y = 1.0 / f64::from(height);
        CompositeMaskParameterFacts {
            bounds: facts.mask_bounds,
            dimensions: facts.mask_dimensions,
            texel_center_facts: [texel_x * 0.5, texel_y * 0.5, texel_x, texel_y],
            sampling: ShaderMaskSamplingKey::new(facts.quality, facts.extend),
        }
    });
    CompositeParameterBytes::try_from_facts(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            facts.layer_point[0] - 0.5,
            facts.layer_point[1] - 0.5,
        ],
        facts.opacity.clamp(0.0, 1.0),
        facts.blend,
        facts.has_clip,
        mask,
    )
    .map(CompositeParameterBytes::into_bytes_for_test)
}

fn validate_narrowed_composite_affine(affine: [f32; 6]) -> Result<()> {
    let scale = affine[0]
        .abs()
        .max(affine[1].abs())
        .max(affine[2].abs())
        .max(affine[3].abs());
    if scale == 0.0 {
        return Err(Error::invalid_value(
            "composite affine mapping",
            "zero linear transform",
            "must remain non-singular after f64-to-f32 narrowing",
        ));
    }
    let a = affine[0] / scale;
    let b = affine[1] / scale;
    let c = affine[2] / scale;
    let d = affine[3] / scale;
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(Error::invalid_value(
            "composite affine mapping",
            determinant,
            "must remain finite and non-singular after f64-to-f32 narrowing",
        ));
    }
    Ok(())
}

const fn blend_parameter_code(blend: BlendMode) -> u32 {
    match blend {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Screen => 2,
        BlendMode::Overlay => 3,
        BlendMode::Darken => 4,
        BlendMode::Lighten => 5,
        BlendMode::Plus => 6,
    }
}

fn write_f32(bytes: &mut [u8; 112], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8; 112], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn narrow_composite_scalar(field: &'static str, value: f64) -> Result<f32> {
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

fn narrow_positive_composite_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = narrow_composite_scalar(field, value)?;
    if narrowed <= 0.0 {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain strictly positive after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
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
