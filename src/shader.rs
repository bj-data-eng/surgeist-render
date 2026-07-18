use super::gpu_transaction::GpuOperationTransaction;
use super::{
    Color, Error, Format, Result, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    image::ResolvedMaskUploadKey,
    resource::WorkingFormat,
    texture::{TextureDescriptor, TextureUsageIntent},
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

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RectShaderPassKind {
    ClearFill,
    IdentityCopy,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectPassBounds {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
impl RectPassBounds {
    pub(crate) fn try_new(
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        source: TextureDescriptor,
        destination: TextureDescriptor,
    ) -> Result<Self> {
        if width == 0 {
            return Err(Error::invalid_value(
                "rect shader pass width",
                width,
                "must be greater than 0 device pixels",
            ));
        }
        if height == 0 {
            return Err(Error::invalid_value(
                "rect shader pass height",
                height,
                "must be greater than 0 device pixels",
            ));
        }
        Self::validate_texture_fit("source", x, y, width, height, source)?;
        Self::validate_texture_fit("destination", x, y, width, height, destination)?;
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    #[must_use]
    pub(crate) const fn x(self) -> u32 {
        self.x
    }

    #[must_use]
    pub(crate) const fn y(self) -> u32 {
        self.y
    }

    #[must_use]
    pub(crate) const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub(crate) const fn height(self) -> u32 {
        self.height
    }

    fn validate_texture_fit(
        texture_role: &'static str,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        descriptor: TextureDescriptor,
    ) -> Result<()> {
        let max_x = x.checked_add(width).ok_or_else(|| {
            Error::invalid_value(
                format!("rect shader pass {texture_role} x extent"),
                format!("{x}+{width}"),
                "must fit in u32 device pixels",
            )
        })?;
        let max_y = y.checked_add(height).ok_or_else(|| {
            Error::invalid_value(
                format!("rect shader pass {texture_role} y extent"),
                format!("{y}+{height}"),
                "must fit in u32 device pixels",
            )
        })?;
        let size = descriptor.physical_size();
        if max_x > size.width() {
            return Err(Error::invalid_value(
                format!("rect shader pass {texture_role} x extent"),
                max_x,
                "must fit inside the texture width",
            ));
        }
        if max_y > size.height() {
            return Err(Error::invalid_value(
                format!("rect shader pass {texture_role} y extent"),
                max_y,
                "must fit inside the texture height",
            ));
        }
        Ok(())
    }

    fn covers(self, descriptor: TextureDescriptor) -> bool {
        self.x == 0
            && self.y == 0
            && self.width == descriptor.physical_size().width()
            && self.height == descriptor.physical_size().height()
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectShaderPassDescriptor {
    source_label: &'static str,
    destination_label: &'static str,
    source: TextureDescriptor,
    destination: TextureDescriptor,
    bounds: RectPassBounds,
    kind: RectShaderPassKind,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
impl RectShaderPassDescriptor {
    pub(crate) fn try_new(
        source_label: &'static str,
        destination_label: &'static str,
        source: TextureDescriptor,
        destination: TextureDescriptor,
        bounds: RectPassBounds,
        kind: RectShaderPassKind,
    ) -> Result<Self> {
        validate_label("rect shader pass source texture", source_label)?;
        validate_label("rect shader pass destination texture", destination_label)?;
        RectPassBounds::validate_texture_fit(
            "source",
            bounds.x(),
            bounds.y(),
            bounds.width(),
            bounds.height(),
            source,
        )?;
        RectPassBounds::validate_texture_fit(
            "destination",
            bounds.x(),
            bounds.y(),
            bounds.width(),
            bounds.height(),
            destination,
        )?;
        if kind == RectShaderPassKind::ClearFill && !bounds.covers(destination) {
            return Err(Error::invalid_value(
                "rect shader clear/fill bounds",
                format!(
                    "{},{},{}x{} over {}x{}",
                    bounds.x(),
                    bounds.y(),
                    bounds.width(),
                    bounds.height(),
                    destination.physical_size().width(),
                    destination.physical_size().height()
                ),
                "must cover the full destination texture",
            ));
        }
        if kind == RectShaderPassKind::IdentityCopy && source.format() != destination.format() {
            return Err(Error::invalid_value(
                "rect shader pass texture formats",
                format!("{:?}->{:?}", source.format(), destination.format()),
                "must match for identity copy passes",
            ));
        }
        Ok(Self {
            source_label,
            destination_label,
            source,
            destination,
            bounds,
            kind,
        })
    }

    #[must_use]
    pub(crate) const fn source_label(self) -> &'static str {
        self.source_label
    }

    #[must_use]
    pub(crate) const fn destination_label(self) -> &'static str {
        self.destination_label
    }

    #[must_use]
    pub(crate) const fn source(self) -> TextureDescriptor {
        self.source
    }

    #[must_use]
    pub(crate) const fn destination(self) -> TextureDescriptor {
        self.destination
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> RectPassBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn kind(self) -> RectShaderPassKind {
        self.kind
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectShaderPipelineKey {
    kind: RectShaderPassKind,
    source_format: Format,
    destination_format: Format,
    source_intent: TextureUsageIntent,
    destination_intent: TextureUsageIntent,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
impl RectShaderPipelineKey {
    #[must_use]
    pub(crate) const fn from_descriptor(descriptor: RectShaderPassDescriptor) -> Self {
        Self {
            kind: descriptor.kind,
            source_format: descriptor.source.format(),
            destination_format: descriptor.destination.format(),
            source_intent: descriptor.source.intent(),
            destination_intent: descriptor.destination.intent(),
        }
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
#[derive(Clone, Copy)]
pub(crate) struct RectShaderPassGpuContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    source_view: &'a wgpu::TextureView,
    destination_view: &'a wgpu::TextureView,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
impl<'a> RectShaderPassGpuContext<'a> {
    #[must_use]
    pub(crate) const fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        source_view: &'a wgpu::TextureView,
        destination_view: &'a wgpu::TextureView,
    ) -> Self {
        Self {
            device,
            queue,
            source_view,
            destination_view,
        }
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
pub(crate) enum RectShaderPassExecution<'a> {
    ContractOnly,
    Gpu(RectShaderPassGpuExecution<'a>),
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
impl<'a> RectShaderPassExecution<'a> {
    #[must_use]
    pub(crate) const fn contract_only() -> Self {
        Self::ContractOnly
    }

    #[must_use]
    pub(crate) const fn gpu(
        context: RectShaderPassGpuContext<'a>,
        transaction: GpuOperationTransaction,
    ) -> Self {
        Self::Gpu(RectShaderPassGpuExecution {
            context,
            transaction,
        })
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
pub(crate) struct RectShaderPassGpuExecution<'a> {
    context: RectShaderPassGpuContext<'a>,
    transaction: GpuOperationTransaction,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
pub(crate) async fn encode_clear_fill_pass(
    execution: RectShaderPassExecution<'_>,
    descriptor: RectShaderPassDescriptor,
    color: Color,
) -> Result<()> {
    if descriptor.kind() != RectShaderPassKind::ClearFill {
        return Err(Error::invalid_value(
            "rect shader pass kind",
            format!("{:?}", descriptor.kind()),
            "must be ClearFill for clear/fill encoding",
        ));
    }
    let RectShaderPassExecution::Gpu(RectShaderPassGpuExecution {
        context,
        transaction,
    }) = execution
    else {
        return Err(Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            "rect/fullscreen shader pass requires an available wgpu device context",
        ));
    };
    let _source_view = context.source_view;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist rect shader pass clear/fill"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(descriptor.destination_label()),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: context.destination_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(color.r()),
                        g: f64::from(color.g()),
                        b: f64::from(color.b()),
                        a: f64::from(color.a()),
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }
    transaction
        .submit_command_buffer(
            context.queue,
            encoder.finish(),
            RuntimeOperation::SurfaceRendering,
        )
        .await
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "T6 removes the existing rect shader probe")
)]
fn validate_label(name: &'static str, value: &'static str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::invalid_value(name, value, "must not be empty"));
    }
    Ok(())
}
