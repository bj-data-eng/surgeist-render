use crate::{
    BackendErrorCode, Color, Error, Format, PhysicalSize, Point, Rect, Result, Transform,
    command::{RenderClip, RenderCommands},
    filter::{RuntimeFilterAmount, RuntimeFilterAngle, RuntimeUnitFilterAmount},
    frame::{GraphLoweringGeneration, GraphLoweringPassId, GraphLoweringResourceId},
    image::ResolvedMaskUploadDescriptor,
    layer::BlendMode,
    renderer::Antialiasing,
    resource::{GaussianKernelKey, WorkingFormat},
    shader::{
        BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderMaskSamplingKey, ShaderModuleKey,
        ShaderTextureFormatKey,
    },
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeGraphGeneration(pub(super) GraphLoweringGeneration);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeResourceId(pub(super) GraphLoweringResourceId);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimePassId(pub(super) GraphLoweringPassId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    ClipCoverage,
    IsolationWorkingImage,
    ImportedImage,
    BackdropCopy,
    FilterIntermediate,
    ShadowImage,
    CompositeResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceFormat {
    VelloCaptureRgba8Unorm,
    ClipCoverageRgba8Unorm,
    Working(WorkingFormat),
    ResolvedMaskRgba8Unorm,
}

impl RuntimeResourceFormat {
    pub(super) const fn shader_key(self) -> ShaderTextureFormatKey {
        match self {
            Self::VelloCaptureRgba8Unorm => ShaderTextureFormatKey::VelloCaptureRgba8Unorm,
            Self::ClipCoverageRgba8Unorm => ShaderTextureFormatKey::ClipCoverageRgba8Unorm,
            Self::Working(format) => ShaderTextureFormatKey::working(format),
            Self::ResolvedMaskRgba8Unorm => ShaderTextureFormatKey::ResolvedMaskRgba8Unorm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeSpatialDescriptor {
    pub(super) logical_bounds: Rect,
    pub(super) device_origin: (i32, i32),
    pub(super) device_extent: PhysicalSize,
    pub(super) texel_origin: Point,
    pub(super) raster_scale: f64,
}

impl RuntimeSpatialDescriptor {
    #[must_use]
    pub(crate) const fn device_extent(self) -> PhysicalSize {
        self.device_extent
    }

    #[must_use]
    pub(crate) const fn texel_origin(self) -> Point {
        self.texel_origin
    }

    #[must_use]
    pub(crate) const fn raster_scale(self) -> f64 {
        self.raster_scale
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceProducer {
    Imported,
    Pass(RuntimePassId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceImport {
    ResolvedAlphaMask(ResolvedMaskUploadDescriptor),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeResourceRequest {
    pub(super) id: RuntimeResourceId,
    pub(super) role: RuntimeResourceRole,
    pub(super) format: RuntimeResourceFormat,
    pub(super) spatial: RuntimeSpatialDescriptor,
    pub(super) producer: RuntimeResourceProducer,
    pub(super) expected_reads: u32,
    pub(super) last_use: RuntimePassId,
    pub(super) import: Option<RuntimeResourceImport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeInitialization {
    SurfaceBaseColor,
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeVelloSpan {
    pub(super) scope: RuntimeVelloSpanScope,
    pub(super) commands: RenderCommands,
    pub(super) capture_transform: Transform,
    pub(super) parent_to_surface: Transform,
    pub(super) antialiasing: Antialiasing,
    pub(super) captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverageElement {
    pub(super) clip: RenderClip,
    pub(super) transform: Transform,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverage {
    pub(super) elements: Vec<RuntimeClipCoverageElement>,
    pub(super) antialiasing: Antialiasing,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeVelloCapture {
    Span(RuntimeVelloSpan),
    ClipCoverage(RuntimeClipCoverage),
}

impl RuntimeVelloCapture {
    pub(super) const fn antialiasing(&self) -> Antialiasing {
        match self {
            Self::Span(span) => span.antialiasing,
            Self::ClipCoverage(coverage) => coverage.antialiasing,
        }
    }

    pub(super) fn span(&self) -> Option<&RuntimeVelloSpan> {
        match self {
            Self::Span(span) => Some(span),
            Self::ClipCoverage(_) => None,
        }
    }

    pub(super) fn clip_coverage(&self) -> Option<&RuntimeClipCoverage> {
        match self {
            Self::Span(_) => None,
            Self::ClipCoverage(coverage) => Some(coverage),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeColorClampBoundary {
    ClampStraightRgbaToUnitThenPremultiply,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeColorOperationKind {
    Brightness(RuntimeFilterAmount),
    Contrast(RuntimeFilterAmount),
    Grayscale(RuntimeUnitFilterAmount),
    HueRotate(RuntimeFilterAngle),
    Invert(RuntimeUnitFilterAmount),
    Opacity(RuntimeUnitFilterAmount),
    Saturate(RuntimeFilterAmount),
    Sepia(RuntimeUnitFilterAmount),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeColorOperation {
    pub(super) operation: RuntimeColorOperationKind,
    pub(super) clamp_boundary: RuntimeColorClampBoundary,
}

impl RuntimeColorOperation {
    #[must_use]
    pub(crate) const fn operation(self) -> RuntimeColorOperationKind {
        self.operation
    }

    #[must_use]
    pub(crate) const fn clamp_boundary(self) -> RuntimeColorClampBoundary {
        self.clamp_boundary
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeSamplingEdge {
    ClampToExtent,
    TransparentBlack,
    SemanticBorderMirror(Rect),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeFilterSpatialMapping {
    pub(super) source: RuntimeSpatialDescriptor,
    pub(super) result: RuntimeSpatialDescriptor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeColorFilter {
    pub(super) operations: Vec<RuntimeColorOperation>,
    pub(super) spatial: RuntimeFilterSpatialMapping,
    pub(super) edge: RuntimeSamplingEdge,
}

impl RuntimeColorFilter {
    #[must_use]
    pub(crate) fn operations(&self) -> &[RuntimeColorOperation] {
        &self.operations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeBlur {
    pub(super) axis: RuntimeBlurAxis,
    pub(super) input: RuntimeBlurInput,
    pub(super) standard_deviation: f64,
    pub(super) support_radius: u32,
    pub(super) kernel: GaussianKernelKey,
    pub(super) spatial: RuntimeFilterSpatialMapping,
    pub(super) edge: RuntimeSamplingEdge,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDropShadow {
    pub(super) offset: Point,
    pub(super) standard_deviation: f64,
    pub(super) color: Color,
    pub(super) support_radius: u32,
    pub(super) spatial: RuntimeFilterSpatialMapping,
    pub(super) edge: RuntimeSamplingEdge,
    pub(super) uses_source_alpha: bool,
    pub(super) uses_continuous_offset: bool,
    pub(super) retains_unchanged_source: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeOuterClip {
    pub(super) clip: RenderClip,
    pub(super) transform: Transform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDestinationToLayerLocal {
    pub(super) affine: Transform,
}

impl RuntimeDestinationToLayerLocal {
    pub(super) fn try_new(affine: Transform) -> Result<Self> {
        if !runtime_affine_is_finite_and_non_singular(affine) {
            return Err(Error::invalid_value(
                "destination-to-layer-local affine mapping",
                format!("{:?}", affine.as_array()),
                "must be finite and non-singular",
            ));
        }
        Ok(Self { affine })
    }

    #[must_use]
    pub(crate) const fn affine(self) -> Transform {
        self.affine
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeMaskTexelCenterFacts {
    pub(super) half_texel_normalized: [f64; 2],
    pub(super) texel_size_normalized: [f64; 2],
}

impl RuntimeMaskTexelCenterFacts {
    pub(super) fn try_new(image_dimensions: PhysicalSize) -> Result<Self> {
        if image_dimensions.width() == 0 || image_dimensions.height() == 0 {
            return Err(Error::invalid_value(
                "composite mask image dimensions",
                format!("{}x{}", image_dimensions.width(), image_dimensions.height()),
                "must be positive before deriving texel-center facts",
            ));
        }
        let texel_size_normalized = [
            1.0 / f64::from(image_dimensions.width()),
            1.0 / f64::from(image_dimensions.height()),
        ];
        let half_texel_normalized = [
            texel_size_normalized[0] * 0.5,
            texel_size_normalized[1] * 0.5,
        ];
        if texel_size_normalized
            .into_iter()
            .chain(half_texel_normalized)
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "composite mask texel-center facts must be finite and positive",
            ));
        }
        Ok(Self {
            half_texel_normalized,
            texel_size_normalized,
        })
    }

    #[must_use]
    pub(crate) const fn half_texel_normalized(self) -> [f64; 2] {
        self.half_texel_normalized
    }

    #[must_use]
    pub(crate) const fn texel_size_normalized(self) -> [f64; 2] {
        self.texel_size_normalized
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeResolvedAlphaMaskComposition {
    pub(super) resource: RuntimeResourceId,
    pub(super) bounds: Rect,
    pub(super) image_dimensions: PhysicalSize,
    pub(super) texel_center_facts: RuntimeMaskTexelCenterFacts,
    pub(super) sampling: ShaderMaskSamplingKey,
}

impl RuntimeResolvedAlphaMaskComposition {
    pub(super) fn try_new(
        resource: RuntimeResourceId,
        bounds: Rect,
        image_dimensions: PhysicalSize,
        sampling: ShaderMaskSamplingKey,
    ) -> Result<Self> {
        let maximum_x = bounds.x() + bounds.width();
        let maximum_y = bounds.y() + bounds.height();
        if !bounds.x().is_finite()
            || !bounds.y().is_finite()
            || !bounds.width().is_finite()
            || !bounds.height().is_finite()
            || bounds.width() <= 0.0
            || bounds.height() <= 0.0
            || !maximum_x.is_finite()
            || !maximum_y.is_finite()
        {
            return Err(Error::invalid_value(
                "composite mask semantic bounds",
                format!(
                    "({}, {}, {}, {})",
                    bounds.x(),
                    bounds.y(),
                    bounds.width(),
                    bounds.height()
                ),
                "must be a finite positive rectangle with a finite maximum",
            ));
        }
        Ok(Self {
            resource,
            bounds,
            image_dimensions,
            texel_center_facts: RuntimeMaskTexelCenterFacts::try_new(image_dimensions)?,
            sampling,
        })
    }

    #[must_use]
    pub(crate) const fn resource(self) -> RuntimeResourceId {
        self.resource
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> Rect {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn image_dimensions(self) -> PhysicalSize {
        self.image_dimensions
    }

    #[must_use]
    pub(crate) const fn texel_center_facts(self) -> RuntimeMaskTexelCenterFacts {
        self.texel_center_facts
    }

    #[must_use]
    pub(crate) const fn sampling(self) -> ShaderMaskSamplingKey {
        self.sampling
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeLayerCompositeParameters {
    pub(super) destination_to_layer_local: RuntimeDestinationToLayerLocal,
    pub(super) opacity: f32,
    pub(super) blend: BlendMode,
    pub(super) has_clip: bool,
    pub(super) alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
}

impl RuntimeLayerCompositeParameters {
    pub(super) fn try_new(
        destination_to_layer_local: Transform,
        opacity: f32,
        blend: BlendMode,
        has_clip: bool,
        alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
    ) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "composite opacity",
                opacity,
                "must be finite before clamping",
            ));
        }
        Ok(Self {
            destination_to_layer_local: RuntimeDestinationToLayerLocal::try_new(
                destination_to_layer_local,
            )?,
            opacity: opacity.clamp(0.0, 1.0),
            blend,
            has_clip,
            alpha_mask,
        })
    }

    #[must_use]
    pub(crate) const fn destination_to_layer_local(self) -> RuntimeDestinationToLayerLocal {
        self.destination_to_layer_local
    }

    #[must_use]
    pub(crate) const fn opacity(self) -> f32 {
        self.opacity
    }

    #[must_use]
    pub(crate) const fn blend(self) -> BlendMode {
        self.blend
    }

    #[must_use]
    pub(crate) const fn has_clip(self) -> bool {
        self.has_clip
    }

    #[must_use]
    pub(crate) const fn alpha_mask(self) -> Option<RuntimeResolvedAlphaMaskComposition> {
        self.alpha_mask
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        parameters: Box<RuntimeLayerCompositeParameters>,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<RuntimeOuterClip>,
        clip_coverage: Option<RuntimeResourceId>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeComposite {
    pub(super) kind: RuntimeCompositeKind,
    pub(super) source_captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimePassKind {
    ClearRoot {
        initialization: RuntimeInitialization,
        color: Color,
    },
    VelloCapture(Option<RuntimeVelloCapture>),
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter(Option<RuntimeColorFilter>),
    BlurHorizontal(Option<RuntimeBlur>),
    BlurVertical(Option<RuntimeBlur>),
    DropShadowColorize(Option<RuntimeDropShadow>),
    Composite(Option<RuntimeComposite>),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeReadRole {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeSamplingFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeReadBinding {
    pub(super) role: RuntimeReadRole,
    pub(super) resource: RuntimeResourceId,
    pub(super) sampling_filter: RuntimeSamplingFilter,
    pub(super) sampling_edge: RuntimeSamplingEdge,
    pub(super) sampler_key: SamplerKey,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these exact immutable runtime read-binding facts"
    )
)]
impl RuntimeReadBinding {
    pub(crate) const fn role(&self) -> RuntimeReadRole {
        self.role
    }

    pub(crate) const fn resource(&self) -> RuntimeResourceId {
        self.resource
    }

    pub(crate) const fn sampling_filter(&self) -> RuntimeSamplingFilter {
        self.sampling_filter
    }

    pub(crate) const fn sampling_edge(&self) -> RuntimeSamplingEdge {
        self.sampling_edge
    }

    pub(crate) const fn sampler_key(&self) -> SamplerKey {
        self.sampler_key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResultBinding {
    Empty,
    Resource(RuntimeResourceId),
    Output(Format),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimePassCacheKeys {
    pub(super) samplers: Vec<SamplerKey>,
    pub(super) layout: BindGroupLayoutKey,
    pub(super) shader: ShaderModuleKey,
    pub(super) pipeline: RenderPipelineKey,
}

impl RuntimePassCacheKeys {
    pub(crate) fn samplers(&self) -> &[SamplerKey] {
        &self.samplers
    }

    pub(crate) const fn layout(&self) -> &BindGroupLayoutKey {
        &self.layout
    }

    pub(crate) const fn shader(&self) -> &ShaderModuleKey {
        &self.shader
    }

    pub(crate) const fn pipeline(&self) -> &RenderPipelineKey {
        &self.pipeline
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimePass {
    pub(super) id: RuntimePassId,
    pub(super) kind: RuntimePassKind,
    pub(super) dependencies: Vec<RuntimePassId>,
    pub(super) reads: Vec<RuntimeReadBinding>,
    pub(super) result: RuntimeResultBinding,
    pub(super) releases: Vec<RuntimeResourceId>,
    pub(super) cache_keys: Option<RuntimePassCacheKeys>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredGraphPlan {
    pub(super) generation: RuntimeGraphGeneration,
    pub(super) working_format: WorkingFormat,
    pub(super) output_format: Format,
    pub(super) resources: Vec<RuntimeResourceRequest>,
    pub(super) passes: Vec<RuntimePass>,
    pub(super) root_working_image: RuntimeResourceId,
    pub(super) final_present: RuntimePassId,
}

pub(super) fn runtime_affine_is_finite_and_non_singular(transform: Transform) -> bool {
    let [a, b, c, d, e, f] = transform.as_array();
    if [a, b, c, d, e, f]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == 0.0 {
        return false;
    }
    let a = a / scale;
    let b = b / scale;
    let c = c / scale;
    let d = d / scale;
    let determinant = a * d - b * c;
    determinant.is_finite() && determinant != 0.0
}
