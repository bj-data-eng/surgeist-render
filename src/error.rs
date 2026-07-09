use std::{error, fmt};

pub type Result<T> = std::result::Result<T, Error>;

/// Stable render diagnostic.
#[derive(Debug)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
    pub source: Option<Box<dyn error::Error + Send + Sync>>,
    invalid_value: Option<Box<InvalidValue>>,
    unsupported_primitive: Option<UnsupportedPrimitive>,
    unresolved_resource: Option<Box<UnresolvedResource>>,
    degraded_quality: Option<Box<DegradedQuality>>,
}

impl Error {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
            invalid_value: None,
            unsupported_primitive: None,
            unresolved_resource: None,
            degraded_quality: None,
        }
    }

    #[must_use]
    pub fn with_source(mut self, source: impl error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    #[must_use]
    pub fn invalid_value(
        name: impl Into<String>,
        value: impl std::fmt::Display,
        rule: &'static str,
    ) -> Self {
        Self::from_invalid_value(InvalidValue::new(name, value, rule))
    }

    #[must_use]
    pub fn from_invalid_value(invalid_value: InvalidValue) -> Self {
        let mut error = Self::new(ErrorCode::InvalidInput, invalid_value.message());
        error.invalid_value = Some(Box::new(invalid_value));
        error
    }

    #[must_use]
    pub fn unsupported_render_primitive(primitive: UnsupportedPrimitive) -> Self {
        let mut error = Self::new(
            ErrorCode::UnsupportedBackend,
            format!(
                "render primitive is unsupported: {} / {}",
                primitive.family().label(),
                primitive.label()
            ),
        );
        error.unsupported_primitive = Some(primitive);
        error
    }

    #[must_use]
    pub fn unresolved_resource(resource: UnresolvedResource) -> Self {
        let mut error = Self::new(ErrorCode::UnresolvedResource, resource.message());
        error.unresolved_resource = Some(Box::new(resource));
        error
    }

    #[must_use]
    pub fn degraded_quality(diagnostic: DegradedQuality) -> Self {
        let mut error = Self::new(ErrorCode::DegradedQuality, diagnostic.message());
        error.degraded_quality = Some(Box::new(diagnostic));
        error
    }

    #[must_use]
    pub const fn unsupported_primitive(&self) -> Option<UnsupportedPrimitive> {
        self.unsupported_primitive
    }

    #[must_use]
    pub const fn invalid_value_diagnostic(&self) -> Option<&InvalidValue> {
        match &self.invalid_value {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }

    #[must_use]
    pub const fn unresolved_resource_diagnostic(&self) -> Option<&UnresolvedResource> {
        match &self.unresolved_resource {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }

    #[must_use]
    pub const fn degraded_quality_diagnostic(&self) -> Option<&DegradedQuality> {
        match &self.degraded_quality {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.source.as_deref().map(|error| error as _)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    AdapterUnavailable,
    DeviceCreateFailed,
    RendererCreateFailed,
    SurfaceCreateFailed,
    SurfaceConfigureFailed,
    SurfaceLost,
    SurfaceOutOfMemory,
    SurfaceTimeout,
    SurfaceOutdated,
    SurfaceUnavailable,
    InvalidInput,
    UnresolvedResource,
    DegradedQuality,
    ImageUploadFailed,
    RenderFailed,
    PresentFailed,
    UnsupportedBackend,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidValue {
    field: String,
    value: String,
    invariant: &'static str,
}

impl InvalidValue {
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        value: impl std::fmt::Display,
        invariant: &'static str,
    ) -> Self {
        Self {
            field: field.into(),
            value: format!("{value}"),
            invariant,
        }
    }

    #[must_use]
    pub fn field(&self) -> &str {
        &self.field
    }

    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    #[must_use]
    pub const fn invariant(&self) -> &'static str {
        self.invariant
    }

    fn message(&self) -> String {
        format!(
            "{} value {} is invalid: {}",
            self.field, self.value, self.invariant
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedPrimitive {
    family: PrimitiveFamily,
    operation: PrimitiveOperation,
}

impl UnsupportedPrimitive {
    #[must_use]
    pub const fn new(family: PrimitiveFamily, operation: PrimitiveOperation) -> Self {
        Self { family, operation }
    }

    #[must_use]
    pub const fn family(self) -> PrimitiveFamily {
        self.family
    }

    #[must_use]
    pub const fn operation(self) -> PrimitiveOperation {
        self.operation
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        self.operation.label()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveFamily {
    GeometryTargets,
    PaintSources,
    ImageSampling,
    Shadows,
    Filters,
    MasksAndClips,
    BoxDecorations,
    OffscreenPipeline,
    Surfaces,
    TransformsAndCoordinateSpaces,
}

impl PrimitiveFamily {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GeometryTargets => "geometry targets",
            Self::PaintSources => "paint sources",
            Self::ImageSampling => "image sampling",
            Self::Shadows => "shadows",
            Self::Filters => "filters",
            Self::MasksAndClips => "masks and clips",
            Self::BoxDecorations => "box decorations",
            Self::OffscreenPipeline => "offscreen pipeline",
            Self::Surfaces => "surfaces",
            Self::TransformsAndCoordinateSpaces => "transforms and coordinate spaces",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveOperation {
    LayerFilter,
    ShapeClip,
    ClipReferenceExecution,
    LayerMask,
    AlphaMaskExecution,
    LuminanceMaskMode,
    MultiLayerMaskComposition,
    MaskCompositeMode,
    NonSolidShadowPaint,
    UnresolvedSymbolicColor,
    ColorMixFunction,
    UnsupportedColorSpace,
    RepeatingGradient,
    BackgroundRepeatRound,
    BackgroundRepeatSpace,
    FilteredImagePaint,
    ColorFilteredImagePaint,
    ImageOrientationConversion,
    ImageColorProfileConversion,
    EllipsePathShadowShape,
    InsetBoxShadow,
    TextShadow,
    InsideOutsidePathStrokeAlignment,
    GeometryBooleanOperation,
    GeometryOffsetOperation,
    WebCanvasSurface,
    Matrix3dTransform,
    PerspectiveTransform,
    Rotate3dTransform,
    TranslateZTransform,
    ScaleZTransform,
    BorderGrooveStyle,
    BorderRidgeStyle,
    BorderInsetStyle,
    BorderOutsetStyle,
    OutlineDoubleStyle,
    OutlineAutoStyle,
    OffscreenLayerRendering,
    TextureCacheUploadLifecycle,
    RectFullscreenShaderPass,
    CpuReferenceBuffer,
    NestedOpacityPlanning,
    MaskExecution,
    FilterExecution,
    ColorFilterClassification,
    ColorFilterPipelineExecution,
    MaterializedImageFilterClassification,
    MaterializedBlurFilterExecution,
    MaterializedDropShadowFilterExecution,
    FilterRegionOutsetPlanning,
    CpuReferenceBlurFallback,
    ColorFilterBlur,
    ColorFilterDropShadow,
    BackdropExecution,
}

impl PrimitiveOperation {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LayerFilter => "layer filter",
            Self::ShapeClip => "shape clip",
            Self::ClipReferenceExecution => "clip reference execution",
            Self::LayerMask => "layer mask",
            Self::AlphaMaskExecution => "alpha mask execution",
            Self::LuminanceMaskMode => "luminance mask mode",
            Self::MultiLayerMaskComposition => "multi-layer mask composition",
            Self::MaskCompositeMode => "mask composite mode",
            Self::NonSolidShadowPaint => "non-solid shadow paint",
            Self::UnresolvedSymbolicColor => "unresolved symbolic color",
            Self::ColorMixFunction => "color-mix function",
            Self::UnsupportedColorSpace => "unsupported color space",
            Self::RepeatingGradient => "repeating gradient",
            Self::BackgroundRepeatRound => "background repeat round",
            Self::BackgroundRepeatSpace => "background repeat space",
            Self::FilteredImagePaint => "filtered image paint",
            Self::ColorFilteredImagePaint => "color-filtered image paint",
            Self::ImageOrientationConversion => "image orientation conversion",
            Self::ImageColorProfileConversion => "image color profile conversion",
            Self::EllipsePathShadowShape => "ellipse/path shadow shape",
            Self::InsetBoxShadow => "inset box shadow",
            Self::TextShadow => "text shadow",
            Self::InsideOutsidePathStrokeAlignment => "inside/outside path stroke alignment",
            Self::GeometryBooleanOperation => "geometry boolean operation",
            Self::GeometryOffsetOperation => "geometry offset operation",
            Self::WebCanvasSurface => "web canvas surface",
            Self::Matrix3dTransform => "matrix3d transform",
            Self::PerspectiveTransform => "perspective transform",
            Self::Rotate3dTransform => "rotate3d transform",
            Self::TranslateZTransform => "translateZ transform",
            Self::ScaleZTransform => "scaleZ transform",
            Self::BorderGrooveStyle => "border groove style",
            Self::BorderRidgeStyle => "border ridge style",
            Self::BorderInsetStyle => "border inset style",
            Self::BorderOutsetStyle => "border outset style",
            Self::OutlineDoubleStyle => "outline double style",
            Self::OutlineAutoStyle => "outline auto style",
            Self::OffscreenLayerRendering => "offscreen layer rendering",
            Self::TextureCacheUploadLifecycle => "texture cache/upload lifecycle",
            Self::RectFullscreenShaderPass => "rect/fullscreen shader pass",
            Self::CpuReferenceBuffer => "CPU reference buffer",
            Self::NestedOpacityPlanning => "nested opacity planning",
            Self::MaskExecution => "mask execution",
            Self::FilterExecution => "filter execution",
            Self::ColorFilterClassification => "color filter classification",
            Self::ColorFilterPipelineExecution => "color filter pipeline execution",
            Self::MaterializedImageFilterClassification => {
                "materialized image filter classification"
            }
            Self::MaterializedBlurFilterExecution => "materialized blur filter execution",
            Self::MaterializedDropShadowFilterExecution => {
                "materialized drop-shadow filter execution"
            }
            Self::FilterRegionOutsetPlanning => "filter-region/outset planning",
            Self::CpuReferenceBlurFallback => "CPU/reference blur fallback",
            Self::ColorFilterBlur => "color filter blur",
            Self::ColorFilterDropShadow => "color filter drop-shadow",
            Self::BackdropExecution => "backdrop execution",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedResource {
    kind: UnresolvedResourceKind,
    identifier: String,
}

impl UnresolvedResource {
    #[must_use]
    pub fn new(kind: UnresolvedResourceKind, identifier: impl Into<String>) -> Self {
        Self {
            kind,
            identifier: identifier.into(),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> UnresolvedResourceKind {
        self.kind
    }

    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    fn message(&self) -> String {
        format!(
            "{} resource {} could not be resolved",
            self.kind.label(),
            self.identifier
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnresolvedResourceKind {
    Image,
    Mask,
    Filter,
    Clip,
}

impl UnresolvedResourceKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Mask => "mask",
            Self::Filter => "filter",
            Self::Clip => "clip",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DegradedQuality {
    kind: DegradedQualityKind,
    value: String,
}

impl DegradedQuality {
    #[must_use]
    pub fn new(kind: DegradedQualityKind, value: impl Into<String>) -> Self {
        Self {
            kind,
            value: value.into(),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> DegradedQualityKind {
        self.kind
    }

    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    fn message(&self) -> String {
        format!(
            "render quality degraded: {} ({})",
            self.kind.label(),
            self.value
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DegradedQualityKind {
    FastBlurClamp,
    SoftwareFallback,
    UnsupportedPaintSpaceConversion,
}

impl DegradedQualityKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FastBlurClamp => "fast blur clamp",
            Self::SoftwareFallback => "software fallback",
            Self::UnsupportedPaintSpaceConversion => "unsupported paint-space conversion",
        }
    }
}
