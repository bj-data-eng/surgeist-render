use std::{error, fmt};

use crate::{EffectQualityPolicy, Format, PhysicalSize};

/// Result type returned by fallible rendering contracts.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable render diagnostic with optional typed semantic or runtime context.
///
/// Callers should branch on [`ErrorCode`] and the matching typed payload rather
/// than parse [`Self::message`]. Frame-operation errors are failure-atomic: the
/// attempted frame is not published, and any previous complete frame remains
/// observable.
#[derive(Debug)]
pub struct Error {
    code: ErrorCode,
    message: String,
    source: Option<BackendErrorSource>,
    invalid_value: Option<Box<InvalidValue>>,
    unsupported_primitive: Option<UnsupportedPrimitive>,
    unresolved_resource: Option<Box<UnresolvedResource>>,
    degraded_quality: Option<Box<DegradedQuality>>,
    runtime_capability_unavailable: Option<RuntimeCapabilityUnavailable>,
}

#[cfg(not(target_arch = "wasm32"))]
type BackendErrorSource = Box<dyn error::Error + Send + Sync + 'static>;

#[cfg(target_arch = "wasm32")]
type BackendErrorSource = Box<dyn error::Error + 'static>;

/// Private code domain accepted by backend error construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendErrorCode {
    DeviceCreateFailed,
    RendererCreateFailed,
    SurfaceCreateFailed,
    SurfaceConfigureFailed,
    SurfaceOutOfMemory,
    SurfaceTimeout,
    SurfaceOutdated,
    ImageUploadFailed,
    RenderFailed,
    /// A texture readback failed during copy, mapping, or row decoding.
    ReadbackFailed,
    PresentFailed,
    UnsupportedBackend,
}

impl BackendErrorCode {
    const ALL: [Self; 12] = [
        Self::DeviceCreateFailed,
        Self::RendererCreateFailed,
        Self::SurfaceCreateFailed,
        Self::SurfaceConfigureFailed,
        Self::SurfaceOutOfMemory,
        Self::SurfaceTimeout,
        Self::SurfaceOutdated,
        Self::ImageUploadFailed,
        Self::RenderFailed,
        Self::ReadbackFailed,
        Self::PresentFailed,
        Self::UnsupportedBackend,
    ];

    const fn error_code(self) -> ErrorCode {
        match self {
            Self::DeviceCreateFailed => ErrorCode::DeviceCreateFailed,
            Self::RendererCreateFailed => ErrorCode::RendererCreateFailed,
            Self::SurfaceCreateFailed => ErrorCode::SurfaceCreateFailed,
            Self::SurfaceConfigureFailed => ErrorCode::SurfaceConfigureFailed,
            Self::SurfaceOutOfMemory => ErrorCode::SurfaceOutOfMemory,
            Self::SurfaceTimeout => ErrorCode::SurfaceTimeout,
            Self::SurfaceOutdated => ErrorCode::SurfaceOutdated,
            Self::ImageUploadFailed => ErrorCode::ImageUploadFailed,
            Self::RenderFailed => ErrorCode::RenderFailed,
            Self::ReadbackFailed => ErrorCode::ReadbackFailed,
            Self::PresentFailed => ErrorCode::PresentFailed,
            Self::UnsupportedBackend => ErrorCode::UnsupportedBackend,
        }
    }
}

impl Error {
    #[must_use]
    pub(crate) fn new(code: BackendErrorCode, message: impl Into<String>) -> Self {
        debug_assert!(BackendErrorCode::ALL.contains(&code));
        Self {
            code: code.error_code(),
            message: message.into(),
            source: None,
            invalid_value: None,
            unsupported_primitive: None,
            unresolved_resource: None,
            degraded_quality: None,
            runtime_capability_unavailable: None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub(crate) fn with_source(mut self, source: impl error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub(crate) fn with_source(mut self, source: impl error::Error + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub(crate) fn append_message(&mut self, suffix: impl fmt::Display) {
        self.message.push_str(&suffix.to_string());
    }

    pub(crate) fn replace_message(&mut self, message: impl Into<String>) {
        self.message = message.into();
    }

    pub(crate) fn invalid_input_message(message: impl Into<String>) -> Self {
        let mut error = Self::from_invalid_value(InvalidValue::new(
            "input",
            "internal validation",
            "must satisfy the requested validation rule",
        ));
        error.replace_message(message);
        error
    }

    pub(crate) fn runtime_unavailable(
        operation: RuntimeOperation,
        reason: RuntimeCapabilityUnavailableReason,
        message: impl Into<String>,
    ) -> Self {
        let diagnostic = RuntimeCapabilityUnavailable::try_new(operation, reason)
            .expect("runtime unavailability constructors use validated operation/reason pairs");
        let mut error = Self::runtime_capability_unavailable(diagnostic);
        error.replace_message(message);
        error
    }

    /// Returns this diagnostic's stable classification.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        self.code
    }

    /// Returns the human-readable diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Creates an invalid-input diagnostic with its structured payload.
    #[must_use]
    pub fn invalid_value(
        name: impl Into<String>,
        value: impl std::fmt::Display,
        rule: &'static str,
    ) -> Self {
        Self::from_invalid_value(InvalidValue::new(name, value, rule))
    }

    /// Creates an invalid-input diagnostic whose structured payload is available through
    /// [`Self::invalid_value_diagnostic`].
    #[must_use]
    pub fn from_invalid_value(invalid_value: InvalidValue) -> Self {
        Self {
            code: ErrorCode::InvalidInput,
            message: invalid_value.message(),
            source: None,
            invalid_value: Some(Box::new(invalid_value)),
            unsupported_primitive: None,
            unresolved_resource: None,
            degraded_quality: None,
            runtime_capability_unavailable: None,
        }
    }

    /// Creates an unsupported-primitive diagnostic whose payload is returned by
    /// [`Self::unsupported_primitive`].
    #[must_use]
    pub fn unsupported_render_primitive(primitive: UnsupportedPrimitive) -> Self {
        Self {
            code: ErrorCode::UnsupportedPrimitive,
            message: format!(
                "render primitive is unsupported: {} / {}",
                primitive.family().label(),
                primitive.label()
            ),
            source: None,
            invalid_value: None,
            unsupported_primitive: Some(primitive),
            unresolved_resource: None,
            degraded_quality: None,
            runtime_capability_unavailable: None,
        }
    }

    /// Creates an unresolved-resource diagnostic whose structured payload is available through
    /// [`Self::unresolved_resource_diagnostic`].
    #[must_use]
    pub fn unresolved_resource(resource: UnresolvedResource) -> Self {
        Self {
            code: ErrorCode::UnresolvedResource,
            message: resource.message(),
            source: None,
            invalid_value: None,
            unsupported_primitive: None,
            unresolved_resource: Some(Box::new(resource)),
            degraded_quality: None,
            runtime_capability_unavailable: None,
        }
    }

    /// Creates a degraded-quality diagnostic whose structured payload is available through
    /// [`Self::degraded_quality_diagnostic`].
    #[must_use]
    pub fn degraded_quality(diagnostic: DegradedQuality) -> Self {
        Self {
            code: ErrorCode::DegradedQuality,
            message: diagnostic.message(),
            source: None,
            invalid_value: None,
            unsupported_primitive: None,
            unresolved_resource: None,
            degraded_quality: Some(Box::new(diagnostic)),
            runtime_capability_unavailable: None,
        }
    }

    /// Creates a runtime-capability diagnostic whose payload is available through
    /// [`Self::runtime_capability_unavailable_diagnostic`].
    #[must_use]
    pub fn runtime_capability_unavailable(value: RuntimeCapabilityUnavailable) -> Self {
        debug_assert!(
            RuntimeCapabilityUnavailable::try_new(value.operation(), value.reason()).is_ok()
        );
        Self {
            code: ErrorCode::RuntimeCapabilityUnavailable,
            message: "runtime capability is unavailable".into(),
            source: None,
            invalid_value: None,
            unsupported_primitive: None,
            unresolved_resource: None,
            degraded_quality: None,
            runtime_capability_unavailable: Some(value),
        }
    }

    /// Returns the unsupported-primitive payload, when this diagnostic carries one.
    #[must_use]
    pub const fn unsupported_primitive(&self) -> Option<UnsupportedPrimitive> {
        self.unsupported_primitive
    }

    /// Returns the invalid-input payload, when this diagnostic carries one.
    #[must_use]
    pub const fn invalid_value_diagnostic(&self) -> Option<&InvalidValue> {
        match &self.invalid_value {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }

    /// Returns the unresolved-resource payload, when this diagnostic carries one.
    #[must_use]
    pub const fn unresolved_resource_diagnostic(&self) -> Option<&UnresolvedResource> {
        match &self.unresolved_resource {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }

    /// Returns the degraded-quality payload, when this diagnostic carries one.
    #[must_use]
    pub const fn degraded_quality_diagnostic(&self) -> Option<&DegradedQuality> {
        match &self.degraded_quality {
            Some(diagnostic) => Some(diagnostic),
            None => None,
        }
    }

    /// Returns the runtime-capability payload, when this diagnostic carries one.
    #[must_use]
    pub const fn runtime_capability_unavailable_diagnostic(
        &self,
    ) -> Option<&RuntimeCapabilityUnavailable> {
        self.runtime_capability_unavailable.as_ref()
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
/// Stable classifications for render diagnostics, suitable for programmatic failure handling.
pub enum ErrorCode {
    DeviceCreateFailed,
    RendererCreateFailed,
    SurfaceCreateFailed,
    SurfaceConfigureFailed,
    SurfaceOutOfMemory,
    SurfaceTimeout,
    SurfaceOutdated,
    InvalidInput,
    /// Reports a render primitive that this renderer cannot represent.
    UnsupportedPrimitive,
    UnresolvedResource,
    DegradedQuality,
    /// Reports that a runtime GPU capability prevented a specific render operation.
    RuntimeCapabilityUnavailable,
    ImageUploadFailed,
    RenderFailed,
    /// Reports failure to copy, map, or decode pixels during GPU readback.
    ReadbackFailed,
    PresentFailed,
    UnsupportedBackend,
}

/// Validated runtime-phase evidence that an operation cannot use a selected GPU capability.
///
/// Each value couples one [`RuntimeOperation`] with a reason permitted for that
/// operation. Callers receive the validated pair through [`Error`]; backend
/// failures cannot manufacture an unrelated operation/reason combination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCapabilityUnavailable {
    operation: RuntimeOperation,
    reason: RuntimeCapabilityUnavailableReason,
}

impl RuntimeCapabilityUnavailable {
    /// Creates a diagnostic only when the operation and reason form are a valid runtime pair.
    pub(crate) fn try_new(
        operation: RuntimeOperation,
        reason: RuntimeCapabilityUnavailableReason,
    ) -> Result<Self> {
        if runtime_capability_pair_is_valid(operation, reason) {
            return Ok(Self { operation, reason });
        }

        Err(Error::invalid_value(
            "runtime capability unavailable pair",
            format!("{operation:?} / {reason:?}"),
            "operation and reason must be a permitted runtime capability unavailable pair",
        ))
    }

    /// Returns the operation that could not use the selected runtime capability.
    #[must_use]
    pub const fn operation(self) -> RuntimeOperation {
        self.operation
    }

    /// Returns the validated runtime reason that prevented the operation.
    #[must_use]
    pub const fn reason(self) -> RuntimeCapabilityUnavailableReason {
        self.reason
    }
}

fn runtime_capability_pair_is_valid(
    operation: RuntimeOperation,
    reason: RuntimeCapabilityUnavailableReason,
) -> bool {
    match operation {
        RuntimeOperation::AdapterSelection => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Occluded
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceReadback => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Uninitialized
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceResume => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectTextureAllocation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectPresentation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
    }
}

/// Runtime operation for which GPU capability availability is diagnosed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RuntimeOperation {
    /// Selecting an adapter for a presented surface.
    AdapterSelection,
    /// Rendering into a surface.
    SurfaceRendering,
    /// Reading pixels from a surface.
    SurfaceReadback,
    /// Resuming a suspended surface.
    SurfaceResume,
    /// Rendering an effect graph.
    EffectRendering,
    /// Allocating an effect texture.
    EffectTextureAllocation,
    /// Presenting an effect result to a surface.
    EffectPresentation,
}

/// Runtime reason that a GPU capability cannot serve an operation.
///
/// These reasons reject the owning GPU operation; there is no CPU fallback or
/// production CPU retry. Semantic unsupported-input diagnostics remain separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RuntimeCapabilityUnavailableReason {
    /// No compatible adapter is available.
    AdapterUnavailable,
    /// The surface is unavailable in the stated lifecycle condition.
    SurfaceUnavailable {
        /// Lifecycle condition that prevents the surface operation.
        state: RenderSurfaceAvailability,
    },
    /// The selected device has been lost.
    DeviceLost {
        /// Reason reported for the device loss.
        reason: DeviceLossReason,
    },
    /// The selected device has entered a terminal fault state.
    DeviceFaulted {
        /// Class of GPU fault observed for the device.
        kind: GpuFaultKind,
    },
    /// The surface belongs to a different renderer or device generation.
    SurfaceIdentityMismatch {
        /// Identity mismatch detected before the operation.
        kind: SurfaceIdentityMismatchKind,
    },
    /// No effect format satisfies the requested precision policy.
    EffectFormatUnavailable {
        /// Effect precision policy that could not be met.
        policy: EffectQualityPolicy,
    },
    /// The requested effect texture exceeds the selected device limit.
    TextureDimensionExceeded {
        /// Requested physical texture dimensions.
        requested: PhysicalSize,
        /// Maximum supported two-dimensional texture dimension.
        maximum: u32,
    },
    /// The surface format cannot receive the effect result.
    SurfaceFormatUnavailable {
        /// Surface format that cannot receive the effect result.
        format: Format,
    },
}

/// Lifecycle condition that makes a render surface unavailable for an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderSurfaceAvailability {
    /// The surface is suspended.
    Suspended,
    /// The surface has no renderable extent.
    NonRenderable,
    /// The surface has no initialized readable publication.
    Uninitialized,
    /// The surface is occluded and cannot acquire a frame.
    Occluded,
    /// The surface has been lost.
    Lost,
}

/// Identity mismatch detected between a renderer operation and a surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceIdentityMismatchKind {
    /// The surface belongs to another renderer instance.
    ForeignRenderer,
    /// The surface references a stale device generation.
    StaleDeviceGeneration,
}

/// Reason reported when a selected GPU device is lost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceLossReason {
    /// The backend did not provide a more specific loss reason.
    Unknown,
    /// The device was explicitly destroyed.
    Destroyed,
}

/// Terminal class of a GPU fault observed by the renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFaultKind {
    /// A GPU validation failure occurred.
    Validation,
    /// GPU memory was exhausted.
    OutOfMemory,
    /// An internal GPU failure occurred.
    Internal,
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
    TextDecorations,
    Compositing,
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
            Self::TextDecorations => "text decorations",
            Self::Compositing => "compositing",
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
    AlphaMaskSourceExecution,
    ResolvedAlphaMaskExecution,
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
    TextDecorationStyle,
    OffscreenLayerRendering,
    PersistentEffectResources,
    BoundedVelloCapture,
    ImagePassExecution,
    CompositePassExecution,
    NestedOpacityComposition,
    MaskExecution,
    LayerFilterExecution,
    OrderedFilterList,
    GpuColorFilterExecution,
    GpuBlurFilterExecution,
    GpuDropShadowFilterExecution,
    FilterRegionPlanning,
    BroadBackdropExecution,
    BoundedBackdropCapture,
    BoundedBackdropFilterExecution,
    BackdropIsolationComposition,
    RootBackdropPolicy,
    BackgroundBlendMode,
    AdditionalMixBlendMode,
    PorterDuffCompositeMode,
}

impl PrimitiveOperation {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LayerFilter => "layer filter",
            Self::ShapeClip => "shape clip",
            Self::ClipReferenceExecution => "clip reference execution",
            Self::LayerMask => "layer mask",
            Self::AlphaMaskSourceExecution => "alpha mask source execution",
            Self::ResolvedAlphaMaskExecution => "resolved alpha-mask execution",
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
            Self::TextDecorationStyle => "text decoration style",
            Self::OffscreenLayerRendering => "offscreen layer rendering",
            Self::PersistentEffectResources => "persistent effect resources",
            Self::BoundedVelloCapture => "bounded Vello capture",
            Self::ImagePassExecution => "image-pass execution",
            Self::CompositePassExecution => "composite-pass execution",
            Self::NestedOpacityComposition => "nested opacity composition",
            Self::MaskExecution => "mask execution",
            Self::LayerFilterExecution => "layer-filter execution",
            Self::OrderedFilterList => "ordered filter list",
            Self::GpuColorFilterExecution => "GPU color-filter execution",
            Self::GpuBlurFilterExecution => "GPU blur filter execution",
            Self::GpuDropShadowFilterExecution => "GPU drop-shadow filter execution",
            Self::FilterRegionPlanning => "filter-region planning",
            Self::BroadBackdropExecution => "broad backdrop execution",
            Self::BoundedBackdropCapture => "bounded backdrop capture",
            Self::BoundedBackdropFilterExecution => "bounded backdrop filter execution",
            Self::BackdropIsolationComposition => "backdrop isolation/composition",
            Self::RootBackdropPolicy => "root backdrop policy",
            Self::BackgroundBlendMode => "background blend mode",
            Self::AdditionalMixBlendMode => "additional mix-blend mode",
            Self::PorterDuffCompositeMode => "Porter-Duff composite mode",
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
    TextRunInkBounds,
}

impl UnresolvedResourceKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Mask => "mask",
            Self::Filter => "filter",
            Self::Clip => "clip",
            Self::TextRunInkBounds => "text run ink bounds",
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
/// Describes a quality limitation reported by a [`ErrorCode::DegradedQuality`] diagnostic.
pub enum DegradedQualityKind {
    /// Reports output produced with reduced intermediate precision.
    ReducedIntermediatePrecision,
    /// Reports inability to perform the requested paint-space conversion.
    UnsupportedPaintSpaceConversion,
}

impl DegradedQualityKind {
    /// Returns a stable human-readable label for this quality limitation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReducedIntermediatePrecision => "reduced intermediate precision",
            Self::UnsupportedPaintSpaceConversion => "unsupported paint-space conversion",
        }
    }
}
