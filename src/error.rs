use std::{error, fmt};

pub type Result<T> = std::result::Result<T, Error>;

/// Stable render diagnostic.
#[derive(Debug)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
    pub source: Option<Box<dyn error::Error + Send + Sync>>,
    unsupported_primitive: Option<UnsupportedPrimitive>,
}

impl Error {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
            unsupported_primitive: None,
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
        Self::new(
            ErrorCode::InvalidInput,
            format!("{} value {value} is invalid: {rule}", name.into()),
        )
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
    pub const fn unsupported_primitive(&self) -> Option<UnsupportedPrimitive> {
        self.unsupported_primitive
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
    ImageUploadFailed,
    RenderFailed,
    PresentFailed,
    UnsupportedBackend,
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
    Filters,
    MasksAndClips,
    Surfaces,
}

impl PrimitiveFamily {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GeometryTargets => "geometry targets",
            Self::PaintSources => "paint sources",
            Self::Filters => "filters",
            Self::MasksAndClips => "masks and clips",
            Self::Surfaces => "surfaces",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveOperation {
    LayerFilter,
    LayerMask,
    NonSolidShadowPaint,
    InsideOutsidePathStrokeAlignment,
    WebCanvasSurface,
}

impl PrimitiveOperation {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LayerFilter => "layer filter",
            Self::LayerMask => "layer mask",
            Self::NonSolidShadowPaint => "non-solid shadow paint",
            Self::InsideOutsidePathStrokeAlignment => "inside/outside path stroke alignment",
            Self::WebCanvasSurface => "web canvas surface",
        }
    }
}
