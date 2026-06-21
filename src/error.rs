use std::{error, fmt};

pub type Result<T> = std::result::Result<T, Error>;

/// Stable render diagnostic.
#[derive(Debug)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
    pub source: Option<Box<dyn error::Error + Send + Sync>>,
}

impl Error {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    #[must_use]
    pub fn with_source(mut self, source: impl error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
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
