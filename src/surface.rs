use super::{
    Color, Error, ErrorCode, PhysicalSize, Result, Size, geometry::physical_size, validation::*,
};

pub struct Surface {
    pub(crate) attachment: Attachment,
    pub(crate) options: SurfaceOptions,
    pub(crate) available: bool,
    pub(crate) last_parameters: Option<Parameters>,
    pub(crate) backend: SurfaceBackend,
}

impl Surface {
    pub(crate) fn with_backend(
        attachment: Attachment,
        options: SurfaceOptions,
        backend: SurfaceBackend,
    ) -> Self {
        Self {
            attachment,
            options,
            available: true,
            last_parameters: None,
            backend,
        }
    }

    pub fn resize(&mut self, size: Size, scale: f64) -> Result<()> {
        validate_size(size, "surface size")?;
        validate_positive_f64(scale, "surface scale")?;
        self.options.size = size;
        self.options.scale = scale;
        let next = physical_size(size, scale);
        match &mut self.backend {
            SurfaceBackend::ContractOnly { physical_size } => {
                *physical_size = next;
            }
            SurfaceBackend::Headless {
                texture,
                view,
                physical_size,
                ..
            } => {
                if *physical_size == next {
                    return Ok(());
                }
                *physical_size = next;
                *texture = None;
                *view = None;
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented {
                surface,
                valid,
                pending_physical_size,
                ..
            } => {
                *valid = next.width > 0 && next.height > 0;
                let current = PhysicalSize {
                    width: surface.config.width,
                    height: surface.config.height,
                };
                if pending_physical_size
                    .as_ref()
                    .is_some_and(|pending| *pending == next)
                {
                    return Ok(());
                }
                *pending_physical_size = (current != next).then_some(next);
            }
        }
        Ok(())
    }

    pub fn suspend(&mut self) -> Result<()> {
        self.available = false;
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { valid, .. } = &mut self.backend {
            *valid = false;
        }
        Ok(())
    }

    pub fn resume(&mut self, attachment: Attachment) -> Result<()> {
        if self.attachment.kind() != attachment.kind() {
            return Err(Error::new(
                ErrorCode::SurfaceCreateFailed,
                "surface cannot resume with an incompatible attachment",
            ));
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { .. } = &self.backend {
            return Err(Error::new(
                ErrorCode::SurfaceUnavailable,
                "presented surfaces must be resumed through Renderer::resume_surface",
            ));
        }
        self.attachment = attachment;
        self.available = true;
        Ok(())
    }

    #[must_use]
    pub const fn size(&self) -> Size {
        self.options.size
    }

    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.options.scale
    }

    #[must_use]
    pub const fn physical_size(&self) -> PhysicalSize {
        match &self.backend {
            SurfaceBackend::ContractOnly { physical_size }
            | SurfaceBackend::Headless { physical_size, .. } => *physical_size,
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { surface, .. } => PhysicalSize {
                width: surface.config.width,
                height: surface.config.height,
            },
        }
    }
}

pub(crate) enum SurfaceBackend {
    ContractOnly {
        physical_size: PhysicalSize,
    },
    Headless {
        dev_id: usize,
        texture: Option<wgpu::Texture>,
        view: Option<wgpu::TextureView>,
        physical_size: PhysicalSize,
    },
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented {
        surface: Box<vello::util::RenderSurface<'static>>,
        valid: bool,
        resizing: bool,
        pending_physical_size: Option<PhysicalSize>,
    },
}

#[derive(Clone, Debug)]
pub enum Attachment {
    Headless,
    #[cfg(feature = "render-window")]
    Window(surgeist_window::Handle),
    WebCanvas(WebCanvas),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttachmentKind {
    Headless,
    #[cfg(feature = "render-window")]
    Window,
    WebCanvas,
}

#[derive(Clone, Debug)]
pub struct WebCanvas {
    id: String,
    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    canvas: Option<wgpu::web_sys::HtmlCanvasElement>,
}

impl Attachment {
    #[must_use]
    pub fn from_web_canvas(id: impl Into<String>) -> Self {
        Self::WebCanvas(WebCanvas::new(id))
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    #[must_use]
    pub fn from_html_canvas(
        id: impl Into<String>,
        canvas: wgpu::web_sys::HtmlCanvasElement,
    ) -> Self {
        Self::WebCanvas(WebCanvas::from_html_canvas(id, canvas))
    }

    #[cfg(feature = "render-window")]
    #[must_use]
    pub fn from_window(handle: surgeist_window::Handle) -> Self {
        Self::Window(handle)
    }

    pub(crate) const fn kind(&self) -> AttachmentKind {
        match self {
            Self::Headless => AttachmentKind::Headless,
            #[cfg(feature = "render-window")]
            Self::Window(_) => AttachmentKind::Window,
            Self::WebCanvas(_) => AttachmentKind::WebCanvas,
        }
    }
}

impl WebCanvas {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
            canvas: None,
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    #[must_use]
    pub fn from_html_canvas(
        id: impl Into<String>,
        canvas: wgpu::web_sys::HtmlCanvasElement,
    ) -> Self {
        Self {
            id: id.into(),
            canvas: Some(canvas),
        }
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceOptions {
    pub size: Size,
    pub scale: f64,
    pub present_mode: PresentMode,
    pub format: Format,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self {
            size: Size::new(1.0, 1.0),
            scale: 1.0,
            present_mode: PresentMode::Auto,
            format: Format::Rgba8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PresentMode {
    #[default]
    Auto,
    Fifo,
    Mailbox,
    Immediate,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Format {
    #[default]
    Rgba8,
    Bgra8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    pub base_color: Color,
    pub debug: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            base_color: Color::TRANSPARENT,
            debug: false,
        }
    }
}

impl From<PresentMode> for wgpu::PresentMode {
    fn from(mode: PresentMode) -> Self {
        match mode {
            PresentMode::Auto => Self::AutoVsync,
            PresentMode::Fifo => Self::Fifo,
            PresentMode::Mailbox => Self::Mailbox,
            PresentMode::Immediate => Self::Immediate,
        }
    }
}

impl From<Format> for wgpu::TextureFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::Rgba8 => Self::Rgba8Unorm,
            Format::Bgra8 => Self::Bgra8Unorm,
        }
    }
}
