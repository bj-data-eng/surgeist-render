use super::{
    Color, Error, ErrorCode, PhysicalSize, Result, Size, geometry::physical_size, validation::*,
};

pub struct Surface {
    pub(crate) attachment: Attachment,
    pub(crate) options: SurfaceOptions,
    pub(crate) state: SurfaceState,
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
            state: SurfaceState::Available,
            last_parameters: None,
            backend,
        }
    }

    pub fn resize(&mut self, size: Size, scale: f64) -> Result<()> {
        validate_size(size, "surface size")?;
        validate_positive_f64(scale, "surface scale")?;
        let next = physical_size(size, scale)?;
        self.options.size = size;
        self.options.scale = scale;
        match &mut self.backend {
            SurfaceBackend::ContractOnly { physical_size } => {
                *physical_size = next;
            }
            SurfaceBackend::Headless {
                resources,
                physical_size,
                ..
            } => {
                if *physical_size == next {
                    return Ok(());
                }
                *physical_size = next;
                *resources = HeadlessResources::Pending;
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented {
                surface, lifecycle, ..
            } => {
                let current = PhysicalSize::new(surface.config.width, surface.config.height);
                *lifecycle = lifecycle.resize_requested(current, next);
            }
        }
        Ok(())
    }

    pub fn suspend(&mut self) -> Result<()> {
        self.state = SurfaceState::Suspended;
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
        self.state = SurfaceState::Available;
        Ok(())
    }

    #[must_use]
    pub const fn state(&self) -> SurfaceState {
        self.state
    }

    pub(crate) fn ensure_available(&self) -> Result<()> {
        if self.state == SurfaceState::Suspended {
            return Err(Error::new(
                ErrorCode::SurfaceUnavailable,
                "surface is suspended",
            ));
        }
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
            SurfaceBackend::Presented { surface, .. } => {
                PhysicalSize::new(surface.config.width, surface.config.height)
            }
        }
    }

    #[must_use]
    pub const fn resource_state(&self) -> SurfaceResourceState {
        match &self.backend {
            SurfaceBackend::ContractOnly { .. } => SurfaceResourceState::ContractOnly,
            SurfaceBackend::Headless { resources, .. } => match resources {
                HeadlessResources::Pending => SurfaceResourceState::PendingAllocation,
                HeadlessResources::Ready { .. } => SurfaceResourceState::Ready,
            },
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => SurfaceResourceState::Presented,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceState {
    Available,
    Suspended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceResourceState {
    ContractOnly,
    PendingAllocation,
    Ready,
    Presented,
}

pub(crate) enum SurfaceBackend {
    ContractOnly {
        physical_size: PhysicalSize,
    },
    Headless {
        dev_id: usize,
        resources: HeadlessResources,
        physical_size: PhysicalSize,
    },
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented {
        surface: Box<vello::util::RenderSurface<'static>>,
        lifecycle: PresentedLifecycle,
    },
}

pub(crate) enum HeadlessResources {
    Pending,
    Ready {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResizeState {
    Idle,
    Resizing,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentedLifecycle {
    Ready {
        resizing: ResizeState,
    },
    ResizePending {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    NonRenderable {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    Occluded {
        resizing: ResizeState,
    },
    Lost,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedLifecycle {
    pub(crate) const fn resizing(self) -> bool {
        matches!(
            self,
            Self::Ready {
                resizing: ResizeState::Resizing
            } | Self::ResizePending {
                resizing: ResizeState::Resizing,
                ..
            } | Self::NonRenderable {
                resizing: ResizeState::Resizing,
                ..
            } | Self::Occluded {
                resizing: ResizeState::Resizing
            }
        )
    }

    pub(crate) const fn resize_state(self) -> ResizeState {
        if self.resizing() {
            ResizeState::Resizing
        } else {
            ResizeState::Idle
        }
    }

    pub(crate) const fn physical_size(self) -> Option<PhysicalSize> {
        match self {
            Self::ResizePending { physical_size, .. }
            | Self::NonRenderable { physical_size, .. } => Some(physical_size),
            Self::Ready { .. } | Self::Occluded { .. } | Self::Lost => None,
        }
    }

    pub(crate) const fn with_resizing(self, resizing: ResizeState) -> Self {
        match self {
            Self::Ready { .. } => Self::Ready { resizing },
            Self::ResizePending { physical_size, .. } => Self::ResizePending {
                physical_size,
                resizing,
            },
            Self::NonRenderable { physical_size, .. } => Self::NonRenderable {
                physical_size,
                resizing,
            },
            Self::Occluded { .. } => Self::Occluded { resizing },
            Self::Lost => Self::Lost,
        }
    }

    pub(crate) fn resize_requested(self, current: PhysicalSize, next: PhysicalSize) -> Self {
        let resizing = self.resize_state();
        if next.width() == 0 || next.height() == 0 {
            return Self::NonRenderable {
                physical_size: next,
                resizing,
            };
        }
        if matches!(self, Self::NonRenderable { .. }) && current == next {
            return Self::Ready { resizing };
        }
        if current == next || self.physical_size() == Some(next) {
            return self;
        }
        Self::ResizePending {
            physical_size: next,
            resizing,
        }
    }
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
