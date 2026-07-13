use super::backend::DeviceSlotIdentity;
use super::{
    BackendErrorCode, Color, Error, PhysicalSize, RenderSurfaceAvailability, Result,
    RuntimeCapabilityUnavailableReason, RuntimeOperation, Size, geometry::physical_size,
    validation::*,
};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct RendererIdentity(Arc<()>);

impl RendererIdentity {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub struct Surface {
    pub(crate) attachment: Attachment,
    pub(crate) options: SurfaceOptions,
    pub(crate) state: SurfaceState,
    pub(crate) last_parameters: Option<Parameters>,
    pub(crate) backend: SurfaceBackend,
    pub(crate) renderer_identity: RendererIdentity,
}

impl Surface {
    pub(crate) fn with_backend(
        attachment: Attachment,
        options: SurfaceOptions,
        backend: SurfaceBackend,
        renderer_identity: RendererIdentity,
    ) -> Self {
        Self {
            attachment,
            options,
            state: SurfaceState::Available,
            last_parameters: None,
            backend,
            renderer_identity,
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
                *resources = HeadlessResources::for_physical_size(next);
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
                BackendErrorCode::SurfaceCreateFailed,
                "surface cannot resume with an incompatible attachment",
            ));
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { .. } = &self.backend {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
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

    pub(crate) fn ensure_available(&self, operation: RuntimeOperation) -> Result<()> {
        if self.state == SurfaceState::Suspended {
            return Err(Error::runtime_unavailable(
                operation,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended,
                },
                "surface is suspended",
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_renderable(&self) -> Result<()> {
        let unavailable = match &self.backend {
            SurfaceBackend::ContractOnly { physical_size }
            | SurfaceBackend::Headless { physical_size, .. }
                if physical_size.width() == 0 || physical_size.height() == 0 =>
            {
                Some(RenderSurfaceAvailability::NonRenderable)
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { lifecycle, .. } => match lifecycle {
                PresentedLifecycle::NonRenderable { .. } => {
                    Some(RenderSurfaceAvailability::NonRenderable)
                }
                PresentedLifecycle::Occluded { .. } => Some(RenderSurfaceAvailability::Occluded),
                PresentedLifecycle::Lost => Some(RenderSurfaceAvailability::Lost),
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::ResizePending { .. } => None,
            },
            SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => None,
        };
        match unavailable {
            Some(state) => Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable { state },
                "surface is not renderable",
            )),
            None => Ok(()),
        }
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
                HeadlessResources::Empty => SurfaceResourceState::Empty,
                HeadlessResources::Pending => SurfaceResourceState::PendingAllocation,
                HeadlessResources::Published { .. } => SurfaceResourceState::Ready,
            },
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => SurfaceResourceState::Presented,
        }
    }

    pub(crate) const fn device_identity(&self) -> Option<DeviceSlotIdentity> {
        match &self.backend {
            SurfaceBackend::ContractOnly { .. } => None,
            SurfaceBackend::Headless {
                device_identity, ..
            } => Some(*device_identity),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented {
                device_identity, ..
            } => Some(*device_identity),
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
    Empty,
    PendingAllocation,
    Ready,
    Presented,
}

pub(crate) enum SurfaceBackend {
    ContractOnly {
        physical_size: PhysicalSize,
    },
    Headless {
        device_identity: DeviceSlotIdentity,
        resources: HeadlessResources,
        physical_size: PhysicalSize,
    },
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented {
        surface: Box<PresentedSurface>,
        device_identity: DeviceSlotIdentity,
        lifecycle: PresentedLifecycle,
    },
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) struct PresentedSurface {
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) target_texture: wgpu::Texture,
    pub(crate) target_view: wgpu::TextureView,
    pub(crate) blitter: wgpu::util::TextureBlitter,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedSurface {
    pub(crate) fn new(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        physical_size: PhysicalSize,
        present_mode: wgpu::PresentMode,
    ) -> Result<Self> {
        let format = surface
            .get_capabilities(adapter)
            .formats
            .into_iter()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                )
            })
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::SurfaceCreateFailed,
                    "the selected adapter does not support an Rgba8 or Bgra8 surface format",
                )
            })?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physical_size.width(),
            height: physical_size.height(),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let (target_texture, target_view) = Self::create_targets(physical_size, device);
        let presented = Self {
            surface,
            config,
            format,
            target_texture,
            target_view,
            blitter: wgpu::util::TextureBlitter::new(device, format),
        };
        presented.configure(device);
        Ok(presented)
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, physical_size: PhysicalSize) {
        let (target_texture, target_view) = Self::create_targets(physical_size, device);
        self.target_texture = target_texture;
        self.target_view = target_view;
        self.config.width = physical_size.width();
        self.config.height = physical_size.height();
        self.configure(device);
    }

    pub(crate) fn configure(&self, device: &wgpu::Device) {
        debug_assert_eq!(self.format, self.config.format);
        self.surface.configure(device, &self.config);
    }

    fn create_targets(
        physical_size: PhysicalSize,
        device: &wgpu::Device,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: physical_size.width(),
                height: physical_size.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

pub(crate) enum HeadlessResources {
    Empty,
    Pending,
    Published {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
}

impl HeadlessResources {
    pub(crate) const fn for_physical_size(physical_size: PhysicalSize) -> Self {
        if physical_size.width() == 0 || physical_size.height() == 0 {
            Self::Empty
        } else {
            Self::Pending
        }
    }
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

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    pub(crate) fn html_canvas(&self) -> Option<wgpu::web_sys::HtmlCanvasElement> {
        self.canvas.clone()
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

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
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
