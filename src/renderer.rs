#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, ResizeState};
use super::{
    backend::*,
    encode::encode_vello_scene,
    geometry::physical_size,
    stats::collect_render_stats,
    surface::{HeadlessResources, SurfaceBackend},
    validation::*,
    *,
};
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

pub struct Renderer {
    options: Options,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    backend: Option<Backend>,
    default_device: Option<usize>,
}

impl Renderer {
    pub async fn new(options: Options) -> Result<Self> {
        let mut context = vello::util::RenderContext::new();
        let default_device = context.device(None).await;
        let backend = default_device.map(|_| Backend {
            context,
            renderers: Vec::new(),
        });

        Ok(Self {
            options,
            stats: Stats::default(),
            uploaded_images: HashSet::new(),
            backend,
            default_device,
        })
    }

    pub fn create_surface(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        match attachment {
            Attachment::Headless => self.create_headless_surface(options),
            Attachment::WebCanvas(canvas) => self.create_web_canvas_surface(canvas, options),
            #[cfg(feature = "render-window")]
            Attachment::Window(handle) => {
                let Some(backend) = self.backend.as_mut() else {
                    return Err(Error::new(
                        ErrorCode::AdapterUnavailable,
                        "no compatible wgpu adapter is available",
                    ));
                };
                let physical_size = physical_size(options.size, options.scale)?;
                let surface = pollster::block_on(backend.context.create_surface(
                    handle.clone(),
                    physical_size.width(),
                    physical_size.height(),
                    options.present_mode.into(),
                ))
                .map_err(|source| {
                    Error::new(
                        ErrorCode::SurfaceCreateFailed,
                        "failed to create native surface",
                    )
                    .with_source(source)
                })?;
                let dev_id = surface.dev_id;
                ensure_vello_renderer(backend, self.options, dev_id)?;
                Ok(Surface::with_backend(
                    Attachment::Window(handle),
                    options,
                    SurfaceBackend::Presented {
                        surface: Box::new(surface),
                        lifecycle: PresentedLifecycle::Ready {
                            resizing: ResizeState::Idle,
                        },
                    },
                ))
            }
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        let Some(html_canvas) = canvas.canvas.clone() else {
            return Err(Error::new(
                ErrorCode::SurfaceCreateFailed,
                format!("web canvas surface '{}' has no canvas handle", canvas.id),
            ));
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::new(
                ErrorCode::AdapterUnavailable,
                "no compatible WebGPU adapter is available",
            ));
        };
        let physical_size = physical_size(options.size, options.scale)?;
        let surface = pollster::block_on(backend.context.create_surface(
            html_canvas,
            physical_size.width(),
            physical_size.height(),
            options.present_mode.into(),
        ))
        .map_err(|source| {
            Error::new(
                ErrorCode::SurfaceCreateFailed,
                "failed to create web canvas surface",
            )
            .with_source(source)
        })?;
        let dev_id = surface.dev_id;
        ensure_vello_renderer(backend, self.options, dev_id)?;
        Ok(Surface::with_backend(
            Attachment::WebCanvas(canvas),
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                lifecycle: PresentedLifecycle::Ready {
                    resizing: ResizeState::Idle,
                },
            },
        ))
    }

    #[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
    fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        _options: SurfaceOptions,
    ) -> Result<Surface> {
        let _ = canvas;
        Capabilities::VELLO_0_9.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))?;
        unreachable!("web canvas support requires the render-web feature on wasm32");
    }

    pub fn create_headless(&mut self, size: Size, scale: f64) -> Result<Surface> {
        let options = SurfaceOptions {
            size,
            scale,
            ..SurfaceOptions::default()
        };
        self.create_headless_surface(options)
    }

    fn create_headless_surface(&mut self, options: SurfaceOptions) -> Result<Surface> {
        validate_surface_options(options)?;
        if options.format != Format::Rgba8 {
            return Err(Error::new(
                ErrorCode::SurfaceCreateFailed,
                "headless surfaces require Rgba8 format for Vello storage rendering",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend =
            if let (Some(backend), Some(dev_id)) = (self.backend.as_mut(), self.default_device) {
                ensure_vello_renderer(backend, self.options, dev_id)?;
                let (texture, view) = create_headless_texture(
                    &backend.context.devices[dev_id].device,
                    physical_size,
                    options.format,
                )?;
                SurfaceBackend::Headless {
                    dev_id,
                    resources: HeadlessResources::Ready { texture, view },
                    physical_size,
                }
            } else {
                SurfaceBackend::ContractOnly { physical_size }
            };

        Ok(Surface::with_backend(
            Attachment::Headless,
            options,
            backend,
        ))
    }

    pub fn set_surface_resizing(&mut self, surface: &mut Surface, resizing: bool) -> Result<()> {
        surface.ensure_available()?;

        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = resizing;

        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented {
            surface: native,
            lifecycle,
            ..
        } = &mut surface.backend
        {
            let next = if resizing {
                ResizeState::Resizing
            } else {
                ResizeState::Idle
            };
            if lifecycle.resize_state() == next {
                return Ok(());
            }
            *lifecycle = lifecycle.with_resizing(next);
            apply_presented_resize_state(self.backend.as_mut(), native, resizing);
        }

        Ok(())
    }

    pub fn render(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<Stats> {
        surface.ensure_available()?;

        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let mut stats = Stats {
            encode_time: Duration::ZERO,
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let normalized = scene.normalize(self.capabilities())?;
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
        let vello_scene = encode_vello_scene(&normalized, surface.scale())?;
        stats.encode_time = encode_start.elapsed();

        if let Some(backend) = self.backend.as_mut() {
            let timings =
                render_vello_surface(backend, self.options, surface, &vello_scene, parameters)?;
            stats.render_time = timings.render_time;
            stats.present_time = timings.present_time;
        }
        stats.frame_time = frame_start.elapsed();

        if parameters.debug || self.options.debug {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        self.stats = stats;
        self.uploaded_images = uploaded_images;
        surface.last_parameters = Some(parameters);
        Ok(stats)
    }

    pub fn resume_surface(&mut self, surface: &mut Surface, attachment: Attachment) -> Result<()> {
        if surface.attachment.kind() != attachment.kind() {
            return Err(Error::new(
                ErrorCode::SurfaceCreateFailed,
                "surface cannot resume with an incompatible attachment",
            ));
        }

        match &surface.backend {
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => {
                let mut next = self.create_surface(attachment, surface.options)?;
                next.last_parameters = surface.last_parameters;
                *surface = next;
                Ok(())
            }
            SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => {
                surface.resume(attachment)
            }
        }
    }

    pub fn read_headless(&mut self, surface: &Surface) -> Result<ImageBuffer> {
        let SurfaceBackend::Headless {
            dev_id,
            resources: HeadlessResources::Ready { texture, .. },
            physical_size,
            ..
        } = &surface.backend
        else {
            return Err(Error::new(
                ErrorCode::UnsupportedBackend,
                "only rendered headless surfaces can be read back",
            ));
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::new(
                ErrorCode::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        let device_handle = &backend.context.devices[*dev_id];
        read_texture_rgba(
            &device_handle.device,
            &device_handle.queue,
            texture,
            *physical_size,
        )
    }

    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    #[must_use]
    pub const fn options(&self) -> Options {
        self.options
    }

    #[must_use]
    pub const fn capabilities(&self) -> Capabilities {
        Capabilities::VELLO_0_9
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Options {
    pub antialiasing: Antialiasing,
    /// Uses Vello's CPU pipeline stages where Vello supports them.
    ///
    /// This is a diagnostic/debug option, not a lower-memory rendering mode.
    /// Vello still uses GPU presentation resources, and CPU mode can increase
    /// total resident memory.
    pub use_cpu: bool,
    pub debug: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antialiasing {
    #[default]
    Area,
    Msaa8,
    Msaa16,
}
