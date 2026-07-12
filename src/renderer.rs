#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, ResizeState};
use super::{
    backend::*,
    command::{RenderCommand, RenderCommands},
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
        if let SurfaceBackend::Presented { lifecycle, .. } = &mut surface.backend {
            let next = if resizing {
                ResizeState::Resizing
            } else {
                ResizeState::Idle
            };
            if lifecycle.resize_state() == next {
                return Ok(());
            }
            *lifecycle = lifecycle.with_resizing(next);
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
        let normalized = RenderCommands::new(self.materialize_resolved_backdrops(
            normalized.commands,
            surface.scale(),
            surface.options.format,
            parameters,
        )?);
        let normalized = RenderCommands::new(self.materialize_resolved_layer_masks(
            normalized.commands,
            surface.scale(),
            surface.options.format,
            parameters,
        )?);
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

        if parameters.debug || self.options.debug() {
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

    pub(crate) fn default_wgpu_device_queue(&mut self) -> Option<(&wgpu::Device, &wgpu::Queue)> {
        let backend = self.backend.as_mut()?;
        let dev_id = self.default_device?;
        let device_handle = &backend.context.devices[dev_id];
        Some((&device_handle.device, &device_handle.queue))
    }

    pub(crate) fn default_offscreen_render_context(
        &mut self,
    ) -> Option<OffscreenRenderGpuContext<'_>> {
        let backend = self.backend.as_mut()?;
        let dev_id = self.default_device?;
        Some(OffscreenRenderGpuContext::new(backend, dev_id))
    }

    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    #[must_use]
    pub const fn options(&self) -> Options {
        self.options
    }

    fn materialize_resolved_backdrops(
        &mut self,
        commands: Vec<RenderCommand>,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Result<Vec<RenderCommand>> {
        commands
            .into_iter()
            .map(|command| self.materialize_resolved_backdrop(command, scale, format, parameters))
            .collect()
    }

    fn materialize_resolved_backdrop(
        &mut self,
        command: RenderCommand,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Result<RenderCommand> {
        let RenderCommand::Layer {
            mut layer,
            children,
        } = command
        else {
            return Ok(command);
        };
        let mut children =
            self.materialize_resolved_backdrops(children, scale, format, parameters)?;
        let Some(backdrop) = layer.backdrop.clone() else {
            return Ok(RenderCommand::Layer { layer, children });
        };

        reject_backdrop_execution(backdrop.source_commands())?;
        let source_commands = self.materialize_resolved_layer_masks(
            backdrop.source_commands().to_vec(),
            scale,
            format,
            parameters,
        )?;
        let bounds = backdrop.capture_bounds();
        let physical_size = physical_size(bounds.rect().size(), scale)?;
        let local_scene = self.backdrop_source_scene(&layer, source_commands, bounds, scale)?;
        let request = OffscreenLocalSceneRenderRequest::new(bounds, scale, format, parameters);
        let options = self.options;
        let mut cache = OffscreenTextureResourceCache::new();
        let Some(context) = self.default_offscreen_render_context() else {
            return Err(Error::new(
                ErrorCode::AdapterUnavailable,
                "materialized backdrop captures require an available wgpu device context",
            ));
        };
        let rendered = render_vello_local_scene_to_offscreen_texture(
            Some(context),
            options,
            &mut cache,
            &local_scene,
            request,
        )?;
        let source = {
            let Some((device, queue)) = self.default_wgpu_device_queue() else {
                return Err(Error::new(
                    ErrorCode::AdapterUnavailable,
                    "materialized backdrop captures require an available wgpu device queue",
                ));
            };
            read_texture_rgba(device, queue, rendered.texture(), physical_size)?
        };
        rendered.release(&mut cache)?;
        let filtered = image::ResolvedMaterializedImageFilterExecution::try_new_for_image_buffer(
            backdrop.filters(),
            &source,
        )?
        .execute_to_image_buffer()?;
        let image = Image::from_rgba(
            Size::new(
                f64::from(filtered.size.width()),
                f64::from(filtered.size.height()),
            ),
            filtered.rgba,
        )?;
        let image_command = RenderCommand::Image {
            image,
            rect: bounds.rect(),
            fit: ImageFit::Stretch,
        };
        let backdrop_command = if let Some(clip) = backdrop.clip().cloned() {
            RenderCommand::Layer {
                layer: command::NormalizedLayer {
                    clip: Some(clip),
                    transform: Transform::identity(),
                    opacity: 1.0,
                    blend: BlendMode::Normal,
                    mask: None,
                    backdrop: None,
                    isolation: command::LayerIsolation::ClipOnly,
                    pass_plan: layer.pass_plan,
                },
                children: vec![image_command],
            }
        } else {
            image_command
        };
        children.insert(0, backdrop_command);
        layer.backdrop = None;

        Ok(RenderCommand::Layer { layer, children })
    }

    fn materialize_resolved_layer_masks(
        &mut self,
        commands: Vec<RenderCommand>,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Result<Vec<RenderCommand>> {
        commands
            .into_iter()
            .map(|command| self.materialize_resolved_layer_mask(command, scale, format, parameters))
            .collect()
    }

    fn materialize_resolved_layer_mask(
        &mut self,
        command: RenderCommand,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Result<RenderCommand> {
        let RenderCommand::Layer { layer, children } = command else {
            return Ok(command);
        };
        let children =
            self.materialize_resolved_layer_masks(children, scale, format, parameters)?;
        if layer.backdrop.is_some() {
            return Err(backdrop_execution_error());
        }
        let Some(mask) = layer.mask.clone() else {
            return Ok(RenderCommand::Layer { layer, children });
        };

        let bounds = layer.pass_plan.bounds().ok_or_else(|| {
            Error::invalid_value(
                "materialized masked layer bounds",
                "unknown",
                "must be explicit before rendering resolved layer alpha masks",
            )
        })?;
        let physical_size = physical_size(bounds.rect().size(), scale)?;
        if mask.alpha_mask().size != physical_size {
            return Err(Error::invalid_value(
                "resolved layer alpha mask size",
                format!(
                    "{}x{}",
                    mask.alpha_mask().size.width(),
                    mask.alpha_mask().size.height()
                ),
                "must match the offscreen layer bounds in device pixels",
            ));
        }

        let local_scene = self.mask_source_scene(&layer, children, bounds, scale)?;
        let request = OffscreenLocalSceneRenderRequest::new(bounds, scale, format, parameters);
        let options = self.options;
        let mut cache = OffscreenTextureResourceCache::new();
        let Some(context) = self.default_offscreen_render_context() else {
            return Err(Error::new(
                ErrorCode::AdapterUnavailable,
                "resolved layer alpha masks require an available wgpu device context",
            ));
        };
        let rendered = render_vello_local_scene_to_offscreen_texture(
            Some(context),
            options,
            &mut cache,
            &local_scene,
            request,
        )?;
        let source = {
            let Some((device, queue)) = self.default_wgpu_device_queue() else {
                return Err(Error::new(
                    ErrorCode::AdapterUnavailable,
                    "resolved layer alpha masks require an available wgpu device queue",
                ));
            };
            read_texture_rgba(device, queue, rendered.texture(), physical_size)?
        };
        rendered.release(&mut cache)?;
        let masked = ResolvedAlphaMaskExecution::try_new(&source, mask.alpha_mask())?
            .execute_to_image_buffer()?;
        let image = Image::from_rgba(
            Size::new(
                f64::from(masked.size.width()),
                f64::from(masked.size.height()),
            ),
            masked.rgba,
        )?;
        let image_command = RenderCommand::Image {
            image,
            rect: bounds.rect(),
            fit: ImageFit::Stretch,
        };
        Ok(RenderCommand::Layer {
            layer: command::NormalizedLayer {
                clip: None,
                mask: None,
                backdrop: None,
                ..layer
            },
            children: vec![image_command],
        })
    }

    fn mask_source_scene(
        &self,
        layer: &command::NormalizedLayer,
        children: Vec<RenderCommand>,
        bounds: command::OffscreenBounds,
        scale: f64,
    ) -> Result<vello::Scene> {
        let source_layer = command::NormalizedLayer {
            clip: layer.clip.clone(),
            transform: Transform::translation(-bounds.rect().x(), -bounds.rect().y())?,
            opacity: 1.0,
            blend: BlendMode::Normal,
            mask: None,
            backdrop: None,
            isolation: if layer.clip.is_some() {
                command::LayerIsolation::ClipOnly
            } else {
                command::LayerIsolation::None
            },
            pass_plan: layer.pass_plan,
        };
        let commands = RenderCommands::new(vec![RenderCommand::Layer {
            layer: source_layer,
            children,
        }]);
        encode_vello_scene(&commands, scale)
    }

    fn backdrop_source_scene(
        &self,
        layer: &command::NormalizedLayer,
        source_commands: Vec<RenderCommand>,
        bounds: command::OffscreenBounds,
        scale: f64,
    ) -> Result<vello::Scene> {
        let source_layer = command::NormalizedLayer {
            clip: None,
            transform: Transform::translation(-bounds.rect().x(), -bounds.rect().y())?,
            opacity: 1.0,
            blend: BlendMode::Normal,
            mask: None,
            backdrop: None,
            isolation: command::LayerIsolation::None,
            pass_plan: layer.pass_plan,
        };
        let commands = RenderCommands::new(vec![RenderCommand::Layer {
            layer: source_layer,
            children: source_commands,
        }]);
        encode_vello_scene(&commands, scale)
    }

    #[must_use]
    pub const fn capabilities(&self) -> Capabilities {
        Capabilities::VELLO_0_9
    }
}

fn reject_backdrop_execution(commands: &[RenderCommand]) -> Result<()> {
    for command in commands {
        let RenderCommand::Layer { layer, children } = command else {
            continue;
        };
        if layer.backdrop.is_some() {
            return Err(backdrop_execution_error());
        }
        reject_backdrop_execution(children)?;
    }
    Ok(())
}

fn backdrop_execution_error() -> Error {
    let mut error = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropExecution,
    ));
    error.message.push_str(
        ": backdrop capture was planned during normalization but render-time backdrop execution is not implemented",
    );
    error
}

/// Renderer configuration that is fixed when a [`Renderer`] is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    antialiasing: Antialiasing,
    debug: bool,
    effect_quality_policy: EffectQualityPolicy,
    resource_cache_budget: ResourceCacheBudget,
}

impl Options {
    /// Creates the default GPU-only renderer configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            antialiasing: Antialiasing::Area,
            debug: false,
            effect_quality_policy: EffectQualityPolicy::RequireHighPrecision,
            resource_cache_budget: ResourceCacheBudget::DEFAULT,
        }
    }

    /// Returns the configured antialiasing method.
    #[must_use]
    pub const fn antialiasing(self) -> Antialiasing {
        self.antialiasing
    }

    /// Returns this configuration with a different antialiasing method.
    #[must_use]
    pub const fn with_antialiasing(mut self, antialiasing: Antialiasing) -> Self {
        self.antialiasing = antialiasing;
        self
    }

    /// Returns whether renderer diagnostics are enabled.
    #[must_use]
    pub const fn debug(self) -> bool {
        self.debug
    }

    /// Returns this configuration with renderer diagnostics enabled or disabled.
    #[must_use]
    pub const fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Returns the policy for effect precision when high precision is unavailable.
    #[must_use]
    pub const fn effect_quality_policy(self) -> EffectQualityPolicy {
        self.effect_quality_policy
    }

    /// Returns this configuration with a different effect precision policy.
    #[must_use]
    pub const fn with_effect_quality_policy(
        mut self,
        effect_quality_policy: EffectQualityPolicy,
    ) -> Self {
        self.effect_quality_policy = effect_quality_policy;
        self
    }

    /// Returns the maximum retained idle effect-resource cache budget.
    #[must_use]
    pub const fn resource_cache_budget(self) -> ResourceCacheBudget {
        self.resource_cache_budget
    }

    /// Returns this configuration with a different idle effect-resource cache budget.
    #[must_use]
    pub const fn with_resource_cache_budget(
        mut self,
        resource_cache_budget: ResourceCacheBudget,
    ) -> Self {
        self.resource_cache_budget = resource_cache_budget;
        self
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy for choosing effect precision on a compatible GPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EffectQualityPolicy {
    /// Require high-precision effect execution.
    #[default]
    RequireHighPrecision,
    /// Prefer high precision and allow reduced precision only when it is unavailable.
    AllowReducedPrecision,
}

/// Byte budget for retaining idle effect resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceCacheBudget(u64);

impl ResourceCacheBudget {
    /// Disables idle effect-resource retention.
    pub const DISABLED: Self = Self(0);

    /// Retains up to 64 MiB of idle effect resources by default.
    pub const DEFAULT: Self = Self(64 * 1024 * 1024);

    /// Creates an idle effect-resource retention budget in bytes.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    /// Returns this retention budget in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.0
    }
}

impl Default for ResourceCacheBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Antialiasing {
    #[default]
    Area,
    Msaa8,
    Msaa16,
}
