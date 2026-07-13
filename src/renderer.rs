#[cfg(test)]
use super::gpu_transaction::GpuOperationTransaction;
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
    gpu_transaction::{GpuOperationDraft, GpuOperationStage},
    stats::collect_render_stats,
    surface::{HeadlessResources, RendererIdentity, SurfaceBackend},
    validation::*,
    *,
};
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

pub struct Renderer {
    identity: RendererIdentity,
    options: Options,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    backend: Option<Backend>,
    default_device: Option<DeviceSlotIdentity>,
}

impl Renderer {
    pub async fn new(options: Options) -> Result<Self> {
        let mut backend = Backend::new();
        let default_device = backend.select_device(None).await?;
        let backend = default_device.map(|_| backend);

        Ok(Self {
            identity: RendererIdentity::new(),
            options,
            stats: Stats::default(),
            uploaded_images: HashSet::new(),
            backend,
            default_device,
        })
    }

    /// Creates a surface and awaits any native or WebGPU surface setup.
    ///
    /// The returned surface is ready for its next lifecycle operation when this
    /// future succeeds. Invalid options and unsupported attachments preserve
    /// their existing diagnostics when the future is awaited. This future does
    /// not promise to be `Send`.
    pub async fn create_surface(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        match attachment {
            Attachment::Headless => self.create_headless_surface(options).await,
            Attachment::WebCanvas(canvas) => self.create_web_canvas_surface(canvas, options).await,
            #[cfg(feature = "render-window")]
            Attachment::Window(handle) => {
                let Some(backend) = self.backend.as_mut() else {
                    return Err(Error::new(
                        BackendErrorCode::AdapterUnavailable,
                        "no compatible wgpu adapter is available",
                    ));
                };
                let physical_size = physical_size(options.size, options.scale)?;
                let (surface, device_identity) = backend
                    .create_presented_surface(
                        handle.clone(),
                        physical_size,
                        options.present_mode.into(),
                    )
                    .await
                    .map_err(|source| {
                        Error::new(
                            BackendErrorCode::SurfaceCreateFailed,
                            "failed to create native surface",
                        )
                        .with_source(source)
                    })?;
                if let Some(error) =
                    backend.terminal_error(device_identity, RuntimeOperation::AdapterSelection)
                {
                    return Err(error);
                }
                ensure_vello_renderer(
                    backend,
                    self.options,
                    device_identity,
                    RuntimeOperation::AdapterSelection,
                )?;
                Ok(Surface::with_backend(
                    Attachment::Window(handle),
                    options,
                    SurfaceBackend::Presented {
                        surface: Box::new(surface),
                        device_identity,
                        lifecycle: PresentedLifecycle::Ready {
                            resizing: ResizeState::Idle,
                        },
                    },
                    self.identity.clone(),
                ))
            }
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    async fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        let Some(html_canvas) = canvas.canvas.clone() else {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                format!("web canvas surface '{}' has no canvas handle", canvas.id),
            ));
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::new(
                BackendErrorCode::AdapterUnavailable,
                "no compatible WebGPU adapter is available",
            ));
        };
        let physical_size = physical_size(options.size, options.scale)?;
        let (surface, device_identity) = backend
            .create_presented_surface(html_canvas, physical_size, options.present_mode.into())
            .await
            .map_err(|source| {
                Error::new(
                    BackendErrorCode::SurfaceCreateFailed,
                    "failed to create web canvas surface",
                )
                .with_source(source)
            })?;
        if let Some(error) =
            backend.terminal_error(device_identity, RuntimeOperation::AdapterSelection)
        {
            return Err(error);
        }
        ensure_vello_renderer(
            backend,
            self.options,
            device_identity,
            RuntimeOperation::AdapterSelection,
        )?;
        Ok(Surface::with_backend(
            Attachment::WebCanvas(canvas),
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                device_identity,
                lifecycle: PresentedLifecycle::Ready {
                    resizing: ResizeState::Idle,
                },
            },
            self.identity.clone(),
        ))
    }

    #[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
    async fn create_web_canvas_surface(
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

    /// Creates a headless surface for a later asynchronous render operation.
    ///
    /// Await this operation before using the surface. Input and format failures
    /// are reported when the future is awaited; readback remains synchronous.
    pub async fn create_headless(&mut self, size: Size, scale: f64) -> Result<Surface> {
        let options = SurfaceOptions {
            size,
            scale,
            ..SurfaceOptions::default()
        };
        self.create_headless_surface(options).await
    }

    async fn create_headless_surface(&mut self, options: SurfaceOptions) -> Result<Surface> {
        validate_surface_options(options)?;
        if options.format != Format::Rgba8 {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "headless surfaces require Rgba8 format for Vello storage rendering",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = if let (Some(backend), Some(device_identity)) =
            (self.backend.as_mut(), self.default_device)
        {
            if let Some(error) =
                backend.terminal_error(device_identity, RuntimeOperation::AdapterSelection)
            {
                return Err(error);
            }
            let renderer_transaction = backend.begin_gpu_operation(
                device_identity,
                GpuOperationStage::RendererCreate,
                RuntimeOperation::AdapterSelection,
            )?;
            let renderer_work = ensure_vello_renderer(
                backend,
                self.options,
                device_identity,
                RuntimeOperation::AdapterSelection,
            );
            let renderer_scope = renderer_transaction
                .finish(RuntimeOperation::AdapterSelection)
                .await;
            backend.observe_device_terminal(device_identity);
            renderer_scope?;
            renderer_work?;

            let texture_transaction = backend.begin_gpu_operation(
                device_identity,
                GpuOperationStage::SurfaceCreate,
                RuntimeOperation::AdapterSelection,
            )?;
            let texture_work = (|| -> Result<(wgpu::Texture, wgpu::TextureView)> {
                let (device, _) =
                    backend.device_queue(device_identity, RuntimeOperation::AdapterSelection)?;
                create_headless_texture(device, physical_size, options.format)
            })();
            let scope_result = texture_transaction
                .finish(RuntimeOperation::AdapterSelection)
                .await;
            backend.observe_device_terminal(device_identity);
            scope_result?;
            let (texture, view) = texture_work?;
            SurfaceBackend::Headless {
                device_identity,
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
            self.identity.clone(),
        ))
    }

    pub fn set_surface_resizing(&mut self, surface: &mut Surface, resizing: bool) -> Result<()> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
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

    /// Submits one render operation for an available surface.
    ///
    /// Awaiting this future returns render statistics after scene validation and
    /// submission, or the existing lifecycle, validation, or backend diagnostic.
    pub async fn render(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<Stats> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;

        let transaction = match surface.device_identity() {
            Some(device_identity) => Some(
                self.backend
                    .as_mut()
                    .expect("device-backed surfaces require their validated backend")
                    .begin_gpu_operation(
                        device_identity,
                        GpuOperationStage::Render,
                        RuntimeOperation::SurfaceRendering,
                    )?,
            ),
            None => None,
        };

        let work = (|| -> Result<(Stats, HashSet<ImageId>)> {
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
            Ok((stats, uploaded_images))
        })();

        if let (Some(transaction), Some(device_identity)) = (transaction, surface.device_identity())
        {
            let backend = self
                .backend
                .as_mut()
                .expect("device-backed surfaces require their validated backend");
            let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
            backend.observe_device_terminal(device_identity);
            scope_result?;
        }
        let (stats, uploaded_images) = work?;
        let mut published = None;
        GpuOperationDraft::new(&mut published, (stats, uploaded_images, parameters)).commit();
        let (stats, uploaded_images, parameters) =
            published.expect("a clean GPU transaction must commit its staged public state");
        self.stats = stats;
        self.uploaded_images = uploaded_images;
        surface.last_parameters = Some(parameters);
        Ok(stats)
    }

    /// Resumes a compatible surface, awaiting recreation when it is presented.
    ///
    /// Await this operation before rendering again. Incompatible attachments and
    /// identity failures preserve their existing error ordering.
    pub async fn resume_surface(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
    ) -> Result<()> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceResume)?;
        if surface.attachment.kind() != attachment.kind() {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "surface cannot resume with an incompatible attachment",
            ));
        }
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceResume)?;

        match &surface.backend {
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => {
                let mut next = self.create_surface(attachment, surface.options).await?;
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
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceReadback)?;
        if !matches!(surface.backend, SurfaceBackend::Headless { .. }) {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "only rendered headless surfaces can be read back",
            ));
        }
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceReadback)?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceReadback)?;
        let SurfaceBackend::Headless {
            device_identity,
            resources: HeadlessResources::Ready { texture, .. },
            physical_size,
            ..
        } = &surface.backend
        else {
            unreachable!("headless backend-kind validation succeeded");
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::new(
                BackendErrorCode::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        let (device, queue) =
            backend.device_queue(*device_identity, RuntimeOperation::SurfaceReadback)?;
        read_texture_rgba(device, queue, texture, *physical_size)
    }

    /// Projects immutable capabilities of the device selected by `surface`.
    ///
    /// This query observes pending terminal device signals but performs no
    /// allocation, submission, mapping, polling, or Vello/WGPU resource call.
    #[must_use]
    pub fn runtime_capabilities(&mut self, surface: &Surface) -> RuntimeCapabilities {
        if !self.identity.matches(&surface.renderer_identity) {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::ForeignRenderer,
                },
            );
        }
        let Some(device_identity) = surface.device_identity() else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            );
        };
        let Some(backend) = self.backend.as_mut() else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        };
        if !backend.has_device_slot(device_identity) {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        }
        if let Some(reason) = backend.terminal_reason(device_identity) {
            return RuntimeCapabilities::Unavailable(reason);
        }
        if let Some(reason) = runtime_surface_unavailable_reason(surface) {
            return RuntimeCapabilities::Unavailable(reason);
        }
        let Some(capabilities) = backend.device_capabilities(device_identity) else {
            return RuntimeCapabilities::Unavailable(
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                    kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
                },
            );
        };
        RuntimeCapabilities::Available(capabilities.runtime_report(runtime_surface_format(surface)))
    }

    pub(crate) fn default_wgpu_device_queue(&mut self) -> Option<(&wgpu::Device, &wgpu::Queue)> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        backend
            .device_queue(device_identity, RuntimeOperation::SurfaceRendering)
            .ok()
    }

    pub(crate) fn default_offscreen_render_context(
        &mut self,
    ) -> Option<OffscreenRenderGpuContext<'_>> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        if backend
            .terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
            .is_some()
        {
            return None;
        }
        Some(OffscreenRenderGpuContext::new(backend, device_identity))
    }

    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    #[must_use]
    pub const fn options(&self) -> Options {
        self.options
    }

    fn validate_surface_renderer_identity(
        &self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        if self.identity.matches(&surface.renderer_identity) {
            return Ok(());
        }
        Err(surface_identity_mismatch(
            operation,
            SurfaceIdentityMismatchKind::ForeignRenderer,
        ))
    }

    fn validate_surface_device_identity(
        &mut self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(device_identity) = surface.device_identity() else {
            return Ok(());
        };
        if self
            .backend
            .as_mut()
            .is_some_and(|backend| backend.has_device_slot(device_identity))
        {
            return Ok(());
        }
        Err(surface_identity_mismatch(
            operation,
            SurfaceIdentityMismatchKind::StaleDeviceGeneration,
        ))
    }

    fn validate_surface_device_terminal(
        &mut self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let Some(device_identity) = surface.device_identity() else {
            return Ok(());
        };
        if let Some(error) = self
            .backend
            .as_mut()
            .and_then(|backend| backend.terminal_error(device_identity, operation))
        {
            return Err(error);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn signal_default_device_loss_for_test(&mut self, reason: DeviceLossReason) {
        if let (Some(backend), Some(device_identity)) = (self.backend.as_mut(), self.default_device)
        {
            backend.signal_loss_for_test(device_identity, reason);
        }
    }

    #[cfg(test)]
    pub(crate) fn arm_default_terminal_signal_after_renderer_creation_for_test(&mut self) {
        if let (Some(backend), Some(device_identity)) = (self.backend.as_mut(), self.default_device)
        {
            backend.arm_terminal_signal_after_renderer_creation_for_test(device_identity);
        }
    }

    #[cfg(test)]
    pub(crate) fn default_device_renderer_released_for_test(&mut self) -> bool {
        match (self.backend.as_mut(), self.default_device) {
            (Some(backend), Some(device_identity)) => {
                backend.renderer_released_for_test(device_identity)
            }
            (None, None) => true,
            _ => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn default_ready_device_state_borrow_for_test(
        &mut self,
    ) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .ready_device_state_borrow_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) async fn deliberate_validation_error_for_test(&mut self) -> Result<Result<()>> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (device, _) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
        let _ = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist deliberate scoped validation failure"),
            size: wgpu::Extent3d {
                width: 0,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
        backend.observe_device_terminal(device_identity);
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) async fn scoped_clear_fill_probe_for_test(&mut self) -> Result<ImageBuffer> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (result, destination_texture) = {
            use super::shader::{
                RectPassBounds, RectShaderPassDescriptor, RectShaderPassExecution,
                RectShaderPassGpuContext, RectShaderPassKind, encode_clear_fill_pass,
            };
            use super::texture::{TextureDescriptor, TextureUsageIntent};

            let source = TextureDescriptor::try_new(
                PhysicalSize::new(2, 2),
                Format::Rgba8,
                TextureUsageIntent::OffscreenLayer,
            )?;
            let destination = TextureDescriptor::try_new(
                PhysicalSize::new(2, 2),
                Format::Rgba8,
                TextureUsageIntent::IntermediatePass,
            )?;
            let bounds = RectPassBounds::try_new(0, 0, 2, 2, source, destination)?;
            let pass = RectShaderPassDescriptor::try_new(
                "scoped source",
                "scoped destination",
                source,
                destination,
                bounds,
                RectShaderPassKind::ClearFill,
            )?;
            let (device, queue) =
                backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
            let (_source_texture, source_view) =
                create_texture(device, "Surgeist scoped shader source", source);
            let (destination_texture, destination_view) =
                create_texture(device, "Surgeist scoped shader destination", destination);
            let context =
                RectShaderPassGpuContext::new(device, queue, &source_view, &destination_view);
            (
                encode_clear_fill_pass(
                    RectShaderPassExecution::gpu(context, transaction),
                    pass,
                    Color::try_rgba(0.25, 0.5, 0.75, 1.0)?,
                )
                .await,
                destination_texture,
            )
        };
        backend.observe_device_terminal(device_identity);
        result?;
        let (device, queue) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceReadback)?;
        read_texture_rgba(device, queue, &destination_texture, PhysicalSize::new(2, 2))
    }

    #[cfg(test)]
    pub(crate) fn start_default_gpu_operation_for_test(
        &mut self,
    ) -> Option<GpuOperationTransaction> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .begin_gpu_operation(
                device_identity,
                GpuOperationStage::Render,
                RuntimeOperation::SurfaceRendering,
            )
            .ok()
    }

    #[cfg(test)]
    pub(crate) async fn submit_prepared_vello_pass_for_test(
        &mut self,
        prepared: &super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<()> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "T6 transaction coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "T6 transaction coverage requires a renderer backend",
            )
        })?;
        backend
            .submit_prepared_vello_pass_for_test(device_identity, prepared, target_extent)
            .await
    }

    #[cfg(test)]
    pub(crate) fn default_device_active_operation_generation_for_test(&mut self) -> Option<u64> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .active_operation_generation_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn default_device_has_no_terminal_signal_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return true;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.terminal_reason(device_identity).is_none())
    }

    #[cfg(test)]
    pub(crate) fn default_device_capabilities_for_test(&mut self) -> AvailableRuntimeCapabilities {
        let device_identity = self.default_device.expect("test requires a default device");
        self.backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .expect("test requires a ready default device")
            .runtime_report(Format::Rgba8)
    }

    #[cfg(test)]
    pub(crate) fn destroy_default_device_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        let Some(backend) = self.backend.as_mut() else {
            return false;
        };
        backend.destroy_device_for_test(device_identity)
    }

    #[cfg(test)]
    pub(crate) fn wait_for_default_terminal_signal_for_test(&mut self, timeout: Duration) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.wait_for_terminal_for_test(device_identity, timeout))
    }

    #[cfg(test)]
    pub(crate) async fn add_donor_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "the renderer has no backend to receive a donor wgpu device",
            )
        })?;
        backend.add_device_slot_for_test().await
    }

    #[cfg(test)]
    pub(crate) async fn submit_scoped_wgpu_probe_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> Result<()> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::new(
                BackendErrorCode::AdapterUnavailable,
                "real second-slot WGPU coverage requires a renderer backend",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let work = (|| -> Result<()> {
            let (device, queue) =
                backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Surgeist second-slot terminal test target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist second-slot terminal test encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Surgeist second-slot terminal test pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
            }
            queue.submit([encoder.finish()]);
            Ok(())
        })();
        let scope_result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
        backend.observe_device_terminal(device_identity);
        scope_result?;
        work
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
                BackendErrorCode::AdapterUnavailable,
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
                    BackendErrorCode::AdapterUnavailable,
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
                BackendErrorCode::AdapterUnavailable,
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
                    BackendErrorCode::AdapterUnavailable,
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

fn runtime_surface_format(surface: &Surface) -> Format {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let SurfaceBackend::Presented {
        surface: native, ..
    } = &surface.backend
    {
        return match native.format {
            wgpu::TextureFormat::Rgba8Unorm => Format::Rgba8,
            wgpu::TextureFormat::Bgra8Unorm => Format::Bgra8,
            _ => surface.options.format,
        };
    }
    surface.options.format
}

fn runtime_surface_unavailable_reason(
    _surface: &Surface,
) -> Option<RuntimeCapabilityUnavailableReason> {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let SurfaceBackend::Presented { lifecycle, .. } = &_surface.backend {
        let state = match lifecycle {
            PresentedLifecycle::NonRenderable { .. } => RenderSurfaceAvailability::NonRenderable,
            PresentedLifecycle::Occluded { .. } => RenderSurfaceAvailability::Occluded,
            PresentedLifecycle::Lost => RenderSurfaceAvailability::Lost,
            PresentedLifecycle::Ready { .. } | PresentedLifecycle::ResizePending { .. } => {
                return None;
            }
        };
        return Some(RuntimeCapabilityUnavailableReason::SurfaceUnavailable { state });
    }
    None
}

fn surface_identity_mismatch(
    operation: RuntimeOperation,
    kind: SurfaceIdentityMismatchKind,
) -> Error {
    let diagnostic = RuntimeCapabilityUnavailable::try_new(
        operation,
        RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { kind },
    )
    .expect("surface identity mismatch is valid for every surface operation");
    Error::runtime_capability_unavailable(diagnostic)
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
    error.append_message(
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
