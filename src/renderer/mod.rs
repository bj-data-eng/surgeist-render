mod dispatch;
mod options;
mod publication;
#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::{
    BoundedBackdropRenderResultForTest, ColorFilterRenderResultForTest,
    ForcedGraphRenderResultForTest, SpatialFilterRenderResultForTest,
    unsupported_graph_diagnostic_for_test,
};

use dispatch::runtime_surface_format;
pub use options::{Antialiasing, EffectQualityPolicy, Options, ResourceCacheBudget};

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, PresentedSurfaceState, ResizeState};
use super::{
    backend::*,
    geometry::physical_size,
    readback::read_texture_rgba,
    surface::{HeadlessResources, RendererIdentity, SurfaceBackend},
    validation::*,
    *,
};
use std::{collections::HashSet, time::Instant};

/// GPU-only renderer and owner of device-scoped resources and frame transactions.
///
/// Effect-free scenes select [`RenderRoute::DirectVello`]. Scenes requiring the
/// implemented resolved-alpha-mask, composition, or bounded-backdrop subset
/// select [`RenderRoute::GpuGraph`]. Both routes encode into one
/// transaction-owned submission. The renderer never retries pixels on a CPU
/// path and never performs implicit readback.
pub struct Renderer {
    identity: RendererIdentity,
    options: Options,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    backend: Option<Backend>,
    default_device: Option<DeviceSlotIdentity>,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
fn ensure_presented_device_available_after_creation(
    backend: &mut Backend,
    device_identity: DeviceSlotIdentity,
    operation: RuntimeOperation,
) -> Result<()> {
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    Ok(())
}

impl Renderer {
    /// Creates a GPU-only renderer with the supplied fixed [`Options`].
    ///
    /// Device selection is asynchronous. When no compatible adapter is
    /// available, construction retains a contract-only headless boundary so
    /// later nonzero operations return typed runtime capability failures.
    pub async fn new(options: Options) -> Result<Self> {
        let mut backend = Backend::new(options.resource_cache_budget());
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

    /// Creates a surface and awaits any native-window or WebGPU host setup.
    ///
    /// The returned surface is ready for its next lifecycle operation when this
    /// future succeeds. Invalid options and unsupported attachments preserve
    /// their existing diagnostics when the future is awaited. Presented host
    /// lifecycle remains owned by the caller and its window or browser event
    /// loop. This future does not promise to be `Send`.
    pub async fn create_surface(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        self.create_surface_with_configuration_operation(
            attachment,
            options,
            RuntimeOperation::SurfaceRendering,
            None,
        )
        .await
    }

    async fn create_surface_with_configuration_operation(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        match attachment {
            Attachment::Headless => self.create_headless_surface(options).await,
            Attachment::WebCanvas(canvas) => {
                self.create_web_canvas_surface(
                    canvas,
                    options,
                    configuration_operation,
                    preferred_device,
                )
                .await
            }
            #[cfg(feature = "render-window")]
            Attachment::Window(handle) => {
                let Some(backend) = self.backend.as_mut() else {
                    return Err(Error::runtime_unavailable(
                        RuntimeOperation::AdapterSelection,
                        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                        "no compatible wgpu adapter is available",
                    ));
                };
                let physical_size = physical_size(options.size, options.scale)?;
                let (surface, device_identity) = backend
                    .create_presented_surface(
                        handle.clone(),
                        preferred_device,
                        configuration_operation,
                    )
                    .await?;
                ensure_presented_device_available_after_creation(
                    backend,
                    device_identity,
                    configuration_operation,
                )?;
                let mut created = Surface::with_backend(
                    Attachment::Window(handle),
                    options,
                    SurfaceBackend::Presented {
                        surface: Box::new(surface),
                        device_identity,
                        state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
                    },
                    self.identity.clone(),
                );
                self.configure_presented_surface_if_needed(&mut created, configuration_operation)
                    .await?;
                Ok(created)
            }
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    async fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        let Some(html_canvas) = canvas.html_canvas() else {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                format!("web canvas surface '{}' has no canvas handle", canvas.id()),
            ));
        };
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible WebGPU adapter is available",
            ));
        };
        let physical_size = physical_size(options.size, options.scale)?;
        let (surface, device_identity) = backend
            .create_presented_surface(
                wgpu::SurfaceTarget::Canvas(html_canvas),
                preferred_device,
                configuration_operation,
            )
            .await?;
        ensure_presented_device_available_after_creation(
            backend,
            device_identity,
            configuration_operation,
        )?;
        let mut created = Surface::with_backend(
            Attachment::WebCanvas(canvas),
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                device_identity,
                state: PresentedSurfaceState::new(physical_size, ResizeState::Idle),
            },
            self.identity.clone(),
        );
        self.configure_presented_surface_if_needed(&mut created, configuration_operation)
            .await?;
        Ok(created)
    }

    #[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
    async fn create_web_canvas_surface(
        &mut self,
        canvas: WebCanvas,
        _options: SurfaceOptions,
        _configuration_operation: RuntimeOperation,
        _preferred_device: Option<DeviceSlotIdentity>,
    ) -> Result<Surface> {
        let _ = canvas;
        Capabilities::CURRENT.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))?;
        unreachable!("web canvas support requires the render-web feature on wasm32");
    }

    /// Creates a headless surface for later asynchronous GPU operations.
    ///
    /// `size` is in logical units and `scale` converts it to physical pixels.
    /// Await this operation before using the surface. Input and `Rgba8` format
    /// failures are reported when the future is awaited; explicit readback is a
    /// separate asynchronous operation.
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
            SurfaceBackend::Headless {
                device_identity,
                resources: HeadlessResources::for_physical_size(physical_size),
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

    /// Updates native presented-resize intent after validating surface identity and lifecycle.
    ///
    /// The flag is host scheduling input, not a resize by itself. Repeating the
    /// same value is idempotent; invalid, unavailable, foreign, or stale surfaces
    /// return their typed diagnostic without changing committed resources.
    pub fn set_surface_resizing(&mut self, surface: &mut Surface, resizing: bool) -> Result<()> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;

        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = resizing;

        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { state, .. } = &mut surface.backend {
            let next = if resizing {
                ResizeState::Resizing
            } else {
                ResizeState::Idle
            };
            if state.lifecycle().resize_state() == next {
                return Ok(());
            }
            state.set_resizing(next);
        }

        Ok(())
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn configure_presented_surface_if_needed(
        &mut self,
        surface: &mut Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let (device_identity, native, requested_physical_size, present_mode, needs_configuration) =
            match &surface.backend {
                SurfaceBackend::Presented {
                    surface: native,
                    device_identity,
                    state,
                } => (
                    *device_identity,
                    native.as_ref(),
                    state.requested_physical_size(),
                    surface.options.present_mode.into(),
                    state.needs_configuration(),
                ),
                SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => {
                    return Ok(());
                }
            };
        if !needs_configuration {
            return Ok(());
        }
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                operation,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            )
        })?;
        let draft = backend
            .configure_presented_surface(
                device_identity,
                operation,
                native,
                requested_physical_size,
                present_mode,
            )
            .await?;
        let publication_signal = backend.publication_signal(device_identity, operation)?;
        let result = publication_signal.commit_if_no_terminal(operation, || {
            let SurfaceBackend::Presented { surface, state, .. } = &mut surface.backend else {
                unreachable!("presented configuration must commit into the originating surface");
            };
            surface.commit_configuration(draft);
            state.commit_configuration();
        });
        if let Err(error) = result {
            backend.observe_device_terminal(device_identity);
            return Err(error);
        };
        Ok(())
    }

    #[cfg(not(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    )))]
    async fn configure_presented_surface_if_needed(
        &mut self,
        _surface: &mut Surface,
        _operation: RuntimeOperation,
    ) -> Result<()> {
        Ok(())
    }

    /// Submits one failure-atomic GPU render operation for an available surface.
    ///
    /// Awaiting this future returns [`Stats`] only after validation, resource
    /// cleanup, submission, and any presentation succeed. On validation,
    /// lifecycle, capability, backend, cancellation, or presentation failure,
    /// no draft frame or statistics publish: the surface and [`Self::stats`]
    /// retain their last successful values. There is no production CPU fallback.
    pub async fn render(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<Stats> {
        let frame_start = Instant::now();
        let (device_identity, publication) = self
            .dispatch_render_frame(surface, scene, parameters)
            .await?;
        self.publish_clean_render_frame(surface, device_identity, publication, frame_start)
    }

    /// Resumes a compatible surface, awaiting host-resource recreation when presented.
    ///
    /// Await this operation before rendering again. Incompatible attachments,
    /// foreign/stale identity, terminal-device, configuration, and host failures
    /// preserve the previously committed surface state and their typed ordering.
    pub async fn resume_surface(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
    ) -> Result<()> {
        #[cfg(not(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        )))]
        let _ = attachment;
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceResume)?;

        match &surface.backend {
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { state, .. } => {
                let action = Surface::presented_resume_action(surface.state, state.lifecycle());
                let resizing = state.lifecycle().resize_state();
                self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceResume)?;
                surface.ensure_attachment_compatible(&attachment)?;
                match action {
                    super::surface::PresentedResumeAction::NoOp => Ok(()),
                    super::surface::PresentedResumeAction::ConfigureExisting => {
                        self.configure_presented_surface_if_needed(
                            surface,
                            RuntimeOperation::SurfaceResume,
                        )
                        .await
                    }
                    super::surface::PresentedResumeAction::Configure => {
                        self.recreate_presented_surface_for_resume(
                            surface, attachment, resizing, true,
                        )
                        .await
                    }
                    super::surface::PresentedResumeAction::Recreate => {
                        self.recreate_presented_surface_for_resume(
                            surface, attachment, resizing, false,
                        )
                        .await
                    }
                }
            }
            SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => unreachable!(),
        }
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    async fn recreate_presented_surface_for_resume(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
        resizing: ResizeState,
        preserve_renderer_identity: bool,
    ) -> Result<()> {
        let preferred_device = surface
            .device_identity()
            .expect("a presented surface must retain its device slot identity");
        let mut next = self
            .create_surface_with_configuration_operation(
                attachment,
                surface.options,
                RuntimeOperation::SurfaceResume,
                Some(preferred_device),
            )
            .await?;
        next.last_parameters = surface.last_parameters;
        if preserve_renderer_identity {
            next.renderer_identity = surface.renderer_identity.clone();
        }
        if let SurfaceBackend::Presented { state, .. } = &mut next.backend {
            state.set_resizing(resizing);
        }
        *surface = next;
        Ok(())
    }

    /// Performs explicit headless readback of the current complete publication.
    ///
    /// The returned [`ImageBuffer`] contains tightly packed straight-alpha RGBA8
    /// physical pixels. A zero-area available surface returns an empty validated
    /// image without GPU work. A nonzero surface without a publication returns
    /// its typed uninitialized diagnostic. Failed or canceled mapping never
    /// changes the published frame. The future is not promised to be `Send`.
    pub async fn read_headless(&mut self, surface: &Surface) -> Result<ImageBuffer> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceReadback)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceReadback)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceReadback)?;
        surface.ensure_available(RuntimeOperation::SurfaceReadback)?;
        let (device_identity, texture, physical_size) = match &surface.backend {
            SurfaceBackend::ContractOnly { physical_size }
                if physical_size.width() == 0 || physical_size.height() == 0 =>
            {
                return ImageBuffer::try_new(*physical_size, Vec::new());
            }
            SurfaceBackend::ContractOnly { .. } => {
                return Err(Error::runtime_unavailable(
                    RuntimeOperation::SurfaceReadback,
                    RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                    "no compatible wgpu adapter is available",
                ));
            }
            SurfaceBackend::Headless {
                physical_size,
                resources: HeadlessResources::Empty,
                ..
            } => {
                return ImageBuffer::try_new(*physical_size, Vec::new());
            }
            SurfaceBackend::Headless {
                resources: HeadlessResources::Pending,
                ..
            } => {
                return Err(Error::runtime_unavailable(
                    RuntimeOperation::SurfaceReadback,
                    RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                        state: RenderSurfaceAvailability::Uninitialized,
                    },
                    "headless surface has no published texture",
                ));
            }
            SurfaceBackend::Headless {
                device_identity,
                resources: HeadlessResources::Ready { texture, .. },
                physical_size,
            } => (*device_identity, texture, *physical_size),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => unreachable!(),
        };
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceReadback)?;
        let Some(backend) = self.backend.as_mut() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        read_texture_rgba(
            backend,
            device_identity,
            texture,
            physical_size,
            RuntimeOperation::SurfaceReadback,
        )
        .await
    }

    /// Projects immutable runtime-phase capabilities of the device selected by `surface`.
    ///
    /// This query observes pending terminal device signals but performs no
    /// allocation, submission, mapping, polling, or Vello/WGPU resource call.
    /// It is separate from semantic [`Capabilities`] and from any Cargo feature:
    /// features select compiled host adapters, while this report describes the
    /// selected device/surface snapshot.
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

    #[must_use]
    /// Returns statistics for the last successful published frame.
    ///
    /// Failed and canceled render attempts do not replace this value.
    pub const fn stats(&self) -> Stats {
        self.stats
    }

    #[must_use]
    /// Returns the fixed configuration supplied when this renderer was created.
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

    fn validate_surface_operation_backend(
        &self,
        surface: &Surface,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let supported = match operation {
            RuntimeOperation::SurfaceReadback => matches!(
                surface.backend,
                SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. }
            ),
            RuntimeOperation::SurfaceResume => {
                #[cfg(any(
                    feature = "render-window",
                    all(feature = "render-web", target_arch = "wasm32")
                ))]
                {
                    matches!(surface.backend, SurfaceBackend::Presented { .. })
                }
                #[cfg(not(any(
                    feature = "render-window",
                    all(feature = "render-web", target_arch = "wasm32")
                )))]
                {
                    false
                }
            }
            _ => true,
        };
        if supported {
            Ok(())
        } else {
            Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "surface backend does not support this operation",
            ))
        }
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

    #[must_use]
    /// Returns the crate's semantic authored-operation capability contract.
    ///
    /// This does not inspect a runtime device or surface; use
    /// [`Self::runtime_capabilities`] for runtime facts.
    pub const fn capabilities(&self) -> Capabilities {
        Capabilities::CURRENT
    }
}

fn runtime_surface_unavailable_reason(
    _surface: &Surface,
) -> Option<RuntimeCapabilityUnavailableReason> {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let SurfaceBackend::Presented { state, .. } = &_surface.backend {
        let state = match state.lifecycle() {
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
