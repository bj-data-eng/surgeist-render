#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::PresentedLifecycle;
use super::surface::{HeadlessResources, SurfaceBackend};
use super::*;
use super::{
    command::OffscreenBounds,
    geometry::physical_size,
    texture::{
        OffscreenTextureCache, OffscreenTextureHandle, TextureCacheKey, TextureDescriptor,
        TextureLifecycleStats, TextureUsageIntent, headless_texture_descriptor,
    },
};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

pub(crate) struct Backend {
    pub(crate) context: vello::util::RenderContext,
    pub(crate) renderers: Vec<Option<vello::Renderer>>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenRenderGpuContext<'a> {
    backend: &'a mut Backend,
    dev_id: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl<'a> OffscreenRenderGpuContext<'a> {
    #[must_use]
    pub(crate) fn new(backend: &'a mut Backend, dev_id: usize) -> Self {
        Self { backend, dev_id }
    }
}

/// Describes a Vello scene that has already been encoded in offscreen-local
/// coordinates; bounds size allocates the target texture, not a scene crop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenLocalSceneRenderRequest {
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
    parameters: Parameters,
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenLocalSceneRenderRequest {
    #[must_use]
    pub(crate) const fn new(
        bounds: OffscreenBounds,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Self {
        Self {
            bounds,
            scale,
            format,
            parameters,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenRenderTarget {
    handle: OffscreenTextureHandle,
    resource_id: u64,
    bounds: OffscreenBounds,
    descriptor: TextureDescriptor,
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenRenderTarget {
    fn new(
        handle: OffscreenTextureHandle,
        resource_id: u64,
        bounds: OffscreenBounds,
        descriptor: TextureDescriptor,
    ) -> Self {
        Self {
            handle,
            resource_id,
            bounds,
            descriptor,
        }
    }

    #[must_use]
    pub(crate) const fn handle(self) -> OffscreenTextureHandle {
        self.handle
    }

    #[must_use]
    pub(crate) const fn resource_id(self) -> u64 {
        self.resource_id
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> OffscreenBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn descriptor(self) -> TextureDescriptor {
        self.descriptor
    }
}

struct OffscreenTextureResource {
    resource_id: u64,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct OffscreenTextureResourceCache {
    lifecycle: OffscreenTextureCache,
    released_by_key: HashMap<TextureCacheKey, VecDeque<OffscreenTextureResource>>,
    next_resource_id: u64,
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenTextureResourceCache {
    pub(crate) fn new() -> Self {
        Self {
            lifecycle: OffscreenTextureCache::new(),
            released_by_key: HashMap::new(),
            next_resource_id: 0,
        }
    }

    fn acquire(
        &mut self,
        device: &wgpu::Device,
        bounds: OffscreenBounds,
        descriptor: TextureDescriptor,
    ) -> Result<OffscreenTextureResourceLease> {
        let key = TextureCacheKey::from_descriptor(descriptor);
        let stats_before = self.lifecycle.stats();
        let handle = self.lifecycle.acquire(descriptor)?;
        let reused = self.lifecycle.stats().hits > stats_before.hits;
        let resource = if reused {
            match self
                .released_by_key
                .get_mut(&key)
                .and_then(VecDeque::pop_front)
            {
                Some(resource) => resource,
                None => {
                    let _ = self.lifecycle.release(handle);
                    return Err(Error::new(
                        ErrorCode::RenderFailed,
                        "offscreen texture resource cache lost a released GPU resource",
                    ));
                }
            }
        } else {
            self.next_resource_id = self.next_resource_id.checked_add(1).ok_or_else(|| {
                Error::invalid_value(
                    "offscreen texture resource id",
                    self.next_resource_id,
                    "must have remaining resource id space",
                )
            })?;
            let (texture, view) =
                create_texture(device, "Surgeist offscreen local scene target", descriptor);
            OffscreenTextureResource {
                resource_id: self.next_resource_id,
                texture,
                view,
            }
        };
        Ok(OffscreenTextureResourceLease {
            target: OffscreenRenderTarget::new(handle, resource.resource_id, bounds, descriptor),
            texture: resource.texture,
            view: resource.view,
        })
    }

    fn release_resource(&mut self, resource: OffscreenTextureResourceLease) -> Result<()> {
        self.lifecycle.release(resource.target.handle())?;
        self.released_by_key
            .entry(TextureCacheKey::from_descriptor(
                resource.target.descriptor(),
            ))
            .or_default()
            .push_back(OffscreenTextureResource {
                resource_id: resource.target.resource_id(),
                texture: resource.texture,
                view: resource.view,
            });
        Ok(())
    }

    #[must_use]
    pub(crate) const fn stats(&self) -> TextureLifecycleStats {
        self.lifecycle.stats()
    }

    #[must_use]
    pub(crate) const fn live_count(&self) -> usize {
        self.lifecycle.live_count()
    }

    #[must_use]
    pub(crate) fn released_resource_count(&self) -> usize {
        self.released_by_key.values().map(VecDeque::len).sum()
    }
}

impl Default for OffscreenTextureResourceCache {
    fn default() -> Self {
        Self::new()
    }
}

struct OffscreenTextureResourceLease {
    target: OffscreenRenderTarget,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use = "offscreen rendered texture leases must be released back to their resource cache"]
pub(crate) struct OffscreenRenderedTextureLease {
    target: OffscreenRenderTarget,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    timings: RenderTimings,
}

impl fmt::Debug for OffscreenRenderedTextureLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OffscreenRenderedTextureLease")
            .field("target", &self.target)
            .field("timings", &self.timings)
            .finish_non_exhaustive()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl OffscreenRenderedTextureLease {
    #[must_use]
    pub(crate) const fn target(&self) -> OffscreenRenderTarget {
        self.target
    }

    #[must_use]
    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[must_use]
    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    #[must_use]
    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    pub(crate) fn release(self, cache: &mut OffscreenTextureResourceCache) -> Result<()> {
        cache.release_resource(OffscreenTextureResourceLease {
            target: self.target,
            texture: self.texture,
            view: self.view,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn offscreen_local_scene_texture_descriptor(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<TextureDescriptor> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    TextureDescriptor::try_new(
        physical_size(bounds.rect().size(), scale)?,
        format,
        TextureUsageIntent::OffscreenLayer,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn render_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    cache: &mut OffscreenTextureResourceCache,
    scene: &vello::Scene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let descriptor =
        offscreen_local_scene_texture_descriptor(request.bounds, request.scale, request.format)?;
    let Some(context) = context else {
        return Err(Error::new(
            ErrorCode::AdapterUnavailable,
            "offscreen Vello local scene rendering requires an available wgpu device context",
        ));
    };
    ensure_vello_renderer(context.backend, options, context.dev_id)?;
    let device_handle = &context.backend.context.devices[context.dev_id];
    let resource = cache.acquire(&device_handle.device, request.bounds, descriptor)?;
    let render_start = Instant::now();
    let result = context.backend.renderers[context.dev_id]
        .as_mut()
        .expect("renderer should exist")
        .render_to_texture(
            &device_handle.device,
            &device_handle.queue,
            scene,
            &resource.view,
            &vello_render_params(
                request.parameters,
                resource.target.descriptor().physical_size(),
                options.antialiasing(),
            ),
        );
    if let Err(source) = result {
        let error =
            Error::new(vello_error_code(&source), vello_error_message(&source)).with_source(source);
        cache.release_resource(resource)?;
        return Err(error);
    }
    Ok(OffscreenRenderedTextureLease {
        target: resource.target,
        texture: resource.texture,
        view: resource.view,
        timings: RenderTimings {
            render_time: render_start.elapsed(),
            present_time: Duration::ZERO,
        },
    })
}

pub(crate) fn render_vello_surface(
    backend: &mut Backend,
    options: Options,
    surface: &mut Surface,
    scene: &vello::Scene,
    parameters: Parameters,
) -> Result<RenderTimings> {
    match &mut surface.backend {
        SurfaceBackend::ContractOnly { .. } => Ok(RenderTimings::default()),
        SurfaceBackend::Headless {
            dev_id,
            resources,
            physical_size,
        } => {
            if physical_size.width() == 0 || physical_size.height() == 0 {
                return Ok(RenderTimings::default());
            }
            ensure_vello_renderer(backend, options, *dev_id)?;
            if matches!(resources, HeadlessResources::Pending) {
                let (next_texture, next_view) = create_headless_texture(
                    &backend.context.devices[*dev_id].device,
                    *physical_size,
                    surface.options.format,
                )?;
                *resources = HeadlessResources::Ready {
                    texture: next_texture,
                    view: next_view,
                };
            }
            let HeadlessResources::Ready { view, .. } = resources else {
                unreachable!("headless resources should be ready after allocation");
            };
            let device_handle = &backend.context.devices[*dev_id];
            let render_start = Instant::now();
            backend.renderers[*dev_id]
                .as_mut()
                .expect("renderer should exist")
                .render_to_texture(
                    &device_handle.device,
                    &device_handle.queue,
                    scene,
                    view,
                    &vello_render_params(parameters, *physical_size, options.antialiasing()),
                )
                .map_err(|source| {
                    Error::new(vello_error_code(&source), vello_error_message(&source))
                        .with_source(source)
                })?;
            Ok(RenderTimings {
                render_time: render_start.elapsed(),
                present_time: Duration::ZERO,
            })
        }
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        SurfaceBackend::Presented {
            surface: native,
            lifecycle,
            ..
        } => {
            match lifecycle {
                PresentedLifecycle::ResizePending {
                    physical_size,
                    resizing,
                } => {
                    backend.context.resize_surface(
                        native,
                        physical_size.width(),
                        physical_size.height(),
                    );
                    *lifecycle = PresentedLifecycle::Ready {
                        resizing: *resizing,
                    };
                }
                PresentedLifecycle::NonRenderable { .. } | PresentedLifecycle::Lost => {
                    return Ok(RenderTimings::default());
                }
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::Occluded { .. } => {}
            }
            ensure_vello_renderer(backend, options, native.dev_id)?;
            let device_handle = &backend.context.devices[native.dev_id];
            let resizing = lifecycle.resize_state();
            let render_start = Instant::now();
            backend.renderers[native.dev_id]
                .as_mut()
                .expect("renderer should exist")
                .render_to_texture(
                    &device_handle.device,
                    &device_handle.queue,
                    scene,
                    &native.target_view,
                    &vello::RenderParams {
                        width: native.config.width,
                        height: native.config.height,
                        base_color: parameters.base_color.into(),
                        antialiasing_method: options.antialiasing().into(),
                    },
                )
                .map_err(|source| {
                    Error::new(vello_error_code(&source), vello_error_message(&source))
                        .with_source(source)
                })?;
            let render_time = render_start.elapsed();

            let present_start = Instant::now();
            let surface_texture = match native.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                    *lifecycle = PresentedLifecycle::Ready { resizing };
                    surface_texture
                }
                wgpu::CurrentSurfaceTexture::Outdated
                | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    backend.context.configure_surface(native);
                    return Err(Error::new(
                        ErrorCode::SurfaceOutdated,
                        "surface is outdated and requires reconfiguration",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    *lifecycle = PresentedLifecycle::Occluded { resizing };
                    return Ok(RenderTimings {
                        render_time,
                        present_time: present_start.elapsed(),
                    });
                }
                wgpu::CurrentSurfaceTexture::Timeout => {
                    return Err(Error::new(
                        ErrorCode::SurfaceTimeout,
                        "timed out acquiring surface texture",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    *lifecycle = PresentedLifecycle::Lost;
                    return Err(Error::new(ErrorCode::SurfaceLost, "surface was lost"));
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    return Err(Error::new(
                        ErrorCode::RenderFailed,
                        "surface texture validation failed",
                    ));
                }
            };

            let mut encoder =
                device_handle
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Surgeist surface blit"),
                    });
            native.blitter.copy(
                &device_handle.device,
                &mut encoder,
                &native.target_view,
                &surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            );
            device_handle.queue.submit([encoder.finish()]);
            surface_texture.present();
            device_handle
                .device
                .poll(wgpu::PollType::Poll)
                .map_err(|source| {
                    Error::new(ErrorCode::PresentFailed, "failed to poll render device")
                        .with_source(source)
                })?;
            Ok(RenderTimings {
                render_time,
                present_time: present_start.elapsed(),
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RenderTimings {
    pub(crate) render_time: Duration,
    pub(crate) present_time: Duration,
}

pub(crate) fn ensure_vello_renderer(
    backend: &mut Backend,
    options: Options,
    dev_id: usize,
) -> Result<()> {
    backend
        .renderers
        .resize_with(backend.context.devices.len(), || None);
    if backend.renderers[dev_id].is_none() {
        let renderer = vello::Renderer::new(
            &backend.context.devices[dev_id].device,
            vello_renderer_options(options),
        )
        .map_err(|source| {
            Error::new(
                ErrorCode::RendererCreateFailed,
                "failed to create Vello renderer",
            )
            .with_source(source)
        })?;
        backend.renderers[dev_id] = Some(renderer);
    }
    Ok(())
}

pub(crate) fn vello_renderer_options(options: Options) -> vello::RendererOptions {
    vello::RendererOptions {
        use_cpu: false,
        antialiasing_support: vello_aa_support(options.antialiasing()),
        num_init_threads: NonZeroUsize::new(1),
        ..vello::RendererOptions::default()
    }
}

pub(crate) fn vello_error_code(error: &vello::Error) -> ErrorCode {
    match error {
        vello::Error::WgpuErrorFromScope(wgpu::Error::OutOfMemory { .. }) => {
            ErrorCode::SurfaceOutOfMemory
        }
        _ => ErrorCode::RenderFailed,
    }
}

pub(crate) fn vello_error_message(error: &vello::Error) -> &'static str {
    match vello_error_code(error) {
        ErrorCode::SurfaceOutOfMemory => "rendering exhausted GPU memory",
        _ => "failed to render scene",
    }
}

pub(crate) fn vello_render_params(
    parameters: Parameters,
    physical_size: PhysicalSize,
    antialiasing: Antialiasing,
) -> vello::RenderParams {
    vello::RenderParams {
        base_color: parameters.base_color.into(),
        width: physical_size.width(),
        height: physical_size.height(),
        antialiasing_method: antialiasing.into(),
    }
}

pub(crate) fn vello_aa_support(antialiasing: Antialiasing) -> vello::AaSupport {
    [vello::AaConfig::from(antialiasing)].into_iter().collect()
}

pub(crate) fn create_headless_texture(
    device: &wgpu::Device,
    physical_size: PhysicalSize,
    format: Format,
) -> Result<(wgpu::Texture, wgpu::TextureView)> {
    let descriptor = headless_texture_descriptor(physical_size, format)?;
    Ok(create_texture(
        device,
        "Surgeist headless target",
        descriptor,
    ))
}

pub(crate) fn create_texture(
    device: &wgpu::Device,
    label: &'static str,
    descriptor: TextureDescriptor,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: descriptor.physical_size().width(),
            height: descriptor.physical_size().height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: descriptor.format().into(),
        usage: descriptor.wgpu_usage(),
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(crate) fn read_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    physical_size: PhysicalSize,
) -> Result<ImageBuffer> {
    let width = physical_size.width().max(1);
    let height = physical_size.height().max(1);
    let padded_bytes_per_row = (width * 4).next_multiple_of(256);
    let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Surgeist headless readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Surgeist headless copy"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|source| {
            Error::new(ErrorCode::RenderFailed, "failed to poll render device").with_source(source)
        })?;
    receiver
        .recv()
        .map_err(|_| {
            Error::new(
                ErrorCode::RenderFailed,
                "headless readback callback dropped",
            )
        })?
        .map_err(|source| {
            Error::new(ErrorCode::RenderFailed, "failed to map headless readback")
                .with_source(source)
        })?;

    let mapped = slice.get_mapped_range();
    let row_bytes = (width * 4) as usize;
    let mut rgba = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        rgba.extend_from_slice(&mapped[start..start + row_bytes]);
    }
    drop(mapped);
    buffer.unmap();

    Ok(ImageBuffer {
        size: physical_size,
        rgba,
    })
}

impl From<Antialiasing> for vello::AaConfig {
    fn from(antialiasing: Antialiasing) -> Self {
        match antialiasing {
            Antialiasing::Area => Self::Area,
            Antialiasing::Msaa8 => Self::Msaa8,
            Antialiasing::Msaa16 => Self::Msaa16,
        }
    }
}
