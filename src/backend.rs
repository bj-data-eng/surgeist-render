use super::{surface::SurfaceBackend, *};
use std::{
    num::NonZeroUsize,
    time::{Duration, Instant},
};

pub(crate) struct Backend {
    pub(crate) context: vello::util::RenderContext,
    pub(crate) renderers: Vec<Option<vello::Renderer>>,
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
            texture,
            view,
            physical_size,
        } => {
            if physical_size.width == 0 || physical_size.height == 0 {
                return Ok(RenderTimings::default());
            }
            ensure_vello_renderer(backend, options, *dev_id)?;
            if texture.is_none() || view.is_none() {
                let (next_texture, next_view) = create_headless_texture(
                    &backend.context.devices[*dev_id].device,
                    *physical_size,
                    surface.options.format,
                );
                *texture = Some(next_texture);
                *view = Some(next_view);
            }
            let device_handle = &backend.context.devices[*dev_id];
            let render_start = Instant::now();
            backend.renderers[*dev_id]
                .as_mut()
                .expect("renderer should exist")
                .render_to_texture(
                    &device_handle.device,
                    &device_handle.queue,
                    scene,
                    view.as_ref().expect("headless view should exist"),
                    &vello_render_params(parameters, *physical_size, options.antialiasing),
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
            valid,
            pending_physical_size,
            ..
        } => {
            if let Some(size) = pending_physical_size.take() {
                if size.width > 0 && size.height > 0 {
                    backend
                        .context
                        .resize_surface(native, size.width, size.height);
                    *valid = true;
                } else {
                    *valid = false;
                }
            }
            if !*valid {
                return Ok(RenderTimings::default());
            }
            ensure_vello_renderer(backend, options, native.dev_id)?;
            let device_handle = &backend.context.devices[native.dev_id];
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
                        antialiasing_method: options.antialiasing.into(),
                    },
                )
                .map_err(|source| {
                    Error::new(vello_error_code(&source), vello_error_message(&source))
                        .with_source(source)
                })?;
            let render_time = render_start.elapsed();

            let present_start = Instant::now();
            let surface_texture = match native.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
                wgpu::CurrentSurfaceTexture::Outdated
                | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                    backend.context.configure_surface(native);
                    return Err(Error::new(
                        ErrorCode::SurfaceOutdated,
                        "surface is outdated and requires reconfiguration",
                    ));
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
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
                    *valid = false;
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

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) fn apply_presented_resize_state(
    backend: Option<&mut Backend>,
    native: &mut vello::util::RenderSurface<'static>,
    resizing: bool,
) {
    #[cfg(target_os = "macos")]
    if let Some(backend) = backend {
        // SAFETY: wgpu checks the backend cast. If the presented surface is not Metal,
        // as_hal returns None and the resize hint is simply unavailable.
        unsafe {
            if let Some(hal_surface) = native.surface.as_hal::<wgpu::hal::api::Metal>() {
                hal_surface
                    .render_layer()
                    .lock()
                    .setPresentsWithTransaction(resizing);
                backend.context.configure_surface(native);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    let _ = (backend, native, resizing);
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
            vello::RendererOptions {
                use_cpu: options.use_cpu,
                antialiasing_support: vello_aa_support(options.antialiasing),
                num_init_threads: NonZeroUsize::new(1),
                ..vello::RendererOptions::default()
            },
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
        width: physical_size.width,
        height: physical_size.height,
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
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surgeist headless target"),
        size: wgpu::Extent3d {
            width: physical_size.width.max(1),
            height: physical_size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.into(),
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
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
    let width = physical_size.width.max(1);
    let height = physical_size.height.max(1);
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
