use crate::texture::{TextureDescriptor, headless_texture_descriptor};
use crate::{Format, PhysicalSize, Result};

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
