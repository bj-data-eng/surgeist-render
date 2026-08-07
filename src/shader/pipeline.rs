use std::borrow::Cow;

use crate::Result;

use super::{
    ShaderMaskExtendKey,
    key::{
        SamplerKey, ShaderCompositePathKey, ShaderMaskSamplingKey, ShaderSamplingEdgeKey,
        ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    validate::{
        BlurAxis, BlurInput, BlurPassDescription, ColorFilterPassDescription,
        CompositePassDescription, CopyBackdropPassDescription, CorePassDescription,
        CorePassProgram, DropShadowColorizePassDescription,
    },
};

const CANONICALIZE_CAPTURE_WGSL: &str = include_str!("../shaders/canonicalize_capture.wgsl");
const SPAN_SOURCE_OVER_WGSL: &str = include_str!("../shaders/span_source_over.wgsl");
const PRESENT_WGSL: &str = include_str!("../shaders/present.wgsl");
const LAYER_COMPOSITE_WGSL: &str = include_str!("../shaders/layer_composite.wgsl");
const COLOR_FILTER_WGSL: &str = include_str!("../shaders/color_filter.wgsl");
pub(super) const BLUR_WGSL: &str = include_str!("../shaders/blur.wgsl");
const DROP_SHADOW_WGSL: &str = include_str!("../shaders/drop_shadow.wgsl");
const COPY_BACKDROP_WGSL: &str = include_str!("../shaders/copy_backdrop.wgsl");

pub(super) fn create_sampler(device: &wgpu::Device, key: SamplerKey) -> wgpu::Sampler {
    device.create_sampler(&sampler_descriptor(key))
}

fn sampler_descriptor(key: SamplerKey) -> wgpu::SamplerDescriptor<'static> {
    let filter = match key.filter {
        ShaderSamplingFilterKey::Nearest => wgpu::FilterMode::Nearest,
        ShaderSamplingFilterKey::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode = match key
        .resolved_mask_sampling
        .map(ShaderMaskSamplingKey::extend)
    {
        None | Some(ShaderMaskExtendKey::Pad) => wgpu::AddressMode::ClampToEdge,
        Some(ShaderMaskExtendKey::Repeat) => wgpu::AddressMode::Repeat,
        Some(ShaderMaskExtendKey::Reflect) => wgpu::AddressMode::MirrorRepeat,
    };
    wgpu::SamplerDescriptor {
        label: Some("Surgeist sampled-image sampler"),
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    }
}

pub(super) fn create_core_pass_bind_group_layout(
    device: &wgpu::Device,
    description: CorePassDescription,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(core_pass_bind_group_layout_label(description.program)),
        entries: &entries,
    })
}

pub(super) fn create_core_pass_shader_module(
    device: &wgpu::Device,
    description: CorePassDescription,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(core_pass_shader_label(description.program)),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(core_pass_shader_source(
            description.program,
        ))),
    })
}

pub(super) fn create_core_pass_render_pipeline(
    device: &wgpu::Device,
    description: CorePassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &'static str,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(core_pass_pipeline_layout_label(description.program)),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let blend = matches!(
        description.program,
        CorePassProgram::SpanSourceOver | CorePassProgram::DropShadowMerge
    )
    .then_some(span_source_over_blend());
    let target = wgpu::ColorTargetState {
        format: texture_format(description.target_format)?,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(core_pass_pipeline_label(description.program)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

pub(super) fn create_composite_bind_group_layout(
    device: &wgpu::Device,
    description: CompositePassDescription,
) -> wgpu::BindGroupLayout {
    let texture_entry = |binding, filterable| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let mut entries = Vec::with_capacity(7);
    entries.push(texture_entry(0, true));
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    if description.path == ShaderCompositePathKey::DestinationSampling {
        entries.push(texture_entry(2, false));
    }
    if description.has_clip_coverage {
        entries.push(texture_entry(3, false));
    }
    if description.has_alpha_mask {
        entries.push(texture_entry(4, false));
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 5,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(48),
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 6,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(112),
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist composition layer-composite bindings"),
        entries: &entries,
    })
}

pub(super) fn create_composite_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surgeist composition layer-composite shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LAYER_COMPOSITE_WGSL)),
    })
}

pub(super) fn create_composite_render_pipeline(
    device: &wgpu::Device,
    description: CompositePassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry_override: Option<&'static str>,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist composition layer-composite pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.target_format)?,
        blend: match description.path {
            ShaderCompositePathKey::Normal => Some(span_source_over_blend()),
            ShaderCompositePathKey::DestinationSampling => None,
        },
        write_mask: wgpu::ColorWrites::ALL,
    };
    let fragment_entry =
        fragment_entry_override.unwrap_or_else(|| composite_fragment_entry(description));
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist composition layer-composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

pub(super) fn create_copy_backdrop_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist backdrop-copy bindings"),
        entries: &entries,
    })
}

pub(super) fn create_copy_backdrop_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surgeist backdrop-copy shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COPY_BACKDROP_WGSL)),
    })
}

pub(super) fn create_copy_backdrop_pipeline(
    device: &wgpu::Device,
    description: CopyBackdropPassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist backdrop-copy pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.working_format)?,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist backdrop-copy pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

pub(super) fn create_color_filter_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist color-filter bindings"),
        entries: &entries,
    })
}

pub(super) fn create_color_filter_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surgeist color-filter shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COLOR_FILTER_WGSL)),
    })
}

pub(super) fn create_color_filter_render_pipeline(
    device: &wgpu::Device,
    description: ColorFilterPassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist color-filter pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.working_format)?,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist color-filter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

pub(super) fn create_blur_bind_group_layout(
    device: &wgpu::Device,
    description: BlurPassDescription,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(8),
            },
            count: None,
        },
    ];
    if description.edge == ShaderSamplingEdgeKey::SemanticBorderMirror {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist checked Gaussian blur bindings"),
        entries: &entries,
    })
}

pub(super) fn create_blur_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surgeist checked Gaussian blur shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLUR_WGSL)),
    })
}

pub(super) fn create_blur_render_pipeline(
    device: &wgpu::Device,
    description: BlurPassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist checked Gaussian blur pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.working_format)?,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist checked Gaussian blur pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(blur_fragment_entry(description)),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

pub(super) fn create_drop_shadow_colorize_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(48),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(32),
            },
            count: None,
        },
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surgeist spatial-filter drop-shadow colorize bindings"),
        entries: &entries,
    })
}

pub(super) fn create_drop_shadow_colorize_shader_module(
    device: &wgpu::Device,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Surgeist spatial-filter drop-shadow colorize shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DROP_SHADOW_WGSL)),
    })
}

pub(super) fn create_drop_shadow_colorize_render_pipeline(
    device: &wgpu::Device,
    description: DropShadowColorizePassDescription,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> Result<wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Surgeist spatial-filter drop-shadow colorize pipeline layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let target = wgpu::ColorTargetState {
        format: texture_format(description.working_format)?,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    };
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surgeist spatial-filter drop-shadow colorize pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(target)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        }),
    )
}

const fn blur_fragment_entry(description: BlurPassDescription) -> &'static str {
    match (description.axis, description.input, description.edge) {
        (BlurAxis::Horizontal, BlurInput::Rgba, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_horizontal_rgba"
        }
        (BlurAxis::Vertical, BlurInput::Rgba, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_vertical_rgba"
        }
        (BlurAxis::Horizontal, BlurInput::SourceAlpha, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_horizontal_source_alpha"
        }
        (BlurAxis::Vertical, BlurInput::SourceAlpha, ShaderSamplingEdgeKey::TransparentBlack) => {
            "fragment_vertical_source_alpha"
        }
        (BlurAxis::Horizontal, BlurInput::Rgba, ShaderSamplingEdgeKey::SemanticBorderMirror) => {
            "fragment_horizontal_rgba_mirror"
        }
        (BlurAxis::Vertical, BlurInput::Rgba, ShaderSamplingEdgeKey::SemanticBorderMirror) => {
            "fragment_vertical_rgba_mirror"
        }
        (
            BlurAxis::Horizontal,
            BlurInput::SourceAlpha,
            ShaderSamplingEdgeKey::SemanticBorderMirror,
        ) => "fragment_horizontal_source_alpha_mirror",
        (
            BlurAxis::Vertical,
            BlurInput::SourceAlpha,
            ShaderSamplingEdgeKey::SemanticBorderMirror,
        ) => "fragment_vertical_source_alpha_mirror",
        (_, _, ShaderSamplingEdgeKey::ClampToExtent) => "invalid_blur_edge_policy",
    }
}

const fn composite_fragment_entry(description: CompositePassDescription) -> &'static str {
    match (
        description.path,
        description.has_clip_coverage,
        description.has_alpha_mask,
    ) {
        (ShaderCompositePathKey::Normal, false, false) => "fragment_normal",
        (ShaderCompositePathKey::Normal, true, false) => "fragment_normal_clip",
        (ShaderCompositePathKey::Normal, false, true) => "fragment_normal_mask",
        (ShaderCompositePathKey::Normal, true, true) => "fragment_normal_clip_mask",
        (ShaderCompositePathKey::DestinationSampling, false, false) => "fragment_destination",
        (ShaderCompositePathKey::DestinationSampling, true, false) => "fragment_destination_clip",
        (ShaderCompositePathKey::DestinationSampling, false, true) => "fragment_destination_mask",
        (ShaderCompositePathKey::DestinationSampling, true, true) => {
            "fragment_destination_clip_mask"
        }
    }
}

const fn core_pass_shader_source(program: CorePassProgram) -> &'static str {
    match program {
        CorePassProgram::CanonicalizeCapture => CANONICALIZE_CAPTURE_WGSL,
        CorePassProgram::SpanSourceOver | CorePassProgram::DropShadowMerge => SPAN_SOURCE_OVER_WGSL,
        CorePassProgram::Present => PRESENT_WGSL,
    }
}

const fn core_pass_shader_label(program: CorePassProgram) -> &'static str {
    match program {
        CorePassProgram::CanonicalizeCapture => "Surgeist core-pass canonicalize-capture shader",
        CorePassProgram::SpanSourceOver => "Surgeist core-pass span source-over shader",
        CorePassProgram::DropShadowMerge => "Surgeist spatial-filter drop-shadow merge shader",
        CorePassProgram::Present => "Surgeist core-pass present shader",
    }
}

const fn core_pass_bind_group_layout_label(program: CorePassProgram) -> &'static str {
    match program {
        CorePassProgram::CanonicalizeCapture => "Surgeist core-pass canonicalize-capture bindings",
        CorePassProgram::SpanSourceOver => "Surgeist core-pass span source-over bindings",
        CorePassProgram::DropShadowMerge => "Surgeist spatial-filter drop-shadow merge bindings",
        CorePassProgram::Present => "Surgeist core-pass present bindings",
    }
}

const fn core_pass_pipeline_layout_label(program: CorePassProgram) -> &'static str {
    match program {
        CorePassProgram::CanonicalizeCapture => {
            "Surgeist core-pass canonicalize-capture pipeline layout"
        }
        CorePassProgram::SpanSourceOver => "Surgeist core-pass span source-over pipeline layout",
        CorePassProgram::DropShadowMerge => {
            "Surgeist spatial-filter drop-shadow merge pipeline layout"
        }
        CorePassProgram::Present => "Surgeist core-pass present pipeline layout",
    }
}

const fn core_pass_pipeline_label(program: CorePassProgram) -> &'static str {
    match program {
        CorePassProgram::CanonicalizeCapture => "Surgeist core-pass canonicalize-capture pipeline",
        CorePassProgram::SpanSourceOver => "Surgeist core-pass span source-over pipeline",
        CorePassProgram::DropShadowMerge => "Surgeist spatial-filter drop-shadow merge pipeline",
        CorePassProgram::Present => "Surgeist core-pass present pipeline",
    }
}

pub(super) const fn span_source_over_blend() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

const fn texture_format(format: ShaderTextureFormatKey) -> Result<wgpu::TextureFormat> {
    match format {
        ShaderTextureFormatKey::VelloCaptureRgba8Unorm
        | ShaderTextureFormatKey::ClipCoverageRgba8Unorm
        | ShaderTextureFormatKey::WorkingReducedPrecisionRgba8Unorm
        | ShaderTextureFormatKey::ResolvedMaskRgba8Unorm
        | ShaderTextureFormatKey::OutputRgba8Unorm => Ok(wgpu::TextureFormat::Rgba8Unorm),
        ShaderTextureFormatKey::WorkingHighPrecisionRgba16Float => {
            Ok(wgpu::TextureFormat::Rgba16Float)
        }
        ShaderTextureFormatKey::OutputBgra8Unorm => Ok(wgpu::TextureFormat::Bgra8Unorm),
    }
}
