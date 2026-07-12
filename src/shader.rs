#![cfg_attr(not(test), allow(dead_code))]

use super::{
    BackendErrorCode, Color, Error, Format, Result,
    texture::{TextureDescriptor, TextureUsageIntent},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RectShaderPassKind {
    ClearFill,
    IdentityCopy,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectPassBounds {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl RectPassBounds {
    pub(crate) fn try_new(
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        source: TextureDescriptor,
        destination: TextureDescriptor,
    ) -> Result<Self> {
        if width == 0 {
            return Err(Error::invalid_value(
                "rect shader pass width",
                width,
                "must be greater than 0 device pixels",
            ));
        }
        if height == 0 {
            return Err(Error::invalid_value(
                "rect shader pass height",
                height,
                "must be greater than 0 device pixels",
            ));
        }
        Self::validate_texture_fit("source", x, y, width, height, source)?;
        Self::validate_texture_fit("destination", x, y, width, height, destination)?;
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    #[must_use]
    pub(crate) const fn x(self) -> u32 {
        self.x
    }

    #[must_use]
    pub(crate) const fn y(self) -> u32 {
        self.y
    }

    #[must_use]
    pub(crate) const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub(crate) const fn height(self) -> u32 {
        self.height
    }

    fn validate_texture_fit(
        texture_role: &'static str,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        descriptor: TextureDescriptor,
    ) -> Result<()> {
        let max_x = x.checked_add(width).ok_or_else(|| {
            Error::invalid_value(
                format!("rect shader pass {texture_role} x extent"),
                format!("{x}+{width}"),
                "must fit in u32 device pixels",
            )
        })?;
        let max_y = y.checked_add(height).ok_or_else(|| {
            Error::invalid_value(
                format!("rect shader pass {texture_role} y extent"),
                format!("{y}+{height}"),
                "must fit in u32 device pixels",
            )
        })?;
        let size = descriptor.physical_size();
        if max_x > size.width() {
            return Err(Error::invalid_value(
                format!("rect shader pass {texture_role} x extent"),
                max_x,
                "must fit inside the texture width",
            ));
        }
        if max_y > size.height() {
            return Err(Error::invalid_value(
                format!("rect shader pass {texture_role} y extent"),
                max_y,
                "must fit inside the texture height",
            ));
        }
        Ok(())
    }

    fn covers(self, descriptor: TextureDescriptor) -> bool {
        self.x == 0
            && self.y == 0
            && self.width == descriptor.physical_size().width()
            && self.height == descriptor.physical_size().height()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectShaderPassDescriptor {
    source_label: &'static str,
    destination_label: &'static str,
    source: TextureDescriptor,
    destination: TextureDescriptor,
    bounds: RectPassBounds,
    kind: RectShaderPassKind,
}

impl RectShaderPassDescriptor {
    pub(crate) fn try_new(
        source_label: &'static str,
        destination_label: &'static str,
        source: TextureDescriptor,
        destination: TextureDescriptor,
        bounds: RectPassBounds,
        kind: RectShaderPassKind,
    ) -> Result<Self> {
        validate_label("rect shader pass source texture", source_label)?;
        validate_label("rect shader pass destination texture", destination_label)?;
        RectPassBounds::validate_texture_fit(
            "source",
            bounds.x(),
            bounds.y(),
            bounds.width(),
            bounds.height(),
            source,
        )?;
        RectPassBounds::validate_texture_fit(
            "destination",
            bounds.x(),
            bounds.y(),
            bounds.width(),
            bounds.height(),
            destination,
        )?;
        if kind == RectShaderPassKind::ClearFill && !bounds.covers(destination) {
            return Err(Error::invalid_value(
                "rect shader clear/fill bounds",
                format!(
                    "{},{},{}x{} over {}x{}",
                    bounds.x(),
                    bounds.y(),
                    bounds.width(),
                    bounds.height(),
                    destination.physical_size().width(),
                    destination.physical_size().height()
                ),
                "must cover the full destination texture",
            ));
        }
        if kind == RectShaderPassKind::IdentityCopy && source.format() != destination.format() {
            return Err(Error::invalid_value(
                "rect shader pass texture formats",
                format!("{:?}->{:?}", source.format(), destination.format()),
                "must match for identity copy passes",
            ));
        }
        Ok(Self {
            source_label,
            destination_label,
            source,
            destination,
            bounds,
            kind,
        })
    }

    #[must_use]
    pub(crate) const fn source_label(self) -> &'static str {
        self.source_label
    }

    #[must_use]
    pub(crate) const fn destination_label(self) -> &'static str {
        self.destination_label
    }

    #[must_use]
    pub(crate) const fn source(self) -> TextureDescriptor {
        self.source
    }

    #[must_use]
    pub(crate) const fn destination(self) -> TextureDescriptor {
        self.destination
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> RectPassBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn kind(self) -> RectShaderPassKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RectShaderPipelineKey {
    kind: RectShaderPassKind,
    source_format: Format,
    destination_format: Format,
    source_intent: TextureUsageIntent,
    destination_intent: TextureUsageIntent,
}

impl RectShaderPipelineKey {
    #[must_use]
    pub(crate) const fn from_descriptor(descriptor: RectShaderPassDescriptor) -> Self {
        Self {
            kind: descriptor.kind,
            source_format: descriptor.source.format(),
            destination_format: descriptor.destination.format(),
            source_intent: descriptor.source.intent(),
            destination_intent: descriptor.destination.intent(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RectShaderPassGpuContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    source_view: &'a wgpu::TextureView,
    destination_view: &'a wgpu::TextureView,
}

impl<'a> RectShaderPassGpuContext<'a> {
    #[must_use]
    pub(crate) const fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        source_view: &'a wgpu::TextureView,
        destination_view: &'a wgpu::TextureView,
    ) -> Self {
        Self {
            device,
            queue,
            source_view,
            destination_view,
        }
    }
}

pub(crate) fn encode_clear_fill_pass(
    context: Option<RectShaderPassGpuContext<'_>>,
    descriptor: RectShaderPassDescriptor,
    color: Color,
) -> Result<()> {
    if descriptor.kind() != RectShaderPassKind::ClearFill {
        return Err(Error::invalid_value(
            "rect shader pass kind",
            format!("{:?}", descriptor.kind()),
            "must be ClearFill for clear/fill encoding",
        ));
    }
    let Some(context) = context else {
        return Err(Error::new(
            BackendErrorCode::AdapterUnavailable,
            "rect/fullscreen shader pass requires an available wgpu device context",
        ));
    };
    let _source_view = context.source_view;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist rect shader pass clear/fill"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(descriptor.destination_label()),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: context.destination_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(color.r()),
                        g: f64::from(color.g()),
                        b: f64::from(color.b()),
                        a: f64::from(color.a()),
                    }),
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
    context.queue.submit([encoder.finish()]);
    context
        .device
        .poll(wgpu::PollType::Poll)
        .map_err(|source| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "failed to poll render device",
            )
            .with_source(source)
        })?;
    Ok(())
}

fn validate_label(name: &'static str, value: &'static str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::invalid_value(name, value, "must not be empty"));
    }
    Ok(())
}
