use super::{Error, Format, PhysicalSize, Result, resource::WorkingFormat};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum EffectTextureRole {
    Capture,
    Working,
    Coverage,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EffectTextureKey {
    role: EffectTextureRole,
    working_format: Option<WorkingFormat>,
    texture_format: wgpu::TextureFormat,
    physical_size: PhysicalSize,
    usage: wgpu::TextureUsages,
}

impl EffectTextureKey {
    pub(crate) const fn from_descriptor(descriptor: EffectTextureDescriptor) -> Self {
        Self {
            role: descriptor.role,
            working_format: descriptor.working_format,
            texture_format: descriptor.texture_format,
            physical_size: descriptor.physical_size,
            usage: descriptor.usage,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EffectTextureDescriptor {
    role: EffectTextureRole,
    working_format: Option<WorkingFormat>,
    texture_format: wgpu::TextureFormat,
    physical_size: PhysicalSize,
    usage: wgpu::TextureUsages,
}

impl EffectTextureDescriptor {
    pub(crate) fn try_capture(
        physical_size: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<Self> {
        Self::try_new(
            EffectTextureRole::Capture,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
            physical_size,
            usage,
        )
    }

    pub(crate) fn try_working(
        working_format: WorkingFormat,
        physical_size: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<Self> {
        Self::try_new(
            EffectTextureRole::Working,
            Some(working_format),
            working_format.texture_format(),
            physical_size,
            usage,
        )
    }

    pub(crate) fn try_coverage(
        physical_size: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<Self> {
        Self::try_new(
            EffectTextureRole::Coverage,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
            physical_size,
            usage,
        )
    }

    fn try_new(
        role: EffectTextureRole,
        working_format: Option<WorkingFormat>,
        texture_format: wgpu::TextureFormat,
        physical_size: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<Self> {
        if physical_size.width() == 0 || physical_size.height() == 0 {
            return Err(Error::invalid_value(
                "effect texture extent",
                format!("{}x{}", physical_size.width(), physical_size.height()),
                "must have positive width and height",
            ));
        }
        if usage.is_empty() {
            return Err(Error::invalid_value(
                "effect texture usage",
                "empty",
                "must contain at least one WGPU texture usage",
            ));
        }
        if !matches!(
            texture_format,
            wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Err(Error::invalid_value(
                "effect texture format",
                format!("{texture_format:?}"),
                "must be Rgba16Float or Rgba8Unorm",
            ));
        }

        Ok(Self {
            role,
            working_format,
            texture_format,
            physical_size,
            usage,
        })
    }

    pub(crate) const fn working_format(self) -> Option<WorkingFormat> {
        self.working_format
    }

    #[cfg(test)]
    pub(crate) const fn role(self) -> EffectTextureRole {
        self.role
    }

    pub(crate) const fn texture_format(self) -> wgpu::TextureFormat {
        self.texture_format
    }

    pub(crate) const fn physical_size(self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn usage(self) -> wgpu::TextureUsages {
        self.usage
    }

    pub(crate) fn checked_byte_len(self) -> Result<u64> {
        let bytes_per_pixel = match self.texture_format {
            wgpu::TextureFormat::Rgba16Float => 8_u64,
            wgpu::TextureFormat::Rgba8Unorm => 4_u64,
            _ => {
                return Err(Error::invalid_value(
                    "effect texture format",
                    format!("{:?}", self.texture_format),
                    "must be Rgba16Float or Rgba8Unorm",
                ));
            }
        };
        let pixel_count = u64::from(self.physical_size.width())
            .checked_mul(u64::from(self.physical_size.height()))
            .ok_or_else(|| {
                Error::invalid_value(
                    "effect texture pixel count",
                    format!(
                        "{}x{}",
                        self.physical_size.width(),
                        self.physical_size.height()
                    ),
                    "must fit in u64",
                )
            })?;
        pixel_count.checked_mul(bytes_per_pixel).ok_or_else(|| {
            Error::invalid_value(
                "effect texture byte length",
                format!("{pixel_count} pixels at {bytes_per_pixel} bytes per pixel"),
                "must fit in u64",
            )
        })
    }

    pub(crate) const fn cache_key(self) -> EffectTextureKey {
        EffectTextureKey::from_descriptor(self)
    }

    pub(crate) const fn label(self) -> &'static str {
        match self.role {
            EffectTextureRole::Capture => "Surgeist retained capture texture",
            EffectTextureRole::Working => "Surgeist retained working texture",
            EffectTextureRole::Coverage => "Surgeist retained coverage texture",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TextureUsageIntent {
    #[cfg(test)]
    IntermediatePass,
    ReadbackReference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureDescriptor {
    physical_size: PhysicalSize,
    format: Format,
    intent: TextureUsageIntent,
}

impl TextureDescriptor {
    pub(crate) fn try_new(
        physical_size: PhysicalSize,
        format: Format,
        intent: TextureUsageIntent,
    ) -> Result<Self> {
        if physical_size.width() == 0 {
            return Err(Error::invalid_value(
                "texture width",
                physical_size.width(),
                "must be greater than 0 device pixels",
            ));
        }
        if physical_size.height() == 0 {
            return Err(Error::invalid_value(
                "texture height",
                physical_size.height(),
                "must be greater than 0 device pixels",
            ));
        }
        let pixel_count = u64::from(physical_size.width())
            .checked_mul(u64::from(physical_size.height()))
            .ok_or_else(|| {
                Error::invalid_value(
                    "texture pixel count",
                    format!("{}x{}", physical_size.width(), physical_size.height()),
                    "must fit in u64",
                )
            })?;
        pixel_count
            .checked_mul(u64::from(format.bytes_per_pixel()))
            .ok_or_else(|| {
                Error::invalid_value(
                    "texture byte length",
                    format!("{} pixels", pixel_count),
                    "must fit in u64",
                )
            })?;
        Ok(Self {
            physical_size,
            format,
            intent,
        })
    }

    pub(crate) const fn physical_size(self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn format(self) -> Format {
        self.format
    }

    #[cfg(test)]
    pub(crate) const fn intent(self) -> TextureUsageIntent {
        self.intent
    }

    pub(crate) const fn wgpu_usage(self) -> wgpu::TextureUsages {
        match (self.intent, self.format) {
            #[cfg(test)]
            (TextureUsageIntent::IntermediatePass, Format::Rgba8) => {
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    .union(wgpu::TextureUsages::STORAGE_BINDING)
                    .union(wgpu::TextureUsages::TEXTURE_BINDING)
                    .union(wgpu::TextureUsages::COPY_SRC)
                    .union(wgpu::TextureUsages::COPY_DST)
            }
            #[cfg(test)]
            (TextureUsageIntent::IntermediatePass, Format::Bgra8) => {
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    .union(wgpu::TextureUsages::TEXTURE_BINDING)
                    .union(wgpu::TextureUsages::COPY_SRC)
                    .union(wgpu::TextureUsages::COPY_DST)
            }
            (TextureUsageIntent::ReadbackReference, _) => wgpu::TextureUsages::RENDER_ATTACHMENT
                .union(wgpu::TextureUsages::STORAGE_BINDING)
                .union(wgpu::TextureUsages::TEXTURE_BINDING)
                .union(wgpu::TextureUsages::COPY_SRC)
                .union(wgpu::TextureUsages::COPY_DST),
        }
    }
}

pub(crate) fn headless_texture_descriptor(
    physical_size: PhysicalSize,
    format: Format,
) -> Result<TextureDescriptor> {
    TextureDescriptor::try_new(
        PhysicalSize::new(physical_size.width().max(1), physical_size.height().max(1)),
        format,
        TextureUsageIntent::ReadbackReference,
    )
}

trait TextureFormatExt {
    fn bytes_per_pixel(self) -> u8;
}

impl TextureFormatExt for Format {
    fn bytes_per_pixel(self) -> u8 {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
        }
    }
}
