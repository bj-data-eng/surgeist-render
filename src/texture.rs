use super::{Error, Format, PhysicalSize, Result};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TextureUsageIntent {
    OffscreenLayer,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C07 runtime pass lowering consumes the intermediate texture role in T5"
        )
    )]
    IntermediatePass,
    ReadbackReference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureDescriptor {
    physical_size: PhysicalSize,
    format: Format,
    intent: TextureUsageIntent,
    byte_len: u64,
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
        let byte_len = pixel_count
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
            byte_len,
        })
    }

    pub(crate) const fn physical_size(self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn format(self) -> Format {
        self.format
    }

    pub(crate) const fn intent(self) -> TextureUsageIntent {
        self.intent
    }

    pub(crate) const fn byte_len(self) -> u64 {
        self.byte_len
    }

    pub(crate) const fn cache_key(self) -> TextureCacheKey {
        TextureCacheKey {
            physical_size: self.physical_size,
            format: self.format,
            intent: self.intent,
        }
    }

    pub(crate) const fn wgpu_usage(self) -> wgpu::TextureUsages {
        match (self.intent, self.format) {
            (
                TextureUsageIntent::OffscreenLayer | TextureUsageIntent::IntermediatePass,
                Format::Rgba8,
            ) => wgpu::TextureUsages::RENDER_ATTACHMENT
                .union(wgpu::TextureUsages::STORAGE_BINDING)
                .union(wgpu::TextureUsages::TEXTURE_BINDING)
                .union(wgpu::TextureUsages::COPY_SRC)
                .union(wgpu::TextureUsages::COPY_DST),
            (
                TextureUsageIntent::OffscreenLayer | TextureUsageIntent::IntermediatePass,
                Format::Bgra8,
            ) => wgpu::TextureUsages::RENDER_ATTACHMENT
                .union(wgpu::TextureUsages::TEXTURE_BINDING)
                .union(wgpu::TextureUsages::COPY_SRC)
                .union(wgpu::TextureUsages::COPY_DST),
            (TextureUsageIntent::ReadbackReference, _) => {
                wgpu::TextureUsages::STORAGE_BINDING.union(wgpu::TextureUsages::COPY_SRC)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureCacheKey {
    physical_size: PhysicalSize,
    format: Format,
    intent: TextureUsageIntent,
}

impl TextureCacheKey {
    pub(crate) const fn from_descriptor(descriptor: TextureDescriptor) -> Self {
        descriptor.cache_key()
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
