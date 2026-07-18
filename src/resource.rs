#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkingFormat {
    HighPrecision,
    ReducedPrecision,
}

impl WorkingFormat {
    pub(crate) const fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            Self::HighPrecision => wgpu::TextureFormat::Rgba16Float,
            Self::ReducedPrecision => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    pub(crate) const fn required_usages(self) -> wgpu::TextureUsages {
        let _ = self;
        wgpu::TextureUsages::RENDER_ATTACHMENT
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST)
    }

    pub(crate) const fn required_format_features(self) -> wgpu::TextureFormatFeatureFlags {
        let _ = self;
        wgpu::TextureFormatFeatureFlags::FILTERABLE
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C07 resource accounting consumes this owned format fact after lifecycle modeling"
        )
    )]
    pub(crate) const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::HighPrecision => 8,
            Self::ReducedPrecision => 4,
        }
    }

    pub(crate) fn is_supported_by(self, features: wgpu::TextureFormatFeatures) -> bool {
        features.allowed_usages.contains(self.required_usages())
            && features.flags.contains(self.required_format_features())
    }
}
