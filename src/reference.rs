#![cfg_attr(not(test), allow(dead_code))]

use super::{Error, PhysicalSize, Result};

/// CPU reference pixel stored as premultiplied RGBA8.
///
/// Color channels are validated to be less than or equal to alpha. Source-over
/// uses integer math and rounds scaled destination terms with `(value + 127) /
/// 255` so oracle tests stay byte-deterministic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PremultipliedRgba8 {
    red: u8,
    green: u8,
    blue: u8,
    alpha: u8,
}

impl PremultipliedRgba8 {
    pub(crate) const TRANSPARENT: Self = Self {
        red: 0,
        green: 0,
        blue: 0,
        alpha: 0,
    };

    pub(crate) fn try_new(red: u8, green: u8, blue: u8, alpha: u8) -> Result<Self> {
        if red > alpha {
            return Err(Error::invalid_value(
                "premultiplied red",
                red,
                "must be less than or equal to alpha",
            ));
        }
        if green > alpha {
            return Err(Error::invalid_value(
                "premultiplied green",
                green,
                "must be less than or equal to alpha",
            ));
        }
        if blue > alpha {
            return Err(Error::invalid_value(
                "premultiplied blue",
                blue,
                "must be less than or equal to alpha",
            ));
        }
        Ok(Self {
            red,
            green,
            blue,
            alpha,
        })
    }

    pub(crate) fn apply_opacity(self, opacity: f32) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "reference opacity",
                opacity,
                "must be finite",
            ));
        }
        let opacity = opacity.clamp(0.0, 1.0);
        Ok(Self {
            red: scale_channel_by_opacity(self.red, opacity),
            green: scale_channel_by_opacity(self.green, opacity),
            blue: scale_channel_by_opacity(self.blue, opacity),
            alpha: scale_channel_by_opacity(self.alpha, opacity),
        })
    }

    pub(crate) const fn source_over(self, destination: Self) -> Self {
        if self.alpha == 0 {
            return destination;
        }
        if self.alpha == u8::MAX {
            return self;
        }
        if destination.alpha == 0 {
            return self;
        }
        let inverse_source_alpha = u8::MAX - self.alpha;
        Self {
            red: self.red + scale_channel_by_alpha(destination.red, inverse_source_alpha),
            green: self.green + scale_channel_by_alpha(destination.green, inverse_source_alpha),
            blue: self.blue + scale_channel_by_alpha(destination.blue, inverse_source_alpha),
            alpha: self.alpha + scale_channel_by_alpha(destination.alpha, inverse_source_alpha),
        }
    }
}

/// CPU reference image buffer with premultiplied RGBA8 pixels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReferencePremultipliedRgba8Buffer {
    physical_size: PhysicalSize,
    byte_len: u64,
    pixels: Vec<PremultipliedRgba8>,
}

impl ReferencePremultipliedRgba8Buffer {
    pub(crate) fn try_new(physical_size: PhysicalSize) -> Result<Self> {
        let pixel_count = validate_size(physical_size)?;
        Ok(Self {
            physical_size,
            byte_len: pixel_count
                .checked_mul(4)
                .expect("validated reference buffer byte length should fit u64"),
            pixels: vec![
                PremultipliedRgba8::TRANSPARENT;
                usize::try_from(pixel_count).expect(
                    "validated reference buffer pixel count should fit addressable memory"
                )
            ],
        })
    }

    pub(crate) fn from_pixels(
        physical_size: PhysicalSize,
        pixels: Vec<PremultipliedRgba8>,
    ) -> Result<Self> {
        let pixel_count = validate_size(physical_size)?;
        let expected_len = usize::try_from(pixel_count).map_err(|_| {
            Error::invalid_value(
                "reference buffer pixel count",
                pixel_count,
                "must fit addressable memory",
            )
        })?;
        if pixels.len() != expected_len {
            return Err(Error::invalid_value(
                "reference buffer pixel data length",
                pixels.len(),
                "must match width multiplied by height",
            ));
        }
        Ok(Self {
            physical_size,
            byte_len: pixel_count
                .checked_mul(4)
                .expect("validated reference buffer byte length should fit u64"),
            pixels,
        })
    }

    pub(crate) const fn physical_size(&self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn pixel(&self, x: u32, y: u32) -> Result<PremultipliedRgba8> {
        Ok(self.pixels[self.pixel_index(x, y)?])
    }

    pub(crate) fn set_pixel(&mut self, x: u32, y: u32, pixel: PremultipliedRgba8) -> Result<()> {
        let index = self.pixel_index(x, y)?;
        self.pixels[index] = pixel;
        Ok(())
    }

    pub(crate) fn apply_opacity(&self, opacity: f32) -> Result<Self> {
        let pixels = self
            .pixels
            .iter()
            .copied()
            .map(|pixel| pixel.apply_opacity(opacity))
            .collect::<Result<Vec<_>>>()?;
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn source_over(&self, destination: &Self) -> Result<Self> {
        if self.physical_size != destination.physical_size {
            return Err(Error::invalid_value(
                "reference source-over destination size",
                format!(
                    "{}x{}",
                    destination.physical_size.width(),
                    destination.physical_size.height()
                ),
                "must match source size",
            ));
        }
        let pixels = self
            .pixels
            .iter()
            .copied()
            .zip(destination.pixels.iter().copied())
            .map(|(source, destination)| source.source_over(destination))
            .collect();
        Self::from_pixels(self.physical_size, pixels)
    }

    fn pixel_index(&self, x: u32, y: u32) -> Result<usize> {
        if x >= self.physical_size.width() {
            return Err(Error::invalid_value(
                "reference buffer x",
                x,
                "must be inside the buffer width",
            ));
        }
        if y >= self.physical_size.height() {
            return Err(Error::invalid_value(
                "reference buffer y",
                y,
                "must be inside the buffer height",
            ));
        }
        let index = u64::from(y)
            .checked_mul(u64::from(self.physical_size.width()))
            .and_then(|row_start| row_start.checked_add(u64::from(x)))
            .ok_or_else(|| {
                Error::invalid_value(
                    "reference buffer pixel index",
                    format!("{x},{y}"),
                    "must fit addressable memory",
                )
            })?;
        usize::try_from(index).map_err(|_| {
            Error::invalid_value(
                "reference buffer pixel index",
                index,
                "must fit addressable memory",
            )
        })
    }
}

fn validate_size(physical_size: PhysicalSize) -> Result<u64> {
    if physical_size.width() == 0 {
        return Err(Error::invalid_value(
            "reference buffer width",
            physical_size.width(),
            "must be greater than 0 device pixels",
        ));
    }
    if physical_size.height() == 0 {
        return Err(Error::invalid_value(
            "reference buffer height",
            physical_size.height(),
            "must be greater than 0 device pixels",
        ));
    }
    let pixel_count = u64::from(physical_size.width())
        .checked_mul(u64::from(physical_size.height()))
        .ok_or_else(|| {
            Error::invalid_value(
                "reference buffer pixel count",
                format!("{}x{}", physical_size.width(), physical_size.height()),
                "must fit in u64",
            )
        })?;
    let _byte_len = pixel_count.checked_mul(4).ok_or_else(|| {
        Error::invalid_value(
            "reference buffer byte length",
            format!("{} pixels", pixel_count),
            "must fit in u64",
        )
    })?;
    usize::try_from(pixel_count).map_err(|_| {
        Error::invalid_value(
            "reference buffer pixel count",
            pixel_count,
            "must fit addressable memory",
        )
    })?;
    Ok(pixel_count)
}

fn scale_channel_by_opacity(channel: u8, opacity: f32) -> u8 {
    (f32::from(channel) * opacity).round().clamp(0.0, 255.0) as u8
}

const fn scale_channel_by_alpha(channel: u8, alpha: u8) -> u8 {
    let scaled = (channel as u16) * (alpha as u16) + 127;
    (scaled / 255) as u8
}
