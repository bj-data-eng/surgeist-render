use super::{BackendErrorCode, Error, PhysicalSize, Result, Size};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ImageId(u64);

impl ImageId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    id: ImageId,
    pub(crate) size: Size,
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) data: peniko::ImageData,
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
}

impl Image {
    pub fn from_rgba(size: Size, data: impl Into<Arc<[u8]>>) -> Result<Self> {
        let data = data.into();
        validate_rgba_image(size, data.len())?;
        let id = stable_hash(&(
            size.width().to_bits(),
            size.height().to_bits(),
            data.as_ref(),
        ));
        let id = ImageId::new(id);
        let width = image_dimension(size.width(), "width")?;
        let height = image_dimension(size.height(), "height")?;
        let image = peniko::ImageData {
            data: peniko::Blob::from_raw_parts(Arc::new(data.to_vec()), id.get()),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::Alpha,
            width,
            height,
        };
        Ok(Self {
            id,
            size,
            bytes: data,
            data: image,
            quality: ImageQuality::Medium,
            extend: Extend::Pad,
        })
    }

    #[must_use]
    pub const fn id(&self) -> ImageId {
        self.id
    }

    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    #[must_use]
    pub const fn quality(mut self, quality: ImageQuality) -> Self {
        self.quality = quality;
        self
    }

    #[must_use]
    pub const fn extend(mut self, extend: Extend) -> Self {
        self.extend = extend;
        self
    }
}

pub(crate) fn validate_rgba_image(size: Size, byte_len: usize) -> Result<()> {
    let width = image_dimension(size.width(), "width")?;
    let height = image_dimension(size.height(), "height")?;
    let expected_len = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(4);
    let actual_len = u64::try_from(byte_len).unwrap_or(u64::MAX);
    if actual_len != expected_len {
        return Err(Error::new(
            BackendErrorCode::ImageUploadFailed,
            format!(
                "RGBA image data length {byte_len} does not match {}x{} image size; expected {expected_len} bytes",
                width, height
            ),
        ));
    }
    Ok(())
}

pub(crate) fn image_dimension(value: f64, name: &str) -> Result<u32> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(Error::new(
            BackendErrorCode::ImageUploadFailed,
            format!("image {name} {value} must be a finite non-negative integer pixel size"),
        ));
    }
    Ok(value as u32)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageQuality {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Extend {
    #[default]
    Pad,
    Repeat,
    Reflect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedMaskUploadKey {
    image_id: ImageId,
    physical_size: PhysicalSize,
    quality: ImageQuality,
    extend: Extend,
}

impl ResolvedMaskUploadKey {
    pub(crate) const fn new(
        image_id: ImageId,
        physical_size: PhysicalSize,
        quality: ImageQuality,
        extend: Extend,
    ) -> Self {
        Self {
            image_id,
            physical_size,
            quality,
            extend,
        }
    }

    #[cfg(test)]
    pub(crate) const fn image_id(self) -> ImageId {
        self.image_id
    }

    #[cfg(test)]
    pub(crate) const fn physical_size(self) -> PhysicalSize {
        self.physical_size
    }

    #[cfg(test)]
    pub(crate) const fn quality(self) -> ImageQuality {
        self.quality
    }

    #[cfg(test)]
    pub(crate) const fn extend(self) -> Extend {
        self.extend
    }
}

impl Hash for ResolvedMaskUploadKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image_id.hash(state);
        self.physical_size.hash(state);
        match self.quality {
            ImageQuality::Low => 0_u8,
            ImageQuality::Medium => 1,
            ImageQuality::High => 2,
        }
        .hash(state);
        match self.extend {
            Extend::Pad => 0_u8,
            Extend::Repeat => 1,
            Extend::Reflect => 2,
        }
        .hash(state);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageFit {
    Fill,
    Contain,
    Cover,
    Stretch,
    #[default]
    None,
}

/// An intrinsically valid headless-readback buffer of straight-alpha RGBA8 physical pixels.
///
/// The byte length always equals `width * height * 4`. Zero-area buffers are
/// represented by an empty byte vector. This resolved CPU-visible value is
/// produced only by explicit [`crate::Renderer::read_headless`] or by validated
/// caller construction; production rendering does not consume it as a fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBuffer {
    size: PhysicalSize,
    rgba: Vec<u8>,
}

impl ImageBuffer {
    /// Creates a buffer when the straight-alpha RGBA8 byte length matches its physical size.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `width * height * 4`
    /// overflows addressable memory or differs from `rgba.len()`.
    pub fn try_new(size: PhysicalSize, rgba: Vec<u8>) -> Result<Self> {
        let expected_len = usize::try_from(size.width())
            .ok()
            .and_then(|width| {
                usize::try_from(size.height())
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixel_count| pixel_count.checked_mul(4))
            .ok_or_else(|| {
                Error::invalid_value(
                    "image buffer byte length",
                    format!("{}x{} RGBA8", size.width(), size.height()),
                    "must fit addressable memory",
                )
            })?;

        if rgba.len() != expected_len {
            return Err(Error::invalid_value(
                "image buffer RGBA data length",
                rgba.len(),
                "must equal width multiplied by height multiplied by 4",
            ));
        }

        Ok(Self { size, rgba })
    }

    /// Returns the physical pixel dimensions.
    #[must_use]
    pub const fn size(&self) -> PhysicalSize {
        self.size
    }

    /// Returns the tightly packed, row-major straight-alpha RGBA8 bytes.
    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Consumes the buffer and returns its tightly packed, row-major straight-alpha RGBA8 bytes.
    #[must_use]
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}

/// Backend-phase upload facts for one normalized resolved alpha-mask image.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedMaskUploadDescriptor {
    image: Image,
    physical_size: PhysicalSize,
    row_bytes: u32,
    byte_len: u64,
}

impl PartialEq for ResolvedMaskUploadDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.cache_key() == other.cache_key() && self.bytes() == other.bytes()
    }
}

impl Eq for ResolvedMaskUploadDescriptor {}

impl ResolvedMaskUploadDescriptor {
    pub(crate) fn try_from_image(image: Image) -> Result<Self> {
        let width = image_dimension(image.size.width(), "width")?;
        let height = image_dimension(image.size.height(), "height")?;
        let physical_size = PhysicalSize::new(width, height);
        let row_bytes = width.checked_mul(4).ok_or_else(|| {
            Error::invalid_value(
                "resolved mask upload row length",
                width,
                "must fit in u32 bytes",
            )
        })?;
        let byte_len = u64::from(row_bytes)
            .checked_mul(u64::from(height))
            .ok_or_else(|| {
                Error::invalid_value(
                    "resolved mask upload byte length",
                    format!("{row_bytes}x{height}"),
                    "must fit in u64",
                )
            })?;
        let descriptor = Self {
            image,
            physical_size,
            row_bytes,
            byte_len,
        };
        descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
        Ok(descriptor)
    }

    pub(crate) fn validate_upload_byte_len(&self, actual_len: usize) -> Result<()> {
        let actual_len = u64::try_from(actual_len).map_err(|_| {
            Error::invalid_value(
                "resolved mask upload byte length",
                actual_len,
                "must fit in u64",
            )
        })?;
        if actual_len != self.byte_len {
            return Err(Error::invalid_value(
                "resolved mask upload byte length",
                actual_len,
                "must equal width multiplied by height multiplied by four",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) const fn image(&self) -> &Image {
        &self.image
    }

    pub(crate) const fn physical_size(&self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn row_bytes(&self) -> u32 {
        self.row_bytes
    }

    pub(crate) const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) const fn quality(&self) -> ImageQuality {
        self.image.quality
    }

    pub(crate) const fn extend(&self) -> Extend {
        self.image.extend
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.image.bytes
    }

    pub(crate) const fn cache_key(&self) -> ResolvedMaskUploadKey {
        ResolvedMaskUploadKey::new(
            self.image.id,
            self.physical_size,
            self.image.quality,
            self.image.extend,
        )
    }
}

#[cfg(test)]
pub(crate) fn validate_image_buffer_rgba_len(size: PhysicalSize, byte_len: usize) -> Result<()> {
    if size.width() == 0 {
        return Err(Error::invalid_value(
            "image buffer width",
            size.width(),
            "must be greater than 0 device pixels",
        ));
    }
    if size.height() == 0 {
        return Err(Error::invalid_value(
            "image buffer height",
            size.height(),
            "must be greater than 0 device pixels",
        ));
    }
    let expected_len = u64::from(size.width())
        .checked_mul(u64::from(size.height()))
        .and_then(|pixel_count| pixel_count.checked_mul(4))
        .ok_or_else(|| {
            Error::invalid_value(
                "image buffer byte length",
                format!("{}x{}", size.width(), size.height()),
                "must fit in u64",
            )
        })?;
    let actual_len = u64::try_from(byte_len).unwrap_or(u64::MAX);
    if actual_len != expected_len {
        return Err(Error::invalid_value(
            "image buffer RGBA data length",
            byte_len,
            "must match width multiplied by height multiplied by 4",
        ));
    }
    Ok(())
}

impl From<ImageQuality> for peniko::ImageQuality {
    fn from(quality: ImageQuality) -> Self {
        match quality {
            ImageQuality::Low => Self::Low,
            ImageQuality::Medium => Self::Medium,
            ImageQuality::High => Self::High,
        }
    }
}

impl From<Extend> for peniko::Extend {
    fn from(extend: Extend) -> Self {
        match extend {
            Extend::Pad => Self::Pad,
            Extend::Repeat => Self::Repeat,
            Extend::Reflect => Self::Reflect,
        }
    }
}

fn stable_hash<T: std::hash::Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
