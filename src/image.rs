use super::{BackendErrorCode, Error, PhysicalSize, Result, Size};
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

/// A compact image fingerprint or caller-supplied resource handle.
///
/// This copyable value is not a collision-free proof of pixel equality and
/// must not be used as the sole identity for a backend cache. Equality compares
/// only the underlying `u64`; the value carries no lifetime, generation, or
/// uniqueness guarantee. [`Image::from_rgba`] derives it deterministically,
/// while [`ImageId::new`] accepts a caller-managed handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ImageId(u64);

impl ImageId {
    /// Creates a compact image fingerprint or caller-supplied resource handle.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the compact underlying value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone)]
pub(crate) struct ImageContentIdentity {
    fingerprint: ImageId,
    content_hash: u64,
    width: u32,
    height: u32,
    bytes: Arc<[u8]>,
}

impl ImageContentIdentity {
    fn new(fingerprint: ImageId, width: u32, height: u32, bytes: Arc<[u8]>) -> Self {
        let content_hash = stable_hash(&(width, height, bytes.as_ref()));
        Self {
            fingerprint,
            content_hash,
            width,
            height,
            bytes,
        }
    }

    #[cfg(test)]
    const fn fingerprint(&self) -> ImageId {
        self.fingerprint
    }
}

impl fmt::Debug for ImageContentIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImageContentIdentity")
            .field("fingerprint", &self.fingerprint)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("byte_len", &self.bytes.len())
            .finish()
    }
}

impl PartialEq for ImageContentIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.bytes == other.bytes
    }
}

impl Eq for ImageContentIdentity {}

impl Hash for ImageContentIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.content_hash.hash(state);
    }
}

/// A validated, shareable RGBA8 image and its sampling configuration.
///
/// Exact render-owned content identity includes the dimensions and shared
/// pixel bytes. [`ImageId`] is retained only as a compact fingerprint, while
/// the Peniko blob carries its own unique backend identity. Image equality uses
/// exact content plus quality and extend policy rather than [`ImageId`] alone.
#[derive(Clone, Debug)]
pub struct Image {
    id: ImageId,
    content_identity: ImageContentIdentity,
    pub(crate) size: Size,
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) data: peniko::ImageData,
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.content_identity == other.content_identity
            && self.quality == other.quality
            && self.extend == other.extend
    }
}

impl Image {
    /// Creates a validated RGBA8 image with a deterministic compact fingerprint.
    ///
    /// Independently constructed images receive distinct backend blob IDs even
    /// when their dimensions and bytes are equal. Returns an image-upload
    /// diagnostic when dimensions are not finite non-negative integer pixels,
    /// exceed `u32`, or the byte length is not exactly `width * height * 4`.
    pub fn from_rgba(size: Size, data: impl Into<Arc<[u8]>>) -> Result<Self> {
        let data = data.into();
        validate_rgba_image(size, data.len())?;
        let id = stable_hash(&(
            size.width().to_bits(),
            size.height().to_bits(),
            data.as_ref(),
        ));
        Self::from_validated_rgba_with_id(size, data, ImageId::new(id))
    }

    fn from_validated_rgba_with_id(size: Size, data: Arc<[u8]>, id: ImageId) -> Result<Self> {
        let width = image_dimension(size.width(), "width")?;
        let height = image_dimension(size.height(), "height")?;
        let content_identity = ImageContentIdentity::new(id, width, height, Arc::clone(&data));
        let image = peniko::ImageData {
            data: peniko::Blob::new(Arc::new(Arc::clone(&data))),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::Alpha,
            width,
            height,
        };
        Ok(Self {
            id,
            content_identity,
            size,
            bytes: data,
            data: image,
            quality: ImageQuality::Medium,
            extend: Extend::Pad,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_rgba_with_id_for_test(
        size: Size,
        data: impl Into<Arc<[u8]>>,
        id: ImageId,
    ) -> Result<Self> {
        let data = data.into();
        validate_rgba_image(size, data.len())?;
        Self::from_validated_rgba_with_id(size, data, id)
    }

    /// Returns the compact content fingerprint.
    ///
    /// This value alone does not prove pixel equality or identify backend
    /// residency without collisions.
    #[must_use]
    pub const fn id(&self) -> ImageId {
        self.id
    }

    #[must_use]
    /// Returns the logical image dimensions, which represent integer pixel counts.
    pub const fn size(&self) -> Size {
        self.size
    }

    #[must_use]
    /// Returns this image with a different sampling-quality hint.
    ///
    /// The exact pixel content and backend blob identity are unchanged.
    pub const fn quality(mut self, quality: ImageQuality) -> Self {
        self.quality = quality;
        self
    }

    #[must_use]
    /// Returns this image with a different out-of-bounds sampling policy.
    ///
    /// The exact pixel content and backend blob identity are unchanged.
    pub const fn extend(mut self, extend: Extend) -> Self {
        self.extend = extend;
        self
    }

    pub(crate) const fn content_identity(&self) -> &ImageContentIdentity {
        &self.content_identity
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

/// A backend sampling-quality hint for an image.
///
/// The default is [`Self::Medium`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageQuality {
    /// Prefer lower-cost, lower-quality sampling.
    Low,
    /// Use the balanced default sampling quality.
    #[default]
    Medium,
    /// Prefer higher-quality sampling.
    High,
}

/// Image sampling behavior outside the source bounds.
///
/// The default is edge padding through [`Self::Pad`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Extend {
    /// Clamp sampling to the nearest edge pixel.
    #[default]
    Pad,
    /// Tile the image in the same orientation.
    Repeat,
    /// Tile the image while alternating orientation at each boundary.
    Reflect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedMaskUploadKey {
    content_identity: ImageContentIdentity,
    physical_size: PhysicalSize,
    quality: ImageQuality,
    extend: Extend,
}

impl ResolvedMaskUploadKey {
    const fn new(
        content_identity: ImageContentIdentity,
        physical_size: PhysicalSize,
        quality: ImageQuality,
        extend: Extend,
    ) -> Self {
        Self {
            content_identity,
            physical_size,
            quality,
            extend,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_image_for_test(
        image: &Image,
        physical_size: PhysicalSize,
        quality: ImageQuality,
        extend: Extend,
    ) -> Self {
        Self::new(
            image.content_identity().clone(),
            physical_size,
            quality,
            extend,
        )
    }

    #[cfg(test)]
    pub(crate) const fn image_id(&self) -> ImageId {
        self.content_identity.fingerprint()
    }

    #[cfg(test)]
    pub(crate) const fn physical_size(&self) -> PhysicalSize {
        self.physical_size
    }

    #[cfg(test)]
    pub(crate) const fn quality(&self) -> ImageQuality {
        self.quality
    }

    #[cfg(test)]
    pub(crate) const fn extend(&self) -> Extend {
        self.extend
    }
}

impl Hash for ResolvedMaskUploadKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.content_identity.hash(state);
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

/// Mapping policy for fitting an image into its destination rectangle.
///
/// The current default [`Self::None`] uses the same independent axis scaling as
/// [`Self::Fill`] and [`Self::Stretch`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageFit {
    /// Scale independently along each axis to fill the destination rectangle.
    Fill,
    /// Scale to fit entirely inside the destination while preserving aspect ratio.
    Contain,
    /// Scale to cover the destination while preserving aspect ratio.
    Cover,
    /// Stretch independently along each axis to match the destination rectangle.
    Stretch,
    /// Use the current default mapping, which scales independently to the destination rectangle.
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

    pub(crate) fn cache_key(&self) -> ResolvedMaskUploadKey {
        ResolvedMaskUploadKey::new(
            self.image.content_identity.clone(),
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

/// Converts the public sampling-quality hint to its equivalent Peniko value.
impl From<ImageQuality> for peniko::ImageQuality {
    fn from(quality: ImageQuality) -> Self {
        match quality {
            ImageQuality::Low => Self::Low,
            ImageQuality::Medium => Self::Medium,
            ImageQuality::High => Self::High,
        }
    }
}

/// Converts the public extension policy to its equivalent Peniko value.
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
