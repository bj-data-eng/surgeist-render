use super::{
    BackendErrorCode, Error, FilterList, FilteredImagePaint, MaskMode, PhysicalSize,
    PrimitiveFamily, PrimitiveOperation, Result, Size, UnsupportedPrimitive,
    filter::{
        BlurPolicy, DevicePixelConversionPolicy, FilterClipBounds, FilterOutset, FilterRegionPlan,
        FilterSourceBounds, MaterializedImageFilterPipeline, MaterializedImageFilterStep,
    },
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
};
use std::{hash::Hasher, sync::Arc};

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

fn validate_rgba_image(size: Size, byte_len: usize) -> Result<()> {
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

fn image_dimension(value: f64, name: &str) -> Result<u32> {
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageFit {
    Fill,
    Contain,
    Cover,
    Stretch,
    #[default]
    None,
}

/// An intrinsically valid, tightly packed straight-alpha RGBA8 pixel buffer.
///
/// The byte length always equals `width * height * 4`. Zero-area buffers are
/// represented by an empty byte vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageBuffer {
    size: PhysicalSize,
    rgba: Vec<u8>,
}

impl ImageBuffer {
    /// Creates an image buffer when its RGBA byte length exactly matches its size.
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

    /// Returns the tightly packed straight-alpha RGBA8 bytes.
    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Consumes the buffer and returns its tightly packed RGBA8 bytes.
    #[must_use]
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}

/// Executable boundary for an already-resolved alpha mask over materialized pixels.
///
/// This type does not resolve CSS mask sources, resource references, placement,
/// shape rasterization, transforms, or luminance conversion. Callers must supply
/// the source pixels and the alpha-mask pixels at the same physical size.
#[derive(Debug)]
pub struct ResolvedAlphaMaskExecution<'a> {
    source: &'a ImageBuffer,
    alpha_mask: &'a ImageBuffer,
}

impl<'a> ResolvedAlphaMaskExecution<'a> {
    pub fn try_new(source: &'a ImageBuffer, alpha_mask: &'a ImageBuffer) -> Result<Self> {
        Self::try_new_with_mode(source, alpha_mask, MaskMode::Alpha)
    }

    pub fn try_new_with_mode(
        source: &'a ImageBuffer,
        alpha_mask: &'a ImageBuffer,
        mode: MaskMode,
    ) -> Result<Self> {
        match mode {
            MaskMode::Alpha => {}
            MaskMode::Luminance => {
                return Err(Error::unsupported_render_primitive(
                    UnsupportedPrimitive::new(
                        PrimitiveFamily::MasksAndClips,
                        PrimitiveOperation::LuminanceMaskMode,
                    ),
                ));
            }
        }

        validate_image_buffer_rgba_len(source.size(), source.rgba().len())?;
        validate_image_buffer_rgba_len(alpha_mask.size(), alpha_mask.rgba().len())?;
        if source.size() != alpha_mask.size() {
            return Err(Error::invalid_value(
                "resolved alpha mask size",
                format!(
                    "{}x{}",
                    alpha_mask.size().width(),
                    alpha_mask.size().height()
                ),
                "must match source size",
            ));
        }

        Ok(Self { source, alpha_mask })
    }

    pub fn execute_to_image_buffer(&self) -> Result<ImageBuffer> {
        let source = straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(self.source)?;
        let alpha_mask =
            straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(self.alpha_mask)?;
        let masked = source.apply_alpha_mask(&alpha_mask)?;
        premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&masked)
    }
}

/// Render-local boundary for a resolved image/filter intent plus materialized RGBA bytes.
///
/// `FilteredImagePaint` names the resolved resource and authored filter list, but the
/// bytes come from the paired `Image`. The execution phase converts straight RGBA8
/// image bytes to premultiplied RGBA8 reference pixels, applies the ordered
/// materialized-image filter pipeline, then emits straight RGBA8 again for
/// paint/upload.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct ResolvedMaterializedImageFilterExecution<'a> {
    source: ResolvedMaterializedImageFilterSource<'a>,
    pipeline: MaterializedImageFilterPipeline,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) type ResolvedImageColorFilterExecution<'a> =
    ResolvedMaterializedImageFilterExecution<'a>;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
enum ResolvedMaterializedImageFilterSource<'a> {
    Image(&'a Image),
    ImageBuffer(&'a ImageBuffer),
}

#[cfg_attr(not(test), allow(dead_code))]
impl<'a> ResolvedMaterializedImageFilterExecution<'a> {
    pub(crate) fn try_new(paint: &FilteredImagePaint, image: &'a Image) -> Result<Self> {
        let pipeline = compile_materialized_image_filter_pipeline(paint.filters())?;
        if paint.resource().id() != image.id() {
            return Err(Error::invalid_value(
                "materialized filtered image id",
                image.id().get(),
                "must match the resolved image resource id",
            ));
        }
        if paint.resource().intrinsic_size() != image.size() {
            return Err(Error::invalid_value(
                "materialized filtered image size",
                format!("{:?}", image.size()),
                "must match the resolved image resource intrinsic size",
            ));
        }
        Ok(Self {
            source: ResolvedMaterializedImageFilterSource::Image(image),
            pipeline,
        })
    }

    pub(crate) fn try_new_for_image_buffer(
        filters: &FilterList,
        image_buffer: &'a ImageBuffer,
    ) -> Result<Self> {
        let pipeline = compile_materialized_image_filter_pipeline(filters)?;
        validate_image_buffer_rgba_len(image_buffer.size(), image_buffer.rgba().len())?;
        Ok(Self {
            source: ResolvedMaterializedImageFilterSource::ImageBuffer(image_buffer),
            pipeline,
        })
    }

    pub(crate) fn execute_to_image(&self) -> Result<Image> {
        let ResolvedMaterializedImageFilterSource::Image(image) = self.source else {
            return Err(Error::invalid_value(
                "color-filtered image execution source",
                "image buffer",
                "must be a materialized Image when producing Image output",
            ));
        };
        let premultiplied = straight_rgba8_image_to_premultiplied_rgba8_reference(image)?;
        let filtered = execute_materialized_filter_pipeline(&premultiplied, &self.pipeline)?;
        let straight = premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&filtered)?;
        let mut filtered_image =
            Image::from_rgba(image.size(), Arc::<[u8]>::from(straight.into_rgba()))?;
        filtered_image.quality = image.quality;
        filtered_image.extend = image.extend;
        Ok(filtered_image)
    }

    pub(crate) fn execute_to_image_buffer(&self) -> Result<ImageBuffer> {
        let ResolvedMaterializedImageFilterSource::ImageBuffer(image_buffer) = self.source else {
            return Err(Error::invalid_value(
                "color-filtered image buffer execution source",
                "image",
                "must be an ImageBuffer when producing ImageBuffer output",
            ));
        };
        let premultiplied =
            straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(image_buffer)?;
        let filtered = execute_materialized_filter_pipeline(&premultiplied, &self.pipeline)?;
        premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&filtered)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn straight_rgba8_image_to_premultiplied_rgba8_reference(
    image: &Image,
) -> Result<ReferencePremultipliedRgba8Buffer> {
    validate_rgba_image(image.size, image.bytes.len())?;
    let size = PhysicalSize::new(
        image_dimension(image.size.width(), "width")?,
        image_dimension(image.size.height(), "height")?,
    );
    straight_rgba8_bytes_to_premultiplied_rgba8_reference(size, &image.bytes)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(
    image_buffer: &ImageBuffer,
) -> Result<ReferencePremultipliedRgba8Buffer> {
    validate_image_buffer_rgba_len(image_buffer.size(), image_buffer.rgba().len())?;
    straight_rgba8_bytes_to_premultiplied_rgba8_reference(image_buffer.size(), image_buffer.rgba())
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(
    buffer: &ReferencePremultipliedRgba8Buffer,
) -> Result<ImageBuffer> {
    let mut rgba = Vec::with_capacity(usize::try_from(buffer.byte_len()).map_err(|_| {
        Error::invalid_value(
            "premultiplied reference buffer byte length",
            buffer.byte_len(),
            "must fit addressable memory",
        )
    })?);
    let size = buffer.physical_size();

    for y in 0..size.height() {
        for x in 0..size.width() {
            let pixel = buffer.pixel(x, y)?;
            rgba.extend_from_slice(&premultiplied_rgba8_pixel_to_straight_rgba8(pixel));
        }
    }

    ImageBuffer::try_new(size, rgba)
}

#[cfg_attr(not(test), allow(dead_code))]
fn compile_materialized_image_filter_pipeline(
    filters: &FilterList,
) -> Result<MaterializedImageFilterPipeline> {
    let pipeline = filters
        .materialized_image_filter_pipeline()?
        .ok_or_else(|| {
            Error::invalid_value(
                "materialized image filters",
                "none",
                "must contain at least one filter operation",
            )
        })?;
    Ok(pipeline)
}

#[cfg_attr(not(test), allow(dead_code))]
fn execute_materialized_filter_pipeline(
    source: &ReferencePremultipliedRgba8Buffer,
    pipeline: &MaterializedImageFilterPipeline,
) -> Result<ReferencePremultipliedRgba8Buffer> {
    let mut current = source.clone();
    for step in pipeline.steps() {
        current = match step {
            MaterializedImageFilterStep::ColorFilters(pipeline) => {
                current.apply_compiled_color_filter_pipeline(pipeline)?
            }
            MaterializedImageFilterStep::Blur(blur) => {
                let planned_size = plan_clipped_materialized_blur_output_size(
                    current.physical_size(),
                    *blur,
                    BlurPolicy::css_filter_default(),
                )?;
                let blurred = current.apply_blur(*blur, BlurPolicy::css_filter_default())?;
                if blurred.physical_size() != planned_size {
                    return Err(Error::invalid_value(
                        "materialized blur output size",
                        format!(
                            "{}x{}",
                            blurred.physical_size().width(),
                            blurred.physical_size().height()
                        ),
                        "must match the clipped materialized image filter region",
                    ));
                }
                blurred
            }
            MaterializedImageFilterStep::DropShadow(shadow) => {
                let planned_size = plan_clipped_materialized_drop_shadow_output_size(
                    current.physical_size(),
                    shadow,
                    BlurPolicy::css_filter_default(),
                )?;
                let shadowed =
                    current.apply_drop_shadow(shadow, BlurPolicy::css_filter_default())?;
                if shadowed.physical_size() != planned_size {
                    return Err(Error::invalid_value(
                        "materialized drop-shadow output size",
                        format!(
                            "{}x{}",
                            shadowed.physical_size().width(),
                            shadowed.physical_size().height()
                        ),
                        "must match the clipped materialized image filter region",
                    ));
                }
                shadowed
            }
        };
    }
    Ok(current)
}

#[cfg_attr(not(test), allow(dead_code))]
fn plan_clipped_materialized_blur_output_size(
    size: PhysicalSize,
    blur: super::FilterBlur,
    policy: BlurPolicy,
) -> Result<PhysicalSize> {
    let source_rect = super::Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height()));
    let source = FilterSourceBounds::try_new(source_rect)?;
    let outset = FilterOutset::from_blur(blur, policy)?;
    let clip = FilterClipBounds::try_new(source_rect)?;
    let region = FilterRegionPlan::try_new(source, outset, Some(clip))?;
    let device_bounds =
        DevicePixelConversionPolicy::outward().convert_region(region.execution_region(), 1.0)?;
    if device_bounds.x() != 0 || device_bounds.y() != 0 {
        return Err(Error::invalid_value(
            "materialized blur output origin",
            format!("{},{}", device_bounds.x(), device_bounds.y()),
            "must remain anchored to the source image origin after clipping",
        ));
    }
    Ok(PhysicalSize::new(
        device_bounds.width(),
        device_bounds.height(),
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
fn plan_clipped_materialized_drop_shadow_output_size(
    size: PhysicalSize,
    shadow: &super::FilterDropShadow,
    policy: BlurPolicy,
) -> Result<PhysicalSize> {
    let source_rect = super::Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height()));
    let source = FilterSourceBounds::try_new(source_rect)?;
    let outset = FilterOutset::from_drop_shadow(shadow, policy)?;
    let clip = FilterClipBounds::try_new(source_rect)?;
    let region = FilterRegionPlan::try_new(source, outset, Some(clip))?;
    let device_bounds =
        DevicePixelConversionPolicy::outward().convert_region(region.execution_region(), 1.0)?;
    if device_bounds.x() != 0 || device_bounds.y() != 0 {
        return Err(Error::invalid_value(
            "materialized drop-shadow output origin",
            format!("{},{}", device_bounds.x(), device_bounds.y()),
            "must remain anchored to the source image origin after clipping",
        ));
    }
    Ok(PhysicalSize::new(
        device_bounds.width(),
        device_bounds.height(),
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
fn straight_rgba8_bytes_to_premultiplied_rgba8_reference(
    size: PhysicalSize,
    rgba: &[u8],
) -> Result<ReferencePremultipliedRgba8Buffer> {
    validate_image_buffer_rgba_len(size, rgba.len())?;
    let pixels = rgba
        .chunks_exact(4)
        .map(|pixel| {
            straight_rgba8_pixel_to_premultiplied_rgba8(pixel[0], pixel[1], pixel[2], pixel[3])
        })
        .collect::<Result<Vec<_>>>()?;
    ReferencePremultipliedRgba8Buffer::from_pixels(size, pixels)
}

#[cfg_attr(not(test), allow(dead_code))]
fn straight_rgba8_pixel_to_premultiplied_rgba8(
    red: u8,
    green: u8,
    blue: u8,
    alpha: u8,
) -> Result<PremultipliedRgba8> {
    if alpha == 0 {
        return Ok(PremultipliedRgba8::TRANSPARENT);
    }

    PremultipliedRgba8::try_new(
        premultiply_straight_rgba8_channel(red, alpha),
        premultiply_straight_rgba8_channel(green, alpha),
        premultiply_straight_rgba8_channel(blue, alpha),
        alpha,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn premultiplied_rgba8_pixel_to_straight_rgba8(pixel: PremultipliedRgba8) -> [u8; 4] {
    if pixel.alpha() == 0 {
        return [0, 0, 0, 0];
    }

    [
        unpremultiply_rgba8_channel(pixel.red(), pixel.alpha()),
        unpremultiply_rgba8_channel(pixel.green(), pixel.alpha()),
        unpremultiply_rgba8_channel(pixel.blue(), pixel.alpha()),
        pixel.alpha(),
    ]
}

#[cfg_attr(not(test), allow(dead_code))]
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

#[cfg_attr(not(test), allow(dead_code))]
fn premultiply_straight_rgba8_channel(channel: u8, alpha: u8) -> u8 {
    round_byte(f64::from(channel) * f64::from(alpha) / 255.0)
}

#[cfg_attr(not(test), allow(dead_code))]
fn unpremultiply_rgba8_channel(channel: u8, alpha: u8) -> u8 {
    round_byte(f64::from(channel) * 255.0 / f64::from(alpha))
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

#[cfg_attr(not(test), allow(dead_code))]
fn round_byte(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}
