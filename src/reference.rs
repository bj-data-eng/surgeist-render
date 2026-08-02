use super::{
    Error, Extend, FilterList, FilteredImagePaint, Image, ImageBuffer, ImageQuality, PhysicalSize,
    Point, Rect, ResolvedLayerAlphaMask, Result,
    filter::{
        BlurPolicy, DevicePixelConversionPolicy, FilterClipBounds, FilterOutset, FilterRegionPlan,
        FilterSourceBounds, TransparentEdgeSamplingPolicy,
    },
    image::{image_dimension, validate_image_buffer_rgba_len, validate_rgba_image},
    layer::BlendMode,
    style::{
        ColorFilterOp, ColorFilterPipeline, FilterBlur, FilterDropShadow, FilterOpKind,
        UnitFilterAmount,
    },
};
use std::sync::Arc;

const GRAYSCALE_LUMA_RED: f64 = 0.2126;
const GRAYSCALE_LUMA_GREEN: f64 = 0.7152;
const GRAYSCALE_LUMA_BLUE: f64 = 0.0722;
const SATURATION_LUMA_RED: f64 = 0.213;
const SATURATION_LUMA_GREEN: f64 = 0.715;
const SATURATION_LUMA_BLUE: f64 = 0.072;

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

    pub(crate) const fn red(self) -> u8 {
        self.red
    }

    pub(crate) const fn green(self) -> u8 {
        self.green
    }

    pub(crate) const fn blue(self) -> u8 {
        self.blue
    }

    pub(crate) const fn alpha(self) -> u8 {
        self.alpha
    }

    pub(crate) fn apply_opacity(self, opacity: f32) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "reference opacity",
                opacity,
                "must be finite",
            ));
        }
        Ok(self.apply_opacity_amount(f64::from(opacity)))
    }

    pub(crate) fn apply_color_filter_pipeline(
        self,
        pipeline: &ColorFilterPipeline,
    ) -> Result<Self> {
        let mut pixel = self;
        for op in pipeline.ops() {
            pixel = pixel.apply_color_filter_op(*op);
        }
        Ok(pixel)
    }

    pub(crate) fn apply_compiled_color_filter_pipeline(
        self,
        pipeline: &CompiledColorFilterPipeline,
    ) -> Result<Self> {
        apply_compiled_color_filter_pipeline_to_pixel(self, pipeline)
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

    pub(crate) fn blend_over(self, destination: Self, mode: BlendMode) -> Self {
        match mode {
            BlendMode::Normal => self.source_over(destination),
            BlendMode::Plus => self.plus_lighter(destination),
            BlendMode::Multiply
            | BlendMode::Screen
            | BlendMode::Overlay
            | BlendMode::Darken
            | BlendMode::Lighten => self.mix_blend_over(destination, mode),
        }
    }

    const fn plus_lighter(self, destination: Self) -> Self {
        Self {
            red: self.red.saturating_add(destination.red),
            green: self.green.saturating_add(destination.green),
            blue: self.blue.saturating_add(destination.blue),
            alpha: self.alpha.saturating_add(destination.alpha),
        }
    }

    fn mix_blend_over(self, destination: Self, mode: BlendMode) -> Self {
        if self.alpha == 0 {
            return destination;
        }
        if destination.alpha == 0 {
            return self;
        }

        let output_alpha = self.source_over(destination).alpha;
        let context = MixBlendContext {
            source_alpha: f64::from(self.alpha) / 255.0,
            destination_alpha: f64::from(destination.alpha) / 255.0,
            mode,
            output_alpha,
        };
        Self {
            red: mix_blend_channel(
                self.red,
                self.alpha,
                destination.red,
                destination.alpha,
                context,
            ),
            green: mix_blend_channel(
                self.green,
                self.alpha,
                destination.green,
                destination.alpha,
                context,
            ),
            blue: mix_blend_channel(
                self.blue,
                self.alpha,
                destination.blue,
                destination.alpha,
                context,
            ),
            alpha: output_alpha,
        }
    }

    pub(crate) const fn source_in_alpha_of(self, destination: Self) -> Self {
        self.scale_by_alpha(destination.alpha)
    }

    pub(crate) const fn destination_in_alpha_of(self, source: Self) -> Self {
        self.scale_by_alpha(source.alpha)
    }

    const fn scale_by_alpha(self, alpha: u8) -> Self {
        if alpha == 0 || self.alpha == 0 {
            return Self::TRANSPARENT;
        }
        if alpha == u8::MAX {
            return self;
        }
        Self {
            red: scale_channel_by_alpha(self.red, alpha),
            green: scale_channel_by_alpha(self.green, alpha),
            blue: scale_channel_by_alpha(self.blue, alpha),
            alpha: scale_channel_by_alpha(self.alpha, alpha),
        }
    }

    fn apply_color_filter_op(self, op: ColorFilterOp) -> Self {
        match op {
            ColorFilterOp::Brightness(amount) => {
                self.apply_straight_color_filter(|rgb| rgb.map(|channel| channel * amount.value()))
            }
            ColorFilterOp::Contrast(amount) => self.apply_straight_color_filter(|rgb| {
                rgb.map(|channel| (channel - 0.5) * amount.value() + 0.5)
            }),
            ColorFilterOp::Grayscale(amount) => self.apply_straight_color_filter(|rgb| {
                let gray = rgb.grayscale_luma();
                rgb.mix(StraightRgb::new(gray, gray, gray), amount.value())
            }),
            ColorFilterOp::HueRotate(angle) => self.apply_straight_color_filter(|rgb| {
                let (sin, cos) = angle.radians().sin_cos();
                StraightRgb::new(
                    (0.213 + cos * 0.787 - sin * 0.213) * rgb.red
                        + (0.715 - cos * 0.715 - sin * 0.715) * rgb.green
                        + (0.072 - cos * 0.072 + sin * 0.928) * rgb.blue,
                    (0.213 - cos * 0.213 + sin * 0.143) * rgb.red
                        + (0.715 + cos * 0.285 + sin * 0.140) * rgb.green
                        + (0.072 - cos * 0.072 - sin * 0.283) * rgb.blue,
                    (0.213 - cos * 0.213 - sin * 0.787) * rgb.red
                        + (0.715 - cos * 0.715 + sin * 0.715) * rgb.green
                        + (0.072 + cos * 0.928 + sin * 0.072) * rgb.blue,
                )
            }),
            ColorFilterOp::Invert(amount) => self.apply_straight_color_filter(|rgb| {
                rgb.map(|channel| {
                    channel * (1.0 - amount.value()) + (1.0 - channel) * amount.value()
                })
            }),
            ColorFilterOp::Opacity(amount) => self.apply_opacity_amount(amount.value()),
            ColorFilterOp::Saturate(amount) => self.apply_straight_color_filter(|rgb| {
                let gray = rgb.saturation_luma();
                StraightRgb::new(gray, gray, gray).mix(rgb, amount.value())
            }),
            ColorFilterOp::Sepia(amount) => self.apply_straight_color_filter(|rgb| {
                let sepia = StraightRgb::new(
                    rgb.red * 0.393 + rgb.green * 0.769 + rgb.blue * 0.189,
                    rgb.red * 0.349 + rgb.green * 0.686 + rgb.blue * 0.168,
                    rgb.red * 0.272 + rgb.green * 0.534 + rgb.blue * 0.131,
                );
                rgb.mix(sepia, amount.value())
            }),
        }
    }

    fn apply_straight_color_filter(self, filter: impl FnOnce(StraightRgb) -> StraightRgb) -> Self {
        let Some(straight) = StraightRgb::from_premultiplied(self) else {
            return Self::TRANSPARENT;
        };
        Self::from_straight_rgb(filter(straight).clamp_unit(), self.alpha)
    }

    pub(crate) fn apply_opacity_amount(self, opacity: f64) -> Self {
        let opacity = opacity.clamp(0.0, 1.0);
        Self {
            red: scale_channel_by_opacity(self.red, opacity),
            green: scale_channel_by_opacity(self.green, opacity),
            blue: scale_channel_by_opacity(self.blue, opacity),
            alpha: scale_channel_by_opacity(self.alpha, opacity),
        }
    }

    pub(crate) fn from_straight_color_channels(red: f64, green: f64, blue: f64, alpha: u8) -> Self {
        Self::from_straight_rgb(StraightRgb::new(red, green, blue).clamp_unit(), alpha)
    }

    fn from_straight_rgb(rgb: StraightRgb, alpha: u8) -> Self {
        if alpha == 0 {
            return Self::TRANSPARENT;
        }
        Self {
            red: premultiply_straight_channel(rgb.red, alpha),
            green: premultiply_straight_channel(rgb.green, alpha),
            blue: premultiply_straight_channel(rgb.blue, alpha),
            alpha,
        }
    }
}

fn apply_compiled_color_filter_pipeline_to_pixel(
    mut pixel: PremultipliedRgba8,
    pipeline: &CompiledColorFilterPipeline,
) -> Result<PremultipliedRgba8> {
    for step in pipeline.executable_steps() {
        pixel = match step {
            CompiledColorFilterStep::Identity => pixel,
            CompiledColorFilterStep::TransparentBlack => PremultipliedRgba8::TRANSPARENT,
            CompiledColorFilterStep::StraightColorRun(transforms) => {
                let mut filtered = pixel;
                for transform in transforms {
                    filtered = apply_compiled_straight_color_transform(filtered, *transform);
                }
                filtered
            }
            CompiledColorFilterStep::Opacity(amount) => pixel.apply_opacity_amount(amount.value()),
        };
    }
    Ok(pixel)
}

fn apply_compiled_straight_color_transform(
    pixel: PremultipliedRgba8,
    transform: StraightColorTransform,
) -> PremultipliedRgba8 {
    if pixel.alpha() == 0 {
        return PremultipliedRgba8::TRANSPARENT;
    }

    let matrix = transform.matrix();
    let alpha = f64::from(pixel.alpha());
    let red = f64::from(pixel.red()) / alpha;
    let green = f64::from(pixel.green()) / alpha;
    let blue = f64::from(pixel.blue()) / alpha;
    PremultipliedRgba8::from_straight_color_channels(
        matrix[0][0] * red + matrix[0][1] * green + matrix[0][2] * blue + matrix[0][3],
        matrix[1][0] * red + matrix[1][1] * green + matrix[1][2] * blue + matrix[1][3],
        matrix[2][0] * red + matrix[2][1] * green + matrix[2][2] * blue + matrix[2][3],
        pixel.alpha(),
    )
}

#[derive(Clone, Copy, Debug)]
struct StraightRgb {
    red: f64,
    green: f64,
    blue: f64,
}

impl StraightRgb {
    const fn new(red: f64, green: f64, blue: f64) -> Self {
        Self { red, green, blue }
    }

    fn from_premultiplied(pixel: PremultipliedRgba8) -> Option<Self> {
        if pixel.alpha == 0 {
            return None;
        }
        let alpha = f64::from(pixel.alpha);
        Some(Self {
            red: f64::from(pixel.red) / alpha,
            green: f64::from(pixel.green) / alpha,
            blue: f64::from(pixel.blue) / alpha,
        })
    }

    fn map(self, map_channel: impl Fn(f64) -> f64) -> Self {
        Self {
            red: map_channel(self.red),
            green: map_channel(self.green),
            blue: map_channel(self.blue),
        }
    }

    fn mix(self, other: Self, amount: f64) -> Self {
        Self {
            red: self.red * (1.0 - amount) + other.red * amount,
            green: self.green * (1.0 - amount) + other.green * amount,
            blue: self.blue * (1.0 - amount) + other.blue * amount,
        }
    }

    fn grayscale_luma(self) -> f64 {
        self.red * GRAYSCALE_LUMA_RED
            + self.green * GRAYSCALE_LUMA_GREEN
            + self.blue * GRAYSCALE_LUMA_BLUE
    }

    fn saturation_luma(self) -> f64 {
        self.red * SATURATION_LUMA_RED
            + self.green * SATURATION_LUMA_GREEN
            + self.blue * SATURATION_LUMA_BLUE
    }

    fn clamp_unit(self) -> Self {
        self.map(|channel| channel.clamp(0.0, 1.0))
    }
}

/// Applies the independent S21 oracle without premultiplied-RGBA8 source
/// quantization. This is the high-working-format comparison owner; reduced
/// comparisons intentionally continue through [`PremultipliedRgba8`].
pub(crate) fn apply_color_filter_pipeline_to_straight_rgba8(
    pixels: &[[u8; 4]],
    pipeline: &ColorFilterPipeline,
) -> Vec<u8> {
    pixels
        .iter()
        .copied()
        .flat_map(|pixel| {
            let mut color = ReferenceStraightRgba::from_rgba8(pixel);
            for operation in pipeline.ops() {
                color = color.apply(*operation);
            }
            color.into_rgba8()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct ReferenceStraightRgba {
    rgb: StraightRgb,
    alpha: f64,
}

impl ReferenceStraightRgba {
    fn from_rgba8(pixel: [u8; 4]) -> Self {
        if pixel[3] == 0 {
            return Self {
                rgb: StraightRgb::new(0.0, 0.0, 0.0),
                alpha: 0.0,
            };
        }
        Self {
            rgb: StraightRgb::new(
                f64::from(pixel[0]) / 255.0,
                f64::from(pixel[1]) / 255.0,
                f64::from(pixel[2]) / 255.0,
            ),
            alpha: f64::from(pixel[3]) / 255.0,
        }
    }

    fn apply(mut self, operation: ColorFilterOp) -> Self {
        match operation {
            ColorFilterOp::Brightness(amount) => {
                self.rgb = self.rgb.map(|channel| channel * amount.value());
            }
            ColorFilterOp::Contrast(amount) => {
                self.rgb = self
                    .rgb
                    .map(|channel| (channel - 0.5) * amount.value() + 0.5);
            }
            ColorFilterOp::Grayscale(amount) => {
                let gray = self.rgb.grayscale_luma();
                self.rgb = self
                    .rgb
                    .mix(StraightRgb::new(gray, gray, gray), amount.value());
            }
            ColorFilterOp::HueRotate(angle) => {
                let (sin, cos) = angle.radians().sin_cos();
                self.rgb = StraightRgb::new(
                    (0.213 + cos * 0.787 - sin * 0.213) * self.rgb.red
                        + (0.715 - cos * 0.715 - sin * 0.715) * self.rgb.green
                        + (0.072 - cos * 0.072 + sin * 0.928) * self.rgb.blue,
                    (0.213 - cos * 0.213 + sin * 0.143) * self.rgb.red
                        + (0.715 + cos * 0.285 + sin * 0.140) * self.rgb.green
                        + (0.072 - cos * 0.072 - sin * 0.283) * self.rgb.blue,
                    (0.213 - cos * 0.213 - sin * 0.787) * self.rgb.red
                        + (0.715 - cos * 0.715 + sin * 0.715) * self.rgb.green
                        + (0.072 + cos * 0.928 + sin * 0.072) * self.rgb.blue,
                );
            }
            ColorFilterOp::Invert(amount) => {
                self.rgb = self.rgb.map(|channel| {
                    channel * (1.0 - amount.value()) + (1.0 - channel) * amount.value()
                });
            }
            ColorFilterOp::Opacity(amount) => {
                self.alpha *= amount.value();
            }
            ColorFilterOp::Saturate(amount) => {
                let gray = self.rgb.saturation_luma();
                self.rgb = StraightRgb::new(gray, gray, gray).mix(self.rgb, amount.value());
            }
            ColorFilterOp::Sepia(amount) => {
                let sepia = StraightRgb::new(
                    self.rgb.red * 0.393 + self.rgb.green * 0.769 + self.rgb.blue * 0.189,
                    self.rgb.red * 0.349 + self.rgb.green * 0.686 + self.rgb.blue * 0.168,
                    self.rgb.red * 0.272 + self.rgb.green * 0.534 + self.rgb.blue * 0.131,
                );
                self.rgb = self.rgb.mix(sepia, amount.value());
            }
        }
        self.rgb = self.rgb.clamp_unit();
        self.alpha = self.alpha.clamp(0.0, 1.0);
        self
    }

    fn into_rgba8(self) -> [u8; 4] {
        if self.alpha == 0.0 {
            return [0, 0, 0, 0];
        }
        [
            unit_to_rgba8(self.rgb.red),
            unit_to_rgba8(self.rgb.green),
            unit_to_rgba8(self.rgb.blue),
            unit_to_rgba8(self.alpha),
        ]
    }
}

fn unit_to_rgba8(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
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

    pub(crate) fn apply_color_filter_pipeline(
        &self,
        pipeline: &ColorFilterPipeline,
    ) -> Result<Self> {
        let pixels = self
            .pixels
            .iter()
            .copied()
            .map(|pixel| pixel.apply_color_filter_pipeline(pipeline))
            .collect::<Result<Vec<_>>>()?;
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn apply_compiled_color_filter_pipeline(
        &self,
        pipeline: &CompiledColorFilterPipeline,
    ) -> Result<Self> {
        self.map_pixels(|pixel| apply_compiled_color_filter_pipeline_to_pixel(pixel, pipeline))
    }

    pub(crate) fn apply_alpha_mask(&self, mask: &Self) -> Result<Self> {
        if self.physical_size != mask.physical_size {
            return Err(Error::invalid_value(
                "reference alpha mask size",
                format!(
                    "{}x{}",
                    mask.physical_size.width(),
                    mask.physical_size.height()
                ),
                "must match source size",
            ));
        }
        let pixels = self
            .pixels
            .iter()
            .copied()
            .zip(mask.pixels.iter().copied())
            .map(|(source, mask)| source.destination_in_alpha_of(mask))
            .collect();
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn apply_resolved_alpha_mask(
        &self,
        source_bounds: Rect,
        mask: &Image,
        mask_bounds: Rect,
    ) -> Result<Self> {
        let width = self.physical_size.width();
        let height = self.physical_size.height();
        let mut pixels = Vec::with_capacity(self.pixels.len());
        for y in 0..height {
            let local_y = source_bounds.y()
                + (f64::from(y) + 0.5) * source_bounds.height() / f64::from(height);
            for x in 0..width {
                let local_x = source_bounds.x()
                    + (f64::from(x) + 0.5) * source_bounds.width() / f64::from(width);
                let mask_alpha = sample_resolved_mask_alpha(mask, mask_bounds, local_x, local_y);
                pixels.push(self.pixel(x, y)?.apply_opacity_amount(mask_alpha));
            }
        }
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn apply_blur(&self, blur: FilterBlur, policy: BlurPolicy) -> Result<Self> {
        let Some(kernel) = BlurKernel::from_policy(blur, policy)? else {
            return Ok(self.clone());
        };

        match policy.edge_sampling() {
            TransparentEdgeSamplingPolicy::TransparentBlack => {
                let width = usize::try_from(self.physical_size.width())
                    .expect("validated reference buffer width should fit addressable memory");
                let height = usize::try_from(self.physical_size.height())
                    .expect("validated reference buffer height should fit addressable memory");
                let mut horizontal = vec![FloatingPremultipliedRgba8::default(); self.pixels.len()];

                for y in 0..height {
                    for x in 0..width {
                        let mut pixel = FloatingPremultipliedRgba8::default();
                        for (offset, weight) in kernel.offset_weights() {
                            let Some(sample_x) = offset_index(x, offset, width) else {
                                continue;
                            };
                            pixel.add_pixel(self.pixels[y * width + sample_x], weight);
                        }
                        horizontal[y * width + x] = pixel;
                    }
                }

                let mut pixels = Vec::with_capacity(self.pixels.len());
                for y in 0..height {
                    for x in 0..width {
                        let mut pixel = FloatingPremultipliedRgba8::default();
                        for (offset, weight) in kernel.offset_weights() {
                            let Some(sample_y) = offset_index(y, offset, height) else {
                                continue;
                            };
                            pixel.add_float(horizontal[sample_y * width + x], weight);
                        }
                        pixels.push(pixel.to_pixel());
                    }
                }

                Self::from_pixels(self.physical_size, pixels)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn apply_mirrored_blur_for_gpu_oracle(
        &self,
        blur: FilterBlur,
        policy: BlurPolicy,
    ) -> Result<Self> {
        let Some(kernel) = BlurKernel::from_policy(blur, policy)? else {
            return Ok(self.clone());
        };
        let width = usize::try_from(self.physical_size.width())
            .expect("validated reference width must fit addressable memory");
        let height = usize::try_from(self.physical_size.height())
            .expect("validated reference height must fit addressable memory");
        let mut horizontal = vec![FloatingPremultipliedRgba8::default(); self.pixels.len()];
        for y in 0..height {
            for x in 0..width {
                for (offset, weight) in kernel.offset_weights() {
                    let sample_x = mirrored_offset_index(x, offset, width);
                    horizontal[y * width + x].add_pixel(self.pixels[y * width + sample_x], weight);
                }
            }
        }
        let mut pixels = Vec::with_capacity(self.pixels.len());
        for y in 0..height {
            for x in 0..width {
                let mut pixel = FloatingPremultipliedRgba8::default();
                for (offset, weight) in kernel.offset_weights() {
                    let sample_y = mirrored_offset_index(y, offset, height);
                    pixel.add_float(horizontal[sample_y * width + x], weight);
                }
                pixels.push(pixel.to_pixel());
            }
        }
        Self::from_pixels(self.physical_size, pixels)
    }

    #[cfg(test)]
    pub(crate) fn apply_blur_to_high_precision_straight_rgba8_for_gpu_oracle(
        &self,
        blur: FilterBlur,
        policy: BlurPolicy,
    ) -> Result<Vec<u8>> {
        let pixels = floating_gaussian_blur(&self.pixels, self.physical_size, blur, policy)?;
        Ok(floating_straight_rgba8(&pixels))
    }

    #[cfg(test)]
    pub(crate) fn apply_fractional_drop_shadow_to_high_precision_straight_rgba8_for_gpu_oracle(
        &self,
        shadow: &FilterDropShadow,
        policy: BlurPolicy,
    ) -> Result<Vec<u8>> {
        let source = self
            .pixels
            .iter()
            .copied()
            .map(FloatingPremultipliedRgba8::from_pixel)
            .collect::<Vec<_>>();
        let alpha_pixels = self
            .pixels
            .iter()
            .copied()
            .map(|pixel| PremultipliedRgba8::try_new(0, 0, 0, pixel.alpha()).unwrap())
            .collect::<Vec<_>>();
        let blurred =
            floating_gaussian_blur(&alpha_pixels, self.physical_size, shadow.blur(), policy)?;
        let shadow_pixels = floating_fractional_shadow(
            &blurred,
            self.physical_size,
            shadow.offset(),
            shadow.color(),
        );
        Ok(floating_straight_rgba8(&floating_source_over(
            &source,
            &shadow_pixels,
        )))
    }

    pub(crate) fn apply_drop_shadow(
        &self,
        shadow: &FilterDropShadow,
        policy: BlurPolicy,
    ) -> Result<Self> {
        let shifted_alpha = self.offset_alpha_mask(shadow)?;
        let blurred_alpha = shifted_alpha.apply_blur(shadow.blur(), policy)?;
        let shadow = blurred_alpha.colorize_alpha_mask(shadow.color())?;
        self.source_over(&shadow)
    }

    #[cfg(test)]
    pub(crate) fn apply_fractional_drop_shadow_for_gpu_oracle(
        &self,
        shadow: &FilterDropShadow,
        policy: BlurPolicy,
    ) -> Result<Self> {
        let source_alpha =
            self.map_pixels(|pixel| PremultipliedRgba8::try_new(0, 0, 0, pixel.alpha()))?;
        let blurred_alpha = source_alpha.apply_blur(shadow.blur(), policy)?;
        let shifted_alpha = blurred_alpha.sample_alpha_at_fractional_offset(shadow.offset())?;
        let colored_shadow = shifted_alpha.colorize_alpha_mask(shadow.color())?;
        self.source_over(&colored_shadow)
    }

    #[cfg(test)]
    fn sample_alpha_at_fractional_offset(&self, offset: Point) -> Result<Self> {
        let mut sampled = Self::try_new(self.physical_size)?;
        for y in 0..self.physical_size.height() {
            for x in 0..self.physical_size.width() {
                let alpha =
                    self.bilinear_alpha(f64::from(x) - offset.x(), f64::from(y) - offset.y());
                sampled.set_pixel(
                    x,
                    y,
                    PremultipliedRgba8::try_new(0, 0, 0, round_byte(alpha))?,
                )?;
            }
        }
        Ok(sampled)
    }

    #[cfg(test)]
    fn bilinear_alpha(&self, x: f64, y: f64) -> f64 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fraction_x = x - x0;
        let fraction_y = y - y0;
        [
            (x0, y0, (1.0 - fraction_x) * (1.0 - fraction_y)),
            (x0 + 1.0, y0, fraction_x * (1.0 - fraction_y)),
            (x0, y0 + 1.0, (1.0 - fraction_x) * fraction_y),
            (x0 + 1.0, y0 + 1.0, fraction_x * fraction_y),
        ]
        .into_iter()
        .map(|(sample_x, sample_y, weight)| {
            self.alpha_if_in_bounds(sample_x as i64, sample_y as i64) * weight
        })
        .sum()
    }

    #[cfg(test)]
    fn alpha_if_in_bounds(&self, x: i64, y: i64) -> f64 {
        if x < 0
            || y < 0
            || x >= i64::from(self.physical_size.width())
            || y >= i64::from(self.physical_size.height())
        {
            return 0.0;
        }
        let index = usize::try_from(y)
            .expect("nonnegative reference y must fit usize")
            .saturating_mul(
                usize::try_from(self.physical_size.width())
                    .expect("validated reference width must fit usize"),
            )
            .saturating_add(usize::try_from(x).expect("nonnegative reference x must fit usize"));
        f64::from(self.pixels[index].alpha())
    }

    fn offset_alpha_mask(&self, shadow: &FilterDropShadow) -> Result<Self> {
        let offset = shadow.offset();
        let offset_policy =
            MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();
        let offset_x = offset_policy.quantize(offset.x(), "filter drop-shadow offset x")?;
        let offset_y = offset_policy.quantize(offset.y(), "filter drop-shadow offset y")?;
        let width = self.physical_size.width();
        let height = self.physical_size.height();
        let mut buffer = Self::try_new(self.physical_size)?;

        for y in 0..height {
            for x in 0..width {
                let sample_x = i64::from(x) - i64::from(offset_x);
                let sample_y = i64::from(y) - i64::from(offset_y);
                if sample_x < 0
                    || sample_y < 0
                    || sample_x >= i64::from(width)
                    || sample_y >= i64::from(height)
                {
                    continue;
                }
                let alpha = self
                    .pixel(
                        u32::try_from(sample_x).expect("validated x should fit u32"),
                        u32::try_from(sample_y).expect("validated y should fit u32"),
                    )?
                    .alpha();
                if alpha != 0 {
                    buffer.set_pixel(
                        x,
                        y,
                        PremultipliedRgba8::try_new(0, 0, 0, alpha)
                            .expect("alpha-only mask pixels are valid premultiplied colors"),
                    )?;
                }
            }
        }

        Ok(buffer)
    }

    fn colorize_alpha_mask(&self, color: super::Color) -> Result<Self> {
        self.map_pixels(|pixel| {
            let alpha = round_byte(f64::from(pixel.alpha()) * f64::from(color.a()));
            Ok(PremultipliedRgba8::from_straight_color_channels(
                f64::from(color.r()),
                f64::from(color.g()),
                f64::from(color.b()),
                alpha,
            ))
        })
    }

    pub(crate) fn map_pixels(
        &self,
        mut map_pixel: impl FnMut(PremultipliedRgba8) -> Result<PremultipliedRgba8>,
    ) -> Result<Self> {
        let pixels = self
            .pixels
            .iter()
            .copied()
            .map(&mut map_pixel)
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

    pub(crate) fn blend_over(&self, destination: &Self, mode: BlendMode) -> Result<Self> {
        if self.physical_size != destination.physical_size {
            return Err(Error::invalid_value(
                "reference blend destination size",
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
            .map(|(source, destination)| source.blend_over(destination, mode))
            .collect();
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn source_in_alpha_of(&self, destination: &Self) -> Result<Self> {
        if self.physical_size != destination.physical_size {
            return Err(Error::invalid_value(
                "reference source-in destination size",
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
            .map(|(source, destination)| source.source_in_alpha_of(destination))
            .collect();
        Self::from_pixels(self.physical_size, pixels)
    }

    pub(crate) fn destination_in_alpha_of(&self, source: &Self) -> Result<Self> {
        if self.physical_size != source.physical_size {
            return Err(Error::invalid_value(
                "reference destination-in source size",
                format!(
                    "{}x{}",
                    source.physical_size.width(),
                    source.physical_size.height()
                ),
                "must match destination size",
            ));
        }
        let pixels = self
            .pixels
            .iter()
            .copied()
            .zip(source.pixels.iter().copied())
            .map(|(destination, source)| destination.destination_in_alpha_of(source))
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

/// Offset quantization policy for materialized CSS drop-shadow execution.
///
/// The materialized CPU/reference path deterministically snaps authored
/// drop-shadow offsets to the nearest device pixel before shifting the alpha
/// mask. Half-device-pixel values follow Rust `f64::round` semantics, which
/// round away from zero. This is the staged materialized CPU path policy until a
/// future subpixel sampling model exists.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaterializedDropShadowOffsetQuantizationPolicy;

impl MaterializedDropShadowOffsetQuantizationPolicy {
    pub(crate) const fn materialized_cpu_reference() -> Self {
        Self
    }

    pub(crate) fn quantize(self, value: f64, name: &'static str) -> Result<i32> {
        let rounded = value.round();
        if !rounded.is_finite() || rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
            return Err(Error::invalid_value(
                name,
                value,
                "must quantize to an i32 nearest-device-pixel offset",
            ));
        }
        Ok(rounded as i32)
    }
}

#[derive(Clone, Debug)]
struct BlurKernel {
    radius: i32,
    weights: Vec<f64>,
}

impl BlurKernel {
    fn from_policy(blur: FilterBlur, policy: BlurPolicy) -> Result<Option<Self>> {
        if blur.radius() == 0.0 {
            return Ok(None);
        }

        let standard_deviation = policy.standard_deviation(blur)?;
        if standard_deviation == 0.0 {
            return Ok(None);
        }

        let support_radius = policy.support_radius(blur)?.ceil();
        if support_radius > f64::from(i32::MAX) {
            return Err(Error::invalid_value(
                "blur kernel support radius",
                support_radius,
                "must fit in i32 device pixels",
            ));
        }
        let radius = support_radius as i32;
        let mut weights = Vec::with_capacity(
            usize::try_from(radius)
                .expect("validated blur kernel support radius should fit usize")
                .saturating_mul(2)
                .saturating_add(1),
        );
        let divisor = 2.0 * standard_deviation * standard_deviation;
        let mut weight_sum = 0.0;

        for offset in -radius..=radius {
            let distance = f64::from(offset);
            let weight = (-(distance * distance) / divisor).exp();
            weights.push(weight);
            weight_sum += weight;
        }
        for weight in &mut weights {
            *weight /= weight_sum;
        }

        Ok(Some(Self { radius, weights }))
    }

    fn offset_weights(&self) -> impl Iterator<Item = (i32, f64)> + '_ {
        self.weights
            .iter()
            .copied()
            .enumerate()
            .map(|(index, weight)| {
                (
                    i32::try_from(index).expect("validated blur kernel index should fit i32")
                        - self.radius,
                    weight,
                )
            })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FloatingPremultipliedRgba8 {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

impl FloatingPremultipliedRgba8 {
    fn from_pixel(pixel: PremultipliedRgba8) -> Self {
        Self {
            red: f64::from(pixel.red()),
            green: f64::from(pixel.green()),
            blue: f64::from(pixel.blue()),
            alpha: f64::from(pixel.alpha()),
        }
    }

    fn add_pixel(&mut self, pixel: PremultipliedRgba8, weight: f64) {
        self.red += f64::from(pixel.red()) * weight;
        self.green += f64::from(pixel.green()) * weight;
        self.blue += f64::from(pixel.blue()) * weight;
        self.alpha += f64::from(pixel.alpha()) * weight;
    }

    fn add_float(&mut self, pixel: Self, weight: f64) {
        self.red += pixel.red * weight;
        self.green += pixel.green * weight;
        self.blue += pixel.blue * weight;
        self.alpha += pixel.alpha * weight;
    }

    fn to_pixel(self) -> PremultipliedRgba8 {
        let alpha = round_byte(self.alpha);
        PremultipliedRgba8::try_new(
            round_byte(self.red).min(alpha),
            round_byte(self.green).min(alpha),
            round_byte(self.blue).min(alpha),
            alpha,
        )
        .expect("weighted premultiplied blur output should stay premultiplied")
    }
}

#[cfg(test)]
fn floating_gaussian_blur(
    pixels: &[PremultipliedRgba8],
    size: PhysicalSize,
    blur: FilterBlur,
    policy: BlurPolicy,
) -> Result<Vec<FloatingPremultipliedRgba8>> {
    let Some(kernel) = BlurKernel::from_policy(blur, policy)? else {
        return Ok(pixels
            .iter()
            .copied()
            .map(FloatingPremultipliedRgba8::from_pixel)
            .collect());
    };
    let width = usize::try_from(size.width()).expect("validated width must fit usize");
    let height = usize::try_from(size.height()).expect("validated height must fit usize");
    let mut horizontal = vec![FloatingPremultipliedRgba8::default(); pixels.len()];
    for y in 0..height {
        for x in 0..width {
            for (offset, weight) in kernel.offset_weights() {
                if let Some(sample_x) = offset_index(x, offset, width) {
                    horizontal[y * width + x].add_pixel(pixels[y * width + sample_x], weight);
                }
            }
        }
    }
    Ok(floating_vertical_blur(&horizontal, width, height, &kernel))
}

#[cfg(test)]
fn floating_vertical_blur(
    horizontal: &[FloatingPremultipliedRgba8],
    width: usize,
    height: usize,
    kernel: &BlurKernel,
) -> Vec<FloatingPremultipliedRgba8> {
    let mut output = vec![FloatingPremultipliedRgba8::default(); horizontal.len()];
    for y in 0..height {
        for x in 0..width {
            for (offset, weight) in kernel.offset_weights() {
                if let Some(sample_y) = offset_index(y, offset, height) {
                    output[y * width + x].add_float(horizontal[sample_y * width + x], weight);
                }
            }
        }
    }
    output
}

#[cfg(test)]
fn floating_fractional_shadow(
    blurred_alpha: &[FloatingPremultipliedRgba8],
    size: PhysicalSize,
    offset: Point,
    color: super::Color,
) -> Vec<FloatingPremultipliedRgba8> {
    let width = usize::try_from(size.width()).expect("validated width must fit usize");
    let height = usize::try_from(size.height()).expect("validated height must fit usize");
    let mut output = Vec::with_capacity(blurred_alpha.len());
    for y in 0..height {
        for x in 0..width {
            let alpha = floating_bilinear_alpha(
                blurred_alpha,
                width,
                height,
                x as f64 - offset.x(),
                y as f64 - offset.y(),
            ) * f64::from(color.a());
            output.push(FloatingPremultipliedRgba8 {
                red: f64::from(color.r()) * alpha,
                green: f64::from(color.g()) * alpha,
                blue: f64::from(color.b()) * alpha,
                alpha,
            });
        }
    }
    output
}

#[cfg(test)]
fn floating_bilinear_alpha(
    pixels: &[FloatingPremultipliedRgba8],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
) -> f64 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fraction_x = x - x0;
    let fraction_y = y - y0;
    [
        (x0, y0, (1.0 - fraction_x) * (1.0 - fraction_y)),
        (x0 + 1.0, y0, fraction_x * (1.0 - fraction_y)),
        (x0, y0 + 1.0, (1.0 - fraction_x) * fraction_y),
        (x0 + 1.0, y0 + 1.0, fraction_x * fraction_y),
    ]
    .into_iter()
    .filter(|(x, y, _)| *x >= 0.0 && *y >= 0.0 && *x < width as f64 && *y < height as f64)
    .map(|(x, y, weight)| pixels[y as usize * width + x as usize].alpha * weight)
    .sum()
}

#[cfg(test)]
fn floating_source_over(
    source: &[FloatingPremultipliedRgba8],
    destination: &[FloatingPremultipliedRgba8],
) -> Vec<FloatingPremultipliedRgba8> {
    source
        .iter()
        .copied()
        .zip(destination.iter().copied())
        .map(|(source, destination)| {
            let retained = 1.0 - source.alpha / 255.0;
            FloatingPremultipliedRgba8 {
                red: source.red + destination.red * retained,
                green: source.green + destination.green * retained,
                blue: source.blue + destination.blue * retained,
                alpha: source.alpha + destination.alpha * retained,
            }
        })
        .collect()
}

#[cfg(test)]
fn floating_straight_rgba8(pixels: &[FloatingPremultipliedRgba8]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|pixel| {
            let alpha = round_byte(pixel.alpha);
            if alpha == 0 || pixel.alpha <= 0.0 {
                return [0, 0, 0, 0];
            }
            [
                round_byte(pixel.red * 255.0 / pixel.alpha),
                round_byte(pixel.green * 255.0 / pixel.alpha),
                round_byte(pixel.blue * 255.0 / pixel.alpha),
                alpha,
            ]
        })
        .collect()
}

fn offset_index(index: usize, offset: i32, len: usize) -> Option<usize> {
    let sample = i64::try_from(index).ok()? + i64::from(offset);
    if sample < 0 || sample >= i64::try_from(len).ok()? {
        return None;
    }
    usize::try_from(sample).ok()
}

#[cfg(test)]
fn mirrored_offset_index(index: usize, offset: i32, len: usize) -> usize {
    let len = i64::try_from(len).expect("validated mirrored extent must fit i64");
    let period = len
        .checked_mul(2)
        .expect("validated mirrored extent period must fit i64");
    let sample = (i64::try_from(index).expect("validated mirrored index must fit i64")
        + i64::from(offset))
    .rem_euclid(period);
    let mirrored = if sample < len {
        sample
    } else {
        period - sample - 1
    };
    usize::try_from(mirrored).expect("mirrored index must fit usize")
}

fn sample_resolved_mask_alpha(mask: &Image, bounds: Rect, x: f64, y: f64) -> f64 {
    let max = bounds.max();
    if x < bounds.x() || x > max.x() || y < bounds.y() || y > max.y() {
        return 0.0;
    }

    let width = mask.data.width;
    let height = mask.data.height;
    if width == 0 || height == 0 {
        return 0.0;
    }

    let sample_x = ((x - bounds.x()) / bounds.width()).mul_add(f64::from(width), -0.5);
    let sample_y = ((y - bounds.y()) / bounds.height()).mul_add(f64::from(height), -0.5);
    match mask.quality {
        ImageQuality::Low => mask_alpha_tap(
            mask,
            (sample_x + 0.5).floor() as i64,
            (sample_y + 0.5).floor() as i64,
        ),
        ImageQuality::Medium => {
            let left = sample_x.floor() as i64;
            let top = sample_y.floor() as i64;
            let horizontal = sample_x - left as f64;
            let vertical = sample_y - top as f64;
            let top_alpha = mask_alpha_tap(mask, left, top).mul_add(
                1.0 - horizontal,
                mask_alpha_tap(mask, left + 1, top) * horizontal,
            );
            let bottom_alpha = mask_alpha_tap(mask, left, top + 1).mul_add(
                1.0 - horizontal,
                mask_alpha_tap(mask, left + 1, top + 1) * horizontal,
            );
            top_alpha.mul_add(1.0 - vertical, bottom_alpha * vertical)
        }
        ImageQuality::High => {
            let base_x = sample_x.floor() as i64;
            let base_y = sample_y.floor() as i64;
            let mut alpha = 0.0;
            for offset_y in -1_i64..=2 {
                let tap_y = base_y + offset_y;
                let weight_y = mitchell_netravali(sample_y - tap_y as f64);
                for offset_x in -1_i64..=2 {
                    let tap_x = base_x + offset_x;
                    let weight_x = mitchell_netravali(sample_x - tap_x as f64);
                    alpha += mask_alpha_tap(mask, tap_x, tap_y) * weight_x * weight_y;
                }
            }
            alpha.clamp(0.0, 1.0)
        }
    }
}

fn mask_alpha_tap(mask: &Image, x: i64, y: i64) -> f64 {
    let Some(x) = extend_mask_index(x, mask.data.width, mask.extend) else {
        return 0.0;
    };
    let Some(y) = extend_mask_index(y, mask.data.height, mask.extend) else {
        return 0.0;
    };
    let alpha_index = u64::from(y)
        .checked_mul(u64::from(mask.data.width))
        .and_then(|row| row.checked_add(u64::from(x)))
        .and_then(|pixel| pixel.checked_mul(4))
        .and_then(|byte| byte.checked_add(3))
        .and_then(|byte| usize::try_from(byte).ok());
    alpha_index
        .and_then(|index| mask.bytes.get(index))
        .map_or(0.0, |alpha| f64::from(*alpha) / 255.0)
}

fn extend_mask_index(index: i64, length: u32, extend: Extend) -> Option<u32> {
    if length == 0 {
        return None;
    }
    let length = i64::from(length);
    let extended = match extend {
        Extend::Pad => index.clamp(0, length - 1),
        Extend::Repeat => index.rem_euclid(length),
        Extend::Reflect => {
            let period = length * 2;
            let reflected = index.rem_euclid(period);
            if reflected < length {
                reflected
            } else {
                period - reflected - 1
            }
        }
    };
    u32::try_from(extended).ok()
}

fn mitchell_netravali(distance: f64) -> f64 {
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;
    let distance = distance.abs();
    if distance < 1.0 {
        ((12.0 - 9.0 * B - 6.0 * C) * distance.powi(3)
            + (-18.0 + 12.0 * B + 6.0 * C) * distance.powi(2)
            + (6.0 - 2.0 * B))
            / 6.0
    } else if distance < 2.0 {
        ((-B - 6.0 * C) * distance.powi(3)
            + (6.0 * B + 30.0 * C) * distance.powi(2)
            + (-12.0 * B - 48.0 * C) * distance
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
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

fn scale_channel_by_opacity(channel: u8, opacity: f64) -> u8 {
    round_byte(f64::from(channel) * opacity)
}

fn premultiply_straight_channel(channel: f64, alpha: u8) -> u8 {
    round_byte(channel * f64::from(alpha))
}

#[derive(Clone, Copy)]
struct MixBlendContext {
    source_alpha: f64,
    destination_alpha: f64,
    mode: BlendMode,
    output_alpha: u8,
}

fn mix_blend_channel(
    source_channel: u8,
    source_alpha_byte: u8,
    destination_channel: u8,
    destination_alpha_byte: u8,
    context: MixBlendContext,
) -> u8 {
    let source = f64::from(source_channel) / f64::from(source_alpha_byte);
    let destination = f64::from(destination_channel) / f64::from(destination_alpha_byte);
    let blended = blend_straight_channel(source, destination, context.mode);
    let premultiplied = (1.0 - context.source_alpha) * context.destination_alpha * destination
        + (1.0 - context.destination_alpha) * context.source_alpha * source
        + context.source_alpha * context.destination_alpha * blended;
    round_byte(premultiplied * 255.0).min(context.output_alpha)
}

fn blend_straight_channel(source: f64, destination: f64, mode: BlendMode) -> f64 {
    match mode {
        BlendMode::Multiply => source * destination,
        BlendMode::Screen => source + destination - source * destination,
        BlendMode::Overlay => {
            if destination <= 0.5 {
                2.0 * source * destination
            } else {
                1.0 - 2.0 * (1.0 - source) * (1.0 - destination)
            }
        }
        BlendMode::Darken => source.min(destination),
        BlendMode::Lighten => source.max(destination),
        BlendMode::Normal | BlendMode::Plus => {
            unreachable!("normal and plus are handled before mix blend math")
        }
    }
}

fn round_byte(value: f64) -> u8 {
    // Reference color filters round half away from zero after clamping, matching
    // Rust's stable `f64::round` behavior so byte oracles are deterministic.
    value.round().clamp(0.0, 255.0) as u8
}

const fn scale_channel_by_alpha(channel: u8, alpha: u8) -> u8 {
    let scaled = (channel as u16) * (alpha as u16) + 127;
    (scaled / 255) as u8
}

/// Test-only reference classifier for materialized image filters.
///
/// This is an execution plan shape, not execution itself. Color-only runs are
/// compiled into the existing color pipeline, while pixel-moving operations
/// remain named steps for later region planning and byte execution.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedImageFilterPipeline {
    steps: Vec<MaterializedImageFilterStep>,
}

#[cfg(test)]
impl MaterializedImageFilterPipeline {
    pub fn try_from_filter_list(filters: &FilterList) -> Result<Option<Self>> {
        let ops = filters.ops();
        if ops.is_empty() {
            return Ok(None);
        }

        let mut steps = Vec::new();
        let mut color_run = Vec::new();

        for op in ops {
            match op.kind() {
                FilterOpKind::Blur(blur) => {
                    flush_materialized_color_run(&mut steps, &mut color_run)?;
                    steps.push(MaterializedImageFilterStep::Blur(*blur));
                }
                FilterOpKind::DropShadow(shadow) => {
                    flush_materialized_color_run(&mut steps, &mut color_run)?;
                    steps.push(MaterializedImageFilterStep::DropShadow(*shadow));
                }
                FilterOpKind::Brightness(amount) => {
                    color_run.push(ColorFilterOp::Brightness(*amount));
                }
                FilterOpKind::Contrast(amount) => {
                    color_run.push(ColorFilterOp::Contrast(*amount));
                }
                FilterOpKind::Grayscale(amount) => {
                    color_run.push(ColorFilterOp::Grayscale(*amount));
                }
                FilterOpKind::HueRotate(angle) => {
                    color_run.push(ColorFilterOp::HueRotate(*angle));
                }
                FilterOpKind::Invert(amount) => {
                    color_run.push(ColorFilterOp::Invert(*amount));
                }
                FilterOpKind::Opacity(amount) => {
                    color_run.push(ColorFilterOp::Opacity(*amount));
                }
                FilterOpKind::Saturate(amount) => {
                    color_run.push(ColorFilterOp::Saturate(*amount));
                }
                FilterOpKind::Sepia(amount) => {
                    color_run.push(ColorFilterOp::Sepia(*amount));
                }
            }
        }

        flush_materialized_color_run(&mut steps, &mut color_run)?;
        Ok(Some(Self { steps }))
    }

    #[must_use]
    pub fn steps(&self) -> &[MaterializedImageFilterStep] {
        &self.steps
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub enum MaterializedImageFilterStep {
    ColorFilters(CompiledColorFilterPipeline),
    Blur(FilterBlur),
    /// Executes an intrinsically valid solid-color filter drop shadow.
    DropShadow(FilterDropShadow),
}

#[cfg(test)]
impl FilterList {
    pub fn materialized_image_filter_pipeline(
        &self,
    ) -> Result<Option<MaterializedImageFilterPipeline>> {
        MaterializedImageFilterPipeline::try_from_filter_list(self)
    }
}

#[cfg(test)]
fn flush_materialized_color_run(
    steps: &mut Vec<MaterializedImageFilterStep>,
    color_run: &mut Vec<ColorFilterOp>,
) -> Result<()> {
    if color_run.is_empty() {
        return Ok(());
    }

    let compiled = CompiledColorFilterPipeline::try_from_ops(std::mem::take(color_run))?;
    steps.push(MaterializedImageFilterStep::ColorFilters(compiled));
    Ok(())
}

/// Test-only reference executable for color-only filter pipelines.
///
/// This is a compiled render/reference phase model, not an authored CSS filter
/// list and not a layer filter graph. It keeps the source operation order for
/// diagnostics/proof while executing grouped color-matrix runs and explicit
/// opacity steps. Opacity is sequenced instead of folded into color runs because
/// it changes premultiplied alpha at its ordered position.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledColorFilterPipeline {
    source_ops: Vec<ColorFilterOp>,
    steps: Vec<CompiledColorFilterStep>,
}

#[cfg(test)]
impl CompiledColorFilterPipeline {
    pub fn try_from_pipeline(pipeline: &ColorFilterPipeline) -> Result<Self> {
        Self::try_from_ops(pipeline.ops().to_vec())
    }

    pub fn try_from_ops(source_ops: Vec<ColorFilterOp>) -> Result<Self> {
        if source_ops.is_empty() {
            return Err(Error::invalid_value(
                "compiled color filter pipeline",
                "[]",
                "must contain at least one color filter operation",
            ));
        }

        Ok(Self {
            steps: compile_steps(&source_ops),
            source_ops,
        })
    }

    #[must_use]
    pub fn source_ops(&self) -> &[ColorFilterOp] {
        &self.source_ops
    }

    pub(crate) fn executable_steps(&self) -> &[CompiledColorFilterStep] {
        &self.steps
    }

    #[cfg(test)]
    pub(crate) fn executable_step_count(&self) -> usize {
        self.steps.len()
    }
}

/// One ordered executable step in a compiled color-filter pipeline.
///
/// Adjacent straight-color filters are fused into `StraightColorRun` so the
/// executable pipeline no longer interprets authored filter variants. The run
/// still stores ordered transforms rather than one collapsed matrix because the
/// reference policy clamps and rounds after each source operation; collapsing
/// those transforms would change CSS-visible order/rounding for some chains.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CompiledColorFilterStep {
    Identity,
    TransparentBlack,
    StraightColorRun(Vec<StraightColorTransform>),
    Opacity(UnitFilterAmount),
}

#[cfg(test)]
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StraightColorTransform {
    matrix: [[f64; 4]; 3],
}

#[cfg(test)]
impl StraightColorTransform {
    pub(crate) const fn matrix(self) -> [[f64; 4]; 3] {
        self.matrix
    }

    fn from_op(op: ColorFilterOp) -> Option<Self> {
        match op {
            ColorFilterOp::Brightness(amount) => Some(Self::brightness(amount.value())),
            ColorFilterOp::Contrast(amount) => Some(Self::contrast(amount.value())),
            ColorFilterOp::Grayscale(amount) => Some(Self::grayscale(amount.value())),
            ColorFilterOp::HueRotate(angle) => Some(Self::hue_rotate(angle.radians())),
            ColorFilterOp::Invert(amount) => Some(Self::invert(amount.value())),
            ColorFilterOp::Opacity(_) => None,
            ColorFilterOp::Saturate(amount) => Some(Self::saturate(amount.value())),
            ColorFilterOp::Sepia(amount) => Some(Self::sepia(amount.value())),
        }
    }

    const fn brightness(amount: f64) -> Self {
        Self {
            matrix: [
                [amount, 0.0, 0.0, 0.0],
                [0.0, amount, 0.0, 0.0],
                [0.0, 0.0, amount, 0.0],
            ],
        }
    }

    const fn contrast(amount: f64) -> Self {
        let intercept = 0.5 - amount * 0.5;
        Self {
            matrix: [
                [amount, 0.0, 0.0, intercept],
                [0.0, amount, 0.0, intercept],
                [0.0, 0.0, amount, intercept],
            ],
        }
    }

    const fn grayscale(amount: f64) -> Self {
        let inverse = 1.0 - amount;
        Self {
            matrix: [
                [
                    inverse + amount * GRAYSCALE_LUMA_RED,
                    amount * GRAYSCALE_LUMA_GREEN,
                    amount * GRAYSCALE_LUMA_BLUE,
                    0.0,
                ],
                [
                    amount * GRAYSCALE_LUMA_RED,
                    inverse + amount * GRAYSCALE_LUMA_GREEN,
                    amount * GRAYSCALE_LUMA_BLUE,
                    0.0,
                ],
                [
                    amount * GRAYSCALE_LUMA_RED,
                    amount * GRAYSCALE_LUMA_GREEN,
                    inverse + amount * GRAYSCALE_LUMA_BLUE,
                    0.0,
                ],
            ],
        }
    }

    fn hue_rotate(radians: f64) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            matrix: [
                [
                    0.213 + cos * 0.787 - sin * 0.213,
                    0.715 - cos * 0.715 - sin * 0.715,
                    0.072 - cos * 0.072 + sin * 0.928,
                    0.0,
                ],
                [
                    0.213 - cos * 0.213 + sin * 0.143,
                    0.715 + cos * 0.285 + sin * 0.140,
                    0.072 - cos * 0.072 - sin * 0.283,
                    0.0,
                ],
                [
                    0.213 - cos * 0.213 - sin * 0.787,
                    0.715 - cos * 0.715 + sin * 0.715,
                    0.072 + cos * 0.928 + sin * 0.072,
                    0.0,
                ],
            ],
        }
    }

    const fn invert(amount: f64) -> Self {
        let scale = 1.0 - amount * 2.0;
        Self {
            matrix: [
                [scale, 0.0, 0.0, amount],
                [0.0, scale, 0.0, amount],
                [0.0, 0.0, scale, amount],
            ],
        }
    }

    const fn saturate(amount: f64) -> Self {
        let inverse = 1.0 - amount;
        Self {
            matrix: [
                [
                    amount + inverse * SATURATION_LUMA_RED,
                    inverse * SATURATION_LUMA_GREEN,
                    inverse * SATURATION_LUMA_BLUE,
                    0.0,
                ],
                [
                    inverse * SATURATION_LUMA_RED,
                    amount + inverse * SATURATION_LUMA_GREEN,
                    inverse * SATURATION_LUMA_BLUE,
                    0.0,
                ],
                [
                    inverse * SATURATION_LUMA_RED,
                    inverse * SATURATION_LUMA_GREEN,
                    amount + inverse * SATURATION_LUMA_BLUE,
                    0.0,
                ],
            ],
        }
    }

    const fn sepia(amount: f64) -> Self {
        let inverse = 1.0 - amount;
        Self {
            matrix: [
                [
                    inverse + amount * 0.393,
                    amount * 0.769,
                    amount * 0.189,
                    0.0,
                ],
                [
                    amount * 0.349,
                    inverse + amount * 0.686,
                    amount * 0.168,
                    0.0,
                ],
                [
                    amount * 0.272,
                    amount * 0.534,
                    inverse + amount * 0.131,
                    0.0,
                ],
            ],
        }
    }
}

#[cfg(test)]
fn compile_steps(source_ops: &[ColorFilterOp]) -> Vec<CompiledColorFilterStep> {
    if source_ops.iter().any(is_zero_opacity) {
        return vec![CompiledColorFilterStep::TransparentBlack];
    }

    let mut steps = Vec::new();
    let mut color_run = Vec::new();

    for op in source_ops.iter().copied() {
        if is_identity_op(op) {
            continue;
        }

        if let Some(transform) = StraightColorTransform::from_op(op) {
            color_run.push(transform);
            continue;
        }

        flush_color_run(&mut steps, &mut color_run);
        if let ColorFilterOp::Opacity(amount) = op {
            steps.push(CompiledColorFilterStep::Opacity(amount));
        }
    }

    flush_color_run(&mut steps, &mut color_run);
    if steps.is_empty() {
        steps.push(CompiledColorFilterStep::Identity);
    }
    steps
}

#[cfg(test)]
fn flush_color_run(
    steps: &mut Vec<CompiledColorFilterStep>,
    color_run: &mut Vec<StraightColorTransform>,
) {
    if !color_run.is_empty() {
        steps.push(CompiledColorFilterStep::StraightColorRun(std::mem::take(
            color_run,
        )));
    }
}

#[cfg(test)]
fn is_zero_opacity(op: &ColorFilterOp) -> bool {
    matches!(op, ColorFilterOp::Opacity(amount) if amount.value() == 0.0)
}

#[cfg(test)]
fn is_identity_op(op: ColorFilterOp) -> bool {
    match op {
        ColorFilterOp::Brightness(amount)
        | ColorFilterOp::Contrast(amount)
        | ColorFilterOp::Saturate(amount) => amount.value() == 1.0,
        ColorFilterOp::Grayscale(amount)
        | ColorFilterOp::Invert(amount)
        | ColorFilterOp::Sepia(amount) => amount.value() == 0.0,
        ColorFilterOp::HueRotate(angle) => angle.radians() == 0.0,
        ColorFilterOp::Opacity(amount) => amount.value() == 1.0,
    }
}

#[cfg(test)]
pub(crate) fn execute_transitional_resolved_mask_bridge_for_test(
    source: &ImageBuffer,
    source_bounds: Rect,
    image: Image,
    mask_bounds: Rect,
) -> Result<ImageBuffer> {
    let mask = ResolvedLayerAlphaMask::try_new(image, mask_bounds)?;
    validate_image_buffer_rgba_len(source.size(), source.rgba().len())?;
    super::validation::validate_point(source_bounds.origin(), "resolved-mask source bounds")?;
    super::validation::validate_positive_f64(
        source_bounds.width(),
        "resolved-mask source bounds width",
    )?;
    super::validation::validate_positive_f64(
        source_bounds.height(),
        "resolved-mask source bounds height",
    )?;
    let source = straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(source)?;
    let masked = source.apply_resolved_alpha_mask(source_bounds, mask.image(), mask.bounds())?;
    premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&masked)
}

/// Test-only reference boundary for resolved image/filter intent and materialized RGBA bytes.
///
/// `FilteredImagePaint` names the resolved resource and authored filter list, but the
/// bytes come from the paired `Image`. The execution phase converts straight RGBA8
/// image bytes to premultiplied RGBA8 reference pixels, applies the ordered
/// materialized-image filter pipeline, then emits straight RGBA8 oracle output.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ResolvedMaterializedImageFilterExecution<'a> {
    source: ResolvedMaterializedImageFilterSource<'a>,
    pipeline: MaterializedImageFilterPipeline,
}

#[cfg(test)]
pub(crate) type ResolvedImageColorFilterExecution<'a> =
    ResolvedMaterializedImageFilterExecution<'a>;

#[cfg(test)]
#[derive(Debug)]
enum ResolvedMaterializedImageFilterSource<'a> {
    Image(&'a Image),
    ImageBuffer(&'a ImageBuffer),
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(
    image_buffer: &ImageBuffer,
) -> Result<ReferencePremultipliedRgba8Buffer> {
    validate_image_buffer_rgba_len(image_buffer.size(), image_buffer.rgba().len())?;
    straight_rgba8_bytes_to_premultiplied_rgba8_reference(image_buffer.size(), image_buffer.rgba())
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

fn premultiply_straight_rgba8_channel(channel: u8, alpha: u8) -> u8 {
    round_byte(f64::from(channel) * f64::from(alpha) / 255.0)
}

fn unpremultiply_rgba8_channel(channel: u8, alpha: u8) -> u8 {
    round_byte(f64::from(channel) * 255.0 / f64::from(alpha))
}
