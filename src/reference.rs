#![cfg_attr(not(test), allow(dead_code))]

use super::{
    Error, Extend, Image, ImageQuality, PhysicalSize, Rect, Result,
    filter::{BlurPolicy, CompiledColorFilterPipeline, TransparentEdgeSamplingPolicy},
    layer::BlendMode,
    style::{ColorFilterOp, ColorFilterPipeline, FilterBlur, FilterDropShadow},
};

const LUMA_RED: f64 = 0.213;
const LUMA_GREEN: f64 = 0.715;
const LUMA_BLUE: f64 = 0.072;

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
        pipeline.apply_to_pixel(self)
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
                let gray = rgb.luma();
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
                let gray = rgb.luma();
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

    fn luma(self) -> f64 {
        self.red * LUMA_RED + self.green * LUMA_GREEN + self.blue * LUMA_BLUE
    }

    fn clamp_unit(self) -> Self {
        self.map(|channel| channel.clamp(0.0, 1.0))
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
        pipeline.apply_to_buffer(self)
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

fn offset_index(index: usize, offset: i32, len: usize) -> Option<usize> {
    let sample = i64::try_from(index).ok()? + i64::from(offset);
    if sample < 0 || sample >= i64::try_from(len).ok()? {
        return None;
    }
    usize::try_from(sample).ok()
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
