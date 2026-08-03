use super::{
    CompiledColorFilterPipeline, PremultipliedRgba8, ReferencePremultipliedRgba8Buffer,
    premultiplied_rgba8_reference_to_straight_rgba8_image_buffer, round_byte,
    straight_rgba8_image_buffer_to_premultiplied_rgba8_reference,
    straight_rgba8_image_to_premultiplied_rgba8_reference,
};
use crate::{
    Color, Error, FilterList, FilteredImagePaint, Image, ImageBuffer, PhysicalSize, Point, Rect,
    Result,
    filter::{
        BlurPolicy, DevicePixelConversionPolicy, FilterClipBounds, FilterOutset, FilterRegionPlan,
        FilterSourceBounds, TransparentEdgeSamplingPolicy,
    },
    image::validate_image_buffer_rgba_len,
    style::{ColorFilterOp, FilterBlur, FilterDropShadow, FilterOpKind},
};
use std::sync::Arc;

impl ReferencePremultipliedRgba8Buffer {
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

    fn colorize_alpha_mask(&self, color: Color) -> Result<Self> {
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
    color: Color,
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
    blur: FilterBlur,
    policy: BlurPolicy,
) -> Result<PhysicalSize> {
    let source_rect = Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height()));
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
    shadow: &FilterDropShadow,
    policy: BlurPolicy,
) -> Result<PhysicalSize> {
    let source_rect = Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height()));
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
