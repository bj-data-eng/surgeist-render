use super::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer, round_byte};
use crate::{
    Error, Image, ImageBuffer, PhysicalSize, Result,
    image::{image_dimension, validate_image_buffer_rgba_len, validate_rgba_image},
    style::{ColorFilterOp, ColorFilterPipeline, UnitFilterAmount},
};

const GRAYSCALE_LUMA_RED: f64 = 0.2126;
const GRAYSCALE_LUMA_GREEN: f64 = 0.7152;
const GRAYSCALE_LUMA_BLUE: f64 = 0.0722;
const SATURATION_LUMA_RED: f64 = 0.213;
const SATURATION_LUMA_GREEN: f64 = 0.715;
const SATURATION_LUMA_BLUE: f64 = 0.072;

impl PremultipliedRgba8 {
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

impl ReferencePremultipliedRgba8Buffer {
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
}

fn premultiply_straight_channel(channel: f64, alpha: u8) -> u8 {
    round_byte(channel * f64::from(alpha))
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
