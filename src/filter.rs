use super::{
    Error, Result,
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    style::{ColorFilterOp, ColorFilterPipeline, UnitFilterAmount},
};

const LUMA_RED: f64 = 0.213;
const LUMA_GREEN: f64 = 0.715;
const LUMA_BLUE: f64 = 0.072;

/// Render-owned executable color-only filter pipeline.
///
/// This is a compiled render/reference phase model, not an authored CSS filter
/// list and not a layer filter graph. It keeps the source operation order for
/// diagnostics/proof while executing grouped color-matrix runs and explicit
/// opacity steps. Opacity is sequenced instead of folded into color runs because
/// it changes premultiplied alpha at its ordered position.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledColorFilterPipeline {
    source_ops: Vec<ColorFilterOp>,
    steps: Vec<CompiledColorFilterStep>,
}

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

    pub(crate) fn apply_to_pixel(&self, pixel: PremultipliedRgba8) -> Result<PremultipliedRgba8> {
        let mut pixel = pixel;
        for step in &self.steps {
            pixel = step.apply_to_pixel(pixel)?;
        }
        Ok(pixel)
    }

    pub(crate) fn apply_to_buffer(
        &self,
        buffer: &ReferencePremultipliedRgba8Buffer,
    ) -> Result<ReferencePremultipliedRgba8Buffer> {
        buffer.map_pixels(|pixel| self.apply_to_pixel(pixel))
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
#[derive(Clone, Debug, PartialEq)]
enum CompiledColorFilterStep {
    Identity,
    TransparentBlack,
    StraightColorRun(Vec<StraightColorTransform>),
    Opacity(UnitFilterAmount),
}

impl CompiledColorFilterStep {
    fn apply_to_pixel(&self, pixel: PremultipliedRgba8) -> Result<PremultipliedRgba8> {
        match self {
            Self::Identity => Ok(pixel),
            Self::TransparentBlack => Ok(PremultipliedRgba8::TRANSPARENT),
            Self::StraightColorRun(transforms) => {
                let mut pixel = pixel;
                for transform in transforms {
                    pixel = transform.apply_to_pixel(pixel);
                }
                Ok(pixel)
            }
            Self::Opacity(amount) => Ok(pixel.apply_opacity_amount(amount.value())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StraightColorTransform {
    matrix: [[f64; 4]; 3],
}

impl StraightColorTransform {
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
                    inverse + amount * LUMA_RED,
                    amount * LUMA_GREEN,
                    amount * LUMA_BLUE,
                    0.0,
                ],
                [
                    amount * LUMA_RED,
                    inverse + amount * LUMA_GREEN,
                    amount * LUMA_BLUE,
                    0.0,
                ],
                [
                    amount * LUMA_RED,
                    amount * LUMA_GREEN,
                    inverse + amount * LUMA_BLUE,
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
                    amount + inverse * LUMA_RED,
                    inverse * LUMA_GREEN,
                    inverse * LUMA_BLUE,
                    0.0,
                ],
                [
                    inverse * LUMA_RED,
                    amount + inverse * LUMA_GREEN,
                    inverse * LUMA_BLUE,
                    0.0,
                ],
                [
                    inverse * LUMA_RED,
                    inverse * LUMA_GREEN,
                    amount + inverse * LUMA_BLUE,
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

    fn apply_to_pixel(self, pixel: PremultipliedRgba8) -> PremultipliedRgba8 {
        if pixel.alpha() == 0 {
            return PremultipliedRgba8::TRANSPARENT;
        }

        let alpha = f64::from(pixel.alpha());
        let red = f64::from(pixel.red()) / alpha;
        let green = f64::from(pixel.green()) / alpha;
        let blue = f64::from(pixel.blue()) / alpha;

        PremultipliedRgba8::from_straight_color_channels(
            self.matrix[0][0] * red
                + self.matrix[0][1] * green
                + self.matrix[0][2] * blue
                + self.matrix[0][3],
            self.matrix[1][0] * red
                + self.matrix[1][1] * green
                + self.matrix[1][2] * blue
                + self.matrix[1][3],
            self.matrix[2][0] * red
                + self.matrix[2][1] * green
                + self.matrix[2][2] * blue
                + self.matrix[2][3],
            pixel.alpha(),
        )
    }
}

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

fn is_zero_opacity(op: &ColorFilterOp) -> bool {
    matches!(op, ColorFilterOp::Opacity(amount) if amount.value() == 0.0)
}

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
