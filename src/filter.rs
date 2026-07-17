use super::{
    Error, Rect, Result,
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    style::{
        ColorFilterOp, ColorFilterPipeline, FilterBlur, FilterDropShadow, FilterList, FilterOpKind,
        UnitFilterAmount,
    },
};

const LUMA_RED: f64 = 0.213;
const LUMA_GREEN: f64 = 0.715;
const LUMA_BLUE: f64 = 0.072;
pub(crate) const CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS: f64 = 2.5;
const DEFAULT_MAX_BLUR_RADIUS: f64 = 256.0;

/// Source bounds for a pixel-moving filter operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterSourceBounds {
    rect: Rect,
}

impl FilterSourceBounds {
    pub fn try_new(rect: Rect) -> Result<Self> {
        validate_filter_bounds(rect, "filter source bounds")?;
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

/// Inflated bounds after applying blur, drop-shadow, or future pixel-moving outsets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterInflatedBounds {
    rect: Rect,
}

impl FilterInflatedBounds {
    fn try_new(rect: Rect) -> Result<Self> {
        validate_filter_bounds(rect, "filter inflated bounds")?;
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

/// Explicit filter-region clip bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterClipBounds {
    rect: Rect,
}

impl FilterClipBounds {
    pub fn try_new(rect: Rect) -> Result<Self> {
        validate_filter_bounds(rect, "filter clip bounds")?;
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

/// Non-empty execution region after inflated bounds are clipped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterExecutionRegion {
    rect: Rect,
}

impl FilterExecutionRegion {
    fn try_new(rect: Rect) -> Result<Self> {
        validate_filter_bounds(rect, "filter execution region")?;
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

/// Complete region plan for one pixel-moving filter step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterRegionPlan {
    source: FilterSourceBounds,
    inflated: FilterInflatedBounds,
    clip: Option<FilterClipBounds>,
    execution: FilterExecutionRegion,
}

impl FilterRegionPlan {
    pub fn try_new(
        source: FilterSourceBounds,
        outset: FilterOutset,
        clip: Option<FilterClipBounds>,
    ) -> Result<Self> {
        let inflated = FilterInflatedBounds::try_new(outset.inflate_rect(source.rect()))?;
        let execution_rect = match clip {
            Some(clip) => intersect_rects(inflated.rect(), clip.rect()).ok_or_else(|| {
                Error::invalid_value(
                    "filter execution region",
                    "empty",
                    "must have positive width and height after clipping",
                )
            })?,
            None => inflated.rect(),
        };
        let execution = FilterExecutionRegion::try_new(execution_rect)?;
        Ok(Self {
            source,
            inflated,
            clip,
            execution,
        })
    }

    #[must_use]
    pub const fn source_bounds(self) -> FilterSourceBounds {
        self.source
    }

    #[must_use]
    pub const fn inflated_bounds(self) -> FilterInflatedBounds {
        self.inflated
    }

    #[must_use]
    pub const fn clip_bounds(self) -> Option<FilterClipBounds> {
        self.clip
    }

    #[must_use]
    pub const fn execution_region(self) -> FilterExecutionRegion {
        self.execution
    }
}

/// Directional pixel-moving filter outset in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FilterOutset {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

impl FilterOutset {
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            left: 0.0,
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
        }
    }

    pub fn try_uniform(amount: f64) -> Result<Self> {
        Self::try_new(amount, amount, amount, amount)
    }

    pub fn try_new(left: f64, top: f64, right: f64, bottom: f64) -> Result<Self> {
        validate_filter_outset_value(left, "filter outset left")?;
        validate_filter_outset_value(top, "filter outset top")?;
        validate_filter_outset_value(right, "filter outset right")?;
        validate_filter_outset_value(bottom, "filter outset bottom")?;
        Ok(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    pub fn from_blur(blur: FilterBlur, policy: BlurPolicy) -> Result<Self> {
        Self::try_uniform(policy.support_radius(blur)?)
    }

    /// Computes the signed logical outsets for an executable filter drop shadow.
    pub fn from_drop_shadow(shadow: &FilterDropShadow, policy: BlurPolicy) -> Result<Self> {
        let support = policy.support_radius(shadow.blur())?;
        let offset = shadow.offset();
        Self::try_new(
            (support - offset.x()).max(0.0),
            (support - offset.y()).max(0.0),
            (support + offset.x()).max(0.0),
            (support + offset.y()).max(0.0),
        )
    }

    #[must_use]
    pub const fn left(self) -> f64 {
        self.left
    }

    #[must_use]
    pub const fn top(self) -> f64 {
        self.top
    }

    #[must_use]
    pub const fn right(self) -> f64 {
        self.right
    }

    #[must_use]
    pub const fn bottom(self) -> f64 {
        self.bottom
    }

    fn inflate_rect(self, rect: Rect) -> Rect {
        Rect::new(
            rect.x() - self.left,
            rect.y() - self.top,
            rect.width() + self.left + self.right,
            rect.height() + self.top + self.bottom,
        )
    }
}

/// How a blur radius value maps to Gaussian standard deviation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlurRadiusInterpretation {
    CssLengthAsStandardDeviation,
    VelloShadowBlurRadiusAsDiameter,
}

impl BlurRadiusInterpretation {
    const fn standard_deviation(self, radius: f64) -> f64 {
        match self {
            Self::CssLengthAsStandardDeviation => radius,
            Self::VelloShadowBlurRadiusAsDiameter => radius * 0.5,
        }
    }
}

/// Kernel support radius measured as a multiple of Gaussian standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelSupportRadius {
    standard_deviation_multiple: f64,
}

impl KernelSupportRadius {
    pub fn try_standard_deviation_multiple(standard_deviation_multiple: f64) -> Result<Self> {
        if !standard_deviation_multiple.is_finite() || standard_deviation_multiple <= 0.0 {
            return Err(Error::invalid_value(
                "blur kernel support radius",
                standard_deviation_multiple,
                "must be finite and greater than 0",
            ));
        }
        Ok(Self {
            standard_deviation_multiple,
        })
    }

    #[must_use]
    pub const fn standard_deviation_multiple(self) -> f64 {
        self.standard_deviation_multiple
    }

    fn support_radius(self, standard_deviation: f64) -> f64 {
        standard_deviation * self.standard_deviation_multiple
    }
}

/// Whether large blur radii are rejected or clamped before kernel planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LargeBlurRadiusAction {
    Reject,
    Clamp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LargeBlurRadiusPolicy {
    action: LargeBlurRadiusAction,
    max_radius: f64,
}

impl LargeBlurRadiusPolicy {
    pub fn try_reject_above(max_radius: f64) -> Result<Self> {
        Self::try_new(LargeBlurRadiusAction::Reject, max_radius)
    }

    pub fn try_clamp_to(max_radius: f64) -> Result<Self> {
        Self::try_new(LargeBlurRadiusAction::Clamp, max_radius)
    }

    fn try_new(action: LargeBlurRadiusAction, max_radius: f64) -> Result<Self> {
        if !max_radius.is_finite() || max_radius <= 0.0 {
            return Err(Error::invalid_value(
                "large blur radius limit",
                max_radius,
                "must be finite and greater than 0",
            ));
        }
        Ok(Self { action, max_radius })
    }

    #[must_use]
    pub const fn action(self) -> LargeBlurRadiusAction {
        self.action
    }

    #[must_use]
    pub const fn max_radius(self) -> f64 {
        self.max_radius
    }

    fn resolve_radius(self, radius: f64) -> Result<f64> {
        if radius <= self.max_radius {
            return Ok(radius);
        }
        match self.action {
            LargeBlurRadiusAction::Reject => Err(Error::invalid_value(
                "filter blur radius",
                radius,
                "must be less than or equal to configured large blur radius limit",
            )),
            LargeBlurRadiusAction::Clamp => Ok(self.max_radius),
        }
    }
}

/// Sampling outside source bounds for blur kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransparentEdgeSamplingPolicy {
    TransparentBlack,
}

/// Blur planning policy for CPU/reference and backend-compatible blur models.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurPolicy {
    radius_interpretation: BlurRadiusInterpretation,
    kernel_support: KernelSupportRadius,
    large_radius: LargeBlurRadiusPolicy,
    edge_sampling: TransparentEdgeSamplingPolicy,
}

impl BlurPolicy {
    pub fn try_new(
        radius_interpretation: BlurRadiusInterpretation,
        kernel_support: KernelSupportRadius,
        large_radius: LargeBlurRadiusPolicy,
        edge_sampling: TransparentEdgeSamplingPolicy,
    ) -> Result<Self> {
        if kernel_support.standard_deviation_multiple() <= 0.0 {
            return Err(Error::invalid_value(
                "blur kernel support radius",
                kernel_support.standard_deviation_multiple(),
                "must be finite and greater than 0",
            ));
        }
        if large_radius.max_radius() <= 0.0 {
            return Err(Error::invalid_value(
                "large blur radius limit",
                large_radius.max_radius(),
                "must be finite and greater than 0",
            ));
        }
        Ok(Self {
            radius_interpretation,
            kernel_support,
            large_radius,
            edge_sampling,
        })
    }

    pub fn css_filter_default() -> Self {
        Self::try_new(
            BlurRadiusInterpretation::CssLengthAsStandardDeviation,
            KernelSupportRadius::try_standard_deviation_multiple(
                CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
            )
            .expect("default kernel support radius is valid"),
            LargeBlurRadiusPolicy::try_reject_above(DEFAULT_MAX_BLUR_RADIUS)
                .expect("default large-radius policy is valid"),
            TransparentEdgeSamplingPolicy::TransparentBlack,
        )
        .expect("default CSS filter blur policy is valid")
    }

    #[must_use]
    pub const fn radius_interpretation(self) -> BlurRadiusInterpretation {
        self.radius_interpretation
    }

    #[must_use]
    pub const fn kernel_support(self) -> KernelSupportRadius {
        self.kernel_support
    }

    #[must_use]
    pub const fn large_radius_policy(self) -> LargeBlurRadiusPolicy {
        self.large_radius
    }

    #[must_use]
    pub const fn edge_sampling(self) -> TransparentEdgeSamplingPolicy {
        self.edge_sampling
    }

    pub fn support_radius(self, blur: FilterBlur) -> Result<f64> {
        Ok(self
            .kernel_support
            .support_radius(self.standard_deviation(blur)?))
    }

    pub(crate) fn standard_deviation(self, blur: FilterBlur) -> Result<f64> {
        let radius = self.large_radius.resolve_radius(blur.radius())?;
        Ok(self.radius_interpretation.standard_deviation(radius))
    }
}

pub(crate) fn vello_outer_shadow_support_radius(blur_radius: f64) -> Result<f64> {
    if !blur_radius.is_finite() || blur_radius < 0.0 {
        return Err(Error::invalid_value(
            "shadow blur",
            blur_radius,
            "must be finite and non-negative",
        ));
    }
    let standard_deviation =
        BlurRadiusInterpretation::VelloShadowBlurRadiusAsDiameter.standard_deviation(blur_radius);
    let support = KernelSupportRadius {
        standard_deviation_multiple: CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
    }
    .support_radius(standard_deviation);
    if !support.is_finite() {
        return Err(Error::invalid_value(
            "box shadow blur support",
            blur_radius,
            "must produce finite Vello-compatible support",
        ));
    }
    Ok(support)
}

/// Outward device-pixel conversion policy for planned filter execution regions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DevicePixelConversionPolicy;

impl DevicePixelConversionPolicy {
    #[must_use]
    pub const fn outward() -> Self {
        Self
    }

    pub fn convert_region(
        self,
        region: FilterExecutionRegion,
        scale: f64,
    ) -> Result<FilterDeviceBounds> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::invalid_value(
                "filter device-pixel scale",
                scale,
                "must be finite and greater than 0",
            ));
        }

        let rect = region.rect();
        let max = rect.max();
        let x = checked_floor_i32(rect.x() * scale, "filter device bounds x")?;
        let y = checked_floor_i32(rect.y() * scale, "filter device bounds y")?;
        let max_x = checked_ceil_i32(max.x() * scale, "filter device bounds max x")?;
        let max_y = checked_ceil_i32(max.y() * scale, "filter device bounds max y")?;
        let width = u32::try_from(i64::from(max_x) - i64::from(x)).map_err(|_| {
            Error::invalid_value(
                "filter device bounds width",
                i64::from(max_x) - i64::from(x),
                "must fit in u32 device pixels",
            )
        })?;
        let height = u32::try_from(i64::from(max_y) - i64::from(y)).map_err(|_| {
            Error::invalid_value(
                "filter device bounds height",
                i64::from(max_y) - i64::from(y),
                "must fit in u32 device pixels",
            )
        })?;
        if width == 0 || height == 0 {
            return Err(Error::invalid_value(
                "filter device bounds",
                format!("{width}x{height}"),
                "must have positive width and height",
            ));
        }
        Ok(FilterDeviceBounds {
            x,
            y,
            width,
            height,
        })
    }
}

/// Device-pixel bounds for a planned execution region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterDeviceBounds {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

impl FilterDeviceBounds {
    #[must_use]
    pub const fn x(self) -> i32 {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> i32 {
        self.y
    }

    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

fn validate_filter_bounds(rect: Rect, name: &str) -> Result<()> {
    validate_finite(rect.x(), &format!("{name} x"))?;
    validate_finite(rect.y(), &format!("{name} y"))?;
    validate_positive_dimension(rect.width(), &format!("{name} width"))?;
    validate_positive_dimension(rect.height(), &format!("{name} height"))
}

fn validate_filter_outset_value(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_value(name, value, "must be finite"));
    }
    Ok(())
}

fn validate_positive_dimension(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and greater than 0",
        ));
    }
    Ok(())
}

fn intersect_rects(a: Rect, b: Rect) -> Option<Rect> {
    let a_max = a.max();
    let b_max = b.max();
    let min_x = a.x().max(b.x());
    let min_y = a.y().max(b.y());
    let max_x = a_max.x().min(b_max.x());
    let max_y = a_max.y().min(b_max.y());
    let width = max_x - min_x;
    let height = max_y - min_y;
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    Some(Rect::new(min_x, min_y, width, height))
}

fn checked_floor_i32(value: f64, name: &str) -> Result<i32> {
    checked_rounded_i32(value.floor(), name)
}

fn checked_ceil_i32(value: f64, name: &str) -> Result<i32> {
    checked_rounded_i32(value.ceil(), name)
}

fn checked_rounded_i32(value: f64, name: &str) -> Result<i32> {
    if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        return Err(Error::invalid_value(
            name,
            value,
            "must fit in i32 device pixels",
        ));
    }
    Ok(value as i32)
}

/// Context-free algorithm phase for one authored filter list.
///
/// This phase preserves authored operation order and per-operation color clamp
/// boundaries without carrying pixels, logical bounds, or backend state.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AlgorithmFilterPlan {
    authored_operation_count: usize,
    steps: Vec<AlgorithmFilterStep>,
}

impl AlgorithmFilterPlan {
    pub(crate) fn from_filter_list(filters: &FilterList) -> Self {
        let authored_operation_count = filters.ops().len();
        let mut steps = Vec::new();
        let mut color_run = Vec::new();

        for op in filters.ops() {
            match op.kind() {
                FilterOpKind::Blur(blur) if blur.radius() == 0.0 => {}
                FilterOpKind::Blur(blur) => {
                    flush_algorithm_color_run(&mut steps, &mut color_run);
                    steps.push(AlgorithmFilterStep::Blur(*blur));
                }
                FilterOpKind::DropShadow(shadow) => {
                    flush_algorithm_color_run(&mut steps, &mut color_run);
                    steps.push(AlgorithmFilterStep::DropShadow(*shadow));
                }
                FilterOpKind::Brightness(amount) => push_algorithm_color_operation(
                    &mut color_run,
                    ColorFilterOp::Brightness(*amount),
                ),
                FilterOpKind::Contrast(amount) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::Contrast(*amount))
                }
                FilterOpKind::Grayscale(amount) => push_algorithm_color_operation(
                    &mut color_run,
                    ColorFilterOp::Grayscale(*amount),
                ),
                FilterOpKind::HueRotate(angle) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::HueRotate(*angle))
                }
                FilterOpKind::Invert(amount) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::Invert(*amount))
                }
                FilterOpKind::Opacity(amount) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::Opacity(*amount))
                }
                FilterOpKind::Saturate(amount) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::Saturate(*amount))
                }
                FilterOpKind::Sepia(amount) => {
                    push_algorithm_color_operation(&mut color_run, ColorFilterOp::Sepia(*amount))
                }
            }
        }

        flush_algorithm_color_run(&mut steps, &mut color_run);
        Self {
            authored_operation_count,
            steps,
        }
    }

    #[must_use]
    pub(crate) const fn authored_operation_count(&self) -> usize {
        self.authored_operation_count
    }

    #[must_use]
    pub(crate) fn steps(&self) -> &[AlgorithmFilterStep] {
        &self.steps
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AlgorithmFilterStep {
    ColorRun(AlgorithmColorFilterRun),
    Blur(FilterBlur),
    DropShadow(FilterDropShadow),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AlgorithmColorFilterRun {
    operations: Vec<ClampedColorFilterOperation>,
}

impl AlgorithmColorFilterRun {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C06 T3 records authored color operations for C07 lowering."
        )
    )]
    #[must_use]
    pub(crate) fn operations(&self) -> &[ClampedColorFilterOperation] {
        &self.operations
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ColorClampBoundary {
    ClampStraightRgbaToUnitThenPremultiply,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClampedColorFilterOperation {
    operation: ColorFilterOp,
    clamp_boundary: ColorClampBoundary,
}

impl ClampedColorFilterOperation {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C06 T3 records each authored color operation for C07 lowering."
        )
    )]
    #[must_use]
    pub(crate) const fn operation(self) -> ColorFilterOp {
        self.operation
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C06 T3 records each authored clamp boundary for C07 lowering."
        )
    )]
    #[must_use]
    pub(crate) const fn clamp_boundary(self) -> ColorClampBoundary {
        self.clamp_boundary
    }
}

fn flush_algorithm_color_run(
    steps: &mut Vec<AlgorithmFilterStep>,
    color_run: &mut Vec<ClampedColorFilterOperation>,
) {
    if !color_run.is_empty() {
        steps.push(AlgorithmFilterStep::ColorRun(AlgorithmColorFilterRun {
            operations: std::mem::take(color_run),
        }));
    }
}

fn push_algorithm_color_operation(
    color_run: &mut Vec<ClampedColorFilterOperation>,
    operation: ColorFilterOp,
) {
    color_run.push(ClampedColorFilterOperation {
        operation,
        clamp_boundary: ColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply,
    });
}

/// Render-owned ordered classifier for materialized image filters.
///
/// This is an execution plan shape, not execution itself. Color-only runs are
/// compiled into the existing color pipeline, while pixel-moving operations
/// remain named steps for later region planning and byte execution.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedImageFilterPipeline {
    steps: Vec<MaterializedImageFilterStep>,
}

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

#[derive(Clone, Debug, PartialEq)]
pub enum MaterializedImageFilterStep {
    ColorFilters(CompiledColorFilterPipeline),
    Blur(FilterBlur),
    /// Executes an intrinsically valid solid-color filter drop shadow.
    DropShadow(FilterDropShadow),
}

impl FilterList {
    pub fn materialized_image_filter_pipeline(
        &self,
    ) -> Result<Option<MaterializedImageFilterPipeline>> {
        MaterializedImageFilterPipeline::try_from_filter_list(self)
    }
}

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
