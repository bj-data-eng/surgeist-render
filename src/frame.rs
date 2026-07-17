use super::{
    error::{Error, Result},
    filter::{
        AlgorithmColorFilterRun, AlgorithmFilterPlan, AlgorithmFilterStep,
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS, DevicePixelConversionPolicy, FilterOutset,
        FilterRegionPlan, FilterSourceBounds,
    },
    geometry::{Point, Rect, Transform},
    style::{FilterBlur, FilterDropShadow, FilterList},
};

#[cfg(test)]
use super::{filter::ColorClampBoundary, style::ColorFilterOp};

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T2 stages the resolved-frame planner that C06 T6 will invoke."
    )
)]
pub(crate) struct FrameContext {
    surface_scale: f64,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T2 stages the resolved-frame planner that C06 T6 will invoke."
    )
)]
impl FrameContext {
    pub(crate) fn try_new(surface_scale: f64) -> Result<Self> {
        if !surface_scale.is_finite() || surface_scale <= 0.0 {
            return Err(Error::invalid_value(
                "frame surface scale",
                surface_scale,
                "must be finite and greater than 0",
            ));
        }
        Ok(Self { surface_scale })
    }

    fn plan_local_bounds(
        self,
        logical_bounds: LogicalBounds,
        transform: Transform,
    ) -> Result<FrameSpatialPlan> {
        let linear_metrics = linear_transform_metrics(transform)?;
        let logical_bounds = match logical_bounds {
            LogicalBounds::Empty(bounds) => {
                return Ok(FrameSpatialPlan::Empty(EmptyFrameSpatialPlan {
                    logical_bounds: LogicalBounds::Empty(bounds),
                }));
            }
            LogicalBounds::NonEmpty(bounds) => bounds,
        };
        if linear_metrics.rank_deficient {
            return Ok(FrameSpatialPlan::Empty(EmptyFrameSpatialPlan {
                logical_bounds: LogicalBounds::NonEmpty(logical_bounds),
            }));
        }

        let raster_scale = RasterScale::try_new(checked_mul(
            self.surface_scale,
            linear_metrics.largest_singular_value,
            "frame local raster scale",
        )?)?;
        let source = FilterSourceBounds::try_new(logical_bounds.rect())?;
        let region = FilterRegionPlan::try_new(source, FilterOutset::zero(), None)?;
        let device = DevicePixelConversionPolicy::outward()
            .convert_region(region.execution_region(), raster_scale.get())?;
        let device_origin = SignedDeviceOrigin::new(device.x(), device.y());
        let device_extent = PositiveDeviceExtent::try_new(device.width(), device.height())?;
        let texel_center_mapping = TexelCenterMapping::try_new(device_origin, raster_scale)?;

        Ok(FrameSpatialPlan::NonEmpty(NonEmptyFrameSpatialPlan {
            logical_bounds,
            device_origin,
            device_extent,
            raster_scale,
            texel_center_mapping,
        }))
    }

    fn plan_filter_list(
        self,
        source_bounds: LogicalBounds,
        transform: Transform,
        filters: &FilterList,
        source_role: FilterSourceRole,
    ) -> Result<ResolvedFrameFilterPlan> {
        let algorithm = AlgorithmFilterPlan::from_filter_list(filters);
        let initial_spatial = self.plan_local_bounds(source_bounds, transform)?;
        let FrameSpatialPlan::NonEmpty(initial_spatial) = initial_spatial else {
            return Ok(ResolvedFrameFilterPlan::Empty(EmptyResolvedFilterPlan {
                source_bounds,
                authored_operation_count: algorithm.authored_operation_count(),
            }));
        };

        let initial_bounds = initial_spatial.logical_bounds;
        let raster_scale = initial_spatial.raster_scale;
        let mut current_bounds = initial_bounds;
        let mut steps = Vec::with_capacity(algorithm.steps().len());

        for algorithm_step in algorithm.steps().iter().cloned() {
            let source_bounds = current_bounds;
            let (result_bounds, edge_policy, operation_intent) = match algorithm_step {
                AlgorithmFilterStep::ColorRun(run) => (
                    source_bounds,
                    FilterEdgePolicy::NoSampling,
                    ResolvedFilterOperationIntent::ColorRun(run),
                ),
                AlgorithmFilterStep::Blur(blur) => {
                    let support = InclusiveFilterKernelSupport::try_new(blur, raster_scale)?;
                    let result_bounds = source_bounds
                        .try_inflate_uniform(support.logical_radius, "filter blur result bounds")?;
                    let edge_policy = match source_role {
                        FilterSourceRole::Ordinary => FilterEdgePolicy::TransparentBlack,
                        FilterSourceRole::Backdrop => FilterEdgePolicy::SemanticBorderMirror {
                            semantic_border: initial_bounds,
                        },
                    };
                    (
                        result_bounds,
                        edge_policy,
                        ResolvedFilterOperationIntent::Blur(ResolvedBlurIntent {
                            authored_blur: blur,
                            support,
                        }),
                    )
                }
                AlgorithmFilterStep::DropShadow(shadow) => {
                    let support =
                        InclusiveFilterKernelSupport::try_new(shadow.blur(), raster_scale)?;
                    let shadow_bounds = source_bounds
                        .try_inflate_uniform(
                            support.logical_radius,
                            "filter drop-shadow alpha bounds",
                        )?
                        .try_translate(shadow.offset(), "filter drop-shadow offset alpha bounds")?;
                    let result_bounds = source_bounds
                        .try_union(shadow_bounds, "filter drop-shadow result bounds")?;
                    (
                        result_bounds,
                        FilterEdgePolicy::TransparentBlack,
                        ResolvedFilterOperationIntent::DropShadow(ResolvedDropShadowIntent {
                            authored_shadow: shadow,
                            alpha_source: DropShadowAlphaSource::SourceAlpha,
                            support,
                            offset_sampling: DropShadowOffsetSampling::ContinuousLinear,
                            source_composition:
                                DropShadowSourceComposition::RetainUnchangedForSourceOver,
                        }),
                    )
                }
            };
            let spatial_mapping = ResolvedFilterSpatialMapping {
                source: self.plan_non_empty_local_bounds(source_bounds, transform)?,
                result: self.plan_non_empty_local_bounds(result_bounds, transform)?,
            };
            steps.push(ResolvedFilterStep {
                source_bounds,
                result_bounds,
                spatial_mapping,
                edge_policy,
                operation_intent,
            });
            current_bounds = result_bounds;
        }

        Ok(ResolvedFrameFilterPlan::NonEmpty(
            NonEmptyResolvedFilterPlan {
                initial_bounds,
                final_bounds: current_bounds,
                authored_operation_count: algorithm.authored_operation_count(),
                steps,
            },
        ))
    }

    fn plan_non_empty_local_bounds(
        self,
        logical_bounds: NonEmptyLogicalBounds,
        transform: Transform,
    ) -> Result<NonEmptyFrameSpatialPlan> {
        match self.plan_local_bounds(LogicalBounds::NonEmpty(logical_bounds), transform)? {
            FrameSpatialPlan::NonEmpty(plan) => Ok(plan),
            FrameSpatialPlan::Empty(_) => Err(Error::invalid_value(
                "filter step spatial mapping",
                "empty",
                "must remain non-empty after the frame transform was validated",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LogicalBounds {
    Empty(EmptyLogicalBounds),
    NonEmpty(NonEmptyLogicalBounds),
}

impl LogicalBounds {
    pub(crate) fn try_from_rect(rect: Rect, name: &str) -> Result<Self> {
        validate_finite(rect.x(), &format!("{name} x"))?;
        validate_finite(rect.y(), &format!("{name} y"))?;
        validate_non_negative(rect.width(), &format!("{name} width"))?;
        validate_non_negative(rect.height(), &format!("{name} height"))?;
        checked_add(rect.x(), rect.width(), &format!("{name} max x"))?;
        checked_add(rect.y(), rect.height(), &format!("{name} max y"))?;

        if rect.width() == 0.0 || rect.height() == 0.0 {
            Ok(Self::Empty(EmptyLogicalBounds { rect }))
        } else {
            Ok(Self::NonEmpty(NonEmptyLogicalBounds { rect }))
        }
    }

    pub(crate) fn try_transform(self, transform: Transform, name: &str) -> Result<Self> {
        let Self::NonEmpty(bounds) = self else {
            return Ok(self);
        };
        let rect = bounds.rect();
        let max = rect.max();
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (x, y) in [
            (rect.x(), rect.y()),
            (max.x(), rect.y()),
            (max.x(), max.y()),
            (rect.x(), max.y()),
        ] {
            let point = transform_point(transform, x, y, name)?;
            min_x = min_x.min(point.x());
            min_y = min_y.min(point.y());
            max_x = max_x.max(point.x());
            max_y = max_y.max(point.y());
        }
        let width = checked_sub(max_x, min_x, &format!("{name} width"))?;
        let height = checked_sub(max_y, min_y, &format!("{name} height"))?;
        let transformed_rect = Rect::new(min_x, min_y, width, height);
        let transformed = Self::try_from_rect(transformed_rect, name)?;
        if linear_transform_is_rank_deficient(transform)?
            && matches!(transformed, Self::NonEmpty(_))
        {
            return Ok(Self::Empty(EmptyLogicalBounds {
                rect: Rect::new(min_x, min_y, 0.0, 0.0),
            }));
        }
        Ok(transformed)
    }

    pub(crate) fn union(self, other: Self, name: &str) -> Result<Self> {
        match (self, other) {
            (Self::Empty(_), bounds) | (bounds, Self::Empty(_)) => Ok(bounds),
            (Self::NonEmpty(a), Self::NonEmpty(b)) => {
                let a = a.rect();
                let b = b.rect();
                let a_max = a.max();
                let b_max = b.max();
                let min_x = a.x().min(b.x());
                let min_y = a.y().min(b.y());
                let max_x = a_max.x().max(b_max.x());
                let max_y = a_max.y().max(b_max.y());
                let width = checked_sub(max_x, min_x, &format!("{name} width"))?;
                let height = checked_sub(max_y, min_y, &format!("{name} height"))?;
                Self::try_from_rect(Rect::new(min_x, min_y, width, height), name)
            }
        }
    }

    #[must_use]
    pub(crate) const fn rect(self) -> Rect {
        match self {
            Self::Empty(bounds) => bounds.rect(),
            Self::NonEmpty(bounds) => bounds.rect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EmptyLogicalBounds {
    rect: Rect,
}

impl EmptyLogicalBounds {
    #[must_use]
    const fn rect(self) -> Rect {
        self.rect
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NonEmptyLogicalBounds {
    rect: Rect,
}

impl NonEmptyLogicalBounds {
    #[must_use]
    const fn rect(self) -> Rect {
        self.rect
    }

    fn try_inflate_uniform(self, amount: f64, name: &str) -> Result<Self> {
        validate_non_negative(amount, &format!("{name} outset"))?;
        let rect = self.rect();
        let doubled = checked_mul(amount, 2.0, &format!("{name} total outset"))?;
        non_empty_logical_bounds(
            Rect::new(
                checked_sub(rect.x(), amount, &format!("{name} x"))?,
                checked_sub(rect.y(), amount, &format!("{name} y"))?,
                checked_add(rect.width(), doubled, &format!("{name} width"))?,
                checked_add(rect.height(), doubled, &format!("{name} height"))?,
            ),
            name,
        )
    }

    fn try_translate(self, offset: Point, name: &str) -> Result<Self> {
        let rect = self.rect();
        non_empty_logical_bounds(
            Rect::new(
                checked_add(rect.x(), offset.x(), &format!("{name} x"))?,
                checked_add(rect.y(), offset.y(), &format!("{name} y"))?,
                rect.width(),
                rect.height(),
            ),
            name,
        )
    }

    fn try_union(self, other: Self, name: &str) -> Result<Self> {
        match LogicalBounds::NonEmpty(self).union(LogicalBounds::NonEmpty(other), name)? {
            LogicalBounds::NonEmpty(bounds) => Ok(bounds),
            LogicalBounds::Empty(_) => Err(Error::invalid_value(
                name,
                "empty",
                "must remain non-empty when two non-empty bounds are united",
            )),
        }
    }
}

fn non_empty_logical_bounds(rect: Rect, name: &str) -> Result<NonEmptyLogicalBounds> {
    match LogicalBounds::try_from_rect(rect, name)? {
        LogicalBounds::NonEmpty(bounds) => Ok(bounds),
        LogicalBounds::Empty(_) => Err(Error::invalid_value(
            name,
            "empty",
            "must have positive width and height",
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SignedDeviceOrigin {
    x: i32,
    y: i32,
}

impl SignedDeviceOrigin {
    #[must_use]
    const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PositiveDeviceExtent {
    width: u32,
    height: u32,
}

impl PositiveDeviceExtent {
    fn try_new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_value(
                "frame device extent",
                format!("{width}x{height}"),
                "must have positive width and height",
            ));
        }
        Ok(Self { width, height })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RasterScale(f64);

impl RasterScale {
    fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_value(
                "frame local raster scale",
                value,
                "must be finite and greater than 0",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TexelCenterMapping {
    origin: Point,
    raster_scale: RasterScale,
}

impl TexelCenterMapping {
    fn try_new(device_origin: SignedDeviceOrigin, raster_scale: RasterScale) -> Result<Self> {
        let origin = Point::try_new(
            checked_div(
                f64::from(device_origin.x),
                raster_scale.get(),
                "frame texel mapping origin x",
            )?,
            checked_div(
                f64::from(device_origin.y),
                raster_scale.get(),
                "frame texel mapping origin y",
            )?,
        )?;
        Ok(Self {
            origin,
            raster_scale,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C06 T2 records texel-center mappings for later resolved pass lowering."
        )
    )]
    fn point_for(self, i: u32, j: u32) -> Result<Point> {
        let x_offset = checked_div(
            checked_add(f64::from(i), 0.5, "frame texel center i")?,
            self.raster_scale.get(),
            "frame texel center x offset",
        )?;
        let y_offset = checked_div(
            checked_add(f64::from(j), 0.5, "frame texel center j")?,
            self.raster_scale.get(),
            "frame texel center y offset",
        )?;
        Point::try_new(
            checked_add(self.origin.x(), x_offset, "frame texel center x")?,
            checked_add(self.origin.y(), y_offset, "frame texel center y")?,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FrameSpatialPlan {
    Empty(EmptyFrameSpatialPlan),
    NonEmpty(NonEmptyFrameSpatialPlan),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T3 stages ordinary and backdrop plans for C06 T5-T6 consumers."
    )
)]
enum FilterSourceRole {
    Ordinary,
    Backdrop,
}

#[derive(Clone, Debug, PartialEq)]
enum ResolvedFrameFilterPlan {
    Empty(EmptyResolvedFilterPlan),
    NonEmpty(NonEmptyResolvedFilterPlan),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EmptyResolvedFilterPlan {
    source_bounds: LogicalBounds,
    authored_operation_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct NonEmptyResolvedFilterPlan {
    initial_bounds: NonEmptyLogicalBounds,
    final_bounds: NonEmptyLogicalBounds,
    authored_operation_count: usize,
    steps: Vec<ResolvedFilterStep>,
}

#[derive(Clone, Debug, PartialEq)]
struct ResolvedFilterStep {
    source_bounds: NonEmptyLogicalBounds,
    result_bounds: NonEmptyLogicalBounds,
    spatial_mapping: ResolvedFilterSpatialMapping,
    edge_policy: FilterEdgePolicy,
    operation_intent: ResolvedFilterOperationIntent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedFilterSpatialMapping {
    source: NonEmptyFrameSpatialPlan,
    result: NonEmptyFrameSpatialPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FilterEdgePolicy {
    NoSampling,
    TransparentBlack,
    SemanticBorderMirror {
        semantic_border: NonEmptyLogicalBounds,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum ResolvedFilterOperationIntent {
    ColorRun(AlgorithmColorFilterRun),
    Blur(ResolvedBlurIntent),
    DropShadow(ResolvedDropShadowIntent),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedBlurIntent {
    authored_blur: FilterBlur,
    support: InclusiveFilterKernelSupport,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedDropShadowIntent {
    authored_shadow: FilterDropShadow,
    alpha_source: DropShadowAlphaSource,
    support: InclusiveFilterKernelSupport,
    offset_sampling: DropShadowOffsetSampling,
    source_composition: DropShadowSourceComposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DropShadowAlphaSource {
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DropShadowOffsetSampling {
    ContinuousLinear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DropShadowSourceComposition {
    RetainUnchangedForSourceOver,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InclusiveFilterKernelSupport {
    device_radius: u32,
    logical_radius: f64,
}

impl InclusiveFilterKernelSupport {
    fn try_new(blur: FilterBlur, raster_scale: RasterScale) -> Result<Self> {
        let scaled_standard_deviation = checked_mul(
            blur.radius(),
            raster_scale.get(),
            "filter blur scaled standard deviation",
        )?;
        let inclusive_radius = checked_mul(
            CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
            scaled_standard_deviation,
            "filter blur inclusive support",
        )?
        .ceil();
        if inclusive_radius < 0.0 || inclusive_radius > f64::from(u32::MAX) {
            return Err(Error::invalid_value(
                "filter blur inclusive support",
                inclusive_radius,
                "must fit in u32 device taps",
            ));
        }
        let device_radius = inclusive_radius as u32;
        let logical_radius = checked_div(
            inclusive_radius,
            raster_scale.get(),
            "filter blur logical support",
        )?;
        Ok(Self {
            device_radius,
            logical_radius,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EmptyFrameSpatialPlan {
    logical_bounds: LogicalBounds,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NonEmptyFrameSpatialPlan {
    logical_bounds: NonEmptyLogicalBounds,
    device_origin: SignedDeviceOrigin,
    device_extent: PositiveDeviceExtent,
    raster_scale: RasterScale,
    texel_center_mapping: TexelCenterMapping,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LinearTransformMetrics {
    largest_singular_value: f64,
    rank_deficient: bool,
}

fn linear_transform_metrics(transform: Transform) -> Result<LinearTransformMetrics> {
    let [a, b, c, d, _, _] = transform.as_array();
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if coefficient_scale == 0.0 {
        return Ok(LinearTransformMetrics {
            largest_singular_value: 0.0,
            rank_deficient: true,
        });
    }

    let a = checked_div(a, coefficient_scale, "frame transform coefficient a")?;
    let b = checked_div(b, coefficient_scale, "frame transform coefficient b")?;
    let c = checked_div(c, coefficient_scale, "frame transform coefficient c")?;
    let d = checked_div(d, coefficient_scale, "frame transform coefficient d")?;
    let frobenius_squared = checked_add(
        checked_add(
            checked_mul(a, a, "frame transform a squared")?,
            checked_mul(b, b, "frame transform b squared")?,
            "frame transform first column norm",
        )?,
        checked_add(
            checked_mul(c, c, "frame transform c squared")?,
            checked_mul(d, d, "frame transform d squared")?,
            "frame transform second column norm",
        )?,
        "frame transform Frobenius norm squared",
    )?;
    let determinant = checked_sub(
        checked_mul(a, d, "frame transform normalized determinant ad")?,
        checked_mul(b, c, "frame transform normalized determinant bc")?,
        "frame transform normalized determinant",
    )?;
    let discriminant = checked_sub(
        checked_mul(
            frobenius_squared,
            frobenius_squared,
            "frame singular-value discriminant norm",
        )?,
        checked_mul(
            4.0,
            checked_mul(
                determinant,
                determinant,
                "frame singular-value determinant squared",
            )?,
            "frame singular-value determinant term",
        )?,
        "frame singular-value discriminant",
    )?
    .max(0.0);
    let largest_eigenvalue = checked_mul(
        0.5,
        checked_add(
            frobenius_squared,
            discriminant.sqrt(),
            "frame largest singular-value eigenvalue",
        )?,
        "frame largest singular-value eigenvalue",
    )?;
    let normalized = largest_eigenvalue.sqrt();
    validate_finite(normalized, "frame normalized largest singular value")?;
    let largest_singular_value = checked_mul(
        coefficient_scale,
        normalized,
        "frame largest singular value",
    )?;
    Ok(LinearTransformMetrics {
        largest_singular_value,
        rank_deficient: determinant == 0.0,
    })
}

fn linear_transform_is_rank_deficient(transform: Transform) -> Result<bool> {
    let [a, b, c, d, _, _] = transform.as_array();
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if coefficient_scale == 0.0 {
        return Ok(true);
    }

    let a = checked_div(a, coefficient_scale, "frame transform coefficient a")?;
    let b = checked_div(b, coefficient_scale, "frame transform coefficient b")?;
    let c = checked_div(c, coefficient_scale, "frame transform coefficient c")?;
    let d = checked_div(d, coefficient_scale, "frame transform coefficient d")?;
    let determinant = checked_sub(
        checked_mul(a, d, "frame transform normalized determinant ad")?,
        checked_mul(b, c, "frame transform normalized determinant bc")?,
        "frame transform normalized determinant",
    )?;
    Ok(determinant == 0.0)
}

fn transform_point(transform: Transform, x: f64, y: f64, name: &str) -> Result<Point> {
    let [a, b, c, d, e, f] = transform.as_array();
    let transformed_x = checked_add(
        checked_add(
            checked_mul(a, x, &format!("{name} x product"))?,
            checked_mul(c, y, &format!("{name} x cross product"))?,
            &format!("{name} x linear value"),
        )?,
        e,
        &format!("{name} x"),
    )?;
    let transformed_y = checked_add(
        checked_add(
            checked_mul(b, x, &format!("{name} y cross product"))?,
            checked_mul(d, y, &format!("{name} y product"))?,
            &format!("{name} y linear value"),
        )?,
        f,
        &format!("{name} y"),
    )?;
    Point::try_new(transformed_x, transformed_y)
}

fn validate_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_value(name, value, "must be finite"));
    }
    Ok(())
}

fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

fn checked_add(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left + right, name, left, "+", right)
}

fn checked_sub(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left - right, name, left, "-", right)
}

fn checked_mul(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left * right, name, left, "*", right)
}

fn checked_div(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left / right, name, left, "/", right)
}

fn checked_finite_result(
    result: f64,
    name: &str,
    left: f64,
    operation: &'static str,
    right: f64,
) -> Result<f64> {
    if !result.is_finite() {
        return Err(Error::invalid_value(
            name,
            format!("{left} {operation} {right}"),
            "must produce a finite value",
        ));
    }
    Ok(result)
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OrderedFilterEdgeObservation {
    NoSampling,
    TransparentBlack,
    SemanticBorderMirror([f64; 4]),
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OrderedFilterIntentObservation {
    ColorRun {
        operations: Vec<ColorFilterOp>,
        clamp_boundaries_after_operation: Vec<usize>,
    },
    Blur {
        standard_deviation: f64,
        inclusive_support_taps: u32,
    },
    DropShadow {
        offset: (f64, f64),
        standard_deviation: f64,
        inclusive_support_taps: u32,
        uses_source_alpha: bool,
        retains_unchanged_source: bool,
        continuous_offset: bool,
    },
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OrderedFilterStepObservation {
    pub(crate) source_bounds: [f64; 4],
    pub(crate) result_bounds: Option<[f64; 4]>,
    pub(crate) source_device_origin: Option<(i32, i32)>,
    pub(crate) source_device_extent: Option<(u32, u32)>,
    pub(crate) result_device_origin: Option<(i32, i32)>,
    pub(crate) result_device_extent: Option<(u32, u32)>,
    pub(crate) edge: OrderedFilterEdgeObservation,
    pub(crate) intent: OrderedFilterIntentObservation,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OrderedFilterPlanObservation {
    pub(crate) initial_bounds: [f64; 4],
    pub(crate) final_bounds: [f64; 4],
    pub(crate) authored_operation_count: usize,
    pub(crate) is_empty: bool,
    pub(crate) has_spatial_mapping: bool,
    pub(crate) steps: Vec<OrderedFilterStepObservation>,
}

#[cfg(test)]
pub(crate) fn ordered_filter_plan_for_test(
    filters: &FilterList,
    source_rect: Rect,
    transform: Transform,
    surface_scale: f64,
    backdrop: bool,
) -> Result<OrderedFilterPlanObservation> {
    let source_bounds = LogicalBounds::try_from_rect(source_rect, "filter plan source bounds")?;
    let source_role = if backdrop {
        FilterSourceRole::Backdrop
    } else {
        FilterSourceRole::Ordinary
    };
    let plan = FrameContext::try_new(surface_scale)?.plan_filter_list(
        source_bounds,
        transform,
        filters,
        source_role,
    )?;

    match plan {
        ResolvedFrameFilterPlan::Empty(plan) => {
            let bounds = logical_rect_values(plan.source_bounds.rect());
            Ok(OrderedFilterPlanObservation {
                initial_bounds: bounds,
                final_bounds: bounds,
                authored_operation_count: plan.authored_operation_count,
                is_empty: true,
                has_spatial_mapping: false,
                steps: Vec::new(),
            })
        }
        ResolvedFrameFilterPlan::NonEmpty(plan) => {
            let steps = plan
                .steps
                .into_iter()
                .map(observe_resolved_filter_step)
                .collect();
            Ok(OrderedFilterPlanObservation {
                initial_bounds: logical_rect_values(plan.initial_bounds.rect()),
                final_bounds: logical_rect_values(plan.final_bounds.rect()),
                authored_operation_count: plan.authored_operation_count,
                is_empty: false,
                has_spatial_mapping: true,
                steps,
            })
        }
    }
}

#[cfg(test)]
fn observe_resolved_filter_step(step: ResolvedFilterStep) -> OrderedFilterStepObservation {
    let source_device_origin = Some((
        step.spatial_mapping.source.device_origin.x,
        step.spatial_mapping.source.device_origin.y,
    ));
    let source_device_extent = Some((
        step.spatial_mapping.source.device_extent.width,
        step.spatial_mapping.source.device_extent.height,
    ));
    let result_device_origin = Some((
        step.spatial_mapping.result.device_origin.x,
        step.spatial_mapping.result.device_origin.y,
    ));
    let result_device_extent = Some((
        step.spatial_mapping.result.device_extent.width,
        step.spatial_mapping.result.device_extent.height,
    ));
    let edge = match step.edge_policy {
        FilterEdgePolicy::NoSampling => OrderedFilterEdgeObservation::NoSampling,
        FilterEdgePolicy::TransparentBlack => OrderedFilterEdgeObservation::TransparentBlack,
        FilterEdgePolicy::SemanticBorderMirror { semantic_border } => {
            OrderedFilterEdgeObservation::SemanticBorderMirror(logical_rect_values(
                semantic_border.rect(),
            ))
        }
    };
    let intent = match step.operation_intent {
        ResolvedFilterOperationIntent::ColorRun(run) => {
            let operations = run
                .operations()
                .iter()
                .copied()
                .map(|operation| operation.operation())
                .collect();
            let clamp_boundaries_after_operation = run
                .operations()
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, operation)| {
                    (operation.clamp_boundary()
                        == ColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply)
                        .then_some(index)
                })
                .collect();
            OrderedFilterIntentObservation::ColorRun {
                operations,
                clamp_boundaries_after_operation,
            }
        }
        ResolvedFilterOperationIntent::Blur(intent) => OrderedFilterIntentObservation::Blur {
            standard_deviation: intent.authored_blur.radius(),
            inclusive_support_taps: intent.support.device_radius,
        },
        ResolvedFilterOperationIntent::DropShadow(intent) => {
            let offset = intent.authored_shadow.offset();
            OrderedFilterIntentObservation::DropShadow {
                offset: (offset.x(), offset.y()),
                standard_deviation: intent.authored_shadow.blur().radius(),
                inclusive_support_taps: intent.support.device_radius,
                uses_source_alpha: intent.alpha_source == DropShadowAlphaSource::SourceAlpha,
                retains_unchanged_source: intent.source_composition
                    == DropShadowSourceComposition::RetainUnchangedForSourceOver,
                continuous_offset: intent.offset_sampling
                    == DropShadowOffsetSampling::ContinuousLinear,
            }
        }
    };

    OrderedFilterStepObservation {
        source_bounds: logical_rect_values(step.source_bounds.rect()),
        result_bounds: Some(logical_rect_values(step.result_bounds.rect())),
        source_device_origin,
        source_device_extent,
        result_device_origin,
        result_device_extent,
        edge,
        intent,
    }
}

#[cfg(test)]
fn logical_rect_values(rect: Rect) -> [f64; 4] {
    [rect.x(), rect.y(), rect.width(), rect.height()]
}

#[cfg(test)]
pub(crate) struct SpatialPrimitivesForTest {
    pub(crate) logical_and_device_phases_are_distinct: bool,
    pub(crate) logical_bounds: Option<[f64; 4]>,
    pub(crate) device_origin: Option<(i32, i32)>,
    pub(crate) device_extent: Option<(u32, u32)>,
    pub(crate) raster_scale: f64,
    pub(crate) texel_center: Option<(f64, f64)>,
    pub(crate) is_empty: bool,
}

#[cfg(test)]
pub(crate) fn transformed_logical_bounds_for_test(
    rect: Rect,
    transform: Transform,
) -> Result<[f64; 4]> {
    let transformed = LogicalBounds::try_from_rect(rect, "frame logical bounds")?
        .try_transform(transform, "frame transformed logical bounds")?
        .rect();
    Ok([
        transformed.x(),
        transformed.y(),
        transformed.width(),
        transformed.height(),
    ])
}

#[cfg(test)]
pub(crate) fn spatial_primitives_for_test(
    rect: Rect,
    transform: Transform,
    surface_scale: f64,
    texel: (u32, u32),
) -> Result<SpatialPrimitivesForTest> {
    let logical_bounds = LogicalBounds::try_from_rect(rect, "frame logical bounds")?;
    let plan =
        FrameContext::try_new(surface_scale)?.plan_local_bounds(logical_bounds, transform)?;
    let logical_rect = logical_bounds.rect();
    let logical_bounds = Some([
        logical_rect.x(),
        logical_rect.y(),
        logical_rect.width(),
        logical_rect.height(),
    ]);

    match plan {
        FrameSpatialPlan::Empty(plan) => {
            let _logical_bounds = plan.logical_bounds;
            Ok(SpatialPrimitivesForTest {
                logical_and_device_phases_are_distinct: true,
                logical_bounds,
                device_origin: None,
                device_extent: None,
                raster_scale: 0.0,
                texel_center: None,
                is_empty: true,
            })
        }
        FrameSpatialPlan::NonEmpty(plan) => {
            let _logical_bounds = plan.logical_bounds;
            let texel_center = plan.texel_center_mapping.point_for(texel.0, texel.1)?;
            Ok(SpatialPrimitivesForTest {
                logical_and_device_phases_are_distinct: true,
                logical_bounds,
                device_origin: Some((plan.device_origin.x, plan.device_origin.y)),
                device_extent: Some((plan.device_extent.width, plan.device_extent.height)),
                raster_scale: plan.raster_scale.get(),
                texel_center: Some((texel_center.x(), texel_center.y())),
                is_empty: false,
            })
        }
    }
}
