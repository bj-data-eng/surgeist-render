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
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use super::{command::LayerPassPlan, filter::ColorClampBoundary, style::ColorFilterOp};

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

static NEXT_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);

type GraphBuildResult<T> = std::result::Result<T, GraphValidationError>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GraphGeneration(u64);

impl GraphGeneration {
    fn try_next() -> GraphBuildResult<Self> {
        NEXT_GRAPH_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .map(Self)
            .map_err(|_| GraphValidationError::GenerationExhausted)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ResourceIndex(u32);

impl ResourceIndex {
    fn try_from_len(len: usize) -> GraphBuildResult<Self> {
        u32::try_from(len)
            .map(Self)
            .map_err(|_| GraphValidationError::ResourceIdentityExhausted)
    }

    fn as_usize(self) -> GraphBuildResult<usize> {
        usize::try_from(self.0).map_err(|_| GraphValidationError::UnknownResourceIndex)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PassIndex(u32);

impl PassIndex {
    fn try_from_len(len: usize) -> GraphBuildResult<Self> {
        u32::try_from(len)
            .map(Self)
            .map_err(|_| GraphValidationError::PassIdentityExhausted)
    }

    fn as_usize(self) -> GraphBuildResult<usize> {
        usize::try_from(self.0).map_err(|_| GraphValidationError::UnknownPassIndex)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SemanticResourceId {
    generation: GraphGeneration,
    index: ResourceIndex,
}

impl SemanticResourceId {
    const fn new(generation: GraphGeneration, index: ResourceIndex) -> Self {
        Self { generation, index }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SemanticPassId {
    generation: GraphGeneration,
    index: PassIndex,
}

impl SemanticPassId {
    const fn new(generation: GraphGeneration, index: PassIndex) -> Self {
        Self { generation, index }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages semantic resource roles for C06 T5-T6 graph planning."
    )
)]
enum SemanticResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    IsolationWorkingImage,
    ImportedImage,
    BackdropCopy,
    FilterIntermediate,
    ShadowImage,
    CompositeResult,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SemanticResourceDescriptor {
    role: SemanticResourceRole,
    logical_bounds: NonEmptyLogicalBounds,
    device_origin: SignedDeviceOrigin,
    device_extent: PositiveDeviceExtent,
    texel_center_mapping: TexelCenterMapping,
    expected_reads: u32,
}

impl SemanticResourceDescriptor {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C06 T4 resource descriptors are constructed by the staged C06 planner tests."
        )
    )]
    const fn new(
        role: SemanticResourceRole,
        spatial: NonEmptyFrameSpatialPlan,
        expected_reads: u32,
    ) -> Self {
        Self {
            role,
            logical_bounds: spatial.logical_bounds,
            device_origin: spatial.device_origin,
            device_extent: spatial.device_extent,
            texel_center_mapping: spatial.texel_center_mapping,
            expected_reads,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages root and transparent initialization intents for C06 T5-T6."
    )
)]
enum WorkingImageInitialization {
    SurfaceBaseColor(super::paint::Color),
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages RGBA and SourceAlpha blur intent for C06 T5-T6."
    )
)]
enum BlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages the finite semantic pass vocabulary for C06 T5-T6."
    )
)]
enum SemanticPassIntent {
    ClearRoot {
        initialization: WorkingImageInitialization,
    },
    VelloCapture {
        initialization: WorkingImageInitialization,
    },
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter,
    BlurHorizontal {
        input: BlurInput,
    },
    BlurVertical {
        input: BlurInput,
    },
    DropShadowColorize,
    Composite,
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages empty and resource-bearing semantic results for C06 T5-T6."
    )
)]
enum SemanticPassResult {
    Empty,
    Resource(SemanticResourceId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticResourceProducer {
    Imported,
    Pass(SemanticPassId),
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticGraphResource {
    id: SemanticResourceId,
    descriptor: SemanticResourceDescriptor,
    producer: Option<SemanticResourceProducer>,
    recorded_reads: u32,
    remaining_reads: Option<u32>,
    releasable_after: Option<SemanticPassId>,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticGraphPass {
    id: SemanticPassId,
    intent: SemanticPassIntent,
    dependencies: Vec<SemanticPassId>,
    reads: Vec<SemanticResourceId>,
    result: SemanticPassResult,
    scheduled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphBuildPhase {
    RecordingConsumers,
    Scheduling,
    FinalPresentScheduled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 typed graph failures are consumed by the staged C06 planner tests."
    )
)]
enum GraphValidationError {
    GenerationExhausted,
    ResourceIdentityExhausted,
    PassIdentityExhausted,
    UnknownResourceIndex,
    UnknownPassIndex,
    WrongResourceGeneration {
        expected: GraphGeneration,
        actual: GraphGeneration,
    },
    WrongPassGeneration {
        expected: GraphGeneration,
        actual: GraphGeneration,
    },
    UnknownResource(SemanticResourceId),
    UnknownPass(SemanticPassId),
    ReleasedResource(SemanticResourceId),
    ForwardDependency(SemanticPassId),
    ForwardRead(SemanticResourceId),
    ReadWriteAlias(SemanticResourceId),
    DuplicateProducer(SemanticResourceId),
    DuplicateDependency(SemanticPassId),
    DuplicateRead(SemanticResourceId),
    MissingProducerDependency {
        resource: SemanticResourceId,
        producer: SemanticPassId,
    },
    ReadCountOverflow(SemanticResourceId),
    DeclaredReadCountMismatch {
        resource: SemanticResourceId,
        declared: u32,
        recorded: u32,
    },
    ResourceWithoutProducer(SemanticResourceId),
    OrphanResult(SemanticResourceId),
    MissingRootWorkingImage,
    DuplicateRootWorkingImage,
    MissingFinalPresent,
    DuplicateFinalPresent,
    DeclarationAfterFinalPresent,
    MissingSurfaceBaseInitialization,
    RepeatedSurfaceBaseInitialization,
    RootMustUseSurfaceBase,
    NonTransparentCaptureBase,
    InvalidClearRootResult,
    InvalidCaptureResult,
    InvalidImportedResourceRole,
    InvalidPassArity,
    InvalidPassResultRole,
    InvalidPresentIntent,
    RootProducedByNonClearPass,
    ConsumersNotSealed,
    ConsumersAlreadySealed,
    PassAlreadyScheduled(SemanticPassId),
    PresentScheduledBeforeOtherPasses(SemanticPassId),
    SchedulingAfterFinalPresent,
    UnscheduledDependency {
        pass: SemanticPassId,
        dependency: SemanticPassId,
    },
    UnscheduledProducer {
        resource: SemanticResourceId,
        producer: SemanticPassId,
    },
    UnscheduledPass(SemanticPassId),
    UnscheduledReads {
        resource: SemanticResourceId,
        remaining: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages the immutable validated graph consumed by C06 T5-T6 and C07."
    )
)]
struct GpuRenderGraph {
    generation: GraphGeneration,
    resources: Vec<SemanticGraphResource>,
    passes: Vec<SemanticGraphPass>,
    root_working_image: SemanticResourceId,
    final_present: SemanticPassId,
}

#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages the private graph builder that C06 T5-T6 will invoke."
    )
)]
struct SemanticGraphBuilder {
    generation: GraphGeneration,
    phase: GraphBuildPhase,
    resources: Vec<SemanticGraphResource>,
    passes: Vec<SemanticGraphPass>,
    root_working_image: Option<SemanticResourceId>,
    final_present: Option<SemanticPassId>,
    surface_base_initializations: u32,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T4 stages the private graph builder that C06 T5-T6 will invoke."
    )
)]
impl SemanticGraphBuilder {
    fn try_new() -> GraphBuildResult<Self> {
        Ok(Self {
            generation: GraphGeneration::try_next()?,
            phase: GraphBuildPhase::RecordingConsumers,
            resources: Vec::new(),
            passes: Vec::new(),
            root_working_image: None,
            final_present: None,
            surface_base_initializations: 0,
        })
    }

    fn declare_resource(
        &mut self,
        descriptor: SemanticResourceDescriptor,
    ) -> GraphBuildResult<SemanticResourceId> {
        self.insert_resource(descriptor, None)
    }

    fn import_resource(
        &mut self,
        descriptor: SemanticResourceDescriptor,
    ) -> GraphBuildResult<SemanticResourceId> {
        if descriptor.role != SemanticResourceRole::ImportedImage {
            return Err(GraphValidationError::InvalidImportedResourceRole);
        }
        self.insert_resource(descriptor, Some(SemanticResourceProducer::Imported))
    }

    fn insert_resource(
        &mut self,
        descriptor: SemanticResourceDescriptor,
        producer: Option<SemanticResourceProducer>,
    ) -> GraphBuildResult<SemanticResourceId> {
        self.require_recording_phase()?;
        if self.final_present.is_some() {
            return Err(GraphValidationError::DeclarationAfterFinalPresent);
        }
        if descriptor.role == SemanticResourceRole::RootWorkingImage
            && self.root_working_image.is_some()
        {
            return Err(GraphValidationError::DuplicateRootWorkingImage);
        }

        let id = SemanticResourceId::new(
            self.generation,
            ResourceIndex::try_from_len(self.resources.len())?,
        );
        self.resources.push(SemanticGraphResource {
            id,
            descriptor,
            producer,
            recorded_reads: 0,
            remaining_reads: None,
            releasable_after: None,
        });
        if descriptor.role == SemanticResourceRole::RootWorkingImage {
            self.root_working_image = Some(id);
        }
        Ok(id)
    }

    fn declare_pass(
        &mut self,
        intent: SemanticPassIntent,
        dependencies: Vec<SemanticPassId>,
        reads: Vec<SemanticResourceId>,
        result: SemanticPassResult,
    ) -> GraphBuildResult<SemanticPassId> {
        self.require_recording_phase()?;
        if self.final_present.is_some() {
            return if intent == SemanticPassIntent::Present {
                Err(GraphValidationError::DuplicateFinalPresent)
            } else {
                Err(GraphValidationError::DeclarationAfterFinalPresent)
            };
        }
        let id = SemanticPassId::new(self.generation, PassIndex::try_from_len(self.passes.len())?);

        let mut seen_dependencies = Vec::with_capacity(dependencies.len());
        for dependency in &dependencies {
            if seen_dependencies.contains(dependency) {
                return Err(GraphValidationError::DuplicateDependency(*dependency));
            }
            self.validate_recorded_dependency(*dependency)?;
            seen_dependencies.push(*dependency);
        }

        let mut read_indices = Vec::with_capacity(reads.len());
        let mut seen_reads = Vec::with_capacity(reads.len());
        for resource in &reads {
            if seen_reads.contains(resource) {
                return Err(GraphValidationError::DuplicateRead(*resource));
            }
            let resource_index = self.validate_resource_id(*resource)?;
            let graph_resource = self
                .resources
                .get(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*resource))?;
            match graph_resource.producer {
                Some(SemanticResourceProducer::Imported) => {}
                Some(SemanticResourceProducer::Pass(producer)) => {
                    if !dependencies.contains(&producer) {
                        return Err(GraphValidationError::MissingProducerDependency {
                            resource: *resource,
                            producer,
                        });
                    }
                }
                None => return Err(GraphValidationError::ForwardRead(*resource)),
            }
            graph_resource
                .recorded_reads
                .checked_add(1)
                .ok_or(GraphValidationError::ReadCountOverflow(*resource))?;
            seen_reads.push(*resource);
            read_indices.push(resource_index);
        }

        let result_index = match result {
            SemanticPassResult::Empty => None,
            SemanticPassResult::Resource(resource) => {
                let resource_index = self.validate_resource_id(resource)?;
                if reads.contains(&resource) {
                    return Err(GraphValidationError::ReadWriteAlias(resource));
                }
                if self
                    .resources
                    .get(resource_index)
                    .and_then(|resource| resource.producer)
                    .is_some()
                {
                    return Err(GraphValidationError::DuplicateProducer(resource));
                }
                Some(resource_index)
            }
        };

        self.validate_pass_shape(intent, &dependencies, &reads, result)?;

        let initializes_surface = matches!(
            intent,
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::SurfaceBaseColor(_)
            }
        );
        let is_present = intent == SemanticPassIntent::Present;
        let mut resources = self.resources.clone();
        if let Some(resource_index) = result_index {
            let resource = resources.get_mut(resource_index).ok_or(match result {
                SemanticPassResult::Resource(resource) => {
                    GraphValidationError::UnknownResource(resource)
                }
                SemanticPassResult::Empty => GraphValidationError::UnknownResourceIndex,
            })?;
            resource.producer = Some(SemanticResourceProducer::Pass(id));
        }
        for (resource, resource_index) in reads.iter().zip(read_indices) {
            let graph_resource = resources
                .get_mut(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*resource))?;
            graph_resource.recorded_reads = graph_resource
                .recorded_reads
                .checked_add(1)
                .ok_or(GraphValidationError::ReadCountOverflow(*resource))?;
        }

        self.resources = resources;
        self.passes.push(SemanticGraphPass {
            id,
            intent,
            dependencies,
            reads,
            result,
            scheduled: false,
        });
        if initializes_surface {
            self.surface_base_initializations = 1;
        }
        if is_present {
            self.final_present = Some(id);
        }
        Ok(id)
    }

    fn validate_pass_shape(
        &self,
        intent: SemanticPassIntent,
        dependencies: &[SemanticPassId],
        reads: &[SemanticResourceId],
        result: SemanticPassResult,
    ) -> GraphBuildResult<()> {
        match intent {
            SemanticPassIntent::ClearRoot { initialization } => {
                if !matches!(
                    initialization,
                    WorkingImageInitialization::SurfaceBaseColor(_)
                ) {
                    return Err(GraphValidationError::RootMustUseSurfaceBase);
                }
                if self.surface_base_initializations != 0 {
                    return Err(GraphValidationError::RepeatedSurfaceBaseInitialization);
                }
                if !dependencies.is_empty() || !reads.is_empty() {
                    return Err(GraphValidationError::InvalidClearRootResult);
                }
                let SemanticPassResult::Resource(resource) = result else {
                    return Err(GraphValidationError::InvalidClearRootResult);
                };
                let resource_index = self.validate_resource_id(resource)?;
                if self.resources.get(resource_index).is_none_or(|resource| {
                    resource.descriptor.role != SemanticResourceRole::RootWorkingImage
                }) {
                    return Err(GraphValidationError::InvalidClearRootResult);
                }
            }
            SemanticPassIntent::VelloCapture { initialization } => {
                if initialization != WorkingImageInitialization::Transparent {
                    return Err(GraphValidationError::NonTransparentCaptureBase);
                }
                if !reads.is_empty() {
                    return Err(GraphValidationError::InvalidCaptureResult);
                }
                if let SemanticPassResult::Resource(resource) = result {
                    let resource_index = self.validate_resource_id(resource)?;
                    if self.resources.get(resource_index).is_none_or(|resource| {
                        !matches!(
                            resource.descriptor.role,
                            SemanticResourceRole::CaptureWorkingImage
                                | SemanticResourceRole::IsolationWorkingImage
                        )
                    }) {
                        return Err(GraphValidationError::InvalidCaptureResult);
                    }
                }
            }
            SemanticPassIntent::Present => {
                if self.final_present.is_some() {
                    return Err(GraphValidationError::DuplicateFinalPresent);
                }
                if reads.len() != 1 || result != SemanticPassResult::Empty {
                    return Err(GraphValidationError::InvalidPresentIntent);
                }
            }
            SemanticPassIntent::CanonicalizeCapture
            | SemanticPassIntent::CopyBackdrop
            | SemanticPassIntent::ColorFilter
            | SemanticPassIntent::BlurHorizontal { .. }
            | SemanticPassIntent::BlurVertical { .. }
            | SemanticPassIntent::DropShadowColorize
            | SemanticPassIntent::Composite => {
                let SemanticPassResult::Resource(resource) = result else {
                    if reads.is_empty() {
                        return Ok(());
                    }
                    return Err(GraphValidationError::InvalidPassArity);
                };
                let reads_are_valid = if intent == SemanticPassIntent::Composite {
                    reads.len() >= 2
                } else {
                    reads.len() == 1
                };
                if !reads_are_valid {
                    return Err(GraphValidationError::InvalidPassArity);
                }
                let expected_role = match intent {
                    SemanticPassIntent::CopyBackdrop => SemanticResourceRole::BackdropCopy,
                    SemanticPassIntent::DropShadowColorize => SemanticResourceRole::ShadowImage,
                    SemanticPassIntent::Composite => SemanticResourceRole::CompositeResult,
                    SemanticPassIntent::CanonicalizeCapture
                    | SemanticPassIntent::ColorFilter
                    | SemanticPassIntent::BlurHorizontal { .. }
                    | SemanticPassIntent::BlurVertical { .. } => {
                        SemanticResourceRole::FilterIntermediate
                    }
                    SemanticPassIntent::ClearRoot { .. }
                    | SemanticPassIntent::VelloCapture { .. }
                    | SemanticPassIntent::Present => {
                        return Err(GraphValidationError::InvalidPassResultRole);
                    }
                };
                let resource_index = self.validate_resource_id(resource)?;
                if self
                    .resources
                    .get(resource_index)
                    .is_none_or(|resource| resource.descriptor.role != expected_role)
                {
                    return Err(GraphValidationError::InvalidPassResultRole);
                }
            }
        }
        Ok(())
    }

    fn begin_scheduling(&mut self) -> GraphBuildResult<()> {
        self.require_recording_phase()?;
        let root = self
            .root_working_image
            .ok_or(GraphValidationError::MissingRootWorkingImage)?;
        self.final_present
            .ok_or(GraphValidationError::MissingFinalPresent)?;
        if self.surface_base_initializations == 0 {
            return Err(GraphValidationError::MissingSurfaceBaseInitialization);
        }
        if self.surface_base_initializations != 1 {
            return Err(GraphValidationError::RepeatedSurfaceBaseInitialization);
        }

        for resource in &self.resources {
            let producer = resource
                .producer
                .ok_or(GraphValidationError::ResourceWithoutProducer(resource.id))?;
            if resource.id == root
                && !matches!(producer, SemanticResourceProducer::Pass(pass) if self
                    .passes
                    .get(pass.index.as_usize()?)
                    .is_some_and(|pass| matches!(pass.intent, SemanticPassIntent::ClearRoot { .. })))
            {
                return Err(GraphValidationError::RootProducedByNonClearPass);
            }
            if resource.descriptor.expected_reads != resource.recorded_reads {
                return Err(GraphValidationError::DeclaredReadCountMismatch {
                    resource: resource.id,
                    declared: resource.descriptor.expected_reads,
                    recorded: resource.recorded_reads,
                });
            }
            if resource.recorded_reads == 0 {
                return Err(GraphValidationError::OrphanResult(resource.id));
            }
        }

        for pass in &self.passes {
            if let SemanticPassIntent::VelloCapture { initialization } = pass.intent
                && initialization != WorkingImageInitialization::Transparent
            {
                return Err(GraphValidationError::NonTransparentCaptureBase);
            }
        }

        for resource in &mut self.resources {
            resource.remaining_reads = Some(resource.descriptor.expected_reads);
        }
        self.phase = GraphBuildPhase::Scheduling;
        Ok(())
    }

    fn schedule_pass(&mut self, id: SemanticPassId) -> GraphBuildResult<()> {
        match self.phase {
            GraphBuildPhase::RecordingConsumers => {
                return Err(GraphValidationError::ConsumersNotSealed);
            }
            GraphBuildPhase::Scheduling => {}
            GraphBuildPhase::FinalPresentScheduled => {
                return Err(GraphValidationError::SchedulingAfterFinalPresent);
            }
        }
        let pass_index = self.validate_existing_pass_id(id)?;
        let pass = self
            .passes
            .get(pass_index)
            .ok_or(GraphValidationError::UnknownPass(id))?;
        if pass.scheduled {
            return Err(GraphValidationError::PassAlreadyScheduled(id));
        }
        let is_present = pass.intent == SemanticPassIntent::Present;
        if is_present {
            if let Some(unscheduled) = self
                .passes
                .iter()
                .find(|candidate| candidate.id != id && !candidate.scheduled)
            {
                return Err(GraphValidationError::PresentScheduledBeforeOtherPasses(
                    unscheduled.id,
                ));
            }
            for resource in &self.resources {
                let required_by_present = u32::from(pass.reads.contains(&resource.id));
                match resource.remaining_reads {
                    Some(remaining) if remaining == required_by_present => {}
                    Some(remaining) => {
                        return Err(GraphValidationError::UnscheduledReads {
                            resource: resource.id,
                            remaining,
                        });
                    }
                    None => return Err(GraphValidationError::ConsumersNotSealed),
                }
            }
        }
        for dependency in &pass.dependencies {
            let dependency_index = self.validate_existing_pass_id(*dependency)?;
            if self
                .passes
                .get(dependency_index)
                .is_none_or(|dependency| !dependency.scheduled)
            {
                return Err(GraphValidationError::UnscheduledDependency {
                    pass: id,
                    dependency: *dependency,
                });
            }
        }

        let mut read_indices = Vec::with_capacity(pass.reads.len());
        for resource in &pass.reads {
            let resource_index = self.validate_resource_id(*resource)?;
            let graph_resource = self
                .resources
                .get(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*resource))?;
            match graph_resource.producer {
                Some(SemanticResourceProducer::Imported) => {}
                Some(SemanticResourceProducer::Pass(producer)) => {
                    let producer_index = self.validate_existing_pass_id(producer)?;
                    if self
                        .passes
                        .get(producer_index)
                        .is_none_or(|producer| !producer.scheduled)
                    {
                        return Err(GraphValidationError::UnscheduledProducer {
                            resource: *resource,
                            producer,
                        });
                    }
                }
                None => return Err(GraphValidationError::ForwardRead(*resource)),
            }
            match graph_resource.remaining_reads {
                Some(0) => return Err(GraphValidationError::ReleasedResource(*resource)),
                Some(_) => {}
                None => return Err(GraphValidationError::ConsumersNotSealed),
            }
            read_indices.push(resource_index);
        }

        let mut resources = self.resources.clone();
        for (resource, resource_index) in pass.reads.iter().zip(read_indices) {
            let graph_resource = resources
                .get_mut(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*resource))?;
            let remaining = graph_resource
                .remaining_reads
                .and_then(|remaining| remaining.checked_sub(1))
                .ok_or(GraphValidationError::ReleasedResource(*resource))?;
            graph_resource.remaining_reads = Some(remaining);
            if remaining == 0 {
                graph_resource.releasable_after = Some(id);
            }
        }
        let mut passes = self.passes.clone();
        let scheduled_pass = passes
            .get_mut(pass_index)
            .ok_or(GraphValidationError::UnknownPass(id))?;
        scheduled_pass.scheduled = true;
        self.resources = resources;
        self.passes = passes;
        if is_present {
            self.phase = GraphBuildPhase::FinalPresentScheduled;
        }
        Ok(())
    }

    fn ensure_resource_readable(&self, resource: SemanticResourceId) -> GraphBuildResult<()> {
        if self.phase == GraphBuildPhase::RecordingConsumers {
            return Err(GraphValidationError::ConsumersNotSealed);
        }
        let resource_index = self.validate_resource_id(resource)?;
        match self
            .resources
            .get(resource_index)
            .ok_or(GraphValidationError::UnknownResource(resource))?
            .remaining_reads
        {
            Some(0) => Err(GraphValidationError::ReleasedResource(resource)),
            Some(_) => Ok(()),
            None => Err(GraphValidationError::ConsumersNotSealed),
        }
    }

    fn finish(self) -> GraphBuildResult<GpuRenderGraph> {
        if self.phase == GraphBuildPhase::RecordingConsumers {
            return Err(GraphValidationError::ConsumersNotSealed);
        }
        for pass in &self.passes {
            if !pass.scheduled {
                return Err(GraphValidationError::UnscheduledPass(pass.id));
            }
        }
        for resource in &self.resources {
            match resource.remaining_reads {
                Some(0) if resource.releasable_after.is_some() => {}
                Some(remaining) => {
                    return Err(GraphValidationError::UnscheduledReads {
                        resource: resource.id,
                        remaining,
                    });
                }
                None => return Err(GraphValidationError::ConsumersNotSealed),
            }
        }
        let root_working_image = self
            .root_working_image
            .ok_or(GraphValidationError::MissingRootWorkingImage)?;
        let final_present = self
            .final_present
            .ok_or(GraphValidationError::MissingFinalPresent)?;
        if self.phase != GraphBuildPhase::FinalPresentScheduled {
            return Err(GraphValidationError::UnscheduledPass(final_present));
        }
        Ok(GpuRenderGraph {
            generation: self.generation,
            resources: self.resources,
            passes: self.passes,
            root_working_image,
            final_present,
        })
    }

    fn require_recording_phase(&self) -> GraphBuildResult<()> {
        match self.phase {
            GraphBuildPhase::RecordingConsumers => Ok(()),
            GraphBuildPhase::Scheduling | GraphBuildPhase::FinalPresentScheduled => {
                Err(GraphValidationError::ConsumersAlreadySealed)
            }
        }
    }

    fn validate_resource_id(&self, id: SemanticResourceId) -> GraphBuildResult<usize> {
        if id.generation != self.generation {
            return Err(GraphValidationError::WrongResourceGeneration {
                expected: self.generation,
                actual: id.generation,
            });
        }
        let index = id.index.as_usize()?;
        if self.resources.get(index).is_none() {
            return Err(GraphValidationError::UnknownResource(id));
        }
        Ok(index)
    }

    fn validate_recorded_dependency(&self, id: SemanticPassId) -> GraphBuildResult<usize> {
        if id.generation != self.generation {
            return Err(GraphValidationError::WrongPassGeneration {
                expected: self.generation,
                actual: id.generation,
            });
        }
        let index = id.index.as_usize()?;
        if index == self.passes.len() {
            return Err(GraphValidationError::ForwardDependency(id));
        }
        if self.passes.get(index).is_none() {
            return Err(GraphValidationError::UnknownPass(id));
        }
        Ok(index)
    }

    fn validate_existing_pass_id(&self, id: SemanticPassId) -> GraphBuildResult<usize> {
        if id.generation != self.generation {
            return Err(GraphValidationError::WrongPassGeneration {
                expected: self.generation,
                actual: id.generation,
            });
        }
        let index = id.index.as_usize()?;
        if self.passes.get(index).is_none() {
            return Err(GraphValidationError::UnknownPass(id));
        }
        Ok(index)
    }
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InvalidSemanticGraphStateForTest {
    StaleResourceIdentity,
    StalePassIdentity,
    UnknownResourceIdentity,
    UnknownPassIdentity,
    ReleasedResourceIdentity,
    ForwardDependency,
    ForwardRead,
    ReadWriteAlias,
    DuplicateProducer,
    DeclaredReadCountMismatch,
    OrphanResult,
    MissingRootWorkingImage,
    DuplicateRootWorkingImage,
    MissingFinalPresent,
    DuplicateFinalPresent,
    NonTransparentCaptureBase,
    RepeatedSurfaceBaseInitialization,
    MissingProducerDependency,
    ScheduleBeforeConsumersAreSealed,
    DeclareConsumerAfterConsumersAreSealed,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphFailureObservation {
    WrongResourceGeneration,
    WrongPassGeneration,
    UnknownResource,
    UnknownPass,
    ReleasedResource,
    ForwardDependency,
    ForwardRead,
    ReadWriteAlias,
    DuplicateProducer,
    DeclaredReadCountMismatch,
    OrphanResult,
    MissingRootWorkingImage,
    DuplicateRootWorkingImage,
    MissingFinalPresent,
    DuplicateFinalPresent,
    NonTransparentCaptureBase,
    RepeatedSurfaceBaseInitialization,
    MissingProducerDependency,
    ConsumersNotSealed,
    ConsumersAlreadySealed,
    DeclarationAfterFinalPresent,
    PresentScheduledBeforeOtherPasses,
    SchedulingAfterFinalPresent,
    OtherTypedFailure,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphOwnerCallObservation {
    Accepted,
    Rejected(GraphFailureObservation),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FinalPresentDeclarationObservation {
    pub(crate) declaration_after_present: GraphOwnerCallObservation,
    pub(crate) completed_after_declaration_attempt: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FinalPresentSchedulingObservation {
    pub(crate) early_present: GraphOwnerCallObservation,
    pub(crate) completed_after_early_present_attempt: bool,
    pub(crate) scheduling_after_present: GraphOwnerCallObservation,
    pub(crate) completed_after_post_present_attempt: bool,
}

#[cfg(test)]
impl From<GraphValidationError> for GraphFailureObservation {
    fn from(error: GraphValidationError) -> Self {
        match error {
            GraphValidationError::WrongResourceGeneration { .. } => Self::WrongResourceGeneration,
            GraphValidationError::WrongPassGeneration { .. } => Self::WrongPassGeneration,
            GraphValidationError::UnknownResource(_)
            | GraphValidationError::UnknownResourceIndex => Self::UnknownResource,
            GraphValidationError::UnknownPass(_) | GraphValidationError::UnknownPassIndex => {
                Self::UnknownPass
            }
            GraphValidationError::ReleasedResource(_) => Self::ReleasedResource,
            GraphValidationError::ForwardDependency(_) => Self::ForwardDependency,
            GraphValidationError::ForwardRead(_) => Self::ForwardRead,
            GraphValidationError::ReadWriteAlias(_) => Self::ReadWriteAlias,
            GraphValidationError::DuplicateProducer(_) => Self::DuplicateProducer,
            GraphValidationError::DeclaredReadCountMismatch { .. } => {
                Self::DeclaredReadCountMismatch
            }
            GraphValidationError::OrphanResult(_) => Self::OrphanResult,
            GraphValidationError::MissingRootWorkingImage => Self::MissingRootWorkingImage,
            GraphValidationError::DuplicateRootWorkingImage => Self::DuplicateRootWorkingImage,
            GraphValidationError::MissingFinalPresent => Self::MissingFinalPresent,
            GraphValidationError::DuplicateFinalPresent => Self::DuplicateFinalPresent,
            GraphValidationError::NonTransparentCaptureBase => Self::NonTransparentCaptureBase,
            GraphValidationError::RepeatedSurfaceBaseInitialization => {
                Self::RepeatedSurfaceBaseInitialization
            }
            GraphValidationError::MissingProducerDependency { .. } => {
                Self::MissingProducerDependency
            }
            GraphValidationError::ConsumersNotSealed => Self::ConsumersNotSealed,
            GraphValidationError::ConsumersAlreadySealed => Self::ConsumersAlreadySealed,
            GraphValidationError::DeclarationAfterFinalPresent => {
                Self::DeclarationAfterFinalPresent
            }
            GraphValidationError::PresentScheduledBeforeOtherPasses(_) => {
                Self::PresentScheduledBeforeOtherPasses
            }
            GraphValidationError::SchedulingAfterFinalPresent => Self::SchedulingAfterFinalPresent,
            GraphValidationError::GenerationExhausted
            | GraphValidationError::ResourceIdentityExhausted
            | GraphValidationError::PassIdentityExhausted
            | GraphValidationError::DuplicateDependency(_)
            | GraphValidationError::DuplicateRead(_)
            | GraphValidationError::ReadCountOverflow(_)
            | GraphValidationError::ResourceWithoutProducer(_)
            | GraphValidationError::MissingSurfaceBaseInitialization
            | GraphValidationError::RootMustUseSurfaceBase
            | GraphValidationError::InvalidClearRootResult
            | GraphValidationError::InvalidCaptureResult
            | GraphValidationError::InvalidImportedResourceRole
            | GraphValidationError::InvalidPassArity
            | GraphValidationError::InvalidPassResultRole
            | GraphValidationError::InvalidPresentIntent
            | GraphValidationError::RootProducedByNonClearPass
            | GraphValidationError::PassAlreadyScheduled(_)
            | GraphValidationError::UnscheduledDependency { .. }
            | GraphValidationError::UnscheduledProducer { .. }
            | GraphValidationError::UnscheduledPass(_)
            | GraphValidationError::UnscheduledReads { .. } => Self::OtherTypedFailure,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SemanticGraphProbeError {
    SpatialSetup,
    UnexpectedGraphFailure(GraphFailureObservation),
    ExpectedRejectionMissing,
    ObservationUnavailable,
}

#[cfg(test)]
type SemanticGraphProbeResult<T> = std::result::Result<T, SemanticGraphProbeError>;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphEdgeLifetimeObservation {
    pub(crate) observes_bounded_offscreen_pass: bool,
    pub(crate) source_expected_reads: u32,
    pub(crate) remaining_before_first_consumer: u32,
    pub(crate) remaining_after_alpha_consumer: u32,
    pub(crate) remaining_before_source_over: u32,
    pub(crate) remaining_after_source_over: u32,
    pub(crate) released_after_source_over: bool,
    pub(crate) post_release_read_rejected: bool,
    pub(crate) every_result_has_one_owner: bool,
    pub(crate) every_read_names_its_producer: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphBaseInitializationObservation {
    pub(crate) observes_bounded_offscreen_pass: bool,
    pub(crate) root_working_images: usize,
    pub(crate) final_present_intents: usize,
    pub(crate) surface_base_initializations: usize,
    pub(crate) surface_base_color: Option<super::paint::Color>,
    pub(crate) isolation_working_images: usize,
    pub(crate) captures_are_transparent: bool,
    pub(crate) empty_results_have_no_descriptor: bool,
    pub(crate) resource_descriptors_are_spatially_complete: bool,
}

#[cfg(test)]
fn graph_probe_descriptor(
    role: SemanticResourceRole,
    expected_reads: u32,
) -> SemanticGraphProbeResult<SemanticResourceDescriptor> {
    let logical_bounds = LogicalBounds::try_from_rect(
        Rect::new(-2.0, 3.0, 8.0, 6.0),
        "semantic graph probe bounds",
    )
    .map_err(|_| SemanticGraphProbeError::SpatialSetup)?;
    let spatial = FrameContext::try_new(2.0)
        .and_then(|context| context.plan_local_bounds(logical_bounds, Transform::identity()))
        .map_err(|_| SemanticGraphProbeError::SpatialSetup)?;
    let FrameSpatialPlan::NonEmpty(spatial) = spatial else {
        return Err(SemanticGraphProbeError::SpatialSetup);
    };
    Ok(SemanticResourceDescriptor::new(
        role,
        spatial,
        expected_reads,
    ))
}

#[cfg(test)]
fn graph_probe_builder() -> SemanticGraphProbeResult<SemanticGraphBuilder> {
    SemanticGraphBuilder::try_new().map_err(|error| {
        SemanticGraphProbeError::UnexpectedGraphFailure(GraphFailureObservation::from(error))
    })
}

#[cfg(test)]
fn graph_probe_value<T>(result: GraphBuildResult<T>) -> SemanticGraphProbeResult<T> {
    result.map_err(|error| {
        SemanticGraphProbeError::UnexpectedGraphFailure(GraphFailureObservation::from(error))
    })
}

#[cfg(test)]
fn graph_probe_failure<T>(
    result: GraphBuildResult<T>,
) -> SemanticGraphProbeResult<GraphFailureObservation> {
    match result {
        Ok(_) => Err(SemanticGraphProbeError::ExpectedRejectionMissing),
        Err(error) => Ok(GraphFailureObservation::from(error)),
    }
}

#[cfg(test)]
fn graph_owner_call_observation<T>(result: GraphBuildResult<T>) -> GraphOwnerCallObservation {
    match result {
        Ok(_) => GraphOwnerCallObservation::Accepted,
        Err(error) => GraphOwnerCallObservation::Rejected(GraphFailureObservation::from(error)),
    }
}

#[cfg(test)]
fn declare_probe_root(
    builder: &mut SemanticGraphBuilder,
    expected_reads: u32,
) -> SemanticGraphProbeResult<(SemanticResourceId, SemanticPassId)> {
    let root = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::RootWorkingImage,
        expected_reads,
    )?))?;
    let clear = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::ClearRoot {
            initialization: WorkingImageInitialization::SurfaceBaseColor(
                super::paint::Color::BLACK,
            ),
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Resource(root),
    ))?;
    Ok((root, clear))
}

#[cfg(test)]
fn declare_probe_capture(
    builder: &mut SemanticGraphBuilder,
    role: SemanticResourceRole,
    expected_reads: u32,
) -> SemanticGraphProbeResult<(SemanticResourceId, SemanticPassId)> {
    let resource =
        graph_probe_value(builder.declare_resource(graph_probe_descriptor(role, expected_reads)?))?;
    let pass = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Resource(resource),
    ))?;
    Ok((resource, pass))
}

#[cfg(test)]
fn declare_probe_present(
    builder: &mut SemanticGraphBuilder,
    resource: SemanticResourceId,
    producer: SemanticPassId,
) -> SemanticGraphProbeResult<SemanticPassId> {
    graph_probe_value(builder.declare_pass(
        SemanticPassIntent::Present,
        vec![producer],
        vec![resource],
        SemanticPassResult::Empty,
    ))
}

#[cfg(test)]
pub(crate) fn final_present_declaration_observation_for_test()
-> SemanticGraphProbeResult<FinalPresentDeclarationObservation> {
    let mut builder = graph_probe_builder()?;
    let (root, clear) = declare_probe_root(&mut builder, 1)?;
    let present = declare_probe_present(&mut builder, root, clear)?;

    let declaration = builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Empty,
    );
    let declared_pass = declaration.as_ref().ok().copied();
    let declaration_after_present = graph_owner_call_observation(declaration);
    graph_probe_value(builder.begin_scheduling())?;
    graph_probe_value(builder.schedule_pass(clear))?;
    if let Some(declared_pass) = declared_pass {
        graph_probe_value(builder.schedule_pass(declared_pass))?;
    }
    graph_probe_value(builder.schedule_pass(present))?;
    let completed_after_declaration_attempt = builder.finish().is_ok();

    Ok(FinalPresentDeclarationObservation {
        declaration_after_present,
        completed_after_declaration_attempt,
    })
}

#[cfg(test)]
pub(crate) fn final_present_scheduling_observation_for_test()
-> SemanticGraphProbeResult<FinalPresentSchedulingObservation> {
    let mut early_builder = graph_probe_builder()?;
    let (early_root, early_clear) = declare_probe_root(&mut early_builder, 1)?;
    let early_independent = graph_probe_value(early_builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Empty,
    ))?;
    let early_present_id = declare_probe_present(&mut early_builder, early_root, early_clear)?;
    graph_probe_value(early_builder.begin_scheduling())?;
    graph_probe_value(early_builder.schedule_pass(early_clear))?;
    let early_present = graph_owner_call_observation(early_builder.schedule_pass(early_present_id));
    graph_probe_value(early_builder.schedule_pass(early_independent))?;
    if matches!(early_present, GraphOwnerCallObservation::Rejected(_)) {
        graph_probe_value(early_builder.schedule_pass(early_present_id))?;
    }
    let completed_after_early_present_attempt = early_builder.finish().is_ok();

    let mut terminal_builder = graph_probe_builder()?;
    let (terminal_root, terminal_clear) = declare_probe_root(&mut terminal_builder, 1)?;
    let terminal_independent = graph_probe_value(terminal_builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Empty,
    ))?;
    let terminal_present =
        declare_probe_present(&mut terminal_builder, terminal_root, terminal_clear)?;
    graph_probe_value(terminal_builder.begin_scheduling())?;
    graph_probe_value(terminal_builder.schedule_pass(terminal_clear))?;
    graph_probe_value(terminal_builder.schedule_pass(terminal_independent))?;
    graph_probe_value(terminal_builder.schedule_pass(terminal_present))?;
    let scheduling_after_present =
        graph_owner_call_observation(terminal_builder.schedule_pass(terminal_clear));
    let completed_after_post_present_attempt = terminal_builder.finish().is_ok();

    Ok(FinalPresentSchedulingObservation {
        early_present,
        completed_after_early_present_attempt,
        scheduling_after_present,
        completed_after_post_present_attempt,
    })
}

#[cfg(test)]
pub(crate) fn invalid_semantic_graph_state_for_test(
    state: InvalidSemanticGraphStateForTest,
) -> SemanticGraphProbeResult<GraphFailureObservation> {
    match state {
        InvalidSemanticGraphStateForTest::StaleResourceIdentity => {
            let mut first = graph_probe_builder()?;
            let stale = graph_probe_value(first.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::FilterIntermediate,
                1,
            )?))?;
            let mut second = graph_probe_builder()?;
            let result = graph_probe_value(second.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::FilterIntermediate,
                1,
            )?))?;
            graph_probe_failure(second.declare_pass(
                SemanticPassIntent::ColorFilter,
                Vec::new(),
                vec![stale],
                SemanticPassResult::Resource(result),
            ))
        }
        InvalidSemanticGraphStateForTest::StalePassIdentity => {
            let mut first = graph_probe_builder()?;
            let (_, stale) = declare_probe_root(&mut first, 1)?;
            let mut second = graph_probe_builder()?;
            graph_probe_failure(second.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent,
                },
                vec![stale],
                Vec::new(),
                SemanticPassResult::Empty,
            ))
        }
        InvalidSemanticGraphStateForTest::UnknownResourceIdentity => {
            let mut builder = graph_probe_builder()?;
            let unknown = SemanticResourceId::new(builder.generation, ResourceIndex(7));
            let result = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::FilterIntermediate,
                1,
            )?))?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::ColorFilter,
                Vec::new(),
                vec![unknown],
                SemanticPassResult::Resource(result),
            ))
        }
        InvalidSemanticGraphStateForTest::UnknownPassIdentity => {
            let mut builder = graph_probe_builder()?;
            let unknown = SemanticPassId::new(builder.generation, PassIndex(1));
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent,
                },
                vec![unknown],
                Vec::new(),
                SemanticPassResult::Empty,
            ))
        }
        InvalidSemanticGraphStateForTest::ReleasedResourceIdentity => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 1)?;
            let present = declare_probe_present(&mut builder, root, clear)?;
            graph_probe_value(builder.begin_scheduling())?;
            graph_probe_value(builder.schedule_pass(clear))?;
            graph_probe_value(builder.schedule_pass(present))?;
            graph_probe_failure(builder.ensure_resource_readable(root))
        }
        InvalidSemanticGraphStateForTest::ForwardDependency => {
            let mut builder = graph_probe_builder()?;
            let future = SemanticPassId::new(builder.generation, PassIndex(0));
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent,
                },
                vec![future],
                Vec::new(),
                SemanticPassResult::Empty,
            ))
        }
        InvalidSemanticGraphStateForTest::ForwardRead => {
            let mut builder = graph_probe_builder()?;
            let source = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::FilterIntermediate,
                1,
            )?))?;
            let result = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::FilterIntermediate,
                1,
            )?))?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::ColorFilter,
                Vec::new(),
                vec![source],
                SemanticPassResult::Resource(result),
            ))
        }
        InvalidSemanticGraphStateForTest::ReadWriteAlias => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 1)?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::ColorFilter,
                vec![clear],
                vec![root],
                SemanticPassResult::Resource(root),
            ))
        }
        InvalidSemanticGraphStateForTest::DuplicateProducer => {
            let mut builder = graph_probe_builder()?;
            let (capture, _) =
                declare_probe_capture(&mut builder, SemanticResourceRole::CaptureWorkingImage, 1)?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::ColorFilter,
                Vec::new(),
                Vec::new(),
                SemanticPassResult::Resource(capture),
            ))
        }
        InvalidSemanticGraphStateForTest::DeclaredReadCountMismatch => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 2)?;
            declare_probe_present(&mut builder, root, clear)?;
            graph_probe_failure(builder.begin_scheduling())
        }
        InvalidSemanticGraphStateForTest::OrphanResult => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 1)?;
            declare_probe_capture(&mut builder, SemanticResourceRole::CaptureWorkingImage, 0)?;
            declare_probe_present(&mut builder, root, clear)?;
            graph_probe_failure(builder.begin_scheduling())
        }
        InvalidSemanticGraphStateForTest::MissingRootWorkingImage => {
            let mut builder = graph_probe_builder()?;
            let (capture, producer) =
                declare_probe_capture(&mut builder, SemanticResourceRole::CaptureWorkingImage, 1)?;
            declare_probe_present(&mut builder, capture, producer)?;
            graph_probe_failure(builder.begin_scheduling())
        }
        InvalidSemanticGraphStateForTest::DuplicateRootWorkingImage => {
            let mut builder = graph_probe_builder()?;
            graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::RootWorkingImage,
                1,
            )?))?;
            graph_probe_failure(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::RootWorkingImage,
                1,
            )?))
        }
        InvalidSemanticGraphStateForTest::MissingFinalPresent => {
            let mut builder = graph_probe_builder()?;
            declare_probe_root(&mut builder, 0)?;
            graph_probe_failure(builder.begin_scheduling())
        }
        InvalidSemanticGraphStateForTest::DuplicateFinalPresent => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 2)?;
            declare_probe_present(&mut builder, root, clear)?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::Present,
                vec![clear],
                vec![root],
                SemanticPassResult::Empty,
            ))
        }
        InvalidSemanticGraphStateForTest::NonTransparentCaptureBase => {
            let mut builder = graph_probe_builder()?;
            let capture = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::IsolationWorkingImage,
                1,
            )?))?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::SurfaceBaseColor(
                        super::paint::Color::BLACK,
                    ),
                },
                Vec::new(),
                Vec::new(),
                SemanticPassResult::Resource(capture),
            ))
        }
        InvalidSemanticGraphStateForTest::RepeatedSurfaceBaseInitialization => {
            let mut builder = graph_probe_builder()?;
            declare_probe_root(&mut builder, 1)?;
            let isolation = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::IsolationWorkingImage,
                1,
            )?))?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::ClearRoot {
                    initialization: WorkingImageInitialization::SurfaceBaseColor(
                        super::paint::Color::BLACK,
                    ),
                },
                Vec::new(),
                Vec::new(),
                SemanticPassResult::Resource(isolation),
            ))
        }
        InvalidSemanticGraphStateForTest::MissingProducerDependency => {
            let mut builder = graph_probe_builder()?;
            let (root, _) = declare_probe_root(&mut builder, 1)?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::Present,
                Vec::new(),
                vec![root],
                SemanticPassResult::Empty,
            ))
        }
        InvalidSemanticGraphStateForTest::ScheduleBeforeConsumersAreSealed => {
            let mut builder = graph_probe_builder()?;
            let (_, clear) = declare_probe_root(&mut builder, 1)?;
            graph_probe_failure(builder.schedule_pass(clear))
        }
        InvalidSemanticGraphStateForTest::DeclareConsumerAfterConsumersAreSealed => {
            let mut builder = graph_probe_builder()?;
            let (root, clear) = declare_probe_root(&mut builder, 1)?;
            declare_probe_present(&mut builder, root, clear)?;
            graph_probe_value(builder.begin_scheduling())?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent,
                },
                Vec::new(),
                Vec::new(),
                SemanticPassResult::Empty,
            ))
        }
    }
}

#[cfg(test)]
struct CompletedDropShadowGraphProbe {
    graph: GpuRenderGraph,
    source: SemanticResourceId,
    source_over: SemanticPassId,
    source_expected_reads: u32,
    remaining_before_first_consumer: u32,
    remaining_after_alpha_consumer: u32,
    remaining_before_source_over: u32,
    remaining_after_source_over: u32,
    post_release_read_rejected: bool,
}

#[cfg(test)]
fn remaining_reads_for_probe(
    builder: &SemanticGraphBuilder,
    resource: SemanticResourceId,
) -> SemanticGraphProbeResult<u32> {
    let resource_index = graph_probe_value(builder.validate_resource_id(resource))?;
    builder
        .resources
        .get(resource_index)
        .and_then(|resource| resource.remaining_reads)
        .ok_or(SemanticGraphProbeError::ObservationUnavailable)
}

#[cfg(test)]
fn build_drop_shadow_graph_probe() -> SemanticGraphProbeResult<CompletedDropShadowGraphProbe> {
    let mut builder = graph_probe_builder()?;
    let (root, clear_root) = declare_probe_root(&mut builder, 1)?;
    graph_probe_value(builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Empty,
    ))?;
    let (source, capture) =
        declare_probe_capture(&mut builder, SemanticResourceRole::IsolationWorkingImage, 2)?;
    let (canonical_source, canonical_capture) =
        declare_probe_capture(&mut builder, SemanticResourceRole::CaptureWorkingImage, 1)?;
    let canonical = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::FilterIntermediate,
        1,
    )?))?;
    let canonicalize = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::CanonicalizeCapture,
        vec![canonical_capture],
        vec![canonical_source],
        SemanticPassResult::Resource(canonical),
    ))?;
    let backdrop = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::BackdropCopy,
        1,
    )?))?;
    let copy_backdrop = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::CopyBackdrop,
        vec![canonicalize],
        vec![canonical],
        SemanticPassResult::Resource(backdrop),
    ))?;
    let rgba_blurred = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::FilterIntermediate,
        1,
    )?))?;
    let rgba_blur = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::BlurHorizontal {
            input: BlurInput::Rgba,
        },
        vec![copy_backdrop],
        vec![backdrop],
        SemanticPassResult::Resource(rgba_blurred),
    ))?;
    let imported = graph_probe_value(builder.import_resource(graph_probe_descriptor(
        SemanticResourceRole::ImportedImage,
        1,
    )?))?;
    let horizontal = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::FilterIntermediate,
        1,
    )?))?;
    let blur_horizontal = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::BlurHorizontal {
            input: BlurInput::SourceAlpha,
        },
        vec![capture],
        vec![source],
        SemanticPassResult::Resource(horizontal),
    ))?;
    let vertical = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::FilterIntermediate,
        1,
    )?))?;
    let blur_vertical = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::BlurVertical {
            input: BlurInput::SourceAlpha,
        },
        vec![blur_horizontal],
        vec![horizontal],
        SemanticPassResult::Resource(vertical),
    ))?;
    let shadow = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::ShadowImage,
        1,
    )?))?;
    let colorize = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::DropShadowColorize,
        vec![blur_vertical],
        vec![vertical],
        SemanticPassResult::Resource(shadow),
    ))?;
    let composed = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
        SemanticResourceRole::CompositeResult,
        1,
    )?))?;
    let source_over = graph_probe_value(builder.declare_pass(
        SemanticPassIntent::Composite,
        vec![clear_root, capture, colorize, rgba_blur],
        vec![root, source, shadow, imported, rgba_blurred],
        SemanticPassResult::Resource(composed),
    ))?;
    let present = declare_probe_present(&mut builder, composed, source_over)?;

    graph_probe_value(builder.begin_scheduling())?;
    let source_expected_reads = builder
        .resources
        .get(graph_probe_value(builder.validate_resource_id(source))?)
        .map(|resource| resource.descriptor.expected_reads)
        .ok_or(SemanticGraphProbeError::ObservationUnavailable)?;
    graph_probe_value(builder.schedule_pass(clear_root))?;
    let empty_capture = builder
        .passes
        .iter()
        .find(|pass| {
            matches!(pass.intent, SemanticPassIntent::VelloCapture { .. })
                && pass.result == SemanticPassResult::Empty
        })
        .map(|pass| pass.id)
        .ok_or(SemanticGraphProbeError::ObservationUnavailable)?;
    graph_probe_value(builder.schedule_pass(empty_capture))?;
    graph_probe_value(builder.schedule_pass(capture))?;
    graph_probe_value(builder.schedule_pass(canonical_capture))?;
    graph_probe_value(builder.schedule_pass(canonicalize))?;
    graph_probe_value(builder.schedule_pass(copy_backdrop))?;
    graph_probe_value(builder.schedule_pass(rgba_blur))?;
    let remaining_before_first_consumer = remaining_reads_for_probe(&builder, source)?;
    graph_probe_value(builder.schedule_pass(blur_horizontal))?;
    let remaining_after_alpha_consumer = remaining_reads_for_probe(&builder, source)?;
    graph_probe_value(builder.schedule_pass(blur_vertical))?;
    graph_probe_value(builder.schedule_pass(colorize))?;
    let remaining_before_source_over = remaining_reads_for_probe(&builder, source)?;
    graph_probe_value(builder.schedule_pass(source_over))?;
    let remaining_after_source_over = remaining_reads_for_probe(&builder, source)?;
    let post_release_read_rejected = matches!(
        builder.ensure_resource_readable(source),
        Err(GraphValidationError::ReleasedResource(released)) if released == source
    );
    graph_probe_value(builder.schedule_pass(present))?;
    let graph = graph_probe_value(builder.finish())?;

    Ok(CompletedDropShadowGraphProbe {
        graph,
        source,
        source_over,
        source_expected_reads,
        remaining_before_first_consumer,
        remaining_after_alpha_consumer,
        remaining_before_source_over,
        remaining_after_source_over,
        post_release_read_rejected,
    })
}

#[cfg(test)]
pub(crate) fn semantic_graph_edge_lifetime_observation_for_test(
    pass_plan: LayerPassPlan,
) -> SemanticGraphProbeResult<GraphEdgeLifetimeObservation> {
    let probe = build_drop_shadow_graph_probe()?;
    let source_index = ResourceIndex::as_usize(probe.source.index)
        .map_err(|_| SemanticGraphProbeError::ObservationUnavailable)?;
    let released_after_source_over = probe
        .graph
        .resources
        .get(source_index)
        .is_some_and(|resource| resource.releasable_after == Some(probe.source_over));
    let every_result_has_one_owner =
        probe
            .graph
            .resources
            .iter()
            .all(|resource| match resource.producer {
                Some(SemanticResourceProducer::Imported) => {
                    resource.id.generation == probe.graph.generation
                }
                Some(SemanticResourceProducer::Pass(producer)) => {
                    resource.id.generation == probe.graph.generation
                        && producer.generation == probe.graph.generation
                        && probe
                            .graph
                            .passes
                            .iter()
                            .filter(|pass| pass.result == SemanticPassResult::Resource(resource.id))
                            .count()
                            == 1
                        && probe.graph.passes.iter().any(|pass| pass.id == producer)
                }
                None => false,
            });
    let every_read_names_its_producer = probe.graph.passes.iter().all(|pass| {
        pass.reads.iter().all(|read| {
            let Ok(index) = read.index.as_usize() else {
                return false;
            };
            probe
                .graph
                .resources
                .get(index)
                .is_some_and(|resource| match resource.producer {
                    Some(SemanticResourceProducer::Imported) => true,
                    Some(SemanticResourceProducer::Pass(producer)) => {
                        pass.dependencies.contains(&producer)
                    }
                    None => false,
                })
        })
    });

    Ok(GraphEdgeLifetimeObservation {
        observes_bounded_offscreen_pass: pass_plan.requires_offscreen_texture()
            && pass_plan.bounds().is_some(),
        source_expected_reads: probe.source_expected_reads,
        remaining_before_first_consumer: probe.remaining_before_first_consumer,
        remaining_after_alpha_consumer: probe.remaining_after_alpha_consumer,
        remaining_before_source_over: probe.remaining_before_source_over,
        remaining_after_source_over: probe.remaining_after_source_over,
        released_after_source_over,
        post_release_read_rejected: probe.post_release_read_rejected,
        every_result_has_one_owner,
        every_read_names_its_producer,
    })
}

#[cfg(test)]
pub(crate) fn semantic_graph_base_initialization_observation_for_test(
    pass_plan: LayerPassPlan,
) -> SemanticGraphProbeResult<GraphBaseInitializationObservation> {
    let probe = build_drop_shadow_graph_probe()?;
    let root_working_images = probe
        .graph
        .resources
        .iter()
        .filter(|resource| {
            resource.descriptor.role == SemanticResourceRole::RootWorkingImage
                && resource.id == probe.graph.root_working_image
        })
        .count();
    let final_present_intents = probe
        .graph
        .passes
        .iter()
        .filter(|pass| {
            pass.intent == SemanticPassIntent::Present && pass.id == probe.graph.final_present
        })
        .count();
    let surface_base_initializations = probe
        .graph
        .passes
        .iter()
        .filter(|pass| {
            matches!(
                pass.intent,
                SemanticPassIntent::ClearRoot {
                    initialization: WorkingImageInitialization::SurfaceBaseColor(_)
                }
            )
        })
        .count();
    let surface_base_color = probe
        .graph
        .passes
        .iter()
        .find_map(|pass| match pass.intent {
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::SurfaceBaseColor(color),
            } => Some(color),
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::Transparent,
            }
            | SemanticPassIntent::VelloCapture { .. }
            | SemanticPassIntent::CanonicalizeCapture
            | SemanticPassIntent::CopyBackdrop
            | SemanticPassIntent::ColorFilter
            | SemanticPassIntent::BlurHorizontal { .. }
            | SemanticPassIntent::BlurVertical { .. }
            | SemanticPassIntent::DropShadowColorize
            | SemanticPassIntent::Composite
            | SemanticPassIntent::Present => None,
        });
    let isolation_working_images = probe
        .graph
        .resources
        .iter()
        .filter(|resource| resource.descriptor.role == SemanticResourceRole::IsolationWorkingImage)
        .count();
    let captures_are_transparent = probe.graph.passes.iter().all(|pass| {
        !matches!(pass.intent, SemanticPassIntent::VelloCapture { .. })
            || matches!(
                pass.intent,
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent
                }
            )
    });
    let pass_resource_results = probe
        .graph
        .passes
        .iter()
        .filter(|pass| matches!(pass.result, SemanticPassResult::Resource(_)))
        .count();
    let imported_resources = probe
        .graph
        .resources
        .iter()
        .filter(|resource| resource.producer == Some(SemanticResourceProducer::Imported))
        .count();
    let empty_result_count = probe
        .graph
        .passes
        .iter()
        .filter(|pass| pass.result == SemanticPassResult::Empty)
        .count();
    let empty_results_have_no_descriptor = empty_result_count > 1
        && probe.graph.resources.len() == pass_resource_results + imported_resources;
    let resource_descriptors_are_spatially_complete =
        probe.graph.resources.iter().all(|resource| {
            let logical = resource.descriptor.logical_bounds.rect();
            let mapped_first_texel = resource
                .descriptor
                .texel_center_mapping
                .point_for(0, 0)
                .ok();
            logical == Rect::new(-2.0, 3.0, 8.0, 6.0)
                && resource.descriptor.device_origin == SignedDeviceOrigin::new(-4, 6)
                && resource.descriptor.device_extent
                    == PositiveDeviceExtent {
                        width: 16,
                        height: 12,
                    }
                && mapped_first_texel == Some(Point::new(-1.75, 3.25))
        });

    Ok(GraphBaseInitializationObservation {
        observes_bounded_offscreen_pass: pass_plan.requires_offscreen_texture()
            && pass_plan.bounds().is_some(),
        root_working_images,
        final_present_intents,
        surface_base_initializations,
        surface_base_color,
        isolation_working_images,
        captures_are_transparent,
        empty_results_have_no_descriptor,
        resource_descriptors_are_spatially_complete,
    })
}
