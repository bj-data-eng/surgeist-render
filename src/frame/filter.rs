use super::{
    FrameContext,
    bounds::{
        FrameSpatialPlan, LogicalBounds, NonEmptyFrameSpatialPlan, NonEmptyLogicalBounds,
        RasterScale, SemanticSourceBounds, SemanticSourceContribution, checked_div, checked_mul,
    },
};
use crate::{
    command::NormalizedLayer,
    error::{Error, Result},
    filter::{
        AlgorithmColorFilterRun, AlgorithmFilterPlan, AlgorithmFilterStep,
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
    },
    geometry::Transform,
    style::{FilterBlur, FilterDropShadow, FilterList},
};

impl FrameContext {
    pub(super) fn plan_filter_list(
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
                    let edge_policy = match source_role {
                        FilterSourceRole::Ordinary => FilterEdgePolicy::TransparentBlack,
                        FilterSourceRole::Backdrop => FilterEdgePolicy::SemanticBorderMirror {
                            semantic_border: initial_bounds,
                        },
                    };
                    (
                        result_bounds,
                        edge_policy,
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

impl SemanticSourceContribution {
    pub(super) fn include_backdrop_contribution(
        mut source_bounds: SemanticSourceBounds,
        current_parent: SemanticSourceBounds,
        layer: &mut NormalizedLayer,
        context: FrameContext,
        layer_to_surface: Transform,
    ) -> Result<SemanticSourceBounds> {
        let Some(backdrop) = layer.backdrop.as_deref() else {
            return Ok(source_bounds);
        };
        let algorithm_filter_plan = AlgorithmFilterPlan::from_filter_list(backdrop.filters());
        let capture_bounds = LogicalBounds::try_from_rect(
            backdrop.capture_bounds().rect(),
            "backdrop capture bounds",
        )?;
        let resolved_filter_plan = context.plan_filter_list(
            capture_bounds,
            layer_to_surface,
            backdrop.filters(),
            FilterSourceRole::Backdrop,
        )?;
        let mut backdrop_contribution = match resolved_filter_plan {
            ResolvedFrameFilterPlan::Empty(_) => SemanticSourceBounds::exactly_empty(),
            ResolvedFrameFilterPlan::NonEmpty(plan) => {
                SemanticSourceBounds::exact_known(plan.final_bounds)
            }
        };
        if let Some(clip) = backdrop.clip() {
            let clip_bounds = SemanticSourceBounds::try_for_clip(clip)?;
            backdrop_contribution = backdrop_contribution
                .try_intersect(clip_bounds, "post-filter backdrop clip intersection")?;
        }
        let captured_parent = current_parent.try_intersect(
            SemanticSourceBounds::from_logical_bounds(capture_bounds),
            "backdrop current-parent intersection",
        )?;
        if captured_parent.is_exactly_empty()
            || algorithm_filter_plan.output_is_always_transparent()
            || backdrop_contribution.is_exactly_empty()
        {
            layer.backdrop = None;
        } else {
            source_bounds = source_bounds.try_union(backdrop_contribution)?;
        }
        Ok(source_bounds)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C06 T3 stages ordinary and backdrop plans for C06 T5-T6 consumers."
    )
)]
pub(super) enum FilterSourceRole {
    Ordinary,
    Backdrop,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ResolvedFrameFilterPlan {
    Empty(EmptyResolvedFilterPlan),
    NonEmpty(NonEmptyResolvedFilterPlan),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EmptyResolvedFilterPlan {
    pub(super) source_bounds: LogicalBounds,
    pub(super) authored_operation_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct NonEmptyResolvedFilterPlan {
    pub(super) initial_bounds: NonEmptyLogicalBounds,
    pub(super) final_bounds: NonEmptyLogicalBounds,
    pub(super) authored_operation_count: usize,
    pub(super) steps: Vec<ResolvedFilterStep>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ResolvedFilterStep {
    pub(super) source_bounds: NonEmptyLogicalBounds,
    pub(super) result_bounds: NonEmptyLogicalBounds,
    pub(super) spatial_mapping: ResolvedFilterSpatialMapping,
    pub(super) edge_policy: FilterEdgePolicy,
    pub(super) operation_intent: ResolvedFilterOperationIntent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ResolvedFilterSpatialMapping {
    pub(super) source: NonEmptyFrameSpatialPlan,
    pub(super) result: NonEmptyFrameSpatialPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum FilterEdgePolicy {
    NoSampling,
    TransparentBlack,
    SemanticBorderMirror {
        semantic_border: NonEmptyLogicalBounds,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ResolvedFilterOperationIntent {
    ColorRun(AlgorithmColorFilterRun),
    Blur(ResolvedBlurIntent),
    DropShadow(ResolvedDropShadowIntent),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ResolvedBlurIntent {
    pub(super) authored_blur: FilterBlur,
    pub(super) support: InclusiveFilterKernelSupport,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ResolvedDropShadowIntent {
    pub(super) authored_shadow: FilterDropShadow,
    pub(super) alpha_source: DropShadowAlphaSource,
    pub(super) support: InclusiveFilterKernelSupport,
    pub(super) offset_sampling: DropShadowOffsetSampling,
    pub(super) source_composition: DropShadowSourceComposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DropShadowAlphaSource {
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DropShadowOffsetSampling {
    ContinuousLinear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DropShadowSourceComposition {
    RetainUnchangedForSourceOver,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct InclusiveFilterKernelSupport {
    pub(super) device_radius: u32,
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
