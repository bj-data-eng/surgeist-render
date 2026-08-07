use std::collections::{BTreeMap, BTreeSet};

use super::lower::{
    runtime_pass_cache_keys, runtime_resource_format, shader_binding_role, shader_sampling_edge,
};
use super::model::{
    LoweredGraphPlan, RuntimeBlur, RuntimeBlurAxis, RuntimeBlurInput, RuntimeClipCoverageElement,
    RuntimeColorClampBoundary, RuntimeColorFilter, RuntimeColorOperation,
    RuntimeColorOperationKind, RuntimeComposite, RuntimeCompositeKind,
    RuntimeDestinationToLayerLocal, RuntimeDropShadow, RuntimeFilterSpatialMapping,
    RuntimeInitialization, RuntimeLayerCompositeParameters, RuntimeMaskTexelCenterFacts,
    RuntimeOuterClip, RuntimePass, RuntimePassId, RuntimePassKind, RuntimeReadBinding,
    RuntimeReadRole, RuntimeResolvedAlphaMaskComposition, RuntimeResourceFormat, RuntimeResourceId,
    RuntimeResourceImport, RuntimeResourceProducer, RuntimeResourceRequest, RuntimeResourceRole,
    RuntimeResultBinding, RuntimeSamplingEdge, RuntimeSamplingFilter, RuntimeSpatialDescriptor,
    RuntimeVelloCapture, RuntimeVelloSpan, RuntimeVelloSpanScope,
    runtime_affine_is_finite_and_non_singular,
};
use crate::{
    BackendErrorCode, BlendMode, Color, Error, Format, PhysicalSize, Point, Result, Transform,
    command::RenderClip,
    renderer::Antialiasing,
    resource::WorkingFormat,
    shader::{SamplerKey, ShaderMaskQualityKey, ShaderMaskSamplingKey, ShaderSamplingFilterKey},
};

#[derive(Clone)]
pub(super) struct ExecutableLayerCompositionFacts {
    pub(super) pass: RuntimePassId,
    pub(super) parent: RuntimeResourceId,
    pub(super) source: RuntimeResourceId,
    pub(super) clip_coverage: Option<RuntimeResourceId>,
    pub(super) alpha_mask: Option<RuntimeResourceId>,
    pub(super) result: RuntimeResourceId,
    pub(super) composite: RuntimeComposite,
}

#[derive(Clone)]
pub(super) struct ExecutableColorFilterFacts {
    pub(super) pass: RuntimePassId,
    pub(super) source: RuntimeResourceId,
    pub(super) result: RuntimeResourceId,
    pub(super) filter: RuntimeColorFilter,
}

#[derive(Clone)]
pub(super) struct ExecutableBlurFacts {
    pub(super) horizontal: RuntimePassId,
    pub(super) vertical: RuntimePassId,
    pub(super) source: RuntimeResourceId,
    pub(super) intermediate: RuntimeResourceId,
    pub(super) result: RuntimeResourceId,
    pub(super) blur: RuntimeBlur,
}

#[derive(Clone)]
pub(super) struct ExecutableDropShadowFacts {
    pub(super) horizontal: RuntimePassId,
    pub(super) vertical: RuntimePassId,
    pub(super) colorize: RuntimePassId,
    pub(super) merge: RuntimePassId,
    pub(super) source: RuntimeResourceId,
    pub(super) horizontal_result: RuntimeResourceId,
    pub(super) vertical_result: RuntimeResourceId,
    pub(super) shadow: RuntimeResourceId,
    pub(super) result: RuntimeResourceId,
    pub(super) blur: RuntimeBlur,
    pub(super) parameters: RuntimeDropShadow,
}

#[derive(Clone)]
pub(super) struct ExecutableBackdropFacts {
    pub(super) copy: RuntimePassId,
    pub(super) completed_parent: RuntimeResourceId,
    pub(super) copied: RuntimeResourceId,
    pub(super) foreground: Option<RuntimeResourceId>,
    pub(super) filter_steps: Vec<ExecutableFilterStepFacts>,
    pub(super) filtered: RuntimeResourceId,
    pub(super) group_clear: RuntimePassId,
    pub(super) backdrop_composite: RuntimePassId,
    pub(super) foreground_composite: Option<RuntimePassId>,
    pub(super) outer_composite: RuntimePassId,
    pub(super) completed_group: RuntimeResourceId,
    pub(super) result: RuntimeResourceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExecutableFilterStepFacts {
    Color(RuntimePassId),
    Blur {
        horizontal: RuntimePassId,
        vertical: RuntimePassId,
    },
    DropShadow {
        horizontal: RuntimePassId,
        vertical: RuntimePassId,
        colorize: RuntimePassId,
        merge: RuntimePassId,
    },
}

#[derive(Clone)]
pub(super) struct ClosedExecutableGraphFacts {
    pub(super) working_format: WorkingFormat,
    pub(super) output_format: Format,
    pub(super) captures: Vec<ExecutableVelloCaptureFacts>,
    pub(super) layer_compositions: Vec<ExecutableLayerCompositionFacts>,
    pub(super) color_filters: Vec<ExecutableColorFilterFacts>,
    pub(super) blurs: Vec<ExecutableBlurFacts>,
    pub(super) drop_shadows: Vec<ExecutableDropShadowFacts>,
    pub(super) filter_steps: Vec<ExecutableFilterStepFacts>,
    pub(super) backdrops: Vec<ExecutableBackdropFacts>,
}

#[derive(Clone, Copy)]
struct ExecutableCompositionContext {
    current: RuntimeResourceId,
    producer: RuntimePassId,
    contains_captured_source: bool,
}

#[must_use = "a closed executable graph must reach dispatch or explicit rejection"]
pub(super) struct ClosedExecutableGraph {
    pub(super) lowered: LoweredGraphPlan,
    pub(super) facts: ClosedExecutableGraphFacts,
}

impl ClosedExecutableGraph {
    pub(super) fn try_from_lowered(
        lowered: LoweredGraphPlan,
    ) -> std::result::Result<Self, LoweredGraphPlan> {
        let Some(facts) = lowered.closed_executable_graph_facts() else {
            return Err(lowered);
        };
        if !facts.proves_exact_facts_for(&lowered) {
            return Err(lowered);
        }
        Ok(Self { lowered, facts })
    }

    pub(super) fn has_layer_composition(&self) -> bool {
        !self.facts.layer_compositions.is_empty()
    }

    fn has_color_filters(&self) -> bool {
        !self.facts.color_filters.is_empty()
    }

    fn has_spatial_filters(&self) -> bool {
        !self.facts.blurs.is_empty() || !self.facts.drop_shadows.is_empty()
    }

    fn has_backdrops(&self) -> bool {
        !self.facts.backdrops.is_empty()
    }
}

impl ClosedExecutableGraphFacts {
    pub(super) fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        if self.working_format != plan.working_format
            || self.output_format != plan.output_format
            || self.captures.is_empty()
        {
            return false;
        }
        let captures_are_exact = self.captures.iter().all(|capture| {
            plan.passes.iter().any(|pass| {
                pass.id == capture.pass()
                    && matches!(
                        &pass.kind,
                        RuntimePassKind::VelloCapture(Some(work))
                            if work == capture.work()
                                && work.antialiasing() == capture.antialiasing()
                    )
                    && pass.result == RuntimeResultBinding::Resource(capture.target())
            })
        });
        let layers_are_exact = self.layer_compositions.iter().all(|layer| {
            let Some(pass) = plan.passes.iter().find(|pass| pass.id == layer.pass) else {
                return false;
            };
            let RuntimePassKind::Composite(Some(composite)) = &pass.kind else {
                return false;
            };
            let mut expected_reads = vec![layer.parent, layer.source];
            if let Some(coverage) = layer.clip_coverage {
                expected_reads.push(coverage);
            }
            if let Some(mask) = layer.alpha_mask {
                expected_reads.push(mask);
            }
            composite == &layer.composite
                && pass
                    .reads
                    .iter()
                    .map(|read| read.resource)
                    .eq(expected_reads)
                && pass.result == RuntimeResultBinding::Resource(layer.result)
        });
        let color_filters_are_exact = self.color_filters.iter().all(|color| {
            plan.passes.iter().any(|pass| {
                pass.id == color.pass
                    && matches!(
                        &pass.kind,
                        RuntimePassKind::ColorFilter(Some(filter)) if filter == &color.filter
                    )
                    && pass.reads.len() == 1
                    && pass.reads[0].resource == color.source
                    && pass.result == RuntimeResultBinding::Resource(color.result)
            })
        }) && self.color_filters.len()
            == plan
                .passes
                .iter()
                .filter(|pass| matches!(pass.kind, RuntimePassKind::ColorFilter(Some(_))))
                .count();
        let blurs_are_exact = self
            .blurs
            .iter()
            .all(|blur| blur.proves_exact_facts_for(plan))
            && self.blurs.len()
                == plan
                    .passes
                    .iter()
                    .filter(|pass| {
                        matches!(
                            pass.kind,
                            RuntimePassKind::BlurHorizontal(Some(RuntimeBlur {
                                input: RuntimeBlurInput::Rgba,
                                ..
                            }))
                        )
                    })
                    .count();
        let drop_shadows_are_exact = self
            .drop_shadows
            .iter()
            .all(|shadow| shadow.proves_exact_facts_for(plan))
            && self.drop_shadows.len()
                == plan
                    .passes
                    .iter()
                    .filter(|pass| {
                        matches!(pass.kind, RuntimePassKind::DropShadowColorize(Some(_)))
                    })
                    .count();
        let filter_order_is_exact =
            executable_filter_step_order(plan) == Some(self.filter_steps.clone());
        let backdrops_are_exact = self
            .backdrops
            .iter()
            .all(|backdrop| backdrop.proves_exact_facts_for(plan));
        captures_are_exact
            && layers_are_exact
            && color_filters_are_exact
            && blurs_are_exact
            && drop_shadows_are_exact
            && filter_order_is_exact
            && backdrops_are_exact
    }
}

impl ExecutableBackdropFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let pass = |id| plan.passes.iter().find(|candidate| candidate.id == id);
        let Some(copy) = pass(self.copy) else {
            return false;
        };
        let foreground_is_distinct = self
            .foreground
            .is_none_or(|foreground| foreground != self.copied && foreground != self.filtered);
        matches!(copy.kind, RuntimePassKind::CopyBackdrop)
            && copy.reads.len() == 1
            && copy.reads[0].resource == self.completed_parent
            && copy.result == RuntimeResultBinding::Resource(self.copied)
            && pass(self.group_clear).is_some_and(|clear| {
                matches!(
                    clear.kind,
                    RuntimePassKind::ClearRoot {
                        initialization: RuntimeInitialization::Transparent,
                        color,
                    } if color == Color::TRANSPARENT
                )
            })
            && pass(self.backdrop_composite).is_some()
            && self
                .foreground_composite
                .is_none_or(|id| pass(id).is_some())
            && pass(self.outer_composite).is_some()
            && self.completed_group != self.completed_parent
            && self.result != self.completed_group
            && foreground_is_distinct
    }
}

impl ExecutableBlurFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let Some(horizontal) = plan.passes.iter().find(|pass| pass.id == self.horizontal) else {
            return false;
        };
        let Some(vertical) = plan.passes.iter().find(|pass| pass.id == self.vertical) else {
            return false;
        };
        matches!(
            &horizontal.kind,
            RuntimePassKind::BlurHorizontal(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Horizontal)
        ) && matches!(
            &vertical.kind,
            RuntimePassKind::BlurVertical(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Vertical)
        ) && horizontal
            .reads
            .first()
            .is_some_and(|read| read.resource == self.source)
            && horizontal.result == RuntimeResultBinding::Resource(self.intermediate)
            && vertical
                .reads
                .first()
                .is_some_and(|read| read.resource == self.intermediate)
            && vertical.result == RuntimeResultBinding::Resource(self.result)
    }
}

impl ExecutableDropShadowFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        let pass = |id| plan.passes.iter().find(|pass| pass.id == id);
        let (Some(horizontal), Some(vertical), Some(colorize), Some(merge)) = (
            pass(self.horizontal),
            pass(self.vertical),
            pass(self.colorize),
            pass(self.merge),
        ) else {
            return false;
        };
        matches!(
            &horizontal.kind,
            RuntimePassKind::BlurHorizontal(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Horizontal)
        ) && matches!(
            &vertical.kind,
            RuntimePassKind::BlurVertical(Some(blur))
                if runtime_blur_matches_axis(blur, &self.blur, RuntimeBlurAxis::Vertical)
        ) && matches!(
            &colorize.kind,
            RuntimePassKind::DropShadowColorize(Some(parameters))
                if parameters == &self.parameters
        ) && matches!(
            &merge.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::DropShadow,
                ..
            }))
        ) && horizontal
            .reads
            .first()
            .is_some_and(|read| read.resource == self.source)
            && horizontal.result == RuntimeResultBinding::Resource(self.horizontal_result)
            && vertical.result == RuntimeResultBinding::Resource(self.vertical_result)
            && colorize.result == RuntimeResultBinding::Resource(self.shadow)
            && merge.result == RuntimeResultBinding::Resource(self.result)
    }
}

fn runtime_blur_matches_axis(
    candidate: &RuntimeBlur,
    expected: &RuntimeBlur,
    axis: RuntimeBlurAxis,
) -> bool {
    let expected_spatial = match axis {
        RuntimeBlurAxis::Horizontal => expected.spatial,
        RuntimeBlurAxis::Vertical => RuntimeFilterSpatialMapping {
            source: expected.spatial.result,
            result: expected.spatial.result,
        },
    };
    candidate.axis == axis
        && candidate.input == expected.input
        && candidate.standard_deviation == expected.standard_deviation
        && candidate.support_radius == expected.support_radius
        && candidate.kernel == expected.kernel
        && candidate.spatial == expected_spatial
        && candidate.edge == expected.edge
}

fn executable_filter_step_order(plan: &LoweredGraphPlan) -> Option<Vec<ExecutableFilterStepFacts>> {
    let mut steps = Vec::new();
    let mut cursor = 0_usize;
    while cursor < plan.passes.len() {
        let pass = &plan.passes[cursor];
        match &pass.kind {
            RuntimePassKind::ColorFilter(Some(_)) => {
                steps.push(ExecutableFilterStepFacts::Color(pass.id));
                cursor = cursor.checked_add(1)?;
            }
            RuntimePassKind::BlurHorizontal(Some(blur)) if blur.input == RuntimeBlurInput::Rgba => {
                let vertical = plan.passes.get(cursor.checked_add(1)?)?;
                steps.push(ExecutableFilterStepFacts::Blur {
                    horizontal: pass.id,
                    vertical: vertical.id,
                });
                cursor = cursor.checked_add(2)?;
            }
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.input == RuntimeBlurInput::SourceAlpha =>
            {
                let vertical = plan.passes.get(cursor.checked_add(1)?)?;
                let colorize = plan.passes.get(cursor.checked_add(2)?)?;
                let merge = plan.passes.get(cursor.checked_add(3)?)?;
                steps.push(ExecutableFilterStepFacts::DropShadow {
                    horizontal: pass.id,
                    vertical: vertical.id,
                    colorize: colorize.id,
                    merge: merge.id,
                });
                cursor = cursor.checked_add(4)?;
            }
            _ => cursor = cursor.checked_add(1)?,
        }
    }
    Some(steps)
}
#[must_use]
pub(crate) struct BaseExecutionFacts {
    pub(super) working_format: WorkingFormat,
    pub(super) output_format: Format,
    pub(super) captures: Vec<ExecutableVelloCaptureFacts>,
}

impl BaseExecutionFacts {
    #[must_use]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.working_format
    }

    #[must_use]
    pub(crate) const fn output_format(&self) -> Format {
        self.output_format
    }

    #[must_use]
    pub(crate) fn captures(&self) -> &[ExecutableVelloCaptureFacts] {
        &self.captures
    }

    pub(super) fn proves_exact_execution_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
        if self.working_format() != plan.working_format
            || self.output_format() != plan.output_format
        {
            return false;
        }
        let mut passes = BTreeSet::new();
        let mut targets = BTreeSet::new();
        self.captures().iter().all(|capture| {
            let Some(pass) = plan.passes.iter().find(|pass| pass.id == capture.pass()) else {
                return false;
            };
            let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) = &pass.kind
            else {
                return false;
            };
            let Some(capture_span) = capture.span() else {
                return false;
            };
            let Some(target) = plan
                .resources
                .iter()
                .find(|resource| resource.id == capture.target())
            else {
                return false;
            };
            passes.insert(capture.pass())
                && targets.insert(capture.target())
                && capture_span.scope == RuntimeVelloSpanScope::CurrentParent
                && capture_span == span
                && capture
                    .initial_transform()
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite())
                && capture.antialiasing() == span.antialiasing
                && capture.target_extent() == target.spatial.device_extent
                && capture.texel_origin() == target.spatial.texel_origin
                && capture.raster_scale() == target.spatial.raster_scale
        })
    }
}

#[must_use]
pub(crate) struct BasePreparableGraph {
    pub(super) lowered: LoweredGraphPlan,
    pub(super) execution: BaseExecutionFacts,
}

impl BasePreparableGraph {
    pub(super) fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if closed.has_layer_composition()
            || closed.has_color_filters()
            || closed.has_spatial_filters()
            || closed.has_backdrops()
            || closed.facts.captures.is_empty()
            || closed.facts.captures.iter().any(|capture| {
                capture
                    .span()
                    .is_none_or(|span| span.scope != RuntimeVelloSpanScope::CurrentParent)
            })
        {
            return Err(Box::new(closed));
        }
        let execution = BaseExecutionFacts {
            working_format: closed.facts.working_format,
            output_format: closed.facts.output_format,
            captures: closed.facts.captures.clone(),
        };
        if !execution.proves_exact_execution_facts_for(&closed.lowered) {
            return Err(Box::new(closed));
        }
        Ok(Self {
            lowered: closed.lowered,
            execution,
        })
    }

    pub(super) fn into_parts(self) -> (LoweredGraphPlan, BaseExecutionFacts) {
        (self.lowered, self.execution)
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.execution.working_format()
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.execution.output_format()
    }

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the base-graph root output resource is missing"))
    }
}

#[must_use]
pub(crate) struct CompositionPreparableGraph {
    closed: ClosedExecutableGraph,
}

impl CompositionPreparableGraph {
    pub(super) fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_layer_composition()
            || closed.has_color_filters()
            || closed.has_spatial_filters()
            || closed.has_backdrops()
        {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    pub(super) fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }
}

#[must_use]
pub(crate) struct ColorFilterPreparableGraph {
    pub(super) closed: ClosedExecutableGraph,
}

impl ColorFilterPreparableGraph {
    pub(super) fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_color_filters() || closed.has_spatial_filters() || closed.has_backdrops() {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    pub(super) fn proves_closed_color_facts(&self) -> bool {
        self.closed.has_color_filters()
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    pub(super) fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }
}

#[must_use]
pub(crate) struct SpatialFilterPreparableGraph {
    pub(super) closed: ClosedExecutableGraph,
}

impl SpatialFilterPreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if !closed.has_spatial_filters() || closed.has_backdrops() {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    pub(super) fn proves_closed_filter_facts(&self) -> bool {
        self.closed.has_spatial_filters()
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    pub(super) fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }
}

#[must_use]
pub(crate) struct BackdropPreparableGraph {
    pub(super) closed: ClosedExecutableGraph,
}

impl BackdropPreparableGraph {
    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        let [backdrop] = closed.facts.backdrops.as_slice() else {
            return Err(Box::new(closed));
        };
        if backdrop.filter_steps != closed.facts.filter_steps {
            return Err(Box::new(closed));
        }
        Ok(Self { closed })
    }

    pub(super) fn proves_closed_backdrop_facts(&self) -> bool {
        let [backdrop] = self.closed.facts.backdrops.as_slice() else {
            return false;
        };
        backdrop.filter_steps == self.closed.facts.filter_steps
            && backdrop.proves_exact_facts_for(&self.closed.lowered)
            && self
                .closed
                .facts
                .proves_exact_facts_for(&self.closed.lowered)
    }

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.closed.facts.working_format
    }

    pub(crate) const fn output_format(&self) -> Format {
        self.closed.facts.output_format
    }

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.closed
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == self.closed.lowered.root_working_image)
            .map(|resource| resource.spatial.device_extent)
            .ok_or_else(|| preparation_error("the backdrop root output resource is missing"))
    }

    pub(super) fn into_closed(self) -> ClosedExecutableGraph {
        self.closed
    }
}
#[derive(Clone)]
pub(crate) struct ExecutableVelloCaptureFacts {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: RuntimeVelloCapture,
    initial_transform: Transform,
    antialiasing: Antialiasing,
    target_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl ExecutableVelloCaptureFacts {
    #[must_use]
    pub(crate) const fn pass(&self) -> RuntimePassId {
        self.pass
    }

    #[must_use]
    pub(crate) const fn target(&self) -> RuntimeResourceId {
        self.target
    }

    #[must_use]
    pub(super) fn span(&self) -> Option<&RuntimeVelloSpan> {
        self.work.span()
    }

    #[must_use]
    pub(super) const fn work(&self) -> &RuntimeVelloCapture {
        &self.work
    }

    #[must_use]
    pub(crate) const fn initial_transform(&self) -> Transform {
        self.initial_transform
    }

    #[must_use]
    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }

    #[must_use]
    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target_extent
    }

    #[must_use]
    pub(crate) const fn texel_origin(&self) -> Point {
        self.texel_origin
    }

    #[must_use]
    pub(crate) const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }
}
struct ClosedGraphMaps<'plan> {
    resource_by_id: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    pass_positions: BTreeMap<RuntimePassId, usize>,
}

impl<'plan> ClosedGraphMaps<'plan> {
    fn try_new(plan: &'plan LoweredGraphPlan) -> Option<Self> {
        let resource_by_id = plan
            .resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();
        if resource_by_id.len() != plan.resources.len() {
            return None;
        }
        let resource_formats = plan
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let pass_positions = plan
            .passes
            .iter()
            .enumerate()
            .map(|(position, pass)| (pass.id, position))
            .collect::<BTreeMap<_, _>>();
        (pass_positions.len() == plan.passes.len()).then_some(Self {
            resource_by_id,
            resource_formats,
            pass_positions,
        })
    }
}

struct ClosedGraphAccounting {
    actual_reads: BTreeMap<RuntimeResourceId, u32>,
    actual_last_reads: BTreeMap<RuntimeResourceId, RuntimePassId>,
    releases: BTreeMap<RuntimeResourceId, RuntimePassId>,
    results: BTreeMap<RuntimeResourceId, RuntimePassId>,
}

fn validate_closed_graph_accounting(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
) -> Option<()> {
    let accounting = validate_closed_pass_accounting(plan, maps)?;
    for resource in &plan.resources {
        if resource.format != runtime_resource_format(resource.role, plan.working_format)
            || resource.expected_reads == 0
            || accounting.actual_reads.get(&resource.id).copied() != Some(resource.expected_reads)
            || accounting.actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
            || accounting.releases.get(&resource.id).copied() != Some(resource.last_use)
            || resource.spatial.device_extent.width() == 0
            || resource.spatial.device_extent.height() == 0
            || !resource.spatial.texel_origin.x().is_finite()
            || !resource.spatial.texel_origin.y().is_finite()
            || !resource.spatial.raster_scale.is_finite()
            || resource.spatial.raster_scale <= 0.0
        {
            return None;
        }
        match (&resource.producer, &resource.import) {
            (
                RuntimeResourceProducer::Imported,
                Some(RuntimeResourceImport::ResolvedAlphaMask(_)),
            ) if resource.role == RuntimeResourceRole::ImportedImage
                && resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm => {}
            (RuntimeResourceProducer::Pass(pass), None)
                if accounting.results.get(&resource.id).copied() == Some(*pass) => {}
            _ => return None,
        }
    }
    Some(())
}

fn validate_closed_pass_accounting(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
) -> Option<ClosedGraphAccounting> {
    let mut accounting = ClosedGraphAccounting {
        actual_reads: BTreeMap::new(),
        actual_last_reads: BTreeMap::new(),
        releases: BTreeMap::new(),
        results: BTreeMap::new(),
    };
    for (position, pass) in plan.passes.iter().enumerate() {
        let mut dependencies = BTreeSet::new();
        if pass.dependencies.iter().any(|dependency| {
            !dependencies.insert(*dependency)
                || maps
                    .pass_positions
                    .get(dependency)
                    .is_none_or(|dependency_position| *dependency_position >= position)
        }) {
            return None;
        }
        let mut pass_reads = BTreeSet::new();
        for read in &pass.reads {
            if !pass_reads.insert(read.resource)
                || !runtime_read_sampler_is_exact(read, &maps.resource_by_id)
            {
                return None;
            }
            validate_closed_read(plan, maps, &mut accounting, pass, read, position)?;
        }
        validate_closed_result(plan, maps, &mut accounting, pass)?;
        let mut pass_releases = BTreeSet::new();
        if pass.releases.iter().any(|resource| {
            !pass_releases.insert(*resource)
                || !pass_reads.contains(resource)
                || accounting.releases.insert(*resource, pass.id).is_some()
        }) {
            return None;
        }
        let expected_cache_keys = runtime_pass_cache_keys(
            &pass.kind,
            &pass.reads,
            pass.result,
            plan.working_format,
            plan.output_format,
            &maps.resource_formats,
        )
        .ok()?;
        if expected_cache_keys != pass.cache_keys {
            return None;
        }
    }
    Some(accounting)
}

fn validate_closed_read(
    _plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
    accounting: &mut ClosedGraphAccounting,
    pass: &RuntimePass,
    read: &RuntimeReadBinding,
    position: usize,
) -> Option<()> {
    let resource = maps.resource_by_id.get(&read.resource).copied()?;
    if pass.result == RuntimeResultBinding::Resource(read.resource) {
        return None;
    }
    if let RuntimeResourceProducer::Pass(producer) = resource.producer
        && (maps
            .pass_positions
            .get(&producer)
            .is_none_or(|producer_position| *producer_position >= position)
            || !pass.dependencies.contains(&producer))
    {
        return None;
    }
    let reads = accounting.actual_reads.entry(read.resource).or_default();
    *reads = reads.checked_add(1)?;
    accounting.actual_last_reads.insert(read.resource, pass.id);
    Some(())
}

fn validate_closed_result(
    plan: &LoweredGraphPlan,
    maps: &ClosedGraphMaps<'_>,
    accounting: &mut ClosedGraphAccounting,
    pass: &RuntimePass,
) -> Option<()> {
    match pass.result {
        RuntimeResultBinding::Resource(resource) => {
            let request = maps.resource_by_id.get(&resource).copied()?;
            if request.producer != RuntimeResourceProducer::Pass(pass.id)
                || accounting.results.insert(resource, pass.id).is_some()
            {
                return None;
            }
        }
        RuntimeResultBinding::Output(format) => {
            if !matches!(pass.kind, RuntimePassKind::Present) || format != plan.output_format {
                return None;
            }
        }
        RuntimeResultBinding::Empty => {}
    }
    Some(())
}

fn closed_graph_root<'plan>(
    plan: &'plan LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
) -> Option<(&'plan RuntimePass, &'plan RuntimeResourceRequest)> {
    let clear = plan.passes.first()?;
    let RuntimePassKind::ClearRoot {
        initialization: RuntimeInitialization::SurfaceBaseColor,
        ..
    } = clear.kind
    else {
        return None;
    };
    if !clear.dependencies.is_empty()
        || !clear.reads.is_empty()
        || !clear.releases.is_empty()
        || clear.cache_keys.is_some()
        || clear.result != RuntimeResultBinding::Resource(plan.root_working_image)
    {
        return None;
    }
    let root = resources.get(&plan.root_working_image).copied()?;
    base_graph_resource_has_fixed_facts(
        root,
        RuntimeResourceRole::RootWorkingImage,
        RuntimeResourceFormat::Working(plan.working_format),
        RuntimeResourceProducer::Pass(clear.id),
    )
    .then_some((clear, root))
}

struct ClosedGraphTraversal<'plan> {
    plan: &'plan LoweredGraphPlan,
    resources: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    contexts: Vec<ExecutableCompositionContext>,
    captures: Vec<ExecutableVelloCaptureFacts>,
    layer_compositions: Vec<ExecutableLayerCompositionFacts>,
    color_filters: Vec<ExecutableColorFilterFacts>,
    blurs: Vec<ExecutableBlurFacts>,
    drop_shadows: Vec<ExecutableDropShadowFacts>,
    filter_steps: Vec<ExecutableFilterStepFacts>,
    backdrops: Vec<ExecutableBackdropFacts>,
    expected_resources: BTreeSet<RuntimeResourceId>,
    cursor: usize,
}

impl<'plan> ClosedGraphTraversal<'plan> {
    fn new(
        plan: &'plan LoweredGraphPlan,
        resources: BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
        clear: &RuntimePass,
        root: &RuntimeResourceRequest,
    ) -> Self {
        Self {
            plan,
            resources,
            contexts: vec![ExecutableCompositionContext {
                current: root.id,
                producer: clear.id,
                contains_captured_source: false,
            }],
            captures: Vec::new(),
            layer_compositions: Vec::new(),
            color_filters: Vec::new(),
            blurs: Vec::new(),
            drop_shadows: Vec::new(),
            filter_steps: Vec::new(),
            backdrops: Vec::new(),
            expected_resources: BTreeSet::from([root.id]),
            cursor: 1,
        }
    }

    fn run(mut self) -> Option<ClosedExecutableGraphFacts> {
        while self.cursor < self.plan.passes.len() {
            self.visit_current_pass()?;
        }
        let clip_coverages_are_exact = self
            .layer_compositions
            .iter()
            .all(|layer| layer_has_exact_clip_coverage_capture(layer, &self.captures))
            && self.captures.iter().all(|capture| {
                capture.work().clip_coverage().is_none()
                    || self
                        .layer_compositions
                        .iter()
                        .any(|layer| layer.clip_coverage == Some(capture.target()))
            });
        if self.captures.is_empty()
            || self.contexts.len() != 1
            || !clip_coverages_are_exact
            || self.expected_resources.len() != self.plan.resources.len()
            || self
                .expected_resources
                .iter()
                .any(|resource| !self.resources.contains_key(resource))
        {
            return None;
        }
        Some(ClosedExecutableGraphFacts {
            working_format: self.plan.working_format,
            output_format: self.plan.output_format,
            captures: self.captures,
            layer_compositions: self.layer_compositions,
            color_filters: self.color_filters,
            blurs: self.blurs,
            drop_shadows: self.drop_shadows,
            filter_steps: self.filter_steps,
            backdrops: self.backdrops,
        })
    }

    fn visit_current_pass(&mut self) -> Option<()> {
        let pass = self.plan.passes.get(self.cursor)?;
        match &pass.kind {
            RuntimePassKind::ClearRoot {
                initialization: RuntimeInitialization::Transparent,
                color,
            } => self.visit_transparent_clear(pass, *color),
            RuntimePassKind::VelloCapture(Some(work)) if work.span().is_some() => {
                self.visit_span_capture(pass, work)
            }
            RuntimePassKind::VelloCapture(Some(work)) if work.clip_coverage().is_some() => {
                self.visit_clip_coverage_capture(pass)
            }
            RuntimePassKind::ColorFilter(Some(filter)) => self.visit_color_filter(pass, filter),
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.axis == RuntimeBlurAxis::Horizontal
                    && blur.input == RuntimeBlurInput::Rgba =>
            {
                self.visit_blur(pass, blur)
            }
            RuntimePassKind::BlurHorizontal(Some(blur))
                if blur.axis == RuntimeBlurAxis::Horizontal
                    && blur.input == RuntimeBlurInput::SourceAlpha =>
            {
                self.visit_drop_shadow(pass, blur)
            }
            RuntimePassKind::CopyBackdrop => self.visit_backdrop(pass),
            RuntimePassKind::Composite(Some(composite))
                if matches!(composite.kind, RuntimeCompositeKind::Layer { .. }) =>
            {
                self.visit_layer_composite(pass)
            }
            RuntimePassKind::Present => self.visit_present(pass),
            RuntimePassKind::ClearRoot {
                initialization: RuntimeInitialization::SurfaceBaseColor,
                ..
            }
            | RuntimePassKind::VelloCapture(None)
            | RuntimePassKind::VelloCapture(Some(_))
            | RuntimePassKind::CanonicalizeCapture
            | RuntimePassKind::ColorFilter(_)
            | RuntimePassKind::BlurHorizontal(_)
            | RuntimePassKind::BlurVertical(_)
            | RuntimePassKind::DropShadowColorize(_)
            | RuntimePassKind::Composite(_) => None,
        }
    }

    fn visit_transparent_clear(&mut self, pass: &RuntimePass, color: Color) -> Option<()> {
        if color != Color::TRANSPARENT
            || !pass.dependencies.is_empty()
            || !pass.reads.is_empty()
            || !pass.releases.is_empty()
            || pass.cache_keys.is_some()
        {
            return None;
        }
        let RuntimeResultBinding::Resource(resource) = pass.result else {
            return None;
        };
        let request = self.resources.get(&resource).copied()?;
        if !base_graph_resource_has_fixed_facts(
            request,
            RuntimeResourceRole::IsolationWorkingImage,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(pass.id),
        ) {
            return None;
        }
        self.expected_resources.insert(resource);
        self.contexts.push(ExecutableCompositionContext {
            current: resource,
            producer: pass.id,
            contains_captured_source: false,
        });
        self.advance(1)
    }

    fn visit_span_capture(&mut self, pass: &RuntimePass, work: &RuntimeVelloCapture) -> Option<()> {
        let span = work.span()?;
        let canonicalize = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let after_canonicalize = self.plan.passes.get(self.cursor.checked_add(2)?)?;
        let (coverage_pass, composite, pass_count) = if matches!(
            after_canonicalize.kind,
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(_)))
        ) {
            (
                Some(after_canonicalize),
                self.plan.passes.get(self.cursor.checked_add(3)?)?,
                4,
            )
        } else {
            (None, after_canonicalize, 3)
        };
        let (capture_target, capture_resource) =
            self.validate_span_capture_source(pass, canonicalize, span)?;
        let (canonical_target, canonical_resource) = self.validate_canonical_capture(
            pass,
            canonicalize,
            composite,
            capture_target,
            capture_resource,
        )?;
        let capture_facts = executable_vello_capture_facts(
            pass.id,
            capture_target,
            work,
            capture_resource.spatial,
        )?;
        let coverage_facts = match coverage_pass {
            Some(coverage) => Some(validate_closed_clip_coverage_capture(
                coverage,
                composite.id,
                &self.resources,
            )?),
            None => None,
        };
        let parent = *self.contexts.last()?;
        let layer = validate_closed_composite(
            composite,
            parent,
            canonical_resource,
            &self.resources,
            self.plan.working_format,
            false,
        )?;
        let RuntimeResultBinding::Resource(result) = composite.result else {
            return None;
        };
        self.record_span_capture(
            composite,
            [capture_target, canonical_target, result],
            capture_facts,
            coverage_facts,
            layer,
        )?;
        self.advance(pass_count)
    }

    fn validate_span_capture_source(
        &self,
        pass: &RuntimePass,
        canonicalize: &RuntimePass,
        span: &RuntimeVelloSpan,
    ) -> Option<(RuntimeResourceId, &'plan RuntimeResourceRequest)> {
        let RuntimeResultBinding::Resource(capture_target) = pass.result else {
            return None;
        };
        let expected_scope = if self.contexts.len() == 1 {
            RuntimeVelloSpanScope::CurrentParent
        } else {
            RuntimeVelloSpanScope::LayerSource
        };
        if !pass.dependencies.is_empty()
            || !pass.reads.is_empty()
            || !pass.releases.is_empty()
            || pass.cache_keys.is_some()
            || span.scope != expected_scope
        {
            return None;
        }
        let resource = self.resources.get(&capture_target).copied()?;
        base_graph_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::CaptureWorkingImage,
            RuntimeResourceFormat::VelloCaptureRgba8Unorm,
            RuntimeResourceProducer::Pass(pass.id),
        )
        .then_some(())
        .filter(|()| resource.expected_reads == 1 && resource.last_use == canonicalize.id)?;
        Some((capture_target, resource))
    }

    fn validate_canonical_capture(
        &self,
        capture: &RuntimePass,
        canonicalize: &RuntimePass,
        composite: &RuntimePass,
        capture_target: RuntimeResourceId,
        capture_resource: &RuntimeResourceRequest,
    ) -> Option<(RuntimeResourceId, &'plan RuntimeResourceRequest)> {
        if !matches!(canonicalize.kind, RuntimePassKind::CanonicalizeCapture)
            || canonicalize.dependencies.as_slice() != [capture.id]
            || canonicalize.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &canonicalize.reads[0],
                RuntimeReadRole::CaptureSource,
                capture_resource,
                RuntimeSamplingFilter::Linear,
                RuntimeSamplingEdge::ClampToExtent,
            )
            || canonicalize.releases.as_slice() != [capture_target]
        {
            return None;
        }
        let RuntimeResultBinding::Resource(canonical_target) = canonicalize.result else {
            return None;
        };
        let resource = self.resources.get(&canonical_target).copied()?;
        base_graph_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::FilterIntermediate,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(canonicalize.id),
        )
        .then_some(())
        .filter(|()| {
            resource.expected_reads == 1
                && resource.last_use == composite.id
                && resource.spatial == capture_resource.spatial
        })?;
        Some((canonical_target, resource))
    }

    fn record_span_capture(
        &mut self,
        composite: &RuntimePass,
        resources: [RuntimeResourceId; 3],
        capture: ExecutableVelloCaptureFacts,
        coverage: Option<(RuntimeResourceId, ExecutableVelloCaptureFacts)>,
        layer: Option<ExecutableLayerCompositionFacts>,
    ) -> Option<()> {
        let context = self.contexts.last_mut()?;
        context.current = resources[2];
        context.producer = composite.id;
        context.contains_captured_source = true;
        self.expected_resources.extend(resources);
        self.captures.push(capture);
        if let Some((resource, facts)) = coverage {
            self.expected_resources.insert(resource);
            self.captures.push(facts);
        }
        if let Some(layer) = layer {
            self.record_layer_resources(&layer);
            self.layer_compositions.push(layer);
        }
        Some(())
    }

    fn visit_clip_coverage_capture(&mut self, pass: &RuntimePass) -> Option<()> {
        let composite = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let (coverage, facts) =
            validate_closed_clip_coverage_capture(pass, composite.id, &self.resources)?;
        self.expected_resources.insert(coverage);
        self.captures.push(facts);
        self.advance(1)
    }

    fn visit_color_filter(
        &mut self,
        pass: &RuntimePass,
        filter: &RuntimeColorFilter,
    ) -> Option<()> {
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let color = validate_closed_color_filter(
            pass,
            context,
            source,
            filter,
            &self.resources,
            self.plan.working_format,
        )?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = color.result;
        runtime_context.producer = pass.id;
        self.expected_resources.insert(color.result);
        self.filter_steps
            .push(ExecutableFilterStepFacts::Color(pass.id));
        self.color_filters.push(color);
        self.advance(1)
    }

    fn visit_blur(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        self.visit_blur_with_edge(horizontal, blur, RuntimeSamplingEdge::TransparentBlack)
    }

    fn visit_blur_with_edge(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
        edge: RuntimeSamplingEdge,
    ) -> Option<()> {
        let vertical = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let validation = ClosedFilterValidation {
            context,
            source,
            resources: &self.resources,
            working_format: self.plan.working_format,
            edge,
        };
        let facts = validate_closed_blur(horizontal, vertical, blur, validation)?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = facts.result;
        runtime_context.producer = facts.vertical;
        self.expected_resources
            .extend([facts.intermediate, facts.result]);
        self.filter_steps.push(ExecutableFilterStepFacts::Blur {
            horizontal: facts.horizontal,
            vertical: facts.vertical,
        });
        self.blurs.push(facts);
        self.advance(2)
    }

    fn visit_drop_shadow(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        self.visit_drop_shadow_with_edge(horizontal, blur, RuntimeSamplingEdge::TransparentBlack)
    }

    fn visit_drop_shadow_with_edge(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
        edge: RuntimeSamplingEdge,
    ) -> Option<()> {
        let vertical = self.plan.passes.get(self.cursor.checked_add(1)?)?;
        let colorize = self.plan.passes.get(self.cursor.checked_add(2)?)?;
        let merge = self.plan.passes.get(self.cursor.checked_add(3)?)?;
        let context = *self.contexts.last()?;
        if !context.contains_captured_source {
            return None;
        }
        let source = self.resources.get(&context.current).copied()?;
        let validation = ClosedFilterValidation {
            context,
            source,
            resources: &self.resources,
            working_format: self.plan.working_format,
            edge,
        };
        let facts =
            validate_closed_drop_shadow([horizontal, vertical, colorize, merge], blur, validation)?;
        let runtime_context = self.contexts.last_mut()?;
        runtime_context.current = facts.result;
        runtime_context.producer = facts.merge;
        self.expected_resources.extend([
            facts.horizontal_result,
            facts.vertical_result,
            facts.shadow,
            facts.result,
        ]);
        self.filter_steps
            .push(ExecutableFilterStepFacts::DropShadow {
                horizontal: facts.horizontal,
                vertical: facts.vertical,
                colorize: facts.colorize,
                merge: facts.merge,
            });
        self.drop_shadows.push(facts);
        self.advance(4)
    }

    fn visit_backdrop(&mut self, copy: &RuntimePass) -> Option<()> {
        let foreground = match self.contexts.len() {
            1 => None,
            2 => Some(self.contexts.pop()?),
            _ => return None,
        };
        if foreground.is_some_and(|context| !context.contains_captured_source) {
            return None;
        }
        let parent = *self.contexts.first()?;
        let copied = self.validate_backdrop_copy(copy, parent)?;
        let filter_start = self.filter_steps.len();
        self.contexts.push(ExecutableCompositionContext {
            current: copied,
            producer: copy.id,
            contains_captured_source: true,
        });
        self.advance(1)?;
        self.visit_backdrop_filters()?;
        let filtered = self.contexts.pop()?;
        let filter_steps = self.filter_steps.get(filter_start..)?.to_vec();
        let group_clear = self.plan.passes.get(self.cursor)?;
        self.visit_transparent_clear(group_clear, Color::TRANSPARENT)?;
        let backdrop_composite = self.visit_backdrop_source_composite(filtered)?;
        let foreground_composite = match foreground {
            Some(foreground) => Some(self.visit_backdrop_foreground(foreground)?),
            None => None,
        };
        let completed_group = *self.contexts.last()?;
        let outer_composite = self.visit_backdrop_outer_composite(parent, completed_group)?;
        let result = self.contexts.first()?.current;
        self.backdrops.push(ExecutableBackdropFacts {
            copy: copy.id,
            completed_parent: parent.current,
            copied,
            foreground: foreground.map(|context| context.current),
            filter_steps,
            filtered: filtered.current,
            group_clear: group_clear.id,
            backdrop_composite,
            foreground_composite,
            outer_composite,
            completed_group: completed_group.current,
            result,
        });
        Some(())
    }

    fn validate_backdrop_copy(
        &mut self,
        copy: &RuntimePass,
        parent: ExecutableCompositionContext,
    ) -> Option<RuntimeResourceId> {
        let parent_resource = self.resources.get(&parent.current).copied()?;
        if copy.dependencies.as_slice() != [parent.producer]
            || copy.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &copy.reads[0],
                RuntimeReadRole::CompletedParent,
                parent_resource,
                RuntimeSamplingFilter::Nearest,
                RuntimeSamplingEdge::TransparentBlack,
            )
            || !copy.releases.is_empty()
            || copy.cache_keys.is_none()
        {
            return None;
        }
        let RuntimeResultBinding::Resource(copied) = copy.result else {
            return None;
        };
        let resource = self.resources.get(&copied).copied()?;
        if !base_graph_resource_has_fixed_facts(
            resource,
            RuntimeResourceRole::BackdropCopy,
            RuntimeResourceFormat::Working(self.plan.working_format),
            RuntimeResourceProducer::Pass(copy.id),
        ) || resource.expected_reads != 1
        {
            return None;
        }
        self.expected_resources.insert(copied);
        Some(copied)
    }

    fn visit_backdrop_filters(&mut self) -> Option<()> {
        loop {
            let pass = self.plan.passes.get(self.cursor)?;
            match &pass.kind {
                RuntimePassKind::ColorFilter(Some(filter)) => {
                    self.visit_color_filter(pass, filter)?;
                }
                RuntimePassKind::BlurHorizontal(Some(blur))
                    if blur.axis == RuntimeBlurAxis::Horizontal
                        && blur.input == RuntimeBlurInput::Rgba =>
                {
                    self.visit_backdrop_blur(pass, blur)?;
                }
                RuntimePassKind::BlurHorizontal(Some(blur))
                    if blur.axis == RuntimeBlurAxis::Horizontal
                        && blur.input == RuntimeBlurInput::SourceAlpha =>
                {
                    self.visit_backdrop_drop_shadow(pass, blur)?;
                }
                RuntimePassKind::ClearRoot {
                    initialization: RuntimeInitialization::Transparent,
                    color,
                } if *color == Color::TRANSPARENT => return Some(()),
                _ => return None,
            }
        }
    }

    fn visit_backdrop_blur(&mut self, horizontal: &RuntimePass, blur: &RuntimeBlur) -> Option<()> {
        let RuntimeSamplingEdge::SemanticBorderMirror(_) = blur.edge else {
            return None;
        };
        self.visit_blur_with_edge(horizontal, blur, blur.edge)
    }

    fn visit_backdrop_drop_shadow(
        &mut self,
        horizontal: &RuntimePass,
        blur: &RuntimeBlur,
    ) -> Option<()> {
        let RuntimeSamplingEdge::SemanticBorderMirror(_) = blur.edge else {
            return None;
        };
        self.visit_drop_shadow_with_edge(horizontal, blur, blur.edge)
    }

    fn visit_backdrop_source_composite(
        &mut self,
        filtered: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.next_layer_composite_with_optional_coverage()?;
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&filtered.current).copied()?;
        let layer = validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
        )??;
        if !runtime_composite_is_backdrop_inner(&layer.composite) {
            return None;
        }
        self.record_backdrop_composite_result(pass, layer)?;
        Some(pass.id)
    }

    fn visit_backdrop_foreground(
        &mut self,
        foreground: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.plan.passes.get(self.cursor)?;
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&foreground.current).copied()?;
        if validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            false,
        )?
        .is_some()
        {
            return None;
        }
        self.record_backdrop_span_result(pass)?;
        Some(pass.id)
    }

    fn visit_backdrop_outer_composite(
        &mut self,
        parent: ExecutableCompositionContext,
        group: ExecutableCompositionContext,
    ) -> Option<RuntimePassId> {
        let pass = self.next_layer_composite_with_optional_coverage()?;
        let source = self.resources.get(&group.current).copied()?;
        let layer = validate_closed_composite_with_parent_reads(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
            2,
        )??;
        if !runtime_composite_is_untransformed_outer(&layer.composite) {
            return None;
        }
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        self.contexts.pop()?;
        let root = self.contexts.first_mut()?;
        root.current = result;
        root.producer = pass.id;
        root.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)?;
        Some(pass.id)
    }

    fn next_layer_composite_with_optional_coverage(&mut self) -> Option<&'plan RuntimePass> {
        let pass = self.plan.passes.get(self.cursor)?;
        if matches!(
            pass.kind,
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(_)))
        ) {
            self.visit_clip_coverage_capture(pass)?;
        }
        let composite = self.plan.passes.get(self.cursor)?;
        matches!(
            composite.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { .. },
                ..
            }))
        )
        .then_some(composite)
    }

    fn record_backdrop_composite_result(
        &mut self,
        pass: &RuntimePass,
        layer: ExecutableLayerCompositionFacts,
    ) -> Option<()> {
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)
    }

    fn record_backdrop_span_result(&mut self, pass: &RuntimePass) -> Option<()> {
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.advance(1)
    }

    fn visit_layer_composite(&mut self, pass: &RuntimePass) -> Option<()> {
        if self.contexts.len() < 2 {
            return None;
        }
        let source_context = self.contexts.pop()?;
        if !source_context.contains_captured_source {
            return None;
        }
        let parent = *self.contexts.last()?;
        let source = self.resources.get(&source_context.current).copied()?;
        let layer = validate_closed_composite(
            pass,
            parent,
            source,
            &self.resources,
            self.plan.working_format,
            true,
        )??;
        let RuntimeResultBinding::Resource(result) = pass.result else {
            return None;
        };
        let context = self.contexts.last_mut()?;
        context.current = result;
        context.producer = pass.id;
        context.contains_captured_source = true;
        self.expected_resources.insert(result);
        self.record_layer_resources(&layer);
        self.layer_compositions.push(layer);
        self.advance(1)
    }

    fn visit_present(&mut self, pass: &RuntimePass) -> Option<()> {
        if self.cursor.checked_add(1)? != self.plan.passes.len()
            || pass.id != self.plan.final_present
            || self.contexts.len() != 1
        {
            return None;
        }
        let parent = self.contexts[0];
        let resource = self.resources.get(&parent.current).copied()?;
        if pass.dependencies.as_slice() != [parent.producer]
            || pass.reads.len() != 1
            || !runtime_read_has_exact_facts(
                &pass.reads[0],
                RuntimeReadRole::FinalWorkingImage,
                resource,
                RuntimeSamplingFilter::Linear,
                RuntimeSamplingEdge::ClampToExtent,
            )
            || pass.result != RuntimeResultBinding::Output(self.plan.output_format)
            || pass.releases.as_slice() != [parent.current]
            || resource.expected_reads != 1
            || resource.last_use != pass.id
        {
            return None;
        }
        self.advance(1)
    }

    fn record_layer_resources(&mut self, layer: &ExecutableLayerCompositionFacts) {
        if let Some(coverage) = layer.clip_coverage {
            self.expected_resources.insert(coverage);
        }
        if let Some(mask) = layer.alpha_mask {
            self.expected_resources.insert(mask);
        }
    }

    fn advance(&mut self, count: usize) -> Option<()> {
        self.cursor = self.cursor.checked_add(count)?;
        Some(())
    }
}
impl LoweredGraphPlan {
    fn closed_executable_graph_facts(&self) -> Option<ClosedExecutableGraphFacts> {
        if !matches!(self.output_format, Format::Rgba8 | Format::Bgra8) || self.passes.len() < 5 {
            return None;
        }
        let maps = ClosedGraphMaps::try_new(self)?;
        validate_closed_graph_accounting(self, &maps)?;
        let (clear, root) = closed_graph_root(self, &maps.resource_by_id)?;
        ClosedGraphTraversal::new(self, maps.resource_by_id, clear, root).run()
    }
}
fn runtime_read_sampler_is_exact(
    read: &RuntimeReadBinding,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> bool {
    let Some(resource) = resources.get(&read.resource).copied() else {
        return false;
    };
    let resolved_mask = match (&read.role, &resource.import) {
        (RuntimeReadRole::AlphaMask, Some(RuntimeResourceImport::ResolvedAlphaMask(upload))) => {
            Some(ShaderMaskSamplingKey::new(
                upload.quality(),
                upload.extend(),
            ))
        }
        (RuntimeReadRole::AlphaMask, None) => return false,
        (RuntimeReadRole::ClipCoverage, Some(_)) => return false,
        (RuntimeReadRole::ClipCoverage, None) => None,
        (_, _) => None,
    };
    read.sampler_key
        == SamplerKey::new(
            shader_binding_role(read.role),
            resource.format.shader_key(),
            match read.sampling_filter {
                RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
            },
            shader_sampling_edge(read.sampling_edge),
            resolved_mask,
        )
}

fn runtime_read_has_exact_facts(
    read: &RuntimeReadBinding,
    role: RuntimeReadRole,
    resource: &RuntimeResourceRequest,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
) -> bool {
    read.role == role
        && read.resource == resource.id
        && read.sampling_filter == sampling_filter
        && read.sampling_edge == sampling_edge
        && runtime_read_sampler_is_exact(read, &BTreeMap::from([(resource.id, resource)]))
}

fn validate_closed_clip_coverage_capture(
    pass: &RuntimePass,
    composite: RuntimePassId,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> Option<(RuntimeResourceId, ExecutableVelloCaptureFacts)> {
    let RuntimePassKind::VelloCapture(Some(work @ RuntimeVelloCapture::ClipCoverage(_))) =
        &pass.kind
    else {
        return None;
    };
    let RuntimeResultBinding::Resource(target) = pass.result else {
        return None;
    };
    if !pass.dependencies.is_empty()
        || !pass.reads.is_empty()
        || !pass.releases.is_empty()
        || pass.cache_keys.is_some()
    {
        return None;
    }
    let resource = resources.get(&target).copied()?;
    if !base_graph_resource_has_fixed_facts(
        resource,
        RuntimeResourceRole::ClipCoverage,
        RuntimeResourceFormat::ClipCoverageRgba8Unorm,
        RuntimeResourceProducer::Pass(pass.id),
    ) || resource.expected_reads != 1
        || resource.last_use != composite
    {
        return None;
    }
    Some((
        target,
        executable_vello_capture_facts(pass.id, target, work, resource.spatial)?,
    ))
}

fn layer_has_exact_clip_coverage_capture(
    layer: &ExecutableLayerCompositionFacts,
    captures: &[ExecutableVelloCaptureFacts],
) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        clip,
        outer_clips,
        clip_coverage,
        ..
    } = &layer.composite.kind
    else {
        return false;
    };
    let mut expected = outer_clips
        .iter()
        .map(|outer| RuntimeClipCoverageElement {
            clip: outer.clip.clone(),
            transform: outer.transform,
        })
        .collect::<Vec<_>>();
    if let Some(clip) = clip {
        expected.push(RuntimeClipCoverageElement {
            clip: (**clip).clone(),
            transform: *transform,
        });
    }
    match (expected.is_empty(), clip_coverage) {
        (true, None) => true,
        (false, Some(coverage)) => {
            let mut matching = captures
                .iter()
                .filter(|capture| capture.target() == *coverage);
            let exact = matching.next().is_some_and(|capture| {
                capture
                    .work()
                    .clip_coverage()
                    .is_some_and(|coverage| coverage.elements == expected)
            });
            exact && matching.next().is_none() && layer.clip_coverage == Some(*coverage)
        }
        (true, Some(_)) | (false, None) => false,
    }
}

fn validate_closed_color_filter(
    pass: &RuntimePass,
    context: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    filter: &RuntimeColorFilter,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
) -> Option<ExecutableColorFilterFacts> {
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || pass.dependencies.as_slice() != [source_producer]
        || pass.reads.len() != 1
        || !runtime_read_has_exact_facts(
            &pass.reads[0],
            RuntimeReadRole::FilterSource,
            source,
            RuntimeSamplingFilter::Nearest,
            RuntimeSamplingEdge::ClampToExtent,
        )
        || pass.releases.as_slice() != [source.id]
        || pass.cache_keys.is_none()
        || source.format != RuntimeResourceFormat::Working(working_format)
        || !matches!(
            source.role,
            RuntimeResourceRole::BackdropCopy
                | RuntimeResourceRole::FilterIntermediate
                | RuntimeResourceRole::CompositeResult
        )
        || source.expected_reads != 1
        || source.last_use != pass.id
        || filter.operations.is_empty()
        || filter.edge != RuntimeSamplingEdge::ClampToExtent
        || filter
            .operations
            .iter()
            .any(|operation| !runtime_color_operation_is_closed(operation))
    {
        return None;
    }
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    if result == source.id {
        return None;
    }
    let result_resource = resources.get(&result).copied()?;
    if !base_graph_resource_has_fixed_facts(
        result_resource,
        RuntimeResourceRole::FilterIntermediate,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    ) || result_resource.expected_reads != 1
        || source.spatial != result_resource.spatial
        || filter.spatial.source != source.spatial
        || filter.spatial.result != result_resource.spatial
    {
        return None;
    }
    Some(ExecutableColorFilterFacts {
        pass: pass.id,
        source: source.id,
        result,
        filter: filter.clone(),
    })
}

fn runtime_color_operation_is_closed(operation: &RuntimeColorOperation) -> bool {
    if operation.clamp_boundary != RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
    {
        return false;
    }
    match operation.operation {
        RuntimeColorOperationKind::Brightness(amount)
        | RuntimeColorOperationKind::Contrast(amount)
        | RuntimeColorOperationKind::Saturate(amount) => {
            if amount.zero() {
                amount.mantissa() == 0.0 && amount.exponent() == 0
            } else {
                amount.mantissa().is_finite() && (0.5..1.0).contains(&amount.mantissa())
            }
        }
        RuntimeColorOperationKind::Grayscale(amount)
        | RuntimeColorOperationKind::Invert(amount)
        | RuntimeColorOperationKind::Opacity(amount)
        | RuntimeColorOperationKind::Sepia(amount) => {
            amount.value().is_finite() && (0.0..=1.0).contains(&amount.value())
        }
        RuntimeColorOperationKind::HueRotate(angle) => {
            angle.sine().is_finite() && angle.cosine().is_finite()
        }
    }
}

#[derive(Clone, Copy)]
struct ClosedFilterValidation<'plan> {
    context: ExecutableCompositionContext,
    source: &'plan RuntimeResourceRequest,
    resources: &'plan BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
    edge: RuntimeSamplingEdge,
}

fn validate_closed_blur(
    horizontal: &RuntimePass,
    vertical: &RuntimePass,
    blur: &RuntimeBlur,
    validation: ClosedFilterValidation<'_>,
) -> Option<ExecutableBlurFacts> {
    let ClosedFilterValidation {
        context,
        source,
        resources,
        working_format,
        edge,
    } = validation;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || blur.axis != RuntimeBlurAxis::Horizontal
        || blur.input != RuntimeBlurInput::Rgba
        || !runtime_blur_is_closed(blur, true, edge)
        || blur.spatial.source != source.spatial
        || !closed_filter_source_is_exact(source, working_format, 1, horizontal.id)
        || !closed_unary_filter_pass_is_exact(
            horizontal,
            source_producer,
            source,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let intermediate = closed_filter_result(
        horizontal,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let RuntimePassKind::BlurVertical(Some(vertical_blur)) = &vertical.kind else {
        return None;
    };
    if !runtime_blur_matches_axis(vertical_blur, blur, RuntimeBlurAxis::Vertical)
        || intermediate.expected_reads != 1
        || intermediate.last_use != vertical.id
        || !closed_unary_filter_pass_is_exact(
            vertical,
            horizontal.id,
            intermediate,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let result = closed_filter_result(
        vertical,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    Some(ExecutableBlurFacts {
        horizontal: horizontal.id,
        vertical: vertical.id,
        source: source.id,
        intermediate: intermediate.id,
        result: result.id,
        blur: blur.clone(),
    })
}

fn validate_closed_drop_shadow(
    passes: [&RuntimePass; 4],
    blur: &RuntimeBlur,
    validation: ClosedFilterValidation<'_>,
) -> Option<ExecutableDropShadowFacts> {
    let ClosedFilterValidation {
        context,
        source,
        resources,
        working_format,
        edge,
    } = validation;
    let [horizontal, vertical, colorize, merge] = passes;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    if source.id != context.current
        || source_producer != context.producer
        || blur.axis != RuntimeBlurAxis::Horizontal
        || blur.input != RuntimeBlurInput::SourceAlpha
        || !runtime_blur_is_closed(blur, false, edge)
        || blur.spatial.source != source.spatial
        || !closed_filter_source_is_exact(source, working_format, 2, merge.id)
        || !closed_unary_filter_pass_is_exact(
            horizontal,
            source_producer,
            source,
            RuntimeReadRole::FilterSource,
            false,
            edge,
        )
    {
        return None;
    }
    let horizontal_result = closed_filter_result(
        horizontal,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let (vertical_result, parameters, shadow) = validate_closed_shadow_tail(
        [vertical, colorize],
        horizontal,
        horizontal_result,
        blur,
        resources,
        working_format,
        edge,
    )?;
    let result = validate_closed_shadow_merge(
        merge,
        ClosedShadowMergeInputs {
            source_producer,
            source,
            colorize,
            shadow,
            result_spatial: parameters.spatial.result,
        },
        resources,
        working_format,
    )?;
    Some(ExecutableDropShadowFacts {
        horizontal: horizontal.id,
        vertical: vertical.id,
        colorize: colorize.id,
        merge: merge.id,
        source: source.id,
        horizontal_result: horizontal_result.id,
        vertical_result: vertical_result.id,
        shadow: shadow.id,
        result: result.id,
        blur: blur.clone(),
        parameters,
    })
}

fn validate_closed_shadow_tail<'plan>(
    passes: [&RuntimePass; 2],
    horizontal: &RuntimePass,
    horizontal_result: &'plan RuntimeResourceRequest,
    blur: &RuntimeBlur,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
    edge: RuntimeSamplingEdge,
) -> Option<(
    &'plan RuntimeResourceRequest,
    RuntimeDropShadow,
    &'plan RuntimeResourceRequest,
)> {
    let [vertical, colorize] = passes;
    let RuntimePassKind::BlurVertical(Some(vertical_blur)) = &vertical.kind else {
        return None;
    };
    if !runtime_blur_matches_axis(vertical_blur, blur, RuntimeBlurAxis::Vertical)
        || horizontal_result.expected_reads != 1
        || horizontal_result.last_use != vertical.id
        || !closed_unary_filter_pass_is_exact(
            vertical,
            horizontal.id,
            horizontal_result,
            RuntimeReadRole::FilterSource,
            true,
            edge,
        )
    {
        return None;
    }
    let vertical_result = closed_filter_result(
        vertical,
        resources,
        RuntimeResourceRole::FilterIntermediate,
        blur.spatial.result,
        working_format,
    )?;
    let RuntimePassKind::DropShadowColorize(Some(parameters)) = &colorize.kind else {
        return None;
    };
    if !runtime_drop_shadow_is_closed(parameters, blur)
        || vertical_result.expected_reads != 1
        || vertical_result.last_use != colorize.id
        || !closed_unary_filter_pass_is_exact(
            colorize,
            vertical.id,
            vertical_result,
            RuntimeReadRole::BlurredSourceAlpha,
            true,
            RuntimeSamplingEdge::TransparentBlack,
        )
    {
        return None;
    }
    let shadow = closed_filter_result(
        colorize,
        resources,
        RuntimeResourceRole::ShadowImage,
        parameters.spatial.result,
        working_format,
    )?;
    Some((vertical_result, *parameters, shadow))
}

struct ClosedShadowMergeInputs<'plan> {
    source_producer: RuntimePassId,
    source: &'plan RuntimeResourceRequest,
    colorize: &'plan RuntimePass,
    shadow: &'plan RuntimeResourceRequest,
    result_spatial: RuntimeSpatialDescriptor,
}

fn validate_closed_shadow_merge<'plan>(
    merge: &RuntimePass,
    inputs: ClosedShadowMergeInputs<'plan>,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    working_format: WorkingFormat,
) -> Option<&'plan RuntimeResourceRequest> {
    let ClosedShadowMergeInputs {
        source_producer,
        source,
        colorize,
        shadow,
        result_spatial,
    } = inputs;
    let RuntimePassKind::Composite(Some(composite)) = &merge.kind else {
        return None;
    };
    if !matches!(composite.kind, RuntimeCompositeKind::DropShadow)
        || !composite.source_captured_before_outer_semantics
        || merge.dependencies.as_slice() != [source_producer, colorize.id]
        || merge.reads.len() != 2
        || !runtime_read_has_exact_facts(
            &merge.reads[0],
            RuntimeReadRole::CompositeSource,
            source,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || !runtime_read_has_exact_facts(
            &merge.reads[1],
            RuntimeReadRole::Shadow,
            shadow,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || !same_resource_set(&merge.releases, &[source.id, shadow.id])
        || merge.cache_keys.is_none()
        || shadow.expected_reads != 1
        || shadow.last_use != merge.id
    {
        return None;
    }
    closed_filter_result(
        merge,
        resources,
        RuntimeResourceRole::CompositeResult,
        result_spatial,
        working_format,
    )
}

fn closed_unary_filter_pass_is_exact(
    pass: &RuntimePass,
    producer: RuntimePassId,
    source: &RuntimeResourceRequest,
    role: RuntimeReadRole,
    releases_source: bool,
    edge: RuntimeSamplingEdge,
) -> bool {
    pass.dependencies.as_slice() == [producer]
        && pass.reads.len() == 1
        && runtime_read_has_exact_facts(
            &pass.reads[0],
            role,
            source,
            RuntimeSamplingFilter::Linear,
            edge,
        )
        && if releases_source {
            pass.releases.as_slice() == [source.id]
        } else {
            pass.releases.is_empty()
        }
        && pass.cache_keys.is_some()
}

fn closed_filter_source_is_exact(
    source: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    expected_reads: u32,
    last_use: RuntimePassId,
) -> bool {
    matches!(
        source.role,
        RuntimeResourceRole::BackdropCopy
            | RuntimeResourceRole::FilterIntermediate
            | RuntimeResourceRole::CompositeResult
    ) && source.format == RuntimeResourceFormat::Working(working_format)
        && source.expected_reads == expected_reads
        && source.last_use == last_use
}

fn closed_filter_result<'plan>(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &'plan RuntimeResourceRequest>,
    role: RuntimeResourceRole,
    spatial: RuntimeSpatialDescriptor,
    working_format: WorkingFormat,
) -> Option<&'plan RuntimeResourceRequest> {
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    let resource = resources.get(&result).copied()?;
    base_graph_resource_has_fixed_facts(
        resource,
        role,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    )
    .then_some(())
    .filter(|()| resource.spatial == spatial)?;
    Some(resource)
}

fn runtime_blur_is_closed(
    blur: &RuntimeBlur,
    require_nonzero: bool,
    edge: RuntimeSamplingEdge,
) -> bool {
    blur.standard_deviation.is_finite()
        && if require_nonzero {
            blur.standard_deviation > 0.0 && blur.support_radius > 0
        } else {
            blur.standard_deviation >= 0.0
        }
        && blur.edge == edge
        && blur.spatial.source.raster_scale.is_finite()
        && blur.spatial.result.raster_scale.is_finite()
}

fn runtime_drop_shadow_is_closed(shadow: &RuntimeDropShadow, blur: &RuntimeBlur) -> bool {
    shadow.standard_deviation == blur.standard_deviation
        && shadow.support_radius == blur.support_radius
        && shadow.spatial.source == blur.spatial.result
        && shadow.spatial.result == blur.spatial.result
        && shadow.edge == blur.edge
        && shadow.uses_source_alpha
        && shadow.uses_continuous_offset
        && shadow.retains_unchanged_source
        && shadow.offset.x().is_finite()
        && shadow.offset.y().is_finite()
        && [
            shadow.color.r(),
            shadow.color.g(),
            shadow.color.b(),
            shadow.color.a(),
        ]
        .into_iter()
        .all(f32::is_finite)
}

fn same_resource_set(actual: &[RuntimeResourceId], expected: &[RuntimeResourceId]) -> bool {
    actual.len() == expected.len()
        && actual.iter().copied().collect::<BTreeSet<_>>()
            == expected.iter().copied().collect::<BTreeSet<_>>()
}

fn validate_closed_composite(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    requires_isolated_source: bool,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    validate_closed_composite_with_parent_reads(
        pass,
        parent,
        source,
        resources,
        working_format,
        requires_isolated_source,
        1,
    )
}

fn runtime_composite_is_backdrop_inner(composite: &RuntimeComposite) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        parameters,
        outer_clips,
        ..
    } = &composite.kind
    else {
        return false;
    };
    *transform == Transform::identity()
        && parameters.destination_to_layer_local().affine() == Transform::identity()
        && parameters.opacity() == 1.0
        && parameters.blend() == BlendMode::Normal
        && outer_clips.is_empty()
        && parameters.alpha_mask().is_none()
}

fn runtime_composite_is_untransformed_outer(composite: &RuntimeComposite) -> bool {
    let RuntimeCompositeKind::Layer {
        transform,
        parameters,
        ..
    } = &composite.kind
    else {
        return false;
    };
    *transform == Transform::identity()
        && parameters.destination_to_layer_local().affine() == Transform::identity()
}

fn validate_closed_composite_with_parent_reads(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    requires_isolated_source: bool,
    parent_expected_reads: u32,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    let RuntimePassKind::Composite(Some(composite)) = &pass.kind else {
        return None;
    };
    let result = validate_closed_composite_base(
        pass,
        parent,
        source,
        resources,
        working_format,
        composite,
        parent_expected_reads,
    )?;

    match &composite.kind {
        RuntimeCompositeKind::SpanSourceOver => {
            if requires_isolated_source
                || pass.reads.len() != 2
                || !matches!(
                    source.role,
                    RuntimeResourceRole::FilterIntermediate | RuntimeResourceRole::CompositeResult
                )
            {
                return None;
            }
            Some(None)
        }
        RuntimeCompositeKind::Layer {
            transform,
            parameters,
            clip,
            outer_clips,
            clip_coverage,
        } => {
            let layer = ClosedLayerCompositeView {
                composite,
                transform,
                parameters,
                clip,
                outer_clips,
                clip_coverage: *clip_coverage,
            };
            validate_closed_layer_composite(
                pass,
                parent,
                source,
                resources,
                requires_isolated_source,
                result,
                layer,
            )
        }
        RuntimeCompositeKind::DropShadow => None,
    }
}

fn validate_closed_composite_base(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    composite: &RuntimeComposite,
    parent_expected_reads: u32,
) -> Option<RuntimeResourceId> {
    if !composite.source_captured_before_outer_semantics || source.id == parent.current {
        return None;
    }
    let parent_resource = resources.get(&parent.current).copied()?;
    let RuntimeResourceProducer::Pass(source_producer) = source.producer else {
        return None;
    };
    let mut expected_dependencies = vec![parent.producer, source_producer];
    if let RuntimeCompositeKind::Layer {
        clip_coverage: Some(coverage),
        ..
    } = &composite.kind
    {
        let coverage_resource = resources.get(coverage).copied()?;
        let RuntimeResourceProducer::Pass(coverage_producer) = coverage_resource.producer else {
            return None;
        };
        expected_dependencies.push(coverage_producer);
    }
    if pass.dependencies != expected_dependencies
        || pass.reads.len() < 2
        || !runtime_read_has_exact_facts(
            &pass.reads[0],
            RuntimeReadRole::CompositeParent,
            parent_resource,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::ClampToExtent,
        )
        || !runtime_read_has_exact_facts(
            &pass.reads[1],
            RuntimeReadRole::CompositeSource,
            source,
            RuntimeSamplingFilter::Linear,
            RuntimeSamplingEdge::TransparentBlack,
        )
        || parent_resource.format != RuntimeResourceFormat::Working(working_format)
        || source.format != RuntimeResourceFormat::Working(working_format)
        || !matches!(
            parent_resource.role,
            RuntimeResourceRole::RootWorkingImage
                | RuntimeResourceRole::IsolationWorkingImage
                | RuntimeResourceRole::CompositeResult
        )
        || parent_resource.expected_reads != parent_expected_reads
        || parent_resource.last_use != pass.id
        || source.expected_reads != 1
        || source.last_use != pass.id
    {
        return None;
    }
    let RuntimeResultBinding::Resource(result) = pass.result else {
        return None;
    };
    let result_resource = resources.get(&result).copied()?;
    base_graph_resource_has_fixed_facts(
        result_resource,
        RuntimeResourceRole::CompositeResult,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    )
    .then_some(())
    .filter(|()| result_resource.spatial == parent_resource.spatial)
    .map(|()| result)
}

struct ClosedLayerCompositeView<'a> {
    composite: &'a RuntimeComposite,
    transform: &'a Transform,
    parameters: &'a RuntimeLayerCompositeParameters,
    clip: &'a Option<Box<RenderClip>>,
    outer_clips: &'a [RuntimeOuterClip],
    clip_coverage: Option<RuntimeResourceId>,
}

fn validate_closed_layer_composite(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    requires_isolated_source: bool,
    result: RuntimeResourceId,
    layer: ClosedLayerCompositeView<'_>,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    let destination_to_layer_local: RuntimeDestinationToLayerLocal =
        layer.parameters.destination_to_layer_local();
    let opacity = layer.parameters.opacity();
    let alpha_mask = layer.parameters.alpha_mask();
    if layer
        .transform
        .as_array()
        .iter()
        .any(|value| !value.is_finite())
        || !opacity.is_finite()
        || !(0.0..=1.0).contains(&opacity)
        || !runtime_affine_is_finite_and_non_singular(destination_to_layer_local.affine())
        || layer.outer_clips.iter().any(|clip| {
            clip.transform
                .as_array()
                .iter()
                .any(|value| !value.is_finite())
        })
        || layer.parameters.has_clip() != (layer.clip.is_some() || !layer.outer_clips.is_empty())
        || layer.parameters.has_clip() != layer.clip_coverage.is_some()
        || !closed_composite_source_is_valid(source, requires_isolated_source, &layer)
    {
        return None;
    }
    let expected_read_count = 2usize
        .checked_add(usize::from(layer.clip_coverage.is_some()))?
        .checked_add(usize::from(alpha_mask.is_some()))?;
    if pass.reads.len() != expected_read_count {
        return None;
    }
    let next_read = validate_closed_clip_coverage(pass, resources, layer.clip_coverage, 2)?;
    validate_closed_alpha_mask(pass, resources, alpha_mask, next_read)?;
    Some(Some(ExecutableLayerCompositionFacts {
        pass: pass.id,
        parent: parent.current,
        source: source.id,
        clip_coverage: layer.clip_coverage,
        alpha_mask: alpha_mask.map(RuntimeResolvedAlphaMaskComposition::resource),
        result,
        composite: layer.composite.clone(),
    }))
}

fn closed_composite_source_is_valid(
    source: &RuntimeResourceRequest,
    requires_isolated_source: bool,
    layer: &ClosedLayerCompositeView<'_>,
) -> bool {
    (requires_isolated_source
        && matches!(
            source.role,
            RuntimeResourceRole::CompositeResult | RuntimeResourceRole::FilterIntermediate
        ))
        || (!requires_isolated_source
            && source.role == RuntimeResourceRole::FilterIntermediate
            && *layer.transform == Transform::identity()
            && layer.parameters.opacity() == 1.0
            && layer.parameters.blend() == BlendMode::Normal
            && layer.clip.is_none()
            && !layer.outer_clips.is_empty()
            && layer.parameters.alpha_mask().is_none())
}

fn validate_closed_clip_coverage(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    clip_coverage: Option<RuntimeResourceId>,
    next_read: usize,
) -> Option<usize> {
    let Some(coverage) = clip_coverage else {
        return Some(next_read);
    };
    let coverage_resource = resources.get(&coverage).copied()?;
    if !runtime_read_has_exact_facts(
        &pass.reads[next_read],
        RuntimeReadRole::ClipCoverage,
        coverage_resource,
        RuntimeSamplingFilter::Linear,
        RuntimeSamplingEdge::TransparentBlack,
    ) || coverage_resource.role != RuntimeResourceRole::ClipCoverage
        || coverage_resource.format != RuntimeResourceFormat::ClipCoverageRgba8Unorm
        || !matches!(coverage_resource.producer, RuntimeResourceProducer::Pass(_))
        || coverage_resource.import.is_some()
        || coverage_resource.expected_reads != 1
        || coverage_resource.last_use != pass.id
    {
        return None;
    }
    next_read.checked_add(1)
}

fn validate_closed_alpha_mask(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
    next_read: usize,
) -> Option<()> {
    let Some(mask) = alpha_mask else {
        return Some(());
    };
    let mask_resource = resources.get(&mask.resource()).copied()?;
    let Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) = &mask_resource.import else {
        return None;
    };
    let mask_filter = match mask.sampling().quality() {
        ShaderMaskQualityKey::Low => RuntimeSamplingFilter::Nearest,
        ShaderMaskQualityKey::Medium | ShaderMaskQualityKey::High => RuntimeSamplingFilter::Linear,
    };
    (runtime_read_has_exact_facts(
        &pass.reads[next_read],
        RuntimeReadRole::AlphaMask,
        mask_resource,
        mask_filter,
        RuntimeSamplingEdge::ClampToExtent,
    ) && mask_resource.role == RuntimeResourceRole::ImportedImage
        && mask_resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm
        && mask_resource.producer == RuntimeResourceProducer::Imported
        && upload.physical_size() == mask.image_dimensions()
        && mask_resource.spatial.device_extent == mask.image_dimensions()
        && mask.sampling() == ShaderMaskSamplingKey::new(upload.quality(), upload.extend())
        && RuntimeMaskTexelCenterFacts::try_new(mask.image_dimensions()).ok()
            == Some(mask.texel_center_facts())
        && mask_resource.expected_reads != 0
        && mask_resource.last_use >= pass.id)
        .then_some(())
}

pub(super) fn base_graph_resource_has_fixed_facts(
    resource: &RuntimeResourceRequest,
    role: RuntimeResourceRole,
    format: RuntimeResourceFormat,
    producer: RuntimeResourceProducer,
) -> bool {
    resource.role == role
        && resource.format == format
        && resource.producer == producer
        && resource.import.is_none()
}

pub(super) fn executable_vello_capture_facts(
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: &RuntimeVelloCapture,
    spatial: RuntimeSpatialDescriptor,
) -> Option<ExecutableVelloCaptureFacts> {
    let valid_work = match work {
        RuntimeVelloCapture::Span(span) => {
            !span.commands.commands.is_empty() && span.captured_before_outer_semantics
        }
        RuntimeVelloCapture::ClipCoverage(coverage) => {
            !coverage.elements.is_empty()
                && coverage.elements.iter().all(|element| {
                    element
                        .transform
                        .as_array()
                        .iter()
                        .all(|value| value.is_finite())
                })
        }
    };
    if !valid_work
        || spatial.device_extent.width() == 0
        || spatial.device_extent.height() == 0
        || !spatial.texel_origin.x().is_finite()
        || !spatial.texel_origin.y().is_finite()
        || !spatial.raster_scale.is_finite()
        || spatial.raster_scale <= 0.0
    {
        return None;
    }
    let expected_device_x = spatial.texel_origin.x() * spatial.raster_scale;
    let expected_device_y = spatial.texel_origin.y() * spatial.raster_scale;
    let tolerance = f64::EPSILON
        * spatial
            .raster_scale
            .abs()
            .max(expected_device_x.abs())
            .max(expected_device_y.abs())
            .max(1.0)
        * 8.0;
    if (expected_device_x - f64::from(spatial.device_origin.0)).abs() > tolerance
        || (expected_device_y - f64::from(spatial.device_origin.1)).abs() > tolerance
    {
        return None;
    }
    let grid_transform =
        Transform::translation(-spatial.texel_origin.x(), -spatial.texel_origin.y())
            .ok()?
            .then(Transform::scale(spatial.raster_scale, spatial.raster_scale).ok()?)
            .ok()?;
    let initial_transform = match work {
        RuntimeVelloCapture::Span(span) => match span.scope {
            RuntimeVelloSpanScope::CurrentParent => span
                .capture_transform
                .then(span.parent_to_surface)
                .ok()?
                .then(grid_transform)
                .ok()?,
            RuntimeVelloSpanScope::LayerSource => {
                span.capture_transform.then(grid_transform).ok()?
            }
        },
        RuntimeVelloCapture::ClipCoverage(_) => grid_transform,
    };
    Some(ExecutableVelloCaptureFacts {
        pass,
        target,
        work: work.clone(),
        initial_transform,
        antialiasing: work.antialiasing(),
        target_extent: spatial.device_extent,
        texel_origin: spatial.texel_origin,
        raster_scale: spatial.raster_scale,
    })
}
pub(super) fn preparation_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchPassSemantics {
    ClosedExecutable,
    FuturePass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GraphPreparationIneligibility {
    OutsideClosedExecutableGraph,
}

impl GraphPreparationIneligibility {
    pub(super) fn into_error(self) -> Error {
        match self {
            Self::OutsideClosedExecutableGraph => preparation_error(
                "a graph outside the closed executable subset cannot enter runtime preparation",
            ),
        }
    }
}

pub(super) enum PrePreparationGraphClassification {
    ExactBase(BasePreparableGraph),
    ExactComposition(ClosedExecutableGraph),
    ExactColorFilter(ColorFilterPreparableGraph),
    ExactSpatialFilter(SpatialFilterPreparableGraph),
    ExactBackdrop(BackdropPreparableGraph),
    FuturePasses,
    Ineligible(GraphPreparationIneligibility),
}
impl PrePreparationGraphClassification {
    pub(super) fn classify(lowered: LoweredGraphPlan) -> Self {
        let closed = match ClosedExecutableGraph::try_from_lowered(lowered) {
            Ok(closed) => closed,
            Err(lowered) => {
                let mut contains_future_passes = false;
                for pass in &lowered.passes {
                    match dispatch_pass_semantics(&pass.kind) {
                        Some(DispatchPassSemantics::ClosedExecutable) => {}
                        Some(DispatchPassSemantics::FuturePass) => {
                            contains_future_passes = true;
                        }
                        None => {
                            return Self::Ineligible(
                                GraphPreparationIneligibility::OutsideClosedExecutableGraph,
                            );
                        }
                    }
                }
                return if contains_future_passes {
                    Self::FuturePasses
                } else {
                    Self::Ineligible(GraphPreparationIneligibility::OutsideClosedExecutableGraph)
                };
            }
        };
        match BasePreparableGraph::try_from_closed(closed) {
            Ok(preparable) => Self::ExactBase(preparable),
            Err(closed) => match BackdropPreparableGraph::try_from_closed(*closed) {
                Ok(preparable) => Self::ExactBackdrop(preparable),
                Err(closed) => match SpatialFilterPreparableGraph::try_from_closed(*closed) {
                    Ok(preparable) => Self::ExactSpatialFilter(preparable),
                    Err(closed) => match ColorFilterPreparableGraph::try_from_closed(*closed) {
                        Ok(preparable) => Self::ExactColorFilter(preparable),
                        Err(closed) => match CompositionPreparableGraph::try_from_closed(*closed) {
                            Ok(preparable) => Self::ExactComposition(preparable.into_closed()),
                            Err(_) => Self::Ineligible(
                                GraphPreparationIneligibility::OutsideClosedExecutableGraph,
                            ),
                        },
                    },
                },
            },
        }
    }
}

fn dispatch_pass_semantics(kind: &RuntimePassKind) -> Option<DispatchPassSemantics> {
    match kind {
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            ..
        }
        | RuntimePassKind::CanonicalizeCapture
        | RuntimePassKind::Present => Some(DispatchPassSemantics::ClosedExecutable),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            color,
        } if *color == Color::TRANSPARENT => Some(DispatchPassSemantics::ClosedExecutable),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            ..
        } => None,
        RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span)))
            if !span.commands.commands.is_empty()
                && span.captured_before_outer_semantics
                && span
                    .capture_transform
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite())
                && span
                    .parent_to_surface
                    .as_array()
                    .iter()
                    .all(|value| value.is_finite()) =>
        {
            Some(DispatchPassSemantics::ClosedExecutable)
        }
        RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(coverage)))
            if !coverage.elements.is_empty()
                && coverage.elements.iter().all(|element| {
                    element
                        .transform
                        .as_array()
                        .iter()
                        .all(|value| value.is_finite())
                }) =>
        {
            Some(DispatchPassSemantics::ClosedExecutable)
        }
        RuntimePassKind::VelloCapture(_) => None,
        RuntimePassKind::CopyBackdrop
        | RuntimePassKind::ColorFilter(Some(_))
        | RuntimePassKind::DropShadowColorize(Some(_)) => Some(DispatchPassSemantics::FuturePass),
        RuntimePassKind::BlurHorizontal(Some(blur)) if blur.axis == RuntimeBlurAxis::Horizontal => {
            Some(DispatchPassSemantics::FuturePass)
        }
        RuntimePassKind::BlurVertical(Some(blur)) if blur.axis == RuntimeBlurAxis::Vertical => {
            Some(DispatchPassSemantics::FuturePass)
        }
        RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(_)
        | RuntimePassKind::BlurVertical(_)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => None,
        RuntimePassKind::Composite(Some(composite))
            if composite.source_captured_before_outer_semantics =>
        {
            Some(match composite.kind {
                RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::Layer { .. } => {
                    DispatchPassSemantics::ClosedExecutable
                }
                RuntimeCompositeKind::DropShadow => DispatchPassSemantics::FuturePass,
            })
        }
        RuntimePassKind::Composite(Some(_)) => None,
    }
}
