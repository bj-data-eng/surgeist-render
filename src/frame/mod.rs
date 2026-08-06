mod bounds;
mod filter;
mod graph;
mod lower;
#[cfg(test)]
mod test_support;
mod validate;

#[cfg(test)]
#[expect(
    unused_imports,
    reason = "preserve existing crate-visible frame test observation paths"
)]
pub(crate) use bounds::{
    SpatialPrimitivesForTest, spatial_primitives_for_test, transformed_logical_bounds_for_test,
};
#[cfg(test)]
pub(crate) use filter::{
    OrderedFilterEdgeObservation, OrderedFilterIntentObservation, OrderedFilterPlanObservation,
    OrderedFilterStepObservation, ordered_filter_plan_for_test,
};
pub(crate) use lower::{
    GraphLoweringBlur, GraphLoweringBlurInput, GraphLoweringClipCoverage, GraphLoweringColorFilter,
    GraphLoweringColorOperation, GraphLoweringComposite, GraphLoweringCompositeKind,
    GraphLoweringDropShadow, GraphLoweringEdgePolicy, GraphLoweringFilterSpatialMapping,
    GraphLoweringGeneration, GraphLoweringImportView, GraphLoweringInitialization,
    GraphLoweringPassId, GraphLoweringPassKind, GraphLoweringPassResult, GraphLoweringPassView,
    GraphLoweringReadBinding, GraphLoweringReadRole, GraphLoweringResourceId,
    GraphLoweringResourceProducer, GraphLoweringResourceRole, GraphLoweringResourceView,
    GraphLoweringSamplingEdge, GraphLoweringSamplingFilter, GraphLoweringSpatialDescriptor,
    GraphLoweringVelloCapture, GraphLoweringVelloSpan, GraphLoweringVelloSpanScope,
};
#[expect(
    unused_imports,
    reason = "preserve existing crate-visible frame lowering paths"
)]
pub(crate) use lower::{
    GraphLoweringDestinationToLayerLocal, GraphLoweringOuterClip,
    GraphLoweringResolvedAlphaMaskComposition, GraphLoweringView,
};
#[cfg(test)]
#[expect(
    unused_imports,
    reason = "preserve existing crate-visible frame test observation paths"
)]
pub(crate) use test_support::{
    BackdropDependencyObservation, FinalPresentDeclarationObservation,
    FinalPresentSchedulingObservation, ForcedC08CaptureMappingForTest,
    ForcedC08GraphCaptureObservationForTest, FramePlanObservation, FramePlanResultObservation,
    FramePlanRouteObservation, FrameSelectionRequirementObservation,
    GraphBaseInitializationObservation, GraphEdgeLifetimeObservation, GraphFailureObservation,
    GraphLoweringFaultForTest, GraphOwnerCallObservation, InvalidSemanticGraphStateForTest,
    SemanticGraphProbeError, VelloCommandObservation, VelloSpanObservation,
    VelloSpanScopeObservation, authored_filter_graph_for_test,
    final_present_declaration_observation_for_test, final_present_scheduling_observation_for_test,
    forced_c08_graph_for_test, forced_c08_graph_with_capture_mapping_for_test,
    frame_plan_result_observation_for_test, invalid_semantic_graph_state_for_test,
    semantic_graph_base_initialization_observation_for_test,
    semantic_graph_edge_lifetime_observation_for_test,
};

pub(crate) use bounds::LogicalBounds;
use bounds::{FrameSpatialPlan, SemanticContributionDomain, SemanticSourceContribution};
pub(crate) use graph::GpuRenderGraph;
use graph::{SemanticFrameGraphPlanner, graph_build};
use validate::validate_semantic_frame_graph;

use super::{
    command::{RenderCommand, RenderCommands},
    error::{Error, Result, UnresolvedResource, UnresolvedResourceKind},
    geometry::{Rect, Size, Transform},
    paint::Color,
    renderer::Antialiasing,
    text::TextRunBoundsKind,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrameContext {
    output_bounds: LogicalBounds,
    surface_scale: f64,
    antialiasing: Antialiasing,
    base_color: Color,
}

impl FrameContext {
    pub(crate) fn try_new(
        surface_size: Size,
        surface_scale: f64,
        antialiasing: Antialiasing,
        base_color: Color,
    ) -> Result<Self> {
        if !surface_scale.is_finite() || surface_scale <= 0.0 {
            return Err(Error::invalid_value(
                "frame surface scale",
                surface_scale,
                "must be finite and greater than 0",
            ));
        }
        let output_bounds = LogicalBounds::try_from_rect(
            Rect::new(0.0, 0.0, surface_size.width(), surface_size.height()),
            "frame output bounds",
        )?;
        Ok(Self {
            output_bounds,
            surface_scale,
            antialiasing,
            base_color,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FramePlan {
    DirectVello(DirectVelloPlan),
    GpuGraph(GpuRenderGraph),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectVelloPlan {
    commands: RenderCommands,
    output_mapping: FrameSpatialPlan,
    antialiasing: Antialiasing,
    base_color: Color,
}

impl DirectVelloPlan {
    pub(crate) fn into_commands(self) -> RenderCommands {
        self.commands
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphSelectionRequirement {
    ResolvedAlphaMask,
    BoundedBackdrop,
}

impl FramePlan {
    pub(crate) fn try_from_commands(
        commands: RenderCommands,
        context: FrameContext,
    ) -> Result<Self> {
        let contribution = SemanticSourceContribution::try_from_commands(
            commands.commands,
            context.initial_parent_contribution(),
            SemanticContributionDomain::RootOutputBounded(context.output_bounds),
            context,
            Transform::identity(),
        )?;
        let commands = RenderCommands::new(contribution.commands);
        let selection_requirements = graph_selection_requirements(&commands.commands);
        if selection_requirements.is_empty() {
            return Ok(Self::DirectVello(DirectVelloPlan {
                commands,
                output_mapping: context.output_spatial_plan()?,
                antialiasing: context.antialiasing,
                base_color: context.base_color,
            }));
        }

        require_graph_text_bounds(&commands.commands, &mut Vec::new())?;
        let output_spatial = match context.output_spatial_plan()? {
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
            FrameSpatialPlan::Empty(_) => {
                return Err(Error::invalid_value(
                    "frame graph output bounds",
                    "empty",
                    "must be non-empty before a custom frame graph is planned",
                ));
            }
        };
        let graph = SemanticFrameGraphPlanner::build(
            commands,
            context,
            output_spatial,
            selection_requirements,
        )?;
        graph_build(validate_semantic_frame_graph(&graph))?;
        Ok(Self::GpuGraph(graph))
    }
}

fn graph_selection_requirements(commands: &[RenderCommand]) -> Vec<GraphSelectionRequirement> {
    let mut requirements = Vec::new();
    collect_graph_selection_requirements(commands, &mut requirements);
    requirements
}

fn collect_graph_selection_requirements(
    commands: &[RenderCommand],
    requirements: &mut Vec<GraphSelectionRequirement>,
) {
    for command in commands {
        let RenderCommand::Layer { layer, children } = command else {
            continue;
        };
        if layer.backdrop.is_some()
            && !requirements.contains(&GraphSelectionRequirement::BoundedBackdrop)
        {
            requirements.push(GraphSelectionRequirement::BoundedBackdrop);
        }
        if layer.mask.is_some()
            && !requirements.contains(&GraphSelectionRequirement::ResolvedAlphaMask)
        {
            requirements.push(GraphSelectionRequirement::ResolvedAlphaMask);
        }
        collect_graph_selection_requirements(children, requirements);
    }
}

fn require_graph_text_bounds(commands: &[RenderCommand], path: &mut Vec<usize>) -> Result<()> {
    for (index, command) in commands.iter().enumerate() {
        path.push(index);
        match command {
            RenderCommand::TextRun { bounds, .. }
                if bounds.kind() == TextRunBoundsKind::Unspecified =>
            {
                let identifier = path
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(".");
                path.pop();
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::TextRunInkBounds,
                    format!("normalized command {identifier}"),
                )));
            }
            RenderCommand::Layer { children, .. } => {
                require_graph_text_bounds(children, path)?;
            }
            RenderCommand::Fill { .. }
            | RenderCommand::Stroke { .. }
            | RenderCommand::Shadow { .. }
            | RenderCommand::Image { .. }
            | RenderCommand::TextRun { .. } => {}
        }
        path.pop();
    }
    Ok(())
}
