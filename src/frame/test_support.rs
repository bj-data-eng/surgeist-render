use super::{
    DirectVelloPlan, FrameContext, FramePlan, GraphSelectionRequirement,
    bounds::{
        FrameSpatialPlan, LogicalBounds, NonEmptyFrameSpatialPlan, PositiveDeviceExtent,
        SignedDeviceOrigin,
    },
    graph::{
        BlurInput, CaptureBoundsCoordinateSpace, GpuRenderGraph, GraphBuildResult, GraphGeneration,
        GraphValidationError, PassIndex, ResourceIndex, SemanticCompositeKind,
        SemanticFrameGraphPlanner, SemanticGraphBuilder, SemanticImportKind, SemanticPassId,
        SemanticPassIntent, SemanticPassResult, SemanticResourceDescriptor, SemanticResourceId,
        SemanticResourceProducer, SemanticResourceRole, SemanticVelloSpanScope,
        WorkingImageInitialization, graph_build,
    },
    validate::validate_semantic_frame_graph,
};
use crate::{
    command::{LayerPassPlan, RenderCommand, RenderCommands},
    error::{Error, Result, UnresolvedResource},
    geometry::{PhysicalSize, Point, Rect, Transform},
    paint::Color,
    renderer::Antialiasing,
    style::FilterList,
    text::TextRunBoundsKind,
};

#[cfg(test)]
pub(crate) fn forced_c08_graph_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> Result<GpuRenderGraph> {
    forced_c08_graph_with_capture_mapping_for_test(
        commands,
        context,
        ForcedC08CaptureMappingForTest::identity(),
    )
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ForcedC08CaptureMappingForTest {
    capture_transform: Transform,
    parent_to_surface: Transform,
}

#[cfg(test)]
impl ForcedC08CaptureMappingForTest {
    pub(crate) const fn identity() -> Self {
        Self {
            capture_transform: Transform::IDENTITY,
            parent_to_surface: Transform::IDENTITY,
        }
    }

    pub(crate) const fn new(capture_transform: Transform, parent_to_surface: Transform) -> Self {
        Self {
            capture_transform,
            parent_to_surface,
        }
    }
}

#[cfg(test)]
pub(crate) fn forced_c08_graph_with_capture_mapping_for_test(
    commands: RenderCommands,
    context: FrameContext,
    mapping: ForcedC08CaptureMappingForTest,
) -> Result<GpuRenderGraph> {
    let output_spatial = match context.output_spatial_plan()? {
        FrameSpatialPlan::NonEmpty(spatial) => spatial,
        FrameSpatialPlan::Empty(_) => {
            return Err(Error::invalid_value(
                "forced C08 graph output bounds",
                "empty",
                "must be non-empty before the private graph fixture is planned",
            ));
        }
    };
    let graph = SemanticFrameGraphPlanner::build_with_capture_mapping(
        commands,
        context,
        output_spatial,
        Vec::new(),
        mapping.capture_transform,
        mapping.parent_to_surface,
        CaptureBoundsCoordinateSpace::ForcedMappedForTest,
    )?;
    graph_build(validate_semantic_frame_graph(&graph))?;
    Ok(graph)
}

/// Test-only authored-filter ingress into the exact graph executor. This starts with
/// authored [`FilterList`] values and ordinary normalized capture input, then
/// uses the production planner's source capture, filter lowering, composition,
/// scheduling, and validation owners. It does not execute or encode the graph.
#[cfg(test)]
pub(crate) fn authored_filter_graph_for_test(
    filters: Vec<FilterList>,
    commands: RenderCommands,
    context: FrameContext,
) -> Result<GpuRenderGraph> {
    let graph =
        SemanticFrameGraphPlanner::build_authored_filter_fixture(filters, commands, context)?;
    graph_build(validate_semantic_frame_graph(&graph))?;
    Ok(graph)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ForcedC08GraphCaptureObservationForTest {
    pub(crate) antialiasing: Antialiasing,
    pub(crate) capture_transform: Transform,
    pub(crate) parent_to_surface: Transform,
    pub(crate) device_origin: (i32, i32),
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
}

impl GpuRenderGraph {
    #[cfg(test)]
    pub(crate) fn forced_capture_observations_for_test(
        &self,
    ) -> Vec<ForcedC08GraphCaptureObservationForTest> {
        self.vello_spans
            .iter()
            .map(|span| {
                let pass = self
                    .passes
                    .iter()
                    .find(|pass| pass.id == span.capture_pass)
                    .expect("a validated C08 span must retain its capture pass");
                let SemanticPassResult::Resource(resource_id) = pass.result else {
                    unreachable!("a validated C08 capture pass must produce one resource");
                };
                let resource = self
                    .resources
                    .iter()
                    .find(|resource| resource.id == resource_id)
                    .expect("a validated C08 capture must retain its output resource");
                let spatial = resource.descriptor;
                ForcedC08GraphCaptureObservationForTest {
                    antialiasing: span.antialiasing,
                    capture_transform: span.capture_transform,
                    parent_to_surface: span.parent_to_surface,
                    device_origin: (spatial.device_origin.x, spatial.device_origin.y),
                    texel_origin: spatial.texel_center_mapping.origin,
                    extent: PhysicalSize::new(
                        spatial.device_extent.width,
                        spatial.device_extent.height,
                    ),
                    raster_scale: spatial.texel_center_mapping.raster_scale.get(),
                }
            })
            .collect()
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
pub(crate) enum GraphLoweringFaultForTest {
    MissingResourceBinding,
    DuplicateReadBinding,
    ForwardDependency,
    StaleResourceGeneration,
    InconsistentLastUse,
}

#[cfg(test)]
impl GpuRenderGraph {
    pub(crate) fn with_lowering_fault_for_test(&self, fault: GraphLoweringFaultForTest) -> Self {
        let mut graph = self.clone();
        match fault {
            GraphLoweringFaultForTest::MissingResourceBinding => {
                if let Some(read) = graph
                    .passes
                    .iter_mut()
                    .find_map(|pass| pass.reads.first_mut())
                {
                    *read = SemanticResourceId::new(graph.generation, ResourceIndex(u32::MAX));
                }
            }
            GraphLoweringFaultForTest::DuplicateReadBinding => {
                if let Some(pass) = graph.passes.iter_mut().find(|pass| !pass.reads.is_empty())
                    && let Some(read) = pass.reads.first().copied()
                {
                    pass.reads.push(read);
                }
            }
            GraphLoweringFaultForTest::ForwardDependency => {
                if let Some(pass) = graph.passes.first_mut() {
                    pass.dependencies.push(graph.final_present);
                }
            }
            GraphLoweringFaultForTest::StaleResourceGeneration => {
                if let Some(read) = graph
                    .passes
                    .iter_mut()
                    .find_map(|pass| pass.reads.first_mut())
                {
                    read.generation = GraphGeneration(graph.generation.0.saturating_add(1));
                }
            }
            GraphLoweringFaultForTest::InconsistentLastUse => {
                if let (Some(resource), Some(pass)) =
                    (graph.resources.first_mut(), graph.passes.first())
                {
                    resource.releasable_after = Some(pass.id);
                }
            }
        }
        graph
    }
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
    pub(crate) surface_base_color: Option<crate::paint::Color>,
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
    let spatial = FrameContext::try_for_spatial_test(2.0)
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
                crate::paint::Color::BLACK,
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
            invalid_stale_resource_identity_for_test()
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
        _ => invalid_semantic_graph_structure_state_for_test(state),
    }
}

#[cfg(test)]
fn invalid_stale_resource_identity_for_test() -> SemanticGraphProbeResult<GraphFailureObservation> {
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

#[cfg(test)]
fn invalid_semantic_graph_structure_state_for_test(
    state: InvalidSemanticGraphStateForTest,
) -> SemanticGraphProbeResult<GraphFailureObservation> {
    match state {
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
        _ => invalid_semantic_graph_lifecycle_state_for_test(state),
    }
}

#[cfg(test)]
fn invalid_semantic_graph_lifecycle_state_for_test(
    state: InvalidSemanticGraphStateForTest,
) -> SemanticGraphProbeResult<GraphFailureObservation> {
    match state {
        InvalidSemanticGraphStateForTest::NonTransparentCaptureBase => {
            let mut builder = graph_probe_builder()?;
            let capture = graph_probe_value(builder.declare_resource(graph_probe_descriptor(
                SemanticResourceRole::IsolationWorkingImage,
                1,
            )?))?;
            graph_probe_failure(builder.declare_pass(
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::SurfaceBaseColor(
                        crate::paint::Color::BLACK,
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
                        crate::paint::Color::BLACK,
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
        _ => unreachable!("identity and structural graph states use earlier probe groups"),
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
struct DropShadowGraphSchedule {
    builder: SemanticGraphBuilder,
    source: SemanticResourceId,
    clear_root: SemanticPassId,
    capture: SemanticPassId,
    canonical_capture: SemanticPassId,
    canonicalize: SemanticPassId,
    copy_backdrop: SemanticPassId,
    rgba_blur: SemanticPassId,
    blur_horizontal: SemanticPassId,
    blur_vertical: SemanticPassId,
    colorize: SemanticPassId,
    source_over: SemanticPassId,
    present: SemanticPassId,
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
    let (root, clear_root, source, capture, canonical_source, canonical_capture) =
        declare_drop_shadow_probe_inputs(&mut builder)?;
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

    finish_drop_shadow_graph_probe(DropShadowGraphSchedule {
        builder,
        source,
        clear_root,
        capture,
        canonical_capture,
        canonicalize,
        copy_backdrop,
        rgba_blur,
        blur_horizontal,
        blur_vertical,
        colorize,
        source_over,
        present,
    })
}

#[cfg(test)]
fn declare_drop_shadow_probe_inputs(
    builder: &mut SemanticGraphBuilder,
) -> SemanticGraphProbeResult<(
    SemanticResourceId,
    SemanticPassId,
    SemanticResourceId,
    SemanticPassId,
    SemanticResourceId,
    SemanticPassId,
)> {
    let (root, clear_root) = declare_probe_root(builder, 1)?;
    graph_probe_value(builder.declare_pass(
        SemanticPassIntent::VelloCapture {
            initialization: WorkingImageInitialization::Transparent,
        },
        Vec::new(),
        Vec::new(),
        SemanticPassResult::Empty,
    ))?;
    let (source, capture) =
        declare_probe_capture(builder, SemanticResourceRole::IsolationWorkingImage, 2)?;
    let (canonical_source, canonical_capture) =
        declare_probe_capture(builder, SemanticResourceRole::CaptureWorkingImage, 1)?;
    Ok((
        root,
        clear_root,
        source,
        capture,
        canonical_source,
        canonical_capture,
    ))
}

#[cfg(test)]
fn finish_drop_shadow_graph_probe(
    schedule: DropShadowGraphSchedule,
) -> SemanticGraphProbeResult<CompletedDropShadowGraphProbe> {
    let DropShadowGraphSchedule {
        mut builder,
        source,
        clear_root,
        capture,
        canonical_capture,
        canonicalize,
        copy_backdrop,
        rgba_blur,
        blur_horizontal,
        blur_vertical,
        colorize,
        source_over,
        present,
    } = schedule;
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
    let captures_are_transparent = graph_probe_captures_are_transparent(&probe.graph);
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
        graph_probe_resources_are_spatially_complete(&probe.graph);

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

#[cfg(test)]
fn graph_probe_resources_are_spatially_complete(graph: &GpuRenderGraph) -> bool {
    graph.resources.iter().all(|resource| {
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
    })
}

#[cfg(test)]
fn graph_probe_captures_are_transparent(graph: &GpuRenderGraph) -> bool {
    graph.passes.iter().all(|pass| {
        !matches!(pass.intent, SemanticPassIntent::VelloCapture { .. })
            || matches!(
                pass.intent,
                SemanticPassIntent::VelloCapture {
                    initialization: WorkingImageInitialization::Transparent
                }
            )
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FramePlanRouteObservation {
    DirectVello,
    GpuGraph,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameSelectionRequirementObservation {
    ResolvedAlphaMask,
    BoundedBackdrop,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloSpanScopeObservation {
    CurrentParent,
    LayerSource,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloCommandObservation {
    Fill,
    Stroke,
    Shadow,
    Image,
    Text,
    LocalLayer,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VelloSpanObservation {
    pub(crate) scope: VelloSpanScopeObservation,
    pub(crate) commands: Vec<VelloCommandObservation>,
    pub(crate) captured_before_outer_semantics: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackdropDependencyObservation {
    None,
    CompletedCurrentParent,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FramePlanObservation {
    pub(crate) route: FramePlanRouteObservation,
    pub(crate) plan_count: usize,
    pub(crate) complete: bool,
    pub(crate) finite: bool,
    pub(crate) backend_free: bool,
    pub(crate) direct_command_count: usize,
    pub(crate) direct_commands: Vec<VelloCommandObservation>,
    pub(crate) output_device_extent: Option<(u32, u32)>,
    pub(crate) antialiasing: Option<crate::renderer::Antialiasing>,
    pub(crate) base_color: Option<crate::paint::Color>,
    pub(crate) selection_requirements: Vec<FrameSelectionRequirementObservation>,
    pub(crate) resolved_alpha_mask_device_extents: Vec<(u32, u32)>,
    pub(crate) vello_spans: Vec<VelloSpanObservation>,
    pub(crate) graph_layer_blends: Vec<crate::layer::BlendMode>,
    pub(crate) backdrop_dependency: BackdropDependencyObservation,
    pub(crate) current_parent_backdrop_reads: usize,
    pub(crate) stores_cloned_command_prefix: bool,
    pub(crate) captures_precede_outer_semantics: bool,
    pub(crate) graph_to_vello_reentry: bool,
    pub(crate) empty_text_resource_count: usize,
    pub(crate) resource_count: usize,
    pub(crate) pass_count: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FramePlanResultObservation {
    pub(crate) plan: Option<FramePlanObservation>,
    pub(crate) error_code: Option<crate::error::ErrorCode>,
    pub(crate) unresolved_resource: Option<crate::error::UnresolvedResourceKind>,
    pub(crate) has_partial_plan: bool,
}

#[cfg(test)]
pub(crate) fn frame_plan_result_observation_for_test(
    commands: crate::command::RenderCommands,
    surface_size: crate::geometry::Size,
    surface_scale: f64,
    antialiasing: crate::renderer::Antialiasing,
    base_color: crate::paint::Color,
) -> FramePlanResultObservation {
    let result = FrameContext::try_new(surface_size, surface_scale, antialiasing, base_color)
        .and_then(|context| commands.plan_for(context));
    match result {
        Ok(FramePlan::DirectVello(plan)) => FramePlanResultObservation {
            plan: Some(observe_direct_frame_plan(plan)),
            error_code: None,
            unresolved_resource: None,
            has_partial_plan: false,
        },
        Ok(FramePlan::GpuGraph(graph)) => FramePlanResultObservation {
            plan: Some(observe_graph_frame_plan(graph)),
            error_code: None,
            unresolved_resource: None,
            has_partial_plan: false,
        },
        Err(error) => FramePlanResultObservation {
            plan: None,
            error_code: Some(error.code()),
            unresolved_resource: error
                .unresolved_resource_diagnostic()
                .map(UnresolvedResource::kind),
            has_partial_plan: false,
        },
    }
}

#[cfg(test)]
fn observe_direct_frame_plan(plan: DirectVelloPlan) -> FramePlanObservation {
    let output_device_extent = match plan.output_mapping {
        FrameSpatialPlan::Empty(_) => None,
        FrameSpatialPlan::NonEmpty(mapping) => {
            Some((mapping.device_extent.width, mapping.device_extent.height))
        }
    };
    FramePlanObservation {
        route: FramePlanRouteObservation::DirectVello,
        plan_count: 1,
        complete: true,
        finite: frame_spatial_plan_is_finite(plan.output_mapping),
        backend_free: true,
        direct_command_count: plan.commands.commands.len(),
        direct_commands: plan
            .commands
            .commands
            .iter()
            .map(observe_vello_command)
            .collect(),
        output_device_extent,
        antialiasing: Some(plan.antialiasing),
        base_color: Some(plan.base_color),
        selection_requirements: Vec::new(),
        resolved_alpha_mask_device_extents: Vec::new(),
        vello_spans: Vec::new(),
        graph_layer_blends: Vec::new(),
        backdrop_dependency: BackdropDependencyObservation::None,
        current_parent_backdrop_reads: 0,
        stores_cloned_command_prefix: false,
        captures_precede_outer_semantics: true,
        graph_to_vello_reentry: false,
        empty_text_resource_count: count_empty_text_commands(&plan.commands.commands),
        resource_count: 0,
        pass_count: 0,
    }
}

#[cfg(test)]
fn observe_graph_frame_plan(graph: GpuRenderGraph) -> FramePlanObservation {
    let root_descriptor = graph
        .resources
        .iter()
        .find(|resource| resource.id == graph.root_working_image)
        .map(|resource| resource.descriptor);
    let selection_requirements = graph
        .selection_requirements
        .iter()
        .map(|requirement| match requirement {
            GraphSelectionRequirement::ResolvedAlphaMask => {
                FrameSelectionRequirementObservation::ResolvedAlphaMask
            }
            GraphSelectionRequirement::BoundedBackdrop => {
                FrameSelectionRequirementObservation::BoundedBackdrop
            }
        })
        .collect();
    let resolved_alpha_mask_device_extents = graph_resolved_alpha_mask_device_extents(&graph);
    let vello_spans = graph
        .vello_spans
        .iter()
        .map(|span| VelloSpanObservation {
            scope: match span.scope {
                SemanticVelloSpanScope::CurrentParent => VelloSpanScopeObservation::CurrentParent,
                SemanticVelloSpanScope::LayerSource => VelloSpanScopeObservation::LayerSource,
            },
            commands: span
                .commands
                .commands
                .iter()
                .map(observe_vello_command)
                .collect(),
            captured_before_outer_semantics: span.captured_before_outer_semantics,
        })
        .collect();
    let graph_layer_blends = graph
        .composites
        .iter()
        .filter_map(|composite| match &composite.kind {
            SemanticCompositeKind::Layer { blend, .. } => Some(*blend),
            SemanticCompositeKind::SpanSourceOver | SemanticCompositeKind::DropShadow => None,
        })
        .collect();
    let complete = graph_frame_plan_is_complete(&graph);
    let finite = graph_frame_plan_is_finite(&graph);
    let graph_to_vello_reentry = graph.passes.iter().any(|pass| {
        matches!(pass.intent, SemanticPassIntent::VelloCapture { .. }) && !pass.reads.is_empty()
    });
    let empty_text_resource_count = graph
        .vello_spans
        .iter()
        .map(|span| count_empty_text_commands(&span.commands.commands))
        .sum();
    let captures_precede_outer_semantics = graph
        .vello_spans
        .iter()
        .all(|span| span.captured_before_outer_semantics)
        && graph
            .composites
            .iter()
            .all(|composite| composite.source_captured_before_outer_semantics);
    let antialiasing = graph.vello_spans.first().map(|span| span.antialiasing);
    let base_color = graph_base_color(&graph);

    FramePlanObservation {
        route: FramePlanRouteObservation::GpuGraph,
        plan_count: 1,
        complete,
        finite,
        backend_free: true,
        direct_command_count: 0,
        direct_commands: Vec::new(),
        output_device_extent: root_descriptor.map(|descriptor| {
            (
                descriptor.device_extent.width,
                descriptor.device_extent.height,
            )
        }),
        antialiasing,
        base_color,
        selection_requirements,
        resolved_alpha_mask_device_extents,
        vello_spans,
        graph_layer_blends,
        backdrop_dependency: if graph.backdrop_reads.is_empty() {
            BackdropDependencyObservation::None
        } else {
            BackdropDependencyObservation::CompletedCurrentParent
        },
        current_parent_backdrop_reads: graph.backdrop_reads.len(),
        stores_cloned_command_prefix: false,
        captures_precede_outer_semantics,
        graph_to_vello_reentry,
        empty_text_resource_count,
        resource_count: graph.resources.len(),
        pass_count: graph.passes.len(),
    }
}

#[cfg(test)]
fn graph_base_color(graph: &GpuRenderGraph) -> Option<Color> {
    graph.passes.iter().find_map(|pass| match pass.intent {
        SemanticPassIntent::ClearRoot {
            initialization: WorkingImageInitialization::SurfaceBaseColor(color),
        } => Some(color),
        _ => None,
    })
}

#[cfg(test)]
fn graph_resolved_alpha_mask_device_extents(graph: &GpuRenderGraph) -> Vec<(u32, u32)> {
    graph
        .imports
        .iter()
        .filter_map(|import| match &import.kind {
            SemanticImportKind::ResolvedAlphaMask { .. } => graph
                .resources
                .iter()
                .find(|resource| resource.id == import.resource)
                .map(|resource| {
                    (
                        resource.descriptor.device_extent.width,
                        resource.descriptor.device_extent.height,
                    )
                }),
        })
        .collect()
}

#[cfg(test)]
fn graph_frame_plan_is_complete(graph: &GpuRenderGraph) -> bool {
    graph.passes.iter().all(|pass| pass.scheduled)
        && graph.resources.iter().all(|resource| {
            resource.producer.is_some()
                && resource.remaining_reads == Some(0)
                && resource.releasable_after.is_some()
        })
        && graph
            .passes
            .last()
            .is_some_and(|pass| pass.id == graph.final_present)
        && validate_semantic_frame_graph(graph).is_ok()
}

#[cfg(test)]
fn graph_frame_plan_is_finite(graph: &GpuRenderGraph) -> bool {
    graph
        .resources
        .iter()
        .all(|resource| semantic_resource_descriptor_is_finite(resource.descriptor))
        && graph.vello_spans.iter().all(|span| {
            span.capture_transform
                .as_array()
                .into_iter()
                .all(f64::is_finite)
                && span
                    .parent_to_surface
                    .as_array()
                    .into_iter()
                    .all(f64::is_finite)
        })
}

#[cfg(test)]
fn frame_spatial_plan_is_finite(plan: FrameSpatialPlan) -> bool {
    match plan {
        FrameSpatialPlan::Empty(plan) => logical_rect_is_finite(plan.logical_bounds.rect()),
        FrameSpatialPlan::NonEmpty(plan) => semantic_spatial_plan_is_finite(plan),
    }
}

#[cfg(test)]
fn semantic_resource_descriptor_is_finite(descriptor: SemanticResourceDescriptor) -> bool {
    logical_rect_is_finite(descriptor.logical_bounds.rect())
        && descriptor.device_extent.width > 0
        && descriptor.device_extent.height > 0
        && descriptor.texel_center_mapping.origin.x().is_finite()
        && descriptor.texel_center_mapping.origin.y().is_finite()
        && descriptor
            .texel_center_mapping
            .raster_scale
            .get()
            .is_finite()
}

#[cfg(test)]
fn semantic_spatial_plan_is_finite(plan: NonEmptyFrameSpatialPlan) -> bool {
    semantic_resource_descriptor_is_finite(SemanticResourceDescriptor::new(
        SemanticResourceRole::CaptureWorkingImage,
        plan,
        0,
    ))
}

#[cfg(test)]
fn logical_rect_is_finite(rect: Rect) -> bool {
    [rect.x(), rect.y(), rect.width(), rect.height()]
        .into_iter()
        .all(f64::is_finite)
}

#[cfg(test)]
fn count_empty_text_commands(commands: &[RenderCommand]) -> usize {
    commands
        .iter()
        .map(|command| match command {
            RenderCommand::TextRun { bounds, .. } if bounds.kind() == TextRunBoundsKind::Empty => 1,
            RenderCommand::Layer { children, .. } => count_empty_text_commands(children),
            RenderCommand::Fill { .. }
            | RenderCommand::Stroke { .. }
            | RenderCommand::Shadow { .. }
            | RenderCommand::Image { .. }
            | RenderCommand::TextRun { .. } => 0,
        })
        .sum()
}

#[cfg(test)]
fn observe_vello_command(command: &crate::command::RenderCommand) -> VelloCommandObservation {
    match command {
        crate::command::RenderCommand::Fill { .. } => VelloCommandObservation::Fill,
        crate::command::RenderCommand::Stroke { .. } => VelloCommandObservation::Stroke,
        crate::command::RenderCommand::Shadow { .. } => VelloCommandObservation::Shadow,
        crate::command::RenderCommand::Image { .. } => VelloCommandObservation::Image,
        crate::command::RenderCommand::TextRun { .. } => VelloCommandObservation::Text,
        crate::command::RenderCommand::Layer { .. } => VelloCommandObservation::LocalLayer,
    }
}
