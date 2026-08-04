mod bounds;
mod filter;

pub(crate) use bounds::LogicalBounds;
use bounds::{
    DestinationToLayerLocalMapping, FrameSpatialPlan, NonEmptyFrameSpatialPlan,
    NonEmptyLogicalBounds, PositiveDeviceExtent, SemanticContributionDomain, SemanticSourceBounds,
    SemanticSourceContribution, SignedDeviceOrigin, TexelCenterMapping,
    destination_to_layer_local_mapping, mask_upload_spatial,
};
use filter::{
    DropShadowAlphaSource, DropShadowOffsetSampling, DropShadowSourceComposition, FilterEdgePolicy,
    FilterSourceRole, ResolvedFilterOperationIntent, ResolvedFilterSpatialMapping,
    ResolvedFilterStep, ResolvedFrameFilterPlan,
};

use super::{
    command::{
        LayerIsolation, NormalizedLayer, RenderClip, RenderCommand, RenderCommands, RenderLayerMask,
    },
    error::{BackendErrorCode, Error, Result, UnresolvedResource, UnresolvedResourceKind},
    filter::ColorClampBoundary,
    geometry::{PhysicalSize, Point, Rect, Size, Transform},
    image::{Extend, ImageQuality, ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey},
    paint::Color,
    renderer::Antialiasing,
    style::{ColorFilterOp, FilterList},
    text::TextRunBoundsKind,
};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use super::command::LayerPassPlan;

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
        SemanticFrameGraphPlanner::build(commands, context, output_spatial, selection_requirements)
            .map(Self::GpuGraph)
    }
}

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
    SemanticFrameGraphPlanner::build_with_capture_mapping(
        commands,
        context,
        output_spatial,
        Vec::new(),
        mapping.capture_transform,
        mapping.parent_to_surface,
        CaptureBoundsCoordinateSpace::ForcedMappedForTest,
    )
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
    SemanticFrameGraphPlanner::build_authored_filter_fixture(filters, commands, context)
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

#[cfg(test)]
static NEXT_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);

type GraphBuildResult<T> = std::result::Result<T, GraphValidationError>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GraphGeneration(u64);

impl GraphGeneration {
    const FRAME_PLAN: Self = Self(1);

    #[cfg(test)]
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
enum SemanticResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    ClipCoverage,
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
enum WorkingImageInitialization {
    SurfaceBaseColor(super::paint::Color),
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticVelloSpan {
    capture_pass: SemanticPassId,
    scope: SemanticVelloSpanScope,
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
    captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticClipCoverage {
    capture_pass: SemanticPassId,
    elements: Vec<SemanticOuterClip>,
    antialiasing: Antialiasing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SemanticResolvedAlphaMaskComposition {
    resource: SemanticResourceId,
    bounds: Rect,
    image_dimensions: PhysicalSize,
    quality: ImageQuality,
    extend: Extend,
}

#[derive(Clone, Debug, PartialEq)]
enum SemanticCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        destination_to_layer_local: DestinationToLayerLocalMapping,
        opacity: f32,
        blend: super::layer::BlendMode,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<SemanticOuterClip>,
        clip_coverage: Option<SemanticResourceId>,
        alpha_mask: Option<Box<SemanticResolvedAlphaMaskComposition>>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticCompositePlan {
    pass: SemanticPassId,
    kind: SemanticCompositeKind,
    source_captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticFilterStepPlan {
    passes: Vec<SemanticPassId>,
    step: ResolvedFilterStep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticBackdropRead {
    pass: SemanticPassId,
    completed_parent: SemanticResourceId,
}

#[derive(Clone, Debug, PartialEq)]
enum SemanticImportKind {
    ResolvedAlphaMask {
        upload: ResolvedMaskUploadDescriptor,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticImportPlan {
    resource: SemanticResourceId,
    kind: SemanticImportKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GpuRenderGraph {
    generation: GraphGeneration,
    resources: Vec<SemanticGraphResource>,
    passes: Vec<SemanticGraphPass>,
    root_working_image: SemanticResourceId,
    final_present: SemanticPassId,
    selection_requirements: Vec<GraphSelectionRequirement>,
    vello_spans: Vec<SemanticVelloSpan>,
    clip_coverages: Vec<SemanticClipCoverage>,
    composites: Vec<SemanticCompositePlan>,
    filter_steps: Vec<SemanticFilterStepPlan>,
    backdrop_reads: Vec<SemanticBackdropRead>,
    imports: Vec<SemanticImportPlan>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphLoweringGeneration(u64);

impl GraphLoweringGeneration {
    const fn from_semantic(generation: GraphGeneration) -> Self {
        Self(generation.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphLoweringResourceId {
    generation: GraphLoweringGeneration,
    index: u32,
}

impl GraphLoweringResourceId {
    const fn from_semantic(id: SemanticResourceId) -> Self {
        Self {
            generation: GraphLoweringGeneration::from_semantic(id.generation),
            index: id.index.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GraphLoweringPassId {
    generation: GraphLoweringGeneration,
    index: u32,
}

impl GraphLoweringPassId {
    const fn from_semantic(id: SemanticPassId) -> Self {
        Self {
            generation: GraphLoweringGeneration::from_semantic(id.generation),
            index: id.index.0,
        }
    }

    #[cfg(test)]
    pub(crate) fn stale_generation_for_test(self) -> Option<Self> {
        self.generation.0.checked_add(1).map(|generation| Self {
            generation: GraphLoweringGeneration(generation),
            index: self.index,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    ClipCoverage,
    IsolationWorkingImage,
    ImportedImage,
    BackdropCopy,
    FilterIntermediate,
    ShadowImage,
    CompositeResult,
}

impl GraphLoweringResourceRole {
    const fn from_semantic(role: SemanticResourceRole) -> Self {
        match role {
            SemanticResourceRole::RootWorkingImage => Self::RootWorkingImage,
            SemanticResourceRole::CaptureWorkingImage => Self::CaptureWorkingImage,
            SemanticResourceRole::ClipCoverage => Self::ClipCoverage,
            SemanticResourceRole::IsolationWorkingImage => Self::IsolationWorkingImage,
            SemanticResourceRole::ImportedImage => Self::ImportedImage,
            SemanticResourceRole::BackdropCopy => Self::BackdropCopy,
            SemanticResourceRole::FilterIntermediate => Self::FilterIntermediate,
            SemanticResourceRole::ShadowImage => Self::ShadowImage,
            SemanticResourceRole::CompositeResult => Self::CompositeResult,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringSpatialDescriptor {
    logical_bounds: Rect,
    device_origin: (i32, i32),
    device_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl GraphLoweringSpatialDescriptor {
    const fn from_semantic(descriptor: SemanticResourceDescriptor) -> Self {
        Self {
            logical_bounds: descriptor.logical_bounds.rect(),
            device_origin: (descriptor.device_origin.x, descriptor.device_origin.y),
            device_extent: PhysicalSize::new(
                descriptor.device_extent.width,
                descriptor.device_extent.height,
            ),
            texel_origin: descriptor.texel_center_mapping.origin,
            raster_scale: descriptor.texel_center_mapping.raster_scale.get(),
        }
    }

    #[must_use]
    pub(crate) const fn logical_bounds(self) -> Rect {
        self.logical_bounds
    }

    #[must_use]
    pub(crate) const fn device_origin(self) -> (i32, i32) {
        self.device_origin
    }

    #[must_use]
    pub(crate) const fn device_extent(self) -> PhysicalSize {
        self.device_extent
    }

    #[must_use]
    pub(crate) const fn texel_origin(self) -> Point {
        self.texel_origin
    }

    #[must_use]
    pub(crate) const fn raster_scale(self) -> f64 {
        self.raster_scale
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringResourceProducer {
    Imported,
    Pass(GraphLoweringPassId),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GraphLoweringImportView<'graph> {
    ResolvedAlphaMask(&'graph ResolvedMaskUploadDescriptor),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphLoweringResourceView<'graph> {
    resource: &'graph SemanticGraphResource,
    import: Option<&'graph SemanticImportPlan>,
}

impl<'graph> GraphLoweringResourceView<'graph> {
    #[must_use]
    pub(crate) const fn id(self) -> GraphLoweringResourceId {
        GraphLoweringResourceId::from_semantic(self.resource.id)
    }

    #[must_use]
    pub(crate) const fn role(self) -> GraphLoweringResourceRole {
        GraphLoweringResourceRole::from_semantic(self.resource.descriptor.role)
    }

    #[must_use]
    pub(crate) const fn spatial(self) -> GraphLoweringSpatialDescriptor {
        GraphLoweringSpatialDescriptor::from_semantic(self.resource.descriptor)
    }

    #[must_use]
    pub(crate) fn producer(self) -> GraphLoweringResourceProducer {
        match self.resource.producer {
            Some(SemanticResourceProducer::Imported) => GraphLoweringResourceProducer::Imported,
            Some(SemanticResourceProducer::Pass(pass)) => {
                GraphLoweringResourceProducer::Pass(GraphLoweringPassId::from_semantic(pass))
            }
            None => unreachable!("validated lowering resources always have a producer"),
        }
    }

    #[must_use]
    pub(crate) const fn expected_reads(self) -> u32 {
        self.resource.descriptor.expected_reads
    }

    #[must_use]
    pub(crate) fn last_use(self) -> GraphLoweringPassId {
        match self.resource.releasable_after {
            Some(pass) => GraphLoweringPassId::from_semantic(pass),
            None => unreachable!("validated lowering resources always have a last use"),
        }
    }

    #[must_use]
    pub(crate) fn import(self) -> Option<GraphLoweringImportView<'graph>> {
        self.import.map(|import| match &import.kind {
            SemanticImportKind::ResolvedAlphaMask { upload } => {
                GraphLoweringImportView::ResolvedAlphaMask(upload)
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringInitialization {
    SurfaceBaseColor,
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringVelloSpan {
    scope: GraphLoweringVelloSpanScope,
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
    captured_before_outer_semantics: bool,
}

impl GraphLoweringVelloSpan {
    #[must_use]
    pub(crate) const fn scope(&self) -> GraphLoweringVelloSpanScope {
        self.scope
    }

    #[must_use]
    pub(crate) fn commands(&self) -> &RenderCommands {
        &self.commands
    }

    #[must_use]
    pub(crate) const fn capture_transform(&self) -> Transform {
        self.capture_transform
    }

    #[must_use]
    pub(crate) const fn parent_to_surface(&self) -> Transform {
        self.parent_to_surface
    }

    #[must_use]
    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }

    #[must_use]
    pub(crate) const fn captured_before_outer_semantics(&self) -> bool {
        self.captured_before_outer_semantics
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringClipCoverageElement {
    clip: RenderClip,
    transform: Transform,
}

impl GraphLoweringClipCoverageElement {
    #[must_use]
    pub(crate) const fn clip(&self) -> &RenderClip {
        &self.clip
    }

    #[must_use]
    pub(crate) const fn transform(&self) -> Transform {
        self.transform
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringClipCoverage {
    elements: Vec<GraphLoweringClipCoverageElement>,
    antialiasing: Antialiasing,
}

impl GraphLoweringClipCoverage {
    #[must_use]
    pub(crate) fn elements(&self) -> &[GraphLoweringClipCoverageElement] {
        &self.elements
    }

    #[must_use]
    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GraphLoweringVelloCapture {
    Span(GraphLoweringVelloSpan),
    ClipCoverage(GraphLoweringClipCoverage),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GraphLoweringEdgePolicy {
    NoSampling,
    TransparentBlack,
    SemanticBorderMirror(Rect),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringFilterSpatialMapping {
    source: GraphLoweringSpatialDescriptor,
    result: GraphLoweringSpatialDescriptor,
}

impl GraphLoweringFilterSpatialMapping {
    #[must_use]
    pub(crate) const fn source(self) -> GraphLoweringSpatialDescriptor {
        self.source
    }

    #[must_use]
    pub(crate) const fn result(self) -> GraphLoweringSpatialDescriptor {
        self.result
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringColorOperation {
    operation: ColorFilterOp,
    clamp_boundary: ColorClampBoundary,
}

impl GraphLoweringColorOperation {
    #[must_use]
    pub(crate) const fn operation(self) -> ColorFilterOp {
        self.operation
    }

    #[must_use]
    pub(crate) const fn clamp_boundary(self) -> ColorClampBoundary {
        self.clamp_boundary
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringColorFilter {
    operations: Vec<GraphLoweringColorOperation>,
    spatial: GraphLoweringFilterSpatialMapping,
    edge: GraphLoweringEdgePolicy,
}

impl GraphLoweringColorFilter {
    #[must_use]
    pub(crate) fn operations(&self) -> &[GraphLoweringColorOperation] {
        &self.operations
    }

    #[must_use]
    pub(crate) const fn spatial(&self) -> GraphLoweringFilterSpatialMapping {
        self.spatial
    }

    #[must_use]
    pub(crate) const fn edge(&self) -> GraphLoweringEdgePolicy {
        self.edge
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringBlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringBlur {
    input: GraphLoweringBlurInput,
    standard_deviation: f64,
    support_radius: u32,
    spatial: GraphLoweringFilterSpatialMapping,
    edge: GraphLoweringEdgePolicy,
}

impl GraphLoweringBlur {
    #[must_use]
    pub(crate) const fn input(self) -> GraphLoweringBlurInput {
        self.input
    }

    #[must_use]
    pub(crate) const fn standard_deviation(self) -> f64 {
        self.standard_deviation
    }

    #[must_use]
    pub(crate) const fn support_radius(self) -> u32 {
        self.support_radius
    }

    #[must_use]
    pub(crate) const fn spatial(self) -> GraphLoweringFilterSpatialMapping {
        self.spatial
    }

    #[must_use]
    pub(crate) const fn edge(self) -> GraphLoweringEdgePolicy {
        self.edge
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringDropShadow {
    offset: Point,
    standard_deviation: f64,
    color: Color,
    support_radius: u32,
    spatial: GraphLoweringFilterSpatialMapping,
    edge: GraphLoweringEdgePolicy,
    source_alpha: bool,
    continuous_offset: bool,
    retains_unchanged_source: bool,
}

impl GraphLoweringDropShadow {
    #[must_use]
    pub(crate) const fn offset(self) -> Point {
        self.offset
    }

    #[must_use]
    pub(crate) const fn standard_deviation(self) -> f64 {
        self.standard_deviation
    }

    #[must_use]
    pub(crate) const fn color(self) -> Color {
        self.color
    }

    #[must_use]
    pub(crate) const fn support_radius(self) -> u32 {
        self.support_radius
    }

    #[must_use]
    pub(crate) const fn spatial(self) -> GraphLoweringFilterSpatialMapping {
        self.spatial
    }

    #[must_use]
    pub(crate) const fn edge(self) -> GraphLoweringEdgePolicy {
        self.edge
    }

    #[must_use]
    pub(crate) const fn uses_source_alpha(self) -> bool {
        self.source_alpha
    }

    #[must_use]
    pub(crate) const fn uses_continuous_offset(self) -> bool {
        self.continuous_offset
    }

    #[must_use]
    pub(crate) const fn retains_unchanged_source(self) -> bool {
        self.retains_unchanged_source
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringOuterClip {
    clip: RenderClip,
    transform: Transform,
}

impl GraphLoweringOuterClip {
    #[must_use]
    pub(crate) const fn clip(&self) -> &RenderClip {
        &self.clip
    }

    #[must_use]
    pub(crate) const fn transform(&self) -> Transform {
        self.transform
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringDestinationToLayerLocal {
    affine: Transform,
}

impl GraphLoweringDestinationToLayerLocal {
    #[must_use]
    pub(crate) const fn affine(self) -> Transform {
        self.affine
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringResolvedAlphaMaskComposition {
    resource: GraphLoweringResourceId,
    bounds: Rect,
    image_dimensions: PhysicalSize,
    quality: ImageQuality,
    extend: Extend,
}

impl GraphLoweringResolvedAlphaMaskComposition {
    #[must_use]
    pub(crate) const fn resource(self) -> GraphLoweringResourceId {
        self.resource
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> Rect {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn image_dimensions(self) -> PhysicalSize {
        self.image_dimensions
    }

    #[must_use]
    pub(crate) const fn quality(self) -> ImageQuality {
        self.quality
    }

    #[must_use]
    pub(crate) const fn extend(self) -> Extend {
        self.extend
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GraphLoweringCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        destination_to_layer_local: GraphLoweringDestinationToLayerLocal,
        opacity: f32,
        blend: super::layer::BlendMode,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<GraphLoweringOuterClip>,
        clip_coverage: Option<GraphLoweringResourceId>,
        alpha_mask: Option<Box<GraphLoweringResolvedAlphaMaskComposition>>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphLoweringComposite {
    kind: GraphLoweringCompositeKind,
    source_captured_before_outer_semantics: bool,
}

impl GraphLoweringComposite {
    #[must_use]
    pub(crate) const fn kind(&self) -> &GraphLoweringCompositeKind {
        &self.kind
    }

    #[must_use]
    pub(crate) const fn source_captured_before_outer_semantics(&self) -> bool {
        self.source_captured_before_outer_semantics
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GraphLoweringPassKind {
    ClearRoot {
        initialization: GraphLoweringInitialization,
        color: Color,
    },
    VelloCapture(Option<GraphLoweringVelloCapture>),
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter(Option<GraphLoweringColorFilter>),
    BlurHorizontal(Option<GraphLoweringBlur>),
    BlurVertical(Option<GraphLoweringBlur>),
    DropShadowColorize(Option<GraphLoweringDropShadow>),
    Composite(Option<GraphLoweringComposite>),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringReadRole {
    CaptureSource,
    CompletedParent,
    FilterSource,
    BlurredSourceAlpha,
    CompositeParent,
    CompositeSource,
    ClipCoverage,
    AlphaMask,
    Shadow,
    FinalWorkingImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringSamplingFilter {
    Nearest,
    Linear,
    GaussianKernel,
    ImportedMask,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GraphLoweringSamplingEdge {
    ClampToExtent,
    TransparentBlack,
    SemanticBorderMirror(Rect),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GraphLoweringReadBinding {
    role: GraphLoweringReadRole,
    resource: GraphLoweringResourceId,
    sampling_filter: GraphLoweringSamplingFilter,
    sampling_edge: GraphLoweringSamplingEdge,
}

impl GraphLoweringReadBinding {
    #[must_use]
    pub(crate) const fn role(self) -> GraphLoweringReadRole {
        self.role
    }

    #[must_use]
    pub(crate) const fn resource(self) -> GraphLoweringResourceId {
        self.resource
    }

    #[must_use]
    pub(crate) const fn sampling_filter(self) -> GraphLoweringSamplingFilter {
        self.sampling_filter
    }

    #[must_use]
    pub(crate) const fn sampling_edge(self) -> GraphLoweringSamplingEdge {
        self.sampling_edge
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphLoweringPassResult {
    Empty,
    Resource(GraphLoweringResourceId),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphLoweringPassView<'graph> {
    graph: &'graph GpuRenderGraph,
    pass: &'graph SemanticGraphPass,
}

impl GraphLoweringPassView<'_> {
    #[must_use]
    pub(crate) const fn id(self) -> GraphLoweringPassId {
        GraphLoweringPassId::from_semantic(self.pass.id)
    }

    #[must_use]
    pub(crate) fn dependencies(self) -> Vec<GraphLoweringPassId> {
        self.pass
            .dependencies
            .iter()
            .copied()
            .map(GraphLoweringPassId::from_semantic)
            .collect()
    }

    pub(crate) fn kind(self) -> Result<GraphLoweringPassKind> {
        graph_build(graph_lowering_pass_kind(self.graph, self.pass))
    }

    pub(crate) fn reads(self) -> Result<Vec<GraphLoweringReadBinding>> {
        graph_build(graph_lowering_read_bindings(self.graph, self.pass))
    }

    #[must_use]
    pub(crate) const fn result(self) -> GraphLoweringPassResult {
        match self.pass.result {
            SemanticPassResult::Empty => GraphLoweringPassResult::Empty,
            SemanticPassResult::Resource(resource) => {
                GraphLoweringPassResult::Resource(GraphLoweringResourceId::from_semantic(resource))
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphLoweringView<'graph> {
    graph: &'graph GpuRenderGraph,
}

impl<'graph> GraphLoweringView<'graph> {
    #[must_use]
    pub(crate) const fn generation(self) -> GraphLoweringGeneration {
        GraphLoweringGeneration::from_semantic(self.graph.generation)
    }

    #[must_use]
    pub(crate) const fn root_working_image(self) -> GraphLoweringResourceId {
        GraphLoweringResourceId::from_semantic(self.graph.root_working_image)
    }

    #[must_use]
    pub(crate) const fn final_present(self) -> GraphLoweringPassId {
        GraphLoweringPassId::from_semantic(self.graph.final_present)
    }

    #[must_use]
    pub(crate) fn resources(self) -> Vec<GraphLoweringResourceView<'graph>> {
        self.graph
            .resources
            .iter()
            .map(|resource| GraphLoweringResourceView {
                resource,
                import: self
                    .graph
                    .imports
                    .iter()
                    .find(|import| import.resource == resource.id),
            })
            .collect()
    }

    #[must_use]
    pub(crate) fn passes(self) -> Vec<GraphLoweringPassView<'graph>> {
        self.graph
            .passes
            .iter()
            .map(|pass| GraphLoweringPassView {
                graph: self.graph,
                pass,
            })
            .collect()
    }
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
    pub(crate) fn lowering_view(&self) -> Result<GraphLoweringView<'_>> {
        graph_build(validate_graph_for_lowering(self))?;
        Ok(GraphLoweringView { graph: self })
    }

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

#[derive(Debug)]
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
    #[cfg(test)]
    fn try_new() -> GraphBuildResult<Self> {
        Self::with_generation(GraphGeneration::try_next()?)
    }

    fn for_frame_plan() -> GraphBuildResult<Self> {
        Self::with_generation(GraphGeneration::FRAME_PLAN)
    }

    fn with_generation(generation: GraphGeneration) -> GraphBuildResult<Self> {
        Ok(Self {
            generation,
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

        self.validate_recorded_dependencies(&dependencies)?;
        let read_indices = self.validate_recorded_reads(&dependencies, &reads)?;

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

    fn validate_recorded_dependencies(
        &self,
        dependencies: &[SemanticPassId],
    ) -> GraphBuildResult<()> {
        let mut seen = Vec::with_capacity(dependencies.len());
        for dependency in dependencies {
            if seen.contains(dependency) {
                return Err(GraphValidationError::DuplicateDependency(*dependency));
            }
            self.validate_recorded_dependency(*dependency)?;
            seen.push(*dependency);
        }
        Ok(())
    }

    fn validate_recorded_reads(
        &self,
        dependencies: &[SemanticPassId],
        reads: &[SemanticResourceId],
    ) -> GraphBuildResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(reads.len());
        let mut seen = Vec::with_capacity(reads.len());
        for resource in reads {
            if seen.contains(resource) {
                return Err(GraphValidationError::DuplicateRead(*resource));
            }
            let resource_index = self.validate_resource_id(*resource)?;
            let graph_resource = self
                .resources
                .get(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*resource))?;
            match graph_resource.producer {
                Some(SemanticResourceProducer::Imported) => {}
                Some(SemanticResourceProducer::Pass(producer))
                    if !dependencies.contains(&producer) =>
                {
                    return Err(GraphValidationError::MissingProducerDependency {
                        resource: *resource,
                        producer,
                    });
                }
                Some(SemanticResourceProducer::Pass(_)) => {}
                None => return Err(GraphValidationError::ForwardRead(*resource)),
            }
            graph_resource
                .recorded_reads
                .checked_add(1)
                .ok_or(GraphValidationError::ReadCountOverflow(*resource))?;
            seen.push(*resource);
            indices.push(resource_index);
        }
        Ok(indices)
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
                self.validate_clear_root_shape(initialization, dependencies, reads, result)?
            }
            SemanticPassIntent::VelloCapture { initialization } => {
                self.validate_capture_shape(initialization, reads, result)?;
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
                self.validate_transform_pass_shape(intent, reads, result)?;
            }
        }
        Ok(())
    }

    fn validate_clear_root_shape(
        &self,
        initialization: WorkingImageInitialization,
        dependencies: &[SemanticPassId],
        reads: &[SemanticResourceId],
        result: SemanticPassResult,
    ) -> GraphBuildResult<()> {
        if matches!(
            initialization,
            WorkingImageInitialization::SurfaceBaseColor(_)
        ) && self.surface_base_initializations != 0
        {
            return Err(GraphValidationError::RepeatedSurfaceBaseInitialization);
        }
        if !dependencies.is_empty() || !reads.is_empty() {
            return Err(GraphValidationError::InvalidClearRootResult);
        }
        let SemanticPassResult::Resource(resource) = result else {
            return Err(GraphValidationError::InvalidClearRootResult);
        };
        let resource_index = self.validate_resource_id(resource)?;
        let Some(resource) = self.resources.get(resource_index) else {
            return Err(GraphValidationError::InvalidClearRootResult);
        };
        match (initialization, resource.descriptor.role) {
            (
                WorkingImageInitialization::SurfaceBaseColor(_),
                SemanticResourceRole::RootWorkingImage,
            )
            | (
                WorkingImageInitialization::Transparent,
                SemanticResourceRole::IsolationWorkingImage,
            ) => Ok(()),
            (WorkingImageInitialization::Transparent, _)
            | (
                WorkingImageInitialization::SurfaceBaseColor(_),
                SemanticResourceRole::IsolationWorkingImage,
            ) => Err(GraphValidationError::RootMustUseSurfaceBase),
            (WorkingImageInitialization::SurfaceBaseColor(_), _) => {
                Err(GraphValidationError::InvalidClearRootResult)
            }
        }
    }

    fn validate_capture_shape(
        &self,
        initialization: WorkingImageInitialization,
        reads: &[SemanticResourceId],
        result: SemanticPassResult,
    ) -> GraphBuildResult<()> {
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
                        | SemanticResourceRole::ClipCoverage
                        | SemanticResourceRole::IsolationWorkingImage
                )
            }) {
                return Err(GraphValidationError::InvalidCaptureResult);
            }
        }
        Ok(())
    }

    fn validate_transform_pass_shape(
        &self,
        intent: SemanticPassIntent,
        reads: &[SemanticResourceId],
        result: SemanticPassResult,
    ) -> GraphBuildResult<()> {
        let SemanticPassResult::Resource(resource) = result else {
            return if reads.is_empty() {
                Ok(())
            } else {
                Err(GraphValidationError::InvalidPassArity)
            };
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
            | SemanticPassIntent::BlurVertical { .. } => SemanticResourceRole::FilterIntermediate,
            _ => return Err(GraphValidationError::InvalidPassResultRole),
        };
        let resource_index = self.validate_resource_id(resource)?;
        if self
            .resources
            .get(resource_index)
            .is_none_or(|resource| resource.descriptor.role != expected_role)
        {
            return Err(GraphValidationError::InvalidPassResultRole);
        }
        Ok(())
    }

    fn seal_recorded_read_counts(&mut self) -> GraphBuildResult<()> {
        self.require_recording_phase()?;
        for resource in &mut self.resources {
            resource.descriptor.expected_reads = resource.recorded_reads;
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
            self.validate_present_schedule(id, &pass.reads)?;
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

        let read_indices = self.validate_scheduled_reads(&pass.reads)?;

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

    fn validate_present_schedule(
        &self,
        id: SemanticPassId,
        present_reads: &[SemanticResourceId],
    ) -> GraphBuildResult<()> {
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
            let required_by_present = u32::from(present_reads.contains(&resource.id));
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
        Ok(())
    }

    fn validate_scheduled_reads(
        &self,
        reads: &[SemanticResourceId],
    ) -> GraphBuildResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(reads.len());
        for resource in reads {
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
            indices.push(resource_index);
        }
        Ok(indices)
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
            selection_requirements: Vec::new(),
            vello_spans: Vec::new(),
            clip_coverages: Vec::new(),
            composites: Vec::new(),
            filter_steps: Vec::new(),
            backdrop_reads: Vec::new(),
            imports: Vec::new(),
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
struct PlannedGraphResource {
    id: SemanticResourceId,
    producer: Option<SemanticPassId>,
    logical_bounds: NonEmptyLogicalBounds,
    spatial: NonEmptyFrameSpatialPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlannedGraphParent {
    current: PlannedGraphResource,
    spatial: NonEmptyFrameSpatialPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureParentLocality {
    External,
    CaptureLocal,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticOuterClip {
    clip: RenderClip,
    transform: Transform,
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticCommandPlanningState {
    parent_locality: CaptureParentLocality,
    span_scope: SemanticVelloSpanScope,
    capture_transform: Transform,
    parent_to_surface: Transform,
    outer_clips: Vec<SemanticOuterClip>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureBoundsCoordinateSpace {
    Local,
    #[cfg(test)]
    ForcedMappedForTest,
}

impl CaptureBoundsCoordinateSpace {
    fn resolve(
        self,
        bounds: NonEmptyLogicalBounds,
        _transform: Transform,
    ) -> Result<Option<NonEmptyLogicalBounds>> {
        match self {
            Self::Local => Ok(Some(bounds)),
            #[cfg(test)]
            Self::ForcedMappedForTest => {
                match LogicalBounds::NonEmpty(bounds)
                    .try_transform(_transform, "forced C08 mapped capture bounds")?
                {
                    LogicalBounds::Empty(_) => Ok(None),
                    LogicalBounds::NonEmpty(bounds) => Ok(Some(bounds)),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VelloSpanFlush {
    parent: PlannedGraphParent,
    next_parent_locality: CaptureParentLocality,
}

struct SemanticFrameGraphPlanner {
    context: FrameContext,
    builder: SemanticGraphBuilder,
    selection_requirements: Vec<GraphSelectionRequirement>,
    vello_spans: Vec<SemanticVelloSpan>,
    clip_coverages: Vec<SemanticClipCoverage>,
    composites: Vec<SemanticCompositePlan>,
    filter_steps: Vec<SemanticFilterStepPlan>,
    backdrop_reads: Vec<SemanticBackdropRead>,
    imports: Vec<SemanticImportPlan>,
    resolved_mask_imports: Vec<(ResolvedMaskUploadKey, PlannedGraphResource)>,
    capture_bounds_coordinate_space: CaptureBoundsCoordinateSpace,
}

impl SemanticFrameGraphPlanner {
    fn build(
        commands: RenderCommands,
        context: FrameContext,
        output_spatial: NonEmptyFrameSpatialPlan,
        selection_requirements: Vec<GraphSelectionRequirement>,
    ) -> Result<GpuRenderGraph> {
        Self::build_with_capture_mapping(
            commands,
            context,
            output_spatial,
            selection_requirements,
            Transform::identity(),
            Transform::identity(),
            CaptureBoundsCoordinateSpace::Local,
        )
    }

    fn build_with_capture_mapping(
        commands: RenderCommands,
        context: FrameContext,
        output_spatial: NonEmptyFrameSpatialPlan,
        selection_requirements: Vec<GraphSelectionRequirement>,
        capture_transform: Transform,
        parent_to_surface: Transform,
        capture_bounds_coordinate_space: CaptureBoundsCoordinateSpace,
    ) -> Result<GpuRenderGraph> {
        let mut planner = Self {
            context,
            builder: graph_build(SemanticGraphBuilder::for_frame_plan())?,
            selection_requirements,
            vello_spans: Vec::new(),
            clip_coverages: Vec::new(),
            composites: Vec::new(),
            filter_steps: Vec::new(),
            backdrop_reads: Vec::new(),
            imports: Vec::new(),
            resolved_mask_imports: Vec::new(),
            capture_bounds_coordinate_space,
        };
        let root_id = graph_build(planner.builder.declare_resource(
            SemanticResourceDescriptor::new(
                SemanticResourceRole::RootWorkingImage,
                output_spatial,
                0,
            ),
        ))?;
        let clear_root = graph_build(planner.builder.declare_pass(
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::SurfaceBaseColor(context.base_color),
            },
            Vec::new(),
            Vec::new(),
            SemanticPassResult::Resource(root_id),
        ))?;
        let root = PlannedGraphResource {
            id: root_id,
            producer: Some(clear_root),
            logical_bounds: output_spatial.logical_bounds,
            spatial: output_spatial,
        };
        let parent = PlannedGraphParent {
            current: root,
            spatial: output_spatial,
        };
        let parent = planner.plan_commands(
            commands.commands,
            parent,
            SemanticCommandPlanningState {
                parent_locality: CaptureParentLocality::External,
                span_scope: SemanticVelloSpanScope::CurrentParent,
                capture_transform,
                parent_to_surface,
                outer_clips: Vec::new(),
            },
        )?;
        let present = graph_build(planner.builder.declare_pass(
            SemanticPassIntent::Present,
            dependencies_for(&[parent.current]),
            vec![parent.current.id],
            SemanticPassResult::Empty,
        ))?;
        debug_assert_eq!(
            planner.builder.final_present,
            Some(present),
            "the declared present must be the graph's terminal pass"
        );

        graph_build(planner.builder.seal_recorded_read_counts())?;
        graph_build(planner.builder.begin_scheduling())?;
        let scheduled = planner
            .builder
            .passes
            .iter()
            .map(|pass| pass.id)
            .collect::<Vec<_>>();
        for pass in scheduled {
            graph_build(planner.builder.schedule_pass(pass))?;
        }
        let mut graph = graph_build(planner.builder.finish())?;
        graph.selection_requirements = planner.selection_requirements;
        graph.vello_spans = planner.vello_spans;
        graph.clip_coverages = planner.clip_coverages;
        graph.composites = planner.composites;
        graph.filter_steps = planner.filter_steps;
        graph.backdrop_reads = planner.backdrop_reads;
        graph.imports = planner.imports;
        graph_build(validate_semantic_frame_graph(&graph))?;
        Ok(graph)
    }

    #[cfg(test)]
    fn build_authored_filter_fixture(
        filters: Vec<FilterList>,
        commands: RenderCommands,
        context: FrameContext,
    ) -> Result<GpuRenderGraph> {
        if filters.is_empty() {
            return Err(Error::invalid_value(
                "authored filter fixture",
                0,
                "must begin with at least one authored FilterList",
            ));
        }
        let output_spatial = match context.output_spatial_plan()? {
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
            FrameSpatialPlan::Empty(_) => {
                return Err(Error::invalid_value(
                    "authored filter fixture output bounds",
                    "empty",
                    "must be non-empty before the private graph fixture is planned",
                ));
            }
        };
        let mut planner = Self {
            context,
            builder: graph_build(SemanticGraphBuilder::for_frame_plan())?,
            selection_requirements: Vec::new(),
            vello_spans: Vec::new(),
            clip_coverages: Vec::new(),
            composites: Vec::new(),
            filter_steps: Vec::new(),
            backdrop_reads: Vec::new(),
            imports: Vec::new(),
            resolved_mask_imports: Vec::new(),
            capture_bounds_coordinate_space: CaptureBoundsCoordinateSpace::Local,
        };
        let root_id = graph_build(planner.builder.declare_resource(
            SemanticResourceDescriptor::new(
                SemanticResourceRole::RootWorkingImage,
                output_spatial,
                0,
            ),
        ))?;
        let clear_root = graph_build(planner.builder.declare_pass(
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::SurfaceBaseColor(context.base_color),
            },
            Vec::new(),
            Vec::new(),
            SemanticPassResult::Resource(root_id),
        ))?;
        let root = PlannedGraphResource {
            id: root_id,
            producer: Some(clear_root),
            logical_bounds: output_spatial.logical_bounds,
            spatial: output_spatial,
        };
        let parent = PlannedGraphParent {
            current: root,
            spatial: output_spatial,
        };
        let mut source = planner
            .plan_layer_source(commands.commands, Transform::identity())?
            .ok_or_else(|| {
                Error::invalid_value(
                    "authored filter fixture capture",
                    "empty",
                    "must contain ordinary capture input",
                )
            })?;
        for authored in &filters {
            source = planner.apply_filter_list(
                source,
                authored,
                FilterSourceRole::Ordinary,
                Transform::identity(),
            )?;
        }
        let parent = planner.composite_into_parent(
            parent,
            source,
            &[],
            SemanticCompositeKind::Layer {
                transform: Transform::identity(),
                destination_to_layer_local: DestinationToLayerLocalMapping {
                    affine: Transform::identity(),
                },
                opacity: 1.0,
                blend: super::layer::BlendMode::Normal,
                clip: None,
                outer_clips: Vec::new(),
                clip_coverage: None,
                alpha_mask: None,
            },
            true,
        )?;
        Self::finish_authored_filter_fixture(planner, parent)
    }

    #[cfg(test)]
    fn finish_authored_filter_fixture(
        mut planner: Self,
        parent: PlannedGraphParent,
    ) -> Result<GpuRenderGraph> {
        let present = graph_build(planner.builder.declare_pass(
            SemanticPassIntent::Present,
            dependencies_for(&[parent.current]),
            vec![parent.current.id],
            SemanticPassResult::Empty,
        ))?;
        debug_assert_eq!(planner.builder.final_present, Some(present));

        graph_build(planner.builder.seal_recorded_read_counts())?;
        graph_build(planner.builder.begin_scheduling())?;
        let scheduled = planner
            .builder
            .passes
            .iter()
            .map(|pass| pass.id)
            .collect::<Vec<_>>();
        for pass in scheduled {
            graph_build(planner.builder.schedule_pass(pass))?;
        }
        let mut graph = graph_build(planner.builder.finish())?;
        graph.vello_spans = planner.vello_spans;
        graph.clip_coverages = planner.clip_coverages;
        graph.composites = planner.composites;
        graph.filter_steps = planner.filter_steps;
        graph.backdrop_reads = planner.backdrop_reads;
        graph.imports = planner.imports;
        graph_build(validate_semantic_frame_graph(&graph))?;
        Ok(graph)
    }

    fn plan_commands(
        &mut self,
        commands: Vec<RenderCommand>,
        mut parent: PlannedGraphParent,
        mut state: SemanticCommandPlanningState,
    ) -> Result<PlannedGraphParent> {
        let mut span = Vec::new();
        for command in commands {
            if command_is_local_to_capture(&command, state.parent_locality) {
                span.push(command);
                continue;
            }

            let flush = self.flush_vello_span(span, parent, &state)?;
            parent = flush.parent;
            state.parent_locality = flush.next_parent_locality;
            span = Vec::new();
            let RenderCommand::Layer { layer, children } = command else {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "frame partition classified a non-layer command as a custom boundary",
                ));
            };
            parent = self.plan_external_layer(layer, children, parent, &state)?;
            state.parent_locality = CaptureParentLocality::External;
        }
        Ok(self.flush_vello_span(span, parent, &state)?.parent)
    }

    fn flush_vello_span(
        &mut self,
        commands: Vec<RenderCommand>,
        parent: PlannedGraphParent,
        state: &SemanticCommandPlanningState,
    ) -> Result<VelloSpanFlush> {
        let raster_transform = state.capture_transform.then(state.parent_to_surface)?;
        let contribution = SemanticSourceContribution::try_from_commands(
            commands,
            SemanticSourceBounds::exactly_empty(),
            SemanticContributionDomain::LocalUnbounded,
            self.context,
            raster_transform,
        )?;
        let Some(logical_bounds) = contribution
            .source_bounds
            .require_non_empty_for_graph("Vello span bounds")?
        else {
            return Ok(VelloSpanFlush {
                parent,
                next_parent_locality: state.parent_locality,
            });
        };
        let Some(logical_bounds) = self
            .capture_bounds_coordinate_space
            .resolve(logical_bounds, raster_transform)?
        else {
            return Ok(VelloSpanFlush {
                parent,
                next_parent_locality: state.parent_locality,
            });
        };
        let spatial = match self
            .context
            .plan_local_bounds(LogicalBounds::NonEmpty(logical_bounds), raster_transform)?
        {
            FrameSpatialPlan::Empty(_) => {
                return Ok(VelloSpanFlush {
                    parent,
                    next_parent_locality: state.parent_locality,
                });
            }
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
        };
        let capture_resource = graph_build(self.builder.declare_resource(
            SemanticResourceDescriptor::new(SemanticResourceRole::CaptureWorkingImage, spatial, 0),
        ))?;
        let capture_pass = graph_build(self.builder.declare_pass(
            SemanticPassIntent::VelloCapture {
                initialization: WorkingImageInitialization::Transparent,
            },
            Vec::new(),
            Vec::new(),
            SemanticPassResult::Resource(capture_resource),
        ))?;
        let commands = RenderCommands::new(contribution.commands);
        self.vello_spans.push(SemanticVelloSpan {
            capture_pass,
            scope: state.span_scope,
            commands,
            capture_transform: state.capture_transform,
            parent_to_surface: state.parent_to_surface,
            antialiasing: self.context.antialiasing,
            captured_before_outer_semantics: true,
        });
        let capture = PlannedGraphResource {
            id: capture_resource,
            producer: Some(capture_pass),
            logical_bounds: spatial.logical_bounds,
            spatial,
        };
        let canonical = self.declare_unary_resource_pass(
            capture,
            SemanticResourceRole::FilterIntermediate,
            spatial,
            SemanticPassIntent::CanonicalizeCapture,
        )?;
        let composite_kind = if state.outer_clips.is_empty() {
            SemanticCompositeKind::SpanSourceOver
        } else {
            SemanticCompositeKind::Layer {
                transform: Transform::identity(),
                destination_to_layer_local: DestinationToLayerLocalMapping {
                    affine: Transform::identity(),
                },
                opacity: 1.0,
                blend: super::layer::BlendMode::Normal,
                clip: None,
                outer_clips: state.outer_clips.clone(),
                clip_coverage: None,
                alpha_mask: None,
            }
        };
        let parent = self.composite_into_parent(parent, canonical, &[], composite_kind, true)?;
        Ok(VelloSpanFlush {
            parent,
            next_parent_locality: CaptureParentLocality::External,
        })
    }

    fn plan_external_layer(
        &mut self,
        layer: NormalizedLayer,
        children: Vec<RenderCommand>,
        parent: PlannedGraphParent,
        state: &SemanticCommandPlanningState,
    ) -> Result<PlannedGraphParent> {
        let layer_transform = layer.transform.then(state.capture_transform)?;
        let layer_to_surface = layer_transform.then(state.parent_to_surface)?;
        if layer.isolation == LayerIsolation::ClipOnly {
            let clip = layer.clip.clone().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "a clip-only layer reached graph planning without clip geometry",
                )
            })?;
            let mut child_state = state.clone();
            child_state.capture_transform = layer_transform;
            child_state.outer_clips.push(SemanticOuterClip {
                clip,
                transform: layer_transform,
            });
            return self.plan_commands(children, parent, child_state);
        }
        let is_transparent_wrapper = layer.clip.is_none()
            && layer.mask.is_none()
            && layer.backdrop.is_none()
            && (layer.opacity - 1.0).abs() < f32::EPSILON
            && layer.blend == super::layer::BlendMode::Normal;
        if is_transparent_wrapper {
            let mut child_state = state.clone();
            child_state.capture_transform = layer_transform;
            return self.plan_commands(children, parent, child_state);
        }

        let Some(destination_to_layer_local) =
            destination_to_layer_local_mapping(layer_to_surface)?
        else {
            return Ok(parent);
        };

        let mut source = self.plan_layer_source(children, layer_to_surface)?;
        if let Some(backdrop) = layer.backdrop.as_deref() {
            source = self.plan_backdrop_group(backdrop, source, parent, layer_to_surface)?;
        }
        let Some(source) = source else {
            return Ok(parent);
        };

        let alpha_mask = match layer.mask.as_ref() {
            Some(mask) => {
                let imported = self.import_alpha_mask(mask)?;
                Some((
                    SemanticResolvedAlphaMaskComposition {
                        resource: imported.id,
                        bounds: mask.bounds(),
                        image_dimensions: mask.upload().physical_size(),
                        quality: mask.upload().quality(),
                        extend: mask.upload().extend(),
                    },
                    imported,
                ))
            }
            None => None,
        };
        let additional_sources = alpha_mask
            .as_ref()
            .map(|(_, resource)| *resource)
            .into_iter()
            .collect::<Vec<_>>();
        self.composite_into_parent(
            parent,
            source,
            &additional_sources,
            SemanticCompositeKind::Layer {
                transform: layer_transform,
                destination_to_layer_local,
                opacity: layer.opacity,
                blend: layer.blend,
                clip: layer.clip.map(Box::new),
                outer_clips: state.outer_clips.clone(),
                clip_coverage: None,
                alpha_mask: alpha_mask.map(|(mask, _)| Box::new(mask)),
            },
            true,
        )
    }

    fn plan_layer_source(
        &mut self,
        children: Vec<RenderCommand>,
        raster_transform: Transform,
    ) -> Result<Option<PlannedGraphResource>> {
        let contribution = SemanticSourceContribution::try_from_commands(
            children,
            SemanticSourceBounds::exactly_empty(),
            SemanticContributionDomain::LocalUnbounded,
            self.context,
            raster_transform,
        )?;
        if contribution.commands.is_empty() {
            return Ok(None);
        }
        let Some(logical_bounds) = contribution
            .source_bounds
            .require_non_empty_for_graph("layer source bounds")?
        else {
            return Ok(None);
        };
        let spatial = match self
            .context
            .plan_local_bounds(LogicalBounds::NonEmpty(logical_bounds), raster_transform)?
        {
            FrameSpatialPlan::Empty(_) => return Ok(None),
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
        };
        let source_parent = self.declare_transparent_parent(spatial)?;
        let source_parent = self.plan_commands(
            contribution.commands,
            source_parent,
            SemanticCommandPlanningState {
                parent_locality: CaptureParentLocality::CaptureLocal,
                span_scope: SemanticVelloSpanScope::LayerSource,
                capture_transform: Transform::identity(),
                parent_to_surface: raster_transform,
                outer_clips: Vec::new(),
            },
        )?;
        Ok(Some(source_parent.current))
    }

    fn plan_backdrop_group(
        &mut self,
        backdrop: &super::command::RenderBackdropCapture,
        foreground: Option<PlannedGraphResource>,
        parent: PlannedGraphParent,
        raster_transform: Transform,
    ) -> Result<Option<PlannedGraphResource>> {
        let capture_bounds = LogicalBounds::try_from_rect(
            backdrop.capture_bounds().rect(),
            "backdrop capture bounds",
        )?;
        let capture_spatial = match self
            .context
            .plan_local_bounds(capture_bounds, raster_transform)?
        {
            FrameSpatialPlan::Empty(_) => return Ok(foreground),
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
        };
        let copied_id = graph_build(self.builder.declare_resource(
            SemanticResourceDescriptor::new(SemanticResourceRole::BackdropCopy, capture_spatial, 0),
        ))?;
        let copy_pass = graph_build(self.builder.declare_pass(
            SemanticPassIntent::CopyBackdrop,
            dependencies_for(&[parent.current]),
            vec![parent.current.id],
            SemanticPassResult::Resource(copied_id),
        ))?;
        self.backdrop_reads.push(SemanticBackdropRead {
            pass: copy_pass,
            completed_parent: parent.current.id,
        });
        let copied = PlannedGraphResource {
            id: copied_id,
            producer: Some(copy_pass),
            logical_bounds: capture_spatial.logical_bounds,
            spatial: capture_spatial,
        };
        let filtered = self.apply_filter_list(
            copied,
            backdrop.filters(),
            FilterSourceRole::Backdrop,
            raster_transform,
        )?;

        let mut backdrop_contribution = SemanticSourceBounds::exact_known(filtered.logical_bounds);
        if let Some(clip) = backdrop.clip() {
            backdrop_contribution = backdrop_contribution.try_intersect(
                SemanticSourceBounds::try_for_clip(clip)?,
                "post-filter backdrop clip intersection",
            )?;
        }
        let backdrop_bounds = backdrop_contribution
            .require_non_empty_for_graph("post-filter backdrop contribution bounds")?
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "post-filter backdrop contribution became empty after semantic planning",
                )
            })?;
        let group_bounds = match foreground {
            Some(foreground) => backdrop_bounds.try_union(
                foreground.logical_bounds,
                "backdrop foreground group bounds",
            )?,
            None => backdrop_bounds,
        };
        let group_spatial = match self
            .context
            .plan_local_bounds(LogicalBounds::NonEmpty(group_bounds), raster_transform)?
        {
            FrameSpatialPlan::Empty(_) => return Ok(None),
            FrameSpatialPlan::NonEmpty(spatial) => spatial,
        };
        let mut group = self.declare_transparent_parent(group_spatial)?;
        group = self.composite_into_parent(
            group,
            filtered,
            &[],
            SemanticCompositeKind::Layer {
                transform: Transform::identity(),
                destination_to_layer_local: DestinationToLayerLocalMapping {
                    affine: Transform::identity(),
                },
                opacity: 1.0,
                blend: super::layer::BlendMode::Normal,
                clip: backdrop.clip().cloned().map(Box::new),
                outer_clips: Vec::new(),
                clip_coverage: None,
                alpha_mask: None,
            },
            true,
        )?;
        if let Some(foreground) = foreground {
            group = self.composite_into_parent(
                group,
                foreground,
                &[],
                SemanticCompositeKind::SpanSourceOver,
                true,
            )?;
        }
        Ok(Some(group.current))
    }

    fn apply_filter_list(
        &mut self,
        mut source: PlannedGraphResource,
        filters: &FilterList,
        source_role: FilterSourceRole,
        raster_transform: Transform,
    ) -> Result<PlannedGraphResource> {
        let plan = self.context.plan_filter_list(
            LogicalBounds::NonEmpty(source.logical_bounds),
            raster_transform,
            filters,
            source_role,
        )?;
        let ResolvedFrameFilterPlan::NonEmpty(plan) = plan else {
            return Ok(source);
        };

        for step in plan.steps {
            let mut passes = Vec::new();
            match step.operation_intent {
                ResolvedFilterOperationIntent::ColorRun(_) => {
                    source = self.declare_unary_resource_pass(
                        source,
                        SemanticResourceRole::FilterIntermediate,
                        step.spatial_mapping.result,
                        SemanticPassIntent::ColorFilter,
                    )?;
                    passes.push(planned_resource_producer(source)?);
                }
                ResolvedFilterOperationIntent::Blur(_) => {
                    let horizontal = self.declare_unary_resource_pass(
                        source,
                        SemanticResourceRole::FilterIntermediate,
                        step.spatial_mapping.result,
                        SemanticPassIntent::BlurHorizontal {
                            input: BlurInput::Rgba,
                        },
                    )?;
                    passes.push(planned_resource_producer(horizontal)?);
                    source = self.declare_unary_resource_pass(
                        horizontal,
                        SemanticResourceRole::FilterIntermediate,
                        step.spatial_mapping.result,
                        SemanticPassIntent::BlurVertical {
                            input: BlurInput::Rgba,
                        },
                    )?;
                    passes.push(planned_resource_producer(source)?);
                }
                ResolvedFilterOperationIntent::DropShadow(_) => {
                    let horizontal = self.declare_unary_resource_pass(
                        source,
                        SemanticResourceRole::FilterIntermediate,
                        step.spatial_mapping.result,
                        SemanticPassIntent::BlurHorizontal {
                            input: BlurInput::SourceAlpha,
                        },
                    )?;
                    passes.push(planned_resource_producer(horizontal)?);
                    let vertical = self.declare_unary_resource_pass(
                        horizontal,
                        SemanticResourceRole::FilterIntermediate,
                        step.spatial_mapping.result,
                        SemanticPassIntent::BlurVertical {
                            input: BlurInput::SourceAlpha,
                        },
                    )?;
                    passes.push(planned_resource_producer(vertical)?);
                    let shadow = self.declare_unary_resource_pass(
                        vertical,
                        SemanticResourceRole::ShadowImage,
                        step.spatial_mapping.result,
                        SemanticPassIntent::DropShadowColorize,
                    )?;
                    passes.push(planned_resource_producer(shadow)?);
                    let result_id = graph_build(self.builder.declare_resource(
                        SemanticResourceDescriptor::new(
                            SemanticResourceRole::CompositeResult,
                            step.spatial_mapping.result,
                            0,
                        ),
                    ))?;
                    let merge = graph_build(self.builder.declare_pass(
                        SemanticPassIntent::Composite,
                        dependencies_for(&[source, shadow]),
                        vec![source.id, shadow.id],
                        SemanticPassResult::Resource(result_id),
                    ))?;
                    passes.push(merge);
                    self.composites.push(SemanticCompositePlan {
                        pass: merge,
                        kind: SemanticCompositeKind::DropShadow,
                        source_captured_before_outer_semantics: true,
                    });
                    source = PlannedGraphResource {
                        id: result_id,
                        producer: Some(merge),
                        logical_bounds: step.result_bounds,
                        spatial: step.spatial_mapping.result,
                    };
                }
            }
            self.filter_steps
                .push(SemanticFilterStepPlan { passes, step });
        }
        Ok(source)
    }

    fn declare_transparent_parent(
        &mut self,
        spatial: NonEmptyFrameSpatialPlan,
    ) -> Result<PlannedGraphParent> {
        let resource = graph_build(self.builder.declare_resource(
            SemanticResourceDescriptor::new(
                SemanticResourceRole::IsolationWorkingImage,
                spatial,
                0,
            ),
        ))?;
        let clear = graph_build(self.builder.declare_pass(
            SemanticPassIntent::ClearRoot {
                initialization: WorkingImageInitialization::Transparent,
            },
            Vec::new(),
            Vec::new(),
            SemanticPassResult::Resource(resource),
        ))?;
        Ok(PlannedGraphParent {
            current: PlannedGraphResource {
                id: resource,
                producer: Some(clear),
                logical_bounds: spatial.logical_bounds,
                spatial,
            },
            spatial,
        })
    }

    fn import_alpha_mask(&mut self, mask: &RenderLayerMask) -> Result<PlannedGraphResource> {
        let key = mask.upload().cache_key();
        if let Some((_, resource)) = self
            .resolved_mask_imports
            .iter()
            .find(|(existing, _)| *existing == key)
        {
            return Ok(*resource);
        }
        let spatial = mask_upload_spatial(mask.upload().physical_size())?;
        let resource = graph_build(
            self.builder
                .import_resource(SemanticResourceDescriptor::new(
                    SemanticResourceRole::ImportedImage,
                    spatial,
                    0,
                )),
        )?;
        self.imports.push(SemanticImportPlan {
            resource,
            kind: SemanticImportKind::ResolvedAlphaMask {
                upload: mask.upload().clone(),
            },
        });
        let planned = PlannedGraphResource {
            id: resource,
            producer: None,
            logical_bounds: spatial.logical_bounds,
            spatial,
        };
        self.resolved_mask_imports.push((key, planned));
        Ok(planned)
    }

    fn declare_unary_resource_pass(
        &mut self,
        source: PlannedGraphResource,
        role: SemanticResourceRole,
        spatial: NonEmptyFrameSpatialPlan,
        intent: SemanticPassIntent,
    ) -> Result<PlannedGraphResource> {
        let resource = graph_build(
            self.builder
                .declare_resource(SemanticResourceDescriptor::new(role, spatial, 0)),
        )?;
        let pass = graph_build(self.builder.declare_pass(
            intent,
            dependencies_for(&[source]),
            vec![source.id],
            SemanticPassResult::Resource(resource),
        ))?;
        Ok(PlannedGraphResource {
            id: resource,
            producer: Some(pass),
            logical_bounds: spatial.logical_bounds,
            spatial,
        })
    }

    fn declare_clip_coverage(
        &mut self,
        spatial: NonEmptyFrameSpatialPlan,
        elements: Vec<SemanticOuterClip>,
    ) -> Result<PlannedGraphResource> {
        if elements.is_empty() {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "clip coverage requires at least one ordered RenderClip",
            ));
        }
        let resource = graph_build(self.builder.declare_resource(
            SemanticResourceDescriptor::new(SemanticResourceRole::ClipCoverage, spatial, 0),
        ))?;
        let capture_pass = graph_build(self.builder.declare_pass(
            SemanticPassIntent::VelloCapture {
                initialization: WorkingImageInitialization::Transparent,
            },
            Vec::new(),
            Vec::new(),
            SemanticPassResult::Resource(resource),
        ))?;
        self.clip_coverages.push(SemanticClipCoverage {
            capture_pass,
            elements,
            antialiasing: self.context.antialiasing,
        });
        Ok(PlannedGraphResource {
            id: resource,
            producer: Some(capture_pass),
            logical_bounds: spatial.logical_bounds,
            spatial,
        })
    }

    fn composite_into_parent(
        &mut self,
        parent: PlannedGraphParent,
        source: PlannedGraphResource,
        additional_sources: &[PlannedGraphResource],
        mut kind: SemanticCompositeKind,
        source_captured_before_outer_semantics: bool,
    ) -> Result<PlannedGraphParent> {
        let clip_elements = match &kind {
            SemanticCompositeKind::Layer {
                transform,
                clip,
                outer_clips,
                clip_coverage,
                ..
            } => {
                if clip_coverage.is_some() {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "clip coverage must be owned by graph composition planning",
                    ));
                }
                let mut elements = outer_clips.clone();
                if let Some(clip) = clip {
                    elements.push(SemanticOuterClip {
                        clip: (**clip).clone(),
                        transform: *transform,
                    });
                }
                elements
            }
            SemanticCompositeKind::SpanSourceOver | SemanticCompositeKind::DropShadow => Vec::new(),
        };
        let clip_coverage = if clip_elements.is_empty() {
            None
        } else {
            Some(self.declare_clip_coverage(parent.spatial, clip_elements)?)
        };
        if let SemanticCompositeKind::Layer {
            clip_coverage: coverage,
            ..
        } = &mut kind
        {
            *coverage = clip_coverage.map(|resource| resource.id);
        }

        let mut sources = Vec::with_capacity(additional_sources.len() + 3);
        sources.push(parent.current);
        sources.push(source);
        if let Some(coverage) = clip_coverage {
            sources.push(coverage);
        }
        sources.extend_from_slice(additional_sources);
        let resource = graph_build(self.builder.declare_resource(
            SemanticResourceDescriptor::new(
                SemanticResourceRole::CompositeResult,
                parent.spatial,
                0,
            ),
        ))?;
        let pass = graph_build(self.builder.declare_pass(
            SemanticPassIntent::Composite,
            dependencies_for(&sources),
            sources.iter().map(|source| source.id).collect(),
            SemanticPassResult::Resource(resource),
        ))?;
        self.composites.push(SemanticCompositePlan {
            pass,
            kind,
            source_captured_before_outer_semantics,
        });
        Ok(PlannedGraphParent {
            current: PlannedGraphResource {
                id: resource,
                producer: Some(pass),
                logical_bounds: parent.spatial.logical_bounds,
                spatial: parent.spatial,
            },
            spatial: parent.spatial,
        })
    }
}

fn command_is_local_to_capture(
    command: &RenderCommand,
    parent_locality: CaptureParentLocality,
) -> bool {
    let RenderCommand::Layer { layer, children } = command else {
        return true;
    };
    if layer.mask.is_some() || layer.backdrop.is_some() {
        return false;
    }
    if layer.blend != super::layer::BlendMode::Normal
        && parent_locality == CaptureParentLocality::External
    {
        return false;
    }
    let child_parent_locality = if parent_locality == CaptureParentLocality::CaptureLocal
        || layer.isolation == LayerIsolation::BackendLayer
    {
        CaptureParentLocality::CaptureLocal
    } else {
        CaptureParentLocality::External
    };
    children
        .iter()
        .all(|child| command_is_local_to_capture(child, child_parent_locality))
}

fn dependencies_for(resources: &[PlannedGraphResource]) -> Vec<SemanticPassId> {
    let mut dependencies = Vec::new();
    for producer in resources.iter().filter_map(|resource| resource.producer) {
        if !dependencies.contains(&producer) {
            dependencies.push(producer);
        }
    }
    dependencies
}

fn planned_resource_producer(resource: PlannedGraphResource) -> Result<SemanticPassId> {
    resource.producer.ok_or_else(|| {
        Error::new(
            BackendErrorCode::RenderFailed,
            "a planned graph result has no producing pass",
        )
    })
}

fn graph_build<T>(result: GraphBuildResult<T>) -> Result<T> {
    result.map_err(|error| {
        Error::new(
            BackendErrorCode::RenderFailed,
            format!("semantic frame graph validation failed: {error:?}"),
        )
    })
}

fn validate_semantic_frame_graph(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    validate_vello_span_metadata(graph)?;
    validate_clip_coverage_metadata(graph)?;
    validate_backdrop_metadata(graph)?;
    validate_import_metadata(graph)?;
    if graph.passes.iter().any(|pass| {
        matches!(pass.intent, SemanticPassIntent::VelloCapture { .. }) && !pass.reads.is_empty()
    }) {
        return Err(GraphValidationError::InvalidCaptureResult);
    }
    Ok(())
}

fn validate_vello_span_metadata(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for span in &graph.vello_spans {
        let pass = graph
            .passes
            .iter()
            .find(|pass| pass.id == span.capture_pass)
            .ok_or(GraphValidationError::UnknownPass(span.capture_pass))?;
        let SemanticPassResult::Resource(capture) = pass.result else {
            return Err(GraphValidationError::InvalidCaptureResult);
        };
        if !matches!(
            pass.intent,
            SemanticPassIntent::VelloCapture {
                initialization: WorkingImageInitialization::Transparent
            }
        ) || !pass.reads.is_empty()
            || !span.captured_before_outer_semantics
        {
            return Err(GraphValidationError::InvalidCaptureResult);
        }
        let canonical_consumers = graph
            .passes
            .iter()
            .filter(|candidate| {
                candidate.intent == SemanticPassIntent::CanonicalizeCapture
                    && candidate.reads == [capture]
            })
            .count();
        if canonical_consumers != 1 {
            return Err(GraphValidationError::InvalidCaptureResult);
        }
    }
    Ok(())
}

fn validate_clip_coverage_metadata(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for coverage in &graph.clip_coverages {
        let pass = graph
            .passes
            .iter()
            .find(|pass| pass.id == coverage.capture_pass)
            .ok_or(GraphValidationError::UnknownPass(coverage.capture_pass))?;
        let SemanticPassResult::Resource(capture) = pass.result else {
            return Err(GraphValidationError::InvalidCaptureResult);
        };
        let resource = graph
            .resources
            .iter()
            .find(|resource| resource.id == capture)
            .ok_or(GraphValidationError::UnknownResource(capture))?;
        if !matches!(
            pass.intent,
            SemanticPassIntent::VelloCapture {
                initialization: WorkingImageInitialization::Transparent
            }
        ) || !pass.reads.is_empty()
            || resource.descriptor.role != SemanticResourceRole::ClipCoverage
            || coverage.elements.is_empty()
        {
            return Err(GraphValidationError::InvalidCaptureResult);
        }
        let composite_consumers = graph
            .composites
            .iter()
            .filter(|composite| {
                matches!(
                    &composite.kind,
                    SemanticCompositeKind::Layer {
                        clip_coverage: Some(resource),
                        ..
                    } if *resource == capture
                ) && graph
                    .passes
                    .iter()
                    .find(|candidate| candidate.id == composite.pass)
                    .is_some_and(|candidate| candidate.reads.contains(&capture))
            })
            .count();
        if composite_consumers != 1 {
            return Err(GraphValidationError::InvalidCaptureResult);
        }
    }
    Ok(())
}

fn validate_backdrop_metadata(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for backdrop in &graph.backdrop_reads {
        let pass = graph
            .passes
            .iter()
            .find(|pass| pass.id == backdrop.pass)
            .ok_or(GraphValidationError::UnknownPass(backdrop.pass))?;
        if pass.intent != SemanticPassIntent::CopyBackdrop
            || pass.reads != [backdrop.completed_parent]
        {
            return Err(GraphValidationError::InvalidPassArity);
        }
    }
    Ok(())
}

fn validate_import_metadata(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for import in &graph.imports {
        let resource = graph
            .resources
            .iter()
            .find(|resource| resource.id == import.resource)
            .ok_or(GraphValidationError::UnknownResource(import.resource))?;
        if resource.descriptor.role != SemanticResourceRole::ImportedImage
            || resource.producer != Some(SemanticResourceProducer::Imported)
        {
            return Err(GraphValidationError::InvalidImportedResourceRole);
        }
    }
    Ok(())
}

fn validate_graph_for_lowering(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    validate_semantic_frame_graph(graph)?;
    if graph.resources.is_empty() {
        return Err(GraphValidationError::MissingRootWorkingImage);
    }
    if graph.passes.is_empty() {
        return Err(GraphValidationError::MissingFinalPresent);
    }

    let mut actual_reads = vec![0_u32; graph.resources.len()];
    let mut last_reads = vec![None; graph.resources.len()];
    validate_lowering_resources(graph)?;
    validate_lowering_imports(graph)?;
    validate_lowering_passes(graph, &mut actual_reads, &mut last_reads)?;
    validate_lowering_lifetimes(graph, &actual_reads, &last_reads)?;
    validate_lowering_anchors(graph)
}

fn validate_lowering_resources(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for (index, resource) in graph.resources.iter().enumerate() {
        let expected_id =
            SemanticResourceId::new(graph.generation, ResourceIndex::try_from_len(index)?);
        if resource.id != expected_id {
            return if resource.id.generation != graph.generation {
                Err(GraphValidationError::WrongResourceGeneration {
                    expected: graph.generation,
                    actual: resource.id.generation,
                })
            } else {
                Err(GraphValidationError::UnknownResource(resource.id))
            };
        }
        if resource.remaining_reads != Some(0) {
            return Err(GraphValidationError::UnscheduledReads {
                resource: resource.id,
                remaining: resource.remaining_reads.unwrap_or(u32::MAX),
            });
        }
        let import_count = graph
            .imports
            .iter()
            .filter(|import| import.resource == resource.id)
            .count();
        match resource.producer {
            Some(SemanticResourceProducer::Imported)
                if resource.descriptor.role == SemanticResourceRole::ImportedImage
                    && import_count == 1 => {}
            Some(SemanticResourceProducer::Imported) => {
                return Err(GraphValidationError::InvalidImportedResourceRole);
            }
            Some(SemanticResourceProducer::Pass(producer)) => {
                if import_count != 0 {
                    return Err(GraphValidationError::InvalidImportedResourceRole);
                }
                validate_graph_pass_id(graph, producer)?;
                let producer_pass = graph
                    .passes
                    .get(producer.index.as_usize()?)
                    .ok_or(GraphValidationError::UnknownPass(producer))?;
                if producer_pass.result != SemanticPassResult::Resource(resource.id) {
                    return Err(GraphValidationError::DuplicateProducer(resource.id));
                }
            }
            None => return Err(GraphValidationError::ResourceWithoutProducer(resource.id)),
        }
    }
    Ok(())
}

fn validate_lowering_imports(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    for import in &graph.imports {
        let index = validate_graph_resource_id(graph, import.resource)?;
        let resource = graph
            .resources
            .get(index)
            .ok_or(GraphValidationError::UnknownResource(import.resource))?;
        if resource.producer != Some(SemanticResourceProducer::Imported)
            || resource.descriptor.role != SemanticResourceRole::ImportedImage
            || graph
                .imports
                .iter()
                .filter(|candidate| candidate.resource == import.resource)
                .count()
                != 1
        {
            return Err(GraphValidationError::InvalidImportedResourceRole);
        }
    }
    Ok(())
}

fn validate_lowering_passes(
    graph: &GpuRenderGraph,
    actual_reads: &mut [u32],
    last_reads: &mut [Option<SemanticPassId>],
) -> GraphBuildResult<()> {
    for (index, pass) in graph.passes.iter().enumerate() {
        let expected_id = SemanticPassId::new(graph.generation, PassIndex::try_from_len(index)?);
        if pass.id != expected_id {
            return if pass.id.generation != graph.generation {
                Err(GraphValidationError::WrongPassGeneration {
                    expected: graph.generation,
                    actual: pass.id.generation,
                })
            } else {
                Err(GraphValidationError::UnknownPass(pass.id))
            };
        }
        if !pass.scheduled {
            return Err(GraphValidationError::UnscheduledPass(pass.id));
        }

        let mut seen_dependencies = Vec::with_capacity(pass.dependencies.len());
        for dependency in &pass.dependencies {
            if seen_dependencies.contains(dependency) {
                return Err(GraphValidationError::DuplicateDependency(*dependency));
            }
            let dependency_index = validate_graph_pass_id(graph, *dependency)?;
            if dependency_index >= index {
                return Err(GraphValidationError::ForwardDependency(*dependency));
            }
            seen_dependencies.push(*dependency);
        }

        let mut seen_reads = Vec::with_capacity(pass.reads.len());
        for read in &pass.reads {
            if seen_reads.contains(read) {
                return Err(GraphValidationError::DuplicateRead(*read));
            }
            let resource_index = validate_graph_resource_id(graph, *read)?;
            let resource = graph
                .resources
                .get(resource_index)
                .ok_or(GraphValidationError::UnknownResource(*read))?;
            if pass.result == SemanticPassResult::Resource(*read) {
                return Err(GraphValidationError::ReadWriteAlias(*read));
            }
            if let Some(SemanticResourceProducer::Pass(producer)) = resource.producer {
                let producer_index = validate_graph_pass_id(graph, producer)?;
                if producer_index >= index {
                    return Err(GraphValidationError::ForwardRead(*read));
                }
                if !pass.dependencies.contains(&producer) {
                    return Err(GraphValidationError::MissingProducerDependency {
                        resource: *read,
                        producer,
                    });
                }
            }
            actual_reads[resource_index] = actual_reads[resource_index]
                .checked_add(1)
                .ok_or(GraphValidationError::ReadCountOverflow(*read))?;
            last_reads[resource_index] = Some(pass.id);
            seen_reads.push(*read);
        }
        if let SemanticPassResult::Resource(result) = pass.result {
            let resource_index = validate_graph_resource_id(graph, result)?;
            let resource = graph
                .resources
                .get(resource_index)
                .ok_or(GraphValidationError::UnknownResource(result))?;
            if resource.producer != Some(SemanticResourceProducer::Pass(pass.id)) {
                return Err(GraphValidationError::DuplicateProducer(result));
            }
        }

        graph_lowering_pass_kind(graph, pass)?;
        graph_lowering_read_bindings(graph, pass)?;
    }
    Ok(())
}

fn validate_lowering_lifetimes(
    graph: &GpuRenderGraph,
    actual_reads: &[u32],
    last_reads: &[Option<SemanticPassId>],
) -> GraphBuildResult<()> {
    for (index, resource) in graph.resources.iter().enumerate() {
        if resource.descriptor.expected_reads != actual_reads[index]
            || resource.recorded_reads != actual_reads[index]
        {
            return Err(GraphValidationError::DeclaredReadCountMismatch {
                resource: resource.id,
                declared: resource.descriptor.expected_reads,
                recorded: actual_reads[index],
            });
        }
        let Some(last_read) = last_reads[index] else {
            return Err(GraphValidationError::OrphanResult(resource.id));
        };
        if resource.releasable_after != Some(last_read) {
            return Err(GraphValidationError::UnscheduledReads {
                resource: resource.id,
                remaining: 0,
            });
        }
    }
    Ok(())
}

fn validate_lowering_anchors(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
    let root_index = validate_graph_resource_id(graph, graph.root_working_image)?;
    if graph
        .resources
        .get(root_index)
        .is_none_or(|resource| resource.descriptor.role != SemanticResourceRole::RootWorkingImage)
    {
        return Err(GraphValidationError::MissingRootWorkingImage);
    }
    let present_index = validate_graph_pass_id(graph, graph.final_present)?;
    if present_index + 1 != graph.passes.len()
        || graph
            .passes
            .get(present_index)
            .is_none_or(|pass| pass.intent != SemanticPassIntent::Present)
    {
        return Err(GraphValidationError::MissingFinalPresent);
    }
    Ok(())
}

fn validate_graph_resource_id(
    graph: &GpuRenderGraph,
    id: SemanticResourceId,
) -> GraphBuildResult<usize> {
    if id.generation != graph.generation {
        return Err(GraphValidationError::WrongResourceGeneration {
            expected: graph.generation,
            actual: id.generation,
        });
    }
    let index = id.index.as_usize()?;
    if graph.resources.get(index).is_none() {
        return Err(GraphValidationError::UnknownResource(id));
    }
    Ok(index)
}

fn validate_graph_pass_id(graph: &GpuRenderGraph, id: SemanticPassId) -> GraphBuildResult<usize> {
    if id.generation != graph.generation {
        return Err(GraphValidationError::WrongPassGeneration {
            expected: graph.generation,
            actual: id.generation,
        });
    }
    let index = id.index.as_usize()?;
    if graph.passes.get(index).is_none() {
        return Err(GraphValidationError::UnknownPass(id));
    }
    Ok(index)
}

fn graph_lowering_pass_kind(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringPassKind> {
    match pass.intent {
        SemanticPassIntent::ClearRoot { initialization } => {
            let (initialization, color) = match initialization {
                WorkingImageInitialization::SurfaceBaseColor(color) => {
                    (GraphLoweringInitialization::SurfaceBaseColor, color)
                }
                WorkingImageInitialization::Transparent => {
                    (GraphLoweringInitialization::Transparent, Color::TRANSPARENT)
                }
            };
            Ok(GraphLoweringPassKind::ClearRoot {
                initialization,
                color,
            })
        }
        SemanticPassIntent::VelloCapture { .. } => graph_lowering_vello_capture(graph, pass),
        SemanticPassIntent::CanonicalizeCapture => {
            reject_unexpected_filter_metadata(graph, pass.id)?;
            Ok(GraphLoweringPassKind::CanonicalizeCapture)
        }
        SemanticPassIntent::CopyBackdrop => {
            let count = graph
                .backdrop_reads
                .iter()
                .filter(|read| read.pass == pass.id)
                .count();
            if (pass.result == SemanticPassResult::Empty && count != 0)
                || (matches!(pass.result, SemanticPassResult::Resource(_)) && count != 1)
            {
                return Err(GraphValidationError::InvalidPassArity);
            }
            reject_unexpected_filter_metadata(graph, pass.id)?;
            Ok(GraphLoweringPassKind::CopyBackdrop)
        }
        SemanticPassIntent::ColorFilter => graph_lowering_color_filter(graph, pass),
        SemanticPassIntent::BlurHorizontal { input }
        | SemanticPassIntent::BlurVertical { input } => {
            graph_lowering_blur_pass(graph, pass, input)
        }
        SemanticPassIntent::DropShadowColorize => graph_lowering_drop_shadow_pass(graph, pass),
        SemanticPassIntent::Composite => graph_lowering_composite_pass(graph, pass),
        SemanticPassIntent::Present => {
            reject_unexpected_filter_metadata(graph, pass.id)?;
            Ok(GraphLoweringPassKind::Present)
        }
    }
}

fn graph_lowering_vello_capture(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringPassKind> {
    let spans = graph
        .vello_spans
        .iter()
        .filter(|span| span.capture_pass == pass.id)
        .collect::<Vec<_>>();
    let coverages = graph
        .clip_coverages
        .iter()
        .filter(|coverage| coverage.capture_pass == pass.id)
        .collect::<Vec<_>>();
    let work = match pass.result {
        SemanticPassResult::Empty if spans.is_empty() && coverages.is_empty() => None,
        SemanticPassResult::Resource(_) if spans.len() == 1 && coverages.is_empty() => Some(
            GraphLoweringVelloCapture::Span(graph_lowering_vello_span(spans[0])),
        ),
        SemanticPassResult::Resource(_) if spans.is_empty() && coverages.len() == 1 => Some(
            GraphLoweringVelloCapture::ClipCoverage(graph_lowering_clip_coverage(coverages[0])),
        ),
        SemanticPassResult::Empty | SemanticPassResult::Resource(_) => {
            return Err(GraphValidationError::InvalidCaptureResult);
        }
    };
    Ok(GraphLoweringPassKind::VelloCapture(work))
}

fn graph_lowering_color_filter(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringPassKind> {
    let step = filter_step_for_pass(graph, pass)?;
    let filter = match (pass.result, step) {
        (SemanticPassResult::Empty, None) => None,
        (SemanticPassResult::Resource(_), Some(step)) => {
            let ResolvedFilterOperationIntent::ColorRun(run) = &step.step.operation_intent else {
                return Err(GraphValidationError::InvalidPassArity);
            };
            if step.passes != [pass.id] {
                return Err(GraphValidationError::InvalidPassArity);
            }
            Some(GraphLoweringColorFilter {
                operations: run
                    .operations()
                    .iter()
                    .copied()
                    .map(|operation| GraphLoweringColorOperation {
                        operation: operation.operation(),
                        clamp_boundary: operation.clamp_boundary(),
                    })
                    .collect(),
                spatial: graph_lowering_filter_spatial(step.step.spatial_mapping),
                edge: graph_lowering_edge(step.step.edge_policy),
            })
        }
        (SemanticPassResult::Empty, Some(_)) | (SemanticPassResult::Resource(_), None) => {
            return Err(GraphValidationError::InvalidPassArity);
        }
    };
    Ok(GraphLoweringPassKind::ColorFilter(filter))
}

fn graph_lowering_blur_pass(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
    input: BlurInput,
) -> GraphBuildResult<GraphLoweringPassKind> {
    let step = filter_step_for_pass(graph, pass)?;
    let blur = match (pass.result, step) {
        (SemanticPassResult::Empty, None) => None,
        (SemanticPassResult::Resource(_), Some(step)) => {
            Some(graph_lowering_blur(graph, pass, input, step)?)
        }
        (SemanticPassResult::Empty, Some(_)) | (SemanticPassResult::Resource(_), None) => {
            return Err(GraphValidationError::InvalidPassArity);
        }
    };
    if matches!(pass.intent, SemanticPassIntent::BlurHorizontal { .. }) {
        Ok(GraphLoweringPassKind::BlurHorizontal(blur))
    } else {
        Ok(GraphLoweringPassKind::BlurVertical(blur))
    }
}

fn graph_lowering_drop_shadow_pass(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringPassKind> {
    let step = filter_step_for_pass(graph, pass)?;
    let shadow = match (pass.result, step) {
        (SemanticPassResult::Empty, None) => None,
        (SemanticPassResult::Resource(_), Some(step)) => {
            Some(graph_lowering_drop_shadow(graph, pass, step)?)
        }
        (SemanticPassResult::Empty, Some(_)) | (SemanticPassResult::Resource(_), None) => {
            return Err(GraphValidationError::InvalidPassArity);
        }
    };
    Ok(GraphLoweringPassKind::DropShadowColorize(shadow))
}

fn graph_lowering_composite_pass(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringPassKind> {
    let composites = graph
        .composites
        .iter()
        .filter(|composite| composite.pass == pass.id)
        .collect::<Vec<_>>();
    let composite = match pass.result {
        SemanticPassResult::Empty if composites.is_empty() => None,
        SemanticPassResult::Resource(_) if composites.len() == 1 => {
            let composite = composites[0];
            if composite.kind == SemanticCompositeKind::DropShadow {
                let step = filter_step_for_pass(graph, pass)?
                    .ok_or(GraphValidationError::InvalidPassArity)?;
                let colorize = step
                    .passes
                    .get(2)
                    .and_then(|colorize| {
                        graph
                            .passes
                            .iter()
                            .find(|candidate| candidate.id == *colorize)
                    })
                    .ok_or(GraphValidationError::InvalidPassArity)?;
                graph_lowering_drop_shadow(graph, colorize, step)?;
            } else {
                reject_unexpected_filter_metadata(graph, pass.id)?;
            }
            Some(graph_lowering_composite(composite))
        }
        SemanticPassResult::Empty | SemanticPassResult::Resource(_) => {
            return Err(GraphValidationError::InvalidPassArity);
        }
    };
    Ok(GraphLoweringPassKind::Composite(composite))
}

fn graph_lowering_vello_span(span: &SemanticVelloSpan) -> GraphLoweringVelloSpan {
    GraphLoweringVelloSpan {
        scope: match span.scope {
            SemanticVelloSpanScope::CurrentParent => GraphLoweringVelloSpanScope::CurrentParent,
            SemanticVelloSpanScope::LayerSource => GraphLoweringVelloSpanScope::LayerSource,
        },
        commands: span.commands.clone(),
        capture_transform: span.capture_transform,
        parent_to_surface: span.parent_to_surface,
        antialiasing: span.antialiasing,
        captured_before_outer_semantics: span.captured_before_outer_semantics,
    }
}

fn graph_lowering_clip_coverage(coverage: &SemanticClipCoverage) -> GraphLoweringClipCoverage {
    GraphLoweringClipCoverage {
        elements: coverage
            .elements
            .iter()
            .map(|element| GraphLoweringClipCoverageElement {
                clip: element.clip.clone(),
                transform: element.transform,
            })
            .collect(),
        antialiasing: coverage.antialiasing,
    }
}

fn filter_step_for_pass<'graph>(
    graph: &'graph GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<Option<&'graph SemanticFilterStepPlan>> {
    let matches = graph
        .filter_steps
        .iter()
        .filter(|step| step.passes.contains(&pass.id))
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(GraphValidationError::InvalidPassArity);
    }
    Ok(matches.first().copied())
}

fn reject_unexpected_filter_metadata(
    graph: &GpuRenderGraph,
    pass: SemanticPassId,
) -> GraphBuildResult<()> {
    if graph
        .filter_steps
        .iter()
        .any(|step| step.passes.contains(&pass))
    {
        return Err(GraphValidationError::InvalidPassArity);
    }
    Ok(())
}

fn graph_lowering_filter_spatial(
    spatial: ResolvedFilterSpatialMapping,
) -> GraphLoweringFilterSpatialMapping {
    GraphLoweringFilterSpatialMapping {
        source: graph_lowering_spatial(spatial.source),
        result: graph_lowering_spatial(spatial.result),
    }
}

fn graph_lowering_spatial(spatial: NonEmptyFrameSpatialPlan) -> GraphLoweringSpatialDescriptor {
    GraphLoweringSpatialDescriptor {
        logical_bounds: spatial.logical_bounds.rect(),
        device_origin: (spatial.device_origin.x, spatial.device_origin.y),
        device_extent: PhysicalSize::new(spatial.device_extent.width, spatial.device_extent.height),
        texel_origin: spatial.texel_center_mapping.origin,
        raster_scale: spatial.raster_scale.get(),
    }
}

fn graph_lowering_edge(edge: FilterEdgePolicy) -> GraphLoweringEdgePolicy {
    match edge {
        FilterEdgePolicy::NoSampling => GraphLoweringEdgePolicy::NoSampling,
        FilterEdgePolicy::TransparentBlack => GraphLoweringEdgePolicy::TransparentBlack,
        FilterEdgePolicy::SemanticBorderMirror { semantic_border } => {
            GraphLoweringEdgePolicy::SemanticBorderMirror(semantic_border.rect())
        }
    }
}

fn graph_lowering_blur(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
    input: BlurInput,
    step: &SemanticFilterStepPlan,
) -> GraphBuildResult<GraphLoweringBlur> {
    let (standard_deviation, support_radius, expected_len) =
        match (&step.step.operation_intent, input) {
            (ResolvedFilterOperationIntent::Blur(intent), BlurInput::Rgba) => (
                intent.authored_blur.radius(),
                intent.support.device_radius,
                2,
            ),
            (ResolvedFilterOperationIntent::DropShadow(intent), BlurInput::SourceAlpha) => (
                intent.authored_shadow.blur().radius(),
                intent.support.device_radius,
                4,
            ),
            (ResolvedFilterOperationIntent::ColorRun(_), _)
            | (ResolvedFilterOperationIntent::Blur(_), BlurInput::SourceAlpha)
            | (ResolvedFilterOperationIntent::DropShadow(_), BlurInput::Rgba) => {
                return Err(GraphValidationError::InvalidPassArity);
            }
        };
    let axis_offset = usize::from(matches!(
        pass.intent,
        SemanticPassIntent::BlurVertical { .. }
    ));
    if step.passes.len() != expected_len || step.passes.get(axis_offset) != Some(&pass.id) {
        return Err(GraphValidationError::InvalidPassArity);
    }
    Ok(GraphLoweringBlur {
        input: match input {
            BlurInput::Rgba => GraphLoweringBlurInput::Rgba,
            BlurInput::SourceAlpha => GraphLoweringBlurInput::SourceAlpha,
        },
        standard_deviation,
        support_radius,
        spatial: graph_lowering_pass_spatial(graph, pass)?,
        edge: graph_lowering_edge(step.step.edge_policy),
    })
}

fn graph_lowering_drop_shadow(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
    step: &SemanticFilterStepPlan,
) -> GraphBuildResult<GraphLoweringDropShadow> {
    let ResolvedFilterOperationIntent::DropShadow(intent) = step.step.operation_intent else {
        return Err(GraphValidationError::InvalidPassArity);
    };
    if step.passes.len() != 4 {
        return Err(GraphValidationError::InvalidPassArity);
    }
    Ok(GraphLoweringDropShadow {
        offset: intent.authored_shadow.offset(),
        standard_deviation: intent.authored_shadow.blur().radius(),
        color: intent.authored_shadow.color(),
        support_radius: intent.support.device_radius,
        spatial: graph_lowering_pass_spatial(graph, pass)?,
        edge: graph_lowering_edge(step.step.edge_policy),
        source_alpha: intent.alpha_source == DropShadowAlphaSource::SourceAlpha,
        continuous_offset: intent.offset_sampling == DropShadowOffsetSampling::ContinuousLinear,
        retains_unchanged_source: intent.source_composition
            == DropShadowSourceComposition::RetainUnchangedForSourceOver,
    })
}

fn graph_lowering_pass_spatial(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<GraphLoweringFilterSpatialMapping> {
    let [source] = pass.reads.as_slice() else {
        return Err(GraphValidationError::InvalidPassArity);
    };
    let SemanticPassResult::Resource(result) = pass.result else {
        return Err(GraphValidationError::InvalidPassArity);
    };
    let source = graph
        .resources
        .get(validate_graph_resource_id(graph, *source)?)
        .ok_or(GraphValidationError::UnknownResource(*source))?;
    let result = graph
        .resources
        .get(validate_graph_resource_id(graph, result)?)
        .ok_or(GraphValidationError::UnknownResource(result))?;
    Ok(GraphLoweringFilterSpatialMapping {
        source: GraphLoweringSpatialDescriptor::from_semantic(source.descriptor),
        result: GraphLoweringSpatialDescriptor::from_semantic(result.descriptor),
    })
}

fn graph_lowering_composite(composite: &SemanticCompositePlan) -> GraphLoweringComposite {
    GraphLoweringComposite {
        kind: match &composite.kind {
            SemanticCompositeKind::SpanSourceOver => GraphLoweringCompositeKind::SpanSourceOver,
            SemanticCompositeKind::Layer {
                transform,
                destination_to_layer_local,
                opacity,
                blend,
                clip,
                outer_clips,
                clip_coverage,
                alpha_mask,
            } => GraphLoweringCompositeKind::Layer {
                transform: *transform,
                destination_to_layer_local: GraphLoweringDestinationToLayerLocal {
                    affine: destination_to_layer_local.affine,
                },
                opacity: *opacity,
                blend: *blend,
                clip: clip.clone(),
                outer_clips: outer_clips
                    .iter()
                    .map(|clip| GraphLoweringOuterClip {
                        clip: clip.clip.clone(),
                        transform: clip.transform,
                    })
                    .collect(),
                clip_coverage: clip_coverage.map(GraphLoweringResourceId::from_semantic),
                alpha_mask: alpha_mask.as_deref().map(|mask| {
                    Box::new(GraphLoweringResolvedAlphaMaskComposition {
                        resource: GraphLoweringResourceId::from_semantic(mask.resource),
                        bounds: mask.bounds,
                        image_dimensions: mask.image_dimensions,
                        quality: mask.quality,
                        extend: mask.extend,
                    })
                }),
            },
            SemanticCompositeKind::DropShadow => GraphLoweringCompositeKind::DropShadow,
        },
        source_captured_before_outer_semantics: composite.source_captured_before_outer_semantics,
    }
}

fn graph_lowering_read_bindings(
    graph: &GpuRenderGraph,
    pass: &SemanticGraphPass,
) -> GraphBuildResult<Vec<GraphLoweringReadBinding>> {
    let kind = graph_lowering_pass_kind(graph, pass)?;
    let make = |index: usize,
                role: GraphLoweringReadRole,
                sampling_filter: GraphLoweringSamplingFilter,
                sampling_edge: GraphLoweringSamplingEdge|
     -> GraphBuildResult<GraphLoweringReadBinding> {
        let resource = pass
            .reads
            .get(index)
            .copied()
            .ok_or(GraphValidationError::InvalidPassArity)?;
        Ok(GraphLoweringReadBinding {
            role,
            resource: GraphLoweringResourceId::from_semantic(resource),
            sampling_filter,
            sampling_edge,
        })
    };
    let bindings = match kind {
        GraphLoweringPassKind::ClearRoot { .. } | GraphLoweringPassKind::VelloCapture(None) => {
            if !pass.reads.is_empty() {
                return Err(GraphValidationError::InvalidPassArity);
            }
            Vec::new()
        }
        GraphLoweringPassKind::VelloCapture(Some(_)) => {
            if !pass.reads.is_empty() {
                return Err(GraphValidationError::InvalidCaptureResult);
            }
            Vec::new()
        }
        GraphLoweringPassKind::CanonicalizeCapture => vec![make(
            0,
            GraphLoweringReadRole::CaptureSource,
            GraphLoweringSamplingFilter::Linear,
            GraphLoweringSamplingEdge::ClampToExtent,
        )?],
        GraphLoweringPassKind::CopyBackdrop => vec![make(
            0,
            GraphLoweringReadRole::CompletedParent,
            GraphLoweringSamplingFilter::Nearest,
            GraphLoweringSamplingEdge::TransparentBlack,
        )?],
        GraphLoweringPassKind::ColorFilter(Some(filter)) => vec![make(
            0,
            GraphLoweringReadRole::FilterSource,
            GraphLoweringSamplingFilter::Nearest,
            sampling_edge_for_filter(filter.edge()),
        )?],
        GraphLoweringPassKind::BlurHorizontal(Some(blur))
        | GraphLoweringPassKind::BlurVertical(Some(blur)) => vec![make(
            0,
            GraphLoweringReadRole::FilterSource,
            GraphLoweringSamplingFilter::GaussianKernel,
            sampling_edge_for_filter(blur.edge()),
        )?],
        GraphLoweringPassKind::DropShadowColorize(Some(_)) => vec![make(
            0,
            GraphLoweringReadRole::BlurredSourceAlpha,
            GraphLoweringSamplingFilter::Linear,
            GraphLoweringSamplingEdge::TransparentBlack,
        )?],
        GraphLoweringPassKind::ColorFilter(None)
        | GraphLoweringPassKind::BlurHorizontal(None)
        | GraphLoweringPassKind::BlurVertical(None)
        | GraphLoweringPassKind::DropShadowColorize(None)
        | GraphLoweringPassKind::Composite(None) => {
            if !pass.reads.is_empty() {
                return Err(GraphValidationError::InvalidPassArity);
            }
            Vec::new()
        }
        GraphLoweringPassKind::Composite(Some(composite)) => {
            graph_lowering_composite_read_bindings(pass, &composite)?
        }
        GraphLoweringPassKind::Present => vec![make(
            0,
            GraphLoweringReadRole::FinalWorkingImage,
            GraphLoweringSamplingFilter::Linear,
            GraphLoweringSamplingEdge::ClampToExtent,
        )?],
    };
    if bindings.len() != pass.reads.len() {
        return Err(GraphValidationError::InvalidPassArity);
    }
    Ok(bindings)
}

fn graph_lowering_composite_read_bindings(
    pass: &SemanticGraphPass,
    composite: &GraphLoweringComposite,
) -> GraphBuildResult<Vec<GraphLoweringReadBinding>> {
    let make = |index: usize,
                role: GraphLoweringReadRole,
                sampling_filter: GraphLoweringSamplingFilter,
                sampling_edge: GraphLoweringSamplingEdge|
     -> GraphBuildResult<GraphLoweringReadBinding> {
        let resource = pass
            .reads
            .get(index)
            .copied()
            .ok_or(GraphValidationError::InvalidPassArity)?;
        Ok(GraphLoweringReadBinding {
            role,
            resource: GraphLoweringResourceId::from_semantic(resource),
            sampling_filter,
            sampling_edge,
        })
    };
    let parent = || {
        make(
            0,
            GraphLoweringReadRole::CompositeParent,
            GraphLoweringSamplingFilter::Linear,
            GraphLoweringSamplingEdge::ClampToExtent,
        )
    };
    let source = |index| {
        make(
            index,
            GraphLoweringReadRole::CompositeSource,
            GraphLoweringSamplingFilter::Linear,
            GraphLoweringSamplingEdge::TransparentBlack,
        )
    };
    match composite.kind() {
        GraphLoweringCompositeKind::SpanSourceOver => Ok(vec![parent()?, source(1)?]),
        GraphLoweringCompositeKind::Layer {
            clip_coverage,
            alpha_mask,
            ..
        } => {
            let mut bindings = vec![parent()?, source(1)?];
            let mut next_read = 2;
            if let Some(clip_coverage) = clip_coverage {
                let binding = make(
                    next_read,
                    GraphLoweringReadRole::ClipCoverage,
                    GraphLoweringSamplingFilter::Linear,
                    GraphLoweringSamplingEdge::TransparentBlack,
                )?;
                if binding.resource != *clip_coverage {
                    return Err(GraphValidationError::InvalidPassArity);
                }
                bindings.push(binding);
                next_read += 1;
            }
            if let Some(alpha_mask) = alpha_mask {
                let binding = make(
                    next_read,
                    GraphLoweringReadRole::AlphaMask,
                    GraphLoweringSamplingFilter::ImportedMask,
                    GraphLoweringSamplingEdge::ClampToExtent,
                )?;
                if binding.resource != alpha_mask.resource() {
                    return Err(GraphValidationError::InvalidPassArity);
                }
                bindings.push(binding);
            }
            Ok(bindings)
        }
        GraphLoweringCompositeKind::DropShadow => Ok(vec![
            source(0)?,
            make(
                1,
                GraphLoweringReadRole::Shadow,
                GraphLoweringSamplingFilter::Linear,
                GraphLoweringSamplingEdge::TransparentBlack,
            )?,
        ]),
    }
}

fn sampling_edge_for_filter(edge: GraphLoweringEdgePolicy) -> GraphLoweringSamplingEdge {
    match edge {
        GraphLoweringEdgePolicy::NoSampling => GraphLoweringSamplingEdge::ClampToExtent,
        GraphLoweringEdgePolicy::TransparentBlack => GraphLoweringSamplingEdge::TransparentBlack,
        GraphLoweringEdgePolicy::SemanticBorderMirror(bounds) => {
            GraphLoweringSamplingEdge::SemanticBorderMirror(bounds)
        }
    }
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
    let plan = FrameContext::try_for_spatial_test(surface_scale)?.plan_filter_list(
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
    let plan = FrameContext::try_for_spatial_test(surface_scale)?
        .plan_local_bounds(logical_bounds, transform)?;
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
    pub(crate) antialiasing: Option<super::renderer::Antialiasing>,
    pub(crate) base_color: Option<super::paint::Color>,
    pub(crate) selection_requirements: Vec<FrameSelectionRequirementObservation>,
    pub(crate) resolved_alpha_mask_device_extents: Vec<(u32, u32)>,
    pub(crate) vello_spans: Vec<VelloSpanObservation>,
    pub(crate) graph_layer_blends: Vec<super::layer::BlendMode>,
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
    pub(crate) error_code: Option<super::error::ErrorCode>,
    pub(crate) unresolved_resource: Option<super::error::UnresolvedResourceKind>,
    pub(crate) has_partial_plan: bool,
}

#[cfg(test)]
pub(crate) fn frame_plan_result_observation_for_test(
    commands: super::command::RenderCommands,
    surface_size: super::geometry::Size,
    surface_scale: f64,
    antialiasing: super::renderer::Antialiasing,
    base_color: super::paint::Color,
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
fn observe_vello_command(command: &super::command::RenderCommand) -> VelloCommandObservation {
    match command {
        super::command::RenderCommand::Fill { .. } => VelloCommandObservation::Fill,
        super::command::RenderCommand::Stroke { .. } => VelloCommandObservation::Stroke,
        super::command::RenderCommand::Shadow { .. } => VelloCommandObservation::Shadow,
        super::command::RenderCommand::Image { .. } => VelloCommandObservation::Image,
        super::command::RenderCommand::TextRun { .. } => VelloCommandObservation::Text,
        super::command::RenderCommand::Layer { .. } => VelloCommandObservation::LocalLayer,
    }
}
