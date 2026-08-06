use super::{
    bounds::NonEmptyFrameSpatialPlan,
    filter::{
        DropShadowAlphaSource, DropShadowOffsetSampling, DropShadowSourceComposition,
        FilterEdgePolicy, ResolvedFilterOperationIntent, ResolvedFilterSpatialMapping,
    },
    graph::{
        BlurInput, GpuRenderGraph, GraphBuildResult, GraphGeneration, GraphValidationError,
        SemanticClipCoverage, SemanticCompositeKind, SemanticCompositePlan, SemanticFilterStepPlan,
        SemanticGraphPass, SemanticGraphResource, SemanticImportKind, SemanticImportPlan,
        SemanticPassId, SemanticPassIntent, SemanticPassResult, SemanticResourceDescriptor,
        SemanticResourceId, SemanticResourceProducer, SemanticResourceRole, SemanticVelloSpan,
        SemanticVelloSpanScope, WorkingImageInitialization, graph_build,
    },
    validate::LoweringValidationState,
};
use crate::{
    command::{RenderClip, RenderCommands},
    error::Result,
    filter::ColorClampBoundary,
    geometry::{PhysicalSize, Point, Rect, Transform},
    image::{Extend, ImageQuality, ResolvedMaskUploadDescriptor},
    layer::BlendMode,
    paint::Color,
    renderer::Antialiasing,
    style::ColorFilterOp,
};

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

impl GpuRenderGraph {
    pub(crate) fn lowering_view(&self) -> Result<GraphLoweringView<'_>> {
        let mut validation = graph_build(LoweringValidationState::begin(self))?;
        for pass in &self.passes {
            graph_build(validation.validate_pass(pass))?;
            graph_build(graph_lowering_pass_kind(self, pass))?;
            graph_build(graph_lowering_read_bindings(self, pass))?;
        }
        graph_build(validation.finish())?;
        Ok(GraphLoweringView { graph: self })
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
        blend: BlendMode,
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
    pub(super) graph: &'graph GpuRenderGraph,
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

pub(super) fn graph_lowering_pass_kind(
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
        .get(source.index.as_usize()?)
        .ok_or(GraphValidationError::UnknownResource(*source))?;
    let result = graph
        .resources
        .get(result.index.as_usize()?)
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

pub(super) fn graph_lowering_read_bindings(
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
