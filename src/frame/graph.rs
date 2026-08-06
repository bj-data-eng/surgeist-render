#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use super::bounds::{
    DestinationToLayerLocalMapping, FrameSpatialPlan, LogicalBounds, NonEmptyFrameSpatialPlan,
    NonEmptyLogicalBounds, PositiveDeviceExtent, SemanticContributionDomain, SemanticSourceBounds,
    SemanticSourceContribution, SignedDeviceOrigin, TexelCenterMapping,
    destination_to_layer_local_mapping, mask_upload_spatial,
};
use super::filter::{
    FilterSourceRole, ResolvedFilterOperationIntent, ResolvedFilterStep, ResolvedFrameFilterPlan,
};
use super::{FrameContext, GraphSelectionRequirement, validate::validate_semantic_frame_graph};
use crate::command::{
    LayerIsolation, NormalizedLayer, RenderClip, RenderCommand, RenderCommands, RenderLayerMask,
};
use crate::error::{BackendErrorCode, Error, Result};
use crate::geometry::{PhysicalSize, Rect, Transform};
use crate::image::{Extend, ImageQuality, ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey};
use crate::renderer::Antialiasing;
use crate::style::FilterList;

#[cfg(test)]
static NEXT_GRAPH_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(super) type GraphBuildResult<T> = std::result::Result<T, GraphValidationError>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct GraphGeneration(pub(super) u64);

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
pub(super) struct ResourceIndex(pub(super) u32);

impl ResourceIndex {
    pub(super) fn try_from_len(len: usize) -> GraphBuildResult<Self> {
        u32::try_from(len)
            .map(Self)
            .map_err(|_| GraphValidationError::ResourceIdentityExhausted)
    }

    pub(super) fn as_usize(self) -> GraphBuildResult<usize> {
        usize::try_from(self.0).map_err(|_| GraphValidationError::UnknownResourceIndex)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct PassIndex(pub(super) u32);

impl PassIndex {
    pub(super) fn try_from_len(len: usize) -> GraphBuildResult<Self> {
        u32::try_from(len)
            .map(Self)
            .map_err(|_| GraphValidationError::PassIdentityExhausted)
    }

    pub(super) fn as_usize(self) -> GraphBuildResult<usize> {
        usize::try_from(self.0).map_err(|_| GraphValidationError::UnknownPassIndex)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SemanticResourceId {
    pub(super) generation: GraphGeneration,
    pub(super) index: ResourceIndex,
}

impl SemanticResourceId {
    pub(super) const fn new(generation: GraphGeneration, index: ResourceIndex) -> Self {
        Self { generation, index }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct SemanticPassId {
    pub(super) generation: GraphGeneration,
    pub(super) index: PassIndex,
}

impl SemanticPassId {
    pub(super) const fn new(generation: GraphGeneration, index: PassIndex) -> Self {
        Self { generation, index }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SemanticResourceRole {
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
pub(super) struct SemanticResourceDescriptor {
    pub(super) role: SemanticResourceRole,
    pub(super) logical_bounds: NonEmptyLogicalBounds,
    pub(super) device_origin: SignedDeviceOrigin,
    pub(super) device_extent: PositiveDeviceExtent,
    pub(super) texel_center_mapping: TexelCenterMapping,
    pub(super) expected_reads: u32,
}

impl SemanticResourceDescriptor {
    pub(super) const fn new(
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
pub(super) enum WorkingImageInitialization {
    SurfaceBaseColor(crate::paint::Color),
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SemanticPassIntent {
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
pub(super) enum SemanticPassResult {
    Empty,
    Resource(SemanticResourceId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SemanticResourceProducer {
    Imported,
    Pass(SemanticPassId),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticGraphResource {
    pub(super) id: SemanticResourceId,
    pub(super) descriptor: SemanticResourceDescriptor,
    pub(super) producer: Option<SemanticResourceProducer>,
    pub(super) recorded_reads: u32,
    pub(super) remaining_reads: Option<u32>,
    pub(super) releasable_after: Option<SemanticPassId>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticGraphPass {
    pub(super) id: SemanticPassId,
    pub(super) intent: SemanticPassIntent,
    pub(super) dependencies: Vec<SemanticPassId>,
    pub(super) reads: Vec<SemanticResourceId>,
    pub(super) result: SemanticPassResult,
    pub(super) scheduled: bool,
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
pub(super) enum GraphValidationError {
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
pub(super) enum SemanticVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticVelloSpan {
    pub(super) capture_pass: SemanticPassId,
    pub(super) scope: SemanticVelloSpanScope,
    pub(super) commands: RenderCommands,
    pub(super) capture_transform: Transform,
    pub(super) parent_to_surface: Transform,
    pub(super) antialiasing: Antialiasing,
    pub(super) captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticClipCoverage {
    pub(super) capture_pass: SemanticPassId,
    pub(super) elements: Vec<SemanticOuterClip>,
    pub(super) antialiasing: Antialiasing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SemanticResolvedAlphaMaskComposition {
    pub(super) resource: SemanticResourceId,
    pub(super) bounds: Rect,
    pub(super) image_dimensions: PhysicalSize,
    pub(super) quality: ImageQuality,
    pub(super) extend: Extend,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SemanticCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        destination_to_layer_local: DestinationToLayerLocalMapping,
        opacity: f32,
        blend: crate::layer::BlendMode,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<SemanticOuterClip>,
        clip_coverage: Option<SemanticResourceId>,
        alpha_mask: Option<Box<SemanticResolvedAlphaMaskComposition>>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticCompositePlan {
    pub(super) pass: SemanticPassId,
    pub(super) kind: SemanticCompositeKind,
    pub(super) source_captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticFilterStepPlan {
    pub(super) passes: Vec<SemanticPassId>,
    pub(super) step: ResolvedFilterStep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SemanticBackdropRead {
    pub(super) pass: SemanticPassId,
    pub(super) completed_parent: SemanticResourceId,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum SemanticImportKind {
    ResolvedAlphaMask {
        upload: ResolvedMaskUploadDescriptor,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticImportPlan {
    pub(super) resource: SemanticResourceId,
    pub(super) kind: SemanticImportKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GpuRenderGraph {
    pub(super) generation: GraphGeneration,
    pub(super) resources: Vec<SemanticGraphResource>,
    pub(super) passes: Vec<SemanticGraphPass>,
    pub(super) root_working_image: SemanticResourceId,
    pub(super) final_present: SemanticPassId,
    pub(super) selection_requirements: Vec<GraphSelectionRequirement>,
    pub(super) vello_spans: Vec<SemanticVelloSpan>,
    pub(super) clip_coverages: Vec<SemanticClipCoverage>,
    pub(super) composites: Vec<SemanticCompositePlan>,
    pub(super) filter_steps: Vec<SemanticFilterStepPlan>,
    pub(super) backdrop_reads: Vec<SemanticBackdropRead>,
    pub(super) imports: Vec<SemanticImportPlan>,
}

#[derive(Debug)]
pub(super) struct SemanticGraphBuilder {
    pub(super) generation: GraphGeneration,
    phase: GraphBuildPhase,
    pub(super) resources: Vec<SemanticGraphResource>,
    pub(super) passes: Vec<SemanticGraphPass>,
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
    pub(super) fn try_new() -> GraphBuildResult<Self> {
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

    pub(super) fn declare_resource(
        &mut self,
        descriptor: SemanticResourceDescriptor,
    ) -> GraphBuildResult<SemanticResourceId> {
        self.insert_resource(descriptor, None)
    }

    pub(super) fn import_resource(
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

    pub(super) fn declare_pass(
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

    pub(super) fn begin_scheduling(&mut self) -> GraphBuildResult<()> {
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

    pub(super) fn schedule_pass(&mut self, id: SemanticPassId) -> GraphBuildResult<()> {
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

    pub(super) fn ensure_resource_readable(
        &self,
        resource: SemanticResourceId,
    ) -> GraphBuildResult<()> {
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

    pub(super) fn finish(self) -> GraphBuildResult<GpuRenderGraph> {
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

    pub(super) fn validate_resource_id(&self, id: SemanticResourceId) -> GraphBuildResult<usize> {
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
pub(super) struct SemanticOuterClip {
    pub(super) clip: RenderClip,
    pub(super) transform: Transform,
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
pub(super) enum CaptureBoundsCoordinateSpace {
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

pub(super) struct SemanticFrameGraphPlanner {
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
    pub(super) fn build(
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

    pub(super) fn build_with_capture_mapping(
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
    pub(super) fn build_authored_filter_fixture(
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
                blend: crate::layer::BlendMode::Normal,
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
                blend: crate::layer::BlendMode::Normal,
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
            && layer.blend == crate::layer::BlendMode::Normal;
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
        backdrop: &crate::command::RenderBackdropCapture,
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
                blend: crate::layer::BlendMode::Normal,
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
    if layer.blend != crate::layer::BlendMode::Normal
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

pub(super) fn graph_build<T>(result: GraphBuildResult<T>) -> Result<T> {
    result.map_err(|error| {
        Error::new(
            BackendErrorCode::RenderFailed,
            format!("semantic frame graph validation failed: {error:?}"),
        )
    })
}
