use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use super::{
    BackendErrorCode, Color, Error, Format, PhysicalSize, Point, Rect, Result, Transform,
    backend::DeviceCapabilities,
    command::{RenderClip, RenderCommands},
    filter::{CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS, ColorClampBoundary},
    frame::{
        GpuRenderGraph, GraphLoweringBlur, GraphLoweringBlurInput, GraphLoweringColorFilter,
        GraphLoweringComposite, GraphLoweringCompositeKind, GraphLoweringDropShadow,
        GraphLoweringEdgePolicy, GraphLoweringGeneration, GraphLoweringImportView,
        GraphLoweringInitialization, GraphLoweringPassId, GraphLoweringPassKind,
        GraphLoweringPassResult, GraphLoweringReadBinding, GraphLoweringReadRole,
        GraphLoweringResourceId, GraphLoweringResourceProducer, GraphLoweringResourceRole,
        GraphLoweringSamplingEdge, GraphLoweringSamplingFilter, GraphLoweringSpatialDescriptor,
        GraphLoweringVelloSpan, GraphLoweringVelloSpanScope,
    },
    image::ResolvedMaskUploadDescriptor,
    layer::BlendMode,
    renderer::{Antialiasing, EffectQualityPolicy},
    resource::{
        FrameCleanup, FrameResourceScope, GaussianKernelKey, GaussianKernelPlan,
        GaussianKernelSamplingForm, ResourceAllocationPreflight, ResourceIdentity, ResourceLease,
        ResourceManager, WorkingFormat,
    },
    shader::{
        BindGroupLayoutKey, DevicePassCache, PassSpatialUniformBytes, RenderPipelineKey,
        SamplerKey, ShaderBindingRoleKey, ShaderBlendKey, ShaderCompositeKey, ShaderDataBindingKey,
        ShaderModuleKey, ShaderProgramKey, ShaderSamplingEdgeKey, ShaderSamplingFilterKey,
        ShaderTextureFormatKey,
    },
    style::ColorFilterOp,
    texture::EffectTextureDescriptor,
};

#[cfg(test)]
use super::texture::EffectTextureRole;

#[cfg(test)]
use super::frame::{FrameContext, FramePlan};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08ExecutableSubsetObservationForTest {
    pub(crate) accepts_exact_rgba_and_bgra: bool,
    pub(crate) rejects_every_other_pass_kind_and_composite_payload: bool,
    pub(crate) rejects_missing_or_reordered_spine_passes: bool,
    pub(crate) rejects_malformed_dependencies_reads_results_and_releases: bool,
    pub(crate) rejects_later_cycle_plan: bool,
    pub(crate) preserves_direct_and_transitional_planner_routes: bool,
}

#[cfg(test)]
pub(crate) fn c08_executable_subset_observation_for_test(
    c08_commands: RenderCommands,
    later_cycle_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C08ExecutableSubsetObservationForTest {
    c08_executable_subset_observation(c08_commands, later_cycle_commands, context, capabilities)
        .unwrap_or_default()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BoundedCaptureTransformObservationForTest {
    pub(crate) preserves_application_order_formula: bool,
    pub(crate) preserves_signed_texel_center_mapping: bool,
    pub(crate) covers_required_raster_scales: bool,
    pub(crate) preserves_capture_execution_facts: bool,
    pub(crate) lowers_scene_with_explicit_initial_transform: bool,
}

#[cfg(test)]
pub(crate) fn bounded_capture_transform_observation_for_test(
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
) -> BoundedCaptureTransformObservationForTest {
    bounded_capture_transform_observation(
        commands,
        capture_transform,
        parent_to_surface,
        antialiasing,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn pass_spatial_uniform_bytes_for_test(
    source_origin: Point,
    source_raster_scale: f64,
    source_extent: PhysicalSize,
    destination_origin: Point,
    destination_raster_scale: f64,
    destination_extent: PhysicalSize,
) -> Result<[u8; 48]> {
    let source = RuntimeSpatialDescriptor {
        logical_bounds: Rect::new(0.0, 0.0, 1.0, 1.0),
        device_origin: (0, 0),
        device_extent: source_extent,
        texel_origin: source_origin,
        raster_scale: source_raster_scale,
    };
    let destination = RuntimeSpatialDescriptor {
        logical_bounds: Rect::new(0.0, 0.0, 1.0, 1.0),
        device_origin: (0, 0),
        device_extent: destination_extent,
        texel_origin: destination_origin,
        raster_scale: destination_raster_scale,
    };
    super::shader::PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(
        source,
        destination,
    )
    .map(super::shader::PassSpatialUniformBytes::into_bytes_for_test)
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeGraphGeneration(GraphLoweringGeneration);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimeResourceId(GraphLoweringResourceId);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RuntimePassId(GraphLoweringPassId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceRole {
    RootWorkingImage,
    CaptureWorkingImage,
    IsolationWorkingImage,
    ImportedImage,
    BackdropCopy,
    FilterIntermediate,
    ShadowImage,
    CompositeResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceFormat {
    VelloCaptureRgba8Unorm,
    Working(WorkingFormat),
    ResolvedMaskRgba8Unorm,
}

impl RuntimeResourceFormat {
    const fn shader_key(self) -> ShaderTextureFormatKey {
        match self {
            Self::VelloCaptureRgba8Unorm => ShaderTextureFormatKey::VelloCaptureRgba8Unorm,
            Self::Working(format) => ShaderTextureFormatKey::working(format),
            Self::ResolvedMaskRgba8Unorm => ShaderTextureFormatKey::ResolvedMaskRgba8Unorm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeSpatialDescriptor {
    logical_bounds: Rect,
    device_origin: (i32, i32),
    device_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl RuntimeSpatialDescriptor {
    fn from_graph(spatial: GraphLoweringSpatialDescriptor) -> Self {
        Self {
            logical_bounds: spatial.logical_bounds(),
            device_origin: spatial.device_origin(),
            device_extent: spatial.device_extent(),
            texel_origin: spatial.texel_origin(),
            raster_scale: spatial.raster_scale(),
        }
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
pub(crate) enum RuntimeResourceProducer {
    Imported,
    Pass(RuntimePassId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResourceImport {
    ResolvedAlphaMask(ResolvedMaskUploadDescriptor),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeResourceRequest {
    id: RuntimeResourceId,
    role: RuntimeResourceRole,
    format: RuntimeResourceFormat,
    spatial: RuntimeSpatialDescriptor,
    producer: RuntimeResourceProducer,
    expected_reads: u32,
    last_use: RuntimePassId,
    import: Option<RuntimeResourceImport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeInitialization {
    SurfaceBaseColor,
    Transparent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeVelloSpanScope {
    CurrentParent,
    LayerSource,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeVelloSpan {
    scope: RuntimeVelloSpanScope,
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
    captured_before_outer_semantics: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeColorClampBoundary {
    ClampStraightRgbaToUnitThenPremultiply,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeColorOperation {
    operation: ColorFilterOp,
    clamp_boundary: RuntimeColorClampBoundary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RuntimeSamplingEdge {
    ClampToExtent,
    TransparentBlack,
    SemanticBorderMirror(Rect),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeFilterSpatialMapping {
    source: RuntimeSpatialDescriptor,
    result: RuntimeSpatialDescriptor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeColorFilter {
    operations: Vec<RuntimeColorOperation>,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurInput {
    Rgba,
    SourceAlpha,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBlurAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeBlur {
    axis: RuntimeBlurAxis,
    input: RuntimeBlurInput,
    standard_deviation: f64,
    support_radius: u32,
    kernel: GaussianKernelKey,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDropShadow {
    offset: Point,
    standard_deviation: f64,
    color: Color,
    support_radius: u32,
    spatial: RuntimeFilterSpatialMapping,
    edge: RuntimeSamplingEdge,
    uses_source_alpha: bool,
    uses_continuous_offset: bool,
    retains_unchanged_source: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeOuterClip {
    clip: RenderClip,
    transform: Transform,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        opacity: f32,
        blend: BlendMode,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<RuntimeOuterClip>,
        alpha_mask: Option<RuntimeResourceId>,
    },
    DropShadow,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeComposite {
    kind: RuntimeCompositeKind,
    source_captured_before_outer_semantics: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimePassKind {
    ClearRoot {
        initialization: RuntimeInitialization,
        color: Color,
    },
    VelloCapture(Option<RuntimeVelloSpan>),
    CanonicalizeCapture,
    CopyBackdrop,
    ColorFilter(Option<RuntimeColorFilter>),
    BlurHorizontal(Option<RuntimeBlur>),
    BlurVertical(Option<RuntimeBlur>),
    DropShadowColorize(Option<RuntimeDropShadow>),
    Composite(Option<RuntimeComposite>),
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeReadRole {
    CaptureSource,
    CompletedParent,
    FilterSource,
    BlurredSourceAlpha,
    CompositeParent,
    CompositeSource,
    AlphaMask,
    Shadow,
    FinalWorkingImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeSamplingFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeReadBinding {
    role: RuntimeReadRole,
    resource: RuntimeResourceId,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
    sampler_key: SamplerKey,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these exact immutable runtime read-binding facts"
    )
)]
impl RuntimeReadBinding {
    pub(crate) const fn role(&self) -> RuntimeReadRole {
        self.role
    }

    pub(crate) const fn resource(&self) -> RuntimeResourceId {
        self.resource
    }

    pub(crate) const fn sampling_filter(&self) -> RuntimeSamplingFilter {
        self.sampling_filter
    }

    pub(crate) const fn sampling_edge(&self) -> RuntimeSamplingEdge {
        self.sampling_edge
    }

    pub(crate) const fn sampler_key(&self) -> SamplerKey {
        self.sampler_key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeResultBinding {
    Empty,
    Resource(RuntimeResourceId),
    Output(Format),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimePassCacheKeys {
    samplers: Vec<SamplerKey>,
    layout: BindGroupLayoutKey,
    shader: ShaderModuleKey,
    pipeline: RenderPipelineKey,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these exact immutable device pass-cache keys"
    )
)]
impl RuntimePassCacheKeys {
    pub(crate) fn samplers(&self) -> &[SamplerKey] {
        &self.samplers
    }

    pub(crate) const fn layout(&self) -> &BindGroupLayoutKey {
        &self.layout
    }

    pub(crate) const fn shader(&self) -> &ShaderModuleKey {
        &self.shader
    }

    pub(crate) const fn pipeline(&self) -> &RenderPipelineKey {
        &self.pipeline
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimePass {
    id: RuntimePassId,
    kind: RuntimePassKind,
    dependencies: Vec<RuntimePassId>,
    reads: Vec<RuntimeReadBinding>,
    result: RuntimeResultBinding,
    releases: Vec<RuntimeResourceId>,
    cache_keys: Option<RuntimePassCacheKeys>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoweredGraphPlan {
    generation: RuntimeGraphGeneration,
    working_format: WorkingFormat,
    output_format: Format,
    resources: Vec<RuntimeResourceRequest>,
    passes: Vec<RuntimePass>,
    root_working_image: RuntimeResourceId,
    final_present: RuntimePassId,
}

#[must_use]
pub(crate) struct C08ExecutableSubset<'plan> {
    working_format: WorkingFormat,
    output_format: Format,
    captures: Vec<C08VelloCaptureExecutionFacts<'plan>>,
}

impl C08ExecutableSubset<'_> {
    #[must_use]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.working_format
    }

    #[must_use]
    pub(crate) const fn output_format(&self) -> Format {
        self.output_format
    }

    #[must_use]
    pub(crate) fn captures(&self) -> &[C08VelloCaptureExecutionFacts<'_>] {
        &self.captures
    }

    fn proves_exact_execution_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
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
            let RuntimePassKind::VelloCapture(Some(span)) = &pass.kind else {
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
                && capture.commands() == &span.commands
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

#[derive(Clone, Copy)]
pub(crate) struct C08VelloCaptureExecutionFacts<'plan> {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    commands: &'plan RenderCommands,
    initial_transform: Transform,
    antialiasing: Antialiasing,
    target_extent: PhysicalSize,
    texel_origin: Point,
    raster_scale: f64,
}

impl C08VelloCaptureExecutionFacts<'_> {
    #[must_use]
    pub(crate) const fn pass(&self) -> RuntimePassId {
        self.pass
    }

    #[must_use]
    pub(crate) const fn target(&self) -> RuntimeResourceId {
        self.target
    }

    #[must_use]
    pub(crate) const fn commands(&self) -> &RenderCommands {
        self.commands
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08PassClass {
    ClearRoot,
    VelloCapture,
    CanonicalizeCapture,
    SpanSourceOver,
    Present,
}

fn c08_pass_class(kind: &RuntimePassKind) -> Option<C08PassClass> {
    match kind {
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            ..
        } => Some(C08PassClass::ClearRoot),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            ..
        } => None,
        RuntimePassKind::VelloCapture(Some(_)) => Some(C08PassClass::VelloCapture),
        RuntimePassKind::VelloCapture(None) => None,
        RuntimePassKind::CanonicalizeCapture => Some(C08PassClass::CanonicalizeCapture),
        RuntimePassKind::CopyBackdrop
        | RuntimePassKind::ColorFilter(_)
        | RuntimePassKind::BlurHorizontal(_)
        | RuntimePassKind::BlurVertical(_)
        | RuntimePassKind::DropShadowColorize(_) => None,
        RuntimePassKind::Composite(Some(composite)) => match &composite.kind {
            RuntimeCompositeKind::SpanSourceOver
                if composite.source_captured_before_outer_semantics =>
            {
                Some(C08PassClass::SpanSourceOver)
            }
            RuntimeCompositeKind::SpanSourceOver
            | RuntimeCompositeKind::Layer { .. }
            | RuntimeCompositeKind::DropShadow => None,
        },
        RuntimePassKind::Composite(None) => None,
        RuntimePassKind::Present => Some(C08PassClass::Present),
    }
}

impl LoweredGraphPlan {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the complete private runtime lowering entry point"
        )
    )]
    pub(crate) fn try_lower_validated_graph(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
        capabilities: &DeviceCapabilities,
    ) -> Result<Self> {
        let view = graph.lowering_view()?;
        let resource_views = view.resources();
        let pass_views = view.passes();
        let pass_ids = pass_views
            .iter()
            .map(|pass| RuntimePassId(pass.id()))
            .collect::<BTreeSet<_>>();

        let mut resources = Vec::with_capacity(resource_views.len());
        let mut resource_ids = BTreeSet::new();
        let mut resource_formats = BTreeMap::new();
        for resource in resource_views {
            let id = RuntimeResourceId(resource.id());
            if !resource_ids.insert(id) {
                return Err(lowering_error("duplicate runtime resource binding"));
            }
            let role = runtime_resource_role(resource.role());
            let spatial = RuntimeSpatialDescriptor::from_graph(resource.spatial());
            capabilities.validate_effect_texture_extent(spatial.device_extent)?;
            let format = runtime_resource_format(role, working_format);
            if let RuntimeResourceFormat::Working(format) = format {
                capabilities.validate_effect_texture_allocation(
                    spatial.device_extent,
                    Some(format),
                    format.texture_format(),
                    format.required_usages(),
                )?;
            }
            let producer = match resource.producer() {
                GraphLoweringResourceProducer::Imported => RuntimeResourceProducer::Imported,
                GraphLoweringResourceProducer::Pass(pass) => {
                    let pass = RuntimePassId(pass);
                    if !pass_ids.contains(&pass) {
                        return Err(lowering_error("resource producer binding is missing"));
                    }
                    RuntimeResourceProducer::Pass(pass)
                }
            };
            let import = resource.import().map(|import| match import {
                GraphLoweringImportView::ResolvedAlphaMask(upload) => {
                    RuntimeResourceImport::ResolvedAlphaMask(upload.clone())
                }
            });
            if matches!(producer, RuntimeResourceProducer::Imported) != import.is_some() {
                return Err(lowering_error(
                    "imported runtime resource binding is inconsistent",
                ));
            }
            let last_use = RuntimePassId(resource.last_use());
            if !pass_ids.contains(&last_use) {
                return Err(lowering_error("resource last-use binding is missing"));
            }
            resource_formats.insert(id, format);
            resources.push(RuntimeResourceRequest {
                id,
                role,
                format,
                spatial,
                producer,
                expected_reads: resource.expected_reads(),
                last_use,
                import,
            });
        }

        let mut releases = BTreeMap::<RuntimePassId, Vec<RuntimeResourceId>>::new();
        for resource in &resources {
            releases
                .entry(resource.last_use)
                .or_default()
                .push(resource.id);
        }
        let resource_by_id = resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();

        let mut seen_passes = BTreeSet::new();
        let mut passes = Vec::with_capacity(pass_views.len());
        for pass in pass_views {
            let id = RuntimePassId(pass.id());
            if seen_passes.contains(&id) {
                return Err(lowering_error("duplicate runtime pass binding"));
            }
            let dependencies = pass
                .dependencies()
                .into_iter()
                .map(RuntimePassId)
                .collect::<Vec<_>>();
            if dependencies
                .iter()
                .any(|dependency| !seen_passes.contains(dependency))
            {
                return Err(lowering_error(
                    "runtime pass has a missing or forward dependency",
                ));
            }
            let graph_kind = pass.kind()?;
            let kind = runtime_pass_kind(graph_kind, working_format);
            let graph_reads = pass.reads()?;
            let reads = lower_read_bindings(&graph_reads, &resource_by_id, &resource_formats)?;
            let result = match pass.result() {
                GraphLoweringPassResult::Empty if matches!(kind, RuntimePassKind::Present) => {
                    RuntimeResultBinding::Output(output_format)
                }
                GraphLoweringPassResult::Empty => RuntimeResultBinding::Empty,
                GraphLoweringPassResult::Resource(resource) => {
                    let resource = RuntimeResourceId(resource);
                    if !resource_by_id.contains_key(&resource)
                        || reads.iter().any(|read| read.resource == resource)
                    {
                        return Err(lowering_error(
                            "runtime pass result binding is inconsistent",
                        ));
                    }
                    RuntimeResultBinding::Resource(resource)
                }
            };
            let pass_releases = releases.remove(&id).unwrap_or_default();
            if pass_releases
                .iter()
                .any(|resource| !reads.iter().any(|binding| binding.resource == *resource))
            {
                return Err(lowering_error(
                    "runtime release is not the resource's last read",
                ));
            }
            let cache_keys = runtime_pass_cache_keys(
                &kind,
                &reads,
                result,
                working_format,
                output_format,
                &resource_formats,
            )?;
            passes.push(RuntimePass {
                id,
                kind,
                dependencies,
                reads,
                result,
                releases: pass_releases,
                cache_keys,
            });
            seen_passes.insert(id);
        }
        if !releases.is_empty() {
            return Err(lowering_error(
                "one or more release bindings have no runtime pass",
            ));
        }

        let root_working_image = RuntimeResourceId(view.root_working_image());
        if resource_formats.get(&root_working_image)
            != Some(&RuntimeResourceFormat::Working(working_format))
        {
            return Err(lowering_error(
                "root working image does not use the graph format",
            ));
        }
        let final_present = RuntimePassId(view.final_present());
        if passes.last().is_none_or(|pass| {
            pass.id != final_present || !matches!(pass.kind, RuntimePassKind::Present)
        }) {
            return Err(lowering_error(
                "the runtime plan has no terminal present pass",
            ));
        }

        let lowered = Self {
            generation: RuntimeGraphGeneration(view.generation()),
            working_format,
            output_format,
            resources,
            passes,
            root_working_image,
            final_present,
        };
        let _ = lowered.c08_executable_subset();
        Ok(lowered)
    }

    pub(crate) fn c08_executable_subset(&self) -> Option<C08ExecutableSubset<'_>> {
        if ![Format::Rgba8, Format::Bgra8].contains(&self.output_format) {
            return None;
        }
        let resource_by_id = self
            .resources
            .iter()
            .map(|resource| (resource.id, resource))
            .collect::<BTreeMap<_, _>>();
        if resource_by_id.len() != self.resources.len() {
            return None;
        }
        let resource_formats = self
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let pass_ids = self
            .passes
            .iter()
            .map(|pass| pass.id)
            .collect::<BTreeSet<_>>();
        if pass_ids.len() != self.passes.len() {
            return None;
        }

        let clear = self.passes.first()?;
        if c08_pass_class(&clear.kind) != Some(C08PassClass::ClearRoot)
            || !clear.dependencies.is_empty()
            || !clear.reads.is_empty()
            || !clear.releases.is_empty()
            || clear.cache_keys.is_some()
            || clear.result != RuntimeResultBinding::Resource(self.root_working_image)
        {
            return None;
        }
        let root = resource_by_id.get(&self.root_working_image).copied()?;
        if !c08_resource_has_fixed_facts(
            root,
            RuntimeResourceRole::RootWorkingImage,
            RuntimeResourceFormat::Working(self.working_format),
            RuntimeResourceProducer::Pass(clear.id),
        ) {
            return None;
        }

        let mut cursor = 1;
        let mut parent = root;
        let mut parent_producer = clear.id;
        let mut captures = Vec::new();
        let mut expected_resources = BTreeSet::from([self.root_working_image]);
        while let Some(capture) = self.passes.get(cursor)
            && c08_pass_class(&capture.kind) == Some(C08PassClass::VelloCapture)
        {
            let canonicalize = self.passes.get(cursor.checked_add(1)?)?;
            let composite = self.passes.get(cursor.checked_add(2)?)?;
            let RuntimePassKind::VelloCapture(Some(span)) = &capture.kind else {
                return None;
            };
            let RuntimeResultBinding::Resource(capture_target) = capture.result else {
                return None;
            };
            if !capture.dependencies.is_empty()
                || !capture.reads.is_empty()
                || !capture.releases.is_empty()
                || capture.cache_keys.is_some()
            {
                return None;
            }
            let capture_resource = resource_by_id.get(&capture_target).copied()?;
            if !c08_resource_has_fixed_facts(
                capture_resource,
                RuntimeResourceRole::CaptureWorkingImage,
                RuntimeResourceFormat::VelloCaptureRgba8Unorm,
                RuntimeResourceProducer::Pass(capture.id),
            ) || capture_resource.expected_reads != 1
                || capture_resource.last_use != canonicalize.id
            {
                return None;
            }

            if c08_pass_class(&canonicalize.kind) != Some(C08PassClass::CanonicalizeCapture)
                || canonicalize.dependencies.as_slice() != [capture.id]
                || canonicalize.reads.len() != 1
                || !c08_read_is_exact(
                    &canonicalize.reads[0],
                    RuntimeReadRole::CaptureSource,
                    capture_target,
                    RuntimeSamplingFilter::Linear,
                    RuntimeSamplingEdge::ClampToExtent,
                    capture_resource.format,
                )
                || canonicalize.releases.as_slice() != [capture_target]
            {
                return None;
            }
            let RuntimeResultBinding::Resource(canonical_target) = canonicalize.result else {
                return None;
            };
            let canonical_resource = resource_by_id.get(&canonical_target).copied()?;
            if !c08_resource_has_fixed_facts(
                canonical_resource,
                RuntimeResourceRole::FilterIntermediate,
                RuntimeResourceFormat::Working(self.working_format),
                RuntimeResourceProducer::Pass(canonicalize.id),
            ) || canonical_resource.expected_reads != 1
                || canonical_resource.last_use != composite.id
                || canonical_resource.spatial != capture_resource.spatial
            {
                return None;
            }

            if c08_pass_class(&composite.kind) != Some(C08PassClass::SpanSourceOver)
                || composite.dependencies.as_slice() != [parent_producer, canonicalize.id]
                || composite.reads.len() != 2
                || !c08_read_is_exact(
                    &composite.reads[0],
                    RuntimeReadRole::CompositeParent,
                    parent.id,
                    RuntimeSamplingFilter::Linear,
                    RuntimeSamplingEdge::ClampToExtent,
                    parent.format,
                )
                || !c08_read_is_exact(
                    &composite.reads[1],
                    RuntimeReadRole::CompositeSource,
                    canonical_target,
                    RuntimeSamplingFilter::Linear,
                    RuntimeSamplingEdge::TransparentBlack,
                    canonical_resource.format,
                )
                || composite.releases.as_slice() != [parent.id, canonical_target]
                || parent.expected_reads != 1
                || parent.last_use != composite.id
            {
                return None;
            }
            let RuntimeResultBinding::Resource(composite_target) = composite.result else {
                return None;
            };
            let composite_resource = resource_by_id.get(&composite_target).copied()?;
            if !c08_resource_has_fixed_facts(
                composite_resource,
                RuntimeResourceRole::CompositeResult,
                RuntimeResourceFormat::Working(self.working_format),
                RuntimeResourceProducer::Pass(composite.id),
            ) || composite_resource.spatial != root.spatial
            {
                return None;
            }

            expected_resources.extend([capture_target, canonical_target, composite_target]);
            captures.push(c08_capture_execution_facts(
                capture.id,
                capture_target,
                span,
                capture_resource.spatial,
            )?);
            parent = composite_resource;
            parent_producer = composite.id;
            cursor = cursor.checked_add(3)?;
        }

        let present = self.passes.get(cursor)?;
        if cursor.checked_add(1)? != self.passes.len()
            || present.id != self.final_present
            || c08_pass_class(&present.kind) != Some(C08PassClass::Present)
            || present.dependencies.as_slice() != [parent_producer]
            || present.reads.len() != 1
            || !c08_read_is_exact(
                &present.reads[0],
                RuntimeReadRole::FinalWorkingImage,
                parent.id,
                RuntimeSamplingFilter::Linear,
                RuntimeSamplingEdge::ClampToExtent,
                parent.format,
            )
            || present.result != RuntimeResultBinding::Output(self.output_format)
            || present.releases.as_slice() != [parent.id]
            || parent.expected_reads != 1
            || parent.last_use != present.id
        {
            return None;
        }
        if expected_resources.len() != self.resources.len()
            || expected_resources
                .iter()
                .any(|resource| !resource_by_id.contains_key(resource))
        {
            return None;
        }
        for pass in &self.passes {
            let expected_cache_keys = runtime_pass_cache_keys(
                &pass.kind,
                &pass.reads,
                pass.result,
                self.working_format,
                self.output_format,
                &resource_formats,
            )
            .ok()?;
            if pass.cache_keys != expected_cache_keys {
                return None;
            }
        }

        let subset = C08ExecutableSubset {
            working_format: self.working_format,
            output_format: self.output_format,
            captures,
        };
        subset
            .proves_exact_execution_facts_for(self)
            .then_some(subset)
    }

    #[cfg(test)]
    pub(crate) fn with_duplicate_preparation_resource_for_test(&self) -> Self {
        let mut invalid = self.clone();
        if invalid.resources.len() > 1 {
            invalid.resources[1].id = invalid.resources[0].id;
        }
        invalid
    }
}

fn c08_resource_has_fixed_facts(
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

fn c08_read_is_exact(
    read: &RuntimeReadBinding,
    role: RuntimeReadRole,
    resource: RuntimeResourceId,
    sampling_filter: RuntimeSamplingFilter,
    sampling_edge: RuntimeSamplingEdge,
    source_format: RuntimeResourceFormat,
) -> bool {
    let sampler_key = SamplerKey::new(
        shader_binding_role(role),
        source_format.shader_key(),
        match sampling_filter {
            RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
            RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
        },
        shader_sampling_edge(sampling_edge),
        None,
    );
    read.role == role
        && read.resource == resource
        && read.sampling_filter == sampling_filter
        && read.sampling_edge == sampling_edge
        && read.sampler_key == sampler_key
}

fn c08_capture_execution_facts<'plan>(
    pass: RuntimePassId,
    target: RuntimeResourceId,
    span: &'plan RuntimeVelloSpan,
    spatial: RuntimeSpatialDescriptor,
) -> Option<C08VelloCaptureExecutionFacts<'plan>> {
    if span.scope != RuntimeVelloSpanScope::CurrentParent
        || span.commands.commands.is_empty()
        || !span.captured_before_outer_semantics
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
    let initial_transform = span
        .capture_transform
        .then(span.parent_to_surface)
        .ok()?
        .then(Transform::translation(-spatial.texel_origin.x(), -spatial.texel_origin.y()).ok()?)
        .ok()?
        .then(Transform::scale(spatial.raster_scale, spatial.raster_scale).ok()?)
        .ok()?;
    Some(C08VelloCaptureExecutionFacts {
        pass,
        target,
        commands: &span.commands,
        initial_transform,
        antialiasing: span.antialiasing,
        target_extent: spatial.device_extent,
        texel_origin: spatial.texel_origin,
        raster_scale: spatial.raster_scale,
    })
}

#[cfg(test)]
fn c08_executable_subset_observation(
    c08_commands: RenderCommands,
    later_cycle_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C08ExecutableSubsetObservationForTest> {
    let direct_route = matches!(
        c08_commands.clone().plan_for(context),
        Ok(FramePlan::DirectVello(_))
    );
    let later_cycle_plan = later_cycle_commands.clone().plan_for(context).ok()?;
    let transitional_route = matches!(&later_cycle_plan, FramePlan::GpuGraph(_));
    let FramePlan::GpuGraph(later_cycle_graph) = later_cycle_plan else {
        return None;
    };
    let c08_graph = super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
    let rgba = LoweredGraphPlan::try_lower_validated_graph(
        &c08_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )
    .ok()?;
    let bgra = LoweredGraphPlan::try_lower_validated_graph(
        &c08_graph,
        WorkingFormat::HighPrecision,
        Format::Bgra8,
        &capabilities,
    )
    .ok()?;
    let later_cycle = LoweredGraphPlan::try_lower_validated_graph(
        &later_cycle_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
        &capabilities,
    )
    .ok()?;
    let rgba_subset = rgba.c08_executable_subset()?;
    let bgra_subset = bgra.c08_executable_subset()?;
    let accepts_exact_rgba_and_bgra = rgba_subset.working_format() == WorkingFormat::HighPrecision
        && rgba_subset.output_format() == Format::Rgba8
        && bgra_subset.working_format() == WorkingFormat::HighPrecision
        && bgra_subset.output_format() == Format::Bgra8
        && !rgba_subset.captures().is_empty()
        && rgba_subset.captures().len() == bgra_subset.captures().len()
        && rgba_subset.captures().iter().all(|capture| {
            capture.target_extent().width() > 0
                && capture.target_extent().height() > 0
                && capture.raster_scale().is_finite()
                && capture.raster_scale() > 0.0
        });

    Some(C08ExecutableSubsetObservationForTest {
        accepts_exact_rgba_and_bgra,
        rejects_every_other_pass_kind_and_composite_payload:
            c08_rejects_every_other_pass_kind_and_composite_payload(&rgba),
        rejects_missing_or_reordered_spine_passes: c08_rejects_missing_or_reordered_spine_passes(
            &rgba,
        ),
        rejects_malformed_dependencies_reads_results_and_releases: c08_rejects_malformed_bindings(
            &rgba,
        ),
        rejects_later_cycle_plan: later_cycle.c08_executable_subset().is_none(),
        preserves_direct_and_transitional_planner_routes: direct_route && transitional_route,
    })
}

#[cfg(test)]
fn c08_rejects_every_other_pass_kind_and_composite_payload(plan: &LoweredGraphPlan) -> bool {
    let forbidden_kinds = [
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            color: Color::TRANSPARENT,
        },
        RuntimePassKind::VelloCapture(None),
        RuntimePassKind::CopyBackdrop,
        RuntimePassKind::ColorFilter(None),
        RuntimePassKind::BlurHorizontal(None),
        RuntimePassKind::BlurVertical(None),
        RuntimePassKind::DropShadowColorize(None),
        RuntimePassKind::Composite(None),
    ];
    if forbidden_kinds
        .iter()
        .any(|kind| c08_pass_class(kind).is_some())
    {
        return false;
    }
    let Some(composite_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::SpanSourceOver))
    else {
        return false;
    };
    let composite_payloads = [
        None,
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver,
            source_captured_before_outer_semantics: false,
        }),
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer {
                transform: Transform::identity(),
                opacity: 1.0,
                blend: BlendMode::Normal,
                clip: None,
                outer_clips: Vec::new(),
                alpha_mask: None,
            },
            source_captured_before_outer_semantics: true,
        }),
        Some(RuntimeComposite {
            kind: RuntimeCompositeKind::DropShadow,
            source_captured_before_outer_semantics: true,
        }),
    ];
    composite_payloads.into_iter().all(|payload| {
        let mut invalid = plan.clone();
        invalid.passes[composite_index].kind = RuntimePassKind::Composite(payload);
        invalid.c08_executable_subset().is_none()
    })
}

#[cfg(test)]
fn c08_rejects_missing_or_reordered_spine_passes(plan: &LoweredGraphPlan) -> bool {
    let Some(capture_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::VelloCapture))
    else {
        return false;
    };
    let Some(canonicalize_index) = capture_index.checked_add(1) else {
        return false;
    };
    let Some(present_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::Present))
    else {
        return false;
    };

    let mut missing_canonicalize = plan.clone();
    missing_canonicalize.passes.remove(canonicalize_index);
    let mut reordered_pair = plan.clone();
    reordered_pair
        .passes
        .swap(capture_index, canonicalize_index);
    let mut repeated_clear = plan.clone();
    repeated_clear
        .passes
        .insert(capture_index, repeated_clear.passes[0].clone());
    let mut missing_present = plan.clone();
    missing_present.passes.remove(present_index);
    let mut nonterminal_present = plan.clone();
    nonterminal_present
        .passes
        .swap(present_index - 1, present_index);

    [
        missing_canonicalize,
        reordered_pair,
        repeated_clear,
        missing_present,
        nonterminal_present,
    ]
    .iter()
    .all(|invalid| invalid.c08_executable_subset().is_none())
}

#[cfg(test)]
fn c08_rejects_malformed_bindings(plan: &LoweredGraphPlan) -> bool {
    let Some(capture_index) = plan
        .passes
        .iter()
        .position(|pass| c08_pass_class(&pass.kind) == Some(C08PassClass::VelloCapture))
    else {
        return false;
    };
    let canonicalize_index = capture_index + 1;
    let composite_index = capture_index + 2;
    let present_index = plan.passes.len() - 1;
    let RuntimeResultBinding::Resource(capture_target) = plan.passes[capture_index].result else {
        return false;
    };
    let RuntimeResultBinding::Resource(canonical_target) = plan.passes[canonicalize_index].result
    else {
        return false;
    };
    let Some(capture_resource_index) = plan
        .resources
        .iter()
        .position(|resource| resource.id == capture_target)
    else {
        return false;
    };

    let mut invalid_plans = Vec::new();

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].dependencies.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].reads[0].role = RuntimeReadRole::FinalWorkingImage;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].reads[0].resource = plan.root_working_image;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].result = RuntimeResultBinding::Empty;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[canonicalize_index].releases.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].reads.swap(0, 1);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].result = RuntimeResultBinding::Resource(canonical_target);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[composite_index].cache_keys = None;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[present_index].result = RuntimeResultBinding::Output(Format::Bgra8);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.passes[present_index].releases.clear();
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index].expected_reads = 2;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index].format =
        RuntimeResourceFormat::Working(plan.working_format);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.resources[capture_resource_index]
        .spatial
        .texel_origin = Point::new(-0.25, 0.5);
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    invalid.final_present = invalid.passes[0].id;
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    if let RuntimePassKind::VelloCapture(Some(span)) = &mut invalid.passes[capture_index].kind {
        span.scope = RuntimeVelloSpanScope::LayerSource;
    }
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    if let RuntimePassKind::VelloCapture(Some(span)) = &mut invalid.passes[capture_index].kind {
        span.captured_before_outer_semantics = false;
    }
    invalid_plans.push(invalid);

    invalid_plans
        .iter()
        .all(|invalid| invalid.c08_executable_subset().is_none())
}

#[cfg(test)]
fn bounded_capture_transform_observation(
    commands: RenderCommands,
    capture_transform: Transform,
    parent_to_surface: Transform,
    antialiasing: Antialiasing,
) -> Option<BoundedCaptureTransformObservationForTest> {
    let capabilities = DeviceCapabilities::from_test_facts(true, true, 4_096);
    let scales = [1.0, 1.25, 2.0];
    let mut preserves_application_order_formula = true;
    let mut preserves_signed_texel_center_mapping = true;
    let mut preserves_capture_execution_facts = true;
    let mut lowers_scene_with_explicit_initial_transform = true;

    for raster_scale in scales {
        let context = FrameContext::try_new(
            super::Size::new(64.0, 64.0),
            raster_scale,
            antialiasing,
            Color::TRANSPARENT,
        )
        .ok()?;
        let graph = super::frame::forced_c08_graph_for_test(commands.clone(), context).ok()?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            WorkingFormat::HighPrecision,
            Format::Rgba8,
            &capabilities,
        )
        .ok()?;
        let subset = lowered.c08_executable_subset()?;
        let actual_capture = subset.captures().first()?;
        let capture_pass = lowered
            .passes
            .iter()
            .find(|pass| pass.id == actual_capture.pass())?;
        let RuntimePassKind::VelloCapture(Some(actual_span)) = &capture_pass.kind else {
            return None;
        };
        let mut span = actual_span.clone();
        span.capture_transform = capture_transform;
        span.parent_to_surface = parent_to_surface;
        let target = actual_capture.target();
        let mut spatial = lowered
            .resources
            .iter()
            .find(|resource| resource.id == target)?
            .spatial;
        spatial.device_origin = (-3, -2);
        spatial.texel_origin = Point::new(-3.0 / raster_scale, -2.0 / raster_scale);
        spatial.raster_scale = raster_scale;
        let facts = c08_capture_execution_facts(capture_pass.id, target, &span, spatial)?;

        let expected_transform = capture_transform
            .then(parent_to_surface)
            .ok()?
            .then(
                Transform::translation(-spatial.texel_origin.x(), -spatial.texel_origin.y())
                    .ok()?,
            )
            .ok()?
            .then(Transform::scale(raster_scale, raster_scale).ok()?)
            .ok()?;
        preserves_application_order_formula &= transforms_are_close(
            facts.initial_transform(),
            expected_transform,
            f64::EPSILON * 32.0,
        );

        let local_to_surface = capture_transform.then(parent_to_surface).ok()?;
        let texel = (2_u32, 3_u32);
        let mapped_center = Point::new(
            spatial.texel_origin.x() + (f64::from(texel.0) + 0.5) / raster_scale,
            spatial.texel_origin.y() + (f64::from(texel.1) + 0.5) / raster_scale,
        );
        let local_center = inverse_transform_point(local_to_surface, mapped_center)?;
        let encoded_center = apply_transform(facts.initial_transform(), local_center);
        preserves_signed_texel_center_mapping &= (encoded_center.x() - 2.5).abs() <= 1.0e-12
            && (encoded_center.y() - 3.5).abs() <= 1.0e-12
            && facts.texel_origin() == spatial.texel_origin;

        preserves_capture_execution_facts &= facts.pass() == capture_pass.id
            && facts.target() == target
            && facts.commands() == &commands
            && facts.antialiasing() == antialiasing
            && facts.target_extent() == spatial.device_extent
            && facts.raster_scale() == raster_scale;

        let encoded = super::encode::encode_vello_scene_with_initial_transform(
            facts.commands(),
            facts.initial_transform(),
        )
        .ok()?;
        let encoded_transform = encoded
            .observation_for_test()
            .first_glyph_run_for_test()?
            .transform_components_for_test();
        lowers_scene_with_explicit_initial_transform &= encoded_transform
            .iter()
            .zip(facts.initial_transform().as_array())
            .all(|(actual, expected)| (*actual - expected as f32).abs() <= 1.0e-5);
    }

    Some(BoundedCaptureTransformObservationForTest {
        preserves_application_order_formula,
        preserves_signed_texel_center_mapping,
        covers_required_raster_scales: scales == [1.0, 1.25, 2.0],
        preserves_capture_execution_facts,
        lowers_scene_with_explicit_initial_transform,
    })
}

#[cfg(test)]
fn transforms_are_close(left: Transform, right: Transform, tolerance: f64) -> bool {
    left.as_array()
        .into_iter()
        .zip(right.as_array())
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

#[cfg(test)]
fn apply_transform(transform: Transform, point: Point) -> Point {
    let [a, b, c, d, e, f] = transform.as_array();
    Point::new(
        a * point.x() + c * point.y() + e,
        b * point.x() + d * point.y() + f,
    )
}

#[cfg(test)]
fn inverse_transform_point(transform: Transform, point: Point) -> Option<Point> {
    let [a, b, c, d, e, f] = transform.as_array();
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return None;
    }
    let x = point.x() - e;
    let y = point.y() - f;
    Some(Point::new(
        (d * x - c * y) / determinant,
        (-b * x + a * y) / determinant,
    ))
}

const VELLO_CAPTURE_TEXTURE_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::STORAGE_BINDING
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);
const RESOLVED_MASK_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST);

#[derive(Clone, Debug, Eq, PartialEq)]
enum RuntimeAllocationRequest {
    EffectTexture(EffectTextureDescriptor),
    ResolvedMask(ResolvedMaskUploadDescriptor),
}

impl RuntimeAllocationRequest {
    fn preflight(&self) -> Result<ResourceAllocationPreflight> {
        match self {
            Self::EffectTexture(descriptor) => {
                ResourceAllocationPreflight::effect_texture(*descriptor)
            }
            Self::ResolvedMask(descriptor) => {
                ResourceAllocationPreflight::resolved_mask(descriptor)
            }
        }
    }

    fn acquire(
        &self,
        frame_scope: &mut FrameResourceScope,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capabilities: &DeviceCapabilities,
    ) -> Result<ResourceLease> {
        match self {
            Self::EffectTexture(descriptor) => {
                frame_scope.acquire_effect_texture(device, capabilities, *descriptor)
            }
            Self::ResolvedMask(descriptor) => {
                frame_scope.acquire_resolved_mask_upload(device, queue, capabilities, descriptor)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeResourcePreparationRequest {
    runtime: RuntimeResourceRequest,
    allocation: RuntimeAllocationRequest,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeKernelPreparationRequest {
    key: GaussianKernelKey,
    plan: GaussianKernelPlan,
    last_use: RuntimePassId,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimePassPreparationRequest {
    runtime: RuntimePass,
    spatial_uniform: Option<PassSpatialUniformBytes>,
    cache_keys: Option<RuntimePassCacheKeys>,
    kernel: Option<GaussianKernelKey>,
    kernel_releases: Vec<GaussianKernelKey>,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeGraphPreparationPlan {
    generation: RuntimeGraphGeneration,
    working_format: WorkingFormat,
    output_format: Format,
    resources: Vec<RuntimeResourcePreparationRequest>,
    kernels: Vec<RuntimeKernelPreparationRequest>,
    passes: Vec<RuntimePassPreparationRequest>,
    root_working_image: RuntimeResourceId,
    final_present: RuntimePassId,
    allocation_preflights: Vec<ResourceAllocationPreflight>,
}

impl RuntimeGraphPreparationPlan {
    fn try_derive(
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
    ) -> Result<Self> {
        let selected = capabilities.resolve_effect_working_format(policy)?;
        if selected != lowered.working_format {
            return Err(preparation_error(
                "the lowered graph working format does not match immutable device policy",
            ));
        }

        let mut resource_by_id = BTreeMap::new();
        let mut resource_formats = BTreeMap::new();
        let mut resources = Vec::with_capacity(lowered.resources.len());
        let mut allocation_preflights = Vec::with_capacity(lowered.resources.len());
        for resource in &lowered.resources {
            if resource_by_id.insert(resource.id, resource).is_some() {
                return Err(preparation_error(
                    "duplicate runtime resource reached graph preparation",
                ));
            }
            if runtime_resource_format(resource.role, lowered.working_format) != resource.format {
                return Err(preparation_error(
                    "runtime resource role and working format are inconsistent",
                ));
            }
            if resource.expected_reads == 0 {
                return Err(preparation_error(
                    "a prepared runtime resource has no scheduled reader",
                ));
            }
            let extent = resource.spatial.device_extent;
            if extent.width() == 0 || extent.height() == 0 {
                return Err(preparation_error(
                    "a concrete runtime resource has an empty allocation extent",
                ));
            }
            capabilities.validate_effect_texture_extent(extent)?;

            let allocation = match (&resource.format, &resource.import) {
                (RuntimeResourceFormat::VelloCaptureRgba8Unorm, None)
                    if resource.role == RuntimeResourceRole::CaptureWorkingImage
                        && matches!(resource.producer, RuntimeResourceProducer::Pass(_)) =>
                {
                    let descriptor =
                        EffectTextureDescriptor::try_capture(extent, VELLO_CAPTURE_TEXTURE_USAGES)?;
                    capabilities.validate_effect_texture_allocation(
                        extent,
                        None,
                        descriptor.texture_format(),
                        descriptor.usage(),
                    )?;
                    RuntimeAllocationRequest::EffectTexture(descriptor)
                }
                (RuntimeResourceFormat::Working(format), None)
                    if *format == lowered.working_format
                        && resource.role != RuntimeResourceRole::CaptureWorkingImage
                        && resource.role != RuntimeResourceRole::ImportedImage =>
                {
                    let descriptor = EffectTextureDescriptor::try_working(
                        *format,
                        extent,
                        format.required_usages(),
                    )?;
                    capabilities.validate_effect_texture_allocation(
                        extent,
                        Some(*format),
                        descriptor.texture_format(),
                        descriptor.usage(),
                    )?;
                    RuntimeAllocationRequest::EffectTexture(descriptor)
                }
                (
                    RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
                    Some(RuntimeResourceImport::ResolvedAlphaMask(descriptor)),
                ) if resource.role == RuntimeResourceRole::ImportedImage
                    && matches!(resource.producer, RuntimeResourceProducer::Imported) =>
                {
                    if descriptor.physical_size() != extent {
                        return Err(preparation_error(
                            "resolved-mask upload extent differs from its runtime resource",
                        ));
                    }
                    descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
                    capabilities.validate_effect_texture_allocation(
                        extent,
                        None,
                        wgpu::TextureFormat::Rgba8Unorm,
                        RESOLVED_MASK_TEXTURE_USAGES,
                    )?;
                    RuntimeAllocationRequest::ResolvedMask(descriptor.clone())
                }
                _ => {
                    return Err(preparation_error(
                        "runtime resource has no exact concrete preparation request",
                    ));
                }
            };
            allocation_preflights.push(allocation.preflight()?);
            resource_formats.insert(resource.id, resource.format);
            resources.push(RuntimeResourcePreparationRequest {
                runtime: resource.clone(),
                allocation,
            });
        }

        let mut pass_positions = BTreeMap::new();
        for (position, pass) in lowered.passes.iter().enumerate() {
            if pass_positions.insert(pass.id, position).is_some() {
                return Err(preparation_error(
                    "duplicate runtime pass reached graph preparation",
                ));
            }
        }
        let mut actual_reads = BTreeMap::<RuntimeResourceId, u32>::new();
        let mut actual_last_reads = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        let mut release_passes = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        let mut produced_results = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        let mut kernel_by_pass = BTreeMap::<RuntimePassId, GaussianKernelKey>::new();
        let mut kernels = BTreeMap::<GaussianKernelKey, RuntimeKernelPreparationRequest>::new();

        for (position, pass) in lowered.passes.iter().enumerate() {
            if pass.dependencies.iter().any(|dependency| {
                pass_positions
                    .get(dependency)
                    .is_none_or(|dependency_position| *dependency_position >= position)
            }) {
                return Err(preparation_error(
                    "prepared pass has a missing or forward dependency",
                ));
            }
            let mut pass_reads = BTreeSet::new();
            for read in &pass.reads {
                if !pass_reads.insert(read.resource) {
                    return Err(preparation_error(
                        "prepared pass contains a duplicate runtime read binding",
                    ));
                }
                let resource = resource_by_id.get(&read.resource).ok_or_else(|| {
                    preparation_error("prepared pass names a missing runtime resource")
                })?;
                if let RuntimeResourceProducer::Pass(producer) = resource.producer
                    && pass_positions
                        .get(&producer)
                        .is_none_or(|producer_position| *producer_position >= position)
                {
                    return Err(preparation_error(
                        "prepared pass reads before its runtime resource producer",
                    ));
                }
                let reads = actual_reads.entry(read.resource).or_default();
                *reads = reads
                    .checked_add(1)
                    .ok_or_else(|| preparation_error("prepared runtime read count overflowed"))?;
                actual_last_reads.insert(read.resource, pass.id);
            }
            match pass.result {
                RuntimeResultBinding::Resource(resource_id) => {
                    let resource = resource_by_id.get(&resource_id).ok_or_else(|| {
                        preparation_error("prepared pass result resource is missing")
                    })?;
                    if resource.producer != RuntimeResourceProducer::Pass(pass.id)
                        || pass_reads.contains(&resource_id)
                        || produced_results.insert(resource_id, pass.id).is_some()
                    {
                        return Err(preparation_error(
                            "prepared pass result binding has no unique matching producer",
                        ));
                    }
                }
                RuntimeResultBinding::Output(format) => {
                    if !matches!(pass.kind, RuntimePassKind::Present)
                        || format != lowered.output_format
                    {
                        return Err(preparation_error(
                            "prepared output binding differs from the terminal present target",
                        ));
                    }
                }
                RuntimeResultBinding::Empty => {}
            }
            let expected_cache_keys = runtime_pass_cache_keys(
                &pass.kind,
                &pass.reads,
                pass.result,
                lowered.working_format,
                lowered.output_format,
                &resource_formats,
            )?;
            if expected_cache_keys != pass.cache_keys {
                return Err(preparation_error(
                    "prepared pass cache keys differ from exact runtime lowering",
                ));
            }
            for resource in &pass.releases {
                if !pass_reads.contains(resource)
                    || release_passes.insert(*resource, pass.id).is_some()
                {
                    return Err(preparation_error(
                        "prepared pass release is missing, duplicate, or not a last read",
                    ));
                }
            }

            if let Some(blur) = runtime_blur_for_kernel(&pass.kind) {
                let kernel_plan = GaussianKernelPlan::try_new(
                    blur.standard_deviation,
                    blur.spatial.result.raster_scale,
                    CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
                    GaussianKernelSamplingForm::PairedLinear,
                )?;
                if kernel_plan.key() != blur.kernel
                    || kernel_plan.byte_len() == 0
                    || kernel_plan.byte_len() > device.limits().max_buffer_size
                {
                    return Err(preparation_error(
                        "Gaussian kernel preparation differs from the exact runtime blur plan",
                    ));
                }
                kernel_by_pass.insert(pass.id, blur.kernel);
                match kernels.entry(blur.kernel) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(RuntimeKernelPreparationRequest {
                            key: blur.kernel,
                            plan: kernel_plan,
                            last_use: pass.id,
                        });
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        if entry.get().plan != kernel_plan {
                            return Err(preparation_error(
                                "one Gaussian kernel identity names conflicting plans",
                            ));
                        }
                        entry.get_mut().last_use = pass.id;
                    }
                }
            }
        }

        for resource in &lowered.resources {
            if actual_reads.get(&resource.id).copied().unwrap_or(0) != resource.expected_reads
                || actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
                || release_passes.get(&resource.id).copied() != Some(resource.last_use)
            {
                return Err(preparation_error(
                    "prepared runtime resource lifetime differs from exact lowering",
                ));
            }
            match resource.producer {
                RuntimeResourceProducer::Imported if resource.import.is_some() => {}
                RuntimeResourceProducer::Pass(pass)
                    if resource.import.is_none()
                        && produced_results.get(&resource.id).copied() == Some(pass) => {}
                RuntimeResourceProducer::Imported | RuntimeResourceProducer::Pass(_) => {
                    return Err(preparation_error(
                        "prepared runtime resource producer/import binding is inconsistent",
                    ));
                }
            }
        }

        let root = resource_by_id
            .get(&lowered.root_working_image)
            .ok_or_else(|| preparation_error("prepared root working resource is missing"))?;
        if root.format != RuntimeResourceFormat::Working(lowered.working_format) {
            return Err(preparation_error(
                "prepared root resource does not use the selected working format",
            ));
        }
        if lowered.passes.last().is_none_or(|pass| {
            pass.id != lowered.final_present
                || !matches!(pass.kind, RuntimePassKind::Present)
                || pass.result != RuntimeResultBinding::Output(lowered.output_format)
        }) {
            return Err(preparation_error(
                "prepared graph has no exact terminal present binding",
            ));
        }

        let kernel_releases = kernels
            .values()
            .map(|kernel| (kernel.last_use, kernel.key))
            .fold(
                BTreeMap::<RuntimePassId, Vec<GaussianKernelKey>>::new(),
                |mut releases, (pass, kernel)| {
                    releases.entry(pass).or_default().push(kernel);
                    releases
                },
            );
        let passes = lowered
            .passes
            .iter()
            .map(|pass| {
                let spatial_uniform = prepared_pass_spatial_uniform(
                    pass,
                    &resource_by_id,
                    lowered.root_working_image,
                )?;
                if spatial_uniform.is_some() != pass.cache_keys.is_some() {
                    return Err(preparation_error(
                        "prepared pass spatial bytes and executable cache keys disagree",
                    ));
                }
                Ok(RuntimePassPreparationRequest {
                    runtime: pass.clone(),
                    spatial_uniform,
                    cache_keys: pass.cache_keys.clone(),
                    kernel: kernel_by_pass.get(&pass.id).copied(),
                    kernel_releases: kernel_releases.get(&pass.id).cloned().unwrap_or_default(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let kernels = kernels.into_values().collect::<Vec<_>>();
        for kernel in &kernels {
            allocation_preflights.push(ResourceAllocationPreflight::gaussian_kernel(&kernel.plan)?);
        }

        Ok(Self {
            generation: lowered.generation,
            working_format: lowered.working_format,
            output_format: lowered.output_format,
            resources,
            kernels,
            passes,
            root_working_image: lowered.root_working_image,
            final_present: lowered.final_present,
            allocation_preflights,
        })
    }
}

fn runtime_blur_for_kernel(kind: &RuntimePassKind) -> Option<&RuntimeBlur> {
    match kind {
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            Some(blur)
        }
        _ => None,
    }
}

fn prepared_pass_spatial_uniform(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    root_working_image: RuntimeResourceId,
) -> Result<Option<PassSpatialUniformBytes>> {
    if pass.cache_keys.is_none() {
        return Ok(None);
    }
    let result_spatial = || -> Result<RuntimeSpatialDescriptor> {
        let RuntimeResultBinding::Resource(resource) = pass.result else {
            return Err(preparation_error(
                "custom pass has no concrete runtime result resource",
            ));
        };
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass result spatial binding is missing"))
    };
    let read_spatial = |role| -> Result<RuntimeSpatialDescriptor> {
        let resource = pass
            .reads
            .iter()
            .find(|read| read.role == role)
            .map(|read| read.resource)
            .ok_or_else(|| preparation_error("custom pass source spatial binding is missing"))?;
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass source resource is missing"))
    };

    let (source, destination) = match &pass.kind {
        RuntimePassKind::CanonicalizeCapture => (
            read_spatial(RuntimeReadRole::CaptureSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::CopyBackdrop => (
            read_spatial(RuntimeReadRole::CompletedParent)?,
            result_spatial()?,
        ),
        RuntimePassKind::ColorFilter(Some(filter)) => {
            (filter.spatial.source, filter.spatial.result)
        }
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            (blur.spatial.source, blur.spatial.result)
        }
        RuntimePassKind::DropShadowColorize(Some(shadow)) => {
            (shadow.spatial.source, shadow.spatial.result)
        }
        RuntimePassKind::Composite(Some(_)) => (
            read_spatial(RuntimeReadRole::CompositeSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::Present => {
            let source = read_spatial(RuntimeReadRole::FinalWorkingImage)?;
            let destination = resources
                .get(&root_working_image)
                .map(|resource| resource.spatial)
                .ok_or_else(|| preparation_error("present destination spatial is missing"))?;
            (source, destination)
        }
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => {
            return Err(preparation_error(
                "non-executable pass unexpectedly requested spatial serialization",
            ));
        }
    };
    PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(source, destination).map(Some)
}

fn preparation_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

struct PreparedResourceBinding {
    allocation: RuntimeAllocationRequest,
    lease: Option<ResourceLease>,
}

struct PreparedKernelBinding {
    lease: Option<ResourceLease>,
}

/// One allocation-backed, generation-bound C07 handoff. Its lifetime prevents
/// the ready device bundle from transitioning while C08 owns its frame scope.
pub(crate) struct PreparedGraph<'device> {
    plan: RuntimeGraphPreparationPlan,
    resource_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
    frame_scope: Option<FrameResourceScope>,
    next_pass: usize,
    _ready_device: PhantomData<(
        &'device wgpu::Device,
        &'device wgpu::Queue,
        &'device ResourceManager,
        &'device DevicePassCache,
    )>,
}

impl<'device> PreparedGraph<'device> {
    pub(crate) fn try_prepare(
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        _pass_cache: &'device DevicePassCache,
    ) -> Result<Self> {
        let plan = RuntimeGraphPreparationPlan::try_derive(lowered, policy, capabilities, device)?;
        resources.preflight_graph_acquisitions(&plan.allocation_preflights)?;

        let mut frame_scope = resources.begin_frame()?;
        let mut resource_bindings = BTreeMap::new();
        for request in &plan.resources {
            let lease =
                request
                    .allocation
                    .acquire(&mut frame_scope, device, queue, capabilities)?;
            if resource_bindings
                .insert(
                    request.runtime.id,
                    PreparedResourceBinding {
                        allocation: request.allocation.clone(),
                        lease: Some(lease),
                    },
                )
                .is_some()
            {
                return Err(preparation_error(
                    "one runtime resource acquired more than one concrete binding",
                ));
            }
        }
        let mut kernel_bindings = BTreeMap::new();
        for request in &plan.kernels {
            let lease = frame_scope.acquire_gaussian_kernel_buffer(device, &request.plan)?;
            if kernel_bindings
                .insert(request.key, PreparedKernelBinding { lease: Some(lease) })
                .is_some()
            {
                return Err(preparation_error(
                    "one Gaussian kernel acquired more than one concrete binding",
                ));
            }
        }

        Ok(Self {
            plan,
            resource_bindings,
            kernel_bindings,
            frame_scope: Some(frame_scope),
            next_pass: 0,
            _ready_device: PhantomData,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the typed prepared graph generation for stale-binding checks"
        )
    )]
    pub(crate) const fn generation(&self) -> RuntimeGraphGeneration {
        self.plan.generation
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the selected prepared working format"
        )
    )]
    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.plan.working_format
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "C08 consumes the prepared output format")
    )]
    pub(crate) const fn output_format(&self) -> Format {
        self.plan.output_format
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes the typed prepared root and terminal identities"
        )
    )]
    pub(crate) const fn root_and_final(&self) -> (RuntimeResourceId, RuntimePassId) {
        (self.plan.root_working_image, self.plan.final_present)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes prepared pass requests through this narrow iterator"
        )
    )]
    pub(crate) fn current_pass(&self) -> Option<PreparedPassView<'_>> {
        self.plan
            .passes
            .get(self.next_pass)
            .map(|request| PreparedPassView { request })
    }

    fn require_current_pass(&self, pass: RuntimePassId) -> Result<&RuntimePassPreparationRequest> {
        let request = self
            .plan
            .passes
            .get(self.next_pass)
            .ok_or_else(|| preparation_error("prepared graph has no remaining pass"))?;
        if request.runtime.id != pass {
            return Err(preparation_error(
                "prepared pass request is missing, stale, duplicate, or out of order",
            ));
        }
        Ok(request)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 inspects exact prepared texture bindings before encoding each pass"
        )
    )]
    pub(crate) fn texture_binding_for_pass(
        &self,
        pass: RuntimePassId,
        resource: RuntimeResourceId,
    ) -> Result<PreparedTextureBinding<'_>> {
        let request = self.require_current_pass(pass)?;
        let bound = request
            .runtime
            .reads
            .iter()
            .any(|read| read.resource == resource)
            || request.runtime.result == RuntimeResultBinding::Resource(resource);
        if !bound {
            return Err(preparation_error(
                "runtime resource is not bound to the requested prepared pass",
            ));
        }
        let binding = self
            .resource_bindings
            .get(&resource)
            .ok_or_else(|| preparation_error("prepared runtime resource binding is missing"))?;
        let lease = binding.lease.as_ref().ok_or_else(|| {
            preparation_error("prepared runtime resource binding is stale or already released")
        })?;
        let frame_scope = self
            .frame_scope
            .as_ref()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?;
        let (texture, view) = match &binding.allocation {
            RuntimeAllocationRequest::EffectTexture(_) => frame_scope.effect_texture(lease)?,
            RuntimeAllocationRequest::ResolvedMask(_) => {
                frame_scope.resolved_mask_texture(lease)?
            }
        };
        Ok(PreparedTextureBinding {
            runtime_resource: resource,
            allocation_resource: lease.resource_identity(),
            texture,
            view,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 inspects exact prepared Gaussian bindings before blur encoding"
        )
    )]
    pub(crate) fn gaussian_kernel_binding_for_pass(
        &self,
        pass: RuntimePassId,
    ) -> Result<Option<PreparedGaussianKernelBinding<'_>>> {
        let request = self.require_current_pass(pass)?;
        let Some(kernel) = request.kernel else {
            return Ok(None);
        };
        let binding = self
            .kernel_bindings
            .get(&kernel)
            .ok_or_else(|| preparation_error("prepared Gaussian kernel binding is missing"))?;
        let lease = binding.lease.as_ref().ok_or_else(|| {
            preparation_error("prepared Gaussian kernel binding is stale or already released")
        })?;
        let frame_scope = self
            .frame_scope
            .as_ref()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?;
        Ok(Some(PreparedGaussianKernelBinding {
            key: kernel,
            allocation_resource: lease.resource_identity(),
            buffer: frame_scope.gaussian_kernel_buffer(lease)?,
        }))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 resolves prepared leases at each validated runtime last-use point"
        )
    )]
    pub(crate) fn complete_pass(&mut self, pass: RuntimePassId) -> Result<()> {
        let request = self.require_current_pass(pass)?;
        let resource_releases = request.runtime.releases.clone();
        let kernel_releases = request.kernel_releases.clone();

        for resource in &resource_releases {
            let binding = self
                .resource_bindings
                .get(resource)
                .ok_or_else(|| preparation_error("prepared runtime release binding is missing"))?;
            if binding.lease.is_none() {
                return Err(preparation_error(
                    "prepared runtime release is stale or duplicate",
                ));
            }
        }
        for kernel in &kernel_releases {
            let binding = self
                .kernel_bindings
                .get(kernel)
                .ok_or_else(|| preparation_error("prepared Gaussian release binding is missing"))?;
            if binding.lease.is_none() {
                return Err(preparation_error(
                    "prepared Gaussian release is stale or duplicate",
                ));
            }
        }

        let Self {
            resource_bindings,
            kernel_bindings,
            frame_scope,
            ..
        } = self;
        let mut leases = Vec::with_capacity(resource_releases.len() + kernel_releases.len());
        for resource in &resource_releases {
            let lease = resource_bindings
                .get(resource)
                .and_then(|binding| binding.lease.as_ref())
                .expect("prepared resource releases were validated before atomic resolution");
            leases.push(lease);
        }
        for kernel in &kernel_releases {
            let lease = kernel_bindings
                .get(kernel)
                .and_then(|binding| binding.lease.as_ref())
                .expect("prepared kernel releases were validated before atomic resolution");
            leases.push(lease);
        }
        frame_scope
            .as_mut()
            .ok_or_else(|| preparation_error("prepared frame resource scope is closed"))?
            .resolve_leases_atomically(&leases)?;
        for resource in resource_releases {
            let _ = resource_bindings
                .get_mut(&resource)
                .and_then(|binding| binding.lease.take())
                .expect("atomically resolved prepared resource must remain bound");
        }
        for kernel in kernel_releases {
            let _ = kernel_bindings
                .get_mut(&kernel)
                .and_then(|binding| binding.lease.take())
                .expect("atomically resolved prepared kernel must remain bound");
        }
        self.next_pass = self.next_pass.saturating_add(1);
        Ok(())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 finishes a completely released prepared frame scope after execution"
        )
    )]
    pub(crate) fn finish(mut self) -> Result<FrameCleanup> {
        if self.next_pass != self.plan.passes.len()
            || self
                .resource_bindings
                .values()
                .any(|binding| binding.lease.is_some())
            || self
                .kernel_bindings
                .values()
                .any(|binding| binding.lease.is_some())
        {
            return Err(preparation_error(
                "prepared graph cannot finish before every pass and last-use release",
            ));
        }
        self.frame_scope
            .take()
            .ok_or_else(|| preparation_error("prepared frame resource scope is already closed"))
            .map(FrameResourceScope::finish)
    }

    #[cfg(test)]
    pub(crate) fn allocation_identities_for_test(&self) -> PreparedAllocationIdentitiesForTest {
        PreparedAllocationIdentitiesForTest {
            resources: self
                .resource_bindings
                .iter()
                .map(|(runtime, binding)| {
                    (
                        *runtime,
                        binding
                            .lease
                            .as_ref()
                            .expect("new prepared resources must own live leases")
                            .resource_identity(),
                    )
                })
                .collect(),
            kernels: self
                .kernel_bindings
                .iter()
                .map(|(kernel, binding)| {
                    (
                        *kernel,
                        binding
                            .lease
                            .as_ref()
                            .expect("new prepared kernels must own live leases")
                            .resource_identity(),
                    )
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn exercise_for_test(&mut self) -> Result<PreparedGraphExerciseObservationForTest> {
        let _ = (
            self.generation(),
            self.working_format(),
            self.output_format(),
            self.root_and_final(),
        );
        let mut vocabulary = [false; 10];
        for pass in &self.plan.passes {
            vocabulary[runtime_pass_kind_index(&pass.runtime.kind)] = true;
        }
        let complete_resource_and_pass_handoff = self
            .plan
            .resources
            .iter()
            .find(|resource| resource.runtime.role == RuntimeResourceRole::RootWorkingImage)
            .map(|resource| resource.runtime.id)
            == Some(self.plan.root_working_image)
            && self
                .plan
                .passes
                .last()
                .is_some_and(|pass| pass.runtime.id == self.plan.final_present)
            && vocabulary.into_iter().all(|present| present)
            && !self.plan.resources.is_empty()
            && !self.plan.kernels.is_empty();

        let mut has_capture = false;
        let mut has_working = false;
        let mut has_mask = false;
        let exact_resources = self.plan.resources.iter().all(|request| {
            match (&request.runtime.format, &request.allocation) {
                (
                    RuntimeResourceFormat::VelloCaptureRgba8Unorm,
                    RuntimeAllocationRequest::EffectTexture(descriptor),
                ) => {
                    has_capture = true;
                    descriptor.role() == EffectTextureRole::Capture
                        && descriptor.working_format().is_none()
                        && descriptor.texture_format() == wgpu::TextureFormat::Rgba8Unorm
                        && descriptor.usage() == VELLO_CAPTURE_TEXTURE_USAGES
                }
                (
                    RuntimeResourceFormat::Working(format),
                    RuntimeAllocationRequest::EffectTexture(descriptor),
                ) => {
                    has_working = true;
                    descriptor.role() == EffectTextureRole::Working
                        && descriptor.working_format() == Some(*format)
                        && descriptor.texture_format() == format.texture_format()
                        && descriptor.usage() == format.required_usages()
                }
                (
                    RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
                    RuntimeAllocationRequest::ResolvedMask(descriptor),
                ) => {
                    has_mask = true;
                    matches!(
                        &request.runtime.import,
                        Some(RuntimeResourceImport::ResolvedAlphaMask(runtime))
                            if runtime.cache_key() == descriptor.cache_key()
                                && runtime.physical_size() == descriptor.physical_size()
                    )
                }
                _ => false,
            }
        });
        let exact_kernels = self.plan.kernels.iter().all(|kernel| {
            kernel.key == kernel.plan.key()
                && kernel.plan.byte_len() > 0
                && self
                    .plan
                    .passes
                    .iter()
                    .any(|pass| pass.kernel == Some(kernel.key))
        });
        let exact_capture_working_mask_and_kernel_allocations =
            has_capture && has_working && has_mask && exact_resources && exact_kernels;
        let spatial_bytes_and_cache_keys_preserved = self.plan.passes.iter().all(|pass| {
            pass.cache_keys == pass.runtime.cache_keys
                && pass.spatial_uniform.is_some() == pass.cache_keys.is_some()
                && pass
                    .spatial_uniform
                    .as_ref()
                    .is_none_or(|bytes| bytes.as_bytes().len() == 48)
        });

        let initial_pass = self
            .current_pass()
            .ok_or_else(|| preparation_error("prepared test graph has no first pass"))?
            .id();
        let initial_outstanding = self.outstanding_lease_count_for_test();
        let out_of_order_rejected = (self.plan.final_present != initial_pass)
            && self.complete_pass(self.plan.final_present).is_err()
            && self.next_pass == 0
            && self.outstanding_lease_count_for_test() == initial_outstanding;
        let unrelated_resource = self.plan.resources.iter().find_map(|resource| {
            let bound = self.plan.passes[0]
                .runtime
                .reads
                .iter()
                .any(|read| read.resource == resource.runtime.id)
                || self.plan.passes[0].runtime.result
                    == RuntimeResultBinding::Resource(resource.runtime.id);
            (!bound).then_some(resource.runtime.id)
        });
        let missing_binding_rejected = unrelated_resource.is_some_and(|resource| {
            self.texture_binding_for_pass(initial_pass, resource)
                .is_err()
                && self.next_pass == 0
                && self.outstanding_lease_count_for_test() == initial_outstanding
        });

        let mut all_bindings_inspected = true;
        let mut releases_are_exact = true;
        let mut duplicate_release_rejected = false;
        let mut completed = 0_usize;
        while let Some(pass) = self.current_pass() {
            let pass_id = pass.id();
            let _ = (pass.kind(), pass.dependencies(), pass.result());
            all_bindings_inspected &= pass.reads().iter().all(|read| {
                let _ = (
                    read.role(),
                    read.resource(),
                    read.sampling_filter(),
                    read.sampling_edge(),
                    read.sampler_key(),
                );
                true
            });
            all_bindings_inspected &= pass
                .spatial_uniform()
                .is_some_and(|bytes| bytes.as_bytes().len() == 48)
                == pass.cache_keys().is_some();
            if let Some(keys) = pass.cache_keys() {
                let _ = (
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                );
            }
            let bound_resources = pass.bound_resources_for_test();
            let resource_releases = pass.resource_releases_for_test().to_vec();
            let kernel_releases = pass.kernel_releases_for_test().to_vec();
            for resource in bound_resources {
                let binding = self.texture_binding_for_pass(pass_id, resource)?;
                all_bindings_inspected &= binding.runtime_resource() == resource
                    && binding.allocation_resource().get() > 0
                    && binding.texture().width() > 0;
                let _ = binding.view();
            }
            if let Some(binding) = self.gaussian_kernel_binding_for_pass(pass_id)? {
                all_bindings_inspected &= binding.allocation_resource().get() > 0
                    && self
                        .plan
                        .passes
                        .get(self.next_pass)
                        .is_some_and(|request| request.kernel == Some(binding.key()));
                let _ = binding.buffer();
            }
            self.complete_pass(pass_id)?;
            releases_are_exact &= resource_releases.iter().all(|resource| {
                self.resource_bindings
                    .get(resource)
                    .is_some_and(|binding| binding.lease.is_none())
            }) && kernel_releases.iter().all(|kernel| {
                self.kernel_bindings
                    .get(kernel)
                    .is_some_and(|binding| binding.lease.is_none())
            });
            if completed == 0 {
                let after_first = self.outstanding_lease_count_for_test();
                duplicate_release_rejected = self.complete_pass(pass_id).is_err()
                    && self.outstanding_lease_count_for_test() == after_first
                    && self.next_pass == 1;
            }
            completed = completed.saturating_add(1);
        }
        let typed_bindings_and_last_use_releases = out_of_order_rejected
            && missing_binding_rejected
            && duplicate_release_rejected
            && all_bindings_inspected
            && releases_are_exact
            && completed == self.plan.passes.len()
            && self.outstanding_lease_count_for_test() == 0;

        Ok(PreparedGraphExerciseObservationForTest {
            complete_resource_and_pass_handoff,
            exact_capture_working_mask_and_kernel_allocations,
            typed_bindings_and_last_use_releases,
            spatial_bytes_and_cache_keys_preserved,
        })
    }

    #[cfg(test)]
    fn outstanding_lease_count_for_test(&self) -> usize {
        self.resource_bindings
            .values()
            .filter(|binding| binding.lease.is_some())
            .count()
            + self
                .kernel_bindings
                .values()
                .filter(|binding| binding.lease.is_some())
                .count()
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes this immutable view of the current prepared runtime pass"
    )
)]
pub(crate) struct PreparedPassView<'prepared> {
    request: &'prepared RuntimePassPreparationRequest,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 consumes these narrow immutable prepared-pass facts"
    )
)]
impl PreparedPassView<'_> {
    pub(crate) const fn id(&self) -> RuntimePassId {
        self.request.runtime.id
    }

    pub(crate) const fn kind(&self) -> &RuntimePassKind {
        &self.request.runtime.kind
    }

    pub(crate) fn dependencies(&self) -> &[RuntimePassId] {
        &self.request.runtime.dependencies
    }

    pub(crate) fn reads(&self) -> &[RuntimeReadBinding] {
        &self.request.runtime.reads
    }

    pub(crate) const fn result(&self) -> RuntimeResultBinding {
        self.request.runtime.result
    }

    pub(crate) const fn spatial_uniform(&self) -> Option<&PassSpatialUniformBytes> {
        self.request.spatial_uniform.as_ref()
    }

    pub(crate) const fn cache_keys(&self) -> Option<&RuntimePassCacheKeys> {
        self.request.cache_keys.as_ref()
    }

    #[cfg(test)]
    fn bound_resources_for_test(&self) -> Vec<RuntimeResourceId> {
        let mut resources = self
            .request
            .runtime
            .reads
            .iter()
            .map(|read| read.resource)
            .collect::<Vec<_>>();
        if let RuntimeResultBinding::Resource(resource) = self.request.runtime.result {
            resources.push(resource);
        }
        resources
    }

    #[cfg(test)]
    fn resource_releases_for_test(&self) -> &[RuntimeResourceId] {
        &self.request.runtime.releases
    }

    #[cfg(test)]
    fn kernel_releases_for_test(&self) -> &[GaussianKernelKey] {
        &self.request.kernel_releases
    }
}

pub(crate) struct PreparedTextureBinding<'prepared> {
    runtime_resource: RuntimeResourceId,
    allocation_resource: ResourceIdentity,
    texture: &'prepared wgpu::Texture,
    view: &'prepared wgpu::TextureView,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 reads these exact typed texture binding facts during pass encoding"
    )
)]
impl<'prepared> PreparedTextureBinding<'prepared> {
    pub(crate) const fn runtime_resource(&self) -> RuntimeResourceId {
        self.runtime_resource
    }

    pub(crate) const fn allocation_resource(&self) -> ResourceIdentity {
        self.allocation_resource
    }

    pub(crate) const fn texture(&self) -> &'prepared wgpu::Texture {
        self.texture
    }

    pub(crate) const fn view(&self) -> &'prepared wgpu::TextureView {
        self.view
    }
}

pub(crate) struct PreparedGaussianKernelBinding<'prepared> {
    key: GaussianKernelKey,
    allocation_resource: ResourceIdentity,
    buffer: &'prepared wgpu::Buffer,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C08 reads these exact typed Gaussian binding facts during blur encoding"
    )
)]
impl<'prepared> PreparedGaussianKernelBinding<'prepared> {
    pub(crate) const fn key(&self) -> GaussianKernelKey {
        self.key
    }

    pub(crate) const fn allocation_resource(&self) -> ResourceIdentity {
        self.allocation_resource
    }

    pub(crate) const fn buffer(&self) -> &'prepared wgpu::Buffer {
        self.buffer
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedAllocationIdentitiesForTest {
    resources: Vec<(RuntimeResourceId, ResourceIdentity)>,
    kernels: Vec<(GaussianKernelKey, ResourceIdentity)>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedGraphExerciseObservationForTest {
    pub(crate) complete_resource_and_pass_handoff: bool,
    pub(crate) exact_capture_working_mask_and_kernel_allocations: bool,
    pub(crate) typed_bindings_and_last_use_releases: bool,
    pub(crate) spatial_bytes_and_cache_keys_preserved: bool,
}

fn lowering_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

const fn runtime_resource_role(role: GraphLoweringResourceRole) -> RuntimeResourceRole {
    match role {
        GraphLoweringResourceRole::RootWorkingImage => RuntimeResourceRole::RootWorkingImage,
        GraphLoweringResourceRole::CaptureWorkingImage => RuntimeResourceRole::CaptureWorkingImage,
        GraphLoweringResourceRole::IsolationWorkingImage => {
            RuntimeResourceRole::IsolationWorkingImage
        }
        GraphLoweringResourceRole::ImportedImage => RuntimeResourceRole::ImportedImage,
        GraphLoweringResourceRole::BackdropCopy => RuntimeResourceRole::BackdropCopy,
        GraphLoweringResourceRole::FilterIntermediate => RuntimeResourceRole::FilterIntermediate,
        GraphLoweringResourceRole::ShadowImage => RuntimeResourceRole::ShadowImage,
        GraphLoweringResourceRole::CompositeResult => RuntimeResourceRole::CompositeResult,
    }
}

const fn runtime_resource_format(
    role: RuntimeResourceRole,
    working_format: WorkingFormat,
) -> RuntimeResourceFormat {
    match role {
        RuntimeResourceRole::CaptureWorkingImage => RuntimeResourceFormat::VelloCaptureRgba8Unorm,
        RuntimeResourceRole::ImportedImage => RuntimeResourceFormat::ResolvedMaskRgba8Unorm,
        RuntimeResourceRole::RootWorkingImage
        | RuntimeResourceRole::IsolationWorkingImage
        | RuntimeResourceRole::BackdropCopy
        | RuntimeResourceRole::FilterIntermediate
        | RuntimeResourceRole::ShadowImage
        | RuntimeResourceRole::CompositeResult => RuntimeResourceFormat::Working(working_format),
    }
}

fn runtime_pass_kind(
    kind: GraphLoweringPassKind,
    working_format: WorkingFormat,
) -> RuntimePassKind {
    match kind {
        GraphLoweringPassKind::ClearRoot {
            initialization,
            color,
        } => RuntimePassKind::ClearRoot {
            initialization: match initialization {
                GraphLoweringInitialization::SurfaceBaseColor => {
                    RuntimeInitialization::SurfaceBaseColor
                }
                GraphLoweringInitialization::Transparent => RuntimeInitialization::Transparent,
            },
            color,
        },
        GraphLoweringPassKind::VelloCapture(span) => {
            RuntimePassKind::VelloCapture(span.map(runtime_vello_span))
        }
        GraphLoweringPassKind::CanonicalizeCapture => RuntimePassKind::CanonicalizeCapture,
        GraphLoweringPassKind::CopyBackdrop => RuntimePassKind::CopyBackdrop,
        GraphLoweringPassKind::ColorFilter(filter) => {
            RuntimePassKind::ColorFilter(filter.map(runtime_color_filter))
        }
        GraphLoweringPassKind::BlurHorizontal(blur) => RuntimePassKind::BlurHorizontal(
            blur.map(|blur| runtime_blur(blur, RuntimeBlurAxis::Horizontal, working_format)),
        ),
        GraphLoweringPassKind::BlurVertical(blur) => RuntimePassKind::BlurVertical(
            blur.map(|blur| runtime_blur(blur, RuntimeBlurAxis::Vertical, working_format)),
        ),
        GraphLoweringPassKind::DropShadowColorize(shadow) => {
            RuntimePassKind::DropShadowColorize(shadow.map(runtime_drop_shadow))
        }
        GraphLoweringPassKind::Composite(composite) => {
            RuntimePassKind::Composite(composite.map(runtime_composite))
        }
        GraphLoweringPassKind::Present => RuntimePassKind::Present,
    }
}

fn runtime_vello_span(span: GraphLoweringVelloSpan) -> RuntimeVelloSpan {
    RuntimeVelloSpan {
        scope: match span.scope() {
            GraphLoweringVelloSpanScope::CurrentParent => RuntimeVelloSpanScope::CurrentParent,
            GraphLoweringVelloSpanScope::LayerSource => RuntimeVelloSpanScope::LayerSource,
        },
        commands: span.commands().clone(),
        capture_transform: span.capture_transform(),
        parent_to_surface: span.parent_to_surface(),
        antialiasing: span.antialiasing(),
        captured_before_outer_semantics: span.captured_before_outer_semantics(),
    }
}

fn runtime_filter_spatial(
    spatial: super::frame::GraphLoweringFilterSpatialMapping,
) -> RuntimeFilterSpatialMapping {
    RuntimeFilterSpatialMapping {
        source: RuntimeSpatialDescriptor::from_graph(spatial.source()),
        result: RuntimeSpatialDescriptor::from_graph(spatial.result()),
    }
}

fn runtime_edge(edge: GraphLoweringEdgePolicy) -> RuntimeSamplingEdge {
    match edge {
        GraphLoweringEdgePolicy::NoSampling => RuntimeSamplingEdge::ClampToExtent,
        GraphLoweringEdgePolicy::TransparentBlack => RuntimeSamplingEdge::TransparentBlack,
        GraphLoweringEdgePolicy::SemanticBorderMirror(bounds) => {
            RuntimeSamplingEdge::SemanticBorderMirror(bounds)
        }
    }
}

fn runtime_color_filter(filter: GraphLoweringColorFilter) -> RuntimeColorFilter {
    RuntimeColorFilter {
        operations: filter
            .operations()
            .iter()
            .copied()
            .map(|operation| RuntimeColorOperation {
                operation: operation.operation(),
                clamp_boundary: match operation.clamp_boundary() {
                    ColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply => {
                        RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
                    }
                },
            })
            .collect(),
        spatial: runtime_filter_spatial(filter.spatial()),
        edge: runtime_edge(filter.edge()),
    }
}

fn runtime_blur(
    blur: GraphLoweringBlur,
    axis: RuntimeBlurAxis,
    _working_format: WorkingFormat,
) -> RuntimeBlur {
    let spatial = runtime_filter_spatial(blur.spatial());
    let kernel = GaussianKernelKey::from_exact_plan(
        blur.standard_deviation().to_bits(),
        spatial.result.raster_scale.to_bits(),
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS.to_bits(),
        blur.support_radius(),
        GaussianKernelSamplingForm::PairedLinear,
    );
    RuntimeBlur {
        axis,
        input: match blur.input() {
            GraphLoweringBlurInput::Rgba => RuntimeBlurInput::Rgba,
            GraphLoweringBlurInput::SourceAlpha => RuntimeBlurInput::SourceAlpha,
        },
        standard_deviation: blur.standard_deviation(),
        support_radius: blur.support_radius(),
        kernel,
        spatial,
        edge: runtime_edge(blur.edge()),
    }
}

fn runtime_drop_shadow(shadow: GraphLoweringDropShadow) -> RuntimeDropShadow {
    RuntimeDropShadow {
        offset: shadow.offset(),
        standard_deviation: shadow.standard_deviation(),
        color: shadow.color(),
        support_radius: shadow.support_radius(),
        spatial: runtime_filter_spatial(shadow.spatial()),
        edge: runtime_edge(shadow.edge()),
        uses_source_alpha: shadow.uses_source_alpha(),
        uses_continuous_offset: shadow.uses_continuous_offset(),
        retains_unchanged_source: shadow.retains_unchanged_source(),
    }
}

fn runtime_composite(composite: GraphLoweringComposite) -> RuntimeComposite {
    let kind = match composite.kind() {
        GraphLoweringCompositeKind::SpanSourceOver => RuntimeCompositeKind::SpanSourceOver,
        GraphLoweringCompositeKind::Layer {
            transform,
            opacity,
            blend,
            clip,
            outer_clips,
            alpha_mask,
        } => RuntimeCompositeKind::Layer {
            transform: *transform,
            opacity: *opacity,
            blend: *blend,
            clip: clip.clone(),
            outer_clips: outer_clips
                .iter()
                .map(|clip| RuntimeOuterClip {
                    clip: clip.clip().clone(),
                    transform: clip.transform(),
                })
                .collect(),
            alpha_mask: alpha_mask.map(RuntimeResourceId),
        },
        GraphLoweringCompositeKind::DropShadow => RuntimeCompositeKind::DropShadow,
    };
    RuntimeComposite {
        kind,
        source_captured_before_outer_semantics: composite.source_captured_before_outer_semantics(),
    }
}

fn lower_read_bindings(
    reads: &[GraphLoweringReadBinding],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
) -> Result<Vec<RuntimeReadBinding>> {
    let mut seen = BTreeSet::new();
    reads
        .iter()
        .copied()
        .map(|read| {
            let resource = RuntimeResourceId(read.resource());
            if !seen.insert(resource) {
                return Err(lowering_error("duplicate runtime read binding"));
            }
            let resource_request = resource_by_id
                .get(&resource)
                .ok_or_else(|| lowering_error("runtime read resource is missing"))?;
            let source_format = resource_formats
                .get(&resource)
                .copied()
                .ok_or_else(|| lowering_error("runtime read format is missing"))?;
            let role = runtime_read_role(read.role());
            let sampling_filter = match read.sampling_filter() {
                GraphLoweringSamplingFilter::Nearest => RuntimeSamplingFilter::Nearest,
                GraphLoweringSamplingFilter::Linear
                | GraphLoweringSamplingFilter::GaussianKernel
                | GraphLoweringSamplingFilter::ImportedMask => RuntimeSamplingFilter::Linear,
            };
            let sampling_edge = runtime_sampling_edge(read.sampling_edge());
            let resolved_mask_sampling = match read.sampling_filter() {
                GraphLoweringSamplingFilter::ImportedMask => match &resource_request.import {
                    Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) => {
                        Some(upload.cache_key())
                    }
                    None => {
                        return Err(lowering_error(
                            "mask sampler is not bound to an imported resolved mask",
                        ));
                    }
                },
                GraphLoweringSamplingFilter::Nearest
                | GraphLoweringSamplingFilter::Linear
                | GraphLoweringSamplingFilter::GaussianKernel => None,
            };
            let sampler_key = SamplerKey::new(
                shader_binding_role(role),
                source_format.shader_key(),
                match sampling_filter {
                    RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                    RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
                },
                shader_sampling_edge(sampling_edge),
                resolved_mask_sampling,
            );
            Ok(RuntimeReadBinding {
                role,
                resource,
                sampling_filter,
                sampling_edge,
                sampler_key,
            })
        })
        .collect()
}

const fn runtime_read_role(role: GraphLoweringReadRole) -> RuntimeReadRole {
    match role {
        GraphLoweringReadRole::CaptureSource => RuntimeReadRole::CaptureSource,
        GraphLoweringReadRole::CompletedParent => RuntimeReadRole::CompletedParent,
        GraphLoweringReadRole::FilterSource => RuntimeReadRole::FilterSource,
        GraphLoweringReadRole::BlurredSourceAlpha => RuntimeReadRole::BlurredSourceAlpha,
        GraphLoweringReadRole::CompositeParent => RuntimeReadRole::CompositeParent,
        GraphLoweringReadRole::CompositeSource => RuntimeReadRole::CompositeSource,
        GraphLoweringReadRole::AlphaMask => RuntimeReadRole::AlphaMask,
        GraphLoweringReadRole::Shadow => RuntimeReadRole::Shadow,
        GraphLoweringReadRole::FinalWorkingImage => RuntimeReadRole::FinalWorkingImage,
    }
}

const fn runtime_sampling_edge(edge: GraphLoweringSamplingEdge) -> RuntimeSamplingEdge {
    match edge {
        GraphLoweringSamplingEdge::ClampToExtent => RuntimeSamplingEdge::ClampToExtent,
        GraphLoweringSamplingEdge::TransparentBlack => RuntimeSamplingEdge::TransparentBlack,
        GraphLoweringSamplingEdge::SemanticBorderMirror(bounds) => {
            RuntimeSamplingEdge::SemanticBorderMirror(bounds)
        }
    }
}

const fn shader_binding_role(role: RuntimeReadRole) -> ShaderBindingRoleKey {
    match role {
        RuntimeReadRole::CaptureSource => ShaderBindingRoleKey::CaptureSource,
        RuntimeReadRole::CompletedParent => ShaderBindingRoleKey::CompletedParent,
        RuntimeReadRole::FilterSource => ShaderBindingRoleKey::FilterSource,
        RuntimeReadRole::BlurredSourceAlpha => ShaderBindingRoleKey::BlurredSourceAlpha,
        RuntimeReadRole::CompositeParent => ShaderBindingRoleKey::CompositeParent,
        RuntimeReadRole::CompositeSource => ShaderBindingRoleKey::CompositeSource,
        RuntimeReadRole::AlphaMask => ShaderBindingRoleKey::AlphaMask,
        RuntimeReadRole::Shadow => ShaderBindingRoleKey::Shadow,
        RuntimeReadRole::FinalWorkingImage => ShaderBindingRoleKey::FinalWorkingImage,
    }
}

const fn shader_sampling_edge(edge: RuntimeSamplingEdge) -> ShaderSamplingEdgeKey {
    match edge {
        RuntimeSamplingEdge::ClampToExtent => ShaderSamplingEdgeKey::ClampToExtent,
        RuntimeSamplingEdge::TransparentBlack => ShaderSamplingEdgeKey::TransparentBlack,
        RuntimeSamplingEdge::SemanticBorderMirror(_) => ShaderSamplingEdgeKey::SemanticBorderMirror,
    }
}

fn runtime_pass_cache_keys(
    kind: &RuntimePassKind,
    reads: &[RuntimeReadBinding],
    result: RuntimeResultBinding,
    working_format: WorkingFormat,
    output_format: Format,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
) -> Result<Option<RuntimePassCacheKeys>> {
    if matches!(
        kind,
        RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
    ) || result == RuntimeResultBinding::Empty
    {
        return Ok(None);
    }
    let program = shader_program(kind)?;
    let sampled_textures = reads
        .iter()
        .map(|read| {
            let format = resource_formats
                .get(&read.resource)
                .copied()
                .ok_or_else(|| lowering_error("cache key source format is missing"))?;
            Ok((shader_binding_role(read.role), format.shader_key()))
        })
        .collect::<Result<Vec<_>>>()?;
    let samplers = reads
        .iter()
        .map(|read| read.sampler_key)
        .collect::<Vec<_>>();
    let layout = BindGroupLayoutKey::new(program, &sampled_textures, shader_data_bindings(kind));
    let output_key = matches!(kind, RuntimePassKind::Present)
        .then(|| ShaderTextureFormatKey::output(output_format));
    let shader = ShaderModuleKey::new(
        program,
        layout.clone(),
        samplers.clone(),
        Some(ShaderTextureFormatKey::working(working_format)),
        output_key,
    );
    let target_format = match result {
        RuntimeResultBinding::Resource(resource) => resource_formats
            .get(&resource)
            .copied()
            .map(RuntimeResourceFormat::shader_key)
            .ok_or_else(|| lowering_error("cache key target format is missing"))?,
        RuntimeResultBinding::Output(format) => ShaderTextureFormatKey::output(format),
        RuntimeResultBinding::Empty => {
            return Err(lowering_error("executable pass has no target binding"));
        }
    };
    let pipeline = RenderPipelineKey::new(
        shader.clone(),
        layout.clone(),
        samplers.clone(),
        target_format,
    );
    Ok(Some(RuntimePassCacheKeys {
        samplers,
        layout,
        shader,
        pipeline,
    }))
}

fn shader_program(kind: &RuntimePassKind) -> Result<ShaderProgramKey> {
    match kind {
        RuntimePassKind::CanonicalizeCapture => Ok(ShaderProgramKey::CanonicalizeCapture),
        RuntimePassKind::CopyBackdrop => Ok(ShaderProgramKey::CopyBackdrop),
        RuntimePassKind::ColorFilter(Some(_)) => Ok(ShaderProgramKey::ColorFilter),
        RuntimePassKind::BlurHorizontal(Some(blur)) => Ok(ShaderProgramKey::BlurHorizontal {
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
        }),
        RuntimePassKind::BlurVertical(Some(blur)) => Ok(ShaderProgramKey::BlurVertical {
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
        }),
        RuntimePassKind::DropShadowColorize(Some(_)) => Ok(ShaderProgramKey::DropShadowColorize),
        RuntimePassKind::Composite(Some(composite)) => Ok(ShaderProgramKey::Composite(
            shader_composite_key(&composite.kind),
        )),
        RuntimePassKind::Present => Ok(ShaderProgramKey::Present),
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Err(lowering_error(
            "a specialized or empty pass requested a custom shader key",
        )),
    }
}

fn shader_composite_key(kind: &RuntimeCompositeKind) -> ShaderCompositeKey {
    match kind {
        RuntimeCompositeKind::SpanSourceOver => ShaderCompositeKey::SpanSourceOver,
        RuntimeCompositeKind::Layer {
            blend,
            clip,
            outer_clips,
            alpha_mask,
            ..
        } => ShaderCompositeKey::Layer {
            blend: shader_blend_key(*blend),
            has_clip: clip.is_some(),
            has_outer_clips: !outer_clips.is_empty(),
            has_alpha_mask: alpha_mask.is_some(),
        },
        RuntimeCompositeKind::DropShadow => ShaderCompositeKey::DropShadow,
    }
}

const fn shader_blend_key(blend: BlendMode) -> ShaderBlendKey {
    match blend {
        BlendMode::Normal => ShaderBlendKey::Normal,
        BlendMode::Multiply => ShaderBlendKey::Multiply,
        BlendMode::Screen => ShaderBlendKey::Screen,
        BlendMode::Overlay => ShaderBlendKey::Overlay,
        BlendMode::Darken => ShaderBlendKey::Darken,
        BlendMode::Lighten => ShaderBlendKey::Lighten,
        BlendMode::Plus => ShaderBlendKey::Plus,
    }
}

fn shader_data_bindings(kind: &RuntimePassKind) -> Vec<ShaderDataBindingKey> {
    match kind {
        RuntimePassKind::CanonicalizeCapture | RuntimePassKind::CopyBackdrop => {
            vec![ShaderDataBindingKey::SpatialUniform]
        }
        RuntimePassKind::ColorFilter(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::ColorFilterOperations,
        ],
        RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::GaussianKernel,
        ],
        RuntimePassKind::DropShadowColorize(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::DropShadowParameters,
        ],
        RuntimePassKind::Composite(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::CompositeParameters,
        ],
        RuntimePassKind::Present => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::PresentParameters,
        ],
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Vec::new(),
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RuntimeLoweringObservationForTest {
    pub(crate) has_exact_closed_vocabulary: bool,
    pub(crate) preserves_backend_ready_resource_facts: bool,
    pub(crate) preserves_semantic_pass_facts: bool,
    pub(crate) preserves_topological_bindings: bool,
    pub(crate) preserves_exact_last_use_releases: bool,
    pub(crate) rejects_inconsistent_bindings_atomically: bool,
    pub(crate) has_exact_cache_keys: bool,
    pub(crate) keys_separate_program_layout_sampling_and_edge: bool,
    pub(crate) keys_separate_source_working_and_output_formats: bool,
}

#[cfg(test)]
pub(crate) fn runtime_lowering_observation_for_test(
    commands: RenderCommands,
    surface_size: super::Size,
    surface_scale: f64,
    antialiasing: Antialiasing,
    base_color: Color,
    output_format: Format,
    capabilities: DeviceCapabilities,
) -> Result<RuntimeLoweringObservationForTest> {
    let context = FrameContext::try_new(surface_size, surface_scale, antialiasing, base_color)?;
    let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
        return Err(lowering_error(
            "the lowering fixture did not produce a GPU graph",
        ));
    };
    let plan = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        output_format,
        &capabilities,
    )?;
    let reduced = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::ReducedPrecision,
        output_format,
        &capabilities,
    )?;
    let alternate_output = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Bgra8,
        &capabilities,
    )?;
    let graph_view = graph.lowering_view()?;

    let mut vocabulary = [false; 10];
    for pass in &plan.passes {
        vocabulary[runtime_pass_kind_index(&pass.kind)] = true;
    }
    let has_exact_closed_vocabulary = vocabulary.into_iter().all(|present| present);
    let imported_keys = plan
        .resources
        .iter()
        .filter_map(|resource| {
            resource
                .import
                .as_ref()
                .map(|RuntimeResourceImport::ResolvedAlphaMask(upload)| upload.cache_key())
        })
        .collect::<Vec<_>>();
    let graph_imported_keys = graph_view
        .resources()
        .into_iter()
        .filter_map(|resource| {
            resource
                .import()
                .map(|GraphLoweringImportView::ResolvedAlphaMask(upload)| upload.cache_key())
        })
        .collect::<Vec<_>>();
    let has_distinct_formats = plan
        .resources
        .iter()
        .any(|resource| resource.format == RuntimeResourceFormat::VelloCaptureRgba8Unorm)
        && plan.resources.iter().any(|resource| {
            resource.format == RuntimeResourceFormat::Working(WorkingFormat::HighPrecision)
        })
        && plan
            .resources
            .iter()
            .any(|resource| resource.format == RuntimeResourceFormat::ResolvedMaskRgba8Unorm);
    let over_limit = DeviceCapabilities::from_test_facts(true, true, 1);
    let extent_rejected = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        WorkingFormat::HighPrecision,
        output_format,
        &over_limit,
    )
    .is_err();
    let preserves_backend_ready_resource_facts = plan.working_format
        == WorkingFormat::HighPrecision
        && plan.output_format == output_format
        && plan.resources.iter().all(|resource| {
            resource.spatial.device_extent.width() > 0
                && resource.spatial.device_extent.height() > 0
                && resource.expected_reads > 0
        })
        && has_distinct_formats
        && imported_keys == graph_imported_keys
        && !imported_keys.is_empty()
        && extent_rejected;
    let preserves_semantic_pass_facts = plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::VelloCapture(Some(span))
                if !span.commands.commands.is_empty()
                    && span.captured_before_outer_semantics
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::ColorFilter(Some(filter)) if !filter.operations.is_empty()
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::DropShadowColorize(Some(shadow))
                if shadow.uses_source_alpha
                    && shadow.uses_continuous_offset
                    && shadow.retains_unchanged_source
        )
    }) && plan.passes.iter().any(|pass| {
        matches!(
            &pass.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer {
                    alpha_mask: Some(_),
                    ..
                },
                ..
            }))
        )
    });

    let graph_passes = graph_view.passes();
    let preserves_topological_bindings = graph_passes.len() == plan.passes.len()
        && graph_passes
            .iter()
            .zip(&plan.passes)
            .all(|(graph_pass, runtime_pass)| {
                RuntimePassId(graph_pass.id()) == runtime_pass.id
                    && graph_pass
                        .dependencies()
                        .into_iter()
                        .map(RuntimePassId)
                        .collect::<Vec<_>>()
                        == runtime_pass.dependencies
                    && graph_pass.reads().is_ok_and(|reads| {
                        reads
                            .into_iter()
                            .map(|read| RuntimeResourceId(read.resource()))
                            .collect::<Vec<_>>()
                            == runtime_pass
                                .reads
                                .iter()
                                .map(|read| read.resource)
                                .collect::<Vec<_>>()
                    })
            });
    let expected_releases = graph_view
        .resources()
        .into_iter()
        .map(|resource| {
            (
                RuntimeResourceId(resource.id()),
                RuntimePassId(resource.last_use()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let observed_releases = plan
        .passes
        .iter()
        .flat_map(|pass| {
            pass.releases
                .iter()
                .copied()
                .map(move |resource| (resource, pass.id))
        })
        .collect::<BTreeMap<_, _>>();
    let preserves_exact_last_use_releases = expected_releases == observed_releases
        && plan
            .resources
            .iter()
            .any(|resource| resource.expected_reads > 1);

    let rejects_inconsistent_bindings_atomically =
        lowering_faults_are_rejected(&graph, &capabilities);

    let custom_passes = plan
        .passes
        .iter()
        .filter(|pass| {
            !matches!(
                pass.kind,
                RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
            )
        })
        .collect::<Vec<_>>();
    let has_exact_cache_keys = !custom_passes.is_empty()
        && custom_passes.iter().all(|pass| pass.cache_keys.is_some())
        && plan.passes.iter().all(|pass| {
            matches!(
                pass.kind,
                RuntimePassKind::ClearRoot { .. } | RuntimePassKind::VelloCapture(_)
            ) == pass.cache_keys.is_none()
        })
        && plan.passes.iter().all(|pass| {
            pass.reads.iter().all(|read| {
                let Some(resource) = plan
                    .resources
                    .iter()
                    .find(|resource| resource.id == read.resource)
                else {
                    return false;
                };
                let expected_mask = match &resource.import {
                    Some(RuntimeResourceImport::ResolvedAlphaMask(upload))
                        if read.role == RuntimeReadRole::AlphaMask =>
                    {
                        Some(upload.cache_key())
                    }
                    Some(RuntimeResourceImport::ResolvedAlphaMask(_)) | None => None,
                };
                read.sampler_key.facts_for_test()
                    == (
                        shader_binding_role(read.role),
                        resource.format.shader_key(),
                        match read.sampling_filter {
                            RuntimeSamplingFilter::Nearest => ShaderSamplingFilterKey::Nearest,
                            RuntimeSamplingFilter::Linear => ShaderSamplingFilterKey::Linear,
                        },
                        shader_sampling_edge(read.sampling_edge),
                        expected_mask,
                    )
            })
        });
    let unique_layouts = custom_passes
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| &keys.layout))
        .fold(Vec::new(), |mut unique, key| {
            if !unique.contains(&key) {
                unique.push(key);
            }
            unique
        });
    let unique_shaders = custom_passes
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| &keys.shader))
        .fold(Vec::new(), |mut unique, key| {
            if !unique.contains(&key) {
                unique.push(key);
            }
            unique
        });
    let sampler_edges_are_distinct = custom_passes
        .iter()
        .flat_map(|pass| pass.reads.iter())
        .any(|read| matches!(read.sampling_edge, RuntimeSamplingEdge::TransparentBlack))
        && custom_passes
            .iter()
            .flat_map(|pass| pass.reads.iter())
            .any(|read| {
                matches!(
                    read.sampling_edge,
                    RuntimeSamplingEdge::SemanticBorderMirror(_)
                )
            });
    let keys_separate_program_layout_sampling_and_edge =
        unique_layouts.len() > 3 && unique_shaders.len() > 5 && sampler_edges_are_distinct;

    let main_custom_keys = custom_key_map(&plan);
    let reduced_custom_keys = custom_key_map(&reduced);
    let alternate_output_keys = custom_key_map(&alternate_output);
    let working_changes_every_custom_key = main_custom_keys.iter().all(|(id, keys)| {
        reduced_custom_keys
            .get(id)
            .is_some_and(|other| *other != *keys)
    });
    let output_changes_only_present = plan.passes.iter().all(|pass| {
        let Some(main_keys) = pass.cache_keys.as_ref() else {
            return true;
        };
        let Some(other_keys) = alternate_output_keys.get(&pass.id) else {
            return false;
        };
        if matches!(pass.kind, RuntimePassKind::Present) {
            *other_keys != main_keys
        } else {
            *other_keys == main_keys
        }
    });
    let keys_separate_source_working_and_output_formats = working_changes_every_custom_key
        && output_changes_only_present
        && plan.root_working_image == RuntimeResourceId(graph_view.root_working_image())
        && plan.final_present == RuntimePassId(graph_view.final_present());

    Ok(RuntimeLoweringObservationForTest {
        has_exact_closed_vocabulary,
        preserves_backend_ready_resource_facts,
        preserves_semantic_pass_facts,
        preserves_topological_bindings,
        preserves_exact_last_use_releases,
        rejects_inconsistent_bindings_atomically,
        has_exact_cache_keys,
        keys_separate_program_layout_sampling_and_edge,
        keys_separate_source_working_and_output_formats,
    })
}

#[cfg(test)]
fn runtime_pass_kind_index(kind: &RuntimePassKind) -> usize {
    match kind {
        RuntimePassKind::ClearRoot { .. } => 0,
        RuntimePassKind::VelloCapture(_) => 1,
        RuntimePassKind::CanonicalizeCapture => 2,
        RuntimePassKind::CopyBackdrop => 3,
        RuntimePassKind::ColorFilter(_) => 4,
        RuntimePassKind::BlurHorizontal(_) => 5,
        RuntimePassKind::BlurVertical(_) => 6,
        RuntimePassKind::DropShadowColorize(_) => 7,
        RuntimePassKind::Composite(_) => 8,
        RuntimePassKind::Present => 9,
    }
}

#[cfg(test)]
fn custom_key_map(plan: &LoweredGraphPlan) -> BTreeMap<RuntimePassId, &RuntimePassCacheKeys> {
    plan.passes
        .iter()
        .filter_map(|pass| pass.cache_keys.as_ref().map(|keys| (pass.id, keys)))
        .collect()
}

#[cfg(test)]
fn lowering_faults_are_rejected(graph: &GpuRenderGraph, capabilities: &DeviceCapabilities) -> bool {
    use super::frame::GraphLoweringFaultForTest;

    [
        GraphLoweringFaultForTest::MissingResourceBinding,
        GraphLoweringFaultForTest::DuplicateReadBinding,
        GraphLoweringFaultForTest::ForwardDependency,
        GraphLoweringFaultForTest::StaleResourceGeneration,
        GraphLoweringFaultForTest::InconsistentLastUse,
    ]
    .into_iter()
    .all(|fault| {
        let invalid = graph.with_lowering_fault_for_test(fault);
        LoweredGraphPlan::try_lower_validated_graph(
            &invalid,
            WorkingFormat::HighPrecision,
            Format::Rgba8,
            capabilities,
        )
        .is_err()
    })
}
