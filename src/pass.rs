use std::collections::{BTreeMap, BTreeSet};

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
    renderer::Antialiasing,
    resource::{GaussianKernelKey, GaussianKernelSamplingForm, WorkingFormat},
    shader::{
        BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderBindingRoleKey, ShaderBlendKey,
        ShaderCompositeKey, ShaderDataBindingKey, ShaderModuleKey, ShaderProgramKey,
        ShaderSamplingEdgeKey, ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    style::ColorFilterOp,
};

#[cfg(test)]
use super::frame::{FrameContext, FramePlan};

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

        Ok(Self {
            generation: RuntimeGraphGeneration(view.generation()),
            working_format,
            output_format,
            resources,
            passes,
            root_working_image,
            final_present,
        })
    }
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
