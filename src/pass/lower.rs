use std::collections::{BTreeMap, BTreeSet};

use super::model::{
    LoweredGraphPlan, RuntimeBlur, RuntimeBlurInput, RuntimeColorClampBoundary, RuntimeColorFilter,
    RuntimeColorOperation, RuntimeColorOperationKind, RuntimeComposite, RuntimeCompositeKind,
    RuntimeGraphGeneration, RuntimeInitialization, RuntimeLayerCompositeParameters, RuntimePass,
    RuntimePassCacheKeys, RuntimePassId, RuntimePassKind, RuntimeReadBinding, RuntimeReadRole,
    RuntimeResourceFormat, RuntimeResourceId, RuntimeResourceImport, RuntimeResourceProducer,
    RuntimeResourceRequest, RuntimeResourceRole, RuntimeResultBinding, RuntimeSamplingEdge,
    RuntimeSamplingFilter, RuntimeSpatialDescriptor, RuntimeVelloCapture,
};
use super::{
    RuntimeBlurAxis, RuntimeClipCoverage, RuntimeClipCoverageElement, RuntimeDropShadow,
    RuntimeFilterSpatialMapping, RuntimeOuterClip, RuntimeResolvedAlphaMaskComposition,
    RuntimeVelloSpan, RuntimeVelloSpanScope,
};
use crate::{
    BackendErrorCode, Error, Format, Result,
    backend::DeviceCapabilities,
    filter::{
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS, ColorClampBoundary, RuntimeFilterAmount,
        RuntimeFilterAngle, RuntimeUnitFilterAmount,
    },
    frame::{
        GpuRenderGraph, GraphLoweringBlur, GraphLoweringBlurInput, GraphLoweringClipCoverage,
        GraphLoweringColorFilter, GraphLoweringColorOperation, GraphLoweringComposite,
        GraphLoweringCompositeKind, GraphLoweringDropShadow, GraphLoweringEdgePolicy,
        GraphLoweringFilterSpatialMapping, GraphLoweringImportView, GraphLoweringInitialization,
        GraphLoweringPassKind, GraphLoweringPassResult, GraphLoweringPassView,
        GraphLoweringReadBinding, GraphLoweringReadRole, GraphLoweringResourceProducer,
        GraphLoweringResourceRole, GraphLoweringResourceView, GraphLoweringSamplingEdge,
        GraphLoweringSamplingFilter, GraphLoweringSpatialDescriptor, GraphLoweringVelloCapture,
        GraphLoweringVelloSpan, GraphLoweringVelloSpanScope,
    },
    layer::BlendMode,
    resource::{GaussianKernelKey, GaussianKernelSamplingForm, WorkingFormat},
    shader::{
        BindGroupLayoutKey, RenderPipelineKey, SamplerKey, ShaderBindingRoleKey,
        ShaderCompositeKey, ShaderCompositePathKey, ShaderDataBindingKey, ShaderMaskQualityKey,
        ShaderMaskSamplingKey, ShaderModuleKey, ShaderProgramKey, ShaderSamplingEdgeKey,
        ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    style::ColorFilterOp,
};

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
}

#[derive(Clone, Copy)]
enum GraphLoweringCapabilityValidation<'capabilities> {
    Required(&'capabilities DeviceCapabilities),
    ClassificationOnly,
}

struct LoweredResourceSet {
    resources: Vec<RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
}

fn lower_runtime_resources(
    resource_views: &[GraphLoweringResourceView<'_>],
    pass_ids: &BTreeSet<RuntimePassId>,
    working_format: WorkingFormat,
    capability_validation: GraphLoweringCapabilityValidation<'_>,
) -> Result<LoweredResourceSet> {
    let mut resources = Vec::with_capacity(resource_views.len());
    let mut resource_ids = BTreeSet::new();
    let mut resource_formats = BTreeMap::new();
    for resource in resource_views {
        let resource = *resource;
        let id = RuntimeResourceId(resource.id());
        if !resource_ids.insert(id) {
            return Err(lowering_error("duplicate runtime resource binding"));
        }
        let role = runtime_resource_role(resource.role());
        let spatial = RuntimeSpatialDescriptor::from_graph(resource.spatial());
        let format = runtime_resource_format(role, working_format);
        if let GraphLoweringCapabilityValidation::Required(capabilities) = capability_validation {
            capabilities.validate_effect_texture_extent(spatial.device_extent)?;
            if let RuntimeResourceFormat::Working(format) = format {
                capabilities.validate_effect_texture_allocation(
                    spatial.device_extent,
                    Some(format),
                    format.texture_format(),
                    format.required_usages(),
                )?;
            }
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
    Ok(LoweredResourceSet {
        resources,
        resource_formats,
    })
}

fn lower_runtime_passes(
    pass_views: &[GraphLoweringPassView<'_>],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    releases: &mut BTreeMap<RuntimePassId, Vec<RuntimeResourceId>>,
    working_format: WorkingFormat,
    output_format: Format,
) -> Result<Vec<RuntimePass>> {
    let mut seen_passes = BTreeSet::new();
    let mut passes = Vec::with_capacity(pass_views.len());
    for pass in pass_views {
        let pass = *pass;
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
        let kind = runtime_pass_kind(pass.kind()?, working_format)?;
        let reads = lower_read_bindings(&pass.reads()?, resource_by_id, resource_formats)?;
        let result =
            lower_runtime_result(pass.result(), &kind, &reads, resource_by_id, output_format)?;
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
            resource_formats,
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
    Ok(passes)
}

fn lower_runtime_result(
    result: GraphLoweringPassResult,
    kind: &RuntimePassKind,
    reads: &[RuntimeReadBinding],
    resource_by_id: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    output_format: Format,
) -> Result<RuntimeResultBinding> {
    match result {
        GraphLoweringPassResult::Empty if matches!(kind, RuntimePassKind::Present) => {
            Ok(RuntimeResultBinding::Output(output_format))
        }
        GraphLoweringPassResult::Empty => Ok(RuntimeResultBinding::Empty),
        GraphLoweringPassResult::Resource(resource) => {
            let resource = RuntimeResourceId(resource);
            if !resource_by_id.contains_key(&resource)
                || reads.iter().any(|read| read.resource == resource)
            {
                return Err(lowering_error(
                    "runtime pass result binding is inconsistent",
                ));
            }
            Ok(RuntimeResultBinding::Resource(resource))
        }
    }
}

impl LoweredGraphPlan {
    pub(crate) fn try_lower_validated_graph(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
        capabilities: &DeviceCapabilities,
    ) -> Result<Self> {
        Self::try_lower_validated_graph_inner(
            graph,
            working_format,
            output_format,
            GraphLoweringCapabilityValidation::Required(capabilities),
        )
    }

    pub(super) fn try_lower_for_dispatch_classification(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
    ) -> Result<Self> {
        Self::try_lower_validated_graph_inner(
            graph,
            working_format,
            output_format,
            GraphLoweringCapabilityValidation::ClassificationOnly,
        )
    }

    fn try_lower_validated_graph_inner(
        graph: &GpuRenderGraph,
        working_format: WorkingFormat,
        output_format: Format,
        capability_validation: GraphLoweringCapabilityValidation<'_>,
    ) -> Result<Self> {
        let view = graph.lowering_view()?;
        let resource_views = view.resources();
        let pass_views = view.passes();
        let pass_ids = pass_views
            .iter()
            .map(|pass| RuntimePassId(pass.id()))
            .collect::<BTreeSet<_>>();
        let LoweredResourceSet {
            resources,
            resource_formats,
        } = lower_runtime_resources(
            &resource_views,
            &pass_ids,
            working_format,
            capability_validation,
        )?;

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

        let passes = lower_runtime_passes(
            &pass_views,
            &resource_by_id,
            &resource_formats,
            &mut releases,
            working_format,
            output_format,
        )?;
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

pub(super) fn lowering_error(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

const fn runtime_resource_role(role: GraphLoweringResourceRole) -> RuntimeResourceRole {
    match role {
        GraphLoweringResourceRole::RootWorkingImage => RuntimeResourceRole::RootWorkingImage,
        GraphLoweringResourceRole::CaptureWorkingImage => RuntimeResourceRole::CaptureWorkingImage,
        GraphLoweringResourceRole::ClipCoverage => RuntimeResourceRole::ClipCoverage,
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

pub(super) const fn runtime_resource_format(
    role: RuntimeResourceRole,
    working_format: WorkingFormat,
) -> RuntimeResourceFormat {
    match role {
        RuntimeResourceRole::CaptureWorkingImage => RuntimeResourceFormat::VelloCaptureRgba8Unorm,
        RuntimeResourceRole::ClipCoverage => RuntimeResourceFormat::ClipCoverageRgba8Unorm,
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
) -> Result<RuntimePassKind> {
    Ok(match kind {
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
        GraphLoweringPassKind::VelloCapture(work) => {
            RuntimePassKind::VelloCapture(work.map(runtime_vello_capture))
        }
        GraphLoweringPassKind::CanonicalizeCapture => RuntimePassKind::CanonicalizeCapture,
        GraphLoweringPassKind::CopyBackdrop => RuntimePassKind::CopyBackdrop,
        GraphLoweringPassKind::ColorFilter(filter) => {
            RuntimePassKind::ColorFilter(filter.map(runtime_color_filter).transpose()?)
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
            RuntimePassKind::Composite(composite.map(runtime_composite).transpose()?)
        }
        GraphLoweringPassKind::Present => RuntimePassKind::Present,
    })
}

fn runtime_vello_capture(capture: GraphLoweringVelloCapture) -> RuntimeVelloCapture {
    match capture {
        GraphLoweringVelloCapture::Span(span) => {
            RuntimeVelloCapture::Span(runtime_vello_span(span))
        }
        GraphLoweringVelloCapture::ClipCoverage(coverage) => {
            RuntimeVelloCapture::ClipCoverage(runtime_clip_coverage(coverage))
        }
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

fn runtime_clip_coverage(coverage: GraphLoweringClipCoverage) -> RuntimeClipCoverage {
    RuntimeClipCoverage {
        elements: coverage
            .elements()
            .iter()
            .map(|element| RuntimeClipCoverageElement {
                clip: element.clip().clone(),
                transform: element.transform(),
            })
            .collect(),
        antialiasing: coverage.antialiasing(),
    }
}

fn runtime_filter_spatial(
    spatial: GraphLoweringFilterSpatialMapping,
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

fn runtime_color_filter(filter: GraphLoweringColorFilter) -> Result<RuntimeColorFilter> {
    Ok(RuntimeColorFilter {
        operations: filter
            .operations()
            .iter()
            .copied()
            .map(runtime_color_operation)
            .collect::<Result<Vec<_>>>()?,
        spatial: runtime_filter_spatial(filter.spatial()),
        edge: runtime_edge(filter.edge()),
    })
}

fn runtime_color_operation(
    operation: GraphLoweringColorOperation,
) -> Result<RuntimeColorOperation> {
    let runtime_operation = match operation.operation() {
        ColorFilterOp::Brightness(amount) => {
            RuntimeColorOperationKind::Brightness(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Contrast(amount) => {
            RuntimeColorOperationKind::Contrast(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Grayscale(amount) => RuntimeColorOperationKind::Grayscale(
            RuntimeUnitFilterAmount::try_from_algorithm(amount)?,
        ),
        ColorFilterOp::HueRotate(angle) => {
            RuntimeColorOperationKind::HueRotate(RuntimeFilterAngle::try_from_algorithm(angle)?)
        }
        ColorFilterOp::Invert(amount) => {
            RuntimeColorOperationKind::Invert(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Opacity(amount) => {
            RuntimeColorOperationKind::Opacity(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Saturate(amount) => {
            RuntimeColorOperationKind::Saturate(RuntimeFilterAmount::try_from_algorithm(amount)?)
        }
        ColorFilterOp::Sepia(amount) => {
            RuntimeColorOperationKind::Sepia(RuntimeUnitFilterAmount::try_from_algorithm(amount)?)
        }
    };
    let clamp_boundary = match operation.clamp_boundary() {
        ColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply => {
            RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
        }
    };
    Ok(RuntimeColorOperation {
        operation: runtime_operation,
        clamp_boundary,
    })
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

fn runtime_composite(composite: GraphLoweringComposite) -> Result<RuntimeComposite> {
    let kind = match composite.kind() {
        GraphLoweringCompositeKind::SpanSourceOver => RuntimeCompositeKind::SpanSourceOver,
        GraphLoweringCompositeKind::Layer {
            transform,
            destination_to_layer_local,
            opacity,
            blend,
            clip,
            outer_clips,
            clip_coverage,
            alpha_mask,
        } => {
            let alpha_mask = alpha_mask
                .as_deref()
                .map(|mask| {
                    RuntimeResolvedAlphaMaskComposition::try_new(
                        RuntimeResourceId(mask.resource()),
                        mask.bounds(),
                        mask.image_dimensions(),
                        ShaderMaskSamplingKey::new(mask.quality(), mask.extend()),
                    )
                })
                .transpose()?;
            RuntimeCompositeKind::Layer {
                transform: *transform,
                parameters: Box::new(RuntimeLayerCompositeParameters::try_new(
                    destination_to_layer_local.affine(),
                    *opacity,
                    *blend,
                    clip.is_some() || !outer_clips.is_empty(),
                    alpha_mask,
                )?),
                clip: clip.clone(),
                outer_clips: outer_clips
                    .iter()
                    .map(|clip| RuntimeOuterClip {
                        clip: clip.clip().clone(),
                        transform: clip.transform(),
                    })
                    .collect(),
                clip_coverage: clip_coverage.map(RuntimeResourceId),
            }
        }
        GraphLoweringCompositeKind::DropShadow => RuntimeCompositeKind::DropShadow,
    };
    Ok(RuntimeComposite {
        kind,
        source_captured_before_outer_semantics: composite.source_captured_before_outer_semantics(),
    })
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
            let sampling_edge = runtime_sampling_edge(read.sampling_edge());
            let resolved_mask_sampling = match read.sampling_filter() {
                GraphLoweringSamplingFilter::ImportedMask => match &resource_request.import {
                    Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) => Some(
                        ShaderMaskSamplingKey::new(upload.quality(), upload.extend()),
                    ),
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
            let sampling_filter = match (read.sampling_filter(), resolved_mask_sampling) {
                (GraphLoweringSamplingFilter::Nearest, _) => RuntimeSamplingFilter::Nearest,
                (GraphLoweringSamplingFilter::ImportedMask, Some(sampling))
                    if sampling.quality() == ShaderMaskQualityKey::Low =>
                {
                    RuntimeSamplingFilter::Nearest
                }
                (GraphLoweringSamplingFilter::Linear, _)
                | (GraphLoweringSamplingFilter::GaussianKernel, _)
                | (GraphLoweringSamplingFilter::ImportedMask, Some(_)) => {
                    RuntimeSamplingFilter::Linear
                }
                (GraphLoweringSamplingFilter::ImportedMask, None) => {
                    return Err(lowering_error(
                        "mask sampling policy is missing from an imported mask read",
                    ));
                }
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
        GraphLoweringReadRole::ClipCoverage => RuntimeReadRole::ClipCoverage,
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

pub(super) const fn shader_binding_role(role: RuntimeReadRole) -> ShaderBindingRoleKey {
    match role {
        RuntimeReadRole::CaptureSource => ShaderBindingRoleKey::CaptureSource,
        RuntimeReadRole::CompletedParent => ShaderBindingRoleKey::CompletedParent,
        RuntimeReadRole::FilterSource => ShaderBindingRoleKey::FilterSource,
        RuntimeReadRole::BlurredSourceAlpha => ShaderBindingRoleKey::BlurredSourceAlpha,
        RuntimeReadRole::CompositeParent => ShaderBindingRoleKey::CompositeParent,
        RuntimeReadRole::CompositeSource => ShaderBindingRoleKey::CompositeSource,
        RuntimeReadRole::ClipCoverage => ShaderBindingRoleKey::ClipCoverage,
        RuntimeReadRole::AlphaMask => ShaderBindingRoleKey::AlphaMask,
        RuntimeReadRole::Shadow => ShaderBindingRoleKey::Shadow,
        RuntimeReadRole::FinalWorkingImage => ShaderBindingRoleKey::FinalWorkingImage,
    }
}

pub(super) const fn shader_sampling_edge(edge: RuntimeSamplingEdge) -> ShaderSamplingEdgeKey {
    match edge {
        RuntimeSamplingEdge::ClampToExtent => ShaderSamplingEdgeKey::ClampToExtent,
        RuntimeSamplingEdge::TransparentBlack => ShaderSamplingEdgeKey::TransparentBlack,
        RuntimeSamplingEdge::SemanticBorderMirror(_) => ShaderSamplingEdgeKey::SemanticBorderMirror,
    }
}

const fn runtime_read_uses_shader_sampler(kind: &RuntimePassKind, role: RuntimeReadRole) -> bool {
    match kind {
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { .. },
            ..
        })) => matches!(role, RuntimeReadRole::CompositeSource),
        _ => true,
    }
}

const fn runtime_read_uses_shader_texture(kind: &RuntimePassKind, role: RuntimeReadRole) -> bool {
    !matches!(
        (kind, role),
        (
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::DropShadow,
                ..
            })),
            RuntimeReadRole::Shadow
        )
    )
}

pub(super) fn runtime_pass_cache_keys(
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
    let composite_samples_parent = matches!(
        kind,
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) if parameters.blend() != BlendMode::Normal
    );
    let sampled_reads = reads
        .iter()
        .filter(|read| {
            runtime_read_uses_shader_texture(kind, read.role)
                && (read.role != RuntimeReadRole::CompositeParent
                    || (!matches!(
                        kind,
                        RuntimePassKind::Composite(Some(RuntimeComposite {
                            kind: RuntimeCompositeKind::SpanSourceOver,
                            ..
                        }))
                    ) && composite_samples_parent))
        })
        .collect::<Vec<_>>();
    let sampled_textures = sampled_reads
        .iter()
        .map(|read| {
            let format = resource_formats
                .get(&read.resource)
                .copied()
                .ok_or_else(|| lowering_error("cache key source format is missing"))?;
            Ok((shader_binding_role(read.role), format.shader_key()))
        })
        .collect::<Result<Vec<_>>>()?;
    let samplers = sampled_reads
        .iter()
        .filter(|read| runtime_read_uses_shader_sampler(kind, read.role))
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
            edge: shader_sampling_edge(blur.edge),
        }),
        RuntimePassKind::BlurVertical(Some(blur)) => Ok(ShaderProgramKey::BlurVertical {
            source_alpha: blur.input == RuntimeBlurInput::SourceAlpha,
            edge: shader_sampling_edge(blur.edge),
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
            parameters,
            clip_coverage,
            ..
        } => ShaderCompositeKey::Layer {
            path: if parameters.blend() == BlendMode::Normal {
                ShaderCompositePathKey::Normal
            } else {
                ShaderCompositePathKey::DestinationSampling
            },
            has_clip_coverage: clip_coverage.is_some(),
            has_alpha_mask: parameters.alpha_mask().is_some(),
        },
        RuntimeCompositeKind::DropShadow => ShaderCompositeKey::DropShadow,
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
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            let mut bindings = vec![
                ShaderDataBindingKey::SpatialUniform,
                ShaderDataBindingKey::GaussianKernel,
            ];
            if matches!(blur.edge, RuntimeSamplingEdge::SemanticBorderMirror(_)) {
                bindings.push(ShaderDataBindingKey::BlurEdgeParameters);
            }
            bindings
        }
        RuntimePassKind::DropShadowColorize(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::DropShadowParameters,
        ],
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::DropShadow,
            ..
        })) => vec![ShaderDataBindingKey::SpatialUniform],
        RuntimePassKind::Composite(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::CompositeParameters,
        ],
        RuntimePassKind::Present => vec![ShaderDataBindingKey::SpatialUniform],
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Vec::new(),
    }
}
