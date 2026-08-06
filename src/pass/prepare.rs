use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use super::super::{
    Format, PhysicalSize, Result,
    backend::DeviceCapabilities,
    filter::CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
    frame::GpuRenderGraph,
    image::ResolvedMaskUploadDescriptor,
    renderer::EffectQualityPolicy,
    resource::{
        FrameResourceScope, GaussianKernelBufferLimits, GaussianKernelKey, GaussianKernelPlan,
        GaussianKernelSamplingForm, ResourceAllocationPreflight, ResourceIdentity, ResourceLease,
        ResourceManager, WorkingFormat,
    },
    shader::{
        BlurEdgeParameterBytes, ColorFilterOperationBufferLimits, ColorFilterOperationBytes,
        CompositeParameterBytes, DevicePassCache, DropShadowParameterBytes,
        PassSpatialUniformBytes, ProvisionalCopyBackdropPassObjects,
        ProvisionalDevicePassCacheUpdate,
    },
    texture::EffectTextureDescriptor,
    vello_engine::VelloEngineState,
};
use super::{
    C08CustomSpineEncodingState, C08ExecutionFacts, C08PreparableGraph, C09PreparableGraph,
    C10PreparableGraph, C11PreparableGraph, C12PreparableGraph,
    ExecutableGraphWorkingFormatRequest, LoweredGraphPlan, RuntimeBlur, RuntimeComposite,
    RuntimeCompositeKind, RuntimeGraphGeneration, RuntimePass, RuntimePassCacheKeys, RuntimePassId,
    RuntimePassKind, RuntimeReadBinding, RuntimeResourceFormat, RuntimeResourceId,
    RuntimeResourceImport, RuntimeResourceProducer, RuntimeResourceRequest, RuntimeResourceRole,
    RuntimeResultBinding,
    close::{
        ClosedExecutableGraph, ClosedExecutableGraphFacts, PrePreparationGraphClassification,
        preparation_error,
    },
    lower::{runtime_pass_cache_keys, runtime_resource_format},
    parameters::{
        prepare_blur_edge_parameters, prepare_color_filter_operations,
        prepare_drop_shadow_parameters, prepared_pass_composite_parameters,
        prepared_pass_spatial_uniform,
    },
};

pub(crate) const VELLO_CAPTURE_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::STORAGE_BINDING
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST);
const RESOLVED_MASK_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum RuntimeAllocationRequest {
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
                ResourceAllocationPreflight::resolved_mask(descriptor)?.ok_or_else(|| {
                    preparation_error(
                        "an explicitly empty resolved mask survived graph contribution pruning",
                    )
                })
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
pub(super) struct RuntimeResourcePreparationRequest {
    pub(super) runtime: RuntimeResourceRequest,
    pub(super) allocation: RuntimeAllocationRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RuntimeKernelPreparationRequest {
    pub(super) key: GaussianKernelKey,
    pub(super) plan: GaussianKernelPlan,
    pub(super) last_use: RuntimePassId,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RuntimePassPreparationRequest {
    pub(super) runtime: RuntimePass,
    pub(super) spatial_uniform: Option<PassSpatialUniformBytes>,
    pub(super) blur_edge_parameters: Option<BlurEdgeParameterBytes>,
    pub(super) color_filter_operations: Option<ColorFilterOperationBytes>,
    pub(super) drop_shadow_parameters: Option<DropShadowParameterBytes>,
    pub(super) composite_parameters: Option<CompositeParameterBytes>,
    pub(super) cache_keys: Option<RuntimePassCacheKeys>,
    pub(super) kernel: Option<GaussianKernelKey>,
    pub(super) kernel_releases: Vec<GaussianKernelKey>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RuntimeGraphPreparationPlan {
    pub(super) generation: RuntimeGraphGeneration,
    pub(super) working_format: WorkingFormat,
    pub(super) output_format: Format,
    pub(super) resources: Vec<RuntimeResourcePreparationRequest>,
    pub(super) kernels: Vec<RuntimeKernelPreparationRequest>,
    pub(super) passes: Vec<RuntimePassPreparationRequest>,
    pub(super) root_working_image: RuntimeResourceId,
    pub(super) final_present: RuntimePassId,
    pub(super) allocation_preflights: Vec<ResourceAllocationPreflight>,
}

impl RuntimeGraphPreparationPlan {
    fn try_derive(
        lowered: LoweredGraphPlan,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
    ) -> Result<Self> {
        Self::try_derive_with_color_filter_limits(
            lowered,
            selected_working_format,
            capabilities,
            device,
            ColorFilterOperationBufferLimits::from_device_limits(&device.limits()),
        )
    }

    fn try_derive_with_color_filter_limits(
        lowered: LoweredGraphPlan,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
        color_filter_limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        capabilities.validate_supported_working_format(selected_working_format)?;
        if selected_working_format != lowered.working_format {
            return Err(preparation_error(
                "the lowered graph working format does not match immutable device policy",
            ));
        }
        let mut prepared_resources = prepare_runtime_resources(&lowered, capabilities)?;
        let pass_facts = analyze_runtime_passes(
            &lowered,
            &prepared_resources.resource_by_id,
            &prepared_resources.resource_formats,
            device,
        )?;
        validate_prepared_graph_lifetimes(
            &lowered,
            &prepared_resources.resource_by_id,
            &pass_facts,
        )?;
        let passes = prepare_runtime_pass_requests(
            &lowered,
            &prepared_resources.resource_by_id,
            &pass_facts,
            color_filter_limits,
        )?;
        let kernels = pass_facts.kernels.into_values().collect::<Vec<_>>();
        for kernel in &kernels {
            prepared_resources
                .allocation_preflights
                .push(ResourceAllocationPreflight::gaussian_kernel(&kernel.plan)?);
        }

        Ok(Self {
            generation: lowered.generation,
            working_format: lowered.working_format,
            output_format: lowered.output_format,
            resources: prepared_resources.resources,
            kernels,
            passes,
            root_working_image: lowered.root_working_image,
            final_present: lowered.final_present,
            allocation_preflights: prepared_resources.allocation_preflights,
        })
    }
}

struct PreparedRuntimeResources<'a> {
    resource_by_id: BTreeMap<RuntimeResourceId, &'a RuntimeResourceRequest>,
    resource_formats: BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    resources: Vec<RuntimeResourcePreparationRequest>,
    allocation_preflights: Vec<ResourceAllocationPreflight>,
}

fn prepare_runtime_resources<'a>(
    lowered: &'a LoweredGraphPlan,
    capabilities: &DeviceCapabilities,
) -> Result<PreparedRuntimeResources<'a>> {
    let mut prepared = PreparedRuntimeResources {
        resource_by_id: BTreeMap::new(),
        resource_formats: BTreeMap::new(),
        resources: Vec::with_capacity(lowered.resources.len()),
        allocation_preflights: Vec::with_capacity(lowered.resources.len()),
    };
    for resource in &lowered.resources {
        if prepared
            .resource_by_id
            .insert(resource.id, resource)
            .is_some()
        {
            return Err(preparation_error(
                "duplicate runtime resource reached graph preparation",
            ));
        }
        validate_runtime_resource_request(resource, lowered.working_format, capabilities)?;
        let allocation =
            prepare_runtime_resource_allocation(resource, lowered.working_format, capabilities)?;
        prepared.allocation_preflights.push(allocation.preflight()?);
        prepared
            .resource_formats
            .insert(resource.id, resource.format);
        prepared.resources.push(RuntimeResourcePreparationRequest {
            runtime: resource.clone(),
            allocation,
        });
    }
    Ok(prepared)
}

fn validate_runtime_resource_request(
    resource: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<()> {
    if runtime_resource_format(resource.role, working_format) != resource.format {
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
    capabilities.validate_effect_texture_extent(extent)
}

fn prepare_runtime_resource_allocation(
    resource: &RuntimeResourceRequest,
    working_format: WorkingFormat,
    capabilities: &DeviceCapabilities,
) -> Result<RuntimeAllocationRequest> {
    let extent = resource.spatial.device_extent;
    match (&resource.format, &resource.import) {
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
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
        }
        (RuntimeResourceFormat::ClipCoverageRgba8Unorm, None)
            if resource.role == RuntimeResourceRole::ClipCoverage
                && matches!(resource.producer, RuntimeResourceProducer::Pass(_)) =>
        {
            let descriptor =
                EffectTextureDescriptor::try_coverage(extent, VELLO_CAPTURE_TEXTURE_USAGES)?;
            capabilities.validate_effect_texture_allocation(
                extent,
                None,
                descriptor.texture_format(),
                descriptor.usage(),
            )?;
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
        }
        (RuntimeResourceFormat::Working(format), None)
            if *format == working_format
                && resource.role != RuntimeResourceRole::CaptureWorkingImage
                && resource.role != RuntimeResourceRole::ClipCoverage
                && resource.role != RuntimeResourceRole::ImportedImage =>
        {
            let descriptor =
                EffectTextureDescriptor::try_working(*format, extent, format.required_usages())?;
            capabilities.validate_effect_texture_allocation(
                extent,
                Some(*format),
                descriptor.texture_format(),
                descriptor.usage(),
            )?;
            Ok(RuntimeAllocationRequest::EffectTexture(descriptor))
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
            Ok(RuntimeAllocationRequest::ResolvedMask(descriptor.clone()))
        }
        _ => Err(preparation_error(
            "runtime resource has no exact concrete preparation request",
        )),
    }
}

struct RuntimePassAnalysis {
    actual_reads: BTreeMap<RuntimeResourceId, u32>,
    actual_last_reads: BTreeMap<RuntimeResourceId, RuntimePassId>,
    release_passes: BTreeMap<RuntimeResourceId, RuntimePassId>,
    produced_results: BTreeMap<RuntimeResourceId, RuntimePassId>,
    kernel_by_pass: BTreeMap<RuntimePassId, GaussianKernelKey>,
    kernels: BTreeMap<GaussianKernelKey, RuntimeKernelPreparationRequest>,
}

fn analyze_runtime_passes(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    device: &wgpu::Device,
) -> Result<RuntimePassAnalysis> {
    let mut pass_positions = BTreeMap::new();
    for (position, pass) in lowered.passes.iter().enumerate() {
        if pass_positions.insert(pass.id, position).is_some() {
            return Err(preparation_error(
                "duplicate runtime pass reached graph preparation",
            ));
        }
    }
    let mut facts = RuntimePassAnalysis {
        actual_reads: BTreeMap::new(),
        actual_last_reads: BTreeMap::new(),
        release_passes: BTreeMap::new(),
        produced_results: BTreeMap::new(),
        kernel_by_pass: BTreeMap::new(),
        kernels: BTreeMap::new(),
    };
    for (position, pass) in lowered.passes.iter().enumerate() {
        let pass_reads = analyze_runtime_pass(
            pass,
            position,
            lowered,
            resources,
            resource_formats,
            &pass_positions,
            &mut facts,
        )?;
        analyze_runtime_kernel(pass, device, &mut facts)?;
        debug_assert!(pass_reads.len() == pass.reads.len());
    }
    Ok(facts)
}

fn analyze_runtime_pass(
    pass: &RuntimePass,
    position: usize,
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    resource_formats: &BTreeMap<RuntimeResourceId, RuntimeResourceFormat>,
    pass_positions: &BTreeMap<RuntimePassId, usize>,
    facts: &mut RuntimePassAnalysis,
) -> Result<BTreeSet<RuntimeResourceId>> {
    if pass.dependencies.iter().any(|dependency| {
        pass_positions
            .get(dependency)
            .is_none_or(|dependency_position| *dependency_position >= position)
    }) {
        return Err(preparation_error(
            "prepared pass has a missing or forward dependency",
        ));
    }
    let pass_reads = analyze_runtime_reads(pass, position, resources, pass_positions, facts)?;
    analyze_runtime_result(pass, lowered, resources, &pass_reads, facts)?;
    let expected_cache_keys = runtime_pass_cache_keys(
        &pass.kind,
        &pass.reads,
        pass.result,
        lowered.working_format,
        lowered.output_format,
        resource_formats,
    )?;
    if expected_cache_keys != pass.cache_keys {
        return Err(preparation_error(
            "prepared pass cache keys differ from exact runtime lowering",
        ));
    }
    for resource in &pass.releases {
        if !pass_reads.contains(resource)
            || facts.release_passes.insert(*resource, pass.id).is_some()
        {
            return Err(preparation_error(
                "prepared pass release is missing, duplicate, or not a last read",
            ));
        }
    }
    Ok(pass_reads)
}

fn analyze_runtime_reads(
    pass: &RuntimePass,
    position: usize,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    pass_positions: &BTreeMap<RuntimePassId, usize>,
    facts: &mut RuntimePassAnalysis,
) -> Result<BTreeSet<RuntimeResourceId>> {
    let mut pass_reads = BTreeSet::new();
    for read in &pass.reads {
        if !pass_reads.insert(read.resource) {
            return Err(preparation_error(
                "prepared pass contains a duplicate runtime read binding",
            ));
        }
        let resource = resources
            .get(&read.resource)
            .ok_or_else(|| preparation_error("prepared pass names a missing runtime resource"))?;
        if let RuntimeResourceProducer::Pass(producer) = resource.producer
            && pass_positions
                .get(&producer)
                .is_none_or(|producer_position| *producer_position >= position)
        {
            return Err(preparation_error(
                "prepared pass reads before its runtime resource producer",
            ));
        }
        let reads = facts.actual_reads.entry(read.resource).or_default();
        *reads = reads
            .checked_add(1)
            .ok_or_else(|| preparation_error("prepared runtime read count overflowed"))?;
        facts.actual_last_reads.insert(read.resource, pass.id);
    }
    Ok(pass_reads)
}

fn analyze_runtime_result(
    pass: &RuntimePass,
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    pass_reads: &BTreeSet<RuntimeResourceId>,
    facts: &mut RuntimePassAnalysis,
) -> Result<()> {
    match pass.result {
        RuntimeResultBinding::Resource(resource_id) => {
            let resource = resources
                .get(&resource_id)
                .ok_or_else(|| preparation_error("prepared pass result resource is missing"))?;
            if resource.producer != RuntimeResourceProducer::Pass(pass.id)
                || pass_reads.contains(&resource_id)
                || facts
                    .produced_results
                    .insert(resource_id, pass.id)
                    .is_some()
            {
                return Err(preparation_error(
                    "prepared pass result binding has no unique matching producer",
                ));
            }
        }
        RuntimeResultBinding::Output(format)
            if !matches!(pass.kind, RuntimePassKind::Present)
                || format != lowered.output_format =>
        {
            return Err(preparation_error(
                "prepared output binding differs from the terminal present target",
            ));
        }
        RuntimeResultBinding::Output(_) | RuntimeResultBinding::Empty => {}
    }
    Ok(())
}

fn analyze_runtime_kernel(
    pass: &RuntimePass,
    device: &wgpu::Device,
    facts: &mut RuntimePassAnalysis,
) -> Result<()> {
    let Some(blur) = runtime_blur_for_kernel(&pass.kind) else {
        return Ok(());
    };
    let kernel_plan = GaussianKernelPlan::try_new(
        blur.standard_deviation,
        blur.spatial.result.raster_scale,
        CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS,
        GaussianKernelSamplingForm::PairedLinear,
    )?;
    if kernel_plan.key() != blur.kernel
        || kernel_plan
            .validate_buffer_limits(GaussianKernelBufferLimits::from_device_limits(
                &device.limits(),
            ))
            .is_err()
    {
        return Err(preparation_error(
            "Gaussian kernel preparation differs from the exact runtime blur plan",
        ));
    }
    facts.kernel_by_pass.insert(pass.id, blur.kernel);
    match facts.kernels.entry(blur.kernel) {
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
    Ok(())
}

fn validate_prepared_graph_lifetimes(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    facts: &RuntimePassAnalysis,
) -> Result<()> {
    for resource in &lowered.resources {
        if facts.actual_reads.get(&resource.id).copied().unwrap_or(0) != resource.expected_reads
            || facts.actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
            || facts.release_passes.get(&resource.id).copied() != Some(resource.last_use)
        {
            return Err(preparation_error(
                "prepared runtime resource lifetime differs from exact lowering",
            ));
        }
        match resource.producer {
            RuntimeResourceProducer::Imported if resource.import.is_some() => {}
            RuntimeResourceProducer::Pass(pass)
                if resource.import.is_none()
                    && facts.produced_results.get(&resource.id).copied() == Some(pass) => {}
            RuntimeResourceProducer::Imported | RuntimeResourceProducer::Pass(_) => {
                return Err(preparation_error(
                    "prepared runtime resource producer/import binding is inconsistent",
                ));
            }
        }
    }
    validate_prepared_graph_root(lowered, resources)
}

fn validate_prepared_graph_root(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
) -> Result<()> {
    let root = resources
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
    Ok(())
}

fn prepare_runtime_pass_requests(
    lowered: &LoweredGraphPlan,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    facts: &RuntimePassAnalysis,
    color_filter_limits: ColorFilterOperationBufferLimits,
) -> Result<Vec<RuntimePassPreparationRequest>> {
    let kernel_releases = facts.kernels.values().fold(
        BTreeMap::<RuntimePassId, Vec<GaussianKernelKey>>::new(),
        |mut releases, kernel| {
            releases
                .entry(kernel.last_use)
                .or_default()
                .push(kernel.key);
            releases
        },
    );
    lowered
        .passes
        .iter()
        .map(|pass| {
            let spatial_uniform =
                prepared_pass_spatial_uniform(pass, resources, lowered.root_working_image)?;
            let blur_edge_parameters = prepare_blur_edge_parameters(pass)?;
            let color_filter_operations =
                prepare_color_filter_operations(pass, color_filter_limits)?;
            let drop_shadow_parameters = prepare_drop_shadow_parameters(pass)?;
            let composite_parameters = prepared_pass_composite_parameters(pass)?;
            if spatial_uniform.is_some() != pass.cache_keys.is_some() {
                return Err(preparation_error(
                    "prepared pass spatial bytes and executable cache keys disagree",
                ));
            }
            Ok(RuntimePassPreparationRequest {
                runtime: pass.clone(),
                spatial_uniform,
                blur_edge_parameters,
                color_filter_operations,
                drop_shadow_parameters,
                composite_parameters,
                cache_keys: pass.cache_keys.clone(),
                kernel: facts.kernel_by_pass.get(&pass.id).copied(),
                kernel_releases: kernel_releases.get(&pass.id).cloned().unwrap_or_default(),
            })
        })
        .collect()
}

fn runtime_blur_for_kernel(kind: &RuntimePassKind) -> Option<&RuntimeBlur> {
    match kind {
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            Some(blur)
        }
        _ => None,
    }
}

pub(super) struct PreparedResourceBinding {
    pub(super) allocation: RuntimeAllocationRequest,
    pub(super) lease: Option<ResourceLease>,
}

pub(super) struct PreparedKernelBinding {
    pub(super) lease: Option<ResourceLease>,
}

pub(super) struct PreparedColorFilterOperationBinding {
    pub(super) bytes: ColorFilterOperationBytes,
    pub(super) buffer: Option<wgpu::Buffer>,
}

pub(super) enum PreparedC11PassObjects {
    CopyBackdrop {
        parent_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
    Blur {
        source_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
    DropShadowColorize {
        source_sampler: wgpu::Sampler,
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::RenderPipeline,
    },
}

struct PreparedPassRealization {
    update: Option<ProvisionalDevicePassCacheUpdate>,
    c11_objects: BTreeMap<RuntimePassId, PreparedC11PassObjects>,
}

#[must_use = "the closed graph dispatch result must select exactly one renderer route"]
pub(crate) enum ExecutableGraphDispatchEligibility {
    ExactC08(C08PreparableGraph),
    ExactC09(C09PreparableGraph),
    ExactC12(C12PreparableGraph),
    FuturePasses,
}

impl ExecutableGraphDispatchEligibility {
    fn try_classify_c12(
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
        preparable: C12PreparableGraph,
    ) -> Result<Self> {
        if !preparable.proves_closed_backdrop_facts() {
            return Err(preparation_error(
                "C12 classification lost its closed pre-allocation facts",
            ));
        }
        let working_format = working_format.resolve(capabilities)?;
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            graph,
            working_format,
            output_format,
            capabilities,
        )?;
        match PrePreparationGraphClassification::classify(lowered) {
            PrePreparationGraphClassification::ExactC12(preparable)
                if preparable.proves_closed_backdrop_facts() =>
            {
                Ok(Self::ExactC12(preparable))
            }
            PrePreparationGraphClassification::ExactC08(_)
            | PrePreparationGraphClassification::ExactC09(_)
            | PrePreparationGraphClassification::ExactC10(_)
            | PrePreparationGraphClassification::ExactC11(_)
            | PrePreparationGraphClassification::ExactC12(_)
            | PrePreparationGraphClassification::FuturePasses
            | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                "checked C12 dispatch lowering changed its closed eligibility result",
            )),
        }
    }

    pub(crate) fn try_classify(
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
    ) -> Result<Self> {
        let classification = PrePreparationGraphClassification::classify(
            LoweredGraphPlan::try_lower_for_dispatch_classification(
                graph,
                WorkingFormat::HighPrecision,
                output_format,
            )?,
        );
        match classification {
            PrePreparationGraphClassification::ExactC12(preparable) => Self::try_classify_c12(
                graph,
                output_format,
                working_format,
                capabilities,
                preparable,
            ),
            PrePreparationGraphClassification::ExactC11(preparable) => {
                if !preparable.proves_closed_filter_facts() {
                    return Err(preparation_error(
                        "C11 classification lost its closed pre-allocation facts",
                    ));
                }
                Ok(Self::FuturePasses)
            }
            PrePreparationGraphClassification::ExactC10(preparable) => {
                if !preparable.proves_closed_color_facts() {
                    return Err(preparation_error(
                        "C10 classification lost its closed pre-allocation facts",
                    ));
                }
                Ok(Self::FuturePasses)
            }
            PrePreparationGraphClassification::ExactC09(_) => {
                let working_format = working_format.resolve(capabilities)?;
                let lowered = LoweredGraphPlan::try_lower_validated_graph(
                    graph,
                    working_format,
                    output_format,
                    capabilities,
                )?;
                match PrePreparationGraphClassification::classify(lowered) {
                    PrePreparationGraphClassification::ExactC09(closed) => {
                        C09PreparableGraph::try_from_closed(closed)
                            .map(Self::ExactC09)
                            .map_err(|_| {
                                preparation_error(
                                    "checked C09 dispatch lowering lost its C09-only facts",
                                )
                            })
                    }
                    PrePreparationGraphClassification::ExactC08(_)
                    | PrePreparationGraphClassification::ExactC10(_)
                    | PrePreparationGraphClassification::ExactC11(_)
                    | PrePreparationGraphClassification::ExactC12(_)
                    | PrePreparationGraphClassification::FuturePasses
                    | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                        "checked C09 dispatch lowering changed its closed eligibility result",
                    )),
                }
            }
            PrePreparationGraphClassification::FuturePasses => Ok(Self::FuturePasses),
            PrePreparationGraphClassification::Ineligible(ineligibility) => {
                Err(ineligibility.into_error())
            }
            PrePreparationGraphClassification::ExactC08(_) => {
                let working_format = working_format.resolve(capabilities)?;
                let lowered = LoweredGraphPlan::try_lower_validated_graph(
                    graph,
                    working_format,
                    output_format,
                    capabilities,
                )?;
                match PrePreparationGraphClassification::classify(lowered) {
                    PrePreparationGraphClassification::ExactC08(preparable) => {
                        Ok(Self::ExactC08(preparable))
                    }
                    PrePreparationGraphClassification::ExactC09(_)
                    | PrePreparationGraphClassification::ExactC10(_)
                    | PrePreparationGraphClassification::ExactC11(_)
                    | PrePreparationGraphClassification::ExactC12(_)
                    | PrePreparationGraphClassification::FuturePasses
                    | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                        "checked C08 dispatch lowering changed its closed eligibility result",
                    )),
                }
            }
        }
    }
}

pub(super) enum GraphPreparationSource {
    C08(C08PreparableGraph),
    C09(ClosedExecutableGraph),
    C10 {
        preparable: C10PreparableGraph,
        operation_limits: Option<ColorFilterOperationBufferLimits>,
    },
    C11(C11PreparableGraph),
    C12(C12PreparableGraph),
}

type GraphPreparationParts = (
    LoweredGraphPlan,
    Option<C08ExecutionFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
    Option<ClosedExecutableGraphFacts>,
);

impl GraphPreparationSource {
    const fn color_filter_operation_limits(&self) -> Option<ColorFilterOperationBufferLimits> {
        match self {
            Self::C10 {
                operation_limits, ..
            } => *operation_limits,
            Self::C08(_) | Self::C09(_) | Self::C11(_) | Self::C12(_) => None,
        }
    }

    fn into_parts(self) -> GraphPreparationParts {
        match self {
            Self::C08(preparable) => {
                let (lowered, execution) = preparable.into_parts();
                (lowered, Some(execution), None, None, None, None)
            }
            Self::C09(closed) => (closed.lowered, None, Some(closed.facts), None, None, None),
            Self::C10 { preparable, .. } => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, Some(closed.facts), None, None)
            }
            Self::C11(preparable) => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, None, Some(closed.facts), None)
            }
            Self::C12(preparable) => {
                let closed = preparable.into_closed();
                (closed.lowered, None, None, None, None, Some(closed.facts))
            }
        }
    }
}

pub(crate) struct PreparedGraph<'device> {
    pub(super) plan: RuntimeGraphPreparationPlan,
    pub(super) c08_execution: Option<C08ExecutionFacts>,
    pub(super) c09_execution: Option<ClosedExecutableGraphFacts>,
    pub(super) c10_execution: Option<ClosedExecutableGraphFacts>,
    pub(super) c11_execution: Option<ClosedExecutableGraphFacts>,
    pub(super) c12_execution: Option<ClosedExecutableGraphFacts>,
    pub(super) resource_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    pub(super) kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
    pub(super) color_filter_operation_bindings:
        BTreeMap<RuntimePassId, PreparedColorFilterOperationBinding>,
    pub(super) c11_pass_objects: BTreeMap<RuntimePassId, PreparedC11PassObjects>,
    pub(super) pass_cache_update: Option<ProvisionalDevicePassCacheUpdate>,
    pub(super) frame_scope: Option<FrameResourceScope>,
    pub(super) next_pass: usize,
    pub(super) c08_encoding_state: Option<C08CustomSpineEncodingState>,
    pub(super) c08_completed_session: Option<Arc<()>>,
    pub(super) device: &'device wgpu::Device,
    pub(super) queue: &'device wgpu::Queue,
    pub(super) vello_engine: Option<&'device VelloEngineState>,
    pub(super) resources: &'device ResourceManager,
    pub(super) pass_cache: &'device DevicePassCache,
    _ready_device: PhantomData<&'device ResourceManager>,
}

struct AcquiredGraphBindings {
    runtime_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    gaussian_kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
}

fn acquire_prepared_graph_resources(
    plan: &RuntimeGraphPreparationPlan,
    frame_scope: &mut FrameResourceScope,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    capabilities: &DeviceCapabilities,
) -> Result<AcquiredGraphBindings> {
    let mut runtime_bindings = BTreeMap::new();
    for request in &plan.resources {
        let lease = request
            .allocation
            .acquire(frame_scope, device, queue, capabilities)?;
        if runtime_bindings
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
    let mut gaussian_kernel_bindings = BTreeMap::new();
    for request in &plan.kernels {
        let lease = frame_scope.acquire_gaussian_kernel_buffer(device, &request.plan)?;
        if gaussian_kernel_bindings
            .insert(request.key, PreparedKernelBinding { lease: Some(lease) })
            .is_some()
        {
            return Err(preparation_error(
                "one Gaussian kernel acquired more than one concrete binding",
            ));
        }
    }
    Ok(AcquiredGraphBindings {
        runtime_bindings,
        gaussian_kernel_bindings,
    })
}

fn create_color_filter_operation_bindings(
    plan: &RuntimeGraphPreparationPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<BTreeMap<RuntimePassId, PreparedColorFilterOperationBinding>> {
    let mut bindings = BTreeMap::new();
    for request in &plan.passes {
        let Some(bytes) = request.color_filter_operations.as_ref() else {
            continue;
        };
        if !matches!(request.runtime.kind, RuntimePassKind::ColorFilter(Some(_)))
            || bytes.as_bytes().is_empty()
        {
            return Err(preparation_error(
                "prepared color-filter bytes have no exact runtime pass",
            ));
        }
        let size = u64::try_from(bytes.as_bytes().len()).map_err(|_| {
            preparation_error("prepared color-filter buffer length does not fit u64")
        })?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C10 ordered color-filter operations"),
            size,
            usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytes.as_bytes());
        if bindings
            .insert(
                request.runtime.id,
                PreparedColorFilterOperationBinding {
                    bytes: bytes.clone(),
                    buffer: Some(buffer),
                },
            )
            .is_some()
        {
            return Err(preparation_error(
                "one color-filter pass acquired more than one operation buffer",
            ));
        }
    }
    Ok(bindings)
}

fn realize_prepared_graph_passes(
    plan: &RuntimeGraphPreparationPlan,
    device: &wgpu::Device,
    pass_cache: &DevicePassCache,
    enabled: bool,
) -> Result<PreparedPassRealization> {
    if !enabled {
        return Ok(PreparedPassRealization {
            update: None,
            c11_objects: BTreeMap::new(),
        });
    }
    let mut update = pass_cache.provisional_update();
    let mut realized_pass = false;
    let mut c11_objects = BTreeMap::new();
    for request in &plan.passes {
        let Some(keys) = request.cache_keys.as_ref() else {
            continue;
        };
        let objects = realize_prepared_graph_pass(&mut update, device, pass_cache, request, keys)?;
        if let Some(objects) = objects
            && c11_objects.insert(request.runtime.id, objects).is_some()
        {
            return Err(preparation_error(
                "one C11 pass retained more than one prepared object set",
            ));
        }
        realized_pass = true;
    }
    Ok(PreparedPassRealization {
        update: realized_pass.then_some(update),
        c11_objects,
    })
}

fn realize_prepared_graph_pass(
    update: &mut ProvisionalDevicePassCacheUpdate,
    device: &wgpu::Device,
    pass_cache: &DevicePassCache,
    request: &RuntimePassPreparationRequest,
    keys: &RuntimePassCacheKeys,
) -> Result<Option<PreparedC11PassObjects>> {
    match &request.runtime.kind {
        RuntimePassKind::CopyBackdrop => {
            let objects = realize_copy_backdrop_for_preparation(update, device, pass_cache, keys)?;
            Ok(Some(PreparedC11PassObjects::CopyBackdrop {
                parent_sampler: objects.parent_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::ColorFilter(Some(_)) => {
            update
                .realize_color_filter_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_)) => {
            let objects = update.realize_blur_pass(
                device,
                pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            Ok(Some(PreparedC11PassObjects::Blur {
                source_sampler: objects.source_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::DropShadowColorize(Some(_)) => {
            let objects = update.realize_drop_shadow_colorize_pass(
                device,
                pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )?;
            objects.require_encoding_ready()?;
            Ok(Some(PreparedC11PassObjects::DropShadowColorize {
                source_sampler: objects.source_sampler().clone(),
                layout: objects.bind_group_layout().clone(),
                pipeline: objects.render_pipeline().clone(),
            }))
        }
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { .. },
            ..
        })) => {
            update
                .realize_composite_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::CanonicalizeCapture
        | RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver | RuntimeCompositeKind::DropShadow,
            ..
        }))
        | RuntimePassKind::Present => {
            update
                .realize_c08_pass(
                    device,
                    pass_cache,
                    keys.samplers(),
                    keys.layout(),
                    keys.shader(),
                    keys.pipeline(),
                )?
                .require_encoding_ready()?;
            Ok(None)
        }
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => Err(preparation_error(
            "checked pass realization reached an unsupported graph pass",
        )),
    }
}

fn realize_copy_backdrop_for_preparation<'a>(
    update: &'a mut ProvisionalDevicePassCacheUpdate,
    device: &wgpu::Device,
    pass_cache: &'a DevicePassCache,
    keys: &RuntimePassCacheKeys,
) -> Result<ProvisionalCopyBackdropPassObjects<'a>> {
    let objects = update.realize_copy_backdrop_pass(
        device,
        pass_cache,
        keys.samplers(),
        keys.layout(),
        keys.shader(),
        keys.pipeline(),
    )?;
    objects.require_encoding_ready()?;
    Ok(objects)
}

impl<'device> PreparedGraph<'device> {
    pub(crate) fn try_prepare_c08(
        preparable: C08PreparableGraph,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache: &'device DevicePassCache,
    ) -> Result<Self> {
        let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
        Self::try_prepare_c08_with_working_format(
            preparable,
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            (pass_cache, false),
        )
    }

    pub(crate) fn try_prepare_c08_with_working_format(
        preparable: C08PreparableGraph,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C08(preparable),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c08_execution_facts().is_none() {
            return Err(preparation_error(
                "C08 preparation lost its validated execution facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare_c09(
        preparable: C09PreparableGraph,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let selected_working_format = preparable.working_format();
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C09(preparable.into_closed()),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c09_execution.is_none() {
            return Err(preparation_error(
                "C09 preparation lost its validated closed execution facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare_c12(
        preparable: C12PreparableGraph,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let prepared = Self::try_prepare_inner(
            GraphPreparationSource::C12(preparable),
            selected_working_format,
            capabilities,
            device,
            queue,
            resources,
            pass_cache_phase,
        )?;
        if prepared.c12_execution.is_none() {
            return Err(preparation_error(
                "C12 preparation lost its validated closed backdrop facts",
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn try_prepare(
        lowered: LoweredGraphPlan,
        policy: EffectQualityPolicy,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        match PrePreparationGraphClassification::classify(lowered) {
            PrePreparationGraphClassification::ExactC08(preparable) if pass_cache_phase.1 => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C08(preparable),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c08_execution_facts().is_none() {
                    return Err(preparation_error(
                        "C08 preparation lost its validated execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC08(preparable) => Self::try_prepare_c08(
                preparable,
                policy,
                capabilities,
                device,
                queue,
                resources,
                pass_cache_phase.0,
            ),
            PrePreparationGraphClassification::ExactC09(closed) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                Self::try_prepare_inner(
                    GraphPreparationSource::C09(closed),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )
            }
            PrePreparationGraphClassification::ExactC10(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C10 {
                        preparable,
                        operation_limits: None,
                    },
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c10_execution.is_none() {
                    return Err(preparation_error(
                        "C10 preparation lost its validated closed execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC11(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                let prepared = Self::try_prepare_inner(
                    GraphPreparationSource::C11(preparable),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )?;
                if prepared.c11_execution.is_none() {
                    return Err(preparation_error(
                        "C11 preparation lost its validated closed execution facts",
                    ));
                }
                Ok(prepared)
            }
            PrePreparationGraphClassification::ExactC12(preparable) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                Self::try_prepare_c12(
                    preparable,
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    pass_cache_phase,
                )
            }
            PrePreparationGraphClassification::FuturePasses => Err(preparation_error(
                "a future GPU pass cannot enter C09 resource preparation",
            )),
            PrePreparationGraphClassification::Ineligible(ineligibility) => {
                Err(ineligibility.into_error())
            }
        }
    }

    pub(super) fn try_prepare_inner(
        source: GraphPreparationSource,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let (pass_cache, realize_checked_passes) = pass_cache_phase;
        let color_filter_operation_limits = source.color_filter_operation_limits();
        let (lowered, c08_execution, c09_execution, c10_execution, c11_execution, c12_execution) =
            source.into_parts();
        let plan = match color_filter_operation_limits {
            Some(limits) => RuntimeGraphPreparationPlan::try_derive_with_color_filter_limits(
                lowered,
                selected_working_format,
                capabilities,
                device,
                limits,
            )?,
            None => RuntimeGraphPreparationPlan::try_derive(
                lowered,
                selected_working_format,
                capabilities,
                device,
            )?,
        };
        resources.preflight_graph_acquisitions(&plan.allocation_preflights)?;

        let mut frame_scope = resources.begin_frame()?;
        frame_scope.abort_provisional_on_drop();
        if c08_execution.is_some()
            || c09_execution.is_some()
            || c10_execution.is_some()
            || c11_execution.is_some()
            || c12_execution.is_some()
        {
            frame_scope.discard_on_drop();
        }
        let acquired_resources =
            acquire_prepared_graph_resources(&plan, &mut frame_scope, device, queue, capabilities)?;
        let color_filter_operation_bindings =
            create_color_filter_operation_bindings(&plan, device, queue)?;
        let pass_realization =
            realize_prepared_graph_passes(&plan, device, pass_cache, realize_checked_passes)?;
        let c08_encoding_state = (c08_execution.is_some()
            || c09_execution.is_some()
            || c10_execution.is_some()
            || c11_execution.is_some()
            || c12_execution.is_some())
        .then_some(C08CustomSpineEncodingState::Ready);

        Ok(Self {
            plan,
            c08_execution,
            c09_execution,
            c10_execution,
            c11_execution,
            c12_execution,
            resource_bindings: acquired_resources.runtime_bindings,
            kernel_bindings: acquired_resources.gaussian_kernel_bindings,
            color_filter_operation_bindings,
            c11_pass_objects: pass_realization.c11_objects,
            pass_cache_update: pass_realization.update,
            frame_scope: Some(frame_scope),
            next_pass: 0,
            c08_encoding_state,
            c08_completed_session: None,
            device,
            queue,
            vello_engine: None,
            resources,
            pass_cache,
            _ready_device: PhantomData,
        })
    }

    pub(crate) const fn c08_execution_facts(&self) -> Option<&C08ExecutionFacts> {
        self.c08_execution.as_ref()
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

    pub(crate) const fn working_format(&self) -> WorkingFormat {
        self.plan.working_format
    }

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

    pub(crate) fn output_extent(&self) -> Result<PhysicalSize> {
        self.resource_request(self.plan.root_working_image)
            .map(|resource| resource.spatial.device_extent)
    }
}

impl PreparedGraph<'_> {
    pub(super) fn resource_request(
        &self,
        resource: RuntimeResourceId,
    ) -> Result<&RuntimeResourceRequest> {
        self.plan
            .resources
            .iter()
            .find(|request| request.runtime.id == resource)
            .map(|request| &request.runtime)
            .ok_or_else(|| preparation_error("the prepared C08 resource request is missing"))
    }
}

impl PreparedGraph<'_> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 consumes prepared pass requests through this narrow iterator"
        )
    )]
    pub(crate) fn current_pass(&self) -> Option<super::PreparedPassView<'_>> {
        self.plan
            .passes
            .get(self.next_pass)
            .map(|request| super::PreparedPassView { request })
    }

    pub(super) fn require_current_pass(
        &self,
        pass: RuntimePassId,
    ) -> Result<&RuntimePassPreparationRequest> {
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

    pub(crate) fn gaussian_kernel_binding_for_pass(
        &self,
        pass: RuntimePassId,
    ) -> Result<Option<super::PreparedGaussianKernelBinding<'_>>> {
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
        Ok(Some(super::PreparedGaussianKernelBinding {
            key: kernel,
            allocation_resource: lease.resource_identity(),
            buffer: frame_scope.gaussian_kernel_buffer(lease)?,
        }))
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
    pub(super) request: &'prepared RuntimePassPreparationRequest,
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

    pub(crate) const fn blur_edge_parameters(&self) -> Option<&BlurEdgeParameterBytes> {
        self.request.blur_edge_parameters.as_ref()
    }

    pub(crate) const fn composite_parameters(&self) -> Option<&CompositeParameterBytes> {
        self.request.composite_parameters.as_ref()
    }

    pub(crate) const fn drop_shadow_parameters(&self) -> Option<&DropShadowParameterBytes> {
        self.request.drop_shadow_parameters.as_ref()
    }

    pub(crate) const fn cache_keys(&self) -> Option<&RuntimePassCacheKeys> {
        self.request.cache_keys.as_ref()
    }
}

pub(crate) struct PreparedTextureBinding<'prepared> {
    runtime_resource: RuntimeResourceId,
    allocation_resource: ResourceIdentity,
    texture: &'prepared wgpu::Texture,
    view: &'prepared wgpu::TextureView,
}

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
