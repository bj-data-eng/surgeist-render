use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Arc,
};

use super::{
    BackendErrorCode, Color, Error, Format, PhysicalSize, Point, Rect, Result, Transform,
    backend::DeviceCapabilities,
    command::{RenderClip, RenderCommands},
    encode::{encode_vello_clip_coverage_scene, encode_vello_scene_with_initial_transform},
    filter::{CSS_FILTER_KERNEL_SUPPORT_STANDARD_DEVIATIONS, ColorClampBoundary},
    frame::{
        GpuRenderGraph, GraphLoweringBlur, GraphLoweringBlurInput, GraphLoweringClipCoverage,
        GraphLoweringColorFilter, GraphLoweringComposite, GraphLoweringCompositeKind,
        GraphLoweringDropShadow, GraphLoweringEdgePolicy, GraphLoweringGeneration,
        GraphLoweringImportView, GraphLoweringInitialization, GraphLoweringPassId,
        GraphLoweringPassKind, GraphLoweringPassResult, GraphLoweringReadBinding,
        GraphLoweringReadRole, GraphLoweringResourceId, GraphLoweringResourceProducer,
        GraphLoweringResourceRole, GraphLoweringSamplingEdge, GraphLoweringSamplingFilter,
        GraphLoweringSpatialDescriptor, GraphLoweringVelloCapture, GraphLoweringVelloSpan,
        GraphLoweringVelloSpanScope,
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
        BindGroupLayoutKey, CompositeParameterBytes, DevicePassCache, PassSpatialUniformBytes,
        ProvisionalC08PassObjects, ProvisionalDevicePassCacheUpdate, RenderPipelineKey, SamplerKey,
        ShaderBindingRoleKey, ShaderCompositeKey, ShaderCompositePathKey, ShaderDataBindingKey,
        ShaderMaskQualityKey, ShaderMaskSamplingKey, ShaderModuleKey, ShaderProgramKey,
        ShaderSamplingEdgeKey, ShaderSamplingFilterKey, ShaderTextureFormatKey,
    },
    style::ColorFilterOp,
    texture::EffectTextureDescriptor,
    vello_engine::{
        ActiveVelloEncodingScope, EncodedVelloCaptureProof, PendingVelloResourceCommit,
        RasterParameters, TransactionEncodingState, TransactionTargetIntent, VelloEngineState,
        VelloResourceLeaseAggregate,
    },
};

#[cfg(test)]
use super::texture::EffectTextureRole;

#[cfg(test)]
use super::resource::ResourceAccountingFault;

#[cfg(test)]
use super::frame::{FrameContext, FramePlan};

#[cfg(test)]
use super::vello_engine::{
    prepared_vello_pass_observation_for_test, scene::VelloPathDrawObservationForTest,
};

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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct C09ExecutableGraphObservationForTest {
    pub(crate) accepts_spine_and_layer_composition_for_all_formats: bool,
    pub(crate) layer_composition_reads_are_exact: bool,
    pub(crate) rejects_c10_plus_passes_and_payloads: bool,
    pub(crate) rejects_missing_payloads: bool,
    pub(crate) rejects_malformed_graph_facts: bool,
    pub(crate) rejects_unsupported_output_binding: bool,
    pub(crate) preserves_transitional_c09_dispatch: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompositionOuterOperationObservationForTest {
    SourceMapping,
    ClipCoverage,
    AlphaMask,
    Opacity,
    Blend,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompositionReadObservationForTest {
    Parent,
    Source,
    ClipCoverage,
    AlphaMask,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClipCoverageElementObservationForTest {
    pub(crate) clip: RenderClip,
    pub(crate) transform: Transform,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClipCoverageCaptureObservationForTest {
    pub(crate) elements: Vec<ClipCoverageElementObservationForTest>,
    pub(crate) antialiasing: Antialiasing,
    pub(crate) device_origin: (i32, i32),
    pub(crate) target_extent: PhysicalSize,
    pub(crate) texel_origin: Point,
    pub(crate) raster_scale: f64,
    pub(crate) first_texel_center: Point,
    pub(crate) initial_transform: Transform,
    pub(crate) emitted_draws: Vec<VelloPathDrawObservationForTest>,
    pub(crate) uses_coverage_resource_role: bool,
    pub(crate) uses_rgba8_target: bool,
    pub(crate) uses_transparent_base: bool,
    pub(crate) raster_antialiasing: Antialiasing,
    pub(crate) raster_target_extent: PhysicalSize,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct GraphClipCoverageObservationForTest {
    pub(crate) captures: Vec<ClipCoverageCaptureObservationForTest>,
    pub(crate) all_vello_capture_count: usize,
    pub(crate) composite_coverage_read_count: usize,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerCompositionObservationForTest {
    pub(crate) transform: Transform,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) has_own_clip: bool,
    pub(crate) inherited_outer_clip_count: usize,
    pub(crate) inherited_outer_clip_transforms: Vec<Transform>,
    pub(crate) reads: Vec<CompositionReadObservationForTest>,
    pub(crate) outer_operations: Vec<CompositionOuterOperationObservationForTest>,
    pub(crate) source_captured_before_outer_semantics: bool,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CompositionGraphObservationForTest {
    pub(crate) layers_inner_to_outer: Vec<LayerCompositionObservationForTest>,
    pub(crate) mask_identity_is_preserved: bool,
    pub(crate) root_surface_base_clears: usize,
    pub(crate) root_surface_base_color: Option<Color>,
    pub(crate) transparent_isolation_clears: usize,
    pub(crate) nontransparent_isolation_clears: usize,
}

#[cfg(test)]
pub(crate) fn c09_executable_graph_observation_for_test(
    c08_commands: RenderCommands,
    c09_commands: RenderCommands,
    c10_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C09ExecutableGraphObservationForTest {
    c09_executable_graph_observation(
        c08_commands,
        c09_commands,
        c10_commands,
        context,
        capabilities,
    )
    .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn composition_graph_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> CompositionGraphObservationForTest {
    composition_graph_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn graph_clip_coverage_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> GraphClipCoverageObservationForTest {
    graph_clip_coverage_observation(commands, context, capabilities).unwrap_or_default()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C08PassLayoutObservationForTest {
    pub(crate) canonicalize_binds_capture_and_spatial_only: bool,
    pub(crate) span_source_over_binds_source_and_spatial_only: bool,
    pub(crate) present_binds_final_image_and_spatial_only: bool,
    pub(crate) copy_only_parent_is_not_sampled: bool,
    pub(crate) dummy_parameters_are_not_bound: bool,
    pub(crate) c09_typed_vocabulary_is_preserved: bool,
    pub(crate) output_specialization_is_exact: bool,
}

#[cfg(test)]
pub(crate) struct C08PassCacheRequestsForTest {
    passes: Vec<RuntimePassCacheKeys>,
}

#[cfg(test)]
impl C08PassCacheRequestsForTest {
    pub(crate) fn passes(&self) -> &[RuntimePassCacheKeys] {
        &self.passes
    }
}

#[cfg(test)]
pub(crate) struct C09CompositeCacheRequestsForTest {
    passes: Vec<RuntimePassCacheKeys>,
}

#[cfg(test)]
impl C09CompositeCacheRequestsForTest {
    pub(crate) fn passes(&self) -> &[RuntimePassCacheKeys] {
        &self.passes
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn composite_pass(
        &self,
        path: ShaderCompositePathKey,
        has_clip_coverage: bool,
        has_alpha_mask: bool,
    ) -> Option<&RuntimePassCacheKeys> {
        self.passes.iter().find(|keys| {
            super::shader::c09_composite_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
            .is_some_and(|facts| {
                facts.path == path
                    && facts.has_clip_coverage == has_clip_coverage
                    && facts.has_alpha_mask == has_alpha_mask
            })
        })
    }
}

#[cfg(test)]
pub(crate) fn c09_composite_cache_requests_for_test(
    command_sets: &[RenderCommands],
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> Result<C09CompositeCacheRequestsForTest> {
    let mut passes = Vec::with_capacity(command_sets.len());
    for commands in command_sets {
        let FramePlan::GpuGraph(graph) = commands.clone().plan_for(context)? else {
            return Err(lowering_error(
                "the C09 composite cache fixture did not produce a GPU graph",
            ));
        };
        let lowered = LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            Format::Rgba8,
            &capabilities,
        )?;
        let resource_formats = lowered
            .resources
            .iter()
            .map(|resource| (resource.id, resource.format))
            .collect::<BTreeMap<_, _>>();
        let layer_passes = lowered.passes.iter().filter_map(|pass| {
            matches!(
                &pass.kind,
                RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::Layer { .. },
                    ..
                }))
            )
            .then_some(pass.cache_keys.as_ref())
            .flatten()
        });
        let mut found_layer = false;
        for keys in layer_passes {
            found_layer = true;
            if !passes.contains(keys) {
                passes.push(keys.clone());
            }
        }
        if !found_layer {
            return Err(lowering_error(
                "the C09 composite cache fixture has no layer-composite keys",
            ));
        }
        for pass in &lowered.passes {
            let RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) = &pass.kind
            else {
                continue;
            };
            if parameters.blend() == BlendMode::Normal {
                continue;
            }
            let mut normal_kind = pass.kind.clone();
            let RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) = &mut normal_kind
            else {
                unreachable!("the cloned C09 layer-composite kind must remain a layer")
            };
            parameters.blend = BlendMode::Normal;
            let Some(keys) = runtime_pass_cache_keys(
                &normal_kind,
                &pass.reads,
                pass.result,
                working_format,
                Format::Rgba8,
                &resource_formats,
            )?
            else {
                return Err(lowering_error(
                    "the C09 normal-path cache fixture lost its exact keys",
                ));
            };
            if !passes.contains(&keys) {
                passes.push(keys);
            }
        }
    }
    if passes.is_empty() {
        return Err(lowering_error(
            "the C09 composite cache fixture contains no composite variants",
        ));
    }
    Ok(C09CompositeCacheRequestsForTest { passes })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct C09CompositeLayoutObservationForTest {
    pub(crate) realizes_all_eight_entry_interfaces: bool,
    pub(crate) normal_omits_parent: bool,
    pub(crate) destination_binds_parent: bool,
    pub(crate) optional_clip_is_exact: bool,
    pub(crate) optional_mask_is_exact: bool,
    pub(crate) binds_only_one_source_sampler: bool,
    pub(crate) binds_only_exact_uniforms: bool,
}

#[cfg(test)]
pub(crate) fn c09_composite_layout_observation_for_test(
    requests: &C09CompositeCacheRequestsForTest,
) -> C09CompositeLayoutObservationForTest {
    use super::shader::c09_composite_pass_key_facts_for_test;

    let facts = requests
        .passes()
        .iter()
        .filter_map(|keys| {
            c09_composite_pass_key_facts_for_test(
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
        })
        .collect::<Vec<_>>();
    let realizes_all_eight_entry_interfaces = facts.len() == requests.passes().len()
        && facts.len() == 8
        && [
            ShaderCompositePathKey::Normal,
            ShaderCompositePathKey::DestinationSampling,
        ]
        .into_iter()
        .all(|path| {
            [(false, false), (true, false), (false, true), (true, true)]
                .into_iter()
                .all(|(has_clip, has_mask)| {
                    facts
                        .iter()
                        .filter(|facts| {
                            facts.path == path
                                && facts.has_clip_coverage == has_clip
                                && facts.has_alpha_mask == has_mask
                        })
                        .count()
                        == 1
                })
        });
    let normal_omits_parent = facts.iter().all(|facts| {
        facts.path != ShaderCompositePathKey::Normal
            || (!facts
                .sampled_roles
                .contains(&ShaderBindingRoleKey::CompositeParent)
                && facts.uses_fixed_source_over_blend
                && !facts.uses_replace_blend)
    });
    let destination_binds_parent = facts.iter().all(|facts| {
        facts.path != ShaderCompositePathKey::DestinationSampling
            || (facts
                .sampled_roles
                .iter()
                .filter(|role| **role == ShaderBindingRoleKey::CompositeParent)
                .count()
                == 1
                && facts.uses_replace_blend
                && !facts.uses_fixed_source_over_blend)
    });
    let optional_clip_is_exact = facts.iter().all(|facts| {
        facts
            .sampled_roles
            .iter()
            .filter(|role| **role == ShaderBindingRoleKey::ClipCoverage)
            .count()
            == usize::from(facts.has_clip_coverage)
    });
    let optional_mask_is_exact = facts.iter().all(|facts| {
        facts
            .sampled_roles
            .iter()
            .filter(|role| **role == ShaderBindingRoleKey::AlphaMask)
            .count()
            == usize::from(facts.has_alpha_mask)
    });
    let binds_only_one_source_sampler = facts.iter().all(|facts| {
        facts.has_only_source_sampler
            && facts
                .sampled_roles
                .iter()
                .filter(|role| **role == ShaderBindingRoleKey::CompositeSource)
                .count()
                == 1
    });
    let binds_only_exact_uniforms = facts.iter().all(|facts| {
        facts.has_exact_uniforms
            && facts.working_format == facts.target_format
            && facts.sampled_roles.len()
                == 1 + usize::from(facts.path == ShaderCompositePathKey::DestinationSampling)
                    + usize::from(facts.has_clip_coverage)
                    + usize::from(facts.has_alpha_mask)
    });
    C09CompositeLayoutObservationForTest {
        realizes_all_eight_entry_interfaces,
        normal_omits_parent,
        destination_binds_parent,
        optional_clip_is_exact,
        optional_mask_is_exact,
        binds_only_one_source_sampler,
        binds_only_exact_uniforms,
    }
}

#[cfg(test)]
pub(crate) fn c08_pass_cache_requests_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
    output_format: Format,
) -> Result<C08PassCacheRequestsForTest> {
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        output_format,
        &capabilities,
    )?;
    let preparable = C08PreparableGraph::try_from_lowered(lowered).map_err(|_| {
        lowering_error("the checked C08 cache fixture did not retain exact executable keys")
    })?;
    let passes = preparable
        .lowered
        .passes
        .iter()
        .filter_map(|pass| pass.cache_keys.clone())
        .collect::<Vec<_>>();
    if passes.is_empty() {
        return Err(lowering_error(
            "the checked C08 cache fixture contains no custom pass keys",
        ));
    }
    Ok(C08PassCacheRequestsForTest { passes })
}

#[cfg(test)]
pub(crate) fn c08_pass_layout_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> C08PassLayoutObservationForTest {
    use super::shader::{C08ProgramForTest, c08_pass_key_facts_for_test};

    let mut canonicalize_binds_capture_and_spatial_only = true;
    let mut span_source_over_binds_source_and_spatial_only = true;
    let mut present_binds_final_image_and_spatial_only = true;
    let mut copy_only_parent_is_not_sampled = true;
    let mut dummy_parameters_are_not_bound = true;
    let mut output_specializations = Vec::new();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let Ok(requests) = c08_pass_cache_requests_for_test(
                commands.clone(),
                context,
                capabilities,
                working_format,
                output_format,
            ) else {
                return C08PassLayoutObservationForTest::default();
            };
            let facts = requests
                .passes()
                .iter()
                .filter_map(|keys| {
                    c08_pass_key_facts_for_test(
                        keys.samplers(),
                        keys.layout(),
                        keys.shader(),
                        keys.pipeline(),
                    )
                })
                .collect::<Vec<_>>();
            if facts.len() != requests.passes().len() {
                return C08PassLayoutObservationForTest::default();
            }
            let canonicalize = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::CanonicalizeCapture);
            let source_over = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::SpanSourceOver);
            let present = facts
                .iter()
                .find(|facts| facts.program == C08ProgramForTest::Present);
            canonicalize_binds_capture_and_spatial_only &= canonicalize.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::CaptureSource
                    && facts.source_format == ShaderTextureFormatKey::VelloCaptureRgba8Unorm
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format.is_none()
                    && facts.target_format == ShaderTextureFormatKey::working(working_format)
                    && facts.has_only_spatial_uniform
            });
            span_source_over_binds_source_and_spatial_only &= source_over.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::CompositeSource
                    && facts.source_format == ShaderTextureFormatKey::working(working_format)
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format.is_none()
                    && facts.target_format == ShaderTextureFormatKey::working(working_format)
                    && facts.has_only_spatial_uniform
                    && facts.has_fixed_source_over_blend
            });
            copy_only_parent_is_not_sampled &= facts
                .iter()
                .all(|facts| facts.source_role != ShaderBindingRoleKey::CompositeParent);
            dummy_parameters_are_not_bound &=
                facts.iter().all(|facts| facts.has_only_spatial_uniform);
            present_binds_final_image_and_spatial_only &= present.is_some_and(|facts| {
                facts.source_role == ShaderBindingRoleKey::FinalWorkingImage
                    && facts.source_format == ShaderTextureFormatKey::working(working_format)
                    && facts.working_format == ShaderTextureFormatKey::working(working_format)
                    && facts.output_format == Some(ShaderTextureFormatKey::output(output_format))
                    && facts.target_format == ShaderTextureFormatKey::output(output_format)
                    && facts.has_only_spatial_uniform
            });
            if let Some(facts) = present {
                output_specializations.push((facts.working_format, facts.output_format));
            }
        }
    }
    let output_specialization_is_exact = output_specializations.len() == 4
        && output_specializations.iter().all(|specialization| {
            output_specializations
                .iter()
                .filter(|candidate| *candidate == specialization)
                .count()
                == 1
        });
    let c09_typed_vocabulary_is_preserved = matches!(
        ShaderBindingRoleKey::CompositeParent,
        ShaderBindingRoleKey::CompositeParent
    ) && matches!(
        ShaderDataBindingKey::CompositeParameters,
        ShaderDataBindingKey::CompositeParameters
    ) && matches!(
        ShaderDataBindingKey::PresentParameters,
        ShaderDataBindingKey::PresentParameters
    );

    C08PassLayoutObservationForTest {
        canonicalize_binds_capture_and_spatial_only,
        span_source_over_binds_source_and_spatial_only,
        present_binds_final_image_and_spatial_only,
        copy_only_parent_is_not_sampled,
        dummy_parameters_are_not_bound,
        c09_typed_vocabulary_is_preserved,
        output_specialization_is_exact,
    }
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
pub(crate) fn c08_zero_capture_spine_lowered_for_test(
    commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    policy: EffectQualityPolicy,
) -> Result<LoweredGraphPlan> {
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let working_format = capabilities.resolve_effect_working_format(policy)?;
    let mut lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let clear = lowered
        .passes
        .first()
        .cloned()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no root clear"))?;
    let mut present =
        lowered.passes.last().cloned().ok_or_else(|| {
            lowering_error("the C08 zero-capture fixture has no terminal present")
        })?;
    let mut root = lowered
        .resources
        .iter()
        .find(|resource| resource.id == lowered.root_working_image)
        .cloned()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no root resource"))?;

    root.expected_reads = 1;
    root.last_use = present.id;
    present.dependencies = vec![clear.id];
    let final_read = present
        .reads
        .first_mut()
        .ok_or_else(|| lowering_error("the C08 zero-capture fixture has no final read"))?;
    final_read.resource = root.id;
    present.releases = vec![root.id];
    lowered.resources = vec![root];
    lowered.passes = vec![clear, present];

    Ok(lowered)
}

#[cfg(test)]
pub(crate) fn c08_two_capture_spine_lowered_for_test(
    commands: RenderCommands,
    donor_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
    policy: EffectQualityPolicy,
) -> Result<LoweredGraphPlan> {
    let working_format = capabilities.resolve_effect_working_format(policy)?;
    let graph = super::frame::forced_c08_graph_for_test(commands, context)?;
    let mut lowered = LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;
    let FramePlan::GpuGraph(donor_graph) = donor_commands.plan_for(context)? else {
        return Err(lowering_error(
            "the two-capture C08 fixture requires a graph-shaped identity donor",
        ));
    };
    let donor = LoweredGraphPlan::try_lower_validated_graph(
        &donor_graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )?;

    let existing_passes = lowered
        .passes
        .iter()
        .map(|pass| pass.id)
        .collect::<BTreeSet<_>>();
    let mut donor_passes = donor
        .passes
        .iter()
        .map(|pass| pass.id)
        .filter(|pass| !existing_passes.contains(pass));
    let second_capture_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare capture pass identity")
    })?;
    let second_canonicalize_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare canonical pass identity")
    })?;
    let second_composite_pass = donor_passes.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare composite pass identity")
    })?;

    let existing_resources = lowered
        .resources
        .iter()
        .map(|resource| resource.id)
        .collect::<BTreeSet<_>>();
    let mut donor_resources = donor
        .resources
        .iter()
        .map(|resource| resource.id)
        .filter(|resource| !existing_resources.contains(resource));
    let second_capture_target = donor_resources.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare capture resource identity")
    })?;
    let second_canonical_target = donor_resources.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare canonical resource identity")
    })?;
    let second_composite_target = donor_resources.next().ok_or_else(|| {
        lowering_error("the two-capture C08 fixture has no spare composite resource identity")
    })?;

    if lowered.passes.len() != 5 || lowered.resources.len() != 4 {
        return Err(lowering_error(
            "the two-capture C08 fixture source is not the exact one-capture spine",
        ));
    }
    let first_capture = lowered.passes[1].clone();
    let first_canonicalize = lowered.passes[2].clone();
    let first_composite = lowered.passes[3].clone();
    let mut present = lowered
        .passes
        .pop()
        .ok_or_else(|| lowering_error("the two-capture C08 fixture has no terminal present"))?;
    let RuntimeResultBinding::Resource(first_capture_target) = first_capture.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no capture target",
        ));
    };
    let RuntimeResultBinding::Resource(first_canonical_target) = first_canonicalize.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no canonical target",
        ));
    };
    let RuntimeResultBinding::Resource(first_composite_target) = first_composite.result else {
        return Err(lowering_error(
            "the two-capture C08 fixture source has no composite target",
        ));
    };

    let mut capture_resource = lowered
        .resources
        .iter()
        .find(|resource| resource.id == first_capture_target)
        .cloned()
        .ok_or_else(|| lowering_error("the two-capture C08 fixture lost its capture resource"))?;
    capture_resource.id = second_capture_target;
    capture_resource.producer = RuntimeResourceProducer::Pass(second_capture_pass);
    capture_resource.last_use = second_canonicalize_pass;

    let mut canonical_resource = lowered
        .resources
        .iter()
        .find(|resource| resource.id == first_canonical_target)
        .cloned()
        .ok_or_else(|| lowering_error("the two-capture C08 fixture lost its canonical resource"))?;
    canonical_resource.id = second_canonical_target;
    canonical_resource.producer = RuntimeResourceProducer::Pass(second_canonicalize_pass);
    canonical_resource.last_use = second_composite_pass;

    let mut composite_resource = lowered
        .resources
        .iter()
        .find(|resource| resource.id == first_composite_target)
        .cloned()
        .ok_or_else(|| lowering_error("the two-capture C08 fixture lost its composite resource"))?;
    composite_resource.id = second_composite_target;
    composite_resource.producer = RuntimeResourceProducer::Pass(second_composite_pass);
    composite_resource.last_use = present.id;
    lowered
        .resources
        .iter_mut()
        .find(|resource| resource.id == first_composite_target)
        .ok_or_else(|| lowering_error("the two-capture C08 fixture lost its first result"))?
        .last_use = second_composite_pass;

    let mut second_capture = first_capture;
    second_capture.id = second_capture_pass;
    second_capture.result = RuntimeResultBinding::Resource(second_capture_target);

    let mut second_canonicalize = first_canonicalize;
    second_canonicalize.id = second_canonicalize_pass;
    second_canonicalize.dependencies = vec![second_capture_pass];
    second_canonicalize.reads[0].resource = second_capture_target;
    second_canonicalize.result = RuntimeResultBinding::Resource(second_canonical_target);
    second_canonicalize.releases = vec![second_capture_target];

    let mut second_composite = first_composite;
    second_composite.id = second_composite_pass;
    second_composite.dependencies = vec![lowered.passes[3].id, second_canonicalize_pass];
    second_composite.reads[0].resource = first_composite_target;
    second_composite.reads[1].resource = second_canonical_target;
    second_composite.result = RuntimeResultBinding::Resource(second_composite_target);
    second_composite.releases = vec![first_composite_target, second_canonical_target];

    present.dependencies = vec![second_composite_pass];
    present.reads[0].resource = second_composite_target;
    present.releases = vec![second_composite_target];
    lowered
        .resources
        .extend([capture_resource, canonical_resource, composite_resource]);
    lowered.passes.extend([
        second_capture,
        second_canonicalize,
        second_composite,
        present,
    ]);

    if lowered.c08_execution_facts().is_none() {
        return Err(lowering_error(
            "the two-capture C08 fixture did not preserve the validated executable subset",
        ));
    }
    Ok(lowered)
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

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaskUploadAllocationObservationForTest {
    pub(crate) allocation_extents: Vec<PhysicalSize>,
    pub(crate) retained_upload_count: usize,
}

#[cfg(test)]
pub(crate) fn mask_upload_allocation_observation_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> MaskUploadAllocationObservationForTest {
    let Some(lowered) = lowered_c09_mask_plan_for_test(commands, context) else {
        return MaskUploadAllocationObservationForTest::default();
    };
    let allocation_extents = lowered
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.import,
                Some(RuntimeResourceImport::ResolvedAlphaMask(_))
            )
        })
        .map(|resource| resource.spatial.device_extent)
        .collect::<Vec<_>>();
    MaskUploadAllocationObservationForTest {
        retained_upload_count: allocation_extents.len(),
        allocation_extents,
    }
}

#[cfg(test)]
pub(crate) fn composite_parameter_bytes_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> Option<[u8; 112]> {
    let plan = lowered_c09_mask_plan_for_test(commands, context)?;
    plan.passes.iter().find_map(|pass| {
        let RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) = &pass.kind
        else {
            return None;
        };
        parameters
            .alpha_mask()
            .and_then(|_| CompositeParameterBytes::try_from_runtime_layer(parameters).ok())
            .map(CompositeParameterBytes::into_bytes_for_test)
    })
}

#[cfg(test)]
pub(crate) fn mask_pipeline_keys_exclude_image_identity_for_test(
    first: RenderCommands,
    second: RenderCommands,
    context: FrameContext,
) -> bool {
    fn first_mask_keys(plan: &LoweredGraphPlan) -> Option<&RuntimePassCacheKeys> {
        plan.passes.iter().find_map(|pass| match &pass.kind {
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) if parameters.alpha_mask().is_some() => pass.cache_keys.as_ref(),
            _ => None,
        })
    }

    let Some(first) = lowered_c09_mask_plan_for_test(first, context) else {
        return false;
    };
    let Some(second) = lowered_c09_mask_plan_for_test(second, context) else {
        return false;
    };
    first_mask_keys(&first) == first_mask_keys(&second)
}

#[cfg(test)]
fn lowered_c09_mask_plan_for_test(
    commands: RenderCommands,
    context: FrameContext,
) -> Option<LoweredGraphPlan> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()
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
    ClipCoverage,
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
    ClipCoverageRgba8Unorm,
    Working(WorkingFormat),
    ResolvedMaskRgba8Unorm,
}

impl RuntimeResourceFormat {
    const fn shader_key(self) -> ShaderTextureFormatKey {
        match self {
            Self::VelloCaptureRgba8Unorm => ShaderTextureFormatKey::VelloCaptureRgba8Unorm,
            Self::ClipCoverageRgba8Unorm => ShaderTextureFormatKey::ClipCoverageRgba8Unorm,
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverageElement {
    clip: RenderClip,
    transform: Transform,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeClipCoverage {
    elements: Vec<RuntimeClipCoverageElement>,
    antialiasing: Antialiasing,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeVelloCapture {
    Span(RuntimeVelloSpan),
    ClipCoverage(RuntimeClipCoverage),
}

impl RuntimeVelloCapture {
    const fn antialiasing(&self) -> Antialiasing {
        match self {
            Self::Span(span) => span.antialiasing,
            Self::ClipCoverage(coverage) => coverage.antialiasing,
        }
    }

    fn span(&self) -> Option<&RuntimeVelloSpan> {
        match self {
            Self::Span(span) => Some(span),
            Self::ClipCoverage(_) => None,
        }
    }

    fn clip_coverage(&self) -> Option<&RuntimeClipCoverage> {
        match self {
            Self::Span(_) => None,
            Self::ClipCoverage(coverage) => Some(coverage),
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeDestinationToLayerLocal {
    affine: Transform,
}

impl RuntimeDestinationToLayerLocal {
    fn try_new(affine: Transform) -> Result<Self> {
        if !runtime_affine_is_finite_and_non_singular(affine) {
            return Err(Error::invalid_value(
                "destination-to-layer-local affine mapping",
                format!("{:?}", affine.as_array()),
                "must be finite and non-singular",
            ));
        }
        Ok(Self { affine })
    }

    #[must_use]
    pub(crate) const fn affine(self) -> Transform {
        self.affine
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeMaskTexelCenterFacts {
    half_texel_normalized: [f64; 2],
    texel_size_normalized: [f64; 2],
}

impl RuntimeMaskTexelCenterFacts {
    fn try_new(image_dimensions: PhysicalSize) -> Result<Self> {
        if image_dimensions.width() == 0 || image_dimensions.height() == 0 {
            return Err(Error::invalid_value(
                "composite mask image dimensions",
                format!("{}x{}", image_dimensions.width(), image_dimensions.height()),
                "must be positive before deriving texel-center facts",
            ));
        }
        let texel_size_normalized = [
            1.0 / f64::from(image_dimensions.width()),
            1.0 / f64::from(image_dimensions.height()),
        ];
        let half_texel_normalized = [
            texel_size_normalized[0] * 0.5,
            texel_size_normalized[1] * 0.5,
        ];
        if texel_size_normalized
            .into_iter()
            .chain(half_texel_normalized)
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(lowering_error(
                "composite mask texel-center facts must be finite and positive",
            ));
        }
        Ok(Self {
            half_texel_normalized,
            texel_size_normalized,
        })
    }

    #[must_use]
    pub(crate) const fn half_texel_normalized(self) -> [f64; 2] {
        self.half_texel_normalized
    }

    #[must_use]
    pub(crate) const fn texel_size_normalized(self) -> [f64; 2] {
        self.texel_size_normalized
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeResolvedAlphaMaskComposition {
    resource: RuntimeResourceId,
    bounds: Rect,
    image_dimensions: PhysicalSize,
    texel_center_facts: RuntimeMaskTexelCenterFacts,
    sampling: ShaderMaskSamplingKey,
}

impl RuntimeResolvedAlphaMaskComposition {
    fn try_new(
        resource: RuntimeResourceId,
        bounds: Rect,
        image_dimensions: PhysicalSize,
        sampling: ShaderMaskSamplingKey,
    ) -> Result<Self> {
        let maximum_x = bounds.x() + bounds.width();
        let maximum_y = bounds.y() + bounds.height();
        if !bounds.x().is_finite()
            || !bounds.y().is_finite()
            || !bounds.width().is_finite()
            || !bounds.height().is_finite()
            || bounds.width() <= 0.0
            || bounds.height() <= 0.0
            || !maximum_x.is_finite()
            || !maximum_y.is_finite()
        {
            return Err(Error::invalid_value(
                "composite mask semantic bounds",
                format!(
                    "({}, {}, {}, {})",
                    bounds.x(),
                    bounds.y(),
                    bounds.width(),
                    bounds.height()
                ),
                "must be a finite positive rectangle with a finite maximum",
            ));
        }
        Ok(Self {
            resource,
            bounds,
            image_dimensions,
            texel_center_facts: RuntimeMaskTexelCenterFacts::try_new(image_dimensions)?,
            sampling,
        })
    }

    #[must_use]
    pub(crate) const fn resource(self) -> RuntimeResourceId {
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
    pub(crate) const fn texel_center_facts(self) -> RuntimeMaskTexelCenterFacts {
        self.texel_center_facts
    }

    #[must_use]
    pub(crate) const fn sampling(self) -> ShaderMaskSamplingKey {
        self.sampling
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeLayerCompositeParameters {
    destination_to_layer_local: RuntimeDestinationToLayerLocal,
    opacity: f32,
    blend: BlendMode,
    has_clip: bool,
    alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
}

impl RuntimeLayerCompositeParameters {
    fn try_new(
        destination_to_layer_local: Transform,
        opacity: f32,
        blend: BlendMode,
        has_clip: bool,
        alpha_mask: Option<RuntimeResolvedAlphaMaskComposition>,
    ) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "composite opacity",
                opacity,
                "must be finite before clamping",
            ));
        }
        Ok(Self {
            destination_to_layer_local: RuntimeDestinationToLayerLocal::try_new(
                destination_to_layer_local,
            )?,
            opacity: opacity.clamp(0.0, 1.0),
            blend,
            has_clip,
            alpha_mask,
        })
    }

    #[must_use]
    pub(crate) const fn destination_to_layer_local(self) -> RuntimeDestinationToLayerLocal {
        self.destination_to_layer_local
    }

    #[must_use]
    pub(crate) const fn opacity(self) -> f32 {
        self.opacity
    }

    #[must_use]
    pub(crate) const fn blend(self) -> BlendMode {
        self.blend
    }

    #[must_use]
    pub(crate) const fn has_clip(self) -> bool {
        self.has_clip
    }

    #[must_use]
    pub(crate) const fn alpha_mask(self) -> Option<RuntimeResolvedAlphaMaskComposition> {
        self.alpha_mask
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RuntimeCompositeKind {
    SpanSourceOver,
    Layer {
        transform: Transform,
        parameters: Box<RuntimeLayerCompositeParameters>,
        clip: Option<Box<RenderClip>>,
        outer_clips: Vec<RuntimeOuterClip>,
        clip_coverage: Option<RuntimeResourceId>,
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
    VelloCapture(Option<RuntimeVelloCapture>),
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
    ClipCoverage,
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

#[derive(Clone)]
struct ExecutableLayerCompositionFacts {
    pass: RuntimePassId,
    parent: RuntimeResourceId,
    source: RuntimeResourceId,
    clip_coverage: Option<RuntimeResourceId>,
    alpha_mask: Option<RuntimeResourceId>,
    result: RuntimeResourceId,
    composite: RuntimeComposite,
}

#[derive(Clone)]
struct ClosedExecutableGraphFacts {
    working_format: WorkingFormat,
    output_format: Format,
    captures: Vec<ExecutableVelloCaptureFacts>,
    layer_compositions: Vec<ExecutableLayerCompositionFacts>,
}

#[derive(Clone, Copy)]
struct ExecutableCompositionContext {
    current: RuntimeResourceId,
    producer: RuntimePassId,
    contains_captured_source: bool,
}

#[must_use = "a closed executable graph must reach dispatch or explicit rejection"]
struct ClosedExecutableGraph {
    lowered: LoweredGraphPlan,
    facts: ClosedExecutableGraphFacts,
}

impl ClosedExecutableGraph {
    fn try_from_lowered(lowered: LoweredGraphPlan) -> std::result::Result<Self, LoweredGraphPlan> {
        let Some(facts) = lowered.closed_executable_graph_facts() else {
            return Err(lowered);
        };
        if !facts.proves_exact_facts_for(&lowered) {
            return Err(lowered);
        }
        Ok(Self { lowered, facts })
    }

    fn into_lowered(self) -> LoweredGraphPlan {
        self.lowered
    }

    fn has_layer_composition(&self) -> bool {
        !self.facts.layer_compositions.is_empty()
    }
}

impl ClosedExecutableGraphFacts {
    fn proves_exact_facts_for(&self, plan: &LoweredGraphPlan) -> bool {
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
        captures_are_exact && layers_are_exact
    }
}

#[derive(Clone, Copy)]
enum GraphLoweringCapabilityValidation<'capabilities> {
    Required(&'capabilities DeviceCapabilities),
    ClassificationOnly,
}

#[must_use]
pub(crate) struct C08ExecutionFacts {
    working_format: WorkingFormat,
    output_format: Format,
    captures: Vec<ExecutableVelloCaptureFacts>,
}

impl C08ExecutionFacts {
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
pub(crate) struct C08PreparableGraph {
    lowered: LoweredGraphPlan,
    execution: C08ExecutionFacts,
}

impl C08PreparableGraph {
    #[cfg(test)]
    pub(crate) fn try_from_lowered(
        lowered: LoweredGraphPlan,
    ) -> std::result::Result<Self, LoweredGraphPlan> {
        let closed = ClosedExecutableGraph::try_from_lowered(lowered)?;
        Self::try_from_closed(closed).map_err(|closed| (*closed).into_lowered())
    }

    fn try_from_closed(
        closed: ClosedExecutableGraph,
    ) -> std::result::Result<Self, Box<ClosedExecutableGraph>> {
        if closed.has_layer_composition()
            || closed.facts.captures.is_empty()
            || closed.facts.captures.iter().any(|capture| {
                capture
                    .span()
                    .is_none_or(|span| span.scope != RuntimeVelloSpanScope::CurrentParent)
            })
        {
            return Err(Box::new(closed));
        }
        let execution = C08ExecutionFacts {
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

    fn into_parts(self) -> (LoweredGraphPlan, C08ExecutionFacts) {
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
            .ok_or_else(|| preparation_error("the C08 root output resource is missing"))
    }

    #[cfg(test)]
    pub(crate) fn capture_grids_for_test(&self) -> Vec<C08CaptureGridForTest> {
        self.execution
            .captures()
            .iter()
            .map(|capture| C08CaptureGridForTest {
                texel_origin: capture.texel_origin(),
                extent: capture.target_extent(),
                raster_scale: capture.raster_scale(),
            })
            .collect()
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C08CaptureGridForTest {
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
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
    fn span(&self) -> Option<&RuntimeVelloSpan> {
        self.work.span()
    }

    #[cfg(test)]
    fn commands(&self) -> Option<&RenderCommands> {
        self.span().map(|span| &span.commands)
    }

    #[must_use]
    const fn work(&self) -> &RuntimeVelloCapture {
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08PassClass {
    ClearRoot,
    VelloCapture,
    CanonicalizeCapture,
    SpanSourceOver,
    Present,
}

#[cfg(test)]
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

    fn try_lower_for_dispatch_classification(
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
            let format = runtime_resource_format(role, working_format);
            if let GraphLoweringCapabilityValidation::Required(capabilities) = capability_validation
            {
                capabilities.validate_effect_texture_extent(spatial.device_extent)?;
            }
            if let (
                GraphLoweringCapabilityValidation::Required(capabilities),
                RuntimeResourceFormat::Working(format),
            ) = (capability_validation, format)
            {
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
            let kind = runtime_pass_kind(graph_kind, working_format)?;
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

    fn closed_executable_graph_facts(&self) -> Option<ClosedExecutableGraphFacts> {
        if !matches!(self.output_format, Format::Rgba8 | Format::Bgra8) || self.passes.len() < 5 {
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
        let pass_positions = self
            .passes
            .iter()
            .enumerate()
            .map(|(position, pass)| (pass.id, position))
            .collect::<BTreeMap<_, _>>();
        if pass_positions.len() != self.passes.len() {
            return None;
        }

        let mut actual_reads = BTreeMap::<RuntimeResourceId, u32>::new();
        let mut actual_last_reads = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        let mut releases = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        let mut results = BTreeMap::<RuntimeResourceId, RuntimePassId>::new();
        for (position, pass) in self.passes.iter().enumerate() {
            let mut dependencies = BTreeSet::new();
            if pass.dependencies.iter().any(|dependency| {
                !dependencies.insert(*dependency)
                    || pass_positions
                        .get(dependency)
                        .is_none_or(|dependency_position| *dependency_position >= position)
            }) {
                return None;
            }
            let mut pass_reads = BTreeSet::new();
            for read in &pass.reads {
                if !pass_reads.insert(read.resource)
                    || !runtime_read_sampler_is_exact(read, &resource_by_id)
                {
                    return None;
                }
                let resource = resource_by_id.get(&read.resource).copied()?;
                if pass.result == RuntimeResultBinding::Resource(read.resource) {
                    return None;
                }
                if let RuntimeResourceProducer::Pass(producer) = resource.producer
                    && (pass_positions
                        .get(&producer)
                        .is_none_or(|producer_position| *producer_position >= position)
                        || !pass.dependencies.contains(&producer))
                {
                    return None;
                }
                let reads = actual_reads.entry(read.resource).or_default();
                *reads = reads.checked_add(1)?;
                actual_last_reads.insert(read.resource, pass.id);
            }
            match pass.result {
                RuntimeResultBinding::Resource(resource) => {
                    let request = resource_by_id.get(&resource).copied()?;
                    if request.producer != RuntimeResourceProducer::Pass(pass.id)
                        || results.insert(resource, pass.id).is_some()
                    {
                        return None;
                    }
                }
                RuntimeResultBinding::Output(format) => {
                    if !matches!(pass.kind, RuntimePassKind::Present)
                        || format != self.output_format
                    {
                        return None;
                    }
                }
                RuntimeResultBinding::Empty => {}
            }
            let mut pass_releases = BTreeSet::new();
            if pass.releases.iter().any(|resource| {
                !pass_releases.insert(*resource)
                    || !pass_reads.contains(resource)
                    || releases.insert(*resource, pass.id).is_some()
            }) {
                return None;
            }
            let expected_cache_keys = runtime_pass_cache_keys(
                &pass.kind,
                &pass.reads,
                pass.result,
                self.working_format,
                self.output_format,
                &resource_formats,
            )
            .ok()?;
            if expected_cache_keys != pass.cache_keys {
                return None;
            }
        }
        for resource in &self.resources {
            if resource.format != runtime_resource_format(resource.role, self.working_format)
                || resource.expected_reads == 0
                || actual_reads.get(&resource.id).copied() != Some(resource.expected_reads)
                || actual_last_reads.get(&resource.id).copied() != Some(resource.last_use)
                || releases.get(&resource.id).copied() != Some(resource.last_use)
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
                    if results.get(&resource.id).copied() == Some(*pass) => {}
                _ => return None,
            }
        }

        let clear = self.passes.first()?;
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

        let mut contexts = vec![ExecutableCompositionContext {
            current: root.id,
            producer: clear.id,
            contains_captured_source: false,
        }];
        let mut captures = Vec::new();
        let mut layer_compositions = Vec::new();
        let mut expected_resources = BTreeSet::from([root.id]);
        let mut cursor = 1usize;
        while cursor < self.passes.len() {
            let pass = self.passes.get(cursor)?;
            match &pass.kind {
                RuntimePassKind::ClearRoot {
                    initialization: RuntimeInitialization::Transparent,
                    color,
                } => {
                    if *color != Color::TRANSPARENT
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
                    let request = resource_by_id.get(&resource).copied()?;
                    if !c08_resource_has_fixed_facts(
                        request,
                        RuntimeResourceRole::IsolationWorkingImage,
                        RuntimeResourceFormat::Working(self.working_format),
                        RuntimeResourceProducer::Pass(pass.id),
                    ) {
                        return None;
                    }
                    expected_resources.insert(resource);
                    contexts.push(ExecutableCompositionContext {
                        current: resource,
                        producer: pass.id,
                        contains_captured_source: false,
                    });
                    cursor = cursor.checked_add(1)?;
                }
                RuntimePassKind::VelloCapture(Some(work)) if work.span().is_some() => {
                    let span = work.span()?;
                    let canonicalize = self.passes.get(cursor.checked_add(1)?)?;
                    let after_canonicalize = self.passes.get(cursor.checked_add(2)?)?;
                    let (coverage_pass, composite, pass_count) = if matches!(
                        after_canonicalize.kind,
                        RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::ClipCoverage(_)))
                    ) {
                        (
                            Some(after_canonicalize),
                            self.passes.get(cursor.checked_add(3)?)?,
                            4,
                        )
                    } else {
                        (None, after_canonicalize, 3)
                    };
                    let RuntimeResultBinding::Resource(capture_target) = pass.result else {
                        return None;
                    };
                    if !pass.dependencies.is_empty()
                        || !pass.reads.is_empty()
                        || !pass.releases.is_empty()
                        || pass.cache_keys.is_some()
                        || span.scope
                            != if contexts.len() == 1 {
                                RuntimeVelloSpanScope::CurrentParent
                            } else {
                                RuntimeVelloSpanScope::LayerSource
                            }
                    {
                        return None;
                    }
                    let capture_resource = resource_by_id.get(&capture_target).copied()?;
                    if !c08_resource_has_fixed_facts(
                        capture_resource,
                        RuntimeResourceRole::CaptureWorkingImage,
                        RuntimeResourceFormat::VelloCaptureRgba8Unorm,
                        RuntimeResourceProducer::Pass(pass.id),
                    ) || capture_resource.expected_reads != 1
                        || capture_resource.last_use != canonicalize.id
                    {
                        return None;
                    }
                    let capture_facts = executable_vello_capture_facts(
                        pass.id,
                        capture_target,
                        work,
                        capture_resource.spatial,
                    )?;

                    if !matches!(canonicalize.kind, RuntimePassKind::CanonicalizeCapture)
                        || canonicalize.dependencies.as_slice() != [pass.id]
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
                    let RuntimeResultBinding::Resource(canonical_target) = canonicalize.result
                    else {
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

                    let coverage_facts = match coverage_pass {
                        Some(coverage_pass) => Some(validate_closed_clip_coverage_capture(
                            coverage_pass,
                            composite.id,
                            &resource_by_id,
                        )?),
                        None => None,
                    };

                    let parent = *contexts.last()?;
                    let layer = validate_closed_composite(
                        composite,
                        parent,
                        canonical_resource,
                        &resource_by_id,
                        self.working_format,
                        false,
                    )?;
                    let RuntimeResultBinding::Resource(result) = composite.result else {
                        return None;
                    };
                    let context = contexts.last_mut()?;
                    context.current = result;
                    context.producer = composite.id;
                    context.contains_captured_source = true;
                    expected_resources.extend([capture_target, canonical_target, result]);
                    captures.push(capture_facts);
                    if let Some((coverage, facts)) = coverage_facts {
                        expected_resources.insert(coverage);
                        captures.push(facts);
                    }
                    if let Some(layer) = layer {
                        if let Some(coverage) = layer.clip_coverage {
                            expected_resources.insert(coverage);
                        }
                        if let Some(mask) = layer.alpha_mask {
                            expected_resources.insert(mask);
                        }
                        layer_compositions.push(layer);
                    }
                    cursor = cursor.checked_add(pass_count)?;
                }
                RuntimePassKind::VelloCapture(Some(work)) if work.clip_coverage().is_some() => {
                    let composite = self.passes.get(cursor.checked_add(1)?)?;
                    let (coverage, facts) =
                        validate_closed_clip_coverage_capture(pass, composite.id, &resource_by_id)?;
                    expected_resources.insert(coverage);
                    captures.push(facts);
                    cursor = cursor.checked_add(1)?;
                }
                RuntimePassKind::Composite(Some(composite))
                    if matches!(composite.kind, RuntimeCompositeKind::Layer { .. }) =>
                {
                    if contexts.len() < 2 {
                        return None;
                    }
                    let source_context = contexts.pop()?;
                    if !source_context.contains_captured_source {
                        return None;
                    }
                    let parent = *contexts.last()?;
                    let source = resource_by_id.get(&source_context.current).copied()?;
                    let layer = validate_closed_composite(
                        pass,
                        parent,
                        source,
                        &resource_by_id,
                        self.working_format,
                        true,
                    )??;
                    let RuntimeResultBinding::Resource(result) = pass.result else {
                        return None;
                    };
                    let context = contexts.last_mut()?;
                    context.current = result;
                    context.producer = pass.id;
                    context.contains_captured_source = true;
                    expected_resources.insert(result);
                    if let Some(coverage) = layer.clip_coverage {
                        expected_resources.insert(coverage);
                    }
                    if let Some(mask) = layer.alpha_mask {
                        expected_resources.insert(mask);
                    }
                    layer_compositions.push(layer);
                    cursor = cursor.checked_add(1)?;
                }
                RuntimePassKind::Present => {
                    if cursor.checked_add(1)? != self.passes.len()
                        || pass.id != self.final_present
                        || contexts.len() != 1
                    {
                        return None;
                    }
                    let parent = contexts[0];
                    let parent_resource = resource_by_id.get(&parent.current).copied()?;
                    if pass.dependencies.as_slice() != [parent.producer]
                        || pass.reads.len() != 1
                        || !runtime_read_has_exact_facts(
                            &pass.reads[0],
                            RuntimeReadRole::FinalWorkingImage,
                            parent_resource,
                            RuntimeSamplingFilter::Linear,
                            RuntimeSamplingEdge::ClampToExtent,
                        )
                        || pass.result != RuntimeResultBinding::Output(self.output_format)
                        || pass.releases.as_slice() != [parent.current]
                        || parent_resource.expected_reads != 1
                        || parent_resource.last_use != pass.id
                    {
                        return None;
                    }
                    cursor = cursor.checked_add(1)?;
                }
                RuntimePassKind::ClearRoot {
                    initialization: RuntimeInitialization::SurfaceBaseColor,
                    ..
                }
                | RuntimePassKind::VelloCapture(None)
                | RuntimePassKind::VelloCapture(Some(_))
                | RuntimePassKind::CanonicalizeCapture
                | RuntimePassKind::CopyBackdrop
                | RuntimePassKind::ColorFilter(_)
                | RuntimePassKind::BlurHorizontal(_)
                | RuntimePassKind::BlurVertical(_)
                | RuntimePassKind::DropShadowColorize(_)
                | RuntimePassKind::Composite(_) => return None,
            }
        }
        let clip_coverages_are_exact = layer_compositions
            .iter()
            .all(|layer| layer_has_exact_clip_coverage_capture(layer, &captures))
            && captures.iter().all(|capture| {
                capture.work().clip_coverage().is_none()
                    || layer_compositions
                        .iter()
                        .any(|layer| layer.clip_coverage == Some(capture.target()))
            });
        if captures.is_empty()
            || contexts.len() != 1
            || !clip_coverages_are_exact
            || expected_resources.len() != self.resources.len()
            || expected_resources
                .iter()
                .any(|resource| !resource_by_id.contains_key(resource))
        {
            return None;
        }

        Some(ClosedExecutableGraphFacts {
            working_format: self.working_format,
            output_format: self.output_format,
            captures,
            layer_compositions,
        })
    }

    #[cfg(test)]
    fn c08_execution_facts(&self) -> Option<C08ExecutionFacts> {
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
            let RuntimePassKind::VelloCapture(Some(work @ RuntimeVelloCapture::Span(_))) =
                &capture.kind
            else {
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
            captures.push(executable_vello_capture_facts(
                capture.id,
                capture_target,
                work,
                capture_resource.spatial,
            )?);
            parent = composite_resource;
            parent_producer = composite.id;
            cursor = cursor.checked_add(3)?;
        }

        if captures.is_empty() {
            return None;
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

        let execution = C08ExecutionFacts {
            working_format: self.working_format,
            output_format: self.output_format,
            captures,
        };
        execution
            .proves_exact_execution_facts_for(self)
            .then_some(execution)
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
    if !c08_resource_has_fixed_facts(
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

fn validate_closed_composite(
    pass: &RuntimePass,
    parent: ExecutableCompositionContext,
    source: &RuntimeResourceRequest,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    working_format: WorkingFormat,
    requires_isolated_source: bool,
) -> Option<Option<ExecutableLayerCompositionFacts>> {
    let RuntimePassKind::Composite(Some(composite)) = &pass.kind else {
        return None;
    };
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
        || parent_resource.expected_reads != 1
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
    if !c08_resource_has_fixed_facts(
        result_resource,
        RuntimeResourceRole::CompositeResult,
        RuntimeResourceFormat::Working(working_format),
        RuntimeResourceProducer::Pass(pass.id),
    ) || result_resource.spatial != parent_resource.spatial
    {
        return None;
    }

    match &composite.kind {
        RuntimeCompositeKind::SpanSourceOver => {
            if requires_isolated_source
                || pass.reads.len() != 2
                || source.role != RuntimeResourceRole::FilterIntermediate
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
            let opacity = parameters.opacity();
            let blend = parameters.blend();
            let alpha_mask = parameters.alpha_mask();
            if transform.as_array().iter().any(|value| !value.is_finite())
                || !opacity.is_finite()
                || !(0.0..=1.0).contains(&opacity)
                || !runtime_affine_is_finite_and_non_singular(
                    parameters.destination_to_layer_local().affine(),
                )
                || outer_clips.iter().any(|clip| {
                    clip.transform
                        .as_array()
                        .iter()
                        .any(|value| !value.is_finite())
                })
                || parameters.has_clip() != (clip.is_some() || !outer_clips.is_empty())
                || parameters.has_clip() != clip_coverage.is_some()
                || (requires_isolated_source && source.role != RuntimeResourceRole::CompositeResult)
                || (!requires_isolated_source
                    && (source.role != RuntimeResourceRole::FilterIntermediate
                        || *transform != Transform::identity()
                        || opacity != 1.0
                        || blend != BlendMode::Normal
                        || clip.is_some()
                        || outer_clips.is_empty()
                        || alpha_mask.is_some()))
            {
                return None;
            }
            let expected_read_count = 2usize
                .checked_add(usize::from(clip_coverage.is_some()))?
                .checked_add(usize::from(alpha_mask.is_some()))?;
            if pass.reads.len() != expected_read_count {
                return None;
            }
            let mut next_read = 2;
            if let Some(coverage) = clip_coverage {
                let coverage_resource = resources.get(coverage).copied()?;
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
                next_read = next_read.checked_add(1)?;
            }
            if let Some(mask) = alpha_mask {
                let mask_resource = resources.get(&mask.resource()).copied()?;
                let Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) = &mask_resource.import
                else {
                    return None;
                };
                let mask_filter = match mask.sampling().quality() {
                    ShaderMaskQualityKey::Low => RuntimeSamplingFilter::Nearest,
                    ShaderMaskQualityKey::Medium | ShaderMaskQualityKey::High => {
                        RuntimeSamplingFilter::Linear
                    }
                };
                if !runtime_read_has_exact_facts(
                    &pass.reads[next_read],
                    RuntimeReadRole::AlphaMask,
                    mask_resource,
                    mask_filter,
                    RuntimeSamplingEdge::ClampToExtent,
                ) || mask_resource.role != RuntimeResourceRole::ImportedImage
                    || mask_resource.format != RuntimeResourceFormat::ResolvedMaskRgba8Unorm
                    || mask_resource.producer != RuntimeResourceProducer::Imported
                    || upload.physical_size() != mask.image_dimensions()
                    || mask_resource.spatial.device_extent != mask.image_dimensions()
                    || mask.sampling()
                        != ShaderMaskSamplingKey::new(upload.quality(), upload.extend())
                    || RuntimeMaskTexelCenterFacts::try_new(mask.image_dimensions()).ok()
                        != Some(mask.texel_center_facts())
                    || mask_resource.expected_reads == 0
                    || mask_resource.last_use < pass.id
                {
                    return None;
                }
            }
            Some(Some(ExecutableLayerCompositionFacts {
                pass: pass.id,
                parent: parent.current,
                source: source.id,
                clip_coverage: *clip_coverage,
                alpha_mask: alpha_mask.map(RuntimeResolvedAlphaMaskComposition::resource),
                result,
                composite: composite.clone(),
            }))
        }
        RuntimeCompositeKind::DropShadow => None,
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

#[cfg(test)]
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

fn executable_vello_capture_facts(
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
        RuntimeVelloCapture::Span(span) => span
            .capture_transform
            .then(span.parent_to_surface)
            .ok()?
            .then(grid_transform)
            .ok()?,
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
    let rgba_preparable = C08PreparableGraph::try_from_lowered(rgba.clone()).ok()?;
    let bgra_preparable = C08PreparableGraph::try_from_lowered(bgra).ok()?;
    let rgba_subset = &rgba_preparable.execution;
    let bgra_subset = &bgra_preparable.execution;
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
        rejects_later_cycle_plan: C08PreparableGraph::try_from_lowered(later_cycle).is_err(),
        preserves_direct_and_transitional_planner_routes: direct_route && transitional_route,
    })
}

#[cfg(test)]
fn c09_executable_graph_observation(
    c08_commands: RenderCommands,
    c09_commands: RenderCommands,
    c10_commands: RenderCommands,
    context: FrameContext,
    capabilities: DeviceCapabilities,
) -> Option<C09ExecutableGraphObservationForTest> {
    let c08_graph = super::frame::forced_c08_graph_for_test(c08_commands, context).ok()?;
    let FramePlan::GpuGraph(c09_graph) = c09_commands.plan_for(context).ok()? else {
        return None;
    };
    let FramePlan::GpuGraph(c10_graph) = c10_commands.plan_for(context).ok()? else {
        return None;
    };

    let mut accepts_spine_and_layer_composition_for_all_formats = true;
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        for output_format in [Format::Rgba8, Format::Bgra8] {
            let c08 = LoweredGraphPlan::try_lower_for_dispatch_classification(
                &c08_graph,
                working_format,
                output_format,
            )
            .ok()
            .and_then(|lowered| ClosedExecutableGraph::try_from_lowered(lowered).ok());
            let c09 = LoweredGraphPlan::try_lower_for_dispatch_classification(
                &c09_graph,
                working_format,
                output_format,
            )
            .ok()
            .and_then(|lowered| ClosedExecutableGraph::try_from_lowered(lowered).ok());
            accepts_spine_and_layer_composition_for_all_formats &= c08
                .is_some_and(|closed| !closed.has_layer_composition())
                && c09.is_some_and(|closed| closed.has_layer_composition());
        }
    }

    let c09_lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &c09_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let c09_closed = ClosedExecutableGraph::try_from_lowered(c09_lowered.clone()).ok()?;
    let layer_composition_reads_are_exact = !c09_closed.facts.layer_compositions.is_empty()
        && c09_closed.facts.layer_compositions.iter().all(|layer| {
            let pass = c09_closed
                .lowered
                .passes
                .iter()
                .find(|pass| pass.id == layer.pass);
            pass.is_some_and(|pass| {
                let mut expected = vec![
                    (RuntimeReadRole::CompositeParent, layer.parent),
                    (RuntimeReadRole::CompositeSource, layer.source),
                ];
                if let Some(coverage) = layer.clip_coverage {
                    expected.push((RuntimeReadRole::ClipCoverage, coverage));
                }
                if let Some(mask) = layer.alpha_mask {
                    expected.push((RuntimeReadRole::AlphaMask, mask));
                }
                pass.reads
                    .iter()
                    .map(|read| (read.role, read.resource))
                    .eq(expected)
            })
        });

    let c10_lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &c10_graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let rejects_actual_c10 = ClosedExecutableGraph::try_from_lowered(c10_lowered).is_err();
    let layer_index = c09_lowered.passes.iter().position(|pass| {
        matches!(
            pass.kind,
            RuntimePassKind::Composite(Some(RuntimeComposite {
                kind: RuntimeCompositeKind::Layer { .. },
                ..
            }))
        )
    })?;
    let capture_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::VelloCapture(Some(_))))?;
    let canonicalize_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::CanonicalizeCapture))?;
    let present_index = c09_lowered
        .passes
        .iter()
        .position(|pass| matches!(pass.kind, RuntimePassKind::Present))?;
    let rejects = |invalid| ClosedExecutableGraph::try_from_lowered(invalid).is_err();

    let mut invalid_copy = c09_lowered.clone();
    invalid_copy.passes[layer_index].kind = RuntimePassKind::CopyBackdrop;
    let mut invalid_payload = c09_lowered.clone();
    invalid_payload.passes[layer_index].kind = RuntimePassKind::Composite(Some(RuntimeComposite {
        kind: RuntimeCompositeKind::DropShadow,
        source_captured_before_outer_semantics: true,
    }));
    let rejects_c10_plus_passes_and_payloads =
        rejects_actual_c10 && rejects(invalid_copy) && rejects(invalid_payload);

    let mut missing_capture = c09_lowered.clone();
    missing_capture.passes[capture_index].kind = RuntimePassKind::VelloCapture(None);
    let mut missing_composite = c09_lowered.clone();
    missing_composite.passes[layer_index].kind = RuntimePassKind::Composite(None);
    let rejects_missing_payloads = rejects(missing_capture) && rejects(missing_composite);

    let mut malformed = Vec::new();
    let mut invalid = c09_lowered.clone();
    invalid.passes[canonicalize_index].dependencies.clear();
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.passes[layer_index].reads.swap(0, 1);
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.passes[layer_index].result =
        RuntimeResultBinding::Resource(invalid.passes[layer_index].reads[0].resource);
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.passes[layer_index].releases.clear();
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.resources[0].expected_reads = invalid.resources[0].expected_reads.saturating_add(1);
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.passes.swap(capture_index, canonicalize_index);
    malformed.push(invalid);
    let mut invalid = c09_lowered.clone();
    invalid.resources[1].id = invalid.resources[0].id;
    malformed.push(invalid);
    let rejects_malformed_graph_facts = malformed.into_iter().all(rejects);

    let mut unsupported_output = c09_lowered;
    unsupported_output.passes[present_index].result = RuntimeResultBinding::Output(Format::Bgra8);
    let rejects_unsupported_output_binding = rejects(unsupported_output);
    let preserves_transitional_c09_dispatch = matches!(
        ExecutableGraphDispatchEligibility::try_classify(
            &c09_graph,
            Format::Rgba8,
            ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
            &capabilities,
        ),
        Ok(ExecutableGraphDispatchEligibility::LaterCycleTransitional)
    );

    Some(C09ExecutableGraphObservationForTest {
        accepts_spine_and_layer_composition_for_all_formats,
        layer_composition_reads_are_exact,
        rejects_c10_plus_passes_and_payloads,
        rejects_missing_payloads,
        rejects_malformed_graph_facts,
        rejects_unsupported_output_binding,
        preserves_transitional_c09_dispatch,
    })
}

fn vello_capture_raster_parameters(
    target_extent: PhysicalSize,
    antialiasing: Antialiasing,
) -> Result<RasterParameters> {
    RasterParameters::try_new(target_extent, peniko::Color::TRANSPARENT, antialiasing)
}

#[cfg(test)]
fn graph_clip_coverage_observation(
    commands: RenderCommands,
    context: FrameContext,
    _capabilities: DeviceCapabilities,
) -> Option<GraphClipCoverageObservationForTest> {
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let closed = ClosedExecutableGraph::try_from_lowered(lowered).ok()?;
    let all_vello_capture_count = closed
        .lowered
        .passes
        .iter()
        .filter(|pass| matches!(pass.kind, RuntimePassKind::VelloCapture(Some(_))))
        .count();
    let composite_coverage_read_count = closed
        .lowered
        .passes
        .iter()
        .flat_map(|pass| &pass.reads)
        .filter(|read| read.role == RuntimeReadRole::ClipCoverage)
        .count();
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let captures = closed
        .facts
        .captures
        .iter()
        .filter_map(|capture| {
            let coverage = capture.work().clip_coverage()?;
            let resource = resources.get(&capture.target()).copied()?;
            let elements = coverage
                .elements
                .iter()
                .map(|element| (element.clip.clone(), element.transform))
                .collect::<Vec<_>>();
            let scene = encode_vello_clip_coverage_scene(
                &elements,
                capture.initial_transform(),
                capture.target_extent(),
            )
            .ok()?;
            let emitted_draws = scene.observation_for_test().solid_path_draws_for_test()?;
            let prepared = scene
                .prepare_raster(
                    vello_capture_raster_parameters(
                        capture.target_extent(),
                        capture.antialiasing(),
                    )
                    .ok()?,
                )
                .ok()?;
            let raster = prepared_vello_pass_observation_for_test(&prepared);
            let elements = coverage
                .elements
                .iter()
                .map(|element| ClipCoverageElementObservationForTest {
                    clip: element.clip.clone(),
                    transform: element.transform,
                })
                .collect();
            Some(ClipCoverageCaptureObservationForTest {
                elements,
                antialiasing: capture.antialiasing(),
                device_origin: resource.spatial.device_origin,
                target_extent: capture.target_extent(),
                texel_origin: capture.texel_origin(),
                raster_scale: capture.raster_scale(),
                first_texel_center: Point::new(
                    (f64::from(resource.spatial.device_origin.0) + 0.5) / capture.raster_scale(),
                    (f64::from(resource.spatial.device_origin.1) + 0.5) / capture.raster_scale(),
                ),
                initial_transform: capture.initial_transform(),
                emitted_draws,
                uses_coverage_resource_role: resource.role == RuntimeResourceRole::ClipCoverage,
                uses_rgba8_target: resource.format == RuntimeResourceFormat::ClipCoverageRgba8Unorm,
                uses_transparent_base: raster.transparent_base_for_test(),
                raster_antialiasing: raster.antialiasing_for_test(),
                raster_target_extent: raster.target_extent_for_test(),
            })
        })
        .collect();

    Some(GraphClipCoverageObservationForTest {
        captures,
        all_vello_capture_count,
        composite_coverage_read_count,
    })
}

#[cfg(test)]
fn composition_graph_observation(
    commands: RenderCommands,
    context: FrameContext,
    _capabilities: DeviceCapabilities,
) -> Option<CompositionGraphObservationForTest> {
    let authored_mask_ids = resolved_mask_image_ids_inner_to_outer(&commands.commands);
    let FramePlan::GpuGraph(graph) = commands.plan_for(context).ok()? else {
        return None;
    };
    let lowered = LoweredGraphPlan::try_lower_for_dispatch_classification(
        &graph,
        WorkingFormat::HighPrecision,
        Format::Rgba8,
    )
    .ok()?;
    let closed = ClosedExecutableGraph::try_from_lowered(lowered).ok()?;
    let resources = closed
        .lowered
        .resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect::<BTreeMap<_, _>>();
    let mut observed_mask_ids = Vec::new();
    let layers_inner_to_outer = closed
        .facts
        .layer_compositions
        .iter()
        .map(|layer| {
            let RuntimeCompositeKind::Layer {
                transform,
                parameters,
                clip,
                outer_clips,
                clip_coverage,
            } = &layer.composite.kind
            else {
                return None;
            };
            let alpha_mask = parameters
                .alpha_mask()
                .map(RuntimeResolvedAlphaMaskComposition::resource);
            if clip_coverage != &layer.clip_coverage || alpha_mask != layer.alpha_mask {
                return None;
            }
            if let Some(mask) = alpha_mask {
                let resource = resources.get(&mask).copied()?;
                let Some(RuntimeResourceImport::ResolvedAlphaMask(upload)) = &resource.import
                else {
                    return None;
                };
                observed_mask_ids.push(upload.cache_key().image_id());
            }
            let pass = closed
                .lowered
                .passes
                .iter()
                .find(|pass| pass.id == layer.pass)?;
            let reads = pass
                .reads
                .iter()
                .map(|read| match read.role {
                    RuntimeReadRole::CompositeParent => {
                        Some(CompositionReadObservationForTest::Parent)
                    }
                    RuntimeReadRole::CompositeSource => {
                        Some(CompositionReadObservationForTest::Source)
                    }
                    RuntimeReadRole::ClipCoverage => {
                        Some(CompositionReadObservationForTest::ClipCoverage)
                    }
                    RuntimeReadRole::AlphaMask => {
                        Some(CompositionReadObservationForTest::AlphaMask)
                    }
                    RuntimeReadRole::CaptureSource
                    | RuntimeReadRole::CompletedParent
                    | RuntimeReadRole::FilterSource
                    | RuntimeReadRole::BlurredSourceAlpha
                    | RuntimeReadRole::Shadow
                    | RuntimeReadRole::FinalWorkingImage => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let mut outer_operations =
                vec![CompositionOuterOperationObservationForTest::SourceMapping];
            if clip.is_some() || !outer_clips.is_empty() {
                outer_operations.push(CompositionOuterOperationObservationForTest::ClipCoverage);
            }
            if alpha_mask.is_some() {
                outer_operations.push(CompositionOuterOperationObservationForTest::AlphaMask);
            }
            outer_operations.extend([
                CompositionOuterOperationObservationForTest::Opacity,
                CompositionOuterOperationObservationForTest::Blend,
            ]);
            Some(LayerCompositionObservationForTest {
                transform: *transform,
                opacity: parameters.opacity(),
                blend: parameters.blend(),
                has_own_clip: clip.is_some(),
                inherited_outer_clip_count: outer_clips.len(),
                inherited_outer_clip_transforms: outer_clips
                    .iter()
                    .map(|clip| clip.transform)
                    .collect(),
                reads,
                outer_operations,
                source_captured_before_outer_semantics: layer
                    .composite
                    .source_captured_before_outer_semantics,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut root_surface_base_clears = 0usize;
    let mut root_surface_base_color = None;
    let mut transparent_isolation_clears = 0usize;
    let mut nontransparent_isolation_clears = 0usize;
    for pass in &closed.lowered.passes {
        let RuntimePassKind::ClearRoot {
            initialization,
            color,
        } = pass.kind
        else {
            continue;
        };
        match initialization {
            RuntimeInitialization::SurfaceBaseColor => {
                root_surface_base_clears = root_surface_base_clears.saturating_add(1);
                root_surface_base_color = Some(color);
            }
            RuntimeInitialization::Transparent => {
                if color == Color::TRANSPARENT {
                    transparent_isolation_clears = transparent_isolation_clears.saturating_add(1);
                } else {
                    nontransparent_isolation_clears =
                        nontransparent_isolation_clears.saturating_add(1);
                }
            }
        }
    }

    Some(CompositionGraphObservationForTest {
        layers_inner_to_outer,
        mask_identity_is_preserved: authored_mask_ids == observed_mask_ids,
        root_surface_base_clears,
        root_surface_base_color,
        transparent_isolation_clears,
        nontransparent_isolation_clears,
    })
}

#[cfg(test)]
fn resolved_mask_image_ids_inner_to_outer(
    commands: &[super::command::RenderCommand],
) -> Vec<super::ImageId> {
    fn collect(commands: &[super::command::RenderCommand], image_ids: &mut Vec<super::ImageId>) {
        for command in commands {
            let super::command::RenderCommand::Layer { layer, children } = command else {
                continue;
            };
            collect(children, image_ids);
            if let Some(mask) = &layer.mask {
                image_ids.push(mask.upload().cache_key().image_id());
            }
        }
    }

    let mut image_ids = Vec::new();
    collect(commands, &mut image_ids);
    image_ids
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
    let Ok(layer_parameters) = RuntimeLayerCompositeParameters::try_new(
        Transform::identity(),
        1.0,
        BlendMode::Normal,
        false,
        None,
    ) else {
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
                parameters: Box::new(layer_parameters),
                clip: None,
                outer_clips: Vec::new(),
                clip_coverage: None,
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
        C08PreparableGraph::try_from_lowered(invalid).is_err()
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
    .into_iter()
    .all(|invalid| C08PreparableGraph::try_from_lowered(invalid).is_err())
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
    if let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) =
        &mut invalid.passes[capture_index].kind
    {
        span.scope = RuntimeVelloSpanScope::LayerSource;
    }
    invalid_plans.push(invalid);

    let mut invalid = plan.clone();
    if let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span))) =
        &mut invalid.passes[capture_index].kind
    {
        span.captured_before_outer_semantics = false;
    }
    invalid_plans.push(invalid);

    invalid_plans
        .into_iter()
        .all(|invalid| C08PreparableGraph::try_from_lowered(invalid).is_err())
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
        let preparable = C08PreparableGraph::try_from_lowered(lowered).ok()?;
        let actual_capture = preparable.execution.captures().first()?;
        let capture_pass = preparable
            .lowered
            .passes
            .iter()
            .find(|pass| pass.id == actual_capture.pass())?;
        let RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(actual_span))) =
            &capture_pass.kind
        else {
            return None;
        };
        let mut span = actual_span.clone();
        span.capture_transform = capture_transform;
        span.parent_to_surface = parent_to_surface;
        let target = actual_capture.target();
        let mut spatial = preparable
            .lowered
            .resources
            .iter()
            .find(|resource| resource.id == target)?
            .spatial;
        spatial.device_origin = (-3, -2);
        spatial.texel_origin = Point::new(-3.0 / raster_scale, -2.0 / raster_scale);
        spatial.raster_scale = raster_scale;
        let facts = executable_vello_capture_facts(
            capture_pass.id,
            target,
            &RuntimeVelloCapture::Span(span),
            spatial,
        )?;

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
            && facts.commands() == Some(&commands)
            && facts.antialiasing() == antialiasing
            && facts.target_extent() == spatial.device_extent
            && facts.raster_scale() == raster_scale;

        let encoded = super::encode::encode_vello_scene_with_initial_transform(
            facts.commands()?,
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

pub(crate) const VELLO_CAPTURE_TEXTURE_USAGES: wgpu::TextureUsages =
    wgpu::TextureUsages::STORAGE_BINDING
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
    composite_parameters: Option<CompositeParameterBytes>,
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
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &wgpu::Device,
    ) -> Result<Self> {
        capabilities.validate_supported_working_format(selected_working_format)?;
        if selected_working_format != lowered.working_format {
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
                (RuntimeResourceFormat::ClipCoverageRgba8Unorm, None)
                    if resource.role == RuntimeResourceRole::ClipCoverage
                        && matches!(resource.producer, RuntimeResourceProducer::Pass(_)) =>
                {
                    let descriptor = EffectTextureDescriptor::try_coverage(
                        extent,
                        VELLO_CAPTURE_TEXTURE_USAGES,
                    )?;
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
                        && resource.role != RuntimeResourceRole::ClipCoverage
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
                let composite_parameters = prepared_pass_composite_parameters(pass)?;
                if spatial_uniform.is_some() != pass.cache_keys.is_some() {
                    return Err(preparation_error(
                        "prepared pass spatial bytes and executable cache keys disagree",
                    ));
                }
                Ok(RuntimePassPreparationRequest {
                    runtime: pass.clone(),
                    spatial_uniform,
                    composite_parameters,
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

fn prepared_pass_composite_parameters(
    pass: &RuntimePass,
) -> Result<Option<CompositeParameterBytes>> {
    match &pass.kind {
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) => {
            let bytes = CompositeParameterBytes::try_from_runtime_layer(parameters)?;
            if bytes.as_bytes().len() != 112 {
                return Err(preparation_error(
                    "composite parameter serialization changed its exact WGSL byte length",
                ));
            }
            Ok(Some(bytes))
        }
        _ => Ok(None),
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransitionalPassSemantics {
    ClosedExecutable,
    LaterCycle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphPreparationIneligibility {
    OutsideClosedExecutableGraph,
}

impl GraphPreparationIneligibility {
    fn into_error(self) -> Error {
        match self {
            Self::OutsideClosedExecutableGraph => preparation_error(
                "a graph outside the closed executable subset cannot enter runtime preparation",
            ),
        }
    }
}

enum PrePreparationGraphClassification {
    ExactC08(C08PreparableGraph),
    LaterCycleTransitional(LoweredGraphPlan),
    Ineligible(GraphPreparationIneligibility),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutableGraphWorkingFormatRequest {
    ConfiguredPolicy(EffectQualityPolicy),
    #[cfg(test)]
    Exact(WorkingFormat),
}

impl ExecutableGraphWorkingFormatRequest {
    fn resolve(self, capabilities: &DeviceCapabilities) -> Result<WorkingFormat> {
        match self {
            Self::ConfiguredPolicy(policy) => capabilities.resolve_effect_working_format(policy),
            #[cfg(test)]
            Self::Exact(working_format) => {
                capabilities.validate_supported_working_format(working_format)?;
                Ok(working_format)
            }
        }
    }
}

#[must_use = "the closed graph dispatch result must select exactly one renderer route"]
pub(crate) enum ExecutableGraphDispatchEligibility {
    Exact(C08PreparableGraph),
    LaterCycleTransitional,
}

impl ExecutableGraphDispatchEligibility {
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
            PrePreparationGraphClassification::LaterCycleTransitional(_) => {
                Ok(Self::LaterCycleTransitional)
            }
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
                        Ok(Self::Exact(preparable))
                    }
                    PrePreparationGraphClassification::LaterCycleTransitional(_)
                    | PrePreparationGraphClassification::Ineligible(_) => Err(preparation_error(
                        "checked C08 dispatch lowering changed its closed eligibility result",
                    )),
                }
            }
        }
    }
}

impl PrePreparationGraphClassification {
    fn classify(lowered: LoweredGraphPlan) -> Self {
        let closed = match ClosedExecutableGraph::try_from_lowered(lowered) {
            Ok(closed) => closed,
            Err(lowered) => {
                let mut contains_later_cycle_semantics = false;
                for pass in &lowered.passes {
                    match transitional_pass_semantics(&pass.kind) {
                        Some(TransitionalPassSemantics::ClosedExecutable) => {}
                        Some(TransitionalPassSemantics::LaterCycle) => {
                            contains_later_cycle_semantics = true;
                        }
                        None => {
                            return Self::Ineligible(
                                GraphPreparationIneligibility::OutsideClosedExecutableGraph,
                            );
                        }
                    }
                }
                return if contains_later_cycle_semantics {
                    Self::LaterCycleTransitional(lowered)
                } else {
                    Self::Ineligible(GraphPreparationIneligibility::OutsideClosedExecutableGraph)
                };
            }
        };
        match C08PreparableGraph::try_from_closed(closed) {
            Ok(preparable) => Self::ExactC08(preparable),
            Err(closed) if closed.has_layer_composition() => {
                Self::LaterCycleTransitional((*closed).into_lowered())
            }
            Err(_) => Self::Ineligible(GraphPreparationIneligibility::OutsideClosedExecutableGraph),
        }
    }
}

fn transitional_pass_semantics(kind: &RuntimePassKind) -> Option<TransitionalPassSemantics> {
    match kind {
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            ..
        }
        | RuntimePassKind::CanonicalizeCapture
        | RuntimePassKind::Present => Some(TransitionalPassSemantics::ClosedExecutable),
        RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::Transparent,
            color,
        } if *color == Color::TRANSPARENT => Some(TransitionalPassSemantics::ClosedExecutable),
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
            Some(TransitionalPassSemantics::ClosedExecutable)
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
            Some(TransitionalPassSemantics::ClosedExecutable)
        }
        RuntimePassKind::VelloCapture(_) => None,
        RuntimePassKind::CopyBackdrop
        | RuntimePassKind::ColorFilter(Some(_))
        | RuntimePassKind::DropShadowColorize(Some(_)) => {
            Some(TransitionalPassSemantics::LaterCycle)
        }
        RuntimePassKind::BlurHorizontal(Some(blur)) if blur.axis == RuntimeBlurAxis::Horizontal => {
            Some(TransitionalPassSemantics::LaterCycle)
        }
        RuntimePassKind::BlurVertical(Some(blur)) if blur.axis == RuntimeBlurAxis::Vertical => {
            Some(TransitionalPassSemantics::LaterCycle)
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
                    TransitionalPassSemantics::ClosedExecutable
                }
                RuntimeCompositeKind::DropShadow => TransitionalPassSemantics::LaterCycle,
            })
        }
        RuntimePassKind::Composite(Some(_)) => None,
    }
}

enum GraphPreparationSource {
    C08(C08PreparableGraph),
    Transitional(LoweredGraphPlan),
}

impl GraphPreparationSource {
    fn into_parts(self) -> (LoweredGraphPlan, Option<C08ExecutionFacts>) {
        match self {
            Self::C08(preparable) => {
                let (lowered, execution) = preparable.into_parts();
                (lowered, Some(execution))
            }
            Self::Transitional(lowered) => (lowered, None),
        }
    }
}

pub(crate) struct C08VelloCaptureEncodingHandoff<'prepared> {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    work: &'prepared RuntimeVelloCapture,
    initial_transform: Transform,
    antialiasing: Antialiasing,
    target_extent: PhysicalSize,
    raster_scale: f64,
    texture: &'prepared wgpu::Texture,
    view: &'prepared wgpu::TextureView,
    session: Arc<()>,
}

struct C08VelloCaptureCompletionSeal;

/// Opaque proof that one exact capture finished inside the active C08 encoding
/// session. Only a successfully encoded internal Vello capture can seal it.
#[must_use = "a capture completion receipt must return to the owning C08 scheduler"]
pub(crate) struct C08VelloCaptureCompletionReceipt {
    pass: RuntimePassId,
    target: RuntimeResourceId,
    session: Arc<()>,
    _seal: C08VelloCaptureCompletionSeal,
}

impl C08VelloCaptureEncodingHandoff<'_> {
    pub(crate) const fn target(&self) -> RuntimeResourceId {
        self.target
    }

    const fn work(&self) -> &RuntimeVelloCapture {
        self.work
    }

    fn has_bounded_work(&self) -> bool {
        match self.work() {
            RuntimeVelloCapture::Span(span) => !span.commands.commands.is_empty(),
            RuntimeVelloCapture::ClipCoverage(coverage) => !coverage.elements.is_empty(),
        }
    }

    pub(crate) const fn initial_transform(&self) -> Transform {
        self.initial_transform
    }

    pub(crate) const fn antialiasing(&self) -> Antialiasing {
        self.antialiasing
    }

    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target_extent
    }

    pub(crate) const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }

    pub(crate) const fn texture(&self) -> &wgpu::Texture {
        self.texture
    }

    pub(crate) const fn view(&self) -> &wgpu::TextureView {
        self.view
    }

    fn complete_after_encoded_capture(
        self,
        proof: EncodedVelloCaptureProof,
    ) -> Result<C08VelloCaptureCompletionReceipt> {
        if !proof.proves_capture_contract(
            self.target_extent,
            wgpu::TextureFormat::Rgba8Unorm,
            VELLO_CAPTURE_TEXTURE_USAGES,
            self.antialiasing,
        ) {
            return Err(preparation_error(
                "encoded C08 Vello capture proof changed its exact raster target contract",
            ));
        }
        Ok(C08VelloCaptureCompletionReceipt {
            pass: self.pass,
            target: self.target,
            session: self.session,
            _seal: C08VelloCaptureCompletionSeal,
        })
    }
}

pub(crate) struct C08ExternalOutputView<'output> {
    view: &'output wgpu::TextureView,
    format: Format,
    extent: PhysicalSize,
}

impl<'output> C08ExternalOutputView<'output> {
    pub(crate) fn try_new(
        view: &'output wgpu::TextureView,
        format: Format,
        extent: PhysicalSize,
    ) -> Result<Self> {
        if extent.width() == 0 || extent.height() == 0 {
            return Err(preparation_error(
                "C08 external output view must have a positive exact extent",
            ));
        }
        Ok(Self {
            view,
            format,
            extent,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08ScheduledEncodingKind {
    ClearRoot,
    VelloCapture,
    CanonicalizeCapture,
    SpanSourceOver,
    Present,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum C08CustomSpineEncodingState {
    Ready,
    Encoding,
    Complete,
    AbortOnly,
}

pub(crate) struct C08CustomSpineEncodingSummary {
    pub(crate) encodes_custom_passes_in_order: bool,
    pub(crate) clears_full_root_once: bool,
    pub(crate) uses_exact_prepared_spatial_mapping: bool,
    pub(crate) presents_to_exact_external_output: bool,
    pub(crate) exposes_bounded_capture_handoff: bool,
    pub(crate) validates_checked_capture_completion: bool,
    pub(crate) completes_custom_passes_after_encoding: bool,
    pub(crate) parent_and_result_are_distinct: bool,
    pub(crate) copies_full_parent_before_bounded_source_render: bool,
    pub(crate) samples_only_source_with_fixed_premultiplied_blend: bool,
    pub(crate) preserves_signed_source_origin: bool,
    pub(crate) keeps_cache_update_provisional: bool,
    #[cfg(test)]
    pub(crate) capture_count: usize,
    #[cfg(test)]
    pub(crate) captures_share_one_command_encoder: bool,
    #[cfg(test)]
    pub(crate) captures_share_one_active_vello_scope: bool,
    #[cfg(test)]
    pub(crate) capture_observations: Vec<C08EncodedCaptureObservationForTest>,
}

impl C08CustomSpineEncodingSummary {
    fn proves_complete_submission(&self) -> bool {
        self.encodes_custom_passes_in_order
            && self.clears_full_root_once
            && self.uses_exact_prepared_spatial_mapping
            && self.presents_to_exact_external_output
            && self.exposes_bounded_capture_handoff
            && self.validates_checked_capture_completion
            && self.completes_custom_passes_after_encoding
            && self.parent_and_result_are_distinct
            && self.copies_full_parent_before_bounded_source_render
            && self.samples_only_source_with_fixed_premultiplied_blend
            && self.preserves_signed_source_origin
            && self.keeps_cache_update_provisional
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct C08EncodedCaptureObservationForTest {
    pub(crate) lowers_with_exact_initial_transform: bool,
    pub(crate) uses_transparent_base: bool,
    pub(crate) antialiasing: Antialiasing,
    pub(crate) target_extent: PhysicalSize,
    pub(crate) target_format: wgpu::TextureFormat,
    pub(crate) target_usage: wgpu::TextureUsages,
    pub(crate) target_and_view_are_exact: bool,
    encoder_identity: usize,
    scope_identity: usize,
}

struct C08EncodedCaptureResult {
    receipt: C08VelloCaptureCompletionReceipt,
    #[cfg(test)]
    observation: C08EncodedCaptureObservationForTest,
}

struct C08VelloCaptureEncodingContext<'encoding, 'device> {
    engine: &'device VelloEngineState,
    resources: &'device ResourceManager,
    queue: &'device wgpu::Queue,
    scope: &'encoding mut ActiveVelloEncodingScope<'device>,
    leases: &'encoding mut VelloResourceLeaseAggregate,
}

/// Owns the scope-clean capture leases until T5 gives them to the transaction
/// submission payload. Dropping this value aborts every capture lease.
#[must_use = "encoded C08 graph captures must remain pending until transaction resolution"]
pub(crate) struct C08PendingGraphEncoding {
    summary: C08CustomSpineEncodingSummary,
    resources: PendingVelloResourceCommit,
    session: Arc<()>,
}

#[cfg(test)]
impl C08PendingGraphEncoding {
    pub(crate) fn into_summary_and_resources(
        self,
    ) -> (C08CustomSpineEncodingSummary, PendingVelloResourceCommit) {
        (self.summary, self.resources)
    }
}

#[must_use = "prepared C08 frame state must commit only after graph transaction success"]
pub(crate) struct PendingC08PreparedFrameCommit {
    frame_scope: FrameResourceScope,
    pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

/// Sealed one-shot state proving that the prepared frame and provisional cache
/// can still complete without an accounting or cache-identity fault.
#[must_use = "accounting-ready C08 prepared state must be committed or aborted on drop"]
pub(crate) struct AccountingReadyC08PreparedFrameCommit {
    frame_scope: FrameResourceScope,
    pass_cache_update: ProvisionalDevicePassCacheUpdate,
}

impl PendingC08PreparedFrameCommit {
    pub(crate) fn into_accounting_ready(
        self,
        pass_cache: &DevicePassCache,
    ) -> Result<AccountingReadyC08PreparedFrameCommit> {
        self.pass_cache_update.ensure_commit_ready(pass_cache)?;
        self.frame_scope.ensure_commit_ready(&[])?;
        Ok(AccountingReadyC08PreparedFrameCommit {
            frame_scope: self.frame_scope,
            pass_cache_update: self.pass_cache_update,
        })
    }

    #[cfg(test)]
    pub(crate) fn resource_identities_for_test(&self) -> Vec<ResourceIdentity> {
        self.frame_scope.leased_resource_identities_for_test()
    }

    #[cfg(test)]
    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        self.frame_scope.poison_retained_byte_accounting_for_test()
    }
}

impl AccountingReadyC08PreparedFrameCommit {
    pub(crate) fn ensure_commit_ready(&self, pass_cache: &DevicePassCache) -> Result<()> {
        self.pass_cache_update.ensure_commit_ready(pass_cache)?;
        self.frame_scope.ensure_commit_ready(&[])
    }

    pub(crate) fn commit(self, pass_cache: &mut DevicePassCache) -> Result<FrameCleanup> {
        self.ensure_commit_ready(pass_cache)?;
        let frame_cleanup = self.frame_scope.finish_checked()?;
        self.pass_cache_update.commit(pass_cache)?;
        Ok(frame_cleanup)
    }
}

#[must_use = "C08 graph submission state must remain owned by one transaction payload"]
pub(crate) struct C08PreparedGraphSubmission {
    capture_resources: PendingVelloResourceCommit,
    prepared_frame: PendingC08PreparedFrameCommit,
}

impl C08PreparedGraphSubmission {
    pub(crate) fn into_parts(self) -> (PendingVelloResourceCommit, PendingC08PreparedFrameCommit) {
        (self.capture_resources, self.prepared_frame)
    }
}

#[derive(Clone)]
struct C08PreparedPassEncodingRequest {
    id: RuntimePassId,
    kind: RuntimePassKind,
    reads: Vec<RuntimeReadBinding>,
    result: RuntimeResultBinding,
    spatial_uniform: Option<PassSpatialUniformBytes>,
    cache_keys: Option<RuntimePassCacheKeys>,
}

impl From<&RuntimePassPreparationRequest> for C08PreparedPassEncodingRequest {
    fn from(request: &RuntimePassPreparationRequest) -> Self {
        Self {
            id: request.runtime.id,
            kind: request.runtime.kind.clone(),
            reads: request.runtime.reads.clone(),
            result: request.runtime.result,
            spatial_uniform: request.spatial_uniform.clone(),
            cache_keys: request.cache_keys.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct C08RenderRegion {
    viewport_x: f32,
    viewport_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    scissor_x: u32,
    scissor_y: u32,
    scissor_width: u32,
    scissor_height: u32,
    unclipped_x: f64,
    unclipped_y: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct C08PassEncodingFacts {
    full_target: bool,
    exact_spatial_uniform: bool,
    external_output_exact: bool,
    parent_and_result_distinct: bool,
    copied_full_parent_before_render: bool,
    sampled_only_source: bool,
    fixed_source_over_blend: bool,
    preserved_signed_source_origin: bool,
}

struct C08SampledRenderTarget<'target> {
    view: &'target wgpu::TextureView,
    extent: PhysicalSize,
    region: Option<C08RenderRegion>,
    load: wgpu::LoadOp<wgpu::Color>,
    label: &'static str,
}

impl C08RenderRegion {
    fn full(extent: PhysicalSize) -> Result<Self> {
        if extent.width() == 0 || extent.height() == 0 {
            return Err(preparation_error(
                "the C08 render region requires a positive target extent",
            ));
        }
        let viewport_width = extent.width() as f32;
        let viewport_height = extent.height() as f32;
        if !viewport_width.is_finite() || !viewport_height.is_finite() {
            return Err(preparation_error(
                "the C08 render extent cannot be represented by WGPU viewport coordinates",
            ));
        }
        Ok(Self {
            viewport_x: 0.0,
            viewport_y: 0.0,
            viewport_width,
            viewport_height,
            scissor_x: 0,
            scissor_y: 0,
            scissor_width: extent.width(),
            scissor_height: extent.height(),
            unclipped_x: 0.0,
            unclipped_y: 0.0,
        })
    }

    fn bounded_source(
        source: RuntimeSpatialDescriptor,
        destination: RuntimeSpatialDescriptor,
    ) -> Result<Option<Self>> {
        let source_width = f64::from(source.device_extent.width()) / source.raster_scale;
        let source_height = f64::from(source.device_extent.height()) / source.raster_scale;
        let unclipped_x =
            (source.texel_origin.x() - destination.texel_origin.x()) * destination.raster_scale;
        let unclipped_y =
            (source.texel_origin.y() - destination.texel_origin.y()) * destination.raster_scale;
        let unclipped_end_x = (source.texel_origin.x() + source_width
            - destination.texel_origin.x())
            * destination.raster_scale;
        let unclipped_end_y = (source.texel_origin.y() + source_height
            - destination.texel_origin.y())
            * destination.raster_scale;
        if [unclipped_x, unclipped_y, unclipped_end_x, unclipped_end_y]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(preparation_error(
                "the C08 signed bounded render mapping is non-finite",
            ));
        }

        let destination_width = f64::from(destination.device_extent.width());
        let destination_height = f64::from(destination.device_extent.height());
        let clipped_x = unclipped_x.max(0.0).min(destination_width);
        let clipped_y = unclipped_y.max(0.0).min(destination_height);
        let clipped_end_x = unclipped_end_x.max(0.0).min(destination_width);
        let clipped_end_y = unclipped_end_y.max(0.0).min(destination_height);
        if clipped_end_x <= clipped_x || clipped_end_y <= clipped_y {
            return Ok(None);
        }

        let scissor_x = clipped_x.floor() as u32;
        let scissor_y = clipped_y.floor() as u32;
        let scissor_end_x = clipped_end_x.ceil() as u32;
        let scissor_end_y = clipped_end_y.ceil() as u32;
        let viewport_x = clipped_x as f32;
        let viewport_y = clipped_y as f32;
        let viewport_width = (clipped_end_x - clipped_x) as f32;
        let viewport_height = (clipped_end_y - clipped_y) as f32;
        if !viewport_x.is_finite()
            || !viewport_y.is_finite()
            || !viewport_width.is_finite()
            || !viewport_height.is_finite()
            || viewport_width <= 0.0
            || viewport_height <= 0.0
            || scissor_end_x <= scissor_x
            || scissor_end_y <= scissor_y
        {
            return Err(preparation_error(
                "the C08 bounded viewport or scissor cannot represent its signed mapping",
            ));
        }
        Ok(Some(Self {
            viewport_x,
            viewport_y,
            viewport_width,
            viewport_height,
            scissor_x,
            scissor_y,
            scissor_width: scissor_end_x - scissor_x,
            scissor_height: scissor_end_y - scissor_y,
            unclipped_x,
            unclipped_y,
        }))
    }
}

fn exact_c08_read(
    request: &C08PreparedPassEncodingRequest,
    role: RuntimeReadRole,
) -> Result<&RuntimeReadBinding> {
    let mut matching = request.reads.iter().filter(|read| read.role == role);
    let read = matching
        .next()
        .ok_or_else(|| preparation_error("the C08 prepared source binding is missing"))?;
    if matching.next().is_some() {
        return Err(preparation_error(
            "the C08 prepared source binding is duplicated",
        ));
    }
    Ok(read)
}

fn c08_scheduled_encoding_order_is_exact(
    scheduled: &[C08ScheduledEncodingKind],
    capture_count: usize,
) -> bool {
    if capture_count == 0
        || scheduled.len() != capture_count.saturating_mul(3).saturating_add(2)
        || scheduled.first() != Some(&C08ScheduledEncodingKind::ClearRoot)
        || scheduled.last() != Some(&C08ScheduledEncodingKind::Present)
    {
        return false;
    }
    scheduled[1..scheduled.len() - 1]
        .chunks_exact(3)
        .all(|chunk| {
            chunk
                == [
                    C08ScheduledEncodingKind::VelloCapture,
                    C08ScheduledEncodingKind::CanonicalizeCapture,
                    C08ScheduledEncodingKind::SpanSourceOver,
                ]
        })
}

fn c08_spatial_uniform_preserves_source_origin(
    bytes: &PassSpatialUniformBytes,
    source: RuntimeSpatialDescriptor,
) -> bool {
    let encoded_x = f32::from_le_bytes(bytes.as_bytes()[0..4].try_into().unwrap_or([0; 4]));
    let encoded_y = f32::from_le_bytes(bytes.as_bytes()[4..8].try_into().unwrap_or([0; 4]));
    encoded_x == source.texel_origin.x() as f32 && encoded_y == source.texel_origin.y() as f32
}

fn close_f64(left: f64, right: f64) -> bool {
    let tolerance = f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0;
    (left - right).abs() <= tolerance
}

/// One allocation-backed, generation-bound C07 handoff. Its lifetime prevents
/// the ready device bundle from transitioning while C08 owns its frame scope.
pub(crate) struct PreparedGraph<'device> {
    plan: RuntimeGraphPreparationPlan,
    c08_execution: Option<C08ExecutionFacts>,
    resource_bindings: BTreeMap<RuntimeResourceId, PreparedResourceBinding>,
    kernel_bindings: BTreeMap<GaussianKernelKey, PreparedKernelBinding>,
    pass_cache_update: Option<ProvisionalDevicePassCacheUpdate>,
    frame_scope: Option<FrameResourceScope>,
    next_pass: usize,
    c08_encoding_state: Option<C08CustomSpineEncodingState>,
    c08_completed_session: Option<Arc<()>>,
    #[cfg(test)]
    fail_capture_encoding_after_for_test: Option<usize>,
    #[cfg(test)]
    fail_scope_resolution_for_test: bool,
    #[cfg(test)]
    acquired_capture_lease_count_for_test: usize,
    device: &'device wgpu::Device,
    queue: &'device wgpu::Queue,
    vello_engine: Option<&'device VelloEngineState>,
    resources: &'device ResourceManager,
    pass_cache: &'device DevicePassCache,
    _ready_device: PhantomData<&'device ResourceManager>,
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
            PrePreparationGraphClassification::LaterCycleTransitional(lowered) => {
                let selected_working_format = capabilities.resolve_effect_working_format(policy)?;
                Self::try_prepare_inner(
                    GraphPreparationSource::Transitional(lowered),
                    selected_working_format,
                    capabilities,
                    device,
                    queue,
                    resources,
                    (pass_cache_phase.0, false),
                )
            }
            PrePreparationGraphClassification::Ineligible(ineligibility) => {
                Err(ineligibility.into_error())
            }
        }
    }

    fn try_prepare_inner(
        source: GraphPreparationSource,
        selected_working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
        device: &'device wgpu::Device,
        queue: &'device wgpu::Queue,
        resources: &'device ResourceManager,
        pass_cache_phase: (&'device DevicePassCache, bool),
    ) -> Result<Self> {
        let (pass_cache, realize_checked_passes) = pass_cache_phase;
        let (lowered, c08_execution) = source.into_parts();
        let plan = RuntimeGraphPreparationPlan::try_derive(
            lowered,
            selected_working_format,
            capabilities,
            device,
        )?;
        resources.preflight_graph_acquisitions(&plan.allocation_preflights)?;

        let mut frame_scope = resources.begin_frame()?;
        frame_scope.abort_provisional_on_drop();
        if c08_execution.is_some() {
            frame_scope.discard_on_drop();
        }
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

        let pass_cache_update = if realize_checked_passes && c08_execution.is_some() {
            let mut update = pass_cache.provisional_update();
            for keys in plan
                .passes
                .iter()
                .filter_map(|request| request.cache_keys.as_ref())
            {
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
            }
            Some(update)
        } else if realize_checked_passes {
            let mut update = pass_cache.provisional_update();
            let mut realized_composite = false;
            for request in &plan.passes {
                let RuntimePassKind::Composite(Some(RuntimeComposite {
                    kind: RuntimeCompositeKind::Layer { .. },
                    ..
                })) = &request.runtime.kind
                else {
                    continue;
                };
                let keys = request.cache_keys.as_ref().ok_or_else(|| {
                    preparation_error("C09 composite preparation lost its exact cache keys")
                })?;
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
                realized_composite = true;
            }
            realized_composite.then_some(update)
        } else {
            None
        };
        let c08_encoding_state = c08_execution
            .as_ref()
            .map(|_| C08CustomSpineEncodingState::Ready);

        Ok(Self {
            plan,
            c08_execution,
            resource_bindings,
            kernel_bindings,
            pass_cache_update,
            frame_scope: Some(frame_scope),
            next_pass: 0,
            c08_encoding_state,
            c08_completed_session: None,
            #[cfg(test)]
            fail_capture_encoding_after_for_test: None,
            #[cfg(test)]
            fail_scope_resolution_for_test: false,
            #[cfg(test)]
            acquired_capture_lease_count_for_test: 0,
            device,
            queue,
            vello_engine: None,
            resources,
            pass_cache,
            _ready_device: PhantomData,
        })
    }

    pub(crate) fn with_vello_engine(mut self, engine: &'device VelloEngineState) -> Self {
        self.vello_engine = Some(engine);
        self
    }

    #[cfg(test)]
    pub(crate) fn fail_capture_encoding_for_test(&mut self) {
        self.fail_capture_encoding_after_for_test = Some(0);
    }

    #[cfg(test)]
    pub(crate) fn fail_capture_encoding_after_for_test(&mut self, successful_capture_count: usize) {
        self.fail_capture_encoding_after_for_test = Some(successful_capture_count);
    }

    #[cfg(test)]
    pub(crate) fn fail_scope_resolution_for_test(&mut self) {
        self.fail_scope_resolution_for_test = true;
    }

    #[cfg(test)]
    pub(crate) const fn acquired_capture_lease_count_for_test(&self) -> usize {
        self.acquired_capture_lease_count_for_test
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

    pub(crate) async fn encode_c08_custom_spine(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: C08ExternalOutputView<'_>,
    ) -> Result<C08PendingGraphEncoding> {
        match self.c08_encoding_state {
            Some(C08CustomSpineEncodingState::Ready) => {}
            Some(
                C08CustomSpineEncodingState::Encoding
                | C08CustomSpineEncodingState::Complete
                | C08CustomSpineEncodingState::AbortOnly,
            ) => {
                return Err(preparation_error(
                    "the C08 custom encoding is one-shot; discard this prepared graph and its encoder",
                ));
            }
            None => {
                return Err(preparation_error(
                    "the C08 custom scheduler requires validated execution facts",
                ));
            }
        }
        let execution = self.c08_execution.as_ref().ok_or_else(|| {
            preparation_error("the C08 custom scheduler requires validated execution facts")
        })?;
        if execution.working_format() != self.plan.working_format
            || execution.output_format() != self.plan.output_format
            || output.format != self.plan.output_format
            || output.extent != self.output_extent()?
        {
            return Err(preparation_error(
                "the C08 external output differs from the exact prepared format or extent",
            ));
        }
        if self.pass_cache_update.is_none() {
            return Err(preparation_error(
                "the C08 custom scheduler requires transaction-provisional pass objects",
            ));
        }
        let expected_capture_count = execution.captures().len();
        if expected_capture_count == 0 || self.next_pass != 0 {
            return Err(preparation_error(
                "the C08 custom scheduler requires one unstarted capture spine",
            ));
        }
        let engine = self.vello_engine.ok_or_else(|| {
            preparation_error("the C08 capture scheduler has no ready internal Vello engine")
        })?;
        let resources = self.resources;
        let queue = self.queue;

        let session = Arc::new(());
        let mut scope = ActiveVelloEncodingScope::begin(self.device);
        let mut leases = VelloResourceLeaseAggregate::new();
        self.c08_encoding_state = Some(C08CustomSpineEncodingState::Encoding);
        let result = {
            let mut capture_encoding = C08VelloCaptureEncodingContext {
                engine,
                resources,
                queue,
                scope: &mut scope,
                leases: &mut leases,
            };
            self.encode_c08_custom_spine_once(
                encoder,
                &output,
                expected_capture_count,
                &session,
                &mut capture_encoding,
            )
        };
        let summary = match result {
            Ok(summary) => summary,
            Err(encoding_error) => {
                let _ = leases.abort();
                let scope_result = scope.finish().await;
                self.c08_encoding_state = Some(C08CustomSpineEncodingState::AbortOnly);
                return match scope_result {
                    Ok(()) => Err(encoding_error),
                    Err(scope_error) => Err(scope_error),
                };
            }
        };
        #[cfg(test)]
        if self.fail_scope_resolution_for_test {
            scope.inject_validation_error_for_test();
        }
        let leases = match scope.finish_with_leases(leases).await {
            Ok(leases) => leases,
            Err(failure) => {
                self.c08_encoding_state = Some(C08CustomSpineEncodingState::AbortOnly);
                return Err(failure.into_error_and_aborted_resources().0);
            }
        };
        self.c08_encoding_state = Some(C08CustomSpineEncodingState::Complete);
        self.c08_completed_session = Some(Arc::clone(&session));
        Ok(C08PendingGraphEncoding {
            summary,
            resources: PendingVelloResourceCommit::from_aggregate(leases),
            session,
        })
    }

    fn encode_c08_custom_spine_once(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        output: &C08ExternalOutputView<'_>,
        expected_capture_count: usize,
        session: &Arc<()>,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
    ) -> Result<C08CustomSpineEncodingSummary> {
        let mut scheduled = Vec::with_capacity(self.plan.passes.len());
        let mut capture_count = 0_usize;
        let mut validated_capture_receipts = 0_usize;
        let mut bounded_capture_handoffs = true;
        let mut custom_encoded = 0_usize;
        let mut custom_completed = 0_usize;
        let mut clear_count = 0_usize;
        let mut clears_full_root = true;
        let mut exact_spatial = true;
        let mut exact_external_output = false;
        let mut source_over_count = 0_usize;
        let mut parent_and_result_are_distinct = true;
        let mut full_copy_before_bounded_render = true;
        let mut samples_source_with_fixed_blend = true;
        let mut preserves_signed_origin = true;
        #[cfg(test)]
        let mut capture_observations = Vec::with_capacity(expected_capture_count);

        while let Some(request) = self
            .plan
            .passes
            .get(self.next_pass)
            .map(C08PreparedPassEncodingRequest::from)
        {
            let pass = request.id;
            match &request.kind {
                RuntimePassKind::ClearRoot { .. } => {
                    let facts = self.encode_c08_clear_root(encoder, &request)?;
                    custom_encoded = custom_encoded.saturating_add(1);
                    clear_count = clear_count.saturating_add(1);
                    clears_full_root &= facts.full_target;
                    scheduled.push(C08ScheduledEncodingKind::ClearRoot);
                    self.complete_c08_custom_pass(pass)?;
                    custom_completed = custom_completed.saturating_add(1);
                }
                RuntimePassKind::VelloCapture(Some(_)) => {
                    #[cfg(test)]
                    if self.fail_capture_encoding_after_for_test == Some(capture_count) {
                        return Err(preparation_error(
                            "injected C08 Vello capture encoding failure",
                        ));
                    }
                    let handoff = self.c08_vello_capture_handoff(&request, session)?;
                    let target = handoff.target();
                    bounded_capture_handoffs &= handoff.has_bounded_work()
                        && handoff.target_extent().width() > 0
                        && handoff.target_extent().height() > 0
                        && handoff.texture().width() == handoff.target_extent().width()
                        && handoff.texture().height() == handoff.target_extent().height()
                        && handoff.texture().depth_or_array_layers() == 1
                        && handoff.texture().mip_level_count() == 1
                        && handoff.texture().sample_count() == 1
                        && handoff.raster_scale().is_finite()
                        && handoff.raster_scale() > 0.0
                        && handoff
                            .initial_transform()
                            .as_array()
                            .iter()
                            .all(|value| value.is_finite());
                    scheduled.push(C08ScheduledEncodingKind::VelloCapture);
                    let encoded =
                        Self::encode_c08_vello_capture(handoff, encoder, capture_encoding)?;
                    #[cfg(test)]
                    capture_observations.push(encoded.observation);
                    self.complete_c08_capture(pass, target, session, encoded.receipt)?;
                    capture_count = capture_count.saturating_add(1);
                    #[cfg(test)]
                    {
                        self.acquired_capture_lease_count_for_test =
                            self.acquired_capture_lease_count_for_test.saturating_add(1);
                    }
                    validated_capture_receipts = validated_capture_receipts.saturating_add(1);
                }
                RuntimePassKind::CanonicalizeCapture => {
                    let facts = self.encode_c08_canonicalize(encoder, &request)?;
                    custom_encoded = custom_encoded.saturating_add(1);
                    exact_spatial &= facts.exact_spatial_uniform;
                    scheduled.push(C08ScheduledEncodingKind::CanonicalizeCapture);
                    self.complete_c08_custom_pass(pass)?;
                    custom_completed = custom_completed.saturating_add(1);
                }
                RuntimePassKind::Composite(Some(composite))
                    if matches!(composite.kind, RuntimeCompositeKind::SpanSourceOver) =>
                {
                    let facts = self.encode_c08_span_source_over(encoder, &request)?;
                    custom_encoded = custom_encoded.saturating_add(1);
                    source_over_count = source_over_count.saturating_add(1);
                    exact_spatial &= facts.exact_spatial_uniform;
                    parent_and_result_are_distinct &= facts.parent_and_result_distinct;
                    full_copy_before_bounded_render &= facts.copied_full_parent_before_render;
                    samples_source_with_fixed_blend &=
                        facts.sampled_only_source && facts.fixed_source_over_blend;
                    preserves_signed_origin &= facts.preserved_signed_source_origin;
                    scheduled.push(C08ScheduledEncodingKind::SpanSourceOver);
                    self.complete_c08_custom_pass(pass)?;
                    custom_completed = custom_completed.saturating_add(1);
                }
                RuntimePassKind::Present => {
                    let facts = self.encode_c08_present(encoder, &request, output)?;
                    custom_encoded = custom_encoded.saturating_add(1);
                    exact_spatial &= facts.exact_spatial_uniform;
                    exact_external_output |= facts.external_output_exact;
                    scheduled.push(C08ScheduledEncodingKind::Present);
                    self.complete_c08_custom_pass(pass)?;
                    custom_completed = custom_completed.saturating_add(1);
                }
                RuntimePassKind::VelloCapture(None)
                | RuntimePassKind::CopyBackdrop
                | RuntimePassKind::ColorFilter(_)
                | RuntimePassKind::BlurHorizontal(_)
                | RuntimePassKind::BlurVertical(_)
                | RuntimePassKind::DropShadowColorize(_)
                | RuntimePassKind::Composite(_) => {
                    return Err(preparation_error(
                        "a non-C08 pass reached the custom graph spine scheduler",
                    ));
                }
            }
        }

        let encodes_custom_passes_in_order =
            c08_scheduled_encoding_order_is_exact(&scheduled, expected_capture_count);
        #[cfg(test)]
        let captures_share_one_command_encoder =
            capture_observations.first().is_some_and(|first| {
                capture_observations.len() == expected_capture_count
                    && capture_observations
                        .iter()
                        .all(|capture| capture.encoder_identity == first.encoder_identity)
            });
        #[cfg(test)]
        let captures_share_one_active_vello_scope =
            capture_observations.first().is_some_and(|first| {
                capture_observations.len() == expected_capture_count
                    && capture_observations
                        .iter()
                        .all(|capture| capture.scope_identity == first.scope_identity)
            });
        Ok(C08CustomSpineEncodingSummary {
            encodes_custom_passes_in_order,
            clears_full_root_once: clear_count == 1 && clears_full_root,
            uses_exact_prepared_spatial_mapping: exact_spatial,
            presents_to_exact_external_output: exact_external_output,
            exposes_bounded_capture_handoff: expected_capture_count > 0
                && capture_count == expected_capture_count
                && bounded_capture_handoffs,
            validates_checked_capture_completion: validated_capture_receipts
                == expected_capture_count,
            completes_custom_passes_after_encoding: custom_encoded > 0
                && custom_completed == custom_encoded,
            parent_and_result_are_distinct: source_over_count > 0 && parent_and_result_are_distinct,
            copies_full_parent_before_bounded_source_render: source_over_count > 0
                && full_copy_before_bounded_render,
            samples_only_source_with_fixed_premultiplied_blend: source_over_count > 0
                && samples_source_with_fixed_blend,
            preserves_signed_source_origin: source_over_count > 0 && preserves_signed_origin,
            keeps_cache_update_provisional: self.pass_cache_update.is_some(),
            #[cfg(test)]
            capture_count,
            #[cfg(test)]
            captures_share_one_command_encoder,
            #[cfg(test)]
            captures_share_one_active_vello_scope,
            #[cfg(test)]
            capture_observations,
        })
    }

    fn encode_c08_vello_capture(
        handoff: C08VelloCaptureEncodingHandoff<'_>,
        encoder: &mut wgpu::CommandEncoder,
        capture_encoding: &mut C08VelloCaptureEncodingContext<'_, '_>,
    ) -> Result<C08EncodedCaptureResult> {
        let target_extent = handoff.target_extent();
        let antialiasing = handoff.antialiasing();
        if target_extent.width() == 0
            || target_extent.height() == 0
            || handoff.texture().width() != target_extent.width()
            || handoff.texture().height() != target_extent.height()
            || handoff.texture().depth_or_array_layers() != 1
            || handoff.texture().mip_level_count() != 1
            || handoff.texture().sample_count() != 1
            || handoff.texture().dimension() != wgpu::TextureDimension::D2
            || handoff.texture().format() != wgpu::TextureFormat::Rgba8Unorm
            || handoff.texture().usage() != VELLO_CAPTURE_TEXTURE_USAGES
        {
            return Err(preparation_error(
                "the C08 Vello capture target changed its exact RGBA8 storage contract",
            ));
        }
        let initial_transform = handoff.initial_transform();
        let scene = match handoff.work() {
            RuntimeVelloCapture::Span(span) => {
                encode_vello_scene_with_initial_transform(&span.commands, initial_transform)?
            }
            RuntimeVelloCapture::ClipCoverage(coverage) => {
                let elements = coverage
                    .elements
                    .iter()
                    .map(|element| (element.clip.clone(), element.transform))
                    .collect::<Vec<_>>();
                encode_vello_clip_coverage_scene(&elements, initial_transform, target_extent)?
            }
        };
        #[cfg(test)]
        let lowers_with_exact_initial_transform = match handoff.work() {
            RuntimeVelloCapture::Span(_) => scene
                .observation_for_test()
                .first_glyph_run_for_test()
                .is_some_and(|run| {
                    run.transform_components_for_test()
                        .iter()
                        .zip(initial_transform.as_array())
                        .all(|(actual, expected)| (*actual - expected as f32).abs() <= 1.0e-5)
                }),
            RuntimeVelloCapture::ClipCoverage(_) => true,
        };
        let prepared = scene.prepare_raster(vello_capture_raster_parameters(
            target_extent,
            antialiasing,
        )?)?;
        #[cfg(test)]
        let encoder_identity = std::ptr::from_mut(&mut *encoder) as usize;
        #[cfg(test)]
        let scope_identity = std::ptr::from_ref(&*capture_encoding.scope) as usize;
        #[cfg(test)]
        let target_view_identity = std::ptr::from_ref(handoff.view()) as usize;
        let encoded = {
            let mut encoding = TransactionEncodingState::new_reusable_graph_capture(
                capture_encoding.scope,
                capture_encoding.queue,
                encoder,
                handoff.view(),
                TransactionTargetIntent::new(
                    target_extent,
                    wgpu::TextureFormat::Rgba8Unorm,
                    VELLO_CAPTURE_TEXTURE_USAGES,
                ),
            );
            prepared.encode_capture_into(
                capture_encoding.engine,
                capture_encoding.resources,
                &mut encoding,
            )
        };
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(failure) => return Err(failure.into_error_and_aborted_resources().0),
        };
        let (lease, proof) = encoded.into_resources_and_proof();
        #[cfg(test)]
        let observation = C08EncodedCaptureObservationForTest {
            lowers_with_exact_initial_transform,
            uses_transparent_base: proof.transparent_base_for_test(),
            antialiasing: proof.antialiasing_for_test(),
            target_extent: proof.target_extent_for_test(),
            target_format: proof.target_format_for_test(),
            target_usage: proof.target_usage_for_test(),
            target_and_view_are_exact: handoff.texture().format() == proof.target_format_for_test()
                && handoff.texture().width() == proof.target_extent_for_test().width()
                && handoff.texture().height() == proof.target_extent_for_test().height()
                && proof.target_view_identity_for_test() == target_view_identity,
            encoder_identity,
            scope_identity,
        };
        let receipt = match handoff.complete_after_encoded_capture(proof) {
            Ok(receipt) => receipt,
            Err(error) => {
                let _ = lease.abort();
                return Err(error);
            }
        };
        capture_encoding.leases.push(lease);
        Ok(C08EncodedCaptureResult {
            receipt,
            #[cfg(test)]
            observation,
        })
    }

    fn resource_request(&self, resource: RuntimeResourceId) -> Result<&RuntimeResourceRequest> {
        self.plan
            .resources
            .iter()
            .find(|request| request.runtime.id == resource)
            .map(|request| &request.runtime)
            .ok_or_else(|| preparation_error("the prepared C08 resource request is missing"))
    }

    fn validate_texture_binding(
        &self,
        binding: &PreparedTextureBinding<'_>,
        resource: RuntimeResourceId,
    ) -> Result<RuntimeSpatialDescriptor> {
        let request = self.resource_request(resource)?;
        let texture = binding.texture();
        let expected_format = match request.format {
            RuntimeResourceFormat::VelloCaptureRgba8Unorm
            | RuntimeResourceFormat::ClipCoverageRgba8Unorm
            | RuntimeResourceFormat::ResolvedMaskRgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            RuntimeResourceFormat::Working(format) => format.texture_format(),
        };
        if binding.runtime_resource() != resource
            || texture.width() != request.spatial.device_extent.width()
            || texture.height() != request.spatial.device_extent.height()
            || texture.depth_or_array_layers() != 1
            || texture.mip_level_count() != 1
            || texture.sample_count() != 1
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.format() != expected_format
        {
            return Err(preparation_error(
                "the C08 prepared texture differs from its exact runtime binding",
            ));
        }
        Ok(request.spatial)
    }

    fn c08_pass_objects<'prepared>(
        &'prepared self,
        keys: &RuntimePassCacheKeys,
    ) -> Result<ProvisionalC08PassObjects<'prepared>> {
        self.pass_cache_update
            .as_ref()
            .ok_or_else(|| preparation_error("C08 provisional pass objects are unavailable"))?
            .encoding_objects(
                self.pass_cache,
                keys.samplers(),
                keys.layout(),
                keys.shader(),
                keys.pipeline(),
            )
    }

    fn create_c08_spatial_uniform_buffer(&self, bytes: &PassSpatialUniformBytes) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist C08 pass spatial uniform"),
            size: bytes.as_bytes().len() as u64,
            usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes.as_bytes());
        buffer
    }

    fn c08_vello_capture_handoff<'prepared>(
        &'prepared self,
        request: &C08PreparedPassEncodingRequest,
        session: &Arc<()>,
    ) -> Result<C08VelloCaptureEncodingHandoff<'prepared>> {
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 Vello capture has no exact prepared target",
            ));
        };
        let capture = self
            .c08_execution
            .as_ref()
            .and_then(|execution| {
                execution
                    .captures()
                    .iter()
                    .find(|capture| capture.pass() == request.id && capture.target() == target)
            })
            .ok_or_else(|| preparation_error("the bounded C08 capture handoff is missing"))?;
        let binding = self.texture_binding_for_pass(request.id, target)?;
        let spatial = self.validate_texture_binding(&binding, target)?;
        if spatial.device_extent != capture.target_extent()
            || spatial.texel_origin != capture.texel_origin()
            || spatial.raster_scale != capture.raster_scale()
        {
            return Err(preparation_error(
                "the bounded C08 capture target changed after preparation",
            ));
        }
        Ok(C08VelloCaptureEncodingHandoff {
            pass: request.id,
            target,
            work: capture.work(),
            initial_transform: capture.initial_transform(),
            antialiasing: capture.antialiasing(),
            target_extent: capture.target_extent(),
            raster_scale: capture.raster_scale(),
            texture: binding.texture(),
            view: binding.view(),
            session: Arc::clone(session),
        })
    }

    fn encode_c08_clear_root(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let RuntimePassKind::ClearRoot {
            initialization: RuntimeInitialization::SurfaceBaseColor,
            color,
        } = &request.kind
        else {
            return Err(preparation_error(
                "the C08 root clear changed its initialization contract",
            ));
        };
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error("the C08 root clear has no target"));
        };
        if target != self.plan.root_working_image
            || !request.reads.is_empty()
            || request.spatial_uniform.is_some()
            || request.cache_keys.is_some()
        {
            return Err(preparation_error(
                "the C08 root clear has non-root or sampled bindings",
            ));
        }
        let binding = self.texture_binding_for_pass(request.id, target)?;
        let spatial = self.validate_texture_binding(&binding, target)?;
        let alpha = f64::from(color.a());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Surgeist C08 full root clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: binding.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(color.r()) * alpha,
                            g: f64::from(color.g()) * alpha,
                            b: f64::from(color.b()) * alpha,
                            a: alpha,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
        }
        Ok(C08PassEncodingFacts {
            full_target: spatial.device_extent == self.output_extent()?,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_canonicalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let source = exact_c08_read(request, RuntimeReadRole::CaptureSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 canonicalization pass has no prepared result",
            ));
        };
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource)?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        if source_spatial != target_spatial
            || self.resource_request(source.resource)?.format
                != RuntimeResourceFormat::VelloCaptureRgba8Unorm
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
        {
            return Err(preparation_error(
                "C08 canonicalization changed its exact capture-to-working binding",
            ));
        }
        let region = C08RenderRegion::full(target_spatial.device_extent)?;
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: target_binding.view(),
                extent: target_spatial.device_extent,
                region: Some(region),
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                label: "Surgeist C08 canonicalize capture",
            },
        )?;
        Ok(C08PassEncodingFacts {
            full_target: true,
            exact_spatial_uniform: true,
            fixed_source_over_blend: fixed_blend,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_span_source_over(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
    ) -> Result<C08PassEncodingFacts> {
        let parent = exact_c08_read(request, RuntimeReadRole::CompositeParent)?;
        let source = exact_c08_read(request, RuntimeReadRole::CompositeSource)?;
        let RuntimeResultBinding::Resource(target) = request.result else {
            return Err(preparation_error(
                "the C08 source-over pass has no prepared result",
            ));
        };
        let parent_binding = self.texture_binding_for_pass(request.id, parent.resource)?;
        let parent_spatial = self.validate_texture_binding(&parent_binding, parent.resource)?;
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let source_spatial = self.validate_texture_binding(&source_binding, source.resource)?;
        let target_binding = self.texture_binding_for_pass(request.id, target)?;
        let target_spatial = self.validate_texture_binding(&target_binding, target)?;
        let parent_and_result_distinct = parent.resource != target
            && parent_binding.allocation_resource() != target_binding.allocation_resource();
        if !parent_and_result_distinct
            || parent_spatial != target_spatial
            || parent_binding.texture().format() != target_binding.texture().format()
            || self.resource_request(parent.resource)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(source.resource)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
            || self.resource_request(target)?.format
                != RuntimeResourceFormat::Working(self.plan.working_format)
        {
            return Err(preparation_error(
                "C08 source-over parent, source, and distinct result bindings are inconsistent",
            ));
        }

        let copy_extent = wgpu::Extent3d {
            width: parent_spatial.device_extent.width(),
            height: parent_spatial.device_extent.height(),
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: parent_binding.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: target_binding.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            copy_extent,
        );

        let region = C08RenderRegion::bounded_source(source_spatial, target_spatial)?;
        let preserved_signed_source_origin =
            request.spatial_uniform.as_ref().is_some_and(|bytes| {
                c08_spatial_uniform_preserves_source_origin(bytes, source_spatial)
                    && region.is_none_or(|region| {
                        let expected_x = (source_spatial.texel_origin.x()
                            - target_spatial.texel_origin.x())
                            * target_spatial.raster_scale;
                        let expected_y = (source_spatial.texel_origin.y()
                            - target_spatial.texel_origin.y())
                            * target_spatial.raster_scale;
                        close_f64(region.unclipped_x, expected_x)
                            && close_f64(region.unclipped_y, expected_y)
                    })
            });
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: target_binding.view(),
                extent: target_spatial.device_extent,
                region,
                load: wgpu::LoadOp::Load,
                label: "Surgeist C08 bounded span source-over",
            },
        )?;
        Ok(C08PassEncodingFacts {
            exact_spatial_uniform: true,
            parent_and_result_distinct,
            copied_full_parent_before_render: copy_extent.width == target_binding.texture().width()
                && copy_extent.height == target_binding.texture().height(),
            sampled_only_source: request.reads.len() == 2,
            fixed_source_over_blend: fixed_blend,
            preserved_signed_source_origin,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        output: &C08ExternalOutputView<'_>,
    ) -> Result<C08PassEncodingFacts> {
        let source = exact_c08_read(request, RuntimeReadRole::FinalWorkingImage)?;
        if request.result != RuntimeResultBinding::Output(output.format)
            || output.format != self.plan.output_format
            || output.extent != self.output_extent()?
        {
            return Err(preparation_error(
                "the C08 present pass changed its external output binding",
            ));
        }
        let source_binding = self.texture_binding_for_pass(request.id, source.resource)?;
        let _ = self.validate_texture_binding(&source_binding, source.resource)?;
        let region = C08RenderRegion::full(output.extent)?;
        let fixed_blend = self.encode_c08_sampled_render_pass(
            encoder,
            request,
            source,
            C08SampledRenderTarget {
                view: output.view,
                extent: output.extent,
                region: Some(region),
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                label: "Surgeist C08 present external output",
            },
        )?;
        Ok(C08PassEncodingFacts {
            full_target: true,
            exact_spatial_uniform: true,
            external_output_exact: true,
            fixed_source_over_blend: fixed_blend,
            ..C08PassEncodingFacts::default()
        })
    }

    fn encode_c08_sampled_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &C08PreparedPassEncodingRequest,
        source: &RuntimeReadBinding,
        target: C08SampledRenderTarget<'_>,
    ) -> Result<bool> {
        let spatial = request.spatial_uniform.as_ref().ok_or_else(|| {
            preparation_error("the C08 sampled pass has no exact prepared spatial uniform")
        })?;
        let keys = request.cache_keys.as_ref().ok_or_else(|| {
            preparation_error("the C08 sampled pass has no provisional cache keys")
        })?;
        if keys.samplers() != [source.sampler_key()] {
            return Err(preparation_error(
                "the C08 sampled pass changed its exact source sampler",
            ));
        }
        let source_binding = self.texture_binding_for_pass(request.id, source.resource())?;
        let _ = self.validate_texture_binding(&source_binding, source.resource())?;
        let objects = self.c08_pass_objects(keys)?;
        objects.require_encoding_ready()?;
        let uniform = self.create_c08_spatial_uniform_buffer(spatial);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surgeist C08 sampled pass bindings"),
            layout: objects.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_binding.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(objects.sampler()?),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        if let Some(region) = target.region {
            if region.scissor_x.saturating_add(region.scissor_width) > target.extent.width()
                || region.scissor_y.saturating_add(region.scissor_height) > target.extent.height()
            {
                return Err(preparation_error(
                    "the C08 bounded render region exceeds its exact target extent",
                ));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(target.label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: target.load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(objects.render_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_viewport(
                region.viewport_x,
                region.viewport_y,
                region.viewport_width,
                region.viewport_height,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                region.scissor_x,
                region.scissor_y,
                region.scissor_width,
                region.scissor_height,
            );
            pass.draw(0..3, 0..1);
        }
        Ok(objects.uses_fixed_source_over_blend())
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
        if self.c08_encoding_state.is_some() {
            return Err(preparation_error(
                "C08 pass completion belongs to its one-shot scheduler; discard an aborted graph and encoder",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_c08_custom_pass(&mut self, pass: RuntimePassId) -> Result<()> {
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Encoding) {
            return Err(preparation_error(
                "C08 custom-pass progress requires the active one-shot encoding session",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_c08_capture(
        &mut self,
        pass: RuntimePassId,
        target: RuntimeResourceId,
        session: &Arc<()>,
        receipt: C08VelloCaptureCompletionReceipt,
    ) -> Result<()> {
        let request = self.require_current_pass(pass)?;
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Encoding)
            || !matches!(request.runtime.kind, RuntimePassKind::VelloCapture(Some(_)))
            || request.runtime.result != RuntimeResultBinding::Resource(target)
            || receipt.pass != pass
            || receipt.target != target
            || !Arc::ptr_eq(&receipt.session, session)
        {
            return Err(preparation_error(
                "C08 capture completion does not match the exact pass, target, and encoding session",
            ));
        }
        self.complete_pass_inner(pass)
    }

    fn complete_pass_inner(&mut self, pass: RuntimePassId) -> Result<()> {
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

    pub(crate) fn finish_c08_submission(
        mut self,
        pending: C08PendingGraphEncoding,
    ) -> Result<C08PreparedGraphSubmission> {
        let completed_session = self.c08_completed_session.take().ok_or_else(|| {
            preparation_error("the prepared C08 graph has no completed encoding session")
        })?;
        if self.c08_encoding_state != Some(C08CustomSpineEncodingState::Complete)
            || !Arc::ptr_eq(&completed_session, &pending.session)
            || !pending.summary.proves_complete_submission()
            || self.next_pass != self.plan.passes.len()
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
                "the C08 graph submission does not own one exact completed prepared frame",
            ));
        }
        let pass_cache_update = self.pass_cache_update.take().ok_or_else(|| {
            preparation_error("the completed C08 graph lost its provisional pass-cache update")
        })?;
        let frame_scope = self.frame_scope.take().ok_or_else(|| {
            preparation_error("the completed C08 graph lost its prepared frame scope")
        })?;
        let C08PendingGraphEncoding {
            summary: _,
            resources: capture_resources,
            session: _,
        } = pending;
        Ok(C08PreparedGraphSubmission {
            capture_resources,
            prepared_frame: PendingC08PreparedFrameCommit {
                frame_scope,
                pass_cache_update,
            },
        })
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
        let _ = self.pass_cache_update.take();
        self.frame_scope
            .take()
            .ok_or_else(|| preparation_error("prepared frame resource scope is already closed"))?
            .finish_checked()
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
            all_bindings_inspected &= pass
                .composite_parameters()
                .is_some_and(|bytes| bytes.as_bytes().len() == 112)
                == matches!(
                    pass.kind(),
                    RuntimePassKind::Composite(Some(RuntimeComposite {
                        kind: RuntimeCompositeKind::Layer { .. },
                        ..
                    }))
                );
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

    pub(crate) const fn composite_parameters(&self) -> Option<&CompositeParameterBytes> {
        self.request.composite_parameters.as_ref()
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

fn runtime_affine_is_finite_and_non_singular(transform: Transform) -> bool {
    let [a, b, c, d, e, f] = transform.as_array();
    if [a, b, c, d, e, f]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == 0.0 {
        return false;
    }
    let a = a / scale;
    let b = b / scale;
    let c = c / scale;
    let d = d / scale;
    let determinant = a * d - b * c;
    determinant.is_finite() && determinant != 0.0
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

const fn runtime_resource_format(
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

const fn shader_binding_role(role: RuntimeReadRole) -> ShaderBindingRoleKey {
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

const fn shader_sampling_edge(edge: RuntimeSamplingEdge) -> ShaderSamplingEdgeKey {
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
            read.role != RuntimeReadRole::CompositeParent
                || (!matches!(
                    kind,
                    RuntimePassKind::Composite(Some(RuntimeComposite {
                        kind: RuntimeCompositeKind::SpanSourceOver,
                        ..
                    }))
                ) && composite_samples_parent)
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
        RuntimePassKind::BlurHorizontal(Some(_)) | RuntimePassKind::BlurVertical(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::GaussianKernel,
        ],
        RuntimePassKind::DropShadowColorize(Some(_)) => vec![
            ShaderDataBindingKey::SpatialUniform,
            ShaderDataBindingKey::DropShadowParameters,
        ],
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::SpanSourceOver,
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
            RuntimePassKind::VelloCapture(Some(RuntimeVelloCapture::Span(span)))
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
                kind: RuntimeCompositeKind::Layer { parameters, .. },
                ..
            })) if parameters.alpha_mask().is_some()
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
                        Some(ShaderMaskSamplingKey::new(
                            upload.quality(),
                            upload.extend(),
                        ))
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
