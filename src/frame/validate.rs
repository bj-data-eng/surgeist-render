use super::graph::{
    GpuRenderGraph, GraphBuildResult, GraphValidationError, PassIndex, ResourceIndex,
    SemanticCompositeKind, SemanticGraphPass, SemanticPassId, SemanticPassIntent,
    SemanticPassResult, SemanticResourceId, SemanticResourceProducer, SemanticResourceRole,
    WorkingImageInitialization,
};

pub(super) fn validate_semantic_frame_graph(graph: &GpuRenderGraph) -> GraphBuildResult<()> {
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

pub(super) struct LoweringValidationState<'graph> {
    graph: &'graph GpuRenderGraph,
    actual_reads: Vec<u32>,
    last_reads: Vec<Option<SemanticPassId>>,
    next_pass_index: usize,
}

impl<'graph> LoweringValidationState<'graph> {
    pub(super) fn begin(graph: &'graph GpuRenderGraph) -> GraphBuildResult<Self> {
        validate_semantic_frame_graph(graph)?;
        if graph.resources.is_empty() {
            return Err(GraphValidationError::MissingRootWorkingImage);
        }
        if graph.passes.is_empty() {
            return Err(GraphValidationError::MissingFinalPresent);
        }

        let actual_reads = vec![0_u32; graph.resources.len()];
        let last_reads = vec![None; graph.resources.len()];
        validate_lowering_resources(graph)?;
        validate_lowering_imports(graph)?;
        Ok(Self {
            graph,
            actual_reads,
            last_reads,
            next_pass_index: 0,
        })
    }

    pub(super) fn validate_pass(&mut self, pass: &SemanticGraphPass) -> GraphBuildResult<()> {
        let index = self.next_pass_index;
        let graph = self.graph;
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
            self.actual_reads[resource_index] = self.actual_reads[resource_index]
                .checked_add(1)
                .ok_or(GraphValidationError::ReadCountOverflow(*read))?;
            self.last_reads[resource_index] = Some(pass.id);
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

        self.next_pass_index += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> GraphBuildResult<()> {
        validate_lowering_lifetimes(self.graph, &self.actual_reads, &self.last_reads)?;
        validate_lowering_anchors(self.graph)
    }
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
