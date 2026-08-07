use super::{
    Renderer,
    dispatch::{RendererFrameDispatch, runtime_surface_format},
    publication::RenderPublication,
};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::SurfaceBackend;
use crate::{
    backend::*,
    command::RenderCommands,
    frame::{FrameContext, FramePlan, GpuRenderGraph},
    pass::ExecutableGraphWorkingFormatRequest,
    resource::{ResourceManagerObservationForTest, WorkingFormat},
    stats::collect_render_stats,
    *,
};
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResourcePreparationObservationForTest {
    pub(crate) complete_resource_and_pass_handoff: bool,
    pub(crate) exact_capture_coverage_working_and_mask_allocations: bool,
    pub(crate) typed_bindings_and_last_use_releases: bool,
    pub(crate) spatial_bytes_and_cache_keys_preserved: bool,
    pub(crate) allocation_preflight_is_atomic: bool,
    pub(crate) failure_and_drop_cleanup: bool,
    pub(crate) repeated_reuse_is_exact_and_bounded: bool,
    pub(crate) populated_pass_cache_is_preserved: bool,
}

impl Renderer {
    pub(crate) fn resource_preparation_observation_for_test(
        &mut self,
        commands: RenderCommands,
        surface_size: Size,
        surface_scale: f64,
        base_color: Color,
        output_format: Format,
    ) -> Result<ResourcePreparationObservationForTest> {
        let context = FrameContext::try_new(
            surface_size,
            surface_scale,
            self.options.antialiasing(),
            base_color,
        )?;
        let FramePlan::GpuGraph(graph) = commands.plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the resource preparation fixture did not produce a GPU graph",
            ));
        };
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::EffectRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "resource preparation coverage requires a ready default device",
            )
        })?;
        let policy = self.options.effect_quality_policy();
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::EffectRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "resource preparation coverage requires a renderer backend",
            )
        })?;
        let capabilities = backend
            .device_capabilities(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "resource preparation coverage requires immutable device capabilities",
                )
            })?;
        let working_format = capabilities.resolve_effect_working_format(policy)?;
        let lowered = super::pass::LoweredGraphPlan::try_lower_validated_graph(
            &graph,
            working_format,
            output_format,
            &capabilities,
        )?;
        let pass_cache_before = backend
            .seed_device_pass_cache_sampler_for_test(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "ready device disappeared before pass-cache preservation coverage",
                )
            })?;

        let allocation_preflight_is_atomic =
            preparation_preflight_is_atomic(backend, device_identity, &lowered, policy)?;
        let (first_exercise, repeated_reuse_is_exact_and_bounded) =
            exercise_preparation_reuse(backend, device_identity, &lowered, policy)?;
        let failure_and_drop_cleanup =
            preparation_failure_cleanup(backend, device_identity, lowered, policy)?;
        let pass_cache_after = backend
            .ready_device_state_borrow_for_test(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "ready device disappeared after pass-cache preservation coverage",
                )
            })?
            .device_pass_cache_counts_for_test();
        let populated_pass_cache_is_preserved =
            pass_cache_before.has_exactly_one_sampler() && pass_cache_after == pass_cache_before;

        Ok(ResourcePreparationObservationForTest {
            complete_resource_and_pass_handoff: first_exercise.complete_resource_and_pass_handoff,
            exact_capture_coverage_working_and_mask_allocations: first_exercise
                .exact_capture_coverage_working_and_mask_allocations,
            typed_bindings_and_last_use_releases: first_exercise
                .typed_bindings_and_last_use_releases,
            spatial_bytes_and_cache_keys_preserved: first_exercise
                .spatial_bytes_and_cache_keys_preserved,
            allocation_preflight_is_atomic,
            failure_and_drop_cleanup,
            repeated_reuse_is_exact_and_bounded,
            populated_pass_cache_is_preserved,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ForcedGraphRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) captures: Vec<ForcedGraphCaptureForTest>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ForcedGraphCaptureForTest {
    pub(crate) antialiasing: Antialiasing,
    pub(crate) capture_transform: Transform,
    pub(crate) parent_to_surface: Transform,
    pub(crate) device_origin: (i32, i32),
    pub(crate) texel_origin: Point,
    pub(crate) extent: PhysicalSize,
    pub(crate) raster_scale: f64,
}

impl From<super::frame::ForcedGraphCaptureObservationForTest> for ForcedGraphCaptureForTest {
    fn from(capture: super::frame::ForcedGraphCaptureObservationForTest) -> Self {
        Self {
            antialiasing: capture.antialiasing,
            capture_transform: capture.capture_transform,
            parent_to_surface: capture.parent_to_surface,
            device_origin: capture.device_origin,
            texel_origin: capture.texel_origin,
            extent: capture.extent,
            raster_scale: capture.raster_scale,
        }
    }
}

struct ForcedGraphPreparationForTest {
    device_identity: DeviceSlotIdentity,
    normalized: RenderCommands,
    preparable: super::pass::BasePreparableGraph,
    output_extent: PhysicalSize,
    captures: Vec<ForcedGraphCaptureForTest>,
}

fn preparation_resource_observation(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    missing: &'static str,
) -> Result<ResourceManagerObservationForTest> {
    backend
        .ready_device_state_borrow_for_test(identity)
        .ok_or_else(|| Error::new(BackendErrorCode::RenderFailed, missing))
        .map(|ready| ready.internal_resource_manager_observation_for_test())
}

fn preparation_preflight_is_atomic(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: &super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<bool> {
    let before = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared before preparation preflight",
    )?;
    let rejected = backend
        .prepare_graph_resources(
            identity,
            lowered.with_duplicate_preparation_resource_for_test(),
            policy,
        )
        .is_err();
    let after = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after preparation preflight",
    )?;
    Ok(rejected && before == after)
}

fn exercise_preparation_reuse(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: &super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<(super::pass::PreparedGraphExerciseObservationForTest, bool)> {
    let (first_exercise, first_identities) = {
        let mut prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let identities = prepared.allocation_identities_for_test();
        let exercise = prepared.exercise_for_test()?;
        let _ = prepared.finish()?;
        (exercise, identities)
    };
    let after_first = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after first complete preparation",
    )?;
    let second_identities = {
        let mut prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        let identities = prepared.allocation_identities_for_test();
        let _ = prepared.exercise_for_test()?;
        let _ = prepared.finish()?;
        identities
    };
    let after_second = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after repeated complete preparation",
    )?;
    let reuse = first_identities == second_identities
        && after_second.payload_creation_attempts == after_first.payload_creation_attempts
        && after_second.entry_count == after_first.entry_count
        && after_second.retained_bytes == after_first.retained_bytes;
    Ok((first_exercise, reuse))
}

fn preparation_failure_cleanup(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    lowered: super::pass::LoweredGraphPlan,
    policy: EffectQualityPolicy,
) -> Result<bool> {
    let early_finish_failed = {
        let prepared = backend.prepare_graph_resources(identity, lowered.clone(), policy)?;
        prepared.finish().is_err()
    };
    let after_finish = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after failed prepared finish",
    )?;
    drop(backend.prepare_graph_resources(identity, lowered, policy)?);
    let after_drop = preparation_resource_observation(
        backend,
        identity,
        "ready device disappeared after prepared cancellation",
    )?;
    Ok(early_finish_failed && after_finish.leased_count == 0 && after_drop.leased_count == 0)
}

#[derive(Debug)]
pub(crate) struct ColorFilterRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) source_origin: (i32, i32),
    pub(crate) source_extent: PhysicalSize,
    pub(crate) source_texel_origin: Point,
    pub(crate) source_raster_scale: f64,
}

#[derive(Debug)]
pub(crate) struct SpatialFilterRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) source_spatial: super::pass::ColorFilterSpatialObservationForTest,
    pub(crate) result_spatial: super::pass::ColorFilterSpatialObservationForTest,
}

#[derive(Debug)]
pub(crate) struct BoundedBackdropRenderResultForTest {
    pub(crate) stats: Stats,
    pub(crate) working_format: WorkingFormat,
    pub(crate) output_extent: PhysicalSize,
    pub(crate) parent_spatial: super::pass::ColorFilterSpatialObservationForTest,
    pub(crate) capture_spatial: super::pass::ColorFilterSpatialObservationForTest,
}

struct ColorFilterFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    source_spatial: super::pass::ColorFilterSpatialObservationForTest,
}

struct SpatialFilterFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    source_spatial: super::pass::ColorFilterSpatialObservationForTest,
    result_spatial: super::pass::ColorFilterSpatialObservationForTest,
}

struct BoundedBackdropFixturePreparationForTest {
    device_identity: DeviceSlotIdentity,
    frame_start: Instant,
    encode_start: Instant,
    normalized: RenderCommands,
    graph: ExactSurfaceGraph,
    output_extent: PhysicalSize,
    parent_spatial: super::pass::ColorFilterSpatialObservationForTest,
    capture_spatial: super::pass::ColorFilterSpatialObservationForTest,
}

impl Renderer {
    pub(crate) fn uploaded_images_for_test(&self) -> HashSet<ImageId> {
        self.uploaded_images.clone()
    }
}

impl Renderer {
    pub(crate) async fn render_with_exact_graph_working_format_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<Stats> {
        let frame_start = Instant::now();
        let (device_identity, publication) = self
            .dispatch_render_frame_with_working_format(
                surface,
                scene,
                parameters,
                ExecutableGraphWorkingFormatRequest::Exact(working_format),
            )
            .await?;
        self.publish_clean_render_frame(surface, device_identity, publication, frame_start)
    }

    fn classify_color_filter_fixture_dispatch(
        &self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        let preparable = super::pass::color_filter_preparable_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::ColorFilter(preparable),
        )))
    }

    fn classify_spatial_filter_fixture_dispatch(
        &self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        let preparable = super::pass::spatial_filter_preparable_graph_from_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::SpatialFilter(preparable),
        )))
    }

    fn classify_bounded_backdrop_fixture_dispatch(
        &self,
        graph: &GpuRenderGraph,
        output_format: Format,
        working_format: WorkingFormat,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        let preparable = super::pass::backdrop_preparable_graph_from_graph_for_test(
            graph,
            output_format,
            working_format,
            capabilities,
        )?;
        Ok(RendererFrameDispatch::ExactGraph(Box::new(
            ExactSurfaceGraph::Backdrop(preparable),
        )))
    }

    /// Test-only entry for forcing ordinary commands through the exact
    /// production graph executor without adding a public route or option.
    pub(crate) async fn render_forced_base_graph_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<ForcedGraphRenderResultForTest> {
        self.render_forced_base_graph_with_capture_mapping_for_test(
            surface,
            scene,
            parameters,
            working_format,
            super::frame::ForcedVelloCaptureMappingForTest::identity(),
        )
        .await
    }

    /// Test-only entry that keeps capture and parent mappings distinct while
    /// executing the same production graph path.
    pub(crate) async fn render_forced_base_graph_with_capture_mapping_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
        capture_mapping: super::frame::ForcedVelloCaptureMappingForTest,
    ) -> Result<ForcedGraphRenderResultForTest> {
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let ForcedGraphPreparationForTest {
            device_identity,
            normalized,
            preparable,
            output_extent,
            captures,
        } = self.prepare_forced_base_graph_for_test(
            surface,
            scene,
            parameters,
            working_format,
            capture_mapping,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let (stats, uploaded_images) =
            self.forced_graph_stats_for_test(&normalized, parameters, encode_start);
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("forced base graph preflight confirmed the renderer backend is available");
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(
                        backend,
                        surface,
                        ExactSurfaceGraph::Base(preparable),
                    )
                    .await
                } else {
                    render_exact_headless_graph_surface(
                        backend,
                        surface,
                        ExactSurfaceGraph::Base(preparable),
                    )
                    .await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(
                    backend,
                    surface,
                    ExactSurfaceGraph::Base(preparable),
                )
                .await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(device_identity);
        }
        let frame = match frame {
            Err(error) if error.code() == ErrorCode::SurfaceOutdated => {
                self.configure_presented_surface_if_needed(
                    surface,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                return Err(error);
            }
            Err(error) => return Err(error),
            Ok(frame) => frame,
        };
        let stats = self.publish_clean_render_frame(
            surface,
            device_identity,
            RenderPublication::new(frame, stats, uploaded_images, parameters),
            frame_start,
        )?;
        Ok(ForcedGraphRenderResultForTest {
            stats,
            working_format,
            output_extent,
            captures,
        })
    }

    fn forced_graph_stats_for_test(
        &self,
        normalized: &RenderCommands,
        parameters: Parameters,
        encode_start: Instant,
    ) -> (Stats, HashSet<ImageId>) {
        let mut stats = Stats {
            encode_time: encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        (stats, uploaded_images)
    }

    fn prepare_forced_base_graph_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
        capture_mapping: super::frame::ForcedVelloCaptureMappingForTest,
    ) -> Result<ForcedGraphPreparationForTest> {
        let device_identity = self.validate_forced_graph_surface_for_test(surface)?;
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph = super::frame::forced_base_graph_with_capture_mapping_for_test(
            normalized.clone(),
            context,
            capture_mapping,
        )?;
        let captures = graph.forced_capture_observations_for_test();
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private base graph forced route lost immutable device capabilities",
                )
            })?;
        let dispatch = self.classify_frame_dispatch(
            FramePlan::GpuGraph(graph),
            runtime_surface_format(surface),
            ExecutableGraphWorkingFormatRequest::Exact(working_format),
            &capabilities,
        )?;
        let preparable = match dispatch {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::Base(preparable) => preparable,
                ExactSurfaceGraph::Composition(_)
                | ExactSurfaceGraph::ColorFilter(_)
                | ExactSurfaceGraph::SpatialFilter(_)
                | ExactSurfaceGraph::Backdrop(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private forced graph is outside the exact executable base graph subset",
                    ));
                }
            },
            _ => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private forced graph is outside the exact executable base graph subset",
                ));
            }
        };
        let output_extent = preparable.output_extent()?;
        let prepared_grids = preparable.capture_grids_for_test();
        if captures.len() != prepared_grids.len()
            || captures
                .iter()
                .zip(&prepared_grids)
                .any(|(capture, prepared)| {
                    capture.texel_origin != prepared.texel_origin
                        || capture.extent != prepared.extent
                        || capture.raster_scale != prepared.raster_scale
                })
        {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the prepared Vello capture grid differs from the validated semantic graph",
            ));
        }
        Ok(ForcedGraphPreparationForTest {
            device_identity,
            normalized,
            preparable,
            output_extent,
            captures: captures
                .into_iter()
                .map(ForcedGraphCaptureForTest::from)
                .collect(),
        })
    }

    fn validate_forced_graph_surface_for_test(
        &mut self,
        surface: &Surface,
    ) -> Result<DeviceSlotIdentity> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;
        surface.device_identity().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the private base graph forced route requires a device-backed surface",
            )
        })
    }

    /// Test-only color-filter ingress into the shared exact graph executor.
    pub(crate) async fn render_color_filter_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<ColorFilterRenderResultForTest> {
        let prepared = self.prepare_color_filter_fixture_for_test(
            surface,
            scene,
            filters,
            parameters,
            working_format,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        let frame = {
            let backend = self.backend.as_mut().expect(
                "color-filter fixture preflight confirmed the renderer backend is available",
            );
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(backend, surface, prepared.graph).await
                } else {
                    render_exact_headless_graph_surface(backend, surface, prepared.graph).await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(backend, surface, prepared.graph).await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = match frame {
            Err(error) if error.code() == ErrorCode::SurfaceOutdated => {
                self.configure_presented_surface_if_needed(
                    surface,
                    RuntimeOperation::SurfaceRendering,
                )
                .await?;
                return Err(error);
            }
            Err(error) => return Err(error),
            Ok(frame) => frame,
        };
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication::new(frame, stats, uploaded_images, parameters),
            prepared.frame_start,
        )?;
        Ok(ColorFilterRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            source_origin: prepared.source_spatial.device_origin,
            source_extent: prepared.source_spatial.device_extent,
            source_texel_origin: prepared.source_spatial.texel_origin,
            source_raster_scale: prepared.source_spatial.raster_scale,
        })
    }

    fn prepare_color_filter_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<ColorFilterFixturePreparationForTest> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;
        let device_identity = surface.device_identity().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the private color-filter fixture requires a device-backed surface",
            )
        })?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph =
            super::frame::authored_filter_graph_for_test(filters, normalized.clone(), context)?;
        let capabilities = self
            .backend
            .as_mut()
            .ok_or_else(|| {
                Error::runtime_unavailable(
                    RuntimeOperation::SurfaceRendering,
                    RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                    "the private color-filter fixture requires a renderer backend",
                )
            })?
            .device_capabilities(device_identity)
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private color-filter fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_color_filter_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::ColorFilter(preparable) => preparable,
                ExactSurfaceGraph::Base(_)
                | ExactSurfaceGraph::Composition(_)
                | ExactSurfaceGraph::SpatialFilter(_)
                | ExactSurfaceGraph::Backdrop(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private color-filter fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private color-filter fixture left its exact renderer dispatch route",
                ));
            }
            RendererFrameDispatch::RejectedFutureGraph(error) => return Err(error),
        };
        let output_extent = preparable.output_extent()?;
        let source_spatial = preparable.first_color_spatial_for_test().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "the private color-filter fixture lost its first exact color source",
            )
        })?;
        Ok(ColorFilterFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::ColorFilter(preparable),
            output_extent,
            source_spatial,
        })
    }

    /// Test-only spatial-filter ingress into the shared exact graph executor.
    pub(crate) async fn render_spatial_filter_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<SpatialFilterRenderResultForTest> {
        let prepared = self.prepare_spatial_filter_fixture_for_test(
            surface,
            scene,
            filters,
            parameters,
            working_format,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        let frame = {
            let backend = self.backend.as_mut().expect(
                "spatial-filter fixture preflight confirmed the renderer backend is available",
            );
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            {
                if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                    render_exact_presented_graph_surface(backend, surface, prepared.graph).await
                } else {
                    render_exact_headless_graph_surface(backend, surface, prepared.graph).await
                }
            }
            #[cfg(not(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            )))]
            {
                render_exact_headless_graph_surface(backend, surface, prepared.graph).await
            }
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = frame?;
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication::new(frame, stats, uploaded_images, parameters),
            prepared.frame_start,
        )?;
        Ok(SpatialFilterRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            source_spatial: prepared.source_spatial,
            result_spatial: prepared.result_spatial,
        })
    }

    fn prepare_spatial_filter_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        filters: Vec<FilterList>,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<SpatialFilterFixturePreparationForTest> {
        let device_identity = self.validate_forced_graph_surface_for_test(surface)?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let graph =
            super::frame::authored_filter_graph_for_test(filters, normalized.clone(), context)?;
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private spatial-filter fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_spatial_filter_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::SpatialFilter(preparable) => preparable,
                ExactSurfaceGraph::Base(_)
                | ExactSurfaceGraph::Composition(_)
                | ExactSurfaceGraph::ColorFilter(_)
                | ExactSurfaceGraph::Backdrop(_) => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private spatial-filter fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private spatial-filter fixture left its exact renderer dispatch route",
                ));
            }
            RendererFrameDispatch::RejectedFutureGraph(error) => return Err(error),
        };
        let output_extent = preparable.output_extent()?;
        let (source_spatial, result_spatial) =
            preparable.first_filter_spatial_for_test().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private spatial-filter fixture lost its first spatial mapping",
                )
            })?;
        Ok(SpatialFilterFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::SpatialFilter(preparable),
            output_extent,
            source_spatial,
            result_spatial,
        })
    }

    /// Test-only bounded-backdrop ingress into the shared exact graph executor.
    pub(crate) async fn render_bounded_backdrop_fixture_for_test(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<BoundedBackdropRenderResultForTest> {
        let prepared = self.prepare_bounded_backdrop_fixture_for_test(
            surface,
            scene,
            parameters,
            working_format,
        )?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: prepared.encode_start.elapsed(),
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(
            &prepared.normalized.commands,
            &mut stats,
            &mut uploaded_images,
        );
        let frame = {
            let backend = self.backend.as_mut().expect(
                "bounded-backdrop fixture preflight confirmed the renderer backend is available",
            );
            render_exact_headless_graph_surface(backend, surface, prepared.graph).await
        };
        if frame.is_err()
            && let Some(backend) = self.backend.as_mut()
        {
            backend.observe_device_terminal(prepared.device_identity);
        }
        let frame = frame?;
        let stats = self.publish_clean_render_frame(
            surface,
            prepared.device_identity,
            RenderPublication::new(frame, stats, uploaded_images, parameters),
            prepared.frame_start,
        )?;
        Ok(BoundedBackdropRenderResultForTest {
            stats,
            working_format,
            output_extent: prepared.output_extent,
            parent_spatial: prepared.parent_spatial,
            capture_spatial: prepared.capture_spatial,
        })
    }

    fn prepare_bounded_backdrop_fixture_for_test(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        working_format: WorkingFormat,
    ) -> Result<BoundedBackdropFixturePreparationForTest> {
        let device_identity = self.validate_forced_graph_surface_for_test(surface)?;
        let frame_start = Instant::now();
        let encode_start = Instant::now();
        let normalized = scene.normalize(self.capabilities())?;
        let context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let FramePlan::GpuGraph(graph) = normalized.clone().plan_for(context)? else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "the private bounded-backdrop fixture did not produce a bounded backdrop graph",
            ));
        };
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private bounded-backdrop fixture lost immutable device capabilities",
                )
            })?;
        let preparable = match self.classify_bounded_backdrop_fixture_dispatch(
            &graph,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        )? {
            RendererFrameDispatch::ExactGraph(graph) => match *graph {
                ExactSurfaceGraph::Backdrop(preparable) => preparable,
                _ => {
                    return Err(Error::new(
                        BackendErrorCode::RenderFailed,
                        "the private bounded-backdrop fixture left its exact renderer dispatch route",
                    ));
                }
            },
            RendererFrameDispatch::DirectVello(_) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private bounded-backdrop fixture left its exact renderer dispatch route",
                ));
            }
            RendererFrameDispatch::RejectedFutureGraph(error) => return Err(error),
        };
        let output_extent = preparable.output_extent()?;
        let (parent_spatial, capture_spatial) =
            preparable.backdrop_spatial_for_test().ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the private bounded-backdrop fixture lost its exact backdrop mapping",
                )
            })?;
        Ok(BoundedBackdropFixturePreparationForTest {
            device_identity,
            frame_start,
            encode_start,
            normalized,
            graph: ExactSurfaceGraph::Backdrop(preparable),
            output_extent,
            parent_spatial,
            capture_spatial,
        })
    }
}
