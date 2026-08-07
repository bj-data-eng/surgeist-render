use super::{Renderer, publication::RenderPublication};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::backend::render_exact_presented_graph_surface;
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::SurfaceBackend;
use crate::{
    BackendErrorCode, Error, ErrorCode, Format, Parameters, PrimitiveFamily, PrimitiveOperation,
    RenderRoute, Result, RuntimeCapabilityUnavailableReason, RuntimeOperation, Scene, Stats,
    Surface, UnsupportedPrimitive,
    backend::{
        DeviceCapabilities, DeviceSlotIdentity, ExactSurfaceGraph, SurfaceFrameCommit,
        render_exact_headless_graph_surface, render_internal_vello_surface,
    },
    command::RenderCommands,
    encode::encode_vello_scene,
    frame::{
        FrameContext, FramePlan, GpuRenderGraph, GraphLoweringCompositeKind, GraphLoweringPassKind,
    },
    gpu_transaction::GpuOperationStage,
    pass::{ExecutableGraphDispatchEligibility, ExecutableGraphWorkingFormatRequest},
    stats::collect_render_stats,
    vello_engine::scene::VelloScene,
};
use std::time::{Duration, Instant};

pub(super) fn runtime_surface_format(surface: &Surface) -> Format {
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    if let crate::surface::SurfaceBackend::Presented {
        surface: native, ..
    } = &surface.backend
    {
        return match native.format {
            wgpu::TextureFormat::Rgba8Unorm => Format::Rgba8,
            wgpu::TextureFormat::Bgra8Unorm => Format::Bgra8,
            _ => surface.options.format,
        };
    }
    surface.options.format
}

#[must_use = "the renderer dispatch boundary must resolve to exactly one execution route"]
pub(super) enum RendererFrameDispatch {
    DirectVello(RenderCommands),
    ExactGraph(Box<ExactSurfaceGraph>),
    RejectedFutureGraph(Error),
}

#[must_use = "prepared renderer execution must reach its selected GPU transaction"]
pub(super) enum PreparedRendererExecution {
    DirectVello(Box<VelloScene>),
    ExactGraph(Box<ExactSurfaceGraph>),
}

impl Renderer {
    pub(super) async fn dispatch_render_frame(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<(DeviceSlotIdentity, RenderPublication)> {
        let device_identity = self.render_device_identity(surface)?;
        let encode_start = Instant::now();
        let (normalized, execution) =
            self.prepare_render_execution(surface, scene, parameters, device_identity)?;
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await?;
        let mut stats = Stats {
            encode_time: Duration::ZERO,
            render_time: Duration::ZERO,
            present_time: Duration::ZERO,
            ..Stats::default()
        };
        if matches!(&execution, PreparedRendererExecution::DirectVello(_)) {
            stats.route = Some(RenderRoute::DirectVello);
            stats.vello_passes = stats.vello_passes.saturating_add(1);
        }
        let mut uploaded_images = self.uploaded_images.clone();
        collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
        stats.encode_time = encode_start.elapsed();
        if parameters.debug || self.options.debug() {
            stats.cache_hits = stats.cache_hits.saturating_add(self.stats.cache_hits);
        }
        let frame = self
            .execute_render_frame(surface, execution, parameters, device_identity)
            .await?;
        Ok((
            device_identity,
            RenderPublication::new(frame, stats, uploaded_images, parameters),
        ))
    }

    fn render_device_identity(&mut self, surface: &Surface) -> Result<DeviceSlotIdentity> {
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceRendering)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceRendering)?;
        surface.ensure_available(RuntimeOperation::SurfaceRendering)?;
        surface.ensure_renderable()?;
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceRendering)?;

        let Some(device_identity) = surface.device_identity() else {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        };
        if self.backend.is_none() {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "no compatible wgpu adapter is available",
            ));
        }
        Ok(device_identity)
    }

    fn prepare_render_execution(
        &mut self,
        surface: &Surface,
        scene: &Scene,
        parameters: Parameters,
        device_identity: DeviceSlotIdentity,
    ) -> Result<(RenderCommands, PreparedRendererExecution)> {
        #[cfg(test)]
        {
            self.preexecution_frame_gate_observation =
                super::PreexecutionFrameGateObservationForTest::default();
        }
        let normalized = scene.normalize(self.capabilities())?;
        let graph_source = normalized.clone();
        let frame_context = FrameContext::try_new(
            surface.size(),
            surface.scale(),
            self.options.antialiasing(),
            parameters.base_color,
        )?;
        let frame_plan = normalized.plan_for(frame_context)?;
        #[cfg(test)]
        {
            self.preexecution_frame_gate_observation
                .validated_plan_count += 1;
        }
        let capabilities = self
            .backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::RenderFailed,
                    "the renderer dispatch boundary lost immutable device capabilities",
                )
            })?;
        #[cfg(test)]
        let working_format = match self.exact_graph_working_format {
            Some(working_format) => ExecutableGraphWorkingFormatRequest::Exact(working_format),
            None => ExecutableGraphWorkingFormatRequest::ConfiguredPolicy(
                self.options.effect_quality_policy(),
            ),
        };
        #[cfg(not(test))]
        let working_format = ExecutableGraphWorkingFormatRequest::ConfiguredPolicy(
            self.options.effect_quality_policy(),
        );
        let dispatch = self.classify_frame_dispatch(
            frame_plan,
            runtime_surface_format(surface),
            working_format,
            &capabilities,
        );
        #[cfg(test)]
        self.observe_frame_dispatch_for_test(&dispatch);
        let dispatch = dispatch?;
        Ok(match dispatch {
            RendererFrameDispatch::DirectVello(normalized) => {
                let vello_scene = encode_vello_scene(&normalized, surface.scale())?;
                (
                    normalized,
                    PreparedRendererExecution::DirectVello(Box::new(vello_scene)),
                )
            }
            RendererFrameDispatch::ExactGraph(graph) => {
                (graph_source, PreparedRendererExecution::ExactGraph(graph))
            }
            RendererFrameDispatch::RejectedFutureGraph(error) => return Err(error),
        })
    }

    async fn execute_render_frame(
        &mut self,
        surface: &mut Surface,
        execution: PreparedRendererExecution,
        parameters: Parameters,
        device_identity: DeviceSlotIdentity,
    ) -> Result<SurfaceFrameCommit> {
        let frame = {
            let backend = self
                .backend
                .as_mut()
                .expect("surface preflight confirmed the renderer backend is available");
            match execution {
                PreparedRendererExecution::DirectVello(vello_scene) => {
                    let transaction = backend.begin_gpu_operation(
                        device_identity,
                        GpuOperationStage::Render,
                        RuntimeOperation::SurfaceRendering,
                    )?;
                    render_internal_vello_surface(
                        backend,
                        transaction,
                        surface,
                        &vello_scene,
                        parameters,
                        self.options.antialiasing(),
                    )
                    .await
                }
                PreparedRendererExecution::ExactGraph(graph) => {
                    #[cfg(any(
                        feature = "render-window",
                        all(feature = "render-web", target_arch = "wasm32")
                    ))]
                    {
                        if matches!(&surface.backend, SurfaceBackend::Presented { .. }) {
                            render_exact_presented_graph_surface(backend, surface, *graph).await
                        } else {
                            render_exact_headless_graph_surface(backend, surface, *graph).await
                        }
                    }
                    #[cfg(not(any(
                        feature = "render-window",
                        all(feature = "render-web", target_arch = "wasm32")
                    )))]
                    {
                        render_exact_headless_graph_surface(backend, surface, *graph).await
                    }
                }
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
        Ok(frame)
    }

    pub(super) fn classify_frame_dispatch(
        &self,
        plan: FramePlan,
        output_format: Format,
        working_format: ExecutableGraphWorkingFormatRequest,
        capabilities: &DeviceCapabilities,
    ) -> Result<RendererFrameDispatch> {
        match plan {
            FramePlan::DirectVello(plan) => {
                Ok(RendererFrameDispatch::DirectVello(plan.into_commands()))
            }
            FramePlan::GpuGraph(graph) => match ExecutableGraphDispatchEligibility::try_classify(
                &graph,
                output_format,
                working_format,
                capabilities,
            )? {
                ExecutableGraphDispatchEligibility::ExactBase(preparable) => {
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::Base(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::ExactComposition(preparable) => {
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::Composition(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::ExactBackdrop(preparable) => {
                    Ok(RendererFrameDispatch::ExactGraph(Box::new(
                        ExactSurfaceGraph::Backdrop(preparable),
                    )))
                }
                ExecutableGraphDispatchEligibility::FuturePasses => {
                    let error = match reject_future_graph_with_typed_diagnostic(&graph) {
                        Err(error) => error,
                        Ok(()) => Error::new(
                            BackendErrorCode::RenderFailed,
                            "a future GPU graph had no unavailable execution pass",
                        ),
                    };
                    Ok(RendererFrameDispatch::RejectedFutureGraph(error))
                }
            },
        }
    }
}

pub(super) fn reject_future_graph_with_typed_diagnostic(graph: &GpuRenderGraph) -> Result<()> {
    let mut has_copy_backdrop = false;
    let mut has_color_filter = false;
    let mut has_blur = false;
    let mut has_drop_shadow = false;

    for pass in graph.lowering_view()?.passes() {
        match pass.kind()? {
            GraphLoweringPassKind::ClearRoot { .. }
            | GraphLoweringPassKind::VelloCapture(Some(_))
            | GraphLoweringPassKind::CanonicalizeCapture
            | GraphLoweringPassKind::Present => {}
            GraphLoweringPassKind::CopyBackdrop => has_copy_backdrop = true,
            GraphLoweringPassKind::ColorFilter(Some(_)) => has_color_filter = true,
            GraphLoweringPassKind::BlurHorizontal(Some(_))
            | GraphLoweringPassKind::BlurVertical(Some(_)) => has_blur = true,
            GraphLoweringPassKind::DropShadowColorize(Some(_)) => has_drop_shadow = true,
            GraphLoweringPassKind::Composite(Some(composite)) => match composite.kind() {
                GraphLoweringCompositeKind::SpanSourceOver => {}
                GraphLoweringCompositeKind::Layer { .. } => {}
                GraphLoweringCompositeKind::DropShadow => has_drop_shadow = true,
            },
            GraphLoweringPassKind::VelloCapture(None)
            | GraphLoweringPassKind::ColorFilter(None)
            | GraphLoweringPassKind::BlurHorizontal(None)
            | GraphLoweringPassKind::BlurVertical(None)
            | GraphLoweringPassKind::DropShadowColorize(None)
            | GraphLoweringPassKind::Composite(None) => {
                return Err(Error::new(
                    BackendErrorCode::RenderFailed,
                    "a malformed GPU graph reached production dispatch",
                ));
            }
        }
    }

    let unsupported = if has_copy_backdrop {
        Some((
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    } else if has_drop_shadow {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ))
    } else if has_blur {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ))
    } else if has_color_filter {
        Some((
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ))
    } else {
        None
    };
    if let Some((family, operation)) = unsupported {
        return Err(Error::unsupported_render_primitive(
            UnsupportedPrimitive::new(family, operation),
        ));
    }
    Err(Error::new(
        BackendErrorCode::RenderFailed,
        "a future GPU graph had no unavailable execution pass",
    ))
}
