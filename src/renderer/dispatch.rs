use super::Renderer;
use crate::{
    BackendErrorCode, Error, Format, PrimitiveFamily, PrimitiveOperation, Result, Surface,
    UnsupportedPrimitive,
    backend::{DeviceCapabilities, ExactSurfaceGraph},
    command::RenderCommands,
    frame::{FramePlan, GpuRenderGraph, GraphLoweringCompositeKind, GraphLoweringPassKind},
    pass::{ExecutableGraphDispatchEligibility, ExecutableGraphWorkingFormatRequest},
    vello_engine::scene::VelloScene,
};

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
