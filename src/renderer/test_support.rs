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
    gpu_transaction::GpuOperationStage,
    image::ImageContentIdentity,
    pass::{ExecutableGraphDispatchEligibility, ExecutableGraphWorkingFormatRequest},
    readback::read_texture_rgba,
    resource::{ResourceManagerObservationForTest, WorkingFormat},
    stats::collect_render_stats,
    *,
};
#[cfg(feature = "render-window")]
use crate::{geometry::physical_size, validation::validate_surface_options};
use std::{
    collections::HashSet,
    sync::Arc,
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
    pub(crate) async fn submit_prepared_vello_pass_for_test(
        &mut self,
        prepared: &super::super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<
        super::super::gpu_transaction::test_support::InternalVelloSubmissionObservationForTest,
    > {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello transaction coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello transaction coverage requires a renderer backend",
            )
        })?;
        backend
            .submit_prepared_vello_pass_for_test(device_identity, prepared, target_extent)
            .await
    }

    pub(crate) async fn fail_prepared_vello_pass_after_submit_for_test(
        &mut self,
        prepared: &super::super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
        publication: &mut Option<u64>,
    ) -> Result<()> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello failure coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello failure coverage requires a renderer backend",
            )
        })?;
        backend
            .fail_prepared_vello_pass_after_submit_for_test(
                device_identity,
                prepared,
                target_extent,
                publication,
            )
            .await
    }

    pub(crate) async fn fault_prepared_vello_accounting_after_submit_for_test(
        &mut self,
        prepared: &super::super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<()> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello accounting coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello accounting coverage requires a renderer backend",
            )
        })?;
        backend
            .fault_prepared_vello_accounting_after_submit_for_test(
                device_identity,
                prepared,
                target_extent,
            )
            .await
    }

    pub(crate) async fn cancel_prepared_vello_pass_after_submit_for_test(
        &mut self,
        prepared: &super::super::vello_engine::PreparedVelloPass,
        target_extent: PhysicalSize,
    ) -> Result<ResourceManagerObservationForTest> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello cancellation coverage requires a ready default device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "internal Vello cancellation coverage requires a renderer backend",
            )
        })?;
        backend
            .cancel_prepared_vello_pass_after_submit_for_test(
                device_identity,
                prepared,
                target_extent,
            )
            .await
    }

    pub(crate) fn default_device_active_operation_generation_for_test(&mut self) -> Option<u64> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .active_operation_generation_for_test(device_identity)
    }

    pub(crate) fn default_device_has_no_terminal_signal_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return true;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.terminal_reason(device_identity).is_none())
    }

    pub(crate) fn default_device_capabilities_for_test(&mut self) -> AvailableRuntimeCapabilities {
        let device_identity = self.default_device.expect("test requires a default device");
        self.backend
            .as_mut()
            .and_then(|backend| backend.device_capabilities(device_identity))
            .expect("test requires a ready default device")
            .runtime_report(Format::Rgba8)
    }

    pub(crate) fn override_default_device_effect_precision_facts_for_test(
        &mut self,
        effect_precisions: EffectPrecisionCapabilities,
    ) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        self.backend.as_mut().is_some_and(|backend| {
            backend
                .override_device_effect_precision_facts_for_test(device_identity, effect_precisions)
        })
    }

    pub(crate) fn destroy_default_device_for_test(&mut self) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        let Some(backend) = self.backend.as_mut() else {
            return false;
        };
        backend.destroy_device_for_test(device_identity)
    }

    pub(crate) fn wait_for_default_terminal_signal_for_test(&mut self, timeout: Duration) -> bool {
        let Some(device_identity) = self.default_device else {
            return false;
        };
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.wait_for_terminal_for_test(device_identity, timeout))
    }

    pub(crate) async fn add_donor_device_slot_for_test(&mut self) -> Result<DeviceSlotIdentity> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the renderer has no backend to receive a donor wgpu device",
            )
        })?;
        backend.add_device_slot_for_test().await
    }

    pub(crate) async fn submit_scoped_wgpu_probe_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> Result<()> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real second-slot WGPU coverage requires a renderer backend",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (device, queue) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
        let command_buffer = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Surgeist second-slot terminal test target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist second-slot terminal test encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Surgeist second-slot terminal test pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
            Ok(encoder.finish())
        };
        let scope_result = match command_buffer {
            Ok(command_buffer) => {
                super::super::gpu_transaction::test_support::submit_command_buffer_for_test(
                    transaction,
                    queue,
                    command_buffer,
                    RuntimeOperation::SurfaceRendering,
                )
                .await
            }
            Err(error) => match transaction.finish(RuntimeOperation::SurfaceRendering).await {
                Ok(()) => Err(error),
                Err(scope_error) => Err(scope_error),
            },
        };
        backend.observe_device_terminal(device_identity);
        scope_result
    }
}

pub(crate) fn unsupported_graph_diagnostic_for_test(
    graph: &GpuRenderGraph,
    output_format: Format,
    capabilities: &DeviceCapabilities,
) -> Result<Option<UnsupportedPrimitive>> {
    match ExecutableGraphDispatchEligibility::try_classify(
        graph,
        output_format,
        ExecutableGraphWorkingFormatRequest::Exact(WorkingFormat::HighPrecision),
        capabilities,
    )? {
        ExecutableGraphDispatchEligibility::FuturePasses => {
            let error = super::dispatch::reject_future_graph_with_typed_diagnostic(graph)
                .expect_err("an unsupported graph diagnostic probe must reject before execution");
            Ok(error.unsupported_primitive())
        }
        ExecutableGraphDispatchEligibility::ExactBase(_)
        | ExecutableGraphDispatchEligibility::ExactComposition(_)
        | ExecutableGraphDispatchEligibility::ExactBackdrop(_) => Ok(None),
    }
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

impl Renderer {
    pub(crate) fn default_wgpu_device_queue(&mut self) -> Option<(&wgpu::Device, &wgpu::Queue)> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        backend
            .device_queue(device_identity, RuntimeOperation::SurfaceRendering)
            .ok()
    }

    pub(crate) fn default_offscreen_render_context(
        &mut self,
    ) -> Option<OffscreenRenderGpuContext<'_>> {
        let backend = self.backend.as_mut()?;
        let device_identity = self.default_device?;
        if backend
            .terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
            .is_some()
        {
            return None;
        }
        Some(OffscreenRenderGpuContext::new(backend, device_identity))
    }

    pub(crate) async fn read_render_texture_for_test(
        &mut self,
        texture: &wgpu::Texture,
        physical_size: PhysicalSize,
    ) -> Result<ImageBuffer> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "required render-texture readback needs an available wgpu device",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "required render-texture readback needs an available wgpu backend",
            )
        })?;
        read_texture_rgba(
            backend,
            device_identity,
            texture,
            physical_size,
            RuntimeOperation::SurfaceRendering,
        )
        .await
    }

    pub(crate) fn signal_default_device_loss_for_test(&mut self, reason: DeviceLossReason) {
        if let Some(device_identity) = self.default_device {
            self.signal_device_loss_for_test(device_identity, reason);
        }
    }

    pub(crate) fn signal_device_loss_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
        reason: DeviceLossReason,
    ) {
        if let Some(backend) = self.backend.as_mut() {
            backend.signal_loss_for_test(device_identity, reason);
        }
    }

    pub(crate) fn device_signal_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> Option<Arc<DeviceSignal>> {
        self.backend
            .as_mut()?
            .device_signal_for_test(device_identity)
    }

    pub(crate) fn default_device_signal_for_test(&mut self) -> Option<Arc<DeviceSignal>> {
        self.device_signal_for_test(self.default_device?)
    }

    pub(crate) fn signal_device_uncaptured_fault_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
        kind: GpuFaultKind,
    ) {
        if let Some(backend) = self.backend.as_mut() {
            backend.signal_uncaptured_fault_for_test(device_identity, kind);
        }
    }

    pub(crate) fn default_device_renderer_released_for_test(&mut self) -> bool {
        match (self.backend.as_mut(), self.default_device) {
            (Some(backend), Some(device_identity)) => {
                backend.renderer_released_for_test(device_identity)
            }
            (None, None) => true,
            _ => false,
        }
    }

    pub(crate) fn device_renderer_released_for_test(
        &mut self,
        device_identity: DeviceSlotIdentity,
    ) -> bool {
        self.backend
            .as_mut()
            .is_some_and(|backend| backend.renderer_released_for_test(device_identity))
    }

    pub(crate) fn default_ready_device_state_borrow_for_test(
        &mut self,
    ) -> Option<ReadyDeviceStateBorrowForTest<'_>> {
        let device_identity = self.default_device?;
        self.backend
            .as_mut()?
            .ready_device_state_borrow_for_test(device_identity)
    }

    pub(crate) async fn deliberate_validation_error_for_test(&mut self) -> Result<Result<()>> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU error-scope coverage requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (device, _) =
            backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
        let _ = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist deliberate scoped validation failure"),
            size: wgpu::Extent3d {
                width: 0,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let result = transaction.finish(RuntimeOperation::SurfaceRendering).await;
        backend.observe_device_terminal(device_identity);
        Ok(result)
    }

    pub(crate) async fn scoped_clear_fill_probe_for_test(&mut self) -> Result<ImageBuffer> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "real GPU clear/fill probe requires a host adapter",
            )
        })?;
        let transaction = backend.begin_gpu_operation(
            device_identity,
            GpuOperationStage::Render,
            RuntimeOperation::SurfaceRendering,
        )?;
        let (result, destination_texture) = {
            let destination = super::super::texture::TextureDescriptor::try_new(
                PhysicalSize::new(2, 2),
                Format::Rgba8,
                super::super::texture::TextureUsageIntent::IntermediatePass,
            )?;
            let (device, queue) =
                backend.device_queue(device_identity, RuntimeOperation::SurfaceRendering)?;
            let (destination_texture, destination_view) =
                create_texture(device, "Surgeist scoped clear destination", destination);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Surgeist scoped test-only clear encoder"),
            });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Surgeist scoped test-only clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &destination_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.25,
                                g: 0.5,
                                b: 0.75,
                                a: 1.0,
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
            (
                super::super::gpu_transaction::test_support::submit_command_buffer_for_test(
                    transaction,
                    queue,
                    encoder.finish(),
                    RuntimeOperation::SurfaceRendering,
                )
                .await,
                destination_texture,
            )
        };
        backend.observe_device_terminal(device_identity);
        result?;
        read_texture_rgba(
            backend,
            device_identity,
            &destination_texture,
            PhysicalSize::new(2, 2),
            RuntimeOperation::SurfaceRendering,
        )
        .await
    }
}

#[cfg(feature = "render-window")]
impl Renderer {
    pub(crate) async fn configure_presented_surface_for_test(
        &mut self,
        surface: &mut Surface,
    ) -> Result<()> {
        self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceRendering)
            .await
    }

    pub(crate) fn display_free_presented_surface_for_test(
        &mut self,
        options: SurfaceOptions,
    ) -> Result<Surface> {
        let device_identity = self.default_device.ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a host adapter",
            )
        })?;
        self.display_free_presented_surface_on_device_for_test(
            options,
            device_identity,
            Attachment::from_web_canvas("display-free-presented-test-target"),
        )
    }

    pub(crate) fn display_free_presented_surface_on_device_for_test(
        &mut self,
        options: SurfaceOptions,
        device_identity: DeviceSlotIdentity,
        attachment: Attachment,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        if !matches!(&attachment, Attachment::WebCanvas(_)) {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "the display-free presented fixture requires a web-canvas attachment",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a renderer backend",
            )
        })?;
        if !backend.has_device_slot(device_identity) {
            return Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "the display-free presented fixture requires a current device slot",
            ));
        }
        if let Some(error) =
            backend.terminal_error(device_identity, RuntimeOperation::SurfaceRendering)
        {
            return Err(error);
        }
        Ok(Surface::with_backend(
            attachment,
            options,
            SurfaceBackend::Presented {
                surface: Box::new(
                    super::super::surface::PresentedSurface::display_free_for_test(options.format),
                ),
                device_identity,
                state: super::super::surface::PresentedSurfaceState::new(
                    physical_size,
                    super::super::surface::ResizeState::Idle,
                ),
            },
            self.identity.clone(),
        ))
    }

    pub(crate) async fn resume_display_free_presented_surface_for_test(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
    ) -> Result<()> {
        self.resume_display_free_presented_surface_with_loss_for_test(surface, attachment, None)
            .await
    }

    pub(crate) async fn resume_display_free_presented_surface_after_device_loss_for_test(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
        reason: DeviceLossReason,
    ) -> Result<()> {
        self.resume_display_free_presented_surface_with_loss_for_test(
            surface,
            attachment,
            Some(reason),
        )
        .await
    }

    async fn resume_display_free_presented_surface_with_loss_for_test(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
        loss_before_configuration: Option<DeviceLossReason>,
    ) -> Result<()> {
        if !surface.is_display_free_presented_for_test() {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "display-free resume support requires its own presented fixture",
            ));
        }
        self.validate_surface_renderer_identity(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_operation_backend(surface, RuntimeOperation::SurfaceResume)?;
        self.validate_surface_device_identity(surface, RuntimeOperation::SurfaceResume)?;
        let SurfaceBackend::Presented { state, .. } = &surface.backend else {
            unreachable!("display-free resume support requires a presented surface");
        };
        let action = Surface::presented_resume_action(surface.state, state.lifecycle());
        let resizing = state.lifecycle().resize_state();
        self.validate_surface_device_terminal(surface, RuntimeOperation::SurfaceResume)?;
        surface.ensure_attachment_compatible(&attachment)?;
        match action {
            super::super::surface::PresentedResumeAction::NoOp => Ok(()),
            super::super::surface::PresentedResumeAction::ConfigureExisting => {
                if let Some(reason) = loss_before_configuration {
                    let device_identity = surface
                        .device_identity()
                        .expect("a presented surface must retain its device slot identity");
                    self.signal_device_loss_for_test(device_identity, reason);
                }
                self.configure_presented_surface_if_needed(surface, RuntimeOperation::SurfaceResume)
                    .await
            }
            super::super::surface::PresentedResumeAction::Configure => {
                self.recreate_display_free_presented_surface_for_resume_for_test(
                    surface,
                    attachment,
                    resizing,
                    loss_before_configuration,
                )
                .await
            }
            super::super::surface::PresentedResumeAction::Recreate => {
                self.recreate_display_free_presented_surface_for_resume_for_test(
                    surface,
                    attachment,
                    resizing,
                    loss_before_configuration,
                )
                .await
            }
        }
    }

    async fn recreate_display_free_presented_surface_for_resume_for_test(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
        resizing: super::super::surface::ResizeState,
        loss_before_configuration: Option<DeviceLossReason>,
    ) -> Result<()> {
        let preferred_device = surface
            .device_identity()
            .expect("a presented surface must retain its device slot identity");
        let mut next = self
            .create_display_free_presented_surface_with_configuration_operation_for_test(
                attachment,
                surface.options,
                RuntimeOperation::SurfaceResume,
                Some(preferred_device),
                loss_before_configuration,
            )
            .await?;
        next.last_parameters = surface.last_parameters;
        next.renderer_identity = surface.renderer_identity.clone();
        if let SurfaceBackend::Presented { state, .. } = &mut next.backend {
            state.set_resizing(resizing);
        }
        *surface = next;
        Ok(())
    }

    async fn create_display_free_presented_surface_with_configuration_operation_for_test(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
        configuration_operation: RuntimeOperation,
        preferred_device: Option<DeviceSlotIdentity>,
        loss_before_configuration: Option<DeviceLossReason>,
    ) -> Result<Surface> {
        validate_surface_options(options)?;
        if !matches!(&attachment, Attachment::WebCanvas(_)) {
            return Err(Error::new(
                BackendErrorCode::SurfaceCreateFailed,
                "the display-free presented fixture requires a web-canvas attachment",
            ));
        }
        let physical_size = physical_size(options.size, options.scale)?;
        let backend = self.backend.as_mut().ok_or_else(|| {
            Error::runtime_unavailable(
                RuntimeOperation::AdapterSelection,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
                "display-free presented configuration coverage requires a renderer backend",
            )
        })?;
        let (surface, device_identity) = backend
            .create_display_free_presented_surface_for_test(
                preferred_device,
                configuration_operation,
                options.format,
            )
            .await?;
        if let Some(reason) = loss_before_configuration {
            backend.signal_loss_for_test(device_identity, reason);
        }
        super::ensure_presented_device_available_after_creation(
            backend,
            device_identity,
            configuration_operation,
        )?;
        let mut created = Surface::with_backend(
            attachment,
            options,
            SurfaceBackend::Presented {
                surface: Box::new(surface),
                device_identity,
                state: super::super::surface::PresentedSurfaceState::new(
                    physical_size,
                    super::super::surface::ResizeState::Idle,
                ),
            },
            self.identity.clone(),
        );
        self.configure_presented_surface_if_needed(&mut created, configuration_operation)
            .await?;
        Ok(created)
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
    pub(crate) fn uploaded_images_for_test(&self) -> HashSet<ImageContentIdentity> {
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
    ) -> (Stats, HashSet<ImageContentIdentity>) {
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
