#[cfg(test)]
use super::{Backend, DeviceSlotIdentity, InternalVelloRenderRequest, RenderTimings};
#[cfg(test)]
use crate::{
    BackendErrorCode, Error, Format, Options, Parameters, PhysicalSize, Result,
    RuntimeCapabilityUnavailableReason, RuntimeOperation,
    command::OffscreenBounds,
    geometry::physical_size,
    gpu_transaction::GpuOperationStage,
    resource::{FrameResourceScope, ResourceAccountingFault, ResourceIdentity, ResourceLease},
    texture::EffectTextureDescriptor,
    vello_engine::scene::VelloScene,
};
#[cfg(test)]
use std::{
    cell::RefCell,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(test)]
thread_local! {
    static ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST: RefCell<Option<Arc<AtomicUsize>>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct ScopedOffscreenTextureAcquireObservationForTest {
    count: Arc<AtomicUsize>,
    previous: Option<Arc<AtomicUsize>>,
}

#[cfg(test)]
impl ScopedOffscreenTextureAcquireObservationForTest {
    pub(crate) fn begin() -> Self {
        let count = Arc::new(AtomicUsize::new(0));
        let previous = ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST
            .with(|active| active.replace(Some(Arc::clone(&count))));
        Self { count, previous }
    }

    pub(crate) fn acquire_count_for_test(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl Drop for ScopedOffscreenTextureAcquireObservationForTest {
    fn drop(&mut self) {
        ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
fn record_offscreen_texture_acquire_for_test() {
    ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST.with(|active| {
        if let Some(count) = active.borrow().as_ref() {
            count.fetch_add(1, Ordering::Relaxed);
        }
    });
}

#[cfg(test)]
pub(crate) struct OffscreenRenderGpuContext<'a> {
    backend: &'a mut Backend,
    device_identity: DeviceSlotIdentity,
}

#[cfg(test)]
impl<'a> OffscreenRenderGpuContext<'a> {
    #[must_use]
    pub(crate) fn new(backend: &'a mut Backend, device_identity: DeviceSlotIdentity) -> Self {
        Self {
            backend,
            device_identity,
        }
    }
}

/// Describes a Vello scene that has already been encoded in offscreen-local
/// coordinates; bounds size allocates the target texture, not a scene crop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(test)]
pub(crate) struct OffscreenLocalSceneRenderRequest {
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
    parameters: Parameters,
}

#[cfg(test)]
impl OffscreenLocalSceneRenderRequest {
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn new(
        bounds: OffscreenBounds,
        scale: f64,
        format: Format,
        parameters: Parameters,
    ) -> Self {
        Self {
            bounds,
            scale,
            format,
            parameters,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(test)]
pub(crate) struct OffscreenRenderTarget {
    #[cfg(test)]
    resource_identity: ResourceIdentity,
    #[cfg(test)]
    bounds: OffscreenBounds,
    descriptor: EffectTextureDescriptor,
}

#[cfg(test)]
impl OffscreenRenderTarget {
    fn new(
        _resource_identity: ResourceIdentity,
        _bounds: OffscreenBounds,
        descriptor: EffectTextureDescriptor,
    ) -> Self {
        Self {
            #[cfg(test)]
            resource_identity: _resource_identity,
            #[cfg(test)]
            bounds: _bounds,
            descriptor,
        }
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn resource_id(self) -> u64 {
        self.resource_identity.get()
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn bounds(self) -> OffscreenBounds {
        self.bounds
    }

    #[must_use]
    pub(crate) const fn descriptor(self) -> EffectTextureDescriptor {
        self.descriptor
    }
}

#[must_use = "offscreen rendered texture leases must be resolved by their device resource frame"]
#[cfg(test)]
pub(crate) struct OffscreenRenderedTextureLease {
    target: OffscreenRenderTarget,
    frame_scope: Option<FrameResourceScope>,
    resource: Option<ResourceLease>,
    timings: RenderTimings,
}

#[cfg(test)]
impl fmt::Debug for OffscreenRenderedTextureLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OffscreenRenderedTextureLease")
            .field("target", &self.target)
            .field("timings", &self.timings)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl OffscreenRenderedTextureLease {
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn target(&self) -> OffscreenRenderTarget {
        self.target
    }

    pub(crate) fn texture(&self) -> Result<&wgpu::Texture> {
        self.managed_texture().map(|(texture, _)| texture)
    }

    pub(crate) fn view(&self) -> Result<&wgpu::TextureView> {
        self.managed_texture().map(|(_, view)| view)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) const fn timings(&self) -> RenderTimings {
        self.timings
    }

    #[cfg(test)]
    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        self.frame_scope
            .as_ref()
            .expect("an unresolved offscreen lease must own its resource frame")
            .poison_retained_byte_accounting_for_test()
    }

    pub(crate) fn release(mut self) -> Result<()> {
        let mut frame_scope = self
            .frame_scope
            .take()
            .expect("an unresolved offscreen lease must own its resource frame");
        let resource = self
            .resource
            .take()
            .expect("an unresolved offscreen lease must own its resource lease");
        frame_scope.ensure_commit_ready(&[&resource])?;
        frame_scope.release(resource)?;
        let _ = frame_scope.finish_checked()?;
        Ok(())
    }

    fn managed_texture(&self) -> Result<(&wgpu::Texture, &wgpu::TextureView)> {
        let frame_scope = self.frame_scope.as_ref().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen texture resource frame was already resolved",
            )
        })?;
        let resource = self.resource.as_ref().ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen texture resource lease was already resolved",
            )
        })?;
        frame_scope.effect_texture(resource)
    }

    fn discard(&mut self) {
        let Some(mut frame_scope) = self.frame_scope.take() else {
            return;
        };
        if let Some(resource) = self.resource.take() {
            let _ = frame_scope.discard(resource);
        }
        let _ = frame_scope.finish();
    }
}

#[cfg(test)]
impl Drop for OffscreenRenderedTextureLease {
    fn drop(&mut self) {
        self.discard();
    }
}

#[cfg(test)]
pub(crate) fn offscreen_local_scene_texture_descriptor(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<EffectTextureDescriptor> {
    let physical_size = offscreen_local_scene_physical_size(bounds, scale, format)?;
    offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, format)
}

#[cfg(test)]
fn offscreen_local_scene_physical_size(
    bounds: OffscreenBounds,
    scale: f64,
    format: Format,
) -> Result<PhysicalSize> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    physical_size(bounds.rect().size(), scale)
}

#[cfg(test)]
fn offscreen_local_scene_texture_descriptor_for_physical_size(
    physical_size: PhysicalSize,
    format: Format,
) -> Result<EffectTextureDescriptor> {
    if format != Format::Rgba8 {
        return Err(Error::invalid_value(
            "offscreen Vello scene texture format",
            format!("{format:?}"),
            "must be Rgba8 for minimal offscreen Vello targets",
        ));
    }
    EffectTextureDescriptor::try_capture(
        physical_size,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            .union(wgpu::TextureUsages::STORAGE_BINDING)
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST),
    )
}

#[cfg(test)]
pub(crate) async fn render_internal_vello_local_scene_to_offscreen_texture(
    context: Option<OffscreenRenderGpuContext<'_>>,
    options: Options,
    scene: &VelloScene,
    request: OffscreenLocalSceneRenderRequest,
) -> Result<OffscreenRenderedTextureLease> {
    let physical_size =
        offscreen_local_scene_physical_size(request.bounds, request.scale, request.format)?;
    let Some(context) = context else {
        offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, request.format)?;
        return Err(Error::runtime_unavailable(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            "offscreen Vello local scene rendering requires an available wgpu device context",
        ));
    };
    let capabilities = context
        .backend
        .device_capabilities(context.device_identity)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::RenderFailed,
                "offscreen device capabilities are unavailable before allocation",
            )
        })?;
    capabilities.validate_effect_texture_extent(physical_size)?;
    let descriptor =
        offscreen_local_scene_texture_descriptor_for_physical_size(physical_size, request.format)?;
    let mut rendered = {
        let ready = context.backend.ready_state_mut(
            context.device_identity,
            RuntimeOperation::SurfaceRendering,
            BackendErrorCode::RenderFailed,
            "offscreen device resources are unavailable before allocation",
        )?;
        #[cfg(test)]
        record_offscreen_texture_acquire_for_test();
        let mut frame_scope = ready.resources.begin_frame()?;
        let resource =
            frame_scope.acquire_effect_texture(&ready.device, &capabilities, descriptor)?;
        let target =
            OffscreenRenderTarget::new(resource.resource_identity(), request.bounds, descriptor);
        OffscreenRenderedTextureLease {
            target,
            frame_scope: Some(frame_scope),
            resource: Some(resource),
            timings: RenderTimings::default(),
        }
    };
    let render_start = Instant::now();
    let transaction = context.backend.begin_gpu_operation(
        context.device_identity,
        GpuOperationStage::Render,
        RuntimeOperation::SurfaceRendering,
    )?;
    let result = context
        .backend
        .render_internal_vello_to_texture(
            transaction,
            InternalVelloRenderRequest {
                identity: context.device_identity,
                operation: RuntimeOperation::SurfaceRendering,
                scene,
                target: rendered.view()?,
                target_extent: rendered.target.descriptor().physical_size(),
                base_color: request.parameters.base_color,
                antialiasing: options.antialiasing(),
                target_usage: rendered.target.descriptor().usage(),
            },
        )
        .await;
    context
        .backend
        .observe_device_terminal(context.device_identity);
    result?;
    rendered.timings = RenderTimings {
        render_time: render_start.elapsed(),
        present_time: Duration::ZERO,
    };
    Ok(rendered)
}
