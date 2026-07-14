use super::backend::DeviceSlotIdentity;
use super::{
    BackendErrorCode, Color, Error, PhysicalSize, RenderSurfaceAvailability, Result,
    RuntimeCapabilityUnavailableReason, RuntimeOperation, Size, geometry::physical_size,
    validation::*,
};
use std::sync::Arc;

#[cfg(all(test, feature = "render-window"))]
use std::sync::Mutex;

#[derive(Clone)]
pub(crate) struct RendererIdentity(Arc<()>);

impl RendererIdentity {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub struct Surface {
    pub(crate) attachment: Attachment,
    pub(crate) options: SurfaceOptions,
    pub(crate) state: SurfaceState,
    pub(crate) last_parameters: Option<Parameters>,
    pub(crate) backend: SurfaceBackend,
    pub(crate) renderer_identity: RendererIdentity,
}

impl Surface {
    pub(crate) fn with_backend(
        attachment: Attachment,
        options: SurfaceOptions,
        backend: SurfaceBackend,
        renderer_identity: RendererIdentity,
    ) -> Self {
        Self {
            attachment,
            options,
            state: SurfaceState::Available,
            last_parameters: None,
            backend,
            renderer_identity,
        }
    }

    pub fn resize(&mut self, size: Size, scale: f64) -> Result<()> {
        validate_size(size, "surface size")?;
        validate_positive_f64(scale, "surface scale")?;
        let next = physical_size(size, scale)?;
        self.options.size = size;
        self.options.scale = scale;
        match &mut self.backend {
            SurfaceBackend::ContractOnly { physical_size } => {
                *physical_size = next;
            }
            SurfaceBackend::Headless {
                resources,
                physical_size,
                ..
            } => {
                if *physical_size == next {
                    return Ok(());
                }
                *physical_size = next;
                *resources = HeadlessResources::for_physical_size(next);
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { surface, state, .. } => {
                state.resize_requested(surface.committed_physical_size(), next);
            }
        }
        Ok(())
    }

    pub fn suspend(&mut self) -> Result<()> {
        self.state = SurfaceState::Suspended;
        Ok(())
    }

    pub fn resume(&mut self, attachment: Attachment) -> Result<()> {
        self.ensure_attachment_compatible(&attachment)?;
        #[cfg(any(
            feature = "render-window",
            all(feature = "render-web", target_arch = "wasm32")
        ))]
        if let SurfaceBackend::Presented { .. } = &self.backend {
            return Err(Error::new(
                BackendErrorCode::UnsupportedBackend,
                "presented surfaces must be resumed through Renderer::resume_surface",
            ));
        }
        self.attachment = attachment;
        self.state = SurfaceState::Available;
        Ok(())
    }

    pub(crate) fn ensure_attachment_compatible(&self, attachment: &Attachment) -> Result<()> {
        if self.attachment.kind() == attachment.kind() {
            return Ok(());
        }
        Err(Error::new(
            BackendErrorCode::SurfaceCreateFailed,
            "surface cannot resume with an incompatible attachment",
        ))
    }

    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    pub(crate) fn presented_resume_action(
        state: SurfaceState,
        lifecycle: PresentedLifecycle,
    ) -> PresentedResumeAction {
        if state == SurfaceState::Available && matches!(lifecycle, PresentedLifecycle::Ready { .. })
        {
            PresentedResumeAction::NoOp
        } else if matches!(lifecycle, PresentedLifecycle::Lost) {
            PresentedResumeAction::Recreate
        } else {
            PresentedResumeAction::Configure
        }
    }

    #[must_use]
    pub const fn state(&self) -> SurfaceState {
        self.state
    }

    pub(crate) fn ensure_available(&self, operation: RuntimeOperation) -> Result<()> {
        if self.state == SurfaceState::Suspended {
            return Err(Error::runtime_unavailable(
                operation,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended,
                },
                "surface is suspended",
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_renderable(&self) -> Result<()> {
        let unavailable = match &self.backend {
            SurfaceBackend::ContractOnly { physical_size }
            | SurfaceBackend::Headless { physical_size, .. }
                if physical_size.width() == 0 || physical_size.height() == 0 =>
            {
                Some(RenderSurfaceAvailability::NonRenderable)
            }
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { state, .. } => match state.lifecycle() {
                PresentedLifecycle::NonRenderable { .. } => {
                    Some(RenderSurfaceAvailability::NonRenderable)
                }
                PresentedLifecycle::Occluded { .. } => Some(RenderSurfaceAvailability::Occluded),
                PresentedLifecycle::Lost => Some(RenderSurfaceAvailability::Lost),
                PresentedLifecycle::Ready { .. } | PresentedLifecycle::ResizePending { .. } => None,
            },
            SurfaceBackend::ContractOnly { .. } | SurfaceBackend::Headless { .. } => None,
        };
        match unavailable {
            Some(state) => Err(Error::runtime_unavailable(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable { state },
                "surface is not renderable",
            )),
            None => Ok(()),
        }
    }

    #[must_use]
    pub const fn size(&self) -> Size {
        self.options.size
    }

    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.options.scale
    }

    #[must_use]
    pub const fn physical_size(&self) -> PhysicalSize {
        match &self.backend {
            SurfaceBackend::ContractOnly { physical_size }
            | SurfaceBackend::Headless { physical_size, .. } => *physical_size,
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { state, .. } => state.requested_physical_size(),
        }
    }

    #[must_use]
    pub const fn resource_state(&self) -> SurfaceResourceState {
        match &self.backend {
            SurfaceBackend::ContractOnly { .. } => SurfaceResourceState::ContractOnly,
            SurfaceBackend::Headless { resources, .. } => match resources {
                HeadlessResources::Empty => SurfaceResourceState::Empty,
                HeadlessResources::Pending => SurfaceResourceState::PendingAllocation,
                HeadlessResources::Ready { .. } => SurfaceResourceState::Ready,
            },
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented { .. } => SurfaceResourceState::Presented,
        }
    }

    pub(crate) const fn device_identity(&self) -> Option<DeviceSlotIdentity> {
        match &self.backend {
            SurfaceBackend::ContractOnly { .. } => None,
            SurfaceBackend::Headless {
                device_identity, ..
            } => Some(*device_identity),
            #[cfg(any(
                feature = "render-window",
                all(feature = "render-web", target_arch = "wasm32")
            ))]
            SurfaceBackend::Presented {
                device_identity, ..
            } => Some(*device_identity),
        }
    }

    pub(crate) fn commit_headless_publication(&mut self, publication: HeadlessPublication) {
        let SurfaceBackend::Headless { resources, .. } = &mut self.backend else {
            unreachable!("only a headless surface can commit a headless publication");
        };
        *resources = HeadlessResources::Ready {
            texture: publication.texture,
        };
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceState {
    Available,
    Suspended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceResourceState {
    ContractOnly,
    Empty,
    PendingAllocation,
    Ready,
    Presented,
}

pub(crate) enum SurfaceBackend {
    ContractOnly {
        physical_size: PhysicalSize,
    },
    Headless {
        device_identity: DeviceSlotIdentity,
        resources: HeadlessResources,
        physical_size: PhysicalSize,
    },
    #[cfg(any(
        feature = "render-window",
        all(feature = "render-web", target_arch = "wasm32")
    ))]
    Presented {
        surface: Box<PresentedSurface>,
        device_identity: DeviceSlotIdentity,
        state: PresentedSurfaceState,
    },
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) struct PresentedSurface {
    target: PresentedSurfaceTarget,
    pub(crate) format: wgpu::TextureFormat,
    committed: Option<PresentedResourceBundle>,
}

/// The external host effect owned by a presented surface.
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
enum PresentedSurfaceTarget {
    Host(wgpu::Surface<'static>),
    /// Test-only substitution for the unavailable display-host configure effect.
    ///
    /// Configuration still allocates the production per-surface target bundle
    /// under the real Configure transaction.
    #[cfg(all(test, feature = "render-window"))]
    DisplayFreeHostEffectForTest(Arc<Mutex<DisplayFreePresentedSurfaceStateForTest>>),
}

/// The finite WGPU acquire outcomes exercised by the display-free presentation fixture.
#[cfg(all(test, feature = "render-window"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentedAcquireOutcomeForTest {
    Success,
    Suboptimal,
    Outdated,
    Occluded,
    Timeout,
    Lost,
    Validation,
}

#[cfg(all(test, feature = "render-window"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DisplayFreePresentedSurfaceObservationForTest {
    acquire_count: usize,
    present_count: usize,
    discarded_count: usize,
}

#[cfg(all(test, feature = "render-window"))]
#[derive(Clone)]
pub(crate) struct DisplayFreePresentedSurfaceObservationHandleForTest(
    Arc<Mutex<DisplayFreePresentedSurfaceStateForTest>>,
);

#[cfg(all(test, feature = "render-window"))]
impl DisplayFreePresentedSurfaceObservationHandleForTest {
    pub(crate) fn snapshot_for_test(&self) -> DisplayFreePresentedSurfaceObservationForTest {
        self.0
            .lock()
            .expect("display-free presentation fixture state must remain available")
            .observation
    }
}

#[cfg(all(test, feature = "render-window"))]
impl DisplayFreePresentedSurfaceObservationForTest {
    pub(crate) const fn acquire_count_for_test(self) -> usize {
        self.acquire_count
    }

    pub(crate) const fn present_count_for_test(self) -> usize {
        self.present_count
    }

    pub(crate) const fn discarded_count_for_test(self) -> usize {
        self.discarded_count
    }
}

#[cfg(all(test, feature = "render-window"))]
pub(crate) struct DisplayFreePresentedSurfaceStateForTest {
    next_outcome: PresentedAcquireOutcomeForTest,
    observation: DisplayFreePresentedSurfaceObservationForTest,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) struct PresentedResourceBundle {
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) target_texture: wgpu::Texture,
    pub(crate) target_view: wgpu::TextureView,
    pub(crate) blitter: wgpu::util::TextureBlitter,
    #[cfg(all(test, feature = "render-window"))]
    resource_id: u64,
}

#[must_use = "presented configuration resources must be committed or dropped"]
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) struct PresentedConfigurationDraft {
    resources: PresentedResourceBundle,
}

/// An acquired host frame that returns/discards itself unless it is presented.
#[must_use = "acquired surface textures must be presented or dropped"]
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) enum AcquiredPresentedSurfaceTexture {
    Host(Option<wgpu::SurfaceTexture>),
    #[cfg(all(test, feature = "render-window"))]
    DisplayFree {
        texture: wgpu::Texture,
        state: Arc<Mutex<DisplayFreePresentedSurfaceStateForTest>>,
        presented: bool,
    },
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) enum PresentedSurfaceAcquire {
    Success(AcquiredPresentedSurfaceTexture),
    Suboptimal(AcquiredPresentedSurfaceTexture),
    Outdated,
    Occluded,
    Timeout,
    Lost,
    Validation,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedSurface {
    pub(crate) fn new(surface: wgpu::Surface<'static>, adapter: &wgpu::Adapter) -> Result<Self> {
        let format = surface
            .get_capabilities(adapter)
            .formats
            .into_iter()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                )
            })
            .ok_or_else(|| {
                Error::new(
                    BackendErrorCode::SurfaceCreateFailed,
                    "the selected adapter does not support an Rgba8 or Bgra8 surface format",
                )
            })?;
        Ok(Self {
            target: PresentedSurfaceTarget::Host(surface),
            format,
            committed: None,
        })
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn display_free_for_test() -> Self {
        Self {
            target: PresentedSurfaceTarget::DisplayFreeHostEffectForTest(Arc::new(Mutex::new(
                DisplayFreePresentedSurfaceStateForTest {
                    next_outcome: PresentedAcquireOutcomeForTest::Success,
                    observation: DisplayFreePresentedSurfaceObservationForTest::default(),
                },
            ))),
            format: wgpu::TextureFormat::Rgba8Unorm,
            committed: None,
        }
    }

    pub(crate) fn committed(&self) -> Option<&PresentedResourceBundle> {
        self.committed.as_ref()
    }

    pub(crate) fn committed_physical_size(&self) -> Option<PhysicalSize> {
        self.committed
            .as_ref()
            .map(|resources| PhysicalSize::new(resources.config.width, resources.config.height))
    }

    pub(crate) fn configure_draft(
        &self,
        device: &wgpu::Device,
        physical_size: PhysicalSize,
        present_mode: wgpu::PresentMode,
    ) -> PresentedConfigurationDraft {
        let config = presented_configuration(self.format, physical_size, present_mode);
        match &self.target {
            PresentedSurfaceTarget::Host(surface) => surface.configure(device, &config),
            #[cfg(all(test, feature = "render-window"))]
            PresentedSurfaceTarget::DisplayFreeHostEffectForTest(_) => {}
        }
        PresentedConfigurationDraft {
            resources: PresentedResourceBundle::new(device, config),
        }
    }

    pub(crate) fn acquire_texture(&self, device: &wgpu::Device) -> PresentedSurfaceAcquire {
        #[cfg(not(all(test, feature = "render-window")))]
        let _ = device;
        match &self.target {
            PresentedSurfaceTarget::Host(surface) => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(texture) => PresentedSurfaceAcquire::Success(
                    AcquiredPresentedSurfaceTexture::Host(Some(texture)),
                ),
                wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                    PresentedSurfaceAcquire::Suboptimal(AcquiredPresentedSurfaceTexture::Host(
                        Some(texture),
                    ))
                }
                wgpu::CurrentSurfaceTexture::Outdated => PresentedSurfaceAcquire::Outdated,
                wgpu::CurrentSurfaceTexture::Occluded => PresentedSurfaceAcquire::Occluded,
                wgpu::CurrentSurfaceTexture::Timeout => PresentedSurfaceAcquire::Timeout,
                wgpu::CurrentSurfaceTexture::Lost => PresentedSurfaceAcquire::Lost,
                wgpu::CurrentSurfaceTexture::Validation => PresentedSurfaceAcquire::Validation,
            },
            #[cfg(all(test, feature = "render-window"))]
            PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) => {
                let outcome = {
                    let mut state = state
                        .lock()
                        .expect("display-free presentation fixture state must remain available");
                    let outcome = state.next_outcome;
                    state.next_outcome = PresentedAcquireOutcomeForTest::Success;
                    outcome
                };
                let texture = || {
                    let resources = self.committed.as_ref().expect(
                        "display-free acquire requires committed presented target resources",
                    );
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Surgeist display-free acquired presentation texture"),
                        size: wgpu::Extent3d {
                            width: resources.config.width,
                            height: resources.config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: resources.config.format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    })
                };
                let acquired = || {
                    let mut fixture = state
                        .lock()
                        .expect("display-free presentation fixture state must remain available");
                    fixture.observation.acquire_count =
                        fixture.observation.acquire_count.saturating_add(1);
                    drop(fixture);
                    AcquiredPresentedSurfaceTexture::DisplayFree {
                        texture: texture(),
                        state: Arc::clone(state),
                        presented: false,
                    }
                };
                match outcome {
                    PresentedAcquireOutcomeForTest::Success => {
                        PresentedSurfaceAcquire::Success(acquired())
                    }
                    PresentedAcquireOutcomeForTest::Suboptimal => {
                        PresentedSurfaceAcquire::Suboptimal(acquired())
                    }
                    PresentedAcquireOutcomeForTest::Outdated => PresentedSurfaceAcquire::Outdated,
                    PresentedAcquireOutcomeForTest::Occluded => PresentedSurfaceAcquire::Occluded,
                    PresentedAcquireOutcomeForTest::Timeout => PresentedSurfaceAcquire::Timeout,
                    PresentedAcquireOutcomeForTest::Lost => PresentedSurfaceAcquire::Lost,
                    PresentedAcquireOutcomeForTest::Validation => {
                        PresentedSurfaceAcquire::Validation
                    }
                }
            }
        }
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn set_acquire_outcome_for_test(&mut self, outcome: PresentedAcquireOutcomeForTest) {
        let PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) = &self.target else {
            panic!("only the display-free presented fixture accepts synthetic acquire outcomes");
        };
        state
            .lock()
            .expect("display-free presentation fixture state must remain available")
            .next_outcome = outcome;
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn observation_for_test(&self) -> DisplayFreePresentedSurfaceObservationForTest {
        self.observation_handle_for_test().snapshot_for_test()
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn observation_handle_for_test(
        &self,
    ) -> DisplayFreePresentedSurfaceObservationHandleForTest {
        let PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) = &self.target else {
            panic!("only the display-free presented fixture exposes presentation observations");
        };
        DisplayFreePresentedSurfaceObservationHandleForTest(Arc::clone(state))
    }

    pub(crate) fn commit_configuration(&mut self, draft: PresentedConfigurationDraft) {
        self.committed = Some(draft.resources);
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl AcquiredPresentedSurfaceTexture {
    pub(crate) fn create_view(&self) -> wgpu::TextureView {
        match self {
            Self::Host(texture) => texture
                .as_ref()
                .expect("an unpresented host surface texture must remain available")
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            #[cfg(all(test, feature = "render-window"))]
            Self::DisplayFree { texture, .. } => {
                texture.create_view(&wgpu::TextureViewDescriptor::default())
            }
        }
    }

    pub(crate) fn present(mut self) {
        match &mut self {
            Self::Host(texture) => texture
                .take()
                .expect("a host surface texture is presented at most once")
                .present(),
            #[cfg(all(test, feature = "render-window"))]
            Self::DisplayFree {
                state, presented, ..
            } => {
                *presented = true;
                let mut state = state
                    .lock()
                    .expect("display-free presentation fixture state must remain available");
                state.observation.present_count = state.observation.present_count.saturating_add(1);
            }
        }
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl Drop for AcquiredPresentedSurfaceTexture {
    fn drop(&mut self) {
        #[cfg(all(test, feature = "render-window"))]
        if let Self::DisplayFree {
            state, presented, ..
        } = self
            && !*presented
        {
            let mut state = state
                .lock()
                .expect("display-free presentation fixture state must remain available");
            state.observation.discarded_count = state.observation.discarded_count.saturating_add(1);
        }
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedResourceBundle {
    fn new(device: &wgpu::Device, config: wgpu::SurfaceConfiguration) -> Self {
        let physical_size = PhysicalSize::new(config.width, config.height);
        let format = config.format;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: physical_size.width(),
                height: physical_size.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            config,
            target_texture: texture,
            target_view: view,
            blitter: wgpu::util::TextureBlitter::new(device, format),
            #[cfg(all(test, feature = "render-window"))]
            resource_id: NEXT_PRESENTED_RESOURCE_ID
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[cfg(all(test, feature = "render-window"))]
impl PresentedResourceBundle {
    pub(crate) const fn resource_id_for_test(&self) -> u64 {
        self.resource_id
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
fn presented_configuration(
    format: wgpu::TextureFormat,
    physical_size: PhysicalSize,
    present_mode: wgpu::PresentMode,
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: physical_size.width(),
        height: physical_size.height(),
        present_mode,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    }
}

#[cfg(all(test, feature = "render-window"))]
static NEXT_PRESENTED_RESOURCE_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

pub(crate) enum HeadlessResources {
    Empty,
    Pending,
    Ready { texture: wgpu::Texture },
}

#[must_use = "headless frame publications must be committed or dropped"]
pub(crate) struct HeadlessPublication {
    pub(crate) texture: wgpu::Texture,
}

impl HeadlessPublication {
    pub(crate) const fn new(texture: wgpu::Texture) -> Self {
        Self { texture }
    }
}

impl HeadlessResources {
    pub(crate) const fn for_physical_size(physical_size: PhysicalSize) -> Self {
        if physical_size.width() == 0 || physical_size.height() == 0 {
            Self::Empty
        } else {
            Self::Pending
        }
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResizeState {
    Idle,
    Resizing,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentedLifecycle {
    Ready {
        resizing: ResizeState,
    },
    ResizePending {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    NonRenderable {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    Occluded {
        resizing: ResizeState,
    },
    Lost,
}

/// Requested presented extent/lifecycle kept independently from optional committed resources.
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
pub(crate) struct PresentedSurfaceState {
    requested_physical_size: PhysicalSize,
    lifecycle: PresentedLifecycle,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedSurfaceState {
    pub(crate) fn new(physical_size: PhysicalSize, resizing: ResizeState) -> Self {
        let lifecycle = if physical_size.width() == 0 || physical_size.height() == 0 {
            PresentedLifecycle::NonRenderable {
                physical_size,
                resizing,
            }
        } else {
            PresentedLifecycle::ResizePending {
                physical_size,
                resizing,
            }
        };
        Self {
            requested_physical_size: physical_size,
            lifecycle,
        }
    }

    pub(crate) const fn lifecycle(&self) -> PresentedLifecycle {
        self.lifecycle
    }

    pub(crate) const fn requested_physical_size(&self) -> PhysicalSize {
        self.requested_physical_size
    }

    pub(crate) const fn needs_configuration(&self) -> bool {
        matches!(self.lifecycle, PresentedLifecycle::ResizePending { .. })
    }

    pub(crate) fn resize_requested(
        &mut self,
        committed_physical_size: Option<PhysicalSize>,
        next: PhysicalSize,
    ) {
        let resizing = self.lifecycle.resize_state();
        self.requested_physical_size = next;
        self.lifecycle = if next.width() == 0 || next.height() == 0 {
            PresentedLifecycle::NonRenderable {
                physical_size: next,
                resizing,
            }
        } else if self.lifecycle.physical_size() == Some(next) {
            self.lifecycle
        } else if committed_physical_size == Some(next) {
            PresentedLifecycle::Ready { resizing }
        } else {
            PresentedLifecycle::ResizePending {
                physical_size: next,
                resizing,
            }
        };
    }

    pub(crate) fn commit_configuration(&mut self) {
        let resizing = self.lifecycle.resize_state();
        debug_assert!(self.requested_physical_size.width() > 0);
        debug_assert!(self.requested_physical_size.height() > 0);
        self.lifecycle = PresentedLifecycle::Ready { resizing };
    }

    pub(crate) fn mark_configuration_pending(&mut self) {
        if self.requested_physical_size.width() > 0 && self.requested_physical_size.height() > 0 {
            self.lifecycle = PresentedLifecycle::ResizePending {
                physical_size: self.requested_physical_size,
                resizing: self.lifecycle.resize_state(),
            };
        }
    }

    pub(crate) fn mark_occluded(&mut self) {
        self.lifecycle = PresentedLifecycle::Occluded {
            resizing: self.lifecycle.resize_state(),
        };
    }

    pub(crate) fn mark_lost(&mut self) {
        self.lifecycle = PresentedLifecycle::Lost;
    }

    pub(crate) fn set_resizing(&mut self, resizing: ResizeState) {
        self.lifecycle = self.lifecycle.with_resizing(resizing);
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentedResumeAction {
    NoOp,
    Configure,
    Recreate,
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl PresentedLifecycle {
    pub(crate) const fn resizing(self) -> bool {
        matches!(
            self,
            Self::Ready {
                resizing: ResizeState::Resizing
            } | Self::ResizePending {
                resizing: ResizeState::Resizing,
                ..
            } | Self::NonRenderable {
                resizing: ResizeState::Resizing,
                ..
            } | Self::Occluded {
                resizing: ResizeState::Resizing
            }
        )
    }

    pub(crate) const fn resize_state(self) -> ResizeState {
        if self.resizing() {
            ResizeState::Resizing
        } else {
            ResizeState::Idle
        }
    }

    pub(crate) const fn physical_size(self) -> Option<PhysicalSize> {
        match self {
            Self::ResizePending { physical_size, .. }
            | Self::NonRenderable { physical_size, .. } => Some(physical_size),
            Self::Ready { .. } | Self::Occluded { .. } | Self::Lost => None,
        }
    }

    pub(crate) const fn with_resizing(self, resizing: ResizeState) -> Self {
        match self {
            Self::Ready { .. } => Self::Ready { resizing },
            Self::ResizePending { physical_size, .. } => Self::ResizePending {
                physical_size,
                resizing,
            },
            Self::NonRenderable { physical_size, .. } => Self::NonRenderable {
                physical_size,
                resizing,
            },
            Self::Occluded { .. } => Self::Occluded { resizing },
            Self::Lost => Self::Lost,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Attachment {
    Headless,
    #[cfg(feature = "render-window")]
    Window(surgeist_window::Handle),
    WebCanvas(WebCanvas),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttachmentKind {
    Headless,
    #[cfg(feature = "render-window")]
    Window,
    WebCanvas,
}

#[derive(Clone, Debug)]
pub struct WebCanvas {
    id: String,
    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    canvas: Option<wgpu::web_sys::HtmlCanvasElement>,
}

impl Attachment {
    #[must_use]
    pub fn from_web_canvas(id: impl Into<String>) -> Self {
        Self::WebCanvas(WebCanvas::new(id))
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    #[must_use]
    pub fn from_html_canvas(
        id: impl Into<String>,
        canvas: wgpu::web_sys::HtmlCanvasElement,
    ) -> Self {
        Self::WebCanvas(WebCanvas::from_html_canvas(id, canvas))
    }

    #[cfg(feature = "render-window")]
    #[must_use]
    pub fn from_window(handle: surgeist_window::Handle) -> Self {
        Self::Window(handle)
    }

    pub(crate) const fn kind(&self) -> AttachmentKind {
        match self {
            Self::Headless => AttachmentKind::Headless,
            #[cfg(feature = "render-window")]
            Self::Window(_) => AttachmentKind::Window,
            Self::WebCanvas(_) => AttachmentKind::WebCanvas,
        }
    }
}

impl WebCanvas {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
            canvas: None,
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    #[must_use]
    pub fn from_html_canvas(
        id: impl Into<String>,
        canvas: wgpu::web_sys::HtmlCanvasElement,
    ) -> Self {
        Self {
            id: id.into(),
            canvas: Some(canvas),
        }
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    pub(crate) fn html_canvas(&self) -> Option<wgpu::web_sys::HtmlCanvasElement> {
        self.canvas.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceOptions {
    pub size: Size,
    pub scale: f64,
    pub present_mode: PresentMode,
    pub format: Format,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        Self {
            size: Size::new(1.0, 1.0),
            scale: 1.0,
            present_mode: PresentMode::Auto,
            format: Format::Rgba8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PresentMode {
    #[default]
    Auto,
    Fifo,
    Mailbox,
    Immediate,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Format {
    #[default]
    Rgba8,
    Bgra8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    pub base_color: Color,
    pub debug: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            base_color: Color::TRANSPARENT,
            debug: false,
        }
    }
}

impl From<PresentMode> for wgpu::PresentMode {
    fn from(mode: PresentMode) -> Self {
        match mode {
            PresentMode::Auto => Self::AutoVsync,
            PresentMode::Fifo => Self::Fifo,
            PresentMode::Mailbox => Self::Mailbox,
            PresentMode::Immediate => Self::Immediate,
        }
    }
}

impl From<Format> for wgpu::TextureFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::Rgba8 => Self::Rgba8Unorm,
            Format::Bgra8 => Self::Bgra8Unorm,
        }
    }
}
