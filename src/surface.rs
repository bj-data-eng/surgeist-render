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

/// Runtime surface identity, lifecycle, and complete-publication state.
///
/// A surface belongs to the [`crate::Renderer`] that created it and to one
/// device generation. Foreign or stale use is a typed failure. The render crate
/// owns GPU resources and failure-atomic publication; an application host owns
/// native-window or browser host lifecycle and supplies compatible attachments.
pub struct Surface {
    pub(crate) attachment: Attachment,
    pub(crate) options: SurfaceOptions,
    pub(crate) state: SurfaceState,
    pub(crate) last_parameters: Option<Parameters>,
    pub(crate) backend: SurfaceBackend,
    pub(crate) renderer_identity: RendererIdentity,
    #[cfg(test)]
    headless_publication_count: usize,
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
            #[cfg(test)]
            headless_publication_count: 0,
        }
    }

    /// Updates requested logical size and logical-to-physical scale.
    ///
    /// Invalid or overflowing values fail without changing the request. A
    /// size-changing headless resize discards its readable publication; a
    /// presented resize is committed by the next renderer-owned host operation.
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

    /// Marks the surface suspended without discarding committed resources.
    ///
    /// Repeating this transition is idempotent. Rendering and readback reject a
    /// suspended surface until a compatible resume succeeds.
    pub fn suspend(&mut self) -> Result<()> {
        self.state = SurfaceState::Suspended;
        Ok(())
    }

    /// Resumes a non-presented surface with the same attachment kind.
    ///
    /// Presented host resources must be resumed asynchronously through
    /// [`crate::Renderer::resume_surface`]. An incompatible attachment fails
    /// without changing the committed lifecycle.
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
        match (state, lifecycle) {
            (
                SurfaceState::Available,
                PresentedLifecycle::Ready { .. }
                | PresentedLifecycle::NonRenderable { .. }
                | PresentedLifecycle::Occluded { .. },
            ) => PresentedResumeAction::NoOp,
            (SurfaceState::Available, PresentedLifecycle::ResizePending { .. }) => {
                PresentedResumeAction::ConfigureExisting
            }
            (_, PresentedLifecycle::Lost) => PresentedResumeAction::Recreate,
            (
                SurfaceState::Suspended,
                PresentedLifecycle::Ready { .. }
                | PresentedLifecycle::ResizePending { .. }
                | PresentedLifecycle::NonRenderable { .. }
                | PresentedLifecycle::Occluded { .. },
            ) => PresentedResumeAction::Configure,
        }
    }

    #[must_use]
    /// Returns the caller-visible available or suspended lifecycle state.
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
    /// Returns the requested logical size.
    pub const fn size(&self) -> Size {
        self.options.size
    }

    #[must_use]
    /// Returns the positive logical-to-physical pixel scale.
    pub const fn scale(&self) -> f64 {
        self.options.scale
    }

    #[must_use]
    /// Returns the requested physical pixel extent.
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
    /// Returns the current public resource/publication phase.
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
        #[cfg(test)]
        {
            self.headless_publication_count = self.headless_publication_count.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) const fn headless_publication_count_for_test(&self) -> usize {
        self.headless_publication_count
    }
}

/// Caller-visible surface lifecycle independent of backend resource phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceState {
    /// Operations may proceed when runtime capabilities also permit them.
    Available,
    /// Rendering and readback are rejected until a compatible resume succeeds.
    Suspended,
}

/// Runtime-phase resource and publication state for a headless or presented surface.
///
/// This observation does not expose backend resources. In particular,
/// [`Self::Ready`] means a complete headless publication is readable, while
/// [`Self::Presented`] remains subject to the external host lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceResourceState {
    /// No compatible GPU adapter was available when the contract-only surface was created.
    ContractOnly,
    /// A zero-area headless surface requires no GPU texture.
    Empty,
    /// A nonzero headless surface has no complete published frame yet.
    PendingAllocation,
    /// A headless surface owns a complete readable publication.
    Ready,
    /// The surface is backed by host-presented resources.
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
    acquire_attempt_count: usize,
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
    pub(crate) const fn acquire_attempt_count_for_test(self) -> usize {
        self.acquire_attempt_count
    }

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
    target_identity: u64,
    configuration_count: usize,
    next_outcome: PresentedAcquireOutcomeForTest,
    observation: DisplayFreePresentedSurfaceObservationForTest,
    last_presented_texture: Option<wgpu::Texture>,
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
        texture: Option<wgpu::Texture>,
        state: Arc<Mutex<DisplayFreePresentedSurfaceStateForTest>>,
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
    pub(crate) fn display_free_for_test(format: Format) -> Self {
        Self {
            target: PresentedSurfaceTarget::DisplayFreeHostEffectForTest(Arc::new(Mutex::new(
                DisplayFreePresentedSurfaceStateForTest {
                    target_identity: NEXT_DISPLAY_FREE_PRESENTED_TARGET_ID
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    configuration_count: 0,
                    next_outcome: PresentedAcquireOutcomeForTest::Success,
                    observation: DisplayFreePresentedSurfaceObservationForTest::default(),
                    last_presented_texture: None,
                },
            ))),
            format: format.into(),
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
            PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) => {
                let mut state = state
                    .lock()
                    .expect("display-free presentation fixture state must remain available");
                state.configuration_count = state.configuration_count.saturating_add(1);
            }
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
                    state.observation.acquire_attempt_count =
                        state.observation.acquire_attempt_count.saturating_add(1);
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
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC,
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
                        texture: Some(texture()),
                        state: Arc::clone(state),
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
    pub(crate) fn configuration_count_for_test(&self) -> usize {
        let PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) = &self.target else {
            panic!("only the display-free presented fixture exposes configuration observations");
        };
        state
            .lock()
            .expect("display-free presentation fixture state must remain available")
            .configuration_count
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

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn target_identity_for_test(&self) -> u64 {
        let PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) = &self.target else {
            panic!("only the display-free presented fixture exposes its target identity");
        };
        state
            .lock()
            .expect("display-free presentation fixture state must remain available")
            .target_identity
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn take_last_presented_texture_for_test(&mut self) -> Option<wgpu::Texture> {
        let PresentedSurfaceTarget::DisplayFreeHostEffectForTest(state) = &self.target else {
            panic!("only the display-free presented fixture retains presented textures");
        };
        state
            .lock()
            .expect("display-free presentation fixture state must remain available")
            .last_presented_texture
            .take()
    }

    pub(crate) fn commit_configuration(&mut self, draft: PresentedConfigurationDraft) {
        self.committed = Some(draft.resources);
    }

    #[cfg(all(test, feature = "render-window"))]
    pub(crate) fn is_display_free_for_test(&self) -> bool {
        matches!(
            self.target,
            PresentedSurfaceTarget::DisplayFreeHostEffectForTest(_)
        )
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
            Self::DisplayFree { texture, .. } => texture
                .as_ref()
                .expect("an unpresented display-free texture must remain available")
                .create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    pub(crate) fn present(mut self) {
        match &mut self {
            Self::Host(texture) => texture
                .take()
                .expect("a host surface texture is presented at most once")
                .present(),
            #[cfg(all(test, feature = "render-window"))]
            Self::DisplayFree { texture, state } => {
                let texture = texture
                    .take()
                    .expect("a display-free surface texture is presented at most once");
                let mut state = state
                    .lock()
                    .expect("display-free presentation fixture state must remain available");
                state.last_presented_texture = Some(texture);
                state.observation.present_count = state.observation.present_count.saturating_add(1);
            }
        }
    }
}

#[cfg(all(test, feature = "render-window"))]
impl Surface {
    pub(crate) fn is_display_free_presented_for_test(&self) -> bool {
        matches!(
            &self.backend,
            SurfaceBackend::Presented { surface, .. } if surface.is_display_free_for_test()
        )
    }
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
impl Drop for AcquiredPresentedSurfaceTexture {
    fn drop(&mut self) {
        #[cfg(all(test, feature = "render-window"))]
        if let Self::DisplayFree { texture, state } = self
            && texture.is_some()
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

#[cfg(all(test, feature = "render-window"))]
static NEXT_DISPLAY_FREE_PRESENTED_TARGET_ID: std::sync::atomic::AtomicU64 =
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
        self.requested_physical_size = next;
        if matches!(self.lifecycle, PresentedLifecycle::Lost) {
            return;
        }
        let resizing = self.lifecycle.resize_state();
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
    ConfigureExisting,
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

/// Host attachment requested for a surface.
#[derive(Clone, Debug)]
pub enum Attachment {
    /// Offscreen GPU rendering with pixels available only through explicit readback.
    Headless,
    #[cfg(feature = "render-window")]
    /// Native presented host supplied by `surgeist-window`.
    Window(surgeist_window::Handle),
    /// Browser-canvas descriptor or handle, depending on target and feature.
    WebCanvas(WebCanvas),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttachmentKind {
    Headless,
    #[cfg(feature = "render-window")]
    Window,
    WebCanvas,
}

/// Browser-host canvas attachment boundary.
///
/// A real canvas handle exists only on `wasm32` with `render-web` and must be
/// supplied by the browser host through `WebCanvas::from_html_canvas`. On native
/// targets, or for identifier-only construction, creation remains a typed
/// platform diagnostic rather than browser execution evidence.
#[derive(Clone, Debug)]
pub struct WebCanvas {
    id: String,
    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    canvas: Option<wgpu::web_sys::HtmlCanvasElement>,
}

impl Attachment {
    /// Creates an identifier-only web-canvas attachment.
    ///
    /// This does not create or discover a browser canvas handle.
    #[must_use]
    pub fn from_web_canvas(id: impl Into<String>) -> Self {
        Self::WebCanvas(WebCanvas::new(id))
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    /// Wraps the browser-host canvas used by WebGPU presentation.
    #[must_use]
    pub fn from_html_canvas(
        id: impl Into<String>,
        canvas: wgpu::web_sys::HtmlCanvasElement,
    ) -> Self {
        Self::WebCanvas(WebCanvas::from_html_canvas(id, canvas))
    }

    #[cfg(feature = "render-window")]
    /// Wraps a live native-window handle supplied by the application host.
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
    /// Creates an identifier-only canvas descriptor with no browser handle.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
            canvas: None,
        }
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    /// Creates a canvas attachment from a browser-host `HtmlCanvasElement`.
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
    /// Returns the caller-provided diagnostic identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    #[cfg(all(feature = "render-web", target_arch = "wasm32"))]
    pub(crate) fn html_canvas(&self) -> Option<wgpu::web_sys::HtmlCanvasElement> {
        self.canvas.clone()
    }
}

/// Surface creation options.
///
/// [`Default`] requests one logical unit at scale 1, automatic presentation,
/// and [`Format::Rgba8`]. Size and scale validation occurs during creation;
/// headless surfaces reject [`Format::Bgra8`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceOptions {
    /// Requested logical dimensions.
    pub size: Size,
    /// Positive logical-to-physical pixel scale.
    pub scale: f64,
    /// Requested host presentation policy.
    pub present_mode: PresentMode,
    /// Requested surface pixel format.
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

/// Requested host presentation scheduling mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PresentMode {
    /// Backend-selected vertical-sync policy; the default.
    #[default]
    Auto,
    /// First-in, first-out presentation.
    Fifo,
    /// Mailbox presentation when the host supports it.
    Mailbox,
    /// Immediate presentation when the host supports it.
    Immediate,
}

/// Public surface color format.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Format {
    /// Straight-alpha RGBA8 at public upload/readback boundaries; the default.
    #[default]
    Rgba8,
    /// BGRA8 presented-surface format; unsupported for headless surfaces.
    Bgra8,
}

/// Per-frame render parameters.
///
/// [`Default`] clears to transparent and disables diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parameters {
    /// Base color used to initialize the frame before authored commands.
    pub base_color: Color,
    /// Enables renderer diagnostics for this frame.
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
