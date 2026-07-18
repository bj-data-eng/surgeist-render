// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::{HashMap, HashSet};

use crate::{
    BackendErrorCode, Error, PhysicalSize, Result,
    gpu_transaction::VelloResourceCommitProof,
    resource::{FrameResourceScope, ResourceLease, ResourceManager},
};

use super::encoder::ActiveVelloEncodingScope;
use super::{
    BufferHandle, BufferIntent, BufferRole, ImageHandle, ImageIntent, ImageRetention, ImageRole,
    RasterImageFormat, ResourceIntent, ResourceReference,
};

#[cfg(test)]
use super::RecordingBuilder;

struct ManagedBuffer {
    lease: ResourceLease,
    byte_len: u64,
}

struct ManagedImage {
    lease: ResourceLease,
    extent: PhysicalSize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct VelloBufferKey {
    role: BufferRole,
    byte_len: u64,
    usage: wgpu::BufferUsages,
}

impl VelloBufferKey {
    fn from_intent(intent: &BufferIntent) -> Self {
        Self {
            role: intent.role,
            byte_len: intent.byte_len,
            usage: buffer_usage(intent.role),
        }
    }

    pub(crate) const fn byte_len(self) -> u64 {
        self.byte_len
    }

    pub(crate) const fn usage(self) -> wgpu::BufferUsages {
        self.usage
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct VelloImageKey {
    role: ImageRole,
    extent: PhysicalSize,
    format: RasterImageFormat,
    retention: ImageRetention,
    usage: wgpu::TextureUsages,
}

impl VelloImageKey {
    fn from_intent(intent: &ImageIntent) -> Self {
        Self {
            role: intent.role,
            extent: intent.extent,
            format: intent.format,
            retention: intent.retention,
            usage: image_usage(),
        }
    }

    pub(crate) const fn extent(self) -> PhysicalSize {
        self.extent
    }

    pub(crate) const fn texture_format(self) -> wgpu::TextureFormat {
        texture_format(self.format)
    }

    pub(crate) const fn usage(self) -> wgpu::TextureUsages {
        self.usage
    }

    pub(crate) const fn is_persistent_atlas(self) -> bool {
        matches!(
            (self.role, self.retention),
            (ImageRole::ImageAtlas, ImageRetention::PersistentImageAtlas)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloAtlasOutcome {
    NoAtlas,
    #[cfg(test)]
    Retain,
    MarkDirty,
    Recreate,
}

impl VelloAtlasOutcome {
    pub(crate) const fn merge_pending_recovery(
        pending: Option<Self>,
        latest: Self,
    ) -> Option<Self> {
        match (pending, latest) {
            (Some(Self::Recreate), _) | (Some(Self::MarkDirty), Self::Recreate) => {
                Some(Self::Recreate)
            }
            (Some(Self::MarkDirty), _) => Some(Self::MarkDirty),
            #[cfg(test)]
            (Some(Self::NoAtlas | Self::Retain) | None, Self::MarkDirty | Self::Recreate) => {
                Some(latest)
            }
            #[cfg(not(test))]
            (Some(Self::NoAtlas) | None, Self::MarkDirty | Self::Recreate) => Some(latest),
            #[cfg(test)]
            (Some(Self::NoAtlas | Self::Retain) | None, Self::NoAtlas | Self::Retain) => None,
            #[cfg(not(test))]
            (Some(Self::NoAtlas) | None, Self::NoAtlas) => None,
        }
    }
}

#[derive(Clone, Copy)]
enum PendingPersistentAtlas {
    NoAtlas,
    NewlyAllocated(ImageHandle),
}

impl PendingPersistentAtlas {
    const fn is_present(self) -> bool {
        !matches!(self, Self::NoAtlas)
    }

    const fn resource(self) -> Option<ImageHandle> {
        match self {
            Self::NoAtlas => None,
            Self::NewlyAllocated(resource) => Some(resource),
        }
    }

    #[cfg(test)]
    const fn commit_outcome(self) -> VelloAtlasOutcome {
        match self {
            Self::NoAtlas => VelloAtlasOutcome::NoAtlas,
            Self::NewlyAllocated(_) => VelloAtlasOutcome::Retain,
        }
    }

    const fn abort_outcome(self) -> VelloAtlasOutcome {
        match self {
            Self::NoAtlas => VelloAtlasOutcome::NoAtlas,
            Self::NewlyAllocated(_) => VelloAtlasOutcome::Recreate,
        }
    }
}

struct PendingVelloResources {
    frame_scope: Option<FrameResourceScope>,
    buffers: HashMap<BufferHandle, ManagedBuffer>,
    images: HashMap<ImageHandle, ManagedImage>,
    persistent_image_atlas: PendingPersistentAtlas,
    released_buffers: HashSet<BufferHandle>,
    released_images: HashSet<ImageHandle>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloResourceAllocationRoleForTest {
    InternalVelloRasterBuffer,
    InternalVelloRasterImage,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct VelloResourceAllocationSummaryForTest {
    requested_roles: Vec<VelloResourceAllocationRoleForTest>,
    allocated_roles: Vec<VelloResourceAllocationRoleForTest>,
}

#[cfg(test)]
impl VelloResourceAllocationSummaryForTest {
    fn role_for_intent(intent: &ResourceIntent) -> VelloResourceAllocationRoleForTest {
        match intent {
            ResourceIntent::Buffer(_) => VelloResourceAllocationRoleForTest::InternalVelloRasterBuffer,
            ResourceIntent::Image(_) => VelloResourceAllocationRoleForTest::InternalVelloRasterImage,
        }
    }

    fn record_request(&mut self, intent: &ResourceIntent) {
        self.requested_roles.push(Self::role_for_intent(intent));
    }

    fn record_allocation(&mut self, intent: &ResourceIntent) {
        self.allocated_roles.push(Self::role_for_intent(intent));
    }

    fn role_count(
        roles: &[VelloResourceAllocationRoleForTest],
        role: VelloResourceAllocationRoleForTest,
    ) -> usize {
        roles.iter().filter(|actual| **actual == role).count()
    }

    pub(crate) fn internal_vello_raster_buffer_requests_for_test(&self) -> usize {
        Self::role_count(
            &self.requested_roles,
            VelloResourceAllocationRoleForTest::InternalVelloRasterBuffer,
        )
    }

    pub(crate) fn internal_vello_raster_buffer_allocations_for_test(&self) -> usize {
        Self::role_count(
            &self.allocated_roles,
            VelloResourceAllocationRoleForTest::InternalVelloRasterBuffer,
        )
    }

    pub(crate) fn internal_vello_raster_image_requests_for_test(&self) -> usize {
        Self::role_count(
            &self.requested_roles,
            VelloResourceAllocationRoleForTest::InternalVelloRasterImage,
        )
    }

    pub(crate) fn internal_vello_raster_image_allocations_for_test(&self) -> usize {
        Self::role_count(
            &self.allocated_roles,
            VelloResourceAllocationRoleForTest::InternalVelloRasterImage,
        )
    }

}

#[must_use]
pub(crate) struct VelloResourceLease {
    pending: PendingVelloResources,
    #[cfg(test)]
    allocation_summary: VelloResourceAllocationSummaryForTest,
}

#[must_use = "scope-clean Vello resource leases must be committed or aborted"]
pub(crate) struct ScopeResolvedVelloResourceLease {
    lease: VelloResourceLease,
}

#[must_use]
#[cfg(not(test))]
pub(crate) struct CommittedVelloResources;

#[must_use]
#[cfg(test)]
pub(crate) struct CommittedVelloResources {
    atlas_outcome: VelloAtlasOutcome,
}

/// Keeps a scope-clean resource lease uncertain until the owning GPU transaction succeeds.
#[must_use = "pending Vello resources must be committed by their transaction or aborted on drop"]
pub(crate) struct PendingVelloResourceCommit {
    lease: Option<ScopeResolvedVelloResourceLease>,
}

impl PendingVelloResourceCommit {
    pub(crate) fn new(mut lease: ScopeResolvedVelloResourceLease) -> Self {
        lease.consume_pending_atlas_recovery();
        Self { lease: Some(lease) }
    }

    #[cfg(test)]
    pub(crate) fn allocation_summary_for_test(&self) -> VelloResourceAllocationSummaryForTest {
        self.lease
            .as_ref()
            .expect("pending Vello resource commits must own their scope-resolved lease")
            .allocation_summary_for_test()
    }

    pub(crate) fn commit(mut self, _proof: VelloResourceCommitProof) {
        if let Some(lease) = self.lease.take() {
            let _ = lease.commit();
        }
    }
}

impl Drop for PendingVelloResourceCommit {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            let _ = lease.abort();
        }
    }
}

#[must_use]
#[derive(Debug)]
pub(crate) struct AbortedVelloResources {
    #[cfg(test)]
    discarded_resource_count: usize,
    #[cfg(test)]
    atlas_outcome: VelloAtlasOutcome,
}

impl VelloResourceLease {
    pub(super) fn allocate(
        scope: &ActiveVelloEncodingScope<'_>,
        manager: &ResourceManager,
        intents: &[ResourceIntent],
    ) -> Result<Self> {
        let device = scope.device();
        preflight_resource_intents(&device.limits(), intents)?;
        let mut pending = PendingVelloResources {
            frame_scope: Some(manager.begin_frame()?),
            buffers: HashMap::new(),
            images: HashMap::new(),
            persistent_image_atlas: PendingPersistentAtlas::NoAtlas,
            released_buffers: HashSet::new(),
            released_images: HashSet::new(),
        };
        #[cfg(test)]
        let mut allocation_summary = VelloResourceAllocationSummaryForTest::default();

        for intent in intents {
            #[cfg(test)]
            allocation_summary.record_request(intent);
            match intent {
                ResourceIntent::Buffer(buffer) => allocate_buffer(device, &mut pending, buffer)?,
                ResourceIntent::Image(image) => allocate_image(device, &mut pending, image)?,
            }
            #[cfg(test)]
            allocation_summary.record_allocation(intent);
        }

        Ok(Self {
            pending,
            #[cfg(test)]
            allocation_summary,
        })
    }

    pub(super) fn buffer(&self, handle: BufferHandle) -> Result<&wgpu::Buffer> {
        let managed = self.live_buffer(handle)?;
        self.pending.frame_scope()?.vello_buffer(&managed.lease)
    }

    pub(super) fn buffer_for_upload(
        &self,
        handle: BufferHandle,
        byte_len: usize,
    ) -> Result<&wgpu::Buffer> {
        let allocated = self.live_buffer(handle)?;
        let byte_len = u64::try_from(byte_len).map_err(|_| {
            render_failed("internal Vello buffer upload length does not fit the GPU address space")
        })?;
        if byte_len > allocated.byte_len {
            return Err(render_failed(
                "internal Vello buffer upload exceeds its prepared allocation",
            ));
        }
        self.pending.frame_scope()?.vello_buffer(&allocated.lease)
    }

    pub(super) fn indirect_buffer(
        &self,
        handle: BufferHandle,
        offset: u64,
    ) -> Result<&wgpu::Buffer> {
        if !offset.is_multiple_of(u64::from(std::mem::size_of::<u32>() as u32)) {
            return Err(render_failed(
                "internal Vello indirect dispatch offset is not aligned",
            ));
        }
        let allocated = self.live_buffer(handle)?;
        let required_end = offset.checked_add(3 * u64::from(std::mem::size_of::<u32>() as u32));
        if required_end.is_none_or(|end| end > allocated.byte_len) {
            return Err(render_failed(
                "internal Vello indirect dispatch exceeds its prepared allocation",
            ));
        }
        self.pending.frame_scope()?.vello_buffer(&allocated.lease)
    }

    pub(super) fn image_texture(&self, handle: ImageHandle) -> Result<&wgpu::Texture> {
        let managed = self.live_image(handle)?;
        self.pending
            .frame_scope()?
            .vello_image(&managed.lease)
            .map(|(texture, _)| texture)
    }

    pub(super) fn image_view(&self, handle: ImageHandle) -> Result<&wgpu::TextureView> {
        let managed = self.live_image(handle)?;
        self.pending
            .frame_scope()?
            .vello_image(&managed.lease)
            .map(|(_, view)| view)
    }

    pub(super) fn image_extent(&self, handle: ImageHandle) -> Result<PhysicalSize> {
        self.live_image(handle).map(|allocated| allocated.extent)
    }

    pub(super) fn record_release(&mut self, reference: ResourceReference) -> Result<()> {
        match reference {
            ResourceReference::Buffer(handle) => {
                if !self.pending.buffers.contains_key(&handle) {
                    return Err(render_failed(
                        "internal Vello recording releases an unknown buffer",
                    ));
                }
                if !self.pending.released_buffers.insert(handle) {
                    return Err(render_failed(
                        "internal Vello recording releases a buffer more than once",
                    ));
                }
            }
            ResourceReference::Image(handle) => {
                if !self.pending.images.contains_key(&handle) {
                    return Err(render_failed(
                        "internal Vello recording releases an unknown image",
                    ));
                }
                if !self.pending.released_images.insert(handle) {
                    return Err(render_failed(
                        "internal Vello recording releases an image more than once",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn after_clean_scope(self) -> ScopeResolvedVelloResourceLease {
        ScopeResolvedVelloResourceLease { lease: self }
    }

    fn into_committed_resources(mut self) -> CommittedVelloResources {
        self.pending.commit()
    }

    pub(crate) fn abort(mut self) -> AbortedVelloResources {
        self.pending.abort()
    }

    fn live_buffer(&self, handle: BufferHandle) -> Result<&ManagedBuffer> {
        if self.pending.released_buffers.contains(&handle) {
            return Err(render_failed(
                "internal Vello recording uses a released buffer",
            ));
        }
        self.pending.buffers.get(&handle).ok_or_else(|| {
            render_failed("internal Vello recording uses an unknown buffer")
        })
    }

    fn live_image(&self, handle: ImageHandle) -> Result<&ManagedImage> {
        if self.pending.released_images.contains(&handle) {
            return Err(render_failed(
                "internal Vello recording uses a released image",
            ));
        }
        self.pending.images.get(&handle).ok_or_else(|| {
            render_failed("internal Vello recording uses an unknown image")
        })
    }
}

impl ScopeResolvedVelloResourceLease {
    #[cfg(test)]
    fn allocation_summary_for_test(&self) -> VelloResourceAllocationSummaryForTest {
        self.lease.allocation_summary.clone()
    }
    fn commit(self) -> CommittedVelloResources {
        self.lease.into_committed_resources()
    }

    fn consume_pending_atlas_recovery(&mut self) {
        self.lease.pending.consume_pending_atlas_recovery();
    }

    pub(crate) fn abort(self) -> AbortedVelloResources {
        self.lease.abort()
    }
}

#[cfg(test)]
pub(crate) async fn over_limit_buffer_preflight_for_test(device: &wgpu::Device) -> Result<()> {
    let requested_size = device
        .limits()
        .max_buffer_size
        .checked_add(1)
        .ok_or_else(|| render_failed("test device cannot represent an over-limit buffer request"))?;
    let mut recording = super::RecordingBuilder::default();
    let _buffer = recording.new_buffer(super::BufferRole::Scene, requested_size)?;
    let (_recording, intents) = recording.finish();
    let manager = ResourceManager::default();
    let scope = ActiveVelloEncodingScope::begin(device);
    let allocation = VelloResourceLease::allocate(&scope, &manager, &intents);
    let allocation_result = match allocation {
        Ok(lease) => {
            let _aborted = lease.abort();
            Ok(())
        }
        Err(error) => Err(error),
    };
    scope.finish().await?;
    allocation_result
}

#[cfg(test)]
pub(crate) fn commit_scope_resolved_for_test(
    lease: ScopeResolvedVelloResourceLease,
) -> VelloAtlasOutcome {
    lease.commit().atlas_outcome()
}

#[cfg(test)]
pub(crate) async fn no_atlas_commit_outcome_for_test(
    device: &wgpu::Device,
) -> Result<VelloAtlasOutcome> {
    let intents = no_atlas_resource_intents_for_test()?;
    let manager = ResourceManager::default();
    let scope = ActiveVelloEncodingScope::begin(device);
    let allocation = VelloResourceLease::allocate(&scope, &manager, &intents);
    match allocation {
        Ok(lease) => match scope.finish_with_lease(lease).await {
            Ok(lease) => {
                let committed = lease.commit();
                Ok(committed.atlas_outcome())
            }
            Err(failure) => Err(failure.into_error_and_aborted_resources().0),
        },
        Err(error) => {
            scope.finish().await?;
            Err(error)
        }
    }
}

#[cfg(test)]
pub(crate) async fn no_atlas_abort_outcome_for_test(
    device: &wgpu::Device,
) -> Result<VelloAtlasOutcome> {
    let intents = no_atlas_resource_intents_for_test()?;
    let manager = ResourceManager::default();
    let scope = ActiveVelloEncodingScope::begin(device);
    let allocation = VelloResourceLease::allocate(&scope, &manager, &intents);
    let outcome = allocation.map(|lease| lease.abort().into_atlas_outcome());
    scope.finish().await?;
    outcome
}

#[cfg(test)]
fn no_atlas_resource_intents_for_test() -> Result<Vec<ResourceIntent>> {
    let mut recording = RecordingBuilder::default();
    let _buffer = recording.new_buffer(BufferRole::Scene, 4)?;
    let (_recording, intents) = recording.finish();
    Ok(intents)
}

impl AbortedVelloResources {
    pub(super) fn without_resources() -> Self {
        Self {
            #[cfg(test)]
            discarded_resource_count: 0,
            #[cfg(test)]
            atlas_outcome: VelloAtlasOutcome::NoAtlas,
        }
    }
}

#[cfg(test)]
impl AbortedVelloResources {
    pub(crate) const fn discarded_resource_count_for_test(&self) -> usize {
        self.discarded_resource_count
    }
}

#[cfg(test)]
impl AbortedVelloResources {
    pub(crate) fn into_atlas_outcome(self) -> VelloAtlasOutcome {
        self.atlas_outcome
    }
}

impl CommittedVelloResources {
    #[cfg(test)]
    pub(crate) const fn atlas_outcome(&self) -> VelloAtlasOutcome {
        self.atlas_outcome
    }

}

impl PendingVelloResources {
    fn frame_scope(&self) -> Result<&FrameResourceScope> {
        self.frame_scope.as_ref().ok_or_else(|| {
            render_failed("internal Vello resource frame was already resolved")
        })
    }

    fn frame_scope_mut(&mut self) -> Result<&mut FrameResourceScope> {
        self.frame_scope.as_mut().ok_or_else(|| {
            render_failed("internal Vello resource frame was already resolved")
        })
    }

    fn consume_pending_atlas_recovery(&mut self) {
        if let Some(scope) = self.frame_scope.as_mut() {
            scope.consume_vello_atlas_recovery();
        }
    }

    fn commit(&mut self) -> CommittedVelloResources {
        let atlas_handle = self.persistent_image_atlas.resource();
        #[cfg(test)]
        let atlas_outcome = self.persistent_image_atlas.commit_outcome();
        let mut frame_scope = self
            .frame_scope
            .take()
            .expect("a pending Vello commit must own its resource frame");

        for (_, managed) in self.buffers.drain() {
            frame_scope
                .discard(managed.lease)
                .expect("a Vello buffer must remain leased by its resource frame");
        }

        let mut retained_atlas = None;
        for (handle, managed) in self.images.drain() {
            if Some(handle) == atlas_handle {
                retained_atlas = Some(managed.lease);
            } else {
                frame_scope
                    .discard(managed.lease)
                    .expect("a transient Vello image must remain leased by its resource frame");
            }
        }
        if let Some(atlas) = retained_atlas {
            frame_scope.retire_idle_vello_atlases();
            frame_scope
                .release(atlas)
                .expect("the persistent Vello atlas must remain leased by its resource frame");
        }
        let _ = frame_scope.finish();

        #[cfg(not(test))]
        {
            CommittedVelloResources
        }
        #[cfg(test)]
        {
            CommittedVelloResources { atlas_outcome }
        }
    }

    fn abort(&mut self) -> AbortedVelloResources {
        #[cfg(test)]
        let discarded_resource_count = self.buffers.len().saturating_add(self.images.len());
        let atlas_outcome = self.persistent_image_atlas.abort_outcome();
        if let Some(mut frame_scope) = self.frame_scope.take() {
            for (_, managed) in self.buffers.drain() {
                let result = frame_scope.discard(managed.lease);
                debug_assert!(result.is_ok());
            }
            for (_, managed) in self.images.drain() {
                let result = frame_scope.discard(managed.lease);
                debug_assert!(result.is_ok());
            }
            frame_scope.record_vello_atlas_recovery(atlas_outcome);
            let _ = frame_scope.finish();
        }
        AbortedVelloResources {
            #[cfg(test)]
            discarded_resource_count,
            #[cfg(test)]
            atlas_outcome,
        }
    }
}

impl Drop for PendingVelloResources {
    fn drop(&mut self) {
        if self.frame_scope.is_some() {
            let _ = self.abort();
        }
    }
}

fn preflight_resource_intents(limits: &wgpu::Limits, intents: &[ResourceIntent]) -> Result<()> {
    // Keep every synchronous intent failure before allocation so an error cannot lose
    // fresh-atlas provenance from a partially built lease.
    let mut buffer_handles = HashSet::new();
    let mut image_handles = HashSet::new();
    let mut has_persistent_image_atlas = false;
    for intent in intents {
        match intent {
            ResourceIntent::Buffer(buffer) => {
                preflight_buffer(limits, buffer)?;
                if !buffer_handles.insert(buffer.resource) {
                    return Err(render_failed(
                        "internal Vello resource allocation repeats a buffer identity",
                    ));
                }
            }
            ResourceIntent::Image(image) => {
                preflight_image(limits, image)?;
                if !image_handles.insert(image.resource) {
                    return Err(render_failed(
                        "internal Vello resource allocation repeats an image identity",
                    ));
                }
                if image.retention == ImageRetention::PersistentImageAtlas {
                    if has_persistent_image_atlas {
                        return Err(render_failed(
                            "internal Vello resource allocation repeats the persistent image atlas",
                        ));
                    }
                    has_persistent_image_atlas = true;
                }
            }
        }
    }
    Ok(())
}

fn preflight_buffer(limits: &wgpu::Limits, intent: &BufferIntent) -> Result<()> {
    if intent.byte_len == 0 {
        return Err(render_failed(
            "internal Vello resource allocation received an empty buffer",
        ));
    }
    if intent.byte_len > limits.max_buffer_size {
        return Err(render_failed(
            "internal Vello buffer exceeds the device limit before allocation",
        ));
    }
    let binding_limit = match intent.role {
        BufferRole::Config => limits.max_uniform_buffer_binding_size,
        _ => limits.max_storage_buffer_binding_size,
    };
    if intent.byte_len > binding_limit {
        return Err(render_failed(
            "internal Vello buffer exceeds its device binding-class limit before allocation",
        ));
    }
    Ok(())
}

fn preflight_image(limits: &wgpu::Limits, intent: &ImageIntent) -> Result<()> {
    if intent.extent.width() == 0 || intent.extent.height() == 0 {
        return Err(render_failed(
            "internal Vello resource allocation received an empty image",
        ));
    }
    let max_extent = limits.max_texture_dimension_2d;
    if intent.extent.width() > max_extent || intent.extent.height() > max_extent {
        return Err(render_failed(
            "internal Vello image exceeds the device 2D texture limit before allocation",
        ));
    }
    Ok(())
}

fn allocate_buffer(
    device: &wgpu::Device,
    pending: &mut PendingVelloResources,
    intent: &BufferIntent,
) -> Result<()> {
    if intent.byte_len == 0 {
        return Err(render_failed(
            "internal Vello resource allocation received an empty buffer",
        ));
    }
    if pending.buffers.contains_key(&intent.resource) {
        return Err(render_failed(
            "internal Vello resource allocation repeats a buffer identity",
        ));
    }
    let lease = pending
        .frame_scope_mut()?
        .acquire_vello_buffer(device, VelloBufferKey::from_intent(intent))?;
    pending.buffers.insert(
        intent.resource,
        ManagedBuffer {
            lease,
            byte_len: intent.byte_len,
        },
    );
    Ok(())
}

fn allocate_image(
    device: &wgpu::Device,
    pending: &mut PendingVelloResources,
    intent: &ImageIntent,
) -> Result<()> {
    if intent.extent.width() == 0 || intent.extent.height() == 0 {
        return Err(render_failed(
            "internal Vello resource allocation received an empty image",
        ));
    }
    if pending.images.contains_key(&intent.resource) {
        return Err(render_failed(
            "internal Vello resource allocation repeats an image identity",
        ));
    }
    if intent.retention == ImageRetention::PersistentImageAtlas
        && pending.persistent_image_atlas.is_present()
    {
        return Err(render_failed(
            "internal Vello resource allocation repeats the persistent image atlas",
        ));
    }

    let key = VelloImageKey::from_intent(intent);
    let byte_len = image_byte_len(intent.extent, intent.format)?;
    let lease = pending
        .frame_scope_mut()?
        .acquire_vello_image(device, key, byte_len)?;
    if intent.retention == ImageRetention::PersistentImageAtlas {
        pending.persistent_image_atlas = PendingPersistentAtlas::NewlyAllocated(intent.resource);
    }
    pending.images.insert(
        intent.resource,
        ManagedImage {
            lease,
            extent: intent.extent,
        },
    );
    Ok(())
}

fn image_byte_len(extent: PhysicalSize, format: RasterImageFormat) -> Result<u64> {
    let bytes_per_pixel = match format {
        RasterImageFormat::Rgba8Unorm => 4_u64,
    };
    u64::from(extent.width())
        .checked_mul(u64::from(extent.height()))
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| render_failed("internal Vello image byte length overflows the GPU address space"))
}

const fn texture_format(format: RasterImageFormat) -> wgpu::TextureFormat {
    match format {
        RasterImageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

const fn buffer_usage(role: BufferRole) -> wgpu::BufferUsages {
    match role {
        BufferRole::Config => wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
        _ => wgpu::BufferUsages::STORAGE
            .union(wgpu::BufferUsages::COPY_DST)
            .union(wgpu::BufferUsages::COPY_SRC)
            .union(wgpu::BufferUsages::INDIRECT),
    }
}

const fn image_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST)
}

fn render_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

#[cfg(test)]
mod tests {
    use super::VelloAtlasOutcome;

    #[test]
    fn pending_atlas_recovery_merge_preserves_the_stronger_outcome() {
        use VelloAtlasOutcome::{MarkDirty, NoAtlas, Recreate, Retain};

        assert_eq!(
            VelloAtlasOutcome::merge_pending_recovery(Some(MarkDirty), Recreate),
            Some(Recreate)
        );
        assert_eq!(
            VelloAtlasOutcome::merge_pending_recovery(Some(Recreate), MarkDirty),
            Some(Recreate)
        );
        assert_eq!(
            VelloAtlasOutcome::merge_pending_recovery(Some(Recreate), NoAtlas),
            Some(Recreate)
        );
        assert_eq!(
            VelloAtlasOutcome::merge_pending_recovery(Some(MarkDirty), Retain),
            Some(MarkDirty)
        );
    }
}
