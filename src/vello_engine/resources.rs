// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::{HashMap, HashSet};

use crate::{BackendErrorCode, Error, PhysicalSize, Result};

use super::encoder::ActiveVelloEncodingScope;
use super::{
    BufferHandle, BufferIntent, BufferRole, ImageHandle, ImageIntent, ImageRetention,
    RasterImageFormat, ResourceIntent, ResourceReference,
};

struct AllocatedBuffer {
    buffer: wgpu::Buffer,
    byte_len: u64,
}

struct AllocatedImage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    extent: PhysicalSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloAtlasOutcome {
    NoAtlas,
    Retain,
    MarkDirty,
    Recreate,
}

#[derive(Clone, Copy)]
enum PendingPersistentAtlas {
    NoAtlas,
    NewlyAllocated,
    #[expect(
        dead_code,
        reason = "C03 T4 records the reusable-atlas provenance that T6 will obtain from its resource manager."
    )]
    Reused,
}

impl PendingPersistentAtlas {
    const fn is_present(self) -> bool {
        !matches!(self, Self::NoAtlas)
    }

    const fn commit_outcome(self) -> VelloAtlasOutcome {
        match self {
            Self::NoAtlas => VelloAtlasOutcome::NoAtlas,
            Self::NewlyAllocated | Self::Reused => VelloAtlasOutcome::Retain,
        }
    }

    const fn abort_outcome(self) -> VelloAtlasOutcome {
        match self {
            Self::NoAtlas => VelloAtlasOutcome::NoAtlas,
            // T4 only creates a fresh atlas; it never borrows a reusable one that could be
            // marked dirty. An aborted fresh allocation must therefore be recreated.
            Self::NewlyAllocated => VelloAtlasOutcome::Recreate,
            Self::Reused => VelloAtlasOutcome::MarkDirty,
        }
    }
}

struct PendingVelloResources {
    buffers: HashMap<BufferHandle, AllocatedBuffer>,
    images: HashMap<ImageHandle, AllocatedImage>,
    persistent_image_atlas: PendingPersistentAtlas,
    released_buffers: HashSet<BufferHandle>,
    released_images: HashSet<ImageHandle>,
}

#[must_use]
pub(crate) struct VelloResourceLease {
    pending: PendingVelloResources,
}

#[must_use = "scope-clean Vello resource leases must be committed or aborted"]
pub(crate) struct ScopeResolvedVelloResourceLease {
    lease: VelloResourceLease,
}

#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T4 models the committed internal resource state for T6 transaction routing."
    )
)]
pub(crate) struct CommittedVelloResources {
    #[cfg_attr(
        test,
        expect(
            dead_code,
            reason = "C03 T4 retains committed lease ownership for later T6 transaction routing."
        )
    )]
    pending: PendingVelloResources,
    atlas_outcome: VelloAtlasOutcome,
}

#[must_use]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T4 keeps the consumed abort result typed until T6 transaction routing owns it."
    )
    )]
#[derive(Debug)]
pub(crate) struct AbortedVelloResources {
    discarded_resource_count: usize,
    atlas_outcome: VelloAtlasOutcome,
}

impl VelloResourceLease {
    pub(super) fn allocate(
        scope: &ActiveVelloEncodingScope<'_>,
        intents: &[ResourceIntent],
    ) -> Result<Self> {
        let device = scope.device();
        preflight_resource_intents(&device.limits(), intents)?;
        let mut pending = PendingVelloResources {
            buffers: HashMap::new(),
            images: HashMap::new(),
            persistent_image_atlas: PendingPersistentAtlas::NoAtlas,
            released_buffers: HashSet::new(),
            released_images: HashSet::new(),
        };

        for intent in intents {
            match intent {
                ResourceIntent::Buffer(buffer) => allocate_buffer(device, &mut pending, buffer)?,
                ResourceIntent::Image(image) => allocate_image(device, &mut pending, image)?,
            }
        }

        Ok(Self { pending })
    }

    pub(super) fn buffer(&self, handle: BufferHandle) -> Result<&wgpu::Buffer> {
        self.live_buffer(handle).map(|allocated| &allocated.buffer)
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
        Ok(&allocated.buffer)
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
        Ok(&allocated.buffer)
    }

    pub(super) fn image_texture(&self, handle: ImageHandle) -> Result<&wgpu::Texture> {
        self.live_image(handle).map(|allocated| &allocated.texture)
    }

    pub(super) fn image_view(&self, handle: ImageHandle) -> Result<&wgpu::TextureView> {
        self.live_image(handle).map(|allocated| &allocated.view)
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

    fn into_committed_resources(self) -> CommittedVelloResources {
        let atlas_outcome = self.pending.persistent_image_atlas.commit_outcome();
        CommittedVelloResources {
            pending: self.pending,
            atlas_outcome,
        }
    }

    pub(crate) fn abort(self) -> AbortedVelloResources {
        let discarded_resource_count = self
            .pending
            .buffers
            .len()
            .saturating_add(self.pending.images.len());
        let atlas_outcome = self.pending.persistent_image_atlas.abort_outcome();
        AbortedVelloResources {
            discarded_resource_count,
            atlas_outcome,
        }
    }

    fn live_buffer(&self, handle: BufferHandle) -> Result<&AllocatedBuffer> {
        if self.pending.released_buffers.contains(&handle) {
            return Err(render_failed(
                "internal Vello recording uses a released buffer",
            ));
        }
        self.pending.buffers.get(&handle).ok_or_else(|| {
            render_failed("internal Vello recording uses an unknown buffer")
        })
    }

    fn live_image(&self, handle: ImageHandle) -> Result<&AllocatedImage> {
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
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 preserves the clean-scope pending-to-committed transition for T6."
        )
    )]
    pub(crate) fn commit(self) -> CommittedVelloResources {
        self.lease.into_committed_resources()
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 preserves explicit abort after clean-scope resolution for later terminal-signal handling."
        )
    )]
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
    let scope = ActiveVelloEncodingScope::begin(device);
    let allocation = VelloResourceLease::allocate(&scope, &intents);
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

impl AbortedVelloResources {
    pub(super) fn without_resources() -> Self {
        Self {
            discarded_resource_count: 0,
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

impl AbortedVelloResources {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 keeps typed abort-atlas recovery consumable by the later T6 resource manager."
        )
    )]
    pub(crate) fn into_atlas_outcome(self) -> VelloAtlasOutcome {
        self.atlas_outcome
    }
}

impl CommittedVelloResources {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 keeps typed committed-atlas retention consumable by the later T6 resource manager."
        )
    )]
    pub(crate) const fn atlas_outcome(&self) -> VelloAtlasOutcome {
        self.atlas_outcome
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
    let usage = match intent.role {
        BufferRole::Config => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        _ => {
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::INDIRECT
        }
    };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Surgeist internal Vello buffer"),
        size: intent.byte_len,
        usage,
        mapped_at_creation: false,
    });
    pending.buffers.insert(
        intent.resource,
        AllocatedBuffer {
            buffer,
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

    let format = texture_format(intent.format);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surgeist internal Vello image"),
        size: wgpu::Extent3d {
            width: intent.extent.width(),
            height: intent.extent.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Surgeist internal Vello image view"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: None,
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    });
    if intent.retention == ImageRetention::PersistentImageAtlas {
        pending.persistent_image_atlas = PendingPersistentAtlas::NewlyAllocated;
    }
    pending.images.insert(
        intent.resource,
        AllocatedImage {
            texture,
            view,
            extent: intent.extent,
        },
    );
    Ok(())
}

const fn texture_format(format: RasterImageFormat) -> wgpu::TextureFormat {
    match format {
        RasterImageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

fn render_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}
