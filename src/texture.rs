#![cfg_attr(not(test), allow(dead_code))]

use super::{Error, Format, PhysicalSize, Result};
use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEXTURE_CACHE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TextureUsageIntent {
    OffscreenLayer,
    IntermediatePass,
    ReadbackReference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureDescriptor {
    physical_size: PhysicalSize,
    format: Format,
    intent: TextureUsageIntent,
    byte_len: u64,
}

impl TextureDescriptor {
    pub(crate) fn try_new(
        physical_size: PhysicalSize,
        format: Format,
        intent: TextureUsageIntent,
    ) -> Result<Self> {
        if physical_size.width() == 0 {
            return Err(Error::invalid_value(
                "texture width",
                physical_size.width(),
                "must be greater than 0 device pixels",
            ));
        }
        if physical_size.height() == 0 {
            return Err(Error::invalid_value(
                "texture height",
                physical_size.height(),
                "must be greater than 0 device pixels",
            ));
        }
        let pixel_count = u64::from(physical_size.width())
            .checked_mul(u64::from(physical_size.height()))
            .ok_or_else(|| {
                Error::invalid_value(
                    "texture pixel count",
                    format!("{}x{}", physical_size.width(), physical_size.height()),
                    "must fit in u64",
                )
            })?;
        let byte_len = pixel_count
            .checked_mul(u64::from(format.bytes_per_pixel()))
            .ok_or_else(|| {
                Error::invalid_value(
                    "texture byte length",
                    format!("{} pixels", pixel_count),
                    "must fit in u64",
                )
            })?;
        Ok(Self {
            physical_size,
            format,
            intent,
            byte_len,
        })
    }

    pub(crate) const fn physical_size(self) -> PhysicalSize {
        self.physical_size
    }

    pub(crate) const fn format(self) -> Format {
        self.format
    }

    pub(crate) const fn intent(self) -> TextureUsageIntent {
        self.intent
    }

    pub(crate) const fn byte_len(self) -> u64 {
        self.byte_len
    }

    pub(crate) const fn cache_key(self) -> TextureCacheKey {
        TextureCacheKey {
            physical_size: self.physical_size,
            format: self.format,
            intent: self.intent,
        }
    }

    pub(crate) const fn wgpu_usage(self) -> wgpu::TextureUsages {
        match (self.intent, self.format) {
            (
                TextureUsageIntent::OffscreenLayer | TextureUsageIntent::IntermediatePass,
                Format::Rgba8,
            ) => wgpu::TextureUsages::RENDER_ATTACHMENT
                .union(wgpu::TextureUsages::STORAGE_BINDING)
                .union(wgpu::TextureUsages::TEXTURE_BINDING)
                .union(wgpu::TextureUsages::COPY_SRC)
                .union(wgpu::TextureUsages::COPY_DST),
            (
                TextureUsageIntent::OffscreenLayer | TextureUsageIntent::IntermediatePass,
                Format::Bgra8,
            ) => wgpu::TextureUsages::RENDER_ATTACHMENT
                .union(wgpu::TextureUsages::TEXTURE_BINDING)
                .union(wgpu::TextureUsages::COPY_SRC)
                .union(wgpu::TextureUsages::COPY_DST),
            (TextureUsageIntent::ReadbackReference, _) => {
                wgpu::TextureUsages::STORAGE_BINDING.union(wgpu::TextureUsages::COPY_SRC)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureCacheKey {
    physical_size: PhysicalSize,
    format: Format,
    intent: TextureUsageIntent,
}

impl TextureCacheKey {
    pub(crate) const fn from_descriptor(descriptor: TextureDescriptor) -> Self {
        descriptor.cache_key()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct OffscreenTextureHandle {
    cache_id: u64,
    id: u64,
    lease: u64,
    descriptor: TextureDescriptor,
}

impl OffscreenTextureHandle {
    const fn new(cache_id: u64, id: u64, lease: u64, descriptor: TextureDescriptor) -> Self {
        Self {
            cache_id,
            id,
            lease,
            descriptor,
        }
    }

    pub(crate) const fn descriptor(self) -> TextureDescriptor {
        self.descriptor
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextureAllocationState {
    Live,
    Released,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TextureLifecycleStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) allocations: u64,
    pub(crate) releases: u64,
    pub(crate) evictions: u64,
}

#[derive(Debug)]
struct TextureCacheEntry {
    handle: OffscreenTextureHandle,
    key: TextureCacheKey,
    state: TextureAllocationState,
}

#[derive(Debug)]
pub(crate) struct OffscreenTextureCache {
    cache_id: u64,
    next_id: u64,
    next_lease: u64,
    entries: HashMap<u64, TextureCacheEntry>,
    released_by_key: HashMap<TextureCacheKey, VecDeque<u64>>,
    stats: TextureLifecycleStats,
    live_count: usize,
}

impl OffscreenTextureCache {
    pub(crate) fn new() -> Self {
        Self {
            cache_id: NEXT_TEXTURE_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            next_id: 0,
            next_lease: 0,
            entries: HashMap::new(),
            released_by_key: HashMap::new(),
            stats: TextureLifecycleStats::default(),
            live_count: 0,
        }
    }

    pub(crate) fn acquire(
        &mut self,
        descriptor: TextureDescriptor,
    ) -> Result<OffscreenTextureHandle> {
        let key = descriptor.cache_key();
        while let Some(id) = self
            .released_by_key
            .get_mut(&key)
            .and_then(VecDeque::pop_front)
        {
            if self
                .entries
                .get(&id)
                .is_some_and(|entry| entry.state == TextureAllocationState::Released)
            {
                let handle = self.next_handle(id, descriptor)?;
                let entry = self
                    .entries
                    .get_mut(&id)
                    .expect("released entry should remain available for reuse");
                entry.handle = handle;
                entry.state = TextureAllocationState::Live;
                self.stats.hits = self.stats.hits.saturating_add(1);
                self.live_count = self.live_count.saturating_add(1);
                return Ok(handle);
            }
        }

        let handle = self.allocate_handle(descriptor)?;
        self.stats.misses = self.stats.misses.saturating_add(1);
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        self.live_count = self.live_count.saturating_add(1);
        self.entries.insert(
            handle.id,
            TextureCacheEntry {
                handle,
                key,
                state: TextureAllocationState::Live,
            },
        );
        Ok(handle)
    }

    pub(crate) fn release(&mut self, handle: OffscreenTextureHandle) -> Result<()> {
        if handle.cache_id != self.cache_id {
            return Err(Error::invalid_value(
                "offscreen texture handle",
                handle.id,
                "must belong to this texture cache",
            ));
        }
        let Some(entry) = self.entries.get_mut(&handle.id) else {
            return Err(Error::invalid_value(
                "offscreen texture handle",
                handle.id,
                "must belong to this texture cache",
            ));
        };
        if entry.handle != handle {
            return Err(Error::invalid_value(
                "offscreen texture handle",
                handle.id,
                "must match the cache entry descriptor",
            ));
        }
        if entry.state == TextureAllocationState::Released {
            return Err(Error::invalid_value(
                "offscreen texture handle",
                handle.id,
                "must not be released more than once",
            ));
        }

        entry.state = TextureAllocationState::Released;
        self.released_by_key
            .entry(entry.key)
            .or_default()
            .push_back(handle.id);
        self.stats.releases = self.stats.releases.saturating_add(1);
        self.live_count = self.live_count.saturating_sub(1);
        Ok(())
    }

    pub(crate) fn evict_released(&mut self) -> usize {
        let released: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|(id, entry)| {
                (entry.state == TextureAllocationState::Released).then_some(*id)
            })
            .collect();
        let evicted = released.len();
        for id in released {
            self.entries.remove(&id);
        }
        self.released_by_key.clear();
        self.stats.evictions = self
            .stats
            .evictions
            .saturating_add(u64::try_from(evicted).unwrap_or(u64::MAX));
        evicted
    }

    pub(crate) const fn stats(&self) -> TextureLifecycleStats {
        self.stats
    }

    pub(crate) const fn live_count(&self) -> usize {
        self.live_count
    }

    pub(crate) fn retained_count(&self) -> usize {
        self.entries.len()
    }

    fn allocate_handle(&mut self, descriptor: TextureDescriptor) -> Result<OffscreenTextureHandle> {
        let id = self.next_id.checked_add(1).ok_or_else(|| {
            Error::invalid_value(
                "offscreen texture handle id",
                self.next_id,
                "must have remaining handle id space",
            )
        })?;
        self.next_id = id;
        self.next_handle(id, descriptor)
    }

    fn next_handle(
        &mut self,
        id: u64,
        descriptor: TextureDescriptor,
    ) -> Result<OffscreenTextureHandle> {
        let lease = self.next_lease.checked_add(1).ok_or_else(|| {
            Error::invalid_value(
                "offscreen texture lease",
                self.next_lease,
                "must have remaining lease token space",
            )
        })?;
        self.next_lease = lease;
        Ok(OffscreenTextureHandle::new(
            self.cache_id,
            id,
            lease,
            descriptor,
        ))
    }
}

impl Default for OffscreenTextureCache {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn headless_texture_descriptor(
    physical_size: PhysicalSize,
    format: Format,
) -> Result<TextureDescriptor> {
    TextureDescriptor::try_new(
        PhysicalSize::new(physical_size.width().max(1), physical_size.height().max(1)),
        format,
        TextureUsageIntent::ReadbackReference,
    )
}

trait TextureFormatExt {
    fn bytes_per_pixel(self) -> u8;
}

impl TextureFormatExt for Format {
    fn bytes_per_pixel(self) -> u8 {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
        }
    }
}
