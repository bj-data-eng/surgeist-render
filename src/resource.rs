use super::{Error, ResourceCacheBudget, Result, texture::TextureCacheKey};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkingFormat {
    HighPrecision,
    ReducedPrecision,
}

impl WorkingFormat {
    pub(crate) const fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            Self::HighPrecision => wgpu::TextureFormat::Rgba16Float,
            Self::ReducedPrecision => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    pub(crate) const fn required_usages(self) -> wgpu::TextureUsages {
        let _ = self;
        wgpu::TextureUsages::RENDER_ATTACHMENT
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST)
    }

    pub(crate) const fn required_format_features(self) -> wgpu::TextureFormatFeatureFlags {
        let _ = self;
        wgpu::TextureFormatFeatureFlags::FILTERABLE
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C07 resource accounting consumes this owned format fact after lifecycle modeling"
        )
    )]
    pub(crate) const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::HighPrecision => 8,
            Self::ReducedPrecision => 4,
        }
    }

    pub(crate) fn is_supported_by(self, features: wgpu::TextureFormatFeatures) -> bool {
        features.allowed_usages.contains(self.required_usages())
            && features.flags.contains(self.required_format_features())
    }
}

#[derive(Clone)]
pub(crate) struct ManagerIdentity(Arc<()>);

impl ManagerIdentity {
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl PartialEq for ManagerIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ManagerIdentity {}

impl fmt::Debug for ManagerIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ManagerIdentity(..)")
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct FrameIdentity(u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ResourceIdentity(u64);

impl ResourceIdentity {
    pub(crate) const fn get(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_raw_for_test(raw: u64) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct AllocationGeneration(u64);

impl AllocationGeneration {
    #[cfg(test)]
    pub(crate) const fn get_for_test(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_raw_for_test(raw: u64) -> Self {
        Self(raw)
    }
}

/// One non-interchangeable namespace for every resource role entering the
/// per-device manager. T3 adds descriptor facts to the role-only variants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "T3 supplies concrete descriptors for the modeled C07 resource namespaces"
    )
)]
pub(crate) enum ResourceCacheKey {
    VelloAtlas,
    CaptureTexture,
    WorkingTexture,
    CoverageTexture,
    ResolvedMaskUpload,
    GaussianKernelBuffer,
    TransitionalTexture(TextureCacheKey),
}

impl ResourceCacheKey {
    pub(crate) const fn transitional_texture(key: TextureCacheKey) -> Self {
        Self::TransitionalTexture(key)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResourceLifecycleStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) allocations: u64,
    pub(crate) releases: u64,
    pub(crate) evictions: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceEntryState {
    Idle { last_used_frame: FrameIdentity },
    Leased { frame: FrameIdentity },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResourceEntry {
    key: ResourceCacheKey,
    allocation_generation: AllocationGeneration,
    byte_len: u64,
    state: ResourceEntryState,
}

struct ResourceManagerState {
    identity: ManagerIdentity,
    budget: ResourceCacheBudget,
    next_frame: u64,
    next_resource: u64,
    retained_bytes: u64,
    active_frames: BTreeSet<FrameIdentity>,
    resolved_leases: BTreeSet<(FrameIdentity, ResourceIdentity)>,
    entries: BTreeMap<ResourceIdentity, ResourceEntry>,
    stats: ResourceLifecycleStats,
}

impl ResourceManagerState {
    fn invalid_lease(resource: ResourceIdentity, reason: &'static str) -> Error {
        Error::invalid_value("resource lease", resource.get(), reason)
    }

    fn acquire(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        if self.identity != *manager_identity || !self.active_frames.contains(&frame) {
            return Err(Error::invalid_value(
                "resource frame",
                frame.0,
                "must belong to an active frame of this resource manager",
            ));
        }
        if byte_len == 0 {
            return Err(Error::invalid_value(
                "resource byte length",
                byte_len,
                "must be greater than zero",
            ));
        }

        let reusable = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| match entry.state {
                ResourceEntryState::Idle { last_used_frame }
                    if entry.key == key && entry.byte_len == byte_len =>
                {
                    Some((last_used_frame, *identity))
                }
                ResourceEntryState::Idle { .. } | ResourceEntryState::Leased { .. } => None,
            })
            .min()
            .map(|(_, identity)| identity);

        let (resource, allocation_generation) = if let Some(resource) = reusable {
            let entry = self
                .entries
                .get_mut(&resource)
                .expect("the selected idle resource must remain registered");
            entry.state = ResourceEntryState::Leased { frame };
            self.stats.hits = self.stats.hits.saturating_add(1);
            (resource, entry.allocation_generation)
        } else {
            let next_resource = self.next_resource.checked_add(1).ok_or_else(|| {
                Error::invalid_value(
                    "resource identity",
                    self.next_resource,
                    "must have remaining identity space",
                )
            })?;
            let retained_bytes = self.retained_bytes.checked_add(byte_len).ok_or_else(|| {
                Error::invalid_value(
                    "retained resource byte length",
                    format!("{} + {byte_len}", self.retained_bytes),
                    "must fit in u64",
                )
            })?;
            let resource = ResourceIdentity(next_resource);
            let allocation_generation = AllocationGeneration(1);
            self.next_resource = next_resource;
            self.retained_bytes = retained_bytes;
            self.entries.insert(
                resource,
                ResourceEntry {
                    key,
                    allocation_generation,
                    byte_len,
                    state: ResourceEntryState::Leased { frame },
                },
            );
            self.stats.misses = self.stats.misses.saturating_add(1);
            self.stats.allocations = self.stats.allocations.saturating_add(1);
            (resource, allocation_generation)
        };

        Ok(ResourceLease::new(
            manager_identity.clone(),
            frame,
            resource,
            allocation_generation,
        ))
    }

    fn validate_lease(
        &self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: &ResourceLeaseToken,
    ) -> Result<()> {
        if self.identity != *manager_identity || token.manager_identity != self.identity {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must belong to this resource manager",
            ));
        }
        if token.frame_identity != frame || !self.active_frames.contains(&frame) {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must belong to this active resource frame",
            ));
        }
        if self
            .resolved_leases
            .contains(&(frame, token.resource_identity))
        {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must not have already been resolved by this frame",
            ));
        }
        let Some(entry) = self.entries.get(&token.resource_identity) else {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must name a current resource identity",
            ));
        };
        if entry.allocation_generation != token.allocation_generation {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must name the current allocation generation",
            ));
        }
        if entry.state != (ResourceEntryState::Leased { frame }) {
            return Err(Self::invalid_lease(
                token.resource_identity,
                "must name a resource leased by this frame",
            ));
        }
        Ok(())
    }

    fn release(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: ResourceLeaseToken,
    ) -> Result<()> {
        self.validate_lease(manager_identity, frame, &token)?;
        self.resolved_leases
            .insert((frame, token.resource_identity));
        self.stats.releases = self.stats.releases.saturating_add(1);
        Ok(())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T3 replaces modeled allocations through this generation-checked transition"
        )
    )]
    fn replace(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: ResourceLeaseToken,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        self.validate_lease(manager_identity, frame, &token)?;
        if byte_len == 0 {
            return Err(Error::invalid_value(
                "resource byte length",
                byte_len,
                "must be greater than zero",
            ));
        }
        let entry = self
            .entries
            .get(&token.resource_identity)
            .expect("a validated resource lease must remain registered");
        let allocation_generation =
            AllocationGeneration(entry.allocation_generation.0.checked_add(1).ok_or_else(
                || {
                    Error::invalid_value(
                        "resource allocation generation",
                        entry.allocation_generation.0,
                        "must have remaining generation space",
                    )
                },
            )?);
        let retained_without_replaced = self
            .retained_bytes
            .checked_sub(entry.byte_len)
            .expect("registered resource bytes must not exceed retained accounting");
        let retained_bytes = retained_without_replaced
            .checked_add(byte_len)
            .ok_or_else(|| {
                Error::invalid_value(
                    "retained resource byte length",
                    format!("{retained_without_replaced} + {byte_len}"),
                    "must fit in u64",
                )
            })?;

        let entry = self
            .entries
            .get_mut(&token.resource_identity)
            .expect("a validated resource lease must remain registered");
        entry.key = key;
        entry.allocation_generation = allocation_generation;
        entry.byte_len = byte_len;
        self.retained_bytes = retained_bytes;
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        Ok(ResourceLease::new(
            manager_identity.clone(),
            frame,
            token.resource_identity,
            allocation_generation,
        ))
    }

    fn discard(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: ResourceLeaseToken,
    ) -> Result<()> {
        self.validate_lease(manager_identity, frame, &token)?;
        let entry = self
            .entries
            .remove(&token.resource_identity)
            .expect("a validated resource lease must remain registered");
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(entry.byte_len)
            .expect("registered resource bytes must not exceed retained accounting");
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        Ok(())
    }

    fn cleanup_frame(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
    ) -> FrameCleanup {
        if self.identity != *manager_identity || !self.active_frames.remove(&frame) {
            return FrameCleanup::default();
        }

        for (resource_identity, entry) in &mut self.entries {
            if entry.state == (ResourceEntryState::Leased { frame }) {
                entry.state = ResourceEntryState::Idle {
                    last_used_frame: frame,
                };
                if !self.resolved_leases.remove(&(frame, *resource_identity)) {
                    self.stats.releases = self.stats.releases.saturating_add(1);
                }
            }
        }
        self.trim_idle()
    }

    fn trim_idle(&mut self) -> FrameCleanup {
        let mut idle = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| match entry.state {
                ResourceEntryState::Idle { last_used_frame } => {
                    Some((last_used_frame, *identity, entry.byte_len))
                }
                ResourceEntryState::Leased { .. } => None,
            })
            .collect::<Vec<_>>();
        idle.sort_unstable_by_key(|(last_used_frame, identity, _)| (*last_used_frame, *identity));
        let mut cleanup = FrameCleanup::default();

        for (_, identity, byte_len) in idle {
            if self.retained_bytes <= self.budget.bytes() {
                break;
            }
            let removed = self
                .entries
                .remove(&identity)
                .expect("the selected idle resource must remain registered");
            debug_assert_eq!(removed.byte_len, byte_len);
            self.retained_bytes = self
                .retained_bytes
                .checked_sub(byte_len)
                .expect("retained byte accounting must include every selected resource");
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            cleanup.evicted_resources.push(identity);
        }
        cleanup
    }
}

/// The sole private owner of resource identity, lease state, accounting, and
/// deterministic idle retention for one device generation.
pub(crate) struct ResourceManager {
    state: Arc<Mutex<ResourceManagerState>>,
}

impl ResourceManager {
    pub(crate) fn new(budget: ResourceCacheBudget) -> Self {
        Self {
            state: Arc::new(Mutex::new(ResourceManagerState {
                identity: ManagerIdentity::new(),
                budget,
                next_frame: 0,
                next_resource: 0,
                retained_bytes: 0,
                active_frames: BTreeSet::new(),
                resolved_leases: BTreeSet::new(),
                entries: BTreeMap::new(),
                stats: ResourceLifecycleStats::default(),
            })),
        }
    }

    pub(crate) fn begin_frame(&self) -> Result<FrameResourceScope> {
        let mut state = lock_state(&self.state);
        let next_frame = state.next_frame.checked_add(1).ok_or_else(|| {
            Error::invalid_value(
                "resource frame identity",
                state.next_frame,
                "must have remaining identity space",
            )
        })?;
        let frame = FrameIdentity(next_frame);
        state.next_frame = next_frame;
        state.active_frames.insert(frame);
        Ok(FrameResourceScope {
            state: Arc::clone(&self.state),
            manager_identity: state.identity.clone(),
            frame,
            cleaned: false,
        })
    }

    pub(crate) fn stats(&self) -> ResourceLifecycleStats {
        lock_state(&self.state).stats
    }

    pub(crate) fn live_count(&self) -> usize {
        lock_state(&self.state)
            .entries
            .values()
            .filter(|entry| matches!(entry.state, ResourceEntryState::Leased { .. }))
            .count()
    }

    #[cfg(test)]
    pub(crate) fn retained_count(&self) -> usize {
        lock_state(&self.state).entries.len()
    }

    #[cfg(test)]
    pub(crate) fn observation_for_test(&self) -> ResourceManagerObservationForTest {
        let state = lock_state(&self.state);
        let idle_count = state
            .entries
            .values()
            .filter(|entry| matches!(entry.state, ResourceEntryState::Idle { .. }))
            .count();
        ResourceManagerObservationForTest {
            idle_count,
            leased_count: state.entries.len().saturating_sub(idle_count),
            retained_bytes: state.retained_bytes,
        }
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new(ResourceCacheBudget::DEFAULT)
    }
}

fn lock_state(state: &Arc<Mutex<ResourceManagerState>>) -> MutexGuard<'_, ResourceManagerState> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct ResourceLeaseToken {
    manager_identity: ManagerIdentity,
    frame_identity: FrameIdentity,
    resource_identity: ResourceIdentity,
    allocation_generation: AllocationGeneration,
}

/// A production lease is intentionally neither `Clone` nor `Copy`; every
/// resolving operation consumes it.
#[must_use = "resource leases must be resolved by their owning frame scope"]
pub(crate) struct ResourceLease {
    token: ResourceLeaseToken,
}

impl ResourceLease {
    fn new(
        manager_identity: ManagerIdentity,
        frame_identity: FrameIdentity,
        resource_identity: ResourceIdentity,
        allocation_generation: AllocationGeneration,
    ) -> Self {
        Self {
            token: ResourceLeaseToken {
                manager_identity,
                frame_identity,
                resource_identity,
                allocation_generation,
            },
        }
    }

    pub(crate) const fn resource_identity(&self) -> ResourceIdentity {
        self.token.resource_identity
    }

    #[cfg(test)]
    pub(crate) fn token_for_test(&self) -> ResourceLeaseTokenForTest {
        ResourceLeaseTokenForTest {
            manager_identity: self.token.manager_identity.clone(),
            frame_identity: self.token.frame_identity,
            resource_identity: self.token.resource_identity,
            allocation_generation: self.token.allocation_generation,
        }
    }
}

impl fmt::Debug for ResourceLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceLease")
            .field("manager_identity", &self.token.manager_identity)
            .field("frame_identity", &self.token.frame_identity)
            .field("resource_identity", &self.token.resource_identity)
            .field("allocation_generation", &self.token.allocation_generation)
            .finish()
    }
}

#[must_use = "frame resource scopes clean outstanding leases when resolved or dropped"]
pub(crate) struct FrameResourceScope {
    state: Arc<Mutex<ResourceManagerState>>,
    manager_identity: ManagerIdentity,
    frame: FrameIdentity,
    cleaned: bool,
}

impl FrameResourceScope {
    pub(crate) fn acquire(
        &mut self,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        lock_state(&self.state).acquire(&self.manager_identity, self.frame, key, byte_len)
    }

    pub(crate) fn release(&mut self, lease: ResourceLease) -> Result<()> {
        lock_state(&self.state).release(&self.manager_identity, self.frame, lease.token)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "T3 replaces modeled allocations through this generation-checked transition"
        )
    )]
    pub(crate) fn replace(
        &mut self,
        lease: ResourceLease,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        lock_state(&self.state).replace(
            &self.manager_identity,
            self.frame,
            lease.token,
            key,
            byte_len,
        )
    }

    pub(crate) fn discard(&mut self, lease: ResourceLease) -> Result<()> {
        lock_state(&self.state).discard(&self.manager_identity, self.frame, lease.token)
    }

    pub(crate) fn finish(mut self) -> FrameCleanup {
        self.cleanup()
    }

    fn cleanup(&mut self) -> FrameCleanup {
        if self.cleaned {
            return FrameCleanup::default();
        }
        self.cleaned = true;
        lock_state(&self.state).cleanup_frame(&self.manager_identity, self.frame)
    }

    #[cfg(test)]
    pub(crate) fn frame_identity_for_test(&self) -> FrameIdentity {
        self.frame
    }

    #[cfg(test)]
    pub(crate) fn manager_identity_for_test(&self) -> ManagerIdentity {
        self.manager_identity.clone()
    }

    #[cfg(test)]
    pub(crate) fn release_injected_for_test(
        &mut self,
        token: ResourceLeaseTokenForTest,
    ) -> Result<()> {
        lock_state(&self.state).release(&self.manager_identity, self.frame, token.into_token())
    }

    #[cfg(test)]
    pub(crate) fn trim_idle_for_test(&mut self) -> FrameCleanup {
        lock_state(&self.state).trim_idle()
    }
}

impl Drop for FrameResourceScope {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct FrameCleanup {
    evicted_resources: Vec<ResourceIdentity>,
}

impl FrameCleanup {
    pub(crate) fn evicted_resources(&self) -> &[ResourceIdentity] {
        &self.evicted_resources
    }
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResourceLeaseTokenForTest {
    pub(crate) manager_identity: ManagerIdentity,
    pub(crate) frame_identity: FrameIdentity,
    pub(crate) resource_identity: ResourceIdentity,
    pub(crate) allocation_generation: AllocationGeneration,
}

#[cfg(test)]
impl ResourceLeaseTokenForTest {
    fn into_token(self) -> ResourceLeaseToken {
        ResourceLeaseToken {
            manager_identity: self.manager_identity,
            frame_identity: self.frame_identity,
            resource_identity: self.resource_identity,
            allocation_generation: self.allocation_generation,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceManagerObservationForTest {
    pub(crate) idle_count: usize,
    pub(crate) leased_count: usize,
    pub(crate) retained_bytes: u64,
}
