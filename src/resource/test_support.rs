use super::{
    FrameCleanup, FrameResourceScope, ResourceCacheBudget, ResourceLease, ResourceManager,
    ResourceRetentionOutcome, Result, WorkingFormat,
    lease::ResourceLeaseToken,
    lock_state,
    manager::{
        AllocationGeneration, FrameIdentity, ManagerIdentity, ResourceAccountingFault,
        ResourceAllocationPreflight, ResourceCacheKey, ResourceEntryState, ResourceIdentity,
        ResourceLifecycleStats,
    },
};
use crate::{
    PhysicalSize,
    backend::DeviceCapabilities,
    image::{ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey},
    texture::EffectTextureDescriptor,
    vello_engine::VelloAtlasOutcome,
};

impl ResourceIdentity {
    pub(crate) const fn from_raw_for_test(raw: u64) -> Self {
        Self(raw)
    }
}

impl AllocationGeneration {
    pub(crate) const fn get_for_test(self) -> u64 {
        self.0
    }

    pub(crate) const fn from_raw_for_test(raw: u64) -> Self {
        Self(raw)
    }
}

impl ResourceCacheKey {
    pub(super) const fn accepts_modeled_payload(&self) -> bool {
        matches!(self, Self::VelloAtlas | Self::EffectTexture(_))
    }

    pub(super) const fn is_vello_buffer(&self) -> bool {
        matches!(self, Self::VelloBuffer(_))
    }

    pub(super) const fn is_transient_vello_image(&self) -> bool {
        matches!(self, Self::VelloImage(key) if !key.is_persistent_atlas())
    }
}

impl ResourceAllocationPreflight {
    pub(crate) fn zero_sized_mask_is_explicitly_empty_for_test(
        descriptor: &ResolvedMaskUploadDescriptor,
    ) -> bool {
        Self::resolved_mask(descriptor).is_ok_and(|preflight| preflight.is_none())
    }
}

impl WorkingFormat {
    pub(crate) const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::HighPrecision => 8,
            Self::ReducedPrecision => 4,
        }
    }
}

impl ResourceManager {
    #[cfg(test)]
    pub(crate) fn stats(&self) -> ResourceLifecycleStats {
        lock_state(&self.state).stats
    }

    #[cfg(test)]
    pub(crate) fn live_count(&self) -> usize {
        lock_state(&self.state)
            .entries
            .values()
            .filter(|entry| matches!(entry.state, ResourceEntryState::Leased { .. }))
            .count()
    }

    #[cfg(test)]
    pub(crate) fn is_empty_for_test(&self) -> bool {
        lock_state(&self.state).entries.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn identity_for_test(&self) -> ManagerIdentity {
        lock_state(&self.state).identity.clone()
    }

    #[cfg(test)]
    pub(crate) fn budget_for_test(&self) -> ResourceCacheBudget {
        lock_state(&self.state).budget
    }

    #[cfg(test)]
    pub(crate) fn retained_count(&self) -> usize {
        lock_state(&self.state).entries.len()
    }

    #[cfg(test)]
    pub(crate) fn inject_retained_byte_mismatch_before_discard_for_test(&self) {
        let mut state = lock_state(&self.state);
        state.retained_bytes = state
            .retained_bytes
            .checked_sub(1)
            .expect("the accounting-mismatch fixture requires retained bytes");
    }

    #[cfg(test)]
    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        let mut state = lock_state(&self.state);
        let registered_entry_bytes = state
            .checked_registered_entry_bytes()
            .expect("the accounting-poison fixture requires a representable entry total");
        state.retained_bytes = state
            .retained_bytes
            .checked_sub(1)
            .expect("the accounting-poison fixture requires retained bytes");
        let fault = ResourceAccountingFault::RetainedByteMismatch {
            retained_bytes: state.retained_bytes,
            registered_entry_bytes,
        };
        state.record_accounting_fault(fault);
        fault
    }

    #[cfg(test)]
    pub(crate) fn inject_retained_byte_underflow_before_discard_for_test(&self) {
        lock_state(&self.state).retained_bytes = 0;
    }

    #[cfg(test)]
    pub(crate) fn inject_registered_entry_total_overflow_for_test(
        &self,
        first: ResourceIdentity,
        second: ResourceIdentity,
    ) -> [(ResourceIdentity, u64); 2] {
        let mut state = lock_state(&self.state);
        let first_original = state
            .entries
            .get(&first)
            .expect("the first overflow-fixture resource must remain registered")
            .byte_len;
        let second_original = state
            .entries
            .get(&second)
            .expect("the second overflow-fixture resource must remain registered")
            .byte_len;
        state
            .entries
            .get_mut(&first)
            .expect("the first overflow-fixture resource must remain registered")
            .byte_len = u64::MAX;
        state
            .entries
            .get_mut(&second)
            .expect("the second overflow-fixture resource must remain registered")
            .byte_len = 1;
        [(first, first_original), (second, second_original)]
    }

    #[cfg(test)]
    pub(crate) fn restore_registered_entry_byte_lengths_for_test(
        &self,
        originals: [(ResourceIdentity, u64); 2],
    ) {
        let mut state = lock_state(&self.state);
        for (identity, byte_len) in originals {
            state
                .entries
                .get_mut(&identity)
                .expect("the restored overflow-fixture resource must remain registered")
                .byte_len = byte_len;
        }
    }

    #[cfg(test)]
    pub(crate) fn observation_for_test(&self) -> ResourceManagerObservationForTest {
        let state = lock_state(&self.state);
        let idle_count = state
            .entries
            .values()
            .filter(|entry| matches!(entry.state, ResourceEntryState::Idle { .. }))
            .count();
        let retained_atlas_count = state
            .entries
            .values()
            .filter(|entry| entry.key.is_vello_atlas())
            .count();
        let retained_atlas_byte_len = state
            .entries
            .values()
            .filter(|entry| entry.key.is_vello_atlas())
            .map(|entry| entry.byte_len)
            .sum();
        let committed_transient_buffer_count = state
            .entries
            .values()
            .filter(|entry| entry.key.is_vello_buffer())
            .count();
        let committed_transient_buffer_byte_len = state
            .entries
            .values()
            .filter(|entry| entry.key.is_vello_buffer())
            .map(|entry| entry.byte_len)
            .sum();
        let committed_transient_image_count = state
            .entries
            .values()
            .filter(|entry| entry.key.is_transient_vello_image())
            .count();
        let committed_transient_image_byte_len = state
            .entries
            .values()
            .filter(|entry| entry.key.is_transient_vello_image())
            .map(|entry| entry.byte_len)
            .sum();
        let effect_texture_count = state
            .entries
            .values()
            .filter(|entry| matches!(entry.key, ResourceCacheKey::EffectTexture(_)))
            .count();
        let resolved_mask_upload_keys = state
            .entries
            .values()
            .filter_map(|entry| match &entry.key {
                ResourceCacheKey::ResolvedMaskUpload(key) => Some(key.clone()),
                ResourceCacheKey::VelloAtlas
                | ResourceCacheKey::VelloBuffer(_)
                | ResourceCacheKey::VelloImage(_)
                | ResourceCacheKey::EffectTexture(_)
                | ResourceCacheKey::GaussianKernelBuffer(_) => None,
            })
            .collect();
        let gaussian_kernel_count = state
            .entries
            .values()
            .filter(|entry| matches!(entry.key, ResourceCacheKey::GaussianKernelBuffer(_)))
            .count();
        ResourceManagerObservationForTest {
            idle_count,
            leased_count: state.entries.len().saturating_sub(idle_count),
            retained_bytes: state.retained_bytes,
            accounted_entry_bytes: state.checked_registered_entry_bytes(),
            accounting_fault: state.accounting_fault,
            active_frame_count: state.active_frames.len(),
            resolved_lease_count: state.resolved_leases.len(),
            entry_identities: state.entries.keys().copied().collect(),
            lifecycle_stats: state.stats,
            next_resource: state.next_resource,
            entry_count: state.entries.len(),
            payload_creation_attempts: state.payload_creation_attempts,
            retained_atlas_count,
            retained_atlas_byte_len,
            committed_transient_buffer_count,
            committed_transient_buffer_byte_len,
            committed_transient_image_count,
            committed_transient_image_byte_len,
            effect_texture_count,
            resolved_mask_upload_keys,
            gaussian_kernel_count,
            recovery_outcome: state.pending_vello_atlas_recovery,
        }
    }
}

impl ResourceLease {
    pub(crate) fn token_for_test(&self) -> super::ResourceLeaseTokenForTest {
        super::ResourceLeaseTokenForTest {
            manager_identity: self.token.manager_identity.clone(),
            frame_identity: self.token.frame_identity,
            resource_identity: self.token.resource_identity,
            allocation_generation: self.token.allocation_generation,
        }
    }
}

impl FrameResourceScope {
    pub(crate) fn acquire_working_effect_texture_for_test(
        &mut self,
        device: &wgpu::Device,
        capabilities: &DeviceCapabilities,
        working_format: WorkingFormat,
        physical_size: PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<ResourceLease> {
        let descriptor =
            EffectTextureDescriptor::try_working(working_format, physical_size, usage)?;
        self.acquire_effect_texture(device, capabilities, descriptor)
    }

    pub(crate) fn acquire(
        &mut self,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        let lease =
            lock_state(&self.state).acquire(&self.manager_identity, self.frame, key, byte_len)?;
        Ok(self.record_acquisition(lease))
    }
    pub(crate) fn replace(
        &mut self,
        lease: ResourceLease,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        let lease = lock_state(&self.state).replace(
            &self.manager_identity,
            self.frame,
            lease.token,
            key,
            byte_len,
        )?;
        Ok(self.record_acquisition(lease))
    }
    pub(crate) fn frame_identity_for_test(&self) -> FrameIdentity {
        self.frame
    }

    pub(crate) fn manager_identity_for_test(&self) -> ManagerIdentity {
        self.manager_identity.clone()
    }

    pub(crate) fn release_injected_for_test(
        &mut self,
        token: super::ResourceLeaseTokenForTest,
    ) -> Result<()> {
        lock_state(&self.state).release(&self.manager_identity, self.frame, token.into_token())
    }

    pub(crate) fn trim_idle_for_test(&mut self) -> FrameCleanup {
        lock_state(&self.state).trim_idle()
    }

    pub(crate) fn poison_retained_byte_accounting_for_test(&self) -> ResourceAccountingFault {
        let mut state = lock_state(&self.state);
        let registered_entry_bytes = state
            .checked_registered_entry_bytes()
            .expect("the accounting-poison fixture requires a representable entry total");
        state.retained_bytes = state
            .retained_bytes
            .checked_sub(1)
            .expect("the accounting-poison fixture requires retained bytes");
        let fault = ResourceAccountingFault::RetainedByteMismatch {
            retained_bytes: state.retained_bytes,
            registered_entry_bytes,
        };
        state.record_accounting_fault(fault);
        fault
    }
}

impl FrameCleanup {
    pub(crate) const fn retention(&self) -> ResourceRetentionOutcome {
        self.retention
    }

    pub(crate) fn evicted_resources(&self) -> &[ResourceIdentity] {
        &self.evicted_resources
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResourceLeaseTokenForTest {
    pub(crate) manager_identity: ManagerIdentity,
    pub(crate) frame_identity: FrameIdentity,
    pub(crate) resource_identity: ResourceIdentity,
    pub(crate) allocation_generation: AllocationGeneration,
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResourceManagerObservationForTest {
    pub(crate) idle_count: usize,
    pub(crate) leased_count: usize,
    pub(crate) retained_bytes: u64,
    pub(crate) accounted_entry_bytes: Option<u64>,
    accounting_fault: Option<ResourceAccountingFault>,
    pub(crate) active_frame_count: usize,
    pub(crate) resolved_lease_count: usize,
    entry_identities: Vec<ResourceIdentity>,
    lifecycle_stats: ResourceLifecycleStats,
    pub(crate) next_resource: u64,
    pub(crate) entry_count: usize,
    pub(crate) payload_creation_attempts: u64,
    retained_atlas_count: usize,
    retained_atlas_byte_len: u64,
    committed_transient_buffer_count: usize,
    committed_transient_buffer_byte_len: u64,
    committed_transient_image_count: usize,
    committed_transient_image_byte_len: u64,
    effect_texture_count: usize,
    resolved_mask_upload_keys: Vec<ResolvedMaskUploadKey>,
    gaussian_kernel_count: usize,
    recovery_outcome: Option<VelloAtlasOutcome>,
}

impl ResourceManagerObservationForTest {
    pub(crate) const fn accounting_fault_for_test(&self) -> Option<ResourceAccountingFault> {
        self.accounting_fault
    }

    pub(crate) fn entry_identities_for_test(&self) -> &[ResourceIdentity] {
        &self.entry_identities
    }

    pub(crate) const fn lifecycle_stats_for_test(&self) -> ResourceLifecycleStats {
        self.lifecycle_stats
    }

    pub(crate) const fn retained_count_for_test(&self) -> usize {
        self.entry_count
    }

    pub(crate) const fn retained_atlas_count_for_test(&self) -> usize {
        self.retained_atlas_count
    }

    pub(crate) const fn retained_byte_len_for_test(&self) -> u64 {
        self.retained_bytes
    }

    pub(crate) const fn retained_atlas_byte_len_for_test(&self) -> u64 {
        self.retained_atlas_byte_len
    }

    pub(crate) const fn committed_transient_buffer_count_for_test(&self) -> usize {
        self.committed_transient_buffer_count
    }

    pub(crate) const fn committed_transient_buffer_byte_len_for_test(&self) -> u64 {
        self.committed_transient_buffer_byte_len
    }

    pub(crate) const fn committed_transient_image_count_for_test(&self) -> usize {
        self.committed_transient_image_count
    }

    pub(crate) const fn committed_transient_image_byte_len_for_test(&self) -> u64 {
        self.committed_transient_image_byte_len
    }

    pub(crate) const fn effect_texture_count_for_test(&self) -> usize {
        self.effect_texture_count
    }

    pub(crate) fn resolved_mask_upload_keys_for_test(&self) -> &[ResolvedMaskUploadKey] {
        &self.resolved_mask_upload_keys
    }

    pub(crate) const fn gaussian_kernel_count_for_test(&self) -> usize {
        self.gaussian_kernel_count
    }

    pub(crate) const fn recovery_outcome_for_test(&self) -> Option<VelloAtlasOutcome> {
        self.recovery_outcome
    }
}
