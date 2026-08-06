use super::{
    BackendErrorCode, Error, ResourceCacheBudget, Result,
    gaussian::{GaussianKernelKey, GaussianKernelPlan},
};
use crate::{
    image::{ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey},
    texture::{EffectTextureDescriptor, EffectTextureKey},
    vello_engine::{VelloAtlasOutcome, VelloBufferKey, VelloImageKey},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

#[derive(Clone)]
pub(crate) struct ManagerIdentity(Arc<()>);

impl ManagerIdentity {
    pub(super) fn new() -> Self {
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
pub(crate) struct FrameIdentity(pub(super) u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ResourceIdentity(pub(super) u64);

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
pub(crate) struct AllocationGeneration(pub(super) u64);

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
/// per-device manager.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ResourceCacheKey {
    #[cfg(test)]
    VelloAtlas,
    VelloBuffer(VelloBufferKey),
    VelloImage(VelloImageKey),
    EffectTexture(EffectTextureKey),
    ResolvedMaskUpload(ResolvedMaskUploadKey),
    GaussianKernelBuffer(GaussianKernelKey),
}

impl ResourceCacheKey {
    #[cfg(test)]
    pub(super) const fn accepts_modeled_payload(self) -> bool {
        matches!(self, Self::VelloAtlas | Self::EffectTexture(_))
    }

    pub(super) const fn is_vello_atlas(self) -> bool {
        matches!(self, Self::VelloImage(key) if key.is_persistent_atlas())
    }

    const fn accepts_graph_preparation(self) -> bool {
        matches!(
            self,
            Self::EffectTexture(_) | Self::ResolvedMaskUpload(_) | Self::GaussianKernelBuffer(_)
        )
    }

    #[cfg(test)]
    pub(super) const fn is_vello_buffer(self) -> bool {
        matches!(self, Self::VelloBuffer(_))
    }

    #[cfg(test)]
    pub(super) const fn is_transient_vello_image(self) -> bool {
        matches!(self, Self::VelloImage(key) if !key.is_persistent_atlas())
    }
}

/// Immutable concrete-allocation facts used to preflight a complete graph
/// before opening its frame resource scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceAllocationPreflight {
    key: ResourceCacheKey,
    byte_len: u64,
}

impl ResourceAllocationPreflight {
    pub(crate) fn effect_texture(descriptor: EffectTextureDescriptor) -> Result<Self> {
        Ok(Self {
            key: ResourceCacheKey::EffectTexture(descriptor.cache_key()),
            byte_len: descriptor.checked_byte_len()?,
        })
    }

    pub(crate) fn resolved_mask(descriptor: &ResolvedMaskUploadDescriptor) -> Result<Option<Self>> {
        let size = descriptor.physical_size();
        if size.width() == 0 || size.height() == 0 {
            return Ok(None);
        }
        descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
        Ok(Some(Self {
            key: ResourceCacheKey::ResolvedMaskUpload(descriptor.cache_key()),
            byte_len: descriptor.byte_len(),
        }))
    }

    #[cfg(test)]
    pub(crate) fn zero_sized_mask_is_explicitly_empty_for_test(
        descriptor: &ResolvedMaskUploadDescriptor,
    ) -> bool {
        Self::resolved_mask(descriptor).is_ok_and(|preflight| preflight.is_none())
    }

    pub(crate) fn gaussian_kernel(plan: &GaussianKernelPlan) -> Result<Self> {
        plan.validate_upload_byte_len(plan.upload_bytes().len())?;
        Ok(Self {
            key: ResourceCacheKey::GaussianKernelBuffer(plan.key()),
            byte_len: plan.byte_len(),
        })
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
pub(crate) enum ResourceAccountingFault {
    RetainedByteUnderflow {
        retained_bytes: u64,
        discarded_entry_bytes: u64,
    },
    SurvivingEntryByteTotalOverflow,
    RetainedByteMismatch {
        retained_bytes: u64,
        registered_entry_bytes: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResourceEntryState {
    Idle { last_used_frame: FrameIdentity },
    Leased { frame: FrameIdentity },
}

#[derive(Clone)]
pub(super) enum ResourcePayload {
    #[cfg(test)]
    Modeled,
    VelloBuffer {
        buffer: wgpu::Buffer,
    },
    VelloImage {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    EffectTexture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    ResolvedMaskUpload {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    GaussianKernelBuffer {
        buffer: wgpu::Buffer,
    },
}

impl ResourcePayload {
    pub(super) const fn matches_key(&self, key: ResourceCacheKey) -> bool {
        match (self, key) {
            #[cfg(test)]
            (Self::Modeled, ResourceCacheKey::VelloAtlas | ResourceCacheKey::EffectTexture(_)) => {
                true
            }
            (Self::VelloBuffer { .. }, ResourceCacheKey::VelloBuffer(_))
            | (Self::VelloImage { .. }, ResourceCacheKey::VelloImage(_))
            | (Self::EffectTexture { .. }, ResourceCacheKey::EffectTexture(_))
            | (Self::ResolvedMaskUpload { .. }, ResourceCacheKey::ResolvedMaskUpload(_))
            | (Self::GaussianKernelBuffer { .. }, ResourceCacheKey::GaussianKernelBuffer(_)) => {
                true
            }
            _ => false,
        }
    }

    pub(super) const fn label(&self) -> &'static str {
        match self {
            #[cfg(test)]
            Self::Modeled => "Modeled",
            Self::VelloBuffer { .. } => "VelloBuffer",
            Self::VelloImage { .. } => "VelloImage",
            Self::EffectTexture { .. } => "EffectTexture",
            Self::ResolvedMaskUpload { .. } => "ResolvedMaskUpload",
            Self::GaussianKernelBuffer { .. } => "GaussianKernelBuffer",
        }
    }
}

pub(super) struct ResourceEntry {
    pub(super) key: ResourceCacheKey,
    pub(super) allocation_generation: AllocationGeneration,
    pub(super) byte_len: u64,
    pub(super) state: ResourceEntryState,
    pub(super) payload: ResourcePayload,
}

pub(super) struct ResourceManagerState {
    pub(super) identity: ManagerIdentity,
    pub(super) budget: ResourceCacheBudget,
    pub(super) next_frame: u64,
    pub(super) next_resource: u64,
    pub(super) retained_bytes: u64,
    pub(super) accounting_fault: Option<ResourceAccountingFault>,
    pub(super) active_frames: BTreeSet<FrameIdentity>,
    pub(super) resolved_leases: BTreeSet<(FrameIdentity, ResourceIdentity)>,
    pub(super) provisional_allocations: BTreeSet<(FrameIdentity, ResourceIdentity)>,
    pub(super) entries: BTreeMap<ResourceIdentity, ResourceEntry>,
    pub(super) pending_vello_atlas_recovery: Option<VelloAtlasOutcome>,
    pub(super) stats: ResourceLifecycleStats,
    #[cfg(test)]
    pub(super) payload_creation_attempts: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IdleReuse {
    Allowed,
    Fresh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResourceAcquisitionSource {
    Allocation,
    Reuse,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FrameResourceAcquisitions {
    allocations: usize,
    reuses: usize,
}

impl FrameResourceAcquisitions {
    pub(super) fn record(&mut self, source: ResourceAcquisitionSource) {
        match source {
            ResourceAcquisitionSource::Allocation => {
                self.allocations = self.allocations.saturating_add(1);
            }
            ResourceAcquisitionSource::Reuse => {
                self.reuses = self.reuses.saturating_add(1);
            }
        }
    }

    pub(super) fn followed_by(self, later: Self) -> Self {
        Self {
            allocations: self.allocations.saturating_add(later.allocations),
            reuses: self.reuses.saturating_add(later.reuses),
        }
    }

    pub(crate) const fn allocations(self) -> usize {
        self.allocations
    }

    pub(crate) const fn reuses(self) -> usize {
        self.reuses
    }
}

impl ResourceManagerState {
    const ACCOUNTING_FAULT_MESSAGE: &'static str =
        "resource manager is unavailable after a retained-byte accounting invariant failure";

    pub(super) fn accounting_fault_error() -> Error {
        Error::new(
            BackendErrorCode::RenderFailed,
            Self::ACCOUNTING_FAULT_MESSAGE,
        )
    }

    pub(super) fn ensure_accounting_healthy(&self) -> Result<()> {
        if self.accounting_fault.is_some() {
            return Err(Self::accounting_fault_error());
        }
        Ok(())
    }

    pub(super) fn record_accounting_fault(&mut self, fault: ResourceAccountingFault) {
        if self.accounting_fault.is_none() {
            self.accounting_fault = Some(fault);
        }
    }

    pub(super) fn fail_accounting<T>(&mut self, fault: ResourceAccountingFault) -> Result<T> {
        self.record_accounting_fault(fault);
        Err(Self::accounting_fault_error())
    }

    pub(super) fn checked_registered_entry_bytes(&self) -> Option<u64> {
        self.entries
            .values()
            .try_fold(0_u64, |bytes, entry| bytes.checked_add(entry.byte_len))
    }

    pub(super) fn checked_idle_summary(&self) -> Option<(usize, u64)> {
        let retained_count = self
            .entries
            .values()
            .filter(|entry| matches!(entry.state, ResourceEntryState::Idle { .. }))
            .count();
        let retained_byte_len = self
            .entries
            .values()
            .filter_map(|entry| match entry.state {
                ResourceEntryState::Idle { .. } => Some(entry.byte_len),
                ResourceEntryState::Leased { .. } => None,
            })
            .try_fold(0_u64, u64::checked_add)?;
        Some((retained_count, retained_byte_len))
    }

    pub(super) fn ensure_accounting_exact(&mut self) -> Result<()> {
        self.ensure_accounting_healthy()?;
        let Some(registered_entry_bytes) = self.checked_registered_entry_bytes() else {
            return self.fail_accounting(ResourceAccountingFault::SurvivingEntryByteTotalOverflow);
        };
        if self.retained_bytes != registered_entry_bytes {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteMismatch {
                retained_bytes: self.retained_bytes,
                registered_entry_bytes,
            });
        }
        Ok(())
    }

    pub(super) fn preflight_graph_acquisitions(
        &mut self,
        requests: &[ResourceAllocationPreflight],
    ) -> Result<()> {
        self.ensure_accounting_exact()?;
        let mut selected_idle = BTreeSet::new();
        let mut retained_bytes = self.retained_bytes;
        let mut next_resource = self.next_resource;

        for request in requests {
            if !request.key.accepts_graph_preparation() {
                return Err(Error::invalid_value(
                    "graph resource preflight key",
                    format!("{:?}", request.key),
                    "must use an effect texture, resolved mask, or Gaussian kernel namespace",
                ));
            }
            if request.byte_len == 0 {
                return Err(Error::invalid_value(
                    "graph resource preflight byte length",
                    request.byte_len,
                    "must be greater than zero",
                ));
            }

            let reusable = self
                .entries
                .iter()
                .filter_map(|(identity, entry)| match entry.state {
                    ResourceEntryState::Idle { last_used_frame }
                        if entry.key == request.key
                            && entry.byte_len == request.byte_len
                            && !selected_idle.contains(identity) =>
                    {
                        Some((last_used_frame, *identity))
                    }
                    ResourceEntryState::Idle { .. } | ResourceEntryState::Leased { .. } => None,
                })
                .min()
                .map(|(_, identity)| identity);
            if let Some(identity) = reusable {
                selected_idle.insert(identity);
                continue;
            }

            next_resource = next_resource.checked_add(1).ok_or_else(|| {
                Error::invalid_value(
                    "resource identity",
                    next_resource,
                    "must have remaining identity space for every prepared allocation",
                )
            })?;
            retained_bytes = retained_bytes
                .checked_add(request.byte_len)
                .ok_or_else(|| {
                    Error::invalid_value(
                        "retained resource byte length",
                        format!("{retained_bytes} + {}", request.byte_len),
                        "must fit in u64",
                    )
                })?;
        }

        Ok(())
    }

    pub(super) fn record_vello_atlas_recovery(&mut self, outcome: VelloAtlasOutcome) {
        self.pending_vello_atlas_recovery =
            VelloAtlasOutcome::merge_pending_recovery(self.pending_vello_atlas_recovery, outcome);
    }

    pub(super) fn consume_vello_atlas_recovery(&mut self) {
        let _ = self.pending_vello_atlas_recovery.take();
    }

    pub(super) fn retire_idle_vello_atlases(&mut self) -> Result<()> {
        self.ensure_accounting_exact()?;
        let retired = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| {
                (entry.key.is_vello_atlas()
                    && matches!(entry.state, ResourceEntryState::Idle { .. }))
                .then_some((*identity, entry.byte_len))
            })
            .collect::<Vec<_>>();
        let Some(retired_bytes) = retired
            .iter()
            .try_fold(0_u64, |total, (_, byte_len)| total.checked_add(*byte_len))
        else {
            return self.fail_accounting(ResourceAccountingFault::SurvivingEntryByteTotalOverflow);
        };
        let Some(retained_bytes) = self.retained_bytes.checked_sub(retired_bytes) else {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteUnderflow {
                retained_bytes: self.retained_bytes,
                discarded_entry_bytes: retired_bytes,
            });
        };
        for (identity, byte_len) in retired {
            let removed = self
                .entries
                .remove(&identity)
                .expect("an idle Vello atlas selected for replacement must remain registered");
            debug_assert_eq!(removed.byte_len, byte_len);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.retained_bytes = retained_bytes;
        self.ensure_accounting_exact()
    }
}
