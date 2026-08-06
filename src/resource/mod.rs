mod gaussian;
mod lease;
mod manager;
#[cfg(test)]
mod test_support;

pub(crate) use gaussian::{
    GaussianKernelBufferLimits, GaussianKernelKey, GaussianKernelPlan, GaussianKernelSamplingForm,
};
#[cfg(test)]
pub(crate) use lease::ResourceRetentionOutcome;
pub(crate) use lease::{FrameCleanup, FrameResourceScope, ResourceLease};
pub(crate) use manager::{
    AllocationGeneration, FrameIdentity, FrameResourceAcquisitions, ManagerIdentity,
    ResourceAccountingFault, ResourceAllocationPreflight, ResourceCacheKey, ResourceIdentity,
    ResourceLifecycleStats,
};
#[cfg(test)]
pub(crate) use test_support::{ResourceLeaseTokenForTest, ResourceManagerObservationForTest};

use super::{Error, ResourceCacheBudget, Result};
use manager::ResourceManagerState;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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

    pub(crate) fn is_supported_by(self, features: wgpu::TextureFormatFeatures) -> bool {
        features.allowed_usages.contains(self.required_usages())
            && features.flags.contains(self.required_format_features())
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
                accounting_fault: None,
                active_frames: BTreeSet::new(),
                resolved_leases: BTreeSet::new(),
                provisional_allocations: BTreeSet::new(),
                entries: BTreeMap::new(),
                pending_vello_atlas_recovery: None,
                stats: ResourceLifecycleStats::default(),
                #[cfg(test)]
                payload_creation_attempts: 0,
            })),
        }
    }

    pub(crate) fn begin_frame(&self) -> Result<FrameResourceScope> {
        let mut state = lock_state(&self.state);
        state.ensure_accounting_exact()?;
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
        Ok(FrameResourceScope::new(
            Arc::clone(&self.state),
            state.identity.clone(),
            frame,
        ))
    }

    pub(crate) fn preflight_graph_acquisitions(
        &self,
        requests: &[ResourceAllocationPreflight],
    ) -> Result<()> {
        lock_state(&self.state).preflight_graph_acquisitions(requests)
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
