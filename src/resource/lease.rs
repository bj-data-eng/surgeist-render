use super::{
    AllocationGeneration, FrameIdentity, FrameResourceAcquisitions, ManagerIdentity,
    ResourceAccountingFault, ResourceCacheKey, ResourceIdentity,
    gaussian::{GaussianKernelBufferLimits, GaussianKernelPlan},
    lock_state,
    manager::{
        IdleReuse, ResourceAcquisitionSource, ResourceEntry, ResourceEntryState,
        ResourceManagerState, ResourcePayload,
    },
};
use crate::{
    BackendErrorCode, Error, Result,
    backend::DeviceCapabilities,
    image::ResolvedMaskUploadDescriptor,
    texture::EffectTextureDescriptor,
    vello_engine::{VelloAtlasOutcome, VelloBufferKey, VelloImageKey},
};
use std::{
    collections::BTreeSet,
    fmt,
    sync::{Arc, Mutex},
};
use wgpu::util::DeviceExt;

impl ResourceManagerState {
    fn ensure_frame_resolution_ready(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        tokens: &[&ResourceLeaseToken],
    ) -> Result<()> {
        self.ensure_accounting_exact()?;
        if self.identity != *manager_identity || !self.active_frames.contains(&frame) {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "resource frame is unavailable for clean accounting resolution",
            ));
        }

        let mut covered = self
            .resolved_leases
            .iter()
            .filter_map(|(resolved_frame, resource)| {
                (*resolved_frame == frame).then_some(*resource)
            })
            .collect::<BTreeSet<_>>();
        for token in tokens {
            if !covered.insert(token.resource_identity) {
                return Err(Self::invalid_lease(
                    token.resource_identity,
                    "must occur exactly once in one clean frame resolution",
                ));
            }
            self.validate_lease(manager_identity, frame, token)?;
        }
        let leased = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| {
                (entry.state == (ResourceEntryState::Leased { frame })).then_some(*identity)
            })
            .collect::<BTreeSet<_>>();
        if covered != leased {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "clean resource resolution must own every exact lease in its frame",
            ));
        }
        Ok(())
    }

    fn invalid_lease(resource: ResourceIdentity, reason: &'static str) -> Error {
        Error::invalid_value("resource lease", resource.get(), reason)
    }

    fn acquire_with_payload(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        key: ResourceCacheKey,
        byte_len: u64,
        idle_reuse: IdleReuse,
        create_payload: impl FnOnce() -> Result<ResourcePayload>,
    ) -> Result<ResourceLease> {
        self.ensure_accounting_exact()?;
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

        let reusable = match idle_reuse {
            IdleReuse::Allowed => self
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
                .map(|(_, identity)| identity),
            IdleReuse::Fresh => None,
        };

        let (resource, allocation_generation, payload) = if let Some(resource) = reusable {
            let entry = self
                .entries
                .get_mut(&resource)
                .expect("the selected idle resource must remain registered");
            debug_assert!(entry.payload.matches_key(key));
            entry.state = ResourceEntryState::Leased { frame };
            self.stats.hits = self.stats.hits.saturating_add(1);
            (resource, entry.allocation_generation, entry.payload.clone())
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
            #[cfg(test)]
            {
                self.payload_creation_attempts = self.payload_creation_attempts.saturating_add(1);
            }
            let payload = create_payload()?;
            if !payload.matches_key(key) {
                return Err(Error::invalid_value(
                    "resource payload",
                    payload.label(),
                    "must match its exact resource cache-key namespace",
                ));
            }
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
                    payload: payload.clone(),
                },
            );
            self.provisional_allocations.insert((frame, resource));
            self.stats.misses = self.stats.misses.saturating_add(1);
            self.stats.allocations = self.stats.allocations.saturating_add(1);
            (resource, allocation_generation, payload)
        };

        Ok(ResourceLease::new(
            manager_identity.clone(),
            frame,
            resource,
            allocation_generation,
            payload,
            if reusable.is_some() {
                ResourceAcquisitionSource::Reuse
            } else {
                ResourceAcquisitionSource::Allocation
            },
        ))
    }

    #[cfg(test)]
    pub(super) fn acquire(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        if !key.accepts_modeled_payload() {
            return Err(Error::invalid_value(
                "modeled resource key",
                format!("{key:?}"),
                "must use a Vello atlas or effect texture namespace",
            ));
        }
        self.acquire_with_payload(
            manager_identity,
            frame,
            key,
            byte_len,
            IdleReuse::Allowed,
            || Ok(ResourcePayload::Modeled),
        )
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

    pub(super) fn release(
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

    fn resolve_leases_atomically(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        tokens: &[&ResourceLeaseToken],
    ) -> Result<()> {
        let mut resources = BTreeSet::new();
        for token in tokens {
            if !resources.insert(token.resource_identity) {
                return Err(Self::invalid_lease(
                    token.resource_identity,
                    "must occur at most once in one atomic release",
                ));
            }
            self.validate_lease(manager_identity, frame, token)?;
        }

        for token in tokens {
            self.resolved_leases
                .insert((frame, token.resource_identity));
            self.stats.releases = self.stats.releases.saturating_add(1);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn replace(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: ResourceLeaseToken,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        self.ensure_accounting_healthy()?;
        self.validate_lease(manager_identity, frame, &token)?;
        if !key.accepts_modeled_payload() {
            return Err(Error::invalid_value(
                "modeled resource key",
                format!("{key:?}"),
                "must use a Vello atlas or effect texture namespace",
            ));
        }
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
        let replaced_byte_len = entry.byte_len;
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
        let Some(retained_without_replaced) = self.retained_bytes.checked_sub(replaced_byte_len)
        else {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteUnderflow {
                retained_bytes: self.retained_bytes,
                discarded_entry_bytes: replaced_byte_len,
            });
        };
        let Some(registered_entry_bytes) = self.checked_registered_entry_bytes() else {
            return self.fail_accounting(ResourceAccountingFault::SurvivingEntryByteTotalOverflow);
        };
        if self.retained_bytes != registered_entry_bytes {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteMismatch {
                retained_bytes: self.retained_bytes,
                registered_entry_bytes,
            });
        }
        let Some(retained_bytes) = retained_without_replaced.checked_add(byte_len) else {
            return self.fail_accounting(ResourceAccountingFault::SurvivingEntryByteTotalOverflow);
        };
        let Some(registered_without_replaced) =
            registered_entry_bytes.checked_sub(replaced_byte_len)
        else {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteUnderflow {
                retained_bytes: registered_entry_bytes,
                discarded_entry_bytes: replaced_byte_len,
            });
        };
        let Some(replaced_entry_bytes) = registered_without_replaced.checked_add(byte_len) else {
            return self.fail_accounting(ResourceAccountingFault::SurvivingEntryByteTotalOverflow);
        };
        if retained_bytes != replaced_entry_bytes {
            return self.fail_accounting(ResourceAccountingFault::RetainedByteMismatch {
                retained_bytes,
                registered_entry_bytes: replaced_entry_bytes,
            });
        }

        let entry = self
            .entries
            .get_mut(&token.resource_identity)
            .expect("a validated resource lease must remain registered");
        entry.key = key;
        entry.allocation_generation = allocation_generation;
        entry.byte_len = byte_len;
        entry.payload = ResourcePayload::Modeled;
        self.retained_bytes = retained_bytes;
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        Ok(ResourceLease::new(
            manager_identity.clone(),
            frame,
            token.resource_identity,
            allocation_generation,
            ResourcePayload::Modeled,
            ResourceAcquisitionSource::Allocation,
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
        self.stats.evictions = self.stats.evictions.saturating_add(1);

        if self.accounting_fault.is_some() {
            return Ok(());
        }

        let Some(retained_bytes) = self.retained_bytes.checked_sub(entry.byte_len) else {
            self.record_accounting_fault(ResourceAccountingFault::RetainedByteUnderflow {
                retained_bytes: self.retained_bytes,
                discarded_entry_bytes: entry.byte_len,
            });
            return Err(Self::accounting_fault_error());
        };
        self.retained_bytes = retained_bytes;

        let detected_fault = match self.checked_registered_entry_bytes() {
            None => Some(ResourceAccountingFault::SurvivingEntryByteTotalOverflow),
            Some(registered_entry_bytes) if retained_bytes != registered_entry_bytes => {
                Some(ResourceAccountingFault::RetainedByteMismatch {
                    retained_bytes,
                    registered_entry_bytes,
                })
            }
            Some(_) => None,
        };
        if let Some(fault) = detected_fault {
            self.record_accounting_fault(fault);
            return Err(Self::accounting_fault_error());
        }

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
        self.provisional_allocations
            .retain(|(provisional_frame, _)| *provisional_frame != frame);
        if let Some(fault) = self.accounting_fault {
            return FrameCleanup::accounting_fault(fault);
        }
        self.trim_idle()
    }

    fn abort_provisional_frame(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
    ) -> FrameCleanup {
        if self.identity != *manager_identity || !self.active_frames.contains(&frame) {
            return FrameCleanup::default();
        }

        for (resource_identity, entry) in &mut self.entries {
            if entry.state == (ResourceEntryState::Leased { frame })
                && !self
                    .provisional_allocations
                    .contains(&(frame, *resource_identity))
            {
                entry.state = ResourceEntryState::Idle {
                    last_used_frame: frame,
                };
                if !self.resolved_leases.remove(&(frame, *resource_identity)) {
                    self.stats.releases = self.stats.releases.saturating_add(1);
                }
            }
        }
        self.discard_frame(manager_identity, frame)
    }

    fn discard_frame(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
    ) -> FrameCleanup {
        if self.identity != *manager_identity || !self.active_frames.remove(&frame) {
            return FrameCleanup::default();
        }

        let discarded = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| {
                (entry.state == (ResourceEntryState::Leased { frame })).then_some(*identity)
            })
            .collect::<Vec<_>>();
        let mut cleanup = FrameCleanup::default();
        let mut retained_after_discard = self
            .accounting_fault
            .is_none()
            .then_some(self.retained_bytes);
        let mut detected_fault = None;
        for identity in discarded {
            if let Some(entry) = self.entries.remove(&identity) {
                if let Some(retained_bytes) = retained_after_discard {
                    if let Some(remaining) = retained_bytes.checked_sub(entry.byte_len) {
                        retained_after_discard = Some(remaining);
                    } else {
                        detected_fault = Some(ResourceAccountingFault::RetainedByteUnderflow {
                            retained_bytes,
                            discarded_entry_bytes: entry.byte_len,
                        });
                        retained_after_discard = None;
                    }
                }
                self.stats.evictions = self.stats.evictions.saturating_add(1);
                cleanup.evicted_resources.push(identity);
            }
        }
        if let Some(retained_bytes) = retained_after_discard {
            self.retained_bytes = retained_bytes;
        }
        self.resolved_leases
            .retain(|(resolved_frame, _)| *resolved_frame != frame);
        self.provisional_allocations
            .retain(|(provisional_frame, _)| *provisional_frame != frame);

        if detected_fault.is_none() && self.accounting_fault.is_none() {
            detected_fault = match (
                retained_after_discard,
                self.checked_registered_entry_bytes(),
            ) {
                (_, None) => Some(ResourceAccountingFault::SurvivingEntryByteTotalOverflow),
                (Some(retained_bytes), Some(registered_entry_bytes))
                    if retained_bytes != registered_entry_bytes =>
                {
                    Some(ResourceAccountingFault::RetainedByteMismatch {
                        retained_bytes,
                        registered_entry_bytes,
                    })
                }
                (Some(_), Some(_)) => None,
                (None, Some(registered_entry_bytes)) => {
                    Some(ResourceAccountingFault::RetainedByteMismatch {
                        retained_bytes: self.retained_bytes,
                        registered_entry_bytes,
                    })
                }
            };
        }
        if let Some(fault) = detected_fault {
            self.record_accounting_fault(fault);
        }
        if let Some(fault) = self.accounting_fault {
            cleanup.retention = ResourceRetentionOutcome::AccountingFault { fault };
            return cleanup;
        }

        let Some((retained_count, retained_byte_len)) = self.checked_idle_summary() else {
            let fault = ResourceAccountingFault::SurvivingEntryByteTotalOverflow;
            self.record_accounting_fault(fault);
            cleanup.retention = ResourceRetentionOutcome::AccountingFault { fault };
            return cleanup;
        };
        cleanup.retention = if retained_count == 0 {
            ResourceRetentionOutcome::NoIdleResources
        } else {
            ResourceRetentionOutcome::RetainedReusable {
                resource_count: retained_count,
                byte_len: retained_byte_len,
            }
        };
        cleanup
    }

    pub(super) fn trim_idle(&mut self) -> FrameCleanup {
        if let Some(fault) = self.accounting_fault {
            return FrameCleanup::accounting_fault(fault);
        }
        if self.ensure_accounting_exact().is_err() {
            return FrameCleanup::accounting_fault(
                self.accounting_fault
                    .expect("failed accounting validation must preserve its first fault"),
            );
        }
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
            let Some(retained_bytes) = self.retained_bytes.checked_sub(byte_len) else {
                let fault = ResourceAccountingFault::RetainedByteUnderflow {
                    retained_bytes: self.retained_bytes,
                    discarded_entry_bytes: byte_len,
                };
                self.record_accounting_fault(fault);
                cleanup.retention = ResourceRetentionOutcome::AccountingFault { fault };
                return cleanup;
            };
            self.retained_bytes = retained_bytes;
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            cleanup.evicted_resources.push(identity);
        }
        if self.ensure_accounting_exact().is_err() {
            cleanup.retention = ResourceRetentionOutcome::AccountingFault {
                fault: self
                    .accounting_fault
                    .expect("failed accounting validation must preserve its first fault"),
            };
            return cleanup;
        }
        let Some((retained_count, retained_byte_len)) = self.checked_idle_summary() else {
            let fault = ResourceAccountingFault::SurvivingEntryByteTotalOverflow;
            self.record_accounting_fault(fault);
            cleanup.retention = ResourceRetentionOutcome::AccountingFault { fault };
            return cleanup;
        };
        cleanup.retention = if cleanup.evicted_resources.is_empty() {
            if retained_count == 0 {
                ResourceRetentionOutcome::NoIdleResources
            } else {
                ResourceRetentionOutcome::RetainedReusable {
                    resource_count: retained_count,
                    byte_len: retained_byte_len,
                }
            }
        } else {
            ResourceRetentionOutcome::Trimmed {
                released_count: cleanup.evicted_resources.len(),
                retained_count,
                retained_byte_len,
            }
        };
        cleanup
    }
}

pub(super) struct ResourceLeaseToken {
    pub(super) manager_identity: ManagerIdentity,
    pub(super) frame_identity: FrameIdentity,
    pub(super) resource_identity: ResourceIdentity,
    pub(super) allocation_generation: AllocationGeneration,
}

/// A production lease is intentionally neither `Clone` nor `Copy`; every
/// resolving operation consumes it.
#[must_use = "resource leases must be resolved by their owning frame scope"]
pub(crate) struct ResourceLease {
    pub(super) token: ResourceLeaseToken,
    payload: ResourcePayload,
    acquisition_source: ResourceAcquisitionSource,
}

impl ResourceLease {
    fn new(
        manager_identity: ManagerIdentity,
        frame_identity: FrameIdentity,
        resource_identity: ResourceIdentity,
        allocation_generation: AllocationGeneration,
        payload: ResourcePayload,
        acquisition_source: ResourceAcquisitionSource,
    ) -> Self {
        Self {
            token: ResourceLeaseToken {
                manager_identity,
                frame_identity,
                resource_identity,
                allocation_generation,
            },
            payload,
            acquisition_source,
        }
    }

    pub(crate) const fn resource_identity(&self) -> ResourceIdentity {
        self.token.resource_identity
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
            .field("payload", &self.payload.label())
            .finish()
    }
}

#[must_use = "frame resource scopes clean outstanding leases when resolved or dropped"]
pub(crate) struct FrameResourceScope {
    pub(super) state: Arc<Mutex<ResourceManagerState>>,
    pub(super) manager_identity: ManagerIdentity,
    pub(super) frame: FrameIdentity,
    acquisitions: FrameResourceAcquisitions,
    pending_drop_disposition: Option<FrameResourceDisposition>,
}

#[derive(Clone, Copy)]
enum FrameResourceDisposition {
    ReleaseReusable,
    AbortProvisional,
    Discard,
}

impl FrameResourceScope {
    pub(super) fn new(
        state: Arc<Mutex<ResourceManagerState>>,
        manager_identity: ManagerIdentity,
        frame: FrameIdentity,
    ) -> Self {
        Self {
            state,
            manager_identity,
            frame,
            acquisitions: FrameResourceAcquisitions::default(),
            pending_drop_disposition: Some(FrameResourceDisposition::ReleaseReusable),
        }
    }

    pub(super) fn record_acquisition(&mut self, lease: ResourceLease) -> ResourceLease {
        self.acquisitions.record(lease.acquisition_source);
        lease
    }

    fn validated_payload<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<&'scope ResourcePayload> {
        lock_state(&self.state).validate_lease(&self.manager_identity, self.frame, &lease.token)?;
        Ok(&lease.payload)
    }

    pub(crate) fn vello_buffer<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<&'scope wgpu::Buffer> {
        match self.validated_payload(lease)? {
            ResourcePayload::VelloBuffer { buffer } => Ok(buffer),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain an internal Vello buffer",
            )),
        }
    }

    pub(crate) fn vello_image<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<(&'scope wgpu::Texture, &'scope wgpu::TextureView)> {
        match self.validated_payload(lease)? {
            ResourcePayload::VelloImage { texture, view } => Ok((texture, view)),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain an internal Vello image",
            )),
        }
    }

    pub(crate) fn acquire_vello_buffer(
        &mut self,
        device: &wgpu::Device,
        key: VelloBufferKey,
    ) -> Result<ResourceLease> {
        self.acquire_vello_buffer_with_reuse(device, key, IdleReuse::Fresh)
    }

    pub(crate) fn acquire_reusable_vello_buffer(
        &mut self,
        device: &wgpu::Device,
        key: VelloBufferKey,
    ) -> Result<ResourceLease> {
        self.acquire_vello_buffer_with_reuse(device, key, IdleReuse::Allowed)
    }

    fn acquire_vello_buffer_with_reuse(
        &mut self,
        device: &wgpu::Device,
        key: VelloBufferKey,
        idle_reuse: IdleReuse,
    ) -> Result<ResourceLease> {
        let byte_len = key.byte_len();
        let lease = lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            ResourceCacheKey::VelloBuffer(key),
            byte_len,
            idle_reuse,
            || {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Surgeist internal Vello buffer"),
                    size: byte_len,
                    usage: key.usage(),
                    mapped_at_creation: false,
                });
                Ok(ResourcePayload::VelloBuffer { buffer })
            },
        )?;
        Ok(self.record_acquisition(lease))
    }

    pub(crate) fn acquire_vello_image(
        &mut self,
        device: &wgpu::Device,
        key: VelloImageKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        self.acquire_vello_image_with_reuse(device, key, byte_len, IdleReuse::Fresh)
    }

    pub(crate) fn acquire_reusable_vello_image(
        &mut self,
        device: &wgpu::Device,
        key: VelloImageKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        self.acquire_vello_image_with_reuse(device, key, byte_len, IdleReuse::Allowed)
    }

    fn acquire_vello_image_with_reuse(
        &mut self,
        device: &wgpu::Device,
        key: VelloImageKey,
        byte_len: u64,
        idle_reuse: IdleReuse,
    ) -> Result<ResourceLease> {
        let lease = lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            ResourceCacheKey::VelloImage(key),
            byte_len,
            idle_reuse,
            || {
                let format = key.texture_format();
                let extent = key.extent();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Surgeist internal Vello image"),
                    size: wgpu::Extent3d {
                        width: extent.width(),
                        height: extent.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: key.usage(),
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
                Ok(ResourcePayload::VelloImage { texture, view })
            },
        )?;
        Ok(self.record_acquisition(lease))
    }

    pub(crate) fn effect_texture<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<(&'scope wgpu::Texture, &'scope wgpu::TextureView)> {
        match self.validated_payload(lease)? {
            ResourcePayload::EffectTexture { texture, view } => Ok((texture, view)),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain an effect texture",
            )),
        }
    }

    pub(crate) fn resolved_mask_texture<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<(&'scope wgpu::Texture, &'scope wgpu::TextureView)> {
        match self.validated_payload(lease)? {
            ResourcePayload::ResolvedMaskUpload { texture, view } => Ok((texture, view)),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain a resolved-mask texture",
            )),
        }
    }

    pub(crate) fn gaussian_kernel_buffer<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<&'scope wgpu::Buffer> {
        match self.validated_payload(lease)? {
            ResourcePayload::GaussianKernelBuffer { buffer } => Ok(buffer),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain a Gaussian kernel buffer",
            )),
        }
    }

    pub(crate) fn acquire_effect_texture(
        &mut self,
        device: &wgpu::Device,
        capabilities: &DeviceCapabilities,
        descriptor: EffectTextureDescriptor,
    ) -> Result<ResourceLease> {
        capabilities.validate_effect_texture_allocation(
            descriptor.physical_size(),
            descriptor.working_format(),
            descriptor.texture_format(),
            descriptor.usage(),
        )?;
        let byte_len = descriptor.checked_byte_len()?;
        let key = ResourceCacheKey::EffectTexture(descriptor.cache_key());
        let lease = lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            key,
            byte_len,
            IdleReuse::Allowed,
            || {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(descriptor.label()),
                    size: wgpu::Extent3d {
                        width: descriptor.physical_size().width(),
                        height: descriptor.physical_size().height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: descriptor.texture_format(),
                    usage: descriptor.usage(),
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Ok(ResourcePayload::EffectTexture { texture, view })
            },
        )?;
        Ok(self.record_acquisition(lease))
    }

    pub(crate) fn acquire_resolved_mask_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capabilities: &DeviceCapabilities,
        descriptor: &ResolvedMaskUploadDescriptor,
    ) -> Result<ResourceLease> {
        let physical_size = descriptor.physical_size();
        if physical_size.width() == 0 || physical_size.height() == 0 {
            return Err(Error::invalid_value(
                "resolved mask upload extent",
                format!("{}x{}", physical_size.width(), physical_size.height()),
                "must have positive width and height before GPU allocation",
            ));
        }
        descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST);
        capabilities.validate_effect_texture_allocation(
            physical_size,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
            usage,
        )?;
        let key = ResourceCacheKey::ResolvedMaskUpload(descriptor.cache_key());
        let lease = lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            key,
            descriptor.byte_len(),
            IdleReuse::Allowed,
            || {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Surgeist retained resolved mask upload"),
                    size: wgpu::Extent3d {
                        width: physical_size.width(),
                        height: physical_size.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    descriptor.bytes(),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(descriptor.row_bytes()),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: physical_size.width(),
                        height: physical_size.height(),
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Ok(ResourcePayload::ResolvedMaskUpload { texture, view })
            },
        )?;
        Ok(self.record_acquisition(lease))
    }

    pub(crate) fn acquire_gaussian_kernel_buffer(
        &mut self,
        device: &wgpu::Device,
        plan: &GaussianKernelPlan,
    ) -> Result<ResourceLease> {
        plan.validate_upload_byte_len(plan.upload_bytes().len())?;
        plan.validate_buffer_limits(GaussianKernelBufferLimits::from_device_limits(
            &device.limits(),
        ))?;
        let byte_len = plan.byte_len();
        let key = ResourceCacheKey::GaussianKernelBuffer(plan.key());
        let lease = lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            key,
            byte_len,
            IdleReuse::Allowed,
            || {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Surgeist immutable Gaussian kernel buffer"),
                    contents: plan.upload_bytes(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                Ok(ResourcePayload::GaussianKernelBuffer { buffer })
            },
        )?;
        Ok(self.record_acquisition(lease))
    }

    pub(crate) fn release(&mut self, lease: ResourceLease) -> Result<()> {
        lock_state(&self.state).release(&self.manager_identity, self.frame, lease.token)
    }

    pub(crate) fn resolve_leases_atomically(&mut self, leases: &[&ResourceLease]) -> Result<()> {
        let tokens = leases.iter().map(|lease| &lease.token).collect::<Vec<_>>();
        lock_state(&self.state).resolve_leases_atomically(
            &self.manager_identity,
            self.frame,
            &tokens,
        )
    }

    pub(crate) fn record_vello_atlas_recovery(&mut self, outcome: VelloAtlasOutcome) {
        lock_state(&self.state).record_vello_atlas_recovery(outcome);
    }

    pub(crate) fn consume_vello_atlas_recovery(&mut self) {
        lock_state(&self.state).consume_vello_atlas_recovery();
    }

    pub(crate) fn retire_idle_vello_atlases(&mut self) -> Result<()> {
        lock_state(&self.state).retire_idle_vello_atlases()
    }

    /// Removes the exact validated lease even when accounting becomes faulted.
    /// A newly detected fault returns a bounded error after removal; cleanup
    /// under an existing fault succeeds without replacing its first diagnostic.
    pub(crate) fn discard(&mut self, lease: ResourceLease) -> Result<()> {
        lock_state(&self.state).discard(&self.manager_identity, self.frame, lease.token)
    }

    pub(crate) fn discard_on_drop(&mut self) {
        if self.pending_drop_disposition.is_some() {
            self.pending_drop_disposition = Some(FrameResourceDisposition::Discard);
        }
    }

    pub(crate) fn abort_provisional_on_drop(&mut self) {
        if self.pending_drop_disposition.is_some() {
            self.pending_drop_disposition = Some(FrameResourceDisposition::AbortProvisional);
        }
    }

    pub(crate) fn finish(mut self) -> FrameCleanup {
        self.resolve(FrameResourceDisposition::ReleaseReusable)
    }

    pub(crate) fn ensure_commit_ready(&self, leases: &[&ResourceLease]) -> Result<()> {
        let tokens = leases.iter().map(|lease| &lease.token).collect::<Vec<_>>();
        lock_state(&self.state).ensure_frame_resolution_ready(
            &self.manager_identity,
            self.frame,
            &tokens,
        )
    }

    pub(crate) fn finish_checked(mut self) -> Result<FrameCleanup> {
        self.ensure_commit_ready(&[])?;
        self.resolve(FrameResourceDisposition::ReleaseReusable)
            .into_accounting_result()
    }

    fn resolve(&mut self, disposition: FrameResourceDisposition) -> FrameCleanup {
        if self.pending_drop_disposition.take().is_none() {
            return FrameCleanup::default();
        }
        let mut state = lock_state(&self.state);
        let cleanup = match disposition {
            FrameResourceDisposition::ReleaseReusable => {
                state.cleanup_frame(&self.manager_identity, self.frame)
            }
            FrameResourceDisposition::AbortProvisional => {
                state.abort_provisional_frame(&self.manager_identity, self.frame)
            }
            FrameResourceDisposition::Discard => {
                state.discard_frame(&self.manager_identity, self.frame)
            }
        };
        cleanup.with_acquisitions(self.acquisitions)
    }
}

impl Drop for FrameResourceScope {
    fn drop(&mut self) {
        if let Some(disposition) = self.pending_drop_disposition {
            let _ = self.resolve(disposition);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ResourceRetentionOutcome {
    #[default]
    NoIdleResources,
    RetainedReusable {
        resource_count: usize,
        byte_len: u64,
    },
    Trimmed {
        released_count: usize,
        retained_count: usize,
        retained_byte_len: u64,
    },
    AccountingFault {
        fault: ResourceAccountingFault,
    },
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct FrameCleanup {
    pub(super) evicted_resources: Vec<ResourceIdentity>,
    pub(super) retention: ResourceRetentionOutcome,
    acquisitions: FrameResourceAcquisitions,
}

impl FrameCleanup {
    fn accounting_fault(fault: ResourceAccountingFault) -> Self {
        Self {
            evicted_resources: Vec::new(),
            retention: ResourceRetentionOutcome::AccountingFault { fault },
            acquisitions: FrameResourceAcquisitions::default(),
        }
    }

    pub(crate) fn followed_by(mut self, mut later: Self) -> Self {
        self.evicted_resources.append(&mut later.evicted_resources);
        self.acquisitions = self.acquisitions.followed_by(later.acquisitions);
        self.retention = match (self.retention, later.retention) {
            (ResourceRetentionOutcome::AccountingFault { fault }, _) => {
                ResourceRetentionOutcome::AccountingFault { fault }
            }
            (_, ResourceRetentionOutcome::AccountingFault { fault }) => {
                ResourceRetentionOutcome::AccountingFault { fault }
            }
            (_, later) => later,
        };
        self
    }

    fn with_acquisitions(mut self, acquisitions: FrameResourceAcquisitions) -> Self {
        self.acquisitions = acquisitions;
        self
    }

    pub(crate) const fn acquisitions(&self) -> FrameResourceAcquisitions {
        self.acquisitions
    }

    pub(crate) const fn retained_byte_len(&self) -> u64 {
        match self.retention {
            ResourceRetentionOutcome::NoIdleResources => 0,
            ResourceRetentionOutcome::RetainedReusable { byte_len, .. } => byte_len,
            ResourceRetentionOutcome::Trimmed {
                retained_byte_len, ..
            } => retained_byte_len,
            ResourceRetentionOutcome::AccountingFault { .. } => 0,
        }
    }

    pub(crate) fn into_accounting_result(self) -> Result<Self> {
        if matches!(
            self.retention,
            ResourceRetentionOutcome::AccountingFault { .. }
        ) {
            return Err(ResourceManagerState::accounting_fault_error());
        }
        Ok(self)
    }
}
