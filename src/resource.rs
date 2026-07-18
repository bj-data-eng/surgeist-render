use super::{
    Error, ResourceCacheBudget, Result,
    backend::DeviceCapabilities,
    image::{ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey},
    texture::{
        EffectTextureDescriptor, EffectTextureKey, TextureCacheKey, TransitionalTextureKey,
        TransitionalTextureRole,
    },
    vello_engine::{VelloAtlasOutcome, VelloBufferKey, VelloImageKey},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};
use wgpu::util::DeviceExt;

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

    #[cfg(test)]
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GaussianKernelSamplingForm {
    PairedLinear,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 retains the validated non-filtering kernel route for exact sampling"
        )
    )]
    FullNearest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GaussianKernelKey {
    standard_deviation_bits: u64,
    raster_scale_bits: u64,
    support_multiple_bits: u64,
    support_radius: u32,
    sampling_form: GaussianKernelSamplingForm,
}

impl GaussianKernelKey {
    pub(crate) const fn from_exact_plan(
        standard_deviation_bits: u64,
        raster_scale_bits: u64,
        support_multiple_bits: u64,
        support_radius: u32,
        sampling_form: GaussianKernelSamplingForm,
    ) -> Self {
        Self {
            standard_deviation_bits,
            raster_scale_bits,
            support_multiple_bits,
            support_radius,
            sampling_form,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GaussianKernelPlan {
    key: GaussianKernelKey,
    upload_bytes: Arc<[u8]>,
    byte_len: u64,
}

impl GaussianKernelPlan {
    pub(crate) fn try_new(
        standard_deviation: f64,
        raster_scale: f64,
        support_multiple: f64,
        sampling_form: GaussianKernelSamplingForm,
    ) -> Result<Self> {
        for (field, value) in [
            ("Gaussian standard deviation", standard_deviation),
            ("Gaussian raster scale", raster_scale),
            ("Gaussian support multiple", support_multiple),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::invalid_value(
                    field,
                    value,
                    "must be finite and greater than zero",
                ));
            }
        }
        let device_standard_deviation = standard_deviation * raster_scale;
        if !device_standard_deviation.is_finite() || device_standard_deviation <= 0.0 {
            return Err(Error::invalid_value(
                "Gaussian device standard deviation",
                device_standard_deviation,
                "must be finite and greater than zero",
            ));
        }
        let support = device_standard_deviation * support_multiple;
        if !support.is_finite() || support > f64::from(u32::MAX) {
            return Err(Error::invalid_value(
                "Gaussian support radius",
                support,
                "must be finite and fit in u32 device pixels",
            ));
        }
        let support_radius = support.ceil() as u32;
        let weight_count = usize::try_from(support_radius)
            .ok()
            .and_then(|radius| radius.checked_add(1))
            .ok_or_else(|| {
                Error::invalid_value(
                    "Gaussian kernel weight count",
                    support_radius,
                    "must fit addressable memory",
                )
            })?;
        let mut positive_weights = Vec::new();
        positive_weights
            .try_reserve_exact(weight_count)
            .map_err(|_| {
                Error::invalid_value(
                    "Gaussian kernel weight count",
                    weight_count,
                    "must fit available addressable memory",
                )
            })?;
        for offset in 0..=support_radius {
            let offset = f64::from(offset);
            let ratio = offset / device_standard_deviation;
            positive_weights.push((-0.5 * ratio * ratio).exp());
        }
        let normalization =
            positive_weights[0] + 2.0 * positive_weights.iter().skip(1).sum::<f64>();
        if !normalization.is_finite() || normalization <= 0.0 {
            return Err(Error::invalid_value(
                "Gaussian kernel normalization",
                normalization,
                "must be finite and greater than zero",
            ));
        }
        for weight in &mut positive_weights {
            *weight /= normalization;
        }

        let sample_count = match sampling_form {
            GaussianKernelSamplingForm::FullNearest => support_radius
                .checked_mul(2)
                .and_then(|count| count.checked_add(1)),
            GaussianKernelSamplingForm::PairedLinear => support_radius
                .checked_add(1)
                .and_then(|count| count.checked_div(2))
                .and_then(|pairs| pairs.checked_mul(2))
                .and_then(|paired| paired.checked_add(1)),
        }
        .ok_or_else(|| {
            Error::invalid_value(
                "Gaussian kernel sample count",
                support_radius,
                "must fit in u32",
            )
        })?;
        let byte_capacity = usize::try_from(sample_count)
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| {
                Error::invalid_value(
                    "Gaussian kernel byte length",
                    sample_count,
                    "must fit addressable memory",
                )
            })?;
        let mut upload_bytes = Vec::new();
        upload_bytes.try_reserve_exact(byte_capacity).map_err(|_| {
            Error::invalid_value(
                "Gaussian kernel byte length",
                byte_capacity,
                "must fit available addressable memory",
            )
        })?;
        append_kernel_sample(&mut upload_bytes, 0.0, positive_weights[0])?;
        match sampling_form {
            GaussianKernelSamplingForm::FullNearest => {
                for offset in 1..=support_radius {
                    let offset_index = usize::try_from(offset)
                        .expect("validated u32 Gaussian offsets must fit usize");
                    let weight = positive_weights[offset_index];
                    append_kernel_sample(&mut upload_bytes, f64::from(offset), weight)?;
                    append_kernel_sample(&mut upload_bytes, -f64::from(offset), weight)?;
                }
            }
            GaussianKernelSamplingForm::PairedLinear => {
                let mut first = 1_u32;
                while first <= support_radius {
                    let second = first
                        .checked_add(1)
                        .filter(|value| *value <= support_radius);
                    let first_index = usize::try_from(first)
                        .expect("validated u32 Gaussian offsets must fit usize");
                    let first_weight = positive_weights[first_index];
                    let (offset, weight) = if let Some(second) = second {
                        let second_index = usize::try_from(second)
                            .expect("validated u32 Gaussian offsets must fit usize");
                        let second_weight = positive_weights[second_index];
                        let weight = first_weight + second_weight;
                        if weight <= 0.0 {
                            return Err(Error::invalid_value(
                                "Gaussian paired sample weight",
                                weight,
                                "must remain greater than zero",
                            ));
                        }
                        let offset = (f64::from(first) * first_weight
                            + f64::from(second) * second_weight)
                            / weight;
                        (offset, weight)
                    } else {
                        (f64::from(first), first_weight)
                    };
                    append_kernel_sample(&mut upload_bytes, offset, weight)?;
                    append_kernel_sample(&mut upload_bytes, -offset, weight)?;
                    let Some(next) = first.checked_add(2) else {
                        break;
                    };
                    first = next;
                }
            }
        }
        if upload_bytes.len() != byte_capacity {
            return Err(Error::invalid_value(
                "Gaussian kernel byte length",
                upload_bytes.len(),
                "must match the exact serialized sample count",
            ));
        }

        let byte_len = u64::try_from(upload_bytes.len()).map_err(|_| {
            Error::invalid_value(
                "Gaussian kernel byte length",
                upload_bytes.len(),
                "must fit in u64",
            )
        })?;
        Ok(Self {
            key: GaussianKernelKey::from_exact_plan(
                standard_deviation.to_bits(),
                raster_scale.to_bits(),
                support_multiple.to_bits(),
                support_radius,
                sampling_form,
            ),
            upload_bytes: upload_bytes.into(),
            byte_len,
        })
    }

    pub(crate) const fn key(&self) -> GaussianKernelKey {
        self.key
    }

    pub(crate) fn upload_bytes(&self) -> &[u8] {
        &self.upload_bytes
    }

    pub(crate) const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn validate_upload_byte_len(&self, actual_len: usize) -> Result<()> {
        if actual_len != self.upload_bytes.len() {
            return Err(Error::invalid_value(
                "Gaussian kernel upload byte length",
                actual_len,
                "must equal the exact serialized kernel plan length",
            ));
        }
        Ok(())
    }
}

fn append_kernel_sample(bytes: &mut Vec<u8>, offset: f64, weight: f64) -> Result<()> {
    let offset = offset as f32;
    let weight = weight as f32;
    if !offset.is_finite() || !weight.is_finite() {
        return Err(Error::invalid_value(
            "Gaussian kernel sample",
            format!("offset {offset}, weight {weight}"),
            "must narrow to finite f32 values",
        ));
    }
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&weight.to_le_bytes());
    Ok(())
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
    TransitionalTexture(TransitionalTextureKey),
}

impl ResourceCacheKey {
    #[cfg(test)]
    pub(crate) const fn transitional_texture(key: TextureCacheKey) -> Self {
        Self::transitional_texture_for_role(TransitionalTextureRole::Offscreen, key)
    }

    pub(crate) const fn transitional_texture_for_role(
        role: TransitionalTextureRole,
        key: TextureCacheKey,
    ) -> Self {
        Self::TransitionalTexture(TransitionalTextureKey::new(role, key))
    }

    #[cfg(test)]
    const fn accepts_modeled_payload(self) -> bool {
        matches!(self, Self::VelloAtlas | Self::TransitionalTexture(_))
    }

    const fn is_vello_atlas(self) -> bool {
        matches!(self, Self::VelloImage(key) if key.is_persistent_atlas())
    }

    const fn accepts_graph_preparation(self) -> bool {
        matches!(
            self,
            Self::EffectTexture(_) | Self::ResolvedMaskUpload(_) | Self::GaussianKernelBuffer(_)
        )
    }

    #[cfg(test)]
    const fn is_vello_buffer(self) -> bool {
        matches!(self, Self::VelloBuffer(_))
    }

    #[cfg(test)]
    const fn is_transient_vello_image(self) -> bool {
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

    pub(crate) fn resolved_mask(descriptor: &ResolvedMaskUploadDescriptor) -> Result<Self> {
        descriptor.validate_upload_byte_len(descriptor.bytes().len())?;
        Ok(Self {
            key: ResourceCacheKey::ResolvedMaskUpload(descriptor.cache_key()),
            byte_len: descriptor.byte_len(),
        })
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
enum ResourceEntryState {
    Idle { last_used_frame: FrameIdentity },
    Leased { frame: FrameIdentity },
}

#[derive(Clone)]
enum ResourcePayload {
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
    TransitionalTexture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
}

impl ResourcePayload {
    const fn matches_key(&self, key: ResourceCacheKey) -> bool {
        match (self, key) {
            #[cfg(test)]
            (
                Self::Modeled,
                ResourceCacheKey::VelloAtlas | ResourceCacheKey::TransitionalTexture(_),
            ) => true,
            (Self::VelloBuffer { .. }, ResourceCacheKey::VelloBuffer(_))
            | (Self::VelloImage { .. }, ResourceCacheKey::VelloImage(_))
            | (Self::EffectTexture { .. }, ResourceCacheKey::EffectTexture(_))
            | (Self::ResolvedMaskUpload { .. }, ResourceCacheKey::ResolvedMaskUpload(_))
            | (Self::GaussianKernelBuffer { .. }, ResourceCacheKey::GaussianKernelBuffer(_))
            | (Self::TransitionalTexture { .. }, ResourceCacheKey::TransitionalTexture(_)) => true,
            _ => false,
        }
    }

    const fn label(&self) -> &'static str {
        match self {
            #[cfg(test)]
            Self::Modeled => "Modeled",
            Self::VelloBuffer { .. } => "VelloBuffer",
            Self::VelloImage { .. } => "VelloImage",
            Self::EffectTexture { .. } => "EffectTexture",
            Self::ResolvedMaskUpload { .. } => "ResolvedMaskUpload",
            Self::GaussianKernelBuffer { .. } => "GaussianKernelBuffer",
            Self::TransitionalTexture { .. } => "TransitionalTexture",
        }
    }
}

struct ResourceEntry {
    key: ResourceCacheKey,
    allocation_generation: AllocationGeneration,
    byte_len: u64,
    state: ResourceEntryState,
    payload: ResourcePayload,
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
    pending_vello_atlas_recovery: Option<VelloAtlasOutcome>,
    stats: ResourceLifecycleStats,
    #[cfg(test)]
    payload_creation_attempts: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdleReuse {
    Allowed,
    Fresh,
}

impl ResourceManagerState {
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
        ))
    }

    fn preflight_graph_acquisitions(&self, requests: &[ResourceAllocationPreflight]) -> Result<()> {
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

    #[cfg(test)]
    fn acquire(
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
                "must use a Vello or transitional namespace",
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
    fn replace(
        &mut self,
        manager_identity: &ManagerIdentity,
        frame: FrameIdentity,
        token: ResourceLeaseToken,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        self.validate_lease(manager_identity, frame, &token)?;
        if !key.accepts_modeled_payload() {
            return Err(Error::invalid_value(
                "modeled resource key",
                format!("{key:?}"),
                "must use a Vello or transitional namespace",
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
        entry.payload = ResourcePayload::Modeled;
        self.retained_bytes = retained_bytes;
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        Ok(ResourceLease::new(
            manager_identity.clone(),
            frame,
            token.resource_identity,
            allocation_generation,
            ResourcePayload::Modeled,
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

    fn record_vello_atlas_recovery(&mut self, outcome: VelloAtlasOutcome) {
        self.pending_vello_atlas_recovery =
            VelloAtlasOutcome::merge_pending_recovery(self.pending_vello_atlas_recovery, outcome);
    }

    fn consume_vello_atlas_recovery(&mut self) {
        let _ = self.pending_vello_atlas_recovery.take();
    }

    fn retire_idle_vello_atlases(&mut self) {
        let retired = self
            .entries
            .iter()
            .filter_map(|(identity, entry)| {
                (entry.key.is_vello_atlas()
                    && matches!(entry.state, ResourceEntryState::Idle { .. }))
                .then_some((*identity, entry.byte_len))
            })
            .collect::<Vec<_>>();
        for (identity, byte_len) in retired {
            let removed = self
                .entries
                .remove(&identity)
                .expect("an idle Vello atlas selected for replacement must remain registered");
            debug_assert_eq!(removed.byte_len, byte_len);
            self.retained_bytes = self
                .retained_bytes
                .checked_sub(byte_len)
                .expect("retained accounting must include every replaced Vello atlas");
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
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
                pending_vello_atlas_recovery: None,
                stats: ResourceLifecycleStats::default(),
                #[cfg(test)]
                payload_creation_attempts: 0,
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

    pub(crate) fn preflight_graph_acquisitions(
        &self,
        requests: &[ResourceAllocationPreflight],
    ) -> Result<()> {
        lock_state(&self.state).preflight_graph_acquisitions(requests)
    }

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
        ResourceManagerObservationForTest {
            idle_count,
            leased_count: state.entries.len().saturating_sub(idle_count),
            retained_bytes: state.retained_bytes,
            next_resource: state.next_resource,
            entry_count: state.entries.len(),
            payload_creation_attempts: state.payload_creation_attempts,
            retained_atlas_count,
            retained_atlas_byte_len,
            committed_transient_buffer_count,
            committed_transient_buffer_byte_len,
            committed_transient_image_count,
            committed_transient_image_byte_len,
            recovery_outcome: state.pending_vello_atlas_recovery,
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
    payload: ResourcePayload,
}

impl ResourceLease {
    fn new(
        manager_identity: ManagerIdentity,
        frame_identity: FrameIdentity,
        resource_identity: ResourceIdentity,
        allocation_generation: AllocationGeneration,
        payload: ResourcePayload,
    ) -> Self {
        Self {
            token: ResourceLeaseToken {
                manager_identity,
                frame_identity,
                resource_identity,
                allocation_generation,
            },
            payload,
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
            .field("payload", &self.payload.label())
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
    #[cfg(test)]
    pub(crate) fn acquire(
        &mut self,
        key: ResourceCacheKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        lock_state(&self.state).acquire(&self.manager_identity, self.frame, key, byte_len)
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

    pub(crate) fn transitional_texture<'scope>(
        &'scope self,
        lease: &'scope ResourceLease,
    ) -> Result<(&'scope wgpu::Texture, &'scope wgpu::TextureView)> {
        match self.validated_payload(lease)? {
            ResourcePayload::TransitionalTexture { texture, view } => Ok((texture, view)),
            _ => Err(Error::invalid_value(
                "resource lease payload",
                lease.resource_identity().get(),
                "must contain a transitional offscreen texture",
            )),
        }
    }

    pub(crate) fn acquire_vello_buffer(
        &mut self,
        device: &wgpu::Device,
        key: VelloBufferKey,
    ) -> Result<ResourceLease> {
        let byte_len = key.byte_len();
        lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            ResourceCacheKey::VelloBuffer(key),
            byte_len,
            IdleReuse::Fresh,
            || {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Surgeist internal Vello buffer"),
                    size: byte_len,
                    usage: key.usage(),
                    mapped_at_creation: false,
                });
                Ok(ResourcePayload::VelloBuffer { buffer })
            },
        )
    }

    pub(crate) fn acquire_vello_image(
        &mut self,
        device: &wgpu::Device,
        key: VelloImageKey,
        byte_len: u64,
    ) -> Result<ResourceLease> {
        lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            ResourceCacheKey::VelloImage(key),
            byte_len,
            IdleReuse::Fresh,
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
        )
    }

    pub(crate) fn acquire_transitional_texture(
        &mut self,
        device: &wgpu::Device,
        role: TransitionalTextureRole,
        descriptor: super::texture::TextureDescriptor,
    ) -> Result<ResourceLease> {
        let key = ResourceCacheKey::transitional_texture_for_role(
            role,
            TextureCacheKey::from_descriptor(descriptor),
        );
        lock_state(&self.state).acquire_with_payload(
            &self.manager_identity,
            self.frame,
            key,
            descriptor.byte_len(),
            IdleReuse::Allowed,
            || {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(role.label()),
                    size: wgpu::Extent3d {
                        width: descriptor.physical_size().width(),
                        height: descriptor.physical_size().height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::from(descriptor.format()),
                    usage: descriptor.wgpu_usage(),
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Ok(ResourcePayload::TransitionalTexture { texture, view })
            },
        )
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
        lock_state(&self.state).acquire_with_payload(
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
        )
    }

    #[cfg(test)]
    pub(crate) fn acquire_working_effect_texture_for_test(
        &mut self,
        device: &wgpu::Device,
        capabilities: &DeviceCapabilities,
        working_format: WorkingFormat,
        physical_size: super::PhysicalSize,
        usage: wgpu::TextureUsages,
    ) -> Result<ResourceLease> {
        let descriptor =
            EffectTextureDescriptor::try_working(working_format, physical_size, usage)?;
        self.acquire_effect_texture(device, capabilities, descriptor)
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
        lock_state(&self.state).acquire_with_payload(
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
        )
    }

    pub(crate) fn acquire_gaussian_kernel_buffer(
        &mut self,
        device: &wgpu::Device,
        plan: &GaussianKernelPlan,
    ) -> Result<ResourceLease> {
        plan.validate_upload_byte_len(plan.upload_bytes().len())?;
        let byte_len = plan.byte_len();
        if byte_len == 0 || byte_len > device.limits().max_buffer_size {
            return Err(Error::invalid_value(
                "Gaussian kernel buffer byte length",
                byte_len,
                "must be positive and no greater than the selected device buffer limit",
            ));
        }
        let key = ResourceCacheKey::GaussianKernelBuffer(plan.key());
        lock_state(&self.state).acquire_with_payload(
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
        )
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

    pub(crate) fn retire_idle_vello_atlases(&mut self) {
        lock_state(&self.state).retire_idle_vello_atlases();
    }

    #[cfg(test)]
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
    #[cfg(test)]
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
    pub(crate) next_resource: u64,
    pub(crate) entry_count: usize,
    pub(crate) payload_creation_attempts: u64,
    retained_atlas_count: usize,
    retained_atlas_byte_len: u64,
    committed_transient_buffer_count: usize,
    committed_transient_buffer_byte_len: u64,
    committed_transient_image_count: usize,
    committed_transient_image_byte_len: u64,
    recovery_outcome: Option<VelloAtlasOutcome>,
}

#[cfg(test)]
impl ResourceManagerObservationForTest {
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

    pub(crate) const fn recovery_outcome_for_test(&self) -> Option<VelloAtlasOutcome> {
        self.recovery_outcome
    }
}
