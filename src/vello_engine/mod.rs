pub(crate) mod glyph;
mod shaders;
mod recording {
    include!("recording.rs");

    mod resources {
        include!("resources.rs");
    }

    mod encoder {
        include!("encoder.rs");

        impl VelloEngineState {
            pub(crate) async fn new_for_device_state(device: &wgpu::Device) -> Result<Self> {
                Ok(Self {
                    shaders: CheckedShaderSet::create(device).await?,
                })
            }

            #[cfg(test)]
            pub(crate) fn checked_pipeline_for_test(&self) -> &wgpu::ComputePipeline {
                self.shaders.pipeline(RasterKernel::FineArea).pipeline()
            }
        }
    }

    pub(crate) use encoder::{
        ActiveVelloEncodingScope, TransactionEncodingState, TransactionTargetIntent,
        VelloEncodingFailure, VelloEngineState, encode_recording,
    };
    #[cfg(test)]
    pub(crate) use resources::VelloAtlasOutcome;
    pub(super) use resources::VelloResourceLease;
    pub(crate) use resources::{PendingVelloResourceCommit, VelloResourceManager};
    #[cfg(test)]
    pub(super) use resources::{
        ScopeResolvedVelloResourceLease, commit_scope_resolved_for_test,
        no_atlas_abort_outcome_for_test, no_atlas_commit_outcome_for_test,
        over_limit_buffer_preflight_for_test,
    };
}
mod raster {
    include!("raster.rs");

    impl PreparedVelloPass {
        #[cfg_attr(
            not(test),
            expect(
                dead_code,
                reason = "C03 T4 keeps private checked encoding ready for the later T7 cutover."
            )
        )]
        pub(crate) fn encode_into(
            &self,
            engine: &super::recording::VelloEngineState,
            state: &mut super::recording::TransactionEncodingState<'_, '_>,
        ) -> std::result::Result<
            super::recording::VelloResourceLease,
            super::recording::VelloEncodingFailure,
        > {
            if self.target_intent.extent != state.target_extent()
                || self.target_intent.format != RasterImageFormat::Rgba8Unorm
                || self.target_intent.access != RasterTargetAccess::StorageWrite
                || state.target_format() != wgpu::TextureFormat::Rgba8Unorm
                || !state
                    .target_usage()
                    .contains(wgpu::TextureUsages::STORAGE_BINDING)
            {
                return Err(
                    super::recording::VelloEncodingFailure::before_resource_allocation(Error::new(
                        crate::BackendErrorCode::RenderFailed,
                        "internal Vello encoding target does not match the prepared raster intent",
                    )),
                );
            }

            super::recording::encode_recording(
                engine,
                &self.recording,
                &self.resource_intents,
                state,
            )
        }
    }
}
pub(crate) mod scene;

#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "C03 T3 exposes scene-owned prepared passes for T4 checked realization and T7 cutover."
    )
)]
pub(crate) use raster::{PreparedVelloPass, RasterParameters};

#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "C03 T4 keeps transaction-borrowed encoding state internal until the T7 cutover."
    )
)]
pub(crate) use recording::{
    ActiveVelloEncodingScope, PendingVelloResourceCommit, TransactionEncodingState,
    TransactionTargetIntent, VelloEngineState, VelloResourceManager,
};

#[cfg(test)]
pub(crate) use recording::VelloAtlasOutcome;

#[cfg(test)]
pub(crate) async fn checked_shader_validation_for_test(device: &wgpu::Device) -> crate::Result<()> {
    shaders::checked_shader_validation_for_test(device).await
}

#[cfg(test)]
pub(crate) fn checked_scope_out_of_memory_for_test() -> crate::Error {
    shaders::checked_scope_out_of_memory_for_test()
}

#[cfg(test)]
pub(crate) async fn over_limit_buffer_preflight_for_test(
    device: &wgpu::Device,
) -> crate::Result<()> {
    recording::over_limit_buffer_preflight_for_test(device).await
}

#[cfg(test)]
pub(crate) async fn no_atlas_commit_outcome_for_test(
    device: &wgpu::Device,
) -> crate::Result<VelloAtlasOutcome> {
    recording::no_atlas_commit_outcome_for_test(device).await
}

#[cfg(test)]
pub(crate) fn commit_scope_resolved_for_test(
    lease: recording::ScopeResolvedVelloResourceLease,
) -> VelloAtlasOutcome {
    recording::commit_scope_resolved_for_test(lease)
}

#[cfg(test)]
pub(crate) async fn no_atlas_abort_outcome_for_test(
    device: &wgpu::Device,
) -> crate::Result<VelloAtlasOutcome> {
    recording::no_atlas_abort_outcome_for_test(device).await
}

#[cfg(test)]
pub(crate) fn prepared_vello_pass_observation_for_test(
    pass: &PreparedVelloPass,
) -> PreparedVelloPassObservation {
    pass.observation_for_test()
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassPhaseForTest {
    Coarse,
    Fine,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassOperationForTest {
    PathTagReduce,
    PathTagReduce2,
    PathTagScan1,
    PathTagScan,
    PathTagScanLarge,
    BboxClear,
    Flatten,
    DrawReduce,
    DrawLeaf,
    ClipReduce,
    ClipLeaf,
    Binning,
    TileAlloc,
    PathCountSetup,
    PathCount,
    Backdrop,
    Coarse,
    PathTilingSetup,
    PathTiling,
    FineArea,
    FineMsaa8,
    FineMsaa16,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassBufferRoleForTest {
    Scene,
    Config,
    InfoBinData,
    Tile,
    Segments,
    PerTileCommandList,
    PathReduced,
    PathReduced2,
    PathReducedScan,
    PathMonoids,
    PathBboxes,
    Bump,
    Lines,
    DrawReduced,
    DrawMonoids,
    ClipInputs,
    ClipElements,
    ClipBics,
    ClipBboxes,
    DrawBboxes,
    BinHeaders,
    Paths,
    IndirectCount,
    SegmentCounts,
    BlendSpill,
    MaskLut,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassImageRoleForTest {
    GradientRamp,
    ImageAtlas,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassBindingForTest {
    Buffer(VelloPassBufferRoleForTest),
    Image(VelloPassImageRoleForTest),
    TargetOutput,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VelloPassIndirectDispatchForTest {
    count_buffer_role: VelloPassBufferRoleForTest,
    offset: u64,
}

#[cfg(test)]
impl VelloPassIndirectDispatchForTest {
    pub(in crate::vello_engine) const fn new(
        count_buffer_role: VelloPassBufferRoleForTest,
        offset: u64,
    ) -> Self {
        Self {
            count_buffer_role,
            offset,
        }
    }

    pub(crate) const fn count_buffer_role_for_test(&self) -> VelloPassBufferRoleForTest {
        self.count_buffer_role
    }

    pub(crate) const fn offset_for_test(&self) -> u64 {
        self.offset
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloPassResourceForTest {
    LargePathReduced2,
    LargePathReducedScan,
    ClipInputs,
    ClipElements,
    ClipBics,
    ClipBboxes,
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VelloPassDispatchObservation {
    pub(in crate::vello_engine) phase: VelloPassPhaseForTest,
    pub(in crate::vello_engine) operation: VelloPassOperationForTest,
    pub(in crate::vello_engine) bindings: Vec<VelloPassBindingForTest>,
    pub(in crate::vello_engine) indirect: Option<VelloPassIndirectDispatchForTest>,
}

#[cfg(test)]
impl VelloPassDispatchObservation {
    pub(crate) const fn phase_for_test(&self) -> VelloPassPhaseForTest {
        self.phase
    }

    pub(crate) const fn operation_for_test(&self) -> VelloPassOperationForTest {
        self.operation
    }

    pub(crate) fn bindings_for_test(&self) -> &[VelloPassBindingForTest] {
        &self.bindings
    }

    pub(crate) const fn indirect_for_test(&self) -> Option<VelloPassIndirectDispatchForTest> {
        self.indirect
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VelloPassResourceLifetimeObservation {
    pub(in crate::vello_engine) allocation_after: Option<VelloPassOperationForTest>,
    pub(in crate::vello_engine) first_use: Option<VelloPassOperationForTest>,
    pub(in crate::vello_engine) last_use: Option<VelloPassOperationForTest>,
    pub(in crate::vello_engine) release_after: Option<VelloPassOperationForTest>,
}

#[cfg(test)]
impl VelloPassResourceLifetimeObservation {
    pub(crate) const fn allocated_after_for_test(&self) -> Option<VelloPassOperationForTest> {
        self.allocation_after
    }

    pub(crate) const fn first_use_for_test(&self) -> Option<VelloPassOperationForTest> {
        self.first_use
    }

    pub(crate) const fn last_use_for_test(&self) -> Option<VelloPassOperationForTest> {
        self.last_use
    }

    pub(crate) const fn released_after_for_test(&self) -> Option<VelloPassOperationForTest> {
        self.release_after
    }
}

#[cfg(test)]
pub(crate) struct PreparedVelloPassObservation {
    pub(in crate::vello_engine) target_extent: crate::PhysicalSize,
    pub(in crate::vello_engine) is_rgba8_storage: bool,
    pub(in crate::vello_engine) final_dispatch_targets_output: bool,
    pub(in crate::vello_engine) is_self_consistent: bool,
    pub(in crate::vello_engine) has_persistent_image_atlas: bool,
    pub(in crate::vello_engine) has_transient_buffer: bool,
    pub(in crate::vello_engine) dispatches: Vec<VelloPassDispatchObservation>,
    pub(in crate::vello_engine) resource_lifetimes: Vec<(
        VelloPassResourceForTest,
        VelloPassResourceLifetimeObservation,
    )>,
}

#[cfg(test)]
impl PreparedVelloPassObservation {
    pub(crate) const fn target_extent_for_test(&self) -> crate::PhysicalSize {
        self.target_extent
    }

    pub(crate) const fn is_rgba8_storage_for_test(&self) -> bool {
        self.is_rgba8_storage
    }

    pub(crate) const fn final_dispatch_targets_output_for_test(&self) -> bool {
        self.final_dispatch_targets_output
    }

    pub(crate) const fn is_self_consistent_for_test(&self) -> bool {
        self.is_self_consistent
    }

    pub(crate) const fn has_persistent_image_atlas_for_test(&self) -> bool {
        self.has_persistent_image_atlas
    }

    pub(crate) const fn has_transient_buffer_for_test(&self) -> bool {
        self.has_transient_buffer
    }

    pub(crate) fn dispatches_for_test(&self) -> &[VelloPassDispatchObservation] {
        &self.dispatches
    }

    pub(crate) fn resource_lifetime_for_test(
        &self,
        resource: VelloPassResourceForTest,
    ) -> Option<VelloPassResourceLifetimeObservation> {
        self.resource_lifetimes
            .iter()
            .find_map(|(observed_resource, lifetime)| {
                (*observed_resource == resource).then_some(*lifetime)
            })
    }
}
