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
    pub(crate) use resources::VelloResourceAllocationSummaryForTest;
    pub(super) use resources::VelloResourceLease;
    pub(crate) use resources::{
        AccountingReadyVelloResourceCommit, PendingVelloResourceCommit, VelloAtlasOutcome,
        VelloBufferKey, VelloImageKey, VelloResourceLeaseAggregate,
    };
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
        fn encode_resources_into(
            &self,
            engine: &super::recording::VelloEngineState,
            resources: &crate::resource::ResourceManager,
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
                resources,
                state,
            )
        }

        pub(crate) fn encode_into(
            &self,
            engine: &super::recording::VelloEngineState,
            resources: &crate::resource::ResourceManager,
            state: &mut super::recording::TransactionEncodingState<'_, '_>,
        ) -> std::result::Result<EncodedVelloPass, super::recording::VelloEncodingFailure> {
            self.encode_resources_into(engine, resources, state)
                .map(|resources| EncodedVelloPass {
                    resources,
                    logical_pass: DirectVelloLogicalPass { _prepared_pass: () },
                })
        }

        pub(crate) fn encode_capture_into(
            &self,
            engine: &super::recording::VelloEngineState,
            resources: &crate::resource::ResourceManager,
            state: &mut super::recording::TransactionEncodingState<'_, '_>,
        ) -> std::result::Result<EncodedVelloCapture, super::recording::VelloEncodingFailure>
        {
            let target_extent = state.target_extent();
            let target_format = state.target_format();
            let target_usage = state.target_usage();
            #[cfg(test)]
            let target_view_identity = state.target_view_identity_for_test();
            self.encode_resources_into(engine, resources, state)
                .map(|resources| EncodedVelloCapture {
                    resources,
                    proof: EncodedVelloCaptureProof {
                        target_extent,
                        target_format,
                        target_usage,
                        antialiasing: self.target_intent.antialiasing,
                        transparent_base: self.target_intent.base_color
                            == peniko::Color::TRANSPARENT,
                        #[cfg(test)]
                        target_view_identity,
                    },
                })
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
pub(crate) use raster::{
    DirectVelloLogicalPass, EncodedVelloCaptureProof, EncodedVelloPass, PreparedVelloPass,
    RasterParameters,
};

pub(crate) use recording::{
    AccountingReadyVelloResourceCommit, ActiveVelloEncodingScope, PendingVelloResourceCommit,
    TransactionEncodingState, TransactionTargetIntent, VelloAtlasOutcome, VelloBufferKey,
    VelloEngineState, VelloImageKey, VelloResourceLeaseAggregate,
};

#[cfg(test)]
pub(crate) use recording::VelloResourceAllocationSummaryForTest;

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
) -> crate::Result<VelloAtlasOutcome> {
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
pub(crate) struct PreparedVelloPassObservation {
    pub(in crate::vello_engine) target_extent: crate::PhysicalSize,
    pub(in crate::vello_engine) transparent_base: bool,
    pub(in crate::vello_engine) antialiasing: crate::Antialiasing,
    pub(in crate::vello_engine) is_rgba8_storage: bool,
    pub(in crate::vello_engine) final_dispatch_targets_output: bool,
    pub(in crate::vello_engine) is_self_consistent: bool,
    pub(in crate::vello_engine) has_persistent_image_atlas: bool,
    pub(in crate::vello_engine) has_transient_buffer: bool,
}

#[cfg(test)]
impl PreparedVelloPassObservation {
    pub(crate) const fn target_extent_for_test(&self) -> crate::PhysicalSize {
        self.target_extent
    }

    pub(crate) const fn transparent_base_for_test(&self) -> bool {
        self.transparent_base
    }

    pub(crate) const fn antialiasing_for_test(&self) -> crate::Antialiasing {
        self.antialiasing
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
}
