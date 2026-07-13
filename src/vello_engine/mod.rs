pub(crate) mod glyph;
mod raster;
mod recording;
pub(crate) mod scene;

#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "C03 T3 exposes prepared-pass construction for T4 checked realization and T7 cutover."
    )
)]
pub(crate) use raster::{PreparedVelloPass, RasterParameters, RasterRecorder};

#[cfg(test)]
pub(crate) fn prepared_vello_pass_observation_for_test(
    pass: &PreparedVelloPass,
) -> PreparedVelloPassObservation {
    pass.observation_for_test()
}

#[cfg(test)]
pub(crate) struct PreparedVelloPassObservation {
    pub(in crate::vello_engine) target_extent: crate::PhysicalSize,
    pub(in crate::vello_engine) is_rgba8_storage: bool,
    pub(in crate::vello_engine) final_dispatch_targets_output: bool,
    pub(in crate::vello_engine) is_self_consistent: bool,
    pub(in crate::vello_engine) has_area_schedule: bool,
    pub(in crate::vello_engine) has_msaa8_schedule: bool,
    pub(in crate::vello_engine) has_msaa16_schedule: bool,
    pub(in crate::vello_engine) has_persistent_image_atlas: bool,
    pub(in crate::vello_engine) has_transient_buffer: bool,
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

    pub(crate) const fn has_area_schedule_for_test(&self) -> bool {
        self.has_area_schedule
    }

    pub(crate) const fn has_msaa8_schedule_for_test(&self) -> bool {
        self.has_msaa8_schedule
    }

    pub(crate) const fn has_msaa16_schedule_for_test(&self) -> bool {
        self.has_msaa16_schedule
    }

    pub(crate) const fn has_persistent_image_atlas_for_test(&self) -> bool {
        self.has_persistent_image_atlas
    }

    pub(crate) const fn has_transient_buffer_for_test(&self) -> bool {
        self.has_transient_buffer
    }
}
