pub(crate) mod glyph;
mod raster;
mod recording;
pub(crate) mod scene;

#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "C03 T3 exposes scene-owned prepared passes for T4 checked realization and T7 cutover."
    )
)]
pub(crate) use raster::{PreparedVelloPass, RasterParameters};

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
pub(crate) enum VelloPassBindingForTest {
    Buffer,
    Image,
    TargetOutput,
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
    pub(in crate::vello_engine) indirect: bool,
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

    pub(crate) const fn is_indirect_for_test(&self) -> bool {
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
