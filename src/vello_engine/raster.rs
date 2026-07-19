// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::mem::size_of;

use peniko::Color;
use vello_encoding::{Encoding, RenderConfig, Resolver, make_mask_lut, make_mask_lut_16};

use crate::{Antialiasing, Error, PhysicalSize, Result};

#[cfg(test)]
use super::PreparedVelloPassObservation;
use super::recording::{
    BinningBindings, BufferHandle, BufferRole, ClipLeafBindings, CoarseDispatch,
    CoarseRasterBindings, DrawLeafBindings, FineDispatch, FineDispatchBindings, FineRasterVariant,
    ImageHandle, ImageRole, RasterImageFormat, Recording, RecordingBuilder, ResourceIntent,
};

const VELLO_TILE_EXTENT: u32 = 16;

/// Validated algorithm inputs for one private Vello recording pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RasterParameters {
    target_extent: PhysicalSize,
    base_color: Color,
    antialiasing: Antialiasing,
}

impl RasterParameters {
    pub(crate) fn try_new(
        target_extent: PhysicalSize,
        base_color: Color,
        antialiasing: Antialiasing,
    ) -> Result<Self> {
        validate_target_dimension("raster target width", target_extent.width())?;
        validate_target_dimension("raster target height", target_extent.height())?;
        Ok(Self {
            target_extent,
            base_color,
            antialiasing,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T3 keeps antialiasing selection symbolic until production scene lowering reaches the engine."
        )
    )]
    pub(crate) const fn with_antialiasing(mut self, antialiasing: Antialiasing) -> Self {
        self.antialiasing = antialiasing;
        self
    }
}

/// The required external output contract for one prepared Vello pass.
struct RasterTargetIntent {
    extent: PhysicalSize,
    format: RasterImageFormat,
    access: RasterTargetAccess,
    base_color: Color,
    antialiasing: Antialiasing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RasterTargetAccess {
    StorageWrite,
}

/// An opaque prepared pass that has no runtime resource or submission authority.
pub(crate) struct PreparedVelloPass {
    recording: Recording,
    target_intent: RasterTargetIntent,
    resource_intents: Vec<ResourceIntent>,
}

/// Unforgeable proof that one prepared pass carries one logical direct-Vello pass.
#[must_use]
pub(crate) struct DirectVelloLogicalPass {
    _prepared_pass: (),
}

impl DirectVelloLogicalPass {
    #[cfg(test)]
    pub(crate) const fn cardinality_for_test(&self) -> usize {
        1
    }
}

#[must_use]
pub(crate) struct EncodedVelloPass {
    resources: super::recording::VelloResourceLease,
    logical_pass: DirectVelloLogicalPass,
}

/// Proof produced only after one prepared capture has encoded successfully.
#[must_use]
pub(crate) struct EncodedVelloCaptureProof {
    target_extent: PhysicalSize,
    target_format: wgpu::TextureFormat,
    target_usage: wgpu::TextureUsages,
    antialiasing: Antialiasing,
    transparent_base: bool,
    #[cfg(test)]
    target_view_identity: usize,
}

#[must_use]
pub(crate) struct EncodedVelloCapture {
    resources: super::recording::VelloResourceLease,
    proof: EncodedVelloCaptureProof,
}

impl EncodedVelloPass {
    pub(crate) fn into_resources_and_logical_pass(
        self,
    ) -> (super::recording::VelloResourceLease, DirectVelloLogicalPass) {
        (self.resources, self.logical_pass)
    }
}

impl EncodedVelloCapture {
    pub(crate) fn into_resources_and_proof(
        self,
    ) -> (
        super::recording::VelloResourceLease,
        EncodedVelloCaptureProof,
    ) {
        (self.resources, self.proof)
    }
}

impl EncodedVelloCaptureProof {
    pub(crate) fn proves_capture_contract(
        &self,
        target_extent: PhysicalSize,
        target_format: wgpu::TextureFormat,
        target_usage: wgpu::TextureUsages,
        antialiasing: Antialiasing,
    ) -> bool {
        self.target_extent == target_extent
            && self.target_format == target_format
            && self.target_usage == target_usage
            && self.antialiasing == antialiasing
            && self.transparent_base
    }

    #[cfg(test)]
    pub(crate) const fn transparent_base_for_test(&self) -> bool {
        self.transparent_base
    }

    #[cfg(test)]
    pub(crate) const fn antialiasing_for_test(&self) -> Antialiasing {
        self.antialiasing
    }

    #[cfg(test)]
    pub(crate) const fn target_extent_for_test(&self) -> PhysicalSize {
        self.target_extent
    }

    #[cfg(test)]
    pub(crate) const fn target_format_for_test(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    #[cfg(test)]
    pub(crate) const fn target_usage_for_test(&self) -> wgpu::TextureUsages {
        self.target_usage
    }

    #[cfg(test)]
    pub(crate) const fn target_view_identity_for_test(&self) -> usize {
        self.target_view_identity
    }
}

pub(super) fn prepare(
    encoding: &Encoding,
    parameters: RasterParameters,
) -> Result<PreparedVelloPass> {
    let mut resolver = Resolver::default();
    let mut packed = Vec::new();
    let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
    let config = RenderConfig::new(
        &layout,
        parameters.target_extent.width(),
        parameters.target_extent.height(),
        &parameters.base_color,
    );
    let mut recording = RecordingBuilder::default();

    let gradient_image = if ramps.height == 0 {
        recording.new_transient_image(ImageRole::GradientRamp, PhysicalSize::new(1, 1))?
    } else {
        recording.upload_gradient_ramps(
            PhysicalSize::new(ramps.width, ramps.height),
            ramps.data.to_vec(),
        )?
    };
    let image_atlas = recording
        .request_image_atlas(PhysicalSize::new(images.width.max(1), images.height.max(1)))?;
    for (image, x, y) in images.images {
        recording.write_image(image_atlas, [*x, *y], image.clone());
    }

    if packed.is_empty() {
        packed.resize(size_of::<u32>(), u8::MAX);
    }
    let scene = recording.upload_scene(packed)?;
    let config_buffer = recording.upload_config(config.gpu)?;
    let fine = record_coarse(
        &mut recording,
        CoarseInputs {
            workgroups: config.workgroup_counts,
            sizes: config.buffer_sizes,
            scene,
            config: config_buffer,
            gradient_image,
            image_atlas,
            antialiasing: parameters.antialiasing,
        },
    )?;
    record_fine(&mut recording, fine)?;

    let (recording, resource_intents) = recording.finish();
    Ok(PreparedVelloPass {
        recording,
        target_intent: RasterTargetIntent {
            extent: parameters.target_extent,
            format: RasterImageFormat::Rgba8Unorm,
            access: RasterTargetAccess::StorageWrite,
            base_color: parameters.base_color,
            antialiasing: parameters.antialiasing,
        },
        resource_intents,
    })
}

struct FineSchedule {
    variant: FineRasterVariant,
    workgroups: vello_encoding::WorkgroupSize,
    config: BufferHandle,
    tile: BufferHandle,
    segments: BufferHandle,
    per_tile_command_list: BufferHandle,
    gradient_image: ImageHandle,
    info_bin_data: BufferHandle,
    image_atlas: ImageHandle,
    blend_spill: BufferHandle,
}

struct CoarseInputs {
    workgroups: vello_encoding::WorkgroupCounts,
    sizes: vello_encoding::BufferSizes,
    scene: BufferHandle,
    config: BufferHandle,
    gradient_image: ImageHandle,
    image_atlas: ImageHandle,
    antialiasing: Antialiasing,
}

fn record_coarse(recording: &mut RecordingBuilder, inputs: CoarseInputs) -> Result<FineSchedule> {
    let CoarseInputs {
        workgroups,
        sizes,
        scene,
        config,
        gradient_image,
        image_atlas,
        antialiasing,
    } = inputs;
    let info_bin_data = buffer(
        recording,
        BufferRole::InfoBinData,
        sizes.bin_data.size_in_bytes(),
    )?;
    let tile = buffer(recording, BufferRole::Tile, sizes.tiles.size_in_bytes())?;
    let segments = buffer(
        recording,
        BufferRole::Segments,
        sizes.segments.size_in_bytes(),
    )?;
    let per_tile_command_list = buffer(
        recording,
        BufferRole::PerTileCommandList,
        sizes.ptcl.size_in_bytes(),
    )?;
    let path_reduced = buffer(
        recording,
        BufferRole::PathReduced,
        sizes.path_reduced.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::path_tag_reduce(
        workgroups.path_reduce,
        config,
        scene,
        path_reduced,
    ));

    let mut path_tag_parent = path_reduced;
    let mut large_path_scan = None;
    if workgroups.use_large_path_scan {
        let path_reduced2 = buffer(
            recording,
            BufferRole::PathReduced2,
            sizes.path_reduced2.size_in_bytes(),
        )?;
        recording.record_coarse(CoarseDispatch::path_tag_reduce2(
            workgroups.path_reduce2,
            path_reduced,
            path_reduced2,
        ));
        let path_reduced_scan = buffer(
            recording,
            BufferRole::PathReducedScan,
            sizes.path_reduced_scan.size_in_bytes(),
        )?;
        recording.record_coarse(CoarseDispatch::path_tag_scan1(
            workgroups.path_scan1,
            path_reduced,
            path_reduced2,
            path_reduced_scan,
        ));
        path_tag_parent = path_reduced_scan;
        large_path_scan = Some((path_reduced2, path_reduced_scan));
    }

    let path_monoids = buffer(
        recording,
        BufferRole::PathMonoids,
        sizes.path_monoids.size_in_bytes(),
    )?;
    recording.record_coarse(if workgroups.use_large_path_scan {
        CoarseDispatch::path_tag_scan_large(
            workgroups.path_scan,
            config,
            scene,
            path_tag_parent,
            path_monoids,
        )
    } else {
        CoarseDispatch::path_tag_scan(
            workgroups.path_scan,
            config,
            scene,
            path_tag_parent,
            path_monoids,
        )
    });
    recording.release(path_reduced);
    if let Some((path_reduced2, path_reduced_scan)) = large_path_scan {
        recording.release(path_reduced2);
        recording.release(path_reduced_scan);
    }

    let path_bboxes = buffer(
        recording,
        BufferRole::PathBboxes,
        sizes.path_bboxes.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::bbox_clear(
        workgroups.bbox_clear,
        config,
        path_bboxes,
    ));
    let bump = buffer(
        recording,
        BufferRole::Bump,
        sizes.bump_alloc.size_in_bytes(),
    )?;
    recording.clear_buffer(bump);
    let lines = buffer(recording, BufferRole::Lines, sizes.lines.size_in_bytes())?;
    recording.record_coarse(CoarseDispatch::flatten(
        workgroups.flatten,
        config,
        scene,
        path_monoids,
        path_bboxes,
        bump,
        lines,
    ));
    let draw_reduced = buffer(
        recording,
        BufferRole::DrawReduced,
        sizes.draw_reduced.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::draw_reduce(
        workgroups.draw_reduce,
        config,
        scene,
        draw_reduced,
    ));
    let draw_monoids = buffer(
        recording,
        BufferRole::DrawMonoids,
        sizes.draw_monoids.size_in_bytes(),
    )?;
    let clip_inputs = buffer(
        recording,
        BufferRole::ClipInputs,
        sizes.clip_inps.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::draw_leaf(
        workgroups.draw_leaf,
        DrawLeafBindings {
            config,
            scene,
            draw_reduced,
            path_bboxes,
            draw_monoids,
            info_bin_data,
            clip_inputs,
        },
    ));
    recording.release(draw_reduced);

    let clip_elements = buffer(
        recording,
        BufferRole::ClipElements,
        sizes.clip_els.size_in_bytes(),
    )?;
    let clip_bics = buffer(
        recording,
        BufferRole::ClipBics,
        sizes.clip_bics.size_in_bytes(),
    )?;
    if workgroups.clip_reduce.0 > 0 {
        recording.record_coarse(CoarseDispatch::clip_reduce(
            workgroups.clip_reduce,
            clip_inputs,
            path_bboxes,
            clip_bics,
            clip_elements,
        ));
    }
    let clip_bboxes = buffer(
        recording,
        BufferRole::ClipBboxes,
        sizes.clip_bboxes.size_in_bytes(),
    )?;
    if workgroups.clip_leaf.0 > 0 {
        recording.record_coarse(CoarseDispatch::clip_leaf(
            workgroups.clip_leaf,
            ClipLeafBindings {
                config,
                clip_inputs,
                path_bboxes,
                clip_bics,
                clip_elements,
                draw_monoids,
                clip_bboxes,
            },
        ));
    }
    recording.release(clip_inputs);
    recording.release(clip_bics);
    recording.release(clip_elements);

    let draw_bboxes = buffer(
        recording,
        BufferRole::DrawBboxes,
        sizes.draw_bboxes.size_in_bytes(),
    )?;
    let bin_headers = buffer(
        recording,
        BufferRole::BinHeaders,
        sizes.bin_headers.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::binning(
        workgroups.binning,
        BinningBindings {
            config,
            draw_monoids,
            path_bboxes,
            clip_bboxes,
            draw_bboxes,
            bump,
            info_bin_data,
            bin_headers,
        },
    ));
    recording.release(path_bboxes);
    recording.release(clip_bboxes);

    let paths = buffer(recording, BufferRole::Paths, sizes.paths.size_in_bytes())?;
    recording.record_coarse(CoarseDispatch::tile_alloc(
        workgroups.tile_alloc,
        config,
        scene,
        draw_bboxes,
        bump,
        paths,
        tile,
    ));
    recording.release(draw_bboxes);
    recording.release(path_monoids);

    let indirect_count = buffer(
        recording,
        BufferRole::IndirectCount,
        sizes.indirect_count.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::path_count_setup(
        workgroups.path_count_setup,
        bump,
        indirect_count,
    ));
    let segment_counts = buffer(
        recording,
        BufferRole::SegmentCounts,
        sizes.seg_counts.size_in_bytes(),
    )?;
    recording.record_coarse(CoarseDispatch::path_count(
        indirect_count,
        config,
        bump,
        lines,
        paths,
        tile,
        segment_counts,
    ));
    recording.record_coarse(CoarseDispatch::backdrop(
        workgroups.backdrop,
        config,
        bump,
        paths,
        tile,
    ));
    recording.record_coarse(CoarseDispatch::coarse(
        workgroups.coarse,
        CoarseRasterBindings {
            config,
            scene,
            draw_monoids,
            bin_headers,
            info_bin_data,
            paths,
            tile,
            bump,
            per_tile_command_list,
        },
    ));
    recording.release(draw_monoids);
    recording.release(bin_headers);
    recording.release(scene);
    recording.record_coarse(CoarseDispatch::path_tiling_setup(
        workgroups.path_tiling_setup,
        bump,
        indirect_count,
        per_tile_command_list,
    ));
    recording.record_coarse(CoarseDispatch::path_tiling(
        indirect_count,
        bump,
        segment_counts,
        lines,
        paths,
        tile,
        segments,
    ));
    recording.release(indirect_count);
    recording.release(segment_counts);
    recording.release(lines);
    recording.release(paths);
    recording.release(bump);

    Ok(FineSchedule {
        variant: fine_variant(antialiasing),
        workgroups: workgroups.fine,
        config,
        tile,
        segments,
        per_tile_command_list,
        gradient_image,
        info_bin_data,
        image_atlas,
        blend_spill: buffer(
            recording,
            BufferRole::BlendSpill,
            sizes.blend_spill.size_in_bytes(),
        )?,
    })
}

fn record_fine(recording: &mut RecordingBuilder, fine: FineSchedule) -> Result<()> {
    let bindings = FineDispatchBindings {
        workgroups: fine.workgroups,
        config: fine.config,
        segments: fine.segments,
        per_tile_command_list: fine.per_tile_command_list,
        info_bin_data: fine.info_bin_data,
        blend_spill: fine.blend_spill,
        gradient_image: fine.gradient_image,
        image_atlas: fine.image_atlas,
    };
    match fine.variant {
        FineRasterVariant::Area => recording.record_fine(FineDispatch::area(bindings)),
        FineRasterVariant::Msaa8 => {
            let mask_lut = recording.upload_mask_lut(FineRasterVariant::Msaa8, make_mask_lut())?;
            recording.record_fine(FineDispatch::msaa8(bindings, mask_lut));
            recording.release(mask_lut);
        }
        FineRasterVariant::Msaa16 => {
            let mask_lut =
                recording.upload_mask_lut(FineRasterVariant::Msaa16, make_mask_lut_16())?;
            recording.record_fine(FineDispatch::msaa16(bindings, mask_lut));
            recording.release(mask_lut);
        }
    }
    release_fine_resources(recording, fine);
    Ok(())
}

fn release_fine_resources(recording: &mut RecordingBuilder, fine: FineSchedule) {
    recording.release(fine.config);
    recording.release(fine.tile);
    recording.release(fine.segments);
    recording.release(fine.per_tile_command_list);
    recording.release(fine.gradient_image);
    recording.release(fine.info_bin_data);
    recording.release(fine.blend_spill);
}

fn buffer(
    recording: &mut RecordingBuilder,
    role: BufferRole,
    byte_len: u32,
) -> Result<BufferHandle> {
    recording.new_buffer(role, u64::from(byte_len))
}

const fn fine_variant(antialiasing: Antialiasing) -> FineRasterVariant {
    match antialiasing {
        Antialiasing::Area => FineRasterVariant::Area,
        Antialiasing::Msaa8 => FineRasterVariant::Msaa8,
        Antialiasing::Msaa16 => FineRasterVariant::Msaa16,
    }
}

fn validate_target_dimension(field: &'static str, value: u32) -> Result<()> {
    if value == 0 || value > u32::MAX - (VELLO_TILE_EXTENT - 1) {
        return Err(Error::invalid_value(
            field,
            value,
            "must be nonzero and leave room for Vello tile padding",
        ));
    }
    Ok(())
}

#[cfg(test)]
impl PreparedVelloPass {
    pub(super) fn observation_for_test(&self) -> PreparedVelloPassObservation {
        let (dispatches, resource_lifetimes) = self
            .recording
            .schedule_observations_for_test(&self.resource_intents);
        PreparedVelloPassObservation {
            target_extent: self.target_intent.extent,
            is_rgba8_storage: self.target_intent.is_rgba8_storage_for_test(),
            final_dispatch_targets_output: self.recording.final_dispatch_targets_output_for_test(),
            is_self_consistent: self
                .recording
                .is_self_consistent_for_test(&self.resource_intents),
            has_persistent_image_atlas: self
                .resource_intents
                .iter()
                .any(ResourceIntent::is_persistent_image_atlas_for_test),
            has_transient_buffer: self
                .resource_intents
                .iter()
                .any(ResourceIntent::is_transient_buffer_for_test),
            dispatches,
            resource_lifetimes,
        }
    }
}

#[cfg(test)]
impl RasterTargetIntent {
    const fn is_rgba8_storage_for_test(&self) -> bool {
        matches!(self.format, RasterImageFormat::Rgba8Unorm)
            && matches!(self.access, RasterTargetAccess::StorageWrite)
    }
}
