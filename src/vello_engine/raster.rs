// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::mem::size_of;

use peniko::Color;
use vello_encoding::{Encoding, RenderConfig, Resolver, make_mask_lut, make_mask_lut_16};

use crate::{Antialiasing, Error, PhysicalSize, Result};

use super::recording::{
    BufferHandle, BufferRole, FineRasterVariant, ImageHandle, ImageRole, RasterImageFormat,
    RasterKernel, RasterPhase, Recording, RecordingBuilder, ResourceBinding, ResourceIntent,
};

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 target validation is staged until production Vello lowering reaches this module."
    )
)]
const VELLO_TILE_EXTENT: u32 = 16;

/// Validated algorithm inputs for one private Vello recording pass.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 raster parameters are staged until T7 owns production Vello lowering."
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RasterParameters {
    target_extent: PhysicalSize,
    base_color: Color,
    antialiasing: Antialiasing,
}

impl RasterParameters {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T3 validation is staged until production Vello lowering reaches this module."
        )
    )]
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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 target intent is consumed by the later transaction-owned realization stage."
    )
)]
pub(crate) struct RasterTargetIntent {
    extent: PhysicalSize,
    format: RasterImageFormat,
    access: RasterTargetAccess,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RasterTargetAccess {
    StorageWrite,
}

/// A private prepared pass that has no runtime resource or submission authority.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 prepared passes are intentionally staged until T4 adds checked realization."
    )
)]
pub(crate) struct PreparedVelloPass {
    recording: Recording,
    target_intent: RasterTargetIntent,
    resource_intents: Vec<ResourceIntent>,
}

/// The private Vello algorithm owner for resolving and recording one pass.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 recorder remains private until T7 production Vello cutover."
    )
)]
#[derive(Default)]
pub(crate) struct RasterRecorder {
    resolver: Resolver,
}

impl RasterRecorder {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T3 preparation is retained for T4 checked encoding and T7 cutover."
        )
    )]
    pub(crate) fn prepare(
        &mut self,
        encoding: &Encoding,
        parameters: RasterParameters,
    ) -> Result<PreparedVelloPass> {
        let mut packed = Vec::new();
        let (layout, ramps, images) = self.resolver.resolve(encoding, &mut packed);
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
            },
            resource_intents,
        })
    }
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
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::PathTagReduce,
        workgroups.path_reduce,
        [config.into(), scene.into(), path_reduced.into()],
    );

    let mut path_tag_parent = path_reduced;
    let mut large_path_scan = None;
    if workgroups.use_large_path_scan {
        let path_reduced2 = buffer(
            recording,
            BufferRole::PathReduced2,
            sizes.path_reduced2.size_in_bytes(),
        )?;
        recording.dispatch(
            RasterPhase::Coarse,
            RasterKernel::PathTagReduce2,
            workgroups.path_reduce2,
            [path_reduced.into(), path_reduced2.into()],
        );
        let path_reduced_scan = buffer(
            recording,
            BufferRole::PathReducedScan,
            sizes.path_reduced_scan.size_in_bytes(),
        )?;
        recording.dispatch(
            RasterPhase::Coarse,
            RasterKernel::PathTagScan1,
            workgroups.path_scan1,
            [
                path_reduced.into(),
                path_reduced2.into(),
                path_reduced_scan.into(),
            ],
        );
        path_tag_parent = path_reduced_scan;
        large_path_scan = Some((path_reduced2, path_reduced_scan));
    }

    let path_monoids = buffer(
        recording,
        BufferRole::PathMonoids,
        sizes.path_monoids.size_in_bytes(),
    )?;
    recording.dispatch(
        RasterPhase::Coarse,
        if workgroups.use_large_path_scan {
            RasterKernel::PathTagScanLarge
        } else {
            RasterKernel::PathTagScan
        },
        workgroups.path_scan,
        [
            config.into(),
            scene.into(),
            path_tag_parent.into(),
            path_monoids.into(),
        ],
    );
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
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::BboxClear,
        workgroups.bbox_clear,
        [config.into(), path_bboxes.into()],
    );
    let bump = buffer(
        recording,
        BufferRole::Bump,
        sizes.bump_alloc.size_in_bytes(),
    )?;
    recording.clear_buffer(bump);
    let lines = buffer(recording, BufferRole::Lines, sizes.lines.size_in_bytes())?;
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::Flatten,
        workgroups.flatten,
        [
            config.into(),
            scene.into(),
            path_monoids.into(),
            path_bboxes.into(),
            bump.into(),
            lines.into(),
        ],
    );
    let draw_reduced = buffer(
        recording,
        BufferRole::DrawReduced,
        sizes.draw_reduced.size_in_bytes(),
    )?;
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::DrawReduce,
        workgroups.draw_reduce,
        [config.into(), scene.into(), draw_reduced.into()],
    );
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
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::DrawLeaf,
        workgroups.draw_leaf,
        [
            config.into(),
            scene.into(),
            draw_reduced.into(),
            path_bboxes.into(),
            draw_monoids.into(),
            info_bin_data.into(),
            clip_inputs.into(),
        ],
    );
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
        recording.dispatch(
            RasterPhase::Coarse,
            RasterKernel::ClipReduce,
            workgroups.clip_reduce,
            [
                clip_inputs.into(),
                path_bboxes.into(),
                clip_bics.into(),
                clip_elements.into(),
            ],
        );
    }
    let clip_bboxes = buffer(
        recording,
        BufferRole::ClipBboxes,
        sizes.clip_bboxes.size_in_bytes(),
    )?;
    if workgroups.clip_leaf.0 > 0 {
        recording.dispatch(
            RasterPhase::Coarse,
            RasterKernel::ClipLeaf,
            workgroups.clip_leaf,
            [
                config.into(),
                clip_inputs.into(),
                path_bboxes.into(),
                clip_bics.into(),
                clip_elements.into(),
                draw_monoids.into(),
                clip_bboxes.into(),
            ],
        );
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
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::Binning,
        workgroups.binning,
        [
            config.into(),
            draw_monoids.into(),
            path_bboxes.into(),
            clip_bboxes.into(),
            draw_bboxes.into(),
            bump.into(),
            info_bin_data.into(),
            bin_headers.into(),
        ],
    );
    recording.release(path_bboxes);
    recording.release(clip_bboxes);

    let paths = buffer(recording, BufferRole::Paths, sizes.paths.size_in_bytes())?;
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::TileAlloc,
        workgroups.tile_alloc,
        [
            config.into(),
            scene.into(),
            draw_bboxes.into(),
            bump.into(),
            paths.into(),
            tile.into(),
        ],
    );
    recording.release(draw_bboxes);
    recording.release(path_monoids);

    let indirect_count = buffer(
        recording,
        BufferRole::IndirectCount,
        sizes.indirect_count.size_in_bytes(),
    )?;
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::PathCountSetup,
        workgroups.path_count_setup,
        [bump.into(), indirect_count.into()],
    );
    let segment_counts = buffer(
        recording,
        BufferRole::SegmentCounts,
        sizes.seg_counts.size_in_bytes(),
    )?;
    recording.dispatch_indirect(
        RasterPhase::Coarse,
        RasterKernel::PathCount,
        indirect_count,
        0,
        [
            config.into(),
            bump.into(),
            lines.into(),
            paths.into(),
            tile.into(),
            segment_counts.into(),
        ],
    );
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::Backdrop,
        workgroups.backdrop,
        [config.into(), bump.into(), paths.into(), tile.into()],
    );
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::Coarse,
        workgroups.coarse,
        [
            config.into(),
            scene.into(),
            draw_monoids.into(),
            bin_headers.into(),
            info_bin_data.into(),
            paths.into(),
            tile.into(),
            bump.into(),
            per_tile_command_list.into(),
        ],
    );
    recording.release(draw_monoids);
    recording.release(bin_headers);
    recording.release(scene);
    recording.dispatch(
        RasterPhase::Coarse,
        RasterKernel::PathTilingSetup,
        workgroups.path_tiling_setup,
        [
            bump.into(),
            indirect_count.into(),
            per_tile_command_list.into(),
        ],
    );
    recording.dispatch_indirect(
        RasterPhase::Coarse,
        RasterKernel::PathTiling,
        indirect_count,
        0,
        [
            bump.into(),
            segment_counts.into(),
            lines.into(),
            paths.into(),
            tile.into(),
            segments.into(),
        ],
    );
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
    let mut bindings = vec![
        ResourceBinding::from(fine.config),
        ResourceBinding::from(fine.segments),
        ResourceBinding::from(fine.per_tile_command_list),
        ResourceBinding::from(fine.info_bin_data),
        ResourceBinding::from(fine.blend_spill),
        ResourceBinding::TargetOutput,
        ResourceBinding::from(fine.gradient_image),
        ResourceBinding::from(fine.image_atlas),
    ];
    let kernel = match fine.variant {
        FineRasterVariant::Area => RasterKernel::FineArea,
        FineRasterVariant::Msaa8 | FineRasterVariant::Msaa16 => {
            let mask_lut = recording.upload_mask_lut(
                fine.variant,
                match fine.variant {
                    FineRasterVariant::Msaa8 => make_mask_lut(),
                    FineRasterVariant::Msaa16 => make_mask_lut_16(),
                    FineRasterVariant::Area => Vec::new(),
                },
            )?;
            bindings.push(mask_lut.into());
            let kernel = match fine.variant {
                FineRasterVariant::Msaa8 => RasterKernel::FineMsaa8,
                FineRasterVariant::Msaa16 => RasterKernel::FineMsaa16,
                FineRasterVariant::Area => RasterKernel::FineArea,
            };
            recording.dispatch(RasterPhase::Fine, kernel, fine.workgroups, bindings);
            recording.release(mask_lut);
            release_fine_resources(recording, fine);
            return Ok(());
        }
    };
    recording.dispatch(RasterPhase::Fine, kernel, fine.workgroups, bindings);
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

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 target validation is staged until production Vello lowering reaches this module."
    )
)]
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
    pub(crate) const fn recording_for_test(&self) -> &Recording {
        &self.recording
    }

    pub(crate) const fn target_intent_for_test(&self) -> &RasterTargetIntent {
        &self.target_intent
    }

    pub(crate) fn resource_intents_for_test(&self) -> &[ResourceIntent] {
        &self.resource_intents
    }
}

#[cfg(test)]
impl RasterTargetIntent {
    pub(crate) const fn extent_for_test(&self) -> PhysicalSize {
        self.extent
    }

    pub(crate) const fn is_rgba8_storage_for_test(&self) -> bool {
        matches!(self.format, RasterImageFormat::Rgba8Unorm)
            && matches!(self.access, RasterTargetAccess::StorageWrite)
    }
}
