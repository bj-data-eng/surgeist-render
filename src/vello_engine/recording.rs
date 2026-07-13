// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(test)]
use std::mem::size_of;

use peniko::ImageData;
use vello_encoding::ConfigUniform;

use crate::{BackendErrorCode, Error, PhysicalSize, Result};

#[cfg(test)]
use super::{
    VelloPassBindingForTest, VelloPassBufferRoleForTest, VelloPassDispatchObservation,
    VelloPassImageRoleForTest, VelloPassIndirectDispatchForTest, VelloPassOperationForTest,
    VelloPassPhaseForTest, VelloPassResourceForTest, VelloPassResourceLifetimeObservation,
};

/// A symbolic resource identity within one prepared Vello pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct ResourceId(u64);

/// A symbolic buffer reference used by the compute-dispatch recording.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct BufferHandle(ResourceId);

/// A symbolic image reference used by the compute-dispatch recording.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct ImageHandle(ResourceId);

/// The only image format required by the pinned Vello raster path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RasterImageFormat {
    Rgba8Unorm,
}

/// The fixed algorithm phase associated with a recorded dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RasterPhase {
    Coarse,
    Fine,
}

/// The antialiasing-specific final raster program selected for one pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FineRasterVariant {
    Area,
    Msaa8,
    Msaa16,
}

/// The closed set of compute programs used by the pinned Vello schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RasterKernel {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResourceBinding {
    Buffer(BufferHandle),
    Image(ImageHandle),
    TargetOutput,
}

/// The bindings accepted by the coarse draw-leaf operation.
pub(super) struct DrawLeafBindings {
    pub(super) config: BufferHandle,
    pub(super) scene: BufferHandle,
    pub(super) draw_reduced: BufferHandle,
    pub(super) path_bboxes: BufferHandle,
    pub(super) draw_monoids: BufferHandle,
    pub(super) info_bin_data: BufferHandle,
    pub(super) clip_inputs: BufferHandle,
}

/// The bindings accepted by the coarse clip-leaf operation.
pub(super) struct ClipLeafBindings {
    pub(super) config: BufferHandle,
    pub(super) clip_inputs: BufferHandle,
    pub(super) path_bboxes: BufferHandle,
    pub(super) clip_bics: BufferHandle,
    pub(super) clip_elements: BufferHandle,
    pub(super) draw_monoids: BufferHandle,
    pub(super) clip_bboxes: BufferHandle,
}

/// The bindings accepted by the coarse binning operation.
pub(super) struct BinningBindings {
    pub(super) config: BufferHandle,
    pub(super) draw_monoids: BufferHandle,
    pub(super) path_bboxes: BufferHandle,
    pub(super) clip_bboxes: BufferHandle,
    pub(super) draw_bboxes: BufferHandle,
    pub(super) bump: BufferHandle,
    pub(super) info_bin_data: BufferHandle,
    pub(super) bin_headers: BufferHandle,
}

/// The bindings accepted by the final coarse-raster operation.
pub(super) struct CoarseRasterBindings {
    pub(super) config: BufferHandle,
    pub(super) scene: BufferHandle,
    pub(super) draw_monoids: BufferHandle,
    pub(super) bin_headers: BufferHandle,
    pub(super) info_bin_data: BufferHandle,
    pub(super) paths: BufferHandle,
    pub(super) tile: BufferHandle,
    pub(super) bump: BufferHandle,
    pub(super) per_tile_command_list: BufferHandle,
}

/// A coarse-phase operation whose constructor fixes its kernel and binding layout.
pub(super) struct CoarseDispatch(DispatchIntent);

impl CoarseDispatch {
    pub(super) fn path_tag_reduce(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        path_reduced: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTagReduce,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(path_reduced),
            ],
        ))
    }

    pub(super) fn path_tag_reduce2(
        workgroups: vello_encoding::WorkgroupSize,
        path_reduced: BufferHandle,
        path_reduced2: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTagReduce2,
            workgroups,
            vec![
                ResourceBinding::Buffer(path_reduced),
                ResourceBinding::Buffer(path_reduced2),
            ],
        ))
    }

    pub(super) fn path_tag_scan1(
        workgroups: vello_encoding::WorkgroupSize,
        path_reduced: BufferHandle,
        path_reduced2: BufferHandle,
        path_reduced_scan: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTagScan1,
            workgroups,
            vec![
                ResourceBinding::Buffer(path_reduced),
                ResourceBinding::Buffer(path_reduced2),
                ResourceBinding::Buffer(path_reduced_scan),
            ],
        ))
    }

    pub(super) fn path_tag_scan(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        path_tag_parent: BufferHandle,
        path_monoids: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTagScan,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(path_tag_parent),
                ResourceBinding::Buffer(path_monoids),
            ],
        ))
    }

    pub(super) fn path_tag_scan_large(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        path_tag_parent: BufferHandle,
        path_monoids: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTagScanLarge,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(path_tag_parent),
                ResourceBinding::Buffer(path_monoids),
            ],
        ))
    }

    pub(super) fn bbox_clear(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        path_bboxes: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::BboxClear,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(path_bboxes),
            ],
        ))
    }

    pub(super) fn flatten(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        path_monoids: BufferHandle,
        path_bboxes: BufferHandle,
        bump: BufferHandle,
        lines: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::Flatten,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(path_monoids),
                ResourceBinding::Buffer(path_bboxes),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(lines),
            ],
        ))
    }

    pub(super) fn draw_reduce(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        draw_reduced: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::DrawReduce,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(draw_reduced),
            ],
        ))
    }

    pub(super) fn draw_leaf(
        workgroups: vello_encoding::WorkgroupSize,
        bindings: DrawLeafBindings,
    ) -> Self {
        let DrawLeafBindings {
            config,
            scene,
            draw_reduced,
            path_bboxes,
            draw_monoids,
            info_bin_data,
            clip_inputs,
        } = bindings;
        Self(DispatchIntent::coarse_direct(
            RasterKernel::DrawLeaf,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(draw_reduced),
                ResourceBinding::Buffer(path_bboxes),
                ResourceBinding::Buffer(draw_monoids),
                ResourceBinding::Buffer(info_bin_data),
                ResourceBinding::Buffer(clip_inputs),
            ],
        ))
    }

    pub(super) fn clip_reduce(
        workgroups: vello_encoding::WorkgroupSize,
        clip_inputs: BufferHandle,
        path_bboxes: BufferHandle,
        clip_bics: BufferHandle,
        clip_elements: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::ClipReduce,
            workgroups,
            vec![
                ResourceBinding::Buffer(clip_inputs),
                ResourceBinding::Buffer(path_bboxes),
                ResourceBinding::Buffer(clip_bics),
                ResourceBinding::Buffer(clip_elements),
            ],
        ))
    }

    pub(super) fn clip_leaf(
        workgroups: vello_encoding::WorkgroupSize,
        bindings: ClipLeafBindings,
    ) -> Self {
        let ClipLeafBindings {
            config,
            clip_inputs,
            path_bboxes,
            clip_bics,
            clip_elements,
            draw_monoids,
            clip_bboxes,
        } = bindings;
        Self(DispatchIntent::coarse_direct(
            RasterKernel::ClipLeaf,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(clip_inputs),
                ResourceBinding::Buffer(path_bboxes),
                ResourceBinding::Buffer(clip_bics),
                ResourceBinding::Buffer(clip_elements),
                ResourceBinding::Buffer(draw_monoids),
                ResourceBinding::Buffer(clip_bboxes),
            ],
        ))
    }

    pub(super) fn binning(
        workgroups: vello_encoding::WorkgroupSize,
        bindings: BinningBindings,
    ) -> Self {
        let BinningBindings {
            config,
            draw_monoids,
            path_bboxes,
            clip_bboxes,
            draw_bboxes,
            bump,
            info_bin_data,
            bin_headers,
        } = bindings;
        Self(DispatchIntent::coarse_direct(
            RasterKernel::Binning,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(draw_monoids),
                ResourceBinding::Buffer(path_bboxes),
                ResourceBinding::Buffer(clip_bboxes),
                ResourceBinding::Buffer(draw_bboxes),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(info_bin_data),
                ResourceBinding::Buffer(bin_headers),
            ],
        ))
    }

    pub(super) fn tile_alloc(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        scene: BufferHandle,
        draw_bboxes: BufferHandle,
        bump: BufferHandle,
        paths: BufferHandle,
        tile: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::TileAlloc,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(draw_bboxes),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(paths),
                ResourceBinding::Buffer(tile),
            ],
        ))
    }

    pub(super) fn path_count_setup(
        workgroups: vello_encoding::WorkgroupSize,
        bump: BufferHandle,
        indirect_count: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathCountSetup,
            workgroups,
            vec![
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(indirect_count),
            ],
        ))
    }

    pub(super) fn path_count(
        indirect_count: BufferHandle,
        config: BufferHandle,
        bump: BufferHandle,
        lines: BufferHandle,
        paths: BufferHandle,
        tile: BufferHandle,
        segment_counts: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_indirect(
            RasterKernel::PathCount,
            indirect_count,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(lines),
                ResourceBinding::Buffer(paths),
                ResourceBinding::Buffer(tile),
                ResourceBinding::Buffer(segment_counts),
            ],
        ))
    }

    pub(super) fn backdrop(
        workgroups: vello_encoding::WorkgroupSize,
        config: BufferHandle,
        bump: BufferHandle,
        paths: BufferHandle,
        tile: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::Backdrop,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(paths),
                ResourceBinding::Buffer(tile),
            ],
        ))
    }

    pub(super) fn coarse(
        workgroups: vello_encoding::WorkgroupSize,
        bindings: CoarseRasterBindings,
    ) -> Self {
        let CoarseRasterBindings {
            config,
            scene,
            draw_monoids,
            bin_headers,
            info_bin_data,
            paths,
            tile,
            bump,
            per_tile_command_list,
        } = bindings;
        Self(DispatchIntent::coarse_direct(
            RasterKernel::Coarse,
            workgroups,
            vec![
                ResourceBinding::Buffer(config),
                ResourceBinding::Buffer(scene),
                ResourceBinding::Buffer(draw_monoids),
                ResourceBinding::Buffer(bin_headers),
                ResourceBinding::Buffer(info_bin_data),
                ResourceBinding::Buffer(paths),
                ResourceBinding::Buffer(tile),
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(per_tile_command_list),
            ],
        ))
    }

    pub(super) fn path_tiling_setup(
        workgroups: vello_encoding::WorkgroupSize,
        bump: BufferHandle,
        indirect_count: BufferHandle,
        per_tile_command_list: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_direct(
            RasterKernel::PathTilingSetup,
            workgroups,
            vec![
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(indirect_count),
                ResourceBinding::Buffer(per_tile_command_list),
            ],
        ))
    }

    pub(super) fn path_tiling(
        indirect_count: BufferHandle,
        bump: BufferHandle,
        segment_counts: BufferHandle,
        lines: BufferHandle,
        paths: BufferHandle,
        tile: BufferHandle,
        segments: BufferHandle,
    ) -> Self {
        Self(DispatchIntent::coarse_indirect(
            RasterKernel::PathTiling,
            indirect_count,
            vec![
                ResourceBinding::Buffer(bump),
                ResourceBinding::Buffer(segment_counts),
                ResourceBinding::Buffer(lines),
                ResourceBinding::Buffer(paths),
                ResourceBinding::Buffer(tile),
                ResourceBinding::Buffer(segments),
            ],
        ))
    }

    fn into_dispatch_intent(self) -> DispatchIntent {
        self.0
    }
}

/// The shared resources bound by every fine-phase operation.
pub(super) struct FineDispatchBindings {
    pub(super) workgroups: vello_encoding::WorkgroupSize,
    pub(super) config: BufferHandle,
    pub(super) segments: BufferHandle,
    pub(super) per_tile_command_list: BufferHandle,
    pub(super) info_bin_data: BufferHandle,
    pub(super) blend_spill: BufferHandle,
    pub(super) gradient_image: ImageHandle,
    pub(super) image_atlas: ImageHandle,
}

impl FineDispatchBindings {
    fn into_parts(self) -> (vello_encoding::WorkgroupSize, Vec<ResourceBinding>) {
        (
            self.workgroups,
            vec![
                ResourceBinding::Buffer(self.config),
                ResourceBinding::Buffer(self.segments),
                ResourceBinding::Buffer(self.per_tile_command_list),
                ResourceBinding::Buffer(self.info_bin_data),
                ResourceBinding::Buffer(self.blend_spill),
                ResourceBinding::TargetOutput,
                ResourceBinding::Image(self.gradient_image),
                ResourceBinding::Image(self.image_atlas),
            ],
        )
    }
}

/// A fine-phase operation with a fixed final-raster kernel and binding layout.
pub(super) struct FineDispatch(DispatchIntent);

impl FineDispatch {
    pub(super) fn area(bindings: FineDispatchBindings) -> Self {
        let (workgroups, bindings) = bindings.into_parts();
        Self(DispatchIntent::fine_direct(
            RasterKernel::FineArea,
            workgroups,
            bindings,
        ))
    }

    pub(super) fn msaa8(bindings: FineDispatchBindings, mask_lut: BufferHandle) -> Self {
        let (workgroups, mut bindings) = bindings.into_parts();
        bindings.push(ResourceBinding::Buffer(mask_lut));
        Self(DispatchIntent::fine_direct(
            RasterKernel::FineMsaa8,
            workgroups,
            bindings,
        ))
    }

    pub(super) fn msaa16(bindings: FineDispatchBindings, mask_lut: BufferHandle) -> Self {
        let (workgroups, mut bindings) = bindings.into_parts();
        bindings.push(ResourceBinding::Buffer(mask_lut));
        Self(DispatchIntent::fine_direct(
            RasterKernel::FineMsaa16,
            workgroups,
            bindings,
        ))
    }

    fn into_dispatch_intent(self) -> DispatchIntent {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResourceReference {
    Buffer(BufferHandle),
    Image(ImageHandle),
}

impl From<BufferHandle> for ResourceReference {
    fn from(value: BufferHandle) -> Self {
        Self::Buffer(value)
    }
}

impl From<ImageHandle> for ResourceReference {
    fn from(value: ImageHandle) -> Self {
        Self::Image(value)
    }
}

/// A recorded compute dispatch with only symbolic resource bindings.
pub(super) struct DispatchIntent {
    phase: RasterPhase,
    kernel: RasterKernel,
    workgroups: vello_encoding::WorkgroupSize,
    bindings: Vec<ResourceBinding>,
    indirect: Option<IndirectDispatch>,
}

pub(super) struct IndirectDispatch {
    buffer: BufferHandle,
    offset: u64,
}

/// A typed resource request, not a live allocation.
pub(super) enum ResourceIntent {
    Buffer(BufferIntent),
    Image(ImageIntent),
}

pub(super) struct BufferIntent {
    resource: BufferHandle,
    role: BufferRole,
    byte_len: u64,
    #[cfg(test)]
    allocation_command_index: usize,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 image intent metadata is consumed by the later resource manager."
    )
)]
pub(super) struct ImageIntent {
    resource: ImageHandle,
    role: ImageRole,
    extent: PhysicalSize,
    format: RasterImageFormat,
    retention: ImageRetention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImageRetention {
    Transient,
    PersistentImageAtlas,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BufferRole {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImageRole {
    GradientRamp,
    ImageAtlas,
}

/// A runtime-resource-free sequence of uploads, compute dispatches, and symbolic releases.
pub(super) struct Recording {
    commands: Vec<RasterCommand>,
}

pub(super) enum RasterCommand {
    UploadScene {
        buffer: BufferHandle,
        packed: Vec<u8>,
    },
    UploadConfig {
        buffer: BufferHandle,
        config: ConfigUniform,
    },
    UploadGradientRamps {
        image: ImageHandle,
        ramps: Vec<u32>,
    },
    UploadMaskLut {
        buffer: BufferHandle,
        variant: FineRasterVariant,
        samples: Vec<u8>,
    },
    WriteImage {
        image: ImageHandle,
        origin: [u32; 2],
        image_data: ImageData,
    },
    ClearBuffer(BufferHandle),
    Dispatch(DispatchIntent),
    Release(ResourceReference),
}

/// Builder-only state that turns symbolic requests into a completed recording.
pub(super) struct RecordingBuilder {
    next_resource_id: u64,
    commands: Vec<RasterCommand>,
    resource_intents: Vec<ResourceIntent>,
}

impl Default for RecordingBuilder {
    fn default() -> Self {
        Self {
            next_resource_id: 1,
            commands: Vec::new(),
            resource_intents: Vec::new(),
        }
    }
}

impl RecordingBuilder {
    pub(super) fn upload_scene(&mut self, packed: Vec<u8>) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer(BufferRole::Scene, packed.len() as u64)?;
        self.commands
            .push(RasterCommand::UploadScene { buffer, packed });
        Ok(buffer)
    }

    pub(super) fn upload_config(&mut self, config: ConfigUniform) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer(
            BufferRole::Config,
            std::mem::size_of::<ConfigUniform>() as u64,
        )?;
        self.commands
            .push(RasterCommand::UploadConfig { buffer, config });
        Ok(buffer)
    }

    pub(super) fn upload_gradient_ramps(
        &mut self,
        extent: PhysicalSize,
        ramps: Vec<u32>,
    ) -> Result<ImageHandle> {
        let image = self.allocate_image(
            ImageRole::GradientRamp,
            extent,
            RasterImageFormat::Rgba8Unorm,
            ImageRetention::Transient,
        )?;
        self.commands
            .push(RasterCommand::UploadGradientRamps { image, ramps });
        Ok(image)
    }

    pub(super) fn new_transient_image(
        &mut self,
        role: ImageRole,
        extent: PhysicalSize,
    ) -> Result<ImageHandle> {
        self.allocate_image(
            role,
            extent,
            RasterImageFormat::Rgba8Unorm,
            ImageRetention::Transient,
        )
    }

    pub(super) fn request_image_atlas(&mut self, extent: PhysicalSize) -> Result<ImageHandle> {
        self.allocate_image(
            ImageRole::ImageAtlas,
            extent,
            RasterImageFormat::Rgba8Unorm,
            ImageRetention::PersistentImageAtlas,
        )
    }

    pub(super) fn write_image(
        &mut self,
        image: ImageHandle,
        origin: [u32; 2],
        image_data: ImageData,
    ) {
        self.commands.push(RasterCommand::WriteImage {
            image,
            origin,
            image_data,
        });
    }

    pub(super) fn new_buffer(&mut self, role: BufferRole, byte_len: u64) -> Result<BufferHandle> {
        self.allocate_buffer(role, byte_len)
    }

    pub(super) fn clear_buffer(&mut self, buffer: BufferHandle) {
        self.commands.push(RasterCommand::ClearBuffer(buffer));
    }

    pub(super) fn upload_mask_lut(
        &mut self,
        variant: FineRasterVariant,
        samples: Vec<u8>,
    ) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer(BufferRole::MaskLut, samples.len() as u64)?;
        self.commands.push(RasterCommand::UploadMaskLut {
            buffer,
            variant,
            samples,
        });
        Ok(buffer)
    }

    pub(super) fn record_coarse(&mut self, dispatch: CoarseDispatch) {
        self.commands
            .push(RasterCommand::Dispatch(dispatch.into_dispatch_intent()));
    }

    pub(super) fn record_fine(&mut self, dispatch: FineDispatch) {
        self.commands
            .push(RasterCommand::Dispatch(dispatch.into_dispatch_intent()));
    }

    pub(super) fn release(&mut self, resource: impl Into<ResourceReference>) {
        self.commands.push(RasterCommand::Release(resource.into()));
    }

    pub(super) fn finish(self) -> (Recording, Vec<ResourceIntent>) {
        (
            Recording {
                commands: self.commands,
            },
            self.resource_intents,
        )
    }

    fn allocate_buffer(&mut self, role: BufferRole, byte_len: u64) -> Result<BufferHandle> {
        if byte_len == 0 {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "Vello recording cannot request an empty buffer",
            ));
        }
        let buffer = BufferHandle(self.next_resource_id()?);
        self.resource_intents
            .push(ResourceIntent::Buffer(BufferIntent {
                resource: buffer,
                role,
                byte_len,
                #[cfg(test)]
                allocation_command_index: self.commands.len(),
            }));
        Ok(buffer)
    }

    fn allocate_image(
        &mut self,
        role: ImageRole,
        extent: PhysicalSize,
        format: RasterImageFormat,
        retention: ImageRetention,
    ) -> Result<ImageHandle> {
        if extent.width() == 0 || extent.height() == 0 {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "Vello recording cannot request an empty image",
            ));
        }
        let image = ImageHandle(self.next_resource_id()?);
        self.resource_intents
            .push(ResourceIntent::Image(ImageIntent {
                resource: image,
                role,
                extent,
                format,
                retention,
            }));
        Ok(image)
    }

    fn next_resource_id(&mut self) -> Result<ResourceId> {
        let Some(next) = self.next_resource_id.checked_add(1) else {
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                "Vello recording exhausted its symbolic resource identities",
            ));
        };
        let id = ResourceId(self.next_resource_id);
        self.next_resource_id = next;
        Ok(id)
    }
}

impl DispatchIntent {
    fn coarse_direct(
        kernel: RasterKernel,
        workgroups: vello_encoding::WorkgroupSize,
        bindings: Vec<ResourceBinding>,
    ) -> Self {
        Self {
            phase: RasterPhase::Coarse,
            kernel,
            workgroups,
            bindings,
            indirect: None,
        }
    }

    fn coarse_indirect(
        kernel: RasterKernel,
        buffer: BufferHandle,
        bindings: Vec<ResourceBinding>,
    ) -> Self {
        Self {
            phase: RasterPhase::Coarse,
            kernel,
            workgroups: (0, 0, 0),
            bindings,
            indirect: Some(IndirectDispatch { buffer, offset: 0 }),
        }
    }

    fn fine_direct(
        kernel: RasterKernel,
        workgroups: vello_encoding::WorkgroupSize,
        bindings: Vec<ResourceBinding>,
    ) -> Self {
        Self {
            phase: RasterPhase::Fine,
            kernel,
            workgroups,
            bindings,
            indirect: None,
        }
    }
}

#[cfg(test)]
impl Recording {
    pub(super) fn schedule_observations_for_test(
        &self,
        intents: &[ResourceIntent],
    ) -> (
        Vec<VelloPassDispatchObservation>,
        Vec<(
            VelloPassResourceForTest,
            VelloPassResourceLifetimeObservation,
        )>,
    ) {
        let resource_roles = ResourceRoleResolverForTest { intents };
        let mut released = Vec::new();
        let mut dispatches = Vec::new();
        for command in &self.commands {
            match command {
                RasterCommand::Dispatch(dispatch) => dispatches.push(
                    dispatch_observation_for_test(dispatch, &resource_roles, &released),
                ),
                RasterCommand::Release(reference) => {
                    released.push(resource_reference_id(*reference));
                }
                _ => {}
            }
        }
        let lifetimes = [
            (
                VelloPassResourceForTest::LargePathReduced2,
                BufferRole::PathReduced2,
            ),
            (
                VelloPassResourceForTest::LargePathReducedScan,
                BufferRole::PathReducedScan,
            ),
            (VelloPassResourceForTest::ClipInputs, BufferRole::ClipInputs),
            (
                VelloPassResourceForTest::ClipElements,
                BufferRole::ClipElements,
            ),
            (VelloPassResourceForTest::ClipBics, BufferRole::ClipBics),
            (VelloPassResourceForTest::ClipBboxes, BufferRole::ClipBboxes),
        ]
        .into_iter()
        .filter_map(|(observed_resource, role)| {
            let (resource, allocation_command_index) = buffer_for_role_for_test(intents, role)?;
            Some((
                observed_resource,
                self.resource_lifetime_for_test(buffer_id(resource), allocation_command_index),
            ))
        })
        .collect();
        (dispatches, lifetimes)
    }

    fn resource_lifetime_for_test(
        &self,
        resource: ResourceId,
        allocation_command_index: usize,
    ) -> VelloPassResourceLifetimeObservation {
        let allocation_after = self.previous_dispatch_for_test(allocation_command_index);
        let (first_use, last_use) = self
            .commands
            .iter()
            .filter_map(|command| match command {
                RasterCommand::Dispatch(dispatch)
                    if dispatch_references_resource_for_test(dispatch, resource) =>
                {
                    Some(operation_for_test(dispatch.kernel))
                }
                _ => None,
            })
            .fold((None, None), |(first, _), operation| {
                (first.or(Some(operation)), Some(operation))
            });
        let release_after = self
            .commands
            .iter()
            .position(|command| {
                matches!(
                    command,
                    RasterCommand::Release(reference)
                        if resource_reference_id(*reference) == resource
                )
            })
            .and_then(|index| self.previous_dispatch_for_test(index));
        VelloPassResourceLifetimeObservation {
            allocation_after,
            first_use,
            last_use,
            release_after,
        }
    }

    fn previous_dispatch_for_test(
        &self,
        command_index: usize,
    ) -> Option<VelloPassOperationForTest> {
        self.commands[..command_index]
            .iter()
            .rev()
            .find_map(|command| match command {
                RasterCommand::Dispatch(dispatch) => Some(operation_for_test(dispatch.kernel)),
                _ => None,
            })
    }

    pub(super) fn is_self_consistent_for_test(&self, intents: &[ResourceIntent]) -> bool {
        let mut known = Vec::with_capacity(intents.len());
        let mut persistent = Vec::new();
        for intent in intents {
            let (resource, is_persistent) = match intent {
                ResourceIntent::Buffer(BufferIntent {
                    resource,
                    role,
                    byte_len,
                    ..
                }) => {
                    if *byte_len == 0 {
                        return false;
                    }
                    let _ = role;
                    (buffer_id(*resource), false)
                }
                ResourceIntent::Image(ImageIntent {
                    resource,
                    role,
                    extent,
                    format,
                    retention,
                }) => {
                    if extent.width() == 0 || extent.height() == 0 {
                        return false;
                    }
                    if !matches!(format, RasterImageFormat::Rgba8Unorm) {
                        return false;
                    }
                    let _ = role;
                    (
                        image_id(*resource),
                        matches!(retention, ImageRetention::PersistentImageAtlas),
                    )
                }
            };
            if known.contains(&resource) {
                return false;
            }
            known.push(resource);
            if is_persistent {
                persistent.push(resource);
            }
        }

        let mut released = Vec::new();
        for command in &self.commands {
            match command {
                RasterCommand::UploadScene { buffer, packed } => {
                    if packed.is_empty() || !is_active(buffer_id(*buffer), &known, &released) {
                        return false;
                    }
                }
                RasterCommand::UploadConfig { buffer, config } => {
                    if config.target_width == 0
                        || config.target_height == 0
                        || !is_active(buffer_id(*buffer), &known, &released)
                    {
                        return false;
                    }
                }
                RasterCommand::UploadGradientRamps { image, ramps } => {
                    if ramps.is_empty() || !is_active(image_id(*image), &known, &released) {
                        return false;
                    }
                }
                RasterCommand::UploadMaskLut {
                    buffer,
                    variant,
                    samples,
                } => {
                    if matches!(variant, FineRasterVariant::Area)
                        || samples.is_empty()
                        || !is_active(buffer_id(*buffer), &known, &released)
                    {
                        return false;
                    }
                }
                RasterCommand::WriteImage {
                    image,
                    origin,
                    image_data,
                } => {
                    if !is_active(image_id(*image), &known, &released) {
                        return false;
                    }
                    let _ = (origin, image_data);
                }
                RasterCommand::ClearBuffer(buffer) => {
                    if !is_active(buffer_id(*buffer), &known, &released) {
                        return false;
                    }
                }
                RasterCommand::Dispatch(dispatch) => {
                    if !dispatch_is_well_formed(dispatch, &known, &released) {
                        return false;
                    }
                }
                RasterCommand::Release(resource) => {
                    let id = resource_reference_id(*resource);
                    if !is_active(id, &known, &released) || persistent.contains(&id) {
                        return false;
                    }
                    released.push(id);
                }
            }
        }

        released.len() + persistent.len() == known.len()
    }

    pub(super) fn final_dispatch_targets_output_for_test(&self) -> bool {
        self.commands
            .iter()
            .rev()
            .find_map(|command| match command {
                RasterCommand::Dispatch(dispatch) => Some(
                    dispatch
                        .bindings
                        .iter()
                        .any(|binding| matches!(binding, ResourceBinding::TargetOutput)),
                ),
                _ => None,
            })
            == Some(true)
    }
}

#[cfg(test)]
struct ResourceRoleResolverForTest<'a> {
    intents: &'a [ResourceIntent],
}

#[cfg(test)]
impl ResourceRoleResolverForTest<'_> {
    fn binding_for_test(
        &self,
        binding: &ResourceBinding,
        released: &[ResourceId],
    ) -> VelloPassBindingForTest {
        match binding {
            ResourceBinding::Buffer(buffer) => {
                VelloPassBindingForTest::Buffer(self.buffer_role_for_test(*buffer, released))
            }
            ResourceBinding::Image(image) => {
                VelloPassBindingForTest::Image(self.image_role_for_test(*image, released))
            }
            ResourceBinding::TargetOutput => VelloPassBindingForTest::TargetOutput,
        }
    }

    fn indirect_for_test(
        &self,
        indirect: &IndirectDispatch,
        released: &[ResourceId],
    ) -> VelloPassIndirectDispatchForTest {
        VelloPassIndirectDispatchForTest::new(
            self.buffer_role_for_test(indirect.buffer, released),
            indirect.offset,
        )
    }

    fn buffer_role_for_test(
        &self,
        buffer: BufferHandle,
        released: &[ResourceId],
    ) -> VelloPassBufferRoleForTest {
        match self.live_intent_for_test(buffer_id(buffer), "buffer", released) {
            ResourceIntent::Buffer(BufferIntent { role, .. }) => buffer_role_for_test(*role),
            ResourceIntent::Image(_) => {
                panic!("Vello recording bound an image allocation as a buffer at dispatch")
            }
        }
    }

    fn image_role_for_test(
        &self,
        image: ImageHandle,
        released: &[ResourceId],
    ) -> VelloPassImageRoleForTest {
        match self.live_intent_for_test(image_id(image), "image", released) {
            ResourceIntent::Buffer(_) => {
                panic!("Vello recording bound a buffer allocation as an image at dispatch")
            }
            ResourceIntent::Image(ImageIntent { role, .. }) => image_role_for_test(*role),
        }
    }

    fn live_intent_for_test(
        &self,
        resource: ResourceId,
        binding_kind: &str,
        released: &[ResourceId],
    ) -> &ResourceIntent {
        if released.contains(&resource) {
            panic!(
                "Vello recording observed a released {binding_kind} allocation at dispatch: {resource:?}"
            );
        }
        let mut matching = self
            .intents
            .iter()
            .filter(|intent| resource_intent_id_for_test(intent) == resource);
        let Some(intent) = matching.next() else {
            panic!(
                "Vello recording observed a {binding_kind} allocation without a resource intent: {resource:?}"
            );
        };
        if matching.next().is_some() {
            panic!(
                "Vello recording observed an ambiguous {binding_kind} allocation role at dispatch: {resource:?}"
            );
        }
        intent
    }
}

#[cfg(test)]
fn dispatch_observation_for_test(
    dispatch: &DispatchIntent,
    resource_roles: &ResourceRoleResolverForTest<'_>,
    released: &[ResourceId],
) -> VelloPassDispatchObservation {
    VelloPassDispatchObservation {
        phase: phase_for_test(dispatch.phase),
        operation: operation_for_test(dispatch.kernel),
        bindings: dispatch
            .bindings
            .iter()
            .map(|binding| resource_roles.binding_for_test(binding, released))
            .collect(),
        indirect: dispatch
            .indirect
            .as_ref()
            .map(|indirect| resource_roles.indirect_for_test(indirect, released)),
    }
}

#[cfg(test)]
const fn phase_for_test(phase: RasterPhase) -> VelloPassPhaseForTest {
    match phase {
        RasterPhase::Coarse => VelloPassPhaseForTest::Coarse,
        RasterPhase::Fine => VelloPassPhaseForTest::Fine,
    }
}

#[cfg(test)]
const fn operation_for_test(kernel: RasterKernel) -> VelloPassOperationForTest {
    match kernel {
        RasterKernel::PathTagReduce => VelloPassOperationForTest::PathTagReduce,
        RasterKernel::PathTagReduce2 => VelloPassOperationForTest::PathTagReduce2,
        RasterKernel::PathTagScan1 => VelloPassOperationForTest::PathTagScan1,
        RasterKernel::PathTagScan => VelloPassOperationForTest::PathTagScan,
        RasterKernel::PathTagScanLarge => VelloPassOperationForTest::PathTagScanLarge,
        RasterKernel::BboxClear => VelloPassOperationForTest::BboxClear,
        RasterKernel::Flatten => VelloPassOperationForTest::Flatten,
        RasterKernel::DrawReduce => VelloPassOperationForTest::DrawReduce,
        RasterKernel::DrawLeaf => VelloPassOperationForTest::DrawLeaf,
        RasterKernel::ClipReduce => VelloPassOperationForTest::ClipReduce,
        RasterKernel::ClipLeaf => VelloPassOperationForTest::ClipLeaf,
        RasterKernel::Binning => VelloPassOperationForTest::Binning,
        RasterKernel::TileAlloc => VelloPassOperationForTest::TileAlloc,
        RasterKernel::PathCountSetup => VelloPassOperationForTest::PathCountSetup,
        RasterKernel::PathCount => VelloPassOperationForTest::PathCount,
        RasterKernel::Backdrop => VelloPassOperationForTest::Backdrop,
        RasterKernel::Coarse => VelloPassOperationForTest::Coarse,
        RasterKernel::PathTilingSetup => VelloPassOperationForTest::PathTilingSetup,
        RasterKernel::PathTiling => VelloPassOperationForTest::PathTiling,
        RasterKernel::FineArea => VelloPassOperationForTest::FineArea,
        RasterKernel::FineMsaa8 => VelloPassOperationForTest::FineMsaa8,
        RasterKernel::FineMsaa16 => VelloPassOperationForTest::FineMsaa16,
    }
}

#[cfg(test)]
const fn buffer_role_for_test(role: BufferRole) -> VelloPassBufferRoleForTest {
    match role {
        BufferRole::Scene => VelloPassBufferRoleForTest::Scene,
        BufferRole::Config => VelloPassBufferRoleForTest::Config,
        BufferRole::InfoBinData => VelloPassBufferRoleForTest::InfoBinData,
        BufferRole::Tile => VelloPassBufferRoleForTest::Tile,
        BufferRole::Segments => VelloPassBufferRoleForTest::Segments,
        BufferRole::PerTileCommandList => VelloPassBufferRoleForTest::PerTileCommandList,
        BufferRole::PathReduced => VelloPassBufferRoleForTest::PathReduced,
        BufferRole::PathReduced2 => VelloPassBufferRoleForTest::PathReduced2,
        BufferRole::PathReducedScan => VelloPassBufferRoleForTest::PathReducedScan,
        BufferRole::PathMonoids => VelloPassBufferRoleForTest::PathMonoids,
        BufferRole::PathBboxes => VelloPassBufferRoleForTest::PathBboxes,
        BufferRole::Bump => VelloPassBufferRoleForTest::Bump,
        BufferRole::Lines => VelloPassBufferRoleForTest::Lines,
        BufferRole::DrawReduced => VelloPassBufferRoleForTest::DrawReduced,
        BufferRole::DrawMonoids => VelloPassBufferRoleForTest::DrawMonoids,
        BufferRole::ClipInputs => VelloPassBufferRoleForTest::ClipInputs,
        BufferRole::ClipElements => VelloPassBufferRoleForTest::ClipElements,
        BufferRole::ClipBics => VelloPassBufferRoleForTest::ClipBics,
        BufferRole::ClipBboxes => VelloPassBufferRoleForTest::ClipBboxes,
        BufferRole::DrawBboxes => VelloPassBufferRoleForTest::DrawBboxes,
        BufferRole::BinHeaders => VelloPassBufferRoleForTest::BinHeaders,
        BufferRole::Paths => VelloPassBufferRoleForTest::Paths,
        BufferRole::IndirectCount => VelloPassBufferRoleForTest::IndirectCount,
        BufferRole::SegmentCounts => VelloPassBufferRoleForTest::SegmentCounts,
        BufferRole::BlendSpill => VelloPassBufferRoleForTest::BlendSpill,
        BufferRole::MaskLut => VelloPassBufferRoleForTest::MaskLut,
    }
}

#[cfg(test)]
const fn image_role_for_test(role: ImageRole) -> VelloPassImageRoleForTest {
    match role {
        ImageRole::GradientRamp => VelloPassImageRoleForTest::GradientRamp,
        ImageRole::ImageAtlas => VelloPassImageRoleForTest::ImageAtlas,
    }
}

#[cfg(test)]
const fn resource_intent_id_for_test(intent: &ResourceIntent) -> ResourceId {
    match intent {
        ResourceIntent::Buffer(BufferIntent { resource, .. }) => buffer_id(*resource),
        ResourceIntent::Image(ImageIntent { resource, .. }) => image_id(*resource),
    }
}

#[cfg(test)]
fn buffer_for_role_for_test(
    intents: &[ResourceIntent],
    expected_role: BufferRole,
) -> Option<(BufferHandle, usize)> {
    intents.iter().find_map(|intent| match intent {
        ResourceIntent::Buffer(BufferIntent {
            resource,
            role,
            allocation_command_index,
            ..
        }) if *role == expected_role => Some((*resource, *allocation_command_index)),
        _ => None,
    })
}

#[cfg(test)]
fn dispatch_references_resource_for_test(dispatch: &DispatchIntent, resource: ResourceId) -> bool {
    dispatch.bindings.iter().any(|binding| match binding {
        ResourceBinding::Buffer(buffer) => buffer_id(*buffer) == resource,
        ResourceBinding::Image(image) => image_id(*image) == resource,
        ResourceBinding::TargetOutput => false,
    }) || dispatch
        .indirect
        .as_ref()
        .is_some_and(|indirect| buffer_id(indirect.buffer) == resource)
}

#[cfg(test)]
impl ResourceIntent {
    pub(super) const fn is_persistent_image_atlas_for_test(&self) -> bool {
        matches!(
            self,
            Self::Image(ImageIntent {
                role: ImageRole::ImageAtlas,
                retention: ImageRetention::PersistentImageAtlas,
                ..
            })
        )
    }

    pub(super) const fn is_transient_buffer_for_test(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }
}

#[cfg(test)]
fn dispatch_is_well_formed(
    dispatch: &DispatchIntent,
    known: &[ResourceId],
    released: &[ResourceId],
) -> bool {
    let is_fine_kernel = matches!(
        dispatch.kernel,
        RasterKernel::FineArea | RasterKernel::FineMsaa8 | RasterKernel::FineMsaa16
    );
    if (is_fine_kernel && dispatch.phase != RasterPhase::Fine)
        || (!is_fine_kernel && dispatch.phase != RasterPhase::Coarse)
        || dispatch.bindings.is_empty()
    {
        return false;
    }
    if let Some(IndirectDispatch { buffer, offset }) = dispatch.indirect {
        if dispatch.workgroups != (0, 0, 0)
            || !is_active(buffer_id(buffer), known, released)
            || offset % size_of::<u32>() as u64 != 0
        {
            return false;
        }
    } else if dispatch.workgroups == (0, 0, 0) {
        return false;
    }
    dispatch.bindings.iter().all(|binding| match binding {
        ResourceBinding::Buffer(buffer) => is_active(buffer_id(*buffer), known, released),
        ResourceBinding::Image(image) => is_active(image_id(*image), known, released),
        ResourceBinding::TargetOutput => true,
    })
}

#[cfg(test)]
fn is_active(resource: ResourceId, known: &[ResourceId], released: &[ResourceId]) -> bool {
    known.contains(&resource) && !released.contains(&resource)
}

#[cfg(test)]
const fn resource_reference_id(reference: ResourceReference) -> ResourceId {
    match reference {
        ResourceReference::Buffer(buffer) => buffer_id(buffer),
        ResourceReference::Image(image) => image_id(image),
    }
}

#[cfg(test)]
const fn buffer_id(buffer: BufferHandle) -> ResourceId {
    buffer.0
}

#[cfg(test)]
const fn image_id(image: ImageHandle) -> ResourceId {
    image.0
}
