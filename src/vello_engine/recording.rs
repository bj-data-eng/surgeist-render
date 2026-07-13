// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(test)]
use std::mem::size_of;

use peniko::ImageData;
use vello_encoding::ConfigUniform;

use crate::{BackendErrorCode, Error, PhysicalSize, Result};

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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 dispatch IR is retained for T4 transaction-owned encoding."
    )
)]
pub(super) struct DispatchIntent {
    phase: RasterPhase,
    kernel: RasterKernel,
    workgroups: vello_encoding::WorkgroupSize,
    bindings: Vec<ResourceBinding>,
    indirect: Option<IndirectDispatch>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 indirect-dispatch data is consumed by the later checked realization stage."
    )
)]
pub(super) struct IndirectDispatch {
    buffer: BufferHandle,
    offset: u64,
}

/// A typed resource request, not a live allocation.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 resource intents are deliberately held for later private resource ownership."
    )
)]
pub(super) enum ResourceIntent {
    Buffer(BufferIntent),
    Image(ImageIntent),
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 buffer intent metadata is consumed by the later checked realization stage."
    )
)]
pub(super) struct BufferIntent {
    resource: BufferHandle,
    role: BufferRole,
    byte_len: u64,
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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 recordings are intentionally staged until T4 owns checked encoding."
    )
)]
pub(super) struct Recording {
    commands: Vec<RasterCommand>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 recording commands are intentionally consumed by the later realization stage."
    )
)]
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
    pub(super) fn has_fixed_schedule_for_test(&self, expected_fine: FineRasterVariant) -> bool {
        let dispatches = self
            .commands
            .iter()
            .filter_map(|command| match command {
                RasterCommand::Dispatch(dispatch) => Some(dispatch),
                _ => None,
            })
            .collect::<Vec<_>>();
        let expected_coarse = [
            (RasterKernel::PathTagReduce, 3),
            (RasterKernel::PathTagScan, 4),
            (RasterKernel::BboxClear, 2),
            (RasterKernel::Flatten, 6),
            (RasterKernel::DrawReduce, 3),
            (RasterKernel::DrawLeaf, 7),
            (RasterKernel::Binning, 8),
            (RasterKernel::TileAlloc, 6),
            (RasterKernel::PathCountSetup, 2),
            (RasterKernel::PathCount, 6),
            (RasterKernel::Backdrop, 4),
            (RasterKernel::Coarse, 9),
            (RasterKernel::PathTilingSetup, 3),
            (RasterKernel::PathTiling, 6),
        ];
        if dispatches.len() != expected_coarse.len() + 1 {
            return false;
        }
        if dispatches[..expected_coarse.len()]
            .iter()
            .zip(expected_coarse)
            .any(|(dispatch, (kernel, binding_count))| {
                dispatch.phase != RasterPhase::Coarse
                    || dispatch.kernel != kernel
                    || dispatch.bindings.len() != binding_count
                    || !dispatch
                        .bindings
                        .iter()
                        .all(|binding| matches!(binding, ResourceBinding::Buffer(_)))
            })
        {
            return false;
        }
        let Some(final_dispatch) = dispatches.last() else {
            return false;
        };
        let expected_kernel = match expected_fine {
            FineRasterVariant::Area => RasterKernel::FineArea,
            FineRasterVariant::Msaa8 => RasterKernel::FineMsaa8,
            FineRasterVariant::Msaa16 => RasterKernel::FineMsaa16,
        };
        if final_dispatch.phase != RasterPhase::Fine || final_dispatch.kernel != expected_kernel {
            return false;
        }
        let mut bindings = final_dispatch.bindings.iter();
        if !(0..5).all(|_| matches!(bindings.next(), Some(ResourceBinding::Buffer(_))))
            || !matches!(bindings.next(), Some(ResourceBinding::TargetOutput))
            || !matches!(bindings.next(), Some(ResourceBinding::Image(_)))
            || !matches!(bindings.next(), Some(ResourceBinding::Image(_)))
        {
            return false;
        }
        if !matches!(expected_fine, FineRasterVariant::Area)
            && !matches!(bindings.next(), Some(ResourceBinding::Buffer(_)))
        {
            return false;
        }
        bindings.next().is_none()
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
