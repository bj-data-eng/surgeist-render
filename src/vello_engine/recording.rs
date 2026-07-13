// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(test)]
use std::mem::size_of;

use peniko::ImageData;
use vello_encoding::ConfigUniform;

use crate::{BackendErrorCode, Error, PhysicalSize, Result};

/// A symbolic resource identity within one prepared Vello pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResourceId(u64);

/// A symbolic buffer reference used by the compute-dispatch recording.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BufferHandle(ResourceId);

/// A symbolic image reference used by the compute-dispatch recording.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ImageHandle(ResourceId);

/// The only image format required by the pinned Vello raster path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RasterImageFormat {
    Rgba8Unorm,
}

/// The fixed algorithm phase associated with a recorded dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RasterPhase {
    Coarse,
    Fine,
}

/// The antialiasing-specific final raster program selected for one pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FineRasterVariant {
    Area,
    Msaa8,
    Msaa16,
}

/// The closed set of compute programs used by the pinned Vello schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RasterKernel {
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
pub(crate) enum ResourceBinding {
    Buffer(BufferHandle),
    Image(ImageHandle),
    TargetOutput,
}

impl From<BufferHandle> for ResourceBinding {
    fn from(value: BufferHandle) -> Self {
        Self::Buffer(value)
    }
}

impl From<ImageHandle> for ResourceBinding {
    fn from(value: ImageHandle) -> Self {
        Self::Image(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceReference {
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
pub(crate) struct DispatchIntent {
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
pub(crate) struct IndirectDispatch {
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
pub(crate) enum ResourceIntent {
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
pub(crate) struct BufferIntent {
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
pub(crate) struct ImageIntent {
    resource: ImageHandle,
    role: ImageRole,
    extent: PhysicalSize,
    format: RasterImageFormat,
    retention: ImageRetention,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageRetention {
    Transient,
    PersistentImageAtlas,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferRole {
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
pub(crate) enum ImageRole {
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
pub(crate) struct Recording {
    commands: Vec<RasterCommand>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T3 recording commands are intentionally consumed by the later realization stage."
    )
)]
pub(crate) enum RasterCommand {
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
pub(crate) struct RecordingBuilder {
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
    pub(crate) fn upload_scene(&mut self, packed: Vec<u8>) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer(BufferRole::Scene, packed.len() as u64)?;
        self.commands
            .push(RasterCommand::UploadScene { buffer, packed });
        Ok(buffer)
    }

    pub(crate) fn upload_config(&mut self, config: ConfigUniform) -> Result<BufferHandle> {
        let buffer = self.allocate_buffer(
            BufferRole::Config,
            std::mem::size_of::<ConfigUniform>() as u64,
        )?;
        self.commands
            .push(RasterCommand::UploadConfig { buffer, config });
        Ok(buffer)
    }

    pub(crate) fn upload_gradient_ramps(
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

    pub(crate) fn new_transient_image(
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

    pub(crate) fn request_image_atlas(&mut self, extent: PhysicalSize) -> Result<ImageHandle> {
        self.allocate_image(
            ImageRole::ImageAtlas,
            extent,
            RasterImageFormat::Rgba8Unorm,
            ImageRetention::PersistentImageAtlas,
        )
    }

    pub(crate) fn write_image(
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

    pub(crate) fn new_buffer(&mut self, role: BufferRole, byte_len: u64) -> Result<BufferHandle> {
        self.allocate_buffer(role, byte_len)
    }

    pub(crate) fn clear_buffer(&mut self, buffer: BufferHandle) {
        self.commands.push(RasterCommand::ClearBuffer(buffer));
    }

    pub(crate) fn upload_mask_lut(
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

    pub(crate) fn dispatch(
        &mut self,
        phase: RasterPhase,
        kernel: RasterKernel,
        workgroups: vello_encoding::WorkgroupSize,
        bindings: impl IntoIterator<Item = ResourceBinding>,
    ) {
        self.commands.push(RasterCommand::Dispatch(DispatchIntent {
            phase,
            kernel,
            workgroups,
            bindings: bindings.into_iter().collect(),
            indirect: None,
        }));
    }

    pub(crate) fn dispatch_indirect(
        &mut self,
        phase: RasterPhase,
        kernel: RasterKernel,
        buffer: BufferHandle,
        offset: u64,
        bindings: impl IntoIterator<Item = ResourceBinding>,
    ) {
        self.commands.push(RasterCommand::Dispatch(DispatchIntent {
            phase,
            kernel,
            workgroups: (0, 0, 0),
            bindings: bindings.into_iter().collect(),
            indirect: Some(IndirectDispatch { buffer, offset }),
        }));
    }

    pub(crate) fn release(&mut self, resource: impl Into<ResourceReference>) {
        self.commands.push(RasterCommand::Release(resource.into()));
    }

    pub(crate) fn finish(self) -> (Recording, Vec<ResourceIntent>) {
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

#[cfg(test)]
impl Recording {
    pub(crate) fn dispatches_for_test(&self) -> Vec<&DispatchIntent> {
        self.commands
            .iter()
            .filter_map(|command| match command {
                RasterCommand::Dispatch(dispatch) => Some(dispatch),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn is_self_consistent_for_test(&self, intents: &[ResourceIntent]) -> bool {
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

    pub(crate) fn final_dispatch_targets_output_for_test(&self) -> bool {
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
impl DispatchIntent {
    pub(crate) const fn phase_for_test(&self) -> RasterPhase {
        self.phase
    }

    pub(crate) const fn kernel_for_test(&self) -> RasterKernel {
        self.kernel
    }

    pub(crate) const fn fine_variant_for_test(&self) -> Option<FineRasterVariant> {
        match self.kernel {
            RasterKernel::FineArea => Some(FineRasterVariant::Area),
            RasterKernel::FineMsaa8 => Some(FineRasterVariant::Msaa8),
            RasterKernel::FineMsaa16 => Some(FineRasterVariant::Msaa16),
            _ => None,
        }
    }
}

#[cfg(test)]
impl ResourceIntent {
    pub(crate) const fn is_persistent_image_atlas_for_test(&self) -> bool {
        matches!(
            self,
            Self::Image(ImageIntent {
                role: ImageRole::ImageAtlas,
                retention: ImageRetention::PersistentImageAtlas,
                ..
            })
        )
    }

    pub(crate) const fn is_transient_buffer_for_test(&self) -> bool {
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
