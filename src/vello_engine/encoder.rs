// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::ImageFormat;

use crate::{BackendErrorCode, Error, PhysicalSize, Result};

use super::resources::VelloResourceLease;
use super::{
    DispatchIntent, FineRasterVariant, RasterCommand, RasterKernel, RasterPhase, Recording,
    ResourceBinding, ResourceIntent,
};
use super::super::shaders::CheckedShaderSet;

pub(crate) struct TransactionEncodingState<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    command_encoder: &'a mut wgpu::CommandEncoder,
    target_view: &'a wgpu::TextureView,
    target_extent: PhysicalSize,
    target_format: wgpu::TextureFormat,
}

impl<'a> TransactionEncodingState<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 exposes transaction-owned construction to the later T7 cutover."
        )
    )]
    pub(crate) fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        command_encoder: &'a mut wgpu::CommandEncoder,
        target_view: &'a wgpu::TextureView,
        target_extent: PhysicalSize,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            command_encoder,
            target_view,
            target_extent,
            target_format,
        }
    }

    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target_extent
    }

    pub(crate) const fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T4 keeps checked private engine construction ready for T7 cutover."
    )
)]
pub(crate) struct VelloEngineState {
    shaders: CheckedShaderSet,
}

impl VelloEngineState {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T4 checked engine construction is staged until T7 invokes private lowering."
        )
    )]
    pub(crate) async fn new_checked(device: &wgpu::Device) -> Result<Self> {
        Ok(Self {
            shaders: CheckedShaderSet::create(device).await?,
        })
    }
}

pub(crate) fn encode_recording(
    engine: &VelloEngineState,
    recording: &Recording,
    resource_intents: &[ResourceIntent],
    state: &mut TransactionEncodingState<'_>,
) -> Result<VelloResourceLease> {
    log::trace!("encoding checked internal Vello raster recording");
    let mut lease = VelloResourceLease::allocate(state.device, resource_intents)?;
    let result = recording
        .commands
        .iter()
        .try_for_each(|command| encode_command(engine, command, state, &mut lease));
    match result {
        Ok(()) => Ok(lease),
        Err(error) => {
            let _aborted = lease.abort();
            Err(error)
        }
    }
}

fn encode_command(
    engine: &VelloEngineState,
    command: &RasterCommand,
    state: &mut TransactionEncodingState<'_>,
    lease: &mut VelloResourceLease,
) -> Result<()> {
    match command {
        RasterCommand::UploadScene { buffer, packed } => {
            let buffer = lease.buffer_for_upload(*buffer, packed.len())?;
            state.queue.write_buffer(buffer, 0, packed);
        }
        RasterCommand::UploadConfig { buffer, config } => {
            let bytes = bytemuck::bytes_of(config);
            let buffer = lease.buffer_for_upload(*buffer, bytes.len())?;
            state.queue.write_buffer(buffer, 0, bytes);
        }
        RasterCommand::UploadGradientRamps { image, ramps } => {
            let bytes = bytemuck::cast_slice(ramps.as_slice());
            write_rgba8_texture(state.queue, lease, *image, [0, 0], None, bytes)?;
        }
        RasterCommand::UploadMaskLut {
            buffer,
            variant,
            samples,
        } => {
            if matches!(variant, FineRasterVariant::Area) {
                return Err(render_failed(
                    "internal Vello recording supplies a mask lookup table for area rasterization",
                ));
            }
            let buffer = lease.buffer_for_upload(*buffer, samples.len())?;
            state.queue.write_buffer(buffer, 0, samples);
        }
        RasterCommand::WriteImage {
            image,
            origin,
            image_data,
        } => {
            if image_data.format != ImageFormat::Rgba8 {
                return Err(render_failed(
                    "internal Vello recording requires RGBA8 image data",
                ));
            }
            write_rgba8_texture(
                state.queue,
                lease,
                *image,
                *origin,
                Some((image_data.width, image_data.height)),
                image_data.data.data(),
            )?;
        }
        RasterCommand::ClearBuffer(buffer) => {
            state.command_encoder.clear_buffer(lease.buffer(*buffer)?, 0, None);
        }
        RasterCommand::Dispatch(dispatch) => encode_dispatch(engine, dispatch, state, lease)?,
        RasterCommand::Release(reference) => lease.record_release(*reference)?,
    }
    Ok(())
}

fn write_rgba8_texture(
    queue: &wgpu::Queue,
    lease: &VelloResourceLease,
    image: super::ImageHandle,
    origin: [u32; 2],
    requested_extent: Option<(u32, u32)>,
    bytes: &[u8],
) -> Result<()> {
    let target_extent = lease.image_extent(image)?;
    let (width, height) = requested_extent
        .unwrap_or((target_extent.width(), target_extent.height()));
    if width == 0 || height == 0 {
        return Err(render_failed(
            "internal Vello texture upload has an empty extent",
        ));
    }
    let end_x = origin[0].checked_add(width);
    let end_y = origin[1].checked_add(height);
    if end_x.is_none_or(|end| end > target_extent.width())
        || end_y.is_none_or(|end| end > target_extent.height())
    {
        return Err(render_failed(
            "internal Vello texture upload exceeds its prepared image",
        ));
    }
    let row_bytes = width.checked_mul(4).ok_or_else(|| {
        render_failed("internal Vello texture row length overflows")
    })?;
    let expected_len = usize::try_from(row_bytes)
        .ok()
        .and_then(|row_bytes| {
            usize::try_from(height)
                .ok()
                .and_then(|height| row_bytes.checked_mul(height))
        })
        .ok_or_else(|| render_failed("internal Vello texture upload length overflows"))?;
    if bytes.len() != expected_len {
        return Err(render_failed(
            "internal Vello texture upload does not match its prepared extent",
        ));
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: lease.image_texture(image)?,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: origin[0],
                y: origin[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn encode_dispatch(
    engine: &VelloEngineState,
    dispatch: &DispatchIntent,
    state: &mut TransactionEncodingState<'_>,
    lease: &VelloResourceLease,
) -> Result<()> {
    if !phase_matches_kernel(dispatch.phase, dispatch.kernel) {
        return Err(render_failed(
            "internal Vello recording assigns a kernel to the wrong phase",
        ));
    }
    if dispatch.indirect.is_none()
        && (dispatch.workgroups.0 == 0 || dispatch.workgroups.1 == 0 || dispatch.workgroups.2 == 0)
    {
        return Ok(());
    }

    let pipeline = engine.shaders.pipeline(dispatch.kernel);
    if dispatch.bindings.len() != pipeline.binding_indices().len() {
        return Err(render_failed(
            "internal Vello recording does not match its shader binding layout",
        ));
    }
    let entries = dispatch
        .bindings
        .iter()
        .zip(pipeline.binding_indices().iter().copied())
        .map(|(binding, index)| binding_entry(binding, index, lease, state.target_view))
        .collect::<Result<Vec<_>>>()?;
    let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Surgeist internal Vello dispatch bindings"),
        layout: pipeline.bind_group_layout(),
        entries: &entries,
    });
    let mut compute_pass = state
        .command_encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Surgeist internal Vello compute pass"),
            timestamp_writes: None,
        });
    compute_pass.set_pipeline(pipeline.pipeline());
    compute_pass.set_bind_group(0, &bind_group, &[]);
    if let Some(indirect) = &dispatch.indirect {
        compute_pass.dispatch_workgroups_indirect(
            lease.indirect_buffer(indirect.buffer, indirect.offset)?,
            indirect.offset,
        );
    } else {
        compute_pass.dispatch_workgroups(
            dispatch.workgroups.0,
            dispatch.workgroups.1,
            dispatch.workgroups.2,
        );
    }
    Ok(())
}

fn binding_entry<'a>(
    binding: &ResourceBinding,
    index: u32,
    lease: &'a VelloResourceLease,
    target_view: &'a wgpu::TextureView,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let resource = match binding {
        ResourceBinding::Buffer(buffer) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: lease.buffer(*buffer)?,
            offset: 0,
            size: None,
        }),
        ResourceBinding::Image(image) => wgpu::BindingResource::TextureView(lease.image_view(*image)?),
        ResourceBinding::TargetOutput => wgpu::BindingResource::TextureView(target_view),
    };
    Ok(wgpu::BindGroupEntry {
        binding: index,
        resource,
    })
}

const fn phase_matches_kernel(phase: RasterPhase, kernel: RasterKernel) -> bool {
    matches!(
        (phase, kernel),
        (RasterPhase::Fine, RasterKernel::FineArea)
            | (RasterPhase::Fine, RasterKernel::FineMsaa8)
            | (RasterPhase::Fine, RasterKernel::FineMsaa16)
            | (RasterPhase::Coarse, RasterKernel::PathTagReduce)
            | (RasterPhase::Coarse, RasterKernel::PathTagReduce2)
            | (RasterPhase::Coarse, RasterKernel::PathTagScan1)
            | (RasterPhase::Coarse, RasterKernel::PathTagScan)
            | (RasterPhase::Coarse, RasterKernel::PathTagScanLarge)
            | (RasterPhase::Coarse, RasterKernel::BboxClear)
            | (RasterPhase::Coarse, RasterKernel::Flatten)
            | (RasterPhase::Coarse, RasterKernel::DrawReduce)
            | (RasterPhase::Coarse, RasterKernel::DrawLeaf)
            | (RasterPhase::Coarse, RasterKernel::ClipReduce)
            | (RasterPhase::Coarse, RasterKernel::ClipLeaf)
            | (RasterPhase::Coarse, RasterKernel::Binning)
            | (RasterPhase::Coarse, RasterKernel::TileAlloc)
            | (RasterPhase::Coarse, RasterKernel::PathCountSetup)
            | (RasterPhase::Coarse, RasterKernel::PathCount)
            | (RasterPhase::Coarse, RasterKernel::Backdrop)
            | (RasterPhase::Coarse, RasterKernel::Coarse)
            | (RasterPhase::Coarse, RasterKernel::PathTilingSetup)
            | (RasterPhase::Coarse, RasterKernel::PathTiling)
    )
}

fn render_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}
