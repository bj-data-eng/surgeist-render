use super::{
    BackendErrorCode, Error, ImageBuffer, PhysicalSize, Result, RuntimeOperation,
    backend::{Backend, DeviceSlotIdentity},
    gpu_transaction::GpuOperationStage,
};

const RGBA8_BYTES_PER_PIXEL: u64 = 4;
const COPY_BYTES_PER_ROW_ALIGNMENT: u64 = 256;

struct ReadbackLayout {
    width: u32,
    height: u32,
    row_bytes: usize,
    padded_bytes_per_row: u32,
    buffer_size: u64,
    decoded_len: usize,
}

impl ReadbackLayout {
    fn try_new(size: PhysicalSize) -> Result<Self> {
        let width = size.width();
        let height = size.height();
        let row_bytes_u64 = u64::from(width)
            .checked_mul(RGBA8_BYTES_PER_PIXEL)
            .ok_or_else(|| readback_failed("readback row byte count overflowed"))?;
        let padded_bytes_per_row_u64 = row_bytes_u64
            .checked_add(COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            .map(|bytes| bytes / COPY_BYTES_PER_ROW_ALIGNMENT * COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| readback_failed("aligned readback row byte count overflowed"))?;
        let padded_bytes_per_row = u32::try_from(padded_bytes_per_row_u64)
            .map_err(|_| readback_failed("aligned readback row byte count exceeds WGPU limits"))?;
        let buffer_size = padded_bytes_per_row_u64
            .checked_mul(u64::from(height))
            .ok_or_else(|| readback_failed("readback staging buffer size overflowed"))?;
        let row_bytes = usize::try_from(row_bytes_u64)
            .map_err(|_| readback_failed("readback row byte count exceeds addressable memory"))?;
        let decoded_len = row_bytes
            .checked_mul(
                usize::try_from(height)
                    .map_err(|_| readback_failed("readback height exceeds addressable memory"))?,
            )
            .ok_or_else(|| readback_failed("decoded readback byte count overflowed"))?;
        Ok(Self {
            width,
            height,
            row_bytes,
            padded_bytes_per_row,
            buffer_size,
            decoded_len,
        })
    }
}

pub(crate) async fn read_texture_rgba(
    backend: &mut Backend,
    device_identity: DeviceSlotIdentity,
    texture: &wgpu::Texture,
    physical_size: PhysicalSize,
    operation: RuntimeOperation,
) -> Result<ImageBuffer> {
    if physical_size.width() == 0 || physical_size.height() == 0 {
        return ImageBuffer::try_new(physical_size, Vec::new());
    }

    let transaction =
        backend.begin_gpu_operation(device_identity, GpuOperationStage::Readback, operation)?;
    let layout = match ReadbackLayout::try_new(physical_size) {
        Ok(layout) => layout,
        Err(error) => {
            let scope_result = transaction.finish(operation).await;
            backend.observe_device_terminal(device_identity);
            scope_result?;
            return Err(error);
        }
    };
    let (buffer, submission_result) = {
        let (device, queue) = backend.gpu_operation_device_queue(
            device_identity,
            operation,
            GpuOperationStage::Readback,
        )?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist scoped texture readback"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist scoped texture readback copy"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
        );
        let submission_result = transaction
            .submit_readback(queue, encoder.finish(), operation)
            .await;
        (buffer, submission_result)
    };
    backend.observe_device_terminal(device_identity);
    let submission = submission_result?;

    let decoded = {
        let (device, _) = backend.gpu_operation_device_queue(
            device_identity,
            operation,
            GpuOperationStage::Readback,
        )?;
        map_and_decode(device, &buffer, submission.into_submission_index(), &layout)
    };
    backend.observe_device_terminal(device_identity);
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    let rgba = decoded?;
    ImageBuffer::try_new(physical_size, rgba).map_err(|source| {
        Error::new(
            BackendErrorCode::ReadbackFailed,
            "decoded readback bytes did not form a valid RGBA8 image",
        )
        .with_source(source)
    })
}

fn map_and_decode(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    submission_index: wgpu::SubmissionIndex,
    layout: &ReadbackLayout,
) -> Result<Vec<u8>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "failed while waiting for the texture readback copy",
            )
            .with_source(source)
        })?;
    receiver
        .recv()
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "texture readback map callback was dropped",
            )
            .with_source(source)
        })?
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "failed to map the texture readback staging buffer",
            )
            .with_source(source)
        })?;

    let mapped = slice.get_mapped_range();
    let decoded = (|| {
        let mut rgba = Vec::with_capacity(layout.decoded_len);
        for row in 0..layout.height {
            let start = u64::from(row)
                .checked_mul(u64::from(layout.padded_bytes_per_row))
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| readback_failed("mapped readback row offset overflowed"))?;
            let end = start
                .checked_add(layout.row_bytes)
                .ok_or_else(|| readback_failed("mapped readback row end overflowed"))?;
            let row = mapped
                .get(start..end)
                .ok_or_else(|| readback_failed("mapped readback row was incomplete"))?;
            rgba.extend_from_slice(row);
        }
        if rgba.len() != layout.decoded_len {
            return Err(readback_failed(
                "decoded readback byte count did not match the validated layout",
            ));
        }
        Ok(rgba)
    })();
    drop(mapped);
    buffer.unmap();
    decoded
}

fn readback_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::ReadbackFailed, message)
}
