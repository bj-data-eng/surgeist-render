#[cfg(all(test, not(target_arch = "wasm32")))]
use self::native::take_native_readback_observation_for_test;
use self::{layout::ReadbackLayout, lifecycle::ReadbackOwner, native::ReadbackMapFuture};
use super::{
    BackendErrorCode, Error, ImageBuffer, PhysicalSize, Result, RuntimeOperation,
    backend::{Backend, DeviceSlotIdentity},
    gpu_transaction::GpuOperationStage,
};

mod layout;
mod lifecycle;
mod native;
#[cfg(test)]
mod test_support;

#[cfg(test)]
pub(crate) use test_support::{
    ReadbackCleanupEventForTest, ReadbackCompletionForTest, ReadbackPhaseForTest,
    ReadbackStagingDispositionForTest, ReadbackStateMachineForTest,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use native::{
    NativeReadbackObservationForTest, NativeReadbackObservationSnapshotForTest,
    NativeReadbackPhaseForTest, ScopedNativeReadbackObservationForTest,
};

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

    #[cfg(all(test, not(target_arch = "wasm32")))]
    let observation = take_native_readback_observation_for_test();

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
    let (mut owner, pending_submission) = {
        let (device, queue) = backend.gpu_operation_device_queue(
            device_identity,
            operation,
            GpuOperationStage::Readback,
        )?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surgeist scoped texture readback"),
            size: layout.buffer_size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut owner = ReadbackOwner::allocated(buffer, layout, physical_size);
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &observation {
            observation.attach_staging(&owner.staging_map());
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist scoped texture readback copy"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: owner.staging(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(owner.layout.padded_bytes_per_row()),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: owner.layout.width(),
                height: owner.layout.height(),
                depth_or_array_layers: 1,
            },
        );
        let pending_submission = transaction.submit_readback(queue, encoder.finish());
        let submission_index = pending_submission.submission_index();
        #[cfg(all(test, not(target_arch = "wasm32")))]
        if let Some(observation) = &observation {
            observation.record_copy_submitted(&submission_index);
        }
        owner.copy_submitted(submission_index);
        (owner, pending_submission)
    };

    let submission_result = pending_submission.finish(operation).await;
    backend.observe_device_terminal(device_identity);
    let submission = match submission_result {
        Ok(submission) => submission,
        Err(error) => {
            owner.fail();
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &observation {
                observation.record_phase(NativeReadbackPhaseForTest::Failed);
            }
            return Err(error);
        }
    };

    let device = match backend.gpu_operation_device_queue(
        device_identity,
        operation,
        GpuOperationStage::Readback,
    ) {
        Ok((device, _)) => device.clone(),
        Err(error) => {
            owner.fail();
            #[cfg(all(test, not(target_arch = "wasm32")))]
            if let Some(observation) = &observation {
                observation.record_phase(NativeReadbackPhaseForTest::Failed);
            }
            return Err(error);
        }
    };
    let readback_result = match ReadbackMapFuture::start(
        owner,
        device,
        submission.into_submission_index(),
        #[cfg(all(test, not(target_arch = "wasm32")))]
        observation,
    ) {
        Ok(readback) => readback.await,
        Err(error) => Err(error),
    };

    backend.observe_device_terminal(device_identity);
    if let Some(error) = backend.terminal_error(device_identity, operation) {
        return Err(error);
    }
    readback_result
}
