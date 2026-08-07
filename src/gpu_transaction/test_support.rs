use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
};

use super::{GpuOperationDraft, GpuOperationStage, GpuOperationTransaction};
#[cfg(feature = "render-window")]
use crate::DeviceLossReason;
use crate::{
    Result, RuntimeOperation,
    backend::DeviceSignal,
    resource::{ResourceCacheKey, ResourceManager},
};

thread_local! {
    static ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST: RefCell<Option<GpuOperationPostSubmitControlForTest>> = const { RefCell::new(None) };
}

/// Exercises the real transaction submission, scope resolution, resource abort,
/// and draft-publication boundaries using only explicit test-owned inputs.
pub(crate) async fn graph_scope_failure_after_submission_for_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    signal: Arc<DeviceSignal>,
    resources: &ResourceManager,
    resource_key: ResourceCacheKey,
    publication: &mut Option<u64>,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let transaction =
        GpuOperationTransaction::begin(device, signal, generation, GpuOperationStage::Render);
    let mut frame = resources.begin_frame()?;
    let _lease = frame.acquire(resource_key, 16)?;
    frame.discard_on_drop();
    let draft = GpuOperationDraft::new(publication, 2);
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit graph transaction failure harness"),
        })
        .finish();
    queue.submit([command_buffer]);
    let _ = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surgeist explicit graph transaction validation failure"),
        size: wgpu::Extent3d {
            width: 0,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    transaction
        .finish(RuntimeOperation::EffectRendering)
        .await?;
    let _cleanup = frame.finish_checked()?;
    draft.commit();
    Ok(())
}

/// Holds a real submitted transaction, prepared resource frame, and draft at
/// an explicit test-owned cancellation boundary. Dropping the returned future
/// exercises their normal cancellation cleanup.
pub(crate) async fn graph_cancellation_after_submission_for_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    signal: Arc<DeviceSignal>,
    resources: &ResourceManager,
    resource_key: ResourceCacheKey,
    publication: &mut Option<u64>,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let _transaction =
        GpuOperationTransaction::begin(device, signal, generation, GpuOperationStage::Render);
    let mut frame = resources.begin_frame()?;
    let _lease = frame.acquire(resource_key, 16)?;
    frame.discard_on_drop();
    let _draft = GpuOperationDraft::new(publication, 2);
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit graph cancellation harness"),
        })
        .finish();
    queue.submit([command_buffer]);
    std::future::pending::<()>().await;
    Ok(())
}

/// Exercises clean scope resolution followed by an explicit resource-accounting
/// rejection before the draft publication may commit.
pub(crate) async fn graph_accounting_failure_after_submission_for_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    signal: Arc<DeviceSignal>,
    resources: &ResourceManager,
    resource_key: ResourceCacheKey,
    publication: &mut Option<u64>,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let transaction =
        GpuOperationTransaction::begin(device, signal, generation, GpuOperationStage::Render);
    let mut frame = resources.begin_frame()?;
    let _lease = frame.acquire(resource_key, 16)?;
    frame.discard_on_drop();
    let draft = GpuOperationDraft::new(publication, 2);
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit graph accounting failure harness"),
        })
        .finish();
    queue.submit([command_buffer]);
    let _fault = frame.poison_retained_byte_accounting_for_test();
    transaction
        .finish(RuntimeOperation::EffectRendering)
        .await?;
    let _cleanup = frame.finish_checked()?;
    draft.commit();
    Ok(())
}

/// Exercises a terminal device signal after a real submission and before the
/// transaction or its draft publication can commit.
#[cfg(feature = "render-window")]
pub(crate) async fn graph_terminal_loss_after_submission_for_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    signal: Arc<DeviceSignal>,
    resources: &ResourceManager,
    resource_key: ResourceCacheKey,
    publication: &mut Option<u64>,
) -> Result<()> {
    let generation = signal.next_test_generation()?;
    let transaction = GpuOperationTransaction::begin(
        device,
        Arc::clone(&signal),
        generation,
        GpuOperationStage::Render,
    );
    let mut frame = resources.begin_frame()?;
    let _lease = frame.acquire(resource_key, 16)?;
    frame.discard_on_drop();
    let draft = GpuOperationDraft::new(publication, 2);
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit graph terminal-loss harness"),
        })
        .finish();
    queue.submit([command_buffer]);
    signal.record_loss_for_test(DeviceLossReason::Destroyed);
    transaction
        .finish(RuntimeOperation::EffectRendering)
        .await?;
    let _cleanup = frame.finish_checked()?;
    draft.commit();
    Ok(())
}

/// Private test-only pause immediately after a generic transaction submits work.
#[cfg(test)]
pub(crate) struct ScopedGpuOperationPostSubmitCheckpointForTest {
    observed: Receiver<()>,
    release: Option<Arc<AtomicBool>>,
    previous: Option<GpuOperationPostSubmitControlForTest>,
}

#[cfg(test)]
#[derive(Clone)]
enum GpuOperationPostSubmitControlForTest {
    Pause(SyncSender<()>),
    Yield {
        reached: SyncSender<()>,
        released: Arc<AtomicBool>,
    },
}

#[cfg(test)]
impl ScopedGpuOperationPostSubmitCheckpointForTest {
    pub(crate) fn begin() -> Self {
        let (reached, observed) = sync_channel(1);
        let previous = ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            active.replace(Some(GpuOperationPostSubmitControlForTest::Pause(reached)))
        });
        Self {
            observed,
            release: None,
            previous,
        }
    }

    pub(crate) fn yielding() -> Self {
        let (reached, observed) = sync_channel(1);
        let released = Arc::new(AtomicBool::new(false));
        let previous = ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            active.replace(Some(GpuOperationPostSubmitControlForTest::Yield {
                reached,
                released: Arc::clone(&released),
            }))
        });
        Self {
            observed,
            release: Some(released),
            previous,
        }
    }

    pub(crate) fn wait_for_submission_for_test(&self, deadline: std::time::Duration) {
        self.observed
            .recv_timeout(deadline)
            .expect("the real generic submission did not reach the bounded post-submit checkpoint");
    }

    pub(crate) fn release_for_test(&self) {
        self.release
            .as_ref()
            .expect("only a yielding post-submit checkpoint can resume the submission")
            .store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
impl Drop for ScopedGpuOperationPostSubmitCheckpointForTest {
    fn drop(&mut self) {
        ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous.take();
        });
    }
}

#[cfg(test)]
pub(super) async fn wait_at_active_gpu_operation_post_submit_checkpoint_for_test() {
    let checkpoint =
        ACTIVE_GPU_OPERATION_POST_SUBMIT_CHECKPOINT_FOR_TEST.with(|active| active.borrow().clone());
    match checkpoint {
        Some(GpuOperationPostSubmitControlForTest::Pause(reached)) => {
            reached
                .send(())
                .expect("the generic submission test must observe the post-submit checkpoint");
            std::future::pending::<()>().await;
        }
        Some(GpuOperationPostSubmitControlForTest::Yield { reached, released }) => {
            reached
                .send(())
                .expect("the generic submission test must observe the post-submit checkpoint");
            std::future::poll_fn(|_| {
                released
                    .load(Ordering::SeqCst)
                    .then_some(())
                    .map_or(std::task::Poll::Pending, std::task::Poll::Ready)
            })
            .await;
        }
        None => {}
    }
}
