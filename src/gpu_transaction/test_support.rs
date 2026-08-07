use std::sync::Arc;

use super::{
    GpuOperationDraft, GpuOperationStage, GpuOperationTransaction, InternalVelloPayload,
    VelloResourceCommitProof,
};
#[cfg(feature = "render-window")]
use crate::DeviceLossReason;
use crate::{
    Result, RuntimeOperation,
    backend::DeviceSignal,
    resource::{ResourceCacheKey, ResourceManager},
    vello_engine::{PendingVelloResourceCommit, VelloResourceAllocationSummaryForTest},
};

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

/// Explicit test-owned facts around one real generic transaction submission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuOperationSubmissionObservationForTest {
    queue_submission_count: usize,
    transaction_generation: u64,
    active_generation: Option<u64>,
    scopes_resolved: bool,
}

impl GpuOperationSubmissionObservationForTest {
    pub(crate) const fn queue_submission_count_for_test(&self) -> usize {
        self.queue_submission_count
    }

    pub(crate) const fn transaction_generation_for_test(&self) -> Option<u64> {
        Some(self.transaction_generation)
    }

    pub(crate) const fn active_generation_for_test(&self) -> Option<u64> {
        self.active_generation
    }

    pub(crate) const fn scopes_resolved_for_test(&self) -> bool {
        self.scopes_resolved
    }
}

/// Submits one explicit command buffer through the real transaction and records
/// only facts owned by the test at that stage boundary.
pub(crate) async fn submit_command_buffer_observed_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
    operation: RuntimeOperation,
) -> Result<GpuOperationSubmissionObservationForTest> {
    let transaction_generation = transaction.lease.generation;
    let active_generation = transaction.lease.signal.active_generation_for_test();
    queue.submit([command_buffer]);
    transaction.finish(operation).await?;
    Ok(GpuOperationSubmissionObservationForTest {
        queue_submission_count: 1,
        transaction_generation,
        active_generation,
        scopes_resolved: true,
    })
}

pub(crate) async fn submit_command_buffer_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
    operation: RuntimeOperation,
) -> Result<()> {
    submit_command_buffer_observed_for_test(transaction, queue, command_buffer, operation)
        .await
        .map(|_| ())
}

/// Holds a real generic transaction after queue submission. Dropping the future
/// releases the transaction lease without reporting a successful completion.
pub(crate) async fn hold_command_buffer_after_submit_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
) -> Result<()> {
    queue.submit([command_buffer]);
    let _transaction = transaction;
    std::future::pending::<()>().await;
    Ok(())
}

/// Records a real uncaptured fault only after the explicit queue submission and
/// resolves it through the real transaction error path.
pub(crate) async fn fault_command_buffer_after_submit_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
    signal: &DeviceSignal,
    kind: crate::GpuFaultKind,
    message: &str,
    operation: RuntimeOperation,
) -> Result<()> {
    queue.submit([command_buffer]);
    signal.record_uncaptured_fault_for_test(kind, message);
    transaction.finish(operation).await
}

/// Test-owned facts around one real readback submission and scope transition.
#[derive(Clone, Debug)]
pub(crate) struct ReadbackSubmissionObservationForTest {
    queue_submission_count: usize,
    transaction_generation: u64,
    active_generation: Option<u64>,
    submission_index: wgpu::SubmissionIndex,
    scopes_resolved: bool,
}

impl ReadbackSubmissionObservationForTest {
    pub(crate) const fn queue_submission_count_for_test(&self) -> usize {
        self.queue_submission_count
    }

    pub(crate) const fn transaction_generation_for_test(&self) -> Option<u64> {
        Some(self.transaction_generation)
    }

    pub(crate) const fn active_generation_for_test(&self) -> Option<u64> {
        self.active_generation
    }

    pub(crate) fn submission_index_for_test(&self) -> wgpu::SubmissionIndex {
        self.submission_index.clone()
    }

    pub(crate) const fn scopes_resolved_for_test(&self) -> bool {
        self.scopes_resolved
    }
}

pub(crate) async fn submit_readback_observed_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
    operation: RuntimeOperation,
) -> Result<ReadbackSubmissionObservationForTest> {
    let transaction_generation = transaction.lease.generation;
    let active_generation = transaction.lease.signal.active_generation_for_test();
    let pending = transaction.submit_readback(queue, command_buffer);
    let completed = pending.finish(operation).await?;
    let submission_index = completed.into_submission_index();
    Ok(ReadbackSubmissionObservationForTest {
        queue_submission_count: 1,
        transaction_generation,
        active_generation,
        submission_index,
        scopes_resolved: true,
    })
}

/// Explicit test-owned facts captured from the Vello payload before its real
/// transaction submission consumes it.
#[derive(Clone, Debug)]
pub(crate) struct InternalVelloSubmissionObservationForTest {
    queue_submission_count: usize,
    transaction_generation: u64,
    active_generation: Option<u64>,
    payload_raster_pass_count: usize,
    allocation_summary: VelloResourceAllocationSummaryForTest,
}

pub(crate) enum InternalVelloSubmissionActionForTest<'a> {
    Observe,
    ScopeFailure(&'a mut Option<u64>),
    AccountingFailure,
}

pub(crate) enum InternalVelloSubmissionOutcomeForTest {
    Observed(InternalVelloSubmissionObservationForTest),
    Completed,
}

impl InternalVelloSubmissionObservationForTest {
    pub(crate) const fn queue_submission_count_for_test(&self) -> usize {
        self.queue_submission_count
    }

    pub(crate) const fn transaction_generation_for_test(&self) -> Option<u64> {
        Some(self.transaction_generation)
    }

    pub(crate) const fn active_generation_for_test(&self) -> Option<u64> {
        self.active_generation
    }

    pub(crate) const fn payload_raster_pass_count_for_test(&self) -> usize {
        self.payload_raster_pass_count
    }

    pub(crate) fn allocation_summary_for_test(
        &self,
    ) -> Option<VelloResourceAllocationSummaryForTest> {
        Some(self.allocation_summary.clone())
    }
}

pub(crate) async fn submit_internal_vello_observed_for_test(
    transaction: GpuOperationTransaction,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    payload: InternalVelloPayload,
    operation: RuntimeOperation,
) -> Result<InternalVelloSubmissionObservationForTest> {
    let transaction_generation = transaction.lease.generation;
    let active_generation = transaction.lease.signal.active_generation_for_test();
    let (command_buffer, resources, logical_pass) = payload.into_parts_for_test();
    let payload_raster_pass_count = logical_pass.cardinality_for_test();
    let allocation_summary = resources.allocation_summary_for_test();
    let payload = InternalVelloPayload::new(command_buffer, resources, logical_pass);
    transaction
        .submit_internal_vello(device, queue, payload, operation)
        .await?;
    Ok(InternalVelloSubmissionObservationForTest {
        queue_submission_count: 1,
        transaction_generation,
        active_generation,
        payload_raster_pass_count,
        allocation_summary,
    })
}

/// Resolves a real transaction and commits prepared Vello resources without a
/// queue submission for tests that isolate resource-commit accounting.
pub(crate) async fn finish_vello_resources_without_submission_for_test(
    transaction: GpuOperationTransaction,
    resources: PendingVelloResourceCommit,
    operation: RuntimeOperation,
) -> Result<()> {
    transaction.finish(operation).await?;
    resources
        .into_accounting_ready()?
        .commit(VelloResourceCommitProof::new())?;
    Ok(())
}

/// Submits a real Vello payload, injects a scoped validation error at the
/// explicit post-submit boundary, and commits neither resources nor draft.
pub(crate) async fn vello_scope_failure_after_submission_for_test(
    transaction: GpuOperationTransaction,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    payload: InternalVelloPayload,
    operation: RuntimeOperation,
    publication: &mut Option<u64>,
) -> Result<()> {
    let (command_buffer, resources, _logical_pass) = payload.into_parts_for_test();
    let draft = GpuOperationDraft::new(publication, 2);
    queue.submit([command_buffer]);
    let _ = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Surgeist explicit Vello transaction validation failure"),
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
    transaction.finish(operation).await?;
    resources
        .into_accounting_ready()?
        .commit(VelloResourceCommitProof::new())?;
    draft.commit();
    Ok(())
}

/// Submits a real Vello payload and poisons its resource accounting only at the
/// explicit post-submit boundary.
pub(crate) async fn vello_accounting_failure_after_submission_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    payload: InternalVelloPayload,
    operation: RuntimeOperation,
) -> Result<()> {
    let (command_buffer, resources, _logical_pass) = payload.into_parts_for_test();
    queue.submit([command_buffer]);
    let _fault = resources.poison_retained_byte_accounting_for_test();
    transaction.finish(operation).await?;
    resources
        .into_accounting_ready()?
        .commit(VelloResourceCommitProof::new())?;
    Ok(())
}

/// Holds a real Vello submission, its resources, transaction, and draft at the
/// explicit cancellation boundary. Dropping the future performs normal cleanup.
pub(crate) async fn hold_internal_vello_after_submit_for_test(
    transaction: GpuOperationTransaction,
    queue: &wgpu::Queue,
    payload: InternalVelloPayload,
    publication: &mut Option<u64>,
) -> Result<()> {
    let (command_buffer, resources, logical_pass) = payload.into_parts_for_test();
    queue.submit([command_buffer]);
    let _transaction = transaction;
    let _resources = resources;
    let _logical_pass = logical_pass;
    let _draft = GpuOperationDraft::new(publication, 2);
    std::future::pending::<()>().await;
    Ok(())
}
