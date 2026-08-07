use super::GpuOperationTransaction;
use crate::{
    Result, RuntimeOperation,
    vello_engine::{DirectVelloLogicalPass, PendingVelloResourceCommit},
};

#[must_use = "internal Vello command buffers must remain owned by their GPU transaction"]
pub(crate) struct InternalVelloPayload {
    command_buffer: wgpu::CommandBuffer,
    resources: PendingVelloResourceCommit,
    logical_pass: DirectVelloLogicalPass,
}

/// Proof that an internal Vello submission has reached its clean terminal boundary.
pub(crate) struct VelloResourceCommitProof {
    _private: (),
}

impl VelloResourceCommitProof {
    pub(super) const fn new() -> Self {
        Self { _private: () }
    }
}

impl InternalVelloPayload {
    pub(crate) fn new(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            logical_pass,
        }
    }

    #[cfg(test)]
    pub(super) fn into_parts_for_test(
        self,
    ) -> (
        wgpu::CommandBuffer,
        PendingVelloResourceCommit,
        DirectVelloLogicalPass,
    ) {
        (self.command_buffer, self.resources, self.logical_pass)
    }
}

impl GpuOperationTransaction {
    pub(crate) async fn submit_internal_vello(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        payload: InternalVelloPayload,
        operation: RuntimeOperation,
    ) -> Result<()> {
        let _ = device;
        let InternalVelloPayload {
            command_buffer,
            resources,
            logical_pass,
        } = payload;
        queue.submit([command_buffer]);
        let _ = logical_pass;
        match self.finish(operation).await {
            Ok(()) => {
                resources
                    .into_accounting_ready()?
                    .commit(VelloResourceCommitProof::new())?;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
