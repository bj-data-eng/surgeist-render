use super::GpuOperationTransaction;
#[cfg(test)]
use super::{
    ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST, AfterInternalVelloSubmitCheckpointForTest,
    InternalVelloSubmissionObservationForTest, record_active_internal_vello_submission_for_test,
};
use crate::{
    Result, RuntimeOperation,
    vello_engine::{DirectVelloLogicalPass, PendingVelloResourceCommit},
};

#[must_use = "internal Vello command buffers must remain owned by their GPU transaction"]
pub(crate) struct InternalVelloPayload {
    command_buffer: wgpu::CommandBuffer,
    resources: PendingVelloResourceCommit,
    logical_pass: DirectVelloLogicalPass,
    #[cfg(test)]
    submission_observation: Option<InternalVelloSubmissionObservationForTest>,
    #[cfg(test)]
    after_submit_checkpoint: Option<AfterInternalVelloSubmitCheckpointForTest>,
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
            #[cfg(test)]
            submission_observation: None,
            #[cfg(test)]
            after_submit_checkpoint: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn observed_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
        submission_observation: InternalVelloSubmissionObservationForTest,
    ) -> Self {
        let mut payload = Self::new(command_buffer, resources, logical_pass);
        payload.submission_observation = Some(submission_observation);
        payload
    }

    #[cfg(test)]
    pub(crate) fn paused_after_submit_for_test(
        command_buffer: wgpu::CommandBuffer,
        resources: PendingVelloResourceCommit,
        logical_pass: DirectVelloLogicalPass,
        checkpoint: AfterInternalVelloSubmitCheckpointForTest,
    ) -> Self {
        Self {
            command_buffer,
            resources,
            logical_pass,
            submission_observation: None,
            after_submit_checkpoint: Some(checkpoint),
        }
    }
}

impl GpuOperationTransaction {
    #[cfg(test)]
    pub(crate) async fn finish_vello_resources_without_submission_for_test(
        self,
        resources: PendingVelloResourceCommit,
        operation: RuntimeOperation,
    ) -> Result<()> {
        self.finish(operation).await?;
        resources
            .into_accounting_ready()?
            .commit(VelloResourceCommitProof::new())?;
        Ok(())
    }

    pub(crate) async fn submit_internal_vello(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        payload: InternalVelloPayload,
        operation: RuntimeOperation,
    ) -> Result<()> {
        #[cfg(not(test))]
        let _ = device;
        let InternalVelloPayload {
            command_buffer,
            resources,
            logical_pass,
            #[cfg(test)]
            submission_observation,
            #[cfg(test)]
            after_submit_checkpoint,
        } = payload;
        queue.submit([command_buffer]);
        #[cfg(not(test))]
        let _ = logical_pass;
        #[cfg(test)]
        let active_generation = self.lease.active_generation_for_test();
        #[cfg(test)]
        let allocation_summary = resources.allocation_summary_for_test();
        #[cfg(test)]
        if let Some(observation) = submission_observation {
            observation.record_payload_submission(
                self.lease.generation(),
                active_generation,
                &logical_pass,
                allocation_summary.clone(),
            );
        }
        #[cfg(test)]
        record_active_internal_vello_submission_for_test(
            self.lease.generation(),
            active_generation,
            &logical_pass,
            allocation_summary,
        );
        #[cfg(test)]
        if let Some(checkpoint) = after_submit_checkpoint {
            checkpoint.wait().await;
        }
        #[cfg(test)]
        if let Some(control) = ACTIVE_INTERNAL_VELLO_POST_SUBMIT_CONTROL_FOR_TEST
            .with(|active| active.borrow().clone())
        {
            control.apply(device, &resources).await;
        }
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
