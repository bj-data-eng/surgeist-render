use super::GpuOperationTransaction;
#[cfg(test)]
use super::{
    GpuOperationSubmissionObservationForTest, record_active_gpu_operation_submission_for_test,
    wait_at_active_gpu_operation_post_submit_checkpoint_for_test,
};
use crate::{Result, RuntimeOperation};

/// Transaction-owned result of submitting one texture readback copy.
#[must_use = "the exact readback submission index must drive map completion"]
pub(crate) struct ReadbackSubmission {
    submission_index: wgpu::SubmissionIndex,
}

impl ReadbackSubmission {
    pub(crate) fn into_submission_index(self) -> wgpu::SubmissionIndex {
        self.submission_index
    }
}

/// A submitted readback copy whose WGPU error scopes are still resolving.
#[must_use = "the readback transaction scopes must resolve before mapping"]
pub(crate) struct PendingReadbackSubmission {
    submission_index: wgpu::SubmissionIndex,
    transaction: GpuOperationTransaction,
    #[cfg(test)]
    submission_observation: Option<GpuOperationSubmissionObservationForTest>,
}

impl PendingReadbackSubmission {
    pub(crate) fn submission_index(&self) -> wgpu::SubmissionIndex {
        self.submission_index.clone()
    }

    pub(crate) async fn finish(self, operation: RuntimeOperation) -> Result<ReadbackSubmission> {
        #[cfg(test)]
        wait_at_active_gpu_operation_post_submit_checkpoint_for_test().await;

        let result = self.transaction.finish(operation).await;
        #[cfg(test)]
        if let Some(observation) = self.submission_observation {
            observation.record_scope_resolution(true);
        }
        result.map(|()| ReadbackSubmission {
            submission_index: self.submission_index,
        })
    }
}

impl GpuOperationTransaction {
    /// Submits one texture-copy command and returns its exact queue submission index.
    pub(crate) fn submit_readback(
        self,
        queue: &wgpu::Queue,
        command_buffer: wgpu::CommandBuffer,
    ) -> PendingReadbackSubmission {
        let submission_index = queue.submit([command_buffer]);
        #[cfg(test)]
        let submission_observation = record_active_gpu_operation_submission_for_test(
            self.lease.generation(),
            self.lease.active_generation_for_test(),
            Some(submission_index.clone()),
        );

        PendingReadbackSubmission {
            submission_index,
            transaction: self,
            #[cfg(test)]
            submission_observation,
        }
    }
}
