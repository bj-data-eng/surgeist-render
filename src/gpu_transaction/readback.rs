use super::GpuOperationTransaction;
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
}

impl PendingReadbackSubmission {
    pub(crate) fn submission_index(&self) -> wgpu::SubmissionIndex {
        self.submission_index.clone()
    }

    pub(crate) async fn finish(self, operation: RuntimeOperation) -> Result<ReadbackSubmission> {
        self.transaction
            .finish(operation)
            .await
            .map(|()| ReadbackSubmission {
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

        PendingReadbackSubmission {
            submission_index,
            transaction: self,
        }
    }
}
