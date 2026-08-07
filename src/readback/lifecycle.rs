use super::{BackendErrorCode, Error, ImageBuffer, PhysicalSize, Result, layout::ReadbackLayout};
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

pub(super) enum ReadbackPhase<I> {
    Allocated,
    CopySubmitted { submission_index: I },
    MapPending,
    Mapped,
    PublishedBytes,
    Failed,
    Canceled,
}

pub(super) struct ReadbackLifecycle<I> {
    pub(super) phase: ReadbackPhase<I>,
}

impl<I> ReadbackLifecycle<I> {
    pub(super) const fn allocated() -> Self {
        Self {
            phase: ReadbackPhase::Allocated,
        }
    }

    pub(super) fn copy_submitted(&mut self, submission_index: I) {
        assert!(
            matches!(&self.phase, ReadbackPhase::Allocated),
            "readback copy submission must follow allocation"
        );
        self.phase = ReadbackPhase::CopySubmitted { submission_index };
    }

    pub(super) fn map_pending(&mut self) {
        let previous = std::mem::replace(&mut self.phase, ReadbackPhase::MapPending);
        let ReadbackPhase::CopySubmitted { submission_index } = previous else {
            self.phase = previous;
            panic!("readback mapping must follow copy submission");
        };
        drop(submission_index);
    }

    pub(super) fn mapped(&mut self) {
        assert!(
            matches!(&self.phase, ReadbackPhase::MapPending),
            "mapped readback bytes require callback success"
        );
        self.phase = ReadbackPhase::Mapped;
    }

    pub(super) fn published(&mut self) {
        assert!(
            matches!(&self.phase, ReadbackPhase::Mapped),
            "readback bytes may publish only from the mapped phase"
        );
        self.phase = ReadbackPhase::PublishedBytes;
    }

    pub(super) fn fail(&mut self) -> bool {
        if self.is_uncertain() {
            self.phase = ReadbackPhase::Failed;
            true
        } else {
            false
        }
    }

    pub(super) fn cancel(&mut self) -> bool {
        if self.is_uncertain() {
            self.phase = ReadbackPhase::Canceled;
            true
        } else {
            false
        }
    }

    const fn is_uncertain(&self) -> bool {
        matches!(
            &self.phase,
            ReadbackPhase::Allocated
                | ReadbackPhase::CopySubmitted { .. }
                | ReadbackPhase::MapPending
                | ReadbackPhase::Mapped
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadbackStagingDisposition {
    Idle,
    MapPending,
    MappedActive,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadbackStagingCleanupAction {
    Drop,
    UnmapThenDrop,
    None,
}

pub(super) struct ReadbackStagingMapState {
    disposition: Mutex<ReadbackStagingDisposition>,
}

impl ReadbackStagingMapState {
    pub(super) fn idle() -> Self {
        Self {
            disposition: Mutex::new(ReadbackStagingDisposition::Idle),
        }
    }

    pub(super) fn map_pending(&self) {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            *disposition,
            ReadbackStagingDisposition::Idle,
            "a readback map request requires known-idle staging"
        );
        *disposition = ReadbackStagingDisposition::MapPending;
    }

    pub(super) fn map_callback_completed(&self, succeeded: bool) {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *disposition {
            ReadbackStagingDisposition::MapPending => {
                *disposition = if succeeded {
                    ReadbackStagingDisposition::MappedActive
                } else {
                    ReadbackStagingDisposition::Idle
                };
            }
            ReadbackStagingDisposition::Released => {}
            ReadbackStagingDisposition::Idle | ReadbackStagingDisposition::MappedActive => {
                panic!("a readback map request callback may complete only once")
            }
        }
    }

    pub(super) fn assert_mapped_active(&self) {
        assert_eq!(
            *self
                .disposition
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ReadbackStagingDisposition::MappedActive,
            "map callback success must make staging actively mapped"
        );
    }

    pub(super) fn take_cleanup_action(&self) -> ReadbackStagingCleanupAction {
        let mut disposition = self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match std::mem::replace(&mut *disposition, ReadbackStagingDisposition::Released) {
            ReadbackStagingDisposition::Idle => ReadbackStagingCleanupAction::Drop,
            ReadbackStagingDisposition::MapPending | ReadbackStagingDisposition::MappedActive => {
                ReadbackStagingCleanupAction::UnmapThenDrop
            }
            ReadbackStagingDisposition::Released => ReadbackStagingCleanupAction::None,
        }
    }

    #[cfg(test)]
    pub(super) fn disposition_for_test(&self) -> ReadbackStagingDisposition {
        *self
            .disposition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

pub(super) struct ReadbackOwner {
    lifecycle: ReadbackLifecycle<wgpu::SubmissionIndex>,
    staging: Option<wgpu::Buffer>,
    staging_map: Arc<ReadbackStagingMapState>,
    pub(super) layout: ReadbackLayout,
    physical_size: PhysicalSize,
}

impl ReadbackOwner {
    pub(super) fn allocated(
        staging: wgpu::Buffer,
        layout: ReadbackLayout,
        physical_size: PhysicalSize,
    ) -> Self {
        Self {
            lifecycle: ReadbackLifecycle::allocated(),
            staging: Some(staging),
            staging_map: Arc::new(ReadbackStagingMapState::idle()),
            layout,
            physical_size,
        }
    }

    pub(super) fn staging(&self) -> &wgpu::Buffer {
        self.staging
            .as_ref()
            .expect("an uncertain readback phase must own its staging buffer")
    }

    pub(super) fn copy_submitted(&mut self, submission_index: wgpu::SubmissionIndex) {
        self.lifecycle.copy_submitted(submission_index);
    }

    pub(super) fn map_pending(&mut self) {
        self.lifecycle.map_pending();
        self.staging_map.map_pending();
    }

    pub(super) fn mapped(&mut self) {
        self.staging_map.assert_mapped_active();
        self.lifecycle.mapped();
    }

    pub(super) fn staging_map(&self) -> Arc<ReadbackStagingMapState> {
        Arc::clone(&self.staging_map)
    }

    pub(super) fn mapped_range(&self) -> Range<wgpu::BufferAddress> {
        self.layout.mapped_range()
    }

    pub(super) fn fail(&mut self) {
        if self.lifecycle.fail() {
            self.release_staging();
        }
    }

    pub(super) fn cancel(&mut self) {
        if self.lifecycle.cancel() {
            self.release_staging();
        }
    }

    pub(super) fn publish_mapped(mut self, rgba: Vec<u8>) -> Result<ImageBuffer> {
        self.release_staging();
        match ImageBuffer::try_new(self.physical_size, rgba) {
            Ok(image) => {
                self.lifecycle.published();
                Ok(image)
            }
            Err(source) => {
                self.lifecycle.fail();
                Err(Error::new(
                    BackendErrorCode::ReadbackFailed,
                    "decoded readback bytes did not form a valid RGBA8 image",
                )
                .with_source(source))
            }
        }
    }

    fn release_staging(&mut self) {
        let action = self.staging_map.take_cleanup_action();
        let Some(staging) = self.staging.take() else {
            assert_eq!(
                action,
                ReadbackStagingCleanupAction::None,
                "readback staging cleanup action must be consumed with ownership"
            );
            return;
        };
        match action {
            ReadbackStagingCleanupAction::Drop => drop(staging),
            ReadbackStagingCleanupAction::UnmapThenDrop => {
                staging.unmap();
                drop(staging);
            }
            ReadbackStagingCleanupAction::None => {
                unreachable!("owned readback staging cannot already be released")
            }
        }
    }
}

impl Drop for ReadbackOwner {
    fn drop(&mut self) {
        self.cancel();
    }
}
