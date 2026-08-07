use super::{
    Renderer,
    backend::{DeviceSlotIdentity, SurfaceFrameCommit},
    gpu_transaction::GpuOperationDraft,
};
use crate::{ImageId, Parameters, Result, Stats, Surface};
use std::{collections::HashSet, time::Instant};

pub(super) struct RenderPublication {
    frame: SurfaceFrameCommit,
    stats: Stats,
    uploaded_images: HashSet<ImageId>,
    parameters: Parameters,
}

impl RenderPublication {
    pub(super) fn new(
        frame: SurfaceFrameCommit,
        stats: Stats,
        uploaded_images: HashSet<ImageId>,
        parameters: Parameters,
    ) -> Self {
        Self {
            frame,
            stats,
            uploaded_images,
            parameters,
        }
    }

    fn commit(self, renderer: &mut Renderer, surface: &mut Surface) -> Stats {
        self.frame.commit(surface);
        renderer.stats = self.stats;
        renderer.uploaded_images = self.uploaded_images;
        surface.last_parameters = Some(self.parameters);
        self.stats
    }
}

impl Renderer {
    pub(super) fn publish_clean_render_frame(
        &mut self,
        surface: &mut Surface,
        device_identity: DeviceSlotIdentity,
        mut publication: RenderPublication,
        frame_start: Instant,
    ) -> Result<Stats> {
        publication
            .frame
            .apply_stats_observation(&mut publication.stats);
        let timings = publication.frame.timings();
        publication.stats.render_time = timings.render_time;
        publication.stats.present_time = timings.present_time;
        publication.stats.frame_time = frame_start.elapsed();
        let mut published = None;
        GpuOperationDraft::new(&mut published, publication).commit();
        let publication =
            published.expect("a clean GPU transaction must commit its staged public state");
        #[cfg(test)]
        {
            let Some(publication_signal) = self
                .backend
                .as_mut()
                .and_then(|backend| backend.device_signal_for_test(device_identity))
            else {
                panic!("a clean frame must retain its device signal until publication");
            };
            inject_final_publication_loss_for_test(&publication_signal);
        }
        let stats = publication.commit(self, surface);
        if let Some(backend) = self.backend.as_mut() {
            backend.observe_device_terminal(device_identity);
        }
        Ok(stats)
    }

    #[cfg(test)]
    pub(crate) fn uploaded_images_for_test(&self) -> HashSet<ImageId> {
        self.uploaded_images.clone()
    }
}

/// Private control that injects loss after a clean transaction and before publication.
#[cfg(test)]
pub(crate) struct ScopedFinalPublicationLossForTest {
    previous: bool,
}

#[cfg(test)]
thread_local! {
    static ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST: std::cell::RefCell<bool> =
        const { std::cell::RefCell::new(false) };
}

#[cfg(test)]
impl ScopedFinalPublicationLossForTest {
    pub(crate) fn after_transaction_completion() -> Self {
        let previous = ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| active.replace(true));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for ScopedFinalPublicationLossForTest {
    fn drop(&mut self) {
        ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| {
            *active.borrow_mut() = self.previous;
        });
    }
}

#[cfg(test)]
pub(super) fn inject_final_publication_loss_for_test(signal: &super::backend::DeviceSignal) {
    if ACTIVE_FINAL_PUBLICATION_LOSS_FOR_TEST.with(|active| *active.borrow()) {
        signal.record_loss_for_test(crate::DeviceLossReason::Unknown);
    }
}
