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
        let stats = publication.commit(self, surface);
        if let Some(backend) = self.backend.as_mut() {
            backend.observe_device_terminal(device_identity);
        }
        Ok(stats)
    }
}
