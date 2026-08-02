use super::{
    Image, ImageId, Paint,
    command::{RenderCommand, RenderPaint},
    paint::PaintKind,
    pass::EncodedGpuGraphActivity,
    resource::{FrameCleanup, WorkingFormat},
    scene::Command,
};
use std::time::Duration;

/// GPU execution route used by one successfully published frame.
///
/// [`Self::DirectVello`] is the single-pass effect-free route;
/// [`Self::GpuGraph`] owns the closed supported graph subset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderRoute {
    /// One transaction-owned internal Vello raster pass.
    DirectVello,
    /// The closed crate-owned WGPU image/composite graph.
    GpuGraph,
}

/// Working texture precision selected for a successfully published GPU graph frame.
///
/// High maps to `Rgba16Float`; Reduced maps to `Rgba8Unorm`. DirectVello
/// frames do not select an effect precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectPrecision {
    /// High-precision `Rgba16Float` working textures.
    High,
    /// Explicit reduced-precision `Rgba8Unorm` working textures.
    Reduced,
}

/// Telemetry for the renderer's last successful complete frame publication.
///
/// Failed or canceled attempts do not replace this value. Counts describe work
/// owned by that frame, durations use [`Duration`], and resource fields measure
/// bytes. [`Default`] represents no successful publication yet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Execution route, or `None` before the first successful frame.
    pub route: Option<RenderRoute>,
    /// Graph working precision, or `None` for DirectVello/no successful frame.
    pub effect_precision: Option<EffectPrecision>,
    /// Number of internal Vello raster passes encoded for the frame.
    pub vello_passes: usize,
    /// Number of crate-owned GPU image-processing passes encoded for the frame.
    pub image_passes: usize,
    /// Number of crate-owned GPU composition passes encoded for the frame.
    pub composite_passes: usize,
    /// Number of GPU texture-copy operations encoded for the frame.
    pub copy_operations: usize,
    /// Number of custom graph-to-surface presentation passes.
    pub custom_present_passes: usize,
    /// Number of new effect textures allocated while preparing the frame.
    pub effect_texture_allocations: usize,
    /// Number of compatible retained effect textures reused by the frame.
    pub effect_texture_reuses: usize,
    /// Idle effect-resource bytes retained after successful frame cleanup.
    pub retained_effect_bytes: u64,
    /// Wall-clock duration of the complete successful frame operation.
    pub frame_time: Duration,
    /// Duration spent validating, normalizing, and encoding frame inputs.
    pub encode_time: Duration,
    /// Duration spent preparing and submitting GPU render work.
    pub render_time: Duration,
    /// Duration spent acquiring and presenting a host surface frame.
    pub present_time: Duration,
    /// Number of normalized render commands visited for telemetry.
    pub commands: usize,
    /// Number of normalized fill commands.
    pub fills: usize,
    /// Number of normalized stroke commands.
    pub strokes: usize,
    /// Number of normalized shadow instances.
    pub shadows: usize,
    /// Number of normalized image draws.
    pub images: usize,
    /// Number of normalized glyph instances.
    pub glyphs: usize,
    /// Number of normalized layer commands.
    pub layers: usize,
    /// Number of image identities already observed by this renderer.
    pub cache_hits: usize,
    /// Number of first-observed image identities for this renderer.
    pub cache_misses: usize,
    /// Straight-alpha source-image bytes first observed by this renderer.
    pub uploaded_bytes: u64,
}

/// One complete graph-frame observation, frozen only after resource cleanup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuGraphStatsObservation {
    effect_precision: EffectPrecision,
    activity: EncodedGpuGraphActivity,
    effect_texture_allocations: usize,
    effect_texture_reuses: usize,
    retained_effect_bytes: u64,
}

impl GpuGraphStatsObservation {
    pub(crate) fn after_cleanup(
        working_format: WorkingFormat,
        activity: EncodedGpuGraphActivity,
        cleanup: &FrameCleanup,
    ) -> Self {
        let acquisitions = cleanup.acquisitions();
        Self {
            effect_precision: match working_format {
                WorkingFormat::HighPrecision => EffectPrecision::High,
                WorkingFormat::ReducedPrecision => EffectPrecision::Reduced,
            },
            activity,
            effect_texture_allocations: acquisitions.allocations(),
            effect_texture_reuses: acquisitions.reuses(),
            retained_effect_bytes: cleanup.retained_byte_len(),
        }
    }

    pub(crate) fn apply_to(self, stats: &mut Stats) {
        stats.route = Some(RenderRoute::GpuGraph);
        stats.effect_precision = Some(self.effect_precision);
        stats.vello_passes = self.activity.vello_passes();
        stats.image_passes = self.activity.image_passes();
        stats.composite_passes = self.activity.composite_passes();
        stats.copy_operations = self.activity.copy_operations();
        stats.custom_present_passes = self.activity.custom_present_passes();
        stats.effect_texture_allocations = self.effect_texture_allocations;
        stats.effect_texture_reuses = self.effect_texture_reuses;
        stats.retained_effect_bytes = self.retained_effect_bytes;
    }
}

pub(crate) fn collect_stats(
    commands: &[Command],
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    for command in commands {
        stats.commands = stats.commands.saturating_add(1);
        match command {
            Command::Fill { paint, .. } => {
                stats.fills = stats.fills.saturating_add(1);
                collect_paint_stats(paint, stats, uploaded_images);
            }
            Command::Stroke { paint, .. } => {
                stats.strokes = stats.strokes.saturating_add(1);
                collect_paint_stats(paint, stats, uploaded_images);
            }
            Command::Shadow { shadow, .. } => {
                stats.shadows = stats.shadows.saturating_add(1);
                collect_paint_stats(shadow.paint(), stats, uploaded_images);
            }
            Command::Image { image, .. } => {
                collect_image_stats(image, stats, uploaded_images);
            }
            Command::TextRun { glyphs, .. } => {
                stats.glyphs = stats.glyphs.saturating_add(glyphs.len());
            }
            Command::TextShadowRun {
                glyphs, shadows, ..
            } => {
                stats.glyphs = stats.glyphs.saturating_add(glyphs.len());
                stats.shadows = stats.shadows.saturating_add(shadows.len());
                for shadow in shadows.shadows() {
                    collect_paint_stats(shadow.paint(), stats, uploaded_images);
                }
            }
            Command::Layer { children, .. } => {
                stats.layers = stats.layers.saturating_add(1);
                collect_stats(children, stats, uploaded_images);
            }
        }
    }
}

fn collect_paint_stats(
    paint: &Paint,
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    if let PaintKind::Image(image) = paint.kind() {
        collect_image_stats(image, stats, uploaded_images);
    }
}

pub(crate) fn collect_render_stats(
    commands: &[RenderCommand],
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    for command in commands {
        stats.commands = stats.commands.saturating_add(1);
        match command {
            RenderCommand::Fill { paint, .. } => {
                stats.fills = stats.fills.saturating_add(1);
                collect_render_paint_stats(paint, stats, uploaded_images);
            }
            RenderCommand::Stroke { paint, .. } => {
                stats.strokes = stats.strokes.saturating_add(1);
                collect_render_paint_stats(paint, stats, uploaded_images);
            }
            RenderCommand::Shadow { .. } => {
                stats.shadows = stats.shadows.saturating_add(1);
            }
            RenderCommand::Image { image, .. } => {
                collect_image_stats(image, stats, uploaded_images);
            }
            RenderCommand::TextRun { glyphs, .. } => {
                stats.glyphs = stats.glyphs.saturating_add(glyphs.len());
            }
            RenderCommand::Layer { children, .. } => {
                stats.layers = stats.layers.saturating_add(1);
                collect_render_stats(children, stats, uploaded_images);
            }
        }
    }
}

fn collect_render_paint_stats(
    paint: &RenderPaint,
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    if let RenderPaint::Image(image) = paint {
        collect_image_stats(image, stats, uploaded_images);
    }
}

fn collect_image_stats(
    image: &Image,
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    stats.images = stats.images.saturating_add(1);
    if uploaded_images.insert(image.id()) {
        stats.cache_misses = stats.cache_misses.saturating_add(1);
        stats.uploaded_bytes = stats
            .uploaded_bytes
            .saturating_add(u64::try_from(image.bytes.len()).unwrap_or(u64::MAX));
    } else {
        stats.cache_hits = stats.cache_hits.saturating_add(1);
    }
}
