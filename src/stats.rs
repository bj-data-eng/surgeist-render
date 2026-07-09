use super::{
    Image, ImageId, Paint,
    command::{RenderCommand, RenderPaint},
    paint::PaintKind,
    scene::Command,
};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub frame_time: Duration,
    pub encode_time: Duration,
    pub render_time: Duration,
    pub present_time: Duration,
    pub commands: usize,
    pub fills: usize,
    pub strokes: usize,
    pub shadows: usize,
    pub images: usize,
    pub glyphs: usize,
    pub layers: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub uploaded_bytes: u64,
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
