// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::{BrushRef, Fill, ImageBrushRef};
use vello_encoding::{
    DrawBeginClip, Encoding, Glyph, GlyphRun, Patch, Transform as VelloTransform,
};

use crate::{BackendErrorCode, Error, Result, TextRun, encode::glyph_paint_brush};

use super::glyph::{
    BitmapEncoding, SelectedGlyphRepresentation, ValidatedGlyphRun, preflight_selected_glyphs,
};
use super::raster::{self, PreparedVelloPass, RasterParameters};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VelloRasterScenario {
    Base,
    LargePath,
    Clip,
    LargePathAndClip,
}

#[derive(Default)]
pub(crate) struct VelloScene {
    encoding: Encoding,
}

impl VelloScene {
    pub(crate) fn fill<'a>(
        &mut self,
        style: Fill,
        transform: kurbo::Affine,
        brush: impl Into<BrushRef<'a>>,
        brush_transform: Option<kurbo::Affine>,
        shape: &impl kurbo::Shape,
    ) {
        self.encoding
            .encode_transform(VelloTransform::from_kurbo(&transform));
        self.encoding.encode_fill_style(style);
        if self.encoding.encode_shape(shape, true) {
            if let Some(brush_transform) = brush_transform
                && self
                    .encoding
                    .encode_transform(VelloTransform::from_kurbo(&(transform * brush_transform)))
            {
                self.encoding.swap_last_path_tags();
            }
            self.encoding.encode_brush(brush, 1.0);
        }
    }

    pub(crate) fn stroke<'a>(
        &mut self,
        style: &kurbo::Stroke,
        transform: kurbo::Affine,
        brush: impl Into<BrushRef<'a>>,
        brush_transform: Option<kurbo::Affine>,
        shape: &impl kurbo::Shape,
    ) {
        if style.width == 0.0 {
            return;
        }
        self.encoding
            .encode_transform(VelloTransform::from_kurbo(&transform));
        let encoded_stroke = self.encoding.encode_stroke_style(style);
        debug_assert!(encoded_stroke, "non-zero strokes must encode");
        let encoded_shape = if style.dash_pattern.is_empty() {
            self.encoding.encode_shape(shape, false)
        } else {
            let dashed = kurbo::dash(
                shape.path_elements(0.01),
                style.dash_offset,
                &style.dash_pattern,
            )
            .collect::<Vec<_>>();
            self.encoding
                .encode_path_elements(dashed.into_iter(), false)
        };
        if encoded_shape {
            if let Some(brush_transform) = brush_transform
                && self
                    .encoding
                    .encode_transform(VelloTransform::from_kurbo(&(transform * brush_transform)))
            {
                self.encoding.swap_last_path_tags();
            }
            self.encoding.encode_brush(brush, 1.0);
        }
    }

    pub(crate) fn draw_image<'a>(
        &mut self,
        image: impl Into<ImageBrushRef<'a>>,
        transform: kurbo::Affine,
    ) {
        let image = image.into();
        let rect = kurbo::Rect::new(
            0.0,
            0.0,
            f64::from(image.image.width),
            f64::from(image.image.height),
        );
        self.fill(Fill::NonZero, transform, image, None, &rect);
    }

    pub(crate) fn push_layer(
        &mut self,
        fill: Fill,
        blend: peniko::BlendMode,
        alpha: f32,
        transform: kurbo::Affine,
        clip: &impl kurbo::Shape,
    ) {
        self.push_layer_inner(
            DrawBeginClip::new(blend, alpha.clamp(0.0, 1.0)),
            fill,
            transform,
            clip,
        );
    }

    pub(crate) fn push_clip_layer(
        &mut self,
        fill: Fill,
        transform: kurbo::Affine,
        clip: &impl kurbo::Shape,
    ) {
        self.push_layer_inner(DrawBeginClip::clip(), fill, transform, clip);
    }

    pub(crate) fn pop_layer(&mut self) {
        self.encoding.encode_end_clip();
    }

    pub(crate) fn draw_blurred_rounded_rect(
        &mut self,
        transform: kurbo::Affine,
        rect: kurbo::Rect,
        brush: peniko::Color,
        radius: f64,
        std_dev: f64,
    ) {
        let kernel_size = 2.5 * std_dev;
        self.draw_blurred_rounded_rect_in(
            &rect.inflate(kernel_size, kernel_size),
            transform,
            rect,
            brush,
            radius,
            std_dev,
        );
    }

    pub(crate) fn draw_blurred_rounded_rect_in(
        &mut self,
        shape: &impl kurbo::Shape,
        transform: kurbo::Affine,
        rect: kurbo::Rect,
        brush: peniko::Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.encoding
            .encode_transform(VelloTransform::from_kurbo(&transform));
        self.encoding.encode_fill_style(Fill::NonZero);
        if self.encoding.encode_shape(shape, true) {
            let brush_transform =
                VelloTransform::from_kurbo(&(transform.pre_translate(rect.center().to_vec2())));
            if self.encoding.encode_transform(brush_transform) {
                self.encoding.swap_last_path_tags();
            }
            self.encoding.encode_blurred_rounded_rect(
                brush,
                rect.width() as f32,
                rect.height() as f32,
                radius as f32,
                std_dev as f32,
            );
        }
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T2 private text encoding entry point is intentionally staged for T7 cutover."
        )
    )]
    pub(crate) fn encode_text_run(&mut self, run: &TextRun<'_>) -> Result<()> {
        let validated = preflight_selected_glyphs(run)?;
        self.append_validated_glyph_run(validated, kurbo::Affine::IDENTITY)
    }

    pub(crate) fn encode_text_run_with_transform(
        &mut self,
        run: &TextRun<'_>,
        transform: kurbo::Affine,
    ) -> Result<()> {
        let validated = preflight_selected_glyphs(run)?;
        self.append_validated_glyph_run(validated, transform)
    }

    pub(crate) fn prepare_raster(&self, parameters: RasterParameters) -> Result<PreparedVelloPass> {
        raster::prepare(&self.encoding, parameters)
    }

    fn append_validated_glyph_run(
        &mut self,
        validated: ValidatedGlyphRun<'_>,
        transform: kurbo::Affine,
    ) -> Result<()> {
        if let Some(representation) = validated
            .representations()
            .iter()
            .find(|representation| !matches!(representation, SelectedGlyphRepresentation::Outline))
        {
            let context = match representation {
                SelectedGlyphRepresentation::Outline => "outline glyph",
                SelectedGlyphRepresentation::Colr => "COLR glyph",
                SelectedGlyphRepresentation::Bitmap(BitmapEncoding::Bgra) => "BGRA bitmap glyph",
                SelectedGlyphRepresentation::Bitmap(BitmapEncoding::Png) => "PNG bitmap glyph",
                SelectedGlyphRepresentation::Bitmap(BitmapEncoding::PackedMask) => {
                    "packed-mask bitmap glyph"
                }
            };
            return Err(Error::new(
                BackendErrorCode::RenderFailed,
                format!("internal Vello scene cannot append selected {context}"),
            ));
        }
        let brush = glyph_paint_brush(validated.paint().fill())?;
        let stream_offsets = self.encoding.stream_offsets();
        let glyph_start = self.encoding.resources.glyphs.len();
        self.encoding
            .resources
            .glyphs
            .extend(validated.glyphs().iter().map(|glyph| Glyph {
                id: glyph.id(),
                x: glyph.x(),
                y: glyph.y(),
            }));
        let glyph_end = self.encoding.resources.glyphs.len();
        if glyph_start == glyph_end {
            return Ok(());
        }
        let index = self.encoding.resources.glyph_runs.len();
        self.encoding.resources.glyph_runs.push(GlyphRun {
            font: validated.font_data().data.clone(),
            transform: VelloTransform::from_kurbo(
                &(transform * kurbo::Affine::from(validated.transform())),
            ),
            glyph_transform: None,
            brush_transform: None,
            font_size: validated.size(),
            font_embolden: vello_encoding::FontEmbolden::default(),
            hint: false,
            normalized_coords: self.encoding.resources.normalized_coords.len()
                ..self.encoding.resources.normalized_coords.len(),
            style: Fill::NonZero.into(),
            glyphs: glyph_start..glyph_end,
            stream_offsets,
        });
        self.encoding
            .resources
            .patches
            .push(Patch::GlyphRun { index });
        self.encoding.encode_brush(&brush, 1.0);
        self.encoding.force_next_transform_and_style();
        Ok(())
    }

    fn push_layer_inner(
        &mut self,
        parameters: DrawBeginClip,
        fill: Fill,
        transform: kurbo::Affine,
        clip: &impl kurbo::Shape,
    ) {
        self.encoding
            .encode_transform(VelloTransform::from_kurbo(&transform));
        self.encoding.encode_fill_style(fill);
        if !self.encoding.encode_shape(clip, true) {
            self.encoding.encode_empty_shape();
        }
        self.encoding.encode_begin_clip(parameters);
    }

    #[cfg(test)]
    pub(crate) fn prepare_raster_scenario_for_test(
        scenario: VelloRasterScenario,
        parameters: RasterParameters,
    ) -> Result<PreparedVelloPass> {
        let mut scene = Self::default();
        scene.append_raster_scenario_for_test(scenario);
        scene.prepare_raster(parameters)
    }

    #[cfg(test)]
    pub(crate) fn observation_for_test(&self) -> VelloSceneObservation<'_> {
        VelloSceneObservation {
            glyph_runs: &self.encoding.resources.glyph_runs,
            glyphs: &self.encoding.resources.glyphs,
            patch_count: self.encoding.resources.patches.len(),
            normalized_coordinate_count: self.encoding.resources.normalized_coords.len(),
        }
    }

    #[cfg(test)]
    fn append_raster_scenario_for_test(&mut self, scenario: VelloRasterScenario) {
        match scenario {
            VelloRasterScenario::Base => {}
            VelloRasterScenario::LargePath => self.append_large_path_scenario_for_test(),
            VelloRasterScenario::Clip => self.append_clip_scenario_for_test(),
            VelloRasterScenario::LargePathAndClip => {
                self.append_large_path_scenario_for_test();
                self.append_clip_scenario_for_test();
            }
        }
    }

    #[cfg(test)]
    fn append_large_path_scenario_for_test(&mut self) {
        const LARGE_PATH_SCAN_LINE_SEGMENTS: usize = 4 * 256 * 256 + 1;

        self.encoding.encode_transform(VelloTransform::IDENTITY);
        self.encoding.encode_fill_style(Fill::NonZero);
        let mut path = self.encoding.encode_path(true);
        path.move_to(0.0, 0.0);
        let mut x = 1.0;
        for segment in 0..LARGE_PATH_SCAN_LINE_SEGMENTS {
            path.line_to(x, if segment % 2 == 0 { 1.0 } else { 0.0 });
            x += 1.0;
        }
        path.finish(true);
        self.encoding.encode_color(peniko::Color::BLACK);
    }

    #[cfg(test)]
    fn append_clip_scenario_for_test(&mut self) {
        const CLIP_PAIRS_FOR_REDUCE: usize = 129;

        for _ in 0..CLIP_PAIRS_FOR_REDUCE {
            self.encoding.encode_transform(VelloTransform::IDENTITY);
            self.encoding.encode_fill_style(Fill::NonZero);
            let encoded = self
                .encoding
                .encode_shape(&kurbo::Rect::new(0.0, 0.0, 1.0, 1.0), true);
            debug_assert!(encoded);
            self.encoding
                .encode_begin_clip(vello_encoding::DrawBeginClip::clip());
            self.encoding.encode_end_clip();
        }
    }
}

#[cfg(test)]
pub(crate) struct VelloSceneObservation<'a> {
    glyph_runs: &'a [GlyphRun],
    glyphs: &'a [Glyph],
    patch_count: usize,
    normalized_coordinate_count: usize,
}

#[cfg(test)]
impl VelloSceneObservation<'_> {
    pub(crate) const fn glyph_run_count_for_test(&self) -> usize {
        self.glyph_runs.len()
    }

    pub(crate) const fn glyph_count_for_test(&self) -> usize {
        self.glyphs.len()
    }

    pub(crate) const fn patch_count_for_test(&self) -> usize {
        self.patch_count
    }

    pub(crate) const fn normalized_coordinate_count_for_test(&self) -> usize {
        self.normalized_coordinate_count
    }

    pub(crate) fn first_glyph_run_for_test(&self) -> Option<VelloGlyphRunObservation<'_>> {
        self.glyph_runs
            .first()
            .map(|glyph_run| VelloGlyphRunObservation { glyph_run })
    }

    pub(crate) fn first_glyph_for_test(&self) -> Option<VelloGlyphObservation<'_>> {
        self.glyphs
            .first()
            .map(|glyph| VelloGlyphObservation { glyph })
    }
}

#[cfg(test)]
pub(crate) struct VelloGlyphRunObservation<'a> {
    glyph_run: &'a GlyphRun,
}

#[cfg(test)]
impl VelloGlyphRunObservation<'_> {
    pub(crate) const fn font_collection_index_for_test(&self) -> u32 {
        self.glyph_run.font.index
    }

    pub(crate) fn font_data_matches_for_test(&self, expected: &[u8]) -> bool {
        self.glyph_run.font.data.as_ref() == expected
    }

    pub(crate) const fn transform_components_for_test(&self) -> [f32; 6] {
        let transform = self.glyph_run.transform;
        [
            transform.matrix[0],
            transform.matrix[1],
            transform.matrix[2],
            transform.matrix[3],
            transform.translation[0],
            transform.translation[1],
        ]
    }

    pub(crate) const fn has_glyph_transform_for_test(&self) -> bool {
        self.glyph_run.glyph_transform.is_some()
    }

    pub(crate) const fn has_brush_transform_for_test(&self) -> bool {
        self.glyph_run.brush_transform.is_some()
    }

    pub(crate) const fn font_size_for_test(&self) -> f32 {
        self.glyph_run.font_size
    }

    pub(crate) const fn embolden_amount_for_test(&self) -> [f64; 2] {
        [
            self.glyph_run.font_embolden.amount.xx,
            self.glyph_run.font_embolden.amount.yy,
        ]
    }

    pub(crate) const fn uses_hinting_for_test(&self) -> bool {
        self.glyph_run.hint
    }

    pub(crate) fn normalized_coordinate_range_for_test(&self) -> std::ops::Range<usize> {
        self.glyph_run.normalized_coords.clone()
    }

    pub(crate) fn glyph_range_for_test(&self) -> std::ops::Range<usize> {
        self.glyph_run.glyphs.clone()
    }

    pub(crate) const fn uses_nonzero_fill_for_test(&self) -> bool {
        matches!(self.glyph_run.style, peniko::Style::Fill(Fill::NonZero))
    }
}

#[cfg(test)]
pub(crate) struct VelloGlyphObservation<'a> {
    glyph: &'a Glyph,
}

#[cfg(test)]
impl VelloGlyphObservation<'_> {
    pub(crate) const fn id_for_test(&self) -> u32 {
        self.glyph.id
    }

    pub(crate) const fn x_for_test(&self) -> f32 {
        self.glyph.x
    }

    pub(crate) const fn y_for_test(&self) -> f32 {
        self.glyph.y
    }
}
