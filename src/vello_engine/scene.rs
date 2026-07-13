// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::Fill;
use vello_encoding::{Encoding, Glyph, GlyphRun, Patch, Transform as VelloTransform};

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

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "C03 T2 private Vello scene state is intentionally staged for T7 cutover."
    )
)]
#[derive(Default)]
pub(crate) struct VelloScene {
    encoding: Encoding,
}

impl VelloScene {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T2 private text encoding entry point is intentionally staged for T7 cutover."
        )
    )]
    pub(crate) fn encode_text_run(&mut self, run: &TextRun<'_>) -> Result<()> {
        let validated = preflight_selected_glyphs(run)?;
        self.append_validated_glyph_run(validated)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C03 T3 scene-owned preparation is staged until T4 checked encoding and T7 cutover."
        )
    )]
    pub(crate) fn prepare_raster(&self, parameters: RasterParameters) -> Result<PreparedVelloPass> {
        raster::prepare(&self.encoding, parameters)
    }

    fn append_validated_glyph_run(&mut self, validated: ValidatedGlyphRun<'_>) -> Result<()> {
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
            transform: VelloTransform::from_kurbo(&kurbo::Affine::from(validated.transform())),
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
