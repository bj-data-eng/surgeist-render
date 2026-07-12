// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::Fill;
use vello_encoding::{Encoding, Glyph, GlyphRun, Patch, Transform as VelloTransform};

use crate::{BackendErrorCode, Error, Result, TextRun, encode::glyph_paint_brush};

use super::glyph::{
    BitmapEncoding, SelectedGlyphRepresentation, ValidatedGlyphRun, preflight_selected_glyphs,
};

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
    pub(crate) const fn encoding_for_test(&self) -> &Encoding {
        &self.encoding
    }
}
