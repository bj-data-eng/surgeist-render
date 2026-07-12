// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::io::Cursor;

use png::{BitDepth, ColorType, Transformations};
use skrifa::{
    GlyphId, MetadataProvider,
    bitmap::{BitmapData, BitmapGlyph},
    color::{Brush as ColorBrush, ColorGlyph, ColorPainter, CompositeMode},
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlinePen},
    raw::{TableProvider, tables::cpal::Cpal, types::BoundingBox},
};

use crate::{
    BackendErrorCode, Error, FontData, Result, TextGlyph, TextPaint, TextRun, Transform,
    text::invalid_font_data,
};

const SKRIFA_NORMALIZED_COORDS: &[skrifa::instance::NormalizedCoord] = &[];
const VELLO_NORMALIZED_COORDS: &[vello_encoding::NormalizedCoord] = &[];

pub(crate) struct ValidatedGlyphRun<'a> {
    font_data: &'a FontData,
    glyphs: &'a [TextGlyph],
    size: f32,
    transform: Transform,
    paint: &'a TextPaint,
    representations: Vec<SelectedGlyphRepresentation>,
}

pub(crate) enum SelectedGlyphRepresentation {
    Outline,
    Colr,
    Bitmap(BitmapEncoding),
}

pub(crate) enum BitmapEncoding {
    Bgra,
    Png,
    PackedMask,
}

impl<'a> ValidatedGlyphRun<'a> {
    pub(crate) const fn size(&self) -> f32 {
        self.size
    }

    pub(crate) const fn transform(&self) -> Transform {
        self.transform
    }

    pub(crate) const fn normalized_coords(&self) -> &[vello_encoding::NormalizedCoord] {
        VELLO_NORMALIZED_COORDS
    }

    pub(crate) const fn hinting_enabled(&self) -> bool {
        false
    }

    pub(crate) const fn uses_non_zero_fill(&self) -> bool {
        true
    }

    pub(crate) const fn embolden_amount(&self) -> (f64, f64) {
        (0.0, 0.0)
    }

    pub(crate) const fn font_data(&self) -> &FontData {
        self.font_data
    }

    pub(crate) const fn glyphs(&self) -> &[TextGlyph] {
        self.glyphs
    }

    pub(crate) const fn paint(&self) -> &TextPaint {
        self.paint
    }

    pub(crate) fn representations(&self) -> &[SelectedGlyphRepresentation] {
        self.representations.as_slice()
    }
}

pub(crate) fn preflight_selected_glyphs<'a>(run: &'a TextRun<'a>) -> Result<ValidatedGlyphRun<'a>> {
    let font_data = run
        .font()
        .data
        .as_ref()
        .ok_or_else(|| Error::invalid_input_message("text run font data is required"))?;
    let font = skrifa::FontRef::from_index(font_data.bytes(), font_data.index())
        .map_err(|_| font_data_error(font_data))?;
    let glyph_count = font
        .maxp()
        .map_err(|_| font_data_error(font_data))?
        .num_glyphs() as u32;
    let colr = font
        .table_data(skrifa::Tag::new(b"COLR"))
        .map(|_| font.colr().map_err(|_| font_data_error(font_data)))
        .transpose()?;
    let colors = font.color_glyphs();
    let bitmaps = font.bitmap_strikes();
    let outlines = font.outline_glyphs();
    let mut representations = Vec::with_capacity(run.glyphs().len());

    for glyph in run.glyphs() {
        if glyph.id() >= glyph_count {
            return Err(missing_glyph_error(glyph.id()));
        }
        let glyph_id = GlyphId::new(glyph.id());
        let representation =
            if selected_color_glyph(&colors, colr.as_ref(), glyph_id, font_data)?.is_some() {
                let cpal = font.cpal().map_err(|_| font_data_error(font_data))?;
                let color = selected_color_glyph(&colors, colr.as_ref(), glyph_id, font_data)?
                    .ok_or_else(|| font_data_error(font_data))?;
                preflight_color_glyph(&color, &outlines, &cpal, run.size(), font_data)?;
                SelectedGlyphRepresentation::Colr
            } else if let Some(bitmap) = bitmaps.glyph_for_size(Size::new(run.size()), glyph_id) {
                SelectedGlyphRepresentation::Bitmap(preflight_bitmap(&bitmap, font_data)?)
            } else {
                preflight_outline(&outlines, glyph_id, run.size(), font_data)?;
                SelectedGlyphRepresentation::Outline
            };
        representations.push(representation);
    }

    Ok(ValidatedGlyphRun {
        font_data,
        glyphs: run.glyphs(),
        size: run.size(),
        transform: run.transform(),
        paint: run.paint(),
        representations,
    })
}

fn selected_color_glyph<'a>(
    colors: &skrifa::color::ColorGlyphCollection<'a>,
    colr: Option<&skrifa::raw::tables::colr::Colr<'a>>,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<Option<ColorGlyph<'a>>> {
    let Some(colr) = colr else {
        return Ok(None);
    };
    let selected = match colr.version() {
        0 => colr
            .v0_base_glyph(glyph_id)
            .map_err(|_| font_data_error(font_data))?
            .is_some(),
        1 => colr
            .v1_base_glyph(glyph_id)
            .map_err(|_| font_data_error(font_data))?
            .is_some(),
        _ => return Err(font_data_error(font_data)),
    };
    if !selected {
        return Ok(None);
    }
    colors
        .get(glyph_id)
        .map(Some)
        .ok_or_else(|| font_data_error(font_data))
}

fn preflight_outline(
    outlines: &skrifa::OutlineGlyphCollection<'_>,
    glyph_id: GlyphId,
    size: f32,
    font_data: &FontData,
) -> Result<()> {
    let outline = outlines
        .get(glyph_id)
        .ok_or_else(|| font_data_error(font_data))?;
    let mut pen = ValidationPen;
    outline
        .draw(
            DrawSettings::unhinted(Size::new(size), LocationRef::new(SKRIFA_NORMALIZED_COORDS)),
            &mut pen,
        )
        .map_err(|_| font_data_error(font_data))?;
    Ok(())
}

fn preflight_color_glyph(
    color: &ColorGlyph<'_>,
    outlines: &skrifa::OutlineGlyphCollection<'_>,
    cpal: &Cpal<'_>,
    size: f32,
    font_data: &FontData,
) -> Result<()> {
    let mut painter = PreflightColorPainter {
        font_data,
        outlines,
        cpal,
        size,
        failed: false,
    };
    color
        .paint(LocationRef::new(SKRIFA_NORMALIZED_COORDS), &mut painter)
        .map_err(|_| font_data_error(font_data))?;
    if painter.failed {
        return Err(font_data_error(font_data));
    }
    Ok(())
}

fn preflight_bitmap(bitmap: &BitmapGlyph<'_>, font_data: &FontData) -> Result<BitmapEncoding> {
    let pixel_count = checked_pixel_count(bitmap.width, bitmap.height, font_data)?;
    match &bitmap.data {
        BitmapData::Bgra(data) => {
            let expected = pixel_count
                .checked_mul(4)
                .ok_or_else(|| font_data_error(font_data))?;
            if data.len() != expected {
                return Err(font_data_error(font_data));
            }
            Ok(BitmapEncoding::Bgra)
        }
        BitmapData::Png(data) => preflight_png(data, bitmap.width, bitmap.height, font_data),
        BitmapData::Mask(mask) => {
            if !matches!(mask.bpp, 1 | 2 | 4 | 8) {
                return Err(font_data_error(font_data));
            }
            if !mask.is_packed {
                return Err(unsupported_image_error());
            }
            let bits = pixel_count
                .checked_mul(usize::from(mask.bpp))
                .ok_or_else(|| font_data_error(font_data))?;
            let expected = bits
                .checked_add(7)
                .and_then(|bits| bits.checked_div(8))
                .ok_or_else(|| font_data_error(font_data))?;
            if mask.data.len() != expected {
                return Err(font_data_error(font_data));
            }
            mask.decode(bitmap.width, bitmap.height)
                .map_err(|_| font_data_error(font_data))?;
            Ok(BitmapEncoding::PackedMask)
        }
    }
}

fn preflight_png(
    data: &[u8],
    width: u32,
    height: u32,
    font_data: &FontData,
) -> Result<BitmapEncoding> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(Transformations::ALPHA | Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|_| font_data_error(font_data))?;
    if reader.output_color_type() != (ColorType::Rgba, BitDepth::Eight) {
        return Err(unsupported_image_error());
    }
    let output_size = reader
        .output_buffer_size()
        .ok_or_else(|| font_data_error(font_data))?;
    let mut output = vec![0; output_size];
    let info = reader
        .next_frame(output.as_mut_slice())
        .map_err(|_| font_data_error(font_data))?;
    if info.width != width || info.height != height {
        return Err(font_data_error(font_data));
    }
    Ok(BitmapEncoding::Png)
}

fn checked_pixel_count(width: u32, height: u32, font_data: &FontData) -> Result<usize> {
    usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| font_data_error(font_data))
}

fn font_data_error(font_data: &FontData) -> Error {
    invalid_font_data(font_data.bytes().len(), font_data.index())
}

fn missing_glyph_error(glyph_id: u32) -> Error {
    Error::invalid_value(
        "text_glyph.id",
        glyph_id,
        "must identify a drawable glyph in the selected FontData",
    )
}

fn unsupported_image_error() -> Error {
    Error::new(
        BackendErrorCode::RenderFailed,
        "internal Vello glyph image encoding is unsupported",
    )
}

struct ValidationPen;

impl OutlinePen for ValidationPen {
    fn move_to(&mut self, _: f32, _: f32) {}

    fn line_to(&mut self, _: f32, _: f32) {}

    fn quad_to(&mut self, _: f32, _: f32, _: f32, _: f32) {}

    fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}

    fn close(&mut self) {}
}

struct PreflightColorPainter<'a, 'font> {
    font_data: &'a FontData,
    outlines: &'a skrifa::OutlineGlyphCollection<'font>,
    cpal: &'a Cpal<'font>,
    size: f32,
    failed: bool,
}

impl PreflightColorPainter<'_, '_> {
    fn validate_outline(&mut self, glyph_id: GlyphId) {
        if self.failed {
            return;
        }
        if preflight_outline(self.outlines, glyph_id, self.size, self.font_data).is_err() {
            self.failed = true;
        }
    }

    fn validate_palette_index(&mut self, palette_index: u16) {
        if self.failed || palette_index == u16::MAX {
            return;
        }
        let Some(records) = self.cpal.color_records_array() else {
            self.failed = true;
            return;
        };
        let Ok(records) = records else {
            self.failed = true;
            return;
        };
        if records.get(usize::from(palette_index)).is_none() {
            self.failed = true;
        }
    }

    fn validate_brush(&mut self, brush: ColorBrush<'_>) {
        match brush {
            ColorBrush::Solid { palette_index, .. } => self.validate_palette_index(palette_index),
            ColorBrush::LinearGradient { color_stops, .. }
            | ColorBrush::RadialGradient { color_stops, .. }
            | ColorBrush::SweepGradient { color_stops, .. } => {
                for color_stop in color_stops {
                    self.validate_palette_index(color_stop.palette_index);
                }
            }
        }
    }
}

impl ColorPainter for PreflightColorPainter<'_, '_> {
    fn push_transform(&mut self, _: skrifa::color::Transform) {}

    fn pop_transform(&mut self) {}

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        self.validate_outline(glyph_id);
    }

    fn push_clip_box(&mut self, _: BoundingBox<f32>) {}

    fn pop_clip(&mut self) {}

    fn fill(&mut self, brush: ColorBrush<'_>) {
        self.validate_brush(brush);
    }

    fn fill_glyph(
        &mut self,
        glyph_id: GlyphId,
        _: Option<skrifa::color::Transform>,
        brush: ColorBrush<'_>,
    ) {
        self.validate_outline(glyph_id);
        self.validate_brush(brush);
    }

    fn push_layer(&mut self, _: CompositeMode) {}
}
