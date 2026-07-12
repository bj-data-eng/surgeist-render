// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::io::Cursor;

use png::{BitDepth, ColorType, Transformations};
use skrifa::{
    GlyphId, MetadataProvider, Tag,
    bitmap::MaskData,
    color::{Brush as ColorBrush, ColorGlyph, ColorPainter, CompositeMode},
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlinePen},
    raw::{
        FontData as RawFontData, ReadError as RawReadError, TableProvider,
        tables::{
            bitmap::{
                BitmapContent as RawBitmapContent, BitmapData as RawBitmapData,
                BitmapDataFormat as RawBitmapDataFormat, BitmapLocation,
                BitmapMetrics as RawBitmapMetrics, BitmapSize, IndexSubtable,
            },
            cpal::Cpal,
        },
        types::BoundingBox,
    },
};

use crate::{
    BackendErrorCode, Error, FontData, Result, TextGlyph, TextPaint, TextRun, Transform,
    text::invalid_font_data,
};

const SKRIFA_NORMALIZED_COORDS: &[skrifa::instance::NormalizedCoord] = &[];

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

struct SelectedBitmap<'a> {
    width: u32,
    height: u32,
    data: SelectedBitmapData<'a>,
}

enum SelectedBitmapData<'a> {
    Bgra(&'a [u8]),
    Png {
        data: &'a [u8],
        expected_dimensions: Option<(u32, u32)>,
    },
    Mask {
        bpp: u8,
        is_packed: bool,
        data: &'a [u8],
    },
    Unsupported,
}

impl<'a> ValidatedGlyphRun<'a> {
    pub(crate) const fn size(&self) -> f32 {
        self.size
    }

    pub(crate) const fn transform(&self) -> Transform {
        self.transform
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
    let outlines = font.outline_glyphs();
    let mut representations = Vec::with_capacity(run.glyphs().len());

    for glyph in run.glyphs() {
        if glyph.id() >= glyph_count {
            return Err(missing_glyph_error(glyph.id()));
        }
        let glyph_id = GlyphId::new(glyph.id());
        let representation = if selected_color_glyph(colr.as_ref(), glyph_id, font_data)? {
            preflight_selected_image_head(&font, font_data)?;
            let cpal = font.cpal().map_err(|_| font_data_error(font_data))?;
            let color = font
                .color_glyphs()
                .get(glyph_id)
                .ok_or_else(|| font_data_error(font_data))?;
            preflight_color_glyph(&color, &outlines, &cpal, run.size(), font_data)?;
            SelectedGlyphRepresentation::Colr
        } else if let Some(bitmap) =
            select_bitmap(&font, Size::new(run.size()), glyph_id, font_data)?
        {
            preflight_selected_image_head(&font, font_data)?;
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
    colr: Option<&skrifa::raw::tables::colr::Colr<'a>>,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<bool> {
    let Some(colr) = colr else {
        return Ok(false);
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
    Ok(selected)
}

fn preflight_selected_image_head(font: &skrifa::FontRef<'_>, font_data: &FontData) -> Result<()> {
    font.head()
        .map_err(|_| font_data_error(font_data))?
        .units_per_em();
    Ok(())
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

fn select_bitmap<'a>(
    font: &skrifa::FontRef<'a>,
    size: Size,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<Option<SelectedBitmap<'a>>> {
    if font_has_table(font, Tag::new(b"sbix")) {
        return select_sbix_bitmap(font, size, glyph_id, font_data);
    }
    if bitmap_table_pair_is_present(font, Tag::new(b"CBLC"), Tag::new(b"CBDT"), font_data)? {
        let cblc = font.cblc().map_err(|_| font_data_error(font_data))?;
        let cbdt = font.cbdt().map_err(|_| font_data_error(font_data))?;
        return select_bdt_bitmap(
            cblc.bitmap_sizes(),
            cblc.num_sizes(),
            cblc.offset_data(),
            size,
            glyph_id,
            font_data,
            |location| cbdt.data(location),
        );
    }
    if bitmap_table_pair_is_present(font, Tag::new(b"EBLC"), Tag::new(b"EBDT"), font_data)? {
        let eblc = font.eblc().map_err(|_| font_data_error(font_data))?;
        let ebdt = font.ebdt().map_err(|_| font_data_error(font_data))?;
        return select_bdt_bitmap(
            eblc.bitmap_sizes(),
            eblc.num_sizes(),
            eblc.offset_data(),
            size,
            glyph_id,
            font_data,
            |location| ebdt.data(location),
        );
    }
    Ok(None)
}

fn font_has_table(font: &skrifa::FontRef<'_>, tag: Tag) -> bool {
    font.table_directory()
        .table_records()
        .iter()
        .any(|record| record.tag() == tag)
}

fn bitmap_table_pair_is_present(
    font: &skrifa::FontRef<'_>,
    location_tag: Tag,
    data_tag: Tag,
    font_data: &FontData,
) -> Result<bool> {
    match (
        font_has_table(font, location_tag),
        font_has_table(font, data_tag),
    ) {
        (false, false) => Ok(false),
        (true, true) => Ok(true),
        _ => Err(font_data_error(font_data)),
    }
}

fn select_sbix_bitmap<'a>(
    font: &skrifa::FontRef<'a>,
    size: Size,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<Option<SelectedBitmap<'a>>> {
    let sbix = font.sbix().map_err(|_| font_data_error(font_data))?;
    let strike_count = checked_count(sbix.num_strikes(), 0, font_data)?;
    let strikes = sbix.strikes();
    if strikes.len() != strike_count {
        return Err(font_data_error(font_data));
    }

    let requested_size = size.ppem().unwrap_or(f32::MAX);
    let mut best = None;
    for index in 0..strike_count {
        let strike = strikes.get(index).map_err(|_| font_data_error(font_data))?;
        let strike_size = f32::from(strike.ppem());
        if !should_consider_bitmap_strike(strike_size, requested_size, best.as_ref()) {
            continue;
        }
        if let Some(bitmap) = select_sbix_glyph(&strike, glyph_id, font_data)? {
            best = Some((strike_size, bitmap));
        }
    }
    Ok(best.map(|(_, bitmap)| bitmap))
}

fn select_sbix_glyph<'a>(
    strike: &skrifa::raw::tables::sbix::Strike<'a>,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<Option<SelectedBitmap<'a>>> {
    let Some(glyph) = strike
        .glyph_data(glyph_id)
        .map_err(|_| font_data_error(font_data))?
    else {
        return Ok(None);
    };
    if glyph.graphic_type() != Tag::new(b"png ") {
        return Ok(Some(SelectedBitmap {
            width: 0,
            height: 0,
            data: SelectedBitmapData::Unsupported,
        }));
    }
    Ok(Some(SelectedBitmap {
        width: 0,
        height: 0,
        data: SelectedBitmapData::Png {
            data: glyph.data(),
            expected_dimensions: None,
        },
    }))
}

fn select_bdt_bitmap<'a, F>(
    bitmap_sizes: &'a [BitmapSize],
    declared_size_count: u32,
    offset_data: RawFontData<'a>,
    size: Size,
    glyph_id: GlyphId,
    font_data: &FontData,
    bitmap_data: F,
) -> Result<Option<SelectedBitmap<'a>>>
where
    F: Fn(&BitmapLocation) -> std::result::Result<RawBitmapData<'a>, RawReadError>,
{
    if bitmap_sizes.len() != checked_count(declared_size_count, 0, font_data)? {
        return Err(font_data_error(font_data));
    }

    let requested_size = size.ppem().unwrap_or(f32::MAX);
    let mut best = None;
    for bitmap_size in bitmap_sizes {
        let strike_size = f32::from(bitmap_size.ppem_y());
        if !should_consider_bitmap_strike(strike_size, requested_size, best.as_ref()) {
            continue;
        }
        let Some(location) = select_bdt_location(bitmap_size, offset_data, glyph_id, font_data)?
        else {
            continue;
        };
        if location.is_empty() {
            continue;
        }
        let data = bitmap_data(&location).map_err(|_| font_data_error(font_data))?;
        let bitmap = selected_bdt_glyph(bitmap_size, data, font_data)?;
        best = Some((strike_size, bitmap));
    }
    Ok(best.map(|(_, bitmap)| bitmap))
}

fn should_consider_bitmap_strike(
    candidate_size: f32,
    requested_size: f32,
    best: Option<&(f32, SelectedBitmap<'_>)>,
) -> bool {
    let Some((best_size, _)) = best else {
        return true;
    };
    (candidate_size >= requested_size && candidate_size < *best_size)
        || (*best_size < requested_size && candidate_size > *best_size)
}

fn select_bdt_location(
    bitmap_size: &BitmapSize,
    offset_data: RawFontData<'_>,
    glyph_id: GlyphId,
    font_data: &FontData,
) -> Result<Option<BitmapLocation>> {
    if !(bitmap_size.start_glyph_index()..=bitmap_size.end_glyph_index()).contains(&glyph_id) {
        return Ok(None);
    }
    let subtable_list = bitmap_size
        .index_subtable_list(offset_data)
        .map_err(|_| font_data_error(font_data))?;
    let records = subtable_list.index_subtable_records();
    if records.len() != checked_count(bitmap_size.number_of_index_subtables(), 0, font_data)? {
        return Err(font_data_error(font_data));
    }
    let Some(record) = records.iter().find(|record| {
        (record.first_glyph_index()..=record.last_glyph_index()).contains(&glyph_id)
    }) else {
        return Ok(None);
    };
    let subtable = record
        .index_subtable(subtable_list.offset_data())
        .map_err(|_| font_data_error(font_data))?;
    let glyph_index =
        checked_glyph_index(glyph_id, record.first_glyph_index().to_u32(), font_data)?;
    selected_bdt_location_for_subtable(bitmap_size, &subtable, glyph_id, glyph_index, font_data)
}

fn selected_bdt_location_for_subtable(
    bitmap_size: &BitmapSize,
    subtable: &IndexSubtable<'_>,
    glyph_id: GlyphId,
    glyph_index: usize,
    font_data: &FontData,
) -> Result<Option<BitmapLocation>> {
    let mut location = BitmapLocation {
        bit_depth: bitmap_size.bit_depth(),
        ..BitmapLocation::default()
    };
    match subtable {
        IndexSubtable::Format1(subtable) => {
            let offsets = subtable.sbit_offsets();
            let start = checked_offset(
                subtable.image_data_offset(),
                offsets
                    .get(glyph_index)
                    .ok_or_else(|| font_data_error(font_data))?
                    .get(),
                font_data,
            )?;
            let end = checked_offset(
                subtable.image_data_offset(),
                offsets
                    .get(
                        glyph_index
                            .checked_add(1)
                            .ok_or_else(|| font_data_error(font_data))?,
                    )
                    .ok_or_else(|| font_data_error(font_data))?
                    .get(),
                font_data,
            )?;
            location.format = subtable.image_format();
            location.data_offset = start;
            location.data_size = end
                .checked_sub(start)
                .ok_or_else(|| font_data_error(font_data))?;
        }
        IndexSubtable::Format2(subtable) => {
            location.format = subtable.image_format();
            location.data_size =
                usize::try_from(subtable.image_size()).map_err(|_| font_data_error(font_data))?;
            location.data_offset = checked_indexed_offset(
                subtable.image_data_offset(),
                glyph_index,
                subtable.image_size(),
                font_data,
            )?;
            location.metrics = Some(
                *subtable
                    .big_metrics()
                    .first()
                    .ok_or_else(|| font_data_error(font_data))?,
            );
        }
        IndexSubtable::Format3(subtable) => {
            let offsets = subtable.sbit_offsets();
            let start = checked_offset(
                subtable.image_data_offset(),
                u32::from(
                    offsets
                        .get(glyph_index)
                        .ok_or_else(|| font_data_error(font_data))?
                        .get(),
                ),
                font_data,
            )?;
            let end = checked_offset(
                subtable.image_data_offset(),
                u32::from(
                    offsets
                        .get(
                            glyph_index
                                .checked_add(1)
                                .ok_or_else(|| font_data_error(font_data))?,
                        )
                        .ok_or_else(|| font_data_error(font_data))?
                        .get(),
                ),
                font_data,
            )?;
            location.format = subtable.image_format();
            location.data_offset = start;
            location.data_size = end
                .checked_sub(start)
                .ok_or_else(|| font_data_error(font_data))?;
        }
        IndexSubtable::Format4(subtable) => {
            let glyphs = subtable.glyph_array();
            if glyphs.len() != checked_count(subtable.num_glyphs(), 1, font_data)? {
                return Err(font_data_error(font_data));
            }
            let (sentinel, glyphs) = glyphs
                .split_last()
                .ok_or_else(|| font_data_error(font_data))?;
            if glyphs
                .windows(2)
                .any(|entries| entries[0].glyph_id().to_u32() >= entries[1].glyph_id().to_u32())
            {
                return Err(font_data_error(font_data));
            }
            let glyph_index =
                glyphs.binary_search_by(|entry| entry.glyph_id().to_u32().cmp(&glyph_id.to_u32()));
            let Ok(glyph_index) = glyph_index else {
                return Ok(None);
            };
            let start = usize::from(
                glyphs
                    .get(glyph_index)
                    .ok_or_else(|| font_data_error(font_data))?
                    .sbit_offset(),
            );
            let next_index = glyph_index
                .checked_add(1)
                .ok_or_else(|| font_data_error(font_data))?;
            let end = if let Some(entry) = glyphs.get(next_index) {
                usize::from(entry.sbit_offset())
            } else {
                usize::from(sentinel.sbit_offset())
            };
            location.format = subtable.image_format();
            location.data_offset = start;
            location.data_size = end
                .checked_sub(start)
                .ok_or_else(|| font_data_error(font_data))?;
        }
        IndexSubtable::Format5(subtable) => {
            let glyphs = subtable.glyph_array();
            if glyphs.len() != checked_count(subtable.num_glyphs(), 0, font_data)? {
                return Err(font_data_error(font_data));
            }
            if glyphs
                .windows(2)
                .any(|entries| entries[0].get().to_u32() >= entries[1].get().to_u32())
            {
                return Err(font_data_error(font_data));
            }
            let glyph_index =
                glyphs.binary_search_by(|entry| entry.get().to_u32().cmp(&glyph_id.to_u32()));
            let Ok(glyph_index) = glyph_index else {
                return Ok(None);
            };
            location.format = subtable.image_format();
            location.data_size =
                usize::try_from(subtable.image_size()).map_err(|_| font_data_error(font_data))?;
            location.data_offset = checked_indexed_offset(
                subtable.image_data_offset(),
                glyph_index,
                subtable.image_size(),
                font_data,
            )?;
            location.metrics = Some(
                *subtable
                    .big_metrics()
                    .first()
                    .ok_or_else(|| font_data_error(font_data))?,
            );
        }
    }
    Ok(Some(location))
}

fn selected_bdt_glyph<'a>(
    bitmap_size: &BitmapSize,
    bitmap_data: RawBitmapData<'a>,
    font_data: &FontData,
) -> Result<SelectedBitmap<'a>> {
    let (width, height) = match bitmap_data.metrics {
        RawBitmapMetrics::Small(metrics) => {
            (u32::from(metrics.width()), u32::from(metrics.height()))
        }
        RawBitmapMetrics::Big(metrics) => (u32::from(metrics.width()), u32::from(metrics.height())),
    };
    let data = match (bitmap_size.bit_depth(), bitmap_data.content) {
        (32, RawBitmapContent::Data(RawBitmapDataFormat::Png, data)) => SelectedBitmapData::Png {
            data,
            expected_dimensions: Some((width, height)),
        },
        (32, RawBitmapContent::Data(RawBitmapDataFormat::ByteAligned, data)) => {
            SelectedBitmapData::Bgra(data)
        }
        (1 | 2 | 4 | 8, RawBitmapContent::Data(RawBitmapDataFormat::ByteAligned, data)) => {
            SelectedBitmapData::Mask {
                bpp: bitmap_size.bit_depth(),
                is_packed: false,
                data,
            }
        }
        (1 | 2 | 4 | 8, RawBitmapContent::Data(RawBitmapDataFormat::BitAligned, data)) => {
            SelectedBitmapData::Mask {
                bpp: bitmap_size.bit_depth(),
                is_packed: true,
                data,
            }
        }
        (1 | 2 | 4 | 8 | 32, _) => SelectedBitmapData::Unsupported,
        _ => return Err(font_data_error(font_data)),
    };
    Ok(SelectedBitmap {
        width,
        height,
        data,
    })
}

fn checked_count(count: u32, extra: usize, font_data: &FontData) -> Result<usize> {
    usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_add(extra))
        .ok_or_else(|| font_data_error(font_data))
}

fn checked_glyph_index(
    glyph_id: GlyphId,
    first_glyph_id: u32,
    font_data: &FontData,
) -> Result<usize> {
    usize::try_from(glyph_id.to_u32())
        .ok()
        .and_then(|glyph_id| {
            usize::try_from(first_glyph_id)
                .ok()
                .and_then(|first_glyph_id| glyph_id.checked_sub(first_glyph_id))
        })
        .ok_or_else(|| font_data_error(font_data))
}

fn checked_offset(base: u32, offset: u32, font_data: &FontData) -> Result<usize> {
    usize::try_from(base)
        .ok()
        .and_then(|base| {
            usize::try_from(offset)
                .ok()
                .and_then(|offset| base.checked_add(offset))
        })
        .ok_or_else(|| font_data_error(font_data))
}

fn checked_indexed_offset(
    base: u32,
    index: usize,
    stride: u32,
    font_data: &FontData,
) -> Result<usize> {
    usize::try_from(base)
        .ok()
        .and_then(|base| {
            usize::try_from(stride)
                .ok()
                .and_then(|stride| index.checked_mul(stride))
                .and_then(|offset| base.checked_add(offset))
        })
        .ok_or_else(|| font_data_error(font_data))
}

fn preflight_bitmap(bitmap: &SelectedBitmap<'_>, font_data: &FontData) -> Result<BitmapEncoding> {
    match &bitmap.data {
        SelectedBitmapData::Bgra(data) => {
            let pixel_count = checked_pixel_count(bitmap.width, bitmap.height, font_data)?;
            let expected = pixel_count
                .checked_mul(4)
                .ok_or_else(|| font_data_error(font_data))?;
            if data.len() != expected {
                return Err(font_data_error(font_data));
            }
            Ok(BitmapEncoding::Bgra)
        }
        SelectedBitmapData::Png {
            data,
            expected_dimensions,
        } => preflight_png(data, *expected_dimensions, font_data),
        SelectedBitmapData::Mask {
            bpp,
            is_packed,
            data,
        } => {
            let pixel_count = checked_pixel_count(bitmap.width, bitmap.height, font_data)?;
            if !matches!(bpp, 1 | 2 | 4 | 8) {
                return Err(font_data_error(font_data));
            }
            if !is_packed {
                return Err(unsupported_image_error());
            }
            let bits = pixel_count
                .checked_mul(usize::from(*bpp))
                .ok_or_else(|| font_data_error(font_data))?;
            let expected = bits
                .checked_add(7)
                .and_then(|bits| bits.checked_div(8))
                .ok_or_else(|| font_data_error(font_data))?;
            if data.len() != expected {
                return Err(font_data_error(font_data));
            }
            MaskData {
                bpp: *bpp,
                is_packed: *is_packed,
                data,
            }
            .decode(bitmap.width, bitmap.height)
            .map_err(|_| font_data_error(font_data))?;
            Ok(BitmapEncoding::PackedMask)
        }
        SelectedBitmapData::Unsupported => Err(unsupported_image_error()),
    }
}

fn preflight_png(
    data: &[u8],
    expected_dimensions: Option<(u32, u32)>,
    font_data: &FontData,
) -> Result<BitmapEncoding> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(Transformations::ALPHA | Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|_| font_data_error(font_data))?;
    let output_color_type = reader.output_color_type();
    let output_size = reader
        .output_buffer_size()
        .ok_or_else(|| font_data_error(font_data))?;
    let mut output = vec![0; output_size];
    let info = reader
        .next_frame(output.as_mut_slice())
        .map_err(|_| font_data_error(font_data))?;
    if let Some((width, height)) = expected_dimensions
        && (info.width != width || info.height != height)
    {
        return Err(font_data_error(font_data));
    }
    if output_color_type != (ColorType::Rgba, BitDepth::Eight) {
        return Err(unsupported_image_error());
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
