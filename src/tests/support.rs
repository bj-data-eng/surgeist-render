use crate::{Color, FontData, FontRef, TextGlyph, TextPaint, TextRun, TextRunBounds, Transform};

pub(super) const AHEM_FONT_BYTES: &[u8] =
    include_bytes!("../../tests/fixtures/fonts/ahem/Ahem.ttf");
pub(super) const AHEM_FONT_ID: u64 = 9001;
pub(super) const AHEM_GLYPH_X: u32 = 58;
pub(super) const AHEM_GLYPH_DESCENT_P: u32 = 82;
pub(super) const AHEM_GLYPH_ASCENT_E_ACUTE: u32 = 100;

pub(super) fn text_run_for<'a>(
    font_data: FontData,
    size: f32,
    transform: Transform,
    glyphs: &'a [TextGlyph],
) -> TextRun<'a> {
    TextRun::try_new(
        FontRef::new(AHEM_FONT_ID)
            .named("selected glyph preflight")
            .with_data(font_data),
        size,
        transform,
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap()
}

pub(super) fn ahem_font(name: &'static str) -> FontRef<'static> {
    FontRef::new(AHEM_FONT_ID)
        .named(name)
        .with_data(FontData::try_from_bytes(AHEM_FONT_BYTES.to_vec(), 0).unwrap())
}
