use crate::{
    BackdropCaptureBounds, BackdropFilterInput, BorderEdges, BorderSide, BorderStyle, Color,
    FilterList, FilterOp, FontData, FontRef, Image, ImageBuffer, Layer, PhysicalSize, Rect,
    ResolvedLayerAlphaMask, Scene, Size, TextGlyph, TextPaint, TextRun, TextRunBounds, Transform,
    UnitFilterAmount, reference::PremultipliedRgba8,
};

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

pub(super) fn image_from_buffer(buffer: ImageBuffer) -> Image {
    let size = buffer.size();
    Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        buffer.into_rgba(),
    )
    .unwrap()
}

pub(super) fn resolved_layer_alpha_mask_from_buffer(buffer: ImageBuffer) -> ResolvedLayerAlphaMask {
    let size = buffer.size();
    ResolvedLayerAlphaMask::try_new(
        image_from_buffer(buffer),
        Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height())),
    )
    .unwrap()
}

pub(super) fn add_planning_text(scene: &mut Scene, bounds: TextRunBounds) {
    let glyphs = [TextGlyph::try_new(1, 1.0, 2.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(41).named("frame planning text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        bounds,
    )
    .unwrap();
    scene.text_run(run);
}

pub(super) fn opaque_planning_mask(size: PhysicalSize) -> ResolvedLayerAlphaMask {
    let byte_len = usize::try_from(size.width())
        .unwrap()
        .checked_mul(usize::try_from(size.height()).unwrap())
        .and_then(|pixels| pixels.checked_mul(4))
        .unwrap();
    resolved_layer_alpha_mask_from_buffer(ImageBuffer::try_new(size, vec![255; byte_len]).unwrap())
}

pub(super) fn bounded_planning_backdrop() -> Layer {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap();
    Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap()
}

pub(super) fn assert_premultiplied(pixel: PremultipliedRgba8) {
    assert!(pixel.red() <= pixel.alpha());
    assert!(pixel.green() <= pixel.alpha());
    assert!(pixel.blue() <= pixel.alpha());
}

pub(super) fn pixel_alpha(image: &ImageBuffer, x: u32, y: u32) -> u8 {
    pixel_rgba(image, x, y)[3]
}

pub(super) fn pixel_rgba(image: &ImageBuffer, x: u32, y: u32) -> [u8; 4] {
    let index = ((y * image.size().width() + x) * 4 + 3) as usize;
    [
        image.rgba()[index - 3],
        image.rgba()[index - 2],
        image.rgba()[index - 1],
        image.rgba()[index],
    ]
}

pub(super) fn box_decoration_edges(
    top: BorderSide,
    right: BorderSide,
    bottom: BorderSide,
    left: BorderSide,
) -> BorderEdges {
    BorderEdges::new(top, right, bottom, left)
}

pub(super) fn solid_border(width: f64, color: Color) -> BorderSide {
    BorderSide::try_new(BorderStyle::Solid, width, color).unwrap()
}

pub(super) fn assert_finite_positive_rect(rect: Rect) {
    assert!(rect.x().is_finite());
    assert!(rect.y().is_finite());
    assert!(rect.width().is_finite());
    assert!(rect.height().is_finite());
    assert!(rect.width() > 0.0);
    assert!(rect.height() > 0.0);
}
