use crate::{
    Antialiasing, BackdropCaptureBounds, BackdropFilterInput, BlendMode, BorderEdges, BorderSide,
    BorderStyle, Capabilities, ClipInput, Color, FilterAmount, FilterAngle, FilterBlur,
    FilterDropShadow, FilterList, FilterOp, FontData, FontRef, Image, ImageBuffer, Layer,
    PhysicalSize, Point, Rect, ResolvedLayerAlphaMask, Scene, Shape, Size, Stroke, TextGlyph,
    TextPaint, TextRun, TextRunBounds, Transform, UnitFilterAmount, command,
    reference::PremultipliedRgba8,
};

pub(super) const AHEM_FONT_BYTES: &[u8] =
    include_bytes!("../../tests/fixtures/fonts/ahem/Ahem.ttf");
const AHEM_FONT_ID: u64 = 9001;
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
    ResolvedLayerAlphaMask::try_new(
        image_from_buffer(ImageBuffer::try_new(size, vec![255; byte_len]).unwrap()),
        Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height())),
    )
    .unwrap()
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

pub(super) fn filter_graph_commands_for_test() -> command::RenderCommands {
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(-2.25, 1.5, 4.0, 3.0),
        Color::try_rgba(0.5, 0.25, 0.75, 0.625).unwrap(),
    );
    scene
        .normalize(Capabilities::CURRENT)
        .expect("ordinary color-filter capture input must normalize")
}

pub(super) fn authored_color_filter_runs_for_test() -> Vec<FilterList> {
    vec![
        FilterList::try_ops(vec![
            FilterOp::brightness(FilterAmount::try_new(1.25).unwrap()),
            FilterOp::contrast(FilterAmount::try_new(0.75).unwrap()),
            FilterOp::invert(UnitFilterAmount::try_new(0.25).unwrap()),
        ])
        .unwrap(),
        FilterList::try_ops(vec![
            FilterOp::hue_rotate(FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap()),
            FilterOp::opacity(UnitFilterAmount::try_new(0.625).unwrap()),
            FilterOp::sepia(UnitFilterAmount::try_new(0.5).unwrap()),
        ])
        .unwrap(),
    ]
}

pub(super) fn filter_graph_context_for_test() -> crate::frame::FrameContext {
    crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.25,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap()
}

pub(super) fn color_then_blur_filters_for_test() -> Vec<FilterList> {
    vec![
        authored_color_filter_runs_for_test()[0].clone(),
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap(),
    ]
}

pub(super) fn spatial_filter_authored_filter_steps_for_test() -> Vec<FilterList> {
    vec![
        authored_color_filter_runs_for_test()[0].clone(),
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.25).unwrap())]).unwrap(),
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(0.0).unwrap())]).unwrap(),
        FilterList::try_ops(vec![FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-1.5, 0.75),
                FilterBlur::try_new(0.625).unwrap(),
                Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
            )
            .unwrap(),
        )])
        .unwrap(),
        authored_color_filter_runs_for_test()[1].clone(),
    ]
}

pub(super) fn bounded_backdrop_graph_commands_for_test() -> command::RenderCommands {
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.25).unwrap()),
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-1.25, 0.75),
                FilterBlur::try_new(0.5).unwrap(),
                Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
            )
            .unwrap(),
        ),
    ])
    .unwrap();
    let backdrop_clip = ClipInput::try_shape(Shape::rect(Rect::new(0.5, 0.5, 7.0, 5.0))).unwrap();
    let backdrop = Layer::new()
        .try_clip(Shape::rect(Rect::new(0.25, 0.25, 7.5, 5.5)))
        .unwrap()
        .try_opacity(0.75)
        .unwrap()
        .blend(BlendMode::Screen)
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                Some(backdrop_clip),
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 8.0, 6.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(1.0, 1.0, 2.0, 2.0),
                Color::try_rgba(1.0, 0.0, 0.0, 0.5).unwrap(),
            );
        })
        .fill(
            Rect::new(6.0, 4.0, 1.0, 1.0),
            Color::try_rgba(0.0, 1.0, 0.0, 0.75).unwrap(),
        );
    scene
        .normalize(Capabilities::CURRENT)
        .expect("the exact bounded-backdrop fixture must normalize")
}

pub(super) fn runtime_lowering_commands_for_test() -> command::RenderCommands {
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.25).unwrap()),
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-1.25, 0.75),
                FilterBlur::try_new(0.5).unwrap(),
                Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
            )
            .unwrap(),
        ),
    ])
    .unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let masked =
        Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4)));
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 8.0, 6.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(1.0, 1.0, 2.0, 2.0),
                Color::try_rgba(1.0, 0.0, 0.0, 0.5).unwrap(),
            );
        })
        .layer(masked, |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 4.0, 4.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        });
    scene
        .normalize(Capabilities::CURRENT)
        .expect("the runtime lowering fixture must normalize")
}

pub(super) fn composition_commands_for_test() -> command::RenderCommands {
    let outer_mask = opaque_planning_mask(PhysicalSize::new(4, 4));
    let inner_mask = opaque_planning_mask(PhysicalSize::new(4, 4));
    let outer_clip_transform = Transform::translation(0.5, 0.25).unwrap();
    let inner_clip_transform = Transform::translation(0.25, 0.5).unwrap();
    let outer_transform = Transform::translation(1.0, 0.5).unwrap();
    let inner_transform = Transform::scale(0.75, 0.5).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)))
            .unwrap()
            .try_transform(outer_clip_transform)
            .unwrap(),
        |scene| {
            scene.layer(
                Layer::new()
                    .try_clip(Shape::rect(Rect::new(0.125, 0.125, 3.75, 3.75)))
                    .unwrap()
                    .try_transform(inner_clip_transform)
                    .unwrap(),
                |scene| {
                    scene.layer(
                        Layer::new()
                            .try_clip(Shape::rect(Rect::new(0.25, 0.25, 3.5, 3.5)))
                            .unwrap()
                            .try_transform(outer_transform)
                            .unwrap()
                            .try_opacity(1.5)
                            .unwrap()
                            .blend(BlendMode::Screen)
                            .with_resolved_alpha_mask(outer_mask),
                        |scene| {
                            scene.layer(
                                Layer::new()
                                    .try_clip(Shape::rect(Rect::new(0.5, 0.5, 2.5, 2.5)))
                                    .unwrap()
                                    .try_transform(inner_transform)
                                    .unwrap()
                                    .try_opacity(0.25)
                                    .unwrap()
                                    .blend(BlendMode::Multiply)
                                    .with_resolved_alpha_mask(inner_mask),
                                |scene| {
                                    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
                                },
                            );
                        },
                    );
                },
            );
        },
    );
    scene.normalize(Capabilities::CURRENT).unwrap()
}

pub(super) fn graph_shader_commands_for_test() -> command::RenderCommands {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(-1.25, -0.75, 2.0, 1.5), Color::BLACK)
        .stroke(
            Shape::rect(Rect::new(2.0, 1.0, 3.0, 2.0)),
            Stroke::try_new(0.5).unwrap(),
            Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
        );
    scene.normalize(Capabilities::CURRENT).unwrap()
}

pub(super) fn graph_shader_frame_context_for_test() -> crate::frame::FrameContext {
    crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap()
}
