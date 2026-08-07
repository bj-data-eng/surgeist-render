use std::sync::Arc;

use crate::{
    Antialiasing, BackdropCaptureBounds, BackdropFilterInput, BlendMode, BorderEdges, BorderSide,
    BorderStyle, Capabilities, ClipInput, Color, DeviceLossReason, Error, ErrorCode, Extend,
    FilterAmount, FilterAngle, FilterBlur, FilterDropShadow, FilterList, FilterOp, FontData,
    FontRef, Image, ImageBuffer, ImageFit, ImageQuality, Layer, Options, Parameters, PhysicalSize,
    Point, Rect, RenderSurfaceAvailability, Renderer, ResolvedLayerAlphaMask,
    RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation, Scene,
    Shape, Size, Stats, Stroke, Surface, TextGlyph, TextPaint, TextRun, TextRunBounds, Transform,
    UnitFilterAmount,
    backend::DeviceSignal,
    command,
    filter::BlurPolicy,
    reference::{self, PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    resource::{ResourceCacheKey, WorkingFormat},
    texture::EffectTextureDescriptor,
    vello_engine::{
        PreparedVelloPass, RasterParameters,
        scene::{VelloRasterScenario, VelloScene},
    },
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
    let image = Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        ImageBuffer::try_new(size, vec![255; byte_len])
            .unwrap()
            .into_rgba(),
    )
    .unwrap();
    ResolvedLayerAlphaMask::try_new(
        image,
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

pub(super) fn modeled_resource_key_for_test(discriminator: u32) -> ResourceCacheKey {
    let descriptor = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(discriminator.max(1), 1),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    ResourceCacheKey::EffectTexture(descriptor.cache_key())
}

pub(super) fn explicit_graph_transaction_inputs_for_test(
    renderer: &mut Renderer,
) -> (wgpu::Device, wgpu::Queue, Arc<DeviceSignal>) {
    let (device, queue) = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the explicit transaction harness requires a ready device");
        (
            ready.device_for_test().clone(),
            ready.queue_for_test().clone(),
        )
    };
    let signal = renderer
        .default_device_signal_for_test()
        .expect("the explicit transaction harness requires its device signal");
    (device, queue, signal)
}

pub(super) fn bounded_backdrop_integration_fixture_for_test() -> (
    Scene,
    PhysicalSize,
    Parameters,
    ReferencePremultipliedRgba8Buffer,
) {
    let size = PhysicalSize::new(8, 6);
    let base = [32, 64, 96, 255];
    let prior = [224, 48, 24, 255];
    let foreground = [240, 224, 32, 255];
    let later = [32, 224, 96, 160];
    let invert = UnitFilterAmount::try_new(1.0).unwrap();
    let blur = FilterBlur::try_new(0.75).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::blur(blur),
    ])
    .unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test(prior),
        )
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(3.0, 2.0, 2.0, 2.0),
                color_from_straight_rgba8_for_test(foreground),
            );
        })
        .fill(
            Rect::new(5.0, 1.0, 2.0, 4.0),
            color_from_straight_rgba8_for_test(later),
        );
    let reference_rect = |rect: (u32, u32, u32, u32), straight: [u8; 4]| {
        let mut buffer = ReferencePremultipliedRgba8Buffer::try_new(size).unwrap();
        let pixel = PremultipliedRgba8::try_new(
            premultiply_u8_channel_for_test(straight[0], straight[3]),
            premultiply_u8_channel_for_test(straight[1], straight[3]),
            premultiply_u8_channel_for_test(straight[2], straight[3]),
            straight[3],
        )
        .unwrap();
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                buffer.set_pixel(x, y, pixel).unwrap();
            }
        }
        buffer
    };
    let parent = reference_rect((0, 0, 8, 6), base);
    let parent = reference_rect((0, 1, 3, 4), prior)
        .source_over(&parent)
        .unwrap();
    let invert_pipeline = FilterList::try_ops(vec![FilterOp::invert(invert)])
        .unwrap()
        .color_filter_pipeline()
        .unwrap()
        .unwrap();
    let filtered = parent
        .apply_color_filter_pipeline(&invert_pipeline)
        .and_then(|buffer| {
            buffer.apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        })
        .unwrap();
    let group = reference_rect((3, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let completed = group.source_over(&parent).unwrap();
    let expected = reference_rect((5, 1, 2, 4), later)
        .source_over(&completed)
        .unwrap();
    (
        scene,
        size,
        Parameters {
            base_color: color_from_straight_rgba8_for_test(base),
            ..Parameters::default()
        },
        expected,
    )
}

pub(super) fn spatial_filter_maximum_error_for_test(
    actual: &[u8],
    expected: &[u8],
    working_format: WorkingFormat,
) -> (u8, u8) {
    match working_format {
        WorkingFormat::HighPrecision => {
            let maximum = if actual.len() == expected.len() && actual.len().is_multiple_of(4) {
                actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
                    0,
                    |maximum, (actual, expected)| {
                        let expected_rgb = if actual[3] == 0 {
                            [0, 0, 0]
                        } else {
                            [expected[0], expected[1], expected[2]]
                        };
                        maximum
                            .max(actual[0].abs_diff(expected_rgb[0]))
                            .max(actual[1].abs_diff(expected_rgb[1]))
                            .max(actual[2].abs_diff(expected_rgb[2]))
                            .max(actual[3].abs_diff(expected[3]))
                    },
                )
            } else {
                u8::MAX
            };
            (maximum, 0)
        }
        WorkingFormat::ReducedPrecision => {
            if actual.len() == expected.len() {
                actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
                    (0, 0),
                    |(max_alpha, max_premul), (actual, expected)| {
                        let alpha = max_alpha.max(actual[3].abs_diff(expected[3]));
                        let premul = (0..3).fold(max_premul, |maximum, channel| {
                            maximum.max(
                                premultiply_u8_channel_for_test(actual[channel], actual[3])
                                    .abs_diff(premultiply_u8_channel_for_test(
                                        expected[channel],
                                        expected[3],
                                    )),
                            )
                        });
                        (alpha, premul)
                    },
                )
            } else {
                (u8::MAX, u8::MAX)
            }
        }
    }
}

pub(super) fn spatial_filter_mixed_filter_fixture_for_test()
-> (Scene, Vec<FilterList>, PhysicalSize, Vec<u8>) {
    let size = PhysicalSize::new(15, 13);
    let mut source = ReferencePremultipliedRgba8Buffer::try_new(size).unwrap();
    for (x, y, pixel) in [
        (5, 5, PremultipliedRgba8::try_new(224, 64, 16, 255).unwrap()),
        (6, 5, PremultipliedRgba8::try_new(32, 192, 96, 255).unwrap()),
        (6, 6, PremultipliedRgba8::try_new(48, 80, 240, 255).unwrap()),
    ] {
        source.set_pixel(x, y, pixel).unwrap();
    }
    let blur = FilterBlur::try_new(0.75).unwrap();
    let shadow = FilterDropShadow::try_new(
        Point::new(-1.25, 0.5),
        FilterBlur::try_new(0.5).unwrap(),
        Color::try_rgba(0.25, 0.5, 0.75, 0.625).unwrap(),
    )
    .unwrap();
    let invert = UnitFilterAmount::try_new(0.25).unwrap();
    let opacity = UnitFilterAmount::try_new(0.8).unwrap();
    let invert_pipeline = FilterList::try_ops(vec![FilterOp::invert(invert)])
        .unwrap()
        .color_filter_pipeline()
        .unwrap()
        .unwrap();
    let opacity_pipeline = FilterList::try_ops(vec![FilterOp::opacity(opacity)])
        .unwrap()
        .color_filter_pipeline()
        .unwrap()
        .unwrap();
    let expected = source
        .apply_color_filter_pipeline(&invert_pipeline)
        .and_then(|buffer| buffer.apply_blur(blur, BlurPolicy::css_filter_default()))
        .and_then(|buffer| {
            buffer.apply_fractional_drop_shadow_for_gpu_oracle(
                &shadow,
                BlurPolicy::css_filter_default(),
            )
        })
        .and_then(|buffer| buffer.apply_color_filter_pipeline(&opacity_pipeline))
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let image = Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        reference_straight_bytes_for_test(&source),
    )
    .expect("the spatial-filter pixel fixture must form one RGBA image");
    let mut scene = Scene::new();
    scene.image(
        image,
        Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height())),
        ImageFit::Stretch,
    );
    let filters = vec![
        FilterList::try_ops(vec![
            FilterOp::invert(invert),
            FilterOp::blur(blur),
            FilterOp::drop_shadow(shadow),
            FilterOp::opacity(opacity),
        ])
        .unwrap(),
    ];
    (scene, filters, size, expected)
}

pub(super) fn assert_runtime_device_lost(
    error: Error,
    operation: RuntimeOperation,
    reason: DeviceLossReason,
) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::DeviceLost { reason },
            )
            .unwrap()
        )
    );
}

pub(super) fn assert_surface_unavailable(
    error: Error,
    operation: RuntimeOperation,
    state: RenderSurfaceAvailability,
) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable { state },
            )
            .unwrap()
        )
    );
}

pub(super) fn headless_direct_publication_fixture_for_test()
-> (Renderer, Surface, Scene, ImageBuffer) {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut first = Scene::new();
    first.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &first, Parameters::default()))
        .expect("the first frame must establish a readable publication");
    let published = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first frame publication must be readable");

    let mut replacement = Scene::new();
    replacement.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    );

    (renderer, surface, replacement, published)
}

pub(super) fn prepared_direct_vello_pass_for_test(
    target_extent: PhysicalSize,
) -> PreparedVelloPass {
    VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("the explicit direct Vello target must be non-empty"),
    )
    .expect("the explicit direct Vello scene must prepare without submission authority")
}

pub(super) fn graph_canonical_pixel_for_test(pixel: [u8; 4]) -> [u8; 4] {
    if pixel[3] == 0 { [0, 0, 0, 0] } else { pixel }
}

pub(super) fn premultiply_u8_channel_for_test(color: u8, alpha: u8) -> u8 {
    ((u16::from(color) * u16::from(alpha) + 127) / 255) as u8
}

pub(super) fn default_graph_working_format_for_test(renderer: &mut Renderer) -> WorkingFormat {
    let precisions = renderer
        .default_device_capabilities_for_test()
        .effect_precisions();
    if precisions.supports_high_precision() {
        WorkingFormat::HighPrecision
    } else {
        assert!(
            precisions.supports_reduced_precision(),
            "graph pixel tests require one real supported working format"
        );
        WorkingFormat::ReducedPrecision
    }
}

pub(super) const COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X: i32 = -2;

pub(super) fn color_filter_retention_fixture_for_test() -> (Scene, Vec<FilterList>, Vec<u8>) {
    let source = [
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ];
    let expected = source
        .iter()
        .flat_map(|pixel| [255 - pixel[0], 255 - pixel[1], 255 - pixel[2], pixel[3]])
        .collect();
    let hidden_prefix = [[17, 31, 47, 255], [233, 199, 151, 127]];
    let bytes = hidden_prefix
        .into_iter()
        .chain(source.iter().copied())
        .flat_map(|pixel| pixel.into_iter())
        .collect::<Vec<_>>();
    let source_width = u32::try_from(source.len() + hidden_prefix.len())
        .expect("the color-filter pixel vector must fit u32");
    let image = Image::from_rgba(
        Size::new(f64::from(source_width), 1.0),
        Arc::<[u8]>::from(bytes),
    )
    .expect("the color-filter pixel vector must form one valid image");
    let mut scene = Scene::new();
    scene.image(
        image,
        Rect::new(
            f64::from(COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X),
            0.0,
            f64::from(source_width),
            1.0,
        ),
        ImageFit::Stretch,
    );
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    (scene, vec![filters], expected)
}

pub(super) fn composition_mask_image_from_alpha_for_test(
    size: PhysicalSize,
    alpha: &[u8],
    quality: ImageQuality,
    extend: Extend,
) -> Image {
    let pixel_count = usize::try_from(size.width())
        .unwrap()
        .checked_mul(usize::try_from(size.height()).unwrap())
        .unwrap();
    assert_eq!(alpha.len(), pixel_count);
    let bytes = alpha
        .iter()
        .copied()
        .flat_map(|alpha| [17, 211, 93, alpha])
        .collect::<Vec<_>>();
    Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        bytes,
    )
    .unwrap()
    .quality(quality)
    .extend(extend)
}

pub(super) fn reference_solid_for_test(
    size: PhysicalSize,
    straight: [u8; 4],
) -> ReferencePremultipliedRgba8Buffer {
    let pixel_count = usize::try_from(size.width())
        .unwrap()
        .checked_mul(usize::try_from(size.height()).unwrap())
        .unwrap();
    let pixel = PremultipliedRgba8::try_new(
        premultiply_u8_channel_for_test(straight[0], straight[3]),
        premultiply_u8_channel_for_test(straight[1], straight[3]),
        premultiply_u8_channel_for_test(straight[2], straight[3]),
        straight[3],
    )
    .unwrap();
    ReferencePremultipliedRgba8Buffer::from_pixels(size, vec![pixel; pixel_count]).unwrap()
}

pub(super) fn reference_straight_bytes_for_test(
    buffer: &ReferencePremultipliedRgba8Buffer,
) -> Vec<u8> {
    reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(buffer)
        .unwrap()
        .into_rgba()
}

pub(super) fn color_from_straight_rgba8_for_test(straight: [u8; 4]) -> Color {
    Color::try_rgba(
        f32::from(straight[0]) / 255.0,
        f32::from(straight[1]) / 255.0,
        f32::from(straight[2]) / 255.0,
        f32::from(straight[3]) / 255.0,
    )
    .unwrap()
}

pub(super) fn graph_pixels_match_for_test(
    actual: &[u8],
    expected: &[u8],
    working_format: WorkingFormat,
    tolerance: u8,
) -> bool {
    actual.len() == expected.len()
        && actual
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .all(|(actual, expected)| match working_format {
                WorkingFormat::HighPrecision => actual
                    .iter()
                    .copied()
                    .zip(expected.iter().copied())
                    .all(|(actual, expected)| actual.abs_diff(expected) <= tolerance),
                WorkingFormat::ReducedPrecision => {
                    actual[3].abs_diff(expected[3]) <= tolerance
                        && (0..3).all(|channel| {
                            premultiply_u8_channel_for_test(actual[channel], actual[3]).abs_diff(
                                premultiply_u8_channel_for_test(expected[channel], expected[3]),
                            ) <= tolerance
                        })
                }
            })
}

#[cfg(feature = "render-window")]
pub(super) fn composition_presented_masked_blended_scene_for_test(rect: Rect) -> Scene {
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[160],
        ImageQuality::Low,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .blend(BlendMode::Multiply)
            .with_resolved_alpha_mask(ResolvedLayerAlphaMask::try_new(mask, rect).unwrap()),
        |scene| {
            scene.fill(rect, color_from_straight_rgba8_for_test([224, 64, 32, 192]));
        },
    );
    scene
}

pub(super) fn repeated_graph_scene_for_test() -> Scene {
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.25, 0.5, 5.5, 3.75),
            Color::try_rgba(0.75, 0.25, 0.125, 0.625).unwrap(),
        )
        .stroke(
            Shape::rect(Rect::new(1.0, 1.0, 4.0, 3.0)),
            Stroke::try_new(0.75).unwrap(),
            Color::BLACK,
        );
    scene
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GraphPublicStatsForTest {
    pub(super) commands: usize,
    pub(super) fills: usize,
    pub(super) strokes: usize,
    pub(super) shadows: usize,
    pub(super) images: usize,
    pub(super) glyphs: usize,
    pub(super) layers: usize,
    pub(super) cache_hits: usize,
    pub(super) cache_misses: usize,
    pub(super) uploaded_bytes: u64,
}

impl From<Stats> for GraphPublicStatsForTest {
    fn from(stats: Stats) -> Self {
        Self {
            commands: stats.commands,
            fills: stats.fills,
            strokes: stats.strokes,
            shadows: stats.shadows,
            images: stats.images,
            glyphs: stats.glyphs,
            layers: stats.layers,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            uploaded_bytes: stats.uploaded_bytes,
        }
    }
}

pub(super) fn graph_transform_point_for_test(transform: Transform, point: Point) -> Point {
    let [a, b, c, d, e, f] = transform.as_array();
    Point::new(
        a * point.x() + c * point.y() + e,
        b * point.x() + d * point.y() + f,
    )
}

pub(super) fn graph_supported_working_formats_for_test(
    renderer: &mut Renderer,
) -> Vec<WorkingFormat> {
    let precision = renderer
        .default_device_capabilities_for_test()
        .effect_precisions();
    let mut formats = Vec::with_capacity(2);
    if precision.supports_high_precision() {
        formats.push(WorkingFormat::HighPrecision);
    }
    if precision.supports_reduced_precision() {
        formats.push(WorkingFormat::ReducedPrecision);
    }
    assert!(
        !formats.is_empty(),
        "direct-graph parity requires at least one real supported working format"
    );
    formats
}
