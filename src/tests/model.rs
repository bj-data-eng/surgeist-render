use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use proptest::{prelude::any, prop_assert, prop_assert_eq, proptest};

use crate::{
    BlendMode, Capabilities, Color, CoordinateSpaceId, CoordinateSpaceKind, CoordinateSpaceTag,
    Dash, DegradedQuality, DegradedQualityKind, DeviceLossReason, EffectPrecision,
    EffectQualityPolicy, Error, ErrorCode, Extend, FillRule, FilledPath, Filter, FontData, FontId,
    FontRef, Format, GpuFaultKind, Gradient, GradientStop, HitTestOwnership, Image, ImageBuffer,
    ImageFit, ImageQuality, InvalidValue, Layer, NormalizedPaintLayer, Paint, PaintColor, Path,
    PathElement, PhysicalSize, Point, PrimitiveFamily, PrimitiveOperation, Radii, Rect,
    RenderRoute, RenderSurfaceAvailability, ResolvedLayerAlphaMask, RuntimeCapabilityUnavailable,
    RuntimeCapabilityUnavailableReason, RuntimeOperation, Scene, Shadow, ShadowList, Shape, Size,
    Stats, Stroke, StrokeAlign, SurfaceIdentityMismatchKind, SymbolicColorPolicy,
    TextDecorationLine, TextDecorationLineStyle, TextGlyph, TextPaint, TextRun, TextRunBounds,
    TextRunBoundsKind, TextShadowRun, Transform, UnresolvedResource, UnresolvedResourceKind,
    UnsupportedPrimitive,
};

use crate::{
    command,
    error::BackendErrorCode,
    reference,
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    scene,
    validation::validate_rect,
    vello_engine::{
        glyph::{BitmapSourceForTest, SelectedGlyphTrace, preflight_selected_glyphs},
        scene::VelloScene,
    },
};

use super::{
    UnwrapOrPanicForTest,
    support::{
        AHEM_FONT_BYTES, AHEM_GLYPH_ASCENT_E_ACUTE, AHEM_GLYPH_DESCENT_P, AHEM_GLYPH_X, ahem_font,
        assert_premultiplied, pixel_alpha, text_run_for,
    },
};

const AHEM_FONT_ID: u64 = 9001;

#[test]
fn scene_encoding_is_deterministic() {
    let mut a = Scene::new();
    let mut b = Scene::new();
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    a.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::try_new(1.0).unwrap(), Color::BLACK);
    b.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::try_new(1.0).unwrap(), Color::BLACK);

    assert_eq!(a, b);
}

#[test]
fn scene_stats_report_facts_without_renderer() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255])).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .stroke(
            Rect::new(1.0, 1.0, 2.0, 2.0),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        )
        .shadow(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Shadow::try_new(Point::new(0.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        )
        .image(image, Rect::new(0.0, 0.0, 1.0, 1.0), ImageFit::Stretch)
        .layer(Layer::new(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });

    let stats = scene.stats();

    assert_eq!(stats.commands, 6);
    assert_eq!(stats.fills, 2);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.shadows, 1);
    assert_eq!(stats.images, 1);
    assert_eq!(stats.layers, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn scene_normalization_preserves_stats() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.stroke(
                Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let stats = normalized.stats();

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
}

#[test]
fn paint_colors_convert_srgb_to_concrete_rgba() {
    let color = PaintColor::try_srgb(0.25, 0.5, 0.75, 0.8)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(color, Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap());
}

#[test]
fn paint_colors_convert_hsl_known_vectors() {
    let red = PaintColor::try_hsl(0.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();
    let cyan = PaintColor::try_hsl(180.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(red, Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap());
    assert_eq!(cyan, Color::try_rgba(0.0, 1.0, 1.0, 1.0).unwrap());
}

#[test]
fn paint_colors_reject_invalid_conversion_inputs() {
    assert!(PaintColor::try_srgb(f32::NAN, 0.0, 0.0, 1.0).is_err());
    assert!(PaintColor::try_hsl(f32::NAN, 1.0, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.5, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, -0.1, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, 0.5, f32::INFINITY).is_err());
}

#[test]
fn normalized_paint_layers_preserve_valid_paint_sources() {
    let color = NormalizedPaintLayer::try_new(Paint::from(Color::BLACK)).unwrap();
    let gradient_paint = Paint::from(
        Gradient::try_linear(
            Point::try_new(0.0, 0.0).unwrap(),
            Point::try_new(10.0, 0.0).unwrap(),
            vec![
                GradientStop::try_new(0.0, Color::BLACK).unwrap(),
                GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
            ],
        )
        .unwrap(),
    );
    let gradient = NormalizedPaintLayer::try_new(gradient_paint.clone()).unwrap();

    assert_eq!(color.paint(), &Paint::from(Color::BLACK));
    assert_eq!(gradient.paint(), &gradient_paint);
}

#[test]
fn gradients_reject_non_finite_geometry_before_paint_layer_construction() {
    let error = Gradient::try_linear(
        Point::new(f64::NAN, 0.0),
        Point::try_new(1.0, 0.0).unwrap(),
        vec![GradientStop::try_new(0.0, Color::BLACK).unwrap()],
    )
    .expect_err("invalid gradient construction should fail before paint layer");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn gradients_expose_render_ready_geometry_and_stops() {
    let stops = vec![
        GradientStop::try_new(0.0, Color::BLACK).unwrap(),
        GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
    ];
    let linear = Gradient::try_linear(
        Point::try_new(1.0, 2.0).unwrap(),
        Point::try_new(3.0, 4.0).unwrap(),
        stops.clone(),
    )
    .unwrap();
    let radial =
        Gradient::try_radial(Point::try_new(5.0, 6.0).unwrap(), 7.0, stops.clone()).unwrap();
    let sweep = Gradient::try_sweep(Point::try_new(8.0, 9.0).unwrap(), stops.clone()).unwrap();

    assert_eq!(linear.stops(), stops.as_slice());
    assert_eq!(
        linear.linear_points(),
        Some((
            Point::try_new(1.0, 2.0).unwrap(),
            Point::try_new(3.0, 4.0).unwrap()
        ))
    );
    assert_eq!(
        radial.radial_geometry(),
        Some((Point::try_new(5.0, 6.0).unwrap(), 7.0))
    );
    assert_eq!(
        sweep.sweep_center(),
        Some(Point::try_new(8.0, 9.0).unwrap())
    );
}

#[test]
fn gradients_preserve_transparent_stops() {
    let stop = GradientStop::try_new(0.5, Color::TRANSPARENT).unwrap();

    assert_eq!(stop.color(), Color::TRANSPARENT);
}

#[test]
fn paths_expose_authored_elements() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .close();

    assert_eq!(path.elements().len(), 3);
    assert!(matches!(path.elements()[0], PathElement::MoveTo(_)));
}

#[test]
fn filled_paths_preserve_fill_rule_intent() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 4.0).unwrap())
        .close();
    let filled = FilledPath::try_new(path.clone(), FillRule::EvenOdd).unwrap();

    assert_eq!(filled.path(), &path);
    assert_eq!(filled.fill_rule(), FillRule::EvenOdd);
}

#[test]
fn filled_paths_reject_invalid_path_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = FilledPath::try_new(path, FillRule::NonZero)
        .expect_err("filled paths validate stored path elements");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn geometry_try_constructors_reject_invalid_values() {
    assert!(Point::try_new(f64::NAN, 0.0).is_err());
    assert!(Size::try_new(-1.0, 1.0).is_err());
    assert!(Rect::try_new(0.0, 0.0, 1.0, f64::INFINITY).is_err());
    assert!(Radii::try_all(-0.1).is_err());
    assert!(Transform::try_new([1.0, 0.0, 0.0, f64::NAN, 0.0, 0.0]).is_err());
}

#[test]
fn rect_constructor_rejects_non_finite_derived_maxima() {
    let cases = [
        (
            "x maximum",
            (f64::MAX, 0.0, f64::MAX, 1.0),
            "rectangle max x",
        ),
        (
            "y maximum",
            (0.0, f64::MAX, 1.0, f64::MAX),
            "rectangle max y",
        ),
    ];

    for (case, (x, y, width, height), expected_field) in cases {
        let error = match Rect::try_new(x, y, width, height) {
            Ok(rect) => panic!(
                "{case} overflow unexpectedly constructed a rectangle with max {:?}",
                rect.max()
            ),
            Err(error) => error,
        };

        assert_eq!(error.code(), ErrorCode::InvalidInput, "{case}");
        assert_eq!(
            error.invalid_value_diagnostic().map(InvalidValue::field),
            Some(expected_field),
            "{case}"
        );
    }
}

#[test]
fn canonical_rect_validation_rejects_non_finite_derived_maxima() {
    let cases = [
        (
            "x maximum",
            Rect::new(f64::MAX, 0.0, f64::MAX, 1.0),
            "canonical rectangle max x",
        ),
        (
            "y maximum",
            Rect::new(0.0, f64::MAX, 1.0, f64::MAX),
            "canonical rectangle max y",
        ),
    ];

    for (case, rect, expected_field) in cases {
        let error = validate_rect(rect, "canonical rectangle")
            .expect_err("canonical validation must reject a non-finite derived maximum");

        assert_eq!(error.code(), ErrorCode::InvalidInput, "{case}");
        assert_eq!(
            error.invalid_value_diagnostic().map(InvalidValue::field),
            Some(expected_field),
            "{case}"
        );
    }
}

#[test]
fn rect_constructor_accepts_finite_and_zero_area_boundaries() {
    let cases = [
        ("finite maximum", (-f64::MAX, -1.0, f64::MAX, 1.0)),
        ("zero-area maximum", (f64::MAX, f64::MAX, 0.0, 0.0)),
    ];

    for (case, (x, y, width, height)) in cases {
        let rect = Rect::try_new(x, y, width, height)
            .unwrap_or_else(|error| panic!("{case} should remain valid: {error:?}"));
        let max = rect.max();

        assert!(max.x().is_finite(), "{case} x maximum");
        assert!(max.y().is_finite(), "{case} y maximum");
    }
}

#[test]
fn transform_helpers_preserve_affine_coefficients() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 4.0).unwrap();
    let rotate = Transform::rotation(std::f64::consts::FRAC_PI_2).unwrap();

    assert_eq!(translate.as_array(), [1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(scale.as_array(), [2.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
    assert!(rotate.as_array()[0].abs() < 1.0e-12);
    assert!((rotate.as_array()[1] - 1.0).abs() < 1.0e-12);
    assert!((rotate.as_array()[2] + 1.0).abs() < 1.0e-12);
    assert!(rotate.as_array()[3].abs() < 1.0e-12);
}

#[test]
fn transform_skew_helpers_preserve_tangent_coefficients() {
    let skew_x = Transform::skew_x(std::f64::consts::FRAC_PI_4).unwrap();
    let skew_y = Transform::skew_y(std::f64::consts::FRAC_PI_4).unwrap();

    assert!((skew_x.as_array()[2] - 1.0).abs() < 1.0e-12);
    assert!((skew_y.as_array()[1] - 1.0).abs() < 1.0e-12);
}

#[test]
fn transform_helpers_reject_non_finite_inputs() {
    assert!(Transform::translation(f64::NAN, 0.0).is_err());
    assert!(Transform::scale(1.0, f64::INFINITY).is_err());
    assert!(Transform::rotation(f64::NAN).is_err());
    assert!(Transform::skew_x(f64::INFINITY).is_err());
    assert!(Transform::skew_y(f64::NAN).is_err());
}

#[test]
fn transform_then_composes_in_application_order() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 2.0).unwrap();
    let composed = translate.then(scale).unwrap();

    assert_eq!(composed.as_array(), [2.0, 0.0, 0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn transform_around_wraps_transform_origin() {
    let origin = Point::try_new(10.0, 5.0).unwrap();
    let transform = Transform::scale(2.0, 3.0).unwrap().around(origin).unwrap();

    assert_eq!(transform.as_array(), [2.0, 0.0, 0.0, 3.0, -10.0, -10.0]);
}

#[test]
fn coordinate_space_tags_preserve_kind_and_transform() {
    let named = CoordinateSpaceId::try_new(7).unwrap();
    let transform = Transform::translation(3.0, 4.0).unwrap();
    let tag = CoordinateSpaceTag::try_new(CoordinateSpaceKind::Named(named), transform).unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Named(named));
    assert_eq!(tag.transform(), transform);
}

#[test]
fn coordinate_space_ids_reject_reserved_zero() {
    let error = CoordinateSpaceId::try_new(0).expect_err("zero is reserved");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("coordinate space id")
    );
}

#[test]
fn coordinate_space_viewport_tags_preserve_transforms() {
    let tag = CoordinateSpaceTag::viewport(Transform::translation(4.0, 6.0).unwrap()).unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Viewport);
    assert_eq!(tag.transform().as_array(), [1.0, 0.0, 0.0, 1.0, 4.0, 6.0]);
}

#[test]
fn rect_try_from_kurbo_rejects_invalid_bounds() {
    let rect = kurbo::Rect {
        x0: 1.0,
        y0: 0.0,
        x1: 0.0,
        y1: 1.0,
    };

    assert!(Rect::try_from(rect).is_err());
}

#[test]
fn physical_size_conversion_rejects_invalid_scale_and_device_overflow() {
    let cases = [
        ("zero scale", Size::try_new(10.0, 10.0).unwrap(), 0.0),
        (
            "device-pixel overflow",
            Size::try_new(f64::from(u32::MAX), 1.0).unwrap(),
            2.0,
        ),
    ];

    for (case, logical_size, scale) in cases {
        let error = PhysicalSize::try_from_logical(logical_size, scale)
            .expect_err("invalid physical-size conversion must be rejected");
        assert_eq!(error.code(), ErrorCode::InvalidInput, "{case}");
    }
}

#[test]
fn stats_default_exposes_no_route_precision_or_pass_activity() {
    fn assert_observation_traits<T: Clone + Copy + std::fmt::Debug + Eq + PartialEq>() {}

    assert_observation_traits::<RenderRoute>();
    assert_observation_traits::<EffectPrecision>();

    let stats = Stats::default();

    assert_eq!(stats.route, None);
    assert_eq!(stats.effect_precision, None);
    assert_eq!(stats.vello_passes, 0);
    assert_eq!(stats.image_passes, 0);
    assert_eq!(stats.composite_passes, 0);
    assert_eq!(stats.copy_operations, 0);
    assert_eq!(stats.custom_present_passes, 0);
    assert_eq!(stats.effect_texture_allocations, 0);
    assert_eq!(stats.effect_texture_reuses, 0);
    assert_eq!(stats.retained_effect_bytes, 0);
}

#[test]
fn colors_reject_non_finite_red_components_at_construction() {
    let error = Color::try_rgba(f32::NAN, 0.0, 0.0, 1.0)
        .expect_err("invalid paint should fail at construction");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.message().contains("red channel"));
}
#[test]
fn font_data_rejects_unreadable_bytes_and_out_of_range_collection_indices() {
    let cases = [
        (
            "malformed bytes",
            vec![0x00, 0x01, 0x02],
            7,
            "len=3, index=7".to_string(),
        ),
        (
            "out-of-range collection index",
            AHEM_FONT_BYTES.to_vec(),
            1,
            format!("len={}, index=1", AHEM_FONT_BYTES.len()),
        ),
    ];

    for (case, bytes, index, expected_value) in cases {
        let error = FontData::try_from_bytes(bytes, index)
            .expect_err("invalid font data must not construct FontData");

        assert_font_data_error(&error, &expected_value);
        assert_eq!(
            error.invalid_value_diagnostic().map(InvalidValue::field),
            Some("font_data"),
            "{case} returned the wrong typed diagnostic"
        );
    }
}

proptest! {
    #[test]
    fn font_data_constructor_never_panics_for_arbitrary_bytes_and_indices(
        bytes in proptest::collection::vec(any::<u8>(), 0..2048),
        index in any::<u32>(),
    ) {
        let expected_value = format!("len={}, index={index}", bytes.len());
        let outcome = catch_unwind(AssertUnwindSafe(|| FontData::try_from_bytes(bytes, index)));

        prop_assert!(outcome.is_ok());
        if let Ok(Err(error)) = outcome {
            let diagnostic = error
                .invalid_value_diagnostic()
                .expect("failed font construction must remain typed");
            prop_assert_eq!(diagnostic.field(), "font_data");
            prop_assert_eq!(diagnostic.value(), expected_value);
            prop_assert_eq!(
                diagnostic.invariant(),
                "must contain a readable OpenType font at the requested collection index"
            );
        }
    }
}

#[test]
fn selected_glyph_preflight_rejects_missing_outline_before_external_encoding() {
    let font_data = FontData::try_from_bytes(AHEM_FONT_BYTES.to_vec(), 0).unwrap();
    let glyphs = [TextGlyph::try_new(u32::MAX, 0.0, 16.0, 8.0).unwrap()];
    let run = text_run_for(font_data, 16.0, Transform::identity(), &glyphs);
    let mut scene = VelloScene::default();
    let error = scene
        .encode_text_run(&run)
        .expect_err("a nonexistent glyph must not reach Vello encoding");

    assert_missing_glyph_error(&error, u32::MAX);
    assert_no_glyph_encoding(&scene);
}

#[test]
fn selected_glyph_preflight_validates_exact_outline_draw_settings() {
    let font_data = FontData::try_from_bytes(AHEM_FONT_BYTES.to_vec(), 0).unwrap();
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 3.0, 19.0, 9.0).unwrap()];
    let transform = Transform::try_new([1.25, 0.0, 0.0, 1.25, 2.0, -3.0]).unwrap();
    let run = text_run_for(font_data, 19.5, transform, &glyphs);
    let mut scene = VelloScene::default();
    scene
        .encode_text_run(&run)
        .expect("a valid outline must reach Encoding");

    let observation = scene.observation_for_test();
    assert_eq!(observation.glyph_run_count_for_test(), 1);
    assert_eq!(observation.glyph_count_for_test(), 1);
    assert_eq!(observation.patch_count_for_test(), 1);
    assert_eq!(observation.normalized_coordinate_count_for_test(), 0);

    let glyph_run = observation
        .first_glyph_run_for_test()
        .expect("the Vello scene must retain the glyph-run facts");
    assert_eq!(glyph_run.font_collection_index_for_test(), 0);
    assert!(glyph_run.font_data_matches_for_test(AHEM_FONT_BYTES));
    assert_eq!(
        glyph_run.transform_components_for_test(),
        [1.25, 0.0, 0.0, 1.25, 2.0, -3.0]
    );
    assert!(!glyph_run.has_glyph_transform_for_test());
    assert!(!glyph_run.has_brush_transform_for_test());
    assert_eq!(glyph_run.font_size_for_test(), 19.5);
    assert_eq!(glyph_run.embolden_amount_for_test(), [0.0, 0.0]);
    assert!(!glyph_run.uses_hinting_for_test());
    assert_eq!(glyph_run.normalized_coordinate_range_for_test(), 0..0);
    assert_eq!(glyph_run.glyph_range_for_test(), 0..1);
    assert!(glyph_run.uses_nonzero_fill_for_test());

    let glyph = observation
        .first_glyph_for_test()
        .expect("the Vello scene must retain the selected glyph");
    assert_eq!(glyph.id_for_test(), AHEM_GLYPH_X);
    assert_eq!(glyph.x_for_test(), 3.0);
    assert_eq!(glyph.y_for_test(), 19.0);
}

#[test]
fn selected_glyph_preflight_validates_colr_palette_bitmap_and_png_inputs() {
    assert_colr_glyph_preflight_cases();
    assert_png_bitmap_glyph_preflight_cases();
    assert_malformed_sbix_glyph_preflight_cases();
    assert_malformed_head_glyph_preflight_cases();
}

fn assert_colr_glyph_preflight_cases() {
    let color_font = FontData::try_from_bytes(ahem_color_font(valid_cpal()), 0).unwrap();
    let color_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let color_run = text_run_for(color_font, 16.0, Transform::identity(), &color_glyphs);
    let mut color_scene = VelloScene::default();
    let color_error = color_scene
        .encode_text_run(&color_run)
        .expect_err("valid COLR data must reach the explicit unsupported glyph-rendering boundary");
    assert_render_failed_without_font_diagnostic(&color_error);
    assert_no_glyph_encoding(&color_scene);

    let color_v1_font =
        FontData::try_from_bytes(ahem_colr_v1_font_with_v0_root(valid_cpal()), 0).unwrap();
    let color_v1_run = text_run_for(color_v1_font, 16.0, Transform::identity(), &color_glyphs);
    assert_selected_glyph_trace(&color_v1_run, SelectedGlyphTrace::Colr);
    let mut color_v1_scene = VelloScene::default();
    let color_v1_error = color_v1_scene
        .encode_text_run(&color_v1_run)
        .expect_err("a COLRv1 table with a V0-only selected root must reach COLR omission");
    assert_render_failed_without_font_diagnostic(&color_v1_error);
    assert_no_glyph_encoding(&color_v1_scene);
}

fn assert_png_bitmap_glyph_preflight_cases() {
    let bitmap_font = FontData::try_from_bytes(ahem_sbix_font(rgba_png()), 0).unwrap();
    let bitmap_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let bitmap_run = text_run_for(bitmap_font, 16.0, Transform::identity(), &bitmap_glyphs);
    assert_selected_glyph_trace(
        &bitmap_run,
        SelectedGlyphTrace::Bitmap {
            source: BitmapSourceForTest::Sbix,
            ppem: 16,
        },
    );
    let mut bitmap_scene = VelloScene::default();
    let bitmap_error = bitmap_scene.encode_text_run(&bitmap_run).expect_err(
        "valid PNG bitmap data must reach the explicit unsupported glyph-rendering boundary",
    );
    assert_render_failed_without_font_diagnostic(&bitmap_error);
    assert_no_glyph_encoding(&bitmap_scene);

    let invalid_palette_font =
        FontData::try_from_bytes(ahem_color_font(invalid_cpal()), 0).unwrap();
    let color_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let invalid_palette_run = text_run_for(
        invalid_palette_font,
        16.0,
        Transform::identity(),
        &color_glyphs,
    );
    let mut invalid_palette_scene = VelloScene::default();
    let palette_error = invalid_palette_scene
        .encode_text_run(&invalid_palette_run)
        .expect_err("a selected invalid CPAL reference must be rejected");
    assert_font_data_error(
        &palette_error,
        font_data_value(&invalid_palette_run).as_str(),
    );
    assert_no_glyph_encoding(&invalid_palette_scene);

    let malformed_png_font = FontData::try_from_bytes(ahem_sbix_font(malformed_png()), 0).unwrap();
    let malformed_png_run = text_run_for(
        malformed_png_font,
        16.0,
        Transform::identity(),
        &bitmap_glyphs,
    );
    let mut malformed_png_scene = VelloScene::default();
    let png_error = malformed_png_scene
        .encode_text_run(&malformed_png_run)
        .expect_err("a selected malformed PNG must be rejected");
    assert_font_data_error(&png_error, font_data_value(&malformed_png_run).as_str());
    assert_no_glyph_encoding(&malformed_png_scene);

    let short_header_font = FontData::try_from_bytes(ahem_sbix_font(png_without_height()), 0)
        .expect("the short selected PNG header remains container-readable before glyph lowering");
    let short_header_run = text_run_for(
        short_header_font,
        16.0,
        Transform::identity(),
        &bitmap_glyphs,
    );
    let mut short_header_scene = VelloScene::default();
    let short_header_error = short_header_scene
        .encode_text_run(&short_header_run)
        .expect_err("a selected PNG without a readable height must not fall back to an outline");
    assert_font_data_error(
        &short_header_error,
        font_data_value(&short_header_run).as_str(),
    );
    assert_no_glyph_encoding(&short_header_scene);
}

fn assert_malformed_sbix_glyph_preflight_cases() {
    let bitmap_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let malformed_sbix_font = FontData::try_from_bytes(
        font_with_tables(
            ahem_sbix_font(rgba_png()).as_slice(),
            vec![(*b"sbix", vec![0])],
        ),
        0,
    )
    .expect("the malformed sbix table remains container-readable before glyph lowering");
    let malformed_sbix_run = text_run_for(
        malformed_sbix_font,
        16.0,
        Transform::identity(),
        &bitmap_glyphs,
    );
    let mut malformed_sbix_scene = VelloScene::default();
    let malformed_sbix_error = malformed_sbix_scene
        .encode_text_run(&malformed_sbix_run)
        .expect_err("a malformed selected sbix table must not fall back to an outline");
    assert_font_data_error(
        &malformed_sbix_error,
        font_data_value(&malformed_sbix_run).as_str(),
    );
    assert_no_glyph_encoding(&malformed_sbix_scene);

    let malformed_record_font =
        FontData::try_from_bytes(ahem_sbix_font_with_truncated_selected_record(), 0).expect(
            "the malformed selected sbix record remains container-readable before glyph lowering",
        );
    let malformed_record_run = text_run_for(
        malformed_record_font,
        16.0,
        Transform::identity(),
        &bitmap_glyphs,
    );
    let mut malformed_record_scene = VelloScene::default();
    let malformed_record_error = malformed_record_scene
        .encode_text_run(&malformed_record_run)
        .expect_err("a malformed selected sbix record must not fall back to an outline");
    assert_font_data_error(
        &malformed_record_error,
        font_data_value(&malformed_record_run).as_str(),
    );
    assert_no_glyph_encoding(&malformed_record_scene);

    let no_bitmap_font =
        FontData::try_from_bytes(ahem_sbix_font_without_selected_glyph(rgba_png()), 0)
            .expect("the sbix font without the selected bitmap remains container-readable");
    let no_bitmap_run = text_run_for(no_bitmap_font, 16.0, Transform::identity(), &bitmap_glyphs);
    assert_selected_glyph_trace(&no_bitmap_run, SelectedGlyphTrace::Outline);
    let mut no_bitmap_scene = VelloScene::default();
    no_bitmap_scene
        .encode_text_run(&no_bitmap_run)
        .expect("a valid bitmap strike without the selected glyph must use the outline");
    let no_bitmap_observation = no_bitmap_scene.observation_for_test();
    assert_eq!(no_bitmap_observation.glyph_run_count_for_test(), 1);
    assert_eq!(no_bitmap_observation.glyph_count_for_test(), 1);
}

fn assert_malformed_head_glyph_preflight_cases() {
    let color_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let malformed_colr_head_bytes = ahem_color_font(valid_cpal());
    let malformed_colr_head_font = FontData::try_from_bytes(
        font_with_tables(
            malformed_colr_head_bytes.as_slice(),
            vec![(*b"head", vec![0])],
        ),
        0,
    )
    .expect("the selected COLR font remains container-readable before head access");
    let malformed_colr_head_run = text_run_for(
        malformed_colr_head_font,
        16.0,
        Transform::identity(),
        &color_glyphs,
    );
    let mut malformed_colr_head_scene = VelloScene::default();
    let colr_head_error = malformed_colr_head_scene
        .encode_text_run(&malformed_colr_head_run)
        .expect_err("selected COLR lowering must reject malformed head data before encoding");
    assert_font_data_error(
        &colr_head_error,
        font_data_value(&malformed_colr_head_run).as_str(),
    );
    assert_no_glyph_encoding(&malformed_colr_head_scene);

    let malformed_bitmap_head_bytes = ahem_sbix_font(rgba_png());
    let bitmap_glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let malformed_bitmap_head_font = FontData::try_from_bytes(
        font_with_tables(
            malformed_bitmap_head_bytes.as_slice(),
            vec![(*b"head", vec![0])],
        ),
        0,
    )
    .expect("the selected bitmap font remains container-readable before head access");
    let malformed_bitmap_head_run = text_run_for(
        malformed_bitmap_head_font,
        16.0,
        Transform::identity(),
        &bitmap_glyphs,
    );
    let mut malformed_bitmap_head_scene = VelloScene::default();
    let bitmap_head_error = malformed_bitmap_head_scene
        .encode_text_run(&malformed_bitmap_head_run)
        .expect_err("selected bitmap lowering must reject malformed head data before encoding");
    assert_font_data_error(
        &bitmap_head_error,
        font_data_value(&malformed_bitmap_head_run).as_str(),
    );
    assert_no_glyph_encoding(&malformed_bitmap_head_scene);

    assert_bdt_glyph_preflight_cases(&bitmap_glyphs);
    assert_bitmap_format_selection_cases(&bitmap_glyphs);
}

#[test]
fn selected_glyph_preflight_distinguishes_unsupported_image_from_malformed_data() {
    let font_data = FontData::try_from_bytes(ahem_sbix_font(grayscale_png()), 0).unwrap();
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];
    let run = text_run_for(font_data, 16.0, Transform::identity(), &glyphs);
    let mut unsupported_scene = VelloScene::default();
    let error = unsupported_scene
        .encode_text_run(&run)
        .expect_err("a valid but unsupported image encoding must fail explicitly");

    assert_render_failed_without_font_diagnostic(&error);
    assert_no_glyph_encoding(&unsupported_scene);

    let malformed_font = FontData::try_from_bytes(ahem_sbix_font(malformed_grayscale_png()), 0)
        .expect("the malformed grayscale PNG remains container-readable before frame decode");
    let malformed_run = text_run_for(malformed_font, 16.0, Transform::identity(), &glyphs);
    let mut scene = VelloScene::default();
    let malformed_error = scene
        .encode_text_run(&malformed_run)
        .expect_err("malformed grayscale PNG data must be invalid, not unsupported");
    assert_font_data_error(&malformed_error, font_data_value(&malformed_run).as_str());
    assert_no_glyph_encoding(&scene);
}

#[test]
fn ahem_font_data_validates_at_collection_index_zero() {
    let font_data = FontData::try_from_bytes(AHEM_FONT_BYTES.to_vec(), 0);

    assert!(font_data.is_ok());
}

#[test]
fn malformed_lazy_font_tables_return_typed_errors_without_encoding_glyphs() {
    let malformed_colr_head_bytes = ahem_color_font(valid_cpal());
    let malformed_bitmap_head_bytes = ahem_sbix_font(rgba_png());
    let cases = [
        (
            "malformed outline table",
            ahem_with_tables(vec![(*b"glyf", vec![0])]),
        ),
        (
            "malformed selected COLR head table",
            font_with_tables(
                malformed_colr_head_bytes.as_slice(),
                vec![(*b"head", vec![0])],
            ),
        ),
        (
            "malformed selected bitmap head table",
            font_with_tables(
                malformed_bitmap_head_bytes.as_slice(),
                vec![(*b"head", vec![0])],
            ),
        ),
    ];
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 16.0, 8.0).unwrap()];

    for (case, bytes) in cases {
        let font_data = FontData::try_from_bytes(bytes, 0)
            .expect("the selected lazy-table case must pass initial FontData construction");
        let run = text_run_for(font_data, 16.0, Transform::identity(), &glyphs);
        let expected_value = font_data_value(&run);
        let mut scene = VelloScene::default();
        let outcome = catch_unwind(AssertUnwindSafe(|| scene.encode_text_run(&run)));

        let error = match outcome {
            Ok(Err(error)) => error,
            Ok(Ok(())) => panic!("{case} must not reach Encoding"),
            Err(_) => panic!("{case} must return a typed error instead of panicking"),
        };
        assert_font_data_error(&error, expected_value.as_str());
        assert_no_glyph_encoding(&scene);
    }
}

fn assert_render_failed_without_font_diagnostic(error: &Error) {
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert!(error.invalid_value_diagnostic().is_none());
}

fn assert_no_glyph_encoding(scene: &VelloScene) {
    let observation = scene.observation_for_test();

    assert_eq!(observation.glyph_run_count_for_test(), 0);
    assert_eq!(observation.glyph_count_for_test(), 0);
    assert_eq!(observation.patch_count_for_test(), 0);
}

fn font_data_value(run: &TextRun<'_>) -> String {
    let font_data = run
        .font()
        .data
        .as_ref()
        .expect("text-run fixture must carry FontData");
    format!(
        "len={}, index={}",
        font_data.bytes().len(),
        font_data.index()
    )
}

fn assert_font_data_error(error: &Error, value: &str) {
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    let diagnostic = error
        .invalid_value_diagnostic()
        .expect("font failures must carry InvalidValue diagnostics");
    assert_eq!(diagnostic.field(), "font_data");
    assert_eq!(diagnostic.value(), value);
    assert_eq!(
        diagnostic.invariant(),
        "must contain a readable OpenType font at the requested collection index"
    );
}

fn assert_missing_glyph_error(error: &Error, glyph_id: u32) {
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    let diagnostic = error
        .invalid_value_diagnostic()
        .expect("missing glyph failures must carry InvalidValue diagnostics");
    assert_eq!(diagnostic.field(), "text_glyph.id");
    assert_eq!(diagnostic.value(), glyph_id.to_string());
    assert_eq!(
        diagnostic.invariant(),
        "must identify a drawable glyph in the selected FontData"
    );
}

fn assert_selected_glyph_trace(run: &TextRun<'_>, expected: SelectedGlyphTrace) {
    let validated = preflight_selected_glyphs(run)
        .expect("the selected glyph fixture must complete preflight before encoding");

    assert_eq!(validated.selected_glyph_traces_for_test(), &[expected]);
}

fn assert_bdt_glyph_preflight_cases(glyphs: &[TextGlyph]) {
    for kind in BdtKind::ALL {
        for index_format in BdtIndexFormat::ALL {
            assert_bdt_selected_bitmap(
                kind,
                &[BdtStrikeFixture::new(
                    16,
                    index_format,
                    BdtGlyphFixture::Present,
                )],
                16.0,
                16,
                glyphs,
            );
        }

        for index_format in [BdtIndexFormat::Format4, BdtIndexFormat::Format5] {
            assert_bdt_outline_fallback(
                kind,
                &[BdtStrikeFixture::new(
                    16,
                    index_format,
                    BdtGlyphFixture::SparseMissing,
                )],
                16.0,
                glyphs,
            );
        }

        let competing_present = [
            BdtStrikeFixture::new(12, BdtIndexFormat::Format1, BdtGlyphFixture::Present),
            BdtStrikeFixture::new(16, BdtIndexFormat::Format2, BdtGlyphFixture::Present),
            BdtStrikeFixture::new(20, BdtIndexFormat::Format3, BdtGlyphFixture::Present),
        ];
        for (size, expected_ppem) in [(16.0, 16), (14.0, 16), (22.0, 20)] {
            assert_bdt_selected_bitmap(
                kind,
                competing_present.as_slice(),
                size,
                expected_ppem,
                glyphs,
            );
        }
        assert_bdt_selected_bitmap(
            kind,
            &[
                BdtStrikeFixture::new(16, BdtIndexFormat::Format1, BdtGlyphFixture::Empty),
                BdtStrikeFixture::new(20, BdtIndexFormat::Format2, BdtGlyphFixture::Present),
            ],
            16.0,
            20,
            glyphs,
        );

        assert_bdt_sparse_preflight_cases(kind, glyphs);
    }

    for kind in BdtKind::ALL {
        assert_bdt_outline_fallback(
            kind,
            &[BdtStrikeFixture::new(
                16,
                BdtIndexFormat::Format1,
                BdtGlyphFixture::Empty,
            )],
            16.0,
            glyphs,
        );
    }

    assert_cbdt_precedes_ebdt(glyphs);
}

fn assert_bdt_sparse_preflight_cases(kind: BdtKind, glyphs: &[TextGlyph]) {
    for index_format in [BdtIndexFormat::Format4, BdtIndexFormat::Format5] {
        for glyph in [
            BdtGlyphFixture::SparseDuplicate,
            BdtGlyphFixture::SparseUnsorted,
        ] {
            assert_bdt_sparse_invalid(
                kind,
                &[BdtStrikeFixture::new(16, index_format, glyph)],
                glyphs,
            );
        }
        assert_bdt_selected_bitmap(
            kind,
            &[BdtStrikeFixture::new(
                16,
                index_format,
                BdtGlyphFixture::SparseUnrelatedDisorder,
            )],
            16.0,
            16,
            glyphs,
        );
    }

    assert_bdt_sparse_invalid(
        kind,
        &[BdtStrikeFixture::new(
            16,
            BdtIndexFormat::Format4,
            BdtGlyphFixture::SparseMalformedSentinel,
        )],
        glyphs,
    );
    assert_bdt_selected_bitmap(
        kind,
        &[BdtStrikeFixture::new(
            16,
            BdtIndexFormat::Format4,
            BdtGlyphFixture::SparseUnselectedMalformedSentinel,
        )],
        16.0,
        16,
        glyphs,
    );

    assert_bdt_selected_bitmap(
        kind,
        &[
            BdtStrikeFixture::new(
                16,
                BdtIndexFormat::Format4,
                BdtGlyphFixture::UnselectedSparseUnsorted,
            ),
            BdtStrikeFixture::new(16, BdtIndexFormat::Format1, BdtGlyphFixture::Present),
        ],
        16.0,
        16,
        glyphs,
    );
}

fn assert_bdt_selected_bitmap(
    kind: BdtKind,
    strikes: &[BdtStrikeFixture],
    size: f32,
    expected_ppem: u16,
    glyphs: &[TextGlyph],
) {
    let font_data = FontData::try_from_bytes(ahem_bdt_font(kind, strikes), 0)
        .expect("the BDT fixture must remain readable before selected bitmap lowering");
    let run = text_run_for(font_data, size, Transform::identity(), glyphs);
    assert_selected_glyph_trace(
        &run,
        SelectedGlyphTrace::Bitmap {
            source: kind.trace_source(),
            ppem: expected_ppem,
        },
    );
    let mut scene = VelloScene::default();
    let error = scene.encode_text_run(&run).expect_err(
        "a valid selected BDT bitmap must reach the explicit unsupported glyph-rendering boundary",
    );

    assert_render_failed_without_font_diagnostic(&error);
    assert_no_glyph_encoding(&scene);
}

fn assert_bdt_outline_fallback(
    kind: BdtKind,
    strikes: &[BdtStrikeFixture],
    size: f32,
    glyphs: &[TextGlyph],
) {
    let font_data = FontData::try_from_bytes(ahem_bdt_font(kind, strikes), 0)
        .expect("the BDT fixture must remain readable before selected bitmap lowering");
    let run = text_run_for(font_data, size, Transform::identity(), glyphs);
    assert_selected_glyph_trace(&run, SelectedGlyphTrace::Outline);
    let mut scene = VelloScene::default();

    scene
        .encode_text_run(&run)
        .expect("a valid absent BDT bitmap must fall back to the outline");
    assert_outline_glyph_encoding(&scene, glyphs[0].id());
}

fn assert_bdt_sparse_invalid(kind: BdtKind, strikes: &[BdtStrikeFixture], glyphs: &[TextGlyph]) {
    let font_data = FontData::try_from_bytes(ahem_bdt_font(kind, strikes), 0)
        .expect("the malformed sparse BDT fixture must remain container-readable");
    let run = text_run_for(font_data, 16.0, Transform::identity(), glyphs);
    let expected_value = font_data_value(&run);
    let mut scene = VelloScene::default();
    let error = scene
        .encode_text_run(&run)
        .expect_err("a malformed selected sparse BDT record must not fall back to the outline");

    assert_font_data_error(&error, expected_value.as_str());
    assert_no_glyph_encoding(&scene);
}

fn assert_cbdt_precedes_ebdt(glyphs: &[TextGlyph]) {
    let (cblc, cbdt) = bdt_tables(
        BdtKind::Cbdt,
        &[BdtStrikeFixture::new(
            16,
            BdtIndexFormat::Format1,
            BdtGlyphFixture::Present,
        )],
    );
    let (eblc, ebdt) = bdt_tables(
        BdtKind::Ebdt,
        &[BdtStrikeFixture::new(
            16,
            BdtIndexFormat::Format1,
            BdtGlyphFixture::Empty,
        )],
    );
    let font_data = FontData::try_from_bytes(
        ahem_with_tables(vec![
            (*b"CBLC", cblc),
            (*b"CBDT", cbdt),
            (*b"EBLC", eblc),
            (*b"EBDT", ebdt),
        ]),
        0,
    )
    .expect("the combined BDT fixture must remain readable before glyph lowering");
    let run = text_run_for(font_data, 16.0, Transform::identity(), glyphs);
    assert_selected_glyph_trace(
        &run,
        SelectedGlyphTrace::Bitmap {
            source: BitmapSourceForTest::Cbdt,
            ppem: 16,
        },
    );
    let mut scene = VelloScene::default();
    let error = scene
        .encode_text_run(&run)
        .expect_err("CBLC/CBDT must retain precedence over EBLC/EBDT");

    assert_render_failed_without_font_diagnostic(&error);
    assert_no_glyph_encoding(&scene);
}

fn assert_bitmap_format_selection_cases(glyphs: &[TextGlyph]) {
    let sbix_competing = [
        SbixStrikeFixture::new(12, true),
        SbixStrikeFixture::new(16, true),
        SbixStrikeFixture::new(20, true),
    ];
    let sbix_without_selected = [SbixStrikeFixture::new(16, false)];
    let cbdt_selected = [BdtStrikeFixture::new(
        16,
        BdtIndexFormat::Format1,
        BdtGlyphFixture::Present,
    )];
    let ebdt_selected = [BdtStrikeFixture::new(
        16,
        BdtIndexFormat::Format1,
        BdtGlyphFixture::Present,
    )];
    let cases = [
        BitmapFormatFixture {
            sbix: Some(sbix_competing.as_slice()),
            cbdt: Some(cbdt_selected.as_slice()),
            ebdt: Some(ebdt_selected.as_slice()),
            size: 14.0,
            expected: BitmapFormatExpected::Bitmap {
                source: BitmapSourceForTest::Sbix,
                ppem: 16,
            },
        },
        BitmapFormatFixture {
            sbix: Some(sbix_without_selected.as_slice()),
            cbdt: Some(cbdt_selected.as_slice()),
            ebdt: Some(ebdt_selected.as_slice()),
            size: 16.0,
            expected: BitmapFormatExpected::Outline,
        },
        BitmapFormatFixture {
            sbix: None,
            cbdt: Some(cbdt_selected.as_slice()),
            ebdt: None,
            size: 16.0,
            expected: BitmapFormatExpected::Bitmap {
                source: BitmapSourceForTest::Cbdt,
                ppem: 16,
            },
        },
        BitmapFormatFixture {
            sbix: None,
            cbdt: Some(cbdt_selected.as_slice()),
            ebdt: Some(ebdt_selected.as_slice()),
            size: 16.0,
            expected: BitmapFormatExpected::Bitmap {
                source: BitmapSourceForTest::Cbdt,
                ppem: 16,
            },
        },
        BitmapFormatFixture {
            sbix: None,
            cbdt: None,
            ebdt: Some(ebdt_selected.as_slice()),
            size: 16.0,
            expected: BitmapFormatExpected::Bitmap {
                source: BitmapSourceForTest::Ebdt,
                ppem: 16,
            },
        },
    ];

    for case in cases {
        let font_data =
            FontData::try_from_bytes(ahem_bitmap_format_font(case.sbix, case.cbdt, case.ebdt), 0)
                .expect("the bitmap format fixture must remain readable before glyph lowering");
        let run = text_run_for(font_data, case.size, Transform::identity(), glyphs);
        assert_selected_glyph_trace(&run, case.expected.trace());
        let mut scene = VelloScene::default();

        match case.expected {
            BitmapFormatExpected::Bitmap { .. } => {
                let error = scene
                    .encode_text_run(&run)
                    .expect_err("a selected bitmap must reach the explicit unsupported glyph-rendering boundary");
                assert_render_failed_without_font_diagnostic(&error);
                assert_no_glyph_encoding(&scene);
            }
            BitmapFormatExpected::Outline => {
                scene
                    .encode_text_run(&run)
                    .expect("the chosen bitmap format without the selected glyph must use outline");
                assert_outline_glyph_encoding(&scene, glyphs[0].id());
            }
        }
    }
}

#[derive(Clone, Copy)]
struct BitmapFormatFixture<'a> {
    sbix: Option<&'a [SbixStrikeFixture]>,
    cbdt: Option<&'a [BdtStrikeFixture]>,
    ebdt: Option<&'a [BdtStrikeFixture]>,
    size: f32,
    expected: BitmapFormatExpected,
}

#[derive(Clone, Copy)]
enum BitmapFormatExpected {
    Bitmap {
        source: BitmapSourceForTest,
        ppem: u16,
    },
    Outline,
}

impl BitmapFormatExpected {
    const fn trace(self) -> SelectedGlyphTrace {
        match self {
            Self::Bitmap { source, ppem } => SelectedGlyphTrace::Bitmap { source, ppem },
            Self::Outline => SelectedGlyphTrace::Outline,
        }
    }
}

fn assert_outline_glyph_encoding(scene: &VelloScene, glyph_id: u32) {
    let observation = scene.observation_for_test();

    assert_eq!(observation.glyph_run_count_for_test(), 1);
    assert_eq!(observation.glyph_count_for_test(), 1);
    assert_eq!(observation.patch_count_for_test(), 1);
    assert_eq!(
        observation
            .first_glyph_for_test()
            .expect("the Vello scene must retain the selected glyph")
            .id_for_test(),
        glyph_id
    );
}

#[derive(Clone, Copy)]
enum BdtKind {
    Cbdt,
    Ebdt,
}

impl BdtKind {
    const ALL: [Self; 2] = [Self::Cbdt, Self::Ebdt];

    const fn location_tag(self) -> [u8; 4] {
        match self {
            Self::Cbdt => *b"CBLC",
            Self::Ebdt => *b"EBLC",
        }
    }

    const fn data_tag(self) -> [u8; 4] {
        match self {
            Self::Cbdt => *b"CBDT",
            Self::Ebdt => *b"EBDT",
        }
    }

    const fn location_major_version(self) -> u16 {
        match self {
            Self::Cbdt => 3,
            Self::Ebdt => 2,
        }
    }

    const fn data_major_version(self) -> u16 {
        self.location_major_version()
    }

    const fn trace_source(self) -> BitmapSourceForTest {
        match self {
            Self::Cbdt => BitmapSourceForTest::Cbdt,
            Self::Ebdt => BitmapSourceForTest::Ebdt,
        }
    }
}

#[derive(Clone, Copy)]
enum BdtIndexFormat {
    Format1,
    Format2,
    Format3,
    Format4,
    Format5,
}

impl BdtIndexFormat {
    const ALL: [Self; 5] = [
        Self::Format1,
        Self::Format2,
        Self::Format3,
        Self::Format4,
        Self::Format5,
    ];

    const fn number(self) -> u16 {
        match self {
            Self::Format1 => 1,
            Self::Format2 => 2,
            Self::Format3 => 3,
            Self::Format4 => 4,
            Self::Format5 => 5,
        }
    }

    const fn uses_constant_metrics(self) -> bool {
        matches!(self, Self::Format2 | Self::Format5)
    }
}

#[derive(Clone, Copy)]
enum BdtGlyphFixture {
    Present,
    Empty,
    SparseMissing,
    SparseUnsorted,
    SparseUnrelatedDisorder,
    SparseDuplicate,
    SparseMalformedSentinel,
    SparseUnselectedMalformedSentinel,
    UnselectedSparseUnsorted,
}

#[derive(Clone, Copy)]
struct BdtStrikeFixture {
    ppem: u8,
    index_format: BdtIndexFormat,
    glyph: BdtGlyphFixture,
}

impl BdtStrikeFixture {
    const fn new(ppem: u8, index_format: BdtIndexFormat, glyph: BdtGlyphFixture) -> Self {
        Self {
            ppem,
            index_format,
            glyph,
        }
    }
}

fn ahem_bdt_font(kind: BdtKind, strikes: &[BdtStrikeFixture]) -> Vec<u8> {
    let (location, data) = bdt_tables(kind, strikes);

    ahem_with_tables(vec![
        (kind.location_tag(), location),
        (kind.data_tag(), data),
    ])
}

fn ahem_bitmap_format_font(
    sbix: Option<&[SbixStrikeFixture]>,
    cbdt: Option<&[BdtStrikeFixture]>,
    ebdt: Option<&[BdtStrikeFixture]>,
) -> Vec<u8> {
    let mut tables = Vec::new();
    if let Some(strikes) = sbix {
        let png = rgba_png();
        tables.push((*b"sbix", sbix_table(png.as_slice(), strikes)));
    }
    if let Some(strikes) = cbdt {
        let (location, data) = bdt_tables(BdtKind::Cbdt, strikes);
        tables.push((*b"CBLC", location));
        tables.push((*b"CBDT", data));
    }
    if let Some(strikes) = ebdt {
        let (location, data) = bdt_tables(BdtKind::Ebdt, strikes);
        tables.push((*b"EBLC", location));
        tables.push((*b"EBDT", data));
    }
    ahem_with_tables(tables)
}

fn bdt_tables(kind: BdtKind, strikes: &[BdtStrikeFixture]) -> (Vec<u8>, Vec<u8>) {
    let mut data = Vec::new();
    push_be_u16(&mut data, kind.data_major_version());
    push_be_u16(&mut data, 0);
    let mut strike_parts = Vec::with_capacity(strikes.len());

    for strike in strikes {
        let data_offset = u32::try_from(data.len()).unwrap();
        let (first_glyph, last_glyph, subtable, image_data) =
            bdt_strike_parts(*strike, data_offset);
        data.extend_from_slice(image_data.as_slice());
        strike_parts.push((first_glyph, last_glyph, subtable));
    }

    let mut location = Vec::new();
    push_be_u16(&mut location, kind.location_major_version());
    push_be_u16(&mut location, 0);
    push_be_u32(&mut location, strikes.len().try_into().unwrap());
    let bitmap_sizes_offset = location.len();
    location.resize(bitmap_sizes_offset + strikes.len() * 48, 0);

    for (index, (strike, (first_glyph, last_glyph, subtable))) in
        strikes.iter().zip(strike_parts).enumerate()
    {
        while location.len() % 4 != 0 {
            location.push(0);
        }
        let index_subtable_list_offset = u32::try_from(location.len()).unwrap();
        let list = bdt_index_subtable_list(first_glyph, last_glyph, subtable.as_slice());
        let bitmap_size_offset = bitmap_sizes_offset + index * 48;
        write_bdt_bitmap_size(
            location.as_mut_slice(),
            bitmap_size_offset,
            index_subtable_list_offset,
            u32::try_from(list.len()).unwrap(),
            strike.ppem,
        );
        location.extend_from_slice(list.as_slice());
    }

    (location, data)
}

fn bdt_strike_parts(
    strike: BdtStrikeFixture,
    image_data_offset: u32,
) -> (u16, u16, Vec<u8>, Vec<u8>) {
    let selected_glyph = u16::try_from(AHEM_GLYPH_X).unwrap();
    let (first_glyph, last_glyph) = match strike.glyph {
        BdtGlyphFixture::UnselectedSparseUnsorted => (selected_glyph + 1, selected_glyph + 1),
        _ if matches!(
            strike.index_format,
            BdtIndexFormat::Format1 | BdtIndexFormat::Format2 | BdtIndexFormat::Format3
        ) =>
        {
            (selected_glyph, selected_glyph)
        }
        _ => (selected_glyph, selected_glyph + 2),
    };
    let image_data = bdt_image_data(strike.index_format, strike.glyph);
    let mut subtable = Vec::new();

    push_be_u16(&mut subtable, strike.index_format.number());
    push_be_u16(
        &mut subtable,
        if strike.index_format.uses_constant_metrics() {
            5
        } else {
            2
        },
    );
    push_be_u32(&mut subtable, image_data_offset);

    match strike.index_format {
        BdtIndexFormat::Format1 => {
            push_be_u32(&mut subtable, 0);
            push_be_u32(&mut subtable, image_data.len().try_into().unwrap());
        }
        BdtIndexFormat::Format2 => {
            push_be_u32(&mut subtable, image_data.len().try_into().unwrap());
            push_bdt_big_metrics(&mut subtable);
        }
        BdtIndexFormat::Format3 => {
            push_be_u16(&mut subtable, 0);
            push_be_u16(&mut subtable, image_data.len().try_into().unwrap());
        }
        BdtIndexFormat::Format4 => push_bdt_format4_array(
            &mut subtable,
            selected_glyph,
            strike.glyph,
            image_data_offset.try_into().unwrap(),
            image_data.len().try_into().unwrap(),
        ),
        BdtIndexFormat::Format5 => {
            push_be_u32(&mut subtable, 1);
            push_bdt_big_metrics(&mut subtable);
            push_bdt_format5_array(&mut subtable, selected_glyph, strike.glyph);
        }
    }

    (first_glyph, last_glyph, subtable, image_data)
}

fn bdt_image_data(index_format: BdtIndexFormat, glyph: BdtGlyphFixture) -> Vec<u8> {
    if matches!(glyph, BdtGlyphFixture::Empty) {
        return Vec::new();
    }

    let image = if index_format.uses_constant_metrics() {
        &[0x80][..]
    } else {
        &[1, 1, 0, 1, 1, 0x80][..]
    };
    let count = match glyph {
        BdtGlyphFixture::SparseUnsorted
        | BdtGlyphFixture::SparseUnrelatedDisorder
        | BdtGlyphFixture::SparseDuplicate
        | BdtGlyphFixture::SparseUnselectedMalformedSentinel
        | BdtGlyphFixture::UnselectedSparseUnsorted => 3,
        _ => 1,
    };
    let mut data = Vec::with_capacity(image.len() * count);
    for _ in 0..count {
        data.extend_from_slice(image);
    }
    data
}

fn bdt_index_subtable_list(first_glyph: u16, last_glyph: u16, subtable: &[u8]) -> Vec<u8> {
    let mut list = Vec::new();
    push_be_u16(&mut list, first_glyph);
    push_be_u16(&mut list, last_glyph);
    push_be_u32(&mut list, 8);
    list.extend_from_slice(subtable);
    list
}

fn write_bdt_bitmap_size(
    bytes: &mut [u8],
    offset: usize,
    index_subtable_list_offset: u32,
    index_subtable_list_size: u32,
    ppem: u8,
) {
    write_be_u32(bytes, offset, index_subtable_list_offset);
    write_be_u32(bytes, offset + 4, index_subtable_list_size);
    write_be_u32(bytes, offset + 8, 1);
    let selected_glyph = u16::try_from(AHEM_GLYPH_X).unwrap();
    write_be_u16(bytes, offset + 40, selected_glyph);
    write_be_u16(bytes, offset + 42, selected_glyph + 2);
    bytes[offset + 44] = ppem;
    bytes[offset + 45] = ppem;
    bytes[offset + 46] = 1;
    bytes[offset + 47] = 1;
}

fn push_bdt_big_metrics(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&[1, 1, 0, 1, 1, 0, 1, 1]);
}

fn push_bdt_format4_array(
    bytes: &mut Vec<u8>,
    selected_glyph: u16,
    glyph: BdtGlyphFixture,
    image_data_offset: u16,
    image_data_len: u16,
) {
    match glyph {
        BdtGlyphFixture::SparseMissing => {
            push_be_u32(bytes, 1);
            push_bdt_glyph_offset_pair(bytes, selected_glyph + 1, image_data_offset);
            push_bdt_glyph_offset_pair(bytes, u16::MAX, image_data_offset + image_data_len);
        }
        BdtGlyphFixture::SparseUnsorted | BdtGlyphFixture::UnselectedSparseUnsorted => {
            push_be_u32(bytes, 3);
            push_bdt_glyph_offset_pair(bytes, selected_glyph, image_data_offset);
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph - 1,
                image_data_offset + image_data_len / 3,
            );
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 2,
                image_data_offset + image_data_len / 3 * 2,
            );
            push_bdt_glyph_offset_pair(bytes, u16::MAX, image_data_offset + image_data_len);
        }
        BdtGlyphFixture::SparseUnrelatedDisorder => {
            push_be_u32(bytes, 3);
            push_bdt_glyph_offset_pair(bytes, selected_glyph + 1, image_data_offset);
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph,
                image_data_offset + image_data_len / 3,
            );
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 2,
                image_data_offset + image_data_len / 3 * 2,
            );
            push_bdt_glyph_offset_pair(bytes, u16::MAX, image_data_offset + image_data_len);
        }
        BdtGlyphFixture::SparseDuplicate => {
            push_be_u32(bytes, 3);
            push_bdt_glyph_offset_pair(bytes, selected_glyph, image_data_offset);
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph,
                image_data_offset + image_data_len / 3,
            );
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 2,
                image_data_offset + image_data_len / 3 * 2,
            );
            push_bdt_glyph_offset_pair(bytes, u16::MAX, image_data_offset + image_data_len);
        }
        BdtGlyphFixture::SparseMalformedSentinel => {
            push_be_u32(bytes, 1);
            push_bdt_glyph_offset_pair(bytes, selected_glyph, image_data_offset);
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 1,
                image_data_offset + image_data_len,
            );
        }
        BdtGlyphFixture::SparseUnselectedMalformedSentinel => {
            push_be_u32(bytes, 3);
            push_bdt_glyph_offset_pair(bytes, selected_glyph, image_data_offset);
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 1,
                image_data_offset + image_data_len / 3,
            );
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 2,
                image_data_offset + image_data_len / 3 * 2,
            );
            push_bdt_glyph_offset_pair(
                bytes,
                selected_glyph + 3,
                image_data_offset + image_data_len,
            );
        }
        _ => {
            push_be_u32(bytes, 1);
            push_bdt_glyph_offset_pair(bytes, selected_glyph, image_data_offset);
            push_bdt_glyph_offset_pair(bytes, u16::MAX, image_data_offset + image_data_len);
        }
    }
}

fn push_bdt_format5_array(bytes: &mut Vec<u8>, selected_glyph: u16, glyph: BdtGlyphFixture) {
    match glyph {
        BdtGlyphFixture::SparseMissing => {
            push_be_u32(bytes, 1);
            push_be_u16(bytes, selected_glyph + 1);
        }
        BdtGlyphFixture::SparseUnsorted | BdtGlyphFixture::UnselectedSparseUnsorted => {
            push_be_u32(bytes, 3);
            push_be_u16(bytes, selected_glyph);
            push_be_u16(bytes, selected_glyph - 1);
            push_be_u16(bytes, selected_glyph + 2);
        }
        BdtGlyphFixture::SparseUnrelatedDisorder => {
            push_be_u32(bytes, 3);
            push_be_u16(bytes, selected_glyph + 1);
            push_be_u16(bytes, selected_glyph);
            push_be_u16(bytes, selected_glyph + 2);
        }
        BdtGlyphFixture::SparseDuplicate => {
            push_be_u32(bytes, 3);
            push_be_u16(bytes, selected_glyph);
            push_be_u16(bytes, selected_glyph);
            push_be_u16(bytes, selected_glyph + 2);
        }
        _ => {
            push_be_u32(bytes, 1);
            push_be_u16(bytes, selected_glyph);
        }
    }
}

fn push_bdt_glyph_offset_pair(bytes: &mut Vec<u8>, glyph_id: u16, offset: u16) {
    push_be_u16(bytes, glyph_id);
    push_be_u16(bytes, offset);
}

fn ahem_with_tables(replacements: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
    font_with_tables(AHEM_FONT_BYTES, replacements)
}

fn font_with_tables(font_bytes: &[u8], replacements: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
    let table_count = read_be_u16(font_bytes, 4) as usize;
    let mut tables = (0..table_count)
        .map(|index| {
            let record = 12 + index * 16;
            let tag = font_bytes[record..record + 4].try_into().unwrap();
            let offset = read_be_u32(font_bytes, record + 8) as usize;
            let length = read_be_u32(font_bytes, record + 12) as usize;
            (tag, font_bytes[offset..offset + length].to_vec())
        })
        .collect::<Vec<([u8; 4], Vec<u8>)>>();

    for (tag, replacement) in replacements {
        if let Some((_, table)) = tables.iter_mut().find(|(existing, _)| *existing == tag) {
            *table = replacement;
        } else {
            tables.push((tag, replacement));
        }
    }
    tables.sort_by_key(|(tag, _)| *tag);

    let count = tables.len();
    let mut output = vec![0; 12 + count * 16];
    output[0..4].copy_from_slice(&font_bytes[0..4]);
    write_be_u16(&mut output, 4, count.try_into().unwrap());
    let mut power = 1usize;
    let mut selector = 0usize;
    while power * 2 <= count {
        power *= 2;
        selector += 1;
    }
    write_be_u16(&mut output, 6, (power * 16).try_into().unwrap());
    write_be_u16(&mut output, 8, selector.try_into().unwrap());
    write_be_u16(&mut output, 10, ((count - power) * 16).try_into().unwrap());

    let mut offset = output.len();
    for (index, (tag, table)) in tables.into_iter().enumerate() {
        let padding = (4 - offset % 4) % 4;
        output.resize(offset + padding, 0);
        offset += padding;
        let record = 12 + index * 16;
        output[record..record + 4].copy_from_slice(&tag);
        write_be_u32(&mut output, record + 4, 0);
        write_be_u32(&mut output, record + 8, offset.try_into().unwrap());
        write_be_u32(&mut output, record + 12, table.len().try_into().unwrap());
        output.extend_from_slice(&table);
        offset += table.len();
    }
    output
}

fn ahem_color_font(cpal: Vec<u8>) -> Vec<u8> {
    let mut colr = Vec::new();
    push_be_u16(&mut colr, 0);
    push_be_u16(&mut colr, 1);
    push_be_u32(&mut colr, 14);
    push_be_u32(&mut colr, 20);
    push_be_u16(&mut colr, 1);
    push_be_u16(&mut colr, AHEM_GLYPH_X.try_into().unwrap());
    push_be_u16(&mut colr, 0);
    push_be_u16(&mut colr, 1);
    push_be_u16(&mut colr, AHEM_GLYPH_X.try_into().unwrap());
    push_be_u16(&mut colr, 0);

    ahem_with_tables(vec![(*b"COLR", colr), (*b"CPAL", cpal)])
}

fn ahem_colr_v1_font_with_v0_root(cpal: Vec<u8>) -> Vec<u8> {
    let mut colr = Vec::new();
    push_be_u16(&mut colr, 1);
    push_be_u16(&mut colr, 1);
    push_be_u32(&mut colr, 34);
    push_be_u32(&mut colr, 40);
    push_be_u16(&mut colr, 1);
    push_be_u32(&mut colr, 44);
    for _ in 0..4 {
        push_be_u32(&mut colr, 0);
    }
    push_be_u16(&mut colr, AHEM_GLYPH_X.try_into().unwrap());
    push_be_u16(&mut colr, 0);
    push_be_u16(&mut colr, 1);
    push_be_u16(&mut colr, AHEM_GLYPH_X.try_into().unwrap());
    push_be_u16(&mut colr, 0);
    push_be_u32(&mut colr, 0);

    ahem_with_tables(vec![(*b"COLR", colr), (*b"CPAL", cpal)])
}

fn valid_cpal() -> Vec<u8> {
    let mut cpal = Vec::new();
    push_be_u16(&mut cpal, 0);
    push_be_u16(&mut cpal, 1);
    push_be_u16(&mut cpal, 1);
    push_be_u16(&mut cpal, 1);
    push_be_u32(&mut cpal, 14);
    push_be_u16(&mut cpal, 0);
    cpal.extend_from_slice(&[0, 0, 255, 255]);
    cpal
}

fn invalid_cpal() -> Vec<u8> {
    let mut cpal = valid_cpal();
    write_be_u32(&mut cpal, 8, u32::MAX);
    cpal
}

fn ahem_sbix_font(png: Vec<u8>) -> Vec<u8> {
    ahem_sbix_font_with_selected_bitmap(png, true)
}

fn ahem_sbix_font_without_selected_glyph(png: Vec<u8>) -> Vec<u8> {
    ahem_sbix_font_with_selected_bitmap(png, false)
}

fn ahem_sbix_font_with_truncated_selected_record() -> Vec<u8> {
    let glyph_count = ahem_num_glyphs();
    let bitmap_offset = 4 + (glyph_count + 1) * 4;
    let bitmap_end = bitmap_offset + 7;
    let mut sbix = Vec::new();
    push_be_u16(&mut sbix, 1);
    push_be_u16(&mut sbix, 1);
    push_be_u32(&mut sbix, 1);
    push_be_u32(&mut sbix, 12);
    push_be_u16(&mut sbix, 16);
    push_be_u16(&mut sbix, 72);
    for glyph_id in 0..=glyph_count {
        let offset = if glyph_id <= AHEM_GLYPH_X as usize {
            bitmap_offset
        } else {
            bitmap_end
        };
        push_be_u32(&mut sbix, offset.try_into().unwrap());
    }
    sbix.extend_from_slice(&[0; 7]);

    ahem_with_tables(vec![(*b"sbix", sbix)])
}

fn ahem_sbix_font_with_selected_bitmap(png: Vec<u8>, selected_bitmap: bool) -> Vec<u8> {
    ahem_sbix_font_with_strikes(png, &[SbixStrikeFixture::new(16, selected_bitmap)])
}

#[derive(Clone, Copy)]
struct SbixStrikeFixture {
    ppem: u16,
    selected: bool,
}

impl SbixStrikeFixture {
    const fn new(ppem: u16, selected: bool) -> Self {
        Self { ppem, selected }
    }
}

fn ahem_sbix_font_with_strikes(png: Vec<u8>, strikes: &[SbixStrikeFixture]) -> Vec<u8> {
    ahem_with_tables(vec![(*b"sbix", sbix_table(png.as_slice(), strikes))])
}

fn sbix_table(png: &[u8], strikes: &[SbixStrikeFixture]) -> Vec<u8> {
    let mut sbix = Vec::new();
    push_be_u16(&mut sbix, 1);
    push_be_u16(&mut sbix, 1);
    push_be_u32(&mut sbix, strikes.len().try_into().unwrap());
    let strike_offsets_start = sbix.len();
    sbix.resize(strike_offsets_start + strikes.len() * 4, 0);

    for (index, strike) in strikes.iter().enumerate() {
        let strike_offset = u32::try_from(sbix.len()).unwrap();
        write_be_u32(
            sbix.as_mut_slice(),
            strike_offsets_start + index * 4,
            strike_offset,
        );
        sbix.extend_from_slice(sbix_strike(png, *strike).as_slice());
    }

    sbix
}

fn sbix_strike(png: &[u8], strike: SbixStrikeFixture) -> Vec<u8> {
    let glyph_count = ahem_num_glyphs();
    let glyph_record_len = 8 + png.len();
    let bitmap_offset = 4 + (glyph_count + 1) * 4;
    let bitmap_end = bitmap_offset + glyph_record_len;
    let mut sbix_strike = Vec::new();
    push_be_u16(&mut sbix_strike, strike.ppem);
    push_be_u16(&mut sbix_strike, 72);
    for glyph_id in 0..=glyph_count {
        let offset = if glyph_id < AHEM_GLYPH_X as usize
            || strike.selected && glyph_id == AHEM_GLYPH_X as usize
        {
            bitmap_offset
        } else {
            bitmap_end
        };
        push_be_u32(&mut sbix_strike, offset.try_into().unwrap());
    }
    push_be_u16(&mut sbix_strike, 0);
    push_be_u16(&mut sbix_strike, 0);
    sbix_strike.extend_from_slice(b"png ");
    sbix_strike.extend_from_slice(png);

    sbix_strike
}

fn ahem_num_glyphs() -> usize {
    let table_count = read_be_u16(AHEM_FONT_BYTES, 4) as usize;
    let maxp_record = (0..table_count)
        .map(|index| 12 + index * 16)
        .find(|record| &AHEM_FONT_BYTES[*record..*record + 4] == b"maxp")
        .unwrap();
    let offset = read_be_u32(AHEM_FONT_BYTES, maxp_record + 8) as usize;
    read_be_u16(AHEM_FONT_BYTES, offset + 4) as usize
}

fn rgba_png() -> Vec<u8> {
    encoded_png(png::ColorType::Rgba, &[255, 0, 0, 255])
}

fn grayscale_png() -> Vec<u8> {
    encoded_png(png::ColorType::Grayscale, &[127])
}

fn malformed_png() -> Vec<u8> {
    let mut png = rgba_png();
    png.truncate(png.len() - 12);
    png
}

fn png_without_height() -> Vec<u8> {
    let mut png = rgba_png();
    png.truncate(20);
    png
}

fn malformed_grayscale_png() -> Vec<u8> {
    let mut png = grayscale_png();
    png.truncate(png.len() - 12);
    png
}

fn encoded_png(color_type: png::ColorType, pixels: &[u8]) -> Vec<u8> {
    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, 1, 1);
    encoder.set_color(color_type);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixels).unwrap();
    drop(writer);
    png
}

fn read_be_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn write_be_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
}

fn write_be_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn push_be_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_be_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

#[derive(Debug)]
struct ErrorSourceFixture;

impl std::fmt::Display for ErrorSourceFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("error source fixture")
    }
}

impl std::error::Error for ErrorSourceFixture {}

#[test]
fn runtime_errors_distinguish_semantic_unsupported_from_device_unavailable() {
    let unsupported = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::LayerFilter,
    ));
    let unavailable = Error::runtime_capability_unavailable(
        RuntimeCapabilityUnavailable::try_new(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::DeviceLost {
                reason: DeviceLossReason::Destroyed,
            },
        )
        .unwrap(),
    );

    assert_eq!(unsupported.code(), ErrorCode::UnsupportedPrimitive);
    assert!(unsupported.unsupported_primitive().is_some());
    assert_eq!(
        unsupported.runtime_capability_unavailable_diagnostic(),
        None
    );
    assert_eq!(unavailable.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(unavailable.unsupported_primitive(), None);
    assert_eq!(
        unavailable
            .runtime_capability_unavailable_diagnostic()
            .map(|diagnostic| diagnostic.operation()),
        Some(RuntimeOperation::SurfaceRendering)
    );
}

#[test]
fn runtime_diagnostic_constructor_rejects_every_unlisted_operation_reason_pair() {
    let operations = [
        RuntimeOperation::AdapterSelection,
        RuntimeOperation::SurfaceRendering,
        RuntimeOperation::SurfaceReadback,
        RuntimeOperation::SurfaceResume,
        RuntimeOperation::EffectRendering,
        RuntimeOperation::EffectTextureAllocation,
        RuntimeOperation::EffectPresentation,
    ];
    let reasons = [
        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Suspended,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::NonRenderable,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Uninitialized,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Occluded,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Lost,
        },
        RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        },
        RuntimeCapabilityUnavailableReason::DeviceFaulted {
            kind: GpuFaultKind::Validation,
        },
        RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
            kind: SurfaceIdentityMismatchKind::ForeignRenderer,
        },
        RuntimeCapabilityUnavailableReason::EffectFormatUnavailable {
            policy: EffectQualityPolicy::RequireHighPrecision,
        },
        RuntimeCapabilityUnavailableReason::TextureDimensionExceeded {
            requested: PhysicalSize::new(17, 19),
            maximum: 16,
        },
        RuntimeCapabilityUnavailableReason::SurfaceFormatUnavailable {
            format: Format::Bgra8,
        },
    ];

    for operation in operations {
        for reason in reasons {
            let result = RuntimeCapabilityUnavailable::try_new(operation, reason);
            if runtime_pair_is_listed(operation, reason) {
                let diagnostic = result.unwrap();
                assert_eq!(diagnostic.operation(), operation);
                assert_eq!(diagnostic.reason(), reason);
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidInput);
                assert!(error.invalid_value_diagnostic().is_some());
            }
        }
    }
}

#[test]
fn typed_error_codes_cannot_exist_without_their_matching_payload() {
    let runtime = RuntimeCapabilityUnavailable::try_new(
        RuntimeOperation::SurfaceRendering,
        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
    )
    .unwrap();
    let errors = [
        Error::invalid_value("field", "value", "must be valid"),
        Error::unsupported_render_primitive(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        )),
        Error::unresolved_resource(UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            "image",
        )),
        Error::degraded_quality(DegradedQuality::new(
            DegradedQualityKind::ReducedIntermediatePrecision,
            "reduced",
        )),
        Error::runtime_capability_unavailable(runtime),
    ];

    for error in &errors {
        let typed_payloads = [
            error.invalid_value_diagnostic().is_some(),
            error.unsupported_primitive().is_some(),
            error.unresolved_resource_diagnostic().is_some(),
            error.degraded_quality_diagnostic().is_some(),
            error.runtime_capability_unavailable_diagnostic().is_some(),
        ];
        assert_eq!(typed_payloads.iter().filter(|present| **present).count(), 1);
        match error.code() {
            ErrorCode::InvalidInput => assert!(typed_payloads[0]),
            ErrorCode::UnsupportedPrimitive => assert!(typed_payloads[1]),
            ErrorCode::UnresolvedResource => assert!(typed_payloads[2]),
            ErrorCode::DegradedQuality => assert!(typed_payloads[3]),
            ErrorCode::RuntimeCapabilityUnavailable => assert!(typed_payloads[4]),
            _ => panic!("semantic constructor returned a non-semantic code"),
        }
    }

    let backend_codes = [
        BackendErrorCode::DeviceCreateFailed,
        BackendErrorCode::RendererCreateFailed,
        BackendErrorCode::SurfaceCreateFailed,
        BackendErrorCode::SurfaceConfigureFailed,
        BackendErrorCode::SurfaceOutOfMemory,
        BackendErrorCode::SurfaceTimeout,
        BackendErrorCode::SurfaceOutdated,
        BackendErrorCode::ImageUploadFailed,
        BackendErrorCode::RenderFailed,
        BackendErrorCode::ReadbackFailed,
        BackendErrorCode::PresentFailed,
        BackendErrorCode::UnsupportedBackend,
    ];
    for code in backend_codes {
        let error = Error::new(code, "backend failure");
        assert!(!matches!(
            error.code(),
            ErrorCode::InvalidInput
                | ErrorCode::UnsupportedPrimitive
                | ErrorCode::UnresolvedResource
                | ErrorCode::DegradedQuality
                | ErrorCode::RuntimeCapabilityUnavailable
        ));
        assert!(error.invalid_value_diagnostic().is_none());
        assert!(error.unsupported_primitive().is_none());
        assert!(error.unresolved_resource_diagnostic().is_none());
        assert!(error.degraded_quality_diagnostic().is_none());
        assert!(error.runtime_capability_unavailable_diagnostic().is_none());
    }
}

#[test]
fn semantic_error_accessors_preserve_payloads() {
    let invalid = Error::invalid_value("radius", -1, "must be non-negative");
    let unsupported = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::LayerFilter,
    ));
    let unresolved = Error::unresolved_resource(UnresolvedResource::new(
        UnresolvedResourceKind::Image,
        "hero-image",
    ));
    let degraded = Error::degraded_quality(DegradedQuality::new(
        DegradedQualityKind::ReducedIntermediatePrecision,
        "rgba16float unavailable",
    ));

    assert_eq!(invalid.code(), ErrorCode::InvalidInput);
    assert_eq!(
        invalid.message(),
        "radius value -1 is invalid: must be non-negative"
    );
    assert_eq!(
        invalid.invalid_value_diagnostic().map(InvalidValue::field),
        Some("radius")
    );
    assert_eq!(unsupported.code(), ErrorCode::UnsupportedPrimitive);
    assert!(unsupported.unsupported_primitive().is_some());
    assert_eq!(unresolved.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        unresolved
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("hero-image")
    );
    assert_eq!(degraded.code(), ErrorCode::DegradedQuality);
    assert_eq!(
        degraded
            .degraded_quality_diagnostic()
            .map(DegradedQuality::kind),
        Some(DegradedQualityKind::ReducedIntermediatePrecision)
    );
}

#[test]
fn native_and_wasm_error_source_storage_preserves_source_contract() {
    #[cfg(not(target_arch = "wasm32"))]
    fn assert_send_sync<T: Send + Sync>() {}

    #[cfg(not(target_arch = "wasm32"))]
    assert_send_sync::<Error>();

    let error = Error::new(BackendErrorCode::RenderFailed, "backend failed")
        .with_source(ErrorSourceFixture);

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(error.message(), "backend failed");
    assert_eq!(
        std::error::Error::source(&error)
            .map(ToString::to_string)
            .as_deref(),
        Some("error source fixture")
    );
}

#[test]
fn text_run_bounds_distinguish_unspecified_empty_and_ink() {
    let unspecified = TextRunBounds::unspecified();
    let empty = TextRunBounds::empty();
    let ink_rect = Rect::new(-2.0, -3.0, 4.0, 5.0);
    let ink = TextRunBounds::try_ink(ink_rect).unwrap();

    assert_eq!(unspecified.kind(), TextRunBoundsKind::Unspecified);
    assert_eq!(empty.kind(), TextRunBoundsKind::Empty);
    assert_eq!(ink.kind(), TextRunBoundsKind::Ink);
    assert_eq!(unspecified.ink_rect(), None);
    assert_eq!(empty.ink_rect(), None);
    assert_eq!(ink.ink_rect(), Some(ink_rect));
    let non_finite_x = TextRunBounds::try_ink(Rect::new(f64::NAN, 0.0, 1.0, 1.0)).unwrap_err();
    let non_finite_y = TextRunBounds::try_ink(Rect::new(0.0, f64::INFINITY, 1.0, 1.0)).unwrap_err();
    let non_finite_width = TextRunBounds::try_ink(Rect::new(0.0, 0.0, f64::NAN, 1.0)).unwrap_err();
    let non_finite_height =
        TextRunBounds::try_ink(Rect::new(0.0, 0.0, 1.0, f64::NEG_INFINITY)).unwrap_err();
    let zero_width = TextRunBounds::try_ink(Rect::new(0.0, 0.0, 0.0, 1.0)).unwrap_err();
    let zero_height = TextRunBounds::try_ink(Rect::new(0.0, 0.0, 1.0, 0.0)).unwrap_err();
    assert_eq!(non_finite_x.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_y.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_width.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_height.code(), ErrorCode::InvalidInput);
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);
    assert_eq!(zero_height.code(), ErrorCode::InvalidInput);
    assert_eq!(
        non_finite_x
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds x")
    );
    assert_eq!(
        non_finite_y
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds y")
    );
    assert_eq!(
        non_finite_width
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds width")
    );
    assert_eq!(
        non_finite_height
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds height")
    );
    assert_eq!(
        zero_width
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds width")
    );
    assert_eq!(
        zero_height
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds height")
    );
    assert_eq!(
        UnresolvedResourceKind::TextRunInkBounds.label(),
        "text run ink bounds"
    );

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Bounded text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        ink,
    )
    .unwrap();
    let shadowed = TextShadowRun::try_new(
        run,
        ShadowList::try_new(vec![
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();

    assert_eq!(shadowed.run().bounds(), ink);
}

#[test]
fn reference_buffer_allocation_validates_positive_size_and_overflow() {
    let buffer = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 3)).unwrap();

    assert_eq!(buffer.physical_size(), PhysicalSize::new(2, 3));
    assert_eq!(buffer.byte_len(), 24);
    assert_eq!(buffer.pixel(1, 2).unwrap(), PremultipliedRgba8::TRANSPARENT);

    let zero_width = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(0, 1))
        .expect_err("zero-width reference buffers should be rejected");
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);

    let overflow =
        ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(u32::MAX, u32::MAX))
            .expect_err("overflow-sized reference buffers should be rejected before allocation");
    assert_eq!(overflow.code(), ErrorCode::InvalidInput);

    let wrong_data_len = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![PremultipliedRgba8::TRANSPARENT],
    )
    .expect_err("raw pixel data should match validated dimensions");
    assert_eq!(wrong_data_len.code(), ErrorCode::InvalidInput);
}

#[test]
fn reference_buffer_pixel_access_preserves_bounds_checks() {
    let mut buffer = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let pixel = PremultipliedRgba8::try_new(10, 20, 30, 40).unwrap();

    buffer.set_pixel(1, 1, pixel).unwrap();

    assert_eq!(buffer.pixel(1, 1).unwrap(), pixel);
    assert_eq!(
        buffer
            .pixel(2, 0)
            .expect_err("x outside width should fail")
            .code(),
        ErrorCode::InvalidInput
    );
    assert_eq!(
        buffer
            .set_pixel(0, 2, pixel)
            .expect_err("y outside height should fail")
            .code(),
        ErrorCode::InvalidInput
    );
}

#[test]
fn reference_premultiplied_pixels_apply_clamped_finite_opacity() {
    let pixel = PremultipliedRgba8::try_new(100, 60, 20, 200).unwrap();

    let invalid_pixel =
        PremultipliedRgba8::try_new(200, 0, 0, 128).expect_err("red must be premultiplied");
    assert_eq!(invalid_pixel.code(), ErrorCode::InvalidInput);

    assert_eq!(
        pixel.apply_opacity(0.5).unwrap(),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
    assert_eq!(pixel.apply_opacity(3.0).unwrap(), pixel);
    assert_eq!(
        pixel.apply_opacity(-1.0).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
    assert_eq!(
        pixel
            .apply_opacity(f32::NAN)
            .expect_err("non-finite opacity should be rejected")
            .code(),
        ErrorCode::InvalidInput
    );

    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![pixel, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    assert_eq!(
        buffer.apply_opacity(0.5).unwrap().pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
}

#[test]
fn reference_source_over_composition_handles_alpha_edges() {
    let destination = PremultipliedRgba8::try_new(20, 40, 60, 128).unwrap();
    assert_eq!(
        PremultipliedRgba8::TRANSPARENT.source_over(destination),
        destination
    );

    let source = PremultipliedRgba8::try_new(20, 10, 5, 64).unwrap();
    assert_eq!(source.source_over(PremultipliedRgba8::TRANSPARENT), source);

    let opaque_source = PremultipliedRgba8::try_new(120, 80, 40, 255).unwrap();
    assert_eq!(opaque_source.source_over(destination), opaque_source);

    let partial_source = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let partial_destination = PremultipliedRgba8::try_new(0, 64, 0, 128).unwrap();
    assert_eq!(
        partial_source.source_over(partial_destination),
        PremultipliedRgba8::try_new(128, 32, 0, 192).unwrap()
    );
}

#[test]
fn reference_buffer_source_over_preserves_transparent_edges() {
    let mut source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let mut destination =
        ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let green = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();

    source.set_pixel(0, 0, red).unwrap();
    destination.set_pixel(1, 1, green).unwrap();
    let composed = source.source_over(&destination).unwrap();

    assert_eq!(composed.pixel(0, 0).unwrap(), red);
    assert_eq!(composed.pixel(1, 1).unwrap(), green);
    assert_eq!(
        composed.pixel(0, 1).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
}

#[test]
fn reference_pixels_apply_plus_lighter_and_blend_modes_deterministically() {
    let source = PremultipliedRgba8::try_new(60, 30, 10, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 80, 40, 160).unwrap();
    let cases = [
        (
            BlendMode::Normal,
            PremultipliedRgba8::try_new(70, 70, 30, 208).unwrap(),
        ),
        (
            BlendMode::Plus,
            PremultipliedRgba8::try_new(80, 110, 50, 255).unwrap(),
        ),
        (
            BlendMode::Multiply,
            PremultipliedRgba8::try_new(37, 60, 25, 208).unwrap(),
        ),
        (
            BlendMode::Screen,
            PremultipliedRgba8::try_new(75, 101, 48, 208).unwrap(),
        ),
        (
            BlendMode::Overlay,
            PremultipliedRgba8::try_new(42, 70, 27, 208).unwrap(),
        ),
        (
            BlendMode::Darken,
            PremultipliedRgba8::try_new(42, 70, 30, 208).unwrap(),
        ),
        (
            BlendMode::Lighten,
            PremultipliedRgba8::try_new(70, 91, 44, 208).unwrap(),
        ),
    ];

    for (mode, expected) in cases {
        let blended = source.blend_over(destination, mode);

        assert_eq!(blended, expected, "unexpected {mode:?} blend result");
        assert_premultiplied(blended);
    }
}

#[test]
fn reference_blends_handle_transparent_and_opaque_alpha_edges() {
    let transparent = PremultipliedRgba8::TRANSPARENT;
    let source = PremultipliedRgba8::try_new(64, 32, 16, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 80, 40, 160).unwrap();
    let opaque_source = PremultipliedRgba8::try_new(200, 100, 50, 255).unwrap();
    let opaque_destination = PremultipliedRgba8::try_new(50, 150, 200, 255).unwrap();

    assert_eq!(
        transparent.blend_over(destination, BlendMode::Multiply),
        destination
    );
    assert_eq!(source.blend_over(transparent, BlendMode::Screen), source);
    assert_eq!(
        opaque_source.blend_over(opaque_destination, BlendMode::Multiply),
        PremultipliedRgba8::try_new(39, 59, 39, 255).unwrap()
    );
    assert_eq!(
        opaque_source.blend_over(opaque_destination, BlendMode::Overlay),
        PremultipliedRgba8::try_new(78, 127, 167, 255).unwrap()
    );
}

#[test]
fn reference_buffer_blend_over_rejects_mismatched_destination_size() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let error = source
        .blend_over(&destination, BlendMode::Multiply)
        .expect_err("blend buffers must map one-to-one to destination pixels");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("reference blend destination size")
    );
}

#[test]
fn reference_buffer_alpha_composites_reject_mismatched_buffer_sizes() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let source_in_error = source
        .source_in_alpha_of(&destination)
        .expect_err("source-in buffers must map one-to-one to destination alpha");
    let source_in_diagnostic = source_in_error
        .invalid_value_diagnostic()
        .expect("source-in mismatch should include invalid value details");

    assert_eq!(source_in_error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        source_in_diagnostic.field(),
        "reference source-in destination size"
    );
    assert_eq!(source_in_diagnostic.value(), "1x2");
    assert_eq!(source_in_diagnostic.invariant(), "must match source size");

    let destination_in_error = destination
        .destination_in_alpha_of(&source)
        .expect_err("destination-in buffers must map one-to-one to source alpha");
    let destination_in_diagnostic = destination_in_error
        .invalid_value_diagnostic()
        .expect("destination-in mismatch should include invalid value details");

    assert_eq!(destination_in_error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        destination_in_diagnostic.field(),
        "reference destination-in source size"
    );
    assert_eq!(destination_in_diagnostic.value(), "2x1");
    assert_eq!(
        destination_in_diagnostic.invariant(),
        "must match destination size"
    );
}

#[test]
fn reference_buffer_blend_over_and_alpha_composites_cover_partial_masks() {
    let red_half = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let green_half = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();
    let blue_opaque = PremultipliedRgba8::try_new(0, 0, 255, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![red_half, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![green_half, blue_opaque],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ],
    )
    .unwrap();

    let blended = source.blend_over(&destination, BlendMode::Lighten).unwrap();
    let source_in = source.source_in_alpha_of(&mask).unwrap();
    let destination_in = destination.destination_in_alpha_of(&mask).unwrap();

    assert_eq!(
        blended.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(128, 128, 0, 192).unwrap()
    );
    assert_eq!(blended.pixel(1, 0).unwrap(), blue_opaque);
    assert_eq!(
        source_in.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(64, 0, 0, 64).unwrap()
    );
    assert_eq!(
        destination_in.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(0, 64, 0, 64).unwrap()
    );
}

#[test]
fn reference_pixels_apply_source_in_and_destination_in_alpha_multiplication() {
    let source = PremultipliedRgba8::try_new(100, 60, 20, 200).unwrap();
    let destination = PremultipliedRgba8::try_new(0, 80, 40, 128).unwrap();

    assert_eq!(
        source.source_in_alpha_of(destination),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
    assert_eq!(
        destination.destination_in_alpha_of(source),
        PremultipliedRgba8::try_new(0, 63, 31, 100).unwrap()
    );
}

#[test]
fn reference_alpha_masks_handle_opaque_transparent_and_partial_mask_pixels() {
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let green = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();
    let blue = PremultipliedRgba8::try_new(0, 0, 200, 200).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![red, green, blue],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![
            PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(16, 8, 4, 64).unwrap(),
        ],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(masked.pixel(0, 0).unwrap(), red);
    assert_eq!(masked.pixel(1, 0).unwrap(), PremultipliedRgba8::TRANSPARENT);
    assert_eq!(
        masked.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(0, 0, 50, 50).unwrap()
    );
}

#[test]
fn reference_alpha_masks_preserve_premultiplied_color_ratios() {
    let source_pixel = PremultipliedRgba8::try_new(100, 50, 25, 200).unwrap();
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![source_pixel])
            .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 1),
        vec![PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap()],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(
        masked.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(50, 25, 13, 100).unwrap()
    );
    assert_premultiplied(masked.pixel(0, 0).unwrap());
}

#[test]
fn reference_alpha_masks_preserve_transparent_edges() {
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            red,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(
                masked.pixel(x, y).unwrap(),
                PremultipliedRgba8::TRANSPARENT,
                "unexpected masked edge at {x},{y}"
            );
        }
    }
}

#[test]
fn reference_alpha_masks_are_deterministic_across_repeated_runs() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(100, 20, 10, 100).unwrap(),
            PremultipliedRgba8::try_new(0, 64, 128, 128).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(10, 40, 80, 160).unwrap(),
        ],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 64).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let first = source.apply_alpha_mask(&mask).unwrap();
    let second = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(first, second);
}

#[test]
fn reference_alpha_masks_reject_mismatched_mask_buffer_size() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let error = source
        .apply_alpha_mask(&mask)
        .expect_err("mask buffers must map one-to-one to source pixels");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("reference alpha mask size")
    );
}

#[test]
fn image_buffer_rejects_short_long_and_overflowing_byte_lengths() {
    let errors = [
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![0; 7])
            .expect_err("short RGBA data must be rejected"),
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![0; 9])
            .expect_err("long RGBA data must be rejected"),
        ImageBuffer::try_new(PhysicalSize::new(0, 2), vec![0])
            .expect_err("zero-area image buffers must reject nonempty RGBA data"),
        ImageBuffer::try_new(PhysicalSize::new(u32::MAX, u32::MAX), Vec::new())
            .expect_err("overflowing RGBA byte lengths must be rejected"),
    ];

    for error in errors {
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert!(error.invalid_value_diagnostic().is_some());
    }
}

#[test]
fn image_buffer_accepts_exact_and_zero_area_lengths_and_round_trips_bytes() {
    let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let image = ImageBuffer::try_new(PhysicalSize::new(2, 1), rgba.clone()).unwrap();

    assert_eq!(image.size(), PhysicalSize::new(2, 1));
    assert_eq!(image.rgba(), rgba.as_slice());
    assert_eq!(image.into_rgba(), rgba);

    for size in [PhysicalSize::new(0, 2), PhysicalSize::new(2, 0)] {
        let empty = ImageBuffer::try_new(size, Vec::new()).unwrap();
        assert_eq!(empty.size(), size);
        assert!(empty.rgba().is_empty());
        assert!(empty.into_rgba().is_empty());
    }
}

#[test]
fn resolved_alpha_mask_execution_applies_materialized_alpha_buffer() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255,
        ],
    )
    .unwrap();
    let mask = Image::from_rgba(
        Size::new(3.0, 1.0),
        vec![
            0, 0, 0, 255, //
            0, 0, 0, 0, //
            0, 0, 0, 128,
        ],
    )
    .unwrap();

    let masked = reference::execute_transitional_resolved_mask_bridge_for_test(
        &source,
        Rect::new(0.0, 0.0, 3.0, 1.0),
        mask,
        Rect::new(0.0, 0.0, 3.0, 1.0),
    )
    .unwrap();

    assert_eq!(masked.size(), source.size());
    assert_eq!(
        masked.rgba(),
        &[
            255, 0, 0, 255, //
            0, 0, 0, 0, //
            0, 0, 255, 128,
        ]
    );
}

#[test]
fn resolved_alpha_mask_execution_accepts_independent_image_extent() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 0, 0, 255, 0, 255, 0, 255],
    )
    .unwrap();
    let mask = Image::from_rgba(Size::new(1.0, 2.0), vec![0, 0, 0, 255, 0, 0, 0, 255]).unwrap();

    let masked = reference::execute_transitional_resolved_mask_bridge_for_test(
        &source,
        Rect::new(0.0, 0.0, 2.0, 1.0),
        mask,
        Rect::new(0.0, 0.0, 2.0, 1.0),
    )
    .unwrap_or_panic_for_test("mask storage extent must remain independent from semantic bounds");

    assert_eq!(masked, source);
}

#[test]
fn resolved_alpha_mask_requires_finite_positive_local_bounds() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255])).unwrap();
    let invalid_bounds = [
        Rect::new(f64::NAN, 0.0, 1.0, 1.0),
        Rect::new(0.0, f64::INFINITY, 1.0, 1.0),
        Rect::new(0.0, 0.0, f64::NEG_INFINITY, 1.0),
        Rect::new(0.0, 0.0, 1.0, f64::NAN),
        Rect::new(0.0, 0.0, 0.0, 1.0),
        Rect::new(0.0, 0.0, 1.0, 0.0),
        Rect::new(0.0, 0.0, -1.0, 1.0),
        Rect::new(0.0, 0.0, 1.0, -1.0),
    ];
    let rejects_invalid_bounds = invalid_bounds
        .into_iter()
        .all(|bounds| ResolvedLayerAlphaMask::try_new(image.clone(), bounds).is_err());
    let zero_sized_image = Image::from_rgba(Size::new(0.0, 0.0), Arc::<[u8]>::from([])).unwrap();
    let accepts_zero_sized_image =
        ResolvedLayerAlphaMask::try_new(zero_sized_image, Rect::new(2.0, 3.0, 4.0, 5.0)).is_ok();

    assert!(
        rejects_invalid_bounds && accepts_zero_sized_image,
        "resolved masks accept invalid local bounds"
    );
}

#[test]
fn resolved_mask_normalization_preserves_image_identity_sampling_and_local_bounds() {
    let image = Image::from_rgba(
        Size::new(3.0, 2.0),
        Arc::<[u8]>::from([
            255, 255, 255, 255, 0, 0, 0, 64, 0, 0, 0, 128, 0, 0, 0, 192, 0, 0, 0, 224, 0, 0, 0, 255,
        ]),
    )
    .unwrap()
    .quality(ImageQuality::High)
    .extend(Extend::Reflect);
    let bounds = Rect::new(10.0, -4.0, 6.0, 3.0);
    let expected_id = image.id();
    let expected_bytes = image.bytes.clone();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(image, bounds).unwrap_or_panic_for_test(
                "the valid resolved-mask normalization fixture must install",
            ),
        ),
        |scene| {
            scene.fill(bounds, Color::BLACK);
        },
    );
    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test("the valid resolved-mask fixture must normalize");
    let [command::RenderCommand::Layer { layer, .. }] = normalized.commands.as_slice() else {
        panic!("the resolved-mask fixture must normalize to one layer");
    };
    let mask = layer
        .mask
        .as_ref()
        .unwrap_or_panic_for_test("the normalized layer must retain its resolved mask");
    let upload = mask.upload();
    let key = upload.cache_key();
    let preserves_contract = key.image_id() == expected_id
        && key.physical_size() == PhysicalSize::new(3, 2)
        && key.quality() == ImageQuality::High
        && key.extend() == Extend::Reflect
        && upload.bytes() == expected_bytes.as_ref()
        && upload.row_bytes() == 12
        && upload.byte_len() == 24
        && mask.semantic_bounds_for_contract_test() == bounds;

    assert!(
        preserves_contract,
        "mask normalization collapsed storage and semantic bounds"
    );
}

#[test]
fn transitional_resolved_mask_bridge_preserves_bounds_quality_extend_and_transform() {
    let source =
        ImageBuffer::try_new(PhysicalSize::new(9, 1), [255, 255, 255, 255].repeat(9)).unwrap();
    let mask_bytes = [128_u8, 0, 64, 128, 192, 48, 160, 255, 64]
        .into_iter()
        .flat_map(|alpha| [0, 0, 0, alpha])
        .collect::<Vec<_>>();
    let source_bounds = Rect::new(0.0, 0.0, 9.0, 1.0);
    let mask_bounds = Rect::new(1.4, 0.0, 6.2, 1.0);
    let mut outputs = Vec::new();
    for quality in [ImageQuality::Low, ImageQuality::Medium, ImageQuality::High] {
        for extend in [Extend::Pad, Extend::Repeat, Extend::Reflect] {
            let image = Image::from_rgba(Size::new(9.0, 1.0), mask_bytes.clone())
                .unwrap()
                .quality(quality)
                .extend(extend);
            let output = reference::execute_transitional_resolved_mask_bridge_for_test(
                &source,
                source_bounds,
                image,
                mask_bounds,
            )
            .unwrap_or_panic_for_test("the staged resolved-mask fixture must execute");
            outputs.push((
                quality,
                extend,
                (0..9)
                    .map(|x| pixel_alpha(&output, x, 0))
                    .collect::<Vec<_>>(),
            ));
        }
    }
    let observed = |quality, extend| {
        outputs
            .iter()
            .find_map(|(candidate_quality, candidate_extend, alpha)| {
                (*candidate_quality == quality && *candidate_extend == extend).then_some(alpha)
            })
            .unwrap_or_panic_for_test("every quality/extend pair must have one staged result")
    };
    let low_pad = observed(ImageQuality::Low, Extend::Pad);
    let low_repeat = observed(ImageQuality::Low, Extend::Repeat);
    let low_reflect = observed(ImageQuality::Low, Extend::Reflect);
    let medium_pad = observed(ImageQuality::Medium, Extend::Pad);
    let medium_repeat = observed(ImageQuality::Medium, Extend::Repeat);
    let medium_reflect = observed(ImageQuality::Medium, Extend::Reflect);
    let high_pad = observed(ImageQuality::High, Extend::Pad);
    let high_repeat = observed(ImageQuality::High, Extend::Repeat);
    let high_reflect = observed(ImageQuality::High, Extend::Reflect);

    let transform = Transform::translation(3.0, -2.0).unwrap();
    let transformed_mask = Image::from_rgba(Size::new(9.0, 1.0), mask_bytes)
        .unwrap()
        .quality(ImageQuality::High)
        .extend(Extend::Reflect);
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_transform(transform)
            .unwrap()
            .with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(transformed_mask, mask_bounds)
                    .unwrap_or_panic_for_test("the transformed staged mask fixture must install"),
            ),
        |scene| {
            scene.fill(source_bounds, Color::BLACK);
        },
    );
    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test("the transformed staged mask fixture must normalize");
    let [command::RenderCommand::Layer { layer, .. }] = normalized.commands.as_slice() else {
        panic!("the transformed mask fixture must normalize to one layer");
    };

    let outside_is_transparent = outputs
        .iter()
        .all(|(_, _, alpha)| alpha[0] == 0 && alpha[8] == 0);
    let sampling_is_preserved = low_pad == low_repeat
        && low_pad == low_reflect
        && medium_pad == medium_reflect
        && medium_pad != medium_repeat
        && high_pad != high_repeat
        && high_pad != high_reflect
        && high_repeat != high_reflect
        && low_pad != medium_pad
        && medium_pad != high_pad;
    assert!(
        outside_is_transparent && sampling_is_preserved && layer.transform == transform,
        "the staged bridge changed new mask semantics"
    );
}

#[test]
fn reference_composition_buffers_are_deterministic() {
    let red_half = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let blue_half = PremultipliedRgba8::try_new(0, 0, 128, 128).unwrap();
    let green = PremultipliedRgba8::try_new(0, 255, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![red_half, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![blue_half, green],
    )
    .unwrap();

    let first = source.source_over(&destination).unwrap();
    let second = source.source_over(&destination).unwrap();
    let faded = first.apply_opacity(0.5).unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(128, 0, 64, 192).unwrap()
    );
    assert_eq!(first.pixel(1, 0).unwrap(), green);
    assert_eq!(
        faded.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(64, 0, 32, 96).unwrap()
    );
}

#[test]
fn authored_layer_mask_and_filter_inputs_return_typed_diagnostics() {
    let cases = [
        (
            "layer mask",
            Layer::new()
                .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
                .unwrap(),
            UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ),
        ),
        (
            "layer filter",
            Layer::new()
                .try_filter(Filter::try_blur(2.0).unwrap())
                .unwrap(),
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        ),
    ];

    for (label, layer, unsupported) in cases {
        let mut scene = Scene::new();
        scene.layer(layer, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        });

        let error = scene
            .normalize(Capabilities::CURRENT)
            .expect_err("authored inputs must not imply mask or layer-effect execution");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(
            error.unsupported_primitive(),
            Some(unsupported),
            "{label} must retain its typed diagnostic"
        );
        assert!(
            error.message().contains(unsupported.label()),
            "{label} must retain its diagnostic label"
        );
    }
}

#[test]
fn invalid_value_errors_name_rejected_value() {
    let error = Error::invalid_value(
        "rectangle width",
        f64::NAN,
        "must be finite and non-negative",
    );

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(
        error.message().contains("rectangle width"),
        "error should name the rejected field: {}",
        error.message()
    );
    assert!(
        error.message().contains("NaN"),
        "error should include the rejected value: {}",
        error.message()
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rectangle width")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("NaN")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite and non-negative")
    );
}

#[test]
fn invalid_value_diagnostic_captures_non_finite_constructor_value() {
    let error =
        Point::try_new(f64::NAN, 0.0).expect_err("non-finite point coordinates should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "point x value NaN is invalid: must be finite"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("point x")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("NaN")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite")
    );
}

#[test]
fn invalid_value_diagnostic_captures_impossible_geometry_constructor_value() {
    let error = Rect::try_new(0.0, 0.0, -1.0, 1.0)
        .expect_err("negative rectangle dimensions should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "rectangle width value -1 is invalid: must be finite and non-negative"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rectangle width")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("-1")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite and non-negative")
    );
}

#[test]
fn invalid_value_constructor_captures_empty_list_invariant() {
    let error = Error::invalid_value("gradient stops", "[]", "must not be empty");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "gradient stops value [] is invalid: must not be empty"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("gradient stops")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("[]")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must not be empty")
    );
}

#[test]
fn invalid_value_existing_empty_list_constructor_preserves_invalid_input_message() {
    let error = Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(1.0, 1.0).unwrap(),
        vec![],
    )
    .expect_err("empty gradient stop lists should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.message(), "gradient stops must not be empty");
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("gradient stops")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("[]")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must not be empty")
    );
}

#[test]
fn unsupported_primitive_errors_name_operation() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::LayerMask,
    );
    let error = Error::unsupported_render_primitive(unsupported);

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(
        error.message().contains("layer mask"),
        "message should name the unsupported primitive: {}",
        error.message()
    );
}

#[test]
fn unresolved_resource_diagnostics_preserve_kind_identifier_and_message() {
    let cases = [
        (
            "image",
            UnresolvedResourceKind::Image,
            "hero.png",
            "image resource hero.png could not be resolved",
        ),
        (
            "mask",
            UnresolvedResourceKind::Mask,
            "#avatar-mask",
            "mask resource #avatar-mask could not be resolved",
        ),
        (
            "filter",
            UnresolvedResourceKind::Filter,
            "#blur",
            "filter resource #blur could not be resolved",
        ),
        (
            "clip",
            UnresolvedResourceKind::Clip,
            "#content-clip",
            "clip resource #content-clip could not be resolved",
        ),
    ];

    for (case, kind, identifier, expected_message) in cases {
        let diagnostic = UnresolvedResource::new(kind, identifier);
        let error = Error::unresolved_resource(diagnostic.clone());

        assert_eq!(
            (
                error.code(),
                error.unresolved_resource_diagnostic(),
                diagnostic.kind(),
                diagnostic.kind().label(),
                diagnostic.identifier(),
                error.message(),
            ),
            (
                ErrorCode::UnresolvedResource,
                Some(&diagnostic),
                kind,
                case,
                identifier,
                expected_message,
            ),
            "{case} resource diagnostic changed"
        );
    }
}

#[test]
fn degraded_quality_diagnostics_preserve_kind_value_and_message() {
    let cases = [
        (
            "reduced precision",
            DegradedQualityKind::ReducedIntermediatePrecision,
            "reduced intermediate precision",
            "rgba16float unavailable",
            "render quality degraded: reduced intermediate precision (rgba16float unavailable)",
        ),
        (
            "paint-space conversion",
            DegradedQualityKind::UnsupportedPaintSpaceConversion,
            "unsupported paint-space conversion",
            "display-p3 -> srgb",
            "render quality degraded: unsupported paint-space conversion (display-p3 -> srgb)",
        ),
    ];

    for (case, kind, expected_label, value, expected_message) in cases {
        let diagnostic = DegradedQuality::new(kind, value);
        let error = Error::degraded_quality(diagnostic.clone());

        assert_eq!(
            (
                error.code(),
                error.degraded_quality_diagnostic(),
                diagnostic.kind(),
                diagnostic.kind().label(),
                diagnostic.value(),
                error.message(),
            ),
            (
                ErrorCode::DegradedQuality,
                Some(&diagnostic),
                kind,
                expected_label,
                value,
                expected_message,
            ),
            "{case} degraded-quality diagnostic changed"
        );
    }
}

#[test]
fn transform_capabilities_name_2d_origin_skew_and_coordinate_tags() {
    let capabilities = Capabilities::CURRENT.transform_coordinate_spaces();

    assert!(capabilities.supports_affine_2d());
    assert!(capabilities.supports_transform_origin());
    assert!(capabilities.supports_skew());
    assert!(capabilities.supports_coordinate_space_tags());
    assert!(!capabilities.supports_transform_3d());
}

#[test]
fn geometry_capabilities_name_boolean_offset_and_hit_test_boundaries() {
    let capabilities = Capabilities::CURRENT;

    assert!(!capabilities.geometry_targets().supports_geometry_booleans());
    assert!(!capabilities.geometry_targets().supports_geometry_offsets());
    assert_eq!(
        capabilities.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned
    );
    assert_eq!(HitTestOwnership::RootOwned, HitTestOwnership::RootOwned);
}

#[test]
fn paint_capabilities_name_color_policy_and_conversion_boundaries() {
    let capabilities = Capabilities::CURRENT.paint_sources();

    assert!(capabilities.supports_solid_rgba());
    assert!(capabilities.supports_gradients());
    assert!(capabilities.supports_srgb_color_conversion());
    assert!(capabilities.supports_hsl_color_conversion());
    assert_eq!(
        capabilities.symbolic_color_policy(),
        SymbolicColorPolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_unresolved_symbolic_colors());
    assert!(!capabilities.supports_color_mix());
    assert!(!capabilities.supports_repeating_gradients());
}

#[test]
fn capabilities_map_unsupported_primitives_to_typed_errors() {
    let capabilities = Capabilities::CURRENT;
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::LayerMask,
    );

    let error = capabilities
        .ensure_supported(unsupported)
        .expect_err("layer masks are not supported in this milestone");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message().contains("layer mask"));
}

#[test]
fn unsupported_geometry_operations_report_typed_diagnostics() {
    let boolean = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryBooleanOperation,
    );
    let offset = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryOffsetOperation,
    );

    for unsupported in [boolean, offset] {
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("geometry operation should be explicitly unsupported");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn unsupported_symbolic_color_inputs_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::UnresolvedSymbolicColor,
        PrimitiveOperation::ColorMixFunction,
        PrimitiveOperation::UnsupportedColorSpace,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::PaintSources, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("symbolic or unsupported color input is not render-resolved");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn repeating_gradients_report_typed_diagnostics() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::PaintSources,
        PrimitiveOperation::RepeatingGradient,
    );

    let error = Capabilities::CURRENT
        .ensure_supported(unsupported)
        .expect_err("repeating gradients require unsupported normalization");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}

#[test]
fn unsupported_3d_transforms_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::Matrix3dTransform,
        PrimitiveOperation::PerspectiveTransform,
        PrimitiveOperation::Rotate3dTransform,
        PrimitiveOperation::TranslateZTransform,
        PrimitiveOperation::ScaleZTransform,
    ] {
        let unsupported =
            UnsupportedPrimitive::new(PrimitiveFamily::TransformsAndCoordinateSpaces, operation);

        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("3D transforms are unsupported in this render phase");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn draw_value_try_constructors_reject_invalid_values() {
    assert!(Shape::try_circle(Point::try_new(0.0, 0.0).unwrap(), -1.0).is_err());
    assert!(Color::try_rgba(2.0, 0.0, 0.0, 1.0).is_err());
    assert!(Stroke::try_new(0.0).is_err());
    assert!(Dash::try_new(0.0, &[1.0, f64::NAN]).is_err());
    assert!(GradientStop::try_new(1.5, Color::BLACK).is_err());
    assert!(
        Gradient::try_linear(
            Point::try_new(0.0, 0.0).unwrap(),
            Point::try_new(1.0, 1.0).unwrap(),
            vec![],
        )
        .is_err()
    );
    assert!(Layer::new().try_opacity(f32::NAN).is_err());
    assert!(Shadow::try_new(Point::try_new(0.0, 0.0).unwrap(), -1.0, 0.0, Color::BLACK).is_err());
    assert!(TextGlyph::try_new(1, 0.0, f32::NAN, 1.0).is_err());
    assert!(
        TextRun::try_new(
            FontRef::new(1),
            -1.0,
            Transform::identity(),
            TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap(),
            &[],
            TextRunBounds::unspecified(),
        )
        .is_err()
    );
}

#[test]
fn draw_value_constructors_preserve_valid_values() {
    let stroke = Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside);
    let stop = GradientStop::try_new(0.5, Color::BLACK).unwrap();
    let layer = Layer::new().try_opacity(0.5).unwrap();
    let text_paint = TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap();
    let glyph = TextGlyph::try_new(7, 1.0, 2.0, 3.0).unwrap();
    let glyphs = [glyph];
    let text_run = TextRun::try_new(
        FontRef::new(1),
        12.0,
        Transform::identity(),
        text_paint.clone(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();

    assert_eq!(stroke.width(), 2.0);
    assert_eq!(stop.offset(), 0.5);
    assert_eq!(layer.opacity(), 0.5);
    assert_eq!(text_paint.fill(), &Paint::color(Color::BLACK));
    assert_eq!(glyph.id(), 7);
    assert_eq!(text_run.size(), 12.0);
}

#[test]
fn image_ids_are_typed_resource_handles() {
    let image = Image::from_rgba(
        Size::try_new(1.0, 1.0).unwrap(),
        Arc::<[u8]>::from([0, 0, 0, 255]),
    )
    .unwrap();
    let id = image.id();

    assert_eq!(id.get(), image.id().get());
}

#[test]
fn font_refs_use_typed_font_ids() {
    let font = FontRef::new(FontId::new(42));

    assert_eq!(font.id(), FontId::new(42));
}

#[test]
fn text_shadow_run_model_preserves_text_run_and_shadow_order() {
    let glyph = TextGlyph::try_new(7, 1.0, 2.0, 3.0).unwrap();
    let glyphs = [glyph];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        12.0,
        Transform::identity(),
        TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(0.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();

    let text_shadow = TextShadowRun::try_new(run.clone(), shadows).unwrap();

    assert_eq!(text_shadow.run(), &run);
    assert_eq!(text_shadow.shadows().len(), 2);
    assert_eq!(text_shadow.shadows().shadows()[0], first);
    assert_eq!(text_shadow.shadows().shadows()[1], second);
}

#[test]
fn zero_blur_multi_text_shadow_preserves_authored_order_but_rejects_execution() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem ordered zero blur text shadows"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(
        Point::new(-2.0, 3.0),
        0.0,
        0.0,
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let text_shadow = TextShadowRun::try_new(run, shadows).unwrap();

    assert_eq!(
        text_shadow.shadows().shadows(),
        &[first.clone(), second.clone()]
    );

    let mut scene = Scene::new();
    scene.text_shadow_run(text_shadow);

    match &scene.commands[0] {
        scene::Command::TextShadowRun { shadows, .. } => {
            assert_eq!(shadows.shadows(), &[first, second]);
        }
        command => panic!("expected stored TextShadowRun, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("zero-blur text-shadow candidates must not emit render commands yet");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("zero-blur solid text shadows"));
    assert!(error.message().contains("not claimed or enabled"));
}

#[test]
fn transformed_text_shadow_inputs_are_stored_but_not_claimed_as_shifted_glyph_execution() {
    let text_transform = Transform::translation(4.0, 5.0)
        .unwrap()
        .then(Transform::skew_x(0.25).unwrap())
        .unwrap();
    let layer_transform = Transform::translation(10.0, -3.0).unwrap();
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 2.0, 10.0, 10.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem transformed text shadow"),
        16.0,
        text_transform,
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(2.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();

    scene.transform(layer_transform, |scene| {
        scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());
    });

    match &scene.commands[0] {
        scene::Command::Layer { layer, children } => {
            assert_eq!(layer.transform(), layer_transform);
            match &children[0] {
                scene::Command::TextShadowRun {
                    transform, glyphs, ..
                } => {
                    assert_eq!(*transform, text_transform);
                    assert_eq!(glyphs[0].id(), AHEM_GLYPH_X);
                }
                command => panic!("expected stored transformed TextShadowRun, got {command:?}"),
            }
        }
        command => panic!("expected transformed layer, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("transform-aware shifted glyph text-shadow execution is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("repeated shifted glyph draws"));
    assert!(error.message().contains("not claimed or enabled"));
}

#[test]
fn non_solid_or_spread_text_shadow_stays_on_glyph_alpha_offscreen_diagnostic_path() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(8.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap()).unwrap(),
        ],
    )
    .unwrap();
    let cases = [
        (
            "gradient text shadow paint",
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Paint::gradient(gradient)).unwrap(),
        ),
        (
            "spread text shadow",
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 2.0, Color::BLACK).unwrap(),
        ),
        (
            "blurred text shadow",
            Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        ),
    ];

    for (label, shadow) in cases {
        let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap()];
        let run = TextRun::try_new(
            ahem_font(label),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.text_shadow_run(
            TextShadowRun::try_new(run, ShadowList::try_new(vec![shadow]).unwrap()).unwrap(),
        );

        let error = match scene.normalize(Capabilities::CURRENT) {
            Ok(_) => panic!("{label} should stay unsupported"),
            Err(error) => error,
        };
        assert!(
            error
                .message()
                .contains("glyph-alpha/offscreen text capture"),
            "{label} used the wrong text-shadow diagnostic: {}",
            error.message()
        );
        assert!(
            !error.message().contains("zero-blur solid text shadows"),
            "{label} should not be classified as the shifted-glyph candidate path"
        );
    }
}

#[test]
fn text_shadow_run_reports_typed_unsupported_diagnostic() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem zero blur text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("text-shadow execution is not implemented in this phase");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("text shadow"));
    assert!(error.message().contains("zero-blur solid"));
    assert!(error.message().contains("repeated shifted glyph draws"));
}

#[test]
fn text_shadow_capability_claim_matches_current_diagnostic_boundary() {
    let unsupported =
        UnsupportedPrimitive::new(PrimitiveFamily::Shadows, PrimitiveOperation::TextShadow);
    assert!(!Capabilities::CURRENT.shadows().supports_text_shadows());

    let capability_error = Capabilities::CURRENT
        .ensure_supported(unsupported)
        .expect_err("text-shadow capability should stay false until execution exists");
    assert_eq!(capability_error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(capability_error.unsupported_primitive(), Some(unsupported));

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let normalize_error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("normalization should report the same unsupported text-shadow boundary");
    assert_eq!(normalize_error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(normalize_error.unsupported_primitive(), Some(unsupported));
    assert_eq!(
        normalize_error.unsupported_primitive(),
        capability_error.unsupported_primitive()
    );
    assert!(normalize_error.message().contains("zero-blur solid"));
    assert!(
        normalize_error
            .message()
            .contains("repeated shifted glyph draws")
    );
}

#[test]
fn blurred_text_shadow_reports_same_typed_boundary() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem blurred text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 4.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("blurred text-shadow needs glyph-alpha capture before pixel-moving blur");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("text shadow"));
    assert!(
        error
            .message()
            .contains("glyph-alpha/offscreen text capture")
    );
}

#[test]
fn text_shadow_run_command_storage_preserves_shadow_order_font_data_and_glyphs() {
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 12.0, 10.0, 10.0).unwrap(),
    ];
    let run = TextRun::try_new(
        ahem_font("Ahem stored text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(3.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(0.0, 4.0), 2.0, 0.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let mut scene = Scene::new();

    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    assert_eq!(scene.commands.len(), 1);
    match &scene.commands[0] {
        scene::Command::TextShadowRun {
            font,
            glyphs,
            shadows,
            ..
        } => {
            assert_eq!(font.id(), FontId::new(AHEM_FONT_ID));
            assert_eq!(font.name.as_deref(), Some("Ahem stored text shadow"));
            assert!(font.data.is_some());
            assert_eq!(glyphs.len(), 2);
            assert_eq!(glyphs[0].id(), AHEM_GLYPH_X);
            assert_eq!(glyphs[1].id(), AHEM_GLYPH_DESCENT_P);
            assert_eq!(shadows.shadows(), &[first, second]);
        }
        command => panic!("expected stored TextShadowRun, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("stored text-shadow ordering should be rejected only at normalization");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
}

#[test]
fn ordinary_text_run_normalization_remains_unaffected_by_text_shadow_boundary() {
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);

    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .expect("ordinary text runs should not use the text-shadow diagnostic");

    assert_eq!(normalized.commands.len(), 1);
    assert_eq!(normalized.stats().glyphs, 1);
    assert_eq!(normalized.stats().shadows, 0);
    assert!(matches!(
        normalized.commands[0],
        command::RenderCommand::TextRun { .. }
    ));
}

#[test]
fn ahem_text_run_preserves_font_data_and_stable_glyph_stream() {
    assert_eq!(AHEM_GLYPH_X, 58);
    assert_eq!(AHEM_GLYPH_DESCENT_P, 82);
    assert_eq!(AHEM_GLYPH_ASCENT_E_ACUTE, 100);

    let expected_glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_X, 2.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 14.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 26.0, 10.0, 10.0).unwrap(),
    ];
    let run = TextRun::try_new(
        ahem_font("Ahem stable glyph stream"),
        10.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &expected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);

    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .expect("Ahem text run with prepared glyphs should normalize");

    let [
        command::RenderCommand::TextRun {
            font,
            glyphs: encoded_glyphs,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("Ahem text should normalize as one text run");
    };
    assert_eq!(font.id(), FontId::new(AHEM_FONT_ID));
    assert!(font.data.is_some());
    assert_eq!(encoded_glyphs, &expected_glyphs);
}

#[test]
fn text_decoration_line_preserves_paint_thickness_transform_and_text_order() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 12.0),
        Point::new(32.0, 12.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let decoration = TextDecorationLine::try_solid(
        Point::new(2.0, 12.0),
        Point::new(34.0, 12.0),
        2.5,
        Transform::translation(3.0, 4.0).unwrap(),
        Paint::gradient(gradient.clone()),
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(1, 4.0, 10.0, 8.0).unwrap()];
    let text = TextRun::try_new(
        FontRef::new(1).named("Decoration order"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_decoration_line(decoration).text_run(text);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands.len(), 2);
    assert!(matches!(
        normalized.commands[1],
        command::RenderCommand::TextRun { .. }
    ));
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[0] else {
        panic!("transformed decoration should lower through a layer");
    };
    assert_eq!(layer.transform, Transform::translation(3.0, 4.0).unwrap());
    let [
        command::RenderCommand::Stroke {
            shape,
            stroke,
            paint,
        },
    ] = children.as_slice()
    else {
        panic!("decoration layer should contain one stroke command");
    };
    assert_eq!(stroke.width, 2.5);
    assert!(matches!(shape, command::RenderStrokeShape::Path(_)));
    assert_eq!(paint, &command::RenderPaint::Gradient(gradient));
}

#[test]
fn text_decoration_line_supports_solid_color_without_extra_text_semantics() {
    let decoration = TextDecorationLine::try_new(
        Point::new(1.0, 5.0),
        Point::new(9.0, 5.0),
        1.0,
        Transform::identity(),
        Color::BLACK.into(),
        TextDecorationLineStyle::Solid,
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_decoration_line(decoration);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    let [command::RenderCommand::Stroke { stroke, paint, .. }] = normalized.commands.as_slice()
    else {
        panic!("identity decoration should lower to a plain stroke");
    };
    assert_eq!(stroke.width, 1.0);
    assert_eq!(paint, &command::RenderPaint::Color(Color::BLACK));
}

#[test]
fn non_solid_text_decoration_styles_report_typed_boundary() {
    for style in [
        TextDecorationLineStyle::Double,
        TextDecorationLineStyle::Dotted,
        TextDecorationLineStyle::Dashed,
        TextDecorationLineStyle::Wavy,
    ] {
        let error = TextDecorationLine::try_new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            1.0,
            Transform::identity(),
            Color::BLACK.into(),
            style,
        )
        .expect_err("non-solid decoration styles require root/text expansion");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::TextDecorations,
                PrimitiveOperation::TextDecorationStyle,
            ))
        );
        assert!(error.message().contains("text decoration style"));
        assert!(error.message().contains("root/text"));
    }
}

#[test]
fn selection_and_generated_text_buckets_use_plain_render_capabilities() {
    let capabilities = Capabilities::CURRENT;
    assert!(capabilities.geometry_targets().supports_rect_fill_stroke());
    assert!(capabilities.paint_sources().supports_solid_rgba());
    assert!(
        !capabilities.shadows().supports_text_shadows(),
        "materialized selection/generated text buckets must not depend on text-shadow execution"
    );

    let selected_glyphs = [TextGlyph::try_new(10, 2.0, 10.0, 6.0).unwrap()];
    let generated_glyphs = [TextGlyph::try_new(11, 14.0, 10.0, 5.0).unwrap()];
    let selected_run = TextRun::try_new(
        FontRef::new(1).named("Selection"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap().into()).unwrap(),
        &selected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let generated_run = TextRun::try_new(
        FontRef::new(2).named("Generated"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &generated_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();

    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 12.0, 16.0), Color::BLACK)
        .text_run(selected_run)
        .text_run(generated_run);

    let normalized = scene
        .normalize(capabilities)
        .expect("materialized selection/generated content should normalize as ordinary commands");

    assert_eq!(normalized.commands.len(), 3);
    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().glyphs, 2);
    assert_eq!(normalized.stats().shadows, 0);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Fill { .. },
            command::RenderCommand::TextRun { .. },
            command::RenderCommand::TextRun { .. },
        ]
    ));
}

#[test]
fn materialized_selection_background_and_text_foreground_stay_ordered_commands() {
    let selected_glyphs = [
        TextGlyph::try_new(21, 4.0, 14.0, 7.0).unwrap(),
        TextGlyph::try_new(22, 11.0, 14.0, 6.0).unwrap(),
    ];
    let selected_text_paint =
        TextPaint::try_fill(Color::try_rgba(0.9, 0.96, 1.0, 1.0).unwrap().into()).unwrap();
    let selected_run = TextRun::try_new(
        FontRef::new(21).named("Root materialized selection text"),
        16.0,
        Transform::identity(),
        selected_text_paint.clone(),
        &selected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let selection_background = Rect::new(2.0, 2.0, 18.0, 18.0);
    let selection_background_paint = Color::try_rgba(0.0, 0.26, 0.72, 1.0).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(selection_background, selection_background_paint)
        .text_run(selected_run);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands.len(), 2);
    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().glyphs, 2);
    let [
        command::RenderCommand::Fill { shape, paint },
        command::RenderCommand::TextRun {
            font,
            paint: text_paint,
            glyphs,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("selection bucket should remain a fill followed by selected glyphs");
    };
    assert_eq!(
        shape,
        &command::RenderShape::Rect(selection_background),
        "selection highlight geometry is ordinary fill geometry"
    );
    assert_eq!(
        paint,
        &command::RenderPaint::Color(selection_background_paint),
        "selection highlight paint is ordinary fill paint"
    );
    assert_eq!(font.id(), FontId::new(21));
    assert_eq!(text_paint, &selected_text_paint);
    assert_eq!(glyphs, &selected_glyphs);
}

#[test]
fn materialized_generated_text_content_preserves_render_command_order() {
    let before_glyphs = [TextGlyph::try_new(31, 0.0, 12.0, 5.0).unwrap()];
    let principal_glyphs = [TextGlyph::try_new(32, 6.0, 12.0, 8.0).unwrap()];
    let after_glyphs = [TextGlyph::try_new(33, 15.0, 12.0, 5.0).unwrap()];
    let before = TextRun::try_new(
        FontRef::new(31).named("Generated before"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &before_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let principal = TextRun::try_new(
        FontRef::new(32).named("Principal text"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::try_rgba(0.1, 0.1, 0.1, 1.0).unwrap().into()).unwrap(),
        &principal_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let after = TextRun::try_new(
        FontRef::new(33).named("Generated after"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &after_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(before).text_run(principal).text_run(after);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.stats().glyphs, 3);
    let [
        command::RenderCommand::TextRun {
            font: before_font, ..
        },
        command::RenderCommand::TextRun {
            font: principal_font,
            ..
        },
        command::RenderCommand::TextRun {
            font: after_font, ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("generated and principal text should all normalize as text runs");
    };
    assert_eq!(before_font.id(), FontId::new(31));
    assert_eq!(principal_font.id(), FontId::new(32));
    assert_eq!(after_font.id(), FontId::new(33));
}

#[test]
fn materialized_generated_image_marker_and_text_content_are_ordinary_image_text_commands() {
    let marker_image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]),
    )
    .unwrap();
    let marker_id = marker_image.id();
    let marker_rect = Rect::new(0.0, 3.0, 4.0, 4.0);
    let item_glyphs = [TextGlyph::try_new(41, 8.0, 14.0, 9.0).unwrap()];
    let item_text = TextRun::try_new(
        FontRef::new(41).named("Generated list item text"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &item_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene
        .image(marker_image, marker_rect, ImageFit::Contain)
        .text_run(item_text);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().glyphs, 1);
    assert_eq!(normalized.stats().cache_misses, 1);
    let [
        command::RenderCommand::Image { image, rect, fit },
        command::RenderCommand::TextRun { font, glyphs, .. },
    ] = normalized.commands.as_slice()
    else {
        panic!("generated marker image and text should remain ordinary image/text commands");
    };
    assert_eq!(image.id(), marker_id);
    assert_eq!(*rect, marker_rect);
    assert_eq!(*fit, ImageFit::Contain);
    assert_eq!(font.id(), FontId::new(41));
    assert_eq!(glyphs, &item_glyphs);
}

#[test]
fn images_reject_incorrect_byte_lengths_and_fractional_dimensions() {
    let error = Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([0, 0, 0, 255]))
        .expect_err("wrong byte length should fail");

    assert_eq!(error.code(), ErrorCode::ImageUploadFailed);
    assert!(error.message().contains("expected 16 bytes"));

    let error = Image::from_rgba(Size::new(1.5, 2.0), Arc::<[u8]>::from([]))
        .expect_err("fractional source image size should fail");

    assert_eq!(error.code(), ErrorCode::ImageUploadFailed);
    assert!(error.message().contains("integer pixel size"));
}

fn runtime_pair_is_listed(
    operation: RuntimeOperation,
    reason: RuntimeCapabilityUnavailableReason,
) -> bool {
    match operation {
        RuntimeOperation::AdapterSelection => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Occluded
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceReadback => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Uninitialized
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceResume => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectTextureAllocation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectPresentation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
    }
}
