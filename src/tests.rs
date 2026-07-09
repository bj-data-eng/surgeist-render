use super::{
    backend::*,
    encode::*,
    surface::{HeadlessResources, SurfaceBackend},
};
use std::{sync::Arc, time::Duration};

use super::*;

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, ResizeState};
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
fn scene_normalization_rejects_unsupported_commands_before_encoding() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("unsupported masks should fail during normalization");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
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

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let stats = normalized.stats();

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
}

#[test]
fn surface_tracks_size_and_scale() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();

    surface.resize(Size::new(20.0, 30.0), 2.0).unwrap();

    assert_eq!(surface.size(), Size::new(20.0, 30.0));
    assert_eq!(surface.scale(), 2.0);
}

#[test]
fn surface_state_reports_availability_without_bool_peeking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(1.0, 1.0).unwrap(), 1.0)
        .unwrap();

    assert_eq!(surface.state(), SurfaceState::Available);
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
}

#[test]
fn headless_backend_resource_state_tracks_readiness() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(2.0, 2.0).unwrap(), 1.0)
        .unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    surface
        .resize(Size::try_new(3.0, 3.0).unwrap(), 1.0)
        .unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn presented_surface_lifecycle_state_names_pending_resize() {
    let state = PresentedLifecycle::ResizePending {
        physical_size: PhysicalSize::new(20, 10),
        resizing: ResizeState::Idle,
    };

    assert_eq!(
        state,
        PresentedLifecycle::ResizePending {
            physical_size: PhysicalSize::new(20, 10),
            resizing: ResizeState::Idle,
        }
    );
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn presented_surface_lifecycle_recovers_from_zero_size_at_current_native_size() {
    let state = PresentedLifecycle::NonRenderable {
        physical_size: PhysicalSize::new(0, 0),
        resizing: ResizeState::Resizing,
    };

    assert_eq!(
        state.resize_requested(PhysicalSize::new(640, 480), PhysicalSize::new(640, 480)),
        PresentedLifecycle::Ready {
            resizing: ResizeState::Resizing,
        }
    );
}

#[test]
fn headless_resize_keeps_target_when_physical_size_is_unchanged() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();

    surface.resize(Size::new(10.4, 10.4), 1.0).unwrap();

    assert_eq!(surface.size(), Size::new(10.4, 10.4));
    assert_eq!(surface.physical_size(), PhysicalSize::new(10, 10));
    assert!(matches!(
        &surface.backend,
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { .. },
            ..
        }
    ));
}

#[test]
fn create_surface_headless_preserves_surface_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let surface = renderer
        .create_surface(
            Attachment::Headless,
            SurfaceOptions {
                size: Size::new(10.0, 20.0),
                scale: 2.0,
                present_mode: PresentMode::Immediate,
                format: Format::Rgba8,
            },
        )
        .unwrap();

    assert_eq!(surface.size(), Size::new(10.0, 20.0));
    assert_eq!(surface.scale(), 2.0);
    assert_eq!(surface.options.present_mode, PresentMode::Immediate);
    assert_eq!(surface.options.format, Format::Rgba8);
    assert_eq!(surface.physical_size(), PhysicalSize::new(20, 40));
}

#[test]
fn rejects_invalid_surface_geometry() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let error = match renderer.create_headless(Size::new(f64::NAN, 10.0), 1.0) {
        Ok(_) => panic!("non-finite surface size should fail before physical conversion"),
        Err(error) => error,
    };

    assert_eq!(error.code, ErrorCode::InvalidInput);

    let mut surface = renderer.create_headless(Size::new(1.0, 1.0), 1.0).unwrap();
    let error = surface
        .resize(Size::new(1.0, 1.0), 0.0)
        .expect_err("invalid scale should fail before resize");

    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[test]
fn invalid_value_errors_name_rejected_value() {
    let error = Error::invalid_value(
        "rectangle width",
        f64::NAN,
        "must be finite and non-negative",
    );

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(
        error.message.contains("rectangle width"),
        "error should name the rejected field: {}",
        error.message
    );
    assert!(
        error.message.contains("NaN"),
        "error should include the rejected value: {}",
        error.message
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
fn error_type_stays_below_clippy_large_err_threshold() {
    assert!(
        std::mem::size_of::<Error>() <= 128,
        "Error should stay compact enough for crate-wide Result<T, Error>: {} bytes",
        std::mem::size_of::<Error>()
    );
}

#[test]
fn style_color_inputs_are_root_resolved_concrete_colors() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let input = StyleColor::new(color);

    assert_eq!(input.color(), color);
}

#[test]
fn symbolic_color_policy_keeps_style_colors_root_resolved() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let style_color = StyleColor::new(color);

    assert_eq!(style_color.color(), color);
    assert_eq!(
        StyleColor::symbolic_policy(),
        SymbolicColorPolicy::RootResolvedOnly
    );
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
fn normalized_paint_layers_reject_invalid_paint_sources() {
    let error = Gradient::try_linear(
        Point::new(f64::NAN, 0.0),
        Point::try_new(1.0, 0.0).unwrap(),
        vec![GradientStop::try_new(0.0, Color::BLACK).unwrap()],
    )
    .expect_err("invalid gradient construction should fail before paint layer");

    assert_eq!(error.code, ErrorCode::InvalidInput);
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
fn style_reference_identifiers_must_not_be_empty() {
    let error = StyleResourceRef::try_new("  ").expect_err("empty identifiers are invalid");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("style resource reference")
    );
}

#[test]
fn resolved_image_resources_preserve_handle_and_intrinsic_size() {
    let resource = ResolvedImageResource::try_new(ImageId::new(7), Size::new(24.0, 12.0)).unwrap();

    assert_eq!(resource.id(), ImageId::new(7));
    assert_eq!(resource.intrinsic_size(), Size::new(24.0, 12.0));
}

#[test]
fn resolved_image_resources_carry_root_resolved_metadata_policy() {
    let resource = ResolvedImageResource::try_new(ImageId::new(12), Size::new(40.0, 20.0))
        .unwrap()
        .with_density(ImageResourceDensity::try_new(2.0).unwrap());

    assert_eq!(resource.id(), ImageId::new(12));
    assert_eq!(resource.intrinsic_size(), Size::new(40.0, 20.0));
    assert_eq!(
        resource.density().map(ImageResourceDensity::value),
        Some(2.0)
    );
    assert_eq!(
        resource.orientation_policy(),
        ImageOrientationPolicy::RootResolvedOnly
    );
    assert_eq!(
        resource.color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
}

#[test]
fn image_resource_density_rejects_invalid_values() {
    let error = ImageResourceDensity::try_new(0.0)
        .expect_err("image density must be positive when supplied");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image resource density")
    );
}

#[test]
fn unresolved_style_image_sources_report_image_resource_diagnostics() {
    let reference = StyleResourceRef::try_new("hero.png").unwrap();
    let source = StyleImageSource::unresolved(reference.clone());

    assert_eq!(
        source.kind(),
        &StyleImageSourceKind::Unresolved(reference.clone())
    );

    let error = source
        .require_resolved()
        .expect_err("unresolved image source must report an image resource diagnostic");
    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(
        error.unresolved_resource_diagnostic(),
        Some(&UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            reference.identifier()
        ))
    );
}

#[test]
fn css_image_layers_preserve_sampling_inputs_without_lowering() {
    let resource = ResolvedImageResource::try_new(ImageId::new(11), Size::new(8.0, 8.0)).unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .with_position(BackgroundPosition::percent(0.25, 0.75).unwrap())
        .with_size(BackgroundSize::cover())
        .with_repeat(BackgroundRepeat::repeat_x())
        .with_origin(BackgroundBox::Padding)
        .with_clip(BackgroundBox::Content)
        .with_attachment(BackgroundAttachment::Fixed);

    assert_eq!(
        layer.source().kind(),
        &StyleImageSourceKind::Resolved(resource)
    );
    assert_eq!(layer.position().x().kind(), PositionComponentKind::Percent);
    assert_eq!(layer.position().y().value(), 0.75);
    assert_eq!(layer.size(), BackgroundSize::cover());
    assert_eq!(layer.repeat(), BackgroundRepeat::repeat_x());
    assert_eq!(layer.origin(), BackgroundBox::Padding);
    assert_eq!(layer.clip(), BackgroundBox::Content);
    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
}

#[test]
fn image_placement_auto_uses_intrinsic_size_and_position_ratio() {
    let input = ImagePlacementInput::try_new(
        Rect::new(10.0, 20.0, 100.0, 50.0),
        Size::new(20.0, 10.0),
        BackgroundPosition::percent(0.5, 1.0).unwrap(),
        BackgroundSize::auto(),
    )
    .unwrap();

    let placement = input.resolve().unwrap();

    assert_eq!(placement.paint_rect(), Rect::new(10.0, 20.0, 100.0, 50.0));
    assert_eq!(placement.tile_rect(), Rect::new(50.0, 60.0, 20.0, 10.0));
}

#[test]
fn image_placement_cover_and_contain_preserve_aspect_ratio() {
    let paint_rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let intrinsic = Size::new(20.0, 20.0);

    let cover = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::cover(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(cover.tile_rect(), Rect::new(0.0, -25.0, 100.0, 100.0));

    let contain = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::contain(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(contain.tile_rect(), Rect::new(25.0, 0.0, 50.0, 50.0));
}

#[test]
fn image_placement_explicit_size_resolves_lengths_percents_and_auto_axis() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::length(5.0, 10.0).unwrap(),
        BackgroundSize::explicit(
            SizeComponent::try_percent(0.5).unwrap(),
            SizeComponent::auto(),
        ),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(5.0, 10.0, 100.0, 50.0));
}

#[test]
fn image_placement_edge_offsets_represent_four_component_positions() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(-20.0, -10.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::edge_offsets(
            PositionEdgeOffset::end(15.0).unwrap(),
            PositionEdgeOffset::end(5.0).unwrap(),
        ),
        BackgroundSize::auto(),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(125.0, 65.0, 40.0, 20.0));
}

#[test]
fn image_placement_rejects_invalid_paint_or_intrinsic_size() {
    let error = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 0.0, 100.0),
        Size::new(10.0, 10.0),
        BackgroundPosition::default(),
        BackgroundSize::auto(),
    )
    .expect_err("paint rect must be positive");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image placement paint rect")
    );
}

#[test]
fn image_repeat_plan_maps_css_repeat_axes() {
    let cases = [
        (BackgroundRepeat::no_repeat(), ImageRepeatMode::NoRepeat),
        (BackgroundRepeat::repeat_x(), ImageRepeatMode::RepeatX),
        (BackgroundRepeat::repeat_y(), ImageRepeatMode::RepeatY),
        (BackgroundRepeat::repeat(), ImageRepeatMode::RepeatBoth),
    ];

    for (repeat, expected) in cases {
        let plan = ImageRepeatPlan::try_new(repeat, Capabilities::VELLO_0_9).unwrap();
        assert_eq!(plan.repeat(), repeat);
        assert_eq!(plan.mode(), expected);
    }
}

#[test]
fn image_repeat_plan_resolves_tile_rects_inside_clip_rect() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 70.0, 40.0),
        Rect::new(0.0, 5.0, 20.0, 10.0),
    )
    .unwrap();

    let repeat_x = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(repeat_x.clip_rect(), Rect::new(0.0, 0.0, 70.0, 40.0));
    assert_eq!(
        repeat_x.tile_rects(),
        &[
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(20.0, 5.0, 20.0, 10.0),
            Rect::new(40.0, 5.0, 20.0, 10.0),
            Rect::new(60.0, 5.0, 20.0, 10.0),
        ]
    );

    let repeat_y = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_y(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(
        repeat_y.tile_rects(),
        &[
            Rect::new(0.0, -5.0, 20.0, 10.0),
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(0.0, 15.0, 20.0, 10.0),
            Rect::new(0.0, 25.0, 20.0, 10.0),
            Rect::new(0.0, 35.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_includes_tiles_before_the_anchor_when_visible() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 50.0, 20.0),
        Rect::new(15.0, 0.0, 20.0, 10.0),
    )
    .unwrap();

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();

    assert_eq!(
        repeated.tile_rects(),
        &[
            Rect::new(-5.0, 0.0, 20.0, 10.0),
            Rect::new(15.0, 0.0, 20.0, 10.0),
            Rect::new(35.0, 0.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_fast_forwards_from_huge_negative_tile_origin() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 40.0, 10.0),
        Rect::new(-1_000_000_000_000.0, 0.0, 10.0, 10.0),
    )
    .unwrap();

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();

    assert_eq!(
        repeated.tile_rects(),
        &[
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(10.0, 0.0, 10.0, 10.0),
            Rect::new(20.0, 0.0, 10.0, 10.0),
            Rect::new(30.0, 0.0, 10.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_rejects_excessive_resolved_tile_count() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 250.25, 1_000.0),
        Rect::new(0.0, 0.0, 0.25, 1.0),
    )
    .unwrap();

    let error = ImageRepeatPlan::try_new(BackgroundRepeat::repeat(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .expect_err("excessive repeat tiling must be rejected before allocation");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image repeat tile count")
    );
}

#[test]
fn css_image_layer_normalizes_placement_repeat_and_attachment_together() {
    let resource = ResolvedImageResource::try_new(ImageId::new(90), Size::new(25.0, 10.0)).unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
        .with_size(BackgroundSize::explicit(
            SizeComponent::try_length(50.0).unwrap(),
            SizeComponent::auto(),
        ))
        .with_repeat(BackgroundRepeat::repeat_x())
        .with_attachment(BackgroundAttachment::Fixed)
        .with_coordinate_space(
            CoordinateSpaceTag::viewport(Transform::translation(2.0, 3.0).unwrap()).unwrap(),
        );

    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        resource.intrinsic_size(),
        layer.position(),
        layer.size(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    let repeat = ImageRepeatPlan::try_new(layer.repeat(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    let attachment =
        ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space()).unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(70.0, 0.0, 50.0, 20.0));
    assert_eq!(repeat.clip_rect(), Rect::new(0.0, 0.0, 120.0, 80.0));
    assert_eq!(
        repeat.tile_rects(),
        &[
            Rect::new(-30.0, 0.0, 50.0, 20.0),
            Rect::new(20.0, 0.0, 50.0, 20.0),
            Rect::new(70.0, 0.0, 50.0, 20.0),
        ]
    );
    assert_eq!(
        attachment.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn image_repeat_plan_rejects_round_and_space_with_typed_diagnostics() {
    let round = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::Round, RepeatMode::Repeat),
        Capabilities::VELLO_0_9,
    )
    .expect_err("round repeat is not supported yet");
    assert_eq!(
        round.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatRound
        ))
    );

    let space = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::NoRepeat, RepeatMode::Space),
        Capabilities::VELLO_0_9,
    )
    .expect_err("space repeat is not supported yet");
    assert_eq!(
        space.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatSpace
        ))
    );
}

#[test]
fn fixed_background_layers_can_carry_viewport_coordinate_space() {
    let layer =
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_attachment(BackgroundAttachment::Fixed)
            .with_coordinate_space(
                CoordinateSpaceTag::viewport(Transform::translation(10.0, 20.0).unwrap()).unwrap(),
            );

    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        layer.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn image_attachment_plan_uses_root_resolved_scroll_and_local_coordinates() {
    let scroll = ImageAttachmentPlan::try_new(BackgroundAttachment::Scroll, None).unwrap();
    assert_eq!(scroll.attachment(), BackgroundAttachment::Scroll);
    assert_eq!(
        scroll.coordinate_space().map(CoordinateSpaceTag::kind),
        None
    );

    let local_tag = CoordinateSpaceTag::local();
    let local = ImageAttachmentPlan::try_new(BackgroundAttachment::Local, Some(local_tag)).unwrap();
    assert_eq!(local.attachment(), BackgroundAttachment::Local);
    assert_eq!(
        local.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Local)
    );
}

#[test]
fn fixed_image_attachment_requires_viewport_coordinate_tag() {
    let missing = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, None)
        .expect_err("fixed backgrounds require an explicit viewport tag");
    assert_eq!(missing.code, ErrorCode::InvalidInput);
    assert_eq!(
        missing.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let surface = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let wrong = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(surface))
        .expect_err("fixed backgrounds must be tagged in viewport coordinates");
    assert_eq!(
        wrong.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let viewport = CoordinateSpaceTag::viewport(Transform::translation(3.0, 4.0).unwrap()).unwrap();
    let fixed = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(viewport)).unwrap();
    assert_eq!(fixed.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        fixed.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn resolved_image_resources_reject_invalid_intrinsic_size() {
    let error = ResolvedImageResource::try_new(ImageId::new(7), Size::new(f64::NAN, 12.0))
        .expect_err("invalid intrinsic size should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("resolved image intrinsic size width")
    );
}

#[test]
fn background_position_rejects_non_finite_percent() {
    let error = BackgroundPosition::percent(f64::NAN, 0.0)
        .expect_err("non-finite percentages should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background position x percent")
    );
}

#[test]
fn background_size_rejects_negative_length() {
    let error = SizeComponent::try_length(-1.0)
        .expect_err("negative explicit background sizes should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background size length")
    );
}

#[test]
fn filter_lists_distinguish_none_from_ordered_ops() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
    ])
    .unwrap();

    assert!(!list.is_none());
    assert_eq!(list.ops().len(), 2);
    assert!(matches!(
        list.ops()[0].kind(),
        FilterOpKind::Brightness(amount) if amount.value() == 1.2
    ));
    assert!(matches!(
        list.ops()[1].kind(),
        FilterOpKind::Blur(blur) if blur.radius() == 4.0
    ));
    assert!(FilterList::none().is_none());
    assert!(FilterList::none().ops().is_empty());
}

#[test]
fn filter_lists_reject_empty_ordered_ops() {
    let error = FilterList::try_ops(Vec::new()).expect_err("empty op lists must use none");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter operations")
    );
}

#[test]
fn filtered_image_paint_preserves_resolved_image_and_filter_list() {
    let resource = ResolvedImageResource::try_new(ImageId::new(30), Size::new(16.0, 16.0)).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.25).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource.clone(), filters.clone()).unwrap();

    assert_eq!(paint.resource(), &resource);
    assert_eq!(paint.filters(), &filters);
}

#[test]
fn filtered_image_paint_rejects_none_filter_list_and_reports_execution_boundary() {
    let resource = ResolvedImageResource::try_new(ImageId::new(31), Size::new(8.0, 8.0)).unwrap();
    let error = FilteredImagePaint::try_new(resource.clone(), FilterList::none())
        .expect_err("filtered image paint requires a non-empty filter list");
    assert_eq!(error.code, ErrorCode::InvalidInput);

    let filters = FilterList::try_ops(vec![FilterOp::contrast(
        FilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();
    let unsupported = paint
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("filtered image paint execution belongs to filter phases");
    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn filter_blur_rejects_negative_radius() {
    let error = FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_unit_amount_rejects_out_of_range_value() {
    let error = UnitFilterAmount::try_new(1.5)
        .expect_err("unit filter amounts must be clamped before render");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter unit amount")
    );
}

#[test]
fn filter_angle_rejects_nan() {
    let error = FilterAngle::try_radians(f64::NAN).expect_err("filter angles must be finite");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter angle")
    );
}

#[test]
fn clip_inputs_preserve_shape_or_unresolved_reference() {
    let shape = Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap());
    let clip = ClipInput::try_shape(shape.clone()).unwrap();
    let reference = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());

    assert_eq!(clip.shape(), Some(&shape));
    assert_eq!(
        reference.reference_ref().map(StyleResourceRef::identifier),
        Some("#clip")
    );
}

#[test]
fn mask_inputs_preserve_mode_and_source() {
    let mask = MaskInput::try_shape(
        Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap()),
        MaskMode::Luminance,
    )
    .unwrap();

    assert_eq!(mask.mode(), MaskMode::Luminance);
    assert!(matches!(mask.source().kind(), MaskSourceKind::Shape(_)));
}

#[test]
fn masks_and_clips_can_carry_coordinate_space_tags() {
    let tag = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap()
        .with_coordinate_space(tag);
    let mask = MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)), MaskMode::Alpha)
        .unwrap()
        .with_coordinate_space(tag);

    assert_eq!(clip.coordinate_space(), Some(tag));
    assert_eq!(mask.coordinate_space(), Some(tag));
}

#[test]
fn clip_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid clip paths fail");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn mask_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = MaskInput::try_shape(Shape::path(path), MaskMode::Alpha)
        .expect_err("invalid mask paths fail");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn paths_expose_elements_without_exposing_mutation() {
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

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn border_edges_preserve_four_independent_sides() {
    let top = BorderSide::try_new(BorderStyle::Solid, 1.0, Color::BLACK).unwrap();
    let right = BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::BLACK).unwrap();
    let bottom = BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap();
    let left = BorderSide::try_new(BorderStyle::Double, 4.0, Color::BLACK).unwrap();
    let edges = BorderEdges::new(top.clone(), right.clone(), bottom.clone(), left.clone());

    assert_eq!(edges.top(), &top);
    assert_eq!(edges.right(), &right);
    assert_eq!(edges.bottom(), &bottom);
    assert_eq!(edges.left(), &left);
}

#[test]
fn background_stacks_preserve_color_behind_ordered_layers() {
    let layer_a = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let layer_b = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack =
        BackgroundStack::try_new(Some(Color::BLACK), vec![layer_a.clone(), layer_b.clone()])
            .unwrap();

    assert_eq!(stack.color(), Some(Color::BLACK));
    assert_eq!(stack.layers(), &[layer_a, layer_b]);
}

#[test]
fn background_areas_select_origin_and_clip_boxes() {
    let areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        Rect::new(10.0, 8.0, 100.0, 60.0),
        Rect::new(20.0, 18.0, 80.0, 40.0),
    )
    .unwrap();

    assert_eq!(
        areas.rect_for(BackgroundBox::Border),
        Rect::new(0.0, 0.0, 120.0, 80.0)
    );
    assert_eq!(
        areas.rect_for(BackgroundBox::Padding),
        Rect::new(10.0, 8.0, 100.0, 60.0)
    );
    assert_eq!(
        areas.rect_for(BackgroundBox::Content),
        Rect::new(20.0, 18.0, 80.0, 40.0)
    );
}

#[test]
fn background_areas_reject_invalid_rects() {
    let error = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 100.0),
        Rect::new(0.0, 0.0, 0.0, 50.0),
        Rect::new(0.0, 0.0, 10.0, 10.0),
    )
    .expect_err("background areas require positive boxes");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background padding box")
    );
}

#[test]
fn background_clip_geometry_preserves_box_or_shape_inputs() {
    let rect_clip = BackgroundClipGeometry::try_rect(Rect::new(0.0, 0.0, 12.0, 8.0)).unwrap();
    assert_eq!(
        rect_clip.kind(),
        &BackgroundClipGeometryKind::Rect(Rect::new(0.0, 0.0, 12.0, 8.0))
    );

    let shape = Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0));
    let shape_clip = BackgroundClipGeometry::try_shape(shape.clone()).unwrap();
    assert_eq!(shape_clip.shape(), Some(&shape));
}

#[test]
fn background_stack_normalization_paints_color_behind_layers() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top, back]).unwrap();
    let input = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    assert_eq!(normalized.commands().len(), 3);
    let NormalizedBackgroundCommandKind::ColorFill { color, .. } = normalized.commands()[0].kind()
    else {
        panic!("expected background color command");
    };
    assert_eq!(*color, Color::BLACK);
    assert!(matches!(
        normalized.commands()[1].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
    assert!(matches!(
        normalized.commands()[2].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
}

#[test]
fn background_normalization_mixes_color_paint_and_image_layers_in_render_order() {
    let image = Image::from_rgba(Size::new(10.0, 10.0), vec![255; 10 * 10 * 4]).unwrap();
    let top_image = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image).unwrap())
            .unwrap()
            .with_size(BackgroundSize::auto())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let back_paint = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top_image, back_paint]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert!(matches!(
        normalized.commands()[0].kind(),
        NormalizedBackgroundCommandKind::ColorFill { .. }
    ));
    let NormalizedBackgroundCommandKind::Layer { layer: back_layer } =
        normalized.commands()[1].kind()
    else {
        panic!("expected back layer command");
    };
    assert!(matches!(
        back_layer.source(),
        NormalizedBackgroundLayerSource::Paint(_)
    ));

    let NormalizedBackgroundCommandKind::Layer { layer: top_layer } =
        normalized.commands()[2].kind()
    else {
        panic!("expected top layer command");
    };
    assert!(matches!(
        top_layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
}

#[test]
fn background_stack_normalization_preserves_top_layer_as_last_render_command() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Content),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Padding),
    );
    let stack = BackgroundStack::try_new(None, vec![top, back]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let last = normalized.commands().last().unwrap();
    assert_eq!(last.clip().rect(), Some(Rect::new(8.0, 8.0, 84.0, 44.0)));
}

#[test]
fn background_stack_normalization_preserves_paint_layer_sampling_semantics() {
    let paint_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 1.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_percent(0.5).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_y())
            .with_attachment(BackgroundAttachment::Local)
            .with_coordinate_space(CoordinateSpaceTag::local()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![paint_layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 60.0),
            Rect::new(20.0, 20.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind() else {
        panic!("expected normalized paint-backed layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Paint(_)
    ));
    assert_eq!(
        layer.placement().paint_rect(),
        Rect::new(20.0, 20.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(60.0, 40.0, 40.0, 20.0)
    );
    assert_eq!(
        layer.repeat().clip_rect(),
        Rect::new(20.0, 20.0, 80.0, 40.0)
    );
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Local);
}

#[test]
fn background_stack_normalizes_image_layers_with_origin_clip_repeat_and_attachment() {
    let image = Image::from_rgba(Size::new(20.0, 10.0), vec![255; 20 * 10 * 4]).unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image.clone()).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_length(40.0).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_x())
            .with_attachment(BackgroundAttachment::Fixed)
            .with_coordinate_space(
                CoordinateSpaceTag::viewport(Transform::translation(1.0, 2.0).unwrap()).unwrap(),
            ),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(5.0, 5.0, 90.0, 50.0),
            Rect::new(10.0, 10.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let command = normalized.commands().first().unwrap();
    assert_eq!(command.clip().rect(), Some(Rect::new(5.0, 5.0, 90.0, 50.0)));
    let NormalizedBackgroundCommandKind::Layer { layer } = command.kind() else {
        panic!("expected normalized image layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
    assert_eq!(
        layer.placement().paint_rect(),
        Rect::new(10.0, 10.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(50.0, 10.0, 40.0, 20.0)
    );
    assert_eq!(
        layer.repeat().clip_rect(),
        Rect::new(10.0, 10.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.repeat().tile_rects(),
        &[
            Rect::new(10.0, 10.0, 40.0, 20.0),
            Rect::new(50.0, 10.0, 40.0, 20.0),
        ]
    );
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Fixed);
}

#[test]
fn background_stack_normalizes_resolved_image_layers_with_intrinsic_size() {
    let resource =
        ResolvedImageResource::try_new(ImageId::new(400), Size::new(30.0, 10.0)).unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
            .unwrap()
            .with_origin(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(0.5, 0.5).unwrap())
            .with_size(BackgroundSize::contain())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 50.0),
            Rect::new(20.0, 20.0, 80.0, 30.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind() else {
        panic!("expected normalized layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::ResolvedImage(_)
    ));
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(10.0, 18.333333333333332, 100.0, 33.333333333333336)
    );
}

#[test]
fn background_stack_reports_unresolved_image_layers() {
    let source = StyleImageSource::unresolved(StyleResourceRef::try_new("hero.png").unwrap());
    let layer = BackgroundLayer::new(StyleImageLayer::try_new(source).unwrap());
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .expect_err("unresolved image layer should fail normalization");

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    let diagnostic = error.unresolved_resource_diagnostic().unwrap();
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Image);
    assert_eq!(diagnostic.identifier(), "hero.png");
}

#[test]
fn background_normalization_rejects_clip_override_length_mismatch() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(Vec::new())
    .expect_err("clip override list must match background layer count");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background layer clip overrides")
    );
}

#[test]
fn background_normalization_accepts_shape_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let shape = Shape::rect(Rect::new(1.0, 1.0, 8.0, 8.0));
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(
        BackgroundClipGeometry::try_shape(shape.clone()).unwrap(),
    )])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}

#[test]
fn background_normalization_accepts_path_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let mut path = Path::new();
    path.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0))
        .line_to(Point::new(10.0, 10.0))
        .close();
    let shape = Shape::path(path);
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(
        BackgroundClipGeometry::try_shape(shape.clone()).unwrap(),
    )])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}

#[test]
fn core_style_models_compose_without_backend_lowering() {
    let color = StyleColor::new(Color::BLACK);
    let paint = Paint::from(color.color());
    let image_layer = StyleImageLayer::try_new(StyleImageSource::paint(paint).unwrap()).unwrap();
    let background = BackgroundStack::try_new(
        Some(Color::TRANSPARENT),
        vec![BackgroundLayer::new(image_layer.clone())],
    )
    .unwrap();
    let filter = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(0.5).unwrap(),
    )])
    .unwrap();
    let mask = MaskInput::image_layer(image_layer, MaskMode::Alpha);
    let outline = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, 2.0).unwrap();

    assert_eq!(background.layers().len(), 1);
    assert_eq!(filter.ops().len(), 1);
    assert_eq!(mask.mode(), MaskMode::Alpha);
    assert_eq!(outline.offset(), 2.0);
}

#[test]
fn border_sides_reject_negative_width() {
    let error = BorderSide::try_new(BorderStyle::Solid, -1.0, Color::BLACK)
        .expect_err("negative border widths should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("border side width")
    );
}

#[test]
fn outlines_reject_non_finite_offset() {
    let error = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, f64::NAN)
        .expect_err("outline offset must be finite");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("outline offset")
    );
}

fn box_decoration_test_areas() -> BackgroundAreas {
    BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 40.0),
        Rect::new(4.0, 4.0, 92.0, 32.0),
        Rect::new(8.0, 8.0, 84.0, 24.0),
    )
    .unwrap()
}

#[test]
fn box_decoration_fragments_normalize_border_box_radii_on_construction() {
    let areas = box_decoration_test_areas();
    let radii = Radii::try_new(10.0, 12.0, 14.0, 16.0).unwrap();

    let fragment = BoxDecorationFragment::try_new(areas, radii, BoxDecorationBreak::Slice).unwrap();

    assert_eq!(fragment.areas(), areas);
    assert_eq!(fragment.radii().border_box(), areas.border_box());
    assert_eq!(fragment.radii().radii(), radii);
    assert_eq!(fragment.break_mode(), BoxDecorationBreak::Slice);
    assert_eq!(fragment.border_clip_override(), None);
}

#[test]
fn box_decoration_inputs_reject_empty_fragments() {
    let error = BoxDecorationInput::try_new(None, None, Vec::new())
        .expect_err("box decoration inputs require at least one fragment");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("box decoration fragments")
    );
}

#[test]
fn box_decoration_inputs_preserve_border_outline_and_break_mode() {
    let side = BorderSide::try_new(BorderStyle::Solid, 2.0, Color::BLACK).unwrap();
    let edges = BorderEdges::new(side.clone(), side.clone(), side.clone(), side);
    let outline = Outline::try_new(OutlineStyle::Dashed, 3.0, Color::TRANSPARENT, 1.5).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();

    let input = BoxDecorationInput::try_new(
        Some(edges.clone()),
        Some(outline.clone()),
        vec![fragment.clone()],
    )
    .unwrap();

    assert_eq!(input.border_edges(), Some(&edges));
    assert_eq!(input.outline(), Some(&outline));
    assert_eq!(input.fragments(), &[fragment]);
    assert_eq!(input.fragments()[0].break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_radii_scale_against_horizontal_and_vertical_limits() {
    let areas = box_decoration_test_areas();
    let radii = Radii::try_new(80.0, 80.0, 20.0, 20.0).unwrap();

    let fragment = BoxDecorationFragment::try_new(areas, radii, BoxDecorationBreak::Slice).unwrap();

    assert_eq!(
        fragment.radii().radii(),
        Radii::try_new(32.0, 32.0, 8.0, 8.0).unwrap()
    );
}

#[test]
fn box_decoration_fragments_validate_clip_override_geometry() {
    let error = BackgroundClipGeometry::try_rect(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("clip overrides reuse background clip validation");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background clip rect")
    );
}

#[test]
fn box_decoration_fragments_preserve_border_clip_override() {
    let shape = Shape::rect(Rect::new(1.0, 2.0, 30.0, 20.0));
    let clip = BackgroundClipGeometry::try_shape(shape.clone()).unwrap();

    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(5.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap()
    .with_border_clip_override(clip.clone());

    assert_eq!(fragment.border_clip_override(), Some(&clip));
    assert_eq!(
        fragment
            .border_clip_override()
            .and_then(|clip| clip.shape()),
        Some(&shape)
    );
}

#[test]
fn background_stacks_reject_empty_and_colorless_inputs() {
    let error = BackgroundStack::try_new(None, Vec::new())
        .expect_err("empty transparent background stacks should use no value");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background stack")
    );
}

#[test]
fn invalid_value_diagnostic_captures_non_finite_constructor_value() {
    let error =
        Point::try_new(f64::NAN, 0.0).expect_err("non-finite point coordinates should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.message,
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

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.message,
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

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.message,
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

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(error.message, "gradient stops must not be empty");
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

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(
        error.message.contains("layer mask"),
        "message should name the unsupported primitive: {}",
        error.message
    );
}

#[test]
fn unresolved_resource_diagnostics_name_image_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Image, "hero.png");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Image);
    assert_eq!(diagnostic.kind().label(), "image");
    assert_eq!(diagnostic.identifier(), "hero.png");
    assert_eq!(
        error.message,
        "image resource hero.png could not be resolved"
    );
}

#[test]
fn unresolved_resource_diagnostics_name_mask_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Mask, "#avatar-mask");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Mask);
    assert_eq!(diagnostic.kind().label(), "mask");
    assert_eq!(diagnostic.identifier(), "#avatar-mask");
    assert_eq!(
        error.message,
        "mask resource #avatar-mask could not be resolved"
    );
}

#[test]
fn unresolved_resource_diagnostics_name_filter_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Filter, "#blur");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Filter);
    assert_eq!(diagnostic.kind().label(), "filter");
    assert_eq!(diagnostic.identifier(), "#blur");
    assert_eq!(error.message, "filter resource #blur could not be resolved");
}

#[test]
fn unresolved_resource_diagnostics_name_clip_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Clip, "#content-clip");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Clip);
    assert_eq!(diagnostic.kind().label(), "clip");
    assert_eq!(diagnostic.identifier(), "#content-clip");
    assert_eq!(
        error.message,
        "clip resource #content-clip could not be resolved"
    );
}

#[test]
fn degraded_quality_diagnostics_name_fast_blur_clamps() {
    let diagnostic =
        DegradedQuality::new(DegradedQualityKind::FastBlurClamp, "radius 512px -> 128px");
    let error = Error::degraded_quality(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::DegradedQuality);
    assert_eq!(error.degraded_quality_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), DegradedQualityKind::FastBlurClamp);
    assert_eq!(diagnostic.kind().label(), "fast blur clamp");
    assert_eq!(diagnostic.value(), "radius 512px -> 128px");
    assert_eq!(
        error.message,
        "render quality degraded: fast blur clamp (radius 512px -> 128px)"
    );
}

#[test]
fn degraded_quality_diagnostics_name_software_fallbacks() {
    let diagnostic = DegradedQuality::new(DegradedQualityKind::SoftwareFallback, "layer filter");
    let error = Error::degraded_quality(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::DegradedQuality);
    assert_eq!(error.degraded_quality_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), DegradedQualityKind::SoftwareFallback);
    assert_eq!(diagnostic.kind().label(), "software fallback");
    assert_eq!(diagnostic.value(), "layer filter");
    assert_eq!(
        error.message,
        "render quality degraded: software fallback (layer filter)"
    );
}

#[test]
fn degraded_quality_diagnostics_name_unsupported_paint_space_conversions() {
    let diagnostic = DegradedQuality::new(
        DegradedQualityKind::UnsupportedPaintSpaceConversion,
        "display-p3 -> srgb",
    );
    let error = Error::degraded_quality(diagnostic.clone());

    assert_eq!(error.code, ErrorCode::DegradedQuality);
    assert_eq!(error.degraded_quality_diagnostic(), Some(&diagnostic));
    assert_eq!(
        diagnostic.kind(),
        DegradedQualityKind::UnsupportedPaintSpaceConversion
    );
    assert_eq!(
        diagnostic.kind().label(),
        "unsupported paint-space conversion"
    );
    assert_eq!(diagnostic.value(), "display-p3 -> srgb");
    assert_eq!(
        error.message,
        "render quality degraded: unsupported paint-space conversion (display-p3 -> srgb)"
    );
}

#[test]
fn renderer_reports_backend_capabilities_by_family() {
    let renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let capabilities = renderer.capabilities();

    assert!(capabilities.geometry_targets().supports_rect_fill_stroke());
    assert!(
        capabilities
            .geometry_targets()
            .supports_rounded_rect_fill_stroke()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_circle_ellipse_fill_stroke()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_arbitrary_path_fill()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_arbitrary_path_centered_stroke()
    );
    assert!(
        !capabilities
            .geometry_targets()
            .supports_arbitrary_path_inside_outside_stroke()
    );

    assert!(capabilities.paint_sources().supports_solid_rgba());
    assert!(capabilities.paint_sources().supports_gradients());
    assert!(capabilities.paint_sources().supports_image_paint());
    assert!(
        !capabilities
            .paint_sources()
            .supports_non_solid_shadow_paint()
    );
    assert!(
        capabilities
            .shadows()
            .supports_rect_rounded_circle_shadows()
    );
    assert!(!capabilities.shadows().supports_ellipse_path_shadows());

    assert!(!capabilities.filters().supports_layer_filters());
    assert!(capabilities.masks_clips().supports_shape_clips());
    assert!(!capabilities.masks_clips().supports_layer_masks());
    assert!(capabilities.compositing().supports_layer_opacity());
    assert!(capabilities.compositing().supports_blend_modes());
    assert!(capabilities.surfaces().supports_headless_surfaces());
    assert_eq!(
        capabilities.surfaces().supports_web_canvas_surfaces(),
        cfg!(all(feature = "render-web", target_arch = "wasm32"))
    );
}

#[test]
fn transform_capabilities_name_2d_origin_skew_and_coordinate_tags() {
    let capabilities = Capabilities::VELLO_0_9.transform_coordinate_spaces();

    assert!(capabilities.supports_affine_2d());
    assert!(capabilities.supports_transform_origin());
    assert!(capabilities.supports_skew());
    assert!(capabilities.supports_coordinate_space_tags());
    assert!(!capabilities.supports_transform_3d());
}

#[test]
fn geometry_capabilities_name_boolean_offset_and_hit_test_boundaries() {
    let capabilities = Capabilities::VELLO_0_9;

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
    let capabilities = Capabilities::VELLO_0_9.paint_sources();

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
fn image_sampling_capabilities_name_css_sampling_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.image_sampling();

    assert!(capabilities.supports_image_fit());
    assert!(capabilities.supports_background_position());
    assert!(capabilities.supports_background_size());
    assert!(capabilities.supports_repeat_xy());
    assert_eq!(
        capabilities.attachment_coordinate_policy(),
        BackgroundAttachmentCoordinatePolicy::RootResolvedOrTagged
    );
    assert_eq!(
        capabilities.image_orientation_policy(),
        ImageOrientationPolicy::RootResolvedOnly
    );
    assert_eq!(
        capabilities.image_color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_repeat_round());
    assert!(!capabilities.supports_repeat_space());
    assert!(!capabilities.supports_filtered_image_paint());
    assert!(!capabilities.supports_image_orientation_conversion());
    assert!(!capabilities.supports_image_color_profile_conversion());
}

#[test]
fn box_decoration_capability_accessors_name_supported_paint_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.box_decorations();

    assert!(capabilities.supports_border_none_hidden_styles());
    assert!(capabilities.supports_border_solid_style());
    assert!(capabilities.supports_border_dashed_dotted_styles());
    assert!(capabilities.supports_border_double_style());
    assert!(capabilities.supports_border_radii());
    assert!(capabilities.supports_outlines());
    assert!(capabilities.supports_outline_none_style());
    assert!(capabilities.supports_outline_solid_style());
    assert!(capabilities.supports_outline_dashed_dotted_styles());
    assert!(capabilities.supports_fragments());
}

#[test]
fn box_decoration_capability_accessors_name_unsupported_style_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.box_decorations();

    assert!(!capabilities.supports_border_groove_style());
    assert!(!capabilities.supports_border_ridge_style());
    assert!(!capabilities.supports_border_inset_style());
    assert!(!capabilities.supports_border_outset_style());
    assert!(!capabilities.supports_outline_double_style());
    assert!(!capabilities.supports_outline_auto_style());
}

#[test]
fn hit_test_geometry_is_root_owned_not_render_lowered() {
    assert_eq!(
        Capabilities::VELLO_0_9.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned
    );
}

#[test]
fn capabilities_map_unsupported_primitives_to_typed_errors() {
    let capabilities = Capabilities::VELLO_0_9;
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::LayerMask,
    );

    let error = capabilities
        .ensure_supported(unsupported)
        .expect_err("layer masks are not supported in this milestone");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message.contains("layer mask"));
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
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("geometry operation should be explicitly unsupported");
        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
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
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("symbolic or unsupported color input is not render-resolved");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn repeating_gradients_report_typed_diagnostics() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::PaintSources,
        PrimitiveOperation::RepeatingGradient,
    );

    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("repeating gradients require later normalization");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}

#[test]
fn unsupported_image_sampling_operations_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::BackgroundRepeatRound,
        PrimitiveOperation::BackgroundRepeatSpace,
        PrimitiveOperation::FilteredImagePaint,
        PrimitiveOperation::ImageOrientationConversion,
        PrimitiveOperation::ImageColorProfileConversion,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::ImageSampling, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this image sampling primitive");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message.contains(unsupported.label()));
    }
}

#[test]
fn unsupported_box_decoration_style_capability_diagnostics_are_typed() {
    for operation in [
        PrimitiveOperation::BorderGrooveStyle,
        PrimitiveOperation::BorderRidgeStyle,
        PrimitiveOperation::BorderInsetStyle,
        PrimitiveOperation::BorderOutsetStyle,
        PrimitiveOperation::OutlineDoubleStyle,
        PrimitiveOperation::OutlineAutoStyle,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this box-decoration style");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message.contains("box decorations"));
        assert!(error.message.contains(unsupported.label()));
    }
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

        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("3D transforms are unsupported in this render phase");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn vello_baseline_reports_current_unsupported_primitives() {
    let capabilities = Capabilities::VELLO_0_9;
    let cases = [
        UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ),
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::BoxDecorations,
            PrimitiveOperation::BorderGrooveStyle,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::BoxDecorations,
            PrimitiveOperation::OutlineAutoStyle,
        ),
    ];

    for unsupported in cases {
        let error = capabilities
            .ensure_supported(unsupported)
            .expect_err("Vello 0.9 should reject this primitive");
        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message.contains(unsupported.label()));
    }
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_unsupported_off_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("web canvas surfaces require render-web on wasm32");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message.contains("web canvas surface"));
}

#[cfg(all(feature = "render-web", target_arch = "wasm32"))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_supported_on_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect("web canvas surfaces are available with render-web on wasm32");
}

#[test]
fn unsupported_layer_masks_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(4.0, 2.0).unwrap(), 1.0)
        .unwrap();
    let mut scene = Scene::new();

    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert!(error.message.contains("layer mask"));
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

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("coordinate space id")
    );
}

#[test]
fn coordinate_space_tags_model_future_backdrop_capture_space() {
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
fn physical_size_try_from_logical_size_rejects_invalid_scale() {
    let error = PhysicalSize::try_from_logical(Size::try_new(10.0, 10.0).unwrap(), 0.0)
        .expect_err("scale zero should be rejected before conversion");
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[test]
fn physical_size_try_from_logical_size_rejects_u32_overflow() {
    let error =
        PhysicalSize::try_from_logical(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0)
            .expect_err("physical device pixels should fit in u32");
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[test]
fn create_headless_rejects_physical_size_overflow() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error =
        match renderer.create_headless(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0) {
            Ok(_) => panic!("physical device pixels should fit in u32"),
            Err(error) => error,
        };

    assert_eq!(error.code, ErrorCode::InvalidInput);
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
fn surface_resize_rejects_physical_size_overflow_without_mutating_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 20.0), 1.5)
        .unwrap();

    let error = surface
        .resize(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0)
        .expect_err("physical device pixels should fit in u32");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(surface.size(), Size::new(10.0, 20.0));
    assert_eq!(surface.scale(), 1.5);
    assert_eq!(surface.physical_size(), PhysicalSize::new(15, 30));
}

#[test]
fn vello_out_of_memory_maps_to_stable_surface_error() {
    let error = vello::Error::WgpuErrorFromScope(wgpu::Error::OutOfMemory {
        source: Box::new(std::io::Error::other("oom")),
    });

    assert_eq!(vello_error_code(&error), ErrorCode::SurfaceOutOfMemory);
    assert!(vello_error_message(&error).contains("memory"));
}

#[test]
fn create_headless_reports_unsupported_format() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error = match renderer.create_surface(
        Attachment::Headless,
        SurfaceOptions {
            format: Format::Bgra8,
            ..SurfaceOptions::default()
        },
    ) {
        Ok(_) => panic!("unsupported headless format should fail before wgpu validation"),
        Err(error) => error,
    };

    assert_eq!(error.code, ErrorCode::SurfaceCreateFailed);
    assert!(error.message.contains("Rgba8"));
}

#[test]
fn surface_suspend_and_resume_preserve_attachment_kind() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();
    let scene = Scene::new();

    surface.suspend().unwrap();
    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("suspended surfaces should be unavailable");

    assert_eq!(error.code, ErrorCode::SurfaceUnavailable);

    renderer
        .resume_surface(&mut surface, Attachment::Headless)
        .unwrap();
    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("resumed headless surface should render");

    let error = surface
        .resume(Attachment::from_web_canvas("canvas"))
        .expect_err("surface backend kind should not change on resume");

    assert_eq!(error.code, ErrorCode::SurfaceCreateFailed);
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn unsupported_web_canvas_attachment_reports_target_requirement() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let canvas = WebCanvas::new("preview");

    assert_eq!(canvas.id(), "preview");

    let error = match renderer.create_surface(
        Attachment::WebCanvas(canvas),
        SurfaceOptions {
            size: Size::new(10.0, 10.0),
            ..SurfaceOptions::default()
        },
    ) {
        Ok(_) => panic!("native test targets should not create web canvas surfaces"),
        Err(error) => error,
    };

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))
    );
    assert!(error.message.contains("web canvas surface"));
}

#[test]
fn render_reports_command_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 5.0, 5.0), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.stroke(
                Rect::new(1.0, 1.0, 3.0, 3.0),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("headless render should report stats");

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
    assert!(stats.frame_time >= stats.encode_time);
    assert!(stats.frame_time >= stats.render_time);
    assert_eq!(stats.present_time, Duration::ZERO);
}

#[test]
fn render_scales_logical_scene_to_physical_surface() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(20.0, 20.0), 2.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(output.size, PhysicalSize::new(40, 40));
    assert!(pixel_alpha(&output, 18, 18) > 0);
    assert_eq!(pixel_alpha(&output, 22, 22), 0);
}

#[test]
fn warm_image_reuse_reports_cache_hit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    assert_eq!(image_data(&image), image_data(&image.clone()));
    let mut scene = Scene::new();
    scene.image(
        image.clone(),
        Rect::new(0.0, 0.0, 1.0, 1.0),
        ImageFit::Stretch,
    );

    let cold = renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let warm = renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();

    assert_eq!(cold.cache_misses, 1);
    assert_eq!(warm.cache_hits, 1);
}

#[test]
fn failed_render_does_not_warm_image_reuse_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 4.0), 1.0).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut failing = Scene::new();
    failing.image(
        image.clone(),
        Rect::new(0.0, 0.0, 1.0, 1.0),
        ImageFit::Stretch,
    );
    failing.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &failing, Parameters::default())
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);

    let mut valid = Scene::new();
    valid.image(image, Rect::new(0.0, 0.0, 1.0, 1.0), ImageFit::Stretch);

    let stats = renderer
        .render(&mut surface, &valid, Parameters::default())
        .expect("valid render should still see cold image");

    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn rejects_malformed_rgba_images() {
    let error = Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([0, 0, 0, 255]))
        .expect_err("wrong byte length should fail");

    assert_eq!(error.code, ErrorCode::ImageUploadFailed);
    assert!(error.message.contains("expected 16 bytes"));

    let error = Image::from_rgba(Size::new(1.5, 2.0), Arc::<[u8]>::from([]))
        .expect_err("fractional source image size should fail");

    assert_eq!(error.code, ErrorCode::ImageUploadFailed);
    assert!(error.message.contains("integer pixel size"));
}

#[test]
fn rejects_malformed_scene_values() {
    let error = Color::try_rgba(f32::NAN, 0.0, 0.0, 1.0)
        .expect_err("invalid paint should fail at construction");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(error.message.contains("red channel"));
}

#[test]
fn concrete_color_paint_renders_without_color_realization() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("concrete color paint should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn gradient_paint_renders_with_transparent_stop() {
    let gradient = Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(2.0, 0.0).unwrap(),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), gradient);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("gradient paint should render");
}

#[test]
fn image_paint_lowers_to_brush() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Paint::image(image));

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.fills, 1);
    assert_eq!(stats.images, 1);
    assert!(pixel_alpha(&output, 0, 0) > 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
}

#[test]
fn image_brush_preserves_sampling_and_extend() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255]))
        .unwrap()
        .quality(ImageQuality::High)
        .extend(Extend::Reflect);

    let brush = image_brush(&image);

    assert_eq!(brush.sampler.quality, peniko::ImageQuality::High);
    assert_eq!(brush.sampler.x_extend, peniko::Extend::Reflect);
    assert_eq!(brush.sampler.y_extend, peniko::Extend::Reflect);
}

#[test]
fn cover_image_fit_clips_to_target_rect() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut pixels = Vec::new();
    for _ in 0..8 {
        pixels.extend_from_slice(&[255, 0, 0, 255]);
    }
    let image = Image::from_rgba(Size::new(4.0, 2.0), Arc::<[u8]>::from(pixels)).unwrap();
    let mut scene = Scene::new();
    scene.image(image, Rect::new(1.0, 0.0, 2.0, 2.0), ImageFit::Cover);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert_eq!(pixel_alpha(&output, 3, 0), 0);
}

#[test]
fn image_fit_transforms_use_uniform_scale() {
    let contain = image_transform(
        Size::new(4.0, 2.0),
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Contain,
    )
    .unwrap()
    .as_coeffs();
    let cover = image_transform(
        Size::new(4.0, 2.0),
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Cover,
    )
    .unwrap()
    .as_coeffs();

    assert_eq!(contain[0], 0.5);
    assert_eq!(contain[3], 0.5);
    assert_eq!(contain[5], 0.5);
    assert_eq!(cover[0], 1.0);
    assert_eq!(cover[3], 1.0);
    assert_eq!(cover[4], -1.0);
}

#[test]
fn layer_transform_moves_child_content() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.transform(
        Transform::try_new([1.0, 0.0, 0.0, 1.0, 2.0, 0.0]).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        },
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn composed_layer_transforms_render_in_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(6.0, 2.0), 1.0).unwrap();
    let transform = Transform::translation(1.0, 0.0)
        .unwrap()
        .then(Transform::scale(2.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 2.0), Color::BLACK);
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("composed transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn origin_wrapped_layer_transform_renders_about_origin() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 4.0), 1.0).unwrap();
    let transform = Transform::scale(2.0, 2.0)
        .unwrap()
        .around(Point::try_new(1.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(1.0, 1.0, 1.0, 1.0), Color::BLACK);
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("origin-wrapped transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
    assert!(pixel_alpha(&output, 2, 2) > 0);
}

#[test]
fn transformed_shape_clips_render_in_layer_space() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_transform(Transform::translation(2.0, 0.0).unwrap())
            .unwrap()
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("transformed clip should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn transformed_images_render_in_layer_space() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.transform(Transform::translation(2.0, 0.0).unwrap(), |scene| {
        scene.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("transformed image should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
}

#[test]
fn pure_transform_does_not_require_backend_layer() {
    let transform = Layer::new()
        .try_transform(Transform::try_new([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap())
        .unwrap();
    let clip = Layer::new()
        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap();
    let opacity = Layer::new().try_opacity(0.5).unwrap();

    let mut scene = Scene::new();
    scene
        .layer(transform, |_| {})
        .layer(clip, |_| {})
        .layer(opacity, |_| {});

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let isolations: Vec<_> = normalized
        .commands
        .iter()
        .map(|command| match command {
            command::RenderCommand::Layer { layer, .. } => layer.isolation,
            _ => panic!("expected layer command"),
        })
        .collect();

    assert_eq!(
        isolations,
        [
            command::LayerIsolation::None,
            command::LayerIsolation::ClipOnly,
            command::LayerIsolation::BackendLayer,
        ]
    );
}

#[test]
fn layer_default_is_visible() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::default(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("default layer should render visible content");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.layers, 1);
    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn layer_opacity_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("opacity layer should render");
    let output = renderer.read_headless(&surface).unwrap();
    let [_, _, _, alpha] = pixel_rgba(&output, 0, 0);

    assert_eq!(stats.layers, 1);
    assert!(alpha > 0);
    assert!(alpha < 255);
}

#[test]
fn layer_blend_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
    );
    scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap(),
        );
    });

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("blend layer should render");
    let output = renderer.read_headless(&surface).unwrap();
    let [red, green, blue, alpha] = pixel_rgba(&output, 0, 0);

    assert_eq!(stats.layers, 1);
    assert!(red < 32, "red channel should be multiplied down: {red}");
    assert!(
        green < 32,
        "green channel should be multiplied down: {green}"
    );
    assert!(blue < 32, "blue channel should be multiplied down: {blue}");
    assert!(alpha > 0);
}

#[test]
fn text_run_requires_font_data() {
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            FontRef::new(1).named("Test"),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("prepared glyphs cannot render without font data");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(error.message.contains("font data"));
}

#[test]
fn inside_and_outside_strokes_lower_for_builtin_shapes() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene
        .stroke(
            Rect::new(4.0, 4.0, 16.0, 16.0),
            Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
            Color::BLACK,
        )
        .stroke(
            Shape::try_circle(Point::new(12.0, 12.0), 6.0).unwrap(),
            Stroke::try_new(2.0).unwrap().align(StrokeAlign::Outside),
            Color::BLACK,
        );

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();

    assert_eq!(stats.strokes, 2);
}

#[test]
fn aligned_rect_strokes_do_not_cross_source_edge() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(12.0, 12.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let inside = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&inside, 2, 6), 0);
    assert!(pixel_alpha(&inside, 3, 6) > 0);

    let mut surface = renderer
        .create_headless(Size::new(12.0, 12.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Outside),
        Color::BLACK,
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let outside = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&outside, 2, 6) > 0);
    assert_eq!(pixel_alpha(&outside, 4, 6), 0);
}

#[test]
fn circle_shadows_lower_to_blurred_round_rect() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_circle(Point::new(12.0, 12.0), 4.0).unwrap(),
        Shadow::try_new(Point::new(1.0, 1.0), 4.0, 1.0, Color::BLACK).unwrap(),
    );

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn non_uniform_rounded_rect_shadows_render_with_corner_partition() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(40.0, 36.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_rounded_rect(
            Rect::new(8.0, 8.0, 16.0, 14.0),
            Radii::new(0.0, 5.0, 10.0, 0.0),
        )
        .unwrap(),
        Shadow::try_new(Point::new(4.0, 5.0), 8.0, 0.0, Color::BLACK).unwrap(),
    );

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("non-uniform rounded shadow should render through corner partitioning");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn direct_geometry_targets_render_without_unsupported_diagnostics() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(32.0, 32.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 30.0).unwrap())
        .close();

    scene.fill(
        Shape::rect(Rect::try_new(1.0, 1.0, 4.0, 4.0).unwrap()),
        Color::BLACK,
    );
    scene.stroke(
        Shape::rect(Rect::try_new(1.0, 7.0, 4.0, 4.0).unwrap()),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 1.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 7.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_circle(Point::try_new(4.0, 14.0).unwrap(), 2.0).unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_circle(Point::try_new(4.0, 20.0).unwrap(), 2.0).unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_ellipse(
            Point::try_new(14.0, 14.0).unwrap(),
            Size::try_new(3.0, 2.0).unwrap(),
        )
        .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_ellipse(
            Point::try_new(14.0, 20.0).unwrap(),
            Size::try_new(3.0, 2.0).unwrap(),
        )
        .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(Shape::path(path), Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("direct geometry targets should render");
}

#[test]
fn centered_path_strokes_support_join_cap_and_dash_inputs() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 20.0).unwrap());
    let stroke = Stroke::try_new(2.0)
        .unwrap()
        .join(LineJoin::Round)
        .caps(LineCap::Round, LineCap::Square)
        .try_dash(Dash::try_new(0.0, &[2.0, 1.0]).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.stroke(Shape::path(path), stroke, Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("centered path strokes should render");
}

#[test]
fn inside_outside_path_strokes_keep_typed_geometry_diagnostic() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(8.0, 8.0), 1.0).unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(1.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 6.0).unwrap())
        .close();
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(1.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("inside path stroke alignment requires offset lowering");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
}

#[test]
fn unsupported_aligned_path_strokes_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(10.0, 10.0));
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("path offsetting is deliberately explicit");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
    assert!(
        error
            .message
            .contains("inside/outside path stroke alignment")
    );
}

#[test]
fn unsupported_layer_masks_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("mask lowering should be explicit until implemented");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert!(error.message.contains("layer mask"));
}

#[test]
fn unsupported_layer_filters_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_filter(Filter::try_blur(4.0).unwrap())
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("filter lowering should be explicit until implemented");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );
    assert!(error.message.contains("layer filter"));
}

#[test]
fn unsupported_non_solid_shadow_paint_reports_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 4.0), 1.0).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Paint::gradient(gradient)).unwrap(),
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("shadow lowering requires solid paint in this milestone");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
    assert!(error.message.contains("non-solid shadow paint"));
}

#[test]
fn unsupported_shadow_shapes_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(8.0, 8.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_ellipse(Point::new(4.0, 4.0), Size::new(2.0, 1.0)).unwrap(),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("ellipse shadows should remain unsupported in this milestone");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ))
    );
    assert!(error.message.contains("ellipse/path shadow shape"));
}

#[test]
fn unsupported_path_shadows_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(8.0, 8.0), 1.0).unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(6.0, 1.0))
        .line_to(Point::new(6.0, 6.0))
        .close();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::path(path),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("path shadows should remain unsupported in this milestone");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ))
    );
    assert!(error.message.contains("ellipse/path shadow shape"));
}

#[test]
fn headless_render_can_be_read_back() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 4.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .unwrap();
    let image = renderer.read_headless(&surface).unwrap();

    assert_eq!(image.size, PhysicalSize::new(4, 4));
    assert_eq!(image.rgba.len(), 4 * 4 * 4);
    assert!(image.rgba.iter().any(|channel| *channel != 0));
}

fn pixel_alpha(image: &ImageBuffer, x: u32, y: u32) -> u8 {
    pixel_rgba(image, x, y)[3]
}

fn pixel_rgba(image: &ImageBuffer, x: u32, y: u32) -> [u8; 4] {
    let index = ((y * image.size.width() + x) * 4 + 3) as usize;
    [
        image.rgba[index - 3],
        image.rgba[index - 2],
        image.rgba[index - 1],
        image.rgba[index],
    ]
}
