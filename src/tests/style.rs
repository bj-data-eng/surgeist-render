use crate::{
    BackgroundAreas, BackgroundAttachment, BackgroundBlendList, BackgroundBlendMode, BackgroundBox,
    BackgroundClipGeometry, BackgroundClipGeometryKind, BackgroundLayer,
    BackgroundNormalizationInput, BackgroundPosition, BackgroundRepeat, BackgroundSize,
    BackgroundStack, BorderEdges, BorderSide, BorderStyle, BoxDecorationBreak,
    BoxDecorationFragment, BoxDecorationInput, BoxSide, Capabilities, Color, CoordinateSpaceKind,
    CoordinateSpaceTag, ErrorCode, Image, ImageAttachmentPlan, ImageColorProfilePolicy, ImageId,
    ImageOrientationPolicy, ImagePlacementInput, ImageRepeatMode, ImageRepeatPlan,
    ImageResourceDensity, InvalidValue, NormalizedBackgroundCommandKind,
    NormalizedBackgroundLayerSource, NormalizedBorderCommand, NormalizedBorderStyle,
    NormalizedBoxDecorationCommand, NormalizedBoxDecorationCommandKind,
    NormalizedDoubleBorderBands, NormalizedOutlineCommand, NormalizedOutlineStyle, Outline,
    OutlineStyle, Paint, Path, Point, PositionComponentKind, PositionEdgeOffset, PrimitiveFamily,
    PrimitiveOperation, Radii, Rect, RepeatMode, ResolvedImagePlacement, ResolvedImageResource,
    Shape, Size, SizeComponent, StyleImageLayer, StyleImageSource, StyleImageSourceKind,
    StyleResourceRef, Transform, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
};

use super::{
    UnwrapOrPanicForTest,
    support::{box_decoration_edges, solid_border},
};

#[test]
fn style_reference_identifiers_must_not_be_empty() {
    let error = StyleResourceRef::try_new("  ").expect_err("empty identifiers are invalid");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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
    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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
        let plan = ImageRepeatPlan::try_new(repeat, Capabilities::CURRENT).unwrap();
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

    let repeat_x = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::CURRENT)
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

    let repeat_y = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_y(), Capabilities::CURRENT)
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

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::CURRENT)
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

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::CURRENT)
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

    let error = ImageRepeatPlan::try_new(BackgroundRepeat::repeat(), Capabilities::CURRENT)
        .unwrap()
        .resolve(placement)
        .expect_err("excessive repeat tiling must be rejected before allocation");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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
    let repeat = ImageRepeatPlan::try_new(layer.repeat(), Capabilities::CURRENT)
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
        Capabilities::CURRENT,
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
        Capabilities::CURRENT,
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
    assert_eq!(missing.code(), ErrorCode::InvalidInput);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("resolved image intrinsic size width")
    );
}

#[test]
fn background_position_rejects_non_finite_percent() {
    let error = BackgroundPosition::percent(f64::NAN, 0.0)
        .expect_err("non-finite percentages should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background position x percent")
    );
}

#[test]
fn background_size_rejects_negative_length() {
    let error = SizeComponent::try_length(-1.0)
        .expect_err("negative explicit background sizes should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background size length")
    );
}

#[test]
fn background_blend_lists_model_normal_layers_and_reject_blend_modes() {
    let list = BackgroundBlendList::try_new(vec![
        BackgroundBlendMode::Normal,
        BackgroundBlendMode::Normal,
    ])
    .unwrap_or_panic_for_test("normal-only background blending is a no-op model");

    assert_eq!(
        list.modes(),
        &[BackgroundBlendMode::Normal, BackgroundBlendMode::Normal]
    );

    let error = BackgroundBlendList::try_new(vec![
        BackgroundBlendMode::Normal,
        BackgroundBlendMode::Multiply,
    ])
    .expect_err("non-normal background blend execution is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::BackgroundBlendMode,
        ))
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();
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
    .normalize(Capabilities::CURRENT)
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
    .normalize(Capabilities::CURRENT)
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
    .normalize(Capabilities::CURRENT)
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
    .normalize(Capabilities::CURRENT)
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
    .normalize(Capabilities::CURRENT)
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
    .normalize(Capabilities::CURRENT)
    .expect_err("unresolved image layer should fail normalization");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background layer clip overrides")
    );
}

#[test]
fn background_normalization_preserves_authored_clip_override_geometry() {
    let mut path = Path::new();
    path.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0))
        .line_to(Point::new(10.0, 10.0))
        .close();
    let cases = [
        ("rectangle", Shape::rect(Rect::new(1.0, 1.0, 8.0, 8.0))),
        ("path", Shape::path(path)),
    ];

    for (case, shape) in cases {
        let layer = BackgroundLayer::new(
            StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
                .unwrap(),
        );
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
        .normalize(Capabilities::CURRENT)
        .unwrap();

        assert_eq!(
            normalized.commands()[0].clip().shape(),
            Some(&shape),
            "{case} clip geometry was not preserved",
        );
    }
}

#[test]
fn border_sides_reject_negative_width() {
    let error = BorderSide::try_new(BorderStyle::Solid, -1.0, Color::BLACK)
        .expect_err("negative border widths should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("border side width")
    );
}

#[test]
fn outlines_reject_non_finite_offset() {
    let error = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, f64::NAN)
        .expect_err("outline offset must be finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

fn normalized_border_command(command: &NormalizedBoxDecorationCommand) -> &NormalizedBorderCommand {
    match command.kind() {
        NormalizedBoxDecorationCommandKind::Border(border) => border,
        NormalizedBoxDecorationCommandKind::Outline(_) => panic!("expected border command"),
    }
}

fn normalized_outline_command(
    command: &NormalizedBoxDecorationCommand,
) -> &NormalizedOutlineCommand {
    match command.kind() {
        NormalizedBoxDecorationCommandKind::Outline(outline) => outline,
        NormalizedBoxDecorationCommandKind::Border(_) => panic!("expected outline command"),
    }
}

#[test]
fn box_decoration_normalization_emits_four_independent_border_sides_in_order() {
    let top = BorderSide::try_new(BorderStyle::Solid, 1.0, Color::BLACK).unwrap();
    let right = BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::TRANSPARENT).unwrap();
    let bottom = BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap();
    let left = BorderSide::try_new(BorderStyle::Double, 4.0, Color::TRANSPARENT).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(6.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            top.clone(),
            right.clone(),
            bottom.clone(),
            left.clone(),
        )),
        None,
        vec![fragment.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();
    let commands = normalized.commands();

    assert_eq!(commands.len(), 4);
    let top_command = normalized_border_command(&commands[0]);
    let right_command = normalized_border_command(&commands[1]);
    let bottom_command = normalized_border_command(&commands[2]);
    let left_command = normalized_border_command(&commands[3]);
    assert_eq!(top_command.side(), BoxSide::Top);
    assert_eq!(right_command.side(), BoxSide::Right);
    assert_eq!(bottom_command.side(), BoxSide::Bottom);
    assert_eq!(left_command.side(), BoxSide::Left);
    assert_eq!(top_command.width(), 1.0);
    assert_eq!(right_command.width(), 2.0);
    assert_eq!(bottom_command.width(), 3.0);
    assert_eq!(left_command.width(), 4.0);
    assert_eq!(top_command.paint(), top.paint());
    assert_eq!(right_command.paint(), right.paint());
    assert_eq!(bottom_command.paint(), bottom.paint());
    assert_eq!(left_command.paint(), left.paint());
    assert_eq!(top_command.style(), &NormalizedBorderStyle::Solid);
    assert_eq!(right_command.style(), &NormalizedBorderStyle::Dashed);
    assert_eq!(bottom_command.style(), &NormalizedBorderStyle::Dotted);
    assert!(matches!(
        left_command.style(),
        NormalizedBorderStyle::Double(_)
    ));
    assert_eq!(
        top_command.target_rect(),
        box_decoration_test_areas().border_box()
    );
    assert_eq!(top_command.fragment_index(), 0);
    assert_eq!(
        top_command.clip().rect(),
        Some(box_decoration_test_areas().border_box())
    );
    assert_eq!(top_command.radii(), fragment.radii());
    assert_eq!(top_command.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_suppresses_none_hidden_and_zero_width_borders() {
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::None, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Hidden, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Solid, 0.0, Color::BLACK).unwrap(),
            solid_border(3.0, Color::BLACK),
        )),
        None,
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands().len(), 1);
    let border = normalized_border_command(&normalized.commands()[0]);
    assert_eq!(border.side(), BoxSide::Left);
    assert_eq!(border.width(), 3.0);
}

#[test]
fn box_decoration_normalization_preserves_dashed_and_dotted_styles() {
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Hidden, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(
        normalized_border_command(&normalized.commands()[0]).style(),
        &NormalizedBorderStyle::Dashed
    );
    assert_eq!(
        normalized_border_command(&normalized.commands()[1]).style(),
        &NormalizedBorderStyle::Dotted
    );
}

fn assert_double_bands(bands: NormalizedDoubleBorderBands, width: f64) {
    assert_eq!(bands.original_width(), width);
    assert!(bands.outer_width() >= 0.0);
    assert!(bands.gap_width() >= 0.0);
    assert!(bands.inner_width() >= 0.0);
    let sum = bands.outer_width() + bands.gap_width() + bands.inner_width();
    assert!(
        (sum - width).abs() < f64::EPSILON,
        "double bands should sum to {width}, got {sum}"
    );
}

#[test]
fn box_decoration_normalization_computes_double_bands_for_thin_medium_and_large_widths() {
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(48.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::Double, 1.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Double, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Double, 9.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![fragment],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands().len(), 3);
    let thin = normalized_border_command(&normalized.commands()[0]);
    let medium = normalized_border_command(&normalized.commands()[1]);
    let large = normalized_border_command(&normalized.commands()[2]);
    let NormalizedBorderStyle::Double(thin_bands) = thin.style() else {
        panic!("expected thin double border bands");
    };
    let NormalizedBorderStyle::Double(medium_bands) = medium.style() else {
        panic!("expected medium double border bands");
    };
    let NormalizedBorderStyle::Double(large_bands) = large.style() else {
        panic!("expected large double border bands");
    };

    assert_double_bands(*thin_bands, 1.0);
    assert_double_bands(*medium_bands, 2.0);
    assert_double_bands(*large_bands, 9.0);
    assert!(thin_bands.outer_width() > 0.0);
    assert_eq!(large_bands.outer_width(), 3.0);
    assert_eq!(large_bands.gap_width(), 3.0);
    assert_eq!(large_bands.inner_width(), 3.0);
    assert_eq!(thin.radii().radii(), Radii::try_all(20.0).unwrap());
}

#[test]
fn box_decoration_normalization_reports_unsupported_border_styles() {
    for (style, operation) in [
        (BorderStyle::Groove, PrimitiveOperation::BorderGrooveStyle),
        (BorderStyle::Ridge, PrimitiveOperation::BorderRidgeStyle),
        (BorderStyle::Inset, PrimitiveOperation::BorderInsetStyle),
        (BorderStyle::Outset, PrimitiveOperation::BorderOutsetStyle),
    ] {
        let input = BoxDecorationInput::try_new(
            Some(box_decoration_edges(
                BorderSide::try_new(style, 2.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            )),
            None,
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let error = input
            .normalize(Capabilities::CURRENT)
            .expect_err("unsupported border styles should report typed diagnostics");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::BoxDecorations,
                operation,
            ))
        );
    }
}

#[test]
fn box_decoration_normalization_emits_borders_for_multiple_fragments_in_order() {
    let first = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(2.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(120.0, 0.0, 60.0, 40.0),
        Rect::new(124.0, 4.0, 52.0, 32.0),
        Rect::new(128.0, 8.0, 44.0, 24.0),
    )
    .unwrap();
    let shape = Shape::rect(Rect::new(120.0, 0.0, 60.0, 40.0));
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap()
    .with_border_clip_override(BackgroundClipGeometry::try_shape(shape.clone()).unwrap());
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![first, second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();
    let commands: Vec<_> = normalized
        .commands()
        .iter()
        .map(normalized_border_command)
        .collect();

    assert_eq!(
        commands
            .iter()
            .map(|command| (command.fragment_index(), command.side()))
            .collect::<Vec<_>>(),
        vec![
            (0, BoxSide::Top),
            (0, BoxSide::Bottom),
            (1, BoxSide::Top),
            (1, BoxSide::Bottom),
        ]
    );
    assert_eq!(commands[2].target_rect(), second_areas.border_box());
    assert_eq!(commands[2].clip().shape(), Some(&shape));
    assert_eq!(commands[2].radii(), second.radii());
    assert_eq!(commands[2].break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_expands_outline_target_by_offset_only() {
    let outline = Outline::try_new(OutlineStyle::Solid, 5.0, Color::BLACK, 3.0).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(6.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input =
        BoxDecorationInput::try_new(None, Some(outline.clone()), vec![fragment.clone()]).unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands().len(), 1);
    let command = normalized_outline_command(&normalized.commands()[0]);
    assert_eq!(command.fragment_index(), 0);
    assert_eq!(command.width(), 5.0);
    assert_eq!(command.offset(), 3.0);
    assert_eq!(command.paint(), outline.paint());
    assert_eq!(command.style(), NormalizedOutlineStyle::Solid);
    assert_eq!(command.target_rect(), Rect::new(-3.0, -3.0, 106.0, 46.0));
    assert_eq!(
        command.clip().rect(),
        Some(box_decoration_test_areas().border_box())
    );
    assert_eq!(command.radii(), fragment.radii());
    assert_eq!(command.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_keeps_outline_width_out_of_geometry() {
    let thin = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, 2.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();
    let thick = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 12.0, Color::BLACK, 2.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();

    assert_eq!(
        normalized_outline_command(&thin.commands()[0]).target_rect(),
        Rect::new(-2.0, -2.0, 104.0, 44.0)
    );
    assert_eq!(
        normalized_outline_command(&thick.commands()[0]).target_rect(),
        Rect::new(-2.0, -2.0, 104.0, 44.0)
    );
    assert_eq!(
        normalized_outline_command(&thick.commands()[0]).width(),
        12.0
    );
}

#[test]
fn box_decoration_normalization_preserves_dashed_and_dotted_outline_styles() {
    for (style, normalized_style) in [
        (OutlineStyle::Dashed, NormalizedOutlineStyle::Dashed),
        (OutlineStyle::Dotted, NormalizedOutlineStyle::Dotted),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(Outline::try_new(style, 2.0, Color::BLACK, 0.0).unwrap()),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let normalized = input.normalize(Capabilities::CURRENT).unwrap();

        assert_eq!(
            normalized_outline_command(&normalized.commands()[0]).style(),
            normalized_style
        );
    }
}

#[test]
fn box_decoration_normalization_reports_unsupported_outline_styles() {
    for (style, operation) in [
        (OutlineStyle::Double, PrimitiveOperation::OutlineDoubleStyle),
        (OutlineStyle::Auto, PrimitiveOperation::OutlineAutoStyle),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(Outline::try_new(style, 2.0, Color::BLACK, 0.0).unwrap()),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let error = input
            .normalize(Capabilities::CURRENT)
            .expect_err("unsupported outline styles should report typed diagnostics");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::BoxDecorations,
                operation,
            ))
        );
    }
}

#[test]
fn box_decoration_normalization_suppresses_none_and_zero_width_outlines() {
    for outline in [
        Outline::try_new(OutlineStyle::None, 2.0, Color::BLACK, 0.0).unwrap(),
        Outline::try_new(OutlineStyle::Solid, 0.0, Color::BLACK, 0.0).unwrap(),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(outline),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let normalized = input.normalize(Capabilities::CURRENT).unwrap();

        assert!(normalized.commands().is_empty());
    }
}

#[test]
fn box_decoration_normalization_handles_negative_outline_offsets_deterministically() {
    let valid = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 2.0, Color::BLACK, -4.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();

    assert_eq!(
        normalized_outline_command(&valid.commands()[0]).target_rect(),
        Rect::new(4.0, 4.0, 92.0, 32.0)
    );

    let invalid = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 2.0, Color::BLACK, -30.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let error = invalid
        .normalize(Capabilities::CURRENT)
        .expect_err("over-contracted outline target rects should be invalid");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("outline target rect")
    );
}

#[test]
fn box_decoration_normalization_emits_outline_after_borders_for_each_fragment() {
    let first = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(2.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(120.0, 0.0, 60.0, 40.0),
        Rect::new(124.0, 4.0, 52.0, 32.0),
        Rect::new(128.0, 8.0, 44.0, 24.0),
    )
    .unwrap();
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Solid, 3.0, Color::TRANSPARENT, 1.0).unwrap()),
        vec![first, second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.commands().len(), 4);
    assert_eq!(
        normalized_border_command(&normalized.commands()[0]).fragment_index(),
        0
    );
    assert_eq!(
        normalized_outline_command(&normalized.commands()[1]).fragment_index(),
        0
    );
    assert_eq!(
        normalized_border_command(&normalized.commands()[2]).fragment_index(),
        1
    );
    let second_outline = normalized_outline_command(&normalized.commands()[3]);
    assert_eq!(second_outline.fragment_index(), 1);
    assert_eq!(
        second_outline.target_rect(),
        Rect::new(119.0, -1.0, 62.0, 42.0)
    );
    assert_eq!(second_outline.radii(), second.radii());
    assert_eq!(second_outline.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn background_and_box_decoration_normalization_reuse_border_box_area() {
    let areas = BackgroundAreas::try_new(
        Rect::new(20.0, 30.0, 160.0, 90.0),
        Rect::new(26.0, 36.0, 148.0, 78.0),
        Rect::new(34.0, 44.0, 132.0, 62.0),
    )
    .unwrap();
    let background_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Border),
    );
    let background = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![background_layer]).unwrap(),
        areas,
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();
    let fragment = BoxDecorationFragment::try_new(
        areas,
        Radii::try_new(12.0, 14.0, 16.0, 18.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let decoration = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![fragment.clone()],
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();

    assert_eq!(background.commands().len(), 1);
    assert_eq!(
        background.commands()[0].clip().rect(),
        Some(areas.border_box())
    );
    let NormalizedBackgroundCommandKind::Layer { layer } = background.commands()[0].kind() else {
        panic!("expected mixed background layer command");
    };
    assert_eq!(layer.placement().paint_rect(), areas.content_box());

    let border = normalized_border_command(&decoration.commands()[0]);
    assert_eq!(border.target_rect(), areas.rect_for(BackgroundBox::Border));
    assert_eq!(border.clip().rect(), Some(areas.border_box()));
    assert_eq!(border.radii().border_box(), areas.border_box());
    assert_eq!(border.radii(), fragment.radii());
}

#[test]
fn background_box_decoration_integration_preserves_command_boundaries_across_fragments() {
    let first_areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 40.0),
        Rect::new(5.0, 5.0, 90.0, 30.0),
        Rect::new(10.0, 10.0, 80.0, 20.0),
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(110.0, 8.0, 70.0, 54.0),
        Rect::new(116.0, 14.0, 58.0, 42.0),
        Rect::new(122.0, 20.0, 46.0, 30.0),
    )
    .unwrap();
    let first = BoxDecorationFragment::try_new(
        first_areas,
        Radii::try_all(10.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_clip_shape = Shape::rect(Rect::new(111.0, 9.0, 68.0, 52.0));
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_new(18.0, 12.0, 10.0, 8.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap()
    .with_border_clip_override(
        BackgroundClipGeometry::try_shape(second_clip_shape.clone()).unwrap(),
    );
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            solid_border(3.0, Color::TRANSPARENT),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Dotted, 2.0, Color::BLACK, 1.5).unwrap()),
        vec![first.clone(), second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();
    let repeated = input.normalize(Capabilities::CURRENT).unwrap();

    assert_box_decoration_fragment_commands(
        normalized.commands(),
        repeated.commands(),
        &first,
        &second,
        first_areas,
        second_areas,
        &second_clip_shape,
    );
}

fn assert_box_decoration_fragment_commands(
    commands: &[NormalizedBoxDecorationCommand],
    repeated: &[NormalizedBoxDecorationCommand],
    first: &BoxDecorationFragment,
    second: &BoxDecorationFragment,
    first_areas: BackgroundAreas,
    second_areas: BackgroundAreas,
    second_clip_shape: &Shape,
) {
    assert_eq!(commands, repeated);
    assert_eq!(commands.len(), 6);
    assert_eq!(
        commands
            .iter()
            .map(|command| match command.kind() {
                NormalizedBoxDecorationCommandKind::Border(border) => {
                    (
                        "border",
                        border.fragment_index(),
                        Some(border.side()),
                        border.break_mode(),
                    )
                }
                NormalizedBoxDecorationCommandKind::Outline(outline) => {
                    (
                        "outline",
                        outline.fragment_index(),
                        None,
                        outline.break_mode(),
                    )
                }
            })
            .collect::<Vec<_>>(),
        vec![
            ("border", 0, Some(BoxSide::Top), BoxDecorationBreak::Slice),
            (
                "border",
                0,
                Some(BoxSide::Bottom),
                BoxDecorationBreak::Slice
            ),
            ("outline", 0, None, BoxDecorationBreak::Slice),
            ("border", 1, Some(BoxSide::Top), BoxDecorationBreak::Clone),
            (
                "border",
                1,
                Some(BoxSide::Bottom),
                BoxDecorationBreak::Clone
            ),
            ("outline", 1, None, BoxDecorationBreak::Clone),
        ]
    );

    for command in &commands[0..3] {
        match command.kind() {
            NormalizedBoxDecorationCommandKind::Border(border) => {
                assert_eq!(border.target_rect(), first_areas.border_box());
                assert_eq!(border.clip().rect(), Some(first_areas.border_box()));
                assert_eq!(border.radii(), first.radii());
            }
            NormalizedBoxDecorationCommandKind::Outline(outline) => {
                assert_eq!(outline.clip().rect(), Some(first_areas.border_box()));
                assert_eq!(outline.radii(), first.radii());
            }
        }
    }
    for command in &commands[3..] {
        match command.kind() {
            NormalizedBoxDecorationCommandKind::Border(border) => {
                assert_eq!(border.target_rect(), second_areas.border_box());
                assert_eq!(border.clip().shape(), Some(second_clip_shape));
                assert_eq!(border.radii(), second.radii());
            }
            NormalizedBoxDecorationCommandKind::Outline(outline) => {
                assert_eq!(outline.clip().shape(), Some(second_clip_shape));
                assert_eq!(outline.radii(), second.radii());
            }
        }
    }
}

#[test]
fn background_stacks_reject_empty_and_colorless_inputs() {
    let error = BackgroundStack::try_new(None, Vec::new())
        .expect_err("empty transparent background stacks should use no value");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background stack")
    );
}
