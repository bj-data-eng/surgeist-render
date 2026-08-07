use std::sync::Arc;

use crate::{
    BackdropCaptureBounds, BackdropFilterInput, BackgroundAreas, BackgroundAttachment,
    BackgroundAttachmentCoordinatePolicy, BackgroundBlendList, BackgroundBlendMode, BackgroundBox,
    BackgroundClipGeometry, BackgroundClipGeometryKind, BackgroundLayer,
    BackgroundNormalizationInput, BackgroundPosition, BackgroundRepeat, BackgroundSize,
    BackgroundStack, BorderEdges, BorderSide, BorderStyle, BoxDecorationBreak,
    BoxDecorationFragment, BoxDecorationInput, BoxSide, Capabilities, ClipGeometryKind, ClipInput,
    Color, CoordinateSpaceKind, CoordinateSpaceTag, ErrorCode, FillRule, FilledPath, Filter,
    FilterAmount, FilterAngle, FilterBlur, FilterDropShadow, FilterList, FilterOp, FilterOpKind,
    FilteredImagePaint, FontRef, Gradient, GradientStop, Image, ImageAttachmentPlan,
    ImageColorProfilePolicy, ImageId, ImageOrientationPolicy, ImagePlacementInput, ImageRepeatMode,
    ImageRepeatPlan, ImageResourceDensity, InvalidValue, Layer, MaskCompositeMode, MaskInput,
    MaskLayer, MaskLayerStack, MaskMode, MaskSourceKind, NormalizedBackgroundCommandKind,
    NormalizedBackgroundLayerSource, NormalizedBorderCommand, NormalizedBorderStyle,
    NormalizedBoxDecorationCommand, NormalizedBoxDecorationCommandKind,
    NormalizedDoubleBorderBands, NormalizedOutlineCommand, NormalizedOutlineStyle, Outline,
    OutlineStyle, Paint, Path, Point, PositionComponentKind, PositionEdgeOffset, PrimitiveFamily,
    PrimitiveOperation, Radii, Rect, RepeatMode, ResolvedImagePlacement, ResolvedImageResource,
    Scene, Shadow, ShadowList, Shape, Size, SizeComponent, Stroke, StrokeAlign, StyleColor,
    StyleImageLayer, StyleImageSource, StyleImageSourceKind, StyleResourceRef, SymbolicColorPolicy,
    TextGlyph, TextPaint, TextRun, TextRunBounds, TextShadowRun, Transform, UnitFilterAmount,
    UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive, command,
    filter::{
        BlurPolicy, BlurRadiusInterpretation, FilterClipBounds, FilterOutset, FilterRegionPlan,
        FilterSourceBounds, KernelSupportRadius, LargeBlurRadiusAction, LargeBlurRadiusPolicy,
        TransparentEdgeSamplingPolicy,
    },
    style::ColorFilterOp,
};

use super::{
    UnwrapOrPanicForTest,
    support::{assert_finite_positive_rect, box_decoration_edges, solid_border},
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

#[test]
fn css_drop_shadow_rejects_non_zero_spread() {
    let error = FilterOp::try_drop_shadow(
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 1.0, Color::BLACK).unwrap(),
    )
    .expect_err("CSS drop-shadow must not silently treat spread like box-shadow spread");

    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter drop-shadow spread")
    );
}

#[test]
fn css_drop_shadow_rejects_inset_shadow_with_typed_diagnostic() {
    let error = FilterOp::try_drop_shadow(
        Shadow::try_inset(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )
    .expect_err("CSS drop-shadow must not execute inset shadows as outer alpha shadows");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
}

#[test]
fn css_drop_shadow_rejects_non_solid_shadow_paint() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let error = FilterOp::try_drop_shadow(
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Paint::gradient(gradient)).unwrap(),
    )
    .expect_err("CSS drop-shadow currently requires a solid shadow paint");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
}

#[test]
fn filter_region_models_preserve_blur_and_drop_shadow_bounds() {
    let blur = FilterBlur::try_new(2.0).unwrap();
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 10.0, 4.0, 4.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(8.0, 8.0, 12.0, 12.0)).unwrap();
    let blur_outset = FilterOutset::from_blur(blur, BlurPolicy::css_filter_default()).unwrap();
    let blur_region = FilterRegionPlan::try_new(source, blur_outset, Some(clip)).unwrap();
    assert_eq!(blur_region.source_bounds(), source);
    assert_eq!(
        blur_region.inflated_bounds().rect(),
        Rect::new(5.0, 5.0, 14.0, 14.0)
    );
    assert_eq!(
        blur_region.execution_region().rect(),
        Rect::new(8.0, 8.0, 11.0, 11.0)
    );

    let shadow = FilterDropShadow::try_from_shadow(
        Shadow::try_new(Point::new(2.0, -1.0), 4.0, 0.0, Color::BLACK).unwrap(),
    )
    .unwrap();
    let shadow_outset =
        FilterOutset::from_drop_shadow(&shadow, BlurPolicy::css_filter_default()).unwrap();
    assert_eq!(shadow_outset.left(), 8.0);
    assert_eq!(shadow_outset.top(), 11.0);
    assert_eq!(shadow_outset.right(), 12.0);
    assert_eq!(shadow_outset.bottom(), 9.0);
}

#[test]
fn authored_shadow_normalization_preserves_order_and_typed_boundaries() {
    let mut scene = Scene::new();
    scene.shadows(
        Rect::new(1.0, 1.0, 6.0, 6.0),
        ShadowList::try_new(vec![
            Shadow::try_new(Point::new(1.0, 0.0), 2.0, 1.0, Color::BLACK).unwrap(),
            Shadow::try_new(Point::new(-1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ])
        .unwrap(),
    );
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    assert_eq!(normalized.stats().shadows, 2);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Shadow { .. },
            command::RenderCommand::Shadow { .. }
        ]
    ));

    let mut inset_scene = Scene::new();
    inset_scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );
    let inset_error = inset_scene.normalize(Capabilities::CURRENT).unwrap_err();
    assert_eq!(
        inset_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let text_run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let text_shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut text_scene = Scene::new();
    text_scene.text_shadow_run(TextShadowRun::try_new(text_run, text_shadows).unwrap());
    let text_error = text_scene.normalize(Capabilities::CURRENT).unwrap_err();
    assert_eq!(
        text_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(
        text_error
            .message()
            .contains("glyph-alpha/offscreen text capture")
    );

    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut non_solid_scene = Scene::new();
    non_solid_scene.shadow(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Paint::gradient(gradient)).unwrap(),
    );
    let non_solid_error = non_solid_scene
        .normalize(Capabilities::CURRENT)
        .unwrap_err();
    assert_eq!(
        non_solid_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
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
fn filter_lists_classify_ordered_color_filter_pipelines() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(0.8).unwrap()),
        FilterOp::grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(0.5).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(0.4).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::saturate(FilterAmount::try_new(1.5).unwrap()),
        FilterOp::sepia(UnitFilterAmount::try_new(0.6).unwrap()),
    ])
    .unwrap();

    let pipeline = list
        .color_filter_pipeline()
        .expect("color-only filter lists should classify")
        .expect("color-only filter lists should produce a pipeline");

    assert_eq!(
        pipeline.ops(),
        &[
            ColorFilterOp::Brightness(FilterAmount::try_new(1.2).unwrap()),
            ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
            ColorFilterOp::HueRotate(FilterAngle::try_radians(0.5).unwrap()),
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.4).unwrap()),
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
            ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.6).unwrap()),
        ]
    );
}

#[test]
fn filter_none_has_no_executable_color_pipeline() {
    assert_eq!(FilterList::none().color_filter_pipeline(), Ok(None));
}

#[test]
fn drop_shadow_model_cannot_express_inset_spread_or_non_solid_paint() {
    fn assert_model_traits<T: Clone + Copy + std::fmt::Debug + PartialEq>() {}

    assert_model_traits::<FilterDropShadow>();
    let offset = Point::try_new(1.0, 2.0).unwrap();
    let blur = FilterBlur::try_new(3.0).unwrap();
    let direct = FilterDropShadow::try_new(offset, blur, Color::BLACK).unwrap();
    assert_eq!(direct.offset(), offset);
    assert_eq!(direct.blur(), blur);
    assert_eq!(direct.color(), Color::BLACK);
    assert!(FilterDropShadow::try_new(Point::new(f64::NAN, 0.0), blur, Color::BLACK).is_err());

    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let invalid_shadows = [
        Shadow::try_inset(Point::new(1.0, 2.0), 3.0, 0.0, Color::BLACK).unwrap(),
        Shadow::try_new(Point::new(1.0, 2.0), 3.0, 1.0, Color::BLACK).unwrap(),
        Shadow::try_new(Point::new(1.0, 2.0), 3.0, 0.0, Paint::gradient(gradient)).unwrap(),
    ];

    assert!(
        invalid_shadows.into_iter().all(|shadow| {
            !crate::style::filter_drop_shadow_payload_accepts_shadow_for_test(shadow)
        }),
        "broad filter drop-shadow payload remains constructible"
    );
}

#[test]
fn filter_blur_rejects_values_above_256_without_clamping() {
    let next_above_256 = f64::from_bits(256.0_f64.to_bits() + 1);

    assert_eq!(FilterBlur::try_new(256.0).unwrap().radius(), 256.0);
    assert!(
        FilterBlur::try_new(next_above_256).is_err(),
        "next representable value above 256 was accepted"
    );
}

#[test]
fn box_shadow_bounds_do_not_reuse_capped_css_filter_blur() {
    let mut scene = Scene::new();
    scene.shadow(
        Rect::new(10.0, 20.0, 30.0, 40.0),
        Shadow::try_new(Point::new(3.0, -2.0), 512.0, 4.0, Color::BLACK).unwrap(),
    );
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let bounds = command::outer_box_shadow_bounds_for_test(&normalized.commands[0]).unwrap();

    assert!(
        bounds.is_some(),
        "box-shadow bounds still depend on CSS FilterBlur validation"
    );
    assert_finite_positive_rect(bounds.unwrap());
}

#[test]
fn filter_blur_policy_zero_radius_produces_zero_outset() {
    let policy = BlurPolicy::css_filter_default();
    let outset = FilterOutset::from_blur(FilterBlur::try_new(0.0).unwrap(), policy).unwrap();

    assert_eq!(outset, FilterOutset::zero());
    assert_eq!(
        policy.radius_interpretation(),
        BlurRadiusInterpretation::CssLengthAsStandardDeviation
    );
    assert_eq!(
        policy.edge_sampling(),
        TransparentEdgeSamplingPolicy::TransparentBlack
    );
}

#[test]
fn filter_blur_region_inflates_bounds_deterministically() {
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 20.0, 30.0, 40.0)).unwrap();
    let outset = FilterOutset::from_blur(
        FilterBlur::try_new(4.0).unwrap(),
        BlurPolicy::css_filter_default(),
    )
    .unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, None).unwrap();

    assert_eq!(outset, FilterOutset::try_uniform(10.0).unwrap());
    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(0.0, 10.0, 50.0, 60.0)
    );
    assert_eq!(
        plan.execution_region().rect(),
        Rect::new(0.0, 10.0, 50.0, 60.0)
    );
}

#[test]
fn drop_shadow_outset_combines_offset_and_blur_support() {
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 10.0, 20.0, 10.0)).unwrap();
    let shadow = FilterDropShadow::try_from_shadow(
        Shadow::try_new(Point::new(3.0, -2.0), 2.0, 0.0, Color::BLACK).unwrap(),
    )
    .unwrap();
    let outset = FilterOutset::from_drop_shadow(&shadow, BlurPolicy::css_filter_default()).unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, None).unwrap();

    assert_eq!(outset, FilterOutset::try_new(2.0, 7.0, 8.0, 3.0).unwrap());
    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(8.0, 3.0, 30.0, 20.0)
    );
}

#[test]
fn filter_drop_shadow_conversion_rejects_inset_shadow_with_typed_diagnostic() {
    let shadow = Shadow::try_inset(Point::new(3.0, -2.0), 2.0, 0.0, Color::BLACK).unwrap();
    let error = FilterDropShadow::try_from_shadow(shadow)
        .expect_err("CSS drop-shadow conversion does not support inset shadows");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
}

#[test]
fn filter_region_plan_clips_inflated_bounds_to_explicit_filter_region() {
    let source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 20.0, 20.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(-5.0, -2.0, 30.0, 18.0)).unwrap();
    let outset = FilterOutset::from_blur(
        FilterBlur::try_new(4.0).unwrap(),
        BlurPolicy::css_filter_default(),
    )
    .unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, Some(clip)).unwrap();

    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(-10.0, -10.0, 40.0, 40.0)
    );
    assert_eq!(plan.clip_bounds(), Some(clip));
    assert_eq!(
        plan.execution_region().rect(),
        Rect::new(-5.0, -2.0, 30.0, 18.0)
    );
}

#[test]
fn filter_blur_policy_names_large_radius_clamp_and_rejection() {
    let clamp = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_clamp_to(8.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();
    let reject = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_reject_above(8.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();

    assert_eq!(
        clamp.large_radius_policy().action(),
        LargeBlurRadiusAction::Clamp
    );
    assert_eq!(
        FilterOutset::from_blur(FilterBlur::try_new(12.0).unwrap(), clamp).unwrap(),
        FilterOutset::try_uniform(20.0).unwrap()
    );

    let error = FilterOutset::from_blur(FilterBlur::try_new(12.0).unwrap(), reject)
        .expect_err("rejecting large blur radii should report a typed invalid value");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_region_models_reject_invalid_bounds_and_radii() {
    let zero_source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("filter source bounds must have area");
    assert_eq!(zero_source.code(), ErrorCode::InvalidInput);
    assert_eq!(
        zero_source
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter source bounds width")
    );

    let non_finite_clip = FilterClipBounds::try_new(Rect::new(f64::INFINITY, 0.0, 1.0, 1.0))
        .expect_err("unbounded sentinel filter regions should be rejected");
    assert_eq!(
        non_finite_clip
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter clip bounds x")
    );

    let negative_outset =
        FilterOutset::try_new(-1.0, 0.0, 0.0, 0.0).expect_err("outsets cannot be negative");
    assert_eq!(
        negative_outset
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter outset left")
    );

    let negative_radius =
        FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");
    assert_eq!(
        negative_radius
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter blur radius")
    );

    let source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(20.0, 20.0, 5.0, 5.0)).unwrap();
    let empty_execution = FilterRegionPlan::try_new(source, FilterOutset::zero(), Some(clip))
        .expect_err("clipping to an empty region should be rejected");
    assert_eq!(
        empty_execution
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter execution region")
    );
}

#[test]
fn color_filter_pipeline_rejects_pixel_moving_operations_with_typed_diagnostics() {
    let shadow = FilterDropShadow::try_from_shadow(
        Shadow::try_new(Point::new(1.0, 2.0), 3.0, 0.0, Color::BLACK).unwrap(),
    )
    .unwrap();
    let cases = [
        (
            "blur",
            FilterList::try_ops(vec![
                FilterOp::brightness(FilterAmount::try_new(1.0).unwrap()),
                FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
                FilterOp::contrast(FilterAmount::try_new(1.0).unwrap()),
            ])
            .unwrap(),
            PrimitiveOperation::GpuBlurFilterExecution,
            "GPU blur filter execution",
        ),
        (
            "drop shadow",
            FilterList::try_ops(vec![
                FilterOp::saturate(FilterAmount::try_new(1.0).unwrap()),
                FilterOp::drop_shadow(shadow),
                FilterOp::sepia(UnitFilterAmount::try_new(0.25).unwrap()),
            ])
            .unwrap(),
            PrimitiveOperation::GpuDropShadowFilterExecution,
            "GPU drop-shadow filter execution",
        ),
    ];

    for (case, list, operation, label) in cases {
        let unsupported = list
            .color_filter_pipeline()
            .expect_err("pixel-moving operations are not color-only filters");

        assert_eq!(
            unsupported,
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, operation),
            "{case} returned the wrong typed diagnostic",
        );
        assert_eq!(
            unsupported.label(),
            label,
            "{case} returned the wrong label"
        );
    }
}

#[test]
fn filter_lists_reject_empty_ordered_ops() {
    let error = FilterList::try_ops(Vec::new()).expect_err("empty op lists must use none");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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
    assert_eq!(error.code(), ErrorCode::InvalidInput);

    let filters = FilterList::try_ops(vec![FilterOp::contrast(
        FilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();
    let unsupported = paint
        .ensure_supported(Capabilities::CURRENT)
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
fn backdrop_filter_input_preserves_supported_filters_bounds_and_clip() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 1.0, 12.0, 8.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(1.0, 2.0, 4.0, 5.0))).unwrap();

    let input = BackdropFilterInput::try_new(filters.clone(), bounds, Some(clip.clone())).unwrap();

    assert_eq!(input.filters(), &filters);
    assert_eq!(input.capture_bounds(), bounds);
    assert_eq!(input.clip(), Some(&clip));
}

#[test]
fn backdrop_filter_input_rejects_empty_filters() {
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let error = BackdropFilterInput::try_new(FilterList::none(), bounds, None)
        .expect_err("backdrop filters must be an explicit non-empty filter list");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("backdrop filter input filters")
    );
}

#[test]
fn backdrop_capture_bounds_reject_invalid_rectangles() {
    let zero = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("backdrop capture bounds must have positive area");

    assert_eq!(zero.code(), ErrorCode::InvalidInput);
    assert_eq!(
        zero.invalid_value_diagnostic().map(InvalidValue::field),
        Some("backdrop capture bounds width")
    );

    let non_finite = BackdropCaptureBounds::try_new(Rect::new(f64::INFINITY, 0.0, 1.0, 1.0))
        .expect_err("backdrop capture bounds must be finite");
    assert_eq!(
        non_finite
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("backdrop capture bounds x")
    );
}

#[test]
fn backdrop_filter_input_rejects_unresolved_clip_references() {
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.1).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let clip = ClipInput::reference(StyleResourceRef::try_new("#backdrop-clip").unwrap());

    let error = BackdropFilterInput::try_new(filters, bounds, Some(clip))
        .expect_err("backdrop clip geometry must already be render-owned");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );
}

#[test]
fn backdrop_filter_root_policy_reports_explicit_diagnostic() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let error = BackdropFilterInput::try_root_backdrop(filters, None)
        .expect_err("root backdrop capture is not render-owned yet");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ))
    );
}

#[test]
fn backdrop_layer_normalization_plans_bounded_capture_without_broad_execution() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(1.0, 2.0, 8.0, 6.0)).unwrap();
    let backdrop = BackdropFilterInput::try_new(filters.clone(), bounds, None).unwrap();
    let layer = Layer::new().try_backdrop_filter(backdrop).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(2.0, 3.0, 4.0, 2.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[1] else {
        panic!("expected backdrop layer command");
    };

    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::BoundedBackdropCapture
    );
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::OffscreenTexture
    );
    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(bounds.rect())
    );
    let capture = layer
        .backdrop
        .as_ref()
        .unwrap_or_panic_for_test("backdrop capture is planned");
    assert_eq!(capture.filters(), &filters);
    assert_eq!(capture.capture_bounds().rect(), bounds.rect());
    assert!(matches!(
        normalized.commands[0],
        command::RenderCommand::Fill { .. }
    ));
    let offscreen = Capabilities::CURRENT.offscreen_pipeline();
    assert!(offscreen.supports_bounded_backdrop_capture());
    assert!(offscreen.supports_bounded_backdrop_filter_execution());
    assert!(!offscreen.supports_broad_backdrop_execution());
}

#[test]
fn backdrop_layer_normalization_preserves_command_order_for_capture_sources() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(2.0, 0.0, 1.0, 1.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        })
        .fill(Rect::new(4.0, 0.0, 1.0, 1.0), Color::BLACK);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[1] else {
        panic!("expected backdrop layer command");
    };
    assert!(matches!(
        normalized.commands[0],
        command::RenderCommand::Fill { .. }
    ));
    assert!(layer.backdrop.is_some());
    assert_eq!(children.len(), 1);
    assert!(matches!(
        normalized.commands[2],
        command::RenderCommand::Fill { .. }
    ));
}

#[test]
fn nested_backdrop_layer_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new(), |scene| {
        scene.layer(backdrop, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });
    });

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("nested backdrop capture is outside the normalization boundary");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    );
    assert!(error.message().contains("nested backdrop capture"));
}

#[test]
fn transformed_backdrop_layer_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let backdrop = Layer::new()
        .try_transform(Transform::translation(2.0, 0.0).unwrap())
        .unwrap()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(backdrop, |_| {});

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("transformed backdrop capture needs coordinate-space reconciliation");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    );
    assert!(error.message().contains("transformed backdrop capture"));
}

#[test]
fn repeated_top_level_backdrop_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let first_backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters.clone(), bounds, None).unwrap())
        .unwrap();
    let second_backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(first_backdrop, |_| {})
        .layer(second_backdrop, |_| {});

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("repeated top-level backdrop captures need staged source reconciliation");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    );
    assert!(
        error
            .message()
            .contains("repeated top-level backdrop capture")
    );
}

#[test]
fn backdrop_layer_normalization_carries_rounded_and_path_clip_planning() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 20.0, 20.0)).unwrap();
    let rounded_clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 2.0, 8.0, 6.0),
        Radii::all(2.0),
    ))
    .unwrap();
    let rounded_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(filters.clone(), bounds, Some(rounded_clip)).unwrap(),
        )
        .unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(3.0, 4.0))
        .line_to(Point::new(7.0, 4.0))
        .line_to(Point::new(7.0, 9.0))
        .close();
    let filled = FilledPath::try_new(path, FillRule::EvenOdd).unwrap();
    let path_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                bounds,
                Some(ClipInput::try_filled_path(filled).unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
    let mut rounded_scene = Scene::new();
    rounded_scene.layer(rounded_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    let rounded_normalized = rounded_scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer {
        layer: rounded_layer,
        ..
    } = &rounded_normalized.commands[0]
    else {
        panic!("expected rounded backdrop layer command");
    };
    let rounded_capture = rounded_layer
        .backdrop
        .as_ref()
        .unwrap_or_panic_for_test("rounded backdrop capture is planned");

    let mut path_scene = Scene::new();
    path_scene.layer(path_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    let path_normalized = path_scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer {
        layer: path_layer, ..
    } = &path_normalized.commands[0]
    else {
        panic!("expected path backdrop layer command");
    };
    let path_capture = path_layer
        .backdrop
        .as_ref()
        .unwrap_or_panic_for_test("path backdrop capture is planned");

    assert!(matches!(
        rounded_capture.clip().map(command::RenderClip::geometry),
        Some(command::RenderClipGeometry::RoundedRect { .. })
    ));
    assert!(matches!(
        path_capture.clip().map(command::RenderClip::geometry),
        Some(command::RenderClipGeometry::Path {
            fill_rule: FillRule::EvenOdd,
            ..
        })
    ));
}

#[test]
fn backdrop_isolation_and_bounded_group_diagnostics_are_explicit() {
    let unsupported_isolation = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropIsolationComposition,
    );
    let unsupported_broad = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );
    for unsupported in [unsupported_isolation, unsupported_broad] {
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("broad backdrop execution must stay diagnostic");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }

    fn backdrop_layer() -> Layer {
        let filters =
            FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
        let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
        Layer::new()
            .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
            .unwrap()
    }

    let mut nested_scene = Scene::new();
    nested_scene.layer(Layer::new(), |scene| {
        scene.layer(backdrop_layer(), |_| {});
    });
    let nested = nested_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("nested backdrop capture crosses the bounded execution path");
    assert_eq!(nested.unsupported_primitive(), Some(unsupported_broad));
    assert!(nested.message().contains("nested backdrop capture"));

    let mut repeated_scene = Scene::new();
    repeated_scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(backdrop_layer(), |_| {})
        .layer(backdrop_layer(), |_| {});
    let repeated = repeated_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("repeated top-level backdrop capture remains bounded");
    assert_eq!(repeated.unsupported_primitive(), Some(unsupported_broad));
    assert!(
        repeated
            .message()
            .contains("repeated top-level backdrop capture")
    );

    let mut transformed_scene = Scene::new();
    transformed_scene.layer(
        backdrop_layer()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap(),
        |_| {},
    );
    let transformed = transformed_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("transformed backdrop capture needs coordinate reconciliation");
    assert_eq!(transformed.unsupported_primitive(), Some(unsupported_broad));
    assert!(
        transformed
            .message()
            .contains("transformed backdrop capture")
    );
}

#[test]
fn filter_blur_rejects_negative_radius() {
    let error = FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_unit_amount_rejects_out_of_range_value() {
    let error = UnitFilterAmount::try_new(1.5)
        .expect_err("unit filter amounts must be clamped before render");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter unit amount")
    );
}

#[test]
fn filter_angle_rejects_nan() {
    let error = FilterAngle::try_radians(f64::NAN).expect_err("filter angles must be finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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
fn repeated_mask_layers_remain_distinct_in_authored_order() {
    let mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();

    let stack = MaskLayerStack::try_new([
        MaskLayer::new(mask.clone()),
        MaskLayer::new(mask.clone()),
        MaskLayer::new(mask),
    ])
    .unwrap();

    assert_eq!(stack.len(), 3);
    assert_eq!(stack.layers()[0], stack.layers()[1]);
    assert_eq!(stack.layers()[1], stack.layers()[2]);
}

#[test]
fn ordered_mask_layer_stacks_preserve_layer_and_composite_lists() {
    let first =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();
    let second =
        MaskInput::try_shape(Shape::rect(Rect::new(1.0, 0.0, 3.0, 4.0)), MaskMode::Alpha).unwrap();

    let stack = MaskLayerStack::try_new([
        MaskLayer::new(first.clone()),
        MaskLayer::try_new(second.clone(), MaskCompositeMode::Add).unwrap(),
    ])
    .unwrap();

    assert_eq!(stack.layers()[0].input(), &first);
    assert_eq!(stack.layers()[1].input(), &second);
    assert_eq!(stack.layers()[0].composite_mode(), MaskCompositeMode::Add);
    assert_eq!(stack.layers()[1].composite_mode(), MaskCompositeMode::Add);
}

#[test]
fn mask_layer_stacks_validate_empty_lists_and_single_layer_diagnostics() {
    let error = MaskLayerStack::try_new([]).expect_err("mask layer lists must not be empty");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("mask layer stack")
    );

    let stack = MaskLayerStack::single(
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap(),
    );
    let error = stack
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("single authored alpha masks still stop at source execution");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );
}

#[test]
fn mask_layer_stacks_report_specific_luminance_and_composite_diagnostics() {
    let luminance = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let luminance_error = luminance
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("luminance mask stacks need a typed unsupported diagnostic");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let composite = MaskLayerStack::single(
        MaskLayer::try_new(
            MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha)
                .unwrap(),
            MaskCompositeMode::Intersect,
        )
        .unwrap(),
    );
    let composite_error = composite
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("non-default mask composite modes are not implemented");
    assert_eq!(
        composite_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ))
    );
}

#[test]
fn multi_layer_mask_stacks_report_composition_boundary_after_input_validation() {
    let first =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();
    let second =
        MaskInput::try_shape(Shape::rect(Rect::new(1.0, 0.0, 3.0, 4.0)), MaskMode::Alpha).unwrap();
    let stack = MaskLayerStack::try_new([MaskLayer::new(first), MaskLayer::new(second)]).unwrap();

    let error = stack
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("true multi-layer mask composition is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ))
    );

    let unresolved = MaskLayerStack::try_new([
        MaskLayer::new(
            MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha)
                .unwrap(),
        ),
        MaskLayer::new(MaskInput::reference(
            StyleResourceRef::try_new("#stack-mask").unwrap(),
            MaskMode::Alpha,
        )),
    ])
    .unwrap();

    let error = unresolved
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("unresolved references remain a narrower diagnostic than composition");
    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("#stack-mask")
    );
}

#[test]
fn mask_layer_stack_model_does_not_change_unmasked_render_paths() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.fill(Rect::new(1.0, 1.0, 2.0, 2.0), Color::BLACK);
        });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(scene.stats().fills, 2);
    assert_eq!(scene.stats().layers, 1);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Fill { .. },
            command::RenderCommand::Layer { .. }
        ]
    ));
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
fn clip_inputs_diagnose_unresolved_reference_boundaries() {
    let clip = ClipInput::reference(StyleResourceRef::try_new("#content-clip").unwrap());

    let error = clip
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("clip references must be root-resolved before render execution");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    let diagnostic = error
        .unresolved_resource_diagnostic()
        .unwrap_or_panic_for_test("clip references should report an unresolved resource");
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Clip);
    assert_eq!(diagnostic.identifier(), "#content-clip");
}

#[test]
fn shape_clip_inputs_match_current_capability_contract() {
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0))).unwrap();

    clip.ensure_supported(Capabilities::CURRENT)
        .unwrap_or_panic_for_test("shape clips are supported by the current Vello layer path");
    assert!(Capabilities::CURRENT.masks_clips().supports_shape_clips());
}

#[test]
fn mask_inputs_diagnose_current_unexecuted_boundaries() {
    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)), MaskMode::Alpha).unwrap();
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255])).unwrap();
    let image_layer = StyleImageLayer::try_new(StyleImageSource::image(image).unwrap()).unwrap();
    let image_mask = MaskInput::image_layer(image_layer, MaskMode::Alpha);
    let luminance_mask = MaskInput::try_shape(
        Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)),
        MaskMode::Luminance,
    )
    .unwrap();
    let transformed_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)), MaskMode::Alpha)
            .unwrap()
            .with_coordinate_space(
                CoordinateSpaceTag::surface(Transform::translation(1.0, 0.0).unwrap()).unwrap(),
            );
    let reference_mask = MaskInput::reference(
        StyleResourceRef::try_new("#alpha-mask").unwrap(),
        MaskMode::Alpha,
    );

    let alpha_error = alpha_mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("shape masks need a real rasterization path before execution");
    assert_eq!(
        alpha_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let image_error = image_mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("image-layer masks need materialized placement before execution");
    assert_eq!(
        image_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let transformed_error = transformed_mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("transformed authored masks need materialized execution inputs");
    assert_eq!(
        transformed_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let luminance_error = luminance_mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("luminance mask mode remains unsupported");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let reference_error = reference_mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("mask references must be root-resolved before render execution");
    assert_eq!(reference_error.code(), ErrorCode::UnresolvedResource);
    let diagnostic = reference_error
        .unresolved_resource_diagnostic()
        .expect("mask references should report an unresolved resource");
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Mask);
    assert_eq!(diagnostic.identifier(), "#alpha-mask");
}

#[test]
fn unresolved_and_unmaterialized_clip_mask_inputs_return_typed_diagnostics() {
    let clip = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());
    let clip_error = clip
        .normalize(Capabilities::CURRENT)
        .expect_err("unresolved clip references remain root-owned");
    assert_eq!(clip_error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        clip_error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );

    let luminance_stack = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let luminance_error = luminance_stack
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("luminance mask conversion is unsupported");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)), MaskMode::Alpha).unwrap();
    let source_error = MaskLayerStack::single(alpha_mask.clone())
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("authored alpha mask sources still need materialization before execution");
    assert_eq!(
        source_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let multi_layer_error = MaskLayerStack::try_new([
        MaskLayer::new(alpha_mask.clone()),
        MaskLayer::new(alpha_mask.clone()),
    ])
    .unwrap()
    .ensure_supported(Capabilities::CURRENT)
    .expect_err("multi-layer mask composition has a typed boundary");
    assert_eq!(
        multi_layer_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ))
    );

    let composite_error =
        MaskLayerStack::single(MaskLayer::try_new(alpha_mask, MaskCompositeMode::Exclude).unwrap())
            .ensure_supported(Capabilities::CURRENT)
            .expect_err("non-default mask composites have a typed boundary");
    assert_eq!(
        composite_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ))
    );
}

#[test]
fn clip_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid clip paths fail");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
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

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn clip_input_normalization_lowers_concrete_shape_geometry() {
    let rect = Rect::new(1.0, 2.0, 3.0, 4.0);
    let radii = Radii::new(1.0, 2.0, 3.0, 4.0);
    let circle_center = Point::new(8.0, 9.0);
    let ellipse_center = Point::new(12.0, 13.0);
    let ellipse_radii = Size::new(4.0, 5.0);
    let cases = [
        (
            ClipInput::try_shape(Shape::rect(rect)).unwrap(),
            ClipGeometryKind::Rect(rect),
        ),
        (
            ClipInput::try_shape(Shape::try_rounded_rect(rect, radii).unwrap()).unwrap(),
            ClipGeometryKind::RoundedRect { rect, radii },
        ),
        (
            ClipInput::try_shape(Shape::try_circle(circle_center, 3.0).unwrap()).unwrap(),
            ClipGeometryKind::Circle {
                center: circle_center,
                radius: 3.0,
            },
        ),
        (
            ClipInput::try_shape(Shape::try_ellipse(ellipse_center, ellipse_radii).unwrap())
                .unwrap(),
            ClipGeometryKind::Ellipse {
                center: ellipse_center,
                radii: ellipse_radii,
            },
        ),
    ];

    for (input, expected) in cases {
        let normalized = input.normalize(Capabilities::CURRENT).unwrap();

        assert_eq!(normalized.geometry().kind(), &expected);
        assert_eq!(normalized.coordinate_space(), None);
    }
}

#[test]
fn clip_input_normalization_preserves_path_fill_rules_and_bounds() {
    let mut path = Path::new();
    path.move_to(Point::new(2.0, 3.0))
        .line_to(Point::new(6.0, 3.0))
        .line_to(Point::new(6.0, 8.0))
        .close();
    let filled = FilledPath::try_new(path.clone(), FillRule::EvenOdd).unwrap();
    let input = ClipInput::try_filled_path(filled.clone()).unwrap();

    let normalized = input.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(
        normalized.geometry().kind(),
        &ClipGeometryKind::Path(filled)
    );

    let layer = Layer::new()
        .try_clip_input(
            ClipInput::try_filled_path(FilledPath::try_new(path, FillRule::NonZero).unwrap())
                .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(-10.0, -10.0, 40.0, 40.0), Color::BLACK);
    });
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(2.0, 3.0, 4.0, 5.0))
    );
    assert!(matches!(
        layer.clip.as_ref().map(|clip| clip.geometry()),
        Some(command::RenderClipGeometry::Path {
            fill_rule: FillRule::NonZero,
            ..
        })
    ));
}

#[test]
fn clip_input_normalization_reports_reference_and_invalid_path_diagnostics() {
    let reference = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());
    let error = reference
        .normalize(Capabilities::CURRENT)
        .expect_err("unresolved clip references should stay a typed diagnostic");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("#clip")
    );

    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));
    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid path points fail");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn clip_input_normalization_preserves_coordinate_space_tags_and_rejects_nonfinite_bounds() {
    let tag = CoordinateSpaceTag::surface(Transform::translation(4.0, 5.0).unwrap()).unwrap();
    let normalized = ClipInput::try_shape(Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0)))
        .unwrap()
        .with_coordinate_space(tag)
        .normalize(Capabilities::CURRENT)
        .unwrap();

    assert_eq!(normalized.coordinate_space(), Some(tag));

    let huge = ClipInput::try_shape(Shape::rect(Rect::new(f64::MAX, 0.0, 1.0, 1.0)))
        .unwrap()
        .with_coordinate_space(
            CoordinateSpaceTag::surface(Transform::scale(2.0, 1.0).unwrap()).unwrap(),
        );
    let error = huge
        .normalize(Capabilities::CURRENT)
        .expect_err("transformed clip bounds must remain finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("clip transformed bounds")
    );
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
fn image_sampling_capabilities_name_css_sampling_boundaries() {
    let capabilities = Capabilities::CURRENT.image_sampling();

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
    assert!(!capabilities.supports_color_filtered_image_paint());
    assert!(!capabilities.supports_image_orientation_conversion());
    assert!(!capabilities.supports_image_color_profile_conversion());
}

#[test]
fn box_decoration_capability_accessors_name_supported_paint_boundaries() {
    let capabilities = Capabilities::CURRENT.box_decorations();

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
    let capabilities = Capabilities::CURRENT.box_decorations();

    assert!(!capabilities.supports_border_groove_style());
    assert!(!capabilities.supports_border_ridge_style());
    assert!(!capabilities.supports_border_inset_style());
    assert!(!capabilities.supports_border_outset_style());
    assert!(!capabilities.supports_outline_double_style());
    assert!(!capabilities.supports_outline_auto_style());
}

#[test]
fn mask_clip_capabilities_name_narrow_alpha_execution_boundaries() {
    let capabilities = Capabilities::CURRENT.masks_clips();

    assert!(capabilities.supports_shape_clips());
    assert!(!capabilities.supports_clip_reference_execution());
    assert!(!capabilities.supports_layer_masks());
    assert!(capabilities.supports_resolved_alpha_mask_execution());
    assert!(
        Capabilities::CURRENT
            .ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::ResolvedAlphaMaskExecution,
            ))
            .is_ok(),
        "resolved alpha-mask execution is supported by the GPU graph boundary"
    );
    assert!(!capabilities.supports_luminance_mask_mode());
    assert!(!capabilities.supports_multi_layer_mask_composition());
    assert!(!capabilities.supports_mask_composite_modes());
}

#[test]
fn current_capabilities_report_supported_gpu_filters_and_diagnostic_only_effects() {
    let capabilities = Capabilities::CURRENT;
    let filters = capabilities.filters();
    assert_eq!(
        [
            (
                "supports_gpu_color_filter_execution",
                filters.supports_gpu_color_filter_execution(),
            ),
            (
                "supports_gpu_blur_filter_execution",
                filters.supports_gpu_blur_filter_execution(),
            ),
            (
                "supports_gpu_drop_shadow_filter_execution",
                filters.supports_gpu_drop_shadow_filter_execution(),
            ),
        ],
        [
            ("supports_gpu_color_filter_execution", true),
            ("supports_gpu_blur_filter_execution", true),
            ("supports_gpu_drop_shadow_filter_execution", true),
        ],
        "the delivered GPU filter routes must publish their final semantic truths"
    );

    assert!(!filters.supports_layer_filters());
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_layer_filter_execution()
    );
    assert!(!capabilities.offscreen_pipeline().supports_mask_execution());
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_broad_backdrop_execution()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_backdrop_isolation_composition()
    );
    assert!(
        !capabilities
            .image_sampling()
            .supports_color_filtered_image_paint(),
        "materialized color-filtered image paint must remain diagnostic"
    );
}

#[test]
fn color_filter_capability_names_granular_execution_without_broad_effects() {
    let capabilities = Capabilities::CURRENT;

    assert!(capabilities.filters().supports_ordered_filter_lists());
    assert!(capabilities.filters().supports_gpu_color_filter_execution());
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(
        !capabilities
            .image_sampling()
            .supports_color_filtered_image_paint()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_layer_filter_execution()
    );
}

#[test]
fn pixel_moving_filter_capability_names_advertise_materialized_execution_only() {
    let capabilities = Capabilities::CURRENT;

    assert!(capabilities.filters().supports_ordered_filter_lists());
    assert!(capabilities.filters().supports_gpu_blur_filter_execution());
    assert!(
        capabilities
            .filters()
            .supports_gpu_drop_shadow_filter_execution()
    );
    assert!(capabilities.filters().supports_filter_region_planning());
    assert!(!capabilities.shadows().supports_inset_box_shadows());
    assert!(!capabilities.shadows().supports_text_shadows());
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_layer_filter_execution()
    );
}

#[test]
fn pixel_moving_filter_and_shadow_diagnostics_have_granular_names() {
    let supported_cases = [
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
            "GPU blur filter execution",
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
            "GPU drop-shadow filter execution",
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::FilterRegionPlanning,
            "filter-region planning",
        ),
    ];
    let unsupported_cases = [
        (
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
            "inset box shadow",
        ),
        (
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
            "text shadow",
        ),
    ];

    for (family, operation, label) in supported_cases {
        let supported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(supported.label(), label);
        assert!(
            Capabilities::CURRENT.ensure_supported(supported).is_ok(),
            "delivered GPU filter behavior and region planning must stay supported"
        );
    }

    for (family, operation, label) in unsupported_cases {
        let unsupported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(unsupported.label(), label);

        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported shadow operations stay named without execution");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(label));
    }
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
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this image sampling primitive");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(unsupported.label()));
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
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this box-decoration style");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("box decorations"));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn backdrop_and_advanced_compositing_diagnostics_have_granular_names() {
    let cases = [
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
            "root backdrop policy",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::BackgroundBlendMode,
            "background blend mode",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::AdditionalMixBlendMode,
            "additional mix-blend mode",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::PorterDuffCompositeMode,
            "Porter-Duff composite mode",
        ),
    ];

    for (family, operation, label) in cases {
        let unsupported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(unsupported.label(), label);

        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("advanced compositing operations remain typed boundaries");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("compositing"));
        assert!(error.message().contains(label));
    }
}

#[test]
fn mask_clip_capability_diagnostics_report_unsupported_operations() {
    for operation in [
        PrimitiveOperation::ClipReferenceExecution,
        PrimitiveOperation::LayerMask,
        PrimitiveOperation::AlphaMaskSourceExecution,
        PrimitiveOperation::LuminanceMaskMode,
        PrimitiveOperation::MultiLayerMaskComposition,
        PrimitiveOperation::MaskCompositeMode,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::MasksAndClips, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported mask and clip operations remain typed boundaries");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("masks and clips"));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn layer_mask_filter_parent_diagnostics_win_over_unsupported_children() {
    let cases = [
        (
            Layer::new()
                .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
                .unwrap(),
            UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ),
        ),
        (
            Layer::new()
                .try_filter(Filter::try_blur(4.0).unwrap())
                .unwrap(),
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        ),
    ];

    for (layer, primitive) in cases {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(8.0, 0.0));
        let mut scene = Scene::new();
        scene.layer(layer, |scene| {
            scene.stroke(
                Shape::path(path),
                Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
                Color::BLACK,
            );
        });

        let error = scene
            .normalize(Capabilities::CURRENT)
            .expect_err("parent layer diagnostic should be reported before child geometry");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(primitive));
    }
}

#[test]
fn unsupported_porter_duff_css_and_mask_composite_policy_stays_typed() {
    let compositing = Capabilities::CURRENT.compositing();
    assert!(!compositing.supports_background_blend_modes());
    assert!(!compositing.supports_additional_mix_blend_modes());
    assert!(!compositing.supports_porter_duff_composite_modes());

    for operation in [
        PrimitiveOperation::BackgroundBlendMode,
        PrimitiveOperation::AdditionalMixBlendMode,
        PrimitiveOperation::PorterDuffCompositeMode,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::Compositing, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported CSS and Porter-Duff composite policy stays typed");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("compositing"));
        assert!(error.message().contains(unsupported.label()));
    }

    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)), MaskMode::Alpha).unwrap();
    for mode in [
        MaskCompositeMode::Subtract,
        MaskCompositeMode::Intersect,
        MaskCompositeMode::Exclude,
    ] {
        let stack = MaskLayerStack::single(MaskLayer::try_new(alpha_mask.clone(), mode).unwrap());
        let error = stack
            .ensure_supported(Capabilities::CURRENT)
            .expect_err("non-default mask composites remain unsupported until fully implemented");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MaskCompositeMode,
            ))
        );
    }
}

#[test]
fn outer_box_shadow_list_normalizes_offset_blur_spread_and_order() {
    let first = Shadow::try_new(Point::new(3.0, -2.0), 6.0, 1.5, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(-4.0, 5.0), 0.0, -1.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let mut scene = Scene::new();

    scene.shadows(Rect::new(8.0, 8.0, 10.0, 10.0), shadows);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    assert_eq!(normalized.commands.len(), 2);
    assert_eq!(normalized.stats().shadows, 2);

    let command::RenderCommand::Shadow { shadow, .. } = &normalized.commands[0] else {
        panic!("first shadow-list entry should lower to a render shadow");
    };
    assert_eq!(shadow.offset, first.offset());
    assert_eq!(shadow.blur, first.blur());
    assert_eq!(shadow.spread, first.spread());

    let command::RenderCommand::Shadow { shadow, .. } = &normalized.commands[1] else {
        panic!("second shadow-list entry should lower to a render shadow");
    };
    assert_eq!(shadow.offset, second.offset());
    assert_eq!(shadow.blur, second.blur());
    assert_eq!(shadow.spread, second.spread());
}

#[test]
fn non_uniform_rounded_outer_shadow_preserves_authored_radii() {
    let radii = Radii::new(0.0, 4.0, 8.0, 12.0);
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_rounded_rect(Rect::new(4.0, 4.0, 16.0, 12.0), radii).unwrap(),
        Shadow::try_new(Point::new(2.0, 2.0), 4.0, 1.0, Color::BLACK).unwrap(),
    );

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Shadow { shape, .. } = &normalized.commands[0] else {
        panic!("rounded rect shadow should lower to a render shadow");
    };
    let command::ShadowShape::RoundedRect {
        radii: lowered_radii,
        ..
    } = shape
    else {
        panic!("rounded rect shadow should preserve rounded geometry");
    };
    assert_eq!(*lowered_radii, radii);
}

#[test]
fn inset_box_shadow_reports_typed_unsupported_diagnostic() {
    let mut scene = Scene::new();
    scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = scene
        .normalize(Capabilities::CURRENT)
        .expect_err("inset shadow execution is not implemented in this phase");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
    assert!(error.message().contains("inset box shadow"));
}
