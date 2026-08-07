use crate::{
    Antialiasing, BackdropCaptureBounds, BackdropFilterInput, Capabilities, ClipInput, Color,
    ErrorCode, FilterAmount, FilterBlur, FilterDropShadow, FilterList, FilterOp, Image, Layer,
    Path, PhysicalSize, Point, PrimitiveFamily, PrimitiveOperation, Rect, ResolvedLayerAlphaMask,
    Scene, Shape, Size, Stroke, TextGlyph, TextPaint, TextRun, TextRunBounds, Transform,
    UnitFilterAmount, UnresolvedResourceKind, UnsupportedPrimitive, error::Result,
    style::ColorFilterOp,
};

use crate::frame::{
    BackdropDependencyObservation, FramePlanResultObservation, FramePlanRouteObservation,
    FrameSelectionRequirementObservation, OrderedFilterEdgeObservation,
    OrderedFilterIntentObservation, OrderedFilterPlanObservation, OrderedFilterStepObservation,
    VelloCommandObservation,
};

use super::{
    UnwrapOrPanicForTest,
    support::{
        AHEM_GLYPH_X, add_planning_text, ahem_font, bounded_planning_backdrop, opaque_planning_mask,
    },
};

#[derive(Debug)]
struct SpatialPrimitiveObservation {
    logical_and_device_phases_are_distinct: bool,
    logical_bounds: Option<[f64; 4]>,
    device_origin: Option<(i32, i32)>,
    device_extent: Option<(u32, u32)>,
    raster_scale: f64,
    texel_center: Option<(f64, f64)>,
    is_empty: bool,
}

fn observe_spatial_primitives(
    rect: Rect,
    transform: Transform,
    surface_scale: f64,
    texel: (u32, u32),
) -> Result<SpatialPrimitiveObservation> {
    let observed =
        crate::frame::spatial_primitives_for_test(rect, transform, surface_scale, texel)?;
    Ok(SpatialPrimitiveObservation {
        logical_and_device_phases_are_distinct: observed.logical_and_device_phases_are_distinct,
        logical_bounds: observed.logical_bounds,
        device_origin: observed.device_origin,
        device_extent: observed.device_extent,
        raster_scale: observed.raster_scale,
        texel_center: observed.texel_center,
        is_empty: observed.is_empty,
    })
}

#[test]
fn signed_device_bounds_floor_minima_and_ceil_maxima() {
    let rect = Rect::new(-1.25, 2.125, 3.5, 4.25);
    let observed = observe_spatial_primitives(rect, Transform::identity(), 2.0, (0, 0)).unwrap();

    assert!(
        observed.logical_and_device_phases_are_distinct,
        "logical and device spatial phases remain collapsed"
    );
    assert_eq!(observed.logical_bounds, Some([-1.25, 2.125, 3.5, 4.25]));
    assert_eq!(observed.device_origin, Some((-3, 4)));
    assert_eq!(observed.device_extent, Some((8, 9)));

    let largest_extent = observe_spatial_primitives(
        Rect::new(f64::from(i32::MIN), 0.0, f64::from(u32::MAX), 1.0),
        Transform::identity(),
        1.0,
        (0, 0),
    )
    .unwrap();
    assert_eq!(largest_extent.device_origin, Some((i32::MIN, 0)));
    assert_eq!(largest_extent.device_extent, Some((u32::MAX, 1)));

    for (rect, scale) in [
        (Rect::new(f64::NAN, 0.0, 1.0, 1.0), 1.0),
        (Rect::new(f64::MAX, 0.0, f64::MAX, 1.0), 1.0),
        (Rect::new(f64::from(i32::MAX), 0.0, 1.0, 1.0), 2.0),
        (
            Rect::new(f64::from(i32::MIN), 0.0, f64::from(u32::MAX) + 1.0, 1.0),
            1.0,
        ),
    ] {
        let error = observe_spatial_primitives(rect, Transform::identity(), scale, (0, 0))
            .expect_err("overflowing spatial values must be rejected");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert!(error.invalid_value_diagnostic().is_some());
    }
}

#[test]
fn negative_and_fractional_origins_preserve_texel_center_mapping() {
    let observed = observe_spatial_primitives(
        Rect::new(-1.25, -0.75, 2.0, 1.5),
        Transform::identity(),
        2.0,
        (2, 3),
    )
    .unwrap();

    assert_eq!(
        observed.texel_center,
        Some((-0.25, 0.75)),
        "texel-center mapping is absent"
    );
    assert_eq!(observed.device_origin, Some((-3, -2)));
    assert_eq!(observed.device_extent, Some((5, 4)));
}

#[test]
fn bounded_capture_transform_preserves_signed_origin_texel_centers_and_scale() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.25, 0.5, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("bounded capture transform"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::try_ink(Rect::new(-1.25, -0.75, 2.0, 1.5)).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);
    let commands = scene.normalize(Capabilities::CURRENT).unwrap();
    let observed = crate::pass::bounded_capture_transform_observation_for_test(
        commands,
        Transform::translation(0.375, -0.625).unwrap(),
        Transform::translation(-2.0, 1.25).unwrap(),
        Antialiasing::Msaa16,
    );

    assert!(
        observed.preserves_application_order_formula
            && observed.preserves_signed_texel_center_mapping
            && observed.covers_required_raster_scales
            && observed.preserves_capture_execution_facts
            && observed.lowers_scene_with_explicit_initial_transform,
        "bounded capture transform changed the signed texel-center mapping"
    );
}

#[test]
fn largest_singular_value_raster_scale_preserves_local_effect_space() {
    let transform = Transform::try_new([2.0, 1.0, 1.0, 3.0, 4.0, -2.0]).unwrap();
    let observed =
        observe_spatial_primitives(Rect::new(-1.0, 2.0, 4.0, 3.0), transform, 1.25, (0, 0))
            .unwrap();
    let expected = ((5.0_f64 + 5.0_f64.sqrt()) * 0.5) * 1.25;

    assert!(
        (observed.raster_scale - expected).abs() <= f64::EPSILON * expected,
        "local raster scale does not use the largest singular value"
    );
    assert_eq!(observed.device_origin, Some((-5, 9)));
    assert_eq!(observed.device_extent, Some((19, 14)));

    let error = observe_spatial_primitives(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        Transform::identity(),
        f64::INFINITY,
        (0, 0),
    )
    .expect_err("non-finite surface scale must be rejected");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.invalid_value_diagnostic().is_some());

    let huge_transform = Transform::try_new([f64::MAX, 0.0, 0.0, f64::MAX, 0.0, 0.0]).unwrap();
    let error =
        observe_spatial_primitives(Rect::new(0.0, 0.0, 1.0, 1.0), huge_transform, 2.0, (0, 0))
            .expect_err("overflowing local raster scale must be rejected");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.invalid_value_diagnostic().is_some());
}

#[test]
fn zero_singular_value_produces_an_empty_plan() {
    let zero_transform = Transform::scale(0.0, 0.0).unwrap();
    let observed =
        observe_spatial_primitives(Rect::new(-2.0, 3.0, 4.0, 5.0), zero_transform, 2.0, (0, 0))
            .unwrap();

    assert!(
        observed.is_empty,
        "degenerate spatial output was erased instead of represented as empty"
    );
    assert_eq!(observed.device_origin, None);
    assert_eq!(observed.device_extent, None);

    let degenerate_bounds = observe_spatial_primitives(
        Rect::new(1.0, 2.0, 0.0, 3.0),
        Transform::identity(),
        2.0,
        (0, 0),
    )
    .unwrap();
    assert!(degenerate_bounds.is_empty);
    assert_eq!(degenerate_bounds.device_origin, None);
    assert_eq!(degenerate_bounds.device_extent, None);
}

#[test]
fn rank_deficient_transform_produces_explicit_empty_spatial_plan() {
    let rank_deficient_transform = Transform::scale(0.0, 1.0).unwrap();
    let observed = observe_spatial_primitives(
        Rect::new(-2.0, 3.0, 4.0, 5.0),
        rank_deficient_transform,
        2.0,
        (0, 0),
    )
    .unwrap();

    assert!(
        observed.is_empty,
        "rank-deficient output was not represented as explicit Empty"
    );
    assert_eq!(observed.device_origin, None);
    assert_eq!(observed.device_extent, None);
}

#[test]
fn logical_bounds_preserve_large_finite_translation_until_frame_scale_resolution() {
    let transformed = crate::frame::transformed_logical_bounds_for_test(
        Rect::new(0.0, 0.0, 4.0, 2.0),
        Transform::translation(3_000_000_000.0, 0.0).unwrap(),
    );

    assert!(
        transformed.is_ok(),
        "finite logical transform was rejected before frame scale resolution: {transformed:?}"
    );
    assert_eq!(transformed.unwrap(), [3_000_000_000.0, 0.0, 4.0, 2.0]);

    let resolved = observe_spatial_primitives(
        Rect::new(3_000_000_000.0, 0.0, 4.0, 2.0),
        Transform::identity(),
        0.5,
        (0, 0),
    )
    .unwrap();
    assert_eq!(resolved.device_origin, Some((1_500_000_000, 0)));
    assert_eq!(resolved.device_extent, Some((2, 1)));
}

fn observe_ordered_filter_plan(
    filters: &FilterList,
    source_rect: Rect,
    transform: Transform,
    surface_scale: f64,
    backdrop: bool,
) -> Result<OrderedFilterPlanObservation> {
    crate::frame::ordered_filter_plan_for_test(
        filters,
        source_rect,
        transform,
        surface_scale,
        backdrop,
    )
}

#[test]
fn filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order() {
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.25).unwrap()),
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::blur(FilterBlur::try_new(0.0).unwrap()),
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-3.25, 4.5),
                FilterBlur::try_new(0.5).unwrap(),
                Color::BLACK,
            )
            .unwrap(),
        ),
        FilterOp::sepia(UnitFilterAmount::try_new(0.25).unwrap()),
    ])
    .unwrap();
    let observed = observe_ordered_filter_plan(
        &filters,
        Rect::new(10.25, -4.5, 20.0, 10.0),
        Transform::identity(),
        2.0,
        false,
    )
    .unwrap();

    assert!(
        observed
            .steps
            .iter()
            .all(|step| step.result_bounds.is_some()),
        "legacy filter classifiers do not produce ordered result-bound records"
    );
    assert_eq!(observed.authored_operation_count, 5);
    assert!(!observed.is_empty);
    assert!(observed.has_spatial_mapping);
    assert_eq!(observed.initial_bounds, [10.25, -4.5, 20.0, 10.0]);
    assert_eq!(observed.final_bounds, [3.0, -7.0, 29.75, 21.0]);
    assert_eq!(observed.steps.len(), 4, "zero blur must be elided");
    assert_ordered_filter_plan_steps(&observed);
    assert_ordered_filter_edge_cases(&filters);
}

fn assert_ordered_filter_plan_steps(observed: &OrderedFilterPlanObservation) {
    assert_eq!(
        observed.steps[0],
        OrderedFilterStepObservation {
            source_bounds: [10.25, -4.5, 20.0, 10.0],
            result_bounds: Some([10.25, -4.5, 20.0, 10.0]),
            source_device_origin: Some((20, -9)),
            source_device_extent: Some((41, 20)),
            result_device_origin: Some((20, -9)),
            result_device_extent: Some((41, 20)),
            edge: OrderedFilterEdgeObservation::NoSampling,
            intent: OrderedFilterIntentObservation::ColorRun {
                operations: vec![ColorFilterOp::Brightness(
                    FilterAmount::try_new(1.25).unwrap(),
                )],
                clamp_boundaries_after_operation: vec![0],
            },
        }
    );
    assert_eq!(
        observed.steps[1],
        OrderedFilterStepObservation {
            source_bounds: [10.25, -4.5, 20.0, 10.0],
            result_bounds: Some([7.75, -7.0, 25.0, 15.0]),
            source_device_origin: Some((20, -9)),
            source_device_extent: Some((41, 20)),
            result_device_origin: Some((15, -14)),
            result_device_extent: Some((51, 30)),
            edge: OrderedFilterEdgeObservation::TransparentBlack,
            intent: OrderedFilterIntentObservation::Blur {
                standard_deviation: 1.0,
                inclusive_support_taps: 5,
            },
        }
    );
    assert_eq!(
        observed.steps[2],
        OrderedFilterStepObservation {
            source_bounds: [7.75, -7.0, 25.0, 15.0],
            result_bounds: Some([3.0, -7.0, 29.75, 21.0]),
            source_device_origin: Some((15, -14)),
            source_device_extent: Some((51, 30)),
            result_device_origin: Some((6, -14)),
            result_device_extent: Some((60, 42)),
            edge: OrderedFilterEdgeObservation::TransparentBlack,
            intent: OrderedFilterIntentObservation::DropShadow {
                offset: (-3.25, 4.5),
                standard_deviation: 0.5,
                inclusive_support_taps: 3,
                uses_source_alpha: true,
                retains_unchanged_source: true,
                continuous_offset: true,
            },
        }
    );
    assert_eq!(
        observed.steps[3],
        OrderedFilterStepObservation {
            source_bounds: [3.0, -7.0, 29.75, 21.0],
            result_bounds: Some([3.0, -7.0, 29.75, 21.0]),
            source_device_origin: Some((6, -14)),
            source_device_extent: Some((60, 42)),
            result_device_origin: Some((6, -14)),
            result_device_extent: Some((60, 42)),
            edge: OrderedFilterEdgeObservation::NoSampling,
            intent: OrderedFilterIntentObservation::ColorRun {
                operations: vec![ColorFilterOp::Sepia(
                    UnitFilterAmount::try_new(0.25).unwrap(),
                )],
                clamp_boundaries_after_operation: vec![0],
            },
        }
    );
}

fn assert_ordered_filter_edge_cases(filters: &FilterList) {
    let backdrop = observe_ordered_filter_plan(
        &FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap(),
        Rect::new(0.0, 0.0, 4.0, 3.0),
        Transform::identity(),
        2.0,
        true,
    )
    .unwrap();
    assert_eq!(
        backdrop.steps[0].edge,
        OrderedFilterEdgeObservation::SemanticBorderMirror([0.0, 0.0, 4.0, 3.0])
    );
    assert_eq!(backdrop.final_bounds, [-2.5, -2.5, 9.0, 8.0]);

    for transform in [Transform::identity(), Transform::scale(0.0, 1.0).unwrap()] {
        let source = if transform == Transform::identity() {
            Rect::new(1.0, 2.0, 0.0, 3.0)
        } else {
            Rect::new(1.0, 2.0, 4.0, 3.0)
        };
        let empty = observe_ordered_filter_plan(filters, source, transform, 2.0, false).unwrap();
        assert!(empty.is_empty);
        assert!(!empty.has_spatial_mapping);
        assert!(empty.steps.is_empty());
    }

    let support_error = observe_ordered_filter_plan(
        &FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(256.0).unwrap())]).unwrap(),
        Rect::new(0.0, 0.0, 1.0e-12, 1.0e-12),
        Transform::identity(),
        f64::from(u32::MAX),
        false,
    )
    .expect_err("unrepresentable raster-aware support must remain a typed failure");
    assert_eq!(support_error.code(), ErrorCode::InvalidInput);
    assert!(support_error.invalid_value_diagnostic().is_some());
}

#[test]
fn color_filter_fusion_preserves_each_source_clamp() {
    let operations = vec![
        ColorFilterOp::Brightness(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(2.0).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ];
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.0).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(2.0).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let observed = observe_ordered_filter_plan(
        &filters,
        Rect::new(-1.0, 2.0, 2.0, 3.0),
        Transform::identity(),
        1.5,
        false,
    )
    .unwrap();

    let OrderedFilterIntentObservation::ColorRun {
        operations: observed_operations,
        clamp_boundaries_after_operation,
    } = &observed.steps[0].intent
    else {
        panic!("adjacent authored color operations must share one semantic pass intent");
    };
    assert_eq!(
        clamp_boundaries_after_operation,
        &[0, 1, 2, 3],
        "fused intent lost authored clamp boundaries"
    );
    assert_eq!(observed.steps.len(), 1);
    assert_eq!(observed_operations, &operations);
    assert_eq!(observed.steps[0].source_bounds, [-1.0, 2.0, 2.0, 3.0]);
    assert_eq!(observed.steps[0].result_bounds, Some([-1.0, 2.0, 2.0, 3.0]));
    assert_eq!(
        observed.steps[0].edge,
        OrderedFilterEdgeObservation::NoSampling
    );
}

fn zero_sized_transparent_mask(bounds: Rect) -> ResolvedLayerAlphaMask {
    ResolvedLayerAlphaMask::try_new(
        Image::from_rgba(Size::new(0.0, 0.0), Vec::<u8>::new()).unwrap(),
        bounds,
    )
    .unwrap()
}

fn transparent_planning_mask(size: PhysicalSize) -> ResolvedLayerAlphaMask {
    zero_sized_transparent_mask(Rect::new(
        0.0,
        0.0,
        f64::from(size.width()),
        f64::from(size.height()),
    ))
}

pub(super) fn observe_frame_plan(
    scene: &Scene,
    surface_size: Size,
    surface_scale: f64,
    antialiasing: Antialiasing,
    base_color: Color,
) -> FramePlanResultObservation {
    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test(
            "the planning fixture must normalize before resolved-frame planning",
        );
    crate::frame::frame_plan_result_observation_for_test(
        normalized,
        surface_size,
        surface_scale,
        antialiasing,
        base_color,
    )
}

#[test]
fn direct_vello_is_the_least_powerful_plan_for_effect_free_scenes() {
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 3.0), Color::BLACK);
    add_planning_text(&mut scene, TextRunBounds::unspecified());

    let result = observe_frame_plan(
        &scene,
        Size::new(12.0, 8.0),
        2.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );
    let plan = result
        .plan
        .as_ref()
        .unwrap_or_panic_for_test("the observation must be complete");

    assert_eq!(
        plan.route,
        FramePlanRouteObservation::DirectVello,
        "effect-free scene has no direct frame plan"
    );
    assert_eq!(plan.plan_count, 1);
    assert!(plan.complete && plan.finite && plan.backend_free);
    assert_eq!(plan.direct_command_count, 2);
    assert_eq!(plan.output_device_extent, Some((24, 16)));
    assert_eq!(plan.antialiasing, Some(Antialiasing::Msaa8));
    assert_eq!(
        plan.base_color,
        Some(Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap())
    );
    assert!(plan.selection_requirements.is_empty());
    assert!(result.error_code.is_none());
}

#[test]
fn transparent_resolved_alpha_mask_annihilates_unspecified_text_without_graph_selection() {
    let transparent_mask = zero_sized_transparent_mask(Rect::new(0.0, 0.0, 2.0, 2.0));
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new().with_resolved_alpha_mask(transparent_mask),
            |scene| add_planning_text(scene, TextRunBounds::unspecified()),
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.unresolved_resource.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.direct_commands == [VelloCommandObservation::Fill]
                    && plan.selection_requirements.is_empty()
                    && plan.resource_count == 0
                    && plan.pass_count == 0
            }),
        "transparent resolved mask retained graph selection or unresolved text bounds"
    );
}

#[test]
fn transparent_mask_under_nonempty_clip_prunes_mixed_unresolved_source() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(4.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_clip(Shape::rect(Rect::new(0.0, 0.0, 2.0, 1.0)))
                .unwrap()
                .with_resolved_alpha_mask(transparent_planning_mask(PhysicalSize::new(2, 1))),
            |scene| {
                scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
                add_planning_text(scene, TextRunBounds::unspecified());
            },
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(6.0, 2.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.unresolved_resource.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.direct_commands == [VelloCommandObservation::Fill]
                    && plan.selection_requirements.is_empty()
                    && plan.resource_count == 0
                    && plan.pass_count == 0
            }),
        "transparent 2x1 mask retained unclipped mixed sizing or unresolved graph text instead of preserving the sibling DirectVello plan"
    );
}

#[test]
fn exact_empty_outer_clip_skips_mask_size_validation_and_preserves_sibling() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(4.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_clip(Shape::rect(Rect::new(0.0, 0.0, 0.0, 1.0)))
                .unwrap()
                .with_resolved_alpha_mask(transparent_planning_mask(PhysicalSize::new(7, 3))),
            |scene| {
                scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
            },
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(6.0, 2.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.unresolved_resource.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.direct_commands == [VelloCommandObservation::Fill]
                    && plan.selection_requirements.is_empty()
                    && plan.resource_count == 0
                    && plan.pass_count == 0
            }),
        "exact-empty outer clip retained resolved-mask size validation or failed to preserve the sibling DirectVello plan"
    );
}

#[test]
fn transparent_resolved_mask_image_extent_is_independent_from_local_bounds() {
    let transparent_mask = zero_sized_transparent_mask(Rect::new(0.0, 0.0, 4.0, 4.0));
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(transparent_mask),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
        },
    );
    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test("the mismatched transparent-mask fixture must normalize");
    let context = crate::frame::FrameContext::try_new(
        Size::new(8.0, 8.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    )
    .unwrap_or_panic_for_test("the mismatched transparent-mask frame context must resolve");
    assert!(
        normalized.plan_for(context).is_ok(),
        "mask image extent was incorrectly coupled to layer-local bounds"
    );
}

#[test]
fn transparent_resolved_mask_prunes_mixed_source_independent_of_image_extent() {
    let transparent_mask = zero_sized_transparent_mask(Rect::new(0.0, 0.0, 4.0, 4.0));
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(transparent_mask),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
            add_planning_text(scene, TextRunBounds::unspecified());
        },
    );
    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test("the mixed-source transparent-mask fixture must normalize");
    let context = crate::frame::FrameContext::try_new(
        Size::new(8.0, 8.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    )
    .unwrap_or_panic_for_test("the mixed-source transparent-mask frame context must resolve");
    assert!(
        normalized.plan_for(context).is_ok(),
        "transparent mask image extent prevented semantic source pruning"
    );
}

#[test]
fn bounded_blur_backdrop_over_empty_parent_is_pruned_without_erasing_foreground() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 4.0, 4.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    });

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.direct_commands == [VelloCommandObservation::LocalLayer]
                    && plan.selection_requirements.is_empty()
                    && plan.current_parent_backdrop_reads == 0
                    && plan.resource_count == 0
                    && plan.pass_count == 0
            }),
        "bounded blur backdrop over an exact empty parent retained a graph boundary"
    );
}

#[test]
fn root_output_domain_prunes_off_surface_backdrop_dependency_and_retains_partial_overlap() {
    let observe_source_and_capture = |source: Rect, capture: Rect| {
        let filters = FilterList::try_ops(vec![FilterOp::invert(
            UnitFilterAmount::try_new(1.0).unwrap(),
        )])
        .unwrap();
        let bounds = BackdropCaptureBounds::try_new(capture).unwrap();
        let backdrop = Layer::new()
            .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
            .unwrap();
        let mut scene = Scene::new();
        scene
            .fill(Rect::new(1.0, 1.0, 1.0, 1.0), Color::BLACK)
            .fill(source, Color::BLACK)
            .layer(backdrop, |_| {});
        observe_frame_plan(
            &scene,
            Size::new(8.0, 6.0),
            1.0,
            Antialiasing::Area,
            Color::TRANSPARENT,
        )
    };

    let off_surface =
        observe_source_and_capture(Rect::new(8.0, 0.0, 2.0, 2.0), Rect::new(8.0, 0.0, 2.0, 2.0));
    let partial_overlap =
        observe_source_and_capture(Rect::new(7.0, 0.0, 2.0, 2.0), Rect::new(7.0, 0.0, 2.0, 2.0));
    let off_surface = off_surface.plan.as_ref().unwrap_or_panic_for_test(
        "the off-surface backdrop fixture must produce one complete plan",
    );
    let partial_overlap = partial_overlap.plan.as_ref().unwrap_or_panic_for_test(
        "the partial-overlap backdrop fixture must produce one complete plan",
    );

    assert!(
        off_surface.route == FramePlanRouteObservation::DirectVello
            && off_surface.direct_commands == [VelloCommandObservation::Fill]
            && off_surface.selection_requirements.is_empty()
            && off_surface.current_parent_backdrop_reads == 0
            && off_surface.resource_count == 0
            && off_surface.pass_count == 0
            && partial_overlap.route == FramePlanRouteObservation::GpuGraph
            && partial_overlap.selection_requirements
                == [FrameSelectionRequirementObservation::BoundedBackdrop]
            && partial_overlap.backdrop_dependency
                == BackdropDependencyObservation::CompletedCurrentParent
            && partial_overlap.current_parent_backdrop_reads == 1,
        "root output domain retained off-surface backdrop graph work or removed the partially visible control: off_surface={off_surface:?}, partial_overlap={partial_overlap:?}"
    );
}

#[test]
fn post_filter_backdrop_clip_retains_expanded_halo_outside_capture() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 4.0, 4.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(4.5, 1.0, 1.0, 1.0))).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, Some(clip)).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(3.0, 1.0, 1.0, 1.0), Color::BLACK)
        .layer(layer, |_| {});

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 8.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::GpuGraph
                    && plan.selection_requirements
                        == [FrameSelectionRequirementObservation::BoundedBackdrop]
                    && plan.backdrop_dependency
                        == BackdropDependencyObservation::CompletedCurrentParent
                    && plan.current_parent_backdrop_reads == 1
                    && plan.captures_precede_outer_semantics
            }),
        "post-filter backdrop halo was pruned before its outside-capture clip"
    );
}

#[test]
fn zero_opacity_backdrop_preserves_foreground_without_graph_boundary() {
    let filters = FilterList::try_ops(vec![
        FilterOp::opacity(UnitFilterAmount::try_new(0.0).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(1.0, -1.0),
                FilterBlur::try_new(0.5).unwrap(),
                Color::BLACK,
            )
            .unwrap(),
        ),
    ])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 6.0, 4.0)).unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.stroke(
                Shape::rect(Rect::new(2.0, 0.0, 2.0, 2.0)),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result.plan.as_ref().unwrap_or_panic_for_test(
        "the zero-opacity backdrop fixture must produce a complete frame plan",
    );

    assert!(
        plan.route == FramePlanRouteObservation::DirectVello
            && plan.direct_commands
                == [
                    VelloCommandObservation::Fill,
                    VelloCommandObservation::LocalLayer,
                ]
            && plan.selection_requirements.is_empty()
            && plan.vello_spans.is_empty()
            && plan.resource_count == 0
            && plan.pass_count == 0,
        "zero-opacity backdrop retained a graph boundary or erased its nonempty Vello foreground"
    );
}

#[test]
fn zero_opacity_backdrop_preserves_resolved_device_range_failure() {
    let plan_with_opacity = |opacity| {
        let filters = FilterList::try_ops(vec![FilterOp::opacity(
            UnitFilterAmount::try_new(opacity).unwrap(),
        )])
        .unwrap();
        let bounds =
            BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 3_000_000_000.0, 1.0)).unwrap();
        let backdrop = Layer::new()
            .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
            .unwrap();
        let mut scene = Scene::new();
        scene
            .fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK)
            .layer(backdrop, |_| {});
        let normalized = scene
            .normalize(Capabilities::CURRENT)
            .unwrap_or_panic_for_test("the huge-backdrop fixture must normalize");
        let context = crate::frame::FrameContext::try_new(
            Size::new(8.0, 8.0),
            1.0,
            Antialiasing::Area,
            Color::TRANSPARENT,
        )
        .unwrap_or_panic_for_test("the huge-backdrop frame context must resolve");
        normalized.plan_for(context)
    };
    let transparent = plan_with_opacity(0.0);
    let opaque = plan_with_opacity(1.0);
    let observe = |result: &Result<_>| {
        result.as_ref().err().map(|error| {
            let diagnostic = error.invalid_value_diagnostic();
            (
                error.code(),
                diagnostic.map(|value| value.field().to_owned()),
                diagnostic.map(|value| value.value().to_owned()),
                diagnostic.map(|value| value.invariant().to_owned()),
            )
        })
    };
    let expected = Some((
        ErrorCode::InvalidInput,
        Some("filter device bounds max x".to_owned()),
        Some("3000000000".to_owned()),
        Some("must fit in i32 device pixels".to_owned()),
    ));

    assert_eq!(
        (observe(&transparent), observe(&opaque)),
        (expected.clone(), expected),
        "transparent backdrop pruning bypassed the resolved device-range failure"
    );
}

#[test]
fn empty_masked_subtree_does_not_select_graph_or_split_vello_span() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(1, 1))),
            |scene| add_planning_text(scene, TextRunBounds::empty()),
        )
        .stroke(
            Shape::rect(Rect::new(3.0, 0.0, 2.0, 2.0)),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result.plan.as_ref().unwrap_or_panic_for_test(
        "the empty-source planning fixture must produce one complete plan",
    );

    assert!(
        plan.route == FramePlanRouteObservation::DirectVello
            && plan.selection_requirements.is_empty()
            && plan.resource_count == 0
            && plan.pass_count == 0
            && plan.vello_spans.is_empty()
            && plan.direct_commands
                == [
                    VelloCommandObservation::Fill,
                    VelloCommandObservation::Stroke,
                ],
        "empty masked subtree selected graph or split Vello span"
    );
}

#[test]
fn zero_area_masked_source_does_not_select_graph_or_split_vello_span() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(1, 1))),
            |scene| {
                scene.fill(Rect::new(0.0, 3.0, 0.0, 2.0), Color::BLACK);
            },
        )
        .stroke(
            Shape::rect(Rect::new(3.0, 0.0, 2.0, 2.0)),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result
        .plan
        .as_ref()
        .unwrap_or_panic_for_test("the zero-area-source fixture must produce one complete plan");

    assert!(
        plan.route == FramePlanRouteObservation::DirectVello
            && plan.selection_requirements.is_empty()
            && plan.resource_count == 0
            && plan.pass_count == 0
            && plan.vello_spans.is_empty()
            && plan.direct_commands
                == [
                    VelloCommandObservation::Fill,
                    VelloCommandObservation::Stroke,
                ],
        "zero-area masked source selected graph or split Vello span"
    );
}

#[test]
fn rank_deficient_masked_source_does_not_select_graph_or_split_vello_span() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_transform(Transform::scale(0.0, 1.0).unwrap())
                .unwrap()
                .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(1, 1))),
            |scene| {
                scene.fill(Rect::new(0.0, 3.0, 2.0, 2.0), Color::BLACK);
            },
        )
        .stroke(
            Shape::rect(Rect::new(3.0, 0.0, 2.0, 2.0)),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result.plan.as_ref().unwrap_or_panic_for_test(
        "the rank-deficient-source fixture must produce one complete plan",
    );

    assert!(
        plan.route == FramePlanRouteObservation::DirectVello
            && plan.selection_requirements.is_empty()
            && plan.resource_count == 0
            && plan.pass_count == 0
            && plan.vello_spans.is_empty()
            && plan.direct_commands
                == [
                    VelloCommandObservation::Fill,
                    VelloCommandObservation::Stroke,
                ],
        "rank-deficient masked source selected graph or split Vello span"
    );
}

#[test]
fn empty_stroked_path_mask_source_does_not_select_graph_or_split_vello_span() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
            |scene| {
                scene.stroke(
                    Shape::path(Path::new()),
                    Stroke::try_new(1.0).unwrap(),
                    Color::BLACK,
                );
            },
        )
        .stroke(
            Shape::rect(Rect::new(3.0, 0.0, 2.0, 2.0)),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.unresolved_resource.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.selection_requirements.is_empty()
                    && plan.resource_count == 0
                    && plan.pass_count == 0
                    && plan.vello_spans.is_empty()
                    && plan.direct_commands
                        == [
                            VelloCommandObservation::Fill,
                            VelloCommandObservation::Stroke,
                        ]
            }),
        "empty stroked path mask source selected graph or split Vello span"
    );
}

#[test]
fn empty_clip_short_circuits_unresolved_masked_text_bounds() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(
            Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(1, 1))),
            |scene| {
                scene.layer(
                    Layer::new()
                        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 0.0, 4.0)))
                        .unwrap(),
                    |scene| add_planning_text(scene, TextRunBounds::unspecified()),
                );
            },
        )
        .stroke(
            Shape::rect(Rect::new(3.0, 0.0, 2.0, 2.0)),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.unresolved_resource.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::DirectVello
                    && plan.selection_requirements.is_empty()
                    && plan.resource_count == 0
                    && plan.pass_count == 0
                    && plan.vello_spans.is_empty()
                    && plan.direct_commands
                        == [
                            VelloCommandObservation::Fill,
                            VelloCommandObservation::Stroke,
                        ]
            }),
        "empty clip did not short-circuit unresolved masked text bounds"
    );
}

#[test]
fn gpu_graph_is_selected_only_for_supported_custom_requirements() {
    let mask = opaque_planning_mask(PhysicalSize::new(4, 4));
    let mut scene = Scene::new();
    scene.layer(Layer::new().with_resolved_alpha_mask(mask), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    });

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 8.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result
        .plan
        .as_ref()
        .unwrap_or_panic_for_test("the observation must be complete");

    assert_eq!(
        plan.route,
        FramePlanRouteObservation::GpuGraph,
        "custom requirement has no semantic graph plan"
    );
    assert_eq!(
        plan.selection_requirements,
        vec![FrameSelectionRequirementObservation::ResolvedAlphaMask]
    );
    assert!(plan.resource_count > 0 && plan.pass_count > 0);
    assert_eq!(plan.plan_count, 1);

    let unsupported_layer = Layer::new()
        .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
        .unwrap();
    let mut unsupported = Scene::new();
    unsupported.layer(unsupported_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });
    let error = unsupported
        .normalize(Capabilities::CURRENT)
        .expect_err("unsupported authored masks must retain their typed diagnostic");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
}

#[test]
fn supported_scenes_produce_one_finite_backend_free_frame_plan() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 8.0, 6.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            scene.layer(
                Layer::new()
                    .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(8, 6))),
                |scene| {
                    scene.fill(Rect::new(1.0, 1.0, 4.0, 3.0), Color::BLACK);
                },
            );
        });
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let observe = |commands| {
        crate::frame::frame_plan_result_observation_for_test(
            commands,
            Size::new(8.0, 6.0),
            2.0,
            Antialiasing::Msaa16,
            Color::try_rgba(0.1, 0.2, 0.3, 1.0).unwrap(),
        )
    };
    let first = observe(normalized.clone());
    let plan = first.plan.as_ref();

    assert!(
        plan.is_some_and(|plan| plan.plan_count == 1 && plan.complete && plan.finite),
        "supported scene has no finite frame plan"
    );
    let second = observe(normalized);
    assert_eq!(first, second, "repeated planning must be deterministic");
    let plan = first.plan.as_ref().unwrap();
    assert_eq!(plan.route, FramePlanRouteObservation::GpuGraph);
    assert!(plan.backend_free);
    assert!(plan.resource_count > 0 && plan.pass_count > 0);
    assert!(!plan.graph_to_vello_reentry);
    assert!(plan.captures_precede_outer_semantics);
    assert_eq!(
        plan.selection_requirements,
        vec![
            FrameSelectionRequirementObservation::BoundedBackdrop,
            FrameSelectionRequirementObservation::ResolvedAlphaMask,
        ]
    );

    let mut failing = Scene::new();
    failing
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            add_planning_text(scene, TextRunBounds::unspecified());
        });
    let failure = observe_frame_plan(
        &failing,
        Size::new(8.0, 6.0),
        2.0,
        Antialiasing::Msaa16,
        Color::TRANSPARENT,
    );
    assert!(failure.plan.is_none());
    assert!(!failure.has_partial_plan);
    assert_eq!(
        failure.unresolved_resource,
        Some(UnresolvedResourceKind::TextRunInkBounds)
    );
}
