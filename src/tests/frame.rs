use crate::{
    Antialiasing, BackdropCaptureBounds, BackdropFilterInput, BlendMode, Capabilities, ClipInput,
    Color, CoordinateSpaceTag, EffectQualityPolicy, ErrorCode, FillRule, FilledPath, FilterAmount,
    FilterAngle, FilterBlur, FilterDropShadow, FilterList, FilterOp, Format, Image, ImageFit,
    Layer, Options, Path, PhysicalSize, Point, PrimitiveFamily, PrimitiveOperation, Radii, Rect,
    Renderer, ResolvedLayerAlphaMask, ResourceCacheBudget, Scene, Shadow, Shape, Size, Stroke,
    TextGlyph, TextPaint, TextRun, TextRunBounds, Transform, UnitFilterAmount,
    UnresolvedResourceKind, UnsupportedPrimitive, command, error::Result, style::ColorFilterOp,
};

use crate::{
    backend::DeviceCapabilities,
    resource::ResourceManager,
    shader::DevicePassCache,
    vello_engine::scene::{
        VelloDrawObservationForTest, VelloFillRuleObservationForTest,
        VelloPathDrawObservationForTest, VelloPathElementObservationForTest,
    },
};

use crate::frame::{
    BackdropDependencyObservation, FramePlanResultObservation, FramePlanRouteObservation,
    FrameSelectionRequirementObservation, GraphFailureObservation, GraphOwnerCallObservation,
    InvalidSemanticGraphStateForTest, OrderedFilterEdgeObservation, OrderedFilterIntentObservation,
    OrderedFilterPlanObservation, OrderedFilterStepObservation, VelloCommandObservation,
    VelloSpanObservation, VelloSpanScopeObservation,
};

use super::{
    UnwrapOrPanicForTest,
    support::{
        AHEM_GLYPH_X, add_planning_text, ahem_font, authored_color_filter_runs_for_test,
        bounded_backdrop_graph_commands_for_test, bounded_planning_backdrop,
        color_then_blur_filters_for_test, composition_commands_for_test,
        filter_graph_commands_for_test, filter_graph_context_for_test,
        graph_shader_commands_for_test, graph_shader_frame_context_for_test, opaque_planning_mask,
        runtime_lowering_commands_for_test, spatial_filter_authored_filter_steps_for_test,
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

#[test]
fn filter_scalar_lowering_handles_f32_f64_exponents_and_huge_angles_finitely() {
    let mantissa_renormalization_boundary = f64::from_bits(1.0_f64.to_bits() - (1_u64 << 28));
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.0).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(f64::from_bits(1)).unwrap()),
        FilterOp::grayscale(UnitFilterAmount::try_new(0.1).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(f64::MAX).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(0.25).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::saturate(FilterAmount::try_new(f64::MAX).unwrap()),
        FilterOp::sepia(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::brightness(FilterAmount::try_new(f64::from(f32::MAX)).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(mantissa_renormalization_boundary).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(-f64::MAX).unwrap()),
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
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 8.0, 6.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(1.0, 1.0, 2.0, 2.0),
                Color::try_rgba(0.5, 0.5, 0.5, 1.0).unwrap(),
            );
        });
    let commands = scene.normalize(Capabilities::CURRENT).unwrap();
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap();
    let observed = crate::pass::runtime_color_filter_observation_for_test(
        commands,
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .expect("the authored color list must reach runtime lowering");

    assert_runtime_color_filter_lowering(&observed);
}

fn assert_runtime_color_filter_lowering(
    observed: &crate::pass::RuntimeColorFilterObservationForTest,
) {
    use crate::pass::{
        RuntimeColorOperationTagForTest as Tag, RuntimeColorScalarObservationForTest as Scalar,
        RuntimeFilterAmountObservationForTest as Amount,
    };

    assert!(
        observed.operations.len() == 11
            && observed
                .operations
                .iter()
                .all(|operation| operation.scalar.is_finite_normalized())
            && observed
                .operations
                .iter()
                .all(|operation| operation.clamps_straight_rgba_then_premultiplies),
        "runtime color scalars are not finite normalized reference color-matrix values"
    );

    assert_eq!(
        observed
            .operations
            .iter()
            .map(|operation| operation.tag)
            .collect::<Vec<_>>(),
        vec![
            Tag::Brightness,
            Tag::Contrast,
            Tag::Grayscale,
            Tag::HueRotate,
            Tag::Invert,
            Tag::Opacity,
            Tag::Saturate,
            Tag::Sepia,
            Tag::Brightness,
            Tag::Contrast,
            Tag::HueRotate,
        ],
        "runtime lowering changed authored operation tags or order"
    );
    assert_eq!(
        observed.operations[0].scalar,
        Scalar::Amount(Amount {
            zero: true,
            mantissa: 0.0,
            exponent: 0,
        })
    );
    assert_eq!(
        observed.operations[1].scalar,
        Scalar::Amount(Amount {
            zero: false,
            mantissa: 0.5,
            exponent: -1073,
        })
    );
    assert!(matches!(
        observed.operations[2].scalar,
        Scalar::Unit(value) if value.to_bits() == (0.1_f64 as f32).to_bits()
    ));
    assert_eq!(
        observed.operations[6].scalar,
        Scalar::Amount(Amount {
            zero: false,
            mantissa: 0.5,
            exponent: 1025,
        })
    );
    assert_eq!(
        observed.operations[8].scalar,
        Scalar::Amount(Amount {
            zero: false,
            mantissa: f32::from_bits(0x3f7f_ffff),
            exponent: 128,
        })
    );
    assert_eq!(
        observed.operations[9].scalar,
        Scalar::Amount(Amount {
            zero: false,
            mantissa: 0.5,
            exponent: 1,
        })
    );

    for (index, angle) in [(3, f64::MAX), (10, -f64::MAX)] {
        let reduced = angle.rem_euclid(std::f64::consts::TAU) as f32;
        let (expected_sine, expected_cosine) = reduced.sin_cos();
        assert!(matches!(
            observed.operations[index].scalar,
            Scalar::Angle { sine, cosine }
                if sine.to_bits() == expected_sine.to_bits()
                    && cosine.to_bits() == expected_cosine.to_bits()
        ));
    }
}

#[test]
fn gpu_graph_executor_accepts_only_spine_composition_and_ordered_color_filters() {
    let observed = crate::pass::color_filter_executable_graph_observation_for_test(
        authored_color_filter_runs_for_test(),
        color_then_blur_filters_for_test(),
        color_then_drop_shadow_filters_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.accepts_spine_composition_and_color_for_all_formats
            && observed.accepts_multiple_ordered_color_runs
            && observed.rejects_empty_missing_and_malformed_color_facts
            && observed.rejects_copy_blur_shadow_and_drop_shadow_composite
            && observed.rejects_unsupported_output
            && observed.preserves_public_composition_dispatch_boundary,
        "the ordered color-filter executor has no closed pre-allocation subset"
    );
}

#[test]
fn color_filter_graph_preserves_authored_order_clamps_and_exact_lifetimes() {
    use crate::pass::RuntimeColorOperationTagForTest as Tag;

    let observed = crate::pass::color_filter_graph_observation_for_test(
        authored_color_filter_runs_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );
    let expected_operations = vec![
        vec![Tag::Brightness, Tag::Contrast, Tag::Invert],
        vec![Tag::HueRotate, Tag::Opacity, Tag::Sepia],
    ];
    let expected_spatial = crate::pass::ColorFilterSpatialObservationForTest {
        logical_bounds: [-2.25, 1.5, 4.0, 3.0],
        device_origin: (-3, 1),
        device_extent: PhysicalSize::new(6, 5),
        texel_origin: Point::new(-2.4, 0.8),
        raster_scale: 1.25,
    };

    assert!(
        observed.operation_tags_by_run == expected_operations
            && observed.first_source_spatial == Some(expected_spatial)
            && observed.every_run_has_one_source_and_distinct_result
            && observed.every_run_preserves_exact_spatial_descriptor
            && observed.every_operation_retains_one_clamp
            && observed.current_resource_advances_after_each_run
            && observed.dependencies_and_last_use_are_exact
            && observed.closed_color_facts_match_runtime_passes,
        "the color-filter graph changed operation order, clamp, or last use"
    );
}

#[test]
fn gpu_graph_executor_accepts_only_color_blur_and_drop_shadow_filter_graphs() {
    let observed = crate::pass::spatial_filter_executable_graph_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.accepts_color_blur_and_drop_shadow_for_all_formats
            && observed.preserves_ordered_nonzero_filter_steps
            && observed.rejects_empty_missing_and_malformed_spatial_facts
            && observed.rejects_wrong_axes_inputs_edges_and_aliases
            && observed.rejects_copy_backdrop_stale_forward_and_backdrop_plus
            && observed.rejects_before_resource_acquisition,
        "the spatial-filter executor has no closed pre-allocation color, blur, and drop-shadow subset"
    );
}

#[test]
fn blur_and_drop_shadow_graph_preserves_order_edges_and_lifetimes() {
    use crate::pass::SpatialFilterPassTagForTest as Tag;

    let observed = crate::pass::spatial_filter_graph_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.pass_order
            == [
                Tag::Color,
                Tag::BlurHorizontalRgba,
                Tag::BlurVerticalRgba,
                Tag::BlurHorizontalSourceAlpha,
                Tag::BlurVerticalSourceAlpha,
                Tag::DropShadowColorize,
                Tag::DropShadowMerge,
                Tag::Color,
            ]
            && observed.ordinary_blur_uses_transparent_black
            && observed.drop_shadow_uses_source_alpha_and_continuous_offset
            && observed.spatial_mappings_are_exact
            && observed.sources_and_results_are_distinct
            && observed.source_alpha_fanout_reads_original_twice
            && observed.original_source_releases_only_after_merge
            && observed.dependencies_and_last_use_are_exact,
        "the spatial-filter graph lost authored order, edge, spatial, or lifetime facts"
    );
}

#[test]
fn gpu_graph_executor_accepts_only_bounded_top_level_backdrop_graphs() {
    let observed = crate::pass::backdrop_executable_graph_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        crate::frame::FrameContext::try_new(
            Size::new(16.0, 12.0),
            1.0,
            Antialiasing::Msaa8,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        )
        .unwrap(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.accepts_bounded_top_level_backdrop
            && observed.rejects_outside_bounded_subset
            && observed.rejects_before_resource_acquisition,
        "production dispatch still rejects every CopyBackdrop"
    );
}

#[test]
fn backdrop_graph_reads_completed_parent_once_and_preserves_group_order() {
    let observed = crate::pass::backdrop_graph_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        crate::frame::FrameContext::try_new(
            Size::new(16.0, 12.0),
            1.0,
            Antialiasing::Msaa8,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        )
        .unwrap(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.closed_subset_receipt
            && observed.reads_completed_parent_once
            && observed.copy_precedes_authored_filters
            && observed.post_filter_clip_precedes_foreground
            && observed.foreground_precedes_outer_composition
            && observed.later_sibling_depends_on_completed_group,
        "the bounded backdrop graph lost its closed classification or dependency receipt: {observed:?}"
    );
}

fn bounded_offscreen_pass_plan_for_graph_probe() -> command::LayerPassPlan {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(-2.0, 3.0, 8.0, 6.0)).unwrap();
    let backdrop = BackdropFilterInput::try_new(filters, bounds, None).unwrap();
    let layer = Layer::new().try_backdrop_filter(backdrop).unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected one normalized offscreen layer");
    };
    layer.pass_plan
}

#[test]
fn graph_builder_rejects_forward_stale_and_read_write_aliases() {
    let pass_plan = bounded_offscreen_pass_plan_for_graph_probe();
    let edge_observation =
        crate::frame::semantic_graph_edge_lifetime_observation_for_test(pass_plan)
            .expect("the semantic graph probe must construct its stable spatial primitives");
    assert!(edge_observation.observes_bounded_offscreen_pass);

    let expected = [
        (
            InvalidSemanticGraphStateForTest::StaleResourceIdentity,
            GraphFailureObservation::WrongResourceGeneration,
        ),
        (
            InvalidSemanticGraphStateForTest::StalePassIdentity,
            GraphFailureObservation::WrongPassGeneration,
        ),
        (
            InvalidSemanticGraphStateForTest::UnknownResourceIdentity,
            GraphFailureObservation::UnknownResource,
        ),
        (
            InvalidSemanticGraphStateForTest::UnknownPassIdentity,
            GraphFailureObservation::UnknownPass,
        ),
        (
            InvalidSemanticGraphStateForTest::ReleasedResourceIdentity,
            GraphFailureObservation::ReleasedResource,
        ),
        (
            InvalidSemanticGraphStateForTest::ForwardDependency,
            GraphFailureObservation::ForwardDependency,
        ),
        (
            InvalidSemanticGraphStateForTest::ForwardRead,
            GraphFailureObservation::ForwardRead,
        ),
        (
            InvalidSemanticGraphStateForTest::ReadWriteAlias,
            GraphFailureObservation::ReadWriteAlias,
        ),
        (
            InvalidSemanticGraphStateForTest::DuplicateProducer,
            GraphFailureObservation::DuplicateProducer,
        ),
        (
            InvalidSemanticGraphStateForTest::DeclaredReadCountMismatch,
            GraphFailureObservation::DeclaredReadCountMismatch,
        ),
        (
            InvalidSemanticGraphStateForTest::OrphanResult,
            GraphFailureObservation::OrphanResult,
        ),
        (
            InvalidSemanticGraphStateForTest::MissingRootWorkingImage,
            GraphFailureObservation::MissingRootWorkingImage,
        ),
        (
            InvalidSemanticGraphStateForTest::DuplicateRootWorkingImage,
            GraphFailureObservation::DuplicateRootWorkingImage,
        ),
        (
            InvalidSemanticGraphStateForTest::MissingFinalPresent,
            GraphFailureObservation::MissingFinalPresent,
        ),
        (
            InvalidSemanticGraphStateForTest::DuplicateFinalPresent,
            GraphFailureObservation::DuplicateFinalPresent,
        ),
        (
            InvalidSemanticGraphStateForTest::NonTransparentCaptureBase,
            GraphFailureObservation::NonTransparentCaptureBase,
        ),
        (
            InvalidSemanticGraphStateForTest::RepeatedSurfaceBaseInitialization,
            GraphFailureObservation::RepeatedSurfaceBaseInitialization,
        ),
        (
            InvalidSemanticGraphStateForTest::MissingProducerDependency,
            GraphFailureObservation::MissingProducerDependency,
        ),
        (
            InvalidSemanticGraphStateForTest::ScheduleBeforeConsumersAreSealed,
            GraphFailureObservation::ConsumersNotSealed,
        ),
        (
            InvalidSemanticGraphStateForTest::DeclareConsumerAfterConsumersAreSealed,
            GraphFailureObservation::ConsumersAlreadySealed,
        ),
    ];
    let observed = expected.map(|(state, _)| {
        (
            state,
            crate::frame::invalid_semantic_graph_state_for_test(state)
                .expect("each stable invalid state must produce one typed graph failure"),
        )
    });

    assert_eq!(
        observed, expected,
        "no closed graph validator rejected the invalid edge sequence"
    );
    assert!(edge_observation.every_result_has_one_owner);
    assert!(edge_observation.every_read_names_its_producer);
}

#[test]
fn graph_builder_rejects_declaration_after_final_present() {
    let observed = crate::frame::final_present_declaration_observation_for_test()
        .expect("the terminal declaration probe must reach the graph owner");

    assert_eq!(
        (
            observed.declaration_after_present,
            observed.completed_after_declaration_attempt,
        ),
        (
            GraphOwnerCallObservation::Rejected(
                GraphFailureObservation::DeclarationAfterFinalPresent,
            ),
            true,
        ),
        "graph declaration after final present was accepted"
    );
}

#[test]
fn graph_builder_rejects_scheduling_after_final_present() {
    let observed = crate::frame::final_present_scheduling_observation_for_test()
        .expect("the terminal scheduling probe must reach the graph owner");

    assert_eq!(
        (
            observed.early_present,
            observed.completed_after_early_present_attempt,
            observed.scheduling_after_present,
            observed.completed_after_post_present_attempt,
        ),
        (
            GraphOwnerCallObservation::Rejected(
                GraphFailureObservation::PresentScheduledBeforeOtherPasses,
            ),
            true,
            GraphOwnerCallObservation::Rejected(
                GraphFailureObservation::SchedulingAfterFinalPresent,
            ),
            true,
        ),
        "graph scheduling after final present was accepted"
    );
}

#[test]
fn drop_shadow_source_fanout_lives_through_both_consumers() {
    let observed = crate::frame::semantic_graph_edge_lifetime_observation_for_test(
        bounded_offscreen_pass_plan_for_graph_probe(),
    )
    .expect("the drop-shadow lifetime graph must validate");
    assert!(observed.observes_bounded_offscreen_pass);
    assert!(
        observed.source_expected_reads == 2
            && observed.remaining_before_first_consumer == 2
            && observed.remaining_after_alpha_consumer == 1
            && observed.remaining_before_source_over == 1
            && observed.remaining_after_source_over == 0
            && observed.released_after_source_over
            && observed.post_release_read_rejected,
        "drop-shadow source has no two-consumer lifetime"
    );
    assert!(observed.every_result_has_one_owner);
    assert!(observed.every_read_names_its_producer);
}

fn observe_runtime_lowering_for_test() -> crate::pass::RuntimeLoweringObservationForTest {
    crate::pass::runtime_lowering_observation_for_test(
        runtime_lowering_commands_for_test(),
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        Format::Rgba8,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .expect("the complete semantic graph must reach runtime lowering")
}

#[test]
fn base_graph_executor_accepts_only_clear_capture_canonicalize_source_over_and_present() {
    let mut base_graph_scene = Scene::new();
    base_graph_scene
        .fill(Rect::new(-1.25, -0.75, 2.0, 1.5), Color::BLACK)
        .stroke(
            Shape::rect(Rect::new(2.0, 1.0, 3.0, 2.0)),
            Stroke::try_new(0.5).unwrap(),
            Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
        );
    let base_graph_commands = base_graph_scene.normalize(Capabilities::CURRENT).unwrap();
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap();
    let observed = crate::pass::base_graph_executable_subset_observation_for_test(
        base_graph_commands,
        runtime_lowering_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.accepts_exact_rgba_and_bgra
            && observed.rejects_every_other_pass_kind_and_composite_payload
            && observed.rejects_missing_or_reordered_spine_passes
            && observed.rejects_malformed_dependencies_reads_results_and_releases
            && observed.rejects_graph_outside_base_subset
            && observed.preserves_direct_and_graph_planner_routes,
        "the base graph executor has no closed pre-allocation executable subset"
    );
}

fn composition_ordered_nonzero_clip_path_for_test() -> Path {
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(7.0, 1.0))
        .line_to(Point::new(7.0, 7.0))
        .line_to(Point::new(1.0, 7.0))
        .close()
        .move_to(Point::new(3.0, 3.0))
        .line_to(Point::new(5.0, 3.0))
        .line_to(Point::new(5.0, 5.0))
        .line_to(Point::new(3.0, 5.0))
        .close();
    path
}

fn composition_signed_even_odd_clip_path_for_test() -> Path {
    let mut path = Path::new();
    path.move_to(Point::new(-2.0, -1.0))
        .line_to(Point::new(1.0, -1.0))
        .line_to(Point::new(1.0, 1.0))
        .line_to(Point::new(-2.0, 1.0))
        .close()
        .move_to(Point::new(-1.25, -0.5))
        .line_to(Point::new(0.25, -0.5))
        .line_to(Point::new(0.25, 0.5))
        .line_to(Point::new(-1.25, 0.5))
        .close();
    path
}

fn emitted_vello_fill_geometry_for_test(
    shape: &impl kurbo::Shape,
) -> Vec<VelloPathElementObservationForTest> {
    let mut geometry = shape
        .path_elements(0.1)
        .map(|element| match element {
            kurbo::PathEl::MoveTo(point) => {
                VelloPathElementObservationForTest::MoveTo([point.x as f32, point.y as f32])
            }
            kurbo::PathEl::LineTo(point) => {
                VelloPathElementObservationForTest::LineTo([point.x as f32, point.y as f32])
            }
            kurbo::PathEl::QuadTo(control, point) => VelloPathElementObservationForTest::QuadTo(
                [control.x as f32, control.y as f32],
                [point.x as f32, point.y as f32],
            ),
            kurbo::PathEl::CurveTo(first, second, point) => {
                VelloPathElementObservationForTest::CubicTo(
                    [first.x as f32, first.y as f32],
                    [second.x as f32, second.y as f32],
                    [point.x as f32, point.y as f32],
                )
            }
            kurbo::PathEl::ClosePath => VelloPathElementObservationForTest::Close,
        })
        .collect::<Vec<_>>();
    if !matches!(
        geometry.last(),
        Some(VelloPathElementObservationForTest::Close)
    ) {
        geometry.push(VelloPathElementObservationForTest::Close);
    }
    geometry
}

fn emitted_vello_transform_for_test(transform: Transform) -> [f32; 6] {
    transform.as_array().map(|component| component as f32)
}

fn emitted_vello_clip_is_exact_for_test(
    observed: &VelloPathDrawObservationForTest,
    expected_geometry: &[VelloPathElementObservationForTest],
    expected_transform: [f32; 6],
    expected_fill_rule: VelloFillRuleObservationForTest,
) -> bool {
    observed.geometry == expected_geometry
        && observed.transform == expected_transform
        && observed.fill_rule == expected_fill_rule
        && matches!(
            observed.draw,
            VelloDrawObservationForTest::BeginClip { blend_mode, alpha }
                if blend_mode == vello_encoding::DrawBeginClip::CLIP_BLEND_MODE
                    && alpha == 1.0
        )
}

fn emitted_vello_coverage_fill_is_exact_for_test(
    observed: &VelloPathDrawObservationForTest,
    target_extent: PhysicalSize,
) -> bool {
    let expected_geometry = emitted_vello_fill_geometry_for_test(&kurbo::Rect::new(
        0.0,
        0.0,
        f64::from(target_extent.width()),
        f64::from(target_extent.height()),
    ));
    observed.geometry == expected_geometry
        && observed.transform == emitted_vello_transform_for_test(Transform::identity())
        && observed.fill_rule == VelloFillRuleObservationForTest::NonZero
        && matches!(
            observed.draw,
            VelloDrawObservationForTest::SolidColor { rgba } if rgba == u32::MAX
        )
}

fn emitted_vello_end_clip_is_exact_for_test(observed: &VelloPathDrawObservationForTest) -> bool {
    observed.geometry.is_empty() && observed.draw == VelloDrawObservationForTest::EndClip
}

fn composition_ordered_clip_coverage_commands_for_test() -> command::RenderCommands {
    let transforms = [
        Transform::translation(0.25, 0.5).unwrap(),
        Transform::translation(0.5, 0.25).unwrap(),
        Transform::translation(0.125, 0.25).unwrap(),
        Transform::translation(0.25, 0.125).unwrap(),
        Transform::translation(0.125, 0.125).unwrap(),
    ];
    let path = composition_ordered_nonzero_clip_path_for_test();
    let path_clip =
        ClipInput::try_filled_path(FilledPath::try_new(path, FillRule::NonZero).unwrap()).unwrap();

    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 8.0, 8.0)))
            .unwrap()
            .try_transform(transforms[0])
            .unwrap(),
        |scene| {
            scene.layer(
                Layer::new()
                    .try_clip(
                        Shape::try_rounded_rect(
                            Rect::new(0.25, 0.25, 7.5, 7.5),
                            Radii::new(0.5, 0.75, 1.0, 1.25),
                        )
                        .unwrap(),
                    )
                    .unwrap()
                    .try_transform(transforms[1])
                    .unwrap(),
                |scene| {
                    scene.layer(
                        Layer::new()
                            .try_clip(Shape::try_circle(Point::new(4.0, 4.0), 3.5).unwrap())
                            .unwrap()
                            .try_transform(transforms[2])
                            .unwrap(),
                        |scene| {
                            scene.layer(
                                Layer::new()
                                    .try_clip(
                                        Shape::try_ellipse(
                                            Point::new(4.0, 4.0),
                                            Size::new(3.25, 3.0),
                                        )
                                        .unwrap(),
                                    )
                                    .unwrap()
                                    .try_transform(transforms[3])
                                    .unwrap(),
                                |scene| {
                                    scene.layer(
                                        Layer::new()
                                            .try_clip_input(path_clip)
                                            .unwrap()
                                            .try_transform(transforms[4])
                                            .unwrap()
                                            .with_resolved_alpha_mask(opaque_planning_mask(
                                                PhysicalSize::new(8, 8),
                                            )),
                                        |scene| {
                                            scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
                                        },
                                    );
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

fn composition_signed_path_clip_coverage_commands_for_test()
-> (command::RenderCommands, CoordinateSpaceTag) {
    let path = composition_signed_even_odd_clip_path_for_test();
    let coordinate_space =
        CoordinateSpaceTag::surface(Transform::translation(0.25, -0.25).unwrap()).unwrap();
    let clip = ClipInput::try_filled_path(FilledPath::try_new(path, FillRule::EvenOdd).unwrap())
        .unwrap()
        .with_coordinate_space(coordinate_space);

    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.layer(
            Layer::new()
                .try_clip_input(clip)
                .unwrap()
                .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(5, 3))),
            |scene| {
                scene.fill(Rect::new(-3.0, -2.0, 6.0, 4.0), Color::BLACK);
            },
        );
    });
    (
        scene.normalize(Capabilities::CURRENT).unwrap(),
        coordinate_space,
    )
}

#[test]
fn graph_clip_coverage_is_one_vello_capture_of_ordered_render_clips() {
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap();
    let observed = crate::pass::graph_clip_coverage_observation_for_test(
        composition_ordered_clip_coverage_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );
    assert_ordered_graph_clip_coverage(&observed);
}

fn assert_ordered_graph_clip_coverage(observed: &crate::pass::GraphClipCoverageObservationForTest) {
    let authored_transforms = [
        Transform::translation(0.25, 0.5).unwrap(),
        Transform::translation(0.5, 0.25).unwrap(),
        Transform::translation(0.125, 0.25).unwrap(),
        Transform::translation(0.25, 0.125).unwrap(),
        Transform::translation(0.125, 0.125).unwrap(),
    ];
    let mut accumulated = Transform::identity();
    let expected_transforms = authored_transforms.map(|transform| {
        accumulated = transform.then(accumulated).unwrap();
        accumulated
    });
    let expected_geometries = [
        emitted_vello_fill_geometry_for_test(&kurbo::Rect::new(0.0, 0.0, 8.0, 8.0)),
        emitted_vello_fill_geometry_for_test(&command::kurbo_rounded_rect(
            Rect::new(0.25, 0.25, 7.5, 7.5),
            Radii::new(0.5, 0.75, 1.0, 1.25),
        )),
        emitted_vello_fill_geometry_for_test(&kurbo::Circle::new((4.0, 4.0), 3.5)),
        emitted_vello_fill_geometry_for_test(&kurbo::Ellipse::new((4.0, 4.0), (3.25, 3.0), 0.0)),
        emitted_vello_fill_geometry_for_test(
            &composition_ordered_nonzero_clip_path_for_test().to_kurbo(),
        ),
    ];
    let expected_fill_rules = [
        VelloFillRuleObservationForTest::NonZero,
        VelloFillRuleObservationForTest::NonZero,
        VelloFillRuleObservationForTest::NonZero,
        VelloFillRuleObservationForTest::NonZero,
        VelloFillRuleObservationForTest::NonZero,
    ];
    let ordered_capture = observed.captures.as_slice().first().is_some_and(|capture| {
        let expected_emitted_transforms = expected_transforms.map(|transform| {
            emitted_vello_transform_for_test(transform.then(capture.initial_transform).unwrap())
        });
        let emitted_clips_are_exact = capture.emitted_draws.get(..5).is_some_and(|draws| {
            draws
                .iter()
                .zip(&expected_geometries)
                .zip(expected_emitted_transforms)
                .zip(expected_fill_rules)
                .all(|(((draw, geometry), transform), fill_rule)| {
                    emitted_vello_clip_is_exact_for_test(draw, geometry, transform, fill_rule)
                })
        });
        let emitted_fill_is_exact = capture.emitted_draws.get(5).is_some_and(|draw| {
            emitted_vello_coverage_fill_is_exact_for_test(draw, capture.target_extent)
        });
        let emitted_end_clips_are_exact = capture
            .emitted_draws
            .get(6..11)
            .is_some_and(|draws| draws.iter().all(emitted_vello_end_clip_is_exact_for_test));
        capture.elements.len() == 5
            && capture.emitted_draws.len() == 11
            && matches!(
                capture.elements[0].clip.geometry(),
                command::RenderClipGeometry::Rect(_)
            )
            && matches!(
                capture.elements[1].clip.geometry(),
                command::RenderClipGeometry::RoundedRect { .. }
            )
            && matches!(
                capture.elements[2].clip.geometry(),
                command::RenderClipGeometry::Circle { .. }
            )
            && matches!(
                capture.elements[3].clip.geometry(),
                command::RenderClipGeometry::Ellipse { .. }
            )
            && matches!(
                capture.elements[4].clip.geometry(),
                command::RenderClipGeometry::Path {
                    fill_rule: FillRule::NonZero,
                    ..
                }
            )
            && capture
                .elements
                .iter()
                .map(|element| element.transform)
                .eq(expected_transforms)
            && emitted_clips_are_exact
            && emitted_fill_is_exact
            && emitted_end_clips_are_exact
            && capture.uses_coverage_resource_role
            && capture.uses_rgba8_target
            && capture.uses_transparent_base
            && capture.raster_antialiasing == Antialiasing::Msaa8
            && capture.raster_target_extent == capture.target_extent
    });

    assert!(
        observed.captures.len() == 1
            && observed.all_vello_capture_count == 2
            && observed.composite_coverage_read_count == 1
            && ordered_capture,
        "graph clips have no bounded Vello coverage capture"
    );
}

#[test]
fn clip_coverage_preserves_fill_rule_antialiasing_and_signed_mapping() {
    let antialiasing = Antialiasing::Msaa16;
    let (commands, coordinate_space) = composition_signed_path_clip_coverage_commands_for_test();
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.25,
        antialiasing,
        Color::TRANSPARENT,
    )
    .unwrap();
    let observed = crate::pass::graph_clip_coverage_observation_for_test(
        commands,
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );
    let expected_texel_origin = Point::new(-2.4, -1.6);
    let expected_initial_transform = Transform::translation(2.4, 1.6)
        .unwrap()
        .then(Transform::scale(1.25, 1.25).unwrap())
        .unwrap();
    let expected_effective_transform = coordinate_space
        .transform()
        .then(expected_initial_transform)
        .unwrap();
    let expected_geometry = emitted_vello_fill_geometry_for_test(
        &composition_signed_even_odd_clip_path_for_test().to_kurbo(),
    );
    let preserves_geometry_and_grid = observed.captures.as_slice().first().is_some_and(|capture| {
        let emitted_clip_is_exact = capture.emitted_draws.first().is_some_and(|draw| {
            emitted_vello_clip_is_exact_for_test(
                draw,
                &expected_geometry,
                emitted_vello_transform_for_test(expected_effective_transform),
                VelloFillRuleObservationForTest::EvenOdd,
            )
        });
        let emitted_fill_is_exact = capture.emitted_draws.get(1).is_some_and(|draw| {
            emitted_vello_coverage_fill_is_exact_for_test(draw, capture.target_extent)
        });
        let emitted_end_clip_is_exact = capture
            .emitted_draws
            .get(2)
            .is_some_and(emitted_vello_end_clip_is_exact_for_test);
        capture.elements.as_slice().first().is_some_and(|element| {
            matches!(
                element.clip.geometry(),
                command::RenderClipGeometry::Path {
                    fill_rule: FillRule::EvenOdd,
                    ..
                }
            ) && element.clip.coordinate_space() == Some(coordinate_space)
                && element.transform == Transform::identity()
        }) && capture.elements.len() == 1
            && capture.emitted_draws.len() == 3
            && emitted_clip_is_exact
            && emitted_fill_is_exact
            && emitted_end_clip_is_exact
            && capture.antialiasing == antialiasing
            && capture.device_origin == (-3, -2)
            && capture.target_extent == PhysicalSize::new(5, 3)
            && capture.texel_origin == expected_texel_origin
            && capture.raster_scale == 1.25
            && capture.first_texel_center == Point::new(-2.0, -1.2)
            && capture.initial_transform == expected_initial_transform
            && capture.uses_transparent_base
            && capture.raster_antialiasing == antialiasing
            && capture.raster_target_extent == capture.target_extent
    });

    assert!(
        observed.captures.len() == 1 && preserves_geometry_and_grid,
        "clip coverage differs from authored Vello geometry or grid"
    );
}

#[test]
fn clip_coverage_is_bound_before_mask_and_opacity() {
    use crate::pass::{
        CompositionOuterOperationObservationForTest as Operation,
        CompositionReadObservationForTest as Read,
    };

    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap();
    let observed = crate::pass::composition_graph_observation_for_test(
        composition_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );
    let expected_reads = [
        Read::Parent,
        Read::Source,
        Read::ClipCoverage,
        Read::AlphaMask,
    ];
    let expected_operations = [
        Operation::SourceMapping,
        Operation::ClipCoverage,
        Operation::AlphaMask,
        Operation::Opacity,
        Operation::Blend,
    ];

    assert!(
        observed.layers_inner_to_outer.len() == 2
            && observed.layers_inner_to_outer.iter().all(|layer| {
                layer.reads == expected_reads && layer.outer_operations == expected_operations
            }),
        "clip coverage lost its ordered composite role"
    );
}

#[test]
fn composition_graph_executor_accepts_only_spine_and_ordered_layer_composition() {
    let mut base_graph_scene = Scene::new();
    base_graph_scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let base_graph_commands = base_graph_scene.normalize(Capabilities::CURRENT).unwrap();
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap();
    let observed = crate::pass::composition_executable_graph_observation_for_test(
        base_graph_commands,
        composition_commands_for_test(),
        runtime_lowering_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.accepts_spine_and_layer_composition_for_all_formats
            && observed.layer_composition_reads_are_exact
            && observed.rejects_color_filter_plus_passes_and_payloads
            && observed.rejects_missing_payloads
            && observed.rejects_malformed_graph_facts
            && observed.rejects_unsupported_output_binding
            && observed.preserves_exact_composition_dispatch,
        "the composition graph executor has no closed pre-allocation subset"
    );
}

#[test]
fn composition_graph_orders_clip_mask_opacity_blend_and_nested_layers() {
    use crate::pass::{
        CompositionOuterOperationObservationForTest as Operation,
        CompositionReadObservationForTest as Read,
    };

    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap();
    let observed = crate::pass::composition_graph_observation_for_test(
        composition_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );
    let expected_operations = [
        Operation::SourceMapping,
        Operation::ClipCoverage,
        Operation::AlphaMask,
        Operation::Opacity,
        Operation::Blend,
    ];
    let outer_clip_transform = Transform::translation(0.5, 0.25).unwrap();
    let inherited_inner_clip_transform = Transform::translation(0.25, 0.5)
        .unwrap()
        .then(outer_clip_transform)
        .unwrap();
    let expected_outer_transform = Transform::translation(1.0, 0.5)
        .unwrap()
        .then(inherited_inner_clip_transform)
        .unwrap();

    assert!(
        observed.layers_inner_to_outer.len() == 2
            && observed.layers_inner_to_outer[0].transform == Transform::scale(0.75, 0.5).unwrap()
            && observed.layers_inner_to_outer[0].opacity == 0.25
            && observed.layers_inner_to_outer[0].blend == BlendMode::Multiply
            && observed.layers_inner_to_outer[0].has_own_clip
            && observed.layers_inner_to_outer[0].inherited_outer_clip_count == 0
            && observed.layers_inner_to_outer[0].reads
                == [
                    Read::Parent,
                    Read::Source,
                    Read::ClipCoverage,
                    Read::AlphaMask
                ]
            && observed.layers_inner_to_outer[0].outer_operations == expected_operations
            && observed.layers_inner_to_outer[0].source_captured_before_outer_semantics
            && observed.layers_inner_to_outer[1].transform == expected_outer_transform
            && observed.layers_inner_to_outer[1].opacity == 1.0
            && observed.layers_inner_to_outer[1].blend == BlendMode::Screen
            && observed.layers_inner_to_outer[1].has_own_clip
            && observed.layers_inner_to_outer[1].inherited_outer_clip_count == 2
            && observed.layers_inner_to_outer[1].inherited_outer_clip_transforms
                == [outer_clip_transform, inherited_inner_clip_transform]
            && observed.layers_inner_to_outer[1].reads
                == [
                    Read::Parent,
                    Read::Source,
                    Read::ClipCoverage,
                    Read::AlphaMask
                ]
            && observed.layers_inner_to_outer[1].outer_operations == expected_operations
            && observed.layers_inner_to_outer[1].source_captured_before_outer_semantics
            && observed.mask_identity_is_preserved,
        "composition graph changed authored outer-operation order"
    );
}

#[test]
fn composition_isolation_starts_from_transparent_black() {
    let base_color = Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap();
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        base_color,
    )
    .unwrap();
    let observed = crate::pass::composition_graph_observation_for_test(
        composition_commands_for_test(),
        context,
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.root_surface_base_clears == 1
            && observed.root_surface_base_color == Some(base_color)
            && observed.transparent_isolation_clears == 2
            && observed.nontransparent_isolation_clears == 0,
        "isolated composition inherited root base color"
    );
}

#[test]
fn graph_preparation_rejects_unsupported_passes_without_resource_or_cache_mutation() {
    let policy = EffectQualityPolicy::AllowReducedPrecision;
    let options = Options::default()
        .with_effect_quality_policy(policy)
        .with_resource_cache_budget(ResourceCacheBudget::new(1024 * 1024));
    let mut renderer = pollster::block_on(Renderer::new(options)).unwrap_or_panic_for_test(
        "graph preparation rejection requires a real selected WGPU device",
    );
    let _surface = pollster::block_on(renderer.create_headless(Size::new(16.0, 12.0), 1.0))
        .unwrap_or_panic_for_test(
            "graph preparation rejection requires a device-backed headless surface",
        );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test("graph preparation rejection requires one ready device bundle");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let working_format = capabilities
        .resolve_effect_working_format(policy)
        .unwrap_or_panic_for_test(
            "the selected graph device must resolve its immutable working format",
        );
    let context = crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
    )
    .unwrap();
    let crate::frame::FramePlan::GpuGraph(graph) = runtime_lowering_commands_for_test()
        .plan_for(context)
        .unwrap_or_panic_for_test("the unsupported-pass fixture must form a validated graph plan")
    else {
        panic!("the unsupported-pass fixture must select the graph route");
    };
    let lowered = crate::pass::LoweredGraphPlan::try_lower_validated_graph(
        &graph,
        working_format,
        Format::Rgba8,
        &capabilities,
    )
    .unwrap_or_panic_for_test("the unsupported-pass fixture must reach runtime lowering");
    let resources = ResourceManager::new(ResourceCacheBudget::new(1024 * 1024));
    let mut pass_cache = DevicePassCache::new();
    let _ = pass_cache.seed_sampler_for_test(ready.device_for_test());
    let resources_before = resources.observation_for_test();
    let pass_cache_before = pass_cache.counts_for_test();

    let preparation = match crate::pass::BasePreparableGraph::try_from_lowered(lowered) {
        Ok(preparable) => crate::pass::PreparedGraph::try_prepare_base(
            preparable,
            policy,
            &capabilities,
            ready.device_for_test(),
            ready.queue_for_test(),
            &resources,
            &pass_cache,
        )
        .map_err(|_| ()),
        Err(_) => Err(()),
    };
    let unsupported_graph_reached_preparation = preparation.is_ok();
    drop(preparation);
    let resources_after = resources.observation_for_test();
    let pass_cache_after = pass_cache.counts_for_test();

    assert!(
        !unsupported_graph_reached_preparation
            && resources_after == resources_before
            && pass_cache_after == pass_cache_before,
        "an unsupported graph reached resource or cache preparation"
    );
}

#[test]
fn semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary() {
    let observed = observe_runtime_lowering_for_test();
    assert!(
        observed.has_exact_closed_vocabulary
            && observed.preserves_backend_ready_resource_facts
            && observed.preserves_semantic_pass_facts,
        "semantic graph has no backend-ready closed lowering"
    );
}

#[test]
fn runtime_lowering_preserves_dependencies_and_last_use_releases() {
    let observed = observe_runtime_lowering_for_test();
    assert!(
        observed.preserves_topological_bindings
            && observed.preserves_exact_last_use_releases
            && observed.rejects_inconsistent_bindings_atomically,
        "runtime lowering changed graph order or lifetime"
    );
}

#[test]
fn runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys() {
    let observed = observe_runtime_lowering_for_test();
    assert!(
        observed.has_exact_cache_keys
            && observed.keys_separate_program_layout_sampling_and_edge
            && observed.keys_separate_source_working_and_output_formats,
        "lowered pass omitted its exact cache keys"
    );
}

#[test]
fn zero_capture_graph_spine_is_rejected_before_preparation() {
    let policy = EffectQualityPolicy::AllowReducedPrecision;
    let options = Options::default()
        .with_effect_quality_policy(policy)
        .with_resource_cache_budget(ResourceCacheBudget::new(1024 * 1024));
    let mut renderer = pollster::block_on(Renderer::new(options)).unwrap_or_panic_for_test(
        "zero-capture graph rejection requires a real selected WGPU device",
    );
    let _surface = pollster::block_on(renderer.create_headless(Size::new(16.0, 12.0), 1.0))
        .unwrap_or_panic_for_test(
            "zero-capture graph rejection requires a device-backed headless surface",
        );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test("zero-capture graph rejection requires one ready device bundle");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let resources = ResourceManager::new(ResourceCacheBudget::new(1024 * 1024));
    let mut pass_cache = DevicePassCache::new();
    let _ = pass_cache.seed_sampler_for_test(ready.device_for_test());
    let resources_before = resources.observation_for_test();
    let pass_cache_before = pass_cache.counts_for_test();

    let lowered = crate::pass::zero_capture_spine_lowered_for_test(
        graph_shader_commands_for_test(),
        graph_shader_frame_context_for_test(),
        capabilities,
        policy,
    )
    .unwrap_or_panic_for_test("the zero-capture fixture must reach validated runtime lowering");
    let preparation = crate::pass::PreparedGraph::try_prepare(
        lowered,
        policy,
        &capabilities,
        ready.device_for_test(),
        ready.queue_for_test(),
        &resources,
        (&pass_cache, false),
    );
    let rejected = preparation.is_err();
    drop(preparation);
    let resources_after = resources.observation_for_test();
    let pass_cache_after = pass_cache.counts_for_test();

    assert!(
        rejected && resources_after == resources_before && pass_cache_after == pass_cache_before,
        "the zero-capture graph spine reached preparation"
    );
}

#[test]
fn graph_base_color_is_initialized_once_and_isolation_is_transparent() {
    let observed = crate::frame::semantic_graph_base_initialization_observation_for_test(
        bounded_offscreen_pass_plan_for_graph_probe(),
    )
    .unwrap_or_panic_for_test("the initialization graph must validate");
    assert!(observed.observes_bounded_offscreen_pass);
    assert!(
        observed.surface_base_initializations == 1
            && observed.isolation_working_images == 1
            && observed.captures_are_transparent,
        "surface base and isolation clears are not modeled exactly once"
    );
    assert_eq!(observed.root_working_images, 1);
    assert_eq!(observed.final_present_intents, 1);
    assert_eq!(observed.surface_base_color, Some(Color::BLACK));
    assert!(observed.empty_results_have_no_descriptor);
    assert!(observed.resource_descriptors_are_spatially_complete);
}

#[test]
fn nested_non_normal_blend_stays_in_masked_layer_source_vello_span() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
        |scene| {
            scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
                scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
            });
        },
    );

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
        .unwrap_or_panic_for_test("the masked blend fixture must produce a complete frame plan");

    assert_eq!(
        plan.vello_spans,
        vec![VelloSpanObservation {
            scope: VelloSpanScopeObservation::LayerSource,
            commands: vec![VelloCommandObservation::LocalLayer],
            captured_before_outer_semantics: true,
        }],
        "capture-local blend group was treated as external to its masked layer source"
    );
}

#[test]
fn nested_mask_boundary_makes_following_multiply_an_ordered_graph_composite() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
        |scene| {
            scene
                .layer(
                    Layer::new()
                        .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
                    |scene| {
                        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
                    },
                )
                .layer(Layer::new().blend(BlendMode::Multiply), |scene| {
                    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
                });
        },
    );

    let result = observe_frame_plan(
        &scene,
        Size::new(8.0, 8.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result.plan.as_ref().unwrap_or_panic_for_test(
        "the nested-mask blend fixture must produce a complete frame plan",
    );

    assert_eq!(
        (&plan.vello_spans, &plan.graph_layer_blends),
        (
            &vec![
                VelloSpanObservation {
                    scope: VelloSpanScopeObservation::LayerSource,
                    commands: vec![VelloCommandObservation::Fill],
                    captured_before_outer_semantics: true,
                },
                VelloSpanObservation {
                    scope: VelloSpanScopeObservation::LayerSource,
                    commands: vec![VelloCommandObservation::Fill],
                    captured_before_outer_semantics: true,
                },
            ],
            &vec![BlendMode::Normal, BlendMode::Multiply, BlendMode::Normal],
        ),
        "following Multiply remained capture-local after the nested mask materialized its parent"
    );
}

#[test]
fn clip_only_wrapper_does_not_make_nested_multiply_capture_local() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
        |scene| {
            scene
                .layer(
                    Layer::new()
                        .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4))),
                    |scene| {
                        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
                    },
                )
                .layer(
                    Layer::new()
                        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)))
                        .unwrap(),
                    |scene| {
                        scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
                            scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
                        });
                    },
                );
        },
    );

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
        .unwrap_or_panic_for_test("the clip-only blend fixture must produce a complete frame plan");

    assert_eq!(
        (&plan.vello_spans, &plan.graph_layer_blends),
        (
            &vec![
                VelloSpanObservation {
                    scope: VelloSpanScopeObservation::LayerSource,
                    commands: vec![VelloCommandObservation::Fill],
                    captured_before_outer_semantics: true,
                },
                VelloSpanObservation {
                    scope: VelloSpanScopeObservation::LayerSource,
                    commands: vec![VelloCommandObservation::Fill],
                    captured_before_outer_semantics: true,
                },
            ],
            &vec![BlendMode::Normal, BlendMode::Multiply, BlendMode::Normal],
        ),
        "ClipOnly made the nested Multiply capture-local after a materialized graph boundary"
    );
}

#[test]
fn clipped_known_mask_source_uses_post_clip_extent_for_validation_and_import() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(Rect::new(1.0, 0.0, 2.0, 1.0)))
            .unwrap()
            .with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(2, 1))),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
        },
    );

    let result = observe_frame_plan(
        &scene,
        Size::new(4.0, 2.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert!(
        result.error_code.is_none()
            && result.plan.as_ref().is_some_and(|plan| {
                plan.route == FramePlanRouteObservation::GpuGraph
                    && plan.plan_count == 1
                    && plan.complete
                    && plan.selection_requirements
                        == [FrameSelectionRequirementObservation::ResolvedAlphaMask]
                    && plan.resolved_alpha_mask_device_extents == [(2, 1)]
            }),
        "known 3x1 source was not validated and imported as the post-clip 2x1 mask extent"
    );
}

#[test]
fn maximal_vello_spans_preserve_authored_command_order() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .stroke(
            Shape::try_circle(Point::new(3.0, 1.0), 1.0).unwrap(),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        )
        .layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(4.0, 0.0, 2.0, 2.0), Color::BLACK);
        })
        .layer(
            Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(3, 3))),
            |scene| {
                scene
                    .fill(Rect::new(0.0, 3.0, 1.0, 1.0), Color::BLACK)
                    .stroke(
                        Shape::rect(Rect::new(1.0, 3.0, 1.0, 1.0)),
                        Stroke::try_new(1.0).unwrap(),
                        Color::BLACK,
                    );
            },
        )
        .image(
            Image::from_rgba(Size::new(1.0, 1.0), vec![255, 255, 255, 255]).unwrap(),
            Rect::new(6.0, 0.0, 1.0, 1.0),
            ImageFit::Stretch,
        )
        .shadow(
            Rect::new(7.0, 0.0, 1.0, 1.0),
            Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        );

    let result = observe_frame_plan(
        &scene,
        Size::new(10.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let plan = result
        .plan
        .as_ref()
        .unwrap_or_panic_for_test("the observation must be complete");
    let expected = vec![
        VelloSpanObservation {
            scope: VelloSpanScopeObservation::CurrentParent,
            commands: vec![
                VelloCommandObservation::Fill,
                VelloCommandObservation::Stroke,
                VelloCommandObservation::LocalLayer,
            ],
            captured_before_outer_semantics: true,
        },
        VelloSpanObservation {
            scope: VelloSpanScopeObservation::LayerSource,
            commands: vec![
                VelloCommandObservation::Fill,
                VelloCommandObservation::Stroke,
            ],
            captured_before_outer_semantics: true,
        },
        VelloSpanObservation {
            scope: VelloSpanScopeObservation::CurrentParent,
            commands: vec![
                VelloCommandObservation::Image,
                VelloCommandObservation::Shadow,
            ],
            captured_before_outer_semantics: true,
        },
    ];

    assert_eq!(
        plan.vello_spans, expected,
        "authored Vello commands are not partitioned into maximal spans"
    );
    assert!(plan.captures_precede_outer_semantics);
    assert!(!plan.graph_to_vello_reentry);
}

#[test]
fn backdrop_plan_depends_on_current_parent_not_cloned_commands() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            scene.fill(Rect::new(2.0, 0.0, 2.0, 2.0), Color::BLACK);
        })
        .fill(Rect::new(4.0, 0.0, 2.0, 2.0), Color::BLACK);

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
        .unwrap_or_panic_for_test("the observation must be complete");

    assert_eq!(
        plan.backdrop_dependency,
        BackdropDependencyObservation::CompletedCurrentParent,
        "backdrop dependency is stored as cloned commands instead of current parent"
    );
    assert_eq!(plan.current_parent_backdrop_reads, 1);
    assert!(!plan.stores_cloned_command_prefix);
}

#[test]
fn graph_planning_requires_explicit_text_ink_bounds_only_for_bounded_subtrees() {
    let mut unspecified = Scene::new();
    unspecified
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            add_planning_text(scene, TextRunBounds::unspecified());
        });
    let unresolved = observe_frame_plan(
        &unspecified,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );

    assert_eq!(
        unresolved.unresolved_resource,
        Some(UnresolvedResourceKind::TextRunInkBounds),
        "bounded graph text lacks an exact unresolved-bounds result"
    );
    assert_eq!(unresolved.error_code, Some(ErrorCode::UnresolvedResource));
    assert!(unresolved.plan.is_none());
    assert!(!unresolved.has_partial_plan);

    let mut ink = Scene::new();
    ink.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK).layer(
        bounded_planning_backdrop(),
        |scene| {
            add_planning_text(
                scene,
                TextRunBounds::try_ink(Rect::new(1.0, 1.0, 4.0, 2.0)).unwrap(),
            );
        },
    );
    let ink_result = observe_frame_plan(
        &ink,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    assert_eq!(
        ink_result.plan.as_ref().map(|plan| plan.route),
        Some(FramePlanRouteObservation::GpuGraph)
    );

    let mut empty = Scene::new();
    empty
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            add_planning_text(scene, TextRunBounds::empty());
        });
    let empty_result = observe_frame_plan(
        &empty,
        Size::new(8.0, 6.0),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    );
    let empty_plan = empty_result
        .plan
        .as_ref()
        .unwrap_or_panic_for_test("empty text must still permit the surrounding supported graph");
    assert_eq!(empty_plan.empty_text_resource_count, 0);
    assert!(
        empty_plan
            .vello_spans
            .iter()
            .all(|span| { !span.commands.contains(&VelloCommandObservation::Text) })
    );
}

fn color_then_drop_shadow_filters_for_test() -> Vec<FilterList> {
    vec![
        authored_color_filter_runs_for_test()[0].clone(),
        FilterList::try_ops(vec![FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(0.5, -0.25),
                FilterBlur::try_new(0.75).unwrap(),
                Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
            )
            .unwrap(),
        )])
        .unwrap(),
    ]
}
