mod frame;
mod model;
mod style;
mod support;

#[cfg(feature = "render-window")]
use super::gpu_transaction::test_support::graph_terminal_loss_after_submission_for_test;
use super::gpu_transaction::test_support::{
    fault_command_buffer_after_submit_for_test, graph_accounting_failure_after_submission_for_test,
    graph_cancellation_after_submission_for_test, graph_scope_failure_after_submission_for_test,
    hold_command_buffer_after_submit_for_test, submit_command_buffer_observed_for_test,
    submit_readback_observed_for_test,
};
use super::gpu_transaction::{GpuOperationLease, GpuOperationStage};
use super::image::{ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey};
#[cfg(not(target_arch = "wasm32"))]
use super::readback::{
    NativeReadbackLateCallbackStageForTest, NativeReadbackStageForTest,
    NativeReadbackStagePhaseForTest,
};
#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::PresentedSurfaceState;
#[cfg(feature = "render-window")]
use super::surface::{
    DisplayFreePresentedSurfaceObservationForTest,
    DisplayFreePresentedSurfaceObservationHandleForTest, PresentedAcquireOutcomeForTest,
};
use super::vello_engine::{
    ActiveVelloEncodingScope, PreparedVelloPass, PreparedVelloPassObservation, RasterParameters,
    TransactionEncodingState, TransactionTargetIntent, VelloAtlasOutcome, VelloEngineState,
    scene::{VelloRasterScenario, VelloScene},
};
use super::{
    backend::*,
    command,
    encode::*,
    filter::{
        BlurPolicy, BlurRadiusInterpretation, KernelSupportRadius, LargeBlurRadiusPolicy,
        TransparentEdgeSamplingPolicy,
    },
    pass::pass_spatial_uniform_bytes_for_test,
    reference::{
        CompiledColorFilterPipeline, MaterializedDropShadowOffsetQuantizationPolicy,
        MaterializedImageFilterStep, PremultipliedRgba8, ReferencePremultipliedRgba8Buffer,
    },
    resource::{
        AllocationGeneration, GaussianKernelBufferLimits, GaussianKernelKey, GaussianKernelPlan,
        GaussianKernelSamplingForm, ResourceAccountingFault, ResourceCacheKey, ResourceIdentity,
        ResourceManager, ResourceRetentionOutcome, WorkingFormat,
    },
    shader::device_pass_cache_owns_exact_key_spaces_for_test,
    style::{ColorFilterOp, ColorFilterPipeline},
    surface::{HeadlessResources, SurfaceBackend},
    texture::{
        EffectTextureDescriptor, EffectTextureRole, TextureDescriptor, TextureUsageIntent,
        headless_texture_descriptor,
    },
};

use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
    time::Duration,
};

#[cfg(not(target_arch = "wasm32"))]
use std::{
    pin::Pin,
    sync::{Condvar, Mutex},
    time::Instant,
};

use super::error::BackendErrorCode;
use super::*;
#[cfg(feature = "render-window")]
use support::add_planning_text;
use support::{
    AHEM_FONT_BYTES, AHEM_GLYPH_ASCENT_E_ACUTE, AHEM_GLYPH_DESCENT_P, AHEM_GLYPH_X, ahem_font,
    assert_finite_positive_rect, assert_premultiplied, authored_color_filter_runs_for_test,
    bounded_backdrop_graph_commands_for_test, bounded_planning_backdrop, box_decoration_edges,
    color_then_blur_filters_for_test, composition_commands_for_test,
    filter_graph_commands_for_test, filter_graph_context_for_test, graph_shader_commands_for_test,
    graph_shader_frame_context_for_test, image_from_buffer, opaque_planning_mask, pixel_alpha,
    pixel_rgba, runtime_lowering_commands_for_test, solid_border,
    spatial_filter_authored_filter_steps_for_test, text_run_for,
};

fn resolved_layer_alpha_mask_from_buffer(buffer: ImageBuffer) -> ResolvedLayerAlphaMask {
    let size = buffer.size();
    ResolvedLayerAlphaMask::try_new(
        image_from_buffer(buffer),
        Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height())),
    )
    .unwrap()
}

trait UnwrapOrPanicForTest<T> {
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T;
}

impl<T> UnwrapOrPanicForTest<T> for Option<T> {
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{message}"),
        }
    }
}

impl<T, E> UnwrapOrPanicForTest<T> for std::result::Result<T, E>
where
    E: std::fmt::Debug,
{
    #[track_caller]
    fn unwrap_or_panic_for_test(self, message: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{message}: {error:?}"),
        }
    }
}
#[test]
fn prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority() {
    let parameters = RasterParameters::try_new(
        PhysicalSize::new(64, 48),
        peniko::Color::BLACK,
        Antialiasing::Area,
    )
    .expect("a non-empty target must prepare");

    for (scenario, antialiasing) in [
        (VelloRasterScenario::Base, Antialiasing::Area),
        (VelloRasterScenario::Base, Antialiasing::Msaa8),
        (VelloRasterScenario::Base, Antialiasing::Msaa16),
        (VelloRasterScenario::LargePath, Antialiasing::Area),
        (VelloRasterScenario::Clip, Antialiasing::Area),
        (VelloRasterScenario::LargePathAndClip, Antialiasing::Area),
    ] {
        let prepared = VelloScene::prepare_raster_scenario_for_test(
            scenario,
            parameters.with_antialiasing(antialiasing),
        )
        .expect("recording preparation must not require a runtime resource");
        let observation = prepared.observation_for_test();
        assert_prepared_vello_pass_basics(&observation);
    }

    let font_data = FontData::try_from_bytes(AHEM_FONT_BYTES.to_vec(), 0)
        .expect("the Ahem fixture must pass selected-glyph preflight");
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 3.0, 19.0, 8.0).unwrap()];
    let run = text_run_for(font_data, 16.0, Transform::identity(), &glyphs);
    let mut scene = VelloScene::default();
    scene
        .encode_text_run(&run)
        .expect("the validated Ahem glyph must encode into the Vello scene");
    let prepared = scene
        .prepare_raster(parameters)
        .expect("only the validated Vello scene may prepare a pass");
    let observation = prepared.observation_for_test();
    assert_prepared_vello_pass_basics(&observation);

    let zero_width = RasterParameters::try_new(
        PhysicalSize::new(0, 48),
        peniko::Color::BLACK,
        Antialiasing::Area,
    )
    .expect_err("a zero-width raster target must be rejected before recording");
    let diagnostic = zero_width
        .invalid_value_diagnostic()
        .expect("an invalid target must retain an invalid-value diagnostic");
    assert_eq!(diagnostic.field(), "raster target width");

    for extent in [
        PhysicalSize::new(u32::MAX - 15, 1),
        PhysicalSize::new(1, u32::MAX - 15),
    ] {
        assert!(
            RasterParameters::try_new(extent, peniko::Color::BLACK, Antialiasing::Area).is_ok(),
            "the largest dimension with room for tile padding must be accepted"
        );
    }

    for (extent, field) in [
        (PhysicalSize::new(u32::MAX - 14, 1), "raster target width"),
        (PhysicalSize::new(1, u32::MAX - 14), "raster target height"),
    ] {
        let error = RasterParameters::try_new(extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect_err("the first dimension without room for tile padding must be rejected");
        let diagnostic = error
            .invalid_value_diagnostic()
            .expect("an oversized target must retain an invalid-value diagnostic");
        assert_eq!(diagnostic.field(), field);
        assert_eq!(diagnostic.value(), (u32::MAX - 14).to_string());
    }
}

fn assert_prepared_vello_pass_basics(observation: &PreparedVelloPassObservation) {
    assert_eq!(
        observation.target_extent_for_test(),
        PhysicalSize::new(64, 48)
    );
    assert!(observation.is_rgba8_storage_for_test());
    assert!(observation.final_dispatch_targets_output_for_test());
    assert!(observation.is_self_consistent_for_test());
    assert!(observation.has_persistent_image_atlas_for_test());
    assert!(observation.has_transient_buffer_for_test());
}

#[test]
fn runtime_capability_report_keeps_precision_flags_independent() {
    let combinations = [(true, true), (true, false), (false, true), (false, false)];

    for (high_precision, reduced_precision) in combinations {
        let precisions = EffectPrecisionCapabilities::new(high_precision, reduced_precision);
        let available = AvailableRuntimeCapabilities::new(Format::Bgra8, precisions, 8_192);
        let report = RuntimeCapabilities::Available(available);

        assert_eq!(precisions.supports_high_precision(), high_precision);
        assert_eq!(precisions.supports_reduced_precision(), reduced_precision);
        assert_eq!(available.surface_format(), Format::Bgra8);
        assert_eq!(available.effect_precisions(), precisions);
        assert_eq!(available.max_effect_texture_dimension_2d(), 8_192);
        assert_eq!(report.available(), Some(available));
        assert_eq!(report.unavailable_reason(), None);
    }

    let unavailable_reason = RuntimeCapabilityUnavailableReason::AdapterUnavailable;
    let unavailable = RuntimeCapabilities::Unavailable(unavailable_reason);
    assert_eq!(unavailable.available(), None);
    assert_eq!(unavailable.unavailable_reason(), Some(unavailable_reason));

    fn assert_report_traits<T: Clone + Copy + std::fmt::Debug + Eq + PartialEq>() {}
    assert_report_traits::<RuntimeCapabilities>();
    assert_report_traits::<AvailableRuntimeCapabilities>();
    assert_report_traits::<EffectPrecisionCapabilities>();
}

#[test]
fn precision_resolver_covers_both_high_only_reduced_only_and_neither() {
    assert_working_format_contracts();
    assert_precision_resolution_matrix();
}

fn assert_working_format_contracts() {
    let required_usages = wgpu::TextureUsages::RENDER_ATTACHMENT
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST);
    for (format, texture_format, bytes_per_pixel) in [
        (
            WorkingFormat::HighPrecision,
            wgpu::TextureFormat::Rgba16Float,
            8,
        ),
        (
            WorkingFormat::ReducedPrecision,
            wgpu::TextureFormat::Rgba8Unorm,
            4,
        ),
    ] {
        assert_eq!(format.texture_format(), texture_format);
        assert_eq!(format.required_usages(), required_usages);
        assert_eq!(
            format.required_format_features(),
            wgpu::TextureFormatFeatureFlags::FILTERABLE,
        );
        assert_eq!(format.bytes_per_pixel(), bytes_per_pixel);

        let complete_features = wgpu::TextureFormatFeatures {
            allowed_usages: required_usages,
            flags: wgpu::TextureFormatFeatureFlags::FILTERABLE,
        };
        assert!(format.is_supported_by(complete_features));
        for required_usage in [
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::TextureUsages::COPY_SRC,
            wgpu::TextureUsages::COPY_DST,
        ] {
            assert!(!format.is_supported_by(wgpu::TextureFormatFeatures {
                allowed_usages: required_usages.difference(required_usage),
                ..complete_features
            }));
        }
        assert!(!format.is_supported_by(wgpu::TextureFormatFeatures {
            flags: wgpu::TextureFormatFeatureFlags::empty(),
            ..complete_features
        }));
    }
}

fn assert_precision_resolution_matrix() {
    let cases = [
        (
            true,
            true,
            EffectQualityPolicy::RequireHighPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            true,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            false,
            EffectQualityPolicy::RequireHighPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            false,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (false, true, EffectQualityPolicy::RequireHighPrecision, None),
        (
            false,
            true,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::ReducedPrecision),
        ),
        (
            false,
            false,
            EffectQualityPolicy::RequireHighPrecision,
            None,
        ),
        (
            false,
            false,
            EffectQualityPolicy::AllowReducedPrecision,
            None,
        ),
    ];

    for (high_precision, reduced_precision, policy, expected) in cases {
        let capabilities =
            DeviceCapabilities::from_test_facts(high_precision, reduced_precision, 8_192);
        let resolution = capabilities.resolve_effect_working_format(policy);

        match expected {
            Some(expected_format) => assert_eq!(
                resolution.expect("the available format should satisfy the requested policy"),
                expected_format,
            ),
            None => {
                let error = resolution
                    .expect_err("the unavailable format should reject the requested policy");
                let expected_diagnostic = RuntimeCapabilityUnavailable::try_new(
                    RuntimeOperation::EffectRendering,
                    RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { policy },
                )
                .unwrap();
                assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
                assert_eq!(
                    error.runtime_capability_unavailable_diagnostic(),
                    Some(&expected_diagnostic),
                );
            }
        }
    }
}

#[test]
fn effect_texture_dimension_is_rejected_before_allocation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("effect texture dimension coverage requires a selected host adapter");
    let maximum = renderer
        .default_device_capabilities_for_test()
        .max_effect_texture_dimension_2d();
    let requested = PhysicalSize::new(u32::MAX, u32::MAX);
    assert!(
        maximum < requested.width(),
        "the selected device must expose a finite over-limit extent"
    );

    let bounds = command::OffscreenBounds::try_new(Rect::new(
        0.0,
        0.0,
        f64::from(requested.width()),
        f64::from(requested.height()),
    ))
    .unwrap();
    let scene = VelloScene::default();
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let options = renderer.options();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("effect texture dimension coverage requires a selected device context")
        .internal_resource_manager_observation_for_test();
    let context = renderer
        .default_offscreen_render_context()
        .expect("effect texture dimension coverage requires a selected device context");

    let error = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .expect_err("an over-limit effect extent should be rejected");
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("effect texture dimension coverage requires retained device resources")
        .internal_resource_manager_observation_for_test();

    assert_eq!(
        resources_after.payload_creation_attempts, resources_before.payload_creation_attempts,
        "over-limit effect extent reached allocation"
    );
    assert_eq!(resources_after.entry_count, resources_before.entry_count);
    let expected_diagnostic = RuntimeCapabilityUnavailable::try_new(
        RuntimeOperation::EffectTextureAllocation,
        RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { requested, maximum },
    )
    .unwrap();
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(&expected_diagnostic),
    );

    let capabilities = DeviceCapabilities::from_test_facts(true, true, maximum);
    for requested in [PhysicalSize::new(1, 1), PhysicalSize::new(maximum, maximum)] {
        capabilities
            .validate_effect_texture_extent(requested)
            .expect("an in-limit nonempty effect extent should pass validation");
    }

    for requested in [
        PhysicalSize::new(0, 0),
        PhysicalSize::new(0, maximum + 1),
        PhysicalSize::new(maximum + 1, 0),
    ] {
        capabilities
            .validate_effect_texture_extent(requested)
            .expect("an empty effect extent should not require texture allocation validation");
    }
}

#[test]
fn options_default_requires_high_precision_and_bounds_retention() {
    let options = Options::new();

    assert_eq!(options, Options::default());
    assert_eq!(options.antialiasing(), Antialiasing::Area);
    assert!(!options.debug());
    assert_eq!(
        options.effect_quality_policy(),
        EffectQualityPolicy::RequireHighPrecision
    );
    assert_eq!(
        options.resource_cache_budget(),
        ResourceCacheBudget::DEFAULT
    );

    let configured = options
        .with_antialiasing(Antialiasing::Msaa16)
        .with_debug(true)
        .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
        .with_resource_cache_budget(ResourceCacheBudget::new(8));

    assert_eq!(configured.antialiasing(), Antialiasing::Msaa16);
    assert!(configured.debug());
    assert_eq!(
        configured.effect_quality_policy(),
        EffectQualityPolicy::AllowReducedPrecision
    );
    assert_eq!(configured.resource_cache_budget().bytes(), 8);
    assert_eq!(
        pollster::block_on(Renderer::new(configured))
            .unwrap()
            .options(),
        configured
    );
}

#[test]
fn resource_cache_budget_zero_disables_idle_retention() {
    let disabled = ResourceCacheBudget::new(0);
    let manager = ResourceManager::new(disabled);
    let mut frame = manager.begin_frame().unwrap();
    let lease = frame.acquire(modeled_resource_key_for_test(1), 4).unwrap();
    frame.release(lease).unwrap();
    let cleanup = frame.finish();

    assert_eq!(disabled, ResourceCacheBudget::DISABLED);
    assert_eq!(disabled.bytes(), 0);
    assert_eq!(ResourceCacheBudget::default(), ResourceCacheBudget::DEFAULT);
    assert_eq!(ResourceCacheBudget::DEFAULT.bytes(), 64 * 1024 * 1024);
    assert_eq!(
        manager.retained_count(),
        0,
        "zero budget retained an idle byte-accounted resource"
    );
    assert_eq!(cleanup.evicted_resources().len(), 1);
}

#[test]
fn scene_lowering_preserves_authored_text_run_bounds() {
    let bounds = TextRunBounds::try_ink(Rect::new(-2.0, -3.0, 4.0, 5.0)).unwrap();
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Bounded scene text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        bounds,
    )
    .unwrap();

    let mut scene = Scene::new();
    scene.text_run(run);

    let [
        scene::Command::TextRun {
            bounds: scene_bounds,
            ..
        },
    ] = scene.commands.as_slice()
    else {
        panic!("direct text run should retain authored bounds in the scene");
    };
    assert_eq!(*scene_bounds, bounds);

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let [
        command::RenderCommand::TextRun {
            bounds: normalized_bounds,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("direct text run should retain authored bounds after normalization");
    };
    assert_eq!(*normalized_bounds, bounds);

    let shadowed_run = TextRun::try_new(
        FontRef::new(1).named("Bounded shadow text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        bounds,
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut shadow_scene = Scene::new();
    shadow_scene.text_shadow_run(TextShadowRun::try_new(shadowed_run, shadows).unwrap());

    let [
        scene::Command::TextShadowRun {
            bounds: shadow_bounds,
            ..
        },
    ] = shadow_scene.commands.as_slice()
    else {
        panic!("text shadow run should retain wrapped authored bounds in the scene");
    };
    assert_eq!(*shadow_bounds, bounds);
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, PresentedResumeAction, ResizeState};
#[test]
fn resolved_alpha_masks_leave_luminance_to_authored_mask_diagnostics() {
    let mask = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let error = mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("luminance masks remain an authored-mask diagnostic");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );
}

#[test]
fn layer_resolved_alpha_mask_applies_after_children_before_parent_composite() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![
            255, 255, 255, 255, //
            255, 255, 255, 128, //
            0, 0, 0, 0,
        ],
    )
    .unwrap();
    let layer = Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 3.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        );
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!(pixel_rgba(&output, 0, 0)[0] > 200);
    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn nested_resolved_alpha_masked_layers_compose_in_child_then_parent_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0)).unwrap();
    let inner_mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 255, 255, 255, 128],
    )
    .unwrap();
    let outer_mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 128, 255, 255, 255, 255],
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(outer_mask)),
        |scene| {
            scene.layer(
                Layer::new()
                    .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(inner_mask)),
                |scene| {
                    scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
                },
            );
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!((96..=160).contains(&pixel_alpha(&output, 0, 0)));
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

#[test]
fn layer_resolved_alpha_mask_respects_layer_clip_before_masking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 255, 255, 255, 255],
    )
    .unwrap();
    let mask =
        ResolvedLayerAlphaMask::try_new(image_from_buffer(mask), Rect::new(1.0, 0.0, 2.0, 1.0))
            .unwrap();
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(1.0, 0.0, 2.0, 1.0)))
        .unwrap()
        .with_resolved_alpha_mask(mask);
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert!(pixel_alpha(&output, 2, 0) > 200);
}

#[test]
fn layer_resolved_alpha_mask_composites_after_layer_transform() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 255, 255, 255]).unwrap();
    let layer = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn layer_resolved_alpha_mask_combines_mask_child_opacity_and_layer_opacity() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 255, 255, 128]).unwrap();
    let layer = Layer::new()
        .try_opacity(0.5)
        .unwrap()
        .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    let alpha = pixel_alpha(&output, 0, 0);

    assert!((24..=40).contains(&alpha), "unexpected alpha {alpha}");
}

#[test]
fn reference_blur_zero_radius_is_identity() {
    let pixels = vec![
        PremultipliedRgba8::TRANSPARENT,
        PremultipliedRgba8::try_new(20, 40, 60, 80).unwrap(),
        PremultipliedRgba8::try_new(10, 5, 0, 10).unwrap(),
        PremultipliedRgba8::try_new(255, 128, 64, 255).unwrap(),
    ];
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(2, 2), pixels).unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(0.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(blurred, source);
}

#[test]
fn reference_blur_small_radius_spreads_impulse_deterministically() {
    let impulse = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 3),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            impulse,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    let expected = [15, 25, 15, 25, 41, 25, 15, 25, 15];
    for y in 0..3 {
        for x in 0..3 {
            let value = expected[(y * 3 + x) as usize];
            assert_eq!(
                blurred.pixel(x, y).unwrap(),
                PremultipliedRgba8::try_new(value, 0, 0, value).unwrap(),
                "unexpected blurred impulse at {x},{y}",
            );
        }
    }
}

#[test]
fn reference_blur_samples_outside_source_as_transparent_black() {
    let opaque = PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap();
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![opaque])
            .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(
        blurred.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(41, 41, 41, 41).unwrap()
    );
}

#[test]
fn reference_blur_preserves_partially_transparent_colored_invariants() {
    let partial = PremultipliedRgba8::try_new(80, 40, 20, 128).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            partial,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(
        blurred.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(8, 4, 2, 12).unwrap()
    );
    assert_eq!(
        blurred.pixel(1, 0).unwrap(),
        PremultipliedRgba8::try_new(13, 6, 3, 20).unwrap()
    );
    assert_eq!(
        blurred.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(8, 4, 2, 12).unwrap()
    );
    for x in 0..3 {
        assert_premultiplied(blurred.pixel(x, 0).unwrap());
    }
}

#[test]
fn reference_blur_uses_large_radius_policy() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 1),
        vec![PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap()],
    )
    .unwrap();
    let reject = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_reject_above(1.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();
    let clamp = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_clamp_to(1.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();

    let error = source
        .apply_blur(FilterBlur::try_new(2.0).unwrap(), reject)
        .expect_err("reject policy should reject large blur radius");
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
    assert_eq!(
        source
            .apply_blur(FilterBlur::try_new(2.0).unwrap(), clamp)
            .unwrap()
            .pixel(0, 0)
            .unwrap(),
        PremultipliedRgba8::try_new(41, 0, 0, 41).unwrap()
    );
}

#[test]
fn reference_blur_is_deterministic_across_repeated_runs() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(10, 20, 30, 40).unwrap(),
            PremultipliedRgba8::try_new(200, 0, 0, 200).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(0, 80, 20, 100).unwrap(),
        ],
    )
    .unwrap();

    let first = source
        .apply_blur(
            FilterBlur::try_new(1.25).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let second = source
        .apply_blur(
            FilterBlur::try_new(1.25).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn reference_buffers_compare_with_deterministic_equality() {
    let pixel = PremultipliedRgba8::try_new(8, 4, 2, 16).unwrap();
    let first = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![PremultipliedRgba8::TRANSPARENT, pixel],
    )
    .unwrap();
    let same = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![PremultipliedRgba8::TRANSPARENT, pixel],
    )
    .unwrap();
    let different = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![pixel, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();

    assert_eq!(first, same);
    assert_ne!(first, different);
}

#[test]
fn reference_color_filter_identity_ops_preserve_pixels_byte_for_byte() {
    let pixel = PremultipliedRgba8::try_new(64, 32, 16, 128).unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.0).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(0.0).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(std::f64::consts::TAU).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.0).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.0).unwrap()),
    ]);

    let buffer =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![pixel])
            .unwrap();

    assert_eq!(pixel.apply_color_filter_pipeline(&pipeline).unwrap(), pixel);
    assert_eq!(
        buffer.apply_color_filter_pipeline(&pipeline).unwrap(),
        buffer
    );
}

#[test]
fn color_filter_known_vectors_use_spec_constants_not_oracle_constants() {
    assert_literal_color_filter_primary_vectors();
    assert_literal_color_filter_identity_and_matrix_vectors();
    assert_literal_color_filter_scalar_and_clamp_vectors();
}

fn assert_literal_color_filter_primary_vectors() {
    let red_primary = PremultipliedRgba8::try_new(54, 0, 0, 54).unwrap();
    let grayscale = color_filter_pipeline([ColorFilterOp::Grayscale(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )]);
    assert_eq!(
        red_primary.apply_color_filter_pipeline(&grayscale).unwrap(),
        PremultipliedRgba8::try_new(11, 11, 11, 54).unwrap(),
        "the grayscale primary vector differs from the literal reference color-matrix result"
    );

    let green_primary = PremultipliedRgba8::try_new(0, 79, 0, 79).unwrap();
    assert_eq!(
        green_primary
            .apply_color_filter_pipeline(&grayscale)
            .unwrap(),
        PremultipliedRgba8::try_new(57, 57, 57, 79).unwrap(),
        "the grayscale green primary differs from the literal reference color matrix"
    );
    let blue_primary = PremultipliedRgba8::try_new(0, 0, 104, 104).unwrap();
    assert_eq!(
        blue_primary
            .apply_color_filter_pipeline(&grayscale)
            .unwrap(),
        PremultipliedRgba8::try_new(8, 8, 8, 104).unwrap(),
        "the grayscale blue primary differs from the literal reference color matrix"
    );

    let saturation_zero =
        color_filter_pipeline([ColorFilterOp::Saturate(FilterAmount::try_new(0.0).unwrap())]);
    assert_eq!(
        red_primary
            .apply_color_filter_pipeline(&saturation_zero)
            .unwrap(),
        PremultipliedRgba8::try_new(12, 12, 12, 54).unwrap(),
        "saturation reused the grayscale luma constants"
    );
}

fn assert_literal_color_filter_identity_and_matrix_vectors() {
    let identity_sample = PremultipliedRgba8::try_new(101, 67, 23, 127).unwrap();
    for identity in [
        color_filter_pipeline([ColorFilterOp::Saturate(FilterAmount::try_new(1.0).unwrap())]),
        color_filter_pipeline([ColorFilterOp::HueRotate(
            FilterAngle::try_radians(0.0).unwrap(),
        )]),
        color_filter_pipeline([ColorFilterOp::Sepia(
            UnitFilterAmount::try_new(0.0).unwrap(),
        )]),
    ] {
        assert_eq!(
            identity_sample
                .apply_color_filter_pipeline(&identity)
                .unwrap(),
            identity_sample,
            "a literal reference color-matrix zero or identity vector changed the source"
        );
    }

    let opaque_red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let hue_quarter_turn = color_filter_pipeline([ColorFilterOp::HueRotate(
        FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
    )]);
    assert_eq!(
        opaque_red
            .apply_color_filter_pipeline(&hue_quarter_turn)
            .unwrap(),
        PremultipliedRgba8::try_new(0, 91, 0, 255).unwrap(),
        "quarter-turn hue rotation differs from the literal reference color matrix"
    );

    let sepia = color_filter_pipeline([ColorFilterOp::Sepia(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )]);
    assert_eq!(
        opaque_red.apply_color_filter_pipeline(&sepia).unwrap(),
        PremultipliedRgba8::try_new(100, 89, 69, 255).unwrap(),
        "full sepia differs from the literal reference color matrix"
    );
}

fn assert_literal_color_filter_scalar_and_clamp_vectors() {
    let identity_sample = PremultipliedRgba8::try_new(101, 67, 23, 127).unwrap();
    let brightness = color_filter_pipeline([ColorFilterOp::Brightness(
        FilterAmount::try_new(2.0).unwrap(),
    )]);
    assert_eq!(
        PremultipliedRgba8::try_new(192, 64, 1, 255)
            .unwrap()
            .apply_color_filter_pipeline(&brightness)
            .unwrap(),
        PremultipliedRgba8::try_new(255, 128, 2, 255).unwrap(),
        "brightness did not clamp at its source-operation boundary"
    );
    let clamped_brightness_chain = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
    ]);
    assert_eq!(
        identity_sample
            .apply_color_filter_pipeline(&clamped_brightness_chain)
            .unwrap(),
        PremultipliedRgba8::try_new(64, 64, 23, 127).unwrap(),
        "the oracle omitted a source-operation clamp and premultiply boundary"
    );

    let contrast =
        color_filter_pipeline([ColorFilterOp::Contrast(FilterAmount::try_new(2.0).unwrap())]);
    assert_eq!(
        PremultipliedRgba8::try_new(0, 128, 255, 255)
            .unwrap()
            .apply_color_filter_pipeline(&contrast)
            .unwrap(),
        PremultipliedRgba8::try_new(0, 129, 255, 255).unwrap(),
        "contrast did not clamp at its source-operation boundary"
    );

    let near_gray_saturation = color_filter_pipeline([ColorFilterOp::Saturate(
        FilterAmount::try_new(f64::MAX).unwrap(),
    )]);
    assert_eq!(
        PremultipliedRgba8::try_new(128, 127, 128, 255)
            .unwrap()
            .apply_color_filter_pipeline(&near_gray_saturation)
            .unwrap(),
        PremultipliedRgba8::try_new(255, 0, 255, 255).unwrap(),
        "near-gray saturation did not clamp finitely at its operation boundary"
    );
}

#[test]
fn reference_color_filter_partial_ops_match_deterministic_bytes() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(200, 122, 186, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(-std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(86, 164, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(146, 161, 167, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let pipeline = color_filter_pipeline([op]);
        assert_eq!(
            pixel.apply_color_filter_pipeline(&pipeline).unwrap(),
            expected,
            "unexpected output for {op:?}"
        );
    }
}

fn bounded_backdrop_scene_for_test() -> Scene {
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
}

#[test]
fn reference_color_filter_extreme_ops_clamp_to_valid_premultiplied_pixels() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ),
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(200, 255, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(73, 173, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(143, 143, 143, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(155, 105, 55, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::TRANSPARENT,
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(143, 143, 143, 255).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(57, 157, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(192, 171, 134, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let filtered = pixel
            .apply_color_filter_pipeline(&color_filter_pipeline([op]))
            .unwrap();
        assert_eq!(filtered, expected, "unexpected output for {op:?}");
        assert_premultiplied(filtered);
    }
}

#[test]
fn reference_color_filter_buffer_preserves_transparency_and_partial_alpha_invariants() {
    let partial = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![PremultipliedRgba8::TRANSPARENT, partial],
    )
    .unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ]);

    let filtered = buffer.apply_color_filter_pipeline(&pipeline).unwrap();
    let transparent = filtered.pixel(0, 0).unwrap();
    let partial = filtered.pixel(1, 0).unwrap();

    assert_eq!(transparent, PremultipliedRgba8::TRANSPARENT);
    assert_eq!(partial, PremultipliedRgba8::try_new(26, 7, 0, 64).unwrap());
    assert_premultiplied(transparent);
    assert_premultiplied(partial);
}

#[test]
fn compiled_color_filter_pipeline_matches_per_op_reference_chain() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(0.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.4).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
        ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.6).unwrap()),
    ]);
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(compiled.source_ops(), pipeline.ops());
    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&compiled)
            .unwrap(),
        pixel.apply_color_filter_pipeline(&pipeline).unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_applies_to_reference_buffers() {
    let first = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let second = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![first, second],
    )
    .unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
    ]);
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(
        buffer
            .apply_compiled_color_filter_pipeline(&compiled)
            .unwrap(),
        buffer.apply_color_filter_pipeline(&pipeline).unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_fuses_adjacent_color_steps() {
    let fused_color_run = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);
    let opacity_boundary = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);

    assert_eq!(
        CompiledColorFilterPipeline::try_from_pipeline(&fused_color_run)
            .unwrap()
            .executable_step_count(),
        1
    );
    assert_eq!(
        CompiledColorFilterPipeline::try_from_pipeline(&opacity_boundary)
            .unwrap()
            .executable_step_count(),
        3
    );
}

#[test]
fn compiled_color_filter_pipeline_preserves_order_sensitivity() {
    let pixel = PremultipliedRgba8::try_new(90, 130, 210, 255).unwrap();
    let contrast_then_brightness = color_filter_pipeline([
        ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
    ]);
    let brightness_then_contrast = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
    ]);
    let contrast_then_brightness =
        CompiledColorFilterPipeline::try_from_pipeline(&contrast_then_brightness).unwrap();
    let brightness_then_contrast =
        CompiledColorFilterPipeline::try_from_pipeline(&brightness_then_contrast).unwrap();

    assert_ne!(
        pixel
            .apply_compiled_color_filter_pipeline(&contrast_then_brightness)
            .unwrap(),
        pixel
            .apply_compiled_color_filter_pipeline(&brightness_then_contrast)
            .unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_sequences_opacity_with_color_steps() {
    let pixel = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let opacity_then_invert = color_filter_pipeline([
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ]);
    let invert_then_opacity = color_filter_pipeline([
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
    ]);
    let opacity_then_invert_compiled =
        CompiledColorFilterPipeline::try_from_pipeline(&opacity_then_invert).unwrap();
    let invert_then_opacity_compiled =
        CompiledColorFilterPipeline::try_from_pipeline(&invert_then_opacity).unwrap();

    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&opacity_then_invert_compiled)
            .unwrap(),
        pixel
            .apply_color_filter_pipeline(&opacity_then_invert)
            .unwrap()
    );
    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&invert_then_opacity_compiled)
            .unwrap(),
        pixel
            .apply_color_filter_pipeline(&invert_then_opacity)
            .unwrap()
    );
    assert_ne!(
        pixel
            .apply_compiled_color_filter_pipeline(&opacity_then_invert_compiled)
            .unwrap(),
        pixel
            .apply_compiled_color_filter_pipeline(&invert_then_opacity_compiled)
            .unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_rejects_empty_construction() {
    let error = CompiledColorFilterPipeline::try_from_ops(Vec::new())
        .expect_err("empty compiled pipelines should be unconstructable");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("compiled color filter pipeline")
    );
}

#[test]
fn image_straight_rgba8_converts_to_premultiplied_and_back_deterministically() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![90, 120, 150, 0, 64, 128, 255, 128, 255, 10, 20, 255],
    )
    .unwrap();

    let premultiplied =
        reference::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
    assert_eq!(
        premultiplied.pixel(0, 0).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
    assert_eq!(
        premultiplied.pixel(1, 0).unwrap(),
        PremultipliedRgba8::try_new(32, 64, 128, 128).unwrap()
    );
    assert_eq!(
        premultiplied.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(255, 10, 20, 255).unwrap()
    );

    let straight =
        reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&premultiplied)
            .unwrap();

    assert_eq!(straight.size(), PhysicalSize::new(3, 1));
    assert_eq!(
        straight.rgba(),
        &[0, 0, 0, 0, 64, 128, 255, 128, 255, 10, 20, 255]
    );
}

#[test]
fn image_color_filter_execution_applies_color_chain_to_one_pixel_image() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(0.5).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(1.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[50, 75, 100, 255]);
}

#[test]
fn image_color_filter_execution_applies_color_chain_to_multi_pixel_buffer() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 2),
        vec![
            64, 128, 255, 128, 10, 20, 30, 0, 100, 150, 200, 255, 20, 40, 80, 64,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.5).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(2, 2));
    assert_eq!(
        filtered.rgba(),
        &[
            32, 64, 128, 64, 0, 0, 0, 0, 50, 76, 100, 128, 16, 24, 40, 32,
        ]
    );
}

#[test]
fn image_color_filter_execution_preserves_buffer_size_and_rgba_order() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![10, 20, 30, 40, 200, 150, 100, 255],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(2, 1));
    assert_eq!(filtered.rgba(), &[13, 19, 32, 40, 200, 150, 100, 255]);
}

#[test]
fn image_color_filter_execution_changes_image_identity_when_bytes_change() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_ne!(filtered.id(), image.id());
    assert_eq!(filtered.bytes.as_ref(), &[155, 105, 55, 255]);
}

#[test]
fn image_filter_execution_blurs_one_pixel_transparent_and_opaque_images() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 0])).unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let transparent = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(transparent.size(), Size::new(1.0, 1.0));
    assert_eq!(transparent.bytes.as_ref(), &[0, 0, 0, 0]);
    assert_eq!(
        transparent.id(),
        image.id(),
        "identity stays stable when blur leaves bytes unchanged"
    );

    let opaque =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let opaque_paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(opaque.id(), opaque.size()).unwrap(),
        filters,
    )
    .unwrap();

    let blurred = reference::ResolvedImageColorFilterExecution::try_new(&opaque_paint, &opaque)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(blurred.size(), Size::new(1.0, 1.0));
    assert_eq!(blurred.bytes.as_ref(), &[100, 149, 199, 41]);
    assert_ne!(
        blurred.id(),
        opaque.id(),
        "filtered output identity changes when blur changes bytes"
    );
}

#[test]
fn image_filter_execution_blurs_multi_pixel_image_with_transparent_edges() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0],
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();

    let blurred =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(blurred.size(), PhysicalSize::new(3, 1));
    assert_eq!(
        blurred.rgba(),
        &[255, 0, 0, 25, 255, 0, 0, 41, 255, 0, 0, 25]
    );
}

#[test]
fn filtered_image_paint_executes_blur_with_matching_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 41, 255, 0, 0, 25]);

    let wrong_id = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(ImageId::new(image.id().get() + 1), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();
    let wrong_size = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), Size::new(1.0, 1.0)).unwrap(),
        filters,
    )
    .unwrap();

    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn materialized_image_filters_preserve_color_and_blur_order() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![200, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0],
    )
    .unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let color_before_blur = FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap();
    let blur_before_color = FilterList::try_ops(vec![blur, brightness]).unwrap();

    let color_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_blur,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let blur_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &blur_before_color,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();

    assert_eq!(color_before.size(), PhysicalSize::new(3, 1));
    assert_eq!(blur_before.size(), PhysicalSize::new(3, 1));
    assert_ne!(color_before.rgba(), blur_before.rgba());
}

#[test]
fn materialized_image_blur_keeps_output_clipped_to_source_region() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 2),
        vec![255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(4.0).unwrap())]).unwrap();

    let blurred =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(
        blurred.size(),
        source.size(),
        "materialized image blur inflates for sampling but clips output to source image extent"
    );
    assert_eq!(blurred.rgba().len(), source.rgba().len());
}

#[test]
fn materialized_drop_shadow_quantizes_positive_fractional_offsets_to_nearest_device_pixel() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy
            .quantize(1.25, "filter drop-shadow offset x")
            .unwrap(),
        1
    );
    assert_eq!(
        policy
            .quantize(1.75, "filter drop-shadow offset x")
            .unwrap(),
        2
    );
}

#[test]
fn materialized_drop_shadow_quantizes_negative_fractional_offsets_to_nearest_device_pixel() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy
            .quantize(-1.25, "filter drop-shadow offset x")
            .unwrap(),
        -1
    );
    assert_eq!(
        policy
            .quantize(-1.75, "filter drop-shadow offset x")
            .unwrap(),
        -2
    );
}

#[test]
fn materialized_drop_shadow_quantizes_half_pixel_offsets_away_from_zero() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy.quantize(0.5, "filter drop-shadow offset x").unwrap(),
        1
    );
    assert_eq!(
        policy
            .quantize(-0.5, "filter drop-shadow offset x")
            .unwrap(),
        -1
    );
}

#[test]
fn materialized_drop_shadow_uses_alpha_mask_not_source_bounds() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 3),
        vec![
            0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(3, 3));
    assert_eq!(pixel_rgba(&filtered, 1, 0), [255, 0, 0, 255]);
    assert_eq!(pixel_rgba(&filtered, 2, 0), [0, 0, 0, 255]);
    assert_eq!(pixel_rgba(&filtered, 2, 1), [0, 0, 0, 255]);
    assert_eq!(
        pixel_rgba(&filtered, 1, 2),
        [0, 0, 0, 0],
        "CSS drop-shadow follows the source alpha mask, not the image border box"
    );
}

#[test]
fn materialized_drop_shadow_clips_offset_and_blur_to_source_extent() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 3),
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), source.size());
    assert_eq!(filtered.rgba().len(), source.rgba().len());
    assert_eq!(pixel_rgba(&filtered, 1, 1), [255, 255, 255, 255]);
    assert!(
        pixel_alpha(&filtered, 2, 0) > 0,
        "blurred offset shadow should contribute inside the clipped source extent"
    );
}

#[test]
fn materialized_drop_shadow_composites_shadow_behind_source() {
    let source = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 0, 0, 128]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[170, 0, 0, 192]);
}

#[test]
fn filtered_image_paint_executes_drop_shadow_with_matching_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 255, 0, 0, 0, 255]);

    let wrong_id = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(ImageId::new(image.id().get() + 1), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();
    let wrong_size = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), Size::new(1.0, 1.0)).unwrap(),
        filters,
    )
    .unwrap();

    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn resource_only_drop_shadow_filtered_image_paint_stays_rejected() {
    let resource = ResolvedImageResource::try_new(ImageId::new(41), Size::new(2.0, 1.0)).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();

    let unsupported = paint
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("resource-only filtered image paint is not materialized bytes");

    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn materialized_filters_after_drop_shadow_apply_to_composed_output() {
    let source =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[0, 255, 255, 255, 255, 255, 255, 255]);
}

#[test]
fn materialized_filters_before_drop_shadow_shape_current_alpha_mask() {
    let source =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[255, 0, 0, 128, 0, 0, 0, 128]);
}

#[test]
fn materialized_image_filter_reference_preserves_nonzero_blur_then_drop_shadow() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 41, 103, 0, 0, 62]);
    assert_ne!(
        filtered.id(),
        image.id(),
        "materialized filtered output identity should reflect nonzero blur/drop-shadow bytes"
    );
}

#[test]
fn mixed_color_and_pixel_filters_preserve_authored_order() {
    let image_buffer =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let color_before_pixel = FilterList::try_ops(vec![
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let pixel_before_color = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let color_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_pixel,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let pixel_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &pixel_before_color,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    assert_ne!(
        color_before.rgba(),
        pixel_before.rgba(),
        "mixed color and pixel-moving filters must preserve authored order"
    );
}

#[test]
fn color_filter_operations_match_compiled_and_reference_bytes() {
    let source = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(200, 122, 186, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(146, 161, 167, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let pipeline = color_filter_pipeline([op]);
        let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

        assert_eq!(
            source
                .apply_compiled_color_filter_pipeline(&compiled)
                .unwrap(),
            expected,
            "unexpected compiled output for {op:?}"
        );
        assert_eq!(
            source
                .apply_compiled_color_filter_pipeline(&compiled)
                .unwrap(),
            source.apply_color_filter_pipeline(&pipeline).unwrap(),
            "compiled and CPU reference paths should agree for {op:?}"
        );
    }
}

#[test]
fn materialized_color_filter_fusion_matches_reference_bytes() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![100, 150, 200, 255, 64, 128, 255, 128],
    )
    .unwrap();
    let filters = color_filter_list([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);
    let pipeline = filters
        .color_filter_pipeline()
        .unwrap()
        .expect("color-only filters should produce an executable color pipeline");
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(compiled.executable_step_count(), 1);

    let premultiplied =
        reference::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
    let reference = premultiplied
        .apply_compiled_color_filter_pipeline(&compiled)
        .unwrap();
    let expected =
        reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&reference)
            .unwrap();
    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered, expected);
    assert_ne!(filtered.rgba(), source.rgba());
}

#[test]
fn materialized_filter_support_does_not_enable_layer_effect_execution() {
    let image_buffer =
        ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![100, 150, 200, 255]).unwrap();
    let shadow = Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap();
    let drop_shadow =
        FilterList::try_ops(vec![FilterOp::try_drop_shadow(shadow).unwrap()]).unwrap();

    let drop_shadow_output =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
            &drop_shadow,
            &image_buffer,
        )
        .unwrap()
        .execute_to_image_buffer()
        .unwrap();
    assert_eq!(drop_shadow_output.size(), image_buffer.size());

    let layer_filter_error = normalize_single_layer_error(
        Layer::new()
            .try_filter(Filter::try_blur(2.0).unwrap())
            .unwrap(),
    );
    assert_eq!(
        layer_filter_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );

    let layer_mask_error = normalize_single_layer_error(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
            .unwrap(),
    );
    assert_eq!(
        layer_mask_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );

    for unsupported in [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaskExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
    ] {
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported compositor execution must remain typed");

        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn unfiltered_images_preserve_direct_sampling_normalization() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let mut scene = Scene::new();
    scene.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(scene.stats().images, 1);
    assert_eq!(scene.stats().layers, 0);
    assert!(matches!(
        normalized.commands.as_slice(),
        [command::RenderCommand::Image { .. }]
    ));

    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 10.0, 4.0),
        Size::new(2.0, 2.0),
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::contain(),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(3.0, 0.0, 4.0, 4.0));
}

fn color_filter_pipeline<const N: usize>(ops: [ColorFilterOp; N]) -> ColorFilterPipeline {
    color_filter_list(ops)
        .color_filter_pipeline()
        .unwrap()
        .unwrap()
}

fn color_filter_list<const N: usize>(ops: [ColorFilterOp; N]) -> FilterList {
    let ops = ops
        .into_iter()
        .map(|op| match op {
            ColorFilterOp::Brightness(amount) => FilterOp::brightness(amount),
            ColorFilterOp::Contrast(amount) => FilterOp::contrast(amount),
            ColorFilterOp::Grayscale(amount) => FilterOp::grayscale(amount),
            ColorFilterOp::HueRotate(angle) => FilterOp::hue_rotate(angle),
            ColorFilterOp::Invert(amount) => FilterOp::invert(amount),
            ColorFilterOp::Opacity(amount) => FilterOp::opacity(amount),
            ColorFilterOp::Saturate(amount) => FilterOp::saturate(amount),
            ColorFilterOp::Sepia(amount) => FilterOp::sepia(amount),
        })
        .collect();
    FilterList::try_ops(ops).unwrap()
}

fn normalize_single_layer_error(layer: Layer) -> Error {
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    scene
        .normalize(Capabilities::CURRENT)
        .expect_err("unsupported layer effects should reject during normalization")
}

#[test]
fn texture_descriptor_equality_uses_size_format_and_intent() {
    let size = PhysicalSize::new(32, 16);
    let readback =
        TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::ReadbackReference)
            .unwrap();
    let same =
        TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::ReadbackReference)
            .unwrap();
    let different_intent =
        TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::IntermediatePass)
            .unwrap();

    assert_eq!(readback, same);
    assert_ne!(readback, different_intent);
    assert_eq!(readback.physical_size(), size);
    assert_eq!(readback.format(), Format::Rgba8);
    assert_eq!(readback.intent(), TextureUsageIntent::ReadbackReference);
}

#[test]
fn texture_cache_keys_are_stable_without_raw_resources() {
    let descriptor = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(8, 4),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();

    assert_eq!(descriptor.cache_key(), descriptor.cache_key(),);
    assert_ne!(
        descriptor.cache_key(),
        EffectTextureDescriptor::try_coverage(
            PhysicalSize::new(8, 4),
            wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .unwrap()
        .cache_key()
    );
}

#[test]
fn texture_cache_records_misses_reuse_hits_and_live_count() {
    let descriptor = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(4, 4),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    let key = ResourceCacheKey::EffectTexture(descriptor.cache_key());
    let byte_len = descriptor.checked_byte_len().unwrap();
    let manager = ResourceManager::default();

    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame.acquire(key, byte_len).unwrap();
    assert_eq!(manager.stats().allocations, 1);
    assert_eq!(manager.stats().misses, 1);
    assert_eq!(manager.stats().hits, 0);
    assert_eq!(manager.live_count(), 1);

    first_frame.release(first).unwrap();
    let _ = first_frame.finish();
    let mut second_frame = manager.begin_frame().unwrap();
    let _second = second_frame.acquire(key, byte_len).unwrap();

    assert_eq!(manager.stats().allocations, 1);
    assert_eq!(manager.stats().misses, 1);
    assert_eq!(manager.stats().hits, 1);
    assert_eq!(manager.live_count(), 1);
}

fn modeled_resource_key_for_test(discriminator: u32) -> ResourceCacheKey {
    modeled_effect_texture_for_test(PhysicalSize::new(discriminator.max(1), 1)).0
}

fn modeled_effect_texture_for_test(physical_size: PhysicalSize) -> (ResourceCacheKey, u64) {
    let descriptor =
        EffectTextureDescriptor::try_capture(physical_size, wgpu::TextureUsages::TEXTURE_BINDING)
            .unwrap();
    (
        ResourceCacheKey::EffectTexture(descriptor.cache_key()),
        descriptor.checked_byte_len().unwrap(),
    )
}

#[test]
fn resource_leases_reject_stale_generation_and_double_release_by_model() {
    let manager = ResourceManager::default();
    let foreign_manager = ResourceManager::default();
    let mut frame = manager.begin_frame().unwrap();
    let foreign_manager_frame = foreign_manager.begin_frame().unwrap();
    let foreign_frame = manager.begin_frame().unwrap();
    let lease = frame.acquire(ResourceCacheKey::VelloAtlas, 4).unwrap();
    let stale = lease.token_for_test();

    assert_eq!(stale.manager_identity, frame.manager_identity_for_test());
    assert_eq!(stale.frame_identity, frame.frame_identity_for_test());
    assert_eq!(stale.resource_identity, lease.resource_identity());
    assert_eq!(stale.allocation_generation.get_for_test(), 1);

    let current = frame
        .replace(lease, modeled_resource_key_for_test(1), 8)
        .unwrap();
    let current_token = current.token_for_test();
    let replay_after_release = current_token.clone();
    assert_eq!(current_token.resource_identity, stale.resource_identity);
    assert!(
        current_token.allocation_generation.get_for_test()
            > stale.allocation_generation.get_for_test(),
        "replacement allocation generation must advance monotonically"
    );

    let before_rejections = manager.observation_for_test();
    let stale_error = frame
        .release_injected_for_test(stale)
        .expect_err("a stale allocation generation must be rejected");

    let mut foreign_manager_token = current_token.clone();
    foreign_manager_token.manager_identity = foreign_manager_frame.manager_identity_for_test();
    let foreign_manager_error = frame
        .release_injected_for_test(foreign_manager_token)
        .expect_err("a foreign manager identity must be rejected");

    let mut foreign_frame_token = current_token.clone();
    foreign_frame_token.frame_identity = foreign_frame.frame_identity_for_test();
    let foreign_frame_error = frame
        .release_injected_for_test(foreign_frame_token)
        .expect_err("a foreign frame identity must be rejected");

    let mut foreign_resource_token = current_token.clone();
    foreign_resource_token.resource_identity = ResourceIdentity::from_raw_for_test(u64::MAX);
    let foreign_resource_error = frame
        .release_injected_for_test(foreign_resource_token)
        .expect_err("a foreign resource identity must be rejected");

    let mut foreign_allocation_token = current_token;
    foreign_allocation_token.allocation_generation =
        AllocationGeneration::from_raw_for_test(u64::MAX);
    let foreign_allocation_error = frame
        .release_injected_for_test(foreign_allocation_token)
        .expect_err("a foreign allocation generation must be rejected");

    for error in [
        stale_error,
        foreign_manager_error,
        foreign_frame_error,
        foreign_resource_error,
        foreign_allocation_error,
    ] {
        assert_eq!(error.code(), ErrorCode::InvalidInput);
    }
    assert_eq!(manager.observation_for_test(), before_rejections);

    // The production operation consumes this non-Clone lease. A second release
    // cannot be expressed without the explicitly test-only injected token path.
    frame.release(current).unwrap();
    let releases_after_production_release = manager.stats().releases;
    let replay_error = frame
        .release_injected_for_test(replay_after_release)
        .expect_err("an injected replay of a consumed lease must be rejected");
    assert_eq!(replay_error.code(), ErrorCode::InvalidInput);
    assert_eq!(manager.stats().releases, releases_after_production_release);
    let _ = frame.finish();
    assert_eq!(manager.observation_for_test().idle_count, 1);
}

#[test]
fn resource_trim_order_is_last_used_then_resource_identity() {
    let manager = ResourceManager::new(ResourceCacheBudget::new(4));
    let mut older_frame = manager.begin_frame().unwrap();
    let older = older_frame
        .acquire(ResourceCacheKey::VelloAtlas, 4)
        .unwrap();
    let older_identity = older.resource_identity();
    older_frame.release(older).unwrap();
    assert!(older_frame.finish().evicted_resources().is_empty());

    let mut current_frame = manager.begin_frame().unwrap();
    let first_equal_age = current_frame
        .acquire(modeled_resource_key_for_test(1), 4)
        .unwrap();
    let second_equal_age = current_frame
        .acquire(modeled_resource_key_for_test(2), 4)
        .unwrap();
    let first_equal_age_identity = first_equal_age.resource_identity();
    let second_equal_age_identity = second_equal_age.resource_identity();
    let active_cleanup = current_frame.trim_idle_for_test();
    let mut evicted_resources = active_cleanup.evicted_resources().to_vec();
    assert_eq!(evicted_resources, &[older_identity]);
    assert_eq!(manager.observation_for_test().leased_count, 2);
    assert_eq!(manager.observation_for_test().retained_bytes, 8);
    current_frame.release(second_equal_age).unwrap();
    current_frame.release(first_equal_age).unwrap();
    let cleanup = current_frame.finish();
    evicted_resources.extend_from_slice(cleanup.evicted_resources());

    assert_eq!(
        evicted_resources,
        &[older_identity, first_equal_age_identity],
        "idle trim order is not deterministic"
    );
    assert_eq!(manager.retained_count(), 1);
    assert_eq!(manager.observation_for_test().retained_bytes, 4);
    assert_eq!(second_equal_age_identity.get(), 3);
}

#[test]
fn resource_frame_scope_cleanup_covers_success_error_and_cancellation() {
    let success_manager = ResourceManager::default();
    let mut success_scope = success_manager.begin_frame().unwrap();
    let _success_lease = success_scope
        .acquire(modeled_resource_key_for_test(1), 4)
        .unwrap();
    let _ = success_scope.finish();
    assert_eq!(success_manager.observation_for_test().idle_count, 1);
    assert_eq!(success_manager.observation_for_test().leased_count, 0);

    let error_manager = ResourceManager::default();
    let error_result: Result<()> = (|| {
        let mut error_scope = error_manager.begin_frame()?;
        let _error_lease = error_scope.acquire(modeled_resource_key_for_test(2), 8)?;
        Err(Error::invalid_value(
            "resource frame fixture",
            "error",
            "must exercise scope-owned error cleanup",
        ))
    })();
    assert_eq!(error_result.unwrap_err().code(), ErrorCode::InvalidInput);
    assert_eq!(error_manager.observation_for_test().idle_count, 1);
    assert_eq!(error_manager.observation_for_test().leased_count, 0);

    let cancellation_manager = ResourceManager::default();
    let mut canceled_scope = cancellation_manager.begin_frame().unwrap();
    let _canceled_lease = canceled_scope
        .acquire(modeled_resource_key_for_test(3), 16)
        .unwrap();
    drop(canceled_scope);
    assert_eq!(cancellation_manager.observation_for_test().idle_count, 1);
    assert_eq!(cancellation_manager.observation_for_test().leased_count, 0);
}

#[test]
fn resource_byte_accounting_overflow_is_typed() {
    let manager = ResourceManager::default();
    let mut first_frame = manager.begin_frame().unwrap();
    let _maximum = first_frame
        .acquire(ResourceCacheKey::VelloAtlas, u64::MAX)
        .unwrap();
    let mut second_frame = manager.begin_frame().unwrap();

    let error = second_frame
        .acquire(modeled_resource_key_for_test(4), 1)
        .expect_err("retained resource byte overflow must be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("retained resource byte length")
    );
    assert_eq!(manager.observation_for_test().retained_bytes, u64::MAX);
    assert_eq!(manager.observation_for_test().leased_count, 1);
}

#[test]
fn discard_accounting_mismatch_faults_resource_manager_without_clamping() {
    let manager = ResourceManager::default();
    let mut unrelated_frame = manager.begin_frame().unwrap();
    let unrelated_lease = unrelated_frame
        .acquire(modeled_resource_key_for_test(41), 8)
        .unwrap();
    let unrelated_identity = unrelated_lease.resource_identity();
    let mut discarded_frame = manager.begin_frame().unwrap();
    let discarded_lease = discarded_frame
        .acquire(modeled_resource_key_for_test(42), 16)
        .unwrap();
    let discarded_identity = discarded_lease.resource_identity();
    discarded_frame.discard_on_drop();
    manager.inject_retained_byte_mismatch_before_discard_for_test();

    let drop_result = catch_unwind(AssertUnwindSafe(|| drop(discarded_frame)));
    assert!(drop_result.is_ok(), "discard-on-drop must not panic");

    let after_discard = manager.observation_for_test();
    assert_eq!(after_discard.retained_bytes, 7);
    assert_eq!(after_discard.accounted_entry_bytes, Some(8));
    let expected_fault = ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes: 7,
        registered_entry_bytes: 8,
    };
    assert_eq!(
        after_discard.accounting_fault_for_test(),
        Some(expected_fault)
    );
    assert_eq!(after_discard.active_frame_count, 1);
    assert_eq!(after_discard.leased_count, 1);
    assert!(
        after_discard
            .entry_identities_for_test()
            .contains(&unrelated_identity),
        "discard removed a resource leased by an unrelated active frame"
    );
    assert!(
        !after_discard
            .entry_identities_for_test()
            .contains(&discarded_identity),
        "the discarded resource remained registered for reuse"
    );

    let begin_error = match manager.begin_frame() {
        Ok(_) => panic!("discard accepted a retained-byte mismatch by saturation/clamping"),
        Err(error) => error,
    };
    let preflight_error = manager
        .preflight_graph_acquisitions(&[])
        .expect_err("faulted resource accounting must block graph preflight");
    let acquire_error = unrelated_frame
        .acquire(modeled_resource_key_for_test(45), 4)
        .expect_err("faulted resource accounting must block acquisition and reuse");
    for error in [&begin_error, &preflight_error, &acquire_error] {
        assert_eq!(error.code(), ErrorCode::RenderFailed);
        assert_eq!(
            error.message(),
            "resource manager is unavailable after a retained-byte accounting invariant failure"
        );
    }
    assert_eq!(manager.observation_for_test(), after_discard);

    unrelated_frame.release(unrelated_lease).unwrap();
    assert_eq!(manager.observation_for_test().resolved_lease_count, 1);
    let finish_result = catch_unwind(AssertUnwindSafe(|| unrelated_frame.finish()));
    assert!(
        finish_result.is_ok(),
        "an unrelated active frame must remain safely resolvable after the fault"
    );
    assert_eq!(
        finish_result.unwrap().retention(),
        ResourceRetentionOutcome::AccountingFault {
            fault: expected_fault,
        }
    );
    let after_unrelated_finish = manager.observation_for_test();
    assert_eq!(after_unrelated_finish.active_frame_count, 0);
    assert_eq!(after_unrelated_finish.resolved_lease_count, 0);
    assert_eq!(after_unrelated_finish.leased_count, 0);
    assert_eq!(after_unrelated_finish.idle_count, 1);
    assert_eq!(
        after_unrelated_finish.accounting_fault_for_test(),
        Some(expected_fault)
    );
    assert!(
        after_unrelated_finish
            .entry_identities_for_test()
            .contains(&unrelated_identity),
        "resolving the unrelated frame must not clear its resource to recover accounting"
    );
    assert!(manager.begin_frame().is_err());
}

#[test]
fn resource_manager_observation_reports_checked_entry_total_overflow() {
    let manager = ResourceManager::default();
    let mut frame = manager.begin_frame().unwrap();
    let first = frame.acquire(modeled_resource_key_for_test(43), 1).unwrap();
    let second = frame.acquire(modeled_resource_key_for_test(44), 1).unwrap();
    let originals = manager.inject_registered_entry_total_overflow_for_test(
        first.resource_identity(),
        second.resource_identity(),
    );

    let accounted_entry_bytes = manager.observation_for_test().accounted_entry_bytes;
    manager.restore_registered_entry_byte_lengths_for_test(originals);
    frame.release(first).unwrap();
    frame.release(second).unwrap();
    let _ = frame.finish();

    assert_eq!(
        accounted_entry_bytes, None,
        "resource observation saturated an overflowing registered-entry total"
    );
}

#[test]
fn discard_records_underflow_and_surviving_total_overflow_without_panicking() {
    let underflow_manager = ResourceManager::default();
    let mut underflow_frame = underflow_manager.begin_frame().unwrap();
    let underflow_lease = underflow_frame
        .acquire(modeled_resource_key_for_test(46), 8)
        .unwrap();
    let underflow_identity = underflow_lease.resource_identity();
    underflow_frame.discard_on_drop();
    underflow_manager.inject_retained_byte_underflow_before_discard_for_test();

    let underflow_drop = catch_unwind(AssertUnwindSafe(|| drop(underflow_frame)));
    assert!(underflow_drop.is_ok(), "discard underflow must not panic");
    let underflow_observation = underflow_manager.observation_for_test();
    assert_eq!(
        underflow_observation.accounting_fault_for_test(),
        Some(ResourceAccountingFault::RetainedByteUnderflow {
            retained_bytes: 0,
            discarded_entry_bytes: 8,
        })
    );
    assert!(
        !underflow_observation
            .entry_identities_for_test()
            .contains(&underflow_identity)
    );
    assert_eq!(underflow_observation.accounted_entry_bytes, Some(0));
    assert!(underflow_manager.begin_frame().is_err());

    let overflow_manager = ResourceManager::default();
    let mut survivor_frame = overflow_manager.begin_frame().unwrap();
    let first_survivor = survivor_frame
        .acquire(modeled_resource_key_for_test(47), 1)
        .unwrap();
    let second_survivor = survivor_frame
        .acquire(modeled_resource_key_for_test(48), 1)
        .unwrap();
    let mut overflow_discard_frame = overflow_manager.begin_frame().unwrap();
    let overflow_discarded = overflow_discard_frame
        .acquire(modeled_resource_key_for_test(49), 1)
        .unwrap();
    let overflow_discarded_identity = overflow_discarded.resource_identity();
    let _originals = overflow_manager.inject_registered_entry_total_overflow_for_test(
        first_survivor.resource_identity(),
        second_survivor.resource_identity(),
    );
    overflow_discard_frame.discard_on_drop();

    let overflow_drop = catch_unwind(AssertUnwindSafe(|| drop(overflow_discard_frame)));
    assert!(
        overflow_drop.is_ok(),
        "surviving-entry total overflow during discard must not panic"
    );
    let overflow_observation = overflow_manager.observation_for_test();
    assert_eq!(
        overflow_observation.accounting_fault_for_test(),
        Some(ResourceAccountingFault::SurvivingEntryByteTotalOverflow)
    );
    assert_eq!(overflow_observation.accounted_entry_bytes, None);
    assert!(
        !overflow_observation
            .entry_identities_for_test()
            .contains(&overflow_discarded_identity)
    );
    assert!(
        overflow_observation
            .entry_identities_for_test()
            .contains(&first_survivor.resource_identity())
    );
    assert!(
        overflow_observation
            .entry_identities_for_test()
            .contains(&second_survivor.resource_identity())
    );
    assert!(overflow_manager.begin_frame().is_err());

    survivor_frame.release(first_survivor).unwrap();
    survivor_frame.release(second_survivor).unwrap();
    let survivor_finish = catch_unwind(AssertUnwindSafe(|| survivor_frame.finish()));
    assert!(survivor_finish.is_ok());
}

#[test]
fn per_lease_discard_after_accounting_fault_removes_exact_lease_without_panicking() {
    let manager = ResourceManager::default();
    let mut unrelated_frame = manager.begin_frame().unwrap();
    let unrelated_lease = unrelated_frame
        .acquire(modeled_resource_key_for_test(50), 8)
        .unwrap();
    let unrelated_identity = unrelated_lease.resource_identity();
    let mut faulting_frame = manager.begin_frame().unwrap();
    let _faulting_lease = faulting_frame
        .acquire(modeled_resource_key_for_test(51), 16)
        .unwrap();
    faulting_frame.discard_on_drop();
    manager.inject_retained_byte_mismatch_before_discard_for_test();
    drop(faulting_frame);

    let expected_fault = ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes: 7,
        registered_entry_bytes: 8,
    };
    assert_eq!(
        manager.observation_for_test().accounting_fault_for_test(),
        Some(expected_fault)
    );

    let discard_result = catch_unwind(AssertUnwindSafe(|| {
        unrelated_frame.discard(unrelated_lease)
    }));
    assert!(
        discard_result.is_ok(),
        "per-lease cleanup after an accounting fault must not panic"
    );
    assert!(
        discard_result.unwrap().is_ok(),
        "an already-recorded accounting fault must allow exact lease cleanup"
    );

    let after_discard = manager.observation_for_test();
    assert_eq!(after_discard.retained_bytes, 7);
    assert_eq!(after_discard.accounted_entry_bytes, Some(0));
    assert_eq!(
        after_discard.accounting_fault_for_test(),
        Some(expected_fault),
        "per-lease cleanup replaced the first accounting diagnostic"
    );
    assert_eq!(after_discard.active_frame_count, 1);
    assert_eq!(after_discard.leased_count, 0);
    assert!(
        !after_discard
            .entry_identities_for_test()
            .contains(&unrelated_identity),
        "per-lease cleanup retained its exact uncertain lease"
    );

    let begin_error = match manager.begin_frame() {
        Ok(_) => panic!("faulted per-lease cleanup reenabled frame acquisition"),
        Err(error) => error,
    };
    let preflight_error = manager
        .preflight_graph_acquisitions(&[])
        .expect_err("faulted per-lease cleanup reenabled graph preflight");
    let acquire_error = unrelated_frame
        .acquire(modeled_resource_key_for_test(52), 4)
        .expect_err("faulted per-lease cleanup reenabled resource acquisition");
    for error in [&begin_error, &preflight_error, &acquire_error] {
        assert_eq!(error.code(), ErrorCode::RenderFailed);
        assert_eq!(
            error.message(),
            "resource manager is unavailable after a retained-byte accounting invariant failure"
        );
    }

    let finish_result = catch_unwind(AssertUnwindSafe(|| unrelated_frame.finish()));
    assert!(
        finish_result.is_ok(),
        "the active frame must remain resolvable after fault-aware per-lease cleanup"
    );
    assert_eq!(
        finish_result.unwrap().retention(),
        ResourceRetentionOutcome::AccountingFault {
            fault: expected_fault,
        }
    );
    let after_finish = manager.observation_for_test();
    assert_eq!(after_finish.active_frame_count, 0);
    assert_eq!(after_finish.resolved_lease_count, 0);
    assert_eq!(
        after_finish.accounting_fault_for_test(),
        Some(expected_fault)
    );
    assert!(manager.begin_frame().is_err());
}

#[test]
fn per_lease_discard_detects_accounting_fault_and_returns_bounded_error() {
    assert_per_lease_mismatch_fault();
    assert_per_lease_underflow_fault();
    assert_per_lease_overflow_fault();
}

fn assert_bounded_resource_accounting_fault(error: &Error) {
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(
        error.message(),
        "resource manager is unavailable after a retained-byte accounting invariant failure"
    );
}

fn assert_per_lease_mismatch_fault() {
    let mismatch_manager = ResourceManager::default();
    let mut mismatch_frame = mismatch_manager.begin_frame().unwrap();
    let mismatch_survivor = mismatch_frame
        .acquire(modeled_resource_key_for_test(53), 8)
        .unwrap();
    let mismatch_survivor_identity = mismatch_survivor.resource_identity();
    let mismatch_discarded = mismatch_frame
        .acquire(modeled_resource_key_for_test(54), 16)
        .unwrap();
    let mismatch_discarded_identity = mismatch_discarded.resource_identity();
    mismatch_manager.inject_retained_byte_mismatch_before_discard_for_test();

    let mismatch_error = mismatch_frame
        .discard(mismatch_discarded)
        .expect_err("per-lease discard silently accepted a retained-byte mismatch");
    assert_bounded_resource_accounting_fault(&mismatch_error);
    let mismatch_fault = ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes: 7,
        registered_entry_bytes: 8,
    };
    let mismatch_observation = mismatch_manager.observation_for_test();
    assert_eq!(mismatch_observation.retained_bytes, 7);
    assert_eq!(mismatch_observation.accounted_entry_bytes, Some(8));
    assert_eq!(
        mismatch_observation.accounting_fault_for_test(),
        Some(mismatch_fault)
    );
    assert!(
        !mismatch_observation
            .entry_identities_for_test()
            .contains(&mismatch_discarded_identity)
    );
    assert!(
        mismatch_observation
            .entry_identities_for_test()
            .contains(&mismatch_survivor_identity),
        "per-lease mismatch handling removed a surviving lease"
    );
    assert!(mismatch_manager.begin_frame().is_err());
    assert!(mismatch_manager.preflight_graph_acquisitions(&[]).is_err());
    assert!(
        mismatch_frame
            .acquire(modeled_resource_key_for_test(55), 4)
            .is_err()
    );
    mismatch_frame.release(mismatch_survivor).unwrap();
    let mismatch_finish = catch_unwind(AssertUnwindSafe(|| mismatch_frame.finish()));
    assert!(mismatch_finish.is_ok());
    assert_eq!(
        mismatch_finish.unwrap().retention(),
        ResourceRetentionOutcome::AccountingFault {
            fault: mismatch_fault,
        }
    );
}

fn assert_per_lease_underflow_fault() {
    let underflow_manager = ResourceManager::default();
    let mut underflow_frame = underflow_manager.begin_frame().unwrap();
    let underflow_discarded = underflow_frame
        .acquire(modeled_resource_key_for_test(56), 8)
        .unwrap();
    let underflow_discarded_identity = underflow_discarded.resource_identity();
    underflow_manager.inject_retained_byte_underflow_before_discard_for_test();

    let underflow_result = catch_unwind(AssertUnwindSafe(|| {
        underflow_frame.discard(underflow_discarded)
    }));
    assert!(
        underflow_result.is_ok(),
        "per-lease retained-byte underflow must return an error instead of panicking"
    );
    let underflow_error = underflow_result
        .unwrap()
        .expect_err("per-lease discard silently accepted retained-byte underflow");
    assert_bounded_resource_accounting_fault(&underflow_error);
    let underflow_fault = ResourceAccountingFault::RetainedByteUnderflow {
        retained_bytes: 0,
        discarded_entry_bytes: 8,
    };
    let underflow_observation = underflow_manager.observation_for_test();
    assert_eq!(underflow_observation.retained_bytes, 0);
    assert_eq!(underflow_observation.accounted_entry_bytes, Some(0));
    assert_eq!(
        underflow_observation.accounting_fault_for_test(),
        Some(underflow_fault)
    );
    assert!(
        !underflow_observation
            .entry_identities_for_test()
            .contains(&underflow_discarded_identity)
    );
    assert!(underflow_manager.begin_frame().is_err());
    assert!(underflow_manager.preflight_graph_acquisitions(&[]).is_err());
    assert!(
        underflow_frame
            .acquire(modeled_resource_key_for_test(57), 4)
            .is_err()
    );
    let underflow_finish = catch_unwind(AssertUnwindSafe(|| underflow_frame.finish()));
    assert!(underflow_finish.is_ok());
    assert_eq!(
        underflow_finish.unwrap().retention(),
        ResourceRetentionOutcome::AccountingFault {
            fault: underflow_fault,
        }
    );
}

fn assert_per_lease_overflow_fault() {
    let overflow_manager = ResourceManager::default();
    let mut overflow_frame = overflow_manager.begin_frame().unwrap();
    let first_overflow_survivor = overflow_frame
        .acquire(modeled_resource_key_for_test(58), 1)
        .unwrap();
    let second_overflow_survivor = overflow_frame
        .acquire(modeled_resource_key_for_test(59), 1)
        .unwrap();
    let overflow_discarded = overflow_frame
        .acquire(modeled_resource_key_for_test(60), 1)
        .unwrap();
    let overflow_discarded_identity = overflow_discarded.resource_identity();
    let _originals = overflow_manager.inject_registered_entry_total_overflow_for_test(
        first_overflow_survivor.resource_identity(),
        second_overflow_survivor.resource_identity(),
    );

    let overflow_error = overflow_frame
        .discard(overflow_discarded)
        .expect_err("per-lease discard silently accepted a surviving-entry total overflow");
    assert_bounded_resource_accounting_fault(&overflow_error);
    let overflow_fault = ResourceAccountingFault::SurvivingEntryByteTotalOverflow;
    let overflow_observation = overflow_manager.observation_for_test();
    assert_eq!(overflow_observation.retained_bytes, 2);
    assert_eq!(overflow_observation.accounted_entry_bytes, None);
    assert_eq!(
        overflow_observation.accounting_fault_for_test(),
        Some(overflow_fault)
    );
    assert!(
        !overflow_observation
            .entry_identities_for_test()
            .contains(&overflow_discarded_identity)
    );
    assert!(
        overflow_observation
            .entry_identities_for_test()
            .contains(&first_overflow_survivor.resource_identity())
    );
    assert!(
        overflow_observation
            .entry_identities_for_test()
            .contains(&second_overflow_survivor.resource_identity())
    );
    assert!(overflow_manager.begin_frame().is_err());
    assert!(overflow_manager.preflight_graph_acquisitions(&[]).is_err());
    assert!(
        overflow_frame
            .acquire(modeled_resource_key_for_test(61), 4)
            .is_err()
    );
    overflow_frame.release(first_overflow_survivor).unwrap();
    overflow_frame.release(second_overflow_survivor).unwrap();
    let overflow_finish = catch_unwind(AssertUnwindSafe(|| overflow_frame.finish()));
    assert!(overflow_finish.is_ok());
    assert_eq!(
        overflow_finish.unwrap().retention(),
        ResourceRetentionOutcome::AccountingFault {
            fault: overflow_fault,
        }
    );
}

#[test]
fn healthy_per_lease_discard_subtracts_exact_bytes_without_false_fault() {
    let manager = ResourceManager::default();
    let mut discarded_frame = manager.begin_frame().unwrap();
    let discarded = discarded_frame
        .acquire(modeled_resource_key_for_test(62), 16)
        .unwrap();
    let discarded_identity = discarded.resource_identity();
    let survivor = discarded_frame
        .acquire(modeled_resource_key_for_test(63), 8)
        .unwrap();
    let survivor_identity = survivor.resource_identity();
    let mut unrelated_frame = manager.begin_frame().unwrap();
    let unrelated = unrelated_frame
        .acquire(modeled_resource_key_for_test(64), 4)
        .unwrap();
    let unrelated_identity = unrelated.resource_identity();

    discarded_frame.discard(discarded).unwrap();

    let after_discard = manager.observation_for_test();
    assert_eq!(after_discard.retained_bytes, 12);
    assert_eq!(after_discard.accounted_entry_bytes, Some(12));
    assert_eq!(after_discard.accounting_fault_for_test(), None);
    assert_eq!(after_discard.active_frame_count, 2);
    assert_eq!(after_discard.leased_count, 2);
    assert!(
        !after_discard
            .entry_identities_for_test()
            .contains(&discarded_identity)
    );
    assert!(
        after_discard
            .entry_identities_for_test()
            .contains(&survivor_identity)
    );
    assert!(
        after_discard
            .entry_identities_for_test()
            .contains(&unrelated_identity),
        "healthy per-lease discard removed an unrelated frame's lease"
    );

    discarded_frame.release(survivor).unwrap();
    let _ = discarded_frame.finish();
    unrelated_frame.release(unrelated).unwrap();
    let _ = unrelated_frame.finish();

    let mut reuse_frame = manager.begin_frame().unwrap();
    let reused = reuse_frame
        .acquire(modeled_resource_key_for_test(63), 8)
        .unwrap();
    assert_eq!(reused.resource_identity(), survivor_identity);
    reuse_frame.release(reused).unwrap();
    let _ = reuse_frame.finish();
    let final_observation = manager.observation_for_test();
    assert_eq!(final_observation.retained_bytes, 12);
    assert_eq!(final_observation.accounted_entry_bytes, Some(12));
    assert_eq!(final_observation.accounting_fault_for_test(), None);
}

#[test]
fn replace_rejects_existing_accounting_fault_before_mutation() {
    let manager = ResourceManager::default();
    let mut frame = manager.begin_frame().unwrap();
    let lease = frame.acquire(modeled_resource_key_for_test(65), 8).unwrap();
    let resource_identity = lease.resource_identity();
    let expected_fault = manager.poison_retained_byte_accounting_for_test();
    let poisoned = manager.observation_for_test();

    let replacement = catch_unwind(AssertUnwindSafe(|| {
        frame.replace(lease, modeled_resource_key_for_test(66), 4)
    }));

    assert!(
        replacement.is_ok(),
        "replace after a recorded accounting fault must return a bounded error"
    );
    let error = replacement
        .unwrap()
        .expect_err("replace must reject an existing accounting fault");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(
        error.message(),
        "resource manager is unavailable after a retained-byte accounting invariant failure"
    );
    assert_eq!(manager.observation_for_test(), poisoned);
    assert_eq!(poisoned.accounting_fault_for_test(), Some(expected_fault));
    assert!(
        poisoned
            .entry_identities_for_test()
            .contains(&resource_identity),
        "rejected replace mutated or removed the original resource"
    );
    assert_eq!(poisoned.accounted_entry_bytes, Some(8));

    let cleanup = catch_unwind(AssertUnwindSafe(|| drop(frame)));
    assert!(cleanup.is_ok(), "faulted replace cleanup must not panic");
    assert_eq!(
        manager.observation_for_test().accounting_fault_for_test(),
        Some(expected_fault),
        "faulted replace cleanup replaced the first accounting diagnostic"
    );
}

#[test]
fn replace_records_mismatch_underflow_and_overflow_without_panicking() {
    assert_replace_mismatch_fault();
    assert_replace_underflow_fault();
    assert_replace_overflow_fault();
}

fn assert_replace_mismatch_fault() {
    let mismatch_manager = ResourceManager::default();
    let mut mismatch_frame = mismatch_manager.begin_frame().unwrap();
    let mismatch_survivor = mismatch_frame
        .acquire(modeled_resource_key_for_test(67), 8)
        .unwrap();
    let mismatch_replaced = mismatch_frame
        .acquire(modeled_resource_key_for_test(68), 16)
        .unwrap();
    let mismatch_replaced_identity = mismatch_replaced.resource_identity();
    mismatch_manager.inject_retained_byte_mismatch_before_discard_for_test();
    let mismatch_result = catch_unwind(AssertUnwindSafe(|| {
        mismatch_frame.replace(mismatch_replaced, modeled_resource_key_for_test(69), 4)
    }));
    assert!(mismatch_result.is_ok(), "replace mismatch must not panic");
    let mismatch_error = mismatch_result
        .unwrap()
        .expect_err("replace silently accepted mismatched retained accounting");
    assert_bounded_resource_accounting_fault(&mismatch_error);
    let mismatch_fault = ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes: 23,
        registered_entry_bytes: 24,
    };
    let mismatch_observation = mismatch_manager.observation_for_test();
    assert_eq!(
        mismatch_observation.accounting_fault_for_test(),
        Some(mismatch_fault)
    );
    assert_eq!(mismatch_observation.accounted_entry_bytes, Some(24));
    assert!(
        mismatch_observation
            .entry_identities_for_test()
            .contains(&mismatch_replaced_identity),
        "mismatched replace removed or replaced the original entry"
    );
    mismatch_frame.release(mismatch_survivor).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| drop(mismatch_frame))).is_ok());
}

fn assert_replace_underflow_fault() {
    let underflow_manager = ResourceManager::default();
    let mut underflow_frame = underflow_manager.begin_frame().unwrap();
    let underflow_replaced = underflow_frame
        .acquire(modeled_resource_key_for_test(70), 8)
        .unwrap();
    let underflow_identity = underflow_replaced.resource_identity();
    underflow_manager.inject_retained_byte_underflow_before_discard_for_test();
    let underflow_result = catch_unwind(AssertUnwindSafe(|| {
        underflow_frame.replace(underflow_replaced, modeled_resource_key_for_test(71), 4)
    }));
    assert!(underflow_result.is_ok(), "replace underflow must not panic");
    let underflow_error = underflow_result
        .unwrap()
        .expect_err("replace silently accepted retained-byte underflow");
    assert_bounded_resource_accounting_fault(&underflow_error);
    let underflow_fault = ResourceAccountingFault::RetainedByteUnderflow {
        retained_bytes: 0,
        discarded_entry_bytes: 8,
    };
    let underflow_observation = underflow_manager.observation_for_test();
    assert_eq!(
        underflow_observation.accounting_fault_for_test(),
        Some(underflow_fault)
    );
    assert_eq!(underflow_observation.accounted_entry_bytes, Some(8));
    assert!(
        underflow_observation
            .entry_identities_for_test()
            .contains(&underflow_identity)
    );
    assert!(catch_unwind(AssertUnwindSafe(|| drop(underflow_frame))).is_ok());
}

fn assert_replace_overflow_fault() {
    let overflow_manager = ResourceManager::default();
    let mut overflow_frame = overflow_manager.begin_frame().unwrap();
    let overflow_survivor = overflow_frame
        .acquire(modeled_resource_key_for_test(72), 1)
        .unwrap();
    let overflow_replaced = overflow_frame
        .acquire(modeled_resource_key_for_test(73), 1)
        .unwrap();
    let overflow_identity = overflow_replaced.resource_identity();
    let overflow_result = catch_unwind(AssertUnwindSafe(|| {
        overflow_frame.replace(
            overflow_replaced,
            modeled_resource_key_for_test(74),
            u64::MAX,
        )
    }));
    assert!(overflow_result.is_ok(), "replace overflow must not panic");
    let overflow_error = overflow_result
        .unwrap()
        .expect_err("replace overflow must fault accounting instead of mutating");
    assert_bounded_resource_accounting_fault(&overflow_error);
    let overflow_observation = overflow_manager.observation_for_test();
    assert_eq!(
        overflow_observation.accounting_fault_for_test(),
        Some(ResourceAccountingFault::SurvivingEntryByteTotalOverflow)
    );
    assert_eq!(overflow_observation.accounted_entry_bytes, Some(2));
    assert!(
        overflow_observation
            .entry_identities_for_test()
            .contains(&overflow_identity)
    );
    overflow_frame.release(overflow_survivor).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| drop(overflow_frame))).is_ok());
}

#[test]
fn extreme_effect_extent_reports_device_dimension_before_descriptor_byte_overflow() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("extreme effect extent coverage requires a selected host adapter");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("extreme effect extent coverage requires a ready WGPU device");
    let device = ready.device_for_test();
    let requested = PhysicalSize::new(u32::MAX, u32::MAX);
    let maximum = 1;
    let capabilities = DeviceCapabilities::from_test_facts(true, true, maximum);
    let manager = ResourceManager::default();
    let mut frame = manager.begin_frame().unwrap();
    let observation_before = manager.observation_for_test();
    let stats_before = manager.stats();

    let error = frame
        .acquire_working_effect_texture_for_test(
            device,
            &capabilities,
            WorkingFormat::HighPrecision,
            requested,
            WorkingFormat::HighPrecision.required_usages(),
        )
        .expect_err("an extreme effect extent must be rejected by selected-device limits");

    assert_eq!(
        manager.observation_for_test(),
        observation_before,
        "extreme effect extent reached concrete GPU payload creation"
    );
    assert_eq!(manager.stats(), stats_before);
    assert_eq!(
        error.code(),
        ErrorCode::RuntimeCapabilityUnavailable,
        "extreme effect extent returned the wrong diagnostic before selected-device validation"
    );
    let expected = RuntimeCapabilityUnavailable::try_new(
        RuntimeOperation::EffectTextureAllocation,
        RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { requested, maximum },
    )
    .unwrap();
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(&expected)
    );
}

#[test]
fn retained_byte_overflow_preflights_all_concrete_payload_creation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("concrete resource preflight coverage requires a selected host adapter");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("concrete resource preflight coverage requires a ready WGPU device");
    let device = ready.device_for_test();
    let queue = ready.queue_for_test();
    let capabilities = DeviceCapabilities::from_device(ready.adapter_for_test(), device);
    let working_format = capabilities
        .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)
        .expect("the selected test device must support one effect working format");
    let effect_descriptor = EffectTextureDescriptor::try_working(
        working_format,
        PhysicalSize::new(1, 1),
        working_format.required_usages(),
    )
    .unwrap();
    let mask_buffer = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![0, 0, 0, 255]).unwrap();
    let mask_descriptor =
        ResolvedMaskUploadDescriptor::try_from_image(image_from_buffer(mask_buffer)).unwrap();
    let kernel_plan =
        GaussianKernelPlan::try_new(1.0, 1.0, 2.5, GaussianKernelSamplingForm::PairedLinear)
            .unwrap();
    let manager = ResourceManager::default();
    let mut retained_frame = manager.begin_frame().unwrap();
    let _maximum = retained_frame
        .acquire(ResourceCacheKey::VelloAtlas, u64::MAX)
        .unwrap();
    let mut concrete_frame = manager.begin_frame().unwrap();
    let observation_before = manager.observation_for_test();
    let stats_before = manager.stats();

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let errors = [
        concrete_frame
            .acquire_effect_texture(device, &capabilities, effect_descriptor)
            .expect_err("effect texture retained-byte overflow must be rejected"),
        concrete_frame
            .acquire_resolved_mask_upload(device, queue, &capabilities, &mask_descriptor)
            .expect_err("mask upload retained-byte overflow must be rejected"),
        concrete_frame
            .acquire_gaussian_kernel_buffer(device, &kernel_plan)
            .expect_err("Gaussian buffer retained-byte overflow must be rejected"),
    ];
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "concrete retained-byte overflow probe emitted a WGPU validation error"
    );

    for error in errors {
        assert_eq!(error.code(), ErrorCode::InvalidInput);
        assert_eq!(
            error.invalid_value_diagnostic().map(InvalidValue::field),
            Some("retained resource byte length")
        );
    }
    assert_eq!(manager.stats(), stats_before);
    assert_eq!(
        manager.observation_for_test(),
        observation_before,
        "retained-byte overflow reached concrete GPU payload creation or upload"
    );
}

#[test]
fn resource_role_keys_keep_allocation_namespaces_distinct() {
    let capture = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(2, 2),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    let coverage = EffectTextureDescriptor::try_coverage(
        PhysicalSize::new(2, 2),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    let working = EffectTextureDescriptor::try_working(
        WorkingFormat::ReducedPrecision,
        PhysicalSize::new(2, 2),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    let mask = ResolvedMaskUploadKey::new(
        ImageId::new(1),
        PhysicalSize::new(2, 2),
        ImageQuality::Medium,
        Extend::Pad,
    );
    let kernel = GaussianKernelKey::from_exact_plan(
        1.0_f64.to_bits(),
        1.0_f64.to_bits(),
        2.5_f64.to_bits(),
        3,
        GaussianKernelSamplingForm::PairedLinear,
    );
    let keys = std::collections::HashSet::from([
        ResourceCacheKey::VelloAtlas,
        ResourceCacheKey::EffectTexture(capture.cache_key()),
        ResourceCacheKey::EffectTexture(working.cache_key()),
        ResourceCacheKey::EffectTexture(coverage.cache_key()),
        ResourceCacheKey::ResolvedMaskUpload(mask),
        ResourceCacheKey::GaussianKernelBuffer(kernel),
        modeled_resource_key_for_test(5),
    ]);

    assert_eq!(keys.len(), 7);
}

#[test]
fn effect_texture_keys_separate_format_extent_usage_and_role() {
    assert_effect_texture_cache_keys_are_distinct();
    assert_effect_texture_allocation_reuse_and_rejection();
}

fn assert_effect_texture_cache_keys_are_distinct() {
    let baseline = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(8, 4),
        wgpu::TextureUsages::TEXTURE_BINDING,
    )
    .unwrap();
    let keys = std::collections::HashSet::from([
        ResourceCacheKey::EffectTexture(baseline.cache_key()),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_coverage(
                PhysicalSize::new(8, 4),
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .unwrap()
            .cache_key(),
        ),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_working(
                WorkingFormat::HighPrecision,
                PhysicalSize::new(8, 4),
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .unwrap()
            .cache_key(),
        ),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_working(
                WorkingFormat::ReducedPrecision,
                PhysicalSize::new(8, 4),
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .unwrap()
            .cache_key(),
        ),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_capture(
                PhysicalSize::new(9, 4),
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .unwrap()
            .cache_key(),
        ),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_capture(
                PhysicalSize::new(8, 5),
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .unwrap()
            .cache_key(),
        ),
        ResourceCacheKey::EffectTexture(
            EffectTextureDescriptor::try_capture(
                PhysicalSize::new(8, 4),
                wgpu::TextureUsages::COPY_DST,
            )
            .unwrap()
            .cache_key(),
        ),
    ]);

    assert_eq!(
        keys.len(),
        7,
        "effect textures can alias across semantic roles"
    );
}

fn assert_effect_texture_allocation_reuse_and_rejection() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("effect texture allocation coverage requires a selected host adapter");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("effect texture allocation coverage requires a ready WGPU device");
    let device = ready.device_for_test();
    let capabilities = DeviceCapabilities::from_device(ready.adapter_for_test(), device);
    let working_format = capabilities
        .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)
        .expect("the selected test device must support one effect working format");
    let descriptor = EffectTextureDescriptor::try_working(
        working_format,
        PhysicalSize::new(3, 2),
        working_format.required_usages(),
    )
    .unwrap();
    let manager = ResourceManager::new(ResourceCacheBudget::new(1_024 * 1_024));

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame
        .acquire_effect_texture(device, &capabilities, descriptor)
        .unwrap();
    let first_identity = first.resource_identity();
    let stale = first.token_for_test();
    assert!(first_frame.effect_texture(&first).is_ok());
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        descriptor.checked_byte_len().unwrap()
    );
    first_frame.release(first).unwrap();
    let first_cleanup = first_frame.finish();
    assert!(first_cleanup.evicted_resources().is_empty());
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "validated effect texture creation emitted a WGPU validation error"
    );

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let allocations_before_reuse = manager.stats().allocations;
    let mut second_frame = manager.begin_frame().unwrap();
    let reused = second_frame
        .acquire_effect_texture(device, &capabilities, descriptor)
        .unwrap();
    assert_eq!(reused.resource_identity(), first_identity);
    assert_eq!(manager.stats().allocations, allocations_before_reuse);
    assert_eq!(manager.stats().hits, 1);
    let stale_error = second_frame
        .release_injected_for_test(stale)
        .expect_err("an effect texture lease from the prior frame must be stale");
    assert_eq!(stale_error.code(), ErrorCode::InvalidInput);

    let coverage_descriptor = EffectTextureDescriptor::try_coverage(
        PhysicalSize::new(3, 2),
        wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST),
    )
    .unwrap();
    let coverage = second_frame
        .acquire_effect_texture(device, &capabilities, coverage_descriptor)
        .unwrap();
    assert_ne!(coverage.resource_identity(), first_identity);
    let capture_descriptor = EffectTextureDescriptor::try_capture(
        PhysicalSize::new(3, 2),
        wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST),
    )
    .unwrap();
    let capture = second_frame
        .acquire_effect_texture(device, &capabilities, capture_descriptor)
        .unwrap();
    assert_ne!(capture.resource_identity(), first_identity);
    assert_ne!(capture.resource_identity(), coverage.resource_identity());
    assert_eq!(manager.stats().allocations, allocations_before_reuse + 2);
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        descriptor.checked_byte_len().unwrap()
            + coverage_descriptor.checked_byte_len().unwrap()
            + capture_descriptor.checked_byte_len().unwrap()
    );
    second_frame.release(capture).unwrap();
    second_frame.release(coverage).unwrap();
    second_frame.release(reused).unwrap();
    let _ = second_frame.finish();
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "exact-key effect texture reuse emitted a WGPU validation error"
    );

    assert_effect_texture_rejections_preserve_state(device, &manager, descriptor);
}

fn assert_effect_texture_rejections_preserve_state(
    device: &wgpu::Device,
    manager: &ResourceManager,
    descriptor: EffectTextureDescriptor,
) {
    let stats_before_rejection = manager.stats();
    let retained_before_rejection = manager.observation_for_test().retained_bytes;
    let mut rejected_frame = manager.begin_frame().unwrap();
    let over_limit = DeviceCapabilities::from_test_facts(true, true, 1);
    let dimension_error = rejected_frame
        .acquire_effect_texture(device, &over_limit, descriptor)
        .expect_err("an over-limit effect texture must fail before allocation");
    assert_eq!(
        dimension_error.code(),
        ErrorCode::RuntimeCapabilityUnavailable
    );
    assert_eq!(manager.stats(), stats_before_rejection);
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        retained_before_rejection
    );

    let unsupported = DeviceCapabilities::from_test_facts(false, false, u32::MAX);
    let format_error = rejected_frame
        .acquire_effect_texture(device, &unsupported, descriptor)
        .expect_err("an unsupported effect format must fail before allocation");
    assert_eq!(format_error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(manager.stats(), stats_before_rejection);
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        retained_before_rejection
    );
}

#[test]
fn resolved_mask_upload_keys_include_identity_dimensions_and_sampling() {
    assert_resolved_mask_cache_keys_are_distinct();
    assert_resolved_mask_upload_lifecycle();
}

fn assert_resolved_mask_cache_keys_are_distinct() {
    let image_id = ImageId::new(17);
    let keys = std::collections::HashSet::from([
        ResourceCacheKey::ResolvedMaskUpload(ResolvedMaskUploadKey::new(
            image_id,
            PhysicalSize::new(8, 4),
            ImageQuality::Medium,
            Extend::Pad,
        )),
        ResourceCacheKey::ResolvedMaskUpload(ResolvedMaskUploadKey::new(
            ImageId::new(18),
            PhysicalSize::new(8, 4),
            ImageQuality::Medium,
            Extend::Pad,
        )),
        ResourceCacheKey::ResolvedMaskUpload(ResolvedMaskUploadKey::new(
            image_id,
            PhysicalSize::new(9, 4),
            ImageQuality::Medium,
            Extend::Pad,
        )),
        ResourceCacheKey::ResolvedMaskUpload(ResolvedMaskUploadKey::new(
            image_id,
            PhysicalSize::new(8, 4),
            ImageQuality::High,
            Extend::Pad,
        )),
        ResourceCacheKey::ResolvedMaskUpload(ResolvedMaskUploadKey::new(
            image_id,
            PhysicalSize::new(8, 4),
            ImageQuality::Medium,
            Extend::Reflect,
        )),
    ]);

    assert_eq!(keys.len(), 5, "mask upload key omits semantic image facts");
}

fn assert_resolved_mask_upload_lifecycle() {
    let mask_bytes = vec![0, 0, 0, 255, 12, 34, 56, 128];
    let mask_buffer = ImageBuffer::try_new(PhysicalSize::new(2, 1), mask_bytes.clone()).unwrap();
    let mask_image = image_from_buffer(mask_buffer.clone());
    let descriptor = ResolvedMaskUploadDescriptor::try_from_image(mask_image.clone()).unwrap();
    assert_eq!(descriptor.physical_size(), mask_buffer.size());
    assert_eq!(descriptor.row_bytes(), 8);
    assert_eq!(descriptor.byte_len(), 8);
    assert_eq!(descriptor.bytes(), mask_bytes);
    assert_eq!(descriptor.cache_key().image_id(), descriptor.image().id());
    assert_eq!(
        descriptor.cache_key().physical_size(),
        PhysicalSize::new(2, 1)
    );
    assert_eq!(descriptor.cache_key().quality(), ImageQuality::Medium);
    assert_eq!(descriptor.cache_key().extend(), Extend::Pad);
    for invalid_len in [7, 9] {
        let error = descriptor
            .validate_upload_byte_len(invalid_len)
            .expect_err("short and long resolved-mask uploads must be rejected");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
    }

    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(mask_image, Rect::new(0.0, 0.0, 2.0, 1.0)).unwrap(),
        ),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
        },
    );
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let [command::RenderCommand::Layer { layer, .. }] = normalized.commands.as_slice() else {
        panic!("the resolved-mask fixture must normalize to one layer command");
    };
    let normalized_mask = layer
        .mask
        .as_ref()
        .expect("the normalized layer must retain its resolved alpha mask");
    assert_eq!(normalized_mask.image().bytes.as_ref(), mask_buffer.rgba());
    assert_eq!(normalized_mask.bounds(), Rect::new(0.0, 0.0, 2.0, 1.0));
    assert_eq!(normalized_mask.upload().bytes(), mask_bytes);
    assert_eq!(normalized_mask.upload().cache_key(), descriptor.cache_key());

    let sampled_descriptor = ResolvedMaskUploadDescriptor::try_from_image(
        descriptor
            .image()
            .clone()
            .quality(ImageQuality::High)
            .extend(Extend::Reflect),
    )
    .unwrap();
    assert_eq!(sampled_descriptor.image().id(), descriptor.image().id());
    assert_ne!(sampled_descriptor.cache_key(), descriptor.cache_key());

    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("resolved-mask upload coverage requires a selected host adapter");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("resolved-mask upload coverage requires a ready WGPU device");
    let device = ready.device_for_test();
    let queue = ready.queue_for_test();
    let capabilities = DeviceCapabilities::from_device(ready.adapter_for_test(), device);
    let manager = ResourceManager::new(ResourceCacheBudget::new(1_024 * 1_024));

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame
        .acquire_resolved_mask_upload(device, queue, &capabilities, &descriptor)
        .unwrap();
    let first_identity = first.resource_identity();
    assert!(first_frame.resolved_mask_texture(&first).is_ok());
    assert_eq!(manager.observation_for_test().retained_bytes, 8);
    first_frame.release(first).unwrap();
    let _ = first_frame.finish();
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "validated resolved-mask upload emitted a WGPU validation error"
    );

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let allocations_before_reuse = manager.stats().allocations;
    let mut second_frame = manager.begin_frame().unwrap();
    let reused = second_frame
        .acquire_resolved_mask_upload(device, queue, &capabilities, &descriptor)
        .unwrap();
    assert_eq!(reused.resource_identity(), first_identity);
    assert_eq!(manager.stats().allocations, allocations_before_reuse);
    let differently_sampled = second_frame
        .acquire_resolved_mask_upload(device, queue, &capabilities, &sampled_descriptor)
        .unwrap();
    assert_ne!(differently_sampled.resource_identity(), first_identity);
    assert_eq!(manager.stats().allocations, allocations_before_reuse + 1);
    assert_eq!(manager.observation_for_test().retained_bytes, 16);
    second_frame.release(differently_sampled).unwrap();
    second_frame.release(reused).unwrap();
    let _ = second_frame.finish();
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "exact-key resolved-mask reuse emitted a WGPU validation error"
    );

    assert_resolved_mask_over_limit_rejection(device, queue, &manager, &descriptor);
}

fn assert_resolved_mask_over_limit_rejection(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    manager: &ResourceManager,
    descriptor: &ResolvedMaskUploadDescriptor,
) {
    let stats_before_rejection = manager.stats();
    let mut rejected_frame = manager.begin_frame().unwrap();
    let over_limit = DeviceCapabilities::from_test_facts(true, true, 1);
    let error = rejected_frame
        .acquire_resolved_mask_upload(device, queue, &over_limit, descriptor)
        .expect_err("an over-limit mask upload must fail before texture allocation");
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(manager.stats(), stats_before_rejection);
}

#[test]
fn gaussian_kernel_buffer_keys_include_the_exact_plan() {
    let standard_deviation = 2.0_f64;
    let raster_scale = 1.5_f64;
    let support_multiple = 2.5_f64;
    assert_gaussian_kernel_cache_keys(standard_deviation, raster_scale, support_multiple);
    assert_gaussian_kernel_upload_lifecycle(standard_deviation, raster_scale, support_multiple);
}

fn assert_gaussian_kernel_cache_keys(
    standard_deviation: f64,
    raster_scale: f64,
    support_multiple: f64,
) {
    let keys = std::collections::HashSet::from([
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            raster_scale.to_bits(),
            support_multiple.to_bits(),
            8,
            GaussianKernelSamplingForm::PairedLinear,
        )),
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            f64::from_bits(standard_deviation.to_bits() + 1).to_bits(),
            raster_scale.to_bits(),
            support_multiple.to_bits(),
            8,
            GaussianKernelSamplingForm::PairedLinear,
        )),
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            f64::from_bits(raster_scale.to_bits() + 1).to_bits(),
            support_multiple.to_bits(),
            8,
            GaussianKernelSamplingForm::PairedLinear,
        )),
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            raster_scale.to_bits(),
            f64::from_bits(support_multiple.to_bits() + 1).to_bits(),
            8,
            GaussianKernelSamplingForm::PairedLinear,
        )),
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            raster_scale.to_bits(),
            support_multiple.to_bits(),
            9,
            GaussianKernelSamplingForm::PairedLinear,
        )),
        ResourceCacheKey::GaussianKernelBuffer(GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            raster_scale.to_bits(),
            support_multiple.to_bits(),
            8,
            GaussianKernelSamplingForm::FullNearest,
        )),
    ]);

    assert_eq!(
        keys.len(),
        6,
        "kernel buffer key omits exact planning facts"
    );
}

fn assert_gaussian_kernel_upload_lifecycle(
    standard_deviation: f64,
    raster_scale: f64,
    support_multiple: f64,
) {
    let paired = GaussianKernelPlan::try_new(
        standard_deviation,
        raster_scale,
        support_multiple,
        GaussianKernelSamplingForm::PairedLinear,
    )
    .unwrap();
    assert_eq!(
        paired.key(),
        GaussianKernelKey::from_exact_plan(
            standard_deviation.to_bits(),
            raster_scale.to_bits(),
            support_multiple.to_bits(),
            8,
            GaussianKernelSamplingForm::PairedLinear,
        )
    );
    assert_eq!(paired.upload_bytes().len() % 8, 0);
    assert_eq!(paired.byte_len(), paired.upload_bytes().len() as u64);
    for invalid_len in [
        paired.upload_bytes().len() - 1,
        paired.upload_bytes().len() + 1,
    ] {
        let error = paired
            .validate_upload_byte_len(invalid_len)
            .expect_err("short and long Gaussian uploads must be rejected");
        assert_eq!(error.code(), ErrorCode::InvalidInput);
    }
    let weight_sum = paired
        .upload_bytes()
        .chunks_exact(8)
        .map(|sample| f32::from_le_bytes(sample[4..8].try_into().unwrap()))
        .sum::<f32>();
    assert!((weight_sum - 1.0).abs() <= 1.0e-6);

    let full = GaussianKernelPlan::try_new(
        standard_deviation,
        raster_scale,
        support_multiple,
        GaussianKernelSamplingForm::FullNearest,
    )
    .unwrap();
    assert_ne!(paired.key(), full.key());
    assert_ne!(paired.upload_bytes(), full.upload_bytes());

    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("Gaussian kernel allocation coverage requires a selected host adapter");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("Gaussian kernel allocation coverage requires a ready WGPU device");
    let device = ready.device_for_test();
    let manager = ResourceManager::new(ResourceCacheBudget::new(1_024 * 1_024));

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame
        .acquire_gaussian_kernel_buffer(device, &paired)
        .unwrap();
    let first_identity = first.resource_identity();
    assert_eq!(
        first_frame.gaussian_kernel_buffer(&first).unwrap().usage(),
        wgpu::BufferUsages::STORAGE,
        "immutable kernel payloads must not retain a write usage"
    );
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        paired.byte_len()
    );
    first_frame.release(first).unwrap();
    let _ = first_frame.finish();
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "validated Gaussian kernel upload emitted a WGPU validation error"
    );

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let allocations_before_reuse = manager.stats().allocations;
    let mut second_frame = manager.begin_frame().unwrap();
    let reused = second_frame
        .acquire_gaussian_kernel_buffer(device, &paired)
        .unwrap();
    assert_eq!(reused.resource_identity(), first_identity);
    assert_eq!(manager.stats().allocations, allocations_before_reuse);
    let full_lease = second_frame
        .acquire_gaussian_kernel_buffer(device, &full)
        .unwrap();
    assert_ne!(full_lease.resource_identity(), first_identity);
    assert_eq!(manager.stats().allocations, allocations_before_reuse + 1);
    assert_eq!(
        manager.observation_for_test().retained_bytes,
        paired.byte_len() + full.byte_len()
    );
    second_frame.release(full_lease).unwrap();
    second_frame.release(reused).unwrap();
    let _ = second_frame.finish();
    assert!(
        pollster::block_on(error_scope.pop()).is_none(),
        "immutable Gaussian kernel reuse emitted a WGPU validation error"
    );
}

#[test]
fn texture_cache_release_and_eviction_accounting_is_deterministic() {
    let (key, byte_len) = modeled_effect_texture_for_test(PhysicalSize::new(2, 2));
    let manager = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut frame = manager.begin_frame().unwrap();
    let lease = frame.acquire(key, byte_len).unwrap();
    frame.release(lease).unwrap();
    let cleanup = frame.finish();

    assert_eq!(cleanup.evicted_resources().len(), 1);
    assert_eq!(manager.live_count(), 0);
    assert_eq!(manager.retained_count(), 0);
    assert_eq!(manager.stats().releases, 1);
    assert_eq!(manager.stats().evictions, 1);
}

#[test]
fn texture_cache_rejects_stale_handle_after_reuse() {
    let (key, byte_len) = modeled_effect_texture_for_test(PhysicalSize::new(3, 3));
    let manager = ResourceManager::default();
    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame.acquire(key, byte_len).unwrap();
    let stale = first.token_for_test();
    first_frame.release(first).unwrap();
    let _ = first_frame.finish();
    let mut second_frame = manager.begin_frame().unwrap();
    let current = second_frame.acquire(key, byte_len).unwrap();
    let error = second_frame
        .release_injected_for_test(stale)
        .expect_err("stale frame leases must not release a new lease");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(manager.live_count(), 1);
    assert_eq!(manager.stats().releases, 1);
    second_frame.release(current).unwrap();
    assert_eq!(manager.stats().releases, 2);
}

#[test]
fn texture_cache_rejects_foreign_release_for_matching_descriptor() {
    let (key, byte_len) = modeled_effect_texture_for_test(PhysicalSize::new(5, 5));
    let first_manager = ResourceManager::default();
    let second_manager = ResourceManager::default();
    let mut first_frame = first_manager.begin_frame().unwrap();
    let mut second_frame = second_manager.begin_frame().unwrap();

    let foreign = first_frame.acquire(key, byte_len).unwrap();
    let local = second_frame.acquire(key, byte_len).unwrap();
    let error = second_frame
        .release(foreign)
        .expect_err("foreign handles must not release matching local entries");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(second_manager.live_count(), 1);
    assert_eq!(second_manager.stats().releases, 0);
    second_frame.release(local).unwrap();
    assert_eq!(second_manager.stats().releases, 1);
}

#[test]
fn texture_descriptors_reject_zero_size_and_overflow() {
    let zero_width = TextureDescriptor::try_new(
        PhysicalSize::new(0, 1),
        Format::Rgba8,
        TextureUsageIntent::ReadbackReference,
    )
    .expect_err("zero-width textures should be rejected");
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);

    let overflow = TextureDescriptor::try_new(
        PhysicalSize::new(u32::MAX, u32::MAX),
        Format::Rgba8,
        TextureUsageIntent::ReadbackReference,
    )
    .expect_err("overflow-sized textures should be rejected");
    assert_eq!(overflow.code(), ErrorCode::InvalidInput);
}

#[test]
fn headless_texture_descriptor_uses_allocation_size_without_surface_rewrite() {
    let zero_surface_descriptor =
        headless_texture_descriptor(PhysicalSize::new(0, 0), Format::Rgba8).unwrap();
    let nonzero_surface_descriptor =
        headless_texture_descriptor(PhysicalSize::new(12, 6), Format::Rgba8).unwrap();

    assert_eq!(
        zero_surface_descriptor.physical_size(),
        PhysicalSize::new(1, 1)
    );
    assert_eq!(
        zero_surface_descriptor.intent(),
        TextureUsageIntent::ReadbackReference
    );
    assert_eq!(
        nonzero_surface_descriptor.physical_size(),
        PhysicalSize::new(12, 6)
    );
}

#[test]
fn texture_lifecycle_accounting_is_separate_from_image_cache_stats() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 1.0, 1.0), Paint::image(image.clone()))
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Paint::image(image));
    let image_stats = scene.stats();

    let (key, byte_len) = modeled_effect_texture_for_test(PhysicalSize::new(4, 4));
    let manager = ResourceManager::default();
    let mut first_frame = manager.begin_frame().unwrap();
    let first = first_frame.acquire(key, byte_len).unwrap();
    first_frame.release(first).unwrap();
    let _ = first_frame.finish();
    let mut second_frame = manager.begin_frame().unwrap();
    let _second = second_frame.acquire(key, byte_len).unwrap();

    assert_eq!(image_stats.images, 2);
    assert_eq!(image_stats.cache_misses, 1);
    assert_eq!(image_stats.cache_hits, 1);
    assert_eq!(image_stats.uploaded_bytes, 4);
    assert_eq!(manager.stats().misses, 1);
    assert_eq!(manager.stats().hits, 1);
}

#[test]
fn shader_clear_fill_pass_encodes_when_gpu_context_is_available() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "real GPU clear/fill coverage requires a host adapter"
    );
    let output = pollster::block_on(renderer.scoped_clear_fill_probe_for_test())
        .expect("available GPU clear/fill work must resolve through its transaction");
    let [red, green, blue, alpha] = pixel_rgba(&output, 0, 0);
    assert!(
        (60..=68).contains(&red),
        "red channel should be cleared: {red}"
    );
    assert!(
        (124..=132).contains(&green),
        "green channel should be cleared: {green}"
    );
    assert!(
        (187..=195).contains(&blue),
        "blue channel should be cleared: {blue}"
    );
    assert_eq!(alpha, 255);
}

#[test]
fn non_readback_gpu_submissions_are_owned_by_gpu_operation_transactions() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("real GPU transaction submission coverage requires a host adapter");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let generation = signal.next_test_generation().unwrap();
    let transaction = super::gpu_transaction::GpuOperationTransaction::begin(
        &device,
        Arc::clone(&signal),
        generation,
        GpuOperationStage::Render,
    );
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit generic transaction observation"),
        })
        .finish();
    let submission = pollster::block_on(submit_command_buffer_observed_for_test(
        transaction,
        &queue,
        command_buffer,
        RuntimeOperation::SurfaceRendering,
    ))
    .expect("the explicit real transaction must submit and resolve its scopes");
    assert_eq!(
        submission.queue_submission_count_for_test(),
        1,
        "the real clear/fill command buffer must submit through a GPU operation transaction"
    );
    assert_eq!(
        submission.transaction_generation_for_test(),
        submission.active_generation_for_test(),
        "the transaction generation must remain active at the real queue submission"
    );
    assert!(
        submission.scopes_resolved_for_test(),
        "the transaction must resolve its nested WGPU scopes before returning"
    );
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn canceled_generic_submission_after_real_submit_clears_ownership_without_public_result() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("generic transaction cancellation coverage requires a host adapter");
    let stats_before = renderer.stats();
    let uploaded_images_before = renderer.uploaded_images_for_test();
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let generation = signal.next_test_generation().unwrap();
    let transaction = super::gpu_transaction::GpuOperationTransaction::begin(
        &device,
        Arc::clone(&signal),
        generation,
        GpuOperationStage::Render,
    );
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit generic transaction cancellation"),
        })
        .finish();

    {
        let future = hold_command_buffer_after_submit_for_test(transaction, &queue, command_buffer);
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }

    assert_eq!(signal.active_generation_for_test(), None);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "dropping the pending generic submission must clear its active generation"
    );
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(renderer.uploaded_images_for_test(), uploaded_images_before);
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn offscreen_texture_allocation_uses_explicit_bounded_layer_descriptor() {
    let bounds = command::OffscreenBounds::try_new(Rect::new(2.0, 3.0, 10.0, 6.0)).unwrap();

    let descriptor = offscreen_local_scene_texture_descriptor(bounds, 2.0, Format::Rgba8).unwrap();

    assert_eq!(descriptor.physical_size(), PhysicalSize::new(20, 12));
    assert_eq!(descriptor.texture_format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(descriptor.role(), EffectTextureRole::Capture);
}

#[test]
fn offscreen_texture_rejects_missing_gpu_context_with_adapter_diagnostic() {
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 1.0, 1.0)).unwrap();
    let mut scene = VelloScene::default();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 1.0, 1.0),
    );

    let error = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        None,
        Options::default(),
        &scene,
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default()),
    ))
    .expect_err("contract-only offscreen render should report missing GPU context");

    assert_runtime_adapter_unavailable(&error, RuntimeOperation::SurfaceRendering);
    assert!(error.message().contains("offscreen Vello local scene"));
}

#[test]
fn offscreen_local_vello_scene_renders_to_texture_when_gpu_context_is_available() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let bounds = command::OffscreenBounds::try_new(Rect::new(12.0, 8.0, 2.0, 2.0)).unwrap();
    let mut scene = VelloScene::default();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 2.0, 2.0),
    );
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let options = renderer.options();
    let context = renderer
        .default_offscreen_render_context()
        .expect("offscreen texture rendering requires a host adapter");

    let output = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .unwrap();
    assert_eq!(output.target().bounds(), bounds);
    assert_eq!(output.target().resource_id(), 1);
    assert_eq!(
        output.target().descriptor().physical_size(),
        PhysicalSize::new(2, 2)
    );
    assert_eq!(output.timings().present_time, Duration::ZERO);
    let view_debug = format!("{:?}", output.view().unwrap());
    assert!(!view_debug.is_empty());

    let image = pollster::block_on(renderer.read_render_texture_for_test(
        output.texture().unwrap(),
        output.target().descriptor().physical_size(),
    ))
    .expect("offscreen texture readback requires the same host adapter");
    assert!(pixel_alpha(&image, 0, 0) > 0);

    output.release().unwrap();
    let resources = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("offscreen texture release must retain the ready device manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources.leased_count, 0);
    assert!(resources.idle_count >= 1);
}

#[test]
fn explicit_offscreen_release_reports_accounting_fault_while_drop_remains_nonpanicking() {
    let render_output = |renderer: &mut Renderer| {
        let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 2.0, 2.0)).unwrap();
        let mut scene = VelloScene::default();
        scene.fill(
            peniko::Fill::NonZero,
            kurbo::Affine::IDENTITY,
            peniko::Color::BLACK,
            None,
            &kurbo::Rect::new(0.0, 0.0, 2.0, 2.0),
        );
        let request = OffscreenLocalSceneRenderRequest::new(
            bounds,
            1.0,
            Format::Rgba8,
            Parameters::default(),
        );
        let options = renderer.options();
        let context = renderer
            .default_offscreen_render_context()
            .expect("offscreen accounting coverage requires a host adapter");
        pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
            Some(context),
            options,
            &scene,
            request,
        ))
        .expect("offscreen accounting coverage requires a rendered texture lease")
    };

    let mut explicit_renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("explicit offscreen release coverage requires a host adapter");
    let explicit = render_output(&mut explicit_renderer);
    let expected_fault = explicit.poison_retained_byte_accounting_for_test();
    let error = explicit
        .release()
        .expect_err("explicit offscreen release silently ignored terminal accounting cleanup");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(
        error.message(),
        "resource manager is unavailable after a retained-byte accounting invariant failure"
    );
    assert_eq!(
        explicit_renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the accounting fault must retain the ready device for diagnosis")
            .internal_resource_manager_observation_for_test()
            .accounting_fault_for_test(),
        Some(expected_fault)
    );

    let mut drop_renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("offscreen drop coverage requires a host adapter");
    let dropped = render_output(&mut drop_renderer);
    let drop_fault = dropped.poison_retained_byte_accounting_for_test();
    let drop_result = catch_unwind(AssertUnwindSafe(|| drop(dropped)));
    assert!(
        drop_result.is_ok(),
        "dropping a poisoned offscreen texture lease must remain best-effort and nonpanicking"
    );
    assert_eq!(
        drop_renderer
            .default_ready_device_state_borrow_for_test()
            .expect("offscreen drop must retain the ready device for diagnosis")
            .internal_resource_manager_observation_for_test()
            .accounting_fault_for_test(),
        Some(drop_fault),
        "offscreen drop replaced the first accounting diagnostic"
    );
}

#[test]
fn offscreen_local_scene_texture_descriptor_rejects_bgra8_for_vello_target() {
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 2.0, 2.0)).unwrap();
    let error = offscreen_local_scene_texture_descriptor(bounds, 1.0, Format::Bgra8)
        .expect_err("minimal offscreen Vello targets are Rgba8-only");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("offscreen Vello scene texture format")
    );
}

#[test]
fn offscreen_bgra8_render_request_rejects_without_cache_allocation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("Bgra8 offscreen validation coverage requires a real selected device");
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 2.0, 2.0)).unwrap();
    let scene = VelloScene::default();
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Bgra8, Parameters::default());
    let options = renderer.options();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("Bgra8 offscreen validation coverage requires real device resources")
        .internal_resource_manager_observation_for_test();
    let context = renderer
        .default_offscreen_render_context()
        .expect("Bgra8 offscreen validation coverage requires a real device context");

    let error = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .expect_err("Bgra8 should be rejected before GPU context allocation");
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("Bgra8 rejection must retain real device resources")
        .internal_resource_manager_observation_for_test();

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("offscreen Vello scene texture format")
    );
    assert_eq!(resources_after, resources_before);
}

#[test]
fn offscreen_nested_layer_opacity_stays_on_direct_vello_surface_path() {
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.75).unwrap(), |scene| {
        scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        });
    });
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[0]
    else {
        panic!("expected outer opacity layer");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[0] else {
        panic!("expected inner opacity layer");
    };

    assert_eq!(
        outer.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
    assert_eq!(
        inner.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
    assert!(!outer.pass_plan.requires_offscreen_texture());
    assert!(!inner.pass_plan.requires_offscreen_texture());

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    let alpha = pixel_alpha(&output, 0, 0);

    assert_eq!(stats.layers, 2);
    assert!(alpha > 0);
    assert!(alpha < 255);
}

#[test]
fn offscreen_reuses_resources_across_repeated_bounded_requests() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 3.0, 2.0)).unwrap();
    let mut scene = VelloScene::default();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 3.0, 2.0),
    );
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let options = renderer.options();
    let context = renderer
        .default_offscreen_render_context()
        .expect("offscreen texture reuse requires a host adapter");
    let first = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .unwrap();
    let first_resource_id = first.target().resource_id();
    let first_descriptor = first.target().descriptor();
    first.release().unwrap();

    let context = renderer.default_offscreen_render_context().unwrap();
    let second = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .unwrap();

    assert_eq!(second.target().descriptor(), first_descriptor);
    assert_eq!(second.target().resource_id(), first_resource_id);
    let resources = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("reused offscreen texture must remain leased from the ready device manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources.leased_count, 1);
    let image = pollster::block_on(renderer.read_render_texture_for_test(
        second.texture().unwrap(),
        second.target().descriptor().physical_size(),
    ))
    .expect("reused offscreen texture readback requires the same host adapter");
    assert_eq!(image.size(), PhysicalSize::new(3, 2));
    assert!(
        image.rgba().chunks_exact(4).all(|pixel| pixel[3] > 0),
        "the reused offscreen texture must contain rendered pixels"
    );
    second.release().unwrap();
    let resources = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("offscreen texture release must retain the ready device manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources.leased_count, 0);
}

#[test]
fn offscreen_no_allocation_when_layer_isolation_is_unnecessary() {
    let mut scene = Scene::new();
    scene.layer(Layer::new(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };
    let manager = ResourceManager::default();

    assert_eq!(layer.pass_plan.kind(), command::LayerPassKind::None);
    assert!(!layer.pass_plan.requires_offscreen_texture());
    assert_eq!(manager.stats().allocations, 0);
    assert_eq!(manager.live_count(), 0);
}

#[test]
fn direct_vello_output_matches_ordinary_scene_baseline() {
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(
            Rect::new(2.0, 0.0, 2.0, 2.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    assert!(
        normalized
            .commands
            .iter()
            .all(|command| { !matches!(command, command::RenderCommand::Layer { .. }) })
    );

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut first_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut second_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();

    let first_stats =
        pollster::block_on(renderer.render(&mut first_surface, &scene, Parameters::default()))
            .unwrap();
    let first_output = pollster::block_on(renderer.read_headless(&first_surface)).unwrap();
    let second_stats =
        pollster::block_on(renderer.render(&mut second_surface, &scene, Parameters::default()))
            .unwrap();
    let second_output = pollster::block_on(renderer.read_headless(&second_surface)).unwrap();

    assert_eq!(first_stats.layers, 0);
    assert_eq!(second_stats.layers, 0);
    assert_eq!(first_output.rgba(), second_output.rgba());
    assert!(pixel_rgba(&first_output, 0, 0)[0] > 200);
    assert!(pixel_rgba(&first_output, 3, 0)[1] > 200);
}

#[test]
fn effect_free_layers_keep_finite_bounds_without_offscreen_plan() {
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.fill(Rect::new(1.0, 2.0, 4.0, 3.0), Color::BLACK);
        scene.layer(
            Layer::new()
                .try_transform(Transform::translation(6.0, 0.0).unwrap())
                .unwrap()
                .blend(BlendMode::Screen),
            |scene| {
                scene.fill(Rect::new(0.0, 1.0, 2.0, 2.0), Color::BLACK);
            },
        );
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[0]
    else {
        panic!("expected outer direct Vello layer");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[1] else {
        panic!("expected inner direct Vello layer");
    };

    for layer in [outer, inner] {
        let bounds = layer
            .pass_plan
            .bounds()
            .expect("direct layer plans should carry explicit child bounds")
            .rect();
        assert_finite_positive_rect(bounds);
        assert_eq!(
            layer.pass_plan.kind(),
            command::LayerPassKind::DirectVelloLayer
        );
        assert!(!layer.pass_plan.requires_offscreen_texture());
    }
    assert_eq!(
        outer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(1.0, 1.0, 7.0, 4.0))
    );
}

#[test]
fn retained_capture_texture_lifecycle_is_deterministic_for_nested_layer_bounds() {
    let outer_bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap();
    let inner_bounds = command::OffscreenBounds::try_new(Rect::new(2.0, 1.0, 3.0, 2.0)).unwrap();
    let outer = offscreen_local_scene_texture_descriptor(outer_bounds, 1.0, Format::Rgba8).unwrap();
    let inner = offscreen_local_scene_texture_descriptor(inner_bounds, 1.0, Format::Rgba8).unwrap();
    let outer_key = ResourceCacheKey::EffectTexture(outer.cache_key());
    let inner_key = ResourceCacheKey::EffectTexture(inner.cache_key());
    let outer_byte_len = outer.checked_byte_len().unwrap();
    let inner_byte_len = inner.checked_byte_len().unwrap();
    let manager = ResourceManager::default();

    let mut first_frame = manager.begin_frame().unwrap();
    let outer_first = first_frame.acquire(outer_key, outer_byte_len).unwrap();
    let inner_first = first_frame.acquire(inner_key, inner_byte_len).unwrap();
    let outer_identity = outer_first.resource_identity();
    let inner_identity = inner_first.resource_identity();
    first_frame.release(inner_first).unwrap();
    first_frame.release(outer_first).unwrap();
    let _ = first_frame.finish();
    let mut second_frame = manager.begin_frame().unwrap();
    let outer_second = second_frame.acquire(outer_key, outer_byte_len).unwrap();
    let inner_second = second_frame.acquire(inner_key, inner_byte_len).unwrap();

    assert_eq!(outer_second.resource_identity(), outer_identity);
    assert_eq!(inner_second.resource_identity(), inner_identity);
    assert_eq!(manager.stats().allocations, 2);
    assert_eq!(manager.stats().misses, 2);
    assert_eq!(manager.stats().hits, 2);
    assert_eq!(manager.stats().releases, 2);
    assert_eq!(manager.live_count(), 2);
    assert_eq!(manager.retained_count(), 2);
}

#[test]
fn surface_tracks_size_and_scale() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    surface.resize(Size::new(20.0, 30.0), 2.0).unwrap();

    assert_eq!(surface.size(), Size::new(20.0, 30.0));
    assert_eq!(surface.scale(), 2.0);
}

#[test]
fn surface_state_reports_availability_without_bool_peeking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::try_new(1.0, 1.0).unwrap(), 1.0))
            .unwrap();

    assert_eq!(surface.state(), SurfaceState::Available);
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
}

#[test]
fn headless_backend_resource_state_tracks_readiness() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::try_new(2.0, 2.0).unwrap(), 1.0))
            .unwrap();

    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
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
    let idle = PresentedLifecycle::ResizePending {
        physical_size: PhysicalSize::new(20, 10),
        resizing: ResizeState::Idle,
    };
    let resizing = idle.with_resizing(ResizeState::Resizing);

    assert_eq!(
        resizing,
        PresentedLifecycle::ResizePending {
            physical_size: PhysicalSize::new(20, 10),
            resizing: ResizeState::Resizing,
        }
    );
    assert_eq!(
        resizing.with_resizing(ResizeState::Resizing),
        resizing,
        "repeating the resizing hint is idempotent"
    );
    assert_eq!(resizing.with_resizing(ResizeState::Idle), idle);
    assert_eq!(
        idle.with_resizing(ResizeState::Idle),
        idle,
        "repeating the idle hint is idempotent"
    );
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn presented_surface_lifecycle_recovers_from_zero_size_at_current_native_size() {
    let mut state = PresentedSurfaceState::new(PhysicalSize::new(0, 0), ResizeState::Resizing);
    state.resize_requested(
        Some(PhysicalSize::new(640, 480)),
        PhysicalSize::new(640, 480),
    );

    assert_eq!(
        state.lifecycle(),
        PresentedLifecycle::Ready {
            resizing: ResizeState::Resizing,
        }
    );
}

#[test]
fn headless_resize_keeps_target_when_physical_size_is_unchanged() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    surface.resize(Size::new(10.4, 10.4), 1.0).unwrap();

    assert_eq!(surface.size(), Size::new(10.4, 10.4));
    assert_eq!(surface.physical_size(), PhysicalSize::new(10, 10));
    assert!(matches!(
        &surface.backend,
        SurfaceBackend::Headless {
            resources: HeadlessResources::Pending,
            ..
        }
    ));
}

#[test]
fn create_surface_headless_preserves_surface_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let surface = pollster::block_on(renderer.create_surface(
        Attachment::Headless,
        SurfaceOptions {
            size: Size::new(10.0, 20.0),
            scale: 2.0,
            present_mode: PresentMode::Immediate,
            format: Format::Rgba8,
        },
    ))
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
    let error = match pollster::block_on(renderer.create_headless(Size::new(f64::NAN, 10.0), 1.0)) {
        Ok(_) => panic!("non-finite surface size should fail before physical conversion"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::InvalidInput);

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    let error = surface
        .resize(Size::new(1.0, 1.0), 0.0)
        .expect_err("invalid scale should fail before resize");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

fn color_filter_commands_for_shader_test() -> command::RenderCommands {
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.0).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(f64::from_bits(1)).unwrap()),
        FilterOp::grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(0.5).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::saturate(FilterAmount::try_new(f64::MAX).unwrap()),
        FilterOp::sepia(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(-2.0, 3.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 12.0, 10.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(-1.0, 4.0, 3.0, 2.0),
                Color::try_rgba(0.5, 0.25, 0.75, 0.5).unwrap(),
            );
        });
    scene.normalize(Capabilities::CURRENT).unwrap()
}

fn color_filter_frame_context_for_shader_test() -> super::frame::FrameContext {
    super::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap()
}

fn color_filter_expected_operation_bytes_for_test() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + 8 * 32);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&[0_u8; 12]);
    let mut push_record = |tag: u32, zero: u32, exponent: i32, payload: [f32; 4]| {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&zero.to_le_bytes());
        bytes.extend_from_slice(&exponent.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        for value in payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    };
    push_record(0, 1, 0, [0.0, 0.0, 0.0, 0.0]);
    push_record(1, 0, -1073, [0.5, 0.0, 0.0, 0.0]);
    push_record(2, 0, 0, [0.25, 0.0, 0.0, 0.0]);
    let reduced_angle = std::f64::consts::FRAC_PI_2.rem_euclid(std::f64::consts::TAU) as f32;
    let (sine, cosine) = reduced_angle.sin_cos();
    push_record(3, 0, 0, [sine, cosine, 0.0, 0.0]);
    push_record(4, 0, 0, [0.5, 0.0, 0.0, 0.0]);
    push_record(5, 0, 0, [0.75, 0.0, 0.0, 0.0]);
    push_record(6, 0, 1025, [0.5, 0.0, 0.0, 0.0]);
    push_record(7, 0, 0, [1.0, 0.0, 0.0, 0.0]);
    bytes
}

#[test]
fn color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries() {
    let observed = super::pass::color_filter_operation_bytes_observation_for_test(
        color_filter_commands_for_shader_test(),
        color_filter_frame_context_for_shader_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .unwrap_or_panic_for_test(
        "the color-filter operation-byte fixture must lower without allocation",
    );

    assert!(
        observed.bytes == color_filter_expected_operation_bytes_for_test()
            && observed.preserves_one_clamp_per_record,
        "color operation bytes lost an authored finite scalar or clamp"
    );
}

#[test]
fn color_filter_operation_buffer_limits_return_exact_invalid_input_before_allocation() {
    let observed = super::pass::color_filter_operation_buffer_limit_observation_for_test();

    assert!(
        observed.count_overflow_is_exact
            && observed.max_buffer_size_is_exact
            && observed.max_storage_binding_size_is_exact
            && observed.equality_at_both_limits_is_accepted
            && observed.rejects_before_any_allocation_or_cache_action,
        "an oversized color-filter buffer lacks its exact pre-allocation diagnostic"
    );
}

#[test]
fn color_filter_cache_realizes_checked_high_and_reduced_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires backend selection",
        )
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires a host adapter",
        );
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires a ready device",
        );
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        super::pass::color_filter_cache_realization_observation_for_test(
            ready.device_for_test(),
            color_filter_commands_for_shader_test(),
            color_filter_frame_context_for_shader_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked color-filter shader must reach real WGPU realization");

    assert!(
        observed.realizes_high_precision
            && observed.realizes_reduced_precision
            && observed.checked_scope_is_clean
            && observed.publishes_only_color_filter_entries,
        "the checked color-filter shader program is unrealized"
    );
}

#[test]
fn color_filter_layout_binds_exact_source_spatial_and_operations() {
    let observed = super::pass::color_filter_layout_observation_for_test(
        color_filter_commands_for_shader_test(),
        color_filter_frame_context_for_shader_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_filter_source
            && observed.binds_exact_nearest_sampler
            && observed.binds_spatial_and_read_only_operations
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the color-filter layout has a missing or dummy binding"
    );
}

#[test]
fn mixed_color_and_spatial_filters_preserve_the_unsupported_operation_diagnostic() {
    let observed = super::pass::mixed_color_unsupported_diagnostic_observation_for_test(
        authored_color_filter_runs_for_test(),
        color_then_blur_filters_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.pure_color_retains_gpu_color_diagnostic
            && observed.color_then_blur_reports_gpu_blur_diagnostic
            && observed.mixed_graph_stays_outside_color_filter_preparation,
        "a spatial-filter pass was admitted or color filtering masked its diagnostic"
    );
}

struct BoundedBackdropProductionFrameForTest {
    output: ImageBuffer,
    result: super::renderer::BoundedBackdropRenderResultForTest,
    publication_count: usize,
}

fn render_bounded_backdrop_fixture_for_test(
    scene: &Scene,
    size: PhysicalSize,
    parameters: Parameters,
    working_format: WorkingFormat,
) -> BoundedBackdropProductionFrameForTest {
    let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
    let publication_before = surface.headless_publication_count_for_test();
    let result = pollster::block_on(renderer.render_bounded_backdrop_fixture_for_test(
        &mut surface,
        scene,
        parameters,
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!("the bounded-backdrop fixture must use the production graph: {error}")
    });
    BoundedBackdropProductionFrameForTest {
        output: pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the published bounded-backdrop fixture must be readable: {error}")
        }),
        result,
        publication_count: surface
            .headless_publication_count_for_test()
            .saturating_sub(publication_before),
    }
}

fn bounded_backdrop_reference_rect_for_test(
    size: PhysicalSize,
    rect: (u32, u32, u32, u32),
    straight: [u8; 4],
) -> ReferencePremultipliedRgba8Buffer {
    let mut buffer = ReferencePremultipliedRgba8Buffer::try_new(size).unwrap();
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            buffer
                .set_pixel(x, y, reference_premultiplied_pixel_for_test(straight))
                .unwrap();
        }
    }
    buffer
}

fn bounded_backdrop_capture_reference_for_test(
    source: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    extent: PhysicalSize,
) -> ReferencePremultipliedRgba8Buffer {
    let mut capture = ReferencePremultipliedRgba8Buffer::try_new(extent).unwrap();
    for y in 0..extent.height() {
        for x in 0..extent.width() {
            let source_x = origin.0 + i32::try_from(x).unwrap();
            let source_y = origin.1 + i32::try_from(y).unwrap();
            if source_x >= 0
                && source_y >= 0
                && source_x < i32::try_from(source.physical_size().width()).unwrap()
                && source_y < i32::try_from(source.physical_size().height()).unwrap()
            {
                capture
                    .set_pixel(
                        x,
                        y,
                        source
                            .pixel(
                                u32::try_from(source_x).unwrap(),
                                u32::try_from(source_y).unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    capture
}

fn bounded_backdrop_place_capture_for_test(
    capture: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    output_size: PhysicalSize,
) -> ReferencePremultipliedRgba8Buffer {
    let mut placed = ReferencePremultipliedRgba8Buffer::try_new(output_size).unwrap();
    for y in 0..capture.physical_size().height() {
        for x in 0..capture.physical_size().width() {
            let output_x = origin.0 + i32::try_from(x).unwrap();
            let output_y = origin.1 + i32::try_from(y).unwrap();
            if output_x >= 0
                && output_y >= 0
                && output_x < i32::try_from(output_size.width()).unwrap()
                && output_y < i32::try_from(output_size.height()).unwrap()
            {
                placed
                    .set_pixel(
                        u32::try_from(output_x).unwrap(),
                        u32::try_from(output_y).unwrap(),
                        capture.pixel(x, y).unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    placed
}

fn bounded_backdrop_clip_reference_for_test(
    source: &ReferencePremultipliedRgba8Buffer,
    rect: (u32, u32, u32, u32),
) -> ReferencePremultipliedRgba8Buffer {
    let mut clipped = ReferencePremultipliedRgba8Buffer::try_new(source.physical_size()).unwrap();
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            clipped
                .set_pixel(x, y, source.pixel(x, y).unwrap())
                .unwrap();
        }
    }
    clipped
}

fn bounded_backdrop_frame_matches_for_test(
    frame: &BoundedBackdropProductionFrameForTest,
    expected: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    extent: PhysicalSize,
    tolerance: u8,
) -> bool {
    let expected_size = expected.physical_size();
    let expected = reference_straight_bytes_for_test(expected);
    let energy_tolerance = match frame.result.working_format {
        WorkingFormat::HighPrecision => 0.015,
        WorkingFormat::ReducedPrecision => 0.025,
    };
    frame.result.output_extent == expected_size
        && frame.output.size() == frame.result.output_extent
        && frame.result.parent_spatial.device_origin == (0, 0)
        && frame.result.parent_spatial.device_extent == frame.result.output_extent
        && frame.result.parent_spatial.texel_origin == Point::new(0.0, 0.0)
        && frame.result.parent_spatial.raster_scale == 1.0
        && frame.result.capture_spatial.device_origin == origin
        && frame.result.capture_spatial.device_extent == extent
        && frame.result.capture_spatial.texel_origin
            == Point::new(f64::from(origin.0), f64::from(origin.1))
        && frame.result.capture_spatial.logical_bounds
            == [
                f64::from(origin.0),
                f64::from(origin.1),
                f64::from(extent.width()),
                f64::from(extent.height()),
            ]
        && frame.result.capture_spatial.raster_scale == 1.0
        && frame.result.stats.layers == 1
        && frame.result.stats.route == Some(RenderRoute::GpuGraph)
        && frame.publication_count == 1
        && spatial_filter_alpha_energy_error_for_test(frame.output.rgba(), &expected)
            <= energy_tolerance
        && graph_pixels_match_for_test(
            frame.output.rgba(),
            &expected,
            frame.result.working_format,
            tolerance,
        )
}

#[test]
fn backdrop_blur_mirrors_at_semantic_bounds_not_allocation_padding() {
    let size = PhysicalSize::new(8, 6);
    let origin = (2, 1);
    let extent = PhysicalSize::new(4, 4);
    let base = [32, 64, 96, 255];
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(2.0, 1.0, 1.0, 4.0),
            color_from_straight_rgba8_for_test([240, 32, 16, 255]),
        )
        .fill(
            Rect::new(3.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test([16, 80, 224, 255]),
        )
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                        BackdropCaptureBounds::try_new(Rect::new(2.0, 1.0, 4.0, 4.0)).unwrap(),
                        Some(
                            ClipInput::try_shape(Shape::rect(Rect::new(2.0, 1.0, 4.0, 4.0)))
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (2, 1, 1, 4), [240, 32, 16, 255])
        .source_over(&parent)
        .unwrap();
    let parent = bounded_backdrop_reference_rect_for_test(size, (3, 1, 3, 4), [16, 80, 224, 255])
        .source_over(&parent)
        .unwrap();
    let filtered = bounded_backdrop_capture_reference_for_test(&parent, origin, extent)
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let expected = bounded_backdrop_place_capture_for_test(&filtered, origin, size)
        .source_over(&parent)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, extent, 4),
            "semantic-edge backdrop blur differs from its mirrored oracle: format={working_format:?}"
        );
    }
}

#[test]
fn backdrop_reads_only_completed_prior_content_and_base_once() {
    let size = PhysicalSize::new(8, 6);
    let origin = (-1, 1);
    let extent = PhysicalSize::new(7, 4);
    let base = [32, 64, 96, 255];
    let prior = [224, 48, 24, 255];
    let later = [32, 224, 80, 160];
    let operations = [
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
    ];
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test(prior),
        )
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        color_filter_list(operations),
                        BackdropCaptureBounds::try_new(Rect::new(-1.0, 1.0, 7.0, 4.0)).unwrap(),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        )
        .fill(
            Rect::new(4.0, 2.0, 3.0, 2.0),
            color_from_straight_rgba8_for_test(later),
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 1, 3, 4), prior)
        .source_over(&parent)
        .unwrap();
    let filtered = bounded_backdrop_capture_reference_for_test(&parent, origin, extent)
        .apply_color_filter_pipeline(&color_filter_pipeline(operations))
        .unwrap();
    let completed = bounded_backdrop_place_capture_for_test(&filtered, origin, size)
        .source_over(&parent)
        .unwrap();
    let expected = bounded_backdrop_reference_rect_for_test(size, (4, 2, 3, 2), later)
        .source_over(&completed)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, extent, 2),
            "bounded backdrop capture included a later sibling, omitted a prior sibling, or changed the signed base mapping: format={working_format:?}"
        );
    }
}

#[test]
fn backdrop_foreground_is_not_filtered_and_composites_above_backdrop() {
    let size = PhysicalSize::new(8, 6);
    let origin = (0, 0);
    let base = [32, 64, 192, 255];
    let foreground = [240, 32, 16, 255];
    let blur = FilterBlur::try_new(1.25).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_backdrop_filter(
                BackdropFilterInput::try_new(
                    single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                    BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                    None,
                )
                .unwrap(),
            )
            .unwrap(),
        |scene| {
            scene.fill(
                Rect::new(3.0, 2.0, 2.0, 2.0),
                color_from_straight_rgba8_for_test(foreground),
            );
        },
    );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let filtered = parent
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let group = bounded_backdrop_reference_rect_for_test(size, (3, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let expected = group.source_over(&parent).unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, size, 4),
            "bounded backdrop execution filtered or under-composited its foreground: format={working_format:?}"
        );
    }
}

#[test]
fn later_siblings_observe_completed_backdrop_group() {
    let size = PhysicalSize::new(8, 6);
    let origin = (0, 0);
    let base = [40, 80, 160, 255];
    let foreground = [224, 32, 48, 255];
    let later = [32, 224, 96, 160];
    let invert = ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap());
    let mut scene = Scene::new();
    scene
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        color_filter_list([invert]),
                        BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |scene| {
                scene.fill(
                    Rect::new(2.0, 2.0, 2.0, 2.0),
                    color_from_straight_rgba8_for_test(foreground),
                );
            },
        )
        .fill(
            Rect::new(3.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test(later),
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let filtered = parent
        .apply_color_filter_pipeline(&color_filter_pipeline([invert]))
        .unwrap();
    let group = bounded_backdrop_reference_rect_for_test(size, (2, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let completed = group.source_over(&parent).unwrap();
    let expected = bounded_backdrop_reference_rect_for_test(size, (3, 1, 3, 4), later)
        .source_over(&completed)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, size, 2),
            "a later sibling failed to observe the complete bounded backdrop group: format={working_format:?}"
        );
    }
}

#[test]
fn outer_clip_precedes_mask_and_opacity_but_follows_filter() {
    let size = PhysicalSize::new(8, 6);
    let base = [32, 64, 192, 255];
    let prior = [240, 32, 16, 255];
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mask_alpha = 128_u8;
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[mask_alpha],
        ImageQuality::Low,
        Extend::Pad,
    );
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(2.0, 1.0, 4.0, 4.0)))
        .unwrap()
        .try_opacity(0.5)
        .unwrap()
        .with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(mask, Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
        )
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(1.0, 0.0, 1.0, 6.0),
            color_from_straight_rgba8_for_test(prior),
        )
        .layer(layer, |_| {});
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (1, 0, 1, 6), prior)
        .source_over(&parent)
        .unwrap();
    let filtered = parent
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let clipped = bounded_backdrop_clip_reference_for_test(&filtered, (2, 1, 4, 4));
    let masked = clipped
        .apply_opacity(f32::from(mask_alpha) / 255.0)
        .unwrap()
        .apply_opacity(0.5)
        .unwrap();
    let expected = masked.source_over(&parent).unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, (0, 0), size, 4),
            "bounded backdrop outer clip, mask, opacity, or filter order differs from the oracle: format={working_format:?}"
        );
    }
}

fn bounded_backdrop_integration_fixture_for_test() -> (
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
    let invert = ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap());
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
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 1, 3, 4), prior)
        .source_over(&parent)
        .unwrap();
    let filtered = parent
        .apply_color_filter_pipeline(&color_filter_pipeline([invert]))
        .and_then(|buffer| {
            buffer.apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        })
        .unwrap();
    let group = bounded_backdrop_reference_rect_for_test(size, (3, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let completed = group.source_over(&parent).unwrap();
    let expected = bounded_backdrop_reference_rect_for_test(size, (5, 1, 2, 4), later)
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

fn bounded_backdrop_broad_capabilities_remain_diagnostic_for_test() -> bool {
    let capabilities = Capabilities::CURRENT;
    let offscreen = capabilities.offscreen_pipeline();
    let unsupported = [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ),
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::LayerFilterExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ),
    ];
    offscreen.supports_bounded_backdrop_filter_execution()
        && !offscreen.supports_broad_backdrop_execution()
        && !offscreen.supports_backdrop_isolation_composition()
        && !offscreen.supports_layer_filter_execution()
        && !capabilities.filters().supports_layer_filters()
        && capabilities
            .ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::OffscreenPipeline,
                PrimitiveOperation::BoundedBackdropFilterExecution,
            ))
            .is_ok()
        && unsupported.into_iter().all(|expected| {
            capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
        })
}

fn bounded_backdrop_repeated_resources_stabilize_for_test(
    scene: &Scene,
    size: PhysicalSize,
    parameters: Parameters,
    expected: &ReferencePremultipliedRgba8Buffer,
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(512 * 1024 * 1024)),
    ))
    .expect("bounded-backdrop retained-resource coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("bounded-backdrop retained-resource coverage requires a surface");
    for _ in 0..2 {
        pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
            &mut surface,
            scene,
            parameters,
            WorkingFormat::ReducedPrecision,
        ))
        .expect("bounded-backdrop retained-resource warm-up must succeed");
    }
    let warmed_output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the warmed bounded-backdrop publication must be readable");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the warmed bounded-backdrop device must remain ready");
    let warmed_resources = ready.internal_resource_manager_observation_for_test();
    let warmed_cache = ready.device_pass_cache_counts_for_test();
    let mut resources = Vec::new();
    let mut caches = Vec::new();
    let mut stats = Vec::new();
    for _ in 0..3 {
        let frame = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
            &mut surface,
            scene,
            parameters,
            WorkingFormat::ReducedPrecision,
        ))
        .expect("repeated public bounded-backdrop frames must succeed");
        stats.push(GraphPublicStatsForTest::from(frame));
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("repeated public bounded-backdrop frames must retain the ready device");
        resources.push(ready.internal_resource_manager_observation_for_test());
        caches.push(ready.device_pass_cache_counts_for_test());
    }
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated bounded-backdrop publication must remain readable");
    let expected = reference_straight_bytes_for_test(expected);
    color_filter_repeated_resource_observations_are_stable_for_test(&resources, &warmed_resources)
        && resources.iter().all(|actual| {
            actual.gaussian_kernel_count_for_test()
                == warmed_resources.gaussian_kernel_count_for_test()
        })
        && warmed_resources.gaussian_kernel_count_for_test() > 0
        && warmed_resources.effect_texture_count_for_test() > 0
        && warmed_cache.has_render_pipelines()
        && caches.iter().all(|actual| *actual == warmed_cache)
        && stats
            .first()
            .is_some_and(|first| stats.iter().all(|actual| actual == first))
        && output.rgba() == warmed_output.rgba()
        && graph_pixels_match_for_test(output.rgba(), &expected, WorkingFormat::ReducedPrecision, 4)
}

fn bounded_backdrop_zero_budget_releases_idle_resources_for_test(
    scene: &Scene,
    size: PhysicalSize,
    parameters: Parameters,
    expected: &ReferencePremultipliedRgba8Buffer,
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("bounded-backdrop zero-budget coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("bounded-backdrop zero-budget coverage requires a surface");
    pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the first bounded-backdrop zero-budget frame must succeed");
    let first = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first bounded-backdrop zero-budget publication must be readable");
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the first bounded-backdrop zero-budget frame must retain its device")
        .device_pass_cache_counts_for_test();
    pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the repeated bounded-backdrop zero-budget frame must succeed");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the repeated bounded-backdrop zero-budget frame must retain its device");
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let second = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated bounded-backdrop zero-budget publication must be readable");
    let expected = reference_straight_bytes_for_test(expected);

    resources.leased_count == 0
        && resources.idle_count == 0
        && resources.active_frame_count == 0
        && resources.resolved_lease_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
        && resources.accounted_entry_bytes == Some(0)
        && resources.effect_texture_count_for_test() == 0
        && resources.gaussian_kernel_count_for_test() == 0
        && cache_before == cache_after
        && cache_after.has_render_pipelines()
        && first.rgba() == second.rgba()
        && graph_pixels_match_for_test(second.rgba(), &expected, WorkingFormat::ReducedPrecision, 4)
}

fn bounded_backdrop_diagnostic_backdrop_layer_for_test() -> Layer {
    Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                single_filter_list_for_test(FilterOp::blur(FilterBlur::try_new(0.5).unwrap()))[0]
                    .clone(),
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap()
}

fn bounded_backdrop_broad_inputs_reject_before_allocation_for_test(
    renderer: &mut Renderer,
    surface: &mut Surface,
) -> bool {
    let mut nested = Scene::new();
    nested.layer(Layer::new(), |scene| {
        scene.layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        );
    });
    let mut transformed = Scene::new();
    transformed.layer(
        bounded_backdrop_diagnostic_backdrop_layer_for_test()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap(),
        |_| {},
    );
    let mut repeated = Scene::new();
    repeated
        .layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        )
        .layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        );
    let mut layer_filter = Scene::new();
    layer_filter.layer(
        Layer::new()
            .try_filter(Filter::try_blur(0.5).unwrap())
            .unwrap(),
        |_| {},
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("bounded-backdrop broad diagnostic coverage requires a ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let publication_before = surface.headless_publication_count_for_test();
    let nested = pollster::block_on(renderer.render(surface, &nested, Parameters::default()));
    let transformed =
        pollster::block_on(renderer.render(surface, &transformed, Parameters::default()));
    let repeated = pollster::block_on(renderer.render(surface, &repeated, Parameters::default()));
    let layer_filter =
        pollster::block_on(renderer.render(surface, &layer_filter, Parameters::default()));
    let root = BackdropFilterInput::try_root_backdrop(
        single_filter_list_for_test(FilterOp::blur(FilterBlur::try_new(0.5).unwrap()))[0].clone(),
        None,
    );
    let reference =
        UnresolvedResource::new(UnresolvedResourceKind::Filter, "#broad_backdrop-filter");
    let reference_error = Error::unresolved_resource(reference.clone());
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("bounded-backdrop broad rejections must preserve the ready device");
    let resources_after = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let broad = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );
    let layer =
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter);
    let root_policy = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::RootBackdropPolicy,
    );
    nested
        .as_ref()
        .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && transformed
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && repeated
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && layer_filter
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(layer))
        && root
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(root_policy))
        && reference_error.unresolved_resource_diagnostic() == Some(&reference)
        && resources_after == resources_before
        && cache_after == cache_before
        && surface.headless_publication_count_for_test() == publication_before
}

#[test]
fn bounded_backdrop_fixture_executes_while_broad_capabilities_remain_diagnostic() {
    let (scene, size, parameters, expected) = bounded_backdrop_integration_fixture_for_test();
    let frame = render_bounded_backdrop_fixture_for_test(
        &scene,
        size,
        parameters,
        WorkingFormat::ReducedPrecision,
    );

    assert!(bounded_backdrop_frame_matches_for_test(
        &frame,
        &expected,
        (0, 0),
        size,
        4
    ));
    assert!(bounded_backdrop_broad_capabilities_remain_diagnostic_for_test());
    assert!(bounded_backdrop_repeated_resources_stabilize_for_test(
        &scene, size, parameters, &expected,
    ));
    assert!(
        bounded_backdrop_zero_budget_releases_idle_resources_for_test(
            &scene, size, parameters, &expected,
        )
    );
}

#[cfg(feature = "render-window")]
#[test]
fn render_window_smoke_executes_bounded_backdrop_fixture() {
    let (scene, size, parameters, expected) = bounded_backdrop_integration_fixture_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("presented bounded-backdrop coverage requires a renderer");
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(f64::from(size.width()), f64::from(size.height())),
            format: Format::Rgba8,
            ..SurfaceOptions::default()
        },
    );
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect("presented bounded-backdrop coverage must configure");
    let presentation = presented_observation_handle_for_test(&surface);
    let rendered = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        &scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ));
    let presented = take_last_presented_texture_for_test(&mut surface)
        .and_then(|texture| {
            pollster::block_on(renderer.read_render_texture_for_test(&texture, size)).ok()
        })
        .map(|image| image.into_rgba());
    let presentation = presentation.snapshot_for_test();
    let expected = reference_straight_bytes_for_test(&expected);

    assert!(
        rendered
            .as_ref()
            .is_ok_and(|stats| stats.route == Some(RenderRoute::GpuGraph))
            && presentation.acquire_count_for_test() == 1
            && presentation.present_count_for_test() == 1
            && presentation.discarded_count_for_test() == 0
            && presented.as_deref().is_some_and(|actual| {
                graph_pixels_match_for_test(actual, &expected, WorkingFormat::ReducedPrecision, 4)
            }),
        "the presented bounded backdrop did not execute atomically"
    );
}

#[test]
fn public_dispatch_enables_only_bounded_backdrop_execution() {
    let (scene, size, parameters, expected) = bounded_backdrop_integration_fixture_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("public bounded-backdrop dispatch coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("public bounded-backdrop dispatch coverage requires a surface");
    let rendered = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        &scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ));
    let actual = pollster::block_on(renderer.read_headless(&surface))
        .expect("the public bounded-backdrop publication must be readable");
    let expected = reference_straight_bytes_for_test(&expected);

    assert!(
        rendered
            .as_ref()
            .is_ok_and(|stats| stats.route == Some(RenderRoute::GpuGraph))
            && graph_pixels_match_for_test(
                actual.rgba(),
                &expected,
                WorkingFormat::ReducedPrecision,
                4,
            )
            && bounded_backdrop_broad_capabilities_remain_diagnostic_for_test()
            && bounded_backdrop_broad_inputs_reject_before_allocation_for_test(
                &mut renderer,
                &mut surface
            ),
        "public dispatch did not enable only the exact bounded-backdrop graph"
    );
}

#[test]
fn copy_backdrop_layout_binds_parent_and_spatial_mapping() {
    let observed = super::pass::copy_backdrop_layout_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        super::frame::FrameContext::try_new(
            Size::new(16.0, 12.0),
            1.0,
            Antialiasing::Msaa8,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        )
        .unwrap(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_completed_parent
            && observed.binds_only_one_nearest_transparent_sampler
            && observed.binds_only_spatial_uniform
            && observed.targets_only_the_working_format
            && observed.source_and_result_are_distinct,
        "the copy-backdrop layout is not exact"
    );
}

#[test]
fn copy_backdrop_cache_realizes_checked_working_format_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires backend selection")
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        super::pass::copy_backdrop_cache_realization_observation_for_test(
            ready.device_for_test(),
            bounded_backdrop_graph_commands_for_test(),
            super::frame::FrameContext::try_new(
                Size::new(16.0, 12.0),
                1.0,
                Antialiasing::Msaa8,
                Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
            )
            .unwrap(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked copy-backdrop shader must reach real WGPU realization");

    assert!(
        observed.realizes_high_precision
            && observed.realizes_reduced_precision
            && observed.checked_scope_is_clean
            && observed.publishes_only_copy_backdrop_entries
            && observed.rejects_unsupported_format_before_publication,
        "the copy-backdrop working-format program is unrealized"
    );
}

#[test]
fn prepared_copy_backdrop_objects_expose_exact_encoding_handles() {
    fn require_copy_handles(objects: &super::shader::ProvisionalCopyBackdropPassObjects<'_>) {
        let _: &wgpu::Sampler = objects.parent_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    let _ = require_copy_handles;
}

#[test]
fn backdrop_blur_cache_separates_transparent_and_mirrored_edge_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires backend selection")
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        super::pass::backdrop_blur_cache_realization_observation_for_test(
            ready.device_for_test(),
            spatial_filter_authored_filter_steps_for_test(),
            filter_graph_commands_for_test(),
            bounded_backdrop_graph_commands_for_test(),
            filter_graph_context_for_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test(
        "transparent and mirrored backdrop-blur programs must reach realization",
    );

    assert!(
        observed.realizes_all_transparent_and_mirrored_programs
            && observed.checked_scope_is_clean
            && observed.publishes_exact_edge_programs
            && observed.edge_program_keys_are_distinct,
        "the checked blur cache does not distinguish transparent and mirrored edge programs"
    );
}

#[test]
fn backdrop_blur_layout_carries_semantic_mirror_bounds() {
    let observed = super::pass::backdrop_blur_layout_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_all_axis_input_precision_and_edge_keys
            && observed.binds_exact_working_source
            && observed.binds_only_one_linear_mirror_sampler
            && observed.binds_spatial_kernel_and_semantic_bounds
            && observed.targets_only_the_working_format
            && observed.semantic_bounds_match_every_mirrored_read
            && observed.shader_mirrors_logical_bounds_before_texture_mapping,
        "the backdrop blur layout omits exact semantic mirror bounds or program facts"
    );
}

#[test]
fn backdrop_filter_chain_preserves_authored_order_and_clamp_boundaries() {
    use super::pass::SpatialFilterPassTagForTest as Tag;

    let observed = super::pass::backdrop_filter_chain_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
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
            ]
            && observed.every_backdrop_blur_uses_mirror
            && observed.source_alpha_blur_uses_mirror
            && observed.every_color_operation_retains_one_clamp
            && observed.semantic_bounds_are_exact
            && observed.every_mirrored_stage_is_realizable,
        "the authored backdrop filter chain contains unrealizable mirrored stages"
    );
}

#[test]
fn backdrop_graph_encodes_copy_filter_clip_foreground_and_group_in_order() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the bounded backdrop fixture must reach its shared GPU graph executor",
    );

    assert!(
        observed.encodes_copy_filter_clip_foreground_and_group_in_order
            && observed.parent_is_copied_once
            && observed.releases_at_validated_last_use,
        "the scheduler has no bounded backdrop encoding route"
    );
}

#[test]
fn backdrop_copy_filter_and_group_use_distinct_resources() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the backdrop alias fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.copy_filter_foreground_and_group_are_distinct
            && observed.releases_at_validated_last_use,
        "the backdrop copy, filters, foreground, or group alias"
    );
}

#[test]
fn later_sibling_dependency_follows_completed_backdrop_group() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the backdrop dependency fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.later_sibling_reads_completed_group
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "the later sibling did not follow the committed backdrop-group transition"
    );
}

#[test]
fn backdrop_encode_failure_preserves_resources_cache_and_publication() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_failure_preservation_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test("the backdrop failure fixture must reach its atomic abort path");
    assert!(
        observed.encode_failure_is_reported
            && observed.resources_are_unchanged
            && observed.cache_is_unchanged
            && observed.publication_is_unchanged,
        "the backdrop encode abort changed provisional or published state"
    );
}

#[test]
fn gaussian_kernel_bytes_are_symmetric_normalized_and_exactly_cached() {
    assert_gaussian_kernel_upload_lifecycle(2.0, 1.5, 2.5);
    let plan = GaussianKernelPlan::try_new(2.0, 1.5, 2.5, GaussianKernelSamplingForm::PairedLinear)
        .unwrap();
    let samples = plan
        .upload_bytes()
        .chunks_exact(8)
        .map(|sample| {
            (
                f32::from_le_bytes(sample[0..4].try_into().unwrap()),
                f32::from_le_bytes(sample[4..8].try_into().unwrap()),
            )
        })
        .collect::<Vec<_>>();
    let symmetric = samples[1..].chunks_exact(2).all(|pair| {
        pair[0].0.to_bits() == (-pair[1].0).to_bits() && pair[0].1.to_bits() == pair[1].1.to_bits()
    });
    let normalized = (samples.iter().map(|sample| sample.1).sum::<f32>() - 1.0).abs() <= 1.0e-6;
    let storage_limit_is_checked = plan
        .validate_buffer_limits(GaussianKernelBufferLimits::for_test(
            plan.byte_len(),
            plan.byte_len() - 1,
        ))
        .is_err();

    assert!(
        symmetric && normalized && storage_limit_is_checked,
        "Gaussian sample bytes or their exact pre-allocation cache contract differ"
    );
}

#[test]
fn blur_layout_binds_exact_source_spatial_and_kernel() {
    let observed = super::pass::blur_layout_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_all_axis_input_and_precision_keys
            && observed.binds_exact_working_source
            && observed.binds_only_one_linear_sampler
            && observed.binds_spatial_and_read_only_kernel
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the blur layout has a missing or dummy binding"
    );
}

#[test]
fn blur_cache_realizes_checked_axis_input_and_precision_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked blur realization requires backend selection")
        .unwrap_or_panic_for_test("checked blur realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked blur realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(super::pass::blur_cache_realization_observation_for_test(
        ready.device_for_test(),
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        capabilities,
    ))
    .unwrap_or_panic_for_test("the checked blur shader must reach real WGPU realization");

    assert!(
        observed.realizes_all_eight_programs
            && observed.checked_scope_is_clean
            && observed.publishes_only_blur_entries,
        "the checked blur programs are unrealized"
    );
}

#[test]
fn drop_shadow_parameter_bytes_preserve_fractional_offset_and_solid_color() {
    let bytes = super::shader::drop_shadow_parameter_bytes_for_test(
        Point::new(-1.5, 0.75),
        Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
    )
    .unwrap();
    let scalar = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    assert!(
        scalar(0).to_bits() == (-1.5_f32).to_bits()
            && scalar(4).to_bits() == 0.75_f32.to_bits()
            && bytes[8..16] == [0; 8]
            && scalar(16).to_bits() == 0.125_f32.to_bits()
            && scalar(20).to_bits() == 0.25_f32.to_bits()
            && scalar(24).to_bits() == 0.375_f32.to_bits()
            && scalar(28).to_bits() == 0.5_f32.to_bits()
            && super::shader::drop_shadow_parameter_bytes_for_test(
                Point::new(f64::NAN, 0.0),
                Color::BLACK,
            )
            .is_err(),
        "drop-shadow parameters lost a fractional offset, finite layout, or solid premultiplied color"
    );
}

#[test]
fn drop_shadow_layout_binds_blurred_alpha_spatial_and_parameters() {
    let observed = super::pass::drop_shadow_layout_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_blurred_source_alpha
            && observed.binds_only_one_linear_transparent_sampler
            && observed.binds_spatial_and_parameters
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the drop-shadow colorize layout has a missing or dummy binding"
    );
}

#[test]
fn drop_shadow_cache_realizes_checked_colorize_and_merge_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked drop-shadow realization requires backend selection")
        .unwrap_or_panic_for_test("checked drop-shadow realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked drop-shadow realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        super::pass::drop_shadow_cache_realization_observation_for_test(
            ready.device_for_test(),
            spatial_filter_authored_filter_steps_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked drop-shadow programs must reach WGPU realization");

    assert!(
        observed.realizes_checked_colorize_and_merge_programs
            && observed.checked_scope_is_clean
            && observed.merge_uses_fixed_premultiplied_source_over
            && observed.merge_omits_destination_sample
            && observed.publishes_only_drop_shadow_entries,
        "the checked drop-shadow colorize or merge programs are unrealized"
    );
}

#[test]
fn prepared_spatial_filter_objects_expose_exact_encoding_handles() {
    fn require_blur_handles(objects: &super::shader::ProvisionalBlurPassObjects<'_>) {
        let _: &wgpu::Sampler = objects.source_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    fn require_drop_shadow_handles(
        objects: &super::shader::ProvisionalDropShadowColorizePassObjects<'_>,
    ) {
        let _: &wgpu::Sampler = objects.source_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    let _ = (require_blur_handles, require_drop_shadow_handles);
}

#[test]
fn spatial_filter_graph_encodes_blur_and_drop_shadow_in_authored_order() {
    use super::pass::SpatialFilterPassTagForTest as Tag;

    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the spatial-filter fixture must reach its shared GPU graph executor",
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
            && observed.each_pass_advances_once
            && observed.binds_exact_prepared_resources
            && observed.uses_signed_viewport_and_scissor
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "the scheduler has no ordered spatial-filter encoding route"
    );
}

#[test]
fn blur_passes_use_distinct_source_intermediate_and_result() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test("the blur fixture must reach its shared GPU graph executor");
    assert!(
        observed.blur_pass_count == 4
            && observed.blur_sources_intermediates_and_results_are_distinct
            && observed.binds_exact_prepared_resources
            && observed.kernels_release_at_validated_last_use
            && observed.textures_release_at_validated_last_use,
        "the blur scheduler has no exact pass receipts"
    );
}

#[test]
fn drop_shadow_reads_source_twice_and_releases_after_merge() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test("the drop-shadow fixture must reach its shared GPU graph executor");

    assert!(
        observed.drop_shadow_colorize_count == 1
            && observed.drop_shadow_merge_count == 1
            && observed.drop_shadow_reads_original_source_twice
            && observed.original_source_releases_after_merge
            && observed.textures_release_at_validated_last_use
            && observed.each_pass_advances_once,
        "the drop-shadow scheduler lost its exact lease transition"
    );
}

#[test]
fn spatial_filter_encode_and_scope_failures_preserve_resources_cache_and_publication() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.spatial_filter_failure_preservation_observation_for_test(
            identity,
            spatial_filter_authored_filter_steps_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test("the spatial-filter failure fixture must exercise both abort paths");

    assert!(
        observed.encode_failure_is_reported
            && observed.scope_failure_is_reported
            && observed.resources_are_unchanged
            && observed.cache_is_unchanged
            && observed.publication_is_unchanged,
        "failed spatial-filter encoding changed provisional or published state"
    );
}

struct SpatialFilterProductionFrameForTest {
    output: ImageBuffer,
    result: super::renderer::SpatialFilterRenderResultForTest,
    publication_count: usize,
}

fn single_filter_list_for_test(operation: FilterOp) -> Vec<FilterList> {
    vec![
        FilterList::try_ops(vec![operation])
            .expect("the spatial-filter operation must form one filter"),
    ]
}

fn graph_pixel_renderer_for_test(
    working_format: WorkingFormat,
    size: PhysicalSize,
) -> (Renderer, Surface) {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("spatial-filter pixel execution requires a real renderer: {error}")
    });
    assert!(
        graph_supported_working_formats_for_test(&mut renderer).contains(&working_format),
        "spatial-filter pixel execution requires the requested real working format"
    );
    let surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .unwrap_or_else(|error| {
        panic!("spatial-filter pixel execution requires a headless surface: {error}")
    });
    (renderer, surface)
}

fn render_spatial_filter_fixture_for_test(
    renderer: &mut Renderer,
    surface: &mut Surface,
    scene: &Scene,
    filters: Vec<FilterList>,
    working_format: WorkingFormat,
) -> SpatialFilterProductionFrameForTest {
    let publication_before = surface.headless_publication_count_for_test();
    let result = pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
        surface,
        scene,
        filters,
        Parameters::default(),
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!("the spatial-filter fixture must use the production graph: {error}")
    });
    SpatialFilterProductionFrameForTest {
        output: pollster::block_on(renderer.read_headless(surface)).unwrap_or_else(|error| {
            panic!("the published spatial-filter fixture must be readable: {error}")
        }),
        result,
        publication_count: surface
            .headless_publication_count_for_test()
            .saturating_sub(publication_before),
    }
}

fn spatial_filter_reference_buffer_for_test(
    size: PhysicalSize,
    opaque_pixels: &[(u32, u32, PremultipliedRgba8)],
) -> ReferencePremultipliedRgba8Buffer {
    let mut source = ReferencePremultipliedRgba8Buffer::try_new(size).unwrap();
    for &(x, y, pixel) in opaque_pixels {
        source.set_pixel(x, y, pixel).unwrap();
    }
    source
}

fn spatial_filter_image_scene_for_test(
    size: PhysicalSize,
    pixels: Vec<u8>,
    destination: Rect,
) -> Scene {
    let image = Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        Arc::<[u8]>::from(pixels),
    )
    .expect("the spatial-filter pixel fixture must form one RGBA image");
    let mut scene = Scene::new();
    scene.image(image, destination, ImageFit::Stretch);
    scene
}

fn spatial_filter_maximum_error_for_test(
    actual: &[u8],
    expected: &[u8],
    working_format: WorkingFormat,
) -> (u8, u8) {
    match working_format {
        WorkingFormat::HighPrecision => (
            high_precision_terminal_error_for_test(actual, expected).unwrap_or(u8::MAX),
            0,
        ),
        WorkingFormat::ReducedPrecision => {
            reduced_precision_terminal_error_for_test(actual, expected)
                .unwrap_or((u8::MAX, u8::MAX))
        }
    }
}

fn spatial_filter_alpha_energy_error_for_test(actual: &[u8], expected: &[u8]) -> f64 {
    let actual = actual
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    let expected = expected
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    actual.abs_diff(expected) as f64 / expected.max(1) as f64
}

fn spatial_filter_alpha_energy_relative_to_for_test(actual: &[u8], expected_energy: u64) -> f64 {
    let actual = actual
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    actual.abs_diff(expected_energy) as f64 / expected_energy.max(1) as f64
}

fn spatial_filter_frame_has_exact_execution_for_test(
    frame: &SpatialFilterProductionFrameForTest,
    size: PhysicalSize,
    working_format: WorkingFormat,
) -> bool {
    frame.output.size() == size
        && frame.result.output_extent == size
        && frame.result.working_format == working_format
        && frame.result.stats.route == Some(RenderRoute::GpuGraph)
        && frame.publication_count == 1
        && frame.result.stats.commands > 0
        && terminal_straight_rgba8_is_canonical_for_test(frame.output.rgba())
}

fn spatial_filter_pixels_match_oracle_for_test(
    frame: &SpatialFilterProductionFrameForTest,
    expected: &[u8],
) -> bool {
    let (alpha_error, color_error) = spatial_filter_maximum_error_for_test(
        frame.output.rgba(),
        expected,
        frame.result.working_format,
    );
    let energy_error = spatial_filter_alpha_energy_error_for_test(frame.output.rgba(), expected);
    let energy_tolerance = match frame.result.working_format {
        WorkingFormat::HighPrecision => 0.015,
        WorkingFormat::ReducedPrecision => 0.025,
    };
    alpha_error <= 4 && color_error <= 4 && energy_error <= energy_tolerance
}

fn spatial_filter_alpha_centroid_for_test(bytes: &[u8], size: PhysicalSize) -> Point {
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    let mut energy = 0.0;
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        let alpha = f64::from(pixel[3]);
        let index = u32::try_from(index).expect("the spatial-filter centroid index must fit u32");
        weighted_x += (f64::from(index % size.width()) + 0.5) * alpha;
        weighted_y += (f64::from(index / size.width()) + 0.5) * alpha;
        energy += alpha;
    }
    Point::new(weighted_x / energy, weighted_y / energy)
}

fn spatial_filter_blur_identity_and_transparency_are_exact_for_test(
    working_format: WorkingFormat,
    size: PhysicalSize,
    scene: &Scene,
    blurred: &SpatialFilterProductionFrameForTest,
) -> bool {
    let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let with_identity = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![
            FilterList::try_ops(vec![FilterOp::blur(blur)]).unwrap(),
            FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(0.0).unwrap())]).unwrap(),
        ],
        working_format,
    );
    if !spatial_filter_frame_has_exact_execution_for_test(&with_identity, size, working_format)
        || with_identity.output.rgba() != blurred.output.rgba()
    {
        return false;
    }
    let transparent = spatial_filter_image_scene_for_test(
        PhysicalSize::new(1, 1),
        vec![17, 31, 47, 0],
        Rect::new(6.0, 6.0, 1.0, 1.0),
    );
    let transparent = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &transparent,
        single_filter_list_for_test(FilterOp::blur(blur)),
        working_format,
    );
    spatial_filter_frame_has_exact_execution_for_test(&transparent, size, working_format)
        && transparent.output.rgba().iter().all(|byte| *byte == 0)
}

fn spatial_filter_drop_shadow_order_is_authored_for_test(
    working_format: WorkingFormat,
    size: PhysicalSize,
    scene: &Scene,
    shadow: FilterDropShadow,
) -> bool {
    let shadow = FilterList::try_ops(vec![FilterOp::drop_shadow(shadow)]).unwrap();
    let blur =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(0.5).unwrap())]).unwrap();
    let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
    let shadow_then_blur = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![shadow.clone(), blur.clone()],
        working_format,
    );
    let blur_then_shadow = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![blur, shadow],
        working_format,
    );
    spatial_filter_frame_has_exact_execution_for_test(&shadow_then_blur, size, working_format)
        && spatial_filter_frame_has_exact_execution_for_test(
            &blur_then_shadow,
            size,
            working_format,
        )
        && shadow_then_blur.output.rgba() != blur_then_shadow.output.rgba()
}

#[test]
fn blur_impulse_is_symmetric_normalized_and_matches_oracle() {
    let size = PhysicalSize::new(17, 17);
    let center = (8, 8);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let source = spatial_filter_reference_buffer_for_test(
        size,
        &[(
            center.0,
            center.1,
            PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap(),
        )],
    );
    let expected = source
        .apply_blur(blur, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut impulse_pixels = vec![0; 7 * 7 * 4];
    impulse_pixels[(3 * 7 + 3) * 4..][..4].copy_from_slice(&[255; 4]);
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(7, 7),
        impulse_pixels,
        Rect::new(5.0, 5.0, 7.0, 7.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let alpha = |x: u32, y: u32| frame.output.rgba()[((y * 17 + x) * 4 + 3) as usize];
        let symmetric = (0..17).all(|coordinate| {
            alpha(coordinate, center.1) == alpha(16 - coordinate, center.1)
                && alpha(center.0, coordinate) == alpha(center.0, 16 - coordinate)
        });
        let (alpha_error, color_error) =
            spatial_filter_maximum_error_for_test(frame.output.rgba(), &expected, working_format);
        let energy_tolerance = match working_format {
            WorkingFormat::HighPrecision => 0.015,
            WorkingFormat::ReducedPrecision => 0.025,
        };
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (5, 5)
                && frame.result.source_spatial.device_extent == PhysicalSize::new(7, 7)
                && frame.result.result_spatial.device_origin == (2, 2)
                && frame.result.result_spatial.device_extent == PhysicalSize::new(13, 13)
                && symmetric
                && alpha_error <= 4
                && color_error <= 4
                && spatial_filter_alpha_energy_relative_to_for_test(frame.output.rgba(), 255)
                    <= energy_tolerance,
            "blur impulse production-GPU comparison exceeds its exact grid, symmetry, or oracle tolerance"
        );
    }
}

#[test]
fn ordinary_blur_samples_transparent_black_at_all_edges() {
    let size = PhysicalSize::new(13, 13);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let opaque = PremultipliedRgba8::try_new(64, 128, 255, 255).unwrap();
    let pixels = (5..=7)
        .flat_map(|y| (5..=7).map(move |x| (x, y, opaque)))
        .collect::<Vec<_>>();
    let source = spatial_filter_reference_buffer_for_test(size, &pixels);
    let expected_high = source
        .apply_blur_to_high_precision_straight_rgba8_for_gpu_oracle(
            blur,
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let expected_reduced = source
        .apply_blur(blur, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut edge_pixels = vec![0; 9 * 9 * 4];
    for y in 3..=5 {
        for x in 3..=5 {
            edge_pixels[(y * 9 + x) * 4..][..4].copy_from_slice(&[64, 128, 255, 255]);
        }
    }
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(9, 9),
        edge_pixels,
        Rect::new(2.0, 2.0, 9.0, 9.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let expected = match working_format {
            WorkingFormat::HighPrecision => &expected_high,
            WorkingFormat::ReducedPrecision => &expected_reduced,
        };
        let alpha = frame
            .output
            .rgba()
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        let transparent_surface_edge = (0..13).all(|coordinate| {
            alpha[coordinate] == 0
                && alpha[12 * 13 + coordinate] == 0
                && alpha[coordinate * 13] == 0
                && alpha[coordinate * 13 + 12] == 0
        });
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (2, 2)
                && frame.result.result_spatial.device_origin == (-1, -1)
                && frame.result.result_spatial.device_extent == PhysicalSize::new(15, 15)
                && transparent_surface_edge
                && spatial_filter_blur_identity_and_transparency_are_exact_for_test(
                    working_format,
                    size,
                    &scene,
                    &frame,
                )
                && spatial_filter_pixels_match_oracle_for_test(&frame, expected),
            "ordinary blur edge production-GPU comparison violates transparent-black sampling"
        );
    }
}

#[test]
fn drop_shadow_preserves_source_uses_fractional_offset_and_expands_signed_bounds() {
    let size = PhysicalSize::new(17, 17);
    let source_pixel = PremultipliedRgba8::try_new(224, 64, 16, 255).unwrap();
    let source = spatial_filter_reference_buffer_for_test(
        size,
        &[(3, 7, source_pixel), (4, 7, source_pixel)],
    );
    let shadow = FilterDropShadow::try_new(
        Point::new(-1.5, 0.75),
        FilterBlur::try_new(1.0).unwrap(),
        Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
    )
    .unwrap();
    let expected_high = source
        .apply_fractional_drop_shadow_to_high_precision_straight_rgba8_for_gpu_oracle(
            &shadow,
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let expected_reduced = source
        .apply_fractional_drop_shadow_for_gpu_oracle(&shadow, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut shadow_pixels = vec![0; 8 * 7 * 4];
    shadow_pixels[(3 * 8 + 3) * 4..][..4].copy_from_slice(&[224, 64, 16, 255]);
    shadow_pixels[(3 * 8 + 4) * 4..][..4].copy_from_slice(&[224, 64, 16, 255]);
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(8, 7),
        shadow_pixels,
        Rect::new(0.0, 4.0, 8.0, 7.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::drop_shadow(shadow)),
            working_format,
        );
        let expected = match working_format {
            WorkingFormat::HighPrecision => &expected_high,
            WorkingFormat::ReducedPrecision => &expected_reduced,
        };
        let source_is_unchanged = [(3, 7), (4, 7)].into_iter().all(|(x, y)| {
            frame.output.rgba()[((y * 17 + x) * 4) as usize..][..4] == [224, 64, 16, 255]
        });
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (0, 4)
                && frame.result.result_spatial.device_origin.0 < 0
                && frame.result.result_spatial.logical_bounds[0] < 0.0
                && source_is_unchanged
                && spatial_filter_drop_shadow_order_is_authored_for_test(
                    working_format,
                    size,
                    &scene,
                    shadow,
                )
                && spatial_filter_pixels_match_oracle_for_test(&frame, expected),
            "drop-shadow production-GPU comparison loses SourceAlpha, fractional offset, signed bounds, or source merge"
        );
    }
}

#[test]
fn nonuniform_scale_and_skew_preserve_local_blur_shape() {
    let size = PhysicalSize::new(25, 21);
    let transform = Transform::try_new([2.0, 0.0, 0.5, 1.25, 4.0, 3.0]).unwrap();
    let local_bounds = Rect::new(2.0, 4.0, 2.0, 2.0);
    let expected_center = graph_transform_point_for_test(transform, Point::new(3.0, 5.0));
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(local_bounds, Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap());
    });

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let centroid = spatial_filter_alpha_centroid_for_test(frame.output.rgba(), size);
        let tolerance = match working_format {
            WorkingFormat::HighPrecision => 0.25,
            WorkingFormat::ReducedPrecision => 0.35,
        };
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.raster_scale == 1.0
                && frame.result.result_spatial.raster_scale == 1.0
                && (centroid.x() - expected_center.x()).abs() <= tolerance
                && (centroid.y() - expected_center.y()).abs() <= tolerance,
            "transformed blur production-GPU comparison exceeds the local-shape centroid tolerance"
        );
    }
}

fn spatial_filter_mixed_filter_fixture_for_test() -> (Scene, Vec<FilterList>, PhysicalSize, Vec<u8>)
{
    let size = PhysicalSize::new(15, 13);
    let source = spatial_filter_reference_buffer_for_test(
        size,
        &[
            (5, 5, PremultipliedRgba8::try_new(224, 64, 16, 255).unwrap()),
            (6, 5, PremultipliedRgba8::try_new(32, 192, 96, 255).unwrap()),
            (6, 6, PremultipliedRgba8::try_new(48, 80, 240, 255).unwrap()),
        ],
    );
    let blur = FilterBlur::try_new(0.75).unwrap();
    let shadow = FilterDropShadow::try_new(
        Point::new(-1.25, 0.5),
        FilterBlur::try_new(0.5).unwrap(),
        Color::try_rgba(0.25, 0.5, 0.75, 0.625).unwrap(),
    )
    .unwrap();
    let invert = UnitFilterAmount::try_new(0.25).unwrap();
    let opacity = UnitFilterAmount::try_new(0.8).unwrap();
    let expected = source
        .apply_color_filter_pipeline(&color_filter_pipeline([ColorFilterOp::Invert(invert)]))
        .and_then(|buffer| buffer.apply_blur(blur, BlurPolicy::css_filter_default()))
        .and_then(|buffer| {
            buffer.apply_fractional_drop_shadow_for_gpu_oracle(
                &shadow,
                BlurPolicy::css_filter_default(),
            )
        })
        .and_then(|buffer| {
            buffer.apply_color_filter_pipeline(&color_filter_pipeline([ColorFilterOp::Opacity(
                opacity,
            )]))
        })
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let scene = spatial_filter_image_scene_for_test(
        size,
        reference_straight_bytes_for_test(&source),
        Rect::new(0.0, 0.0, f64::from(size.width()), f64::from(size.height())),
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

fn spatial_filter_public_spatial_graph_diagnostic_for_test(
    scene: &Scene,
    operation: FilterOp,
    size: PhysicalSize,
) -> Option<UnsupportedPrimitive> {
    let commands = scene
        .normalize(Capabilities::CURRENT)
        .expect("the public spatial-filter diagnostic fixture must normalize capture input");
    let context = super::frame::FrameContext::try_new(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
        Antialiasing::Area,
        Color::TRANSPARENT,
    )
    .expect("the public spatial-filter diagnostic fixture must form a frame context");
    let graph = super::frame::authored_filter_graph_for_test(
        single_filter_list_for_test(operation),
        commands,
        context,
    )
    .expect("the public spatial-filter diagnostic fixture must form an authored graph");
    super::renderer::unsupported_graph_diagnostic_for_test(
        &graph,
        Format::Rgba8,
        &DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .expect("the retained public dispatch classifier must diagnose a spatial-filter graph")
}

fn repeated_spatial_filter_resources_are_stable_for_test(
    scene: &Scene,
    filters: &[FilterList],
    size: PhysicalSize,
    expected: &[u8],
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect("spatial-filter retained-resource coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("spatial-filter retained-resource coverage requires a surface");
    for _ in 0..2 {
        pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
            &mut surface,
            scene,
            filters.to_vec(),
            Parameters::default(),
            WorkingFormat::ReducedPrecision,
        ))
        .expect("spatial-filter retained-resource warm-up must succeed");
    }
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("spatial-filter retained-resource warm-up must keep its device");
    let warmed_resources = ready.internal_resource_manager_observation_for_test();
    let warmed_cache = ready.device_pass_cache_counts_for_test();
    let mut resources = Vec::new();
    let mut caches = Vec::new();
    for _ in 0..3 {
        pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
            &mut surface,
            scene,
            filters.to_vec(),
            Parameters::default(),
            WorkingFormat::ReducedPrecision,
        ))
        .expect("repeated spatial-filter retained-resource frames must succeed");
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("repeated spatial-filter frames must keep their device");
        resources.push(ready.internal_resource_manager_observation_for_test());
        caches.push(ready.device_pass_cache_counts_for_test());
    }
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated spatial-filter publication must remain readable");

    color_filter_repeated_resource_observations_are_stable_for_test(&resources, &warmed_resources)
        && resources.iter().all(|actual| {
            actual.gaussian_kernel_count_for_test()
                == warmed_resources.gaussian_kernel_count_for_test()
        })
        && warmed_resources.gaussian_kernel_count_for_test() > 0
        && warmed_resources.effect_texture_count_for_test() > 0
        && warmed_cache.has_render_pipelines()
        && caches.iter().all(|actual| *actual == warmed_cache)
        && spatial_filter_maximum_error_for_test(
            output.rgba(),
            expected,
            WorkingFormat::ReducedPrecision,
        ) <= (4, 4)
}

fn spatial_filter_zero_budget_releases_all_frame_resources_for_test(
    scene: &Scene,
    filters: &[FilterList],
    size: PhysicalSize,
    expected: &[u8],
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("spatial-filter zero-budget coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("spatial-filter zero-budget coverage requires a surface");
    pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
        &mut surface,
        scene,
        filters.to_vec(),
        Parameters::default(),
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the first spatial-filter zero-budget frame must succeed");
    let first = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first spatial-filter zero-budget publication must be readable");
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the first spatial-filter zero-budget frame must keep its device")
        .device_pass_cache_counts_for_test();
    pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
        &mut surface,
        scene,
        filters.to_vec(),
        Parameters::default(),
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the repeated spatial-filter zero-budget frame must succeed");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the repeated spatial-filter zero-budget frame must keep its device");
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let second = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated spatial-filter zero-budget publication must be readable");

    resources.leased_count == 0
        && resources.idle_count == 0
        && resources.active_frame_count == 0
        && resources.resolved_lease_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
        && resources.accounted_entry_bytes == Some(0)
        && resources.committed_transient_buffer_count_for_test() == 0
        && resources.committed_transient_image_count_for_test() == 0
        && resources.effect_texture_count_for_test() == 0
        && resources.gaussian_kernel_count_for_test() == 0
        && cache_before == cache_after
        && cache_after.has_render_pipelines()
        && first.rgba() == second.rgba()
        && spatial_filter_maximum_error_for_test(
            second.rgba(),
            expected,
            WorkingFormat::ReducedPrecision,
        ) <= (4, 4)
}

#[test]
fn spatial_filter_fixture_executes_while_public_capabilities_remain_diagnostic() {
    let (scene, filters, size, expected) = spatial_filter_mixed_filter_fixture_for_test();
    let (mut renderer, mut surface) =
        graph_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, size);
    let frame = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &scene,
        filters.clone(),
        WorkingFormat::ReducedPrecision,
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the executed spatial-filter fixture must retain its ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let publication_before = surface.headless_publication_count_for_test();
    let blur = spatial_filter_public_spatial_graph_diagnostic_for_test(
        &scene,
        FilterOp::blur(FilterBlur::try_new(0.75).unwrap()),
        size,
    );
    let shadow = spatial_filter_public_spatial_graph_diagnostic_for_test(
        &scene,
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-1.25, 0.5),
                FilterBlur::try_new(0.5).unwrap(),
                Color::BLACK,
            )
            .unwrap(),
        ),
        size,
    );
    let (resources_after, cache_after) = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("public spatial-filter diagnostics must retain the ready device");
        (
            ready.internal_resource_manager_observation_for_test(),
            ready.device_pass_cache_counts_for_test(),
        )
    };

    assert!(
        spatial_filter_frame_has_exact_execution_for_test(
            &frame,
            size,
            WorkingFormat::ReducedPrecision
        ) && spatial_filter_pixels_match_oracle_for_test(&frame, &expected)
            && blur
                == Some(UnsupportedPrimitive::new(
                    PrimitiveFamily::Filters,
                    PrimitiveOperation::GpuBlurFilterExecution,
                ))
            && shadow
                == Some(UnsupportedPrimitive::new(
                    PrimitiveFamily::Filters,
                    PrimitiveOperation::GpuDropShadowFilterExecution,
                ))
            && retained_public_filter_diagnostics_are_exact_for_test()
            && repeated_spatial_filter_resources_are_stable_for_test(
                &scene, &filters, size, &expected,
            )
            && spatial_filter_zero_budget_releases_all_frame_resources_for_test(
                &scene, &filters, size, &expected,
            )
            && resources_after == resources_before
            && cache_after == cache_before
            && surface.headless_publication_count_for_test() == publication_before,
        "the spatial-filter fixture did not execute while public capabilities remained diagnostic"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn render_window_smoke_executes_gaussian_and_drop_shadow_fixture() {
    let (scene, filters, size, expected) = spatial_filter_mixed_filter_fixture_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("presented spatial-filter coverage requires a renderer: {error}")
    });
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(f64::from(size.width()), f64::from(size.height())),
            format: Format::Rgba8,
            ..SurfaceOptions::default()
        },
    );
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface)).unwrap_or_else(
        |error| panic!("presented spatial-filter coverage must configure: {error}"),
    );
    let presentation = presented_observation_handle_for_test(&surface);
    let rendered = pollster::block_on(renderer.render_spatial_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters,
        Parameters::default(),
        WorkingFormat::ReducedPrecision,
    ));
    let presented = take_last_presented_texture_for_test(&mut surface)
        .and_then(|texture| {
            pollster::block_on(renderer.read_render_texture_for_test(&texture, size)).ok()
        })
        .map(|image| image.into_rgba());
    let presentation = presentation.snapshot_for_test();
    let pixels_match = presented.as_deref().is_some_and(|actual| {
        spatial_filter_maximum_error_for_test(actual, &expected, WorkingFormat::ReducedPrecision)
            <= (4, 4)
    });

    assert!(
        rendered.as_ref().is_ok_and(|frame| {
            frame.working_format == WorkingFormat::ReducedPrecision
                && frame.output_extent == size
                && frame.stats == renderer.stats()
        }) && presentation.acquire_count_for_test() == 1
            && presentation.present_count_for_test() == 1
            && presentation.discarded_count_for_test() == 0
            && pixels_match,
        "the presented fixture did not execute Gaussian blur and drop shadow atomically"
    );
}

#[test]
fn public_dispatch_routes_composition_and_spatial_filters_but_rejects_broad_backdrop() {
    let (scene, filters, size, expected) = spatial_filter_mixed_filter_fixture_for_test();
    let (mut renderer, mut spatial_filter_surface) =
        graph_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, size);
    let spatial_filter = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut spatial_filter_surface,
        &scene,
        filters,
        WorkingFormat::ReducedPrecision,
    );
    let (composition_scene, composition_size, _, _) = composition_reuse_scene_and_oracle_for_test();
    let mut composition_surface = pollster::block_on(renderer.create_headless(
        Size::new(
            f64::from(composition_size.width()),
            f64::from(composition_size.height()),
        ),
        1.0,
    ))
    .expect("masked-composition dispatch coverage requires a surface");
    let composition = pollster::block_on(renderer.render(
        &mut composition_surface,
        &composition_scene,
        Parameters::default(),
    ));
    let unsupported_scene = color_filter_unsupported_backdrop_scene_for_test();
    let mut unsupported_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
            .expect("broad-backdrop rejection coverage requires a surface");
    let unsupported = pollster::block_on(renderer.render(
        &mut unsupported_surface,
        &unsupported_scene,
        Parameters::default(),
    ));
    let expected_backdrop = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );

    assert!(
        spatial_filter_pixels_match_oracle_for_test(&spatial_filter, &expected)
            && composition.is_ok()
            && unsupported
                .as_ref()
                .is_err_and(|error| error.unsupported_primitive() == Some(expected_backdrop))
            && unsupported_surface.headless_publication_count_for_test() == 0,
        "public dispatch misrouted masked composition, spatial filters, or broad backdrop"
    );
}

fn graph_encoding_backend_for_test() -> (Backend, DeviceSlotIdentity) {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("color-filter ordered encoding requires backend selection")
        .unwrap_or_panic_for_test("color-filter ordered encoding requires a host adapter");
    (backend, identity)
}

#[test]
fn color_filter_graph_encodes_fused_operations_in_authored_order() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.ordered_color_filter_graph_encoding_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test("the color-filter fixture must reach its shared GPU graph executor");

    assert!(
        observed.color_pass_count == 2
            && observed.fused_runs_preserve_authored_order
            && observed.binds_exact_source_spatial_and_operations
            && observed.uses_validated_viewport_and_scissor
            && observed.releases_every_resource_at_last_use,
        "the scheduler has no ordered GPU color-filter pass"
    );
}

#[test]
fn color_filter_pass_uses_distinct_source_and_result() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.ordered_color_filter_graph_encoding_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test(
        "the color-filter alias fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.color_pass_count == 2
            && observed.source_and_result_are_distinct
            && observed.binds_exact_source_spatial_and_operations
            && observed.releases_every_resource_at_last_use,
        "the color-filter pass aliases source and result"
    );
}

#[test]
fn multiple_color_runs_share_one_graph_encoder_and_transaction_commit() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.ordered_color_filter_graph_encoding_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            composition_commands_for_test(),
            composition_frame_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test(
        "the ordered color-filter transaction must reach one shared graph executor",
    );
    assert!(
        observed.color_pass_count == 2
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "ordered color filtering split the frame transaction"
    );
}

#[test]
fn oversized_color_filter_buffer_preserves_resources_cache_and_publication() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.color_filter_oversized_buffer_preservation_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test(
        "the oversized color-filter fixture must reject through immutable preparation",
    );
    assert!(
        observed.returns_exact_limit_error
            && observed.resources_are_unchanged
            && observed.cache_is_unchanged
            && observed.publication_is_unchanged,
        "the color-filter limit rejection changed GPU or published state"
    );
}

fn composition_mask_image_for_test(
    size: PhysicalSize,
    byte_seed: u8,
    quality: ImageQuality,
    extend: Extend,
) -> Image {
    let byte_len = usize::try_from(size.width())
        .unwrap()
        .checked_mul(usize::try_from(size.height()).unwrap())
        .and_then(|pixels| pixels.checked_mul(4))
        .unwrap();
    let mut bytes = vec![byte_seed; byte_len];
    for alpha in bytes.iter_mut().skip(3).step_by(4) {
        *alpha = 255;
    }
    Image::from_rgba(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        bytes,
    )
    .unwrap()
    .quality(quality)
    .extend(extend)
}

fn composition_single_mask_composition_commands_for_test(
    image: Image,
    bounds: Rect,
    transform: Transform,
    opacity: f32,
    blend: BlendMode,
    with_clip: bool,
) -> command::RenderCommands {
    let mut layer = Layer::new()
        .try_transform(transform)
        .unwrap()
        .try_opacity(opacity)
        .unwrap()
        .blend(blend)
        .with_resolved_alpha_mask(ResolvedLayerAlphaMask::try_new(image, bounds).unwrap());
    if with_clip {
        layer = layer
            .try_clip(Shape::rect(Rect::new(-3.0, -2.0, 12.0, 9.0)))
            .unwrap();
    }
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(-1.0, 0.5, 6.0, 3.0), Color::BLACK);
    });
    scene.normalize(Capabilities::CURRENT).unwrap()
}

fn composition_frame_context_for_test() -> super::frame::FrameContext {
    super::frame::FrameContext::try_new(
        Size::new(64.0, 48.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap()
}

#[test]
fn mask_upload_allocation_uses_image_extent_not_local_bounds() {
    let image = composition_mask_image_for_test(
        PhysicalSize::new(3, 2),
        17,
        ImageQuality::Medium,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    for (bounds, content) in [
        (
            Rect::new(-40.0, 10.0, 37.0, 19.0),
            Rect::new(0.0, 0.0, 8.0, 6.0),
        ),
        (
            Rect::new(12.0, -9.0, 23.0, 31.0),
            Rect::new(1.0, 2.0, 11.0, 4.0),
        ),
    ] {
        scene.layer(
            Layer::new().with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(image.clone(), bounds).unwrap(),
            ),
            |scene| {
                scene.fill(content, Color::BLACK);
            },
        );
    }
    let observed = super::pass::mask_upload_allocation_observation_for_test(
        scene.normalize(Capabilities::CURRENT).unwrap(),
        composition_frame_context_for_test(),
    );

    assert!(
        observed.retained_upload_count == 1
            && observed.allocation_extents == [PhysicalSize::new(3, 2)],
        "mask allocation still aliases semantic bounds"
    );
}

fn expected_composite_parameter_bytes_for_test() -> [u8; 112] {
    fn write_f32(bytes: &mut [u8; 112], offset: usize, value: f32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn write_u32(bytes: &mut [u8; 112], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    let mut bytes = [0_u8; 112];
    for (offset, value) in [0.48_f32, -0.16, 0.08, 0.64].into_iter().enumerate() {
        write_f32(&mut bytes, offset * 4, value);
    }
    write_f32(&mut bytes, 16, -1.12);
    write_f32(&mut bytes, 20, 3.04);
    for (offset, value) in [-2.5_f32, 1.25, 7.5, 3.75].into_iter().enumerate() {
        write_f32(&mut bytes, 32 + offset * 4, value);
    }
    write_u32(&mut bytes, 48, 3);
    write_u32(&mut bytes, 52, 2);
    for (offset, value) in [1.0_f32 / 6.0, 0.25, 1.0 / 3.0, 0.5]
        .into_iter()
        .enumerate()
    {
        write_f32(&mut bytes, 64 + offset * 4, value);
    }
    write_f32(&mut bytes, 80, 1.0);
    write_u32(&mut bytes, 84, 3);
    write_u32(&mut bytes, 88, 2);
    write_u32(&mut bytes, 92, 2);
    write_u32(&mut bytes, 96, 1);
    write_u32(&mut bytes, 100, 1);
    bytes
}

#[test]
fn composite_parameter_bytes_preserve_affine_mask_mapping_quality_and_extend() {
    let image = composition_mask_image_for_test(
        PhysicalSize::new(3, 2),
        41,
        ImageQuality::High,
        Extend::Reflect,
    );
    let commands = composition_single_mask_composition_commands_for_test(
        image,
        Rect::new(-2.5, 1.25, 7.5, 3.75),
        Transform::try_new([2.0, 0.5, -0.25, 1.5, 3.0, -4.0]).unwrap(),
        1.25,
        BlendMode::Overlay,
        true,
    );
    let observed = super::pass::composite_parameter_bytes_for_test(
        commands,
        composition_frame_context_for_test(),
    );

    assert!(
        observed == Some(expected_composite_parameter_bytes_for_test()),
        "composite bytes lost typed mask mapping or sampling"
    );
}

#[test]
fn zero_sized_mask_image_annihilates_without_texture_allocation() {
    let zero_image = Image::from_rgba(Size::new(0.0, 7.0), Vec::<u8>::new()).unwrap();
    let descriptor = ResolvedMaskUploadDescriptor::try_from_image(zero_image).unwrap();

    assert!(
        super::resource::ResourceAllocationPreflight::zero_sized_mask_is_explicitly_empty_for_test(
            &descriptor,
        ),
        "zero mask allocated a substitute texture"
    );
}

#[test]
fn mask_pipeline_keys_exclude_image_identity() {
    let first = composition_single_mask_composition_commands_for_test(
        composition_mask_image_for_test(
            PhysicalSize::new(4, 3),
            13,
            ImageQuality::Medium,
            Extend::Repeat,
        ),
        Rect::new(-1.0, 2.0, 8.0, 6.0),
        Transform::translation(2.0, -3.0).unwrap(),
        0.75,
        BlendMode::Screen,
        false,
    );
    let second = composition_single_mask_composition_commands_for_test(
        composition_mask_image_for_test(
            PhysicalSize::new(4, 3),
            29,
            ImageQuality::Medium,
            Extend::Repeat,
        ),
        Rect::new(-1.0, 2.0, 8.0, 6.0),
        Transform::translation(2.0, -3.0).unwrap(),
        0.75,
        BlendMode::Screen,
        false,
    );

    assert!(
        super::pass::mask_pipeline_keys_exclude_image_identity_for_test(
            first,
            second,
            composition_frame_context_for_test(),
        ),
        "pipeline caching is keyed by retained image identity"
    );
}

#[test]
fn resource_preparation_is_allocation_safe_and_submission_free() {
    let options = Options::default()
        .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
        .with_resource_cache_budget(ResourceCacheBudget::new(1024 * 1024));
    let mut renderer = pollster::block_on(Renderer::new(options)).unwrap_or_panic_for_test(
        "resource preparation coverage requires a real selected WGPU device",
    );
    let surface = pollster::block_on(renderer.create_headless(Size::new(16.0, 12.0), 1.0))
        .unwrap_or_panic_for_test(
            "resource preparation coverage requires a device-backed headless surface",
        );
    let stats_before = renderer.stats();
    let capabilities_before = renderer.runtime_capabilities(&surface);
    let surface_state_before = surface.resource_state();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test("resource preparation coverage requires one ready device bundle")
        .internal_resource_manager_observation_for_test();
    let observed = renderer
        .resource_preparation_observation_for_test(
            composition_commands_for_test(),
            Size::new(16.0, 12.0),
            1.0,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
            Format::Rgba8,
        )
        .unwrap_or_panic_for_test(
            "the representative graph must reach the preparation observation",
        );

    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test("resource preparation must leave the selected device ready")
        .internal_resource_manager_observation_for_test();
    let public_state_unchanged = renderer.stats() == stats_before
        && renderer.runtime_capabilities(&surface) == capabilities_before
        && surface.resource_state() == surface_state_before
        && surface.last_parameters.is_none();
    let bounded_after_cleanup = resources_before.leased_count == 0
        && resources_after.leased_count == 0
        && resources_after.active_frame_count == 0
        && resources_after.resolved_lease_count == 0
        && resources_after.accounted_entry_bytes == Some(resources_after.retained_bytes)
        && resources_after.retained_bytes <= options.resource_cache_budget().bytes();

    assert!(
        observed.complete_resource_and_pass_handoff
            && observed.exact_capture_coverage_working_and_mask_allocations
            && observed.typed_bindings_and_last_use_releases
            && observed.spatial_bytes_and_cache_keys_preserved
            && observed.allocation_preflight_is_atomic
            && observed.failure_and_drop_cleanup
            && observed.repeated_reuse_is_exact_and_bounded
            && observed.populated_pass_cache_is_preserved
            && public_state_unchanged
            && bounded_after_cleanup,
        "the graph has no complete resource and pass preparation handoff: observed={observed:?}, resources_before={resources_before:?}, resources_after={resources_after:?}, public_state_unchanged={public_state_unchanged}, bounded_after_cleanup={bounded_after_cleanup}"
    );
}

#[test]
fn resource_budget_and_device_loss_preserve_public_stats_contract() {
    let commands = composition_commands_for_test();
    let route_before = resource_lifecycle_route_for_test(commands.clone());

    let ordinary_options = Options::default()
        .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
        .with_resource_cache_budget(ResourceCacheBudget::new(1024 * 1024));
    let mut ordinary = pollster::block_on(Renderer::new(ordinary_options))
        .unwrap_or_panic_for_test(
            "ordinary-budget preparation coverage requires a real selected WGPU device",
        );
    let ordinary_surface = pollster::block_on(ordinary.create_headless(Size::new(16.0, 12.0), 1.0))
        .unwrap_or_panic_for_test(
            "ordinary-budget coverage requires a device-backed headless surface",
        );
    let ordinary_stats = ordinary.stats();
    let ordinary_capabilities = ordinary.runtime_capabilities(&ordinary_surface);
    let ordinary_observation = ordinary
        .resource_preparation_observation_for_test(
            commands.clone(),
            Size::new(16.0, 12.0),
            1.0,
            Color::BLACK,
            Format::Rgba8,
        )
        .unwrap_or_panic_for_test("ordinary-budget preparation must reach the resource handoff");
    let ordinary_resources = ordinary
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test("ordinary-budget preparation must retain one ready device")
        .internal_resource_manager_observation_for_test();
    let ordinary_public_unchanged = ordinary.stats() == ordinary_stats
        && ordinary.runtime_capabilities(&ordinary_surface) == ordinary_capabilities
        && ordinary_surface.resource_state() == SurfaceResourceState::PendingAllocation;

    let disabled_options =
        ordinary_options.with_resource_cache_budget(ResourceCacheBudget::DISABLED);
    let mut disabled = pollster::block_on(Renderer::new(disabled_options))
        .unwrap_or_panic_for_test(
            "zero-budget preparation coverage requires a real selected WGPU device",
        );
    let disabled_surface = pollster::block_on(disabled.create_headless(Size::new(16.0, 12.0), 1.0))
        .unwrap_or_panic_for_test("zero-budget coverage requires a device-backed headless surface");
    let disabled_stats = disabled.stats();
    let disabled_capabilities = disabled.runtime_capabilities(&disabled_surface);
    let disabled_observation = disabled
        .resource_preparation_observation_for_test(
            commands.clone(),
            Size::new(16.0, 12.0),
            1.0,
            Color::BLACK,
            Format::Rgba8,
        )
        .unwrap_or_panic_for_test("zero-budget preparation must reach the resource handoff");
    let zero_budget_resources = disabled
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_panic_for_test(
            "zero-budget preparation must retain one ready device before loss",
        )
        .internal_resource_manager_observation_for_test();
    let disabled_public_before_loss = disabled.stats() == disabled_stats
        && disabled.runtime_capabilities(&disabled_surface) == disabled_capabilities
        && disabled_surface.resource_state() == SurfaceResourceState::PendingAllocation;

    disabled.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    disabled.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let terminal_capabilities = disabled.runtime_capabilities(&disabled_surface);
    let terminal_cleanup_once = disabled.default_device_renderer_released_for_test()
        && disabled.default_device_renderer_released_for_test();
    let route_after = resource_lifecycle_route_for_test(commands);

    assert!(
        ordinary_observation.failure_and_drop_cleanup
            && disabled_observation.failure_and_drop_cleanup
            && ordinary_public_unchanged
            && disabled_public_before_loss
            && prepared_resources_are_resolved_and_bounded(
                &ordinary_resources,
                ordinary_options.resource_cache_budget(),
            )
            && prepared_resources_are_fully_released(&zero_budget_resources)
            && disabled.stats() == disabled_stats
            && terminal_capabilities
                == RuntimeCapabilities::Unavailable(
                    RuntimeCapabilityUnavailableReason::DeviceLost {
                        reason: DeviceLossReason::Destroyed,
                    },
                )
            && terminal_cleanup_once
            && route_after == route_before,
        "resource lifecycle leaked into final public stats"
    );
}

fn resource_lifecycle_route_for_test(
    commands: command::RenderCommands,
) -> super::frame::FramePlanResultObservation {
    super::frame::frame_plan_result_observation_for_test(
        commands,
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::BLACK,
    )
}

fn prepared_resources_are_resolved_and_bounded(
    resources: &super::resource::ResourceManagerObservationForTest,
    budget: ResourceCacheBudget,
) -> bool {
    resources.leased_count == 0
        && resources.active_frame_count == 0
        && resources.resolved_lease_count == 0
        && resources.accounted_entry_bytes == Some(resources.retained_bytes)
        && resources.retained_bytes <= budget.bytes()
}

fn prepared_resources_are_fully_released(
    resources: &super::resource::ResourceManagerObservationForTest,
) -> bool {
    prepared_resources_are_resolved_and_bounded(resources, ResourceCacheBudget::DISABLED)
        && resources.idle_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
}

#[test]
fn pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod() {
    let source_extent = PhysicalSize::new(0x0102_0304, 0x0a0b_0c0d);
    let destination_extent = PhysicalSize::new(0x1020_3040, 0x5060_7080);
    let serialized = pass_spatial_uniform_bytes_for_test(
        Point::new(1.5, -2.25),
        2.5,
        source_extent,
        Point::new(-4.5, 8.25),
        0.5,
        destination_extent,
    );
    let expected = [
        0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x90, 0xc0, 0x00, 0x00, 0x04, 0x41, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x03, 0x02, 0x01, 0x0d, 0x0c, 0x0b, 0x0a, 0x40, 0x30, 0x20, 0x10, 0x80,
        0x70, 0x60, 0x50,
    ];
    let exact_layout = serialized.as_ref().is_ok_and(|bytes| {
        bytes == &expected && bytes[12..16] == [0; 4] && bytes[28..32] == [0; 4]
    });

    let finite_overflow_cases = [
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(f64::MAX, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source origin x",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, f64::MAX),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source origin y",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                f64::MAX,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source raster scale",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(f64::MAX, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial destination origin x",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, f64::MAX),
                1.0,
                destination_extent,
            ),
            "pass spatial destination origin y",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                f64::MAX,
                destination_extent,
            ),
            "pass spatial destination raster scale",
        ),
    ];
    let finite_overflow_is_typed = finite_overflow_cases.into_iter().all(|(result, field)| {
        result.as_ref().is_err_and(|error| {
            error.code() == ErrorCode::InvalidInput
                && error.invalid_value_diagnostic().map(InvalidValue::field) == Some(field)
        })
    });

    assert!(
        exact_layout && finite_overflow_is_typed,
        "pass spatial serialization has no explicit 48-byte contract"
    );
}

#[test]
fn pass_spatial_uniform_rejects_f32_underflowing_raster_scales() {
    let underflowing_scale = f64::from_bits(1);
    assert!(underflowing_scale.is_finite() && underflowing_scale > 0.0);
    assert_eq!(underflowing_scale as f32, 0.0);

    let source_error = pass_spatial_uniform_bytes_for_test(
        Point::new(0.0, 0.0),
        underflowing_scale,
        PhysicalSize::new(1, 1),
        Point::new(0.0, 0.0),
        1.0,
        PhysicalSize::new(1, 1),
    );
    let destination_error = pass_spatial_uniform_bytes_for_test(
        Point::new(0.0, 0.0),
        1.0,
        PhysicalSize::new(1, 1),
        Point::new(0.0, 0.0),
        underflowing_scale,
        PhysicalSize::new(1, 1),
    );
    let rejects_scale = |result: &Result<[u8; 48]>, field| {
        result.as_ref().is_err_and(|error| {
            error.code() == ErrorCode::InvalidInput
                && error.invalid_value_diagnostic().map(InvalidValue::field) == Some(field)
        })
    };

    assert!(
        rejects_scale(&source_error, "pass spatial source raster scale")
            && rejects_scale(&destination_error, "pass spatial destination raster scale"),
        "positive f64 raster scale narrowed to zero"
    );
}

#[test]
fn device_pass_cache_starts_with_no_realized_entries() {
    assert!(device_pass_cache_owns_exact_key_spaces_for_test());
}

fn graph_vello_capture_commands_for_test() -> command::RenderCommands {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 1.5, 9.5, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Vello capture raster contract"),
        8.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::try_ink(Rect::new(1.0, 1.0, 8.0, 10.0)).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);
    scene.normalize(Capabilities::CURRENT).unwrap()
}

fn composition_shader_composite_commands_for_test(
    blend: BlendMode,
    has_clip: bool,
    has_mask: bool,
) -> command::RenderCommands {
    let mut layer = Layer::new().try_opacity(0.75).unwrap().blend(blend);
    if has_clip {
        layer = layer
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 12.0, 10.0)))
            .unwrap();
    }
    if has_mask {
        let mask = composition_mask_image_for_test(
            PhysicalSize::new(4, 1),
            53,
            ImageQuality::High,
            Extend::Reflect,
        );
        layer = layer.with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(mask, Rect::new(0.0, 0.0, 4.0, 1.0)).unwrap(),
        );
    }
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 4.0, 1.0),
            Color::try_rgba(0.8, 0.4, 0.2, 1.0).unwrap(),
        );
    });
    if !has_mask {
        let graph_trigger = ResolvedLayerAlphaMask::try_new(
            composition_mask_image_for_test(
                PhysicalSize::new(1, 1),
                97,
                ImageQuality::Low,
                Extend::Pad,
            ),
            Rect::new(7.0, 7.0, 1.0, 1.0),
        )
        .unwrap();
        scene.layer(
            Layer::new().with_resolved_alpha_mask(graph_trigger),
            |scene| {
                scene.fill(Rect::new(7.0, 7.0, 1.0, 1.0), Color::BLACK);
            },
        );
    }
    scene.normalize(Capabilities::CURRENT).unwrap()
}

fn composition_shader_composite_command_variants_for_test() -> Vec<command::RenderCommands> {
    let mut variants = Vec::with_capacity(8);
    for blend in [BlendMode::Normal, BlendMode::Multiply] {
        for (has_clip, has_mask) in [(false, false), (true, false), (false, true), (true, true)] {
            variants.push(composition_shader_composite_commands_for_test(
                blend, has_clip, has_mask,
            ));
        }
    }
    variants
}

fn composition_composite_requests_for_test(
    capabilities: DeviceCapabilities,
    working_format: WorkingFormat,
) -> super::pass::LayerCompositeCacheRequestsForTest {
    super::pass::layer_composite_cache_requests_for_test(
        &composition_shader_composite_command_variants_for_test(),
        composition_frame_context_for_test(),
        capabilities,
        working_format,
    )
    .unwrap()
}

fn composition_selected_backend_and_requests_for_test() -> (
    Backend,
    DeviceSlotIdentity,
    super::pass::LayerCompositeCacheRequestsForTest,
) {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap()
        .unwrap();
    let capabilities = {
        let ready = backend
            .ready_device_state_borrow_for_test(identity)
            .unwrap();
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test())
    };
    let working_format = capabilities
        .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)
        .unwrap();
    let requests = composition_composite_requests_for_test(capabilities, working_format);
    (backend, identity, requests)
}

#[test]
fn composition_graph_encodes_clip_mask_opacity_and_blend_in_authored_order() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_commands_for_test(),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "the composition graph must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.encodes_clip_mask_opacity_and_blend_in_authored_order,
        "layer composition has no one-shot GPU encoding"
    );
}

#[test]
fn normal_composition_uses_fixed_premultiplied_blend_without_parent_sampling() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_shader_composite_commands_for_test(BlendMode::Normal, true, true),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "normal composition must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.normal_uses_fixed_premultiplied_blend && observed.normal_omits_parent_sample,
        "normal composition sampled its parent or used wrong factors"
    );
}

#[test]
fn non_normal_blends_copy_parent_and_never_read_write_one_texture() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_shader_composite_commands_for_test(BlendMode::Multiply, true, true),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "destination sampling must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.destination_copies_full_parent && observed.destination_avoids_read_write_alias,
        "destination sampling aliases its output"
    );
}

#[test]
fn multiple_composites_share_one_graph_encoder_and_transaction_commit() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_commands_for_test(),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "multiple composites must reach their checked one-shot encoding observation: {error:?}"
        ),
    };
    assert!(
        observed.composite_count == 2
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "composition split the frame transaction"
    );
}

#[test]
fn base_graph_shader_cache_realizes_checked_programs_without_publishing_failed_entries() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test(
            "checked base-graph shader realization requires backend selection",
        )
        .unwrap_or_panic_for_test("checked base-graph shader realization requires a host adapter");
    let capabilities = {
        let ready = backend
            .ready_device_state_borrow_for_test(identity)
            .unwrap_or_panic_for_test(
                "checked base-graph shader realization requires a ready device",
            );
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test())
    };
    let working_format = capabilities
        .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)
        .unwrap_or_panic_for_test(
            "checked base-graph shader realization requires one supported working format",
        );
    let commands = graph_shader_commands_for_test();
    let context = graph_shader_frame_context_for_test();
    let rgba_requests = super::pass::core_pass_cache_requests_for_test(
        commands.clone(),
        context,
        capabilities,
        working_format,
        Format::Rgba8,
    )
    .unwrap_or_panic_for_test(
        "RGBA shader realization requires exact lowered base-graph pass keys",
    );
    let bgra_requests = super::pass::core_pass_cache_requests_for_test(
        commands,
        context,
        capabilities,
        working_format,
        Format::Bgra8,
    )
    .unwrap_or_panic_for_test(
        "BGRA shader realization requires exact lowered base-graph pass keys",
    );
    let observed = pollster::block_on(
        backend.core_pass_shader_cache_realization_observation_for_test(
            identity,
            &rgba_requests,
            &bgra_requests,
        ),
    )
    .unwrap_or_panic_for_test(
        "checked base-graph shader realization must reach its transaction observation",
    );

    assert!(
        observed.realizes_all_checked_programs
            && observed.provisional_handles_are_encoding_ready
            && observed.commits_only_after_clean_transaction
            && observed.reuses_exact_committed_entries
            && observed.failed_validation_publishes_none
            && observed.cancellation_publishes_none
            && observed.device_transition_publishes_none
            && observed.specializes_rgba_and_bgra_outputs,
        "base-graph pass objects are not transactionally cached"
    );
}

#[test]
fn composite_cache_realizes_exact_normal_and_destination_sampling_programs() {
    let (mut backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let observed = pollster::block_on(
        backend.layer_composite_cache_realization_observation_for_test(identity, &requests),
    )
    .unwrap();

    assert!(
        observed.realizes_normal_and_destination_programs
            && observed.realizes_all_optional_binding_combinations
            && observed.normal_uses_fixed_premultiplied_source_over
            && observed.destination_uses_replace_blending
            && observed.commits_only_after_clean_transaction
            && observed.reuses_exact_committed_entries
            && observed.failed_validation_publishes_none
            && observed.cancellation_publishes_none
            && observed.device_transition_publishes_none,
        "the compositor has no checked pipeline realization"
    );
}

#[test]
fn composite_layouts_bind_no_dummy_parent_clip_or_mask() {
    let requests = composition_composite_requests_for_test(
        DeviceCapabilities::from_test_facts(true, true, 4_096),
        WorkingFormat::HighPrecision,
    );
    let observed = super::pass::layer_composite_layout_observation_for_test(&requests);

    assert!(
        observed.realizes_all_eight_entry_interfaces
            && observed.normal_omits_parent
            && observed.destination_binds_parent
            && observed.optional_clip_is_exact
            && observed.optional_mask_is_exact
            && observed.binds_only_one_source_sampler
            && observed.binds_only_exact_uniforms,
        "composite layout contains an absent semantic binding"
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_f16_to_f32_for_test(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut unbiased = -14_i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased -= 1;
            }
            normalized &= 0x03ff;
            sign | (u32::try_from(unbiased + 127).unwrap() << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_read_gpu_vectors_for_test(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    working_format: WorkingFormat,
    count: usize,
) -> Result<Vec<[f32; 4]>> {
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector readback did not make bounded device progress",
            )
            .with_source(source)
        })?;
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector map callback missed its diagnostic deadline",
            )
            .with_source(source)
        })?
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector staging buffer could not be mapped",
            )
            .with_source(source)
        })?;
    let mapped = slice.get_mapped_range();
    let stride = usize::try_from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT).unwrap();
    let mut rgba = Vec::with_capacity(count);
    for index in 0..count {
        let offset = index.checked_mul(stride).ok_or_else(|| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector offset overflowed",
            )
        })?;
        let pixel = match working_format {
            WorkingFormat::HighPrecision => {
                let bytes = mapped.get(offset..offset + 8).ok_or_else(|| {
                    Error::new(
                        BackendErrorCode::ReadbackFailed,
                        "high-precision composite-shader vector readback is truncated",
                    )
                })?;
                let mut pixel = [0.0; 4];
                for (channel, pair) in bytes.chunks_exact(2).enumerate() {
                    pixel[channel] =
                        composition_f16_to_f32_for_test(u16::from_le_bytes([pair[0], pair[1]]));
                }
                pixel
            }
            WorkingFormat::ReducedPrecision => {
                let bytes = mapped.get(offset..offset + 4).ok_or_else(|| {
                    Error::new(
                        BackendErrorCode::ReadbackFailed,
                        "reduced-precision composite-shader vector readback is truncated",
                    )
                })?;
                [
                    f32::from(bytes[0]) / 255.0,
                    f32::from(bytes[1]) / 255.0,
                    f32::from(bytes[2]) / 255.0,
                    f32::from(bytes[3]) / 255.0,
                ]
            }
        };
        rgba.push(pixel);
    }
    drop(mapped);
    staging.unmap();
    Ok(rgba)
}

#[cfg(not(target_arch = "wasm32"))]
async fn composition_submit_and_read_gpu_vectors_for_test(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    transaction: super::gpu_transaction::GpuOperationTransaction,
    prepared: CompositionPreparedGpuVectorsForTest,
) -> Result<CompositionGpuVectorResultsForTest> {
    let CompositionPreparedGpuVectorsForTest {
        device,
        queue,
        working_format,
        mut encoder,
        outputs,
        pass_cache_update,
    } = prepared;
    let output_stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging_size = u64::from(output_stride)
        .checked_mul(u64::try_from(outputs.len()).map_err(|_| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector count does not fit the staging address space",
            )
        })?)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector staging size overflowed",
            )
        })?;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Surgeist composite-shader GPU vector readback"),
        size: staging_size,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
        mapped_at_creation: false,
    });
    for (index, output) in outputs.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: u64::from(output_stride) * u64::try_from(index).unwrap(),
                    bytes_per_row: Some(output_stride),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    super::gpu_transaction::test_support::submit_command_buffer_for_test(
        transaction,
        &queue,
        encoder.finish(),
        RuntimeOperation::EffectRendering,
    )
    .await?;
    backend.commit_checked_pass_cache_update_for_test(identity, pass_cache_update)?;
    let rgba =
        composition_read_gpu_vectors_for_test(&device, &staging, working_format, outputs.len())?;
    Ok(CompositionGpuVectorResultsForTest {
        working_format,
        rgba,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_mask_boundary_vectors_for_test()
-> (Vec<CompositionMaskSamplingVectorForTest>, Vec<[f32; 4]>) {
    let mut vectors = Vec::new();
    let mut expected = Vec::new();
    let rows = [
        (
            ImageQuality::Low,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.0, 0.0),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.666_666_7,
        ),
        (
            ImageQuality::Medium,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.5, 0.5),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.5,
        ),
        (
            ImageQuality::High,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.5, 0.5),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.5,
        ),
    ];
    for (quality, extend_rows, vertical_boundary) in rows {
        for (extend, left_boundary, right_boundary) in extend_rows {
            for (point, alpha) in [
                (Point::new(0.0, 0.5), left_boundary),
                (Point::new(4.0, 0.5), right_boundary),
                (Point::new(2.0, 0.0), vertical_boundary),
                (Point::new(2.0, 1.0), vertical_boundary),
                (Point::new(-0.000_1, 0.5), 0.0),
                (Point::new(4.000_1, 0.5), 0.0),
                (Point::new(2.0, -0.000_1), 0.0),
                (Point::new(2.0, 1.000_1), 0.0),
            ] {
                vectors.push(CompositionMaskSamplingVectorForTest {
                    quality,
                    extend,
                    layer_point: point,
                    clip_alpha: None,
                    opacity: 1.0,
                });
                expected.push([alpha; 4]);
            }
        }
    }
    vectors.push(CompositionMaskSamplingVectorForTest {
        quality: ImageQuality::Medium,
        extend: Extend::Repeat,
        layer_point: Point::new(0.0, 0.5),
        clip_alpha: Some(0.5),
        opacity: 0.5,
    });
    expected.push([0.125; 4]);
    (vectors, expected)
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_gpu_vectors_match(
    observed: &CompositionGpuVectorResultsForTest,
    expected: &[[f32; 4]],
    tolerance: f32,
) -> bool {
    observed.rgba.len() == expected.len()
        && observed
            .rgba
            .iter()
            .zip(expected)
            .all(|(actual, expected)| {
                actual.iter().all(|channel| channel.is_finite())
                    && actual[3] >= -tolerance
                    && actual[3] <= 1.0 + tolerance
                    && actual[..3]
                        .iter()
                        .all(|channel| *channel >= -tolerance && *channel <= actual[3] + tolerance)
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
            })
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mask_sampling_shader_matches_independent_boundary_vectors() {
    let (mut backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let (vectors, expected) = composition_mask_boundary_vectors_for_test();
    let observed = pollster::block_on(async {
        let transaction = backend.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let prepared = backend.composition_shader_mask_sampling_preparation_for_test(
            identity,
            &requests,
            &CompositionMaskSamplingInputForTest {
                mask_size: PhysicalSize::new(4, 1),
                mask_rgba: vec![255, 0, 0, 0, 0, 255, 0, 85, 0, 0, 255, 170, 17, 33, 65, 255],
                mask_bounds: Rect::new(0.0, 0.0, 4.0, 1.0),
                source: [1.0; 4],
                vectors,
            },
        )?;
        composition_submit_and_read_gpu_vectors_for_test(
            &mut backend,
            identity,
            transaction,
            prepared,
        )
        .await
    })
    .unwrap();
    let tolerance = match observed.working_format {
        WorkingFormat::HighPrecision | WorkingFormat::ReducedPrecision => 2.0 / 255.0,
    };

    assert!(
        composition_gpu_vectors_match(&observed, &expected, tolerance),
        "GPU mask sampling differs from independent constants"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn blend_shaders_match_independent_known_vectors() {
    let (backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let source = [0.4, 0.1, 0.3, 0.5];
    let parent = [0.2, 0.6, 0.32, 0.8];
    let vectors = [
        CompositionBlendVectorForTest {
            blend: BlendMode::Normal,
            source,
            parent,
            opacity: 1.25,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Multiply,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Screen,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Overlay,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Darken,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Lighten,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Plus,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Plus,
            source: [0.8, 0.2, 0.7, 0.8],
            parent: [0.6, 0.9, 0.4, 0.9],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Multiply,
            source: [0.0; 4],
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Screen,
            source,
            parent: [0.0; 4],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Overlay,
            source: [0.0; 4],
            parent: [0.0; 4],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Normal,
            source,
            parent,
            opacity: -0.25,
        },
    ];
    let expected = [
        [0.5, 0.4, 0.46, 0.9],
        [0.26, 0.38, 0.316, 0.9],
        [0.52, 0.64, 0.524, 0.9],
        [0.34, 0.56, 0.412, 0.9],
        [0.28, 0.4, 0.38, 0.9],
        [0.5, 0.62, 0.46, 0.9],
        [0.6, 0.7, 0.62, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        parent,
        source,
        [0.0; 4],
        parent,
    ];
    assert_composition_blend_gpu_vectors(backend, identity, &requests, &vectors, &expected);
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_composition_blend_gpu_vectors(
    mut backend: Backend,
    identity: DeviceSlotIdentity,
    requests: &super::pass::LayerCompositeCacheRequestsForTest,
    vectors: &[CompositionBlendVectorForTest],
    expected: &[[f32; 4]],
) {
    let observed = pollster::block_on(async {
        let transaction = backend.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let prepared =
            backend.composition_shader_blend_preparation_for_test(identity, requests, vectors)?;
        composition_submit_and_read_gpu_vectors_for_test(
            &mut backend,
            identity,
            transaction,
            prepared,
        )
        .await
    })
    .unwrap();
    let tolerance = match observed.working_format {
        WorkingFormat::HighPrecision | WorkingFormat::ReducedPrecision => 3.0 / 255.0,
    };

    assert!(
        composition_gpu_vectors_match(&observed, expected, tolerance),
        "GPU blend math differs from independent constants"
    );
}

#[test]
fn base_graph_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms() {
    let observed = super::pass::core_pass_layout_observation_for_test(
        graph_shader_commands_for_test(),
        graph_shader_frame_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.canonicalize_binds_capture_and_spatial_only
            && observed.span_source_over_binds_source_and_spatial_only
            && observed.present_binds_final_image_and_spatial_only
            && observed.copy_only_parent_is_not_sampled
            && observed.dummy_parameters_are_not_bound
            && observed.composition_typed_vocabulary_is_preserved
            && observed.output_specialization_is_exact,
        "the base-graph pass layout contains a copy-only or dummy binding"
    );
}

fn observe_graph_custom_spine_encoding_for_test() -> CustomSpineEncodingObservationForTest {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("custom-spine encoding requires backend selection")
        .unwrap_or_panic_for_test("custom-spine encoding requires a host adapter");
    pollster::block_on(backend.custom_spine_encoding_observation_for_test(
        identity,
        graph_shader_commands_for_test(),
        graph_shader_frame_context_for_test(),
        Format::Rgba8,
    ))
    .unwrap_or_panic_for_test("custom-spine encoding must reach its encoding observation")
}

#[test]
fn custom_spine_encodes_clear_canonicalize_copy_source_over_and_present_in_order() {
    let observed = observe_graph_custom_spine_encoding_for_test();

    assert!(
        observed.encodes_custom_passes_in_order
            && observed.clears_full_root_once
            && observed.uses_exact_prepared_spatial_mapping
            && observed.presents_to_exact_external_output
            && observed.exposes_bounded_capture_handoff
            && observed.validates_checked_capture_completion
            && observed.completes_custom_passes_after_encoding
            && observed.keeps_cache_update_provisional
            && observed.encodes_without_submission_or_sync,
        "the custom pass scheduler has no executable ordered spine"
    );
}

#[test]
fn span_source_over_copies_parent_then_uses_fixed_premultiplied_blend() {
    let observed = observe_graph_custom_spine_encoding_for_test();

    assert!(
        observed.parent_and_result_are_distinct
            && observed.copies_full_parent_before_bounded_source_render
            && observed.samples_only_source_with_fixed_premultiplied_blend
            && observed.preserves_signed_source_origin,
        "normal source-over sampled or overwrote its parent incorrectly"
    );
}

#[test]
fn multiple_vello_captures_share_one_graph_encoder_and_transaction_commit() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("multiple Vello captures require backend selection")
        .unwrap_or_panic_for_test("multiple Vello captures require a host adapter");
    let observed = pollster::block_on(
        backend.multiple_vello_capture_encoding_observation_for_test(
            identity,
            graph_shader_commands_for_test(),
            runtime_lowering_commands_for_test(),
            graph_shader_frame_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test("the validated two-capture fixture must reach graph encoding");
    assert!(
        observed.exact_capture_count
            && observed.one_graph_command_encoder
            && observed.one_gpu_transaction
            && observed.one_active_vello_scope
            && observed.aggregate_pending_commit
            && observed.commits_every_capture_after_transaction_success
            && observed.aborts_every_capture_on_drop,
        "bounded Vello captures cannot share one graph transaction"
    );
}

#[test]
fn later_two_capture_encode_failure_aborts_all_leases_and_rejects_retry_without_submission() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("later Vello capture failure requires backend selection")
        .unwrap_or_panic_for_test("later Vello capture failure requires a host adapter");
    let observed = pollster::block_on(backend.two_capture_failure_observation_for_test(
        identity,
        graph_shader_commands_for_test(),
        runtime_lowering_commands_for_test(),
        graph_shader_frame_context_for_test(),
        TwoCaptureFailureForTest::LaterCaptureEncoding,
    ))
    .unwrap_or_panic_for_test("later Vello capture failure must reach its failure observation");
    assert!(
        observed.acquired_capture_lease_count == 1
            && observed.failure_is_reported
            && observed.produces_no_pending_commit
            && observed.retry_is_rejected
            && observed.resource_creation_was_observed
            && observed.remaining_leased_resource_count == 0
            && observed.remaining_resource_count == 0
            && observed.atlas_recovery_outcome == Some(VelloAtlasOutcome::Recreate)
            && observed.transaction_lease_is_released,
        "later two-capture encode failure did not abort every acquired lease and resource"
    );
}

#[test]
fn shared_two_capture_scope_failure_aborts_all_leases_and_rejects_retry_without_submission() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("shared Vello scope failure requires backend selection")
        .unwrap_or_panic_for_test("shared Vello scope failure requires a host adapter");
    let observed = pollster::block_on(backend.two_capture_failure_observation_for_test(
        identity,
        graph_shader_commands_for_test(),
        runtime_lowering_commands_for_test(),
        graph_shader_frame_context_for_test(),
        TwoCaptureFailureForTest::SharedScopeResolution,
    ))
    .unwrap_or_panic_for_test("shared Vello scope failure must reach its failure observation");
    assert!(
        observed.acquired_capture_lease_count == 2
            && observed.failure_is_reported
            && observed.produces_no_pending_commit
            && observed.retry_is_rejected
            && observed.resource_creation_was_observed
            && observed.remaining_leased_resource_count == 0
            && observed.remaining_resource_count == 0
            && observed.atlas_recovery_outcome == Some(VelloAtlasOutcome::Recreate)
            && observed.transaction_lease_is_released,
        "shared two-capture scope failure did not abort every acquired lease and resource"
    );
}

#[test]
fn vello_capture_uses_transparent_base_requested_aa_and_exact_bounded_extent() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("capture raster contracts require backend selection")
        .unwrap_or_panic_for_test("capture raster contracts require a host adapter");
    let commands = graph_vello_capture_commands_for_test();
    let mut contract_is_exact = true;
    for antialiasing in [
        Antialiasing::Area,
        Antialiasing::Msaa8,
        Antialiasing::Msaa16,
    ] {
        let context = super::frame::FrameContext::try_new(
            Size::new(16.0, 12.0),
            1.25,
            antialiasing,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        )
        .unwrap();
        let observed = pollster::block_on(
            backend.vello_capture_raster_contract_observation_for_test(
                identity,
                commands.clone(),
                context,
                antialiasing,
            ),
        )
        .unwrap_or_panic_for_test("the bounded capture must reach its raster contract observation");
        contract_is_exact &= observed.lowers_with_exact_initial_transform
            && observed.uses_transparent_base
            && observed.uses_requested_antialiasing
            && observed.uses_exact_positive_extent
            && observed.uses_exact_rgba8_target_and_view
            && observed.uses_exact_capture_usage
            && observed.has_unforgeable_encoded_capture_proof;
    }

    assert!(
        contract_is_exact,
        "Vello capture changed its raster contract"
    );
}

#[test]
fn capture_failure_aborts_and_rejects_retry_on_new_encoder() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("capture-failure coverage requires backend selection")
        .unwrap_or_panic_for_test("capture-failure coverage requires a host adapter");
    let observed = pollster::block_on(backend.vello_capture_failure_observation_for_test(
        identity,
        graph_shader_commands_for_test(),
        graph_shader_frame_context_for_test(),
        Format::Rgba8,
    ))
    .unwrap_or_panic_for_test("capture-failure coverage must reach its failure observation");
    assert!(
        observed.capture_failure_is_reported
            && observed.complete_pass_is_rejected
            && observed.retry_on_new_encoder_is_rejected,
        "failed capture completed or retried on a new encoder"
    );
}

#[test]
fn bounded_backdrop_render_succeeds_after_complete_frame_validation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(8.0, 6.0), 1.0))
        .unwrap_or_panic_for_test(
            "the pre-execution frame-gate fixture requires a headless surface",
        );
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 8.0, 6.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            scene.fill(Rect::new(1.0, 1.0, 4.0, 3.0), Color::BLACK);
        });

    let result = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()));

    assert!(
        result.is_ok(),
        "bounded backdrop execution must succeed after complete frame validation"
    );
}

#[test]
fn materialized_image_filter_classifier_preserves_mixed_filter_order() {
    let shadow = FilterDropShadow::try_from_shadow(
        Shadow::try_new(Point::new(2.0, 3.0), 4.0, 0.0, Color::BLACK).unwrap(),
    )
    .unwrap();
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(0.8).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::drop_shadow(shadow),
        FilterOp::sepia(UnitFilterAmount::try_new(0.25).unwrap()),
    ])
    .unwrap();

    let pipeline = list
        .materialized_image_filter_pipeline()
        .unwrap_or_panic_for_test("materialized image filters should classify")
        .unwrap_or_panic_for_test("non-empty filter lists should produce a pipeline");

    assert_eq!(pipeline.steps().len(), 5);
    assert!(matches!(
        &pipeline.steps()[0],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [
                    ColorFilterOp::Brightness(FilterAmount::try_new(1.2).unwrap()),
                    ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
                ]
    ));
    assert!(matches!(
        pipeline.steps()[1],
        MaterializedImageFilterStep::Blur(blur) if blur.radius() == 4.0
    ));
    assert!(matches!(
        &pipeline.steps()[2],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap())]
    ));
    assert!(matches!(
        &pipeline.steps()[3],
        MaterializedImageFilterStep::DropShadow(classified) if classified == &shadow
    ));
    assert!(matches!(
        &pipeline.steps()[4],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.25).unwrap())]
    ));
}

#[test]
fn filter_none_has_no_materialized_image_filter_pipeline() {
    assert_eq!(
        FilterList::none()
            .materialized_image_filter_pipeline()
            .unwrap(),
        None
    );
}

#[test]
fn materialized_image_filter_classifier_accepts_blur_and_drop_shadow() {
    let shadow = FilterDropShadow::try_from_shadow(
        Shadow::try_new(Point::new(1.0, 2.0), 3.0, 0.0, Color::BLACK).unwrap(),
    )
    .unwrap();
    let list = FilterList::try_ops(vec![
        FilterOp::blur(FilterBlur::try_new(2.0).unwrap()),
        FilterOp::drop_shadow(shadow),
    ])
    .unwrap();

    let pipeline = list
        .materialized_image_filter_pipeline()
        .unwrap_or_panic_for_test("blur and drop-shadow should classify")
        .unwrap_or_panic_for_test("non-empty materialized filter lists should produce a pipeline");

    assert_eq!(
        pipeline.steps(),
        &[
            MaterializedImageFilterStep::Blur(FilterBlur::try_new(2.0).unwrap()),
            MaterializedImageFilterStep::DropShadow(shadow),
        ]
    );
}

#[test]
fn materialized_filter_classification_does_not_make_resource_handles_bytes() {
    let resource = ResolvedImageResource::try_new(ImageId::new(41), Size::new(8.0, 8.0)).unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();

    assert!(
        paint
            .filters()
            .materialized_image_filter_pipeline()
            .unwrap()
            .is_some()
    );

    let unsupported = paint
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("resource-only filtered image paint is still not materialized bytes");
    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn bounded_backdrop_capture_normalizes_and_executes_through_public_route() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 2.0, 1.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |_| {})
        .fill(
            Rect::new(1.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene
        .normalize(Capabilities::CURRENT)
        .unwrap_or_panic_for_test(
            "backdrop planning should remain inspectable through normalization",
        );
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[1] else {
        panic!("expected normalized backdrop layer");
    };
    assert!(layer.backdrop.is_some());

    assert_bounded_backdrop_filter_execution_is_public(&scene, Size::new(2.0, 1.0));
}

#[test]
fn authored_backdrop_filter_orders_execute_through_public_route() {
    let source_rect = Rect::new(0.0, 0.0, 3.0, 1.0);
    let bounds = BackdropCaptureBounds::try_new(source_rect).unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let mut color_before_blur = Scene::new();
    color_before_blur
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    let mut blur_before_color = Scene::new();
    blur_before_color
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![blur, brightness]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    assert_bounded_backdrop_filter_execution_is_public(&color_before_blur, Size::new(3.0, 1.0));
    assert_bounded_backdrop_filter_execution_is_public(&blur_before_color, Size::new(3.0, 1.0));
}

#[test]
fn clipped_backdrop_filter_executes_through_public_route() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 5.0, 5.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 1.0, 3.0, 3.0),
        Radii::all(1.5),
    ))
    .unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, Some(clip)).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 5.0, 5.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |_| {});

    assert_bounded_backdrop_filter_execution_is_public(&scene, Size::new(5.0, 5.0));
}

#[test]
fn backdrop_foreground_composition_executes_through_public_route() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 3.0, 1.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 3.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |scene| {
            scene.fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK);
        });

    assert_bounded_backdrop_filter_execution_is_public(&scene, Size::new(3.0, 1.0));
}

#[test]
fn bounded_backdrop_normalization_orders_capture_before_execution() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 3.0, 1.0)).unwrap();
    let backdrop_layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(backdrop_layer, |scene| {
            scene.fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK);
        })
        .fill(
            Rect::new(2.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[1] else {
        panic!("expected bounded backdrop layer command");
    };
    let capture = layer
        .backdrop
        .as_ref()
        .unwrap_or_panic_for_test("backdrop capture planned");
    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::BoundedBackdropCapture
    );
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::OffscreenTexture
    );
    assert_eq!(capture.capture_bounds().rect(), bounds.rect());
    assert!(matches!(
        normalized.commands[0],
        command::RenderCommand::Fill { .. }
    ));
    assert_eq!(children.len(), 1);
    assert!(matches!(
        normalized.commands[2],
        command::RenderCommand::Fill { .. }
    ));

    assert_bounded_backdrop_filter_execution_is_public(&scene, Size::new(3.0, 1.0));
}

#[test]
fn backdrop_filter_chain_preserves_authored_inputs() {
    let source_rect = Rect::new(0.0, 0.0, 3.0, 1.0);
    let bounds = BackdropCaptureBounds::try_new(source_rect).unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let mut color_before_blur = Scene::new();
    color_before_blur
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );
    let mut blur_before_color = Scene::new();
    blur_before_color
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![blur, brightness]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    assert_bounded_backdrop_filter_execution_is_public(&color_before_blur, Size::new(3.0, 1.0));
    assert_bounded_backdrop_filter_execution_is_public(&blur_before_color, Size::new(3.0, 1.0));

    let clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 1.0, 3.0, 3.0),
        Radii::all(1.5),
    ))
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let clipped_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 5.0, 5.0)).unwrap(),
                Some(clip),
            )
            .unwrap(),
        )
        .unwrap();
    let mut clipped_scene = Scene::new();
    clipped_scene
        .fill(
            Rect::new(0.0, 0.0, 5.0, 5.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(clipped_layer, |_| {});
    assert_bounded_backdrop_filter_execution_is_public(&clipped_scene, Size::new(5.0, 5.0));
}

#[test]
fn supported_mix_blend_modes_use_direct_vello_and_extra_modes_are_typed() {
    let blend_modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    assert_eq!(
        blend_modes.len(),
        7,
        "public layer BlendMode additions require encoding and reference coverage"
    );

    let source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for mode in blend_modes {
        let mut scene = Scene::new();
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(destination),
        );
        scene.layer(Layer::new().blend(mode), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(source),
            );
        });

        let output = render_scene_pixel(&mut renderer, &scene);
        assert_rgba_near_reference_pixel(
            output,
            source.blend_over(destination, mode),
            2,
            &format!("direct Vello blend mode {mode:?} should match reference"),
        );
    }

    let backdrop = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let outer_child_backdrop = PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap();
    let inner_source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let expected_inner = inner_source.blend_over(outer_child_backdrop, BlendMode::Multiply);
    let expected_outer = expected_inner.blend_over(backdrop, BlendMode::Screen);
    let mut nested_scene = Scene::new();
    nested_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(backdrop),
    );
    nested_scene.layer(Layer::new().blend(BlendMode::Screen), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(outer_child_backdrop),
        );
        scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(inner_source),
            );
        });
    });
    let normalized = nested_scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer: outer, .. } = &normalized.commands[1] else {
        panic!("expected outer blend layer");
    };
    assert_eq!(
        outer.pass_plan.requirement(),
        command::LayerPassRequirement::DirectVelloBlend
    );
    let nested_output = render_scene_pixel(&mut renderer, &nested_scene);
    assert_rgba_near_reference_pixel(
        nested_output,
        expected_outer,
        2,
        "nested direct Vello blend groups stay implemented in command order",
    );

    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::AdditionalMixBlendMode,
    );
    let error = Capabilities::CURRENT
        .ensure_supported(unsupported)
        .expect_err("mix-blend modes outside BlendMode remain diagnostic");
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}

#[test]
fn root_background_and_composite_boundaries_remain_typed() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let root = BackdropFilterInput::try_root_backdrop(filters, None)
        .expect_err("root backdrop policy is not render-owned");
    assert_eq!(
        root.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ))
    );

    let normal_background =
        BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal]).unwrap();
    assert_eq!(normal_background.modes(), &[BackgroundBlendMode::Normal]);
    for mode in [
        BackgroundBlendMode::Multiply,
        BackgroundBlendMode::Screen,
        BackgroundBlendMode::Overlay,
        BackgroundBlendMode::Darken,
        BackgroundBlendMode::Lighten,
        BackgroundBlendMode::Plus,
    ] {
        let error = BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal, mode])
            .expect_err("non-normal background blend lists remain diagnostic");
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::BackgroundBlendMode,
            ))
        );
    }

    let source = PremultipliedRgba8::try_new(80, 40, 20, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 30, 40, 96).unwrap();
    let mask = PremultipliedRgba8::try_new(0, 0, 0, 64).unwrap();
    for pixel in [
        source.blend_over(destination, BlendMode::Normal),
        source.blend_over(destination, BlendMode::Plus),
        source.source_in_alpha_of(mask),
        destination.destination_in_alpha_of(mask),
    ] {
        assert_premultiplied(pixel);
    }

    let porter_duff = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::PorterDuffCompositeMode,
    );
    let error = Capabilities::CURRENT
        .ensure_supported(porter_duff)
        .expect_err("Porter-Duff CSS operators stay behind a typed boundary");
    assert_eq!(error.unsupported_primitive(), Some(porter_duff));

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
            .expect_err("non-default mask composites remain diagnostic");
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
fn shape_and_basic_shape_clips_normalize_and_render_from_owned_geometry() {
    let rect = Rect::new(0.0, 0.0, 2.0, 2.0);
    let rounded = Shape::try_rounded_rect(rect, Radii::try_all(0.5).unwrap()).unwrap();
    let circle = Shape::try_circle(Point::new(1.0, 1.0), 1.0).unwrap();
    let ellipse = Shape::try_ellipse(Point::new(1.0, 1.0), Size::new(1.0, 0.75)).unwrap();
    let clips = [
        (Shape::rect(rect), ClipGeometryKind::Rect(rect)),
        (
            rounded,
            ClipGeometryKind::RoundedRect {
                rect,
                radii: Radii::try_all(0.5).unwrap(),
            },
        ),
        (
            circle,
            ClipGeometryKind::Circle {
                center: Point::new(1.0, 1.0),
                radius: 1.0,
            },
        ),
        (
            ellipse,
            ClipGeometryKind::Ellipse {
                center: Point::new(1.0, 1.0),
                radii: Size::new(1.0, 0.75),
            },
        ),
    ];

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for (shape, expected_geometry) in clips {
        let normalized = ClipInput::try_shape(shape.clone())
            .unwrap()
            .normalize(Capabilities::CURRENT)
            .unwrap();
        assert_eq!(normalized.geometry().kind(), &expected_geometry);

        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(3.0, 2.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(Layer::new().try_clip(shape).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 3.0, 2.0), Color::BLACK);
        });

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("authored shape clips should render through layer clipping");
        let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

        assert!(pixel_alpha(&output, 0, 0) > 0);
        assert_eq!(pixel_alpha(&output, 2, 0), 0);
    }
}

#[test]
fn path_clip_rendering_preserves_authored_fill_rule() {
    fn nested_rect_path() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(5.0, 0.0))
            .line_to(Point::new(5.0, 5.0))
            .line_to(Point::new(0.0, 5.0))
            .close()
            .move_to(Point::new(1.0, 1.0))
            .line_to(Point::new(4.0, 1.0))
            .line_to(Point::new(4.0, 4.0))
            .line_to(Point::new(1.0, 4.0))
            .close();
        path
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut outputs = Vec::new();
    for fill_rule in [FillRule::EvenOdd, FillRule::NonZero] {
        let filled_path = FilledPath::try_new(nested_rect_path(), fill_rule).unwrap();
        let normalized = ClipInput::try_filled_path(filled_path.clone())
            .unwrap()
            .normalize(Capabilities::CURRENT)
            .unwrap();
        assert_eq!(
            normalized.geometry().kind(),
            &ClipGeometryKind::Path(filled_path.clone())
        );

        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(5.0, 5.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(
            Layer::new()
                .try_clip_input(ClipInput::try_filled_path(filled_path).unwrap())
                .unwrap(),
            |scene| {
                scene.fill(Rect::new(0.0, 0.0, 5.0, 5.0), Color::BLACK);
            },
        );

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("path clips should render with their authored fill rule");
        outputs.push(pollster::block_on(renderer.read_headless(&surface)).unwrap());
    }

    assert_eq!(pixel_alpha(&outputs[0], 2, 2), 0);
    assert!(pixel_alpha(&outputs[1], 2, 2) > 0);
}

#[test]
fn resolved_alpha_masks_match_reference_and_rendered_layer_output() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 0, 0, 255, 0, 255, 0, 255],
    )
    .unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 0, 0, 0, 128],
    )
    .unwrap();
    let masked = reference::execute_transitional_resolved_mask_bridge_for_test(
        &source,
        Rect::new(0.0, 0.0, 2.0, 1.0),
        image_from_buffer(mask.clone()),
        Rect::new(0.0, 0.0, 2.0, 1.0),
    )
    .unwrap();
    assert_eq!(masked.rgba(), &[255, 0, 0, 255, 0, 255, 0, 128]);

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask)),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("resolved layer alpha masks should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
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
    assert!(
        capabilities
            .masks_clips()
            .supports_resolved_alpha_mask_execution()
    );
    assert!(capabilities.compositing().supports_layer_opacity());
    assert!(capabilities.compositing().supports_blend_modes());
    assert!(
        capabilities
            .offscreen_pipeline()
            .supports_direct_vello_opacity_isolation()
    );
    assert!(
        capabilities
            .offscreen_pipeline()
            .supports_direct_vello_blend_isolation()
    );
    assert!(capabilities.surfaces().supports_headless_surfaces());
    assert_eq!(
        capabilities.surfaces().supports_web_canvas_surfaces(),
        cfg!(all(feature = "render-web", target_arch = "wasm32"))
    );
}

#[test]
fn offscreen_pipeline_capability_accessors_report_supported_operations() {
    let capabilities = Capabilities::CURRENT.offscreen_pipeline();

    assert!(capabilities.supports_direct_vello_opacity_isolation());
    assert!(capabilities.supports_direct_vello_blend_isolation());
    assert!(!capabilities.supports_offscreen_layer_rendering());
    assert!(capabilities.supports_persistent_effect_resources());
    assert!(capabilities.supports_bounded_vello_capture());
    assert!(capabilities.supports_image_pass_execution());
    assert!(capabilities.supports_composite_pass_execution());
    assert!(capabilities.supports_nested_opacity_composition());
    assert!(!capabilities.supports_mask_execution());
    assert!(!capabilities.supports_layer_filter_execution());
    assert!(!capabilities.supports_broad_backdrop_execution());
}

#[test]
fn backdrop_capability_accessors_claim_only_narrow_materialized_execution() {
    let capabilities = Capabilities::CURRENT.offscreen_pipeline();

    assert!(capabilities.supports_bounded_backdrop_capture());
    assert!(capabilities.supports_bounded_backdrop_filter_execution());
    assert!(!capabilities.supports_backdrop_isolation_composition());
    assert!(!capabilities.supports_broad_backdrop_execution());
}

#[test]
fn blend_capability_accessors_preserve_direct_vello_claims_without_background_blend() {
    let compositing = Capabilities::CURRENT.compositing();
    let offscreen = Capabilities::CURRENT.offscreen_pipeline();

    assert!(compositing.supports_layer_opacity());
    assert!(compositing.supports_blend_modes());
    assert!(offscreen.supports_direct_vello_opacity_isolation());
    assert!(offscreen.supports_direct_vello_blend_isolation());
    assert!(!compositing.supports_root_backdrop_policy());
    assert!(!compositing.supports_background_blend_modes());
    assert!(!compositing.supports_additional_mix_blend_modes());
    assert!(!compositing.supports_porter_duff_composite_modes());
}

#[test]
fn direct_render_reports_stats_and_failed_mask_preserves_them() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("direct-route statistics coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("direct-route statistics coverage requires a headless surface");
    let mut successful_scene = Scene::new();
    successful_scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let successful =
        pollster::block_on(renderer.render(&mut surface, &successful_scene, Parameters::default()))
            .expect("the direct GPU route must succeed");
    assert_eq!(successful.route, Some(RenderRoute::DirectVello));
    assert_eq!(successful.effect_precision, None);
    assert_eq!(successful.vello_passes, 1);
    assert_eq!(
        (
            successful.image_passes,
            successful.composite_passes,
            successful.copy_operations,
            successful.custom_present_passes,
            successful.effect_texture_allocations,
            successful.effect_texture_reuses,
            successful.retained_effect_bytes,
        ),
        (0, 0, 0, 0, 0, 0, 0)
    );

    let mut failing_scene = Scene::new();
    failing_scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
            .expect("the diagnostic mask input must be intrinsically valid"),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        },
    );
    let error =
        pollster::block_on(renderer.render(&mut surface, &failing_scene, Parameters::default()))
            .expect_err("the broad mask boundary must remain an exact failure");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert_eq!(renderer.stats(), successful);
}

#[test]
fn affected_capability_queries_map_one_to_one_to_primitive_operations() {
    let capabilities = Capabilities::CURRENT;
    let offscreen = capabilities.offscreen_pipeline();
    let cases = [
        (
            capabilities.filters().supports_gpu_color_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ),
        (
            capabilities.filters().supports_gpu_blur_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ),
        (
            capabilities
                .filters()
                .supports_gpu_drop_shadow_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ),
        (
            capabilities
                .masks_clips()
                .supports_resolved_alpha_mask_execution(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::ResolvedAlphaMaskExecution,
        ),
        (
            offscreen.supports_image_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::ImagePassExecution,
        ),
        (
            offscreen.supports_composite_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::CompositePassExecution,
        ),
        (
            offscreen.supports_nested_opacity_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::NestedOpacityComposition,
        ),
    ];

    for (query, family, operation) in cases {
        assert_eq!(
            capabilities
                .ensure_supported(UnsupportedPrimitive::new(family, operation))
                .is_ok(),
            query,
            "capability query must map one-to-one to {operation:?}",
        );
    }
}

#[test]
fn offscreen_pipeline_capability_diagnostics_report_unsupported_operations() {
    for operation in [
        PrimitiveOperation::OffscreenLayerRendering,
        PrimitiveOperation::MaskExecution,
        PrimitiveOperation::LayerFilterExecution,
        PrimitiveOperation::BroadBackdropExecution,
        PrimitiveOperation::BackdropIsolationComposition,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::OffscreenPipeline, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("offscreen pipeline operation is not implemented in this phase");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("offscreen pipeline"));
        assert!(error.message().contains(unsupported.label()));
    }
    Capabilities::CURRENT
        .ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropFilterExecution,
        ))
        .expect(
            "bounded backdrop filter execution is the narrow implemented bounded-backdrop subset",
        );
}

#[test]
fn vello_baseline_reports_current_unsupported_primitives() {
    let capabilities = Capabilities::CURRENT;
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
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_unsupported_off_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    let error = Capabilities::CURRENT
        .ensure_supported(unsupported)
        .expect_err("web canvas surfaces require render-web on wasm32");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message().contains("web canvas surface"));
}

#[cfg(all(feature = "render-web", target_arch = "wasm32"))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_supported_on_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    Capabilities::CURRENT
        .ensure_supported(unsupported)
        .expect("web canvas surfaces are available with render-web on wasm32");
}

#[test]
fn create_headless_rejects_physical_size_overflow() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error = match pollster::block_on(
        renderer.create_headless(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0),
    ) {
        Ok(_) => panic!("physical device pixels should fit in u32"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn text_fill_paint_matches_concrete_render_paint_surface() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(4.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let cases = [
        ("solid color", Paint::color(Color::BLACK)),
        ("gradient", Paint::gradient(gradient)),
        ("image", Paint::image(image)),
    ];

    for (label, paint) in cases {
        let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
        let run = TextRun::try_new(
            FontRef::new(1).named(label),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(paint.clone()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.text_run(run);

        let brush = glyph_paint_brush(&paint)
            .unwrap_or_else(|_| panic!("{label} should encode as a glyph brush"));
        match (&paint, brush) {
            (paint, peniko::Brush::Solid(_)) if paint == &Paint::color(Color::BLACK) => {}
            (paint, peniko::Brush::Gradient(_))
                if matches!(paint.kind(), paint::PaintKind::Gradient(_)) => {}
            (paint, peniko::Brush::Image(_))
                if matches!(paint.kind(), paint::PaintKind::Image(_)) => {}
            _ => panic!("{label} encoded to the wrong glyph brush kind"),
        }

        let normalized = scene
            .normalize(Capabilities::CURRENT)
            .unwrap_or_else(|_| panic!("{label} text fill should normalize"));

        match &normalized.commands[0] {
            command::RenderCommand::TextRun {
                paint: text_paint,
                glyphs,
                ..
            } => {
                assert_eq!(text_paint.fill(), &paint);
                assert_eq!(glyphs.len(), 1);
            }
            command => panic!("{label} should normalize to a text run, got {command:?}"),
        }
    }
}

#[test]
fn ahem_font_data_renders_ascent_and_descent_glyph_bands() {
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 1.0, 9.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 13.0, 9.0, 10.0).unwrap(),
    ];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            ahem_font("Ahem ascent and descent bands"),
            10.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(25.0, 12.0), 1.0)).unwrap();
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("required headless text rendering needs an available host adapter");
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("required headless text readback must complete");

    assert!(
        pixel_alpha(&output, 6, 5) > 200,
        "E-acute gid 100 should paint the ascent band"
    );
    assert_eq!(
        pixel_alpha(&output, 6, 10),
        0,
        "E-acute gid 100 should not paint the descent band"
    );
    assert!(
        pixel_alpha(&output, 18, 10) > 200,
        "p gid 82 should paint the descent band"
    );
    assert_eq!(
        pixel_alpha(&output, 18, 5),
        0,
        "p gid 82 should not paint the ascent band"
    );
}

#[test]
fn matrix_full_background_box_image_text_stack_preserves_render_order() {
    let areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 64.0, 32.0),
        Rect::new(4.0, 4.0, 56.0, 24.0),
        Rect::new(8.0, 8.0, 48.0, 16.0),
    )
    .unwrap();
    let background_image =
        Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([255; 16])).unwrap();
    let background_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(background_image.clone()).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_length(12.0).unwrap(),
                SizeComponent::try_length(8.0).unwrap(),
            ))
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let background = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(
            Some(Color::try_rgba(0.1, 0.2, 0.3, 1.0).unwrap()),
            vec![background_layer],
        )
        .unwrap(),
        areas,
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();
    let decoration = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Solid, 1.0, Color::TRANSPARENT, 1.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                areas,
                Radii::try_all(3.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::CURRENT)
    .unwrap();
    let decoration_line = TextDecorationLine::try_solid(
        Point::new(10.0, 24.0),
        Point::new(42.0, 24.0),
        1.5,
        Transform::identity(),
        Paint::color(Color::BLACK),
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(71, 12.0, 22.0, 9.0).unwrap()];
    let text = TextRun::try_new(
        FontRef::new(71).named("Matrix paint stack"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            areas.border_box(),
            Color::try_rgba(0.1, 0.2, 0.3, 1.0).unwrap(),
        )
        .image(
            background_image,
            Rect::new(8.0, 8.0, 12.0, 8.0),
            ImageFit::Stretch,
        )
        .stroke(
            areas.border_box(),
            Stroke::try_new(2.0).unwrap(),
            Color::BLACK,
        )
        .text_decoration_line(decoration_line)
        .text_run(text);

    assert_matrix_background_and_decoration(&background, &decoration, areas);
    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    assert_matrix_render_command_order(&normalized, areas);
}

fn assert_matrix_background_and_decoration(
    background: &NormalizedBackgroundStack,
    decoration: &NormalizedBoxDecoration,
    areas: BackgroundAreas,
) {
    assert_eq!(background.commands().len(), 2);
    assert!(matches!(
        background.commands()[0].kind(),
        NormalizedBackgroundCommandKind::ColorFill { .. }
    ));
    let NormalizedBackgroundCommandKind::Layer { layer } = background.commands()[1].kind() else {
        panic!("expected normalized image layer after background color");
    };
    assert_eq!(
        background.commands()[1].clip().rect(),
        Some(areas.padding_box())
    );
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
    assert_eq!(layer.placement().paint_rect(), areas.content_box());
    assert_eq!(
        decoration
            .commands()
            .iter()
            .map(|command| match command.kind() {
                NormalizedBoxDecorationCommandKind::Border(_) => "border",
                NormalizedBoxDecorationCommandKind::Outline(_) => "outline",
            })
            .collect::<Vec<_>>(),
        ["border", "outline"]
    );
}

fn assert_matrix_render_command_order(
    normalized: &command::RenderCommands,
    areas: BackgroundAreas,
) {
    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().strokes, 2);
    assert_eq!(normalized.stats().glyphs, 1);
    let [
        command::RenderCommand::Fill { .. },
        command::RenderCommand::Image { .. },
        command::RenderCommand::Stroke {
            shape: border_shape,
            stroke: border_stroke,
            paint: border_paint,
        },
        command::RenderCommand::Stroke {
            shape: decoration_shape,
            stroke: decoration_stroke,
            paint: decoration_paint,
        },
        command::RenderCommand::TextRun { .. },
    ] = normalized.commands.as_slice()
    else {
        panic!("expected fill, image, border stroke, decoration stroke, and text run in order");
    };
    assert_eq!(
        border_shape,
        &command::RenderStrokeShape::Rect(kurbo::Rect::from(areas.border_box()))
    );
    assert_eq!(border_stroke.width, 2.0);
    assert_eq!(border_paint, &command::RenderPaint::Color(Color::BLACK));
    let command::RenderStrokeShape::Path(decoration_path) = decoration_shape else {
        panic!("expected text decoration to lower to a path stroke");
    };
    assert_eq!(decoration_path.elements().len(), 2);
    assert_eq!(
        decoration_path.elements()[0],
        kurbo::PathEl::MoveTo(kurbo::Point::new(10.0, 24.0))
    );
    assert_eq!(
        decoration_path.elements()[1],
        kurbo::PathEl::LineTo(kurbo::Point::new(42.0, 24.0))
    );
    assert_eq!(decoration_stroke.width, 1.5);
    assert_eq!(decoration_paint, &command::RenderPaint::Color(Color::BLACK));
}

#[test]
fn matrix_full_transform_clip_opacity_image_gradient_stack_plans_layers() {
    let image = Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([255; 16])).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let outer_transform = Transform::translation(3.0, 4.0).unwrap();
    let clip_shape = Shape::rect(Rect::new(2.0, 2.0, 18.0, 14.0));
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_transform(outer_transform).unwrap(),
        |scene| {
            scene.layer(Layer::new().try_clip(clip_shape).unwrap(), |scene| {
                scene.layer(Layer::new().try_opacity(0.625).unwrap(), |scene| {
                    scene.image(image, Rect::new(4.0, 5.0, 8.0, 6.0), ImageFit::Contain);
                    scene.fill(
                        Rect::new(6.0, 7.0, 10.0, 3.0),
                        Paint::gradient(gradient.clone()),
                    );
                });
            });
        },
    );

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();

    assert_eq!(normalized.stats().layers, 3);
    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().fills, 1);
    let [
        command::RenderCommand::Layer {
            layer: transform_layer,
            children: transform_children,
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("expected transform layer at the root");
    };
    assert_eq!(transform_layer.transform, outer_transform);
    assert_eq!(transform_layer.isolation, command::LayerIsolation::None);
    let [
        command::RenderCommand::Layer {
            layer: clip_layer,
            children: clip_children,
        },
    ] = transform_children.as_slice()
    else {
        panic!("expected clip layer inside transform layer");
    };
    assert_eq!(clip_layer.isolation, command::LayerIsolation::ClipOnly);
    assert_eq!(
        clip_layer.pass_plan.kind(),
        command::LayerPassKind::ClipOnly
    );
    assert_eq!(
        clip_layer
            .pass_plan
            .bounds()
            .map(command::OffscreenBounds::rect),
        Some(Rect::new(2.0, 2.0, 18.0, 14.0))
    );
    let [
        command::RenderCommand::Layer {
            layer: opacity_layer,
            children: opacity_children,
        },
    ] = clip_children.as_slice()
    else {
        panic!("expected opacity layer inside clip layer");
    };
    assert_eq!(opacity_layer.opacity, 0.625);
    assert_eq!(
        opacity_layer.isolation,
        command::LayerIsolation::BackendLayer
    );
    assert_eq!(
        opacity_layer.pass_plan.requirement(),
        command::LayerPassRequirement::DirectVelloOpacity
    );
    assert!(matches!(
        opacity_children.as_slice(),
        [
            command::RenderCommand::Image { .. },
            command::RenderCommand::Fill {
                paint: command::RenderPaint::Gradient(_),
                ..
            },
        ]
    ));
}

#[test]
fn matrix_full_effect_stack_diagnostics_stop_at_unsupported_boundaries() {
    let filter_layer = Layer::new()
        .try_filter(Filter::try_blur(4.0).unwrap())
        .unwrap();
    let mut filter_scene = Scene::new();
    filter_scene.layer(filter_layer, |scene| {
        scene.shadow(
            Rect::new(0.0, 0.0, 8.0, 8.0),
            Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        );
    });
    let filter_error = filter_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("layer filters remain a typed full-stack diagnostic boundary");
    assert_eq!(
        filter_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );

    let mut inset_shadow_scene = Scene::new();
    inset_shadow_scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );
    let inset_shadow_error = inset_shadow_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("inset box shadows remain a typed shadow diagnostic boundary");
    assert_eq!(
        inset_shadow_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );

    let mask_layer = Layer::new()
        .try_mask(Shape::rect(Rect::new(0.0, 0.0, 6.0, 6.0)))
        .unwrap();
    let mut mask_scene = Scene::new();
    mask_scene.layer(mask_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    });
    let mask_error = mask_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("authored layer masks remain a typed full-stack diagnostic boundary");
    assert_eq!(
        mask_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );

    let backdrop_filters = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let backdrop = BackdropFilterInput::try_new(
        backdrop_filters,
        BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 8.0)).unwrap(),
        Some(ClipInput::try_shape(Shape::rect(Rect::new(1.0, 1.0, 6.0, 6.0))).unwrap()),
    )
    .unwrap();
    let mut backdrop_scene = Scene::new();
    backdrop_scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
    backdrop_scene.layer(
        Layer::new()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap()
            .try_backdrop_filter(backdrop)
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(1.0, 1.0, 4.0, 4.0), Color::TRANSPARENT);
        },
    );
    let backdrop_error = backdrop_scene
        .normalize(Capabilities::CURRENT)
        .expect_err("transformed backdrop stacks remain explicitly unsupported");
    assert_eq!(
        backdrop_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ))
    );
    assert!(
        backdrop_error
            .message()
            .contains("transformed backdrop capture")
    );
}

#[test]
fn non_readback_renderer_front_door_is_async() {
    pollster::block_on(async {
        let mut renderer = Renderer::new(Options::default()).await.unwrap();
        let mut surface = renderer
            .create_surface(Attachment::Headless, SurfaceOptions::default())
            .await
            .unwrap();
        renderer
            .render(&mut surface, &Scene::new(), Parameters::default())
            .await
            .unwrap();
        surface.resume(Attachment::Headless).unwrap();

        let headless = renderer
            .create_headless(Size::new(1.0, 1.0), 1.0)
            .await
            .unwrap();
        let _: Result<ImageBuffer> = renderer.read_headless(&headless).await;
    });
}

#[test]
fn surface_resize_rejects_physical_size_overflow_without_mutating_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 20.0), 1.5)).unwrap();

    let error = surface
        .resize(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0)
        .expect_err("physical device pixels should fit in u32");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(surface.size(), Size::new(10.0, 20.0));
    assert_eq!(surface.scale(), 1.5);
    assert_eq!(surface.physical_size(), PhysicalSize::new(15, 30));
}

#[test]
fn gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage() {
    let stages = [
        (GpuOperationStage::Render, BackendErrorCode::RenderFailed),
        (
            GpuOperationStage::Configure,
            BackendErrorCode::SurfaceConfigureFailed,
        ),
        (GpuOperationStage::Present, BackendErrorCode::PresentFailed),
    ];
    let faults = [
        GpuFaultKind::Validation,
        GpuFaultKind::OutOfMemory,
        GpuFaultKind::Internal,
    ];

    for (stage, expected_code) in stages {
        for fault in faults {
            let error = stage.classify_fault_for_test(fault, "injected GPU error");
            assert_eq!(
                error.code(),
                if fault == GpuFaultKind::OutOfMemory {
                    ErrorCode::SurfaceOutOfMemory
                } else {
                    Error::new(expected_code, "expected stage error").code()
                }
            );
        }
    }
}

#[test]
fn readback_transaction_maps_validation_internal_oom_and_terminal_failures() {
    use super::gpu_transaction::ReadbackSubmission;

    let _transaction_result_contract: Option<ReadbackSubmission> = None;
    for fault in [GpuFaultKind::Validation, GpuFaultKind::Internal] {
        let error = GpuOperationStage::Readback
            .classify_fault_for_test(fault, "injected readback GPU error");
        assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    }
    assert_eq!(
        GpuOperationStage::Readback
            .classify_fault_for_test(GpuFaultKind::OutOfMemory, "injected readback OOM")
            .code(),
        ErrorCode::SurfaceOutOfMemory
    );
    assert_eq!(
        Error::new(BackendErrorCode::ReadbackFailed, "readback failed").code(),
        ErrorCode::ReadbackFailed
    );

    let lost_signal = DeviceSignal::new_for_test();
    lost_signal.record_loss_for_test(DeviceLossReason::Destroyed);
    let lost = lost_signal
        .first_terminal()
        .expect("the injected readback loss must be terminal")
        .error(RuntimeOperation::SurfaceReadback);
    assert_eq!(
        lost.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::DeviceLost {
                    reason: DeviceLossReason::Destroyed,
                },
            )
            .unwrap()
        )
    );

    let faulted_signal = DeviceSignal::new_for_test();
    faulted_signal.record_uncaptured_fault_for_test(
        GpuFaultKind::Internal,
        "injected terminal readback fault",
    );
    let faulted = faulted_signal
        .first_terminal()
        .expect("the injected readback fault must be terminal")
        .error(RuntimeOperation::SurfaceReadback);
    assert_eq!(
        faulted.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::DeviceFaulted {
                    kind: GpuFaultKind::Internal,
                },
            )
            .unwrap()
        )
    );

    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("readback transaction coverage requires a host adapter");
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the readback transaction fixture must publish a headless texture");
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the scoped readback copy must complete");
    assert_eq!(output.size(), PhysicalSize::new(1, 1));

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let generation = signal.next_test_generation().unwrap();
    let transaction = super::gpu_transaction::GpuOperationTransaction::begin(
        &device,
        Arc::clone(&signal),
        generation,
        GpuOperationStage::Readback,
    );
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit readback transaction observation"),
        })
        .finish();
    let submission = pollster::block_on(submit_readback_observed_for_test(
        transaction,
        &queue,
        command_buffer,
        RuntimeOperation::SurfaceReadback,
    ))
    .expect("the explicit readback transaction must resolve its real scopes");
    assert_eq!(submission.queue_submission_count_for_test(), 1);
    assert_eq!(
        submission.transaction_generation_for_test(),
        submission.active_generation_for_test(),
        "the readback copy must submit while its transaction generation is active"
    );
    assert!(
        submission.scopes_resolved_for_test(),
        "the readback copy must resolve its scopes before completing"
    );
    let submission_index = submission.submission_index_for_test();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(Duration::from_secs(2)),
        })
        .expect("the retained readback submission index must name the completed copy");
}

#[test]
fn readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers() {
    assert_readback_submission_index_retained();
    assert_readback_idle_cleanup();
    assert_readback_pending_cleanup();
    assert_readback_callback_cleanup();
    assert_readback_mapped_completion();
}

use super::readback::{
    ReadbackCleanupEventForTest as Cleanup, ReadbackPhaseForTest,
    ReadbackStagingDispositionForTest as StagingDisposition, ReadbackStateMachineForTest,
};

fn readback_state_at(phase: ReadbackPhaseForTest) -> ReadbackStateMachineForTest {
    let mut state = ReadbackStateMachineForTest::allocated();
    match phase {
        ReadbackPhaseForTest::Allocated => {}
        ReadbackPhaseForTest::CopySubmitted { submission_index } => {
            state.copy_submitted_for_test(submission_index);
        }
        ReadbackPhaseForTest::MapPending => {
            state.copy_submitted_for_test(17);
            state.map_pending_for_test();
        }
        ReadbackPhaseForTest::Mapped => {
            state.copy_submitted_for_test(17);
            state.map_pending_for_test();
            state.map_callback_succeeded_for_test();
            state.mapped_for_test();
        }
        ReadbackPhaseForTest::PublishedBytes
        | ReadbackPhaseForTest::Failed
        | ReadbackPhaseForTest::Canceled => {
            panic!("the fixture accepts only uncertain readback phases")
        }
    }
    state
}

fn assert_readback_submission_index_retained() {
    let submitted = readback_state_at(ReadbackPhaseForTest::CopySubmitted {
        submission_index: 91,
    });
    assert_eq!(
        submitted.phase_for_test(),
        ReadbackPhaseForTest::CopySubmitted {
            submission_index: 91,
        },
        "the owner must retain the exact queue submission index"
    );
}

fn assert_readback_idle_cleanup() {
    for idle_phase in [
        ReadbackPhaseForTest::Allocated,
        ReadbackPhaseForTest::CopySubmitted {
            submission_index: 17,
        },
    ] {
        let mut failed = readback_state_at(idle_phase);
        failed.fail_for_test();
        assert_eq!(failed.phase_for_test(), ReadbackPhaseForTest::Failed);
        assert_eq!(
            failed.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "pre-map failure must drop idle staging without invalid unmap"
        );
        assert_eq!(
            failed.staging_disposition_for_test(),
            StagingDisposition::Released
        );
        failed.cancel_for_test();
        assert_eq!(
            failed.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "terminal cleanup must consume staging ownership exactly once"
        );
        assert_eq!(
            failed.staging_disposition_for_test(),
            StagingDisposition::Released
        );

        let mut canceled = readback_state_at(idle_phase);
        canceled.cancel_for_test();
        assert_eq!(canceled.phase_for_test(), ReadbackPhaseForTest::Canceled);
        assert_eq!(
            canceled.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "pre-map cancellation must drop idle staging without invalid unmap"
        );
        canceled.fail_for_test();
        assert_eq!(
            canceled.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "terminal cleanup must consume staging ownership exactly once"
        );
        assert_eq!(
            canceled.staging_disposition_for_test(),
            StagingDisposition::Released
        );
    }
}

fn assert_readback_pending_cleanup() {
    let mut pending_failure = readback_state_at(ReadbackPhaseForTest::MapPending);
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::MapPending
    );
    pending_failure.fail_for_test();
    assert_eq!(
        pending_failure.phase_for_test(),
        ReadbackPhaseForTest::Failed
    );
    assert_eq!(
        pending_failure.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "wrong-index or other pending-map failure must abort the request before dropping staging"
    );
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::Released
    );
    pending_failure.map_callback_succeeded_for_test();
    pending_failure.cancel_for_test();
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::Released,
        "a late callback cannot reacquire released staging"
    );
    assert_eq!(
        pending_failure.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "late delivery and second terminal cleanup cannot act on staging again"
    );

    let mut pending_cancellation = readback_state_at(ReadbackPhaseForTest::MapPending);
    pending_cancellation.cancel_for_test();
    assert_eq!(
        pending_cancellation.phase_for_test(),
        ReadbackPhaseForTest::Canceled
    );
    assert_eq!(
        pending_cancellation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "pending-map cancellation must abort the request before dropping staging"
    );
    assert_eq!(
        pending_cancellation.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

fn assert_readback_callback_cleanup() {
    for terminal_phase in [ReadbackPhaseForTest::Failed, ReadbackPhaseForTest::Canceled] {
        let mut callback_error = readback_state_at(ReadbackPhaseForTest::MapPending);
        callback_error.map_callback_failed_for_test();
        assert_eq!(
            callback_error.phase_for_test(),
            ReadbackPhaseForTest::MapPending,
            "callback delivery must not overwrite the lifecycle before the owner consumes it"
        );
        assert_eq!(
            callback_error.staging_disposition_for_test(),
            StagingDisposition::Idle,
            "a map callback error returns physical staging to known idle"
        );
        match terminal_phase {
            ReadbackPhaseForTest::Failed => callback_error.fail_for_test(),
            ReadbackPhaseForTest::Canceled => callback_error.cancel_for_test(),
            _ => unreachable!("the fixture selects only terminal cleanup phases"),
        }
        assert_eq!(callback_error.phase_for_test(), terminal_phase);
        assert_eq!(
            callback_error.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "callback-error-idle cleanup must not call unmap"
        );
        assert_eq!(
            callback_error.staging_disposition_for_test(),
            StagingDisposition::Released
        );
    }

    let mut callback_success_cancellation = readback_state_at(ReadbackPhaseForTest::MapPending);
    callback_success_cancellation.map_callback_succeeded_for_test();
    assert_eq!(
        callback_success_cancellation.phase_for_test(),
        ReadbackPhaseForTest::MapPending
    );
    assert_eq!(
        callback_success_cancellation.staging_disposition_for_test(),
        StagingDisposition::MappedActive
    );
    callback_success_cancellation.cancel_for_test();
    assert_eq!(
        callback_success_cancellation.phase_for_test(),
        ReadbackPhaseForTest::Canceled
    );
    assert_eq!(
        callback_success_cancellation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "cancellation racing callback success must unmap active staging before drop"
    );

    let dropped = readback_state_at(ReadbackPhaseForTest::MapPending);
    let drop_observation = dropped.observation_for_test();
    drop(dropped);
    assert_eq!(
        drop_observation.terminal_phase_for_test(),
        Some(ReadbackPhaseForTest::Canceled)
    );
    assert_eq!(
        drop_observation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped]
    );
    assert_eq!(
        drop_observation.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

fn assert_readback_mapped_completion() {
    let mut incomplete = readback_state_at(ReadbackPhaseForTest::Mapped);
    let error = incomplete
        .finish_mapped_for_test(PhysicalSize::new(1, 2), &[0; 256])
        .expect_err("a missing padded row must fail through checked decoding");
    assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    assert_eq!(incomplete.phase_for_test(), ReadbackPhaseForTest::Failed);
    assert_eq!(
        incomplete.staging_disposition_for_test(),
        StagingDisposition::Released
    );
    assert_eq!(
        incomplete.cleanup_events_for_test(),
        vec![
            Cleanup::MappedViewDropped,
            Cleanup::StagingUnmapped,
            Cleanup::StagingDropped,
        ],
        "the mapped view must drop before staging is unmapped"
    );

    let mut mapped = vec![0; 512];
    mapped[0..4].copy_from_slice(&[1, 2, 3, 4]);
    mapped[256..260].copy_from_slice(&[5, 6, 7, 8]);
    let mut published = readback_state_at(ReadbackPhaseForTest::Mapped);
    let image = published
        .finish_mapped_for_test(PhysicalSize::new(1, 2), &mapped)
        .expect("complete checked rows must publish one validated image");
    assert_eq!(image.rgba(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        published.phase_for_test(),
        ReadbackPhaseForTest::PublishedBytes
    );
    assert_eq!(
        published.cleanup_events_for_test(),
        vec![
            Cleanup::MappedViewDropped,
            Cleanup::StagingUnmapped,
            Cleanup::StagingDropped,
            Cleanup::PublishedBytes,
        ]
    );
    assert_eq!(
        published.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

#[test]
fn readback_map_callback_publishes_once_and_wakes_latest_waker() {
    use super::readback::ReadbackCompletionForTest;

    struct WakeCount(AtomicUsize);

    impl std::task::Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let first_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
    let latest_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
    let first_waker = Waker::from(Arc::clone(&first_wakes));
    let latest_waker = Waker::from(Arc::clone(&latest_wakes));
    let completion = ReadbackCompletionForTest::new();
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&first_waker)),
        Poll::Pending
    ));
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&latest_waker)),
        Poll::Pending
    ));

    completion.invoke_map_callback_for_test(Ok(()));
    assert_eq!(first_wakes.0.load(Ordering::SeqCst), 0);
    assert_eq!(latest_wakes.0.load(Ordering::SeqCst), 1);
    completion.deliver_late_map_result_for_test(Err(wgpu::BufferAsyncError));
    assert_eq!(completion.accepted_result_count_for_test(), 1);
    assert_eq!(completion.discarded_result_count_for_test(), 1);
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&latest_waker)),
        Poll::Ready(Ok(()))
    ));
    completion.deliver_late_map_result_for_test(Ok(()));
    assert_eq!(completion.accepted_result_count_for_test(), 1);
    assert_eq!(completion.discarded_result_count_for_test(), 2);

    let callback_error = ReadbackCompletionForTest::new();
    callback_error.invoke_map_callback_for_test(Err(wgpu::BufferAsyncError));
    let Poll::Ready(Err(error)) =
        callback_error.poll_for_test(&mut Context::from_waker(Waker::noop()))
    else {
        panic!("the callback error must be consumed exactly once")
    };
    assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    assert!(std::error::Error::source(&error).is_some());

    let canceled = ReadbackCompletionForTest::new();
    canceled.cancel_for_test();
    canceled.deliver_late_map_result_for_test(Ok(()));
    assert!(canceled.is_canceled_for_test());
    assert_eq!(canceled.accepted_result_count_for_test(), 0);
    assert_eq!(canceled.discarded_result_count_for_test(), 1);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let poll_completion = ReadbackCompletionForTest::new();
        let poll_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
        let poll_waker = Waker::from(Arc::clone(&poll_wakes));
        assert!(matches!(
            poll_completion.poll_for_test(&mut Context::from_waker(&poll_waker)),
            Poll::Pending
        ));
        assert!(poll_completion.timeout_slice_for_test());
        assert!(poll_completion.timeout_slice_for_test());
        assert_eq!(poll_completion.accepted_result_count_for_test(), 0);
        poll_completion.wrong_submission_index_for_test(9, 8);
        assert_eq!(poll_wakes.0.load(Ordering::SeqCst), 1);
        let Poll::Ready(Err(error)) =
            poll_completion.poll_for_test(&mut Context::from_waker(&poll_waker))
        else {
            panic!("a wrong submission index must terminate readback")
        };
        assert_eq!(error.code(), ErrorCode::ReadbackFailed);
        assert!(std::error::Error::source(&error).is_some());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct NativeReadbackDiagnosticDeadlineForTest {
    expires_at: Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeReadbackDiagnosticDeadlineForTest {
    fn begin() -> Self {
        Self {
            expires_at: Instant::now()
                .checked_add(Duration::from_secs(5))
                .expect("the native readback diagnostic deadline must be representable"),
        }
    }

    fn remaining(self) -> Option<Duration> {
        self.expires_at.checked_duration_since(Instant::now())
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeReadbackWakeConditionForTest {
    notified: Mutex<bool>,
    changed: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeReadbackWakeForTest {
    condition: Arc<NativeReadbackWakeConditionForTest>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeReadbackWakeForTest {
    fn fresh() -> Self {
        Self {
            condition: Arc::new(NativeReadbackWakeConditionForTest {
                notified: Mutex::new(false),
                changed: Condvar::new(),
            }),
        }
    }

    fn prepare_for_poll(&self) {
        *self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = false;
    }

    fn wait_for_wake(
        &self,
        deadline: NativeReadbackDiagnosticDeadlineForTest,
        stage: &NativeReadbackStageForTest,
        device_signal: &DeviceSignal,
    ) {
        let notified = self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(remaining) = deadline.remaining() else {
            panic!(
                "native readback diagnostic deadline expired: {}",
                native_readback_diagnostic_for_test(stage, device_signal)
            );
        };
        let (notified, timeout) = self
            .condition
            .changed
            .wait_timeout_while(notified, remaining, |notified| !*notified)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if timeout.timed_out() && !*notified {
            panic!(
                "native readback diagnostic deadline expired: {}",
                native_readback_diagnostic_for_test(stage, device_signal)
            );
        }
    }

    fn notify(&self) {
        let mut notified = self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *notified = true;
        self.condition.changed.notify_all();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::task::Wake for NativeReadbackWakeForTest {
    fn wake(self: Arc<Self>) {
        self.notify();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn drive_native_readback_for_test(
    stage: &mut NativeReadbackStageForTest,
    deadline: NativeReadbackDiagnosticDeadlineForTest,
    device_signal: &Arc<DeviceSignal>,
) -> Result<ImageBuffer> {
    let wake = Arc::new(NativeReadbackWakeForTest::fresh());
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    loop {
        wake.prepare_for_poll();
        match Pin::new(&mut *stage).poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => {}
        }
        wake.wait_for_wake(deadline, stage, device_signal);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_readback_diagnostic_for_test(
    stage: &NativeReadbackStageForTest,
    device_signal: &DeviceSignal,
) -> String {
    format!(
        "stage_phase={:?}; staging_disposition={:?}; submission_index={:?}; device_active_generation={:?}; device_terminal_signal={:?}",
        stage.phase_for_test(),
        stage.staging_disposition_for_test(),
        stage.submission_index_for_test(),
        device_signal.active_generation_for_test(),
        device_signal.first_terminal(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn headless_publication_texture_for_test(surface: &Surface) -> wgpu::Texture {
    match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the real headless fixture must retain one readable publication"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_readback_callback_progresses_and_cleans_up_with_diagnostic_deadline() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "native callback progress coverage requires an available host adapter"
    );
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("the callback progress fixture must publish a real headless texture");

    let device_signal = renderer
        .default_device_signal_for_test()
        .expect("native callback progress requires a ready device signal");
    let texture = headless_publication_texture_for_test(&surface);
    let (device, queue) = renderer
        .default_wgpu_device_queue()
        .expect("native callback progress requires a ready device and queue");
    let device = device.clone();
    let queue = queue.clone();
    let mut progress =
        NativeReadbackStageForTest::begin(&device, &queue, &texture, PhysicalSize::new(4, 4))
            .expect("the explicit native map stage must start from a real submitted texture copy");
    assert_eq!(
        progress.phase_for_test(),
        NativeReadbackStagePhaseForTest::MapPending
    );
    let deadline = NativeReadbackDiagnosticDeadlineForTest::begin();
    let image = drive_native_readback_for_test(&mut progress, deadline, &device_signal)
        .expect("the native callback must progress the real publication readback");
    assert_eq!(
        progress.phase_for_test(),
        NativeReadbackStagePhaseForTest::PublishedBytes
    );
    assert_eq!(progress.staging_disposition_for_test(), None);
    assert!(progress.staging_state_dropped_for_test());
    assert_eq!(device_signal.active_generation_for_test(), None);
    assert!(device_signal.first_terminal().is_none());
    assert_eq!(image.size(), PhysicalSize::new(4, 4));
    assert!(image.rgba().iter().any(|channel| *channel != 0));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canceled_native_readback_discards_late_callback_without_publication_change() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "native cancellation coverage requires an available host adapter"
    );
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 4.0, 4.0),
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .expect("the cancellation fixture must publish a real headless texture");
    let pixels_before = pollster::block_on(renderer.read_headless(&surface))
        .expect("the cancellation fixture publication must be readable");
    let publication_before = headless_publication_texture_for_test(&surface);
    let stats_before = renderer.stats();
    let renderer_options_before = renderer.options();
    let uploaded_images_before = renderer.uploaded_images_for_test();
    let parameters_before = surface.last_parameters;
    let surface_state_before = surface.state();
    let resource_state_before = surface.resource_state();
    let physical_size_before = surface.physical_size();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the published fixture must retain its ready device resources")
        .internal_resource_manager_observation_for_test();
    let device_signal = renderer
        .default_device_signal_for_test()
        .expect("native cancellation requires a ready device signal");
    let (device, queue) = renderer
        .default_wgpu_device_queue()
        .expect("native cancellation requires a ready device and queue");
    let device = device.clone();
    let queue = queue.clone();

    let mut canceled_future = NativeReadbackStageForTest::begin(
        &device,
        &queue,
        &publication_before,
        physical_size_before,
    )
    .expect("the explicit native future stage must start from a real submitted texture copy");
    let canceled_submission = canceled_future.submission_index_for_test();
    canceled_future.cancel_for_test();
    assert_eq!(
        canceled_future.phase_for_test(),
        NativeReadbackStagePhaseForTest::Canceled
    );
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(canceled_submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("the canceled native future must release its helper and staging request");
    assert!(canceled_future.staging_state_dropped_for_test());

    let late_callback = NativeReadbackLateCallbackStageForTest::cancel_before_poll(
        &device,
        &queue,
        &publication_before,
        physical_size_before,
    )
    .expect("the late-callback stage must register a real map before cancellation");
    assert!(
        matches!(
            late_callback.staging_disposition_for_test(),
            Some(super::readback::ReadbackStagingDispositionForTest::Released) | None
        ),
        "cancellation must release staging whether callback delivery is immediate or poll-driven"
    );
    late_callback
        .deliver_late_callback_for_test()
        .expect("native polling must deliver the real callback after cancellation");
    assert!(
        late_callback.callback_result_was_discarded_for_test(),
        "a real late callback must leave completion canceled and release its staging capture"
    );

    let pixels_after = pollster::block_on(renderer.read_headless(&surface))
        .expect("the preserved publication must remain readable after cancellation");

    let publication_after = headless_publication_texture_for_test(&surface);
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("canceled readback must retain the ready device resources")
        .internal_resource_manager_observation_for_test();
    assert_eq!(publication_after, publication_before);
    assert_eq!(pixels_after, pixels_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(renderer.options(), renderer_options_before);
    assert_eq!(renderer.uploaded_images_for_test(), uploaded_images_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(surface.state(), surface_state_before);
    assert_eq!(surface.resource_state(), resource_state_before);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(resources_after, resources_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    assert!(device_signal.first_terminal().is_none());
}

#[test]
fn uncaptured_faults_observe_active_and_released_generations() {
    let signal = DeviceSignal::new_for_test();
    let lease = GpuOperationLease::begin_for_test(&signal).unwrap();
    let generation = lease.generation_for_test();

    signal.record_uncaptured_fault_for_test(GpuFaultKind::Validation, "active fault");
    let terminal = signal
        .finish_active_generation_for_test(generation)
        .unwrap();
    assert_eq!(terminal.operation_generation_for_test(), Some(generation));
    assert_eq!(signal.active_generation_for_test(), None);

    let late_signal = DeviceSignal::new_for_test();
    let late_lease = GpuOperationLease::begin_for_test(&late_signal).unwrap();
    let late_generation = late_lease.generation_for_test();
    assert!(
        late_signal
            .finish_active_generation_for_test(late_generation)
            .is_none()
    );
    late_signal.record_uncaptured_fault_for_test(GpuFaultKind::Internal, "late fault");
    assert_eq!(
        late_signal
            .first_terminal()
            .expect("late fault must terminally affect the next operation")
            .operation_generation_for_test(),
        None
    );
}

#[test]
fn terminal_record_snapshots_share_identity_and_keep_the_first_record() {
    let signal = DeviceSignal::new_for_test();
    let lease = GpuOperationLease::begin_for_test(&signal).unwrap();
    let generation = lease.generation_for_test();

    signal.record_uncaptured_fault_for_test(GpuFaultKind::Validation, "first terminal record");
    signal.record_uncaptured_fault_for_test(GpuFaultKind::Internal, "later terminal record");

    let first_snapshot = signal
        .first_terminal()
        .expect("the first terminal signal must be recorded");
    let repeated_snapshot = signal
        .first_terminal()
        .expect("repeated terminal snapshots must remain available");
    let finished_snapshot = signal
        .finish_active_generation_for_test(generation)
        .expect("finishing the active generation must observe the terminal record");

    assert!(Arc::ptr_eq(&first_snapshot, &repeated_snapshot));
    assert!(Arc::ptr_eq(&first_snapshot, &finished_snapshot));
    assert!(matches!(
        first_snapshot.as_ref(),
        DeviceTerminalSignal::Faulted {
            kind: GpuFaultKind::Validation,
            message,
            operation_generation: Some(observed_generation),
        } if message == "first terminal record" && *observed_generation == generation
    ));
}

#[test]
fn dropped_gpu_operation_future_aborts_draft_state_and_leases() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    let resources = pollster::block_on(
        renderer.cancel_prepared_vello_pass_after_submit_for_test(&prepared, target_extent),
    )
    .expect("the explicit canceled Vello transaction must release its owned resources");
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "dropping the production render future must release its active transaction lease"
    );
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
    let error = pollster::block_on(renderer.read_headless(&surface))
        .expect_err("a canceled first frame must not publish readable bytes");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceReadback,
        RenderSurfaceAvailability::Uninitialized,
    );
}

#[test]
fn real_gpu_error_scope_captures_deliberate_validation_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let result = pollster::block_on(renderer.deliberate_validation_error_for_test())
        .expect("real GPU error-scope coverage requires a host adapter");
    let error = result.expect_err("the deliberate invalid texture must be captured by the scope");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn internal_vello_checked_shader_creation_reports_validation_without_unsafe() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let validation_result = {
        let (device, _) = renderer
            .default_wgpu_device_queue()
            .expect("checked internal Vello shader coverage requires a host adapter");
        pollster::block_on(super::vello_engine::checked_shader_validation_for_test(
            device,
        ))
    };

    let error = validation_result
        .expect_err("invalid internal Vello WGSL must fail through a checked scope");
    assert_eq!(error.code(), ErrorCode::RenderFailed);

    let out_of_memory = super::vello_engine::checked_scope_out_of_memory_for_test();
    assert_eq!(out_of_memory.code(), ErrorCode::SurfaceOutOfMemory);

    let preflight_error = {
        let (device, _) = renderer
            .default_wgpu_device_queue()
            .expect("checked internal Vello resource coverage requires a host adapter");
        pollster::block_on(super::vello_engine::over_limit_buffer_preflight_for_test(
            device,
        ))
    }
    .expect_err("an over-limit internal Vello buffer must fail before WGPU allocation");
    assert_eq!(preflight_error.code(), ErrorCode::RenderFailed);
    assert!(preflight_error.message().contains("device limit"));

    {
        let (device, queue) = renderer
            .default_wgpu_device_queue()
            .expect("checked internal Vello encoding coverage requires a host adapter");
        let engine = pollster::block_on(VelloEngineState::new_checked(device))
            .expect("internal Vello shaders must create through checked scopes");
        let resources = ResourceManager::default();
        let target_extent = PhysicalSize::new(64, 48);
        let target_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist checked internal Vello target"),
            size: wgpu::Extent3d {
                width: target_extent.width(),
                height: target_extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: target_usage,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let area_parameters =
            RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
                .expect("a non-empty internal Vello target must prepare");
        let area_pass = VelloScene::prepare_raster_scenario_for_test(
            VelloRasterScenario::Base,
            area_parameters,
        )
        .expect("the base scene must prepare for internal checked encoding");
        let msaa8_pass = VelloScene::prepare_raster_scenario_for_test(
            VelloRasterScenario::Base,
            area_parameters.with_antialiasing(Antialiasing::Msaa8),
        )
        .expect("the MSAA8 scene must prepare for internal checked encoding");
        let fixture = CheckedVelloFixture {
            device,
            queue,
            engine: &engine,
            resources: &resources,
            target: &target_view,
            extent: target_extent,
            usage: target_usage,
        };
        assert_checked_vello_commit(&fixture, &msaa8_pass);
        assert_checked_vello_abort(&fixture, &area_pass);

        assert_checked_vello_no_atlas_outcomes(device);

        assert_checked_vello_target_mismatch(&fixture, &area_pass);

        let invalid_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Surgeist checked internal Vello invalid storage target"),
            size: wgpu::Extent3d {
                width: target_extent.width(),
                height: target_extent.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let invalid_target_view =
            invalid_target.create_view(&wgpu::TextureViewDescriptor::default());
        let invalid_fixture = CheckedVelloFixture {
            target: &invalid_target_view,
            ..fixture
        };
        assert_checked_vello_invalid_target(&invalid_fixture, &area_pass);
    }

    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[derive(Clone, Copy)]
struct CheckedVelloFixture<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    engine: &'a VelloEngineState,
    resources: &'a ResourceManager,
    target: &'a wgpu::TextureView,
    extent: PhysicalSize,
    usage: wgpu::TextureUsages,
}

fn assert_checked_vello_commit(fixture: &CheckedVelloFixture<'_>, pass: &PreparedVelloPass) {
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist checked internal Vello committed command encoding"),
        });
    let mut scope = ActiveVelloEncodingScope::begin(fixture.device);
    let (lease, _logical_pass) = {
        let mut state = TransactionEncodingState::new(
            &mut scope,
            fixture.queue,
            &mut encoder,
            fixture.target,
            TransactionTargetIntent::new(
                fixture.extent,
                wgpu::TextureFormat::Rgba8Unorm,
                fixture.usage,
            ),
        );
        pass.encode_into(fixture.engine, fixture.resources, &mut state)
            .expect("an MSAA8 pass must encode through an active checked scope")
            .into_resources_and_logical_pass()
    };
    drop(encoder.finish());
    let lease = pollster::block_on(scope.finish_with_lease(lease))
        .expect("the caller must resolve a clean checked encoding scope");
    assert_eq!(
        super::vello_engine::commit_scope_resolved_for_test(lease)
            .expect("the scope-resolved Vello commit must keep accounting clean"),
        VelloAtlasOutcome::Retain
    );
}

fn assert_checked_vello_abort(fixture: &CheckedVelloFixture<'_>, pass: &PreparedVelloPass) {
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist checked internal Vello aborted command encoding"),
        });
    let mut scope = ActiveVelloEncodingScope::begin(fixture.device);
    let outcome = {
        let mut state = TransactionEncodingState::new(
            &mut scope,
            fixture.queue,
            &mut encoder,
            fixture.target,
            TransactionTargetIntent::new(
                fixture.extent,
                wgpu::TextureFormat::Rgba8Unorm,
                fixture.usage,
            ),
        );
        let (lease, _logical_pass) = pass
            .encode_into(fixture.engine, fixture.resources, &mut state)
            .expect("an area pass must encode through an active checked scope")
            .into_resources_and_logical_pass();
        let aborted = lease.abort();
        assert!(aborted.discarded_resource_count_for_test() > 0);
        aborted.into_atlas_outcome()
    };
    drop(encoder.finish());
    pollster::block_on(scope.finish())
        .expect("the caller must resolve an aborted checked encoding scope");
    assert_eq!(outcome, VelloAtlasOutcome::Recreate);
}

fn assert_checked_vello_no_atlas_outcomes(device: &wgpu::Device) {
    let committed = pollster::block_on(super::vello_engine::no_atlas_commit_outcome_for_test(
        device,
    ))
    .expect("a no-atlas lease commit must resolve through checked scopes");
    assert_eq!(committed, VelloAtlasOutcome::NoAtlas);
    let aborted = pollster::block_on(super::vello_engine::no_atlas_abort_outcome_for_test(device))
        .expect("a no-atlas lease abort must resolve through checked scopes");
    assert_eq!(aborted, VelloAtlasOutcome::NoAtlas);
}

fn assert_checked_vello_target_mismatch(
    fixture: &CheckedVelloFixture<'_>,
    pass: &PreparedVelloPass,
) {
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist checked internal Vello mismatched target encoding"),
        });
    let mut scope = ActiveVelloEncodingScope::begin(fixture.device);
    let failure = {
        let mut state = TransactionEncodingState::new(
            &mut scope,
            fixture.queue,
            &mut encoder,
            fixture.target,
            TransactionTargetIntent::new(
                PhysicalSize::new(63, 48),
                wgpu::TextureFormat::Rgba8Unorm,
                fixture.usage,
            ),
        );
        match pass.encode_into(fixture.engine, fixture.resources, &mut state) {
            Ok(encoded) => {
                let (lease, _logical_pass) = encoded.into_resources_and_logical_pass();
                let _ = lease.abort();
                panic!("a mismatched transaction target must fail before allocation");
            }
            Err(failure) => failure,
        }
    };
    drop(encoder.finish());
    pollster::block_on(scope.finish())
        .expect("a preflight target mismatch must leave checked scopes clean");
    assert_eq!(failure.error().code(), ErrorCode::RenderFailed);
    assert_eq!(
        failure.into_aborted_resources().into_atlas_outcome(),
        VelloAtlasOutcome::NoAtlas
    );
}

fn assert_checked_vello_invalid_target(
    fixture: &CheckedVelloFixture<'_>,
    pass: &PreparedVelloPass,
) {
    let mut encoder = fixture
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist checked internal Vello invalid target encoding"),
        });
    let mut scope = ActiveVelloEncodingScope::begin(fixture.device);
    let (lease, _logical_pass) = {
        let mut state = TransactionEncodingState::new(
            &mut scope,
            fixture.queue,
            &mut encoder,
            fixture.target,
            TransactionTargetIntent::new(
                fixture.extent,
                wgpu::TextureFormat::Rgba8Unorm,
                fixture.usage,
            ),
        );
        pass.encode_into(fixture.engine, fixture.resources, &mut state)
            .expect("the active scope must own actual target-view validation")
            .into_resources_and_logical_pass()
    };
    drop(encoder.finish());
    let failure = match pollster::block_on(scope.finish_with_lease(lease)) {
        Ok(lease) => {
            let _ = lease.abort();
            panic!("an invalid target view must be captured by the active checked scope");
        }
        Err(failure) => failure,
    };
    assert_eq!(failure.error().code(), ErrorCode::RenderFailed);
    assert_eq!(
        failure.into_aborted_resources().into_atlas_outcome(),
        VelloAtlasOutcome::Recreate
    );
}

#[test]
fn real_gpu_smoke_emits_no_uncaptured_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "real GPU smoke coverage requires a host adapter"
    );
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .expect("real GPU smoke coverage requires a host adapter");
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the production Renderer::create_headless + Renderer::render path must be clean");
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn headless_bgra8_remains_a_surface_create_diagnostic() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error = match pollster::block_on(renderer.create_surface(
        Attachment::Headless,
        SurfaceOptions {
            format: Format::Bgra8,
            ..SurfaceOptions::default()
        },
    )) {
        Ok(_) => panic!("unsupported headless format should fail before wgpu validation"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
    assert!(error.message().contains("Rgba8"));
}

#[cfg(feature = "render-window")]
#[test]
fn presented_surface_without_compatible_adapter_reports_typed_adapter_unavailable() {
    let error = require_presented_device_identity_for_test(None)
        .expect_err("a presented surface without a compatible adapter must be rejected");

    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    let diagnostic = error
        .runtime_capability_unavailable_diagnostic()
        .expect("adapter selection failure must carry its typed runtime diagnostic");
    assert_eq!(diagnostic.operation(), RuntimeOperation::AdapterSelection);
    assert_eq!(
        diagnostic.reason(),
        RuntimeCapabilityUnavailableReason::AdapterUnavailable
    );
}

#[test]
fn surface_operation_matrix_covers_every_kind_state_and_duplicate_transition() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();

    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "a nonzero headless surface has no publication before its first render"
    );
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the first headless render should publish a readable texture");
    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);

    surface.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::Ready,
        "same-physical resize retains the current publication"
    );
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "a physical-size change invalidates the old publication"
    );

    surface.suspend().unwrap();
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
    surface.resume(Attachment::Headless).unwrap();
    surface.resume(Attachment::Headless).unwrap();
    assert_eq!(surface.state(), SurfaceState::Available);

    let error = pollster::block_on(renderer.resume_surface(&mut surface, Attachment::Headless))
        .expect_err("renderer resume is not the headless lifecycle operation");
    assert_eq!(error.code(), ErrorCode::UnsupportedBackend);
}

#[test]
fn completed_headless_render_retains_ready_resources() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();

    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("a completed headless render must retain ready resources");

    assert!(matches!(
        &surface.backend,
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { .. },
            ..
        }
    ));
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn available_presented_resume_keeps_the_installed_attachment_without_recreating() {
    let action = Surface::presented_resume_action(
        SurfaceState::Available,
        PresentedLifecycle::Ready {
            resizing: ResizeState::Idle,
        },
    );

    assert!(
        matches!(action, PresentedResumeAction::NoOp),
        "an available presented surface must retain its attachment without WGPU recreation"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn planner_failure_precedes_pending_presented_surface_configuration() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented planner-gate coverage requires a compatible device");
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(8.0, 6.0),
            ..SurfaceOptions::default()
        },
    );
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let configuration_count_before = presented_configuration_count_for_test(&surface);
    let presented_before = presented_observation_for_test(&surface);
    let stats_before = renderer.stats();
    let parameters_before = surface.last_parameters;
    assert!(matches!(
        lifecycle_before,
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(resource_before, None);
    assert_eq!(configuration_count_before, 0);

    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK)
        .layer(bounded_planning_backdrop(), |scene| {
            add_planning_text(scene, TextRunBounds::unspecified());
        });

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("unresolved bounded text must fail the complete frame plan");
    let expected = UnresolvedResource::new(
        UnresolvedResourceKind::TextRunInkBounds,
        "normalized command 1.0",
    );
    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&expected));

    assert_eq!(
        presented_configuration_count_for_test(&surface),
        configuration_count_before,
        "presented configuration occurred before planner failure"
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        resource_before,
        "planner failure published presented configuration resources"
    );
    assert_eq!(
        presented_lifecycle_for_test(&surface),
        lifecycle_before,
        "planner failure changed the pending presented lifecycle"
    );
    assert_eq!(presented_observation_for_test(&surface), presented_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(surface.last_parameters, parameters_before);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_setup_and_resize_commit_only_after_clean_configuration() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented configuration coverage requires a compatible device");
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(2.0, 2.0),
            ..SurfaceOptions::default()
        },
    );

    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(presented_resource_id_for_test(&surface), None);

    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect("initial presented configuration must commit only after clean scopes");
    let initial_resource = presented_resource_id_for_test(&surface)
        .expect("clean configuration must commit one resource bundle");
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    let error = pollster::block_on(presented_configuration_validation_failure_stage_for_test(
        &mut renderer,
        &surface,
        RuntimeOperation::SurfaceRendering,
    ))
    .expect_err("a real Configure validation failure must leave the requested resize pending");
    assert_eq!(error.code(), ErrorCode::SurfaceConfigureFailed);
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));

    discard_presented_configuration_stage_for_test(&mut renderer, &surface)
        .expect("an explicit Configure draft must be discardable before publication");
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect_err("a terminal device must leave the pending configuration uncommitted");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
}

#[cfg(feature = "render-window")]
#[test]
fn presented_acquire_outcomes_map_every_surface_result_before_commit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented acquire coverage requires a compatible device");
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };

    let mut success = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut success, PresentedAcquireOutcomeForTest::Success);
    let stats = pollster::block_on(renderer.render(&mut success, &Scene::new(), parameters))
        .expect("a successful acquire must present and publish the frame");
    assert_eq!(renderer.stats(), stats);
    assert_eq!(success.last_parameters, Some(parameters));
    assert_eq!(
        presented_observation_for_test(&success).present_count_for_test(),
        1,
        "a successful acquired texture must be presented exactly once"
    );

    for outcome in [
        PresentedAcquireOutcomeForTest::Suboptimal,
        PresentedAcquireOutcomeForTest::Outdated,
    ] {
        let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
        let stats_before = renderer.stats();
        let parameters_before = surface.last_parameters;
        let resource_before = presented_resource_id_for_test(&surface);
        set_presented_acquire_outcome_for_test(&mut surface, outcome);

        let error = pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
            .expect_err("suboptimal and outdated acquisition must retry configuration then fail");
        assert_eq!(error.code(), ErrorCode::SurfaceOutdated);
        assert_eq!(renderer.stats(), stats_before);
        assert_eq!(surface.last_parameters, parameters_before);
        assert!(matches!(
            presented_lifecycle_for_test(&surface),
            PresentedLifecycle::Ready { .. }
        ));
        assert_ne!(presented_resource_id_for_test(&surface), resource_before);
        let observation = presented_observation_for_test(&surface);
        assert_eq!(observation.present_count_for_test(), 0);
        assert_eq!(
            observation.discarded_count_for_test(),
            if outcome == PresentedAcquireOutcomeForTest::Suboptimal {
                1
            } else {
                0
            },
            "only an acquired suboptimal texture needs RAII discard"
        );
    }

    for outcome in [
        PresentedAcquireOutcomeForTest::Timeout,
        PresentedAcquireOutcomeForTest::Validation,
    ] {
        let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
        let stats_before = renderer.stats();
        set_presented_acquire_outcome_for_test(&mut surface, outcome);
        let error = pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
            .expect_err("failed acquire must not publish frame state");
        assert_eq!(
            error.code(),
            match outcome {
                PresentedAcquireOutcomeForTest::Timeout => ErrorCode::SurfaceTimeout,
                PresentedAcquireOutcomeForTest::Validation => ErrorCode::PresentFailed,
                _ => unreachable!(),
            }
        );
        assert_eq!(renderer.stats(), stats_before);
        assert_eq!(surface.last_parameters, None);
        assert_eq!(
            presented_observation_for_test(&surface).present_count_for_test(),
            0
        );
    }

    let mut occluded = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut occluded, PresentedAcquireOutcomeForTest::Occluded);
    let error = pollster::block_on(renderer.render(&mut occluded, &Scene::new(), parameters))
        .expect_err("occluded acquire must not report a successful frame");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&occluded),
        PresentedLifecycle::Occluded { .. }
    ));
    assert_eq!(occluded.last_parameters, None);

    let mut lost = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut lost, PresentedAcquireOutcomeForTest::Lost);
    let error = pollster::block_on(renderer.render(&mut lost, &Scene::new(), parameters))
        .expect_err("surface loss must not report a successful frame");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&lost),
        PresentedLifecycle::Lost
    ));
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[cfg(feature = "render-window")]
#[test]
fn presented_blit_and_present_remain_scoped_until_frame_commit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented transaction coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let stats_before = renderer.stats();
    let parameters = Parameters {
        base_color: Color::TRANSPARENT,
        debug: true,
    };

    let observation = presented_observation_handle_for_test(&surface);
    let scene = Scene::new();
    let stats = pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .expect("scoped present must publish only after transaction completion");
    let observation = observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats);
    assert_ne!(renderer.stats(), stats_before);
    assert_eq!(renderer.stats(), stats);
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn render_window_smoke_executes_direct_and_graph_presented_frames() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("presented direct-and-graph smoke coverage requires a compatible device");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let observation = presented_observation_handle_for_test(&surface);
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);

    let direct = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()));
    let after_direct = observation.snapshot_for_test();
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ));
    let after_graph = observation.snapshot_for_test();

    let direct_presented = direct.is_ok()
        && after_direct.acquire_count_for_test() == 1
        && after_direct.present_count_for_test() == 1
        && after_direct.discarded_count_for_test() == 0;
    let graph_presented = graph.is_ok()
        && after_graph.acquire_count_for_test() == 2
        && after_graph.present_count_for_test() == 2
        && after_graph.discarded_count_for_test() == 0
        && graph.as_ref().is_ok_and(|frame| {
            frame.stats.route == Some(RenderRoute::GpuGraph) && frame.stats == renderer.stats()
        })
        && surface.headless_publication_count_for_test() == 0;

    assert!(
        direct_presented && graph_presented,
        "the presented graph did not acquire, submit, and present through one transaction"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_output_specializes_rgba_and_bgra_without_channel_swap() {
    let expected = [191_u8, 64, 16, 255];
    let mut preserves_rgba_semantics = true;

    for format in [Format::Rgba8, Format::Bgra8] {
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default()
                .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
        ))
        .expect("presented format coverage requires a compatible device");
        let working_format = default_graph_working_format_for_test(&mut renderer);
        let mut surface = display_free_presented_surface_for_test(
            &mut renderer,
            SurfaceOptions {
                size: Size::new(4.0, 4.0),
                format,
                ..SurfaceOptions::default()
            },
        );
        pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
            .expect("presented format coverage requires a configured output");
        let advertised_format_is_exact = matches!(
            &surface.backend,
            SurfaceBackend::Presented { surface, .. }
                if surface.format == wgpu::TextureFormat::from(format)
        );
        let mut scene = Scene::new();
        scene.fill(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Color::try_rgba(0.75, 0.25, 0.0625, 1.0)
                .expect("the asymmetric test color must be valid"),
        );

        let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
            &mut surface,
            &scene,
            Parameters::default(),
            working_format,
        ));
        let presented_texture = take_last_presented_texture_for_test(&mut surface);
        let semantic_pixel = presented_texture.as_ref().and_then(|texture| {
            pollster::block_on(
                renderer.read_render_texture_for_test(texture, PhysicalSize::new(4, 4)),
            )
            .ok()
            .and_then(|image| {
                let offset = (4 + 1) * 4;
                let raw: [u8; 4] = image.rgba().get(offset..offset + 4)?.try_into().ok()?;
                Some(match format {
                    Format::Rgba8 => raw,
                    Format::Bgra8 => [raw[2], raw[1], raw[0], raw[3]],
                })
            })
        });
        preserves_rgba_semantics &= graph.is_ok()
            && advertised_format_is_exact
            && surface.headless_publication_count_for_test() == 0
            && semantic_pixel.is_some_and(|actual| {
                actual
                    .into_iter()
                    .zip(expected)
                    .all(|(actual, expected)| actual.abs_diff(expected) <= 4)
            });
    }

    assert!(
        preserves_rgba_semantics,
        "presented output format conversion changed RGBA semantics"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_acquire_error_leaks_no_prepared_or_public_state() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("presented acquire-failure coverage requires a compatible device");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let stats_before = renderer.stats();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the configured surface must retain a ready device")
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the configured surface must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Timeout);
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);

    let error = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect_err("the injected acquire timeout must abort the prepared graph");

    assert_eq!(error.code(), ErrorCode::SurfaceTimeout);
    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 1);
    assert_eq!(presented.acquire_count_for_test(), 0);
    assert_eq!(presented.present_count_for_test(), 0);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(surface.last_parameters, None);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert_eq!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("an acquire timeout must retain the ready device")
            .device_pass_cache_counts_for_test(),
        cache_before
    );
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("an acquire timeout must return every prepared lease")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after.leased_count, 0);
    assert_eq!(
        resources_after.retained_count_for_test(),
        resources_before.retained_count_for_test()
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_scope_failure_suppresses_presentation_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented scope-failure coverage requires a compatible device");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(910),
        &mut presentation_commit,
    ))
    .expect_err("scope failure after a real submit must abort the host-effect draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(presentation_commit, Some(1));
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_accounting_fault_before_authorization_suppresses_present_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented accounting-fault coverage requires a compatible device");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let mut presentation_commit = Some(1);
    let error = pollster::block_on(graph_accounting_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(911),
        &mut presentation_commit,
    ))
    .expect_err("accounting rejection after submit must abort the host-effect draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(presentation_commit, Some(1));
    let after_fault = resources.observation_for_test();
    let Some(ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes,
        registered_entry_bytes,
    }) = after_fault.accounting_fault_for_test()
    else {
        panic!("the presented transaction must preserve the exact injected accounting fault");
    };
    assert_eq!(retained_bytes.checked_add(1), Some(registered_entry_bytes));
    assert_eq!(after_fault.active_frame_count, 0);
    assert_eq!(after_fault.leased_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_cancellation_after_submit_discards_without_presentation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented cancellation coverage requires a compatible device");
    let surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let observation = presented_observation_handle_for_test(&surface);
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);

    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &resources,
            modeled_resource_key_for_test(913),
            &mut presentation_commit,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }

    assert_eq!(presentation_commit, Some(1));
    let canceled = observation.snapshot_for_test();
    assert_eq!(canceled.acquire_attempt_count_for_test(), 0);
    assert_eq!(canceled.acquire_count_for_test(), 0);
    assert_eq!(canceled.present_count_for_test(), 0);
    assert_eq!(canceled.discarded_count_for_test(), 0);
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_terminal_loss_suppresses_presentation_and_transitions_device() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("presented terminal-loss coverage requires a compatible device");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);

    let error = pollster::block_on(graph_terminal_loss_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(912),
        &mut presentation_commit,
    ))
    .expect_err("terminal device loss after submit must suppress the host-effect draft");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::EffectRendering,
        DeviceLossReason::Destroyed,
    );
    assert_eq!(presentation_commit, Some(1));
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 0);
    assert_eq!(presented.acquire_count_for_test(), 0);
    assert_eq!(presented.present_count_for_test(), 0);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(surface.last_parameters, None);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert!(matches!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Destroyed
        })
    ));
    let repeated = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect_err("the terminal device generation must reject every later frame");
    assert_runtime_device_lost(
        repeated,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );
    assert_eq!(
        presented_observation_for_test(&surface),
        presented,
        "a terminal device generation must not reacquire or present"
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn surface_resize_suspend_resume_and_two_surfaces_own_resources() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented configuration coverage requires a compatible device");
    let mut first = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(0.0, 2.0),
            ..SurfaceOptions::default()
        },
    );
    let mut second = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(2.0, 2.0),
            ..SurfaceOptions::default()
        },
    );
    renderer.set_surface_resizing(&mut first, true).unwrap();

    assert!(matches!(
        presented_lifecycle_for_test(&first),
        PresentedLifecycle::NonRenderable { .. }
    ));
    assert_eq!(presented_resource_id_for_test(&first), None);
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut first))
        .expect("zero-area presented setup must avoid configuration and target allocation");
    assert_eq!(presented_resource_id_for_test(&first), None);

    first.resize(Size::new(2.0, 2.0), 1.0).unwrap();
    first.suspend().unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&first),
        PresentedLifecycle::ResizePending { .. }
    ));
    let error =
        pollster::block_on(renderer.render(&mut first, &Scene::new(), Parameters::default()))
            .expect_err(
                "a suspended resize must retain its requested configuration without WGPU work",
            );
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(presented_resource_id_for_test(&first), None);
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut first,
        Attachment::from_web_canvas("display-free-presented-test-target"),
    ))
    .expect("resuming a nonzero requested surface must configure it transactionally");
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut second))
        .expect("each ready presented surface must configure its own resource bundle");

    let first_resource =
        presented_resource_id_for_test(&first).expect("first surface must own a committed bundle");
    let second_resource = presented_resource_id_for_test(&second)
        .expect("second surface must own a committed bundle");
    assert_ne!(first_resource, second_resource);

    first.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    assert_eq!(presented_resource_id_for_test(&first), Some(first_resource));
    assert!(matches!(
        presented_lifecycle_for_test(&first),
        PresentedLifecycle::Ready {
            resizing: ResizeState::Resizing
        }
    ));
}

#[cfg(feature = "render-window")]
#[test]
fn surface_loss_can_resume_but_device_loss_requires_a_new_renderer() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("surface lifecycle coverage requires a compatible device");
    let mut other = configured_display_free_presented_surface_for_test(&mut renderer);
    let other_device = presented_device_identity_for_test(&other);
    let other_resource = presented_resource_id_for_test(&other)
        .expect("the default fixture must commit its initial target");
    let donor_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("surface-loss recreation coverage requires a non-default ready device slot");
    assert_ne!(donor_device, other_device);
    let initial_attachment = "display-free-donor-initial";
    let mut surface = configured_display_free_presented_surface_on_device_for_test(
        &mut renderer,
        donor_device,
        Attachment::from_web_canvas(initial_attachment),
    );
    let original_options = surface.options;
    let original_renderer_identity = surface.renderer_identity.clone();
    let original_parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), original_parameters))
        .expect("the donor surface must render before loss");
    let initial_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must commit its initial target");
    assert_eq!(presented_device_identity_for_test(&surface), donor_device);

    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("surface loss must not terminally lose its ready device");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Lost
    ));
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(renderer.default_device_has_no_terminal_signal_for_test());

    let replacement_attachment = "display-free-donor-replacement";
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas(replacement_attachment),
    ))
    .expect("a lost surface must recreate on its same ready device");
    let resumed_resource = presented_resource_id_for_test(&surface)
        .expect("resuming the lost surface must configure a new target");
    assert_ne!(resumed_resource, initial_resource);
    assert_eq!(presented_device_identity_for_test(&surface), donor_device);
    assert_eq!(surface.options, original_options);
    assert!(
        surface
            .renderer_identity
            .matches(&original_renderer_identity)
    );
    assert_eq!(surface.last_parameters, Some(original_parameters));
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("the recreated display-free surface must retain a web-canvas attachment"),
        },
        replacement_attachment
    );
    assert_eq!(presented_resource_id_for_test(&other), Some(other_resource));
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the recreated surface must render through the original ready device");
    pollster::block_on(renderer.render(&mut other, &Scene::new(), Parameters::default()))
        .expect("the default surface must remain coherent after donor-surface recreation");

    renderer.signal_device_loss_for_test(donor_device, DeviceLossReason::Destroyed);
    let error = pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("display-free-presented-test-target"),
    ))
    .expect_err("resume must not revive a terminal device generation");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Destroyed,
    );
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Destroyed,
        }),
    );

    let mut replacement = pollster::block_on(Renderer::new(Options::default()))
        .expect("a new renderer is the explicit recovery path after device loss");
    let mut replacement_surface =
        configured_display_free_presented_surface_for_test(&mut replacement);
    pollster::block_on(replacement.render(
        &mut replacement_surface,
        &Scene::new(),
        Parameters::default(),
    ))
    .expect("a replacement renderer must own a fresh ready device generation");
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resize_preserves_lost_recovery_gate_for_same_and_changed_extents() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost-resize coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let committed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let committed_target = presented_target_identity_for_test(&surface);

    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("acquire loss must close the surface recovery gate");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );
    assert_eq!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Lost
    );

    let stats_before = renderer.stats();
    let parameters_before = surface.last_parameters;
    let observation_before = presented_observation_for_test(&surface);

    surface.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    let same_physical_size = surface.physical_size();
    let same_lifecycle = presented_lifecycle_for_test(&surface);
    let same_capabilities = renderer.runtime_capabilities(&surface);
    let same_render = pollster::block_on(renderer.render(
        &mut surface,
        &Scene::new(),
        presented_black_debug_parameters_for_test(),
    ));
    let same_resource = presented_resource_id_for_test(&surface);
    let same_target = presented_target_identity_for_test(&surface);
    let same_observation = presented_observation_for_test(&surface);
    let same_stats = renderer.stats();
    let same_parameters = surface.last_parameters;
    let same_active_generation = renderer.default_device_active_operation_generation_for_test();

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    let changed_physical_size = surface.physical_size();
    let changed_lifecycle = presented_lifecycle_for_test(&surface);
    let changed_capabilities = renderer.runtime_capabilities(&surface);
    let changed_render = pollster::block_on(renderer.render(
        &mut surface,
        &Scene::new(),
        presented_black_debug_parameters_for_test(),
    ));
    let changed_resource = presented_resource_id_for_test(&surface);
    let changed_target = presented_target_identity_for_test(&surface);
    let changed_observation = presented_observation_for_test(&surface);
    let changed_stats = renderer.stats();
    let changed_parameters = surface.last_parameters;
    let changed_active_generation = renderer.default_device_active_operation_generation_for_test();

    assert_eq!(
        [same_lifecycle, changed_lifecycle],
        [PresentedLifecycle::Lost, PresentedLifecycle::Lost],
        "same- and changed-extent resize must not bypass explicit lost-surface recovery"
    );
    assert_eq!(same_physical_size, PhysicalSize::new(2, 2));
    assert_eq!(changed_physical_size, PhysicalSize::new(3, 2));
    let lost_capabilities =
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Lost,
        });
    assert_eq!(same_capabilities, lost_capabilities);
    assert_eq!(changed_capabilities, lost_capabilities);
    for result in [same_render, changed_render] {
        let error = result.expect_err("resize must not make a lost surface renderable");
        assert_surface_unavailable(
            error,
            RuntimeOperation::SurfaceRendering,
            RenderSurfaceAvailability::Lost,
        );
    }
    assert_eq!(
        [same_resource, changed_resource],
        [Some(committed_resource), Some(committed_resource)],
        "resize while lost must not publish a replacement configuration"
    );
    assert_eq!([same_target, changed_target], [committed_target; 2]);
    assert_eq!(
        [same_observation, changed_observation],
        [observation_before; 2],
        "rejected lost-surface renders must not acquire or present a frame"
    );
    assert_eq!([same_stats, changed_stats], [stats_before; 2]);
    assert_eq!(
        [same_parameters, changed_parameters],
        [parameters_before; 2]
    );
    assert_eq!(
        [same_active_generation, changed_active_generation],
        [None; 2]
    );

    assert_explicit_lost_resize_recovery(
        &mut renderer,
        &mut surface,
        committed_resource,
        committed_target,
    );
}

#[cfg(feature = "render-window")]
fn assert_explicit_lost_resize_recovery(
    renderer: &mut Renderer,
    surface: &mut Surface,
    committed_resource: u64,
    committed_target: u64,
) {
    let replacement_attachment = "lost-resize-replacement";
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        surface,
        Attachment::from_web_canvas(replacement_attachment),
    ))
    .expect("explicit resume must recover at the final requested extent");
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(3, 2));
    assert!(matches!(
        presented_lifecycle_for_test(surface),
        PresentedLifecycle::Ready { .. }
    ));
    let committed_physical_size = match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.committed_physical_size(),
        _ => panic!("the fixture must retain a presented surface backend"),
    };
    assert_eq!(committed_physical_size, Some(PhysicalSize::new(3, 2)));
    assert_ne!(
        presented_resource_id_for_test(surface),
        Some(committed_resource)
    );
    assert_ne!(
        presented_target_identity_for_test(surface),
        committed_target
    );
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("lost recovery must install a compatible presented attachment"),
        },
        replacement_attachment
    );
    assert!(matches!(
        renderer.runtime_capabilities(surface),
        RuntimeCapabilities::Available(_)
    ));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    pollster::block_on(renderer.render(surface, &Scene::new(), Parameters::default()))
        .expect("the explicitly resumed surface must render on its ready device");
}

#[cfg(feature = "render-window")]
#[test]
fn available_resize_pending_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available-resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("compatible-resume-candidate"),
    ))
    .expect("available resume must configure the pending extent on the installed target");

    let attachment_after = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id(),
        _ => panic!("available resume must retain the installed attachment kind"),
    };
    assert_eq!(
        (
            attachment_after,
            presented_target_identity_for_test(&surface)
        ),
        (installed_attachment.as_str(), installed_target),
        "available pending resume must retain the installed attachment and target identities"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(3, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    let configured_resource = presented_resource_id_for_test(&surface)
        .expect("pending resume must commit a configured target bundle");
    assert_ne!(configured_resource, installed_resource);
    let committed_physical_size = match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.committed_physical_size(),
        _ => panic!("the fixture must retain a presented surface backend"),
    };
    assert_eq!(committed_physical_size, Some(PhysicalSize::new(3, 2)));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pending configuration must return its transaction generation"
    );
    assert!(matches!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Available(_)
    ));

    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the configured existing target must remain renderable");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn available_nonrenderable_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available nonrenderable resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    surface.resize(Size::new(0.0, 2.0), 1.0).unwrap();
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(0, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::NonRenderable {
            physical_size,
            resizing: ResizeState::Idle,
        } if physical_size == PhysicalSize::new(0, 2)
    ));

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-nonrenderable-resume-candidate"),
    ))
    .expect("available nonrenderable resume must be an idempotent compatible success");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("available resume must retain the installed attachment kind"),
        },
        installed_attachment
    );
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target,
        "available nonrenderable resume must retain the installed host target"
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource),
        "available nonrenderable resume must retain the installed target resources"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(0, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::NonRenderable { .. }
    ));
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::NonRenderable,
        })
    );
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the retained zero-area surface must remain nonrenderable");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::NonRenderable,
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "an idempotent available resume and rejected render must start no GPU transaction"
    );

    surface.resize(Size::new(2.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("restoring the installed extent must render through the retained target");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn available_occluded_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available occluded resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Occluded);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the synthetic occlusion must enter the occluded lifecycle");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Occluded { .. }
    ));

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-occluded-resume-candidate"),
    ))
    .expect("available occluded resume may remain occluded on its installed target");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("available resume must retain the installed attachment kind"),
        },
        installed_attachment
    );
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target,
        "available occluded resume must retain the installed host target"
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource),
        "available occluded resume must retain the installed target resources"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Occluded { .. }
    ));
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Occluded,
        })
    );
    let observation_before_rejected_render = installed_observation.snapshot_for_test();
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("an occluded surface must remain unavailable until explicit recovery");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert_eq!(
        installed_observation.snapshot_for_test(),
        observation_before_rejected_render,
        "an occluded render rejection must not attempt another acquire"
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "an idempotent available resume and rejected render must start no GPU transaction"
    );

    surface.resize(Size::new(2.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("same-extent recovery must render through the retained target");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn suspended_presented_replacement_terminal_loss_before_configuration_uses_surface_resume() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("suspended replacement attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = presented_black_debug_parameters_for_test();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before replacement");
    surface.suspend().unwrap();

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let device_before = presented_device_identity_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    let error = pollster::block_on(
        renderer.resume_display_free_presented_surface_after_device_loss_for_test(
            &mut surface,
            Attachment::from_web_canvas("suspended-replacement-candidate"),
            DeviceLossReason::Unknown,
        ),
    )
    .expect_err("terminal loss before replacement configuration must abort resume");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
    assert_eq!(surface.state(), SurfaceState::Suspended);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(presented_device_identity_for_test(&surface), device_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed replacement must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pre-configuration loss must not begin a Configure transaction"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn lost_presented_recreation_terminal_loss_before_configuration_uses_surface_resume() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost recreation attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before loss");
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("synthetic acquire loss must require explicit recreation");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let device_before = presented_device_identity_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    let error = pollster::block_on(
        renderer.resume_display_free_presented_surface_after_device_loss_for_test(
            &mut surface,
            Attachment::from_web_canvas("lost-recreation-candidate"),
            DeviceLossReason::Unknown,
        ),
    )
    .expect_err("terminal loss before recreation configuration must abort resume");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(presented_device_identity_for_test(&surface), device_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Lost
    );
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed recreation must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pre-configuration loss must not begin a Configure transaction"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resume_prefers_installed_compatible_slot_over_earlier_donor_slot() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented selection coverage requires a compatible device");
    let mut earlier = configured_display_free_presented_surface_for_test(&mut renderer);
    let earlier_device = presented_device_identity_for_test(&earlier);
    let earlier_resource = presented_resource_id_for_test(&earlier);
    let earlier_target = presented_target_identity_for_test(&earlier);
    let installed_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("presented selection coverage requires a later ready device slot");
    assert_ne!(installed_device, earlier_device);
    let mut surface = configured_display_free_presented_surface_on_device_for_test(
        &mut renderer,
        installed_device,
        Attachment::from_web_canvas("installed-slot-target"),
    );
    surface.suspend().unwrap();

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("installed-slot-replacement"),
    ))
    .expect("resume must configure a replacement on the installed compatible slot");

    assert_eq!(
        presented_device_identity_for_test(&surface),
        installed_device,
        "an earlier compatible slot must not capture a surface from its installed ready slot"
    );
    assert_eq!(presented_resource_id_for_test(&earlier), earlier_resource);
    assert_eq!(presented_target_identity_for_test(&earlier), earlier_target);
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the resumed surface must render through its installed device slot");
    pollster::block_on(renderer.render(&mut earlier, &Scene::new(), Parameters::default()))
        .expect("the earlier donor surface must retain coherent resources");
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resume_skips_terminal_compatible_donor_for_later_healthy_slot() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("terminal donor selection coverage requires a compatible device");
    let terminal_donor_surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let terminal_donor = presented_device_identity_for_test(&terminal_donor_surface);
    let terminal_donor_resource = presented_resource_id_for_test(&terminal_donor_surface);
    let terminal_donor_target = presented_target_identity_for_test(&terminal_donor_surface);
    let installed_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("terminal donor selection coverage requires an installed device slot");
    let mut surface = configured_display_free_presented_surface_on_device_for_test(
        &mut renderer,
        installed_device,
        Attachment::from_web_canvas("terminal-donor-installed-target"),
    );
    let parameters = presented_black_debug_parameters_for_test();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the installed surface must establish public frame state before replacement");
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the installed surface must own committed resources");
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_options = surface.options;
    let installed_physical_size = surface.physical_size();
    let installed_renderer_identity = surface.renderer_identity.clone();
    let installed_stats = renderer.stats();

    let healthy_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("terminal donor selection coverage requires a later healthy device slot");
    assert_ne!(terminal_donor, installed_device);
    assert_ne!(terminal_donor, healthy_device);
    assert_ne!(installed_device, healthy_device);

    surface.suspend().unwrap();
    renderer.signal_device_loss_for_test(terminal_donor, DeviceLossReason::Destroyed);
    assert!(
        renderer
            .device_signal_for_test(terminal_donor)
            .expect("the terminal donor must retain its callback signal")
            .first_terminal()
            .is_some(),
        "the earlier donor must record terminal loss before selection"
    );
    let selected_device = select_display_free_presented_device_for_test(
        &mut renderer,
        installed_device,
        &[
            DisplayFreePresentedDeviceCompatibilityForTest::compatible(terminal_donor),
            DisplayFreePresentedDeviceCompatibilityForTest::incompatible(installed_device),
            DisplayFreePresentedDeviceCompatibilityForTest::compatible(healthy_device),
        ],
    )
    .expect("the explicit compatibility stage must find the later healthy slot");
    assert_eq!(selected_device, healthy_device);
    let SurfaceBackend::Presented {
        device_identity, ..
    } = &mut surface.backend
    else {
        panic!("the compatibility fixture must retain a presented surface");
    };
    *device_identity = selected_device;
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("terminal-donor-replacement-target"),
    ))
    .expect("resume must skip the terminal donor and publish through the later healthy slot");

    assert!(renderer.device_renderer_released_for_test(terminal_donor));
    assert_eq!(presented_device_identity_for_test(&surface), healthy_device);
    assert_eq!(surface.state(), SurfaceState::Available);
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    assert_ne!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource)
    );
    assert_ne!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(surface.options, installed_options);
    assert_eq!(surface.physical_size(), installed_physical_size);
    assert!(
        surface
            .renderer_identity
            .matches(&installed_renderer_identity)
    );
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(renderer.stats(), installed_stats);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("the replacement must retain a web-canvas attachment"),
        },
        "terminal-donor-replacement-target"
    );
    assert_eq!(
        presented_resource_id_for_test(&terminal_donor_surface),
        terminal_donor_resource
    );
    assert_eq!(
        presented_target_identity_for_test(&terminal_donor_surface),
        terminal_donor_target
    );
    pollster::block_on(renderer.submit_scoped_wgpu_probe_for_test(installed_device))
        .expect("replacement incompatibility must not disable the installed healthy slot");
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the resumed surface must render through the later healthy slot");
}

#[cfg(feature = "render-window")]
#[test]
fn available_resize_pending_resume_terminal_loss_preserves_surface_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("pending resume attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before the resume race");
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let state_before = surface.state();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-pending-resume-candidate"),
    ))
    .expect_err("terminal loss must abort the pending resume configuration");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed resume must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(surface.state(), state_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "terminal resume preflight must not leave an active operation generation"
    );
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        })
    );
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
}

#[cfg(feature = "render-window")]
#[test]
fn lost_recreation_resume_terminal_loss_preserves_surface_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost recreation attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before surface loss");
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the synthetic acquire loss must require explicit recreation");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let state_before = surface.state();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-lost-recreation-candidate"),
    ))
    .expect_err("terminal loss must abort replacement installation");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed recreation must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(lifecycle_before, PresentedLifecycle::Lost);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(surface.state(), state_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "terminal recreation preflight must not leave an active operation generation"
    );
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        })
    );
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
}

#[cfg(feature = "render-window")]
#[test]
fn resize_suspend_resume_and_two_surfaces_keep_device_resources_coherent() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented lifecycle coverage requires a compatible device");
    let mut first = configured_display_free_presented_surface_for_test(&mut renderer);
    let mut second = configured_display_free_presented_surface_for_test(&mut renderer);
    let first_initial = presented_resource_id_for_test(&first).unwrap();
    let second_initial = presented_resource_id_for_test(&second).unwrap();
    let first_target_initial = presented_target_identity_for_test(&first);
    let second_target_initial = presented_target_identity_for_test(&second);

    first.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    assert_eq!(presented_resource_id_for_test(&first), Some(first_initial));
    assert!(matches!(
        presented_lifecycle_for_test(&first),
        PresentedLifecycle::Ready { .. }
    ));

    first.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert_eq!(presented_resource_id_for_test(&first), Some(first_initial));
    assert!(matches!(
        presented_lifecycle_for_test(&first),
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(
        presented_resource_id_for_test(&second),
        Some(second_initial)
    );

    first.suspend().unwrap();
    first.suspend().unwrap();
    let error =
        pollster::block_on(renderer.render(&mut first, &Scene::new(), Parameters::default()))
            .expect_err("suspended surfaces must fail before configuring or rendering");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Suspended,
    );
    assert_eq!(presented_resource_id_for_test(&first), Some(first_initial));

    let attachment_kind_before = first.attachment.kind();
    let attachment_identity_before = match &first.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must retain a web-canvas attachment"),
    };
    let lifecycle_before = presented_lifecycle_for_test(&first);
    let parameters_before = first.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&first);
    let old_target_observation = presented_observation_handle_for_test(&first);
    let failed_candidate = display_free_presented_surface_on_device_for_test(
        &mut renderer,
        first.options,
        presented_device_identity_for_test(&first),
        Attachment::from_web_canvas("failed-resume-replacement"),
    );
    let error = pollster::block_on(presented_configuration_validation_failure_stage_for_test(
        &mut renderer,
        &failed_candidate,
        RuntimeOperation::SurfaceResume,
    ))
    .expect_err("a failed replacement configuration must preserve the installed surface state");
    assert_eq!(error.code(), ErrorCode::SurfaceConfigureFailed);
    assert_eq!(presented_resource_id_for_test(&failed_candidate), None);
    drop(failed_candidate);
    assert_eq!(first.attachment.kind(), attachment_kind_before);
    assert_eq!(
        match &first.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("the failed resume must retain its original attachment kind"),
        },
        attachment_identity_before
    );
    assert_eq!(first.state(), SurfaceState::Suspended);
    assert_eq!(presented_lifecycle_for_test(&first), lifecycle_before);
    assert_eq!(presented_resource_id_for_test(&first), Some(first_initial));
    assert_eq!(
        presented_target_identity_for_test(&first),
        first_target_initial
    );
    assert_eq!(first.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&first), observation_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "a failed configure transaction must return its active generation"
    );
    assert_eq!(
        presented_resource_id_for_test(&second),
        Some(second_initial),
        "a failed resume must not disturb another surface's committed target"
    );
    assert_eq!(
        presented_target_identity_for_test(&second),
        second_target_initial,
        "a failed resume must not disturb another surface's host target"
    );

    assert_successful_presented_resume_coherence(PresentedResumeCoherenceContextForTest {
        renderer: &mut renderer,
        first: &mut first,
        second: &mut second,
        first_initial,
        second_initial,
        first_target_initial,
        second_target_initial,
        observation_before,
        old_target_observation,
    });
}

#[cfg(feature = "render-window")]
struct PresentedResumeCoherenceContextForTest<'a> {
    renderer: &'a mut Renderer,
    first: &'a mut Surface,
    second: &'a mut Surface,
    first_initial: u64,
    second_initial: u64,
    first_target_initial: u64,
    second_target_initial: u64,
    observation_before: DisplayFreePresentedSurfaceObservationForTest,
    old_target_observation: DisplayFreePresentedSurfaceObservationHandleForTest,
}

#[cfg(feature = "render-window")]
fn assert_successful_presented_resume_coherence(
    context: PresentedResumeCoherenceContextForTest<'_>,
) {
    let resumed_attachment = "display-free-resumed-target";
    pollster::block_on(
        context
            .renderer
            .resume_display_free_presented_surface_for_test(
                context.first,
                Attachment::from_web_canvas(resumed_attachment),
            ),
    )
    .expect("resume must atomically install and configure the replacement host target");
    let first_resized = presented_resource_id_for_test(context.first).unwrap();
    assert_ne!(first_resized, context.first_initial);
    let first_target_resumed = presented_target_identity_for_test(context.first);
    assert_ne!(first_target_resumed, context.first_target_initial);
    assert_eq!(
        match &context.first.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("the resumed display-free surface must retain a web-canvas attachment"),
        },
        resumed_attachment
    );
    let resumed_target_observation = presented_observation_handle_for_test(context.first);
    assert_eq!(
        context.old_target_observation.snapshot_for_test(),
        context.observation_before
    );
    assert_eq!(
        presented_resource_id_for_test(context.second),
        Some(context.second_initial)
    );
    assert_eq!(
        presented_target_identity_for_test(context.second),
        context.second_target_initial,
        "resuming the first surface must not replace the other surface's host target"
    );
    assert_eq!(
        context
            .renderer
            .default_device_active_operation_generation_for_test(),
        None,
        "a committed resume configuration must return its active generation"
    );
    pollster::block_on(
        context
            .renderer
            .resume_display_free_presented_surface_for_test(
                context.first,
                Attachment::from_web_canvas("display-free-presented-test-target"),
            ),
    )
    .expect("a compatible duplicate resume must retain the committed target");
    assert_eq!(
        presented_resource_id_for_test(context.first),
        Some(first_resized)
    );
    pollster::block_on(context.renderer.render(
        context.first,
        &Scene::new(),
        Parameters::default(),
    ))
    .expect("the resized surface must render with its own committed target");
    assert_eq!(
        context.old_target_observation.snapshot_for_test(),
        context.observation_before
    );
    assert_eq!(
        resumed_target_observation
            .snapshot_for_test()
            .acquire_count_for_test(),
        1,
        "the replacement host target must receive the resumed surface's frame"
    );
    pollster::block_on(context.renderer.render(
        context.second,
        &Scene::new(),
        Parameters::default(),
    ))
    .expect("the untouched surface must retain and render with its own target");
}

#[cfg(feature = "render-window")]
fn presented_black_debug_parameters_for_test() -> Parameters {
    Parameters {
        base_color: Color::BLACK,
        debug: true,
    }
}

#[test]
fn zero_size_headless_render_diagnoses_and_read_returns_empty() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(0.0, 2.0), 1.0)).unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Empty);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("zero-area headless rendering must be rejected before planning");
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::NonRenderable,
                },
            )
            .unwrap()
        )
    );

    let image = pollster::block_on(renderer.read_headless(&surface))
        .expect("zero-area headless readback returns a validated empty image");
    assert_eq!(image.size(), PhysicalSize::new(0, 2));
    assert!(image.rgba().is_empty());
}

#[test]
fn nonzero_headless_read_before_publication_reports_uninitialized_without_map() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();

    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "creation must defer headless texture allocation"
    );
    let error = pollster::block_on(renderer.read_headless(&surface))
        .expect_err("a nonzero headless surface has no readable publication before render");
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Uninitialized,
                },
            )
            .unwrap()
        )
    );
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
}

#[test]
fn surface_suspend_and_resume_preserve_attachment_kind() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
    let scene = Scene::new();

    surface.suspend().unwrap();
    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("suspended surfaces should be unavailable");

    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Suspended,
    );

    surface.resume(Attachment::Headless).unwrap();
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("resumed headless surface should render");

    let error = surface
        .resume(Attachment::from_web_canvas("canvas"))
        .expect_err("surface backend kind should not change on resume");

    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
}

#[test]
fn foreign_and_stale_surfaces_fail_before_device_slot_access() {
    let mut owner = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut foreign_renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut foreign_surface =
        pollster::block_on(owner.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();

    if let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut foreign_surface.backend
    {
        device_identity.mark_stale_for_test();
    }

    assert_eq!(
        foreign_renderer.runtime_capabilities(&foreign_surface),
        RuntimeCapabilities::Unavailable(
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                kind: SurfaceIdentityMismatchKind::ForeignRenderer,
            }
        ),
        "a foreign renderer must reject the surface before consulting its stale device identity"
    );

    let error = pollster::block_on(foreign_renderer.render(
        &mut foreign_surface,
        &Scene::new(),
        Parameters::default(),
    ))
    .expect_err("foreign surfaces must fail before indexing their device slot");

    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceRendering,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );
    let error = pollster::block_on(foreign_renderer.read_headless(&foreign_surface))
        .expect_err("foreign readback must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceReadback,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );
    let error = pollster::block_on(
        foreign_renderer.resume_surface(&mut foreign_surface, Attachment::Headless),
    )
    .expect_err("foreign resume must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceResume,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );

    let mut stale_surface =
        pollster::block_on(owner.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut stale_surface.backend
    else {
        panic!("the test environment must provide a device-backed headless surface");
    };
    device_identity.mark_stale_for_test();

    assert_eq!(
        owner.runtime_capabilities(&stale_surface),
        RuntimeCapabilities::Unavailable(
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
            }
        ),
        "a stale surface must be rejected before runtime capability projection"
    );

    let error =
        pollster::block_on(owner.render(&mut stale_surface, &Scene::new(), Parameters::default()))
            .expect_err("stale rendering must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceRendering,
        SurfaceIdentityMismatchKind::StaleDeviceGeneration,
    );
    let error = pollster::block_on(owner.read_headless(&stale_surface))
        .expect_err("stale readback must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceReadback,
        SurfaceIdentityMismatchKind::StaleDeviceGeneration,
    );
    let error = pollster::block_on(owner.resume_surface(
        &mut stale_surface,
        Attachment::from_web_canvas("incompatible-canvas"),
    ))
    .expect_err("headless resume must reject its backend before attachment or stale validation");
    assert_eq!(error.code(), ErrorCode::UnsupportedBackend);
    let error = pollster::block_on(owner.resume_surface(&mut stale_surface, Attachment::Headless))
        .expect_err("headless resume must reject its backend before stale validation");
    assert_eq!(error.code(), ErrorCode::UnsupportedBackend);
}

#[test]
fn device_loss_is_terminal_idempotent_and_releases_device_resources() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width = u32::try_from(expected.len() / 4).expect("the loss fixture width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .unwrap();
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap();
    let warmed = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters,
        Parameters::default(),
        working_format,
    ))
    .expect("device-loss coverage must first retain one color-filter graph frame");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the warmed color-filter device must remain ready before loss");
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache = ready.device_pass_cache_counts_for_test();
    assert!(
        warmed.output_extent == PhysicalSize::new(width, 1)
            && resources.entry_count > 0
            && resources.effect_texture_count_for_test() > 0
            && cache.has_render_pipelines(),
        "device-loss coverage did not first retain exact color-filter resources"
    );

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("a signaled device loss must prevent further Vello use");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );
    assert!(renderer.default_device_renderer_released_for_test());
    assert!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .is_none()
    );
}

#[test]
fn uncaptured_gpu_error_faults_only_its_device_generation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut faulted =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let healthy_slot = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("device-isolation coverage requires a second ready device slot");
    let mut healthy =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut healthy.backend
    else {
        panic!("the test environment must create a device-backed healthy surface");
    };
    *device_identity = healthy_slot;

    let idle_slot = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("no-active-generation coverage requires a third ready device slot");
    let mut idle = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    assert_eq!(
        idle.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "the idle donor surface must not carry resources created by another device"
    );
    let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut idle.backend
    else {
        panic!("the idle donor test requires a pending device-backed surface");
    };
    *device_identity = idle_slot;
    assert_eq!(idle.device_identity(), Some(idle_slot));

    let (device, queue, active_signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let generation = active_signal.next_test_generation().unwrap();
    let transaction = super::gpu_transaction::GpuOperationTransaction::begin(
        &device,
        Arc::clone(&active_signal),
        generation,
        GpuOperationStage::Render,
    );
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit active-generation fault transaction"),
        })
        .finish();
    let error = pollster::block_on(fault_command_buffer_after_submit_for_test(
        transaction,
        &queue,
        command_buffer,
        &active_signal,
        GpuFaultKind::Validation,
        "active fault",
        RuntimeOperation::SurfaceRendering,
    ))
    .expect_err("an active-generation uncaptured fault must fail its transaction");
    assert_eq!(
        active_signal
            .first_terminal()
            .expect("the uncaptured error must terminally fault the active device")
            .operation_generation_for_test(),
        Some(generation)
    );
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(active_signal.active_generation_for_test(), None);
    assert!(renderer.default_device_renderer_released_for_test());
    assert_eq!(
        renderer.runtime_capabilities(&faulted),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceFaulted {
            kind: GpuFaultKind::Validation,
        }),
    );
    let error =
        pollster::block_on(renderer.render(&mut faulted, &Scene::new(), Parameters::default()))
            .expect_err(
                "the next default-device operation must report the terminal uncaptured fault",
            );
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::DeviceFaulted {
                    kind: GpuFaultKind::Validation,
                },
            )
            .unwrap()
        )
    );

    pollster::block_on(renderer.render(&mut healthy, &Scene::new(), Parameters::default()))
        .expect("a healthy device slot and its surface must continue after another slot faults");
    assert!(matches!(
        renderer.runtime_capabilities(&healthy),
        RuntimeCapabilities::Available(_)
    ));

    assert_idle_uncaptured_fault_is_isolated(&mut renderer, idle_slot, &mut idle, &mut healthy);
}

fn assert_idle_uncaptured_fault_is_isolated(
    renderer: &mut Renderer,
    idle_slot: DeviceSlotIdentity,
    idle: &mut Surface,
    healthy: &mut Surface,
) {
    let signal = renderer
        .device_signal_for_test(idle_slot)
        .expect("the idle device slot must retain its real DeviceSignal");
    assert_eq!(signal.active_generation_for_test(), None);
    renderer.signal_device_uncaptured_fault_for_test(idle_slot, GpuFaultKind::Internal);
    assert_eq!(
        signal
            .first_terminal()
            .expect("an idle uncaptured fault must terminally affect its own device slot")
            .operation_generation_for_test(),
        None
    );
    let error = pollster::block_on(renderer.render(idle, &Scene::new(), Parameters::default()))
        .expect_err("the next operation naming the idle faulted slot must reject it");
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::DeviceFaulted {
                    kind: GpuFaultKind::Internal,
                },
            )
            .unwrap()
        )
    );
    assert_eq!(
        signal.active_generation_for_test(),
        None,
        "terminal preflight must not begin an idle-slot GPU operation"
    );
    assert!(
        renderer.device_renderer_released_for_test(idle_slot),
        "terminal preflight must release the idle slot without resource use"
    );
    pollster::block_on(renderer.render(healthy, &Scene::new(), Parameters::default()))
        .expect("the healthy slot must remain usable after active and idle faults elsewhere");
}

#[test]
fn surgeist_device_state_owns_selected_wgpu_handles() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("ready-device ownership coverage requires a real selected WGPU device");

    assert_ready_device_state_exposes_owned_wgpu_handles(&ready);
    assert!(
        std::ptr::eq(
            ready.checked_pipeline_for_test(),
            ready.checked_pipeline_for_test()
        ),
        "the ready DeviceState must retain one checked internal-engine pipeline"
    );
    assert!(
        ready.internal_resources_empty_for_test(),
        "a newly selected device must begin with a valid empty internal resource owner"
    );
}

#[test]
fn terminal_device_cleanup_drops_internal_engine_resources() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let _surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("terminal device cleanup coverage requires a real selected WGPU device");

        assert_ready_device_state_exposes_owned_wgpu_handles(&ready);
        assert!(
            std::ptr::eq(
                ready.checked_pipeline_for_test(),
                ready.checked_pipeline_for_test()
            ),
            "the ready DeviceState must retain its checked internal-engine pipeline"
        );
        assert!(
            ready.internal_resources_empty_for_test(),
            "the ready DeviceState must retain an accessible internal resource owner"
        );
    }

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    assert!(renderer.default_device_renderer_released_for_test());
    assert!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .is_none(),
        "the terminal transition must make the typed ready ownership borrow inaccessible"
    );
}

#[test]
fn one_ready_device_owns_one_raster_and_effect_resource_manager() {
    let options = Options::default().with_resource_cache_budget(ResourceCacheBudget::DISABLED);
    let mut renderer = pollster::block_on(Renderer::new(options))
        .expect("resource ownership coverage requires a real selected WGPU device");
    let manager_identity = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("resource ownership coverage requires one ready device state");
        let manager_identity = ready.sole_resource_manager_identity_for_test();
        assert_eq!(
            ready.resource_cache_budget_for_test(),
            ResourceCacheBudget::DISABLED,
            "the ready device manager must retain the renderer's fixed zero budget"
        );
        manager_identity
    };
    assert!(
        manager_identity.is_some(),
        "raster and effect allocations still have competing owners"
    );

    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 1.0, 1.0)).unwrap();
    let mut scene = VelloScene::default();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 1.0, 1.0),
    );
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let context = renderer
        .default_offscreen_render_context()
        .expect("resource ownership coverage requires one ready device context");
    let output = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &scene,
        request,
    ))
    .expect("raster and transitional effect allocations must use the ready device manager");

    let observation = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("resource ownership coverage requires the same ready device state");
        assert_eq!(
            ready.sole_resource_manager_identity_for_test(),
            manager_identity,
            "raster and effect allocations must retain one manager identity"
        );
        ready.internal_resource_manager_observation_for_test()
    };
    assert_eq!(observation.leased_count, 1);
    assert_eq!(observation.idle_count, 0);
    assert!(
        observation.next_resource > output.target().resource_id(),
        "Vello raster allocations and the transitional texture must share one identity sequence"
    );

    output.release().unwrap();
    let cleaned = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("resource ownership coverage requires the same ready device state")
        .internal_resource_manager_observation_for_test();
    assert_eq!(cleaned.leased_count, 0);
    assert_eq!(cleaned.idle_count, 0);
    assert_eq!(cleaned.entry_count, 0);
}

#[test]
fn encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("internal Vello transaction coverage requires a real selected WGPU device");
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the base direct scene must prepare without WGPU submission authority");

    assert!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("internal Vello transaction coverage requires the owned per-device Vello state")
            .internal_resources_empty_for_test(),
        "the actual per-device manager must begin empty before the transaction owns the lease"
    );

    pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
        .expect("the transaction must submit and finish the checked internal Vello pass cleanly");

    assert!(
        !renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the selected device must remain ready after clean scopes")
            .internal_resources_empty_for_test(),
        "a checked Vello lease must be submitted and explicitly adopted by the per-device manager"
    );
}

#[test]
fn direct_vello_submission_reports_accounting_fault_after_real_submit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("direct Vello accounting coverage requires a real selected WGPU device");
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the direct Vello accounting fixture must prepare without submission");
    let error = match pollster::block_on(
        renderer.fault_prepared_vello_accounting_after_submit_for_test(&prepared, target_extent),
    ) {
        Ok(_) => {
            panic!("direct Vello submission silently ignored terminal resource accounting cleanup")
        }
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(
        error.message(),
        "resource manager is unavailable after a retained-byte accounting invariant failure"
    );
    let after_fault = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the accounting fault must retain the ready device for diagnosis")
        .internal_resource_manager_observation_for_test();
    assert!(matches!(
        after_fault.accounting_fault_for_test(),
        Some(ResourceAccountingFault::RetainedByteMismatch { .. })
    ));
    assert_eq!(after_fault.active_frame_count, 0);
    assert_eq!(after_fault.leased_count, 0);
    assert_eq!(
        after_fault.entry_count, 0,
        "the failed direct Vello commit retained submitted-but-uncertain identities"
    );

    let retry = match pollster::block_on(
        renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent),
    ) {
        Ok(_) => panic!("a faulted direct Vello manager must block later acquisition"),
        Err(error) => error,
    };
    assert_eq!(retry.code(), ErrorCode::RenderFailed);
}

#[test]
fn internal_vello_encoding_shares_the_frame_transaction_submission() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).expect(
        "internal Vello transaction submission coverage requires a real selected WGPU device",
    );
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the base direct scene must prepare without WGPU submission authority");

    let observation =
        pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
            .expect("the frame transaction must submit the checked internal Vello payload");

    assert_eq!(
        observation.queue_submission_count_for_test(),
        1,
        "the internal payload must use exactly one real frame transaction queue submission"
    );
    assert_eq!(
        observation.payload_raster_pass_count_for_test(),
        1,
        "the one consumed internal payload command buffer must be the direct raster pass"
    );
    assert_eq!(
        observation.active_generation_for_test(),
        observation.transaction_generation_for_test(),
        "the real queue submission must retain the active DeviceSignal generation for its transaction lease"
    );
    assert!(
        observation
            .transaction_generation_for_test()
            .is_some_and(|generation| generation != 0),
        "the real queue submission must retain its nonzero frame operation generation"
    );
    assert_eq!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the selected device must remain ready after a clean transaction")
            .internal_resource_manager_observation_for_test()
            .retained_count_for_test(),
        1,
        "the clean transaction must adopt its one committed internal resource lease"
    );
}

#[test]
fn direct_vello_scene_uses_one_pass_and_no_effect_allocation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("direct-raster allocation coverage requires a real selected WGPU device");
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the base direct scene must prepare without effect-graph authority");

    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the direct scene must begin with a real ready resource manager")
        .internal_resource_manager_observation_for_test();
    let observation =
        pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
            .expect("the direct scene must submit through its one internal raster payload");
    assert_eq!(
        observation.payload_raster_pass_count_for_test(),
        1,
        "the effect-free direct scene must consume exactly one internal raster payload pass"
    );

    let allocation_summary = observation.allocation_summary_for_test();
    assert!(
        allocation_summary.as_ref().is_some_and(|summary| {
            summary.internal_vello_raster_buffer_requests_for_test() > 0
                && summary.internal_vello_raster_buffer_allocations_for_test() > 0
                && summary.internal_vello_raster_image_requests_for_test() > 0
                && summary.internal_vello_raster_image_allocations_for_test() > 0
        }),
        "the transaction-owned direct payload must carry actual internal Vello buffer/image allocation roles"
    );
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the direct scene must retain its real ready resource manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_before.effect_texture_count_for_test(), 0);
    assert_eq!(resources_after.effect_texture_count_for_test(), 0);
}

#[test]
fn direct_vello_succeeds_when_effect_working_format_is_unavailable() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("direct format-independence coverage requires a real selected WGPU device");
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .expect("direct format-independence coverage requires a real headless surface");
    let mut baseline = Scene::new();
    baseline.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &baseline, Parameters::default()))
        .expect("the direct baseline must establish a readable publication");
    let baseline_pixels = pollster::block_on(renderer.read_headless(&surface))
        .expect("the direct baseline publication must be readable");
    let publication_before = surface.headless_publication_count_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the direct baseline must retain its ready device")
        .internal_resource_manager_observation_for_test();
    assert!(
        renderer.override_default_device_effect_precision_facts_for_test(
            EffectPrecisionCapabilities::new(false, false),
        ),
        "the real renderer must accept the scoped no-effect-format capability facts"
    );

    let mut replacement = Scene::new();
    replacement.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    );
    let result =
        pollster::block_on(renderer.render(&mut surface, &replacement, Parameters::default()));
    if let Err(error) = &result {
        let expected = RuntimeCapabilityUnavailable::try_new(
            RuntimeOperation::EffectRendering,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable {
                policy: EffectQualityPolicy::RequireHighPrecision,
            },
        )
        .unwrap();
        assert_eq!(
            error.runtime_capability_unavailable_diagnostic(),
            Some(&expected),
            "the direct regression must fail only at the premature effect-format gate"
        );
    }
    let stats = result.expect("direct Vello must not require an effect working format");
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the direct replacement must retain its ready device")
        .internal_resource_manager_observation_for_test();
    let no_effect_resources = resources_before.effect_texture_count_for_test() == 0
        && resources_after.effect_texture_count_for_test() == 0
        && resources_before
            .resolved_mask_upload_keys_for_test()
            .is_empty()
        && resources_after
            .resolved_mask_upload_keys_for_test()
            .is_empty()
        && resources_before.gaussian_kernel_count_for_test() == 0
        && resources_after.gaussian_kernel_count_for_test() == 0;
    let selected_direct_vello = stats.route == Some(RenderRoute::DirectVello);
    let replacement_pixels = pollster::block_on(renderer.read_headless(&surface))
        .expect("the successful direct replacement publication must be readable");

    assert!(
        selected_direct_vello
            && no_effect_resources
            && surface.headless_publication_count_for_test()
                == publication_before.saturating_add(1)
            && renderer.stats() == stats
            && replacement_pixels != baseline_pixels
            && replacement_pixels
                .rgba()
                .chunks_exact(4)
                .all(|pixel| pixel == [255, 255, 255, 255]),
        "direct Vello used graph resources, selected the wrong route, or corrupted publication"
    );
}

#[test]
fn repeated_direct_renders_keep_internal_vello_retention_bounded() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("retention coverage requires a real selected WGPU device");
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("retention coverage requires a real headless surface");
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    let mut observations = Vec::new();
    for _ in 0..4 {
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("each production direct raster render must succeed");
        observations.push(
            renderer
                .default_ready_device_state_borrow_for_test()
                .expect("the selected device must remain ready after direct rendering")
                .internal_resource_manager_observation_for_test(),
        );
    }

    let retained_counts = observations
        .iter()
        .map(|observation| observation.retained_count_for_test())
        .collect::<Vec<_>>();
    let retained_byte_lengths = observations
        .iter()
        .map(|observation| observation.retained_byte_len_for_test())
        .collect::<Vec<_>>();
    assert_eq!(
        retained_counts,
        vec![1; observations.len()],
        "equal direct production frames must retain one current allocation; observed retained counts {retained_counts:?}, bytes {retained_byte_lengths:?}"
    );
    assert!(
        retained_byte_lengths
            .windows(2)
            .all(|pair| pair[0] == pair[1]),
        "equal direct production frames must not increase retained bytes; observed retained counts {retained_counts:?}, bytes {retained_byte_lengths:?}"
    );

    for observation in observations {
        assert_eq!(
            observation.retained_atlas_count_for_test(),
            1,
            "each clean direct frame must retain exactly one current persistent atlas"
        );
        assert!(
            observation.retained_atlas_byte_len_for_test() > 0,
            "the retained atlas must report only its known Rgba8Unorm byte length"
        );
        assert_eq!(
            observation.committed_transient_buffer_count_for_test(),
            0,
            "clean commits must discard every transient buffer"
        );
        assert_eq!(
            observation.committed_transient_buffer_byte_len_for_test(),
            0,
            "clean commits must discard transient buffer bytes"
        );
        assert_eq!(
            observation.committed_transient_image_count_for_test(),
            0,
            "clean commits must discard every transient image"
        );
        assert_eq!(
            observation.committed_transient_image_byte_len_for_test(),
            0,
            "clean commits must discard transient image bytes"
        );
    }
}

#[test]
fn canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("Vello cancellation coverage requires a real selected WGPU device");
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the base direct scene must prepare without WGPU submission authority");

    let initial = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("Vello cancellation coverage requires the owned per-device state")
        .internal_resource_manager_observation_for_test();
    assert_eq!(initial.retained_count_for_test(), 0);
    assert_eq!(initial.recovery_outcome_for_test(), None);

    pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
        .expect("the first clean pass must retain its current persistent atlas");
    let prior_clean = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the selected device must remain ready after the first clean pass")
        .internal_resource_manager_observation_for_test();
    assert_eq!(prior_clean.retained_count_for_test(), 1);
    assert_eq!(prior_clean.retained_atlas_count_for_test(), 1);
    assert!(prior_clean.retained_atlas_byte_len_for_test() > 0);

    let canceled = pollster::block_on(
        renderer.cancel_prepared_vello_pass_after_submit_for_test(&prepared, target_extent),
    )
    .expect(
        "the cancellation adapter must encode, submit, reach its post-submit checkpoint, and drop locally",
    );
    assert_eq!(
        canceled.retained_count_for_test(),
        prior_clean.retained_count_for_test(),
        "a canceled new lease must preserve the prior clean retained atlas"
    );
    assert_eq!(
        canceled.retained_atlas_byte_len_for_test(),
        prior_clean.retained_atlas_byte_len_for_test(),
        "a canceled new lease must not replace or drop the prior clean atlas"
    );
    assert_eq!(
        canceled.recovery_outcome_for_test(),
        Some(VelloAtlasOutcome::Recreate),
        "the fresh atlas allocation must derive Recreate from its aborted lease provenance"
    );

    pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
        .expect("the next clean pass must recover before retaining fresh internal resources");
    let recovered = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the selected device must remain ready after recovery")
        .internal_resource_manager_observation_for_test();
    assert_eq!(recovered.retained_count_for_test(), 1);
    assert_eq!(recovered.retained_atlas_count_for_test(), 1);
    assert_eq!(
        recovered.retained_atlas_byte_len_for_test(),
        prior_clean.retained_atlas_byte_len_for_test(),
        "the later clean transaction must replace the atlas without increasing retention"
    );
    assert_eq!(
        recovered.recovery_outcome_for_test(),
        None,
        "the next clean pass must consume the prior atlas recovery before retaining fresh resources"
    );
}

#[test]
fn canceled_vello_atlas_recovery_survives_preallocation_failure() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("Vello atlas recovery coverage requires a real selected WGPU device");
    let target_extent = PhysicalSize::new(64, 48);
    let prepared = VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("a non-empty direct Vello target must prepare"),
    )
    .expect("the base direct scene must prepare without WGPU submission authority");

    let canceled = pollster::block_on(
        renderer.cancel_prepared_vello_pass_after_submit_for_test(&prepared, target_extent),
    )
    .expect("the real submitted cancellation must establish atlas recovery");
    assert_eq!(
        canceled.recovery_outcome_for_test(),
        Some(VelloAtlasOutcome::Recreate)
    );

    let preallocation_failure = match pollster::block_on(
        renderer.submit_prepared_vello_pass_for_test(&prepared, PhysicalSize::new(63, 48)),
    ) {
        Ok(_) => panic!("a mismatched target must fail before internal Vello resource allocation"),
        Err(error) => error,
    };
    assert_eq!(preallocation_failure.code(), ErrorCode::RenderFailed);

    let pending = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the selected device must remain ready after the pre-allocation failure")
        .internal_resource_manager_observation_for_test();
    assert_eq!(pending.retained_count_for_test(), 0);
    assert_eq!(
        pending.recovery_outcome_for_test(),
        Some(VelloAtlasOutcome::Recreate),
        "a pre-allocation failure must not clear recovery from the canceled submitted pass"
    );

    pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
        .expect("the next clean pass must consume recovery before retaining its lease");
    let recovered = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the selected device must remain ready after recovery")
        .internal_resource_manager_observation_for_test();
    assert_eq!(recovered.retained_count_for_test(), 1);
    assert_eq!(recovered.recovery_outcome_for_test(), None);
}

fn assert_ready_device_state_exposes_owned_wgpu_handles(ready: &ReadyDeviceStateBorrowForTest<'_>) {
    let adapter = ready.adapter_for_test();
    let device = ready.device_for_test();
    let queue = ready.queue_for_test();

    assert!(
        adapter.features().contains(device.features()),
        "the ready DeviceState device must expose only features supported by its selected adapter"
    );
    assert!(
        device.limits().max_texture_dimension_2d <= adapter.limits().max_texture_dimension_2d,
        "the ready DeviceState device limits must come from its selected adapter"
    );
    assert!(
        queue.get_timestamp_period().is_finite(),
        "the ready DeviceState queue must be directly accessible through the selected handle bundle"
    );
}

#[test]
fn terminal_default_device_rejects_headless_without_disabling_ready_slots() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    let error = match pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)) {
        Ok(_) => panic!("a terminal default device must not be replaced automatically"),
        Err(error) => error,
    };
    assert_runtime_device_lost(
        error,
        RuntimeOperation::AdapterSelection,
        DeviceLossReason::Destroyed,
    );
}

#[test]
fn runtime_capabilities_project_the_selected_surface_without_gpu_work() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();

    let report = renderer.runtime_capabilities(&surface);
    let available = report
        .available()
        .expect("a device-backed headless surface must project immutable capabilities");
    assert_eq!(available.surface_format(), Format::Rgba8);
    assert_eq!(
        available,
        renderer.default_device_capabilities_for_test(),
        "the query must project the snapshotted state without another GPU call"
    );
}

#[test]
fn destroyed_device_callback_reports_terminal_loss_without_stale_resource_use() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let ready_slot = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("the destroyed-device test requires a second real WGPU device slot");
    let device_signal = renderer
        .default_device_signal_for_test()
        .expect("the destroyed-device test requires the default device callback signal");

    assert!(renderer.destroy_default_device_for_test());
    let terminal_timeout = Duration::from_secs(5);
    let terminal_wait_started = std::time::Instant::now();
    let terminal_observed = renderer.wait_for_default_terminal_signal_for_test(terminal_timeout);
    let terminal_wait =
        device_signal.terminal_wait_observation_for_test(terminal_timeout, terminal_wait_started);
    assert!(
        terminal_observed,
        "device destruction did not invoke the loss callback within the diagnostic deadline: final_terminal={:?}; active_operation_generation={:?}; requested_timeout={:?}; elapsed={:?}",
        terminal_wait.final_terminal,
        terminal_wait.active_operation_generation,
        terminal_wait.requested_timeout,
        terminal_wait.elapsed,
    );

    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("a destroyed device must be observed before any stale Vello use");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );

    let error = match pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)) {
        Ok(_) => panic!("a destroyed device must not create another headless surface"),
        Err(error) => error,
    };
    assert_runtime_device_lost(
        error,
        RuntimeOperation::AdapterSelection,
        DeviceLossReason::Destroyed,
    );

    assert!(renderer.default_device_renderer_released_for_test());
    pollster::block_on(renderer.submit_scoped_wgpu_probe_for_test(ready_slot))
        .expect("a ready second slot must submit and finish a real scoped WGPU operation");
}

fn assert_runtime_device_lost(error: Error, operation: RuntimeOperation, reason: DeviceLossReason) {
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

fn assert_runtime_adapter_unavailable(error: &Error, operation: RuntimeOperation) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::AdapterUnavailable,
            )
            .unwrap()
        )
    );
}

fn assert_surface_unavailable(
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

fn assert_surface_identity_mismatch(
    error: Error,
    operation: RuntimeOperation,
    kind: SurfaceIdentityMismatchKind,
) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { kind },
            )
            .unwrap()
        )
    );
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn unsupported_web_canvas_attachment_reports_target_requirement() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let canvas = WebCanvas::new("preview");

    assert_eq!(canvas.id(), "preview");

    let error = match pollster::block_on(renderer.create_surface(
        Attachment::WebCanvas(canvas),
        SurfaceOptions {
            size: Size::new(10.0, 10.0),
            ..SurfaceOptions::default()
        },
    )) {
        Ok(_) => panic!("native test targets should not create web canvas surfaces"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))
    );
    assert!(error.message().contains("web canvas surface"));
}

#[test]
fn direct_vello_stats_report_exact_route_and_single_raster_pass() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("a direct internal-Vello frame should publish render observations");

    assert_eq!(stats.route, Some(RenderRoute::DirectVello));
    assert_eq!(stats.effect_precision, None);
    assert_eq!(stats.vello_passes, 1);
    assert_eq!(stats.image_passes, 0);
    assert_eq!(stats.composite_passes, 0);
    assert_eq!(stats.copy_operations, 0);
    assert_eq!(stats.custom_present_passes, 0);
    assert_eq!(stats.effect_texture_allocations, 0);
    assert_eq!(stats.effect_texture_reuses, 0);
    assert_eq!(stats.retained_effect_bytes, 0);
    assert_eq!(renderer.stats(), stats);
}

#[test]
fn non_render_operations_do_not_mutate_last_successful_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert_eq!(renderer.stats(), Stats::default());
    let _ = renderer.capabilities();
    assert_eq!(renderer.stats(), Stats::default());

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    assert_eq!(renderer.stats(), Stats::default());

    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let last_successful =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("the baseline direct frame should publish stats");
    assert_eq!(last_successful.route, Some(RenderRoute::DirectVello));

    let _ = renderer.capabilities();
    assert_eq!(renderer.stats(), last_successful);
    let _ = renderer.runtime_capabilities(&surface);
    assert_eq!(renderer.stats(), last_successful);
    let _ = pollster::block_on(renderer.read_headless(&surface))
        .expect("explicit readback should observe the published frame");
    assert_eq!(renderer.stats(), last_successful);

    let _other = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .expect("surface creation should remain independent from render stats");
    assert_eq!(renderer.stats(), last_successful);
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert_eq!(renderer.stats(), last_successful);
    surface.suspend().unwrap();
    assert_eq!(renderer.stats(), last_successful);
    surface.resume(Attachment::Headless).unwrap();
    assert_eq!(renderer.stats(), last_successful);
}

#[test]
fn gpu_graph_stats_count_exact_backdrop_passes_copies_resources_and_precision() {
    let (scene, size, parameters, _) = bounded_backdrop_integration_fixture_for_test();
    for (working_format, precision) in [
        (WorkingFormat::HighPrecision, EffectPrecision::High),
        (WorkingFormat::ReducedPrecision, EffectPrecision::Reduced),
    ] {
        let stats =
            render_bounded_backdrop_fixture_for_test(&scene, size, parameters, working_format)
                .result
                .stats;
        assert_eq!(
            (
                stats.route,
                stats.effect_precision,
                stats.vello_passes,
                stats.image_passes,
                stats.composite_passes,
                stats.copy_operations,
                stats.custom_present_passes,
            ),
            (Some(RenderRoute::GpuGraph), Some(precision), 3, 9, 6, 1, 1,)
        );
        assert!(stats.effect_texture_allocations > 0);
        assert_eq!(stats.effect_texture_reuses, 0);
        assert!(stats.retained_effect_bytes > 0);
    }
}

#[test]
fn resource_stats_report_acquisition_source_and_post_trim_retention() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect("resource-stat coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(6.0, 4.0), 1.0))
        .expect("resource-stat coverage requires a headless surface");
    let scene = repeated_graph_scene_for_test();

    let first = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the first resource-stat graph must succeed");
    let second = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the repeated resource-stat graph must succeed");
    let resources = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("resource-stat coverage must retain its ready device")
        .internal_resource_manager_observation_for_test();

    assert!(first.stats.effect_texture_allocations > 0);
    assert_eq!(first.stats.effect_texture_reuses, 0);
    assert!(second.stats.effect_texture_reuses > 0);
    assert_eq!(second.stats.retained_effect_bytes, resources.retained_bytes);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.active_frame_count, 0);
}

#[test]
fn failed_and_canceled_graph_frames_preserve_last_successful_stats() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect("graph stats failure coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(6.0, 4.0), 1.0))
        .expect("graph stats failure coverage requires a headless surface");
    let scene = repeated_graph_scene_for_test();
    let successful = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the graph stats baseline must succeed")
    .stats;
    assert_eq!(successful.route, Some(RenderRoute::GpuGraph));
    assert!(successful.effect_texture_allocations > 0);

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::default();
    let mut publication = Some(1);
    pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(904),
        &mut publication,
    ))
    .expect_err("the explicit submitted transaction failure must not publish stats");
    assert_eq!(publication, Some(1));
    assert_eq!(renderer.stats(), successful);

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let canceled_resources = ResourceManager::default();
    let mut canceled_publication = Some(1);
    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &canceled_resources,
            modeled_resource_key_for_test(905),
            &mut canceled_publication,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }
    assert_eq!(canceled_publication, Some(1));
    let canceled_resources = canceled_resources.observation_for_test();
    assert_eq!(canceled_resources.active_frame_count, 0);
    assert_eq!(canceled_resources.leased_count, 0);
    assert_eq!(canceled_resources.entry_count, 0);
    assert_eq!(renderer.stats(), successful);
}

#[test]
fn render_reports_command_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
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

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
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
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(20.0, 20.0), 2.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(output.size(), PhysicalSize::new(40, 40));
    assert!(pixel_alpha(&output, 18, 18) > 0);
    assert_eq!(pixel_alpha(&output, 22, 22), 0);
}

#[test]
fn warm_image_reuse_reports_cache_hit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    assert_eq!(image_data(&image), image_data(&image.clone()));
    let mut scene = Scene::new();
    scene.image(
        image.clone(),
        Rect::new(0.0, 0.0, 1.0, 1.0),
        ImageFit::Stretch,
    );

    let cold =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let warm =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();

    assert_eq!(cold.cache_misses, 1);
    assert_eq!(warm.cache_hits, 1);
}

#[test]
fn failed_render_does_not_warm_image_reuse_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
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

    let error = pollster::block_on(renderer.render(&mut surface, &failing, Parameters::default()))
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);

    let mut valid = Scene::new();
    valid.image(image, Rect::new(0.0, 0.0, 1.0, 1.0), ImageFit::Stretch);

    let stats = pollster::block_on(renderer.render(&mut surface, &valid, Parameters::default()))
        .expect("valid render should still see cold image");

    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn concrete_color_paint_renders_without_color_realization() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("concrete color paint should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

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
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), gradient);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("gradient paint should render");
}

#[test]
fn image_paint_lowers_to_brush() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Paint::image(image));

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

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
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut pixels = Vec::new();
    for _ in 0..8 {
        pixels.extend_from_slice(&[255, 0, 0, 255]);
    }
    let image = Image::from_rgba(Size::new(4.0, 2.0), Arc::<[u8]>::from(pixels)).unwrap();
    let mut scene = Scene::new();
    scene.image(image, Rect::new(1.0, 0.0, 2.0, 2.0), ImageFit::Cover);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

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
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.transform(
        Transform::try_new([1.0, 0.0, 0.0, 1.0, 2.0, 0.0]).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn composed_layer_transforms_render_in_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 2.0), 1.0)).unwrap();
    let transform = Transform::translation(1.0, 0.0)
        .unwrap()
        .then(Transform::scale(2.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 2.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("composed transform should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn origin_wrapped_layer_transform_renders_about_origin() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let transform = Transform::scale(2.0, 2.0)
        .unwrap()
        .around(Point::try_new(1.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(1.0, 1.0, 1.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("origin-wrapped transform should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
    assert!(pixel_alpha(&output, 2, 2) > 0);
}

#[test]
fn transformed_shape_clips_render_in_layer_space() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
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

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("transformed clip should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn path_clip_fill_rules_execute_even_odd_and_nonzero() {
    fn nested_rect_path() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(5.0, 0.0))
            .line_to(Point::new(5.0, 5.0))
            .line_to(Point::new(0.0, 5.0))
            .close()
            .move_to(Point::new(1.0, 1.0))
            .line_to(Point::new(4.0, 1.0))
            .line_to(Point::new(4.0, 4.0))
            .line_to(Point::new(1.0, 4.0))
            .close();
        path
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut even_odd_surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 5.0), 1.0)).unwrap();
    let even_odd_clip = ClipInput::try_filled_path(
        FilledPath::try_new(nested_rect_path(), FillRule::EvenOdd).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_clip_input(even_odd_clip).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 6.0, 5.0), Color::BLACK);
        },
    );
    pollster::block_on(renderer.render(&mut even_odd_surface, &scene, Parameters::default()))
        .expect("even-odd path clip should render");
    let even_odd = pollster::block_on(renderer.read_headless(&even_odd_surface)).unwrap();

    let mut nonzero_surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 5.0), 1.0)).unwrap();
    let nonzero_clip = ClipInput::try_filled_path(
        FilledPath::try_new(nested_rect_path(), FillRule::NonZero).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_clip_input(nonzero_clip).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 6.0, 5.0), Color::BLACK);
        },
    );
    pollster::block_on(renderer.render(&mut nonzero_surface, &scene, Parameters::default()))
        .expect("nonzero path clip should render");
    let nonzero = pollster::block_on(renderer.read_headless(&nonzero_surface)).unwrap();

    assert!(pixel_alpha(&even_odd, 0, 0) > 0);
    assert_eq!(pixel_alpha(&even_odd, 2, 2), 0);
    assert!(pixel_alpha(&nonzero, 2, 2) > 0);
}

#[test]
fn builtin_shape_clips_execute_for_layer_clipping() {
    let clips = [
        Shape::try_rounded_rect(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Radii::new(1.0, 1.0, 1.0, 1.0),
        )
        .unwrap(),
        Shape::try_circle(Point::new(2.0, 2.0), 2.0).unwrap(),
        Shape::try_ellipse(Point::new(2.0, 2.0), Size::new(2.0, 1.5)).unwrap(),
    ];

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for clip in clips {
        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(Layer::new().try_clip(clip).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
        });

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("builtin shape clip should render as a layer clip");
        let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

        assert!(
            output.rgba().chunks_exact(4).any(|pixel| pixel[3] > 0),
            "builtin shape clip should leave visible clipped content"
        );
    }
}

#[test]
fn nested_clips_render_only_the_intersection() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(5.0, 2.0), 1.0)).unwrap();
    let mut inner_path = Path::new();
    inner_path
        .move_to(Point::new(2.0, 0.0))
        .line_to(Point::new(5.0, 0.0))
        .line_to(Point::new(5.0, 2.0))
        .line_to(Point::new(2.0, 2.0))
        .close();
    let inner_clip =
        ClipInput::try_filled_path(FilledPath::try_new(inner_path, FillRule::NonZero).unwrap())
            .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(Rect::new(1.0, 0.0, 3.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.layer(Layer::new().try_clip_input(inner_clip).unwrap(), |scene| {
                scene.fill(Rect::new(0.0, 0.0, 5.0, 2.0), Color::BLACK);
            });
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("nested clips should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
    assert_eq!(pixel_alpha(&output, 4, 0), 0);
}

#[test]
fn coordinate_space_tag_transform_affects_layer_clip() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
        .unwrap()
        .with_coordinate_space(
            CoordinateSpaceTag::surface(Transform::translation(2.0, 0.0).unwrap()).unwrap(),
        );
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_clip_input(clip).unwrap(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("coordinate-space clip transform should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn scene_clip_convenience_still_uses_shape_layer_clips() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.clip(Rect::new(1.0, 0.0, 1.0, 1.0), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("existing Scene::clip convenience should keep working");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 0);
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn transformed_images_render_in_layer_space() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.transform(Transform::translation(2.0, 0.0).unwrap(), |scene| {
        scene.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("transformed image should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

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

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
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
fn layer_pass_plan_uses_clip_bounds_before_child_geometry() {
    let clip = Layer::new()
        .try_clip(Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0)))
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(clip, |scene| {
        scene.fill(Rect::new(-10.0, -10.0, 50.0, 50.0), Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(layer.isolation, command::LayerIsolation::ClipOnly);
    assert_eq!(layer.pass_plan.kind(), command::LayerPassKind::ClipOnly);
    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::ClipOnly
    );
    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(1.0, 2.0, 3.0, 4.0))
    );
}

#[test]
fn layer_pass_plan_names_opacity_and_blend_direct_layers() {
    let opacity = Layer::new().try_opacity(0.5).unwrap();
    let blend = Layer::new().blend(BlendMode::Multiply);
    let mut scene = Scene::new();
    scene
        .layer(opacity, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        })
        .layer(blend, |scene| {
            scene.fill(Rect::new(4.0, 0.0, 2.0, 2.0), Color::BLACK);
        });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let plans: Vec<_> = normalized
        .commands
        .iter()
        .map(|command| match command {
            command::RenderCommand::Layer { layer, .. } => (
                layer.isolation,
                layer.pass_plan.kind(),
                layer.pass_plan.requirement(),
            ),
            _ => panic!("expected layer command"),
        })
        .collect();

    assert_eq!(
        plans,
        [
            (
                command::LayerIsolation::BackendLayer,
                command::LayerPassKind::DirectVelloLayer,
                command::LayerPassRequirement::DirectVelloOpacity,
            ),
            (
                command::LayerIsolation::BackendLayer,
                command::LayerPassKind::DirectVelloLayer,
                command::LayerPassRequirement::DirectVelloBlend,
            ),
        ]
    );
}

#[test]
fn nested_layer_pass_plan_aggregates_transformed_child_bounds() {
    let outer = Layer::new().try_opacity(0.5).unwrap();
    let inner = Layer::new()
        .try_transform(Transform::translation(4.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(outer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        scene.layer(inner, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 3.0, 2.0), Color::BLACK);
        });
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected outer layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(0.0, 0.0, 7.0, 3.0))
    );
}

#[test]
fn path_stroke_layer_bounds_include_miter_limit_conservatively() {
    let mut path = Path::new();
    path.move_to(Point::new(10.0, 10.0))
        .line_to(Point::new(20.0, 10.0));
    let stroke = Stroke::try_new(4.0).unwrap().try_miter_limit(10.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.stroke(Shape::path(path), stroke, Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(-10.0, -10.0, 50.0, 40.0))
    );
}

#[test]
fn exact_epsilon_opacity_with_clip_keeps_backend_layer_isolation() {
    let opacity = 1.0 - f32::EPSILON;
    assert_eq!((opacity - 1.0).abs(), f32::EPSILON);
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap()
        .try_opacity(opacity)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(layer.isolation, command::LayerIsolation::BackendLayer);
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
}

#[test]
fn layer_default_is_visible() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::default(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("default layer should render visible content");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(stats.layers, 1);
    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn layer_opacity_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("opacity layer should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    let [_, _, _, alpha] = pixel_rgba(&output, 0, 0);

    assert_eq!(stats.layers, 1);
    assert!(alpha > 0);
    assert!(alpha < 255);
}

#[test]
fn layer_blend_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
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

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("blend layer should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
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
fn direct_vello_blend_modes_match_reference_oracle_for_opaque_pixels() {
    let source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    for mode in [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ] {
        let mut scene = Scene::new();
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(destination),
        );
        scene.layer(Layer::new().blend(mode), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(source),
            );
        });

        let output = render_scene_pixel(&mut renderer, &scene);
        let expected = source.blend_over(destination, mode);

        assert_rgba_near_reference_pixel(
            output,
            expected,
            2,
            &format!("direct Vello {mode:?} blend should stay aligned with the CPU oracle"),
        );
    }
}

#[test]
fn blend_layer_isolation_changes_backdrop_composition_from_normal_paint_order() {
    let source = PremultipliedRgba8::try_new(64, 128, 192, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(192, 128, 64, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let mut normal_scene = Scene::new();
    normal_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(destination),
    );
    normal_scene.layer(Layer::new(), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(source),
        );
    });

    let mut blended_scene = Scene::new();
    blended_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(destination),
    );
    blended_scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(source),
        );
    });

    let normal_output = render_scene_pixel(&mut renderer, &normal_scene);
    let blended_output = render_scene_pixel(&mut renderer, &blended_scene);
    let expected_blend = source.blend_over(destination, BlendMode::Multiply);

    assert_rgba_near_reference_pixel(
        normal_output,
        source,
        2,
        "non-isolated normal layer should paint its children in command order",
    );
    assert_rgba_near_reference_pixel(
        blended_output,
        expected_blend,
        2,
        "blend layer should isolate its child output before blending with prior backdrop",
    );
    assert_ne!(
        normal_output, blended_output,
        "multiply isolation should produce a different pixel than normal child painting"
    );
}

#[test]
fn nested_direct_vello_blend_groups_match_nested_reference_oracle() {
    let backdrop = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let outer_child_backdrop = PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap();
    let inner_source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let expected_inner = inner_source.blend_over(outer_child_backdrop, BlendMode::Multiply);
    let expected_outer = expected_inner.blend_over(backdrop, BlendMode::Screen);

    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(backdrop),
    );
    scene.layer(Layer::new().blend(BlendMode::Screen), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(outer_child_backdrop),
        );
        scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(inner_source),
            );
        });
    });

    let normalized = scene.normalize(Capabilities::CURRENT).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[1]
    else {
        panic!("expected outer blend layer command");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[1] else {
        panic!("expected nested blend layer command");
    };
    for layer in [outer, inner] {
        assert_eq!(layer.isolation, command::LayerIsolation::BackendLayer);
        assert_eq!(
            layer.pass_plan.requirement(),
            command::LayerPassRequirement::DirectVelloBlend
        );
        assert_eq!(
            layer.pass_plan.kind(),
            command::LayerPassKind::DirectVelloLayer
        );
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let output = render_scene_pixel(&mut renderer, &scene);

    assert_rgba_near_reference_pixel(
        output,
        expected_outer,
        2,
        "nested direct Vello blend groups should compose in command order",
    );
}

#[test]
fn unsupported_blend_and_composite_boundaries_remain_typed_diagnostics() {
    let public_layer_modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    assert_eq!(
        public_layer_modes.len(),
        7,
        "Task 6 should not expand layer BlendMode without encoding and tests"
    );

    for mode in [
        BackgroundBlendMode::Multiply,
        BackgroundBlendMode::Screen,
        BackgroundBlendMode::Overlay,
        BackgroundBlendMode::Darken,
        BackgroundBlendMode::Lighten,
        BackgroundBlendMode::Plus,
    ] {
        let error = BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal, mode])
            .expect_err("background-layer blending is not routed through layer BlendMode");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::BackgroundBlendMode,
            ))
        );
    }

    for operation in [
        PrimitiveOperation::AdditionalMixBlendMode,
        PrimitiveOperation::PorterDuffCompositeMode,
        PrimitiveOperation::RootBackdropPolicy,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::Compositing, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported blend/composite policy must stay behind typed diagnostics");

        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(
            error.message().contains(unsupported.label()),
            "diagnostic should name unsupported compositing boundary: {}",
            error.message()
        );
    }
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
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("prepared glyphs cannot render without font data");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.message().contains("font data"));
}

#[test]
fn text_run_with_gradient_fill_still_requires_font_data_before_brush_encoding() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            FontRef::new(1).named("Test"),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Paint::gradient(gradient)).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("prepared glyphs cannot render without font data");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.unsupported_primitive(), None);
    assert!(error.message().contains("font data"));
}

#[test]
fn inside_and_outside_strokes_lower_for_builtin_shapes() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
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

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();

    assert_eq!(stats.strokes, 2);
}

#[test]
fn aligned_rect_strokes_do_not_cross_source_edge() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(12.0, 12.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let inside = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&inside, 2, 6), 0);
    assert!(pixel_alpha(&inside, 3, 6) > 0);

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(12.0, 12.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Outside),
        Color::BLACK,
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let outside = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!(pixel_alpha(&outside, 2, 6) > 0);
    assert_eq!(pixel_alpha(&outside, 4, 6), 0);
}

#[test]
fn circle_shadows_lower_to_blurred_round_rect() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_circle(Point::new(12.0, 12.0), 4.0).unwrap(),
        Shadow::try_new(Point::new(1.0, 1.0), 4.0, 1.0, Color::BLACK).unwrap(),
    );

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba().chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn non_uniform_rounded_rect_shadows_render_with_corner_partition() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(40.0, 36.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_rounded_rect(
            Rect::new(8.0, 8.0, 16.0, 14.0),
            Radii::new(0.0, 5.0, 10.0, 0.0),
        )
        .unwrap(),
        Shadow::try_new(Point::new(4.0, 5.0), 8.0, 0.0, Color::BLACK).unwrap(),
    );

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("non-uniform rounded shadow should render through corner partitioning");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba().chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn multiple_outer_shadows_render_in_authored_order() {
    let red = Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap();
    let blue = Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, red).unwrap(),
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, blue).unwrap(),
    ])
    .unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadows(Rect::new(1.0, 1.0, 6.0, 6.0), shadows);

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    let overlap = pixel_rgba(&output, 4, 4);

    assert_eq!(stats.shadows, 2);
    assert!(
        overlap[2] > overlap[0],
        "last overlapping shadow should be composited above earlier shadows: {overlap:?}"
    );
}

#[test]
fn direct_geometry_targets_render_without_unsupported_diagnostics() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(32.0, 32.0), 1.0)).unwrap();
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

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("direct geometry targets should render");
}

#[test]
fn centered_path_strokes_support_join_cap_and_dash_inputs() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
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

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("centered path strokes should render");
}

#[test]
fn unsupported_aligned_path_strokes_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(10.0, 10.0));
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("path offsetting is deliberately explicit");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
    assert!(
        error
            .message()
            .contains("inside/outside path stroke alignment")
    );
}

#[test]
fn unsupported_layer_masks_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("mask lowering should be explicit until implemented");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert!(error.message().contains("layer mask"));
}

#[test]
fn unsupported_layer_filters_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_filter(Filter::try_blur(4.0).unwrap())
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("filter lowering should be explicit until implemented");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );
    assert!(error.message().contains("layer filter"));
}

#[test]
fn unsupported_non_solid_shadow_paint_reports_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
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

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("shadow lowering requires solid paint in this milestone");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
    assert!(error.message().contains("non-solid shadow paint"));
}

#[test]
fn ellipse_and_path_shadows_report_typed_error() {
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(6.0, 1.0))
        .line_to(Point::new(6.0, 6.0))
        .close();
    let cases = [
        (
            "ellipse",
            Shape::try_ellipse(Point::new(4.0, 4.0), Size::new(2.0, 1.0)).unwrap(),
        ),
        ("path", Shape::path(path)),
    ];

    for (label, shape) in cases {
        let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.shadow(
            shape,
            Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
        );

        let error = match pollster::block_on(renderer.render(
            &mut surface,
            &scene,
            Parameters::default(),
        )) {
            Err(error) => error,
            Ok(stats) => panic!("{label} shadow unexpectedly rendered with {stats:?}"),
        };
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive, "{label}");
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Shadows,
                PrimitiveOperation::EllipsePathShadowShape,
            )),
            "{label} shadow must retain its typed diagnostic"
        );
        assert!(
            error.message().contains("ellipse/path shadow shape"),
            "{label} shadow must retain its diagnostic label"
        );
    }
}

fn headless_direct_publication_fixture_for_test() -> (Renderer, Surface, Scene, ImageBuffer) {
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

fn prepared_direct_vello_pass_for_test(target_extent: PhysicalSize) -> PreparedVelloPass {
    VelloScene::prepare_raster_scenario_for_test(
        VelloRasterScenario::Base,
        RasterParameters::try_new(target_extent, peniko::Color::BLACK, Antialiasing::Area)
            .expect("the explicit direct Vello target must be non-empty"),
    )
    .expect("the explicit direct Vello scene must prepare without submission authority")
}

#[test]
fn headless_direct_post_submit_failure_preserves_previous_and_initial_publication() {
    let (mut renderer, surface, _replacement, published) =
        headless_direct_publication_fixture_for_test();
    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    let stats_before = renderer.stats();
    let mut publication = Some(1);
    let error = pollster::block_on(renderer.fail_prepared_vello_pass_after_submit_for_test(
        &prepared,
        target_extent,
        &mut publication,
    ))
    .expect_err("the explicit post-submit failure must abort the replacement draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    assert_eq!(
        pollster::block_on(renderer.read_headless(&surface))
            .expect("a failed frame must retain the previous publication")
            .rgba(),
        published.rgba(),
        "a failed submitted frame must not overwrite readable published pixels"
    );

    let uninitialized =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut initial_publication = None;
    pollster::block_on(renderer.fail_prepared_vello_pass_after_submit_for_test(
        &prepared,
        target_extent,
        &mut initial_publication,
    ))
    .expect_err("a failed first direct transaction must not commit its publication draft");
    assert_eq!(initial_publication, None);
    assert_eq!(
        uninitialized.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
    let error = pollster::block_on(renderer.read_headless(&uninitialized))
        .expect_err("a failed first frame must remain unreadable");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceReadback,
        RenderSurfaceAvailability::Uninitialized,
    );
}

#[test]
fn headless_direct_cancellation_after_submit_preserves_previous_publication() {
    let (mut renderer, surface, _replacement, published) =
        headless_direct_publication_fixture_for_test();
    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    pollster::block_on(
        renderer.cancel_prepared_vello_pass_after_submit_for_test(&prepared, target_extent),
    )
    .expect("the explicit canceled Vello submission must release its resources");
    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    assert_eq!(
        pollster::block_on(renderer.read_headless(&surface))
            .expect("a canceled frame must retain the previous publication")
            .rgba(),
        published.rgba(),
        "a canceled submitted frame must not overwrite readable published pixels"
    );
}

#[test]
fn headless_graph_post_submit_failure_leaves_first_frame_unpublished() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("first-frame graph failure coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::default();
    let mut publication = None;
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(901),
        &mut publication,
    ))
    .expect_err("a submitted transaction scope failure must not publish its draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, None);
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[test]
fn headless_accounting_fault_after_submit_suppresses_publication_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("headless accounting coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let mut publication = Some(1);
    let error = pollster::block_on(graph_accounting_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(902),
        &mut publication,
    ))
    .expect_err("accounting poison after submit must abort draft publication");

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    let after_fault = resources.observation_for_test();
    assert!(matches!(
        after_fault.accounting_fault_for_test(),
        Some(ResourceAccountingFault::RetainedByteMismatch { .. })
    ));
    assert_eq!(after_fault.active_frame_count, 0);
    assert_eq!(after_fault.leased_count, 0);
}

fn graph_white_replacement_scene_for_test() -> Scene {
    let mut replacement = Scene::new();
    replacement.fill(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    );
    replacement
}

fn explicit_graph_transaction_inputs_for_test(
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

fn graph_replacement_parameters_for_test() -> Parameters {
    Parameters {
        base_color: Color::TRANSPARENT,
        debug: true,
    }
}

struct GraphAbortFixtureForTest {
    renderer: Renderer,
    surface: Surface,
    replacement: Scene,
    replacement_parameters: Parameters,
    working_format: WorkingFormat,
    baseline_pixels: ImageBuffer,
    baseline_stats: Stats,
    baseline_parameters: Option<Parameters>,
    baseline_uploaded_images: std::collections::HashSet<ImageId>,
    baseline_publication_count: usize,
    baseline_cache: super::shader::DevicePassCacheCountsForTest,
    resources_before: super::resource::ResourceManagerObservationForTest,
}

fn graph_abort_fixture_for_test(
    renderer_expectation: &'static str,
    surface_expectation: &'static str,
    baseline_render_expectation: &'static str,
    baseline_read_expectation: &'static str,
    ready_device_expectation: &'static str,
    resource_manager_expectation: &'static str,
) -> GraphAbortFixtureForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect(renderer_expectation);
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0))
        .expect(surface_expectation);
    let mut baseline_scene = Scene::new();
    baseline_scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &baseline_scene, Parameters::default()))
        .expect(baseline_render_expectation);
    let baseline_pixels =
        pollster::block_on(renderer.read_headless(&surface)).expect(baseline_read_expectation);
    let baseline_stats = renderer.stats();
    let baseline_parameters = surface.last_parameters;
    let baseline_uploaded_images = renderer.uploaded_images_for_test();
    let baseline_publication_count = surface.headless_publication_count_for_test();
    let baseline_cache = renderer
        .default_ready_device_state_borrow_for_test()
        .expect(ready_device_expectation)
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect(resource_manager_expectation)
        .internal_resource_manager_observation_for_test();
    GraphAbortFixtureForTest {
        renderer,
        surface,
        replacement: graph_white_replacement_scene_for_test(),
        replacement_parameters: graph_replacement_parameters_for_test(),
        working_format,
        baseline_pixels,
        baseline_stats,
        baseline_parameters,
        baseline_uploaded_images,
        baseline_publication_count,
        baseline_cache,
        resources_before,
    }
}

#[test]
fn post_submit_scope_failure_discards_prepared_resources_with_nonzero_budget() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("post-submit graph abort coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let resources_before = resources.observation_for_test();
    let mut publication = Some(1);
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(903),
        &mut publication,
    ))
    .expect_err("the explicit post-submit scope failure must abort its resource frame");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    let resources_after = resources.observation_for_test();
    assert_eq!(resources_after.active_frame_count, 0);
    assert_eq!(resources_after.leased_count, 0);
    assert_eq!(resources_after.entry_count, 0);
    assert!(
        resources_after.lifecycle_stats_for_test().evictions
            > resources_before.lifecycle_stats_for_test().evictions
    );
    assert_eq!(resources_after.accounted_entry_bytes, Some(0));
}

#[test]
fn canceled_graph_after_real_submit_discards_prepared_resources_and_retries_fresh() {
    let GraphAbortFixtureForTest {
        mut renderer,
        mut surface,
        replacement,
        replacement_parameters,
        working_format,
        baseline_pixels,
        baseline_stats,
        baseline_parameters,
        baseline_uploaded_images,
        baseline_publication_count,
        baseline_cache,
        resources_before,
    } = graph_abort_fixture_for_test(
        "submitted graph cancellation coverage requires a renderer",
        "submitted graph cancellation coverage requires a headless surface",
        "the direct baseline frame must publish before cancellation coverage",
        "the cancellation baseline publication must be readable",
        "the cancellation baseline must retain a ready device",
        "the cancellation baseline must retain one resource manager",
    );
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let canceled_resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let canceled_resources_before = canceled_resources.observation_for_test();
    let mut canceled_publication = Some(1);
    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &canceled_resources,
            modeled_resource_key_for_test(904),
            &mut canceled_publication,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }
    assert_eq!(canceled_publication, Some(1));
    let canceled_resources_after = canceled_resources.observation_for_test();
    assert_eq!(canceled_resources_after.active_frame_count, 0);
    assert_eq!(canceled_resources_after.leased_count, 0);
    assert_eq!(canceled_resources_after.entry_count, 0);
    assert!(
        canceled_resources_after
            .lifecycle_stats_for_test()
            .evictions
            > canceled_resources_before
                .lifecycle_stats_for_test()
                .evictions
    );
    assert_eq!(renderer.stats(), baseline_stats);
    assert_eq!(surface.last_parameters, baseline_parameters);
    assert_eq!(
        renderer.uploaded_images_for_test(),
        baseline_uploaded_images
    );
    assert_eq!(
        surface.headless_publication_count_for_test(),
        baseline_publication_count
    );
    assert_eq!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the canceled frame must retain the ready device")
            .device_pass_cache_counts_for_test(),
        baseline_cache
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    let resources_after_abort = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the canceled frame must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after_abort.active_frame_count, 0);
    assert_eq!(resources_after_abort.leased_count, 0);
    assert_eq!(resources_after_abort.resolved_lease_count, 0);
    assert_eq!(
        resources_after_abort.retained_bytes,
        resources_after_abort
            .accounted_entry_bytes
            .expect("canceled frame resource accounting must have an exact total")
    );
    assert_eq!(resources_after_abort, resources_before);
    assert_eq!(
        pollster::block_on(renderer.read_headless(&surface))
            .expect("the canceled graph must preserve the baseline publication")
            .rgba(),
        baseline_pixels.rgba()
    );

    assert_graph_retry_after_abort(
        GraphRetryContextForTest {
            renderer: &mut renderer,
            surface: &mut surface,
            replacement: &replacement,
            replacement_parameters,
            working_format,
            baseline_cache,
            baseline_publication_count,
            baseline_pixels: &baseline_pixels,
        },
        GraphRetryExpectationsForTest {
            success: "a clean graph retry must succeed after submitted cancellation",
            readable: "the clean retry publication must be readable",
        },
    );
}

struct GraphRetryContextForTest<'a> {
    renderer: &'a mut Renderer,
    surface: &'a mut Surface,
    replacement: &'a Scene,
    replacement_parameters: Parameters,
    working_format: WorkingFormat,
    baseline_cache: super::shader::DevicePassCacheCountsForTest,
    baseline_publication_count: usize,
    baseline_pixels: &'a ImageBuffer,
}

struct GraphRetryExpectationsForTest {
    success: &'static str,
    readable: &'static str,
}

fn assert_graph_retry_after_abort(
    context: GraphRetryContextForTest<'_>,
    expectations: GraphRetryExpectationsForTest,
) {
    let retry = pollster::block_on(context.renderer.render_forced_base_graph_for_test(
        context.surface,
        context.replacement,
        context.replacement_parameters,
        context.working_format,
    ))
    .expect(expectations.success);
    let resources_after_retry = context
        .renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the clean retry must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after_retry.active_frame_count, 0);
    assert_eq!(resources_after_retry.resolved_lease_count, 0);
    assert_eq!(resources_after_retry.leased_count, 0);
    assert_eq!(
        resources_after_retry.retained_bytes,
        resources_after_retry
            .accounted_entry_bytes
            .expect("clean retry resource accounting must have an exact total")
    );
    assert!(resources_after_retry.entry_count > 0);
    assert_ne!(
        context
            .renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the clean retry must retain its committed pass cache")
            .device_pass_cache_counts_for_test(),
        context.baseline_cache
    );
    assert_eq!(context.renderer.stats(), retry.stats);
    assert_eq!(
        context.surface.last_parameters,
        Some(context.replacement_parameters)
    );
    assert_eq!(
        context.surface.headless_publication_count_for_test(),
        context.baseline_publication_count + 1
    );
    assert_ne!(
        pollster::block_on(context.renderer.read_headless(context.surface))
            .expect(expectations.readable)
            .rgba(),
        context.baseline_pixels.rgba()
    );
}

#[test]
fn terminal_signal_after_successful_headless_publication_preserves_frame_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255]))
        .expect("the baseline image must be valid");
    let mut first = Scene::new();
    first.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    let first_parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &first, first_parameters))
        .expect("the first frame must establish the public state to preserve");
    let _prior_pixels = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first frame must establish readable pixels");
    let prior_texture = match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the readable headless frame must retain its published texture"),
    };
    let prior_parameters = surface.last_parameters;
    let prior_uploaded_images = renderer.uploaded_images_for_test();
    let prior_publication_count = surface.headless_publication_count_for_test();

    let replacement =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255]))
            .expect("the replacement image must be valid");
    let mut next = Scene::new();
    next.image(
        replacement,
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Stretch,
    );
    let next_parameters = Parameters {
        base_color: Color::TRANSPARENT,
        debug: false,
    };
    let current = pollster::block_on(renderer.render(&mut surface, &next, next_parameters))
        .unwrap_or_else(|error| panic!("the replacement frame must publish: {error}"));
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    let current_texture = match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the completed frame must install its headless publication"),
    };
    assert_ne!(
        current_texture, prior_texture,
        "the completed frame must replace the prior published texture"
    );
    assert_eq!(renderer.stats(), current);
    assert_eq!(surface.last_parameters, Some(next_parameters));
    assert_ne!(surface.last_parameters, prior_parameters);
    assert_ne!(renderer.uploaded_images_for_test(), prior_uploaded_images);
    assert_eq!(
        surface.headless_publication_count_for_test(),
        prior_publication_count + 1
    );

    let committed_stats = renderer.stats();
    let committed_parameters = surface.last_parameters;
    let committed_uploaded_images = renderer.uploaded_images_for_test();
    let committed_publication_count = surface.headless_publication_count_for_test();
    let error = pollster::block_on(renderer.render(&mut surface, &next, Parameters::default()))
        .expect_err("the operation after an idle terminal signal must fail deterministically");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(renderer.stats(), committed_stats);
    assert_eq!(surface.last_parameters, committed_parameters);
    assert_eq!(
        renderer.uploaded_images_for_test(),
        committed_uploaded_images
    );
    assert_eq!(
        surface.headless_publication_count_for_test(),
        committed_publication_count
    );
    match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => assert_eq!(texture, &current_texture),
        _ => panic!("the rejected next operation must preserve the completed publication"),
    }
}

#[test]
fn failed_frame_returns_all_leases_and_preserves_last_successful_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut first = Scene::new();
    first.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    let last_successful =
        pollster::block_on(renderer.render(&mut surface, &first, Parameters::default()))
            .expect("the first frame must commit stats before failure coverage");
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the successful frame must retain a ready device")
        .internal_resource_manager_observation_for_test();

    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    let mut publication = Some(1);
    let error = pollster::block_on(renderer.fail_prepared_vello_pass_after_submit_for_test(
        &prepared,
        target_extent,
        &mut publication,
    ))
    .expect_err("the explicit post-submit failure must abort the frame draft");

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    assert_eq!(renderer.stats(), last_successful);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "the failed frame must return its transaction lease"
    );
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("a scoped frame failure must not terminally lose the device")
        .internal_resource_manager_observation_for_test();
    assert_eq!(
        resources_after.retained_count_for_test(),
        resources_before.retained_count_for_test(),
        "the failed frame must not retain an additional internal resource lease"
    );
    assert_eq!(
        resources_after.retained_atlas_byte_len_for_test(),
        resources_before.retained_atlas_byte_len_for_test(),
        "the failed frame must preserve the prior committed resource allocation"
    );
}

#[test]
fn headless_render_can_be_read_back() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let image = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    assert_eq!(image.size(), PhysicalSize::new(4, 4));
    assert_eq!(image.rgba().len(), 4 * 4 * 4);
    assert!(image.rgba().iter().any(|channel| *channel != 0));
}

#[derive(Clone, Copy, Debug)]
struct VelloPixelCharacterizationCase {
    antialiasing: Antialiasing,
    scale: f64,
    logical_dimensions: [u32; 2],
    physical_origin: [u32; 2],
    physical_dimensions: [u32; 2],
    solid_fill: [u8; 4],
    stroke: [u8; 4],
    gradient_left: [u8; 4],
    gradient_right: [u8; 4],
    image_top_left: [u8; 4],
    image_top_right: [u8; 4],
    clip_inside: [u8; 4],
    clip_excluded: [u8; 4],
    transformed_inside: [u8; 4],
    transformed_excluded: [u8; 4],
    ahem_ascent_ink: [u8; 4],
    ahem_descent_ink: [u8; 4],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
}

#[derive(Clone, Copy, Debug)]
struct AlphaSupport {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    centroid_x_hundredths: i32,
    centroid_y_hundredths: i32,
}

#[derive(Clone, Copy, Debug)]
struct VelloPixelVariation {
    physical_dimensions: [u32; 2],
    stroke_alpha: u8,
    gradient_left: [u8; 2],
    gradient_right: [u8; 2],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
}

// Each row is `{AA, scale, physical dimensions, stroke alpha, gradient left/right,
// solid edge support, stroke edge support}`. Other samples are stable across all rows.
const VELLO_PIXEL_CHARACTERIZATION_CASES: &[VelloPixelCharacterizationCase] = &[
    vello_pixel_case(
        Antialiasing::Area,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Area,
        1.25,
        variation(
            [90, 60],
            175,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 731, 731),
            edge(15, 0, 30, 15, 2293, 793),
            edge(67, 21, 77, 29, 7199, 2511),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Area,
        2.0,
        variation(
            [144, 96],
            127,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3699, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa8,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa8,
        1.25,
        variation(
            [90, 60],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 737, 730),
            edge(16, 1, 30, 15, 2299, 796),
            edge(67, 21, 77, 29, 7200, 2511),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa8,
        2.0,
        variation(
            [144, 96],
            128,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3700, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa16,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa16,
        1.25,
        variation(
            [90, 60],
            175,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 731, 731),
            edge(15, 0, 30, 15, 2293, 794),
            edge(67, 21, 77, 29, 7200, 2511),
        ),
    ),
    vello_pixel_case(
        Antialiasing::Msaa16,
        2.0,
        variation(
            [144, 96],
            128,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3700, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
];

const fn vello_pixel_case(
    antialiasing: Antialiasing,
    scale: f64,
    variation: VelloPixelVariation,
) -> VelloPixelCharacterizationCase {
    VelloPixelCharacterizationCase {
        antialiasing,
        scale,
        logical_dimensions: [72, 48],
        physical_origin: [0, 0],
        physical_dimensions: variation.physical_dimensions,
        solid_fill: [203, 52, 26, 128],
        stroke: [26, 64, 230, variation.stroke_alpha],
        gradient_left: [
            variation.gradient_left[0],
            0,
            variation.gradient_left[1],
            255,
        ],
        gradient_right: [
            variation.gradient_right[0],
            0,
            variation.gradient_right[1],
            255,
        ],
        image_top_left: [255, 0, 0, 255],
        image_top_right: [0, 255, 0, 255],
        clip_inside: [255, 255, 0, 255],
        clip_excluded: [0, 0, 0, 0],
        transformed_inside: [0, 255, 255, 255],
        transformed_excluded: [0, 0, 0, 0],
        ahem_ascent_ink: [0, 0, 0, 255],
        ahem_descent_ink: [0, 0, 0, 255],
        solid_edge: variation.solid_edge,
        stroke_edge: variation.stroke_edge,
        transformed_placement: variation.transformed_placement,
    }
}

const fn variation(
    physical_dimensions: [u32; 2],
    stroke_alpha: u8,
    gradient_left: [u8; 2],
    gradient_right: [u8; 2],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
) -> VelloPixelVariation {
    VelloPixelVariation {
        physical_dimensions,
        stroke_alpha,
        gradient_left,
        gradient_right,
        solid_edge,
        stroke_edge,
        transformed_placement,
    }
}

const fn edge(
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    centroid_x_hundredths: i32,
    centroid_y_hundredths: i32,
) -> AlphaSupport {
    AlphaSupport {
        min_x,
        min_y,
        max_x,
        max_y,
        centroid_x_hundredths,
        centroid_y_hundredths,
    }
}

#[test]
fn direct_vello_pixels_match_characterization_cases() {
    let configurations = [
        (Antialiasing::Area, 1.0),
        (Antialiasing::Area, 1.25),
        (Antialiasing::Area, 2.0),
        (Antialiasing::Msaa8, 1.0),
        (Antialiasing::Msaa8, 1.25),
        (Antialiasing::Msaa8, 2.0),
        (Antialiasing::Msaa16, 1.0),
        (Antialiasing::Msaa16, 1.25),
        (Antialiasing::Msaa16, 2.0),
    ];
    let mut observed = Vec::with_capacity(configurations.len());

    for (antialiasing, scale) in configurations {
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default().with_antialiasing(antialiasing),
        ))
        .expect("Vello pixel characterization requires a host adapter");
        let scene = vello_pixel_characterization_scene();
        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(72.0, 48.0), scale))
                .expect("Vello pixel characterization requires a real headless surface");
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("pixel characterization must render through the production Vello route");
        let output = pollster::block_on(renderer.read_headless(&surface))
            .expect("pixel characterization must read the rendered headless surface");
        observed.push(observe_vello_pixel_characterization(
            antialiasing,
            &surface,
            &output,
        ));
    }

    assert_eq!(
        observed.len(),
        VELLO_PIXEL_CHARACTERIZATION_CASES.len(),
        "missing Vello pixel characterization samples; observed rows: {observed:#?}"
    );
    assert_eq!(
        VELLO_PIXEL_CHARACTERIZATION_CASES.len(),
        configurations.len(),
        "the pixel table must cover every AA/scale Cartesian pair"
    );
    for (actual, expected) in observed.iter().zip(VELLO_PIXEL_CHARACTERIZATION_CASES) {
        assert_vello_pixel_characterization_case(*actual, *expected);
    }
}

#[test]
fn direct_render_submits_one_transaction_owned_raster_pass() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("production submission coverage requires a renderer");
    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    let submission =
        pollster::block_on(renderer.submit_prepared_vello_pass_for_test(&prepared, target_extent))
            .expect("the explicit real transaction must submit its internal raster payload");
    assert_eq!(
        submission.queue_submission_count_for_test(),
        1,
        "Renderer::render must use exactly one transaction-owned internal raster submission"
    );
    assert_eq!(
        submission.transaction_generation_for_test(),
        submission.active_generation_for_test(),
        "the observed submission must remain inside its transaction lease"
    );
    assert_eq!(
        submission.payload_raster_pass_count_for_test(),
        1,
        "the submitted transaction payload must contain the direct raster pass"
    );
    assert!(
        submission.allocation_summary_for_test().is_some(),
        "the observed submission must carry the internal raster resource lease"
    );
}

fn graph_alpha_extreme_pixels_for_test() -> Vec<[u8; 4]> {
    let mut pixels = vec![[128, 0, 0, 1]];
    for alpha in [0, 1, 2, 15, 16, 127, 254, 255] {
        for rgb in [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ] {
            pixels.push([rgb[0], rgb[1], rgb[2], alpha]);
        }
    }
    pixels
}

fn graph_alpha_extreme_scene_for_test(pixels: &[[u8; 4]]) -> Scene {
    let bytes = pixels
        .iter()
        .flat_map(|pixel| pixel.iter().copied())
        .collect::<Vec<_>>();
    let width = u32::try_from(pixels.len()).expect("the graph pixel vector must fit in u32");
    let image = Image::from_rgba(Size::new(f64::from(width), 1.0), Arc::<[u8]>::from(bytes))
        .expect("the alpha vector must form one valid image");
    let mut scene = Scene::new();
    scene.image(
        image,
        Rect::new(0.0, 0.0, f64::from(width), 1.0),
        ImageFit::Stretch,
    );
    scene
}

fn graph_canonical_pixel_for_test(pixel: [u8; 4]) -> [u8; 4] {
    if pixel[3] == 0 { [0, 0, 0, 0] } else { pixel }
}

fn premultiply_u8_channel_for_test(color: u8, alpha: u8) -> u8 {
    ((u16::from(color) * u16::from(alpha) + 127) / 255) as u8
}

fn graph_channel_error_for_test(actual: u8, expected: u8) -> u8 {
    actual.abs_diff(expected)
}

struct GraphAlphaVectorOutputForTest {
    output: ImageBuffer,
    graph: super::renderer::ForcedGraphRenderResultForTest,
    used_graph_transaction: bool,
    publication_count: usize,
}

fn default_graph_working_format_for_test(renderer: &mut Renderer) -> WorkingFormat {
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

fn render_graph_alpha_vector_for_test(
    expected: &[[u8; 4]],
    requested_working_format: Option<WorkingFormat>,
) -> GraphAlphaVectorOutputForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("alpha-vector graph execution requires a renderer");
    let working_format = requested_working_format
        .unwrap_or_else(|| default_graph_working_format_for_test(&mut renderer));
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .expect("alpha-vector graph execution requires a headless surface");
    let publication_before = surface.headless_publication_count_for_test();
    let scene = graph_alpha_extreme_scene_for_test(expected);
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the forced graph entry must execute the production headless graph");
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the already-published graph frame must be explicitly readable");
    let used_graph_transaction = graph.working_format == working_format
        && graph.stats.route == Some(RenderRoute::GpuGraph)
        && publication_count == 1;
    GraphAlphaVectorOutputForTest {
        output,
        graph,
        used_graph_transaction,
        publication_count,
    }
}

fn graph_alpha_vector_has_exact_grid_for_test(
    graph: &super::renderer::ForcedGraphRenderResultForTest,
    width: u32,
) -> bool {
    graph.output_extent == PhysicalSize::new(width, 1)
        && matches!(
            graph.captures.as_slice(),
            [capture]
                if capture.texel_origin == Point::new(0.0, 0.0)
                    && capture.extent == PhysicalSize::new(width, 1)
                    && capture.raster_scale == 1.0
        )
}

#[test]
fn capture_canonicalize_present_round_trips_transparent_partial_and_opaque_pixels() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered = render_graph_alpha_vector_for_test(&expected, None);
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let canonical_pixels =
        rendered
            .output
            .rgba()
            .chunks_exact(4)
            .zip(&expected)
            .all(|(actual, expected)| {
                let expected = graph_canonical_pixel_for_test(*expected);
                if rendered.graph.working_format == WorkingFormat::HighPrecision {
                    actual
                        .iter()
                        .copied()
                        .zip(expected)
                        .all(|(actual, expected)| {
                            graph_channel_error_for_test(actual, expected) <= 2
                        })
                } else {
                    graph_channel_error_for_test(actual[3], expected[3]) <= 1
                        && (0..3).all(|channel| {
                            graph_channel_error_for_test(
                                premultiply_u8_channel_for_test(actual[channel], actual[3]),
                                premultiply_u8_channel_for_test(expected[channel], expected[3]),
                            ) <= 1
                        })
                }
            });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && exact_extent_and_origin
            && canonical_pixels,
        "headless graph pixels do not satisfy canonical output"
    );
}

#[test]
fn reduced_precision_low_alpha_pixels_use_alpha_and_premul8_tolerances() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered =
        render_graph_alpha_vector_for_test(&expected, Some(WorkingFormat::ReducedPrecision));
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let reduced_pixels =
        rendered
            .output
            .rgba()
            .chunks_exact(4)
            .zip(&expected)
            .all(|(actual, expected)| {
                let expected = graph_canonical_pixel_for_test(*expected);
                graph_channel_error_for_test(actual[3], expected[3]) <= 1
                    && (0..3).all(|channel| {
                        graph_channel_error_for_test(
                            premultiply_u8_channel_for_test(actual[channel], actual[3]),
                            premultiply_u8_channel_for_test(expected[channel], expected[3]),
                        ) <= 1
                    })
            });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && rendered.graph.working_format == WorkingFormat::ReducedPrecision
            && exact_extent_and_origin
            && reduced_pixels,
        "reduced-precision graph output violates alpha or premul8 tolerance"
    );
}

#[test]
fn high_precision_low_alpha_pixels_preserve_straight_rgb() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered =
        render_graph_alpha_vector_for_test(&expected, Some(WorkingFormat::HighPrecision));
    let width = u32::try_from(expected.len())
        .unwrap_or_panic_for_test("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let stable_straight_rgb = rendered
        .output
        .rgba()
        .chunks_exact(4)
        .zip(&expected)
        .filter(|(_, expected)| expected[3] > 0 && expected[3] <= 16)
        .all(|(actual, expected)| {
            graph_channel_error_for_test(actual[3], expected[3]) <= 2
                && (0..3).all(|channel| {
                    graph_channel_error_for_test(actual[channel], expected[channel]) <= 2
                })
        });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && rendered.graph.working_format == WorkingFormat::HighPrecision
            && exact_extent_and_origin
            && stable_straight_rgb,
        "high-precision graph output lost stable straight RGB"
    );
}

const COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X: i32 = -2;

struct ColorFilterProductionFrameForTest {
    output: ImageBuffer,
    stats: Stats,
    working_format: WorkingFormat,
    output_extent: PhysicalSize,
    source_origin: Option<(i32, i32)>,
    source_extent: Option<PhysicalSize>,
    source_texel_origin: Option<Point>,
    source_raster_scale: Option<f64>,
    publication_count: usize,
}

fn color_filter_boundary_pixels_for_test() -> Vec<[u8; 4]> {
    graph_alpha_extreme_pixels_for_test()
}

fn color_filter_signed_source_scene_for_test(visible_pixels: &[[u8; 4]]) -> Scene {
    let hidden_prefix = [[17, 31, 47, 255], [233, 199, 151, 127]];
    let bytes = hidden_prefix
        .into_iter()
        .chain(visible_pixels.iter().copied())
        .flat_map(|pixel| pixel.into_iter())
        .collect::<Vec<_>>();
    let source_width = u32::try_from(visible_pixels.len() + hidden_prefix.len())
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
    scene
}

fn color_filter_operation_matrix_for_test() -> [(&'static str, ColorFilterOp); 8] {
    [
        (
            "brightness",
            ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
        ),
        (
            "contrast",
            ColorFilterOp::Contrast(FilterAmount::try_new(1.75).unwrap()),
        ),
        (
            "grayscale",
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(1.0).unwrap()),
        ),
        (
            "hue-rotate",
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
        ),
        (
            "invert",
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.75).unwrap()),
        ),
        (
            "opacity",
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ),
        (
            "saturate",
            ColorFilterOp::Saturate(FilterAmount::try_new(2.0).unwrap()),
        ),
        (
            "sepia",
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(1.0).unwrap()),
        ),
    ]
}

fn color_filter_reference_straight_pixels_for_test(
    source_pixels: &[[u8; 4]],
    operations: &[ColorFilterOp],
    working_format: WorkingFormat,
) -> Vec<u8> {
    let size = PhysicalSize::new(
        u32::try_from(source_pixels.len()).expect("the color-filter oracle width must fit u32"),
        1,
    );
    let premultiplied_source = ReferencePremultipliedRgba8Buffer::from_pixels(
        size,
        source_pixels
            .iter()
            .copied()
            .map(reference_premultiplied_pixel_for_test)
            .collect(),
    )
    .expect("the color-filter source pixels must form one premultiplied oracle buffer");
    let filter = FilterList::try_ops(
        operations
            .iter()
            .copied()
            .map(|operation| match operation {
                ColorFilterOp::Brightness(amount) => FilterOp::brightness(amount),
                ColorFilterOp::Contrast(amount) => FilterOp::contrast(amount),
                ColorFilterOp::Grayscale(amount) => FilterOp::grayscale(amount),
                ColorFilterOp::HueRotate(angle) => FilterOp::hue_rotate(angle),
                ColorFilterOp::Invert(amount) => FilterOp::invert(amount),
                ColorFilterOp::Opacity(amount) => FilterOp::opacity(amount),
                ColorFilterOp::Saturate(amount) => FilterOp::saturate(amount),
                ColorFilterOp::Sepia(amount) => FilterOp::sepia(amount),
            })
            .collect(),
    )
    .expect("the color-filter oracle operations must form one authored filter");
    let pipeline = filter
        .color_filter_pipeline()
        .expect("the color-filter oracle fixture contains only color functions")
        .expect("the color-filter oracle fixture contains one nonempty color run");
    match working_format {
        WorkingFormat::HighPrecision => {
            super::reference::apply_color_filter_pipeline_to_straight_rgba8(
                source_pixels,
                &pipeline,
            )
        }
        WorkingFormat::ReducedPrecision => {
            let filtered = premultiplied_source
                .apply_color_filter_pipeline(&pipeline)
                .expect("the color-filter CPU oracle must evaluate every authored operation");
            reference_straight_bytes_for_test(&filtered)
        }
    }
}

fn render_color_filter_fixture_for_test(
    renderer: &mut Renderer,
    surface: &mut Surface,
    scene: &Scene,
    filters: Vec<FilterList>,
    parameters: Parameters,
    working_format: WorkingFormat,
) -> ColorFilterProductionFrameForTest {
    let publication_before = surface.headless_publication_count_for_test();
    let graph = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        surface,
        scene,
        filters,
        parameters,
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!("the color-filter fixture must execute through the shared exact graph: {error}")
    });
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let output = pollster::block_on(renderer.read_headless(surface)).unwrap_or_else(|error| {
        panic!(
            "the already-published color-filter RED fixture must be explicitly readable: {error}"
        )
    });
    ColorFilterProductionFrameForTest {
        output,
        stats: graph.stats,
        working_format: graph.working_format,
        output_extent: graph.output_extent,
        source_origin: Some(graph.source_origin),
        source_extent: Some(graph.source_extent),
        source_texel_origin: Some(graph.source_texel_origin),
        source_raster_scale: Some(graph.source_raster_scale),
        publication_count,
    }
}

fn color_filter_frame_has_exact_extent_origin_and_submission_for_test(
    rendered: &ColorFilterProductionFrameForTest,
    visible_width: u32,
) -> bool {
    rendered.output.size() == PhysicalSize::new(visible_width, 1)
        && rendered.output_extent == PhysicalSize::new(visible_width, 1)
        && rendered.source_origin == Some((COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X, 0))
        && rendered.source_extent
            == Some(PhysicalSize::new(
                visible_width + COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X.unsigned_abs(),
                1,
            ))
        && rendered.source_texel_origin
            == Some(Point::new(
                f64::from(COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X),
                0.0,
            ))
        && rendered.source_raster_scale == Some(1.0)
        && rendered.stats.route == Some(RenderRoute::GpuGraph)
        && rendered.publication_count == 1
        && rendered.stats.commands > 0
}

fn terminal_straight_rgba8_is_canonical_for_test(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(4)
        && bytes
            .chunks_exact(4)
            .all(|pixel| pixel[3] != 0 || pixel[..3] == [0, 0, 0])
}

#[test]
fn terminal_straight_rgba8_rejects_rgb_leakage_at_zero_alpha() {
    let accepted = [
        0, 0, 0, 0, 255, 0, 128, 1, 17, 31, 47, 127, 255, 255, 255, 255,
    ];

    assert!(
        terminal_straight_rgba8_is_canonical_for_test(&accepted),
        "terminal straight RGBA8 rejected canonical transparent black or a nonzero-alpha pixel"
    );
    assert!(
        !terminal_straight_rgba8_is_canonical_for_test(&[0, 0, 0]),
        "terminal straight RGBA8 accepted a byte-misaligned sample"
    );
    assert!(
        !terminal_straight_rgba8_is_canonical_for_test(&[128, 0, 0, 0]),
        "terminal straight RGBA8 accepted RGB leakage at zero alpha"
    );
}

fn high_precision_terminal_error_for_test(actual: &[u8], expected: &[u8]) -> Option<u8> {
    (actual.len() == expected.len() && actual.len().is_multiple_of(4)).then(|| {
        actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
            0,
            |maximum, (actual, expected)| {
                // The caller first proves canonical terminal bytes. Once target
                // alpha quantizes to zero, straight RGB has only the black form.
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
    })
}

fn reduced_precision_terminal_error_for_test(actual: &[u8], expected: &[u8]) -> Option<(u8, u8)> {
    (actual.len() == expected.len()).then(|| {
        actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
            (0, 0),
            |(max_alpha, max_premul), (actual, expected)| {
                let alpha = max_alpha.max(actual[3].abs_diff(expected[3]));
                let premul = (0..3).fold(max_premul, |maximum, channel| {
                    maximum.max(
                        premultiply_u8_channel_for_test(actual[channel], actual[3]).abs_diff(
                            premultiply_u8_channel_for_test(expected[channel], expected[3]),
                        ),
                    )
                });
                (alpha, premul)
            },
        )
    })
}

fn color_filter_pixel_renderer_for_test(
    working_format: WorkingFormat,
    width: u32,
) -> (Renderer, Surface) {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("color-filter pixel execution requires a real renderer: {error}")
    });
    let supported = graph_supported_working_formats_for_test(&mut renderer);
    assert!(
        supported.contains(&working_format),
        "color-filter pixel execution requires the requested real working format"
    );
    let adapter = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("color-filter pixel execution requires one ready real adapter")
        .adapter_for_test()
        .get_info();
    eprintln!(
        "color-filter real adapter name={} backend={:?} device_type={:?} driver={} driver_info={}",
        adapter.name, adapter.backend, adapter.device_type, adapter.driver, adapter.driver_info
    );
    let surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap_or_else(|error| {
                panic!("color-filter pixel execution requires a headless surface: {error}")
            });
    (renderer, surface)
}

#[test]
fn high_precision_color_functions_match_cpu_oracle_for_boundary_pixels() {
    let source = color_filter_boundary_pixels_for_test();
    let width =
        u32::try_from(source.len()).expect("the high-precision color matrix width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let (mut renderer, mut surface) =
        color_filter_pixel_renderer_for_test(WorkingFormat::HighPrecision, width);
    let mut maximum_error = 0;
    let mut exact_execution = true;

    for (name, operation) in color_filter_operation_matrix_for_test() {
        let expected = color_filter_reference_straight_pixels_for_test(
            &source,
            &[operation],
            WorkingFormat::HighPrecision,
        );
        let rendered = render_color_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            vec![color_filter_list([operation])],
            Parameters::default(),
            WorkingFormat::HighPrecision,
        );
        let exact_grid =
            color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width);
        let terminal_canonical =
            terminal_straight_rgba8_is_canonical_for_test(rendered.output.rgba());
        exact_execution &= rendered.working_format == WorkingFormat::HighPrecision
            && exact_grid
            && terminal_canonical;
        let error = high_precision_terminal_error_for_test(rendered.output.rgba(), &expected)
            .unwrap_or(u8::MAX);
        maximum_error = maximum_error.max(error);
        eprintln!(
            "high-precision color operation={name} max_terminal_straight_rgba8_error={error} exact_grid={exact_grid} terminal_canonical={terminal_canonical} output_extent={:?} source_origin={:?} source_extent={:?} source_texel_origin={:?} source_raster_scale={:?} publication_count={}",
            rendered.output_extent,
            rendered.source_origin,
            rendered.source_extent,
            rendered.source_texel_origin,
            rendered.source_raster_scale,
            rendered.publication_count,
        );
    }

    assert!(
        exact_execution && maximum_error <= 2,
        "high-precision color-filter pixels exceed the declared color tolerance"
    );
}

#[test]
fn reduced_precision_color_functions_match_cpu_oracle_with_declared_tolerance() {
    let source = color_filter_boundary_pixels_for_test();
    let width =
        u32::try_from(source.len()).expect("the reduced-precision color matrix width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let (mut renderer, mut surface) =
        color_filter_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, width);
    let mut maximum_alpha_error = 0;
    let mut maximum_premul_error = 0;
    let mut exact_execution = true;

    for (name, operation) in color_filter_operation_matrix_for_test() {
        let expected = color_filter_reference_straight_pixels_for_test(
            &source,
            &[operation],
            WorkingFormat::ReducedPrecision,
        );
        let rendered = render_color_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            vec![color_filter_list([operation])],
            Parameters::default(),
            WorkingFormat::ReducedPrecision,
        );
        let terminal_canonical =
            terminal_straight_rgba8_is_canonical_for_test(rendered.output.rgba());
        exact_execution &= rendered.working_format == WorkingFormat::ReducedPrecision
            && color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width)
            && terminal_canonical;
        let (alpha_error, premul_error) =
            reduced_precision_terminal_error_for_test(rendered.output.rgba(), &expected)
                .unwrap_or((u8::MAX, u8::MAX));
        maximum_alpha_error = maximum_alpha_error.max(alpha_error);
        maximum_premul_error = maximum_premul_error.max(premul_error);
        eprintln!(
            "reduced-precision color operation={name} max_alpha_error={alpha_error} max_premul8_error={premul_error} terminal_canonical={terminal_canonical}"
        );
    }

    assert!(
        exact_execution && maximum_alpha_error <= 2 && maximum_premul_error <= 2,
        "reduced-precision color-filter alpha or premul8 exceeds the declared tolerance"
    );
}

#[test]
fn filter_function_order_changes_output_and_matches_ordered_oracle() {
    let source = vec![
        [255, 37, 173, 0],
        [224, 72, 16, 127],
        [192, 64, 46, 255],
        [96, 32, 23, 127],
        [17, 231, 93, 255],
    ];
    let width = u32::try_from(source.len()).expect("the ordered color-filter width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let chain_pairs = [
        (
            "noncommuting contrast/brightness",
            [
                ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
            ],
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
                ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
            ],
        ),
        (
            "source-clamp-sensitive brightness chain",
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            ],
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
            ],
        ),
    ];
    let mut ordered_results_are_exact = true;

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) =
            color_filter_pixel_renderer_for_test(working_format, width);
        for (name, first, second) in chain_pairs {
            let first_expected =
                color_filter_reference_straight_pixels_for_test(&source, &first, working_format);
            let second_expected =
                color_filter_reference_straight_pixels_for_test(&source, &second, working_format);
            let first_rendered = render_color_filter_fixture_for_test(
                &mut renderer,
                &mut surface,
                &scene,
                vec![color_filter_list(first)],
                Parameters::default(),
                working_format,
            );
            let second_rendered = render_color_filter_fixture_for_test(
                &mut renderer,
                &mut surface,
                &scene,
                vec![color_filter_list(second)],
                Parameters::default(),
                working_format,
            );
            let first_terminal_canonical =
                terminal_straight_rgba8_is_canonical_for_test(first_rendered.output.rgba());
            let second_terminal_canonical =
                terminal_straight_rgba8_is_canonical_for_test(second_rendered.output.rgba());
            let exact_frames =
                color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                    &first_rendered,
                    width,
                ) && color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                    &second_rendered,
                    width,
                ) && first_terminal_canonical
                    && second_terminal_canonical;
            let first_matches =
                color_filter_rendered_output_matches_for_test(&first_rendered, &first_expected);
            let second_matches =
                color_filter_rendered_output_matches_for_test(&second_rendered, &second_expected);
            ordered_results_are_exact &= first_expected != second_expected
                && first_rendered.output.rgba() != second_rendered.output.rgba()
                && first_matches
                && second_matches
                && exact_frames;
            eprintln!(
                "ordered color-filter chain={name} working_format={working_format:?} first_terminal_canonical={first_terminal_canonical} second_terminal_canonical={second_terminal_canonical}"
            );
        }
    }

    assert!(
        ordered_results_are_exact,
        "the GPU lost authored order or a source clamp"
    );
}

fn color_filter_rendered_output_matches_for_test(
    rendered: &ColorFilterProductionFrameForTest,
    expected: &[u8],
) -> bool {
    match rendered.working_format {
        WorkingFormat::HighPrecision => {
            high_precision_terminal_error_for_test(rendered.output.rgba(), expected)
                .is_some_and(|error| error <= 2)
        }
        WorkingFormat::ReducedPrecision => {
            reduced_precision_terminal_error_for_test(rendered.output.rgba(), expected)
                .is_some_and(|(alpha, premul)| alpha <= 2 && premul <= 2)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ColorFilterShaderFailureObservationForTest {
    failure_is_reported: bool,
    prior_pixels_are_preserved: bool,
    prior_publication_is_preserved: bool,
    public_state_is_preserved: bool,
    pass_cache_is_preserved: bool,
    draft_resources_are_aborted: bool,
}

fn color_filter_shader_failure_observation_for_test() -> ColorFilterShaderFailureObservationForTest
{
    let source = vec![[17, 31, 47, 0], [224, 72, 16, 127], [192, 64, 46, 255]];
    let width = u32::try_from(source.len()).expect("the color-filter failure width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let (mut renderer, mut surface) =
        color_filter_pixel_renderer_for_test(WorkingFormat::HighPrecision, width);
    assert!(
        graph_supported_working_formats_for_test(&mut renderer)
            .contains(&WorkingFormat::ReducedPrecision),
        "color-filter shader-failure coverage requires the real reduced working format"
    );
    let filters = vec![color_filter_list([
        ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
    ])];
    let _baseline = render_color_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &scene,
        filters.clone(),
        Parameters::default(),
        WorkingFormat::HighPrecision,
    );
    let published = pollster::block_on(renderer.read_headless(&surface))
        .expect("the color-filter failure baseline must be readable");
    let stats_before = renderer.stats();
    let parameters_before = surface.last_parameters;
    let uploaded_images_before = renderer.uploaded_images_for_test();
    let publication_before = surface.headless_publication_count_for_test();
    let (cache_before, resources_before) = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the color-filter failure baseline must retain a ready device");
        (
            ready.device_pass_cache_counts_for_test(),
            ready.internal_resource_manager_observation_for_test(),
        )
    };

    let shader_failure =
        super::pass::ScopedColorFilterShaderFailureForTest::after_checked_realization();
    let failure = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters,
        Parameters {
            base_color: Color::TRANSPARENT,
            debug: true,
        },
        WorkingFormat::ReducedPrecision,
    ));
    drop(shader_failure);

    let current = pollster::block_on(renderer.read_headless(&surface))
        .expect("the failed color-filter attempt must leave the prior publication readable");
    let (cache_after, resources_after) = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the failed color-filter attempt must retain its ready device");
        (
            ready.device_pass_cache_counts_for_test(),
            ready.internal_resource_manager_observation_for_test(),
        )
    };
    let failure_is_reported = is_injected_color_filter_shader_failure(failure);
    ColorFilterShaderFailureObservationForTest {
        failure_is_reported,
        prior_pixels_are_preserved: current.rgba() == published.rgba(),
        prior_publication_is_preserved: surface.headless_publication_count_for_test()
            == publication_before,
        public_state_is_preserved: renderer.stats() == stats_before
            && surface.last_parameters == parameters_before
            && renderer.uploaded_images_for_test() == uploaded_images_before,
        pass_cache_is_preserved: cache_after == cache_before,
        draft_resources_are_aborted: resources_after.leased_count == 0
            && resources_after.active_frame_count == 0
            && resources_after.resolved_lease_count == 0
            && resources_after.accounting_fault_for_test().is_none()
            && resources_after
                .entry_identities_for_test()
                .iter()
                .all(|identity| {
                    resources_before
                        .entry_identities_for_test()
                        .contains(identity)
                }),
    }
}

fn is_injected_color_filter_shader_failure(
    failure: Result<super::renderer::ColorFilterRenderResultForTest>,
) -> bool {
    failure.is_err_and(|error| {
        error.code() == ErrorCode::RenderFailed
            && error
                .message()
                .contains("injected color-filter shader failure")
    })
}

#[test]
fn color_filter_shader_failure_preserves_prior_publication_and_cache() {
    let observed = color_filter_shader_failure_observation_for_test();
    eprintln!("color-filter shader failure observation={observed:?}");

    assert!(
        observed.failure_is_reported
            && observed.prior_pixels_are_preserved
            && observed.prior_publication_is_preserved
            && observed.public_state_is_preserved
            && observed.pass_cache_is_preserved
            && observed.draft_resources_are_aborted,
        "failed color-filter execution published draft state"
    );
}

fn color_filter_retention_fixture_for_test() -> (Scene, Vec<FilterList>, Vec<u8>) {
    let source = vec![
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
    (
        color_filter_signed_source_scene_for_test(&source),
        vec![color_filter_list([ColorFilterOp::Invert(
            UnitFilterAmount::try_new(1.0).unwrap(),
        )])],
        expected,
    )
}

fn color_filter_repeated_resource_observations_are_stable_for_test(
    observations: &[super::resource::ResourceManagerObservationForTest],
    warmed: &super::resource::ResourceManagerObservationForTest,
) -> bool {
    observations.iter().all(|observation| {
        observation.leased_count == 0
            && observation.active_frame_count == 0
            && observation.resolved_lease_count == 0
            && observation.next_resource == warmed.next_resource
            && observation.entry_count == warmed.entry_count
            && observation.retained_bytes == warmed.retained_bytes
            && observation.payload_creation_attempts == warmed.payload_creation_attempts
            && observation.entry_identities_for_test() == warmed.entry_identities_for_test()
            && observation.committed_transient_buffer_count_for_test()
                == warmed.committed_transient_buffer_count_for_test()
            && observation.committed_transient_image_count_for_test()
                == warmed.committed_transient_image_count_for_test()
            && observation.effect_texture_count_for_test() == warmed.effect_texture_count_for_test()
    })
}

#[test]
fn repeated_color_filter_frames_reuse_passes_without_growth_or_readback() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width =
        u32::try_from(expected.len() / 4).expect("the color-filter retention width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| {
        panic!("repeated color-filter reuse coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap_or_else(|error| {
                panic!("repeated color-filter reuse coverage requires a headless surface: {error}")
            });

    for _ in 0..2 {
        pollster::block_on(renderer.render_color_filter_fixture_for_test(
            &mut surface,
            &scene,
            filters.clone(),
            Parameters::default(),
            working_format,
        ))
        .unwrap_or_else(|error| panic!("color-filter reuse warm-up frames must succeed: {error}"));
    }
    let warmed_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the warmed color-filter publication must be readable: {error}")
        });
    let warmed = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("the warmed color-filter device must remain ready"));
    let warmed_resources = warmed.internal_resource_manager_observation_for_test();
    let warmed_cache = warmed.device_pass_cache_counts_for_test();

    let mut resource_observations = Vec::new();
    let mut cache_observations = Vec::new();
    let mut stats = Vec::new();
    for _ in 0..3 {
        let frame = pollster::block_on(renderer.render_color_filter_fixture_for_test(
            &mut surface,
            &scene,
            filters.clone(),
            Parameters::default(),
            working_format,
        ))
        .unwrap_or_else(|error| panic!("repeated color-filter frames must succeed: {error}"));
        stats.push(GraphPublicStatsForTest::from(frame.stats));
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .unwrap_or_else(|| panic!("repeated color-filter frames must retain the ready device"));
        resource_observations.push(ready.internal_resource_manager_observation_for_test());
        cache_observations.push(ready.device_pass_cache_counts_for_test());
    }
    let stable_resource_set = color_filter_repeated_resource_observations_are_stable_for_test(
        &resource_observations,
        &warmed_resources,
    );
    let stable_cache = warmed_cache.has_render_pipelines()
        && cache_observations
            .iter()
            .all(|observation| *observation == warmed_cache);
    let stable_stats = stats
        .first()
        .is_some_and(|first| stats.iter().all(|actual| actual == first));
    let actual = pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
        panic!("the repeated color-filter publication must remain readable: {error}")
    });
    eprintln!("color-filter retained cache={warmed_cache:?} resources={warmed_resources:?}");

    assert!(
        stable_resource_set
            && warmed_resources.committed_transient_buffer_count_for_test() > 0
            && warmed_resources.committed_transient_image_count_for_test() > 0
            && warmed_resources.effect_texture_count_for_test() > 0
            && stable_cache
            && stable_stats
            && warmed_output.rgba() == actual.rgba()
            && actual.rgba() == expected,
        "repeated color-filter frames grew passes or resources or entered readback"
    );
}

#[test]
fn budget_zero_releases_color_filter_frame_resources_without_changing_pixels() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width =
        u32::try_from(expected.len() / 4).expect("the color-filter zero-budget width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .unwrap_or_else(|error| {
        panic!("zero-retention color-filter coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap_or_else(|error| {
                panic!("zero-retention color-filter coverage requires a headless surface: {error}")
            });

    let first = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters.clone(),
        Parameters::default(),
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!("the first zero-retention color-filter frame must succeed: {error}")
    });
    let first_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the first zero-retention color-filter publication must be readable: {error}")
        });
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| {
            panic!("the first zero-retention color-filter frame must retain its device")
        })
        .device_pass_cache_counts_for_test();

    let second = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters,
        Parameters::default(),
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!("the repeated zero-retention color-filter frame must succeed: {error}")
    });
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| {
            panic!("the repeated zero-retention color-filter device must remain ready")
        });
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let all_idle_resources_are_released = resources.leased_count == 0
        && resources.idle_count == 0
        && resources.active_frame_count == 0
        && resources.resolved_lease_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
        && resources.committed_transient_buffer_count_for_test() == 0
        && resources.committed_transient_image_count_for_test() == 0
        && resources.effect_texture_count_for_test() == 0;
    let second_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the repeated zero-retention color-filter publication must be readable: {error}")
        });
    eprintln!(
        "color-filter zero-budget cache_before={cache_before:?} cache_after={cache_after:?} resources={resources:?} first_stats={:?} second_stats={:?} first={:?} second={:?} expected={expected:?}",
        GraphPublicStatsForTest::from(first.stats),
        GraphPublicStatsForTest::from(second.stats),
        first_output.rgba(),
        second_output.rgba(),
    );

    assert!(
        all_idle_resources_are_released
            && cache_before == cache_after
            && cache_after.has_render_pipelines()
            && first.stats.commands == second.stats.commands
            && first.stats.images == second.stats.images
            && first_output.rgba() == second_output.rgba()
            && second_output.rgba() == expected,
        "zero retention changed color-filter pixels or kept idle frame resources"
    );
}

fn color_filter_public_color_graph_diagnostic_for_test(
    scene: &Scene,
    filters: Vec<FilterList>,
    size: Size,
) -> Option<UnsupportedPrimitive> {
    let commands = scene
        .normalize(Capabilities::CURRENT)
        .expect("the public color-filter diagnostic fixture must normalize ordinary capture input");
    let context =
        super::frame::FrameContext::try_new(size, 1.0, Antialiasing::Area, Color::TRANSPARENT)
            .expect("the public color-filter diagnostic fixture must form a frame context");
    let graph = super::frame::authored_filter_graph_for_test(filters, commands, context)
        .expect("the public diagnostic fixture must form the same authored color-filter graph");
    super::renderer::unsupported_graph_diagnostic_for_test(
        &graph,
        Format::Rgba8,
        &DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .expect("the retained public dispatch classifier must diagnose a color-filter graph")
}

fn retained_public_filter_diagnostics_are_exact_for_test() -> bool {
    let capabilities = Capabilities::CURRENT;
    let supported = [
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ),
    ];
    let unsupported = [
        (PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        (
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint,
        ),
        (
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::ColorFilteredImagePaint,
        ),
        (
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::LayerFilterExecution,
        ),
        (
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
    ];
    let supported_are_exact = supported.into_iter().all(|(family, operation)| {
        capabilities
            .ensure_supported(UnsupportedPrimitive::new(family, operation))
            .is_ok()
    });
    let unsupported_are_exact = unsupported.into_iter().all(|(family, operation)| {
        let expected = UnsupportedPrimitive::new(family, operation);
        capabilities
            .ensure_supported(expected)
            .is_err_and(|error| error.unsupported_primitive() == Some(expected))
    });
    let reference =
        UnresolvedResource::new(UnresolvedResourceKind::Filter, "#color_filter-reference");
    let reference_error = Error::unresolved_resource(reference.clone());
    supported_are_exact
        && unsupported_are_exact
        && capabilities.filters().supports_gpu_color_filter_execution()
        && capabilities.filters().supports_gpu_blur_filter_execution()
        && capabilities
            .filters()
            .supports_gpu_drop_shadow_filter_execution()
        && !capabilities.filters().supports_layer_filters()
        && !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
        && !capabilities
            .image_sampling()
            .supports_color_filtered_image_paint()
        && !capabilities
            .offscreen_pipeline()
            .supports_layer_filter_execution()
        && capabilities
            .offscreen_pipeline()
            .supports_bounded_backdrop_filter_execution()
        && reference_error.code() == ErrorCode::UnresolvedResource
        && reference_error.unresolved_resource_diagnostic() == Some(&reference)
}

fn color_filter_unsupported_backdrop_scene_for_test() -> Scene {
    let backdrop_filters = color_filter_list([ColorFilterOp::Invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )]);
    let backdrop = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                backdrop_filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 4.0, 4.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(1.0, 1.0, 2.0, 2.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        });
    scene
}

#[test]
fn color_filter_fixture_executes_while_public_capability_remains_diagnostic() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width =
        u32::try_from(expected.len() / 4).expect("the color-filter ingress width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| panic!("color-filter ingress coverage requires a renderer: {error}"));
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap_or_else(|error| {
                panic!("color-filter ingress coverage requires a surface: {error}")
            });
    let rendered = render_color_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &scene,
        filters.clone(),
        Parameters::default(),
        working_format,
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the executed color-filter fixture must retain its ready device");
    let resources_before_diagnostic = ready.internal_resource_manager_observation_for_test();
    let cache_before_diagnostic = ready.device_pass_cache_counts_for_test();
    let publication_before_diagnostic = surface.headless_publication_count_for_test();
    let public_diagnostic = color_filter_public_color_graph_diagnostic_for_test(
        &scene,
        filters,
        Size::new(f64::from(width), 1.0),
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the public color-filter rejection must retain its ready device");
    let resources_after_diagnostic = ready.internal_resource_manager_observation_for_test();
    let cache_after_diagnostic = ready.device_pass_cache_counts_for_test();
    let expected_diagnostic = UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::GpuColorFilterExecution,
    );

    assert!(
        color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width)
            && rendered.output.rgba() == expected
            && rendered.stats == renderer.stats()
            && public_diagnostic == Some(expected_diagnostic)
            && retained_public_filter_diagnostics_are_exact_for_test()
            && resources_after_diagnostic == resources_before_diagnostic
            && cache_after_diagnostic == cache_before_diagnostic
            && surface.headless_publication_count_for_test() == publication_before_diagnostic,
        "the color-filter fixture did not execute through retained graph ingress while the public capability remained diagnostic"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn render_window_smoke_executes_ordered_color_filter_fixture_through_production_graph() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width =
        u32::try_from(expected.len() / 4).expect("the presented fixture width must fit u32");
    let parameters = Parameters::default();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| panic!("presented color-filter coverage requires a renderer: {error}"));
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(f64::from(width), 1.0),
            format: Format::Rgba8,
            ..SurfaceOptions::default()
        },
    );
    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .unwrap_or_else(|error| panic!("presented color-filter coverage must configure: {error}"));
    let presentation = presented_observation_handle_for_test(&surface);
    let rendered = pollster::block_on(renderer.render_color_filter_fixture_for_test(
        &mut surface,
        &scene,
        filters,
        parameters,
        working_format,
    ));
    let one_production_submission = rendered
        .as_ref()
        .is_ok_and(|frame| frame.stats.route == Some(RenderRoute::GpuGraph));
    let presentation = presentation.snapshot_for_test();
    let presented = take_last_presented_texture_for_test(&mut surface)
        .and_then(|texture| {
            pollster::block_on(
                renderer.read_render_texture_for_test(&texture, PhysicalSize::new(width, 1)),
            )
            .ok()
        })
        .map(|image| image.into_rgba());
    let exact_graph = rendered.as_ref().is_ok_and(|rendered| {
        rendered.working_format == working_format
            && rendered.output_extent == PhysicalSize::new(width, 1)
            && rendered.source_origin == (COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X, 0)
            && rendered.source_extent
                == PhysicalSize::new(
                    width + COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X.unsigned_abs(),
                    1,
                )
            && rendered.source_texel_origin
                == Point::new(f64::from(COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X), 0.0)
            && rendered.source_raster_scale == 1.0
            && rendered.stats == renderer.stats()
    });

    assert!(
        exact_graph
            && one_production_submission
            && presentation.acquire_count_for_test() == 1
            && presentation.present_count_for_test() == 1
            && presentation.discarded_count_for_test() == 0
            && surface.headless_publication_count_for_test() == 0
            && surface.last_parameters == Some(parameters)
            && presented.as_deref() == Some(expected.as_slice()),
        "the presented color-filter fixture did not use the production graph transaction and host effects"
    );
}

#[test]
fn public_dispatch_routes_composition_and_color_filters_but_rejects_broad_backdrop() {
    let (color_filter_scene, color_filters, color_filter_expected) =
        color_filter_retention_fixture_for_test();
    let color_filter_width = u32::try_from(color_filter_expected.len() / 4)
        .expect("the color-filter width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("composition and color-filter dispatch coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);

    let mut color_filter_surface = pollster::block_on(
        renderer.create_headless(Size::new(f64::from(color_filter_width), 1.0), 1.0),
    )
    .unwrap_or_else(|error| panic!("color-filter dispatch coverage requires a surface: {error}"));
    let color_filter = render_color_filter_fixture_for_test(
        &mut renderer,
        &mut color_filter_surface,
        &color_filter_scene,
        color_filters.clone(),
        Parameters::default(),
        working_format,
    );

    let (composition_scene, composition_size, _, _) = composition_reuse_scene_and_oracle_for_test();
    let mut composition_surface = pollster::block_on(renderer.create_headless(
        Size::new(
            f64::from(composition_size.width()),
            f64::from(composition_size.height()),
        ),
        1.0,
    ))
    .unwrap_or_else(|error| panic!("masked-composition coverage requires a surface: {error}"));
    let composition = pollster::block_on(renderer.render(
        &mut composition_surface,
        &composition_scene,
        Parameters::default(),
    ));

    let unsupported_scene = color_filter_unsupported_backdrop_scene_for_test();
    let mut unsupported_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap_or_else(
            |error| panic!("broad-backdrop rejection coverage requires a surface: {error}"),
        );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("dispatch coverage must retain its ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let unsupported = pollster::block_on(renderer.render(
        &mut unsupported_surface,
        &unsupported_scene,
        Parameters::default(),
    ));
    let color_diagnostic = color_filter_public_color_graph_diagnostic_for_test(
        &color_filter_scene,
        color_filters,
        Size::new(f64::from(color_filter_width), 1.0),
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("retained public diagnostics must keep the ready device");
    let resources_after = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let expected_color = UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::GpuColorFilterExecution,
    );
    let expected_backdrop = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );

    assert!(
        color_filter.output.rgba() == color_filter_expected
            && color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                &color_filter,
                color_filter_width
            )
            && composition.is_ok()
            && unsupported
                .as_ref()
                .is_err_and(|error| error.unsupported_primitive() == Some(expected_backdrop))
            && color_diagnostic == Some(expected_color)
            && resources_after == resources_before
            && cache_after == cache_before
            && unsupported_surface.headless_publication_count_for_test() == 0,
        "public dispatch misrouted masked composition, color filters, or broad backdrop"
    );
}

#[test]
fn graph_render_submits_one_transaction_and_publishes_once() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_panic_for_test("graph submission coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .unwrap_or_panic_for_test("graph submission coverage requires a headless surface");
    let publication_before = surface.headless_publication_count_for_test();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .unwrap_or_panic_for_test("the forced graph route must invoke the production graph executor");

    let production_graph_transaction = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before)
        == 1
        && graph.output_extent == PhysicalSize::new(2, 2)
        && graph.stats.route == Some(RenderRoute::GpuGraph)
        && graph.stats == renderer.stats();
    assert!(
        production_graph_transaction,
        "the production graph did not commit one transaction and publication"
    );
}

#[derive(Debug)]
struct CompositionProductionFrameForTest {
    output: ImageBuffer,
    stats: Stats,
    working_format: WorkingFormat,
    publication_count: usize,
}

fn composition_mask_image_from_alpha_for_test(
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

fn reference_premultiplied_pixel_for_test(straight: [u8; 4]) -> PremultipliedRgba8 {
    PremultipliedRgba8::try_new(
        premultiply_u8_channel_for_test(straight[0], straight[3]),
        premultiply_u8_channel_for_test(straight[1], straight[3]),
        premultiply_u8_channel_for_test(straight[2], straight[3]),
        straight[3],
    )
    .unwrap()
}

fn reference_solid_for_test(
    size: PhysicalSize,
    straight: [u8; 4],
) -> ReferencePremultipliedRgba8Buffer {
    let pixel_count = usize::try_from(size.width())
        .unwrap()
        .checked_mul(usize::try_from(size.height()).unwrap())
        .unwrap();
    ReferencePremultipliedRgba8Buffer::from_pixels(
        size,
        vec![reference_premultiplied_pixel_for_test(straight); pixel_count],
    )
    .unwrap()
}

fn reference_straight_bytes_for_test(buffer: &ReferencePremultipliedRgba8Buffer) -> Vec<u8> {
    reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(buffer)
        .unwrap()
        .into_rgba()
}

fn color_from_straight_rgba8_for_test(straight: [u8; 4]) -> Color {
    Color::try_rgba(
        f32::from(straight[0]) / 255.0,
        f32::from(straight[1]) / 255.0,
        f32::from(straight[2]) / 255.0,
        f32::from(straight[3]) / 255.0,
    )
    .unwrap()
}

fn graph_pixels_match_for_test(
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

fn render_composition_headless_for_test(
    scene: &Scene,
    size: Size,
    parameters: Parameters,
    working_format: WorkingFormat,
) -> CompositionProductionFrameForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("masked-composition production execution requires a compatible renderer: {error}")
    });
    let mut surface =
        pollster::block_on(renderer.create_headless(size, 1.0)).unwrap_or_else(|error| {
            panic!("masked-composition production execution requires a headless surface: {error}")
        });
    let publication_before = surface.headless_publication_count_for_test();
    let stats = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!(
            "the masked-composition fixture must reach its current production render route: {error}"
        )
    });
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
        panic!("the masked-composition fixture publication must be explicitly readable: {error}")
    });
    CompositionProductionFrameForTest {
        output,
        stats,
        working_format,
        publication_count,
    }
}

fn composition_frame_used_one_atomic_graph_submission_for_test(
    rendered: &CompositionProductionFrameForTest,
) -> bool {
    rendered.publication_count == 1
        && rendered.stats.route == Some(RenderRoute::GpuGraph)
        && rendered.stats.commands > 0
}

fn composition_supported_working_formats_for_test() -> Vec<WorkingFormat> {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("masked-composition format coverage requires a compatible renderer: {error}")
    });
    graph_supported_working_formats_for_test(&mut renderer)
}

fn composition_reuse_scene_and_oracle_for_test()
-> (Scene, PhysicalSize, Vec<u8>, ResolvedMaskUploadKey) {
    let size = PhysicalSize::new(4, 4);
    let bounds = Rect::new(0.0, 0.0, 4.0, 4.0);
    let source = [224, 64, 32, 192];
    let destination = [48, 160, 208, 255];
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(2, 2),
        &[160, 160, 160, 160],
        ImageQuality::High,
        Extend::Reflect,
    );
    let mask_key = ResolvedMaskUploadDescriptor::try_from_image(mask.clone())
        .unwrap_or_else(|error| {
            panic!("the masked-composition reuse mask must produce an upload key: {error}")
        })
        .cache_key();
    let layer = Layer::new()
        .try_clip(Shape::rect(bounds))
        .unwrap_or_else(|error| panic!("the masked-composition reuse clip must be valid: {error}"))
        .try_opacity(0.75)
        .unwrap_or_else(|error| {
            panic!("the masked-composition reuse opacity must be valid: {error}")
        })
        .blend(BlendMode::Multiply)
        .with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(mask.clone(), bounds).unwrap_or_else(|error| {
                panic!("the masked-composition reuse mask bounds must be valid: {error}")
            }),
        );
    let mut scene = Scene::new();
    scene
        .fill(bounds, color_from_straight_rgba8_for_test(destination))
        .layer(layer, |scene| {
            scene.fill(bounds, color_from_straight_rgba8_for_test(source));
        });

    let source = reference_solid_for_test(size, source)
        .apply_resolved_alpha_mask(bounds, &mask, bounds)
        .unwrap_or_else(|error| {
            panic!("the masked-composition reuse mask oracle must resolve: {error}")
        })
        .apply_opacity(0.75)
        .unwrap_or_else(|error| {
            panic!("the masked-composition reuse opacity oracle must resolve: {error}")
        });
    let destination = reference_solid_for_test(size, destination);
    let expected = source
        .blend_over(&destination, BlendMode::Multiply)
        .unwrap_or_else(|error| {
            panic!("the masked-composition reuse blend oracle must resolve: {error}")
        });
    (
        scene,
        size,
        reference_straight_bytes_for_test(&expected),
        mask_key,
    )
}

#[test]
fn resolved_alpha_mask_preserves_partial_alpha_and_nested_order() {
    let size = PhysicalSize::new(4, 2);
    let bounds = Rect::new(0.0, 0.0, 4.0, 2.0);
    let inner_mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[128],
        ImageQuality::Low,
        Extend::Pad,
    );
    let outer_mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[160],
        ImageQuality::Low,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(outer_mask.clone(), bounds).unwrap(),
        ),
        |scene| {
            scene.fill(
                bounds,
                color_from_straight_rgba8_for_test([32, 64, 224, 255]),
            );
            scene.layer(
                Layer::new().with_resolved_alpha_mask(
                    ResolvedLayerAlphaMask::try_new(inner_mask.clone(), bounds).unwrap(),
                ),
                |scene| {
                    scene.fill(
                        bounds,
                        color_from_straight_rgba8_for_test([240, 48, 16, 255]),
                    );
                },
            );
        },
    );

    let inner = reference_solid_for_test(size, [240, 48, 16, 255])
        .apply_resolved_alpha_mask(bounds, &inner_mask, bounds)
        .unwrap();
    let parent = reference_solid_for_test(size, [32, 64, 224, 255]);
    let expected = inner
        .source_over(&parent)
        .unwrap()
        .apply_resolved_alpha_mask(bounds, &outer_mask, bounds)
        .unwrap();
    let expected = reference_straight_bytes_for_test(&expected);

    let preserves_nested_order = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(4.0, 2.0),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    2,
                )
        });

    assert!(
        preserves_nested_order,
        "resolved masks do not compose inner to outer on GPU"
    );
}

fn composition_mask_quality_scene_and_oracle_for_test() -> (Scene, PhysicalSize, Vec<u8>) {
    let qualities = [ImageQuality::Low, ImageQuality::Medium, ImageQuality::High];
    let extends = [Extend::Pad, Extend::Repeat, Extend::Reflect];
    let tile_size = PhysicalSize::new(12, 8);
    let mut expected = Vec::with_capacity(12 * 8 * qualities.len() * extends.len() * 4);
    let mut scene = Scene::new();
    let mut case_index = 0_u32;
    for quality in qualities {
        for extend in extends {
            let mask = composition_mask_image_from_alpha_for_test(
                PhysicalSize::new(2, 2),
                &[16, 96, 160, 240],
                quality,
                extend,
            );
            let mask_bounds = Rect::new(1.0, 1.0, 4.0, 2.0);
            let source_bounds = Rect::new(0.0, 0.0, 6.0, 4.0);
            let transform = Transform::try_new([
                2.0,
                0.0,
                0.0,
                2.0,
                0.0,
                f64::from(case_index * tile_size.height()),
            ])
            .unwrap();
            scene.layer(
                Layer::new()
                    .try_transform(transform)
                    .unwrap()
                    .with_resolved_alpha_mask(
                        ResolvedLayerAlphaMask::try_new(mask.clone(), mask_bounds).unwrap(),
                    ),
                |scene| {
                    scene.fill(
                        source_bounds,
                        color_from_straight_rgba8_for_test([255, 255, 255, 255]),
                    );
                },
            );
            let reference = reference_solid_for_test(tile_size, [255, 255, 255, 255])
                .apply_resolved_alpha_mask(source_bounds, &mask, mask_bounds)
                .unwrap();
            expected.extend(reference_straight_bytes_for_test(&reference));
            case_index += 1;
        }
    }
    (
        scene,
        PhysicalSize::new(tile_size.width(), case_index * tile_size.height()),
        expected,
    )
}

#[test]
fn resolved_alpha_mask_low_medium_high_and_extend_modes_match_boundary_oracle() {
    let (scene, size, expected) = composition_mask_quality_scene_and_oracle_for_test();
    let matches_boundary_oracle = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(f64::from(size.width()), f64::from(size.height())),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    2,
                )
        });

    assert!(
        matches_boundary_oracle,
        "GPU mask quality or edge sampling exceeds the GPU edge tolerance"
    );
}

fn composition_blend_scene_and_oracle_for_test() -> (Scene, PhysicalSize, Vec<u8>) {
    let modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    let source = [204, 64, 153, 160];
    let opaque_destination = [51, 179, 102, 255];
    let tile_width = 2_u32;
    let tile_height = 2_u32;
    let size = PhysicalSize::new(
        tile_width * u32::try_from(modes.len()).unwrap(),
        tile_height * 2,
    );
    let mut expected = vec![0_u8; usize::try_from(size.width() * size.height() * 4).unwrap()];
    let mut scene = Scene::new();
    for (base_index, destination) in [[0, 0, 0, 0], opaque_destination].into_iter().enumerate() {
        for (mode_index, mode) in modes.into_iter().enumerate() {
            let x = f64::from(u32::try_from(mode_index).unwrap() * tile_width);
            let y = f64::from(u32::try_from(base_index).unwrap() * tile_height);
            let rect = Rect::new(x, y, f64::from(tile_width), f64::from(tile_height));
            if destination[3] != 0 {
                scene.fill(rect, color_from_straight_rgba8_for_test(destination));
            }
            let mask = composition_mask_image_from_alpha_for_test(
                PhysicalSize::new(1, 1),
                &[255],
                ImageQuality::Low,
                Extend::Pad,
            );
            scene.layer(
                Layer::new()
                    .blend(mode)
                    .with_resolved_alpha_mask(ResolvedLayerAlphaMask::try_new(mask, rect).unwrap()),
                |scene| {
                    scene.fill(rect, color_from_straight_rgba8_for_test(source));
                },
            );
            let expected_pixel = reference_premultiplied_pixel_for_test(source)
                .blend_over(reference_premultiplied_pixel_for_test(destination), mode);
            let expected_pixel = reference_straight_bytes_for_test(
                &ReferencePremultipliedRgba8Buffer::from_pixels(
                    PhysicalSize::new(1, 1),
                    vec![expected_pixel],
                )
                .unwrap(),
            );
            for pixel_y in 0..tile_height {
                for pixel_x in 0..tile_width {
                    let output_x = u32::try_from(mode_index).unwrap() * tile_width + pixel_x;
                    let output_y = u32::try_from(base_index).unwrap() * tile_height + pixel_y;
                    let offset = usize::try_from((output_y * size.width() + output_x) * 4).unwrap();
                    expected[offset..offset + 4].copy_from_slice(&expected_pixel);
                }
            }
        }
    }
    (scene, size, expected)
}

#[test]
fn all_supported_blends_match_oracle_over_transparent_and_opaque_bases() {
    let (scene, size, expected) = composition_blend_scene_and_oracle_for_test();
    let blends_match = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(f64::from(size.width()), f64::from(size.height())),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    3,
                )
        });

    assert!(
        blends_match,
        "GPU blend output exceeds the GPU pixel tolerance"
    );
}

#[test]
fn plus_blend_clamps_high_precision_results() {
    let source = [255, 128, 64, 204];
    let destination = [128, 255, 192, 204];
    let rect = Rect::new(0.0, 0.0, 4.0, 4.0);
    let mut scene = Scene::new();
    scene.fill(rect, color_from_straight_rgba8_for_test(destination));
    scene.layer(
        Layer::new()
            .blend(BlendMode::Plus)
            .with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(
                    composition_mask_image_from_alpha_for_test(
                        PhysicalSize::new(1, 1),
                        &[255],
                        ImageQuality::Low,
                        Extend::Pad,
                    ),
                    rect,
                )
                .unwrap(),
            ),
        |scene| {
            scene.fill(rect, color_from_straight_rgba8_for_test(source));
        },
    );
    let expected_pixel = reference_premultiplied_pixel_for_test(source).blend_over(
        reference_premultiplied_pixel_for_test(destination),
        BlendMode::Plus,
    );
    let expected = reference_straight_bytes_for_test(
        &ReferencePremultipliedRgba8Buffer::from_pixels(
            PhysicalSize::new(4, 4),
            vec![expected_pixel; 16],
        )
        .unwrap(),
    );
    let rendered = render_composition_headless_for_test(
        &scene,
        Size::new(4.0, 4.0),
        Parameters::default(),
        WorkingFormat::HighPrecision,
    );

    assert!(
        composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
            && graph_pixels_match_for_test(
                rendered.output.rgba(),
                &expected,
                WorkingFormat::HighPrecision,
                3,
            ),
        "Plus exceeded the unit interval"
    );
}

#[test]
fn outer_clip_precedes_mask_and_opacity_on_unfiltered_sources() {
    let surface_size = PhysicalSize::new(4, 2);
    let source_bounds = Rect::new(0.0, 0.0, 4.0, 2.0);
    let clip_bounds = Rect::new(1.0, 0.0, 2.0, 2.0);
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[128],
        ImageQuality::Low,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(clip_bounds))
            .unwrap()
            .try_opacity(0.5)
            .unwrap()
            .with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(mask.clone(), source_bounds).unwrap(),
            ),
        |scene| {
            scene.fill(
                source_bounds,
                color_from_straight_rgba8_for_test([224, 48, 16, 255]),
            );
        },
    );
    let masked = reference_solid_for_test(surface_size, [224, 48, 16, 255])
        .apply_resolved_alpha_mask(source_bounds, &mask, source_bounds)
        .unwrap()
        .apply_opacity(0.5)
        .unwrap();
    let mut expected = reference_straight_bytes_for_test(&masked);
    for y in 0..surface_size.height() {
        for x in [0_u32, 3] {
            let offset = usize::try_from((y * surface_size.width() + x) * 4).unwrap();
            expected[offset..offset + 4].fill(0);
        }
    }
    let rendered = render_composition_headless_for_test(
        &scene,
        Size::new(4.0, 2.0),
        Parameters::default(),
        WorkingFormat::HighPrecision,
    );

    assert!(
        composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
            && rendered.output.size() == surface_size
            && graph_pixels_match_for_test(
                rendered.output.rgba(),
                &expected,
                WorkingFormat::HighPrecision,
                2,
            ),
        "outer composition operations changed order"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn render_window_smoke_executes_masked_and_blended_graph_frames() {
    let source = [224, 64, 32, 192];
    let destination = [48, 160, 208, 255];
    let mask_alpha = 160_u8;
    let rect = Rect::new(0.0, 0.0, 4.0, 4.0);
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[mask_alpha],
        ImageQuality::Low,
        Extend::Pad,
    );
    let scene = composition_presented_masked_blended_scene_for_test(rect);
    let expected_source = reference_solid_for_test(PhysicalSize::new(1, 1), source)
        .apply_resolved_alpha_mask(rect, &mask, rect)
        .unwrap();
    let expected = expected_source
        .blend_over(
            &reference_solid_for_test(PhysicalSize::new(1, 1), destination),
            BlendMode::Multiply,
        )
        .unwrap();
    let expected = reference_straight_bytes_for_test(&expected);
    let parameters = Parameters {
        base_color: color_from_straight_rgba8_for_test(destination),
        debug: false,
    };

    let presented_atomically = [Format::Rgba8, Format::Bgra8].into_iter().all(|format| {
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default()
                .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
        ))
        .unwrap_or_else(|error| {
            panic!("presented masked-composition coverage requires a compatible renderer: {error}")
        });
        let working_format = default_graph_working_format_for_test(&mut renderer);
        let mut surface = display_free_presented_surface_for_test(
            &mut renderer,
            SurfaceOptions {
                size: Size::new(4.0, 4.0),
                format,
                ..SurfaceOptions::default()
            },
        );
        pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
            .unwrap_or_else(|error| {
                panic!(
                    "presented masked-composition coverage requires a configured output: {error}"
                )
            });
        let observation = presented_observation_handle_for_test(&surface);
        let stats = pollster::block_on(renderer.render(&mut surface, &scene, parameters));
        let submitted_atomically = stats.is_ok()
            && stats
                .as_ref()
                .is_ok_and(|stats| stats.route == Some(RenderRoute::GpuGraph));
        let presentation = observation.snapshot_for_test();
        let presented_texture = take_last_presented_texture_for_test(&mut surface);
        let pixel = presented_texture.and_then(|texture| {
            pollster::block_on(
                renderer.read_render_texture_for_test(&texture, PhysicalSize::new(4, 4)),
            )
            .ok()
            .and_then(|image| {
                let offset = (4 + 1) * 4;
                let raw: [u8; 4] = image.rgba().get(offset..offset + 4)?.try_into().ok()?;
                Some(match format {
                    Format::Rgba8 => raw,
                    Format::Bgra8 => [raw[2], raw[1], raw[0], raw[3]],
                })
            })
        });
        submitted_atomically
            && presentation.acquire_count_for_test() == 1
            && presentation.present_count_for_test() == 1
            && presentation.discarded_count_for_test() == 0
            && surface.headless_publication_count_for_test() == 0
            && renderer.stats() == stats.unwrap()
            && surface.last_parameters == Some(parameters)
            && pixel.is_some_and(|pixel| {
                graph_pixels_match_for_test(&pixel, &expected, working_format, 3)
            })
    });

    assert!(
        presented_atomically,
        "the presented masked composition did not commit atomically"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_terminal_signal_after_publication_fails_the_next_operation() {
    let rect = Rect::new(0.0, 0.0, 2.0, 2.0);
    let scene = composition_presented_masked_blended_scene_for_test(rect);

    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| {
        panic!("presented terminal-signal coverage requires a compatible renderer: {error}")
    });
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: color_from_straight_rgba8_for_test([48, 160, 208, 255]),
        debug: true,
    };
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .unwrap_or_else(|error| panic!("the presented graph frame must publish: {error}"));
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 1);
    assert_eq!(presented.acquire_count_for_test(), 1);
    assert_eq!(presented.present_count_for_test(), 1);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats);
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.resource_state(), SurfaceResourceState::Presented);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(surface.headless_publication_count_for_test(), 0);

    let committed_stats = renderer.stats();
    let committed_parameters = surface.last_parameters;
    let committed_lifecycle = presented_lifecycle_for_test(&surface);
    let committed_target = presented_target_identity_for_test(&surface);
    let committed_resource = presented_resource_id_for_test(&surface);
    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("the operation after an idle terminal signal must fail deterministically");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(presented_observation_for_test(&surface), presented);
    assert_eq!(renderer.stats(), committed_stats);
    assert_eq!(surface.last_parameters, committed_parameters);
    assert_eq!(presented_lifecycle_for_test(&surface), committed_lifecycle);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        committed_target
    );
    assert_eq!(presented_resource_id_for_test(&surface), committed_resource);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert!(take_last_presented_texture_for_test(&mut surface).is_some());
}

#[cfg(feature = "render-window")]
fn composition_presented_masked_blended_scene_for_test(rect: Rect) -> Scene {
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

fn unsupported_broad_backdrop_scene(size: Size, inner_bounds: Rect) -> Scene {
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.25).unwrap(),
    )])
    .unwrap();
    let backdrop = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, size.width(), size.height()))
                    .unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, size.width(), size.height()),
            Color::BLACK,
        )
        .layer(backdrop, |scene| {
            scene.fill(inner_bounds, Color::BLACK);
        });
    scene
}

#[test]
fn broad_backdrop_graph_returns_exact_unsupported_diagnostic_without_publication() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(64 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| {
        panic!("broad-backdrop diagnostic coverage requires a compatible renderer: {error}")
    });
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(8.0, 6.0), 1.0))
        .unwrap_or_else(|error| {
            panic!("broad-backdrop diagnostic coverage requires a headless surface: {error}")
        });
    let mut baseline = Scene::new();
    baseline.fill(
        Rect::new(0.0, 0.0, 8.0, 6.0),
        color_from_straight_rgba8_for_test([32, 64, 96, 255]),
    );
    pollster::block_on(renderer.render(&mut surface, &baseline, Parameters::default()))
        .unwrap_or_else(|error| {
            panic!("broad-backdrop diagnostic coverage requires a published baseline: {error}")
        });
    let pixels_before =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the broad-backdrop diagnostic baseline must be readable: {error}")
        });
    let stats_before = renderer.stats();
    let publication_before = surface.headless_publication_count_for_test();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap()
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap()
        .internal_resource_manager_observation_for_test();

    let unsupported =
        unsupported_broad_backdrop_scene(Size::new(8.0, 6.0), Rect::new(2.0, 1.0, 3.0, 3.0));
    let result =
        pollster::block_on(renderer.render(&mut surface, &unsupported, Parameters::default()));
    let pixels_after =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("a rejected broad-backdrop graph must retain its prior publication: {error}")
        });
    let diagnostic_is_exact = result.as_ref().err().is_some_and(|error| {
        error.code() == ErrorCode::UnsupportedPrimitive
            && error.unsupported_primitive().is_some_and(|diagnostic| {
                diagnostic.family() == PrimitiveFamily::OffscreenPipeline
                    && diagnostic.operation().label() == "broad backdrop execution"
            })
    });
    let cache_after = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap()
        .device_pass_cache_counts_for_test();
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap()
        .internal_resource_manager_observation_for_test();

    assert!(
        diagnostic_is_exact
            && renderer.stats() == stats_before
            && surface.headless_publication_count_for_test() == publication_before
            && pixels_after == pixels_before
            && cache_after == cache_before
            && resources_after == resources_before,
        "an unsupported graph entered CPU execution or changed publication"
    );
}

#[test]
fn broad_backdrop_diagnostic_precedes_unavailable_effect_working_format() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("diagnostic ordering requires a real selected WGPU device");
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("diagnostic ordering requires a real headless surface");
    let mut baseline = Scene::new();
    baseline.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &baseline, Parameters::default()))
        .expect("diagnostic ordering requires a published direct baseline");
    let pixels_before = pollster::block_on(renderer.read_headless(&surface))
        .expect("the diagnostic baseline must be readable");
    let stats_before = renderer.stats();
    let publication_before = surface.headless_publication_count_for_test();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the diagnostic baseline must retain its ready cache")
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the diagnostic baseline must retain its ready resources")
        .internal_resource_manager_observation_for_test();
    assert!(
        renderer.override_default_device_effect_precision_facts_for_test(
            EffectPrecisionCapabilities::new(false, false),
        ),
        "the real renderer must accept the scoped no-effect-format capability facts"
    );

    let unsupported =
        unsupported_broad_backdrop_scene(Size::new(4.0, 4.0), Rect::new(1.0, 1.0, 2.0, 2.0));
    let error =
        pollster::block_on(renderer.render(&mut surface, &unsupported, Parameters::default()))
            .expect_err("a broad-backdrop graph must retain its typed unsupported-pass diagnostic");
    let expected = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );
    assert_eq!(
        error.unsupported_primitive(),
        Some(expected),
        "the broad-backdrop diagnostic must precede effect-format resolution: {error:?}"
    );
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        None,
        "effect-format unavailability preempted the broad-backdrop diagnostic"
    );
    let pixels_after = pollster::block_on(renderer.read_headless(&surface))
        .expect("the rejected broad-backdrop graph must retain its prior publication");
    let cache_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the broad-backdrop rejection must retain its ready cache")
        .device_pass_cache_counts_for_test();
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the broad-backdrop rejection must retain its ready resources")
        .internal_resource_manager_observation_for_test();

    assert!(
        renderer.stats() == stats_before
            && surface.headless_publication_count_for_test() == publication_before
            && pixels_after == pixels_before
            && cache_after == cache_before
            && resources_after == resources_before,
        "broad-backdrop rejection allocated, submitted, or changed publication"
    );
}

#[test]
fn capabilities_report_supported_gpu_operations_and_broad_mask_diagnostic() {
    let capabilities = Capabilities::CURRENT;
    let filters = capabilities.filters();
    let masks = capabilities.masks_clips();
    let offscreen = capabilities.offscreen_pipeline();
    let supported = composition_supported_capability_rows(filters, masks, offscreen);
    let rejected = composition_rejected_capability_rows(filters, masks, offscreen);
    let supported_are_exact = supported.into_iter().all(|(query, family, operation)| {
        query
            && capabilities
                .ensure_supported(UnsupportedPrimitive::new(family, operation))
                .is_ok()
    });
    let rejected_are_exact = rejected.into_iter().all(|(query, family, operation)| {
        let expected = UnsupportedPrimitive::new(family, operation);
        !query
            && capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
    });
    let diagnostic_only_rejections = [UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::AlphaMaskSourceExecution,
    )];
    let diagnostic_only_rejections_are_exact =
        diagnostic_only_rejections.into_iter().all(|expected| {
            capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
        });
    assert!(
        supported_are_exact && rejected_are_exact && diagnostic_only_rejections_are_exact,
        "capability surface overclaims broad support"
    );
}

type CapabilityRowForTest = (bool, PrimitiveFamily, PrimitiveOperation);

fn composition_supported_capability_rows(
    filters: FilterCapabilities,
    masks: MaskClipCapabilities,
    offscreen: OffscreenPipelineCapabilities,
) -> [CapabilityRowForTest; 13] {
    [
        (
            filters.supports_gpu_color_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ),
        (
            filters.supports_gpu_blur_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ),
        (
            filters.supports_gpu_drop_shadow_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ),
        (
            masks.supports_resolved_alpha_mask_execution(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::ResolvedAlphaMaskExecution,
        ),
        (
            offscreen.supports_image_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::ImagePassExecution,
        ),
        (
            offscreen.supports_composite_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::CompositePassExecution,
        ),
        (
            offscreen.supports_nested_opacity_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::NestedOpacityComposition,
        ),
        (
            filters.supports_ordered_filter_lists(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::OrderedFilterList,
        ),
        (
            filters.supports_filter_region_planning(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::FilterRegionPlanning,
        ),
        (
            offscreen.supports_persistent_effect_resources(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::PersistentEffectResources,
        ),
        (
            offscreen.supports_bounded_vello_capture(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedVelloCapture,
        ),
        (
            offscreen.supports_bounded_backdrop_capture(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropCapture,
        ),
        (
            offscreen.supports_bounded_backdrop_filter_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropFilterExecution,
        ),
    ]
}

fn composition_rejected_capability_rows(
    _filters: FilterCapabilities,
    masks: MaskClipCapabilities,
    offscreen: OffscreenPipelineCapabilities,
) -> [CapabilityRowForTest; 9] {
    [
        (
            offscreen.supports_layer_filter_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::LayerFilterExecution,
        ),
        (
            offscreen.supports_broad_backdrop_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
        (
            masks.supports_layer_masks(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ),
        (
            offscreen.supports_mask_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaskExecution,
        ),
        (
            masks.supports_luminance_mask_mode(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ),
        (
            masks.supports_multi_layer_mask_composition(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ),
        (
            masks.supports_mask_composite_modes(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ),
        (
            offscreen.supports_offscreen_layer_rendering(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::OffscreenLayerRendering,
        ),
        (
            offscreen.supports_backdrop_isolation_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ),
    ]
}

#[test]
fn repeated_masked_and_blended_frames_reuse_resources_without_growth_or_readback() {
    let (scene, size, expected, mask_key) = composition_reuse_scene_and_oracle_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(512 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| {
        panic!("repeated composition reuse coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .unwrap_or_else(|error| {
        panic!("repeated composition reuse coverage requires a headless surface: {error}")
    });

    for _ in 0..2 {
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .unwrap_or_else(|error| {
                panic!("composition reuse warm-up frames must succeed: {error}")
            });
    }
    let warmed_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the warmed composition publication must be readable: {error}")
        });
    let warmed = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("the warmed composition device must remain ready"));
    let warmed_resources = warmed.internal_resource_manager_observation_for_test();
    let warmed_cache = warmed.device_pass_cache_counts_for_test();

    let mut resource_observations = Vec::new();
    let mut cache_observations = Vec::new();
    let mut stats = Vec::new();
    for _ in 0..3 {
        let frame =
            pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
                .unwrap_or_else(|error| {
                    panic!("repeated composition frames must succeed: {error}")
                });
        stats.push(GraphPublicStatsForTest::from(frame));
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .unwrap_or_else(|| panic!("repeated composition frames must retain the ready device"));
        resource_observations.push(ready.internal_resource_manager_observation_for_test());
        cache_observations.push(ready.device_pass_cache_counts_for_test());
    }
    let stable_resource_set =
        composition_resource_observations_are_stable(&resource_observations, &warmed_resources);
    let exact_mask_key_is_retained = warmed_resources.resolved_mask_upload_keys_for_test()
        == [mask_key]
        && mask_key.physical_size() == PhysicalSize::new(2, 2)
        && mask_key.quality() == ImageQuality::High
        && mask_key.extend() == Extend::Reflect;
    let stable_cache = warmed_cache.has_render_pipelines()
        && cache_observations
            .iter()
            .all(|observation| *observation == warmed_cache);
    let stable_stats = stats
        .first()
        .is_some_and(|first| stats.iter().all(|actual| actual == first));
    let actual = pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
        panic!("the repeated composition publication must remain readable: {error}")
    });

    assert!(
        stable_resource_set
            && exact_mask_key_is_retained
            && warmed_resources.effect_texture_count_for_test() > 0
            && stable_cache
            && stable_stats
            && warmed_output.rgba() == actual.rgba()
            && graph_pixels_match_for_test(actual.rgba(), &expected, working_format, 3),
        "composition resources grow or enter readback"
    );
}

fn composition_resource_observations_are_stable(
    observations: &[super::resource::ResourceManagerObservationForTest],
    warmed: &super::resource::ResourceManagerObservationForTest,
) -> bool {
    observations.iter().all(|observation| {
        observation.leased_count == 0
            && observation.active_frame_count == 0
            && observation.resolved_lease_count == 0
            && observation.next_resource == warmed.next_resource
            && observation.entry_count == warmed.entry_count
            && observation.retained_bytes == warmed.retained_bytes
            && observation.payload_creation_attempts == warmed.payload_creation_attempts
            && observation.entry_identities_for_test() == warmed.entry_identities_for_test()
            && observation.effect_texture_count_for_test() == warmed.effect_texture_count_for_test()
            && observation.resolved_mask_upload_keys_for_test()
                == warmed.resolved_mask_upload_keys_for_test()
            && observation.gaussian_kernel_count_for_test() == 0
    })
}

#[test]
fn budget_zero_releases_composition_resources_without_changing_pixels() {
    let (scene, size, expected, _) = composition_reuse_scene_and_oracle_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .unwrap_or_else(|error| {
        panic!("zero-retention composition coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .unwrap_or_else(|error| {
        panic!("zero-retention composition coverage requires a headless surface: {error}")
    });
    let first = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .unwrap_or_else(|error| {
            panic!("the first zero-retention composition frame must succeed: {error}")
        });
    let first_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the first zero-retention composition publication must be readable: {error}")
        });
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("the zero-retention composition device must remain ready"))
        .device_pass_cache_counts_for_test();

    let second = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .unwrap_or_else(|error| {
            panic!("the repeated zero-retention composition frame must succeed: {error}")
        });
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| {
            panic!("the repeated zero-retention composition device must remain ready")
        });
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let all_idle_resources_are_released = resources.leased_count == 0
        && resources.idle_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
        && resources.effect_texture_count_for_test() == 0
        && resources.resolved_mask_upload_keys_for_test().is_empty()
        && resources.gaussian_kernel_count_for_test() == 0;
    let second_output =
        pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
            panic!("the repeated zero-retention composition publication must be readable: {error}")
        });

    assert!(
        all_idle_resources_are_released
            && cache_before == cache_after
            && cache_after.has_render_pipelines()
            && GraphPublicStatsForTest::from(first) == GraphPublicStatsForTest::from(second)
            && first_output.rgba() == second_output.rgba()
            && graph_pixels_match_for_test(second_output.rgba(), &expected, working_format, 3),
        "zero retention changed composition pixels or kept idle resources"
    );
}

#[test]
fn renderer_dispatches_supported_graphs_and_rejects_unsupported_effects() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| panic!("renderer dispatch coverage requires a renderer: {error}"));
    let working_format = default_graph_working_format_for_test(&mut renderer);

    let mut direct_surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .unwrap_or_else(|error| panic!("direct dispatch coverage requires a surface: {error}"));
    let mut direct_scene = Scene::new();
    direct_scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let direct = pollster::block_on(renderer.render(
        &mut direct_surface,
        &direct_scene,
        Parameters::default(),
    ));

    let mut forced_graph_surface = pollster::block_on(
        renderer.create_headless(Size::new(4.0, 4.0), 1.0),
    )
    .unwrap_or_else(|error| panic!("forced-graph dispatch coverage requires a surface: {error}"));
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut forced_graph_surface,
        &direct_scene,
        Parameters::default(),
        working_format,
    ));

    let (composition_scene, _, _, _) = composition_reuse_scene_and_oracle_for_test();
    let mut composition_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap_or_else(
            |error| panic!("masked-composition dispatch coverage requires a surface: {error}"),
        );
    let composition = pollster::block_on(renderer.render(
        &mut composition_surface,
        &composition_scene,
        Parameters::default(),
    ));

    let (blur_scene, backdrop_scene) = composition_unsupported_dispatch_scenes_for_test();
    let mut blur_surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .unwrap_or_else(|error| panic!("unsupported blur coverage requires a surface: {error}"));
    let mut backdrop_surface = pollster::block_on(
        renderer.create_headless(Size::new(4.0, 4.0), 1.0),
    )
    .unwrap_or_else(|error| panic!("unsupported backdrop coverage requires a surface: {error}"));
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("unsupported dispatch coverage requires the ready device"))
        .internal_resource_manager_observation_for_test();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("unsupported dispatch coverage requires the ready cache"))
        .device_pass_cache_counts_for_test();
    let blur_result =
        pollster::block_on(renderer.render(&mut blur_surface, &blur_scene, Parameters::default()));
    let backdrop_result = pollster::block_on(renderer.render(
        &mut backdrop_surface,
        &backdrop_scene,
        Parameters::default(),
    ));
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("unsupported rejection must retain the ready device"))
        .internal_resource_manager_observation_for_test();
    let cache_after = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("unsupported rejection must retain the ready cache"))
        .device_pass_cache_counts_for_test();
    let exact_unsupported_diagnostics = blur_result.as_ref().is_err_and(|error| {
        error.unsupported_primitive()
            == Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::LayerFilter,
            ))
    }) && backdrop_result.as_ref().is_err_and(|error| {
        error.unsupported_primitive()
            == Some(UnsupportedPrimitive::new(
                PrimitiveFamily::OffscreenPipeline,
                PrimitiveOperation::BroadBackdropExecution,
            ))
    });

    assert!(
        direct.is_ok()
            && graph.is_ok()
            && composition.is_ok()
            && exact_unsupported_diagnostics
            && resources_after == resources_before
            && cache_after == cache_before
            && blur_surface.headless_publication_count_for_test() == 0
            && backdrop_surface.headless_publication_count_for_test() == 0,
        "dispatch misrouted supported graphs or admitted unsupported effects"
    );
}

fn composition_unsupported_dispatch_scenes_for_test() -> (Scene, Scene) {
    let blur = Filter::try_blur(1.0)
        .unwrap_or_else(|error| panic!("the unsupported blur fixture must be valid: {error}"));
    let blur_layer = Layer::new()
        .try_filter(blur)
        .unwrap_or_else(|error| panic!("the unsupported blur layer must be valid: {error}"));
    let mut blur_scene = Scene::new();
    blur_scene.layer(blur_layer, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            color_from_straight_rgba8_for_test([255, 255, 255, 255]),
        );
    });
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.25)
            .unwrap_or_else(|error| panic!("the unsupported color amount must be valid: {error}")),
    )])
    .unwrap_or_else(|error| panic!("the unsupported filter list must be valid: {error}"));
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 4.0, 4.0))
        .unwrap_or_else(|error| panic!("the unsupported backdrop bounds must be valid: {error}"));
    let input = BackdropFilterInput::try_new(filters, bounds, None)
        .unwrap_or_else(|error| panic!("the unsupported backdrop input must be valid: {error}"));
    let layer = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .try_backdrop_filter(input)
        .unwrap_or_else(|error| panic!("the unsupported backdrop layer must be valid: {error}"));
    let mut backdrop_scene = Scene::new();
    backdrop_scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(1.0, 1.0, 2.0, 2.0),
                color_from_straight_rgba8_for_test([255, 255, 255, 255]),
            );
        });
    (blur_scene, backdrop_scene)
}

#[test]
fn repeated_frames_reuse_resources_without_growth_or_readback() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| panic!("repeated graph reuse coverage requires a renderer: {error}"));
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(8.0, 6.0), 1.0))
        .unwrap_or_else(|error| {
            panic!("repeated graph reuse coverage requires a headless surface: {error}")
        });
    let scene = repeated_graph_scene_for_test();

    for _ in 0..2 {
        pollster::block_on(renderer.render_forced_base_graph_for_test(
            &mut surface,
            &scene,
            Parameters::default(),
            working_format,
        ))
        .unwrap_or_else(|error| panic!("graph reuse warm-up frames must succeed: {error}"));
    }
    let expected = pollster::block_on(renderer.read_headless(&surface))
        .unwrap_or_else(|error| panic!("the warmed graph publication must be readable: {error}"));
    let warmed_resources = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("the warmed graph device must remain ready"))
        .internal_resource_manager_observation_for_test();
    let warmed_cache = renderer
        .default_ready_device_state_borrow_for_test()
        .unwrap_or_else(|| panic!("the warmed graph device must retain its pass cache"))
        .device_pass_cache_counts_for_test();

    let mut resource_observations = Vec::new();
    let mut cache_observations = Vec::new();
    let mut public_stats = Vec::new();
    for _ in 0..3 {
        let result = pollster::block_on(renderer.render_forced_base_graph_for_test(
            &mut surface,
            &scene,
            Parameters::default(),
            working_format,
        ))
        .unwrap_or_else(|error| panic!("repeated graph frames must succeed: {error}"));
        public_stats.push(GraphPublicStatsForTest::from(result.stats));
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .unwrap_or_else(|| panic!("repeated graph frames must retain the ready device"));
        resource_observations.push(ready.internal_resource_manager_observation_for_test());
        cache_observations.push(ready.device_pass_cache_counts_for_test());
    }

    let no_post_warmup_growth =
        graph_resource_observations_are_stable(&resource_observations, &warmed_resources);
    let reusable_vello_resources_are_retained =
        warmed_resources.committed_transient_buffer_count_for_test() > 0
            && warmed_resources.committed_transient_image_count_for_test() > 0;
    let reusable_graph_frame_resources_are_retained =
        graph_frame_resources_are_retained(&warmed_resources);
    let stable_cache_and_pipelines = warmed_cache.has_render_pipelines()
        && cache_observations
            .iter()
            .all(|observation| *observation == warmed_cache);
    let stable_public_report = public_stats
        .first()
        .is_some_and(|first| public_stats.iter().all(|actual| actual == first));
    let actual = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated graph publication must remain readable");

    assert!(
        no_post_warmup_growth
            && reusable_vello_resources_are_retained
            && reusable_graph_frame_resources_are_retained
            && stable_cache_and_pipelines
            && stable_public_report
            && actual.rgba() == expected.rgba(),
        "repeated graph frames grew resources or entered readback"
    );
}

fn repeated_graph_scene_for_test() -> Scene {
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

fn graph_frame_resources_are_retained(
    resources: &super::resource::ResourceManagerObservationForTest,
) -> bool {
    resources.entry_count
        > resources
            .committed_transient_buffer_count_for_test()
            .saturating_add(resources.committed_transient_image_count_for_test())
            .saturating_add(resources.retained_atlas_count_for_test())
}

fn graph_resource_observations_are_stable(
    observations: &[super::resource::ResourceManagerObservationForTest],
    warmed: &super::resource::ResourceManagerObservationForTest,
) -> bool {
    observations.iter().all(|observation| {
        observation.leased_count == 0
            && observation.next_resource == warmed.next_resource
            && observation.entry_count == warmed.entry_count
            && observation.retained_bytes == warmed.retained_bytes
            && observation.payload_creation_attempts == warmed.payload_creation_attempts
            && observation.committed_transient_buffer_count_for_test()
                == warmed.committed_transient_buffer_count_for_test()
            && observation.committed_transient_image_count_for_test()
                == warmed.committed_transient_image_count_for_test()
    })
}

#[test]
fn budget_zero_releases_idle_resources_without_changing_pixels() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("zero-retention graph coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(6.0, 4.0), 1.0))
        .expect("zero-retention graph coverage requires a headless surface");
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 6.0, 4.0),
        Color::try_rgba(0.125, 0.5, 0.875, 0.75).unwrap(),
    );

    let first = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the first zero-retention graph frame must succeed");
    let expected = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first zero-retention graph publication must be readable");
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the first zero-retention frame must retain the ready device")
        .device_pass_cache_counts_for_test();

    let second = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the repeated zero-retention graph frame must succeed");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the repeated zero-retention frame must retain the ready device");
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let released_all_idle = resources.leased_count == 0
        && resources.idle_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0;
    let actual = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated zero-retention graph publication must be readable");

    assert!(
        released_all_idle
            && cache_before == cache_after
            && cache_after.has_render_pipelines()
            && GraphPublicStatsForTest::from(first.stats)
                == GraphPublicStatsForTest::from(second.stats)
            && actual.rgba() == expected.rgba(),
        "zero retention changed graph pixels or retained idle resources"
    );
}

#[test]
fn renderer_public_dispatch_validates_direct_and_masked_composition_routes() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("renderer public-dispatch coverage requires a selected device");
    let working_format = default_graph_working_format_for_test(&mut renderer);

    let mut direct_surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("direct public-dispatch coverage requires a headless surface");
    let mut direct_scene = Scene::new();
    direct_scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let direct = pollster::block_on(renderer.render(
        &mut direct_surface,
        &direct_scene,
        Parameters::default(),
    ));

    let mut exact_surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("forced-graph dispatch coverage requires a headless surface");
    let exact = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut exact_surface,
        &direct_scene,
        Parameters::default(),
        working_format,
    ));

    let mut later_surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
        .expect("masked-composition dispatch coverage requires a headless surface");
    let masked =
        Layer::new().with_resolved_alpha_mask(opaque_planning_mask(PhysicalSize::new(4, 4)));
    let mut later_scene = Scene::new();
    later_scene.layer(masked, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
        );
    });
    let later = pollster::block_on(renderer.render(
        &mut later_surface,
        &later_scene,
        Parameters::default(),
    ));

    assert!(
        direct.is_ok() && exact.is_ok() && later.is_ok(),
        "public dispatch did not validate and route direct, forced-graph, and masked-composition frames"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphParityFixtureForTest {
    SolidShape,
    StableAhemGlyph,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphParityConfigurationForTest {
    antialiasing: Antialiasing,
    scale: f64,
}

const GRAPH_PARITY_CONFIGURATIONS_FOR_TEST: [GraphParityConfigurationForTest; 9] = [
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Area,
        scale: 1.0,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Area,
        scale: 1.25,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Area,
        scale: 2.0,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa8,
        scale: 1.0,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa8,
        scale: 1.25,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa8,
        scale: 2.0,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa16,
        scale: 1.0,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa16,
        scale: 1.25,
    },
    GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa16,
        scale: 2.0,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphParityScenarioForTest {
    Matrix,
    CaptureTransform,
    ParentTransform,
    OrderedCaptureThenParent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphParityCaseForTest {
    fixture: GraphParityFixtureForTest,
    scenario: GraphParityScenarioForTest,
    antialiasing: Antialiasing,
    scale: f64,
    working_format: WorkingFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphParityFailureStageForTest {
    Setup,
    RequestedAntialiasing,
    OutputDimensions,
    CaptureGrid,
    DirectRoute,
    GraphRoute,
    PublicStats,
    InteriorPixels,
    AntialiasedBoundaryPixels,
    InkSupport,
    AlphaWeightedCentroid,
    MatrixCoverage,
}

#[derive(Debug)]
struct GraphParityFailureForTest {
    case: GraphParityCaseForTest,
    stage: GraphParityFailureStageForTest,
    detail: String,
}

impl GraphParityFailureForTest {
    fn new(
        case: GraphParityCaseForTest,
        stage: GraphParityFailureStageForTest,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            case,
            stage,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for GraphParityFailureForTest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "case={:?} stage={:?}: {}",
            self.case, self.stage, self.detail
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphPixelMetricForTest {
    HighPrecisionStraightRgba8,
    ReducedPrecisionAlphaAndPremul8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphParityToleranceForTest {
    metric: GraphPixelMetricForTest,
    interior_levels: u8,
    boundary_levels: u8,
    centroid_device_pixels: f64,
}

impl GraphParityToleranceForTest {
    const fn for_working_format(working_format: WorkingFormat) -> Self {
        match working_format {
            WorkingFormat::HighPrecision => Self {
                metric: GraphPixelMetricForTest::HighPrecisionStraightRgba8,
                interior_levels: 2,
                boundary_levels: 4,
                centroid_device_pixels: 0.25,
            },
            WorkingFormat::ReducedPrecision => Self {
                metric: GraphPixelMetricForTest::ReducedPrecisionAlphaAndPremul8,
                interior_levels: 2,
                boundary_levels: 4,
                centroid_device_pixels: 0.35,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphCaptureMappingForTest {
    capture_transform: Transform,
    parent_to_surface: Transform,
}

impl GraphCaptureMappingForTest {
    const fn identity() -> Self {
        Self {
            capture_transform: Transform::IDENTITY,
            parent_to_surface: Transform::IDENTITY,
        }
    }

    fn combined(self) -> Transform {
        self.capture_transform
            .then(self.parent_to_surface)
            .expect("graph fixture transforms must compose")
    }

    const fn as_frame_mapping(self) -> super::frame::ForcedVelloCaptureMappingForTest {
        super::frame::ForcedVelloCaptureMappingForTest::new(
            self.capture_transform,
            self.parent_to_surface,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphExpectedCaptureGridForTest {
    device_origin: (i32, i32),
    texel_origin: Point,
    extent: PhysicalSize,
    raster_scale: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphPublicStatsForTest {
    commands: usize,
    fills: usize,
    strokes: usize,
    shadows: usize,
    images: usize,
    glyphs: usize,
    layers: usize,
    cache_hits: usize,
    cache_misses: usize,
    uploaded_bytes: u64,
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

#[derive(Debug)]
struct GraphDirectParityOutputForTest {
    image: ImageBuffer,
    stats: GraphPublicStatsForTest,
    planned_antialiasing: Antialiasing,
}

#[derive(Debug)]
struct GraphParityOutputForTest {
    image: ImageBuffer,
    result: super::renderer::ForcedGraphRenderResultForTest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphPixelComparisonProfileForTest {
    FixtureInteriorAndBoundary,
    PlacementBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphDeviceRegionForTest {
    min_x: u32,
    min_y: u32,
    max_x_exclusive: u32,
    max_y_exclusive: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphPixelCoordinateForTest {
    x: u32,
    y: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphPixelMismatchForTest {
    coordinate: GraphPixelCoordinateForTest,
    direct: [u8; 4],
    graph: [u8; 4],
    metric_error: [u8; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphPixelMismatchSummaryForTest {
    mismatch_count: usize,
    maximum_metric_error: [u8; 4],
    first: Option<GraphPixelMismatchForTest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphTileTranslationMismatchForTest {
    surface_coordinate: GraphPixelCoordinateForTest,
    capture_coordinate: GraphPixelCoordinateForTest,
    surface_pixel: [u8; 4],
    capture_pixel: [u8; 4],
    metric_error: [u8; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphAlphaWeightedCentroidForTest {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphNonEmptyAlphaSupportForTest {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    alpha_sum: u64,
    centroid: GraphAlphaWeightedCentroidForTest,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GraphAlphaSupportForTest {
    Empty,
    NonEmpty(GraphNonEmptyAlphaSupportForTest),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GraphCaptureRequestForTest {
    Identity,
    DistinctMapping,
}

fn graph_parity_surface_size_for_test() -> Size {
    Size::new(32.0, 24.0)
}

fn graph_transformed_parity_surface_size_for_test() -> Size {
    Size::new(20.0, 16.0)
}

fn graph_parity_ink_bounds_for_test(fixture: GraphParityFixtureForTest) -> Rect {
    match fixture {
        GraphParityFixtureForTest::SolidShape => Rect::new(4.25, 3.5, 13.5, 11.25),
        GraphParityFixtureForTest::StableAhemGlyph => Rect::new(8.25, 8.5, 10.0, 10.0),
    }
}

fn graph_parity_interior_bounds_for_test(fixture: GraphParityFixtureForTest) -> Rect {
    match fixture {
        GraphParityFixtureForTest::SolidShape => Rect::new(7.0, 6.0, 7.0, 5.0),
        GraphParityFixtureForTest::StableAhemGlyph => Rect::new(11.0, 11.0, 4.0, 4.0),
    }
}

fn graph_parity_scene_for_test(fixture: GraphParityFixtureForTest) -> Scene {
    let mut scene = Scene::new();
    match fixture {
        GraphParityFixtureForTest::SolidShape => {
            scene.fill(
                Rect::new(4.25, 3.5, 13.5, 11.25),
                Color::try_rgba(0.8, 0.2, 0.1, 0.75).unwrap(),
            );
        }
        GraphParityFixtureForTest::StableAhemGlyph => {
            assert_eq!(
                AHEM_GLYPH_X, 58,
                "the parity fixture must retain the proven Ahem X glyph id"
            );
            let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 8.25, 16.5, 10.0).unwrap()];
            scene.text_run(
                TextRun::try_new(
                    ahem_font("direct-graph parity stable Ahem glyph"),
                    10.0,
                    Transform::identity(),
                    TextPaint::try_fill(Color::BLACK.into()).unwrap(),
                    &glyphs,
                    TextRunBounds::try_ink(Rect::new(8.25, 8.5, 10.0, 10.0)).unwrap(),
                )
                .unwrap(),
            );
        }
    }
    scene
}

fn graph_supported_working_formats_for_test(renderer: &mut Renderer) -> Vec<WorkingFormat> {
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

fn graph_frame_plan_antialiasing_for_test(
    scene: &Scene,
    size: Size,
    configuration: GraphParityConfigurationForTest,
) -> Option<Antialiasing> {
    let commands = scene.normalize(Capabilities::CURRENT).ok()?;
    super::frame::frame_plan_result_observation_for_test(
        commands,
        size,
        configuration.scale,
        configuration.antialiasing,
        Color::TRANSPARENT,
    )
    .plan
    .and_then(|plan| {
        (plan.route == super::frame::FramePlanRouteObservation::DirectVello)
            .then_some(plan.antialiasing)
            .flatten()
    })
}

fn graph_render_direct_parity_for_test(
    renderer: &mut Renderer,
    scene: &Scene,
    size: Size,
    configuration: GraphParityConfigurationForTest,
    case: GraphParityCaseForTest,
) -> std::result::Result<GraphDirectParityOutputForTest, GraphParityFailureForTest> {
    let planned_antialiasing = graph_frame_plan_antialiasing_for_test(scene, size, configuration)
        .ok_or_else(|| {
        GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::RequestedAntialiasing,
            "the fixture did not produce one direct plan with an observable AA request",
        )
    })?;
    let mut surface = pollster::block_on(renderer.create_headless(size, configuration.scale))
        .map_err(|error| {
            GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::Setup,
                format!("direct headless surface creation failed: {error}"),
            )
        })?;
    let publication_before = surface.headless_publication_count_for_test();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("a created headless surface must retain its ready device")
        .device_pass_cache_counts_for_test();
    let stats = pollster::block_on(renderer.render(&mut surface, scene, Parameters::default()))
        .map_err(|error| {
            GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::DirectRoute,
                format!("production direct rendering failed: {error}"),
            )
        })?;
    let cache_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("a clean direct frame must retain its ready device")
        .device_pass_cache_counts_for_test();
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let direct_route_is_exact = stats.route == Some(RenderRoute::DirectVello)
        && cache_before == cache_after
        && publication_count == 1
        && renderer.stats() == stats;
    if !direct_route_is_exact {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::DirectRoute,
            format!(
                "direct route changed: public_route={:?}, publication_count={}, cache_before={cache_before:?}, cache_after={cache_after:?}",
                stats.route, publication_count,
            ),
        ));
    }
    let image = pollster::block_on(renderer.read_headless(&surface)).map_err(|error| {
        GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::DirectRoute,
            format!("direct publication readback failed: {error}"),
        )
    })?;
    Ok(GraphDirectParityOutputForTest {
        image,
        stats: stats.into(),
        planned_antialiasing,
    })
}

fn graph_render_graph_parity_for_test(
    renderer: &mut Renderer,
    scene: &Scene,
    size: Size,
    configuration: GraphParityConfigurationForTest,
    case: GraphParityCaseForTest,
    mapping: GraphCaptureMappingForTest,
    request: GraphCaptureRequestForTest,
) -> std::result::Result<GraphParityOutputForTest, GraphParityFailureForTest> {
    let mut surface = pollster::block_on(renderer.create_headless(size, configuration.scale))
        .map_err(|error| {
            GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::Setup,
                format!("graph headless surface creation failed: {error}"),
            )
        })?;
    let publication_before = surface.headless_publication_count_for_test();
    let result = match request {
        GraphCaptureRequestForTest::Identity => {
            pollster::block_on(renderer.render_forced_base_graph_for_test(
                &mut surface,
                scene,
                Parameters::default(),
                case.working_format,
            ))
        }
        GraphCaptureRequestForTest::DistinctMapping => pollster::block_on(
            renderer.render_forced_base_graph_with_capture_mapping_for_test(
                &mut surface,
                scene,
                Parameters::default(),
                case.working_format,
                mapping.as_frame_mapping(),
            ),
        ),
    }
    .map_err(|error| {
        GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::GraphRoute,
            format!("production forced-graph rendering failed: {error}"),
        )
    })?;
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let graph_route_is_exact = publication_count == 1
        && result.stats.route == Some(RenderRoute::GpuGraph)
        && result.stats == renderer.stats()
        && result.working_format == case.working_format;
    if !graph_route_is_exact {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::GraphRoute,
            format!(
                "graph route changed: public_route={:?}, publication_count={}",
                result.stats.route, publication_count,
            ),
        ));
    }
    let image = pollster::block_on(renderer.read_headless(&surface)).map_err(|error| {
        GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::GraphRoute,
            format!("graph publication readback failed: {error}"),
        )
    })?;
    Ok(GraphParityOutputForTest { image, result })
}

fn graph_transform_point_for_test(transform: Transform, point: Point) -> Point {
    let [a, b, c, d, e, f] = transform.as_array();
    Point::new(
        a * point.x() + c * point.y() + e,
        b * point.x() + d * point.y() + f,
    )
}

fn graph_transform_rect_for_test(rect: Rect, transform: Transform) -> Rect {
    let corners = [
        Point::new(rect.x(), rect.y()),
        Point::new(rect.x() + rect.width(), rect.y()),
        Point::new(rect.x(), rect.y() + rect.height()),
        Point::new(rect.x() + rect.width(), rect.y() + rect.height()),
    ]
    .map(|point| graph_transform_point_for_test(transform, point));
    let min_x = corners
        .iter()
        .map(|point| point.x())
        .fold(f64::INFINITY, f64::min);
    let min_y = corners
        .iter()
        .map(|point| point.y())
        .fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|point| point.x())
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = corners
        .iter()
        .map(|point| point.y())
        .fold(f64::NEG_INFINITY, f64::max);
    Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
}

fn graph_largest_singular_value_for_test(transform: Transform) -> f64 {
    let [a, b, c, d, _, _] = transform.as_array();
    let frobenius_squared = a * a + b * b + c * c + d * d;
    let determinant = a * d - b * c;
    let discriminant =
        (frobenius_squared * frobenius_squared - 4.0 * determinant * determinant).max(0.0);
    ((frobenius_squared + discriminant.sqrt()) * 0.5).sqrt()
}

fn graph_expected_capture_grid_for_test(
    local_bounds: Rect,
    mapping: GraphCaptureMappingForTest,
    surface_scale: f64,
) -> GraphExpectedCaptureGridForTest {
    let combined = mapping.combined();
    let mapped_bounds = graph_transform_rect_for_test(local_bounds, combined);
    let raster_scale = surface_scale * graph_largest_singular_value_for_test(combined);
    let device_min_x = (mapped_bounds.x() * raster_scale).floor() as i32;
    let device_min_y = (mapped_bounds.y() * raster_scale).floor() as i32;
    let device_max_x = ((mapped_bounds.x() + mapped_bounds.width()) * raster_scale).ceil() as i32;
    let device_max_y = ((mapped_bounds.y() + mapped_bounds.height()) * raster_scale).ceil() as i32;
    GraphExpectedCaptureGridForTest {
        device_origin: (device_min_x, device_min_y),
        texel_origin: Point::new(
            f64::from(device_min_x) / raster_scale,
            f64::from(device_min_y) / raster_scale,
        ),
        extent: PhysicalSize::new(
            u32::try_from(device_max_x - device_min_x)
                .expect("the bounded parity width must be positive"),
            u32::try_from(device_max_y - device_min_y)
                .expect("the bounded parity height must be positive"),
        ),
        raster_scale,
    }
}

fn graph_device_region_for_test(
    logical: Rect,
    scale: f64,
    output: PhysicalSize,
) -> GraphDeviceRegionForTest {
    GraphDeviceRegionForTest {
        min_x: ((logical.x() * scale).floor().max(0.0) as u32).min(output.width()),
        min_y: ((logical.y() * scale).floor().max(0.0) as u32).min(output.height()),
        max_x_exclusive: (((logical.x() + logical.width()) * scale).ceil().max(0.0) as u32)
            .min(output.width()),
        max_y_exclusive: (((logical.y() + logical.height()) * scale).ceil().max(0.0) as u32)
            .min(output.height()),
    }
}

fn graph_metric_error_for_test(
    direct: [u8; 4],
    graph: [u8; 4],
    metric: GraphPixelMetricForTest,
) -> [u8; 4] {
    match metric {
        GraphPixelMetricForTest::HighPrecisionStraightRgba8 => {
            // Straight RGB has no defined color at zero alpha. Canonicalize that
            // one semantic value before comparing independently quantized RGBA8
            // outputs; nonzero alpha always retains the straight RGB oracle.
            let direct = graph_canonical_pixel_for_test(direct);
            let graph = graph_canonical_pixel_for_test(graph);
            [
                direct[0].abs_diff(graph[0]),
                direct[1].abs_diff(graph[1]),
                direct[2].abs_diff(graph[2]),
                direct[3].abs_diff(graph[3]),
            ]
        }
        GraphPixelMetricForTest::ReducedPrecisionAlphaAndPremul8 => [
            premultiply_u8_channel_for_test(direct[0], direct[3])
                .abs_diff(premultiply_u8_channel_for_test(graph[0], graph[3])),
            premultiply_u8_channel_for_test(direct[1], direct[3])
                .abs_diff(premultiply_u8_channel_for_test(graph[1], graph[3])),
            premultiply_u8_channel_for_test(direct[2], direct[3])
                .abs_diff(premultiply_u8_channel_for_test(graph[2], graph[3])),
            direct[3].abs_diff(graph[3]),
        ],
    }
}

fn graph_compare_pixel_region_for_test(
    direct: &ImageBuffer,
    graph: &ImageBuffer,
    region: GraphDeviceRegionForTest,
    metric: GraphPixelMetricForTest,
    tolerance: u8,
) -> Option<GraphPixelMismatchSummaryForTest> {
    let mut mismatch_count = 0_usize;
    let mut maximum_metric_error = [0_u8; 4];
    let mut first = None;
    for y in region.min_y..region.max_y_exclusive {
        for x in region.min_x..region.max_x_exclusive {
            let direct_pixel = pixel_rgba(direct, x, y);
            let graph_pixel = pixel_rgba(graph, x, y);
            let metric_error = graph_metric_error_for_test(direct_pixel, graph_pixel, metric);
            let mismatched = metric_error.iter().any(|error| *error > tolerance);
            if !mismatched {
                continue;
            }
            mismatch_count = mismatch_count.saturating_add(1);
            for channel in 0..4 {
                maximum_metric_error[channel] =
                    maximum_metric_error[channel].max(metric_error[channel]);
            }
            first.get_or_insert(GraphPixelMismatchForTest {
                coordinate: GraphPixelCoordinateForTest { x, y },
                direct: direct_pixel,
                graph: graph_pixel,
                metric_error,
            });
        }
    }
    (mismatch_count > 0).then_some(GraphPixelMismatchSummaryForTest {
        mismatch_count,
        maximum_metric_error,
        first,
    })
}

fn graph_alpha_support_for_test(image: &ImageBuffer) -> GraphAlphaSupportForTest {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0_u32;
    let mut max_y = 0_u32;
    let mut alpha_sum = 0_u64;
    let mut weighted_x = 0.0_f64;
    let mut weighted_y = 0.0_f64;
    for y in 0..image.size().height() {
        for x in 0..image.size().width() {
            let alpha = u64::from(pixel_alpha(image, x, y));
            if alpha == 0 {
                continue;
            }
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            alpha_sum = alpha_sum.saturating_add(alpha);
            weighted_x += alpha as f64 * (f64::from(x) + 0.5);
            weighted_y += alpha as f64 * (f64::from(y) + 0.5);
        }
    }
    if alpha_sum == 0 {
        GraphAlphaSupportForTest::Empty
    } else {
        GraphAlphaSupportForTest::NonEmpty(GraphNonEmptyAlphaSupportForTest {
            min_x,
            min_y,
            max_x,
            max_y,
            alpha_sum,
            centroid: GraphAlphaWeightedCentroidForTest {
                x: weighted_x / alpha_sum as f64,
                y: weighted_y / alpha_sum as f64,
            },
        })
    }
}

fn graph_has_antialiased_boundary_for_test(image: &ImageBuffer) -> bool {
    let maximum_alpha = image
        .rgba()
        .chunks_exact(4)
        .map(|pixel| pixel[3])
        .max()
        .unwrap_or(0);
    maximum_alpha > 0
        && image
            .rgba()
            .chunks_exact(4)
            .any(|pixel| pixel[3] > 0 && pixel[3] < maximum_alpha)
}

fn graph_compare_support_for_test(
    case: GraphParityCaseForTest,
    direct: GraphAlphaSupportForTest,
    graph: GraphAlphaSupportForTest,
    centroid_tolerance: f64,
) -> std::result::Result<(), GraphParityFailureForTest> {
    let (GraphAlphaSupportForTest::NonEmpty(direct), GraphAlphaSupportForTest::NonEmpty(graph)) =
        (direct, graph)
    else {
        return match (direct, graph) {
            (GraphAlphaSupportForTest::Empty, GraphAlphaSupportForTest::Empty) => Ok(()),
            _ => Err(GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::InkSupport,
                format!("support emptiness differs: direct={direct:?}, graph={graph:?}"),
            )),
        };
    };
    for (axis, direct_edge, graph_edge) in [
        ("min_x", direct.min_x, graph.min_x),
        ("min_y", direct.min_y, graph.min_y),
        ("max_x", direct.max_x, graph.max_x),
        ("max_y", direct.max_y, graph.max_y),
    ] {
        if direct_edge.abs_diff(graph_edge) > 1 {
            return Err(GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::InkSupport,
                format!(
                    "nonzero support {axis} differs by more than one pixel: direct={direct:?}, graph={graph:?}"
                ),
            ));
        }
    }
    let centroid_delta_x = (direct.centroid.x - graph.centroid.x).abs();
    let centroid_delta_y = (direct.centroid.y - graph.centroid.y).abs();
    if centroid_delta_x > centroid_tolerance || centroid_delta_y > centroid_tolerance {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::AlphaWeightedCentroid,
            format!(
                "centroid delta=({centroid_delta_x:.6},{centroid_delta_y:.6}) exceeds {centroid_tolerance:.2}; direct={direct:?}, graph={graph:?}"
            ),
        ));
    }
    Ok(())
}

fn graph_compare_parity_outputs_for_test(
    case: GraphParityCaseForTest,
    direct: GraphDirectParityOutputForTest,
    graph: GraphParityOutputForTest,
    surface_size: Size,
    expected_capture: GraphExpectedCaptureGridForTest,
    expected_mapping: GraphCaptureMappingForTest,
    profile: GraphPixelComparisonProfileForTest,
) -> std::result::Result<(), GraphParityFailureForTest> {
    let tolerance = GraphParityToleranceForTest::for_working_format(case.working_format);
    if direct.planned_antialiasing != case.antialiasing
        || !matches!(
            graph.result.captures.as_slice(),
            [capture] if capture.antialiasing == case.antialiasing
        )
    {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::RequestedAntialiasing,
            format!(
                "requested={:?}, direct_plan={:?}, graph_captures={:?}",
                case.antialiasing, direct.planned_antialiasing, graph.result.captures
            ),
        ));
    }

    let expected_output = PhysicalSize::try_from_logical(surface_size, case.scale)
        .expect("the parity surface size must be valid");
    if direct.image.size() != expected_output
        || graph.image.size() != expected_output
        || graph.result.output_extent != expected_output
    {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::OutputDimensions,
            format!(
                "expected={expected_output:?}, direct={:?}, graph={:?}, graph_root={:?}",
                direct.image.size(),
                graph.image.size(),
                graph.result.output_extent
            ),
        ));
    }

    let [capture] = graph.result.captures.as_slice() else {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::CaptureGrid,
            format!(
                "expected one capture grid, observed {}",
                graph.result.captures.len()
            ),
        ));
    };
    let scale_tolerance = f64::EPSILON * expected_capture.raster_scale.abs().max(1.0) * 32.0;
    let origin_tolerance = f64::EPSILON
        * expected_capture
            .texel_origin
            .x()
            .abs()
            .max(expected_capture.texel_origin.y().abs())
            .max(1.0)
        * 32.0;
    if capture.device_origin != expected_capture.device_origin
        || capture.extent != expected_capture.extent
        || (capture.raster_scale - expected_capture.raster_scale).abs() > scale_tolerance
        || (capture.texel_origin.x() - expected_capture.texel_origin.x()).abs() > origin_tolerance
        || (capture.texel_origin.y() - expected_capture.texel_origin.y()).abs() > origin_tolerance
        || capture.capture_transform != expected_mapping.capture_transform
        || capture.parent_to_surface != expected_mapping.parent_to_surface
    {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::CaptureGrid,
            format!(
                "expected_grid={expected_capture:?}, actual_capture={capture:?}, expected_mapping={expected_mapping:?}"
            ),
        ));
    }

    let graph_stats = GraphPublicStatsForTest::from(graph.result.stats);
    let expected_direct_stats = match profile {
        GraphPixelComparisonProfileForTest::FixtureInteriorAndBoundary => graph_stats,
        GraphPixelComparisonProfileForTest::PlacementBoundary => GraphPublicStatsForTest {
            commands: graph_stats.commands.saturating_add(1),
            layers: graph_stats.layers.saturating_add(1),
            ..graph_stats
        },
    };
    if direct.stats != expected_direct_stats {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::PublicStats,
            format!(
                "direct={:?}, graph={graph_stats:?}, expected_direct={expected_direct_stats:?}",
                direct.stats
            ),
        ));
    }

    graph_compare_parity_pixel_regions(
        case,
        &direct.image,
        &graph.image,
        expected_output,
        profile,
        tolerance,
    )?;

    graph_compare_parity_support(case, &direct.image, &graph.image, tolerance)
}

fn graph_compare_parity_support(
    case: GraphParityCaseForTest,
    direct: &ImageBuffer,
    graph: &ImageBuffer,
    tolerance: GraphParityToleranceForTest,
) -> std::result::Result<(), GraphParityFailureForTest> {
    if !graph_has_antialiased_boundary_for_test(direct)
        || !graph_has_antialiased_boundary_for_test(graph)
    {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::AntialiasedBoundaryPixels,
            "the fixture did not retain an observable partial-alpha AA boundary",
        ));
    }
    graph_compare_support_for_test(
        case,
        graph_alpha_support_for_test(direct),
        graph_alpha_support_for_test(graph),
        tolerance.centroid_device_pixels,
    )
}

fn graph_compare_parity_pixel_regions(
    case: GraphParityCaseForTest,
    direct: &ImageBuffer,
    graph: &ImageBuffer,
    output: PhysicalSize,
    profile: GraphPixelComparisonProfileForTest,
    tolerance: GraphParityToleranceForTest,
) -> std::result::Result<(), GraphParityFailureForTest> {
    let full_output = GraphDeviceRegionForTest {
        min_x: 0,
        min_y: 0,
        max_x_exclusive: output.width(),
        max_y_exclusive: output.height(),
    };
    if let Some(summary) = graph_compare_pixel_region_for_test(
        direct,
        graph,
        full_output,
        tolerance.metric,
        tolerance.boundary_levels,
    ) {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::AntialiasedBoundaryPixels,
            format!(
                "metric={:?}, tolerance={}, summary={summary:?}",
                tolerance.metric, tolerance.boundary_levels
            ),
        ));
    }
    if profile != GraphPixelComparisonProfileForTest::FixtureInteriorAndBoundary {
        return Ok(());
    }
    let interior = graph_device_region_for_test(
        graph_parity_interior_bounds_for_test(case.fixture),
        case.scale,
        output,
    );
    let contains_ink = (interior.min_y..interior.max_y_exclusive)
        .any(|y| (interior.min_x..interior.max_x_exclusive).any(|x| pixel_alpha(direct, x, y) > 0));
    if !contains_ink {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::InteriorPixels,
            format!("fixture interior contains no direct ink: {interior:?}"),
        ));
    }
    if let Some(summary) = graph_compare_pixel_region_for_test(
        direct,
        graph,
        interior,
        tolerance.metric,
        tolerance.interior_levels,
    ) {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::InteriorPixels,
            format!(
                "metric={:?}, tolerance={}, region={interior:?}, summary={summary:?}",
                tolerance.metric, tolerance.interior_levels
            ),
        ));
    }
    Ok(())
}

fn run_graph_parity_matrix_for_test(
    fixture: GraphParityFixtureForTest,
) -> std::result::Result<Vec<GraphParityCaseForTest>, GraphParityFailureForTest> {
    let scene = graph_parity_scene_for_test(fixture);
    let mapping = GraphCaptureMappingForTest::identity();
    let mut completed = Vec::new();
    let mut configuration_index = 0_usize;
    for antialiasing in [
        Antialiasing::Area,
        Antialiasing::Msaa8,
        Antialiasing::Msaa16,
    ] {
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default()
                .with_antialiasing(antialiasing)
                .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
        ))
        .expect("the parity matrix requires a real selected WGPU device");
        let working_formats = graph_supported_working_formats_for_test(&mut renderer);
        for scale in [1.0, 1.25, 2.0] {
            let configuration = GraphParityConfigurationForTest {
                antialiasing,
                scale,
            };
            if GRAPH_PARITY_CONFIGURATIONS_FOR_TEST[configuration_index] != configuration {
                let case = GraphParityCaseForTest {
                    fixture,
                    scenario: GraphParityScenarioForTest::Matrix,
                    antialiasing,
                    scale,
                    working_format: working_formats[0],
                };
                return Err(GraphParityFailureForTest::new(
                    case,
                    GraphParityFailureStageForTest::MatrixCoverage,
                    "the AA/scale case table no longer matches execution order",
                ));
            }
            configuration_index += 1;
            for working_format in working_formats.iter().copied() {
                let case = GraphParityCaseForTest {
                    fixture,
                    scenario: GraphParityScenarioForTest::Matrix,
                    antialiasing,
                    scale,
                    working_format,
                };
                let direct = graph_render_direct_parity_for_test(
                    &mut renderer,
                    &scene,
                    graph_parity_surface_size_for_test(),
                    configuration,
                    case,
                )?;
                let graph = graph_render_graph_parity_for_test(
                    &mut renderer,
                    &scene,
                    graph_parity_surface_size_for_test(),
                    configuration,
                    case,
                    mapping,
                    GraphCaptureRequestForTest::Identity,
                )?;
                graph_compare_parity_outputs_for_test(
                    case,
                    direct,
                    graph,
                    graph_parity_surface_size_for_test(),
                    graph_expected_capture_grid_for_test(
                        graph_parity_ink_bounds_for_test(fixture),
                        mapping,
                        scale,
                    ),
                    mapping,
                    GraphPixelComparisonProfileForTest::FixtureInteriorAndBoundary,
                )?;
                completed.push(case);
            }
        }
    }
    Ok(completed)
}

fn graph_expected_matrix_cases_for_test() -> Vec<GraphParityCaseForTest> {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("parity-matrix completeness requires a real selected WGPU device");
    let working_formats = graph_supported_working_formats_for_test(&mut renderer);
    let mut expected = Vec::new();
    for fixture in [
        GraphParityFixtureForTest::SolidShape,
        GraphParityFixtureForTest::StableAhemGlyph,
    ] {
        for configuration in GRAPH_PARITY_CONFIGURATIONS_FOR_TEST {
            for working_format in working_formats.iter().copied() {
                expected.push(GraphParityCaseForTest {
                    fixture,
                    scenario: GraphParityScenarioForTest::Matrix,
                    antialiasing: configuration.antialiasing,
                    scale: configuration.scale,
                    working_format,
                });
            }
        }
    }
    expected
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GraphTransformedPlacementCaseForTest {
    scenario: GraphParityScenarioForTest,
    mapping: GraphCaptureMappingForTest,
}

fn graph_transformed_placement_cases_for_test() -> [GraphTransformedPlacementCaseForTest; 3] {
    let capture = Transform::try_new([0.0, 1.0, -1.0, 0.0, 10.375, -6.125]).unwrap();
    let parent = Transform::try_new([0.0, -1.0, 1.0, 0.0, -6.25, 14.375]).unwrap();
    [
        GraphTransformedPlacementCaseForTest {
            scenario: GraphParityScenarioForTest::CaptureTransform,
            mapping: GraphCaptureMappingForTest {
                capture_transform: capture,
                parent_to_surface: Transform::identity(),
            },
        },
        GraphTransformedPlacementCaseForTest {
            scenario: GraphParityScenarioForTest::ParentTransform,
            mapping: GraphCaptureMappingForTest {
                capture_transform: Transform::identity(),
                parent_to_surface: parent,
            },
        },
        GraphTransformedPlacementCaseForTest {
            scenario: GraphParityScenarioForTest::OrderedCaptureThenParent,
            mapping: GraphCaptureMappingForTest {
                capture_transform: capture,
                parent_to_surface: parent,
            },
        },
    ]
}

fn graph_transformed_direct_solid_scene_for_test(mapping: GraphCaptureMappingForTest) -> Scene {
    let bounds = graph_parity_ink_bounds_for_test(GraphParityFixtureForTest::SolidShape);
    let transform = mapping.combined();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(bounds, Color::try_rgba(0.8, 0.2, 0.1, 0.75).unwrap());
    });
    scene
}

fn graph_run_transformed_parity_for_test()
-> std::result::Result<Vec<GraphParityCaseForTest>, GraphParityFailureForTest> {
    let configuration = GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa16,
        scale: 1.25,
    };
    let graph_scene = graph_parity_scene_for_test(GraphParityFixtureForTest::SolidShape);
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_antialiasing(configuration.antialiasing)
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("transformed direct-graph parity requires a real selected WGPU device");
    let working_formats = graph_supported_working_formats_for_test(&mut renderer);
    let mut completed = Vec::new();
    for transformed in graph_transformed_placement_cases_for_test() {
        let expected_grid = graph_expected_capture_grid_for_test(
            graph_parity_ink_bounds_for_test(GraphParityFixtureForTest::SolidShape),
            transformed.mapping,
            configuration.scale,
        );
        graph_validate_transformed_grid(
            transformed,
            expected_grid,
            configuration,
            working_formats[0],
        )?;
        let direct_scene = graph_transformed_direct_solid_scene_for_test(transformed.mapping);
        for working_format in working_formats.iter().copied() {
            let case = GraphParityCaseForTest {
                fixture: GraphParityFixtureForTest::SolidShape,
                scenario: transformed.scenario,
                antialiasing: configuration.antialiasing,
                scale: configuration.scale,
                working_format,
            };
            let direct = graph_render_direct_parity_for_test(
                &mut renderer,
                &direct_scene,
                graph_transformed_parity_surface_size_for_test(),
                configuration,
                case,
            )?;
            let graph = graph_render_graph_parity_for_test(
                &mut renderer,
                &graph_scene,
                graph_transformed_parity_surface_size_for_test(),
                configuration,
                case,
                transformed.mapping,
                GraphCaptureRequestForTest::DistinctMapping,
            )?;
            graph_compare_parity_outputs_for_test(
                case,
                direct,
                graph,
                graph_transformed_parity_surface_size_for_test(),
                expected_grid,
                transformed.mapping,
                GraphPixelComparisonProfileForTest::PlacementBoundary,
            )?;
            completed.push(case);
        }
    }
    Ok(completed)
}

fn graph_validate_transformed_grid(
    transformed: GraphTransformedPlacementCaseForTest,
    grid: GraphExpectedCaptureGridForTest,
    configuration: GraphParityConfigurationForTest,
    working_format: WorkingFormat,
) -> std::result::Result<(), GraphParityFailureForTest> {
    let case = GraphParityCaseForTest {
        fixture: GraphParityFixtureForTest::SolidShape,
        scenario: transformed.scenario,
        antialiasing: configuration.antialiasing,
        scale: configuration.scale,
        working_format,
    };
    if grid.device_origin.0 >= 0 && grid.device_origin.1 >= 0 {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::CaptureGrid,
            format!("transformed fixture did not retain a signed origin: {grid:?}"),
        ));
    }
    let fractional = grid.texel_origin.x().fract().abs() > f64::EPSILON
        || grid.texel_origin.y().fract().abs() > f64::EPSILON;
    if !fractional {
        return Err(GraphParityFailureForTest::new(
            case,
            GraphParityFailureStageForTest::CaptureGrid,
            format!("transformed fixture did not retain a fractional texel origin: {grid:?}"),
        ));
    }
    if transformed.scenario == GraphParityScenarioForTest::OrderedCaptureThenParent {
        let reverse = transformed
            .mapping
            .parent_to_surface
            .then(transformed.mapping.capture_transform)
            .expect("the reverse-order probe transforms must compose");
        if transformed.mapping.combined() == reverse {
            return Err(GraphParityFailureForTest::new(
                case,
                GraphParityFailureStageForTest::CaptureGrid,
                "the ordered transform probe accidentally commutes",
            ));
        }
    }
    Ok(())
}

#[test]
fn direct_and_graph_routes_match_each_fixture_configuration_and_pixel_oracle() {
    let mut actual = Vec::new();
    for (label, fixture) in [
        (
            "solid-shape interior and antialiased edges",
            GraphParityFixtureForTest::SolidShape,
        ),
        (
            "Ahem glyph ink extent and capture grid",
            GraphParityFixtureForTest::StableAhemGlyph,
        ),
    ] {
        actual.extend(
            run_graph_parity_matrix_for_test(fixture)
                .unwrap_or_else(|failure| panic!("{label} parity failed: {failure}")),
        );
    }
    let expected = graph_expected_matrix_cases_for_test();
    assert_eq!(
        actual,
        expected,
        "direct/graph parity matrix is incomplete: actual_count={}, expected_count={}",
        actual.len(),
        expected.len()
    );
}

#[test]
fn negative_bounds_and_subpixel_transforms_do_not_shift_capture() {
    let completed = graph_run_transformed_parity_for_test().unwrap_or_else(|failure| {
        panic!("transformed signed capture placement exceeds GPU pixel tolerance: {failure}")
    });
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("transformed parity coverage requires a real selected WGPU device");
    let expected_count = graph_supported_working_formats_for_test(&mut renderer).len() * 3;
    assert_eq!(
        completed.len(),
        expected_count,
        "transformed signed capture placement exceeds GPU pixel tolerance: completed_count={}, expected_count={expected_count}",
        completed.len()
    );
}

#[test]
fn internal_vello_msaa8_mask_lut_ties_are_tile_translation_invariant() {
    let fixture = GraphParityFixtureForTest::SolidShape;
    let configuration = GraphParityConfigurationForTest {
        antialiasing: Antialiasing::Msaa8,
        scale: 1.25,
    };
    let case = GraphParityCaseForTest {
        fixture,
        scenario: GraphParityScenarioForTest::Matrix,
        antialiasing: configuration.antialiasing,
        scale: configuration.scale,
        working_format: WorkingFormat::HighPrecision,
    };
    let scene = graph_parity_scene_for_test(fixture);
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_antialiasing(configuration.antialiasing),
    ))
    .expect("the focused Vello tile-translation regression requires a real selected device");
    let direct = graph_render_direct_parity_for_test(
        &mut renderer,
        &scene,
        graph_parity_surface_size_for_test(),
        configuration,
        case,
    )
    .unwrap_or_else(|failure| panic!("focused direct Vello setup failed: {failure}"));
    let normalized = scene
        .normalize(renderer.capabilities())
        .expect("the focused solid fixture must normalize");
    let grid = graph_expected_capture_grid_for_test(
        graph_parity_ink_bounds_for_test(fixture),
        GraphCaptureMappingForTest::identity(),
        configuration.scale,
    );
    let initial_transform = Transform::translation(-grid.texel_origin.x(), -grid.texel_origin.y())
        .and_then(|translation| {
            translation.then(Transform::scale(configuration.scale, configuration.scale)?)
        })
        .expect("the focused capture transform must remain finite");
    let local_scene = encode_vello_scene_with_initial_transform(&normalized, initial_transform)
        .expect("the focused capture scene must encode");
    let local_bounds = command::OffscreenBounds::try_new(Rect::new(
        0.0,
        0.0,
        f64::from(grid.extent.width()),
        f64::from(grid.extent.height()),
    ))
    .expect("the focused capture extent must form positive offscreen bounds");
    let options = renderer.options();
    let context = renderer
        .default_offscreen_render_context()
        .expect("the focused Vello tile-translation regression requires its selected device");
    let local = pollster::block_on(render_internal_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &local_scene,
        OffscreenLocalSceneRenderRequest::new(
            local_bounds,
            1.0,
            Format::Rgba8,
            Parameters::default(),
        ),
    ))
    .expect("the focused bounded Vello capture must render");
    let local_image = pollster::block_on(
        renderer.read_render_texture_for_test(
            local
                .texture()
                .expect("the focused capture lease must own its texture"),
            grid.extent,
        ),
    )
    .expect("the focused bounded Vello capture must be readable after submission");

    let (mismatch_count, first) =
        graph_tile_translation_mismatches(&direct.image, &local_image, grid);
    local
        .release()
        .expect("the focused capture lease must release");
    assert_eq!(
        mismatch_count, 0,
        "internal Vello MSAA8 LUT-boundary coverage changed under integer tile translation: first={first:?}"
    );
}

fn graph_tile_translation_mismatches(
    direct: &ImageBuffer,
    local: &ImageBuffer,
    grid: GraphExpectedCaptureGridForTest,
) -> (usize, Option<GraphTileTranslationMismatchForTest>) {
    let mut count = 0usize;
    let mut first = None;
    for local_y in 0..grid.extent.height() {
        for local_x in 0..grid.extent.width() {
            let surface_x = u32::try_from(i64::from(grid.device_origin.0) + i64::from(local_x))
                .expect("the focused identity capture must remain on the positive surface");
            let surface_y = u32::try_from(i64::from(grid.device_origin.1) + i64::from(local_y))
                .expect("the focused identity capture must remain on the positive surface");
            let direct_pixel = pixel_rgba(direct, surface_x, surface_y);
            let local_pixel = pixel_rgba(local, local_x, local_y);
            let error = graph_metric_error_for_test(
                direct_pixel,
                local_pixel,
                GraphPixelMetricForTest::HighPrecisionStraightRgba8,
            );
            if error.iter().any(|channel| *channel > 4) {
                count = count.saturating_add(1);
                first.get_or_insert(GraphTileTranslationMismatchForTest {
                    surface_coordinate: GraphPixelCoordinateForTest {
                        x: surface_x,
                        y: surface_y,
                    },
                    capture_coordinate: GraphPixelCoordinateForTest {
                        x: local_x,
                        y: local_y,
                    },
                    surface_pixel: direct_pixel,
                    capture_pixel: local_pixel,
                    metric_error: error,
                });
            }
        }
    }
    (count, first)
}

#[test]
fn gpu_mask_render_preserves_single_transaction_generation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("materialized-mask transaction coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0))
        .expect("materialized-mask transaction coverage requires a headless surface");
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 0, 0, 0, 128],
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask)),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
        },
    );

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("materialized masks must render through the production path");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(stats.route, Some(RenderRoute::GpuGraph));
    assert_eq!(renderer.stats(), stats);
    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

fn vello_pixel_characterization_scene() -> Scene {
    let partial_red = Color::try_rgba(0.8, 0.2, 0.1, 0.5).unwrap();
    let blue = Color::try_rgba(0.1, 0.25, 0.9, 1.0).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(2.0, 16.0),
        Point::new(18.0, 16.0),
        vec![
            GradientStop::try_new(0.0, Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap()).unwrap(),
            GradientStop::try_new(1.0, Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap()).unwrap(),
        ],
    )
    .unwrap();
    let image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ]),
    )
    .unwrap();
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 2.0, 38.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 14.0, 38.0, 10.0).unwrap(),
    ];
    let mut scene = Scene::new();

    scene.fill(Rect::new(2.25, 2.25, 8.0, 8.0), partial_red);
    scene.stroke(
        Rect::new(14.25, 2.25, 9.0, 9.0),
        Stroke::try_new(3.0).unwrap(),
        blue,
    );
    scene.fill(Rect::new(2.0, 16.0, 16.0, 8.0), Paint::gradient(gradient));
    scene.image(image, Rect::new(22.0, 16.0, 8.0, 8.0), ImageFit::Stretch);
    scene.clip(Rect::new(36.0, 16.0, 6.0, 8.0), |scene| {
        scene.fill(
            Rect::new(32.0, 14.0, 14.0, 12.0),
            Color::try_rgba(1.0, 1.0, 0.0, 1.0).unwrap(),
        );
    });
    scene.transform(Transform::translation(6.0, 3.0).unwrap(), |scene| {
        scene.fill(
            Rect::new(48.0, 14.0, 8.0, 7.0),
            Color::try_rgba(0.0, 1.0, 1.0, 1.0).unwrap(),
        );
    });
    scene.text_run(
        TextRun::try_new(
            ahem_font("Vello pixel characterization"),
            10.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    scene
}

fn observe_vello_pixel_characterization(
    antialiasing: Antialiasing,
    surface: &Surface,
    image: &ImageBuffer,
) -> VelloPixelCharacterizationCase {
    let logical_size = surface.size();
    let scale = surface.scale();
    let surface_physical_size = surface.physical_size();
    let frame_bounds = Rect::new(0.0, 0.0, logical_size.width(), logical_size.height());
    let physical_origin = [
        (frame_bounds.x() * scale).floor() as u32,
        (frame_bounds.y() * scale).floor() as u32,
    ];
    let physical_dimensions = [image.size().width(), image.size().height()];

    assert_eq!(
        physical_dimensions,
        [
            surface_physical_size.width(),
            surface_physical_size.height()
        ],
        "headless image dimensions must match the created surface"
    );

    VelloPixelCharacterizationCase {
        antialiasing,
        scale,
        logical_dimensions: [logical_size.width() as u32, logical_size.height() as u32],
        physical_origin,
        physical_dimensions,
        solid_fill: characterization_pixel(image, scale, 5.0, 5.0),
        stroke: characterization_pixel(image, scale, 15.0, 5.0),
        gradient_left: characterization_pixel(image, scale, 4.0, 20.0),
        gradient_right: characterization_pixel(image, scale, 16.0, 20.0),
        image_top_left: characterization_pixel(image, scale, 23.0, 17.0),
        image_top_right: characterization_pixel(image, scale, 28.0, 17.0),
        clip_inside: characterization_pixel(image, scale, 38.0, 20.0),
        clip_excluded: characterization_pixel(image, scale, 33.0, 20.0),
        transformed_inside: characterization_pixel(image, scale, 56.0, 20.0),
        transformed_excluded: characterization_pixel(image, scale, 50.0, 20.0),
        ahem_ascent_ink: characterization_pixel(image, scale, 7.0, 34.0),
        ahem_descent_ink: characterization_pixel(image, scale, 19.0, 39.0),
        solid_edge: characterization_alpha_support(image, scale, 1.0, 1.0, 11.0, 11.0),
        stroke_edge: characterization_alpha_support(image, scale, 12.0, 0.0, 25.0, 13.0),
        transformed_placement: characterization_alpha_support(image, scale, 54.0, 17.0, 8.0, 7.0),
    }
}

fn characterization_pixel(image: &ImageBuffer, scale: f64, x: f64, y: f64) -> [u8; 4] {
    let x = ((x + 0.5) * scale).floor() as u32;
    let y = ((y + 0.5) * scale).floor() as u32;
    pixel_rgba(image, x, y)
}

fn characterization_alpha_support(
    image: &ImageBuffer,
    scale: f64,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> AlphaSupport {
    let x_start = (x * scale).floor() as u32;
    let y_start = (y * scale).floor() as u32;
    let x_end = ((x + width) * scale).ceil() as u32;
    let y_end = ((y + height) * scale).ceil() as u32;
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut alpha_sum = 0_u64;
    let mut weighted_x = 0_u64;
    let mut weighted_y = 0_u64;

    for pixel_y in y_start..y_end {
        for pixel_x in x_start..x_end {
            let alpha = u64::from(pixel_alpha(image, pixel_x, pixel_y));
            if alpha == 0 {
                continue;
            }
            min_x = min_x.min(pixel_x);
            min_y = min_y.min(pixel_y);
            max_x = max_x.max(pixel_x);
            max_y = max_y.max(pixel_y);
            alpha_sum += alpha;
            weighted_x += alpha * u64::from(pixel_x);
            weighted_y += alpha * u64::from(pixel_y);
        }
    }

    assert!(
        alpha_sum > 0,
        "characterization edge region must contain ink"
    );
    AlphaSupport {
        min_x,
        min_y,
        max_x,
        max_y,
        centroid_x_hundredths: ((weighted_x * 100) / alpha_sum) as i32,
        centroid_y_hundredths: ((weighted_y * 100) / alpha_sum) as i32,
    }
}

fn assert_vello_pixel_characterization_case(
    actual: VelloPixelCharacterizationCase,
    expected: VelloPixelCharacterizationCase,
) {
    assert_eq!(actual.antialiasing, expected.antialiasing);
    assert_eq!(actual.scale, expected.scale);
    assert_eq!(actual.logical_dimensions, expected.logical_dimensions);
    assert_eq!(actual.physical_origin, expected.physical_origin);
    assert_eq!(actual.physical_dimensions, expected.physical_dimensions);

    assert_partial_alpha_straight_rgba8(
        actual.solid_fill,
        expected.solid_fill,
        "partial-alpha solid fill",
    );

    for (name, actual, expected) in [
        ("stroke", actual.stroke, expected.stroke),
        (
            "gradient left",
            actual.gradient_left,
            expected.gradient_left,
        ),
        (
            "gradient right",
            actual.gradient_right,
            expected.gradient_right,
        ),
        (
            "image top left",
            actual.image_top_left,
            expected.image_top_left,
        ),
        (
            "image top right",
            actual.image_top_right,
            expected.image_top_right,
        ),
        ("clip inside", actual.clip_inside, expected.clip_inside),
        (
            "transformed inside",
            actual.transformed_inside,
            expected.transformed_inside,
        ),
        (
            "Ahem ascent ink",
            actual.ahem_ascent_ink,
            expected.ahem_ascent_ink,
        ),
        (
            "Ahem descent ink",
            actual.ahem_descent_ink,
            expected.ahem_descent_ink,
        ),
    ] {
        assert_rgba_within(actual, expected, 2, name);
    }

    assert_eq!(actual.clip_excluded, [0, 0, 0, 0]);
    assert_eq!(actual.transformed_excluded, [0, 0, 0, 0]);
    assert!(
        actual.ahem_ascent_ink[3] > 0,
        "Ahem ascent sample must contain ink"
    );
    assert!(
        actual.ahem_descent_ink[3] > 0,
        "Ahem descent sample must contain ink"
    );
    assert_alpha_support_within(actual.solid_edge, expected.solid_edge, "solid fill edge");
    assert_alpha_support_within(actual.stroke_edge, expected.stroke_edge, "stroke edge");
    assert_transformed_placement_within(
        actual.transformed_placement,
        expected.transformed_placement,
    );
    assert!(actual.gradient_left[0] > actual.gradient_left[2]);
    assert!(actual.gradient_right[2] > actual.gradient_right[0]);
    assert!(actual.image_top_left[0] > actual.image_top_left[1]);
    assert!(actual.image_top_right[1] > actual.image_top_right[0]);
}

fn assert_partial_alpha_straight_rgba8(actual: [u8; 4], expected: [u8; 4], name: &str) {
    assert_rgba_within(actual, expected, 2, name);
    assert!(
        actual[3] > 0 && actual[3] < u8::MAX,
        "{name} must remain partially transparent: {actual:?}"
    );
    assert!(
        actual[0] > actual[3],
        "{name} must retain its straight red channel above alpha: {actual:?}"
    );

    let premultiplied = [
        ((u16::from(actual[0]) * u16::from(actual[3]) + 127) / 255) as u8,
        ((u16::from(actual[1]) * u16::from(actual[3]) + 127) / 255) as u8,
        ((u16::from(actual[2]) * u16::from(actual[3]) + 127) / 255) as u8,
        actual[3],
    ];
    assert!(
        premultiplied[0] <= premultiplied[3] && actual[0].abs_diff(premultiplied[0]) >= 32,
        "{name} must differ materially from its premultiplied representation: {actual:?} -> {premultiplied:?}"
    );
}

fn assert_rgba_within(actual: [u8; 4], expected: [u8; 4], tolerance: u8, name: &str) {
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "{name} channel {channel} expected {expected} +/- {tolerance}, got {actual}"
        );
    }
}

fn assert_alpha_support_within(actual: AlphaSupport, expected: AlphaSupport, name: &str) {
    for (component, actual, expected) in [
        ("min_x", actual.min_x, expected.min_x),
        ("min_y", actual.min_y, expected.min_y),
        ("max_x", actual.max_x, expected.max_x),
        ("max_y", actual.max_y, expected.max_y),
    ] {
        assert!(
            actual.abs_diff(expected) <= 1,
            "{name} nonzero support {component} expected {expected} +/- 1, got {actual}"
        );
    }
    assert!(
        (actual.centroid_x_hundredths - expected.centroid_x_hundredths).abs() <= 35,
        "{name} centroid x exceeds the 0.35-device-pixel GPU centroid tolerance"
    );
    assert!(
        (actual.centroid_y_hundredths - expected.centroid_y_hundredths).abs() <= 35,
        "{name} centroid y exceeds the 0.35-device-pixel GPU centroid tolerance"
    );
}

fn assert_transformed_placement_within(actual: AlphaSupport, expected: AlphaSupport) {
    for (component, actual, expected) in [
        ("min_x", actual.min_x, expected.min_x),
        ("min_y", actual.min_y, expected.min_y),
        ("max_x", actual.max_x, expected.max_x),
        ("max_y", actual.max_y, expected.max_y),
    ] {
        assert!(
            actual.abs_diff(expected) <= 1,
            "transformed rectangle nonzero support {component} expected {expected} +/- 1, got {actual}"
        );
    }
    assert!(
        (actual.centroid_x_hundredths - expected.centroid_x_hundredths).abs() <= 35,
        "transformed rectangle centroid x exceeds the 0.35-device-pixel GPU centroid tolerance"
    );
    assert!(
        (actual.centroid_y_hundredths - expected.centroid_y_hundredths).abs() <= 35,
        "transformed rectangle centroid y exceeds the 0.35-device-pixel GPU centroid tolerance"
    );
}

fn assert_bounded_backdrop_filter_execution_is_public(scene: &Scene, size: Size) {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap();
    let mut surface = pollster::block_on(renderer.create_headless(size, 1.0)).unwrap();
    let publication_before = surface.headless_publication_count_for_test();
    let stats = pollster::block_on(renderer.render(&mut surface, scene, Parameters::default()))
        .expect("bounded backdrop execution must use the public GPU-graph route");
    assert_eq!(stats.route, Some(RenderRoute::GpuGraph));
    assert_eq!(renderer.stats(), stats);
    assert_eq!(
        surface.headless_publication_count_for_test(),
        publication_before.saturating_add(1)
    );
}

fn render_scene_pixel(renderer: &mut Renderer, scene: &Scene) -> [u8; 4] {
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    pollster::block_on(renderer.render(&mut surface, scene, Parameters::default()))
        .expect("single-pixel blend scene should render through the direct Vello path");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    pixel_rgba(&output, 0, 0)
}

fn color_from_opaque_rgba8(pixel: PremultipliedRgba8) -> Color {
    assert_eq!(
        pixel.alpha(),
        u8::MAX,
        "test helper only accepts opaque straight-compatible pixels"
    );
    Color::try_rgba(
        f32::from(pixel.red()) / 255.0,
        f32::from(pixel.green()) / 255.0,
        f32::from(pixel.blue()) / 255.0,
        1.0,
    )
    .unwrap()
}

fn assert_rgba_near_reference_pixel(
    actual: [u8; 4],
    expected: PremultipliedRgba8,
    tolerance: u8,
    message: &str,
) {
    let expected = [
        expected.red(),
        expected.green(),
        expected.blue(),
        expected.alpha(),
    ];
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let delta = actual.abs_diff(expected);
        assert!(
            delta <= tolerance,
            "{message}: channel {channel} expected {expected} +/- {tolerance}, got {actual}"
        );
    }
}
