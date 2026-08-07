mod frame;
mod gpu;
mod model;
mod style;
mod support;

use super::gpu_transaction::GpuOperationStage;
#[cfg(feature = "render-window")]
use super::gpu_transaction::test_support::graph_terminal_loss_after_submission_for_test;
use super::gpu_transaction::test_support::{
    fault_command_buffer_after_submit_for_test, graph_accounting_failure_after_submission_for_test,
    graph_cancellation_after_submission_for_test, graph_scope_failure_after_submission_for_test,
    submit_readback_observed_for_test,
};
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
    filter::BlurPolicy,
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    resource::{ResourceAccountingFault, ResourceCacheKey, ResourceManager, WorkingFormat},
    style::{ColorFilterOp, ColorFilterPipeline},
    surface::{HeadlessResources, SurfaceBackend},
    texture::EffectTextureDescriptor,
};

use std::{
    future::Future,
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
use support::{
    AHEM_FONT_BYTES, AHEM_GLYPH_ASCENT_E_ACUTE, AHEM_GLYPH_DESCENT_P, AHEM_GLYPH_X, ahem_font,
    box_decoration_edges, pixel_alpha, pixel_rgba, solid_border, text_run_for,
};
#[cfg(feature = "render-window")]
use support::{add_planning_text, bounded_planning_backdrop};

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

fn graph_canonical_pixel_for_test(pixel: [u8; 4]) -> [u8; 4] {
    if pixel[3] == 0 { [0, 0, 0, 0] } else { pixel }
}

fn premultiply_u8_channel_for_test(color: u8, alpha: u8) -> u8 {
    ((u16::from(color) * u16::from(alpha) + 127) / 255) as u8
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
