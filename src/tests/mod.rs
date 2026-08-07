mod frame;
mod gpu;
mod model;
mod platform;
mod style;
mod support;
mod surface;
mod vello;

use super::image::{ResolvedMaskUploadDescriptor, ResolvedMaskUploadKey};
use super::vello_engine::{
    PreparedVelloPass, RasterParameters,
    scene::{VelloRasterScenario, VelloScene},
};
use super::{
    backend::*,
    command,
    encode::*,
    filter::BlurPolicy,
    reference::{PremultipliedRgba8, ReferencePremultipliedRgba8Buffer},
    resource::{ResourceCacheKey, WorkingFormat},
    style::{ColorFilterOp, ColorFilterPipeline},
    surface::{HeadlessResources, SurfaceBackend},
    texture::EffectTextureDescriptor,
};

use std::sync::Arc;

use super::*;

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

fn graph_transform_point_for_test(transform: Transform, point: Point) -> Point {
    let [a, b, c, d, e, f] = transform.as_array();
    Point::new(
        a * point.x() + c * point.y() + e,
        b * point.x() + d * point.y() + f,
    )
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
