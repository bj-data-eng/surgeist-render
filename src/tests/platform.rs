use crate::{
    Attachment, Capabilities, ErrorCode, Format, Options, Parameters, PrimitiveFamily,
    PrimitiveOperation, Renderer, Scene, Size, SurfaceOptions, UnsupportedPrimitive,
};

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
use crate::WebCanvas;

#[cfg(feature = "render-window")]
use crate::{
    BlendMode, Color, EffectQualityPolicy, Extend, ImageQuality, PhysicalSize, Point, Rect,
    RenderRoute, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    backend::{
        configured_display_free_presented_surface_for_test,
        display_free_presented_surface_for_test, presented_observation_handle_for_test,
        require_presented_device_identity_for_test, take_last_presented_texture_for_test,
    },
    resource::WorkingFormat,
};

#[cfg(feature = "render-window")]
use super::{
    COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X, bounded_backdrop_integration_fixture_for_test,
    color_filter_retention_fixture_for_test, color_from_straight_rgba8_for_test,
    composition_mask_image_from_alpha_for_test,
    composition_presented_masked_blended_scene_for_test, default_graph_working_format_for_test,
    graph_pixels_match_for_test, reference_solid_for_test, reference_straight_bytes_for_test,
    spatial_filter_maximum_error_for_test, spatial_filter_mixed_filter_fixture_for_test,
};

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
