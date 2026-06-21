use super::{backend::*, encode::*, surface::SurfaceBackend};
use std::{sync::Arc, time::Duration};

use super::*;
#[test]
fn scene_encoding_is_deterministic() {
    let mut a = Scene::new();
    let mut b = Scene::new();
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    a.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::new(1.0), Color::BLACK);
    b.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::new(1.0), Color::BLACK);

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
            Stroke::new(1.0),
            Color::BLACK,
        )
        .shadow(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Shadow::new(Point::new(0.0, 1.0), 2.0, 0.0, Color::BLACK),
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
fn headless_resize_keeps_target_when_physical_size_is_unchanged() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(10.0, 10.0), 1.0)
        .unwrap();

    surface.resize(Size::new(10.4, 10.4), 1.0).unwrap();

    assert_eq!(surface.size(), Size::new(10.4, 10.4));
    assert_eq!(
        surface.physical_size(),
        PhysicalSize {
            width: 10,
            height: 10,
        }
    );
    assert!(matches!(
        &surface.backend,
        SurfaceBackend::Headless {
            texture: Some(_),
            view: Some(_),
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
    assert_eq!(
        surface.physical_size(),
        PhysicalSize {
            width: 20,
            height: 40,
        }
    );
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
fn web_canvas_attachment_reports_target_requirement() {
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
    assert!(error.message.contains("wasm32"));
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
                Stroke::new(1.0),
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

    assert_eq!(
        output.size,
        PhysicalSize {
            width: 40,
            height: 40,
        }
    );
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
        Layer {
            mask: Some(Shape::Rect(Rect::new(0.0, 0.0, 1.0, 1.0))),
            ..Layer::new()
        },
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
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        Color::rgba(f32::NAN, 0.0, 0.0, 1.0),
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("invalid paint should fail during scene encoding");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(error.message.contains("red channel"));
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
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Paint::Image(image));

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
    scene.transform(Transform::translate(2.0, 0.0), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

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
fn pure_transform_does_not_require_backend_layer() {
    let transform = Layer {
        transform: Transform::translate(1.0, 1.0),
        ..Layer::new()
    };
    let clip = Layer {
        clip: Some(Shape::Rect(Rect::new(0.0, 0.0, 1.0, 1.0))),
        ..Layer::new()
    };
    let opacity = Layer {
        opacity: 0.5,
        ..Layer::new()
    };

    assert!(!requires_vello_layer(&transform));
    assert!(requires_vello_layer(&clip));
    assert!(requires_vello_layer(&opacity));
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
    scene.layer(
        Layer {
            opacity: 0.5,
            ..Layer::new()
        },
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        },
    );

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
        Color::rgba(1.0, 0.0, 0.0, 1.0),
    );
    scene.layer(
        Layer {
            blend: BlendMode::Multiply,
            ..Layer::new()
        },
        |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 2.0, 2.0),
                Color::rgba(0.0, 0.0, 1.0, 1.0),
            );
        },
    );

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
    let glyphs = [TextGlyph {
        id: 1,
        x: 0.0,
        y: 0.0,
        advance: 5.0,
    }];
    let mut scene = Scene::new();
    scene.text_run(TextRun {
        font: FontRef::new(1).named("Test"),
        size: 16.0,
        transform: Transform::identity(),
        paint: TextPaint {
            fill: Color::BLACK.into(),
        },
        glyphs: &glyphs,
    });
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
            Stroke::new(2.0).align(StrokeAlign::Inside),
            Color::BLACK,
        )
        .stroke(
            Shape::Circle {
                center: Point::new(12.0, 12.0),
                radius: 6.0,
            },
            Stroke::new(2.0).align(StrokeAlign::Outside),
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
        Stroke::new(2.0).align(StrokeAlign::Inside),
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
        Stroke::new(2.0).align(StrokeAlign::Outside),
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
        Shape::Circle {
            center: Point::new(12.0, 12.0),
            radius: 4.0,
        },
        Shadow::new(Point::new(1.0, 1.0), 4.0, 1.0, Color::BLACK),
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
        Shape::RoundedRect {
            rect: Rect::new(8.0, 8.0, 16.0, 14.0),
            radii: Radii {
                top_left: 0.0,
                top_right: 5.0,
                bottom_right: 10.0,
                bottom_left: 0.0,
            },
        },
        Shadow::new(Point::new(4.0, 5.0), 8.0, 0.0, Color::BLACK),
    );

    let stats = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("non-uniform rounded shadow should render through corner partitioning");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn aligned_path_strokes_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(10.0, 10.0));
    let mut scene = Scene::new();
    scene.stroke(
        Shape::Path(path),
        Stroke::new(2.0).align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("path offsetting is deliberately explicit");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
}

#[test]
fn layer_masks_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer {
            mask: Some(Shape::Rect(Rect::new(0.0, 0.0, 2.0, 2.0))),
            ..Layer::new()
        },
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("mask lowering should be explicit until implemented");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert!(error.message.contains("masks"));
}

#[test]
fn layer_filters_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::new(24.0, 24.0), 1.0)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer {
            filter: Some(Filter::Blur { radius: 4.0 }),
            ..Layer::new()
        },
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("filter lowering should be explicit until implemented");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert!(error.message.contains("filters"));
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

    assert_eq!(
        image.size,
        PhysicalSize {
            width: 4,
            height: 4
        }
    );
    assert_eq!(image.rgba.len(), 4 * 4 * 4);
    assert!(image.rgba.iter().any(|channel| *channel != 0));
}

fn pixel_alpha(image: &ImageBuffer, x: u32, y: u32) -> u8 {
    pixel_rgba(image, x, y)[3]
}

fn pixel_rgba(image: &ImageBuffer, x: u32, y: u32) -> [u8; 4] {
    let index = ((y * image.size.width + x) * 4 + 3) as usize;
    [
        image.rgba[index - 3],
        image.rgba[index - 2],
        image.rgba[index - 1],
        image.rgba[index],
    ]
}
