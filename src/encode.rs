use super::{
    command::{
        LayerIsolation, NormalizedLayer, RenderClipGeometry, RenderCommand, RenderCommands,
        RenderPaint, RenderShadow, RenderShape, RenderStroke, RenderStrokeShape, ShadowShape,
        kurbo_rounded_rect,
    },
    geometry::{expand_rect, offset_radii},
    paint::PaintKind,
    validation::*,
    *,
};

pub(crate) fn encode_vello_scene(commands: &RenderCommands, scale: f64) -> Result<vello::Scene> {
    let mut encoded = vello::Scene::new();
    encode_vello_commands(
        &commands.commands,
        &mut encoded,
        kurbo::Affine::scale(scale),
    )?;
    Ok(encoded)
}

fn encode_vello_commands(
    commands: &[RenderCommand],
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
) -> Result<()> {
    for command in commands {
        match command {
            RenderCommand::Fill { shape, paint } => encode_fill(scene, transform, shape, paint)?,
            RenderCommand::Stroke {
                shape,
                stroke,
                paint,
            } => encode_stroke(scene, transform, shape, stroke, paint)?,
            RenderCommand::Shadow { shape, shadow } => {
                encode_shadow(scene, transform, shape, shadow)?
            }
            RenderCommand::Image { image, rect, fit } => {
                encode_image(scene, transform, image, *rect, *fit)?
            }
            RenderCommand::TextRun {
                font,
                size,
                transform: run_transform,
                paint,
                glyphs,
            } => encode_text_run(scene, transform, font, *size, *run_transform, paint, glyphs)?,
            RenderCommand::Layer { layer, children } => {
                let layer_transform = transform * kurbo::Affine::from(layer.transform);
                if layer.isolation != LayerIsolation::None {
                    encode_layer_start(scene, layer, layer_transform);
                    encode_vello_commands(children, scene, layer_transform)?;
                    scene.pop_layer();
                } else {
                    encode_vello_commands(children, scene, layer_transform)?;
                }
            }
        }
    }
    Ok(())
}

fn encode_fill(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &RenderShape,
    paint: &RenderPaint,
) -> Result<()> {
    let brush = render_paint_brush(paint)?;
    encode_solid_fill(scene, transform, shape, &brush);
    Ok(())
}

fn encode_solid_fill(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &RenderShape,
    brush: &peniko::Brush,
) {
    match shape {
        RenderShape::Rect(rect) => scene.fill(
            peniko::Fill::NonZero,
            transform,
            brush,
            None,
            &kurbo::Rect::from(*rect),
        ),
        RenderShape::RoundedRect { rect, radii } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            brush,
            None,
            &kurbo_rounded_rect(*rect, *radii),
        ),
        RenderShape::Circle { center, radius } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            brush,
            None,
            &kurbo::Circle::new((center.x(), center.y()), *radius),
        ),
        RenderShape::Ellipse { center, radii } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            brush,
            None,
            &kurbo::Ellipse::new(
                (center.x(), center.y()),
                (radii.width(), radii.height()),
                0.0,
            ),
        ),
        RenderShape::Path(path) => scene.fill(
            peniko::Fill::NonZero,
            transform,
            brush,
            None,
            &path.to_kurbo(),
        ),
    }
}

fn encode_stroke(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &RenderStrokeShape,
    stroke: &RenderStroke,
    paint: &RenderPaint,
) -> Result<()> {
    let brush = render_paint_brush(paint)?;
    let vello_stroke = vello_stroke(stroke);
    match shape {
        RenderStrokeShape::Rect(rect) => scene.stroke(&vello_stroke, transform, &brush, None, rect),
        RenderStrokeShape::RoundedRect(rect) => {
            scene.stroke(&vello_stroke, transform, &brush, None, rect)
        }
        RenderStrokeShape::Circle(circle) => {
            scene.stroke(&vello_stroke, transform, &brush, None, circle)
        }
        RenderStrokeShape::Ellipse(ellipse) => {
            scene.stroke(&vello_stroke, transform, &brush, None, ellipse)
        }
        RenderStrokeShape::Path(path) => scene.stroke(&vello_stroke, transform, &brush, None, path),
    }
    Ok(())
}

fn encode_shadow(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &ShadowShape,
    shadow: &RenderShadow,
) -> Result<()> {
    let color = peniko::Color::from(shadow.color);
    let std_dev = shadow.blur * 0.5;
    match shape {
        ShadowShape::Rect(rect) => {
            let rect = offset_rect(expand_rect(*rect, shadow.spread), shadow.offset);
            scene.draw_blurred_rounded_rect(transform, rect.into(), color, 0.0, std_dev);
        }
        ShadowShape::RoundedRect { rect, radii } => {
            let rect = offset_rect(expand_rect(*rect, shadow.spread), shadow.offset);
            if let Some(radius) = radii.uniform() {
                scene.draw_blurred_rounded_rect(
                    transform,
                    rect.into(),
                    color,
                    (radius + shadow.spread).max(0.0),
                    std_dev,
                );
            } else {
                encode_non_uniform_rounded_shadow(
                    scene,
                    transform,
                    rect,
                    offset_radii(*radii, shadow.spread),
                    shadow.blur,
                    shadow.color,
                )?;
            }
        }
        ShadowShape::Circle { center, radius } => {
            let radius = (radius + shadow.spread).max(0.0);
            let rect = Rect::new(
                center.x() - radius + shadow.offset.x(),
                center.y() - radius + shadow.offset.y(),
                radius * 2.0,
                radius * 2.0,
            );
            scene.draw_blurred_rounded_rect(transform, rect.into(), color, radius, std_dev);
        }
    }
    Ok(())
}

fn encode_non_uniform_rounded_shadow(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    rect: Rect,
    radii: Radii,
    blur: f64,
    color: Color,
) -> Result<()> {
    if blur == 0.0 {
        let brush = peniko::Brush::Solid(color.into());
        encode_solid_fill(
            scene,
            transform,
            &RenderShape::RoundedRect { rect, radii },
            &brush,
        );
        return Ok(());
    }

    let std_dev = blur * 0.5;
    let kernel = 2.5 * std_dev;
    let support = kurbo::Rect::from(rect).inflate(kernel, kernel);
    let mid_x = rect.x() + rect.width() * 0.5;
    let mid_y = rect.y() + rect.height() * 0.5;
    let shadow_rect = kurbo::Rect::from(rect);
    let color = peniko::Color::from(color);

    let regions = [
        (
            kurbo::Rect::new(support.x0, support.y0, mid_x, mid_y),
            radii.top_left(),
        ),
        (
            kurbo::Rect::new(mid_x, support.y0, support.x1, mid_y),
            radii.top_right(),
        ),
        (
            kurbo::Rect::new(mid_x, mid_y, support.x1, support.y1),
            radii.bottom_right(),
        ),
        (
            kurbo::Rect::new(support.x0, mid_y, mid_x, support.y1),
            radii.bottom_left(),
        ),
    ];

    for (clip, radius) in regions {
        scene.draw_blurred_rounded_rect_in(
            &clip,
            transform,
            shadow_rect,
            color,
            radius.max(0.0),
            std_dev,
        );
    }

    Ok(())
}

fn encode_image(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    image: &Image,
    rect: Rect,
    fit: ImageFit,
) -> Result<()> {
    let brush = image_brush(image);
    let image_transform = transform * image_transform(image.size, rect, fit)?;
    let clip_to_target = matches!(fit, ImageFit::Contain | ImageFit::Cover);
    if clip_to_target {
        scene.push_clip_layer(peniko::Fill::NonZero, transform, &kurbo::Rect::from(rect));
    }
    scene.draw_image(&brush, image_transform);
    if clip_to_target {
        scene.pop_layer();
    }
    Ok(())
}

pub(crate) fn image_brush(image: &Image) -> peniko::ImageBrush {
    peniko::ImageBrush::new(image_data(image))
        .with_quality(image.quality.into())
        .with_extend(image.extend.into())
}

pub(crate) fn image_data(image: &Image) -> peniko::ImageData {
    image.data.clone()
}

fn encode_text_run(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    font: &FontRef<'static>,
    size: f32,
    run_transform: Transform,
    paint: &TextPaint,
    glyphs: &[TextGlyph],
) -> Result<()> {
    let Some(data) = font.data.as_ref().map(|data| &data.data) else {
        return Err(invalid_input(
            "text run font data is required for rendering prepared glyphs",
        ));
    };
    let brush = glyph_paint_brush(paint.fill())?;
    scene
        .draw_glyphs(data)
        .font_size(size)
        .transform(transform * kurbo::Affine::from(run_transform))
        .brush(&brush)
        .draw(
            peniko::Fill::NonZero,
            glyphs.iter().map(|glyph| vello::Glyph {
                id: glyph.id(),
                x: glyph.x(),
                y: glyph.y(),
            }),
        );
    Ok(())
}

fn encode_layer_start(scene: &mut vello::Scene, layer: &NormalizedLayer, transform: kurbo::Affine) {
    let blend = vello_blend(layer.blend);
    let alpha = layer.opacity.clamp(0.0, 1.0);
    let use_clip = layer.isolation == LayerIsolation::ClipOnly;
    let default_clip = RenderClipGeometry::Rect(Rect::new(-1.0e9, -1.0e9, 2.0e9, 2.0e9));
    let (clip, clip_transform) = layer
        .clip
        .as_ref()
        .map(|clip| {
            let clip_transform = clip
                .coordinate_space()
                .map(|coordinate_space| {
                    transform * kurbo::Affine::from(coordinate_space.transform())
                })
                .unwrap_or(transform);
            (clip.geometry(), clip_transform)
        })
        .unwrap_or((&default_clip, transform));
    match clip {
        RenderClipGeometry::Rect(rect) => push_vello_layer(
            scene,
            use_clip,
            peniko::Fill::NonZero,
            blend,
            alpha,
            clip_transform,
            &kurbo::Rect::from(*rect),
        ),
        RenderClipGeometry::RoundedRect { rect, radii } => push_vello_layer(
            scene,
            use_clip,
            peniko::Fill::NonZero,
            blend,
            alpha,
            clip_transform,
            &kurbo_rounded_rect(*rect, *radii),
        ),
        RenderClipGeometry::Circle { center, radius } => push_vello_layer(
            scene,
            use_clip,
            peniko::Fill::NonZero,
            blend,
            alpha,
            clip_transform,
            &kurbo::Circle::new((center.x(), center.y()), *radius),
        ),
        RenderClipGeometry::Ellipse { center, radii } => push_vello_layer(
            scene,
            use_clip,
            peniko::Fill::NonZero,
            blend,
            alpha,
            clip_transform,
            &kurbo::Ellipse::new(
                (center.x(), center.y()),
                (radii.width(), radii.height()),
                0.0,
            ),
        ),
        RenderClipGeometry::Path { path, fill_rule } => push_vello_layer(
            scene,
            use_clip,
            vello_fill_rule(*fill_rule),
            blend,
            alpha,
            clip_transform,
            &path.to_kurbo(),
        ),
    }
}

fn push_vello_layer(
    scene: &mut vello::Scene,
    use_clip: bool,
    fill: peniko::Fill,
    blend: peniko::BlendMode,
    alpha: f32,
    transform: kurbo::Affine,
    shape: &impl kurbo::Shape,
) {
    if use_clip {
        scene.push_clip_layer(fill, transform, shape);
    } else {
        scene.push_layer(fill, blend, alpha, transform, shape);
    }
}

pub(crate) fn glyph_paint_brush(paint: &Paint) -> Result<peniko::Brush> {
    match paint.kind() {
        PaintKind::Color(color) => Ok(peniko::Brush::Solid((*color).into())),
        PaintKind::Gradient(gradient) => Ok(peniko::Brush::Gradient(gradient.clone().into())),
        PaintKind::Image(image) => Ok(peniko::Brush::Image(image_brush(image))),
    }
}

fn render_paint_brush(paint: &RenderPaint) -> Result<peniko::Brush> {
    match paint {
        RenderPaint::Color(color) => Ok(peniko::Brush::Solid((*color).into())),
        RenderPaint::Gradient(gradient) => Ok(peniko::Brush::Gradient(gradient.clone().into())),
        RenderPaint::Image(image) => Ok(peniko::Brush::Image(image_brush(image))),
    }
}

fn vello_stroke(stroke: &RenderStroke) -> kurbo::Stroke {
    let mut vello = kurbo::Stroke::new(stroke.width)
        .with_join(stroke.join.into())
        .with_start_cap(stroke.start_cap.into())
        .with_end_cap(stroke.end_cap.into())
        .with_miter_limit(stroke.miter_limit);
    if let Some(dash) = stroke.dash {
        vello.dash_offset = dash.offset();
        vello.dash_pattern = dash.intervals().to_vec().into();
    }
    vello
}

fn vello_blend(blend: BlendMode) -> peniko::BlendMode {
    match blend {
        BlendMode::Normal => peniko::Mix::Normal.into(),
        BlendMode::Multiply => peniko::Mix::Multiply.into(),
        BlendMode::Screen => peniko::Mix::Screen.into(),
        BlendMode::Overlay => peniko::Mix::Overlay.into(),
        BlendMode::Darken => peniko::Mix::Darken.into(),
        BlendMode::Lighten => peniko::Mix::Lighten.into(),
        BlendMode::Plus => peniko::Compose::Plus.into(),
    }
}

fn vello_fill_rule(fill_rule: FillRule) -> peniko::Fill {
    match fill_rule {
        FillRule::NonZero => peniko::Fill::NonZero,
        FillRule::EvenOdd => peniko::Fill::EvenOdd,
    }
}

fn offset_rect(rect: Rect, offset: Point) -> Rect {
    Rect::new(
        rect.x() + offset.x(),
        rect.y() + offset.y(),
        rect.width(),
        rect.height(),
    )
}

pub(crate) fn image_transform(size: Size, rect: Rect, fit: ImageFit) -> Result<kurbo::Affine> {
    if size.width() <= 0.0 || size.height() <= 0.0 {
        return Err(Error::new(
            ErrorCode::ImageUploadFailed,
            "image size must be positive",
        ));
    }
    let scale_x = rect.width() / size.width();
    let scale_y = rect.height() / size.height();
    let (fit_scale_x, fit_scale_y, tx, ty) = match fit {
        ImageFit::Fill | ImageFit::Stretch | ImageFit::None => {
            (scale_x, scale_y, rect.x(), rect.y())
        }
        ImageFit::Contain => {
            let scale = scale_x.min(scale_y);
            (
                scale,
                scale,
                rect.x() + (rect.width() - size.width() * scale) * 0.5,
                rect.y() + (rect.height() - size.height() * scale) * 0.5,
            )
        }
        ImageFit::Cover => {
            let scale = scale_x.max(scale_y);
            (
                scale,
                scale,
                rect.x() + (rect.width() - size.width() * scale) * 0.5,
                rect.y() + (rect.height() - size.height() * scale) * 0.5,
            )
        }
    };
    Ok(kurbo::Affine::translate((tx, ty))
        * kurbo::Affine::scale_non_uniform(fit_scale_x, fit_scale_y))
}
