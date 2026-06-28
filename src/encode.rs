use super::{
    geometry::{expand_rect, offset_radii},
    paint::PaintKind,
    scene::Command,
    shape::ShapeKind,
    validation::*,
    *,
};

pub(crate) fn encode_vello_scene(scene: &Scene, scale: f64) -> Result<vello::Scene> {
    let mut encoded = vello::Scene::new();
    encode_vello_commands(&scene.commands, &mut encoded, kurbo::Affine::scale(scale))?;
    Ok(encoded)
}

fn encode_vello_commands(
    commands: &[Command],
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
) -> Result<()> {
    for command in commands {
        match command {
            Command::Fill { shape, paint } => encode_fill(scene, transform, shape, paint)?,
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => encode_stroke(scene, transform, shape, *stroke, paint)?,
            Command::Shadow { shape, shadow } => encode_shadow(scene, transform, shape, shadow)?,
            Command::Image { image, rect, fit } => {
                encode_image(scene, transform, image, *rect, *fit)?
            }
            Command::TextRun {
                font,
                size,
                transform: run_transform,
                paint,
                glyphs,
            } => encode_text_run(scene, transform, font, *size, *run_transform, paint, glyphs)?,
            Command::Layer { layer, children } => {
                validate_layer(layer)?;
                let layer_transform = transform * kurbo::Affine::from(layer.transform());
                if requires_vello_layer(layer) {
                    let clip = layer
                        .clip()
                        .cloned()
                        .unwrap_or_else(|| Shape::rect(Rect::new(-1.0e9, -1.0e9, 2.0e9, 2.0e9)));
                    encode_layer_start(scene, layer, layer_transform, &clip)?;
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
    shape: &Shape,
    paint: &Paint,
) -> Result<()> {
    validate_shape(shape)?;
    validate_paint(paint)?;
    let brush = paint_brush(paint)?;
    match shape.kind() {
        ShapeKind::Rect(rect) => scene.fill(
            peniko::Fill::NonZero,
            transform,
            &brush,
            None,
            &kurbo::Rect::from(*rect),
        ),
        ShapeKind::RoundedRect { rect, radii } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            &brush,
            None,
            &kurbo_rounded_rect(*rect, *radii),
        ),
        ShapeKind::Circle { center, radius } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            &brush,
            None,
            &kurbo::Circle::new((center.x(), center.y()), *radius),
        ),
        ShapeKind::Ellipse { center, radii } => scene.fill(
            peniko::Fill::NonZero,
            transform,
            &brush,
            None,
            &kurbo::Ellipse::new(
                (center.x(), center.y()),
                (radii.width(), radii.height()),
                0.0,
            ),
        ),
        ShapeKind::Path(path) => scene.fill(
            peniko::Fill::NonZero,
            transform,
            &brush,
            None,
            &path.to_kurbo(),
        ),
    }
    Ok(())
}

fn encode_stroke(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &Shape,
    stroke: Stroke,
    paint: &Paint,
) -> Result<()> {
    validate_shape(shape)?;
    validate_stroke(stroke)?;
    validate_paint(paint)?;
    let brush = paint_brush(paint)?;
    let (shape, stroke) = aligned_stroke_shape(shape, stroke)?;
    let vello_stroke = vello_stroke(stroke);
    match shape {
        AlignedShape::Rect(rect) => scene.stroke(&vello_stroke, transform, &brush, None, &rect),
        AlignedShape::RoundedRect(rect) => {
            scene.stroke(&vello_stroke, transform, &brush, None, &rect)
        }
        AlignedShape::Circle(circle) => {
            scene.stroke(&vello_stroke, transform, &brush, None, &circle)
        }
        AlignedShape::Ellipse(ellipse) => {
            scene.stroke(&vello_stroke, transform, &brush, None, &ellipse)
        }
        AlignedShape::Path(path) => scene.stroke(&vello_stroke, transform, &brush, None, &path),
    }
    Ok(())
}

enum AlignedShape {
    Rect(kurbo::Rect),
    RoundedRect(kurbo::RoundedRect),
    Circle(kurbo::Circle),
    Ellipse(kurbo::Ellipse),
    Path(kurbo::BezPath),
}

fn aligned_stroke_shape(shape: &Shape, mut stroke: Stroke) -> Result<(AlignedShape, Stroke)> {
    let align = stroke.align_kind();
    let width = stroke.width();
    let offset = match align {
        StrokeAlign::Center => 0.0,
        StrokeAlign::Inside => -width * 0.5,
        StrokeAlign::Outside => width * 0.5,
    };
    if align != StrokeAlign::Center {
        stroke = stroke.align(StrokeAlign::Center);
    }
    let shape = match shape.kind() {
        ShapeKind::Rect(rect) => {
            AlignedShape::Rect(kurbo::Rect::from(*rect).inflate(offset, offset))
        }
        ShapeKind::RoundedRect { rect, radii } => {
            let radii = offset_radii(*radii, offset);
            let rect = kurbo::Rect::from(*rect).inflate(offset, offset);
            AlignedShape::RoundedRect(kurbo_rounded_rect(
                Rect::new(rect.x0, rect.y0, rect.width(), rect.height()),
                radii,
            ))
        }
        ShapeKind::Circle { center, radius } => AlignedShape::Circle(kurbo::Circle::new(
            (center.x(), center.y()),
            (*radius + offset).max(0.0),
        )),
        ShapeKind::Ellipse { center, radii } => AlignedShape::Ellipse(kurbo::Ellipse::new(
            (center.x(), center.y()),
            (
                (radii.width() + offset).max(0.0),
                (radii.height() + offset).max(0.0),
            ),
            0.0,
        )),
        ShapeKind::Path(path) if offset == 0.0 => AlignedShape::Path(path.to_kurbo()),
        ShapeKind::Path(_) => {
            return Err(Error::new(
                ErrorCode::UnsupportedBackend,
                "inside and outside stroke alignment for arbitrary paths requires a shape layer",
            ));
        }
    };
    Ok((shape, stroke))
}

fn encode_shadow(
    scene: &mut vello::Scene,
    transform: kurbo::Affine,
    shape: &Shape,
    shadow: &Shadow,
) -> Result<()> {
    validate_shape(shape)?;
    validate_shadow(shadow)?;
    let color = paint_color(shadow.paint())?;
    let std_dev = shadow.blur() * 0.5;
    match shape.kind() {
        ShapeKind::Rect(rect) => {
            let rect = offset_rect(expand_rect(*rect, shadow.spread()), shadow.offset());
            scene.draw_blurred_rounded_rect(transform, rect.into(), color, 0.0, std_dev);
        }
        ShapeKind::RoundedRect { rect, radii } => {
            let rect = offset_rect(expand_rect(*rect, shadow.spread()), shadow.offset());
            if let Some(radius) = radii.uniform() {
                scene.draw_blurred_rounded_rect(
                    transform,
                    rect.into(),
                    color,
                    (radius + shadow.spread()).max(0.0),
                    std_dev,
                );
            } else {
                let color = solid_color(shadow.paint())?;
                encode_non_uniform_rounded_shadow(
                    scene,
                    transform,
                    rect,
                    offset_radii(*radii, shadow.spread()),
                    shadow.blur(),
                    color,
                )?;
            }
        }
        ShapeKind::Circle { center, radius } => {
            let radius = (radius + shadow.spread()).max(0.0);
            let rect = Rect::new(
                center.x() - radius + shadow.offset().x(),
                center.y() - radius + shadow.offset().y(),
                radius * 2.0,
                radius * 2.0,
            );
            scene.draw_blurred_rounded_rect(transform, rect.into(), color, radius, std_dev);
        }
        _ => {
            return Err(Error::new(
                ErrorCode::UnsupportedBackend,
                "only rectangle, rounded rectangle, and circle shadows lower to Vello in this milestone",
            ));
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
        return encode_fill(
            scene,
            transform,
            &Shape::rounded_rect(rect, radii),
            &Paint::color(color),
        );
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
    validate_rect(rect, "image target rectangle")?;
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
    validate_text_run(size, run_transform, glyphs)?;
    validate_paint(paint.fill())?;
    let Some(data) = font.data.as_ref().map(|data| &data.data) else {
        return Err(invalid_input(
            "text run font data is required for rendering prepared glyphs",
        ));
    };
    let brush = paint_brush(paint.fill())?;
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

fn encode_layer_start(
    scene: &mut vello::Scene,
    layer: &Layer,
    transform: kurbo::Affine,
    clip: &Shape,
) -> Result<()> {
    validate_layer(layer)?;
    if layer.filter().is_some() {
        return Err(Error::new(
            ErrorCode::UnsupportedBackend,
            "layer filters are not implemented yet",
        ));
    }
    if layer.mask().is_some() {
        return Err(Error::new(
            ErrorCode::UnsupportedBackend,
            "layer masks are not implemented yet",
        ));
    }
    let blend = vello_blend(layer.blend_mode());
    let alpha = layer.opacity().clamp(0.0, 1.0);
    let use_clip = layer.blend_mode() == BlendMode::Normal && (alpha - 1.0).abs() < f32::EPSILON;
    match clip.kind() {
        ShapeKind::Rect(rect) => push_vello_layer(
            scene,
            use_clip,
            blend,
            alpha,
            transform,
            &kurbo::Rect::from(*rect),
        ),
        ShapeKind::RoundedRect { rect, radii } => push_vello_layer(
            scene,
            use_clip,
            blend,
            alpha,
            transform,
            &kurbo_rounded_rect(*rect, *radii),
        ),
        ShapeKind::Circle { center, radius } => push_vello_layer(
            scene,
            use_clip,
            blend,
            alpha,
            transform,
            &kurbo::Circle::new((center.x(), center.y()), *radius),
        ),
        ShapeKind::Ellipse { center, radii } => push_vello_layer(
            scene,
            use_clip,
            blend,
            alpha,
            transform,
            &kurbo::Ellipse::new(
                (center.x(), center.y()),
                (radii.width(), radii.height()),
                0.0,
            ),
        ),
        ShapeKind::Path(path) => {
            push_vello_layer(scene, use_clip, blend, alpha, transform, &path.to_kurbo())
        }
    }
    Ok(())
}

pub(crate) fn requires_vello_layer(layer: &Layer) -> bool {
    layer.clip().is_some()
        || layer.mask().is_some()
        || layer.filter().is_some()
        || layer.blend_mode() != BlendMode::Normal
        || (layer.opacity() - 1.0).abs() > f32::EPSILON
}

fn push_vello_layer(
    scene: &mut vello::Scene,
    use_clip: bool,
    blend: peniko::BlendMode,
    alpha: f32,
    transform: kurbo::Affine,
    shape: &impl kurbo::Shape,
) {
    if use_clip {
        scene.push_clip_layer(peniko::Fill::NonZero, transform, shape);
    } else {
        scene.push_layer(peniko::Fill::NonZero, blend, alpha, transform, shape);
    }
}

fn kurbo_rounded_rect(rect: Rect, radii: Radii) -> kurbo::RoundedRect {
    kurbo::RoundedRect::from_rect(
        rect.into(),
        kurbo::RoundedRectRadii::new(
            radii.top_left(),
            radii.top_right(),
            radii.bottom_right(),
            radii.bottom_left(),
        ),
    )
}

fn paint_color(paint: &Paint) -> Result<peniko::Color> {
    Ok(solid_color(paint)?.into())
}

fn solid_color(paint: &Paint) -> Result<Color> {
    match paint.kind() {
        PaintKind::Color(color) => Ok(*color),
        PaintKind::Gradient(_) | PaintKind::Image(_) => Err(Error::new(
            ErrorCode::UnsupportedBackend,
            "this operation requires a solid color paint",
        )),
    }
}

fn paint_brush(paint: &Paint) -> Result<peniko::Brush> {
    match paint.kind() {
        PaintKind::Color(color) => Ok(peniko::Brush::Solid((*color).into())),
        PaintKind::Gradient(gradient) => Ok(peniko::Brush::Gradient(gradient.clone().into())),
        PaintKind::Image(image) => Ok(peniko::Brush::Image(image_brush(image))),
    }
}

fn vello_stroke(stroke: Stroke) -> kurbo::Stroke {
    let (width, join, start_cap, end_cap, miter_limit, dash, _) = stroke.parts();
    let mut vello = kurbo::Stroke::new(width)
        .with_join(join.into())
        .with_start_cap(start_cap.into())
        .with_end_cap(end_cap.into())
        .with_miter_limit(miter_limit);
    if let Some(dash) = dash {
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
