use super::*;
use super::{
    paint::{GradientKind, PaintKind},
    shape::ShapeKind,
};

pub(crate) fn validate_surface_options(options: SurfaceOptions) -> Result<()> {
    validate_size(options.size, "surface size")?;
    validate_positive_f64(options.scale, "surface scale")
}

pub(crate) fn validate_shape(shape: &Shape) -> Result<()> {
    match shape.kind() {
        ShapeKind::Rect(rect) => validate_rect(*rect, "rectangle"),
        ShapeKind::RoundedRect { rect, radii } => {
            validate_rect(*rect, "rounded rectangle")?;
            validate_radii(*radii, "rounded rectangle radii")
        }
        ShapeKind::Circle { center, radius } => {
            validate_point(*center, "circle center")?;
            validate_non_negative_f64(*radius, "circle radius")
        }
        ShapeKind::Ellipse { center, radii } => {
            validate_point(*center, "ellipse center")?;
            validate_size(*radii, "ellipse radii")
        }
        ShapeKind::Path(path) => validate_path(path),
    }
}

pub(crate) fn validate_path(path: &Path) -> Result<()> {
    for element in path.elements() {
        match *element {
            PathElement::MoveTo(point) | PathElement::LineTo(point) => {
                validate_point(point, "path point")?;
            }
            PathElement::QuadTo(control, point) => {
                validate_point(control, "path control point")?;
                validate_point(point, "path point")?;
            }
            PathElement::CubicTo(a, b, point) => {
                validate_point(a, "path control point")?;
                validate_point(b, "path control point")?;
                validate_point(point, "path point")?;
            }
            PathElement::Close => {}
        }
    }
    Ok(())
}

pub(crate) fn validate_stroke(stroke: Stroke) -> Result<()> {
    validate_positive_f64(stroke.width(), "stroke width")?;
    validate_positive_f64(stroke.miter_limit(), "stroke miter limit")?;
    if let Some(dash) = stroke.dash() {
        validate_dash(dash)?;
    }
    Ok(())
}

pub(crate) fn validate_dash(dash: Dash) -> Result<()> {
    validate_finite_f64(dash.offset(), "dash offset")?;
    for interval in dash.intervals() {
        validate_non_negative_f64(*interval, "dash interval")?;
    }
    Ok(())
}

pub(crate) fn validate_paint(paint: &Paint) -> Result<()> {
    match paint.kind() {
        PaintKind::Color(color) => validate_color(*color, "color paint"),
        PaintKind::Gradient(gradient) => validate_gradient(gradient),
        PaintKind::Image(image) => validate_size(image.size(), "image size"),
    }
}

pub(crate) fn validate_gradient(gradient: &Gradient) -> Result<()> {
    match gradient.kind() {
        GradientKind::Linear { start, end, stops } => {
            validate_point(*start, "linear gradient start")?;
            validate_point(*end, "linear gradient end")?;
            validate_gradient_stops(stops)
        }
        GradientKind::Radial {
            center,
            radius,
            stops,
        } => {
            validate_point(*center, "radial gradient center")?;
            validate_positive_f64(*radius, "radial gradient radius")?;
            validate_gradient_stops(stops)
        }
        GradientKind::Sweep { center, stops } => {
            validate_point(*center, "sweep gradient center")?;
            validate_gradient_stops(stops)
        }
    }
}

pub(crate) fn validate_gradient_stops(stops: &[GradientStop]) -> Result<()> {
    if stops.is_empty() {
        let mut error = Error::invalid_value("gradient stops", "[]", "must not be empty");
        error.message = "gradient stops must not be empty".to_owned();
        return Err(error);
    }
    for stop in stops {
        if !stop.offset().is_finite() || !(0.0..=1.0).contains(&stop.offset()) {
            return Err(Error::invalid_value(
                "gradient stop offset",
                stop.offset(),
                "must be finite and between 0 and 1",
            ));
        }
        validate_color(stop.color(), "gradient stop color")?;
    }
    Ok(())
}

pub(crate) fn validate_shadow(shadow: &Shadow) -> Result<()> {
    validate_point(shadow.offset(), "shadow offset")?;
    validate_non_negative_f64(shadow.blur(), "shadow blur")?;
    validate_finite_f64(shadow.spread(), "shadow spread")?;
    validate_paint(shadow.paint())
}

pub(crate) fn validate_layer(layer: &Layer) -> Result<()> {
    validate_transform(layer.transform(), "layer transform")?;
    if !layer.opacity().is_finite() {
        return Err(invalid_input("layer opacity must be finite"));
    }
    if let Some(clip) = layer.clip_input() {
        super::style::validate_clip_input(clip)?;
    }
    if let Some(mask) = layer.mask() {
        validate_shape(mask)?;
    }
    if let Some(mask) = layer.resolved_alpha_mask() {
        super::image::validate_image_buffer_rgba_len(
            mask.alpha_mask().size,
            mask.alpha_mask().rgba.len(),
        )?;
    }
    if let Some(filter) = layer.filter() {
        validate_filter(filter)?;
    }
    if let Some(backdrop_filter) = layer.backdrop_filter() {
        backdrop_filter.ensure_supported_for_planning(Capabilities::VELLO_0_9)?;
    }
    Ok(())
}

pub(crate) fn validate_filter(filter: Filter) -> Result<()> {
    validate_non_negative_f64(filter.blur_radius(), "layer blur radius")
}

pub(crate) fn validate_text_run(
    size: f32,
    transform: Transform,
    glyphs: &[TextGlyph],
) -> Result<()> {
    if !size.is_finite() || size <= 0.0 {
        return Err(invalid_input(
            "text run size must be finite and greater than 0",
        ));
    }
    validate_transform(transform, "text transform")?;
    for glyph in glyphs {
        if !glyph.x().is_finite() || !glyph.y().is_finite() || !glyph.advance().is_finite() {
            return Err(invalid_input(
                "text glyph positions and advances must be finite",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_rect(rect: Rect, name: &str) -> Result<()> {
    validate_point(rect.origin(), name)?;
    validate_size(rect.size(), name)
}

pub(crate) fn validate_size(size: Size, name: &str) -> Result<()> {
    validate_non_negative_f64(size.width(), &format!("{name} width"))?;
    validate_non_negative_f64(size.height(), &format!("{name} height"))
}

pub(crate) fn validate_radii(radii: Radii, name: &str) -> Result<()> {
    validate_non_negative_f64(radii.top_left(), &format!("{name} top-left"))?;
    validate_non_negative_f64(radii.top_right(), &format!("{name} top-right"))?;
    validate_non_negative_f64(radii.bottom_right(), &format!("{name} bottom-right"))?;
    validate_non_negative_f64(radii.bottom_left(), &format!("{name} bottom-left"))
}

pub(crate) fn validate_point(point: Point, name: &str) -> Result<()> {
    validate_finite_f64(point.x(), &format!("{name} x"))?;
    validate_finite_f64(point.y(), &format!("{name} y"))
}

pub(crate) fn validate_transform(transform: Transform, name: &str) -> Result<()> {
    for value in transform.as_array() {
        validate_finite_f64(value, name)?;
    }
    Ok(())
}

pub(crate) fn validate_color(color: Color, name: &str) -> Result<()> {
    for (channel, value) in [
        ("red", color.r()),
        ("green", color.g()),
        ("blue", color.b()),
        ("alpha", color.a()),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Error::invalid_value(
                format!("{name} {channel} channel"),
                value,
                "must be finite and between 0 and 1",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_positive_f64(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and greater than 0",
        ));
    }
    Ok(())
}

pub(crate) fn validate_non_negative_f64(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

pub(crate) fn validate_finite_f64(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_value(name, value, "must be finite"));
    }
    Ok(())
}

pub(crate) fn invalid_input(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidInput, message)
}
