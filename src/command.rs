use super::{
    geometry::offset_radii, paint::PaintKind, scene::Command, shape::ShapeKind,
    stats::collect_render_stats, validation::*, *,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderCommands {
    pub(crate) commands: Vec<RenderCommand>,
}

impl RenderCommands {
    #[must_use]
    pub(crate) fn new(commands: Vec<RenderCommand>) -> Self {
        Self { commands }
    }

    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        let mut uploaded_images = std::collections::HashSet::new();
        collect_render_stats(&self.commands, &mut stats, &mut uploaded_images);
        stats
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderCommand {
    Fill {
        shape: RenderShape,
        paint: RenderPaint,
    },
    Stroke {
        shape: RenderStrokeShape,
        stroke: RenderStroke,
        paint: RenderPaint,
    },
    Shadow {
        shape: ShadowShape,
        shadow: RenderShadow,
    },
    Image {
        image: Image,
        rect: Rect,
        fit: ImageFit,
    },
    TextRun {
        font: FontRef<'static>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: Vec<TextGlyph>,
    },
    Layer {
        layer: NormalizedLayer,
        children: Vec<RenderCommand>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderShape {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(Path),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderStrokeShape {
    Rect(kurbo::Rect),
    RoundedRect(kurbo::RoundedRect),
    Circle(kurbo::Circle),
    Ellipse(kurbo::Ellipse),
    Path(kurbo::BezPath),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderStroke {
    pub(crate) width: f64,
    pub(crate) join: LineJoin,
    pub(crate) start_cap: LineCap,
    pub(crate) end_cap: LineCap,
    pub(crate) miter_limit: f64,
    pub(crate) dash: Option<Dash>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderPaint {
    Color(Color),
    Gradient(Gradient),
    Image(Image),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShadowShape {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderShadow {
    pub(crate) offset: Point,
    pub(crate) blur: f64,
    pub(crate) spread: f64,
    pub(crate) color: Color,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NormalizedLayer {
    pub(crate) clip: Option<RenderShape>,
    pub(crate) transform: Transform,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) isolation: LayerIsolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerIsolation {
    None,
    ClipOnly,
    BackendLayer,
}

pub(crate) fn normalize_commands(
    commands: &[Command],
    capabilities: Capabilities,
) -> Result<Vec<RenderCommand>> {
    let mut normalized = Vec::with_capacity(commands.len());
    for command in commands {
        normalized.push(match command {
            Command::Fill { shape, paint } => RenderCommand::Fill {
                shape: RenderShape::try_from(shape.clone())?,
                paint: RenderPaint::try_from(paint.clone())?,
            },
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => RenderCommand::Stroke {
                shape: RenderStrokeShape::from_authored(shape, *stroke, capabilities)?,
                stroke: RenderStroke::try_from(*stroke)?,
                paint: RenderPaint::try_from(paint.clone())?,
            },
            Command::Shadow { shape, shadow } => RenderCommand::Shadow {
                shape: ShadowShape::from_authored(shape.clone(), capabilities)?,
                shadow: RenderShadow::from_authored(shadow.clone(), capabilities)?,
            },
            Command::Image { image, rect, fit } => {
                validate_rect(*rect, "image target rectangle")?;
                RenderCommand::Image {
                    image: image.clone(),
                    rect: *rect,
                    fit: *fit,
                }
            }
            Command::TextRun {
                font,
                size,
                transform,
                paint,
                glyphs,
            } => {
                validate_text_run(*size, *transform, glyphs)?;
                validate_paint(paint.fill())?;
                RenderCommand::TextRun {
                    font: font.clone(),
                    size: *size,
                    transform: *transform,
                    paint: paint.clone(),
                    glyphs: glyphs.clone(),
                }
            }
            Command::Layer { layer, children } => RenderCommand::Layer {
                layer: NormalizedLayer::from_authored(layer, capabilities)?,
                children: normalize_commands(children, capabilities)?,
            },
        });
    }
    Ok(normalized)
}

impl TryFrom<Shape> for RenderShape {
    type Error = Error;

    fn try_from(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(match shape.kind() {
            ShapeKind::Rect(rect) => Self::Rect(*rect),
            ShapeKind::RoundedRect { rect, radii } => Self::RoundedRect {
                rect: *rect,
                radii: *radii,
            },
            ShapeKind::Circle { center, radius } => Self::Circle {
                center: *center,
                radius: *radius,
            },
            ShapeKind::Ellipse { center, radii } => Self::Ellipse {
                center: *center,
                radii: *radii,
            },
            ShapeKind::Path(path) => Self::Path(path.clone()),
        })
    }
}

impl RenderStrokeShape {
    fn from_authored(shape: &Shape, stroke: Stroke, capabilities: Capabilities) -> Result<Self> {
        validate_shape(shape)?;
        validate_stroke(stroke)?;

        let align = stroke.align_kind();
        let width = stroke.width();
        let offset = match align {
            StrokeAlign::Center => 0.0,
            StrokeAlign::Inside => -width * 0.5,
            StrokeAlign::Outside => width * 0.5,
        };

        Ok(match shape.kind() {
            ShapeKind::Rect(rect) => Self::Rect(kurbo::Rect::from(*rect).inflate(offset, offset)),
            ShapeKind::RoundedRect { rect, radii } => {
                let radii = offset_radii(*radii, offset);
                let rect = kurbo::Rect::from(*rect).inflate(offset, offset);
                Self::RoundedRect(kurbo_rounded_rect(
                    Rect::new(rect.x0, rect.y0, rect.width(), rect.height()),
                    radii,
                ))
            }
            ShapeKind::Circle { center, radius } => Self::Circle(kurbo::Circle::new(
                (center.x(), center.y()),
                (*radius + offset).max(0.0),
            )),
            ShapeKind::Ellipse { center, radii } => Self::Ellipse(kurbo::Ellipse::new(
                (center.x(), center.y()),
                (
                    (radii.width() + offset).max(0.0),
                    (radii.height() + offset).max(0.0),
                ),
                0.0,
            )),
            ShapeKind::Path(path) if align == StrokeAlign::Center => Self::Path(path.to_kurbo()),
            ShapeKind::Path(_) => {
                capabilities.ensure_supported(UnsupportedPrimitive::new(
                    PrimitiveFamily::GeometryTargets,
                    PrimitiveOperation::InsideOutsidePathStrokeAlignment,
                ))?;
                unreachable!("path stroke alignment support requires path offset lowering");
            }
        })
    }
}

impl TryFrom<Stroke> for RenderStroke {
    type Error = Error;

    fn try_from(stroke: Stroke) -> Result<Self> {
        validate_stroke(stroke)?;
        Ok(Self {
            width: stroke.width(),
            join: stroke.join_kind(),
            start_cap: stroke.start_cap(),
            end_cap: stroke.end_cap(),
            miter_limit: stroke.miter_limit(),
            dash: stroke.dash(),
        })
    }
}

impl TryFrom<Paint> for RenderPaint {
    type Error = Error;

    fn try_from(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(match paint.kind() {
            PaintKind::Color(color) => Self::Color(*color),
            PaintKind::Gradient(gradient) => Self::Gradient(gradient.clone()),
            PaintKind::Image(image) => Self::Image(image.clone()),
        })
    }
}

impl ShadowShape {
    fn from_authored(shape: Shape, capabilities: Capabilities) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(match shape.kind() {
            ShapeKind::Rect(rect) => Self::Rect(*rect),
            ShapeKind::RoundedRect { rect, radii } => Self::RoundedRect {
                rect: *rect,
                radii: *radii,
            },
            ShapeKind::Circle { center, radius } => Self::Circle {
                center: *center,
                radius: *radius,
            },
            ShapeKind::Ellipse { .. } | ShapeKind::Path(_) => {
                capabilities.ensure_supported(UnsupportedPrimitive::new(
                    PrimitiveFamily::Shadows,
                    PrimitiveOperation::EllipsePathShadowShape,
                ))?;
                unreachable!("ellipse/path shadow support requires shadow geometry lowering");
            }
        })
    }
}

impl RenderShadow {
    fn from_authored(shadow: Shadow, capabilities: Capabilities) -> Result<Self> {
        validate_shadow(&shadow)?;
        Ok(Self {
            offset: shadow.offset(),
            blur: shadow.blur(),
            spread: shadow.spread(),
            color: solid_shadow_color(shadow.paint(), capabilities)?,
        })
    }
}

impl NormalizedLayer {
    fn from_authored(layer: &Layer, capabilities: Capabilities) -> Result<Self> {
        validate_layer(layer)?;
        if layer.mask().is_some() {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ))?;
        }
        if layer.filter().is_some() {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::LayerFilter,
            ))?;
        }
        let isolation = if layer.clip().is_some()
            && layer.blend_mode() == BlendMode::Normal
            && (layer.opacity() - 1.0).abs() < f32::EPSILON
        {
            LayerIsolation::ClipOnly
        } else if layer.clip().is_some()
            || layer.blend_mode() != BlendMode::Normal
            || (layer.opacity() - 1.0).abs() > f32::EPSILON
        {
            LayerIsolation::BackendLayer
        } else {
            LayerIsolation::None
        };
        Ok(Self {
            clip: layer
                .clip()
                .cloned()
                .map(RenderShape::try_from)
                .transpose()?,
            transform: layer.transform(),
            opacity: layer.opacity(),
            blend: layer.blend_mode(),
            isolation,
        })
    }
}

fn solid_shadow_color(paint: &Paint, capabilities: Capabilities) -> Result<Color> {
    match paint.kind() {
        PaintKind::Color(color) => Ok(*color),
        PaintKind::Gradient(_) | PaintKind::Image(_) => {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::PaintSources,
                PrimitiveOperation::NonSolidShadowPaint,
            ))?;
            unreachable!("non-solid shadow paint support requires shadow paint lowering");
        }
    }
}

pub(crate) fn kurbo_rounded_rect(rect: Rect, radii: Radii) -> kurbo::RoundedRect {
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
