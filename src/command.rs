use super::{
    geometry::offset_radii,
    paint::PaintKind,
    scene::Command,
    shape::ShapeKind,
    stats::collect_render_stats,
    style::{ClipGeometryKind, FilterList, NormalizedClip},
    validation::*,
    *,
};
use kurbo::Shape as KurboShape;

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
pub(crate) struct RenderClip {
    geometry: RenderClipGeometry,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderClipGeometry {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path { path: Path, fill_rule: FillRule },
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
    pub(crate) clip: Option<RenderClip>,
    pub(crate) transform: Transform,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) mask: Option<RenderLayerMask>,
    pub(crate) backdrop: Option<Box<RenderBackdropCapture>>,
    pub(crate) isolation: LayerIsolation,
    pub(crate) pass_plan: LayerPassPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RenderLayerMask {
    alpha_mask: ImageBuffer,
}

impl RenderLayerMask {
    fn from_resolved(mask: &ResolvedLayerAlphaMask, capabilities: Capabilities) -> Result<Self> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaterializedAlphaMaskExecution,
        ))?;
        Ok(Self {
            alpha_mask: mask.alpha_mask().clone(),
        })
    }

    #[must_use]
    pub(crate) const fn alpha_mask(&self) -> &ImageBuffer {
        &self.alpha_mask
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderBackdropCapture {
    filters: FilterList,
    capture_bounds: OffscreenBounds,
    clip: Option<RenderClip>,
    source_commands: Vec<RenderCommand>,
}

impl RenderBackdropCapture {
    fn from_input(
        input: &BackdropFilterInput,
        source_commands: &[RenderCommand],
        capabilities: Capabilities,
    ) -> Result<Self> {
        input.ensure_supported_for_planning(capabilities)?;
        let capture_bounds = OffscreenBounds::try_new(input.capture_bounds().rect())?;
        let clip = input
            .clip()
            .map(|clip| RenderClip::from_input(clip, capabilities))
            .transpose()?;
        Ok(Self {
            filters: input.filters().clone(),
            capture_bounds,
            clip,
            source_commands: source_commands.to_vec(),
        })
    }

    #[must_use]
    pub(crate) const fn filters(&self) -> &FilterList {
        &self.filters
    }

    #[must_use]
    pub(crate) const fn capture_bounds(&self) -> OffscreenBounds {
        self.capture_bounds
    }

    #[must_use]
    pub(crate) const fn clip(&self) -> Option<&RenderClip> {
        self.clip.as_ref()
    }

    #[must_use]
    pub(crate) fn source_commands(&self) -> &[RenderCommand] {
        &self.source_commands
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerIsolation {
    None,
    ClipOnly,
    BackendLayer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerPassRequirement {
    None,
    ClipOnly,
    DirectVelloOpacity,
    DirectVelloBlend,
    DirectVelloOpacityBlend,
    BoundedBackdropCapture,
    OffscreenTexture(PrimitiveOperation),
    DiagnosticBoundary(PrimitiveOperation),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerPassKind {
    None,
    ClipOnly,
    DirectVelloLayer,
    OffscreenTexture,
    DiagnosticBoundary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OffscreenBounds {
    rect: Rect,
}

impl OffscreenBounds {
    pub(crate) fn try_new(rect: Rect) -> Result<Self> {
        validate_rect(rect, "offscreen layer bounds")?;
        Ok(Self { rect })
    }

    #[must_use]
    pub(crate) const fn rect(self) -> Rect {
        self.rect
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerPassPlan {
    requirement: LayerPassRequirement,
    kind: LayerPassKind,
    bounds: Option<OffscreenBounds>,
}

impl LayerPassPlan {
    fn new(
        requirement: LayerPassRequirement,
        kind: LayerPassKind,
        bounds: Option<OffscreenBounds>,
    ) -> Self {
        Self {
            requirement,
            kind,
            bounds,
        }
    }

    fn from_authored(
        layer: &Layer,
        clip: Option<&RenderClip>,
        children: &[RenderCommand],
        capabilities: Capabilities,
    ) -> Result<Self> {
        if let Some(unsupported) = unsupported_layer_effect(layer) {
            return Self::diagnostic_boundary(unsupported);
        }
        if let Some(backdrop) = layer.backdrop_filter() {
            let bounds = OffscreenBounds::try_new(backdrop.capture_bounds().rect())?;
            return Ok(Self::new(
                LayerPassRequirement::BoundedBackdropCapture,
                LayerPassKind::OffscreenTexture,
                Some(bounds),
            ));
        }
        let bounds = clip_bounds(clip).or_else(|| commands_bounds(children));

        let has_clip = clip.is_some();
        let has_resolved_mask = layer.resolved_alpha_mask().is_some();
        let opacity_delta = (layer.opacity() - 1.0).abs();
        let opacity_is_clip_identity = opacity_delta < f32::EPSILON;
        let has_opacity = opacity_delta > f32::EPSILON;
        let has_blend = layer.blend_mode() != BlendMode::Normal;

        if has_resolved_mask {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MaterializedAlphaMaskExecution,
            ))?;
            let bounds = bounds.ok_or_else(|| {
                Error::invalid_value(
                    "materialized masked layer bounds",
                    "unknown",
                    "must be explicit for resolved layer alpha masks",
                )
            })?;
            return Ok(Self::new(
                LayerPassRequirement::OffscreenTexture(
                    PrimitiveOperation::MaterializedAlphaMaskExecution,
                ),
                LayerPassKind::OffscreenTexture,
                Some(bounds),
            ));
        }

        if has_clip && opacity_is_clip_identity && !has_blend {
            return Ok(Self::new(
                LayerPassRequirement::ClipOnly,
                LayerPassKind::ClipOnly,
                bounds,
            ));
        }

        if has_clip || has_opacity || has_blend {
            let requirement = match (has_opacity, has_blend) {
                (true, true) => LayerPassRequirement::DirectVelloOpacityBlend,
                (true, false) => LayerPassRequirement::DirectVelloOpacity,
                (false, true) => LayerPassRequirement::DirectVelloBlend,
                (false, false) => LayerPassRequirement::DirectVelloOpacity,
            };
            if direct_vello_layer_supported(requirement, capabilities) {
                return Ok(Self::new(
                    requirement,
                    LayerPassKind::DirectVelloLayer,
                    bounds,
                ));
            }
            return Self::future_offscreen(
                PrimitiveOperation::OffscreenLayerRendering,
                bounds,
                capabilities,
            );
        }

        Ok(Self::new(
            LayerPassRequirement::None,
            LayerPassKind::None,
            bounds,
        ))
    }

    fn diagnostic_boundary(unsupported: UnsupportedPrimitive) -> Result<Self> {
        let _boundary = Self::new(
            LayerPassRequirement::DiagnosticBoundary(unsupported.operation()),
            LayerPassKind::DiagnosticBoundary,
            None,
        );
        Err(Error::unsupported_render_primitive(unsupported))
    }

    fn future_offscreen(
        operation: PrimitiveOperation,
        bounds: Option<OffscreenBounds>,
        capabilities: Capabilities,
    ) -> Result<Self> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            operation,
        ))?;
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::OffscreenLayerRendering,
        ))?;
        let bounds = bounds.ok_or_else(|| {
            Error::invalid_value(
                "offscreen layer bounds",
                "unknown",
                "must be explicit for offscreen texture passes",
            )
        })?;
        Ok(Self::new(
            LayerPassRequirement::OffscreenTexture(operation),
            LayerPassKind::OffscreenTexture,
            Some(bounds),
        ))
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn requirement(self) -> LayerPassRequirement {
        self.requirement
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn kind(self) -> LayerPassKind {
        self.kind
    }

    #[must_use]
    pub(crate) const fn bounds(self) -> Option<OffscreenBounds> {
        self.bounds
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn requires_offscreen_texture(self) -> bool {
        matches!(self.kind, LayerPassKind::OffscreenTexture)
    }

    #[must_use]
    const fn isolation(self) -> LayerIsolation {
        match self.kind {
            LayerPassKind::None => LayerIsolation::None,
            LayerPassKind::ClipOnly => LayerIsolation::ClipOnly,
            LayerPassKind::DirectVelloLayer
            | LayerPassKind::OffscreenTexture
            | LayerPassKind::DiagnosticBoundary => LayerIsolation::BackendLayer,
        }
    }
}

pub(crate) fn normalize_commands(
    commands: &[Command],
    capabilities: Capabilities,
) -> Result<Vec<RenderCommand>> {
    normalize_commands_in_context(commands, capabilities, 0)
}

fn normalize_commands_in_context(
    commands: &[Command],
    capabilities: Capabilities,
    layer_depth: usize,
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
            Command::TextShadowRun {
                size,
                transform,
                paint,
                glyphs,
                shadows,
                ..
            } => {
                validate_text_run(*size, *transform, glyphs)?;
                validate_paint(paint.fill())?;
                for shadow in shadows.shadows() {
                    validate_shadow(shadow)?;
                }
                return Err(unsupported_text_shadow_error(plan_text_shadow_run(
                    shadows,
                    capabilities,
                )));
            }
            Command::Layer { layer, children } => {
                validate_layer(layer)?;
                reject_unsupported_layer_effect(layer)?;
                if layer.backdrop_filter().is_some() && layer_depth > 0 {
                    return Err(nested_backdrop_capture_error());
                }
                if layer.backdrop_filter().is_some() && layer.transform() != Transform::identity() {
                    return Err(transformed_backdrop_capture_error());
                }
                let previous_siblings = normalized.clone();
                if layer.backdrop_filter().is_some()
                    && commands_contain_backdrop_capture(&previous_siblings)
                {
                    return Err(repeated_top_level_backdrop_capture_error());
                }
                let children =
                    normalize_commands_in_context(children, capabilities, layer_depth + 1)?;
                RenderCommand::Layer {
                    layer: NormalizedLayer::from_authored(
                        layer,
                        &children,
                        &previous_siblings,
                        capabilities,
                    )?,
                    children,
                }
            }
        });
    }
    Ok(normalized)
}

fn commands_contain_backdrop_capture(commands: &[RenderCommand]) -> bool {
    commands.iter().any(command_contains_backdrop_capture)
}

fn command_contains_backdrop_capture(command: &RenderCommand) -> bool {
    match command {
        RenderCommand::Layer { layer, children } => {
            layer.backdrop.is_some() || commands_contain_backdrop_capture(children)
        }
        RenderCommand::Fill { .. }
        | RenderCommand::Stroke { .. }
        | RenderCommand::Shadow { .. }
        | RenderCommand::Image { .. }
        | RenderCommand::TextRun { .. } => false,
    }
}

fn nested_backdrop_capture_error() -> Error {
    let mut error = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropExecution,
    ));
    error.message.push_str(
        ": nested backdrop capture crosses a layer isolation boundary and is not normalized in this task",
    );
    error
}

fn transformed_backdrop_capture_error() -> Error {
    let mut error = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropExecution,
    ));
    error.message.push_str(
        ": transformed backdrop capture requires coordinate-space reconciliation before materialized execution",
    );
    error
}

fn repeated_top_level_backdrop_capture_error() -> Error {
    let mut error = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropExecution,
    ));
    error.message.push_str(
        ": repeated top-level backdrop capture requires staged source reconciliation before materialized execution",
    );
    error
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextShadowPlan {
    SupportedZeroBlurSolid,
    UnsupportedZeroBlurSolid,
    UnsupportedGlyphAlphaCapture,
}

fn plan_text_shadow_run(shadows: &ShadowList, capabilities: Capabilities) -> TextShadowPlan {
    if is_zero_blur_solid_text_shadow_subset(shadows) {
        if capabilities.shadows().supports_text_shadows() {
            TextShadowPlan::SupportedZeroBlurSolid
        } else {
            TextShadowPlan::UnsupportedZeroBlurSolid
        }
    } else {
        TextShadowPlan::UnsupportedGlyphAlphaCapture
    }
}

fn is_zero_blur_solid_text_shadow_subset(shadows: &ShadowList) -> bool {
    shadows.shadows().iter().all(|shadow| {
        shadow.kind() == ShadowKind::Outer
            && shadow.blur() == 0.0
            && shadow.spread() == 0.0
            && matches!(shadow.paint().kind(), PaintKind::Color(_))
    })
}

fn unsupported_text_shadow_error(plan: TextShadowPlan) -> Error {
    let mut error = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::Shadows,
        PrimitiveOperation::TextShadow,
    ));
    match plan {
        TextShadowPlan::SupportedZeroBlurSolid | TextShadowPlan::UnsupportedZeroBlurSolid => {
            error.message.push_str(
                ": zero-blur solid text shadows could be represented as repeated shifted glyph draws behind text, but this renderer has not claimed or enabled that executable subset yet",
            );
        }
        TextShadowPlan::UnsupportedGlyphAlphaCapture => {
            error.message.push_str(
                ": text-shadow execution depends on glyph-alpha/offscreen text capture before blurred shadows can be composited behind text",
            );
        }
    }
    error
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

impl RenderClip {
    fn from_input(input: &ClipInput, capabilities: Capabilities) -> Result<Self> {
        let normalized = input.normalize(capabilities)?;
        Ok(Self::from_normalized(&normalized))
    }

    fn from_normalized(clip: &NormalizedClip) -> Self {
        Self {
            geometry: match clip.geometry().kind() {
                ClipGeometryKind::Rect(rect) => RenderClipGeometry::Rect(*rect),
                ClipGeometryKind::RoundedRect { rect, radii } => RenderClipGeometry::RoundedRect {
                    rect: *rect,
                    radii: *radii,
                },
                ClipGeometryKind::Circle { center, radius } => RenderClipGeometry::Circle {
                    center: *center,
                    radius: *radius,
                },
                ClipGeometryKind::Ellipse { center, radii } => RenderClipGeometry::Ellipse {
                    center: *center,
                    radii: *radii,
                },
                ClipGeometryKind::Path(path) => RenderClipGeometry::Path {
                    path: path.path().clone(),
                    fill_rule: path.fill_rule(),
                },
            },
            coordinate_space: clip.coordinate_space(),
        }
    }

    #[must_use]
    pub(crate) const fn geometry(&self) -> &RenderClipGeometry {
        &self.geometry
    }

    #[must_use]
    pub(crate) const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
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
        if shadow.kind() == ShadowKind::Inset {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::Shadows,
                PrimitiveOperation::InsetBoxShadow,
            ))?;
            unreachable!("inset shadow support requires clipped inner shadow lowering");
        }
        Ok(Self {
            offset: shadow.offset(),
            blur: shadow.blur(),
            spread: shadow.spread(),
            color: solid_shadow_color(shadow.paint(), capabilities)?,
        })
    }
}

impl NormalizedLayer {
    fn from_authored(
        layer: &Layer,
        children: &[RenderCommand],
        previous_siblings: &[RenderCommand],
        capabilities: Capabilities,
    ) -> Result<Self> {
        validate_layer(layer)?;
        let clip = layer
            .clip_input()
            .map(|clip| RenderClip::from_input(clip, capabilities))
            .transpose()?;
        let mask = layer
            .resolved_alpha_mask()
            .map(|mask| RenderLayerMask::from_resolved(mask, capabilities))
            .transpose()?;
        let backdrop = layer
            .backdrop_filter()
            .map(|backdrop| {
                RenderBackdropCapture::from_input(backdrop, previous_siblings, capabilities)
                    .map(Box::new)
            })
            .transpose()?;
        let pass_plan = LayerPassPlan::from_authored(layer, clip.as_ref(), children, capabilities)?;
        Ok(Self {
            clip,
            transform: layer.transform(),
            opacity: layer.opacity(),
            blend: layer.blend_mode(),
            mask,
            backdrop,
            isolation: pass_plan.isolation(),
            pass_plan,
        })
    }
}

fn direct_vello_layer_supported(
    requirement: LayerPassRequirement,
    capabilities: Capabilities,
) -> bool {
    match requirement {
        LayerPassRequirement::DirectVelloOpacity => capabilities
            .offscreen_pipeline()
            .supports_direct_vello_opacity_isolation(),
        LayerPassRequirement::DirectVelloBlend => capabilities
            .offscreen_pipeline()
            .supports_direct_vello_blend_isolation(),
        LayerPassRequirement::DirectVelloOpacityBlend => {
            capabilities
                .offscreen_pipeline()
                .supports_direct_vello_opacity_isolation()
                && capabilities
                    .offscreen_pipeline()
                    .supports_direct_vello_blend_isolation()
        }
        LayerPassRequirement::None
        | LayerPassRequirement::ClipOnly
        | LayerPassRequirement::BoundedBackdropCapture
        | LayerPassRequirement::OffscreenTexture(_)
        | LayerPassRequirement::DiagnosticBoundary(_) => false,
    }
}

fn reject_unsupported_layer_effect(layer: &Layer) -> Result<()> {
    if let Some(unsupported) = unsupported_layer_effect(layer) {
        return Err(Error::unsupported_render_primitive(unsupported));
    }
    Ok(())
}

fn unsupported_layer_effect(layer: &Layer) -> Option<UnsupportedPrimitive> {
    if layer.mask().is_some() {
        return Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ));
    }
    if layer.filter().is_some() {
        return Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ));
    }
    None
}

fn clip_bounds(clip: Option<&RenderClip>) -> Option<OffscreenBounds> {
    clip.and_then(render_clip_bounds)
}

fn commands_bounds(commands: &[RenderCommand]) -> Option<OffscreenBounds> {
    commands
        .iter()
        .filter_map(command_bounds)
        .reduce(union_bounds)
}

fn command_bounds(command: &RenderCommand) -> Option<OffscreenBounds> {
    match command {
        RenderCommand::Fill { shape, .. } => render_shape_bounds(shape),
        RenderCommand::Stroke { shape, stroke, .. } => stroke_shape_bounds(shape, stroke),
        RenderCommand::Shadow { shape, shadow } => shadow_bounds(shape, shadow),
        RenderCommand::Image { rect, .. } => OffscreenBounds::try_new(*rect).ok(),
        RenderCommand::TextRun { .. } => None,
        RenderCommand::Layer { layer, .. } => layer
            .pass_plan
            .bounds()
            .and_then(|bounds| transform_bounds(bounds.rect(), layer.transform)),
    }
}

fn render_shape_bounds(shape: &RenderShape) -> Option<OffscreenBounds> {
    match shape {
        RenderShape::Rect(rect) | RenderShape::RoundedRect { rect, .. } => {
            OffscreenBounds::try_new(*rect).ok()
        }
        RenderShape::Circle { center, radius } => OffscreenBounds::try_new(Rect::new(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        ))
        .ok(),
        RenderShape::Ellipse { center, radii } => OffscreenBounds::try_new(Rect::new(
            center.x() - radii.width(),
            center.y() - radii.height(),
            radii.width() * 2.0,
            radii.height() * 2.0,
        ))
        .ok(),
        RenderShape::Path(path) => kurbo_bounds(path.to_kurbo().bounding_box()),
    }
}

fn render_clip_bounds(clip: &RenderClip) -> Option<OffscreenBounds> {
    let bounds = match clip.geometry() {
        RenderClipGeometry::Rect(rect) | RenderClipGeometry::RoundedRect { rect, .. } => {
            OffscreenBounds::try_new(*rect).ok()
        }
        RenderClipGeometry::Circle { center, radius } => OffscreenBounds::try_new(Rect::new(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        ))
        .ok(),
        RenderClipGeometry::Ellipse { center, radii } => OffscreenBounds::try_new(Rect::new(
            center.x() - radii.width(),
            center.y() - radii.height(),
            radii.width() * 2.0,
            radii.height() * 2.0,
        ))
        .ok(),
        RenderClipGeometry::Path { path, .. } => kurbo_bounds(path.to_kurbo().bounding_box()),
    }?;
    match clip.coordinate_space() {
        Some(coordinate_space) => transform_bounds(bounds.rect(), coordinate_space.transform()),
        None => Some(bounds),
    }
}

fn stroke_shape_bounds(
    shape: &RenderStrokeShape,
    stroke: &RenderStroke,
) -> Option<OffscreenBounds> {
    let half_width = stroke.width * 0.5;
    let (bounds, inflate) = match shape {
        RenderStrokeShape::Rect(rect) => (*rect, half_width),
        RenderStrokeShape::RoundedRect(rect) => (rect.bounding_box(), half_width),
        RenderStrokeShape::Circle(circle) => (circle.bounding_box(), half_width),
        RenderStrokeShape::Ellipse(ellipse) => (ellipse.bounding_box(), half_width),
        RenderStrokeShape::Path(path) => (
            path.bounding_box(),
            half_width * stroke.miter_limit.max(1.0),
        ),
    };
    kurbo_bounds(bounds.inflate(inflate, inflate))
}

fn shadow_bounds(shape: &ShadowShape, shadow: &RenderShadow) -> Option<OffscreenBounds> {
    let base = match shape {
        ShadowShape::Rect(rect) | ShadowShape::RoundedRect { rect, .. } => *rect,
        ShadowShape::Circle { center, radius } => Rect::new(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        ),
    };
    let offset = Rect::new(
        base.x() + shadow.offset.x(),
        base.y() + shadow.offset.y(),
        base.width(),
        base.height(),
    );
    let blur = FilterBlur::try_new(shadow.blur).ok()?;
    let blur_support = BlurPolicy::vello_outer_shadow_compatibility()
        .support_radius(blur)
        .ok()?;
    let support = shadow.spread + blur_support;
    OffscreenBounds::try_new(geometry::expand_rect(offset, support)).ok()
}

fn transform_bounds(rect: Rect, transform: Transform) -> Option<OffscreenBounds> {
    let [a, b, c, d, e, f] = transform.as_array();
    let max = rect.max();
    let corners = [
        (rect.x(), rect.y()),
        (max.x(), rect.y()),
        (max.x(), max.y()),
        (rect.x(), max.y()),
    ];
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let transformed_x = a * x + c * y + e;
        let transformed_y = b * x + d * y + f;
        if !transformed_x.is_finite() || !transformed_y.is_finite() {
            return None;
        }
        min_x = min_x.min(transformed_x);
        min_y = min_y.min(transformed_y);
        max_x = max_x.max(transformed_x);
        max_y = max_y.max(transformed_y);
    }
    OffscreenBounds::try_new(Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)).ok()
}

fn union_bounds(a: OffscreenBounds, b: OffscreenBounds) -> OffscreenBounds {
    let a = a.rect();
    let b = b.rect();
    let a_max = a.max();
    let b_max = b.max();
    let min_x = a.x().min(b.x());
    let min_y = a.y().min(b.y());
    let max_x = a_max.x().max(b_max.x());
    let max_y = a_max.y().max(b_max.y());
    OffscreenBounds::try_new(Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
        .expect("union of valid finite bounds remains valid")
}

fn kurbo_bounds(bounds: kurbo::Rect) -> Option<OffscreenBounds> {
    if !bounds.x0.is_finite()
        || !bounds.y0.is_finite()
        || !bounds.x1.is_finite()
        || !bounds.y1.is_finite()
    {
        return None;
    }
    OffscreenBounds::try_new(Rect::new(
        bounds.x0,
        bounds.y0,
        bounds.width(),
        bounds.height(),
    ))
    .ok()
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
