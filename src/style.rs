use super::{
    Capabilities, Color, CoordinateSpaceTag, Error, Image, ImageColorProfilePolicy, ImageId,
    ImageOrientationPolicy, Paint, PrimitiveFamily, PrimitiveOperation, Rect, Result, Shadow,
    Shape, Size, SymbolicColorPolicy, UnresolvedResource, UnresolvedResourceKind,
    UnsupportedPrimitive,
    validation::{
        validate_finite_f64, validate_non_negative_f64, validate_paint, validate_shape,
        validate_size,
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StyleColor {
    color: Color,
}

impl StyleColor {
    #[must_use]
    pub const fn new(color: Color) -> Self {
        Self { color }
    }

    #[must_use]
    pub const fn color(self) -> Color {
        self.color
    }

    #[must_use]
    pub const fn symbolic_policy() -> SymbolicColorPolicy {
        SymbolicColorPolicy::RootResolvedOnly
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StyleResourceRef {
    identifier: String,
}

impl StyleResourceRef {
    pub fn try_new(identifier: impl Into<String>) -> Result<Self> {
        let identifier = identifier.into();
        if identifier.trim().is_empty() {
            return Err(Error::invalid_value(
                "style resource reference",
                identifier,
                "must not be empty",
            ));
        }
        Ok(Self { identifier })
    }

    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedImageResource {
    id: ImageId,
    intrinsic_size: Size,
    density: Option<ImageResourceDensity>,
}

impl ResolvedImageResource {
    pub fn try_new(id: ImageId, intrinsic_size: Size) -> Result<Self> {
        validate_size(intrinsic_size, "resolved image intrinsic size")?;
        Ok(Self {
            id,
            intrinsic_size,
            density: None,
        })
    }

    #[must_use]
    pub const fn id(&self) -> ImageId {
        self.id
    }

    #[must_use]
    pub const fn intrinsic_size(&self) -> Size {
        self.intrinsic_size
    }

    #[must_use]
    pub const fn with_density(mut self, density: ImageResourceDensity) -> Self {
        self.density = Some(density);
        self
    }

    #[must_use]
    pub const fn density(&self) -> Option<ImageResourceDensity> {
        self.density
    }

    #[must_use]
    pub const fn orientation_policy(&self) -> ImageOrientationPolicy {
        ImageOrientationPolicy::RootResolvedOnly
    }

    #[must_use]
    pub const fn color_profile_policy(&self) -> ImageColorProfilePolicy {
        ImageColorProfilePolicy::RootResolvedOnly
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageResourceDensity {
    value: f64,
}

impl ImageResourceDensity {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_value(
                "image resource density",
                value,
                "must be finite and positive",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StyleImageSource {
    kind: StyleImageSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StyleImageSourceKind {
    Image(Image),
    Resolved(ResolvedImageResource),
    Paint(Paint),
    Unresolved(StyleResourceRef),
}

impl StyleImageSource {
    pub fn image(image: Image) -> Result<Self> {
        validate_size(image.size(), "image size")?;
        Ok(Self {
            kind: StyleImageSourceKind::Image(image),
        })
    }

    pub fn paint(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(Self {
            kind: StyleImageSourceKind::Paint(paint),
        })
    }

    #[must_use]
    pub const fn resolved(resource: ResolvedImageResource) -> Self {
        Self {
            kind: StyleImageSourceKind::Resolved(resource),
        }
    }

    #[must_use]
    pub fn unresolved(reference: StyleResourceRef) -> Self {
        Self {
            kind: StyleImageSourceKind::Unresolved(reference),
        }
    }

    pub fn require_resolved(&self) -> Result<()> {
        if let StyleImageSourceKind::Unresolved(reference) = &self.kind {
            return Err(Error::unresolved_resource(UnresolvedResource::new(
                UnresolvedResourceKind::Image,
                reference.identifier(),
            )));
        }
        Ok(())
    }

    #[must_use]
    pub const fn kind(&self) -> &StyleImageSourceKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StyleImageLayer {
    source: StyleImageSource,
    position: BackgroundPosition,
    size: BackgroundSize,
    repeat: BackgroundRepeat,
    origin: BackgroundBox,
    clip: BackgroundBox,
    attachment: BackgroundAttachment,
    coordinate_space: Option<CoordinateSpaceTag>,
}

impl StyleImageLayer {
    pub fn try_new(source: StyleImageSource) -> Result<Self> {
        Ok(Self {
            source,
            position: BackgroundPosition::default(),
            size: BackgroundSize::auto(),
            repeat: BackgroundRepeat::repeat(),
            origin: BackgroundBox::Padding,
            clip: BackgroundBox::Border,
            attachment: BackgroundAttachment::Scroll,
            coordinate_space: None,
        })
    }

    #[must_use]
    pub fn with_position(mut self, position: BackgroundPosition) -> Self {
        self.position = position;
        self
    }

    #[must_use]
    pub fn with_size(mut self, size: BackgroundSize) -> Self {
        self.size = size;
        self
    }

    #[must_use]
    pub fn with_repeat(mut self, repeat: BackgroundRepeat) -> Self {
        self.repeat = repeat;
        self
    }

    #[must_use]
    pub fn with_origin(mut self, origin: BackgroundBox) -> Self {
        self.origin = origin;
        self
    }

    #[must_use]
    pub fn with_clip(mut self, clip: BackgroundBox) -> Self {
        self.clip = clip;
        self
    }

    #[must_use]
    pub fn with_attachment(mut self, attachment: BackgroundAttachment) -> Self {
        self.attachment = attachment;
        self
    }

    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    #[must_use]
    pub const fn source(&self) -> &StyleImageSource {
        &self.source
    }

    #[must_use]
    pub const fn position(&self) -> BackgroundPosition {
        self.position
    }

    #[must_use]
    pub const fn size(&self) -> BackgroundSize {
        self.size
    }

    #[must_use]
    pub const fn repeat(&self) -> BackgroundRepeat {
        self.repeat
    }

    #[must_use]
    pub const fn origin(&self) -> BackgroundBox {
        self.origin
    }

    #[must_use]
    pub const fn clip(&self) -> BackgroundBox {
        self.clip
    }

    #[must_use]
    pub const fn attachment(&self) -> BackgroundAttachment {
        self.attachment
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundPosition {
    x: PositionComponent,
    y: PositionComponent,
}

impl BackgroundPosition {
    pub fn percent(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_percent_for(x, "background position x percent")?,
            y: PositionComponent::try_percent_for(y, "background position y percent")?,
        })
    }

    pub fn length(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_length_for(x, "background position x length")?,
            y: PositionComponent::try_length_for(y, "background position y length")?,
        })
    }

    pub fn components(x: PositionComponent, y: PositionComponent) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn edge_offsets(x: PositionEdgeOffset, y: PositionEdgeOffset) -> Self {
        Self {
            x: PositionComponent::edge_offset(x),
            y: PositionComponent::edge_offset(y),
        }
    }

    #[must_use]
    pub const fn x(self) -> PositionComponent {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> PositionComponent {
        self.y
    }
}

impl Default for BackgroundPosition {
    fn default() -> Self {
        Self {
            x: PositionComponent::percent_unchecked(0.0),
            y: PositionComponent::percent_unchecked(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionComponent {
    kind: PositionComponentKind,
    value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionComponentKind {
    Length,
    Percent,
    EdgeOffset(PositionEdge),
}

impl PositionComponent {
    pub fn try_percent(value: f64) -> Result<Self> {
        Self::try_percent_for(value, "background position percent")
    }

    pub fn try_length(value: f64) -> Result<Self> {
        Self::try_length_for(value, "background position length")
    }

    fn try_percent_for(value: f64, field: &str) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid_value(field, value, "must be finite"));
        }
        Ok(Self::percent_unchecked(value))
    }

    fn try_length_for(value: f64, field: &str) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid_value(field, value, "must be finite"));
        }
        Ok(Self {
            kind: PositionComponentKind::Length,
            value,
        })
    }

    const fn percent_unchecked(value: f64) -> Self {
        Self {
            kind: PositionComponentKind::Percent,
            value,
        }
    }

    #[must_use]
    pub const fn edge_offset(offset: PositionEdgeOffset) -> Self {
        Self {
            kind: PositionComponentKind::EdgeOffset(offset.edge()),
            value: offset.offset(),
        }
    }

    #[must_use]
    pub const fn kind(self) -> PositionComponentKind {
        self.kind
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionEdge {
    Start,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionEdgeOffset {
    edge: PositionEdge,
    offset: f64,
}

impl PositionEdgeOffset {
    pub fn start(offset: f64) -> Result<Self> {
        Self::try_new(PositionEdge::Start, offset)
    }

    pub fn end(offset: f64) -> Result<Self> {
        Self::try_new(PositionEdge::End, offset)
    }

    fn try_new(edge: PositionEdge, offset: f64) -> Result<Self> {
        if !offset.is_finite() {
            return Err(Error::invalid_value(
                "background position edge offset",
                offset,
                "must be finite",
            ));
        }
        Ok(Self { edge, offset })
    }

    #[must_use]
    pub const fn edge(self) -> PositionEdge {
        self.edge
    }

    #[must_use]
    pub const fn offset(self) -> f64 {
        self.offset
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundSize {
    kind: BackgroundSizeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BackgroundSizeKind {
    Auto,
    Cover,
    Contain,
    Explicit {
        width: SizeComponent,
        height: SizeComponent,
    },
}

impl BackgroundSize {
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: BackgroundSizeKind::Auto,
        }
    }

    #[must_use]
    pub const fn cover() -> Self {
        Self {
            kind: BackgroundSizeKind::Cover,
        }
    }

    #[must_use]
    pub const fn contain() -> Self {
        Self {
            kind: BackgroundSizeKind::Contain,
        }
    }

    #[must_use]
    pub const fn explicit(width: SizeComponent, height: SizeComponent) -> Self {
        Self {
            kind: BackgroundSizeKind::Explicit { width, height },
        }
    }

    #[must_use]
    pub const fn kind(self) -> BackgroundSizeKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeComponent {
    kind: SizeComponentKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizeComponentKind {
    Auto,
    Length(f64),
    Percent(f64),
}

impl SizeComponent {
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: SizeComponentKind::Auto,
        }
    }

    pub fn try_length(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "background size length",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: SizeComponentKind::Length(value),
        })
    }

    pub fn try_percent(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "background size percent",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: SizeComponentKind::Percent(value),
        })
    }

    #[must_use]
    pub const fn kind(self) -> SizeComponentKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePlacementInput {
    paint_rect: Rect,
    intrinsic_size: Size,
    position: BackgroundPosition,
    size: BackgroundSize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedImagePlacement {
    paint_rect: Rect,
    tile_rect: Rect,
}

impl ImagePlacementInput {
    pub fn try_new(
        paint_rect: Rect,
        intrinsic_size: Size,
        position: BackgroundPosition,
        size: BackgroundSize,
    ) -> Result<Self> {
        validate_placement_rect(paint_rect, "image placement paint rect")?;
        validate_positive_size(intrinsic_size, "image placement intrinsic size")?;
        Ok(Self {
            paint_rect,
            intrinsic_size,
            position,
            size,
        })
    }

    #[must_use]
    pub const fn paint_rect(self) -> Rect {
        self.paint_rect
    }

    #[must_use]
    pub const fn intrinsic_size(self) -> Size {
        self.intrinsic_size
    }

    #[must_use]
    pub const fn position(self) -> BackgroundPosition {
        self.position
    }

    #[must_use]
    pub const fn size(self) -> BackgroundSize {
        self.size
    }

    pub fn resolve(self) -> Result<ResolvedImagePlacement> {
        let tile_size = resolve_background_size(self.paint_rect, self.intrinsic_size, self.size);
        let tile_rect = Rect::new(
            resolve_position_component(
                self.paint_rect.x(),
                self.paint_rect.width(),
                tile_size.width(),
                self.position.x(),
            ),
            resolve_position_component(
                self.paint_rect.y(),
                self.paint_rect.height(),
                tile_size.height(),
                self.position.y(),
            ),
            tile_size.width(),
            tile_size.height(),
        );
        ResolvedImagePlacement::from_parts(self.paint_rect, tile_rect)
    }
}

impl ResolvedImagePlacement {
    pub fn from_parts(paint_rect: Rect, tile_rect: Rect) -> Result<Self> {
        validate_placement_rect(paint_rect, "image placement paint rect")?;
        validate_placement_rect(tile_rect, "image placement tile rect")?;
        Ok(Self {
            paint_rect,
            tile_rect,
        })
    }

    #[must_use]
    pub const fn paint_rect(self) -> Rect {
        self.paint_rect
    }

    #[must_use]
    pub const fn tile_rect(self) -> Rect {
        self.tile_rect
    }
}

fn validate_placement_rect(rect: Rect, field: &str) -> Result<()> {
    validate_finite_f64(rect.x(), &format!("{field} x"))?;
    validate_finite_f64(rect.y(), &format!("{field} y"))?;
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return Err(Error::invalid_value(
            field,
            format!("{rect:?}"),
            "must have finite positive width and height",
        ));
    }
    validate_positive_size(rect.size(), field)
}

fn validate_positive_size(size: Size, field: &str) -> Result<()> {
    if !size.width().is_finite()
        || !size.height().is_finite()
        || size.width() <= 0.0
        || size.height() <= 0.0
    {
        return Err(Error::invalid_value(
            field,
            format!("{size:?}"),
            "must have finite positive width and height",
        ));
    }
    Ok(())
}

fn resolve_background_size(paint_rect: Rect, intrinsic_size: Size, size: BackgroundSize) -> Size {
    let intrinsic_width = intrinsic_size.width();
    let intrinsic_height = intrinsic_size.height();
    let scale_x = paint_rect.width() / intrinsic_width;
    let scale_y = paint_rect.height() / intrinsic_height;

    match size.kind() {
        BackgroundSizeKind::Auto => intrinsic_size,
        BackgroundSizeKind::Cover => {
            let scale = scale_x.max(scale_y);
            Size::new(intrinsic_width * scale, intrinsic_height * scale)
        }
        BackgroundSizeKind::Contain => {
            let scale = scale_x.min(scale_y);
            Size::new(intrinsic_width * scale, intrinsic_height * scale)
        }
        BackgroundSizeKind::Explicit { width, height } => {
            let width = resolve_size_component(width, paint_rect.width());
            let height = resolve_size_component(height, paint_rect.height());
            match (width, height) {
                (Some(width), Some(height)) => Size::new(width, height),
                (Some(width), None) => Size::new(width, width * intrinsic_height / intrinsic_width),
                (None, Some(height)) => {
                    Size::new(height * intrinsic_width / intrinsic_height, height)
                }
                (None, None) => intrinsic_size,
            }
        }
    }
}

fn resolve_size_component(component: SizeComponent, axis: f64) -> Option<f64> {
    match component.kind() {
        SizeComponentKind::Auto => None,
        SizeComponentKind::Length(value) => Some(value),
        SizeComponentKind::Percent(value) => Some(axis * value),
    }
}

fn resolve_position_component(
    origin: f64,
    axis: f64,
    tile_axis: f64,
    component: PositionComponent,
) -> f64 {
    match component.kind() {
        PositionComponentKind::Length => origin + component.value(),
        PositionComponentKind::Percent => origin + (axis - tile_axis) * component.value(),
        PositionComponentKind::EdgeOffset(PositionEdge::Start) => origin + component.value(),
        PositionComponentKind::EdgeOffset(PositionEdge::End) => {
            origin + axis - tile_axis - component.value()
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackgroundRepeat {
    x: RepeatMode,
    y: RepeatMode,
}

impl BackgroundRepeat {
    #[must_use]
    pub const fn new(x: RepeatMode, y: RepeatMode) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn repeat() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::Repeat)
    }

    #[must_use]
    pub const fn repeat_x() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::NoRepeat)
    }

    #[must_use]
    pub const fn repeat_y() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::Repeat)
    }

    #[must_use]
    pub const fn no_repeat() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::NoRepeat)
    }

    #[must_use]
    pub const fn x(self) -> RepeatMode {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> RepeatMode {
        self.y
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepeatMode {
    Repeat,
    NoRepeat,
    Round,
    Space,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageRepeatMode {
    NoRepeat,
    RepeatX,
    RepeatY,
    RepeatBoth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageRepeatPlan {
    repeat: BackgroundRepeat,
    mode: ImageRepeatMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedImageRepeat {
    clip_rect: Rect,
    tile_rects: Vec<Rect>,
}

impl ImageRepeatPlan {
    pub fn try_new(repeat: BackgroundRepeat, capabilities: Capabilities) -> Result<Self> {
        if matches!(repeat.x(), RepeatMode::Round) || matches!(repeat.y(), RepeatMode::Round) {
            let unsupported = UnsupportedPrimitive::new(
                PrimitiveFamily::ImageSampling,
                PrimitiveOperation::BackgroundRepeatRound,
            );
            capabilities.ensure_supported(unsupported)?;
            return Err(Error::unsupported_render_primitive(unsupported));
        }
        if matches!(repeat.x(), RepeatMode::Space) || matches!(repeat.y(), RepeatMode::Space) {
            let unsupported = UnsupportedPrimitive::new(
                PrimitiveFamily::ImageSampling,
                PrimitiveOperation::BackgroundRepeatSpace,
            );
            capabilities.ensure_supported(unsupported)?;
            return Err(Error::unsupported_render_primitive(unsupported));
        }

        let mode = match (repeat.x(), repeat.y()) {
            (RepeatMode::NoRepeat, RepeatMode::NoRepeat) => ImageRepeatMode::NoRepeat,
            (RepeatMode::Repeat, RepeatMode::NoRepeat) => ImageRepeatMode::RepeatX,
            (RepeatMode::NoRepeat, RepeatMode::Repeat) => ImageRepeatMode::RepeatY,
            (RepeatMode::Repeat, RepeatMode::Repeat) => ImageRepeatMode::RepeatBoth,
            _ => unreachable!("round and space are rejected before repeat mode mapping"),
        };
        Ok(Self { repeat, mode })
    }

    #[must_use]
    pub const fn repeat(self) -> BackgroundRepeat {
        self.repeat
    }

    #[must_use]
    pub const fn mode(self) -> ImageRepeatMode {
        self.mode
    }

    pub fn resolve(self, placement: ResolvedImagePlacement) -> Result<ResolvedImageRepeat> {
        let repeats_x = matches!(
            self.mode,
            ImageRepeatMode::RepeatX | ImageRepeatMode::RepeatBoth
        );
        let repeats_y = matches!(
            self.mode,
            ImageRepeatMode::RepeatY | ImageRepeatMode::RepeatBoth
        );
        let x_positions = repeat_positions(
            placement.paint_rect().x(),
            placement.paint_rect().width(),
            placement.tile_rect().x(),
            placement.tile_rect().width(),
            repeats_x,
        )?;
        let y_positions = repeat_positions(
            placement.paint_rect().y(),
            placement.paint_rect().height(),
            placement.tile_rect().y(),
            placement.tile_rect().height(),
            repeats_y,
        )?;

        let mut tile_rects = Vec::new();
        for y in y_positions {
            for x in &x_positions {
                tile_rects.push(Rect::new(
                    *x,
                    y,
                    placement.tile_rect().width(),
                    placement.tile_rect().height(),
                ));
            }
        }

        Ok(ResolvedImageRepeat {
            clip_rect: placement.paint_rect(),
            tile_rects,
        })
    }
}

impl ResolvedImageRepeat {
    #[must_use]
    pub const fn clip_rect(&self) -> Rect {
        self.clip_rect
    }

    #[must_use]
    pub fn tile_rects(&self) -> &[Rect] {
        &self.tile_rects
    }
}

fn repeat_positions(
    clip_origin: f64,
    clip_axis: f64,
    tile_origin: f64,
    tile_axis: f64,
    repeats: bool,
) -> Result<Vec<f64>> {
    if tile_axis <= 0.0 || !tile_axis.is_finite() {
        return Err(Error::invalid_value(
            "image repeat tile size",
            tile_axis,
            "must be finite and positive",
        ));
    }

    let clip_end = clip_origin + clip_axis;
    if !repeats {
        return Ok(
            if tile_origin < clip_end && tile_origin + tile_axis > clip_origin {
                vec![tile_origin]
            } else {
                Vec::new()
            },
        );
    }

    let mut origin = tile_origin;
    while origin > clip_origin {
        origin -= tile_axis;
    }

    let mut positions = Vec::new();
    while origin < clip_end {
        if origin + tile_axis > clip_origin {
            positions.push(origin);
        }
        origin += tile_axis;
    }
    Ok(positions)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundBox {
    Border,
    Padding,
    Content,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachment {
    Scroll,
    Fixed,
    Local,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterList {
    ops: Option<Vec<FilterOp>>,
}

impl FilterList {
    #[must_use]
    pub const fn none() -> Self {
        Self { ops: None }
    }

    pub fn try_ops(ops: Vec<FilterOp>) -> Result<Self> {
        if ops.is_empty() {
            return Err(Error::invalid_value(
                "filter operations",
                "[]",
                "must not be empty",
            ));
        }
        Ok(Self { ops: Some(ops) })
    }

    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.ops.is_none()
    }

    #[must_use]
    pub fn ops(&self) -> &[FilterOp] {
        self.ops.as_deref().unwrap_or(&[])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterOp {
    kind: FilterOpKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FilterOpKind {
    Blur(FilterBlur),
    Brightness(FilterAmount),
    Contrast(FilterAmount),
    Grayscale(UnitFilterAmount),
    HueRotate(FilterAngle),
    Invert(UnitFilterAmount),
    Opacity(UnitFilterAmount),
    Saturate(FilterAmount),
    Sepia(UnitFilterAmount),
    DropShadow(Shadow),
}

impl FilterOp {
    #[must_use]
    pub const fn blur(blur: FilterBlur) -> Self {
        Self {
            kind: FilterOpKind::Blur(blur),
        }
    }

    #[must_use]
    pub const fn brightness(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Brightness(amount),
        }
    }

    #[must_use]
    pub const fn contrast(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Contrast(amount),
        }
    }

    #[must_use]
    pub const fn grayscale(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Grayscale(amount),
        }
    }

    #[must_use]
    pub const fn hue_rotate(angle: FilterAngle) -> Self {
        Self {
            kind: FilterOpKind::HueRotate(angle),
        }
    }

    #[must_use]
    pub const fn invert(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Invert(amount),
        }
    }

    #[must_use]
    pub const fn opacity(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Opacity(amount),
        }
    }

    #[must_use]
    pub const fn saturate(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Saturate(amount),
        }
    }

    #[must_use]
    pub const fn sepia(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Sepia(amount),
        }
    }

    #[must_use]
    pub const fn drop_shadow(shadow: Shadow) -> Self {
        Self {
            kind: FilterOpKind::DropShadow(shadow),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> &FilterOpKind {
        &self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterBlur {
    radius: f64,
}

impl FilterBlur {
    pub fn try_new(radius: f64) -> Result<Self> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::invalid_value(
                "filter blur radius",
                radius,
                "must be finite and non-negative",
            ));
        }
        Ok(Self { radius })
    }

    #[must_use]
    pub const fn radius(self) -> f64 {
        self.radius
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAmount {
    value: f64,
}

impl FilterAmount {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "filter amount",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitFilterAmount {
    value: f64,
}

impl UnitFilterAmount {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Error::invalid_value(
                "filter unit amount",
                value,
                "must be finite and between 0 and 1",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAngle {
    radians: f64,
}

impl FilterAngle {
    pub fn try_radians(radians: f64) -> Result<Self> {
        if !radians.is_finite() {
            return Err(Error::invalid_value(
                "filter angle",
                radians,
                "must be finite",
            ));
        }
        Ok(Self { radians })
    }

    #[must_use]
    pub const fn radians(self) -> f64 {
        self.radians
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipInput {
    kind: ClipInputKind,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClipInputKind {
    Shape(Shape),
    Reference(StyleResourceRef),
}

impl ClipInput {
    pub fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: ClipInputKind::Shape(shape),
            coordinate_space: None,
        })
    }

    #[must_use]
    pub const fn reference(reference: StyleResourceRef) -> Self {
        Self {
            kind: ClipInputKind::Reference(reference),
            coordinate_space: None,
        }
    }

    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    #[must_use]
    pub const fn kind(&self) -> &ClipInputKind {
        &self.kind
    }

    #[must_use]
    pub const fn shape(&self) -> Option<&Shape> {
        match &self.kind {
            ClipInputKind::Shape(shape) => Some(shape),
            ClipInputKind::Reference(_) => None,
        }
    }

    #[must_use]
    pub const fn reference_ref(&self) -> Option<&StyleResourceRef> {
        match &self.kind {
            ClipInputKind::Shape(_) => None,
            ClipInputKind::Reference(reference) => Some(reference),
        }
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskInput {
    source: MaskSource,
    mode: MaskMode,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskSource {
    kind: MaskSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaskSourceKind {
    Shape(Shape),
    ImageLayer(StyleImageLayer),
    Reference(StyleResourceRef),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskMode {
    Alpha,
    Luminance,
}

impl MaskInput {
    pub fn try_shape(shape: Shape, mode: MaskMode) -> Result<Self> {
        Ok(Self {
            source: MaskSource::try_shape(shape)?,
            mode,
            coordinate_space: None,
        })
    }

    #[must_use]
    pub const fn image_layer(layer: StyleImageLayer, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::image_layer(layer),
            mode,
            coordinate_space: None,
        }
    }

    #[must_use]
    pub const fn reference(reference: StyleResourceRef, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::reference(reference),
            mode,
            coordinate_space: None,
        }
    }

    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    #[must_use]
    pub const fn source(&self) -> &MaskSource {
        &self.source
    }

    #[must_use]
    pub const fn mode(&self) -> MaskMode {
        self.mode
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

impl MaskSource {
    fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: MaskSourceKind::Shape(shape),
        })
    }

    const fn image_layer(layer: StyleImageLayer) -> Self {
        Self {
            kind: MaskSourceKind::ImageLayer(layer),
        }
    }

    const fn reference(reference: StyleResourceRef) -> Self {
        Self {
            kind: MaskSourceKind::Reference(reference),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> &MaskSourceKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BorderSide {
    style: BorderStyle,
    width: f64,
    paint: Paint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BorderStyle {
    None,
    Hidden,
    Solid,
    Dashed,
    Dotted,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
}

impl BorderSide {
    pub fn try_new(style: BorderStyle, width: f64, paint: impl Into<Paint>) -> Result<Self> {
        validate_non_negative_f64(width, "border side width")?;
        let paint = paint.into();
        validate_paint(&paint)?;
        Ok(Self {
            style,
            width,
            paint,
        })
    }

    #[must_use]
    pub const fn style(&self) -> BorderStyle {
        self.style
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BorderEdges {
    top: BorderSide,
    right: BorderSide,
    bottom: BorderSide,
    left: BorderSide,
}

impl BorderEdges {
    #[must_use]
    pub const fn new(
        top: BorderSide,
        right: BorderSide,
        bottom: BorderSide,
        left: BorderSide,
    ) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    #[must_use]
    pub const fn top(&self) -> &BorderSide {
        &self.top
    }

    #[must_use]
    pub const fn right(&self) -> &BorderSide {
        &self.right
    }

    #[must_use]
    pub const fn bottom(&self) -> &BorderSide {
        &self.bottom
    }

    #[must_use]
    pub const fn left(&self) -> &BorderSide {
        &self.left
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outline {
    style: OutlineStyle,
    width: f64,
    paint: Paint,
    offset: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutlineStyle {
    None,
    Solid,
    Dashed,
    Dotted,
    Double,
    Auto,
}

impl Outline {
    pub fn try_new(
        style: OutlineStyle,
        width: f64,
        paint: impl Into<Paint>,
        offset: f64,
    ) -> Result<Self> {
        validate_non_negative_f64(width, "outline width")?;
        validate_finite_f64(offset, "outline offset")?;
        let paint = paint.into();
        validate_paint(&paint)?;
        Ok(Self {
            style,
            width,
            paint,
            offset,
        })
    }

    #[must_use]
    pub const fn style(&self) -> OutlineStyle {
        self.style
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundLayer {
    image: StyleImageLayer,
}

impl BackgroundLayer {
    #[must_use]
    pub const fn new(image: StyleImageLayer) -> Self {
        Self { image }
    }

    #[must_use]
    pub const fn image(&self) -> &StyleImageLayer {
        &self.image
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundStack {
    color: Option<Color>,
    layers: Vec<BackgroundLayer>,
}

impl BackgroundStack {
    pub fn try_new(color: Option<Color>, layers: Vec<BackgroundLayer>) -> Result<Self> {
        if color.is_none() && layers.is_empty() {
            return Err(Error::invalid_value(
                "background stack",
                "none + []",
                "must include a color or at least one layer",
            ));
        }
        Ok(Self { color, layers })
    }

    #[must_use]
    pub const fn color(&self) -> Option<Color> {
        self.color
    }

    #[must_use]
    pub fn layers(&self) -> &[BackgroundLayer] {
        &self.layers
    }
}
