use super::{
    Capabilities, Color, CoordinateSpaceKind, CoordinateSpaceTag, Error, FillRule, FilledPath,
    Image, ImageColorProfilePolicy, ImageId, ImageOrientationPolicy, Paint, Path, Point,
    PrimitiveFamily, PrimitiveOperation, Radii, Rect, Result, Shadow, Shape, Size,
    SymbolicColorPolicy, Transform, UnresolvedResource, UnresolvedResourceKind,
    UnsupportedPrimitive,
    shape::ShapeKind,
    validation::{
        validate_finite_f64, validate_non_negative_f64, validate_paint, validate_path,
        validate_shape, validate_size,
    },
};
use kurbo::Shape as KurboShape;

const MAX_IMAGE_REPEAT_TILES: usize = 1_000_000;
const MAX_IMAGE_REPEAT_TILES_RULE: &str = "must not exceed 1000000";

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
        let tile_count = x_positions
            .len()
            .checked_mul(y_positions.len())
            .ok_or_else(|| image_repeat_tile_count_error("overflow"))?;
        validate_image_repeat_tile_count(tile_count)?;

        let mut tile_rects = Vec::with_capacity(tile_count);
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
    if !clip_end.is_finite() {
        return Err(Error::invalid_value(
            "image repeat geometry",
            format!("origin {clip_origin}, axis {clip_axis}"),
            "resolved clip extent must be finite",
        ));
    }
    if !repeats {
        return Ok(
            if tile_origin < clip_end && tile_origin + tile_axis > clip_origin {
                vec![tile_origin]
            } else {
                Vec::new()
            },
        );
    }

    let origin = first_repeated_tile_origin(clip_origin, tile_origin, tile_axis);
    let count = repeat_position_count(origin, clip_end, tile_axis)?;
    validate_image_repeat_tile_count(count)?;

    let mut positions = Vec::with_capacity(count);
    for index in 0..count {
        positions.push(origin + tile_axis * index as f64);
    }
    Ok(positions)
}

fn first_repeated_tile_origin(clip_origin: f64, tile_origin: f64, tile_axis: f64) -> f64 {
    let offset = ((clip_origin - tile_origin) / tile_axis).floor();
    let mut origin = tile_origin + offset * tile_axis;
    if origin + tile_axis <= clip_origin {
        origin += tile_axis;
    }
    origin
}

fn repeat_position_count(origin: f64, clip_end: f64, tile_axis: f64) -> Result<usize> {
    if origin >= clip_end {
        return Ok(0);
    }
    let count = ((clip_end - origin) / tile_axis).ceil();
    if !count.is_finite() || count < 0.0 || count > usize::MAX as f64 {
        return Err(image_repeat_tile_count_error(count));
    }
    Ok(count as usize)
}

fn validate_image_repeat_tile_count(tile_count: usize) -> Result<()> {
    if tile_count > MAX_IMAGE_REPEAT_TILES {
        return Err(image_repeat_tile_count_error(tile_count));
    }
    Ok(())
}

fn image_repeat_tile_count_error(value: impl std::fmt::Display) -> Error {
    Error::invalid_value(
        "image repeat tile count",
        value,
        MAX_IMAGE_REPEAT_TILES_RULE,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundBox {
    Border,
    Padding,
    Content,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundAreas {
    border_box: Rect,
    padding_box: Rect,
    content_box: Rect,
}

impl BackgroundAreas {
    pub fn try_new(border_box: Rect, padding_box: Rect, content_box: Rect) -> Result<Self> {
        validate_background_rect(border_box, "background border box")?;
        validate_background_rect(padding_box, "background padding box")?;
        validate_background_rect(content_box, "background content box")?;
        Ok(Self {
            border_box,
            padding_box,
            content_box,
        })
    }

    #[must_use]
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    #[must_use]
    pub const fn padding_box(self) -> Rect {
        self.padding_box
    }

    #[must_use]
    pub const fn content_box(self) -> Rect {
        self.content_box
    }

    #[must_use]
    pub const fn rect_for(self, box_kind: BackgroundBox) -> Rect {
        match box_kind {
            BackgroundBox::Border => self.border_box,
            BackgroundBox::Padding => self.padding_box,
            BackgroundBox::Content => self.content_box,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundClipGeometry {
    kind: BackgroundClipGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BackgroundClipGeometryKind {
    Rect(Rect),
    Shape(Shape),
}

impl BackgroundClipGeometry {
    pub fn try_rect(rect: Rect) -> Result<Self> {
        validate_background_rect(rect, "background clip rect")?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Rect(rect),
        })
    }

    pub fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Shape(shape),
        })
    }

    #[must_use]
    pub fn kind(&self) -> &BackgroundClipGeometryKind {
        &self.kind
    }

    #[must_use]
    pub fn rect(&self) -> Option<Rect> {
        match &self.kind {
            BackgroundClipGeometryKind::Rect(rect) => Some(*rect),
            BackgroundClipGeometryKind::Shape(_) => None,
        }
    }

    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        match &self.kind {
            BackgroundClipGeometryKind::Rect(_) => None,
            BackgroundClipGeometryKind::Shape(shape) => Some(shape),
        }
    }
}

fn validate_background_rect(rect: Rect, field: &str) -> Result<()> {
    validate_finite_f64(rect.x(), &format!("{field} x"))?;
    validate_finite_f64(rect.y(), &format!("{field} y"))?;
    if !rect.width().is_finite()
        || !rect.height().is_finite()
        || rect.width() <= 0.0
        || rect.height() <= 0.0
    {
        return Err(Error::invalid_value(
            field,
            format!("{rect:?}"),
            "must have finite positive width and height",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachment {
    Scroll,
    Fixed,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageAttachmentPlan {
    attachment: BackgroundAttachment,
    coordinate_space: Option<CoordinateSpaceTag>,
}

impl ImageAttachmentPlan {
    pub fn try_new(
        attachment: BackgroundAttachment,
        coordinate_space: Option<CoordinateSpaceTag>,
    ) -> Result<Self> {
        if matches!(attachment, BackgroundAttachment::Fixed) {
            let Some(tag) = coordinate_space else {
                return Err(Error::invalid_value(
                    "background attachment coordinate space",
                    "none",
                    "fixed backgrounds require a viewport coordinate tag",
                ));
            };
            if tag.kind() != CoordinateSpaceKind::Viewport {
                return Err(Error::invalid_value(
                    "background attachment coordinate space",
                    format!("{:?}", tag.kind()),
                    "fixed backgrounds require a viewport coordinate tag",
                ));
            }
        }
        Ok(Self {
            attachment,
            coordinate_space,
        })
    }

    #[must_use]
    pub const fn attachment(self) -> BackgroundAttachment {
        self.attachment
    }

    #[must_use]
    pub const fn coordinate_space(self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
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

    pub fn color_filter_pipeline(
        &self,
    ) -> std::result::Result<Option<ColorFilterPipeline>, UnsupportedPrimitive> {
        let Some(ops) = self.ops.as_deref() else {
            return Ok(None);
        };

        let mut color_ops = Vec::with_capacity(ops.len());
        for op in ops {
            color_ops.push(ColorFilterOp::try_from_filter_op(op)?);
        }

        Ok(Some(ColorFilterPipeline { ops: color_ops }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColorFilterPipeline {
    ops: Vec<ColorFilterOp>,
}

impl ColorFilterPipeline {
    #[must_use]
    pub fn ops(&self) -> &[ColorFilterOp] {
        &self.ops
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorFilterOp {
    Brightness(FilterAmount),
    Contrast(FilterAmount),
    Grayscale(UnitFilterAmount),
    HueRotate(FilterAngle),
    Invert(UnitFilterAmount),
    Opacity(UnitFilterAmount),
    Saturate(FilterAmount),
    Sepia(UnitFilterAmount),
}

impl ColorFilterOp {
    fn try_from_filter_op(op: &FilterOp) -> std::result::Result<Self, UnsupportedPrimitive> {
        match op.kind() {
            FilterOpKind::Brightness(amount) => Ok(Self::Brightness(*amount)),
            FilterOpKind::Contrast(amount) => Ok(Self::Contrast(*amount)),
            FilterOpKind::Grayscale(amount) => Ok(Self::Grayscale(*amount)),
            FilterOpKind::HueRotate(angle) => Ok(Self::HueRotate(*angle)),
            FilterOpKind::Invert(amount) => Ok(Self::Invert(*amount)),
            FilterOpKind::Opacity(amount) => Ok(Self::Opacity(*amount)),
            FilterOpKind::Saturate(amount) => Ok(Self::Saturate(*amount)),
            FilterOpKind::Sepia(amount) => Ok(Self::Sepia(*amount)),
            FilterOpKind::Blur(_) => Err(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::ColorFilterBlur,
            )),
            FilterOpKind::DropShadow(_) => Err(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::ColorFilterDropShadow,
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilteredImagePaint {
    resource: ResolvedImageResource,
    filters: FilterList,
}

impl FilteredImagePaint {
    pub fn try_new(resource: ResolvedImageResource, filters: FilterList) -> Result<Self> {
        if filters.is_none() {
            return Err(Error::invalid_value(
                "filtered image paint filters",
                "none",
                "must contain at least one filter operation",
            ));
        }
        Ok(Self { resource, filters })
    }

    #[must_use]
    pub const fn resource(&self) -> &ResolvedImageResource {
        &self.resource
    }

    #[must_use]
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint,
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropCaptureBounds {
    rect: Rect,
}

impl BackdropCaptureBounds {
    pub fn try_new(rect: Rect) -> Result<Self> {
        validate_finite_f64(rect.x(), "backdrop capture bounds x")?;
        validate_finite_f64(rect.y(), "backdrop capture bounds y")?;
        if !rect.width().is_finite() || rect.width() <= 0.0 {
            return Err(Error::invalid_value(
                "backdrop capture bounds width",
                rect.width(),
                "must be finite and positive",
            ));
        }
        if !rect.height().is_finite() || rect.height() <= 0.0 {
            return Err(Error::invalid_value(
                "backdrop capture bounds height",
                rect.height(),
                "must be finite and positive",
            ));
        }
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackdropFilterInput {
    filters: FilterList,
    capture_bounds: BackdropCaptureBounds,
    clip: Option<ClipInput>,
}

impl BackdropFilterInput {
    pub fn try_new(
        filters: FilterList,
        capture_bounds: BackdropCaptureBounds,
        clip: Option<ClipInput>,
    ) -> Result<Self> {
        validate_backdrop_filter_list(&filters)?;
        validate_backdrop_clip(clip.as_ref())?;
        Ok(Self {
            filters,
            capture_bounds,
            clip,
        })
    }

    pub fn try_root_backdrop(filters: FilterList, clip: Option<ClipInput>) -> Result<Self> {
        validate_backdrop_filter_list(&filters)?;
        validate_backdrop_clip(clip.as_ref())?;
        Err(Error::unsupported_render_primitive(
            UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::RootBackdropPolicy,
            ),
        ))
    }

    #[must_use]
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    #[must_use]
    pub const fn capture_bounds(&self) -> BackdropCaptureBounds {
        self.capture_bounds
    }

    #[must_use]
    pub const fn clip(&self) -> Option<&ClipInput> {
        self.clip.as_ref()
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropCapture,
        ))?;
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaterializedBackdropFilterExecution,
        ))?;
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ))
    }
}

fn validate_backdrop_filter_list(filters: &FilterList) -> Result<()> {
    if filters.is_none() {
        return Err(Error::invalid_value(
            "backdrop filter input filters",
            "none",
            "must contain at least one supported filter operation",
        ));
    }
    filters.materialized_image_filter_pipeline()?;
    Ok(())
}

fn validate_backdrop_clip(clip: Option<&ClipInput>) -> Result<()> {
    let Some(clip) = clip else {
        return Ok(());
    };
    clip.ensure_supported(Capabilities::VELLO_0_9)
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
    Path(FilledPath),
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

    pub fn try_filled_path(path: FilledPath) -> Result<Self> {
        validate_path_clip(path.path())?;
        Ok(Self {
            kind: ClipInputKind::Path(path),
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
            ClipInputKind::Path(_) => None,
            ClipInputKind::Reference(_) => None,
        }
    }

    #[must_use]
    pub const fn filled_path(&self) -> Option<&FilledPath> {
        match &self.kind {
            ClipInputKind::Shape(_) => None,
            ClipInputKind::Path(path) => Some(path),
            ClipInputKind::Reference(_) => None,
        }
    }

    #[must_use]
    pub const fn reference_ref(&self) -> Option<&StyleResourceRef> {
        match &self.kind {
            ClipInputKind::Shape(_) => None,
            ClipInputKind::Path(_) => None,
            ClipInputKind::Reference(reference) => Some(reference),
        }
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        match &self.kind {
            ClipInputKind::Shape(_) | ClipInputKind::Path(_) => {
                capabilities.ensure_supported(UnsupportedPrimitive::new(
                    PrimitiveFamily::MasksAndClips,
                    PrimitiveOperation::ShapeClip,
                ))
            }
            ClipInputKind::Reference(reference) => Err(Error::unresolved_resource(
                UnresolvedResource::new(UnresolvedResourceKind::Clip, reference.identifier()),
            )),
        }
    }

    pub fn normalize(&self, capabilities: Capabilities) -> Result<NormalizedClip> {
        self.ensure_supported(capabilities)?;
        let geometry = match &self.kind {
            ClipInputKind::Shape(shape) => ClipGeometry::from_shape(shape)?,
            ClipInputKind::Path(path) => ClipGeometry::try_path(path.clone())?,
            ClipInputKind::Reference(reference) => {
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::Clip,
                    reference.identifier(),
                )));
            }
        };
        NormalizedClip::try_new(geometry, self.coordinate_space)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedClip {
    geometry: ClipGeometry,
    coordinate_space: Option<CoordinateSpaceTag>,
}

impl NormalizedClip {
    pub fn try_new(
        geometry: ClipGeometry,
        coordinate_space: Option<CoordinateSpaceTag>,
    ) -> Result<Self> {
        validate_clip_geometry(&geometry)?;
        validate_clip_transformed_bounds(&geometry, coordinate_space)?;
        Ok(Self {
            geometry,
            coordinate_space,
        })
    }

    #[must_use]
    pub const fn geometry(&self) -> &ClipGeometry {
        &self.geometry
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipGeometry {
    kind: ClipGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClipGeometryKind {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(FilledPath),
}

impl ClipGeometry {
    pub fn try_path(path: FilledPath) -> Result<Self> {
        validate_path_clip(path.path())?;
        Ok(Self {
            kind: ClipGeometryKind::Path(path),
        })
    }

    fn from_shape(shape: &Shape) -> Result<Self> {
        validate_shape(shape)?;
        Ok(Self {
            kind: match shape.kind() {
                ShapeKind::Rect(rect) => ClipGeometryKind::Rect(*rect),
                ShapeKind::RoundedRect { rect, radii } => ClipGeometryKind::RoundedRect {
                    rect: *rect,
                    radii: *radii,
                },
                ShapeKind::Circle { center, radius } => ClipGeometryKind::Circle {
                    center: *center,
                    radius: *radius,
                },
                ShapeKind::Ellipse { center, radii } => ClipGeometryKind::Ellipse {
                    center: *center,
                    radii: *radii,
                },
                ShapeKind::Path(path) => {
                    ClipGeometryKind::Path(FilledPath::try_new(path.clone(), FillRule::NonZero)?)
                }
            },
        })
    }

    #[must_use]
    pub const fn kind(&self) -> &ClipGeometryKind {
        &self.kind
    }
}

pub(crate) fn validate_clip_input(input: &ClipInput) -> Result<()> {
    match input.kind() {
        ClipInputKind::Shape(shape) => validate_shape(shape),
        ClipInputKind::Path(path) => validate_path_clip(path.path()),
        ClipInputKind::Reference(_) => Ok(()),
    }?;
    if let Some(coordinate_space) = input.coordinate_space() {
        validate_clip_transform_values(coordinate_space.transform())?;
    }
    Ok(())
}

fn validate_clip_geometry(geometry: &ClipGeometry) -> Result<()> {
    match geometry.kind() {
        ClipGeometryKind::Rect(rect) => validate_shape(&Shape::rect(*rect)),
        ClipGeometryKind::RoundedRect { rect, radii } => {
            validate_shape(&Shape::try_rounded_rect(*rect, *radii)?)
        }
        ClipGeometryKind::Circle { center, radius } => {
            validate_shape(&Shape::try_circle(*center, *radius)?)
        }
        ClipGeometryKind::Ellipse { center, radii } => {
            validate_shape(&Shape::try_ellipse(*center, *radii)?)
        }
        ClipGeometryKind::Path(path) => validate_path_clip(path.path()),
    }
}

fn validate_path_clip(path: &Path) -> Result<()> {
    validate_path(path)
}

fn validate_clip_transform_values(transform: Transform) -> Result<()> {
    for value in transform.as_array() {
        validate_finite_f64(value, "clip coordinate-space transform")?;
    }
    Ok(())
}

fn validate_clip_transformed_bounds(
    geometry: &ClipGeometry,
    coordinate_space: Option<CoordinateSpaceTag>,
) -> Result<()> {
    let bounds = clip_geometry_bounds(geometry)?;
    let transform = coordinate_space
        .map(CoordinateSpaceTag::transform)
        .unwrap_or_else(Transform::identity);
    validate_clip_transform_values(transform)?;
    transformed_clip_bounds(bounds, transform).map(|_| ())
}

fn clip_geometry_bounds(geometry: &ClipGeometry) -> Result<Rect> {
    match geometry.kind() {
        ClipGeometryKind::Rect(rect) | ClipGeometryKind::RoundedRect { rect, .. } => Ok(*rect),
        ClipGeometryKind::Circle { center, radius } => finite_clip_rect(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        ),
        ClipGeometryKind::Ellipse { center, radii } => finite_clip_rect(
            center.x() - radii.width(),
            center.y() - radii.height(),
            radii.width() * 2.0,
            radii.height() * 2.0,
        ),
        ClipGeometryKind::Path(path) => {
            let bounds = path.path().to_kurbo().bounding_box();
            finite_clip_rect(bounds.x0, bounds.y0, bounds.width(), bounds.height())
        }
    }
}

fn transformed_clip_bounds(rect: Rect, transform: Transform) -> Result<Rect> {
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
            return Err(Error::invalid_value(
                "clip transformed bounds",
                "non-finite",
                "must remain finite after coordinate-space transform",
            ));
        }
        min_x = min_x.min(transformed_x);
        min_y = min_y.min(transformed_y);
        max_x = max_x.max(transformed_x);
        max_y = max_y.max(transformed_y);
    }
    finite_clip_rect(min_x, min_y, max_x - min_x, max_y - min_y)
}

fn finite_clip_rect(x: f64, y: f64, width: f64, height: f64) -> Result<Rect> {
    if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
        return Err(Error::invalid_value(
            "clip transformed bounds",
            format!("x {x}, y {y}, width {width}, height {height}"),
            "must remain finite after coordinate-space transform",
        ));
    }
    if width < 0.0 || height < 0.0 {
        return Err(Error::invalid_value(
            "clip transformed bounds",
            format!("width {width}, height {height}"),
            "must be non-negative",
        ));
    }
    Ok(Rect::new(x, y, width, height))
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

#[derive(Clone, Debug, PartialEq)]
pub struct MaskLayerStack {
    layers: Vec<MaskLayer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskLayer {
    input: MaskInput,
    composite_mode: MaskCompositeMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskCompositeMode {
    Add,
    Subtract,
    Intersect,
    Exclude,
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

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.source.kind() {
            MaskSourceKind::Reference(reference) => {
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::Mask,
                    reference.identifier(),
                )));
            }
            MaskSourceKind::ImageLayer(layer) => {
                layer.source().require_resolved()?;
            }
            MaskSourceKind::Shape(_) => {}
        }

        match self.mode {
            MaskMode::Alpha => Err(Error::unsupported_render_primitive(
                UnsupportedPrimitive::new(
                    PrimitiveFamily::MasksAndClips,
                    PrimitiveOperation::AlphaMaskSourceExecution,
                ),
            )),
            MaskMode::Luminance => capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LuminanceMaskMode,
            )),
        }
    }

    fn ensure_stack_input_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.source.kind() {
            MaskSourceKind::Reference(reference) => {
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::Mask,
                    reference.identifier(),
                )));
            }
            MaskSourceKind::ImageLayer(layer) => {
                layer.source().require_resolved()?;
            }
            MaskSourceKind::Shape(_) => {}
        }

        match self.mode {
            MaskMode::Alpha => Ok(()),
            MaskMode::Luminance => capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LuminanceMaskMode,
            )),
        }
    }
}

impl MaskLayerStack {
    pub fn try_new(layers: impl IntoIterator<Item = MaskLayer>) -> Result<Self> {
        let layers = layers.into_iter().collect::<Vec<_>>();
        if layers.is_empty() {
            return Err(Error::invalid_value(
                "mask layer stack",
                0,
                "must contain at least one layer",
            ));
        }
        Ok(Self { layers })
    }

    #[must_use]
    pub fn single(layer: impl Into<MaskLayer>) -> Self {
        Self {
            layers: vec![layer.into()],
        }
    }

    #[must_use]
    pub fn layers(&self) -> &[MaskLayer] {
        &self.layers
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        for layer in &self.layers {
            layer.input.ensure_stack_input_supported(capabilities)?;
            layer.ensure_composite_supported(capabilities)?;
        }

        if self.layers.len() > 1 {
            return capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MultiLayerMaskComposition,
            ));
        }

        self.layers[0].input.ensure_supported(capabilities)
    }
}

impl MaskLayer {
    #[must_use]
    pub const fn new(input: MaskInput) -> Self {
        Self {
            input,
            composite_mode: MaskCompositeMode::Add,
        }
    }

    pub const fn try_new(input: MaskInput, composite_mode: MaskCompositeMode) -> Result<Self> {
        Ok(Self {
            input,
            composite_mode,
        })
    }

    #[must_use]
    pub const fn input(&self) -> &MaskInput {
        &self.input
    }

    #[must_use]
    pub const fn composite_mode(&self) -> MaskCompositeMode {
        self.composite_mode
    }

    fn ensure_composite_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.composite_mode {
            MaskCompositeMode::Add => Ok(()),
            MaskCompositeMode::Subtract
            | MaskCompositeMode::Intersect
            | MaskCompositeMode::Exclude => {
                capabilities.ensure_supported(UnsupportedPrimitive::new(
                    PrimitiveFamily::MasksAndClips,
                    PrimitiveOperation::MaskCompositeMode,
                ))
            }
        }
    }
}

impl From<MaskInput> for MaskLayer {
    fn from(input: MaskInput) -> Self {
        Self::new(input)
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxDecorationBreak {
    Slice,
    Clone,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedBoxRadii {
    border_box: Rect,
    radii: Radii,
}

impl NormalizedBoxRadii {
    pub fn try_new(border_box: Rect, radii: Radii) -> Result<Self> {
        validate_background_rect(border_box, "box decoration border box")?;
        validate_box_decoration_radii(radii)?;
        Ok(Self {
            border_box,
            radii: scale_box_radii(border_box, radii),
        })
    }

    #[must_use]
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    #[must_use]
    pub const fn radii(self) -> Radii {
        self.radii
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationFragment {
    areas: BackgroundAreas,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
    border_clip_override: Option<BackgroundClipGeometry>,
}

impl BoxDecorationFragment {
    pub fn try_new(
        areas: BackgroundAreas,
        radii: Radii,
        break_mode: BoxDecorationBreak,
    ) -> Result<Self> {
        Ok(Self {
            areas,
            radii: NormalizedBoxRadii::try_new(areas.border_box(), radii)?,
            break_mode,
            border_clip_override: None,
        })
    }

    #[must_use]
    pub fn with_border_clip_override(
        mut self,
        border_clip_override: BackgroundClipGeometry,
    ) -> Self {
        self.border_clip_override = Some(border_clip_override);
        self
    }

    #[must_use]
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }

    #[must_use]
    pub const fn border_clip_override(&self) -> Option<&BackgroundClipGeometry> {
        self.border_clip_override.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationInput {
    border_edges: Option<BorderEdges>,
    outline: Option<Outline>,
    fragments: Vec<BoxDecorationFragment>,
}

impl BoxDecorationInput {
    pub fn try_new(
        border_edges: Option<BorderEdges>,
        outline: Option<Outline>,
        fragments: Vec<BoxDecorationFragment>,
    ) -> Result<Self> {
        if fragments.is_empty() {
            return Err(Error::invalid_value(
                "box decoration fragments",
                "[]",
                "must contain at least one fragment",
            ));
        }
        Ok(Self {
            border_edges,
            outline,
            fragments,
        })
    }

    #[must_use]
    pub const fn border_edges(&self) -> Option<&BorderEdges> {
        self.border_edges.as_ref()
    }

    #[must_use]
    pub const fn outline(&self) -> Option<&Outline> {
        self.outline.as_ref()
    }

    #[must_use]
    pub fn fragments(&self) -> &[BoxDecorationFragment] {
        &self.fragments
    }

    pub fn normalize(&self, _capabilities: Capabilities) -> Result<NormalizedBoxDecoration> {
        let mut commands = Vec::new();

        for (fragment_index, fragment) in self.fragments.iter().enumerate() {
            let target_rect = fragment.areas().border_box();
            let clip = border_clip_geometry(fragment)?;

            if let Some(border_edges) = &self.border_edges {
                for (side, border_side) in border_sides(border_edges) {
                    if let Some(style) = normalize_border_style(border_side)? {
                        commands.push(NormalizedBoxDecorationCommand {
                            kind: NormalizedBoxDecorationCommandKind::Border(
                                NormalizedBorderCommand {
                                    fragment_index,
                                    side,
                                    width: border_side.width(),
                                    paint: border_side.paint().clone(),
                                    style,
                                    target_rect,
                                    clip: clip.clone(),
                                    radii: fragment.radii(),
                                    break_mode: fragment.break_mode(),
                                },
                            ),
                        });
                    }
                }
            }

            if let Some(outline) = &self.outline
                && let Some(style) = normalize_outline_style(outline)?
            {
                commands.push(NormalizedBoxDecorationCommand {
                    kind: NormalizedBoxDecorationCommandKind::Outline(NormalizedOutlineCommand {
                        fragment_index,
                        width: outline.width(),
                        paint: outline.paint().clone(),
                        offset: outline.offset(),
                        style,
                        target_rect: outline_target_rect(target_rect, outline.offset())?,
                        clip,
                        radii: fragment.radii(),
                        break_mode: fragment.break_mode(),
                    }),
                });
            }
        }

        Ok(NormalizedBoxDecoration { commands })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxSide {
    Top,
    Right,
    Bottom,
    Left,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedDoubleBorderBands {
    original_width: f64,
    outer_width: f64,
    gap_width: f64,
    inner_width: f64,
}

impl NormalizedDoubleBorderBands {
    #[must_use]
    fn from_width(width: f64) -> Self {
        let outer_width = width / 3.0;
        let gap_width = width / 3.0;
        let inner_width = width - outer_width - gap_width;
        Self {
            original_width: width,
            outer_width,
            gap_width,
            inner_width,
        }
    }

    #[must_use]
    pub const fn original_width(self) -> f64 {
        self.original_width
    }

    #[must_use]
    pub const fn outer_width(self) -> f64 {
        self.outer_width
    }

    #[must_use]
    pub const fn gap_width(self) -> f64 {
        self.gap_width
    }

    #[must_use]
    pub const fn inner_width(self) -> f64 {
        self.inner_width
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBorderStyle {
    Solid,
    Dashed,
    Dotted,
    Double(NormalizedDoubleBorderBands),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizedOutlineStyle {
    Solid,
    Dashed,
    Dotted,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBorderCommand {
    fragment_index: usize,
    side: BoxSide,
    width: f64,
    paint: Paint,
    style: NormalizedBorderStyle,
    target_rect: Rect,
    clip: BackgroundClipGeometry,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
}

impl NormalizedBorderCommand {
    #[must_use]
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
    }

    #[must_use]
    pub const fn side(&self) -> BoxSide {
        self.side
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
    pub const fn style(&self) -> &NormalizedBorderStyle {
        &self.style
    }

    #[must_use]
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedOutlineCommand {
    fragment_index: usize,
    width: f64,
    paint: Paint,
    offset: f64,
    style: NormalizedOutlineStyle,
    target_rect: Rect,
    clip: BackgroundClipGeometry,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
}

impl NormalizedOutlineCommand {
    #[must_use]
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
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

    #[must_use]
    pub const fn style(&self) -> NormalizedOutlineStyle {
        self.style
    }

    #[must_use]
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecoration {
    commands: Vec<NormalizedBoxDecorationCommand>,
}

impl NormalizedBoxDecoration {
    #[must_use]
    pub fn commands(&self) -> &[NormalizedBoxDecorationCommand] {
        &self.commands
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecorationCommand {
    kind: NormalizedBoxDecorationCommandKind,
}

impl NormalizedBoxDecorationCommand {
    #[must_use]
    pub const fn kind(&self) -> &NormalizedBoxDecorationCommandKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBoxDecorationCommandKind {
    Border(NormalizedBorderCommand),
    Outline(NormalizedOutlineCommand),
}

fn border_sides(edges: &BorderEdges) -> [(BoxSide, &BorderSide); 4] {
    [
        (BoxSide::Top, edges.top()),
        (BoxSide::Right, edges.right()),
        (BoxSide::Bottom, edges.bottom()),
        (BoxSide::Left, edges.left()),
    ]
}

fn border_clip_geometry(fragment: &BoxDecorationFragment) -> Result<BackgroundClipGeometry> {
    if let Some(clip) = fragment.border_clip_override() {
        return Ok(clip.clone());
    }
    BackgroundClipGeometry::try_rect(fragment.areas().border_box())
}

fn normalize_border_style(side: &BorderSide) -> Result<Option<NormalizedBorderStyle>> {
    if side.width() == 0.0 || matches!(side.style(), BorderStyle::None | BorderStyle::Hidden) {
        return Ok(None);
    }

    let style = match side.style() {
        BorderStyle::None | BorderStyle::Hidden => unreachable!("suppressed before style mapping"),
        BorderStyle::Solid => NormalizedBorderStyle::Solid,
        BorderStyle::Dashed => NormalizedBorderStyle::Dashed,
        BorderStyle::Dotted => NormalizedBorderStyle::Dotted,
        BorderStyle::Double => {
            NormalizedBorderStyle::Double(NormalizedDoubleBorderBands::from_width(side.width()))
        }
        BorderStyle::Groove => {
            return unsupported_border_style(PrimitiveOperation::BorderGrooveStyle);
        }
        BorderStyle::Ridge => {
            return unsupported_border_style(PrimitiveOperation::BorderRidgeStyle);
        }
        BorderStyle::Inset => {
            return unsupported_border_style(PrimitiveOperation::BorderInsetStyle);
        }
        BorderStyle::Outset => {
            return unsupported_border_style(PrimitiveOperation::BorderOutsetStyle);
        }
    };

    Ok(Some(style))
}

fn unsupported_border_style(
    operation: PrimitiveOperation,
) -> Result<Option<NormalizedBorderStyle>> {
    let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
    Err(Error::unsupported_render_primitive(unsupported))
}

fn normalize_outline_style(outline: &Outline) -> Result<Option<NormalizedOutlineStyle>> {
    if outline.width() == 0.0 || matches!(outline.style(), OutlineStyle::None) {
        return Ok(None);
    }

    let style = match outline.style() {
        OutlineStyle::None => unreachable!("suppressed before style mapping"),
        OutlineStyle::Solid => NormalizedOutlineStyle::Solid,
        OutlineStyle::Dashed => NormalizedOutlineStyle::Dashed,
        OutlineStyle::Dotted => NormalizedOutlineStyle::Dotted,
        OutlineStyle::Double => {
            return unsupported_outline_style(PrimitiveOperation::OutlineDoubleStyle);
        }
        OutlineStyle::Auto => {
            return unsupported_outline_style(PrimitiveOperation::OutlineAutoStyle);
        }
    };

    Ok(Some(style))
}

fn unsupported_outline_style(
    operation: PrimitiveOperation,
) -> Result<Option<NormalizedOutlineStyle>> {
    let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
    Err(Error::unsupported_render_primitive(unsupported))
}

fn outline_target_rect(border_box: Rect, offset: f64) -> Result<Rect> {
    let x = border_box.x() - offset;
    let y = border_box.y() - offset;
    let width = border_box.width() + offset * 2.0;
    let height = border_box.height() + offset * 2.0;

    if !x.is_finite()
        || !y.is_finite()
        || !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
    {
        return Err(Error::invalid_value(
            "outline target rect",
            format!("border box {border_box:?}, offset {offset}"),
            "must resolve to finite positive width and height",
        ));
    }

    Ok(Rect::new(x, y, width, height))
}

fn validate_box_decoration_radii(radii: Radii) -> Result<()> {
    for (field, value) in [
        ("box decoration top-left radius", radii.top_left()),
        ("box decoration top-right radius", radii.top_right()),
        ("box decoration bottom-right radius", radii.bottom_right()),
        ("box decoration bottom-left radius", radii.bottom_left()),
    ] {
        validate_non_negative_f64(value, field)?;
    }
    Ok(())
}

fn scale_box_radii(border_box: Rect, radii: Radii) -> Radii {
    let mut scale: f64 = 1.0;
    scale = scale.min(corner_scale(
        border_box.width(),
        radii.top_left() + radii.top_right(),
    ));
    scale = scale.min(corner_scale(
        border_box.width(),
        radii.bottom_left() + radii.bottom_right(),
    ));
    scale = scale.min(corner_scale(
        border_box.height(),
        radii.top_left() + radii.bottom_left(),
    ));
    scale = scale.min(corner_scale(
        border_box.height(),
        radii.top_right() + radii.bottom_right(),
    ));

    if scale >= 1.0 {
        return radii;
    }

    Radii::new(
        radii.top_left() * scale,
        radii.top_right() * scale,
        radii.bottom_right() * scale,
        radii.bottom_left() * scale,
    )
}

fn corner_scale(available: f64, requested: f64) -> f64 {
    if requested <= available || requested == 0.0 {
        1.0
    } else {
        available / requested
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

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundBlendList {
    modes: Vec<BackgroundBlendMode>,
}

impl BackgroundBlendList {
    pub fn try_new(modes: Vec<BackgroundBlendMode>) -> Result<Self> {
        if modes.is_empty() {
            return Err(Error::invalid_value(
                "background blend list",
                "[]",
                "must contain at least one mode",
            ));
        }
        if modes
            .iter()
            .any(|mode| *mode != BackgroundBlendMode::Normal)
        {
            return Err(Error::unsupported_render_primitive(
                UnsupportedPrimitive::new(
                    PrimitiveFamily::Compositing,
                    PrimitiveOperation::BackgroundBlendMode,
                ),
            ));
        }
        Ok(Self { modes })
    }

    #[must_use]
    pub fn modes(&self) -> &[BackgroundBlendMode] {
        &self.modes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundBlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Plus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundNormalizationInput {
    stack: BackgroundStack,
    areas: BackgroundAreas,
    layer_clip_overrides: Vec<Option<BackgroundClipGeometry>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundStack {
    commands: Vec<NormalizedBackgroundCommand>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundCommand {
    clip: BackgroundClipGeometry,
    kind: NormalizedBackgroundCommandKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundLayer {
    source: NormalizedBackgroundLayerSource,
    placement: ResolvedImagePlacement,
    repeat: ResolvedImageRepeat,
    attachment: ImageAttachmentPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBackgroundLayerSource {
    Paint(Paint),
    Image(Image),
    ResolvedImage(ResolvedImageResource),
}

#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "background commands keep their planned public shape for direct matching"
)]
pub enum NormalizedBackgroundCommandKind {
    ColorFill { rect: Rect, color: Color },
    Layer { layer: NormalizedBackgroundLayer },
}

impl BackgroundNormalizationInput {
    pub fn try_new(stack: BackgroundStack, areas: BackgroundAreas) -> Result<Self> {
        let layer_clip_overrides = vec![None; stack.layers().len()];
        Ok(Self {
            stack,
            areas,
            layer_clip_overrides,
        })
    }

    #[must_use]
    pub fn stack(&self) -> &BackgroundStack {
        &self.stack
    }

    #[must_use]
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    #[must_use]
    pub fn layer_clip_overrides(&self) -> &[Option<BackgroundClipGeometry>] {
        &self.layer_clip_overrides
    }

    pub fn with_layer_clip_overrides(
        mut self,
        layer_clip_overrides: Vec<Option<BackgroundClipGeometry>>,
    ) -> Result<Self> {
        if layer_clip_overrides.len() != self.stack.layers().len() {
            return Err(Error::invalid_value(
                "background layer clip overrides",
                layer_clip_overrides.len(),
                "must match background layer count",
            ));
        }
        self.layer_clip_overrides = layer_clip_overrides;
        Ok(self)
    }

    pub fn normalize(&self, capabilities: Capabilities) -> Result<NormalizedBackgroundStack> {
        let mut commands = Vec::new();
        if let Some(color) = self.stack.color() {
            let rect = self.areas.border_box();
            commands.push(NormalizedBackgroundCommand {
                clip: BackgroundClipGeometry::try_rect(rect)?,
                kind: NormalizedBackgroundCommandKind::ColorFill { rect, color },
            });
        }

        for (layer_index, layer) in self.stack.layers().iter().enumerate().rev() {
            commands.push(self.normalize_layer(layer_index, layer.image(), capabilities)?);
        }

        Ok(NormalizedBackgroundStack { commands })
    }

    fn normalize_layer(
        &self,
        layer_index: usize,
        layer: &StyleImageLayer,
        capabilities: Capabilities,
    ) -> Result<NormalizedBackgroundCommand> {
        let clip = self.layer_clip_geometry(layer_index, layer)?;
        let origin_rect = self.areas.rect_for(layer.origin());
        let (source, intrinsic_size) = match layer.source().kind() {
            StyleImageSourceKind::Paint(paint) => {
                validate_paint(paint)?;
                (
                    NormalizedBackgroundLayerSource::Paint(paint.clone()),
                    origin_rect.size(),
                )
            }
            StyleImageSourceKind::Image(image) => (
                NormalizedBackgroundLayerSource::Image(image.clone()),
                image.size(),
            ),
            StyleImageSourceKind::Resolved(resource) => (
                NormalizedBackgroundLayerSource::ResolvedImage(resource.clone()),
                resource.intrinsic_size(),
            ),
            StyleImageSourceKind::Unresolved(_) => {
                layer.source().require_resolved()?;
                unreachable!("unresolved image sources return an error")
            }
        };
        let placement = ImagePlacementInput::try_new(
            origin_rect,
            intrinsic_size,
            layer.position(),
            layer.size(),
        )?
        .resolve()?;
        let repeat = ImageRepeatPlan::try_new(layer.repeat(), capabilities)?.resolve(placement)?;
        let attachment =
            ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space())?;
        Ok(NormalizedBackgroundCommand {
            clip,
            kind: NormalizedBackgroundCommandKind::Layer {
                layer: NormalizedBackgroundLayer {
                    source,
                    placement,
                    repeat,
                    attachment,
                },
            },
        })
    }

    fn layer_clip_geometry(
        &self,
        layer_index: usize,
        layer: &StyleImageLayer,
    ) -> Result<BackgroundClipGeometry> {
        if let Some(override_clip) = &self.layer_clip_overrides[layer_index] {
            return Ok(override_clip.clone());
        }
        BackgroundClipGeometry::try_rect(self.areas.rect_for(layer.clip()))
    }
}

impl NormalizedBackgroundStack {
    #[must_use]
    pub fn commands(&self) -> &[NormalizedBackgroundCommand] {
        &self.commands
    }
}

impl NormalizedBackgroundCommand {
    #[must_use]
    pub fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub fn kind(&self) -> &NormalizedBackgroundCommandKind {
        &self.kind
    }
}

impl NormalizedBackgroundLayer {
    #[must_use]
    pub fn source(&self) -> &NormalizedBackgroundLayerSource {
        &self.source
    }

    #[must_use]
    pub const fn placement(&self) -> ResolvedImagePlacement {
        self.placement
    }

    #[must_use]
    pub fn repeat(&self) -> &ResolvedImageRepeat {
        &self.repeat
    }

    #[must_use]
    pub const fn attachment(&self) -> ImageAttachmentPlan {
        self.attachment
    }
}
