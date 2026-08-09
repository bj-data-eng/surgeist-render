use super::super::{
    Capabilities, CoordinateSpaceKind, CoordinateSpaceTag, Error, Image, ImageColorProfilePolicy,
    ImageId, ImageOrientationPolicy, Paint, PrimitiveFamily, PrimitiveOperation, Rect, Result,
    Shape, Size, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
    validation::{validate_finite_f64, validate_paint, validate_shape, validate_size},
};

const MAX_IMAGE_REPEAT_TILES: usize = 1_000_000;
const MAX_IMAGE_REPEAT_TILES_RULE: &str = "must not exceed 1000000";

#[derive(Clone, Debug, Eq, PartialEq)]
/// A symbolic reference to a style resource that requires surrounding context.
///
/// Equality compares the stored identifier exactly. Construction preserves the
/// caller's string, including non-empty surrounding whitespace.
pub struct StyleResourceRef {
    identifier: String,
}

impl StyleResourceRef {
    /// Creates a symbolic resource reference from a non-blank identifier.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the identifier is empty
    /// or consists only of whitespace.
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

    /// Returns the stored identifier exactly as supplied.
    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

#[derive(Clone, Debug, PartialEq)]
/// An image resource already resolved by the surrounding rendering context.
///
/// The descriptor carries an opaque image handle, finite non-negative logical
/// intrinsic dimensions, and optional intrinsic density. It does not carry pixel
/// bytes or perform resource lookup.
pub struct ResolvedImageResource {
    id: ImageId,
    intrinsic_size: Size,
    density: Option<ImageResourceDensity>,
}

impl ResolvedImageResource {
    /// Creates a resolved image descriptor.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when either intrinsic dimension
    /// is negative or non-finite. Zero dimensions remain representable here,
    /// although placement requires positive intrinsic dimensions.
    pub fn try_new(id: ImageId, intrinsic_size: Size) -> Result<Self> {
        validate_size(intrinsic_size, "resolved image intrinsic size")?;
        Ok(Self {
            id,
            intrinsic_size,
            density: None,
        })
    }

    /// Returns the opaque image resource handle or compact fingerprint.
    #[must_use]
    pub const fn id(&self) -> ImageId {
        self.id
    }

    /// Returns the finite non-negative intrinsic size in logical rendering units.
    #[must_use]
    pub const fn intrinsic_size(&self) -> Size {
        self.intrinsic_size
    }

    /// Associates a validated positive density with the resource.
    #[must_use]
    pub const fn with_density(mut self, density: ImageResourceDensity) -> Self {
        self.density = Some(density);
        self
    }

    /// Returns the optional intrinsic resource density.
    #[must_use]
    pub const fn density(&self) -> Option<ImageResourceDensity> {
        self.density
    }

    /// Returns the current requirement for resolving image orientation.
    #[must_use]
    pub const fn orientation_policy(&self) -> ImageOrientationPolicy {
        ImageOrientationPolicy::RootResolvedOnly
    }

    /// Returns the current requirement for resolving image color profiles.
    #[must_use]
    pub const fn color_profile_policy(&self) -> ImageColorProfilePolicy {
        ImageColorProfilePolicy::RootResolvedOnly
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A finite positive density scalar attached to a resolved image resource.
pub struct ImageResourceDensity {
    value: f64,
}

impl ImageResourceDensity {
    /// Creates a density from a finite value greater than zero.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for zero, negative, infinite,
    /// or NaN values.
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

    /// Returns the positive density scalar.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
/// An authored image-style source in concrete, resolved, or symbolic form.
pub struct StyleImageSource {
    kind: StyleImageSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
/// The phase and payload represented by a [`StyleImageSource`].
pub enum StyleImageSourceKind {
    /// Concrete validated image pixels and sampling configuration.
    Image(Image),
    /// A resource descriptor already resolved by the surrounding context.
    Resolved(ResolvedImageResource),
    /// A validated concrete paint source.
    Paint(Paint),
    /// A symbolic resource reference that cannot be normalized directly.
    Unresolved(StyleResourceRef),
}

impl StyleImageSource {
    /// Creates a source from a concrete image after validating its size.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if the image dimensions are
    /// negative or non-finite.
    pub fn image(image: Image) -> Result<Self> {
        validate_size(image.size(), "image size")?;
        Ok(Self {
            kind: StyleImageSourceKind::Image(image),
        })
    }

    /// Creates a source from a validated concrete paint.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when paint validation fails.
    pub fn paint(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(Self {
            kind: StyleImageSourceKind::Paint(paint),
        })
    }

    /// Creates a source from a context-resolved image descriptor.
    #[must_use]
    pub const fn resolved(resource: ResolvedImageResource) -> Self {
        Self {
            kind: StyleImageSourceKind::Resolved(resource),
        }
    }

    /// Creates a symbolic image-resource source.
    #[must_use]
    pub fn unresolved(reference: StyleResourceRef) -> Self {
        Self {
            kind: StyleImageSourceKind::Unresolved(reference),
        }
    }

    /// Requires the source to be concrete or context-resolved.
    ///
    /// Returns an unresolved image-resource diagnostic only for
    /// [`StyleImageSourceKind::Unresolved`].
    pub fn require_resolved(&self) -> Result<()> {
        if let StyleImageSourceKind::Unresolved(reference) = &self.kind {
            return Err(Error::unresolved_resource(UnresolvedResource::new(
                UnresolvedResourceKind::Image,
                reference.identifier(),
            )));
        }
        Ok(())
    }

    /// Returns the source phase and payload.
    #[must_use]
    pub const fn kind(&self) -> &StyleImageSourceKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
/// An authored image layer with symbolic placement, repetition, and attachment.
///
/// New layers start at position `0% 0%`, use intrinsic sizing, repeat on both
/// axes, originate in the padding box, clip to the border box, scroll with the
/// content, and have no explicit coordinate-space tag.
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
    /// Creates an image layer with the documented authored defaults.
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

    /// Replaces the authored background position.
    #[must_use]
    pub fn with_position(mut self, position: BackgroundPosition) -> Self {
        self.position = position;
        self
    }

    /// Replaces the authored background size.
    #[must_use]
    pub fn with_size(mut self, size: BackgroundSize) -> Self {
        self.size = size;
        self
    }

    /// Replaces the authored repeat modes.
    #[must_use]
    pub fn with_repeat(mut self, repeat: BackgroundRepeat) -> Self {
        self.repeat = repeat;
        self
    }

    /// Replaces the box used as the placement area.
    #[must_use]
    pub fn with_origin(mut self, origin: BackgroundBox) -> Self {
        self.origin = origin;
        self
    }

    /// Replaces the box used to derive default clip geometry.
    #[must_use]
    pub fn with_clip(mut self, clip: BackgroundBox) -> Self {
        self.clip = clip;
        self
    }

    /// Replaces the authored attachment choice.
    #[must_use]
    pub fn with_attachment(mut self, attachment: BackgroundAttachment) -> Self {
        self.attachment = attachment;
        self
    }

    /// Associates the layer with a tagged coordinate space.
    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    /// Returns the image-style source.
    #[must_use]
    pub const fn source(&self) -> &StyleImageSource {
        &self.source
    }

    /// Returns the authored position.
    #[must_use]
    pub const fn position(&self) -> BackgroundPosition {
        self.position
    }

    /// Returns the authored size.
    #[must_use]
    pub const fn size(&self) -> BackgroundSize {
        self.size
    }

    /// Returns the authored repeat modes.
    #[must_use]
    pub const fn repeat(&self) -> BackgroundRepeat {
        self.repeat
    }

    /// Returns the placement-origin box.
    #[must_use]
    pub const fn origin(&self) -> BackgroundBox {
        self.origin
    }

    /// Returns the default clipping box.
    #[must_use]
    pub const fn clip(&self) -> BackgroundBox {
        self.clip
    }

    /// Returns the authored attachment choice.
    #[must_use]
    pub const fn attachment(&self) -> BackgroundAttachment {
        self.attachment
    }

    /// Returns the optional coordinate-space tag.
    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A symbolic two-axis image position.
///
/// Percent values are fractions of the available space after subtracting the
/// tile size: `0.0` aligns the start and `1.0` aligns the end. Lengths and edge
/// offsets are in logical rendering units.
pub struct BackgroundPosition {
    x: PositionComponent,
    y: PositionComponent,
}

impl BackgroundPosition {
    /// Creates a two-axis fractional position from finite values.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if either value is non-finite.
    pub fn percent(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_percent_for(x, "background position x percent")?,
            y: PositionComponent::try_percent_for(y, "background position y percent")?,
        })
    }

    /// Creates a two-axis position in finite logical lengths.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if either value is non-finite.
    pub fn length(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_length_for(x, "background position x length")?,
            y: PositionComponent::try_length_for(y, "background position y length")?,
        })
    }

    /// Creates a position from independently authored axis components.
    pub fn components(x: PositionComponent, y: PositionComponent) -> Self {
        Self { x, y }
    }

    /// Creates a position from explicit start- or end-edge offsets on both axes.
    #[must_use]
    pub const fn edge_offsets(x: PositionEdgeOffset, y: PositionEdgeOffset) -> Self {
        Self {
            x: PositionComponent::edge_offset(x),
            y: PositionComponent::edge_offset(y),
        }
    }

    /// Returns the horizontal component.
    #[must_use]
    pub const fn x(self) -> PositionComponent {
        self.x
    }

    /// Returns the vertical component.
    #[must_use]
    pub const fn y(self) -> PositionComponent {
        self.y
    }
}

impl Default for BackgroundPosition {
    /// Returns the start-aligned `0% 0%` position.
    fn default() -> Self {
        Self {
            x: PositionComponent::percent_unchecked(0.0),
            y: PositionComponent::percent_unchecked(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// One finite symbolic image-position component.
///
/// The stored value is a logical length for [`PositionComponentKind::Length`]
/// and edge offsets, or a fractional multiplier for percent positioning.
pub struct PositionComponent {
    kind: PositionComponentKind,
    value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The interpretation of a [`PositionComponent`] value.
pub enum PositionComponentKind {
    /// A signed offset in logical rendering units from the axis origin.
    Length,
    /// A fractional multiplier of the available axis space.
    Percent,
    /// A signed logical offset from the selected axis edge.
    EdgeOffset(PositionEdge),
}

impl PositionComponent {
    /// Creates a fractional position component from a finite value.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `value` is non-finite.
    pub fn try_percent(value: f64) -> Result<Self> {
        Self::try_percent_for(value, "background position percent")
    }

    /// Creates a logical-length position component from a finite value.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `value` is non-finite.
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

    /// Converts a validated edge offset into a position component without loss.
    #[must_use]
    pub const fn edge_offset(offset: PositionEdgeOffset) -> Self {
        Self {
            kind: PositionComponentKind::EdgeOffset(offset.edge()),
            value: offset.offset(),
        }
    }

    /// Returns how the stored value is interpreted.
    #[must_use]
    pub const fn kind(self) -> PositionComponentKind {
        self.kind
    }

    /// Returns the finite logical length or fractional multiplier.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The reference edge for an image-position offset.
pub enum PositionEdge {
    /// The axis start edge.
    Start,
    /// The axis end edge.
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A finite signed logical offset from a selected position edge.
pub struct PositionEdgeOffset {
    edge: PositionEdge,
    offset: f64,
}

impl PositionEdgeOffset {
    /// Creates an offset from the start edge.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `offset` is non-finite.
    pub fn start(offset: f64) -> Result<Self> {
        Self::try_new(PositionEdge::Start, offset)
    }

    /// Creates an offset from the end edge.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `offset` is non-finite.
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

    /// Returns the reference edge.
    #[must_use]
    pub const fn edge(self) -> PositionEdge {
        self.edge
    }

    /// Returns the finite signed offset in logical rendering units.
    #[must_use]
    pub const fn offset(self) -> f64 {
        self.offset
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A symbolic image-tile sizing choice.
pub struct BackgroundSize {
    kind: BackgroundSizeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// The sizing rule represented by [`BackgroundSize`].
pub enum BackgroundSizeKind {
    /// Uses the image's intrinsic width and height.
    Auto,
    /// Preserves aspect ratio while covering the entire paint rectangle.
    Cover,
    /// Preserves aspect ratio while fitting entirely inside the paint rectangle.
    Contain,
    /// Resolves width and height independently from authored components.
    Explicit {
        /// The symbolic horizontal size.
        width: SizeComponent,
        /// The symbolic vertical size.
        height: SizeComponent,
    },
}

impl BackgroundSize {
    /// Creates intrinsic automatic sizing.
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: BackgroundSizeKind::Auto,
        }
    }

    /// Creates aspect-preserving cover sizing.
    #[must_use]
    pub const fn cover() -> Self {
        Self {
            kind: BackgroundSizeKind::Cover,
        }
    }

    /// Creates aspect-preserving contain sizing.
    #[must_use]
    pub const fn contain() -> Self {
        Self {
            kind: BackgroundSizeKind::Contain,
        }
    }

    /// Creates explicit two-axis symbolic sizing.
    #[must_use]
    pub const fn explicit(width: SizeComponent, height: SizeComponent) -> Self {
        Self {
            kind: BackgroundSizeKind::Explicit { width, height },
        }
    }

    /// Returns the symbolic sizing rule.
    #[must_use]
    pub const fn kind(self) -> BackgroundSizeKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// One validated symbolic image-size component.
pub struct SizeComponent {
    kind: SizeComponentKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// The interpretation of one image-size axis.
pub enum SizeComponentKind {
    /// Uses the intrinsic aspect ratio and the other resolved axis when present.
    Auto,
    /// A finite non-negative length in logical rendering units.
    Length(f64),
    /// A finite non-negative fraction of the corresponding paint-rectangle axis.
    Percent(f64),
}

impl SizeComponent {
    /// Creates an automatic size component.
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: SizeComponentKind::Auto,
        }
    }

    /// Creates a logical length component.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `value` is negative or
    /// non-finite. Zero is accepted here, but placement resolution requires a
    /// positive final tile size.
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

    /// Creates a fractional size component.
    ///
    /// `1.0` represents the full corresponding paint-rectangle axis. Returns
    /// [`crate::ErrorCode::InvalidInput`] when `value` is negative or non-finite.
    /// Zero is accepted here, but placement resolution requires a positive final tile size.
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

    /// Returns the symbolic size rule and value.
    #[must_use]
    pub const fn kind(self) -> SizeComponentKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Validated context for resolving symbolic image placement.
///
/// The paint rectangle and intrinsic size use logical rendering units and must
/// both have finite positive dimensions.
pub struct ImagePlacementInput {
    paint_rect: Rect,
    intrinsic_size: Size,
    position: BackgroundPosition,
    size: BackgroundSize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Concrete logical geometry for an image paint area and its first tile.
///
/// Both rectangles have finite origins and finite positive dimensions.
pub struct ResolvedImagePlacement {
    paint_rect: Rect,
    tile_rect: Rect,
}

impl ImagePlacementInput {
    /// Creates placement context from logical geometry and authored choices.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] unless the paint rectangle and
    /// intrinsic size have finite positive dimensions and the rectangle origin
    /// is finite.
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

    /// Returns the logical rectangle available for painting.
    #[must_use]
    pub const fn paint_rect(self) -> Rect {
        self.paint_rect
    }

    /// Returns the positive intrinsic image size in logical units.
    #[must_use]
    pub const fn intrinsic_size(self) -> Size {
        self.intrinsic_size
    }

    /// Returns the authored symbolic position.
    #[must_use]
    pub const fn position(self) -> BackgroundPosition {
        self.position
    }

    /// Returns the authored symbolic size.
    #[must_use]
    pub const fn size(self) -> BackgroundSize {
        self.size
    }

    /// Resolves symbolic size and position against the stored logical geometry.
    ///
    /// The result contains the paint rectangle and first tile rectangle. Returns
    /// [`crate::ErrorCode::InvalidInput`] if arithmetic produces a non-finite or
    /// non-positive tile rectangle.
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
    /// Validates concrete paint and first-tile rectangles.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] unless both rectangles have
    /// finite origins and finite positive dimensions.
    pub fn from_parts(paint_rect: Rect, tile_rect: Rect) -> Result<Self> {
        validate_placement_rect(paint_rect, "image placement paint rect")?;
        validate_placement_rect(tile_rect, "image placement tile rect")?;
        Ok(Self {
            paint_rect,
            tile_rect,
        })
    }

    /// Returns the logical clipping and repetition area.
    #[must_use]
    pub const fn paint_rect(self) -> Rect {
        self.paint_rect
    }

    /// Returns the concrete first-tile rectangle in logical coordinates.
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
/// Independent authored repetition choices for the horizontal and vertical axes.
pub struct BackgroundRepeat {
    x: RepeatMode,
    y: RepeatMode,
}

impl BackgroundRepeat {
    /// Creates repetition choices for both axes.
    #[must_use]
    pub const fn new(x: RepeatMode, y: RepeatMode) -> Self {
        Self { x, y }
    }

    /// Repeats tiles on both axes.
    #[must_use]
    pub const fn repeat() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::Repeat)
    }

    /// Repeats horizontally and draws a single vertical tile.
    #[must_use]
    pub const fn repeat_x() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::NoRepeat)
    }

    /// Draws a single horizontal tile and repeats vertically.
    #[must_use]
    pub const fn repeat_y() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::Repeat)
    }

    /// Draws at most one tile on each axis.
    #[must_use]
    pub const fn no_repeat() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::NoRepeat)
    }

    /// Returns the horizontal repeat choice.
    #[must_use]
    pub const fn x(self) -> RepeatMode {
        self.x
    }

    /// Returns the vertical repeat choice.
    #[must_use]
    pub const fn y(self) -> RepeatMode {
        self.y
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// An authored repetition choice for one image axis.
pub enum RepeatMode {
    /// Repeats fixed-size tiles to cover the clip extent.
    Repeat,
    /// Retains only the positioned tile when it intersects the clip extent.
    NoRepeat,
    /// Resizes tiles to fit an integer count; currently rejected as unsupported.
    Round,
    /// Distributes space between whole tiles; currently rejected as unsupported.
    Space,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// The intrinsically normalized repetition axes supported by the current planner.
pub enum ImageRepeatMode {
    /// Does not repeat either axis.
    NoRepeat,
    /// Repeats the horizontal axis only.
    RepeatX,
    /// Repeats the vertical axis only.
    RepeatY,
    /// Repeats both axes.
    RepeatBoth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// A validated repeat plan with authored and normalized axis choices.
///
/// Only `Repeat` and `NoRepeat` axis combinations can currently form this plan.
pub struct ImageRepeatPlan {
    repeat: BackgroundRepeat,
    mode: ImageRepeatMode,
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete image repetition geometry in logical rendering units.
///
/// Tile rectangles are ordered by increasing generated vertical position, with
/// horizontal positions varying inside each row. The list may be empty when a
/// non-repeated positioned tile does not intersect the clip rectangle.
pub struct ResolvedImageRepeat {
    clip_rect: Rect,
    tile_rects: Vec<Rect>,
}

impl ImageRepeatPlan {
    /// Normalizes supported authored repeat modes.
    ///
    /// `Round` and `Space` return their current unsupported-primitive diagnostic.
    /// The supplied capabilities are consulted for that semantic operation, but
    /// those modes do not currently produce an [`ImageRepeatPlan`].
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

    /// Returns the original authored repeat choices.
    #[must_use]
    pub const fn repeat(self) -> BackgroundRepeat {
        self.repeat
    }

    /// Returns the normalized repetition axes.
    #[must_use]
    pub const fn mode(self) -> ImageRepeatMode {
        self.mode
    }

    /// Resolves repetition against concrete placement geometry.
    ///
    /// The placement paint rectangle becomes the clip rectangle. Returns
    /// [`crate::ErrorCode::InvalidInput`] for invalid repetition arithmetic or
    /// when an axis or total tile count exceeds 1,000,000.
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
    /// Returns the logical rectangle that clips all generated tiles.
    #[must_use]
    pub const fn clip_rect(&self) -> Rect {
        self.clip_rect
    }

    /// Returns the ordered concrete tile rectangles.
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
/// A semantic box used for background placement or default clipping.
pub enum BackgroundBox {
    /// The border-box rectangle.
    Border,
    /// The padding-box rectangle.
    Padding,
    /// The content-box rectangle.
    Content,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Logical border, padding, and content rectangles used during normalization.
///
/// Each rectangle is validated independently as finite and positive-area; this
/// value does not enforce containment or nesting between the three rectangles.
pub struct BackgroundAreas {
    border_box: Rect,
    padding_box: Rect,
    content_box: Rect,
}

impl BackgroundAreas {
    /// Creates a set of independently validated background areas.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if any rectangle has a
    /// non-finite origin or a non-positive or non-finite dimension.
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

    /// Returns the logical border-box rectangle.
    #[must_use]
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    /// Returns the logical padding-box rectangle.
    #[must_use]
    pub const fn padding_box(self) -> Rect {
        self.padding_box
    }

    /// Returns the logical content-box rectangle.
    #[must_use]
    pub const fn content_box(self) -> Rect {
        self.content_box
    }

    /// Selects the rectangle for a semantic background box.
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
/// Validated concrete geometry used to clip a normalized background command.
pub struct BackgroundClipGeometry {
    kind: BackgroundClipGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
/// The concrete geometry choice stored by [`BackgroundClipGeometry`].
pub enum BackgroundClipGeometryKind {
    /// A finite positive-area logical rectangle.
    Rect(Rect),
    /// Validated logical shape geometry.
    Shape(Shape),
}

impl BackgroundClipGeometry {
    /// Creates rectangular clip geometry.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for a non-finite origin or a
    /// non-positive or non-finite dimension.
    pub fn try_rect(rect: Rect) -> Result<Self> {
        validate_background_rect(rect, "background clip rect")?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Rect(rect),
        })
    }

    /// Creates clip geometry from a validated shape.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when shape validation fails.
    pub fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Shape(shape),
        })
    }

    /// Returns the concrete geometry choice.
    #[must_use]
    pub fn kind(&self) -> &BackgroundClipGeometryKind {
        &self.kind
    }

    /// Returns the rectangle, if this is rectangular clip geometry.
    #[must_use]
    pub fn rect(&self) -> Option<Rect> {
        match &self.kind {
            BackgroundClipGeometryKind::Rect(rect) => Some(*rect),
            BackgroundClipGeometryKind::Shape(_) => None,
        }
    }

    /// Returns the shape, if this is shape clip geometry.
    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        match &self.kind {
            BackgroundClipGeometryKind::Rect(_) => None,
            BackgroundClipGeometryKind::Shape(shape) => Some(shape),
        }
    }
}

pub(super) fn validate_background_rect(rect: Rect, field: &str) -> Result<()> {
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
/// An authored relationship between a background layer and scrolling context.
pub enum BackgroundAttachment {
    /// Selects ordinary scrolling attachment.
    Scroll,
    /// Selects attachment to an explicitly tagged viewport coordinate space.
    Fixed,
    /// Selects element-local scrolling attachment.
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A validated attachment choice and optional coordinate-space context.
pub struct ImageAttachmentPlan {
    attachment: BackgroundAttachment,
    coordinate_space: Option<CoordinateSpaceTag>,
}

impl ImageAttachmentPlan {
    /// Validates an authored attachment and its optional coordinate-space tag.
    ///
    /// A fixed attachment requires a viewport tag. Missing or non-viewport
    /// context returns [`crate::ErrorCode::InvalidInput`]. Scroll and local
    /// attachments retain any supplied tag without this viewport restriction.
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

    /// Returns the authored attachment choice.
    #[must_use]
    pub const fn attachment(self) -> BackgroundAttachment {
        self.attachment
    }

    /// Returns the optional coordinate-space tag.
    #[must_use]
    pub const fn coordinate_space(self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}
