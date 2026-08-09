use super::{
    Point, Radii, Rect, Result, Size,
    validation::{
        validate_dash, validate_finite_f64, validate_non_negative_f64, validate_path,
        validate_point, validate_positive_f64, validate_radii, validate_rect, validate_size,
    },
};

/// A renderer-facing logical-space shape with one closed geometry kind.
///
/// Public construction preserves finite point, size, radius, and path-element invariants.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    kind: ShapeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShapeKind {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(Path),
}

impl Shape {
    #[must_use]
    /// Creates an axis-aligned rectangle shape.
    pub const fn rect(rect: Rect) -> Self {
        Self {
            kind: ShapeKind::Rect(rect),
        }
    }

    #[must_use]
    /// Creates a rounded rectangle from already validated geometry.
    pub const fn rounded_rect(rect: Rect, radii: Radii) -> Self {
        Self {
            kind: ShapeKind::RoundedRect { rect, radii },
        }
    }

    /// Canonically validates and creates a rounded rectangle.
    ///
    /// Invalid rectangle maxima or negative or non-finite radii return a typed
    /// input diagnostic.
    pub fn try_rounded_rect(rect: Rect, radii: Radii) -> Result<Self> {
        validate_rect(rect, "rounded rectangle")?;
        validate_radii(radii, "rounded rectangle radii")?;
        Ok(Self::rounded_rect(rect, radii))
    }

    /// Creates a circle with a finite center and finite, non-negative logical radius.
    ///
    /// Invalid geometry returns [`crate::ErrorCode::InvalidInput`].
    pub fn try_circle(center: Point, radius: f64) -> Result<Self> {
        validate_point(center, "circle center")?;
        validate_non_negative_f64(radius, "circle radius")?;
        Ok(Self {
            kind: ShapeKind::Circle { center, radius },
        })
    }

    /// Creates an ellipse from a finite center and non-negative logical axis radii.
    ///
    /// Invalid geometry returns [`crate::ErrorCode::InvalidInput`]; zero radii are valid.
    pub fn try_ellipse(center: Point, radii: Size) -> Result<Self> {
        validate_point(center, "ellipse center")?;
        validate_size(radii, "ellipse radii")?;
        Ok(Self {
            kind: ShapeKind::Ellipse { center, radii },
        })
    }

    #[must_use]
    /// Creates a shape from an authored logical path without changing its elements.
    pub fn path(path: Path) -> Self {
        Self {
            kind: ShapeKind::Path(path),
        }
    }

    pub(crate) const fn kind(&self) -> &ShapeKind {
        &self.kind
    }
}

/// Converts a logical rectangle to its corresponding shape without loss.
impl From<Rect> for Shape {
    fn from(rect: Rect) -> Self {
        Self::rect(rect)
    }
}

/// An authored ordered sequence of logical-space path elements.
///
/// The default path is empty. Builder methods append elements without implicit
/// closing, normalization, or coordinate conversion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    elements: Vec<PathElement>,
}

impl Path {
    #[must_use]
    /// Creates an empty path.
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    /// Returns the authored elements in append order.
    pub fn elements(&self) -> &[PathElement] {
        &self.elements
    }

    /// Appends a new-subpath move to a finite logical point.
    pub fn move_to(&mut self, point: Point) -> &mut Self {
        self.elements.push(PathElement::MoveTo(point));
        self
    }

    /// Appends a straight segment to a finite logical point.
    pub fn line_to(&mut self, point: Point) -> &mut Self {
        self.elements.push(PathElement::LineTo(point));
        self
    }

    /// Appends a quadratic Bézier segment with one control point and endpoint.
    pub fn quad_to(&mut self, control: Point, point: Point) -> &mut Self {
        self.elements.push(PathElement::QuadTo(control, point));
        self
    }

    /// Appends a cubic Bézier segment with two control points and an endpoint.
    pub fn cubic_to(&mut self, a: Point, b: Point, point: Point) -> &mut Self {
        self.elements.push(PathElement::CubicTo(a, b, point));
        self
    }

    /// Appends an instruction that closes the current subpath.
    pub fn close(&mut self) -> &mut Self {
        self.elements.push(PathElement::Close);
        self
    }
}

/// One authored logical-space path instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathElement {
    /// Start a new subpath at the supplied point.
    MoveTo(Point),
    /// Add a straight segment ending at the supplied point.
    LineTo(Point),
    /// Add a quadratic segment with `(control, endpoint)`.
    QuadTo(Point, Point),
    /// Add a cubic segment with `(first control, second control, endpoint)`.
    CubicTo(Point, Point, Point),
    /// Close the current subpath.
    Close,
}

/// Rule for determining which regions of a path are filled.
///
/// The default is [`Self::NonZero`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FillRule {
    /// Use the non-zero winding rule.
    #[default]
    NonZero,
    /// Use the even-odd crossing rule.
    EvenOdd,
}

/// A canonically validated authored path paired with its fill rule.
#[derive(Clone, Debug, PartialEq)]
pub struct FilledPath {
    path: Path,
    fill_rule: FillRule,
}

impl FilledPath {
    /// Validates every path element and preserves the authored fill rule.
    ///
    /// Returns a typed input diagnostic for any non-finite path coordinate.
    pub fn try_new(path: Path, fill_rule: FillRule) -> Result<Self> {
        validate_path(&path)?;
        Ok(Self { path, fill_rule })
    }

    #[must_use]
    /// Returns the validated path.
    pub const fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    /// Returns the path's fill rule.
    pub const fn fill_rule(&self) -> FillRule {
        self.fill_rule
    }
}

/// Validated logical-space stroke parameters.
///
/// Width and miter limit are finite and strictly positive. The default
/// configuration created by [`Self::try_new`] uses a miter join, butt caps, a
/// miter limit of `4`, no dash, and centered alignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    width: f64,
    join: LineJoin,
    start_cap: LineCap,
    end_cap: LineCap,
    miter_limit: f64,
    dash: Option<Dash>,
    align: StrokeAlign,
}

impl Stroke {
    #[must_use]
    const fn new(width: f64) -> Self {
        Self {
            width,
            join: LineJoin::Miter,
            start_cap: LineCap::Butt,
            end_cap: LineCap::Butt,
            miter_limit: 4.0,
            dash: None,
            align: StrokeAlign::Center,
        }
    }

    /// Creates a centered solid stroke with a finite, positive logical width.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for zero, negative, or non-finite width.
    pub fn try_new(width: f64) -> Result<Self> {
        validate_positive_f64(width, "stroke width")?;
        Ok(Self::new(width))
    }

    #[must_use]
    /// Returns the stroke width in logical units.
    pub const fn width(self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns this stroke with a different alignment relative to the path.
    pub const fn align(mut self, align: StrokeAlign) -> Self {
        self.align = align;
        self
    }

    #[must_use]
    /// Returns this stroke with a different line join.
    pub const fn join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    #[must_use]
    /// Returns this stroke with separate start and end caps.
    pub const fn caps(mut self, start: LineCap, end: LineCap) -> Self {
        self.start_cap = start;
        self.end_cap = end;
        self
    }

    /// Returns this stroke with a finite, strictly positive miter limit.
    ///
    /// Invalid values return [`crate::ErrorCode::InvalidInput`].
    pub fn try_miter_limit(mut self, miter_limit: f64) -> Result<Self> {
        validate_positive_f64(miter_limit, "stroke miter limit")?;
        self.miter_limit = miter_limit;
        Ok(self)
    }

    /// Returns this stroke with a validated dash pattern.
    ///
    /// A non-finite offset or negative or non-finite interval returns
    /// [`crate::ErrorCode::InvalidInput`].
    pub fn try_dash(mut self, dash: Dash) -> Result<Self> {
        validate_dash(dash)?;
        self.dash = Some(dash);
        Ok(self)
    }

    #[must_use]
    /// Returns the configured line join.
    pub const fn join_kind(self) -> LineJoin {
        self.join
    }

    #[must_use]
    /// Returns the start cap.
    pub const fn start_cap(self) -> LineCap {
        self.start_cap
    }

    #[must_use]
    /// Returns the end cap.
    pub const fn end_cap(self) -> LineCap {
        self.end_cap
    }

    #[must_use]
    /// Returns the positive miter limit.
    pub const fn miter_limit(self) -> f64 {
        self.miter_limit
    }

    #[must_use]
    /// Returns the dash pattern, if configured.
    pub const fn dash(self) -> Option<Dash> {
        self.dash
    }

    #[must_use]
    /// Returns the stroke alignment.
    pub const fn align_kind(self) -> StrokeAlign {
        self.align
    }
}

/// A logical-space stroke dash offset and static interval pattern.
///
/// The offset is finite and each interval is finite and non-negative. Empty
/// patterns and zero-length intervals are currently accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dash {
    offset: f64,
    intervals: &'static [f64],
}

impl Dash {
    /// Creates a dash pattern from a finite offset and non-negative intervals.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for any non-finite value or
    /// negative interval. The interval slice must have static lifetime.
    pub fn try_new(offset: f64, intervals: &'static [f64]) -> Result<Self> {
        validate_finite_f64(offset, "dash offset")?;
        for interval in intervals {
            validate_non_negative_f64(*interval, "dash interval")?;
        }
        Ok(Self { offset, intervals })
    }

    #[must_use]
    /// Returns the logical dash phase offset.
    pub const fn offset(self) -> f64 {
        self.offset
    }

    #[must_use]
    /// Returns the logical interval sequence.
    pub const fn intervals(self) -> &'static [f64] {
        self.intervals
    }
}

/// Geometry used where consecutive stroke segments meet.
///
/// The default is [`Self::Miter`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LineJoin {
    /// Extend segment edges until they meet, subject to the stroke's miter limit.
    #[default]
    Miter,
    /// Join segments with a round arc.
    Round,
    /// Join segments with a bevel edge.
    Bevel,
}

/// Converts the public line join to its equivalent Kurbo value.
impl From<LineJoin> for kurbo::Join {
    fn from(join: LineJoin) -> Self {
        match join {
            LineJoin::Miter => Self::Miter,
            LineJoin::Round => Self::Round,
            LineJoin::Bevel => Self::Bevel,
        }
    }
}

/// Geometry applied to an open stroke endpoint.
///
/// The default is [`Self::Butt`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LineCap {
    /// End at the endpoint without extension.
    #[default]
    Butt,
    /// Extend the endpoint with a semicircle.
    Round,
    /// Extend the endpoint with a half-width square.
    Square,
}

/// Converts the public line cap to its equivalent Kurbo value.
impl From<LineCap> for kurbo::Cap {
    fn from(cap: LineCap) -> Self {
        match cap {
            LineCap::Butt => Self::Butt,
            LineCap::Round => Self::Round,
            LineCap::Square => Self::Square,
        }
    }
}

/// Placement of a stroke relative to its source path.
///
/// The default is [`Self::Center`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StrokeAlign {
    /// Center the stroke on the path.
    #[default]
    Center,
    /// Place the stroke on the path's inside side.
    Inside,
    /// Place the stroke on the path's outside side.
    Outside,
}

impl Path {
    pub(crate) fn to_kurbo(&self) -> kurbo::BezPath {
        let mut path = kurbo::BezPath::new();
        for element in &self.elements {
            match element {
                PathElement::MoveTo(point) => path.move_to((point.x(), point.y())),
                PathElement::LineTo(point) => path.line_to((point.x(), point.y())),
                PathElement::QuadTo(control, point) => {
                    path.quad_to((control.x(), control.y()), (point.x(), point.y()));
                }
                PathElement::CubicTo(a, b, point) => {
                    path.curve_to((a.x(), a.y()), (b.x(), b.y()), (point.x(), point.y()));
                }
                PathElement::Close => path.close_path(),
            }
        }
        path
    }
}
