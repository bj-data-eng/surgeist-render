use super::{
    Point, Radii, Rect, Result, Size,
    validation::{
        validate_dash, validate_finite_f64, validate_non_negative_f64, validate_point,
        validate_positive_f64, validate_radii, validate_rect, validate_size,
    },
};

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
    pub const fn rect(rect: Rect) -> Self {
        Self {
            kind: ShapeKind::Rect(rect),
        }
    }

    #[must_use]
    pub const fn rounded_rect(rect: Rect, radii: Radii) -> Self {
        Self {
            kind: ShapeKind::RoundedRect { rect, radii },
        }
    }

    pub fn try_rounded_rect(rect: Rect, radii: Radii) -> Result<Self> {
        validate_rect(rect, "rounded rectangle")?;
        validate_radii(radii, "rounded rectangle radii")?;
        Ok(Self::rounded_rect(rect, radii))
    }

    pub fn try_circle(center: Point, radius: f64) -> Result<Self> {
        validate_point(center, "circle center")?;
        validate_non_negative_f64(radius, "circle radius")?;
        Ok(Self {
            kind: ShapeKind::Circle { center, radius },
        })
    }

    pub fn try_ellipse(center: Point, radii: Size) -> Result<Self> {
        validate_point(center, "ellipse center")?;
        validate_size(radii, "ellipse radii")?;
        Ok(Self {
            kind: ShapeKind::Ellipse { center, radii },
        })
    }

    #[must_use]
    pub fn path(path: Path) -> Self {
        Self {
            kind: ShapeKind::Path(path),
        }
    }

    pub(crate) const fn kind(&self) -> &ShapeKind {
        &self.kind
    }
}

impl From<Rect> for Shape {
    fn from(rect: Rect) -> Self {
        Self::rect(rect)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    pub(crate) elements: Vec<PathElement>,
}

impl Path {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn move_to(&mut self, point: Point) -> &mut Self {
        self.elements.push(PathElement::MoveTo(point));
        self
    }

    pub fn line_to(&mut self, point: Point) -> &mut Self {
        self.elements.push(PathElement::LineTo(point));
        self
    }

    pub fn quad_to(&mut self, control: Point, point: Point) -> &mut Self {
        self.elements.push(PathElement::QuadTo(control, point));
        self
    }

    pub fn cubic_to(&mut self, a: Point, b: Point, point: Point) -> &mut Self {
        self.elements.push(PathElement::CubicTo(a, b, point));
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.elements.push(PathElement::Close);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathElement {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CubicTo(Point, Point, Point),
    Close,
}

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

    pub fn try_new(width: f64) -> Result<Self> {
        validate_positive_f64(width, "stroke width")?;
        Ok(Self::new(width))
    }

    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn align(mut self, align: StrokeAlign) -> Self {
        self.align = align;
        self
    }

    #[must_use]
    pub const fn join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    #[must_use]
    pub const fn caps(mut self, start: LineCap, end: LineCap) -> Self {
        self.start_cap = start;
        self.end_cap = end;
        self
    }

    pub fn try_miter_limit(mut self, miter_limit: f64) -> Result<Self> {
        validate_positive_f64(miter_limit, "stroke miter limit")?;
        self.miter_limit = miter_limit;
        Ok(self)
    }

    pub fn try_dash(mut self, dash: Dash) -> Result<Self> {
        validate_dash(dash)?;
        self.dash = Some(dash);
        Ok(self)
    }

    #[must_use]
    pub const fn join_kind(self) -> LineJoin {
        self.join
    }

    #[must_use]
    pub const fn start_cap(self) -> LineCap {
        self.start_cap
    }

    #[must_use]
    pub const fn end_cap(self) -> LineCap {
        self.end_cap
    }

    #[must_use]
    pub const fn miter_limit(self) -> f64 {
        self.miter_limit
    }

    #[must_use]
    pub const fn dash(self) -> Option<Dash> {
        self.dash
    }

    #[must_use]
    pub const fn align_kind(self) -> StrokeAlign {
        self.align
    }

    pub(crate) const fn parts(
        self,
    ) -> (
        f64,
        LineJoin,
        LineCap,
        LineCap,
        f64,
        Option<Dash>,
        StrokeAlign,
    ) {
        (
            self.width,
            self.join,
            self.start_cap,
            self.end_cap,
            self.miter_limit,
            self.dash,
            self.align,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dash {
    offset: f64,
    intervals: &'static [f64],
}

impl Dash {
    pub fn try_new(offset: f64, intervals: &'static [f64]) -> Result<Self> {
        validate_finite_f64(offset, "dash offset")?;
        for interval in intervals {
            validate_non_negative_f64(*interval, "dash interval")?;
        }
        Ok(Self { offset, intervals })
    }

    #[must_use]
    pub const fn offset(self) -> f64 {
        self.offset
    }

    #[must_use]
    pub const fn intervals(self) -> &'static [f64] {
        self.intervals
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}

impl From<LineJoin> for kurbo::Join {
    fn from(join: LineJoin) -> Self {
        match join {
            LineJoin::Miter => Self::Miter,
            LineJoin::Round => Self::Round,
            LineJoin::Bevel => Self::Bevel,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}

impl From<LineCap> for kurbo::Cap {
    fn from(cap: LineCap) -> Self {
        match cap {
            LineCap::Butt => Self::Butt,
            LineCap::Round => Self::Round,
            LineCap::Square => Self::Square,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StrokeAlign {
    #[default]
    Center,
    Inside,
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
