use super::{Point, Radii, Rect, Size};

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(Path),
}

impl Shape {
    #[must_use]
    pub const fn rect(rect: Rect) -> Self {
        Self::Rect(rect)
    }

    #[must_use]
    pub const fn rounded_rect(rect: Rect, radii: Radii) -> Self {
        Self::RoundedRect { rect, radii }
    }
}

impl From<Rect> for Shape {
    fn from(rect: Rect) -> Self {
        Self::Rect(rect)
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
    pub width: f64,
    pub join: LineJoin,
    pub start_cap: LineCap,
    pub end_cap: LineCap,
    pub miter_limit: f64,
    pub dash: Option<Dash>,
    pub align: StrokeAlign,
}

impl Stroke {
    #[must_use]
    pub const fn new(width: f64) -> Self {
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

    #[must_use]
    pub const fn align(mut self, align: StrokeAlign) -> Self {
        self.align = align;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dash {
    pub offset: f64,
    pub intervals: &'static [f64],
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
                PathElement::MoveTo(point) => path.move_to((point.x, point.y)),
                PathElement::LineTo(point) => path.line_to((point.x, point.y)),
                PathElement::QuadTo(control, point) => {
                    path.quad_to((control.x, control.y), (point.x, point.y));
                }
                PathElement::CubicTo(a, b, point) => {
                    path.curve_to((a.x, a.y), (b.x, b.y), (point.x, point.y));
                }
                PathElement::Close => path.close_path(),
            }
        }
        path
    }
}
