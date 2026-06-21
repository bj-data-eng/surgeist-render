#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    #[must_use]
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    #[must_use]
    pub const fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(width, height),
        }
    }

    #[must_use]
    pub fn min(self) -> Point {
        self.origin
    }

    #[must_use]
    pub fn max(self) -> Point {
        Point::new(
            self.origin.x + self.size.width,
            self.origin.y + self.size.height,
        )
    }
}

impl From<Rect> for kurbo::Rect {
    fn from(rect: Rect) -> Self {
        let max = rect.max();
        Self::new(rect.origin.x, rect.origin.y, max.x, max.y)
    }
}

impl From<kurbo::Rect> for Rect {
    fn from(rect: kurbo::Rect) -> Self {
        Self::new(rect.x0, rect.y0, rect.width(), rect.height())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Radii {
    pub top_left: f64,
    pub top_right: f64,
    pub bottom_right: f64,
    pub bottom_left: f64,
}

impl Radii {
    #[must_use]
    pub const fn all(radius: f64) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    #[must_use]
    pub fn uniform(self) -> Option<f64> {
        (self.top_left == self.top_right
            && self.top_left == self.bottom_right
            && self.top_left == self.bottom_left)
            .then_some(self.top_left)
    }
}

pub(crate) fn offset_radii(radii: Radii, offset: f64) -> Radii {
    Radii {
        top_left: (radii.top_left + offset).max(0.0),
        top_right: (radii.top_right + offset).max(0.0),
        bottom_right: (radii.bottom_right + offset).max(0.0),
        bottom_left: (radii.bottom_left + offset).max(0.0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform(pub [f64; 6]);

impl Transform {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    #[must_use]
    pub const fn translate(x: f64, y: f64) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, x, y])
    }

    #[must_use]
    pub const fn scale(x: f64, y: f64) -> Self {
        Self([x, 0.0, 0.0, y, 0.0, 0.0])
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Transform> for kurbo::Affine {
    fn from(transform: Transform) -> Self {
        Self::new(transform.0)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhysicalSize {
    pub width: u32,
    pub height: u32,
}

pub(crate) fn physical_size(size: Size, scale: f64) -> PhysicalSize {
    let scale = if scale > 0.0 { scale } else { 1.0 };
    PhysicalSize {
        width: (size.width.max(0.0) * scale).round() as u32,
        height: (size.height.max(0.0) * scale).round() as u32,
    }
}

pub(crate) fn expand_rect(rect: Rect, amount: f64) -> Rect {
    Rect::new(
        rect.origin.x - amount,
        rect.origin.y - amount,
        rect.size.width + amount * 2.0,
        rect.size.height + amount * 2.0,
    )
}
