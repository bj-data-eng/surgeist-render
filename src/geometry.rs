use super::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn try_new(x: f64, y: f64) -> Result<Self> {
        if !x.is_finite() {
            return Err(Error::invalid_value("point x", x, "must be finite"));
        }
        if !y.is_finite() {
            return Err(Error::invalid_value("point y", y, "must be finite"));
        }
        Ok(Self { x, y })
    }

    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    #[must_use]
    pub(crate) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    width: f64,
    height: f64,
}

impl Size {
    pub fn try_new(width: f64, height: f64) -> Result<Self> {
        if !width.is_finite() || width < 0.0 {
            return Err(Error::invalid_value(
                "size width",
                width,
                "must be finite and non-negative",
            ));
        }
        if !height.is_finite() || height < 0.0 {
            return Err(Error::invalid_value(
                "size height",
                height,
                "must be finite and non-negative",
            ));
        }
        Ok(Self { width, height })
    }

    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }

    #[must_use]
    pub(crate) const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    origin: Point,
    size: Size,
}

impl Rect {
    pub fn try_new(x: f64, y: f64, width: f64, height: f64) -> Result<Self> {
        let origin = Point::try_new(x, y)?;
        if !width.is_finite() || width < 0.0 {
            return Err(Error::invalid_value(
                "rectangle width",
                width,
                "must be finite and non-negative",
            ));
        }
        if !height.is_finite() || height < 0.0 {
            return Err(Error::invalid_value(
                "rectangle height",
                height,
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            origin,
            size: Size::new(width, height),
        })
    }

    #[must_use]
    pub const fn origin(self) -> Point {
        self.origin
    }

    #[must_use]
    pub const fn size(self) -> Size {
        self.size
    }

    #[must_use]
    pub const fn x(self) -> f64 {
        self.origin.x()
    }

    #[must_use]
    pub const fn y(self) -> f64 {
        self.origin.y()
    }

    #[must_use]
    pub const fn width(self) -> f64 {
        self.size.width()
    }

    #[must_use]
    pub const fn height(self) -> f64 {
        self.size.height()
    }

    #[must_use]
    pub const fn min(self) -> Point {
        self.origin
    }

    #[must_use]
    pub fn max(self) -> Point {
        Point::new(self.x() + self.width(), self.y() + self.height())
    }

    #[must_use]
    pub(crate) const fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(width, height),
        }
    }
}

impl From<Rect> for kurbo::Rect {
    fn from(rect: Rect) -> Self {
        let max = rect.max();
        Self::new(rect.x(), rect.y(), max.x(), max.y())
    }
}

impl TryFrom<kurbo::Rect> for Rect {
    type Error = Error;

    fn try_from(rect: kurbo::Rect) -> Result<Self> {
        Self::try_new(rect.x0, rect.y0, rect.width(), rect.height())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Radii {
    top_left: f64,
    top_right: f64,
    bottom_right: f64,
    bottom_left: f64,
}

impl Radii {
    pub fn try_new(
        top_left: f64,
        top_right: f64,
        bottom_right: f64,
        bottom_left: f64,
    ) -> Result<Self> {
        for (name, value) in [
            ("top-left corner radius", top_left),
            ("top-right corner radius", top_right),
            ("bottom-right corner radius", bottom_right),
            ("bottom-left corner radius", bottom_left),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid_value(
                    name,
                    value,
                    "must be finite and non-negative",
                ));
            }
        }
        Ok(Self {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        })
    }

    pub fn try_all(radius: f64) -> Result<Self> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::invalid_value(
                "corner radius",
                radius,
                "must be finite and non-negative",
            ));
        }
        Ok(Self::all(radius))
    }

    #[must_use]
    pub const fn top_left(self) -> f64 {
        self.top_left
    }

    #[must_use]
    pub const fn top_right(self) -> f64 {
        self.top_right
    }

    #[must_use]
    pub const fn bottom_right(self) -> f64 {
        self.bottom_right
    }

    #[must_use]
    pub const fn bottom_left(self) -> f64 {
        self.bottom_left
    }

    #[must_use]
    pub fn uniform(self) -> Option<f64> {
        (self.top_left == self.top_right
            && self.top_left == self.bottom_right
            && self.top_left == self.bottom_left)
            .then_some(self.top_left)
    }

    #[must_use]
    pub(crate) const fn new(
        top_left: f64,
        top_right: f64,
        bottom_right: f64,
        bottom_left: f64,
    ) -> Self {
        Self {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        }
    }

    #[must_use]
    pub(crate) const fn all(radius: f64) -> Self {
        Self::new(radius, radius, radius, radius)
    }
}

pub(crate) fn offset_radii(radii: Radii, offset: f64) -> Radii {
    Radii::new(
        (radii.top_left() + offset).max(0.0),
        (radii.top_right() + offset).max(0.0),
        (radii.bottom_right() + offset).max(0.0),
        (radii.bottom_left() + offset).max(0.0),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform([f64; 6]);

impl Transform {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    #[must_use]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    pub fn try_new(values: [f64; 6]) -> Result<Self> {
        for value in values {
            if !value.is_finite() {
                return Err(Error::invalid_value(
                    "transform",
                    value,
                    "must contain only finite values",
                ));
            }
        }
        Ok(Self(values))
    }

    pub fn translation(x: f64, y: f64) -> Result<Self> {
        Self::try_new([1.0, 0.0, 0.0, 1.0, x, y])
    }

    pub fn scale(x: f64, y: f64) -> Result<Self> {
        Self::try_new([x, 0.0, 0.0, y, 0.0, 0.0])
    }

    pub fn rotation(radians: f64) -> Result<Self> {
        let (sin, cos) = radians.sin_cos();
        Self::try_new([cos, sin, -sin, cos, 0.0, 0.0])
    }

    pub fn skew_x(radians: f64) -> Result<Self> {
        Self::try_new([1.0, 0.0, radians.tan(), 1.0, 0.0, 0.0])
    }

    pub fn skew_y(radians: f64) -> Result<Self> {
        Self::try_new([1.0, radians.tan(), 0.0, 1.0, 0.0, 0.0])
    }

    pub fn then(self, next: Self) -> Result<Self> {
        let [a, b, c, d, e, f] = self.0;
        let [na, nb, nc, nd, ne, nf] = next.0;
        Self::try_new([
            na * a + nc * b,
            nb * a + nd * b,
            na * c + nc * d,
            nb * c + nd * d,
            na * e + nc * f + ne,
            nb * e + nd * f + nf,
        ])
    }

    pub fn around(self, origin: Point) -> Result<Self> {
        Transform::translation(-origin.x(), -origin.y())?
            .then(self)?
            .then(Transform::translation(origin.x(), origin.y())?)
    }

    #[must_use]
    pub const fn as_array(self) -> [f64; 6] {
        self.0
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Transform> for kurbo::Affine {
    fn from(transform: Transform) -> Self {
        Self::new(transform.as_array())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateSpaceId {
    value: u64,
}

impl CoordinateSpaceId {
    pub fn try_new(value: u64) -> Result<Self> {
        if value == 0 {
            return Err(Error::invalid_value(
                "coordinate space id",
                value,
                "must be non-zero",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateSpaceKind {
    Local,
    Viewport,
    Surface,
    Named(CoordinateSpaceId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateSpaceTag {
    kind: CoordinateSpaceKind,
    transform: Transform,
}

impl CoordinateSpaceTag {
    pub fn try_new(kind: CoordinateSpaceKind, transform: Transform) -> Result<Self> {
        let transform = Transform::try_new(transform.as_array())?;
        Ok(Self { kind, transform })
    }

    #[must_use]
    pub fn local() -> Self {
        Self {
            kind: CoordinateSpaceKind::Local,
            transform: Transform::identity(),
        }
    }

    pub fn viewport(transform: Transform) -> Result<Self> {
        Self::try_new(CoordinateSpaceKind::Viewport, transform)
    }

    pub fn surface(transform: Transform) -> Result<Self> {
        Self::try_new(CoordinateSpaceKind::Surface, transform)
    }

    #[must_use]
    pub const fn kind(self) -> CoordinateSpaceKind {
        self.kind
    }

    #[must_use]
    pub const fn transform(self) -> Transform {
        self.transform
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PhysicalSize {
    width: u32,
    height: u32,
}

impl PhysicalSize {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn try_from_logical(size: Size, scale: f64) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::invalid_value(
                "surface scale",
                scale,
                "must be finite and greater than 0",
            ));
        }
        let width = size.width() * scale;
        let height = size.height() * scale;
        if width > f64::from(u32::MAX) {
            return Err(Error::invalid_value(
                "physical width",
                width,
                "must fit in u32 device pixels",
            ));
        }
        if height > f64::from(u32::MAX) {
            return Err(Error::invalid_value(
                "physical height",
                height,
                "must fit in u32 device pixels",
            ));
        }
        Ok(Self {
            width: width.round() as u32,
            height: height.round() as u32,
        })
    }

    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

pub(crate) fn physical_size(size: Size, scale: f64) -> Result<PhysicalSize> {
    PhysicalSize::try_from_logical(size, scale)
}

pub(crate) fn expand_rect(rect: Rect, amount: f64) -> Rect {
    Rect::new(
        rect.x() - amount,
        rect.y() - amount,
        rect.width() + amount * 2.0,
        rect.height() + amount * 2.0,
    )
}
