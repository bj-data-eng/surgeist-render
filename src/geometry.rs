use super::{Error, Result};

/// A finite point in logical rendering coordinates.
///
/// The default is the logical origin `(0, 0)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    /// Creates a logical point from finite `x` and `y` coordinates.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when either coordinate is not finite.
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
    /// Returns the logical x-coordinate.
    pub const fn x(self) -> f64 {
        self.x
    }

    #[must_use]
    /// Returns the logical y-coordinate.
    pub const fn y(self) -> f64 {
        self.y
    }

    #[must_use]
    pub(crate) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A finite, non-negative extent in logical rendering units.
///
/// Zero width or height is valid, and the default is a zero-area size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    width: f64,
    height: f64,
}

impl Size {
    /// Creates a logical size with finite, non-negative dimensions.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when either dimension is negative or non-finite.
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
    /// Returns the logical width.
    pub const fn width(self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns the logical height.
    pub const fn height(self) -> f64 {
        self.height
    }

    #[must_use]
    pub(crate) const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

/// An axis-aligned rectangle in logical rendering coordinates.
///
/// Public construction guarantees a finite origin, finite non-negative size,
/// and finite derived maximum coordinates. Zero-area rectangles are valid.
/// The default is the zero-area rectangle at the logical origin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    origin: Point,
    size: Size,
}

impl Rect {
    /// Creates a rectangle with finite components and finite derived maximum coordinates.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for a non-finite origin,
    /// negative or non-finite size, or non-finite `x + width` or `y + height`.
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
        let rect = Self {
            origin,
            size: Size::new(width, height),
        };
        validate_rect_maxima(rect, "rectangle")?;
        Ok(rect)
    }

    #[must_use]
    /// Returns the rectangle's logical origin.
    pub const fn origin(self) -> Point {
        self.origin
    }

    #[must_use]
    /// Returns the rectangle's logical size.
    pub const fn size(self) -> Size {
        self.size
    }

    #[must_use]
    /// Returns the minimum logical x-coordinate.
    pub const fn x(self) -> f64 {
        self.origin.x()
    }

    #[must_use]
    /// Returns the minimum logical y-coordinate.
    pub const fn y(self) -> f64 {
        self.origin.y()
    }

    #[must_use]
    /// Returns the logical width.
    pub const fn width(self) -> f64 {
        self.size.width()
    }

    #[must_use]
    /// Returns the logical height.
    pub const fn height(self) -> f64 {
        self.size.height()
    }

    #[must_use]
    /// Returns the minimum corner, equal to [`Self::origin`].
    pub const fn min(self) -> Point {
        self.origin
    }

    #[must_use]
    /// Returns the derived maximum coordinates, which are finite for publicly constructed rectangles.
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

pub(crate) fn validate_rect_maxima(rect: Rect, name: &str) -> Result<()> {
    for (axis, maximum) in [
        ("x", rect.x() + rect.width()),
        ("y", rect.y() + rect.height()),
    ] {
        if !maximum.is_finite() {
            return Err(Error::invalid_value(
                format!("{name} max {axis}"),
                maximum,
                "must be finite",
            ));
        }
    }
    Ok(())
}

/// Converts the logical rectangle to Kurbo without loss.
impl From<Rect> for kurbo::Rect {
    fn from(rect: Rect) -> Self {
        let max = rect.max();
        Self::new(rect.x(), rect.y(), max.x(), max.y())
    }
}

/// Validates a Kurbo rectangle as a logical [`Rect`].
///
/// Conversion rejects negative or non-finite dimensions and non-finite derived maxima.
impl TryFrom<kurbo::Rect> for Rect {
    type Error = Error;

    fn try_from(rect: kurbo::Rect) -> Result<Self> {
        Self::try_new(rect.x0, rect.y0, rect.width(), rect.height())
    }
}

/// Four finite, non-negative corner radii in logical rendering units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Radii {
    top_left: f64,
    top_right: f64,
    bottom_right: f64,
    bottom_left: f64,
}

impl Radii {
    /// Creates independently specified corner radii.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when any radius is negative or non-finite.
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

    /// Creates four equal corner radii.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `radius` is negative or non-finite.
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
    /// Returns the top-left radius in logical units.
    pub const fn top_left(self) -> f64 {
        self.top_left
    }

    #[must_use]
    /// Returns the top-right radius in logical units.
    pub const fn top_right(self) -> f64 {
        self.top_right
    }

    #[must_use]
    /// Returns the bottom-right radius in logical units.
    pub const fn bottom_right(self) -> f64 {
        self.bottom_right
    }

    #[must_use]
    /// Returns the bottom-left radius in logical units.
    pub const fn bottom_left(self) -> f64 {
        self.bottom_left
    }

    #[must_use]
    /// Returns the common radius when all four corners are exactly equal.
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

/// A finite two-dimensional affine transform for logical rendering coordinates.
///
/// Values use Kurbo's `[a, b, c, d, e, f]` coefficient order. The default is
/// the identity transform; composition rejects any non-finite derived coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform([f64; 6]);

impl Transform {
    /// The identity transform.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    #[must_use]
    /// Returns the identity transform.
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Creates a transform from finite affine coefficients in Kurbo order.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when any coefficient is non-finite.
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

    /// Creates a logical translation by finite `x` and `y` offsets.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when either offset is non-finite.
    pub fn translation(x: f64, y: f64) -> Result<Self> {
        Self::try_new([1.0, 0.0, 0.0, 1.0, x, y])
    }

    /// Creates a finite scale transform for the supplied axis factors.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when either factor is non-finite.
    pub fn scale(x: f64, y: f64) -> Result<Self> {
        Self::try_new([x, 0.0, 0.0, y, 0.0, 0.0])
    }

    /// Creates a rotation by a finite angle in radians.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the angle produces a
    /// non-finite coefficient.
    pub fn rotation(radians: f64) -> Result<Self> {
        let (sin, cos) = radians.sin_cos();
        Self::try_new([cos, sin, -sin, cos, 0.0, 0.0])
    }

    /// Creates an x-axis skew by a finite angle in radians.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the angle produces a
    /// non-finite coefficient.
    pub fn skew_x(radians: f64) -> Result<Self> {
        Self::try_new([1.0, 0.0, radians.tan(), 1.0, 0.0, 0.0])
    }

    /// Creates a y-axis skew by a finite angle in radians.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the angle produces a
    /// non-finite coefficient.
    pub fn skew_y(radians: f64) -> Result<Self> {
        Self::try_new([1.0, radians.tan(), 0.0, 1.0, 0.0, 0.0])
    }

    /// Composes this transform followed by `next`.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if composition produces a
    /// non-finite coefficient.
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

    /// Conjugates this transform so it operates around `origin`.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if composition produces a
    /// non-finite coefficient.
    pub fn around(self, origin: Point) -> Result<Self> {
        Transform::translation(-origin.x(), -origin.y())?
            .then(self)?
            .then(Transform::translation(origin.x(), origin.y())?)
    }

    #[must_use]
    /// Returns the affine coefficients in Kurbo order.
    pub const fn as_array(self) -> [f64; 6] {
        self.0
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Converts this finite logical transform to the equivalent Kurbo affine transform.
impl From<Transform> for kurbo::Affine {
    fn from(transform: Transform) -> Self {
        Self::new(transform.as_array())
    }
}

/// A non-zero caller-defined identifier for a named rendering coordinate space.
///
/// Equality compares only the opaque numeric value; zero is reserved and rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateSpaceId {
    value: u64,
}

impl CoordinateSpaceId {
    /// Creates an opaque coordinate-space identifier.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when `value` is zero.
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
    /// Returns the non-zero underlying value.
    pub const fn get(self) -> u64 {
        self.value
    }
}

/// The semantic frame in which tagged rendering geometry is expressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateSpaceKind {
    /// Coordinates local to the current rendered object or layer.
    Local,
    /// Coordinates relative to the viewport.
    Viewport,
    /// Coordinates relative to the render surface.
    Surface,
    /// Coordinates in a caller-defined named space.
    Named(CoordinateSpaceId),
}

/// A coordinate-space kind paired with the finite transform used during lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateSpaceTag {
    kind: CoordinateSpaceKind,
    transform: Transform,
}

impl CoordinateSpaceTag {
    /// Creates a tag after revalidating every transform coefficient.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for a non-finite transform.
    pub fn try_new(kind: CoordinateSpaceKind, transform: Transform) -> Result<Self> {
        let transform = Transform::try_new(transform.as_array())?;
        Ok(Self { kind, transform })
    }

    #[must_use]
    /// Returns the identity tag for local logical coordinates.
    pub fn local() -> Self {
        Self {
            kind: CoordinateSpaceKind::Local,
            transform: Transform::identity(),
        }
    }

    /// Creates a viewport-space tag with the supplied finite transform.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if revalidation finds a
    /// non-finite coefficient.
    pub fn viewport(transform: Transform) -> Result<Self> {
        Self::try_new(CoordinateSpaceKind::Viewport, transform)
    }

    /// Creates a surface-space tag with the supplied finite transform.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] if revalidation finds a
    /// non-finite coefficient.
    pub fn surface(transform: Transform) -> Result<Self> {
        Self::try_new(CoordinateSpaceKind::Surface, transform)
    }

    #[must_use]
    /// Returns the tagged coordinate-space kind.
    pub const fn kind(self) -> CoordinateSpaceKind {
        self.kind
    }

    #[must_use]
    /// Returns the transform associated with the coordinate space.
    pub const fn transform(self) -> Transform {
        self.transform
    }
}

/// A width and height in integer physical device pixels.
///
/// Zero-area sizes are valid, and the default is `0` by `0` device pixels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PhysicalSize {
    width: u32,
    height: u32,
}

impl PhysicalSize {
    #[must_use]
    /// Creates a physical size in device pixels.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Resolves a logical size to rounded physical device-pixel dimensions.
    ///
    /// `scale` must be finite and greater than zero. Returns
    /// [`crate::ErrorCode::InvalidInput`] when the scale is invalid or a scaled
    /// dimension exceeds `u32` device pixels.
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
    /// Returns the width in physical device pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    #[must_use]
    /// Returns the height in physical device pixels.
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
