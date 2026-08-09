use super::super::{
    Capabilities, CoordinateSpaceTag, Error, FillRule, FilledPath, Path, Point, PrimitiveFamily,
    PrimitiveOperation, Radii, Rect, Result, Shape, Size, Transform, UnresolvedResource,
    UnresolvedResourceKind, UnsupportedPrimitive,
    shape::ShapeKind,
    validation::{validate_finite_f64, validate_path, validate_shape},
};
use super::image::StyleResourceRef;
use kurbo::Shape as KurboShape;

#[derive(Clone, Debug, PartialEq)]
/// An authored clip source with an optional tagged coordinate space.
///
/// Concrete shapes and paths can be normalized locally. A resource reference
/// remains symbolic and produces an unresolved-resource diagnostic until the
/// surrounding context replaces it with concrete geometry.
pub struct ClipInput {
    kind: ClipInputKind,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
/// The authored source represented by a [`ClipInput`].
pub enum ClipInputKind {
    /// Validated shape geometry.
    Shape(Shape),
    /// A validated path paired with its fill rule.
    Path(FilledPath),
    /// A symbolic clip-resource reference.
    Reference(StyleResourceRef),
}

impl ClipInput {
    /// Creates a clip from validated logical shape geometry.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the shape is invalid.
    pub fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: ClipInputKind::Shape(shape),
            coordinate_space: None,
        })
    }

    /// Creates a clip from a filled logical path.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when path validation fails.
    pub fn try_filled_path(path: FilledPath) -> Result<Self> {
        validate_path_clip(path.path())?;
        Ok(Self {
            kind: ClipInputKind::Path(path),
            coordinate_space: None,
        })
    }

    /// Creates a symbolic clip-resource input.
    #[must_use]
    pub const fn reference(reference: StyleResourceRef) -> Self {
        Self {
            kind: ClipInputKind::Reference(reference),
            coordinate_space: None,
        }
    }

    /// Associates the input with a tagged coordinate space.
    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    /// Returns the authored clip source.
    #[must_use]
    pub const fn kind(&self) -> &ClipInputKind {
        &self.kind
    }

    /// Returns the shape source, if this input contains one.
    #[must_use]
    pub const fn shape(&self) -> Option<&Shape> {
        match &self.kind {
            ClipInputKind::Shape(shape) => Some(shape),
            ClipInputKind::Path(_) => None,
            ClipInputKind::Reference(_) => None,
        }
    }

    /// Returns the filled-path source, if this input contains one.
    #[must_use]
    pub const fn filled_path(&self) -> Option<&FilledPath> {
        match &self.kind {
            ClipInputKind::Shape(_) => None,
            ClipInputKind::Path(path) => Some(path),
            ClipInputKind::Reference(_) => None,
        }
    }

    /// Returns the symbolic resource reference, if this input contains one.
    #[must_use]
    pub const fn reference_ref(&self) -> Option<&StyleResourceRef> {
        match &self.kind {
            ClipInputKind::Shape(_) => None,
            ClipInputKind::Path(_) => None,
            ClipInputKind::Reference(reference) => Some(reference),
        }
    }

    /// Returns the optional coordinate-space tag.
    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }

    /// Checks whether the current capability contract accepts this clip source.
    ///
    /// Concrete geometry is checked as shape clipping. A symbolic reference
    /// returns an unresolved clip-resource diagnostic.
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

    /// Converts a supported concrete input into normalized clip geometry.
    ///
    /// This validates the source and its finite transformed logical bounds.
    /// Unsupported geometry returns a capability diagnostic, while symbolic
    /// references return an unresolved-resource diagnostic.
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
/// Intrinsically validated clip geometry and its optional coordinate-space tag.
///
/// Construction guarantees valid geometry, finite transform coefficients, and
/// finite logical bounds after applying the tagged transform.
pub struct NormalizedClip {
    geometry: ClipGeometry,
    coordinate_space: Option<CoordinateSpaceTag>,
}

impl NormalizedClip {
    /// Validates normalized clip geometry in its optional coordinate space.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for invalid geometry or for
    /// transform coefficients or transformed bounds that are not finite.
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

    /// Returns the normalized clip geometry.
    #[must_use]
    pub const fn geometry(&self) -> &ClipGeometry {
        &self.geometry
    }

    /// Returns the optional coordinate-space tag used for the geometry.
    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Intrinsically validated clip geometry in logical rendering units.
pub struct ClipGeometry {
    kind: ClipGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
/// The closed geometry choice stored by [`ClipGeometry`].
pub enum ClipGeometryKind {
    /// An axis-aligned logical rectangle.
    Rect(Rect),
    /// A logical rectangle with corner radii.
    RoundedRect {
        /// The rectangle bounds.
        rect: Rect,
        /// The logical corner radii.
        radii: Radii,
    },
    /// A circle in logical coordinates.
    Circle {
        /// The logical center point.
        center: Point,
        /// The logical radius.
        radius: f64,
    },
    /// An ellipse in logical coordinates.
    Ellipse {
        /// The logical center point.
        center: Point,
        /// The horizontal and vertical logical radii.
        radii: Size,
    },
    /// A logical path paired with its fill rule.
    Path(FilledPath),
}

impl ClipGeometry {
    /// Validates and stores filled path geometry.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when path validation fails.
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

    /// Returns the normalized geometry choice.
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
