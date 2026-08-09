use super::{
    Error, Image, Point, Result,
    validation::{
        validate_color, validate_gradient_stops, validate_paint, validate_point,
        validate_positive_f64,
    },
};

/// A validated numeric-sRGB color with straight red, green, blue, and alpha channels.
///
/// Every channel is finite and in the inclusive range `0..=1`. The default is
/// [`Self::TRANSPARENT`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    /// Opaque numeric-sRGB black.
    pub const BLACK: Self = Self::rgba(0.0, 0.0, 0.0, 1.0);
    /// Fully transparent numeric-sRGB black.
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    #[must_use]
    const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates a numeric-sRGB color from straight normalized channels.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when any channel is non-finite
    /// or outside the inclusive range `0..=1`.
    pub fn try_rgba(r: f32, g: f32, b: f32, a: f32) -> Result<Self> {
        let color = Self::rgba(r, g, b, a);
        validate_color(color, "color")?;
        Ok(color)
    }

    #[must_use]
    /// Returns the normalized red channel.
    pub const fn r(self) -> f32 {
        self.r
    }

    #[must_use]
    /// Returns the normalized green channel.
    pub const fn g(self) -> f32 {
        self.g
    }

    #[must_use]
    /// Returns the normalized blue channel.
    pub const fn b(self) -> f32 {
        self.b
    }

    #[must_use]
    /// Returns the normalized alpha channel.
    pub const fn a(self) -> f32 {
        self.a
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::TRANSPARENT
    }
}

/// Converts the numeric-sRGB channels to the equivalent Peniko color.
impl From<Color> for peniko::Color {
    fn from(color: Color) -> Self {
        Self::new([color.r, color.g, color.b, color.a])
    }
}

/// The authored color representation carried by [`PaintColor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaintColorSpace {
    /// Straight numeric-sRGB red, green, blue, and alpha channels.
    Srgb,
    /// Hue in degrees followed by normalized saturation, lightness, and alpha.
    Hsl,
}

/// A validated authored color that preserves its sRGB or HSL representation.
///
/// Call [`Self::to_color`] to resolve it to the numeric-sRGB [`Color`] consumed
/// by rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintColor {
    space: PaintColorSpace,
    channels: [f32; 4],
}

impl PaintColor {
    /// Creates an authored numeric-sRGB color from straight normalized channels.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for non-finite or out-of-range channels.
    pub fn try_srgb(r: f32, g: f32, b: f32, a: f32) -> Result<Self> {
        Color::try_rgba(r, g, b, a)?;
        Ok(Self {
            space: PaintColorSpace::Srgb,
            channels: [r, g, b, a],
        })
    }

    /// Creates an authored HSL color.
    ///
    /// Hue is a finite angle in degrees and may be outside one turn; resolution
    /// wraps it modulo 360. Saturation, lightness, and alpha must be finite and
    /// in `0..=1`, otherwise construction returns [`crate::ErrorCode::InvalidInput`].
    pub fn try_hsl(hue_degrees: f32, saturation: f32, lightness: f32, alpha: f32) -> Result<Self> {
        validate_finite_channel(hue_degrees, "hsl hue")?;
        validate_unit_channel(saturation, "hsl saturation")?;
        validate_unit_channel(lightness, "hsl lightness")?;
        validate_unit_channel(alpha, "hsl alpha")?;
        Ok(Self {
            space: PaintColorSpace::Hsl,
            channels: [hue_degrees, saturation, lightness, alpha],
        })
    }

    #[must_use]
    /// Returns the preserved authored color space.
    pub const fn space(self) -> PaintColorSpace {
        self.space
    }

    #[must_use]
    /// Returns the four preserved channels in the order defined by [`Self::space`].
    pub const fn channels(self) -> [f32; 4] {
        self.channels
    }

    /// Resolves this authored value to a validated numeric-sRGB [`Color`].
    ///
    /// HSL hue is wrapped modulo 360 degrees. A typed input diagnostic is
    /// returned if the resulting channels cannot satisfy [`Color`]'s invariant.
    pub fn to_color(self) -> Result<Color> {
        match self.space {
            PaintColorSpace::Srgb => {
                let [r, g, b, a] = self.channels;
                Color::try_rgba(r, g, b, a)
            }
            PaintColorSpace::Hsl => {
                let [hue_degrees, saturation, lightness, alpha] = self.channels;
                let [r, g, b] = hsl_to_srgb(hue_degrees, saturation, lightness);
                Color::try_rgba(r, g, b, alpha)
            }
        }
    }
}

fn validate_finite_channel(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_value(name, value, "must be finite"));
    }
    Ok(())
}

fn validate_unit_channel(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and between 0 and 1",
        ));
    }
    Ok(())
}

fn hsl_to_srgb(hue_degrees: f32, saturation: f32, lightness: f32) -> [f32; 3] {
    if saturation == 0.0 {
        return [lightness, lightness, lightness];
    }

    let hue = hue_degrees.rem_euclid(360.0);
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let hue_sector = hue / 60.0;
    let x = chroma * (1.0 - (hue_sector.rem_euclid(2.0) - 1.0).abs());
    let m = lightness - chroma / 2.0;

    let [r, g, b] = match hue_sector {
        sector if sector < 1.0 => [chroma, x, 0.0],
        sector if sector < 2.0 => [x, chroma, 0.0],
        sector if sector < 3.0 => [0.0, chroma, x],
        sector if sector < 4.0 => [0.0, x, chroma],
        sector if sector < 5.0 => [x, 0.0, chroma],
        _ => [chroma, 0.0, x],
    };

    [
        (r + m).clamp(0.0, 1.0),
        (g + m).clamp(0.0, 1.0),
        (b + m).clamp(0.0, 1.0),
    ]
}

/// A renderer-facing paint source: a solid color, gradient, or image.
///
/// Each public constructor preserves the validation already encoded by its input type.
#[derive(Clone, Debug, PartialEq)]
pub struct Paint {
    kind: PaintKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PaintKind {
    Color(Color),
    Gradient(Gradient),
    Image(Image),
}

impl Paint {
    #[must_use]
    /// Creates a solid-color paint without changing its numeric-sRGB channels.
    pub const fn color(color: Color) -> Self {
        Self {
            kind: PaintKind::Color(color),
        }
    }

    #[must_use]
    /// Creates a gradient paint from a validated gradient.
    pub fn gradient(gradient: Gradient) -> Self {
        Self {
            kind: PaintKind::Gradient(gradient),
        }
    }

    #[must_use]
    /// Creates an image paint retaining the image's exact content and sampling policy.
    pub fn image(image: Image) -> Self {
        Self {
            kind: PaintKind::Image(image),
        }
    }

    pub(crate) const fn kind(&self) -> &PaintKind {
        &self.kind
    }
}

/// Converts a color to its solid [`Paint`] representation without loss.
impl From<Color> for Paint {
    fn from(color: Color) -> Self {
        Self::color(color)
    }
}

/// Converts a gradient to its [`Paint`] representation without loss.
impl From<Gradient> for Paint {
    fn from(gradient: Gradient) -> Self {
        Self::gradient(gradient)
    }
}

/// Converts an image to its [`Paint`] representation without loss.
impl From<Image> for Paint {
    fn from(image: Image) -> Self {
        Self::image(image)
    }
}

/// A normalized paint layer whose source has passed canonical paint validation.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedPaintLayer {
    paint: Paint,
}

impl NormalizedPaintLayer {
    /// Canonically validates a renderer-facing paint source.
    ///
    /// Returns a typed input diagnostic when any nested color, gradient,
    /// geometry, stop, or image-size invariant is invalid.
    pub fn try_new(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(Self { paint })
    }

    #[must_use]
    /// Returns the validated paint source.
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

/// A validated logical-space linear, radial, or sweep gradient.
///
/// Every gradient contains at least one valid stop. Geometry is finite, and a
/// radial radius is strictly positive.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    kind: GradientKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GradientKind {
    Linear {
        start: Point,
        end: Point,
        stops: Vec<GradientStop>,
    },
    Radial {
        center: Point,
        radius: f64,
        stops: Vec<GradientStop>,
    },
    Sweep {
        center: Point,
        stops: Vec<GradientStop>,
    },
}

impl Gradient {
    /// Creates a linear gradient between finite logical points.
    ///
    /// Returns a typed input diagnostic for invalid points, an empty stop list,
    /// or any invalid stop.
    pub fn try_linear(start: Point, end: Point, stops: Vec<GradientStop>) -> Result<Self> {
        validate_point(start, "linear gradient start")?;
        validate_point(end, "linear gradient end")?;
        validate_gradient_stops(&stops)?;
        Ok(Self {
            kind: GradientKind::Linear { start, end, stops },
        })
    }

    /// Creates a radial gradient in logical coordinates.
    ///
    /// `radius` must be finite and greater than zero. Invalid geometry, an
    /// empty stop list, or an invalid stop returns a typed input diagnostic.
    pub fn try_radial(center: Point, radius: f64, stops: Vec<GradientStop>) -> Result<Self> {
        validate_point(center, "radial gradient center")?;
        validate_positive_f64(radius, "radial gradient radius")?;
        validate_gradient_stops(&stops)?;
        Ok(Self {
            kind: GradientKind::Radial {
                center,
                radius,
                stops,
            },
        })
    }

    /// Creates a full-turn sweep gradient around a finite logical center.
    ///
    /// An invalid center, empty stop list, or invalid stop returns a typed input diagnostic.
    pub fn try_sweep(center: Point, stops: Vec<GradientStop>) -> Result<Self> {
        validate_point(center, "sweep gradient center")?;
        validate_gradient_stops(&stops)?;
        Ok(Self {
            kind: GradientKind::Sweep { center, stops },
        })
    }

    #[must_use]
    /// Returns the gradient stops in their supplied order.
    pub fn stops(&self) -> &[GradientStop] {
        match &self.kind {
            GradientKind::Linear { stops, .. }
            | GradientKind::Radial { stops, .. }
            | GradientKind::Sweep { stops, .. } => stops,
        }
    }

    #[must_use]
    /// Returns the start and end points for a linear gradient.
    pub const fn linear_points(&self) -> Option<(Point, Point)> {
        match &self.kind {
            GradientKind::Linear { start, end, .. } => Some((*start, *end)),
            GradientKind::Radial { .. } | GradientKind::Sweep { .. } => None,
        }
    }

    #[must_use]
    /// Returns the center and logical radius for a radial gradient.
    pub const fn radial_geometry(&self) -> Option<(Point, f64)> {
        match &self.kind {
            GradientKind::Radial { center, radius, .. } => Some((*center, *radius)),
            GradientKind::Linear { .. } | GradientKind::Sweep { .. } => None,
        }
    }

    #[must_use]
    /// Returns the logical center for a sweep gradient.
    pub const fn sweep_center(&self) -> Option<Point> {
        match &self.kind {
            GradientKind::Sweep { center, .. } => Some(*center),
            GradientKind::Linear { .. } | GradientKind::Radial { .. } => None,
        }
    }

    pub(crate) const fn kind(&self) -> &GradientKind {
        &self.kind
    }
}

/// A validated gradient color stop at a normalized offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    offset: f32,
    color: Color,
}

impl GradientStop {
    /// Creates a stop at a finite offset in the inclusive range `0..=1`.
    ///
    /// An out-of-range offset or invalid numeric-sRGB color returns
    /// [`crate::ErrorCode::InvalidInput`].
    pub fn try_new(offset: f32, color: Color) -> Result<Self> {
        if !offset.is_finite() || !(0.0..=1.0).contains(&offset) {
            return Err(super::Error::invalid_value(
                "gradient stop offset",
                offset,
                "must be finite and between 0 and 1",
            ));
        }
        validate_color(color, "gradient stop color")?;
        Ok(Self { offset, color })
    }

    #[must_use]
    /// Returns the normalized stop offset.
    pub const fn offset(self) -> f32 {
        self.offset
    }

    #[must_use]
    /// Returns the stop color.
    pub const fn color(self) -> Color {
        self.color
    }
}

/// Lowers the validated gradient to Peniko, converting logical `f64` geometry to `f32`.
impl From<Gradient> for peniko::Gradient {
    fn from(gradient: Gradient) -> Self {
        let stops: Vec<_> = match gradient.kind() {
            GradientKind::Linear { stops, .. }
            | GradientKind::Radial { stops, .. }
            | GradientKind::Sweep { stops, .. } => stops
                .iter()
                .map(|stop| (stop.offset, peniko::Color::from(stop.color)))
                .collect(),
        };

        match gradient.kind {
            GradientKind::Linear { start, end, .. } => peniko::Gradient::new_linear(
                (start.x() as f32, start.y() as f32),
                (end.x() as f32, end.y() as f32),
            )
            .with_stops(stops.as_slice()),
            GradientKind::Radial { center, radius, .. } => {
                peniko::Gradient::new_radial((center.x() as f32, center.y() as f32), radius as f32)
                    .with_stops(stops.as_slice())
            }
            GradientKind::Sweep { center, .. } => peniko::Gradient::new_sweep(
                (center.x() as f32, center.y() as f32),
                0.0,
                std::f32::consts::TAU,
            )
            .with_stops(stops.as_slice()),
        }
    }
}
