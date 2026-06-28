use super::{
    Image, Point, Result,
    validation::{validate_color, validate_gradient_stops, validate_point, validate_positive_f64},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    pub const BLACK: Self = Self::rgba(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    #[must_use]
    const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn try_rgba(r: f32, g: f32, b: f32, a: f32) -> Result<Self> {
        let color = Self::rgba(r, g, b, a);
        validate_color(color, "color")?;
        Ok(color)
    }

    #[must_use]
    pub const fn r(self) -> f32 {
        self.r
    }

    #[must_use]
    pub const fn g(self) -> f32 {
        self.g
    }

    #[must_use]
    pub const fn b(self) -> f32 {
        self.b
    }

    #[must_use]
    pub const fn a(self) -> f32 {
        self.a
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::TRANSPARENT
    }
}

impl From<Color> for peniko::Color {
    fn from(color: Color) -> Self {
        Self::new([color.r, color.g, color.b, color.a])
    }
}

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
    pub const fn color(color: Color) -> Self {
        Self {
            kind: PaintKind::Color(color),
        }
    }

    #[must_use]
    pub fn gradient(gradient: Gradient) -> Self {
        Self {
            kind: PaintKind::Gradient(gradient),
        }
    }

    #[must_use]
    pub fn image(image: Image) -> Self {
        Self {
            kind: PaintKind::Image(image),
        }
    }

    pub(crate) const fn kind(&self) -> &PaintKind {
        &self.kind
    }
}

impl From<Color> for Paint {
    fn from(color: Color) -> Self {
        Self::color(color)
    }
}

impl From<Gradient> for Paint {
    fn from(gradient: Gradient) -> Self {
        Self::gradient(gradient)
    }
}

impl From<Image> for Paint {
    fn from(image: Image) -> Self {
        Self::image(image)
    }
}

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
    pub fn try_linear(start: Point, end: Point, stops: Vec<GradientStop>) -> Result<Self> {
        validate_point(start, "linear gradient start")?;
        validate_point(end, "linear gradient end")?;
        validate_gradient_stops(&stops)?;
        Ok(Self {
            kind: GradientKind::Linear { start, end, stops },
        })
    }

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

    pub fn try_sweep(center: Point, stops: Vec<GradientStop>) -> Result<Self> {
        validate_point(center, "sweep gradient center")?;
        validate_gradient_stops(&stops)?;
        Ok(Self {
            kind: GradientKind::Sweep { center, stops },
        })
    }

    pub(crate) const fn kind(&self) -> &GradientKind {
        &self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    offset: f32,
    color: Color,
}

impl GradientStop {
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
    pub const fn offset(self) -> f32 {
        self.offset
    }

    #[must_use]
    pub const fn color(self) -> Color {
        self.color
    }
}

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
