use super::{
    Error, Image, Point, Result,
    validation::{
        validate_color, validate_gradient_stops, validate_paint, validate_point,
        validate_positive_f64,
    },
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaintColorSpace {
    Srgb,
    Hsl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintColor {
    space: PaintColorSpace,
    channels: [f32; 4],
}

impl PaintColor {
    pub fn try_srgb(r: f32, g: f32, b: f32, a: f32) -> Result<Self> {
        Color::try_rgba(r, g, b, a)?;
        Ok(Self {
            space: PaintColorSpace::Srgb,
            channels: [r, g, b, a],
        })
    }

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
    pub const fn space(self) -> PaintColorSpace {
        self.space
    }

    #[must_use]
    pub const fn channels(self) -> [f32; 4] {
        self.channels
    }

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
pub struct NormalizedPaintLayer {
    paint: Paint,
}

impl NormalizedPaintLayer {
    pub fn try_new(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(Self { paint })
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
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
