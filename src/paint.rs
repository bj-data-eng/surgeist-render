use super::{Image, Point};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const BLACK: Self = Self::rgba(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
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
pub enum Paint {
    Color(Color),
    Gradient(Gradient),
    Image(Image),
}

impl Paint {
    #[must_use]
    pub const fn color(color: Color) -> Self {
        Self::Color(color)
    }
}

impl From<Color> for Paint {
    fn from(color: Color) -> Self {
        Self::Color(color)
    }
}

impl From<Gradient> for Paint {
    fn from(gradient: Gradient) -> Self {
        Self::Gradient(gradient)
    }
}

impl From<Image> for Paint {
    fn from(image: Image) -> Self {
        Self::Image(image)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Gradient {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

impl From<Gradient> for peniko::Gradient {
    fn from(gradient: Gradient) -> Self {
        let stops: Vec<_> = match &gradient {
            Gradient::Linear { stops, .. }
            | Gradient::Radial { stops, .. }
            | Gradient::Sweep { stops, .. } => stops
                .iter()
                .map(|stop| (stop.offset, peniko::Color::from(stop.color)))
                .collect(),
        };

        match gradient {
            Gradient::Linear { start, end, .. } => peniko::Gradient::new_linear(
                (start.x as f32, start.y as f32),
                (end.x as f32, end.y as f32),
            )
            .with_stops(stops.as_slice()),
            Gradient::Radial { center, radius, .. } => {
                peniko::Gradient::new_radial((center.x as f32, center.y as f32), radius as f32)
                    .with_stops(stops.as_slice())
            }
            Gradient::Sweep { center, .. } => peniko::Gradient::new_sweep(
                (center.x as f32, center.y as f32),
                0.0,
                std::f32::consts::TAU,
            )
            .with_stops(stops.as_slice()),
        }
    }
}
