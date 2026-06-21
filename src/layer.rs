use super::{Paint, Point, Shape, Transform};

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub clip: Option<Shape>,
    pub transform: Transform,
    pub opacity: f32,
    pub blend: BlendMode,
    pub mask: Option<Shape>,
    pub filter: Option<Filter>,
}

impl Layer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            clip: None,
            transform: Transform::identity(),
            opacity: 1.0,
            blend: BlendMode::Normal,
            mask: None,
            filter: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Plus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    Blur { radius: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shadow {
    pub offset: Point,
    pub blur: f64,
    pub spread: f64,
    pub paint: Paint,
}

impl Shadow {
    #[must_use]
    pub fn new(offset: Point, blur: f64, spread: f64, paint: impl Into<Paint>) -> Self {
        Self {
            offset,
            blur,
            spread,
            paint: paint.into(),
        }
    }
}
