use super::{
    Error, Paint, Point, Result, Shape, Transform,
    validation::{
        validate_filter, validate_finite_f64, validate_non_negative_f64, validate_paint,
        validate_point, validate_shape, validate_transform,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    clip: Option<Shape>,
    transform: Transform,
    opacity: f32,
    blend: BlendMode,
    mask: Option<Shape>,
    filter: Option<Filter>,
}

impl Layer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_clip(mut self, clip: Shape) -> Result<Self> {
        validate_shape(&clip)?;
        self.clip = Some(clip);
        Ok(self)
    }

    pub fn try_mask(mut self, mask: Shape) -> Result<Self> {
        validate_shape(&mask)?;
        self.mask = Some(mask);
        Ok(self)
    }

    pub fn try_filter(mut self, filter: Filter) -> Result<Self> {
        validate_filter(filter)?;
        self.filter = Some(filter);
        Ok(self)
    }

    pub fn try_transform(mut self, transform: Transform) -> Result<Self> {
        validate_transform(transform, "layer transform")?;
        self.transform = transform;
        Ok(self)
    }

    #[must_use]
    pub const fn blend(mut self, blend: BlendMode) -> Self {
        self.blend = blend;
        self
    }

    pub fn try_opacity(mut self, opacity: f32) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value(
                "layer opacity",
                opacity,
                "must be finite",
            ));
        }
        self.opacity = opacity;
        Ok(self)
    }

    #[must_use]
    pub fn clip(&self) -> Option<&Shape> {
        self.clip.as_ref()
    }

    #[must_use]
    pub fn mask(&self) -> Option<&Shape> {
        self.mask.as_ref()
    }

    #[must_use]
    pub const fn filter(&self) -> Option<Filter> {
        self.filter
    }

    #[must_use]
    pub const fn transform(&self) -> Transform {
        self.transform
    }

    #[must_use]
    pub const fn opacity(&self) -> f32 {
        self.opacity
    }

    #[must_use]
    pub const fn blend_mode(&self) -> BlendMode {
        self.blend
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
pub struct Filter {
    radius: f64,
}

impl Filter {
    pub fn try_blur(radius: f64) -> Result<Self> {
        validate_non_negative_f64(radius, "layer blur radius")?;
        Ok(Self { radius })
    }

    #[must_use]
    pub const fn blur_radius(self) -> f64 {
        self.radius
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shadow {
    kind: ShadowKind,
    offset: Point,
    blur: f64,
    spread: f64,
    paint: Paint,
}

impl Shadow {
    fn new(
        kind: ShadowKind,
        offset: Point,
        blur: f64,
        spread: f64,
        paint: impl Into<Paint>,
    ) -> Self {
        Self {
            kind,
            offset,
            blur,
            spread,
            paint: paint.into(),
        }
    }

    pub fn try_new(offset: Point, blur: f64, spread: f64, paint: impl Into<Paint>) -> Result<Self> {
        Self::try_with_kind(ShadowKind::Outer, offset, blur, spread, paint)
    }

    pub fn try_inset(
        offset: Point,
        blur: f64,
        spread: f64,
        paint: impl Into<Paint>,
    ) -> Result<Self> {
        Self::try_with_kind(ShadowKind::Inset, offset, blur, spread, paint)
    }

    pub fn try_with_kind(
        kind: ShadowKind,
        offset: Point,
        blur: f64,
        spread: f64,
        paint: impl Into<Paint>,
    ) -> Result<Self> {
        validate_point(offset, "shadow offset")?;
        validate_non_negative_f64(blur, "shadow blur")?;
        validate_finite_f64(spread, "shadow spread")?;
        let paint = paint.into();
        validate_paint(&paint)?;
        Ok(Self::new(kind, offset, blur, spread, paint))
    }

    #[must_use]
    pub const fn kind(&self) -> ShadowKind {
        self.kind
    }

    #[must_use]
    pub const fn offset(&self) -> Point {
        self.offset
    }

    #[must_use]
    pub const fn blur(&self) -> f64 {
        self.blur
    }

    #[must_use]
    pub const fn spread(&self) -> f64 {
        self.spread
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ShadowKind {
    #[default]
    Outer,
    Inset,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShadowList {
    shadows: Vec<Shadow>,
}

impl ShadowList {
    pub fn try_new(shadows: Vec<Shadow>) -> Result<Self> {
        if shadows.is_empty() {
            return Err(Error::invalid_value(
                "shadow list",
                "[]",
                "must contain at least one shadow",
            ));
        }
        for shadow in &shadows {
            validate_point(shadow.offset(), "shadow offset")?;
            validate_non_negative_f64(shadow.blur(), "shadow blur")?;
            validate_finite_f64(shadow.spread(), "shadow spread")?;
            validate_paint(shadow.paint())?;
        }
        Ok(Self { shadows })
    }

    #[must_use]
    pub fn shadows(&self) -> &[Shadow] {
        &self.shadows
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.shadows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shadows.is_empty()
    }

    pub(crate) fn into_vec(self) -> Vec<Shadow> {
        self.shadows
    }
}
