use super::{Paint, Result, Transform, validation::*};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FontId(u64);

impl FontId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for FontId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextRun<'a> {
    font: FontRef<'a>,
    size: f32,
    transform: Transform,
    paint: TextPaint,
    glyphs: &'a [TextGlyph],
}

impl<'a> TextRun<'a> {
    pub fn try_new(
        font: FontRef<'a>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: &'a [TextGlyph],
    ) -> Result<Self> {
        validate_text_run(size, transform, glyphs)?;
        validate_paint(paint.fill())?;
        Ok(Self {
            font,
            size,
            transform,
            paint,
            glyphs,
        })
    }

    #[must_use]
    pub const fn font(&self) -> &FontRef<'a> {
        &self.font
    }

    #[must_use]
    pub const fn size(&self) -> f32 {
        self.size
    }

    #[must_use]
    pub const fn transform(&self) -> Transform {
        self.transform
    }

    #[must_use]
    pub const fn paint(&self) -> &TextPaint {
        &self.paint
    }

    #[must_use]
    pub const fn glyphs(&self) -> &'a [TextGlyph] {
        self.glyphs
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGlyph {
    id: u32,
    x: f32,
    y: f32,
    advance: f32,
}

impl TextGlyph {
    pub fn try_new(id: u32, x: f32, y: f32, advance: f32) -> Result<Self> {
        if !x.is_finite() || !y.is_finite() || !advance.is_finite() {
            return Err(super::Error::invalid_value(
                "text glyph positions and advances",
                format_args!("x={x}, y={y}, advance={advance}"),
                "must be finite",
            ));
        }
        Ok(Self { id, x, y, advance })
    }

    #[must_use]
    pub const fn id(self) -> u32 {
        self.id
    }

    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }

    #[must_use]
    pub const fn advance(self) -> f32 {
        self.advance
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FontRef<'a> {
    id: FontId,
    pub name: Option<Cow<'a, str>>,
    pub(crate) data: Option<FontData>,
}

impl<'a> FontRef<'a> {
    #[must_use]
    pub fn new(id: impl Into<FontId>) -> Self {
        Self {
            id: id.into(),
            name: None,
            data: None,
        }
    }

    #[must_use]
    pub const fn id(&self) -> FontId {
        self.id
    }

    #[must_use]
    pub fn named(mut self, name: impl Into<Cow<'a, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_data(mut self, data: FontData) -> Self {
        self.data = Some(data);
        self
    }

    pub(crate) fn to_owned_static(&self) -> FontRef<'static> {
        FontRef {
            id: self.id,
            name: self
                .name
                .as_ref()
                .map(|name| Cow::Owned(name.clone().into_owned())),
            data: self.data.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FontData {
    pub(crate) data: peniko::FontData,
}

impl FontData {
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>, index: u32) -> Self {
        Self {
            data: peniko::FontData::new(bytes.into(), index),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextPaint {
    fill: Paint,
}

impl TextPaint {
    pub fn try_fill(fill: Paint) -> Result<Self> {
        validate_paint(&fill)?;
        Ok(Self { fill })
    }

    #[must_use]
    pub const fn fill(&self) -> &Paint {
        &self.fill
    }
}
