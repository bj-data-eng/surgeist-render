use super::{Paint, Transform};
use std::borrow::Cow;

#[derive(Clone, Debug, PartialEq)]
pub struct TextRun<'a> {
    pub font: FontRef<'a>,
    pub size: f32,
    pub transform: Transform,
    pub paint: TextPaint,
    pub glyphs: &'a [TextGlyph],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGlyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub advance: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FontRef<'a> {
    pub id: u64,
    pub name: Option<Cow<'a, str>>,
    pub(crate) data: Option<FontData>,
}

impl<'a> FontRef<'a> {
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            id,
            name: None,
            data: None,
        }
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
    pub fill: Paint,
}
