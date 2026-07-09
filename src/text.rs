use super::{
    Paint, Point, PrimitiveFamily, PrimitiveOperation, Result, ShadowList, Transform,
    UnsupportedPrimitive, validation::*,
};
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

#[derive(Clone, Debug, PartialEq)]
pub struct TextShadowRun<'a> {
    run: TextRun<'a>,
    shadows: ShadowList,
}

impl<'a> TextShadowRun<'a> {
    pub fn try_new(run: TextRun<'a>, shadows: ShadowList) -> Result<Self> {
        validate_text_run(run.size(), run.transform(), run.glyphs())?;
        validate_paint(run.paint().fill())?;
        for shadow in shadows.shadows() {
            validate_shadow(shadow)?;
        }
        Ok(Self { run, shadows })
    }

    #[must_use]
    pub const fn run(&self) -> &TextRun<'a> {
        &self.run
    }

    #[must_use]
    pub const fn shadows(&self) -> &ShadowList {
        &self.shadows
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextDecorationLine {
    start: Point,
    end: Point,
    thickness: f64,
    transform: Transform,
    paint: Paint,
}

impl TextDecorationLine {
    pub fn try_solid(
        start: Point,
        end: Point,
        thickness: f64,
        transform: Transform,
        paint: Paint,
    ) -> Result<Self> {
        Self::try_new(
            start,
            end,
            thickness,
            transform,
            paint,
            TextDecorationLineStyle::Solid,
        )
    }

    pub fn try_new(
        start: Point,
        end: Point,
        thickness: f64,
        transform: Transform,
        paint: Paint,
        style: TextDecorationLineStyle,
    ) -> Result<Self> {
        validate_point(start, "text decoration start")?;
        validate_point(end, "text decoration end")?;
        if start == end {
            return Err(super::Error::invalid_value(
                "text decoration line",
                "zero-length",
                "must have distinct start and end points",
            ));
        }
        validate_positive_f64(thickness, "text decoration thickness")?;
        validate_transform(transform, "text decoration transform")?;
        validate_paint(&paint)?;
        if style != TextDecorationLineStyle::Solid {
            return Err(unsupported_text_decoration_style_error(style));
        }
        Ok(Self {
            start,
            end,
            thickness,
            transform,
            paint,
        })
    }

    #[must_use]
    pub const fn start(&self) -> Point {
        self.start
    }

    #[must_use]
    pub const fn end(&self) -> Point {
        self.end
    }

    #[must_use]
    pub const fn thickness(&self) -> f64 {
        self.thickness
    }

    #[must_use]
    pub const fn transform(&self) -> Transform {
        self.transform
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextDecorationLineStyle {
    Solid,
    Double,
    Dotted,
    Dashed,
    Wavy,
}

impl TextDecorationLineStyle {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Solid => "solid",
            Self::Double => "double",
            Self::Dotted => "dotted",
            Self::Dashed => "dashed",
            Self::Wavy => "wavy",
        }
    }
}

fn unsupported_text_decoration_style_error(style: TextDecorationLineStyle) -> super::Error {
    let mut error = super::Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::TextDecorations,
        PrimitiveOperation::TextDecorationStyle,
    ));
    error.message.push_str(&format!(
        ": text decoration style '{}' requires root/text to expand the decoration into explicit render geometry",
        style.label()
    ));
    error
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
