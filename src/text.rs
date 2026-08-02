use super::{
    Paint, Point, PrimitiveFamily, PrimitiveOperation, Rect, Result, ShadowList, Transform,
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

/// Authored ink bounds for a text run in run-local logical coordinates.
///
/// Direct rendering permits [`Self::unspecified`]. A text run captured by the
/// GPU graph must instead carry validated ink bounds or be explicitly empty;
/// the render crate never estimates ink geometry from glyph advances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextRunBounds {
    value: TextRunBoundsValue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TextRunBoundsValue {
    Unspecified,
    Empty,
    Ink(Rect),
}

/// The payload-free authored state of [`TextRunBounds`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextRunBoundsKind {
    /// No ink extent was supplied because direct rendering does not require one.
    ///
    /// GPU-graph capture rejects this state with a typed unresolved-bounds error.
    Unspecified,
    /// The run is known to contribute no ink.
    Empty,
    /// The run has a validated positive-area ink rectangle.
    Ink,
}

impl TextRunBounds {
    /// Returns bounds whose ink extent is intentionally not authored.
    ///
    /// This is valid for direct rendering but unresolved for GPU-graph capture.
    #[must_use]
    pub const fn unspecified() -> Self {
        Self {
            value: TextRunBoundsValue::Unspecified,
        }
    }

    /// Returns bounds for a run known to contribute no ink.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            value: TextRunBoundsValue::Empty,
        }
    }

    /// Validates a finite, positive-area run-local logical ink rectangle.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for non-finite or non-positive bounds.
    pub fn try_ink(rect: Rect) -> Result<Self> {
        validate_rect(rect, "text run ink bounds")?;
        validate_positive_f64(rect.width(), "text run ink bounds width")?;
        validate_positive_f64(rect.height(), "text run ink bounds height")?;
        Ok(Self {
            value: TextRunBoundsValue::Ink(rect),
        })
    }

    /// Returns the authored bounds state without exposing its ink payload.
    #[must_use]
    pub const fn kind(self) -> TextRunBoundsKind {
        match self.value {
            TextRunBoundsValue::Unspecified => TextRunBoundsKind::Unspecified,
            TextRunBoundsValue::Empty => TextRunBoundsKind::Empty,
            TextRunBoundsValue::Ink(_) => TextRunBoundsKind::Ink,
        }
    }

    /// Returns the validated run-local ink rectangle when one was authored.
    #[must_use]
    pub const fn ink_rect(self) -> Option<Rect> {
        match self.value {
            TextRunBoundsValue::Ink(rect) => Some(rect),
            TextRunBoundsValue::Unspecified | TextRunBoundsValue::Empty => None,
        }
    }
}

/// An authored text run with a font, glyphs, paint, transform, and ink bounds.
///
/// Glyph positions, size, and bounds are logical run-local values before
/// `transform`. Text shaping and ink-bound calculation belong to the caller;
/// this crate validates and lowers the supplied authored facts.
#[derive(Clone, Debug, PartialEq)]
pub struct TextRun<'a> {
    font: FontRef<'a>,
    size: f32,
    transform: Transform,
    paint: TextPaint,
    glyphs: &'a [TextGlyph],
    bounds: TextRunBounds,
}

impl<'a> TextRun<'a> {
    /// Creates a valid authored run with bounds in logical coordinates before `transform`.
    ///
    /// Invalid size, transform, glyph values, or paint return typed input diagnostics.
    pub fn try_new(
        font: FontRef<'a>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: &'a [TextGlyph],
        bounds: TextRunBounds,
    ) -> Result<Self> {
        validate_text_run(size, transform, glyphs)?;
        validate_paint(paint.fill())?;
        Ok(Self {
            font,
            size,
            transform,
            paint,
            glyphs,
            bounds,
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

    /// Returns the authored run-local logical ink bounds without estimating glyph geometry.
    #[must_use]
    pub const fn bounds(&self) -> TextRunBounds {
        self.bounds
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
    error.append_message(format!(
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

/// Validated, owned OpenType font bytes and collection index.
///
/// Construction verifies that the requested face is readable before GPU scene
/// lowering. Selected glyph tables receive additional fallible preflight during
/// rendering; malformed data never becomes an unwrap, fallback font, or silent omission.
#[derive(Clone, Debug, PartialEq)]
pub struct FontData {
    pub(crate) data: peniko::FontData,
}

impl FontData {
    /// Validates and owns OpenType bytes at the requested collection index.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] before GPU work when the bytes
    /// are malformed or the collection index is out of range.
    pub fn try_from_bytes(bytes: Vec<u8>, index: u32) -> Result<Self> {
        let byte_len = bytes.len();
        skrifa::FontRef::from_index(bytes.as_slice(), index)
            .map_err(|_| invalid_font_data(byte_len, index))?;
        Ok(Self {
            data: peniko::FontData::new(bytes.into(), index),
        })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.data.data.as_ref()
    }

    pub(crate) const fn index(&self) -> u32 {
        self.data.index
    }
}

pub(crate) fn invalid_font_data(byte_len: usize, index: u32) -> super::Error {
    super::Error::invalid_value(
        "font_data",
        format_args!("len={byte_len}, index={index}"),
        "must contain a readable OpenType font at the requested collection index",
    )
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
