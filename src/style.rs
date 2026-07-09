use super::{
    Color, Error, Image, ImageId, Paint, Result, Size,
    validation::{validate_paint, validate_size},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StyleColor {
    color: Color,
}

impl StyleColor {
    #[must_use]
    pub const fn new(color: Color) -> Self {
        Self { color }
    }

    #[must_use]
    pub const fn color(self) -> Color {
        self.color
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StyleResourceRef {
    identifier: String,
}

impl StyleResourceRef {
    pub fn try_new(identifier: impl Into<String>) -> Result<Self> {
        let identifier = identifier.into();
        if identifier.trim().is_empty() {
            return Err(Error::invalid_value(
                "style resource reference",
                identifier,
                "must not be empty",
            ));
        }
        Ok(Self { identifier })
    }

    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedImageResource {
    id: ImageId,
    intrinsic_size: Size,
}

impl ResolvedImageResource {
    pub fn try_new(id: ImageId, intrinsic_size: Size) -> Result<Self> {
        validate_size(intrinsic_size, "resolved image intrinsic size")?;
        Ok(Self { id, intrinsic_size })
    }

    #[must_use]
    pub const fn id(&self) -> ImageId {
        self.id
    }

    #[must_use]
    pub const fn intrinsic_size(&self) -> Size {
        self.intrinsic_size
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StyleImageSource {
    kind: StyleImageSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StyleImageSourceKind {
    Image(Image),
    Resolved(ResolvedImageResource),
    Paint(Paint),
}

impl StyleImageSource {
    pub fn image(image: Image) -> Result<Self> {
        validate_size(image.size(), "image size")?;
        Ok(Self {
            kind: StyleImageSourceKind::Image(image),
        })
    }

    pub fn paint(paint: Paint) -> Result<Self> {
        validate_paint(&paint)?;
        Ok(Self {
            kind: StyleImageSourceKind::Paint(paint),
        })
    }

    #[must_use]
    pub const fn resolved(resource: ResolvedImageResource) -> Self {
        Self {
            kind: StyleImageSourceKind::Resolved(resource),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> &StyleImageSourceKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StyleImageLayer {
    source: StyleImageSource,
    position: BackgroundPosition,
    size: BackgroundSize,
    repeat: BackgroundRepeat,
    origin: BackgroundBox,
    clip: BackgroundBox,
    attachment: BackgroundAttachment,
}

impl StyleImageLayer {
    pub fn try_new(source: StyleImageSource) -> Result<Self> {
        Ok(Self {
            source,
            position: BackgroundPosition::default(),
            size: BackgroundSize::auto(),
            repeat: BackgroundRepeat::repeat(),
            origin: BackgroundBox::Padding,
            clip: BackgroundBox::Border,
            attachment: BackgroundAttachment::Scroll,
        })
    }

    #[must_use]
    pub fn with_position(mut self, position: BackgroundPosition) -> Self {
        self.position = position;
        self
    }

    #[must_use]
    pub fn with_size(mut self, size: BackgroundSize) -> Self {
        self.size = size;
        self
    }

    #[must_use]
    pub fn with_repeat(mut self, repeat: BackgroundRepeat) -> Self {
        self.repeat = repeat;
        self
    }

    #[must_use]
    pub fn with_origin(mut self, origin: BackgroundBox) -> Self {
        self.origin = origin;
        self
    }

    #[must_use]
    pub fn with_clip(mut self, clip: BackgroundBox) -> Self {
        self.clip = clip;
        self
    }

    #[must_use]
    pub fn with_attachment(mut self, attachment: BackgroundAttachment) -> Self {
        self.attachment = attachment;
        self
    }

    #[must_use]
    pub const fn source(&self) -> &StyleImageSource {
        &self.source
    }

    #[must_use]
    pub const fn position(&self) -> BackgroundPosition {
        self.position
    }

    #[must_use]
    pub const fn size(&self) -> BackgroundSize {
        self.size
    }

    #[must_use]
    pub const fn repeat(&self) -> BackgroundRepeat {
        self.repeat
    }

    #[must_use]
    pub const fn origin(&self) -> BackgroundBox {
        self.origin
    }

    #[must_use]
    pub const fn clip(&self) -> BackgroundBox {
        self.clip
    }

    #[must_use]
    pub const fn attachment(&self) -> BackgroundAttachment {
        self.attachment
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundPosition {
    x: PositionComponent,
    y: PositionComponent,
}

impl BackgroundPosition {
    pub fn percent(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_percent_for(x, "background position x percent")?,
            y: PositionComponent::try_percent_for(y, "background position y percent")?,
        })
    }

    pub fn length(x: f64, y: f64) -> Result<Self> {
        Ok(Self {
            x: PositionComponent::try_length_for(x, "background position x length")?,
            y: PositionComponent::try_length_for(y, "background position y length")?,
        })
    }

    pub fn components(x: PositionComponent, y: PositionComponent) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn x(self) -> PositionComponent {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> PositionComponent {
        self.y
    }
}

impl Default for BackgroundPosition {
    fn default() -> Self {
        Self {
            x: PositionComponent::percent_unchecked(0.0),
            y: PositionComponent::percent_unchecked(0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionComponent {
    kind: PositionComponentKind,
    value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionComponentKind {
    Length,
    Percent,
}

impl PositionComponent {
    pub fn try_percent(value: f64) -> Result<Self> {
        Self::try_percent_for(value, "background position percent")
    }

    pub fn try_length(value: f64) -> Result<Self> {
        Self::try_length_for(value, "background position length")
    }

    fn try_percent_for(value: f64, field: &str) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid_value(field, value, "must be finite"));
        }
        Ok(Self::percent_unchecked(value))
    }

    fn try_length_for(value: f64, field: &str) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid_value(field, value, "must be finite"));
        }
        Ok(Self {
            kind: PositionComponentKind::Length,
            value,
        })
    }

    const fn percent_unchecked(value: f64) -> Self {
        Self {
            kind: PositionComponentKind::Percent,
            value,
        }
    }

    #[must_use]
    pub const fn kind(self) -> PositionComponentKind {
        self.kind
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundSize {
    kind: BackgroundSizeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BackgroundSizeKind {
    Auto,
    Cover,
    Contain,
    Explicit {
        width: SizeComponent,
        height: SizeComponent,
    },
}

impl BackgroundSize {
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: BackgroundSizeKind::Auto,
        }
    }

    #[must_use]
    pub const fn cover() -> Self {
        Self {
            kind: BackgroundSizeKind::Cover,
        }
    }

    #[must_use]
    pub const fn contain() -> Self {
        Self {
            kind: BackgroundSizeKind::Contain,
        }
    }

    #[must_use]
    pub const fn explicit(width: SizeComponent, height: SizeComponent) -> Self {
        Self {
            kind: BackgroundSizeKind::Explicit { width, height },
        }
    }

    #[must_use]
    pub const fn kind(self) -> BackgroundSizeKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeComponent {
    kind: SizeComponentKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizeComponentKind {
    Auto,
    Length(f64),
    Percent(f64),
}

impl SizeComponent {
    #[must_use]
    pub const fn auto() -> Self {
        Self {
            kind: SizeComponentKind::Auto,
        }
    }

    pub fn try_length(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "background size length",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: SizeComponentKind::Length(value),
        })
    }

    pub fn try_percent(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "background size percent",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: SizeComponentKind::Percent(value),
        })
    }

    #[must_use]
    pub const fn kind(self) -> SizeComponentKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackgroundRepeat {
    x: RepeatMode,
    y: RepeatMode,
}

impl BackgroundRepeat {
    #[must_use]
    pub const fn new(x: RepeatMode, y: RepeatMode) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn repeat() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::Repeat)
    }

    #[must_use]
    pub const fn repeat_x() -> Self {
        Self::new(RepeatMode::Repeat, RepeatMode::NoRepeat)
    }

    #[must_use]
    pub const fn repeat_y() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::Repeat)
    }

    #[must_use]
    pub const fn no_repeat() -> Self {
        Self::new(RepeatMode::NoRepeat, RepeatMode::NoRepeat)
    }

    #[must_use]
    pub const fn x(self) -> RepeatMode {
        self.x
    }

    #[must_use]
    pub const fn y(self) -> RepeatMode {
        self.y
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepeatMode {
    Repeat,
    NoRepeat,
    Round,
    Space,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundBox {
    Border,
    Padding,
    Content,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachment {
    Scroll,
    Fixed,
    Local,
}
