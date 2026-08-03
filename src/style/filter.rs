use super::super::{
    Capabilities, Color, Error, Point, PrimitiveFamily, PrimitiveOperation, Rect, Result, Shadow,
    ShadowKind, UnsupportedPrimitive,
    paint::PaintKind,
    validation::{validate_color, validate_finite_f64, validate_point},
};
use super::{clip::ClipInput, image::ResolvedImageResource};

const MAX_FILTER_BLUR_RADIUS: f64 = 256.0;

#[derive(Clone, Debug, PartialEq)]
pub struct FilterList {
    ops: Option<Vec<FilterOp>>,
}

impl FilterList {
    #[must_use]
    pub const fn none() -> Self {
        Self { ops: None }
    }

    pub fn try_ops(ops: Vec<FilterOp>) -> Result<Self> {
        if ops.is_empty() {
            return Err(Error::invalid_value(
                "filter operations",
                "[]",
                "must not be empty",
            ));
        }
        Ok(Self { ops: Some(ops) })
    }

    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.ops.is_none()
    }

    #[must_use]
    pub fn ops(&self) -> &[FilterOp] {
        self.ops.as_deref().unwrap_or(&[])
    }

    pub fn color_filter_pipeline(
        &self,
    ) -> std::result::Result<Option<ColorFilterPipeline>, UnsupportedPrimitive> {
        let Some(ops) = self.ops.as_deref() else {
            return Ok(None);
        };

        let mut color_ops = Vec::with_capacity(ops.len());
        for op in ops {
            color_ops.push(ColorFilterOp::try_from_filter_op(op)?);
        }

        Ok(Some(ColorFilterPipeline { ops: color_ops }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColorFilterPipeline {
    ops: Vec<ColorFilterOp>,
}

impl ColorFilterPipeline {
    #[must_use]
    pub fn ops(&self) -> &[ColorFilterOp] {
        &self.ops
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorFilterOp {
    Brightness(FilterAmount),
    Contrast(FilterAmount),
    Grayscale(UnitFilterAmount),
    HueRotate(FilterAngle),
    Invert(UnitFilterAmount),
    Opacity(UnitFilterAmount),
    Saturate(FilterAmount),
    Sepia(UnitFilterAmount),
}

impl ColorFilterOp {
    fn try_from_filter_op(op: &FilterOp) -> std::result::Result<Self, UnsupportedPrimitive> {
        match op.kind() {
            FilterOpKind::Brightness(amount) => Ok(Self::Brightness(*amount)),
            FilterOpKind::Contrast(amount) => Ok(Self::Contrast(*amount)),
            FilterOpKind::Grayscale(amount) => Ok(Self::Grayscale(*amount)),
            FilterOpKind::HueRotate(angle) => Ok(Self::HueRotate(*angle)),
            FilterOpKind::Invert(amount) => Ok(Self::Invert(*amount)),
            FilterOpKind::Opacity(amount) => Ok(Self::Opacity(*amount)),
            FilterOpKind::Saturate(amount) => Ok(Self::Saturate(*amount)),
            FilterOpKind::Sepia(amount) => Ok(Self::Sepia(*amount)),
            FilterOpKind::Blur(_) => Err(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::GpuBlurFilterExecution,
            )),
            FilterOpKind::DropShadow(_) => Err(UnsupportedPrimitive::new(
                PrimitiveFamily::Filters,
                PrimitiveOperation::GpuDropShadowFilterExecution,
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilteredImagePaint {
    resource: ResolvedImageResource,
    filters: FilterList,
}

impl FilteredImagePaint {
    pub fn try_new(resource: ResolvedImageResource, filters: FilterList) -> Result<Self> {
        if filters.is_none() {
            return Err(Error::invalid_value(
                "filtered image paint filters",
                "none",
                "must contain at least one filter operation",
            ));
        }
        Ok(Self { resource, filters })
    }

    #[must_use]
    pub const fn resource(&self) -> &ResolvedImageResource {
        &self.resource
    }

    #[must_use]
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint,
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropCaptureBounds {
    rect: Rect,
}

impl BackdropCaptureBounds {
    pub fn try_new(rect: Rect) -> Result<Self> {
        validate_finite_f64(rect.x(), "backdrop capture bounds x")?;
        validate_finite_f64(rect.y(), "backdrop capture bounds y")?;
        if !rect.width().is_finite() || rect.width() <= 0.0 {
            return Err(Error::invalid_value(
                "backdrop capture bounds width",
                rect.width(),
                "must be finite and positive",
            ));
        }
        if !rect.height().is_finite() || rect.height() <= 0.0 {
            return Err(Error::invalid_value(
                "backdrop capture bounds height",
                rect.height(),
                "must be finite and positive",
            ));
        }
        Ok(Self { rect })
    }

    #[must_use]
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackdropFilterInput {
    filters: FilterList,
    capture_bounds: BackdropCaptureBounds,
    clip: Option<ClipInput>,
}

impl BackdropFilterInput {
    pub fn try_new(
        filters: FilterList,
        capture_bounds: BackdropCaptureBounds,
        clip: Option<ClipInput>,
    ) -> Result<Self> {
        validate_backdrop_filter_list(&filters)?;
        validate_backdrop_clip(clip.as_ref())?;
        Ok(Self {
            filters,
            capture_bounds,
            clip,
        })
    }

    pub fn try_root_backdrop(filters: FilterList, clip: Option<ClipInput>) -> Result<Self> {
        validate_backdrop_filter_list(&filters)?;
        validate_backdrop_clip(clip.as_ref())?;
        Err(Error::unsupported_render_primitive(
            UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::RootBackdropPolicy,
            ),
        ))
    }

    #[must_use]
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    #[must_use]
    pub const fn capture_bounds(&self) -> BackdropCaptureBounds {
        self.capture_bounds
    }

    #[must_use]
    pub const fn clip(&self) -> Option<&ClipInput> {
        self.clip.as_ref()
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropCapture,
        ))?;
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropFilterExecution,
        ))
    }

    pub(crate) fn ensure_supported_for_planning(&self, capabilities: Capabilities) -> Result<()> {
        validate_backdrop_filter_list(&self.filters)?;
        if let Some(clip) = &self.clip {
            clip.normalize(capabilities)?;
        }
        Ok(())
    }
}

fn validate_backdrop_filter_list(filters: &FilterList) -> Result<()> {
    if filters.is_none() {
        return Err(Error::invalid_value(
            "backdrop filter input filters",
            "none",
            "must contain at least one supported filter operation",
        ));
    }
    let _ordered_plan = filters.ordered_filter_plan();
    Ok(())
}

fn validate_backdrop_clip(clip: Option<&ClipInput>) -> Result<()> {
    let Some(clip) = clip else {
        return Ok(());
    };
    clip.ensure_supported(Capabilities::CURRENT)
}

/// An executable CSS filter drop shadow.
///
/// This normalized payload contains only an outer, zero-spread, solid-color
/// shadow. Offsets and blur are measured in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterDropShadow {
    offset: Point,
    blur: FilterBlur,
    color: Color,
}

impl FilterDropShadow {
    /// Creates an executable drop shadow from already-separated filter values.
    ///
    /// The offset and color must be finite. `blur` has already been validated
    /// against the CSS filter-blur range by [`FilterBlur::try_new`].
    pub fn try_new(offset: Point, blur: FilterBlur, color: Color) -> Result<Self> {
        validate_point(offset, "filter drop-shadow offset")?;
        validate_color(color, "filter drop-shadow color")?;
        Ok(Self {
            offset,
            blur,
            color,
        })
    }

    /// Converts a broad authored shadow into the executable filter form.
    ///
    /// Inset shadows, nonzero spread, non-solid paint, and blur outside the CSS
    /// filter range retain their existing typed diagnostics.
    pub fn try_from_shadow(shadow: Shadow) -> Result<Self> {
        if shadow.kind() == ShadowKind::Inset {
            return Err(Error::unsupported_render_primitive(
                UnsupportedPrimitive::new(
                    PrimitiveFamily::Shadows,
                    PrimitiveOperation::InsetBoxShadow,
                ),
            ));
        }
        if shadow.spread() != 0.0 {
            return Err(Error::invalid_value(
                "filter drop-shadow spread",
                shadow.spread(),
                "must be zero for CSS drop-shadow filter planning",
            ));
        }
        let color = match shadow.paint().kind() {
            PaintKind::Color(color) => *color,
            PaintKind::Gradient(_) | PaintKind::Image(_) => {
                return Err(Error::unsupported_render_primitive(
                    UnsupportedPrimitive::new(
                        PrimitiveFamily::PaintSources,
                        PrimitiveOperation::NonSolidShadowPaint,
                    ),
                ));
            }
        };
        let blur = FilterBlur::try_new(shadow.blur())?;
        Self::try_new(shadow.offset(), blur, color)
    }

    /// Returns the continuous logical-pixel offset.
    #[must_use]
    pub const fn offset(self) -> Point {
        self.offset
    }

    /// Returns the validated CSS Gaussian standard deviation.
    #[must_use]
    pub const fn blur(self) -> FilterBlur {
        self.blur
    }

    /// Returns the solid shadow color.
    #[must_use]
    pub const fn color(self) -> Color {
        self.color
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterOp {
    kind: FilterOpKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FilterOpKind {
    Blur(FilterBlur),
    Brightness(FilterAmount),
    Contrast(FilterAmount),
    Grayscale(UnitFilterAmount),
    HueRotate(FilterAngle),
    Invert(UnitFilterAmount),
    Opacity(UnitFilterAmount),
    Saturate(FilterAmount),
    Sepia(UnitFilterAmount),
    /// An intrinsically valid executable filter drop shadow.
    DropShadow(FilterDropShadow),
}

impl FilterOp {
    #[must_use]
    pub const fn blur(blur: FilterBlur) -> Self {
        Self {
            kind: FilterOpKind::Blur(blur),
        }
    }

    #[must_use]
    pub const fn brightness(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Brightness(amount),
        }
    }

    #[must_use]
    pub const fn contrast(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Contrast(amount),
        }
    }

    #[must_use]
    pub const fn grayscale(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Grayscale(amount),
        }
    }

    #[must_use]
    pub const fn hue_rotate(angle: FilterAngle) -> Self {
        Self {
            kind: FilterOpKind::HueRotate(angle),
        }
    }

    #[must_use]
    pub const fn invert(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Invert(amount),
        }
    }

    #[must_use]
    pub const fn opacity(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Opacity(amount),
        }
    }

    #[must_use]
    pub const fn saturate(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Saturate(amount),
        }
    }

    #[must_use]
    pub const fn sepia(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Sepia(amount),
        }
    }

    /// Creates a filter operation from an executable drop-shadow payload.
    #[must_use]
    pub const fn drop_shadow(shadow: FilterDropShadow) -> Self {
        Self {
            kind: FilterOpKind::DropShadow(shadow),
        }
    }

    /// Converts a broad authored shadow into a filter drop-shadow operation.
    pub fn try_drop_shadow(shadow: Shadow) -> Result<Self> {
        Ok(Self::drop_shadow(FilterDropShadow::try_from_shadow(
            shadow,
        )?))
    }

    #[must_use]
    pub const fn kind(&self) -> &FilterOpKind {
        &self.kind
    }
}

#[cfg(test)]
pub(crate) fn filter_drop_shadow_payload_accepts_shadow_for_test(shadow: Shadow) -> bool {
    FilterOp::try_drop_shadow(shadow).is_ok()
}

/// A CSS filter blur standard deviation in logical pixels.
///
/// Valid values are finite and lie in the closed interval `[0, 256]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterBlur {
    radius: f64,
}

impl FilterBlur {
    /// Creates a validated CSS filter blur without clamping.
    pub fn try_new(radius: f64) -> Result<Self> {
        if !radius.is_finite() || !(0.0..=MAX_FILTER_BLUR_RADIUS).contains(&radius) {
            return Err(Error::invalid_value(
                "filter blur radius",
                radius,
                "must be finite and between 0 and 256",
            ));
        }
        Ok(Self { radius })
    }

    /// Returns the Gaussian standard deviation in logical pixels.
    #[must_use]
    pub const fn radius(self) -> f64 {
        self.radius
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAmount {
    value: f64,
}

impl FilterAmount {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::invalid_value(
                "filter amount",
                value,
                "must be finite and non-negative",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitFilterAmount {
    value: f64,
}

impl UnitFilterAmount {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Error::invalid_value(
                "filter unit amount",
                value,
                "must be finite and between 0 and 1",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAngle {
    radians: f64,
}

impl FilterAngle {
    pub fn try_radians(radians: f64) -> Result<Self> {
        if !radians.is_finite() {
            return Err(Error::invalid_value(
                "filter angle",
                radians,
                "must be finite",
            ));
        }
        Ok(Self { radians })
    }

    #[must_use]
    pub const fn radians(self) -> f64 {
        self.radians
    }
}
