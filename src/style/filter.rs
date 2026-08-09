use super::super::{
    Capabilities, Color, Error, Point, PrimitiveFamily, PrimitiveOperation, Rect, Result, Shadow,
    ShadowKind, UnsupportedPrimitive,
    paint::PaintKind,
    validation::{validate_color, validate_finite_f64, validate_point},
};
use super::{clip::ClipInput, image::ResolvedImageResource};

const MAX_FILTER_BLUR_RADIUS: f64 = 256.0;

/// An authored filter-list state that is either `none` or non-empty.
///
/// A non-empty list preserves operation order exactly. Planning and execution
/// apply each operation to the previous operation's result; adjacent color
/// operations may be grouped internally without changing their ordered clamp
/// boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterList {
    ops: Option<Vec<FilterOp>>,
}

impl FilterList {
    /// Creates the authored `none` filter state.
    ///
    /// This is distinct from an empty operation list, which is not constructible
    /// through [`Self::try_ops`].
    #[must_use]
    pub const fn none() -> Self {
        Self { ops: None }
    }

    /// Creates a non-empty authored filter list in the supplied order.
    ///
    /// An empty vector returns [`crate::ErrorCode::InvalidInput`] with the
    /// `filter operations` diagnostic field. Each [`FilterOp`] already contains
    /// intrinsically validated operation data.
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
    /// Returns whether this is the authored `none` state.
    pub const fn is_none(&self) -> bool {
        self.ops.is_none()
    }

    #[must_use]
    /// Returns operations in authored execution order, or an empty slice for
    /// [`Self::none`].
    pub fn ops(&self) -> &[FilterOp] {
        self.ops.as_deref().unwrap_or(&[])
    }

    /// Projects this list into a color-only pipeline without reordering it.
    ///
    /// Returns `Ok(None)` for [`Self::none`] and `Ok(Some(_))` when every
    /// operation is a color operation. A blur returns
    /// the `Filters` /
    /// [`crate::PrimitiveOperation::GpuBlurFilterExecution`] identity, and a
    /// drop shadow returns `Filters` /
    /// [`crate::PrimitiveOperation::GpuDropShadowFilterExecution`], as a bare
    /// [`UnsupportedPrimitive`] rather than a crate [`Error`].
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

/// A normalized, non-empty, color-only filter pipeline.
///
/// Operations retain the source [`FilterList`] order. This projection excludes
/// blur and drop-shadow operations; use [`FilterList::color_filter_pipeline`]
/// to obtain it with the current typed diagnostic behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorFilterPipeline {
    ops: Vec<ColorFilterOp>,
}

impl ColorFilterPipeline {
    #[must_use]
    /// Returns color operations in execution order.
    pub fn ops(&self) -> &[ColorFilterOp] {
        &self.ops
    }
}

/// One intrinsically validated color-filter operation.
///
/// Amount and angle wrappers encode each operation's finite range. Variants are
/// applied in list order and retain their individual color-clamp boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorFilterOp {
    /// Multiplies source color channels by a non-negative brightness factor;
    /// `1` is the identity amount.
    Brightness(FilterAmount),
    /// Adjusts contrast by a non-negative factor; `1` is the identity amount.
    Contrast(FilterAmount),
    /// Interpolates toward grayscale by a unit-interval amount; `0` is the
    /// identity amount.
    Grayscale(UnitFilterAmount),
    /// Rotates hue by a finite angle in radians without normalizing the stored
    /// angle.
    HueRotate(FilterAngle),
    /// Interpolates toward inverted color by a unit-interval amount; `0` is the
    /// identity amount.
    Invert(UnitFilterAmount),
    /// Multiplies alpha by a unit-interval amount; `1` is the identity amount.
    Opacity(UnitFilterAmount),
    /// Adjusts saturation by a non-negative factor; `1` is the identity amount.
    Saturate(FilterAmount),
    /// Interpolates toward sepia by a unit-interval amount; `0` is the identity
    /// amount.
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

/// A resolved image resource paired with a non-empty authored filter list.
///
/// This model is currently diagnostic-only at the image-paint boundary:
/// [`Self::ensure_supported`] reports
/// [`crate::PrimitiveOperation::FilteredImagePaint`] for
/// [`Capabilities::CURRENT`]. It does not imply image-filter execution.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredImagePaint {
    resource: ResolvedImageResource,
    filters: FilterList,
}

impl FilteredImagePaint {
    /// Creates filtered image-paint input from a resolved resource and filters.
    ///
    /// The authored `none` state returns [`crate::ErrorCode::InvalidInput`] with
    /// the `filtered image paint filters` diagnostic field. A non-empty list is
    /// retained unchanged; support is checked separately.
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
    /// Returns the resolved image resource.
    pub const fn resource(&self) -> &ResolvedImageResource {
        &self.resource
    }

    #[must_use]
    /// Returns the non-empty filter list in authored order.
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    /// Checks current semantic support for filtered image paint.
    ///
    /// Unsupported capabilities return [`crate::ErrorCode::UnsupportedPrimitive`]
    /// carrying the `ImageSampling` / `FilteredImagePaint` identity. In
    /// particular, [`Capabilities::CURRENT`] currently returns that diagnostic.
    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint,
        ))
    }
}

/// A finite positive-area logical rectangle captured for bounded backdrop filtering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackdropCaptureBounds {
    rect: Rect,
}

impl BackdropCaptureBounds {
    /// Creates bounded backdrop capture geometry.
    ///
    /// The rectangle origin must be finite and both dimensions must be finite
    /// and positive. Violations return [`crate::ErrorCode::InvalidInput`] with
    /// the corresponding `backdrop capture bounds ...` diagnostic field.
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
    /// Returns the logical capture rectangle.
    pub const fn rect(self) -> Rect {
        self.rect
    }
}

/// Validated input for filtering one explicitly bounded backdrop region.
///
/// The non-empty filter list executes in authored order over the captured
/// logical rectangle. An optional validated clip is applied to the filtered
/// backdrop result. This bounded form is supported by the current semantic
/// capability contract; unbounded root backdrop policy is diagnostic-only.
#[derive(Clone, Debug, PartialEq)]
pub struct BackdropFilterInput {
    filters: FilterList,
    capture_bounds: BackdropCaptureBounds,
    clip: Option<ClipInput>,
}

impl BackdropFilterInput {
    /// Creates a bounded backdrop-filter input.
    ///
    /// A `none` filter list returns [`crate::ErrorCode::InvalidInput`] with the
    /// `backdrop filter input filters` diagnostic field. An unsupported clip
    /// returns its existing typed unsupported diagnostic. The capture bounds
    /// are already intrinsically validated by [`BackdropCaptureBounds::try_new`].
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

    /// Validates root-backdrop inputs and returns the current policy diagnostic.
    ///
    /// This constructor never returns `Ok`: after rejecting a `none` list or an
    /// unsupported clip as applicable, it returns
    /// [`crate::ErrorCode::UnsupportedPrimitive`] carrying `Compositing` /
    /// [`crate::PrimitiveOperation::RootBackdropPolicy`].
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
    /// Returns the non-empty filter list in authored execution order.
    pub const fn filters(&self) -> &FilterList {
        &self.filters
    }

    #[must_use]
    /// Returns the bounded logical capture region.
    pub const fn capture_bounds(&self) -> BackdropCaptureBounds {
        self.capture_bounds
    }

    #[must_use]
    /// Returns the optional post-filter clip input.
    pub const fn clip(&self) -> Option<&ClipInput> {
        self.clip.as_ref()
    }

    /// Checks the two semantic capabilities required by bounded backdrop filtering.
    ///
    /// The check first requires `OffscreenPipeline` /
    /// [`crate::PrimitiveOperation::BoundedBackdropCapture`], then
    /// `OffscreenPipeline` /
    /// [`crate::PrimitiveOperation::BoundedBackdropFilterExecution`], returning
    /// the first unavailable operation as an `UnsupportedPrimitive` error. Both
    /// operations are supported by [`Capabilities::CURRENT`].
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
    /// The offset and every color channel must be finite, and color channels
    /// must lie in `0..=1`; violations return
    /// [`crate::ErrorCode::InvalidInput`]. `blur` has already been validated
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
    /// An inset shadow returns the `Shadows` /
    /// [`crate::PrimitiveOperation::InsetBoxShadow`] unsupported diagnostic. A
    /// nonzero spread returns [`crate::ErrorCode::InvalidInput`] for `filter
    /// drop-shadow spread`. Gradient or image paint returns the `PaintSources`
    /// / [`crate::PrimitiveOperation::NonSolidShadowPaint`] unsupported
    /// diagnostic, and blur outside `0..=256` returns its filter-blur input
    /// diagnostic.
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

/// One authored filter operation with intrinsically validated payload data.
///
/// Constructors preserve the chosen operation without execution or reordering.
/// Ordered execution is supplied by [`FilterList`].
#[derive(Clone, Debug, PartialEq)]
pub struct FilterOp {
    kind: FilterOpKind,
}

/// The closed operation choice stored by [`FilterOp`].
#[derive(Clone, Debug, PartialEq)]
pub enum FilterOpKind {
    /// Applies Gaussian blur with a logical-pixel standard deviation.
    Blur(FilterBlur),
    /// Applies a non-negative brightness factor.
    Brightness(FilterAmount),
    /// Applies a non-negative contrast factor.
    Contrast(FilterAmount),
    /// Applies a unit-interval grayscale amount.
    Grayscale(UnitFilterAmount),
    /// Applies a finite hue rotation in radians.
    HueRotate(FilterAngle),
    /// Applies a unit-interval color-inversion amount.
    Invert(UnitFilterAmount),
    /// Applies a unit-interval opacity multiplier.
    Opacity(UnitFilterAmount),
    /// Applies a non-negative saturation factor.
    Saturate(FilterAmount),
    /// Applies a unit-interval sepia amount.
    Sepia(UnitFilterAmount),
    /// An intrinsically valid executable filter drop shadow.
    DropShadow(FilterDropShadow),
}

impl FilterOp {
    /// Creates a blur operation from a validated logical-pixel standard deviation.
    #[must_use]
    pub const fn blur(blur: FilterBlur) -> Self {
        Self {
            kind: FilterOpKind::Blur(blur),
        }
    }

    /// Creates a brightness operation from a validated non-negative factor.
    #[must_use]
    pub const fn brightness(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Brightness(amount),
        }
    }

    /// Creates a contrast operation from a validated non-negative factor.
    #[must_use]
    pub const fn contrast(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Contrast(amount),
        }
    }

    /// Creates a grayscale operation from a validated unit-interval amount.
    #[must_use]
    pub const fn grayscale(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Grayscale(amount),
        }
    }

    /// Creates a hue-rotation operation from a validated radian angle.
    #[must_use]
    pub const fn hue_rotate(angle: FilterAngle) -> Self {
        Self {
            kind: FilterOpKind::HueRotate(angle),
        }
    }

    /// Creates an inversion operation from a validated unit-interval amount.
    #[must_use]
    pub const fn invert(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Invert(amount),
        }
    }

    /// Creates an opacity operation from a validated unit-interval multiplier.
    #[must_use]
    pub const fn opacity(amount: UnitFilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Opacity(amount),
        }
    }

    /// Creates a saturation operation from a validated non-negative factor.
    #[must_use]
    pub const fn saturate(amount: FilterAmount) -> Self {
        Self {
            kind: FilterOpKind::Saturate(amount),
        }
    }

    /// Creates a sepia operation from a validated unit-interval amount.
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
    ///
    /// This preserves the exact diagnostics documented by
    /// [`FilterDropShadow::try_from_shadow`].
    pub fn try_drop_shadow(shadow: Shadow) -> Result<Self> {
        Ok(Self::drop_shadow(FilterDropShadow::try_from_shadow(
            shadow,
        )?))
    }

    #[must_use]
    /// Returns the stored operation choice.
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

/// A finite non-negative factor for brightness, contrast, or saturation.
///
/// The value is not clamped; every finite representable non-negative value is
/// accepted. For each of these factor operations, `1` is the identity amount.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAmount {
    value: f64,
}

impl FilterAmount {
    /// Creates a non-negative filter factor without clamping.
    ///
    /// A negative or non-finite value returns
    /// [`crate::ErrorCode::InvalidInput`] with the `filter amount` diagnostic
    /// field.
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
    /// Returns the finite non-negative factor.
    pub const fn value(self) -> f64 {
        self.value
    }
}

/// A finite filter amount in the inclusive unit interval `0..=1`.
///
/// Grayscale, invert, opacity, and sepia use this intrinsically bounded value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitFilterAmount {
    value: f64,
}

impl UnitFilterAmount {
    /// Creates a unit-interval filter amount without clamping.
    ///
    /// A non-finite value or one outside `0..=1` returns
    /// [`crate::ErrorCode::InvalidInput`] with the `filter unit amount`
    /// diagnostic field.
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
    /// Returns the finite amount in `0..=1`.
    pub const fn value(self) -> f64 {
        self.value
    }
}

/// A finite hue-rotation angle stored in radians.
///
/// Construction does not wrap or otherwise normalize the angle, so negative
/// and multi-turn values retain their authored numeric value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterAngle {
    radians: f64,
}

impl FilterAngle {
    /// Creates a hue-rotation angle from radians.
    ///
    /// A non-finite angle returns [`crate::ErrorCode::InvalidInput`] with the
    /// `filter angle` diagnostic field.
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
    /// Returns the finite, unnormalized angle in radians.
    pub const fn radians(self) -> f64 {
        self.radians
    }
}
