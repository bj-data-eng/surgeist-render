use super::super::{
    Capabilities, Error, Paint, PrimitiveFamily, PrimitiveOperation, Radii, Rect, Result,
    UnsupportedPrimitive,
    validation::{validate_finite_f64, validate_non_negative_f64, validate_paint},
};
use super::image::{BackgroundAreas, BackgroundClipGeometry, validate_background_rect};

/// One authored border edge with a style, logical-pixel width, and paint.
///
/// Construction guarantees a finite non-negative width and a valid paint. The
/// style remains authored until [`BoxDecorationInput::normalize`] either emits
/// normalized command data, suppresses the edge, or reports an unsupported
/// style.
#[derive(Clone, Debug, PartialEq)]
pub struct BorderSide {
    style: BorderStyle,
    width: f64,
    paint: Paint,
}

/// The authored line style of one border edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BorderStyle {
    /// Suppresses the border edge during normalization.
    None,
    /// Suppresses the border edge during normalization.
    Hidden,
    /// Selects a solid normalized border.
    Solid,
    /// Selects a dashed normalized border.
    Dashed,
    /// Selects a dotted normalized border.
    Dotted,
    /// Selects a normalized three-band double border.
    Double,
    /// Selects a groove border, which currently produces a
    /// [`crate::PrimitiveOperation::BorderGrooveStyle`] unsupported diagnostic
    /// during normalization.
    Groove,
    /// Selects a ridge border, which currently produces a
    /// [`crate::PrimitiveOperation::BorderRidgeStyle`] unsupported diagnostic
    /// during normalization.
    Ridge,
    /// Selects an inset border, which currently produces a
    /// [`crate::PrimitiveOperation::BorderInsetStyle`] unsupported diagnostic
    /// during normalization.
    Inset,
    /// Selects an outset border, which currently produces a
    /// [`crate::PrimitiveOperation::BorderOutsetStyle`] unsupported diagnostic
    /// during normalization.
    Outset,
}

impl BorderSide {
    /// Creates an authored border edge.
    ///
    /// `width` is measured in logical pixels. Returns
    /// [`crate::ErrorCode::InvalidInput`] if it is negative or non-finite, or
    /// if the converted paint violates its intrinsic invariants.
    pub fn try_new(style: BorderStyle, width: f64, paint: impl Into<Paint>) -> Result<Self> {
        validate_non_negative_f64(width, "border side width")?;
        let paint = paint.into();
        validate_paint(&paint)?;
        Ok(Self {
            style,
            width,
            paint,
        })
    }

    #[must_use]
    /// Returns the authored border style.
    pub const fn style(&self) -> BorderStyle {
        self.style
    }

    #[must_use]
    /// Returns the finite non-negative width in logical pixels.
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns the validated border paint.
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

/// Four independently authored border edges.
///
/// Normalization visits them in top, right, bottom, then left order for each
/// box-decoration fragment.
#[derive(Clone, Debug, PartialEq)]
pub struct BorderEdges {
    top: BorderSide,
    right: BorderSide,
    bottom: BorderSide,
    left: BorderSide,
}

impl BorderEdges {
    /// Groups four already validated border edges by physical side.
    #[must_use]
    pub const fn new(
        top: BorderSide,
        right: BorderSide,
        bottom: BorderSide,
        left: BorderSide,
    ) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    #[must_use]
    /// Returns the top edge.
    pub const fn top(&self) -> &BorderSide {
        &self.top
    }

    #[must_use]
    /// Returns the right edge.
    pub const fn right(&self) -> &BorderSide {
        &self.right
    }

    #[must_use]
    /// Returns the bottom edge.
    pub const fn bottom(&self) -> &BorderSide {
        &self.bottom
    }

    #[must_use]
    /// Returns the left edge.
    pub const fn left(&self) -> &BorderSide {
        &self.left
    }
}

/// An authored outline around each box-decoration fragment.
///
/// Width and offset are logical-pixel values. Normalization suppresses `None`
/// and zero-width outlines, accepts solid, dashed, and dotted styles, and
/// reports typed unsupported diagnostics for double and automatic styles.
#[derive(Clone, Debug, PartialEq)]
pub struct Outline {
    style: OutlineStyle,
    width: f64,
    paint: Paint,
    offset: f64,
}

/// The authored line style of an outline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutlineStyle {
    /// Suppresses the outline during normalization.
    None,
    /// Selects a solid normalized outline.
    Solid,
    /// Selects a dashed normalized outline.
    Dashed,
    /// Selects a dotted normalized outline.
    Dotted,
    /// Selects a double outline, which currently produces a
    /// [`crate::PrimitiveOperation::OutlineDoubleStyle`] unsupported diagnostic
    /// during normalization.
    Double,
    /// Selects an automatic outline, which currently produces a
    /// [`crate::PrimitiveOperation::OutlineAutoStyle`] unsupported diagnostic
    /// during normalization.
    Auto,
}

impl Outline {
    /// Creates an authored outline.
    ///
    /// `width` must be finite and non-negative, `offset` must be finite, and
    /// both are measured in logical pixels. Invalid numeric input or paint
    /// returns [`crate::ErrorCode::InvalidInput`]. Negative offsets are accepted
    /// here but can be rejected by normalization if they contract a fragment to
    /// a non-positive target rectangle.
    pub fn try_new(
        style: OutlineStyle,
        width: f64,
        paint: impl Into<Paint>,
        offset: f64,
    ) -> Result<Self> {
        validate_non_negative_f64(width, "outline width")?;
        validate_finite_f64(offset, "outline offset")?;
        let paint = paint.into();
        validate_paint(&paint)?;
        Ok(Self {
            style,
            width,
            paint,
            offset,
        })
    }

    #[must_use]
    /// Returns the authored outline style.
    pub const fn style(&self) -> OutlineStyle {
        self.style
    }

    #[must_use]
    /// Returns the finite non-negative width in logical pixels.
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns the validated outline paint.
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    /// Returns the finite logical-pixel offset from the border box.
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

/// The authored fragmentation mode carried into normalized decoration commands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxDecorationBreak {
    /// Carries slice semantics for this fragment.
    Slice,
    /// Carries clone semantics for this fragment.
    Clone,
}

/// Corner radii normalized against one finite positive-area border box.
///
/// Radii are non-negative logical-pixel values. When adjacent requested radii
/// exceed an available box dimension, all four radii are scaled by one common
/// factor so every adjacent pair fits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedBoxRadii {
    border_box: Rect,
    radii: Radii,
}

impl NormalizedBoxRadii {
    /// Validates a logical border box and normalizes its radii.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for a non-finite origin, a
    /// non-positive or non-finite box dimension, or any negative or non-finite
    /// radius.
    pub fn try_new(border_box: Rect, radii: Radii) -> Result<Self> {
        validate_background_rect(border_box, "box decoration border box")?;
        validate_box_decoration_radii(radii)?;
        Ok(Self {
            border_box,
            radii: scale_box_radii(border_box, radii),
        })
    }

    #[must_use]
    /// Returns the logical border box used as the normalization basis.
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    #[must_use]
    /// Returns the non-negative radii after proportional scaling, if required.
    pub const fn radii(self) -> Radii {
        self.radii
    }
}

/// One validated fragment used to normalize border and outline command data.
///
/// The background areas and radii are expressed in logical coordinates. By
/// default the border box supplies rectangular clip geometry; a validated clip
/// override can replace it without changing the fragment's areas or radii.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationFragment {
    areas: BackgroundAreas,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
    border_clip_override: Option<BackgroundClipGeometry>,
}

impl BoxDecorationFragment {
    /// Creates a fragment and normalizes `radii` against its border box.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the border box or radii
    /// violate [`NormalizedBoxRadii`] invariants.
    pub fn try_new(
        areas: BackgroundAreas,
        radii: Radii,
        break_mode: BoxDecorationBreak,
    ) -> Result<Self> {
        Ok(Self {
            areas,
            radii: NormalizedBoxRadii::try_new(areas.border_box(), radii)?,
            break_mode,
            border_clip_override: None,
        })
    }

    #[must_use]
    /// Returns this fragment with validated border clip geometry installed.
    ///
    /// The new geometry replaces any previous override.
    pub fn with_border_clip_override(
        mut self,
        border_clip_override: BackgroundClipGeometry,
    ) -> Self {
        self.border_clip_override = Some(border_clip_override);
        self
    }

    #[must_use]
    /// Returns the fragment's logical border, padding, and content areas.
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    #[must_use]
    /// Returns the radii normalized against the fragment's border box.
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    /// Returns the authored fragmentation mode.
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }

    #[must_use]
    /// Returns the explicit border clip override, if one was installed.
    pub const fn border_clip_override(&self) -> Option<&BackgroundClipGeometry> {
        self.border_clip_override.as_ref()
    }
}

/// Authored border and outline facts for one or more validated fragments.
///
/// Normalization preserves fragment order. Within each fragment it emits
/// non-suppressed borders in top, right, bottom, left order, followed by the
/// outline when present and non-suppressed.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationInput {
    border_edges: Option<BorderEdges>,
    outline: Option<Outline>,
    fragments: Vec<BoxDecorationFragment>,
}

impl BoxDecorationInput {
    /// Creates authored box-decoration input with at least one fragment.
    ///
    /// An empty fragment list returns [`crate::ErrorCode::InvalidInput`] with
    /// the `box decoration fragments` diagnostic field. Border, outline, and
    /// fragment values have already been validated by their constructors.
    pub fn try_new(
        border_edges: Option<BorderEdges>,
        outline: Option<Outline>,
        fragments: Vec<BoxDecorationFragment>,
    ) -> Result<Self> {
        if fragments.is_empty() {
            return Err(Error::invalid_value(
                "box decoration fragments",
                "[]",
                "must contain at least one fragment",
            ));
        }
        Ok(Self {
            border_edges,
            outline,
            fragments,
        })
    }

    #[must_use]
    /// Returns the four authored border edges, if present.
    pub const fn border_edges(&self) -> Option<&BorderEdges> {
        self.border_edges.as_ref()
    }

    #[must_use]
    /// Returns the authored outline, if present.
    pub const fn outline(&self) -> Option<&Outline> {
        self.outline.as_ref()
    }

    #[must_use]
    /// Returns the non-empty fragments in authored order.
    pub fn fragments(&self) -> &[BoxDecorationFragment] {
        &self.fragments
    }

    /// Converts authored decoration facts into ordered normalized command data.
    ///
    /// `None`, `Hidden`, and zero-width borders, plus `None` and zero-width
    /// outlines, emit no command. Groove, ridge, inset, and outset borders and
    /// double or automatic outlines return their exact `BoxDecorations`
    /// [`crate::UnsupportedPrimitive`] diagnostic. A finite negative outline
    /// offset returns [`crate::ErrorCode::InvalidInput`] if the resulting target
    /// rectangle is non-finite or has non-positive extent. The current
    /// normalization is context-free; `capabilities` is reserved input and does
    /// not alter these results.
    pub fn normalize(&self, _capabilities: Capabilities) -> Result<NormalizedBoxDecoration> {
        let mut commands = Vec::new();

        for (fragment_index, fragment) in self.fragments.iter().enumerate() {
            let target_rect = fragment.areas().border_box();
            let clip = border_clip_geometry(fragment)?;

            if let Some(border_edges) = &self.border_edges {
                for (side, border_side) in border_sides(border_edges) {
                    if let Some(style) = normalize_border_style(border_side)? {
                        commands.push(NormalizedBoxDecorationCommand {
                            kind: NormalizedBoxDecorationCommandKind::Border(
                                NormalizedBorderCommand {
                                    fragment_index,
                                    side,
                                    width: border_side.width(),
                                    paint: border_side.paint().clone(),
                                    style,
                                    target_rect,
                                    clip: clip.clone(),
                                    radii: fragment.radii(),
                                    break_mode: fragment.break_mode(),
                                },
                            ),
                        });
                    }
                }
            }

            if let Some(outline) = &self.outline
                && let Some(style) = normalize_outline_style(outline)?
            {
                commands.push(NormalizedBoxDecorationCommand {
                    kind: NormalizedBoxDecorationCommandKind::Outline(NormalizedOutlineCommand {
                        fragment_index,
                        width: outline.width(),
                        paint: outline.paint().clone(),
                        offset: outline.offset(),
                        style,
                        target_rect: outline_target_rect(target_rect, outline.offset())?,
                        clip,
                        radii: fragment.radii(),
                        break_mode: fragment.break_mode(),
                    }),
                });
            }
        }

        Ok(NormalizedBoxDecoration { commands })
    }
}

/// A physical side selected by a normalized border command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxSide {
    /// The top border edge.
    Top,
    /// The right border edge.
    Right,
    /// The bottom border edge.
    Bottom,
    /// The left border edge.
    Left,
}

/// Logical-pixel band widths for a normalized double border.
///
/// The outer and gap widths are each one third of the original width. The inner
/// width is the remainder, so the three non-negative bands sum to the original
/// finite non-negative width without integer rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedDoubleBorderBands {
    original_width: f64,
    outer_width: f64,
    gap_width: f64,
    inner_width: f64,
}

impl NormalizedDoubleBorderBands {
    #[must_use]
    fn from_width(width: f64) -> Self {
        let outer_width = width / 3.0;
        let gap_width = width / 3.0;
        let inner_width = width - outer_width - gap_width;
        Self {
            original_width: width,
            outer_width,
            gap_width,
            inner_width,
        }
    }

    #[must_use]
    /// Returns the original logical-pixel border width.
    pub const fn original_width(self) -> f64 {
        self.original_width
    }

    #[must_use]
    /// Returns the outer painted band's width in logical pixels.
    pub const fn outer_width(self) -> f64 {
        self.outer_width
    }

    #[must_use]
    /// Returns the unpainted gap width in logical pixels.
    pub const fn gap_width(self) -> f64 {
        self.gap_width
    }

    #[must_use]
    /// Returns the inner painted band's width in logical pixels.
    pub const fn inner_width(self) -> f64 {
        self.inner_width
    }
}

/// A border style accepted into normalized box-decoration command data.
#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBorderStyle {
    /// A solid border style.
    Solid,
    /// A dashed border style, preserved without defining a dash algorithm here.
    Dashed,
    /// A dotted border style, preserved without defining a dot algorithm here.
    Dotted,
    /// A double border with three normalized logical-pixel bands.
    Double(NormalizedDoubleBorderBands),
}

/// An outline style accepted into normalized box-decoration command data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizedOutlineStyle {
    /// A solid outline style.
    Solid,
    /// A dashed outline style, preserved without defining a dash algorithm here.
    Dashed,
    /// A dotted outline style, preserved without defining a dot algorithm here.
    Dotted,
}

/// Normalized backend-facing data for one border edge of one fragment.
///
/// Geometry and widths are in logical coordinates. The value records validated
/// paint, clip geometry, normalized corner radii, and fragmentation mode; it
/// does not prescribe a border rasterization or dash-placement algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBorderCommand {
    fragment_index: usize,
    side: BoxSide,
    width: f64,
    paint: Paint,
    style: NormalizedBorderStyle,
    target_rect: Rect,
    clip: BackgroundClipGeometry,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
}

impl NormalizedBorderCommand {
    #[must_use]
    /// Returns the zero-based index of the source fragment.
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
    }

    #[must_use]
    /// Returns the physical border side.
    pub const fn side(&self) -> BoxSide {
        self.side
    }

    #[must_use]
    /// Returns the finite non-negative width in logical pixels.
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns the validated border paint.
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    /// Returns the normalized border style.
    pub const fn style(&self) -> &NormalizedBorderStyle {
        &self.style
    }

    #[must_use]
    /// Returns the fragment border box in logical coordinates.
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    /// Returns the border clip geometry selected for the fragment.
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    /// Returns the corner radii normalized against the fragment border box.
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    /// Returns the source fragment's break mode.
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

/// Normalized backend-facing data for one outline around one fragment.
///
/// Width, offset, and target geometry are in logical coordinates. The target
/// rectangle is expanded or contracted by the offset only; the outline width
/// remains a separate value. This type does not prescribe a line or dash
/// rasterization algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedOutlineCommand {
    fragment_index: usize,
    width: f64,
    paint: Paint,
    offset: f64,
    style: NormalizedOutlineStyle,
    target_rect: Rect,
    clip: BackgroundClipGeometry,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
}

impl NormalizedOutlineCommand {
    #[must_use]
    /// Returns the zero-based index of the source fragment.
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
    }

    #[must_use]
    /// Returns the finite non-negative width in logical pixels.
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    /// Returns the validated outline paint.
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    /// Returns the finite logical-pixel offset from the border box.
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    #[must_use]
    /// Returns the normalized outline style.
    pub const fn style(&self) -> NormalizedOutlineStyle {
        self.style
    }

    #[must_use]
    /// Returns the finite positive-area logical rectangle derived from the
    /// fragment border box and outline offset.
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    /// Returns the fragment's border clip geometry.
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    /// Returns the corner radii normalized against the fragment border box.
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    /// Returns the source fragment's break mode.
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

/// Ordered normalized border and outline command data.
///
/// Commands retain fragment order; each fragment's border commands precede its
/// optional outline command. An authored input whose styles are all suppressed
/// produces an empty command slice.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecoration {
    commands: Vec<NormalizedBoxDecorationCommand>,
}

impl NormalizedBoxDecoration {
    #[must_use]
    /// Returns normalized commands in deterministic emission order.
    pub fn commands(&self) -> &[NormalizedBoxDecorationCommand] {
        &self.commands
    }
}

/// One normalized box-decoration command.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecorationCommand {
    kind: NormalizedBoxDecorationCommandKind,
}

impl NormalizedBoxDecorationCommand {
    #[must_use]
    /// Returns the command payload.
    pub const fn kind(&self) -> &NormalizedBoxDecorationCommandKind {
        &self.kind
    }
}

/// The normalized payload selected for a box-decoration command.
#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBoxDecorationCommandKind {
    /// Carries one normalized border-edge command.
    Border(NormalizedBorderCommand),
    /// Carries one normalized outline command.
    Outline(NormalizedOutlineCommand),
}

fn border_sides(edges: &BorderEdges) -> [(BoxSide, &BorderSide); 4] {
    [
        (BoxSide::Top, edges.top()),
        (BoxSide::Right, edges.right()),
        (BoxSide::Bottom, edges.bottom()),
        (BoxSide::Left, edges.left()),
    ]
}

fn border_clip_geometry(fragment: &BoxDecorationFragment) -> Result<BackgroundClipGeometry> {
    if let Some(clip) = fragment.border_clip_override() {
        return Ok(clip.clone());
    }
    BackgroundClipGeometry::try_rect(fragment.areas().border_box())
}

fn normalize_border_style(side: &BorderSide) -> Result<Option<NormalizedBorderStyle>> {
    if side.width() == 0.0 || matches!(side.style(), BorderStyle::None | BorderStyle::Hidden) {
        return Ok(None);
    }

    let style = match side.style() {
        BorderStyle::None | BorderStyle::Hidden => unreachable!("suppressed before style mapping"),
        BorderStyle::Solid => NormalizedBorderStyle::Solid,
        BorderStyle::Dashed => NormalizedBorderStyle::Dashed,
        BorderStyle::Dotted => NormalizedBorderStyle::Dotted,
        BorderStyle::Double => {
            NormalizedBorderStyle::Double(NormalizedDoubleBorderBands::from_width(side.width()))
        }
        BorderStyle::Groove => {
            return unsupported_border_style(PrimitiveOperation::BorderGrooveStyle);
        }
        BorderStyle::Ridge => {
            return unsupported_border_style(PrimitiveOperation::BorderRidgeStyle);
        }
        BorderStyle::Inset => {
            return unsupported_border_style(PrimitiveOperation::BorderInsetStyle);
        }
        BorderStyle::Outset => {
            return unsupported_border_style(PrimitiveOperation::BorderOutsetStyle);
        }
    };

    Ok(Some(style))
}

fn unsupported_border_style(
    operation: PrimitiveOperation,
) -> Result<Option<NormalizedBorderStyle>> {
    let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
    Err(Error::unsupported_render_primitive(unsupported))
}

fn normalize_outline_style(outline: &Outline) -> Result<Option<NormalizedOutlineStyle>> {
    if outline.width() == 0.0 || matches!(outline.style(), OutlineStyle::None) {
        return Ok(None);
    }

    let style = match outline.style() {
        OutlineStyle::None => unreachable!("suppressed before style mapping"),
        OutlineStyle::Solid => NormalizedOutlineStyle::Solid,
        OutlineStyle::Dashed => NormalizedOutlineStyle::Dashed,
        OutlineStyle::Dotted => NormalizedOutlineStyle::Dotted,
        OutlineStyle::Double => {
            return unsupported_outline_style(PrimitiveOperation::OutlineDoubleStyle);
        }
        OutlineStyle::Auto => {
            return unsupported_outline_style(PrimitiveOperation::OutlineAutoStyle);
        }
    };

    Ok(Some(style))
}

fn unsupported_outline_style(
    operation: PrimitiveOperation,
) -> Result<Option<NormalizedOutlineStyle>> {
    let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
    Err(Error::unsupported_render_primitive(unsupported))
}

fn outline_target_rect(border_box: Rect, offset: f64) -> Result<Rect> {
    let x = border_box.x() - offset;
    let y = border_box.y() - offset;
    let width = border_box.width() + offset * 2.0;
    let height = border_box.height() + offset * 2.0;

    if !x.is_finite()
        || !y.is_finite()
        || !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
    {
        return Err(Error::invalid_value(
            "outline target rect",
            format!("border box {border_box:?}, offset {offset}"),
            "must resolve to finite positive width and height",
        ));
    }

    Ok(Rect::new(x, y, width, height))
}

fn validate_box_decoration_radii(radii: Radii) -> Result<()> {
    for (field, value) in [
        ("box decoration top-left radius", radii.top_left()),
        ("box decoration top-right radius", radii.top_right()),
        ("box decoration bottom-right radius", radii.bottom_right()),
        ("box decoration bottom-left radius", radii.bottom_left()),
    ] {
        validate_non_negative_f64(value, field)?;
    }
    Ok(())
}

fn scale_box_radii(border_box: Rect, radii: Radii) -> Radii {
    let mut scale: f64 = 1.0;
    scale = scale.min(corner_scale(
        border_box.width(),
        radii.top_left() + radii.top_right(),
    ));
    scale = scale.min(corner_scale(
        border_box.width(),
        radii.bottom_left() + radii.bottom_right(),
    ));
    scale = scale.min(corner_scale(
        border_box.height(),
        radii.top_left() + radii.bottom_left(),
    ));
    scale = scale.min(corner_scale(
        border_box.height(),
        radii.top_right() + radii.bottom_right(),
    ));

    if scale >= 1.0 {
        return radii;
    }

    Radii::new(
        radii.top_left() * scale,
        radii.top_right() * scale,
        radii.bottom_right() * scale,
        radii.bottom_left() * scale,
    )
}

fn corner_scale(available: f64, requested: f64) -> f64 {
    if requested <= available || requested == 0.0 {
        1.0
    } else {
        available / requested
    }
}
