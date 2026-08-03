use super::super::{
    Capabilities, Error, Paint, PrimitiveFamily, PrimitiveOperation, Radii, Rect, Result,
    UnsupportedPrimitive,
    validation::{validate_finite_f64, validate_non_negative_f64, validate_paint},
};
use super::image::{BackgroundAreas, BackgroundClipGeometry, validate_background_rect};

#[derive(Clone, Debug, PartialEq)]
pub struct BorderSide {
    style: BorderStyle,
    width: f64,
    paint: Paint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BorderStyle {
    None,
    Hidden,
    Solid,
    Dashed,
    Dotted,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
}

impl BorderSide {
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
    pub const fn style(&self) -> BorderStyle {
        self.style
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BorderEdges {
    top: BorderSide,
    right: BorderSide,
    bottom: BorderSide,
    left: BorderSide,
}

impl BorderEdges {
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
    pub const fn top(&self) -> &BorderSide {
        &self.top
    }

    #[must_use]
    pub const fn right(&self) -> &BorderSide {
        &self.right
    }

    #[must_use]
    pub const fn bottom(&self) -> &BorderSide {
        &self.bottom
    }

    #[must_use]
    pub const fn left(&self) -> &BorderSide {
        &self.left
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outline {
    style: OutlineStyle,
    width: f64,
    paint: Paint,
    offset: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutlineStyle {
    None,
    Solid,
    Dashed,
    Dotted,
    Double,
    Auto,
}

impl Outline {
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
    pub const fn style(&self) -> OutlineStyle {
        self.style
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxDecorationBreak {
    Slice,
    Clone,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizedBoxRadii {
    border_box: Rect,
    radii: Radii,
}

impl NormalizedBoxRadii {
    pub fn try_new(border_box: Rect, radii: Radii) -> Result<Self> {
        validate_background_rect(border_box, "box decoration border box")?;
        validate_box_decoration_radii(radii)?;
        Ok(Self {
            border_box,
            radii: scale_box_radii(border_box, radii),
        })
    }

    #[must_use]
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    #[must_use]
    pub const fn radii(self) -> Radii {
        self.radii
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationFragment {
    areas: BackgroundAreas,
    radii: NormalizedBoxRadii,
    break_mode: BoxDecorationBreak,
    border_clip_override: Option<BackgroundClipGeometry>,
}

impl BoxDecorationFragment {
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
    pub fn with_border_clip_override(
        mut self,
        border_clip_override: BackgroundClipGeometry,
    ) -> Self {
        self.border_clip_override = Some(border_clip_override);
        self
    }

    #[must_use]
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }

    #[must_use]
    pub const fn border_clip_override(&self) -> Option<&BackgroundClipGeometry> {
        self.border_clip_override.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxDecorationInput {
    border_edges: Option<BorderEdges>,
    outline: Option<Outline>,
    fragments: Vec<BoxDecorationFragment>,
}

impl BoxDecorationInput {
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
    pub const fn border_edges(&self) -> Option<&BorderEdges> {
        self.border_edges.as_ref()
    }

    #[must_use]
    pub const fn outline(&self) -> Option<&Outline> {
        self.outline.as_ref()
    }

    #[must_use]
    pub fn fragments(&self) -> &[BoxDecorationFragment] {
        &self.fragments
    }

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxSide {
    Top,
    Right,
    Bottom,
    Left,
}

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
    pub const fn original_width(self) -> f64 {
        self.original_width
    }

    #[must_use]
    pub const fn outer_width(self) -> f64 {
        self.outer_width
    }

    #[must_use]
    pub const fn gap_width(self) -> f64 {
        self.gap_width
    }

    #[must_use]
    pub const fn inner_width(self) -> f64 {
        self.inner_width
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBorderStyle {
    Solid,
    Dashed,
    Dotted,
    Double(NormalizedDoubleBorderBands),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizedOutlineStyle {
    Solid,
    Dashed,
    Dotted,
}

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
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
    }

    #[must_use]
    pub const fn side(&self) -> BoxSide {
        self.side
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    pub const fn style(&self) -> &NormalizedBorderStyle {
        &self.style
    }

    #[must_use]
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

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
    pub const fn fragment_index(&self) -> usize {
        self.fragment_index
    }

    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn paint(&self) -> &Paint {
        &self.paint
    }

    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    #[must_use]
    pub const fn style(&self) -> NormalizedOutlineStyle {
        self.style
    }

    #[must_use]
    pub const fn target_rect(&self) -> Rect {
        self.target_rect
    }

    #[must_use]
    pub const fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub const fn radii(&self) -> NormalizedBoxRadii {
        self.radii
    }

    #[must_use]
    pub const fn break_mode(&self) -> BoxDecorationBreak {
        self.break_mode
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecoration {
    commands: Vec<NormalizedBoxDecorationCommand>,
}

impl NormalizedBoxDecoration {
    #[must_use]
    pub fn commands(&self) -> &[NormalizedBoxDecorationCommand] {
        &self.commands
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBoxDecorationCommand {
    kind: NormalizedBoxDecorationCommandKind,
}

impl NormalizedBoxDecorationCommand {
    #[must_use]
    pub const fn kind(&self) -> &NormalizedBoxDecorationCommandKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBoxDecorationCommandKind {
    Border(NormalizedBorderCommand),
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
