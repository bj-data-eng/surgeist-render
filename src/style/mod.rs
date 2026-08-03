use super::{Color, SymbolicColorPolicy};

mod background;
mod clip;
mod decoration;
mod filter;
mod image;
mod mask;

pub use self::background::{
    BackgroundBlendList, BackgroundBlendMode, BackgroundLayer, BackgroundNormalizationInput,
    BackgroundStack, NormalizedBackgroundCommand, NormalizedBackgroundCommandKind,
    NormalizedBackgroundLayer, NormalizedBackgroundLayerSource, NormalizedBackgroundStack,
};
pub(crate) use self::clip::validate_clip_input;
pub use self::clip::{ClipGeometry, ClipGeometryKind, ClipInput, ClipInputKind, NormalizedClip};
pub use self::decoration::{
    BorderEdges, BorderSide, BorderStyle, BoxDecorationBreak, BoxDecorationFragment,
    BoxDecorationInput, BoxSide, NormalizedBorderCommand, NormalizedBorderStyle,
    NormalizedBoxDecoration, NormalizedBoxDecorationCommand, NormalizedBoxDecorationCommandKind,
    NormalizedBoxRadii, NormalizedDoubleBorderBands, NormalizedOutlineCommand,
    NormalizedOutlineStyle, Outline, OutlineStyle,
};
#[cfg(test)]
pub(crate) use self::filter::ColorFilterPipeline;
#[cfg(test)]
pub(crate) use self::filter::filter_drop_shadow_payload_accepts_shadow_for_test;
pub use self::filter::{
    BackdropCaptureBounds, BackdropFilterInput, ColorFilterOp, FilterAmount, FilterAngle,
    FilterBlur, FilterDropShadow, FilterList, FilterOp, FilterOpKind, FilteredImagePaint,
    UnitFilterAmount,
};
pub use self::image::{
    BackgroundAreas, BackgroundAttachment, BackgroundBox, BackgroundClipGeometry,
    BackgroundClipGeometryKind, BackgroundPosition, BackgroundRepeat, BackgroundSize,
    BackgroundSizeKind, ImageAttachmentPlan, ImagePlacementInput, ImageRepeatMode, ImageRepeatPlan,
    ImageResourceDensity, PositionComponent, PositionComponentKind, PositionEdge,
    PositionEdgeOffset, RepeatMode, ResolvedImagePlacement, ResolvedImageRepeat,
    ResolvedImageResource, SizeComponent, SizeComponentKind, StyleImageLayer, StyleImageSource,
    StyleImageSourceKind, StyleResourceRef,
};
pub use self::mask::{
    MaskCompositeMode, MaskInput, MaskLayer, MaskLayerStack, MaskMode, MaskSource, MaskSourceKind,
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

    #[must_use]
    pub const fn symbolic_policy() -> SymbolicColorPolicy {
        SymbolicColorPolicy::RootResolvedOnly
    }
}
