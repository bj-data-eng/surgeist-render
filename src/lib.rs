//! Vello-backed rendering boundary for Surgeist.
//!
//! This module owns renderer-facing visual facts: surfaces, scenes, drawing
//! commands, paint, layers, strokes, images, shadows, text runs, and diagnostics.
//! The implementation keeps scene encoding deterministic and headless-testable
//! while reserving the backend boundary for Vello/wgpu submission.

mod backend;
mod capability;
mod command;
mod encode;
mod error;
mod geometry;
mod image;
mod layer;
mod paint;
mod reference;
mod renderer;
mod scene;
mod shader;
mod shape;
mod stats;
mod style;
mod surface;
mod text;
mod texture;
mod validation;

pub use capability::{
    BackgroundAttachmentCoordinatePolicy, BoxDecorationCapabilities, Capabilities,
    CompositingCapabilities, FilterCapabilities, GeometryTargetCapabilities, HitTestOwnership,
    ImageColorProfilePolicy, ImageOrientationPolicy, ImageSamplingCapabilities,
    MaskClipCapabilities, OffscreenPipelineCapabilities, PaintSourceCapabilities,
    ShadowCapabilities, SurfaceCapabilities, SymbolicColorPolicy,
    TransformCoordinateSpaceCapabilities,
};
pub use error::{
    DegradedQuality, DegradedQualityKind, Error, ErrorCode, InvalidValue, PrimitiveFamily,
    PrimitiveOperation, Result, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
};
pub use geometry::{
    CoordinateSpaceId, CoordinateSpaceKind, CoordinateSpaceTag, PhysicalSize, Point, Radii, Rect,
    Size, Transform,
};
pub use image::{Extend, Image, ImageBuffer, ImageFit, ImageId, ImageQuality};
pub use layer::{BlendMode, Filter, Layer, Shadow};
pub use paint::{
    Color, Gradient, GradientStop, NormalizedPaintLayer, Paint, PaintColor, PaintColorSpace,
};
pub use renderer::{Antialiasing, Options, Renderer};
pub use scene::Scene;
pub use shape::{
    Dash, FillRule, FilledPath, LineCap, LineJoin, Path, PathElement, Shape, Stroke, StrokeAlign,
};
pub use stats::Stats;
pub use style::{
    BackgroundAreas, BackgroundAttachment, BackgroundBox, BackgroundClipGeometry,
    BackgroundClipGeometryKind, BackgroundLayer, BackgroundNormalizationInput, BackgroundPosition,
    BackgroundRepeat, BackgroundSize, BackgroundSizeKind, BackgroundStack, BorderEdges, BorderSide,
    BorderStyle, BoxDecorationBreak, BoxDecorationFragment, BoxDecorationInput, BoxSide, ClipInput,
    ClipInputKind, ColorFilterOp, ColorFilterPipeline, FilterAmount, FilterAngle, FilterBlur,
    FilterList, FilterOp, FilterOpKind, FilteredImagePaint, ImageAttachmentPlan,
    ImagePlacementInput, ImageRepeatMode, ImageRepeatPlan, ImageResourceDensity, MaskInput,
    MaskMode, MaskSource, MaskSourceKind, NormalizedBackgroundCommand,
    NormalizedBackgroundCommandKind, NormalizedBackgroundLayer, NormalizedBackgroundLayerSource,
    NormalizedBackgroundStack, NormalizedBorderCommand, NormalizedBorderStyle,
    NormalizedBoxDecoration, NormalizedBoxDecorationCommand, NormalizedBoxDecorationCommandKind,
    NormalizedBoxRadii, NormalizedDoubleBorderBands, NormalizedOutlineCommand,
    NormalizedOutlineStyle, Outline, OutlineStyle, PositionComponent, PositionComponentKind,
    PositionEdge, PositionEdgeOffset, RepeatMode, ResolvedImagePlacement, ResolvedImageRepeat,
    ResolvedImageResource, SizeComponent, SizeComponentKind, StyleColor, StyleImageLayer,
    StyleImageSource, StyleImageSourceKind, StyleResourceRef, UnitFilterAmount,
};
pub use surface::{
    Attachment, Format, Parameters, PresentMode, Surface, SurfaceOptions, SurfaceResourceState,
    SurfaceState, WebCanvas,
};
pub use text::{FontData, FontId, FontRef, TextGlyph, TextPaint, TextRun};

#[cfg(test)]
mod tests;
