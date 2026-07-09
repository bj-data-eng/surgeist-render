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
mod renderer;
mod scene;
mod shape;
mod stats;
mod style;
mod surface;
mod text;
mod validation;

pub use capability::{
    Capabilities, CompositingCapabilities, FilterCapabilities, GeometryTargetCapabilities,
    HitTestOwnership, MaskClipCapabilities, PaintSourceCapabilities, ShadowCapabilities,
    SurfaceCapabilities, SymbolicColorPolicy, TransformCoordinateSpaceCapabilities,
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
    BackgroundAttachment, BackgroundBox, BackgroundLayer, BackgroundPosition, BackgroundRepeat,
    BackgroundSize, BackgroundSizeKind, BackgroundStack, BorderEdges, BorderSide, BorderStyle,
    ClipInput, ClipInputKind, FilterAmount, FilterAngle, FilterBlur, FilterList, FilterOp,
    FilterOpKind, MaskInput, MaskMode, MaskSource, MaskSourceKind, Outline, OutlineStyle,
    PositionComponent, PositionComponentKind, RepeatMode, ResolvedImageResource, SizeComponent,
    SizeComponentKind, StyleColor, StyleImageLayer, StyleImageSource, StyleImageSourceKind,
    StyleResourceRef, UnitFilterAmount,
};
pub use surface::{
    Attachment, Format, Parameters, PresentMode, Surface, SurfaceOptions, SurfaceResourceState,
    SurfaceState, WebCanvas,
};
pub use text::{FontData, FontId, FontRef, TextGlyph, TextPaint, TextRun};

#[cfg(test)]
mod tests;
