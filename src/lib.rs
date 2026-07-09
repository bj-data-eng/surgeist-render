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
mod surface;
mod text;
mod validation;

pub use capability::{
    Capabilities, CompositingCapabilities, FilterCapabilities, GeometryTargetCapabilities,
    MaskClipCapabilities, PaintSourceCapabilities, SurfaceCapabilities,
};
pub use error::{
    DegradedQuality, DegradedQualityKind, Error, ErrorCode, InvalidValue, PrimitiveFamily,
    PrimitiveOperation, Result, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
};
pub use geometry::{PhysicalSize, Point, Radii, Rect, Size, Transform};
pub use image::{Extend, Image, ImageBuffer, ImageFit, ImageId, ImageQuality};
pub use layer::{BlendMode, Filter, Layer, Shadow};
pub use paint::{Color, Gradient, GradientStop, Paint};
pub use renderer::{Antialiasing, Options, Renderer};
pub use scene::Scene;
pub use shape::{Dash, LineCap, LineJoin, Path, PathElement, Shape, Stroke, StrokeAlign};
pub use stats::Stats;
pub use surface::{
    Attachment, Format, Parameters, PresentMode, Surface, SurfaceOptions, SurfaceResourceState,
    SurfaceState, WebCanvas,
};
pub use text::{FontData, FontId, FontRef, TextGlyph, TextPaint, TextRun};

#[cfg(test)]
mod tests;
