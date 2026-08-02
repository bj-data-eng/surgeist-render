#![forbid(unsafe_code)]

//! GPU-only rendering boundary for Surgeist.
//!
//! The crate owns renderer-facing visual facts, safe WGPU resources, and two
//! observable execution routes. Effect-free scenes use one transaction-owned
//! [`RenderRoute::DirectVello`] raster pass. Resolved image alpha masks,
//! supported blend/composition, and bounded backdrop filter lists use
//! [`RenderRoute::GpuGraph`] image/composite passes. Production rendering has no
//! CPU pixel path or CPU fallback; CPU reference algorithms compile only for tests.
//!
//! Effect graphs use high precision by default. [`EffectQualityPolicy::RequireHighPrecision`]
//! rejects a device that cannot provide it, while
//! [`EffectQualityPolicy::AllowReducedPrecision`] permits reduced precision,
//! observably, only when high precision is unavailable. [`Capabilities::CURRENT`]
//! reports semantic capabilities for authored operations; [`Renderer::runtime_capabilities`]
//! reports immutable runtime capabilities for a selected device and surface.
//! Cargo features select host adapters rather than changing those meanings.
//!
//! Renderer and surface GPU operations are asynchronous and failure-atomic. A
//! failed or canceled operation publishes neither partial pixels nor partial
//! statistics, so [`Renderer::stats`] continues to describe the last successful
//! frame. CPU-visible pixels require explicit headless readback through
//! [`Renderer::read_headless`]; rendering never enters that path implicitly.
//!
//! The native window presentation path is host lifecycle owned.
//! The tracked `render_window_smoke` example exercises the public native window lifecycle
//! and presentation path. The
//! `wasm32-unknown-unknown` leaf boundary is compile-only under `render-web`;
//! real canvas execution requires a browser host and is root integration work.
//! The root `surgeist` repository owns the root facade, cross-crate adapters,
//! generated API artifacts, browser evidence, and this leaf's gitlink.

mod backend;
mod capability;
mod command;
mod encode;
mod error;
mod filter;
mod frame;
mod geometry;
mod gpu_transaction;
mod image;
mod layer;
mod paint;
mod pass;
mod readback;
#[cfg(test)]
mod reference;
mod renderer;
mod resource;
mod scene;
mod shader;
mod shape;
mod stats;
mod style;
mod surface;
mod text;
mod texture;
mod validation;
mod vello_engine;

pub(crate) use error::BackendErrorCode;

pub use capability::{
    AvailableRuntimeCapabilities, BackgroundAttachmentCoordinatePolicy, BoxDecorationCapabilities,
    Capabilities, CompositingCapabilities, EffectPrecisionCapabilities, FilterCapabilities,
    GeometryTargetCapabilities, HitTestOwnership, ImageColorProfilePolicy, ImageOrientationPolicy,
    ImageSamplingCapabilities, MaskClipCapabilities, OffscreenPipelineCapabilities,
    PaintSourceCapabilities, RuntimeCapabilities, ShadowCapabilities, SurfaceCapabilities,
    SymbolicColorPolicy, TransformCoordinateSpaceCapabilities,
};
pub use error::{
    DegradedQuality, DegradedQualityKind, DeviceLossReason, Error, ErrorCode, GpuFaultKind,
    InvalidValue, PrimitiveFamily, PrimitiveOperation, RenderSurfaceAvailability, Result,
    RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    SurfaceIdentityMismatchKind, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
};
pub use geometry::{
    CoordinateSpaceId, CoordinateSpaceKind, CoordinateSpaceTag, PhysicalSize, Point, Radii, Rect,
    Size, Transform,
};
pub use image::{Extend, Image, ImageBuffer, ImageFit, ImageId, ImageQuality};
pub use layer::{BlendMode, Filter, Layer, ResolvedLayerAlphaMask, Shadow, ShadowKind, ShadowList};
pub use paint::{
    Color, Gradient, GradientStop, NormalizedPaintLayer, Paint, PaintColor, PaintColorSpace,
};
pub use renderer::{Antialiasing, EffectQualityPolicy, Options, Renderer, ResourceCacheBudget};
pub use scene::Scene;
pub use shape::{
    Dash, FillRule, FilledPath, LineCap, LineJoin, Path, PathElement, Shape, Stroke, StrokeAlign,
};
pub use stats::{EffectPrecision, RenderRoute, Stats};
pub use style::{
    BackdropCaptureBounds, BackdropFilterInput, BackgroundAreas, BackgroundAttachment,
    BackgroundBlendList, BackgroundBlendMode, BackgroundBox, BackgroundClipGeometry,
    BackgroundClipGeometryKind, BackgroundLayer, BackgroundNormalizationInput, BackgroundPosition,
    BackgroundRepeat, BackgroundSize, BackgroundSizeKind, BackgroundStack, BorderEdges, BorderSide,
    BorderStyle, BoxDecorationBreak, BoxDecorationFragment, BoxDecorationInput, BoxSide,
    ClipGeometry, ClipGeometryKind, ClipInput, ClipInputKind, FilterAmount, FilterAngle,
    FilterBlur, FilterDropShadow, FilterList, FilterOp, FilterOpKind, FilteredImagePaint,
    ImageAttachmentPlan, ImagePlacementInput, ImageRepeatMode, ImageRepeatPlan,
    ImageResourceDensity, MaskCompositeMode, MaskInput, MaskLayer, MaskLayerStack, MaskMode,
    MaskSource, MaskSourceKind, NormalizedBackgroundCommand, NormalizedBackgroundCommandKind,
    NormalizedBackgroundLayer, NormalizedBackgroundLayerSource, NormalizedBackgroundStack,
    NormalizedBorderCommand, NormalizedBorderStyle, NormalizedBoxDecoration,
    NormalizedBoxDecorationCommand, NormalizedBoxDecorationCommandKind, NormalizedBoxRadii,
    NormalizedClip, NormalizedDoubleBorderBands, NormalizedOutlineCommand, NormalizedOutlineStyle,
    Outline, OutlineStyle, PositionComponent, PositionComponentKind, PositionEdge,
    PositionEdgeOffset, RepeatMode, ResolvedImagePlacement, ResolvedImageRepeat,
    ResolvedImageResource, SizeComponent, SizeComponentKind, StyleColor, StyleImageLayer,
    StyleImageSource, StyleImageSourceKind, StyleResourceRef, UnitFilterAmount,
};
pub use surface::{
    Attachment, Format, Parameters, PresentMode, Surface, SurfaceOptions, SurfaceResourceState,
    SurfaceState, WebCanvas,
};
pub use text::{
    FontData, FontId, FontRef, TextDecorationLine, TextDecorationLineStyle, TextGlyph, TextPaint,
    TextRun, TextRunBounds, TextRunBoundsKind, TextShadowRun,
};

#[cfg(test)]
mod tests;
