#![forbid(unsafe_code)]

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
mod filter;
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

pub(crate) use error::BackendErrorCode;

pub use capability::{
    BackgroundAttachmentCoordinatePolicy, BoxDecorationCapabilities, Capabilities,
    CompositingCapabilities, FilterCapabilities, GeometryTargetCapabilities, HitTestOwnership,
    ImageColorProfilePolicy, ImageOrientationPolicy, ImageSamplingCapabilities,
    MaskClipCapabilities, OffscreenPipelineCapabilities, PaintSourceCapabilities,
    ShadowCapabilities, SurfaceCapabilities, SymbolicColorPolicy,
    TransformCoordinateSpaceCapabilities,
};
pub use error::{
    DegradedQuality, DegradedQualityKind, DeviceLossReason, Error, ErrorCode, GpuFaultKind,
    InvalidValue, PrimitiveFamily, PrimitiveOperation, RenderSurfaceAvailability, Result,
    RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    SurfaceIdentityMismatchKind, UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive,
};
pub use filter::{
    BlurPolicy, BlurRadiusInterpretation, CompiledColorFilterPipeline, DevicePixelConversionPolicy,
    FilterClipBounds, FilterDeviceBounds, FilterExecutionRegion, FilterInflatedBounds,
    FilterOutset, FilterRegionPlan, FilterSourceBounds, KernelSupportRadius, LargeBlurRadiusAction,
    LargeBlurRadiusPolicy, MaterializedImageFilterPipeline, MaterializedImageFilterStep,
    TransparentEdgeSamplingPolicy,
};
pub use geometry::{
    CoordinateSpaceId, CoordinateSpaceKind, CoordinateSpaceTag, PhysicalSize, Point, Radii, Rect,
    Size, Transform,
};
pub use image::{
    Extend, Image, ImageBuffer, ImageFit, ImageId, ImageQuality, ResolvedAlphaMaskExecution,
};
pub use layer::{BlendMode, Filter, Layer, ResolvedLayerAlphaMask, Shadow, ShadowKind, ShadowList};
pub use paint::{
    Color, Gradient, GradientStop, NormalizedPaintLayer, Paint, PaintColor, PaintColorSpace,
};
pub use renderer::{Antialiasing, EffectQualityPolicy, Options, Renderer, ResourceCacheBudget};
pub use scene::Scene;
pub use shape::{
    Dash, FillRule, FilledPath, LineCap, LineJoin, Path, PathElement, Shape, Stroke, StrokeAlign,
};
pub use stats::Stats;
pub use style::{
    BackdropCaptureBounds, BackdropFilterInput, BackgroundAreas, BackgroundAttachment,
    BackgroundBlendList, BackgroundBlendMode, BackgroundBox, BackgroundClipGeometry,
    BackgroundClipGeometryKind, BackgroundLayer, BackgroundNormalizationInput, BackgroundPosition,
    BackgroundRepeat, BackgroundSize, BackgroundSizeKind, BackgroundStack, BorderEdges, BorderSide,
    BorderStyle, BoxDecorationBreak, BoxDecorationFragment, BoxDecorationInput, BoxSide,
    ClipGeometry, ClipGeometryKind, ClipInput, ClipInputKind, ColorFilterOp, ColorFilterPipeline,
    FilterAmount, FilterAngle, FilterBlur, FilterList, FilterOp, FilterOpKind, FilteredImagePaint,
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
