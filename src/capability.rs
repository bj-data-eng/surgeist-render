use super::{
    Error, Format, PrimitiveFamily, PrimitiveOperation, Result, RuntimeCapabilityUnavailableReason,
    UnsupportedPrimitive,
};

/// Runtime-phase facts reported for a selected device and surface through safe WGPU.
///
/// These facts describe the selected runtime device/surface. They are not
/// semantic rendering support or enabled Cargo features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeCapabilities {
    /// The selected device or surface cannot provide a runtime capability report.
    ///
    /// This is runtime device/surface evidence, not a statement about semantic
    /// rendering support or Cargo features.
    Unavailable(RuntimeCapabilityUnavailableReason),
    /// Runtime facts for an available selected device and surface.
    ///
    /// These are device/surface facts rather than semantic rendering support or
    /// Cargo features.
    Available(AvailableRuntimeCapabilities),
}

impl RuntimeCapabilities {
    /// Returns the selected device/surface facts when the runtime report is available.
    ///
    /// The returned value describes runtime device/surface facts, not semantic
    /// rendering support or Cargo features.
    #[must_use]
    pub const fn available(self) -> Option<AvailableRuntimeCapabilities> {
        match self {
            Self::Unavailable(_) => None,
            Self::Available(capabilities) => Some(capabilities),
        }
    }

    /// Returns why the selected device or surface could not provide a runtime report.
    ///
    /// The reason concerns runtime device/surface availability, not semantic
    /// rendering support or Cargo features.
    #[must_use]
    pub const fn unavailable_reason(self) -> Option<RuntimeCapabilityUnavailableReason> {
        match self {
            Self::Unavailable(reason) => Some(reason),
            Self::Available(_) => None,
        }
    }
}

/// Available runtime-phase facts for a selected safe WGPU device and surface.
///
/// The fields are runtime device/surface facts rather than semantic rendering
/// support or enabled Cargo features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AvailableRuntimeCapabilities {
    surface_format: Format,
    effect_precisions: EffectPrecisionCapabilities,
    max_effect_texture_dimension_2d: u32,
}

impl AvailableRuntimeCapabilities {
    pub(crate) const fn new(
        surface_format: Format,
        effect_precisions: EffectPrecisionCapabilities,
        max_effect_texture_dimension_2d: u32,
    ) -> Self {
        Self {
            surface_format,
            effect_precisions,
            max_effect_texture_dimension_2d,
        }
    }

    /// Returns the format of the selected runtime surface.
    ///
    /// This is a selected device/surface fact, not semantic rendering support or
    /// a Cargo feature.
    #[must_use]
    pub const fn surface_format(self) -> Format {
        self.surface_format
    }

    /// Returns the effect texture precisions supported by the selected runtime device.
    ///
    /// These are device facts, not semantic rendering support or Cargo features.
    #[must_use]
    pub const fn effect_precisions(self) -> EffectPrecisionCapabilities {
        self.effect_precisions
    }

    /// Returns the selected device's maximum two-dimensional effect texture dimension.
    ///
    /// This runtime device limit is not semantic rendering support or a Cargo feature.
    #[must_use]
    pub const fn max_effect_texture_dimension_2d(self) -> u32 {
        self.max_effect_texture_dimension_2d
    }
}

/// Runtime effect texture precision facts for a selected safe WGPU device.
///
/// Each flag is independent and describes runtime device support rather than
/// semantic rendering support or enabled Cargo features. Selection between high
/// and reduced precision is controlled separately by [`crate::EffectQualityPolicy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectPrecisionCapabilities {
    high_precision: bool,
    reduced_precision: bool,
}

impl EffectPrecisionCapabilities {
    pub(crate) const fn new(high_precision: bool, reduced_precision: bool) -> Self {
        Self {
            high_precision,
            reduced_precision,
        }
    }

    /// Returns whether the selected runtime device supports high-precision effect textures.
    ///
    /// This is a runtime device fact, not semantic rendering support or a Cargo feature.
    #[must_use]
    pub const fn supports_high_precision(self) -> bool {
        self.high_precision
    }

    /// Returns whether the selected runtime device supports reduced-precision effect textures.
    ///
    /// This is a runtime device fact, not semantic rendering support or a Cargo feature.
    #[must_use]
    pub const fn supports_reduced_precision(self) -> bool {
        self.reduced_precision
    }
}

/// Semantic support for normalized authored rendering operations.
///
/// This report is fixed by the crate's public contract. It does not describe a
/// selected runtime device, surface, or enabled Cargo features; use
/// [`crate::Renderer::runtime_capabilities`] for those runtime facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    geometry_targets: GeometryTargetCapabilities,
    paint_sources: PaintSourceCapabilities,
    image_sampling: ImageSamplingCapabilities,
    shadows: ShadowCapabilities,
    filters: FilterCapabilities,
    masks_clips: MaskClipCapabilities,
    box_decorations: BoxDecorationCapabilities,
    compositing: CompositingCapabilities,
    offscreen_pipeline: OffscreenPipelineCapabilities,
    surfaces: SurfaceCapabilities,
    transform_coordinate_spaces: TransformCoordinateSpaceCapabilities,
}

impl Capabilities {
    /// The current semantic rendering contract.
    ///
    /// The value describes which authored operations this crate accepts and how
    /// it owns them. It is not a backend-name, runtime-device, or Cargo-feature
    /// probe. Unsupported authored operations fail through
    /// [`Self::ensure_supported`].
    pub const CURRENT: Self = Self {
        geometry_targets: GeometryTargetCapabilities {
            rect_fill_stroke: true,
            rounded_rect_fill_stroke: true,
            circle_ellipse_fill_stroke: true,
            arbitrary_path_fill: true,
            arbitrary_path_centered_stroke: true,
            arbitrary_path_inside_outside_stroke: false,
            geometry_booleans: false,
            geometry_offsets: false,
            hit_testing: HitTestOwnership::RootOwned,
        },
        paint_sources: PaintSourceCapabilities {
            solid_rgba: true,
            gradients: true,
            image_paint: true,
            non_solid_shadow_paint: false,
            srgb_color_conversion: true,
            hsl_color_conversion: true,
            unresolved_symbolic_colors: false,
            color_mix: false,
            repeating_gradients: false,
            symbolic_color_policy: SymbolicColorPolicy::RootResolvedOnly,
        },
        image_sampling: ImageSamplingCapabilities {
            image_fit: true,
            background_position: true,
            background_size: true,
            repeat_xy: true,
            repeat_round: false,
            repeat_space: false,
            filtered_image_paint: false,
            color_filtered_image_paint: false,
            image_orientation_conversion: false,
            image_color_profile_conversion: false,
            attachment_coordinate_policy:
                BackgroundAttachmentCoordinatePolicy::RootResolvedOrTagged,
            image_orientation_policy: ImageOrientationPolicy::RootResolvedOnly,
            image_color_profile_policy: ImageColorProfilePolicy::RootResolvedOnly,
        },
        shadows: ShadowCapabilities {
            rect_rounded_circle_shadows: true,
            ellipse_path_shadows: false,
            inset_box_shadows: false,
            text_shadows: false,
        },
        filters: FilterCapabilities {
            layer_filters: false,
            ordered_filter_lists: true,
            gpu_color_filter_execution: true,
            gpu_blur_filter_execution: true,
            gpu_drop_shadow_filter_execution: true,
            filter_region_planning: true,
        },
        masks_clips: MaskClipCapabilities {
            shape_clips: true,
            clip_reference_execution: false,
            layer_masks: false,
            resolved_alpha_mask_execution: true,
            luminance_mask_mode: false,
            multi_layer_mask_composition: false,
            mask_composite_modes: false,
        },
        box_decorations: BoxDecorationCapabilities {
            border_none_hidden_styles: true,
            border_solid_style: true,
            border_dashed_dotted_styles: true,
            border_double_style: true,
            border_groove_style: false,
            border_ridge_style: false,
            border_inset_style: false,
            border_outset_style: false,
            border_radii: true,
            outlines: true,
            outline_none_style: true,
            outline_solid_style: true,
            outline_dashed_dotted_styles: true,
            outline_double_style: false,
            outline_auto_style: false,
            fragments: true,
        },
        compositing: CompositingCapabilities {
            layer_opacity: true,
            blend_modes: true,
            root_backdrop_policy: false,
            background_blend_modes: false,
            additional_mix_blend_modes: false,
            porter_duff_composite_modes: false,
        },
        offscreen_pipeline: OffscreenPipelineCapabilities {
            direct_vello_opacity_isolation: true,
            direct_vello_blend_isolation: true,
            offscreen_layer_rendering: false,
            persistent_effect_resources: true,
            bounded_vello_capture: true,
            image_pass_execution: true,
            composite_pass_execution: true,
            nested_opacity_composition: true,
            mask_execution: false,
            layer_filter_execution: false,
            broad_backdrop_execution: false,
            bounded_backdrop_capture: true,
            bounded_backdrop_filter_execution: true,
            backdrop_isolation_composition: false,
        },
        surfaces: SurfaceCapabilities {
            headless_surfaces: true,
            web_canvas_surfaces: cfg!(all(feature = "render-web", target_arch = "wasm32")),
        },
        transform_coordinate_spaces: TransformCoordinateSpaceCapabilities {
            affine_2d: true,
            transform_origin: true,
            skew: true,
            transform_3d: false,
            coordinate_space_tags: true,
        },
    };

    /// Returns semantic support for geometry targets and geometry operations.
    #[must_use]
    pub const fn geometry_targets(self) -> GeometryTargetCapabilities {
        self.geometry_targets
    }

    /// Returns semantic support for paint sources and color handling.
    #[must_use]
    pub const fn paint_sources(self) -> PaintSourceCapabilities {
        self.paint_sources
    }

    /// Returns semantic support and resolution policy for image sampling.
    #[must_use]
    pub const fn image_sampling(self) -> ImageSamplingCapabilities {
        self.image_sampling
    }

    /// Returns semantic support for shadow shapes and kinds.
    #[must_use]
    pub const fn shadows(self) -> ShadowCapabilities {
        self.shadows
    }

    /// Returns semantic support for filter lists, planning, and GPU execution.
    #[must_use]
    pub const fn filters(self) -> FilterCapabilities {
        self.filters
    }

    /// Returns semantic support for clips, masks, and resolved mask execution.
    #[must_use]
    pub const fn masks_clips(self) -> MaskClipCapabilities {
        self.masks_clips
    }

    /// Returns semantic support for borders, outlines, radii, and fragments.
    #[must_use]
    pub const fn box_decorations(self) -> BoxDecorationCapabilities {
        self.box_decorations
    }

    /// Returns semantic support for opacity, blending, and compositing modes.
    #[must_use]
    pub const fn compositing(self) -> CompositingCapabilities {
        self.compositing
    }

    /// Returns semantic support for direct and GPU-graph offscreen phases.
    #[must_use]
    pub const fn offscreen_pipeline(self) -> OffscreenPipelineCapabilities {
        self.offscreen_pipeline
    }

    /// Returns semantic surface-construction support for this compiled target.
    ///
    /// This does not report whether a selected runtime device or surface is
    /// currently available.
    #[must_use]
    pub const fn surfaces(self) -> SurfaceCapabilities {
        self.surfaces
    }

    /// Returns semantic support for transforms and coordinate-space tagging.
    #[must_use]
    pub const fn transform_coordinate_spaces(self) -> TransformCoordinateSpaceCapabilities {
        self.transform_coordinate_spaces
    }

    /// Accepts a supported authored operation or returns its typed unsupported diagnostic.
    ///
    /// An unavailable operation produces [`crate::ErrorCode::UnsupportedPrimitive`]
    /// with `primitive` available from [`Error::unsupported_primitive`]. This
    /// semantic check does not query enabled Cargo features or a runtime device.
    pub fn ensure_supported(self, primitive: UnsupportedPrimitive) -> Result<()> {
        if self.supports(primitive) {
            Ok(())
        } else {
            Err(Error::unsupported_render_primitive(primitive))
        }
    }

    const fn supports(self, primitive: UnsupportedPrimitive) -> bool {
        match (primitive.family(), primitive.operation()) {
            (PrimitiveFamily::GeometryTargets, operation) => {
                self.supports_geometry_operation(operation)
            }
            (PrimitiveFamily::PaintSources, operation) => self.supports_paint_operation(operation),
            (PrimitiveFamily::ImageSampling, operation) => self.supports_image_operation(operation),
            (PrimitiveFamily::Shadows, operation) => self.supports_shadow_operation(operation),
            (PrimitiveFamily::Filters, operation) => self.supports_filter_operation(operation),
            (PrimitiveFamily::MasksAndClips, operation) => {
                self.supports_mask_clip_operation(operation)
            }
            (PrimitiveFamily::BoxDecorations, operation) => {
                self.supports_box_decoration_operation(operation)
            }
            (PrimitiveFamily::Compositing, operation) => {
                self.supports_compositing_operation(operation)
            }
            (PrimitiveFamily::OffscreenPipeline, operation) => {
                self.supports_offscreen_operation(operation)
            }
            (PrimitiveFamily::Surfaces, PrimitiveOperation::WebCanvasSurface) => {
                self.surfaces.supports_web_canvas_surfaces()
            }
            (PrimitiveFamily::TransformsAndCoordinateSpaces, operation) => {
                self.supports_transform_operation(operation)
            }
            _ => false,
        }
    }

    const fn supports_geometry_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::InsideOutsidePathStrokeAlignment => self
                .geometry_targets
                .supports_arbitrary_path_inside_outside_stroke(),
            PrimitiveOperation::GeometryBooleanOperation => {
                self.geometry_targets.supports_geometry_booleans()
            }
            PrimitiveOperation::GeometryOffsetOperation => {
                self.geometry_targets.supports_geometry_offsets()
            }
            _ => false,
        }
    }

    const fn supports_paint_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::NonSolidShadowPaint => {
                self.paint_sources.supports_non_solid_shadow_paint()
            }
            PrimitiveOperation::UnresolvedSymbolicColor => {
                self.paint_sources.supports_unresolved_symbolic_colors()
            }
            PrimitiveOperation::ColorMixFunction => self.paint_sources.supports_color_mix(),
            PrimitiveOperation::RepeatingGradient => {
                self.paint_sources.supports_repeating_gradients()
            }
            PrimitiveOperation::UnsupportedColorSpace => false,
            _ => false,
        }
    }

    const fn supports_image_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::BackgroundRepeatRound => {
                self.image_sampling.supports_repeat_round()
            }
            PrimitiveOperation::BackgroundRepeatSpace => {
                self.image_sampling.supports_repeat_space()
            }
            PrimitiveOperation::FilteredImagePaint => {
                self.image_sampling.supports_filtered_image_paint()
            }
            PrimitiveOperation::ColorFilteredImagePaint => {
                self.image_sampling.supports_color_filtered_image_paint()
            }
            PrimitiveOperation::ImageOrientationConversion => {
                self.image_sampling.supports_image_orientation_conversion()
            }
            PrimitiveOperation::ImageColorProfileConversion => self
                .image_sampling
                .supports_image_color_profile_conversion(),
            _ => false,
        }
    }

    const fn supports_shadow_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::EllipsePathShadowShape => {
                self.shadows.supports_ellipse_path_shadows()
            }
            PrimitiveOperation::InsetBoxShadow => self.shadows.supports_inset_box_shadows(),
            PrimitiveOperation::TextShadow => self.shadows.supports_text_shadows(),
            _ => false,
        }
    }

    const fn supports_filter_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::LayerFilter => self.filters.supports_layer_filters(),
            PrimitiveOperation::OrderedFilterList => self.filters.supports_ordered_filter_lists(),
            PrimitiveOperation::GpuColorFilterExecution => {
                self.filters.supports_gpu_color_filter_execution()
            }
            PrimitiveOperation::GpuBlurFilterExecution => {
                self.filters.supports_gpu_blur_filter_execution()
            }
            PrimitiveOperation::GpuDropShadowFilterExecution => {
                self.filters.supports_gpu_drop_shadow_filter_execution()
            }
            PrimitiveOperation::FilterRegionPlanning => {
                self.filters.supports_filter_region_planning()
            }
            _ => false,
        }
    }

    const fn supports_mask_clip_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::ShapeClip => self.masks_clips.supports_shape_clips(),
            PrimitiveOperation::ClipReferenceExecution => {
                self.masks_clips.supports_clip_reference_execution()
            }
            PrimitiveOperation::LayerMask => self.masks_clips.supports_layer_masks(),
            PrimitiveOperation::ResolvedAlphaMaskExecution => {
                self.masks_clips.supports_resolved_alpha_mask_execution()
            }
            PrimitiveOperation::LuminanceMaskMode => {
                self.masks_clips.supports_luminance_mask_mode()
            }
            PrimitiveOperation::MultiLayerMaskComposition => {
                self.masks_clips.supports_multi_layer_mask_composition()
            }
            PrimitiveOperation::MaskCompositeMode => {
                self.masks_clips.supports_mask_composite_modes()
            }
            PrimitiveOperation::AlphaMaskSourceExecution => false,
            _ => false,
        }
    }

    const fn supports_box_decoration_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::BorderGrooveStyle => {
                self.box_decorations.supports_border_groove_style()
            }
            PrimitiveOperation::BorderRidgeStyle => {
                self.box_decorations.supports_border_ridge_style()
            }
            PrimitiveOperation::BorderInsetStyle => {
                self.box_decorations.supports_border_inset_style()
            }
            PrimitiveOperation::BorderOutsetStyle => {
                self.box_decorations.supports_border_outset_style()
            }
            PrimitiveOperation::OutlineDoubleStyle => {
                self.box_decorations.supports_outline_double_style()
            }
            PrimitiveOperation::OutlineAutoStyle => {
                self.box_decorations.supports_outline_auto_style()
            }
            _ => false,
        }
    }

    const fn supports_compositing_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::RootBackdropPolicy => {
                self.compositing.supports_root_backdrop_policy()
            }
            PrimitiveOperation::BackgroundBlendMode => {
                self.compositing.supports_background_blend_modes()
            }
            PrimitiveOperation::AdditionalMixBlendMode => {
                self.compositing.supports_additional_mix_blend_modes()
            }
            PrimitiveOperation::PorterDuffCompositeMode => {
                self.compositing.supports_porter_duff_composite_modes()
            }
            _ => false,
        }
    }

    const fn supports_offscreen_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::OffscreenLayerRendering => {
                self.offscreen_pipeline.supports_offscreen_layer_rendering()
            }
            PrimitiveOperation::PersistentEffectResources => self
                .offscreen_pipeline
                .supports_persistent_effect_resources(),
            PrimitiveOperation::BoundedVelloCapture => {
                self.offscreen_pipeline.supports_bounded_vello_capture()
            }
            PrimitiveOperation::ImagePassExecution => {
                self.offscreen_pipeline.supports_image_pass_execution()
            }
            PrimitiveOperation::CompositePassExecution => {
                self.offscreen_pipeline.supports_composite_pass_execution()
            }
            PrimitiveOperation::NestedOpacityComposition => self
                .offscreen_pipeline
                .supports_nested_opacity_composition(),
            PrimitiveOperation::MaskExecution => self.offscreen_pipeline.supports_mask_execution(),
            PrimitiveOperation::LayerFilterExecution => {
                self.offscreen_pipeline.supports_layer_filter_execution()
            }
            PrimitiveOperation::BroadBackdropExecution => {
                self.offscreen_pipeline.supports_broad_backdrop_execution()
            }
            PrimitiveOperation::BoundedBackdropCapture => {
                self.offscreen_pipeline.supports_bounded_backdrop_capture()
            }
            PrimitiveOperation::BoundedBackdropFilterExecution => self
                .offscreen_pipeline
                .supports_bounded_backdrop_filter_execution(),
            PrimitiveOperation::BackdropIsolationComposition => self
                .offscreen_pipeline
                .supports_backdrop_isolation_composition(),
            _ => false,
        }
    }

    const fn supports_transform_operation(self, operation: PrimitiveOperation) -> bool {
        match operation {
            PrimitiveOperation::Matrix3dTransform
            | PrimitiveOperation::PerspectiveTransform
            | PrimitiveOperation::Rotate3dTransform
            | PrimitiveOperation::TranslateZTransform
            | PrimitiveOperation::ScaleZTransform => {
                self.transform_coordinate_spaces.supports_transform_3d()
            }
            _ => false,
        }
    }
}

/// Identifies the layer responsible for hit testing authored geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitTestOwnership {
    /// Hit testing is resolved outside this rendering crate before rendering.
    RootOwned,
}

/// Policy for symbolic colors received by the rendering boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicColorPolicy {
    /// Symbolic colors must be resolved before they reach this crate.
    RootResolvedOnly,
}

/// Policy for background-attachment coordinate spaces at this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachmentCoordinatePolicy {
    /// Attachment coordinates must be resolved or explicitly tagged before rendering.
    RootResolvedOrTagged,
}

/// Policy for image orientation at this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageOrientationPolicy {
    /// Image orientation must be resolved before it reaches this crate.
    RootResolvedOnly,
}

/// Policy for image color profiles at this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageColorProfilePolicy {
    /// Image color profiles must be resolved before they reach this crate.
    RootResolvedOnly,
}

/// Semantic support for authored geometry targets and geometry operations.
///
/// A `true` accessor result means the operation is supported by the current
/// rendering contract; `false` means it is unavailable and is rejected when
/// represented by [`UnsupportedPrimitive`]. These are not runtime device facts
/// or Cargo-feature flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryTargetCapabilities {
    rect_fill_stroke: bool,
    rounded_rect_fill_stroke: bool,
    circle_ellipse_fill_stroke: bool,
    arbitrary_path_fill: bool,
    arbitrary_path_centered_stroke: bool,
    arbitrary_path_inside_outside_stroke: bool,
    geometry_booleans: bool,
    geometry_offsets: bool,
    hit_testing: HitTestOwnership,
}

impl GeometryTargetCapabilities {
    /// Returns whether rectangles support fill and stroke rendering.
    #[must_use]
    pub const fn supports_rect_fill_stroke(self) -> bool {
        self.rect_fill_stroke
    }

    /// Returns whether rounded rectangles support fill and stroke rendering.
    #[must_use]
    pub const fn supports_rounded_rect_fill_stroke(self) -> bool {
        self.rounded_rect_fill_stroke
    }

    /// Returns whether circles and ellipses support fill and stroke rendering.
    #[must_use]
    pub const fn supports_circle_ellipse_fill_stroke(self) -> bool {
        self.circle_ellipse_fill_stroke
    }

    /// Returns whether arbitrary paths support fill rendering.
    #[must_use]
    pub const fn supports_arbitrary_path_fill(self) -> bool {
        self.arbitrary_path_fill
    }

    /// Returns whether arbitrary paths support centered strokes.
    #[must_use]
    pub const fn supports_arbitrary_path_centered_stroke(self) -> bool {
        self.arbitrary_path_centered_stroke
    }

    /// Returns whether arbitrary paths support inside or outside stroke alignment.
    #[must_use]
    pub const fn supports_arbitrary_path_inside_outside_stroke(self) -> bool {
        self.arbitrary_path_inside_outside_stroke
    }

    /// Returns whether geometry boolean operations are supported.
    #[must_use]
    pub const fn supports_geometry_booleans(self) -> bool {
        self.geometry_booleans
    }

    /// Returns whether geometry offset operations are supported.
    #[must_use]
    pub const fn supports_geometry_offsets(self) -> bool {
        self.geometry_offsets
    }

    /// Returns the layer responsible for geometry hit testing.
    #[must_use]
    pub const fn hit_testing(self) -> HitTestOwnership {
        self.hit_testing
    }
}

/// Semantic support for authored paint sources and color handling.
///
/// Boolean accessors report current rendering support, not runtime GPU facts or
/// Cargo-feature selection. Policy accessors state what must be resolved before
/// paint reaches this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaintSourceCapabilities {
    solid_rgba: bool,
    gradients: bool,
    image_paint: bool,
    non_solid_shadow_paint: bool,
    srgb_color_conversion: bool,
    hsl_color_conversion: bool,
    unresolved_symbolic_colors: bool,
    color_mix: bool,
    repeating_gradients: bool,
    symbolic_color_policy: SymbolicColorPolicy,
}

impl PaintSourceCapabilities {
    /// Returns whether solid RGBA paint is supported.
    #[must_use]
    pub const fn supports_solid_rgba(self) -> bool {
        self.solid_rgba
    }

    /// Returns whether gradient paint is supported.
    #[must_use]
    pub const fn supports_gradients(self) -> bool {
        self.gradients
    }

    /// Returns whether image paint is supported.
    #[must_use]
    pub const fn supports_image_paint(self) -> bool {
        self.image_paint
    }

    /// Returns whether shadows accept non-solid paint.
    #[must_use]
    pub const fn supports_non_solid_shadow_paint(self) -> bool {
        self.non_solid_shadow_paint
    }

    /// Returns whether sRGB color conversion is supported.
    #[must_use]
    pub const fn supports_srgb_color_conversion(self) -> bool {
        self.srgb_color_conversion
    }

    /// Returns whether HSL color conversion is supported.
    #[must_use]
    pub const fn supports_hsl_color_conversion(self) -> bool {
        self.hsl_color_conversion
    }

    /// Returns whether unresolved symbolic colors are accepted.
    #[must_use]
    pub const fn supports_unresolved_symbolic_colors(self) -> bool {
        self.unresolved_symbolic_colors
    }

    /// Returns whether authored color-mix functions are supported.
    #[must_use]
    pub const fn supports_color_mix(self) -> bool {
        self.color_mix
    }

    /// Returns whether repeating gradients are supported.
    #[must_use]
    pub const fn supports_repeating_gradients(self) -> bool {
        self.repeating_gradients
    }

    /// Returns the resolution policy for symbolic colors.
    #[must_use]
    pub const fn symbolic_color_policy(self) -> SymbolicColorPolicy {
        self.symbolic_color_policy
    }
}

/// Semantic support and resolution policy for authored image sampling.
///
/// Boolean accessors distinguish supported operations from operations rejected
/// as unavailable by the current contract. They do not describe a selected GPU
/// device or enabled Cargo feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageSamplingCapabilities {
    image_fit: bool,
    background_position: bool,
    background_size: bool,
    repeat_xy: bool,
    repeat_round: bool,
    repeat_space: bool,
    filtered_image_paint: bool,
    color_filtered_image_paint: bool,
    image_orientation_conversion: bool,
    image_color_profile_conversion: bool,
    attachment_coordinate_policy: BackgroundAttachmentCoordinatePolicy,
    image_orientation_policy: ImageOrientationPolicy,
    image_color_profile_policy: ImageColorProfilePolicy,
}

impl ImageSamplingCapabilities {
    /// Returns whether image-fit sizing is supported.
    #[must_use]
    pub const fn supports_image_fit(self) -> bool {
        self.image_fit
    }

    /// Returns whether authored background positions are supported.
    #[must_use]
    pub const fn supports_background_position(self) -> bool {
        self.background_position
    }

    /// Returns whether authored background sizes are supported.
    #[must_use]
    pub const fn supports_background_size(self) -> bool {
        self.background_size
    }

    /// Returns whether independent x/y image repetition is supported.
    #[must_use]
    pub const fn supports_repeat_xy(self) -> bool {
        self.repeat_xy
    }

    /// Returns whether `round` image repetition is supported.
    #[must_use]
    pub const fn supports_repeat_round(self) -> bool {
        self.repeat_round
    }

    /// Returns whether `space` image repetition is supported.
    #[must_use]
    pub const fn supports_repeat_space(self) -> bool {
        self.repeat_space
    }

    /// Returns whether filtered image paint is supported.
    #[must_use]
    pub const fn supports_filtered_image_paint(self) -> bool {
        self.filtered_image_paint
    }

    /// Returns whether color-filtered image paint is supported.
    #[must_use]
    pub const fn supports_color_filtered_image_paint(self) -> bool {
        self.color_filtered_image_paint
    }

    /// Returns whether this crate performs image-orientation conversion.
    #[must_use]
    pub const fn supports_image_orientation_conversion(self) -> bool {
        self.image_orientation_conversion
    }

    /// Returns whether this crate performs image color-profile conversion.
    #[must_use]
    pub const fn supports_image_color_profile_conversion(self) -> bool {
        self.image_color_profile_conversion
    }

    /// Returns the coordinate-resolution policy for background attachment.
    #[must_use]
    pub const fn attachment_coordinate_policy(self) -> BackgroundAttachmentCoordinatePolicy {
        self.attachment_coordinate_policy
    }

    /// Returns the image-orientation resolution policy.
    #[must_use]
    pub const fn image_orientation_policy(self) -> ImageOrientationPolicy {
        self.image_orientation_policy
    }

    /// Returns the image color-profile resolution policy.
    #[must_use]
    pub const fn image_color_profile_policy(self) -> ImageColorProfilePolicy {
        self.image_color_profile_policy
    }
}

/// Semantic support for authored shadow shapes and kinds.
///
/// Each accessor reports current contract support rather than a runtime GPU fact
/// or Cargo-feature setting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowCapabilities {
    rect_rounded_circle_shadows: bool,
    ellipse_path_shadows: bool,
    inset_box_shadows: bool,
    text_shadows: bool,
}

impl ShadowCapabilities {
    /// Returns whether rectangle, rounded-rectangle, and circle shadows are supported.
    #[must_use]
    pub const fn supports_rect_rounded_circle_shadows(self) -> bool {
        self.rect_rounded_circle_shadows
    }

    /// Returns whether ellipse and arbitrary-path shadows are supported.
    #[must_use]
    pub const fn supports_ellipse_path_shadows(self) -> bool {
        self.ellipse_path_shadows
    }

    /// Returns whether inset box shadows are supported.
    #[must_use]
    pub const fn supports_inset_box_shadows(self) -> bool {
        self.inset_box_shadows
    }

    /// Returns whether text shadows are supported.
    #[must_use]
    pub const fn supports_text_shadows(self) -> bool {
        self.text_shadows
    }
}

/// Semantic support for authored filter lists and their GPU execution phases.
///
/// Each boolean distinguishes a supported operation from an unavailable one in
/// the current rendering contract. The report is not a selected-device query or
/// a Cargo-feature inventory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilterCapabilities {
    layer_filters: bool,
    ordered_filter_lists: bool,
    gpu_color_filter_execution: bool,
    gpu_blur_filter_execution: bool,
    gpu_drop_shadow_filter_execution: bool,
    filter_region_planning: bool,
}

impl FilterCapabilities {
    /// Returns whether authored layer filters are supported.
    #[must_use]
    pub const fn supports_layer_filters(self) -> bool {
        self.layer_filters
    }

    /// Returns whether authored filter lists preserve their exact operation order.
    #[must_use]
    pub const fn supports_ordered_filter_lists(self) -> bool {
        self.ordered_filter_lists
    }

    /// Returns whether color-filter graph passes execute on the GPU.
    #[must_use]
    pub const fn supports_gpu_color_filter_execution(self) -> bool {
        self.gpu_color_filter_execution
    }

    /// Returns whether blur graph passes execute on the GPU.
    #[must_use]
    pub const fn supports_gpu_blur_filter_execution(self) -> bool {
        self.gpu_blur_filter_execution
    }

    /// Returns whether drop-shadow graph passes execute on the GPU.
    #[must_use]
    pub const fn supports_gpu_drop_shadow_filter_execution(self) -> bool {
        self.gpu_drop_shadow_filter_execution
    }

    /// Returns whether filter execution regions and outsets are planned before execution.
    #[must_use]
    pub const fn supports_filter_region_planning(self) -> bool {
        self.filter_region_planning
    }
}

/// Semantic support for authored clips, masks, and resolved mask execution.
///
/// Each boolean reports current contract support rather than runtime device
/// availability or Cargo-feature selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskClipCapabilities {
    shape_clips: bool,
    clip_reference_execution: bool,
    layer_masks: bool,
    resolved_alpha_mask_execution: bool,
    luminance_mask_mode: bool,
    multi_layer_mask_composition: bool,
    mask_composite_modes: bool,
}

impl MaskClipCapabilities {
    /// Returns whether authored shape clips are supported.
    #[must_use]
    pub const fn supports_shape_clips(self) -> bool {
        self.shape_clips
    }

    /// Returns whether referenced clips execute at this boundary.
    #[must_use]
    pub const fn supports_clip_reference_execution(self) -> bool {
        self.clip_reference_execution
    }

    /// Returns whether authored layer masks are supported.
    #[must_use]
    pub const fn supports_layer_masks(self) -> bool {
        self.layer_masks
    }

    /// Returns whether resolved image alpha masks execute in the GPU composition graph.
    #[must_use]
    pub const fn supports_resolved_alpha_mask_execution(self) -> bool {
        self.resolved_alpha_mask_execution
    }

    /// Returns whether luminance mask mode is supported.
    #[must_use]
    pub const fn supports_luminance_mask_mode(self) -> bool {
        self.luminance_mask_mode
    }

    /// Returns whether multiple mask layers can be composed.
    #[must_use]
    pub const fn supports_multi_layer_mask_composition(self) -> bool {
        self.multi_layer_mask_composition
    }

    /// Returns whether authored mask composite modes are supported.
    #[must_use]
    pub const fn supports_mask_composite_modes(self) -> bool {
        self.mask_composite_modes
    }
}

/// Semantic support for authored borders, outlines, radii, and fragments.
///
/// A `true` accessor result means the current rendering contract supports the
/// named form; `false` means that form is unavailable. These values are not
/// selected-device facts or Cargo-feature flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoxDecorationCapabilities {
    border_none_hidden_styles: bool,
    border_solid_style: bool,
    border_dashed_dotted_styles: bool,
    border_double_style: bool,
    border_groove_style: bool,
    border_ridge_style: bool,
    border_inset_style: bool,
    border_outset_style: bool,
    border_radii: bool,
    outlines: bool,
    outline_none_style: bool,
    outline_solid_style: bool,
    outline_dashed_dotted_styles: bool,
    outline_double_style: bool,
    outline_auto_style: bool,
    fragments: bool,
}

impl BoxDecorationCapabilities {
    /// Returns whether `none` and `hidden` border styles are supported.
    #[must_use]
    pub const fn supports_border_none_hidden_styles(self) -> bool {
        self.border_none_hidden_styles
    }

    /// Returns whether solid borders are supported.
    #[must_use]
    pub const fn supports_border_solid_style(self) -> bool {
        self.border_solid_style
    }

    /// Returns whether dashed and dotted borders are supported.
    #[must_use]
    pub const fn supports_border_dashed_dotted_styles(self) -> bool {
        self.border_dashed_dotted_styles
    }

    /// Returns whether double borders are supported.
    #[must_use]
    pub const fn supports_border_double_style(self) -> bool {
        self.border_double_style
    }

    /// Returns whether groove borders are supported.
    #[must_use]
    pub const fn supports_border_groove_style(self) -> bool {
        self.border_groove_style
    }

    /// Returns whether ridge borders are supported.
    #[must_use]
    pub const fn supports_border_ridge_style(self) -> bool {
        self.border_ridge_style
    }

    /// Returns whether inset borders are supported.
    #[must_use]
    pub const fn supports_border_inset_style(self) -> bool {
        self.border_inset_style
    }

    /// Returns whether outset borders are supported.
    #[must_use]
    pub const fn supports_border_outset_style(self) -> bool {
        self.border_outset_style
    }

    /// Returns whether border radii are supported.
    #[must_use]
    pub const fn supports_border_radii(self) -> bool {
        self.border_radii
    }

    /// Returns whether outlines are supported.
    #[must_use]
    pub const fn supports_outlines(self) -> bool {
        self.outlines
    }

    /// Returns whether the `none` outline style is supported.
    #[must_use]
    pub const fn supports_outline_none_style(self) -> bool {
        self.outline_none_style
    }

    /// Returns whether solid outlines are supported.
    #[must_use]
    pub const fn supports_outline_solid_style(self) -> bool {
        self.outline_solid_style
    }

    /// Returns whether dashed and dotted outlines are supported.
    #[must_use]
    pub const fn supports_outline_dashed_dotted_styles(self) -> bool {
        self.outline_dashed_dotted_styles
    }

    /// Returns whether double outlines are supported.
    #[must_use]
    pub const fn supports_outline_double_style(self) -> bool {
        self.outline_double_style
    }

    /// Returns whether automatic outlines are supported.
    #[must_use]
    pub const fn supports_outline_auto_style(self) -> bool {
        self.outline_auto_style
    }

    /// Returns whether fragmented box decorations are supported.
    #[must_use]
    pub const fn supports_fragments(self) -> bool {
        self.fragments
    }
}

/// Semantic support for opacity, blending, and compositing modes.
///
/// Accessors report current rendering support, not runtime GPU capabilities or
/// enabled Cargo features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositingCapabilities {
    layer_opacity: bool,
    blend_modes: bool,
    root_backdrop_policy: bool,
    background_blend_modes: bool,
    additional_mix_blend_modes: bool,
    porter_duff_composite_modes: bool,
}

impl CompositingCapabilities {
    /// Returns whether layer opacity is supported.
    #[must_use]
    pub const fn supports_layer_opacity(self) -> bool {
        self.layer_opacity
    }

    /// Returns whether the current layer blend modes are supported.
    #[must_use]
    pub const fn supports_blend_modes(self) -> bool {
        self.blend_modes
    }

    /// Returns whether root-backdrop policy is supported.
    #[must_use]
    pub const fn supports_root_backdrop_policy(self) -> bool {
        self.root_backdrop_policy
    }

    /// Returns whether per-background blend modes are supported.
    #[must_use]
    pub const fn supports_background_blend_modes(self) -> bool {
        self.background_blend_modes
    }

    /// Returns whether additional mix-blend modes are supported.
    #[must_use]
    pub const fn supports_additional_mix_blend_modes(self) -> bool {
        self.additional_mix_blend_modes
    }

    /// Returns whether Porter-Duff composite modes are supported.
    #[must_use]
    pub const fn supports_porter_duff_composite_modes(self) -> bool {
        self.porter_duff_composite_modes
    }
}

/// Semantic support for direct and GPU-graph offscreen phases.
///
/// These flags describe operations implemented by the crate, not formats or
/// limits available on a selected runtime device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OffscreenPipelineCapabilities {
    direct_vello_opacity_isolation: bool,
    direct_vello_blend_isolation: bool,
    offscreen_layer_rendering: bool,
    persistent_effect_resources: bool,
    bounded_vello_capture: bool,
    image_pass_execution: bool,
    composite_pass_execution: bool,
    nested_opacity_composition: bool,
    mask_execution: bool,
    layer_filter_execution: bool,
    broad_backdrop_execution: bool,
    bounded_backdrop_capture: bool,
    bounded_backdrop_filter_execution: bool,
    backdrop_isolation_composition: bool,
}

impl OffscreenPipelineCapabilities {
    /// Returns whether the direct Vello route isolates opacity.
    #[must_use]
    pub const fn supports_direct_vello_opacity_isolation(self) -> bool {
        self.direct_vello_opacity_isolation
    }

    /// Returns whether the direct Vello route isolates blend operations.
    #[must_use]
    pub const fn supports_direct_vello_blend_isolation(self) -> bool {
        self.direct_vello_blend_isolation
    }

    /// Returns whether general offscreen layer rendering is supported.
    #[must_use]
    pub const fn supports_offscreen_layer_rendering(self) -> bool {
        self.offscreen_layer_rendering
    }

    /// Returns whether effect textures and uploads use persistent device-owned resources.
    #[must_use]
    pub const fn supports_persistent_effect_resources(self) -> bool {
        self.persistent_effect_resources
    }

    /// Returns whether bounded Vello spans can be captured into graph resources.
    #[must_use]
    pub const fn supports_bounded_vello_capture(self) -> bool {
        self.bounded_vello_capture
    }

    /// Returns whether image-processing graph passes execute on the GPU.
    #[must_use]
    pub const fn supports_image_pass_execution(self) -> bool {
        self.image_pass_execution
    }

    /// Returns whether graph composition passes execute on the GPU.
    #[must_use]
    pub const fn supports_composite_pass_execution(self) -> bool {
        self.composite_pass_execution
    }

    /// Returns whether nested opacity is composed in ordered GPU passes.
    #[must_use]
    pub const fn supports_nested_opacity_composition(self) -> bool {
        self.nested_opacity_composition
    }

    /// Returns whether general mask execution is supported by the offscreen pipeline.
    #[must_use]
    pub const fn supports_mask_execution(self) -> bool {
        self.mask_execution
    }

    /// Returns whether broad authored layer filters execute through the GPU graph.
    #[must_use]
    pub const fn supports_layer_filter_execution(self) -> bool {
        self.layer_filter_execution
    }

    /// Returns whether unbounded or root/nested backdrop forms execute through the GPU graph.
    #[must_use]
    pub const fn supports_broad_backdrop_execution(self) -> bool {
        self.broad_backdrop_execution
    }

    /// Returns whether bounded backdrop content can be captured for filtering.
    #[must_use]
    pub const fn supports_bounded_backdrop_capture(self) -> bool {
        self.bounded_backdrop_capture
    }

    /// Returns whether the supported bounded backdrop filter subset executes on the GPU.
    #[must_use]
    pub const fn supports_bounded_backdrop_filter_execution(self) -> bool {
        self.bounded_backdrop_filter_execution
    }

    /// Returns whether separate backdrop isolation and composition are supported.
    #[must_use]
    pub const fn supports_backdrop_isolation_composition(self) -> bool {
        self.backdrop_isolation_composition
    }
}

/// Semantic surface-construction support for the compiled target.
///
/// This report states which surface adapters the current build exposes. In
/// particular, web-canvas support requires the `render-web` feature on
/// `wasm32`. It does not state whether a selected runtime adapter, device, or
/// surface is currently available; use [`RuntimeCapabilities`] for runtime facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceCapabilities {
    headless_surfaces: bool,
    web_canvas_surfaces: bool,
}

impl SurfaceCapabilities {
    /// Returns whether headless surface construction is supported.
    #[must_use]
    pub const fn supports_headless_surfaces(self) -> bool {
        self.headless_surfaces
    }

    /// Returns whether web-canvas construction is compiled for this target.
    #[must_use]
    pub const fn supports_web_canvas_surfaces(self) -> bool {
        self.web_canvas_surfaces
    }
}

/// Semantic support for transforms and explicit coordinate-space tagging.
///
/// The flags distinguish supported authored operations from unavailable ones;
/// they are not runtime device capabilities or Cargo-feature settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformCoordinateSpaceCapabilities {
    affine_2d: bool,
    transform_origin: bool,
    skew: bool,
    transform_3d: bool,
    coordinate_space_tags: bool,
}

impl TransformCoordinateSpaceCapabilities {
    /// Returns whether two-dimensional affine transforms are supported.
    #[must_use]
    pub const fn supports_affine_2d(self) -> bool {
        self.affine_2d
    }

    /// Returns whether transform origins are supported.
    #[must_use]
    pub const fn supports_transform_origin(self) -> bool {
        self.transform_origin
    }

    /// Returns whether skew transforms are supported.
    #[must_use]
    pub const fn supports_skew(self) -> bool {
        self.skew
    }

    /// Returns whether three-dimensional transforms are supported.
    #[must_use]
    pub const fn supports_transform_3d(self) -> bool {
        self.transform_3d
    }

    /// Returns whether explicit coordinate-space tags are supported.
    #[must_use]
    pub const fn supports_coordinate_space_tags(self) -> bool {
        self.coordinate_space_tags
    }
}
