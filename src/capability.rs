use super::{
    Error, Format, PrimitiveFamily, PrimitiveOperation, Result, RuntimeCapabilityUnavailableReason,
    UnsupportedPrimitive,
};

/// Runtime facts reported for a selected safe WGPU device and surface.
///
/// These facts describe the selected runtime device/surface rather than semantic
/// rendering support or enabled Cargo features.
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

/// Available runtime facts for a selected safe WGPU device and surface.
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
/// semantic rendering support or enabled Cargo features.
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
            color_filtered_image_paint: true,
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
            gpu_color_filter_execution: false,
            gpu_blur_filter_execution: false,
            gpu_drop_shadow_filter_execution: false,
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

    #[must_use]
    pub const fn geometry_targets(self) -> GeometryTargetCapabilities {
        self.geometry_targets
    }

    #[must_use]
    pub const fn paint_sources(self) -> PaintSourceCapabilities {
        self.paint_sources
    }

    #[must_use]
    pub const fn image_sampling(self) -> ImageSamplingCapabilities {
        self.image_sampling
    }

    #[must_use]
    pub const fn shadows(self) -> ShadowCapabilities {
        self.shadows
    }

    #[must_use]
    pub const fn filters(self) -> FilterCapabilities {
        self.filters
    }

    #[must_use]
    pub const fn masks_clips(self) -> MaskClipCapabilities {
        self.masks_clips
    }

    #[must_use]
    pub const fn box_decorations(self) -> BoxDecorationCapabilities {
        self.box_decorations
    }

    #[must_use]
    pub const fn compositing(self) -> CompositingCapabilities {
        self.compositing
    }

    #[must_use]
    pub const fn offscreen_pipeline(self) -> OffscreenPipelineCapabilities {
        self.offscreen_pipeline
    }

    #[must_use]
    pub const fn surfaces(self) -> SurfaceCapabilities {
        self.surfaces
    }

    #[must_use]
    pub const fn transform_coordinate_spaces(self) -> TransformCoordinateSpaceCapabilities {
        self.transform_coordinate_spaces
    }

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitTestOwnership {
    RootOwned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicColorPolicy {
    RootResolvedOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachmentCoordinatePolicy {
    RootResolvedOrTagged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageOrientationPolicy {
    RootResolvedOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageColorProfilePolicy {
    RootResolvedOnly,
}

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
    #[must_use]
    pub const fn supports_rect_fill_stroke(self) -> bool {
        self.rect_fill_stroke
    }

    #[must_use]
    pub const fn supports_rounded_rect_fill_stroke(self) -> bool {
        self.rounded_rect_fill_stroke
    }

    #[must_use]
    pub const fn supports_circle_ellipse_fill_stroke(self) -> bool {
        self.circle_ellipse_fill_stroke
    }

    #[must_use]
    pub const fn supports_arbitrary_path_fill(self) -> bool {
        self.arbitrary_path_fill
    }

    #[must_use]
    pub const fn supports_arbitrary_path_centered_stroke(self) -> bool {
        self.arbitrary_path_centered_stroke
    }

    #[must_use]
    pub const fn supports_arbitrary_path_inside_outside_stroke(self) -> bool {
        self.arbitrary_path_inside_outside_stroke
    }

    #[must_use]
    pub const fn supports_geometry_booleans(self) -> bool {
        self.geometry_booleans
    }

    #[must_use]
    pub const fn supports_geometry_offsets(self) -> bool {
        self.geometry_offsets
    }

    #[must_use]
    pub const fn hit_testing(self) -> HitTestOwnership {
        self.hit_testing
    }
}

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
    #[must_use]
    pub const fn supports_solid_rgba(self) -> bool {
        self.solid_rgba
    }

    #[must_use]
    pub const fn supports_gradients(self) -> bool {
        self.gradients
    }

    #[must_use]
    pub const fn supports_image_paint(self) -> bool {
        self.image_paint
    }

    #[must_use]
    pub const fn supports_non_solid_shadow_paint(self) -> bool {
        self.non_solid_shadow_paint
    }

    #[must_use]
    pub const fn supports_srgb_color_conversion(self) -> bool {
        self.srgb_color_conversion
    }

    #[must_use]
    pub const fn supports_hsl_color_conversion(self) -> bool {
        self.hsl_color_conversion
    }

    #[must_use]
    pub const fn supports_unresolved_symbolic_colors(self) -> bool {
        self.unresolved_symbolic_colors
    }

    #[must_use]
    pub const fn supports_color_mix(self) -> bool {
        self.color_mix
    }

    #[must_use]
    pub const fn supports_repeating_gradients(self) -> bool {
        self.repeating_gradients
    }

    #[must_use]
    pub const fn symbolic_color_policy(self) -> SymbolicColorPolicy {
        self.symbolic_color_policy
    }
}

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
    #[must_use]
    pub const fn supports_image_fit(self) -> bool {
        self.image_fit
    }

    #[must_use]
    pub const fn supports_background_position(self) -> bool {
        self.background_position
    }

    #[must_use]
    pub const fn supports_background_size(self) -> bool {
        self.background_size
    }

    #[must_use]
    pub const fn supports_repeat_xy(self) -> bool {
        self.repeat_xy
    }

    #[must_use]
    pub const fn supports_repeat_round(self) -> bool {
        self.repeat_round
    }

    #[must_use]
    pub const fn supports_repeat_space(self) -> bool {
        self.repeat_space
    }

    #[must_use]
    pub const fn supports_filtered_image_paint(self) -> bool {
        self.filtered_image_paint
    }

    #[must_use]
    pub const fn supports_color_filtered_image_paint(self) -> bool {
        self.color_filtered_image_paint
    }

    #[must_use]
    pub const fn supports_image_orientation_conversion(self) -> bool {
        self.image_orientation_conversion
    }

    #[must_use]
    pub const fn supports_image_color_profile_conversion(self) -> bool {
        self.image_color_profile_conversion
    }

    #[must_use]
    pub const fn attachment_coordinate_policy(self) -> BackgroundAttachmentCoordinatePolicy {
        self.attachment_coordinate_policy
    }

    #[must_use]
    pub const fn image_orientation_policy(self) -> ImageOrientationPolicy {
        self.image_orientation_policy
    }

    #[must_use]
    pub const fn image_color_profile_policy(self) -> ImageColorProfilePolicy {
        self.image_color_profile_policy
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowCapabilities {
    rect_rounded_circle_shadows: bool,
    ellipse_path_shadows: bool,
    inset_box_shadows: bool,
    text_shadows: bool,
}

impl ShadowCapabilities {
    #[must_use]
    pub const fn supports_rect_rounded_circle_shadows(self) -> bool {
        self.rect_rounded_circle_shadows
    }

    #[must_use]
    pub const fn supports_ellipse_path_shadows(self) -> bool {
        self.ellipse_path_shadows
    }

    #[must_use]
    pub const fn supports_inset_box_shadows(self) -> bool {
        self.inset_box_shadows
    }

    #[must_use]
    pub const fn supports_text_shadows(self) -> bool {
        self.text_shadows
    }
}

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
    #[must_use]
    pub const fn supports_shape_clips(self) -> bool {
        self.shape_clips
    }

    #[must_use]
    pub const fn supports_clip_reference_execution(self) -> bool {
        self.clip_reference_execution
    }

    #[must_use]
    pub const fn supports_layer_masks(self) -> bool {
        self.layer_masks
    }

    /// Returns whether resolved image alpha masks execute in the GPU composition graph.
    #[must_use]
    pub const fn supports_resolved_alpha_mask_execution(self) -> bool {
        self.resolved_alpha_mask_execution
    }

    #[must_use]
    pub const fn supports_luminance_mask_mode(self) -> bool {
        self.luminance_mask_mode
    }

    #[must_use]
    pub const fn supports_multi_layer_mask_composition(self) -> bool {
        self.multi_layer_mask_composition
    }

    #[must_use]
    pub const fn supports_mask_composite_modes(self) -> bool {
        self.mask_composite_modes
    }
}

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
    #[must_use]
    pub const fn supports_border_none_hidden_styles(self) -> bool {
        self.border_none_hidden_styles
    }

    #[must_use]
    pub const fn supports_border_solid_style(self) -> bool {
        self.border_solid_style
    }

    #[must_use]
    pub const fn supports_border_dashed_dotted_styles(self) -> bool {
        self.border_dashed_dotted_styles
    }

    #[must_use]
    pub const fn supports_border_double_style(self) -> bool {
        self.border_double_style
    }

    #[must_use]
    pub const fn supports_border_groove_style(self) -> bool {
        self.border_groove_style
    }

    #[must_use]
    pub const fn supports_border_ridge_style(self) -> bool {
        self.border_ridge_style
    }

    #[must_use]
    pub const fn supports_border_inset_style(self) -> bool {
        self.border_inset_style
    }

    #[must_use]
    pub const fn supports_border_outset_style(self) -> bool {
        self.border_outset_style
    }

    #[must_use]
    pub const fn supports_border_radii(self) -> bool {
        self.border_radii
    }

    #[must_use]
    pub const fn supports_outlines(self) -> bool {
        self.outlines
    }

    #[must_use]
    pub const fn supports_outline_none_style(self) -> bool {
        self.outline_none_style
    }

    #[must_use]
    pub const fn supports_outline_solid_style(self) -> bool {
        self.outline_solid_style
    }

    #[must_use]
    pub const fn supports_outline_dashed_dotted_styles(self) -> bool {
        self.outline_dashed_dotted_styles
    }

    #[must_use]
    pub const fn supports_outline_double_style(self) -> bool {
        self.outline_double_style
    }

    #[must_use]
    pub const fn supports_outline_auto_style(self) -> bool {
        self.outline_auto_style
    }

    #[must_use]
    pub const fn supports_fragments(self) -> bool {
        self.fragments
    }
}

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
    #[must_use]
    pub const fn supports_layer_opacity(self) -> bool {
        self.layer_opacity
    }

    #[must_use]
    pub const fn supports_blend_modes(self) -> bool {
        self.blend_modes
    }

    #[must_use]
    pub const fn supports_root_backdrop_policy(self) -> bool {
        self.root_backdrop_policy
    }

    #[must_use]
    pub const fn supports_background_blend_modes(self) -> bool {
        self.background_blend_modes
    }

    #[must_use]
    pub const fn supports_additional_mix_blend_modes(self) -> bool {
        self.additional_mix_blend_modes
    }

    #[must_use]
    pub const fn supports_porter_duff_composite_modes(self) -> bool {
        self.porter_duff_composite_modes
    }
}

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
    #[must_use]
    pub const fn supports_direct_vello_opacity_isolation(self) -> bool {
        self.direct_vello_opacity_isolation
    }

    #[must_use]
    pub const fn supports_direct_vello_blend_isolation(self) -> bool {
        self.direct_vello_blend_isolation
    }

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

    #[must_use]
    pub const fn supports_mask_execution(self) -> bool {
        self.mask_execution
    }

    #[must_use]
    pub const fn supports_layer_filter_execution(self) -> bool {
        self.layer_filter_execution
    }

    #[must_use]
    pub const fn supports_broad_backdrop_execution(self) -> bool {
        self.broad_backdrop_execution
    }

    #[must_use]
    pub const fn supports_bounded_backdrop_capture(self) -> bool {
        self.bounded_backdrop_capture
    }

    #[must_use]
    pub const fn supports_bounded_backdrop_filter_execution(self) -> bool {
        self.bounded_backdrop_filter_execution
    }

    #[must_use]
    pub const fn supports_backdrop_isolation_composition(self) -> bool {
        self.backdrop_isolation_composition
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceCapabilities {
    headless_surfaces: bool,
    web_canvas_surfaces: bool,
}

impl SurfaceCapabilities {
    #[must_use]
    pub const fn supports_headless_surfaces(self) -> bool {
        self.headless_surfaces
    }

    #[must_use]
    pub const fn supports_web_canvas_surfaces(self) -> bool {
        self.web_canvas_surfaces
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformCoordinateSpaceCapabilities {
    affine_2d: bool,
    transform_origin: bool,
    skew: bool,
    transform_3d: bool,
    coordinate_space_tags: bool,
}

impl TransformCoordinateSpaceCapabilities {
    #[must_use]
    pub const fn supports_affine_2d(self) -> bool {
        self.affine_2d
    }

    #[must_use]
    pub const fn supports_transform_origin(self) -> bool {
        self.transform_origin
    }

    #[must_use]
    pub const fn supports_skew(self) -> bool {
        self.skew
    }

    #[must_use]
    pub const fn supports_transform_3d(self) -> bool {
        self.transform_3d
    }

    #[must_use]
    pub const fn supports_coordinate_space_tags(self) -> bool {
        self.coordinate_space_tags
    }
}
