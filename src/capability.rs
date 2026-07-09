use super::{Error, PrimitiveFamily, PrimitiveOperation, Result, UnsupportedPrimitive};

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
    pub const VELLO_0_9: Self = Self {
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
            color_filter_classification: true,
            color_filter_pipeline_execution: true,
            materialized_image_filter_classification: true,
            materialized_blur_filter_execution: true,
            materialized_drop_shadow_filter_execution: false,
            filter_region_outset_planning: true,
            cpu_reference_blur_fallback: true,
        },
        masks_clips: MaskClipCapabilities {
            shape_clips: true,
            layer_masks: false,
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
        },
        offscreen_pipeline: OffscreenPipelineCapabilities {
            direct_vello_opacity_isolation: true,
            direct_vello_blend_isolation: true,
            offscreen_layer_rendering: false,
            texture_cache_upload_lifecycle: false,
            rect_fullscreen_shader_passes: false,
            cpu_reference_buffers: false,
            nested_opacity_planning: false,
            mask_execution: false,
            filter_execution: false,
            backdrop_execution: false,
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
            (
                PrimitiveFamily::GeometryTargets,
                PrimitiveOperation::InsideOutsidePathStrokeAlignment,
            ) => self
                .geometry_targets
                .supports_arbitrary_path_inside_outside_stroke(),
            (PrimitiveFamily::GeometryTargets, PrimitiveOperation::GeometryBooleanOperation) => {
                self.geometry_targets.supports_geometry_booleans()
            }
            (PrimitiveFamily::GeometryTargets, PrimitiveOperation::GeometryOffsetOperation) => {
                self.geometry_targets.supports_geometry_offsets()
            }
            (PrimitiveFamily::PaintSources, PrimitiveOperation::NonSolidShadowPaint) => {
                self.paint_sources.supports_non_solid_shadow_paint()
            }
            (PrimitiveFamily::PaintSources, PrimitiveOperation::UnresolvedSymbolicColor) => {
                self.paint_sources.supports_unresolved_symbolic_colors()
            }
            (PrimitiveFamily::PaintSources, PrimitiveOperation::ColorMixFunction) => {
                self.paint_sources.supports_color_mix()
            }
            (PrimitiveFamily::PaintSources, PrimitiveOperation::UnsupportedColorSpace) => false,
            (PrimitiveFamily::PaintSources, PrimitiveOperation::RepeatingGradient) => {
                self.paint_sources.supports_repeating_gradients()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::BackgroundRepeatRound) => {
                self.image_sampling.supports_repeat_round()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::BackgroundRepeatSpace) => {
                self.image_sampling.supports_repeat_space()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::FilteredImagePaint) => {
                self.image_sampling.supports_filtered_image_paint()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::ColorFilteredImagePaint) => {
                self.image_sampling.supports_color_filtered_image_paint()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::ImageOrientationConversion) => {
                self.image_sampling.supports_image_orientation_conversion()
            }
            (PrimitiveFamily::ImageSampling, PrimitiveOperation::ImageColorProfileConversion) => {
                self.image_sampling
                    .supports_image_color_profile_conversion()
            }
            (PrimitiveFamily::Shadows, PrimitiveOperation::EllipsePathShadowShape) => {
                self.shadows.supports_ellipse_path_shadows()
            }
            (PrimitiveFamily::Shadows, PrimitiveOperation::InsetBoxShadow) => {
                self.shadows.supports_inset_box_shadows()
            }
            (PrimitiveFamily::Shadows, PrimitiveOperation::TextShadow) => {
                self.shadows.supports_text_shadows()
            }
            (PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter) => {
                self.filters.supports_layer_filters()
            }
            (PrimitiveFamily::Filters, PrimitiveOperation::ColorFilterClassification) => {
                self.filters.supports_color_filter_classification()
            }
            (PrimitiveFamily::Filters, PrimitiveOperation::ColorFilterPipelineExecution) => {
                self.filters.supports_color_filter_pipeline_execution()
            }
            (
                PrimitiveFamily::Filters,
                PrimitiveOperation::MaterializedImageFilterClassification,
            ) => self
                .filters
                .supports_materialized_image_filter_classification(),
            (PrimitiveFamily::Filters, PrimitiveOperation::MaterializedBlurFilterExecution) => {
                self.filters.supports_materialized_blur_filter_execution()
            }
            (
                PrimitiveFamily::Filters,
                PrimitiveOperation::MaterializedDropShadowFilterExecution,
            ) => self
                .filters
                .supports_materialized_drop_shadow_filter_execution(),
            (PrimitiveFamily::Filters, PrimitiveOperation::FilterRegionOutsetPlanning) => {
                self.filters.supports_filter_region_outset_planning()
            }
            (PrimitiveFamily::Filters, PrimitiveOperation::CpuReferenceBlurFallback) => {
                self.filters.supports_cpu_reference_blur_fallback()
            }
            (
                PrimitiveFamily::Filters,
                PrimitiveOperation::ColorFilterBlur | PrimitiveOperation::ColorFilterDropShadow,
            ) => false,
            (PrimitiveFamily::MasksAndClips, PrimitiveOperation::LayerMask) => {
                self.masks_clips.supports_layer_masks()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::BorderGrooveStyle) => {
                self.box_decorations.supports_border_groove_style()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::BorderRidgeStyle) => {
                self.box_decorations.supports_border_ridge_style()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::BorderInsetStyle) => {
                self.box_decorations.supports_border_inset_style()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::BorderOutsetStyle) => {
                self.box_decorations.supports_border_outset_style()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::OutlineDoubleStyle) => {
                self.box_decorations.supports_outline_double_style()
            }
            (PrimitiveFamily::BoxDecorations, PrimitiveOperation::OutlineAutoStyle) => {
                self.box_decorations.supports_outline_auto_style()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::OffscreenLayerRendering) => {
                self.offscreen_pipeline.supports_offscreen_layer_rendering()
            }
            (
                PrimitiveFamily::OffscreenPipeline,
                PrimitiveOperation::TextureCacheUploadLifecycle,
            ) => self
                .offscreen_pipeline
                .supports_texture_cache_upload_lifecycle(),
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::RectFullscreenShaderPass) => {
                self.offscreen_pipeline
                    .supports_rect_fullscreen_shader_passes()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::CpuReferenceBuffer) => {
                self.offscreen_pipeline.supports_cpu_reference_buffers()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::NestedOpacityPlanning) => {
                self.offscreen_pipeline.supports_nested_opacity_planning()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::MaskExecution) => {
                self.offscreen_pipeline.supports_mask_execution()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::FilterExecution) => {
                self.offscreen_pipeline.supports_filter_execution()
            }
            (PrimitiveFamily::OffscreenPipeline, PrimitiveOperation::BackdropExecution) => {
                self.offscreen_pipeline.supports_backdrop_execution()
            }
            (PrimitiveFamily::Surfaces, PrimitiveOperation::WebCanvasSurface) => {
                self.surfaces.supports_web_canvas_surfaces()
            }
            (
                PrimitiveFamily::TransformsAndCoordinateSpaces,
                PrimitiveOperation::Matrix3dTransform
                | PrimitiveOperation::PerspectiveTransform
                | PrimitiveOperation::Rotate3dTransform
                | PrimitiveOperation::TranslateZTransform
                | PrimitiveOperation::ScaleZTransform,
            ) => self.transform_coordinate_spaces.supports_transform_3d(),
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
    color_filter_classification: bool,
    color_filter_pipeline_execution: bool,
    materialized_image_filter_classification: bool,
    materialized_blur_filter_execution: bool,
    materialized_drop_shadow_filter_execution: bool,
    filter_region_outset_planning: bool,
    cpu_reference_blur_fallback: bool,
}

impl FilterCapabilities {
    #[must_use]
    pub const fn supports_layer_filters(self) -> bool {
        self.layer_filters
    }

    #[must_use]
    pub const fn supports_color_filter_classification(self) -> bool {
        self.color_filter_classification
    }

    #[must_use]
    pub const fn supports_color_filter_pipeline_execution(self) -> bool {
        self.color_filter_pipeline_execution
    }

    #[must_use]
    pub const fn supports_materialized_image_filter_classification(self) -> bool {
        self.materialized_image_filter_classification
    }

    #[must_use]
    pub const fn supports_materialized_blur_filter_execution(self) -> bool {
        self.materialized_blur_filter_execution
    }

    #[must_use]
    pub const fn supports_materialized_drop_shadow_filter_execution(self) -> bool {
        self.materialized_drop_shadow_filter_execution
    }

    #[must_use]
    pub const fn supports_filter_region_outset_planning(self) -> bool {
        self.filter_region_outset_planning
    }

    #[must_use]
    pub const fn supports_cpu_reference_blur_fallback(self) -> bool {
        self.cpu_reference_blur_fallback
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskClipCapabilities {
    shape_clips: bool,
    layer_masks: bool,
}

impl MaskClipCapabilities {
    #[must_use]
    pub const fn supports_shape_clips(self) -> bool {
        self.shape_clips
    }

    #[must_use]
    pub const fn supports_layer_masks(self) -> bool {
        self.layer_masks
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OffscreenPipelineCapabilities {
    direct_vello_opacity_isolation: bool,
    direct_vello_blend_isolation: bool,
    offscreen_layer_rendering: bool,
    texture_cache_upload_lifecycle: bool,
    rect_fullscreen_shader_passes: bool,
    cpu_reference_buffers: bool,
    nested_opacity_planning: bool,
    mask_execution: bool,
    filter_execution: bool,
    backdrop_execution: bool,
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

    #[must_use]
    pub const fn supports_texture_cache_upload_lifecycle(self) -> bool {
        self.texture_cache_upload_lifecycle
    }

    #[must_use]
    pub const fn supports_rect_fullscreen_shader_passes(self) -> bool {
        self.rect_fullscreen_shader_passes
    }

    #[must_use]
    pub const fn supports_cpu_reference_buffers(self) -> bool {
        self.cpu_reference_buffers
    }

    #[must_use]
    pub const fn supports_nested_opacity_planning(self) -> bool {
        self.nested_opacity_planning
    }

    #[must_use]
    pub const fn supports_mask_execution(self) -> bool {
        self.mask_execution
    }

    #[must_use]
    pub const fn supports_filter_execution(self) -> bool {
        self.filter_execution
    }

    #[must_use]
    pub const fn supports_backdrop_execution(self) -> bool {
        self.backdrop_execution
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
