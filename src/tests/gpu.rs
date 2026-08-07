use crate::{
    Antialiasing, AvailableRuntimeCapabilities, BackdropCaptureBounds, BackdropFilterInput,
    BlendMode, Capabilities, ClipInput, Color, EffectPrecisionCapabilities, EffectQualityPolicy,
    Error, ErrorCode, Extend, Filter, FilterAmount, FilterAngle, FilterBlur, FilterCapabilities,
    FilterDropShadow, FilterList, FilterOp, FilteredImagePaint, Format, Image, ImageBuffer,
    ImageFit, ImageId, ImageQuality, InvalidValue, Layer, MaskClipCapabilities, MaskInput,
    MaskLayerStack, MaskMode, OffscreenPipelineCapabilities, Options, Parameters, PhysicalSize,
    Point, PrimitiveFamily, PrimitiveOperation, Rect, RenderRoute, Renderer, ResolvedImageResource,
    ResolvedLayerAlphaMask, ResourceCacheBudget, Result, RuntimeCapabilities,
    RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation, Scene,
    Shadow, Shape, Size, Stats, Surface, Transform, UnitFilterAmount, UnresolvedResource,
    UnresolvedResourceKind, UnsupportedPrimitive,
    backend::{
        Backend, CompositionBlendVectorForTest, CompositionGpuVectorResultsForTest,
        CompositionMaskSamplingInputForTest, CompositionMaskSamplingVectorForTest,
        CompositionPreparedGpuVectorsForTest, DeviceCapabilities, DeviceSlotIdentity,
    },
    command,
    filter::{
        BlurPolicy, BlurRadiusInterpretation, KernelSupportRadius, LargeBlurRadiusPolicy,
        TransparentEdgeSamplingPolicy,
    },
    pass::pass_spatial_uniform_bytes_for_test,
    reference::{
        self, CompiledColorFilterPipeline, MaterializedDropShadowOffsetQuantizationPolicy,
        PremultipliedRgba8, ReferencePremultipliedRgba8Buffer,
    },
    resource::{
        GaussianKernelBufferLimits, GaussianKernelPlan, GaussianKernelSamplingForm, WorkingFormat,
    },
    shader::device_pass_cache_owns_exact_key_spaces_for_test,
    style::ColorFilterOp,
};

use super::premultiply_u8_channel_for_test;
use super::{
    BoundedBackdropProductionFrameForTest, COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X,
    ColorFilterProductionFrameForTest, GraphPublicStatsForTest,
    SpatialFilterProductionFrameForTest, UnwrapOrPanicForTest,
    assert_gaussian_kernel_upload_lifecycle, bounded_backdrop_integration_fixture_for_test,
    bounded_backdrop_reference_rect_for_test, color_filter_list, color_filter_pipeline,
    color_filter_pixel_renderer_for_test, color_filter_public_color_graph_diagnostic_for_test,
    color_filter_repeated_resource_observations_are_stable_for_test,
    color_filter_retention_fixture_for_test, color_filter_signed_source_scene_for_test,
    color_filter_unsupported_backdrop_scene_for_test, color_from_straight_rgba8_for_test,
    composition_composite_requests_for_test, composition_frame_context_for_test,
    composition_mask_image_for_test, composition_mask_image_from_alpha_for_test,
    composition_reuse_scene_and_oracle_for_test,
    composition_selected_backend_and_requests_for_test,
    composition_shader_composite_commands_for_test, default_graph_working_format_for_test,
    graph_canonical_pixel_for_test, graph_encoding_backend_for_test, graph_pixel_renderer_for_test,
    graph_pixels_match_for_test, graph_supported_working_formats_for_test,
    graph_transform_point_for_test, high_precision_terminal_error_for_test,
    normalize_single_layer_error, reduced_precision_terminal_error_for_test,
    reference_premultiplied_pixel_for_test, reference_solid_for_test,
    reference_straight_bytes_for_test, render_bounded_backdrop_fixture_for_test,
    render_color_filter_fixture_for_test, render_spatial_filter_fixture_for_test,
    repeated_spatial_filter_resources_are_stable_for_test, resolved_layer_alpha_mask_from_buffer,
    retained_public_filter_diagnostics_are_exact_for_test, single_filter_list_for_test,
    spatial_filter_image_scene_for_test, spatial_filter_maximum_error_for_test,
    spatial_filter_mixed_filter_fixture_for_test,
    spatial_filter_public_spatial_graph_diagnostic_for_test,
    spatial_filter_reference_buffer_for_test,
    spatial_filter_zero_budget_releases_all_frame_resources_for_test,
    support::{
        assert_premultiplied, authored_color_filter_runs_for_test,
        bounded_backdrop_graph_commands_for_test, color_then_blur_filters_for_test,
        composition_commands_for_test, filter_graph_commands_for_test,
        filter_graph_context_for_test, graph_shader_commands_for_test,
        graph_shader_frame_context_for_test, image_from_buffer, pixel_alpha, pixel_rgba,
        spatial_filter_authored_filter_steps_for_test,
    },
};
use super::{error::BackendErrorCode, gpu_transaction::GpuOperationStage};

use std::{sync::Arc, time::Duration};

#[test]
fn shader_clear_fill_pass_encodes_when_gpu_context_is_available() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "real GPU clear/fill coverage requires a host adapter"
    );
    let output = pollster::block_on(renderer.scoped_clear_fill_probe_for_test())
        .expect("available GPU clear/fill work must resolve through its transaction");
    let [red, green, blue, alpha] = pixel_rgba(&output, 0, 0);
    assert!(
        (60..=68).contains(&red),
        "red channel should be cleared: {red}"
    );
    assert!(
        (124..=132).contains(&green),
        "green channel should be cleared: {green}"
    );
    assert!(
        (187..=195).contains(&blue),
        "blue channel should be cleared: {blue}"
    );
    assert_eq!(alpha, 255);
}

fn color_filter_commands_for_shader_test() -> command::RenderCommands {
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.0).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(f64::from_bits(1)).unwrap()),
        FilterOp::grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(0.5).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::saturate(FilterAmount::try_new(f64::MAX).unwrap()),
        FilterOp::sepia(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(-2.0, 3.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 12.0, 10.0), Color::BLACK)
        .layer(backdrop, |scene| {
            scene.fill(
                Rect::new(-1.0, 4.0, 3.0, 2.0),
                Color::try_rgba(0.5, 0.25, 0.75, 0.5).unwrap(),
            );
        });
    scene.normalize(Capabilities::CURRENT).unwrap()
}

fn color_filter_frame_context_for_shader_test() -> crate::frame::FrameContext {
    crate::frame::FrameContext::try_new(
        Size::new(16.0, 12.0),
        1.0,
        Antialiasing::Msaa8,
        Color::TRANSPARENT,
    )
    .unwrap()
}

fn color_filter_expected_operation_bytes_for_test() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + 8 * 32);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&[0_u8; 12]);
    let mut push_record = |tag: u32, zero: u32, exponent: i32, payload: [f32; 4]| {
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(&zero.to_le_bytes());
        bytes.extend_from_slice(&exponent.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        for value in payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    };
    push_record(0, 1, 0, [0.0, 0.0, 0.0, 0.0]);
    push_record(1, 0, -1073, [0.5, 0.0, 0.0, 0.0]);
    push_record(2, 0, 0, [0.25, 0.0, 0.0, 0.0]);
    let reduced_angle = std::f64::consts::FRAC_PI_2.rem_euclid(std::f64::consts::TAU) as f32;
    let (sine, cosine) = reduced_angle.sin_cos();
    push_record(3, 0, 0, [sine, cosine, 0.0, 0.0]);
    push_record(4, 0, 0, [0.5, 0.0, 0.0, 0.0]);
    push_record(5, 0, 0, [0.75, 0.0, 0.0, 0.0]);
    push_record(6, 0, 1025, [0.5, 0.0, 0.0, 0.0]);
    push_record(7, 0, 0, [1.0, 0.0, 0.0, 0.0]);
    bytes
}

#[test]
fn color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries() {
    let observed = crate::pass::color_filter_operation_bytes_observation_for_test(
        color_filter_commands_for_shader_test(),
        color_filter_frame_context_for_shader_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    )
    .unwrap_or_panic_for_test(
        "the color-filter operation-byte fixture must lower without allocation",
    );

    assert!(
        observed.bytes == color_filter_expected_operation_bytes_for_test()
            && observed.preserves_one_clamp_per_record,
        "color operation bytes lost an authored finite scalar or clamp"
    );
}

#[test]
fn color_filter_operation_buffer_limits_return_exact_invalid_input_before_allocation() {
    let observed = crate::pass::color_filter_operation_buffer_limit_observation_for_test();

    assert!(
        observed.count_overflow_is_exact
            && observed.max_buffer_size_is_exact
            && observed.max_storage_binding_size_is_exact
            && observed.equality_at_both_limits_is_accepted
            && observed.rejects_before_any_allocation_or_cache_action,
        "an oversized color-filter buffer lacks its exact pre-allocation diagnostic"
    );
}

#[test]
fn color_filter_cache_realizes_checked_high_and_reduced_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires backend selection",
        )
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires a host adapter",
        );
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test(
            "checked color-filter shader realization requires a ready device",
        );
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        crate::pass::color_filter_cache_realization_observation_for_test(
            ready.device_for_test(),
            color_filter_commands_for_shader_test(),
            color_filter_frame_context_for_shader_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked color-filter shader must reach real WGPU realization");

    assert!(
        observed.realizes_high_precision
            && observed.realizes_reduced_precision
            && observed.checked_scope_is_clean
            && observed.publishes_only_color_filter_entries,
        "the checked color-filter shader program is unrealized"
    );
}

#[test]
fn color_filter_layout_binds_exact_source_spatial_and_operations() {
    let observed = crate::pass::color_filter_layout_observation_for_test(
        color_filter_commands_for_shader_test(),
        color_filter_frame_context_for_shader_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_filter_source
            && observed.binds_exact_nearest_sampler
            && observed.binds_spatial_and_read_only_operations
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the color-filter layout has a missing or dummy binding"
    );
}

#[test]
fn mixed_color_and_spatial_filters_preserve_the_unsupported_operation_diagnostic() {
    let observed = crate::pass::mixed_color_unsupported_diagnostic_observation_for_test(
        authored_color_filter_runs_for_test(),
        color_then_blur_filters_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.pure_color_retains_gpu_color_diagnostic
            && observed.color_then_blur_reports_gpu_blur_diagnostic
            && observed.mixed_graph_stays_outside_color_filter_preparation,
        "a spatial-filter pass was admitted or color filtering masked its diagnostic"
    );
}

#[test]
fn copy_backdrop_layout_binds_parent_and_spatial_mapping() {
    let observed = crate::pass::copy_backdrop_layout_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        crate::frame::FrameContext::try_new(
            Size::new(16.0, 12.0),
            1.0,
            Antialiasing::Msaa8,
            Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
        )
        .unwrap(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_completed_parent
            && observed.binds_only_one_nearest_transparent_sampler
            && observed.binds_only_spatial_uniform
            && observed.targets_only_the_working_format
            && observed.source_and_result_are_distinct,
        "the copy-backdrop layout is not exact"
    );
}

#[test]
fn copy_backdrop_cache_realizes_checked_working_format_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires backend selection")
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked copy-backdrop realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        crate::pass::copy_backdrop_cache_realization_observation_for_test(
            ready.device_for_test(),
            bounded_backdrop_graph_commands_for_test(),
            crate::frame::FrameContext::try_new(
                Size::new(16.0, 12.0),
                1.0,
                Antialiasing::Msaa8,
                Color::try_rgba(0.125, 0.25, 0.5, 1.0).unwrap(),
            )
            .unwrap(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked copy-backdrop shader must reach real WGPU realization");

    assert!(
        observed.realizes_high_precision
            && observed.realizes_reduced_precision
            && observed.checked_scope_is_clean
            && observed.publishes_only_copy_backdrop_entries
            && observed.rejects_unsupported_format_before_publication,
        "the copy-backdrop working-format program is unrealized"
    );
}

#[test]
fn prepared_copy_backdrop_objects_expose_exact_encoding_handles() {
    fn require_copy_handles(objects: &crate::shader::ProvisionalCopyBackdropPassObjects<'_>) {
        let _: &wgpu::Sampler = objects.parent_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    let _ = require_copy_handles;
}

#[test]
fn backdrop_blur_cache_separates_transparent_and_mirrored_edge_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires backend selection")
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked backdrop-blur realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        crate::pass::backdrop_blur_cache_realization_observation_for_test(
            ready.device_for_test(),
            spatial_filter_authored_filter_steps_for_test(),
            filter_graph_commands_for_test(),
            bounded_backdrop_graph_commands_for_test(),
            filter_graph_context_for_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test(
        "transparent and mirrored backdrop-blur programs must reach realization",
    );

    assert!(
        observed.realizes_all_transparent_and_mirrored_programs
            && observed.checked_scope_is_clean
            && observed.publishes_exact_edge_programs
            && observed.edge_program_keys_are_distinct,
        "the checked blur cache does not distinguish transparent and mirrored edge programs"
    );
}

#[test]
fn backdrop_blur_layout_carries_semantic_mirror_bounds() {
    let observed = crate::pass::backdrop_blur_layout_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_all_axis_input_precision_and_edge_keys
            && observed.binds_exact_working_source
            && observed.binds_only_one_linear_mirror_sampler
            && observed.binds_spatial_kernel_and_semantic_bounds
            && observed.targets_only_the_working_format
            && observed.semantic_bounds_match_every_mirrored_read
            && observed.shader_mirrors_logical_bounds_before_texture_mapping,
        "the backdrop blur layout omits exact semantic mirror bounds or program facts"
    );
}

#[test]
fn backdrop_filter_chain_preserves_authored_order_and_clamp_boundaries() {
    use crate::pass::SpatialFilterPassTagForTest as Tag;

    let observed = crate::pass::backdrop_filter_chain_observation_for_test(
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.pass_order
            == [
                Tag::Color,
                Tag::BlurHorizontalRgba,
                Tag::BlurVerticalRgba,
                Tag::BlurHorizontalSourceAlpha,
                Tag::BlurVerticalSourceAlpha,
                Tag::DropShadowColorize,
                Tag::DropShadowMerge,
            ]
            && observed.every_backdrop_blur_uses_mirror
            && observed.source_alpha_blur_uses_mirror
            && observed.every_color_operation_retains_one_clamp
            && observed.semantic_bounds_are_exact
            && observed.every_mirrored_stage_is_realizable,
        "the authored backdrop filter chain contains unrealizable mirrored stages"
    );
}

#[test]
fn backdrop_graph_encodes_copy_filter_clip_foreground_and_group_in_order() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the bounded backdrop fixture must reach its shared GPU graph executor",
    );

    assert!(
        observed.encodes_copy_filter_clip_foreground_and_group_in_order
            && observed.parent_is_copied_once
            && observed.releases_at_validated_last_use,
        "the scheduler has no bounded backdrop encoding route"
    );
}

#[test]
fn backdrop_copy_filter_and_group_use_distinct_resources() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the backdrop alias fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.copy_filter_foreground_and_group_are_distinct
            && observed.releases_at_validated_last_use,
        "the backdrop copy, filters, foreground, or group alias"
    );
}

#[test]
fn later_sibling_dependency_follows_completed_backdrop_group() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.backdrop_graph_encoding_observation_for_test(
        identity,
        bounded_backdrop_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the backdrop dependency fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.later_sibling_reads_completed_group
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "the later sibling did not follow the committed backdrop-group transition"
    );
}

#[test]
fn gaussian_kernel_bytes_are_symmetric_normalized_and_exactly_cached() {
    assert_gaussian_kernel_upload_lifecycle(2.0, 1.5, 2.5);
    let plan = GaussianKernelPlan::try_new(2.0, 1.5, 2.5, GaussianKernelSamplingForm::PairedLinear)
        .unwrap();
    let samples = plan
        .upload_bytes()
        .chunks_exact(8)
        .map(|sample| {
            (
                f32::from_le_bytes(sample[0..4].try_into().unwrap()),
                f32::from_le_bytes(sample[4..8].try_into().unwrap()),
            )
        })
        .collect::<Vec<_>>();
    let symmetric = samples[1..].chunks_exact(2).all(|pair| {
        pair[0].0.to_bits() == (-pair[1].0).to_bits() && pair[0].1.to_bits() == pair[1].1.to_bits()
    });
    let normalized = (samples.iter().map(|sample| sample.1).sum::<f32>() - 1.0).abs() <= 1.0e-6;
    let storage_limit_is_checked = plan
        .validate_buffer_limits(GaussianKernelBufferLimits::for_test(
            plan.byte_len(),
            plan.byte_len() - 1,
        ))
        .is_err();

    assert!(
        symmetric && normalized && storage_limit_is_checked,
        "Gaussian sample bytes or their exact pre-allocation cache contract differ"
    );
}

#[test]
fn blur_layout_binds_exact_source_spatial_and_kernel() {
    let observed = crate::pass::blur_layout_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_all_axis_input_and_precision_keys
            && observed.binds_exact_working_source
            && observed.binds_only_one_linear_sampler
            && observed.binds_spatial_and_read_only_kernel
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the blur layout has a missing or dummy binding"
    );
}

#[test]
fn blur_cache_realizes_checked_axis_input_and_precision_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked blur realization requires backend selection")
        .unwrap_or_panic_for_test("checked blur realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked blur realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(crate::pass::blur_cache_realization_observation_for_test(
        ready.device_for_test(),
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        capabilities,
    ))
    .unwrap_or_panic_for_test("the checked blur shader must reach real WGPU realization");

    assert!(
        observed.realizes_all_eight_programs
            && observed.checked_scope_is_clean
            && observed.publishes_only_blur_entries,
        "the checked blur programs are unrealized"
    );
}

#[test]
fn drop_shadow_parameter_bytes_preserve_fractional_offset_and_solid_color() {
    let bytes = crate::shader::drop_shadow_parameter_bytes_for_test(
        Point::new(-1.5, 0.75),
        Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
    )
    .unwrap();
    let scalar = |offset| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    assert!(
        scalar(0).to_bits() == (-1.5_f32).to_bits()
            && scalar(4).to_bits() == 0.75_f32.to_bits()
            && bytes[8..16] == [0; 8]
            && scalar(16).to_bits() == 0.125_f32.to_bits()
            && scalar(20).to_bits() == 0.25_f32.to_bits()
            && scalar(24).to_bits() == 0.375_f32.to_bits()
            && scalar(28).to_bits() == 0.5_f32.to_bits()
            && crate::shader::drop_shadow_parameter_bytes_for_test(
                Point::new(f64::NAN, 0.0),
                Color::BLACK,
            )
            .is_err(),
        "drop-shadow parameters lost a fractional offset, finite layout, or solid premultiplied color"
    );
}

#[test]
fn drop_shadow_layout_binds_blurred_alpha_spatial_and_parameters() {
    let observed = crate::pass::drop_shadow_layout_observation_for_test(
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.realizes_both_working_formats
            && observed.binds_exact_blurred_source_alpha
            && observed.binds_only_one_linear_transparent_sampler
            && observed.binds_spatial_and_parameters
            && observed.targets_only_the_working_format
            && observed.contains_no_dummy_binding,
        "the drop-shadow colorize layout has a missing or dummy binding"
    );
}

#[test]
fn drop_shadow_cache_realizes_checked_colorize_and_merge_programs() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test("checked drop-shadow realization requires backend selection")
        .unwrap_or_panic_for_test("checked drop-shadow realization requires a host adapter");
    let ready = backend
        .ready_device_state_borrow_for_test(identity)
        .unwrap_or_panic_for_test("checked drop-shadow realization requires a ready device");
    let capabilities =
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test());
    let observed = pollster::block_on(
        crate::pass::drop_shadow_cache_realization_observation_for_test(
            ready.device_for_test(),
            spatial_filter_authored_filter_steps_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
            capabilities,
        ),
    )
    .unwrap_or_panic_for_test("the checked drop-shadow programs must reach WGPU realization");

    assert!(
        observed.realizes_checked_colorize_and_merge_programs
            && observed.checked_scope_is_clean
            && observed.merge_uses_fixed_premultiplied_source_over
            && observed.merge_omits_destination_sample
            && observed.publishes_only_drop_shadow_entries,
        "the checked drop-shadow colorize or merge programs are unrealized"
    );
}

#[test]
fn prepared_spatial_filter_objects_expose_exact_encoding_handles() {
    fn require_blur_handles(objects: &crate::shader::ProvisionalBlurPassObjects<'_>) {
        let _: &wgpu::Sampler = objects.source_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    fn require_drop_shadow_handles(
        objects: &crate::shader::ProvisionalDropShadowColorizePassObjects<'_>,
    ) {
        let _: &wgpu::Sampler = objects.source_sampler();
        let _: &wgpu::BindGroupLayout = objects.bind_group_layout();
        let _: &wgpu::RenderPipeline = objects.render_pipeline();
    }

    let _ = (require_blur_handles, require_drop_shadow_handles);
}

#[test]
fn spatial_filter_graph_encodes_blur_and_drop_shadow_in_authored_order() {
    use crate::pass::SpatialFilterPassTagForTest as Tag;

    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test(
        "the spatial-filter fixture must reach its shared GPU graph executor",
    );

    assert!(
        observed.pass_order
            == [
                Tag::Color,
                Tag::BlurHorizontalRgba,
                Tag::BlurVerticalRgba,
                Tag::BlurHorizontalSourceAlpha,
                Tag::BlurVerticalSourceAlpha,
                Tag::DropShadowColorize,
                Tag::DropShadowMerge,
                Tag::Color,
            ]
            && observed.each_pass_advances_once
            && observed.binds_exact_prepared_resources
            && observed.uses_signed_viewport_and_scissor
            && observed.one_graph_command_encoder
            && observed.transaction_committed,
        "the scheduler has no ordered spatial-filter encoding route"
    );
}

#[test]
fn blur_passes_use_distinct_source_intermediate_and_result() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test("the blur fixture must reach its shared GPU graph executor");
    assert!(
        observed.blur_pass_count == 4
            && observed.blur_sources_intermediates_and_results_are_distinct
            && observed.binds_exact_prepared_resources
            && observed.kernels_release_at_validated_last_use
            && observed.textures_release_at_validated_last_use,
        "the blur scheduler has no exact pass receipts"
    );
}

#[test]
fn drop_shadow_reads_source_twice_and_releases_after_merge() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(backend.spatial_filter_graph_encoding_observation_for_test(
        identity,
        spatial_filter_authored_filter_steps_for_test(),
        filter_graph_commands_for_test(),
        filter_graph_context_for_test(),
    ))
    .unwrap_or_panic_for_test("the drop-shadow fixture must reach its shared GPU graph executor");

    assert!(
        observed.drop_shadow_colorize_count == 1
            && observed.drop_shadow_merge_count == 1
            && observed.drop_shadow_reads_original_source_twice
            && observed.original_source_releases_after_merge
            && observed.textures_release_at_validated_last_use
            && observed.each_pass_advances_once,
        "the drop-shadow scheduler lost its exact lease transition"
    );
}

#[test]
fn color_filter_graph_encodes_fused_operations_in_authored_order() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.ordered_color_filter_graph_encoding_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test("the color-filter fixture must reach its shared GPU graph executor");

    assert!(
        observed.color_pass_count == 2
            && observed.fused_runs_preserve_authored_order
            && observed.binds_exact_source_spatial_and_operations
            && observed.uses_validated_viewport_and_scissor
            && observed.releases_every_resource_at_last_use,
        "the scheduler has no ordered GPU color-filter pass"
    );
}

#[test]
fn color_filter_pass_uses_distinct_source_and_result() {
    let (mut backend, identity) = graph_encoding_backend_for_test();
    let observed = pollster::block_on(
        backend.ordered_color_filter_graph_encoding_observation_for_test(
            identity,
            authored_color_filter_runs_for_test(),
            filter_graph_commands_for_test(),
            filter_graph_context_for_test(),
        ),
    )
    .unwrap_or_panic_for_test(
        "the color-filter alias fixture must reach its shared GPU graph executor",
    );
    assert!(
        observed.color_pass_count == 2
            && observed.source_and_result_are_distinct
            && observed.binds_exact_source_spatial_and_operations
            && observed.releases_every_resource_at_last_use,
        "the color-filter pass aliases source and result"
    );
}

fn expected_composite_parameter_bytes_for_test() -> [u8; 112] {
    fn write_f32(bytes: &mut [u8; 112], offset: usize, value: f32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn write_u32(bytes: &mut [u8; 112], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    let mut bytes = [0_u8; 112];
    for (offset, value) in [0.48_f32, -0.16, 0.08, 0.64].into_iter().enumerate() {
        write_f32(&mut bytes, offset * 4, value);
    }
    write_f32(&mut bytes, 16, -1.12);
    write_f32(&mut bytes, 20, 3.04);
    for (offset, value) in [-2.5_f32, 1.25, 7.5, 3.75].into_iter().enumerate() {
        write_f32(&mut bytes, 32 + offset * 4, value);
    }
    write_u32(&mut bytes, 48, 3);
    write_u32(&mut bytes, 52, 2);
    for (offset, value) in [1.0_f32 / 6.0, 0.25, 1.0 / 3.0, 0.5]
        .into_iter()
        .enumerate()
    {
        write_f32(&mut bytes, 64 + offset * 4, value);
    }
    write_f32(&mut bytes, 80, 1.0);
    write_u32(&mut bytes, 84, 3);
    write_u32(&mut bytes, 88, 2);
    write_u32(&mut bytes, 92, 2);
    write_u32(&mut bytes, 96, 1);
    write_u32(&mut bytes, 100, 1);
    bytes
}

fn composition_single_mask_composition_commands_for_test(
    image: Image,
    bounds: Rect,
    transform: Transform,
    opacity: f32,
    blend: BlendMode,
    with_clip: bool,
) -> command::RenderCommands {
    let mut layer = Layer::new()
        .try_transform(transform)
        .unwrap()
        .try_opacity(opacity)
        .unwrap()
        .blend(blend)
        .with_resolved_alpha_mask(ResolvedLayerAlphaMask::try_new(image, bounds).unwrap());
    if with_clip {
        layer = layer
            .try_clip(Shape::rect(Rect::new(-3.0, -2.0, 12.0, 9.0)))
            .unwrap();
    }
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(-1.0, 0.5, 6.0, 3.0), Color::BLACK);
    });
    scene.normalize(Capabilities::CURRENT).unwrap()
}

#[test]
fn composite_parameter_bytes_preserve_affine_mask_mapping_quality_and_extend() {
    let image = composition_mask_image_for_test(
        PhysicalSize::new(3, 2),
        41,
        ImageQuality::High,
        Extend::Reflect,
    );
    let commands = composition_single_mask_composition_commands_for_test(
        image,
        Rect::new(-2.5, 1.25, 7.5, 3.75),
        Transform::try_new([2.0, 0.5, -0.25, 1.5, 3.0, -4.0]).unwrap(),
        1.25,
        BlendMode::Overlay,
        true,
    );
    let observed = crate::pass::composite_parameter_bytes_for_test(
        commands,
        composition_frame_context_for_test(),
    );

    assert!(
        observed == Some(expected_composite_parameter_bytes_for_test()),
        "composite bytes lost typed mask mapping or sampling"
    );
}

#[test]
fn mask_pipeline_keys_exclude_image_identity() {
    let first = composition_single_mask_composition_commands_for_test(
        composition_mask_image_for_test(
            PhysicalSize::new(4, 3),
            13,
            ImageQuality::Medium,
            Extend::Repeat,
        ),
        Rect::new(-1.0, 2.0, 8.0, 6.0),
        Transform::translation(2.0, -3.0).unwrap(),
        0.75,
        BlendMode::Screen,
        false,
    );
    let second = composition_single_mask_composition_commands_for_test(
        composition_mask_image_for_test(
            PhysicalSize::new(4, 3),
            29,
            ImageQuality::Medium,
            Extend::Repeat,
        ),
        Rect::new(-1.0, 2.0, 8.0, 6.0),
        Transform::translation(2.0, -3.0).unwrap(),
        0.75,
        BlendMode::Screen,
        false,
    );

    assert!(
        crate::pass::mask_pipeline_keys_exclude_image_identity_for_test(
            first,
            second,
            composition_frame_context_for_test(),
        ),
        "pipeline caching is keyed by retained image identity"
    );
}

#[test]
fn pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod() {
    let source_extent = PhysicalSize::new(0x0102_0304, 0x0a0b_0c0d);
    let destination_extent = PhysicalSize::new(0x1020_3040, 0x5060_7080);
    let serialized = pass_spatial_uniform_bytes_for_test(
        Point::new(1.5, -2.25),
        2.5,
        source_extent,
        Point::new(-4.5, 8.25),
        0.5,
        destination_extent,
    );
    let expected = [
        0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x90, 0xc0, 0x00, 0x00, 0x04, 0x41, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x03, 0x02, 0x01, 0x0d, 0x0c, 0x0b, 0x0a, 0x40, 0x30, 0x20, 0x10, 0x80,
        0x70, 0x60, 0x50,
    ];
    let exact_layout = serialized.as_ref().is_ok_and(|bytes| {
        bytes == &expected && bytes[12..16] == [0; 4] && bytes[28..32] == [0; 4]
    });

    let finite_overflow_cases = [
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(f64::MAX, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source origin x",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, f64::MAX),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source origin y",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                f64::MAX,
                source_extent,
                Point::new(0.0, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial source raster scale",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(f64::MAX, 0.0),
                1.0,
                destination_extent,
            ),
            "pass spatial destination origin x",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, f64::MAX),
                1.0,
                destination_extent,
            ),
            "pass spatial destination origin y",
        ),
        (
            pass_spatial_uniform_bytes_for_test(
                Point::new(0.0, 0.0),
                1.0,
                source_extent,
                Point::new(0.0, 0.0),
                f64::MAX,
                destination_extent,
            ),
            "pass spatial destination raster scale",
        ),
    ];
    let finite_overflow_is_typed = finite_overflow_cases.into_iter().all(|(result, field)| {
        result.as_ref().is_err_and(|error| {
            error.code() == ErrorCode::InvalidInput
                && error.invalid_value_diagnostic().map(InvalidValue::field) == Some(field)
        })
    });

    assert!(
        exact_layout && finite_overflow_is_typed,
        "pass spatial serialization has no explicit 48-byte contract"
    );
}

#[test]
fn pass_spatial_uniform_rejects_f32_underflowing_raster_scales() {
    let underflowing_scale = f64::from_bits(1);
    assert!(underflowing_scale.is_finite() && underflowing_scale > 0.0);
    assert_eq!(underflowing_scale as f32, 0.0);

    let source_error = pass_spatial_uniform_bytes_for_test(
        Point::new(0.0, 0.0),
        underflowing_scale,
        PhysicalSize::new(1, 1),
        Point::new(0.0, 0.0),
        1.0,
        PhysicalSize::new(1, 1),
    );
    let destination_error = pass_spatial_uniform_bytes_for_test(
        Point::new(0.0, 0.0),
        1.0,
        PhysicalSize::new(1, 1),
        Point::new(0.0, 0.0),
        underflowing_scale,
        PhysicalSize::new(1, 1),
    );
    let rejects_scale = |result: &Result<[u8; 48]>, field| {
        result.as_ref().is_err_and(|error| {
            error.code() == ErrorCode::InvalidInput
                && error.invalid_value_diagnostic().map(InvalidValue::field) == Some(field)
        })
    };

    assert!(
        rejects_scale(&source_error, "pass spatial source raster scale")
            && rejects_scale(&destination_error, "pass spatial destination raster scale"),
        "positive f64 raster scale narrowed to zero"
    );
}

#[test]
fn device_pass_cache_starts_with_no_realized_entries() {
    assert!(device_pass_cache_owns_exact_key_spaces_for_test());
}

#[test]
fn composition_graph_encodes_clip_mask_opacity_and_blend_in_authored_order() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_commands_for_test(),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "the composition graph must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.encodes_clip_mask_opacity_and_blend_in_authored_order,
        "layer composition has no one-shot GPU encoding"
    );
}

#[test]
fn normal_composition_uses_fixed_premultiplied_blend_without_parent_sampling() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_shader_composite_commands_for_test(BlendMode::Normal, true, true),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "normal composition must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.normal_uses_fixed_premultiplied_blend && observed.normal_omits_parent_sample,
        "normal composition sampled its parent or used wrong factors"
    );
}

#[test]
fn non_normal_blends_copy_parent_and_never_read_write_one_texture() {
    let (mut backend, identity, _) = composition_selected_backend_and_requests_for_test();
    let observed = match pollster::block_on(
        backend.composition_ordered_graph_encoding_observation_for_test(
            identity,
            composition_shader_composite_commands_for_test(BlendMode::Multiply, true, true),
            composition_frame_context_for_test(),
        ),
    ) {
        Ok(observed) => observed,
        Err(error) => panic!(
            "destination sampling must reach its checked one-shot encoding observation: {error:?}"
        ),
    };

    assert!(
        observed.destination_copies_full_parent && observed.destination_avoids_read_write_alias,
        "destination sampling aliases its output"
    );
}

#[test]
fn base_graph_shader_cache_realizes_checked_programs_without_publishing_failed_entries() {
    let mut backend = Backend::new(ResourceCacheBudget::DISABLED);
    let identity = pollster::block_on(backend.select_device(None))
        .unwrap_or_panic_for_test(
            "checked base-graph shader realization requires backend selection",
        )
        .unwrap_or_panic_for_test("checked base-graph shader realization requires a host adapter");
    let capabilities = {
        let ready = backend
            .ready_device_state_borrow_for_test(identity)
            .unwrap_or_panic_for_test(
                "checked base-graph shader realization requires a ready device",
            );
        DeviceCapabilities::from_device(ready.adapter_for_test(), ready.device_for_test())
    };
    let working_format = capabilities
        .resolve_effect_working_format(EffectQualityPolicy::AllowReducedPrecision)
        .unwrap_or_panic_for_test(
            "checked base-graph shader realization requires one supported working format",
        );
    let commands = graph_shader_commands_for_test();
    let context = graph_shader_frame_context_for_test();
    let rgba_requests = crate::pass::core_pass_cache_requests_for_test(
        commands.clone(),
        context,
        capabilities,
        working_format,
        Format::Rgba8,
    )
    .unwrap_or_panic_for_test(
        "RGBA shader realization requires exact lowered base-graph pass keys",
    );
    let bgra_requests = crate::pass::core_pass_cache_requests_for_test(
        commands,
        context,
        capabilities,
        working_format,
        Format::Bgra8,
    )
    .unwrap_or_panic_for_test(
        "BGRA shader realization requires exact lowered base-graph pass keys",
    );
    let observed = pollster::block_on(
        backend.core_pass_shader_cache_realization_observation_for_test(
            identity,
            &rgba_requests,
            &bgra_requests,
        ),
    )
    .unwrap_or_panic_for_test(
        "checked base-graph shader realization must reach its transaction observation",
    );

    assert!(
        observed.realizes_all_checked_programs
            && observed.provisional_handles_are_encoding_ready
            && observed.commits_only_after_clean_transaction
            && observed.reuses_exact_committed_entries
            && observed.failed_validation_publishes_none
            && observed.cancellation_publishes_none
            && observed.device_transition_publishes_none
            && observed.specializes_rgba_and_bgra_outputs,
        "base-graph pass objects are not transactionally cached"
    );
}

#[test]
fn composite_cache_realizes_exact_normal_and_destination_sampling_programs() {
    let (mut backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let observed = pollster::block_on(
        backend.layer_composite_cache_realization_observation_for_test(identity, &requests),
    )
    .unwrap();

    assert!(
        observed.realizes_normal_and_destination_programs
            && observed.realizes_all_optional_binding_combinations
            && observed.normal_uses_fixed_premultiplied_source_over
            && observed.destination_uses_replace_blending
            && observed.commits_only_after_clean_transaction
            && observed.reuses_exact_committed_entries
            && observed.failed_validation_publishes_none
            && observed.cancellation_publishes_none
            && observed.device_transition_publishes_none,
        "the compositor has no checked pipeline realization"
    );
}

#[test]
fn composite_layouts_bind_no_dummy_parent_clip_or_mask() {
    let requests = composition_composite_requests_for_test(
        DeviceCapabilities::from_test_facts(true, true, 4_096),
        WorkingFormat::HighPrecision,
    );
    let observed = crate::pass::layer_composite_layout_observation_for_test(&requests);

    assert!(
        observed.realizes_all_eight_entry_interfaces
            && observed.normal_omits_parent
            && observed.destination_binds_parent
            && observed.optional_clip_is_exact
            && observed.optional_mask_is_exact
            && observed.binds_only_one_source_sampler
            && observed.binds_only_exact_uniforms,
        "composite layout contains an absent semantic binding"
    );
}

#[test]
fn base_graph_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms() {
    let observed = crate::pass::core_pass_layout_observation_for_test(
        graph_shader_commands_for_test(),
        graph_shader_frame_context_for_test(),
        DeviceCapabilities::from_test_facts(true, true, 4_096),
    );

    assert!(
        observed.canonicalize_binds_capture_and_spatial_only
            && observed.span_source_over_binds_source_and_spatial_only
            && observed.present_binds_final_image_and_spatial_only
            && observed.copy_only_parent_is_not_sampled
            && observed.dummy_parameters_are_not_bound
            && observed.composition_typed_vocabulary_is_preserved
            && observed.output_specialization_is_exact,
        "the base-graph pass layout contains a copy-only or dummy binding"
    );
}

#[test]
fn offscreen_pipeline_capability_accessors_report_supported_operations() {
    let capabilities = Capabilities::CURRENT.offscreen_pipeline();

    assert!(capabilities.supports_direct_vello_opacity_isolation());
    assert!(capabilities.supports_direct_vello_blend_isolation());
    assert!(!capabilities.supports_offscreen_layer_rendering());
    assert!(capabilities.supports_persistent_effect_resources());
    assert!(capabilities.supports_bounded_vello_capture());
    assert!(capabilities.supports_image_pass_execution());
    assert!(capabilities.supports_composite_pass_execution());
    assert!(capabilities.supports_nested_opacity_composition());
    assert!(!capabilities.supports_mask_execution());
    assert!(!capabilities.supports_layer_filter_execution());
    assert!(!capabilities.supports_broad_backdrop_execution());
}

#[test]
fn backdrop_capability_accessors_claim_only_narrow_materialized_execution() {
    let capabilities = Capabilities::CURRENT.offscreen_pipeline();

    assert!(capabilities.supports_bounded_backdrop_capture());
    assert!(capabilities.supports_bounded_backdrop_filter_execution());
    assert!(!capabilities.supports_backdrop_isolation_composition());
    assert!(!capabilities.supports_broad_backdrop_execution());
}

#[test]
fn affected_capability_queries_map_one_to_one_to_primitive_operations() {
    let capabilities = Capabilities::CURRENT;
    let offscreen = capabilities.offscreen_pipeline();
    let cases = [
        (
            capabilities.filters().supports_gpu_color_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ),
        (
            capabilities.filters().supports_gpu_blur_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ),
        (
            capabilities
                .filters()
                .supports_gpu_drop_shadow_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ),
        (
            capabilities
                .masks_clips()
                .supports_resolved_alpha_mask_execution(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::ResolvedAlphaMaskExecution,
        ),
        (
            offscreen.supports_image_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::ImagePassExecution,
        ),
        (
            offscreen.supports_composite_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::CompositePassExecution,
        ),
        (
            offscreen.supports_nested_opacity_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::NestedOpacityComposition,
        ),
    ];

    for (query, family, operation) in cases {
        assert_eq!(
            capabilities
                .ensure_supported(UnsupportedPrimitive::new(family, operation))
                .is_ok(),
            query,
            "capability query must map one-to-one to {operation:?}",
        );
    }
}

#[test]
fn offscreen_pipeline_capability_diagnostics_report_unsupported_operations() {
    for operation in [
        PrimitiveOperation::OffscreenLayerRendering,
        PrimitiveOperation::MaskExecution,
        PrimitiveOperation::LayerFilterExecution,
        PrimitiveOperation::BroadBackdropExecution,
        PrimitiveOperation::BackdropIsolationComposition,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::OffscreenPipeline, operation);
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("offscreen pipeline operation is not implemented in this phase");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("offscreen pipeline"));
        assert!(error.message().contains(unsupported.label()));
    }
    Capabilities::CURRENT
        .ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropFilterExecution,
        ))
        .expect(
            "bounded backdrop filter execution is the narrow implemented bounded-backdrop subset",
        );
}

#[test]
fn graph_render_submits_one_transaction_and_publishes_once() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_panic_for_test("graph submission coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .unwrap_or_panic_for_test("graph submission coverage requires a headless surface");
    let publication_before = surface.headless_publication_count_for_test();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .unwrap_or_panic_for_test("the forced graph route must invoke the production graph executor");

    let production_graph_transaction = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before)
        == 1
        && graph.output_extent == PhysicalSize::new(2, 2)
        && graph.stats.route == Some(RenderRoute::GpuGraph)
        && graph.stats == renderer.stats();
    assert!(
        production_graph_transaction,
        "the production graph did not commit one transaction and publication"
    );
}

#[test]
fn capabilities_report_supported_gpu_operations_and_broad_mask_diagnostic() {
    let capabilities = Capabilities::CURRENT;
    let filters = capabilities.filters();
    let masks = capabilities.masks_clips();
    let offscreen = capabilities.offscreen_pipeline();
    let supported = composition_supported_capability_rows(filters, masks, offscreen);
    let rejected = composition_rejected_capability_rows(filters, masks, offscreen);
    let supported_are_exact = supported.into_iter().all(|(query, family, operation)| {
        query
            && capabilities
                .ensure_supported(UnsupportedPrimitive::new(family, operation))
                .is_ok()
    });
    let rejected_are_exact = rejected.into_iter().all(|(query, family, operation)| {
        let expected = UnsupportedPrimitive::new(family, operation);
        !query
            && capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
    });
    let diagnostic_only_rejections = [UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::AlphaMaskSourceExecution,
    )];
    let diagnostic_only_rejections_are_exact =
        diagnostic_only_rejections.into_iter().all(|expected| {
            capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
        });
    assert!(
        supported_are_exact && rejected_are_exact && diagnostic_only_rejections_are_exact,
        "capability surface overclaims broad support"
    );
}

type CapabilityRowForTest = (bool, PrimitiveFamily, PrimitiveOperation);

fn composition_supported_capability_rows(
    filters: FilterCapabilities,
    masks: MaskClipCapabilities,
    offscreen: OffscreenPipelineCapabilities,
) -> [CapabilityRowForTest; 13] {
    [
        (
            filters.supports_gpu_color_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuColorFilterExecution,
        ),
        (
            filters.supports_gpu_blur_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuBlurFilterExecution,
        ),
        (
            filters.supports_gpu_drop_shadow_filter_execution(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::GpuDropShadowFilterExecution,
        ),
        (
            masks.supports_resolved_alpha_mask_execution(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::ResolvedAlphaMaskExecution,
        ),
        (
            offscreen.supports_image_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::ImagePassExecution,
        ),
        (
            offscreen.supports_composite_pass_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::CompositePassExecution,
        ),
        (
            offscreen.supports_nested_opacity_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::NestedOpacityComposition,
        ),
        (
            filters.supports_ordered_filter_lists(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::OrderedFilterList,
        ),
        (
            filters.supports_filter_region_planning(),
            PrimitiveFamily::Filters,
            PrimitiveOperation::FilterRegionPlanning,
        ),
        (
            offscreen.supports_persistent_effect_resources(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::PersistentEffectResources,
        ),
        (
            offscreen.supports_bounded_vello_capture(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedVelloCapture,
        ),
        (
            offscreen.supports_bounded_backdrop_capture(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropCapture,
        ),
        (
            offscreen.supports_bounded_backdrop_filter_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropFilterExecution,
        ),
    ]
}

#[test]
fn resolved_alpha_mask_preserves_partial_alpha_and_nested_order() {
    let size = PhysicalSize::new(4, 2);
    let bounds = Rect::new(0.0, 0.0, 4.0, 2.0);
    let inner_mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[128],
        ImageQuality::Low,
        Extend::Pad,
    );
    let outer_mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[160],
        ImageQuality::Low,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(outer_mask.clone(), bounds).unwrap(),
        ),
        |scene| {
            scene.fill(
                bounds,
                color_from_straight_rgba8_for_test([32, 64, 224, 255]),
            );
            scene.layer(
                Layer::new().with_resolved_alpha_mask(
                    ResolvedLayerAlphaMask::try_new(inner_mask.clone(), bounds).unwrap(),
                ),
                |scene| {
                    scene.fill(
                        bounds,
                        color_from_straight_rgba8_for_test([240, 48, 16, 255]),
                    );
                },
            );
        },
    );

    let inner = reference_solid_for_test(size, [240, 48, 16, 255])
        .apply_resolved_alpha_mask(bounds, &inner_mask, bounds)
        .unwrap();
    let parent = reference_solid_for_test(size, [32, 64, 224, 255]);
    let expected = inner
        .source_over(&parent)
        .unwrap()
        .apply_resolved_alpha_mask(bounds, &outer_mask, bounds)
        .unwrap();
    let expected = reference_straight_bytes_for_test(&expected);

    let preserves_nested_order = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(4.0, 2.0),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    2,
                )
        });

    assert!(
        preserves_nested_order,
        "resolved masks do not compose inner to outer on GPU"
    );
}

fn composition_mask_quality_scene_and_oracle_for_test() -> (Scene, PhysicalSize, Vec<u8>) {
    let qualities = [ImageQuality::Low, ImageQuality::Medium, ImageQuality::High];
    let extends = [Extend::Pad, Extend::Repeat, Extend::Reflect];
    let tile_size = PhysicalSize::new(12, 8);
    let mut expected = Vec::with_capacity(12 * 8 * qualities.len() * extends.len() * 4);
    let mut scene = Scene::new();
    let mut case_index = 0_u32;
    for quality in qualities {
        for extend in extends {
            let mask = composition_mask_image_from_alpha_for_test(
                PhysicalSize::new(2, 2),
                &[16, 96, 160, 240],
                quality,
                extend,
            );
            let mask_bounds = Rect::new(1.0, 1.0, 4.0, 2.0);
            let source_bounds = Rect::new(0.0, 0.0, 6.0, 4.0);
            let transform = Transform::try_new([
                2.0,
                0.0,
                0.0,
                2.0,
                0.0,
                f64::from(case_index * tile_size.height()),
            ])
            .unwrap();
            scene.layer(
                Layer::new()
                    .try_transform(transform)
                    .unwrap()
                    .with_resolved_alpha_mask(
                        ResolvedLayerAlphaMask::try_new(mask.clone(), mask_bounds).unwrap(),
                    ),
                |scene| {
                    scene.fill(
                        source_bounds,
                        color_from_straight_rgba8_for_test([255, 255, 255, 255]),
                    );
                },
            );
            let reference = reference_solid_for_test(tile_size, [255, 255, 255, 255])
                .apply_resolved_alpha_mask(source_bounds, &mask, mask_bounds)
                .unwrap();
            expected.extend(reference_straight_bytes_for_test(&reference));
            case_index += 1;
        }
    }
    (
        scene,
        PhysicalSize::new(tile_size.width(), case_index * tile_size.height()),
        expected,
    )
}

#[test]
fn resolved_alpha_mask_low_medium_high_and_extend_modes_match_boundary_oracle() {
    let (scene, size, expected) = composition_mask_quality_scene_and_oracle_for_test();
    let matches_boundary_oracle = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(f64::from(size.width()), f64::from(size.height())),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    2,
                )
        });

    assert!(
        matches_boundary_oracle,
        "GPU mask quality or edge sampling exceeds the GPU edge tolerance"
    );
}

fn composition_blend_scene_and_oracle_for_test() -> (Scene, PhysicalSize, Vec<u8>) {
    let modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    let source = [204, 64, 153, 160];
    let opaque_destination = [51, 179, 102, 255];
    let tile_width = 2_u32;
    let tile_height = 2_u32;
    let size = PhysicalSize::new(
        tile_width * u32::try_from(modes.len()).unwrap(),
        tile_height * 2,
    );
    let mut expected = vec![0_u8; usize::try_from(size.width() * size.height() * 4).unwrap()];
    let mut scene = Scene::new();
    for (base_index, destination) in [[0, 0, 0, 0], opaque_destination].into_iter().enumerate() {
        for (mode_index, mode) in modes.into_iter().enumerate() {
            let x = f64::from(u32::try_from(mode_index).unwrap() * tile_width);
            let y = f64::from(u32::try_from(base_index).unwrap() * tile_height);
            let rect = Rect::new(x, y, f64::from(tile_width), f64::from(tile_height));
            if destination[3] != 0 {
                scene.fill(rect, color_from_straight_rgba8_for_test(destination));
            }
            let mask = composition_mask_image_from_alpha_for_test(
                PhysicalSize::new(1, 1),
                &[255],
                ImageQuality::Low,
                Extend::Pad,
            );
            scene.layer(
                Layer::new()
                    .blend(mode)
                    .with_resolved_alpha_mask(ResolvedLayerAlphaMask::try_new(mask, rect).unwrap()),
                |scene| {
                    scene.fill(rect, color_from_straight_rgba8_for_test(source));
                },
            );
            let expected_pixel = reference_premultiplied_pixel_for_test(source)
                .blend_over(reference_premultiplied_pixel_for_test(destination), mode);
            let expected_pixel = reference_straight_bytes_for_test(
                &ReferencePremultipliedRgba8Buffer::from_pixels(
                    PhysicalSize::new(1, 1),
                    vec![expected_pixel],
                )
                .unwrap(),
            );
            for pixel_y in 0..tile_height {
                for pixel_x in 0..tile_width {
                    let output_x = u32::try_from(mode_index).unwrap() * tile_width + pixel_x;
                    let output_y = u32::try_from(base_index).unwrap() * tile_height + pixel_y;
                    let offset = usize::try_from((output_y * size.width() + output_x) * 4).unwrap();
                    expected[offset..offset + 4].copy_from_slice(&expected_pixel);
                }
            }
        }
    }
    (scene, size, expected)
}

#[test]
fn all_supported_blends_match_oracle_over_transparent_and_opaque_bases() {
    let (scene, size, expected) = composition_blend_scene_and_oracle_for_test();
    let blends_match = composition_supported_working_formats_for_test()
        .into_iter()
        .all(|working_format| {
            let rendered = render_composition_headless_for_test(
                &scene,
                Size::new(f64::from(size.width()), f64::from(size.height())),
                Parameters::default(),
                working_format,
            );
            rendered.output.size() == size
                && composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
                && graph_pixels_match_for_test(
                    rendered.output.rgba(),
                    &expected,
                    rendered.working_format,
                    3,
                )
        });

    assert!(
        blends_match,
        "GPU blend output exceeds the GPU pixel tolerance"
    );
}

#[test]
fn plus_blend_clamps_high_precision_results() {
    let source = [255, 128, 64, 204];
    let destination = [128, 255, 192, 204];
    let rect = Rect::new(0.0, 0.0, 4.0, 4.0);
    let mut scene = Scene::new();
    scene.fill(rect, color_from_straight_rgba8_for_test(destination));
    scene.layer(
        Layer::new()
            .blend(BlendMode::Plus)
            .with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(
                    composition_mask_image_from_alpha_for_test(
                        PhysicalSize::new(1, 1),
                        &[255],
                        ImageQuality::Low,
                        Extend::Pad,
                    ),
                    rect,
                )
                .unwrap(),
            ),
        |scene| {
            scene.fill(rect, color_from_straight_rgba8_for_test(source));
        },
    );
    let expected_pixel = reference_premultiplied_pixel_for_test(source).blend_over(
        reference_premultiplied_pixel_for_test(destination),
        BlendMode::Plus,
    );
    let expected = reference_straight_bytes_for_test(
        &ReferencePremultipliedRgba8Buffer::from_pixels(
            PhysicalSize::new(4, 4),
            vec![expected_pixel; 16],
        )
        .unwrap(),
    );
    let rendered = render_composition_headless_for_test(
        &scene,
        Size::new(4.0, 4.0),
        Parameters::default(),
        WorkingFormat::HighPrecision,
    );

    assert!(
        composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
            && graph_pixels_match_for_test(
                rendered.output.rgba(),
                &expected,
                WorkingFormat::HighPrecision,
                3,
            ),
        "Plus exceeded the unit interval"
    );
}

#[test]
fn outer_clip_precedes_mask_and_opacity_on_unfiltered_sources() {
    let surface_size = PhysicalSize::new(4, 2);
    let source_bounds = Rect::new(0.0, 0.0, 4.0, 2.0);
    let clip_bounds = Rect::new(1.0, 0.0, 2.0, 2.0);
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[128],
        ImageQuality::Low,
        Extend::Pad,
    );
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(clip_bounds))
            .unwrap()
            .try_opacity(0.5)
            .unwrap()
            .with_resolved_alpha_mask(
                ResolvedLayerAlphaMask::try_new(mask.clone(), source_bounds).unwrap(),
            ),
        |scene| {
            scene.fill(
                source_bounds,
                color_from_straight_rgba8_for_test([224, 48, 16, 255]),
            );
        },
    );
    let masked = reference_solid_for_test(surface_size, [224, 48, 16, 255])
        .apply_resolved_alpha_mask(source_bounds, &mask, source_bounds)
        .unwrap()
        .apply_opacity(0.5)
        .unwrap();
    let mut expected = reference_straight_bytes_for_test(&masked);
    for y in 0..surface_size.height() {
        for x in [0_u32, 3] {
            let offset = usize::try_from((y * surface_size.width() + x) * 4).unwrap();
            expected[offset..offset + 4].fill(0);
        }
    }
    let rendered = render_composition_headless_for_test(
        &scene,
        Size::new(4.0, 2.0),
        Parameters::default(),
        WorkingFormat::HighPrecision,
    );

    assert!(
        composition_frame_used_one_atomic_graph_submission_for_test(&rendered)
            && rendered.output.size() == surface_size
            && graph_pixels_match_for_test(
                rendered.output.rgba(),
                &expected,
                WorkingFormat::HighPrecision,
                2,
            ),
        "outer composition operations changed order"
    );
}

fn render_composition_headless_for_test(
    scene: &Scene,
    size: Size,
    parameters: Parameters,
    working_format: WorkingFormat,
) -> CompositionProductionFrameForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("masked-composition production execution requires a compatible renderer: {error}")
    });
    let mut surface =
        pollster::block_on(renderer.create_headless(size, 1.0)).unwrap_or_else(|error| {
            panic!("masked-composition production execution requires a headless surface: {error}")
        });
    let publication_before = surface.headless_publication_count_for_test();
    let stats = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        working_format,
    ))
    .unwrap_or_else(|error| {
        panic!(
            "the masked-composition fixture must reach its current production render route: {error}"
        )
    });
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap_or_else(|error| {
        panic!("the masked-composition fixture publication must be explicitly readable: {error}")
    });
    CompositionProductionFrameForTest {
        output,
        stats,
        working_format,
        publication_count,
    }
}

fn composition_frame_used_one_atomic_graph_submission_for_test(
    rendered: &CompositionProductionFrameForTest,
) -> bool {
    rendered.publication_count == 1
        && rendered.stats.route == Some(RenderRoute::GpuGraph)
        && rendered.stats.commands > 0
}

fn composition_supported_working_formats_for_test() -> Vec<WorkingFormat> {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("masked-composition format coverage requires a compatible renderer: {error}")
    });
    graph_supported_working_formats_for_test(&mut renderer)
}

#[test]
fn public_dispatch_routes_composition_and_color_filters_but_rejects_broad_backdrop() {
    let (color_filter_scene, color_filters, color_filter_expected) =
        color_filter_retention_fixture_for_test();
    let color_filter_width = u32::try_from(color_filter_expected.len() / 4)
        .expect("the color-filter width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| {
        panic!("composition and color-filter dispatch coverage requires a renderer: {error}")
    });
    let working_format = default_graph_working_format_for_test(&mut renderer);

    let mut color_filter_surface = pollster::block_on(
        renderer.create_headless(Size::new(f64::from(color_filter_width), 1.0), 1.0),
    )
    .unwrap_or_else(|error| panic!("color-filter dispatch coverage requires a surface: {error}"));
    let color_filter = render_color_filter_fixture_for_test(
        &mut renderer,
        &mut color_filter_surface,
        &color_filter_scene,
        color_filters.clone(),
        Parameters::default(),
        working_format,
    );

    let (composition_scene, composition_size, _, _) = composition_reuse_scene_and_oracle_for_test();
    let mut composition_surface = pollster::block_on(renderer.create_headless(
        Size::new(
            f64::from(composition_size.width()),
            f64::from(composition_size.height()),
        ),
        1.0,
    ))
    .unwrap_or_else(|error| panic!("masked-composition coverage requires a surface: {error}"));
    let composition = pollster::block_on(renderer.render(
        &mut composition_surface,
        &composition_scene,
        Parameters::default(),
    ));

    let unsupported_scene = color_filter_unsupported_backdrop_scene_for_test();
    let mut unsupported_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap_or_else(
            |error| panic!("broad-backdrop rejection coverage requires a surface: {error}"),
        );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("dispatch coverage must retain its ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let unsupported = pollster::block_on(renderer.render(
        &mut unsupported_surface,
        &unsupported_scene,
        Parameters::default(),
    ));
    let color_diagnostic = color_filter_public_color_graph_diagnostic_for_test(
        &color_filter_scene,
        color_filters,
        Size::new(f64::from(color_filter_width), 1.0),
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("retained public diagnostics must keep the ready device");
    let resources_after = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let expected_color = UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::GpuColorFilterExecution,
    );
    let expected_backdrop = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );

    assert!(
        color_filter.output.rgba() == color_filter_expected
            && color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                &color_filter,
                color_filter_width
            )
            && composition.is_ok()
            && unsupported
                .as_ref()
                .is_err_and(|error| error.unsupported_primitive() == Some(expected_backdrop))
            && color_diagnostic == Some(expected_color)
            && resources_after == resources_before
            && cache_after == cache_before
            && unsupported_surface.headless_publication_count_for_test() == 0,
        "public dispatch misrouted masked composition, color filters, or broad backdrop"
    );
}

#[derive(Debug)]
struct CompositionProductionFrameForTest {
    output: ImageBuffer,
    stats: Stats,
    working_format: WorkingFormat,
    publication_count: usize,
}

#[test]
fn color_filter_fixture_executes_while_public_capability_remains_diagnostic() {
    let (scene, filters, expected) = color_filter_retention_fixture_for_test();
    let width =
        u32::try_from(expected.len() / 4).expect("the color-filter ingress width must fit u32");
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .unwrap_or_else(|error| panic!("color-filter ingress coverage requires a renderer: {error}"));
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .unwrap_or_else(|error| {
                panic!("color-filter ingress coverage requires a surface: {error}")
            });
    let rendered = render_color_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &scene,
        filters.clone(),
        Parameters::default(),
        working_format,
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the executed color-filter fixture must retain its ready device");
    let resources_before_diagnostic = ready.internal_resource_manager_observation_for_test();
    let cache_before_diagnostic = ready.device_pass_cache_counts_for_test();
    let publication_before_diagnostic = surface.headless_publication_count_for_test();
    let public_diagnostic = color_filter_public_color_graph_diagnostic_for_test(
        &scene,
        filters,
        Size::new(f64::from(width), 1.0),
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the public color-filter rejection must retain its ready device");
    let resources_after_diagnostic = ready.internal_resource_manager_observation_for_test();
    let cache_after_diagnostic = ready.device_pass_cache_counts_for_test();
    let expected_diagnostic = UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::GpuColorFilterExecution,
    );

    assert!(
        color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width)
            && rendered.output.rgba() == expected
            && rendered.stats == renderer.stats()
            && public_diagnostic == Some(expected_diagnostic)
            && retained_public_filter_diagnostics_are_exact_for_test()
            && resources_after_diagnostic == resources_before_diagnostic
            && cache_after_diagnostic == cache_before_diagnostic
            && surface.headless_publication_count_for_test() == publication_before_diagnostic,
        "the color-filter fixture did not execute through retained graph ingress while the public capability remained diagnostic"
    );
}

fn render_graph_alpha_vector_for_test(
    expected: &[[u8; 4]],
    requested_working_format: Option<WorkingFormat>,
) -> GraphAlphaVectorOutputForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("alpha-vector graph execution requires a renderer");
    let working_format = requested_working_format
        .unwrap_or_else(|| default_graph_working_format_for_test(&mut renderer));
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(f64::from(width), 1.0), 1.0))
            .expect("alpha-vector graph execution requires a headless surface");
    let publication_before = surface.headless_publication_count_for_test();
    let scene = graph_alpha_extreme_scene_for_test(expected);
    let graph = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the forced graph entry must execute the production headless graph");
    let publication_count = surface
        .headless_publication_count_for_test()
        .saturating_sub(publication_before);
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the already-published graph frame must be explicitly readable");
    let used_graph_transaction = graph.working_format == working_format
        && graph.stats.route == Some(RenderRoute::GpuGraph)
        && publication_count == 1;
    GraphAlphaVectorOutputForTest {
        output,
        graph,
        used_graph_transaction,
        publication_count,
    }
}

fn graph_alpha_vector_has_exact_grid_for_test(
    graph: &super::renderer::ForcedGraphRenderResultForTest,
    width: u32,
) -> bool {
    graph.output_extent == PhysicalSize::new(width, 1)
        && matches!(
            graph.captures.as_slice(),
            [capture]
                if capture.texel_origin == Point::new(0.0, 0.0)
                    && capture.extent == PhysicalSize::new(width, 1)
                    && capture.raster_scale == 1.0
        )
}

#[test]
fn capture_canonicalize_present_round_trips_transparent_partial_and_opaque_pixels() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered = render_graph_alpha_vector_for_test(&expected, None);
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let canonical_pixels =
        rendered
            .output
            .rgba()
            .chunks_exact(4)
            .zip(&expected)
            .all(|(actual, expected)| {
                let expected = graph_canonical_pixel_for_test(*expected);
                if rendered.graph.working_format == WorkingFormat::HighPrecision {
                    actual
                        .iter()
                        .copied()
                        .zip(expected)
                        .all(|(actual, expected)| {
                            graph_channel_error_for_test(actual, expected) <= 2
                        })
                } else {
                    graph_channel_error_for_test(actual[3], expected[3]) <= 1
                        && (0..3).all(|channel| {
                            graph_channel_error_for_test(
                                premultiply_u8_channel_for_test(actual[channel], actual[3]),
                                premultiply_u8_channel_for_test(expected[channel], expected[3]),
                            ) <= 1
                        })
                }
            });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && exact_extent_and_origin
            && canonical_pixels,
        "headless graph pixels do not satisfy canonical output"
    );
}

#[test]
fn reduced_precision_low_alpha_pixels_use_alpha_and_premul8_tolerances() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered =
        render_graph_alpha_vector_for_test(&expected, Some(WorkingFormat::ReducedPrecision));
    let width = u32::try_from(expected.len()).expect("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let reduced_pixels =
        rendered
            .output
            .rgba()
            .chunks_exact(4)
            .zip(&expected)
            .all(|(actual, expected)| {
                let expected = graph_canonical_pixel_for_test(*expected);
                graph_channel_error_for_test(actual[3], expected[3]) <= 1
                    && (0..3).all(|channel| {
                        graph_channel_error_for_test(
                            premultiply_u8_channel_for_test(actual[channel], actual[3]),
                            premultiply_u8_channel_for_test(expected[channel], expected[3]),
                        ) <= 1
                    })
            });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && rendered.graph.working_format == WorkingFormat::ReducedPrecision
            && exact_extent_and_origin
            && reduced_pixels,
        "reduced-precision graph output violates alpha or premul8 tolerance"
    );
}

#[test]
fn high_precision_low_alpha_pixels_preserve_straight_rgb() {
    let expected = graph_alpha_extreme_pixels_for_test();
    let rendered =
        render_graph_alpha_vector_for_test(&expected, Some(WorkingFormat::HighPrecision));
    let width = u32::try_from(expected.len())
        .unwrap_or_panic_for_test("the graph pixel vector must fit in u32");
    let exact_extent_and_origin = rendered.output.size() == PhysicalSize::new(width, 1)
        && graph_alpha_vector_has_exact_grid_for_test(&rendered.graph, width);
    let stable_straight_rgb = rendered
        .output
        .rgba()
        .chunks_exact(4)
        .zip(&expected)
        .filter(|(_, expected)| expected[3] > 0 && expected[3] <= 16)
        .all(|(actual, expected)| {
            graph_channel_error_for_test(actual[3], expected[3]) <= 2
                && (0..3).all(|channel| {
                    graph_channel_error_for_test(actual[channel], expected[channel]) <= 2
                })
        });

    assert!(
        rendered.used_graph_transaction
            && rendered.publication_count == 1
            && rendered.graph.working_format == WorkingFormat::HighPrecision
            && exact_extent_and_origin
            && stable_straight_rgb,
        "high-precision graph output lost stable straight RGB"
    );
}

fn color_filter_operation_matrix_for_test() -> [(&'static str, ColorFilterOp); 8] {
    [
        (
            "brightness",
            ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
        ),
        (
            "contrast",
            ColorFilterOp::Contrast(FilterAmount::try_new(1.75).unwrap()),
        ),
        (
            "grayscale",
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(1.0).unwrap()),
        ),
        (
            "hue-rotate",
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
        ),
        (
            "invert",
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.75).unwrap()),
        ),
        (
            "opacity",
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ),
        (
            "saturate",
            ColorFilterOp::Saturate(FilterAmount::try_new(2.0).unwrap()),
        ),
        (
            "sepia",
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(1.0).unwrap()),
        ),
    ]
}

fn color_filter_reference_straight_pixels_for_test(
    source_pixels: &[[u8; 4]],
    operations: &[ColorFilterOp],
    working_format: WorkingFormat,
) -> Vec<u8> {
    let size = PhysicalSize::new(
        u32::try_from(source_pixels.len()).expect("the color-filter oracle width must fit u32"),
        1,
    );
    let premultiplied_source = ReferencePremultipliedRgba8Buffer::from_pixels(
        size,
        source_pixels
            .iter()
            .copied()
            .map(reference_premultiplied_pixel_for_test)
            .collect(),
    )
    .expect("the color-filter source pixels must form one premultiplied oracle buffer");
    let filter = FilterList::try_ops(
        operations
            .iter()
            .copied()
            .map(|operation| match operation {
                ColorFilterOp::Brightness(amount) => FilterOp::brightness(amount),
                ColorFilterOp::Contrast(amount) => FilterOp::contrast(amount),
                ColorFilterOp::Grayscale(amount) => FilterOp::grayscale(amount),
                ColorFilterOp::HueRotate(angle) => FilterOp::hue_rotate(angle),
                ColorFilterOp::Invert(amount) => FilterOp::invert(amount),
                ColorFilterOp::Opacity(amount) => FilterOp::opacity(amount),
                ColorFilterOp::Saturate(amount) => FilterOp::saturate(amount),
                ColorFilterOp::Sepia(amount) => FilterOp::sepia(amount),
            })
            .collect(),
    )
    .expect("the color-filter oracle operations must form one authored filter");
    let pipeline = filter
        .color_filter_pipeline()
        .expect("the color-filter oracle fixture contains only color functions")
        .expect("the color-filter oracle fixture contains one nonempty color run");
    match working_format {
        WorkingFormat::HighPrecision => {
            super::reference::apply_color_filter_pipeline_to_straight_rgba8(
                source_pixels,
                &pipeline,
            )
        }
        WorkingFormat::ReducedPrecision => {
            let filtered = premultiplied_source
                .apply_color_filter_pipeline(&pipeline)
                .expect("the color-filter CPU oracle must evaluate every authored operation");
            reference_straight_bytes_for_test(&filtered)
        }
    }
}

fn color_filter_frame_has_exact_extent_origin_and_submission_for_test(
    rendered: &ColorFilterProductionFrameForTest,
    visible_width: u32,
) -> bool {
    rendered.output.size() == PhysicalSize::new(visible_width, 1)
        && rendered.output_extent == PhysicalSize::new(visible_width, 1)
        && rendered.source_origin == Some((COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X, 0))
        && rendered.source_extent
            == Some(PhysicalSize::new(
                visible_width + COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X.unsigned_abs(),
                1,
            ))
        && rendered.source_texel_origin
            == Some(Point::new(
                f64::from(COLOR_FILTER_PIXEL_FIXTURE_SIGNED_X),
                0.0,
            ))
        && rendered.source_raster_scale == Some(1.0)
        && rendered.stats.route == Some(RenderRoute::GpuGraph)
        && rendered.publication_count == 1
        && rendered.stats.commands > 0
}

fn terminal_straight_rgba8_is_canonical_for_test(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(4)
        && bytes
            .chunks_exact(4)
            .all(|pixel| pixel[3] != 0 || pixel[..3] == [0, 0, 0])
}

#[test]
fn terminal_straight_rgba8_rejects_rgb_leakage_at_zero_alpha() {
    let accepted = [
        0, 0, 0, 0, 255, 0, 128, 1, 17, 31, 47, 127, 255, 255, 255, 255,
    ];

    assert!(
        terminal_straight_rgba8_is_canonical_for_test(&accepted),
        "terminal straight RGBA8 rejected canonical transparent black or a nonzero-alpha pixel"
    );
    assert!(
        !terminal_straight_rgba8_is_canonical_for_test(&[0, 0, 0]),
        "terminal straight RGBA8 accepted a byte-misaligned sample"
    );
    assert!(
        !terminal_straight_rgba8_is_canonical_for_test(&[128, 0, 0, 0]),
        "terminal straight RGBA8 accepted RGB leakage at zero alpha"
    );
}

#[test]
fn high_precision_color_functions_match_cpu_oracle_for_boundary_pixels() {
    let source = color_filter_boundary_pixels_for_test();
    let width =
        u32::try_from(source.len()).expect("the high-precision color matrix width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let (mut renderer, mut surface) =
        color_filter_pixel_renderer_for_test(WorkingFormat::HighPrecision, width);
    let mut maximum_error = 0;
    let mut exact_execution = true;

    for (name, operation) in color_filter_operation_matrix_for_test() {
        let expected = color_filter_reference_straight_pixels_for_test(
            &source,
            &[operation],
            WorkingFormat::HighPrecision,
        );
        let rendered = render_color_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            vec![color_filter_list([operation])],
            Parameters::default(),
            WorkingFormat::HighPrecision,
        );
        let exact_grid =
            color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width);
        let terminal_canonical =
            terminal_straight_rgba8_is_canonical_for_test(rendered.output.rgba());
        exact_execution &= rendered.working_format == WorkingFormat::HighPrecision
            && exact_grid
            && terminal_canonical;
        let error = high_precision_terminal_error_for_test(rendered.output.rgba(), &expected)
            .unwrap_or(u8::MAX);
        maximum_error = maximum_error.max(error);
        eprintln!(
            "high-precision color operation={name} max_terminal_straight_rgba8_error={error} exact_grid={exact_grid} terminal_canonical={terminal_canonical} output_extent={:?} source_origin={:?} source_extent={:?} source_texel_origin={:?} source_raster_scale={:?} publication_count={}",
            rendered.output_extent,
            rendered.source_origin,
            rendered.source_extent,
            rendered.source_texel_origin,
            rendered.source_raster_scale,
            rendered.publication_count,
        );
    }

    assert!(
        exact_execution && maximum_error <= 2,
        "high-precision color-filter pixels exceed the declared color tolerance"
    );
}

#[test]
fn reduced_precision_color_functions_match_cpu_oracle_with_declared_tolerance() {
    let source = color_filter_boundary_pixels_for_test();
    let width =
        u32::try_from(source.len()).expect("the reduced-precision color matrix width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let (mut renderer, mut surface) =
        color_filter_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, width);
    let mut maximum_alpha_error = 0;
    let mut maximum_premul_error = 0;
    let mut exact_execution = true;

    for (name, operation) in color_filter_operation_matrix_for_test() {
        let expected = color_filter_reference_straight_pixels_for_test(
            &source,
            &[operation],
            WorkingFormat::ReducedPrecision,
        );
        let rendered = render_color_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            vec![color_filter_list([operation])],
            Parameters::default(),
            WorkingFormat::ReducedPrecision,
        );
        let terminal_canonical =
            terminal_straight_rgba8_is_canonical_for_test(rendered.output.rgba());
        exact_execution &= rendered.working_format == WorkingFormat::ReducedPrecision
            && color_filter_frame_has_exact_extent_origin_and_submission_for_test(&rendered, width)
            && terminal_canonical;
        let (alpha_error, premul_error) =
            reduced_precision_terminal_error_for_test(rendered.output.rgba(), &expected)
                .unwrap_or((u8::MAX, u8::MAX));
        maximum_alpha_error = maximum_alpha_error.max(alpha_error);
        maximum_premul_error = maximum_premul_error.max(premul_error);
        eprintln!(
            "reduced-precision color operation={name} max_alpha_error={alpha_error} max_premul8_error={premul_error} terminal_canonical={terminal_canonical}"
        );
    }

    assert!(
        exact_execution && maximum_alpha_error <= 2 && maximum_premul_error <= 2,
        "reduced-precision color-filter alpha or premul8 exceeds the declared tolerance"
    );
}

#[test]
fn filter_function_order_changes_output_and_matches_ordered_oracle() {
    let source = vec![
        [255, 37, 173, 0],
        [224, 72, 16, 127],
        [192, 64, 46, 255],
        [96, 32, 23, 127],
        [17, 231, 93, 255],
    ];
    let width = u32::try_from(source.len()).expect("the ordered color-filter width must fit u32");
    let scene = color_filter_signed_source_scene_for_test(&source);
    let chain_pairs = [
        (
            "noncommuting contrast/brightness",
            [
                ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
            ],
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
                ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
            ],
        ),
        (
            "source-clamp-sensitive brightness chain",
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            ],
            [
                ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
                ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
            ],
        ),
    ];
    let mut ordered_results_are_exact = true;

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) =
            color_filter_pixel_renderer_for_test(working_format, width);
        for (name, first, second) in chain_pairs {
            let first_expected =
                color_filter_reference_straight_pixels_for_test(&source, &first, working_format);
            let second_expected =
                color_filter_reference_straight_pixels_for_test(&source, &second, working_format);
            let first_rendered = render_color_filter_fixture_for_test(
                &mut renderer,
                &mut surface,
                &scene,
                vec![color_filter_list(first)],
                Parameters::default(),
                working_format,
            );
            let second_rendered = render_color_filter_fixture_for_test(
                &mut renderer,
                &mut surface,
                &scene,
                vec![color_filter_list(second)],
                Parameters::default(),
                working_format,
            );
            let first_terminal_canonical =
                terminal_straight_rgba8_is_canonical_for_test(first_rendered.output.rgba());
            let second_terminal_canonical =
                terminal_straight_rgba8_is_canonical_for_test(second_rendered.output.rgba());
            let exact_frames =
                color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                    &first_rendered,
                    width,
                ) && color_filter_frame_has_exact_extent_origin_and_submission_for_test(
                    &second_rendered,
                    width,
                ) && first_terminal_canonical
                    && second_terminal_canonical;
            let first_matches =
                color_filter_rendered_output_matches_for_test(&first_rendered, &first_expected);
            let second_matches =
                color_filter_rendered_output_matches_for_test(&second_rendered, &second_expected);
            ordered_results_are_exact &= first_expected != second_expected
                && first_rendered.output.rgba() != second_rendered.output.rgba()
                && first_matches
                && second_matches
                && exact_frames;
            eprintln!(
                "ordered color-filter chain={name} working_format={working_format:?} first_terminal_canonical={first_terminal_canonical} second_terminal_canonical={second_terminal_canonical}"
            );
        }
    }

    assert!(
        ordered_results_are_exact,
        "the GPU lost authored order or a source clamp"
    );
}

fn color_filter_rendered_output_matches_for_test(
    rendered: &ColorFilterProductionFrameForTest,
    expected: &[u8],
) -> bool {
    match rendered.working_format {
        WorkingFormat::HighPrecision => {
            high_precision_terminal_error_for_test(rendered.output.rgba(), expected)
                .is_some_and(|error| error <= 2)
        }
        WorkingFormat::ReducedPrecision => {
            reduced_precision_terminal_error_for_test(rendered.output.rgba(), expected)
                .is_some_and(|(alpha, premul)| alpha <= 2 && premul <= 2)
        }
    }
}

fn graph_alpha_extreme_pixels_for_test() -> Vec<[u8; 4]> {
    let mut pixels = vec![[128, 0, 0, 1]];
    for alpha in [0, 1, 2, 15, 16, 127, 254, 255] {
        for rgb in [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ] {
            pixels.push([rgb[0], rgb[1], rgb[2], alpha]);
        }
    }
    pixels
}

fn color_filter_boundary_pixels_for_test() -> Vec<[u8; 4]> {
    graph_alpha_extreme_pixels_for_test()
}

fn graph_alpha_extreme_scene_for_test(pixels: &[[u8; 4]]) -> Scene {
    let bytes = pixels
        .iter()
        .flat_map(|pixel| pixel.iter().copied())
        .collect::<Vec<_>>();
    let width = u32::try_from(pixels.len()).expect("the graph pixel vector must fit in u32");
    let image = Image::from_rgba(Size::new(f64::from(width), 1.0), Arc::<[u8]>::from(bytes))
        .expect("the alpha vector must form one valid image");
    let mut scene = Scene::new();
    scene.image(
        image,
        Rect::new(0.0, 0.0, f64::from(width), 1.0),
        ImageFit::Stretch,
    );
    scene
}

fn graph_channel_error_for_test(actual: u8, expected: u8) -> u8 {
    actual.abs_diff(expected)
}

struct GraphAlphaVectorOutputForTest {
    output: ImageBuffer,
    graph: super::renderer::ForcedGraphRenderResultForTest,
    used_graph_transaction: bool,
    publication_count: usize,
}

#[test]
fn resolved_alpha_masks_match_reference_and_rendered_layer_output() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 0, 0, 255, 0, 255, 0, 255],
    )
    .unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 0, 0, 0, 128],
    )
    .unwrap();
    let masked = reference::execute_transitional_resolved_mask_bridge_for_test(
        &source,
        Rect::new(0.0, 0.0, 2.0, 1.0),
        image_from_buffer(mask.clone()),
        Rect::new(0.0, 0.0, 2.0, 1.0),
    )
    .unwrap();
    assert_eq!(masked.rgba(), &[255, 0, 0, 255, 0, 255, 0, 128]);

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask)),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("resolved layer alpha masks should render");
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_f16_to_f32_for_test(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut unbiased = -14_i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased -= 1;
            }
            normalized &= 0x03ff;
            sign | (u32::try_from(unbiased + 127).unwrap() << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_read_gpu_vectors_for_test(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    working_format: WorkingFormat,
    count: usize,
) -> Result<Vec<[f32; 4]>> {
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector readback did not make bounded device progress",
            )
            .with_source(source)
        })?;
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector map callback missed its diagnostic deadline",
            )
            .with_source(source)
        })?
        .map_err(|source| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector staging buffer could not be mapped",
            )
            .with_source(source)
        })?;
    let mapped = slice.get_mapped_range();
    let stride = usize::try_from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT).unwrap();
    let mut rgba = Vec::with_capacity(count);
    for index in 0..count {
        let offset = index.checked_mul(stride).ok_or_else(|| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector offset overflowed",
            )
        })?;
        let pixel = match working_format {
            WorkingFormat::HighPrecision => {
                let bytes = mapped.get(offset..offset + 8).ok_or_else(|| {
                    Error::new(
                        BackendErrorCode::ReadbackFailed,
                        "high-precision composite-shader vector readback is truncated",
                    )
                })?;
                let mut pixel = [0.0; 4];
                for (channel, pair) in bytes.chunks_exact(2).enumerate() {
                    pixel[channel] =
                        composition_f16_to_f32_for_test(u16::from_le_bytes([pair[0], pair[1]]));
                }
                pixel
            }
            WorkingFormat::ReducedPrecision => {
                let bytes = mapped.get(offset..offset + 4).ok_or_else(|| {
                    Error::new(
                        BackendErrorCode::ReadbackFailed,
                        "reduced-precision composite-shader vector readback is truncated",
                    )
                })?;
                [
                    f32::from(bytes[0]) / 255.0,
                    f32::from(bytes[1]) / 255.0,
                    f32::from(bytes[2]) / 255.0,
                    f32::from(bytes[3]) / 255.0,
                ]
            }
        };
        rgba.push(pixel);
    }
    drop(mapped);
    staging.unmap();
    Ok(rgba)
}

#[cfg(not(target_arch = "wasm32"))]
async fn composition_submit_and_read_gpu_vectors_for_test(
    backend: &mut Backend,
    identity: DeviceSlotIdentity,
    transaction: super::gpu_transaction::GpuOperationTransaction,
    prepared: CompositionPreparedGpuVectorsForTest,
) -> Result<CompositionGpuVectorResultsForTest> {
    let CompositionPreparedGpuVectorsForTest {
        device,
        queue,
        working_format,
        mut encoder,
        outputs,
        pass_cache_update,
    } = prepared;
    let output_stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging_size = u64::from(output_stride)
        .checked_mul(u64::try_from(outputs.len()).map_err(|_| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector count does not fit the staging address space",
            )
        })?)
        .ok_or_else(|| {
            Error::new(
                BackendErrorCode::ReadbackFailed,
                "composite-shader vector staging size overflowed",
            )
        })?;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Surgeist composite-shader GPU vector readback"),
        size: staging_size,
        usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
        mapped_at_creation: false,
    });
    for (index, output) in outputs.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: u64::from(output_stride) * u64::try_from(index).unwrap(),
                    bytes_per_row: Some(output_stride),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    super::gpu_transaction::test_support::submit_command_buffer_for_test(
        transaction,
        &queue,
        encoder.finish(),
        RuntimeOperation::EffectRendering,
    )
    .await?;
    backend.commit_checked_pass_cache_update_for_test(identity, pass_cache_update)?;
    let rgba =
        composition_read_gpu_vectors_for_test(&device, &staging, working_format, outputs.len())?;
    Ok(CompositionGpuVectorResultsForTest {
        working_format,
        rgba,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_mask_boundary_vectors_for_test()
-> (Vec<CompositionMaskSamplingVectorForTest>, Vec<[f32; 4]>) {
    let mut vectors = Vec::new();
    let mut expected = Vec::new();
    let rows = [
        (
            ImageQuality::Low,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.0, 0.0),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.666_666_7,
        ),
        (
            ImageQuality::Medium,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.5, 0.5),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.5,
        ),
        (
            ImageQuality::High,
            [
                (Extend::Pad, 0.0, 1.0),
                (Extend::Repeat, 0.5, 0.5),
                (Extend::Reflect, 0.0, 1.0),
            ],
            0.5,
        ),
    ];
    for (quality, extend_rows, vertical_boundary) in rows {
        for (extend, left_boundary, right_boundary) in extend_rows {
            for (point, alpha) in [
                (Point::new(0.0, 0.5), left_boundary),
                (Point::new(4.0, 0.5), right_boundary),
                (Point::new(2.0, 0.0), vertical_boundary),
                (Point::new(2.0, 1.0), vertical_boundary),
                (Point::new(-0.000_1, 0.5), 0.0),
                (Point::new(4.000_1, 0.5), 0.0),
                (Point::new(2.0, -0.000_1), 0.0),
                (Point::new(2.0, 1.000_1), 0.0),
            ] {
                vectors.push(CompositionMaskSamplingVectorForTest {
                    quality,
                    extend,
                    layer_point: point,
                    clip_alpha: None,
                    opacity: 1.0,
                });
                expected.push([alpha; 4]);
            }
        }
    }
    vectors.push(CompositionMaskSamplingVectorForTest {
        quality: ImageQuality::Medium,
        extend: Extend::Repeat,
        layer_point: Point::new(0.0, 0.5),
        clip_alpha: Some(0.5),
        opacity: 0.5,
    });
    expected.push([0.125; 4]);
    (vectors, expected)
}

#[cfg(not(target_arch = "wasm32"))]
fn composition_gpu_vectors_match(
    observed: &CompositionGpuVectorResultsForTest,
    expected: &[[f32; 4]],
    tolerance: f32,
) -> bool {
    observed.rgba.len() == expected.len()
        && observed
            .rgba
            .iter()
            .zip(expected)
            .all(|(actual, expected)| {
                actual.iter().all(|channel| channel.is_finite())
                    && actual[3] >= -tolerance
                    && actual[3] <= 1.0 + tolerance
                    && actual[..3]
                        .iter()
                        .all(|channel| *channel >= -tolerance && *channel <= actual[3] + tolerance)
                    && actual
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
            })
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mask_sampling_shader_matches_independent_boundary_vectors() {
    let (mut backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let (vectors, expected) = composition_mask_boundary_vectors_for_test();
    let observed = pollster::block_on(async {
        let transaction = backend.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let prepared = backend.composition_shader_mask_sampling_preparation_for_test(
            identity,
            &requests,
            &CompositionMaskSamplingInputForTest {
                mask_size: PhysicalSize::new(4, 1),
                mask_rgba: vec![255, 0, 0, 0, 0, 255, 0, 85, 0, 0, 255, 170, 17, 33, 65, 255],
                mask_bounds: Rect::new(0.0, 0.0, 4.0, 1.0),
                source: [1.0; 4],
                vectors,
            },
        )?;
        composition_submit_and_read_gpu_vectors_for_test(
            &mut backend,
            identity,
            transaction,
            prepared,
        )
        .await
    })
    .unwrap();
    let tolerance = match observed.working_format {
        WorkingFormat::HighPrecision | WorkingFormat::ReducedPrecision => 2.0 / 255.0,
    };

    assert!(
        composition_gpu_vectors_match(&observed, &expected, tolerance),
        "GPU mask sampling differs from independent constants"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn blend_shaders_match_independent_known_vectors() {
    let (backend, identity, requests) = composition_selected_backend_and_requests_for_test();
    let source = [0.4, 0.1, 0.3, 0.5];
    let parent = [0.2, 0.6, 0.32, 0.8];
    let vectors = [
        CompositionBlendVectorForTest {
            blend: BlendMode::Normal,
            source,
            parent,
            opacity: 1.25,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Multiply,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Screen,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Overlay,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Darken,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Lighten,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Plus,
            source,
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Plus,
            source: [0.8, 0.2, 0.7, 0.8],
            parent: [0.6, 0.9, 0.4, 0.9],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Multiply,
            source: [0.0; 4],
            parent,
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Screen,
            source,
            parent: [0.0; 4],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Overlay,
            source: [0.0; 4],
            parent: [0.0; 4],
            opacity: 1.0,
        },
        CompositionBlendVectorForTest {
            blend: BlendMode::Normal,
            source,
            parent,
            opacity: -0.25,
        },
    ];
    let expected = [
        [0.5, 0.4, 0.46, 0.9],
        [0.26, 0.38, 0.316, 0.9],
        [0.52, 0.64, 0.524, 0.9],
        [0.34, 0.56, 0.412, 0.9],
        [0.28, 0.4, 0.38, 0.9],
        [0.5, 0.62, 0.46, 0.9],
        [0.6, 0.7, 0.62, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        parent,
        source,
        [0.0; 4],
        parent,
    ];
    assert_composition_blend_gpu_vectors(backend, identity, &requests, &vectors, &expected);
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_composition_blend_gpu_vectors(
    mut backend: Backend,
    identity: DeviceSlotIdentity,
    requests: &super::pass::LayerCompositeCacheRequestsForTest,
    vectors: &[CompositionBlendVectorForTest],
    expected: &[[f32; 4]],
) {
    let observed = pollster::block_on(async {
        let transaction = backend.begin_gpu_operation(
            identity,
            GpuOperationStage::Render,
            RuntimeOperation::EffectRendering,
        )?;
        let prepared =
            backend.composition_shader_blend_preparation_for_test(identity, requests, vectors)?;
        composition_submit_and_read_gpu_vectors_for_test(
            &mut backend,
            identity,
            transaction,
            prepared,
        )
        .await
    })
    .unwrap();
    let tolerance = match observed.working_format {
        WorkingFormat::HighPrecision | WorkingFormat::ReducedPrecision => 3.0 / 255.0,
    };

    assert!(
        composition_gpu_vectors_match(&observed, expected, tolerance),
        "GPU blend math differs from independent constants"
    );
}

#[test]
fn public_dispatch_routes_composition_and_spatial_filters_but_rejects_broad_backdrop() {
    let (scene, filters, size, expected) = spatial_filter_mixed_filter_fixture_for_test();
    let (mut renderer, mut spatial_filter_surface) =
        graph_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, size);
    let spatial_filter = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut spatial_filter_surface,
        &scene,
        filters,
        WorkingFormat::ReducedPrecision,
    );
    let (composition_scene, composition_size, _, _) = composition_reuse_scene_and_oracle_for_test();
    let mut composition_surface = pollster::block_on(renderer.create_headless(
        Size::new(
            f64::from(composition_size.width()),
            f64::from(composition_size.height()),
        ),
        1.0,
    ))
    .expect("masked-composition dispatch coverage requires a surface");
    let composition = pollster::block_on(renderer.render(
        &mut composition_surface,
        &composition_scene,
        Parameters::default(),
    ));
    let unsupported_scene = color_filter_unsupported_backdrop_scene_for_test();
    let mut unsupported_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0))
            .expect("broad-backdrop rejection coverage requires a surface");
    let unsupported = pollster::block_on(renderer.render(
        &mut unsupported_surface,
        &unsupported_scene,
        Parameters::default(),
    ));
    let expected_backdrop = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );

    assert!(
        spatial_filter_pixels_match_oracle_for_test(&spatial_filter, &expected)
            && composition.is_ok()
            && unsupported
                .as_ref()
                .is_err_and(|error| error.unsupported_primitive() == Some(expected_backdrop))
            && unsupported_surface.headless_publication_count_for_test() == 0,
        "public dispatch misrouted masked composition, spatial filters, or broad backdrop"
    );
}

#[test]
fn spatial_filter_fixture_executes_while_public_capabilities_remain_diagnostic() {
    let (scene, filters, size, expected) = spatial_filter_mixed_filter_fixture_for_test();
    let (mut renderer, mut surface) =
        graph_pixel_renderer_for_test(WorkingFormat::ReducedPrecision, size);
    let frame = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &scene,
        filters.clone(),
        WorkingFormat::ReducedPrecision,
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the executed spatial-filter fixture must retain its ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let publication_before = surface.headless_publication_count_for_test();
    let blur = spatial_filter_public_spatial_graph_diagnostic_for_test(
        &scene,
        FilterOp::blur(FilterBlur::try_new(0.75).unwrap()),
        size,
    );
    let shadow = spatial_filter_public_spatial_graph_diagnostic_for_test(
        &scene,
        FilterOp::drop_shadow(
            FilterDropShadow::try_new(
                Point::new(-1.25, 0.5),
                FilterBlur::try_new(0.5).unwrap(),
                Color::BLACK,
            )
            .unwrap(),
        ),
        size,
    );
    let (resources_after, cache_after) = {
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("public spatial-filter diagnostics must retain the ready device");
        (
            ready.internal_resource_manager_observation_for_test(),
            ready.device_pass_cache_counts_for_test(),
        )
    };

    assert!(
        spatial_filter_frame_has_exact_execution_for_test(
            &frame,
            size,
            WorkingFormat::ReducedPrecision
        ) && spatial_filter_pixels_match_oracle_for_test(&frame, &expected)
            && blur
                == Some(UnsupportedPrimitive::new(
                    PrimitiveFamily::Filters,
                    PrimitiveOperation::GpuBlurFilterExecution,
                ))
            && shadow
                == Some(UnsupportedPrimitive::new(
                    PrimitiveFamily::Filters,
                    PrimitiveOperation::GpuDropShadowFilterExecution,
                ))
            && retained_public_filter_diagnostics_are_exact_for_test()
            && repeated_spatial_filter_resources_are_stable_for_test(
                &scene, &filters, size, &expected,
            )
            && spatial_filter_zero_budget_releases_all_frame_resources_for_test(
                &scene, &filters, size, &expected,
            )
            && resources_after == resources_before
            && cache_after == cache_before
            && surface.headless_publication_count_for_test() == publication_before,
        "the spatial-filter fixture did not execute while public capabilities remained diagnostic"
    );
}

fn spatial_filter_alpha_energy_error_for_test(actual: &[u8], expected: &[u8]) -> f64 {
    let actual = actual
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    let expected = expected
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    actual.abs_diff(expected) as f64 / expected.max(1) as f64
}

fn spatial_filter_alpha_energy_relative_to_for_test(actual: &[u8], expected_energy: u64) -> f64 {
    let actual = actual
        .chunks_exact(4)
        .map(|pixel| u64::from(pixel[3]))
        .sum::<u64>();
    actual.abs_diff(expected_energy) as f64 / expected_energy.max(1) as f64
}

fn spatial_filter_frame_has_exact_execution_for_test(
    frame: &SpatialFilterProductionFrameForTest,
    size: PhysicalSize,
    working_format: WorkingFormat,
) -> bool {
    frame.output.size() == size
        && frame.result.output_extent == size
        && frame.result.working_format == working_format
        && frame.result.stats.route == Some(RenderRoute::GpuGraph)
        && frame.publication_count == 1
        && frame.result.stats.commands > 0
        && terminal_straight_rgba8_is_canonical_for_test(frame.output.rgba())
}

fn spatial_filter_pixels_match_oracle_for_test(
    frame: &SpatialFilterProductionFrameForTest,
    expected: &[u8],
) -> bool {
    let (alpha_error, color_error) = spatial_filter_maximum_error_for_test(
        frame.output.rgba(),
        expected,
        frame.result.working_format,
    );
    let energy_error = spatial_filter_alpha_energy_error_for_test(frame.output.rgba(), expected);
    let energy_tolerance = match frame.result.working_format {
        WorkingFormat::HighPrecision => 0.015,
        WorkingFormat::ReducedPrecision => 0.025,
    };
    alpha_error <= 4 && color_error <= 4 && energy_error <= energy_tolerance
}

fn spatial_filter_alpha_centroid_for_test(bytes: &[u8], size: PhysicalSize) -> Point {
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;
    let mut energy = 0.0;
    for (index, pixel) in bytes.chunks_exact(4).enumerate() {
        let alpha = f64::from(pixel[3]);
        let index = u32::try_from(index).expect("the spatial-filter centroid index must fit u32");
        weighted_x += (f64::from(index % size.width()) + 0.5) * alpha;
        weighted_y += (f64::from(index / size.width()) + 0.5) * alpha;
        energy += alpha;
    }
    Point::new(weighted_x / energy, weighted_y / energy)
}

fn spatial_filter_blur_identity_and_transparency_are_exact_for_test(
    working_format: WorkingFormat,
    size: PhysicalSize,
    scene: &Scene,
    blurred: &SpatialFilterProductionFrameForTest,
) -> bool {
    let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let with_identity = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![
            FilterList::try_ops(vec![FilterOp::blur(blur)]).unwrap(),
            FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(0.0).unwrap())]).unwrap(),
        ],
        working_format,
    );
    if !spatial_filter_frame_has_exact_execution_for_test(&with_identity, size, working_format)
        || with_identity.output.rgba() != blurred.output.rgba()
    {
        return false;
    }
    let transparent = spatial_filter_image_scene_for_test(
        PhysicalSize::new(1, 1),
        vec![17, 31, 47, 0],
        Rect::new(6.0, 6.0, 1.0, 1.0),
    );
    let transparent = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        &transparent,
        single_filter_list_for_test(FilterOp::blur(blur)),
        working_format,
    );
    spatial_filter_frame_has_exact_execution_for_test(&transparent, size, working_format)
        && transparent.output.rgba().iter().all(|byte| *byte == 0)
}

fn spatial_filter_drop_shadow_order_is_authored_for_test(
    working_format: WorkingFormat,
    size: PhysicalSize,
    scene: &Scene,
    shadow: FilterDropShadow,
) -> bool {
    let shadow = FilterList::try_ops(vec![FilterOp::drop_shadow(shadow)]).unwrap();
    let blur =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(0.5).unwrap())]).unwrap();
    let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
    let shadow_then_blur = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![shadow.clone(), blur.clone()],
        working_format,
    );
    let blur_then_shadow = render_spatial_filter_fixture_for_test(
        &mut renderer,
        &mut surface,
        scene,
        vec![blur, shadow],
        working_format,
    );
    spatial_filter_frame_has_exact_execution_for_test(&shadow_then_blur, size, working_format)
        && spatial_filter_frame_has_exact_execution_for_test(
            &blur_then_shadow,
            size,
            working_format,
        )
        && shadow_then_blur.output.rgba() != blur_then_shadow.output.rgba()
}

#[test]
fn blur_impulse_is_symmetric_normalized_and_matches_oracle() {
    let size = PhysicalSize::new(17, 17);
    let center = (8, 8);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let source = spatial_filter_reference_buffer_for_test(
        size,
        &[(
            center.0,
            center.1,
            PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap(),
        )],
    );
    let expected = source
        .apply_blur(blur, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut impulse_pixels = vec![0; 7 * 7 * 4];
    impulse_pixels[(3 * 7 + 3) * 4..][..4].copy_from_slice(&[255; 4]);
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(7, 7),
        impulse_pixels,
        Rect::new(5.0, 5.0, 7.0, 7.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let alpha = |x: u32, y: u32| frame.output.rgba()[((y * 17 + x) * 4 + 3) as usize];
        let symmetric = (0..17).all(|coordinate| {
            alpha(coordinate, center.1) == alpha(16 - coordinate, center.1)
                && alpha(center.0, coordinate) == alpha(center.0, 16 - coordinate)
        });
        let (alpha_error, color_error) =
            spatial_filter_maximum_error_for_test(frame.output.rgba(), &expected, working_format);
        let energy_tolerance = match working_format {
            WorkingFormat::HighPrecision => 0.015,
            WorkingFormat::ReducedPrecision => 0.025,
        };
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (5, 5)
                && frame.result.source_spatial.device_extent == PhysicalSize::new(7, 7)
                && frame.result.result_spatial.device_origin == (2, 2)
                && frame.result.result_spatial.device_extent == PhysicalSize::new(13, 13)
                && symmetric
                && alpha_error <= 4
                && color_error <= 4
                && spatial_filter_alpha_energy_relative_to_for_test(frame.output.rgba(), 255)
                    <= energy_tolerance,
            "blur impulse production-GPU comparison exceeds its exact grid, symmetry, or oracle tolerance"
        );
    }
}

#[test]
fn ordinary_blur_samples_transparent_black_at_all_edges() {
    let size = PhysicalSize::new(13, 13);
    let blur = FilterBlur::try_new(1.0).unwrap();
    let opaque = PremultipliedRgba8::try_new(64, 128, 255, 255).unwrap();
    let pixels = (5..=7)
        .flat_map(|y| (5..=7).map(move |x| (x, y, opaque)))
        .collect::<Vec<_>>();
    let source = spatial_filter_reference_buffer_for_test(size, &pixels);
    let expected_high = source
        .apply_blur_to_high_precision_straight_rgba8_for_gpu_oracle(
            blur,
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let expected_reduced = source
        .apply_blur(blur, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut edge_pixels = vec![0; 9 * 9 * 4];
    for y in 3..=5 {
        for x in 3..=5 {
            edge_pixels[(y * 9 + x) * 4..][..4].copy_from_slice(&[64, 128, 255, 255]);
        }
    }
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(9, 9),
        edge_pixels,
        Rect::new(2.0, 2.0, 9.0, 9.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let expected = match working_format {
            WorkingFormat::HighPrecision => &expected_high,
            WorkingFormat::ReducedPrecision => &expected_reduced,
        };
        let alpha = frame
            .output
            .rgba()
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        let transparent_surface_edge = (0..13).all(|coordinate| {
            alpha[coordinate] == 0
                && alpha[12 * 13 + coordinate] == 0
                && alpha[coordinate * 13] == 0
                && alpha[coordinate * 13 + 12] == 0
        });
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (2, 2)
                && frame.result.result_spatial.device_origin == (-1, -1)
                && frame.result.result_spatial.device_extent == PhysicalSize::new(15, 15)
                && transparent_surface_edge
                && spatial_filter_blur_identity_and_transparency_are_exact_for_test(
                    working_format,
                    size,
                    &scene,
                    &frame,
                )
                && spatial_filter_pixels_match_oracle_for_test(&frame, expected),
            "ordinary blur edge production-GPU comparison violates transparent-black sampling"
        );
    }
}

#[test]
fn drop_shadow_preserves_source_uses_fractional_offset_and_expands_signed_bounds() {
    let size = PhysicalSize::new(17, 17);
    let source_pixel = PremultipliedRgba8::try_new(224, 64, 16, 255).unwrap();
    let source = spatial_filter_reference_buffer_for_test(
        size,
        &[(3, 7, source_pixel), (4, 7, source_pixel)],
    );
    let shadow = FilterDropShadow::try_new(
        Point::new(-1.5, 0.75),
        FilterBlur::try_new(1.0).unwrap(),
        Color::try_rgba(0.25, 0.5, 0.75, 0.5).unwrap(),
    )
    .unwrap();
    let expected_high = source
        .apply_fractional_drop_shadow_to_high_precision_straight_rgba8_for_gpu_oracle(
            &shadow,
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let expected_reduced = source
        .apply_fractional_drop_shadow_for_gpu_oracle(&shadow, BlurPolicy::css_filter_default())
        .map(|buffer| reference_straight_bytes_for_test(&buffer))
        .unwrap();
    let mut shadow_pixels = vec![0; 8 * 7 * 4];
    shadow_pixels[(3 * 8 + 3) * 4..][..4].copy_from_slice(&[224, 64, 16, 255]);
    shadow_pixels[(3 * 8 + 4) * 4..][..4].copy_from_slice(&[224, 64, 16, 255]);
    let scene = spatial_filter_image_scene_for_test(
        PhysicalSize::new(8, 7),
        shadow_pixels,
        Rect::new(0.0, 4.0, 8.0, 7.0),
    );

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::drop_shadow(shadow)),
            working_format,
        );
        let expected = match working_format {
            WorkingFormat::HighPrecision => &expected_high,
            WorkingFormat::ReducedPrecision => &expected_reduced,
        };
        let source_is_unchanged = [(3, 7), (4, 7)].into_iter().all(|(x, y)| {
            frame.output.rgba()[((y * 17 + x) * 4) as usize..][..4] == [224, 64, 16, 255]
        });
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.device_origin == (0, 4)
                && frame.result.result_spatial.device_origin.0 < 0
                && frame.result.result_spatial.logical_bounds[0] < 0.0
                && source_is_unchanged
                && spatial_filter_drop_shadow_order_is_authored_for_test(
                    working_format,
                    size,
                    &scene,
                    shadow,
                )
                && spatial_filter_pixels_match_oracle_for_test(&frame, expected),
            "drop-shadow production-GPU comparison loses SourceAlpha, fractional offset, signed bounds, or source merge"
        );
    }
}

#[test]
fn nonuniform_scale_and_skew_preserve_local_blur_shape() {
    let size = PhysicalSize::new(25, 21);
    let transform = Transform::try_new([2.0, 0.0, 0.5, 1.25, 4.0, 3.0]).unwrap();
    let local_bounds = Rect::new(2.0, 4.0, 2.0, 2.0);
    let expected_center = graph_transform_point_for_test(transform, Point::new(3.0, 5.0));
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(local_bounds, Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap());
    });

    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let (mut renderer, mut surface) = graph_pixel_renderer_for_test(working_format, size);
        let frame = render_spatial_filter_fixture_for_test(
            &mut renderer,
            &mut surface,
            &scene,
            single_filter_list_for_test(FilterOp::blur(blur)),
            working_format,
        );
        let centroid = spatial_filter_alpha_centroid_for_test(frame.output.rgba(), size);
        let tolerance = match working_format {
            WorkingFormat::HighPrecision => 0.25,
            WorkingFormat::ReducedPrecision => 0.35,
        };
        assert!(
            spatial_filter_frame_has_exact_execution_for_test(&frame, size, working_format)
                && frame.result.source_spatial.raster_scale == 1.0
                && frame.result.result_spatial.raster_scale == 1.0
                && (centroid.x() - expected_center.x()).abs() <= tolerance
                && (centroid.y() - expected_center.y()).abs() <= tolerance,
            "transformed blur production-GPU comparison exceeds the local-shape centroid tolerance"
        );
    }
}

#[test]
fn public_dispatch_enables_only_bounded_backdrop_execution() {
    let (scene, size, parameters, expected) = bounded_backdrop_integration_fixture_for_test();
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("public bounded-backdrop dispatch coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("public bounded-backdrop dispatch coverage requires a surface");
    let rendered = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        &scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ));
    let actual = pollster::block_on(renderer.read_headless(&surface))
        .expect("the public bounded-backdrop publication must be readable");
    let expected = reference_straight_bytes_for_test(&expected);

    assert!(
        rendered
            .as_ref()
            .is_ok_and(|stats| stats.route == Some(RenderRoute::GpuGraph))
            && graph_pixels_match_for_test(
                actual.rgba(),
                &expected,
                WorkingFormat::ReducedPrecision,
                4,
            )
            && bounded_backdrop_broad_capabilities_remain_diagnostic_for_test()
            && bounded_backdrop_broad_inputs_reject_before_allocation_for_test(
                &mut renderer,
                &mut surface
            ),
        "public dispatch did not enable only the exact bounded-backdrop graph"
    );
}

fn bounded_backdrop_broad_capabilities_remain_diagnostic_for_test() -> bool {
    let capabilities = Capabilities::CURRENT;
    let offscreen = capabilities.offscreen_pipeline();
    let unsupported = [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ),
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::LayerFilterExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ),
    ];
    offscreen.supports_bounded_backdrop_filter_execution()
        && !offscreen.supports_broad_backdrop_execution()
        && !offscreen.supports_backdrop_isolation_composition()
        && !offscreen.supports_layer_filter_execution()
        && !capabilities.filters().supports_layer_filters()
        && capabilities
            .ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::OffscreenPipeline,
                PrimitiveOperation::BoundedBackdropFilterExecution,
            ))
            .is_ok()
        && unsupported.into_iter().all(|expected| {
            capabilities
                .ensure_supported(expected)
                .is_err_and(|error| error.unsupported_primitive() == Some(expected))
        })
}

fn bounded_backdrop_repeated_resources_stabilize_for_test(
    scene: &Scene,
    size: PhysicalSize,
    parameters: Parameters,
    expected: &ReferencePremultipliedRgba8Buffer,
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(512 * 1024 * 1024)),
    ))
    .expect("bounded-backdrop retained-resource coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("bounded-backdrop retained-resource coverage requires a surface");
    for _ in 0..2 {
        pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
            &mut surface,
            scene,
            parameters,
            WorkingFormat::ReducedPrecision,
        ))
        .expect("bounded-backdrop retained-resource warm-up must succeed");
    }
    let warmed_output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the warmed bounded-backdrop publication must be readable");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the warmed bounded-backdrop device must remain ready");
    let warmed_resources = ready.internal_resource_manager_observation_for_test();
    let warmed_cache = ready.device_pass_cache_counts_for_test();
    let mut resources = Vec::new();
    let mut caches = Vec::new();
    let mut stats = Vec::new();
    for _ in 0..3 {
        let frame = pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
            &mut surface,
            scene,
            parameters,
            WorkingFormat::ReducedPrecision,
        ))
        .expect("repeated public bounded-backdrop frames must succeed");
        stats.push(GraphPublicStatsForTest::from(frame));
        let ready = renderer
            .default_ready_device_state_borrow_for_test()
            .expect("repeated public bounded-backdrop frames must retain the ready device");
        resources.push(ready.internal_resource_manager_observation_for_test());
        caches.push(ready.device_pass_cache_counts_for_test());
    }
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated bounded-backdrop publication must remain readable");
    let expected = reference_straight_bytes_for_test(expected);
    color_filter_repeated_resource_observations_are_stable_for_test(&resources, &warmed_resources)
        && resources.iter().all(|actual| {
            actual.gaussian_kernel_count_for_test()
                == warmed_resources.gaussian_kernel_count_for_test()
        })
        && warmed_resources.gaussian_kernel_count_for_test() > 0
        && warmed_resources.effect_texture_count_for_test() > 0
        && warmed_cache.has_render_pipelines()
        && caches.iter().all(|actual| *actual == warmed_cache)
        && stats
            .first()
            .is_some_and(|first| stats.iter().all(|actual| actual == first))
        && output.rgba() == warmed_output.rgba()
        && graph_pixels_match_for_test(output.rgba(), &expected, WorkingFormat::ReducedPrecision, 4)
}

fn bounded_backdrop_zero_budget_releases_idle_resources_for_test(
    scene: &Scene,
    size: PhysicalSize,
    parameters: Parameters,
    expected: &ReferencePremultipliedRgba8Buffer,
) -> bool {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("bounded-backdrop zero-budget coverage requires a renderer");
    let mut surface = pollster::block_on(renderer.create_headless(
        Size::new(f64::from(size.width()), f64::from(size.height())),
        1.0,
    ))
    .expect("bounded-backdrop zero-budget coverage requires a surface");
    pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the first bounded-backdrop zero-budget frame must succeed");
    let first = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first bounded-backdrop zero-budget publication must be readable");
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the first bounded-backdrop zero-budget frame must retain its device")
        .device_pass_cache_counts_for_test();
    pollster::block_on(renderer.render_with_exact_graph_working_format_for_test(
        &mut surface,
        scene,
        parameters,
        WorkingFormat::ReducedPrecision,
    ))
    .expect("the repeated bounded-backdrop zero-budget frame must succeed");
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the repeated bounded-backdrop zero-budget frame must retain its device");
    let resources = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let second = pollster::block_on(renderer.read_headless(&surface))
        .expect("the repeated bounded-backdrop zero-budget publication must be readable");
    let expected = reference_straight_bytes_for_test(expected);

    resources.leased_count == 0
        && resources.idle_count == 0
        && resources.active_frame_count == 0
        && resources.resolved_lease_count == 0
        && resources.entry_count == 0
        && resources.retained_bytes == 0
        && resources.accounted_entry_bytes == Some(0)
        && resources.effect_texture_count_for_test() == 0
        && resources.gaussian_kernel_count_for_test() == 0
        && cache_before == cache_after
        && cache_after.has_render_pipelines()
        && first.rgba() == second.rgba()
        && graph_pixels_match_for_test(second.rgba(), &expected, WorkingFormat::ReducedPrecision, 4)
}

fn bounded_backdrop_diagnostic_backdrop_layer_for_test() -> Layer {
    Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                single_filter_list_for_test(FilterOp::blur(FilterBlur::try_new(0.5).unwrap()))[0]
                    .clone(),
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap()
}

fn bounded_backdrop_broad_inputs_reject_before_allocation_for_test(
    renderer: &mut Renderer,
    surface: &mut Surface,
) -> bool {
    let mut nested = Scene::new();
    nested.layer(Layer::new(), |scene| {
        scene.layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        );
    });
    let mut transformed = Scene::new();
    transformed.layer(
        bounded_backdrop_diagnostic_backdrop_layer_for_test()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap(),
        |_| {},
    );
    let mut repeated = Scene::new();
    repeated
        .layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        )
        .layer(
            bounded_backdrop_diagnostic_backdrop_layer_for_test(),
            |_| {},
        );
    let mut layer_filter = Scene::new();
    layer_filter.layer(
        Layer::new()
            .try_filter(Filter::try_blur(0.5).unwrap())
            .unwrap(),
        |_| {},
    );
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("bounded-backdrop broad diagnostic coverage requires a ready device");
    let resources_before = ready.internal_resource_manager_observation_for_test();
    let cache_before = ready.device_pass_cache_counts_for_test();
    let publication_before = surface.headless_publication_count_for_test();
    let nested = pollster::block_on(renderer.render(surface, &nested, Parameters::default()));
    let transformed =
        pollster::block_on(renderer.render(surface, &transformed, Parameters::default()));
    let repeated = pollster::block_on(renderer.render(surface, &repeated, Parameters::default()));
    let layer_filter =
        pollster::block_on(renderer.render(surface, &layer_filter, Parameters::default()));
    let root = BackdropFilterInput::try_root_backdrop(
        single_filter_list_for_test(FilterOp::blur(FilterBlur::try_new(0.5).unwrap()))[0].clone(),
        None,
    );
    let reference =
        UnresolvedResource::new(UnresolvedResourceKind::Filter, "#broad_backdrop-filter");
    let reference_error = Error::unresolved_resource(reference.clone());
    let ready = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("bounded-backdrop broad rejections must preserve the ready device");
    let resources_after = ready.internal_resource_manager_observation_for_test();
    let cache_after = ready.device_pass_cache_counts_for_test();
    let broad = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BroadBackdropExecution,
    );
    let layer =
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter);
    let root_policy = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::RootBackdropPolicy,
    );
    nested
        .as_ref()
        .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && transformed
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && repeated
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(broad))
        && layer_filter
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(layer))
        && root
            .as_ref()
            .is_err_and(|error| error.unsupported_primitive() == Some(root_policy))
        && reference_error.unresolved_resource_diagnostic() == Some(&reference)
        && resources_after == resources_before
        && cache_after == cache_before
        && surface.headless_publication_count_for_test() == publication_before
}

#[test]
fn bounded_backdrop_fixture_executes_while_broad_capabilities_remain_diagnostic() {
    let (scene, size, parameters, expected) = bounded_backdrop_integration_fixture_for_test();
    let frame = render_bounded_backdrop_fixture_for_test(
        &scene,
        size,
        parameters,
        WorkingFormat::ReducedPrecision,
    );

    assert!(bounded_backdrop_frame_matches_for_test(
        &frame,
        &expected,
        (0, 0),
        size,
        4
    ));
    assert!(bounded_backdrop_broad_capabilities_remain_diagnostic_for_test());
    assert!(bounded_backdrop_repeated_resources_stabilize_for_test(
        &scene, size, parameters, &expected,
    ));
    assert!(
        bounded_backdrop_zero_budget_releases_idle_resources_for_test(
            &scene, size, parameters, &expected,
        )
    );
}

fn bounded_backdrop_capture_reference_for_test(
    source: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    extent: PhysicalSize,
) -> ReferencePremultipliedRgba8Buffer {
    let mut capture = ReferencePremultipliedRgba8Buffer::try_new(extent).unwrap();
    for y in 0..extent.height() {
        for x in 0..extent.width() {
            let source_x = origin.0 + i32::try_from(x).unwrap();
            let source_y = origin.1 + i32::try_from(y).unwrap();
            if source_x >= 0
                && source_y >= 0
                && source_x < i32::try_from(source.physical_size().width()).unwrap()
                && source_y < i32::try_from(source.physical_size().height()).unwrap()
            {
                capture
                    .set_pixel(
                        x,
                        y,
                        source
                            .pixel(
                                u32::try_from(source_x).unwrap(),
                                u32::try_from(source_y).unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    capture
}

fn bounded_backdrop_place_capture_for_test(
    capture: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    output_size: PhysicalSize,
) -> ReferencePremultipliedRgba8Buffer {
    let mut placed = ReferencePremultipliedRgba8Buffer::try_new(output_size).unwrap();
    for y in 0..capture.physical_size().height() {
        for x in 0..capture.physical_size().width() {
            let output_x = origin.0 + i32::try_from(x).unwrap();
            let output_y = origin.1 + i32::try_from(y).unwrap();
            if output_x >= 0
                && output_y >= 0
                && output_x < i32::try_from(output_size.width()).unwrap()
                && output_y < i32::try_from(output_size.height()).unwrap()
            {
                placed
                    .set_pixel(
                        u32::try_from(output_x).unwrap(),
                        u32::try_from(output_y).unwrap(),
                        capture.pixel(x, y).unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    placed
}

fn bounded_backdrop_clip_reference_for_test(
    source: &ReferencePremultipliedRgba8Buffer,
    rect: (u32, u32, u32, u32),
) -> ReferencePremultipliedRgba8Buffer {
    let mut clipped = ReferencePremultipliedRgba8Buffer::try_new(source.physical_size()).unwrap();
    for y in rect.1..rect.1 + rect.3 {
        for x in rect.0..rect.0 + rect.2 {
            clipped
                .set_pixel(x, y, source.pixel(x, y).unwrap())
                .unwrap();
        }
    }
    clipped
}

fn bounded_backdrop_frame_matches_for_test(
    frame: &BoundedBackdropProductionFrameForTest,
    expected: &ReferencePremultipliedRgba8Buffer,
    origin: (i32, i32),
    extent: PhysicalSize,
    tolerance: u8,
) -> bool {
    let expected_size = expected.physical_size();
    let expected = reference_straight_bytes_for_test(expected);
    let energy_tolerance = match frame.result.working_format {
        WorkingFormat::HighPrecision => 0.015,
        WorkingFormat::ReducedPrecision => 0.025,
    };
    frame.result.output_extent == expected_size
        && frame.output.size() == frame.result.output_extent
        && frame.result.parent_spatial.device_origin == (0, 0)
        && frame.result.parent_spatial.device_extent == frame.result.output_extent
        && frame.result.parent_spatial.texel_origin == Point::new(0.0, 0.0)
        && frame.result.parent_spatial.raster_scale == 1.0
        && frame.result.capture_spatial.device_origin == origin
        && frame.result.capture_spatial.device_extent == extent
        && frame.result.capture_spatial.texel_origin
            == Point::new(f64::from(origin.0), f64::from(origin.1))
        && frame.result.capture_spatial.logical_bounds
            == [
                f64::from(origin.0),
                f64::from(origin.1),
                f64::from(extent.width()),
                f64::from(extent.height()),
            ]
        && frame.result.capture_spatial.raster_scale == 1.0
        && frame.result.stats.layers == 1
        && frame.result.stats.route == Some(RenderRoute::GpuGraph)
        && frame.publication_count == 1
        && spatial_filter_alpha_energy_error_for_test(frame.output.rgba(), &expected)
            <= energy_tolerance
        && graph_pixels_match_for_test(
            frame.output.rgba(),
            &expected,
            frame.result.working_format,
            tolerance,
        )
}

#[test]
fn backdrop_blur_mirrors_at_semantic_bounds_not_allocation_padding() {
    let size = PhysicalSize::new(8, 6);
    let origin = (2, 1);
    let extent = PhysicalSize::new(4, 4);
    let base = [32, 64, 96, 255];
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(2.0, 1.0, 1.0, 4.0),
            color_from_straight_rgba8_for_test([240, 32, 16, 255]),
        )
        .fill(
            Rect::new(3.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test([16, 80, 224, 255]),
        )
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                        BackdropCaptureBounds::try_new(Rect::new(2.0, 1.0, 4.0, 4.0)).unwrap(),
                        Some(
                            ClipInput::try_shape(Shape::rect(Rect::new(2.0, 1.0, 4.0, 4.0)))
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (2, 1, 1, 4), [240, 32, 16, 255])
        .source_over(&parent)
        .unwrap();
    let parent = bounded_backdrop_reference_rect_for_test(size, (3, 1, 3, 4), [16, 80, 224, 255])
        .source_over(&parent)
        .unwrap();
    let filtered = bounded_backdrop_capture_reference_for_test(&parent, origin, extent)
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let expected = bounded_backdrop_place_capture_for_test(&filtered, origin, size)
        .source_over(&parent)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, extent, 4),
            "semantic-edge backdrop blur differs from its mirrored oracle: format={working_format:?}"
        );
    }
}

#[test]
fn backdrop_reads_only_completed_prior_content_and_base_once() {
    let size = PhysicalSize::new(8, 6);
    let origin = (-1, 1);
    let extent = PhysicalSize::new(7, 4);
    let base = [32, 64, 96, 255];
    let prior = [224, 48, 24, 255];
    let later = [32, 224, 80, 160];
    let operations = [
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
    ];
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test(prior),
        )
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        color_filter_list(operations),
                        BackdropCaptureBounds::try_new(Rect::new(-1.0, 1.0, 7.0, 4.0)).unwrap(),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        )
        .fill(
            Rect::new(4.0, 2.0, 3.0, 2.0),
            color_from_straight_rgba8_for_test(later),
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 1, 3, 4), prior)
        .source_over(&parent)
        .unwrap();
    let filtered = bounded_backdrop_capture_reference_for_test(&parent, origin, extent)
        .apply_color_filter_pipeline(&color_filter_pipeline(operations))
        .unwrap();
    let completed = bounded_backdrop_place_capture_for_test(&filtered, origin, size)
        .source_over(&parent)
        .unwrap();
    let expected = bounded_backdrop_reference_rect_for_test(size, (4, 2, 3, 2), later)
        .source_over(&completed)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, extent, 2),
            "bounded backdrop capture included a later sibling, omitted a prior sibling, or changed the signed base mapping: format={working_format:?}"
        );
    }
}

#[test]
fn backdrop_foreground_is_not_filtered_and_composites_above_backdrop() {
    let size = PhysicalSize::new(8, 6);
    let origin = (0, 0);
    let base = [32, 64, 192, 255];
    let foreground = [240, 32, 16, 255];
    let blur = FilterBlur::try_new(1.25).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_backdrop_filter(
                BackdropFilterInput::try_new(
                    single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                    BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                    None,
                )
                .unwrap(),
            )
            .unwrap(),
        |scene| {
            scene.fill(
                Rect::new(3.0, 2.0, 2.0, 2.0),
                color_from_straight_rgba8_for_test(foreground),
            );
        },
    );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let filtered = parent
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let group = bounded_backdrop_reference_rect_for_test(size, (3, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let expected = group.source_over(&parent).unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, size, 4),
            "bounded backdrop execution filtered or under-composited its foreground: format={working_format:?}"
        );
    }
}

#[test]
fn later_siblings_observe_completed_backdrop_group() {
    let size = PhysicalSize::new(8, 6);
    let origin = (0, 0);
    let base = [40, 80, 160, 255];
    let foreground = [224, 32, 48, 255];
    let later = [32, 224, 96, 160];
    let invert = ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap());
    let mut scene = Scene::new();
    scene
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        color_filter_list([invert]),
                        BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |scene| {
                scene.fill(
                    Rect::new(2.0, 2.0, 2.0, 2.0),
                    color_from_straight_rgba8_for_test(foreground),
                );
            },
        )
        .fill(
            Rect::new(3.0, 1.0, 3.0, 4.0),
            color_from_straight_rgba8_for_test(later),
        );
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let filtered = parent
        .apply_color_filter_pipeline(&color_filter_pipeline([invert]))
        .unwrap();
    let group = bounded_backdrop_reference_rect_for_test(size, (2, 2, 2, 2), foreground)
        .source_over(&filtered)
        .unwrap();
    let completed = group.source_over(&parent).unwrap();
    let expected = bounded_backdrop_reference_rect_for_test(size, (3, 1, 3, 4), later)
        .source_over(&completed)
        .unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, origin, size, 2),
            "a later sibling failed to observe the complete bounded backdrop group: format={working_format:?}"
        );
    }
}

#[test]
fn outer_clip_precedes_mask_and_opacity_but_follows_filter() {
    let size = PhysicalSize::new(8, 6);
    let base = [32, 64, 192, 255];
    let prior = [240, 32, 16, 255];
    let blur = FilterBlur::try_new(1.0).unwrap();
    let mask_alpha = 128_u8;
    let mask = composition_mask_image_from_alpha_for_test(
        PhysicalSize::new(1, 1),
        &[mask_alpha],
        ImageQuality::Low,
        Extend::Pad,
    );
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(2.0, 1.0, 4.0, 4.0)))
        .unwrap()
        .try_opacity(0.5)
        .unwrap()
        .with_resolved_alpha_mask(
            ResolvedLayerAlphaMask::try_new(mask, Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
        )
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                single_filter_list_for_test(FilterOp::blur(blur))[0].clone(),
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(1.0, 0.0, 1.0, 6.0),
            color_from_straight_rgba8_for_test(prior),
        )
        .layer(layer, |_| {});
    let parent = bounded_backdrop_reference_rect_for_test(size, (0, 0, 8, 6), base);
    let parent = bounded_backdrop_reference_rect_for_test(size, (1, 0, 1, 6), prior)
        .source_over(&parent)
        .unwrap();
    let filtered = parent
        .apply_mirrored_blur_for_gpu_oracle(blur, BlurPolicy::css_filter_default())
        .unwrap();
    let clipped = bounded_backdrop_clip_reference_for_test(&filtered, (2, 1, 4, 4));
    let masked = clipped
        .apply_opacity(f32::from(mask_alpha) / 255.0)
        .unwrap()
        .apply_opacity(0.5)
        .unwrap();
    let expected = masked.source_over(&parent).unwrap();
    for working_format in [
        WorkingFormat::HighPrecision,
        WorkingFormat::ReducedPrecision,
    ] {
        let frame = render_bounded_backdrop_fixture_for_test(
            &scene,
            size,
            Parameters {
                base_color: color_from_straight_rgba8_for_test(base),
                ..Parameters::default()
            },
            working_format,
        );
        assert!(
            bounded_backdrop_frame_matches_for_test(&frame, &expected, (0, 0), size, 4),
            "bounded backdrop outer clip, mask, opacity, or filter order differs from the oracle: format={working_format:?}"
        );
    }
}

#[test]
fn resolved_alpha_masks_leave_luminance_to_authored_mask_diagnostics() {
    let mask = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let error = mask
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("luminance masks remain an authored-mask diagnostic");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );
}

#[test]
fn layer_resolved_alpha_mask_applies_after_children_before_parent_composite() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![
            255, 255, 255, 255, //
            255, 255, 255, 128, //
            0, 0, 0, 0,
        ],
    )
    .unwrap();
    let layer = Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 3.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        );
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!(pixel_rgba(&output, 0, 0)[0] > 200);
    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn nested_resolved_alpha_masked_layers_compose_in_child_then_parent_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0)).unwrap();
    let inner_mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 255, 255, 255, 128],
    )
    .unwrap();
    let outer_mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 128, 255, 255, 255, 255],
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(outer_mask)),
        |scene| {
            scene.layer(
                Layer::new()
                    .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(inner_mask)),
                |scene| {
                    scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
                },
            );
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert!((96..=160).contains(&pixel_alpha(&output, 0, 0)));
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

#[test]
fn layer_resolved_alpha_mask_respects_layer_clip_before_masking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![255, 255, 255, 255, 255, 255, 255, 255],
    )
    .unwrap();
    let mask =
        ResolvedLayerAlphaMask::try_new(image_from_buffer(mask), Rect::new(1.0, 0.0, 2.0, 1.0))
            .unwrap();
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(1.0, 0.0, 2.0, 1.0)))
        .unwrap()
        .with_resolved_alpha_mask(mask);
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert!(pixel_alpha(&output, 2, 0) > 200);
}

#[test]
fn layer_resolved_alpha_mask_composites_after_layer_transform() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 255, 255, 255]).unwrap();
    let layer = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn layer_resolved_alpha_mask_combines_mask_child_opacity_and_layer_opacity() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 255, 255, 128]).unwrap();
    let layer = Layer::new()
        .try_opacity(0.5)
        .unwrap()
        .with_resolved_alpha_mask(resolved_layer_alpha_mask_from_buffer(mask));
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = pollster::block_on(renderer.read_headless(&surface)).unwrap();
    let alpha = pixel_alpha(&output, 0, 0);

    assert!((24..=40).contains(&alpha), "unexpected alpha {alpha}");
}

#[test]
fn reference_blur_zero_radius_is_identity() {
    let pixels = vec![
        PremultipliedRgba8::TRANSPARENT,
        PremultipliedRgba8::try_new(20, 40, 60, 80).unwrap(),
        PremultipliedRgba8::try_new(10, 5, 0, 10).unwrap(),
        PremultipliedRgba8::try_new(255, 128, 64, 255).unwrap(),
    ];
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(2, 2), pixels).unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(0.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(blurred, source);
}

#[test]
fn reference_blur_small_radius_spreads_impulse_deterministically() {
    let impulse = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 3),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            impulse,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    let expected = [15, 25, 15, 25, 41, 25, 15, 25, 15];
    for y in 0..3 {
        for x in 0..3 {
            let value = expected[(y * 3 + x) as usize];
            assert_eq!(
                blurred.pixel(x, y).unwrap(),
                PremultipliedRgba8::try_new(value, 0, 0, value).unwrap(),
                "unexpected blurred impulse at {x},{y}",
            );
        }
    }
}

#[test]
fn reference_blur_samples_outside_source_as_transparent_black() {
    let opaque = PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap();
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![opaque])
            .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(
        blurred.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(41, 41, 41, 41).unwrap()
    );
}

#[test]
fn reference_blur_preserves_partially_transparent_colored_invariants() {
    let partial = PremultipliedRgba8::try_new(80, 40, 20, 128).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            partial,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let blurred = source
        .apply_blur(
            FilterBlur::try_new(1.0).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(
        blurred.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(8, 4, 2, 12).unwrap()
    );
    assert_eq!(
        blurred.pixel(1, 0).unwrap(),
        PremultipliedRgba8::try_new(13, 6, 3, 20).unwrap()
    );
    assert_eq!(
        blurred.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(8, 4, 2, 12).unwrap()
    );
    for x in 0..3 {
        assert_premultiplied(blurred.pixel(x, 0).unwrap());
    }
}

#[test]
fn reference_blur_uses_large_radius_policy() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 1),
        vec![PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap()],
    )
    .unwrap();
    let reject = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_reject_above(1.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();
    let clamp = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_clamp_to(1.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();

    let error = source
        .apply_blur(FilterBlur::try_new(2.0).unwrap(), reject)
        .expect_err("reject policy should reject large blur radius");
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
    assert_eq!(
        source
            .apply_blur(FilterBlur::try_new(2.0).unwrap(), clamp)
            .unwrap()
            .pixel(0, 0)
            .unwrap(),
        PremultipliedRgba8::try_new(41, 0, 0, 41).unwrap()
    );
}

#[test]
fn reference_blur_is_deterministic_across_repeated_runs() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(10, 20, 30, 40).unwrap(),
            PremultipliedRgba8::try_new(200, 0, 0, 200).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(0, 80, 20, 100).unwrap(),
        ],
    )
    .unwrap();

    let first = source
        .apply_blur(
            FilterBlur::try_new(1.25).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();
    let second = source
        .apply_blur(
            FilterBlur::try_new(1.25).unwrap(),
            BlurPolicy::css_filter_default(),
        )
        .unwrap();

    assert_eq!(first, second);
}

#[test]
fn reference_buffers_compare_with_deterministic_equality() {
    let pixel = PremultipliedRgba8::try_new(8, 4, 2, 16).unwrap();
    let first = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![PremultipliedRgba8::TRANSPARENT, pixel],
    )
    .unwrap();
    let same = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![PremultipliedRgba8::TRANSPARENT, pixel],
    )
    .unwrap();
    let different = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 2),
        vec![pixel, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();

    assert_eq!(first, same);
    assert_ne!(first, different);
}

#[test]
fn reference_color_filter_identity_ops_preserve_pixels_byte_for_byte() {
    let pixel = PremultipliedRgba8::try_new(64, 32, 16, 128).unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.0).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(0.0).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(std::f64::consts::TAU).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.0).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.0).unwrap()),
    ]);

    let buffer =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![pixel])
            .unwrap();

    assert_eq!(pixel.apply_color_filter_pipeline(&pipeline).unwrap(), pixel);
    assert_eq!(
        buffer.apply_color_filter_pipeline(&pipeline).unwrap(),
        buffer
    );
}

#[test]
fn color_filter_known_vectors_use_spec_constants_not_oracle_constants() {
    assert_literal_color_filter_primary_vectors();
    assert_literal_color_filter_identity_and_matrix_vectors();
    assert_literal_color_filter_scalar_and_clamp_vectors();
}

fn assert_literal_color_filter_primary_vectors() {
    let red_primary = PremultipliedRgba8::try_new(54, 0, 0, 54).unwrap();
    let grayscale = color_filter_pipeline([ColorFilterOp::Grayscale(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )]);
    assert_eq!(
        red_primary.apply_color_filter_pipeline(&grayscale).unwrap(),
        PremultipliedRgba8::try_new(11, 11, 11, 54).unwrap(),
        "the grayscale primary vector differs from the literal reference color-matrix result"
    );

    let green_primary = PremultipliedRgba8::try_new(0, 79, 0, 79).unwrap();
    assert_eq!(
        green_primary
            .apply_color_filter_pipeline(&grayscale)
            .unwrap(),
        PremultipliedRgba8::try_new(57, 57, 57, 79).unwrap(),
        "the grayscale green primary differs from the literal reference color matrix"
    );
    let blue_primary = PremultipliedRgba8::try_new(0, 0, 104, 104).unwrap();
    assert_eq!(
        blue_primary
            .apply_color_filter_pipeline(&grayscale)
            .unwrap(),
        PremultipliedRgba8::try_new(8, 8, 8, 104).unwrap(),
        "the grayscale blue primary differs from the literal reference color matrix"
    );

    let saturation_zero =
        color_filter_pipeline([ColorFilterOp::Saturate(FilterAmount::try_new(0.0).unwrap())]);
    assert_eq!(
        red_primary
            .apply_color_filter_pipeline(&saturation_zero)
            .unwrap(),
        PremultipliedRgba8::try_new(12, 12, 12, 54).unwrap(),
        "saturation reused the grayscale luma constants"
    );
}

fn assert_literal_color_filter_identity_and_matrix_vectors() {
    let identity_sample = PremultipliedRgba8::try_new(101, 67, 23, 127).unwrap();
    for identity in [
        color_filter_pipeline([ColorFilterOp::Saturate(FilterAmount::try_new(1.0).unwrap())]),
        color_filter_pipeline([ColorFilterOp::HueRotate(
            FilterAngle::try_radians(0.0).unwrap(),
        )]),
        color_filter_pipeline([ColorFilterOp::Sepia(
            UnitFilterAmount::try_new(0.0).unwrap(),
        )]),
    ] {
        assert_eq!(
            identity_sample
                .apply_color_filter_pipeline(&identity)
                .unwrap(),
            identity_sample,
            "a literal reference color-matrix zero or identity vector changed the source"
        );
    }

    let opaque_red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let hue_quarter_turn = color_filter_pipeline([ColorFilterOp::HueRotate(
        FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
    )]);
    assert_eq!(
        opaque_red
            .apply_color_filter_pipeline(&hue_quarter_turn)
            .unwrap(),
        PremultipliedRgba8::try_new(0, 91, 0, 255).unwrap(),
        "quarter-turn hue rotation differs from the literal reference color matrix"
    );

    let sepia = color_filter_pipeline([ColorFilterOp::Sepia(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )]);
    assert_eq!(
        opaque_red.apply_color_filter_pipeline(&sepia).unwrap(),
        PremultipliedRgba8::try_new(100, 89, 69, 255).unwrap(),
        "full sepia differs from the literal reference color matrix"
    );
}

fn assert_literal_color_filter_scalar_and_clamp_vectors() {
    let identity_sample = PremultipliedRgba8::try_new(101, 67, 23, 127).unwrap();
    let brightness = color_filter_pipeline([ColorFilterOp::Brightness(
        FilterAmount::try_new(2.0).unwrap(),
    )]);
    assert_eq!(
        PremultipliedRgba8::try_new(192, 64, 1, 255)
            .unwrap()
            .apply_color_filter_pipeline(&brightness)
            .unwrap(),
        PremultipliedRgba8::try_new(255, 128, 2, 255).unwrap(),
        "brightness did not clamp at its source-operation boundary"
    );
    let clamped_brightness_chain = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
    ]);
    assert_eq!(
        identity_sample
            .apply_color_filter_pipeline(&clamped_brightness_chain)
            .unwrap(),
        PremultipliedRgba8::try_new(64, 64, 23, 127).unwrap(),
        "the oracle omitted a source-operation clamp and premultiply boundary"
    );

    let contrast =
        color_filter_pipeline([ColorFilterOp::Contrast(FilterAmount::try_new(2.0).unwrap())]);
    assert_eq!(
        PremultipliedRgba8::try_new(0, 128, 255, 255)
            .unwrap()
            .apply_color_filter_pipeline(&contrast)
            .unwrap(),
        PremultipliedRgba8::try_new(0, 129, 255, 255).unwrap(),
        "contrast did not clamp at its source-operation boundary"
    );

    let near_gray_saturation = color_filter_pipeline([ColorFilterOp::Saturate(
        FilterAmount::try_new(f64::MAX).unwrap(),
    )]);
    assert_eq!(
        PremultipliedRgba8::try_new(128, 127, 128, 255)
            .unwrap()
            .apply_color_filter_pipeline(&near_gray_saturation)
            .unwrap(),
        PremultipliedRgba8::try_new(255, 0, 255, 255).unwrap(),
        "near-gray saturation did not clamp finitely at its operation boundary"
    );
}

#[test]
fn reference_color_filter_partial_ops_match_deterministic_bytes() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(200, 122, 186, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(-std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(86, 164, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(146, 161, 167, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let pipeline = color_filter_pipeline([op]);
        assert_eq!(
            pixel.apply_color_filter_pipeline(&pipeline).unwrap(),
            expected,
            "unexpected output for {op:?}"
        );
    }
}

#[test]
fn reference_color_filter_extreme_ops_clamp_to_valid_premultiplied_pixels() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ),
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(200, 255, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(73, 173, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(143, 143, 143, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(155, 105, 55, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::TRANSPARENT,
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.0).unwrap()),
            PremultipliedRgba8::try_new(143, 143, 143, 255).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(2.0).unwrap()),
            PremultipliedRgba8::try_new(57, 157, 255, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(1.0).unwrap()),
            PremultipliedRgba8::try_new(192, 171, 134, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let filtered = pixel
            .apply_color_filter_pipeline(&color_filter_pipeline([op]))
            .unwrap();
        assert_eq!(filtered, expected, "unexpected output for {op:?}");
        assert_premultiplied(filtered);
    }
}

#[test]
fn reference_color_filter_buffer_preserves_transparency_and_partial_alpha_invariants() {
    let partial = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![PremultipliedRgba8::TRANSPARENT, partial],
    )
    .unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ]);

    let filtered = buffer.apply_color_filter_pipeline(&pipeline).unwrap();
    let transparent = filtered.pixel(0, 0).unwrap();
    let partial = filtered.pixel(1, 0).unwrap();

    assert_eq!(transparent, PremultipliedRgba8::TRANSPARENT);
    assert_eq!(partial, PremultipliedRgba8::try_new(26, 7, 0, 64).unwrap());
    assert_premultiplied(transparent);
    assert_premultiplied(partial);
}

#[test]
fn compiled_color_filter_pipeline_matches_per_op_reference_chain() {
    let pixel = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        ColorFilterOp::HueRotate(FilterAngle::try_radians(0.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.4).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
        ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.6).unwrap()),
    ]);
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(compiled.source_ops(), pipeline.ops());
    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&compiled)
            .unwrap(),
        pixel.apply_color_filter_pipeline(&pipeline).unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_applies_to_reference_buffers() {
    let first = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let second = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![first, second],
    )
    .unwrap();
    let pipeline = color_filter_pipeline([
        ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
    ]);
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(
        buffer
            .apply_compiled_color_filter_pipeline(&compiled)
            .unwrap(),
        buffer.apply_color_filter_pipeline(&pipeline).unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_fuses_adjacent_color_steps() {
    let fused_color_run = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);
    let opacity_boundary = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);

    assert_eq!(
        CompiledColorFilterPipeline::try_from_pipeline(&fused_color_run)
            .unwrap()
            .executable_step_count(),
        1
    );
    assert_eq!(
        CompiledColorFilterPipeline::try_from_pipeline(&opacity_boundary)
            .unwrap()
            .executable_step_count(),
        3
    );
}

#[test]
fn compiled_color_filter_pipeline_preserves_order_sensitivity() {
    let pixel = PremultipliedRgba8::try_new(90, 130, 210, 255).unwrap();
    let contrast_then_brightness = color_filter_pipeline([
        ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
        ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
    ]);
    let brightness_then_contrast = color_filter_pipeline([
        ColorFilterOp::Brightness(FilterAmount::try_new(0.7).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(1.8).unwrap()),
    ]);
    let contrast_then_brightness =
        CompiledColorFilterPipeline::try_from_pipeline(&contrast_then_brightness).unwrap();
    let brightness_then_contrast =
        CompiledColorFilterPipeline::try_from_pipeline(&brightness_then_contrast).unwrap();

    assert_ne!(
        pixel
            .apply_compiled_color_filter_pipeline(&contrast_then_brightness)
            .unwrap(),
        pixel
            .apply_compiled_color_filter_pipeline(&brightness_then_contrast)
            .unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_sequences_opacity_with_color_steps() {
    let pixel = PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap();
    let opacity_then_invert = color_filter_pipeline([
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ]);
    let invert_then_opacity = color_filter_pipeline([
        ColorFilterOp::Invert(UnitFilterAmount::try_new(1.0).unwrap()),
        ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
    ]);
    let opacity_then_invert_compiled =
        CompiledColorFilterPipeline::try_from_pipeline(&opacity_then_invert).unwrap();
    let invert_then_opacity_compiled =
        CompiledColorFilterPipeline::try_from_pipeline(&invert_then_opacity).unwrap();

    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&opacity_then_invert_compiled)
            .unwrap(),
        pixel
            .apply_color_filter_pipeline(&opacity_then_invert)
            .unwrap()
    );
    assert_eq!(
        pixel
            .apply_compiled_color_filter_pipeline(&invert_then_opacity_compiled)
            .unwrap(),
        pixel
            .apply_color_filter_pipeline(&invert_then_opacity)
            .unwrap()
    );
    assert_ne!(
        pixel
            .apply_compiled_color_filter_pipeline(&opacity_then_invert_compiled)
            .unwrap(),
        pixel
            .apply_compiled_color_filter_pipeline(&invert_then_opacity_compiled)
            .unwrap()
    );
}

#[test]
fn compiled_color_filter_pipeline_rejects_empty_construction() {
    let error = CompiledColorFilterPipeline::try_from_ops(Vec::new())
        .expect_err("empty compiled pipelines should be unconstructable");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("compiled color filter pipeline")
    );
}

#[test]
fn image_straight_rgba8_converts_to_premultiplied_and_back_deterministically() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![90, 120, 150, 0, 64, 128, 255, 128, 255, 10, 20, 255],
    )
    .unwrap();

    let premultiplied =
        reference::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
    assert_eq!(
        premultiplied.pixel(0, 0).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
    assert_eq!(
        premultiplied.pixel(1, 0).unwrap(),
        PremultipliedRgba8::try_new(32, 64, 128, 128).unwrap()
    );
    assert_eq!(
        premultiplied.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(255, 10, 20, 255).unwrap()
    );

    let straight =
        reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&premultiplied)
            .unwrap();

    assert_eq!(straight.size(), PhysicalSize::new(3, 1));
    assert_eq!(
        straight.rgba(),
        &[0, 0, 0, 0, 64, 128, 255, 128, 255, 10, 20, 255]
    );
}

#[test]
fn image_color_filter_execution_applies_color_chain_to_one_pixel_image() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(0.5).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(1.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[50, 75, 100, 255]);
}

#[test]
fn image_color_filter_execution_applies_color_chain_to_multi_pixel_buffer() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 2),
        vec![
            64, 128, 255, 128, 10, 20, 30, 0, 100, 150, 200, 255, 20, 40, 80, 64,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.5).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(2, 2));
    assert_eq!(
        filtered.rgba(),
        &[
            32, 64, 128, 64, 0, 0, 0, 0, 50, 76, 100, 128, 16, 24, 40, 32,
        ]
    );
}

#[test]
fn image_color_filter_execution_preserves_buffer_size_and_rgba_order() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![10, 20, 30, 40, 200, 150, 100, 255],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(2, 1));
    assert_eq!(filtered.rgba(), &[13, 19, 32, 40, 200, 150, 100, 255]);
}

#[test]
fn image_color_filter_execution_changes_image_identity_when_bytes_change() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_ne!(filtered.id(), image.id());
    assert_eq!(filtered.bytes.as_ref(), &[155, 105, 55, 255]);
}

#[test]
fn image_filter_execution_blurs_one_pixel_transparent_and_opaque_images() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 0])).unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let transparent = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(transparent.size(), Size::new(1.0, 1.0));
    assert_eq!(transparent.bytes.as_ref(), &[0, 0, 0, 0]);
    assert_eq!(
        transparent.id(),
        image.id(),
        "identity stays stable when blur leaves bytes unchanged"
    );

    let opaque =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([100, 150, 200, 255])).unwrap();
    let opaque_paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(opaque.id(), opaque.size()).unwrap(),
        filters,
    )
    .unwrap();

    let blurred = reference::ResolvedImageColorFilterExecution::try_new(&opaque_paint, &opaque)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(blurred.size(), Size::new(1.0, 1.0));
    assert_eq!(blurred.bytes.as_ref(), &[100, 149, 199, 41]);
    assert_ne!(
        blurred.id(),
        opaque.id(),
        "filtered output identity changes when blur changes bytes"
    );
}

#[test]
fn image_filter_execution_blurs_multi_pixel_image_with_transparent_edges() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0],
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();

    let blurred =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(blurred.size(), PhysicalSize::new(3, 1));
    assert_eq!(
        blurred.rgba(),
        &[255, 0, 0, 25, 255, 0, 0, 41, 255, 0, 0, 25]
    );
}

#[test]
fn filtered_image_paint_executes_blur_with_matching_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 41, 255, 0, 0, 25]);

    let wrong_id = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(ImageId::new(image.id().get() + 1), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();
    let wrong_size = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), Size::new(1.0, 1.0)).unwrap(),
        filters,
    )
    .unwrap();

    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn materialized_image_filters_preserve_color_and_blur_order() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 1),
        vec![200, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0],
    )
    .unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let color_before_blur = FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap();
    let blur_before_color = FilterList::try_ops(vec![blur, brightness]).unwrap();

    let color_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_blur,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let blur_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &blur_before_color,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();

    assert_eq!(color_before.size(), PhysicalSize::new(3, 1));
    assert_eq!(blur_before.size(), PhysicalSize::new(3, 1));
    assert_ne!(color_before.rgba(), blur_before.rgba());
}

#[test]
fn materialized_image_blur_keeps_output_clipped_to_source_region() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 2),
        vec![255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    .unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(4.0).unwrap())]).unwrap();

    let blurred =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(
        blurred.size(),
        source.size(),
        "materialized image blur inflates for sampling but clips output to source image extent"
    );
    assert_eq!(blurred.rgba().len(), source.rgba().len());
}

#[test]
fn materialized_drop_shadow_quantizes_positive_fractional_offsets_to_nearest_device_pixel() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy
            .quantize(1.25, "filter drop-shadow offset x")
            .unwrap(),
        1
    );
    assert_eq!(
        policy
            .quantize(1.75, "filter drop-shadow offset x")
            .unwrap(),
        2
    );
}

#[test]
fn materialized_drop_shadow_quantizes_negative_fractional_offsets_to_nearest_device_pixel() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy
            .quantize(-1.25, "filter drop-shadow offset x")
            .unwrap(),
        -1
    );
    assert_eq!(
        policy
            .quantize(-1.75, "filter drop-shadow offset x")
            .unwrap(),
        -2
    );
}

#[test]
fn materialized_drop_shadow_quantizes_half_pixel_offsets_away_from_zero() {
    let policy = MaterializedDropShadowOffsetQuantizationPolicy::materialized_cpu_reference();

    assert_eq!(
        policy.quantize(0.5, "filter drop-shadow offset x").unwrap(),
        1
    );
    assert_eq!(
        policy
            .quantize(-0.5, "filter drop-shadow offset x")
            .unwrap(),
        -1
    );
}

#[test]
fn materialized_drop_shadow_uses_alpha_mask_not_source_bounds() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 3),
        vec![
            0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), PhysicalSize::new(3, 3));
    assert_eq!(pixel_rgba(&filtered, 1, 0), [255, 0, 0, 255]);
    assert_eq!(pixel_rgba(&filtered, 2, 0), [0, 0, 0, 255]);
    assert_eq!(pixel_rgba(&filtered, 2, 1), [0, 0, 0, 255]);
    assert_eq!(
        pixel_rgba(&filtered, 1, 2),
        [0, 0, 0, 0],
        "CSS drop-shadow follows the source alpha mask, not the image border box"
    );
}

#[test]
fn materialized_drop_shadow_clips_offset_and_blur_to_source_extent() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(3, 3),
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size(), source.size());
    assert_eq!(filtered.rgba().len(), source.rgba().len());
    assert_eq!(pixel_rgba(&filtered, 1, 1), [255, 255, 255, 255]);
    assert!(
        pixel_alpha(&filtered, 2, 0) > 0,
        "blurred offset shadow should contribute inside the clipped source extent"
    );
}

#[test]
fn materialized_drop_shadow_composites_shadow_behind_source() {
    let source = ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![255, 0, 0, 128]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[170, 0, 0, 192]);
}

#[test]
fn filtered_image_paint_executes_drop_shadow_with_matching_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 255, 0, 0, 0, 255]);

    let wrong_id = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(ImageId::new(image.id().get() + 1), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();
    let wrong_size = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), Size::new(1.0, 1.0)).unwrap(),
        filters,
    )
    .unwrap();

    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        reference::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn resource_only_drop_shadow_filtered_image_paint_stays_rejected() {
    let resource = ResolvedImageResource::try_new(ImageId::new(41), Size::new(2.0, 1.0)).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();

    let unsupported = paint
        .ensure_supported(Capabilities::CURRENT)
        .expect_err("resource-only filtered image paint is not materialized bytes");

    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn materialized_filters_after_drop_shadow_apply_to_composed_output() {
    let source =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[0, 255, 255, 255, 255, 255, 255, 255]);
}

#[test]
fn materialized_filters_before_drop_shadow_shape_current_alpha_mask() {
    let source =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();

    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba(), &[255, 0, 0, 128, 0, 0, 0, 128]);
}

#[test]
fn materialized_image_filter_reference_preserves_nonzero_blur_then_drop_shadow() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = reference::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(2.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[255, 0, 0, 41, 103, 0, 0, 62]);
    assert_ne!(
        filtered.id(),
        image.id(),
        "materialized filtered output identity should reflect nonzero blur/drop-shadow bytes"
    );
}

#[test]
fn mixed_color_and_pixel_filters_preserve_authored_order() {
    let image_buffer =
        ImageBuffer::try_new(PhysicalSize::new(2, 1), vec![255, 0, 0, 255, 0, 0, 0, 0]).unwrap();
    let color_before_pixel = FilterList::try_ops(vec![
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    let pixel_before_color = FilterList::try_ops(vec![
        FilterOp::try_drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        )
        .unwrap(),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let color_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_pixel,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let pixel_before = reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &pixel_before_color,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    assert_ne!(
        color_before.rgba(),
        pixel_before.rgba(),
        "mixed color and pixel-moving filters must preserve authored order"
    );
}

#[test]
fn color_filter_operations_match_compiled_and_reference_bytes() {
    let source = PremultipliedRgba8::try_new(100, 150, 200, 255).unwrap();
    let cases = [
        (
            ColorFilterOp::Brightness(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 255).unwrap(),
        ),
        (
            ColorFilterOp::Contrast(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::HueRotate(
                FilterAngle::try_radians(std::f64::consts::FRAC_PI_2).unwrap(),
            ),
            PremultipliedRgba8::try_new(200, 122, 186, 255).unwrap(),
        ),
        (
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.25).unwrap()),
            PremultipliedRgba8::try_new(114, 139, 164, 255).unwrap(),
        ),
        (
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(50, 75, 100, 128).unwrap(),
        ),
        (
            ColorFilterOp::Saturate(FilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(121, 146, 171, 255).unwrap(),
        ),
        (
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.5).unwrap()),
            PremultipliedRgba8::try_new(146, 161, 167, 255).unwrap(),
        ),
    ];

    for (op, expected) in cases {
        let pipeline = color_filter_pipeline([op]);
        let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

        assert_eq!(
            source
                .apply_compiled_color_filter_pipeline(&compiled)
                .unwrap(),
            expected,
            "unexpected compiled output for {op:?}"
        );
        assert_eq!(
            source
                .apply_compiled_color_filter_pipeline(&compiled)
                .unwrap(),
            source.apply_color_filter_pipeline(&pipeline).unwrap(),
            "compiled and CPU reference paths should agree for {op:?}"
        );
    }
}

#[test]
fn materialized_color_filter_fusion_matches_reference_bytes() {
    let source = ImageBuffer::try_new(
        PhysicalSize::new(2, 1),
        vec![100, 150, 200, 255, 64, 128, 255, 128],
    )
    .unwrap();
    let filters = color_filter_list([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);
    let pipeline = filters
        .color_filter_pipeline()
        .unwrap()
        .expect("color-only filters should produce an executable color pipeline");
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(compiled.executable_step_count(), 1);

    let premultiplied =
        reference::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
    let reference = premultiplied
        .apply_compiled_color_filter_pipeline(&compiled)
        .unwrap();
    let expected =
        reference::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&reference)
            .unwrap();
    let filtered =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered, expected);
    assert_ne!(filtered.rgba(), source.rgba());
}

#[test]
fn materialized_filter_support_does_not_enable_layer_effect_execution() {
    let image_buffer =
        ImageBuffer::try_new(PhysicalSize::new(1, 1), vec![100, 150, 200, 255]).unwrap();
    let shadow = Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap();
    let drop_shadow =
        FilterList::try_ops(vec![FilterOp::try_drop_shadow(shadow).unwrap()]).unwrap();

    let drop_shadow_output =
        reference::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
            &drop_shadow,
            &image_buffer,
        )
        .unwrap()
        .execute_to_image_buffer()
        .unwrap();
    assert_eq!(drop_shadow_output.size(), image_buffer.size());

    let layer_filter_error = normalize_single_layer_error(
        Layer::new()
            .try_filter(Filter::try_blur(2.0).unwrap())
            .unwrap(),
    );
    assert_eq!(
        layer_filter_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );

    let layer_mask_error = normalize_single_layer_error(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
            .unwrap(),
    );
    assert_eq!(
        layer_mask_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );

    for unsupported in [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaskExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
    ] {
        let error = Capabilities::CURRENT
            .ensure_supported(unsupported)
            .expect_err("unsupported compositor execution must remain typed");

        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn runtime_capability_report_keeps_precision_flags_independent() {
    let combinations = [(true, true), (true, false), (false, true), (false, false)];

    for (high_precision, reduced_precision) in combinations {
        let precisions = EffectPrecisionCapabilities::new(high_precision, reduced_precision);
        let available = AvailableRuntimeCapabilities::new(Format::Bgra8, precisions, 8_192);
        let report = RuntimeCapabilities::Available(available);

        assert_eq!(precisions.supports_high_precision(), high_precision);
        assert_eq!(precisions.supports_reduced_precision(), reduced_precision);
        assert_eq!(available.surface_format(), Format::Bgra8);
        assert_eq!(available.effect_precisions(), precisions);
        assert_eq!(available.max_effect_texture_dimension_2d(), 8_192);
        assert_eq!(report.available(), Some(available));
        assert_eq!(report.unavailable_reason(), None);
    }

    let unavailable_reason = RuntimeCapabilityUnavailableReason::AdapterUnavailable;
    let unavailable = RuntimeCapabilities::Unavailable(unavailable_reason);
    assert_eq!(unavailable.available(), None);
    assert_eq!(unavailable.unavailable_reason(), Some(unavailable_reason));

    fn assert_report_traits<T: Clone + Copy + std::fmt::Debug + Eq + PartialEq>() {}
    assert_report_traits::<RuntimeCapabilities>();
    assert_report_traits::<AvailableRuntimeCapabilities>();
    assert_report_traits::<EffectPrecisionCapabilities>();
}

#[test]
fn precision_resolver_covers_both_high_only_reduced_only_and_neither() {
    assert_working_format_contracts();
    assert_precision_resolution_matrix();
}

fn assert_working_format_contracts() {
    let required_usages = wgpu::TextureUsages::RENDER_ATTACHMENT
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST);
    for (format, texture_format, bytes_per_pixel) in [
        (
            WorkingFormat::HighPrecision,
            wgpu::TextureFormat::Rgba16Float,
            8,
        ),
        (
            WorkingFormat::ReducedPrecision,
            wgpu::TextureFormat::Rgba8Unorm,
            4,
        ),
    ] {
        assert_eq!(format.texture_format(), texture_format);
        assert_eq!(format.required_usages(), required_usages);
        assert_eq!(
            format.required_format_features(),
            wgpu::TextureFormatFeatureFlags::FILTERABLE,
        );
        assert_eq!(format.bytes_per_pixel(), bytes_per_pixel);

        let complete_features = wgpu::TextureFormatFeatures {
            allowed_usages: required_usages,
            flags: wgpu::TextureFormatFeatureFlags::FILTERABLE,
        };
        assert!(format.is_supported_by(complete_features));
        for required_usage in [
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::TextureUsages::COPY_SRC,
            wgpu::TextureUsages::COPY_DST,
        ] {
            assert!(!format.is_supported_by(wgpu::TextureFormatFeatures {
                allowed_usages: required_usages.difference(required_usage),
                ..complete_features
            }));
        }
        assert!(!format.is_supported_by(wgpu::TextureFormatFeatures {
            flags: wgpu::TextureFormatFeatureFlags::empty(),
            ..complete_features
        }));
    }
}

fn assert_precision_resolution_matrix() {
    let cases = [
        (
            true,
            true,
            EffectQualityPolicy::RequireHighPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            true,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            false,
            EffectQualityPolicy::RequireHighPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (
            true,
            false,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::HighPrecision),
        ),
        (false, true, EffectQualityPolicy::RequireHighPrecision, None),
        (
            false,
            true,
            EffectQualityPolicy::AllowReducedPrecision,
            Some(WorkingFormat::ReducedPrecision),
        ),
        (
            false,
            false,
            EffectQualityPolicy::RequireHighPrecision,
            None,
        ),
        (
            false,
            false,
            EffectQualityPolicy::AllowReducedPrecision,
            None,
        ),
    ];

    for (high_precision, reduced_precision, policy, expected) in cases {
        let capabilities =
            DeviceCapabilities::from_test_facts(high_precision, reduced_precision, 8_192);
        let resolution = capabilities.resolve_effect_working_format(policy);

        match expected {
            Some(expected_format) => assert_eq!(
                resolution.expect("the available format should satisfy the requested policy"),
                expected_format,
            ),
            None => {
                let error = resolution
                    .expect_err("the unavailable format should reject the requested policy");
                let expected_diagnostic = RuntimeCapabilityUnavailable::try_new(
                    RuntimeOperation::EffectRendering,
                    RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { policy },
                )
                .unwrap();
                assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
                assert_eq!(
                    error.runtime_capability_unavailable_diagnostic(),
                    Some(&expected_diagnostic),
                );
            }
        }
    }
}

fn composition_rejected_capability_rows(
    _filters: FilterCapabilities,
    masks: MaskClipCapabilities,
    offscreen: OffscreenPipelineCapabilities,
) -> [CapabilityRowForTest; 9] {
    [
        (
            offscreen.supports_layer_filter_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::LayerFilterExecution,
        ),
        (
            offscreen.supports_broad_backdrop_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BroadBackdropExecution,
        ),
        (
            masks.supports_layer_masks(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ),
        (
            offscreen.supports_mask_execution(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaskExecution,
        ),
        (
            masks.supports_luminance_mask_mode(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ),
        (
            masks.supports_multi_layer_mask_composition(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ),
        (
            masks.supports_mask_composite_modes(),
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ),
        (
            offscreen.supports_offscreen_layer_rendering(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::OffscreenLayerRendering,
        ),
        (
            offscreen.supports_backdrop_isolation_composition(),
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ),
    ]
}
