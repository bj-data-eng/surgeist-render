use crate::{
    Antialiasing, BackdropCaptureBounds, BackdropFilterInput, BlendMode, Capabilities, Color,
    EffectQualityPolicy, ErrorCode, Extend, FilterAmount, FilterAngle, FilterCapabilities,
    FilterList, FilterOp, Format, Image, ImageQuality, InvalidValue, Layer, MaskClipCapabilities,
    OffscreenPipelineCapabilities, Options, Parameters, PhysicalSize, Point, PrimitiveFamily,
    PrimitiveOperation, Rect, RenderRoute, Renderer, ResolvedLayerAlphaMask, ResourceCacheBudget,
    Result, Scene, Shape, Size, Transform, UnitFilterAmount, UnsupportedPrimitive,
    backend::{Backend, DeviceCapabilities},
    command,
    pass::pass_spatial_uniform_bytes_for_test,
    resource::{
        GaussianKernelBufferLimits, GaussianKernelPlan, GaussianKernelSamplingForm, WorkingFormat,
    },
    shader::device_pass_cache_owns_exact_key_spaces_for_test,
};

use super::{
    UnwrapOrPanicForTest, assert_gaussian_kernel_upload_lifecycle,
    composition_composite_requests_for_test, composition_frame_context_for_test,
    composition_mask_image_for_test, composition_selected_backend_and_requests_for_test,
    composition_shader_composite_commands_for_test, default_graph_working_format_for_test,
    graph_encoding_backend_for_test,
    support::{
        authored_color_filter_runs_for_test, bounded_backdrop_graph_commands_for_test,
        color_then_blur_filters_for_test, composition_commands_for_test,
        filter_graph_commands_for_test, filter_graph_context_for_test,
        graph_shader_commands_for_test, graph_shader_frame_context_for_test, pixel_rgba,
        spatial_filter_authored_filter_steps_for_test,
    },
};

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
