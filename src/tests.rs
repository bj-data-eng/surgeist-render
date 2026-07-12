use super::gpu_transaction::{GpuOperationDraft, GpuOperationLease, GpuOperationStage};
use super::{
    backend::*,
    command,
    encode::*,
    reference::{
        MaterializedDropShadowOffsetQuantizationPolicy, PremultipliedRgba8,
        ReferencePremultipliedRgba8Buffer,
    },
    shader::{
        RectPassBounds, RectShaderPassDescriptor, RectShaderPassExecution, RectShaderPassKind,
        RectShaderPipelineKey, encode_clear_fill_pass,
    },
    surface::{HeadlessResources, SurfaceBackend},
    texture::{
        OffscreenTextureCache, TextureCacheKey, TextureDescriptor, TextureUsageIntent,
        headless_texture_descriptor,
    },
};
use std::{
    future::{Future, pending},
    sync::Arc,
    task::{Context, Poll, Waker},
    time::Duration,
};

use super::error::BackendErrorCode;
use super::*;

const AHEM_FONT_BYTES: &[u8] = include_bytes!("../tests/fixtures/fonts/ahem/Ahem.ttf");
const AHEM_FONT_ID: u64 = 9001;
const AHEM_GLYPH_X: u32 = 58;
const AHEM_GLYPH_DESCENT_P: u32 = 82;
const AHEM_GLYPH_ASCENT_E_ACUTE: u32 = 100;

#[derive(Debug)]
struct ErrorSourceFixture;

impl std::fmt::Display for ErrorSourceFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("error source fixture")
    }
}

impl std::error::Error for ErrorSourceFixture {}

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
fn runtime_errors_distinguish_semantic_unsupported_from_device_unavailable() {
    let unsupported = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::LayerFilter,
    ));
    let unavailable = Error::runtime_capability_unavailable(
        RuntimeCapabilityUnavailable::try_new(
            RuntimeOperation::SurfaceRendering,
            RuntimeCapabilityUnavailableReason::DeviceLost {
                reason: DeviceLossReason::Destroyed,
            },
        )
        .unwrap(),
    );

    assert_eq!(unsupported.code(), ErrorCode::UnsupportedPrimitive);
    assert!(unsupported.unsupported_primitive().is_some());
    assert_eq!(
        unsupported.runtime_capability_unavailable_diagnostic(),
        None
    );
    assert_eq!(unavailable.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(unavailable.unsupported_primitive(), None);
    assert_eq!(
        unavailable
            .runtime_capability_unavailable_diagnostic()
            .map(|diagnostic| diagnostic.operation()),
        Some(RuntimeOperation::SurfaceRendering)
    );
}

#[test]
fn runtime_diagnostic_constructor_rejects_every_unlisted_operation_reason_pair() {
    let operations = [
        RuntimeOperation::AdapterSelection,
        RuntimeOperation::SurfaceRendering,
        RuntimeOperation::SurfaceReadback,
        RuntimeOperation::SurfaceResume,
        RuntimeOperation::EffectRendering,
        RuntimeOperation::EffectTextureAllocation,
        RuntimeOperation::EffectPresentation,
    ];
    let reasons = [
        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Suspended,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::NonRenderable,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Uninitialized,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Occluded,
        },
        RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Lost,
        },
        RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        },
        RuntimeCapabilityUnavailableReason::DeviceFaulted {
            kind: GpuFaultKind::Validation,
        },
        RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
            kind: SurfaceIdentityMismatchKind::ForeignRenderer,
        },
        RuntimeCapabilityUnavailableReason::EffectFormatUnavailable {
            policy: EffectQualityPolicy::RequireHighPrecision,
        },
        RuntimeCapabilityUnavailableReason::TextureDimensionExceeded {
            requested: PhysicalSize::new(17, 19),
            maximum: 16,
        },
        RuntimeCapabilityUnavailableReason::SurfaceFormatUnavailable {
            format: Format::Bgra8,
        },
    ];

    for operation in operations {
        for reason in reasons {
            let result = RuntimeCapabilityUnavailable::try_new(operation, reason);
            if runtime_pair_is_listed(operation, reason) {
                let diagnostic = result.unwrap();
                assert_eq!(diagnostic.operation(), operation);
                assert_eq!(diagnostic.reason(), reason);
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidInput);
                assert!(error.invalid_value_diagnostic().is_some());
            }
        }
    }
}

fn runtime_pair_is_listed(
    operation: RuntimeOperation,
    reason: RuntimeCapabilityUnavailableReason,
) -> bool {
    match operation {
        RuntimeOperation::AdapterSelection => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Occluded
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceReadback => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::AdapterUnavailable
                | RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Suspended
                        | RenderSurfaceAvailability::NonRenderable
                        | RenderSurfaceAvailability::Uninitialized
                        | RenderSurfaceAvailability::Lost,
                }
                | RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::SurfaceResume => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectRendering => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::EffectFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectTextureAllocation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::TextureDimensionExceeded { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
        RuntimeOperation::EffectPresentation => matches!(
            reason,
            RuntimeCapabilityUnavailableReason::SurfaceFormatUnavailable { .. }
                | RuntimeCapabilityUnavailableReason::DeviceLost { .. }
                | RuntimeCapabilityUnavailableReason::DeviceFaulted { .. }
        ),
    }
}

#[test]
fn typed_error_codes_cannot_exist_without_their_matching_payload() {
    let runtime = RuntimeCapabilityUnavailable::try_new(
        RuntimeOperation::SurfaceRendering,
        RuntimeCapabilityUnavailableReason::AdapterUnavailable,
    )
    .unwrap();
    let errors = [
        Error::invalid_value("field", "value", "must be valid"),
        Error::unsupported_render_primitive(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        )),
        Error::unresolved_resource(UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            "image",
        )),
        Error::degraded_quality(DegradedQuality::new(
            DegradedQualityKind::ReducedIntermediatePrecision,
            "reduced",
        )),
        Error::runtime_capability_unavailable(runtime),
    ];

    for error in &errors {
        let typed_payloads = [
            error.invalid_value_diagnostic().is_some(),
            error.unsupported_primitive().is_some(),
            error.unresolved_resource_diagnostic().is_some(),
            error.degraded_quality_diagnostic().is_some(),
            error.runtime_capability_unavailable_diagnostic().is_some(),
        ];
        assert_eq!(typed_payloads.iter().filter(|present| **present).count(), 1);
        match error.code() {
            ErrorCode::InvalidInput => assert!(typed_payloads[0]),
            ErrorCode::UnsupportedPrimitive => assert!(typed_payloads[1]),
            ErrorCode::UnresolvedResource => assert!(typed_payloads[2]),
            ErrorCode::DegradedQuality => assert!(typed_payloads[3]),
            ErrorCode::RuntimeCapabilityUnavailable => assert!(typed_payloads[4]),
            _ => panic!("semantic constructor returned a non-semantic code"),
        }
    }

    let backend_codes = [
        BackendErrorCode::AdapterUnavailable,
        BackendErrorCode::DeviceCreateFailed,
        BackendErrorCode::RendererCreateFailed,
        BackendErrorCode::SurfaceCreateFailed,
        BackendErrorCode::SurfaceConfigureFailed,
        BackendErrorCode::SurfaceLost,
        BackendErrorCode::SurfaceOutOfMemory,
        BackendErrorCode::SurfaceTimeout,
        BackendErrorCode::SurfaceOutdated,
        BackendErrorCode::SurfaceUnavailable,
        BackendErrorCode::ImageUploadFailed,
        BackendErrorCode::RenderFailed,
        BackendErrorCode::PresentFailed,
        BackendErrorCode::UnsupportedBackend,
    ];
    for code in backend_codes {
        let error = Error::new(code, "backend failure");
        assert!(!matches!(
            error.code(),
            ErrorCode::InvalidInput
                | ErrorCode::UnsupportedPrimitive
                | ErrorCode::UnresolvedResource
                | ErrorCode::DegradedQuality
                | ErrorCode::RuntimeCapabilityUnavailable
        ));
        assert!(error.invalid_value_diagnostic().is_none());
        assert!(error.unsupported_primitive().is_none());
        assert!(error.unresolved_resource_diagnostic().is_none());
        assert!(error.degraded_quality_diagnostic().is_none());
        assert!(error.runtime_capability_unavailable_diagnostic().is_none());
    }
}

#[test]
fn semantic_error_accessors_preserve_payloads() {
    let invalid = Error::invalid_value("radius", -1, "must be non-negative");
    let unsupported = Error::unsupported_render_primitive(UnsupportedPrimitive::new(
        PrimitiveFamily::Filters,
        PrimitiveOperation::LayerFilter,
    ));
    let unresolved = Error::unresolved_resource(UnresolvedResource::new(
        UnresolvedResourceKind::Image,
        "hero-image",
    ));
    let degraded = Error::degraded_quality(DegradedQuality::new(
        DegradedQualityKind::ReducedIntermediatePrecision,
        "rgba16float unavailable",
    ));

    assert_eq!(invalid.code(), ErrorCode::InvalidInput);
    assert_eq!(
        invalid.message(),
        "radius value -1 is invalid: must be non-negative"
    );
    assert_eq!(
        invalid.invalid_value_diagnostic().map(InvalidValue::field),
        Some("radius")
    );
    assert_eq!(unsupported.code(), ErrorCode::UnsupportedPrimitive);
    assert!(unsupported.unsupported_primitive().is_some());
    assert_eq!(unresolved.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        unresolved
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("hero-image")
    );
    assert_eq!(degraded.code(), ErrorCode::DegradedQuality);
    assert_eq!(
        degraded
            .degraded_quality_diagnostic()
            .map(DegradedQuality::kind),
        Some(DegradedQualityKind::ReducedIntermediatePrecision)
    );
}

#[test]
fn native_and_wasm_error_source_storage_preserves_source_contract() {
    #[cfg(not(target_arch = "wasm32"))]
    fn assert_send_sync<T: Send + Sync>() {}

    #[cfg(not(target_arch = "wasm32"))]
    assert_send_sync::<Error>();

    let error = Error::new(BackendErrorCode::RenderFailed, "backend failed")
        .with_source(ErrorSourceFixture);

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(error.message(), "backend failed");
    assert_eq!(
        std::error::Error::source(&error)
            .map(ToString::to_string)
            .as_deref(),
        Some("error source fixture")
    );
}

#[test]
fn options_default_requires_high_precision_and_bounds_retention() {
    let options = Options::new();

    assert_eq!(options, Options::default());
    assert_eq!(options.antialiasing(), Antialiasing::Area);
    assert!(!options.debug());
    assert_eq!(
        options.effect_quality_policy(),
        EffectQualityPolicy::RequireHighPrecision
    );
    assert_eq!(
        options.resource_cache_budget(),
        ResourceCacheBudget::DEFAULT
    );

    let configured = options
        .with_antialiasing(Antialiasing::Msaa16)
        .with_debug(true)
        .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
        .with_resource_cache_budget(ResourceCacheBudget::new(8));

    assert_eq!(configured.antialiasing(), Antialiasing::Msaa16);
    assert!(configured.debug());
    assert_eq!(
        configured.effect_quality_policy(),
        EffectQualityPolicy::AllowReducedPrecision
    );
    assert_eq!(configured.resource_cache_budget().bytes(), 8);
    assert_eq!(
        pollster::block_on(Renderer::new(configured))
            .unwrap()
            .options(),
        configured
    );
}

#[test]
fn resource_cache_budget_zero_disables_idle_retention() {
    let disabled = ResourceCacheBudget::new(0);

    assert_eq!(disabled, ResourceCacheBudget::DISABLED);
    assert_eq!(disabled.bytes(), 0);
    assert_eq!(ResourceCacheBudget::default(), ResourceCacheBudget::DEFAULT);
    assert_eq!(ResourceCacheBudget::DEFAULT.bytes(), 64 * 1024 * 1024);
}

#[test]
fn vello_renderer_options_force_gpu_execution() {
    let options = Options::new().with_antialiasing(Antialiasing::Msaa8);
    let vello_options = vello_renderer_options(options);

    assert!(!vello_options.use_cpu);
    assert_eq!(
        vello_options.antialiasing_support,
        vello_aa_support(Antialiasing::Msaa8)
    );
}

#[test]
fn text_run_bounds_distinguish_unspecified_empty_and_ink() {
    let unspecified = TextRunBounds::unspecified();
    let empty = TextRunBounds::empty();
    let ink_rect = Rect::new(-2.0, -3.0, 4.0, 5.0);
    let ink = TextRunBounds::try_ink(ink_rect).unwrap();

    assert_eq!(unspecified.kind(), TextRunBoundsKind::Unspecified);
    assert_eq!(empty.kind(), TextRunBoundsKind::Empty);
    assert_eq!(ink.kind(), TextRunBoundsKind::Ink);
    assert_eq!(unspecified.ink_rect(), None);
    assert_eq!(empty.ink_rect(), None);
    assert_eq!(ink.ink_rect(), Some(ink_rect));
    let non_finite_x = TextRunBounds::try_ink(Rect::new(f64::NAN, 0.0, 1.0, 1.0)).unwrap_err();
    let non_finite_y = TextRunBounds::try_ink(Rect::new(0.0, f64::INFINITY, 1.0, 1.0)).unwrap_err();
    let non_finite_width = TextRunBounds::try_ink(Rect::new(0.0, 0.0, f64::NAN, 1.0)).unwrap_err();
    let non_finite_height =
        TextRunBounds::try_ink(Rect::new(0.0, 0.0, 1.0, f64::NEG_INFINITY)).unwrap_err();
    let zero_width = TextRunBounds::try_ink(Rect::new(0.0, 0.0, 0.0, 1.0)).unwrap_err();
    let zero_height = TextRunBounds::try_ink(Rect::new(0.0, 0.0, 1.0, 0.0)).unwrap_err();
    assert_eq!(non_finite_x.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_y.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_width.code(), ErrorCode::InvalidInput);
    assert_eq!(non_finite_height.code(), ErrorCode::InvalidInput);
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);
    assert_eq!(zero_height.code(), ErrorCode::InvalidInput);
    assert_eq!(
        non_finite_x
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds x")
    );
    assert_eq!(
        non_finite_y
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds y")
    );
    assert_eq!(
        non_finite_width
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds width")
    );
    assert_eq!(
        non_finite_height
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds height")
    );
    assert_eq!(
        zero_width
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds width")
    );
    assert_eq!(
        zero_height
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("text run ink bounds height")
    );
    assert_eq!(
        UnresolvedResourceKind::TextRunInkBounds.label(),
        "text run ink bounds"
    );

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Bounded text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        ink,
    )
    .unwrap();
    let shadowed = TextShadowRun::try_new(
        run,
        ShadowList::try_new(vec![
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();

    assert_eq!(shadowed.run().bounds(), ink);
}

#[test]
fn scene_lowering_preserves_authored_text_run_bounds() {
    let bounds = TextRunBounds::try_ink(Rect::new(-2.0, -3.0, 4.0, 5.0)).unwrap();
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Bounded scene text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        bounds,
    )
    .unwrap();

    let mut scene = Scene::new();
    scene.text_run(run);

    let [
        scene::Command::TextRun {
            bounds: scene_bounds,
            ..
        },
    ] = scene.commands.as_slice()
    else {
        panic!("direct text run should retain authored bounds in the scene");
    };
    assert_eq!(*scene_bounds, bounds);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let [
        command::RenderCommand::TextRun {
            bounds: normalized_bounds,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("direct text run should retain authored bounds after normalization");
    };
    assert_eq!(*normalized_bounds, bounds);

    let shadowed_run = TextRun::try_new(
        FontRef::new(1).named("Bounded shadow text"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        bounds,
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut shadow_scene = Scene::new();
    shadow_scene.text_shadow_run(TextShadowRun::try_new(shadowed_run, shadows).unwrap());

    let [
        scene::Command::TextShadowRun {
            bounds: shadow_bounds,
            ..
        },
    ] = shadow_scene.commands.as_slice()
    else {
        panic!("text shadow run should retain wrapped authored bounds in the scene");
    };
    assert_eq!(*shadow_bounds, bounds);
}

fn ahem_font(name: &'static str) -> FontRef<'static> {
    FontRef::new(AHEM_FONT_ID)
        .named(name)
        .with_data(FontData::from_bytes(AHEM_FONT_BYTES.to_vec(), 0))
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use super::surface::{PresentedLifecycle, ResizeState};
#[test]
fn scene_encoding_is_deterministic() {
    let mut a = Scene::new();
    let mut b = Scene::new();
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    a.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::try_new(1.0).unwrap(), Color::BLACK);
    b.fill(rect, Color::BLACK)
        .stroke(rect, Stroke::try_new(1.0).unwrap(), Color::BLACK);

    assert_eq!(a, b);
}

#[test]
fn scene_stats_report_facts_without_renderer() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255])).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .stroke(
            Rect::new(1.0, 1.0, 2.0, 2.0),
            Stroke::try_new(1.0).unwrap(),
            Color::BLACK,
        )
        .shadow(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Shadow::try_new(Point::new(0.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        )
        .image(image, Rect::new(0.0, 0.0, 1.0, 1.0), ImageFit::Stretch)
        .layer(Layer::new(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });

    let stats = scene.stats();

    assert_eq!(stats.commands, 6);
    assert_eq!(stats.fills, 2);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.shadows, 1);
    assert_eq!(stats.images, 1);
    assert_eq!(stats.layers, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn reference_buffer_allocation_validates_positive_size_and_overflow() {
    let buffer = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 3)).unwrap();

    assert_eq!(buffer.physical_size(), PhysicalSize::new(2, 3));
    assert_eq!(buffer.byte_len(), 24);
    assert_eq!(buffer.pixel(1, 2).unwrap(), PremultipliedRgba8::TRANSPARENT);

    let zero_width = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(0, 1))
        .expect_err("zero-width reference buffers should be rejected");
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);

    let overflow =
        ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(u32::MAX, u32::MAX))
            .expect_err("overflow-sized reference buffers should be rejected before allocation");
    assert_eq!(overflow.code(), ErrorCode::InvalidInput);

    let wrong_data_len = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![PremultipliedRgba8::TRANSPARENT],
    )
    .expect_err("raw pixel data should match validated dimensions");
    assert_eq!(wrong_data_len.code(), ErrorCode::InvalidInput);
}

#[test]
fn reference_buffer_pixel_access_preserves_bounds_checks() {
    let mut buffer = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let pixel = PremultipliedRgba8::try_new(10, 20, 30, 40).unwrap();

    buffer.set_pixel(1, 1, pixel).unwrap();

    assert_eq!(buffer.pixel(1, 1).unwrap(), pixel);
    assert_eq!(
        buffer
            .pixel(2, 0)
            .expect_err("x outside width should fail")
            .code(),
        ErrorCode::InvalidInput
    );
    assert_eq!(
        buffer
            .set_pixel(0, 2, pixel)
            .expect_err("y outside height should fail")
            .code(),
        ErrorCode::InvalidInput
    );
}

#[test]
fn reference_premultiplied_pixels_apply_clamped_finite_opacity() {
    let pixel = PremultipliedRgba8::try_new(100, 60, 20, 200).unwrap();

    let invalid_pixel =
        PremultipliedRgba8::try_new(200, 0, 0, 128).expect_err("red must be premultiplied");
    assert_eq!(invalid_pixel.code(), ErrorCode::InvalidInput);

    assert_eq!(
        pixel.apply_opacity(0.5).unwrap(),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
    assert_eq!(pixel.apply_opacity(3.0).unwrap(), pixel);
    assert_eq!(
        pixel.apply_opacity(-1.0).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
    assert_eq!(
        pixel
            .apply_opacity(f32::NAN)
            .expect_err("non-finite opacity should be rejected")
            .code(),
        ErrorCode::InvalidInput
    );

    let buffer = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![pixel, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    assert_eq!(
        buffer.apply_opacity(0.5).unwrap().pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
}

#[test]
fn reference_source_over_composition_handles_alpha_edges() {
    let destination = PremultipliedRgba8::try_new(20, 40, 60, 128).unwrap();
    assert_eq!(
        PremultipliedRgba8::TRANSPARENT.source_over(destination),
        destination
    );

    let source = PremultipliedRgba8::try_new(20, 10, 5, 64).unwrap();
    assert_eq!(source.source_over(PremultipliedRgba8::TRANSPARENT), source);

    let opaque_source = PremultipliedRgba8::try_new(120, 80, 40, 255).unwrap();
    assert_eq!(opaque_source.source_over(destination), opaque_source);

    let partial_source = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let partial_destination = PremultipliedRgba8::try_new(0, 64, 0, 128).unwrap();
    assert_eq!(
        partial_source.source_over(partial_destination),
        PremultipliedRgba8::try_new(128, 32, 0, 192).unwrap()
    );
}

#[test]
fn reference_buffer_source_over_preserves_transparent_edges() {
    let mut source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let mut destination =
        ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 2)).unwrap();
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let green = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();

    source.set_pixel(0, 0, red).unwrap();
    destination.set_pixel(1, 1, green).unwrap();
    let composed = source.source_over(&destination).unwrap();

    assert_eq!(composed.pixel(0, 0).unwrap(), red);
    assert_eq!(composed.pixel(1, 1).unwrap(), green);
    assert_eq!(
        composed.pixel(0, 1).unwrap(),
        PremultipliedRgba8::TRANSPARENT
    );
}

#[test]
fn private_composite_reference_helpers_cover_current_internal_operator_boundary() {
    let source = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();
    let mask = PremultipliedRgba8::try_new(0, 0, 0, 64).unwrap();
    let source_over = source.source_over(destination);
    let blend_normal = source.blend_over(destination, BlendMode::Normal);
    let blend_plus = source.blend_over(destination, BlendMode::Plus);
    let source_in = source.source_in_alpha_of(mask);
    let destination_in = destination.destination_in_alpha_of(mask);

    assert_eq!(
        source_over,
        PremultipliedRgba8::try_new(128, 64, 0, 192).unwrap()
    );
    assert_eq!(
        blend_normal, source_over,
        "normal blend-over remains the same private source-over operator"
    );
    assert_eq!(
        blend_plus,
        PremultipliedRgba8::try_new(128, 128, 0, 255).unwrap()
    );
    assert_eq!(
        source_in,
        PremultipliedRgba8::try_new(32, 0, 0, 32).unwrap()
    );
    assert_eq!(
        destination_in,
        PremultipliedRgba8::try_new(0, 32, 0, 32).unwrap()
    );

    for pixel in [
        source_over,
        blend_normal,
        blend_plus,
        source_in,
        destination_in,
    ] {
        assert_premultiplied(pixel);
    }
}

#[test]
fn reference_pixels_apply_plus_lighter_and_blend_modes_deterministically() {
    let source = PremultipliedRgba8::try_new(60, 30, 10, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 80, 40, 160).unwrap();
    let cases = [
        (
            BlendMode::Normal,
            PremultipliedRgba8::try_new(70, 70, 30, 208).unwrap(),
        ),
        (
            BlendMode::Plus,
            PremultipliedRgba8::try_new(80, 110, 50, 255).unwrap(),
        ),
        (
            BlendMode::Multiply,
            PremultipliedRgba8::try_new(37, 60, 25, 208).unwrap(),
        ),
        (
            BlendMode::Screen,
            PremultipliedRgba8::try_new(75, 101, 48, 208).unwrap(),
        ),
        (
            BlendMode::Overlay,
            PremultipliedRgba8::try_new(42, 70, 27, 208).unwrap(),
        ),
        (
            BlendMode::Darken,
            PremultipliedRgba8::try_new(42, 70, 30, 208).unwrap(),
        ),
        (
            BlendMode::Lighten,
            PremultipliedRgba8::try_new(70, 91, 44, 208).unwrap(),
        ),
    ];

    for (mode, expected) in cases {
        let blended = source.blend_over(destination, mode);

        assert_eq!(blended, expected, "unexpected {mode:?} blend result");
        assert_premultiplied(blended);
    }
}

#[test]
fn reference_blends_handle_transparent_and_opaque_alpha_edges() {
    let transparent = PremultipliedRgba8::TRANSPARENT;
    let source = PremultipliedRgba8::try_new(64, 32, 16, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 80, 40, 160).unwrap();
    let opaque_source = PremultipliedRgba8::try_new(200, 100, 50, 255).unwrap();
    let opaque_destination = PremultipliedRgba8::try_new(50, 150, 200, 255).unwrap();

    assert_eq!(
        transparent.blend_over(destination, BlendMode::Multiply),
        destination
    );
    assert_eq!(source.blend_over(transparent, BlendMode::Screen), source);
    assert_eq!(
        opaque_source.blend_over(opaque_destination, BlendMode::Multiply),
        PremultipliedRgba8::try_new(39, 59, 39, 255).unwrap()
    );
    assert_eq!(
        opaque_source.blend_over(opaque_destination, BlendMode::Overlay),
        PremultipliedRgba8::try_new(78, 127, 167, 255).unwrap()
    );
}

#[test]
fn reference_buffer_blend_over_rejects_mismatched_destination_size() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let error = source
        .blend_over(&destination, BlendMode::Multiply)
        .expect_err("blend buffers must map one-to-one to destination pixels");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("reference blend destination size")
    );
}

#[test]
fn reference_buffer_alpha_composites_reject_mismatched_buffer_sizes() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let source_in_error = source
        .source_in_alpha_of(&destination)
        .expect_err("source-in buffers must map one-to-one to destination alpha");
    let source_in_diagnostic = source_in_error
        .invalid_value_diagnostic()
        .expect("source-in mismatch should include invalid value details");

    assert_eq!(source_in_error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        source_in_diagnostic.field(),
        "reference source-in destination size"
    );
    assert_eq!(source_in_diagnostic.value(), "1x2");
    assert_eq!(source_in_diagnostic.invariant(), "must match source size");

    let destination_in_error = destination
        .destination_in_alpha_of(&source)
        .expect_err("destination-in buffers must map one-to-one to source alpha");
    let destination_in_diagnostic = destination_in_error
        .invalid_value_diagnostic()
        .expect("destination-in mismatch should include invalid value details");

    assert_eq!(destination_in_error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        destination_in_diagnostic.field(),
        "reference destination-in source size"
    );
    assert_eq!(destination_in_diagnostic.value(), "2x1");
    assert_eq!(
        destination_in_diagnostic.invariant(),
        "must match destination size"
    );
}

#[test]
fn reference_buffer_blend_over_and_alpha_composites_cover_partial_masks() {
    let red_half = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let green_half = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();
    let blue_opaque = PremultipliedRgba8::try_new(0, 0, 255, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![red_half, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![green_half, blue_opaque],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ],
    )
    .unwrap();

    let blended = source.blend_over(&destination, BlendMode::Lighten).unwrap();
    let source_in = source.source_in_alpha_of(&mask).unwrap();
    let destination_in = destination.destination_in_alpha_of(&mask).unwrap();

    assert_eq!(
        blended.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(128, 128, 0, 192).unwrap()
    );
    assert_eq!(blended.pixel(1, 0).unwrap(), blue_opaque);
    assert_eq!(
        source_in.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(64, 0, 0, 64).unwrap()
    );
    assert_eq!(
        destination_in.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(0, 64, 0, 64).unwrap()
    );
}

#[test]
fn reference_pixels_apply_source_in_and_destination_in_alpha_multiplication() {
    let source = PremultipliedRgba8::try_new(100, 60, 20, 200).unwrap();
    let destination = PremultipliedRgba8::try_new(0, 80, 40, 128).unwrap();

    assert_eq!(
        source.source_in_alpha_of(destination),
        PremultipliedRgba8::try_new(50, 30, 10, 100).unwrap()
    );
    assert_eq!(
        destination.destination_in_alpha_of(source),
        PremultipliedRgba8::try_new(0, 63, 31, 100).unwrap()
    );
}

#[test]
fn reference_alpha_masks_handle_opaque_transparent_and_partial_mask_pixels() {
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let green = PremultipliedRgba8::try_new(0, 128, 0, 128).unwrap();
    let blue = PremultipliedRgba8::try_new(0, 0, 200, 200).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![red, green, blue],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(3, 1),
        vec![
            PremultipliedRgba8::try_new(255, 255, 255, 255).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(16, 8, 4, 64).unwrap(),
        ],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(masked.pixel(0, 0).unwrap(), red);
    assert_eq!(masked.pixel(1, 0).unwrap(), PremultipliedRgba8::TRANSPARENT);
    assert_eq!(
        masked.pixel(2, 0).unwrap(),
        PremultipliedRgba8::try_new(0, 0, 50, 50).unwrap()
    );
}

#[test]
fn reference_alpha_masks_preserve_premultiplied_color_ratios() {
    let source_pixel = PremultipliedRgba8::try_new(100, 50, 25, 200).unwrap();
    let source =
        ReferencePremultipliedRgba8Buffer::from_pixels(PhysicalSize::new(1, 1), vec![source_pixel])
            .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(1, 1),
        vec![PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap()],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(
        masked.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(50, 25, 13, 100).unwrap()
    );
    assert_premultiplied(masked.pixel(0, 0).unwrap());
}

#[test]
fn reference_alpha_masks_preserve_transparent_edges() {
    let red = PremultipliedRgba8::try_new(255, 0, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::TRANSPARENT,
            red,
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
        ],
    )
    .unwrap();

    let masked = source.apply_alpha_mask(&mask).unwrap();

    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(
                masked.pixel(x, y).unwrap(),
                PremultipliedRgba8::TRANSPARENT,
                "unexpected masked edge at {x},{y}"
            );
        }
    }
}

#[test]
fn reference_alpha_masks_are_deterministic_across_repeated_runs() {
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(100, 20, 10, 100).unwrap(),
            PremultipliedRgba8::try_new(0, 64, 128, 128).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
            PremultipliedRgba8::try_new(10, 40, 80, 160).unwrap(),
        ],
    )
    .unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 2),
        vec![
            PremultipliedRgba8::try_new(0, 0, 0, 255).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 128).unwrap(),
            PremultipliedRgba8::try_new(0, 0, 0, 64).unwrap(),
            PremultipliedRgba8::TRANSPARENT,
        ],
    )
    .unwrap();

    let first = source.apply_alpha_mask(&mask).unwrap();
    let second = source.apply_alpha_mask(&mask).unwrap();

    assert_eq!(first, second);
}

#[test]
fn reference_alpha_masks_reject_mismatched_mask_buffer_size() {
    let source = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(2, 1)).unwrap();
    let mask = ReferencePremultipliedRgba8Buffer::try_new(PhysicalSize::new(1, 2)).unwrap();

    let error = source
        .apply_alpha_mask(&mask)
        .expect_err("mask buffers must map one-to-one to source pixels");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("reference alpha mask size")
    );
}

#[test]
fn resolved_alpha_mask_execution_applies_materialized_alpha_buffer() {
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255,
        ],
    };
    let mask = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![
            0, 0, 0, 255, //
            0, 0, 0, 0, //
            0, 0, 0, 128,
        ],
    };

    let masked = ResolvedAlphaMaskExecution::try_new(&source, &mask)
        .unwrap()
        .execute_to_image_buffer()
        .unwrap();

    assert_eq!(masked.size, source.size);
    assert_eq!(
        masked.rgba,
        vec![
            255, 0, 0, 255, //
            0, 0, 0, 0, //
            0, 0, 255, 128,
        ]
    );
}

#[test]
fn resolved_alpha_mask_execution_rejects_non_materialized_luminance_policy() {
    let source = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 0, 0, 255],
    };
    let mask = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 255, 255, 255],
    };

    let error = ResolvedAlphaMaskExecution::try_new_with_mode(&source, &mask, MaskMode::Luminance)
        .expect_err("luminance masks need an explicit conversion policy before execution");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );
}

#[test]
fn resolved_alpha_mask_execution_rejects_mismatched_buffers() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 255, 0, 255],
    };
    let mask = ImageBuffer {
        size: PhysicalSize::new(1, 2),
        rgba: vec![0, 0, 0, 255, 0, 0, 0, 255],
    };

    let error = ResolvedAlphaMaskExecution::try_new(&source, &mask)
        .expect_err("materialized alpha masks must match source buffer size");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("resolved alpha mask size")
    );
}

#[test]
fn layer_resolved_alpha_mask_applies_after_children_before_parent_composite() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![
            255, 255, 255, 255, //
            255, 255, 255, 128, //
            0, 0, 0, 0,
        ],
    };
    let layer = Layer::new().try_resolved_alpha_mask(mask).unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 3.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        );
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

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
    let inner_mask = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 255, 255, 255, 255, 255, 255, 128],
    };
    let outer_mask = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 255, 255, 128, 255, 255, 255, 255],
    };
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_resolved_alpha_mask(outer_mask).unwrap(),
        |scene| {
            scene.layer(
                Layer::new().try_resolved_alpha_mask(inner_mask).unwrap(),
                |scene| {
                    scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
                },
            );
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert!((96..=160).contains(&pixel_alpha(&output, 0, 0)));
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

#[test]
fn layer_resolved_alpha_mask_respects_layer_clip_before_masking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 255, 255, 255, 255, 255, 255, 255],
    };
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(1.0, 0.0, 2.0, 1.0)))
        .unwrap()
        .try_resolved_alpha_mask(mask)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert!(pixel_alpha(&output, 2, 0) > 200);
}

#[test]
fn layer_resolved_alpha_mask_composites_after_layer_transform() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 255, 255, 255],
    };
    let layer = Layer::new()
        .try_transform(Transform::translation(1.0, 0.0).unwrap())
        .unwrap()
        .try_resolved_alpha_mask(mask)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 200);
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn layer_resolved_alpha_mask_combines_mask_child_opacity_and_layer_opacity() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    let mask = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 255, 255, 128],
    };
    let layer = Layer::new()
        .try_opacity(0.5)
        .unwrap()
        .try_resolved_alpha_mask(mask)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();
    let alpha = pixel_alpha(&output, 0, 0);

    assert!((24..=40).contains(&alpha), "unexpected alpha {alpha}");
}

#[test]
fn layer_resolved_alpha_mask_rejects_luminance_mode_without_conversion_policy() {
    let mask = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 255, 255, 255],
    };

    let error = ResolvedLayerAlphaMask::try_new_with_mode(mask, MaskMode::Luminance)
        .expect_err("resolved layer masks do not implement luminance conversion");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );
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
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![90, 120, 150, 0, 64, 128, 255, 128, 255, 10, 20, 255],
    };

    let premultiplied =
        image::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
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
        image::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&premultiplied)
            .unwrap();

    assert_eq!(straight.size, PhysicalSize::new(3, 1));
    assert_eq!(
        straight.rgba,
        vec![0, 0, 0, 0, 64, 128, 255, 128, 255, 10, 20, 255]
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

    let filtered = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
        .unwrap()
        .execute_to_image()
        .unwrap();

    assert_eq!(filtered.size(), Size::new(1.0, 1.0));
    assert_eq!(filtered.bytes.as_ref(), &[50, 75, 100, 255]);
}

#[test]
fn image_color_filter_execution_applies_color_chain_to_multi_pixel_buffer() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 2),
        rgba: vec![
            64, 128, 255, 128, 10, 20, 30, 0, 100, 150, 200, 255, 20, 40, 80, 64,
        ],
    };
    let filters = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(0.5).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
    ])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size, PhysicalSize::new(2, 2));
    assert_eq!(
        filtered.rgba,
        vec![
            32, 64, 128, 64, 0, 0, 0, 0, 50, 76, 100, 128, 16, 24, 40, 32,
        ]
    );
}

#[test]
fn image_color_filter_execution_preserves_buffer_size_and_rgba_order() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![10, 20, 30, 40, 200, 150, 100, 255],
    };
    let filters = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size, PhysicalSize::new(2, 1));
    assert_eq!(filtered.rgba, vec![13, 19, 32, 40, 200, 150, 100, 255]);
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

    let filtered = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
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

    let transparent = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
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

    let blurred = image::ResolvedImageColorFilterExecution::try_new(&opaque_paint, &opaque)
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
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0],
    };
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();

    let blurred =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(blurred.size, PhysicalSize::new(3, 1));
    assert_eq!(
        blurred.rgba,
        vec![255, 0, 0, 25, 255, 0, 0, 41, 255, 0, 0, 25]
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

    let filtered = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
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
        image::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        image::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn materialized_image_filters_preserve_color_and_blur_order() {
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 1),
        rgba: vec![200, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0],
    };
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let color_before_blur = FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap();
    let blur_before_color = FilterList::try_ops(vec![blur, brightness]).unwrap();

    let color_before = image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_blur,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let blur_before = image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &blur_before_color,
        &source,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();

    assert_eq!(color_before.size, PhysicalSize::new(3, 1));
    assert_eq!(blur_before.size, PhysicalSize::new(3, 1));
    assert_ne!(color_before.rgba, blur_before.rgba);
}

#[test]
fn materialized_image_blur_keeps_output_clipped_to_source_region() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 2),
        rgba: vec![255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    };
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(4.0).unwrap())]).unwrap();

    let blurred =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(
        blurred.size, source.size,
        "materialized image blur inflates for sampling but clips output to source image extent"
    );
    assert_eq!(blurred.rgba.len(), source.rgba.len());
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
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 3),
        rgba: vec![
            0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    };
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size, PhysicalSize::new(3, 3));
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
    let source = ImageBuffer {
        size: PhysicalSize::new(3, 3),
        rgba: vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    };
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(1.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.size, source.size);
    assert_eq!(filtered.rgba.len(), source.rgba.len());
    assert_eq!(pixel_rgba(&filtered, 1, 1), [255, 255, 255, 255]);
    assert!(
        pixel_alpha(&filtered, 2, 0) > 0,
        "blurred offset shadow should contribute inside the clipped source extent"
    );
}

#[test]
fn materialized_drop_shadow_composites_shadow_behind_source() {
    let source = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 0, 0, 128],
    };
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba, vec![170, 0, 0, 192]);
}

#[test]
fn filtered_image_paint_executes_drop_shadow_with_matching_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters.clone(),
    )
    .unwrap();

    let filtered = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
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
        image::ResolvedImageColorFilterExecution::try_new(&wrong_id, &image)
            .expect_err("materialized image id should match resolved resource id")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image id")
    );
    assert_eq!(
        image::ResolvedImageColorFilterExecution::try_new(&wrong_size, &image)
            .expect_err("materialized image size should match resolved resource size")
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("materialized filtered image size")
    );
}

#[test]
fn resource_only_drop_shadow_filtered_image_paint_stays_rejected() {
    let resource = ResolvedImageResource::try_new(ImageId::new(41), Size::new(2.0, 1.0)).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();

    let unsupported = paint
        .ensure_supported(Capabilities::VELLO_0_9)
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
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 0, 0, 0],
    };
    let filters = FilterList::try_ops(vec![
        FilterOp::drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba, vec![0, 255, 255, 255, 255, 255, 255, 255]);
}

#[test]
fn materialized_filters_before_drop_shadow_shape_current_alpha_mask() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 0, 0, 0],
    };
    let filters = FilterList::try_ops(vec![
        FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap()),
        FilterOp::drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ),
    ])
    .unwrap();

    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered.rgba, vec![255, 0, 0, 128, 0, 0, 0, 128]);
}

#[test]
fn css_drop_shadow_rejects_non_zero_spread() {
    let source = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 0, 0, 255],
    };
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 1.0, Color::BLACK).unwrap(),
    )])
    .unwrap();

    let error =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .expect_err("CSS drop-shadow must not silently treat spread like box-shadow spread");

    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter drop-shadow spread")
    );
}

#[test]
fn materialized_drop_shadow_rejects_inset_shadow_with_typed_diagnostic() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 0, 0, 0],
    };
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_inset(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
    )])
    .unwrap();

    let error =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .expect_err("CSS drop-shadow must not execute inset shadows as outer alpha shadows");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
}

#[test]
fn css_drop_shadow_rejects_non_solid_shadow_paint() {
    let source = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![255, 0, 0, 255],
    };
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::drop_shadow(
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, Paint::gradient(gradient)).unwrap(),
    )])
    .unwrap();

    let error =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .expect_err("CSS drop-shadow currently requires a solid shadow paint");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
}

#[test]
fn sequence11_capabilities_advertise_narrow_materialized_filters_without_broad_effects() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(
        capabilities
            .filters()
            .supports_materialized_image_filter_classification()
    );
    assert!(
        capabilities
            .filters()
            .supports_materialized_blur_filter_execution()
    );
    assert!(
        capabilities
            .filters()
            .supports_materialized_drop_shadow_filter_execution()
    );
    assert!(
        capabilities
            .filters()
            .supports_filter_region_outset_planning()
    );
    assert!(
        capabilities
            .filters()
            .supports_cpu_reference_blur_fallback()
    );
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(!capabilities.shadows().supports_inset_box_shadows());
    assert!(!capabilities.shadows().supports_text_shadows());
    assert!(!capabilities.masks_clips().supports_layer_masks());
    assert!(
        capabilities
            .masks_clips()
            .supports_materialized_alpha_mask_execution()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_filter_execution()
    );
    assert!(!capabilities.offscreen_pipeline().supports_mask_execution());
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_backdrop_execution()
    );
}

#[test]
fn sequence11_filtered_image_executes_nonzero_blur_then_drop_shadow_with_materialized_image() {
    let image = Image::from_rgba(
        Size::new(2.0, 1.0),
        Arc::<[u8]>::from([255, 0, 0, 255, 0, 0, 0, 0]),
    )
    .unwrap();
    let filters = FilterList::try_ops(vec![
        FilterOp::blur(FilterBlur::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ),
    ])
    .unwrap();
    let paint = FilteredImagePaint::try_new(
        ResolvedImageResource::try_new(image.id(), image.size()).unwrap(),
        filters,
    )
    .unwrap();

    let filtered = image::ResolvedImageColorFilterExecution::try_new(&paint, &image)
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
fn sequence11_matrix_guardrails_cover_filter_shadow_and_diagnostic_rows() {
    let blur = FilterBlur::try_new(2.0).unwrap();
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 10.0, 4.0, 4.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(8.0, 8.0, 12.0, 12.0)).unwrap();
    let blur_outset = FilterOutset::from_blur(blur, BlurPolicy::css_filter_default()).unwrap();
    let blur_region = FilterRegionPlan::try_new(source, blur_outset, Some(clip)).unwrap();
    assert_eq!(
        blur_region.inflated_bounds().rect(),
        Rect::new(5.0, 5.0, 14.0, 14.0)
    );
    assert_eq!(
        blur_region.execution_region().rect(),
        Rect::new(8.0, 8.0, 11.0, 11.0)
    );

    let shadow = Shadow::try_new(Point::new(2.0, -1.0), 4.0, 0.0, Color::BLACK).unwrap();
    let shadow_outset =
        FilterOutset::from_drop_shadow(&shadow, BlurPolicy::css_filter_default()).unwrap();
    assert_eq!(shadow_outset.left(), 8.0);
    assert_eq!(shadow_outset.top(), 11.0);
    assert_eq!(shadow_outset.right(), 12.0);
    assert_eq!(shadow_outset.bottom(), 9.0);

    let image_buffer = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 0, 0, 0],
    };
    let color_before_pixel = FilterList::try_ops(vec![
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ),
    ])
    .unwrap();
    let pixel_before_color = FilterList::try_ops(vec![
        FilterOp::drop_shadow(
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ),
        FilterOp::invert(UnitFilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();
    let color_before = image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &color_before_pixel,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    let pixel_before = image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &pixel_before_color,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    assert_ne!(
        color_before.rgba, pixel_before.rgba,
        "mixed color and pixel-moving filters must preserve authored order"
    );

    let mut scene = Scene::new();
    scene.shadows(
        Rect::new(1.0, 1.0, 6.0, 6.0),
        ShadowList::try_new(vec![
            Shadow::try_new(Point::new(1.0, 0.0), 2.0, 1.0, Color::BLACK).unwrap(),
            Shadow::try_new(Point::new(-1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
        ])
        .unwrap(),
    );
    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    assert_eq!(normalized.stats().shadows, 2);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Shadow { .. },
            command::RenderCommand::Shadow { .. }
        ]
    ));

    let mut inset_scene = Scene::new();
    inset_scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );
    let inset_error = inset_scene.normalize(Capabilities::VELLO_0_9).unwrap_err();
    assert_eq!(
        inset_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let text_run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let text_shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut text_scene = Scene::new();
    text_scene.text_shadow_run(TextShadowRun::try_new(text_run, text_shadows).unwrap());
    let text_error = text_scene.normalize(Capabilities::VELLO_0_9).unwrap_err();
    assert_eq!(
        text_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(
        text_error
            .message()
            .contains("glyph-alpha/offscreen text capture")
    );

    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut non_solid_scene = Scene::new();
    non_solid_scene.shadow(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Paint::gradient(gradient)).unwrap(),
    );
    let non_solid_error = non_solid_scene
        .normalize(Capabilities::VELLO_0_9)
        .unwrap_err();
    assert_eq!(
        non_solid_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
}

#[test]
fn sequence10_matrix_color_filters_execute_with_cpu_reference_bytes() {
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
fn sequence10_matrix_filter_fusion_matches_reference_fallback_for_materialized_image() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![100, 150, 200, 255, 64, 128, 255, 128],
    };
    let filters = color_filter_list([
        ColorFilterOp::Brightness(FilterAmount::try_new(1.25).unwrap()),
        ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
        ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
    ]);
    let pipeline = filters
        .color_filter_pipeline()
        .unwrap()
        .expect("color-only filters should produce a sequence10 pipeline");
    let compiled = CompiledColorFilterPipeline::try_from_pipeline(&pipeline).unwrap();

    assert_eq!(compiled.executable_step_count(), 1);

    let premultiplied =
        image::straight_rgba8_image_buffer_to_premultiplied_rgba8_reference(&source).unwrap();
    let reference = premultiplied
        .apply_compiled_color_filter_pipeline(&compiled)
        .unwrap();
    let expected =
        image::premultiplied_rgba8_reference_to_straight_rgba8_image_buffer(&reference).unwrap();
    let filtered =
        image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(&filters, &source)
            .unwrap()
            .execute_to_image_buffer()
            .unwrap();

    assert_eq!(filtered, expected);
    assert_ne!(filtered.rgba, source.rgba);
}

#[test]
fn sequence10_capabilities_expose_only_granular_color_filter_execution() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(
        capabilities
            .filters()
            .supports_color_filter_classification()
    );
    assert!(
        capabilities
            .filters()
            .supports_color_filter_pipeline_execution()
    );
    assert!(
        capabilities
            .image_sampling()
            .supports_color_filtered_image_paint()
    );
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_filter_execution()
    );
}

#[test]
fn sequence10_guardrail_layer_effect_execution_stays_unsupported() {
    let image_buffer = ImageBuffer {
        size: PhysicalSize::new(1, 1),
        rgba: vec![100, 150, 200, 255],
    };
    let shadow = Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap();
    let drop_shadow = FilterList::try_ops(vec![FilterOp::drop_shadow(shadow)]).unwrap();

    let drop_shadow_output = image::ResolvedImageColorFilterExecution::try_new_for_image_buffer(
        &drop_shadow,
        &image_buffer,
    )
    .unwrap()
    .execute_to_image_buffer()
    .unwrap();
    assert_eq!(drop_shadow_output.size, image_buffer.size);

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
            PrimitiveOperation::BackdropExecution,
        ),
    ] {
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("later compositor execution should remain unsupported");

        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn sequence10_guardrail_unfiltered_images_stay_on_direct_sampling_path() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let mut scene = Scene::new();
    scene.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(scene.stats().images, 1);
    assert_eq!(scene.stats().layers, 0);
    assert!(matches!(
        normalized.commands.as_slice(),
        [command::RenderCommand::Image { .. }]
    ));

    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 10.0, 4.0),
        Size::new(2.0, 2.0),
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::contain(),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(3.0, 0.0, 4.0, 4.0));
}

fn color_filter_pipeline<const N: usize>(ops: [ColorFilterOp; N]) -> ColorFilterPipeline {
    color_filter_list(ops)
        .color_filter_pipeline()
        .unwrap()
        .unwrap()
}

fn color_filter_list<const N: usize>(ops: [ColorFilterOp; N]) -> FilterList {
    let ops = ops
        .into_iter()
        .map(|op| match op {
            ColorFilterOp::Brightness(amount) => FilterOp::brightness(amount),
            ColorFilterOp::Contrast(amount) => FilterOp::contrast(amount),
            ColorFilterOp::Grayscale(amount) => FilterOp::grayscale(amount),
            ColorFilterOp::HueRotate(angle) => FilterOp::hue_rotate(angle),
            ColorFilterOp::Invert(amount) => FilterOp::invert(amount),
            ColorFilterOp::Opacity(amount) => FilterOp::opacity(amount),
            ColorFilterOp::Saturate(amount) => FilterOp::saturate(amount),
            ColorFilterOp::Sepia(amount) => FilterOp::sepia(amount),
        })
        .collect();
    FilterList::try_ops(ops).unwrap()
}

fn normalize_single_layer_error(layer: Layer) -> Error {
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("Sequence 10 guardrail layer should reject during normalization")
}

fn assert_premultiplied(pixel: PremultipliedRgba8) {
    assert!(pixel.red() <= pixel.alpha());
    assert!(pixel.green() <= pixel.alpha());
    assert!(pixel.blue() <= pixel.alpha());
}

#[test]
fn texture_descriptor_equality_uses_size_format_and_intent() {
    let size = PhysicalSize::new(32, 16);
    let layer = TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::OffscreenLayer)
        .unwrap();
    let same = TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::OffscreenLayer)
        .unwrap();
    let different_intent =
        TextureDescriptor::try_new(size, Format::Rgba8, TextureUsageIntent::IntermediatePass)
            .unwrap();

    assert_eq!(layer, same);
    assert_ne!(layer, different_intent);
    assert_eq!(layer.physical_size(), size);
    assert_eq!(layer.format(), Format::Rgba8);
    assert_eq!(layer.intent(), TextureUsageIntent::OffscreenLayer);
    assert_eq!(layer.byte_len(), 32 * 16 * 4);
}

#[test]
fn texture_cache_keys_are_stable_without_raw_resources() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(8, 4),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();

    assert_eq!(
        TextureCacheKey::from_descriptor(descriptor),
        TextureCacheKey::from_descriptor(descriptor)
    );
    assert_ne!(
        TextureCacheKey::from_descriptor(descriptor),
        TextureCacheKey::from_descriptor(
            TextureDescriptor::try_new(
                PhysicalSize::new(8, 4),
                Format::Rgba8,
                TextureUsageIntent::ReadbackReference,
            )
            .unwrap()
        )
    );
}

#[test]
fn texture_cache_records_misses_reuse_hits_and_live_count() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(4, 4),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let mut cache = OffscreenTextureCache::new();

    let first = cache.acquire(descriptor).unwrap();
    assert_eq!(cache.stats().allocations, 1);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.live_count(), 1);

    cache.release(first).unwrap();
    let second = cache.acquire(descriptor).unwrap();

    assert_eq!(second.descriptor(), descriptor);
    assert_eq!(cache.stats().allocations, 1);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.live_count(), 1);
}

#[test]
fn texture_cache_release_and_eviction_accounting_is_deterministic() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(2, 2),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let mut cache = OffscreenTextureCache::new();

    let handle = cache.acquire(descriptor).unwrap();
    cache.release(handle).unwrap();
    let evicted = cache.evict_released();

    assert_eq!(evicted, 1);
    assert_eq!(cache.live_count(), 0);
    assert_eq!(cache.retained_count(), 0);
    assert_eq!(cache.stats().releases, 1);
    assert_eq!(cache.stats().evictions, 1);
}

#[test]
fn texture_cache_rejects_stale_handle_after_reuse() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(3, 3),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let mut cache = OffscreenTextureCache::new();

    let stale = cache.acquire(descriptor).unwrap();
    cache.release(stale).unwrap();
    let current = cache.acquire(descriptor).unwrap();
    let error = cache
        .release(stale)
        .expect_err("stale handles must not release a new lease");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(cache.live_count(), 1);
    assert_eq!(cache.stats().releases, 1);
    cache.release(current).unwrap();
    assert_eq!(cache.stats().releases, 2);
}

#[test]
fn texture_cache_rejects_same_descriptor_handle_from_another_cache() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(5, 5),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let mut first_cache = OffscreenTextureCache::new();
    let mut second_cache = OffscreenTextureCache::new();

    let foreign = first_cache.acquire(descriptor).unwrap();
    let local = second_cache.acquire(descriptor).unwrap();
    let error = second_cache
        .release(foreign)
        .expect_err("foreign handles must not release matching local entries");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(second_cache.live_count(), 1);
    assert_eq!(second_cache.stats().releases, 0);
    second_cache.release(local).unwrap();
    assert_eq!(second_cache.stats().releases, 1);
}

#[test]
fn texture_cache_default_construction_rejects_same_descriptor_foreign_release() {
    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(7, 7),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let mut first_cache = OffscreenTextureCache::default();
    let mut second_cache = OffscreenTextureCache::default();

    let foreign = first_cache.acquire(descriptor).unwrap();
    let local = second_cache.acquire(descriptor).unwrap();
    let error = second_cache
        .release(foreign)
        .expect_err("default-constructed caches must still have unique identities");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(second_cache.live_count(), 1);
    assert_eq!(second_cache.stats().releases, 0);
    second_cache.release(local).unwrap();
    assert_eq!(second_cache.stats().releases, 1);
}

#[test]
fn texture_descriptors_reject_zero_size_and_overflow() {
    let zero_width = TextureDescriptor::try_new(
        PhysicalSize::new(0, 1),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .expect_err("zero-width textures should be rejected");
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);

    let overflow = TextureDescriptor::try_new(
        PhysicalSize::new(u32::MAX, u32::MAX),
        Format::Rgba8,
        TextureUsageIntent::ReadbackReference,
    )
    .expect_err("overflow-sized textures should be rejected");
    assert_eq!(overflow.code(), ErrorCode::InvalidInput);
}

#[test]
fn headless_texture_descriptor_uses_allocation_size_without_surface_rewrite() {
    let zero_surface_descriptor =
        headless_texture_descriptor(PhysicalSize::new(0, 0), Format::Rgba8).unwrap();
    let nonzero_surface_descriptor =
        headless_texture_descriptor(PhysicalSize::new(12, 6), Format::Rgba8).unwrap();

    assert_eq!(
        zero_surface_descriptor.physical_size(),
        PhysicalSize::new(1, 1)
    );
    assert_eq!(
        zero_surface_descriptor.intent(),
        TextureUsageIntent::ReadbackReference
    );
    assert_eq!(
        nonzero_surface_descriptor.physical_size(),
        PhysicalSize::new(12, 6)
    );
}

#[test]
fn texture_lifecycle_accounting_is_separate_from_image_cache_stats() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 1.0, 1.0), Paint::image(image.clone()))
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Paint::image(image));
    let image_stats = scene.stats();

    let descriptor = TextureDescriptor::try_new(
        PhysicalSize::new(4, 4),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let mut cache = OffscreenTextureCache::new();
    let handle = cache.acquire(descriptor).unwrap();
    cache.release(handle).unwrap();
    let _ = cache.acquire(descriptor).unwrap();

    assert_eq!(image_stats.images, 2);
    assert_eq!(image_stats.cache_misses, 1);
    assert_eq!(image_stats.cache_hits, 1);
    assert_eq!(image_stats.uploaded_bytes, 4);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
}

#[test]
fn shader_pass_descriptor_names_textures_bounds_and_kind() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(16, 8),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(16, 8),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(2, 1, 6, 4, source, destination).unwrap();

    let pass = RectShaderPassDescriptor::try_new(
        "layer-source",
        "layer-destination",
        source,
        destination,
        bounds,
        RectShaderPassKind::IdentityCopy,
    )
    .unwrap();

    assert_eq!(pass.source_label(), "layer-source");
    assert_eq!(pass.destination_label(), "layer-destination");
    assert_eq!(pass.source(), source);
    assert_eq!(pass.destination(), destination);
    assert_eq!(pass.bounds(), bounds);
    assert_eq!(pass.bounds().x(), 2);
    assert_eq!(pass.bounds().y(), 1);
    assert_eq!(pass.kind(), RectShaderPassKind::IdentityCopy);
}

#[test]
fn shader_pipeline_key_is_stable_and_distinct_from_texture_cache_keys() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(0, 0, 8, 8, source, destination).unwrap();
    let pass = RectShaderPassDescriptor::try_new(
        "source",
        "destination",
        source,
        destination,
        bounds,
        RectShaderPassKind::ClearFill,
    )
    .unwrap();

    let key = RectShaderPipelineKey::from_descriptor(pass);
    assert_eq!(key, RectShaderPipelineKey::from_descriptor(pass));
    assert_ne!(
        key,
        RectShaderPipelineKey::from_descriptor(
            RectShaderPassDescriptor::try_new(
                "source",
                "destination",
                source,
                destination,
                bounds,
                RectShaderPassKind::IdentityCopy,
            )
            .unwrap()
        )
    );
    assert_ne!(
        format!("{key:?}"),
        format!("{:?}", TextureCacheKey::from_descriptor(destination))
    );
}

#[test]
fn shader_rect_bounds_reject_zero_and_out_of_range_regions() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(4, 4),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(4, 4),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();

    let zero_width = RectPassBounds::try_new(0, 0, 0, 1, source, destination)
        .expect_err("zero-width shader bounds should be rejected");
    assert_eq!(zero_width.code(), ErrorCode::InvalidInput);

    let source_overflow = RectPassBounds::try_new(3, 0, 2, 1, source, destination)
        .expect_err("source bounds must fit source texture");
    assert_eq!(source_overflow.code(), ErrorCode::InvalidInput);

    let destination_overflow = RectPassBounds::try_new(
        0,
        3,
        1,
        2,
        source,
        TextureDescriptor::try_new(
            PhysicalSize::new(4, 3),
            Format::Rgba8,
            TextureUsageIntent::IntermediatePass,
        )
        .unwrap(),
    )
    .expect_err("destination bounds must fit destination texture");
    assert_eq!(destination_overflow.code(), ErrorCode::InvalidInput);
}

#[test]
fn shader_pass_descriptor_revalidates_bounds_against_named_textures() {
    let large_source = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let large_destination = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let smaller_destination = TextureDescriptor::try_new(
        PhysicalSize::new(4, 4),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(3, 3, 2, 2, large_source, large_destination).unwrap();

    let error = RectShaderPassDescriptor::try_new(
        "source",
        "smaller-destination",
        large_source,
        smaller_destination,
        bounds,
        RectShaderPassKind::ClearFill,
    )
    .expect_err("descriptor must revalidate bounds against its own destination");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rect shader pass destination x extent")
    );
}

#[test]
fn shader_clear_fill_descriptor_rejects_partial_destination_bounds() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(8, 8),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(2, 1, 4, 4, source, destination).unwrap();

    let error = RectShaderPassDescriptor::try_new(
        "source",
        "destination",
        source,
        destination,
        bounds,
        RectShaderPassKind::ClearFill,
    )
    .expect_err("clear/fill uses attachment clear and must be fullscreen");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rect shader clear/fill bounds")
    );
}

#[test]
fn shader_pass_contract_only_context_reports_adapter_unavailable() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(2, 2),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(2, 2),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(0, 0, 2, 2, source, destination).unwrap();
    let pass = RectShaderPassDescriptor::try_new(
        "source",
        "destination",
        source,
        destination,
        bounds,
        RectShaderPassKind::ClearFill,
    )
    .unwrap();

    let error = pollster::block_on(encode_clear_fill_pass(
        RectShaderPassExecution::contract_only(),
        pass,
        Color::BLACK,
    ))
    .expect_err("contract-only shader pass should report missing GPU context");

    assert_eq!(error.code(), ErrorCode::AdapterUnavailable);
    assert!(error.message().contains("rect/fullscreen shader pass"));
}

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

#[test]
fn offscreen_texture_allocation_uses_explicit_bounded_layer_descriptor() {
    let bounds = command::OffscreenBounds::try_new(Rect::new(2.0, 3.0, 10.0, 6.0)).unwrap();

    let descriptor = offscreen_local_scene_texture_descriptor(bounds, 2.0, Format::Rgba8).unwrap();

    assert_eq!(descriptor.physical_size(), PhysicalSize::new(20, 12));
    assert_eq!(descriptor.format(), Format::Rgba8);
    assert_eq!(descriptor.intent(), TextureUsageIntent::OffscreenLayer);
}

#[test]
fn offscreen_texture_rejects_missing_gpu_context_with_adapter_diagnostic() {
    let mut cache = OffscreenTextureResourceCache::new();
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 1.0, 1.0)).unwrap();
    let mut scene = vello::Scene::new();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 1.0, 1.0),
    );

    let error = render_vello_local_scene_to_offscreen_texture(
        None,
        Options::default(),
        &mut cache,
        &scene,
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default()),
    )
    .expect_err("contract-only offscreen render should report missing GPU context");

    assert_eq!(error.code(), ErrorCode::AdapterUnavailable);
    assert!(error.message().contains("offscreen Vello local scene"));
    assert_eq!(cache.stats().allocations, 0);
    assert_eq!(cache.live_count(), 0);
}

#[test]
fn offscreen_local_vello_scene_renders_to_texture_when_gpu_context_is_available() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let bounds = command::OffscreenBounds::try_new(Rect::new(12.0, 8.0, 2.0, 2.0)).unwrap();
    let mut scene = vello::Scene::new();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 2.0, 2.0),
    );
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let options = renderer.options();
    let mut cache = OffscreenTextureResourceCache::new();
    let Some(context) = renderer.default_offscreen_render_context() else {
        let error = render_vello_local_scene_to_offscreen_texture(
            None, options, &mut cache, &scene, request,
        )
        .expect_err("no GPU machines should report the explicit diagnostic");
        assert_eq!(error.code(), ErrorCode::AdapterUnavailable);
        return;
    };

    let output = render_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &mut cache,
        &scene,
        request,
    )
    .unwrap();
    assert_eq!(output.target().bounds(), bounds);
    assert_eq!(output.target().resource_id(), 1);
    assert_eq!(
        output.target().descriptor().physical_size(),
        PhysicalSize::new(2, 2)
    );
    assert_eq!(output.timings().present_time, Duration::ZERO);
    assert_eq!(cache.stats().allocations, 1);
    let view_debug = format!("{:?}", output.view());
    assert!(!view_debug.is_empty());

    let (device, queue) = renderer.default_wgpu_device_queue().unwrap();
    let image = read_texture_rgba(
        device,
        queue,
        output.texture(),
        output.target().descriptor().physical_size(),
    )
    .unwrap();
    assert!(pixel_alpha(&image, 0, 0) > 0);

    output.release(&mut cache).unwrap();
    assert_eq!(cache.live_count(), 0);
    assert_eq!(cache.released_resource_count(), 1);
}

#[test]
fn offscreen_local_scene_texture_descriptor_rejects_bgra8_for_vello_target() {
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 2.0, 2.0)).unwrap();
    let error = offscreen_local_scene_texture_descriptor(bounds, 1.0, Format::Bgra8)
        .expect_err("minimal offscreen Vello targets are Rgba8-only");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("offscreen Vello scene texture format")
    );
}

#[test]
fn offscreen_bgra8_render_request_rejects_without_cache_allocation() {
    let mut cache = OffscreenTextureResourceCache::new();
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 2.0, 2.0)).unwrap();
    let scene = vello::Scene::new();
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Bgra8, Parameters::default());

    let error = render_vello_local_scene_to_offscreen_texture(
        None,
        Options::default(),
        &mut cache,
        &scene,
        request,
    )
    .expect_err("Bgra8 should be rejected before GPU context allocation");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(cache.stats().allocations, 0);
    assert_eq!(cache.live_count(), 0);
}

#[test]
fn offscreen_rect_shader_pass_descriptor_targets_offscreen_textures() {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(4, 3),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(4, 3),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(0, 0, 4, 3, source, destination).unwrap();

    let pass = RectShaderPassDescriptor::try_new(
        "offscreen-layer",
        "intermediate-pass",
        source,
        destination,
        bounds,
        RectShaderPassKind::IdentityCopy,
    )
    .unwrap();

    assert_eq!(pass.source().intent(), TextureUsageIntent::OffscreenLayer);
    assert_eq!(
        pass.destination().intent(),
        TextureUsageIntent::IntermediatePass
    );
    assert_eq!(pass.kind(), RectShaderPassKind::IdentityCopy);
}

#[test]
fn offscreen_nested_layer_opacity_stays_on_direct_vello_surface_path() {
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.75).unwrap(), |scene| {
        scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        });
    });
    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[0]
    else {
        panic!("expected outer opacity layer");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[0] else {
        panic!("expected inner opacity layer");
    };

    assert_eq!(
        outer.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
    assert_eq!(
        inner.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
    assert!(!outer.pass_plan.requires_offscreen_texture());
    assert!(!inner.pass_plan.requires_offscreen_texture());

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();
    let alpha = pixel_alpha(&output, 0, 0);

    assert_eq!(stats.layers, 2);
    assert!(alpha > 0);
    assert!(alpha < 255);
}

#[test]
fn offscreen_reuses_resources_across_repeated_bounded_requests() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 3.0, 2.0)).unwrap();
    let mut scene = vello::Scene::new();
    scene.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::IDENTITY,
        peniko::Color::BLACK,
        None,
        &kurbo::Rect::new(0.0, 0.0, 3.0, 2.0),
    );
    let request =
        OffscreenLocalSceneRenderRequest::new(bounds, 1.0, Format::Rgba8, Parameters::default());
    let options = renderer.options();
    let mut cache = OffscreenTextureResourceCache::new();
    let Some(context) = renderer.default_offscreen_render_context() else {
        let error = render_vello_local_scene_to_offscreen_texture(
            None, options, &mut cache, &scene, request,
        )
        .expect_err("no GPU machines should report the explicit diagnostic");
        assert_eq!(error.code(), ErrorCode::AdapterUnavailable);
        return;
    };
    let first = render_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &mut cache,
        &scene,
        request,
    )
    .unwrap();
    let first_resource_id = first.target().resource_id();
    let first_descriptor = first.target().descriptor();
    first.release(&mut cache).unwrap();

    let context = renderer.default_offscreen_render_context().unwrap();
    let second = render_vello_local_scene_to_offscreen_texture(
        Some(context),
        options,
        &mut cache,
        &scene,
        request,
    )
    .unwrap();

    assert_eq!(second.target().descriptor(), first_descriptor);
    assert_eq!(second.target().resource_id(), first_resource_id);
    assert_eq!(cache.stats().allocations, 1);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.live_count(), 1);
    assert_eq!(cache.released_resource_count(), 0);
    second.release(&mut cache).unwrap();
    assert_eq!(cache.live_count(), 0);
    assert_eq!(cache.released_resource_count(), 1);
}

#[test]
fn offscreen_no_allocation_when_layer_isolation_is_unnecessary() {
    let mut scene = Scene::new();
    scene.layer(Layer::new(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });
    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };
    let cache = OffscreenTextureCache::new();

    assert_eq!(layer.pass_plan.kind(), command::LayerPassKind::None);
    assert!(!layer.pass_plan.requires_offscreen_texture());
    assert_eq!(cache.stats().allocations, 0);
    assert_eq!(cache.live_count(), 0);
}

#[test]
fn sequence9_offscreen_guardrail_direct_vello_rendering_matches_ordinary_scene_baseline() {
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(
            Rect::new(2.0, 0.0, 2.0, 2.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    assert!(
        normalized
            .commands
            .iter()
            .all(|command| { !matches!(command, command::RenderCommand::Layer { .. }) })
    );

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut first_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut second_surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();

    let first_stats =
        pollster::block_on(renderer.render(&mut first_surface, &scene, Parameters::default()))
            .unwrap();
    let first_output = renderer.read_headless(&first_surface).unwrap();
    let second_stats =
        pollster::block_on(renderer.render(&mut second_surface, &scene, Parameters::default()))
            .unwrap();
    let second_output = renderer.read_headless(&second_surface).unwrap();

    assert_eq!(first_stats.layers, 0);
    assert_eq!(second_stats.layers, 0);
    assert_eq!(first_output.rgba, second_output.rgba);
    assert!(pixel_rgba(&first_output, 0, 0)[0] > 200);
    assert!(pixel_rgba(&first_output, 3, 0)[1] > 200);
}

#[test]
fn sequence9_guardrail_layer_pass_plans_keep_finite_bounds_without_offscreen_texture() {
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.fill(Rect::new(1.0, 2.0, 4.0, 3.0), Color::BLACK);
        scene.layer(
            Layer::new()
                .try_transform(Transform::translation(6.0, 0.0).unwrap())
                .unwrap()
                .blend(BlendMode::Screen),
            |scene| {
                scene.fill(Rect::new(0.0, 1.0, 2.0, 2.0), Color::BLACK);
            },
        );
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[0]
    else {
        panic!("expected outer direct Vello layer");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[1] else {
        panic!("expected inner direct Vello layer");
    };

    for layer in [outer, inner] {
        let bounds = layer
            .pass_plan
            .bounds()
            .expect("direct layer plans should carry explicit child bounds")
            .rect();
        assert_finite_positive_rect(bounds);
        assert_eq!(
            layer.pass_plan.kind(),
            command::LayerPassKind::DirectVelloLayer
        );
        assert!(!layer.pass_plan.requires_offscreen_texture());
    }
    assert_eq!(
        outer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(1.0, 1.0, 7.0, 4.0))
    );
}

#[test]
fn sequence9_guardrail_texture_lifecycle_is_deterministic_for_nested_layer_bounds() {
    let outer_bounds = command::OffscreenBounds::try_new(Rect::new(0.0, 0.0, 8.0, 6.0)).unwrap();
    let inner_bounds = command::OffscreenBounds::try_new(Rect::new(2.0, 1.0, 3.0, 2.0)).unwrap();
    let outer = offscreen_local_scene_texture_descriptor(outer_bounds, 1.0, Format::Rgba8).unwrap();
    let inner = offscreen_local_scene_texture_descriptor(inner_bounds, 1.0, Format::Rgba8).unwrap();
    let mut cache = OffscreenTextureCache::new();

    let outer_first = cache.acquire(outer).unwrap();
    let inner_first = cache.acquire(inner).unwrap();
    cache.release(inner_first).unwrap();
    cache.release(outer_first).unwrap();
    let outer_second = cache.acquire(outer).unwrap();
    let inner_second = cache.acquire(inner).unwrap();

    assert_eq!(outer_second.descriptor(), outer);
    assert_eq!(inner_second.descriptor(), inner);
    assert_eq!(cache.stats().allocations, 2);
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 2);
    assert_eq!(cache.stats().releases, 2);
    assert_eq!(cache.live_count(), 2);
    assert_eq!(cache.retained_count(), 2);
}

#[test]
fn sequence9_guardrail_rect_shader_plumbing_accepts_offscreen_to_intermediate_without_filter_semantics()
 {
    let source = TextureDescriptor::try_new(
        PhysicalSize::new(5, 4),
        Format::Rgba8,
        TextureUsageIntent::OffscreenLayer,
    )
    .unwrap();
    let destination = TextureDescriptor::try_new(
        PhysicalSize::new(5, 4),
        Format::Rgba8,
        TextureUsageIntent::IntermediatePass,
    )
    .unwrap();
    let bounds = RectPassBounds::try_new(0, 0, 5, 4, source, destination).unwrap();

    let pass = RectShaderPassDescriptor::try_new(
        "sequence9-layer-source",
        "sequence9-intermediate",
        source,
        destination,
        bounds,
        RectShaderPassKind::IdentityCopy,
    )
    .unwrap();

    assert_eq!(pass.kind(), RectShaderPassKind::IdentityCopy);
    assert_eq!(pass.source().intent(), TextureUsageIntent::OffscreenLayer);
    assert_eq!(
        pass.destination().intent(),
        TextureUsageIntent::IntermediatePass
    );
    assert_eq!(
        RectShaderPipelineKey::from_descriptor(pass),
        RectShaderPipelineKey::from_descriptor(pass)
    );
}

#[test]
fn sequence9_guardrail_reference_buffers_are_deterministic_composition_oracles() {
    let red_half = PremultipliedRgba8::try_new(128, 0, 0, 128).unwrap();
    let blue_half = PremultipliedRgba8::try_new(0, 0, 128, 128).unwrap();
    let green = PremultipliedRgba8::try_new(0, 255, 0, 255).unwrap();
    let source = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![red_half, PremultipliedRgba8::TRANSPARENT],
    )
    .unwrap();
    let destination = ReferencePremultipliedRgba8Buffer::from_pixels(
        PhysicalSize::new(2, 1),
        vec![blue_half, green],
    )
    .unwrap();

    let first = source.source_over(&destination).unwrap();
    let second = source.source_over(&destination).unwrap();
    let faded = first.apply_opacity(0.5).unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(128, 0, 64, 192).unwrap()
    );
    assert_eq!(first.pixel(1, 0).unwrap(), green);
    assert_eq!(
        faded.pixel(0, 0).unwrap(),
        PremultipliedRgba8::try_new(64, 0, 32, 96).unwrap()
    );
}

#[test]
fn sequence9_guardrail_layer_mask_and_filter_inputs_keep_typed_diagnostics() {
    let cases = [
        (
            Layer::new()
                .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
                .unwrap(),
            UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ),
        ),
        (
            Layer::new()
                .try_filter(Filter::try_blur(2.0).unwrap())
                .unwrap(),
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        ),
    ];

    for (layer, unsupported) in cases {
        let mut scene = Scene::new();
        scene.layer(layer, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        });

        let error = scene
            .normalize(Capabilities::VELLO_0_9)
            .expect_err("Sequence 9 must not execute mask or layer effect semantics");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn scene_normalization_rejects_unsupported_commands_before_encoding() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("unsupported masks should fail during normalization");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
}

#[test]
fn scene_normalization_preserves_stats() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.stroke(
                Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let stats = normalized.stats();

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
}

#[test]
fn surface_tracks_size_and_scale() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    surface.resize(Size::new(20.0, 30.0), 2.0).unwrap();

    assert_eq!(surface.size(), Size::new(20.0, 30.0));
    assert_eq!(surface.scale(), 2.0);
}

#[test]
fn surface_state_reports_availability_without_bool_peeking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::try_new(1.0, 1.0).unwrap(), 1.0))
            .unwrap();

    assert_eq!(surface.state(), SurfaceState::Available);
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
}

#[test]
fn headless_backend_resource_state_tracks_readiness() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::try_new(2.0, 2.0).unwrap(), 1.0))
            .unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    surface
        .resize(Size::try_new(3.0, 3.0).unwrap(), 1.0)
        .unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn presented_surface_lifecycle_state_names_pending_resize() {
    let idle = PresentedLifecycle::ResizePending {
        physical_size: PhysicalSize::new(20, 10),
        resizing: ResizeState::Idle,
    };
    let resizing = idle.with_resizing(ResizeState::Resizing);

    assert_eq!(
        resizing,
        PresentedLifecycle::ResizePending {
            physical_size: PhysicalSize::new(20, 10),
            resizing: ResizeState::Resizing,
        }
    );
    assert_eq!(
        resizing.with_resizing(ResizeState::Resizing),
        resizing,
        "repeating the resizing hint is idempotent"
    );
    assert_eq!(resizing.with_resizing(ResizeState::Idle), idle);
    assert_eq!(
        idle.with_resizing(ResizeState::Idle),
        idle,
        "repeating the idle hint is idempotent"
    );
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn presented_surface_lifecycle_recovers_from_zero_size_at_current_native_size() {
    let state = PresentedLifecycle::NonRenderable {
        physical_size: PhysicalSize::new(0, 0),
        resizing: ResizeState::Resizing,
    };

    assert_eq!(
        state.resize_requested(PhysicalSize::new(640, 480), PhysicalSize::new(640, 480)),
        PresentedLifecycle::Ready {
            resizing: ResizeState::Resizing,
        }
    );
}

#[test]
fn headless_resize_keeps_target_when_physical_size_is_unchanged() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    surface.resize(Size::new(10.4, 10.4), 1.0).unwrap();

    assert_eq!(surface.size(), Size::new(10.4, 10.4));
    assert_eq!(surface.physical_size(), PhysicalSize::new(10, 10));
    assert!(matches!(
        &surface.backend,
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { .. },
            ..
        }
    ));
}

#[test]
fn create_surface_headless_preserves_surface_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let surface = pollster::block_on(renderer.create_surface(
        Attachment::Headless,
        SurfaceOptions {
            size: Size::new(10.0, 20.0),
            scale: 2.0,
            present_mode: PresentMode::Immediate,
            format: Format::Rgba8,
        },
    ))
    .unwrap();

    assert_eq!(surface.size(), Size::new(10.0, 20.0));
    assert_eq!(surface.scale(), 2.0);
    assert_eq!(surface.options.present_mode, PresentMode::Immediate);
    assert_eq!(surface.options.format, Format::Rgba8);
    assert_eq!(surface.physical_size(), PhysicalSize::new(20, 40));
}

#[test]
fn rejects_invalid_surface_geometry() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let error = match pollster::block_on(renderer.create_headless(Size::new(f64::NAN, 10.0), 1.0)) {
        Ok(_) => panic!("non-finite surface size should fail before physical conversion"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::InvalidInput);

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    let error = surface
        .resize(Size::new(1.0, 1.0), 0.0)
        .expect_err("invalid scale should fail before resize");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn invalid_value_errors_name_rejected_value() {
    let error = Error::invalid_value(
        "rectangle width",
        f64::NAN,
        "must be finite and non-negative",
    );

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(
        error.message().contains("rectangle width"),
        "error should name the rejected field: {}",
        error.message()
    );
    assert!(
        error.message().contains("NaN"),
        "error should include the rejected value: {}",
        error.message()
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rectangle width")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("NaN")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite and non-negative")
    );
}

#[test]
fn error_type_stays_below_clippy_large_err_threshold() {
    assert!(
        std::mem::size_of::<Error>() <= 128,
        "Error should stay compact enough for crate-wide Result<T, Error>: {} bytes",
        std::mem::size_of::<Error>()
    );
}

#[test]
fn style_color_inputs_are_root_resolved_concrete_colors() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let input = StyleColor::new(color);

    assert_eq!(input.color(), color);
}

#[test]
fn symbolic_color_policy_keeps_style_colors_root_resolved() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let style_color = StyleColor::new(color);

    assert_eq!(style_color.color(), color);
    assert_eq!(
        StyleColor::symbolic_policy(),
        SymbolicColorPolicy::RootResolvedOnly
    );
}

#[test]
fn paint_colors_convert_srgb_to_concrete_rgba() {
    let color = PaintColor::try_srgb(0.25, 0.5, 0.75, 0.8)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(color, Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap());
}

#[test]
fn paint_colors_convert_hsl_known_vectors() {
    let red = PaintColor::try_hsl(0.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();
    let cyan = PaintColor::try_hsl(180.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(red, Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap());
    assert_eq!(cyan, Color::try_rgba(0.0, 1.0, 1.0, 1.0).unwrap());
}

#[test]
fn paint_colors_reject_invalid_conversion_inputs() {
    assert!(PaintColor::try_srgb(f32::NAN, 0.0, 0.0, 1.0).is_err());
    assert!(PaintColor::try_hsl(f32::NAN, 1.0, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.5, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, -0.1, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, 0.5, f32::INFINITY).is_err());
}

#[test]
fn normalized_paint_layers_preserve_valid_paint_sources() {
    let color = NormalizedPaintLayer::try_new(Paint::from(Color::BLACK)).unwrap();
    let gradient_paint = Paint::from(
        Gradient::try_linear(
            Point::try_new(0.0, 0.0).unwrap(),
            Point::try_new(10.0, 0.0).unwrap(),
            vec![
                GradientStop::try_new(0.0, Color::BLACK).unwrap(),
                GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
            ],
        )
        .unwrap(),
    );
    let gradient = NormalizedPaintLayer::try_new(gradient_paint.clone()).unwrap();

    assert_eq!(color.paint(), &Paint::from(Color::BLACK));
    assert_eq!(gradient.paint(), &gradient_paint);
}

#[test]
fn normalized_paint_layers_reject_invalid_paint_sources() {
    let error = Gradient::try_linear(
        Point::new(f64::NAN, 0.0),
        Point::try_new(1.0, 0.0).unwrap(),
        vec![GradientStop::try_new(0.0, Color::BLACK).unwrap()],
    )
    .expect_err("invalid gradient construction should fail before paint layer");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn gradients_expose_render_ready_geometry_and_stops() {
    let stops = vec![
        GradientStop::try_new(0.0, Color::BLACK).unwrap(),
        GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
    ];
    let linear = Gradient::try_linear(
        Point::try_new(1.0, 2.0).unwrap(),
        Point::try_new(3.0, 4.0).unwrap(),
        stops.clone(),
    )
    .unwrap();
    let radial =
        Gradient::try_radial(Point::try_new(5.0, 6.0).unwrap(), 7.0, stops.clone()).unwrap();
    let sweep = Gradient::try_sweep(Point::try_new(8.0, 9.0).unwrap(), stops.clone()).unwrap();

    assert_eq!(linear.stops(), stops.as_slice());
    assert_eq!(
        linear.linear_points(),
        Some((
            Point::try_new(1.0, 2.0).unwrap(),
            Point::try_new(3.0, 4.0).unwrap()
        ))
    );
    assert_eq!(
        radial.radial_geometry(),
        Some((Point::try_new(5.0, 6.0).unwrap(), 7.0))
    );
    assert_eq!(
        sweep.sweep_center(),
        Some(Point::try_new(8.0, 9.0).unwrap())
    );
}

#[test]
fn gradients_preserve_transparent_stops() {
    let stop = GradientStop::try_new(0.5, Color::TRANSPARENT).unwrap();

    assert_eq!(stop.color(), Color::TRANSPARENT);
}

#[test]
fn style_reference_identifiers_must_not_be_empty() {
    let error = StyleResourceRef::try_new("  ").expect_err("empty identifiers are invalid");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("style resource reference")
    );
}

#[test]
fn resolved_image_resources_preserve_handle_and_intrinsic_size() {
    let resource = ResolvedImageResource::try_new(ImageId::new(7), Size::new(24.0, 12.0)).unwrap();

    assert_eq!(resource.id(), ImageId::new(7));
    assert_eq!(resource.intrinsic_size(), Size::new(24.0, 12.0));
}

#[test]
fn resolved_image_resources_carry_root_resolved_metadata_policy() {
    let resource = ResolvedImageResource::try_new(ImageId::new(12), Size::new(40.0, 20.0))
        .unwrap()
        .with_density(ImageResourceDensity::try_new(2.0).unwrap());

    assert_eq!(resource.id(), ImageId::new(12));
    assert_eq!(resource.intrinsic_size(), Size::new(40.0, 20.0));
    assert_eq!(
        resource.density().map(ImageResourceDensity::value),
        Some(2.0)
    );
    assert_eq!(
        resource.orientation_policy(),
        ImageOrientationPolicy::RootResolvedOnly
    );
    assert_eq!(
        resource.color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
}

#[test]
fn image_resource_density_rejects_invalid_values() {
    let error = ImageResourceDensity::try_new(0.0)
        .expect_err("image density must be positive when supplied");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image resource density")
    );
}

#[test]
fn unresolved_style_image_sources_report_image_resource_diagnostics() {
    let reference = StyleResourceRef::try_new("hero.png").unwrap();
    let source = StyleImageSource::unresolved(reference.clone());

    assert_eq!(
        source.kind(),
        &StyleImageSourceKind::Unresolved(reference.clone())
    );

    let error = source
        .require_resolved()
        .expect_err("unresolved image source must report an image resource diagnostic");
    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error.unresolved_resource_diagnostic(),
        Some(&UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            reference.identifier()
        ))
    );
}

#[test]
fn css_image_layers_preserve_sampling_inputs_without_lowering() {
    let resource = ResolvedImageResource::try_new(ImageId::new(11), Size::new(8.0, 8.0)).unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .with_position(BackgroundPosition::percent(0.25, 0.75).unwrap())
        .with_size(BackgroundSize::cover())
        .with_repeat(BackgroundRepeat::repeat_x())
        .with_origin(BackgroundBox::Padding)
        .with_clip(BackgroundBox::Content)
        .with_attachment(BackgroundAttachment::Fixed);

    assert_eq!(
        layer.source().kind(),
        &StyleImageSourceKind::Resolved(resource)
    );
    assert_eq!(layer.position().x().kind(), PositionComponentKind::Percent);
    assert_eq!(layer.position().y().value(), 0.75);
    assert_eq!(layer.size(), BackgroundSize::cover());
    assert_eq!(layer.repeat(), BackgroundRepeat::repeat_x());
    assert_eq!(layer.origin(), BackgroundBox::Padding);
    assert_eq!(layer.clip(), BackgroundBox::Content);
    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
}

#[test]
fn image_placement_auto_uses_intrinsic_size_and_position_ratio() {
    let input = ImagePlacementInput::try_new(
        Rect::new(10.0, 20.0, 100.0, 50.0),
        Size::new(20.0, 10.0),
        BackgroundPosition::percent(0.5, 1.0).unwrap(),
        BackgroundSize::auto(),
    )
    .unwrap();

    let placement = input.resolve().unwrap();

    assert_eq!(placement.paint_rect(), Rect::new(10.0, 20.0, 100.0, 50.0));
    assert_eq!(placement.tile_rect(), Rect::new(50.0, 60.0, 20.0, 10.0));
}

#[test]
fn image_placement_cover_and_contain_preserve_aspect_ratio() {
    let paint_rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let intrinsic = Size::new(20.0, 20.0);

    let cover = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::cover(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(cover.tile_rect(), Rect::new(0.0, -25.0, 100.0, 100.0));

    let contain = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::contain(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(contain.tile_rect(), Rect::new(25.0, 0.0, 50.0, 50.0));
}

#[test]
fn image_placement_explicit_size_resolves_lengths_percents_and_auto_axis() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::length(5.0, 10.0).unwrap(),
        BackgroundSize::explicit(
            SizeComponent::try_percent(0.5).unwrap(),
            SizeComponent::auto(),
        ),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(5.0, 10.0, 100.0, 50.0));
}

#[test]
fn image_placement_edge_offsets_represent_four_component_positions() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(-20.0, -10.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::edge_offsets(
            PositionEdgeOffset::end(15.0).unwrap(),
            PositionEdgeOffset::end(5.0).unwrap(),
        ),
        BackgroundSize::auto(),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(125.0, 65.0, 40.0, 20.0));
}

#[test]
fn image_placement_rejects_invalid_paint_or_intrinsic_size() {
    let error = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 0.0, 100.0),
        Size::new(10.0, 10.0),
        BackgroundPosition::default(),
        BackgroundSize::auto(),
    )
    .expect_err("paint rect must be positive");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image placement paint rect")
    );
}

#[test]
fn image_repeat_plan_maps_css_repeat_axes() {
    let cases = [
        (BackgroundRepeat::no_repeat(), ImageRepeatMode::NoRepeat),
        (BackgroundRepeat::repeat_x(), ImageRepeatMode::RepeatX),
        (BackgroundRepeat::repeat_y(), ImageRepeatMode::RepeatY),
        (BackgroundRepeat::repeat(), ImageRepeatMode::RepeatBoth),
    ];

    for (repeat, expected) in cases {
        let plan = ImageRepeatPlan::try_new(repeat, Capabilities::VELLO_0_9).unwrap();
        assert_eq!(plan.repeat(), repeat);
        assert_eq!(plan.mode(), expected);
    }
}

#[test]
fn image_repeat_plan_resolves_tile_rects_inside_clip_rect() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 70.0, 40.0),
        Rect::new(0.0, 5.0, 20.0, 10.0),
    )
    .unwrap();

    let repeat_x = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(repeat_x.clip_rect(), Rect::new(0.0, 0.0, 70.0, 40.0));
    assert_eq!(
        repeat_x.tile_rects(),
        &[
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(20.0, 5.0, 20.0, 10.0),
            Rect::new(40.0, 5.0, 20.0, 10.0),
            Rect::new(60.0, 5.0, 20.0, 10.0),
        ]
    );

    let repeat_y = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_y(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(
        repeat_y.tile_rects(),
        &[
            Rect::new(0.0, -5.0, 20.0, 10.0),
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(0.0, 15.0, 20.0, 10.0),
            Rect::new(0.0, 25.0, 20.0, 10.0),
            Rect::new(0.0, 35.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_includes_tiles_before_the_anchor_when_visible() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 50.0, 20.0),
        Rect::new(15.0, 0.0, 20.0, 10.0),
    )
    .unwrap();

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();

    assert_eq!(
        repeated.tile_rects(),
        &[
            Rect::new(-5.0, 0.0, 20.0, 10.0),
            Rect::new(15.0, 0.0, 20.0, 10.0),
            Rect::new(35.0, 0.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_fast_forwards_from_huge_negative_tile_origin() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 40.0, 10.0),
        Rect::new(-1_000_000_000_000.0, 0.0, 10.0, 10.0),
    )
    .unwrap();

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();

    assert_eq!(
        repeated.tile_rects(),
        &[
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(10.0, 0.0, 10.0, 10.0),
            Rect::new(20.0, 0.0, 10.0, 10.0),
            Rect::new(30.0, 0.0, 10.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_rejects_excessive_resolved_tile_count() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 250.25, 1_000.0),
        Rect::new(0.0, 0.0, 0.25, 1.0),
    )
    .unwrap();

    let error = ImageRepeatPlan::try_new(BackgroundRepeat::repeat(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .expect_err("excessive repeat tiling must be rejected before allocation");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image repeat tile count")
    );
}

#[test]
fn css_image_layer_normalizes_placement_repeat_and_attachment_together() {
    let resource = ResolvedImageResource::try_new(ImageId::new(90), Size::new(25.0, 10.0)).unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
        .with_size(BackgroundSize::explicit(
            SizeComponent::try_length(50.0).unwrap(),
            SizeComponent::auto(),
        ))
        .with_repeat(BackgroundRepeat::repeat_x())
        .with_attachment(BackgroundAttachment::Fixed)
        .with_coordinate_space(
            CoordinateSpaceTag::viewport(Transform::translation(2.0, 3.0).unwrap()).unwrap(),
        );

    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        resource.intrinsic_size(),
        layer.position(),
        layer.size(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    let repeat = ImageRepeatPlan::try_new(layer.repeat(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    let attachment =
        ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space()).unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(70.0, 0.0, 50.0, 20.0));
    assert_eq!(repeat.clip_rect(), Rect::new(0.0, 0.0, 120.0, 80.0));
    assert_eq!(
        repeat.tile_rects(),
        &[
            Rect::new(-30.0, 0.0, 50.0, 20.0),
            Rect::new(20.0, 0.0, 50.0, 20.0),
            Rect::new(70.0, 0.0, 50.0, 20.0),
        ]
    );
    assert_eq!(
        attachment.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn image_repeat_plan_rejects_round_and_space_with_typed_diagnostics() {
    let round = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::Round, RepeatMode::Repeat),
        Capabilities::VELLO_0_9,
    )
    .expect_err("round repeat is not supported yet");
    assert_eq!(
        round.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatRound
        ))
    );

    let space = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::NoRepeat, RepeatMode::Space),
        Capabilities::VELLO_0_9,
    )
    .expect_err("space repeat is not supported yet");
    assert_eq!(
        space.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatSpace
        ))
    );
}

#[test]
fn fixed_background_layers_can_carry_viewport_coordinate_space() {
    let layer =
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_attachment(BackgroundAttachment::Fixed)
            .with_coordinate_space(
                CoordinateSpaceTag::viewport(Transform::translation(10.0, 20.0).unwrap()).unwrap(),
            );

    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        layer.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn image_attachment_plan_uses_root_resolved_scroll_and_local_coordinates() {
    let scroll = ImageAttachmentPlan::try_new(BackgroundAttachment::Scroll, None).unwrap();
    assert_eq!(scroll.attachment(), BackgroundAttachment::Scroll);
    assert_eq!(
        scroll.coordinate_space().map(CoordinateSpaceTag::kind),
        None
    );

    let local_tag = CoordinateSpaceTag::local();
    let local = ImageAttachmentPlan::try_new(BackgroundAttachment::Local, Some(local_tag)).unwrap();
    assert_eq!(local.attachment(), BackgroundAttachment::Local);
    assert_eq!(
        local.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Local)
    );
}

#[test]
fn fixed_image_attachment_requires_viewport_coordinate_tag() {
    let missing = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, None)
        .expect_err("fixed backgrounds require an explicit viewport tag");
    assert_eq!(missing.code(), ErrorCode::InvalidInput);
    assert_eq!(
        missing.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let surface = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let wrong = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(surface))
        .expect_err("fixed backgrounds must be tagged in viewport coordinates");
    assert_eq!(
        wrong.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let viewport = CoordinateSpaceTag::viewport(Transform::translation(3.0, 4.0).unwrap()).unwrap();
    let fixed = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(viewport)).unwrap();
    assert_eq!(fixed.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        fixed.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn resolved_image_resources_reject_invalid_intrinsic_size() {
    let error = ResolvedImageResource::try_new(ImageId::new(7), Size::new(f64::NAN, 12.0))
        .expect_err("invalid intrinsic size should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("resolved image intrinsic size width")
    );
}

#[test]
fn background_position_rejects_non_finite_percent() {
    let error = BackgroundPosition::percent(f64::NAN, 0.0)
        .expect_err("non-finite percentages should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background position x percent")
    );
}

#[test]
fn background_size_rejects_negative_length() {
    let error = SizeComponent::try_length(-1.0)
        .expect_err("negative explicit background sizes should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background size length")
    );
}

#[test]
fn filter_lists_distinguish_none_from_ordered_ops() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
    ])
    .unwrap();

    assert!(!list.is_none());
    assert_eq!(list.ops().len(), 2);
    assert!(matches!(
        list.ops()[0].kind(),
        FilterOpKind::Brightness(amount) if amount.value() == 1.2
    ));
    assert!(matches!(
        list.ops()[1].kind(),
        FilterOpKind::Blur(blur) if blur.radius() == 4.0
    ));
    assert!(FilterList::none().is_none());
    assert!(FilterList::none().ops().is_empty());
}

#[test]
fn filter_lists_classify_ordered_color_filter_pipelines() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(0.8).unwrap()),
        FilterOp::grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
        FilterOp::hue_rotate(FilterAngle::try_radians(0.5).unwrap()),
        FilterOp::invert(UnitFilterAmount::try_new(0.4).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::saturate(FilterAmount::try_new(1.5).unwrap()),
        FilterOp::sepia(UnitFilterAmount::try_new(0.6).unwrap()),
    ])
    .unwrap();

    let pipeline = list
        .color_filter_pipeline()
        .expect("color-only filter lists should classify")
        .expect("color-only filter lists should produce a pipeline");

    assert_eq!(
        pipeline.ops(),
        &[
            ColorFilterOp::Brightness(FilterAmount::try_new(1.2).unwrap()),
            ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
            ColorFilterOp::Grayscale(UnitFilterAmount::try_new(0.25).unwrap()),
            ColorFilterOp::HueRotate(FilterAngle::try_radians(0.5).unwrap()),
            ColorFilterOp::Invert(UnitFilterAmount::try_new(0.4).unwrap()),
            ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap()),
            ColorFilterOp::Saturate(FilterAmount::try_new(1.5).unwrap()),
            ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.6).unwrap()),
        ]
    );
}

#[test]
fn filter_none_has_no_executable_color_pipeline() {
    assert_eq!(FilterList::none().color_filter_pipeline(), Ok(None));
}

#[test]
fn materialized_image_filter_classifier_preserves_mixed_filter_order() {
    let shadow = Shadow::try_new(Point::new(2.0, 3.0), 4.0, 0.0, Color::BLACK).unwrap();
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(0.8).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
        FilterOp::opacity(UnitFilterAmount::try_new(0.75).unwrap()),
        FilterOp::drop_shadow(shadow.clone()),
        FilterOp::sepia(UnitFilterAmount::try_new(0.25).unwrap()),
    ])
    .unwrap();

    let pipeline = list
        .materialized_image_filter_pipeline()
        .expect("materialized image filters should classify")
        .expect("non-empty filter lists should produce a pipeline");

    assert_eq!(pipeline.steps().len(), 5);
    assert!(matches!(
        &pipeline.steps()[0],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [
                    ColorFilterOp::Brightness(FilterAmount::try_new(1.2).unwrap()),
                    ColorFilterOp::Contrast(FilterAmount::try_new(0.8).unwrap()),
                ]
    ));
    assert!(matches!(
        pipeline.steps()[1],
        MaterializedImageFilterStep::Blur(blur) if blur.radius() == 4.0
    ));
    assert!(matches!(
        &pipeline.steps()[2],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [ColorFilterOp::Opacity(UnitFilterAmount::try_new(0.75).unwrap())]
    ));
    assert!(matches!(
        &pipeline.steps()[3],
        MaterializedImageFilterStep::DropShadow(classified) if classified == &shadow
    ));
    assert!(matches!(
        &pipeline.steps()[4],
        MaterializedImageFilterStep::ColorFilters(pipeline)
            if pipeline.source_ops()
                == [ColorFilterOp::Sepia(UnitFilterAmount::try_new(0.25).unwrap())]
    ));
}

#[test]
fn filter_none_has_no_materialized_image_filter_pipeline() {
    assert_eq!(
        FilterList::none()
            .materialized_image_filter_pipeline()
            .unwrap(),
        None
    );
}

#[test]
fn materialized_image_filter_classifier_accepts_blur_and_drop_shadow() {
    let shadow = Shadow::try_new(Point::new(1.0, 2.0), 3.0, 0.0, Color::BLACK).unwrap();
    let list = FilterList::try_ops(vec![
        FilterOp::blur(FilterBlur::try_new(2.0).unwrap()),
        FilterOp::drop_shadow(shadow.clone()),
    ])
    .unwrap();

    let pipeline = list
        .materialized_image_filter_pipeline()
        .expect("blur and drop-shadow should classify")
        .expect("non-empty materialized filter lists should produce a pipeline");

    assert_eq!(
        pipeline.steps(),
        &[
            MaterializedImageFilterStep::Blur(FilterBlur::try_new(2.0).unwrap()),
            MaterializedImageFilterStep::DropShadow(shadow),
        ]
    );
}

#[test]
fn materialized_filter_classification_does_not_make_resource_handles_bytes() {
    let resource = ResolvedImageResource::try_new(ImageId::new(41), Size::new(8.0, 8.0)).unwrap();
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();

    assert!(
        paint
            .filters()
            .materialized_image_filter_pipeline()
            .unwrap()
            .is_some()
    );

    let unsupported = paint
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("resource-only filtered image paint is still not materialized bytes");
    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn filter_blur_policy_zero_radius_produces_zero_outset() {
    let policy = BlurPolicy::css_filter_default();
    let outset = FilterOutset::from_blur(FilterBlur::try_new(0.0).unwrap(), policy).unwrap();

    assert_eq!(outset, FilterOutset::zero());
    assert_eq!(
        policy.radius_interpretation(),
        BlurRadiusInterpretation::CssLengthAsStandardDeviation
    );
    assert_eq!(
        policy.edge_sampling(),
        TransparentEdgeSamplingPolicy::TransparentBlack
    );
}

#[test]
fn filter_blur_region_inflates_bounds_deterministically() {
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 20.0, 30.0, 40.0)).unwrap();
    let outset = FilterOutset::from_blur(
        FilterBlur::try_new(4.0).unwrap(),
        BlurPolicy::css_filter_default(),
    )
    .unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, None).unwrap();

    assert_eq!(outset, FilterOutset::try_uniform(10.0).unwrap());
    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(0.0, 10.0, 50.0, 60.0)
    );
    assert_eq!(
        plan.execution_region().rect(),
        Rect::new(0.0, 10.0, 50.0, 60.0)
    );
}

#[test]
fn drop_shadow_outset_combines_offset_and_blur_support() {
    let source = FilterSourceBounds::try_new(Rect::new(10.0, 10.0, 20.0, 10.0)).unwrap();
    let shadow = Shadow::try_new(Point::new(3.0, -2.0), 2.0, 0.0, Color::BLACK).unwrap();
    let outset = FilterOutset::from_drop_shadow(&shadow, BlurPolicy::css_filter_default()).unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, None).unwrap();

    assert_eq!(outset, FilterOutset::try_new(2.0, 7.0, 8.0, 3.0).unwrap());
    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(8.0, 3.0, 30.0, 20.0)
    );
}

#[test]
fn drop_shadow_outset_rejects_inset_shadow_with_typed_diagnostic() {
    let shadow = Shadow::try_inset(Point::new(3.0, -2.0), 2.0, 0.0, Color::BLACK).unwrap();
    let error = FilterOutset::from_drop_shadow(&shadow, BlurPolicy::css_filter_default())
        .expect_err("CSS drop-shadow outset planning does not support inset shadows");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
}

#[test]
fn filter_region_plan_clips_inflated_bounds_to_explicit_filter_region() {
    let source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 20.0, 20.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(-5.0, -2.0, 30.0, 18.0)).unwrap();
    let outset = FilterOutset::from_blur(
        FilterBlur::try_new(4.0).unwrap(),
        BlurPolicy::css_filter_default(),
    )
    .unwrap();

    let plan = FilterRegionPlan::try_new(source, outset, Some(clip)).unwrap();

    assert_eq!(
        plan.inflated_bounds().rect(),
        Rect::new(-10.0, -10.0, 40.0, 40.0)
    );
    assert_eq!(plan.clip_bounds(), Some(clip));
    assert_eq!(
        plan.execution_region().rect(),
        Rect::new(-5.0, -2.0, 30.0, 18.0)
    );
}

#[test]
fn filter_blur_policy_names_large_radius_clamp_and_rejection() {
    let clamp = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_clamp_to(8.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();
    let reject = BlurPolicy::try_new(
        BlurRadiusInterpretation::CssLengthAsStandardDeviation,
        KernelSupportRadius::try_standard_deviation_multiple(2.5).unwrap(),
        LargeBlurRadiusPolicy::try_reject_above(8.0).unwrap(),
        TransparentEdgeSamplingPolicy::TransparentBlack,
    )
    .unwrap();

    assert_eq!(
        clamp.large_radius_policy().action(),
        LargeBlurRadiusAction::Clamp
    );
    assert_eq!(
        FilterOutset::from_blur(FilterBlur::try_new(12.0).unwrap(), clamp).unwrap(),
        FilterOutset::try_uniform(20.0).unwrap()
    );

    let error = FilterOutset::from_blur(FilterBlur::try_new(12.0).unwrap(), reject)
        .expect_err("rejecting large blur radii should report a typed invalid value");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_region_models_reject_invalid_bounds_and_radii() {
    let zero_source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("filter source bounds must have area");
    assert_eq!(zero_source.code(), ErrorCode::InvalidInput);
    assert_eq!(
        zero_source
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter source bounds width")
    );

    let non_finite_clip = FilterClipBounds::try_new(Rect::new(f64::INFINITY, 0.0, 1.0, 1.0))
        .expect_err("unbounded sentinel filter regions should be rejected");
    assert_eq!(
        non_finite_clip
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter clip bounds x")
    );

    let negative_outset =
        FilterOutset::try_new(-1.0, 0.0, 0.0, 0.0).expect_err("outsets cannot be negative");
    assert_eq!(
        negative_outset
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter outset left")
    );

    let negative_radius =
        FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");
    assert_eq!(
        negative_radius
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter blur radius")
    );

    let source = FilterSourceBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let clip = FilterClipBounds::try_new(Rect::new(20.0, 20.0, 5.0, 5.0)).unwrap();
    let empty_execution = FilterRegionPlan::try_new(source, FilterOutset::zero(), Some(clip))
        .expect_err("clipping to an empty region should be rejected");
    assert_eq!(
        empty_execution
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("filter execution region")
    );
}

#[test]
fn filter_color_pipeline_rejects_blur_with_typed_diagnostic() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.0).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
        FilterOp::contrast(FilterAmount::try_new(1.0).unwrap()),
    ])
    .unwrap();

    let unsupported = list
        .color_filter_pipeline()
        .expect_err("blur is not a color-only filter operation");

    assert_eq!(
        unsupported,
        UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::ColorFilterBlur
        )
    );
    assert_eq!(unsupported.label(), "color filter blur");
}

#[test]
fn filter_color_pipeline_rejects_drop_shadow_with_typed_diagnostic() {
    let shadow = Shadow::try_new(Point::new(1.0, 2.0), 3.0, 0.0, Color::BLACK).unwrap();
    let list = FilterList::try_ops(vec![
        FilterOp::saturate(FilterAmount::try_new(1.0).unwrap()),
        FilterOp::drop_shadow(shadow),
        FilterOp::sepia(UnitFilterAmount::try_new(0.25).unwrap()),
    ])
    .unwrap();

    let unsupported = list
        .color_filter_pipeline()
        .expect_err("drop-shadow is not a color-only filter operation");

    assert_eq!(
        unsupported,
        UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::ColorFilterDropShadow
        )
    );
    assert_eq!(unsupported.label(), "color filter drop-shadow");
}

#[test]
fn filter_lists_reject_empty_ordered_ops() {
    let error = FilterList::try_ops(Vec::new()).expect_err("empty op lists must use none");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter operations")
    );
}

#[test]
fn filtered_image_paint_preserves_resolved_image_and_filter_list() {
    let resource = ResolvedImageResource::try_new(ImageId::new(30), Size::new(16.0, 16.0)).unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.25).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource.clone(), filters.clone()).unwrap();

    assert_eq!(paint.resource(), &resource);
    assert_eq!(paint.filters(), &filters);
}

#[test]
fn filtered_image_paint_rejects_none_filter_list_and_reports_execution_boundary() {
    let resource = ResolvedImageResource::try_new(ImageId::new(31), Size::new(8.0, 8.0)).unwrap();
    let error = FilteredImagePaint::try_new(resource.clone(), FilterList::none())
        .expect_err("filtered image paint requires a non-empty filter list");
    assert_eq!(error.code(), ErrorCode::InvalidInput);

    let filters = FilterList::try_ops(vec![FilterOp::contrast(
        FilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();
    let unsupported = paint
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("filtered image paint execution belongs to filter phases");
    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}

#[test]
fn backdrop_filter_input_preserves_supported_filters_bounds_and_clip() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 1.0, 12.0, 8.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(1.0, 2.0, 4.0, 5.0))).unwrap();

    let input = BackdropFilterInput::try_new(filters.clone(), bounds, Some(clip.clone())).unwrap();

    assert_eq!(input.filters(), &filters);
    assert_eq!(input.capture_bounds(), bounds);
    assert_eq!(input.clip(), Some(&clip));
}

#[test]
fn backdrop_filter_input_rejects_empty_filters() {
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let error = BackdropFilterInput::try_new(FilterList::none(), bounds, None)
        .expect_err("backdrop filters must be an explicit non-empty filter list");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("backdrop filter input filters")
    );
}

#[test]
fn backdrop_capture_bounds_reject_invalid_rectangles() {
    let zero = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("backdrop capture bounds must have positive area");

    assert_eq!(zero.code(), ErrorCode::InvalidInput);
    assert_eq!(
        zero.invalid_value_diagnostic().map(InvalidValue::field),
        Some("backdrop capture bounds width")
    );

    let non_finite = BackdropCaptureBounds::try_new(Rect::new(f64::INFINITY, 0.0, 1.0, 1.0))
        .expect_err("backdrop capture bounds must be finite");
    assert_eq!(
        non_finite
            .invalid_value_diagnostic()
            .map(InvalidValue::field),
        Some("backdrop capture bounds x")
    );
}

#[test]
fn backdrop_filter_input_rejects_unresolved_clip_references() {
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.1).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let clip = ClipInput::reference(StyleResourceRef::try_new("#backdrop-clip").unwrap());

    let error = BackdropFilterInput::try_new(filters, bounds, Some(clip))
        .expect_err("backdrop clip geometry must already be render-owned");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );
}

#[test]
fn backdrop_filter_root_policy_reports_explicit_diagnostic() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let error = BackdropFilterInput::try_root_backdrop(filters, None)
        .expect_err("root backdrop capture is not render-owned yet");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ))
    );
}

#[test]
fn backdrop_layer_normalization_plans_bounded_capture_without_broad_execution() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(2.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(1.0, 2.0, 8.0, 6.0)).unwrap();
    let backdrop = BackdropFilterInput::try_new(filters.clone(), bounds, None).unwrap();
    let layer = Layer::new().try_backdrop_filter(backdrop).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(2.0, 3.0, 4.0, 2.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[1] else {
        panic!("expected backdrop layer command");
    };

    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::BoundedBackdropCapture
    );
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::OffscreenTexture
    );
    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(bounds.rect())
    );
    let capture = layer
        .backdrop
        .as_ref()
        .expect("backdrop capture is planned");
    assert_eq!(capture.filters(), &filters);
    assert_eq!(capture.capture_bounds().rect(), bounds.rect());
    assert_eq!(capture.source_commands().len(), 1);
    let offscreen = Capabilities::VELLO_0_9.offscreen_pipeline();
    assert!(offscreen.supports_bounded_backdrop_capture());
    assert!(offscreen.supports_materialized_backdrop_filter_execution());
    assert!(!offscreen.supports_backdrop_execution());
}

#[test]
fn render_materializes_bounded_backdrop_capture_from_prior_siblings() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 2.0, 1.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |_| {})
        .fill(
            Rect::new(1.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect("backdrop planning should remain inspectable through normalization");
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[1] else {
        panic!("expected normalized backdrop layer");
    };
    assert!(layer.backdrop.is_some());

    let Some(output) = render_scene_to_headless_or_skip_no_adapter(&scene, Size::new(2.0, 1.0))
    else {
        return;
    };

    let prior_only_backdrop = pixel_rgba(&output, 0, 0);
    assert!(
        prior_only_backdrop[1] > 200 && prior_only_backdrop[2] > 200,
        "red prior content should be inverted into cyan: {prior_only_backdrop:?}"
    );
    let later_content = pixel_rgba(&output, 1, 0);
    assert!(
        later_content[1] > 200 && later_content[0] < 80 && later_content[2] < 80,
        "later sibling content should render after capture, not into it: {later_content:?}"
    );
}

#[test]
fn render_backdrop_filter_order_is_preserved() {
    let source_rect = Rect::new(0.0, 0.0, 3.0, 1.0);
    let bounds = BackdropCaptureBounds::try_new(source_rect).unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let mut color_before_blur = Scene::new();
    color_before_blur
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    let mut blur_before_color = Scene::new();
    blur_before_color
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![blur, brightness]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    let Some(color_first) =
        render_scene_to_headless_or_skip_no_adapter(&color_before_blur, Size::new(3.0, 1.0))
    else {
        return;
    };
    let Some(blur_first) =
        render_scene_to_headless_or_skip_no_adapter(&blur_before_color, Size::new(3.0, 1.0))
    else {
        return;
    };

    assert_ne!(color_first.rgba, blur_first.rgba);
}

#[test]
fn render_backdrop_clip_limits_filtered_image_to_requested_region() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 5.0, 5.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 1.0, 3.0, 3.0),
        Radii::all(1.5),
    ))
    .unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, Some(clip)).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 5.0, 5.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |_| {});

    let Some(output) = render_scene_to_headless_or_skip_no_adapter(&scene, Size::new(5.0, 5.0))
    else {
        return;
    };

    let outside_clip = pixel_rgba(&output, 0, 0);
    assert!(
        outside_clip[0] > 200 && outside_clip[1] < 80 && outside_clip[2] < 80,
        "filtered backdrop should not leak outside the rounded clip: {outside_clip:?}"
    );
    let inside_clip = pixel_rgba(&output, 2, 2);
    assert!(
        inside_clip[1] > 200 && inside_clip[2] > 200,
        "filtered backdrop should render inside the rounded clip: {inside_clip:?}"
    );
}

#[test]
fn render_backdrop_foreground_composites_over_filtered_backdrop() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 3.0, 1.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 3.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(layer, |scene| {
            scene.fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK);
        });

    let Some(output) = render_scene_to_headless_or_skip_no_adapter(&scene, Size::new(3.0, 1.0))
    else {
        return;
    };

    let backdrop_only = pixel_rgba(&output, 0, 0);
    assert!(
        backdrop_only[1] > 200 && backdrop_only[2] > 200,
        "filtered backdrop should sit behind foreground: {backdrop_only:?}"
    );
    let foreground = pixel_rgba(&output, 1, 0);
    assert!(
        foreground[0] < 80 && foreground[1] < 80 && foreground[2] < 80,
        "foreground content should composite over backdrop: {foreground:?}"
    );
}

#[test]
fn backdrop_layer_normalization_preserves_command_order_for_capture_sources() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(layer, |scene| {
            scene.fill(
                Rect::new(2.0, 0.0, 1.0, 1.0),
                Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            );
        })
        .fill(Rect::new(4.0, 0.0, 1.0, 1.0), Color::BLACK);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[1] else {
        panic!("expected backdrop layer command");
    };
    let capture = layer
        .backdrop
        .as_ref()
        .expect("backdrop capture is planned");

    assert_eq!(capture.source_commands().len(), 1);
    assert!(matches!(
        capture.source_commands()[0],
        command::RenderCommand::Fill { .. }
    ));
    assert_eq!(children.len(), 1);
    assert!(matches!(
        normalized.commands[2],
        command::RenderCommand::Fill { .. }
    ));
}

#[test]
fn nested_backdrop_layer_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new(), |scene| {
        scene.layer(backdrop, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        });
    });

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("nested backdrop capture is outside the normalization boundary");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropExecution,
        ))
    );
    assert!(error.message().contains("nested backdrop capture"));
}

#[test]
fn transformed_backdrop_layer_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let backdrop = Layer::new()
        .try_transform(Transform::translation(2.0, 0.0).unwrap())
        .unwrap()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(backdrop, |_| {});

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("transformed backdrop capture needs coordinate-space reconciliation");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropExecution,
        ))
    );
    assert!(error.message().contains("transformed backdrop capture"));
}

#[test]
fn repeated_top_level_backdrop_normalization_reports_typed_boundary() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
    let first_backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters.clone(), bounds, None).unwrap())
        .unwrap();
    let second_backdrop = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(first_backdrop, |_| {})
        .layer(second_backdrop, |_| {});

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("repeated top-level backdrop captures need staged source reconciliation");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropExecution,
        ))
    );
    assert!(
        error
            .message()
            .contains("repeated top-level backdrop capture")
    );
}

#[test]
fn backdrop_layer_normalization_carries_rounded_and_path_clip_planning() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 20.0, 20.0)).unwrap();
    let rounded_clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 2.0, 8.0, 6.0),
        Radii::all(2.0),
    ))
    .unwrap();
    let rounded_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(filters.clone(), bounds, Some(rounded_clip)).unwrap(),
        )
        .unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(3.0, 4.0))
        .line_to(Point::new(7.0, 4.0))
        .line_to(Point::new(7.0, 9.0))
        .close();
    let filled = FilledPath::try_new(path, FillRule::EvenOdd).unwrap();
    let path_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                bounds,
                Some(ClipInput::try_filled_path(filled).unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
    let mut rounded_scene = Scene::new();
    rounded_scene.layer(rounded_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    let rounded_normalized = rounded_scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer {
        layer: rounded_layer,
        ..
    } = &rounded_normalized.commands[0]
    else {
        panic!("expected rounded backdrop layer command");
    };
    let rounded_capture = rounded_layer
        .backdrop
        .as_ref()
        .expect("rounded backdrop capture is planned");

    let mut path_scene = Scene::new();
    path_scene.layer(path_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });
    let path_normalized = path_scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer {
        layer: path_layer, ..
    } = &path_normalized.commands[0]
    else {
        panic!("expected path backdrop layer command");
    };
    let path_capture = path_layer
        .backdrop
        .as_ref()
        .expect("path backdrop capture is planned");

    assert!(matches!(
        rounded_capture.clip().map(command::RenderClip::geometry),
        Some(command::RenderClipGeometry::RoundedRect { .. })
    ));
    assert!(matches!(
        path_capture.clip().map(command::RenderClip::geometry),
        Some(command::RenderClipGeometry::Path {
            fill_rule: FillRule::EvenOdd,
            ..
        })
    ));
}

#[test]
fn sequence13_bounded_backdrop_capture_materializes_prior_siblings_with_foreground_order() {
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 3.0, 1.0)).unwrap();
    let backdrop_layer = Layer::new()
        .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(backdrop_layer, |scene| {
            scene.fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK);
        })
        .fill(
            Rect::new(2.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.0, 1.0, 0.0, 1.0).unwrap(),
        );

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[1] else {
        panic!("expected bounded backdrop layer command");
    };
    let capture = layer.backdrop.as_ref().expect("backdrop capture planned");
    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::BoundedBackdropCapture
    );
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::OffscreenTexture
    );
    assert_eq!(capture.capture_bounds().rect(), bounds.rect());
    assert_eq!(capture.source_commands().len(), 1);
    assert_eq!(children.len(), 1);
    assert!(matches!(
        normalized.commands[2],
        command::RenderCommand::Fill { .. }
    ));

    let Some(output) = render_scene_to_headless_or_skip_no_adapter(&scene, Size::new(3.0, 1.0))
    else {
        return;
    };

    let prior_backdrop = pixel_rgba(&output, 0, 0);
    assert!(
        prior_backdrop[1] > 200 && prior_backdrop[2] > 200,
        "prior red sibling should be captured then inverted: {prior_backdrop:?}"
    );
    let foreground = pixel_rgba(&output, 1, 0);
    assert!(
        foreground[0] < 80 && foreground[1] < 80 && foreground[2] < 80,
        "foreground child should composite over the filtered backdrop: {foreground:?}"
    );
    let later_sibling = pixel_rgba(&output, 2, 0);
    assert!(
        later_sibling[1] > 200 && later_sibling[0] < 80 && later_sibling[2] < 80,
        "later sibling should paint after backdrop capture, not feed it: {later_sibling:?}"
    );
}

#[test]
fn sequence13_backdrop_filter_chain_preserves_order_and_clipping() {
    let source_rect = Rect::new(0.0, 0.0, 3.0, 1.0);
    let bounds = BackdropCaptureBounds::try_new(source_rect).unwrap();
    let brightness = FilterOp::brightness(FilterAmount::try_new(2.0).unwrap());
    let blur = FilterOp::blur(FilterBlur::try_new(1.0).unwrap());
    let mut color_before_blur = Scene::new();
    color_before_blur
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![brightness.clone(), blur.clone()]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );
    let mut blur_before_color = Scene::new();
    blur_before_color
        .fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            Color::try_rgba(0.8, 0.0, 0.0, 1.0).unwrap(),
        )
        .fill(Rect::new(1.0, 0.0, 1.0, 1.0), Color::BLACK)
        .layer(
            Layer::new()
                .try_backdrop_filter(
                    BackdropFilterInput::try_new(
                        FilterList::try_ops(vec![blur, brightness]).unwrap(),
                        bounds,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap(),
            |_| {},
        );

    let Some(color_first) =
        render_scene_to_headless_or_skip_no_adapter(&color_before_blur, Size::new(3.0, 1.0))
    else {
        return;
    };
    let Some(blur_first) =
        render_scene_to_headless_or_skip_no_adapter(&blur_before_color, Size::new(3.0, 1.0))
    else {
        return;
    };
    assert_ne!(
        color_first.rgba, blur_first.rgba,
        "materialized backdrop filters must execute in authored order"
    );

    let clip = ClipInput::try_shape(Shape::rounded_rect(
        Rect::new(1.0, 1.0, 3.0, 3.0),
        Radii::all(1.5),
    ))
    .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::invert(
        UnitFilterAmount::try_new(1.0).unwrap(),
    )])
    .unwrap();
    let clipped_layer = Layer::new()
        .try_backdrop_filter(
            BackdropFilterInput::try_new(
                filters,
                BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 5.0, 5.0)).unwrap(),
                Some(clip),
            )
            .unwrap(),
        )
        .unwrap();
    let mut clipped_scene = Scene::new();
    clipped_scene
        .fill(
            Rect::new(0.0, 0.0, 5.0, 5.0),
            Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
        )
        .layer(clipped_layer, |_| {});
    let Some(clipped) =
        render_scene_to_headless_or_skip_no_adapter(&clipped_scene, Size::new(5.0, 5.0))
    else {
        return;
    };

    let outside_clip = pixel_rgba(&clipped, 0, 0);
    assert!(
        outside_clip[0] > 200 && outside_clip[1] < 80 && outside_clip[2] < 80,
        "filtered backdrop should stay clipped out at the rounded corner: {outside_clip:?}"
    );
    let inside_clip = pixel_rgba(&clipped, 2, 2);
    assert!(
        inside_clip[1] > 200 && inside_clip[2] > 200,
        "filtered backdrop should appear inside the rounded clip: {inside_clip:?}"
    );
}

#[test]
fn sequence13_backdrop_isolation_and_bounded_group_diagnostics_are_explicit() {
    let unsupported_isolation = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropIsolationComposition,
    );
    let unsupported_broad = UnsupportedPrimitive::new(
        PrimitiveFamily::OffscreenPipeline,
        PrimitiveOperation::BackdropExecution,
    );
    for unsupported in [unsupported_isolation, unsupported_broad] {
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("broad backdrop execution must stay diagnostic");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }

    fn backdrop_layer() -> Layer {
        let filters =
            FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
        let bounds = BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 10.0, 10.0)).unwrap();
        Layer::new()
            .try_backdrop_filter(BackdropFilterInput::try_new(filters, bounds, None).unwrap())
            .unwrap()
    }

    let mut nested_scene = Scene::new();
    nested_scene.layer(Layer::new(), |scene| {
        scene.layer(backdrop_layer(), |_| {});
    });
    let nested = nested_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("nested backdrop capture crosses the bounded Sequence 13 path");
    assert_eq!(nested.unsupported_primitive(), Some(unsupported_broad));
    assert!(nested.message().contains("nested backdrop capture"));

    let mut repeated_scene = Scene::new();
    repeated_scene
        .fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK)
        .layer(backdrop_layer(), |_| {})
        .layer(backdrop_layer(), |_| {});
    let repeated = repeated_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("repeated top-level backdrop capture remains bounded");
    assert_eq!(repeated.unsupported_primitive(), Some(unsupported_broad));
    assert!(
        repeated
            .message()
            .contains("repeated top-level backdrop capture")
    );

    let mut transformed_scene = Scene::new();
    transformed_scene.layer(
        backdrop_layer()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap(),
        |_| {},
    );
    let transformed = transformed_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("transformed backdrop capture needs coordinate reconciliation");
    assert_eq!(transformed.unsupported_primitive(), Some(unsupported_broad));
    assert!(
        transformed
            .message()
            .contains("transformed backdrop capture")
    );
}

#[test]
fn sequence13_mix_blend_set_is_direct_vello_only_with_extra_modes_diagnostic() {
    let blend_modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    assert_eq!(
        blend_modes.len(),
        7,
        "public layer BlendMode additions require encoding and reference coverage"
    );

    let source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for mode in blend_modes {
        let mut scene = Scene::new();
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(destination),
        );
        scene.layer(Layer::new().blend(mode), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(source),
            );
        });

        let output = render_scene_pixel(&mut renderer, &scene);
        assert_rgba_near_reference_pixel(
            output,
            source.blend_over(destination, mode),
            2,
            &format!("Sequence 13 direct Vello blend mode {mode:?} should match reference"),
        );
    }

    let backdrop = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let outer_child_backdrop = PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap();
    let inner_source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let expected_inner = inner_source.blend_over(outer_child_backdrop, BlendMode::Multiply);
    let expected_outer = expected_inner.blend_over(backdrop, BlendMode::Screen);
    let mut nested_scene = Scene::new();
    nested_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(backdrop),
    );
    nested_scene.layer(Layer::new().blend(BlendMode::Screen), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(outer_child_backdrop),
        );
        scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(inner_source),
            );
        });
    });
    let normalized = nested_scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer: outer, .. } = &normalized.commands[1] else {
        panic!("expected outer blend layer");
    };
    assert_eq!(
        outer.pass_plan.requirement(),
        command::LayerPassRequirement::DirectVelloBlend
    );
    let nested_output = render_scene_pixel(&mut renderer, &nested_scene);
    assert_rgba_near_reference_pixel(
        nested_output,
        expected_outer,
        2,
        "nested direct Vello blend groups stay implemented in command order",
    );

    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::AdditionalMixBlendMode,
    );
    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("mix-blend modes outside BlendMode remain diagnostic");
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}

#[test]
fn sequence13_root_background_and_composite_boundaries_remain_typed() {
    let filters =
        FilterList::try_ops(vec![FilterOp::blur(FilterBlur::try_new(1.0).unwrap())]).unwrap();
    let root = BackdropFilterInput::try_root_backdrop(filters, None)
        .expect_err("root backdrop policy is not render-owned");
    assert_eq!(
        root.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ))
    );

    let normal_background =
        BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal]).unwrap();
    assert_eq!(normal_background.modes(), &[BackgroundBlendMode::Normal]);
    for mode in [
        BackgroundBlendMode::Multiply,
        BackgroundBlendMode::Screen,
        BackgroundBlendMode::Overlay,
        BackgroundBlendMode::Darken,
        BackgroundBlendMode::Lighten,
        BackgroundBlendMode::Plus,
    ] {
        let error = BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal, mode])
            .expect_err("non-normal background blend lists remain diagnostic");
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::BackgroundBlendMode,
            ))
        );
    }

    let source = PremultipliedRgba8::try_new(80, 40, 20, 128).unwrap();
    let destination = PremultipliedRgba8::try_new(20, 30, 40, 96).unwrap();
    let mask = PremultipliedRgba8::try_new(0, 0, 0, 64).unwrap();
    for pixel in [
        source.blend_over(destination, BlendMode::Normal),
        source.blend_over(destination, BlendMode::Plus),
        source.source_in_alpha_of(mask),
        destination.destination_in_alpha_of(mask),
    ] {
        assert_premultiplied(pixel);
    }

    let porter_duff = UnsupportedPrimitive::new(
        PrimitiveFamily::Compositing,
        PrimitiveOperation::PorterDuffCompositeMode,
    );
    let error = Capabilities::VELLO_0_9
        .ensure_supported(porter_duff)
        .expect_err("Porter-Duff CSS operators stay behind a typed boundary");
    assert_eq!(error.unsupported_primitive(), Some(porter_duff));

    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)), MaskMode::Alpha).unwrap();
    for mode in [
        MaskCompositeMode::Subtract,
        MaskCompositeMode::Intersect,
        MaskCompositeMode::Exclude,
    ] {
        let stack = MaskLayerStack::single(MaskLayer::try_new(alpha_mask.clone(), mode).unwrap());
        let error = stack
            .ensure_supported(Capabilities::VELLO_0_9)
            .expect_err("non-default mask composites remain diagnostic");
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MaskCompositeMode,
            ))
        );
    }
}

#[test]
fn sequence13_vello_0_9_advertises_exact_narrow_backdrop_and_compositing_contract() {
    let capabilities = Capabilities::VELLO_0_9;
    let offscreen = capabilities.offscreen_pipeline();
    assert!(offscreen.supports_direct_vello_opacity_isolation());
    assert!(offscreen.supports_direct_vello_blend_isolation());
    assert!(offscreen.supports_bounded_backdrop_capture());
    assert!(offscreen.supports_materialized_backdrop_filter_execution());
    assert!(!offscreen.supports_offscreen_layer_rendering());
    assert!(!offscreen.supports_texture_cache_upload_lifecycle());
    assert!(!offscreen.supports_rect_fullscreen_shader_passes());
    assert!(!offscreen.supports_cpu_reference_buffers());
    assert!(!offscreen.supports_nested_opacity_planning());
    assert!(!offscreen.supports_mask_execution());
    assert!(!offscreen.supports_filter_execution());
    assert!(!offscreen.supports_backdrop_execution());
    assert!(!offscreen.supports_backdrop_isolation_composition());

    let compositing = capabilities.compositing();
    assert!(compositing.supports_layer_opacity());
    assert!(compositing.supports_blend_modes());
    assert!(!compositing.supports_root_backdrop_policy());
    assert!(!compositing.supports_background_blend_modes());
    assert!(!compositing.supports_additional_mix_blend_modes());
    assert!(!compositing.supports_porter_duff_composite_modes());

    let masks = capabilities.masks_clips();
    assert!(masks.supports_shape_clips());
    assert!(masks.supports_materialized_alpha_mask_execution());
    assert!(!masks.supports_clip_reference_execution());
    assert!(!masks.supports_layer_masks());
    assert!(!masks.supports_luminance_mask_mode());
    assert!(!masks.supports_multi_layer_mask_composition());
    assert!(!masks.supports_mask_composite_modes());

    for supported in [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BoundedBackdropCapture,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::MaterializedBackdropFilterExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaterializedAlphaMaskExecution,
        ),
    ] {
        capabilities
            .ensure_supported(supported)
            .expect("narrow Sequence 13 capability should be advertised");
    }

    for unsupported in [
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropExecution,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropIsolationComposition,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::BackgroundBlendMode,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::AdditionalMixBlendMode,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::PorterDuffCompositeMode,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ),
    ] {
        let error = capabilities
            .ensure_supported(unsupported)
            .expect_err("broader Sequence 13 behavior must not be advertised");
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn background_blend_lists_model_normal_layers_and_reject_blend_modes() {
    let list = BackgroundBlendList::try_new(vec![
        BackgroundBlendMode::Normal,
        BackgroundBlendMode::Normal,
    ])
    .expect("normal-only background blending is a no-op model");

    assert_eq!(
        list.modes(),
        &[BackgroundBlendMode::Normal, BackgroundBlendMode::Normal]
    );

    let error = BackgroundBlendList::try_new(vec![
        BackgroundBlendMode::Normal,
        BackgroundBlendMode::Multiply,
    ])
    .expect_err("non-normal background blend execution is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Compositing,
            PrimitiveOperation::BackgroundBlendMode,
        ))
    );
}

#[test]
fn filter_blur_rejects_negative_radius() {
    let error = FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_unit_amount_rejects_out_of_range_value() {
    let error = UnitFilterAmount::try_new(1.5)
        .expect_err("unit filter amounts must be clamped before render");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter unit amount")
    );
}

#[test]
fn filter_angle_rejects_nan() {
    let error = FilterAngle::try_radians(f64::NAN).expect_err("filter angles must be finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter angle")
    );
}

#[test]
fn clip_inputs_preserve_shape_or_unresolved_reference() {
    let shape = Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap());
    let clip = ClipInput::try_shape(shape.clone()).unwrap();
    let reference = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());

    assert_eq!(clip.shape(), Some(&shape));
    assert_eq!(
        reference.reference_ref().map(StyleResourceRef::identifier),
        Some("#clip")
    );
}

#[test]
fn mask_inputs_preserve_mode_and_source() {
    let mask = MaskInput::try_shape(
        Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap()),
        MaskMode::Luminance,
    )
    .unwrap();

    assert_eq!(mask.mode(), MaskMode::Luminance);
    assert!(matches!(mask.source().kind(), MaskSourceKind::Shape(_)));
}

#[test]
fn repeated_mask_layers_remain_distinct_in_authored_order() {
    let mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();

    let stack = MaskLayerStack::try_new([
        MaskLayer::new(mask.clone()),
        MaskLayer::new(mask.clone()),
        MaskLayer::new(mask),
    ])
    .unwrap();

    assert_eq!(stack.len(), 3);
    assert_eq!(stack.layers()[0], stack.layers()[1]);
    assert_eq!(stack.layers()[1], stack.layers()[2]);
}

#[test]
fn ordered_mask_layer_stacks_preserve_layer_and_composite_lists() {
    let first =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();
    let second =
        MaskInput::try_shape(Shape::rect(Rect::new(1.0, 0.0, 3.0, 4.0)), MaskMode::Alpha).unwrap();

    let stack = MaskLayerStack::try_new([
        MaskLayer::new(first.clone()),
        MaskLayer::try_new(second.clone(), MaskCompositeMode::Add).unwrap(),
    ])
    .unwrap();

    assert_eq!(stack.layers()[0].input(), &first);
    assert_eq!(stack.layers()[1].input(), &second);
    assert_eq!(stack.layers()[0].composite_mode(), MaskCompositeMode::Add);
    assert_eq!(stack.layers()[1].composite_mode(), MaskCompositeMode::Add);
}

#[test]
fn mask_layer_stacks_validate_empty_lists_and_single_layer_diagnostics() {
    let error = MaskLayerStack::try_new([]).expect_err("mask layer lists must not be empty");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("mask layer stack")
    );

    let stack = MaskLayerStack::single(
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap(),
    );
    let error = stack
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("single authored alpha masks still stop at source execution");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );
}

#[test]
fn mask_layer_stacks_report_specific_luminance_and_composite_diagnostics() {
    let luminance = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let luminance_error = luminance
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("luminance mask stacks need a typed unsupported diagnostic");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let composite = MaskLayerStack::single(
        MaskLayer::try_new(
            MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha)
                .unwrap(),
            MaskCompositeMode::Intersect,
        )
        .unwrap(),
    );
    let composite_error = composite
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("non-default mask composite modes are not implemented");
    assert_eq!(
        composite_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ))
    );
}

#[test]
fn multi_layer_mask_stacks_report_composition_boundary_after_input_validation() {
    let first =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha).unwrap();
    let second =
        MaskInput::try_shape(Shape::rect(Rect::new(1.0, 0.0, 3.0, 4.0)), MaskMode::Alpha).unwrap();
    let stack = MaskLayerStack::try_new([MaskLayer::new(first), MaskLayer::new(second)]).unwrap();

    let error = stack
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("true multi-layer mask composition is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ))
    );

    let unresolved = MaskLayerStack::try_new([
        MaskLayer::new(
            MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 4.0, 4.0)), MaskMode::Alpha)
                .unwrap(),
        ),
        MaskLayer::new(MaskInput::reference(
            StyleResourceRef::try_new("#stack-mask").unwrap(),
            MaskMode::Alpha,
        )),
    ])
    .unwrap();

    let error = unresolved
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("unresolved references remain a narrower diagnostic than composition");
    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("#stack-mask")
    );
}

#[test]
fn mask_layer_stack_model_does_not_change_unmasked_render_paths() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.fill(Rect::new(1.0, 1.0, 2.0, 2.0), Color::BLACK);
        });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(scene.stats().fills, 2);
    assert_eq!(scene.stats().layers, 1);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Fill { .. },
            command::RenderCommand::Layer { .. }
        ]
    ));
}

#[test]
fn masks_and_clips_can_carry_coordinate_space_tags() {
    let tag = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap()
        .with_coordinate_space(tag);
    let mask = MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)), MaskMode::Alpha)
        .unwrap()
        .with_coordinate_space(tag);

    assert_eq!(clip.coordinate_space(), Some(tag));
    assert_eq!(mask.coordinate_space(), Some(tag));
}

#[test]
fn clip_inputs_diagnose_unresolved_reference_boundaries() {
    let clip = ClipInput::reference(StyleResourceRef::try_new("#content-clip").unwrap());

    let error = clip
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("clip references must be root-resolved before render execution");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    let diagnostic = error
        .unresolved_resource_diagnostic()
        .expect("clip references should report an unresolved resource");
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Clip);
    assert_eq!(diagnostic.identifier(), "#content-clip");
}

#[test]
fn shape_clip_inputs_match_current_capability_contract() {
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0))).unwrap();

    clip.ensure_supported(Capabilities::VELLO_0_9)
        .expect("shape clips are supported by the current Vello layer path");
    assert!(Capabilities::VELLO_0_9.masks_clips().supports_shape_clips());
}

#[test]
fn mask_inputs_diagnose_current_unexecuted_boundaries() {
    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)), MaskMode::Alpha).unwrap();
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255])).unwrap();
    let image_layer = StyleImageLayer::try_new(StyleImageSource::image(image).unwrap()).unwrap();
    let image_mask = MaskInput::image_layer(image_layer, MaskMode::Alpha);
    let luminance_mask = MaskInput::try_shape(
        Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)),
        MaskMode::Luminance,
    )
    .unwrap();
    let transformed_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 8.0, 6.0)), MaskMode::Alpha)
            .unwrap()
            .with_coordinate_space(
                CoordinateSpaceTag::surface(Transform::translation(1.0, 0.0).unwrap()).unwrap(),
            );
    let reference_mask = MaskInput::reference(
        StyleResourceRef::try_new("#alpha-mask").unwrap(),
        MaskMode::Alpha,
    );

    let alpha_error = alpha_mask
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("shape masks need a real rasterization path before execution");
    assert_eq!(
        alpha_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let image_error = image_mask
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("image-layer masks need materialized placement before execution");
    assert_eq!(
        image_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let transformed_error = transformed_mask
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("transformed authored masks need materialized execution inputs");
    assert_eq!(
        transformed_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let luminance_error = luminance_mask
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("luminance mask mode is not implemented in Task 1");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let reference_error = reference_mask
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("mask references must be root-resolved before render execution");
    assert_eq!(reference_error.code(), ErrorCode::UnresolvedResource);
    let diagnostic = reference_error
        .unresolved_resource_diagnostic()
        .expect("mask references should report an unresolved resource");
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Mask);
    assert_eq!(diagnostic.identifier(), "#alpha-mask");
}

#[test]
fn sequence12_executes_shape_and_basic_shape_clips_from_render_owned_geometry() {
    let rect = Rect::new(0.0, 0.0, 2.0, 2.0);
    let rounded = Shape::try_rounded_rect(rect, Radii::try_all(0.5).unwrap()).unwrap();
    let circle = Shape::try_circle(Point::new(1.0, 1.0), 1.0).unwrap();
    let ellipse = Shape::try_ellipse(Point::new(1.0, 1.0), Size::new(1.0, 0.75)).unwrap();
    let clips = [
        (Shape::rect(rect), ClipGeometryKind::Rect(rect)),
        (
            rounded,
            ClipGeometryKind::RoundedRect {
                rect,
                radii: Radii::try_all(0.5).unwrap(),
            },
        ),
        (
            circle,
            ClipGeometryKind::Circle {
                center: Point::new(1.0, 1.0),
                radius: 1.0,
            },
        ),
        (
            ellipse,
            ClipGeometryKind::Ellipse {
                center: Point::new(1.0, 1.0),
                radii: Size::new(1.0, 0.75),
            },
        ),
    ];

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for (shape, expected_geometry) in clips {
        let normalized = ClipInput::try_shape(shape.clone())
            .unwrap()
            .normalize(Capabilities::VELLO_0_9)
            .unwrap();
        assert_eq!(normalized.geometry().kind(), &expected_geometry);

        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(3.0, 2.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(Layer::new().try_clip(shape).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 3.0, 2.0), Color::BLACK);
        });

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("Sequence 12 shape/basic-shape clips should execute through layer clipping");
        let output = renderer.read_headless(&surface).unwrap();

        assert!(pixel_alpha(&output, 0, 0) > 0);
        assert_eq!(pixel_alpha(&output, 2, 0), 0);
    }
}

#[test]
fn sequence12_path_clip_execution_preserves_fill_rule_behavior() {
    fn nested_rect_path() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(5.0, 0.0))
            .line_to(Point::new(5.0, 5.0))
            .line_to(Point::new(0.0, 5.0))
            .close()
            .move_to(Point::new(1.0, 1.0))
            .line_to(Point::new(4.0, 1.0))
            .line_to(Point::new(4.0, 4.0))
            .line_to(Point::new(1.0, 4.0))
            .close();
        path
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut outputs = Vec::new();
    for fill_rule in [FillRule::EvenOdd, FillRule::NonZero] {
        let filled_path = FilledPath::try_new(nested_rect_path(), fill_rule).unwrap();
        let normalized = ClipInput::try_filled_path(filled_path.clone())
            .unwrap()
            .normalize(Capabilities::VELLO_0_9)
            .unwrap();
        assert_eq!(
            normalized.geometry().kind(),
            &ClipGeometryKind::Path(filled_path.clone())
        );

        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(5.0, 5.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(
            Layer::new()
                .try_clip_input(ClipInput::try_filled_path(filled_path).unwrap())
                .unwrap(),
            |scene| {
                scene.fill(Rect::new(0.0, 0.0, 5.0, 5.0), Color::BLACK);
            },
        );

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("Sequence 12 path clips should execute with their authored fill rule");
        outputs.push(renderer.read_headless(&surface).unwrap());
    }

    assert_eq!(pixel_alpha(&outputs[0], 2, 2), 0);
    assert!(pixel_alpha(&outputs[1], 2, 2) > 0);
}

#[test]
fn sequence12_reports_typed_clip_and_mask_diagnostics_for_unresolved_or_later_inputs() {
    let clip = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());
    let clip_error = clip
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("unresolved clip references remain root-owned");
    assert_eq!(clip_error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        clip_error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );

    let luminance_stack = MaskLayerStack::single(
        MaskInput::try_shape(
            Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)),
            MaskMode::Luminance,
        )
        .unwrap(),
    );
    let luminance_error = luminance_stack
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("luminance mask conversion is outside Sequence 12");
    assert_eq!(
        luminance_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LuminanceMaskMode,
        ))
    );

    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)), MaskMode::Alpha).unwrap();
    let source_error = MaskLayerStack::single(alpha_mask.clone())
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("authored alpha mask sources still need materialization before execution");
    assert_eq!(
        source_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::AlphaMaskSourceExecution,
        ))
    );

    let multi_layer_error = MaskLayerStack::try_new([
        MaskLayer::new(alpha_mask.clone()),
        MaskLayer::new(alpha_mask.clone()),
    ])
    .unwrap()
    .ensure_supported(Capabilities::VELLO_0_9)
    .expect_err("multi-layer mask composition has a typed Sequence 12 boundary");
    assert_eq!(
        multi_layer_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MultiLayerMaskComposition,
        ))
    );

    let composite_error =
        MaskLayerStack::single(MaskLayer::try_new(alpha_mask, MaskCompositeMode::Exclude).unwrap())
            .ensure_supported(Capabilities::VELLO_0_9)
            .expect_err("non-default mask composites have a typed Sequence 12 boundary");
    assert_eq!(
        composite_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaskCompositeMode,
        ))
    );
}

#[test]
fn sequence12_executes_materialized_alpha_masks_for_resolved_buffers_and_layers() {
    let source = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 0, 0, 255, 0, 255, 0, 255],
    };
    let mask = ImageBuffer {
        size: PhysicalSize::new(2, 1),
        rgba: vec![255, 255, 255, 255, 0, 0, 0, 128],
    };
    let masked = ResolvedAlphaMaskExecution::try_new(&source, &mask)
        .unwrap()
        .execute_to_image_buffer()
        .unwrap();
    assert_eq!(masked.rgba, vec![255, 0, 0, 255, 0, 255, 0, 128]);

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 1.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_resolved_alpha_mask(mask).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 1.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("Sequence 12 resolved layer alpha masks should execute");
    let output = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 200);
    assert!((96..=160).contains(&pixel_alpha(&output, 1, 0)));
}

#[test]
fn sequence12_capabilities_claim_only_implemented_mask_clip_and_no_broad_backdrop_behavior() {
    let capabilities = Capabilities::VELLO_0_9;
    let masks_clips = capabilities.masks_clips();
    assert!(masks_clips.supports_shape_clips());
    assert!(masks_clips.supports_materialized_alpha_mask_execution());
    assert!(!masks_clips.supports_clip_reference_execution());
    assert!(!masks_clips.supports_layer_masks());
    assert!(!masks_clips.supports_luminance_mask_mode());
    assert!(!masks_clips.supports_multi_layer_mask_composition());
    assert!(!masks_clips.supports_mask_composite_modes());

    capabilities
        .ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaterializedAlphaMaskExecution,
        ))
        .expect("materialized alpha-mask execution is the narrow supported mask execution path");
    for operation in [
        PrimitiveOperation::ClipReferenceExecution,
        PrimitiveOperation::LayerMask,
        PrimitiveOperation::AlphaMaskSourceExecution,
        PrimitiveOperation::LuminanceMaskMode,
        PrimitiveOperation::MultiLayerMaskComposition,
        PrimitiveOperation::MaskCompositeMode,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::MasksAndClips, operation);
        assert_eq!(
            capabilities
                .ensure_supported(unsupported)
                .expect_err("broader mask/clip behavior must not be claimed early")
                .unsupported_primitive(),
            Some(unsupported)
        );
    }

    let offscreen = capabilities.offscreen_pipeline();
    assert!(offscreen.supports_direct_vello_opacity_isolation());
    assert!(offscreen.supports_direct_vello_blend_isolation());
    assert!(!offscreen.supports_offscreen_layer_rendering());
    assert!(!offscreen.supports_mask_execution());
    assert!(!offscreen.supports_filter_execution());
    assert!(!offscreen.supports_backdrop_execution());

    assert!(
        capabilities
            .filters()
            .supports_materialized_blur_filter_execution()
    );
    assert!(
        capabilities
            .filters()
            .supports_materialized_drop_shadow_filter_execution()
    );
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(!capabilities.shadows().supports_text_shadows());
}

#[test]
fn clip_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid clip paths fail");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn mask_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = MaskInput::try_shape(Shape::path(path), MaskMode::Alpha)
        .expect_err("invalid mask paths fail");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn paths_expose_elements_without_exposing_mutation() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .close();

    assert_eq!(path.elements().len(), 3);
    assert!(matches!(path.elements()[0], PathElement::MoveTo(_)));
}

#[test]
fn filled_paths_preserve_fill_rule_intent() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 4.0).unwrap())
        .close();
    let filled = FilledPath::try_new(path.clone(), FillRule::EvenOdd).unwrap();

    assert_eq!(filled.path(), &path);
    assert_eq!(filled.fill_rule(), FillRule::EvenOdd);
}

#[test]
fn filled_paths_reject_invalid_path_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = FilledPath::try_new(path, FillRule::NonZero)
        .expect_err("filled paths validate stored path elements");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn clip_input_normalization_lowers_concrete_shape_geometry() {
    let rect = Rect::new(1.0, 2.0, 3.0, 4.0);
    let radii = Radii::new(1.0, 2.0, 3.0, 4.0);
    let circle_center = Point::new(8.0, 9.0);
    let ellipse_center = Point::new(12.0, 13.0);
    let ellipse_radii = Size::new(4.0, 5.0);
    let cases = [
        (
            ClipInput::try_shape(Shape::rect(rect)).unwrap(),
            ClipGeometryKind::Rect(rect),
        ),
        (
            ClipInput::try_shape(Shape::try_rounded_rect(rect, radii).unwrap()).unwrap(),
            ClipGeometryKind::RoundedRect { rect, radii },
        ),
        (
            ClipInput::try_shape(Shape::try_circle(circle_center, 3.0).unwrap()).unwrap(),
            ClipGeometryKind::Circle {
                center: circle_center,
                radius: 3.0,
            },
        ),
        (
            ClipInput::try_shape(Shape::try_ellipse(ellipse_center, ellipse_radii).unwrap())
                .unwrap(),
            ClipGeometryKind::Ellipse {
                center: ellipse_center,
                radii: ellipse_radii,
            },
        ),
    ];

    for (input, expected) in cases {
        let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

        assert_eq!(normalized.geometry().kind(), &expected);
        assert_eq!(normalized.coordinate_space(), None);
    }
}

#[test]
fn clip_input_normalization_preserves_path_fill_rules_and_bounds() {
    let mut path = Path::new();
    path.move_to(Point::new(2.0, 3.0))
        .line_to(Point::new(6.0, 3.0))
        .line_to(Point::new(6.0, 8.0))
        .close();
    let filled = FilledPath::try_new(path.clone(), FillRule::EvenOdd).unwrap();
    let input = ClipInput::try_filled_path(filled.clone()).unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(
        normalized.geometry().kind(),
        &ClipGeometryKind::Path(filled)
    );

    let layer = Layer::new()
        .try_clip_input(
            ClipInput::try_filled_path(FilledPath::try_new(path, FillRule::NonZero).unwrap())
                .unwrap(),
        )
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(-10.0, -10.0, 40.0, 40.0), Color::BLACK);
    });
    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(2.0, 3.0, 4.0, 5.0))
    );
    assert!(matches!(
        layer.clip.as_ref().map(|clip| clip.geometry()),
        Some(command::RenderClipGeometry::Path {
            fill_rule: FillRule::NonZero,
            ..
        })
    ));
}

#[test]
fn clip_input_normalization_reports_reference_and_invalid_path_diagnostics() {
    let reference = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());
    let error = reference
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("unresolved clip references should stay a typed diagnostic");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::kind),
        Some(UnresolvedResourceKind::Clip)
    );
    assert_eq!(
        error
            .unresolved_resource_diagnostic()
            .map(UnresolvedResource::identifier),
        Some("#clip")
    );

    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));
    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid path points fail");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn clip_input_normalization_preserves_coordinate_space_tags_and_rejects_nonfinite_bounds() {
    let tag = CoordinateSpaceTag::surface(Transform::translation(4.0, 5.0).unwrap()).unwrap();
    let normalized = ClipInput::try_shape(Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0)))
        .unwrap()
        .with_coordinate_space(tag)
        .normalize(Capabilities::VELLO_0_9)
        .unwrap();

    assert_eq!(normalized.coordinate_space(), Some(tag));

    let huge = ClipInput::try_shape(Shape::rect(Rect::new(f64::MAX, 0.0, 1.0, 1.0)))
        .unwrap()
        .with_coordinate_space(
            CoordinateSpaceTag::surface(Transform::scale(2.0, 1.0).unwrap()).unwrap(),
        );
    let error = huge
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("transformed clip bounds must remain finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("clip transformed bounds")
    );
}

#[test]
fn border_edges_preserve_four_independent_sides() {
    let top = BorderSide::try_new(BorderStyle::Solid, 1.0, Color::BLACK).unwrap();
    let right = BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::BLACK).unwrap();
    let bottom = BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap();
    let left = BorderSide::try_new(BorderStyle::Double, 4.0, Color::BLACK).unwrap();
    let edges = BorderEdges::new(top.clone(), right.clone(), bottom.clone(), left.clone());

    assert_eq!(edges.top(), &top);
    assert_eq!(edges.right(), &right);
    assert_eq!(edges.bottom(), &bottom);
    assert_eq!(edges.left(), &left);
}

#[test]
fn background_stacks_preserve_color_behind_ordered_layers() {
    let layer_a = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let layer_b = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack =
        BackgroundStack::try_new(Some(Color::BLACK), vec![layer_a.clone(), layer_b.clone()])
            .unwrap();

    assert_eq!(stack.color(), Some(Color::BLACK));
    assert_eq!(stack.layers(), &[layer_a, layer_b]);
}

#[test]
fn background_areas_select_origin_and_clip_boxes() {
    let areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        Rect::new(10.0, 8.0, 100.0, 60.0),
        Rect::new(20.0, 18.0, 80.0, 40.0),
    )
    .unwrap();

    assert_eq!(
        areas.rect_for(BackgroundBox::Border),
        Rect::new(0.0, 0.0, 120.0, 80.0)
    );
    assert_eq!(
        areas.rect_for(BackgroundBox::Padding),
        Rect::new(10.0, 8.0, 100.0, 60.0)
    );
    assert_eq!(
        areas.rect_for(BackgroundBox::Content),
        Rect::new(20.0, 18.0, 80.0, 40.0)
    );
}

#[test]
fn background_areas_reject_invalid_rects() {
    let error = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 100.0),
        Rect::new(0.0, 0.0, 0.0, 50.0),
        Rect::new(0.0, 0.0, 10.0, 10.0),
    )
    .expect_err("background areas require positive boxes");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background padding box")
    );
}

#[test]
fn background_clip_geometry_preserves_box_or_shape_inputs() {
    let rect_clip = BackgroundClipGeometry::try_rect(Rect::new(0.0, 0.0, 12.0, 8.0)).unwrap();
    assert_eq!(
        rect_clip.kind(),
        &BackgroundClipGeometryKind::Rect(Rect::new(0.0, 0.0, 12.0, 8.0))
    );

    let shape = Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0));
    let shape_clip = BackgroundClipGeometry::try_shape(shape.clone()).unwrap();
    assert_eq!(shape_clip.shape(), Some(&shape));
}

#[test]
fn background_stack_normalization_paints_color_behind_layers() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top, back]).unwrap();
    let input = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    assert_eq!(normalized.commands().len(), 3);
    let NormalizedBackgroundCommandKind::ColorFill { color, .. } = normalized.commands()[0].kind()
    else {
        panic!("expected background color command");
    };
    assert_eq!(*color, Color::BLACK);
    assert!(matches!(
        normalized.commands()[1].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
    assert!(matches!(
        normalized.commands()[2].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
}

#[test]
fn background_normalization_mixes_color_paint_and_image_layers_in_render_order() {
    let image = Image::from_rgba(Size::new(10.0, 10.0), vec![255; 10 * 10 * 4]).unwrap();
    let top_image = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image).unwrap())
            .unwrap()
            .with_size(BackgroundSize::auto())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let back_paint = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top_image, back_paint]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert!(matches!(
        normalized.commands()[0].kind(),
        NormalizedBackgroundCommandKind::ColorFill { .. }
    ));
    let NormalizedBackgroundCommandKind::Layer { layer: back_layer } =
        normalized.commands()[1].kind()
    else {
        panic!("expected back layer command");
    };
    assert!(matches!(
        back_layer.source(),
        NormalizedBackgroundLayerSource::Paint(_)
    ));

    let NormalizedBackgroundCommandKind::Layer { layer: top_layer } =
        normalized.commands()[2].kind()
    else {
        panic!("expected top layer command");
    };
    assert!(matches!(
        top_layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
}

#[test]
fn background_stack_normalization_preserves_top_layer_as_last_render_command() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Content),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Padding),
    );
    let stack = BackgroundStack::try_new(None, vec![top, back]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let last = normalized.commands().last().unwrap();
    assert_eq!(last.clip().rect(), Some(Rect::new(8.0, 8.0, 84.0, 44.0)));
}

#[test]
fn background_stack_normalization_preserves_paint_layer_sampling_semantics() {
    let paint_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 1.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_percent(0.5).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_y())
            .with_attachment(BackgroundAttachment::Local)
            .with_coordinate_space(CoordinateSpaceTag::local()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![paint_layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 60.0),
            Rect::new(20.0, 20.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind() else {
        panic!("expected normalized paint-backed layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Paint(_)
    ));
    assert_eq!(
        layer.placement().paint_rect(),
        Rect::new(20.0, 20.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(60.0, 40.0, 40.0, 20.0)
    );
    assert_eq!(
        layer.repeat().clip_rect(),
        Rect::new(20.0, 20.0, 80.0, 40.0)
    );
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Local);
}

#[test]
fn background_stack_normalizes_image_layers_with_origin_clip_repeat_and_attachment() {
    let image = Image::from_rgba(Size::new(20.0, 10.0), vec![255; 20 * 10 * 4]).unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image.clone()).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_length(40.0).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_x())
            .with_attachment(BackgroundAttachment::Fixed)
            .with_coordinate_space(
                CoordinateSpaceTag::viewport(Transform::translation(1.0, 2.0).unwrap()).unwrap(),
            ),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(5.0, 5.0, 90.0, 50.0),
            Rect::new(10.0, 10.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let command = normalized.commands().first().unwrap();
    assert_eq!(command.clip().rect(), Some(Rect::new(5.0, 5.0, 90.0, 50.0)));
    let NormalizedBackgroundCommandKind::Layer { layer } = command.kind() else {
        panic!("expected normalized image layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
    assert_eq!(
        layer.placement().paint_rect(),
        Rect::new(10.0, 10.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(50.0, 10.0, 40.0, 20.0)
    );
    assert_eq!(
        layer.repeat().clip_rect(),
        Rect::new(10.0, 10.0, 80.0, 40.0)
    );
    assert_eq!(
        layer.repeat().tile_rects(),
        &[
            Rect::new(10.0, 10.0, 40.0, 20.0),
            Rect::new(50.0, 10.0, 40.0, 20.0),
        ]
    );
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Fixed);
}

#[test]
fn background_stack_normalizes_resolved_image_layers_with_intrinsic_size() {
    let resource =
        ResolvedImageResource::try_new(ImageId::new(400), Size::new(30.0, 10.0)).unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
            .unwrap()
            .with_origin(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(0.5, 0.5).unwrap())
            .with_size(BackgroundSize::contain())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 50.0),
            Rect::new(20.0, 20.0, 80.0, 30.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind() else {
        panic!("expected normalized layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::ResolvedImage(_)
    ));
    assert_eq!(
        layer.placement().tile_rect(),
        Rect::new(10.0, 18.333333333333332, 100.0, 33.333333333333336)
    );
}

#[test]
fn background_stack_reports_unresolved_image_layers() {
    let source = StyleImageSource::unresolved(StyleResourceRef::try_new("hero.png").unwrap());
    let layer = BackgroundLayer::new(StyleImageLayer::try_new(source).unwrap());
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .expect_err("unresolved image layer should fail normalization");

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    let diagnostic = error.unresolved_resource_diagnostic().unwrap();
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Image);
    assert_eq!(diagnostic.identifier(), "hero.png");
}

#[test]
fn background_normalization_rejects_clip_override_length_mismatch() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(Vec::new())
    .expect_err("clip override list must match background layer count");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background layer clip overrides")
    );
}

#[test]
fn background_normalization_accepts_shape_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let shape = Shape::rect(Rect::new(1.0, 1.0, 8.0, 8.0));
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(
        BackgroundClipGeometry::try_shape(shape.clone()).unwrap(),
    )])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}

#[test]
fn background_normalization_accepts_path_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let mut path = Path::new();
    path.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0))
        .line_to(Point::new(10.0, 10.0))
        .close();
    let shape = Shape::path(path);
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(
        BackgroundClipGeometry::try_shape(shape.clone()).unwrap(),
    )])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}

#[test]
fn core_style_models_compose_without_backend_lowering() {
    let color = StyleColor::new(Color::BLACK);
    let paint = Paint::from(color.color());
    let image_layer = StyleImageLayer::try_new(StyleImageSource::paint(paint).unwrap()).unwrap();
    let background = BackgroundStack::try_new(
        Some(Color::TRANSPARENT),
        vec![BackgroundLayer::new(image_layer.clone())],
    )
    .unwrap();
    let filter = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(0.5).unwrap(),
    )])
    .unwrap();
    let mask = MaskInput::image_layer(image_layer, MaskMode::Alpha);
    let outline = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, 2.0).unwrap();

    assert_eq!(background.layers().len(), 1);
    assert_eq!(filter.ops().len(), 1);
    assert_eq!(mask.mode(), MaskMode::Alpha);
    assert_eq!(outline.offset(), 2.0);
}

#[test]
fn border_sides_reject_negative_width() {
    let error = BorderSide::try_new(BorderStyle::Solid, -1.0, Color::BLACK)
        .expect_err("negative border widths should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("border side width")
    );
}

#[test]
fn outlines_reject_non_finite_offset() {
    let error = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, f64::NAN)
        .expect_err("outline offset must be finite");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("outline offset")
    );
}

fn box_decoration_test_areas() -> BackgroundAreas {
    BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 40.0),
        Rect::new(4.0, 4.0, 92.0, 32.0),
        Rect::new(8.0, 8.0, 84.0, 24.0),
    )
    .unwrap()
}

#[test]
fn box_decoration_fragments_normalize_border_box_radii_on_construction() {
    let areas = box_decoration_test_areas();
    let radii = Radii::try_new(10.0, 12.0, 14.0, 16.0).unwrap();

    let fragment = BoxDecorationFragment::try_new(areas, radii, BoxDecorationBreak::Slice).unwrap();

    assert_eq!(fragment.areas(), areas);
    assert_eq!(fragment.radii().border_box(), areas.border_box());
    assert_eq!(fragment.radii().radii(), radii);
    assert_eq!(fragment.break_mode(), BoxDecorationBreak::Slice);
    assert_eq!(fragment.border_clip_override(), None);
}

#[test]
fn box_decoration_inputs_reject_empty_fragments() {
    let error = BoxDecorationInput::try_new(None, None, Vec::new())
        .expect_err("box decoration inputs require at least one fragment");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("box decoration fragments")
    );
}

#[test]
fn box_decoration_inputs_preserve_border_outline_and_break_mode() {
    let side = BorderSide::try_new(BorderStyle::Solid, 2.0, Color::BLACK).unwrap();
    let edges = BorderEdges::new(side.clone(), side.clone(), side.clone(), side);
    let outline = Outline::try_new(OutlineStyle::Dashed, 3.0, Color::TRANSPARENT, 1.5).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();

    let input = BoxDecorationInput::try_new(
        Some(edges.clone()),
        Some(outline.clone()),
        vec![fragment.clone()],
    )
    .unwrap();

    assert_eq!(input.border_edges(), Some(&edges));
    assert_eq!(input.outline(), Some(&outline));
    assert_eq!(input.fragments(), &[fragment]);
    assert_eq!(input.fragments()[0].break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_radii_scale_against_horizontal_and_vertical_limits() {
    let areas = box_decoration_test_areas();
    let radii = Radii::try_new(80.0, 80.0, 20.0, 20.0).unwrap();

    let fragment = BoxDecorationFragment::try_new(areas, radii, BoxDecorationBreak::Slice).unwrap();

    assert_eq!(
        fragment.radii().radii(),
        Radii::try_new(32.0, 32.0, 8.0, 8.0).unwrap()
    );
}

#[test]
fn box_decoration_fragments_validate_clip_override_geometry() {
    let error = BackgroundClipGeometry::try_rect(Rect::new(0.0, 0.0, 0.0, 10.0))
        .expect_err("clip overrides reuse background clip validation");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background clip rect")
    );
}

#[test]
fn box_decoration_fragments_preserve_border_clip_override() {
    let shape = Shape::rect(Rect::new(1.0, 2.0, 30.0, 20.0));
    let clip = BackgroundClipGeometry::try_shape(shape.clone()).unwrap();

    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(5.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap()
    .with_border_clip_override(clip.clone());

    assert_eq!(fragment.border_clip_override(), Some(&clip));
    assert_eq!(
        fragment
            .border_clip_override()
            .and_then(|clip| clip.shape()),
        Some(&shape)
    );
}

fn box_decoration_edges(
    top: BorderSide,
    right: BorderSide,
    bottom: BorderSide,
    left: BorderSide,
) -> BorderEdges {
    BorderEdges::new(top, right, bottom, left)
}

fn solid_border(width: f64, color: Color) -> BorderSide {
    BorderSide::try_new(BorderStyle::Solid, width, color).unwrap()
}

fn normalized_border_command(command: &NormalizedBoxDecorationCommand) -> &NormalizedBorderCommand {
    match command.kind() {
        NormalizedBoxDecorationCommandKind::Border(border) => border,
        NormalizedBoxDecorationCommandKind::Outline(_) => panic!("expected border command"),
    }
}

fn normalized_outline_command(
    command: &NormalizedBoxDecorationCommand,
) -> &NormalizedOutlineCommand {
    match command.kind() {
        NormalizedBoxDecorationCommandKind::Outline(outline) => outline,
        NormalizedBoxDecorationCommandKind::Border(_) => panic!("expected outline command"),
    }
}

#[test]
fn box_decoration_normalization_emits_four_independent_border_sides_in_order() {
    let top = BorderSide::try_new(BorderStyle::Solid, 1.0, Color::BLACK).unwrap();
    let right = BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::TRANSPARENT).unwrap();
    let bottom = BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap();
    let left = BorderSide::try_new(BorderStyle::Double, 4.0, Color::TRANSPARENT).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(6.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            top.clone(),
            right.clone(),
            bottom.clone(),
            left.clone(),
        )),
        None,
        vec![fragment.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    let commands = normalized.commands();

    assert_eq!(commands.len(), 4);
    let top_command = normalized_border_command(&commands[0]);
    let right_command = normalized_border_command(&commands[1]);
    let bottom_command = normalized_border_command(&commands[2]);
    let left_command = normalized_border_command(&commands[3]);
    assert_eq!(top_command.side(), BoxSide::Top);
    assert_eq!(right_command.side(), BoxSide::Right);
    assert_eq!(bottom_command.side(), BoxSide::Bottom);
    assert_eq!(left_command.side(), BoxSide::Left);
    assert_eq!(top_command.width(), 1.0);
    assert_eq!(right_command.width(), 2.0);
    assert_eq!(bottom_command.width(), 3.0);
    assert_eq!(left_command.width(), 4.0);
    assert_eq!(top_command.paint(), top.paint());
    assert_eq!(right_command.paint(), right.paint());
    assert_eq!(bottom_command.paint(), bottom.paint());
    assert_eq!(left_command.paint(), left.paint());
    assert_eq!(top_command.style(), &NormalizedBorderStyle::Solid);
    assert_eq!(right_command.style(), &NormalizedBorderStyle::Dashed);
    assert_eq!(bottom_command.style(), &NormalizedBorderStyle::Dotted);
    assert!(matches!(
        left_command.style(),
        NormalizedBorderStyle::Double(_)
    ));
    assert_eq!(
        top_command.target_rect(),
        box_decoration_test_areas().border_box()
    );
    assert_eq!(top_command.fragment_index(), 0);
    assert_eq!(
        top_command.clip().rect(),
        Some(box_decoration_test_areas().border_box())
    );
    assert_eq!(top_command.radii(), fragment.radii());
    assert_eq!(top_command.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_suppresses_none_hidden_and_zero_width_borders() {
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::None, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Hidden, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Solid, 0.0, Color::BLACK).unwrap(),
            solid_border(3.0, Color::BLACK),
        )),
        None,
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands().len(), 1);
    let border = normalized_border_command(&normalized.commands()[0]);
    assert_eq!(border.side(), BoxSide::Left);
    assert_eq!(border.width(), 3.0);
}

#[test]
fn box_decoration_normalization_preserves_dashed_and_dotted_styles() {
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Hidden, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(
        normalized_border_command(&normalized.commands()[0]).style(),
        &NormalizedBorderStyle::Dashed
    );
    assert_eq!(
        normalized_border_command(&normalized.commands()[1]).style(),
        &NormalizedBorderStyle::Dotted
    );
}

fn assert_double_bands(bands: NormalizedDoubleBorderBands, width: f64) {
    assert_eq!(bands.original_width(), width);
    assert!(bands.outer_width() >= 0.0);
    assert!(bands.gap_width() >= 0.0);
    assert!(bands.inner_width() >= 0.0);
    let sum = bands.outer_width() + bands.gap_width() + bands.inner_width();
    assert!(
        (sum - width).abs() < f64::EPSILON,
        "double bands should sum to {width}, got {sum}"
    );
}

#[test]
fn box_decoration_normalization_computes_double_bands_for_thin_medium_and_large_widths() {
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(48.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            BorderSide::try_new(BorderStyle::Double, 1.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Double, 2.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::Double, 9.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![fragment],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands().len(), 3);
    let thin = normalized_border_command(&normalized.commands()[0]);
    let medium = normalized_border_command(&normalized.commands()[1]);
    let large = normalized_border_command(&normalized.commands()[2]);
    let NormalizedBorderStyle::Double(thin_bands) = thin.style() else {
        panic!("expected thin double border bands");
    };
    let NormalizedBorderStyle::Double(medium_bands) = medium.style() else {
        panic!("expected medium double border bands");
    };
    let NormalizedBorderStyle::Double(large_bands) = large.style() else {
        panic!("expected large double border bands");
    };

    assert_double_bands(*thin_bands, 1.0);
    assert_double_bands(*medium_bands, 2.0);
    assert_double_bands(*large_bands, 9.0);
    assert!(thin_bands.outer_width() > 0.0);
    assert_eq!(large_bands.outer_width(), 3.0);
    assert_eq!(large_bands.gap_width(), 3.0);
    assert_eq!(large_bands.inner_width(), 3.0);
    assert_eq!(thin.radii().radii(), Radii::try_all(20.0).unwrap());
}

#[test]
fn box_decoration_normalization_reports_unsupported_border_styles() {
    for (style, operation) in [
        (BorderStyle::Groove, PrimitiveOperation::BorderGrooveStyle),
        (BorderStyle::Ridge, PrimitiveOperation::BorderRidgeStyle),
        (BorderStyle::Inset, PrimitiveOperation::BorderInsetStyle),
        (BorderStyle::Outset, PrimitiveOperation::BorderOutsetStyle),
    ] {
        let input = BoxDecorationInput::try_new(
            Some(box_decoration_edges(
                BorderSide::try_new(style, 2.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
                BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            )),
            None,
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let error = input
            .normalize(Capabilities::VELLO_0_9)
            .expect_err("unsupported border styles should report typed diagnostics");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::BoxDecorations,
                operation,
            ))
        );
    }
}

#[test]
fn box_decoration_normalization_emits_borders_for_multiple_fragments_in_order() {
    let first = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(2.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(120.0, 0.0, 60.0, 40.0),
        Rect::new(124.0, 4.0, 52.0, 32.0),
        Rect::new(128.0, 8.0, 44.0, 24.0),
    )
    .unwrap();
    let shape = Shape::rect(Rect::new(120.0, 0.0, 60.0, 40.0));
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap()
    .with_border_clip_override(BackgroundClipGeometry::try_shape(shape.clone()).unwrap());
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![first, second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    let commands: Vec<_> = normalized
        .commands()
        .iter()
        .map(normalized_border_command)
        .collect();

    assert_eq!(
        commands
            .iter()
            .map(|command| (command.fragment_index(), command.side()))
            .collect::<Vec<_>>(),
        vec![
            (0, BoxSide::Top),
            (0, BoxSide::Bottom),
            (1, BoxSide::Top),
            (1, BoxSide::Bottom),
        ]
    );
    assert_eq!(commands[2].target_rect(), second_areas.border_box());
    assert_eq!(commands[2].clip().shape(), Some(&shape));
    assert_eq!(commands[2].radii(), second.radii());
    assert_eq!(commands[2].break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_expands_outline_target_by_offset_only() {
    let outline = Outline::try_new(OutlineStyle::Solid, 5.0, Color::BLACK, 3.0).unwrap();
    let fragment = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(6.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input =
        BoxDecorationInput::try_new(None, Some(outline.clone()), vec![fragment.clone()]).unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands().len(), 1);
    let command = normalized_outline_command(&normalized.commands()[0]);
    assert_eq!(command.fragment_index(), 0);
    assert_eq!(command.width(), 5.0);
    assert_eq!(command.offset(), 3.0);
    assert_eq!(command.paint(), outline.paint());
    assert_eq!(command.style(), NormalizedOutlineStyle::Solid);
    assert_eq!(command.target_rect(), Rect::new(-3.0, -3.0, 106.0, 46.0));
    assert_eq!(
        command.clip().rect(),
        Some(box_decoration_test_areas().border_box())
    );
    assert_eq!(command.radii(), fragment.radii());
    assert_eq!(command.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn box_decoration_normalization_keeps_outline_width_out_of_geometry() {
    let thin = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, 2.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();
    let thick = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 12.0, Color::BLACK, 2.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(
        normalized_outline_command(&thin.commands()[0]).target_rect(),
        Rect::new(-2.0, -2.0, 104.0, 44.0)
    );
    assert_eq!(
        normalized_outline_command(&thick.commands()[0]).target_rect(),
        Rect::new(-2.0, -2.0, 104.0, 44.0)
    );
    assert_eq!(
        normalized_outline_command(&thick.commands()[0]).width(),
        12.0
    );
}

#[test]
fn box_decoration_normalization_preserves_dashed_and_dotted_outline_styles() {
    for (style, normalized_style) in [
        (OutlineStyle::Dashed, NormalizedOutlineStyle::Dashed),
        (OutlineStyle::Dotted, NormalizedOutlineStyle::Dotted),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(Outline::try_new(style, 2.0, Color::BLACK, 0.0).unwrap()),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

        assert_eq!(
            normalized_outline_command(&normalized.commands()[0]).style(),
            normalized_style
        );
    }
}

#[test]
fn box_decoration_normalization_reports_unsupported_outline_styles() {
    for (style, operation) in [
        (OutlineStyle::Double, PrimitiveOperation::OutlineDoubleStyle),
        (OutlineStyle::Auto, PrimitiveOperation::OutlineAutoStyle),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(Outline::try_new(style, 2.0, Color::BLACK, 0.0).unwrap()),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let error = input
            .normalize(Capabilities::VELLO_0_9)
            .expect_err("unsupported outline styles should report typed diagnostics");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::BoxDecorations,
                operation,
            ))
        );
    }
}

#[test]
fn box_decoration_normalization_suppresses_none_and_zero_width_outlines() {
    for outline in [
        Outline::try_new(OutlineStyle::None, 2.0, Color::BLACK, 0.0).unwrap(),
        Outline::try_new(OutlineStyle::Solid, 0.0, Color::BLACK, 0.0).unwrap(),
    ] {
        let input = BoxDecorationInput::try_new(
            None,
            Some(outline),
            vec![
                BoxDecorationFragment::try_new(
                    box_decoration_test_areas(),
                    Radii::try_all(0.0).unwrap(),
                    BoxDecorationBreak::Slice,
                )
                .unwrap(),
            ],
        )
        .unwrap();

        let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

        assert!(normalized.commands().is_empty());
    }
}

#[test]
fn box_decoration_normalization_handles_negative_outline_offsets_deterministically() {
    let valid = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 2.0, Color::BLACK, -4.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(
        normalized_outline_command(&valid.commands()[0]).target_rect(),
        Rect::new(4.0, 4.0, 92.0, 32.0)
    );

    let invalid = BoxDecorationInput::try_new(
        None,
        Some(Outline::try_new(OutlineStyle::Solid, 2.0, Color::BLACK, -30.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                box_decoration_test_areas(),
                Radii::try_all(0.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let error = invalid
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("over-contracted outline target rects should be invalid");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("outline target rect")
    );
}

#[test]
fn box_decoration_normalization_emits_outline_after_borders_for_each_fragment() {
    let first = BoxDecorationFragment::try_new(
        box_decoration_test_areas(),
        Radii::try_all(2.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(120.0, 0.0, 60.0, 40.0),
        Rect::new(124.0, 4.0, 52.0, 32.0),
        Rect::new(128.0, 8.0, 44.0, 24.0),
    )
    .unwrap();
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_all(4.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap();
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Solid, 3.0, Color::TRANSPARENT, 1.0).unwrap()),
        vec![first, second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands().len(), 4);
    assert_eq!(
        normalized_border_command(&normalized.commands()[0]).fragment_index(),
        0
    );
    assert_eq!(
        normalized_outline_command(&normalized.commands()[1]).fragment_index(),
        0
    );
    assert_eq!(
        normalized_border_command(&normalized.commands()[2]).fragment_index(),
        1
    );
    let second_outline = normalized_outline_command(&normalized.commands()[3]);
    assert_eq!(second_outline.fragment_index(), 1);
    assert_eq!(
        second_outline.target_rect(),
        Rect::new(119.0, -1.0, 62.0, 42.0)
    );
    assert_eq!(second_outline.radii(), second.radii());
    assert_eq!(second_outline.break_mode(), BoxDecorationBreak::Clone);
}

#[test]
fn background_and_box_decoration_normalization_reuse_border_box_area() {
    let areas = BackgroundAreas::try_new(
        Rect::new(20.0, 30.0, 160.0, 90.0),
        Rect::new(26.0, 36.0, 148.0, 78.0),
        Rect::new(34.0, 44.0, 132.0, 62.0),
    )
    .unwrap();
    let background_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Border),
    );
    let background = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![background_layer]).unwrap(),
        areas,
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();
    let fragment = BoxDecorationFragment::try_new(
        areas,
        Radii::try_new(12.0, 14.0, 16.0, 18.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let decoration = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        None,
        vec![fragment.clone()],
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(background.commands().len(), 1);
    assert_eq!(
        background.commands()[0].clip().rect(),
        Some(areas.border_box())
    );
    let NormalizedBackgroundCommandKind::Layer { layer } = background.commands()[0].kind() else {
        panic!("expected mixed background layer command");
    };
    assert_eq!(layer.placement().paint_rect(), areas.content_box());

    let border = normalized_border_command(&decoration.commands()[0]);
    assert_eq!(border.target_rect(), areas.rect_for(BackgroundBox::Border));
    assert_eq!(border.clip().rect(), Some(areas.border_box()));
    assert_eq!(border.radii().border_box(), areas.border_box());
    assert_eq!(border.radii(), fragment.radii());
}

#[test]
fn background_box_decoration_integration_preserves_command_boundaries_across_fragments() {
    let first_areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 40.0),
        Rect::new(5.0, 5.0, 90.0, 30.0),
        Rect::new(10.0, 10.0, 80.0, 20.0),
    )
    .unwrap();
    let second_areas = BackgroundAreas::try_new(
        Rect::new(110.0, 8.0, 70.0, 54.0),
        Rect::new(116.0, 14.0, 58.0, 42.0),
        Rect::new(122.0, 20.0, 46.0, 30.0),
    )
    .unwrap();
    let first = BoxDecorationFragment::try_new(
        first_areas,
        Radii::try_all(10.0).unwrap(),
        BoxDecorationBreak::Slice,
    )
    .unwrap();
    let second_clip_shape = Shape::rect(Rect::new(111.0, 9.0, 68.0, 52.0));
    let second = BoxDecorationFragment::try_new(
        second_areas,
        Radii::try_new(18.0, 12.0, 10.0, 8.0).unwrap(),
        BoxDecorationBreak::Clone,
    )
    .unwrap()
    .with_border_clip_override(
        BackgroundClipGeometry::try_shape(second_clip_shape.clone()).unwrap(),
    );
    let input = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(1.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            solid_border(3.0, Color::TRANSPARENT),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Dotted, 2.0, Color::BLACK, 1.5).unwrap()),
        vec![first.clone(), second.clone()],
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    let repeated = input.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands(), repeated.commands());
    assert_eq!(normalized.commands().len(), 6);
    assert_eq!(
        normalized
            .commands()
            .iter()
            .map(|command| match command.kind() {
                NormalizedBoxDecorationCommandKind::Border(border) => {
                    (
                        "border",
                        border.fragment_index(),
                        Some(border.side()),
                        border.break_mode(),
                    )
                }
                NormalizedBoxDecorationCommandKind::Outline(outline) => {
                    (
                        "outline",
                        outline.fragment_index(),
                        None,
                        outline.break_mode(),
                    )
                }
            })
            .collect::<Vec<_>>(),
        vec![
            ("border", 0, Some(BoxSide::Top), BoxDecorationBreak::Slice),
            (
                "border",
                0,
                Some(BoxSide::Bottom),
                BoxDecorationBreak::Slice
            ),
            ("outline", 0, None, BoxDecorationBreak::Slice),
            ("border", 1, Some(BoxSide::Top), BoxDecorationBreak::Clone),
            (
                "border",
                1,
                Some(BoxSide::Bottom),
                BoxDecorationBreak::Clone
            ),
            ("outline", 1, None, BoxDecorationBreak::Clone),
        ]
    );

    for command in &normalized.commands()[0..3] {
        match command.kind() {
            NormalizedBoxDecorationCommandKind::Border(border) => {
                assert_eq!(border.target_rect(), first_areas.border_box());
                assert_eq!(border.clip().rect(), Some(first_areas.border_box()));
                assert_eq!(border.radii(), first.radii());
            }
            NormalizedBoxDecorationCommandKind::Outline(outline) => {
                assert_eq!(outline.clip().rect(), Some(first_areas.border_box()));
                assert_eq!(outline.radii(), first.radii());
            }
        }
    }
    for command in &normalized.commands()[3..] {
        match command.kind() {
            NormalizedBoxDecorationCommandKind::Border(border) => {
                assert_eq!(border.target_rect(), second_areas.border_box());
                assert_eq!(border.clip().shape(), Some(&second_clip_shape));
                assert_eq!(border.radii(), second.radii());
            }
            NormalizedBoxDecorationCommandKind::Outline(outline) => {
                assert_eq!(outline.clip().shape(), Some(&second_clip_shape));
                assert_eq!(outline.radii(), second.radii());
            }
        }
    }
}

#[test]
fn background_stacks_reject_empty_and_colorless_inputs() {
    let error = BackgroundStack::try_new(None, Vec::new())
        .expect_err("empty transparent background stacks should use no value");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background stack")
    );
}

#[test]
fn invalid_value_diagnostic_captures_non_finite_constructor_value() {
    let error =
        Point::try_new(f64::NAN, 0.0).expect_err("non-finite point coordinates should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "point x value NaN is invalid: must be finite"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("point x")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("NaN")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite")
    );
}

#[test]
fn invalid_value_diagnostic_captures_impossible_geometry_constructor_value() {
    let error = Rect::try_new(0.0, 0.0, -1.0, 1.0)
        .expect_err("negative rectangle dimensions should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "rectangle width value -1 is invalid: must be finite and non-negative"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("rectangle width")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("-1")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must be finite and non-negative")
    );
}

#[test]
fn invalid_value_constructor_captures_empty_list_invariant() {
    let error = Error::invalid_value("gradient stops", "[]", "must not be empty");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "gradient stops value [] is invalid: must not be empty"
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("gradient stops")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("[]")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must not be empty")
    );
}

#[test]
fn invalid_value_existing_empty_list_constructor_preserves_invalid_input_message() {
    let error = Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(1.0, 1.0).unwrap(),
        vec![],
    )
    .expect_err("empty gradient stop lists should be rejected");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.message(), "gradient stops must not be empty");
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("gradient stops")
    );
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::value),
        Some("[]")
    );
    assert_eq!(
        error
            .invalid_value_diagnostic()
            .map(InvalidValue::invariant),
        Some("must not be empty")
    );
}

#[test]
fn unsupported_primitive_errors_name_operation() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::LayerMask,
    );
    let error = Error::unsupported_render_primitive(unsupported);

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(
        error.message().contains("layer mask"),
        "message should name the unsupported primitive: {}",
        error.message()
    );
}

#[test]
fn unresolved_resource_diagnostics_name_image_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Image, "hero.png");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Image);
    assert_eq!(diagnostic.kind().label(), "image");
    assert_eq!(diagnostic.identifier(), "hero.png");
    assert_eq!(
        error.message(),
        "image resource hero.png could not be resolved"
    );
}

#[test]
fn unresolved_resource_diagnostics_name_mask_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Mask, "#avatar-mask");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Mask);
    assert_eq!(diagnostic.kind().label(), "mask");
    assert_eq!(diagnostic.identifier(), "#avatar-mask");
    assert_eq!(
        error.message(),
        "mask resource #avatar-mask could not be resolved"
    );
}

#[test]
fn unresolved_resource_diagnostics_name_filter_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Filter, "#blur");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Filter);
    assert_eq!(diagnostic.kind().label(), "filter");
    assert_eq!(diagnostic.identifier(), "#blur");
    assert_eq!(
        error.message(),
        "filter resource #blur could not be resolved"
    );
}

#[test]
fn unresolved_resource_diagnostics_name_clip_resources() {
    let diagnostic = UnresolvedResource::new(UnresolvedResourceKind::Clip, "#content-clip");
    let error = Error::unresolved_resource(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::UnresolvedResource);
    assert_eq!(error.unresolved_resource_diagnostic(), Some(&diagnostic));
    assert_eq!(diagnostic.kind(), UnresolvedResourceKind::Clip);
    assert_eq!(diagnostic.kind().label(), "clip");
    assert_eq!(diagnostic.identifier(), "#content-clip");
    assert_eq!(
        error.message(),
        "clip resource #content-clip could not be resolved"
    );
}

#[test]
fn degraded_quality_diagnostics_name_reduced_intermediate_precision() {
    let diagnostic = DegradedQuality::new(
        DegradedQualityKind::ReducedIntermediatePrecision,
        "rgba16float unavailable",
    );
    let error = Error::degraded_quality(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::DegradedQuality);
    assert_eq!(error.degraded_quality_diagnostic(), Some(&diagnostic));
    assert_eq!(
        diagnostic.kind(),
        DegradedQualityKind::ReducedIntermediatePrecision
    );
    assert_eq!(diagnostic.kind().label(), "reduced intermediate precision");
    assert_eq!(diagnostic.value(), "rgba16float unavailable");
    assert_eq!(
        error.message(),
        "render quality degraded: reduced intermediate precision (rgba16float unavailable)"
    );
}

#[test]
fn degraded_quality_diagnostics_name_unsupported_paint_space_conversions() {
    let diagnostic = DegradedQuality::new(
        DegradedQualityKind::UnsupportedPaintSpaceConversion,
        "display-p3 -> srgb",
    );
    let error = Error::degraded_quality(diagnostic.clone());

    assert_eq!(error.code(), ErrorCode::DegradedQuality);
    assert_eq!(error.degraded_quality_diagnostic(), Some(&diagnostic));
    assert_eq!(
        diagnostic.kind(),
        DegradedQualityKind::UnsupportedPaintSpaceConversion
    );
    assert_eq!(
        diagnostic.kind().label(),
        "unsupported paint-space conversion"
    );
    assert_eq!(diagnostic.value(), "display-p3 -> srgb");
    assert_eq!(
        error.message(),
        "render quality degraded: unsupported paint-space conversion (display-p3 -> srgb)"
    );
}

#[test]
fn renderer_reports_backend_capabilities_by_family() {
    let renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let capabilities = renderer.capabilities();

    assert!(capabilities.geometry_targets().supports_rect_fill_stroke());
    assert!(
        capabilities
            .geometry_targets()
            .supports_rounded_rect_fill_stroke()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_circle_ellipse_fill_stroke()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_arbitrary_path_fill()
    );
    assert!(
        capabilities
            .geometry_targets()
            .supports_arbitrary_path_centered_stroke()
    );
    assert!(
        !capabilities
            .geometry_targets()
            .supports_arbitrary_path_inside_outside_stroke()
    );

    assert!(capabilities.paint_sources().supports_solid_rgba());
    assert!(capabilities.paint_sources().supports_gradients());
    assert!(capabilities.paint_sources().supports_image_paint());
    assert!(
        !capabilities
            .paint_sources()
            .supports_non_solid_shadow_paint()
    );
    assert!(
        capabilities
            .shadows()
            .supports_rect_rounded_circle_shadows()
    );
    assert!(!capabilities.shadows().supports_ellipse_path_shadows());

    assert!(!capabilities.filters().supports_layer_filters());
    assert!(capabilities.masks_clips().supports_shape_clips());
    assert!(!capabilities.masks_clips().supports_layer_masks());
    assert!(
        capabilities
            .masks_clips()
            .supports_materialized_alpha_mask_execution()
    );
    assert!(capabilities.compositing().supports_layer_opacity());
    assert!(capabilities.compositing().supports_blend_modes());
    assert!(
        capabilities
            .offscreen_pipeline()
            .supports_direct_vello_opacity_isolation()
    );
    assert!(
        capabilities
            .offscreen_pipeline()
            .supports_direct_vello_blend_isolation()
    );
    assert!(capabilities.surfaces().supports_headless_surfaces());
    assert_eq!(
        capabilities.surfaces().supports_web_canvas_surfaces(),
        cfg!(all(feature = "render-web", target_arch = "wasm32"))
    );
}

#[test]
fn transform_capabilities_name_2d_origin_skew_and_coordinate_tags() {
    let capabilities = Capabilities::VELLO_0_9.transform_coordinate_spaces();

    assert!(capabilities.supports_affine_2d());
    assert!(capabilities.supports_transform_origin());
    assert!(capabilities.supports_skew());
    assert!(capabilities.supports_coordinate_space_tags());
    assert!(!capabilities.supports_transform_3d());
}

#[test]
fn geometry_capabilities_name_boolean_offset_and_hit_test_boundaries() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(!capabilities.geometry_targets().supports_geometry_booleans());
    assert!(!capabilities.geometry_targets().supports_geometry_offsets());
    assert_eq!(
        capabilities.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned
    );
    assert_eq!(HitTestOwnership::RootOwned, HitTestOwnership::RootOwned);
}

#[test]
fn paint_capabilities_name_color_policy_and_conversion_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.paint_sources();

    assert!(capabilities.supports_solid_rgba());
    assert!(capabilities.supports_gradients());
    assert!(capabilities.supports_srgb_color_conversion());
    assert!(capabilities.supports_hsl_color_conversion());
    assert_eq!(
        capabilities.symbolic_color_policy(),
        SymbolicColorPolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_unresolved_symbolic_colors());
    assert!(!capabilities.supports_color_mix());
    assert!(!capabilities.supports_repeating_gradients());
}

#[test]
fn image_sampling_capabilities_name_css_sampling_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.image_sampling();

    assert!(capabilities.supports_image_fit());
    assert!(capabilities.supports_background_position());
    assert!(capabilities.supports_background_size());
    assert!(capabilities.supports_repeat_xy());
    assert_eq!(
        capabilities.attachment_coordinate_policy(),
        BackgroundAttachmentCoordinatePolicy::RootResolvedOrTagged
    );
    assert_eq!(
        capabilities.image_orientation_policy(),
        ImageOrientationPolicy::RootResolvedOnly
    );
    assert_eq!(
        capabilities.image_color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_repeat_round());
    assert!(!capabilities.supports_repeat_space());
    assert!(!capabilities.supports_filtered_image_paint());
    assert!(capabilities.supports_color_filtered_image_paint());
    assert!(!capabilities.supports_image_orientation_conversion());
    assert!(!capabilities.supports_image_color_profile_conversion());
}

#[test]
fn box_decoration_capability_accessors_name_supported_paint_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.box_decorations();

    assert!(capabilities.supports_border_none_hidden_styles());
    assert!(capabilities.supports_border_solid_style());
    assert!(capabilities.supports_border_dashed_dotted_styles());
    assert!(capabilities.supports_border_double_style());
    assert!(capabilities.supports_border_radii());
    assert!(capabilities.supports_outlines());
    assert!(capabilities.supports_outline_none_style());
    assert!(capabilities.supports_outline_solid_style());
    assert!(capabilities.supports_outline_dashed_dotted_styles());
    assert!(capabilities.supports_fragments());
}

#[test]
fn box_decoration_capability_accessors_name_unsupported_style_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.box_decorations();

    assert!(!capabilities.supports_border_groove_style());
    assert!(!capabilities.supports_border_ridge_style());
    assert!(!capabilities.supports_border_inset_style());
    assert!(!capabilities.supports_border_outset_style());
    assert!(!capabilities.supports_outline_double_style());
    assert!(!capabilities.supports_outline_auto_style());
}

#[test]
fn offscreen_pipeline_capability_accessors_name_current_phase_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.offscreen_pipeline();

    assert!(capabilities.supports_direct_vello_opacity_isolation());
    assert!(capabilities.supports_direct_vello_blend_isolation());
    assert!(!capabilities.supports_offscreen_layer_rendering());
    assert!(!capabilities.supports_texture_cache_upload_lifecycle());
    assert!(!capabilities.supports_rect_fullscreen_shader_passes());
    assert!(!capabilities.supports_cpu_reference_buffers());
    assert!(!capabilities.supports_nested_opacity_planning());
    assert!(!capabilities.supports_mask_execution());
    assert!(!capabilities.supports_filter_execution());
    assert!(!capabilities.supports_backdrop_execution());
}

#[test]
fn backdrop_capability_accessors_claim_only_narrow_materialized_execution() {
    let capabilities = Capabilities::VELLO_0_9.offscreen_pipeline();

    assert!(capabilities.supports_bounded_backdrop_capture());
    assert!(capabilities.supports_materialized_backdrop_filter_execution());
    assert!(!capabilities.supports_backdrop_isolation_composition());
    assert!(!capabilities.supports_backdrop_execution());
}

#[test]
fn blend_capability_accessors_preserve_direct_vello_claims_without_background_blend() {
    let compositing = Capabilities::VELLO_0_9.compositing();
    let offscreen = Capabilities::VELLO_0_9.offscreen_pipeline();

    assert!(compositing.supports_layer_opacity());
    assert!(compositing.supports_blend_modes());
    assert!(offscreen.supports_direct_vello_opacity_isolation());
    assert!(offscreen.supports_direct_vello_blend_isolation());
    assert!(!compositing.supports_root_backdrop_policy());
    assert!(!compositing.supports_background_blend_modes());
    assert!(!compositing.supports_additional_mix_blend_modes());
    assert!(!compositing.supports_porter_duff_composite_modes());
}

#[test]
fn mask_clip_capabilities_name_sequence12_boundaries_with_narrow_alpha_execution() {
    let capabilities = Capabilities::VELLO_0_9.masks_clips();

    assert!(capabilities.supports_shape_clips());
    assert!(!capabilities.supports_clip_reference_execution());
    assert!(!capabilities.supports_layer_masks());
    assert!(capabilities.supports_materialized_alpha_mask_execution());
    Capabilities::VELLO_0_9
        .ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::MaterializedAlphaMaskExecution,
        ))
        .expect("materialized alpha-mask execution is supported by the current CPU boundary");
    assert!(!capabilities.supports_luminance_mask_mode());
    assert!(!capabilities.supports_multi_layer_mask_composition());
    assert!(!capabilities.supports_mask_composite_modes());
}

#[test]
fn color_filter_capability_names_granular_execution_without_broad_effects() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(
        capabilities
            .filters()
            .supports_color_filter_classification()
    );
    assert!(
        capabilities
            .filters()
            .supports_color_filter_pipeline_execution()
    );
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(
        capabilities
            .image_sampling()
            .supports_color_filtered_image_paint()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_filter_execution()
    );
}

#[test]
fn pixel_moving_filter_capability_names_advertise_materialized_execution_only() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(
        capabilities
            .filters()
            .supports_materialized_image_filter_classification()
    );
    assert!(
        capabilities
            .filters()
            .supports_materialized_blur_filter_execution()
    );
    assert!(
        capabilities
            .filters()
            .supports_materialized_drop_shadow_filter_execution()
    );
    assert!(
        capabilities
            .filters()
            .supports_filter_region_outset_planning()
    );
    assert!(
        capabilities
            .filters()
            .supports_cpu_reference_blur_fallback()
    );
    assert!(!capabilities.shadows().supports_inset_box_shadows());
    assert!(!capabilities.shadows().supports_text_shadows());
    assert!(!capabilities.filters().supports_layer_filters());
    assert!(
        !capabilities
            .image_sampling()
            .supports_filtered_image_paint()
    );
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_filter_execution()
    );
}

#[test]
fn pixel_moving_filter_and_shadow_diagnostics_have_granular_names() {
    let supported_cases = [
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::MaterializedBlurFilterExecution,
            "materialized blur filter execution",
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::MaterializedDropShadowFilterExecution,
            "materialized drop-shadow filter execution",
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::FilterRegionOutsetPlanning,
            "filter-region/outset planning",
        ),
        (
            PrimitiveFamily::Filters,
            PrimitiveOperation::CpuReferenceBlurFallback,
            "CPU/reference blur fallback",
        ),
    ];
    let unsupported_cases = [
        (
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
            "inset box shadow",
        ),
        (
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
            "text shadow",
        ),
    ];

    for (family, operation, label) in supported_cases {
        let supported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(supported.label(), label);
        Capabilities::VELLO_0_9
            .ensure_supported(supported)
            .expect("Task 4 enables materialized blur execution and its planning pieces");
    }

    for (family, operation, label) in unsupported_cases {
        let unsupported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(unsupported.label(), label);

        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("later sequence diagnostics stay named without execution");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(label));
    }
}

#[test]
fn hit_test_geometry_is_root_owned_not_render_lowered() {
    assert_eq!(
        Capabilities::VELLO_0_9.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned
    );
}

#[test]
fn capabilities_map_unsupported_primitives_to_typed_errors() {
    let capabilities = Capabilities::VELLO_0_9;
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::MasksAndClips,
        PrimitiveOperation::LayerMask,
    );

    let error = capabilities
        .ensure_supported(unsupported)
        .expect_err("layer masks are not supported in this milestone");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message().contains("layer mask"));
}

#[test]
fn unsupported_geometry_operations_report_typed_diagnostics() {
    let boolean = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryBooleanOperation,
    );
    let offset = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryOffsetOperation,
    );

    for unsupported in [boolean, offset] {
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("geometry operation should be explicitly unsupported");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn unsupported_symbolic_color_inputs_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::UnresolvedSymbolicColor,
        PrimitiveOperation::ColorMixFunction,
        PrimitiveOperation::UnsupportedColorSpace,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::PaintSources, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("symbolic or unsupported color input is not render-resolved");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn repeating_gradients_report_typed_diagnostics() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::PaintSources,
        PrimitiveOperation::RepeatingGradient,
    );

    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("repeating gradients require later normalization");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}

#[test]
fn unsupported_image_sampling_operations_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::BackgroundRepeatRound,
        PrimitiveOperation::BackgroundRepeatSpace,
        PrimitiveOperation::FilteredImagePaint,
        PrimitiveOperation::ImageOrientationConversion,
        PrimitiveOperation::ImageColorProfileConversion,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::ImageSampling, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this image sampling primitive");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn unsupported_box_decoration_style_capability_diagnostics_are_typed() {
    for operation in [
        PrimitiveOperation::BorderGrooveStyle,
        PrimitiveOperation::BorderRidgeStyle,
        PrimitiveOperation::BorderInsetStyle,
        PrimitiveOperation::BorderOutsetStyle,
        PrimitiveOperation::OutlineDoubleStyle,
        PrimitiveOperation::OutlineAutoStyle,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::BoxDecorations, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this box-decoration style");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("box decorations"));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn unsupported_3d_transforms_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::Matrix3dTransform,
        PrimitiveOperation::PerspectiveTransform,
        PrimitiveOperation::Rotate3dTransform,
        PrimitiveOperation::TranslateZTransform,
        PrimitiveOperation::ScaleZTransform,
    ] {
        let unsupported =
            UnsupportedPrimitive::new(PrimitiveFamily::TransformsAndCoordinateSpaces, operation);

        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("3D transforms are unsupported in this render phase");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}

#[test]
fn offscreen_pipeline_capability_diagnostics_report_unsupported_operations() {
    for operation in [
        PrimitiveOperation::OffscreenLayerRendering,
        PrimitiveOperation::TextureCacheUploadLifecycle,
        PrimitiveOperation::RectFullscreenShaderPass,
        PrimitiveOperation::CpuReferenceBuffer,
        PrimitiveOperation::NestedOpacityPlanning,
        PrimitiveOperation::MaskExecution,
        PrimitiveOperation::FilterExecution,
        PrimitiveOperation::BackdropExecution,
        PrimitiveOperation::BackdropIsolationComposition,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::OffscreenPipeline, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("offscreen pipeline operation is not implemented in this phase");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("offscreen pipeline"));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn backdrop_and_advanced_compositing_diagnostics_have_granular_names() {
    let cases = [
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::RootBackdropPolicy,
            "root backdrop policy",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::BackgroundBlendMode,
            "background blend mode",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::AdditionalMixBlendMode,
            "additional mix-blend mode",
        ),
        (
            PrimitiveFamily::Compositing,
            PrimitiveOperation::PorterDuffCompositeMode,
            "Porter-Duff composite mode",
        ),
    ];

    for (family, operation, label) in cases {
        let unsupported = UnsupportedPrimitive::new(family, operation);
        assert_eq!(unsupported.label(), label);

        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Sequence 13 Task 1 only names future compositing boundaries");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("compositing"));
        assert!(error.message().contains(label));
    }
}

#[test]
fn mask_clip_capability_diagnostics_report_sequence12_unsupported_operations() {
    for operation in [
        PrimitiveOperation::ClipReferenceExecution,
        PrimitiveOperation::LayerMask,
        PrimitiveOperation::AlphaMaskSourceExecution,
        PrimitiveOperation::LuminanceMaskMode,
        PrimitiveOperation::MultiLayerMaskComposition,
        PrimitiveOperation::MaskCompositeMode,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::MasksAndClips, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Sequence 12 Task 1 should only name unsupported mask/clip boundaries");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("masks and clips"));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[test]
fn vello_baseline_reports_current_unsupported_primitives() {
    let capabilities = Capabilities::VELLO_0_9;
    let cases = [
        UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ),
        UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::BoxDecorations,
            PrimitiveOperation::BorderGrooveStyle,
        ),
        UnsupportedPrimitive::new(
            PrimitiveFamily::BoxDecorations,
            PrimitiveOperation::OutlineAutoStyle,
        ),
    ];

    for unsupported in cases {
        let error = capabilities
            .ensure_supported(unsupported)
            .expect_err("Vello 0.9 should reject this primitive");
        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains(unsupported.label()));
    }
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_unsupported_off_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("web canvas surfaces require render-web on wasm32");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
    assert!(error.message().contains("web canvas surface"));
}

#[cfg(all(feature = "render-web", target_arch = "wasm32"))]
#[test]
fn vello_baseline_reports_web_canvas_surface_as_supported_on_wasm_web() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::Surfaces,
        PrimitiveOperation::WebCanvasSurface,
    );

    Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect("web canvas surfaces are available with render-web on wasm32");
}

#[test]
fn unsupported_layer_masks_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::try_new(4.0, 2.0).unwrap(), 1.0))
            .unwrap();
    let mut scene = Scene::new();

    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert!(error.message().contains("layer mask"));
}

#[test]
fn geometry_try_constructors_reject_invalid_values() {
    assert!(Point::try_new(f64::NAN, 0.0).is_err());
    assert!(Size::try_new(-1.0, 1.0).is_err());
    assert!(Rect::try_new(0.0, 0.0, 1.0, f64::INFINITY).is_err());
    assert!(Radii::try_all(-0.1).is_err());
    assert!(Transform::try_new([1.0, 0.0, 0.0, f64::NAN, 0.0, 0.0]).is_err());
}

#[test]
fn transform_helpers_preserve_affine_coefficients() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 4.0).unwrap();
    let rotate = Transform::rotation(std::f64::consts::FRAC_PI_2).unwrap();

    assert_eq!(translate.as_array(), [1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(scale.as_array(), [2.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
    assert!(rotate.as_array()[0].abs() < 1.0e-12);
    assert!((rotate.as_array()[1] - 1.0).abs() < 1.0e-12);
    assert!((rotate.as_array()[2] + 1.0).abs() < 1.0e-12);
    assert!(rotate.as_array()[3].abs() < 1.0e-12);
}

#[test]
fn transform_skew_helpers_preserve_tangent_coefficients() {
    let skew_x = Transform::skew_x(std::f64::consts::FRAC_PI_4).unwrap();
    let skew_y = Transform::skew_y(std::f64::consts::FRAC_PI_4).unwrap();

    assert!((skew_x.as_array()[2] - 1.0).abs() < 1.0e-12);
    assert!((skew_y.as_array()[1] - 1.0).abs() < 1.0e-12);
}

#[test]
fn transform_helpers_reject_non_finite_inputs() {
    assert!(Transform::translation(f64::NAN, 0.0).is_err());
    assert!(Transform::scale(1.0, f64::INFINITY).is_err());
    assert!(Transform::rotation(f64::NAN).is_err());
    assert!(Transform::skew_x(f64::INFINITY).is_err());
    assert!(Transform::skew_y(f64::NAN).is_err());
}

#[test]
fn transform_then_composes_in_application_order() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 2.0).unwrap();
    let composed = translate.then(scale).unwrap();

    assert_eq!(composed.as_array(), [2.0, 0.0, 0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn transform_around_wraps_transform_origin() {
    let origin = Point::try_new(10.0, 5.0).unwrap();
    let transform = Transform::scale(2.0, 3.0).unwrap().around(origin).unwrap();

    assert_eq!(transform.as_array(), [2.0, 0.0, 0.0, 3.0, -10.0, -10.0]);
}

#[test]
fn coordinate_space_tags_preserve_kind_and_transform() {
    let named = CoordinateSpaceId::try_new(7).unwrap();
    let transform = Transform::translation(3.0, 4.0).unwrap();
    let tag = CoordinateSpaceTag::try_new(CoordinateSpaceKind::Named(named), transform).unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Named(named));
    assert_eq!(tag.transform(), transform);
}

#[test]
fn coordinate_space_ids_reject_reserved_zero() {
    let error = CoordinateSpaceId::try_new(0).expect_err("zero is reserved");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("coordinate space id")
    );
}

#[test]
fn coordinate_space_tags_model_future_backdrop_capture_space() {
    let tag = CoordinateSpaceTag::viewport(Transform::translation(4.0, 6.0).unwrap()).unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Viewport);
    assert_eq!(tag.transform().as_array(), [1.0, 0.0, 0.0, 1.0, 4.0, 6.0]);
}

#[test]
fn rect_try_from_kurbo_rejects_invalid_bounds() {
    let rect = kurbo::Rect {
        x0: 1.0,
        y0: 0.0,
        x1: 0.0,
        y1: 1.0,
    };

    assert!(Rect::try_from(rect).is_err());
}

#[test]
fn physical_size_try_from_logical_size_rejects_invalid_scale() {
    let error = PhysicalSize::try_from_logical(Size::try_new(10.0, 10.0).unwrap(), 0.0)
        .expect_err("scale zero should be rejected before conversion");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn physical_size_try_from_logical_size_rejects_u32_overflow() {
    let error =
        PhysicalSize::try_from_logical(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0)
            .expect_err("physical device pixels should fit in u32");
    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn create_headless_rejects_physical_size_overflow() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error = match pollster::block_on(
        renderer.create_headless(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0),
    ) {
        Ok(_) => panic!("physical device pixels should fit in u32"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::InvalidInput);
}

#[test]
fn draw_value_try_constructors_reject_invalid_values() {
    assert!(Shape::try_circle(Point::try_new(0.0, 0.0).unwrap(), -1.0).is_err());
    assert!(Color::try_rgba(2.0, 0.0, 0.0, 1.0).is_err());
    assert!(Stroke::try_new(0.0).is_err());
    assert!(Dash::try_new(0.0, &[1.0, f64::NAN]).is_err());
    assert!(GradientStop::try_new(1.5, Color::BLACK).is_err());
    assert!(
        Gradient::try_linear(
            Point::try_new(0.0, 0.0).unwrap(),
            Point::try_new(1.0, 1.0).unwrap(),
            vec![],
        )
        .is_err()
    );
    assert!(Layer::new().try_opacity(f32::NAN).is_err());
    assert!(Shadow::try_new(Point::try_new(0.0, 0.0).unwrap(), -1.0, 0.0, Color::BLACK).is_err());
    assert!(TextGlyph::try_new(1, 0.0, f32::NAN, 1.0).is_err());
    assert!(
        TextRun::try_new(
            FontRef::new(1),
            -1.0,
            Transform::identity(),
            TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap(),
            &[],
            TextRunBounds::unspecified(),
        )
        .is_err()
    );
}

#[test]
fn draw_value_constructors_preserve_valid_values() {
    let stroke = Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside);
    let stop = GradientStop::try_new(0.5, Color::BLACK).unwrap();
    let layer = Layer::new().try_opacity(0.5).unwrap();
    let text_paint = TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap();
    let glyph = TextGlyph::try_new(7, 1.0, 2.0, 3.0).unwrap();
    let glyphs = [glyph];
    let text_run = TextRun::try_new(
        FontRef::new(1),
        12.0,
        Transform::identity(),
        text_paint.clone(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();

    assert_eq!(stroke.width(), 2.0);
    assert_eq!(stop.offset(), 0.5);
    assert_eq!(layer.opacity(), 0.5);
    assert_eq!(text_paint.fill(), &Paint::color(Color::BLACK));
    assert_eq!(glyph.id(), 7);
    assert_eq!(text_run.size(), 12.0);
}

#[test]
fn image_ids_are_typed_resource_handles() {
    let image = Image::from_rgba(
        Size::try_new(1.0, 1.0).unwrap(),
        Arc::<[u8]>::from([0, 0, 0, 255]),
    )
    .unwrap();
    let id = image.id();

    assert_eq!(id.get(), image.id().get());
}

#[test]
fn font_refs_use_typed_font_ids() {
    let font = FontRef::new(FontId::new(42));

    assert_eq!(font.id(), FontId::new(42));
}

#[test]
fn text_shadow_run_model_preserves_text_run_and_shadow_order() {
    let glyph = TextGlyph::try_new(7, 1.0, 2.0, 3.0).unwrap();
    let glyphs = [glyph];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        12.0,
        Transform::identity(),
        TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(0.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();

    let text_shadow = TextShadowRun::try_new(run.clone(), shadows).unwrap();

    assert_eq!(text_shadow.run(), &run);
    assert_eq!(text_shadow.shadows().len(), 2);
    assert_eq!(text_shadow.shadows().shadows()[0], first);
    assert_eq!(text_shadow.shadows().shadows()[1], second);
}

#[test]
fn zero_blur_multi_text_shadow_preserves_authored_order_but_rejects_execution() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem ordered zero blur text shadows"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(
        Point::new(-2.0, 3.0),
        0.0,
        0.0,
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let text_shadow = TextShadowRun::try_new(run, shadows).unwrap();

    assert_eq!(
        text_shadow.shadows().shadows(),
        &[first.clone(), second.clone()]
    );

    let mut scene = Scene::new();
    scene.text_shadow_run(text_shadow);

    match &scene.commands[0] {
        scene::Command::TextShadowRun { shadows, .. } => {
            assert_eq!(shadows.shadows(), &[first, second]);
        }
        command => panic!("expected stored TextShadowRun, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("zero-blur text-shadow candidates must not emit render commands yet");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("zero-blur solid text shadows"));
    assert!(error.message().contains("not claimed or enabled"));
}

#[test]
fn transformed_text_shadow_inputs_are_stored_but_not_claimed_as_shifted_glyph_execution() {
    let text_transform = Transform::translation(4.0, 5.0)
        .unwrap()
        .then(Transform::skew_x(0.25).unwrap())
        .unwrap();
    let layer_transform = Transform::translation(10.0, -3.0).unwrap();
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 2.0, 10.0, 10.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem transformed text shadow"),
        16.0,
        text_transform,
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(2.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();

    scene.transform(layer_transform, |scene| {
        scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());
    });

    match &scene.commands[0] {
        scene::Command::Layer { layer, children } => {
            assert_eq!(layer.transform(), layer_transform);
            match &children[0] {
                scene::Command::TextShadowRun {
                    transform, glyphs, ..
                } => {
                    assert_eq!(*transform, text_transform);
                    assert_eq!(glyphs[0].id(), AHEM_GLYPH_X);
                }
                command => panic!("expected stored transformed TextShadowRun, got {command:?}"),
            }
        }
        command => panic!("expected transformed layer, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("transform-aware shifted glyph text-shadow execution is not implemented");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("repeated shifted glyph draws"));
    assert!(error.message().contains("not claimed or enabled"));
}

#[test]
fn non_solid_or_spread_text_shadow_stays_on_glyph_alpha_offscreen_diagnostic_path() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(8.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap()).unwrap(),
        ],
    )
    .unwrap();
    let cases = [
        (
            "gradient text shadow paint",
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Paint::gradient(gradient)).unwrap(),
        ),
        (
            "spread text shadow",
            Shadow::try_new(Point::new(1.0, 1.0), 0.0, 2.0, Color::BLACK).unwrap(),
        ),
        (
            "blurred text shadow",
            Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        ),
    ];

    for (label, shadow) in cases {
        let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap()];
        let run = TextRun::try_new(
            ahem_font(label),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.text_shadow_run(
            TextShadowRun::try_new(run, ShadowList::try_new(vec![shadow]).unwrap()).unwrap(),
        );

        let error = match scene.normalize(Capabilities::VELLO_0_9) {
            Ok(_) => panic!("{label} should stay unsupported"),
            Err(error) => error,
        };
        assert!(
            error
                .message()
                .contains("glyph-alpha/offscreen text capture"),
            "{label} used the wrong text-shadow diagnostic: {}",
            error.message()
        );
        assert!(
            !error.message().contains("zero-blur solid text shadows"),
            "{label} should not be classified as the shifted-glyph candidate path"
        );
    }
}

#[test]
fn text_shadow_run_reports_typed_unsupported_diagnostic() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem zero blur text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("text-shadow execution is not implemented in this phase");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("text shadow"));
    assert!(error.message().contains("zero-blur solid"));
    assert!(error.message().contains("repeated shifted glyph draws"));
}

#[test]
fn text_shadow_capability_claim_matches_current_diagnostic_boundary() {
    let unsupported =
        UnsupportedPrimitive::new(PrimitiveFamily::Shadows, PrimitiveOperation::TextShadow);
    assert!(!Capabilities::VELLO_0_9.shadows().supports_text_shadows());

    let capability_error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("text-shadow capability should stay false until execution exists");
    assert_eq!(capability_error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(capability_error.unsupported_primitive(), Some(unsupported));

    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let normalize_error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("normalization should report the same unsupported text-shadow boundary");
    assert_eq!(normalize_error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(normalize_error.unsupported_primitive(), Some(unsupported));
    assert_eq!(
        normalize_error.unsupported_primitive(),
        capability_error.unsupported_primitive()
    );
    assert!(normalize_error.message().contains("zero-blur solid"));
    assert!(
        normalize_error
            .message()
            .contains("repeated shifted glyph draws")
    );
}

#[test]
fn blurred_text_shadow_reports_same_typed_boundary() {
    let glyphs = [TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        ahem_font("Ahem blurred text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 4.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut scene = Scene::new();
    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("blurred text-shadow needs glyph-alpha capture before pixel-moving blur");

    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("text shadow"));
    assert!(
        error
            .message()
            .contains("glyph-alpha/offscreen text capture")
    );
}

#[test]
fn text_shadow_run_command_storage_preserves_shadow_order_font_data_and_glyphs() {
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_X, 0.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 12.0, 10.0, 10.0).unwrap(),
    ];
    let run = TextRun::try_new(
        ahem_font("Ahem stored text shadow"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let first = Shadow::try_new(Point::new(3.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(0.0, 4.0), 2.0, 0.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let mut scene = Scene::new();

    scene.text_shadow_run(TextShadowRun::try_new(run, shadows).unwrap());

    assert_eq!(scene.commands.len(), 1);
    match &scene.commands[0] {
        scene::Command::TextShadowRun {
            font,
            glyphs,
            shadows,
            ..
        } => {
            assert_eq!(font.id(), FontId::new(AHEM_FONT_ID));
            assert_eq!(font.name.as_deref(), Some("Ahem stored text shadow"));
            assert!(font.data.is_some());
            assert_eq!(glyphs.len(), 2);
            assert_eq!(glyphs[0].id(), AHEM_GLYPH_X);
            assert_eq!(glyphs[1].id(), AHEM_GLYPH_DESCENT_P);
            assert_eq!(shadows.shadows(), &[first, second]);
        }
        command => panic!("expected stored TextShadowRun, got {command:?}"),
    }

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("stored text-shadow ordering should be rejected only at normalization");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
}

#[test]
fn ordinary_text_run_normalization_remains_unaffected_by_text_shadow_boundary() {
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let run = TextRun::try_new(
        FontRef::new(1).named("Test"),
        16.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);

    let normalized = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect("ordinary text runs should not use the text-shadow diagnostic");

    assert_eq!(normalized.commands.len(), 1);
    assert_eq!(normalized.stats().glyphs, 1);
    assert_eq!(normalized.stats().shadows, 0);
    assert!(matches!(
        normalized.commands[0],
        command::RenderCommand::TextRun { .. }
    ));
}

#[test]
fn text_fill_paint_matches_concrete_render_paint_surface() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(4.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 0, 0, 255])).unwrap();
    let cases = [
        ("solid color", Paint::color(Color::BLACK)),
        ("gradient", Paint::gradient(gradient)),
        ("image", Paint::image(image)),
    ];

    for (label, paint) in cases {
        let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
        let run = TextRun::try_new(
            FontRef::new(1).named(label),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(paint.clone()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.text_run(run);

        let brush = glyph_paint_brush(&paint)
            .unwrap_or_else(|_| panic!("{label} should encode as a glyph brush"));
        match (&paint, brush) {
            (paint, peniko::Brush::Solid(_)) if paint == &Paint::color(Color::BLACK) => {}
            (paint, peniko::Brush::Gradient(_))
                if matches!(paint.kind(), paint::PaintKind::Gradient(_)) => {}
            (paint, peniko::Brush::Image(_))
                if matches!(paint.kind(), paint::PaintKind::Image(_)) => {}
            _ => panic!("{label} encoded to the wrong glyph brush kind"),
        }

        let normalized = scene
            .normalize(Capabilities::VELLO_0_9)
            .unwrap_or_else(|_| panic!("{label} text fill should normalize"));

        match &normalized.commands[0] {
            command::RenderCommand::TextRun {
                paint: text_paint,
                glyphs,
                ..
            } => {
                assert_eq!(text_paint.fill(), &paint);
                assert_eq!(glyphs.len(), 1);
            }
            command => panic!("{label} should normalize to a text run, got {command:?}"),
        }
    }
}

#[test]
fn ahem_text_run_preserves_font_data_and_stable_glyph_stream() {
    assert_eq!(AHEM_GLYPH_X, 58);
    assert_eq!(AHEM_GLYPH_DESCENT_P, 82);
    assert_eq!(AHEM_GLYPH_ASCENT_E_ACUTE, 100);

    let expected_glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_X, 2.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 14.0, 10.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 26.0, 10.0, 10.0).unwrap(),
    ];
    let run = TextRun::try_new(
        ahem_font("Ahem stable glyph stream"),
        10.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &expected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(run);

    let normalized = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect("Ahem text run with prepared glyphs should normalize");

    let [
        command::RenderCommand::TextRun {
            font,
            glyphs: encoded_glyphs,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("Ahem text should normalize as one text run");
    };
    assert_eq!(font.id(), FontId::new(AHEM_FONT_ID));
    assert!(font.data.is_some());
    assert_eq!(encoded_glyphs, &expected_glyphs);
}

#[test]
fn ahem_font_data_renders_ascent_and_descent_glyph_bands() {
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 1.0, 9.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 13.0, 9.0, 10.0).unwrap(),
    ];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            ahem_font("Ahem ascent and descent bands"),
            10.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let Some(output) = render_scene_to_headless_or_skip_no_adapter(&scene, Size::new(25.0, 12.0))
    else {
        return;
    };

    assert!(
        pixel_alpha(&output, 6, 5) > 200,
        "E-acute gid 100 should paint the ascent band"
    );
    assert_eq!(
        pixel_alpha(&output, 6, 10),
        0,
        "E-acute gid 100 should not paint the descent band"
    );
    assert!(
        pixel_alpha(&output, 18, 10) > 200,
        "p gid 82 should paint the descent band"
    );
    assert_eq!(
        pixel_alpha(&output, 18, 5),
        0,
        "p gid 82 should not paint the ascent band"
    );
}

#[test]
fn text_decoration_line_preserves_paint_thickness_transform_and_text_order() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 12.0),
        Point::new(32.0, 12.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let decoration = TextDecorationLine::try_solid(
        Point::new(2.0, 12.0),
        Point::new(34.0, 12.0),
        2.5,
        Transform::translation(3.0, 4.0).unwrap(),
        Paint::gradient(gradient.clone()),
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(1, 4.0, 10.0, 8.0).unwrap()];
    let text = TextRun::try_new(
        FontRef::new(1).named("Decoration order"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_decoration_line(decoration).text_run(text);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands.len(), 2);
    assert!(matches!(
        normalized.commands[1],
        command::RenderCommand::TextRun { .. }
    ));
    let command::RenderCommand::Layer { layer, children } = &normalized.commands[0] else {
        panic!("transformed decoration should lower through a layer");
    };
    assert_eq!(layer.transform, Transform::translation(3.0, 4.0).unwrap());
    let [
        command::RenderCommand::Stroke {
            shape,
            stroke,
            paint,
        },
    ] = children.as_slice()
    else {
        panic!("decoration layer should contain one stroke command");
    };
    assert_eq!(stroke.width, 2.5);
    assert!(matches!(shape, command::RenderStrokeShape::Path(_)));
    assert_eq!(paint, &command::RenderPaint::Gradient(gradient));
}

#[test]
fn text_decoration_line_supports_solid_color_without_extra_text_semantics() {
    let decoration = TextDecorationLine::try_new(
        Point::new(1.0, 5.0),
        Point::new(9.0, 5.0),
        1.0,
        Transform::identity(),
        Color::BLACK.into(),
        TextDecorationLineStyle::Solid,
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_decoration_line(decoration);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    let [command::RenderCommand::Stroke { stroke, paint, .. }] = normalized.commands.as_slice()
    else {
        panic!("identity decoration should lower to a plain stroke");
    };
    assert_eq!(stroke.width, 1.0);
    assert_eq!(paint, &command::RenderPaint::Color(Color::BLACK));
}

#[test]
fn non_solid_text_decoration_styles_report_typed_boundary() {
    for style in [
        TextDecorationLineStyle::Double,
        TextDecorationLineStyle::Dotted,
        TextDecorationLineStyle::Dashed,
        TextDecorationLineStyle::Wavy,
    ] {
        let error = TextDecorationLine::try_new(
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            1.0,
            Transform::identity(),
            Color::BLACK.into(),
            style,
        )
        .expect_err("non-solid decoration styles require root/text expansion");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::TextDecorations,
                PrimitiveOperation::TextDecorationStyle,
            ))
        );
        assert!(error.message().contains("text decoration style"));
        assert!(error.message().contains("root/text"));
    }
}

#[test]
fn selection_and_generated_text_buckets_use_plain_render_capabilities() {
    let capabilities = Capabilities::VELLO_0_9;
    assert!(capabilities.geometry_targets().supports_rect_fill_stroke());
    assert!(capabilities.paint_sources().supports_solid_rgba());
    assert!(
        !capabilities.shadows().supports_text_shadows(),
        "materialized selection/generated text buckets must not depend on text-shadow execution"
    );

    let selected_glyphs = [TextGlyph::try_new(10, 2.0, 10.0, 6.0).unwrap()];
    let generated_glyphs = [TextGlyph::try_new(11, 14.0, 10.0, 5.0).unwrap()];
    let selected_run = TextRun::try_new(
        FontRef::new(1).named("Selection"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap().into()).unwrap(),
        &selected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let generated_run = TextRun::try_new(
        FontRef::new(2).named("Generated"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &generated_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();

    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 12.0, 16.0), Color::BLACK)
        .text_run(selected_run)
        .text_run(generated_run);

    let normalized = scene
        .normalize(capabilities)
        .expect("materialized selection/generated content should normalize as ordinary commands");

    assert_eq!(normalized.commands.len(), 3);
    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().glyphs, 2);
    assert_eq!(normalized.stats().shadows, 0);
    assert!(matches!(
        normalized.commands.as_slice(),
        [
            command::RenderCommand::Fill { .. },
            command::RenderCommand::TextRun { .. },
            command::RenderCommand::TextRun { .. },
        ]
    ));
}

#[test]
fn materialized_selection_background_and_text_foreground_stay_ordered_commands() {
    let selected_glyphs = [
        TextGlyph::try_new(21, 4.0, 14.0, 7.0).unwrap(),
        TextGlyph::try_new(22, 11.0, 14.0, 6.0).unwrap(),
    ];
    let selected_text_paint =
        TextPaint::try_fill(Color::try_rgba(0.9, 0.96, 1.0, 1.0).unwrap().into()).unwrap();
    let selected_run = TextRun::try_new(
        FontRef::new(21).named("Root materialized selection text"),
        16.0,
        Transform::identity(),
        selected_text_paint.clone(),
        &selected_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let selection_background = Rect::new(2.0, 2.0, 18.0, 18.0);
    let selection_background_paint = Color::try_rgba(0.0, 0.26, 0.72, 1.0).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(selection_background, selection_background_paint)
        .text_run(selected_run);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.commands.len(), 2);
    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().glyphs, 2);
    let [
        command::RenderCommand::Fill { shape, paint },
        command::RenderCommand::TextRun {
            font,
            paint: text_paint,
            glyphs,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("selection bucket should remain a fill followed by selected glyphs");
    };
    assert_eq!(
        shape,
        &command::RenderShape::Rect(selection_background),
        "selection highlight geometry is ordinary fill geometry"
    );
    assert_eq!(
        paint,
        &command::RenderPaint::Color(selection_background_paint),
        "selection highlight paint is ordinary fill paint"
    );
    assert_eq!(font.id(), FontId::new(21));
    assert_eq!(text_paint, &selected_text_paint);
    assert_eq!(glyphs, &selected_glyphs);
}

#[test]
fn materialized_generated_text_content_preserves_render_command_order() {
    let before_glyphs = [TextGlyph::try_new(31, 0.0, 12.0, 5.0).unwrap()];
    let principal_glyphs = [TextGlyph::try_new(32, 6.0, 12.0, 8.0).unwrap()];
    let after_glyphs = [TextGlyph::try_new(33, 15.0, 12.0, 5.0).unwrap()];
    let before = TextRun::try_new(
        FontRef::new(31).named("Generated before"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &before_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let principal = TextRun::try_new(
        FontRef::new(32).named("Principal text"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::try_rgba(0.1, 0.1, 0.1, 1.0).unwrap().into()).unwrap(),
        &principal_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let after = TextRun::try_new(
        FontRef::new(33).named("Generated after"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &after_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.text_run(before).text_run(principal).text_run(after);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.stats().glyphs, 3);
    let [
        command::RenderCommand::TextRun {
            font: before_font, ..
        },
        command::RenderCommand::TextRun {
            font: principal_font,
            ..
        },
        command::RenderCommand::TextRun {
            font: after_font, ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("generated and principal text should all normalize as text runs");
    };
    assert_eq!(before_font.id(), FontId::new(31));
    assert_eq!(principal_font.id(), FontId::new(32));
    assert_eq!(after_font.id(), FontId::new(33));
}

#[test]
fn materialized_generated_image_marker_and_text_content_are_ordinary_image_text_commands() {
    let marker_image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]),
    )
    .unwrap();
    let marker_id = marker_image.id();
    let marker_rect = Rect::new(0.0, 3.0, 4.0, 4.0);
    let item_glyphs = [TextGlyph::try_new(41, 8.0, 14.0, 9.0).unwrap()];
    let item_text = TextRun::try_new(
        FontRef::new(41).named("Generated list item text"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &item_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene
        .image(marker_image, marker_rect, ImageFit::Contain)
        .text_run(item_text);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().glyphs, 1);
    assert_eq!(normalized.stats().cache_misses, 1);
    let [
        command::RenderCommand::Image { image, rect, fit },
        command::RenderCommand::TextRun { font, glyphs, .. },
    ] = normalized.commands.as_slice()
    else {
        panic!("generated marker image and text should remain ordinary image/text commands");
    };
    assert_eq!(image.id(), marker_id);
    assert_eq!(*rect, marker_rect);
    assert_eq!(*fit, ImageFit::Contain);
    assert_eq!(font.id(), FontId::new(41));
    assert_eq!(glyphs, &item_glyphs);
}

#[test]
fn sequence14_matrix_rows_normalize_or_report_typed_diagnostics() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(12.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let glyph_fill_glyphs = [TextGlyph::try_new(51, 2.0, 12.0, 7.0).unwrap()];
    let glyph_fill = TextRun::try_new(
        FontRef::new(51).named("Sequence 14 glyph fill"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Paint::gradient(gradient.clone())).unwrap(),
        &glyph_fill_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let decoration = TextDecorationLine::try_solid(
        Point::new(2.0, 16.0),
        Point::new(22.0, 16.0),
        1.5,
        Transform::identity(),
        Paint::color(Color::BLACK),
    )
    .unwrap();
    let generated_image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let generated_glyphs = [TextGlyph::try_new(52, 28.0, 12.0, 6.0).unwrap()];
    let generated = TextRun::try_new(
        FontRef::new(52).named("Sequence 14 generated content"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &generated_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            Rect::new(0.0, 2.0, 26.0, 18.0),
            Color::try_rgba(0.7, 0.82, 1.0, 1.0).unwrap(),
        )
        .text_run(glyph_fill)
        .text_decoration_line(decoration)
        .image(
            generated_image,
            Rect::new(24.0, 4.0, 28.0, 8.0),
            ImageFit::Contain,
        )
        .text_run(generated);

    let normalized = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect("implemented Sequence 14 rows should normalize as ordinary render commands");

    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().strokes, 1);
    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().glyphs, 2);
    let [
        command::RenderCommand::Fill { .. },
        command::RenderCommand::TextRun {
            paint: glyph_paint,
            glyphs: normalized_glyphs,
            ..
        },
        command::RenderCommand::Stroke { stroke, paint, .. },
        command::RenderCommand::Image { fit, .. },
        command::RenderCommand::TextRun {
            font: generated_font,
            ..
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("Sequence 14 implemented rows should keep fill/text/stroke/image/text order");
    };
    assert_eq!(glyph_paint.fill(), &Paint::gradient(gradient));
    assert_eq!(normalized_glyphs, &glyph_fill_glyphs);
    assert_eq!(stroke.width, 1.5);
    assert_eq!(paint, &command::RenderPaint::Color(Color::BLACK));
    assert_eq!(*fit, ImageFit::Contain);
    assert_eq!(generated_font.id(), FontId::new(52));

    let shadow_glyphs = [TextGlyph::try_new(53, 0.0, 12.0, 7.0).unwrap()];
    let shadow_run = TextRun::try_new(
        FontRef::new(53).named("Sequence 14 text shadow"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &shadow_glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(1.0, 1.0), 0.0, 0.0, Color::BLACK).unwrap(),
    ])
    .unwrap();
    let mut shadow_scene = Scene::new();
    shadow_scene.text_shadow_run(TextShadowRun::try_new(shadow_run, shadows).unwrap());

    let error = shadow_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("Sequence 14 text-shadow execution should stay explicitly diagnostic");
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::TextShadow,
        ))
    );
    assert!(error.message().contains("zero-blur solid text shadows"));
}

#[test]
fn sequence14_capabilities_advertise_only_render_owned_text_behavior() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(capabilities.paint_sources().supports_solid_rgba());
    assert!(capabilities.paint_sources().supports_gradients());
    assert!(capabilities.paint_sources().supports_image_paint());
    assert!(capabilities.geometry_targets().supports_rect_fill_stroke());
    assert!(
        capabilities
            .geometry_targets()
            .supports_arbitrary_path_centered_stroke()
    );
    assert!(capabilities.image_sampling().supports_image_fit());

    assert_eq!(
        capabilities.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned,
        "selection ownership must stay outside render; render only accepts materialized fills/text"
    );
    assert_eq!(
        capabilities.paint_sources().symbolic_color_policy(),
        SymbolicColorPolicy::RootResolvedOnly,
        "generated/currentColor style resolution must stay outside render"
    );
    assert!(
        !capabilities
            .paint_sources()
            .supports_unresolved_symbolic_colors()
    );
    assert!(!capabilities.paint_sources().supports_color_mix());
    assert!(!capabilities.shadows().supports_text_shadows());
    assert!(
        !capabilities
            .offscreen_pipeline()
            .supports_filter_execution()
    );
    assert!(!capabilities.offscreen_pipeline().supports_mask_execution());

    let text_shadow =
        UnsupportedPrimitive::new(PrimitiveFamily::Shadows, PrimitiveOperation::TextShadow);
    let error = capabilities
        .ensure_supported(text_shadow)
        .expect_err("capabilities should not claim text-shadow execution");
    assert_eq!(error.unsupported_primitive(), Some(text_shadow));
}

#[test]
fn sequence14_text_shadow_candidates_stay_on_diagnostic_boundary() {
    assert!(!Capabilities::VELLO_0_9.shadows().supports_text_shadows());

    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(8.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let cases = [
        (
            "zero blur",
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Color::BLACK).unwrap(),
            "zero-blur solid text shadows",
        ),
        (
            "blurred glyph alpha",
            Shadow::try_new(Point::new(1.0, 0.0), 3.0, 0.0, Color::BLACK).unwrap(),
            "glyph-alpha/offscreen text capture",
        ),
        (
            "non-solid glyph alpha",
            Shadow::try_new(Point::new(1.0, 0.0), 0.0, 0.0, Paint::gradient(gradient)).unwrap(),
            "glyph-alpha/offscreen text capture",
        ),
    ];

    for (label, shadow, expected_message) in cases {
        let glyphs = [TextGlyph::try_new(61, 0.0, 12.0, 7.0).unwrap()];
        let run = TextRun::try_new(
            FontRef::new(61).named(label),
            14.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.text_shadow_run(
            TextShadowRun::try_new(run, ShadowList::try_new(vec![shadow]).unwrap()).unwrap(),
        );

        let error = match scene.normalize(Capabilities::VELLO_0_9) {
            Ok(_) => panic!("{label} text-shadow should stay unsupported"),
            Err(error) => error,
        };
        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Shadows,
                PrimitiveOperation::TextShadow,
            ))
        );
        assert!(
            error.message().contains(expected_message),
            "{label} text-shadow used the wrong diagnostic: {}",
            error.message()
        );
    }
}

#[test]
fn matrix_full_background_box_image_text_stack_preserves_render_order() {
    let areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 64.0, 32.0),
        Rect::new(4.0, 4.0, 56.0, 24.0),
        Rect::new(8.0, 8.0, 48.0, 16.0),
    )
    .unwrap();
    let background_image =
        Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([255; 16])).unwrap();
    let background_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(background_image.clone()).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_length(12.0).unwrap(),
                SizeComponent::try_length(8.0).unwrap(),
            ))
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let background = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(
            Some(Color::try_rgba(0.1, 0.2, 0.3, 1.0).unwrap()),
            vec![background_layer],
        )
        .unwrap(),
        areas,
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();
    let decoration = BoxDecorationInput::try_new(
        Some(box_decoration_edges(
            solid_border(2.0, Color::BLACK),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
            BorderSide::try_new(BorderStyle::None, 0.0, Color::BLACK).unwrap(),
        )),
        Some(Outline::try_new(OutlineStyle::Solid, 1.0, Color::TRANSPARENT, 1.0).unwrap()),
        vec![
            BoxDecorationFragment::try_new(
                areas,
                Radii::try_all(3.0).unwrap(),
                BoxDecorationBreak::Slice,
            )
            .unwrap(),
        ],
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();
    let decoration_line = TextDecorationLine::try_solid(
        Point::new(10.0, 24.0),
        Point::new(42.0, 24.0),
        1.5,
        Transform::identity(),
        Paint::color(Color::BLACK),
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(71, 12.0, 22.0, 9.0).unwrap()];
    let text = TextRun::try_new(
        FontRef::new(71).named("Matrix paint stack"),
        14.0,
        Transform::identity(),
        TextPaint::try_fill(Color::BLACK.into()).unwrap(),
        &glyphs,
        TextRunBounds::unspecified(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene
        .fill(
            areas.border_box(),
            Color::try_rgba(0.1, 0.2, 0.3, 1.0).unwrap(),
        )
        .image(
            background_image,
            Rect::new(8.0, 8.0, 12.0, 8.0),
            ImageFit::Stretch,
        )
        .stroke(
            areas.border_box(),
            Stroke::try_new(2.0).unwrap(),
            Color::BLACK,
        )
        .text_decoration_line(decoration_line)
        .text_run(text);

    assert_eq!(background.commands().len(), 2);
    assert!(matches!(
        background.commands()[0].kind(),
        NormalizedBackgroundCommandKind::ColorFill { .. }
    ));
    let NormalizedBackgroundCommandKind::Layer { layer } = background.commands()[1].kind() else {
        panic!("expected normalized image layer after background color");
    };
    assert_eq!(
        background.commands()[1].clip().rect(),
        Some(areas.padding_box())
    );
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::Image(_)
    ));
    assert_eq!(layer.placement().paint_rect(), areas.content_box());
    assert_eq!(
        decoration
            .commands()
            .iter()
            .map(|command| match command.kind() {
                NormalizedBoxDecorationCommandKind::Border(_) => "border",
                NormalizedBoxDecorationCommandKind::Outline(_) => "outline",
            })
            .collect::<Vec<_>>(),
        ["border", "outline"]
    );

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.stats().fills, 1);
    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().strokes, 2);
    assert_eq!(normalized.stats().glyphs, 1);
    let [
        command::RenderCommand::Fill { .. },
        command::RenderCommand::Image { .. },
        command::RenderCommand::Stroke {
            shape: border_shape,
            stroke: border_stroke,
            paint: border_paint,
        },
        command::RenderCommand::Stroke {
            shape: decoration_shape,
            stroke: decoration_stroke,
            paint: decoration_paint,
        },
        command::RenderCommand::TextRun { .. },
    ] = normalized.commands.as_slice()
    else {
        panic!("expected fill, image, border stroke, decoration stroke, and text run in order");
    };
    assert_eq!(
        border_shape,
        &command::RenderStrokeShape::Rect(kurbo::Rect::from(areas.border_box()))
    );
    assert_eq!(border_stroke.width, 2.0);
    assert_eq!(border_paint, &command::RenderPaint::Color(Color::BLACK));
    let command::RenderStrokeShape::Path(decoration_path) = decoration_shape else {
        panic!("expected text decoration to lower to a path stroke");
    };
    assert_eq!(decoration_path.elements().len(), 2);
    assert_eq!(
        decoration_path.elements()[0],
        kurbo::PathEl::MoveTo(kurbo::Point::new(10.0, 24.0))
    );
    assert_eq!(
        decoration_path.elements()[1],
        kurbo::PathEl::LineTo(kurbo::Point::new(42.0, 24.0))
    );
    assert_eq!(decoration_stroke.width, 1.5);
    assert_eq!(decoration_paint, &command::RenderPaint::Color(Color::BLACK));
}

#[test]
fn matrix_full_transform_clip_opacity_image_gradient_stack_plans_layers() {
    let image = Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([255; 16])).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let outer_transform = Transform::translation(3.0, 4.0).unwrap();
    let clip_shape = Shape::rect(Rect::new(2.0, 2.0, 18.0, 14.0));
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_transform(outer_transform).unwrap(),
        |scene| {
            scene.layer(Layer::new().try_clip(clip_shape).unwrap(), |scene| {
                scene.layer(Layer::new().try_opacity(0.625).unwrap(), |scene| {
                    scene.image(image, Rect::new(4.0, 5.0, 8.0, 6.0), ImageFit::Contain);
                    scene.fill(
                        Rect::new(6.0, 7.0, 10.0, 3.0),
                        Paint::gradient(gradient.clone()),
                    );
                });
            });
        },
    );

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();

    assert_eq!(normalized.stats().layers, 3);
    assert_eq!(normalized.stats().images, 1);
    assert_eq!(normalized.stats().fills, 1);
    let [
        command::RenderCommand::Layer {
            layer: transform_layer,
            children: transform_children,
        },
    ] = normalized.commands.as_slice()
    else {
        panic!("expected transform layer at the root");
    };
    assert_eq!(transform_layer.transform, outer_transform);
    assert_eq!(transform_layer.isolation, command::LayerIsolation::None);
    let [
        command::RenderCommand::Layer {
            layer: clip_layer,
            children: clip_children,
        },
    ] = transform_children.as_slice()
    else {
        panic!("expected clip layer inside transform layer");
    };
    assert_eq!(clip_layer.isolation, command::LayerIsolation::ClipOnly);
    assert_eq!(
        clip_layer.pass_plan.kind(),
        command::LayerPassKind::ClipOnly
    );
    assert_eq!(
        clip_layer
            .pass_plan
            .bounds()
            .map(command::OffscreenBounds::rect),
        Some(Rect::new(2.0, 2.0, 18.0, 14.0))
    );
    let [
        command::RenderCommand::Layer {
            layer: opacity_layer,
            children: opacity_children,
        },
    ] = clip_children.as_slice()
    else {
        panic!("expected opacity layer inside clip layer");
    };
    assert_eq!(opacity_layer.opacity, 0.625);
    assert_eq!(
        opacity_layer.isolation,
        command::LayerIsolation::BackendLayer
    );
    assert_eq!(
        opacity_layer.pass_plan.requirement(),
        command::LayerPassRequirement::DirectVelloOpacity
    );
    assert!(matches!(
        opacity_children.as_slice(),
        [
            command::RenderCommand::Image { .. },
            command::RenderCommand::Fill {
                paint: command::RenderPaint::Gradient(_),
                ..
            },
        ]
    ));
}

#[test]
fn matrix_full_effect_stack_diagnostics_stop_at_unsupported_boundaries() {
    let filter_layer = Layer::new()
        .try_filter(Filter::try_blur(4.0).unwrap())
        .unwrap();
    let mut filter_scene = Scene::new();
    filter_scene.layer(filter_layer, |scene| {
        scene.shadow(
            Rect::new(0.0, 0.0, 8.0, 8.0),
            Shadow::try_new(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
        );
    });
    let filter_error = filter_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("layer filters remain a typed full-stack diagnostic boundary");
    assert_eq!(
        filter_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );

    let mut inset_shadow_scene = Scene::new();
    inset_shadow_scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );
    let inset_shadow_error = inset_shadow_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("inset box shadows remain a typed shadow diagnostic boundary");
    assert_eq!(
        inset_shadow_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );

    let mask_layer = Layer::new()
        .try_mask(Shape::rect(Rect::new(0.0, 0.0, 6.0, 6.0)))
        .unwrap();
    let mut mask_scene = Scene::new();
    mask_scene.layer(mask_layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    });
    let mask_error = mask_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("authored layer masks remain a typed full-stack diagnostic boundary");
    assert_eq!(
        mask_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );

    let backdrop_filters = FilterList::try_ops(vec![FilterOp::opacity(
        UnitFilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let backdrop = BackdropFilterInput::try_new(
        backdrop_filters,
        BackdropCaptureBounds::try_new(Rect::new(0.0, 0.0, 8.0, 8.0)).unwrap(),
        Some(ClipInput::try_shape(Shape::rect(Rect::new(1.0, 1.0, 6.0, 6.0))).unwrap()),
    )
    .unwrap();
    let mut backdrop_scene = Scene::new();
    backdrop_scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
    backdrop_scene.layer(
        Layer::new()
            .try_transform(Transform::translation(1.0, 0.0).unwrap())
            .unwrap()
            .try_backdrop_filter(backdrop)
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(1.0, 1.0, 4.0, 4.0), Color::TRANSPARENT);
        },
    );
    let backdrop_error = backdrop_scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("transformed backdrop stacks remain explicitly unsupported");
    assert_eq!(
        backdrop_error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::OffscreenPipeline,
            PrimitiveOperation::BackdropExecution,
        ))
    );
    assert!(
        backdrop_error
            .message()
            .contains("transformed backdrop capture")
    );
}

#[test]
fn non_readback_renderer_front_door_is_async() {
    pollster::block_on(async {
        let mut renderer = Renderer::new(Options::default()).await.unwrap();
        let mut surface = renderer
            .create_surface(Attachment::Headless, SurfaceOptions::default())
            .await
            .unwrap();
        renderer
            .render(&mut surface, &Scene::new(), Parameters::default())
            .await
            .unwrap();
        renderer
            .resume_surface(&mut surface, Attachment::Headless)
            .await
            .unwrap();

        let headless = renderer
            .create_headless(Size::new(1.0, 1.0), 1.0)
            .await
            .unwrap();
        let _: Result<ImageBuffer> = renderer.read_headless(&headless);
    });
}

#[test]
fn surface_resize_rejects_physical_size_overflow_without_mutating_options() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 20.0), 1.5)).unwrap();

    let error = surface
        .resize(Size::try_new(f64::from(u32::MAX), 1.0).unwrap(), 2.0)
        .expect_err("physical device pixels should fit in u32");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(surface.size(), Size::new(10.0, 20.0));
    assert_eq!(surface.scale(), 1.5);
    assert_eq!(surface.physical_size(), PhysicalSize::new(15, 30));
}

#[test]
fn vello_out_of_memory_maps_to_stable_surface_error() {
    let error = vello::Error::WgpuErrorFromScope(wgpu::Error::OutOfMemory {
        source: Box::new(std::io::Error::other("oom")),
    });

    assert_eq!(
        vello_error_code(&error),
        BackendErrorCode::SurfaceOutOfMemory
    );
    assert!(vello_error_message(&error).contains("memory"));
}

#[test]
fn gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage() {
    let stages = [
        (
            GpuOperationStage::SurfaceCreate,
            BackendErrorCode::SurfaceCreateFailed,
        ),
        (
            GpuOperationStage::RendererCreate,
            BackendErrorCode::RendererCreateFailed,
        ),
        (GpuOperationStage::Render, BackendErrorCode::RenderFailed),
        (GpuOperationStage::Present, BackendErrorCode::PresentFailed),
    ];
    let faults = [
        GpuFaultKind::Validation,
        GpuFaultKind::OutOfMemory,
        GpuFaultKind::Internal,
    ];

    for (stage, expected_code) in stages {
        for fault in faults {
            let error = stage.classify_fault_for_test(fault, "injected GPU error");
            assert_eq!(
                error.code(),
                if fault == GpuFaultKind::OutOfMemory {
                    ErrorCode::SurfaceOutOfMemory
                } else {
                    Error::new(expected_code, "expected stage error").code()
                }
            );
        }
    }
}

#[test]
fn uncaptured_faults_observe_active_and_released_generations() {
    let signal = DeviceSignal::new_for_test();
    let lease = GpuOperationLease::begin_for_test(&signal).unwrap();
    let generation = lease.generation_for_test();

    signal.record_uncaptured_fault_for_test(GpuFaultKind::Validation, "active fault");
    let terminal = signal
        .finish_active_generation_for_test(generation)
        .unwrap();
    assert_eq!(terminal.operation_generation_for_test(), Some(generation));
    assert_eq!(signal.active_generation_for_test(), None);

    let late_signal = DeviceSignal::new_for_test();
    let late_lease = GpuOperationLease::begin_for_test(&late_signal).unwrap();
    let late_generation = late_lease.generation_for_test();
    assert!(
        late_signal
            .finish_active_generation_for_test(late_generation)
            .is_none()
    );
    late_signal.record_uncaptured_fault_for_test(GpuFaultKind::Internal, "late fault");
    assert_eq!(
        late_signal
            .first_terminal()
            .expect("late fault must terminally affect the next operation")
            .operation_generation_for_test(),
        None
    );
}

#[test]
fn terminal_record_snapshots_share_identity_and_keep_the_first_record() {
    let signal = DeviceSignal::new_for_test();
    let lease = GpuOperationLease::begin_for_test(&signal).unwrap();
    let generation = lease.generation_for_test();

    signal.record_uncaptured_fault_for_test(GpuFaultKind::Validation, "first terminal record");
    signal.record_uncaptured_fault_for_test(GpuFaultKind::Internal, "later terminal record");

    let first_snapshot = signal
        .first_terminal()
        .expect("the first terminal signal must be recorded");
    let repeated_snapshot = signal
        .first_terminal()
        .expect("repeated terminal snapshots must remain available");
    let finished_snapshot = signal
        .finish_active_generation_for_test(generation)
        .expect("finishing the active generation must observe the terminal record");

    assert!(Arc::ptr_eq(&first_snapshot, &repeated_snapshot));
    assert!(Arc::ptr_eq(&first_snapshot, &finished_snapshot));
    assert!(matches!(
        first_snapshot.as_ref(),
        DeviceTerminalSignal::Faulted {
            kind: GpuFaultKind::Validation,
            message,
            operation_generation: Some(observed_generation),
        } if message == "first terminal record" && *observed_generation == generation
    ));
}

#[test]
fn dropped_gpu_operation_future_aborts_draft_state_and_leases() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    if let Some(transaction) = renderer.start_default_gpu_operation_for_test() {
        let mut published = None;
        {
            let future = async {
                let draft = GpuOperationDraft::new(&mut published, true);
                pending::<()>().await;
                transaction
                    .finish(RuntimeOperation::SurfaceRendering)
                    .await?;
                draft.commit();
                Result::<()>::Ok(())
            };
            let mut future = std::pin::pin!(future);
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(
                Future::poll(future.as_mut(), &mut context),
                Poll::Pending
            ));
            assert!(
                renderer
                    .default_device_active_operation_generation_for_test()
                    .is_some()
            );
        }
        assert_eq!(published, None);
        assert_eq!(
            renderer.default_device_active_operation_generation_for_test(),
            None
        );
        return;
    }

    let signal = super::backend::DeviceSignal::new_for_test();
    let mut published = None;
    let future = async {
        let draft = GpuOperationDraft::new(&mut published, true);
        let lease = GpuOperationLease::begin_for_test(&signal).unwrap();
        assert!(signal.active_generation_for_test().is_some());
        pending::<()>().await;
        drop(lease);
        draft.commit();
    };
    {
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert_eq!(Future::poll(future.as_mut(), &mut context), Poll::Pending);
    }
    assert_eq!(published, None);
    assert_eq!(signal.active_generation_for_test(), None);
}

#[test]
fn real_gpu_error_scope_captures_deliberate_validation_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let result = pollster::block_on(renderer.deliberate_validation_error_for_test())
        .expect("real GPU error-scope coverage requires a host adapter");
    let error = result.expect_err("the deliberate invalid texture must be captured by the scope");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn real_gpu_smoke_emits_no_uncaptured_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "real GPU smoke coverage requires a host adapter"
    );
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .expect("real GPU smoke coverage requires a host adapter");
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the production Renderer::create_headless + Renderer::render path must be clean");
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[test]
fn create_headless_reports_unsupported_format() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let error = match pollster::block_on(renderer.create_surface(
        Attachment::Headless,
        SurfaceOptions {
            format: Format::Bgra8,
            ..SurfaceOptions::default()
        },
    )) {
        Ok(_) => panic!("unsupported headless format should fail before wgpu validation"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
    assert!(error.message().contains("Rgba8"));
}

#[test]
fn surface_suspend_and_resume_preserve_attachment_kind() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
    let scene = Scene::new();

    surface.suspend().unwrap();
    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("suspended surfaces should be unavailable");

    assert_eq!(error.code(), ErrorCode::SurfaceUnavailable);

    pollster::block_on(renderer.resume_surface(&mut surface, Attachment::Headless)).unwrap();
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("resumed headless surface should render");

    let error = surface
        .resume(Attachment::from_web_canvas("canvas"))
        .expect_err("surface backend kind should not change on resume");

    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
}

#[test]
fn foreign_and_stale_surfaces_fail_before_device_slot_access() {
    let mut owner = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut foreign_renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut foreign_surface =
        pollster::block_on(owner.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();

    if let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut foreign_surface.backend
    {
        device_identity.mark_stale_for_test();
    }

    assert_eq!(
        foreign_renderer.runtime_capabilities(&foreign_surface),
        RuntimeCapabilities::Unavailable(
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                kind: SurfaceIdentityMismatchKind::ForeignRenderer,
            }
        ),
        "a foreign renderer must reject the surface before consulting its stale device identity"
    );

    let error = pollster::block_on(foreign_renderer.render(
        &mut foreign_surface,
        &Scene::new(),
        Parameters::default(),
    ))
    .expect_err("foreign surfaces must fail before indexing their device slot");

    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceRendering,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );
    let error = foreign_renderer
        .read_headless(&foreign_surface)
        .expect_err("foreign readback must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceReadback,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );
    let error = pollster::block_on(
        foreign_renderer.resume_surface(&mut foreign_surface, Attachment::Headless),
    )
    .expect_err("foreign resume must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceResume,
        SurfaceIdentityMismatchKind::ForeignRenderer,
    );

    let mut stale_surface =
        pollster::block_on(owner.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let SurfaceBackend::Headless {
        device_identity, ..
    } = &mut stale_surface.backend
    else {
        panic!("the test environment must provide a device-backed headless surface");
    };
    device_identity.mark_stale_for_test();

    assert_eq!(
        owner.runtime_capabilities(&stale_surface),
        RuntimeCapabilities::Unavailable(
            RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch {
                kind: SurfaceIdentityMismatchKind::StaleDeviceGeneration,
            }
        ),
        "a stale surface must be rejected before runtime capability projection"
    );

    let error =
        pollster::block_on(owner.render(&mut stale_surface, &Scene::new(), Parameters::default()))
            .expect_err("stale rendering must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceRendering,
        SurfaceIdentityMismatchKind::StaleDeviceGeneration,
    );
    let error = owner
        .read_headless(&stale_surface)
        .expect_err("stale readback must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceReadback,
        SurfaceIdentityMismatchKind::StaleDeviceGeneration,
    );
    let error = pollster::block_on(owner.resume_surface(
        &mut stale_surface,
        Attachment::from_web_canvas("incompatible-canvas"),
    ))
    .expect_err("incompatible resume must fail before stale device validation");
    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
    let error = pollster::block_on(owner.resume_surface(&mut stale_surface, Attachment::Headless))
        .expect_err("stale resume must fail before indexing the device slot");
    assert_surface_identity_mismatch(
        error,
        RuntimeOperation::SurfaceResume,
        SurfaceIdentityMismatchKind::StaleDeviceGeneration,
    );
}

#[test]
fn device_loss_is_terminal_idempotent_and_releases_device_resources() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("a signaled device loss must prevent further Vello use");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );
    assert!(renderer.default_device_renderer_released_for_test());
}

#[test]
fn terminal_default_device_rejects_headless_without_disabling_ready_slots() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Destroyed);
    let error = match pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)) {
        Ok(_) => panic!("a terminal default device must not be replaced automatically"),
        Err(error) => error,
    };
    assert_runtime_device_lost(
        error,
        RuntimeOperation::AdapterSelection,
        DeviceLossReason::Destroyed,
    );
}

#[test]
fn terminal_signal_during_renderer_creation_aborts_before_followup_gpu_work() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    renderer.arm_default_terminal_signal_after_renderer_creation_for_test();

    let error = match pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)) {
        Ok(_) => panic!("a terminal signal during renderer creation must abort headless creation"),
        Err(error) => error,
    };

    assert_runtime_device_lost(
        error,
        RuntimeOperation::AdapterSelection,
        DeviceLossReason::Destroyed,
    );
    assert!(renderer.default_device_renderer_released_for_test());
}

#[test]
fn runtime_capabilities_project_the_selected_surface_without_gpu_work() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let surface = pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();

    let report = renderer.runtime_capabilities(&surface);
    let available = report
        .available()
        .expect("a device-backed headless surface must project immutable capabilities");
    assert_eq!(available.surface_format(), Format::Rgba8);
    assert_eq!(
        available,
        renderer.default_device_capabilities_for_test(),
        "the query must project the snapshotted state without another GPU call"
    );
}

#[test]
fn destroyed_device_callback_reports_terminal_loss_without_stale_resource_use() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let ready_slot = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("the destroyed-device test requires a second real WGPU device slot");

    assert!(renderer.destroy_default_device_for_test());
    assert!(
        renderer.wait_for_default_terminal_signal_for_test(Duration::from_secs(5)),
        "device destruction did not invoke the loss callback within the diagnostic deadline"
    );

    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("a destroyed device must be observed before any stale Vello use");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );

    let error = match pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)) {
        Ok(_) => panic!("a destroyed device must not create another headless surface"),
        Err(error) => error,
    };
    assert_runtime_device_lost(
        error,
        RuntimeOperation::AdapterSelection,
        DeviceLossReason::Destroyed,
    );

    assert!(renderer.default_device_renderer_released_for_test());
    pollster::block_on(renderer.submit_scoped_wgpu_probe_for_test(ready_slot))
        .expect("a ready second slot must submit and finish a real scoped WGPU operation");
}

fn assert_runtime_device_lost(error: Error, operation: RuntimeOperation, reason: DeviceLossReason) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::DeviceLost { reason },
            )
            .unwrap()
        )
    );
}

fn assert_surface_identity_mismatch(
    error: Error,
    operation: RuntimeOperation,
    kind: SurfaceIdentityMismatchKind,
) {
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                operation,
                RuntimeCapabilityUnavailableReason::SurfaceIdentityMismatch { kind },
            )
            .unwrap()
        )
    );
}

#[cfg(not(all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn unsupported_web_canvas_attachment_reports_target_requirement() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let canvas = WebCanvas::new("preview");

    assert_eq!(canvas.id(), "preview");

    let error = match pollster::block_on(renderer.create_surface(
        Attachment::WebCanvas(canvas),
        SurfaceOptions {
            size: Size::new(10.0, 10.0),
            ..SurfaceOptions::default()
        },
    )) {
        Ok(_) => panic!("native test targets should not create web canvas surfaces"),
        Err(error) => error,
    };

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Surfaces,
            PrimitiveOperation::WebCanvasSurface,
        ))
    );
    assert!(error.message().contains("web canvas surface"));
}

#[test]
fn render_reports_command_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene
        .fill(Rect::new(0.0, 0.0, 5.0, 5.0), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.stroke(
                Rect::new(1.0, 1.0, 3.0, 3.0),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("headless render should report stats");

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
    assert!(stats.frame_time >= stats.encode_time);
    assert!(stats.frame_time >= stats.render_time);
    assert_eq!(stats.present_time, Duration::ZERO);
}

#[test]
fn render_scales_logical_scene_to_physical_surface() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(20.0, 20.0), 2.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 10.0, 10.0), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(output.size, PhysicalSize::new(40, 40));
    assert!(pixel_alpha(&output, 18, 18) > 0);
    assert_eq!(pixel_alpha(&output, 22, 22), 0);
}

#[test]
fn warm_image_reuse_reports_cache_hit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    assert_eq!(image_data(&image), image_data(&image.clone()));
    let mut scene = Scene::new();
    scene.image(
        image.clone(),
        Rect::new(0.0, 0.0, 1.0, 1.0),
        ImageFit::Stretch,
    );

    let cold =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let warm =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();

    assert_eq!(cold.cache_misses, 1);
    assert_eq!(warm.cache_hits, 1);
}

#[test]
fn failed_render_does_not_warm_image_reuse_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut failing = Scene::new();
    failing.image(
        image.clone(),
        Rect::new(0.0, 0.0, 1.0, 1.0),
        ImageFit::Stretch,
    );
    failing.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &failing, Parameters::default()))
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);

    let mut valid = Scene::new();
    valid.image(image, Rect::new(0.0, 0.0, 1.0, 1.0), ImageFit::Stretch);

    let stats = pollster::block_on(renderer.render(&mut surface, &valid, Parameters::default()))
        .expect("valid render should still see cold image");

    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn rejects_malformed_rgba_images() {
    let error = Image::from_rgba(Size::new(2.0, 2.0), Arc::<[u8]>::from([0, 0, 0, 255]))
        .expect_err("wrong byte length should fail");

    assert_eq!(error.code(), ErrorCode::ImageUploadFailed);
    assert!(error.message().contains("expected 16 bytes"));

    let error = Image::from_rgba(Size::new(1.5, 2.0), Arc::<[u8]>::from([]))
        .expect_err("fractional source image size should fail");

    assert_eq!(error.code(), ErrorCode::ImageUploadFailed);
    assert!(error.message().contains("integer pixel size"));
}

#[test]
fn rejects_malformed_scene_values() {
    let error = Color::try_rgba(f32::NAN, 0.0, 0.0, 1.0)
        .expect_err("invalid paint should fail at construction");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.message().contains("red channel"));
}

#[test]
fn concrete_color_paint_renders_without_color_realization() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("concrete color paint should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn gradient_paint_renders_with_transparent_stop() {
    let gradient = Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(2.0, 0.0).unwrap(),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), gradient);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("gradient paint should render");
}

#[test]
fn image_paint_lowers_to_brush() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Paint::image(image));

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.fills, 1);
    assert_eq!(stats.images, 1);
    assert!(pixel_alpha(&output, 0, 0) > 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
}

#[test]
fn image_brush_preserves_sampling_and_extend() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255]))
        .unwrap()
        .quality(ImageQuality::High)
        .extend(Extend::Reflect);

    let brush = image_brush(&image);

    assert_eq!(brush.sampler.quality, peniko::ImageQuality::High);
    assert_eq!(brush.sampler.x_extend, peniko::Extend::Reflect);
    assert_eq!(brush.sampler.y_extend, peniko::Extend::Reflect);
}

#[test]
fn cover_image_fit_clips_to_target_rect() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut pixels = Vec::new();
    for _ in 0..8 {
        pixels.extend_from_slice(&[255, 0, 0, 255]);
    }
    let image = Image::from_rgba(Size::new(4.0, 2.0), Arc::<[u8]>::from(pixels)).unwrap();
    let mut scene = Scene::new();
    scene.image(image, Rect::new(1.0, 0.0, 2.0, 2.0), ImageFit::Cover);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert_eq!(pixel_alpha(&output, 3, 0), 0);
}

#[test]
fn image_fit_transforms_use_uniform_scale() {
    let contain = image_transform(
        Size::new(4.0, 2.0),
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Contain,
    )
    .unwrap()
    .as_coeffs();
    let cover = image_transform(
        Size::new(4.0, 2.0),
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Cover,
    )
    .unwrap()
    .as_coeffs();

    assert_eq!(contain[0], 0.5);
    assert_eq!(contain[3], 0.5);
    assert_eq!(contain[5], 0.5);
    assert_eq!(cover[0], 1.0);
    assert_eq!(cover[3], 1.0);
    assert_eq!(cover[4], -1.0);
}

#[test]
fn layer_transform_moves_child_content() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.transform(
        Transform::try_new([1.0, 0.0, 0.0, 1.0, 2.0, 0.0]).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn composed_layer_transforms_render_in_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 2.0), 1.0)).unwrap();
    let transform = Transform::translation(1.0, 0.0)
        .unwrap()
        .then(Transform::scale(2.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 2.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("composed transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn origin_wrapped_layer_transform_renders_about_origin() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let transform = Transform::scale(2.0, 2.0)
        .unwrap()
        .around(Point::try_new(1.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(1.0, 1.0, 1.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("origin-wrapped transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
    assert!(pixel_alpha(&output, 2, 2) > 0);
}

#[test]
fn transformed_shape_clips_render_in_layer_space() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_transform(Transform::translation(2.0, 0.0).unwrap())
            .unwrap()
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("transformed clip should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn path_clip_fill_rules_execute_even_odd_and_nonzero() {
    fn nested_rect_path() -> Path {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(5.0, 0.0))
            .line_to(Point::new(5.0, 5.0))
            .line_to(Point::new(0.0, 5.0))
            .close()
            .move_to(Point::new(1.0, 1.0))
            .line_to(Point::new(4.0, 1.0))
            .line_to(Point::new(4.0, 4.0))
            .line_to(Point::new(1.0, 4.0))
            .close();
        path
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut even_odd_surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 5.0), 1.0)).unwrap();
    let even_odd_clip = ClipInput::try_filled_path(
        FilledPath::try_new(nested_rect_path(), FillRule::EvenOdd).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_clip_input(even_odd_clip).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 6.0, 5.0), Color::BLACK);
        },
    );
    pollster::block_on(renderer.render(&mut even_odd_surface, &scene, Parameters::default()))
        .expect("even-odd path clip should render");
    let even_odd = renderer.read_headless(&even_odd_surface).unwrap();

    let mut nonzero_surface =
        pollster::block_on(renderer.create_headless(Size::new(6.0, 5.0), 1.0)).unwrap();
    let nonzero_clip = ClipInput::try_filled_path(
        FilledPath::try_new(nested_rect_path(), FillRule::NonZero).unwrap(),
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new().try_clip_input(nonzero_clip).unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 6.0, 5.0), Color::BLACK);
        },
    );
    pollster::block_on(renderer.render(&mut nonzero_surface, &scene, Parameters::default()))
        .expect("nonzero path clip should render");
    let nonzero = renderer.read_headless(&nonzero_surface).unwrap();

    assert!(pixel_alpha(&even_odd, 0, 0) > 0);
    assert_eq!(pixel_alpha(&even_odd, 2, 2), 0);
    assert!(pixel_alpha(&nonzero, 2, 2) > 0);
}

#[test]
fn builtin_shape_clips_execute_for_layer_clipping() {
    let clips = [
        Shape::try_rounded_rect(
            Rect::new(0.0, 0.0, 4.0, 4.0),
            Radii::new(1.0, 1.0, 1.0, 1.0),
        )
        .unwrap(),
        Shape::try_circle(Point::new(2.0, 2.0), 2.0).unwrap(),
        Shape::try_ellipse(Point::new(2.0, 2.0), Size::new(2.0, 1.5)).unwrap(),
    ];

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    for clip in clips {
        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
        let mut scene = Scene::new();
        scene.layer(Layer::new().try_clip(clip).unwrap(), |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
        });

        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("builtin shape clip should render as a layer clip");
        let output = renderer.read_headless(&surface).unwrap();

        assert!(
            output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0),
            "builtin shape clip should leave visible clipped content"
        );
    }
}

#[test]
fn nested_clips_render_only_the_intersection() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(5.0, 2.0), 1.0)).unwrap();
    let mut inner_path = Path::new();
    inner_path
        .move_to(Point::new(2.0, 0.0))
        .line_to(Point::new(5.0, 0.0))
        .line_to(Point::new(5.0, 2.0))
        .line_to(Point::new(2.0, 2.0))
        .close();
    let inner_clip =
        ClipInput::try_filled_path(FilledPath::try_new(inner_path, FillRule::NonZero).unwrap())
            .unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_clip(Shape::rect(Rect::new(1.0, 0.0, 3.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.layer(Layer::new().try_clip_input(inner_clip).unwrap(), |scene| {
                scene.fill(Rect::new(0.0, 0.0, 5.0, 2.0), Color::BLACK);
            });
        },
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("nested clips should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
    assert_eq!(pixel_alpha(&output, 4, 0), 0);
}

#[test]
fn coordinate_space_tag_transform_affects_layer_clip() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
        .unwrap()
        .with_coordinate_space(
            CoordinateSpaceTag::surface(Transform::translation(2.0, 0.0).unwrap()).unwrap(),
        );
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_clip_input(clip).unwrap(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("coordinate-space clip transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn scene_clip_convenience_still_uses_shape_layer_clips() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(3.0, 1.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.clip(Rect::new(1.0, 0.0, 1.0, 1.0), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 3.0, 1.0), Color::BLACK);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("existing Scene::clip convenience should keep working");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 0) > 0);
    assert_eq!(pixel_alpha(&output, 2, 0), 0);
}

#[test]
fn transformed_images_render_in_layer_space() {
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.transform(Transform::translation(2.0, 0.0).unwrap(), |scene| {
        scene.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    });

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("transformed image should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
}

#[test]
fn pure_transform_does_not_require_backend_layer() {
    let transform = Layer::new()
        .try_transform(Transform::try_new([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap())
        .unwrap();
    let clip = Layer::new()
        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap();
    let opacity = Layer::new().try_opacity(0.5).unwrap();

    let mut scene = Scene::new();
    scene
        .layer(transform, |_| {})
        .layer(clip, |_| {})
        .layer(opacity, |_| {});

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let isolations: Vec<_> = normalized
        .commands
        .iter()
        .map(|command| match command {
            command::RenderCommand::Layer { layer, .. } => layer.isolation,
            _ => panic!("expected layer command"),
        })
        .collect();

    assert_eq!(
        isolations,
        [
            command::LayerIsolation::None,
            command::LayerIsolation::ClipOnly,
            command::LayerIsolation::BackendLayer,
        ]
    );
}

#[test]
fn layer_pass_plan_uses_clip_bounds_before_child_geometry() {
    let clip = Layer::new()
        .try_clip(Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0)))
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(clip, |scene| {
        scene.fill(Rect::new(-10.0, -10.0, 50.0, 50.0), Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(layer.isolation, command::LayerIsolation::ClipOnly);
    assert_eq!(layer.pass_plan.kind(), command::LayerPassKind::ClipOnly);
    assert_eq!(
        layer.pass_plan.requirement(),
        command::LayerPassRequirement::ClipOnly
    );
    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(1.0, 2.0, 3.0, 4.0))
    );
}

#[test]
fn layer_pass_plan_names_opacity_and_blend_direct_layers() {
    let opacity = Layer::new().try_opacity(0.5).unwrap();
    let blend = Layer::new().blend(BlendMode::Multiply);
    let mut scene = Scene::new();
    scene
        .layer(opacity, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        })
        .layer(blend, |scene| {
            scene.fill(Rect::new(4.0, 0.0, 2.0, 2.0), Color::BLACK);
        });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let plans: Vec<_> = normalized
        .commands
        .iter()
        .map(|command| match command {
            command::RenderCommand::Layer { layer, .. } => (
                layer.isolation,
                layer.pass_plan.kind(),
                layer.pass_plan.requirement(),
            ),
            _ => panic!("expected layer command"),
        })
        .collect();

    assert_eq!(
        plans,
        [
            (
                command::LayerIsolation::BackendLayer,
                command::LayerPassKind::DirectVelloLayer,
                command::LayerPassRequirement::DirectVelloOpacity,
            ),
            (
                command::LayerIsolation::BackendLayer,
                command::LayerPassKind::DirectVelloLayer,
                command::LayerPassRequirement::DirectVelloBlend,
            ),
        ]
    );
}

#[test]
fn nested_layer_pass_plan_aggregates_transformed_child_bounds() {
    let outer = Layer::new().try_opacity(0.5).unwrap();
    let inner = Layer::new()
        .try_transform(Transform::translation(4.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(outer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        scene.layer(inner, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 3.0, 2.0), Color::BLACK);
        });
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected outer layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(0.0, 0.0, 7.0, 3.0))
    );
}

#[test]
fn layer_pass_plan_rejects_mask_filter_boundaries_with_typed_diagnostics() {
    let cases = [
        (
            Layer::new()
                .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
                .unwrap(),
            UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ),
        ),
        (
            Layer::new()
                .try_filter(Filter::try_blur(4.0).unwrap())
                .unwrap(),
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        ),
    ];

    for (layer, primitive) in cases {
        let mut scene = Scene::new();
        scene.layer(layer, |scene| {
            scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
        });

        let error = scene
            .normalize(Capabilities::VELLO_0_9)
            .expect_err("mask/filter layer pass planning should stop at diagnostic boundary");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(primitive));
    }
}

#[test]
fn layer_mask_filter_parent_diagnostics_win_over_unsupported_children() {
    let cases = [
        (
            Layer::new()
                .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
                .unwrap(),
            UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LayerMask,
            ),
        ),
        (
            Layer::new()
                .try_filter(Filter::try_blur(4.0).unwrap())
                .unwrap(),
            UnsupportedPrimitive::new(PrimitiveFamily::Filters, PrimitiveOperation::LayerFilter),
        ),
    ];

    for (layer, primitive) in cases {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(8.0, 0.0));
        let mut scene = Scene::new();
        scene.layer(layer, |scene| {
            scene.stroke(
                Shape::path(path),
                Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
                Color::BLACK,
            );
        });

        let error = scene
            .normalize(Capabilities::VELLO_0_9)
            .expect_err("parent layer diagnostic should be reported before child geometry");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(primitive));
    }
}

#[test]
fn path_stroke_layer_bounds_include_miter_limit_conservatively() {
    let mut path = Path::new();
    path.move_to(Point::new(10.0, 10.0))
        .line_to(Point::new(20.0, 10.0));
    let stroke = Stroke::try_new(4.0).unwrap().try_miter_limit(10.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.stroke(Shape::path(path), stroke, Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(
        layer.pass_plan.bounds().map(command::OffscreenBounds::rect),
        Some(Rect::new(-10.0, -10.0, 50.0, 40.0))
    );
}

#[test]
fn exact_epsilon_opacity_with_clip_keeps_backend_layer_isolation() {
    let opacity = 1.0 - f32::EPSILON;
    assert_eq!((opacity - 1.0).abs(), f32::EPSILON);
    let layer = Layer::new()
        .try_clip(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap()
        .try_opacity(opacity)
        .unwrap();
    let mut scene = Scene::new();
    scene.layer(layer, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 1.0), Color::BLACK);
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer { layer, .. } = &normalized.commands[0] else {
        panic!("expected layer command");
    };

    assert_eq!(layer.isolation, command::LayerIsolation::BackendLayer);
    assert_eq!(
        layer.pass_plan.kind(),
        command::LayerPassKind::DirectVelloLayer
    );
}

#[test]
fn layer_default_is_visible() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::default(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("default layer should render visible content");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.layers, 1);
    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn layer_opacity_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(Layer::new().try_opacity(0.5).unwrap(), |scene| {
        scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);
    });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("opacity layer should render");
    let output = renderer.read_headless(&surface).unwrap();
    let [_, _, _, alpha] = pixel_rgba(&output, 0, 0);

    assert_eq!(stats.layers, 1);
    assert!(alpha > 0);
    assert!(alpha < 255);
}

#[test]
fn layer_blend_isolates_child_output() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
    );
    scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 2.0, 2.0),
            Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap(),
        );
    });

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("blend layer should render");
    let output = renderer.read_headless(&surface).unwrap();
    let [red, green, blue, alpha] = pixel_rgba(&output, 0, 0);

    assert_eq!(stats.layers, 1);
    assert!(red < 32, "red channel should be multiplied down: {red}");
    assert!(
        green < 32,
        "green channel should be multiplied down: {green}"
    );
    assert!(blue < 32, "blue channel should be multiplied down: {blue}");
    assert!(alpha > 0);
}

#[test]
fn direct_vello_blend_modes_match_reference_oracle_for_opaque_pixels() {
    let source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    for mode in [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ] {
        let mut scene = Scene::new();
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(destination),
        );
        scene.layer(Layer::new().blend(mode), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(source),
            );
        });

        let output = render_scene_pixel(&mut renderer, &scene);
        let expected = source.blend_over(destination, mode);

        assert_rgba_near_reference_pixel(
            output,
            expected,
            2,
            &format!("direct Vello {mode:?} blend should stay aligned with the CPU oracle"),
        );
    }
}

#[test]
fn blend_layer_isolation_changes_backdrop_composition_from_normal_paint_order() {
    let source = PremultipliedRgba8::try_new(64, 128, 192, 255).unwrap();
    let destination = PremultipliedRgba8::try_new(192, 128, 64, 255).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();

    let mut normal_scene = Scene::new();
    normal_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(destination),
    );
    normal_scene.layer(Layer::new(), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(source),
        );
    });

    let mut blended_scene = Scene::new();
    blended_scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(destination),
    );
    blended_scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(source),
        );
    });

    let normal_output = render_scene_pixel(&mut renderer, &normal_scene);
    let blended_output = render_scene_pixel(&mut renderer, &blended_scene);
    let expected_blend = source.blend_over(destination, BlendMode::Multiply);

    assert_rgba_near_reference_pixel(
        normal_output,
        source,
        2,
        "non-isolated normal layer should paint its children in command order",
    );
    assert_rgba_near_reference_pixel(
        blended_output,
        expected_blend,
        2,
        "blend layer should isolate its child output before blending with prior backdrop",
    );
    assert_ne!(
        normal_output, blended_output,
        "multiply isolation should produce a different pixel than normal child painting"
    );
}

#[test]
fn nested_direct_vello_blend_groups_match_nested_reference_oracle() {
    let backdrop = PremultipliedRgba8::try_new(64, 192, 96, 255).unwrap();
    let outer_child_backdrop = PremultipliedRgba8::try_new(128, 128, 128, 255).unwrap();
    let inner_source = PremultipliedRgba8::try_new(192, 64, 128, 255).unwrap();
    let expected_inner = inner_source.blend_over(outer_child_backdrop, BlendMode::Multiply);
    let expected_outer = expected_inner.blend_over(backdrop, BlendMode::Screen);

    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 1.0, 1.0),
        color_from_opaque_rgba8(backdrop),
    );
    scene.layer(Layer::new().blend(BlendMode::Screen), |scene| {
        scene.fill(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            color_from_opaque_rgba8(outer_child_backdrop),
        );
        scene.layer(Layer::new().blend(BlendMode::Multiply), |scene| {
            scene.fill(
                Rect::new(0.0, 0.0, 1.0, 1.0),
                color_from_opaque_rgba8(inner_source),
            );
        });
    });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Layer {
        layer: outer,
        children,
    } = &normalized.commands[1]
    else {
        panic!("expected outer blend layer command");
    };
    let command::RenderCommand::Layer { layer: inner, .. } = &children[1] else {
        panic!("expected nested blend layer command");
    };
    for layer in [outer, inner] {
        assert_eq!(layer.isolation, command::LayerIsolation::BackendLayer);
        assert_eq!(
            layer.pass_plan.requirement(),
            command::LayerPassRequirement::DirectVelloBlend
        );
        assert_eq!(
            layer.pass_plan.kind(),
            command::LayerPassKind::DirectVelloLayer
        );
    }

    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let output = render_scene_pixel(&mut renderer, &scene);

    assert_rgba_near_reference_pixel(
        output,
        expected_outer,
        2,
        "nested direct Vello blend groups should compose in command order",
    );
}

#[test]
fn unsupported_blend_and_composite_boundaries_remain_typed_diagnostics() {
    let public_layer_modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Plus,
    ];
    assert_eq!(
        public_layer_modes.len(),
        7,
        "Task 6 should not expand layer BlendMode without encoding and tests"
    );

    for mode in [
        BackgroundBlendMode::Multiply,
        BackgroundBlendMode::Screen,
        BackgroundBlendMode::Overlay,
        BackgroundBlendMode::Darken,
        BackgroundBlendMode::Lighten,
        BackgroundBlendMode::Plus,
    ] {
        let error = BackgroundBlendList::try_new(vec![BackgroundBlendMode::Normal, mode])
            .expect_err("background-layer blending is not routed through layer BlendMode");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::Compositing,
                PrimitiveOperation::BackgroundBlendMode,
            ))
        );
    }

    for operation in [
        PrimitiveOperation::AdditionalMixBlendMode,
        PrimitiveOperation::PorterDuffCompositeMode,
        PrimitiveOperation::RootBackdropPolicy,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::Compositing, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("future blend/composite policy must stay behind typed diagnostics");

        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(
            error.message().contains(unsupported.label()),
            "diagnostic should name unsupported compositing boundary: {}",
            error.message()
        );
    }
}

#[test]
fn unsupported_porter_duff_css_and_mask_composite_policy_stays_typed() {
    let compositing = Capabilities::VELLO_0_9.compositing();
    assert!(!compositing.supports_background_blend_modes());
    assert!(!compositing.supports_additional_mix_blend_modes());
    assert!(!compositing.supports_porter_duff_composite_modes());

    for operation in [
        PrimitiveOperation::BackgroundBlendMode,
        PrimitiveOperation::AdditionalMixBlendMode,
        PrimitiveOperation::PorterDuffCompositeMode,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::Compositing, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("unsupported CSS and Porter-Duff composite policy stays typed");

        assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message().contains("compositing"));
        assert!(error.message().contains(unsupported.label()));
    }

    let alpha_mask =
        MaskInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)), MaskMode::Alpha).unwrap();
    for mode in [
        MaskCompositeMode::Subtract,
        MaskCompositeMode::Intersect,
        MaskCompositeMode::Exclude,
    ] {
        let stack = MaskLayerStack::single(MaskLayer::try_new(alpha_mask.clone(), mode).unwrap());
        let error = stack
            .ensure_supported(Capabilities::VELLO_0_9)
            .expect_err("non-default mask composites remain unsupported until fully implemented");

        assert_eq!(
            error.unsupported_primitive(),
            Some(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MaskCompositeMode,
            ))
        );
    }
}

#[test]
fn text_run_requires_font_data() {
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            FontRef::new(1).named("Test"),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("prepared glyphs cannot render without font data");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert!(error.message().contains("font data"));
}

#[test]
fn text_run_with_gradient_fill_still_requires_font_data_before_brush_encoding() {
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(10.0, 0.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let glyphs = [TextGlyph::try_new(1, 0.0, 0.0, 5.0).unwrap()];
    let mut scene = Scene::new();
    scene.text_run(
        TextRun::try_new(
            FontRef::new(1).named("Test"),
            16.0,
            Transform::identity(),
            TextPaint::try_fill(Paint::gradient(gradient)).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(10.0, 10.0), 1.0)).unwrap();

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("prepared glyphs cannot render without font data");

    assert_eq!(error.code(), ErrorCode::InvalidInput);
    assert_eq!(error.unsupported_primitive(), None);
    assert!(error.message().contains("font data"));
}

#[test]
fn inside_and_outside_strokes_lower_for_builtin_shapes() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene
        .stroke(
            Rect::new(4.0, 4.0, 16.0, 16.0),
            Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
            Color::BLACK,
        )
        .stroke(
            Shape::try_circle(Point::new(12.0, 12.0), 6.0).unwrap(),
            Stroke::try_new(2.0).unwrap().align(StrokeAlign::Outside),
            Color::BLACK,
        );

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();

    assert_eq!(stats.strokes, 2);
}

#[test]
fn aligned_rect_strokes_do_not_cross_source_edge() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(12.0, 12.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let inside = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&inside, 2, 6), 0);
    assert!(pixel_alpha(&inside, 3, 6) > 0);

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(12.0, 12.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.stroke(
        Rect::new(3.0, 3.0, 6.0, 6.0),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Outside),
        Color::BLACK,
    );

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let outside = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&outside, 2, 6) > 0);
    assert_eq!(pixel_alpha(&outside, 4, 6), 0);
}

#[test]
fn circle_shadows_lower_to_blurred_round_rect() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_circle(Point::new(12.0, 12.0), 4.0).unwrap(),
        Shadow::try_new(Point::new(1.0, 1.0), 4.0, 1.0, Color::BLACK).unwrap(),
    );

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn non_uniform_rounded_rect_shadows_render_with_corner_partition() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(40.0, 36.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_rounded_rect(
            Rect::new(8.0, 8.0, 16.0, 14.0),
            Radii::new(0.0, 5.0, 10.0, 0.0),
        )
        .unwrap(),
        Shadow::try_new(Point::new(4.0, 5.0), 8.0, 0.0, Color::BLACK).unwrap(),
    );

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("non-uniform rounded shadow should render through corner partitioning");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(stats.shadows, 1);
    assert!(output.rgba.chunks_exact(4).any(|pixel| pixel[3] > 0));
}

#[test]
fn outer_box_shadow_list_normalizes_offset_blur_spread_and_order() {
    let first = Shadow::try_new(Point::new(3.0, -2.0), 6.0, 1.5, Color::BLACK).unwrap();
    let second = Shadow::try_new(Point::new(-4.0, 5.0), 0.0, -1.0, Color::BLACK).unwrap();
    let shadows = ShadowList::try_new(vec![first.clone(), second.clone()]).unwrap();
    let mut scene = Scene::new();

    scene.shadows(Rect::new(8.0, 8.0, 10.0, 10.0), shadows);

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    assert_eq!(normalized.commands.len(), 2);
    assert_eq!(normalized.stats().shadows, 2);

    let command::RenderCommand::Shadow { shadow, .. } = &normalized.commands[0] else {
        panic!("first shadow-list entry should lower to a render shadow");
    };
    assert_eq!(shadow.offset, first.offset());
    assert_eq!(shadow.blur, first.blur());
    assert_eq!(shadow.spread, first.spread());

    let command::RenderCommand::Shadow { shadow, .. } = &normalized.commands[1] else {
        panic!("second shadow-list entry should lower to a render shadow");
    };
    assert_eq!(shadow.offset, second.offset());
    assert_eq!(shadow.blur, second.blur());
    assert_eq!(shadow.spread, second.spread());
}

#[test]
fn non_uniform_rounded_outer_shadow_preserves_authored_radii() {
    let radii = Radii::new(0.0, 4.0, 8.0, 12.0);
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_rounded_rect(Rect::new(4.0, 4.0, 16.0, 12.0), radii).unwrap(),
        Shadow::try_new(Point::new(2.0, 2.0), 4.0, 1.0, Color::BLACK).unwrap(),
    );

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let command::RenderCommand::Shadow { shape, .. } = &normalized.commands[0] else {
        panic!("rounded rect shadow should lower to a render shadow");
    };
    let command::ShadowShape::RoundedRect {
        radii: lowered_radii,
        ..
    } = shape
    else {
        panic!("rounded rect shadow should preserve rounded geometry");
    };
    assert_eq!(*lowered_radii, radii);
}

#[test]
fn multiple_outer_shadows_render_in_authored_order() {
    let red = Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap();
    let blue = Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap();
    let shadows = ShadowList::try_new(vec![
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, red).unwrap(),
        Shadow::try_new(Point::new(0.0, 0.0), 0.0, 0.0, blue).unwrap(),
    ])
    .unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadows(Rect::new(1.0, 1.0, 6.0, 6.0), shadows);

    let stats =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let output = renderer.read_headless(&surface).unwrap();
    let overlap = pixel_rgba(&output, 4, 4);

    assert_eq!(stats.shadows, 2);
    assert!(
        overlap[2] > overlap[0],
        "last overlapping shadow should be composited above earlier shadows: {overlap:?}"
    );
}

#[test]
fn inset_box_shadow_reports_typed_unsupported_diagnostic() {
    let mut scene = Scene::new();
    scene.shadow(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Shadow::try_inset(Point::new(1.0, 1.0), 2.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("inset shadow execution is not implemented in this phase");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::InsetBoxShadow,
        ))
    );
    assert!(error.message().contains("inset box shadow"));
}

#[test]
fn direct_geometry_targets_render_without_unsupported_diagnostics() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(32.0, 32.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 30.0).unwrap())
        .close();

    scene.fill(
        Shape::rect(Rect::try_new(1.0, 1.0, 4.0, 4.0).unwrap()),
        Color::BLACK,
    );
    scene.stroke(
        Shape::rect(Rect::try_new(1.0, 7.0, 4.0, 4.0).unwrap()),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 1.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 7.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_circle(Point::try_new(4.0, 14.0).unwrap(), 2.0).unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_circle(Point::try_new(4.0, 20.0).unwrap(), 2.0).unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_ellipse(
            Point::try_new(14.0, 14.0).unwrap(),
            Size::try_new(3.0, 2.0).unwrap(),
        )
        .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_ellipse(
            Point::try_new(14.0, 20.0).unwrap(),
            Size::try_new(3.0, 2.0).unwrap(),
        )
        .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(Shape::path(path), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("direct geometry targets should render");
}

#[test]
fn centered_path_strokes_support_join_cap_and_dash_inputs() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 20.0).unwrap());
    let stroke = Stroke::try_new(2.0)
        .unwrap()
        .join(LineJoin::Round)
        .caps(LineCap::Round, LineCap::Square)
        .try_dash(Dash::try_new(0.0, &[2.0, 1.0]).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.stroke(Shape::path(path), stroke, Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("centered path strokes should render");
}

#[test]
fn inside_outside_path_strokes_keep_typed_geometry_diagnostic() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(1.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 6.0).unwrap())
        .close();
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(1.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("inside path stroke alignment requires offset lowering");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
}

#[test]
fn unsupported_aligned_path_strokes_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(10.0, 10.0));
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("path offsetting is deliberately explicit");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
    assert!(
        error
            .message()
            .contains("inside/outside path stroke alignment")
    );
}

#[test]
fn unsupported_layer_masks_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 2.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("mask lowering should be explicit until implemented");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::MasksAndClips,
            PrimitiveOperation::LayerMask,
        ))
    );
    assert!(error.message().contains("layer mask"));
}

#[test]
fn unsupported_layer_filters_report_explicit_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(24.0, 24.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_filter(Filter::try_blur(4.0).unwrap())
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
        },
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("filter lowering should be explicit until implemented");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Filters,
            PrimitiveOperation::LayerFilter,
        ))
    );
    assert!(error.message().contains("layer filter"));
}

#[test]
fn unsupported_non_solid_shadow_paint_reports_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Rect::new(0.0, 0.0, 2.0, 2.0),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Paint::gradient(gradient)).unwrap(),
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("shadow lowering requires solid paint in this milestone");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::PaintSources,
            PrimitiveOperation::NonSolidShadowPaint,
        ))
    );
    assert!(error.message().contains("non-solid shadow paint"));
}

#[test]
fn unsupported_shadow_shapes_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::try_ellipse(Point::new(4.0, 4.0), Size::new(2.0, 1.0)).unwrap(),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("ellipse shadows should remain unsupported in this milestone");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ))
    );
    assert!(error.message().contains("ellipse/path shadow shape"));
}

#[test]
fn unsupported_path_shadows_report_typed_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0)).unwrap();
    let mut path = Path::new();
    path.move_to(Point::new(1.0, 1.0))
        .line_to(Point::new(6.0, 1.0))
        .line_to(Point::new(6.0, 6.0))
        .close();
    let mut scene = Scene::new();
    scene.shadow(
        Shape::path(path),
        Shadow::try_new(Point::new(0.0, 0.0), 1.0, 0.0, Color::BLACK).unwrap(),
    );

    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("path shadows should remain unsupported in this milestone");

    assert_eq!(error.code(), ErrorCode::UnsupportedPrimitive);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::Shadows,
            PrimitiveOperation::EllipsePathShadowShape,
        ))
    );
    assert!(error.message().contains("ellipse/path shadow shape"));
}

#[test]
fn headless_render_can_be_read_back() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let image = renderer.read_headless(&surface).unwrap();

    assert_eq!(image.size, PhysicalSize::new(4, 4));
    assert_eq!(image.rgba.len(), 4 * 4 * 4);
    assert!(image.rgba.iter().any(|channel| *channel != 0));
}

#[derive(Clone, Copy, Debug)]
struct PinnedVelloCharacterizationCase {
    antialiasing: Antialiasing,
    scale: f64,
    logical_dimensions: [u32; 2],
    physical_origin: [u32; 2],
    physical_dimensions: [u32; 2],
    solid_fill: [u8; 4],
    stroke: [u8; 4],
    gradient_left: [u8; 4],
    gradient_right: [u8; 4],
    image_top_left: [u8; 4],
    image_top_right: [u8; 4],
    clip_inside: [u8; 4],
    clip_excluded: [u8; 4],
    transformed_inside: [u8; 4],
    transformed_excluded: [u8; 4],
    ahem_ascent_ink: [u8; 4],
    ahem_descent_ink: [u8; 4],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
}

#[derive(Clone, Copy, Debug)]
struct AlphaSupport {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    centroid_x_hundredths: i32,
    centroid_y_hundredths: i32,
}

#[derive(Clone, Copy, Debug)]
struct PinnedVelloVariation {
    physical_dimensions: [u32; 2],
    stroke_alpha: u8,
    gradient_left: [u8; 2],
    gradient_right: [u8; 2],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
}

// Each row is `{AA, scale, physical dimensions, stroke alpha, gradient left/right,
// solid edge support, stroke edge support}`. Other samples are stable across all rows.
const PINNED_VELLO_CHARACTERIZATION_CASES: &[PinnedVelloCharacterizationCase] = &[
    pinned_vello_case(
        Antialiasing::Area,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Area,
        1.25,
        variation(
            [90, 60],
            175,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 731, 731),
            edge(15, 0, 30, 15, 2293, 793),
            edge(67, 21, 77, 29, 7199, 2511),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Area,
        2.0,
        variation(
            [144, 96],
            127,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3699, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa8,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa8,
        1.25,
        variation(
            [90, 60],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 737, 730),
            edge(16, 1, 30, 15, 2299, 796),
            edge(67, 21, 77, 29, 7200, 2511),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa8,
        2.0,
        variation(
            [144, 96],
            128,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3700, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa16,
        1.0,
        variation(
            [72, 48],
            191,
            [223, 32],
            [32, 223],
            edge(2, 2, 10, 10, 575, 575),
            edge(12, 0, 24, 12, 1824, 624),
            edge(54, 17, 61, 23, 5750, 2000),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa16,
        1.25,
        variation(
            [90, 60],
            175,
            [223, 32],
            [32, 223],
            edge(2, 2, 12, 12, 731, 731),
            edge(15, 0, 30, 15, 2293, 794),
            edge(67, 21, 77, 29, 7200, 2511),
        ),
    ),
    pinned_vello_case(
        Antialiasing::Msaa16,
        2.0,
        variation(
            [144, 96],
            128,
            [215, 40],
            [24, 231],
            edge(4, 4, 20, 20, 1200, 1200),
            edge(25, 1, 49, 25, 3700, 1300),
            edge(108, 34, 123, 47, 11550, 4050),
        ),
    ),
];

const fn pinned_vello_case(
    antialiasing: Antialiasing,
    scale: f64,
    variation: PinnedVelloVariation,
) -> PinnedVelloCharacterizationCase {
    PinnedVelloCharacterizationCase {
        antialiasing,
        scale,
        logical_dimensions: [72, 48],
        physical_origin: [0, 0],
        physical_dimensions: variation.physical_dimensions,
        solid_fill: [203, 52, 26, 128],
        stroke: [26, 64, 230, variation.stroke_alpha],
        gradient_left: [
            variation.gradient_left[0],
            0,
            variation.gradient_left[1],
            255,
        ],
        gradient_right: [
            variation.gradient_right[0],
            0,
            variation.gradient_right[1],
            255,
        ],
        image_top_left: [255, 0, 0, 255],
        image_top_right: [0, 255, 0, 255],
        clip_inside: [255, 255, 0, 255],
        clip_excluded: [0, 0, 0, 0],
        transformed_inside: [0, 255, 255, 255],
        transformed_excluded: [0, 0, 0, 0],
        ahem_ascent_ink: [0, 0, 0, 255],
        ahem_descent_ink: [0, 0, 0, 255],
        solid_edge: variation.solid_edge,
        stroke_edge: variation.stroke_edge,
        transformed_placement: variation.transformed_placement,
    }
}

const fn variation(
    physical_dimensions: [u32; 2],
    stroke_alpha: u8,
    gradient_left: [u8; 2],
    gradient_right: [u8; 2],
    solid_edge: AlphaSupport,
    stroke_edge: AlphaSupport,
    transformed_placement: AlphaSupport,
) -> PinnedVelloVariation {
    PinnedVelloVariation {
        physical_dimensions,
        stroke_alpha,
        gradient_left,
        gradient_right,
        solid_edge,
        stroke_edge,
        transformed_placement,
    }
}

const fn edge(
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    centroid_x_hundredths: i32,
    centroid_y_hundredths: i32,
) -> AlphaSupport {
    AlphaSupport {
        min_x,
        min_y,
        max_x,
        max_y,
        centroid_x_hundredths,
        centroid_y_hundredths,
    }
}

#[test]
fn pinned_vello_characterization_cases_are_source_readable() {
    let configurations = [
        (Antialiasing::Area, 1.0),
        (Antialiasing::Area, 1.25),
        (Antialiasing::Area, 2.0),
        (Antialiasing::Msaa8, 1.0),
        (Antialiasing::Msaa8, 1.25),
        (Antialiasing::Msaa8, 2.0),
        (Antialiasing::Msaa16, 1.0),
        (Antialiasing::Msaa16, 1.25),
        (Antialiasing::Msaa16, 2.0),
    ];
    let mut observed = Vec::with_capacity(configurations.len());

    for (antialiasing, scale) in configurations {
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default().with_antialiasing(antialiasing),
        ))
        .expect("pinned Vello characterization requires a host adapter");
        let scene = pinned_vello_characterization_scene();
        let mut surface =
            pollster::block_on(renderer.create_headless(Size::new(72.0, 48.0), scale))
                .expect("pinned Vello characterization requires a real headless surface");
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("pinned Vello characterization must render through the production Vello route");
        let output = renderer
            .read_headless(&surface)
            .expect("pinned Vello characterization must read the rendered headless surface");
        observed.push(observe_pinned_vello_characterization(
            antialiasing,
            &surface,
            &output,
        ));
    }

    assert_eq!(
        observed.len(),
        PINNED_VELLO_CHARACTERIZATION_CASES.len(),
        "missing source-readable pinned Vello samples; observed rows: {observed:#?}"
    );
    assert_eq!(
        PINNED_VELLO_CHARACTERIZATION_CASES.len(),
        configurations.len(),
        "the pinned table must cover every AA/scale Cartesian pair"
    );
    for (actual, expected) in observed.iter().zip(PINNED_VELLO_CHARACTERIZATION_CASES) {
        assert_pinned_vello_characterization_case(*actual, *expected);
    }
}

fn pinned_vello_characterization_scene() -> Scene {
    let partial_red = Color::try_rgba(0.8, 0.2, 0.1, 0.5).unwrap();
    let blue = Color::try_rgba(0.1, 0.25, 0.9, 1.0).unwrap();
    let gradient = Gradient::try_linear(
        Point::new(2.0, 16.0),
        Point::new(18.0, 16.0),
        vec![
            GradientStop::try_new(0.0, Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap()).unwrap(),
            GradientStop::try_new(1.0, Color::try_rgba(0.0, 0.0, 1.0, 1.0).unwrap()).unwrap(),
        ],
    )
    .unwrap();
    let image = Image::from_rgba(
        Size::new(2.0, 2.0),
        Arc::<[u8]>::from([
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ]),
    )
    .unwrap();
    let glyphs = [
        TextGlyph::try_new(AHEM_GLYPH_ASCENT_E_ACUTE, 2.0, 38.0, 10.0).unwrap(),
        TextGlyph::try_new(AHEM_GLYPH_DESCENT_P, 14.0, 38.0, 10.0).unwrap(),
    ];
    let mut scene = Scene::new();

    scene.fill(Rect::new(2.25, 2.25, 8.0, 8.0), partial_red);
    scene.stroke(
        Rect::new(14.25, 2.25, 9.0, 9.0),
        Stroke::try_new(3.0).unwrap(),
        blue,
    );
    scene.fill(Rect::new(2.0, 16.0, 16.0, 8.0), Paint::gradient(gradient));
    scene.image(image, Rect::new(22.0, 16.0, 8.0, 8.0), ImageFit::Stretch);
    scene.clip(Rect::new(36.0, 16.0, 6.0, 8.0), |scene| {
        scene.fill(
            Rect::new(32.0, 14.0, 14.0, 12.0),
            Color::try_rgba(1.0, 1.0, 0.0, 1.0).unwrap(),
        );
    });
    scene.transform(Transform::translation(6.0, 3.0).unwrap(), |scene| {
        scene.fill(
            Rect::new(48.0, 14.0, 8.0, 7.0),
            Color::try_rgba(0.0, 1.0, 1.0, 1.0).unwrap(),
        );
    });
    scene.text_run(
        TextRun::try_new(
            ahem_font("C03 pinned Vello characterization"),
            10.0,
            Transform::identity(),
            TextPaint::try_fill(Color::BLACK.into()).unwrap(),
            &glyphs,
            TextRunBounds::unspecified(),
        )
        .unwrap(),
    );
    scene
}

fn observe_pinned_vello_characterization(
    antialiasing: Antialiasing,
    surface: &Surface,
    image: &ImageBuffer,
) -> PinnedVelloCharacterizationCase {
    let logical_size = surface.size();
    let scale = surface.scale();
    let surface_physical_size = surface.physical_size();
    let frame_bounds = Rect::new(0.0, 0.0, logical_size.width(), logical_size.height());
    let physical_origin = [
        (frame_bounds.x() * scale).floor() as u32,
        (frame_bounds.y() * scale).floor() as u32,
    ];
    let physical_dimensions = [image.size.width(), image.size.height()];

    assert_eq!(
        physical_dimensions,
        [
            surface_physical_size.width(),
            surface_physical_size.height()
        ],
        "headless image dimensions must match the created surface"
    );

    PinnedVelloCharacterizationCase {
        antialiasing,
        scale,
        logical_dimensions: [logical_size.width() as u32, logical_size.height() as u32],
        physical_origin,
        physical_dimensions,
        solid_fill: characterization_pixel(image, scale, 5.0, 5.0),
        stroke: characterization_pixel(image, scale, 15.0, 5.0),
        gradient_left: characterization_pixel(image, scale, 4.0, 20.0),
        gradient_right: characterization_pixel(image, scale, 16.0, 20.0),
        image_top_left: characterization_pixel(image, scale, 23.0, 17.0),
        image_top_right: characterization_pixel(image, scale, 28.0, 17.0),
        clip_inside: characterization_pixel(image, scale, 38.0, 20.0),
        clip_excluded: characterization_pixel(image, scale, 33.0, 20.0),
        transformed_inside: characterization_pixel(image, scale, 56.0, 20.0),
        transformed_excluded: characterization_pixel(image, scale, 50.0, 20.0),
        ahem_ascent_ink: characterization_pixel(image, scale, 7.0, 34.0),
        ahem_descent_ink: characterization_pixel(image, scale, 19.0, 39.0),
        solid_edge: characterization_alpha_support(image, scale, 1.0, 1.0, 11.0, 11.0),
        stroke_edge: characterization_alpha_support(image, scale, 12.0, 0.0, 25.0, 13.0),
        transformed_placement: characterization_alpha_support(image, scale, 54.0, 17.0, 8.0, 7.0),
    }
}

fn characterization_pixel(image: &ImageBuffer, scale: f64, x: f64, y: f64) -> [u8; 4] {
    let x = ((x + 0.5) * scale).floor() as u32;
    let y = ((y + 0.5) * scale).floor() as u32;
    pixel_rgba(image, x, y)
}

fn characterization_alpha_support(
    image: &ImageBuffer,
    scale: f64,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> AlphaSupport {
    let x_start = (x * scale).floor() as u32;
    let y_start = (y * scale).floor() as u32;
    let x_end = ((x + width) * scale).ceil() as u32;
    let y_end = ((y + height) * scale).ceil() as u32;
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut alpha_sum = 0_u64;
    let mut weighted_x = 0_u64;
    let mut weighted_y = 0_u64;

    for pixel_y in y_start..y_end {
        for pixel_x in x_start..x_end {
            let alpha = u64::from(pixel_alpha(image, pixel_x, pixel_y));
            if alpha == 0 {
                continue;
            }
            min_x = min_x.min(pixel_x);
            min_y = min_y.min(pixel_y);
            max_x = max_x.max(pixel_x);
            max_y = max_y.max(pixel_y);
            alpha_sum += alpha;
            weighted_x += alpha * u64::from(pixel_x);
            weighted_y += alpha * u64::from(pixel_y);
        }
    }

    assert!(
        alpha_sum > 0,
        "characterization edge region must contain ink"
    );
    AlphaSupport {
        min_x,
        min_y,
        max_x,
        max_y,
        centroid_x_hundredths: ((weighted_x * 100) / alpha_sum) as i32,
        centroid_y_hundredths: ((weighted_y * 100) / alpha_sum) as i32,
    }
}

fn assert_pinned_vello_characterization_case(
    actual: PinnedVelloCharacterizationCase,
    expected: PinnedVelloCharacterizationCase,
) {
    assert_eq!(actual.antialiasing, expected.antialiasing);
    assert_eq!(actual.scale, expected.scale);
    assert_eq!(actual.logical_dimensions, expected.logical_dimensions);
    assert_eq!(actual.physical_origin, expected.physical_origin);
    assert_eq!(actual.physical_dimensions, expected.physical_dimensions);

    assert_partial_alpha_straight_rgba8(
        actual.solid_fill,
        expected.solid_fill,
        "partial-alpha solid fill",
    );

    for (name, actual, expected) in [
        ("stroke", actual.stroke, expected.stroke),
        (
            "gradient left",
            actual.gradient_left,
            expected.gradient_left,
        ),
        (
            "gradient right",
            actual.gradient_right,
            expected.gradient_right,
        ),
        (
            "image top left",
            actual.image_top_left,
            expected.image_top_left,
        ),
        (
            "image top right",
            actual.image_top_right,
            expected.image_top_right,
        ),
        ("clip inside", actual.clip_inside, expected.clip_inside),
        (
            "transformed inside",
            actual.transformed_inside,
            expected.transformed_inside,
        ),
        (
            "Ahem ascent ink",
            actual.ahem_ascent_ink,
            expected.ahem_ascent_ink,
        ),
        (
            "Ahem descent ink",
            actual.ahem_descent_ink,
            expected.ahem_descent_ink,
        ),
    ] {
        assert_rgba_within(actual, expected, 2, name);
    }

    assert_eq!(actual.clip_excluded, [0, 0, 0, 0]);
    assert_eq!(actual.transformed_excluded, [0, 0, 0, 0]);
    assert!(
        actual.ahem_ascent_ink[3] > 0,
        "Ahem ascent sample must contain ink"
    );
    assert!(
        actual.ahem_descent_ink[3] > 0,
        "Ahem descent sample must contain ink"
    );
    assert_alpha_support_within(actual.solid_edge, expected.solid_edge, "solid fill edge");
    assert_alpha_support_within(actual.stroke_edge, expected.stroke_edge, "stroke edge");
    assert_transformed_placement_within(
        actual.transformed_placement,
        expected.transformed_placement,
    );
    assert!(actual.gradient_left[0] > actual.gradient_left[2]);
    assert!(actual.gradient_right[2] > actual.gradient_right[0]);
    assert!(actual.image_top_left[0] > actual.image_top_left[1]);
    assert!(actual.image_top_right[1] > actual.image_top_right[0]);
}

fn assert_partial_alpha_straight_rgba8(actual: [u8; 4], expected: [u8; 4], name: &str) {
    assert_rgba_within(actual, expected, 2, name);
    assert!(
        actual[3] > 0 && actual[3] < u8::MAX,
        "{name} must remain partially transparent: {actual:?}"
    );
    assert!(
        actual[0] > actual[3],
        "{name} must retain its straight red channel above alpha: {actual:?}"
    );

    let premultiplied = [
        ((u16::from(actual[0]) * u16::from(actual[3]) + 127) / 255) as u8,
        ((u16::from(actual[1]) * u16::from(actual[3]) + 127) / 255) as u8,
        ((u16::from(actual[2]) * u16::from(actual[3]) + 127) / 255) as u8,
        actual[3],
    ];
    assert!(
        premultiplied[0] <= premultiplied[3] && actual[0].abs_diff(premultiplied[0]) >= 32,
        "{name} must differ materially from its premultiplied representation: {actual:?} -> {premultiplied:?}"
    );
}

fn assert_rgba_within(actual: [u8; 4], expected: [u8; 4], tolerance: u8, name: &str) {
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "{name} channel {channel} expected {expected} +/- {tolerance}, got {actual}"
        );
    }
}

fn assert_alpha_support_within(actual: AlphaSupport, expected: AlphaSupport, name: &str) {
    for (component, actual, expected) in [
        ("min_x", actual.min_x, expected.min_x),
        ("min_y", actual.min_y, expected.min_y),
        ("max_x", actual.max_x, expected.max_x),
        ("max_y", actual.max_y, expected.max_y),
    ] {
        assert!(
            actual.abs_diff(expected) <= 1,
            "{name} nonzero support {component} expected {expected} +/- 1, got {actual}"
        );
    }
    assert!(
        (actual.centroid_x_hundredths - expected.centroid_x_hundredths).abs() <= 35,
        "{name} centroid x exceeds the S34 0.35-device-pixel tolerance"
    );
    assert!(
        (actual.centroid_y_hundredths - expected.centroid_y_hundredths).abs() <= 35,
        "{name} centroid y exceeds the S34 0.35-device-pixel tolerance"
    );
}

fn assert_transformed_placement_within(actual: AlphaSupport, expected: AlphaSupport) {
    for (component, actual, expected) in [
        ("min_x", actual.min_x, expected.min_x),
        ("min_y", actual.min_y, expected.min_y),
        ("max_x", actual.max_x, expected.max_x),
        ("max_y", actual.max_y, expected.max_y),
    ] {
        assert!(
            actual.abs_diff(expected) <= 1,
            "transformed rectangle nonzero support {component} expected {expected} +/- 1, got {actual}"
        );
    }
    assert!(
        (actual.centroid_x_hundredths - expected.centroid_x_hundredths).abs() <= 35,
        "transformed rectangle centroid x exceeds the S34 0.35-device-pixel tolerance"
    );
    assert!(
        (actual.centroid_y_hundredths - expected.centroid_y_hundredths).abs() <= 35,
        "transformed rectangle centroid y exceeds the S34 0.35-device-pixel tolerance"
    );
}

fn render_scene_to_headless_or_skip_no_adapter(scene: &Scene, size: Size) -> Option<ImageBuffer> {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = pollster::block_on(renderer.create_headless(size, 1.0)).unwrap();
    match pollster::block_on(renderer.render(&mut surface, scene, Parameters::default())) {
        Ok(_) => {}
        Err(error) if error.code() == ErrorCode::AdapterUnavailable => {
            assert!(
                error.message().contains("backdrop") || error.message().contains("offscreen"),
                "adapter diagnostic should name the offscreen backdrop boundary: {}",
                error.message()
            );
            return None;
        }
        Err(error) => panic!("render failed unexpectedly: {error:?}"),
    }
    match renderer.read_headless(&surface) {
        Ok(output) => Some(output),
        Err(error)
            if matches!(
                error.code(),
                ErrorCode::AdapterUnavailable | ErrorCode::UnsupportedBackend
            ) =>
        {
            assert!(
                error.message().contains("headless") || error.message().contains("readback"),
                "readback diagnostic should name the headless readback boundary: {}",
                error.message()
            );
            None
        }
        Err(error) => panic!("headless readback failed unexpectedly: {error:?}"),
    }
}

fn render_scene_pixel(renderer: &mut Renderer, scene: &Scene) -> [u8; 4] {
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    pollster::block_on(renderer.render(&mut surface, scene, Parameters::default()))
        .expect("single-pixel blend scene should render through the direct Vello path");
    let output = renderer.read_headless(&surface).unwrap();
    pixel_rgba(&output, 0, 0)
}

fn color_from_opaque_rgba8(pixel: PremultipliedRgba8) -> Color {
    assert_eq!(
        pixel.alpha(),
        u8::MAX,
        "test helper only accepts opaque straight-compatible pixels"
    );
    Color::try_rgba(
        f32::from(pixel.red()) / 255.0,
        f32::from(pixel.green()) / 255.0,
        f32::from(pixel.blue()) / 255.0,
        1.0,
    )
    .unwrap()
}

fn assert_rgba_near_reference_pixel(
    actual: [u8; 4],
    expected: PremultipliedRgba8,
    tolerance: u8,
    message: &str,
) {
    let expected = [
        expected.red(),
        expected.green(),
        expected.blue(),
        expected.alpha(),
    ];
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let delta = actual.abs_diff(expected);
        assert!(
            delta <= tolerance,
            "{message}: channel {channel} expected {expected} +/- {tolerance}, got {actual}"
        );
    }
}

fn pixel_alpha(image: &ImageBuffer, x: u32, y: u32) -> u8 {
    pixel_rgba(image, x, y)[3]
}

fn pixel_rgba(image: &ImageBuffer, x: u32, y: u32) -> [u8; 4] {
    let index = ((y * image.size.width() + x) * 4 + 3) as usize;
    [
        image.rgba[index - 3],
        image.rgba[index - 2],
        image.rgba[index - 1],
        image.rgba[index],
    ]
}

fn assert_finite_positive_rect(rect: Rect) {
    assert!(rect.x().is_finite());
    assert!(rect.y().is_finite());
    assert!(rect.width().is_finite());
    assert!(rect.height().is_finite());
    assert!(rect.width() > 0.0);
    assert!(rect.height() > 0.0);
}
