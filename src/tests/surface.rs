use crate::{
    Attachment, Color, DeviceLossReason, EffectQualityPolicy, Error, ErrorCode, Format,
    GpuFaultKind, Image, ImageBuffer, ImageFit, ImageId, Options, Parameters, PhysicalSize,
    PresentMode, Rect, RenderRoute, RenderSurfaceAvailability, Renderer, ResourceCacheBudget,
    Result, RuntimeCapabilityUnavailable, RuntimeCapabilityUnavailableReason, RuntimeOperation,
    Scene, Size, Stats, Surface, SurfaceOptions, SurfaceResourceState, SurfaceState,
};

use crate::{
    backend::DeviceSignal,
    error::BackendErrorCode,
    gpu_transaction::{
        GpuOperationStage,
        test_support::{
            graph_accounting_failure_after_submission_for_test,
            graph_cancellation_after_submission_for_test,
            graph_scope_failure_after_submission_for_test, submit_readback_observed_for_test,
        },
    },
    resource::{ResourceAccountingFault, ResourceManager, WorkingFormat},
    surface::{HeadlessResources, SurfaceBackend},
};

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
use crate::surface::{
    PresentedLifecycle, PresentedResumeAction, PresentedSurfaceState, ResizeState,
};

#[cfg(feature = "render-window")]
use crate::{
    RuntimeCapabilities,
    backend::{
        DisplayFreePresentedDeviceCompatibilityForTest,
        configured_display_free_presented_surface_for_test,
        configured_display_free_presented_surface_on_device_for_test,
        discard_presented_configuration_stage_for_test, display_free_presented_surface_for_test,
        presented_configuration_validation_failure_stage_for_test,
        presented_device_identity_for_test, presented_lifecycle_for_test,
        presented_observation_for_test, presented_observation_handle_for_test,
        presented_resource_id_for_test, presented_target_identity_for_test,
        select_display_free_presented_device_for_test, set_presented_acquire_outcome_for_test,
        take_last_presented_texture_for_test,
    },
    gpu_transaction::test_support::graph_terminal_loss_after_submission_for_test,
    surface::PresentedAcquireOutcomeForTest,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::readback::{
    NativeReadbackLateCallbackStageForTest, NativeReadbackStageForTest,
    NativeReadbackStagePhaseForTest,
};

use std::{
    future::Future,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
    time::Duration,
};

#[cfg(not(target_arch = "wasm32"))]
use std::{
    pin::Pin,
    sync::{Condvar, Mutex},
    time::Instant,
};

use super::support::{
    assert_runtime_device_lost, assert_surface_unavailable, default_graph_working_format_for_test,
    explicit_graph_transaction_inputs_for_test, headless_direct_publication_fixture_for_test,
    modeled_resource_key_for_test, prepared_direct_vello_pass_for_test,
    repeated_graph_scene_for_test,
};

#[cfg(feature = "render-window")]
use super::support::{
    color_from_straight_rgba8_for_test, composition_presented_masked_blended_scene_for_test,
};

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
    let mut state = PresentedSurfaceState::new(PhysicalSize::new(0, 0), ResizeState::Resizing);
    state.resize_requested(
        Some(PhysicalSize::new(640, 480)),
        PhysicalSize::new(640, 480),
    );

    assert_eq!(
        state.lifecycle(),
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
            resources: HeadlessResources::Pending,
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
        surface.resume(Attachment::Headless).unwrap();

        let headless = renderer
            .create_headless(Size::new(1.0, 1.0), 1.0)
            .await
            .unwrap();
        let _: Result<ImageBuffer> = renderer.read_headless(&headless).await;
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
fn readback_transaction_maps_validation_internal_oom_and_terminal_failures() {
    use crate::gpu_transaction::ReadbackSubmission;

    let _transaction_result_contract: Option<ReadbackSubmission> = None;
    for fault in [GpuFaultKind::Validation, GpuFaultKind::Internal] {
        let error = GpuOperationStage::Readback
            .classify_fault_for_test(fault, "injected readback GPU error");
        assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    }
    assert_eq!(
        GpuOperationStage::Readback
            .classify_fault_for_test(GpuFaultKind::OutOfMemory, "injected readback OOM")
            .code(),
        ErrorCode::SurfaceOutOfMemory
    );
    assert_eq!(
        Error::new(BackendErrorCode::ReadbackFailed, "readback failed").code(),
        ErrorCode::ReadbackFailed
    );

    let lost_signal = DeviceSignal::new_for_test();
    lost_signal.record_loss_for_test(DeviceLossReason::Destroyed);
    let lost = lost_signal
        .first_terminal()
        .expect("the injected readback loss must be terminal")
        .error(RuntimeOperation::SurfaceReadback);
    assert_eq!(
        lost.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::DeviceLost {
                    reason: DeviceLossReason::Destroyed,
                },
            )
            .unwrap()
        )
    );

    let faulted_signal = DeviceSignal::new_for_test();
    faulted_signal.record_uncaptured_fault_for_test(
        GpuFaultKind::Internal,
        "injected terminal readback fault",
    );
    let faulted = faulted_signal
        .first_terminal()
        .expect("the injected readback fault must be terminal")
        .error(RuntimeOperation::SurfaceReadback);
    assert_eq!(
        faulted.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::DeviceFaulted {
                    kind: GpuFaultKind::Internal,
                },
            )
            .unwrap()
        )
    );

    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("readback transaction coverage requires a host adapter");
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(1.0, 1.0), 1.0)).unwrap();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the readback transaction fixture must publish a headless texture");
    let output = pollster::block_on(renderer.read_headless(&surface))
        .expect("the scoped readback copy must complete");
    assert_eq!(output.size(), PhysicalSize::new(1, 1));

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let generation = signal.next_test_generation().unwrap();
    let transaction = crate::gpu_transaction::GpuOperationTransaction::begin(
        &device,
        Arc::clone(&signal),
        generation,
        GpuOperationStage::Readback,
    );
    let command_buffer = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surgeist explicit readback transaction observation"),
        })
        .finish();
    let submission = pollster::block_on(submit_readback_observed_for_test(
        transaction,
        &queue,
        command_buffer,
        RuntimeOperation::SurfaceReadback,
    ))
    .expect("the explicit readback transaction must resolve its real scopes");
    assert_eq!(submission.queue_submission_count_for_test(), 1);
    assert_eq!(
        submission.transaction_generation_for_test(),
        submission.active_generation_for_test(),
        "the readback copy must submit while its transaction generation is active"
    );
    assert!(
        submission.scopes_resolved_for_test(),
        "the readback copy must resolve its scopes before completing"
    );
    let submission_index = submission.submission_index_for_test();
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(Duration::from_secs(2)),
        })
        .expect("the retained readback submission index must name the completed copy");
}

#[test]
fn readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers() {
    assert_readback_submission_index_retained();
    assert_readback_idle_cleanup();
    assert_readback_pending_cleanup();
    assert_readback_callback_cleanup();
    assert_readback_mapped_completion();
}

use crate::readback::{
    ReadbackCleanupEventForTest as Cleanup, ReadbackPhaseForTest,
    ReadbackStagingDispositionForTest as StagingDisposition, ReadbackStateMachineForTest,
};

fn readback_state_at(phase: ReadbackPhaseForTest) -> ReadbackStateMachineForTest {
    let mut state = ReadbackStateMachineForTest::allocated();
    match phase {
        ReadbackPhaseForTest::Allocated => {}
        ReadbackPhaseForTest::CopySubmitted { submission_index } => {
            state.copy_submitted_for_test(submission_index);
        }
        ReadbackPhaseForTest::MapPending => {
            state.copy_submitted_for_test(17);
            state.map_pending_for_test();
        }
        ReadbackPhaseForTest::Mapped => {
            state.copy_submitted_for_test(17);
            state.map_pending_for_test();
            state.map_callback_succeeded_for_test();
            state.mapped_for_test();
        }
        ReadbackPhaseForTest::PublishedBytes
        | ReadbackPhaseForTest::Failed
        | ReadbackPhaseForTest::Canceled => {
            panic!("the fixture accepts only uncertain readback phases")
        }
    }
    state
}

fn assert_readback_submission_index_retained() {
    let submitted = readback_state_at(ReadbackPhaseForTest::CopySubmitted {
        submission_index: 91,
    });
    assert_eq!(
        submitted.phase_for_test(),
        ReadbackPhaseForTest::CopySubmitted {
            submission_index: 91,
        },
        "the owner must retain the exact queue submission index"
    );
}

fn assert_readback_idle_cleanup() {
    for idle_phase in [
        ReadbackPhaseForTest::Allocated,
        ReadbackPhaseForTest::CopySubmitted {
            submission_index: 17,
        },
    ] {
        let mut failed = readback_state_at(idle_phase);
        failed.fail_for_test();
        assert_eq!(failed.phase_for_test(), ReadbackPhaseForTest::Failed);
        assert_eq!(
            failed.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "pre-map failure must drop idle staging without invalid unmap"
        );
        assert_eq!(
            failed.staging_disposition_for_test(),
            StagingDisposition::Released
        );
        failed.cancel_for_test();
        assert_eq!(
            failed.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "terminal cleanup must consume staging ownership exactly once"
        );
        assert_eq!(
            failed.staging_disposition_for_test(),
            StagingDisposition::Released
        );

        let mut canceled = readback_state_at(idle_phase);
        canceled.cancel_for_test();
        assert_eq!(canceled.phase_for_test(), ReadbackPhaseForTest::Canceled);
        assert_eq!(
            canceled.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "pre-map cancellation must drop idle staging without invalid unmap"
        );
        canceled.fail_for_test();
        assert_eq!(
            canceled.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "terminal cleanup must consume staging ownership exactly once"
        );
        assert_eq!(
            canceled.staging_disposition_for_test(),
            StagingDisposition::Released
        );
    }
}

fn assert_readback_pending_cleanup() {
    let mut pending_failure = readback_state_at(ReadbackPhaseForTest::MapPending);
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::MapPending
    );
    pending_failure.fail_for_test();
    assert_eq!(
        pending_failure.phase_for_test(),
        ReadbackPhaseForTest::Failed
    );
    assert_eq!(
        pending_failure.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "wrong-index or other pending-map failure must abort the request before dropping staging"
    );
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::Released
    );
    pending_failure.map_callback_succeeded_for_test();
    pending_failure.cancel_for_test();
    assert_eq!(
        pending_failure.staging_disposition_for_test(),
        StagingDisposition::Released,
        "a late callback cannot reacquire released staging"
    );
    assert_eq!(
        pending_failure.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "late delivery and second terminal cleanup cannot act on staging again"
    );

    let mut pending_cancellation = readback_state_at(ReadbackPhaseForTest::MapPending);
    pending_cancellation.cancel_for_test();
    assert_eq!(
        pending_cancellation.phase_for_test(),
        ReadbackPhaseForTest::Canceled
    );
    assert_eq!(
        pending_cancellation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "pending-map cancellation must abort the request before dropping staging"
    );
    assert_eq!(
        pending_cancellation.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

fn assert_readback_callback_cleanup() {
    for terminal_phase in [ReadbackPhaseForTest::Failed, ReadbackPhaseForTest::Canceled] {
        let mut callback_error = readback_state_at(ReadbackPhaseForTest::MapPending);
        callback_error.map_callback_failed_for_test();
        assert_eq!(
            callback_error.phase_for_test(),
            ReadbackPhaseForTest::MapPending,
            "callback delivery must not overwrite the lifecycle before the owner consumes it"
        );
        assert_eq!(
            callback_error.staging_disposition_for_test(),
            StagingDisposition::Idle,
            "a map callback error returns physical staging to known idle"
        );
        match terminal_phase {
            ReadbackPhaseForTest::Failed => callback_error.fail_for_test(),
            ReadbackPhaseForTest::Canceled => callback_error.cancel_for_test(),
            _ => unreachable!("the fixture selects only terminal cleanup phases"),
        }
        assert_eq!(callback_error.phase_for_test(), terminal_phase);
        assert_eq!(
            callback_error.cleanup_events_for_test(),
            vec![Cleanup::StagingDropped],
            "callback-error-idle cleanup must not call unmap"
        );
        assert_eq!(
            callback_error.staging_disposition_for_test(),
            StagingDisposition::Released
        );
    }

    let mut callback_success_cancellation = readback_state_at(ReadbackPhaseForTest::MapPending);
    callback_success_cancellation.map_callback_succeeded_for_test();
    assert_eq!(
        callback_success_cancellation.phase_for_test(),
        ReadbackPhaseForTest::MapPending
    );
    assert_eq!(
        callback_success_cancellation.staging_disposition_for_test(),
        StagingDisposition::MappedActive
    );
    callback_success_cancellation.cancel_for_test();
    assert_eq!(
        callback_success_cancellation.phase_for_test(),
        ReadbackPhaseForTest::Canceled
    );
    assert_eq!(
        callback_success_cancellation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped],
        "cancellation racing callback success must unmap active staging before drop"
    );

    let dropped = readback_state_at(ReadbackPhaseForTest::MapPending);
    let drop_observation = dropped.observation_for_test();
    drop(dropped);
    assert_eq!(
        drop_observation.terminal_phase_for_test(),
        Some(ReadbackPhaseForTest::Canceled)
    );
    assert_eq!(
        drop_observation.cleanup_events_for_test(),
        vec![Cleanup::StagingUnmapped, Cleanup::StagingDropped]
    );
    assert_eq!(
        drop_observation.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

fn assert_readback_mapped_completion() {
    let mut incomplete = readback_state_at(ReadbackPhaseForTest::Mapped);
    let error = incomplete
        .finish_mapped_for_test(PhysicalSize::new(1, 2), &[0; 256])
        .expect_err("a missing padded row must fail through checked decoding");
    assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    assert_eq!(incomplete.phase_for_test(), ReadbackPhaseForTest::Failed);
    assert_eq!(
        incomplete.staging_disposition_for_test(),
        StagingDisposition::Released
    );
    assert_eq!(
        incomplete.cleanup_events_for_test(),
        vec![
            Cleanup::MappedViewDropped,
            Cleanup::StagingUnmapped,
            Cleanup::StagingDropped,
        ],
        "the mapped view must drop before staging is unmapped"
    );

    let mut mapped = vec![0; 512];
    mapped[0..4].copy_from_slice(&[1, 2, 3, 4]);
    mapped[256..260].copy_from_slice(&[5, 6, 7, 8]);
    let mut published = readback_state_at(ReadbackPhaseForTest::Mapped);
    let image = published
        .finish_mapped_for_test(PhysicalSize::new(1, 2), &mapped)
        .expect("complete checked rows must publish one validated image");
    assert_eq!(image.rgba(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        published.phase_for_test(),
        ReadbackPhaseForTest::PublishedBytes
    );
    assert_eq!(
        published.cleanup_events_for_test(),
        vec![
            Cleanup::MappedViewDropped,
            Cleanup::StagingUnmapped,
            Cleanup::StagingDropped,
            Cleanup::PublishedBytes,
        ]
    );
    assert_eq!(
        published.staging_disposition_for_test(),
        StagingDisposition::Released
    );
}

#[test]
fn readback_map_callback_publishes_once_and_wakes_latest_waker() {
    use crate::readback::ReadbackCompletionForTest;

    struct WakeCount(AtomicUsize);

    impl std::task::Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let first_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
    let latest_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
    let first_waker = Waker::from(Arc::clone(&first_wakes));
    let latest_waker = Waker::from(Arc::clone(&latest_wakes));
    let completion = ReadbackCompletionForTest::new();
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&first_waker)),
        Poll::Pending
    ));
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&latest_waker)),
        Poll::Pending
    ));

    completion.invoke_map_callback_for_test(Ok(()));
    assert_eq!(first_wakes.0.load(Ordering::SeqCst), 0);
    assert_eq!(latest_wakes.0.load(Ordering::SeqCst), 1);
    completion.deliver_late_map_result_for_test(Err(wgpu::BufferAsyncError));
    assert_eq!(completion.accepted_result_count_for_test(), 1);
    assert_eq!(completion.discarded_result_count_for_test(), 1);
    assert!(matches!(
        completion.poll_for_test(&mut Context::from_waker(&latest_waker)),
        Poll::Ready(Ok(()))
    ));
    completion.deliver_late_map_result_for_test(Ok(()));
    assert_eq!(completion.accepted_result_count_for_test(), 1);
    assert_eq!(completion.discarded_result_count_for_test(), 2);

    let callback_error = ReadbackCompletionForTest::new();
    callback_error.invoke_map_callback_for_test(Err(wgpu::BufferAsyncError));
    let Poll::Ready(Err(error)) =
        callback_error.poll_for_test(&mut Context::from_waker(Waker::noop()))
    else {
        panic!("the callback error must be consumed exactly once")
    };
    assert_eq!(error.code(), ErrorCode::ReadbackFailed);
    assert!(std::error::Error::source(&error).is_some());

    let canceled = ReadbackCompletionForTest::new();
    canceled.cancel_for_test();
    canceled.deliver_late_map_result_for_test(Ok(()));
    assert!(canceled.is_canceled_for_test());
    assert_eq!(canceled.accepted_result_count_for_test(), 0);
    assert_eq!(canceled.discarded_result_count_for_test(), 1);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let poll_completion = ReadbackCompletionForTest::new();
        let poll_wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
        let poll_waker = Waker::from(Arc::clone(&poll_wakes));
        assert!(matches!(
            poll_completion.poll_for_test(&mut Context::from_waker(&poll_waker)),
            Poll::Pending
        ));
        assert!(poll_completion.timeout_slice_for_test());
        assert!(poll_completion.timeout_slice_for_test());
        assert_eq!(poll_completion.accepted_result_count_for_test(), 0);
        poll_completion.wrong_submission_index_for_test(9, 8);
        assert_eq!(poll_wakes.0.load(Ordering::SeqCst), 1);
        let Poll::Ready(Err(error)) =
            poll_completion.poll_for_test(&mut Context::from_waker(&poll_waker))
        else {
            panic!("a wrong submission index must terminate readback")
        };
        assert_eq!(error.code(), ErrorCode::ReadbackFailed);
        assert!(std::error::Error::source(&error).is_some());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct NativeReadbackDiagnosticDeadlineForTest {
    expires_at: Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeReadbackDiagnosticDeadlineForTest {
    fn begin() -> Self {
        Self {
            expires_at: Instant::now()
                .checked_add(Duration::from_secs(5))
                .expect("the native readback diagnostic deadline must be representable"),
        }
    }

    fn remaining(self) -> Option<Duration> {
        self.expires_at.checked_duration_since(Instant::now())
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeReadbackWakeConditionForTest {
    notified: Mutex<bool>,
    changed: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeReadbackWakeForTest {
    condition: Arc<NativeReadbackWakeConditionForTest>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeReadbackWakeForTest {
    fn fresh() -> Self {
        Self {
            condition: Arc::new(NativeReadbackWakeConditionForTest {
                notified: Mutex::new(false),
                changed: Condvar::new(),
            }),
        }
    }

    fn prepare_for_poll(&self) {
        *self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = false;
    }

    fn wait_for_wake(
        &self,
        deadline: NativeReadbackDiagnosticDeadlineForTest,
        stage: &NativeReadbackStageForTest,
        device_signal: &DeviceSignal,
    ) {
        let notified = self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(remaining) = deadline.remaining() else {
            panic!(
                "native readback diagnostic deadline expired: {}",
                native_readback_diagnostic_for_test(stage, device_signal)
            );
        };
        let (notified, timeout) = self
            .condition
            .changed
            .wait_timeout_while(notified, remaining, |notified| !*notified)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if timeout.timed_out() && !*notified {
            panic!(
                "native readback diagnostic deadline expired: {}",
                native_readback_diagnostic_for_test(stage, device_signal)
            );
        }
    }

    fn notify(&self) {
        let mut notified = self
            .condition
            .notified
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *notified = true;
        self.condition.changed.notify_all();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::task::Wake for NativeReadbackWakeForTest {
    fn wake(self: Arc<Self>) {
        self.notify();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn drive_native_readback_for_test(
    stage: &mut NativeReadbackStageForTest,
    deadline: NativeReadbackDiagnosticDeadlineForTest,
    device_signal: &Arc<DeviceSignal>,
) -> Result<ImageBuffer> {
    let wake = Arc::new(NativeReadbackWakeForTest::fresh());
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    loop {
        wake.prepare_for_poll();
        match Pin::new(&mut *stage).poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => {}
        }
        wake.wait_for_wake(deadline, stage, device_signal);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_readback_diagnostic_for_test(
    stage: &NativeReadbackStageForTest,
    device_signal: &DeviceSignal,
) -> String {
    format!(
        "stage_phase={:?}; staging_disposition={:?}; submission_index={:?}; device_active_generation={:?}; device_terminal_signal={:?}",
        stage.phase_for_test(),
        stage.staging_disposition_for_test(),
        stage.submission_index_for_test(),
        device_signal.active_generation_for_test(),
        device_signal.first_terminal(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn headless_publication_texture_for_test(surface: &Surface) -> wgpu::Texture {
    match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the real headless fixture must retain one readable publication"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_readback_callback_progresses_and_cleans_up_with_diagnostic_deadline() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "native callback progress coverage requires an available host adapter"
    );
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("the callback progress fixture must publish a real headless texture");

    let device_signal = renderer
        .default_device_signal_for_test()
        .expect("native callback progress requires a ready device signal");
    let texture = headless_publication_texture_for_test(&surface);
    let (device, queue) = renderer
        .default_wgpu_device_queue()
        .expect("native callback progress requires a ready device and queue");
    let device = device.clone();
    let queue = queue.clone();
    let mut progress =
        NativeReadbackStageForTest::begin(&device, &queue, &texture, PhysicalSize::new(4, 4))
            .expect("the explicit native map stage must start from a real submitted texture copy");
    assert_eq!(
        progress.phase_for_test(),
        NativeReadbackStagePhaseForTest::MapPending
    );
    let deadline = NativeReadbackDiagnosticDeadlineForTest::begin();
    let image = drive_native_readback_for_test(&mut progress, deadline, &device_signal)
        .expect("the native callback must progress the real publication readback");
    assert_eq!(
        progress.phase_for_test(),
        NativeReadbackStagePhaseForTest::PublishedBytes
    );
    assert_eq!(progress.staging_disposition_for_test(), None);
    assert!(progress.staging_state_dropped_for_test());
    assert_eq!(device_signal.active_generation_for_test(), None);
    assert!(device_signal.first_terminal().is_none());
    assert_eq!(image.size(), PhysicalSize::new(4, 4));
    assert!(image.rgba().iter().any(|channel| *channel != 0));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canceled_native_readback_discards_late_callback_without_publication_change() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert!(
        renderer.default_wgpu_device_queue().is_some(),
        "native cancellation coverage requires an available host adapter"
    );
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(
        Rect::new(0.0, 0.0, 4.0, 4.0),
        Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap(),
    );
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .expect("the cancellation fixture must publish a real headless texture");
    let pixels_before = pollster::block_on(renderer.read_headless(&surface))
        .expect("the cancellation fixture publication must be readable");
    let publication_before = headless_publication_texture_for_test(&surface);
    let stats_before = renderer.stats();
    let renderer_options_before = renderer.options();
    let uploaded_images_before = renderer.uploaded_images_for_test();
    let parameters_before = surface.last_parameters;
    let surface_state_before = surface.state();
    let resource_state_before = surface.resource_state();
    let physical_size_before = surface.physical_size();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the published fixture must retain its ready device resources")
        .internal_resource_manager_observation_for_test();
    let device_signal = renderer
        .default_device_signal_for_test()
        .expect("native cancellation requires a ready device signal");
    let (device, queue) = renderer
        .default_wgpu_device_queue()
        .expect("native cancellation requires a ready device and queue");
    let device = device.clone();
    let queue = queue.clone();

    let mut canceled_future = NativeReadbackStageForTest::begin(
        &device,
        &queue,
        &publication_before,
        physical_size_before,
    )
    .expect("the explicit native future stage must start from a real submitted texture copy");
    let canceled_submission = canceled_future.submission_index_for_test();
    canceled_future.cancel_for_test();
    assert_eq!(
        canceled_future.phase_for_test(),
        NativeReadbackStagePhaseForTest::Canceled
    );
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(canceled_submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("the canceled native future must release its helper and staging request");
    assert!(canceled_future.staging_state_dropped_for_test());

    let late_callback = NativeReadbackLateCallbackStageForTest::cancel_before_poll(
        &device,
        &queue,
        &publication_before,
        physical_size_before,
    )
    .expect("the late-callback stage must register a real map before cancellation");
    assert!(
        matches!(
            late_callback.staging_disposition_for_test(),
            Some(crate::readback::ReadbackStagingDispositionForTest::Released) | None
        ),
        "cancellation must release staging whether callback delivery is immediate or poll-driven"
    );
    late_callback
        .deliver_late_callback_for_test()
        .expect("native polling must deliver the real callback after cancellation");
    assert!(
        late_callback.callback_result_was_discarded_for_test(),
        "a real late callback must leave completion canceled and release its staging capture"
    );

    let pixels_after = pollster::block_on(renderer.read_headless(&surface))
        .expect("the preserved publication must remain readable after cancellation");

    let publication_after = headless_publication_texture_for_test(&surface);
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("canceled readback must retain the ready device resources")
        .internal_resource_manager_observation_for_test();
    assert_eq!(publication_after, publication_before);
    assert_eq!(pixels_after, pixels_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(renderer.options(), renderer_options_before);
    assert_eq!(renderer.uploaded_images_for_test(), uploaded_images_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(surface.state(), surface_state_before);
    assert_eq!(surface.resource_state(), resource_state_before);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(resources_after, resources_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    assert!(device_signal.first_terminal().is_none());
}

#[test]
fn surface_operation_matrix_covers_every_kind_state_and_duplicate_transition() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();

    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "a nonzero headless surface has no publication before its first render"
    );
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the first headless render should publish a readable texture");
    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);

    surface.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::Ready,
        "same-physical resize retains the current publication"
    );
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "a physical-size change invalidates the old publication"
    );

    surface.suspend().unwrap();
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
    surface.resume(Attachment::Headless).unwrap();
    surface.resume(Attachment::Headless).unwrap();
    assert_eq!(surface.state(), SurfaceState::Available);

    let error = pollster::block_on(renderer.resume_surface(&mut surface, Attachment::Headless))
        .expect_err("renderer resume is not the headless lifecycle operation");
    assert_eq!(error.code(), ErrorCode::UnsupportedBackend);
}

#[cfg(any(
    feature = "render-window",
    all(feature = "render-web", target_arch = "wasm32")
))]
#[test]
fn available_presented_resume_keeps_the_installed_attachment_without_recreating() {
    let action = Surface::presented_resume_action(
        SurfaceState::Available,
        PresentedLifecycle::Ready {
            resizing: ResizeState::Idle,
        },
    );

    assert!(
        matches!(action, PresentedResumeAction::NoOp),
        "an available presented surface must retain its attachment without WGPU recreation"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_setup_and_resize_commit_only_after_clean_configuration() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented configuration coverage requires a compatible device");
    let mut surface = display_free_presented_surface_for_test(
        &mut renderer,
        SurfaceOptions {
            size: Size::new(2.0, 2.0),
            ..SurfaceOptions::default()
        },
    );

    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(presented_resource_id_for_test(&surface), None);

    pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect("initial presented configuration must commit only after clean scopes");
    let initial_resource = presented_resource_id_for_test(&surface)
        .expect("clean configuration must commit one resource bundle");
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    let error = pollster::block_on(presented_configuration_validation_failure_stage_for_test(
        &mut renderer,
        &surface,
        RuntimeOperation::SurfaceRendering,
    ))
    .expect_err("a real Configure validation failure must leave the requested resize pending");
    assert_eq!(error.code(), ErrorCode::SurfaceConfigureFailed);
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));

    discard_presented_configuration_stage_for_test(&mut renderer, &surface)
        .expect("an explicit Configure draft must be discardable before publication");
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.configure_presented_surface_for_test(&mut surface))
        .expect_err("a terminal device must leave the pending configuration uncommitted");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(initial_resource)
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
}

#[cfg(feature = "render-window")]
#[test]
fn presented_acquire_outcomes_map_every_surface_result_before_commit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented acquire coverage requires a compatible device");
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };

    let mut success = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut success, PresentedAcquireOutcomeForTest::Success);
    let stats = pollster::block_on(renderer.render(&mut success, &Scene::new(), parameters))
        .expect("a successful acquire must present and publish the frame");
    assert_eq!(renderer.stats(), stats);
    assert_eq!(success.last_parameters, Some(parameters));
    assert_eq!(
        presented_observation_for_test(&success).present_count_for_test(),
        1,
        "a successful acquired texture must be presented exactly once"
    );

    for outcome in [
        PresentedAcquireOutcomeForTest::Suboptimal,
        PresentedAcquireOutcomeForTest::Outdated,
    ] {
        let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
        let stats_before = renderer.stats();
        let parameters_before = surface.last_parameters;
        let resource_before = presented_resource_id_for_test(&surface);
        set_presented_acquire_outcome_for_test(&mut surface, outcome);

        let error = pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
            .expect_err("suboptimal and outdated acquisition must retry configuration then fail");
        assert_eq!(error.code(), ErrorCode::SurfaceOutdated);
        assert_eq!(renderer.stats(), stats_before);
        assert_eq!(surface.last_parameters, parameters_before);
        assert!(matches!(
            presented_lifecycle_for_test(&surface),
            PresentedLifecycle::Ready { .. }
        ));
        assert_ne!(presented_resource_id_for_test(&surface), resource_before);
        let observation = presented_observation_for_test(&surface);
        assert_eq!(observation.present_count_for_test(), 0);
        assert_eq!(
            observation.discarded_count_for_test(),
            if outcome == PresentedAcquireOutcomeForTest::Suboptimal {
                1
            } else {
                0
            },
            "only an acquired suboptimal texture needs RAII discard"
        );
    }

    for outcome in [
        PresentedAcquireOutcomeForTest::Timeout,
        PresentedAcquireOutcomeForTest::Validation,
    ] {
        let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
        let stats_before = renderer.stats();
        set_presented_acquire_outcome_for_test(&mut surface, outcome);
        let error = pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
            .expect_err("failed acquire must not publish frame state");
        assert_eq!(
            error.code(),
            match outcome {
                PresentedAcquireOutcomeForTest::Timeout => ErrorCode::SurfaceTimeout,
                PresentedAcquireOutcomeForTest::Validation => ErrorCode::PresentFailed,
                _ => unreachable!(),
            }
        );
        assert_eq!(renderer.stats(), stats_before);
        assert_eq!(surface.last_parameters, None);
        assert_eq!(
            presented_observation_for_test(&surface).present_count_for_test(),
            0
        );
    }

    let mut occluded = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut occluded, PresentedAcquireOutcomeForTest::Occluded);
    let error = pollster::block_on(renderer.render(&mut occluded, &Scene::new(), parameters))
        .expect_err("occluded acquire must not report a successful frame");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&occluded),
        PresentedLifecycle::Occluded { .. }
    ));
    assert_eq!(occluded.last_parameters, None);

    let mut lost = configured_display_free_presented_surface_for_test(&mut renderer);
    set_presented_acquire_outcome_for_test(&mut lost, PresentedAcquireOutcomeForTest::Lost);
    let error = pollster::block_on(renderer.render(&mut lost, &Scene::new(), parameters))
        .expect_err("surface loss must not report a successful frame");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&lost),
        PresentedLifecycle::Lost
    ));
    assert!(renderer.default_device_has_no_terminal_signal_for_test());
}

#[cfg(feature = "render-window")]
#[test]
fn presented_blit_and_present_remain_scoped_until_frame_commit() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented transaction coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let stats_before = renderer.stats();
    let parameters = Parameters {
        base_color: Color::TRANSPARENT,
        debug: true,
    };

    let observation = presented_observation_handle_for_test(&surface);
    let scene = Scene::new();
    let stats = pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .expect("scoped present must publish only after transaction completion");
    let observation = observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats);
    assert_ne!(renderer.stats(), stats_before);
    assert_eq!(renderer.stats(), stats);
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_acquire_error_leaks_no_prepared_or_public_state() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::DISABLED),
    ))
    .expect("presented acquire-failure coverage requires a compatible device");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let stats_before = renderer.stats();
    let cache_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the configured surface must retain a ready device")
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the configured surface must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Timeout);
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);

    let error = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect_err("the injected acquire timeout must abort the prepared graph");

    assert_eq!(error.code(), ErrorCode::SurfaceTimeout);
    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 1);
    assert_eq!(presented.acquire_count_for_test(), 0);
    assert_eq!(presented.present_count_for_test(), 0);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(surface.last_parameters, None);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert_eq!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("an acquire timeout must retain the ready device")
            .device_pass_cache_counts_for_test(),
        cache_before
    );
    let resources_after = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("an acquire timeout must return every prepared lease")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after.leased_count, 0);
    assert_eq!(
        resources_after.retained_count_for_test(),
        resources_before.retained_count_for_test()
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_scope_failure_suppresses_presentation_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented scope-failure coverage requires a compatible device");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(910),
        &mut presentation_commit,
    ))
    .expect_err("scope failure after a real submit must abort the host-effect draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(presentation_commit, Some(1));
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_accounting_fault_before_authorization_suppresses_present_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented accounting-fault coverage requires a compatible device");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let mut presentation_commit = Some(1);
    let error = pollster::block_on(graph_accounting_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(911),
        &mut presentation_commit,
    ))
    .expect_err("accounting rejection after submit must abort the host-effect draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(presentation_commit, Some(1));
    let after_fault = resources.observation_for_test();
    let Some(ResourceAccountingFault::RetainedByteMismatch {
        retained_bytes,
        registered_entry_bytes,
    }) = after_fault.accounting_fault_for_test()
    else {
        panic!("the presented transaction must preserve the exact injected accounting fault");
    };
    assert_eq!(retained_bytes.checked_add(1), Some(registered_entry_bytes));
    assert_eq!(after_fault.active_frame_count, 0);
    assert_eq!(after_fault.leased_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_cancellation_after_submit_discards_without_presentation() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented cancellation coverage requires a compatible device");
    let surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let observation = presented_observation_handle_for_test(&surface);
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);

    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &resources,
            modeled_resource_key_for_test(913),
            &mut presentation_commit,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }

    assert_eq!(presentation_commit, Some(1));
    let canceled = observation.snapshot_for_test();
    assert_eq!(canceled.acquire_attempt_count_for_test(), 0);
    assert_eq!(canceled.acquire_count_for_test(), 0);
    assert_eq!(canceled.present_count_for_test(), 0);
    assert_eq!(canceled.discarded_count_for_test(), 0);
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[cfg(feature = "render-window")]
#[test]
fn presented_graph_terminal_loss_suppresses_presentation_and_transitions_device() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default().with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
    ))
    .expect("presented terminal-loss coverage requires a compatible device");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::DISABLED);
    let mut presentation_commit = Some(1);
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLACK);

    let error = pollster::block_on(graph_terminal_loss_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(912),
        &mut presentation_commit,
    ))
    .expect_err("terminal device loss after submit must suppress the host-effect draft");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::EffectRendering,
        DeviceLossReason::Destroyed,
    );
    assert_eq!(presentation_commit, Some(1));
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 0);
    assert_eq!(presented.acquire_count_for_test(), 0);
    assert_eq!(presented.present_count_for_test(), 0);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(surface.last_parameters, None);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert!(matches!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Destroyed
        })
    ));
    let repeated = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect_err("the terminal device generation must reject every later frame");
    assert_runtime_device_lost(
        repeated,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Destroyed,
    );
    assert_eq!(
        presented_observation_for_test(&surface),
        presented,
        "a terminal device generation must not reacquire or present"
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resize_preserves_lost_recovery_gate_for_same_and_changed_extents() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost-resize coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let committed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let committed_target = presented_target_identity_for_test(&surface);

    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("acquire loss must close the surface recovery gate");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );
    assert_eq!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Lost
    );

    let stats_before = renderer.stats();
    let parameters_before = surface.last_parameters;
    let observation_before = presented_observation_for_test(&surface);

    surface.resize(Size::new(1.0, 1.0), 2.0).unwrap();
    let same_physical_size = surface.physical_size();
    let same_lifecycle = presented_lifecycle_for_test(&surface);
    let same_capabilities = renderer.runtime_capabilities(&surface);
    let same_render = pollster::block_on(renderer.render(
        &mut surface,
        &Scene::new(),
        presented_black_debug_parameters_for_test(),
    ));
    let same_resource = presented_resource_id_for_test(&surface);
    let same_target = presented_target_identity_for_test(&surface);
    let same_observation = presented_observation_for_test(&surface);
    let same_stats = renderer.stats();
    let same_parameters = surface.last_parameters;
    let same_active_generation = renderer.default_device_active_operation_generation_for_test();

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    let changed_physical_size = surface.physical_size();
    let changed_lifecycle = presented_lifecycle_for_test(&surface);
    let changed_capabilities = renderer.runtime_capabilities(&surface);
    let changed_render = pollster::block_on(renderer.render(
        &mut surface,
        &Scene::new(),
        presented_black_debug_parameters_for_test(),
    ));
    let changed_resource = presented_resource_id_for_test(&surface);
    let changed_target = presented_target_identity_for_test(&surface);
    let changed_observation = presented_observation_for_test(&surface);
    let changed_stats = renderer.stats();
    let changed_parameters = surface.last_parameters;
    let changed_active_generation = renderer.default_device_active_operation_generation_for_test();

    assert_eq!(
        [same_lifecycle, changed_lifecycle],
        [PresentedLifecycle::Lost, PresentedLifecycle::Lost],
        "same- and changed-extent resize must not bypass explicit lost-surface recovery"
    );
    assert_eq!(same_physical_size, PhysicalSize::new(2, 2));
    assert_eq!(changed_physical_size, PhysicalSize::new(3, 2));
    let lost_capabilities =
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Lost,
        });
    assert_eq!(same_capabilities, lost_capabilities);
    assert_eq!(changed_capabilities, lost_capabilities);
    for result in [same_render, changed_render] {
        let error = result.expect_err("resize must not make a lost surface renderable");
        assert_surface_unavailable(
            error,
            RuntimeOperation::SurfaceRendering,
            RenderSurfaceAvailability::Lost,
        );
    }
    assert_eq!(
        [same_resource, changed_resource],
        [Some(committed_resource), Some(committed_resource)],
        "resize while lost must not publish a replacement configuration"
    );
    assert_eq!([same_target, changed_target], [committed_target; 2]);
    assert_eq!(
        [same_observation, changed_observation],
        [observation_before; 2],
        "rejected lost-surface renders must not acquire or present a frame"
    );
    assert_eq!([same_stats, changed_stats], [stats_before; 2]);
    assert_eq!(
        [same_parameters, changed_parameters],
        [parameters_before; 2]
    );
    assert_eq!(
        [same_active_generation, changed_active_generation],
        [None; 2]
    );

    assert_explicit_lost_resize_recovery(
        &mut renderer,
        &mut surface,
        committed_resource,
        committed_target,
    );
}

#[cfg(feature = "render-window")]
fn assert_explicit_lost_resize_recovery(
    renderer: &mut Renderer,
    surface: &mut Surface,
    committed_resource: u64,
    committed_target: u64,
) {
    let replacement_attachment = "lost-resize-replacement";
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        surface,
        Attachment::from_web_canvas(replacement_attachment),
    ))
    .expect("explicit resume must recover at the final requested extent");
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(3, 2));
    assert!(matches!(
        presented_lifecycle_for_test(surface),
        PresentedLifecycle::Ready { .. }
    ));
    let committed_physical_size = match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.committed_physical_size(),
        _ => panic!("the fixture must retain a presented surface backend"),
    };
    assert_eq!(committed_physical_size, Some(PhysicalSize::new(3, 2)));
    assert_ne!(
        presented_resource_id_for_test(surface),
        Some(committed_resource)
    );
    assert_ne!(
        presented_target_identity_for_test(surface),
        committed_target
    );
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("lost recovery must install a compatible presented attachment"),
        },
        replacement_attachment
    );
    assert!(matches!(
        renderer.runtime_capabilities(surface),
        RuntimeCapabilities::Available(_)
    ));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    pollster::block_on(renderer.render(surface, &Scene::new(), Parameters::default()))
        .expect("the explicitly resumed surface must render on its ready device");
}

#[cfg(feature = "render-window")]
#[test]
fn available_resize_pending_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available-resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::ResizePending { .. }
    ));
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("compatible-resume-candidate"),
    ))
    .expect("available resume must configure the pending extent on the installed target");

    let attachment_after = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id(),
        _ => panic!("available resume must retain the installed attachment kind"),
    };
    assert_eq!(
        (
            attachment_after,
            presented_target_identity_for_test(&surface)
        ),
        (installed_attachment.as_str(), installed_target),
        "available pending resume must retain the installed attachment and target identities"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(3, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    let configured_resource = presented_resource_id_for_test(&surface)
        .expect("pending resume must commit a configured target bundle");
    assert_ne!(configured_resource, installed_resource);
    let committed_physical_size = match &surface.backend {
        SurfaceBackend::Presented { surface, .. } => surface.committed_physical_size(),
        _ => panic!("the fixture must retain a presented surface backend"),
    };
    assert_eq!(committed_physical_size, Some(PhysicalSize::new(3, 2)));
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pending configuration must return its transaction generation"
    );
    assert!(matches!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Available(_)
    ));

    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the configured existing target must remain renderable");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn available_nonrenderable_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available nonrenderable resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    surface.resize(Size::new(0.0, 2.0), 1.0).unwrap();
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(0, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::NonRenderable {
            physical_size,
            resizing: ResizeState::Idle,
        } if physical_size == PhysicalSize::new(0, 2)
    ));

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-nonrenderable-resume-candidate"),
    ))
    .expect("available nonrenderable resume must be an idempotent compatible success");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("available resume must retain the installed attachment kind"),
        },
        installed_attachment
    );
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target,
        "available nonrenderable resume must retain the installed host target"
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource),
        "available nonrenderable resume must retain the installed target resources"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), PhysicalSize::new(0, 2));
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::NonRenderable { .. }
    ));
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::NonRenderable,
        })
    );
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the retained zero-area surface must remain nonrenderable");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::NonRenderable,
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "an idempotent available resume and rejected render must start no GPU transaction"
    );

    surface.resize(Size::new(2.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("restoring the installed extent must render through the retained target");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn available_occluded_resume_retains_installed_attachment_and_target() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("available occluded resume coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let installed_attachment = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the fixture must begin with a committed target bundle");
    let installed_observation = presented_observation_handle_for_test(&surface);

    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Occluded);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the synthetic occlusion must enter the occluded lifecycle");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Occluded { .. }
    ));

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-occluded-resume-candidate"),
    ))
    .expect("available occluded resume may remain occluded on its installed target");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("available resume must retain the installed attachment kind"),
        },
        installed_attachment
    );
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target,
        "available occluded resume must retain the installed host target"
    );
    assert_eq!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource),
        "available occluded resume must retain the installed target resources"
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Occluded { .. }
    ));
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
            state: RenderSurfaceAvailability::Occluded,
        })
    );
    let observation_before_rejected_render = installed_observation.snapshot_for_test();
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("an occluded surface must remain unavailable until explicit recovery");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Occluded,
    );
    assert_eq!(
        installed_observation.snapshot_for_test(),
        observation_before_rejected_render,
        "an occluded render rejection must not attempt another acquire"
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "an idempotent available resume and rejected render must start no GPU transaction"
    );

    surface.resize(Size::new(2.0, 2.0), 1.0).unwrap();
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("same-extent recovery must render through the retained target");
    let observation = installed_observation.snapshot_for_test();
    assert_eq!(observation.acquire_count_for_test(), 1);
    assert_eq!(observation.present_count_for_test(), 1);
    assert_eq!(observation.discarded_count_for_test(), 0);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
}

#[cfg(feature = "render-window")]
#[test]
fn suspended_presented_replacement_terminal_loss_before_configuration_uses_surface_resume() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("suspended replacement attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = presented_black_debug_parameters_for_test();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before replacement");
    surface.suspend().unwrap();

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let device_before = presented_device_identity_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    let error = pollster::block_on(
        renderer.resume_display_free_presented_surface_after_device_loss_for_test(
            &mut surface,
            Attachment::from_web_canvas("suspended-replacement-candidate"),
            DeviceLossReason::Unknown,
        ),
    )
    .expect_err("terminal loss before replacement configuration must abort resume");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
    assert_eq!(surface.state(), SurfaceState::Suspended);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(presented_device_identity_for_test(&surface), device_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed replacement must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pre-configuration loss must not begin a Configure transaction"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn lost_presented_recreation_terminal_loss_before_configuration_uses_surface_resume() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost recreation attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before loss");
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("synthetic acquire loss must require explicit recreation");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let device_before = presented_device_identity_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    let error = pollster::block_on(
        renderer.resume_display_free_presented_surface_after_device_loss_for_test(
            &mut surface,
            Attachment::from_web_canvas("lost-recreation-candidate"),
            DeviceLossReason::Unknown,
        ),
    )
    .expect_err("terminal loss before recreation configuration must abort resume");

    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(presented_device_identity_for_test(&surface), device_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Lost
    );
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed recreation must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "pre-configuration loss must not begin a Configure transaction"
    );
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resume_prefers_installed_compatible_slot_over_earlier_donor_slot() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("presented selection coverage requires a compatible device");
    let mut earlier = configured_display_free_presented_surface_for_test(&mut renderer);
    let earlier_device = presented_device_identity_for_test(&earlier);
    let earlier_resource = presented_resource_id_for_test(&earlier);
    let earlier_target = presented_target_identity_for_test(&earlier);
    let installed_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("presented selection coverage requires a later ready device slot");
    assert_ne!(installed_device, earlier_device);
    let mut surface = configured_display_free_presented_surface_on_device_for_test(
        &mut renderer,
        installed_device,
        Attachment::from_web_canvas("installed-slot-target"),
    );
    surface.suspend().unwrap();

    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("installed-slot-replacement"),
    ))
    .expect("resume must configure a replacement on the installed compatible slot");

    assert_eq!(
        presented_device_identity_for_test(&surface),
        installed_device,
        "an earlier compatible slot must not capture a surface from its installed ready slot"
    );
    assert_eq!(presented_resource_id_for_test(&earlier), earlier_resource);
    assert_eq!(presented_target_identity_for_test(&earlier), earlier_target);
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the resumed surface must render through its installed device slot");
    pollster::block_on(renderer.render(&mut earlier, &Scene::new(), Parameters::default()))
        .expect("the earlier donor surface must retain coherent resources");
}

#[cfg(feature = "render-window")]
#[test]
fn presented_resume_skips_terminal_compatible_donor_for_later_healthy_slot() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("terminal donor selection coverage requires a compatible device");
    let terminal_donor_surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let terminal_donor = presented_device_identity_for_test(&terminal_donor_surface);
    let terminal_donor_resource = presented_resource_id_for_test(&terminal_donor_surface);
    let terminal_donor_target = presented_target_identity_for_test(&terminal_donor_surface);
    let installed_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("terminal donor selection coverage requires an installed device slot");
    let mut surface = configured_display_free_presented_surface_on_device_for_test(
        &mut renderer,
        installed_device,
        Attachment::from_web_canvas("terminal-donor-installed-target"),
    );
    let parameters = presented_black_debug_parameters_for_test();
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the installed surface must establish public frame state before replacement");
    let installed_resource = presented_resource_id_for_test(&surface)
        .expect("the installed surface must own committed resources");
    let installed_target = presented_target_identity_for_test(&surface);
    let installed_options = surface.options;
    let installed_physical_size = surface.physical_size();
    let installed_renderer_identity = surface.renderer_identity.clone();
    let installed_stats = renderer.stats();

    let healthy_device = pollster::block_on(renderer.add_donor_device_slot_for_test())
        .expect("terminal donor selection coverage requires a later healthy device slot");
    assert_ne!(terminal_donor, installed_device);
    assert_ne!(terminal_donor, healthy_device);
    assert_ne!(installed_device, healthy_device);

    surface.suspend().unwrap();
    renderer.signal_device_loss_for_test(terminal_donor, DeviceLossReason::Destroyed);
    assert!(
        renderer
            .device_signal_for_test(terminal_donor)
            .expect("the terminal donor must retain its callback signal")
            .first_terminal()
            .is_some(),
        "the earlier donor must record terminal loss before selection"
    );
    let selected_device = select_display_free_presented_device_for_test(
        &mut renderer,
        installed_device,
        &[
            DisplayFreePresentedDeviceCompatibilityForTest::compatible(terminal_donor),
            DisplayFreePresentedDeviceCompatibilityForTest::incompatible(installed_device),
            DisplayFreePresentedDeviceCompatibilityForTest::compatible(healthy_device),
        ],
    )
    .expect("the explicit compatibility stage must find the later healthy slot");
    assert_eq!(selected_device, healthy_device);
    let SurfaceBackend::Presented {
        device_identity, ..
    } = &mut surface.backend
    else {
        panic!("the compatibility fixture must retain a presented surface");
    };
    *device_identity = selected_device;
    pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("terminal-donor-replacement-target"),
    ))
    .expect("resume must skip the terminal donor and publish through the later healthy slot");

    assert!(renderer.device_renderer_released_for_test(terminal_donor));
    assert_eq!(presented_device_identity_for_test(&surface), healthy_device);
    assert_eq!(surface.state(), SurfaceState::Available);
    assert!(matches!(
        presented_lifecycle_for_test(&surface),
        PresentedLifecycle::Ready { .. }
    ));
    assert_ne!(
        presented_resource_id_for_test(&surface),
        Some(installed_resource)
    );
    assert_ne!(
        presented_target_identity_for_test(&surface),
        installed_target
    );
    assert_eq!(surface.options, installed_options);
    assert_eq!(surface.physical_size(), installed_physical_size);
    assert!(
        surface
            .renderer_identity
            .matches(&installed_renderer_identity)
    );
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(renderer.stats(), installed_stats);
    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("the replacement must retain a web-canvas attachment"),
        },
        "terminal-donor-replacement-target"
    );
    assert_eq!(
        presented_resource_id_for_test(&terminal_donor_surface),
        terminal_donor_resource
    );
    assert_eq!(
        presented_target_identity_for_test(&terminal_donor_surface),
        terminal_donor_target
    );
    pollster::block_on(renderer.submit_scoped_wgpu_probe_for_test(installed_device))
        .expect("replacement incompatibility must not disable the installed healthy slot");
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
        .expect("the resumed surface must render through the later healthy slot");
}

#[cfg(feature = "render-window")]
#[test]
fn available_resize_pending_resume_terminal_loss_preserves_surface_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("pending resume attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before the resume race");
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let state_before = surface.state();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-pending-resume-candidate"),
    ))
    .expect_err("terminal loss must abort the pending resume configuration");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed resume must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(surface.state(), state_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "terminal resume preflight must not leave an active operation generation"
    );
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        })
    );
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
}

#[cfg(feature = "render-window")]
#[test]
fn lost_recreation_resume_terminal_loss_preserves_surface_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("lost recreation attribution coverage requires a compatible device");
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &Scene::new(), parameters))
        .expect("the fixture must establish public frame state before surface loss");
    set_presented_acquire_outcome_for_test(&mut surface, PresentedAcquireOutcomeForTest::Lost);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("the synthetic acquire loss must require explicit recreation");
    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Lost,
    );

    let attachment_before = match &surface.attachment {
        Attachment::WebCanvas(canvas) => canvas.id().to_owned(),
        _ => panic!("the display-free fixture must own a web-canvas attachment"),
    };
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let physical_size_before = surface.physical_size();
    let state_before = surface.state();
    let parameters_before = surface.last_parameters;
    let stats_before = renderer.stats();
    let observation_before = presented_observation_for_test(&surface);

    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);
    let error = pollster::block_on(renderer.resume_display_free_presented_surface_for_test(
        &mut surface,
        Attachment::from_web_canvas("different-lost-recreation-candidate"),
    ))
    .expect_err("terminal loss must abort replacement installation");

    assert_eq!(
        match &surface.attachment {
            Attachment::WebCanvas(canvas) => canvas.id(),
            _ => panic!("failed recreation must retain the installed attachment kind"),
        },
        attachment_before
    );
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(lifecycle_before, PresentedLifecycle::Lost);
    assert_eq!(surface.physical_size(), physical_size_before);
    assert_eq!(surface.state(), state_before);
    assert_eq!(surface.last_parameters, parameters_before);
    assert_eq!(renderer.stats(), stats_before);
    assert_eq!(presented_observation_for_test(&surface), observation_before);
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None,
        "terminal recreation preflight must not leave an active operation generation"
    );
    assert_eq!(
        renderer.runtime_capabilities(&surface),
        RuntimeCapabilities::Unavailable(RuntimeCapabilityUnavailableReason::DeviceLost {
            reason: DeviceLossReason::Unknown,
        })
    );
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceResume,
        DeviceLossReason::Unknown,
    );
}

#[cfg(feature = "render-window")]
fn presented_black_debug_parameters_for_test() -> Parameters {
    Parameters {
        base_color: Color::BLACK,
        debug: true,
    }
}

#[test]
fn zero_size_headless_render_diagnoses_and_read_returns_empty() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(0.0, 2.0), 1.0)).unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Empty);
    let error =
        pollster::block_on(renderer.render(&mut surface, &Scene::new(), Parameters::default()))
            .expect_err("zero-area headless rendering must be rejected before planning");
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceRendering,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::NonRenderable,
                },
            )
            .unwrap()
        )
    );

    let image = pollster::block_on(renderer.read_headless(&surface))
        .expect("zero-area headless readback returns a validated empty image");
    assert_eq!(image.size(), PhysicalSize::new(0, 2));
    assert!(image.rgba().is_empty());
}

#[test]
fn nonzero_headless_read_before_publication_reports_uninitialized_without_map() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let surface = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();

    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation,
        "creation must defer headless texture allocation"
    );
    let error = pollster::block_on(renderer.read_headless(&surface))
        .expect_err("a nonzero headless surface has no readable publication before render");
    assert_eq!(error.code(), ErrorCode::RuntimeCapabilityUnavailable);
    assert_eq!(
        error.runtime_capability_unavailable_diagnostic(),
        Some(
            &RuntimeCapabilityUnavailable::try_new(
                RuntimeOperation::SurfaceReadback,
                RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Uninitialized,
                },
            )
            .unwrap()
        )
    );
    assert_eq!(
        surface.resource_state(),
        SurfaceResourceState::PendingAllocation
    );
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

    assert_surface_unavailable(
        error,
        RuntimeOperation::SurfaceRendering,
        RenderSurfaceAvailability::Suspended,
    );

    surface.resume(Attachment::Headless).unwrap();
    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect("resumed headless surface should render");

    let error = surface
        .resume(Attachment::from_web_canvas("canvas"))
        .expect_err("surface backend kind should not change on resume");

    assert_eq!(error.code(), ErrorCode::SurfaceCreateFailed);
}

#[test]
fn non_render_operations_do_not_mutate_last_successful_stats() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    assert_eq!(renderer.stats(), Stats::default());
    let _ = renderer.capabilities();
    assert_eq!(renderer.stats(), Stats::default());

    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    assert_eq!(renderer.stats(), Stats::default());

    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);
    let last_successful =
        pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
            .expect("the baseline direct frame should publish stats");
    assert_eq!(last_successful.route, Some(RenderRoute::DirectVello));

    let _ = renderer.capabilities();
    assert_eq!(renderer.stats(), last_successful);
    let _ = renderer.runtime_capabilities(&surface);
    assert_eq!(renderer.stats(), last_successful);
    let _ = pollster::block_on(renderer.read_headless(&surface))
        .expect("explicit readback should observe the published frame");
    assert_eq!(renderer.stats(), last_successful);

    let _other = pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0))
        .expect("surface creation should remain independent from render stats");
    assert_eq!(renderer.stats(), last_successful);
    surface.resize(Size::new(3.0, 2.0), 1.0).unwrap();
    assert_eq!(renderer.stats(), last_successful);
    surface.suspend().unwrap();
    assert_eq!(renderer.stats(), last_successful);
    surface.resume(Attachment::Headless).unwrap();
    assert_eq!(renderer.stats(), last_successful);
}

#[test]
fn failed_and_canceled_graph_frames_preserve_last_successful_stats() {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect("graph stats failure coverage requires a renderer");
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(6.0, 4.0), 1.0))
        .expect("graph stats failure coverage requires a headless surface");
    let scene = repeated_graph_scene_for_test();
    let successful = pollster::block_on(renderer.render_forced_base_graph_for_test(
        &mut surface,
        &scene,
        Parameters::default(),
        working_format,
    ))
    .expect("the graph stats baseline must succeed")
    .stats;
    assert_eq!(successful.route, Some(RenderRoute::GpuGraph));
    assert!(successful.effect_texture_allocations > 0);

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::default();
    let mut publication = Some(1);
    pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(904),
        &mut publication,
    ))
    .expect_err("the explicit submitted transaction failure must not publish stats");
    assert_eq!(publication, Some(1));
    assert_eq!(renderer.stats(), successful);

    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let canceled_resources = ResourceManager::default();
    let mut canceled_publication = Some(1);
    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &canceled_resources,
            modeled_resource_key_for_test(905),
            &mut canceled_publication,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }
    assert_eq!(canceled_publication, Some(1));
    let canceled_resources = canceled_resources.observation_for_test();
    assert_eq!(canceled_resources.active_frame_count, 0);
    assert_eq!(canceled_resources.leased_count, 0);
    assert_eq!(canceled_resources.entry_count, 0);
    assert_eq!(renderer.stats(), successful);
}

#[test]
fn headless_direct_cancellation_after_submit_preserves_previous_publication() {
    let (mut renderer, surface, _replacement, published) =
        headless_direct_publication_fixture_for_test();
    let target_extent = PhysicalSize::new(2, 2);
    let prepared = prepared_direct_vello_pass_for_test(target_extent);
    pollster::block_on(
        renderer.cancel_prepared_vello_pass_after_submit_for_test(&prepared, target_extent),
    )
    .expect("the explicit canceled Vello submission must release its resources");
    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    assert_eq!(
        pollster::block_on(renderer.read_headless(&surface))
            .expect("a canceled frame must retain the previous publication")
            .rgba(),
        published.rgba(),
        "a canceled submitted frame must not overwrite readable published pixels"
    );
}

#[test]
fn headless_graph_post_submit_failure_leaves_first_frame_unpublished() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("first-frame graph failure coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::default();
    let mut publication = None;
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(901),
        &mut publication,
    ))
    .expect_err("a submitted transaction scope failure must not publish its draft");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, None);
    let resources = resources.observation_for_test();
    assert_eq!(resources.active_frame_count, 0);
    assert_eq!(resources.leased_count, 0);
    assert_eq!(resources.entry_count, 0);
}

#[test]
fn headless_accounting_fault_after_submit_suppresses_publication_and_commits() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("headless accounting coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let mut publication = Some(1);
    let error = pollster::block_on(graph_accounting_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(902),
        &mut publication,
    ))
    .expect_err("accounting poison after submit must abort draft publication");

    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    let after_fault = resources.observation_for_test();
    assert!(matches!(
        after_fault.accounting_fault_for_test(),
        Some(ResourceAccountingFault::RetainedByteMismatch { .. })
    ));
    assert_eq!(after_fault.active_frame_count, 0);
    assert_eq!(after_fault.leased_count, 0);
}

fn graph_white_replacement_scene_for_test() -> Scene {
    let mut replacement = Scene::new();
    replacement.fill(
        Rect::new(0.0, 0.0, 8.0, 8.0),
        Color::try_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    );
    replacement
}

fn graph_replacement_parameters_for_test() -> Parameters {
    Parameters {
        base_color: Color::TRANSPARENT,
        debug: true,
    }
}

struct GraphAbortFixtureForTest {
    renderer: Renderer,
    surface: Surface,
    replacement: Scene,
    replacement_parameters: Parameters,
    working_format: WorkingFormat,
    baseline_pixels: ImageBuffer,
    baseline_stats: Stats,
    baseline_parameters: Option<Parameters>,
    baseline_uploaded_images: std::collections::HashSet<ImageId>,
    baseline_publication_count: usize,
    baseline_cache: crate::shader::DevicePassCacheCountsForTest,
    resources_before: crate::resource::ResourceManagerObservationForTest,
}

fn graph_abort_fixture_for_test(
    renderer_expectation: &'static str,
    surface_expectation: &'static str,
    baseline_render_expectation: &'static str,
    baseline_read_expectation: &'static str,
    ready_device_expectation: &'static str,
    resource_manager_expectation: &'static str,
) -> GraphAbortFixtureForTest {
    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .expect(renderer_expectation);
    let working_format = default_graph_working_format_for_test(&mut renderer);
    let mut surface = pollster::block_on(renderer.create_headless(Size::new(8.0, 8.0), 1.0))
        .expect(surface_expectation);
    let mut baseline_scene = Scene::new();
    baseline_scene.fill(Rect::new(0.0, 0.0, 8.0, 8.0), Color::BLACK);
    pollster::block_on(renderer.render(&mut surface, &baseline_scene, Parameters::default()))
        .expect(baseline_render_expectation);
    let baseline_pixels =
        pollster::block_on(renderer.read_headless(&surface)).expect(baseline_read_expectation);
    let baseline_stats = renderer.stats();
    let baseline_parameters = surface.last_parameters;
    let baseline_uploaded_images = renderer.uploaded_images_for_test();
    let baseline_publication_count = surface.headless_publication_count_for_test();
    let baseline_cache = renderer
        .default_ready_device_state_borrow_for_test()
        .expect(ready_device_expectation)
        .device_pass_cache_counts_for_test();
    let resources_before = renderer
        .default_ready_device_state_borrow_for_test()
        .expect(resource_manager_expectation)
        .internal_resource_manager_observation_for_test();
    GraphAbortFixtureForTest {
        renderer,
        surface,
        replacement: graph_white_replacement_scene_for_test(),
        replacement_parameters: graph_replacement_parameters_for_test(),
        working_format,
        baseline_pixels,
        baseline_stats,
        baseline_parameters,
        baseline_uploaded_images,
        baseline_publication_count,
        baseline_cache,
        resources_before,
    }
}

#[test]
fn post_submit_scope_failure_discards_prepared_resources_with_nonzero_budget() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default()))
        .expect("post-submit graph abort coverage requires a renderer");
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let resources_before = resources.observation_for_test();
    let mut publication = Some(1);
    let error = pollster::block_on(graph_scope_failure_after_submission_for_test(
        &device,
        &queue,
        signal,
        &resources,
        modeled_resource_key_for_test(903),
        &mut publication,
    ))
    .expect_err("the explicit post-submit scope failure must abort its resource frame");
    assert_eq!(error.code(), ErrorCode::RenderFailed);
    assert_eq!(publication, Some(1));
    let resources_after = resources.observation_for_test();
    assert_eq!(resources_after.active_frame_count, 0);
    assert_eq!(resources_after.leased_count, 0);
    assert_eq!(resources_after.entry_count, 0);
    assert!(
        resources_after.lifecycle_stats_for_test().evictions
            > resources_before.lifecycle_stats_for_test().evictions
    );
    assert_eq!(resources_after.accounted_entry_bytes, Some(0));
}

#[test]
fn canceled_graph_after_real_submit_discards_prepared_resources_and_retries_fresh() {
    let GraphAbortFixtureForTest {
        mut renderer,
        mut surface,
        replacement,
        replacement_parameters,
        working_format,
        baseline_pixels,
        baseline_stats,
        baseline_parameters,
        baseline_uploaded_images,
        baseline_publication_count,
        baseline_cache,
        resources_before,
    } = graph_abort_fixture_for_test(
        "submitted graph cancellation coverage requires a renderer",
        "submitted graph cancellation coverage requires a headless surface",
        "the direct baseline frame must publish before cancellation coverage",
        "the cancellation baseline publication must be readable",
        "the cancellation baseline must retain a ready device",
        "the cancellation baseline must retain one resource manager",
    );
    let (device, queue, signal) = explicit_graph_transaction_inputs_for_test(&mut renderer);
    let canceled_resources = ResourceManager::new(ResourceCacheBudget::new(256 * 1024 * 1024));
    let canceled_resources_before = canceled_resources.observation_for_test();
    let mut canceled_publication = Some(1);
    {
        let future = graph_cancellation_after_submission_for_test(
            &device,
            &queue,
            signal,
            &canceled_resources,
            modeled_resource_key_for_test(904),
            &mut canceled_publication,
        );
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }
    assert_eq!(canceled_publication, Some(1));
    let canceled_resources_after = canceled_resources.observation_for_test();
    assert_eq!(canceled_resources_after.active_frame_count, 0);
    assert_eq!(canceled_resources_after.leased_count, 0);
    assert_eq!(canceled_resources_after.entry_count, 0);
    assert!(
        canceled_resources_after
            .lifecycle_stats_for_test()
            .evictions
            > canceled_resources_before
                .lifecycle_stats_for_test()
                .evictions
    );
    assert_eq!(renderer.stats(), baseline_stats);
    assert_eq!(surface.last_parameters, baseline_parameters);
    assert_eq!(
        renderer.uploaded_images_for_test(),
        baseline_uploaded_images
    );
    assert_eq!(
        surface.headless_publication_count_for_test(),
        baseline_publication_count
    );
    assert_eq!(
        renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the canceled frame must retain the ready device")
            .device_pass_cache_counts_for_test(),
        baseline_cache
    );
    assert_eq!(
        renderer.default_device_active_operation_generation_for_test(),
        None
    );
    let resources_after_abort = renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the canceled frame must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after_abort.active_frame_count, 0);
    assert_eq!(resources_after_abort.leased_count, 0);
    assert_eq!(resources_after_abort.resolved_lease_count, 0);
    assert_eq!(
        resources_after_abort.retained_bytes,
        resources_after_abort
            .accounted_entry_bytes
            .expect("canceled frame resource accounting must have an exact total")
    );
    assert_eq!(resources_after_abort, resources_before);
    assert_eq!(
        pollster::block_on(renderer.read_headless(&surface))
            .expect("the canceled graph must preserve the baseline publication")
            .rgba(),
        baseline_pixels.rgba()
    );

    assert_graph_retry_after_abort(
        GraphRetryContextForTest {
            renderer: &mut renderer,
            surface: &mut surface,
            replacement: &replacement,
            replacement_parameters,
            working_format,
            baseline_cache,
            baseline_publication_count,
            baseline_pixels: &baseline_pixels,
        },
        GraphRetryExpectationsForTest {
            success: "a clean graph retry must succeed after submitted cancellation",
            readable: "the clean retry publication must be readable",
        },
    );
}

struct GraphRetryContextForTest<'a> {
    renderer: &'a mut Renderer,
    surface: &'a mut Surface,
    replacement: &'a Scene,
    replacement_parameters: Parameters,
    working_format: WorkingFormat,
    baseline_cache: crate::shader::DevicePassCacheCountsForTest,
    baseline_publication_count: usize,
    baseline_pixels: &'a ImageBuffer,
}

struct GraphRetryExpectationsForTest {
    success: &'static str,
    readable: &'static str,
}

fn assert_graph_retry_after_abort(
    context: GraphRetryContextForTest<'_>,
    expectations: GraphRetryExpectationsForTest,
) {
    let retry = pollster::block_on(context.renderer.render_forced_base_graph_for_test(
        context.surface,
        context.replacement,
        context.replacement_parameters,
        context.working_format,
    ))
    .expect(expectations.success);
    let resources_after_retry = context
        .renderer
        .default_ready_device_state_borrow_for_test()
        .expect("the clean retry must retain one resource manager")
        .internal_resource_manager_observation_for_test();
    assert_eq!(resources_after_retry.active_frame_count, 0);
    assert_eq!(resources_after_retry.resolved_lease_count, 0);
    assert_eq!(resources_after_retry.leased_count, 0);
    assert_eq!(
        resources_after_retry.retained_bytes,
        resources_after_retry
            .accounted_entry_bytes
            .expect("clean retry resource accounting must have an exact total")
    );
    assert!(resources_after_retry.entry_count > 0);
    assert_ne!(
        context
            .renderer
            .default_ready_device_state_borrow_for_test()
            .expect("the clean retry must retain its committed pass cache")
            .device_pass_cache_counts_for_test(),
        context.baseline_cache
    );
    assert_eq!(context.renderer.stats(), retry.stats);
    assert_eq!(
        context.surface.last_parameters,
        Some(context.replacement_parameters)
    );
    assert_eq!(
        context.surface.headless_publication_count_for_test(),
        context.baseline_publication_count + 1
    );
    assert_ne!(
        pollster::block_on(context.renderer.read_headless(context.surface))
            .expect(expectations.readable)
            .rgba(),
        context.baseline_pixels.rgba()
    );
}

#[test]
fn terminal_signal_after_successful_headless_publication_preserves_frame_state() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(2.0, 2.0), 1.0)).unwrap();
    let image = Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255]))
        .expect("the baseline image must be valid");
    let mut first = Scene::new();
    first.image(image, Rect::new(0.0, 0.0, 2.0, 2.0), ImageFit::Stretch);
    let first_parameters = Parameters {
        base_color: Color::BLACK,
        debug: true,
    };
    pollster::block_on(renderer.render(&mut surface, &first, first_parameters))
        .expect("the first frame must establish the public state to preserve");
    let _prior_pixels = pollster::block_on(renderer.read_headless(&surface))
        .expect("the first frame must establish readable pixels");
    let prior_texture = match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the readable headless frame must retain its published texture"),
    };
    let prior_parameters = surface.last_parameters;
    let prior_uploaded_images = renderer.uploaded_images_for_test();
    let prior_publication_count = surface.headless_publication_count_for_test();

    let replacement =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([255, 255, 255, 255]))
            .expect("the replacement image must be valid");
    let mut next = Scene::new();
    next.image(
        replacement,
        Rect::new(0.0, 0.0, 2.0, 2.0),
        ImageFit::Stretch,
    );
    let next_parameters = Parameters {
        base_color: Color::TRANSPARENT,
        debug: false,
    };
    let current = pollster::block_on(renderer.render(&mut surface, &next, next_parameters))
        .unwrap_or_else(|error| panic!("the replacement frame must publish: {error}"));
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    let current_texture = match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => texture.clone(),
        _ => panic!("the completed frame must install its headless publication"),
    };
    assert_ne!(
        current_texture, prior_texture,
        "the completed frame must replace the prior published texture"
    );
    assert_eq!(renderer.stats(), current);
    assert_eq!(surface.last_parameters, Some(next_parameters));
    assert_ne!(surface.last_parameters, prior_parameters);
    assert_ne!(renderer.uploaded_images_for_test(), prior_uploaded_images);
    assert_eq!(
        surface.headless_publication_count_for_test(),
        prior_publication_count + 1
    );

    let committed_stats = renderer.stats();
    let committed_parameters = surface.last_parameters;
    let committed_uploaded_images = renderer.uploaded_images_for_test();
    let committed_publication_count = surface.headless_publication_count_for_test();
    let error = pollster::block_on(renderer.render(&mut surface, &next, Parameters::default()))
        .expect_err("the operation after an idle terminal signal must fail deterministically");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(renderer.stats(), committed_stats);
    assert_eq!(surface.last_parameters, committed_parameters);
    assert_eq!(
        renderer.uploaded_images_for_test(),
        committed_uploaded_images
    );
    assert_eq!(
        surface.headless_publication_count_for_test(),
        committed_publication_count
    );
    match &surface.backend {
        SurfaceBackend::Headless {
            resources: HeadlessResources::Ready { texture },
            ..
        } => assert_eq!(texture, &current_texture),
        _ => panic!("the rejected next operation must preserve the completed publication"),
    }
}

#[test]
fn headless_render_can_be_read_back() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface =
        pollster::block_on(renderer.create_headless(Size::new(4.0, 4.0), 1.0)).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 4.0, 4.0), Color::BLACK);

    pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default())).unwrap();
    let image = pollster::block_on(renderer.read_headless(&surface)).unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    assert_eq!(image.size(), PhysicalSize::new(4, 4));
    assert_eq!(image.rgba().len(), 4 * 4 * 4);
    assert!(image.rgba().iter().any(|channel| *channel != 0));
}

#[cfg(feature = "render-window")]
#[test]
fn presented_terminal_signal_after_publication_fails_the_next_operation() {
    let rect = Rect::new(0.0, 0.0, 2.0, 2.0);
    let scene = composition_presented_masked_blended_scene_for_test(rect);

    let mut renderer = pollster::block_on(Renderer::new(
        Options::default()
            .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision)
            .with_resource_cache_budget(ResourceCacheBudget::new(256 * 1024 * 1024)),
    ))
    .unwrap_or_else(|error| {
        panic!("presented terminal-signal coverage requires a compatible renderer: {error}")
    });
    let mut surface = configured_display_free_presented_surface_for_test(&mut renderer);
    let parameters = Parameters {
        base_color: color_from_straight_rgba8_for_test([48, 160, 208, 255]),
        debug: true,
    };
    let lifecycle_before = presented_lifecycle_for_test(&surface);
    let target_before = presented_target_identity_for_test(&surface);
    let resource_before = presented_resource_id_for_test(&surface);

    let stats = pollster::block_on(renderer.render(&mut surface, &scene, parameters))
        .unwrap_or_else(|error| panic!("the presented graph frame must publish: {error}"));
    renderer.signal_default_device_loss_for_test(DeviceLossReason::Unknown);

    let presented = presented_observation_for_test(&surface);
    assert_eq!(presented.acquire_attempt_count_for_test(), 1);
    assert_eq!(presented.acquire_count_for_test(), 1);
    assert_eq!(presented.present_count_for_test(), 1);
    assert_eq!(presented.discarded_count_for_test(), 0);
    assert_eq!(renderer.stats(), stats);
    assert_eq!(surface.last_parameters, Some(parameters));
    assert_eq!(surface.state(), SurfaceState::Available);
    assert_eq!(surface.resource_state(), SurfaceResourceState::Presented);
    assert_eq!(presented_lifecycle_for_test(&surface), lifecycle_before);
    assert_eq!(presented_target_identity_for_test(&surface), target_before);
    assert_eq!(presented_resource_id_for_test(&surface), resource_before);
    assert_eq!(surface.headless_publication_count_for_test(), 0);

    let committed_stats = renderer.stats();
    let committed_parameters = surface.last_parameters;
    let committed_lifecycle = presented_lifecycle_for_test(&surface);
    let committed_target = presented_target_identity_for_test(&surface);
    let committed_resource = presented_resource_id_for_test(&surface);
    let error = pollster::block_on(renderer.render(&mut surface, &scene, Parameters::default()))
        .expect_err("the operation after an idle terminal signal must fail deterministically");
    assert_runtime_device_lost(
        error,
        RuntimeOperation::SurfaceRendering,
        DeviceLossReason::Unknown,
    );
    assert_eq!(presented_observation_for_test(&surface), presented);
    assert_eq!(renderer.stats(), committed_stats);
    assert_eq!(surface.last_parameters, committed_parameters);
    assert_eq!(presented_lifecycle_for_test(&surface), committed_lifecycle);
    assert_eq!(
        presented_target_identity_for_test(&surface),
        committed_target
    );
    assert_eq!(presented_resource_id_for_test(&surface), committed_resource);
    assert_eq!(surface.headless_publication_count_for_test(), 0);
    assert!(take_last_presented_texture_for_test(&mut surface).is_some());
}
