# GPU Render Pipeline C05 Surface Readback

## Header
- Cycle: `C05`; owner: `surgeist-render`; status: `complete`.
- Cycle base and published prerequisite: `24e7375883188bc64a92529b3dfe2d8ed556ade0` (C04).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: the `ImageBuffer` portion of S09, readback portions of S13-S13A, S25-S26, readback rows of S28-S29, and C05-applicable S31-S35 evidence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `c1a203393b2549603c0a0d5698099f55018abe2e`, normalized SHA-256 `70b345be31ac5bf4fcae72e2d3c5901c8e04ad0b2c26080851bd2c50d7150cde`, entry `C05 Surface Readback`.
- Outcome: replace the temporary blocking texture-download helper with one private, transaction-owned, native/wasm async readback state machine over C04 publications, expose only validated complete bytes, and make cancellation clean every uncertain staging state.

## Boundary
- C04 supplies async atomic rendering, stable published headless textures, exact surface/device preflight, and transaction-owned non-readback submission. C05 neither reopens those decisions nor changes publication on read success, failure, or cancellation.
- `ImageBuffer` is the public readback/fixture value. Its fields become private and its sole public constructor enforces exact checked RGBA8 length. `read_headless` becomes async and returns this validated value; its future is not promised to be `Send`.
- A zero-area available headless surface returns a validated empty image without WGPU work. A nonzero unpublished surface remains typed `Uninitialized`; a published surface copies only its current publication. Existing identity, backend-kind, generation, suspension, lifecycle, and terminal-device ordering remains authoritative.
- Readback allocation, copy encoding, submission, map registration, progress, row decoding, completion, and cleanup have one owner in private `src/readback.rs`. Copy work is scoped to a readback GPU transaction; validation/internal/map/decode/wrong-index failures become `ReadbackFailed`, out-of-memory remains `SurfaceOutOfMemory`, and terminal device loss/fault takes precedence for the owning runtime operation.
- Native progress uses one short-lived helper thread and bounded 50 ms `Device::poll(PollType::Wait { submission_index: Some(index), timeout: Some(...) })` slices. Timeout means continue after checking completion/cancellation; `WrongSubmissionIndex` is terminal. Wasm uses the event-loop callback and contains no `Device::poll` call.
- One `map_async` callback stores at most one terminal result in an `Arc` completion cell and wakes the latest registered task waker. Bytes are copied from one validated nonempty aligned mapped range, row padding is stripped with checked arithmetic, the view is dropped, the buffer is unmapped, and only then may complete bytes be published as an `ImageBuffer`.
- Dropping any in-flight readback future marks cancellation, safely unmaps and releases its staging buffer, causes late callback delivery to be discarded, and lets the native helper exit after at most its current bounded slice. No uncertain staging buffer is pooled and no readback mutates surface publication, renderer stats, parameters, or resource state.
- Every named test that exercises the real GPU readback or a temporary materialized download treats an unavailable adapter as a required-host failure. Contract-only adapter behavior remains covered by separate deterministic tests; no required execution test returns early or passes by skipping.
- The existing temporary materialized mask/backdrop/filter callers retain their current observable behavior until their reviewed GPU replacements. In C05 they may await this same private readback owner, with `SurfaceRendering` as the terminal-device operation, but they may not allocate, submit, map, poll, decode, or block independently. C13 still owns removal of those transitional CPU/materialized routes and stale public phases.
- C05 excludes C06+ graph planning/effect resources/passes, C13 cutover and capability reconciliation, C14 docs/platform smoke, root integration/API artifacts, compatibility shims, CPU fallback, dependencies, acquisition, and generated artifacts. Owned Rust remains free of `unsafe`.

## Impacts
| Area | C05 record |
| --- | --- |
| Public API | Breaking/corrective: `ImageBuffer` fields become private with `try_new`, `size`, `rgba`, and `into_rgba`; `Renderer::read_headless` becomes async; `ErrorCode::ReadbackFailed` is additive. No shim is retained. |
| Dependencies/features | Unchanged. Normal, dev, target-specific dev roles, and all four native plus one wasm-supported feature states remain exactly S36. |
| Artifacts/docs/MSRV | No generated, fixture, license, README, or example delta; changed public items receive source docs. Rust 1.97 and Rust 2024 remain required. |
| Root/handoff | No root edit. The published candidate reports the breaking async/private-field API so root may adapt later; C06 receives stable validated readback and surface states. |

## Tasks
Define these functions in each task shell before focused commands; they reject zero or ambiguous test selection and owned unsafe source.
```sh
run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(CARGO_NET_OFFLINE=true cargo test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    CARGO_NET_OFFLINE=true cargo test -p surgeist-render "tests::$name" -- --exact
  fi
}

run_required_download_matrix() {
  local name
  for name in \
    offscreen_local_vello_scene_renders_to_texture_when_gpu_context_is_available \
    offscreen_reuses_resources_across_repeated_bounded_requests \
    shader_clear_fill_pass_encodes_when_gpu_context_is_available \
    render_materializes_bounded_backdrop_capture_from_prior_siblings \
    render_backdrop_filter_order_is_preserved \
    render_backdrop_clip_limits_filtered_image_to_requested_region \
    render_backdrop_foreground_composites_over_filtered_backdrop \
    sequence13_bounded_backdrop_capture_materializes_prior_siblings_with_foreground_order \
    sequence13_backdrop_filter_chain_preserves_order_and_clipping \
    ahem_font_data_renders_ascent_and_descent_glyph_bands
  do
    run_exact_test "$name"
  done
}

assert_no_owned_unsafe() {
  local output scan_status
  if output="$(git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg -n --pcre2 '#\s*!?\[\s*(?:allow|expect)\s*\([^)]*\bunsafe(?:_[A-Za-z0-9_]+)?\b|#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' 2>&1)"; then
    printf '%s\n' "$output" >&2
    return 1
  else
    scan_status=$?
    test "$scan_status" -eq 1 && return 0
    printf '%s\n' "$output" >&2
    return "$scan_status"
  fi
}
```

**C05-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
stable_version="$(CARGO_NET_OFFLINE=true rustc +stable --version)"
printf '%s\n' "$stable_version"
case "$stable_version" in
  "rustc 1.97."*) ;;
  *) printf 'expected Rust 1.97.x, found %s\n' "$stable_version" >&2; exit 1 ;;
esac
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(CARGO_NET_OFFLINE=true cargo tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
native_getrandom_tree="$(CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
test -z "$(git ls-files -- Cargo.lock)"
assert_no_owned_unsafe
```
`rustc +stable --version` must report Rust `1.97.x`. The wasm command is compile evidence only; browser callback execution remains root-owned. `assert_no_owned_unsafe` converts only ripgrep's no-match status to success and rejects matches or command failures.

T1-T5 use the `default` profile for every named test below.

**T1. Make every `ImageBuffer` intrinsically valid**
- Area/outcome: `src/{image.rs,layer.rs,command.rs,validation.rs,backend.rs,renderer.rs,tests.rs}` and public source docs; make both fields private, add the exact S09 constructor/accessors, and migrate every direct construction, validation, and observation through the validated boundary without changing temporary effect semantics. Validation may use the accessors or remove only checks made redundant by the sole validated constructor.
- RED/acceptance: add `image_buffer_rejects_short_long_and_overflowing_byte_lengths` first; it cannot compile against the missing constructor. Pass requires checked `width * height * 4`, exact zero/nonzero lengths, typed `InvalidValue` for short/long/overflow, no panic/saturation, and `image_buffer_accepts_exact_and_zero_area_lengths_and_round_trips_bytes` proving accessors/consuming extraction. Existing mask/filter/readback fixtures construct only valid buffers.
- Commands: run both named tests separately with `run_exact_test`; run `run_exact_test resolved_alpha_mask_execution_applies_materialized_alpha_buffer`; run `run_exact_test materialized_image_filters_preserve_color_and_blur_order`; then `C05-CHECK`.
- Depends: none. Commit: `Validate image buffer construction`.

**T2. Scope and centralize every readback copy**
- Area/outcome: `src/{lib.rs,error.rs,gpu_transaction.rs,backend.rs,readback.rs,renderer.rs,tests.rs}`; add private `readback`, `ReadbackFailed`, a readback GPU stage/submission result retaining `SubmissionIndex`, and async `read_headless`. Move all current helper callers to one async entry so allocation, encoding, and queue submission occur under readback scopes while preserving public and transitional materialized results.
- RED/acceptance: add `readback_transaction_maps_validation_internal_oom_and_terminal_failures` first; it fails because there is no readback stage/code. Pass maps readback validation/internal to `ReadbackFailed`, OOM globally, terminal loss/fault with operation precedence, and proves the copy submission is observed with its active transaction generation and exact submission index. Replace `render_scene_to_headless_or_skip_no_adapter` globally with a required-host helper returning `ImageBuffer`, remove every caller's optional early return, and make both offscreen GPU-context tests require the adapter while retaining the separate missing-context diagnostic test. Every `run_required_download_matrix` member fails on unavailable or failed render/readback. All synchronous test callers use the already-pinned dev-only `pollster`; production contains no blocking executor.
- Commands: `run_exact_test readback_transaction_maps_validation_internal_oom_and_terminal_failures`; `run_exact_test headless_render_can_be_read_back`; `run_exact_test zero_size_headless_render_diagnoses_and_read_returns_empty`; `run_exact_test nonzero_headless_read_before_publication_reports_uninitialized_without_map`; `run_required_download_matrix`; `run_exact_test layer_resolved_alpha_mask_applies_after_children_before_parent_composite`; then `C05-CHECK`.
- Depends: T1. Commit: `Centralize scoped readback copies`.

**T3. Implement the cancellation-safe native/wasm map state machine**
- Area/outcome: `src/{readback.rs,gpu_transaction.rs,backend.rs,renderer.rs,tests.rs}`; replace the remaining blocking completion with the S13A `Allocated -> CopySubmitted -> MapPending -> Mapped -> PublishedBytes` owner and terminal `Failed/Canceled` cleanup, using a latest-waker completion cell, one callback, checked row decoding, native bounded poll helper, and wasm event-loop progress.
- RED/acceptance: add `readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers` first against a private deterministic seam and show the absent state owner. Pass proves exactly-once completion, latest-waker replacement, callback error, wrong-index error, checked row failure, mapped-view-before-unmap ordering, cleanup from every uncertain state, late-result discard, and no buffer reuse. Native timeout slices continue; wasm compiles with no thread or poll path.
- Commands: `run_exact_test readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers`; `run_exact_test readback_map_callback_publishes_once_and_wakes_latest_waker`; `run_exact_test headless_render_can_be_read_back`; then `C05-CHECK`.
- Depends: T2. Commit: `Implement async readback state machine`.

**T4. Prove native callback progress and cancellation on real publications**
- Area/outcome: `src/{readback.rs,renderer.rs,surface.rs,tests.rs}`; add condition-based real-GPU evidence over C04 publications and test-only observations needed to diagnose state/submission/device signal without exposing production API.
- RED/acceptance: add `native_readback_callback_progresses_and_cleans_up_with_diagnostic_deadline` first and show missing callback progress evidence, then add `canceled_native_readback_discards_late_callback_without_publication_change`. Pass uses fresh condition cells and one five-second test-only diagnostic deadline, reports state/index/signal on expiry, cancels after real copy submission while map is pending, proves helper/callback cleanup, and preserves publication identity/bytes, stats, parameters, and resource state. No elapsed sleep, skipped adapter, public timeout, or production test hook is permitted.
- Commands: run both named tests separately with `run_exact_test`; `run_exact_test headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`; `run_exact_test terminal_signal_after_transaction_completion_preserves_public_frame_state`; then `C05-CHECK`.
- Depends: T3. Commit: `Prove readback progress and cancellation`.

**T5. Close readback reachability, error, and target evidence**
- Area/outcome: integrated `src/`, `Cargo.toml`, and tests; close public preflight/terminal/error behavior, preserve temporary materialized outputs through the single owner, and make static reachability mechanically inspectable across native and wasm.
- RED/acceptance: strengthen `render_path_submits_without_map_or_cpu_wait` and add `readback_static_paths_confine_map_poll_and_copy_submission` before final cleanup. Pass requires `map_async`, `MAP_READ`, row decode, and production native `Device::poll` only in `readback.rs`; no `wait_indefinitely`; no map/poll in renderer/backend/shader/internal Vello; every queue submit only in `gpu_transaction.rs`; explicit readback is async; zero/uninitialized/foreign/stale/suspended/terminal states remain exact; and transitional mask/backdrop paths still match their current pixels while using no second readback implementation. The old adapter-skip helper and both offscreen early-return branches are absent, every required download matrix row still fails on missing execution, and contract-only behavior remains in its separate tests.
- Commands: `run_exact_test render_path_submits_without_map_or_cpu_wait`; `run_exact_test readback_static_paths_confine_map_poll_and_copy_submission`; `run_exact_test surface_operation_matrix_covers_every_kind_state_and_duplicate_transition`; `run_exact_test foreign_and_stale_surfaces_fail_before_device_slot_access`; `run_required_download_matrix`; `run_exact_test sequence12_executes_materialized_alpha_masks_for_resolved_buffers_and_layers`; run the final guards below; then `C05-CHECK`.
- Depends: T4. Commit: `Close surface readback evidence`.

## Completion
- Require all five ordered task ranges and clean task reviews. `ImageBuffer` has no invalid public state; one async owner performs every current download; public readback observes only the current complete publication; all uncertain map states clean safely; native progress cannot block the async executor indefinitely; wasm contains no poll helper; failures/cancellation publish no bytes or frame state.
- Run `C05-CHECK` and every named matrix test above through `run_exact_test`; require a clean worktree. Final guards are: `rg -n 'wait_indefinitely|std::sync::mpsc::channel|\.recv\(\)|pollster::block_on' src/readback.rs src/backend.rs src/renderer.rs`; `rg -n 'map_async|MAP_READ|PollType::Wait|get_mapped_range' src/backend.rs src/renderer.rs src/shader.rs src/vello_engine`; `rg -n 'queue\.submit' src --glob '*.rs' | rg -v '^src/gpu_transaction\.rs:'`; `rg -n 'render_scene_to_headless_or_skip_no_adapter|no GPU machines should report the explicit diagnostic' src/tests.rs`; each exits `1` clean. The first guard forbids the old blocking channel construction/receive without rejecting C04's unrelated cfg-test presentation control; the last forbids success-by-skip GPU/download evidence. Require `map_async`, `MAP_READ`, checked row decode, and the cfg-native bounded `PollType::Wait` implementation in `src/readback.rs`; no `Device::poll` symbol in its wasm-compiled branch; and no direct public `ImageBuffer` field access compiles.
- After cycle acceptance, follow `$surgeist-agent`'s canonical implementation-cycle and automated landing/publication gates, then its crate-candidate handoff contract for C06.
- Block only for an unowned worktree conflict, contradiction in the reviewed spec/sequence, unavailable required native GPU execution, missing already-authorized Rust 1.97/wasm target support, or unavailable required Surgeist agent profile. No substitute target, skipped adapter, browser-execution claim, dependency acquisition, or compatibility shim is allowed.
