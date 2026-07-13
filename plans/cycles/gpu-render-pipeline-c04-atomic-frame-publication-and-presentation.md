# GPU Render Pipeline C04 Atomic Frame Publication And Presentation

## Header
- Cycle: `C04`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `cffc37b909d63c5382e6ef3b91653ed2943f3cbf` (C03).
- Specification: `plans/specs/gpu-render-pipeline.md` at `1e6517e4e33669d97b1f45c0df9c1de78ec4d07e`, normalized SHA-256 `db78f70e03a31430e949ac06de6628ca24a03cd53cf5dec453b43bcf4fbe53be`: S12-S13A, publication/lifecycle portions of S25-S26, applicable S29, and C04-applicable S31-S35 evidence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `b46c9c2afb6f705fdaf928d640b3821a8e29c0c9`, normalized SHA-256 `3dab5afdeb5084026f4863a3f0f4dfa18de47441a2560e1f5cbd1562732d8bdf`, entry `C04 Atomic Frame Publication And Presentation`.
- Outcome: make headless publication frame-atomic; place presented setup, configuration, acquisition, output submission, and presentation under Surgeist transactions; close the non-readback surface lifecycle and error matrix.

## Boundary
- C03 supplies the private internal Vello engine, device owner, raster leases, and transaction-owned raster submission. C04 changes surface/frame ownership around that engine; it does not reopen raster provenance, glyph lowering, scheduling, or retained-atlas policy.
- A device-backed headless surface starts as `Empty` at zero physical area or `Pending` at nonzero area. It allocates only a transient draft during render, and only a clean frame commit may replace its optional published texture.
- A presented frame renders into a Surgeist-owned intermediate. Configuration and output work use real stage-specific transaction scopes; the output transaction owns any acquired `wgpu::SurfaceTexture`, submits the blit, calls `present`, awaits scopes/signals, and only then permits public frame commit. An attempted present is not rolled back when the final scope/signal check fails.
- Surface identity, kind, generation, suspension, private lifecycle, and runtime capability are checked in the S26 order before scene planning or WGPU work. State fields are mutated only by private transitions; failure and cancellation publish no draft stats, parameters, pixels, or lifecycle success.
- C04 may change only readback preflight: zero-size returns a validated empty image, nonzero without publication is typed `Uninitialized`, and a published texture reaches the existing synchronous helper. C05 owns the async copy/map/poll state machine and `ReadbackFailed`; the temporary helper is neither redesigned nor counted as C04 evidence.
- C04 excludes C05 readback execution, C06+ graph/effect/resource-budget work, C14 browser/window smoke and docs, root integration, new dependencies, compatibility shims, CPU fallback, and acquisition. Owned Rust remains free of `unsafe`.

## Impacts
| Area | C04 record |
| --- | --- |
| Public API | Breaking/corrective: add `SurfaceResourceState::Empty`; remove legacy `ErrorCode::{AdapterUnavailable, SurfaceLost, SurfaceUnavailable}` in favor of exact runtime diagnostics; retain synchronous `read_headless` until C05. |
| Dependencies/features | Unchanged exact C03 manifest; default, `render-window`, `render-web`, and combined native feature states remain supported. |
| Artifacts/docs/MSRV | No generated or license artifact delta; docs/examples remain C14-owned; Rust 1.97 and Rust 2024 remain required. |
| Root/handoff | No root edit. A published C04 candidate hands stable publication/lifecycle semantics to C05. |

## Tasks
Define this function in each task shell before focused commands; it rejects zero or ambiguous selection.
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

**C04-CHECK (run verbatim after every task)**
```sh
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
CARGO_NET_OFFLINE=true rustc +stable --version
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
test -z "$(git ls-files -- Cargo.lock)"
assert_no_owned_unsafe
```
`rustc +stable --version` must report Rust `1.97.x`. `assert_no_owned_unsafe` converts only ripgrep's no-match status to success and rejects matches or command failures.

T1 uses its authoritative wasm compile failure as RED/GREEN evidence. For the remaining matrix, invoke `run_exact_test name` for `default` and `run_exact_test name profile` otherwise.
| Owner | Profile | Exact `src/tests.rs` names |
| --- | --- | --- |
| T2 | `default` | `surface_operation_matrix_covers_every_kind_state_and_duplicate_transition`; `zero_size_headless_render_diagnoses_and_read_returns_empty`; `nonzero_headless_read_before_publication_reports_uninitialized_without_map`; `headless_bgra8_remains_a_surface_create_diagnostic`; `foreign_and_stale_surfaces_fail_before_device_slot_access` |
| T3 | `default` | `headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`; `failed_frame_returns_all_leases_and_preserves_last_successful_stats`; `dropped_gpu_operation_future_aborts_draft_state_and_leases`; `headless_render_can_be_read_back`; `render_path_submits_without_map_or_cpu_wait` |
| T4 | `default` | `non_readback_gpu_submissions_are_owned_by_gpu_operation_transactions`; `gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage`; `real_gpu_error_scope_captures_deliberate_validation_error`; `real_gpu_smoke_emits_no_uncaptured_error` |
| T5 | `render-window` | `presented_setup_and_resize_commit_only_after_clean_configuration`; `surface_resize_suspend_resume_and_two_surfaces_own_resources` |
| T6 | `render-window` | `presented_acquire_outcomes_map_every_surface_result_before_commit`; `presented_blit_and_present_remain_scoped_until_frame_commit` |
| T7 | `render-window` | `surface_loss_can_resume_but_device_loss_requires_a_new_renderer`; `resize_suspend_resume_and_two_surfaces_keep_device_resources_coherent` |
| T7 | `default` | `uncaptured_gpu_error_faults_only_its_device_generation`; `device_loss_is_terminal_idempotent_and_releases_device_resources`; `terminal_default_device_rejects_headless_without_disabling_ready_slots`; `destroyed_device_callback_reports_terminal_loss_without_stale_resource_use` |

**T1. Restore the wasm WebCanvas ownership boundary**
- Area/outcome: wasm-gated `src/{surface.rs,renderer.rs}`; add one crate-private `WebCanvas::html_canvas` clone accessor and make the renderer use it plus public `id()`, without exposing fields or changing native diagnostics.
- RED/acceptance: `CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --lib --target wasm32-unknown-unknown --features render-web` fails at C03 with `E0616` for direct access to private `canvas` and `id`. The same command must pass after both accesses cross their owning methods; no new public API, target branch, dependency, or browser execution claim is permitted.
- Commands: run the exact wasm compile command for RED and GREEN; then `C04-CHECK`.
- Depends: none. Commit: `Restore wasm WebCanvas ownership`.

**T2. Model exact surface states and operation preflight**
- Area/outcome: `src/{surface.rs,renderer.rs,error.rs,lib.rs}` and focused tests; add public `Empty`, private headless empty/pending/published phases, deferred headless allocation, exact resize/suspend/resume projections, and S26 operation ordering. Remove the three superseded public/backend error codes and route public conditions through validated S13 diagnostics without changing actual readback execution.
- RED/acceptance: the named nonzero-uninitialized and zero-size tests fail because C03 eagerly allocates and treats zero render as success. Pass requires no create-time headless texture, no planning/WGPU work for rejected states, same-physical resize/publication retention, size-changing invalidation, idempotent compatible transitions, exact contract-only/identity diagnostics, and no public legacy code.
- Commands: run every T2 matrix name separately with `run_exact_test`; `rg -n '(?:ErrorCode|BackendErrorCode)::(?:AdapterUnavailable|SurfaceLost|SurfaceUnavailable)' src --glob '*.rs'` exits `1`; then `C04-CHECK`.
- Depends: T1. Commit: `Model exact surface publication states`.

**T3. Publish headless frames only at the clean linearization point**
- Area/outcome: `src/{backend.rs,renderer.rs,surface.rs,gpu_transaction.rs}` and tests; allocate a distinct draft under active scopes, rasterize only into that draft, and return an explicit commit value so pixels, stats, uploaded-image state, and parameters become visible together after clean scopes/signals.
- RED/acceptance: a production-path test pauses/cancels after real submission and injects a scoped failure; C03 either writes the current readable target or lacks a publication seam. Pass preserves prior bytes/state exactly, leaves no-publication surfaces pending, drops/quarantines drafts and leases, invalidates only on changed physical size, and retains successful direct pixels with no map/poll.
- Commands: run every T3 matrix name separately with `run_exact_test`; then `C04-CHECK`.
- Depends: T2. Commit: `Make headless frame publication atomic`.

**T4. Centralize non-readback submission in GPU transactions**
- Area/outcome: `src/{gpu_transaction.rs,shader.rs,backend.rs,renderer.rs}` and tests; add stage-typed transaction submission primitives, make configure/present stages production classifications, and route every current non-readback command-buffer submission through them. Preserve internal Vello lease commit/abort semantics and the C05-owned readback submit exception.
- RED/acceptance: a transaction submission observation around the real clear/fill path reports bypass in C03. Pass proves active-generation ownership through submit and scope completion; captured validation/internal map by stage, OOM maps globally, terminal signals win, and cancellation clears ownership without public commit.
- Commands: run every T4 matrix name separately with `run_exact_test`; `rg -n 'queue\.submit' src/shader.rs src/renderer.rs` exits `1`; then `C04-CHECK`.
- Depends: T3. Commit: `Centralize scoped GPU submission`.

**T5. Make presented setup and resize configuration transactional**
- Area/outcome: feature-gated `src/{surface.rs,backend.rs,renderer.rs}` plus transaction plumbing/tests; separate desired config from committed target resources, permit zero-size/nonrenderable construction without target allocation, and configure/create replacement targets only inside a real surface-configure transaction.
- RED/acceptance: the named commit test exposes C03 constructor/configure mutation before scope resolution. Pass keeps failed resize pending, commits config/target/lifecycle only after clean scopes/signals, creates no zero-area target, preserves resizing hints, and leaves no constructor or transition method performing unscoped WGPU work.
- Commands: run every T5 matrix name separately with `run_exact_test`; then `C04-CHECK`.
- Depends: T4. Commit: `Transact presented surface configuration`.

**T6. Own acquire, blit, and present through frame commit**
- Area/outcome: feature-gated `src/{backend.rs,surface.rs,gpu_transaction.rs,renderer.rs}` and tests; map every WGPU acquire outcome, retain only a successful texture under RAII ownership, encode the intermediate-to-surface blit, submit and call `present` inside the present-stage transaction, then recheck scopes/signals before returning a frame commit.
- RED/acceptance: C03 mutates lifecycle on acquire, submits directly, and returns success for occlusion. Pass presents `Success`; safely discards and reconfigures `Suboptimal`/`Outdated` before `SurfaceOutdated`; maps timeout, occlusion, loss, validation, scoped OOM, and terminal signals exactly; discards every unpresented texture on error/cancel; preserves the attempted-present exception; publishes stats/parameters only after the post-present check; and leaves no direct present submit outside `gpu_transaction.rs`.
- Commands: run every T6 matrix name separately with `run_exact_test`; `test "$(rg -n 'queue\.submit' src/backend.rs | wc -l | tr -d ' ')" -eq 1` proves only C05's readback helper remains; then `C04-CHECK`.
- Depends: T5. Commit: `Transact presented frame publication`.

**T7. Close lifecycle, device-isolation, and static-path evidence**
- Area/outcome: `src/{backend.rs,renderer.rs,surface.rs,gpu_transaction.rs,tests.rs}`; complete lost/occluded/suspended/resume behavior, two-surface ownership, one-slot terminal isolation, uncaptured generation attribution, duplicate transition behavior, and cancellation cleanup across the integrated C04 path.
- RED/acceptance: the absent named matrix tests expose C03's incomplete presented and publication behavior. Pass covers every S26 row and S31 C04 row, retains first terminal records and healthy slots, allows surface-loss recreation but never terminal-device revival, performs no rejected-state GPU work, and leaves readback as the sole production submit/map/poll exception.
- Commands: run every T7 matrix name separately with `run_exact_test`; run T2's operation-matrix test and T3's publication test again; run the final guards below; then `C04-CHECK`.
- Depends: T6. Commit: `Close atomic surface lifecycle semantics`.

## Completion
- Require all seven ordered task ranges and clean task reviews. Headless frames expose only clean publications; presented configuration and output are transaction-owned; cancellation/failure retains prior public state; surface and device terminality are distinct; no C05 or later-cycle behavior enters the range.
- Run `C04-CHECK`, every matrix name through `run_exact_test` with its recorded profile, and require a clean worktree. Final guards are: `rg -n '^vello\s*=|^glifo\s*=' Cargo.toml`; `rg -n '\bvello::' src --glob '*.rs'`; `rg -n 'create_shader_module_trusted|queue\.submit|map_async|\.poll\(|use_cpu|CpuShader' src/vello_engine`; `rg -n 'queue\.submit' src/shader.rs src/renderer.rs`; each exits `1` clean. Require exactly one `queue.submit` in `src/backend.rs`, lexically inside `read_texture_rgba`, and every other submit in `src/gpu_transaction.rs`.
- After cycle acceptance, follow `$surgeist-agent`'s canonical implementation-cycle and automated landing/publication gates, then its crate-candidate handoff contract for C05.
- Block only for unowned worktree conflict, reviewed-packet contradiction, unavailable required native GPU evidence, missing Rust 1.97/already-authorized tooling, or unavailable already-authorized `wasm32-unknown-unknown` standard-library support. No substitute target or browser claim is allowed.
