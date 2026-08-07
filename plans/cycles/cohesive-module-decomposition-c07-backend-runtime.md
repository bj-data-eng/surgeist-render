# P02-I02-S01-C07 Backend Runtime

## 1 Header

- Cycle: `P02/I02/S01/C07`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base: published/read-back C06 `d138f05d83a6739ebebe64d3146154c66fa58a47`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `bd25c89790358054a2b51c77c5c2b83f71859cf1`, SHA-256 `186eb7cf9366302ea5f16476720b3fc996083ea73a0af159d7794d3b0fb13e93`;
  M01-M04, M05.3 backend table, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `b7ce6d17a20c70dc06f68882d5347086e7c5546f`, SHA-256 `e4b731ecb2c38543a6011402235d4e3ebc6a587d41badb876206d9f7f703d72a`;
  `C07 Backend Runtime`.
- Outcome: replace `src/backend.rs` with narrow `backend/mod.rs`, `device.rs`,
  `execute.rs`, `offscreen.rs`, `present.rs`, `texture.rs`, and test-only
  `test_support.rs`, preserving crate-visible paths and runtime behavior.

## 2 Boundary

- Front door: `Backend`, its instance/device slots/cache budget, and only
  coordination requiring complete backend state remain in `backend/mod.rs`.
- Device owner: `DeviceState`, ready/terminal lifecycle, device-slot identity,
  capabilities/signals/callbacks, compatible selection facts, and device/queue
  access move to `backend/device.rs`.
- Texture owner: safe descriptor realization and shared headless/target texture
  construction move to `backend/texture.rs`.
- Offscreen owner: local-scene request/context, bounded target construction,
  managed lease/cleanup, and local rendering move to `backend/offscreen.rs`.
- Presented owner: presented device selection, surface creation/configuration,
  acquisition mapping, target validation, resize/recovery, blit, and presentation
  move to `backend/present.rs`.
- Execution owner: exact graph selection, direct Vello and prepared-graph
  execution, targets, encoding/submission, timings, and commit results move to
  `backend/execute.rs`.
- Backend test owner: backend-owned failure controls, fixtures, observations,
  stage harnesses, and test-only `impl Backend` methods move to `test_support.rs`.
- Preserve device generation/terminal precedence, capabilities/formats,
  transactions, cache/resources, pass order, targets, acquire/configure state,
  presentation, publication, statistics, cleanup, cancellation, and atomicity.
- Preserve M06 backend edges. Imports name the owning front door or child; no
  trait/callback indirection, dynamic dispatch, duplicated state, generic helper,
  compatibility module, `include!`, or `#[path]` may disguise a cycle.
- M04.5 applies only in production-move tasks T01, T03, T05, and T07. A minimal
  fact may travel only until T02, T04, T06, or T08. Final production children
  own no test-support dependency, fixture, control, aggregation, registry, guard,
  or support callback.
- A support task replaces hidden zero-argument guards or production recorders
  with explicit test-owned inputs at the natural stage. Product outcomes, error
  mapping, resource/publication effects, and public-route coverage remain;
  instrumentation-only timing/identity/count/wiring assertions may be retired.
- `src/lib.rs`, `Cargo.toml`, `README.md`, `examples/`, settled private
  hierarchies, `src/renderer.rs`, exports, dependencies, features, errors, and
  product expectations are protected. Only narrow M04.5 test rewrites may vary.
- Root/sibling integration, API artifacts, unrelated cleanup, semantic changes,
  renderer decomposition, and focused-test cycles are excluded.

## 3 Effects And Evidence Policy

- API effect: none; `src/lib.rs` and crate-visible backend paths remain compatible.
- Dependency/feature effect: none; `Cargo.toml` and resolved trees are unchanged.
- Behavior and oracle effect: none. This is a mechanical ownership move backed
  by pre/post characterization; no artificial RED applies.
- Generated-artifact effect: none. Root owns API artifacts and is excluded.
- Test effect: product conditions, outcomes, and public-route coverage remain.
  Support tasks replace hidden backend wiring with explicit test-owned stages;
  test names continue to state their triggering condition and observable result.
- Structural inspection is transient workflow evidence. Add no parser,
  source-text assertion, plan-closure test, committed inventory, ledger,
  generated index, lint, CI rule, or file-size/count gate.
- Workers record exact pre/post focused commands, moved ownership, visibility
  changes, file creation/deletion, protected-surface diff, feature gates, and
  every test disposition. Each task is one coherent commit and receives a
  separate task review before the next task begins.

## 4 Preconditions And Landing

- Local `main`, `origin/main`, and authority-remote `main` must equal the cycle
  base before T01.
- The worktree must be clean. The implementation stays in this repository and
  current worktree; no root or sibling repository is edited.
- Use installed tooling offline. Do not acquire or update dependencies, targets,
  toolchains, linters, or system software.
- Implementation commits land directly on leaf `main` but are not pushed until
  all tasks, final checks, and holistic review are clean.
- Each task starts from the reviewed predecessor head. A finding is fixed by a
  fresh worker span and complete ordered-range task review before proceeding.
- After T08 is clean, the coordinator makes a separate status-only `complete`
  commit, runs the final matrix, obtains a distinct holistic review over the
  exact cycle range, repeats the matrix at the unchanged reviewed head, and
  publishes with a compare-and-swap push plus authority-remote readback.

## 5 Ordered Tasks

### 5.1 T01 Establish Backend Front Door And Device Owner

- Start only from the published C06 base. Replace `src/backend.rs` with
  `src/backend/mod.rs`; add `src/backend/device.rs`.
- Move device state/lifecycle, ready resources, slot identity, capabilities,
  terminal signal/state, callback registration, compatible-device selection,
  device creation, terminal observation, and device/queue access to `device.rs`.
- Keep not-yet-moved execution, offscreen, presentation, texture, and backend
  support directly in `mod.rs`. Device-coupled test facts may travel attached
  only until T02 under M04.5.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render surgeist_device_state_owns_selected_wgpu_handles
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render device_loss_is_terminal_idempotent_and_releases_device_resources
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render uncaptured_gpu_error_faults_only_its_device_generation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render terminal_device_cleanup_drops_internal_engine_resources
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render one_ready_device_owns_one_raster_and_effect_resource_manager
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_capabilities_project_the_selected_surface_without_gpu_work
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
  ```

- Acceptance: the old file is gone; `device.rs` owns each named device concern;
  the front door retains only the backend state and staged later-cycle owners;
  device generations, terminal precedence, capabilities, callbacks, resources,
  and crate-visible paths are unchanged.
- Intended commit: one backend-front-door/device-owner move.

### 5.2 T02 Extract Device Test Support

- Start only from the reviewed T01 head. Create test-gated
  `backend/test_support.rs`; move device drop witnesses, borrows, callbacks,
  fault controls, generation probes, cache/resource observations, and test-only
  backend device helpers out of production ownership.
- Replace any device global guard or production recorder with an explicit
  test-owned device/signal stage while preserving loss, fault, cleanup,
  generation, capability, and resource outcomes.
- Run all T01 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render destroyed_device_callback_reports_terminal_loss_without_stale_resource_use
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render terminal_default_device_rejects_headless_without_disabling_ready_slots
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: device support is test-only; production `device.rs` imports no
  support and owns no fault control, observation aggregation, global bridge, or
  support callback; device product oracles remain unchanged.
- Intended commit: one device-test-support extraction.

### 5.3 T03 Move Texture And Offscreen Owners

- Start only from the reviewed T02 head. Add `backend/texture.rs` and
  `backend/offscreen.rs`.
- Move safe backend texture realization to `texture.rs`. Move offscreen context,
  request, target, managed lease/drop/release, descriptor/physical-size
  validation, and local Vello scene rendering to `offscreen.rs`.
- Preserve exact extent/format/usage/role, allocation diagnostics, resource
  accounting, lease cleanup, bounded local-scene output, and zero-allocation
  behavior. Offscreen-coupled acquisition observation may travel attached only
  until T04.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_texture_descriptor_uses_allocation_size_without_surface_rewrite
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_texture_allocation_uses_explicit_bounded_layer_descriptor
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_texture_rejects_missing_gpu_context_with_adapter_diagnostic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_local_vello_scene_renders_to_texture_when_gpu_context_is_available
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render explicit_offscreen_release_reports_accounting_fault_while_drop_remains_nonpanicking
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_local_scene_texture_descriptor_rejects_bgra8_for_vello_target
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: texture construction and offscreen ownership are in their named
  children; resource/lease/accounting semantics and output bytes are unchanged;
  `mod.rs` no longer owns those implementations.
- Intended commit: one texture/offscreen owner move.

### 5.4 T04 Extract Offscreen Test Support

- Start only from the reviewed T03 head. Move offscreen fixtures, accounting
  probes, and acquisition observations to `backend/test_support.rs`.
- Remove `ACTIVE_OFFSCREEN_TEXTURE_ACQUIRE_OBSERVATION_FOR_TEST`, its
  zero-argument scoped guard, and production recorder. Rewrite affected tests
  with explicit request/resource/publication facts at the offscreen stage;
  retain allocation/no-allocation, lease, cleanup, accounting, bytes, and route
  outcomes while retiring only hidden call-count wiring.
- Run all T03 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_no_allocation_when_layer_isolation_is_unnecessary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render offscreen_reuses_resources_across_repeated_bounded_requests
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_output_matches_ordinary_scene_baseline
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: production texture/offscreen children have no support dependency,
  global observation, count recorder, or fault control; explicit real stages
  preserve product-visible offscreen outcomes.
- Intended commit: one offscreen-test-support reconciliation.

### 5.5 T05 Move Presented Runtime Owner

- Start only from the reviewed T04 head. Add `backend/present.rs`.
- Move presented-compatible device selection, surface creation, configuration,
  acquisition result mapping, presented target validation, resize/recovery
  coordination, blit, and host presentation to `present.rs`.
- Preserve configuration draft/commit atomicity, acquisition error mapping,
  surface lifecycle, device-slot choice, terminal precedence, exactly-once host
  presentation, and suppression after failure/cancellation. Presented configure
  controls may travel attached only until T06.
- Before and after, run with `render-window`:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_setup_and_resize_commit_only_after_clean_configuration
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_acquire_outcomes_map_every_surface_result_before_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_blit_and_present_remain_scoped_until_frame_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window surface_resize_suspend_resume_and_two_surfaces_own_resources
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window surface_loss_can_resume_but_device_loss_requires_a_new_renderer
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_resume_prefers_installed_compatible_slot_over_earlier_donor_slot
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
  ```

- Acceptance: presented ownership is in `present.rs`; acquisition/configuration,
  lifecycle, terminal, and presentation behavior is identical on native/window
  and wasm compilation paths; the front door owns no presented implementation.
- Intended commit: one presented-runtime-owner move.

### 5.6 T06 Extract Presented Test Support

- Start only from the reviewed T05 head. Move presented failure controls,
  configuration observations, display-free compatibility fixtures, acquisition
  fixtures, and test-only presented backend methods to `test_support.rs`.
- Remove the active configure-control and display-free incompatibility
  thread-locals, zero-argument scoped guards, and production callbacks. Replace
  bridge-dependent harnesses with explicit test-owned configuration/acquisition
  stages. Preserve configuration failure mapping, no-publication guarantees,
  resume/donor selection, recovery, and presentation outcomes; retire only
  hidden checkpoint/count/identity wiring.
- Run all T05 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window planner_failure_precedes_pending_presented_surface_configuration
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_acquire_error_leaks_no_prepared_or_public_state
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_scope_failure_suppresses_presentation_and_commits
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_cancellation_after_submit_discards_without_presentation
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
  ```

- Acceptance: production `present.rs` imports no support and owns no global
  control, observation registry, or support callback; real lifecycle,
  acquisition, transaction, resource, and publication outcomes remain covered.
- Intended commit: one presented-test-support reconciliation.

### 5.7 T07 Move Direct And Graph Execution Owner

- Start only from the reviewed T06 head. Add `backend/execute.rs`.
- Move `ExactSurfaceGraph`, internal Vello requests, direct and prepared-graph
  execution, preparation/encoding/submission handoff, exact headless/presented
  targets, frame timings, `SurfaceFrameCommit`, statistics application, and
  publication commit results to `execute.rs`.
- Preserve direct/graph route behavior, operation stages, exact target facts,
  cache/resource commit, transaction submission, pass order, statistics,
  presentation authorization, headless publication, cancellation, and failure
  atomicity. Execution-coupled observations may travel attached only until T08.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render multiple_color_runs_share_one_graph_encoder_and_transaction_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_scene_uses_one_pass_and_no_effect_allocation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_direct_post_submit_failure_preserves_previous_and_initial_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_graph_post_submit_failure_leaves_first_frame_unpublished
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_graph_after_real_submit_discards_prepared_resources_and_retries_fresh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_output_specializes_rgba_and_bgra_without_channel_swap
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_terminal_loss_suppresses_presentation_and_transitions_device
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: execution ownership is in `execute.rs`; front door and other
  children own no graph/direct executor or commit-result implementation;
  transaction, resource, publication, statistics, and pixel outcomes are
  unchanged.
- Intended commit: one direct/graph-execution-owner move.

### 5.8 T08 Complete Backend Test Support And Reconcile Front Door

- Start only from the reviewed T07 head. Move execution fixtures, malformed
  plans, failure injections, observations, aggregation, and remaining test-only
  backend methods to `backend/test_support.rs`.
- Reconcile `backend/mod.rs` to child declarations, explicit crate-visible
  reexports, `Backend` state, and only genuine complete-state coordination.
- Replace remaining hidden-transition wiring with explicit test-owned stages;
  preserve direct/graph errors, order, resource/cache/publication effects,
  pixels, statistics, cancellation, retry, and terminal outcomes.
- Before and after, run all T01-T07 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render oversized_color_filter_buffer_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_encode_and_scope_failures_preserve_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_encode_failure_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render internal_vello_encoding_shares_the_frame_transaction_submission
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
  ```

- Acceptance: all seven backend files exist; test support is test-only; no
  production child depends on support or owns a fixture, fault control,
  observation model/aggregation, global bridge, or support callback; `mod.rs`
  is a narrow backend state front door; focused/default/window behavior and
  product oracles are unchanged.
- Intended commit: one backend-test-support/front-door reconciliation.

## 6 Verification And Completion

Each task records passing pre-move characterization and identical post-move
operation/oracle results. Module ownership is assessed directly in task and
holistic review, not encoded as a parser or closure gate. Each task requires a
separate task-review `CLEAN` verdict.

After all tasks are clean, the coordinator makes a status-only `complete`
commit, runs this matrix, obtains a distinct holistic `CLEAN` review over the
exact cycle range, repeats the matrix at the unchanged reviewed head, and
publishes with authority-remote readback:

```sh
set -euo pipefail
test -z "$(git diff d138f05d83a6739ebebe64d3146154c66fa58a47 -- \
  src/lib.rs Cargo.toml README.md examples src/frame src/pass src/shader \
  src/resource src/gpu_transaction src/readback src/style src/reference \
  src/renderer.rs src/surface.rs src/texture.rs src/vello_engine)"
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
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
rustc +1.97.0 --version
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4
test -z "$(git ls-files -- Cargo.lock)"
owned_rust_files=("${(@f)$(
  {
    git ls-files -- '*.rs'
    git ls-files --others --exclude-standard -- '*.rs'
  } | sort -u
)}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then
  exit 1
else
  test "$?" -eq 1
fi
git diff --check d138f05d83a6739ebebe64d3146154c66fa58a47..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

Both native smoke executables should render and exit on the native host. The
active goal records the current macOS window/session exception: if the smoke
hang recurs, record it once, do not rerun it until the user requests, and
continue all non-smoke implementation and verification work. Every unsafe-scan
match is classified; any executable match blocks completion. The publication
head is immutable after holistic review. Root integration remains excluded.

The C07-to-C08 leaf handoff reports the immutable published C07 candidate and
authority-remote readback SHA, the exact reviewed planning revision, clean task
and holistic verdicts, the stable backend front door and runtime children,
clean status, the recorded native-smoke disposition, and explicit exclusion of
root integration.
