# P02-I02-S01-C08 Renderer Orchestration

## 1 Header

- Cycle: `P02/I02/S01/C08`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base: published/read-back C07
  `a8719e7633bc6445542bb4c5d3b2ac16294b117b`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `bd25c89790358054a2b51c77c5c2b83f71859cf1`, SHA-256
  `186eb7cf9366302ea5f16476720b3fc996083ea73a0af159d7794d3b0fb13e93`;
  M01-M04, M05.3 renderer table, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `b7ce6d17a20c70dc06f68882d5347086e7c5546f`, SHA-256
  `e4b731ecb2c38543a6011402235d4e3ebc6a587d41badb876206d9f7f703d72a`;
  `C08 Renderer Orchestration`.
- Outcome: replace `src/renderer.rs` with narrow `renderer/mod.rs` plus
  `dispatch.rs`, `publication.rs`, `options.rs`, and test-only
  `test_support.rs`, preserving the public renderer front door and behavior.

## 2 Boundary

- Front door: `Renderer`, its public construction, surface lifecycle, render,
  readback, and runtime-capability methods, and only coordination requiring the
  complete renderer state remain in `renderer/mod.rs`.
- Options owner: `Options`, `EffectQualityPolicy`, `ResourceCacheBudget`, and
  `Antialiasing`, including their constructors, accessors, defaults, and docs,
  move to `renderer/options.rs`.
- Dispatch owner: route classification, pre-execution validation and gating,
  device identity selection, preparation, direct/graph execution dispatch, and
  typed unsupported-graph translation move to `renderer/dispatch.rs`.
- Publication owner: successful-frame publication, uploaded-image/statistics
  commit, and failure-atomic publication state move to
  `renderer/publication.rs`.
- Renderer test owner: renderer fixture preparation, publication fault
  controls, dispatch observations, renderer-only stage probes, and test-only
  `impl Renderer` methods move to `renderer/test_support.rs`.
- Preserve public types and root reexports, validation precedence, route
  selection, surface/device identity, typed diagnostics, graph facts,
  transaction submission, resource/cache effects, publication atomicity,
  statistics, cancellation, terminal behavior, pixels, and capability results.
- Preserve the M06 renderer edges. Imports name an owning front door or child;
  no trait/callback indirection, dynamic dispatch, duplicated state, generic
  helper, compatibility module, `include!`, or `#[path]` may disguise a cycle.
- M04.5 permits a production value and its attached test fact to move together
  only until the immediately following support task. Final production children
  and `Renderer` own no test-support dependency, fixture, fault control,
  observation aggregation, global bridge, or support callback.
- Support extraction replaces hidden scoped guards, counters, or recorders with
  explicit test-owned input at a natural renderer/backend/pass stage. Preserve
  product outcomes and public-route coverage; instrumentation-only wiring facts
  may be retired instead of forcing forbidden production coupling.
- `src/lib.rs`, `Cargo.toml`, `README.md`, `examples/`, settled private module
  hierarchies, dependencies, features, errors, and product expectations are
  protected. Only imports, necessary narrow visibility, and M04.5 test rewrites
  may vary.
- Root/sibling integration, API artifacts, unrelated cleanup, semantic change,
  crate-level test decomposition, and new retained-render architecture are
  excluded.

## 3 Effects And Evidence Policy

- API effect: none. `src/lib.rs` reexports and public renderer/options paths,
  signatures, documentation identity, defaults, and behavior remain compatible.
- Dependency/feature effect: none. `Cargo.toml` and resolved trees remain fixed.
- Behavior and oracle effect: none. This is a mechanical ownership move with
  pre/post characterization; no artificial RED applies.
- Generated-artifact effect: none. Root owns API artifacts and is excluded.
- Test effect: product conditions, operations, inputs, assertions, and oracles
  remain. Support rewrites may remove only hidden instrumentation assertions
  whose preservation would violate M04.5.
- Structural inspection is transient workflow evidence. Add no source parser,
  plan-closure test, committed inventory, generated index, file-size/count gate,
  or architectural assertion encoded as raw-source matching.
- Workers record exact pre/post focused commands, moved ownership, visibility
  changes, protected-surface diff, test dispositions, and public API effect.
  Each task is one coherent commit and receives a fresh task review before the
  next task begins.
- The active goal carries the macOS native-window exception: do not rerun a
  hanging smoke until the user requests it. All non-smoke gates remain required.

## 4 Preconditions And Landing

- `origin/main` and authority-remote `main` equal the C07 base before T01.
  Local `main` is the reviewed planning/status head descended from that base;
  the worktree is clean.
- Work remains in this leaf repository and current worktree. Do not edit root or
  siblings and do not create a separate worktree.
- Use installed tooling offline. Do not acquire or update dependencies,
  targets, toolchains, linters, or system software.
- Implementation commits land directly on leaf `main` but remain unpushed until
  all tasks, final checks, and holistic review are clean.
- Every task starts from its reviewed predecessor. A finding receives a fresh
  worker fix span and complete ordered-range task review before proceeding.
- After T08 is task-clean, make a separate status-only `complete` commit, run
  the final matrix, obtain a distinct holistic review over the exact cycle
  range, repeat the matrix unchanged, then CAS-push and read back authority
  `main`.

## 5 Ordered Tasks

### 5.1 T01 Establish Renderer Front Door And Options Owner

- Replace `src/renderer.rs` with `src/renderer/mod.rs`; add
  `src/renderer/options.rs`.
- Move all four public option types with intrinsic constructors, accessors,
  defaults, derives, and docs. Retain explicit front-door reexports so crate
  callers and `src/lib.rs` remain unchanged.
- Keep dispatch, publication, public orchestration, and renderer test support
  directly in `mod.rs`; add no forwarding compatibility layer.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render options_default_requires_high_precision_and_bounds_retention
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_cache_budget_zero_disables_idle_retention
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_budget_and_device_loss_preserve_public_stats_contract
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window create_surface_headless_preserves_surface_options
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: option APIs/defaults and root paths are unchanged; sibling owners
  import the real options owner; non-option source is a faithful move.

### 5.2 T02 Move Successful Publication Owner

- Add `renderer/publication.rs`. Move `RenderPublication`, successful render
  state commit, uploaded-image/statistics publication, and intrinsic helpers.
- Publication state stays provisional until execution succeeds; every error,
  cancellation, scope/accounting fault, and terminal path preserves the prior
  surface publication and last successful stats.
- Attached publication test facts may travel only until T03 under M04.5.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_render_reports_stats_and_failed_mask_preserves_them
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_render_operations_do_not_mutate_last_successful_stats
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_and_canceled_graph_frames_preserve_last_successful_stats
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_direct_post_submit_failure_preserves_previous_and_initial_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_render_does_not_warm_image_reuse_stats
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: one publication owner commits surface state, stats, and image
  reuse only after success; all failure-atomic outcomes are unchanged.

### 5.3 T03 Extract Publication Test Support

- Add test-gated `renderer/test_support.rs`. Move publication fault controls,
  fixtures, observations, and test-only Renderer methods attached in T02.
- Replace production-to-support callbacks, thread-local controls, or zero-input
  guards with explicit support-owned stages; retire wiring-only observations
  when required by M04.5 without weakening product oracles.
- Before and after run all T02 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_direct_cancellation_after_submit_preserves_previous_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_graph_post_submit_failure_leaves_first_frame_unpublished
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_accounting_fault_after_submit_suppresses_publication_and_commits
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_graph_after_real_submit_discards_prepared_resources_and_retries_fresh
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: publication production code imports no support, owns no control or
  observation, and the real commit/failure paths still establish every oracle.

### 5.4 T04 Move Route Classification And Typed Gating

- Add `renderer/dispatch.rs`. Move route enums, frame classification,
  pre-execution eligibility, device/surface gating used by classification, and
  typed unsupported-graph translation.
- Preserve validation precedence and exact Direct Vello, executable graph,
  fixture-only test entry, and unsupported routes. Keep test-only classification
  fixtures, observations, and entry points in `mod.rs`; T04 moves no test fact
  into `dispatch.rs`.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render renderer_dispatches_supported_graphs_and_rejects_unsupported_effects
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render renderer_public_dispatch_validates_direct_and_masked_composition_routes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render public_dispatch_routes_composition_and_color_filters_but_rejects_broad_backdrop
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render public_dispatch_routes_composition_and_spatial_filters_but_rejects_broad_backdrop
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render broad_backdrop_graph_returns_exact_unsupported_diagnostic_without_publication
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: classification invokes one route, rejects future graphs before
  resource mutation, and preserves exact typed errors and precedence.

### 5.5 T05 Move Preparation And Execution Dispatch

- Move renderer-owned device identity selection, preparation, direct/graph
  execution dispatch, and their private coordination to `dispatch.rs`.
- Retain public `Renderer::render` in `mod.rs` as the crate-facing orchestrator;
  it calls the dispatch and publication owners without duplicating either.
- Execution-attached test facts may travel with their production value only
  until the immediately following T06 support extraction under M04.5.
- Preserve one transaction, prepared resources, cache/resource accounting,
  pixels, statistics, cancellation, terminal signals, and failure order.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_stats_report_exact_route_and_single_raster_pass
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gpu_graph_stats_count_exact_backdrop_passes_copies_resources_and_precision
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_and_graph_routes_match_each_fixture_configuration_and_pixel_oracle
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_frame_returns_all_leases_and_preserves_last_successful_stats
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `dispatch.rs` owns selection through execution outcome;
  `mod.rs` remains the public orchestrator; result and failure semantics match.

### 5.6 T06 Extract Dispatch Fixture Support

- Move forced-graph, color-filter, spatial-filter, and bounded-backdrop fixture
  preparation, observations, mutations, and test-only Renderer entry points to
  `test_support.rs`.
- Replace hidden dispatch counters and production fixture branches with
  explicit test-owned inputs around real classification, preparation,
  execution, and publication stages. Retire wiring-only route counts if their
  preservation would require forbidden production coupling.
- Before and after run all T04-T05 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_fixture_executes_while_public_capability_remains_diagnostic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_fixture_executes_while_public_capabilities_remain_diagnostic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render bounded_backdrop_fixture_executes_while_broad_capabilities_remain_diagnostic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render oversized_color_filter_buffer_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_encode_and_scope_failures_preserve_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_encode_failure_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: dispatch production code owns no fixture, fault control,
  observation aggregation, test callback, or global bridge; public and fixture
  paths exercise the same real stages and preserve product outcomes.

### 5.7 T07 Complete Renderer Test Support

- Move remaining renderer-owned device/surface/readback/capability probes and
  test-only Renderer methods to `test_support.rs`.
- Remove test-only state fields from production `Renderer` when explicit
  support-owned setup or downstream observations provide the same product
  oracle. Do not duplicate backend/pass/resource test support.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render renderer_reports_backend_capabilities_by_family
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_capabilities_project_the_selected_surface_without_gpu_work
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render foreign_and_stale_surfaces_fail_before_device_slot_access
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render device_loss_is_terminal_idempotent_and_releases_device_resources
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_transaction_maps_validation_internal_oom_and_terminal_failures
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window surface_loss_can_resume_but_device_loss_requires_a_new_renderer
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: renderer support is test-only and owns only renderer-domain
  support; production `Renderer` and children contain no support wiring.

### 5.8 T08 Reconcile Narrow Renderer Orchestrator

- Reconcile `renderer/mod.rs` to child declarations, explicit reexports,
  `Renderer` state, public methods, complete-state lifecycle coordination, and
  no not-yet-moved implementation.
- Remove transitional visibility, attached test facts, unused imports, and lint
  suppressions introduced solely for staged moves. Add no shim or new API.
- Before and after run all T01-T07 focused conditions plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_readback_renderer_front_door_is_async
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render surface_operation_matrix_covers_every_kind_state_and_duplicate_transition
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render render_reports_command_stats
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
  CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
  ```

- Acceptance: all five renderer files exist; `mod.rs` is the narrow public
  orchestrator; each child owns exactly its M05.3 responsibility; public API,
  product behavior, and all allowed edges remain unchanged.

## 6 Verification And Completion

After all tasks are task-review `CLEAN`, make the status-only `complete` commit,
run this matrix, obtain a distinct holistic `CLEAN` review of the exact cycle
range, repeat the matrix at the unchanged reviewed head, and publish with
authority-remote readback:

```sh
set -euo pipefail
test -z "$(git diff a8719e7633bc6445542bb4c5d3b2ac16294b117b -- . \
  ':(exclude)src/renderer.rs' ':(exclude)src/renderer/**' \
  ':(exclude)src/tests.rs' \
  ':(exclude)plans/cycles/cohesive-module-decomposition-c08-renderer-orchestration.md')"
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
CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
rustc +1.97.0 --version
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
test -z "$(git ls-files -- Cargo.lock)"
owned_rust_files=("${(@f)$( { git ls-files -- '*.rs'; git ls-files --others --exclude-standard -- '*.rs'; } | sort -u )}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then exit 1; else test "$?" -eq 1; fi
git diff --check a8719e7633bc6445542bb4c5d3b2ac16294b117b..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The public-surface comparison is direct review of `src/lib.rs`, exported item
definitions, signatures, docs, and the base-to-head diff; it is not a parser
test. The two native smokes must render and exit when the user requests their
rerun; until then the active macOS session exception is recorded and all other
gates proceed. Every unsafe-scan match is classified; executable owned unsafe
blocks completion. Root integration remains excluded.

The C08-to-C09 handoff reports the immutable published/read-back candidate,
reviewed planning revision, task and holistic verdicts, renderer hierarchy,
public-surface comparison, test-support disposition, smoke disposition, clean
status, and explicit root-integration exclusion.
