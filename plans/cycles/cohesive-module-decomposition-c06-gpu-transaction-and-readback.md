# P02-I02-S01-C06 GPU Transaction And Readback

## 1 Header

- Cycle: `P02/I02/S01/C06`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base: `9673fcda13b614cfac3bd74f23fcf4435ec869ef`, the published
  C05 candidate verified on local and authority-remote `main`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `bd25c89790358054a2b51c77c5c2b83f71859cf1`, SHA-256
  `186eb7cf9366302ea5f16476720b3fc996083ea73a0af159d7794d3b0fb13e93`;
  sections M01-M04, M05.3 transaction table, M05.5 readback table, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `b7ce6d17a20c70dc06f68882d5347086e7c5546f`, SHA-256
  `e4b731ecb2c38543a6011402235d4e3ebc6a587d41badb876206d9f7f703d72a`;
  entry `C06 GPU Transaction And Readback`.
- Outcome: replace `src/gpu_transaction.rs` with a narrow transaction front
  door and the required graph, Vello, readback, and test-support children; then
  replace `src/readback.rs` with a narrow operation front door and the required
  layout, lifecycle, native, and test-support children without changing public
  behavior, transaction proofs, readback bytes, or test oracles.

## 2 Boundary

- Transaction shared owner: `GpuOperationStage`, `GpuOperationDraft`,
  `GpuOperationLease`, `GpuOperationTransaction`, shared scope/error
  classification, and operation-wide begin/finish/drop coordination remain in
  `gpu_transaction/mod.rs`.
- Graph owner: graph submission payload and command/resource readiness,
  accounting, host effects, graph output commit, and graph submission commit
  proof move to `gpu_transaction/graph.rs`.
- Vello owner: `InternalVelloPayload`, Vello submission, and
  `VelloResourceCommitProof` move to `gpu_transaction/vello.rs`.
- Transaction-readback owner: `ReadbackSubmission`,
  `PendingReadbackSubmission`, and their commit transition move to
  `gpu_transaction/readback.rs`.
- Transaction test owner: post-submit controls plus operation, graph, and Vello
  submission observations move to test-only
  `gpu_transaction/test_support.rs`.
- Readback layout owner: validated row layout, mapped-range validation, and
  padded-row decode move to `readback/layout.rs`.
- Readback lifecycle owner: readback phase, owner, staging disposition, cleanup,
  mapping state, and lifecycle transitions move to `readback/lifecycle.rs`.
- Readback native owner: completion state/callback, native polling/helper,
  readback future, and native platform behavior move to `readback/native.rs`.
- Readback test owner: native observations and standalone lifecycle/completion
  probes move to test-only `readback/test_support.rs`.
- `readback/mod.rs` retains only the readback operation front door and genuine
  coordination spanning layout, lifecycle, and native completion.
- Preserve transaction stages; validation, internal and out-of-memory error
  mapping; device-terminal precedence; exactly-once submission and commit;
  post-submit behavior; generation and lease ownership; graph/Vello resource
  readiness; and cancellation/drop cleanup.
- Preserve row alignment, byte counts, mapped ranges, RGBA decode order,
  staging-buffer phases and disposition, callback and wake behavior, native
  polling deadlines, future ownership, late-callback cancellation behavior,
  and failure-atomic image publication.
- Preserve the M06 transaction/backend mutual edge. Child imports must name the
  owning front door or child explicitly; no trait, callback abstraction,
  dynamic dispatch, duplicated state, compatibility module, `include!`, or
  `#[path]` bridge may disguise an ownership cycle.
- M04.5 applies only in production-move tasks T01, T03, T05, and T07: a minimal
  intrinsic test raw fact may travel only until the immediately following T02,
  T04, T06, or T08 extraction. Final production children import no test support
  and own no fixture, fault control, observation aggregation, or global bridge.
- `src/lib.rs`, `Cargo.toml`, `README.md`, `examples/`, settled hierarchies,
  backend/renderer production code, public exports, dependencies, features,
  error codes, and product expectations are protected surfaces. A test-support
  task may narrowly rewrite affected `src/tests.rs` harnesses, existing
  `pass/test_support.rs` and `resource/test_support.rs` support, and
  `#[cfg(test)]` harnesses in `backend.rs` and `renderer.rs` under M04.5. Task
  and holistic review inspect those two mixed files directly and require their
  production behavior to remain unchanged; no source-text predicate attempts
  to infer Rust configuration boundaries.
- Root integration, sibling repositories, API artifacts, unrelated cleanup,
  algorithm changes, error-policy changes, and the future backend, renderer,
  and focused-test cycles are excluded.

## 3 Effects And Evidence Policy

- API effect: none. Existing public and crate-visible paths remain explicit at
  the new module front doors; no public visibility is added.
- Dependency and feature effect: none. `Cargo.toml` and the resolved trees are
  unchanged.
- Behavior and oracle effect: none. This is a mechanical ownership move backed
  by pre/post characterization; no artificial RED applies.
- Generated-artifact effect: none. Root owns API artifacts and is excluded.
- Test effect: product conditions/outcomes and public-route coverage remain.
  Test-support tasks replace zero-argument global guards and hidden-transition
  wiring assertions with explicit test-owned setup at natural stage boundaries;
  test names continue to state their condition and observable outcome.
- Structural inspection is transient workflow evidence. Add no parser, source-
  text assertion, plan-closure test, committed inventory, ledger, generated
  index, lint, CI rule, or file-size/count gate.
- Workers record exact pre/post focused commands, moved-item ownership,
  visibility deltas, file deletion/creation, protected-surface diff, and the
  absence of production-algorithm/product-oracle changes. Each task is one
  logical commit and a separately reviewed exact range.

## 4 Ordered Tasks

### 4.1 T01 Establish Transaction Front Door And Graph Owner

- Start from the reviewed cycle base. Replace `src/gpu_transaction.rs` with
  `src/gpu_transaction/mod.rs`; retain only shared operation-stage,
  draft/lease/transaction coordination there. Move graph payload, submitted
  command/resources, readiness/accounting receipts, host effects, graph output
  commit, and graph commit transitions to `graph.rs` with the narrowest required
  visibility.
- Keep Vello, transaction-readback, and test-support implementation temporarily
  in `mod.rs`. Graph-coupled observations may remain attached under M04.5 until
  the immediately following T02; do not detach them into global state or
  redesign the graph commit path.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render multiple_composites_share_one_graph_encoder_and_transaction_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_graph_after_real_submit_discards_prepared_resources_and_retries_fresh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_graph_post_submit_failure_leaves_first_frame_unpublished
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_scope_failure_suppresses_presentation_and_commits
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: the old file is gone; graph production ownership is in
  `graph.rs`; `mod.rs` contains shared transaction coordination rather than a
  copied graph implementation; graph submission, readiness, commit, failure,
  cancellation, and publication observations are identical.
- Intended commit: one transaction-front-door/graph-owner move.

### 4.2 T02 Extract Graph Transaction Test Support

- Start only from the reviewed T01 head. Create test-gated
  `gpu_transaction/test_support.rs` and move graph submission observations,
  graph post-submit controls, scoped guards, checkpoints, and graph recorders
  out of `graph.rs`; also move separable operation-wide test support.
- Production files retain only minimal intrinsic per-value raw facts/accessors;
  they do not import test support or own a fault control, observation
  model/aggregation, or global bridge.
- Rewrite affected graph tests to pass explicit test-owned inputs around natural
  transaction stage boundaries. Retire only hidden timing/identity/wiring
  assertions whose sole implementation is the forbidden production callback;
  preserve product failures, cleanup, publication effects, and public-route
  coverage through real production transitions.
- Before and after, run all T01 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render post_submit_scope_failure_discards_prepared_resources_with_nonzero_budget
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: graph/operation test support is test-only immediately after the
  graph move; graph/shared transaction owners have no test-support dependency
  or global bridge; focused product conditions and outcomes are preserved
  without simulated results or production fault-control fields.
- Intended commit: one graph-transaction-test-support extraction.

### 4.3 T03 Move Vello And Transaction-Readback Owners

- Start only from the reviewed T02 head. Move internal Vello payload,
  submission, and resource commit proof to `vello.rs`. Move pending/committed
  readback submission facts and transitions to `readback.rs`. Reconcile shared
  transaction methods in `mod.rs` through explicit child contracts.
- Preserve transaction generation, scope resolution, queue submission,
  post-submit ordering, lease commit/discard, device-signal precedence, and
  readback submission index. Vello-coupled observations may remain attached
  only until the immediately following T04.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_readback_gpu_submissions_are_owned_by_gpu_operation_transactions
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render dropped_gpu_operation_future_aborts_draft_state_and_leases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_render_submits_one_transaction_owned_raster_pass
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render internal_vello_encoding_shares_the_frame_transaction_submission
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_transaction_maps_validation_internal_oom_and_terminal_failures
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: all four production transaction files exist; each M05.3 owner is
  distinct; shared coordination remains narrow; operation, Vello, and readback
  submission behavior and observations are identical; no new crate edge exists.
- Intended commit: one Vello/readback transaction-owner move.

### 4.4 T04 Complete Transaction Test Support And Reconcile Front Door

- Start only from the reviewed T03 head. Move remaining Vello observation
  models/aggregation, post-submit controls, scoped guards, checkpoints, and
  recorders to the existing test-gated `gpu_transaction/test_support.rs`;
  reconcile any remaining operation-wide support there.
- Leave production children only minimal intrinsic per-value raw facts or
  accessors that cannot be derived from production state without changing
  semantics. Production children may not import test support or own a fixture,
  fault control, observation model/aggregation, or global bridge.
- Apply the M04.5 explicit-harness replacement to affected Vello/operation tests;
  preserve product outcomes and public-route coverage, not hidden wiring.
- Before and after, run all T01 and T03 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_direct_post_submit_failure_preserves_previous_and_initial_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render post_submit_scope_failure_discards_prepared_resources_with_nonzero_budget
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render terminal_signal_after_transaction_completion_preserves_public_frame_state
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_post_transaction_terminal_signal_commits_current_frame_and_fails_next_operation
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: `test_support.rs` is compiled only for tests; the front door is
  shared transaction coordination plus explicit reexports; production children
  have no test-support dependency or behavior-changing test control; the full
  default suite and all focused product outcomes remain equivalent.
- Intended commit: one transaction-test-support reconciliation.

### 4.5 T05 Establish Readback Front Door, Layout, And Lifecycle Owners

- Start only from the reviewed T04 head. Replace `src/readback.rs` with
  `src/readback/mod.rs`. Move row-layout construction/validation, mapped-range
  validation, and padded-row decode to `layout.rs`. Move phase, owner, staging
  disposition, cleanup, map state, and lifecycle transitions to `lifecycle.rs`.
- Keep native completion/polling/future and readback test support temporarily in
  `mod.rs`. Lifecycle-coupled test facts may remain attached under M04.5 until
  the immediately following T06; row bytes, ranges, cleanup actions, and state
  transitions do not change.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_map_callback_publishes_once_and_wakes_latest_waker
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_render_can_be_read_back
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render high_precision_low_alpha_pixels_preserve_straight_rgb
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render nonzero_headless_read_before_publication_reports_uninitialized_without_map
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: the old file is gone; layout and lifecycle responsibilities have
  one owner each; `mod.rs` retains the operation front door and temporarily
  staged native/test support only; decoded bytes, state transitions, cleanup,
  waking, and publication behavior are unchanged.
- Intended commit: one readback-front-door/layout/lifecycle move.

### 4.6 T06 Extract Readback Lifecycle Test Support

- Start only from the reviewed T05 head. Create test-gated
  `readback/test_support.rs`; move lifecycle probes/observations/aggregation out
  of production. Native observations remain with the staged operation until T07.
- Replace any lifecycle global guard with explicit test-owned stage setup while
  preserving mapped/failed/canceled cleanup outcomes.
- Before and after, run all T05 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: lifecycle support is test-only immediately after its production
  move; layout/lifecycle have no test-support dependency or global bridge;
  native observations remain attached to their staged native operation.
- Intended commit: one readback-lifecycle-test-support extraction.

### 4.7 T07 Move Native Readback Owner

- Start only from the reviewed T06 head. Move completion callback/state, native
  polling decision and helper, helper/callback ownership, `ReadbackMapFuture`,
  and native completion behavior to `native.rs`. Keep the operation entry point
  in `mod.rs` and communicate through explicit layout/lifecycle/native facts.
  Native-coupled test facts may remain attached only until the immediately
  following T08.
- Preserve callback-at-most-once behavior, latest-waker replacement, native
  polling deadline, helper lifetime, cancel/drop behavior, late callback
  discard, staging cleanup, and diagnostic text/error conditions.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render native_readback_callback_progresses_and_cleans_up_with_diagnostic_deadline
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_native_readback_discards_late_callback_without_publication_change
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_map_callback_publishes_once_and_wakes_latest_waker
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_render_can_be_read_back
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
  ```

- Acceptance: native ownership is in `native.rs`; `mod.rs` has no callback,
  poll-helper, or future implementation; native and wasm compilation plus all
  focused completion/cancellation behavior and diagnostics are unchanged.
- Intended commit: one native-readback-owner move.

### 4.8 T08 Complete Readback Test Support And Reconcile Front Door

- Start only from the reviewed T07 head. Move native observation models,
  standalone state-machine/completion probes, scoped guards, test lifetimes,
  and test-only aggregation to test-gated `readback/test_support.rs`.
- Reconcile `readback/mod.rs` to test-gated child declaration/reexports, the
  readback operation entry point, and only genuine coordination spanning
  production children. Apply the same final production/test boundary as T04.
- Replace native hidden-transition wiring with explicit test-owned stage setup;
  preserve callback, cancellation, cleanup, diagnostic, and publication outcomes.
- Before and after, run all T05 and T07 focused tests plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_transaction_maps_validation_internal_oom_and_terminal_failures
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_native_readback_discards_late_callback_without_publication_change
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
  ```

- Acceptance: all five readback files exist; test support is test-only;
  production children have no test-support dependency, fault control,
  observation model/aggregation, or global bridge; `mod.rs` is a narrow readback
  operation front door; focused/default/window behavior and oracles are
  identical.
- Intended commit: one readback-test-support/front-door reconciliation.

## 5 Verification And Completion

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
test -z "$(git diff 9673fcda13b614cfac3bd74f23fcf4435ec869ef -- \
  src/lib.rs Cargo.toml README.md examples src/frame src/shader \
  ':(top)src/pass' ':(top,exclude)src/pass/test_support.rs' \
  ':(top)src/resource' ':(top,exclude)src/resource/test_support.rs')"
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
git diff --check 9673fcda13b614cfac3bd74f23fcf4435ec869ef..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

Both native smoke executables must render and exit on the native host. If a
known macOS session condition causes an environmental failure, record that
single result and follow the active goal note rather than repeatedly rerunning
it; all non-smoke gates remain mandatory. Every unsafe-scan match is classified;
any executable match blocks completion. The publication head is immutable after
holistic review. Root integration remains excluded.

The C06-to-C07 leaf handoff reports the immutable published C06 candidate and
authority-remote readback SHA, the exact reviewed planning revision, clean task
and holistic verdicts, the stable transaction/readback private hierarchies,
clean status, and explicit exclusion of root integration.
