# P02-I02-S01-C10 Runtime Platform Test Hierarchy And Reconciliation

## 1 Header

- Cycle: `P02/I02/S01/C10`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: published/read-back C09
  `1e57d07d2595be95949caeff7b76a573a457723a`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `bd25c89790358054a2b51c77c5c2b83f71859cf1`, SHA-256
  `186eb7cf9366302ea5f16476720b3fc996083ea73a0af159d7794d3b0fb13e93`;
  M01-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `b7ce6d17a20c70dc06f68882d5347086e7c5546f`, SHA-256
  `e4b731ecb2c38543a6011402235d4e3ebc6a587d41badb876206d9f7f703d72a`;
  `C10 Runtime Platform Test Hierarchy And Reconciliation`.
- Outcome: complete M05.6 by moving the C09 remainder into private `gpu`,
  `surface`, `platform`, and `vello` test domains, reconciling cross-domain
  support, and reducing `tests/mod.rs` to declarations and genuine suite-level
  coordination without changing behavior or the public surface.

## 2 Boundary

- Work is mechanical and test-only. It may change `src/tests/mod.rs`, add
  `src/tests/{gpu,surface,platform,vello}.rs`, and reconcile
  `src/tests/{model,style,frame,support}.rs` only when a real caller or helper
  owner requires it.
- The immutable C09 handoff classifies the `mod.rs` remainder as 371 tests and
  226 helpers: GPU 126/100, resource/cache 74/31, transaction 24/4,
  surface/publication/cancellation/readback 48/28, platform 11/0, and
  Vello/runtime execution 88/63. The categories are a direct disposition
  inventory and may overlap only at helpers that become `support.rs` items;
  every test has exactly one final child owner.
- Base test evidence is 671 raw `#[test]` attributes plus one proptest case.
  Default and `render-web` compile 637 leaf names with SHA-256
  `e833ab0a4e876abbcb909fb5c52a71f11946c519ca2bceba5866faf110bd01d7`;
  `render-window` and combined compile 670 leaf names with SHA-256
  `eabeb5e28335d8f0918904a814eaae1158dab272975bc10b734d473a707087df`.
  Workers compare these transiently; no inventory is committed.
- A test moves by its owned observable: shader/graph execution, pixels,
  precision, resources, caches, and transactions belong to `gpu.rs`;
  headless/presented lifecycle, publication, cancellation, and readback belong
  to `surface.rs`; target, feature, and host-example behavior belongs to
  `platform.rs`; internal Vello recording, characterization, and route parity
  belongs to `vello.rs`.
- A helper with one final-domain caller moves with that caller. A helper remains
  in `support.rs` only with direct callers in at least two sibling test domains;
  it is `pub(super)` and named for its domain operation. Do not create a generic
  utility owner or keep transitional visibility.
- Test names, cfgs, operations, inputs, assertions, timing semantics, oracles,
  and fixture bytes remain unchanged except the smallest module-path and
  visibility repair required by the move. Do not alter production code.
- No source parser, plan-closure test, plan identifier or path outside `plans/`,
  committed inventory, generated index, file/item/line-count gate, forwarding
  wrapper, `include!`, `#[path]`, glob import, compatibility shim, duplicate
  helper, or broad lint suppression is allowed.
- Root, siblings, adapters, API artifacts, gitlinks, the public hierarchical
  front-door initiative, algorithms, dependencies, features, and generated
  artifacts are excluded.
- The active macOS exception remains controlling: do not execute either native
  `render_window_smoke` cargo-run command until the user requests it. Record
  both as deferred and run every other configured gate. Implementation and
  task review continue, but final holistic review and publication stop until
  the user authorizes both commands and both render and exit successfully.

## 3 Impacts And Preconditions

- Public API: internal-only and source-compatible; `src/lib.rs`, public items,
  reexports, docs, defaults, and diagnostics do not change.
- Dependencies/features: unchanged; all four supported feature configurations
  remain verification inputs.
- Generated artifacts, docs/examples, and migration: none. The example source
  does not change.
- MSRV: unchanged; installed Rust 1.97 remains required.
- Unsafe: no Surgeist-owned unsafe may be introduced or retained.
- Root follow-up: none in this cycle. Return only the published leaf candidate;
  root integration remains separately owned and excluded.
- Before T01, local `main` is clean and equals `origin/main` at the cycle base.
  Every task depends on the reviewed preceding task head and contributes one
  logical relocation commit.
- Behavior-preserving relocation uses pre/post characterization; artificial RED
  is not applicable. Each focused condition must execute before and after its
  task and retain the same observable outcome.

## 4 Tasks

### 4.1 T01 Establish GPU Shader And Graph-Execution Ownership

- Add `src/tests/gpu.rs` and move shader parameter bytes, key/layout/cache
  realization, prepared-pass encoding, executable graph ordering, and related
  GPU capability tests plus their single-domain helpers from `mod.rs`.
- Keep CPU pixel/reference oracles and precision cases for T02; resource/cache
  manager and transaction-lifecycle cases for T03; surface, platform, and
  Vello-owned cases remain in `mod.rs`.
- Dependency/intended commit: reviewed C10 plan head; one GPU front-door and
  shader/graph-execution relocation commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_graph_encodes_blur_and_drop_shadow_in_authored_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render shader_clear_fill_pass_encodes_when_gpu_context_is_available
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: `gpu.rs` owns shader and executable-graph tests without absorbing
  transaction, surface, platform, or Vello lifecycle owners; names and behavior
  match the base.

### 4.2 T02 Complete GPU Pixel And Precision Ownership

- Move CPU reference buffers used as GPU pixel oracles, image/filter/mask/blend
  result cases, edge and precision policies, known-vector comparisons, and live
  graph pixel execution cases into `gpu.rs` with single-domain helpers.
- Internal Vello recorder characterization and direct-versus-graph route parity
  remain for T06 even when they compare pixels.
- Dependency/intended commit: reviewed T01 head; one pixel/precision relocation
  commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render reference_color_filter_partial_ops_match_deterministic_bytes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render blur_impulse_is_symmetric_normalized_and_matches_oracle
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render high_precision_color_functions_match_cpu_oracle_for_boundary_pixels
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resolved_alpha_mask_low_medium_high_and_extend_modes_match_boundary_oracle
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render plus_blend_clamps_high_precision_results
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: all GPU pixel and precision oracles have one owner; Vello parity
  remains unmoved; compiled leaf-name inventories match.

### 4.3 T03 Complete GPU Resource, Cache, And Transaction Ownership

- Move resource identity/accounting/leasing/retention, texture/effect/resource
  caches, GPU operation transaction stages, submission/commit, graph encoding
  transaction behavior, and transaction-owned cancellation cases into `gpu.rs`.
- Shader key/layout/pipeline cache realization is already owned by T01 and does
  not move in this task.
- Surface publication/readback cancellation stays for T04; Vello atlas and
  direct-render transaction characterization stays for T06.
- Dependency/intended commit: reviewed T02 head; one resource/cache/transaction
  relocation commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_leases_reject_stale_generation_and_double_release_by_model
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render texture_cache_release_and_eviction_accounting_is_deterministic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_readback_gpu_submissions_are_owned_by_gpu_operation_transactions
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render canceled_generic_submission_after_real_submit_clears_ownership_without_public_result
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_preparation_is_allocation_safe_and_submission_free
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `gpu.rs` completely owns the M05.6 GPU domain; surface and Vello
  lifecycle cases remain for their children; helpers and inventories match.

### 4.4 T04 Establish Surface Lifecycle And Publication Ownership

- Add `src/tests/surface.rs` and move headless/presented creation, resize,
  suspend/resume, device/surface loss, publication atomicity, cancellation,
  readback state/future/callback, statistics publication, and surface-owned
  renderer dispatch cases with their single-domain helpers.
- Feature/target diagnostics and host-facing example conditions remain for T05;
  Vello internal transaction/atlas cases remain for T06.
- Dependency/intended commit: reviewed T03 head; one surface-domain relocation
  commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_direct_cancellation_after_submit_preserves_previous_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render terminal_signal_after_successful_headless_publication_preserves_frame_state
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window surface_operation_matrix_covers_every_kind_state_and_duplicate_transition
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_graph_cancellation_after_submit_discards_without_presentation
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
  ```

- Acceptance: surface lifecycle, publication, cancellation, and readback have
  one child owner without swallowing feature/target or Vello internals.

### 4.5 T05 Establish Platform And Feature Ownership

- Add `src/tests/platform.rs` and move the eleven target/feature/host-example
  conditions from the C09 disposition inventory, including off-wasm/wasm web
  capability, native GPU/presentation diagnostics, and the five feature-gated
  window-smoke conditions. Move only their single-domain helpers.
- This task runs feature-gated test functions; it does not execute the deferred
  native example cargo-run commands.
- Dependency/intended commit: reviewed T04 head; one platform-domain relocation
  commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web vello_baseline_reports_web_canvas_surface_as_unsupported_off_wasm_web
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web unsupported_web_canvas_attachment_reports_target_requirement
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_bgra8_remains_a_surface_create_diagnostic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render real_gpu_smoke_emits_no_uncaptured_error
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window render_window_smoke_executes_direct_and_graph_presented_frames
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true RUSTFLAGS="-D warnings" cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
  ```

- Acceptance: all eleven platform conditions have one owner and their cfgs are
  unchanged; native example execution remains explicitly deferred.

### 4.6 T06 Establish Internal Vello Ownership

- Add `src/tests/vello.rs` and move internal Vello recording/preparation,
  atlas/recovery, direct-render characterization, retained-engine behavior,
  and direct-versus-graph route/pixel parity cases with their single-domain
  helpers.
- Do not move generic GPU shader/resource/transaction tests or surface
  publication behavior merely because Vello participates in the path.
- Dependency/intended commit: reviewed T05 head; one Vello-domain relocation
  commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_pixels_match_characterization_cases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_and_graph_routes_match_each_fixture_configuration_and_pixel_oracle
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render internal_vello_msaa8_mask_lut_ties_are_tile_translation_invariant
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render repeated_direct_renders_keep_internal_vello_retention_bounded
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `vello.rs` owns only internal Vello characterization/parity;
  generic GPU and surface contracts stay with their owners.

### 4.7 T07 Reconcile The Complete Test Hierarchy

- Reconcile all nine test files. Reduce `mod.rs` to declarations, imports, and
  genuine suite-level coordination only; every remaining test/helper receives
  an explicit final disposition. Remove transitional imports/visibility and
  move every single-domain helper out of `support.rs`.
- Record the final test/helper disposition and support caller inventory only in
  task/handoff evidence. Commit no inventory, parser, enforcement test, or size
  fact.
- Dependency/intended commit: reviewed T06 head; one hierarchy reconciliation
  commit.
- Before and after run all T01-T06 focused conditions plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render headless_render_can_be_read_back
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window render_window_smoke_executes_masked_and_blended_graph_frames
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_and_graph_routes_match_each_fixture_configuration_and_pixel_oracle
  CARGO_NET_OFFLINE=true cargo fmt --check
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

- Acceptance: M05.6 is complete; every test/helper has one final owner;
  `support.rs` contains only proven multi-sibling fixtures/oracles; the base and
  final test inventories and observable behavior are identical.

## 5 Verification And Completion

After all tasks are task-review `CLEAN`, make the status-only `complete` commit,
run this matrix, obtain a distinct holistic `CLEAN` review, repeat at unchanged
HEAD, and CAS-publish with authority readback:

```sh
set -euo pipefail
test -z "$(git diff 1e57d07d2595be95949caeff7b76a573a457723a -- . \
  ':(exclude)src/tests/**' \
  ':(exclude)plans/cycles/cohesive-module-decomposition-c10-runtime-platform-test-hierarchy.md')"
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
owned_rust_files=("${(@f)$( { git ls-files -- '*.rs'; git ls-files --others --exclude-standard -- '*.rs'; } | sort -u )}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then exit 1; else test "$?" -eq 1; fi
non_plan_code=("${(@f)$(git ls-files -- '*.rs' '*.wgsl' | rg -v '^plans/')}")
test "${#non_plan_code[@]}" -gt 0
if rg -n --pcre2 '(?<![A-Za-z0-9_])(?:[PISCT][0-9]{2}[A-Za-z0-9_]*|[pisct][0-9]{2}_[A-Za-z0-9_]*)(?![A-Za-z0-9_])' "${non_plan_code[@]}"; then exit 1; else test "$?" -eq 1; fi
test -z "$(git ls-files | rg -v '^plans/' | rg --pcre2 '(?<![A-Za-z0-9_])(?:[PISCT][0-9]{2}[A-Za-z0-9_]*|[pisct][0-9]{2}_[A-Za-z0-9_]*)(?![A-Za-z0-9_])' || true)"
test -z "$(git ls-files | rg -v '^plans/' | rg --pcre2 '(?i)(?:^|[/_.-])[pisct][0-9]{2}(?=$|[/_.-])|sequence[0-9]+' || true)"
git diff --check 1e57d07d2595be95949caeff7b76a573a457723a..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The two specification M07 native commands remain required but deferred until
the user requests them:

```sh
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
```

All other final gates, task reviews, and implementation may proceed while they
are deferred. Do not start final holistic review or publication until the user
authorizes these commands and both render and exit successfully.

Before and after each task and at final verification, record raw/equivalent test
counts and sorted leaf names for default, `render-window`, `render-web`, and
combined configurations as ephemeral equality evidence. Compare the public
surface directly from `src/lib.rs`, public definitions, and the base-to-head
diff; add no parser or artifact.

Completion requires seven task `CLEAN` verdicts, holistic `CLEAN`, unchanged
behavior/public surface/dependencies/features/artifacts, complete M05.6
ownership, empty planning-identifier scans, no owned unsafe, clean status,
publication to leaf `main`, authority readback, and the P02-I02 leaf candidate
handoff. Root integration is excluded. A missing non-smoke prerequisite is a
blocker. If native-smoke authorization is still absent after implementation and
task review, stop before holistic review and publication with the clean task
head preserved; do not convert the deferral into completion.
