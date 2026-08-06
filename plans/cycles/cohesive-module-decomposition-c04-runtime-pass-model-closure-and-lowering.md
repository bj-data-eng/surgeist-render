# P02-I02-S01-C04 Runtime Pass Model Closure And Lowering

## 1 Header

- Cycle: `P02/I02/S01/C04`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base: `fcc03f3c0b0156fef6423302e7b7c0233d1a9286`, the published
  C03 candidate verified on local and authority-remote `main`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `3365399fa2411efd5cd8fcfdfe74d4b756cd6a79`, SHA-256
  `892c5c1c2162a07c83bc124c3687e05ff62c6906930c154f25117792ec63d035`;
  sections M01-M04, M05.2, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, SHA-256
  `a45655293a54b5bf0986d508d6d2b278af68e5267dc373d8f5d56cb84e58c74b`;
  entry `C04 Runtime Pass Model Closure And Lowering`.
- Outcome: replace `src/pass.rs` with `src/pass/mod.rs` plus `model.rs`,
  `close.rs`, and `lower.rs`, assigning runtime facts, executable closure and
  accounting, and frame-to-runtime conversion to their M05.2 owners while
  preserving every current runtime value, diagnostic, order, and contract.

## 2 Boundary

- Owned input: `src/pass.rs` and only the import and visibility repairs required
  by faithful relocation of its existing items.
- Required output: `src/pass/{mod,model,close,lower}.rs`. `model.rs` owns runtime
  resource, pass, read, result, filter, composite, spatial, and cache-key facts,
  including intrinsic constructors and validation. `close.rs` owns executable-
  subset closure, allowed runtime-pass shapes, topology/accounting validation,
  and the closed/preparable phase contracts. `lower.rs` owns conversion from
  frame graph-lowering views into the runtime model and its typed failures.
- Intermediate C05 state: preparation and allocation analysis, preflight,
  realization and prepared bindings, parameter construction, encoding and
  capture handoff, scheduling and receipts, and all pass-owned fixtures,
  injections, malformed-plan construction, and behavioral observations remain
  directly in `pass/mod.rs`. They are not copied, wrapped, renamed, or moved to
  a temporary owner. The C04 handoff records their exact retained inventory.
- Child direction is runtime model -> closure/lowering -> retained preparation
  -> retained encoding. `model.rs` may not depend on `close.rs`, `lower.rs`,
  preparation, encoding, backend, or test support. Production children may not
  depend on test-only support. Genuine pass orchestration and explicit current-
  contract reexports may remain in `mod.rs`.
- Existing pass/backend, pass/shader, pass/renderer, and frame/pass edges remain
  unchanged. Imports name the owning front door or child explicitly and no new
  module-directory mutual edge is introduced.
- No public API, runtime model value, closure diagnostic or precedence, pass
  order, dependency, read/result binding, release/accounting result, lowering
  fact, allowed executable subset, test operation, or test oracle changes.
- No semantic rename, compatibility shim, forwarding-only layer, copied
  definition, `include!`, `#[path]`, generated concatenation, glob-reexport
  maze, generic helper module, source parser, inventory test, or numerical
  size/count gate is allowed.
- Root, sibling, adapter, API-artifact, gitlink, hierarchical-public-front-door,
  dependency, feature, target, example, manifest, and correctness-fix work is
  excluded. Commands use installed artifacts offline; acquisition, installation,
  bootstrap, and update remain unauthorized.

## 3 Impacts

- Public API and caller migration: none; `src/lib.rs` and public paths remain
  unchanged. Current crate-visible pass paths remain available through explicit
  front-door reexports.
- Behavior and diagnostics: unchanged.
- Dependencies, features, targets, MSRV, docs, and examples: unchanged.
- Test impact: import/module relocation only; test names, operations, inputs,
  assertions, and oracles remain unchanged. No new test is required because the
  existing focused tests characterize the mechanically moved behavior.
- Generated artifacts: none in this leaf; root-owned artifacts remain untouched.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Establish The Runtime Model Owner

- Convert `src/pass.rs` to `src/pass/mod.rs` without changing the crate module
  name. Move runtime graph/resource/pass identities, resource roles and formats,
  spatial and Vello capture facts, color/filter/blur/drop-shadow/composite facts,
  read and result bindings, sampling choices, cache keys, `RuntimePass`, and the
  `LoweredGraphPlan` data model to `model.rs` with their intrinsic constructors,
  validation, and private helpers.
- Leave closure, lowering operations, preparation, encoding, parameter
  construction, and pass-owned test support in `mod.rs` until their owning C04
  or C05 task. Use explicit imports and the narrowest visibility; do not split an
  intrinsic type implementation merely to balance files or duplicate a model to
  bridge the move.
- This is a behavior-preserving refactor, so fabricated RED is not applicable.
  Before editing and after the move, run and record:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_preserves_dependencies_and_last_use_releases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render mask_pipeline_keys_exclude_image_identity
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `src/pass/{mod,model}.rs` exist; each assigned runtime fact has one
  definition in `model.rs`; `model.rs` has no higher-phase dependency; focused
  behavior and default/combined checks pass; public, manifest, docs, and example
  surfaces are unchanged.
- Intended commit: one complete runtime-model ownership point.

### 4.2 T02 Move Executable Closure And Accounting

- Start only from the reviewed T01 head. Move executable graph facts, closed
  graph state, executable-subset classification, allowed pass-shape validation,
  traversal/maps, root and lifetime checks, read/result/release accounting, and
  the closed/preparable phase contracts to `close.rs` with their owned methods.
- Preserve diagnostic precedence, pass and filter ordering, exact allowed base,
  composition, color-filter, blur/drop-shadow, and bounded-backdrop subsets, and
  rejection before allocation or cache mutation. Leave the test-only fixture and
  observation implementations in `mod.rs` for C05; they may call the closure
  front door but production closure may not depend on them.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_executor_accepts_only_clear_capture_canonicalize_source_over_and_present
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composition_graph_executor_accepts_only_spine_and_ordered_layer_composition
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gpu_graph_executor_accepts_only_spine_composition_and_ordered_color_filters
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gpu_graph_executor_accepts_only_color_blur_and_drop_shadow_filter_graphs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gpu_graph_executor_accepts_only_bounded_top_level_backdrop_graphs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_preparation_rejects_unsupported_passes_without_resource_or_cache_mutation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render zero_capture_graph_spine_is_rejected_before_preparation
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `src/pass/close.rs` exists; closure, allowed-subset, and runtime
  accounting responsibilities have one owner; malformed bindings and unsupported
  graphs retain their exact rejection and failure-atomic behavior; child imports
  preserve model-to-closure direction; all commands pass unchanged.
- Intended commit: one executable closure/accounting ownership point.

### 4.3 T03 Move Frame-To-Runtime Lowering And Reconcile The Front Door

- Start only from the reviewed T02 head. Move resource/pass/result conversion,
  frame-view traversal, runtime pass-kind and read-binding conversion, spatial,
  Vello, clip, filter, composite, cache-key, and typed lowering-error construction
  to `lower.rs`. Keep the `LoweredGraphPlan` data model in `model.rs`; place only
  its conversion behavior in `lower.rs`.
- Reconcile `pass/mod.rs` to child declarations, explicit current-contract
  reexports, genuine cross-child orchestration, and the exact C05 intermediate
  inventory named in section 2. Remove no retained C05 item and introduce no
  temporary alternative model, wrapper, callback, trait, or helper bucket.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_preserves_dependencies_and_last_use_releases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_graph_preserves_authored_order_clamps_and_exact_lifetimes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render blur_and_drop_shadow_graph_preserves_order_edges_and_lifetimes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composition_graph_orders_clip_mask_opacity_blend_and_nested_layers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_graph_reads_completed_parent_once_and_preserves_group_order
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: all four C04 pass files exist and `src/pass.rs` is absent;
  frame-to-runtime conversion has one owner; runtime vocabulary, topology,
  lifetime releases, spatial/filter/composite facts, and cache keys are exact;
  `mod.rs` contains no model, closure, accounting, or lowering implementation;
  the worker returns the exact retained C05 item inventory; all focused commands
  and the full C04 matrix pass.
- Intended commit: one lowering/front-door/intermediate-inventory ownership point.

## 5 Verification And Completion

Each task records the required passing pre-move characterization and identical
post-move operation/oracle result; source and file checks are structural evidence
only. Each task requires a separate task-review `CLEAN` verdict. After all tasks
are clean, the coordinator makes a status-only `complete` commit, runs this
matrix, obtains a distinct holistic `CLEAN` review over the exact cycle range,
repeats the matrix at the unchanged reviewed head, and publishes with authority-
remote readback:

```sh
set -euo pipefail
test ! -e src/pass.rs
for required_file in \
  src/pass/mod.rs src/pass/model.rs src/pass/close.rs src/pass/lower.rs; do
  test -f "$required_file"
done
test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/pass || true)"
test -z "$(git diff fcc03f3c0b0156fef6423302e7b7c0233d1a9286 -- src/lib.rs Cargo.toml README.md examples)"
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
git diff --check fcc03f3c0b0156fef6423302e7b7c0233d1a9286..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The live smoke executables remain required and must render and exit on the native
host. Every unsafe-scan match is classified; any executable match blocks
completion. The publication head is immutable after holistic review. Root
integration remains excluded.

User-authorized deferred-gate note (2026-08-06): retain both native smoke
commands above, but ignore their current noncompletion for C04 progression. The
user reports that the remotely accessed Mac aggressively attempts to sleep and
its power settings cannot be changed until they are physically present; rerun
these two gates later. This deferral is not a Rust-test failure and changes no
implementation or test contract.

The C04-to-C05 leaf handoff reports the immutable published C04 candidate and
authority-remote readback SHA, the exact reviewed planning revision, clean task
and holistic verdicts, the stable pass model/closure/lowering children, and the
exact preparation, encoding, parameter, and test-support inventory retained in
`pass/mod.rs`. It confirms clean status and preserves the explicit exclusion of
root integration.
