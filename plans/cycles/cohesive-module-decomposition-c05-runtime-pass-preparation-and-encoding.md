# P02-I02-S01-C05 Runtime Pass Preparation And Encoding

## 1 Header

- Cycle: `P02/I02/S01/C05`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: `14b0ab5f8d7fbb2d93e2a958587e1075657f0f7b`, the published
  C04 candidate verified on local and authority-remote `main`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `3365399fa2411efd5cd8fcfdfe74d4b756cd6a79`, SHA-256
  `892c5c1c2162a07c83bc124c3687e05ff62c6906930c154f25117792ec63d035`;
  sections M01-M04, M05.2, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, SHA-256
  `a45655293a54b5bf0986d508d6d2b278af68e5267dc373d8f5d56cb84e58c74b`;
  entry `C05 Runtime Pass Preparation And Encoding`.
- Outcome: complete the private runtime-pass hierarchy required by M05.2 by
  moving pass-owned parameter construction, preparation and realization,
  encoding and receipts, and pass-owned test support from `src/pass/mod.rs` to
  `parameters.rs`, `prepare.rs`, `encode.rs`, and `test_support.rs`, leaving only
  narrow orchestration and explicit current-contract reexports in `mod.rs`.

## 2 Boundary

- Published input: `src/pass/{mod,model,close,lower}.rs` at the cycle base.
  C04 established the runtime model, executable closure/accounting, and frame-
  to-runtime lowering owners; this cycle does not reopen or redesign them.
- C04-retained preparation inventory: allocation/resource/kernel/pass request
  models and analysis; resource validation, preflight, allocation and acquisition;
  lifetime/root validation; prepared resource/kernel/color-filter/pass bindings;
  pass realization; dispatch eligibility and graph-preparation source/routes;
  `PreparedGraph`, prepared views, and preparation observations.
- C04-retained parameter inventory: blur-edge, drop-shadow, color-filter,
  spatial, composite, and encoding-local uniform/parameter construction owned
  above shader byte serialization.
- C04-retained encoding inventory: Vello capture handoff and completion;
  external output, scheduling and activity; custom-spine progress and summaries;
  pending/accounting-ready frame commits and submissions; render regions and
  pass-specific encoding facts; graph/pass encoders; receipts and completion.
- C04-retained test-support inventory: pass-owned fixtures and donor identities;
  malformed-plan construction and rejection probes; shader/preparation/encoding
  fault controls; runtime-lowering, closure, layout, cache, filter, backdrop,
  composition, clip, mask, prepared-binding, and encoding observations.
- Required output: `src/pass/{mod,model,close,lower,parameters,prepare,encode,
  test_support}.rs`. `test_support` is compiled only for tests. Child direction is
  `model -> close/lower -> parameters/prepare -> encode -> test_support` where
  test support may consume production children but no production child consumes
  test support. Shared parameter facts sit below their preparation/encoding users.
- A type remains with the phase that owns its invariant. Inherent implementations
  may be separated only where existing methods genuinely belong to distinct
  preparation and encoding phases; they are not split to balance file length.
- Existing pass/backend, pass/shader, pass/renderer, and frame/pass edges remain
  unchanged. Imports name the owning front door or child explicitly and no new
  module-directory mutual edge is introduced.
- No public API, preparation failure atomicity or diagnostic precedence, cache or
  resource publication, allocation/accounting, parameter bytes, encoded pass
  order, capture/receipt fact, cancellation/cleanup, test operation, or oracle
  change is allowed.
- No semantic rename, compatibility shim, forwarding-only layer, copied
  definition, `include!`, `#[path]`, generated concatenation, glob-reexport maze,
  callback/trait indirection, generic helper module, source parser, inventory
  test, planning path/name test, or numerical size/count gate is allowed.
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
- Test impact: relocation and import paths only; test names, operations, inputs,
  assertions, and oracles remain unchanged. Existing focused tests characterize
  every moved behavior, so no new test is required.
- Generated artifacts: none in this leaf; root-owned artifacts remain untouched.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Establish Pass-Owned Parameter Construction

- Move blur-edge, drop-shadow, color-filter-operation, spatial-uniform,
  composite, and encoding-local pass parameter construction from `pass/mod.rs`
  to `parameters.rs`. Shader-owned byte models and serialization remain in
  `shader::parameters`; this child owns only semantic construction above them.
- Leave preparation, realization, encoding, and test support in `mod.rs` until
  their tasks. Give their existing callers explicit imports from the parameter
  owner and preserve checked narrowing, layout bytes, mapping, edge policy,
  format specialization, and validation order exactly.
- This is a behavior-preserving refactor, so fabricated RED is not applicable.
  Before editing and after the move, run and record:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render pass_spatial_uniform_rejects_f32_underflowing_raster_scales
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render drop_shadow_parameter_bytes_preserve_fractional_offset_and_solid_color
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composite_parameter_bytes_preserve_affine_mask_mapping_quality_and_extend
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_blur_layout_carries_semantic_mirror_bounds
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `src/pass/parameters.rs` exists; each pass-owned parameter builder
  has one owner; shader serialization remains unchanged; the parameter child has
  no preparation, encoding, backend, or test-support dependency; all commands
  pass with exact bytes and errors preserved.
- Intended commit: one complete pass-parameter ownership point.

### 4.2 T02 Move Preparation, Preflight, Realization, And Bindings

- Start only from the reviewed T01 head. Move allocation/resource/kernel/pass
  requests and analysis, resource validation/preflight/acquisition, lifetime/root
  validation, pass realization, dispatch eligibility, preparation source/routes,
  `PreparedGraph`, prepared resource/kernel/color-filter/pass bindings and views,
  and preparation-owned methods to `prepare.rs`.
- Keep encoding-owned types and methods in `mod.rs` until T03. Where one existing
  inherent implementation crosses phases, move methods by their actual phase
  owner without duplicating state or exposing a new API. Preparation may consume
  model, closure, lowering, parameters, shader/cache, resource, capability, and
  policy contracts; none may depend back on preparation merely for relocation.
- Preserve immutable preflight, exact validation precedence, failure atomicity,
  allocation identities, cache/resource publication, lease cleanup, and the
  executable-subset boundary. Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_preparation_is_allocation_safe_and_submission_free
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_preparation_rejects_unsupported_passes_without_resource_or_cache_mutation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render zero_capture_graph_spine_is_rejected_before_preparation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render prepared_copy_backdrop_objects_expose_exact_encoding_handles
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render prepared_spatial_filter_objects_expose_exact_encoding_handles
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render oversized_color_filter_buffer_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `src/pass/prepare.rs` exists; preparation, realization, binding,
  and dispatch-eligibility responsibilities have one owner; preparation performs
  no submission/publication; rejection leaves resources/cache/publication exact;
  all commands pass unchanged.
- Intended commit: one preparation/realization ownership point.

### 4.3 T03 Move Encoding, Capture Handoff, Scheduling, And Receipts

- Start only from the reviewed T02 head. Move Vello capture handoff/completion,
  external output and render regions, scheduling/activity, custom-spine state and
  progress, pass-specific encoding facts, graph/pass encoders, pending and
  accounting-ready commits, submission payloads, receipts, completion, and
  encoding-owned methods to `encode.rs`.
- Encoding consumes prepared bindings and parameter facts without taking
  preparation ownership. Preserve exact clear/capture/canonicalize/copy/filter/
  blur/shadow/composite/present order, one-encoder and one-transaction behavior,
  capture boundaries, output specialization, failure/cancellation cleanup,
  commit authorization, and publication atomicity. Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render custom_spine_encodes_clear_canonicalize_copy_source_over_and_present_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render multiple_vello_captures_share_one_graph_encoder_and_transaction_commit
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render later_two_capture_encode_failure_aborts_all_leases_and_rejects_retry_without_submission
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render capture_failure_aborts_and_rejects_retry_on_new_encoder
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composition_graph_encodes_clip_mask_opacity_and_blend_in_authored_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_graph_encodes_fused_operations_in_authored_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_graph_encodes_blur_and_drop_shadow_in_authored_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_graph_encodes_copy_filter_clip_foreground_and_group_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_encode_and_scope_failures_preserve_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_encode_failure_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `src/pass/encode.rs` exists; all M05.2 encoding responsibilities
  have one owner; preparation owns no submission method and encoding owns no
  allocation/preflight policy; exact order, receipts, failure atomicity, and
  cleanup remain characterized; all commands pass.
- Intended commit: one encoding/receipt ownership point.

### 4.4 T04 Move Pass Test Support And Reconcile The Front Door

- Start only from the reviewed T03 head. Move every pass-owned fixture, donor
  identity, malformed-plan/rejection probe, fault control, and runtime-lowering,
  closure, layout, cache, filter, backdrop, composition, clip, mask, prepared-
  binding, and encoding observation to test-only `test_support.rs`.
- Reconcile `pass/mod.rs` to test-gated child declaration/reexports, explicit
  current production-contract reexports, and only genuine narrow orchestration
  spanning production children. Helpers used solely inside one production child
  remain with that child; no production child imports `test_support`.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_executor_accepts_only_clear_capture_canonicalize_source_over_and_present
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_preserves_dependencies_and_last_use_releases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_preparation_is_allocation_safe_and_submission_free
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render custom_spine_encodes_clear_canonicalize_copy_source_over_and_present_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_shader_failure_preserves_prior_publication_and_cache
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render spatial_filter_encode_and_scope_failures_preserve_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_encode_failure_preserves_resources_cache_and_publication
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: all eight M05.2 files exist; `test_support.rs` is test-only;
  `mod.rs` contains no child-owned preparation, parameter, encoding, or test
  implementation; every current crate-visible path remains explicit; all focused
  commands and the full C05 matrix pass with no test/oracle delta.
- Intended commit: one test-support/front-door reconciliation point.

## 5 Verification And Completion

Each task records passing pre-move characterization and identical post-move
operation/oracle results; structural source checks are workflow evidence only and
are not tests. Each task requires a separate task-review `CLEAN` verdict. After
all tasks are clean, the coordinator makes a status-only `complete` commit, runs
this matrix, obtains a distinct holistic `CLEAN` review over the exact cycle
range, repeats the matrix at the unchanged reviewed head, and publishes with
authority-remote readback:

```sh
set -euo pipefail
test ! -e src/pass.rs
for required_file in \
  src/pass/mod.rs src/pass/model.rs src/pass/close.rs src/pass/lower.rs \
  src/pass/parameters.rs src/pass/prepare.rs src/pass/encode.rs \
  src/pass/test_support.rs; do
  test -f "$required_file"
done
test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/pass || true)"
test -z "$(git diff 14b0ab5f8d7fbb2d93e2a958587e1075657f0f7b -- src/lib.rs Cargo.toml README.md examples)"
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
git diff --check 14b0ab5f8d7fbb2d93e2a958587e1075657f0f7b..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

Both native smoke executables must render and exit on the native host; they are
verified available at the cycle base and are not deferred. Every unsafe-scan
match is classified; any executable match blocks completion. The publication
head is immutable after holistic review. Root integration remains excluded.

The C05-to-C06 leaf handoff reports the immutable published C05 candidate and
authority-remote readback SHA, the exact reviewed planning revision, clean task
and holistic verdicts, and the complete stable runtime-pass private hierarchy.
It confirms clean status and preserves the explicit exclusion of root integration.
