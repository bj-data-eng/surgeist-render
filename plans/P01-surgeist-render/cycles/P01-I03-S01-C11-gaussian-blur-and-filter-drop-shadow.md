# P01-I03-S01-C11 Gaussian Blur And Filter Drop Shadow

## 1 Header

- Cycle: `P01/I03/S01/C11`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base and published prerequisite: `3345fe6bb1face00ee3364ae4345b5c0782053e2`
  (`P01/I03/S01/C10`, remotely verified and explicitly closed).
- Specification: `plans/P01-surgeist-render/initiatives/P01-I03-gpu-render-pipeline.md`
  at `e96614d3eff2667a6b07cbb145873122d4f1f22d`, revision `P01/I03/V03`,
  `sha256:ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`;
  clauses S16, S18-S22, S25, S27-S30, and the C11-applicable S31-S34 evidence.
- Sequence: `plans/P01-surgeist-render/sequences/P01-I03-S01-gpu-render-pipeline.md`
  at `e96614d3eff2667a6b07cbb145873122d4f1f22d`,
  `sha256:9fb83aeebf2bcd2a581241e97c7cdde58942c8001f54c195b5a54a090908f1ba`;
  entry `C11 Gaussian Blur And Filter Drop Shadow`.
- Outcome: execute ordinary Gaussian blur and solid-color CSS filter drop
  shadow as ordered GPU image passes in both working formats, preserving
  transparent-black edges, continuous offsets, SourceAlpha fan-out, signed
  bounds, exact resource lifetimes, and the C12/C13 public diagnostic boundary.

## 2 Boundary

- The C10 exact graph subset is extended only for nonempty authored filter runs
  containing color operations, nonzero `FilterBlur`, and executable
  `FilterDropShadow`. Zero blur remains an identity and creates no blur pass.
- Ordinary blur samples transparent black outside the semantic source bounds.
  Backdrop mirror-edge execution, `CopyBackdrop`, and backdrop group composition
  remain C12 work and stay rejected before preparation.
- One blur is two distinct passes with a distinct intermediate texture. The
  existing immutable `GaussianKernelPlan` remains the sole kernel-byte owner
  and is cached by its exact standard-deviation, raster-scale, support, and
  sampling-form facts.
- One drop shadow keeps the original source alive for two reads, blurs
  SourceAlpha, samples it at the authored continuous logical offset, colorizes
  with the finite solid premultiplied color, then source-overs the unchanged
  source above the shadow before the next authored filter.
- The existing test-only authored-filter fixture is the sole C11 ingress. Broad
  layer filters, `FilteredImagePaint`, bounded backdrop execution, non-solid
  shadow paint, inset/spread shadow input, reference filters, and every C12+
  graph remain exact typed diagnostics.
- `supports_gpu_blur_filter_execution()` and
  `supports_gpu_drop_shadow_filter_execution()` remain false in this cycle.
  C13 owns final capability and inventory reconciliation.
- All passes encode into the caller-owned graph transaction. Production blur
  and shadow code contains no queue submission, readback, map, poll, CPU pixel
  execution, Vello atlas re-entry, replay, wait, or fallback.
- No public test seam, alternate executor, unchecked buffer path, hidden
  downscale, dependency, feature, compatibility shim, or `unsafe` is added.

## 3 Impacts

| Area | C11 effect |
| --- | --- |
| Public API and anticipated SemVer | Internal-only; authored filter and diagnostic surfaces remain unchanged |
| Dependencies and features | Unchanged; all commands use already-present artifacts offline |
| Generated artifacts | None; root owns API artifacts |
| Source artifacts | Add only tracked blur and drop-shadow WGSL owned by `shader.rs` |
| Docs and examples | Unchanged; C14 owns final public documentation |
| MSRV | Installed stable Rust 1.97.x remains required |
| Root follow-up | None in C11; final leaf candidate handoff remains C14 |
| Safety | `#![forbid(unsafe_code)]` remains effective and all owned Rust remains free of unsafe |

## 4 Tasks

### 4.1 T01 Close The C11 Executable Graph Subset

- Area: `src/{frame.rs,pass.rs,renderer.rs,tests.rs}`.
- Outcome: admit only validated C10 graph content plus ordered nonzero ordinary
  blur and executable drop-shadow nodes; preserve exact order, edge policy,
  spatial facts, SourceAlpha fan-out, and last-use releases.
- RED: `c11_executor_accepts_only_color_blur_and_drop_shadow_filter_graphs`
  fails only because the C11 subset is not preparable;
  `c11_blur_and_drop_shadow_graph_preserve_order_edges_and_lifetimes` fails only
  because the closed graph does not yet retain the exact C11 facts;
  `c12_plus_graph_diagnostic_precedes_unavailable_effect_working_format` remains
  green as the next-cycle boundary.
- Acceptance: empty/malformed steps, wrong axes or inputs, mirror edges,
  `CopyBackdrop`, missing payloads, source/result aliasing, stale/forward
  dependencies, and every C12+ pass fail before resource acquisition. A valid
  drop shadow reads the original source exactly twice and releases it only
  after the merge.
- Commands: run the three named tests separately, then `C11-CHECK`.
- Depends on: none.
- Intended commit: `feat(graph): close gaussian and drop shadow subset`.

### 4.2 T02 Realize Checked Gaussian Blur Programs

- Area: `src/{resource.rs,shader.rs,pass.rs,tests.rs}` and
  `src/shaders/blur.wgsl`.
- Outcome: make the existing normalized Gaussian sample bytes the only immutable
  kernel input and realize checked horizontal/vertical, RGBA/SourceAlpha,
  high/reduced blur pipelines.
- RED: `gaussian_kernel_bytes_are_symmetric_normalized_and_exactly_cached`
  fails only at the exact sample-byte/cache assertion;
  `c11_blur_layout_binds_exact_source_spatial_and_kernel` fails only at a
  missing or dummy binding; `c11_blur_cache_realizes_checked_axis_input_and_precision_programs`
  fails only because no executable blur program exists.
- Acceptance: layout binds one linear-sampled working source, one spatial
  uniform, and one read-only kernel buffer; shader-key facts select axis and
  RGBA versus SourceAlpha without runtime source; kernel length and device
  limits fail before allocation; both working formats use the same semantics.
- Commands: run all three named tests separately; run
  `resource_preparation_is_private_allocation_safe_and_submission_free`; then
  `C11-CHECK`.
- Depends on: T01.
- Intended commit: `feat(shader): realize checked gaussian blur programs`.

### 4.3 T03 Realize Checked Drop Shadow Colorize And Merge

- Area: `src/{shader.rs,pass.rs,tests.rs}` and
  `src/shaders/drop_shadow.wgsl`.
- Outcome: safely serialize continuous offset and solid premultiplied color,
  realize drop-shadow colorization, and reuse the checked compositor for the
  unchanged-source-over-shadow merge.
- RED: `drop_shadow_parameter_bytes_preserve_fractional_offset_and_solid_color`
  fails only at the exact finite WGSL byte layout;
  `c11_drop_shadow_layout_binds_blurred_alpha_spatial_and_parameters` fails only
  at a missing or dummy binding;
  `c11_drop_shadow_cache_realizes_checked_colorize_and_merge_programs` fails
  only because the C11 shader/composite keys remain unrealized.
- Acceptance: colorization samples blurred SourceAlpha linearly at the
  continuous logical offset, emits transparent black outside its semantic
  source, multiplies solid premultiplied color and alpha, and never samples its
  result target. Merge reads distinct unchanged source and shadow textures and
  performs fixed premultiplied source-over without a destination read.
- Commands: run all three named tests separately; run
  `c09_composite_layouts_bind_no_dummy_parent_clip_or_mask`; then `C11-CHECK`.
- Depends on: T02.
- Intended commit: `feat(shader): realize checked drop shadow programs`.

### 4.4 T04 Encode Spatial Filters In One Graph Transaction

- Area: `src/{pass.rs,backend.rs,gpu_transaction.rs,tests.rs}`.
- Outcome: prepare and encode every validated blur/colorize/merge pass in order
  through the existing caller-owned encoder and checked GPU scope.
- RED: `c11_graph_encodes_blur_and_drop_shadow_in_authored_order` fails only
  because the scheduler has no C11 encoding route;
  `blur_passes_use_distinct_source_intermediate_and_result_without_readback`
  fails only at the missing pass receipts;
  `drop_shadow_reads_source_twice_and_releases_after_merge` fails only at the
  missing exact lease transition;
  `c11_encode_failure_preserves_resources_cache_and_publication` fails only
  because the C11 abort path is absent.
- Acceptance: each pass advances once, binds its exact prepared resources and
  signed viewport/scissor mapping, releases textures and kernels only at
  validated last use, submits once through `GpuOperationTransaction`, and
  atomically aborts provisional resources/cache entries and publication on
  encode or scope failure.
- Commands: run all four named tests separately; run
  `multiple_color_runs_share_one_graph_encoder_and_transaction_commit`,
  `graph_render_path_submits_without_map_or_cpu_wait`, and `C11-CHECK`.
- Depends on: T03.
- Intended commit: `feat(gpu): encode spatial filters in one transaction`.

### 4.5 T05 Prove Gaussian And Drop Shadow GPU Quality

- Area: `src/{renderer.rs,backend.rs,pass.rs,reference.rs,tests.rs}`.
- Outcome: route only the C11 test fixture through the shared production graph
  executor and prove high/reduced pixels, signed bounds, transparent edges,
  local transforms, SourceAlpha, fractional offset, source merge, and authored
  order against independent/oracle evidence.
- RED: `blur_impulse_is_symmetric_normalized_and_matches_oracle`,
  `ordinary_blur_samples_transparent_black_at_all_edges`,
  `drop_shadow_preserves_source_uses_fractional_offset_and_expands_signed_bounds`,
  and `nonuniform_scale_and_skew_preserve_local_blur_shape` fail only at their
  named production-GPU comparisons.
- Acceptance: exact dimensions and origins precede pixel comparison; high
  precision stays within four straight-RGBA8 levels and 1.5% alpha energy;
  reduced stays within four alpha/premul8 levels and 2.5% energy; transform
  centroid error stays within 0.25/0.35 device pixel; identity/transparent
  invariants remain exact; no missing adapter is treated as a skip.
- Commands: run all four named tests separately; run
  `high_precision_color_functions_match_cpu_oracle_for_boundary_pixels`,
  `reduced_precision_color_functions_match_cpu_oracle_with_declared_tolerance`,
  `filter_function_order_changes_output_and_matches_ordered_oracle`, and
  `C11-CHECK`.
- Depends on: T04.
- Intended commit: `test(gpu): prove gaussian and drop shadow quality`.

### 4.6 T06 Close C11 Integration Reuse And Diagnostics

- Area: integrated
  `src/{capability.rs,frame.rs,pass.rs,shader.rs,backend.rs,gpu_transaction.rs,renderer.rs,tests.rs}`.
- Outcome: prove mixed color/blur/shadow execution, presented delivery, reuse,
  zero-budget cleanup, atomic failure, retained public diagnostics, and the
  exact C12 handoff without enabling a public filter route.
- RED: `c11_fixture_executes_spatial_filters_while_public_capabilities_remain_diagnostic`,
  `render_window_smoke_executes_gaussian_and_drop_shadow_fixture`,
  and `public_dispatch_retains_c09_boundary_while_c11_fixture_uses_shared_executor`
  fail only at their fixture, presented, or public-boundary assertions.
- Acceptance: repeated frames stabilize pass/kernel/resource counts; zero
  budget retains no idle byte-accounted C11 resources; one submission produces
  exact pixels; failure preserves prior publication; public blur/drop-shadow,
  backdrop, broad layer/reference/filtered-image execution remains false with
  exact diagnostics before allocation; production source has no CPU/replay/
  readback/re-entry route.
- Commands: run the three named tests separately, using `render-window` for the
  presented test; run
  `repeated_frames_reuse_resources_without_growth_or_readback`,
  `budget_zero_releases_idle_resources_without_changing_pixels`,
  `device_loss_is_terminal_idempotent_and_releases_device_resources`; run the
  completion guards; then `C11-CHECK`.
- Depends on: T05.
- Intended commit: `feat(gpu): complete gaussian and drop shadow execution`.

## 5 Verification

All commands use an already-installed stable Rust 1.97.x toolchain and
`CARGO_NET_OFFLINE=true`; acquisition is not authorized.

`C11-CHECK` is:

```sh
set -euo pipefail
rustup run 1.97-aarch64-apple-darwin rustc --version
CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo fmt --check
CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo check -p surgeist-render
for features in '' render-window render-web render-window,render-web; do
  if test -n "$features"; then
    CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo test -p surgeist-render --features "$features"
    CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo clippy -p surgeist-render --all-targets --features "$features" -- -F unsafe-code -D warnings -D clippy::too_many_lines
  else
    CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo test -p surgeist-render
    CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings -D clippy::too_many_lines
  fi
done
CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true rustup run 1.97-aarch64-apple-darwin cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
test -z "$(git ls-files -- Cargo.lock)"
```

The owned-Rust unsafe scan uses the canonical workflow manifest and pattern.
Native dependency-tree evidence requires no `getrandom/wasm_js`; the wasm
`render-web` tree requires exactly one target-scoped `getrandom/wasm_js`.

Completion guards require:

- only `src/shaders/blur.wgsl` and `src/shaders/drop_shadow.wgsl` are added to
  the C10 shader inventory, with exactly one `include_str!` owner each;
- `Cargo.toml`, public `src/lib.rs` reexports, features, and dependency roles
  equal the cycle base;
- public blur/drop-shadow capability accessors remain false and their exact
  typed diagnostics remain reachable before allocation;
- no production match for `queue.submit`, `map_async`, `Device::poll`,
  `register_texture`, `override_image`, `reference::`, a CPU filter executor,
  graph replay, or a lint allowance/expectation added by this range;
- all T01-T06 named evidence and the complete four-feature test/strict-Clippy
  matrix pass with the exact test-name inventory preserved except for the named
  C11 additions.

## 6 Completion

- All six ordered task ranges have fresh `CLEAN` task reviews and coordinator
  acceptance against their exact current attribution.
- A separate status-only commit changes this plan from `in_progress` to
  `complete`; the exact cycle range then passes `C11-CHECK`, every named C11
  test, completion guards, and a clean-worktree check.
- A fresh holistic reviewer returns `CLEAN` for the exact cycle range; its
  retained artifact is recorded with `record_cycle_review`. The complete final
  command set passes again on the exact reviewed head.
- The immutable reviewed head is published to authority `origin/main`, fresh
  remote readback agrees with local `main` and its tracking ref, publication
  evidence is retained, and `close_cycle` records C11 as closed.
- Handoff: C12 receives the complete supported ordered filter chain and the
  unchanged bounded-backdrop diagnostic boundary.
- Genuine blockers are limited to unowned conflicting state, a reviewed-source
  contradiction, unavailable required native GPU/presented execution, missing
  installed Rust 1.97.x or authorized wasm target, required unsafe, unavailable
  Surgeist profiles, or publication credentials/remote history failure.
