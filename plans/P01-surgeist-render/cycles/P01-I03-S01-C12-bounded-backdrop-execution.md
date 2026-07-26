# P01-I03-S01-C12 Bounded Backdrop Execution

## 1 Header

- Cycle: `P01/I03/S01/C12`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base and published prerequisite: `4d8a33a3cb7047a0f13725b2e53dc9712492c684`
  (`P01/I03/S01/C11`, remotely verified and explicitly closed).
- Specification: `plans/P01-surgeist-render/initiatives/P01-I03-gpu-render-pipeline.md`
  at `e96614d3eff2667a6b07cbb145873122d4f1f22d`, revision `P01/I03/V03`,
  `sha256:ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`;
  clauses S16-S17, S19-S20, S23-S24, S27-S28, and the C12-applicable
  S30-S34 evidence.
- Sequence: `plans/P01-surgeist-render/sequences/P01-I03-S01-gpu-render-pipeline.md`
  at `e96614d3eff2667a6b07cbb145873122d4f1f22d`,
  `sha256:9fb83aeebf2bcd2a581241e97c7cdde58942c8001f54c195b5a54a090908f1ba`;
  entry `C12 Bounded Backdrop Execution`.
- Outcome: copy the completed current root parent once, apply the authored GPU
  filter chain with semantic backdrop edges, clip the filtered backdrop, keep
  foreground unfiltered, and composite the complete bounded group in authored
  order without replay or production readback.

## 2 Boundary

- The executable subset is only a bounded, top-level, untransformed,
  non-repeated backdrop layer whose filters are already executable by C11.
- `CopyBackdrop` reads the completed current root parent at the layer paint
  position. It includes the root base color and every completed prior sibling,
  uses signed device mapping, and returns transparent black outside the surface.
- Backdrop blur mirrors at the semantic capture or border-box edge, never the
  padded allocation edge. Ordinary blur keeps C11 transparent-black semantics.
- The authored filter list executes before the backdrop clip. Foreground is
  rendered separately to transparent, stays unfiltered, and is source-over the
  clipped filtered backdrop inside a transparent local group.
- The layer's outer clip, resolved alpha mask, opacity, and blend apply exactly
  once to the complete local group before it is composited into the root.
- Later siblings observe the completed backdrop group. No result re-enters
  Vello, and no prior command is cloned or replayed.
- Root backdrop policy, nested or isolated backdrop semantics, transformed
  backdrops, repeated top-level capture, unresolved clip resources, broad layer
  filters, and C13 cutover remain typed diagnostics before allocation.
- C12 may make the narrow bounded-backdrop runtime capability truthful. It does
  not enable broad backdrop, backdrop-root, layer-filter, reference-filter, or
  CPU/materialized execution.

## 3 Impacts

| Surface | C12 disposition |
| --- | --- |
| Public API and reexports | No signature, type, reexport, or feature change; only the existing narrow bounded-backdrop runtime capability becomes truthful after implementation. |
| Dependencies and artifacts | No dependency, feature, lockfile, generated API artifact, or root-owned integration change. |
| Documentation and examples | No user-facing docs or example edit in C12; C14 owns final GPU-pipeline documentation and platform evidence after C13 completes the public cutover. |
| MSRV | Unchanged at Rust 1.97.x; C12 adds no language, library, dependency, or target requirement beyond the published base. |
| Safety | No owned unsafe code, unsafe attribute, extern block, or lint allowance/expectation. |

## 4 Global Invariants

- Preserve one transaction-owned drawing submission stage, caller-owned command
  encoding, one async error-scope resolution boundary, and atomic publication.
- Preserve the C11 working-pixel contract, exact filter order and clamp
  boundaries, high/reduced selection, signed origins, texel centers, resource
  identities, validated last-use releases, and deterministic budget trimming.
- A pass never samples its result target. Copy, filtered backdrop, foreground,
  local group, old parent, and new parent remain distinct whenever their
  lifetimes overlap.
- No production pass maps, polls, waits, downloads pixels, invokes a CPU filter
  executor, clones/replays source commands, or registers graph output back into
  the raster atlas.
- A failed plan, preparation, encode, scope, signal, acquisition, or present
  step aborts every provisional C12 resource and cache entry and preserves prior
  publication, stats, and surface state.
- Public and private inputs are validated before device limits, allocation,
  cache mutation, encoding, submission, presentation, or publication.
- Source remains authoritative; no dependency, feature, public reexport, API
  artifact, unsafe code, lint suppression, or root-owned integration change is
  in this cycle.
- Every implementation task uses a fresh worker, exact RED evidence, one
  focused validated commit, direct Git attribution, fresh task review, and
  coordinator acceptance before completion.

## 5 Ordered Tasks

### 5.1 T01 Close The C12 Bounded Backdrop Graph Subset

- Area: `src/{command.rs,frame.rs,pass.rs,renderer.rs,tests.rs}`.
- Outcome: admit only the supported bounded backdrop graph and retain exact
  current-parent, capture, filter, clip, foreground, group, and later-sibling
  dependencies.
- RED: `c12_executor_accepts_only_bounded_top_level_backdrop_graphs` fails only
  because production dispatch still rejects every `CopyBackdrop`;
  `c12_backdrop_graph_reads_completed_parent_once_and_preserves_group_order`
  fails only at the closed-subset classification and dependency receipt;
  `c13_plus_backdrop_diagnostic_precedes_unavailable_effect_working_format`
  fails only because later/broad backdrop classification is not yet separated
  from the C12 subset.
- Acceptance: exact C12 graphs contain one completed-parent read per backdrop,
  one copy, the authored filter pass sequence, post-filter backdrop clip,
  separate foreground capture, transparent local group composition, one outer
  layer composition, and dependencies to every later observer. Malformed,
  root, nested, transformed, repeated, unresolved, and C13+ graphs retain exact
  typed diagnostics before preparation.
- Commands: run all three named tests separately; run
  `backdrop_plan_depends_on_current_parent_not_cloned_commands`,
  `post_filter_backdrop_clip_retains_expanded_halo_outside_capture`, and
  `C12-CHECK`.
- Depends on: none.
- Intended commit: `feat(graph): close bounded backdrop subset`.

### 5.2 T02 Realize The Checked Backdrop Copy Program

- Area: `src/{shader.rs,pass.rs,resource.rs,tests.rs}` and new
  `src/shaders/copy_backdrop.wgsl`.
- Outcome: realize checked high/reduced `CopyBackdrop` objects that sample the
  exact completed parent through signed surface-to-capture mapping.
- RED: `c12_copy_backdrop_layout_binds_parent_and_spatial_mapping` fails only
  at the missing exact layout;
  `c12_copy_backdrop_cache_realizes_checked_working_format_programs` fails only
  at the unrealized program;
  `copy_backdrop_maps_signed_bounds_and_transparent_surface_edges` fails only at
  the missing static shader semantics.
- Acceptance: the program binds one working-format parent source, one exact
  sampler, and one spatial uniform; writes the selected working format; maps
  signed capture texel centers to surface texel centers; emits transparent
  black outside the surface; never samples its result; and has one canonical
  `include_str!` owner. Validation and device limits precede allocation/cache
  publication.
- Commands: run all three named tests separately; run
  `c08_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms`,
  `resource_preparation_is_private_allocation_safe_and_submission_free`, and
  `C12-CHECK`.
- Depends on: T01.
- Intended commit: `feat(shader): realize checked backdrop copy program`.

### 5.3 T03 Realize Semantic Backdrop Edge Programs

- Area: `src/{shader.rs,pass.rs,tests.rs}` and
  `src/shaders/blur.wgsl`.
- Outcome: extend the checked Gaussian programs so backdrop filter blur,
  including SourceAlpha blur inside drop shadow, mirrors at semantic bounds
  while ordinary filter blur stays transparent black.
- RED: `c12_blur_cache_separates_transparent_and_mirrored_edge_programs` fails
  only because the checked cache does not distinguish edge policy;
  `c12_backdrop_blur_layout_carries_semantic_mirror_bounds` fails only at the
  missing exact layout/program facts;
  `c12_backdrop_filter_chain_preserves_authored_order_and_clamp_boundaries`
  fails only because mirrored backdrop stages are not realizable.
- Acceptance: every backdrop Gaussian sample mirrors its logical coordinate at
  the semantic capture/border-box bounds before texture mapping, never at
  allocation padding; horizontal/vertical, RGBA/SourceAlpha, high/reduced, and
  transparent/mirror program identities are exact; ordinary C11 shaders and
  quality remain unchanged.
- Commands: run all three named tests separately; run
  `ordinary_blur_samples_transparent_black_at_all_edges`,
  `c11_blur_cache_realizes_checked_axis_input_and_precision_programs`, and
  `C12-CHECK`.
- Depends on: T02.
- Intended commit: `feat(shader): realize backdrop mirror edge filters`.

### 5.4 T04 Encode The Backdrop Group In One Transaction

- Area: `src/{pass.rs,backend.rs,gpu_transaction.rs,tests.rs}`.
- Outcome: encode copy, ordered filters, backdrop clip, foreground, local-group
  merge, outer operations, and root composition through the existing
  caller-owned encoder and transaction.
- RED: `c12_graph_encodes_copy_filter_clip_foreground_and_group_in_order` fails
  only because the scheduler has no C12 encoding route;
  `backdrop_copy_filter_and_group_use_distinct_resources_without_readback`
  fails only at missing pass receipts;
  `later_sibling_dependency_follows_completed_backdrop_group` fails only at the
  missing committed dependency transition;
  `c12_backdrop_encode_failure_preserves_resources_cache_and_publication` fails
  only because the C12 abort path is absent.
- Acceptance: each pass advances once and binds exact prepared resources and
  signed viewport/scissor mapping; the completed parent is copied once; the
  filtered backdrop and unfiltered foreground stay distinct until local merge;
  later siblings read the new completed parent; releases occur only at exact
  last use; the transaction submits once and aborts all provisional state on
  failure.
- Commands: run all four named tests separately; run
  `multiple_color_runs_share_one_graph_encoder_and_transaction_commit`,
  `graph_render_path_submits_without_map_or_cpu_wait`, and `C12-CHECK`.
- Depends on: T03.
- Intended commit: `feat(gpu): encode bounded backdrop group`.

### 5.5 T05 Prove Bounded Backdrop GPU Quality And Ordering

- Area: `src/{renderer.rs,backend.rs,pass.rs,reference.rs,tests.rs}`.
- Outcome: route only the C12 fixture through the shared production graph
  executor and prove semantic edges, completed-parent capture, base/sibling
  order, unfiltered foreground, clip order, and later observers against
  independent/oracle evidence.
- RED: `backdrop_blur_mirrors_at_semantic_bounds_not_allocation_padding`,
  `backdrop_reads_only_completed_prior_content_and_base_once`,
  `backdrop_foreground_is_not_filtered_and_composites_above_backdrop`, and
  `later_siblings_observe_completed_backdrop_group` fail only at their named
  production-GPU comparisons.
- Acceptance: exact dimensions and signed origins precede pixel comparison;
  base color appears once; prior siblings are captured and later siblings are
  not; filtered backdrop obeys authored filter order and mirror edges;
  foreground remains crisp and source-over above it; clipping happens after
  filtering; later siblings observe the completed group. Applicable S34
  high/reduced filter, composition, and placement tolerances hold, and no
  missing adapter is treated as a skip.
- Commands: run all four named tests separately; run
  `filter_function_order_changes_output_and_matches_ordered_oracle`,
  `negative_bounds_and_subpixel_transforms_do_not_shift_capture`,
  `outer_clip_precedes_mask_and_opacity_but_follows_filter`, and
  `C12-CHECK`.
- Depends on: T04.
- Intended commit: `test(gpu): prove bounded backdrop quality`.

### 5.6 T06 Close C12 Integration Reuse And Diagnostics

- Area: integrated
  `src/{capability.rs,command.rs,frame.rs,pass.rs,shader.rs,backend.rs,gpu_transaction.rs,renderer.rs,tests.rs}`.
- Outcome: prove mixed/presented bounded-backdrop delivery, reuse, zero-budget
  cleanup, atomic failure, truthful narrow capability, retained broad
  diagnostics, and the exact C13 handoff.
- RED: `c12_fixture_executes_bounded_backdrop_while_broad_capabilities_remain_diagnostic`,
  `render_window_smoke_executes_bounded_backdrop_fixture`, and
  `public_dispatch_enables_only_bounded_backdrop_execution` fail only at their
  fixture, presented, or public-boundary assertions.
- Acceptance: repeated frames stabilize pass/kernel/resource counts; zero
  budget retains no idle byte-accounted C12 resource; one submission produces
  exact pixels; failure preserves prior publication; the narrow bounded
  backdrop capability is true; root, nested, transformed, repeated, broad
  isolation, layer/reference filter, and CPU/materialized routes remain false
  with exact diagnostics before allocation; production contains no replay,
  readback, re-entry, or cloned backdrop commands.
- Commands: run the three named tests separately, using `render-window` for the
  presented test; run
  `repeated_frames_reuse_resources_without_growth_or_readback`,
  `budget_zero_releases_idle_resources_without_changing_pixels`,
  `device_loss_is_terminal_idempotent_and_releases_device_resources`; run the
  completion guards; then `C12-CHECK`.
- Depends on: T05.
- Intended commit: `feat(gpu): complete bounded backdrop execution`.

## 6 Verification

All commands use the already-installed stable Rust 1.97.x toolchain and
`CARGO_NET_OFFLINE=true`; acquisition is not authorized.

`C12-CHECK` is:

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

- only `src/shaders/copy_backdrop.wgsl` is added to the C11 shader inventory,
  with exactly one `include_str!` owner;
- `Cargo.toml`, public `src/lib.rs` reexports, features, and dependency roles
  equal the cycle base;
- narrow bounded-backdrop capability and execution are true while broad,
  root, nested, transformed, repeated, layer/reference-filter, and
  materialized/CPU routes retain exact typed diagnostics before allocation;
- `RenderBackdropCapture` contains no cloned `source_commands`;
- no production match for new `queue.submit`, `map_async`, `Device::poll`,
  `register_texture`, `override_image`, `reference::`, CPU filter execution,
  graph replay, or lint allowance/expectation is added by this range;
- all T01-T06 named evidence and the complete four-feature test/strict-Clippy
  matrix pass with the exact test-name inventory preserved except for named C12
  additions.

## 7 Completion

- All six ordered task ranges have fresh `CLEAN` task reviews and coordinator
  acceptance against their exact current attribution.
- A separate status-only commit changes this plan from `in_progress` to
  `complete`; the exact cycle range then passes `C12-CHECK`, every named C12
  test, completion guards, and a clean-worktree check.
- A fresh holistic reviewer returns `CLEAN` for the exact cycle range; its
  retained artifact is recorded with `record_cycle_review`. The complete final
  command set passes again on the exact reviewed head.
- The immutable reviewed head is published to authority `origin/main`, fresh
  remote readback agrees with local `main` and its tracking ref, publication
  evidence is retained, and `close_cycle` records C12 as closed.
- Handoff: C13 receives every independently GPU-complete replacement primitive,
  the truthful narrow bounded-backdrop capability, and unchanged broad
  diagnostic boundaries.
- Genuine blockers are limited to unowned conflicting state, a reviewed-source
  contradiction, unavailable required native GPU/presented execution, missing
  installed Rust 1.97.x or authorized wasm target, required unsafe, unavailable
  Surgeist profiles, or publication credentials/remote history failure.
