# P02-I02-S01-C02 Frame Planning

## 1 Header

- Cycle: `P02/I02/S01/C02`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: `92b664bdb91f927bf38a4732c42ea89a5b822618`, the published
  P02-I02 C01 candidate verified on local and authority-remote `main`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `3365399fa2411efd5cd8fcfdfe74d4b756cd6a79`, normalized SHA-256
  `892c5c1c2162a07c83bc124c3687e05ff62c6906930c154f25117792ec63d035`;
  sections M01-M05.1 and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, normalized SHA-256
  `a45655293a54b5bf0986d508d6d2b278af68e5267dc373d8f5d56cb84e58c74b`;
  entry `C02 Frame Planning`.
- Outcome: replace `src/frame.rs` with an explicit private frame hierarchy
  while preserving plan selection, semantic bounds and filter facts, graph
  identities and construction, validation precedence, lowering views, and all
  crate-visible frame contracts.

## 2 Boundary

- Owned input: `src/frame.rs` and only import/visibility repairs required by
  faithful relocation of its existing items.
- Required output:
  `src/frame/{mod,bounds,filter,graph,validate,lower,test_support}.rs`.
- `frame/mod.rs` retains `FrameContext`, `FramePlan`, `DirectVelloPlan`, and the
  genuine coordination that selects and returns a plan. Each child owns exactly
  the responsibility assigned by specification M05.1.
- Existing `src/lib.rs` declarations/reexports, public paths, renderer/pass
  callers, diagnostics, validation order, graph identity values, resource and
  pass order, lowering facts, and frame-test observations remain unchanged.
- The allowed `frame`/`renderer` edge remains: frame coordination consumes
  `renderer::options::Antialiasing`, and renderer dispatch consumes the frame
  front door. No new mutual module-directory edge is introduced.
- No semantic rename, algorithm or oracle change, compatibility shim,
  forwarding-only layer, copied definition, `include!`, `#[path]`, generated
  concatenation, glob-reexport maze, generic helper module, source parser,
  inventory test, or numerical size/count gate is allowed.
- Root, sibling, adapter, API-artifact, gitlink, public hierarchical-front-door,
  dependency, feature, target, example, manifest, and correctness-fix work is
  excluded.
- Commands use installed artifacts offline. No acquisition, installation,
  bootstrap, or update is authorized.

## 3 Impacts

- Public API and caller migration: none; crate-root exports and construction
  paths remain unchanged.
- Behavior and diagnostics: none.
- Dependencies, features, targets, MSRV, docs, and examples: unchanged.
- Test impact: import/module relocation only; test names, operations, inputs,
  assertions, and oracles remain unchanged.
- Generated artifacts: none in this leaf; root-owned artifacts remain untouched.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Establish Bounds And Filter Planning Owners

- Convert `src/frame.rs` to `src/frame/mod.rs` without changing its crate module
  name. Move semantic command contributions, logical bounds, finite spatial
  primitives, coordinate transforms, and checked arithmetic to `bounds.rs`.
- Move resolved filter intent, kernel support, edge policy, filter roles, and
  filter spatial planning to `filter.rs`. Keep `FrameContext` and plan-selection
  coordination in `mod.rs`; use explicit imports and the narrowest visibility.
- Before moving items, run and record these characterization commands; each must
  select at least one test and pass. Repeat the same commands after the move:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render signed_device_bounds_floor_minima_and_ceil_maxima
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render bounded_capture_transform_preserves_signed_origin_texel_centers_and_scale
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render rank_deficient_transform_produces_explicit_empty_spatial_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render logical_bounds_preserve_large_finite_translation_until_frame_scale_resolution
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_fusion_preserves_each_source_clamp
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filter_scalar_lowering_handles_f32_f64_exponents_and_huge_angles_finitely
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_blur_mirrors_at_semantic_bounds_not_allocation_padding
  ```

- Acceptance:

  ```sh
  test ! -e src/frame.rs
  for required in src/frame/mod.rs src/frame/bounds.rs src/frame/filter.rs; do test -f "$required"; done
  test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/frame || true)"
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  CARGO_NET_OFFLINE=true cargo check -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  git diff --check
  ```

- Intended commit: one mechanical bounds/filter ownership point with a complete
  move and visibility inventory.

### 4.2 T02 Move Semantic Graph Construction

- Start only from the reviewed T01 head. Move semantic graph generations,
  identities, resources, passes, results, builders, planners, capture-locality
  facts, Vello spans, clip coverage, composites, imports, backdrop reads, and
  graph construction to `graph.rs`.
- Keep plan selection and returned plan coordination in `mod.rs`. Do not move
  validation or graph-to-runtime lowering into `graph.rs`, and do not create an
  alternate graph model for temporary compilation.
- Run before and after:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render supported_scenes_produce_one_finite_backend_free_frame_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gpu_graph_is_selected_only_for_supported_custom_requirements
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_builder_rejects_declaration_after_final_present
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_builder_rejects_forward_stale_and_read_write_aliases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composition_graph_orders_clip_mask_opacity_blend_and_nested_layers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render maximal_vello_spans_preserve_authored_command_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_clip_coverage_is_one_vello_capture_of_ordered_render_clips
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render backdrop_graph_reads_completed_parent_once_and_preserves_group_order
  ```

- Acceptance: `src/frame/graph.rs` exists; every M05.1 graph-construction owner
  has one definition there; the same common Cargo/structural acceptance commands
  from T01 pass; `src/lib.rs`, `Cargo.toml`, README, and examples remain unchanged.
- Intended commit: one semantic-graph ownership point with its complete move and
  visibility inventory.

### 4.3 T03 Move Validation And Lowering

- Start only from the reviewed T02 head. Move semantic graph structure,
  metadata, import, lifetime, anchor, and lowering-precondition validation to
  `validate.rs`.
- Move graph-lowering public-to-crate views and conversion from semantic graph
  facts to lowering facts to `lower.rs`. Preserve validation precedence,
  diagnostic payloads, resource/pass order, last-use facts, sampling facts, and
  finite spatial descriptors byte-for-byte and value-for-value.
- Run before and after:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_preserves_dependencies_and_last_use_releases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render blur_and_drop_shadow_graph_preserves_order_edges_and_lifetimes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_graph_preserves_authored_order_clamps_and_exact_lifetimes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_planning_requires_explicit_text_ink_bounds_only_for_bounded_subtrees
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render zero_capture_graph_spine_is_rejected_before_preparation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_executor_accepts_only_clear_capture_canonicalize_source_over_and_present
  ```

- Acceptance: `src/frame/{validate,lower}.rs` exist; graph validation and
  lowering responsibilities have one explicit owner each; the common
  Cargo/structural acceptance commands from T01 pass; no validation expectation
  or runtime-lowering oracle changes.
- Intended commit: one validation/lowering ownership point with its complete
  move and visibility inventory.

### 4.4 T04 Move Test Support And Reconcile The Frame Front Door

- Start only from the reviewed T03 head. Move frame-owned observations,
  malformed-graph probes, fault inputs, forced graph fixtures, and standalone
  finite/completeness observations to `test_support.rs` under `#[cfg(test)]`.
- Reconcile `frame/mod.rs` to the required M05.1 result: child declarations,
  explicit current-contract reexports, `FrameContext`, `FramePlan`,
  `DirectVelloPlan`, and only genuine plan-selection/return coordination. Move a
  one-domain test helper into its owning child rather than making it shared.
- Run before and after:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_builder_rejects_scheduling_after_final_present
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_builder_rejects_forward_stale_and_read_write_aliases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_base_color_is_initialized_once_and_isolation_is_transparent
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render supported_scenes_produce_one_finite_backend_free_frame_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render negative_bounds_and_subpixel_transforms_do_not_shift_capture
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render transparent_resolved_alpha_mask_annihilates_unspecified_text_without_graph_selection
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render zero_opacity_backdrop_preserves_foreground_without_graph_boundary
  ```

- Acceptance: all seven required `src/frame` files exist; `test_support.rs` is
  test-only; `mod.rs` contains no M05.1 responsibility assigned to a child; the
  common Cargo/structural acceptance commands from T01 pass; the full C02 matrix
  below passes.
- Intended commit: one frame-front-door reconciliation point with the complete
  test-support move and final visibility inventory.

## 5 Verification And Completion

Before each task, the worker records the exact focused characterization results;
after the move, the same operations and oracles pass. Each task requires a
separate task-review `CLEAN` verdict. After all tasks are clean, the coordinator
makes a status-only `complete` commit, runs this matrix, obtains a distinct
holistic `CLEAN` review over the exact cycle range, repeats this matrix at the
unchanged reviewed head, and publishes with authority-remote readback:

```sh
set -euo pipefail
test ! -e src/frame.rs
for required in src/frame/mod.rs src/frame/bounds.rs src/frame/filter.rs \
  src/frame/graph.rs src/frame/validate.rs src/frame/lower.rs \
  src/frame/test_support.rs; do
  test -f "$required"
done
test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/frame || true)"
test -z "$(git diff 92b664bdb91f927bf38a4732c42ea89a5b822618 -- src/lib.rs Cargo.toml README.md examples)"
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
git diff --check 92b664bdb91f927bf38a4732c42ea89a5b822618..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The live smoke executables must render and exit on the native host. Every
unsafe-scan match is classified; any executable match blocks completion. The
publication head is immutable after holistic review. Root integration remains
excluded.

The C02-to-C03 leaf handoff reports the immutable published C02 candidate and
authority-remote readback SHA, the exact planning revision and clean task and
holistic verdicts, and the stable semantic-planning and graph-lowering front
doors now available to C03 shader/resource work. It confirms clean status and
retains the explicit exclusion of root integration.
