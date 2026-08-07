# P02-I02-S01-C09 Test Support And Model Planning Domains

## 1 Header

- Cycle: `P02/I02/S01/C09`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: published/read-back C08
  `8be93c6fa6cc2aba15760faf40136061febb275a`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `bd25c89790358054a2b51c77c5c2b83f71859cf1`, SHA-256
  `186eb7cf9366302ea5f16476720b3fc996083ea73a0af159d7794d3b0fb13e93`;
  M01-M04, M05.6, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `b7ce6d17a20c70dc06f68882d5347086e7c5546f`, SHA-256
  `e4b731ecb2c38543a6011402235d4e3ebc6a587d41badb876206d9f7f703d72a`;
  `C09 Test Support And Model Planning Domains`.
- Outcome: replace `src/tests.rs` with `src/tests/mod.rs`; establish
  `support.rs`; move model/style/frame-owned tests to `model.rs`, `style.rs`,
  and `frame.rs`; leave the exact runtime/platform/Vello remainder for C10.

## 2 Boundary

- Front door: `tests/mod.rs` declares the focused children and temporarily owns
  only GPU, resource, transaction, surface, readback, platform, internal-Vello,
  and their single-domain fixtures reserved for C10.
- Shared support: `tests/support.rs` owns only fixtures or semantic oracles used
  by at least two sibling test domains. A helper used by one child moves with
  that child; no `common`, `util`, or convenience aggregation is introduced.
- Model owner: geometry, core values and validation, scene, paint, image,
  layer, text, authored command, and model-only reference contract tests move to
  `tests/model.rs` with their single-domain fixtures.
- Style owner: image placement/repeat, backgrounds, decorations, filters,
  clips, masks, authored normalization, and style diagnostic tests move to
  `tests/style.rs` with their single-domain fixtures.
- Frame owner: route selection, semantic bounds/filter planning, graph
  construction/validation, graph-lowering views, runtime lowering, and pass
  closure tests move to `tests/frame.rs` with their single-domain fixtures.
- Preserve every test's name, feature/target cfg, operation, input, assertion,
  oracle, async behavior, and test attribute. Only module imports, paths, and
  the smallest real sibling visibility may change.
- A helper shared only because a move split its callers may enter `support.rs`
  only after direct caller inventory proves at least two sibling domains. It
  keeps its semantic name and may not wrap or forward the old monolith.
- Exact pre/post test counts and sorted names are transient relocation evidence,
  never a committed test, manifest, inventory, generated file, or closure gate.
- The C10 remainder is reported in the C09 handoff by domain, helper ownership,
  test count, and immutable candidate; no committed inventory or forwarding
  module is added.
- Production code, `src/lib.rs`, renderer hierarchy, manifests, docs, examples,
  dependencies, features, public API, test behavior, and product expectations
  are protected.
- Root/sibling integration, API artifacts, semantic cleanup, test deletion or
  consolidation, production refactoring, and C10 runtime/platform/Vello moves
  are excluded.

## 3 Evidence Policy And Landing

- API/dependency/feature/generated-artifact effect: none.
- Behavior/oracle effect: none. Every task is mechanical relocation backed by
  identical pre/post names, counts, commands, inputs, and assertions; no
  artificial RED applies.
- Structural name/count/helper-use inspection is transient workflow evidence.
  Add no raw-source parser test, plan-path test, closure test, architectural
  assertion, committed inventory, or file-size/count gate.
- Each worker records exact moved tests/helpers/imports/cfgs, helper caller
  inventory, visibility changes, pre/post focused/full results, sorted-name and
  count equality, protected diff, and every test disposition. Each task is one
  coherent commit with a fresh exact-range task review before its successor.
- `origin/main` and authority-remote `main` equal C08 before T01. Local `main`
  is the reviewed planning/status descendant; the worktree is clean.
- Work remains in this leaf and current worktree. Use installed tooling offline;
  do not acquire dependencies, targets, toolchains, linters, or software.
- Implementation remains unpushed until all tasks, final matrices, and holistic
  review are clean. Publish by compare-and-swap and authority readback.
- The active macOS window-smoke exception remains: do not rerun a hanging native
  smoke until the user requests; every non-smoke gate remains required.

## 4 Ordered Tasks

### 4.1 T01 Establish Test Front Door And Shared Support

- Replace `src/tests.rs` with `src/tests/mod.rs`; add test-private `support.rs`.
- Move only fixtures/oracles already proven to have callers in at least two
  exact final sibling domains: `model`, `style`, `frame`, `gpu`, `surface`,
  `platform`, or `vello`. Keep all tests and every
  single-domain helper in `mod.rs` for later tasks.
- Preserve the complete import/cfg surface; add no glob reexport or forwarding
  wrapper. Record the support helper and direct sibling-caller inventory.
- Dependency/intended commit: published C08 plus reviewed C09 plan; one test
  front-door and proven shared-support commit.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render font_data_rejects_unreadable_bytes_and_out_of_range_collection_indices
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render background_stack_normalization_paints_color_behind_layers
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_is_the_least_powerful_plan_for_effect_free_scenes
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: the full suite is identical; `mod.rs` is the real test front door;
  support contains no single-domain or convenience helper.

### 4.2 T02 Move Core Model, Geometry, Scene, And Paint Tests

- Add `tests/model.rs`. Move geometry/core-value validation, scene construction
  and deterministic encoding/stats, command, paint/color/gradient, and their
  single-domain fixtures.
- Keep image/layer/text tests in `mod.rs` until T03; move no style normalization,
  frame planning, GPU execution, or platform behavior.
- Dependency/intended commit: reviewed T01 head; one core-model test move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render scene_encoding_is_deterministic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render scene_stats_report_facts_without_renderer
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render paint_colors_convert_hsl_known_vectors
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gradients_expose_render_ready_geometry_and_stops
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render rect_try_from_kurbo_rejects_invalid_bounds
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: model-owned tests and helpers have one owner; names, cfgs,
  assertions, sorted full-suite names, and count are unchanged.

### 4.3 T03 Complete Image, Layer, Text, And Validation Model Tests

- Move public image/buffer/sampling model, layer composition/mask/opacity,
  text/font/glyph model, authored command, and model validation tests to
  `model.rs` with their single-domain fixtures.
- CPU reference helpers used only as a model oracle move with the tests; helpers
  also used by an exact `style`, `frame`, `gpu`, `surface`, `platform`, or
  `vello` sibling name `support.rs` directly after recorded caller proof.
- Dependency/intended commit: reviewed T02 head; one remaining-model test move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_buffer_accepts_exact_and_zero_area_lengths_and_round_trips_bytes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render layer_resolved_alpha_mask_applies_after_children_before_parent_composite
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render text_run_bounds_distinguish_unspecified_empty_and_ink
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render selected_glyph_preflight_validates_exact_outline_draw_settings
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render authored_layer_mask_and_filter_inputs_return_typed_diagnostics
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `model.rs` owns the complete M05.6 model domain; no runtime/GPU,
  style-normalization, or frame-planning test travels with it.

### 4.4 T04 Move Style Image, Background, And Decoration Tests

- Add `tests/style.rs`. Move style resource/image placement/repeat/attachment,
  background areas/stacks/normalization, border/outline/radius/fragment, and
  box-decoration tests with single-domain helpers.
- Keep filters/clips/masks in `mod.rs` until T05. Preserve authored order,
  diagnostics, normalized commands, and every existing assertion.
- Dependency/intended commit: reviewed T03 head; one image/background/decoration
  test move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_placement_auto_uses_intrinsic_size_and_position_ratio
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render image_repeat_plan_resolves_tile_rects_inside_clip_rect
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render background_stack_normalizes_image_layers_with_origin_clip_repeat_and_attachment
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render box_decoration_normalization_emits_four_independent_border_sides_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render background_box_decoration_integration_preserves_command_boundaries_across_fragments
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: style image/background/decoration tests have one semantic owner;
  test names, order-sensitive assertions, cfgs, full names, and count match.

### 4.5 T05 Complete Style Filter, Clip, And Mask Tests

- Move authored filters/drop shadows/filter regions, clip inputs/normalization,
  mask inputs/stacks/composition, and style capability diagnostics to
  `style.rs` with their single-domain fixtures.
- Frame graph and GPU/materialized execution tests remain in their existing
  owners even when they consume styled values; ownership follows tested
  condition and oracle, not shared vocabulary.
- Dependency/intended commit: reviewed T04 head; one filter/clip/mask test move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render authored_shadow_normalization_preserves_order_and_typed_boundaries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render clip_input_normalization_preserves_path_fill_rules_and_bounds
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render ordered_mask_layer_stacks_preserve_layer_and_composite_lists
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render mask_layer_stacks_report_specific_luminance_and_composite_diagnostics
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filter_region_models_reject_invalid_bounds_and_radii
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `style.rs` owns the complete M05.6 style domain without taking
  frame/runtime/GPU product oracles; the complete suite is unchanged.

### 4.6 T06 Move Frame Route, Bounds, And Filter-Planning Tests

- Add `tests/frame.rs`. Move plan selection, semantic command contributions,
  bounds/coordinate mapping, filter intent/kernel/edge/spatial planning, and
  their single-domain fixtures.
- Keep graph construction/validation/lowering/closure in `mod.rs` until T07.
- Dependency/intended commit: reviewed T05 head; one route/bounds/filter-plan
  test move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render direct_vello_is_the_least_powerful_plan_for_effect_free_scenes
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render signed_device_bounds_floor_minima_and_ceil_maxima
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render supported_scenes_produce_one_finite_backend_free_frame_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render transparent_resolved_alpha_mask_annihilates_unspecified_text_without_graph_selection
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
  ```

- Acceptance: route/bounds/filter-plan tests own no runtime encoder, backend,
  surface, or GPU oracle; names, cfgs, assertions, and full suite match.

### 4.7 T07 Complete Graph, Lowering, And Closure Tests

- Move semantic graph builder/validation/import/lifetime tests, graph-lowering
  views, runtime lowering/accounting/key facts, and pass executable-closure tests
  to `frame.rs` with their single-domain fixtures.
- Encoding, shader bytes, GPU pixels, resources, transactions, surfaces,
  readback, platform, and Vello tests remain in `mod.rs` for C10.
- Dependency/intended commit: reviewed T06 head; one graph/lowering/closure test
  move.
- Before and after run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_builder_rejects_forward_stale_and_read_write_aliases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_preserves_dependencies_and_last_use_releases
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_preparation_rejects_unsupported_passes_without_resource_or_cache_mutation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composition_graph_orders_clip_mask_opacity_blend_and_nested_layers
  CARGO_NET_OFFLINE=true cargo fmt --check
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
  CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
  ```

- Acceptance: `frame.rs` owns the complete M05.6 frame domain; runtime execution
  tests do not move; complete sorted names/count and behavior are identical.

### 4.8 T08 Reconcile Support And Record C10 Remainder

- Reconcile `mod.rs`, `model.rs`, `style.rs`, `frame.rs`, and `support.rs`.
  Remove transitional imports/visibility and any helper from support with fewer
  than two sibling-domain callers; add no shim, glob, lint suppression, or test.
- Directly inventory the remaining `mod.rs` tests/helpers by GPU, resource/cache,
  transaction, surface/publication/cancellation/readback, platform, and Vello
  domain for the C10 handoff. Record it in the task/handoff evidence only.
- Dependency/intended commit: reviewed T07 head; one support/front-door
  reconciliation commit.
- Before and after run all T01-T07 focused conditions plus:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_readback_renderer_front_door_is_async
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_render_submits_one_transaction_and_publishes_once
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window surface_operation_matrix_covers_every_kind_state_and_duplicate_transition
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

- Acceptance: five test files exist with exact M05.6 ownership; support contains
  only multi-sibling helpers; the C10 remainder is exact and uncommitted; no
  test, name, cfg, product oracle, or public/production artifact changed.

## 5 Verification And Completion

After all tasks are task-review `CLEAN`, make the status-only `complete` commit,
run this matrix, obtain a distinct holistic `CLEAN` review, repeat at unchanged
HEAD, and CAS-publish with authority readback:

```sh
set -euo pipefail
test -z "$(git diff 8be93c6fa6cc2aba15760faf40136061febb275a -- . \
  ':(exclude)src/tests.rs' ':(exclude)src/tests/**' \
  ':(exclude)plans/cycles/cohesive-module-decomposition-c09-test-model-planning-domains.md')"
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
test -z "$(git ls-files -- Cargo.lock)"
owned_rust_files=("${(@f)$( { git ls-files -- '*.rs'; git ls-files --others --exclude-standard -- '*.rs'; } | sort -u )}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then exit 1; else test "$?" -eq 1; fi
git diff --check 8be93c6fa6cc2aba15760faf40136061febb275a..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

Before and after every task and final matrix, record exact `#[test]`/equivalent
attribute counts and sorted names for each feature configuration as ephemeral
evidence; equality is required, but no source-parser test or artifact is added.
The two native smoke executables remain under the active user-requested rerun
exception; all non-smoke checks proceed. Root integration remains excluded.

The C09-to-C10 handoff reports immutable candidate/readback SHA, reviewed plan
revision, task/holistic verdicts, exact child ownership, helper caller inventory,
ephemeral name/count equality, the domain-classified C10 remainder, smoke
disposition, clean status, and explicit root-integration exclusion.
