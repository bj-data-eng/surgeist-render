# GPU Render Pipeline C06 Frame Spatial And Filter Planning

## Header
- Cycle: `C06`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `d5e4c2a0c1fe879d98b6cabdf86a667e4277be0f` (C05).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: S15, S17, S19, planning portions of S20 and S22, S28, and C06-applicable S31-S32 evidence; inherited S07 resolved-frame phases, S10 bounded-text behavior, S13 typed unresolved/runtime diagnostics, and applicable S36-S37 feature/dependency/verification rules apply where those sections define a named C06 invariant or gate.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `c1a203393b2549603c0a0d5698099f55018abe2e`, normalized SHA-256 `70b345be31ac5bf4fcae72e2d3c5901c8e04ad0b2c26080851bd2c50d7150cde`, entry `C06 Frame Spatial And Filter Planning`.
- Outcome: make one private, backend-free resolved-frame planner choose the least-powerful direct plan or a finite validated semantic GPU graph with exact command dependencies, maximal Vello spans, signed spatial mappings, ordered filter bounds, explicit text-effect bounds, and fan-out lifetimes suitable for C07 lowering.

## Boundary
- C05 supplies stable normalized public input, surfaces, parameters, internal Vello raster execution, atomic publication, and async readback. C06 consumes only normalized semantic commands plus a private immutable `FrameContext`; planning owns no WGPU object, queue, encoder, submission, map, poll, readback, pipeline, shader, or texture lease.
- `FramePlan` is the private resolved-frame phase and is exactly `DirectVello(DirectVelloPlan)` or `GpuGraph(GpuRenderGraph)`. Direct contains one normalized Vello-encodable tree, output mapping, antialiasing, and base color. Graph resources and pass intents contain semantic roles, logical bounds, signed origins, positive extents, mappings, dependencies, and read counts, but no concrete WGPU format or allocation.
- The planner consumes an owned `RenderCommands` value through a named fallible `plan_for(FrameContext)` conversion. It returns no partial plan. Direct planning does not require text ink bounds; any command subtree that must become a bounded graph capture requires explicit `TextRunBounds::Ink`, treats `Empty` as no pixels, and reports `UnresolvedResourceKind::TextRunInkBounds` for `Unspecified` without estimating glyph geometry.
- Spatial planning preserves logical bounds separately from signed device origin and positive extent. Outward conversion floors minima and ceils maxima with checked arithmetic; texel centers retain the signed origin. Local effects use surface scale times the largest affine singular value; a zero singular value is an explicit empty result. C07 owns selected-device dimension-limit and working-format resolution against the already-computed extent.
- Filter planning records every authored operation in order. A color run may share one semantic pass intent only while retaining each source operation and clamp boundary. Zero blur is identity; nonzero blur uses inclusive support through `ceil(2.5 * sigma * raster_scale)` and transparent-black ordinary or semantic-border mirror backdrop edges. Drop shadow unions the unchanged source with continuously offset blurred SourceAlpha and retains the source through both consumers.
- C06 removes command-owned cloned backdrop source lists. The graph references the completed current-parent resource. The transitional materialized backdrop/mask execution may derive its temporary prefix during traversal to preserve current pixels until C12/C13, but it is not graph evidence and may not become part of `FramePlan`.
- C06 excludes C07 working-format selection, device-limit rejection, concrete texture/resource leasing, shaders, executable pass lowering, and resource statistics; C08+ graph execution and pixels; C13 removal of transitional CPU/materialized execution and superseded public algorithm phases; C14 docs/platform closure; root integration/API artifacts; dependencies/acquisition; compatibility shims; and generated artifacts. Owned Rust remains free of `unsafe`.

## Impacts
| Area | C06 record |
| --- | --- |
| Public API | Breaking/corrective: add `FilterDropShadow`; change `FilterOpKind::DropShadow` and `FilterOp::drop_shadow` to that intrinsically valid payload; add fallible `try_drop_shadow(Shadow)`; reject `FilterBlur` values above 256. No shim. Existing public materialized/compiled algorithm phases remain only until C13. |
| Dependencies/features | Unchanged. Normal, dev, target-specific dev roles and all four native plus one wasm-supported feature states remain S36. |
| Artifacts/docs/MSRV | No dependency, fixture, license, generated, README, or example delta; changed public items receive source docs. Rust 1.97 and Rust 2024 remain required. |
| Unsafe | Unchanged: `#![forbid(unsafe_code)]` remains effective and all Surgeist-owned Rust must pass the explicit unsafe-absence scan. |
| Root/handoff | No root edit. The published candidate reports the filter payload/range break; C07 receives only validated semantic resources, pass intents, mappings, and extents. |

## Tasks
Define these functions in each task shell before focused commands; they reject zero or ambiguous test selection and owned unsafe source.
```sh
prepare_c06_toolchain() {
  local candidate installed_toolchains version
  installed_toolchains="$(rustup toolchain list)" || return $?
  C06_TOOLCHAIN=""
  for candidate in $(printf '%s\n' "$installed_toolchains" | awk '{ print $1 }'); do
    version="$(rustup run "$candidate" rustc --version)" || continue
    case "$version" in
      "rustc 1.97."*) C06_TOOLCHAIN="$candidate"; break ;;
    esac
  done
  test -n "$C06_TOOLCHAIN" || { printf 'required installed Rust 1.97.x toolchain is unavailable; acquisition is not authorized\n' >&2; return 1; }
  export C06_TOOLCHAIN
}

require_c06_wasm_target() {
  local sysroot
  prepare_c06_toolchain || return $?
  sysroot="$(rustup run "$C06_TOOLCHAIN" rustc --print sysroot)" || return $?
  test -d "$sysroot/lib/rustlib/wasm32-unknown-unknown/lib" || { printf 'wasm32-unknown-unknown is not installed for %s; acquisition is not authorized\n' "$C06_TOOLCHAIN" >&2; return 1; }
}

cargo_c06() {
  prepare_c06_toolchain || return $?
  CARGO_NET_OFFLINE=true rustup run "$C06_TOOLCHAIN" cargo "$@"
}

cargo_c06_wasm() {
  require_c06_wasm_target || return $?
  CARGO_NET_OFFLINE=true rustup run "$C06_TOOLCHAIN" cargo "$@"
}

run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  prepare_c06_toolchain || return $?
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(cargo_c06 test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(cargo_c06 test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    cargo_c06 test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    cargo_c06 test -p surgeist-render "tests::$name" -- --exact
  fi
}

assert_no_owned_unsafe() {
  local output scan_status
  if output="$(git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg -n --pcre2 '#\s*!?\[\s*(?:allow|expect)\s*\([^)]*\bunsafe(?:_[A-Za-z0-9_]+)?\b|#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' 2>&1)"; then
    printf '%s\n' "$output" >&2
    return 1
  else
    scan_status=$?
    test "$scan_status" -eq 1 && return 0
    printf '%s\n' "$output" >&2
    return "$scan_status"
  fi
}
```

**C06-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
prepare_c06_toolchain
cargo_c06 fmt --check
cargo_c06 check -p surgeist-render
cargo_c06 test -p surgeist-render
cargo_c06 clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo_c06 test -p surgeist-render --features render-window
cargo_c06 clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
cargo_c06 test -p surgeist-render --features render-web
cargo_c06 clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
cargo_c06 test -p surgeist-render --features render-window,render-web
cargo_c06 clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
cargo_c06 check -p surgeist-render --all-targets
cargo_c06 check -p surgeist-render --all-targets --features render-window,render-web
native_getrandom_tree="$(cargo_c06 tree -p surgeist-render -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
test -z "$(git ls-files -- Cargo.lock)"
assert_no_owned_unsafe
cargo_c06_wasm check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(cargo_c06_wasm tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
```
`prepare_c06_toolchain` runs before every native Cargo process and selects the first exact installed toolchain name whose compiler reports Rust `1.97.x`, so an advanced `stable` cannot replace MSRV evidence. `cargo_c06_wasm` separately proves the wasm target in that selected toolchain's existing sysroot immediately before each wasm Cargo process; a missing target blocks only after the preceding native evidence. No `+stable`, bare Cargo, rustup target mutation, or acquisition command is permitted. The wasm command is compile evidence only; browser execution remains root-owned. `assert_no_owned_unsafe` converts only ripgrep's no-match status to success and rejects matches or command failures. T1-T6 use the `default` profile for every named test below.

For each task's RED, first add the named test and only the narrow `#[cfg(test)]` observation needed to express the pre-task behavior with stable primitive/test-only values. That observation must wrap the existing owner, contain no final algorithm or unconditionally compiled API, and be retained as a production-owner observation or removed after GREEN. Run the stated `run_exact_test` command before production changes and require the stated assertion failure; compilation, listing, lint, panic-before-assertion, or harness failure is invalid RED evidence. Retain the same behavioral test for GREEN.

**T1. Make filter drop shadow intrinsically executable**
- Area/outcome: `src/{lib.rs,style.rs,filter.rs,command.rs,image.rs,reference.rs,tests.rs}` and public source docs; add the exact S22 `FilterDropShadow`, store it in `FilterOpKind`, make direct construction infallible only from that valid type, and retain one fallible authored `Shadow` conversion. Enforce the S20 filter-blur range while adapting transitional private materialization/oracle code without changing pixels. Box-shadow bounds use a distinct private uncapped Vello-compatibility calculation over validated `Shadow::blur`, never `FilterBlur`.
- RED/acceptance: add three baseline-compiling probes. `drop_shadow_model_cannot_express_inset_spread_or_non_solid_paint` uses authored `Shadow` plus a cfg-test observation of whether the current broad payload admits each invalid state; before production edits, its exact command must fail only at `broad filter drop-shadow payload remains constructible`. `filter_blur_rejects_values_above_256_without_clamping` uses the existing constructor; its exact command must fail only at `next representable value above 256 was accepted`. `box_shadow_bounds_do_not_reuse_capped_css_filter_blur` observes the current bounds dependency and a finite blur above 256; its exact command must fail only at `box-shadow bounds still depend on CSS FilterBlur validation`. Pass requires private fields, exact constructor/accessors/traits, finite offset/color, outer kind, zero spread, solid paint, typed existing diagnostics, closed `[0, 256]` filter-blur acceptance, and no alias/forwarder from broad `Shadow`; the large box shadow retains fallible explicit bounds rather than becoming unknown, and box-shadow constructor/render behavior remains unchanged.
- Commands: before production edits run each RED command exactly: `run_exact_test drop_shadow_model_cannot_express_inset_spread_or_non_solid_paint`; `run_exact_test filter_blur_rejects_values_above_256_without_clamping`; `run_exact_test box_shadow_bounds_do_not_reuse_capped_css_filter_blur`. For GREEN rerun all three, then `run_exact_test css_drop_shadow_rejects_non_zero_spread`; `run_exact_test css_drop_shadow_rejects_non_solid_shadow_paint`; `run_exact_test materialized_image_filters_preserve_color_and_blur_order`; then `C06-CHECK`.
- Depends: none. Commit: `Model executable filter drop shadows`.

**T2. Establish checked signed spatial planning**
- Area/outcome: private `src/frame.rs`, `src/{lib.rs,command.rs,tests.rs}`; introduce the resolved-frame `FrameContext` and least-powerful logical-empty/nonempty bounds, signed device origin, positive extent, raster scale, and texel-center mapping types. Move fallible bounds contribution out of lossy `Option`/`.ok()` paths so overflow/non-finite input is never guessed away.
- RED/acceptance: add the four named probes against a cfg-test primitive observation that wraps the current `DevicePixelConversionPolicy` and lossy command-bounds path. Before production edits, run each exact command and require only its named assertion failure: `run_exact_test signed_device_bounds_floor_minima_and_ceil_maxima` at `logical and device spatial phases remain collapsed`; `run_exact_test negative_and_fractional_origins_preserve_texel_center_mapping` at `texel-center mapping is absent`; `run_exact_test largest_singular_value_raster_scale_preserves_local_effect_space` at `local raster scale does not use the largest singular value`; and `run_exact_test zero_singular_value_produces_an_empty_plan` at `degenerate spatial output was erased instead of represented as empty`. Pass preserves negative/fractional origins separately from allocation extent; uses checked floor/ceil/difference conversion; maps texel `(i,j)` at `origin + ((i+0.5)/scale,(j+0.5)/scale)`; computes local raster scale from the largest 2D affine singular value times finite positive surface scale; returns explicit empty for zero singular value/degenerate bounds; and returns typed input failure for non-finite, integer-overflowing, or non-`u32` extents without a 1x1 substitute.
- Commands: `run_exact_test signed_device_bounds_floor_minima_and_ceil_maxima`; `run_exact_test negative_and_fractional_origins_preserve_texel_center_mapping`; `run_exact_test largest_singular_value_raster_scale_preserves_local_effect_space`; `run_exact_test zero_singular_value_produces_an_empty_plan`; then `C06-CHECK`.
- Depends: T1. Commit: `Establish signed frame spatial mapping`.

**T3. Fold authored filters into ordered semantic plans**
- Area/outcome: `src/{filter.rs,frame.rs,image.rs,reference.rs,tests.rs}`; add private algorithm-phase filter plans whose steps record source/result logical bounds, spatial mapping, edge policy, and operation intent in authored order. Preserve transitional byte execution behind separate private types; no pixel buffer enters the frame plan.
- RED/acceptance: add both named tests against a cfg-test ordered-step observation derived only from the current `MaterializedImageFilterPipeline`, `FilterRegionPlan`, and authored operations. Before production edits, `run_exact_test filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order` must compile and fail only at `legacy filter classifiers do not produce ordered result-bound records`; `run_exact_test color_filter_fusion_preserves_each_source_clamp` must fail only at `fused intent lost authored clamp boundaries`. Pass elides zero blur; computes nonzero support with inclusive `ceil(2.5 * sigma * raster_scale)` taps; preserves ordinary transparent-black versus backdrop semantic-border mirror edges; keeps every source color operation/clamp boundary in a fused run; unions drop-shadow output with the unchanged source at continuous signed offset; and reports an empty result rather than allocating for degenerate input.
- Commands: run both named tests separately with `run_exact_test`; `run_exact_test filter_blur_policy_zero_radius_produces_zero_outset`; `run_exact_test materialized_filters_before_drop_shadow_shape_current_alpha_mask`; `run_exact_test materialized_filters_after_drop_shadow_apply_to_composed_output`; then `C06-CHECK`.
- Depends: T2. Commit: `Plan ordered filter bounds`.

**T4. Validate graph identities, dependencies, and fan-out**
- Area/outcome: `src/{frame.rs,tests.rs}`; implement one closed graph builder with generation-aware private resource/pass IDs, semantic resource descriptors, finite pass intents, explicit producer/read edges, remaining-read lifetimes, one root working image, and one final present intent. C07—not this builder—resolves formats, allocations, pipelines, or executable passes.
- RED/acceptance: add all three probes against a cfg-test edge/lifetime observation over the pre-task `LayerPassPlan`/offscreen model, using stable primitive states rather than missing graph symbols. Before production edits, require only these assertion failures: `run_exact_test graph_builder_rejects_forward_stale_and_read_write_aliases` at `no closed graph validator rejected the invalid edge sequence`; `run_exact_test drop_shadow_source_fanout_lives_through_both_consumers` at `drop-shadow source has no two-consumer lifetime`; and `run_exact_test graph_base_color_is_initialized_once_and_isolation_is_transparent` at `surface base and isolation clears are not modeled exactly once`. Pass rejects wrong-generation, unknown, released, forward, read/write-same-subresource, duplicate-producer, read-count, orphan-result, missing/duplicate root, missing/duplicate present, nontransparent capture-base, and repeated surface-base initialization states. All consumers are recorded before scheduling; reads decrement only when scheduled; resources become releasable exactly after the last read; drop shadow reads the same immutable source once for SourceAlpha and once for unchanged source-over; empty/no-op results have no texture descriptor.
- Commands: `run_exact_test graph_builder_rejects_forward_stale_and_read_write_aliases`; `run_exact_test drop_shadow_source_fanout_lives_through_both_consumers`; `run_exact_test graph_base_color_is_initialized_once_and_isolation_is_transparent`; then `C06-CHECK`.
- Depends: T3. Commit: `Validate semantic render graph lifetimes`.

**T5. Partition maximal Vello spans against current-parent dependencies**
- Area/outcome: `src/{command.rs,frame.rs,renderer.rs,tests.rs}`; classify normalized trees, choose direct only when no custom result is required, partition graph scenes into maximal consecutive Vello spans without crossing semantic group boundaries, and replace command-owned backdrop source clones with a current-parent graph edge. Preserve the narrow transitional materialized executor by deriving any temporary prior-sibling prefix during traversal rather than storing it in normalized commands or the graph.
- RED/acceptance: add the six named probes against a cfg-test route/dependency observation that wraps current `LayerPassPlan`, command order, text bounds, and `RenderBackdropCapture` without calling a missing partitioner. Before production edits, require only these assertion failures: `run_exact_test direct_vello_is_the_least_powerful_plan_for_effect_free_scenes` at `effect-free scene has no direct frame plan`; `run_exact_test gpu_graph_is_selected_only_for_supported_custom_requirements` at `custom requirement has no semantic graph plan`; `run_exact_test maximal_vello_spans_preserve_authored_command_order` at `authored Vello commands are not partitioned into maximal spans`; `run_exact_test backdrop_plan_depends_on_current_parent_not_cloned_commands` at `backdrop dependency is stored as cloned commands instead of current parent`; `run_exact_test graph_planning_requires_explicit_text_ink_bounds_only_for_bounded_subtrees` at `bounded graph text lacks an exact unresolved-bounds result`; and `run_exact_test supported_scenes_produce_one_finite_backend_free_frame_plan` at `supported scene has no finite frame plan`. Pass selects graph only for supported alpha-mask/backdrop/image/composite requirements; keeps fully local Vello-only groups within a capture; separates external-parent blend/effect ownership; preserves authored command order; captures effect sources before outer filter/clip/mask/opacity/blend; never creates graph-to-Vello image re-entry; represents backdrop as a read of the completed parent; permits unspecified text bounds only on direct paths; treats empty text as no pixels; and returns `TextRunInkBounds` unresolved for an unspecified bounded graph subtree.
- Commands: run all six RED tests separately with `run_exact_test`; then `C06-CHECK`.
- Depends: T4. Commit: `Partition frame plans by semantic dependency`.

**T6. Make planning the required pre-execution frame gate**
- Area/outcome: integrated `src/{renderer.rs,frame.rs,command.rs,filter.rs,tests.rs}`; invoke the owned `RenderCommands::plan_for(FrameContext)` conversion after normalization and before transitional materialization or any frame GPU operation. Direct encoding consumes only the validated direct tree; graph transitional execution may use the normalized source retained by orchestration, but cannot mutate, backfill, or bypass the validated plan.
- RED/acceptance: retain T5's green `supported_scenes_produce_one_finite_backend_free_frame_plan`, then add `render_plans_before_transitional_effect_execution` against a cfg-test renderer-stage observation initialized from the current render flow. Before production edits, `run_exact_test render_plans_before_transitional_effect_execution` must compile and fail only at `renderer entered transitional effect execution before one validated plan`. Pass proves every currently supported direct/mask/bounded-backdrop scene yields exactly one complete direct/graph plan deterministically; failures expose no partial graph; repeated planning has no renderer/backend/resource side effect; `frame.rs` imports no backend/WGPU execution authority; current direct/mask/backdrop render/readback pixels and failure atomicity remain unchanged; and C07 can inspect complete semantic resource intents, pass intents, mappings, extents, and lifetimes without re-walking commands or inventing bounds.
- Commands: `run_exact_test render_plans_before_transitional_effect_execution`; `run_exact_test supported_scenes_produce_one_finite_backend_free_frame_plan`; `run_exact_test render_materializes_bounded_backdrop_capture_from_prior_siblings`; `run_exact_test layer_resolved_alpha_mask_applies_after_children_before_parent_composite`; `run_exact_test headless_render_can_be_read_back`; run the final guards below; then `C06-CHECK`.
- Depends: T5. Commit: `Require validated frame planning`.

## Completion
- Require all six ordered task ranges and clean task reviews. Every supported normalized scene returns one finite validated direct or graph plan before effect/backend execution; direct remains least-powerful; graph IDs and lifetimes are generation-safe; filter/spatial bounds are exact and signed; text bounds are never inferred; backdrop depends on current parent; and planning owns no WGPU execution authority.
- Run `C06-CHECK` and every T1-T6 named test above through `run_exact_test`; require a clean worktree. Final guards are: `rg -n '\bwgpu\b|GpuOperation|queue\.|\.submit\(|map_async|Device::poll|read_texture_rgba|pollster' src/frame.rs`; `rg -n 'source_commands|previous_siblings\s*=\s*normalized\.clone' src/command.rs`; `rg -n 'pub use .*frame|FramePlan|GpuRenderGraph' src/lib.rs`; each exits `1` clean. Require `src/frame.rs` to contain `FrameContext`, `FramePlan`, direct/graph variants, checked signed spatial mapping, generation-aware resource/pass identities, recorded reads, and completed-graph validation; require no raw graph identity to cross `frame.rs` or the public front door.
- After cycle acceptance, follow `$surgeist-agent`'s canonical implementation-cycle and automated landing/publication gates, then its crate-candidate handoff contract for C07.
- Block only for an unowned worktree conflict, contradiction in the reviewed specification/sequence, unavailable required native GPU execution, missing already-installed Rust 1.97/wasm target support, or unavailable required Surgeist agent profile. No substitute target, skipped adapter, dependency/toolchain acquisition, compatibility shim, CPU fallback expansion, guessed bounds, or backend-bearing plan is allowed.
