# GPU Render Pipeline C10 Ordered Color Filter Execution

## Header
- Cycle: `C10`; owner: `surgeist-render`; status: `reviewed`.
- Cycle base and published prerequisite: `1c4ae3b2547fbe3c42d5519caee902303269ccd1` (C09).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: S16, S18, S20-S21, S27-S28, C10-applicable S30-S34, and per-cycle S36-S37 evidence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `562478db06184de64d6d5fad7ed134d99e2ab0f9`, normalized SHA-256 `9fb83aeebf2bcd2a581241e97c7cdde58942c8001f54c195b5a54a090908f1ba`, entry `C10 Ordered Color Filter Execution`.
- Outcome: execute every supported authored color function and legal adjacent fusion through one ordered GPU color-filter pass, in high and reduced working formats, with exact S21 clamp and finite-scalar semantics.

## Boundary
- C10 accepts the C09 executable graph plus nonempty `ColorFilter(Some(_))` passes whose one source and distinct result preserve the same finite spatial descriptor, use nearest sampling with `ClampToExtent`, contain only the eight S21 operations, and retain `ClampStraightRgbaToUnitThenPremultiply` after every authored operation. It still rejects `CopyBackdrop`, either blur pass, `DropShadowColorize`, `Composite(DropShadow)`, missing payloads, malformed order/bindings, and unsupported output before allocation or encoding.
- The authored `FilterList`, `FilterOp`, and `FilterOpKind` remain the public semantic input. Algorithm color runs remain private and preserve authored order. Graph lowering performs one explicit algorithm-to-runtime conversion into private finite GPU scalar forms; backend types, operation records, buffers, shader keys, and working formats remain private.
- `UnitFilterAmount` lowers to a checked nearest `f32`. A nonnegative `FilterAmount` lowers from its finite `f64` bits to `{ zero, mantissa: f32, exponent: i32 }`, with positive mantissa in `[0.5, 1)`, and renormalizes a rounded `1.0` to `0.5` plus one exponent. `FilterAngle` first applies `rem_euclid(TAU)` in `f64`, then stores checked finite `f32` sine and cosine. No unbounded amount is converted directly to `f32`.
- One tracked `src/shaders/color_filter.wgsl` samples the premultiplied source, safely unpremultiplies, applies each S21 operation in authored order, clamps straight RGB and alpha after each operation, and premultiplies before continuing. Opacity scales premultiplied RGB and alpha at its authored position. The amount helper compares normalized products with clamp distances before `ldexp`, so `f64::MAX`, subnormal amounts, and near-gray saturation never create infinity or NaN.
- A fused run means one GPU pass with one ordered operation buffer; it never means matrix collapse or removal of a source clamp. Identity operations remain explicit in C10. The buffer is exactly one 16-byte header (`operation_count: u32` plus three zero `u32` pads) followed by `N` 32-byte WGSL-aligned records carrying only the operation tag, zero flag, exponent, finite scalar payload, and zero padding. Construction first converts `N` to `u32`, then computes `16 + 32 * N` in checked `u64`; the result must be no greater than both `Device::limits().max_buffer_size` and `u64::from(Device::limits().max_storage_buffer_binding_size)`.
- Count conversion failure returns `ErrorCode::InvalidInput` with `InvalidValue.field() == "color filter operation count"`; byte computation or either device-limit failure returns the same code with field `"color filter operation buffer byte length"`. Both occur during immutable preparation-plan derivation before `ResourceManager::begin_frame`, WGPU buffer creation, provisional cache creation, submission, or frame publication. Test-only limit inputs exercise count overflow and each device limit without allocating an oversized vector. No new public error variant is added.
- Color-only execution preserves source bounds, signed origin, extent, raster scale, and texel centers. It renders to a distinct working texture, never reads and writes one texture, and participates in the existing exact dependency and last-use release model.
- C10 reuses C09's caller-owned encoder, checked GPU scope, resource frame, provisional pass cache, submission, and headless/presented host effect. Only `GpuOperationTransaction` submits. Production color execution contains no map, poll, CPU pixel execution, re-upload, Vello atlas reentry, inter-pass wait, or CPU fallback.
- The private CPU oracle is corrected to the literal S21 constants and remains `cfg(test)` only. Literal known-vector tests are independent from both oracle and WGSL constants. GPU comparisons first require exact dimensions/origin and premultiplied invariants, then use S34: at most 2 straight RGBA8 levels for high precision, and at most 2 alpha/`premul8` levels for reduced precision.
- No public scene ingress at the C10 base can produce a color-only executable graph: broad layer filters and `FilteredImagePaint` are diagnostics, while the only public scene `FilterList` ingress is bounded backdrop and remains a C12 diagnostic. C10 therefore keeps `supports_gpu_color_filter_execution()` false and preserves `GpuColorFilterExecution` as unsupported. A `#[cfg(test)]` fixture built from an authored `FilterList` plus ordinary capture/composition inputs is the sole C10 ingress; it enters the same non-test graph lowering, preparation, shader, encoding, transaction, headless/presented publication, and failure paths without bypassing validation. C12 supplies the first public bounded-filter ingress, and C13 owns the final capability/inventory flip.
- Blur, drop shadow, bounded backdrop completion, broad layer/reference filters, `FilteredImagePaint` execution, and every broad mask/backdrop surface remain their exact typed diagnostics. The separate legacy authored/model query `supports_color_filtered_image_paint()` and its C13 inventory reconciliation are unchanged; C10 adds no filtered-image render route.

## Impacts
| Area | C10 record |
| --- | --- |
| Public API | No type, signature, capability value, or reachable public render route changes. The existing granular GPU color capability remains false until C12/C13 supplies and reconciles a public ingress; C10's fixture is crate-private and test-only. |
| Dependencies/features | Unchanged reviewed dependency and feature set; no acquisition, lockfile, build script, fixture, or generated artifact delta. |
| Modules | `filter.rs` owns finite scalar lowering; `frame.rs`/`pass.rs` own closed graph/runtime phases; `shader.rs` owns operation bytes and checked program objects; backend/transaction/renderer own atomic execution; `reference.rs` remains test-only. |
| Artifacts | Add only `src/shaders/color_filter.wgsl`; root owns later facade and API-artifact adaptation. No sibling or root edit. |
| Docs/examples | Unchanged; C10 changes no public behavior claim, and final architecture/platform documentation belongs to C14. |
| MSRV/platform | Preserve Rust 1.97.x, Rust 2024, all four native feature states, wasm `render-web` compilation, native headless execution, and `render-window` presentation. |
| Unsafe | `#![forbid(unsafe_code)]` remains effective; no owned unsafe, unsafe attribute/extern, lint allowance, pointer cast, or owned POD implementation. |

## Tasks
Define these functions in each task shell before focused commands. They select an installed stable Rust 1.97.x release, require the already-authorized wasm target, reject ambiguous tests, and fail on owned unsafe.
```sh
prepare_c10_toolchain() {
  local candidate host installed_toolchains patch verbose_version version_number
  installed_toolchains="$(rustup toolchain list)" || return $?
  C10_TOOLCHAIN=""
  for candidate in $(printf '%s\n' "$installed_toolchains" | awk '{ print $1 }'); do
    verbose_version="$(rustup run "$candidate" rustc -vV)" || continue
    version_number="$(printf '%s\n' "$verbose_version" | awk '/^rustc / { print $2 }')" || continue
    host="$(printf '%s\n' "$verbose_version" | awk '/^host: / { print $2 }')" || continue
    test -n "$host" || continue
    case "$version_number" in
      1.97.*)
        patch="${version_number#1.97.}"
        case "$patch" in
          ''|*[!0-9]*) continue ;;
          *)
            case "$candidate" in
              stable|"stable-$host"|"1.97.$patch"|"1.97.$patch-$host") C10_TOOLCHAIN="$candidate"; break ;;
            esac
            ;;
        esac
        ;;
    esac
  done
  test -n "$C10_TOOLCHAIN" || { printf 'required installed stable Rust 1.97.x release toolchain is unavailable; acquisition is not authorized\n' >&2; return 1; }
  export C10_TOOLCHAIN
}

require_c10_wasm_target() {
  local sysroot
  prepare_c10_toolchain || return $?
  sysroot="$(rustup run "$C10_TOOLCHAIN" rustc --print sysroot)" || return $?
  test -d "$sysroot/lib/rustlib/wasm32-unknown-unknown/lib" || { printf 'wasm32-unknown-unknown is not installed for %s; acquisition is not authorized\n' "$C10_TOOLCHAIN" >&2; return 1; }
}

cargo_c10() {
  prepare_c10_toolchain || return $?
  CARGO_NET_OFFLINE=true rustup run "$C10_TOOLCHAIN" cargo "$@"
}

cargo_c10_wasm() {
  require_c10_wasm_target || return $?
  CARGO_NET_OFFLINE=true rustup run "$C10_TOOLCHAIN" cargo "$@"
}

run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(cargo_c10 test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(cargo_c10 test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    cargo_c10 test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    cargo_c10 test -p surgeist-render "tests::$name" -- --exact
  fi
}

assert_no_owned_unsafe() {
  local file manifest output scan_status
  manifest="$(mktemp "${TMPDIR:-/tmp}/surgeist-c10-owned-rust.XXXXXX")" || return $?
  git ls-files -z --cached --others --exclude-standard -- '*.rs' >"$manifest" || { scan_status=$?; unlink "$manifest"; return "$scan_status"; }
  test -s "$manifest" || { unlink "$manifest"; printf 'owned Rust manifest is empty\n' >&2; return 1; }
  while IFS= read -r -d '' file; do
    if output="$(rg -n --pcre2 '#\s*!?\[\s*(?:allow|expect)\s*\([^)]*\bunsafe(?:_[A-Za-z0-9_]+)?\b|#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' -- "$file" 2>&1)"; then
      unlink "$manifest"; printf '%s\n' "$output" >&2; return 1
    else
      scan_status=$?
      test "$scan_status" -eq 1 || { unlink "$manifest"; printf '%s\n' "$output" >&2; return "$scan_status"; }
    fi
  done <"$manifest"
  unlink "$manifest"
}

assert_no_match() {
  local output scan_status
  if output="$(rg "$@" 2>&1)"; then printf '%s\n' "$output" >&2; return 1; fi
  scan_status=$?
  test "$scan_status" -eq 1 && return 0
  printf '%s\n' "$output" >&2
  return "$scan_status"
}
```
No helper installs, updates, or acquires software. Every changed behavior first receives its named test and the narrowest `#[cfg(test)]` observation needed, demonstrates the stated assertion RED, then is implemented. Compile failure, setup error, unavailable adapter, panic-before-assertion, or unrelated failure is not RED. Every worker runs its focused commands and `C10-CHECK`; the coordinator records its exact commit span and obtains a fresh task review before advancing.

**C10-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
prepare_c10_toolchain
cargo_c10 fmt --check
cargo_c10 check -p surgeist-render
cargo_c10 test -p surgeist-render
cargo_c10 clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo_c10 test -p surgeist-render --features render-window
cargo_c10 clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
cargo_c10 test -p surgeist-render --features render-web
cargo_c10 clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
cargo_c10 test -p surgeist-render --features render-window,render-web
cargo_c10 clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
cargo_c10 check -p surgeist-render --all-targets
cargo_c10 check -p surgeist-render --all-targets --features render-window,render-web
native_getrandom_tree="$(cargo_c10 tree -p surgeist-render -e features -i getrandom@0.3.4)"
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
test -z "$(git ls-files -- Cargo.lock)"
assert_no_owned_unsafe
cargo_c10_wasm check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(cargo_c10_wasm tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)"
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
```

**T1. Lower exact color semantics to finite runtime values**
- Area/outcome: `src/{filter.rs,frame.rs,pass.rs,reference.rs,tests.rs}`; correct the test oracle to S21 literals and add one explicit algorithm-to-runtime conversion for all eight operations, finite normalized amounts, reduced angles, and per-operation clamp identity.
- RED/acceptance: `color_filter_known_vectors_use_spec_constants_not_oracle_constants` fails only at `the grayscale primary vector differs from the literal S21 result`; `filter_scalar_lowering_handles_f32_f64_exponents_and_huge_angles_finitely` fails only at `runtime color scalars are not finite normalized S21 values`; the existing `color_filter_fusion_preserves_each_source_clamp` remains green. Pass covers zero, positive subnormal, `f32::MAX`, `f64::MAX`, mantissa-rounding renormalization, huge positive/negative angles, near-gray saturation, exact operation order, and a clamp after every operation without exposing a production test API.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order`; then `C10-CHECK`.
- Depends: none. Commit: `Lower finite color filter semantics`.

**T2. Serialize and realize the checked color-filter program**
- Area/outcome: `src/{filter.rs,pass.rs,shader.rs,tests.rs}` plus `src/shaders/color_filter.wgsl`; add the sole operation-buffer byte owner, exact bind-group layout, checked shader module, and high/reduced render-pipeline realization without making the pass production-dispatchable yet.
- RED/acceptance: `color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries` fails only at `color operation bytes lost an authored finite scalar or clamp`; `color_filter_operation_buffer_limits_return_exact_invalid_input_before_allocation` fails only at `an oversized C10 buffer lacks its exact pre-allocation diagnostic`; `c10_color_filter_cache_realizes_checked_high_and_reduced_programs` fails only at `the C10 checked shader program is unrealized`; `c10_color_filter_layout_binds_exact_source_spatial_and_operations` fails only at `the C10 layout has a missing or dummy binding`. Pass uses the exact 16-plus-32N layout, checks `u32` count plus `max_buffer_size` and `max_storage_buffer_binding_size` separately through test-only limit inputs, returns the two exact `InvalidValue` fields, and creates no frame scope/buffer/cache entry on rejection. The accepted layout uses one nearest `FilterSource` texture/sampler, one spatial uniform, one read-only operation storage buffer, one working-format target, no runtime shader source, and no unrelated shader/cache key.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test c09_composite_layouts_bind_no_dummy_parent_clip_or_mask`; `run_exact_test c08_shader_cache_realizes_checked_programs_without_publishing_failed_entries`; then `C10-CHECK`.
- Depends: T1. Commit: `Add the checked color filter shader`.

**T3. Close the C10 executable graph subset**
- Area/outcome: `src/{frame.rs,pass.rs,renderer.rs,tests.rs}`; add the exact `#[cfg(test)]` authored-filter graph fixture, extend closed graph facts and preparation with ordered color-filter source/result transitions, and keep public production dispatch at the C09 diagnostic boundary for the whole cycle.
- RED/acceptance: `c10_executor_accepts_only_spine_composition_and_ordered_color_filters` fails only at `C10 has no closed pre-allocation subset`; `color_filter_graph_preserves_authored_order_clamps_and_exact_lifetimes` fails only at `the C10 graph changed operation order clamp or last use`; `mixed_color_and_future_passes_preserve_the_next_unavailable_diagnostic` fails only at `a C11 plus pass was admitted or color masked its diagnostic`. Pass accepts any finite sequence of valid color runs composed with C08/C09 nodes, updates the current resource after each run, validates one distinct source/result with exact same spatial facts, and rejects malformed/empty payloads and every C11+ pass before preparation. The fixture starts from an authored `FilterList` and ordinary captured commands, invokes normal planning/lowering, and exposes no production API or alternate executor.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test renderer_dispatch_routes_c08_and_c09_to_gpu_but_future_passes_to_typed_diagnostics`; `run_exact_test c10_plus_graph_diagnostic_precedes_unavailable_effect_working_format`; then `C10-CHECK`.
- Depends: T2. Commit: `Close the C10 color graph subset`.

**T4. Encode ordered color passes in one graph transaction**
- Area/outcome: `src/{pass.rs,shader.rs,backend.rs,gpu_transaction.rs,tests.rs}`; prepare exact operation buffers and encode every validated color run into a distinct working texture in the existing caller-owned encoder and checked scope, still through a private staged entry.
- RED/acceptance: `c10_graph_encodes_fused_color_filters_in_authored_order` fails only at `the C10 scheduler has no ordered GPU color pass`; `color_filter_pass_uses_distinct_source_and_result_without_readback` fails only at `the C10 pass aliases source or reaches CPU visibility`; `multiple_color_runs_share_one_graph_encoder_and_transaction_commit` fails only at `C10 split the frame transaction`; `oversized_color_filter_buffer_preserves_resources_cache_and_publication` fails only at `C10 limit rejection changed GPU or published state`. Pass binds exact source/spatial/operation resources, uses the signed bounded viewport/scissor, advances each pass once, releases resources at validated last use, keeps all runs in one submission, and aborts resources/cache entries on encode or scope failure without map/poll/wait/submit in pass code. Each count/limit failure returns its T2 diagnostic before resource acquisition, cache publication, submission, or frame publication.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test multiple_composites_share_one_graph_encoder_and_transaction_commit`; `run_exact_test graph_render_path_submits_without_map_or_cpu_wait`; then `C10-CHECK`.
- Depends: T3. Commit: `Encode ordered GPU color filters`.

**T5. Prove color filters through the shared production executor**
- Area/outcome: `src/{renderer.rs,backend.rs,pass.rs,reference.rs,tests.rs}`; route only the T3 test fixture into the same non-test exact-graph backend and prove all operations, order, clamps, and precision through real headless GPU output while public dispatch and capabilities remain unchanged.
- RED/acceptance: `high_precision_color_functions_match_cpu_oracle_for_boundary_pixels` fails only at `high precision C10 pixels exceed the S34 color tolerance`; `reduced_precision_color_functions_match_cpu_oracle_with_declared_tolerance` fails only at `reduced C10 alpha or premul8 exceeds the S34 tolerance`; `filter_function_order_changes_output_and_matches_ordered_oracle` fails only at `the GPU lost authored order or a source clamp`; `color_filter_shader_failure_preserves_prior_publication_and_cache` fails only at `failed C10 execution published draft state`. Pass covers transparent/partial/opaque boundary pixels and every operation, exact dimensions/origin and premultiplied invariants, real high and reduced paths, noncommuting and clamp-sensitive chains, one transaction submission, and atomic failure with no CPU retry.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test capture_canonicalize_present_round_trips_transparent_partial_and_opaque_pixels`; `run_exact_test headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`; then `C10-CHECK`.
- Depends: T4. Commit: `Prove GPU color filter execution`.

**T6. Close private ingress integration and retained public diagnostics**
- Area/outcome: integrated `src/{capability.rs,error.rs,frame.rs,pass.rs,shader.rs,backend.rs,gpu_transaction.rs,renderer.rs,tests.rs}`; prove the test-only ingress, presented path, reuse/zero-budget behavior, unchanged public capability truth, exact C10/C11+ diagnostics, and production-path guards without enabling any public filter surface.
- RED/characterization/acceptance: before T6 changes, `repeated_color_filter_frames_reuse_passes_without_growth_or_readback` and `budget_zero_releases_color_filter_frame_resources_without_changing_pixels` pass as behavior-preserving characterization evidence established by T1-T5; `c10_fixture_executes_while_public_color_capability_remains_diagnostic`, `render_window_smoke_executes_ordered_color_filter_fixture_through_production_graph`, and `public_dispatch_retains_c09_boundary_while_c10_fixture_uses_shared_executor` fail only at their named fixture, presented, or public-boundary assertion. After implementation all five pass. Pass proves stable pass-cache/resource counts after warm-up, zero idle retention at budget zero, exact pixels and one submission, the fixture alone reaches the shared executor, `supports_gpu_color_filter_execution()` remains false with exact `GpuColorFilterExecution`, blur/drop-shadow/backdrop/layer/reference/`FilteredImagePaint` execution remains false before allocation, the legacy color-filtered-image model query is unchanged, and no CPU materialization route exists.
- Commands: run all five names separately with `run_exact_test`, using `render-window` for the presented fixture test; `run_exact_test repeated_frames_reuse_resources_without_growth_or_readback`; `run_exact_test budget_zero_releases_idle_resources_without_changing_pixels`; `run_exact_test device_loss_is_terminal_idempotent_and_releases_device_resources`; run the completion guards; then `C10-CHECK`.
- Depends: T5. Commit: `Complete ordered color filter execution`.

## Completion
- Require all six ordered task ranges and clean fresh task reviews. The coordinator then changes only this plan status from `in_progress` to `complete` and commits that status separately. Run `C10-CHECK` and every T1-T6 named test with its stated feature set. Require exact extent/origin, S34 high/reduced evidence, one transaction submission, atomic headless/presented delivery, stable reuse, zero-budget cleanup, exact C11+ diagnostics, and a clean worktree.
- Run these fail-closed guards: `test "$(rg -n '^    pub const fn supports_gpu_color_filter_execution\(' src/capability.rs | wc -l | tr -d ' ')" -eq 1`; require `run_exact_test c10_fixture_executes_while_public_color_capability_remains_diagnostic`; `assert_no_match -n 'ResolvedImageColorFilterExecution|CompiledColorFilterPipeline|apply_color_filter_pipeline|apply_compiled_color_filter_pipeline' src/renderer.rs src/backend.rs src/pass.rs src/shader.rs src/gpu_transaction.rs`; `assert_no_match -n 'queue\.submit|map_async|Device::poll' src/pass.rs src/shader.rs src/vello_engine --glob '*.rs'`; `assert_no_match -n 'register_texture|override_image' src/pass.rs src/shader.rs src/gpu_transaction.rs src/vello_engine/encoder.rs`; `assert_no_match -n 'use .*reference|reference::' src/renderer.rs src/backend.rs src/pass.rs src/shader.rs src/filter.rs`; `assert_no_match -n 'pub use .*pass|pub use .*shader|PreparedGraph|WorkingFormat|GpuRenderGraph|RuntimeColor' src/lib.rs`; and require `git diff --unified=0 1c4ae3b2547fbe3c42d5519caee902303269ccd1..HEAD -- '*.rs' | rg '^\+(?:\s*(?:allow|expect)\s*\(|.*#\s*!?\[[^]]*(?:allow|expect)\s*\()'` to return no match. Require `git ls-files -- 'src/shaders/**' | LC_ALL=C sort` to equal the four C09 shader paths plus `src/shaders/color_filter.wgsl`, each with exactly one `include_str!` owner; `Cargo.toml` must equal the cycle base and no lockfile may be tracked.
- After final checks, run a fresh `surgeist-holistic-reviewer` against this complete-status plan, exact specification/sequence revisions, crate boundary, full `1c4ae3b2547fbe3c42d5519caee902303269ccd1..HEAD` diff, tests, and Rust modeling guidance. Only CLEAN permits a second complete final-check run, canonical landing, immutable publication to authority `origin/main`, fresh fetch/readback, and handoff. C11 receives one executable ordered color path in the shared graph; root receives no C10 edit.
- Block only for an unowned conflicting change, reviewed-source contradiction, unavailable required native GPU/presented execution, missing installed Rust 1.97.x or authorized wasm target, or unavailable required Surgeist custom profile. No acquisition, CPU production renderer/fallback, skipped adapter, compatibility shim, hidden quality reduction, unsafe, unchecked shader/buffer path, public capability overclaim, or partial C11+ execution is allowed.
