# GPU Render Pipeline C09 Composition Clip Mask And Blend Passes

## Header
- Cycle: `C09`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `44fd908f60a4b0d1b073c7f9a11ebab8c1472ee6` (C08).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: S09, C09-applicable S11, S16, S18-S19, S23, S25, S27-S30, C09-applicable S31-S34, and per-cycle S36-S37 evidence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `562478db06184de64d6d5fad7ed134d99e2ab0f9`, normalized SHA-256 `9fb83aeebf2bcd2a581241e97c7cdde58942c8001f54c195b5a54a090908f1ba`, entry `C09 Composition Clip Mask And Blend Passes`.
- Outcome: extend the published source-to-output GPU graph with Vello-generated outer-clip coverage and an exact WGPU composition pass for resolved alpha masks, opacity, isolation, normal source-over, and every currently supported blend mode.

## Boundary
- The executable C09 vocabulary is the C08 spine plus `VelloCapture` carrying a closed clip-coverage payload and `Composite(Some(Layer { .. }))`. It still rejects `CopyBackdrop`, `ColorFilter`, both blur passes, `DropShadowColorize`, `Composite(DropShadow)`, missing payloads, malformed order/bindings, and unsupported output before allocation or encoding. The existing transitional route remains only through T6; T7 replaces every C10+ route with the exact typed no-publication diagnostic below. C09 neither executes nor claims the filter portion of `outer_clip_precedes_mask_and_opacity_but_follows_filter`.
- `ResolvedLayerAlphaMask` becomes exactly `Image` plus finite positive-area layer-local `Rect`, with private fields, `try_new`, `image`, and `bounds`; `Layer::with_resolved_alpha_mask` is infallible. The old ImageBuffer/mode constructors and accessors, `Layer::try_resolved_alpha_mask`, public/production `ResolvedAlphaMaskExecution`, and `Eq` implementation are removed without aliases or shims. Through T6 only, one crate-private full-S09 staging adapter preserves the new bounds, transform, quality, and extend semantics by calling the same independent CPU oracle later gated to tests; T7 removes its production call sites and compilation.
- Mask storage and mask meaning are distinct typed facts. The retained upload allocates the exact image pixel extent and is keyed by `ImageId`, dimensions, quality, and extend. Composition separately owns local semantic bounds, validated destination-to-layer-local mapping, texel-center mapping, and image dimensions. A zero-sized image is a transparent/annihilating mask with no texture allocation; no 1x1 substitute, bounds-derived upload extent, pixel scan, or guessed mapping is allowed.
- Every graph layer captures content before its own outer semantics. Its existing clip plus inherited ordered clips are rasterized together by the internal Vello engine into one bounded antialiased RGBA8 coverage image using the authored geometry, fill rule, transforms, requested antialiasing, transparent base, and opaque coverage paint. The composite samples coverage alpha; direct-only clips remain on Vello's native path.
- Composition applies source mapping, then clip coverage, resolved mask, clamped layer opacity, and blend into the parent. Mask points outside semantic bounds are transparent before extend is considered. Inside bounds, Low is nearest, Medium is per-tap bilinear, and High is 4x4 Mitchell-Netravali with `B=C=1/3`; Pad, Repeat, and Reflect apply to every out-of-domain tap, and High alpha clamps to `[0,1]`. RGB mask channels are ignored.
- Normal source-over uses fixed premultiplied `One`/`OneMinusSrcAlpha` blending and never samples its parent. Multiply, Screen, Overlay, Darken, Lighten, and Plus copy the parent to a distinct result, sample source and old parent, evaluate the CSS premultiplied numeric-sRGB formula, clamp, preserve pixels outside the bounded region, and swap parent identity. Plus takes this destination-sampling path so channel and alpha sums clamp to one. No pass samples the texture it writes; isolated groups begin at transparent black.
- Add only `src/shaders/layer_composite.wgsl`. Its entry points expose exact normal versus destination-sampling and optional clip/mask binding sets, so no dummy texture, sampler, or parameter binding is needed. Blend choice and mask quality/extend are validated typed parameters; upload identity never enters a shader/pipeline key. Uniform bytes use explicit checked little-endian serialization with WGSL alignment and no pointer cast or owned POD implementation.
- C09 reuses one caller-owned graph encoder, checked GPU scope, aggregate Vello/effect leases, provisional pass cache, submission, and headless/presented host effect from C08. Only `GpuOperationTransaction` submits. Failure, cancellation, surface transition, or device terminality publishes no draft/cache/lease state and preserves the prior successful surface state. Production composition contains no map, poll, CPU pixel execution, re-upload, atlas re-entry, or inter-pass wait.
- High and reduced working formats use the same graph, shader semantics, and ordering. GPU comparisons first require exact extent/origin and premultiplied invariants, then apply S34: normal composition at most 2 levels per channel and supported blends at most 3, with reduced comparisons using alpha and `premul8`. After T7, all CPU pixel/filter implementations and `reference` compile only under `cfg(test)`.
- C09 performs only the S11/S29 capability changes whose behavior becomes real here: resolved alpha-mask execution, image-pass execution, composite-pass execution, and nested opacity composition use their final semantic names and exact operations. Broad layer masks, mask execution/composite modes, layer filters, offscreen layers, broad backdrops, and backdrop isolation remain false diagnostics. Final remaining inventory totals/routes/statistics and capability reconciliation remain C13 work.

The staged capability and dispatch transition is exact:

| Stage | Public capability state | Render dispatch state |
| --- | --- | --- |
| Base through T6 | Preserve `supports_materialized_alpha_mask_execution() == true` and `PrimitiveOperation::MaterializedAlphaMaskExecution` only for the temporary full-semantics staging adapter. Preserve `supports_rect_fullscreen_shader_passes() == false` and `supports_nested_opacity_planning() == false`; no new GPU capability is claimed. | C08 executes on GPU; masks and bounded backdrop/filter graphs may use the existing private transition. No intermediate commit is publishable. |
| T7 and T8 | Remove every old/materialized/CPU accessor and operation. Add `supports_resolved_alpha_mask_execution() == true` / `ResolvedAlphaMaskExecution`, `supports_image_pass_execution() == true` / `ImagePassExecution`, `supports_composite_pass_execution() == true` / `CompositePassExecution`, and `supports_nested_opacity_composition() == true` / `NestedOpacityComposition`. Planning/resource facts `supports_ordered_filter_lists()`, `supports_filter_region_planning()`, `supports_persistent_effect_resources()`, `supports_bounded_vello_capture()`, and `supports_bounded_backdrop_capture()` remain true. | C08/C09 execute on GPU. `ColorFilter`, blur, drop-shadow, and bounded-backdrop graph requirements return the exact false operation below before allocation, configuration/acquisition, encoding, submission, or publication. |

At the T7 boundary the unavailable execution queries are exactly `supports_gpu_color_filter_execution()`, `supports_gpu_blur_filter_execution()`, `supports_gpu_drop_shadow_filter_execution()`, and `supports_bounded_backdrop_filter_execution()`, all `false`; `supports_layer_filter_execution()` and `supports_broad_backdrop_execution()` also remain `false`. Dispatch maps `ColorFilter` to `(Filters, GpuColorFilterExecution)`, either blur pass to `(Filters, GpuBlurFilterExecution)`, `DropShadowColorize` or `Composite(DropShadow)` to `(Filters, GpuDropShadowFilterExecution)`, and `CopyBackdrop`/bounded-backdrop completion to `(OffscreenPipeline, BoundedBackdropFilterExecution)`, each as `UnsupportedPrimitive`. Malformed graph states remain typed `RenderFailed`; no unavailable or malformed case changes publication/stats.

## Impacts
| Area | C09 record |
| --- | --- |
| Public API | Intentional breaking S09 mask model/removals plus the C09-owned S29 semantic capability/operation renames; no deprecated aliases or compatibility feature. |
| Dependencies/features | Unchanged reviewed dependency and feature set; no acquisition, lockfile, build script, fixture, or generated artifact delta. |
| Modules | `layer.rs`/`image.rs`/`command.rs` own mask phases; `frame.rs`/`pass.rs` own closed composition planning and lowering; `encode.rs`/internal Vello own clip coverage; `shader.rs` owns exact composite programs; `resource.rs` owns retained uploads/leases; backend/transaction/renderer own atomic execution. |
| Artifacts | Add only `src/shaders/layer_composite.wgsl`; root owns later facade and API-artifact adaptation. No sibling or root edit. |
| Docs/examples | Update rustdoc on every changed public mask/capability surface. `README.md` is unchanged because it documents neither surface; no example target exists. |
| MSRV/platform | Preserve Rust 1.97, Rust 2024, all four native feature states, wasm `render-web` compilation, native headless execution, and `render-window` presentation. |
| Unsafe | `#![forbid(unsafe_code)]` remains effective; no owned unsafe, unsafe attribute/extern, lint allowance, unchecked shader path, or raw backend escape. |

## Tasks
Define these functions in each task shell before focused commands. They select an exact installed Rust 1.97 release toolchain, require the already-authorized wasm target, reject ambiguous tests, and fail on owned unsafe.
```sh
prepare_c09_toolchain() {
  local candidate host installed_toolchains patch verbose_version version_number
  installed_toolchains="$(rustup toolchain list)" || return $?
  C09_TOOLCHAIN=""
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
              stable|"stable-$host"|"1.97.$patch"|"1.97.$patch-$host") C09_TOOLCHAIN="$candidate"; break ;;
            esac
            ;;
        esac
        ;;
    esac
  done
  test -n "$C09_TOOLCHAIN" || { printf 'required installed stable Rust 1.97.x release toolchain is unavailable; acquisition is not authorized\n' >&2; return 1; }
  export C09_TOOLCHAIN
}

require_c09_wasm_target() {
  local sysroot
  prepare_c09_toolchain || return $?
  sysroot="$(rustup run "$C09_TOOLCHAIN" rustc --print sysroot)" || return $?
  test -d "$sysroot/lib/rustlib/wasm32-unknown-unknown/lib" || { printf 'wasm32-unknown-unknown is not installed for %s; acquisition is not authorized\n' "$C09_TOOLCHAIN" >&2; return 1; }
}

cargo_c09() {
  prepare_c09_toolchain || return $?
  CARGO_NET_OFFLINE=true rustup run "$C09_TOOLCHAIN" cargo "$@"
}

cargo_c09_wasm() {
  require_c09_wasm_target || return $?
  CARGO_NET_OFFLINE=true rustup run "$C09_TOOLCHAIN" cargo "$@"
}

run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(cargo_c09 test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(cargo_c09 test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    cargo_c09 test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    cargo_c09 test -p surgeist-render "tests::$name" -- --exact
  fi
}

assert_no_owned_unsafe() {
  local file manifest output scan_status
  manifest="$(mktemp "${TMPDIR:-/tmp}/surgeist-c09-owned-rust.XXXXXX")" || return $?
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
No helper installs, updates, or acquires software. Every changed behavior first receives its named test and the narrowest `#[cfg(test)]` observation needed, demonstrates the stated assertion RED, then is implemented. Adapter absence, setup error, panic-before-assertion, or unrelated failure is not RED. T8 alone first adds its explicitly named capability, reuse, and zero-budget behavior-preserving proof tests; they must pass before and after its separate dispatch RED and implementation. Every worker runs its focused commands and `C09-CHECK`; the coordinator records its exact commit span and obtains a fresh task review before advancing.

**C09-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
prepare_c09_toolchain
cargo_c09 fmt --check
cargo_c09 check -p surgeist-render
cargo_c09 test -p surgeist-render
cargo_c09 clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo_c09 test -p surgeist-render --features render-window
cargo_c09 clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
cargo_c09 test -p surgeist-render --features render-web
cargo_c09 clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
cargo_c09 test -p surgeist-render --features render-window,render-web
cargo_c09 clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
cargo_c09 check -p surgeist-render --all-targets
cargo_c09 check -p surgeist-render --all-targets --features render-window,render-web
native_getrandom_tree="$(cargo_c09 tree -p surgeist-render -e features -i getrandom@0.3.4)"
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
test -z "$(git ls-files -- Cargo.lock)"
assert_no_owned_unsafe
cargo_c09_wasm check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(cargo_c09_wasm tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)"
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
```

**T1. Install the resolved image-mask contract**
- Area/outcome: `src/{layer.rs,image.rs,command.rs,renderer.rs,reference.rs,lib.rs,tests.rs}`; replace the public and normalized mask phases with validated `Image` plus local `Rect`, preserve upload identity/sampling facts, remove the old public execution type, and adapt the temporary private bridge to the complete new semantics without changing its old capability name before GPU cutover.
- RED/acceptance: `resolved_alpha_mask_requires_finite_positive_local_bounds` fails only at `resolved masks accept invalid local bounds`; `resolved_alpha_mask_public_model_uses_image_bounds_and_infallible_layer_installation` fails only at `resolved mask public phases still expose buffer or mode semantics`; `resolved_mask_normalization_preserves_image_identity_sampling_and_local_bounds` fails only at `mask normalization collapsed storage and semantic bounds`; `transitional_resolved_mask_bridge_preserves_bounds_quality_extend_and_transform` fails only at `the staged bridge changed new mask semantics`. Pass rejects every non-finite/zero/negative-area bound, preserves zero-sized images as valid transparent inputs, removes `Eq` and old constructor/accessor/reexport aliases, retains broad/luminance/multi-layer diagnostics, and leaves the old materialized capability/operation unchanged solely for the bounded T1-T6 bridge.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test affected_capability_queries_map_one_to_one_to_primitive_operations`; then `C09-CHECK`.
- Depends: none. Commit: `Adopt resolved image mask semantics`.

**T2. Close the C09 composition graph model**
- Area/outcome: `src/{frame.rs,pass.rs,renderer.rs,tests.rs}`; generalize cycle-numbered spine eligibility into one stable closed executable-graph classification, model layer composition as exact source/parent/optional-coverage/optional-mask reads, and preserve the explicit later-cycle transitional class.
- RED/acceptance: `c09_executor_accepts_only_spine_and_ordered_layer_composition` fails only at `C09 has no closed pre-allocation subset`; `composition_graph_orders_clip_mask_opacity_blend_and_nested_layers` fails only at `composition graph changed authored outer-operation order`; `composition_isolation_starts_from_transparent_black` fails only at `isolated composition inherited root base color`. Pass accepts both C08 spine and well-formed C09 layer composites for RGBA/BGRA and both working formats, rejects every C10+ pass/payload or malformed alias/order before preparation, retains one root base clear, and models nested composites inner to outer without applying outer semantics in their capture.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test graph_base_color_is_initialized_once_and_isolation_is_transparent`; `run_exact_test renderer_dispatch_routes_only_closed_c08_graph_subset_to_gpu_executor`; then `C09-CHECK`.
- Depends: T1. Commit: `Close the C09 composition graph subset`.

**T3. Generate ordered Vello clip coverage**
- Area/outcome: `src/{frame.rs,pass.rs,encode.rs,tests.rs}` and `src/vello_engine/{encoder.rs,raster.rs,resources.rs,mod.rs}`; add one closed clip-coverage form under `VelloCapture`, combine the owning and inherited `RenderClip` stack, and bind its RGBA8 alpha as a composition read.
- RED/acceptance: `graph_clip_coverage_is_one_vello_capture_of_ordered_render_clips` fails only at `graph clips have no bounded Vello coverage capture`; `clip_coverage_preserves_fill_rule_antialiasing_and_signed_mapping` fails only at `clip coverage differs from authored Vello geometry or grid`; `clip_coverage_is_bound_before_mask_and_opacity` fails only at `clip coverage lost its ordered composite role`. Pass preserves rect/rounded/circle/ellipse/path geometry, nonzero/even-odd fill, coordinate-space transform, requested AA, signed origin, texel centers, and transparent outside coverage; clip captures share the frame's aggregate Vello leases and add no public pass kind or CPU rasterizer.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test multiple_vello_captures_share_one_graph_encoder_and_transaction_commit`; `run_exact_test vello_capture_uses_transparent_base_requested_aa_and_exact_bounded_extent`; then `C09-CHECK`.
- Depends: T2. Commit: `Rasterize graph clip coverage with Vello`.

**T4. Prepare mask uploads and exact composition parameters**
- Area/outcome: `src/{frame.rs,pass.rs,shader.rs,resource.rs,tests.rs}`; allocate retained mask images by pixel extent, carry semantic bounds/mappings separately, serialize all finite composition parameters safely, and specialize cache keys only by actual program/layout/sampling behavior.
- RED/acceptance: `mask_upload_allocation_uses_image_extent_not_local_bounds` fails only at `mask allocation still aliases semantic bounds`; `composite_parameter_bytes_preserve_affine_mask_mapping_quality_and_extend` fails only at `composite bytes lost typed mask mapping or sampling`; `zero_sized_mask_image_annihilates_without_texture_allocation` fails only at `zero mask allocated a substitute texture`; `mask_pipeline_keys_exclude_image_identity` fails only at `pipeline caching is keyed by retained image identity`. Pass validates destination-to-layer-local inversion before allocation, preserves arbitrary finite transforms and texel centers, emits exact WGSL-aligned little-endian bytes, requests one retained upload per exact key, and cleans provisional uploads on failure/cancellation.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod`; `run_exact_test resource_leases_reject_stale_generation_and_double_release_by_model`; then `C09-CHECK`.
- Depends: T3. Commit: `Prepare typed composition resources`.

**T5. Realize the exact GPU compositor**
- Area/outcome: `src/{shader.rs,pass.rs,backend.rs,tests.rs}` plus `src/shaders/layer_composite.wgsl`; realize checked entry-point-specific layouts/pipelines for normal and destination-sampling composition with only present clip/mask bindings, manual mask taps, and every supported blend formula.
- RED/acceptance: `c09_composite_cache_realizes_exact_normal_and_destination_sampling_programs` fails only at `C09 compositor has no checked pipeline realization`; `c09_composite_layouts_bind_no_dummy_parent_clip_or_mask` fails only at `composite layout contains an absent semantic binding`; `c09_shader_mask_sampling_matches_independent_boundary_vectors` fails only at `GPU mask sampling differs from independent constants`; `c09_shader_blend_functions_match_independent_known_vectors` fails only at `GPU blend math differs from independent constants`. Pass uses fixed normal blend without a parent sample; destination paths use replace blending, safe zero-alpha handling, CSS formulas and Plus clamp; Low/Medium/High plus Pad/Repeat/Reflect match S09 and no runtime shader source or unchecked module enters the cache.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test c08_shader_cache_realizes_checked_programs_without_publishing_failed_entries`; `run_exact_test device_pass_cache_owns_exact_sampler_layout_shader_and_pipeline_key_spaces`; then `C09-CHECK`.
- Depends: T4. Commit: `Add the checked GPU compositor`.

**T6. Encode composition in one graph transaction**
- Area/outcome: `src/{pass.rs,shader.rs,backend.rs,gpu_transaction.rs,tests.rs}`; extend the one-shot scheduler to encode clip coverage, parent preservation, bounded normal or destination-sampling composites, and exact last-use releases in the existing caller-owned encoder/scope.
- RED/acceptance: `c09_graph_encodes_clip_mask_opacity_and_blend_in_authored_order` fails only at `C09 composition has no one-shot GPU encoding`; `normal_composition_uses_fixed_premultiplied_blend_without_parent_sampling` fails only at `normal composition sampled its parent or used wrong factors`; `non_normal_blends_copy_parent_and_never_read_write_one_texture` fails only at `destination sampling aliases its output`; `multiple_composites_share_one_graph_encoder_and_transaction_commit` fails only at `composition split the frame transaction`. Pass copies/preserves the full parent where required, bounds viewport/scissor without origin loss, binds exact resources/parameters, advances each pass once, releases at validated last use, and aborts all captures/uploads/effect leases/cache entries on any encode or scope failure without submit/map/poll/wait.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test custom_spine_encodes_clear_canonicalize_copy_source_over_and_present_in_order`; `run_exact_test encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`; then `C09-CHECK`.
- Depends: T5. Commit: `Encode ordered GPU composition`.

**T7. Cut resolved masks and blends to production GPU execution**
- Area/outcome: `src/{capability.rs,error.rs,filter.rs,image.rs,lib.rs,reference.rs,renderer.rs,backend.rs,gpu_transaction.rs,surface.rs,tests.rs}`; dispatch exact C08/C09 graphs through the same atomic headless/presented executor, install the exact final capability state and C10+ diagnostics, delete all transitional materialization/readback/re-upload calls, and gate every CPU reference/filter implementation to tests.
- RED/acceptance: `resolved_alpha_mask_preserves_partial_alpha_and_nested_order` fails only at `resolved masks do not compose inner to outer on GPU`; `resolved_alpha_mask_low_medium_high_and_extend_modes_match_boundary_oracle` fails only at `GPU mask quality or edge sampling exceeds S34`; `all_supported_blends_match_oracle_over_transparent_and_opaque_bases` fails only at `GPU blend output exceeds S34`; `plus_blend_clamps_high_precision_results` fails only at `Plus exceeded the unit interval`; `outer_clip_precedes_mask_and_opacity_on_unfiltered_sources` fails only at `C09 outer operations changed order`; `render_window_smoke_executes_masked_and_blended_graph_frames` fails only at `presented C09 composition did not commit atomically`; `c10_plus_graph_inputs_return_exact_gpu_unavailable_diagnostic_without_publication` fails only at `a future graph entered CPU execution or changed publication`; `cpu_reference_algorithms_are_test_only_after_gpu_cutover` fails only at `CPU pixel code remains in the production module graph`. Pass covers real supported high/reduced formats, arbitrary mask bounds/transforms, every quality/extend combination and RGBA/BGRA presentation, submits one frame stage, applies the transition matrix exactly, and preserves old publication/stats on every unavailable/failure/cancellation result.
- Commands: run the first five names separately with `run_exact_test`; `run_exact_test render_window_smoke_executes_masked_and_blended_graph_frames render-window`; run the final two names separately with `run_exact_test`; `run_exact_test graph_render_path_submits_without_map_or_cpu_wait`; `run_exact_test headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`; then `C09-CHECK`.
- Depends: T6. Commit: `Execute masks and blends on the GPU`.

**T8. Close capability truth reuse and failure evidence**
- Area/outcome: integrated `src/{capability.rs,error.rs,frame.rs,pass.rs,shader.rs,resource.rs,backend.rs,gpu_transaction.rs,renderer.rs,lib.rs,tests.rs}`; verify the T7 semantic capability/diagnostic boundary, prove bounded cache/upload/coverage reuse and zero-budget cleanup, and close dispatch, cancellation, device, and production-path guards.
- RED/characterization/acceptance: before production changes, `c09_capabilities_name_only_gpu_semantics_and_keep_broad_masks_diagnostic`, scoped to the production/public surface including command normalization rather than test source, `repeated_masked_and_blended_frames_reuse_resources_without_growth_or_readback`, and `budget_zero_releases_composition_resources_without_changing_pixels` pass as behavior-preserving characterization evidence for capability truth, reuse, and zero-budget cleanup already established by T1-T7; `renderer_dispatch_routes_c08_and_c09_to_gpu_but_future_passes_to_typed_diagnostics` fails only at `dispatch has no exact C09 boundary`. After implementation all four pass. Pass proves every old operation/accessor absent, resolved mask/image/composite/nested-opacity true, planning/resource facts true, future GPU execution and broad mask/filter/backdrop false with exact operations, stable counts after warm-up, exact pixels, one submission, retained upload reuse by key, deterministic coverage/effect cleanup, terminal-device release, and no CPU materialization/re-upload route.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test repeated_frames_reuse_resources_without_growth_or_readback`; `run_exact_test budget_zero_releases_idle_resources_without_changing_pixels`; `run_exact_test uncaptured_gpu_error_faults_only_its_device_generation`; `run_exact_test device_loss_is_terminal_idempotent_and_releases_device_resources`; run the completion guards; then `C09-CHECK`.
- Depends: T7. Commit: `Complete GPU composition and mask execution`.

## Completion
- Require all eight ordered task ranges and clean fresh task reviews. The coordinator then changes only this plan status from `in_progress` to `complete` and commits that status separately. Run `C09-CHECK` and every T1-T8 named test with its stated feature set; T7 replaces T2's staged `renderer_dispatch_routes_only_closed_c08_graph_subset_to_gpu_executor` name with the final `renderer_dispatch_routes_closed_c08_and_c09_graph_subsets_to_gpu_executor`, so final completion runs the latter exactly, while T8 separately runs its named future-diagnostic dispatch test. Require exact extent/origin, S34 high/reduced evidence, one transaction submission, atomic headless/presented delivery, bounded reuse, correct broad diagnostics, and a clean worktree.
- Run these fail-closed guards: `assert_no_match -n 'pub struct ResolvedAlphaMaskExecution|pub use [^;]*ResolvedAlphaMaskExecution' src/capability.rs src/error.rs src/image.rs src/layer.rs src/lib.rs`; `assert_no_match -n 'MaterializedAlphaMaskExecution|supports_materialized_alpha_mask_execution|try_resolved_alpha_mask|for_resolved_mask|TransitionalTextureRole::ResolvedMask|materialize_resolved_layer_mask' src/capability.rs src/error.rs src/image.rs src/layer.rs src/lib.rs src/command.rs src/renderer.rs src/pass.rs src/resource.rs src/backend.rs`; `test "$(rg -n '^    pub const fn supports_resolved_alpha_mask_execution\(' src/capability.rs | wc -l | tr -d ' ')" -eq 1`; `test "$(rg -n '^    ResolvedAlphaMaskExecution,$' src/error.rs | wc -l | tr -d ' ')" -eq 1`; `assert_no_match -n 'queue\.submit|map_async|Device::poll' src/pass.rs src/shader.rs src/vello_engine --glob '*.rs'`; `assert_no_match -n 'register_texture|override_image' src/pass.rs src/shader.rs src/gpu_transaction.rs src/vello_engine/encoder.rs`; `assert_no_match -n 'pub use .*pass|pub use .*shader|PreparedGraph|WorkingFormat|GpuRenderGraph' src/lib.rs`; and require `git diff --unified=0 44fd908f60a4b0d1b073c7f9a11ebab8c1472ee6..HEAD -- '*.rs' | rg '^\+(?:\s*(?:allow|expect)\s*\(|.*#\s*!?\[[^]]*(?:allow|expect)\s*\()'` to return no match. Require `git ls-files -- 'src/shaders/**' | LC_ALL=C sort` to equal the three C08 shader paths plus `src/shaders/layer_composite.wgsl`, each with exactly one `include_str!` owner; `Cargo.toml` must equal the cycle base and no lockfile may be tracked.
- After final checks, run a fresh `surgeist-holistic-reviewer` against this complete-status plan, exact specification/sequence revisions, crate boundary, full `44fd908f60a4b0d1b073c7f9a11ebab8c1472ee6..HEAD` diff, tests, and Rust modeling guidance. Only CLEAN permits a second complete final-check run, canonical landing, immutable publication to authority `origin/main`, fresh fetch/readback, and handoff. C10 receives one verified GPU composition boundary for color-filter output; root receives no C09 edit.
- Block only for an unowned conflicting change, reviewed-source contradiction, unavailable required native GPU/presented execution, missing installed Rust 1.97 or authorized wasm target, or unavailable required Surgeist custom profile. No acquisition, CPU production renderer/fallback, skipped adapter, compatibility shim, hidden quality reduction, unsafe, guessed allocation/mapping, or partial C10+ execution is allowed.
