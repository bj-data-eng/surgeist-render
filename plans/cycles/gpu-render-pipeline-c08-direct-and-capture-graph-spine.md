# GPU Render Pipeline C08 Direct And Capture Graph Spine

## Header
- Cycle: `C08`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `d57e49d728ccd3ba9d65d3793a89faac00721aef` (C07).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: S15-S19, the root/normal-source-over/present subset of S23, S25-S26, S28, and C08-applicable S31-S35 evidence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `c1a203393b2549603c0a0d5698099f55018abe2e`, normalized SHA-256 `70b345be31ac5bf4fcae72e2d3c5901c8e04ad0b2c26080851bd2c50d7150cde`, entry `C08 Direct And Capture Graph Spine`.
- Outcome: keep effect-free scenes on the one-pass direct internal-Vello route while executing the first complete GPU graph spine from root clear through bounded capture, canonical premultiplication, normal source-over, safe output conversion, and atomic headless or presented delivery.

## Boundary
- C08's executable graph vocabulary is exactly `ClearRoot`, `VelloCapture(Some(_))`, `CanonicalizeCapture`, `Composite(Some(SpanSourceOver))`, and terminal `Present`. A private eligibility check rejects every missing payload, other pass kind, malformed order, unsupported output, or non-C08 binding before graph resource allocation, pass-cache mutation, Vello encoding, surface acquisition, or command-encoder mutation. C09 owns layer/clip/mask/blend composition; C10 owns color filters; C11 owns backdrop, blur, and shadow passes. Existing public graph plans containing those later operations keep their current transitional route until their owning cycle and never execute a partial C08 approximation.
- `DirectVello` remains the least-powerful production plan: one prepared raster pass, one transaction-owned command buffer, no effect texture, no custom shader, and unchanged public pixels/statistics. C08 adds a narrow `#[cfg(test)]` forced-graph constructor for ordinary commands so the production graph executor and pipelines can be compared before a naturally selectable C08-only public scene exists; it does not add a public option or alter planner selection.
- A capture maps local points by applying `capture_transform`, then `parent_to_surface`, then translation by the negative signed capture texel origin, then the positive capture raster scale. In the crate's application-order API this is `capture_transform.then(parent_to_surface)?.then(Transform::translation(-origin.x(), -origin.y())?)?.then(Transform::scale(scale, scale)?)?`. The capture uses the requested `Antialiasing`, transparent base, positive bounded extent, and `Rgba8Unorm` storage target. No origin clamp, hidden crop/downscale, 1x1 substitute, or graph-result Vello re-entry is allowed.
- `ClearRoot` clears the full selected working image once to the surface base color. `CanonicalizeCapture` samples straight RGBA8 and writes finite clamped `vec4(rgb * a, a)` numeric-sRGB values. `SpanSourceOver` first copies the complete old parent into a distinct same-format result and then renders only the bounded source into that result with fixed premultiplied blend factors `One` and `OneMinusSrcAlpha`; the copy-only parent is not a sampled binding. `Present` samples the final premultiplied image, clamps it, emits exact transparent black for zero alpha, otherwise safely unpremultiplies, and writes the selected `Rgba8Unorm` or `Bgra8Unorm` output.
- The C08 shaders are tracked source files `src/shaders/{canonicalize_capture,span_source_over,present}.wgsl`, loaded only with `include_str!`. `shader.rs` realizes exact sampler/layout/module/pipeline keys and safe explicit bytes. Span source-over binds only its sampled source plus `PassSpatialUniformBytes`; present binds only the final image plus that spatial uniform. No dummy composite/present parameter buffer, pointer cast, owned POD derive, hot reload, runtime shader string, or placeholder future-pass shader is permitted.
- One graph operation owns preparation, provisional cache objects, one ordered command encoder, zero or more Vello capture leases, effect leases, output draft/acquired surface image, submission, and host effect. Only `GpuOperationTransaction` submits. All Vello capture leases resolve through one checked encoding scope and become reusable only after the complete transaction is clean. Provisional pass-cache entries and graph frame cleanup commit only on success; error or cancellation drops/quarantines uncertain state and preserves the prior publication/stats.
- Headless graph output targets a draft `Rgba8Unorm` texture with storage, render-attachment, copy-source, and copy-destination usages required by direct rendering, graph present, and explicit later readback. Success atomically publishes the draft; failure/cancellation leaves the old publication byte-for-byte intact. Presented graph output acquires and writes the advertised RGBA8/BGRA8 surface view, presents only as the transaction host effect, and reuses existing lifecycle/error mapping. Production graph execution contains no map, poll, CPU pixels, or inter-pass wait; tests may explicitly read a successfully published headless result afterward.
- High and reduced precision use the same executor, WGSL, bindings, and pass order. Private tests may explicitly select a supported `WorkingFormat` only after validating its real device feature set; they cannot substitute a shader or CPU fake. Exact dimensions/origins precede S34 comparisons. Reduced low-alpha checks use alpha and `premul8`; high precision checks stable straight RGB.

## Impacts
| Area | C08 record |
| --- | --- |
| Public API | No new public type, field, builder, route, report, or reexport. Existing direct and transitional public behavior remains source-compatible inside this cycle. |
| Dependencies/features | Unchanged reviewed dependency and feature set. No acquisition, lockfile, build script, generated artifact, or fixture delta; use the existing licensed/provenanced Ahem fixture. |
| Modules | `pass.rs` owns exact subset validation, executable custom-pass scheduling/bindings/encoding; `shader.rs` owns checked C08 cache realization; `encode.rs`/`vello_engine` own transformed capture preparation/encoding; `backend.rs`/`gpu_transaction.rs` own transaction, output, submission, and publication; `renderer.rs` owns dispatch. |
| Artifacts | Add only the three reviewed WGSL implementation sources. No C09+ shader, API artifact, screenshot, binary fixture, or root/sibling edit. |
| MSRV/platform | Preserve Rust 1.97, Rust 2024, all four native feature states, wasm `render-web` compilation, native headless execution, and `render-window` presented execution. |
| Unsafe | `#![forbid(unsafe_code)]` remains effective. No owned unsafe, unsafe attribute, extern block, unsafe lint allowance, unchecked shader path, or backend-specific surface escape. |

## Tasks
Define these functions in each task shell before focused commands; they select an exact installed Rust 1.97 release toolchain, require the already-authorized wasm target, reject ambiguous tests, and fail on owned unsafe.
```sh
prepare_c08_toolchain() {
  local candidate host installed_toolchains patch verbose_version version_number
  installed_toolchains="$(rustup toolchain list)" || return $?
  C08_TOOLCHAIN=""
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
              stable|"stable-$host"|"1.97.$patch"|"1.97.$patch-$host")
                C08_TOOLCHAIN="$candidate"
                break
                ;;
            esac
            ;;
        esac
        ;;
    esac
  done
  test -n "$C08_TOOLCHAIN" || { printf 'required installed stable Rust 1.97.x release toolchain is unavailable; acquisition is not authorized\n' >&2; return 1; }
  export C08_TOOLCHAIN
}

require_c08_wasm_target() {
  local sysroot
  prepare_c08_toolchain || return $?
  sysroot="$(rustup run "$C08_TOOLCHAIN" rustc --print sysroot)" || return $?
  test -d "$sysroot/lib/rustlib/wasm32-unknown-unknown/lib" || { printf 'wasm32-unknown-unknown is not installed for %s; acquisition is not authorized\n' "$C08_TOOLCHAIN" >&2; return 1; }
}

cargo_c08() {
  prepare_c08_toolchain || return $?
  CARGO_NET_OFFLINE=true rustup run "$C08_TOOLCHAIN" cargo "$@"
}

cargo_c08_wasm() {
  require_c08_wasm_target || return $?
  CARGO_NET_OFFLINE=true rustup run "$C08_TOOLCHAIN" cargo "$@"
}

run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(cargo_c08 test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(cargo_c08 test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    cargo_c08 test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    cargo_c08 test -p surgeist-render "tests::$name" -- --exact
  fi
}

assert_no_owned_unsafe() {
  local file manifest output scan_status
  manifest="$(mktemp "${TMPDIR:-/tmp}/surgeist-c08-owned-rust.XXXXXX")" || return $?
  if git ls-files -z --cached --others --exclude-standard -- '*.rs' >"$manifest"; then
    :
  else
    scan_status=$?
    rm -f "$manifest" || return $?
    return "$scan_status"
  fi
  test -s "$manifest" || { rm -f "$manifest" || return $?; printf 'owned Rust manifest is empty\n' >&2; return 1; }
  while IFS= read -r -d '' file; do
    if output="$(rg -n --pcre2 '#\s*!?\[\s*(?:allow|expect)\s*\([^)]*\bunsafe(?:_[A-Za-z0-9_]+)?\b|#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' -- "$file" 2>&1)"; then
      rm -f "$manifest" || return $?
      printf '%s\n' "$output" >&2
      return 1
    else
      scan_status=$?
      if test "$scan_status" -ne 1; then
        rm -f "$manifest" || return $?
        printf '%s\n' "$output" >&2
        return "$scan_status"
      fi
    fi
  done <"$manifest"
  rm -f "$manifest"
}

assert_no_match() {
  local output scan_status
  if output="$(rg "$@" 2>&1)"; then
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
No helper mutates rustup, installs a target, updates an index, or acquires software. Every behavior RED first adds the exact named test plus only the narrowest `#[cfg(test)]` observation needed. It must compile and fail only at the stated assertion; setup, adapter absence, listing, panic-before-assertion, or unrelated failure is not a valid RED. Characterization tasks remain green. Each worker implements and commits only its assigned task after `C08-CHECK`; the coordinator obtains a fresh `surgeist-task-reviewer` review of the exact task range before advancing.

**C08-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
prepare_c08_toolchain
cargo_c08 fmt --check
cargo_c08 check -p surgeist-render
cargo_c08 test -p surgeist-render
cargo_c08 clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo_c08 test -p surgeist-render --features render-window
cargo_c08 clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
cargo_c08 test -p surgeist-render --features render-web
cargo_c08 clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
cargo_c08 test -p surgeist-render --features render-window,render-web
cargo_c08 clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
cargo_c08 check -p surgeist-render --all-targets
cargo_c08 check -p surgeist-render --all-targets --features render-window,render-web
native_getrandom_tree="$(cargo_c08 tree -p surgeist-render -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
tracked_lockfiles="$(git ls-files -- Cargo.lock)" || exit $?
test -z "$tracked_lockfiles"
assert_no_owned_unsafe
cargo_c08_wasm check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(cargo_c08_wasm tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
```

**T1. Close the C08 subset and exact capture mapping**
- Area/outcome: `src/{frame.rs,pass.rs,encode.rs,tests.rs}`; add one private executable-subset proof consumed by the backend, expose only the exact span/spatial facts needed by execution, and add explicit initial-transform scene lowering. Unsupported/later-cycle plans fail eligibility before allocation while the public transitional route remains available.
- RED/acceptance: `c08_executor_accepts_only_clear_capture_canonicalize_span_source_over_and_present` fails only at `C08 has no closed pre-allocation executable subset`; `bounded_capture_transform_preserves_signed_origin_texel_centers_and_scale` fails only at `bounded capture transform changed the signed texel-center mapping`. Pass exhaustively classifies every `RuntimePassKind`/composite payload, requires one root clear and terminal present, pairs every capture with exactly one canonicalization, preserves topological/read/result bindings, and proves the application-order transform formula for negative/fractional origins and scales `1.0`, `1.25`, and `2.0`.
- Commands: run both names separately with `run_exact_test`; `run_exact_test graph_base_color_is_initialized_once_and_isolation_is_transparent`; `run_exact_test maximal_vello_spans_preserve_authored_command_order`; `run_exact_test negative_and_fractional_origins_preserve_texel_center_mapping`; then `C08-CHECK`.
- Depends: none. Commit: `Close the C08 graph execution subset`.

**T2. Realize checked C08 shaders and provisional caches**
- Area/outcome: `src/{shader.rs,pass.rs,backend.rs,tests.rs}` plus the three exact `src/shaders/*.wgsl` files; implement real sampler/layout/module/pipeline creation for canonicalize, source-over, and present. Add a non-clone provisional cache update whose new handles are usable for encoding but enter the persistent cache only after the owning transaction succeeds.
- RED/acceptance: `c08_shader_cache_realizes_checked_programs_without_publishing_failed_entries` fails only at `C08 pass objects are not transactionally cached`; `c08_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms` fails only at `C08 pass layout contains a copy-only or dummy binding`. Pass loads tracked `include_str!` sources, uses exact format/program/blend keys, specializes RGBA/BGRA outputs, reuses exact committed entries, leaves no provisional entry after validation error/cancellation/device transition, excludes the copy-only composite parent, and removes `CompositeParameters`/`PresentParameters` from these two C08 layouts without weakening C09's typed key space.
- Commands: run both names separately with `run_exact_test`; `run_exact_test device_pass_cache_owns_exact_sampler_layout_shader_and_pipeline_key_spaces`; `run_exact_test pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod`; then `C08-CHECK`.
- Depends: T1. Commit: `Add checked C08 shader pipelines`.

**T3. Encode custom spine passes without submission**
- Area/outcome: `src/{pass.rs,shader.rs,backend.rs,tests.rs}`; make one scheduler encode root clear, canonicalization, copy-then-source-over, and output present into a caller-owned command encoder and external output view. Use exact prepared bindings and advance/release a pass only after its commands encode successfully.
- RED/acceptance: `custom_spine_encodes_clear_canonicalize_copy_source_over_and_present_in_order` fails only at `C08 custom pass scheduler has no executable ordered spine`; `span_source_over_copies_parent_then_uses_fixed_premultiplied_blend` fails only at `normal source-over sampled or overwrote its parent incorrectly`. Pass clears the full root once, uses bounded viewports/scissors without signed-origin truncation, copies full parent to a distinct result, blends source with `One`/`OneMinusSrcAlpha`, never reads/writes one subresource, emits present to the exact external format/extent, and neither submits nor maps/polls/waits.
- Commands: run both names separately with `run_exact_test`; `run_exact_test runtime_lowering_preserves_dependencies_and_last_use_releases`; `run_exact_test resource_preparation_is_private_allocation_safe_and_submission_free`; then `C08-CHECK`.
- Depends: T2. Commit: `Encode the custom graph spine passes`.

**T4. Encode bounded Vello captures into the graph transaction**
- Area/outcome: `src/{encode.rs,pass.rs,backend.rs,gpu_transaction.rs,tests.rs}` and `src/vello_engine/{encoder.rs,raster.rs,resources.rs,mod.rs}`; prepare each validated span with the T1 transform, requested AA, transparent base, and exact capture target, then encode all captures into the same graph encoder/check scope. Replace the single-direct-lease assumption with an aggregate pending commit while preserving the direct one-pass proof.
- RED/acceptance: `multiple_vello_captures_share_one_graph_encoder_and_transaction_commit` fails only at `bounded Vello captures cannot share one graph transaction`; `vello_capture_uses_transparent_base_requested_aa_and_exact_bounded_extent` fails only at `Vello capture changed its raster contract`. Pass aggregates zero-or-more non-clone capture leases, aborts every lease on any encode/scope failure, commits all only after transaction success, keeps direct payload cardinality exactly one, and creates no queue submission, atlas re-entry, RGBA image, or inter-capture wait.
- Commands: run both names separately with `run_exact_test`; `run_exact_test internal_vello_encoding_shares_the_frame_transaction_submission`; `run_exact_test encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`; `run_exact_test canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty`; then `C08-CHECK`.
- Depends: T3. Commit: `Encode bounded Vello graph captures`.

**T5. Submit and atomically publish the headless graph spine**
- Area/outcome: `src/{backend.rs,gpu_transaction.rs,pass.rs,renderer.rs,surface.rs,texture.rs,tests.rs}`; add a graph submission payload that owns the command buffer, prepared frame cleanup, capture commits, provisional cache update, and headless draft host effect. Add render-attachment usage to headless targets and a private test-only forced-graph entry that invokes the same production executor.
- RED/acceptance: `capture_canonicalize_present_round_trips_transparent_partial_and_opaque_pixels` fails only at `headless C08 graph pixels do not satisfy canonical output`; `reduced_precision_low_alpha_pixels_use_alpha_and_premul8_tolerances` fails only at `reduced C08 output violates alpha or premul8 tolerance`; `high_precision_low_alpha_pixels_preserve_straight_rgb` fails only at `high precision C08 output lost stable straight RGB`; `graph_render_path_submits_without_map_or_cpu_wait` fails only at `production C08 graph reached CPU-visible synchronization`. The graph-specific no-wait test is added before T5 introduces the forced graph entry and transaction payload, then records that real graph transaction rather than the existing direct payload. Pass exercises the full alpha/extreme vector in S34 through real supported high/reduced formats, exact dimensions/origins, and the same production shaders; success publishes once after clean scopes/signals and one submission, with no map/poll/inter-pass wait, while error/cancellation preserves prior pixels/stats and commits no cache/resource lease.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test render_path_submits_without_map_or_cpu_wait`; `run_exact_test headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`; `run_exact_test failed_frame_returns_all_leases_and_preserves_last_successful_stats`; then `C08-CHECK`.
- Depends: T4. Commit: `Publish headless graph frames atomically`.

**T6. Prove direct and graph raster parity**
- Area/outcome: `src/{renderer.rs,frame.rs,tests.rs}` and existing `tests/fixtures/fonts/ahem/*`; render the solid-shape and stable Ahem-glyph fixtures through direct and forced C08 graph routes using the real executor. Add source-readable comparison helpers for straight high-precision and alpha/premul8 reduced results without changing the fixture.
- RED/acceptance: `solid_shape_direct_and_graph_routes_match_on_interior_and_aa_edges` fails only at `solid direct/graph pixels exceed S34 tolerance`; `ahem_glyph_direct_and_graph_routes_share_ink_extent_and_capture_grid` fails only at `Ahem direct/graph capture grids or ink extents differ`; `direct_graph_parity_covers_every_antialiasing_and_scale_pair` fails only at `direct/graph parity matrix is incomplete`; `negative_bounds_and_subpixel_transforms_do_not_shift_capture` fails only at `transformed signed capture placement exceeds S34 tolerance`. Pass covers `{Area,Msaa8,Msaa16} x {1.0,1.25,2.0}` for both fixtures and each real supported working format, asserts requested AA and exact grid/origin, and separately renders nonidentity capture and parent transforms with negative/fractional origins. It checks exact output dimensions and signed origin before interior/edge/support tolerances and alpha-weighted centroid deltas of at most `0.25` high or `0.35` reduced device pixels, while leaving direct allocation/pass cardinality unchanged.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test internal_vello_direct_pixels_match_pinned_vello_characterization_cases`; `run_exact_test direct_vello_scene_uses_one_pass_and_no_effect_allocation`; then `C08-CHECK`.
- Depends: T5. Commit: `Prove direct and graph capture parity`.

**T7. Deliver graph frames to presented surfaces**
- Area/outcome: `src/{backend.rs,gpu_transaction.rs,renderer.rs,surface.rs,pass.rs,tests.rs}` under `render-window` and existing wasm compile gates; route an eligible prepared graph to one safely acquired advertised output view and make presentation the transaction host effect. Preserve resize/suspend/occlusion/loss and device terminal-state ordering.
- RED/acceptance: `render_window_smoke_executes_direct_and_graph_presented_frames` fails only at `presented C08 graph did not acquire submit and present through one transaction`; `presented_graph_output_specializes_rgba_and_bgra_without_channel_swap` fails only at `presented output format conversion changed RGBA semantics`. Pass supports advertised Rgba8/Bgra8, acquires only after complete preflight/preparation, presents only after clean submission, maps acquire/present errors through existing lifecycle, performs no headless publication, and leaves direct presented behavior and cache ownership unchanged.
- Commands: run both names separately with `run_exact_test <name> render-window`; `run_exact_test surface_resize_suspend_resume_and_two_surfaces_own_resources render-window`; `run_exact_test surface_operation_matrix_covers_every_kind_state_and_duplicate_transition render-window`; then `C08-CHECK`.
- Depends: T6. Commit: `Present GPU graph spine frames`.

**T8. Close dispatch, reuse, cancellation, and no-readback evidence**
- Area/outcome: integrated `src/{renderer.rs,backend.rs,gpu_transaction.rs,pass.rs,shader.rs,resource.rs,tests.rs}`; dispatch only exact eligible C08 plans, keep later-cycle plans transitional, close all failure/cancellation paths, and prove bounded resource/cache reuse without changing public reports.
- RED/acceptance: `repeated_frames_reuse_resources_without_growth_or_readback` fails only at `repeated C08 frames grew resources or entered readback`; `budget_zero_releases_idle_resources_without_changing_pixels` fails only at `zero retention changed C08 pixels or retained idle resources`; `renderer_dispatch_routes_only_closed_c08_graph_subset_to_gpu_executor` fails only at `renderer has no closed C08 graph dispatch boundary`. The dispatch test injects one validated exact-subset plan and one validated later-cycle plan into the same private production dispatcher: before T8 both lack the required differentiated route; after T8 only the exact subset reaches the T5 executor while the later plan retains transitional materialization. Pass also proves stable allocation/cache counts after warm-up, zero-budget idle release, exact pixels, one transaction submission per graph frame, no map/poll/wait/CPU pixels/atlas re-entry, old publication/stats on injected failure and canceled future, terminal-device cleanup, and unchanged direct/transitional dispatch outside the exact C08 subset.
- Commands: run the three RED names separately with `run_exact_test`; `run_exact_test graph_render_path_submits_without_map_or_cpu_wait`; `run_exact_test render_path_submits_without_map_or_cpu_wait`; `run_exact_test canceled_generic_submission_after_real_submit_clears_ownership_without_public_result`; `run_exact_test uncaptured_gpu_error_faults_only_its_device_generation`; `run_exact_test device_loss_is_terminal_idempotent_and_releases_device_resources`; run the completion guards; then `C08-CHECK`.
- Depends: T7. Commit: `Complete the direct and capture graph spine`.

## Completion
- Require all eight ordered task commits and clean fresh `surgeist-task-reviewer` reviews. The coordinator then changes only this plan's status from `in_progress` to `complete` and commits that status separately before final verification. Run `C08-CHECK` plus every T1-T8 named test above through `run_exact_test` with its stated feature set. Require exact direct/graph dimensions/origins and S34 tolerance evidence in real high/reduced production pipelines, one transaction per graph frame, atomic headless/presented delivery, bounded reuse, and a clean worktree.
- Run these fail-closed guards: `assert_no_match -n 'queue\.submit|map_async|Device::poll' src/pass.rs src/shader.rs src/vello_engine --glob '*.rs'`; `assert_no_match -n 'register_texture|override_image|Image::from_rgba' src/pass.rs src/shader.rs src/gpu_transaction.rs src/vello_engine/encoder.rs`; `assert_no_match -n 'CompositeParameters|PresentParameters' src/shaders`; `assert_no_match -n 'pub use .*pass|pub use .*shader|PreparedGraph|WorkingFormat|GpuRenderGraph' src/lib.rs`; and `assert_no_match -n '\ballow\s*\(\s*dead_code\b' src/pass.rs src/shader.rs src/backend.rs src/gpu_transaction.rs src/vello_engine/encoder.rs`. Separately require `git ls-files -- 'src/shaders/**' | LC_ALL=C sort` to equal exactly the three C08 WGSL paths, every file to have one `include_str!` owner, `Cargo.toml` to remain unchanged from the cycle base, and no tracked lockfile.
- After those final checks pass, run a fresh clean-context `surgeist-holistic-reviewer` against this complete-status plan, the exact specification/sequence revisions, crate boundary, full `d57e49d728ccd3ba9d65d3793a89faac00721aef..HEAD` diff, tests, and Rust modeling guidance. Only CLEAN permits a second complete final-check run, followed by canonical landing, immutable-lease publication to authority `origin/main`, fresh fetch/readback, and crate-candidate handoff. C09 receives the verified source-to-output graph spine; root receives no C08 edit.
- Block only for an unowned conflicting worktree change, reviewed-source contradiction, unavailable required native GPU/presented execution, missing installed Rust 1.97/authorized wasm target, or unavailable required Surgeist custom profile. No dependency/tool acquisition, CPU renderer/fallback, skipped adapter, compatibility shim, hidden quality reduction, unsafe, guessed allocation, or partial later-cycle execution is allowed.
