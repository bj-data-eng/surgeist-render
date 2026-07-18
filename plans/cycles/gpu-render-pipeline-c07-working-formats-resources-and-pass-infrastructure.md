# GPU Render Pipeline C07 Working Formats Resources And Pass Infrastructure

## Header
- Cycle: `C07`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `53e406ab768b96a540bd523f727bdb54f1cace03` (C06).
- Specification: `plans/specs/gpu-render-pipeline.md` at `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized SHA-256 `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`: S12, S16, S18, S25, S28, and C07-applicable S31-S35 evidence; S13 typed effect errors, S20 exact kernel-key facts, and S36-S37 feature/dependency/verification rules apply only where those sections define a named C07 invariant or gate.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `c1a203393b2549603c0a0d5698099f55018abe2e`, normalized SHA-256 `70b345be31ac5bf4fcae72e2d3c5901c8e04ad0b2c26080851bd2c50d7150cde`, entry `C07 Working Formats Resources And Pass Infrastructure`.
- Outcome: resolve one private working format from immutable device facts and policy, replace competing texture/resource caches with one generation-aware per-device manager for internal Vello and effect resources, and lower C06 semantic graphs into a finite backend-ready pass/resource plan with safe shader serialization and cache ownership for C08 execution.

## Boundary
- C06 supplies one validated `DirectVello` or `GpuGraph` plan with complete semantic pass/resource intents, signed mappings, bounded extents, dependencies, and last-use information. C07 may consume that graph through a narrow crate-private lowering view; it does not re-walk authored commands, infer bounds, change ordering, or expose graph/resource identities publicly.
- Private `WorkingFormat::{HighPrecision, ReducedPrecision}` is the backend phase for premultiplied numeric-sRGB intermediates. High maps to `Rgba16Float`, reduced to `Rgba8Unorm`; both require render attachment, sampled/filterable texture, and copy source/destination use. Selection prefers high, permits reduced only under `AllowReducedPrecision`, and otherwise returns `RuntimeCapabilityUnavailable(EffectRendering, EffectFormatUnavailable { policy })`. Every nonempty effect extent is checked against the selected device limit before allocation and returns the exact `EffectTextureAllocation` dimension diagnostic on failure.
- One `ResourceManager` is created with each ready device and the renderer's fixed `ResourceCacheBudget`. It is the sole owner of internal Vello allocations, effect/capture/coverage textures, retained resolved-mask uploads, and Gaussian kernel buffers. Surface swapchain images and the headless draft/published lifecycle remain surface/backend-owned outputs, not idle effect-cache entries.
- Private semantic newtypes distinguish frame, resource, and allocation generations. A non-clone lease belongs to one manager/frame/resource generation; production callers cannot manufacture, copy, or release it twice. Scope-owned frame cleanup returns every outstanding lease on success, error, or cancellation. Replacement increments the resource generation, and test-only owner observations may inject stale tokens without adding a production-visible API.
- Exact cache keys include role as well as descriptor: Vello buffer/image role and allocation facts; effect texture role, format, extent, and usage; resolved mask `ImageId`, dimensions, `ImageQuality`, and `Extend`; Gaussian standard-deviation bits, raster-scale bits, support policy/radius, and sampling form. Distinct Vello atlas, capture, working, coverage, mask, and kernel roles never alias.
- Checked `u64` byte accounting covers retained textures by actual format/extent, mask uploads, and kernel buffers. Idle trimming runs at frame cleanup in ascending `(last_used_frame, resource_id)` order until retained bytes fit the configured budget; zero budget drops all idle byte-accounted resources. Active bytes remain internal evidence and are not rejected by the retention budget. Pipeline objects are device-lifetime caches with no fabricated byte estimate.
- `pass.rs` owns a closed C07 runtime vocabulary corresponding one-to-one with S16 and a validated lowered plan containing exact resource requests, pass dependencies, read bindings, result bindings, last-use releases, working format, and pipeline/sampler/layout keys. `ClearRoot` and `VelloCapture` retain their specialized owners. C07 adds no WGSL file, shader module, render pipeline, or executable custom-pass program: `shader.rs` adds only the exact cache owners/keys and the 48-byte `PassSpatialUniformBytes` serializer described in T6. C08 and each later execution cycle add only their real tracked `include_str!` sources and populate those caches when their pass semantics become executable; no placeholder program is permitted.
- C07 allocates no graph from `Renderer::render`, encodes no C08 graph command buffer, submits no custom pass, maps/polls no resource, performs no effect readback/re-entry, changes no pixels/routes/public report, and claims no C08-C12 execution. Transitional mask/backdrop materialization remains behaviorally intact but obtains any temporary offscreen allocation through the same per-device manager. C07 leaves the CPU oracle untouched; C13 removes production CPU/materialized routes and isolates the oracle under `#[cfg(test)]`. C14 owns final platform/docs evidence; root owns facade/API artifacts/gitlink work.

## Impacts
| Area | C07 record |
| --- | --- |
| Public API | Internal-only. `Options`, runtime capability reports/errors, `Stats`, routes, and public reexports remain unchanged; no working format, resource handle, graph/pass key, WGPU, or Vello type crosses `lib.rs`. |
| Dependencies/features | Unchanged exact S36 set. No dependency, feature, acquisition, lockfile, build script, generated artifact, fixture, or license delta. |
| Modules | Add private `resource.rs` and `pass.rs`; converge `texture.rs`, `shader.rs`, `backend.rs`, and `vello_engine/resources.rs` onto one manager/cache authority. Add no `src/shaders/` artifact in C07. |
| Docs/MSRV/root | No README/example/root edit. Preserve Rust 1.97, Rust 2024, all four native feature states, and wasm `render-web` compilation. C08 receives the private lowering/resource contract after publication. |
| Unsafe | `#![forbid(unsafe_code)]` remains effective; serializers use explicit little-endian bytes and documented WGSL offsets, with no owned `Pod`/`Zeroable`, pointer cast, unsafe lint allowance, or executable owned unsafe. |

## Tasks
Define these functions in each task shell before focused commands; they select an exact installed Rust 1.97 toolchain, reject an absent authorized wasm target, reject ambiguous tests, and fail on owned unsafe.
```sh
prepare_c07_toolchain() {
  local candidate host installed_toolchains patch verbose_version version_number
  installed_toolchains="$(rustup toolchain list)" || return $?
  C07_TOOLCHAIN=""
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
                C07_TOOLCHAIN="$candidate"
                break
                ;;
            esac
            ;;
        esac
        ;;
    esac
  done
  test -n "$C07_TOOLCHAIN" || { printf 'required installed stable Rust 1.97.x release toolchain is unavailable; acquisition is not authorized\n' >&2; return 1; }
  export C07_TOOLCHAIN
}

require_c07_wasm_target() {
  local sysroot
  prepare_c07_toolchain || return $?
  sysroot="$(rustup run "$C07_TOOLCHAIN" rustc --print sysroot)" || return $?
  test -d "$sysroot/lib/rustlib/wasm32-unknown-unknown/lib" || { printf 'wasm32-unknown-unknown is not installed for %s; acquisition is not authorized\n' "$C07_TOOLCHAIN" >&2; return 1; }
}

cargo_c07() {
  prepare_c07_toolchain || return $?
  CARGO_NET_OFFLINE=true rustup run "$C07_TOOLCHAIN" cargo "$@"
}

cargo_c07_wasm() {
  require_c07_wasm_target || return $?
  CARGO_NET_OFFLINE=true rustup run "$C07_TOOLCHAIN" cargo "$@"
}

run_exact_test() {
  local name="$1" features="${2-}" target listing count
  test "$#" -ge 1 && test "$#" -le 2 || return 64
  target="tests::$name: test"
  if test -n "$features"; then
    listing="$(cargo_c07 test -p surgeist-render --features "$features" -- --list)" || return $?
  else
    listing="$(cargo_c07 test -p surgeist-render -- --list)" || return $?
  fi
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  if test -n "$features"; then
    cargo_c07 test -p surgeist-render --features "$features" "tests::$name" -- --exact
  else
    cargo_c07 test -p surgeist-render "tests::$name" -- --exact
  fi
}

assert_no_owned_unsafe() {
  local file manifest output scan_status
  manifest="$(mktemp "${TMPDIR:-/tmp}/surgeist-c07-owned-rust.XXXXXX")" || return $?
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
`prepare_c07_toolchain` admits only installed `stable`, `stable-<reported-host>`, `1.97.<patch>`, or `1.97.<patch>-<reported-host>` names and requires the compiler version token to be exactly `1.97.<digits>` with no prerelease suffix; advanced stable, nightly/beta, and custom aliases are not MSRV evidence. `assert_no_owned_unsafe` first persists and verifies Git's nonempty owned-Rust enumeration, then scans each exact NUL-delimited path without `xargs` and removes the manifest on every handled exit; only per-file ripgrep status `1` continues. No helper mutates rustup, installs a target, or acquires software.

**C07-CHECK (run verbatim after every task)**
```sh
set -euo pipefail
prepare_c07_toolchain
cargo_c07 fmt --check
cargo_c07 check -p surgeist-render
cargo_c07 test -p surgeist-render
cargo_c07 clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo_c07 test -p surgeist-render --features render-window
cargo_c07 clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
cargo_c07 test -p surgeist-render --features render-web
cargo_c07 clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
cargo_c07 test -p surgeist-render --features render-window,render-web
cargo_c07 clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
cargo_c07 check -p surgeist-render --all-targets
cargo_c07 check -p surgeist-render --all-targets --features render-window,render-web
native_getrandom_tree="$(cargo_c07 tree -p surgeist-render -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$native_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 0
tracked_lockfiles="$(git ls-files -- Cargo.lock)" || exit $?
test -z "$tracked_lockfiles"
assert_no_owned_unsafe
cargo_c07_wasm check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
wasm_getrandom_tree="$(cargo_c07_wasm tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4)" || exit $?
test "$(printf '%s\n' "$wasm_getrandom_tree" | awk '/getrandom feature "wasm_js"/ { count += 1 } END { print count + 0 }')" -eq 1
```
Every task uses the default profile for its named tests. For behavior RED, first add the named test and only a narrow `#[cfg(test)]` observation over the current owner. The exact test must compile and fail only at the stated assertion; listing, setup, adapter absence, panic-before-assertion, or unrelated failure is invalid. A refactor task first records the named characterization and keeps it green throughout. No worker acquires tooling or dependencies.

**T1. Resolve effect precision and device limits**
- Area/outcome: `src/{backend.rs,resource.rs,error.rs,tests.rs}`; add private `WorkingFormat`, exact required WGPU features/usages/bytes, and one named contextual resolver on immutable `DeviceCapabilities`. Validate nonempty extents before any manager allocation and preserve independent public capability flags.
- RED/acceptance: `precision_resolver_covers_both_high_only_reduced_only_and_neither` fails only at `effect precision has no typed deterministic resolver`; `effect_texture_dimension_is_rejected_before_allocation` fails only at `over-limit effect extent reached allocation`. Pass covers `{high,reduced} = {TT,TF,FT,FF}` under both policies, exact high preference/reduced permission/error payload, no backend-name/downscale/CPU/Vello fallback, and exact dimension operation/reason.
- Commands: run both names separately with `run_exact_test`; `run_exact_test runtime_capability_report_keeps_precision_flags_independent`; then `C07-CHECK`.
- Depends: none. Commit: `Resolve GPU effect working formats`.

**T2. Model generation-safe leases and deterministic retention**
- Area/outcome: new private `src/resource.rs`, converged lifecycle model from `src/texture.rs`, and tests; add manager/frame/resource identities, exact role-bearing keys, `Idle`/`Leased` state, non-clone leases, checked byte accounting, scope cleanup, and deterministic budget trim without WGPU allocation yet.
- RED/acceptance: `resource_leases_reject_stale_generation_and_double_release_by_model` fails only at `resource leases do not encode manager frame and allocation generations`; `resource_trim_order_is_last_used_then_resource_identity` fails only at `idle trim order is not deterministic`; strengthen `resource_cache_budget_zero_disables_idle_retention` to fail only at `zero budget retained an idle byte-accounted resource`. Pass makes production double release unconstructable, test-only stale injection rejected, active resources untrimmed, replacement generation monotonic, overflow typed, and equal-age trimming ordered by resource identity.
- Commands: run all three names separately with `run_exact_test`; then `C07-CHECK`.
- Depends: T1. Commit: `Model per-device resource leases and retention`.

**T3. Own concrete effect textures masks and kernels**
- Area/outcome: `src/{resource.rs,texture.rs,image.rs,command.rs,frame.rs,backend.rs,tests.rs}`; allocate real effect/capture/coverage textures, retained mask uploads, and immutable Gaussian kernel buffers through one manager. Normalize transitional resolved mask bytes into a private Image-backed upload descriptor without changing the public mask API or pixels.
- RED/acceptance: `effect_texture_keys_separate_format_extent_usage_and_role` fails only at `effect textures can alias across semantic roles`; `resolved_mask_upload_keys_include_identity_dimensions_and_sampling` fails only at `mask upload key omits semantic image facts`; `gaussian_kernel_buffer_keys_include_the_exact_plan` fails only at `kernel buffer key omits exact planning facts`. Pass checks WGPU limits first, accounts exact bytes, safely uploads validated lengths, reuses only exact idle keys, keeps Vello/mask/effect namespaces disjoint, and exposes resources only through generation-checked leases.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test image_buffer_rejects_short_long_and_overflowing_byte_lengths`; then `C07-CHECK`.
- Depends: T2. Commit: `Allocate retained GPU effect resources`.

**T4. Move Vello and transitional offscreen allocations under the manager**
- Area/outcome: `src/{backend.rs,renderer.rs,resource.rs,texture.rs,gpu_transaction.rs,tests.rs}` and `src/vello_engine/{encoder.rs,resources.rs,mod.rs}`; replace `VelloResourceManager` plus per-effect `OffscreenTextureResourceCache` instances with the ready device's sole `ResourceManager`. Preserve Vello abort/atlas recovery and all current transitional mask/backdrop pixels.
- RED/acceptance: first characterize current direct, canceled Vello, mask, and bounded-backdrop outcomes with existing tests. Then `one_ready_device_owns_one_raster_and_effect_resource_manager` fails only at `raster and effect allocations still have competing owners`. Pass routes every internal raster/effect allocation through one device manager, returns scopes on success/error/cancellation, trims only idle entries, preserves prior stats/publication, and terminal transition drops manager/caches once. Remove superseded broad dead-code allowances and duplicate lifecycle/cache types rather than suppressing them.
- Commands: `run_exact_test one_ready_device_owns_one_raster_and_effect_resource_manager`; `run_exact_test encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`; `run_exact_test canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty`; `run_exact_test layer_resolved_alpha_mask_applies_after_children_before_parent_composite`; `run_exact_test render_materializes_bounded_backdrop_capture_from_prior_siblings`; `run_exact_test failed_frame_returns_all_leases_and_preserves_last_successful_stats`; `run_exact_test device_loss_is_terminal_idempotent_and_releases_device_resources`; then `C07-CHECK`.
- Depends: T3. Commit: `Unify per-device raster and effect resources`.

**T5. Lower semantic graphs into the closed runtime pass vocabulary**
- Area/outcome: new private `src/pass.rs`, key model in `src/shader.rs`, narrow crate-private C06 graph view in `src/frame.rs`, and tests; add `SamplerKey`, `BindGroupLayoutKey`, `ShaderModuleKey`, and `RenderPipelineKey`, then consume a complete `GpuRenderGraph` plus resolved format/capabilities into one immutable lowered plan. Preserve all S16 kinds, dependencies, read/result bindings, spatial descriptors, imported upload keys, last-use releases, and exact cache keys without WGPU objects or command authority.
- RED/acceptance: `semantic_graph_lowers_to_finite_runtime_pass_and_resource_vocabulary` fails only at `semantic graph has no backend-ready closed lowering`; `runtime_lowering_preserves_dependencies_and_last_use_releases` fails only at `runtime lowering changed graph order or lifetime`; `runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys` fails only at `lowered pass omitted its exact cache keys`. Pass maps every semantic kind exactly once, selects one format per graph, keeps Vello capture RGBA8 distinct from canonical working images, derives keys from program/layout/sampling/source/working/output format facts, validates extents before requests, rejects missing/duplicate/forward/stale bindings without a partial result, and cannot express unknown/custom escape variants.
- Commands: run all three names separately with `run_exact_test`; `run_exact_test graph_builder_rejects_forward_stale_and_read_write_aliases`; `run_exact_test drop_shadow_source_fanout_lives_through_both_consumers`; then `C07-CHECK`.
- Depends: T4. Commit: `Lower semantic graphs to runtime passes`.

**T6. Add safe pass serialization and device-lifetime cache ownership**
- Area/outcome: `src/{shader.rs,pass.rs,resource.rs,backend.rs,renderer.rs,tests.rs}`; remove the rect-probe model and migrate `Renderer::scoped_clear_fill_probe_for_test` plus every direct test caller to a direct test-only WGPU render-pass clear that preserves transaction, cancellation, fault, and readback evidence without a production custom-pass abstraction. Use T5's final key types, and add exactly `PassSpatialUniformBytes([u8; 48])` plus one `DevicePassCache` per ready device. The 48-byte WGSL-compatible uniform layout is: source origin `vec2<f32>` at bytes `0..8`, source raster scale `f32` at `8..12`, zero pad at `12..16`, destination origin `vec2<f32>` at `16..24`, destination raster scale `f32` at `24..28`, zero pad at `28..32`, source extent `vec2<u32>` at `32..40`, and destination extent `vec2<u32>` at `40..48`. C07 production creates no shader source/module/pipeline and submits no custom pass; later cycles supply validated executable programs to the typed cache owner.
- RED/acceptance: `pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod` fails only at `pass spatial serialization has no explicit 48-byte contract`; `pass_spatial_uniform_rejects_f32_underflowing_raster_scales` fails only at `positive f64 raster scale narrowed to zero`; `device_pass_cache_owns_exact_sampler_layout_shader_and_pipeline_key_spaces` fails only at `device pass cache does not separate exact key spaces`; `c07_contains_no_placeholder_custom_shader_program` fails only at `C07 introduced a custom shader program before executable semantics`. Pass narrows each scalar, then rejects any non-finite `f32` and any source/destination raster scale that is not strictly positive in `f32` with typed `InvalidValue`; the underflow test uses a finite positive `f64` that becomes `0.0f32`, and the layout test covers finite overflow to infinity. It writes the exact offsets/padding, uses no pointer cast/owned POD derive, distinguishes working/output formats and program/layout identities in keys, keeps unknown driver bytes out of resource budgets, adds no production submit/map/poll/readback authority or `src/shaders/` file, removes every `RectShader*`/`RectPassBounds`/`encode_clear_fill_pass` caller and contract-only test, and preserves the existing real-GPU transaction tests through the test-only clear seam.
- Commands: run all four names separately with `run_exact_test`; `run_exact_test shader_clear_fill_pass_encodes_when_gpu_context_is_available`; `run_exact_test non_readback_gpu_submissions_are_owned_by_gpu_operation_transactions`; `run_exact_test canceled_generic_submission_after_real_submit_clears_ownership_without_public_result`; `run_exact_test generic_submission_observation_remains_bound_across_interleaved_scope_resolution`; `run_exact_test uncaptured_gpu_error_faults_only_its_device_generation`; `run_exact_test real_gpu_smoke_emits_no_uncaptured_error`; `assert_no_match -n 'RectShader|RectPassBounds|encode_clear_fill_pass' src --glob '*.rs'`; then `C07-CHECK`.
- Depends: T5. Commit: `Add safe custom pass shader infrastructure`.

**T7. Close the C07 device and C08 handoff contract**
- Area/outcome: integrated `src/{backend.rs,renderer.rs,resource.rs,pass.rs,shader.rs,texture.rs,stats.rs,tests.rs}` and private Vello resource callers; construct exactly one manager and cache bundle per ready device from fixed renderer options, prove lifecycle cleanup/reuse, and expose only a private validated preparation seam for C08.
- RED/acceptance: `resource_preparation_is_private_allocation_safe_and_submission_free` fails only at `C08 has no complete private resource and pass preparation handoff`; `resource_budget_and_device_loss_preserve_public_stats_contract` fails only at `resource lifecycle leaked into final public stats`. Pass proves direct Vello allocates no effect texture, repeated preparation/release is bounded and deterministic, budget zero retains no idle byte-accounted resources, failure/device loss returns or drops every lease, `Stats`/public capability routes remain unchanged, and no current render starts custom graph execution.
- Commands: run both names separately with `run_exact_test`; `run_exact_test direct_vello_scene_uses_one_pass_and_no_effect_allocation`; `run_exact_test resource_cache_budget_zero_disables_idle_retention`; `run_exact_test failed_frame_returns_all_leases_and_preserves_last_successful_stats`; `run_exact_test device_loss_is_terminal_idempotent_and_releases_device_resources`; run the completion guards; then `C07-CHECK`.
- Depends: T6. Commit: `Complete GPU resource and pass preparation`.

## Completion
- Require all seven ordered task ranges and clean task reviews. High/reduced selection and extent rejection are typed; one persistent manager owns raster/effect allocations; leases, generations, accounting, reuse, trimming, and terminal cleanup are deterministic; C06 graphs lower once into the closed S16 runtime vocabulary; safe serialization and device-lifetime caches are ready; no final public report or C08+ pixel claim exists.
- Run `C07-CHECK` and every T1-T7 named test above through `run_exact_test`; require a clean worktree. Run these final guards through the defined fail-closed helper: `assert_no_match -n 'OffscreenTextureResourceCache|VelloResourceManager' src --glob '*.rs'`; `assert_no_match -n '\ballow\s*\(\s*dead_code\b' src/backend.rs src/resource.rs src/texture.rs src/shader.rs src/vello_engine/resources.rs`; `assert_no_match -n 'RectShader|RectPassBounds|encode_clear_fill_pass' src --glob '*.rs'`; `assert_no_match -n 'queue\.submit|map_async|Device::poll|read_texture_rgba|Image::from_rgba' src/resource.rs src/pass.rs src/shader.rs`; `assert_no_match -n 'reference::|super::reference|crate::reference' src/resource.rs src/pass.rs src/shader.rs`; and `assert_no_match -n 'pub use .*resource|pub use .*pass|WorkingFormat|ResourceLease|PipelineKey' src/lib.rs`. `assert_no_match` converts only ripgrep status `1` to success, rejects status `0`, and propagates status greater than `1`. Separately, `tracked_shader_files="$(git ls-files -- 'src/shaders/**')" || exit $?; test -z "$tracked_shader_files"` first propagates enumeration failure and then exits `0` only when C07 has no tracked shader artifact. Require exactly one ready-device `ResourceManager` and one `DevicePassCache` field, and no `Stats` route/precision/pass/resource field added in this range.
- After cycle acceptance, follow `$surgeist-agent`'s canonical status, final-check, holistic-review, landing, publication, remote-readback, and crate-candidate handoff gates. C08 receives the immutable candidate SHA plus private working-format/resource/pass preparation evidence; root receives no C07 edit.
- Block only for unowned worktree conflict, a contradiction in the reviewed specification/sequence, unavailable required native GPU execution, missing already-installed Rust 1.97/authorized wasm target, or unavailable required Surgeist custom agent profile. No dependency/tool acquisition, skipped adapter, fallback target, compatibility shim, CPU renderer, hidden quality reduction, guessed allocation, or unsafe is allowed.
