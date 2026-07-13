# GPU Render Pipeline C03 Internal Vello Raster Engine Cutover

## Header
- Cycle: `C03`; owner: `surgeist-render`; status: `in_progress`.
- Cycle base and published prerequisite: `957582a6f9ceedfcff9a728bca23d26fc108af8f` (C02).
- Specification: `plans/specs/gpu-render-pipeline.md` at `1e6517e4e33669d97b1f45c0df9c1de78ec4d07e`, normalized SHA-256 `db78f70e03a31430e949ac06de6628ca24a03cd53cf5dec453b43bcf4fbe53be`: S04-S06A, S07 raster phase, S10A, raster/internalization portions of S13A and S16-S17, S25-S26 device/resource boundary, S28 vello engine/backend/renderer/text/error rows, S29 FontData/Capabilities/internal-engine rows, and C03-applicable S31-S37 rows.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at `b46c9c2afb6f705fdaf928d640b3821a8e29c0c9`, normalized SHA-256 `3dab5afdeb5084026f4863a3f0f4dfa18de47441a2560e1f5cbd1562732d8bdf`, entry `C03 Internal Vello Raster Engine Cutover`.
- Outcome: characterize pinned output; replace external Vello raster/device ownership with the checked private engine; preflight selected glyphs; retain transaction-owned encoding/submission and leases; close dependencies/provenance.

## Boundary
- C02 is the clean published base: external `vello 0.9` owns raster execution, submission/resources, and device/surface conveniences. C03 creates private `src/vello_engine/`; no Vello identity crosses `src/lib.rs`.
- The sole import source is local pinned `vello-0.9.0`, checksum `261359dbef879f8110ef7e1c442246c838d33d3d91cb05e0ea9288d432760c9f`. `NOTICE-VELLO.md` records package/version/checksum/source URL, each imported file's pre-adaptation SHA-256 and adaptation, every omission, and both preserved license texts.
- No working tree or commit may contain executable Surgeist-owned `unsafe`, including a temporary upstream copy. The first import omits CPU/debug/map/poll/submission paths and replaces the trusted shader site with checked creation; `#![forbid(unsafe_code)]` remains effective.
- `PreparedVelloPass` carries recording, target intent, and resource intents only. Its encoder returns `VelloResourceLease`; only `GpuOperationTransaction` submits internal-raster command buffers, then commits or aborts the lease after scopes/signals resolve.
- C03 excludes C04 publication/present lifecycle and readback, later custom effects, root work, browser execution, and acquisition. `direct_vello_is_the_least_powerful_plan_for_effect_free_scenes` is C06-owned/non-C03; no C06 task detail belongs here. C04 receives only the private raster/device/resource boundary; C14 owns target, presented-smoke, documentation, and final platform evidence.

## Impacts
| Area | C03 record |
| --- | --- |
| Public API | Breaking: `FontData::try_from_bytes`; remove `from_bytes`, `Capabilities::VELLO_0_9`, and external-engine exposure; retain C04 lifecycle contracts unchanged. |
| Authorized dependencies | Exact S36 roles: normal `kurbo`, `peniko`, optional `surgeist-window`, `wgpu`, `bytemuck` (no crate-requested features), `log`, `png`, `skrifa` (`default-features=false`, `autohint_shaping,std`), `vello_encoding`, and WGSL-only `vello_shaders`; dev `pollster` and `proptest`. These exact dependencies are already transitively available and authorized; remove unused `vello` and `glifo`; add nothing else. |
| Artifacts/MSRV | Add only notice and two license artifacts; preserve Rust 1.97/Rust 2024. No generated artifacts, library lockfile artifact, or root detail. |
| RED rule | Behavior RED is a named test that uniquely selects and fails on missing behavior. An explicitly named typed-boundary API-shape compile RED may fail with its stated compiler error while the required public signature is absent; it is neither behavior RED nor setup failure. Provenance/dependency RED uses its stated deterministic artifact predicate. |

## Tasks
Define and source this function in each task shell before focused commands; it rejects zero or ambiguous selection and runs only an exact `tests::`-prefixed target.
```sh
run_exact_test() {
  local name="$1" target listing count
  test "$#" -eq 1 || return 64
  target="tests::$name: test"
  listing="$(CARGO_NET_OFFLINE=true cargo test -p surgeist-render -- --list)" || return $?
  count="$(printf '%s\n' "$listing" | awk -v target="$target" '$0 == target { count += 1 } END { print count + 0 }')"
  test "$count" -eq 1 || { printf 'expected one %s, found %s\n' "$target" "$count" >&2; return 1; }
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render "tests::$name" -- --exact
}
```

**C03-CHECK (run verbatim after every task)**
```sh
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
CARGO_NET_OFFLINE=true rustc +stable --version
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders
test -f NOTICE-VELLO.md && test -f LICENSES/Vello-0.9.0-APACHE-2.0.txt && test -f LICENSES/Vello-0.9.0-MIT.txt
test -z "$(git ls-files -- Cargo.lock)"
git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{'
```
`rustc +stable --version` reports Rust `1.97.x`. The unsafe command exits `1` when clean, `0` for a prohibited owned-Rust match, and any other status for command failure; the no-lockfile guard passes only when `Cargo.lock` is untracked.

**Required exact-test matrix**
| Exact `src/tests.rs` name(s) | Owner | Source |
| --- | --- | --- |
| `pinned_vello_characterization_cases_are_source_readable` | T1 | `src/tests.rs` |
| `font_data_try_from_bytes_api_shape` | T2 | `src/tests.rs` |
| `font_data_rejects_malformed_bytes_before_raster_lowering`<br>`font_data_rejects_out_of_range_collection_index_before_raster_lowering`<br>`font_data_constructor_never_panics_for_arbitrary_bytes_and_indices`<br>`font_lowering_rejects_malformed_lazy_tables_without_panic_or_gpu_work`<br>`selected_glyph_preflight_rejects_missing_outline_before_external_encoding`<br>`selected_glyph_preflight_validates_exact_outline_draw_settings`<br>`selected_glyph_preflight_validates_colr_palette_bitmap_and_png_inputs`<br>`selected_glyph_preflight_distinguishes_unsupported_image_from_malformed_data`<br>`external_glyph_resolver_omission_branches_are_blocked_by_preflight`<br>`unsupported_glyph_image_encoding_returns_render_failed_without_omission`<br>`ahem_font_data_validates_at_collection_index_zero`<br>`internal_vello_font_parsing_is_fallible_and_never_unwraps` | T2 | `src/tests.rs` |
| `prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority` | T3 | `src/tests.rs` |
| `internal_vello_checked_shader_creation_reports_validation_without_unsafe` | T4 | `src/tests.rs` |
| `surgeist_device_state_owns_selected_wgpu_handles`<br>`terminal_device_cleanup_drops_internal_engine_resources` | T5 | `src/tests.rs` |
| `encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`<br>`canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty`<br>`internal_vello_encoding_shares_the_frame_transaction_submission`<br>`direct_vello_scene_uses_one_pass_and_no_effect_allocation` | T6 | `src/tests.rs` |
| `internal_vello_direct_pixels_match_pinned_vello_characterization_cases`<br>`capabilities_current_report_semantics_without_backend_or_cpu_names`<br>`render_path_submits_without_map_or_cpu_wait` | T7 | `src/tests.rs` |
| `internal_vello_provenance_names_exact_package_checksum_source_file_hashes_and_adaptations` | T8 | `src/tests.rs` |

**T1. Pin characterization and provenance scaffold**
- Area/outcome: test-only `src/tests.rs` source-readable tables plus `NOTICE-VELLO.md` and both license artifacts; characterize six authored primitive families across `{Area, Msaa8, Msaa16} x {1.0, 1.25, 2.0}`, asserting S34 origin/dimensions, alpha/premultiplication, interiors, edges, placement, and tolerances.
- RED/acceptance: deterministic characterization-artifact RED, not behavior RED. The real external route is known working; missing source-readable expected tables and notice/license predicates fail, then observed pinned samples are recorded and asserted.
- Commands: `run_exact_test pinned_vello_characterization_cases_are_source_readable`; `test -f NOTICE-VELLO.md && test -f LICENSES/Vello-0.9.0-APACHE-2.0.txt && test -f LICENSES/Vello-0.9.0-MIT.txt`; then `C03-CHECK`.
- Depends: none. Commit: `Characterize pinned Vello raster output`.

**T2. Add fallible font data and private scene lowering**
- Area/outcome: `Cargo.toml`, `src/{text.rs,error.rs,lib.rs}`, `src/vello_engine/{mod.rs,scene.rs,glyph.rs}`, tests, and notice rows; add only authorized `png`, `skrifa`, and `vello_encoding`. `FontData::try_from_bytes(Vec<u8>, u32)` and private `ValidatedGlyphRun` complete all S10A selected-glyph reads before `vello_encoding` append.
- RED/acceptance: first add and run base-compatible `font_data_from_bytes_currently_accepts_malformed_bytes` as an uncommitted characterization probe, then remove it before candidate edits because `from_bytes` must disappear. Next add persistent `font_data_try_from_bytes_api_shape`; its raw exact Cargo command is the public API-shape RED and must fail with `E0599`. After the signature exists, that test and every matrix test below are behavior/API GREEN: exact S10A `InvalidValue`, no panic/raw bytes/fallback/unwrap/expect/omission/GPU work, and valid unsupported images map to `RenderFailed`.
- Commands: probe with `run_exact_test font_data_from_bytes_currently_accepts_malformed_bytes`, then remove that probe; add `font_data_try_from_bytes_api_shape` and run `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::font_data_try_from_bytes_api_shape -- --exact` for the expected `E0599` RED; after adding the signature run `run_exact_test font_data_try_from_bytes_api_shape`; then run `run_exact_test` separately for every T2 normative-matrix name; then `C03-CHECK`.
- Depends: T1. Commit: `Internalize fallible Vello scene glyph lowering`.

**T3. Internalize WGPU-free recording and raster scheduling**
- Area/outcome: `src/vello_engine/{recording.rs,raster.rs,mod.rs}` and tests; retain fixed coarse/fine scheduling and construct `PreparedVelloPass` with target/resource intents but no WGPU object, encoder, lease, or submission authority.
- RED/acceptance: behavior RED because the base delegates opaque recording; pass proves intended phases and WGPU-free preparation.
- Commands: `run_exact_test prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority`; `rg -n 'wgpu::|CommandEncoder|queue\.submit' src/vello_engine/recording.rs src/vello_engine/raster.rs` (exit `1` clean); then `C03-CHECK`.
- Depends: T2. Commit: `Internalize Vello recording and raster schedule`.

**T4. Add checked encoder resources and explicit lease**
- Area/outcome: `Cargo.toml`, `src/vello_engine/{shaders.rs,encoder.rs,resources.rs,mod.rs}`, tests, and notice rows; add authorized direct `bytemuck`, `log`, and WGSL-only `vello_shaders`. Checked resources accept transaction-owned encoding state and return an uncommitted `VelloResourceLease`, never submit.
- RED/acceptance: behavior RED for absent checked internal encoding; validation maps to `RenderFailed`, atlas intent is lease-owned, and the first import has no trusted shader, CPU/debug/hot-reload/profiler/map/poll/direct-submit path.
- Commands: `run_exact_test internal_vello_checked_shader_creation_reports_validation_without_unsafe`; `rg -n 'create_shader_module_trusted|queue\.submit|map_async|\.poll\(|use_cpu|CpuShader' src/vello_engine` (exit `1` clean); then `C03-CHECK`.
- Depends: T3. Commit: `Add checked Vello raster encoder and leases`.

**T5. Replace Vello utility device and surface ownership**
- Area/outcome: `src/{backend.rs,renderer.rs,surface.rs}`, `src/vello_engine/{resources.rs,mod.rs}`, and tests; own WGPU instance/adapter/device/queue, surface handles, per-device engine state, and resources while preserving current behavior/lifecycle. C04 publication/presentation semantics remain unimplemented; the external renderer may temporarily consume owned handles only as a utility/comparison path.
- RED/acceptance: behavior RED for utility-owned identity/cleanup; pass proves selected-device identity, callbacks, terminal cleanup, no utility owner, and unchanged current behavior boundary.
- Commands: `run_exact_test surgeist_device_state_owns_selected_wgpu_handles`; `run_exact_test terminal_device_cleanup_drops_internal_engine_resources`; `rg -n 'vello::util::(RenderContext|DeviceHandle|RenderSurface)' src` (exit `1` clean); then `C03-CHECK`.
- Depends: T4. Commit: `Move raster device and surface ownership into Surgeist`.

**T6. Route internal raster submission through transactions**
- Area/outcome: `src/{backend.rs,gpu_transaction.rs,renderer.rs}`, `src/vello_engine/{encoder.rs,resources.rs}`, and tests; only internal Vello raster buffers move under `GpuOperationTransaction`. Scope/signal success commits a lease; cancellation/error aborts/quarantines it. The existing `backend.rs` submission is C04-present blit, not moved internal raster; readback and later custom-effect submits retain their owners.
- RED/acceptance: behavior RED for the absent transaction-owned internal pass; pass proves one effect-free direct raster pass/submission stage, dirty/recreated uncertain atlas state, and no engine submit/map/poll.
- Commands: `run_exact_test encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`; `run_exact_test canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty`; `run_exact_test internal_vello_encoding_shares_the_frame_transaction_submission`; `run_exact_test direct_vello_scene_uses_one_pass_and_no_effect_allocation`; then `C03-CHECK`.
- Depends: T5. Commit: `Submit internal Vello raster through transactions`.

**T7. Cut production over and rename capability**
- Area/outcome: production callers/tests in `src/{backend.rs,encode.rs,renderer.rs,surface.rs,capability.rs,tests.rs}` and `src/vello_engine/`; private engine becomes the sole direct-raster path, removes `vello::` and temporary external utility/comparison paths, and renames `Capabilities::VELLO_0_9` to `Capabilities::CURRENT` without CPU fallback or silent glyph omission.
- RED/acceptance: behavior RED for external-route dependence; pass requires six-family pinned parity, current capability semantics without backend/CPU names, and submission without map/CPU wait.
- Commands: `run_exact_test internal_vello_direct_pixels_match_pinned_vello_characterization_cases`; `run_exact_test capabilities_current_report_semantics_without_backend_or_cpu_names`; `run_exact_test render_path_submits_without_map_or_cpu_wait`; `rg -n '\bvello::' src --glob '*.rs'` (exit `1` clean); then `C03-CHECK`.
- Depends: T6. Commit: `Cut production raster over to internal Vello engine`.

**T8. Close dependencies and exact provenance**
- Area/outcome: `Cargo.toml`, notice/licenses, imported headers, and deterministic dependency/provenance tests; remove `vello` and `glifo`, set each S36 role exactly once, complete imported hashes/adaptations/omissions, and add no dependency beyond the authorized already-present S36 set.
- RED/acceptance: deterministic artifact RED for missing final roles/provenance; pass parses manifest/notice fixtures for package checksum, files/hashes/adaptations/omissions/licenses, intended direct uses, dev-only `pollster`, featureless crate request for `bytemuck`, and WGSL-only `vello_shaders`.
- Commands: `run_exact_test internal_vello_provenance_names_exact_package_checksum_source_file_hashes_and_adaptations`; `CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1`; `CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1`; `CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck`; `CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders`; `test -z "$(git ls-files -- Cargo.lock)"`; then `C03-CHECK`.
- Depends: T7. Commit: `Remove external Vello and close provenance`.

## Completion
- Require all eight ordered task ranges and clean reviews; the private checked engine owns direct raster/device/resources, selected glyphs preflight, and only transactions submit internal raster. No CPU fallback, direct engine submit/map/poll, trusted shader, executable owned unsafe, browser acquisition, or wasm execution claim is permitted.
- Final evidence runs `C03-CHECK` verbatim, every matrix name through `run_exact_test`, and the T8 exact provenance test plus its offline normal/dev/inverse-feature trees. Re-run `test -z "$(git ls-files -- Cargo.lock)"` and require `git ls-files --error-unmatch NOTICE-VELLO.md LICENSES/Vello-0.9.0-APACHE-2.0.txt LICENSES/Vello-0.9.0-MIT.txt` to succeed after the committed task ranges exist.
- Final targeted engine guards are `rg -n '^vello\s*=|^glifo\s*=' Cargo.toml`, `rg -n '\bvello::' src --glob '*.rs'`, and `rg -n 'create_shader_module_trusted|queue\.submit|map_async|\.poll\(|use_cpu|CpuShader|RenderContext|RenderSurface|DeviceHandle' src/vello_engine`; each exits `1` clean, `0` on prohibited match, and otherwise fails. The C03-CHECK unsafe command uses the same exit semantics and the tracked notice/license predicate is required evidence.
- Handoff/non-goal: after clean plan/task/holistic review, landing, offline gates, publication, and remote readback, C04 receives only the candidate SHA plus C03 raster/device/resource evidence, not presentation lifecycle, readback, custom effects, root integration, browser execution, or final platform evidence. Block only for unowned worktree conflict, reviewed-packet contradiction, required native GPU failure, unavailable exact Rust 1.97, or missing tooling requiring separately authorized acquisition.
