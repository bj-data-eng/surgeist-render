# P01-I03-S01-C13 GPU Only Cutover And Matrix Reconciliation

## 1 Header

- Cycle: `P01/I03/S01/C13`.
- Owning repository: `surgeist-render`.
- Status: `reviewed`.
- Cycle base and published prerequisite: `7bd4fcdaf23ebb0ba137496b17458dcbd1278e13`
  (`P01/I03/S01/C12`, remotely verified and explicitly closed).
- Specification: `plans/specs/gpu-render-pipeline.md`
  at `9903b58e8c96063bd834fb561baaf321116feead`,
  `sha256:d33bb1478eac256c75203b7ecaff450a12adc8158f3f2ba72e4ed891e9c6a9ce`;
  clauses S01, S03, S11, S14, S20, S25, S27, the remaining S29-S30C
  cutover and inventory rows, S35, and S38.
- Sequence: `plans/sequences/gpu-render-pipeline.md`
  at `75ba3b1c0a1d0f1c83734d69663eb6abb3061474`,
  `sha256:7a016389ae9925480c81b951840ca5bacf48d1c5c822e5f78dae472461618e9d`;
  entry `C13 GPU Only Cutover And Matrix Reconciliation`.
- Outcome: publish the final truthful semantic capability and render-statistics
  surface, prove that CPU/reference behavior is test-only, and reconcile every
  primitive, property, and typed-diagnostic inventory row without adding a
  production fallback, graph readback/re-entry path, or platform-evidence claim.

## 2 Boundary

- C12 is the immutable implementation base: direct internal Vello, ordered
  color filters, Gaussian blur, filter drop shadow, resolved alpha masks,
  composition, and bounded backdrop execution are independently GPU-complete.
- C13 changes semantic claims only where the corresponding production behavior
  already executes. GPU color-filter, blur, and drop-shadow queries become
  truthful; broad layer filters, materialized filtered-image paint, CPU
  fallback, broad masks, broad backdrop, and unsupported composite modes remain
  exact diagnostics or oracle-only rows.
- `Stats` gains the exact S14 route, precision, pass, copy, resource-acquisition,
  and post-trim retention observations. These facts describe one successfully
  published frame; they never authorize execution or expose graph, resource,
  shader, WGPU, or internal Vello handles.
- Direct frames report one internal Vello pass and no effect precision or
  effect-resource activity caused by the frame. Graph frames report only
  actually encoded operations and the selected high/reduced working precision.
- Resource allocation/reuse counts come from successful lease acquisition
  sources; retained bytes are the byte-accounted idle total after frame cleanup
  and deterministic trimming. Test-only probes cannot become the public source.
- Last-successful stats publish atomically with the frame. Planning,
  preparation, encode, scope, device-signal, acquisition, submission,
  presentation, cancellation, or publication failure preserves the prior
  public value.
- CPU pixel algorithms and materialized byte executors remain reachable only
  through `#[cfg(test)]` oracle code. Production `Renderer::render` cannot call
  readback, buffer mapping, `Device::poll`, rendered `Image::from_rgba`, graph
  replay, command cloning, or internal-raster atlas re-entry.
- C13 owns the three deterministic S30 inventories: 101 primitive/backend rows
  with exact `69 Supported / 24 Diagnostic / 6 Root / 2 OracleOnly` totals, 22
  property surfaces, and every stable S30C diagnostic subcase exactly once.
- C13 does not add dependencies, features, shaders, generated API artifacts,
  examples, platform-host evidence, README migration prose, root adapters, or a
  version change. C14 owns platform evidence and final user-facing
  documentation; root owns facade/API artifacts and the gitlink.

## 3 Baseline And Impacts

| Surface | Published C12 baseline and C13 disposition |
| --- | --- |
| Semantic capabilities | `Capabilities::CURRENT` and final operation names already exist, but the three implemented GPU filter queries are false and `ColorFilteredImagePaint` still overclaims a retired materialized route. C13 sets only the S11/S30 final truths and keeps every broad boundary false. |
| Public statistics | `Stats` currently carries timings, scene counts, image cache counts, and uploaded bytes only. C13 adds `RenderRoute`, `EffectPrecision`, and the ten exact S14 fields, then reexports the two enums. |
| Execution telemetry | Private frame planning distinguishes direct and graph routes; runtime pass plans, working format, resource lifecycle counts, and retained bytes already exist. C13 derives immutable observations from those owners without adding a second execution model. |
| CPU/reference code | `reference.rs` is test-only and materialized helpers are currently gated by `#[cfg(test)]`; C13 seals that boundary with compile/source evidence and removes or relocates any remaining production-reachable phase. |
| Reconciliation | The exact three S30 inventory tests do not exist. C13 adds ordinary test-owned tables and probes; planning documents never become build inputs. |
| Public API | Additive `RenderRoute` and `EffectPrecision` reexports plus additive `Stats` fields; semantic capability values change only from stale false/true claims to the reviewed final contract. No aliases or deprecated phases. |
| Dependencies and artifacts | `Cargo.toml`, features, lockfile absence, shader inventory, Vello provenance, and root-owned generated artifacts remain unchanged. |
| Documentation and examples | No user-facing documentation or example change; C14 owns final migration/API/example evidence after this cutover is published. |
| MSRV and targets | Unchanged Rust 1.97.x and existing native/wasm target contract; no acquisition is authorized. |
| Safety | No owned unsafe code, unsafe attribute, extern block, or lint allowance/expectation. |

## 4 Global Invariants

- Preserve one transaction-owned drawing submission stage, caller-owned command
  encoding, one async scope/signal resolution boundary, and atomic public
  publication.
- Preserve the C12 working-pixel, filter-order, clamp, signed-spatial,
  sampling, resource-identity, last-use, and deterministic budget-trim
  contracts. Statistics observe those facts and cannot change them.
- Public `Stats` never contains a half-frame, provisional resource, cache
  mutation, failed acquisition, or uncommitted presentation. Saturating
  accumulation cannot influence rendering decisions.
- A direct frame cannot claim graph precision or custom passes. A graph frame
  cannot count semantic plans, empty/no-op passes, failed preparations, or
  resources that did not reach a successful lease acquisition.
- Semantic `Capabilities` remain device-independent. Runtime format/limit
  availability remains in `RuntimeCapabilities`; neither report guesses from
  backend names, Cargo features, or error strings.
- Every remaining `PrimitiveOperation` maps to exactly one family query or
  exact typed false boundary. Root and OracleOnly inventory rows are not
  represented as production render support.
- Production has no CPU selector, CPU shader/materialized-buffer path,
  blocking render helper, external `vello::` path, graph readback/re-entry,
  source-command replay, per-effect resource manager, or unchecked shader path.
- Public and private inputs are validated before device limits, allocation,
  cache mutation, encoding, submission, presentation, or publication.
- Source remains authoritative; no generator or planning document is consumed
  by the build, and no root-owned source or artifact changes in this cycle.
- Every implementation task follows the canonical worker, RED-GREEN-REFACTOR,
  exact-range task-review, and coordinator-acceptance gate before dependent
  work begins.

## 5 Ordered Tasks

### 5.1 T01 Publish GPU Truths Through The Primitive Inventory

- Area: `src/{capability.rs,error.rs,tests.rs}`.
- Outcome: encode S30B as a deterministic test-owned 101-row inventory and use
  its executable probes to make the three independently delivered GPU filter
  claims truthful.
- RED: after defining the reviewed 101 primary rows,
  `final_primitive_inventory_has_101_unique_capability_consistent_rows` fails
  concretely because `supports_gpu_color_filter_execution`,
  `supports_gpu_blur_filter_execution`, and
  `supports_gpu_drop_shadow_filter_execution` are false at the C12 base while
  their `FLT-03`-`FLT-10`, `FLT-02`, and `SHD-04` rows are Supported;
  `c13_semantic_capabilities_match_final_gpu_only_contract` fails at those same
  three exact assertions. No other production truth is changed in this task.
- Acceptance: exactly 101 unique primary IDs resolve to `69 Supported`,
  `24 Diagnostic`, `6 Root`, and `2 OracleOnly`; every supported primary probe
  succeeds, every diagnostic primary probe returns its exact typed boundary,
  Root exposes no render operation, and OracleOnly is absent from production
  capabilities/modules. The three GPU filter queries become true; all broad
  layer-filter, mask, backdrop, composite, and materialized filtered-image
  primary queries retain their reviewed values. Unknown, duplicate, and
  `FutureRender` dispositions are rejected by the test model; no plan parser or
  build input is added.
- Commands: run both named tests separately; run
  `affected_capability_queries_map_one_to_one_to_primitive_operations`,
  `capabilities_map_unsupported_primitives_to_typed_errors`,
  `runtime_capability_report_keeps_precision_flags_independent`, and
  `C13-CHECK`.
- Depends on: none.
- Intended commit: `feat(capability): publish gpu truths through final inventory`.

### 5.2 T02 Reconcile Properties And Typed Diagnostic Subcases

- Area: `src/{capability.rs,error.rs,tests.rs}`.
- Outcome: encode the exact S30A 22-property cross-reference and every S30C
  stable diagnostic subcase, then remove the remaining materialized
  color-filtered image overclaim exposed by those tables.
- RED: after defining the reviewed tables,
  `final_property_inventory_maps_22_surfaces_to_known_primitive_ids` and
  `final_diagnostic_subcase_inventory_maps_every_typed_boundary_once` fail
  concretely because `PNT-09.ColorFilteredImagePaint` is Diagnostic while the
  C12 base reports `supports_color_filtered_image_paint() == true` and accepts
  `PrimitiveOperation::ColorFilteredImagePaint`.
- Acceptance: color-filtered image paint becomes false with exact
  `UnsupportedPrimitive`; ordinary filtered-image paint remains false. Exactly
  22 unique property surfaces reference only known S30B IDs or stable S30C keys
  with matching parents, and no mixed row is blanket-supported when any mapped
  row is Diagnostic or Root. Every S30C key is unique and every final false
  semantic capability operation appears once; native WebCanvas, five
  unresolved-resource kinds, and both degraded-quality kinds are asserted
  separately. Unknown parents, unknown keys, and duplicates are rejected.
- Commands: run both named tests separately; run
  `unsupported_primitive_errors_name_operation`,
  `unresolved_resource_diagnostics_name_filter_resources`,
  `degraded_quality_diagnostics_name_reduced_intermediate_precision`, and
  `C13-CHECK`.
- Depends on: T01.
- Intended commit: `test(matrix): reconcile properties and diagnostics`.

### 5.3 T03 Publish Direct Route And Precision Statistics

- Area: `src/{stats.rs,lib.rs,renderer.rs,backend.rs,resource.rs,tests.rs}`.
- Outcome: add the exact S14 public types and fields and publish correct
  last-successful observations for default state and direct internal-Vello
  frames.
- RED: `stats_default_exposes_no_route_precision_or_pass_activity` fails
  concretely because `Stats` has no route, precision, or S14 pass/resource
  fields; `direct_vello_stats_report_exact_route_and_single_raster_pass` fails
  because a successful direct frame currently publishes no route or Vello-pass
  fact; `non_render_operations_do_not_mutate_last_successful_stats` cannot
  assert the reviewed new fields at the C12 base.
- Acceptance: `RenderRoute::{DirectVello,GpuGraph}` and
  `EffectPrecision::{High,Reduced}` have the exact reviewed traits and are
  reexported; `Stats` adds `route`, `effect_precision`, `vello_passes`,
  `image_passes`, `composite_passes`, `copy_operations`,
  `custom_present_passes`, `effect_texture_allocations`,
  `effect_texture_reuses`, and `retained_effect_bytes`; `Stats::default()` has
  no route/precision and zero new counters;
  one successful direct frame reports `DirectVello`, one Vello pass, no graph
  pass/copy/resource activity, and no effect precision. Capability queries,
  surface creation, resize, suspend/resume, and explicit readback do not mutate
  the last successful value.
- Commands: run all three named tests separately; run
  `render_reports_command_stats`, `failed_render_does_not_warm_image_reuse_stats`,
  and `C13-CHECK`.
- Depends on: T02.
- Intended commit: `feat(stats): expose final render route telemetry`.

### 5.4 T04 Publish Exact GPU Graph And Resource Statistics

- Area:
  `src/{frame.rs,pass.rs,resource.rs,backend.rs,gpu_transaction.rs,renderer.rs,stats.rs,tests.rs}`.
- Outcome: derive one immutable successful-frame observation from the selected
  working format, encoded runtime passes, lease acquisition sources, and
  post-trim resource manager state.
- RED: `gpu_graph_stats_count_exact_c12_passes_copies_resources_and_precision`
  fails concretely because graph frames inherit the T03 default graph counters
  and no precision; `resource_stats_report_acquisition_source_and_post_trim_retention`
  fails because successful resource-manager allocation/reuse deltas and retained
  bytes are not connected to public stats;
  `failed_and_canceled_graph_frames_preserve_last_successful_stats` fails until
  the new in-progress observation is staged behind the existing atomic
  publication boundary.
- Acceptance: actual Vello capture/direct passes, image passes, composites,
  copies, and custom present passes increment exactly per S14; identity/no-op
  work increments nothing; high/reduced precision reflects the selected
  working format. Successful allocation/reuse lease sources and retained idle
  bytes publish only after cleanup/trim. Every failure/cancellation preserves
  the preceding successful stats even when private cleanup or terminal
  diagnostics advance.
- Commands: run all three named tests separately; run
  `budget_zero_releases_idle_resources_without_changing_pixels`,
  `resource_budget_and_device_loss_preserve_public_stats_contract`,
  `terminal_signal_after_transaction_completion_preserves_public_frame_state`,
  and `C13-CHECK`.
- Depends on: T03.
- Intended commit: `feat(stats): publish exact gpu graph activity`.

### 5.5 T05 Consolidate Oracle Ownership And Close The Cutover

- Area:
  `src/{filter.rs,image.rs,reference.rs,renderer.rs,readback.rs,gpu_transaction.rs,vello_engine,tests.rs}`.
- Outcome: move the remaining CPU/materialized pixel-execution phases out of
  `image.rs` and `filter.rs` into the test-only reference owner, then prove the
  capability, stats, execution, failure, and three inventory surfaces form one
  final GPU-only contract.
- RED: `reference_module_exclusively_owns_cpu_pixel_execution` fails concretely
  because C12 `image.rs` still defines
  `ResolvedMaterializedImageFilterExecution`, straight/premultiplied CPU
  conversion, and materialized filter execution, while `filter.rs` still
  defines the materialized/compiled CPU pipeline phases;
  `image_and_filter_modules_have_no_materialized_executor_phases` fails on
  those exact source identities.
- Acceptance: `reference.rs` under `#[cfg(test)]` exclusively owns CPU pixel
  execution and its test-only compiled/oracle phases; `image.rs` owns validated
  images/readback buffers and `filter.rs` owns non-pixel planning/algorithm
  facts without a materialized executor phase. Existing oracle behavior remains
  deterministic. Non-test `Renderer::render` reaches neither reference code,
  readback/map/poll, rendered `Image::from_rgba`, graph replay, nor atlas
  re-entry; the caller-owned transaction remains the sole drawing submission
  authority. Integrated assertions prove the exact inventories agree with
  capabilities and successful/failed stats; no `FutureRender` or stale public
  phase remains.
- Commands: run both named RED tests separately; run
  `c13_gpu_only_cutover_reconciles_capabilities_stats_and_inventories`,
  `cpu_reference_algorithms_are_test_only_after_gpu_cutover`,
  `render_path_submits_without_map_or_cpu_wait`,
  `readback_static_paths_confine_map_poll_and_copy_submission`, every T01-T04
  named test, the completion guards, and `C13-CHECK`.
- Depends on: T04.
- Intended commit: `refactor(test): consolidate cpu oracle ownership`.

## 6 Verification

Implementation commands use the already-configured local Rust toolchain with
`CARGO_NET_OFFLINE=true`; acquisition is not authorized. C13 does not claim
formal Rust-version, feature-combination, wasm-target, presented-host, or
platform evidence; C14 owns those gates.

`C13-CHECK` is:

```sh
set -euo pipefail
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
test -z "$(git ls-files -- Cargo.lock)"
```

Structural Clippy advisories such as `too_many_lines` do not block C13; the
required post-initiative sprawl review evaluates those signals against cohesion
and whole-module structure.

The owned-Rust unsafe scan uses the canonical workflow manifest and pattern.
Dependency-feature, target, host, and provenance execution evidence is deferred
to C14; C13 only proves that its range leaves `Cargo.toml` and the published
dependency/artifact sources unchanged.

Completion guards require:

- `Cargo.toml`, features, dependency roles, Vello provenance/license artifacts,
  and the complete C12 shader inventory remain unchanged; every shader has one
  canonical include owner;
- the only `src/lib.rs` name additions are `RenderRoute` and
  `EffectPrecision`, with no internal graph/resource/shader/backend reexport;
- all three implemented GPU filter capability queries are true, both
  filtered-image materialization claims are false, and every broad diagnostic
  boundary remains false with its exact typed result;
- public stats match actual direct/graph work, selected precision, successful
  resource-acquisition source, and post-trim retention, while all failed and
  non-render operations preserve the last successful value;
- non-test builds contain no reference import, CPU/materialized executor,
  external `vello::`, blocking render helper, graph readback/re-entry, cloned
  source commands, per-effect manager, or unchecked shader path;
- no new production `queue.submit`, `map_async`, `Device::poll`,
  `register_texture`, rendered `Image::from_rgba`, graph replay, unsafe, or lint
  allowance/expectation is added by the range;
- the deterministic inventories have exact `101 = 69 + 24 + 6 + 2` and
  22-property totals, every S30C boundary appears exactly once, and there is no
  unknown, duplicate, blanket mixed-row support, or `FutureRender` disposition;
- all T01-T05 named evidence and the default-feature `C13-CHECK` pass with the
  exact prior test-name inventory preserved except for named C13 additions.

## 7 Completion

- All five ordered task ranges have fresh `CLEAN` task reviews and coordinator
  acceptance against their exact ordered commit spans.
- A separate status-only commit changes this plan from `in_progress` to
  `complete`; the exact cycle range then passes `C13-CHECK`, every named C13
  test, completion guards, and a clean-worktree check.
- A fresh holistic reviewer returns `CLEAN` for the exact cycle range, and the
  complete final command set passes again on the exact reviewed head.
- The immutable reviewed head is published through the canonical landing gate;
  fresh remote readback proves local `main`, its authority tracking ref, and
  observed remote `main` agree.
- Handoff: C14 receives the final GPU-only production architecture, truthful
  public capability and statistics surfaces, exact reconciled inventories, and
  unchanged feature/dependency/platform/root ownership boundaries.
- After C13 closure, the coordinator retains an explicit C14 planning packet
  that names the published C13 head, the two added public enums and S14 fields,
  the exact `101/22/S30C` inventory evidence, and the still-unclaimed
  platform/docs/example/root work. This handoff is planning evidence, not a Rust
  test or build input.
- Genuine blockers are limited to unowned conflicting state, a reviewed-source
  contradiction, unavailable required native GPU execution, required unsafe,
  unavailable Surgeist profiles, or publication credentials/remote history
  failure.
