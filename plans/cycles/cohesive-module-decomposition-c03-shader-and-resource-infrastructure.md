# P02-I02-S01-C03 Shader And Resource Infrastructure

## 1 Header

- Cycle: `P02/I02/S01/C03`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base: `2229779872ba35ac905867c69eae6db146d11bcf`, the published
  C02 candidate verified on local and authority-remote `main`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `3365399fa2411efd5cd8fcfdfe74d4b756cd6a79`, SHA-256
  `892c5c1c2162a07c83bc124c3687e05ff62c6906930c154f25117792ec63d035`;
  sections M01-M04, M05.4, and M06-M09.
- Sequence: `plans/sequences/cohesive-module-decomposition.md` at
  `d5ac5c23b3c66d3fa451bed6b751f1c82275b5d1`, SHA-256
  `a45655293a54b5bf0986d508d6d2b278af68e5267dc373d8f5d56cb84e58c74b`;
  entry `C03 Shader And Resource Infrastructure`.
- Outcome: replace `src/{shader,resource}.rs` with the explicit private
  hierarchies required by M05.4 while preserving serialization, identity,
  validation, realization, Gaussian, allocation, accounting, leasing, cleanup,
  and retention behavior and every existing crate-visible contract.

## 2 Boundary

- Owned input: `src/shader.rs`, `src/resource.rs`, and only the import and
  visibility repairs required by faithful relocation of their existing items.
- Required shader output:
  `src/shader/{mod,parameters,key,validate,pipeline,cache,test_support}.rs`.
  `shader/mod.rs` retains the crate-visible front door and explicit reexports;
  children own exactly the M05.4 shader responsibilities.
- Required resource output:
  `src/resource/{mod,gaussian,manager,lease,test_support}.rs`.
  `resource/mod.rs` retains `WorkingFormat`, `ResourceManager`, and genuine
  resource-front-door coordination. `manager.rs` owns identities, keys,
  entries, preflight, and manager state. `lease.rs` owns acquisition scopes,
  leases, cleanup, retention, accounting outcomes, and state operations that
  consume those contracts. This keeps the child direction
  `gaussian -> manager -> lease`; it does not create `manager <-> lease`.
- Existing shader/pass, resource/backend, resource/renderer, texture/resource,
  transaction/resource, and Vello-resource edges remain unchanged. Imports name
  the owning front door or child explicitly and no new module-directory mutual
  edge is introduced.
- No public API, behavior, diagnostic, validation-precedence, byte layout,
  identity, cache-publication, allocation, lifecycle, accounting, retention,
  test operation, or test oracle change is allowed.
- No semantic rename, compatibility shim, forwarding-only layer, copied
  definition, `include!`, `#[path]`, generated concatenation, glob-reexport
  maze, generic helper module, source parser, inventory test, or numerical
  size/count gate is allowed.
- Root, sibling, adapter, API-artifact, gitlink, hierarchical-public-front-door,
  dependency, feature, target, example, manifest, and correctness-fix work is
  excluded. Commands use installed artifacts offline; acquisition, installation,
  bootstrap, and update remain unauthorized.

## 3 Impacts

- Public API and caller migration: none; `src/lib.rs` and public paths remain
  unchanged. Crate-visible shader/resource paths remain available through
  explicit front-door reexports.
- Behavior and diagnostics: unchanged.
- Dependencies, features, targets, MSRV, docs, and examples: unchanged.
- Test impact: import/module relocation only; test names, operations, inputs,
  assertions, and oracles remain unchanged.
- Generated artifacts: none in this leaf; root-owned artifacts remain untouched.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Establish Shader Parameter And Key Owners

- Convert `src/shader.rs` to `src/shader/mod.rs` without changing the crate
  module name. Move every GPU parameter byte model, checked scalar narrowing,
  byte-length calculation, serialization routine, and vector fact to
  `parameters.rs`. Move shader program, format, binding, sampler, mask, layout,
  module, and render-pipeline key models to `key.rs`.
- Leave validation, WGPU construction, cache realization, provisional updates,
  and test support in `mod.rs` until their ordered tasks. Use explicit imports
  and the narrowest visibility; do not duplicate definitions to bridge the move.
- This is a behavior-preserving refactor, so fabricated RED is not applicable.
  Before editing and after the move, run and record these characterizations:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_operation_bytes_preserve_tags_scalars_and_clamp_boundaries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_operation_buffer_limits_return_exact_invalid_input_before_allocation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render drop_shadow_parameter_bytes_preserve_fractional_offset_and_solid_color
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composite_parameter_bytes_preserve_affine_mask_mapping_quality_and_extend
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render pass_spatial_uniform_bytes_match_the_exact_little_endian_layout_without_pod
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_lowering_derives_exact_sampler_layout_shader_and_pipeline_keys
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render mask_pipeline_keys_exclude_image_identity
  ```

- Acceptance: `src/shader/{mod,parameters,key}.rs` exist; each assigned model has
  one definition in its owner; common structural/default/combined commands from
  section 5 pass; public, manifest, docs, and example surfaces are unchanged.
- Intended commit: one complete shader parameter/key ownership point.

### 4.2 T02 Move Shader Validation And Pipeline Construction

- Start only from the reviewed T01 head. Move pass-key compatibility,
  consistency, semantic pass descriptions, and validation precedence to
  `validate.rs`. Move WGSL selection and WGPU sampler, bind-group-layout,
  pipeline-layout, shader-module, and render-pipeline construction to
  `pipeline.rs`.
- Preserve phase direction: keys and parameter facts feed validation; validated
  descriptions feed construction; neither validation nor construction depends
  on cache state. Preserve exact WGPU labels, descriptors, entry points, blend
  states, binding order, and typed errors.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_layouts_bind_only_sampled_resources_and_exact_spatial_uniforms
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composite_layouts_bind_no_dummy_parent_clip_or_mask
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_layout_binds_exact_source_spatial_and_operations
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render copy_backdrop_layout_binds_parent_and_spatial_mapping
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render blur_layout_binds_exact_source_spatial_and_kernel
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render drop_shadow_layout_binds_blurred_alpha_spatial_and_parameters
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render graph_preparation_rejects_unsupported_passes_without_resource_or_cache_mutation
  ```

- Acceptance: `src/shader/{validate,pipeline}.rs` exist; validation and pipeline
  responsibilities each have one owner; child imports follow parameter/key to
  validation to pipeline direction; common commands pass with unchanged tests.
- Intended commit: one complete shader validation/pipeline ownership point.

### 4.3 T03 Move Shader Cache Support And Reconcile Its Front Door

- Start only from the reviewed T02 head. Move committed/provisional device pass
  cache state, realization, cache commit/rollback readiness, and provisional pass
  objects to `cache.rs`. Move shader/cache observations, vector facts, deliberate
  invalid-fragment ingress, and key-space probes to test-only `test_support.rs`.
- Reconcile `shader/mod.rs` to child declarations and explicit current-contract
  reexports only. Cache may consume parameter/key/validation/pipeline owners;
  those children may not depend on cache or test support.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render device_pass_cache_starts_with_no_realized_entries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render base_graph_shader_cache_realizes_checked_programs_without_publishing_failed_entries
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render composite_cache_realizes_exact_normal_and_destination_sampling_programs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_cache_realizes_checked_high_and_reduced_programs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render copy_backdrop_cache_realizes_checked_working_format_programs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render blur_cache_realizes_checked_axis_input_and_precision_programs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render drop_shadow_cache_realizes_checked_colorize_and_merge_programs
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render color_filter_shader_failure_preserves_prior_publication_and_cache
  ```

- Acceptance: all seven shader files exist; `test_support.rs` is test-only;
  `mod.rs` retains no responsibility assigned to a child; cache publication and
  failure atomicity remain characterized; common commands pass.
- Intended commit: one shader cache/test-support/front-door reconciliation point.

### 4.4 T04 Establish Gaussian And Resource Manager Owners

- Start only from the reviewed T03 head. Convert `src/resource.rs` to
  `src/resource/mod.rs`. Move Gaussian keys, limits, normalized weights,
  sample-count planning, packing, serialization, and upload validation to
  `gaussian.rs`. Move manager/frame/resource identities, cache keys, payloads,
  entries, allocation preflight, manager state, and state operations that do not
  consume lease/cleanup contracts to `manager.rs`.
- Retain `WorkingFormat`, `ResourceManager`, and still-inseparable lease/scope and
  cleanup coordination in `mod.rs` until T05. Do not make `manager.rs` depend on
  `lease.rs`; widen fields or state methods only to the narrowest `pub(super)`
  access required by the final direction.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gaussian_kernel_bytes_are_symmetric_normalized_and_exactly_cached
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render gaussian_kernel_buffer_keys_include_the_exact_plan
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_role_keys_keep_allocation_namespaces_distinct
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render extreme_effect_extent_reports_device_dimension_before_descriptor_byte_overflow
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render retained_byte_overflow_preflights_all_concrete_payload_creation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_preparation_is_allocation_safe_and_submission_free
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_manager_observation_reports_checked_entry_total_overflow
  ```

- Acceptance: `src/resource/{mod,gaussian,manager}.rs` exist; Gaussian and manager
  model/state responsibilities have one owner; no manager-to-lease edge exists;
  common commands pass with exact allocation-preflight behavior preserved.
- Intended commit: one Gaussian/manager ownership point.

### 4.5 T05 Move Resource Lease Support And Reconcile Its Front Door

- Start only from the reviewed T04 head. Move frame acquisition scopes, resource
  leases/tokens, acquisition and resolution operations, cleanup dispositions,
  retention/accounting outcomes, and their manager-state operations to
  `lease.rs`. Move resource observations, injected test tokens/fault controls,
  and test-only accessors to test-only `test_support.rs`.
- Reconcile `resource/mod.rs` to `WorkingFormat`, `ResourceManager`, explicit
  current-contract reexports, and only genuine manager/scope coordination. Keep
  the final child direction `gaussian -> manager -> lease`; do not use a shim,
  callback, trait, copied state, or generic helper to conceal a cycle.
- Before and after, run:

  ```sh
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_leases_reject_stale_generation_and_double_release_by_model
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_trim_order_is_last_used_then_resource_identity
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_frame_scope_cleanup_covers_success_error_and_cancellation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render discard_accounting_mismatch_faults_resource_manager_without_clamping
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render per_lease_discard_detects_accounting_fault_and_returns_bounded_error
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render replace_rejects_existing_accounting_fault_before_mutation
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render texture_cache_release_and_eviction_accounting_is_deterministic
  CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_frame_returns_all_leases_and_preserves_last_successful_stats
  ```

- Acceptance: all five resource files exist; `test_support.rs` is test-only;
  `mod.rs` contains no child-owned implementation; every prior crate-visible
  path remains explicit; accounting and cleanup precedence remain unchanged;
  common commands and the full C03 matrix pass.
- Intended commit: one resource lease/test-support/front-door reconciliation point.

## 5 Verification And Completion

Each task records the required passing pre-move characterization and identical
post-move operation/oracle result; source and file checks are structural evidence
only. Each task requires a separate task-review `CLEAN` verdict. After all tasks
are clean, the coordinator makes a status-only `complete` commit, runs this
matrix, obtains a distinct holistic `CLEAN` review over the exact cycle range,
repeats the matrix at the unchanged reviewed head, and publishes with
authority-remote readback:

```sh
set -euo pipefail
test ! -e src/shader.rs
test ! -e src/resource.rs
for required in \
  src/shader/mod.rs src/shader/parameters.rs src/shader/key.rs \
  src/shader/validate.rs src/shader/pipeline.rs src/shader/cache.rs \
  src/shader/test_support.rs src/resource/mod.rs src/resource/gaussian.rs \
  src/resource/manager.rs src/resource/lease.rs \
  src/resource/test_support.rs; do
  test -f "$required"
done
test -z "$(rg -n 'include!|#\s*\[\s*path\s*=' src/shader src/resource || true)"
test -z "$(git diff 2229779872ba35ac905867c69eae6db146d11bcf -- src/lib.rs Cargo.toml README.md examples)"
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
git diff --check 2229779872ba35ac905867c69eae6db146d11bcf..HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse main)"
test -z "$(git status --porcelain)"
```

The live smoke executables must render and exit on the native host. Every
unsafe-scan match is classified; any executable match blocks completion. The
publication head is immutable after holistic review. Root integration remains
excluded.

The C03-to-C04 leaf handoff reports the immutable published C03 candidate and
authority-remote readback SHA, the exact reviewed planning revision, clean task
and holistic verdicts, and the stable shader/cache and resource-management front
doors consumed by pass-model work. It confirms clean status and preserves the
explicit exclusion of root integration.
