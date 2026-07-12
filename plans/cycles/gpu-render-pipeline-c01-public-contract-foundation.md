# GPU Render Pipeline C01 Public Contract Foundation

## Header

- Cycle: `C01`.
- Owner: `surgeist-render`.
- Status: `complete`.
- Review disposition: the user explicitly waived the remaining cycle-plan review after the final returned finding was limited to the corrected S32/S36/S37 header inventory.
- Cycle base: `d59ad253300b68311f4e81a70e2b6ce73c922a4d`.
- Specification: `plans/specs/gpu-render-pipeline.md` at commit `3826a9098e859874a515bbebaf470a47d754d76c`, content SHA-256 `f01972e19f8a5ddc90936edfc6ea7955feff3d1b1fdf5d181e77e8d10cc1f60a`, sections S07-S08, S10, S12, non-readback S13, safe-resize S26, applicable S29 rows, S35, C01-named model tests from S32, C01 dependency/feature/docs/MSRV limits from S36, and the available native command subset from S37. Its user-authorized review disposition is recorded by the sequence.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at commit `ff0d4a3c478f6f89cceab3962883bd53396cba6b`, content SHA-256 `1ff25dbd5bb0382e2e66573affd058a3e3f939bede561a896ce6e1dea7d73840`, entry `C01 Public Contract Foundation`.
- Outcome: publish the options, text-bounds, error, and runtime-capability model foundations; force Vello GPU execution; and remove the backend-specific unsafe resize hint without changing the current semantic capability, statistics, or pixel-execution route.

## Boundary

- Current `Options` exposes fields and forwards `use_cpu`; C01 replaces it with exact S08 private state and always passes `use_cpu: false` to Vello.
- `apply_presented_resize_state` contains the only owned `unsafe` block and Metal `as_hal` access. C01 removes that side effect, preserves modeled resize-state changes, and makes the crate forbid owned unsafe code.
- `TextRun` has no authored ink-bound state. C01 adds the exact S10 model, constructor argument, accessors, validation, and direct-run preservation.
- `Error` exposes mutable representation and one native-only source bound. C01 makes representation private, separates backend and semantic construction, adds target-correct source storage, and publishes the shared runtime diagnostic vocabulary and validation table.
- C01 publishes the exact S12 report value types and accessors only. C02 owns `Renderer::runtime_capabilities`, selected-surface identity, safe adapter/device projection, and terminal device-state observation as one truthful operation.
- `RuntimeOperation::SurfaceReadback` and its reason pairings may enter as shared model variants, but `ErrorCode::ReadbackFailed`, async readback progress, and readback behavior remain C03. Existing lifecycle/backend codes needed by unmigrated C02/C03 paths remain only until those owning paths are replaced; they are not compatibility shims.
- C01 does not make renderer entry points async, add identity/device terminality/draft publication, alter readback, create frame or executable plans, add GPU graph resources/passes, or remove current materialized/CPU-reference routes.
- `Capabilities::VELLO_0_9`, its family accessors and values, and public `Stats` remain behaviorally unchanged. C11 owns their cutover.
- Work stays in this repository. No root/sibling edit, compatibility shim, dependency acquisition, new dependency, production CPU fallback, or generated artifact is authorized.
- The canonical gate applies without a local commit override: each worker records RED evidence, implements, runs acceptance checks, and creates the intended logical commit; a separate reviewer then inspects that exact task range before the coordinator advances.

## Impacts

- Public API: breaking private-field changes to `Options` and `Error`, breaking `TextRun::try_new`, removal of the CPU selector and obsolete degraded-quality variants, plus additive S08/S10/S12/S13 types, accessors, builders, and reexports. No deprecated aliases, forwarding shims, or C01 runtime query.
- Dependencies/features: declarations remain unchanged; native default, `render-window`, `render-web`, and combined feature states use offline checks.
- Generated artifacts: none.
- Docs/examples: changed public items receive contract rustdocs; README and native presented example remain C12-owned.
- MSRV: use only Rust 1.89-compatible APIs. The absent 1.89 toolchain and wasm target retain their actual C12 gates.
- Root follow-up: none; publication hands the reviewed crate SHA to C02, not root integration.
- Unsafe: remove the owned block, add crate-level `forbid(unsafe_code)`, and require compiler plus repository-wide absence evidence.

## Tasks

Every task runs its focused RED/acceptance command followed by the exact `C01-CHECK` matrix before review:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
```

### 1. Remove The Unsafe Presented Resize Hint

- Files/area: `src/lib.rs`, `src/backend.rs`, `src/renderer.rs`, and presented lifecycle tests in `src/tests.rs`.
- Outcome: remove `as_hal`, Metal-layer mutation, and `unsafe`; retain only the safe S26 resize-state hint; add `#![forbid(unsafe_code)]`.
- RED evidence: before implementation, `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings` fails at the existing resize block.
- Acceptance: `set_surface_resizing` retains idempotent modeled transitions, no backend/native-handle side effect remains, and the owned Rust manifest contains no executable unsafe construct or lint escape.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_surface_lifecycle_state_names_pending_resize`; exact `C01-CHECK`; `git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{'` (no match, exit 1, is clean).
- Dependencies: none. Intended worker commit: `Remove unsafe resize hint`.

### 2. Publish GPU-Only Renderer Options

- Files/area: `src/renderer.rs`, Vello option construction in `src/backend.rs`, `src/lib.rs`, and focused `src/tests.rs` coverage.
- Outcome: implement exact S08 `Options`, `EffectQualityPolicy`, and `ResourceCacheBudget` APIs/defaults; remove `use_cpu`; route antialiasing/debug through accessors; and make one tested Vello-options path always select `use_cpu: false`.
- RED evidence: first add `options_default_requires_high_precision_and_bounds_retention`, `resource_cache_budget_zero_disables_idle_retention`, and `vello_renderer_options_force_gpu_execution`; new API assertions or compilation fail before implementation.
- Acceptance: builders preserve unrelated fields, zero budget is valid, `Renderer::options()` preserves configuration, and no public or private CPU selector remains.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render options_default_requires_high_precision_and_bounds_retention`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render resource_cache_budget_zero_disables_idle_retention`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render vello_renderer_options_force_gpu_execution`; exact `C01-CHECK`.
- Dependencies: task 1. Intended worker commit: `Implement GPU-only renderer options`.

### 3. Add Authored Text Run Bounds

- Files/area: `src/text.rs`, `src/error.rs` unresolved-resource kind, `src/lib.rs`, and all `TextRun::try_new` callers/tests.
- Outcome: implement exact S10 private three-state `TextRunBounds`, positive finite ink validation, appended constructor argument, `bounds()`, `TextShadowRun` preservation, and `TextRunInkBounds`.
- RED evidence: first add `text_run_bounds_distinguish_unspecified_empty_and_ink`; its missing types and constructor signature fail before implementation.
- Acceptance: public construction cannot fabricate ink payloads, invalid/zero-area ink is a typed invalid value, direct text uses explicit `unspecified()`, and no glyph-advance estimate exists.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render text_run_bounds_distinguish_unspecified_empty_and_ink`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render text_run`; exact `C01-CHECK`.
- Dependencies: task 2. Intended worker commit: `Add authored text run bounds`.

### 4. Make Error Ownership And Sources Explicit

- Files/area: `src/error.rs`, all production error construction/context sites, `src/lib.rs`, and accessor migrations in `src/tests.rs`.
- Outcome: privatize `Error`, add exact observation methods, constrain generic backend construction to a private backend-code type, use target-specific source storage, correct semantic unsupported errors to `ErrorCode::UnsupportedPrimitive`, and finalize non-runtime degraded-quality variants.
- RED evidence: first add `semantic_error_accessors_preserve_payloads` and `native_and_wasm_error_source_storage_preserves_source_contract`; accessors, target storage, or native `Send + Sync` assertions fail before implementation.
- Acceptance: only `src/error.rs` writes wrapper fields; backend constructors cannot accept semantic typed codes; safe sources remain observable; native `Error` is `Send + Sync`; tests use accessors; public raw constructors/fields are absent.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render semantic_error_accessors_preserve_payloads`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render native_and_wasm_error_source_storage_preserves_source_contract`; exact `C01-CHECK`.
- Dependencies: task 3. Intended worker commit: `Privatize render errors`.

### 5. Publish Validated Runtime Diagnostics

- Files/area: runtime diagnostic models/validation in `src/error.rs`, reexports in `src/lib.rs`, and exhaustive model tests in `src/tests.rs`.
- Outcome: add complete S13 operation/reason/component enums, private `RuntimeCapabilityUnavailable`, validated internal construction, public accessors/semantic `Error` constructor, and matching typed error-code invariant without changing C02/C03 operation flow.
- RED evidence: first add `runtime_errors_distinguish_semantic_unsupported_from_device_unavailable`, `runtime_diagnostic_constructor_rejects_every_unlisted_operation_reason_pair`, and `typed_error_codes_cannot_exist_without_their_matching_payload`; missing models/pair validation fail before implementation.
- Acceptance: all and only S13 pairs construct; invalid pairs return typed `InvalidValue`; public callers cannot construct the diagnostic directly; payload/code pairing is exact; readback execution/code changes remain absent.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_errors_distinguish_semantic_unsupported_from_device_unavailable`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_diagnostic_constructor_rejects_every_unlisted_operation_reason_pair`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render typed_error_codes_cannot_exist_without_their_matching_payload`; exact `C01-CHECK`.
- Dependencies: task 4. Intended worker commit: `Add validated runtime diagnostics`.

### 6. Publish Runtime Capability Report Models

- Files/area: report value types and private validated construction in `src/capability.rs`, reexports in `src/lib.rs`, and model tests in `src/tests.rs`.
- Outcome: implement exact S12 report traits, private fields, and public accessors while leaving selected-surface querying and adapter/device projection wholly to C02.
- RED evidence: first add `runtime_capability_report_keeps_precision_flags_independent`; missing report types/accessors and independent flags fail before implementation.
- Acceptance: available/unavailable projections and both independent precision flags are representable, both flags false is valid, fields are private, semantic `Capabilities`/`Stats` remain unchanged, and no `Renderer::runtime_capabilities` or backend query path enters C01.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render runtime_capability_report_keeps_precision_flags_independent`; exact `C01-CHECK`.
- Dependencies: task 5. Intended worker commit: `Add runtime capability report models`.

## Completion

- Acceptance: all six task ranges and reviews are clean; cited public models match the specification; Vello cannot select CPU execution; owned source is unsafe-free; current semantic capabilities, statistics, and rendering route remain truthful; no runtime query or later-cycle behavior/artifact entered the range.
- Final commands: `CARGO_NET_OFFLINE=true cargo check -p surgeist-render`; exact `C01-CHECK`; and the task 1 unsafe scan with no executable match.
- Landing: follow canonical task, status, holistic-review, publication, and remote-readback gates. Publish reviewed C01 on authority `origin/main` and hand its exact SHA plus evidence to C02; root receives no C01 handoff.
- Blockers: return only an unprovided design decision, forbidden dependency/acquisition need, unowned repository change, unavailable required native GPU evidence, or material packet contradiction. Missing wasm or Rust 1.89 tooling is recorded for C12 and does not block C01.
