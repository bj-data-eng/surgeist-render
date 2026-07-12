# GPU Render Pipeline C02 Foundation Adoption And Cleanup

## Header

- Cycle: `C02`.
- Owner: `surgeist-render`.
- Status: `in_progress`.
- Cycle base: `5361e3460278dffb877b9d485a2d12977977c3ef`.
- Specification: `plans/specs/gpu-render-pipeline.md` at commit
  `1e6517e4e33669d97b1f45c0df9c1de78ec4d07e`, normalized SHA-256
  `db78f70e03a31430e949ac06de6628ca24a03cd53cf5dec453b43bcf4fbe53be`,
  sections S06B, foundation portions of S07-S13A, identity/terminal portions of
  S25-S26, applicable S29 and S31-S32 rows, and only S35's owned-unsafe and
  C02 production-blocking guards.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at commit
  `b46c9c2afb6f705fdaf928d640b3821a8e29c0c9`, normalized SHA-256
  `3dab5afdeb5084026f4863a3f0f4dfa18de47441a2560e1f5cbd1562732d8bdf`,
  entry `C02 Foundation Adoption And Cleanup`.
- Outcome: audit and correct the provisional post-C01 identity, async API,
  terminal-device, capability, error-scope, and transaction foundation; remove
  the rejected external-Vello presented-setup/no-op seam without starting C03.

## Boundary

- Published C01 is the sole base. Existing implementation commits are
  provisional task-source spans, not accepted results; they are not rebased,
  squashed, reordered, or rewritten.
- Each task reconstructs the exact ephemeral probe below in a cycle-owned
  detached worktree at its pre-span base, then discards the clean probe through
  the canonical resource ledger. Probe files never enter candidate history.
- An existing source span begins its task's ordered range. A worker appends a
  correction commit only when acceptance requires a source/test change; an
  empty or bookkeeping commit is forbidden. The separate reviewer inspects the
  full ordered range either way.
- External `vello = 0.9.0` remains the unchanged temporary production raster
  implementation. C02 adds no Vello ownership and makes no internal-engine,
  selected-glyph, provenance, dependency, or manifest change.
- Headless draft publication, presented transactional publication, remaining
  lifecycle completion, readback, frame/graph planning, custom passes, final
  reports, and CPU/materialization removal remain C03 and later sequence work.
- No root/sibling edit, compatibility shim, generated artifact, acquisition,
  production CPU fallback, blocking executor, or owned `unsafe` is permitted.

## Impacts

- Public API: adopt the provisional async methods, capability query, identity diagnostics, and terminal reports; add nothing else.
- Dependencies/features: unchanged; default, `render-window`, `render-web`, and
  combined native states remain supported.
- Generated artifacts: none.
- Docs/examples: only changed public rustdocs; README and presented example remain C14-owned.
- MSRV: Rust 1.97; current `+stable` must report 1.97.x for compatibility checks.
- Root follow-up: none; the remotely verified C02 SHA becomes C03's base.
- Unsafe: `forbid(unsafe_code)`, Clippy `-F unsafe-code`, and the owned-source scan remain mandatory.

## Tasks

Every task runs its focused command and this exact `C02-CHECK` matrix before
review:

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
rustc +stable --version
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets --features render-window,render-web
```

The base-compatible RED probes are exact:

| Task | Pre-span base and ephemeral test-only files | RED command and expected failure |
| --- | --- | --- |
| 1 | `2b6d4ad4e4af20ba6e11df9bb099c9bfad2bfc6c`; add public integration test `tests/c02_probe_identity.rs` with local `assert_foreign_identity` only | `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --test c02_probe_identity -- --exact foreign_surface_reports_identity_before_backend`; assertion receives a non-identity/backend result, not `ForeignRenderer` |
| 2 | `7e2a31e012d00c25dd4ab8c1c795186b3a4f269b`; add `tests/c02_probe_async.rs` with local generic `requires_future` only | `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --test c02_probe_async`; compile fails because `create_headless` returns `Result<Surface>`, not `Future<Output = Result<Surface>>` |
| 3 | `7b638c624d1ad9a1677ab5a4f56b4cf592b4d9aa`; add only cfg-test `Renderer::destroy_default_device_for_c02_probe` in `src/renderer.rs`, which destroys the existing default WGPU device then performs one bounded test-only `Device::poll(Wait)` progress call, plus `c02_probe_destroyed_device_reports_typed_loss` in `src/tests.rs` | `CARGO_NET_OFFLINE=true cargo test -p surgeist-render c02_probe_destroyed_device_reports_typed_loss`; the next headless operation does not return typed `DeviceLost` |
| 4 | `ed3355f54411f4fcc15b617c516680867c6b6473`; add only cfg-test `Renderer::submit_invalid_copy_for_c02_probe` in `src/renderer.rs` and `c02_probe_validation_is_captured_without_fault` in `src/tests.rs`, using pre-span WGPU handles and no new scope | `CARGO_NET_OFFLINE=true cargo test -p surgeist-render c02_probe_validation_is_captured_without_fault`; validation escapes/panics or faults the device instead of returning captured `RenderFailed` while the slot remains ready |

### 1. Adopt Renderer Surface And Device Identity

- Existing spans: `2b6d4ad4e4af20ba6e11df9bb099c9bfad2bfc6c..64bc5cb1ee3d7e9747120a8c02c2d6205f0c1400`
  and `6621c6264fb95c9ea804fc92e259987d10f379bd..7e2a31e012d00c25dd4ab8c1c795186b3a4f269b`.
- Files/area: `src/backend.rs`, `src/renderer.rs`, `src/surface.rs`,
  `src/error.rs`, and identity tests in `src/tests.rs`.
- Outcome: make renderer identity and generation-bearing device-slot identity
  authoritative before every slot access or WGPU/Vello call.
- RED: run task 1's public probe at its exact base; only the stated non-identity
  result is acceptable. In a second probe at `6621c62`, append only the
  seven-line incompatible-stale-resume case to `src/tests.rs`; the focused
  identity test must return stale identity instead of required
  `SurfaceCreateFailed` before the `7e2a31e` ordering correction.
- Acceptance: contract-only surfaces name no device; all render/read/resume/
  capability paths validate renderer, kind, slot, and generation in S26 order;
  foreign/stale failures make no backend call.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  foreign_and_stale_surfaces_fail_before_device_slot_access`; `C02-CHECK`.
- Dependencies: none. Intended logical point: `Adopt renderer and device identity`.

### 2. Adopt The Async Non-Readback Front Door

- Existing span: `7e2a31e012d00c25dd4ab8c1c795186b3a4f269b..7b638c624d1ad9a1677ab5a4f56b4cf592b4d9aa`.
- Files/area: public methods/helpers in `src/renderer.rs` and all crate-owned
  callers/tests in `src/tests.rs`.
- Outcome: retain exact async create-surface, create-headless, render, and
  resume signatures while leaving synchronous readback unchanged.
- RED: run task 2's public compile probe at its exact base; only the stated
  `Result<Surface>: Future` trait failure is acceptable.
- Acceptance: production contains no `pollster::block_on` or Vello blocking
  executor; futures promise no `Send`; validation/error order is unchanged.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  non_readback_renderer_front_door_is_async`; `rg -n
  'pollster::block_on|block_on_wgpu' src --glob '!tests.rs'` (exit 1 is clean);
  `C02-CHECK`.
- Dependencies: task 1. Intended logical point: `Adopt async renderer operations`.

### 3. Adopt Terminal Device Signals And Runtime Reports

- Existing span: `7b638c624d1ad9a1677ab5a4f56b4cf592b4d9aa..ed3355f54411f4fcc15b617c516680867c6b6473`.
- Files/area: device/callback state in `src/backend.rs`, projections in
  `src/renderer.rs`, diagnostics in `src/error.rs`, and focused `src/tests.rs`.
- Outcome: retain immutable selected-device capabilities and first-record-wins
  per-generation loss/fault state with slot-local terminal cleanup.
- RED: run task 3's destroy-and-bounded-poll probe at its exact base; only the
  stated missing typed `DeviceLost` result is acceptable RED.
- Acceptance: callbacks record portable typed facts, poison recovery retains the
  first record, terminal observation drops only that slot, ready slots continue,
  and construction races cannot publish a renderer after terminal signal. Expand
  `destroyed_device_callback_reports_terminal_loss_without_stale_resource_use`
  to wait at most five seconds, assert two same-reason post-destroy failures,
  renderer cleanup, and preservation of another ready slot.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  device_loss_is_terminal_idempotent_and_releases_device_resources`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  runtime_capabilities_project_the_selected_surface_without_gpu_work`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  terminal_default_device_rejects_headless_without_disabling_ready_slots`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  terminal_signal_during_renderer_creation_aborts_before_followup_gpu_work`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  destroyed_device_callback_reports_terminal_loss_without_stale_resource_use`;
  `C02-CHECK`.
- Dependencies: task 2. Intended logical point: `Adopt terminal device state`.

### 4. Adopt Scoped Transactions And Remove Rejected Setup Seam

- Existing span: `ed3355f54411f4fcc15b617c516680867c6b6473..9aa1d97c30a75d0bd552d6b07b62d8d87a7bd39b`.
- Files/area: `src/gpu_transaction.rs`, current transaction use in
  `src/backend.rs`, `src/renderer.rs`, `src/shader.rs`, diagnostics, and tests.
- Outcome: retain backend-neutral generations, nested scopes, typed stage
  mapping, cancellation cleanup, and leases; forward-remove
  `PresentedSetupStep`, `PRESENTED_SETUP_STEPS`, `orchestrate_presented_setup`,
  `presented_setup_transaction_stages_for_test`,
  `presented_setup_assigns_each_device_owned_step_to_a_transaction_stage`, and
  sole-use `GpuOperationStage::SurfaceConfigure` plumbing/classification.
- RED: run task 4's pre-span probe; only the stated escaped/terminal validation
  failure is acceptable RED. The prior clean-context finding separately proves
  the setup-stage test is a no-op seam to remove rather than bless.
- Acceptance: captured errors do not fault a device; active/unattributed
  uncaptured errors do; loss wins terminal precedence; cancellation clears the
  generation and leases; retained draw/probe calls remain scoped; no rejected
  setup symbol, stage-only helper/test, `GpuOperationStage::SurfaceConfigure`,
  fake setup-scope claim, or non-readback production poll remains. Worker and
  reviewer inspect every changed feature-gated and cfg-test renderer helper so
  renaming cannot satisfy this criterion.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  dropped_gpu_operation_future_aborts_draft_state_and_leases`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  real_gpu_error_scope_captures_deliberate_validation_error`;
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render
  real_gpu_smoke_emits_no_uncaptured_error`; `rg -n
  'PresentedSetup|presented_setup|presented setup orchestration|GpuOperationStage::SurfaceConfigure'
  src` (exit 1 is clean); `C02-CHECK`.
- Dependencies: task 3. Intended logical point: `Adopt scoped GPU transactions`.

## Completion

- Acceptance: all four ordered adoption ranges and reviews are clean; every
  provisional implementation hunk is owned once; rejected setup code is gone;
  identity/async/terminal/transaction foundations satisfy the referenced spec;
  external Vello remains only unchanged temporary raster behavior; later-cycle
  claims do not enter the range.
- Final commands: exact `C02-CHECK`; task 2 and task 4 static scans; and
  `git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg
  -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{'`
  (no executable match; exit 1 is clean).
- Handoff: after canonical complete-status, final-check, holistic-review,
  publication, and remote-readback gates, provide the exact C02 SHA and evidence
  as C03's base; root receives no handoff.
- Blockers: only an unprovided design decision, unowned change, forbidden
  acquisition, unavailable required native GPU evidence, or material
  contradiction in the reviewed packet. Wasm target/browser evidence remains
  C14-owned and does not block C02.
