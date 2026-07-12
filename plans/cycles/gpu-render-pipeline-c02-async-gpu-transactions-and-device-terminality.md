# GPU Render Pipeline C02 Async GPU Transactions And Device Terminality

## Header

- Cycle: `C02`.
- Owner: `surgeist-render`.
- Status: `in_progress`.
- Cycle base: `5361e3460278dffb877b9d485a2d12977977c3ef`.
- Specification: `plans/specs/gpu-render-pipeline.md` at commit `3826a9098e859874a515bbebaf470a47d754d76c`, content SHA-256 `f01972e19f8a5ddc90936edfc6ea7955feff3d1b1fdf5d181e77e8d10cc1f60a`, sections S12-S13, non-readback S13A, identity/device/publication portions of S25-S26, applicable S29 rows, S31, C02-applicable named evidence in S32-S33, S35, and the native feature/tooling limits in S36-S37.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at commit `ff0d4a3c478f6f89cceab3962883bd53396cba6b`, content SHA-256 `1ff25dbd5bb0382e2e66573affd058a3e3f939bede561a896ce6e1dea7d73840`, entry `C02 Async GPU Transactions And Device Terminality`.
- Outcome: make non-readback GPU entry points asynchronous; validate renderer and device identity before slot access; make each selected device terminal on its first loss or uncaptured fault; scope and classify owned GPU work; and commit headless or presented frame state only at the specified clean transaction boundary.

## Boundary

- Current surfaces carry raw `dev_id` values and backend code indexes parallel Vello vectors directly. C02 replaces that authority with private renderer identity and generation-bearing device-state slots.
- `Renderer::new` is async, but create, render, and resume block or expose synchronous GPU operations. C02 changes exactly `create_surface`, `create_headless`, `render`, and `resume_surface` to the S13A async forms and removes production executor waits from those paths.
- The stable Vello 0.9 `render_to_texture` call remains the direct GPU submission primitive inside async render transactions. C02 does not use Vello's deprecated async renderer or add a CPU path.
- C02 registers WGPU 29 safe callbacks and uses its non-`Send` error-scope guards. Callback cells retain portable typed records and prose, not target-dependent raw uncaptured `wgpu::Error` values.
- `read_headless`, its map/copy/poll helper, and the two temporary internal materialization readbacks remain synchronous C03-owned progress. C02 may add identity and terminal guards around them, but does not change their signature, staging state, mapping, polling, row decoding, or error code.
- C02 establishes headless draft-versus-published ownership, including `SurfaceResourceState::Empty`, but does not add C05's persistent effect resource manager or pool old publications.
- No C04 graph planning, custom effect pass, final route/stat/capability cutover, README/example work, wasm target proof, or root adaptation enters this cycle.
- Work stays in this repository with the pinned dependencies. No compatibility shim, dependency/acquisition, production CPU fallback, generated artifact, or owned `unsafe` is authorized.

## Impacts

- Public API: breaking sync-to-async changes for the four non-readback renderer methods; additive `Renderer::runtime_capabilities`; additive `SurfaceResourceState::Empty`; corrected runtime/lifecycle errors and removal of obsolete adapter/surface backend codes once all current uses are migrated. `read_headless` stays synchronous until C03.
- Dependencies/features: unchanged. Default, `render-window`, `render-web`, and their additive combination remain supported native check states.
- Generated artifacts: none.
- Docs/examples: changed public methods and lifecycle values receive contract rustdocs; README and the native presented example remain C12-owned.
- MSRV: use Rust 1.89-compatible APIs; the absent 1.89 toolchain and wasm target remain C12 gates.
- Root follow-up: none; the published C02 SHA becomes C03's cycle base.
- Unsafe: crate-level `forbid(unsafe_code)`, Clippy `-F unsafe-code`, and the owned-source scan remain mandatory.

## Tasks

Every task runs its focused RED/acceptance commands and the exact `C02-CHECK` matrix before review:

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

### 1. Install Renderer Identity And Device-State Slots

- Files/area: `src/renderer.rs`, `src/backend.rs`, `src/surface.rs`, `src/error.rs`, and focused `src/tests.rs` model coverage.
- Outcome: add private `RendererIdentity(Arc<()>)`, generation-bearing `DeviceSlotIdentity`, and one backend `DeviceState` per Vello slot; every renderer operation checks renderer identity, backend-kind compatibility, slot bounds, and generation in S26 order before indexing or calling WGPU/Vello.
- RED evidence: first add `foreign_and_stale_surfaces_fail_before_device_slot_access`; the current raw slot model cannot represent either typed failure and indexes directly.
- Acceptance: contract-only surfaces carry no device identity; device-backed surfaces carry no public slot; foreign/stale render, read, and resume produce their exact validated runtime diagnostic without backend access; valid current surfaces preserve behavior.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render foreign_and_stale_surfaces_fail_before_device_slot_access`; exact `C02-CHECK`.
- Dependencies: none. Intended worker commit: `Add renderer and device slot identity`.

### 2. Make Non-Readback Renderer Operations Async

- Files/area: public methods and helpers in `src/renderer.rs`, `src/backend.rs`, `src/surface.rs`, all crate-owned callers in `src/tests.rs`, and public rustdocs/reexports as applicable.
- Outcome: implement the exact async signatures for create, headless create, render, and resume; await Vello surface creation directly; migrate test callers through pinned `pollster`; retain synchronous readback unchanged.
- RED evidence: first add `non_readback_renderer_front_door_is_async`; its awaited public calls fail to compile against the current signatures.
- Acceptance: production code contains no `pollster::block_on` or Vello blocking executor; returned futures have no `Send` promise; behavior and error ordering remain unchanged except where task 1 already corrected identity.
- Commands: `CARGO_NET_OFFLINE=true cargo test -p surgeist-render non_readback_renderer_front_door_is_async`; `rg -n 'pollster::block_on|block_on_wgpu' src --glob '!tests.rs'` (no match, exit 1, is clean); exact `C02-CHECK`.
- Dependencies: task 1. Intended worker commit: `Make renderer GPU operations async`.

### 3. Add Terminal Device Signals And Runtime Reports

- Files/area: device state/callback ownership in `src/backend.rs`, runtime projection in `src/renderer.rs` and `src/capability.rs`, typed diagnostics in `src/error.rs`, and model/real-device tests in `src/tests.rs`.
- Outcome: snapshot selected adapter format features and device limits; register first-record-wins loss and uncaptured-error callbacks for each new slot; implement idempotent per-generation `Ready -> Lost/Faulted` cleanup; and add allocation-free `Renderer::runtime_capabilities`.
- RED evidence: first add `device_loss_is_terminal_idempotent_and_releases_device_resources`, `terminal_default_device_rejects_headless_without_disabling_ready_slots`, and `runtime_capabilities_project_the_selected_surface_without_gpu_work`; current backend has no signal, terminal state, or query.
- Acceptance: poison recovery retains the first record; terminal observation drops only that slot's Vello renderer and blocks later GPU calls; another ready slot remains usable; reports use adapter features plus device limits and honor the complete S26 identity/lifecycle order.
- Commands: the three named focused tests; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render destroyed_device_callback_reports_terminal_loss_without_stale_resource_use`; exact `C02-CHECK`.
- Dependencies: task 2. Intended worker commit: `Track terminal GPU device state`.

### 4. Scope And Classify Owned GPU Operations

- Files/area: a private transaction module, `src/backend.rs`, `src/renderer.rs`, `src/shader.rs`, `src/error.rs`, and focused transaction/GPU tests in `src/tests.rs`.
- Outcome: add monotonic active operation generations, RAII draft/lease cleanup, and nested `Internal -> OutOfMemory -> Validation` WGPU scopes popped and awaited in reverse; map captured and uncaptured errors by owning create/render/present stage with loss precedence; remove non-readback `Device::poll`.
- RED evidence: first add `gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage`, `dropped_gpu_operation_future_aborts_draft_state_and_leases`, and `real_gpu_error_scope_captures_deliberate_validation_error`; current code has no scope or cancellation boundary.
- Acceptance: create/configure, Vello creation/draw, current submissions, and the clear/fill probe use the transaction boundary; captured errors do not fault the device; active/no-active uncaptured errors do; cancellation clears active generation and cannot publish draft state.
- Commands: the three named focused tests; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render real_gpu_smoke_emits_no_uncaptured_error`; exact `C02-CHECK`.
- Dependencies: task 3. Intended worker commit: `Add scoped GPU operation transactions`.

### 5. Publish Headless Frames Atomically

- Files/area: headless state in `src/surface.rs`, allocation/render commit flow in `src/backend.rs` and `src/renderer.rs`, and focused lifecycle/pixel tests in `src/tests.rs`.
- Outcome: represent zero-size `Empty`, nonzero unpublished `Pending`, and one readable publication; render every nonzero frame to a separate draft and swap only after scopes/signals are clean; drop failed/canceled drafts and commit stats, parameters, uploads, and publication together.
- RED evidence: first add `headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames` and `zero_size_headless_state_is_empty_and_render_diagnoses_without_gpu_work`; direct in-place rendering and the current successful zero-size render fail those contracts.
- Acceptance: creation and physical-size changes allocate no publication; same-size resize preserves it; failure/cancellation preserves prior pixels or remains pending; zero-size render diagnoses before adapter lookup; C03-owned readback progress remains unchanged.
- Commands: the two named focused tests; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render failed_frame_returns_all_leases_and_preserves_last_successful_stats`; exact `C02-CHECK`.
- Dependencies: task 4. Intended worker commit: `Publish headless frames atomically`.

### 6. Commit Presented Frames After Draw And Present Scopes

- Files/area: presented create/resize/acquire/render/blit/present flow in `src/renderer.rs`, `src/backend.rs`, and `src/surface.rs`, plus feature-gated lifecycle tests in `src/tests.rs`.
- Outcome: keep the acquired surface texture in transaction ownership; finish Vello drawing under one scope set; perform blit, submit, and `present` under a second scope set; then commit last-successful public state only after the final terminal-signal check.
- RED evidence: first add `surface_loss_can_resume_but_device_loss_requires_a_new_renderer` and a deterministic presented transaction test proving a late fault after attempted presentation does not commit stats/parameters; current path has one unscoped sequence and polls the device.
- Acceptance: unpresented textures discard on every exit; configure failure leaves resize pending; timeout/outdated/occluded/lost use exact lifecycle mappings; compatible resume is idempotent; an attempted external present is the sole documented non-rollback side effect. Live-window execution remains C12 evidence.
- Commands: the two focused tests under `render-window`; `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window presented_surface_lifecycle`; exact `C02-CHECK`.
- Dependencies: task 5. Intended worker commit: `Make presented frames transactional`.

### 7. Close The Non-Readback Lifecycle And Error Matrix

- Files/area: `src/error.rs`, `src/renderer.rs`, `src/backend.rs`, `src/surface.rs`, `src/shader.rs`, and matrix/static tests in `src/tests.rs`.
- Outcome: complete S13/S26/S31 non-readback ordering, duplicate transitions, terminal isolation, stage mappings, and backend-code removal; constrain the only remaining production poll/map path to C03's explicit and temporary readback helper.
- RED evidence: first add `surface_operation_matrix_covers_every_kind_state_and_duplicate_transition` and `uncaptured_gpu_error_faults_only_its_device_generation`; current lifecycle and global backend state fail the matrix.
- Acceptance: every C02-owned matrix cell is covered; no generic adapter/surface code substitutes for typed runtime evidence; no production blocking executor remains; production polling appears only in the C03-owned readback helper; no later-cycle capability, route, effect, or readback claim enters the range.
- Commands: the two named focused tests; `rg -n 'pollster::block_on|block_on_wgpu' src --glob '!tests.rs'` (no match); `rg -n '\.poll\(wgpu::PollType|map_async|get_mapped_range' src -g '*.rs'` (all production matches inspected and confined to the C03 readback helper); exact `C02-CHECK`.
- Dependencies: task 6. Intended worker commit: `Complete async lifecycle error matrix`.

## Completion

- Acceptance: all seven ordered task ranges and reviews are clean; identity always precedes slot access; non-readback public GPU work is async, scoped, cancellation-safe, and terminal per device; failed/canceled headless frames preserve publication; presented state commits only at its final clean boundary; runtime reports are truthful; C03 readback ownership is intact.
- Final commands: `CARGO_NET_OFFLINE=true cargo check -p surgeist-render`; exact `C02-CHECK`; both task 7 static scans; and `git ls-files -z --cached --others --exclude-standard -- '*.rs' | xargs -0 rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{'` (no executable match; exit 1 is clean).
- Landing: use the canonical task, status, holistic-review, publication, and remote-readback gates. Publish reviewed C02 on authority `origin/main` and hand its exact SHA plus evidence to C03; root receives no C02 handoff.
- Blockers: return only an unprovided design decision, forbidden dependency/acquisition need, unowned repository change, unavailable required native GPU evidence, or material contradiction in the reviewed packet. Missing wasm or Rust 1.89 tooling remains C12-owned and does not block C02.
