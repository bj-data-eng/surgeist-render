# Offscreen Layer And Texture Pipeline Implementation Plan

Date: 2026-07-09

## Scope

Implement Sequence 9 from
`plans/2026-07-09-render-css-implementation-sequence.md`: the render-local
offscreen layer and texture infrastructure required by later filters, masks,
blend isolation, and backdrop effects.

Source matrix:

- `plans/2026-07-08-render-css-support-matrix.md`

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

This phase is render-local. It must not add sibling crate dependencies, edit
sibling crates, update root submodule pointers, or implement CSS filter/blur,
mask compositing, or backdrop behavior that belongs to later sequence items.
Backwards compatibility shims are not required.

## Existing Inputs

The crate already owns these relevant pieces:

- `Layer` with clip, transform, opacity, blend, mask, and filter inputs.
- Internal `NormalizedLayer` and `LayerIsolation` in `src/command.rs`.
- Vello layer encoding for clip-only, opacity, and blend isolation.
- Headless/presented surface texture allocation in `src/backend.rs` and
  `src/surface.rs`.
- Typed unsupported diagnostics for layer masks and layer filters.
- Pixel readback tests for headless rendering, opacity, and blend behavior.

Sequence 9 should build infrastructure around these models instead of
replacing them. Existing direct Vello rendering must remain the fast path when
no offscreen pipeline is needed.

## Target Design

Add explicit render-owned infrastructure for future compositor passes:

- Capability and diagnostic names for offscreen layers, texture cache/upload,
  rect shader passes, CPU reference buffers, isolation groups, and nested
  opacity.
- Offscreen pass planning types that describe why a layer needs isolation,
  which bounds it needs, and whether it can remain a Vello direct layer.
- Texture allocation/cache lifecycle models with stable descriptors, keys, and
  reuse/release accounting.
- CPU reference RGBA buffers and compositing helpers suitable for deterministic
  later filter and compositing tests.
- Backend helpers that can render a normalized subtree into an offscreen texture
  with explicit physical bounds, without changing the public CSS/style surface.

This phase may use Vello's existing `push_layer` behavior for current opacity
and blend behavior. It should introduce explicit offscreen texture
infrastructure for later phases, but it must not implement color filters, blur,
masks, backdrop capture, or new blend algorithms beyond current behavior.

## Task Sequence

Each scoped task must follow the AGENTS coordinator workflow before the next
task begins:

1. Assign one implementation worker for the scoped task.
2. Have a separate clean-context reviewer inspect that worker's changes.
3. Reconcile any findings with follow-up worker/reviewer cycles.
4. Run the focused checks listed for the task.
5. Commit the task only after the worker result, reviewer result, and focused
   checks are clean.

### 1. Offscreen Pipeline Capabilities And Diagnostics

Worker scope:

- Extend the capability model with explicit offscreen/compositor pipeline
  capabilities. Prefer a focused `OffscreenPipelineCapabilities` type unless
  local code strongly favors extending `CompositingCapabilities`.
- Name support for:
  - direct Vello layer isolation for opacity and blend
  - offscreen layer rendering
  - texture cache/upload lifecycle
  - rect/fullscreen shader passes
  - CPU reference buffers
  - nested opacity planning
  - mask/filter/backdrop execution remaining unsupported in this phase
- Add precise `PrimitiveOperation` variants when current diagnostics need more
  specific operation names than the existing `LayerMask` and `LayerFilter`.
- Preserve current `Capabilities::VELLO_0_9` behavior: opacity and blend are
  supported; masks and filters still reject through typed diagnostics.
- Export new capability types as needed.
- Add tests for capability accessors and representative diagnostics.

Focused checks:

```sh
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Offscreen Bounds And Layer Pass Planning

Worker scope:

- Add render-owned planning models for offscreen layer requirements, likely in
  `src/command.rs` or a small crate-local module:
  - `LayerPassRequirement`
  - `LayerPassKind`
  - `OffscreenBounds`
  - `LayerPassPlan`
- Preserve existing `LayerIsolation` semantics while making the reason for
  isolation explicit: none, clip-only, opacity/blend direct layer, future
  offscreen texture pass, mask/filter/backdrop diagnostic boundary.
- Compute explicit bounds from root-supplied layer clips or child command
  geometry where available. If bounds cannot be known for a future offscreen
  requirement, return a typed invalid/unsupported diagnostic rather than using
  unbounded sentinel rectangles.
- Keep direct Vello encoding unchanged for current no-offscreen and direct-layer
  cases.
- Add tests for nested layers, clip bounds, opacity/blend pass planning, and
  unsupported mask/filter pass requirements.

Focused checks:

```sh
cargo test -p surgeist-render layer
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Texture Allocation And Cache Lifecycle Models

Worker scope:

- Add internal texture lifecycle models that can be used by backend code and
  later shader/filter passes:
  - texture descriptor with physical size, format, and usage intent
  - stable texture cache key
  - allocation state and reuse/release accounting
  - offscreen texture handle that does not expose raw `wgpu` resources publicly
- Integrate the model with existing headless texture allocation where useful,
  without rewriting presented/headless surface behavior.
- Preserve `Renderer` image upload accounting and do not conflate image uploads
  with offscreen pass textures.
- Add tests for descriptor equality, reuse hits, release/eviction accounting,
  invalid zero-size/overflow descriptors, and separation from image cache state.

Focused checks:

```sh
cargo test -p surgeist-render texture
cargo test -p surgeist-render headless
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Rect Shader Pass Foundation

Worker scope:

- Add a minimal WGPU rect/fullscreen shader-pass foundation over texture views:
  - pass descriptor naming source texture, destination texture, bounds, and
    shader/pass kind
  - pipeline/cache key models separate from texture cache keys
  - identity/copy pass or clear/fill pass sufficient to prove command encoding
    and texture-view wiring without implementing CSS color filters
  - explicit diagnostics for unavailable GPU/device contexts
- Keep shader pass APIs internal unless a public render-owned model is needed
  for later plan phases.
- Do not implement brightness/contrast/color-matrix, blur, masks, or backdrop
  shaders in this phase.
- Add tests for pass descriptors, pipeline key stability, bounded rect pass
  validation, contract-only behavior without a GPU context, and at least one
  GPU-backed identity/copy or clear/fill pass when a device is available.

Focused checks:

```sh
cargo test -p surgeist-render shader
cargo test -p surgeist-render texture
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. CPU Reference Buffer And Compositing Foundation

Worker scope:

- Add CPU/reference buffer types for deterministic later filter and compositing
  tests:
  - finite positive size validation
  - RGBA8 or premultiplied-RGBA storage policy named in the type
  - pixel access helpers that preserve bounds checks
  - source-over and opacity helper functions needed by later oracle tests
- Keep this as test/reference infrastructure. Do not switch production rendering
  away from Vello.
- Add tests for allocation, pixel access bounds, opacity application,
  source-over composition, transparent edges, and deterministic equality.

Focused checks:

```sh
cargo test -p surgeist-render reference
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 6. Minimal Offscreen Render Entry Point

Worker scope:

- Add backend helpers that can render a Vello scene/subtree into an offscreen
  texture with explicit bounds using the texture lifecycle model.
- Keep the existing direct surface render path unchanged unless an explicit
  offscreen pass is requested.
- Add focused tests that exercise:
  - offscreen texture allocation for a bounded layer
  - rect shader pass descriptors can target offscreen textures without running
    filter-specific shaders
  - nested layer opacity continues to render through the current direct Vello
    path
  - resource reuse across repeated bounded offscreen requests
  - no allocation when isolation is unnecessary
- If GPU access is unavailable, tests must assert the contract-only model and
  diagnostics rather than requiring hardware-specific output.

Focused checks:

```sh
cargo test -p surgeist-render offscreen
cargo test -p surgeist-render layer
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 7. Integration Guardrails And Final Review

Worker scope:

- Add integration tests that prove Sequence 9 infrastructure is ready for later
  filter/mask/blend/backdrop phases without implementing those phases:
  - direct Vello rendering remains unchanged for ordinary scenes
  - offscreen pass planning always carries explicit finite bounds
  - texture lifecycle accounting is deterministic under nested layers
  - rect shader-pass plumbing is available without color-filter/blur semantics
  - CPU reference buffers can act as deterministic filter/composition oracles
  - layer mask/filter inputs still fail with existing typed diagnostics
- Do not add CSS filter, blur, mask compositing, backdrop capture, or
  background-blend behavior here.

Focused checks:

```sh
cargo test -p surgeist-render offscreen
cargo test -p surgeist-render layer
cargo test -p surgeist-render texture
cargo test -p surgeist-render shader
cargo test -p surgeist-render reference
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

After all scoped tasks are committed, assign a final clean-context holistic
reviewer to inspect the complete Sequence 9 result against this plan,
`AGENTS.md`, the support matrix, the modeling guide, crate boundaries, tests,
and git diff.

Final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## Non-Goals

- No sibling crate edits.
- No root submodule pointer updates.
- No CSS color-filter implementation.
- No blur/drop-shadow implementation.
- No mask compositing implementation.
- No backdrop capture or backdrop-filter implementation.
- No new public CSS/style surface.
- No replacement of the current direct Vello path for scenes that do not need
  offscreen infrastructure.
- No backwards compatibility shims.
