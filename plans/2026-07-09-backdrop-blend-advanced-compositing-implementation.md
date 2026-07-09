# Backdrop, Blend, And Advanced Compositing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or the crate-local AGENTS.md coordinator workflow to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Sequence 13 from `plans/2026-07-09-render-css-implementation-sequence.md`: backdrop capture/filtering and the advanced compositing contract needed by backdrop effects, blend behavior, and future mask/effect operators.

**Architecture:** Sequence 13 builds on Sequence 9 offscreen rendering, Sequence 10/11 materialized filter execution, and Sequence 12 mask/clip composition. Render should add explicit backdrop/compositing models, deterministic CPU reference oracles, and a narrow executable backdrop path for resolved bounded layers while keeping root-owned URL loading, CSS stacking-tree construction, and unimplemented blend/composite modes diagnostic-only.

**Tech Stack:** Rust 2024, Vello 0.9 scene encoding, existing `OffscreenLocalSceneRenderRequest`, `ImageBuffer`, `ResolvedAlphaMaskExecution`, materialized filter execution, typed `Error` diagnostics, crate-local tests.

---

## Source Scope

Sequence item:

- `plans/2026-07-09-render-css-implementation-sequence.md`, sequence 13.

Matrix rows:

- `plans/2026-07-08-render-css-support-matrix.md`, sections 7, 9, and backend pipeline rows for the backdrop compositor.

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Current Baseline

- `Layer` already models `BlendMode::{Normal, Multiply, Screen, Overlay, Darken, Lighten, Plus}` and direct Vello layer isolation for opacity/blend.
- `Capabilities::VELLO_0_9` currently advertises direct Vello opacity/blend isolation but not offscreen backdrop execution.
- Sequence 9 introduced offscreen texture descriptors, rect shader pass foundations, and local-scene render/readback helpers.
- Sequence 10/11 introduced deterministic materialized color/blur/drop-shadow filter execution for `ImageBuffer`.
- Sequence 12 introduced resolved layer alpha mask execution by rendering a bounded layer subtree offscreen, reading it back, applying a reference operation, and replacing it with an image command.
- There is no render-owned backdrop input model, backdrop capture command, background blend list model, or public Porter-Duff/composite operator model beyond the reference helpers used internally by tests.

## Boundary Decisions

- Root owns CSS stacking-tree construction, scroll state, URL/resource resolution, and deciding which element/layer receives backdrop behavior.
- Render owns bounded backdrop capture once root/authored inputs provide a layer region and the command stream ordering is explicit.
- Render may execute a narrow backdrop path by materializing previous sibling commands in the same render command list, applying a supported materialized filter list, and compositing the filtered backdrop behind the foreground layer.
- Render must not claim full browser stacking-context behavior, root backdrop capture, URL/SVG filter graphs, or background-blend execution until explicit models and tests exist.
- Vello direct blend behavior remains available for the existing `BlendMode` set; Sequence 13 must add reference coverage and capability honesty, not replace working direct Vello blending unless a tested offscreen path is required for backdrop composition.
- Backwards compatibility shims are not required. Prefer intentional front doors and remove obsolete names if they conflict with the model.

## File Map

- `src/capability.rs`: add compositing/backdrop capability bits and typed support checks for narrow backdrop capture/filter/composite operations.
- `src/error.rs`: add any missing `PrimitiveOperation` variants for backdrop capture, backdrop filter chain, root backdrop policy, background blend mode, or unsupported mix/composite modes.
- `src/style.rs`: add render-owned backdrop filter input, backdrop bounds/policy, background blend list, and composite operation models as needed.
- `src/layer.rs`: add a layer backdrop front door only if the executor can support a bounded resolved backdrop path in this sequence.
- `src/command.rs`: normalize backdrop-bearing layers, plan pass requirements, and keep diagnostic boundaries explicit.
- `src/renderer.rs`: implement ordered backdrop materialization using already-normalized prior commands and existing offscreen/readback/filter helpers.
- `src/reference.rs`: add deterministic reference compositing/blend helpers required by tests.
- `src/lib.rs`: expose only crate-owned public front doors.
- `src/tests.rs`: add focused unit and integration tests for every task.

## Task Sequence

Follow `AGENTS.md` coordinator workflow for every code task:

1. Assign one scoped worker task or tightly coupled task group.
2. Have a separate clean-context reviewer inspect the worker changes.
3. Reconcile reviewer findings before moving on.
4. Run the focused checks listed for the task.
5. Commit the task only after worker result, reviewer result, and focused checks are clean.

### 1. Backdrop/Compositing Capability And Diagnostic Contract

Worker scope:

- Add granular capability names for:
  - bounded backdrop capture
  - materialized backdrop filter execution
  - backdrop isolation/composition
  - root backdrop policy
  - background blend mode
  - unsupported mix/composite modes if the existing `BlendMode` set is not complete enough for the matrix row
- Keep broad `PrimitiveOperation::BackdropExecution` unsupported until the executable path lands.
- Preserve existing direct Vello opacity/blend capability claims.
- Add tests proving `Capabilities::VELLO_0_9` advertises only the behavior implemented at the end of this task.

Implementation notes:

- Prefer adding specific `PrimitiveOperation` variants rather than overloading `BackdropExecution`.
- If a capability is planned but not executable until later tasks, default it to `false` and assert the typed diagnostic.
- Do not change rendering behavior in this task.

Focused checks:

```sh
cargo test -p surgeist-render capability
cargo test -p surgeist-render backdrop
cargo test -p surgeist-render blend
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Reference Blend And Composite Oracle

Worker scope:

- Extend `src/reference.rs` with deterministic premultiplied reference helpers for the compositing operators Sequence 13 needs:
  - source-over, already present, must remain stable
  - plus/lighter if matching `BlendMode::Plus`
  - multiply, screen, overlay, darken, and lighten for existing `BlendMode`
  - source-in and destination-in if needed to keep mask/composite behavior covered
- Keep helper visibility crate-internal unless a public front door is required.
- Add tests for transparent, opaque, partial-alpha, and mismatched-size behavior.

Implementation notes:

- Preserve premultiplied invariants: color channels must never exceed alpha.
- Use integer or deterministic rounded math matching existing reference helpers.
- Do not use GPU/Vello output as the oracle for these tests.

Focused checks:

```sh
cargo test -p surgeist-render reference
cargo test -p surgeist-render blend
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Render-Owned Backdrop And Background Blend Models

Worker scope:

- Add a render-owned `BackdropFilterInput` or equivalent model that carries:
  - a non-empty supported `FilterList`
  - explicit capture bounds or an explicit root/backdrop policy diagnostic
  - optional clip geometry already resolved into render-owned `ClipInput`
- Add a background blend list model or diagnostic front door for per-background-layer blend semantics.
- Add a public layer method only if the model has a supported executor path planned in Task 5; otherwise expose a diagnostic model without wiring it into `Layer`.
- Add constructor tests for empty filters, invalid bounds, unresolved clips, unsupported root backdrop policy, and unsupported background blend lists.

Implementation notes:

- Root-resolved colors and resources remain root-owned.
- Backdrop filter lists may reuse existing `FilterList` and materialized image filter classification, but unsupported filter graph/SVG/resource filters must stay diagnostic.
- Avoid sibling crate dependencies.

Focused checks:

```sh
cargo test -p surgeist-render backdrop
cargo test -p surgeist-render filter
cargo test -p surgeist-render background
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Backdrop Capture Planning And Command Normalization

Worker scope:

- Normalize backdrop-bearing inputs into explicit `RenderCommand` state.
- Add a pass-plan requirement that distinguishes bounded backdrop capture from broad unsupported backdrop execution.
- Validate finite non-empty capture bounds and clip bounds.
- Preserve command order: backdrop capture must only see previously rendered sibling commands, never later foreground commands.
- Add tests for:
  - bounded backdrop pass planning
  - root backdrop rejection or explicit root policy diagnostic
  - command-order preservation
  - nested group and nested backdrop behavior, either implemented or rejected by a typed diagnostic that explains the unsupported nesting boundary
  - rounded/path clip carried into backdrop capture planning

Implementation notes:

- If the executor cannot support a case yet, return a typed diagnostic from normalization.
- Do not introduce global scene graph or root stacking context ownership.
- Keep existing opacity, blend, mask, and clip behavior green.

Focused checks:

```sh
cargo test -p surgeist-render backdrop
cargo test -p surgeist-render layer
cargo test -p surgeist-render clip
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Materialized Backdrop Capture And Filter Execution

Worker scope:

- Implement the narrow executable backdrop path:
  - render prior sibling commands covering the capture bounds into an offscreen texture
  - read the captured backdrop into an `ImageBuffer`
  - apply a supported materialized filter chain using Sequence 10/11 filter execution
  - place the filtered backdrop behind the layer foreground in the correct bounds
- Support color filters and blur where the existing materialized filter executor supports them.
- Preserve clipping so the filtered backdrop is clipped to the requested capture/clip region.
- Add pixel tests for:
  - rect backdrop capture samples only prior content
  - rounded clip does not leak outside the clip
  - nested backdrop/group ordering when supported, or a stable typed diagnostic when nested backdrop capture is still outside render's executable boundary
  - filter order is preserved
  - foreground content composites over filtered backdrop

Implementation notes:

- The first implementation may use the same readback/materialized-image strategy as Sequence 12 layer masks if that is the most reliable path.
- Avoid claiming broad GPU post-processing if this uses CPU/reference materialization internally.
- If GPU context is unavailable, return the same stable adapter diagnostic style used by resolved layer masks.

Focused checks:

```sh
cargo test -p surgeist-render backdrop
cargo test -p surgeist-render filter
cargo test -p surgeist-render offscreen
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 6. Blend Mode And Isolation Reconciliation

Worker scope:

- Add reference-oracle tests for the existing `BlendMode` set:
  - normal
  - multiply
  - screen
  - overlay
  - darken
  - lighten
  - plus
- Confirm Vello direct blend output matches expected qualitative or deterministic reference behavior for representative pixels.
- Add typed diagnostics for blend/composite modes not represented by the public enum or explicitly root-owned/background-owned models.
- Add tests for isolated vs non-isolated blend behavior where render has enough command-order information to assert it.
- Add tests for nested isolated groups when supported, or typed diagnostics when a nested group requires unsupported offscreen/backdrop semantics.

Implementation notes:

- Do not expand the public `BlendMode` enum unless every new variant has an encoding and tests.
- Background blend mode is separate from layer `BlendMode`; do not route background-layer blending through `Layer::blend` without a model that preserves layer-list semantics.

Focused checks:

```sh
cargo test -p surgeist-render blend
cargo test -p surgeist-render layer
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 7. Porter-Duff And Composite Policy Boundary

Worker scope:

- Introduce or complete a render-owned composite operation model only for operators used by implemented masks/backdrop/blend behavior.
- Implement deterministic reference tests for supported operators.
- Add typed diagnostics for unsupported Porter-Duff or CSS composite operations not implemented in this sequence.
- Reconcile `MaskCompositeMode` from Sequence 12 with any new composite model without weakening its diagnostics.

Implementation notes:

- If no public composite model is needed yet, document the private helper boundary through tests and typed diagnostics instead of adding public API.
- Keep `MaskLayerStack` non-default composites unsupported unless this task implements and tests them completely.

Focused checks:

```sh
cargo test -p surgeist-render composite
cargo test -p surgeist-render mask
cargo test -p surgeist-render backdrop
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 8. Integration Guardrails And Sequence 13 Reconciliation

Worker scope:

- Add integration tests named with `sequence13` that prove the matrix rows are either implemented or explicitly diagnostic:
  - backdrop capture
  - backdrop filter chain
  - backdrop isolation
  - sibling ordering and nested backdrop/group behavior, implemented or explicitly diagnostic
  - root backdrop policy
  - mix blend mode
  - background blend mode diagnostics or implementation
  - Porter-Duff/composite ops required by masks/effects
- Confirm Sequence 10 color filter tests, Sequence 11 blur/drop-shadow tests, and Sequence 12 mask/clip tests still pass.
- Confirm `Capabilities::VELLO_0_9` now advertises exactly the narrow Sequence 13 behavior implemented and nothing broader.
- Add plan notes only if implementation discovers a real boundary that root must know.

Focused checks:

```sh
cargo test -p surgeist-render sequence13
cargo test -p surgeist-render backdrop
cargo test -p surgeist-render blend
cargo test -p surgeist-render composite
cargo test -p surgeist-render filter
cargo test -p surgeist-render mask
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## Final Review And Required Checks

After all scoped tasks are implemented, assign a final clean-context holistic reviewer to inspect the complete result against:

- this implementation plan
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- the full git diff for the Sequence 13 implementation

The final reviewer must confirm:

- backdrop capture is ordered and bounded
- backdrop filter chains preserve filter order and clipping
- root backdrop behavior is implemented or explicitly diagnostic
- blend/composite capabilities claim only implemented behavior
- direct Vello blend behavior and any materialized/reference blend behavior are reconciled
- Porter-Duff/composite helpers preserve premultiplied-alpha invariants
- background blend mode is implemented with list semantics or remains typed diagnostic
- public APIs remain intentional and crate-owned
- no sibling crates or root submodule pointers were edited

Run these final checks before declaring Sequence 13 complete:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## References

- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-offscreen-layer-texture-pipeline-implementation.md`
- `plans/2026-07-09-color-filter-pipeline-implementation.md`
- `plans/2026-07-09-blur-drop-shadow-shadow-completion-implementation.md`
- `plans/2026-07-09-masks-clips-masked-composition-implementation.md`
