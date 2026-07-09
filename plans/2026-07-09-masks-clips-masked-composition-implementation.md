# Masks, Clips, And Masked Composition Implementation Plan

## Goal

Implement Sequence 12 from
`plans/2026-07-09-render-css-implementation-sequence.md`: render-owned
clip-path/basic-shape lowering, alpha mask execution for resolved inputs, and
typed diagnostics for mask/clip semantics that remain outside this phase.

## Architecture

Sequence 12 builds on existing render-local shape/path models, Sequence 6 image
sampling, Sequence 9 offscreen texture planning, and the Sequence 11 reference
buffer/compositing oracles. Render remains self-contained: root resolves CSS
URLs, colors, geometry, image bytes, and style lists before calling this crate.
This phase should preserve that boundary while making render capable of
executing already-resolved masks and clips.

Current state:

- `Layer::clip(Shape)` already lowers to Vello clip/layer behavior for simple
  shape clips.
- `ClipInput` and `MaskInput` model shape/image/reference sources with optional
  coordinate-space tags, but they do not yet normalize into executable layer
  commands.
- `Layer::try_mask(Shape)` is currently a typed `LayerMask` diagnostic
  boundary.
- Sequence 9 provides offscreen texture/pass planning but does not execute mask
  compositing.
- Sequence 11 provides deterministic premultiplied reference buffers and
  source-over composition helpers that can be extended for mask-oracle tests.

Sequence 12 should not add backdrop capture, blend-mode expansion, filter graph
execution, CSS URL loading, root integration, or sibling-crate dependencies.
Those remain later sequence or root responsibilities.

## Scope

In scope:

- Shape/path/basic-shape clip normalization and execution where render already
  has concrete geometry.
- Basic-shape coverage in this phase means root-lowered concrete render
  geometry: `Rect`, `RoundedRect`, `Circle`, `Ellipse`, and path/polygon-like
  `Path` values supplied through render-owned shape/path models. This phase does
  not add symbolic CSS `inset()`/`circle()`/`ellipse()`/`polygon()` plus
  reference-box resolution models; root remains responsible for resolving those
  semantics into render geometry before calling this crate.
- Explicit clip URL/reference diagnostics when root has not resolved a clip.
- Alpha mask execution for resolved render-owned mask inputs.
- Luminance mask diagnostics unless a real deterministic luminance conversion
  and execution path is implemented.
- Ordered/multi-layer mask modeling and diagnostics or alpha execution where
  feasible without root-private state.
- Mask composite diagnostics unless required compositor operators are
  implemented in this phase.
- CPU/reference oracle coverage for alpha mask edges and mask compositing.
- Capability reconciliation for shape/path clips, layer masks, and mask
  execution.

Out of scope:

- Backdrop capture/filtering.
- Mix-blend-mode expansion beyond existing behavior.
- CSS resource loading or URL resolution.
- Text clip/mask execution unless root supplies concrete geometry or pixels.
- New dependencies on sibling crates.
- Broad offscreen filter execution.

## Task Sequence

Follow the AGENTS.md coordinator workflow for every code task:

1. Assign one implementation worker for the scoped task or tightly coupled task
   group.
2. Have a separate clean-context reviewer inspect that worker's changes.
3. Reconcile any findings with follow-up worker/reviewer cycles.
4. Run the focused checks listed for the task.
5. Commit the task only after the worker result, reviewer result, and focused
   checks are clean.

### 1. Clip And Mask Capability/Diagnostic Contract

Worker scope:

- Reconcile `MaskClipCapabilities` and offscreen mask capabilities with the
  Sequence 12 target surface.
- Add or refine typed operations only where existing names are too coarse, such
  as:
  - clip reference unresolved/unsupported execution
  - alpha mask execution
  - luminance mask mode
  - multi-layer mask composition
  - unsupported mask composite modes
- Preserve existing `LayerMask` and `MaskExecution` diagnostics where they still
  describe the boundary.
- Add tests proving current support/unsupported boundaries before execution work
  begins.
- Do not claim layer mask execution, mask execution, or broad compositor support
  until the executing tasks land.

Focused checks:

```sh
cargo test -p surgeist-render capability
cargo test -p surgeist-render mask
cargo test -p surgeist-render clip
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Clip Input Normalization And Path/Shape Clip Execution

Worker scope:

- Add render-owned normalization for `ClipInput` into executable concrete clip
  geometry when the input is a shape/path/basic shape.
- Add or refine a path-clip representation so fill-rule semantics are explicit.
  The matrix requires path clips to preserve even-odd/nonzero behavior; do not
  silently route every path clip through a hardcoded nonzero rule if an authored
  fill rule exists.
- Preserve coordinate-space tags in the normalized representation and validate
  that transformed clips remain finite.
- Ensure shape/path clips can be used for layer clipping without going through
  mask execution.
- Add typed diagnostics for `ClipInputKind::Reference` when root has not
  resolved the clip.
- Add tests for:
  - rect, rounded rect, circle/ellipse, and path clips
  - path clip fill rules, including even-odd and nonzero behavior
  - path clip bounds and invalid-path diagnostics
  - nested clips
  - transformed clips and coordinate-space tags
  - reference clip diagnostics
  - ordinary existing `Layer::clip(...)` behavior remaining intact

Focused checks:

```sh
cargo test -p surgeist-render clip
cargo test -p surgeist-render layer
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Reference Alpha Mask Oracle

Worker scope:

- Extend the reference premultiplied buffer path with deterministic alpha mask
  application.
- Support source-in/destination-in style alpha multiplication needed to prove
  mask edges, without over-generalizing into the full Sequence 13 compositor.
- Define how mask buffers map to source buffers:
  - matching size is executable
  - mismatched size is either rejected with a typed invalid-value diagnostic or
    normalized through an explicit placement model from Sequence 6
- Keep luminance mode diagnostic-only unless a real conversion policy is added
  and tested.
- Add CPU/reference tests for:
  - opaque, transparent, and partial alpha masks
  - premultiplied color preservation under mask alpha
  - transparent edge behavior
  - deterministic repeated runs
  - invalid or mismatched mask buffers

Focused checks:

```sh
cargo test -p surgeist-render reference
cargo test -p surgeist-render mask
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Resolved Alpha Mask Execution Boundary

Worker scope:

- Add an executable boundary for resolved alpha masks that can be used without
  root-private state.
- Prefer explicit materialized inputs, for example a source image/layer buffer
  plus a matching alpha mask buffer, rather than claiming arbitrary resource
  handles are executable.
- If `MaskInput::try_shape(...)` / `MaskSourceKind::Shape` is executable,
  define whether it rasterizes through a Vello/offscreen pass or uses an
  existing materialized alpha buffer. Do not silently approximate shape masks
  without a tested rasterization path.
- `MaskInput::image_layer(...)` / `MaskSourceKind::ImageLayer` may reuse
  Sequence 6 placement/sampling only if all required image bytes and placement
  inputs are resolved.
- `MaskInput::reference(...)` / `MaskSourceKind::Reference` must remain a typed
  unresolved/unsupported resource diagnostic.
- Add tests for:
  - resolved alpha mask execution
  - image-backed alpha mask execution or typed boundary
  - shape-backed mask execution or typed boundary
  - reference mask diagnostics
  - transformed mask diagnostics or execution

Focused checks:

```sh
cargo test -p surgeist-render mask
cargo test -p surgeist-render image
cargo test -p surgeist-render offscreen
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Layer Mask Composition Integration

Worker scope:

- Integrate resolved alpha mask execution with `Layer` normalization/rendering.
- Remove the blanket `LayerMask` diagnostic only for cases that now have a real
  executable resolved mask path.
- Preserve typed diagnostics for unsupported mask modes/sources/composites.
- Ensure mask compositing happens after children are rendered into the layer and
  before the layer is composited back into the parent.
- Use Sequence 9 offscreen bounds/texture planning; avoid introducing backdrop
  behavior.
- Add tests for:
  - layer mask alpha edges
  - nested masked layers
  - masks with transforms/clips
  - mask opacity/child opacity interaction
  - unsupported mask modes and unresolved references still diagnosing

Focused checks:

```sh
cargo test -p surgeist-render mask
cargo test -p surgeist-render layer
cargo test -p surgeist-render offscreen
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 6. Multi-Layer Mask And Composite Policy

Worker scope:

- Model ordered mask layer stacks if existing `MaskInput` is insufficient for
  CSS mask lists.
- Implement only composite operators that are needed and fully tested in this
  phase. Otherwise add typed diagnostics for unsupported composite modes.
- Preserve CSS list ordering and repeated mask-layer semantics where inputs are
  resolved.
- Add tests for:
  - repeated mask layers
  - ordered mask layer composition
  - unsupported luminance/composite diagnostics
  - list length/order validation
  - masks not changing unmasked rendering paths

Focused checks:

```sh
cargo test -p surgeist-render mask
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 7. Integration Guardrails And Sequence 12 Reconciliation

Worker scope:

- Add integration tests that verify Sequence 12 satisfies the matrix rows for:
  - shape clip
  - path clip
  - basic shape clip
  - clip reference diagnostics
  - alpha mask execution for resolved inputs
  - luminance mask diagnostic or implementation
  - multi-layer mask support or typed diagnostics
  - mask composite diagnostics
- Confirm `Capabilities::VELLO_0_9` exposes only the mask/clip support that
  actually exists after this phase.
- Confirm Sequence 9 offscreen tests still pass.
- Confirm Sequence 11 blur/drop-shadow tests still pass and broad backdrop
  support remains unsupported.
- Add or update plan notes only if implementation discovers a real boundary or
  blocker.

Focused checks:

```sh
cargo test -p surgeist-render sequence12
cargo test -p surgeist-render clip
cargo test -p surgeist-render mask
cargo test -p surgeist-render offscreen
cargo test -p surgeist-render filter
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## Final Review And Required Checks

After all scoped tasks are implemented, assign a final clean-context holistic
reviewer to inspect the complete result against:

- this implementation plan
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- the full git diff for the Sequence 12 implementation

The final reviewer must confirm:

- clip-path/basic-shape lowering is explicit and crate-owned
- shape/path clip execution is covered for nested/transformed cases
- alpha mask execution is deterministic and has CPU/reference oracle coverage
- unresolved clip/mask references and unsupported mask modes/composites have
  typed diagnostics
- capabilities claim only implemented mask/clip behavior
- broad backdrop, blend expansion, URL loading, and root integration are not
  claimed early
- public APIs remain intentional and crate-owned
- no sibling crates or root submodule pointers were edited

Run these final checks before committing the completed implementation:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## References

- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `AGENTS.md`
