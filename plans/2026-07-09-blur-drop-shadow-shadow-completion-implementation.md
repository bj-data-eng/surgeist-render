# Blur, Drop Shadow, And Shadow Completion Implementation Plan

Date: 2026-07-09

## Scope

Implement Sequence 11 from
`plans/2026-07-09-render-css-implementation-sequence.md`: pixel-moving blur
and drop-shadow filters plus the remaining shadow-surface decisions required by
the CSS/style matrix.

Source matrix:

- `plans/2026-07-08-render-css-support-matrix.md`

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

This phase is render-local. It must not add sibling crate dependencies, edit
sibling crates, update root submodule pointers, implement masks/backdrop, or
claim broad layer-filter execution unless the implemented path actually renders
that layer effect. Backwards compatibility shims are not required.

## Existing Inputs

The crate already owns these relevant pieces:

- `FilterList`, `FilterOp`, `FilterOpKind::Blur`, and
  `FilterOpKind::DropShadow` in `src/style.rs`.
- Sequence 10 color-filter classification, compiled color-filter execution, and
  materialized image byte execution.
- `Image` and `ImageBuffer` straight RGBA8 byte storage in `src/image.rs`.
- `ReferencePremultipliedRgba8Buffer` and deterministic premultiplied RGBA8
  reference helpers in `src/reference.rs`.
- Sequence 9 offscreen texture and rect-pass foundations.
- `Shadow` in `src/layer.rs`, direct `Scene::shadow(...)`, and Vello encoding
  for solid-color outer rect/rounded/circle shadows.
- Typed unsupported diagnostics for non-solid shadow paint and ellipse/path
  shadows.
- `TextRun` in `src/text.rs`, without any text-shadow model.

Sequence 11 should build on these models. It should not treat a
`ResolvedImageResource` as a byte store, and it should not turn the existing
layer `Filter::try_blur(...)` into supported layer-filter execution unless the
full offscreen/layer path is implemented and tested.

## Target Design

Add a typed pixel-moving filter path with deterministic CPU/reference behavior
and honest capability reporting.

The pipeline should:

- Preserve ordered filter-list semantics across color filters, blur, and
  drop-shadow.
- Execute blur and drop-shadow for materialized `Image`/`ImageBuffer` inputs.
- Model filter-region inflation and clipping explicitly for pixel-moving
  filters.
- Use a separable deterministic blur kernel or equivalent reference algorithm
  with transparent-edge behavior and a named large-radius policy.
- Implement drop-shadow from the source alpha mask, not from the source border
  box.
- Preserve the original source when applying drop-shadow, compositing the
  shadow behind the source as CSS filter semantics require.
- Execute blur/drop-shadow filtered image paint when the image phase provides a
  `FilteredImagePaint` intent plus matching materialized `Image` bytes. Continue
  to reject resource handles that are not paired with bytes.
- Treat CSS `drop-shadow(...)` as offset, blur, and color over the source alpha
  mask. The existing `Shadow::spread` field must not silently become CSS
  drop-shadow spread; reject non-zero spread at the CSS filter boundary unless a
  future non-CSS render extension names it explicitly.
- Keep broad layer-filter, mask, and backdrop execution unsupported unless a
  later task in this plan deliberately wires and tests that exact path.
- Complete the shadow surface by either implementing a render-owned primitive or
  adding an explicit typed diagnostic for each remaining shadow matrix row:
  inset box shadow, text shadow, ellipse/path shadow, and non-solid shadow
  paint.

This phase may use a CPU/reference implementation for materialized image
filters if the capability model names that path honestly. Existing Vello direct
solid shape shadows should remain the fast path for supported outer shadows.

## Task Sequence

Each scoped task must follow the AGENTS coordinator workflow before the next
task begins:

1. Assign one implementation worker for the scoped task or tightly coupled task
   group.
2. Have a separate clean-context reviewer inspect that worker's changes.
3. Reconcile any findings with follow-up worker/reviewer cycles.
4. Run the focused checks listed for the task.
5. Commit the task only after the worker result, reviewer result, and focused
   checks are clean.

Workers must not commit. The coordinator commits only after each scoped
worker/reviewer cycle is clean.

### 1. Pixel-Moving Filter Capabilities And Classification

Worker scope:

- Add granular capability and diagnostic names for:
  - materialized blur filter execution
  - materialized drop-shadow filter execution
  - filter-region/outset planning
  - CPU/reference blur fallback
  - inset box-shadow diagnostic or execution
  - text-shadow diagnostic or execution
- Do not enable broad `FilteredImagePaint`, broad `LayerFilter`, or broad
  offscreen `FilterExecution` unless the implementation for that broad behavior
  exists.
- Add an ordered executable filter classifier for `FilterList` that can sequence:
  - color-only runs using `CompiledColorFilterPipeline`
  - blur operations
  - drop-shadow operations
- Keep `FilterList::none()` as no executable filter pipeline.
- Reject any unsupported future filter operation with a typed diagnostic at the
  first unsupported operation.
- Add tests for:
  - mixed color/blur/drop-shadow order preservation
  - `none` producing no pixel-moving pipeline
  - blur and drop-shadow now accepted by the materialized-image classifier
  - layer filters still unsupported unless explicitly implemented later
  - broad filtered resource handles still not treated as bytes

Focused checks:

```sh
cargo test -p surgeist-render filter
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Filter Region, Outset, And Blur Policy Models

Worker scope:

- Add render-owned models for pixel-moving filter regions:
  - source bounds
  - inflated bounds
  - clip bounds
  - blur/drop-shadow outsets
  - device-pixel conversion policy where needed
- Add a blur policy type that names:
  - CSS blur radius/std-deviation interpretation
  - kernel support radius
  - large-radius clamp or rejection policy
  - transparent-edge sampling policy
- Keep invalid states hard to construct: no negative blur radius, no non-finite
  bounds, no unbounded sentinel rectangles, and no zero-area execution region.
- Align existing shadow bounds with the same outset model where possible, while
  preserving current supported outer-shadow rendering behavior.
- Add tests for:
  - zero-radius blur producing zero outset
  - positive blur inflating bounds deterministically
  - drop-shadow combining alpha-mask offset plus blur outset
  - clipping an inflated region back to an explicit filter region
  - large-radius policy
  - invalid bounds/radii diagnostics

Focused checks:

```sh
cargo test -p surgeist-render filter
cargo test -p surgeist-render shadow
cargo test -p surgeist-render offscreen
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Deterministic Separable Blur Reference Path

Worker scope:

- Implement a deterministic CPU/reference blur path for
  `ReferencePremultipliedRgba8Buffer`.
- Use the Task 2 blur policy model for kernel construction, edge sampling, and
  large-radius handling.
- Preserve premultiplied invariants after every blur.
- Define transparent-edge behavior explicitly: pixels outside the source buffer
  are transparent black unless a later typed edge mode says otherwise.
- Ensure radius `0` is an identity operation.
- Add tests for:
  - identity blur
  - small-radius blur over an impulse pixel
  - transparent edges
  - partially transparent colored pixels
  - large-radius clamp or diagnostic policy
  - deterministic equality across repeated runs

Focused checks:

```sh
cargo test -p surgeist-render reference
cargo test -p surgeist-render filter
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Materialized Image Blur And Ordered Filter Execution

Worker scope:

- Extend the Sequence 10 materialized image execution boundary so it can execute
  ordered chains containing color filters and blur.
- Add a positive filtered-image execution path for
  `FilteredImagePaint` paired with a matching materialized `Image`, covering the
  Sequence 11 requirement for resolved image plus pixel-moving filter list.
- Preserve straight RGBA8 image input/output and premultiplied RGBA8 reference
  execution phases.
- Apply filter-region inflation and clipping from Task 2.
- Keep source image identity stable only when bytes are unchanged; filtered
  output identity must reflect filtered bytes.
- Reject layer filters and resource-only filtered-image execution if materialized
  bytes are not provided.
- Add tests for:
  - blur on a 1x1 transparent and opaque image
  - blur on a multi-pixel image with transparent edges
  - `FilteredImagePaint` plus matching `Image` executing blur and preserving
    resource/size validation
  - mixed color-before-blur and blur-before-color order sensitivity
  - output region/size behavior
  - resource-only `FilteredImagePaint` not becoming a byte source

Focused checks:

```sh
cargo test -p surgeist-render image
cargo test -p surgeist-render filter
cargo test -p surgeist-render reference
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Drop-Shadow Filter From Alpha Mask

Worker scope:

- Implement materialized-image `drop-shadow(...)` filter execution.
- Build the shadow from the current input alpha mask after previous filters in
  the ordered chain.
- Apply offset, blur, and solid color.
- Reject non-zero `Shadow::spread` for CSS drop-shadow filters with a typed
  diagnostic unless a separate non-CSS render extension is introduced in a
  later plan.
- Composite the shadow behind the current source using premultiplied source-over
  semantics.
- Distinguish drop-shadow alpha semantics from box-shadow geometry semantics in
  tests.
- Preserve following filter order: filters after drop-shadow apply to the
  composed shadow-plus-source output.
- Keep non-solid shadow paints rejected with the existing typed diagnostic.
- Add tests for:
  - alpha-shaped shadow from a non-rectangular transparent source
  - offset and blur bounds
  - shadow behind source ordering
  - `FilteredImagePaint` plus matching `Image` executing drop-shadow and
    preserving resource/size validation
  - resource-only drop-shadow filtered image paint remaining rejected without
    materialized bytes
  - drop-shadow followed by color filter
  - color filter followed by drop-shadow
  - non-zero spread rejected for CSS drop-shadow
  - non-solid drop-shadow paint diagnostic

Focused checks:

```sh
cargo test -p surgeist-render filter
cargo test -p surgeist-render image
cargo test -p surgeist-render shadow
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 6. Box Shadow Surface Completion

Worker scope:

- Reconcile existing `Shadow`/`Scene::shadow(...)` behavior with the matrix rows
  for outer, inset, multiple, rounded-corner, and non-solid shadows.
- Preserve current supported outer solid rect/rounded/circle shadows.
- Add or refine render-owned models only where they make missing semantics
  explicit, such as:
  - ordered `ShadowList`
  - explicit outer/inset shadow kind
  - explicit inset-shadow unsupported diagnostic if inset execution is not
    implemented in this phase
- If inset shadow execution is implemented, it must include clipped inner blur
  and rounded-corner tests. If not, the diagnostic must be typed and tested.
- Multiple shadows may be represented as ordered repeated commands if that is
  the crate-owned boundary; add tests proving ordering and overlap behavior.
- Keep non-solid shadow paint rejected unless a real rasterization path is
  implemented.
- Add tests for:
  - outer box-shadow offset, blur, spread, and negative/positive offset
  - rounded corners, including non-uniform radii
  - multiple shadow ordering
  - inset shadow execution or typed diagnostic
  - non-solid shadow paint diagnostic remains explicit

Focused checks:

```sh
cargo test -p surgeist-render shadow
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 7. Text Shadow Boundary

Worker scope:

- Reconcile the text-shadow matrix row with the current text model.
- Prefer a render-owned typed model, such as a text run plus ordered shadow
  list, only if it can be validated without borrowing root/private state.
- If text-shadow execution is implemented, it must draw shadows behind text and
  must support blur semantics through the same pixel-moving policy as other
  shadows.
- If text-shadow execution is not implemented in this phase, add a typed
  diagnostic that names text-shadow specifically and explain the remaining
  dependency on glyph-alpha/offscreen text capture.
- Do not alter ordinary `TextRun` rendering behavior.
- Add tests for:
  - text-shadow model validation or diagnostic
  - ordering behind text if execution exists
  - ordinary text runs remaining unaffected
  - blurred text-shadow diagnostic or execution behavior

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render shadow
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 8. Integration Guardrails And Sequence 11 Reconciliation

Worker scope:

- Add integration tests that verify Sequence 11 satisfies the matrix rows for:
  - blur
  - filter region/outsets
  - drop-shadow filter
  - blur/drop-shadow filtered image paint with matching materialized `Image`
  - outer box shadow
  - inset box shadow execution or typed diagnostic
  - multiple shadows
  - text shadow execution or typed diagnostic
  - non-solid shadow diagnostics
- Confirm `Capabilities::VELLO_0_9` exposes only the pixel-moving and shadow
  support that actually exists after this phase.
- Confirm Sequence 10 color-filter tests still pass and mixed color/pixel filter
  order remains correct.
- Confirm Sequence 9 offscreen tests still pass.
- Keep masks and backdrop unsupported.
- Update crate-local plan notes only if implementation discovers a real boundary
  or blocker.

Focused checks:

```sh
cargo test -p surgeist-render sequence11
cargo test -p surgeist-render filter
cargo test -p surgeist-render image
cargo test -p surgeist-render shadow
cargo test -p surgeist-render offscreen
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
- the full git diff for the Sequence 11 implementation

The final reviewer must confirm:

- blur execution is deterministic and has explicit region/outset semantics
- drop-shadow uses alpha-mask semantics rather than box geometry semantics
- ordered chains preserve color/blur/drop-shadow order
- CPU/reference golden coverage exists for blur
- shadow rows are either implemented or have typed diagnostics
- broad layer-filter, mask, and backdrop support are not claimed early
- public APIs remain intentional and crate-owned
- no sibling crates or root submodule pointers were edited

Run these final checks before committing the completed implementation:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```
