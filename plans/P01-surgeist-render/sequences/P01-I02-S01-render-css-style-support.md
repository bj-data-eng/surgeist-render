# Render CSS/Style Implementation Sequence

Date: 2026-07-09

## Goal

Execute the render CSS/style primitive support matrix through a sequence of
reviewed, implementation-sized phases. Each sequence item becomes its own
crate-local implementation plan before code changes begin.

Source matrix:

- `plans/2026-07-08-render-css-support-matrix.md`

Required standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Workflow Contract

For each sequence item:

1. Create a focused implementation plan in `plans/`.
2. Have a clean-context reviewer inspect the plan against this sequence,
   `AGENTS.md`, the support matrix, and the modeling guide.
3. Commit the plan only after review is clean.
4. Execute the committed plan through the coordinator workflow:
   - split the plan into sequential worker tasks
   - assign one implementation worker to one scoped task or tightly coupled
     task group at a time
   - use a separate reviewer for each worker result
   - workers do not commit
   - after each scoped worker/reviewer cycle is clean, run the focused check
     for that scoped task and let the coordinator commit that logical point
5. After all scoped tasks are complete, assign a final clean-context holistic
   reviewer for the sequence item implementation.
6. Run the final focused checks for the sequence item.
7. If final review or checks find issues, create follow-up scoped worker tasks
   and repeat the worker/reviewer/commit cycle before declaring the sequence
   item complete.

The sequence is complete only after all items are implemented, required checks
pass, and the final full-result holistic review is clean.

## Sequence

### 1. Capability And Diagnostic Foundation

Establish the primitive-family capability and error contract that every later
phase will use.

Matrix coverage:

- Diagnostics and capability reporting
- Backend pipeline requirements
- Unsupported primitive diagnostics
- Unresolved resource diagnostics
- Invalid value diagnostics

Deliverables:

- Replace the narrow capability surface with typed render capability families.
- Add typed unsupported/unresolved/degraded diagnostic variants.
- Preserve current Vello baseline behavior through explicit capability values.
- Add tests for capability defaults and representative diagnostics.

Dependencies: none.

### 2. Core Render Primitive Models

Introduce render-owned public models for CSS-facing paint/effect data without
adding lowering or backend behavior yet.

Matrix coverage:

- Paint sources
- CSS image layer model
- Filter list model
- Masks and clips model
- Box decoration, border, outline model
- Resource handles and images

Deliverables:

- Add private-field, constructor-validated types for color inputs, image layer
  inputs, filter operations, masks, clips, border sides, outlines, and
  background layers.
- Decide the symbolic color handoff policy before adding color front doors:
  if root-resolved-only, do not expose symbolic color payloads; if render
  accepts symbolic colors, use phase-specific render color types that name the
  required realization context.
- Keep symbolic or unresolved values explicit and phase-specific.
- Add construction and invalid-state tests.
- Do not add sibling crate dependencies.

Dependencies:

- Sequence 1 diagnostics and capabilities.

### 3. Geometry Target Normalization

Establish render-owned geometry target behavior before higher-level paint
features depend on it.

Matrix coverage:

- Rect fill/stroke
- Rounded rect fill/stroke
- Circle/ellipse fill/stroke
- Arbitrary path fill
- Arbitrary path centered stroke
- Arbitrary path inside/outside stroke diagnostics or expansion
- Geometry boolean/offset support diagnostics or implementation
- Hit-test geometry out-of-scope handling

Deliverables:

- Audit and normalize existing shape, path, stroke, dash, radius, and alignment
  models against the matrix.
- Add explicit diagnostics for geometry operations render will not implement in
  this phase.
- Preserve render-owned geometry front doors without adding sibling
  dependencies.
- Add tests for current direct geometry support and unsupported inside/outside
  path stroke behavior.

Dependencies:

- Sequence 1 diagnostics and capabilities.
- Sequence 2 primitive models.

### 4. Transform And Coordinate Space Normalization

Complete transform-origin and coordinate-space support needed by backgrounds,
image attachment, masks, and later effects.

Matrix coverage:

- 2D affine transforms
- Transform origin
- Skew
- 3D transform diagnostics
- Coordinate-space tagging

Deliverables:

- Normalize transform-origin into explicit transform sequences.
- Add public constructors or normalized inputs for matrix/skew where needed.
- Add diagnostics for unsupported 3D transforms.
- Add coordinate-space data needed for fixed backgrounds, masks, and backdrop
  capture.
- Add tests for transformed clips/images and for coordinate-space tags used by
  fixed backgrounds, masks, and future backdrop capture.

Dependencies:

- Sequence 1 diagnostics.
- Sequence 2 primitive models.
- Sequence 3 geometry target normalization.

### 5. Color Realization And Paint Source Normalization

Implement render-side color and paint-source normalization that can feed Vello
or later pipelines.

Matrix coverage:

- Solid RGBA paint
- Symbolic color token
- Paint-space color conversion
- Linear/radial/conic gradients
- Repeating gradients

Deliverables:

- Add a render-owned normalized paint layer representation.
- Implement the symbolic color handoff policy selected by Sequence 2.
- Normalize CSS-like gradient inputs to Vello/Peniko-compatible data or typed
  diagnostics.
- Add deterministic tests for conversion, invalid values, and gradient edge
  cases.

Dependencies:

- Sequence 2 primitive models.
- Sequence 3 geometry target normalization.

### 6. Image Resources And CSS Image Sampling

Implement resolved image-resource inputs and CSS image sampling semantics.

Matrix coverage:

- Image paint
- Resolved image handle
- Intrinsic image metadata
- Image fit
- Background position
- Background size
- Repeat/no-repeat/round/space
- Background attachment coordinate input
- Filtered image paint boundary and diagnostics
- Image orientation and color-profile ownership policy

Deliverables:

- Introduce render-owned image resource handles or resolved image descriptors.
- Normalize image placement from position/size/repeat/origin inputs.
- Support CSS repeat modes or provide precise diagnostics where unsupported.
- Define whether orientation and color-profile conversion are root-resolved or
  render-realized; add diagnostics or conversion hooks accordingly.
- Represent filtered image paint as a resolved image plus filter list, with
  execution deferred to the filter phases or rejected by typed diagnostics.
- Add tests for fit, intrinsic sizing, repeat modes, fixed/local attachment
  coordinate behavior or documented root-owned scroll adjustment, transformed
  image coordinates, and missing resource diagnostics.

Dependencies:

- Sequence 2 primitive models.
- Sequence 4 coordinate-space normalization.
- Sequence 5 paint normalization for mixed image/gradient layers.

### 7. Background Layer Stack

Implement layered background painting using the paint and image sampling
foundations.

Matrix coverage:

- Background layer stack
- Background origin
- Background clip
- Multi-layer image stack
- Background color behind images

Deliverables:

- Normalize background layers into ordered render commands.
- Support border-box, padding-box, and content-box inputs as root-supplied
  geometry.
- Support shape/path clips for layer clipping where existing backend paths can
  do so.
- Add tests for layer order, list matching, origin/clip geometry, and
  transparent layers.

Dependencies:

- Sequence 5 color and paint normalization.
- Sequence 6 image sampling.

### 8. Border, Outline, Radius, And Box Decoration Paint

Implement box decoration primitives that do not require offscreen effects.

Matrix coverage:

- Border side solid/none/hidden/dashed/dotted/double
- Groove/ridge/inset/outset diagnostics or color-band rendering
- Border radius clipping
- Outline solid/dashed/dotted/auto diagnostics
- Box decoration break with root-supplied fragments

Deliverables:

- Normalize border and outline inputs into render commands.
- Preserve side-specific widths, styles, colors, and radii.
- Add diagnostics for styles not yet rendered.
- Add tests for side independence, radii, outlines, and fragmented decoration
  inputs.

Dependencies:

- Sequence 3 geometry target normalization.
- Sequence 5 paint normalization.
- Sequence 7 background geometry conventions.

### 9. Offscreen Layer And Texture Pipeline

Add the render-local infrastructure required for filters, masks, blending, and
backdrop effects.

Matrix coverage:

- Offscreen layer renderer
- Texture cache/upload
- Fullscreen/rect shader pass foundation
- CPU/reference fallback foundation for deterministic filter and compositing tests
- Isolation group
- Layer opacity behavior

Deliverables:

- Render subtrees into offscreen textures with explicit bounds.
- Add texture allocation/cache lifecycle models.
- Add CPU/reference buffer and compositing helpers that later filter, mask,
  blend, and backdrop phases can use as deterministic test oracles.
- Preserve current direct Vello path when offscreen isolation is unnecessary.
- Add tests for nested layers, opacity, bounds, and resource reuse.

Dependencies:

- Sequence 1 capability/diagnostic foundation.
- Sequence 4 coordinate-space normalization.

### 10. Color Filter Pipeline

Implement color-only CSS filters as a fused render-local pipeline.

Matrix coverage:

- Brightness
- Contrast
- Grayscale
- Hue rotate
- Invert
- Opacity filter
- Saturate
- Sepia
- Filter fusion
- CPU/reference fallback for deterministic tests

Deliverables:

- Add color-filter shader or CPU/reference implementation.
- Preserve filter order.
- Fuse compatible color filters when possible.
- Execute color-only filtered image paint where the image phase produced a
  resolved image plus color-filter list.
- Add pixel/reference tests for identity, partial, and extreme amounts.

Dependencies:

- Sequence 2 filter model.
- Sequence 6 image sampling.
- Sequence 9 offscreen/shader-pass foundation.

### 11. Blur, Drop Shadow, And Shadow Completion

Implement pixel-moving effects and complete the shadow surface.

Matrix coverage:

- Blur
- Filter region/outsets
- Drop shadow filter
- Outer box shadow
- Inset box shadow
- Multiple shadows
- Text shadow
- Non-solid shadow diagnostics

Deliverables:

- Add separable blur or equivalent blur pipeline.
- Implement filter-region inflation and clipping.
- Implement drop-shadow from alpha mask.
- Complete box-shadow behavior, including inset or explicit diagnostics.
- Execute blur/drop-shadow filtered image paint where the image phase produced a
  resolved image plus pixel-moving filter list.
- Add CPU/reference golden coverage for blur so GPU variance does not hide
  filter-region or kernel regressions.
- Add tests for large-radius policy, bounds, rounded corners, shadow ordering,
  and drop-shadow alpha semantics.

Dependencies:

- Sequence 9 offscreen pipeline.
- Sequence 10 color filter pipeline only for shared filter-list execution.

### 12. Masks, Clip Paths, And Masked Composition

Implement alpha/path masking and clip-path lowering.

Matrix coverage:

- Shape clip
- Path clip
- Basic shape clip
- Clip URL/reference diagnostics
- Alpha mask
- Luminance mask diagnostics or implementation
- Multi-layer mask
- Mask composite diagnostics

Deliverables:

- Normalize basic shape clips into render-owned geometry.
- Support path and shape clips in offscreen composition.
- Implement alpha masks for resolved mask inputs.
- Add typed diagnostics for URL/reference and unsupported mask modes.
- Add CPU/reference oracle coverage for mask edges and mask compositing.
- Add tests for nested clips, mask alpha edges, repeated mask layers, and
  transformed masks.

Dependencies:

- Sequence 3 geometry target normalization.
- Sequence 6 image sampling.
- Sequence 9 offscreen pipeline.

### 13. Backdrop, Blend, And Advanced Compositing

Implement effect behavior that depends on previously rendered scene content.

Matrix coverage:

- Backdrop capture
- Backdrop filter chain
- Backdrop isolation
- Root backdrop policy diagnostics
- Mix blend mode
- Background blend mode diagnostics or implementation
- Porter-Duff/composite ops required by masks/effects

Deliverables:

- Capture backdrop regions into textures.
- Apply filter chains to backdrop textures.
- Composite filtered backdrop and foreground content in correct order.
- Expand blend-mode support or provide typed diagnostics per mode.
- Add CPU/reference oracle coverage for required Porter-Duff operators, blend
  behavior, and backdrop compositing order.
- Add tests for backdrop ordering, rounded clips, nested groups, and blend
  mode behavior.

Dependencies:

- Sequence 9 offscreen pipeline.
- Sequence 10 color filters.
- Sequence 11 blur.
- Sequence 12 masks/clips.

### 14. Text Paint Hooks And Materialized Paint Buckets

Support render-side text and materialized generated/selection paint inputs
without taking ownership of text shaping, runtime selection, or pseudo-element
materialization.

Matrix coverage:

- Glyph fill paint
- Text decoration paint
- Text shadow
- Selection paint bucket
- Generated content paint bucket

Deliverables:

- Allow text paint to use the expanded paint/color model where appropriate.
- Add render-side hooks for already-shaped decoration geometry or simple text
  decoration commands.
- Support text shadows through the completed shadow/filter pipeline.
- Document and test that generated content and selection are normal render
  command streams once root materializes them.

Dependencies:

- Sequence 5 paint normalization.
- Sequence 11 shadow completion.

### 15. Full Matrix Reconciliation And Root-Handoff Readiness

Close gaps, update docs, and prepare the final root-facing render contract.

Matrix coverage:

- All matrix rows
- Property cross-reference
- Review checklist

Deliverables:

- Audit every support matrix row and mark it supported, diagnostic, or deferred
  with a named reason.
- Ensure public APIs are intentional front doors from `lib.rs`.
- Remove obsolete APIs because backwards compatibility shims are not required.
- Add integration-style crate tests for representative full paint stacks.
- Produce the final root-handoff note with remaining explicit root
  responsibilities.

Dependencies:

- Sequences 1 through 14.

## Required Checks

Each implementation plan may add focused checks. The standing crate checks are:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Do not use broad feature matrices unless the implementation plan proves they
are valid for the current backend and target constraints.
