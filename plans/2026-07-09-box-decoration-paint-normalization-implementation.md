# Box Decoration Paint Normalization Implementation Plan

Date: 2026-07-09

## Scope

Implement Sequence 8 from
`plans/2026-07-09-render-css-implementation-sequence.md`: border, outline,
radius, and box-decoration paint primitives that do not require offscreen
effects.

Source matrix:

- `plans/2026-07-08-render-css-support-matrix.md`

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

This phase is render-local. It must not add a dependency on `surgeist-shape`,
edit sibling crates, or update root submodule pointers. Backwards
compatibility shims are not required.

## Existing Inputs

The crate already owns validated style-facing value objects:

- `BorderSide`, `BorderStyle`, and `BorderEdges`
- `Outline` and `OutlineStyle`
- `BackgroundAreas` and `BackgroundClipGeometry`
- `Radii`, `Rect`, `Shape`, `Paint`, and geometry validation helpers

Sequence 8 should reuse those models and add normalization outputs, capability
flags, and diagnostics. It should not introduce a parallel border or outline
input model unless the existing model cannot express the needed CSS contract.

## Target Design

Add a box-decoration normalization surface that converts root-supplied
decoration geometry into render commands:

- `BoxDecorationInput`
  - optional `BorderEdges`
  - optional `Outline`
  - one or more root-supplied `BoxDecorationFragment`s
- `BoxDecorationFragment`
  - `BackgroundAreas` for border, padding, and content boxes
  - authored `Radii` for the border box, normalized during input construction
    into clamped border-box radii
  - `BoxDecorationBreak` (`Slice` or `Clone`) to preserve root fragment
    semantics
  - optional border clip override when root has already produced arbitrary
    radius/path geometry
- `NormalizedBoxRadii`
  - clamped `Radii` plus the source border box used for clamping
  - applies the CSS corner scaling rule when adjacent horizontal or vertical
    radii exceed the border box dimensions
- `NormalizedBoxDecoration`
  - ordered `NormalizedBoxDecorationCommand`s
  - commands carry fragment index, clip geometry, paint, style, side, width,
    normalized radii, and target rect as appropriate

Normalize these supported styles:

- border `None` and `Hidden`: suppress paint commands
- border `Solid`: produce one side-specific paint command per non-zero side
- border `Dashed` and `Dotted`: produce side-specific commands with typed dash
  semantics that can later lower to Vello strokes or custom geometry
- border `Double`: produce side-specific commands with explicit outer band, gap,
  and inner band widths, preserving the original side width and normalized radii
- outline `None`: suppress paint commands
- outline `Solid`, `Dashed`, and `Dotted`: produce commands outside the border
  box without affecting layout

Add typed unsupported diagnostics for styles this phase does not render:

- border `Groove`
- border `Ridge`
- border `Inset`
- border `Outset`
- outline `Auto`
- outline `Double`

Do not implement color-band rendering for groove/ridge/inset/outset in this
phase. The diagnostic route keeps the model precise while later phases decide
whether those styles are root-resolved color transforms or render-owned bands.

## Task Sequence

Each scoped task must follow the AGENTS coordinator workflow before the next
task begins:

1. Assign one implementation worker for the scoped task.
2. Have a separate clean-context reviewer inspect that worker's changes.
3. Reconcile any findings with follow-up worker/reviewer cycles.
4. Run the focused checks listed for the task.
5. Commit the task only after the worker result, reviewer result, and focused
   checks are clean.

### 1. Box-Decoration Capabilities And Diagnostics

Worker scope:

- Add `BoxDecorationCapabilities` to `src/capability.rs`.
- Add `PrimitiveFamily::BoxDecorations`.
- Add precise `PrimitiveOperation` variants for unsupported decorative styles:
  - `BorderGrooveStyle`
  - `BorderRidgeStyle`
  - `BorderInsetStyle`
  - `BorderOutsetStyle`
  - `OutlineDoubleStyle`
  - `OutlineAutoStyle`
- Set `Capabilities::VELLO_0_9` to support solid, none/hidden,
  dashed/dotted, double, radii, outlines, and fragments, and to reject the
  unsupported style operations above.
- Export the new capability type in `src/lib.rs`.
- Add tests for capability accessors and representative unsupported diagnostics.

Focused checks:

```sh
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Fragment And Input Models

Worker scope:

- Add `BoxDecorationBreak`, `NormalizedBoxRadii`, `BoxDecorationFragment`, and
  `BoxDecorationInput` to `src/style.rs`.
- Require at least one fragment.
- Validate fragment geometry by reusing `BackgroundAreas`, `Radii`, and
  `BackgroundClipGeometry`.
- Normalize fragment radii by clamping/scaling corner radii against the
  fragment border box when adjacent horizontal or vertical corners exceed the
  available width or height.
- Preserve optional `BorderEdges` and `Outline` rather than replacing the
  existing input types.
- Export the new models in `src/lib.rs`.
- Add tests for construction, empty-fragment rejection, break-mode preservation,
  radius clamping/scaling, background-clip geometry interaction, and clip
  override preservation.

Focused checks:

```sh
cargo test -p surgeist-render box_decoration
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Border Normalization

Worker scope:

- Add normalized border command types in `src/style.rs`:
  - `BoxSide`
  - `NormalizedBorderStyle`
  - `NormalizedDoubleBorderBands`
  - `NormalizedBorderCommand`
  - `NormalizedBoxDecorationCommandKind::Border`
- Implement `BoxDecorationInput::normalize(capabilities)` for border commands.
- Emit commands in deterministic fragment order and side order: top, right,
  bottom, left.
- Preserve side-specific width, paint, style, radii, target rect, fragment
  index, clip, and `BoxDecorationBreak`.
- Suppress zero-width, `None`, and `Hidden` border sides.
- Map `Solid`, `Dashed`, `Dotted`, and `Double` to supported normalized style
  variants.
- Normalize `Double` into render-ready band data:
  - preserve the original CSS border width
  - compute non-negative outer band, gap, and inner band widths
  - ensure the sum of the three bands equals the original side width
  - keep thin borders deterministic rather than dropping paint accidentally
- Return typed unsupported diagnostics for `Groove`, `Ridge`, `Inset`, and
  `Outset`.
- Add tests for four independent sides, none/hidden suppression, zero-width
  suppression, dashed/dotted style preservation, double-band normalization for
  thin/medium/large widths with radii, unsupported styles, and multiple
  fragments.

Focused checks:

```sh
cargo test -p surgeist-render box_decoration
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Outline Normalization

Worker scope:

- Add normalized outline command types in `src/style.rs`:
  - `NormalizedOutlineStyle`
  - `NormalizedOutlineCommand`
  - `NormalizedBoxDecorationCommandKind::Outline`
- Extend `BoxDecorationInput::normalize(capabilities)` to emit outline
  commands after border commands for each fragment.
- Preserve outline width, paint, offset, style, fragment index, target rect,
  clip, radii, and `BoxDecorationBreak`.
- Suppress zero-width and `None` outlines.
- Expand the border box by outline offset to produce the outline target rect;
  keep width as a separate stroke/band value so layout remains root-owned.
- Map `Solid`, `Dashed`, and `Dotted` to supported normalized style variants.
- Return typed unsupported diagnostics for `Double` and `Auto`.
- Add tests for outside-border offset geometry, non-layout behavior,
  dashed/dotted style preservation, double and auto diagnostics, zero-width
  suppression, and per-fragment outlines.

Focused checks:

```sh
cargo test -p surgeist-render box_decoration
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Integration Guardrails And Final Review

Worker scope:

- Add integration tests that combine background normalization conventions with
  box-decoration normalization:
  - background areas are reused for border-box geometry
  - border radii and clip overrides are preserved
  - `BoxDecorationBreak::Clone` and `BoxDecorationBreak::Slice` remain explicit
    on commands
  - commands remain deterministic across multiple fragments
- Do not add backend/offscreen rendering behavior in this phase.

Focused checks:

```sh
cargo test -p surgeist-render box_decoration
cargo test -p surgeist-render background
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

After all scoped tasks are committed, assign a final clean-context holistic
reviewer to inspect the complete Sequence 8 result against this plan,
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
- No Vello backend draw lowering for box decoration commands.
- No offscreen effects, filters, blend isolation, or shadow work.
- No groove/ridge/inset/outset color-band rendering.
- No `OutlineStyle::Double` rendering; normalization must reject it with the
  typed `OutlineDoubleStyle` unsupported diagnostic because the Sequence 8
  matrix only covers solid, dashed, dotted, and auto outline behavior.
- No backwards compatibility shims.
