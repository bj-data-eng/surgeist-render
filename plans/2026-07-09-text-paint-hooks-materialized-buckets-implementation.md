# Text Paint Hooks And Materialized Paint Buckets Implementation

**Goal:** Implement Sequence 14 from
`plans/2026-07-09-render-css-implementation-sequence.md`: text paint hooks
and materialized generated/selection paint buckets.

**Architecture:** Render continues to consume already-shaped glyph runs and
already-materialized command streams. It does not own text shaping, CSS
pseudo-element construction, selection range calculation, or generated-content
materialization. Sequence 14 should expose render-owned models and execution
paths for concrete glyph paint, simple decoration geometry, and executable
text-shadow subsets, while keeping unsupported glyph-alpha capture or CSS text
semantics behind typed diagnostics.

**Tech Stack:** Rust 2024, current Vello text encoding, render-owned `Paint`,
`TextRun`, layer/offscreen, Sequence 10/11 materialized filters/shadows, and
crate-local unit/render tests.

## Inputs And Boundaries

- Matrix:
  - `plans/2026-07-08-render-css-support-matrix.md`, section 11.
- Sequence:
  - `plans/2026-07-09-render-css-implementation-sequence.md`, Sequence 14.
- Guidance:
  - `AGENTS.md`
  - `guidance/surgeist-rust-modeling-guide.md`

Render owns:

- validation and encoding of already-shaped glyph runs
- concrete text fill paint that can be represented by current `Paint`
- render-owned simple decoration commands or geometry supplied by root/text
- materialized text-shadow execution only where the required glyph/alpha pixels
  and filter path are explicit and tested
- tests proving selection/generated buckets are normal command streams once root
  materializes them

Render does not own:

- CSS text shaping, bidi, font fallback, line layout, or glyph lookup
- runtime selection ranges, highlight inheritance, or pseudo-element generation
- `currentColor`, system-color, or symbolic color resolution beyond the existing
  root-resolved concrete color policy
- root stacking tree ownership

Backwards compatibility shims are not required for this phase.

## Existing State

- `TextRun` already stores shaped glyphs, `FontRef`, size, transform, and
  `TextPaint`.
- `TextPaint` currently wraps a fill `Paint`.
- Vello encoding can draw text runs when `FontRef` carries `FontData`.
- `TextShadowRun` exists as a model but currently normalizes to an unsupported
  `Shadows / TextShadow` diagnostic.
- Matrix rows for selection and generated content are render buckets: once root
  materializes them as commands, render should treat them as ordinary fills,
  layers, images, and text runs.

## Tasks

Each task below must follow the crate coordinator workflow from `AGENTS.md`:
assign one scoped worker, have a separate clean-context reviewer inspect the
worker change, reconcile findings, rerun the focused checks, and commit the
task as a traceable logical point before moving to the next task. Workers do
not commit.

### 1. Text Capability And Diagnostic Contract

Worker scope:

- Audit and, if needed, tighten capability names for:
  - glyph fill paint
  - text decoration paint/geometry
  - text shadow execution
  - selection/generated materialized command buckets
- Keep unsupported text-shadow cases behind `Shadows / TextShadow` unless a more
  precise existing operation already fits.
- Add tests proving:
  - ordinary text runs remain separate from text-shadow diagnostics
  - current text-shadow capability claims match the current diagnostic boundary
  - selection/generated buckets do not need special root-owned render capability
    once represented as command streams

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Glyph Fill Paint Reconciliation

Worker scope:

- Confirm `TextPaint` accepts the concrete `Paint` surface supported by Sequence
  5 without widening symbolic-color ownership.
- Add render tests for text fill paint where feasible with embedded font data:
  - solid concrete color
  - gradient or image paint only if Vello encoding and current `Paint` support it
    for glyph brushes without extra policy
- If a paint kind cannot be encoded for glyphs, reject it with a typed
  diagnostic instead of accepting and failing late.
- Preserve existing font-data-required diagnostics for render-time text drawing.

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render paint
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Text Decoration Paint And Geometry Hooks

Worker scope:

- Add a render-owned decoration input only for already-resolved decoration
  geometry or a simple line decoration command whose dimensions are supplied by
  root/text.
- Preserve CSS semantics outside render:
  - underline position
  - line-through metrics
  - skip-ink
  - text-decoration-style expansion beyond what render explicitly models
- Support decoration paint through existing `Paint`/stroke/fill primitives when
  geometry is explicit.
- Add tests for decoration color/paint, thickness, transform, and ordering
  relative to text.
- Add typed diagnostics for decoration variants not modeled.

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render stroke
cargo test -p surgeist-render paint
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Materialized Selection And Generated Content Buckets

Worker scope:

- Add tests or small helper models proving selection and generated content are
  ordinary render command streams after root materializes them.
- Cover:
  - selection highlight background plus selected glyph foreground
  - generated text/content rendered in command order
  - list-marker/image-style generated content as ordinary image/text commands if
    current primitives already support it
- Avoid adding CSS pseudo-element or runtime selection APIs to render.

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render background
cargo test -p surgeist-render image
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Text Shadow Planning Boundary

Worker scope:

- Split `TextShadowRun` handling into explicit supported and unsupported cases.
- Decide, based on the current renderer:
  - execute zero-blur solid text shadows by repeated shifted text draws if this
    preserves CSS ordering and paint constraints, or
  - keep all text shadows diagnostic until glyph-alpha capture is implemented.
- For nonzero blur, either:
  - implement a materialized glyph-alpha/readback path using Sequence 9/10/11
    primitives, or
  - retain the current typed diagnostic with tests explaining that glyph-alpha
    capture is still the missing boundary.
- Add tests for shadow ordering behind text and for unsupported blur/glyph-alpha
  cases.

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render shadow
cargo test -p surgeist-render filter
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 6. Materialized Text Shadow Execution

Worker scope:

- If Task 5 chooses execution for any text-shadow subset, implement it here:
  - preserve authored shadow order
  - draw shadows behind glyph fill
  - respect text transform and layer transform
  - use current shadow/filter machinery only for cases it can represent
  - return stable adapter diagnostics when GPU readback is required but
    unavailable
- If no text-shadow subset is executable without overreach, add guardrail tests
  proving why the current diagnostic remains honest and root-facing.

Focused checks:

```sh
cargo test -p surgeist-render text
cargo test -p surgeist-render shadow
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 7. Sequence 14 Integration Guardrails

Worker scope:

- Add `sequence14` tests proving the matrix rows are either implemented or
  explicitly diagnostic:
  - glyph fill paint
  - text decoration paint
  - text shadow
  - selection paint bucket
  - generated content paint bucket
- Confirm capabilities advertise exactly the text behavior implemented and no
  root-owned selection/generated/text-shaping behavior.
- Confirm text-shadow capability claims match the executable subset or
  diagnostic boundary chosen by Tasks 5 and 6.
- Add plan notes only for real root-facing boundaries discovered during
  implementation.

Focused checks:

```sh
cargo test -p surgeist-render sequence14
cargo test -p surgeist-render text
cargo test -p surgeist-render shadow
cargo test -p surgeist-render paint
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

## Final Review And Required Checks

After all scoped tasks are implemented, assign a final clean-context holistic
reviewer to inspect the complete Sequence 14 result against:

- this implementation plan
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- the full git diff for the Sequence 14 implementation

The final reviewer must confirm:

- render does not take ownership of shaping, selection ranges, or generated
  content construction
- glyph fill paint uses only supported `Paint` behavior
- decoration hooks are geometry/paint hooks, not CSS text-layout ownership
- text-shadow execution is implemented only where fully modeled and tested, or
  remains typed diagnostic
- selection and generated buckets are ordinary command streams after root
  materialization
- public APIs remain intentional and crate-owned
- no sibling crates or root submodule pointers were edited

Run these final checks before declaring Sequence 14 complete:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```
