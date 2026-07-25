# Color Filter Pipeline Implementation Plan

Date: 2026-07-09

## Scope

Implement Sequence 10 from
`plans/2026-07-09-render-css-implementation-sequence.md`: the render-local
color-filter pipeline for color-only CSS filter operations.

Source matrix:

- `plans/2026-07-08-render-css-support-matrix.md`

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

This phase is render-local. It must not add sibling crate dependencies, edit
sibling crates, update root submodule pointers, or implement blur, drop-shadow,
URL/SVG filter graphs, masks, backdrop capture, or layer-filter execution.
Backwards compatibility shims are not required.

## Existing Inputs

The crate already owns these relevant pieces:

- `FilterList`, `FilterOp`, and `FilterOpKind` in `src/style.rs`.
- Color filter amount wrappers:
  - `FilterAmount` for non-negative brightness, contrast, and saturate values.
  - `UnitFilterAmount` for clamped grayscale, invert, opacity, and sepia
    values.
  - `FilterAngle` for finite hue-rotate angles.
- `FilteredImagePaint` as a resolved-resource plus ordered-filter intent.
- `Image` in `src/image.rs`, which stores validated straight RGBA8 bytes for
  already materialized render-local image inputs.
- Sequence 9 offscreen infrastructure:
  - `OffscreenPipelineCapabilities`
  - `TextureDescriptor` and `OffscreenTextureCache`
  - `RectShaderPassDescriptor`
  - `ReferencePremultipliedRgba8Buffer`
  - offscreen Vello render helpers in `src/backend.rs`

Sequence 10 should build on these models instead of replacing them. The
resource-only `FilteredImagePaint` model must remain honest: a resolved image
resource identifier is not itself a byte store. Actual execution in this phase
requires a render-local `Image` or `ImageBuffer` paired with a color-filter
list.

## Target Design

Add a typed, order-preserving color-filter pipeline with deterministic CPU
reference behavior and a clear future GPU pass boundary.

The pipeline should:

- Accept only color-only operations:
  - brightness
  - contrast
  - grayscale
  - hue rotate
  - invert
  - opacity
  - saturate
  - sepia
- Reject or defer pixel-moving and graph operations:
  - blur
  - drop shadow
  - URL/SVG/reference filters when those are modeled later
- Preserve authored filter order.
- Compile compatible color filters into one render-local fused transform when
  possible.
- Apply color filters to real RGBA image bytes when the image pipeline has
  already produced an `Image` or `ImageBuffer` plus filter list.
- Convert explicitly at the image execution boundary: straight RGBA8 image
  bytes become premultiplied RGBA8 reference/filter pixels before color math,
  then convert back to straight RGBA8 bytes for the produced image output.
- Keep broad `LayerFilter`, broad `FilterExecution`, blur, drop-shadow, masks,
  and backdrop capabilities unsupported until their later sequence items.

The first production-capable implementation may use the CPU/reference path for
color-filter execution if the capability model names that path honestly. A GPU
shader path is allowed only if it is complete enough to pass the same pixel
tests and does not overstate unsupported filter behavior.

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

### 1. Color Filter Capabilities, Diagnostics, And Classification

Worker scope:

- Add precise capability and diagnostic names for color-filter execution.
- Do not mark broad `FilteredImagePaint`, `LayerFilter`, or full
  `FilterExecution` support true unless the operation is narrowed to a
  color-only execution path.
- Prefer one of these modeling approaches:
  - Add a granular `ColorFilterPipelineCapabilities` or focused fields under
    the existing filter/offscreen capability models.
  - Add explicit operation names such as color-filter execution and
    color-filtered image paint, while leaving broad filtered-image and layer
    filter operations unsupported.
- Add a typed classifier for `FilterList` that returns an ordered
  color-filter-only pipeline or a typed unsupported primitive for the first
  non-color operation.
- The accepted operations are:
  - `FilterOpKind::Brightness`
  - `FilterOpKind::Contrast`
  - `FilterOpKind::Grayscale`
  - `FilterOpKind::HueRotate`
  - `FilterOpKind::Invert`
  - `FilterOpKind::Opacity`
  - `FilterOpKind::Saturate`
  - `FilterOpKind::Sepia`
- The rejected operations in this phase are:
  - `FilterOpKind::Blur`
  - `FilterOpKind::DropShadow`
- Preserve `FilterList::none()` as no pipeline and preserve
  `FilteredImagePaint::try_new` rejection of `none`.
- Add tests for:
  - accepted color-only lists preserving order
  - `none` producing no executable color pipeline
  - blur rejected with a typed blur/filter diagnostic
  - drop-shadow rejected with a typed drop-shadow/filter diagnostic
  - capabilities remaining honest for broad layer filters and full filter
    execution

Focused checks:

```sh
cargo test -p surgeist-render filter
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 2. Deterministic Color Filter Reference Math

Worker scope:

- Add deterministic CPU color-filter math, either in `src/reference.rs` or a
  small focused module used by `src/reference.rs`.
- Define and test the pixel policy explicitly:
  - Inputs are premultiplied RGBA8 for reference-buffer execution.
  - Color operations unpremultiply non-transparent pixels to straight color,
    apply the operation, clamp to `0..=1`, then premultiply by the current
    alpha.
  - Transparent pixels remain transparent.
  - `opacity()` scales alpha and premultiplied color channels at its position in
    the ordered filter chain.
- Implement reference behavior for:
  - brightness amount `0`, `1`, and greater than `1`
  - contrast amount `0`, `1`, and greater than `1`
  - grayscale amount `0`, partial, and `1`
  - hue-rotate identity, positive angle, negative angle, and full-turn identity
  - invert amount `0`, partial, and `1`
  - opacity amount `0`, partial, and `1`
  - saturate amount `0`, `1`, and greater than `1`
  - sepia amount `0`, partial, and `1`
- Use stable byte rounding so pixel tests do not depend on platform floating
  point formatting or GPU behavior.
- Add tests for:
  - identity filters preserving pixels byte-for-byte
  - partial filters producing expected bytes
  - extreme amounts clamping without invalid premultiplied pixels
  - transparent and partially transparent pixels preserving premultiplied
    invariants

Focused checks:

```sh
cargo test -p surgeist-render reference
cargo test -p surgeist-render filter
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 3. Fused Color Filter Pipeline Model

Worker scope:

- Add a render-owned compiled/fused color-filter model.
- The model must be phase-specific: it represents an executable color-only
  filter pipeline, not an authored CSS filter list and not a layer-filter graph.
- The model must preserve enough information to prove order. A matrix or fused
  transform is acceptable only if ordered chains that are not mathematically
  commutative still produce distinct results where CSS requires them.
- Implement fusion for compatible color filters into one executable transform
  when possible.
- If one exact matrix representation is used, include alpha/opacity behavior in
  the representation or sequence it explicitly so filter-order semantics remain
  correct.
- If exact fusion is not safe for an operation, keep that operation in the
  compiled pipeline as an ordered step and document the reason in the type or
  constructor.
- Add APIs that apply the compiled pipeline to:
  - a single `PremultipliedRgba8`
  - a `ReferencePremultipliedRgba8Buffer`
- Add tests for:
  - chain equivalence between per-op application and compiled pipeline
  - order sensitivity, for example contrast followed by brightness versus
    brightness followed by contrast
  - opacity order interacting correctly with color operations
  - an empty compiled pipeline being unconstructable except through an explicit
    identity/no-op value if such a value is needed internally

Focused checks:

```sh
cargo test -p surgeist-render filter
cargo test -p surgeist-render reference
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 4. Resolved Image Color-Filter Execution Boundary

Worker scope:

- Add an internal render-local execution boundary for already materialized image
  data plus a color-filter pipeline.
- The boundary may wrap `Image` or `ImageBuffer`, but it must not pretend that a
  `ResolvedImageResource` alone contains bytes.
- Model pixel phases explicitly at this boundary:
  - `Image` and `ImageBuffer` bytes are straight RGBA8.
  - reference/filter execution uses premultiplied RGBA8.
  - conversion helpers must be named for their direction and keep transparent
    pixels deterministic.
- Validate image byte length using existing `Image::from_rgba` and
  `ImageBuffer` size conventions.
- Execute color-only filtered image paint by applying the compiled pipeline to
  the converted premultiplied pixels and producing a new render-local image or
  image buffer with straight RGBA8 bytes suitable for later paint/upload.
- Preserve image identity semantics intentionally:
  - if a new `Image` is produced, its stable hash must reflect filtered bytes
  - if an `ImageBuffer` is produced, its size and RGBA byte order must be
    explicit
- Reject non-color filter lists before image execution.
- Keep resource-only `FilteredImagePaint::ensure_supported` from overclaiming
  execution unless there is a typed color-only route with materialized bytes.
- Add tests for:
  - applying a color-only filter chain to a 1x1 image
  - applying a color-only filter chain to a multi-pixel image
  - straight-to-premultiplied-to-straight conversion for transparent and
    partially transparent pixels
  - preserving image size and byte order
  - changing image identity when filtered bytes change
  - rejecting blur and drop-shadow before image byte transformation

Focused checks:

```sh
cargo test -p surgeist-render image
cargo test -p surgeist-render filter
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

### 5. Integration Guardrails And Sequence 10 Reconciliation

Worker scope:

- Add integration tests that verify Sequence 10 satisfies the matrix rows for:
  - brightness
  - contrast
  - grayscale
  - hue rotate
  - invert
  - opacity filter
  - saturate
  - sepia
  - filter fusion
  - CPU/reference fallback for deterministic tests
- Add guardrail tests that later-sequence work remains unsupported:
  - blur execution
  - drop-shadow execution
  - layer filter execution
  - mask execution
  - backdrop execution
- Confirm `Capabilities::VELLO_0_9` exposes only the color-filter support that
  actually exists after this phase.
- Confirm direct Vello rendering, image upload, background/image sampling, and
  Sequence 9 offscreen tests still pass without routing every image through the
  color-filter path.
- Update any crate-local plan notes only if needed to record a real boundary or
  blocker discovered during implementation.

Focused checks:

```sh
cargo test -p surgeist-render sequence10
cargo test -p surgeist-render filter
cargo test -p surgeist-render image
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
- the full git diff for the Sequence 10 implementation

The final reviewer must confirm:

- color-only filter execution is typed and order-preserving
- filter fusion does not erase CSS order semantics
- deterministic CPU/reference pixel tests cover identity, partial, and extreme
  amounts
- color-filtered image execution uses materialized image bytes, not unresolved
  resource handles
- broad filter, layer-filter, blur, drop-shadow, mask, and backdrop support are
  not claimed early
- public APIs remain intentional and crate-owned
- no sibling crates or root submodule pointers were edited

Run these final checks before committing the completed implementation:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```
