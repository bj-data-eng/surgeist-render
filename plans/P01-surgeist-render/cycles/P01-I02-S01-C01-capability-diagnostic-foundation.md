# Capability And Diagnostic Foundation Implementation Plan

Date: 2026-07-09

Sequence item: `plans/2026-07-09-render-css-implementation-sequence.md`
item 1, "Capability And Diagnostic Foundation".

Support matrix source:

- `plans/2026-07-08-render-css-support-matrix.md`

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Goal

Replace the current narrow renderer capability and unsupported-error surface
with a primitive-family capability and diagnostic foundation that later CSS
paint/effect phases can extend without broad enum bags or scattered string
checks.

This plan must preserve current Vello 0.9 behavior. It should not implement new
paint, filter, mask, background, or compositor behavior.

## Scope

In scope:

- Capability categories for render primitive families.
- Typed unsupported primitive diagnostics.
- Typed invalid-value diagnostics.
- Typed unresolved resource diagnostics.
- Typed degraded-quality diagnostics.
- Tests for Vello baseline capability values and representative diagnostics.
- Updates to existing code paths that currently call `Capabilities::ensure`.

Out of scope:

- CSS/style primitive models.
- Symbolic color APIs.
- Resource handles.
- Any new rendering behavior.
- Sibling crate dependencies.
- Root submodule pointer updates.
- Backwards compatibility shims for obsolete capability names.

## Current State

Relevant files:

- `src/capability.rs`
- `src/error.rs`
- `src/command.rs`
- `src/renderer.rs`
- `src/lib.rs`
- `src/tests.rs`

Current capability support is limited to:

- layer masks
- layer filters
- inside/outside path strokes
- web canvas surfaces
- non-solid shadow paint as an always-unsupported ad hoc capability

Current diagnostic support is limited to `UnsupportedCapability` plus broad
`ErrorCode::UnsupportedBackend`.

## Modeling Requirements

- Keep capability concepts phase-specific and render-owned.
- Use typed identifiers for primitive families rather than string-only labels.
- Keep invalid or unsupported states observable through stable constructors and
  typed errors.
- Avoid one broad enum that mixes unsupported primitives, unresolved resources,
  quality degradation, and backend lifecycle failures.
- Keep public front doors intentional through `lib.rs`.
- Do not add dependencies on `surgeist-style`, `surgeist-css`,
  `surgeist-shape`, root, or runtime.

## Target Model

Workers may adapt exact names if review finds a better local fit, but the
implementation should converge on these concepts:

### Capability Report

`Capabilities` should become a render capability report with grouped query
methods. It may remain a concrete struct with private fields, but the public
queries should name primitive families such as:

- geometry targets
- paint sources
- image sampling
- backgrounds/box decoration
- filters
- masks/clips
- compositing/backdrop
- text paint hooks
- surfaces

The report must expose the current Vello 0.9 baseline as an explicit constant
or constructor. The baseline should continue to reject currently unsupported
features:

- layer masks
- layer filters
- inside/outside path stroke alignment
- non-solid shadow paint
- web canvas surfaces on unsupported targets/configurations, while preserving
  current `render-web` plus `wasm32` support

### Unsupported Diagnostics

Add a typed unsupported primitive diagnostic model. It should identify at least:

- the unsupported primitive family
- the specific operation or feature where useful
- a stable label for error messages

Existing normalization paths should return this typed unsupported diagnostic,
not construct ad hoc `UnsupportedBackend` messages.

### Invalid Value Diagnostics

Review and strengthen the existing invalid-value diagnostic path. The current
`Error::invalid_value(...)` helper already names a rejected field, value, and
rule. This phase should make that contract explicit and typed enough for later
primitive constructors to reuse without falling back to ad hoc strings.

The diagnostic should identify at least:

- the rejected field or semantic value
- the invalid value text
- the violated invariant
- `ErrorCode::InvalidInput` unless a narrower code is introduced

This phase should not rewrite every constructor in the crate. It should provide
the typed front door and update representative call sites/tests so later phases
have a stable pattern.

### Unresolved Resource Diagnostics

Add a typed unresolved-resource diagnostic model for future URL/image/mask/filter
resource handoffs. This phase does not add resource handles, but it must provide
the diagnostic type and constructors so later phases have a stable target.

Minimum resource kinds:

- image
- mask
- filter
- clip
- font or text paint resource only if needed by existing render terms

### Degraded Quality Diagnostics

Add a typed degraded-quality diagnostic model for future fast blur clamps,
software fallback, unsupported wide gamut conversion, or similar render-visible
quality changes.

This phase may expose it as a warning/statistics-friendly type without changing
render execution. It must be testable as a stable value and message.

## Worker Task Sequence

The coordinator must assign one task or tightly coupled task group at a time.
Workers must not commit. Workers are not alone in the codebase and must not
revert unrelated changes.

### Task 1: Capability Families And Unsupported Primitive Diagnostics

Files:

- `src/capability.rs`
- `src/error.rs`
- `src/command.rs`
- `src/renderer.rs`
- `src/lib.rs`
- `src/tests.rs`

Steps:

1. Add focused tests that assert the Vello baseline reports support and
   non-support for the primitive families needed by the matrix, including the
   cfg-dependent web canvas surface behavior.
2. Add tests for typed unsupported primitive diagnostics, including existing
   layer mask, layer filter, path stroke alignment, web canvas surface, and
   non-solid shadow paint cases.
3. Replace the narrow private fields with family-level capability fields or
   small nested structs with private fields.
4. Add a typed unsupported primitive value that names primitive family and
   operation.
5. Add an error constructor that preserves `ErrorCode::UnsupportedBackend` for
   backend-incompatible render requests while making the unsupported primitive
   inspectable through type and message.
6. Update direct callers to use the new capability query/ensure API and typed
   unsupported primitive path in the same logical commit.
7. Keep compatibility only where still semantically current; do not preserve
   obsolete public methods just to avoid breakage.

Expected tests:

```sh
cargo test -p surgeist-render capabilities
cargo test -p surgeist-render unsupported
```

### Task 2: Invalid Value Diagnostics

Files:

- `src/error.rs`
- `src/tests.rs`
- `src/lib.rs`

Steps:

1. Add tests for invalid-value diagnostics that cover non-finite values,
   impossible geometry, and empty-list style invariants through representative
   existing constructors.
2. Add a typed invalid-value diagnostic value or equivalent private-field
   struct that captures field, value text, and invariant without exposing
   invalid construction.
3. Update `Error::invalid_value(...)` or add a new constructor so existing
   callers can use the typed path without a crate-wide rewrite.
4. Ensure representative existing invalid-input tests still assert stable
   messages and `ErrorCode::InvalidInput`.

Expected tests:

```sh
cargo test -p surgeist-render invalid_value
```

### Task 3: Unresolved Resource Diagnostics

Files:

- `src/error.rs`
- `src/tests.rs`
- `src/lib.rs`

Steps:

1. Add tests for unresolved image, mask, filter, and clip resource diagnostics.
2. Add a typed unresolved resource kind and diagnostic value.
3. Add an error constructor that uses an appropriate stable `ErrorCode`.
   Prefer a new code if existing codes would blur unresolved caller input with
   backend failure.
4. Keep this phase diagnostic-only; do not add resource loading or handles.

Expected tests:

```sh
cargo test -p surgeist-render unresolved_resource
```

### Task 4: Degraded Quality Diagnostics

Files:

- `src/error.rs`
- `src/stats.rs`
- `src/tests.rs`
- `src/lib.rs`

Steps:

1. Add tests for degraded-quality diagnostic construction and labels.
2. Add typed degraded-quality kind/value for fast blur clamp, software fallback,
   unsupported paint-space conversion, and similar future quality paths.
3. Expose the type through `lib.rs`.
4. Do not change rendering behavior in this phase.

Expected tests:

```sh
cargo test -p surgeist-render degraded
```

### Task 5: Integration Pass And Cleanup

Files:

- `src/capability.rs`
- `src/error.rs`
- `src/command.rs`
- `src/renderer.rs`
- `src/tests.rs`
- `src/lib.rs`

Steps:

1. Remove obsolete capability or diagnostic names and helper methods that no
   longer represent the model.
2. Ensure all existing unsupported paths still fail during normalization or
   renderer creation as before.
3. Add any missing regression tests for current behavior.
4. Run the focused checks below.

Expected tests:

```sh
cargo test -p surgeist-render capabilities
cargo test -p surgeist-render unsupported
cargo test -p surgeist-render invalid_value
cargo test -p surgeist-render unresolved_resource
cargo test -p surgeist-render degraded
```

## Final Review Gate

After task-scoped worker/reviewer cycles and coordinator commits are complete,
assign a final clean-context holistic reviewer to inspect:

- this plan
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-render-css-implementation-sequence.md`
- the final git diff since this implementation plan commit

The reviewer should check:

- capability families are typed and not broad string bags
- diagnostics distinguish unsupported, invalid, unresolved, and degraded paths
- current Vello 0.9 unsupported behavior is preserved
- no sibling dependency or root integration code was added
- tests cover representative current and future-facing diagnostic contracts

## Required Checks

Run after the final holistic review, and repeat if follow-up changes are made:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Do not use `--all-features` for this phase unless a reviewer explicitly
confirms the feature matrix is valid for the current host and backend.
