# Full Matrix Reconciliation And Root-Handoff Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Sequence 15 by reconciling every render CSS/style matrix row, tightening the public root-facing contract, adding representative full-stack guardrails, and producing the final root-handoff note.

**Architecture:** This phase is an audit and integration-readiness phase, not a new primitive expansion phase. Workers should prove the existing matrix rows are supported, diagnostic, or deferred with a named root/render boundary; remove only genuinely obsolete crate-local APIs; and add integration tests that exercise representative combinations without changing render ownership of CSS parsing, shaping, resource loading, or root tree construction.

**Tech Stack:** Rust 2024, crate-local plans and docs, `Capabilities`, typed `UnsupportedPrimitive` diagnostics, `Scene` normalization, headless render tests where stable, and the existing `cargo test -p surgeist-render` / clippy / fmt gates.

---

## Inputs And Boundaries

- Matrix:
  - `plans/2026-07-08-render-css-support-matrix.md`
- Sequence:
  - `plans/2026-07-09-render-css-implementation-sequence.md`, Sequence 15
- Guidance:
  - `AGENTS.md`
  - `guidance/surgeist-rust-modeling-guide.md`
- Prior sequence plans:
  - `plans/2026-07-09-*-implementation.md`
  - `plans/2026-07-09-text-paint-hooks-materialized-buckets-implementation.md`

Render must remain self-contained:

- no sibling crate edits
- no root submodule pointer edits
- no URL/font/image loading policy beyond fixtures and caller-provided bytes
- no CSS parsing, cascade, text shaping, generated-content construction,
  selection range calculation, hit testing, or root stacking-tree ownership
- no backwards compatibility shims for obsolete APIs

## Expected Outputs

- A committed matrix reconciliation artifact in `plans/` that can be handed to
  root.
- Optional source/test changes only where the audit proves a gap in public API
  front doors, typed diagnostics, or representative integration coverage.
- A final root-handoff note naming remaining root responsibilities and render
  responsibilities.
- Final clean-context holistic review and full crate checks.

## Tasks

Each task below must follow the crate coordinator workflow from `AGENTS.md`:
assign one scoped worker, have a separate clean-context reviewer inspect the
worker change, reconcile findings, rerun focused checks, and commit the task as
a traceable logical point before moving to the next task. Workers do not commit.

### 1. Matrix Reconciliation Ledger

Worker scope:

- Create `plans/2026-07-10-render-css-matrix-reconciliation.md`.
- Audit every table row in `plans/2026-07-08-render-css-support-matrix.md`,
  including primitive sections, backend pipeline requirements, and property
  cross-reference entries.
- Audit every bullet in the matrix Review Checklist.
- For each primitive or backend pipeline row, assign exactly one status:
  - `Supported`: implemented and covered by tests.
  - `Diagnostic`: accepted as a render-facing concept but rejected through a
    typed diagnostic/capability boundary.
  - `DeferredToRoot`: root/style/layout/text/runtime must resolve or
    materialize before render sees it.
  - `FutureRender`: still render-owned but intentionally not implemented by the
    completed sequence.
- For every primitive or backend pipeline row, include:
  - the primitive or pipeline name
  - status
  - render contract
  - authoritative evidence: tests, public types, capability flags, diagnostics,
    or plan notes
  - remaining root responsibility, if any
- For every property cross-reference entry, include:
  - CSS/style surface
  - primary render primitive contract
  - root lowering responsibility
  - evidence row or diagnostic status from the primitive/backend ledger
- For every Review Checklist bullet, include:
  - checklist text
  - pass/fail status
  - evidence from the ledger, code, tests, or docs
- Include a summary table grouped by matrix section, plus separate summaries for
  backend pipeline requirements, property cross-reference coverage, and Review
  Checklist coverage.
- Do not change code in this task unless the audit discovers a typo in a plan
  that makes the ledger ambiguous.

Focused checks:

```sh
rg -n "TODO|TBD|FutureRender|Diagnostic|DeferredToRoot|Supported" plans/2026-07-10-render-css-matrix-reconciliation.md
cargo fmt --check
```

Commit message:

```text
docs: reconcile render css matrix
```

### 2. Public API Front-Door Audit

Worker scope:

- Audit `src/lib.rs` exports and public constructors/accessors against the
  reconciliation ledger.
- Ensure root-facing public APIs are intentional front doors for implemented or
  diagnostic render-owned concepts.
- Remove obsolete public APIs only if they are proven unused internally and not
  needed by the final render contract. Backwards compatibility shims are not
  required, but do not churn public API without a concrete finding.
- If no source change is needed, add an audit section to
  `plans/2026-07-10-render-css-matrix-reconciliation.md` naming the public API
  front doors reviewed and why no removal was warranted.
- If source changes are needed, add tests proving the new narrower API still
  exposes the intended render contract.

Focused checks:

```sh
rg -n "pub use|pub struct|pub enum|pub fn" src/lib.rs src/*.rs
cargo test -p surgeist-render capability
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Commit message:

```text
docs: audit render public front doors
```

or, if source changes are made:

```text
render: tighten public front doors
```

### 3. Representative Full Paint Stack Guardrails

Worker scope:

- Add integration-style crate tests for representative full paint/effect stacks
  using existing render-owned primitives.
- Cover at least three representative stacks:
  - background/box decoration/image/text decoration/text run command ordering
  - transform/clip/layer opacity/image or gradient stack behavior
  - filter/shadow/mask/backdrop diagnostic boundaries when a stack crosses a
    still-unsupported render-owned boundary
- Use stable headless rendering only where existing helpers already handle
  adapterless environments. Prefer normalization/stat/capability assertions for
  behavior that should not require a GPU.
- Do not introduce new CSS shorthand models, parser inputs, or root tree APIs.

Focused checks:

```sh
cargo test -p surgeist-render matrix
cargo test -p surgeist-render sequence
cargo test -p surgeist-render render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Commit message:

```text
test: add matrix handoff guardrails
```

### 4. Root-Handoff Contract Note

Worker scope:

- Create `plans/2026-07-10-render-root-handoff.md`.
- Summarize the final root-facing render contract in a format root can consume:
  - render-owned supported primitives
  - typed diagnostic boundaries root must handle
  - values root/text/layout/style/runtime must resolve before render handoff
  - explicit resource policy: render consumes handles/bytes and does not load
    URLs
  - explicit text policy: render consumes prepared glyph runs/font bytes and
    does not shape/layout
  - explicit pseudo-element/selection policy: root materializes ordinary command
    streams
  - known `FutureRender` work items from the reconciliation ledger
- Link to the matrix reconciliation ledger and the active sequence plan.
- Do not describe unsupported behavior as supported merely because root could
  synthesize a workaround.

Focused checks:

```sh
rg -n "root|render|Diagnostic|FutureRender|DeferredToRoot|Supported" plans/2026-07-10-render-root-handoff.md
cargo fmt --check
```

Commit message:

```text
docs: add render root handoff
```

### 5. Sequence 15 Final Reconciliation Review

Worker scope:

- Add final `sequence15` tests or doc assertions only if Tasks 1-4 expose a
  missing invariant.
- Run a final self-audit against:
  - `plans/2026-07-08-render-css-support-matrix.md`
  - `plans/2026-07-09-render-css-implementation-sequence.md`
  - `plans/2026-07-10-render-css-matrix-reconciliation.md`
  - `plans/2026-07-10-render-root-handoff.md`
  - `AGENTS.md`
  - `guidance/surgeist-rust-modeling-guide.md`
- Confirm no generated files were hand-edited and no sibling/root files changed.
- Confirm the working tree contains only intended crate-local changes before
  final review.

Focused checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
git status --short --branch
```

Commit message, only if this task adds final tests/docs:

```text
test: finalize matrix reconciliation
```

## Final Review And Required Checks

After all scoped tasks are implemented, assign a final clean-context holistic
reviewer to inspect Sequence 15 and the complete sequence result against:

- this implementation plan
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-10-render-css-matrix-reconciliation.md`
- `plans/2026-07-10-render-root-handoff.md`
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- the full git diff for Sequence 15
- the current public exports in `src/lib.rs`

The final reviewer must confirm:

- every primitive and backend pipeline matrix row has a status and evidence
- every property cross-reference entry maps to a render contract and root
  lowering responsibility
- every matrix Review Checklist bullet has pass/fail evidence
- final handoff docs distinguish render, root, style, layout, text, and runtime
  ownership clearly
- representative stack tests cover implemented and diagnostic paths
- public APIs are intentional front doors from `lib.rs`
- obsolete APIs were removed or explicitly retained with a current contract
  reason
- no sibling crates or root submodule pointers were edited
- backwards compatibility shims were not added

Run these final checks before declaring Sequence 15 complete:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
git status --short --branch
```
