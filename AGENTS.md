# Agent Guide

Use this guide for automated work in the Surgeist top-level repo.

## Product Compass

Surgeist is a reusable Rust UI framework built around strict, typed, composable primitives for app structure, layout, style, rendering, retained state, text, windows, dialogs, and future template/DSL layers.

Keep Surgeist host-adapter-agnostic unless a crate is explicitly about a host, runtime, or backend. Prefer typed contracts over loose runtime behavior, hidden shared state, or broad framework magic.

Refinement bar:

- Can this API be explained in one paragraph?
- Can this behavior be tested in its owning crate?
- Can this layer be reused outside a single app?
- Can an app override behavior without forking internals?
- Does this boundary reduce coordination, or just move complexity?

Treat public APIs, internals, tests, docs, commands, defaults, and errors as part of the product. Names and boundaries should feel intentional.

## Repository Model

This repo is both:

- the root `surgeist` facade crate at the repo root
- the coordination workspace for sibling Surgeist crate repos linked as submodules

Expected shape:

```text
surgeist/
  Cargo.toml
  src/
  crates/
    surgeist-css/
    surgeist-dialog/
    surgeist-layout/
    surgeist-render/
    surgeist-retained/
    surgeist-shape/
    surgeist-style/
    surgeist-text/
    surgeist-window/
```

The root repo owns workspace wiring, submodule pointers, cross-crate plans, source-derived API coordination, and whole-system verification. Crate implementation work belongs in that crate's own repo and Codex project.

## Project Boundaries

Use one Codex project for this root repo and one Codex project for each crate repo.

Top-level agents may coordinate, inspect submodules, write integration plans, run workspace checks, update submodule pointers, and review cross-crate compatibility.

Top-level agents must not casually edit submodule contents. If a crate needs implementation work, do it from that crate's Codex project unless the user explicitly asks for a top-level integration edit.

Crate project agents edit only their crate repo, run focused checks there, commit there, and expose intentional front-door APIs for integration.

Subagents inherit the project boundary they are launched from. Give each worker one clear repo/crate lane. Reviewers may inspect across crates, but implementation workers should stay in their assigned project.

## Crate Roles

- root `surgeist`: thin facade, integration surface, and public composition layer.
- `surgeist-css`: strict CSS parsing and lowering into typed style data.
- `surgeist-style`: style model, resolution, and visual/layout property contracts.
- `surgeist-layout`: layout algorithms, layout contracts, oracle/parity tests, and fixture tooling.
- `surgeist-retained`: retained identity, retained state, tree identity, and stable handles.
- `surgeist-text`: text shaping, measurement, font-facing abstractions, and text layout contracts.
- `surgeist-render`: rendering contracts and backend-facing draw data.
- `surgeist-window`: window, app host, event-loop, and platform host contracts.
- `surgeist-dialog`: dialog contracts and dialog coordination primitives.
- `surgeist-shape`: shape, geometry, and primitive path data.

Add crates only for real API boundaries, not architecture theater.

## Dependency Direction

Keep dependencies directional and acyclic. If a small change needs edits across many crates, stop and revisit the boundary.

Current intended shape:

```text
root surgeist crate
  -> surgeist-css
  -> surgeist-dialog
  -> surgeist-layout
  -> surgeist-render
  -> surgeist-retained
  -> surgeist-shape
  -> surgeist-style
  -> surgeist-text
  -> surgeist-window

surgeist-css -> surgeist-style
surgeist-style -> surgeist-layout, surgeist-retained, surgeist-text
surgeist-render -> surgeist-shape, surgeist-window optional
surgeist-text -> surgeist-render optional
```

Plan cross-crate API changes at the top level, then implement crate-local pieces in the owning crate projects.

## API Coordination

API coordination is source-derived only. Do not maintain handwritten API truth tables as authority.

Each crate should expose intentional front-door APIs from `lib.rs`, keep internals private by default, and support generated API shape checks when that tooling exists. The root repo may consume generated API reports, but source remains the source of truth.

Prefer typed commands, events, snapshots, reports, and change sets. Avoid sibling crates reaching through private module paths, broad common crates that become dependency sinks, and accidental cycles introduced for convenience.

## Type And Value Modeling

Prefer semantic types over broad enums when a distinction carries invariants, units, phases, or API meaning. Use newtypes and focused structs to make invalid states hard to express and call sites easy to read.

Use enums for genuinely closed choices with different behavior, not as catch-all value bags. If an enum starts accumulating unrelated states, revisit the model before extending it.

For values that may stay symbolic across parsing, style resolution, and layout, prefer a generic value representation such as `Calc<T>` over premature conversion to raw numbers. Preserve enough structure to resolve values at the layer with the right context.

## Upstream Issue Reporting

Workers and reviewers may identify issues in sibling or upstream crates while working from another repo. They must not edit that crate from the wrong Codex project.

When an upstream issue affects correctness, compatibility, integration, or developer workflow:

- Confirm the owning repo/crate.
- Capture reproduction steps, expected behavior, observed behavior, affected APIs/files, and relevant commands/tests/plans/commits.
- Report the issue in the owning GitHub repo.
- If GitHub issue creation is unavailable, write a complete issue draft in the task output and stop for coordinator action.

Issue reports should be specific enough for a crate-local worker to act without rediscovering the problem. Do not file upstream issues for bugs owned by the current repo; fix those locally.

## Plans And Specs

Use Superpowers for workflow guidance. Repo file locations override Superpowers default paths.

Plans go in `/plans` at the root of the repo where the implementation will happen:

- Top-level integration plans: `surgeist/plans/`
- Crate-local plans: that crate repo's `plans/`

If writing specs or design docs, use the same root-local convention unless the user chooses a separate folder. Do not put new plans under `docs/superpowers`.

## Testing

Each crate owns its focused test commands. The root repo coordinates whole-system checks, but tight iteration should happen in the relevant crate project.

Expected command pattern:

```sh
cargo test -p <crate>
cargo clippy -p <crate> --all-targets -- -D warnings
cargo fmt --check
```

Run focused checks before committing. Run broader root checks before updating submodule pointers or declaring cross-crate work complete.

## Subagents

Use subagents for bounded work and clean reviews.

- Assign one clear repo/crate lane per worker.
- Tell workers they are not alone in the codebase and must not revert others' work.
- Do not duplicate a completed subagent's investigation. Review, verify, reconcile, and act on it.
- Use clean reviewers for boundary changes, API changes, and nontrivial cross-crate work.

## Editing And Git

- Use `apply_patch` for manual edits.
- Prefer `rg` and `rg --files`.
- Check status before and after edits: `git status --short --branch`.
- Do not rewrite unrelated files or revert user changes unless explicitly asked.
- Keep `.venv/`, `target/`, `build/`, `dist/`, secrets, host identity, editor residue, and runtime residue out of git.
- Commit logical points with short, concrete messages.

Commit in the repo being changed:

- crate implementation commits inside that crate repo
- submodule pointer and integration commits inside the root Surgeist repo

Never silently edit a sibling submodule from the root project and call it done. If that happens by mistake, stop, report it, and reconcile deliberately.
