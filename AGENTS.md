# Agent Guide

Use this guide for automated work in the `surgeist-render` crate repo.

## Product Compass

Surgeist is a reusable Rust UI framework built around strict, typed,
composable primitives for app structure, layout, style, rendering, retained
state, text, windows, dialogs, and future template/DSL layers.

Keep Surgeist host-adapter-agnostic unless a crate is explicitly about a host,
runtime, or backend. Prefer typed contracts over loose runtime behavior, hidden
shared state, or broad framework magic.

Refinement bar:

- Can this API be explained in one paragraph?
- Can this behavior be tested in its owning crate?
- Can this layer be reused outside a single app?
- Can an app override behavior without forking internals?
- Does this boundary reduce coordination, or just move complexity?

Treat public APIs, internals, tests, docs, commands, defaults, and errors as
part of the product. Names and boundaries should feel intentional.

## Repository Model

This repo owns the `surgeist-render` crate.

The top-level `surgeist` repo is the facade crate and integration workspace. It
links this repo as a submodule under `crates/surgeist-render` and owns workspace wiring,
submodule pointers, cross-crate plans, source-derived API coordination, and
whole-system verification.

This crate repo owns implementation, focused tests, crate-local plans,
crate-local API reports, and the intentional front-door API exposed from
`src/lib.rs`.

## Project Boundaries

Use one Codex project for the top-level repo and one Codex project for each
crate repo.

Agents working in this crate may edit this crate's source, tests, docs, plans,
API artifact, and crate-local tooling. They should run focused checks here
before declaring work complete.

Do not casually edit sibling crates or the top-level repo from this crate
project. If work requires a sibling API change, stop and report a precise
inter-crate issue in that owning repo, or write a complete issue draft if issue
creation is unavailable.

Subagents inherit the project boundary they are launched from. Give each worker
one clear repo/crate lane. Reviewers may inspect across crates, but
implementation workers should stay in their assigned project.

## Crate Role

`surgeist-render` owns rendering contracts and backend-facing draw data.

Keep this crate's public API focused on that role. Do not absorb behavior just
because another crate needs convenience. If a change would make this crate a
dependency sink or introduce a cycle, stop and revisit the boundary with the
top-level coordinator.

## Surgeist Crate Map

- root `surgeist`: thin facade, integration surface, and public composition layer.
- `surgeist-css`: strict CSS parsing and lowering into typed style data.
- `surgeist-style`: style model, resolution, and visual/layout property contracts.
- `surgeist-layout`: layout algorithms, layout contracts, oracle/parity tests,
  and fixture tooling.
- `surgeist-retained`: retained identity, retained state, tree identity, and
  stable handles.
- `surgeist-text`: text shaping, measurement, font-facing abstractions, and text
  layout contracts.
- `surgeist-render`: rendering contracts and backend-facing draw data.
- `surgeist-window`: window, app host, event-loop, and platform host contracts.
- `surgeist-dialog`: dialog contracts and dialog coordination primitives.
- `surgeist-shape`: shape, geometry, and primitive path data.

Add crates only for real API boundaries, not architecture theater.

## Dependency Direction

Keep dependencies directional and acyclic. If a small change needs edits across
many crates, stop and revisit the boundary.

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

Plan cross-crate API changes at the top level, then implement crate-local pieces
in the owning crate projects.

## API Coordination

API coordination is source-derived only. Do not maintain handwritten API truth
tables as authority.

Expose intentional front-door APIs from `lib.rs`, keep internals private by
default, and support generated API shape checks when that tooling exists. The
top-level repo may consume generated API reports, but source remains the source
of truth.

Prefer typed commands, events, snapshots, reports, and change sets. Avoid
sibling crates reaching through private module paths, broad common crates that
become dependency sinks, and accidental cycles introduced for convenience.

## Public API Surface

Treat public APIs as product contracts. Public surface includes `pub` items,
reexports, feature flags, error types, trait impls, docs, examples, and MSRV.
Generated API reports are audit artifacts for that source-derived contract, not
the source of truth.

For release-facing API changes:

- name the owning crate and public entry point
- explain whether the change is additive, breaking, or internal-only
- update docs, examples, and API artifacts when this crate requires them
- run source-derived API checks or `cargo semver-checks` when that tooling is
  configured

## Type And Value Modeling

Prefer semantic types over broad enums when a distinction carries invariants,
units, phases, or API meaning. Use newtypes and focused structs to make invalid
states hard to express and call sites easy to read.

Use enums for genuinely closed choices with different behavior, not as catch-all
value bags. If an enum starts accumulating unrelated states, revisit the model
before extending it.

For values that may stay symbolic across parsing, style resolution, and layout,
prefer a generic value representation such as `Calc<T>` over premature
conversion to raw numbers. Preserve enough structure to resolve values at the
layer with the right context.

## Unsafe Code

Default to no new `unsafe` code.

If `unsafe` is unavoidable:

- get explicit maintainer or user approval before adding it
- keep the unsafe block as small as possible
- document the safety contract at the unsafe boundary
- add focused tests around the invariants the unsafe code relies on
- require reviewer attention on the safety argument

Do not add broad `#[allow(...)]` attributes to quiet warnings. Prefer scoped
`#[expect(...)]` with a reason when a lint exception is intentional.

## Dependencies And Features

Prefer workspace dependencies and existing crate-local dependencies. New
dependencies must justify why they belong at this layer, whether they affect
MSRV, licenses, advisories, binary size, optional features, or dependency
cycles.

Feature checks must match this crate's real feature matrix. Do not assume
`--all-features` is valid when features are mutually exclusive, host-specific,
backend-specific, or generator-only. Document the correct feature combinations
in this guide when broad Cargo commands are misleading.

Run `cargo deny` or equivalent dependency checks when configured by this repo.
Do not introduce secrets, credential bypasses, network shortcuts, or CI bypasses.

## Generated Artifacts

Generated files are owned by their generator. Do not hand-edit generated output
unless this crate explicitly says that is allowed.

When generated API reports, fixtures, snapshots, bindings, or parity artifacts
change:

- run the documented generator or snapshot update command
- commit source inputs and generated outputs together when this crate expects
  both
- explain why the generated delta is expected
- do not weaken tests or snapshots just to make a check pass

## Upstream Issue Reporting

Workers and reviewers may identify issues in sibling or upstream crates while
working from this crate repo. They must not edit that crate from the wrong Codex
project.

When an upstream issue affects correctness, compatibility, integration, or
developer workflow:

- Confirm the owning repo/crate.
- Capture reproduction steps, expected behavior, observed behavior, affected
  APIs/files, and relevant commands/tests/plans/commits.
- Report the issue in the owning GitHub repo.
- If GitHub issue creation is unavailable, write a complete issue draft in the
  task output and stop for coordinator action.

Issue reports should be specific enough for a crate-local worker to act without
rediscovering the problem. Do not file upstream issues for bugs owned by this
crate; fix those locally.

## Plans And Specs

Use Superpowers for workflow guidance. Repo file locations override Superpowers
default paths.

Plans go in `/plans` at the root of the repo where the implementation will
happen:

- Crate-local plans: this crate repo's `plans/`
- Top-level integration plans: `surgeist/plans/`

If writing specs or design docs, use the same root-local convention unless the
user chooses a separate folder. Do not put new plans under `docs/superpowers`.

## Testing

This crate owns its focused test commands. Tight iteration should happen in this
crate project.

Expected command pattern:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

These commands are a baseline, not a substitute for crate-specific guidance.
Use this crate's `README.md`, plans, or task runner when it defines
feature-specific, generated-artifact, snapshot, fuzzing, or dependency checks.

Feature checks must match this crate's actual features. Avoid broad
`--all-features` commands when this crate has mutually exclusive or
host-specific features.

Run focused checks before committing. Ask the top-level coordinator to run
broader root checks before submodule pointer updates or cross-crate work is
declared complete.

## Subagents

Crate coordinators are coordinators first. For implementation work, they assign,
verify, reconcile, and integrate; they do not default to doing the code edit
themselves.

For any requested code change, a crate coordinator must follow this sequence
unless the user explicitly waives it:

1. Check crate status before work begins.
2. Confirm the work belongs in `surgeist-render`.
3. Assign one implementation worker to this crate lane.
4. Wait for the worker's result, including its reported tests and git status.
5. Assign a separate reviewer to inspect the worker's changes.
6. Reconcile the worker and reviewer findings.
7. Run this crate's focused checks.
8. Commit crate-local changes in this repo only.
9. Push when another repo/thread must fetch the commit, when the top-level repo
   needs a submodule pointer update, or when the user requested publication.
10. Hand the completed crate-local result back to the top-level coordinator when
    root integration or submodule pointer updates are needed.

The coordinator may directly edit crate-local planning, documentation,
requirements, or workflow files when the user explicitly asks for that change.

Crate code changes still count as code changes and must go through the worker
and reviewer gate unless the user explicitly waives review.

No coordinator may declare implementation complete until a separate reviewer has
reviewed the changed code, or the user explicitly waives review.

- Assign one clear repo/crate lane per worker.
- Tell workers they are not alone in the codebase and must not revert others'
  work.
- Do not duplicate a completed subagent's investigation. Review, verify,
  reconcile, and act on it.
- Use clean reviewers for code changes, boundary changes, API changes, unsafe
  code, and nontrivial cross-crate work.

## Editing And Git

- Use `apply_patch` for manual edits.
- Prefer `rg` and `rg --files`.
- Check status before and after edits: `git status --short --branch`.
- Before committing, review `git diff --stat` and the relevant detailed diff.
- Do not rewrite unrelated files or revert user changes unless explicitly asked.
- Keep `.venv/`, `target/`, `build/`, `dist/`, secrets, host identity, editor
  residue, and runtime residue out of git.
- Commit logical points with short, concrete messages.
- Push commits only when requested, when handing work to another repo/thread, or
  when the top-level repo must fetch the commit for a pointer update.

Commit in the repo being changed:

- crate implementation commits inside this crate repo
- submodule pointer and integration commits inside the root Surgeist repo

Never silently edit a sibling crate or the top-level repo from this crate project
and call it done. If that happens by mistake, stop, report it, and reconcile
deliberately.
