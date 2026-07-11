# surgeist-render Repository Guide

Use `$surgeist-agent` for every task in this repository.

## Authority Split

This file is the leaf repository's committed discovery entry point. It owns the
mapping from mutable leaf facts to authoritative sources, the intended crate and
architecture boundary, and the configured local command inventory. The sources
named below own their current values.

`$surgeist-agent` is the sole Surgeist workflow authority. It owns scope control,
planning, debugging and TDD, worker/reviewer gates, external-software permission,
the absolute unsafe prohibition, Git landing and publication, and
cross-repository handoffs. This file does not redefine those workflows or grant
authority to mutate, install, commit, or publish.

Resolve an apparent conflict by domain: use this file and the sources below for
mutable repository facts; use `$surgeist-agent` for workflow.
Higher-priority user and system instructions still apply. Do not import another
general development workflow.

## Repository Identity And Ownership

`surgeist-render` is an independent leaf repository. It owns its manifest,
rendering domain implementation, public front door, focused tests and docs,
commits, and published `main` candidate.

Root `surgeist` owns the facade and public composition, cross-crate adapters,
root integration tests and tools, the gitlink, and the API generator and
artifacts. A parent workspace, Codex project, task, branch, or worktree does not
change repository ownership.

## Discover The Current Structure

Read these sources for current facts; do not substitute a cached root roster or
descriptive crate list for them.

| Fact | Authoritative source |
| --- | --- |
| Package identity, edition, dependencies, features, and targets | `Cargo.toml` |
| Public front door | `src/lib.rs` and its reexports |
| Implemented behavior and crate boundary | `README.md` and `src/` |
| Focused verification | Tracked `#[cfg(test)]` modules and, when present, `tests/` and fixtures |
| Additional local commands | Cargo targets and features in `Cargo.toml` and `README.md`; task runner and CI when present |
| Root integration MSRV, repository URL, and compatible pin | Root `Cargo.toml`, root `.gitmodules`, and root's committed gitlink for `crates/surgeist-render`, when root integration is in scope |

When these sources disagree, report the exact paths and revisions. Do not guess,
silently update another document, or widen the task to reconcile them.

## Crate Boundary

`surgeist-render` owns rendering contracts and backend-facing draw data. It
excludes style, layout, and text interpretation and application and window
lifecycle.

Surgeist-to-Surgeist lowering and adapters belong to root. Sibling internals are
not this repository's surface.

## API Artifacts

Source in this repository is authoritative. Root `surgeist` owns the only API
generator and all generated API audit artifacts. This leaf carries no copies.

## Command Inventory

These commands describe local verification capability; `$surgeist-agent`
determines their exact gate, order, feature matrix, and whether already-present
tooling can run without unauthorized acquisition.

```sh
cargo check -p surgeist-render
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
cargo fmt --check
```

Discovery is complete when the owning repository, public front door, dependency
and feature facts, verification sources, API-artifact owner, and applicable
command inventory are identified from the sources above.
