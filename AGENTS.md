# Crate Agent Guide

Use this guide for automated work inside a single Surgeist crate repo.

## Crate Role

This repo owns one Surgeist crate. Keep work inside this crate unless the user
or root coordinator explicitly directs otherwise.

Before starting, identify the crate name from `Cargo.toml` and read the local
`README.md`, active plan, and any referenced guidance files. Treat root-authored
plans as the source of task scope for cross-crate work.

## Boundaries

- Do not edit sibling crate repos from this crate project.
- Do not update root `surgeist` submodule pointers from this crate project.
- If another crate blocks this work, report the blocker with reproduction steps,
  expected behavior, observed behavior, affected APIs/files, and commands run.
- Keep public APIs intentional and crate-owned. Do not reach through sibling
  private module paths.
- Generated files are owned by their generator. Regenerate them with the
  documented command instead of hand-editing output.

## Coordinator Workflow

Crate coordinators coordinate first. They assign, verify, reconcile, and
integrate; they do not default to editing implementation code themselves.

For code changes, follow this sequence unless the user explicitly waives it:

1. Check crate status before work begins.
2. Confirm the work belongs in this crate.
3. If executing an implementation plan, split it into sequential tasks and
   assign workers one task or tightly coupled task group at a time.
4. Assign one implementation worker to the current scoped task.
5. Wait for the worker's result, including reported tests and git status.
6. Assign a separate reviewer to inspect the worker's scoped changes before
   moving to the next task.
7. Reconcile worker and reviewer findings before assigning follow-up work.
8. After a scoped task's worker/reviewer cycle is clean, run the relevant
   focused check and commit that task as a traceable logical point.
9. After all scoped tasks are complete, assign a final holistic reviewer to
   inspect the complete result against the plan, crate boundary, tests, and git
   diff.
10. Run this crate's final focused checks.
11. Push when another repo/thread must fetch the commit, when root needs a
    submodule pointer update, or when the user requested publication.
12. Report completion or blockers back to the root coordinator when root
    integration or pointer updates are needed.

During plan execution, assign one clear scoped task to each worker. Tell workers
they are not alone in the codebase and must not revert others' work.

Do not declare a multi-task implementation plan complete until task-scoped
worker/reviewer cycles and the final holistic review are clean.

## Direct Coordinator Edits

The coordinator may directly edit crate-local planning, documentation,
requirements, or workflow files when the user explicitly asks for that change.

Crate code changes still count as code changes and must go through the worker
and reviewer gate unless the user explicitly waives review.

## Testing

Use the crate's active plan, README, task runner, or local scripts for
crate-specific checks. Baseline commands:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Do not assume `--all-features` is valid when features are mutually exclusive,
host-specific, backend-specific, or generator-only.

If a broad check is skipped, name the reason in the task output.

## Git Hygiene

- Use `apply_patch` for manual edits.
- Prefer `rg` and `rg --files`.
- Check status before and after edits: `git status --short --branch`.
- Review `git diff --stat` and the relevant detailed diff before committing.
- Do not rewrite unrelated files or revert user changes unless explicitly asked.
- Keep `.venv/`, `target/`, `build/`, `dist/`, secrets, editor residue, and
  runtime residue out of git.
- Commit logical points with short, concrete messages.
- Push only when requested, when handing work to another repo/thread, or when
  root needs a fetchable commit for pointer updates.

## Engineering Guidance

When a plan asks for modeling, API, dependency, generated-artifact, unsafe, or
cross-crate boundary review, use the guidance file referenced by that plan. If
no file is referenced and the work changes Rust models, use:

```text
guidance/surgeist-rust-modeling-guide.md
```

Plans should carry the detailed engineering context needed for their task. This
file is only the standing crate-local workflow contract.
