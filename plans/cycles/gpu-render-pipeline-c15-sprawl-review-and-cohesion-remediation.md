# P01-I03-S01-C15 Sprawl Review And Cohesion Remediation

## 1 Header

- Cycle: `P01/I03/S01/C15`.
- Owning repository: `surgeist-render`.
- Status: `complete`.
- Cycle base and published prerequisite:
  `1fd628bbdd5f6e5ed7750a9002aa4be347c90188` (`P01/I03/S01/C14`),
  verified on local `main`, `origin/main`, and the observed authority-remote
  `main` before this plan was written.
- Immutable implementation-source baseline:
  `92cdd9114046115d45451153c6ebad3b425db36e`; the published C14 candidate
  changes only planning state after this source baseline.
- Specification: `plans/specs/gpu-render-pipeline.md` at
  `e88e00bc8bd9325ae82ef1f1db2e4c72de44b28b`,
  `sha256:30dee50db5e8ad2f06df7cbd01ef34c61b5ccd037924a100932053a8507712af`;
  sections S39 and S38 item 12.
- Sequence: `plans/sequences/gpu-render-pipeline.md` at
  `8709e0d6ce1e04b646d41763c5efafdf9ecf7daf`,
  `sha256:7047172964b388c17e6d7fee43fb0be788dd9c5761d51dabc566c605d31a57d1`;
  entry `C15 Sprawl Review And Cohesion Remediation`.
- Outcome: exhaustively audit the two historical function-size experiments,
  retain semantic boundaries, repair every confirmed cohesion regression, and
  publish the final I03 leaf candidate without changing public behavior or
  turning a physical-line advisory into architecture.

## 2 Boundary

- The only experiment inputs are commit
  `9488f2000f0a31485679c39a568ff2a6d9d6f28f` against first parent
  `fa9c738577110c893b7b156c0e68b68fdeec6e51` and commit
  `4225fae1466aef093f5f22f69bd8c17c6e4420d7` against first parent
  `93fbc8e59f4b58a26520915ac1285d3fe8f54622`. They are evidence, not
  desired-state authority.
- The audit covers every Rust item changed by either experiment commit and
  every surviving descended helper/call relationship at the immutable source
  baseline. Adjacent current code is in scope only when needed to judge
  ownership, cohesion, phase boundaries, control flow, state/resource
  lifetimes, diagnostics, or test intent.
- The exact experiment file inventory is
  `src/{backend.rs,capability.rs,frame.rs,gpu_transaction.rs,pass.rs,renderer.rs,resource.rs,shader.rs,tests.rs}`
  and
  `src/vello_engine/{glyph.rs,raster.rs,recording.rs,scene.rs}`. A file may be
  omitted from remediation only after every changed item and surviving
  relationship in it has a recorded disposition.
- Mechanical fragmentation includes single-use forwarding helpers without a
  semantic boundary, argument/return bundle churn, split ordering or cleanup,
  misleading names, duplicated setup/assertion fragments, and ownership moved
  away from the governing module or phase. A helper is not damage merely
  because it is short, long, single-use, or introduced by an experiment.
- Useful boundaries remain when they express a reusable operation, phase or
  failure boundary, invariant, ownership boundary, or independently testable
  concept. No item is split, recombined, retained, or rejected merely to meet
  a physical-line count.
- The cycle may reshape private production and test code and add its tracked
  source-backed disposition ledger and focused closure guards. It adds no
  public API, rendering feature, dependency, feature flag, shader behavior,
  fixture, generated artifact, fallback, browser harness, or root-owned work.
- Public names/signatures/defaults, rendering pixels/routes, diagnostics,
  capabilities, statistics, ordering, transaction/resource lifetimes, test
  names/assertions, dependency/provenance facts, Rust 1.97 compatibility, and
  unsafe absence are preservation constraints.
- `clippy::too_many_lines` and similar structural lints are advisory evidence.
  The ordinary warning-denied Clippy matrices remain Boolean gates; the
  structural advisory output is never itself a pass/fail condition.
- No acquisition, installation, update, bootstrap, display substitute, or
  network dependency resolution is authorized. Required commands use already
  installed artifacts and `CARGO_NET_OFFLINE=true`.
- Root facade adaptation, browser-host execution, API artifact regeneration,
  root integration tests, and gitlink promotion remain outside this leaf cycle.

## 3 Impacts

- Public API: none. Private helper shape may change only while the public
  contract, documented semantics, and exported surface remain byte-for-byte or
  semantically unchanged as applicable.
- Behavior: none intended. Existing focused, full-suite, real-GPU, and native
  presentation evidence is preservation evidence, not permission to rewrite
  expectations around a refactor.
- Dependencies/features/targets: unchanged. No `Cargo.toml` change is expected.
- Artifacts: add only
  `plans/evidence/gpu-render-pipeline-c15-sprawl-dispositions.md` as the tracked
  audit ledger and focused test-owned guards required to keep that ledger
  complete and closed. Root-owned generated API artifacts remain untouched.
- Safety: every tracked or non-ignored repository-owned Rust file remains free
  of executable unsafe and unsafe-enabling lint allowances.
- Handoff: after canonical publication and remote readback, report the final I03
  candidate, unchanged public API, completed experiment dispositions, native
  and wasm evidence, browser-host follow-up, and root-owned
  facade/artifact/gitlink work.

## 4 Ordered Tasks

### 4.1 T01 Build The Exhaustive Experiment Disposition Ledger

- Area: the exact two first-parent Rust diffs, immutable source baseline
  `92cdd9114046115d45451153c6ebad3b425db36e`,
  `plans/evidence/gpu-render-pipeline-c15-sprawl-dispositions.md`, and only the
  minimum focused repository-contract guard in `src/tests.rs`.
- Outcome: independently enumerate every Rust item changed by either experiment
  commit and every surviving descended helper/call relationship, then assign
  each `retain`, `remediate in C15`, or `already superseded` with current source
  evidence and a cohesion rationale. This task changes no production helper.
- Ledger schema: stable entry ID; experiment commit and first parent; file;
  fully qualified item path/kind and historical hunk anchor; current descendant
  or explicit absence; related caller/callee entry IDs; current source anchor;
  disposition; evidence/rationale; and, for remediation, exactly one owning
  task T02-T04. Duplicate changes across the two experiments are linked but not
  silently dropped. A per-commit/per-file manifest records counts and proves
  coverage of all 13 files.
- Historical extraction evidence:
  `git diff --find-renames --function-context <first-parent> <experiment> -- '*.rs'`
  for both exact pairs, plus the corresponding zero-context diff and
  `git diff-tree --no-commit-id --name-status -r`. Current descendant evidence
  is read from the exact baseline commit, not from an uncommitted checkout.
- RED: `test -f plans/evidence/gpu-render-pipeline-c15-sprawl-dispositions.md`
  fails at the cycle base; the focused closure guard then fails on missing
  experiment identities, file-manifest rows, relationship records, invalid
  dispositions, or remediation entries without exactly one task owner.
- Acceptance: the source-backed ledger is exhaustive and internally closed;
  every changed item occurrence and every surviving relationship has one
  disposition; every remediation is assigned once; no entry cites line length
  as its architectural rationale; no production source is changed; the task
  reviewer independently re-extracts both historical diffs and checks the
  current descendants rather than trusting ledger prose.
- Commands: run both exact extraction sets; run
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c15_sprawl_disposition_ledger_is_exhaustive_and_source_backed -- --exact`;
  run `C15-CHECK`.
- Depends on: none.
- Intended commit: `docs(audit): inventory function-size experiment`.

### 4.2 T02 Repair Lifecycle And Resource Cohesion Regressions

- Area: every T01 remediation assigned to T02 in production portions of
  `src/{backend.rs,frame.rs,gpu_transaction.rs,renderer.rs,resource.rs}` and the
  ledger/guards needed to record the exact correction.
- Outcome: repair experiment-caused fragmentation around device/surface/frame
  ownership, transaction ordering and failure cleanup, publication, resource
  leases/accounting/reuse, and renderer dispatch while retaining genuine phase
  and failure boundaries.
- RED: before changing each confirmed regression, the T02 ledger-closure guard
  fails on its still-open entry. When a regression affects an observable
  invariant, an exact focused behavior test fails for the missing preserved
  ownership/order assertion; a pure structural repair uses source/ledger
  evidence and must not fabricate a behavior change.
- Acceptance: every T02 entry is updated from planned remediation to an exact
  corrected descendant/commit and reviewer-checkable rationale; no T02 entry
  remains open; direct/graph selection, device and surface identity, atomic
  publication, cancellation/error attribution, presentation, and resource
  lifetime behavior remain unchanged; no new forwarding/bundle layer replaces
  the one removed; no public or dependency surface changes.
- Commands: run
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c15_lifecycle_resource_remediations_are_closed -- --exact`;
  run all affected lifecycle, transaction, resource, presented, and route tests
  named by the ledger; run `C15-CHECK` with default and `render-window` suites.
- Depends on: T01.
- Intended commit: `refactor(runtime): restore lifecycle cohesion`.

### 4.3 T03 Repair Graph Raster And Model Cohesion Regressions

- Area: every T01 remediation assigned to T03 in production portions of
  `src/{capability.rs,pass.rs,shader.rs}` and
  `src/vello_engine/{glyph.rs,raster.rs,recording.rs,scene.rs}`, plus the
  ledger/guards needed to record the exact correction.
- Outcome: repair experiment-caused fragmentation in graph validation/lowering,
  pass scheduling and resource bindings, shader serialization, capability
  ownership, glyph/scene validation, raster encoding, and recording/resource
  handoff without reopening the closed GPU vocabulary.
- RED: the T03 ledger-closure guard fails on every still-open entry. Observable
  ordering, validation, pixel, or lifetime invariants receive exact focused RED
  coverage before their structural correction; source-only cohesion repairs do
  not invent new semantics or line-count assertions.
- Acceptance: every T03 remediation has an exact corrected descendant and
  evidence; no T03 entry remains open; direct/internal-Vello and supported graph
  pixels, pass vocabulary, dependencies, resource lifetimes, validation order,
  diagnostics, precision, and provenance remain unchanged; useful phase and
  independently testable boundaries remain justified.
- Commands: run
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c15_graph_raster_model_remediations_are_closed -- --exact`;
  run every affected graph, pass, shader, capability, glyph, raster, recording,
  scene, and pixel-oracle test named by the ledger; run `C15-CHECK` with default,
  `render-web`, and combined features.
- Depends on: T02.
- Intended commit: `refactor(gpu): restore graph and raster cohesion`.

### 4.4 T04 Repair Test-Code Cohesion Regressions

- Area: every T01 remediation assigned to T04 in `src/tests.rs` and any
  experiment-descended `#[cfg(test)]`, `for_test`, fixture, source-scan, setup,
  or assertion helper inside the other 12 experiment files, plus the ledger.
- Outcome: remove duplicated or misleading experiment-created test scaffolding,
  recombine setup/assertion fragments that obscure test intent, and retain
  reusable fixtures and independently meaningful source checks. Production
  behavior and public tests are not weakened to make a refactor pass.
- RED: the T04 ledger-closure guard fails on every still-open test-code entry;
  each edited test is first run in its pre-remediation form as preservation
  evidence, and any new focused guard fails only on the precise duplicated or
  misplaced contract it is meant to protect.
- Acceptance: every T04 remediation has an exact corrected descendant and
  evidence; no T04 entry remains open; all existing test names and assertions
  retain their intent and coverage; source scans remain narrow and factual;
  fixtures do not gain opaque argument/return bundles or one-use forwarding
  chains; test-only CPU oracles remain test-only.
- Commands: run
  `CARGO_NET_OFFLINE=true cargo test -p surgeist-render tests::c15_test_code_remediations_are_closed -- --exact`;
  run every edited or transitively affected test named by the ledger; run the
  default, `render-window`, `render-web`, and combined full test suites; run
  `C15-CHECK`.
- Depends on: T03.
- Intended commit: `refactor(test): restore fixture cohesion`.

### 4.5 T05 Reconcile The Closed Audit And Final Preservation Matrix

- Area: the complete disposition ledger, removal of every C15-only raw-source
  parser/string-matching guard from `src/tests.rs`, and only a directly
  falsified C15-owned correction from T01-T04.
- Outcome: prove there is no unrecorded experiment item/relationship, no open or
  multiply owned remediation and no regression in the C14
  architecture/quality/platform contract, then retire the temporary C15 test
  machinery instead of publishing a bespoke Rust parser as permanent code.
- RED: the final-source residue check initially finds the C15-only constants,
  helpers, and tests accumulated by T01-T05. Before removing them, the worker
  runs their accepted focused evidence once on the exact pre-cleanup head and
  records the result; behavioral characterization remains in the ordinary test
  suite after cleanup.
- Acceptance: all historical item occurrences and surviving relationships are
  present; every disposition is final; every `retain` and `already superseded`
  rationale is supported by the ledger and independent review; every
  remediation cites the exact correction; `src/tests.rs` contains no `C15_`
  constant and no function whose name begins `c15_`; no C15-only source parser,
  raw-code relationship matcher, immutable-anchor loader, ledger validator, or
  mutation fixture remains; the T04 behavioral scanner characterization and the
  actual T03/T04 refactors remain; no physical-line ceiling appears as
  acceptance logic; the full feature/target/MSRV/native matrix is green; the
  initiative handoff is exact.
- Commands: on the exact pre-cleanup head, run the six accepted C15 focused
  guards once; after cleanup require
  `test -z "$(git grep -nE 'C15_|fn c15_' -- src/tests.rs || true)"`; rerun both
  historical extraction sets and every command in Section 5. The task reviewer
  independently re-extracts the ledger inputs and inspects the final source and
  remediation descendants without delegating semantic proof to string matches.
- Depends on: T04.
- Intended commit: `test(audit): retire temporary source validators`.

## 5 Verification And Completion

Implementation and final commands use only already-installed artifacts with
`CARGO_NET_OFFLINE=true`. `C15-CHECK` is:

```sh
set -euo pipefail
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
test -z "$(git ls-files -- Cargo.lock)"
```

The complete C15 final command set is:

```sh
set -euo pipefail
cycle_base=1fd628bbdd5f6e5ed7750a9002aa4be347c90188
implementation_source=92cdd9114046115d45451153c6ebad3b425db36e
cycle_head=$(git rev-parse HEAD)
test -z "$(git diff --name-only "$implementation_source" "$cycle_base" -- '*.rs' Cargo.toml README.md examples/)"
for experiment_sha in \
  9488f2000f0a31485679c39a568ff2a6d9d6f28f \
  4225fae1466aef093f5f22f69bd8c17c6e4420d7
do
  first_parent_sha=$(git rev-parse "$experiment_sha^")
  git diff --check "$first_parent_sha..$experiment_sha"
  git diff-tree --no-commit-id --name-status -r "$experiment_sha" -- '*.rs'
  git diff --find-renames --function-context "$first_parent_sha" "$experiment_sha" -- '*.rs' >/dev/null
  git diff --unified=0 "$first_parent_sha" "$experiment_sha" -- '*.rs' >/dev/null
done
test -z "$(git grep -nE 'C15_|fn c15_' -- src/tests.rs || true)"
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window,render-web
CARGO_NET_OFFLINE=true cargo check -p surgeist-render --target wasm32-unknown-unknown --features render-web --lib --tests
rustc +1.97.0 --version
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +1.97.0 check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D warnings" cargo doc -p surgeist-render --no-deps --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render --target wasm32-unknown-unknown --features render-web -e features -i getrandom@0.3.4
test -z "$(git ls-files -- Cargo.lock)"
owned_rs=("${(@f)$(
  {
    git ls-files -- '*.rs'
    git ls-files --others --exclude-standard -- '*.rs'
  } | sort -u
)}")
test "${#owned_rs[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rs[@]}"; then
  exit 1
else
  test "$?" -eq 1
fi
git diff --check "$cycle_base..$cycle_head"
test "$(git rev-parse HEAD)" = "$cycle_head"
test -z "$(git status --porcelain)"
```

Separately collect structural advisory output with
`CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -A warnings -W clippy::too_many_lines`.
Its findings must be considered against the ledger and source, but its warning
count and physical threshold are not Boolean acceptance evidence.

Completion requires all five tasks to have fresh `CLEAN` task reviews and
coordinator acceptance, a separate status-only `complete` commit, the complete
command set on the exact completed head, a fresh `CLEAN` cycle-level holistic
review whose task name ends at `c15`, a post-review repeat of the complete
command set, and canonical publication and remote readback. The final handoff
records the immutable candidate, ledger coverage/dispositions, exact preserved
public and dependency surfaces, feature/MSRV/wasm/native evidence, and remaining
root-owned work. Missing installed tooling, graphical-host capability,
credentials, or stable remote history uses the canonical blocker contract; no
unavailable requirement is counted as green.
