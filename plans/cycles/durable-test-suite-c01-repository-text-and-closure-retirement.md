# P02-I01-S01-C01 Repository-Text And Closure Scaffolding Retirement

## 1 Header

- Cycle: `P02/I01/S01/C01`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base: `8bff3a6af323cf724175bb2e3fd09c801b401804`.
- Specification: `plans/specs/durable-test-suite.md` at
  `79f68da934322a13f286a64d6d7df48213ca5046`, normalized SHA-256
  `100e4972bfe4237f6b7bc89dc9b2821c71a3e09c5cb9a92ccb51ce7985dbabbb`;
  sections S01-S06 and S08-S12.
- Sequence: `plans/sequences/durable-test-suite.md` at
  `50c5a17697dc9ba4b39c366f81e33f7449d1c558`, normalized SHA-256
  `cbc45733a5b8011d0860e2254eaf752d280a59dd687b72b747eb4240e7cd986e`;
  entry `C01 Repository-Text And Closure Scaffolding Retirement`.
- Outcome: remove tests and helpers that prove completed-plan or repository-text
  structure, while retaining each unique observable rendering condition through
  existing front-door or typed behavior.

## 2 Boundary

- The cycle covers the 31 tests that depend directly or indirectly on Rust or
  WGSL source text, the four manifest/synthetic-manifest parsing tests, and
  helpers used only by those tests.
- A source-dependent test may be deleted when adjacent or stronger behavior
  already owns its condition. When a test mixes behavior with source assertions,
  retain the behavioral portion under a condition-and-outcome name and delete
  only the implementation-text oracle.
- Repository text includes Rust, WGSL, README, notice, manifest, example, and
  directory/file inventories. Binary behavior fixtures such as embedded fonts
  or image bytes are not repository-text parsing and remain in scope only when
  transitively affected by helper cleanup.
- The baseline counts are discovery evidence. No committed test, inventory,
  ledger, parser, generated file, or policy asserts them after this cycle.
- C02 owns the complete authored/model/style naming and consolidation pass; C03
  owns the runtime/GPU/surface pass. C01 renames only a retained test directly
  edited to remove source dependence.
- No production-visible test API, generalized mock, parser replacement, lint,
  build script, dependency, feature, rendering behavior, shader behavior,
  diagnostic, or public surface is added.
- P02-I02 file decomposition, root integration, sibling repositories, root API
  artifacts, adapters, and gitlink promotion are outside this cycle.
- Commands use already installed artifacts with `CARGO_NET_OFFLINE=true`. No
  acquisition, installation, update, or bootstrap is authorized.

## 3 Impacts

- Public API: internal-only; no public item or root reexport changes.
- Behavior: unchanged; only non-behavior closure or implementation-text oracles
  are removed.
- Dependencies/features/targets: unchanged; `Cargo.toml` is not edited.
- Generated artifacts and fixtures: none.
- Documentation/examples: unchanged; tests that parse their text are removed.
- MSRV: preserve Rust 1.97 compatibility.
- Root follow-up: evaluate the published leaf pointer only; no adapter or API
  artifact delta is expected.
- Safety: every tracked or non-ignored repository-owned Rust file remains free
  of executable `unsafe` and unsafe-enabling allowances.

## 4 Ordered Tasks

### 4.1 T01 Remove Repository-Metadata And Completed-Closure Tests

- Area: the manifest/provenance/parser region at the beginning of
  `src/tests.rs`; pure source-placement, exact-inventory, documentation/example,
  final-contract, and completed-cutover tests; helpers used only by those tests.
- Outcome: remove tests whose sole oracle is repository wording, paths,
  identifiers, hashes, inventories, module placement, or proof that a P01 task
  once completed. Preserve committed manifests, notices, docs, examples, plans,
  and Git history unchanged as their own provenance sources.
- Characterization evidence: at the assignment base, run every test selected
  for deletion and record its passing result; inspect adjacent behavioral tests
  and name the retained owner of any overlapping condition. This is a
  behavior-preserving deletion and does not fabricate a behavioral RED.
- RED predicate: a direct source inventory over `src/tests.rs` finds the selected
  metadata/closure tests and manifest/provenance helpers at the assignment base;
  the same predicate must be empty after the task for the exact selected names.
- Acceptance: each deleted test has no unique observable rendering oracle;
  `ManifestDependencyRecords`, its parser helpers, private SHA-256 provenance
  support, exact source-file/adaptation inventories, documentation keyword
  assertions, final test-name inventories, and their orphan support are absent
  when no remaining behavior uses them; no source, manifest, README, notice,
  example, or public API changes.
- Commands:
  - run each selected test exactly on the assignment base;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: none.
- Intended commit: `test(cleanup): retire project closure assertions`.

### 4.2 T02 Preserve Behavior Without Rust Or WGSL Text Assertions

- Area: every remaining source-dependent test in `src/tests.rs` and only the
  existing private test-support entry point directly exercised by such a test.
- Outcome: retain a test only for its observable return, pixels, route, typed
  error, statistics, capability, resource state, transaction event, or
  publication condition; remove Rust/WGSL substrings, function-body scans,
  file walks, and internal-name/call-graph assertions.
- Characterization evidence: before editing, run each mixed test and the
  adjacent behavioral owner it overlaps. For a retained condition, keep the
  same behavioral oracle green throughout the refactor. Add a focused RED only
  if concrete inspection proves an enduring condition currently has no
  behavioral coverage; otherwise deletion or oracle removal needs no artificial
  behavior failure.
- RED predicate: the repository-text scan in the task evidence identifies every
  remaining `include_str!`, Rust/WGSL `read_to_string`, source `read_dir`,
  `StaticSourceScanForTest`, and braced-body call reachable from a test.
- Acceptance: no retained test infers resource separation, execution routing,
  readback absence, CPU/GPU ownership, cache structure, shader behavior,
  capability semantics, or failure handling from source text; each retained
  condition is observable and named for its trigger/result; no new test seam or
  production behavior is introduced merely to replace a source assertion.
- Commands:
  - run every affected mixed test and its named behavioral owner before and
    after the edit;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: T01.
- Intended commit: `test(cleanup): assert rendering behavior directly`.

### 4.3 T03 Remove Repository-Text Infrastructure And Reconcile C01

- Area: `src/tests.rs`, `src/vello_engine/resources.rs`, and any test-only helper
  in an in-scope source module that became orphaned through T01-T02.
- Outcome: remove `StaticSourceScanForTest`, source code tokenization,
  braced-body extraction, production-Rust directory traversal, manifest audit,
  closure inventories, and all orphaned helpers; prove C01 left no test that
  reads repository text.
- Characterization evidence: run every surviving affected behavioral test on
  the T02 head before deleting shared infrastructure. The authoritative C01
  residue scan fails before cleanup because the scanner/helper definitions
  remain, then passes after their removal.
- RED predicate: a direct scan at the T02 head finds at least one C01-owned
  repository-text helper or call; the final scan is empty without adding an
  executable test for that fact.
- Acceptance: tests contain no Rust/WGSL/README/notice/manifest/example source
  reads; no source scanner, manifest parser, repository directory walker,
  source-body extractor, closure inventory, or helper used only by them remains;
  binary fixtures and distinct behavioral oracles remain intact; all configured
  C01 evidence is green.
- Commands:
  - `test -z "$(rg -n 'include_str!|read_to_string|read_dir|StaticSourceScanForTest|source_braced_block_from_marker|source_code_only_for_static_reachability|production_rust_sources_for_static_reachability' src/tests.rs src/vello_engine/resources.rs || true)"`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `CARGO_NET_OFFLINE=true cargo check -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web`;
  - `git diff --check`.
- Depends on: T02.
- Intended commit: `test(cleanup): remove repository text scanners`.

## 5 Verification And Completion

C01 acceptance requires:

- every T01-T03 task acceptance criterion and task review is `CLEAN`;
- the final direct residue scan in T03 passes without a committed enforcement
  test;
- each deleted or changed test has task evidence mapping it to a retained
  behavioral owner or stating why no observable condition existed;
- no public API, dependency, feature, target, manifest, example, README, notice,
  shader, fixture, or production behavior delta;
- the complete command matrix below passes on the exact completed cycle head.

```sh
set -euo pipefail
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
```

The final safety gate builds the explicit owned-Rust manifest from tracked and
non-ignored untracked `*.rs`, runs the canonical executable-unsafe scan, and
classifies every textual match. Final completion also requires warning-denied
Clippy with `-F unsafe-code`, a status-only `complete` plan commit, a distinct
holistic `CLEAN` review over the exact cycle range, post-review repetition of
the complete command set, authority-remote publication/readback, and a crate
candidate handoff. Missing installed tooling, graphical-host capability,
credentials, or stable remote history uses the canonical blocker contract.
