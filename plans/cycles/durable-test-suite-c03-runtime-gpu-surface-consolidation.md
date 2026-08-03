# P02-I01-S01-C03 Runtime GPU Surface And Lifecycle Test Consolidation

## 1 Header

- Cycle: `P02/I01/S01/C03`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base and published prerequisite:
  `d6e33a40196f334e8070739ad44473174eef8e6b` (`P02/I01/S01/C02`),
  verified on local `main`, `origin/main`, and observed authority-remote `main`
  before this plan was written.
- Specification: `plans/specs/durable-test-suite.md` at
  `79f68da934322a13f286a64d6d7df48213ca5046`, normalized SHA-256
  `100e4972bfe4237f6b7bc89dc9b2821c71a3e09c5cb9a92ccb51ce7985dbabbb`;
  sections S01-S02, S04-S05, and S07-S12.
- Sequence: `plans/sequences/durable-test-suite.md` at
  `50c5a17697dc9ba4b39c366f81e33f7449d1c558`, normalized SHA-256
  `cbc45733a5b8011d0860e2254eaf752d280a59dd687b72b747eb4240e7cd986e`;
  entry `C03 Runtime GPU Surface And Lifecycle Test Consolidation`.
- Outcome: give every runtime, GPU, surface, and lifecycle test one semantic
  disposition while preserving distinct pixel, route, precision, failure,
  resource, cancellation, publication, presentation, and readback conditions
  under domain-focused names.

## 2 Boundary

- C03 owns behavioral tests of frame and graph planning, pass and shader
  execution, reference pixels, backend and renderer dispatch, GPU resources and
  transactions, Vello runtime state, surfaces, presentation, and explicit
  readback in `src/tests.rs` and `src/vello_engine/resources.rs`.
- C03 may simplify or rename private `#[cfg(test)]` support used directly by an
  affected runtime test. Production algorithms, public APIs, manifests,
  dependencies, features, targets, shaders, examples, docs, binary fixtures,
  root artifacts, and rendering behavior are unchanged.
- C02-authored model/style dispositions are settled. C03 does not revisit them
  merely because a runtime test shares their setup.
- C04 owns complete cross-domain fixture and naming reconciliation. C03 closes
  its runtime-owned identifiers and helpers without making a repository-wide
  enforcement mechanism.
- `src/tests.rs` and the large runtime implementation files remain physically
  in place. Module decomposition belongs exclusively to P02-I02 after C04 is
  published; C03 creates no size threshold, line-count test, or file move.
- Commands use already installed artifacts with `CARGO_NET_OFFLINE=true`. No
  acquisition, installation, update, or bootstrap is authorized.

## 3 Impacts

- Public API, dependency, feature, target, MSRV, and rendering behavior impact:
  none.
- Test impact: retain distinct runtime contracts, consolidate only repeated
  data cases with the same operation and oracle, delete duplicate historical
  matrices, and replace completed-cycle names and messages with conditions.
- Private support impact: remove orphaned runtime fixtures and rename only
  directly affected test seams whose historical spelling would otherwise leak
  into retained tests.
- Baseline evidence at the cycle base: 681 tracked source-level `#[test]`
  functions remain across the two owned files; 51 test names contain a `cNN`
  planning identifier; `src/tests.rs` contains 424 case-insensitive standalone
  `cNN` references. These are planning evidence, not numerical acceptance
  gates.

## 4 Ordered Tasks

### 4.1 T01 Consolidate Graph Pass Shader And Pixel Conditions

- Area: tests and solely owned private support for frame/graph planning, direct
  Vello versus GPU-graph routing, pass dependency and lifetime structure,
  filter/composite/backdrop execution, shader layouts and numeric vectors,
  precision selection, and reference or rendered pixels. Resource accounting,
  transaction failure ownership, renderer dispatch, surfaces, presentation,
  and readback are excluded.
- Outcome: retain distinct route, ordering, bound, precision, pass, shader, and
  pixel contracts; consolidate repeated matrices only when operation and oracle
  are identical; remove historical `c08`-through-`c13` wording from every owned
  test name and assertion message.
- Characterization: run every affected original exactly at the cycle base.
  Renames preserve bodies and oracles apart from provenance-only messages;
  consolidations run every original case and the resulting labeled case table;
  deletions identify a stronger retained owner. Add behavioral RED only for a
  concrete condition found to lack existing evidence.
- Acceptance: direct and graph route selection, graph validation, pass order
  and lifetimes, high/reduced precision, filter/composite/backdrop pixels,
  shader numeric behavior, bounds, and typed execution failures remain
  explicit; authored model tests and transaction/surface domains are untouched;
  no production or public change.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: none.
- Intended commit: `test(runtime): consolidate graph and pixel conditions`.

### 4.2 T02 Consolidate Resource Transaction And Failure Conditions

- Area: tests and solely owned private support for resource identity,
  allocation, leases, cache reuse and eviction, accounting faults, GPU
  transaction scopes, submission ownership, cancellation, failure atomicity,
  terminal device loss, and headless publication state. Surface acquisition,
  presentation, explicit readback, and renderer route dispatch are excluded.
- Outcome: preserve distinct lifecycle transitions and failure phases while
  replacing repeated fault matrices with labeled cases only where they share
  operation and oracle; remove completed-cycle wording from owned names and
  messages.
- Characterization: run every affected original exactly on the T01 head. Each
  retained failure phase must still prove its pre/post state. A consolidation
  runs every original row and resulting labeled table; a deletion names its
  stronger owner. Cancellation and post-submit cases are never merged merely
  because they return the same error.
- Acceptance: allocation identity, accounting, cache retention, lease
  generation, transaction commit/abort, cancellation, device loss, last-good
  statistics, and publication atomicity remain observable; test seams preserve
  production validation and state transitions; no surface/readback or public
  behavior change.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: T01.
- Intended commit: `test(resource): consolidate transaction lifecycles`.

### 4.3 T03 Consolidate Renderer Surface Presentation And Readback Conditions

- Area: tests and solely owned private support for backend selection, renderer
  dispatch, ready-device ownership, Vello atlas recovery, headless and presented
  surfaces, resize/suspend/resume/loss, acquisition and configuration,
  presentation, explicit readback, window smoke behavior, and native/web target
  boundaries.
- Outcome: retain distinct platform, state-transition, presentation, and
  explicit-readback contracts; consolidate same-transition matrices only when
  their state setup and oracle match; remove completed-cycle wording from every
  owned test name and assertion message.
- Characterization: run every affected original exactly on the T02 head,
  including feature-gated originals under their owning feature. Renames preserve
  bodies/oracles apart from provenance-only messages. Consolidations exercise
  all original transition rows; deletions identify an exact stronger owner.
- Acceptance: renderer routes, unavailable/terminal devices, independent
  surfaces, lifecycle recovery, present suppression/commit, cancellation,
  explicit headless readback, atlas recovery, and native/web boundaries remain
  explicit; no application/window lifecycle or root adapter work; no production
  or public change.
- Commands:
  - run every affected original and resulting test exactly under its feature;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web`;
  - `CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: T02.
- Intended commit: `test(surface): consolidate presentation lifecycles`.

### 4.4 T04 Reconcile Runtime Fixtures And Cycle-Free Naming

- Area: the composed T01-T03 result in both owned test files, cross-runtime
  private support used only by C03 domains, all remaining `cNN` test
  identifiers/messages, and only directly affected `#[cfg(test)]` helpers
  outside those files.
- Outcome: remove orphaned and duplicate runtime fixtures, give shared support
  a real domain owner, and close the C03 disposition and naming surface without
  adding a permanent parser, ledger, lint, or count gate.
- Characterization: run every surviving C03-affected test before removing
  shared support. The direct residue predicates fail on the T03 head when any
  runtime test provenance remains and pass after reconciliation.
- Acceptance: every C03-owned test has reviewer-checkable disposition evidence;
  no `#[test]` name or assertion message in the owned files contains a
  standalone `cNN`; shared fixtures remain only when at least two distinct
  tests need the same semantic setup; C02 coverage and C04 cross-domain support
  remain; configured evidence is green.
- Commands:
  - `test -z "$(perl -0777 -ne 'while(/#\\[test\\]\\s*(?:#\\[[^\\]]+\\]\\s*)*fn\\s+([A-Za-z0-9_]+)/g){$name=$1; print qq{$name\\n} if $name =~ /(?:^|_)c\\d+(?:_|$)/}' src/tests.rs src/vello_engine/resources.rs)"`;
  - `test -z "$(rg -n -i '\\bc[0-9]{2}\\b' src/tests.rs src/vello_engine/resources.rs || true)"`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `CARGO_NET_OFFLINE=true cargo check -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web`;
  - `git diff --check`.
- Depends on: T03.
- Intended commit: `test(cleanup): retire cycle-named runtime support`.

## 5 Verification And Completion

C03 acceptance requires every T01-T04 task and task review to be `CLEAN`, every
in-scope deletion/consolidation to have non-committed reviewer evidence, zero
standalone `cNN` test names/messages, unchanged public/dependency/feature/
behavior surfaces, and the complete matrix below on the exact completed cycle
head:

```sh
set -euo pipefail
test -z "$(perl -0777 -ne 'while(/#\\[test\\]\\s*(?:#\\[[^\\]]+\\]\\s*)*fn\\s+([A-Za-z0-9_]+)/g){$name=$1; print qq{$name\\n} if $name =~ /(?:^|_)c\\d+(?:_|$)/}' src/tests.rs src/vello_engine/resources.rs)"
test -z "$(rg -n -i '\\bc[0-9]{2}\\b' src/tests.rs src/vello_engine/resources.rs || true)"
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
test -z "$(git ls-files -- Cargo.lock)"
owned_rust_files=("${(@f)$(
  {
    git ls-files -- '*.rs'
    git ls-files --others --exclude-standard -- '*.rs'
  } | sort -u
)}")
test "${#owned_rust_files[@]}" -gt 0
if rg -n --pcre2 '#\s*\[\s*(?:unsafe\s*\(|no_mangle\b|export_name\b)|\bunsafe\s*(?:\{|fn\b|trait\b|impl\b|extern\b)|\bstatic\s+mut\b|\bextern\s*(?:"[^"]*")?\s*\{' "${owned_rust_files[@]}"; then
  exit 1
else
  test "$?" -eq 1
fi
```

Every unsafe-scan textual match is classified; an executable match blocks
completion. Completion also requires a status-only `complete` plan commit, a
distinct holistic `CLEAN` review over the exact cycle range, post-review
repetition of the complete matrix, authority-remote publication/readback, and a
C04 handoff. Missing installed tooling, graphical-host capability, credentials,
or stable remote history uses the canonical blocker contract.
