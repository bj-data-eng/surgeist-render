# P02-I01-S01-C04 Cross-Domain Test Suite Reconciliation

## 1 Header

- Cycle: `P02/I01/S01/C04`.
- Owning repository: `surgeist-render`.
- Status: `in_progress`.
- Cycle base and published prerequisite:
  `83178d9337b5bab186d3d51deb256f252d78f7a1` (`P02/I01/S01/C03`),
  verified on local `main`, `origin/main`, and observed authority-remote `main`
  before this plan was written.
- Specification: `plans/specs/durable-test-suite.md` at
  `79f68da934322a13f286a64d6d7df48213ca5046`, normalized SHA-256
  `100e4972bfe4237f6b7bc89dc9b2821c71a3e09c5cb9a92ccb51ce7985dbabbb`;
  sections S01-S12.
- Sequence: `plans/sequences/durable-test-suite.md` at
  `50c5a17697dc9ba4b39c366f81e33f7449d1c558`, normalized SHA-256
  `cbc45733a5b8011d0860e2254eaf752d280a59dd687b72b747eb4240e7cd986e`;
  entry `C04 Cross-Domain Suite Reconciliation`.
- Outcome: reconcile the independently cleaned authored and runtime domains so
  every retained test states one enduring condition and observable outcome,
  cross-domain support has a semantic owner, and the complete P02-I01 suite is
  ready for publication without a count, naming, source, or architecture gate.

## 2 Boundary

- C04 owns the composed test suite in `src/tests.rs` and
  `src/vello_engine/resources.rs`, plus only private test support directly used
  by an affected cross-domain condition.
- C01-C03 dispositions are settled evidence. C04 may correct a remaining
  cross-domain duplicate, provenance-shaped name/message, architecture-only
  oracle, or orphan fixture that could not be judged within one earlier domain;
  it does not reopen already distinct behavior for broad rewriting.
- Words such as `Future`, `final present`, and `completed parent` may remain
  when they name a real Rust or rendering lifecycle state. Wording such as
  `source-readable`, `later cycle`, `future pass`, or a pinned historical
  schedule receives a semantic disposition rather than being preserved as
  project provenance.
- Production algorithms, public APIs, manifests, dependencies, features,
  targets, shaders, examples, docs, binary fixtures, root artifacts, and
  rendering behavior are unchanged. Root integration is excluded.
- `src/tests.rs` and large implementation files remain physically in place.
  Hierarchical test/front-door work and cohesive module decomposition belong to
  P02-I02 after this cycle is published. C04 creates no line-count threshold,
  size lint, parser, inventory, ledger, generator, or permanent naming/source
  enforcement test.
- Commands use already installed artifacts with `CARGO_NET_OFFLINE=true`. No
  acquisition, installation, update, or bootstrap is authorized.

## 3 Impacts

- Public API, dependency, feature, target, MSRV, docs/example, generated
  artifact, and caller migration impact: none.
- Test impact: condition-focused rename, consolidation, or deletion only when
  characterization and reviewer evidence prove preservation of every distinct
  behavior.
- Private support impact: remove orphan or forwarding-only test fixtures and
  retain a one-owner helper only when it is itself the typed oracle/transition
  boundary rather than historical setup indirection.
- Safety impact: none; Surgeist-owned Rust remains free of executable `unsafe`
  and unsafe-enabling allowances.
- Baseline evidence at the cycle base: 680 tracked source-level `#[test]`
  functions remain across the two owned files; direct plan-identifier and
  numbered-sequence test-name predicates are already empty;
  `source-readable` references and several ambiguous historical words still
  require semantic adjudication. These facts guide review and are not numerical
  acceptance gates.

## 4 Ordered Tasks

### 4.1 T01 Resolve Provenance And Architecture-Shaped Conditions

- Area: test names, assertion messages, comments, and solely owned private
  support whose wording or oracle describes source readability, a later cycle,
  a future implementation pass, a pinned historical schedule, a private phase,
  or another completed-project fact rather than a rendering condition.
- Outcome: retain genuine Rust/rendering lifecycle terms; rename tests whose
  behavior is durable; delete architecture-only or duplicate tests; replace an
  implementation-shaped oracle only when an uncovered observable condition
  exists. Remove directly affected historical helper vocabulary.
- Characterization: run every affected original exactly at the cycle base.
  Renames preserve bodies/oracles apart from provenance wording;
  consolidations run every original case and resulting labeled case; deletions
  identify a stronger owner or state why no observable contract exists. Do not
  fabricate RED for behavior-preserving cleanup.
- Acceptance: no test name, message, comment, or directly affected test helper
  presents project chronology, source readability, private inventory, module
  placement, or an exact historical schedule as the contract; real `Future`,
  graph-finalization, completed-parent, and frame-planning domain terms remain
  only where their trigger and outcome are observable; no public/production
  change.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: none.
- Intended commit: `test(cleanup): retire historical suite vocabulary`.

### 4.2 T02 Reconcile Cross-Domain Fixtures And Duplicate Oracles

- Area: the composed T01 result, helpers and fixtures shared between authored
  and runtime domains, directly related `#[cfg(test)]` support, and any
  same-operation/same-oracle duplication visible only after C02 and C03 were
  composed.
- Outcome: give shared support one semantic owner, remove orphan and
  forwarding-only fixtures, consolidate only data-equivalent cases, and close
  the complete initiative disposition surface without adding enforcement code.
- Characterization: run every affected surviving test before editing support.
  For a consolidation, run each original and the labeled result. For deletion,
  identify the retained owner with the same operation and oracle. A single-user
  helper remains only when it owns a typed validation, state transition, or
  legible complex oracle rather than merely forwarding setup.
- Acceptance: every retained test names a triggering condition and observable
  outcome; distinct normal, boundary, invalid, failure, lifecycle, capability,
  pixel, precision, route, platform, cancellation, publication, presentation,
  and readback conditions remain; no cross-domain duplicate shares both
  operation and oracle; no orphan parser, scanner, manifest audit, closure,
  inventory, or fixture support remains; configured evidence is green.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-window,render-web -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: T01.
- Intended commit: `test(cleanup): reconcile cross-domain suite support`.

## 5 Verification And Completion

C04 acceptance requires both tasks and task reviews to be `CLEAN`, every
changed/deleted/consolidated test to have non-committed disposition evidence,
all S12 initiative criteria to hold, unchanged public/dependency/feature/
behavior surfaces, and the complete matrix below on the exact completed cycle
head:

```sh
set -euo pipefail
test -z "$(perl -0777 -ne 'while(/#\\[test\\]\\s*(?:#\\[[^\\]]+\\]\\s*)*fn\\s+([A-Za-z0-9_]+)/g){$name=$1; print qq{$name\\n} if $name =~ /(?:^|_)(?:(?:p|i|s|c|t)\\d+|sequence\\d+)(?:_|$)/i}' src/tests.rs src/vello_engine/resources.rs)"
test -z "$(rg -n -i 'plans/|source[-_ ]readable|later[-_ ]cycle|future[-_ ]pass|closed[-_ ](?:cycle|sequence)|plan[-_ ]closure' src/tests.rs src/vello_engine/resources.rs || true)"
test -z "$(rg -n 'include_str!|read_to_string|read_dir|(?:std::)?fs::(?:read|read_to_string)|File::open|Cargo\\.toml|README\\.md' src/tests.rs src/vello_engine/resources.rs || true)"
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

Every unsafe-scan match is classified; an executable match blocks completion.
Completion also requires a status-only `complete` plan commit, a distinct
holistic `CLEAN` review over the exact cycle range, post-review repetition of
the complete matrix, authority-remote publication/readback, the final P02-I01
crate candidate handoff, and confirmation that P02-I02 may restore its draft on
this published base. Missing installed tooling, graphical-host capability,
credentials, or stable remote history uses the canonical blocker contract.
