# P02-I01-S01-C02 Authored Model And Style Test Consolidation

## 1 Header

- Cycle: `P02/I01/S01/C02`.
- Owning repository: `surgeist-render`.
- Status: `draft`.
- Cycle base and published prerequisite:
  `eb3e49249b6b68726b956046ec8511f4c3aa3f5a` (`P02/I01/S01/C01`),
  verified on local `main`, `origin/main`, and observed authority-remote `main`
  before this plan was written.
- Specification: `plans/specs/durable-test-suite.md` at
  `79f68da934322a13f286a64d6d7df48213ca5046`, normalized SHA-256
  `100e4972bfe4237f6b7bc89dc9b2821c71a3e09c5cb9a92ccb51ce7985dbabbb`;
  sections S01-S02, S04-S05, and S07-S12.
- Sequence: `plans/sequences/durable-test-suite.md` at
  `50c5a17697dc9ba4b39c366f81e33f7449d1c558`, normalized SHA-256
  `cbc45733a5b8011d0860e2254eaf752d280a59dd687b72b747eb4240e7cd986e`;
  entry `C02 Authored Model And Style Test Consolidation`.
- Outcome: give every authored/model/style test one semantic disposition,
  remove duplicate oracles and completed-sequence naming, and retain distinct
  construction, normalization, conversion, boundary, and diagnostic behavior
  under condition-focused names.

## 2 Boundary

- C02 owns tests of crate models and authored/normalized behavior for
  capabilities, errors, geometry, transforms, paint, shape, image, layer,
  scene, text, style image placement/repeat/attachment, backgrounds, box
  decoration, filters, clips, and masks.
- C02 also owns shared fixtures used only by those domains and every
  `sequenceN` test name. When a `sequenceN` test primarily exercises runtime GPU
  execution, C02 renames it for its authored input and observable result but
  does not restructure runtime support reserved for C03.
- A test is retained when it uniquely protects a valid, boundary, invalid,
  conversion, normalization, ordering, or typed diagnostic condition.
- Tests consolidate only when they invoke the same operation with the same
  oracle and differ solely by case data. Each consolidated case keeps a short
  diagnostic label. Tests with distinct setup, phase, failure, or outcome remain
  separate.
- A deleted test must be mapped in task evidence to a stronger retained test
  with the same condition and oracle, or identified as completed-plan-only
  evidence. No committed disposition ledger, parser, inventory, snapshot, or
  count assertion is added.
- C03 owns runtime graph/pass/shader/backend/renderer/resource/transaction/
  Vello/surface/presentation/readback consolidation and `cNN` execution names.
  All tests in `src/vello_engine/resources.rs`, including atlas-recovery
  behavior, remain C03-owned and are read-only context in C02.
  C04 owns final cross-domain fixture and complete naming reconciliation.
- `src/tests.rs` remains one file until P02-I02. Production algorithms, public
  APIs, manifests, dependencies, features, targets, docs, examples, fixtures,
  shaders, root artifacts, and rendering behavior are unchanged.
- Commands use already installed artifacts with `CARGO_NET_OFFLINE=true`. No
  acquisition, installation, update, or bootstrap is authorized.

## 3 Impacts

- Public API: internal-only; no public item, signature, path, or reexport change.
- Behavior: unchanged; tests and private test support become more concise.
- Dependencies/features/targets: unchanged.
- Generated artifacts and binary fixtures: unchanged.
- Documentation/examples: unchanged.
- MSRV: preserve Rust 1.97 compatibility.
- Root follow-up: leaf pointer evaluation only; no adapter or API artifact delta.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## 4 Ordered Tasks

### 4.1 T01 Consolidate Core Model And Capability Conditions

- Area: tests and solely owned private test support for capabilities, typed
  errors, geometry, transforms, paint, shapes, image buffers/resources, layers,
  scenes, text models, and their construction/conversion/validation behavior in
  `src/tests.rs`. Runtime/Vello resource tests are excluded.
- Outcome: every in-scope test is retained, consolidated, renamed, or deleted
  according to specification S04-S05; names state the subject, triggering
  condition where needed, and observable result without planning provenance.
- Characterization: run every renamed, consolidated, or deleted original test
  exactly before editing. For consolidation, run every original case and then
  the resulting table/property test. For deletion, identify the retained owner
  with the same operation and oracle. No artificial behavior RED is required.
- Acceptance: distinct constructors, invalid values, boundaries, conversions,
  typed errors, capability reports, ordering, and model invariants remain
  covered; no test is kept solely for implementation structure, exact private
  inventory, or context-compaction value; one-use support removed by a
  disposition is absent; no production or public change.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: none.
- Intended commit: `test(model): consolidate authored conditions`.

### 4.2 T02 Consolidate Style Normalization And Diagnostic Conditions

- Area: tests and solely owned private support for style image placement,
  repeat/attachment, backgrounds, borders/outlines/box decoration, filter
  models, clips, masks, and authored-to-normalized command ordering.
- Outcome: replace repeated same-oracle cases with labeled data cases where
  useful; retain separate semantic boundaries; remove completed-sequence
  wording from every owned test name and assertion message.
- Characterization: run every affected original exactly on the T01 head. A
  rename preserves the exact body/oracle; a consolidation proves every original
  case through the resulting case table; a deletion names its stronger retained
  owner. Add behavioral RED only for a concrete condition found to lack current
  evidence.
- Acceptance: normal, boundary, invalid, unresolved-resource, unsupported,
  ordering, clip/mask/filter, and normalization conditions remain explicit;
  authored model tests do not claim backend execution beyond their observable
  result; no `sequenceN` provenance remains in an owned test name or message;
  duplicate setup/oracle tests and their orphan fixtures are absent.
- Commands:
  - run every affected original and resulting test exactly;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets --features render-web -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `git diff --check`.
- Depends on: T01.
- Intended commit: `test(style): consolidate normalization conditions`.

### 4.3 T03 Reconcile Authored Fixtures And Sequence-Free Naming

- Area: the composed T01-T02 `src/tests.rs` result, cross-domain support used
  only by C02 domains, all remaining `sequenceN` test identifiers/messages, and
  only a directly affected test-only helper outside that file.
- Outcome: remove orphan and duplicate authored fixtures, give shared support
  one real domain owner, and close the C02 naming/disposition surface without
  creating a permanent enforcement test.
- Characterization: run all surviving C02-affected tests before removing shared
  support. The direct sequence-name residue predicate fails on the T02 head when
  any C02-owned provenance remains and passes after reconciliation.
- Acceptance: every C02-owned test has reviewer-checkable disposition evidence;
  no `#[test]` name or assertion message contains `sequenceN`; fixtures shared
  across domains are retained only when at least two distinct tests need the
  same semantic setup; no C03 runtime support is removed; configured evidence
  is green.
- Commands:
  - `test -z "$(perl -0777 -ne 'while(/#\\[test\\]\\s*(?:#\\[[^\\]]+\\]\\s*)*fn\\s+([A-Za-z0-9_]+)/g){print qq{$1\\n} if $1 =~ /(?:^|_)sequence\\d+(?:_|$)/}' src/tests.rs src/vello_engine/resources.rs)"`;
  - `test -z "$(rg -n -i 'sequence[0-9]+' src/tests.rs src/vello_engine/resources.rs || true)"`;
  - `CARGO_NET_OFFLINE=true cargo fmt --check`;
  - `CARGO_NET_OFFLINE=true cargo check -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render`;
  - `CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web`;
  - `CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web`;
  - `git diff --check`.
- Depends on: T02.
- Intended commit: `test(cleanup): retire sequence-named test support`.

## 5 Verification And Completion

C02 acceptance requires every T01-T03 task and task review to be `CLEAN`, every
in-scope deletion/consolidation to have non-committed reviewer evidence, zero
`sequenceN` test names/messages, unchanged public/dependency/feature/behavior
surfaces, and the complete matrix below on the exact completed cycle head:

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
test -z "$(git ls-files -- Cargo.lock)"
```

Final safety evidence builds the explicit owned-Rust manifest, runs the
canonical executable-unsafe scan, and classifies every match. Completion also
requires a status-only `complete` plan commit, a distinct holistic `CLEAN`
review over the exact cycle range, post-review repetition of the complete
matrix, authority-remote publication/readback, and a C03 handoff. Missing
installed tooling, graphical-host capability, credentials, or stable remote
history uses the canonical blocker contract.
