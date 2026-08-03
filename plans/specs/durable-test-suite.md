# P02-I01 Durable Test Suite

## S01 Outcome

`surgeist-render` has a concise, condition-focused test suite whose failures
describe enduring rendering behavior. Completed-plan provenance, raw source
inspection, exact internal inventories, and historical closure proofs do not
remain executable tests. Refactoring private modules, helpers, algorithms, or
execution structure does not require changing a test unless an observable crate
contract changes.

The initiative is complete when:

1. no `#[test]` name encodes a completed project, initiative, sequence, cycle,
   or task identifier;
2. no test reads Rust, WGSL, README, notice, manifest, example, or other
   repository text to infer implementation or architecture;
3. no test exists solely to prove that a completed plan, migration, cutover,
   placeholder removal, documentation task, or inventory reconciliation once
   closed;
4. each retained focused test names a triggering condition and observable
   outcome;
5. repeated cases use one legible table, property, fixture, or helper only when
   they share the same operation and oracle;
6. distinct normal, boundary, invalid, failure, lifecycle, capability, pixel,
   and platform behaviors remain covered;
7. public behavior, API, dependencies, features, targets, MSRV compatibility,
   and the absolute absence of Surgeist-owned `unsafe` remain unchanged.

There is no target number of tests, maximum test-name length, source-line limit,
or required percentage reduction. Concision is judged by distinct behavioral
value and failure clarity, not a numerical gate.

## S02 Ownership And Boundary

The owning repository is `surgeist-render`. It owns its focused tests, private
test fixtures and seams, rendering contracts, backend-facing draw data, and the
published leaf candidate.

In scope:

- tracked `#[test]` functions in `src/tests.rs` and
  `src/vello_engine/resources.rs`;
- private `#[cfg(test)]`, fixture, oracle, and `for_test` support used by an
  affected test;
- removal or simplification of test-only source scanners, manifest parsers,
  inventories, and closure helpers;
- condition-focused renaming and consolidation of tests;
- the smallest private test-support adjustment required to preserve an
  enduring condition without inspecting source text.

Out of scope:

- new rendering behavior, public API, dependency, feature, target, shader
  capability, fallback, or production diagnostic;
- production renaming or restructuring merely because a production identifier
  retains P01 history;
- a replacement parser, linter, test-count gate, naming checker, generator, CI
  rule, or permanent audit ledger;
- root facade, adapter, API artifact, integration-test, or gitlink work;
- sibling repositories and application or window lifecycle behavior.

Root integration, if later requested by root, is limited to evaluating and
promoting the published leaf candidate. This initiative never edits root.

## S03 Current Evidence

At cycle base `8bff3a6af323cf724175bb2e3fd09c801b401804`:

- the crate has 727 `#[test]` functions across two Rust files;
- 89 test names contain completed-plan identifiers: 61 use a `cNN` identifier
  and 28 use a `sequenceN` identifier;
- 31 tests depend directly or indirectly on Rust or WGSL source text;
- four additional tests parse only `Cargo.toml` or synthetic manifest text;
- no test contains a literal `plans/` filesystem path;
- `src/tests.rs` contains the private `StaticSourceScanForTest` parser and
  braced-body extraction support used to reason about source placement and
  call structure;
- P01 is complete and its committed planning artifacts and Git history retain
  the provenance that planning identifiers previously supplied in test names.

These counts establish the baseline only. P02 must not publish tests that assert
the counts, inventory the test suite, parse this specification, or otherwise
turn this evidence into another architecture lock.

## S04 Durable Test Contract

A retained test must exercise a crate-owned condition through one of these
observable forms:

| Form | Acceptable evidence |
| --- | --- |
| Public or crate front door | return value, typed error, state, event, statistics, capability report, or external effect |
| Rendering result | pixels, bounds, route, precision, pass accounting that is itself a documented contract, or publication state |
| Model invariant | valid construction, rejection, normalization, conversion, ordering, or property |
| Lifecycle | allowed transition, failure atomicity, cancellation, stale identity, cleanup, or retained publication |
| Backend contract | typed capability, resource state, command result, or instrumented event already owned by a narrow test boundary |
| Feature or target | successful compilation or execution under the documented feature/target matrix |
| Shader | successful validation/execution and an observed numeric or pixel result |

A test is not justified merely because it detects a textual change, names a
historical milestone, documents an implementation decision, or makes an LLM
rediscover less context.

Private test support remains acceptable when it preserves the same validation,
state transition, failure, resource, and capability semantics as production.
It must not expose a production-visible API or encode a particular source-file,
function-body, helper-call, module-placement, or spelling requirement.

## S05 Disposition Rules

Every existing test receives exactly one semantic disposition during P02:

### S05.1 Retain

Retain a test when it uniquely protects a durable condition and already asserts
observable behavior. Rename it if its name carries planning provenance or fails
to state the condition and result. Mechanical renaming must not alter its body
or oracle.

### S05.2 Consolidate

Consolidate tests only when they perform the same operation, use the same
oracle, and differ solely by data. Each case retains a short diagnostic label so
a failure identifies its condition. Keep separate tests when setup, phase,
failure semantics, lifecycle, or expected outcome differs materially.

Property tests supplement named boundary cases for algebraic invariants,
normalization idempotence, and broad input spaces. They do not replace concrete
failure or lifecycle cases.

### S05.3 Replace

Replace an implementation-text assertion when it contains an enduring behavior
that is not already covered. The replacement exercises the narrowest existing
front door or typed test seam and asserts only the observable condition. Do not
add a production-visible surface or a generalized mock solely to make the
replacement possible.

When a non-observable architectural statement cannot be expressed without
locking implementation structure, remove its test. Code review, module
ownership, compilation, Clippy, rustdoc, and the no-unsafe gate remain valid
evidence even when no runtime test asserts that statement.

### S05.4 Delete

Delete a test when any of these is its only value:

- proving a P01 plan, cycle, task, cutover, inventory, or documentation phase
  completed;
- asserting that a private identifier, helper, module, call, string, or source
  body exists or is absent;
- checking exact internal inventory counts or unique ownership of test names;
- parsing documentation for required wording or future-task residue;
- parsing manifests or copied provenance text as a rendering test;
- validating the source scanner, manifest parser, or another closure-only test
  utility;
- duplicating a stronger behavioral test with the same setup and oracle.

Deletion does not require a replacement when no enduring behavior would be
lost.

## S06 Repository-Text Matrix

| Existing evidence | P02 desired state |
| --- | --- |
| Rust `include_str!`, `read_to_string`, directory walk, substring, token, or braced-body checks | delete, or replace only the enduring behavior |
| WGSL text and shader-file ownership checks | validate/execute through existing shader or render behavior, otherwise delete |
| README, notice, documentation wording, or semantic-doc source checks | delete from the test suite; warning-denied rustdoc remains configured evidence |
| Cargo manifest and synthetic manifest parsing | delete from rendering tests; Cargo resolution, feature checks, and dependency commands remain evidence |
| Example source-body inspection | compile or run the example through documented commands, otherwise delete |
| `#[cfg(test)]`, module placement, private visibility, or CPU-code-location checks | delete; compilation boundaries and review establish placement |
| “no readback/map/poll/copy call” source scans | retain explicit readback and publication behavior tests; delete source-call assertions |
| Exact vendored-source hashes and adaptation lists | delete as runtime tests; committed source, manifest pins, notice, and Git history retain provenance |

P02 removes `StaticSourceScanForTest`, source-directory traversal, braced-body
extraction, and helper code used only by deleted repository-text tests. It does
not replace them with syntax trees, regular expressions, generated inventories,
or build scripts.

## S07 Test Names And Provenance

Test names use the smallest wording that identifies:

1. the subject or operation;
2. the triggering condition when it is not the ordinary path;
3. the observable outcome.

Names do not contain `P02`, `I01`, `S01`, `C01`, `T01`, earlier P01 forms such
as `c08`, sequence numbers, plan filenames, “final,” “planned,” “future task,”
or “closure” when those words describe project history rather than domain
semantics.

Planning provenance remains in committed plans, commit messages, and Git
history. It is not copied into test comments. A historical name may remain only
when the same spelling is a genuine production-domain term rather than a plan
identifier; no current exception is established for `cNN` or `sequenceN` test
names.

The naming rule has no mechanical word, character, or physical-line ceiling.
Descriptive test names are an intentional local exception to ordinary short
item naming when shortening them would hide the condition or outcome.

## S08 Behavioral Preservation

Cleanup must preserve focused evidence for every currently supported category:

- scene/model validation, normalization, geometry, transforms, paint, images,
  layers, backgrounds, borders, clips, masks, filters, and text paint hooks;
- direct Vello and GPU-graph route selection and pixels;
- high- and reduced-precision policy and capability reporting;
- resource identity, cache behavior where externally meaningful, transaction
  accounting, failure atomicity, cancellation, presentation, and retained
  publication;
- explicit headless readback semantics;
- typed unsupported-operation and unavailable-runtime diagnostics;
- shader numeric behavior through existing vectors or rendered output;
- native default, `render-window`, `render-web`, combined-feature, and
  `wasm32-unknown-unknown` compile behavior documented by the repository.

Exact private pipeline keys, cache table shapes, helper ownership, call graphs,
and pass implementation vocabulary are not independently preserved unless they
are already an observable public or crate-front-door contract. A task reviewer
must compare each deletion or consolidation against adjacent tests and affected
behavior, not infer preservation from a green aggregate count.

## S09 Implementation Shape

`src/tests.rs` remains the crate-level behavioral suite during this initiative;
moving it into a hierarchy is reserved for the separately discussed
hierarchical-front-door cleanup. P02 may extract or remove private test helpers
only when that directly improves an affected test's condition-focused evidence.

The implementation records disposition in the reviewed commit diff and task
evidence. It does not add a tracked disposition ledger, test inventory, parser,
snapshot, generated artifact, or policy mirror.

Behavior-preserving cleanup first establishes the existing focused test as
characterization evidence. A replacement behavior test demonstrates RED only
when it expresses a condition that the existing suite did not exercise. Pure
deletion, renaming, or data-table consolidation must not fabricate a behavior
change to satisfy a RED ritual.

## S10 Product Impacts

- Public API: internal-only; no intended source or behavior change.
- Dependencies and features: unchanged; no new dependency or feature.
- Generated artifacts: none in this leaf; root-owned API artifacts remain
  untouched.
- Documentation and examples: unchanged unless a stale test-only reference is
  removed from an existing comment. P02 adds no replacement policy document.
- MSRV: preserve root integration compatibility with Rust 1.97.
- Migration: none for callers.
- Root integration: pointer evaluation only after a published leaf candidate;
  no facade, adapter, or generated-artifact delta is expected.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## S11 Verification Contract

Focused task evidence must include:

- pre-change execution of every affected behavioral test or matrix used as
  characterization;
- post-change execution of each retained, renamed, consolidated, or replacement
  condition;
- an explicit diff-based mapping showing why each deleted test had no unique
  enduring oracle, supplied to reviewers as task evidence rather than committed
  source;
- zero `#[test]` names carrying the plan-identifier patterns described in S07;
- zero test dependencies on repository source or document text described in
  S06;
- no orphaned parser, manifest-audit, source-scan, closure, inventory, or
  fixture support left solely for deleted tests.

The final repository command set is derived from `AGENTS.md`, `Cargo.toml`, and
`README.md` and uses already installed artifacts with Cargo offline:

```sh
CARGO_NET_OFFLINE=true cargo fmt --check
CARGO_NET_OFFLINE=true cargo check -p surgeist-render
CARGO_NET_OFFLINE=true cargo test -p surgeist-render
CARGO_NET_OFFLINE=true cargo clippy -p surgeist-render --all-targets -- -F unsafe-code -D warnings
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-web
CARGO_NET_OFFLINE=true cargo test -p surgeist-render --features render-window,render-web
```

The current installed wasm target, Rust 1.97 toolchain, dependency inspection,
warning-denied rustdoc, native presentation smoke, and canonical owned-Rust
unsafe scan remain applicable final evidence when available under the existing
repository configuration. P02 does not install or update missing software.

## S12 Initiative Acceptance

P02 is accepted when the suite expresses the durable contract in S04-S08, all
repository-text and plan-provenance scaffolding is absent, every removed or
consolidated test has reviewer-checked preservation evidence, all configured
checks pass, and the reviewed leaf candidate is published to the authority
remote `main` with a complete root handoff. A lower test count alone is neither
necessary nor sufficient evidence.
