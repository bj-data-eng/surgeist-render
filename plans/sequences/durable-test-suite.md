# P02-I01-S01 Durable Test Suite Sequence

## 1 Source

- Project: `P02`.
- Initiative: `P02-I01` durable test suite.
- Sequence: `P02-I01-S01`.
- Owning repository: `surgeist-render`.
- Specification: `plans/specs/durable-test-suite.md` at
  `79f68da934322a13f286a64d6d7df48213ca5046`, normalized SHA-256
  `100e4972bfe4237f6b7bc89dc9b2821c71a3e09c5cb9a92ccb51ce7985dbabbb`.
- Initiative outcome: the retained suite tests enduring observable conditions,
  contains no completed-plan provenance or repository-text inspection, and
  remains behaviorally complete without a numerical cleanup gate.

## 2 Ordering Constraints

- C01 removes repository-text infrastructure before broader consolidation so
  later cycles do not preserve or relocate closure scaffolding.
- C02 and C03 partition the behavioral suite by owned domain. They follow C01
  so source-based assertions cannot be mistaken for characterization evidence.
- C03 follows C02 because shared crate-level fixtures currently mix authored
  model inputs with runtime execution and must have one settled owner before
  execution-focused consolidation.
- C04 follows every domain disposition and closes only the cross-domain naming,
  fixture, feature, and preservation surface.
- P02-I02 cohesive module decomposition begins only after C04 is published and
  handed off; no I02 file move belongs to this sequence.

## 3 Ordered Cycles

### C01 Repository-Text And Closure Scaffolding Retirement

- Owning repository: `surgeist-render`.
- Specification sections: S01-S06, S08-S12.
- Prerequisites: specification review `CLEAN`; leaf `main` at the recorded
  cycle base; no active implementation cycle.
- Entry state: 31 tests depend on Rust or WGSL source text, four additional
  tests parse manifest text, and source-scanner/closure support remains in the
  test suite.
- Bounded outcome: delete completed-plan and repository-text-only tests and
  their support; replace only the enduring observable conditions not already
  covered by behavior.
- Exit evidence: affected behavior remains covered through front-door or typed
  observations; no test reads repository text; source-scanner, manifest-audit,
  inventory, and closure-only support is absent; configured checks pass.
- Handoff: publish the C01 leaf candidate and confirm C02 may classify the
  authored/model/style suite without source-text dependencies.

### C02 Authored Model And Style Test Consolidation

- Owning repository: `surgeist-render`.
- Specification sections: S01, S04-S05, S07-S12.
- Prerequisites: published C01 candidate and its complete handoff.
- Entry state: authored and normalized geometry, scene, paint, image, layer,
  text, style, capability, and validation tests use behavioral evidence but
  retain P01-era naming, repetition, and fixture overlap.
- Bounded outcome: give every in-scope test one retain, consolidate, replace,
  or delete disposition; retain distinct normal, boundary, invalid, conversion,
  normalization, and diagnostic conditions with condition-focused names.
- Exit evidence: the authored/model/style domains contain no completed-plan
  identifiers, duplicate oracle cases, or fixtures without a distinct
  behavioral owner; their focused and configured checks pass.
- Handoff: publish the C02 leaf candidate and return the settled shared authored
  fixtures required by runtime execution tests.

### C03 Runtime GPU Surface And Lifecycle Test Consolidation

- Owning repository: `surgeist-render`.
- Specification sections: S01, S04-S05, S07-S12.
- Prerequisites: published C02 candidate and its complete handoff.
- Entry state: frame, graph, pass, shader, backend, renderer, resource,
  transaction, Vello, surface, presentation, and readback tests remain
  behaviorally valid but retain P01-era naming, repeated matrices, and
  fixture-specific overlap.
- Bounded outcome: give every in-scope test one semantic disposition while
  preserving distinct pixels, routes, capabilities, precision, errors,
  ordering, resource lifetimes, failure atomicity, cancellation, publication,
  presentation, and explicit-readback conditions.
- Exit evidence: the runtime/GPU/surface domains contain no completed-plan
  identifiers or duplicate behavior with the same operation and oracle;
  focused GPU, lifecycle, pixel, and configured checks pass.
- Handoff: publish the C03 leaf candidate and return the complete domain
  dispositions for final cross-domain reconciliation.

### C04 Cross-Domain Suite Reconciliation

- Owning repository: `surgeist-render`.
- Specification sections: S01-S12.
- Prerequisites: published C03 candidate and its complete handoff.
- Entry state: every domain has been dispositioned independently; only
  cross-domain fixture ownership, complete naming closure, and final feature,
  target, MSRV, documentation, presentation, and safety evidence remain.
- Bounded outcome: remove orphaned or cross-domain duplicate test support,
  confirm every retained test states a condition and outcome, and reconcile the
  complete suite without adding a count, naming, source, or architecture gate.
- Exit evidence: all initiative acceptance criteria hold; the complete
  configured verification matrix passes; the reviewed leaf candidate is
  published with no public API, dependency, feature, or behavior delta.
- Handoff: return the final P02-I01 crate candidate to root and establish the
  published prerequisite for P02-I02 cohesive module decomposition.

## 4 Sequence Completion

The sequence is complete only after C04 is reviewed, published to the authority
remote `main`, read back, and handed off. Later I02 work uses that immutable
candidate as its baseline and does not reinterpret an unfinished I01 cycle.
