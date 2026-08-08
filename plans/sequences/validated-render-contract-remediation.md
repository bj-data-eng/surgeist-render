# P03-I01-S01 Validated Render Contract Remediation Sequence

## 1 Source

- Project: `P03`.
- Initiative: `P03-I01` validated render contract remediation.
- Sequence: `P03-I01-S01`.
- Owning repository: `surgeist-render`.
- Initiative baseline: published/read-back `main` at
  `b02fa0c372472c88a511f45cb74b1ec0b356d181`.
- Specification: `plans/specs/validated-render-contract-remediation.md` at
  `c8c7fabef9db0494a01cfd2558f5174baa714db5`, normalized SHA-256
  `1b57af4471b8bcea9f73f4bff5723227222083ba0d8ca367e14bc1155603933b`.
- Specification review: `CLEAN` with no findings.
- Initiative outcome: backend image identity and render-owned content reuse are
  collision-safe, public rectangles cannot expose non-finite maxima, the current
  public model is fully documented, and the unused command-statistics path is
  removed without root, sibling, dependency, feature, or permanent tooling work.

## 2 Ordering Constraints

- C01 precedes C02 because public documentation must describe the corrected
  image identity and rectangle invariants rather than an immediately obsolete
  contract.
- C01 owns all behavior and private model changes. C02 is documentation-only and
  may not revisit identity, caching, validation, statistics, or rendering
  behavior.
- C01 publishes and receives authority-remote readback before C02 is planned.
  C02 records that published candidate as its cycle base.
- Both cycles remain inside the leaf. Root facade, adapters, API artifacts, and
  gitlink promotion are excluded and receive only the final candidate handoff.
- Neither cycle adds a dependency, feature, script, generator, parser, CI rule,
  permanent lint, source-text test, plan-closure test, or `unsafe` code.

## 3 Ordered Cycles

### C01 Collision-Safe Identity And Finite Geometry

- Owning repository: `surgeist-render`.
- Specification sections: R01 items 1, 2, and 4; R02; R03.1, R03.2, R03.4;
  R04.1-R04.4, R04.6; applicable R05 rows; R06.1, R06.2, R06.4; R07; and
  R08 items 1-4 and 6-10.
- Prerequisites: specification review `CLEAN`; clean leaf `main`; local and
  authority tracking refs agree at the initiative baseline; installed offline
  Rust 1.97 and wasm target remain available.
- Entry state: a 64-bit content hash is supplied as Peniko/Vello identity and is
  the sole image fact in retained-mask and upload-telemetry identity; public and
  canonical rectangle validation omit derived maxima; the unused command stats
  helper remains allowed as dead code.
- Bounded outcome: Peniko owns unique backend blob IDs; render-owned mask reuse
  and upload telemetry use exact content equality; both rectangle boundaries
  reject non-finite derived maxima; and the dead helper and allowance are gone.
- Exit evidence: deterministic forced-collision behavior, exact-content reuse,
  backend-ID separation, both derived-axis rejection and finite boundary cases,
  active renderer statistics behavior, configured feature/target/MSRV checks,
  no owned unsafe, clean task reviews, clean holistic review, leaf publication,
  and authority readback.
- Handoff: publish the corrected leaf contract as the immutable base for C02;
  report the precise public behavior correction for `Rect::try_new` and the
  clarified non-unique `ImageId` role.

### C02 Current Public Model Documentation

- Owning repository: `surgeist-render`.
- Specification sections: R01 item 3; R02; R03.3; R04.5; R06.3; R07; and R08
  items 5 and 7-10.
- Prerequisite: C01 is reviewed, landed, published, remotely verified, and its
  corrected identity and rectangle contracts remain unchanged.
- Entry state: strict missing-docs rustdoc reports undocumented items across the
  current flat public surface, including geometry, images, capability/error
  models, rendering primitives, and authored-style values still present pending
  the separately deferred cross-crate boundary initiative.
- Bounded outcome: every currently exported item, variant, public field,
  constructor, builder, conversion, default, and behavior-bearing method has
  accurate proportional rustdoc; strict missing-docs rustdoc succeeds without a
  committed lint or inventory mechanism.
- Exit evidence: zero strict missing-docs diagnostics, warning-free rustdoc and
  doctests, configured feature/target/MSRV checks, unchanged behavior and public
  visibility/reexports, no plan identifier outside `plans/`, no owned unsafe,
  clean task reviews, clean holistic review, leaf publication, and authority
  readback.
- Handoff: return the fully documented published leaf candidate to root without
  editing root-owned API artifacts or promoting the gitlink.

## 4 Completion

The initiative is complete only after both cycles satisfy their bounded exit
evidence and the C02 candidate is published on authority-remote `main` with
local-main, tracking-ref, and observed-remote agreement. The final handoff names
the candidate SHA, compatibility effects, verification evidence, and the
explicitly excluded root integration. The deferred authored-style ownership
issue remains separate and is neither resolved nor contradicted by this sequence.
