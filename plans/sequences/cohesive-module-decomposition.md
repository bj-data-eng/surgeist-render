# P02-I02-S01 Cohesive Module Decomposition Sequence

## 1 Source

- Project: `P02`.
- Initiative: `P02-I02` cohesive module decomposition.
- Sequence: `P02-I02-S01`.
- Owning repository: `surgeist-render`.
- Published prerequisite: P02-I01 at
  `4f7fcb8b81d96f16b426f045b336aaba345c4cfa`.
- Specification: `plans/specs/cohesive-module-decomposition.md` at
  `314b8252e8db18130abb8031033b5a0be624c81a`, normalized SHA-256
  `415257797bf18fd6d6a2d3e5a9ffcd07bc42793490da56505a83e7300aa6d1bb`.
- Initiative outcome: private modules own the cohesive responsibilities in
  specification M05, and non-plan filenames and code use rendering-domain names
  instead of planning chronology. Public API, behavior, dependencies, features,
  targets, and safety remain unchanged; diagnostic wording changes only to
  remove planning identifiers, and no numerical size gate is introduced.

## 2 Ordering Constraints

- Every cycle starts from the published, reviewed predecessor and publishes its
  own reviewed leaf candidate before the next cycle starts.
- Moves are ordered by semantic dependency and current caller concentration,
  not by physical line count.
- Authored and CPU-reference domains move first because later planning and test
  modules consume their front doors without owning their definitions.
- Frame planning precedes runtime-pass decomposition because pass lowering
  consumes frame-owned lowering views.
- Shader and resource front doors precede pass preparation because pass
  preparation consumes their realization, cache, key, and lease contracts.
- The pass monolith is split across two cycles. The first establishes runtime
  model, closure, and lowering owners; the second moves preparation, encoding,
  parameter construction, and pass-owned test support. The intermediate
  `pass/mod.rs` may retain the not-yet-moved second-cycle implementation, but it
  may not introduce a temporary shim or alternate model.
- Encoding-coupled test observations may move with their production value during
  C05 only when required to preserve per-value semantics; C05 extracts them to
  pass test support before its final state and introduces no global bridge.
- GPU transaction and readback ownership precede backend decomposition so the
  backend can import stable transaction/readback front doors while it moves.
- Renderer moves after frame, pass, resource, transaction, readback, and backend
  owners are settled because it orchestrates all of them and owns the public
  option types used by several allowed baseline edges.
- The crate-level test hierarchy moves last. This keeps behavioral oracles
  stable while production paths change and prevents repeated test import churn.
  Its first cycle establishes shared support and lower-dependency domains; its
  second cycle completes runtime/platform domains and removes the old monolith.
- The eight allowed bidirectional module-directory edges in specification M06
  may remain. No cycle may add another bidirectional edge or disguise an
  intra-directory cycle.

## 3 Ordered Cycles

### C01 Authored Style And Reference Domains

- Specification sections: M01-M05.5, M06-M09.
- Prerequisite: specification review `CLEAN`; local and authority-remote `main`
  agree at the recorded cycle base.
- Bounded outcome: replace `src/style.rs` with `src/style/mod.rs` plus
  `image.rs`, `filter.rs`, `clip.rs`, `mask.rs`, `decoration.rs`, and
  `background.rs`; replace test-only `src/reference.rs` with
  `src/reference/mod.rs` plus `color.rs`, `filter.rs`, and `mask.rs`.
- Mechanical boundary: preserve every style root reexport, constructor,
  validation order, normalization result, and reference-oracle byte result.
- Exit evidence: pre/post focused style and reference tests; all affected
  native feature tests and Clippy; unchanged `src/lib.rs` export surface;
  configured cycle matrix; task and holistic reviews.
- Handoff: publish stable authored/reference front doors for frame and later
  test-domain moves.

### C02 Frame Planning

- Specification sections: M01-M05.1, M06-M09.
- Prerequisite: published C01 candidate.
- Bounded outcome: replace `src/frame.rs` with `src/frame/mod.rs` plus
  `bounds.rs`, `filter.rs`, `graph.rs`, `validate.rs`, `lower.rs`, and
  `test_support.rs` with the exact responsibilities from M05.1.
- Mechanical boundary: preserve plan selection, graph identities, validation
  precedence, lowering views, spatial/filter facts, and crate-visible frame
  contracts; retain the allowed frame/renderer edge.
- Exit evidence: pre/post route, bounds, graph, validation, and lowering tests;
  default and affected feature checks/Clippy; unchanged front door and graph
  observations; configured cycle matrix; task and holistic reviews.
- Handoff: publish the stable semantic-planning and lowering-view owners used by
  pass decomposition.

### C03 Shader And Resource Infrastructure

- Specification sections: M01-M04, M05.4, M06-M09.
- Prerequisite: published C02 candidate.
- Bounded outcome: replace `src/shader.rs` with `src/shader/mod.rs` plus
  `parameters.rs`, `key.rs`, `validate.rs`, `pipeline.rs`, `cache.rs`, and
  `test_support.rs`; replace `src/resource.rs` with `src/resource/mod.rs` plus
  `gaussian.rs`, `manager.rs`, `lease.rs`, and `test_support.rs`.
- Mechanical boundary: preserve byte serialization, shader/cache identity,
  validation precedence, WGPU construction, Gaussian results, accounting,
  leasing, retention, and the allowed shader/pass, resource/backend, and
  resource/renderer edges.
- Exit evidence: pre/post shader vectors, key/cache, Gaussian, allocation,
  accounting, lease, and retention tests; affected feature checks/Clippy;
  unchanged crate-visible contracts; configured cycle matrix; task and
  holistic reviews.
- Handoff: publish stable realization, cache, key, kernel, and resource owners
  consumed by pass preparation.

### C04 Runtime Pass Model Closure And Lowering

- Specification sections: M01-M04, M05.2, M06-M09.
- Prerequisite: published C03 candidate.
- Bounded outcome: convert `src/pass.rs` to `src/pass/mod.rs`; move runtime
  resource/pass/read/result/filter/composite/spatial/cache-key facts to
  `model.rs`, executable closure/accounting validation to `close.rs`, and frame
  graph conversion to `lower.rs`.
- Explicit intermediate state: preparation, encoding, pass-specific parameter
  construction, and pass-owned test support remain directly owned by
  `pass/mod.rs` for C05; they are not copied, wrapped, or renamed.
- Mechanical boundary: preserve runtime model values, closure diagnostics and
  ordering, accounting validation, lowering facts, and allowed pass edges.
- Exit evidence: pre/post model construction, graph closure, invalid-graph,
  accounting, and lowering tests; all affected feature checks/Clippy; explicit
  inventory of items intentionally retained in `pass/mod.rs`; configured cycle
  matrix; task and holistic reviews.
- Handoff: publish stable runtime model/closure/lowering children and the exact
  remaining C05 inventory.

### C05 Runtime Pass Preparation And Encoding

- Specification sections: M01-M04, M05.2, M06-M09.
- Prerequisite: published C04 candidate and its remaining-item inventory.
- Bounded outcome: move allocation, kernel/pass analysis, preflight,
  realization, and prepared bindings to `prepare.rs`; move graph encoding,
  capture handoff, scheduling, and receipts to `encode.rs`; move pass-specific
  uniform construction to `parameters.rs`; move pass-owned fixtures,
  injections, malformed plans, and observations to `test_support.rs`; replace
  every planning identifier in tracked non-plan filenames and Rust/WGSL code
  with the semantic M04.6 vocabulary.
- Boundary: leave only true pass orchestration and explicit
  crate-visible reexports in `pass/mod.rs`; preserve preparation failure
  atomicity, cache/resource publication, encoded order, receipt facts, and all
  allowed pass edges; preserve per-summary test observations during their staged
  move and change diagnostic prose/labels only to remove planning chronology.
- Exit evidence: pre/post preparation, encoding, parameter, failure,
  cancellation, and observation tests; all feature checks/Clippy; no remaining
  M05.2-owned item in `pass/mod.rs`; empty transient non-plan filename and code
  predicates; configured cycle matrix; task and holistic reviews.
- Handoff: publish the complete runtime-pass private hierarchy and semantically
  named non-plan source.

### C06 GPU Transaction And Readback

- Specification sections: M01-M04, M05.3 transaction table, M05.5 readback
  table, M06-M09.
- Prerequisite: published C05 candidate.
- Bounded outcome: replace `src/gpu_transaction.rs` with
  `src/gpu_transaction/mod.rs` plus `graph.rs`, `vello.rs`, `readback.rs`, and
  `test_support.rs`; replace `src/readback.rs` with `src/readback/mod.rs` plus
  `layout.rs`, `lifecycle.rs`, `native.rs`, and `test_support.rs`.
- Mechanical boundary: preserve transaction stages, submission/commit proofs,
  post-submit behavior, readback row/decode facts, callback/polling behavior,
  future ownership, cleanup, cancellation, and the allowed transaction/backend
  edge.
- Exit evidence: pre/post transaction, submission, commit, cancellation,
  readback layout/lifecycle/native, and future tests; native/feature checks and
  Clippy; configured cycle matrix; task and holistic reviews.
- Handoff: publish stable transaction and readback front doors for backend and
  renderer decomposition.

### C07 Backend Runtime

- Specification sections: M01-M04, M05.3 backend table, M06-M09.
- Prerequisite: published C06 candidate.
- Bounded outcome: replace `src/backend.rs` with `src/backend/mod.rs` plus
  `device.rs`, `execute.rs`, `offscreen.rs`, `present.rs`, `texture.rs`, and
  `test_support.rs`.
- Mechanical boundary: preserve device lifecycle/signals/capabilities, direct
  and graph execution, target construction, surface acquisition/configuration,
  presentation, safe texture creation, failure injection, and all allowed
  backend edges.
- Exit evidence: pre/post device, execution, offscreen, presented lifecycle,
  texture, failure, and publication-adjacent tests; all four feature checks,
  tests, and Clippy configurations; both live native smokes and wasm check;
  configured cycle matrix; task and holistic reviews.
- Handoff: publish the stable backend front door and runtime children.

### C08 Renderer Orchestration

- Specification sections: M01-M04, M05.3 renderer table, M06-M09.
- Prerequisite: published C07 candidate.
- Bounded outcome: replace `src/renderer.rs` with `src/renderer/mod.rs` plus
  `dispatch.rs`, `publication.rs`, `options.rs`, and `test_support.rs`.
- Mechanical boundary: preserve every public renderer/options type and root
  reexport, route selection, typed unsupported translation, preparation and
  execution dispatch, publication/failure atomicity, and all allowed renderer
  edges.
- Exit evidence: pre/post option, route, dispatch, capability, error,
  statistics, publication, and fault-control tests; public-surface comparison;
  all feature/target/host checks; configured cycle matrix; task and holistic
  reviews.
- Handoff: publish the complete production hierarchy and stable public
  orchestrator before moving the crate-level tests.

### C09 Test Support And Model Planning Domains

- Specification sections: M01-M04, M05.6, M06-M09.
- Prerequisite: published C08 candidate.
- Bounded outcome: convert `src/tests.rs` to `src/tests/mod.rs`; establish
  `support.rs` only for fixtures/oracles used by at least two sibling domains;
  move model/geometry/scene/paint/image/layer/text tests to `model.rs`, authored
  normalization tests to `style.rs`, and frame/graph/lowering/closure tests to
  `frame.rs`.
- Explicit intermediate state: GPU, surface, platform, and Vello tests remain
  directly in `tests/mod.rs` for C10 with one recorded disposition inventory;
  no forwarding wrapper or duplicate helper is introduced.
- Mechanical boundary: test names, operations, inputs, assertions, and oracles
  remain unchanged except import paths and the smallest visibility required by
  real sibling ownership.
- Exit evidence: exact pre/post test counts and names for moved domains as
  relocation evidence rather than a permanent gate; focused/full tests; all
  feature checks/Clippy; helper-use inventory; configured cycle matrix; task and
  holistic reviews.
- Handoff: publish settled model/style/frame test owners and the exact C10
  remaining-test inventory.

### C10 Runtime Platform Test Hierarchy And Reconciliation

- Specification sections: M01-M09.
- Prerequisite: published C09 candidate and remaining-test inventory.
- Bounded outcome: move shader/graph execution/pixel/precision/resource/cache/
  transaction tests to `gpu.rs`; surface/publication/cancellation/readback tests
  to `surface.rs`; target/feature/example behavior to `platform.rs`; internal
  Vello characterization/parity tests to `vello.rs`; reconcile `support.rs` and
  reduce `tests/mod.rs` to declarations and genuine suite-level coordination.
- Mechanical boundary: every remaining test keeps the same operation and
  oracle; helpers used by one domain move into that child; no source parser,
  planning name/path, inventory test, size gate, or architecture enforcement
  remains.
- Exit evidence: complete relocation/disposition inventory; unchanged public
  surface and observable behavior; full specification M07 matrix including
  live smokes, wasm, MSRV, rustdoc, dependency views, Cargo.lock absence, and
  canonical unsafe scan; task reviews, final holistic review, publication, and
  authority-remote readback.
- Handoff: return the reviewed published P02-I02 leaf candidate; root work is
  excluded unless separately requested by root.

## 4 Cross-Cycle Invariants

- No cycle changes `src/lib.rs` public exports except a private `mod` file-to-
  directory resolution that preserves the same module name.
- No public signature, default, behavior, dependency, feature, target, example
  contract, or safety policy changes. Diagnostics change only to replace planning
  chronology with equivalent rendering-domain context.
- No algorithm or test oracle changes. A discovered semantic defect is reported
  separately and is not repaired within this sequence.
- No `include!`, `#[path]`, generated concatenation, copied definition,
  compatibility shim, glob-reexport maze, or generic helper bucket is used to
  make a move compile.
- Visibility remains private or uses the smallest `pub(super)`/`pub(crate)`
  widening justified by an existing caller.
- Every cycle records pre-move characterization, post-move focused evidence,
  the affected feature matrix, file moves, visibility changes, and unchanged
  public-surface evidence.
- Planning artifacts and Git history retain relocation provenance. No permanent
  source parser, file-size lint, item inventory, or planning closure test is
  added.

## 5 Completion

P02-I02 completes only after C01-C10 are each task-reviewed, holistically
reviewed, published, and read back in order; every M05 child owns its specified
responsibility; the public front door and observable rendering behavior remain
unchanged; tracked non-plan filenames and Rust/WGSL code contain no planning
identifier; the full M07 evidence is green; and no numerical size enforcement or
root integration change has been introduced.
