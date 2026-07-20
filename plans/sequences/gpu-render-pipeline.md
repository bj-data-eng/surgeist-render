# GPU Render Pipeline Implementation Sequence

## Authority

- Owning repository: `surgeist-render`.
- Desired state: `plans/specs/gpu-render-pipeline.md` at commit
  `fdbee86d599da8a4fba656a260ca1c910e53ac3d`, normalized semantic SHA-256
  `ca32ba5edc2e66b901934e9838facda9c54fdc5106d7f5e355677d61737a1f97`.
- Specification review disposition: `CLEAN` for that exact revision.
- Recovery base: published C01 commit
  `5361e3460278dffb877b9d485a2d12977977c3ef`; the former C02 plan is
  `superseded` and contributes no acceptance evidence.
- Ordering rule: a cycle starts only after every prerequisite cycle is clean,
  committed, published, remotely verified, and handed forward.
- Root facade adaptation, API artifacts, integration tests, and gitlink movement
  remain outside this leaf sequence.

## C01 Public Contract Foundation

- Outcome: establish options, text bounds, runtime/error foundations, GPU-only
  selection, and safe resize behavior without exposing final graph reports.
- Specification: S07-S08, S10, S12, non-readback S13, safe-resize S26, applicable
  S29 rows, and S35.
- Prerequisites: none.
- Entry state: initiative base `d59ad253300b68311f4e81a70e2b6ce73c922a4d` with external Vello ownership and legacy contracts.
- Exit evidence: accepted C01 range and remote publication remain authoritative;
  no successor reopens or rewrites it.
- Handoff: C02 adopts only the provisional post-C01 foundation range under S06B.

## C02 Foundation Adoption And Cleanup

- Outcome: audit and correct the provisional identity, async API, terminal-device,
  capability, error-scope, and transaction foundation; forward-remove the
  rejected external-Vello presented-setup/no-op seam.
- Specification: S06B, foundation portions of S07-S13A, identity/terminal portions
  of S25-S26, applicable S29 and S31-S32 rows, and S35.
- Prerequisites: C01.
- Entry state: C01 is the sole published base; commits `64bc5cb..9aa1d97` are
  provisional source evidence, not accepted implementation.
- Exit evidence: the exact `5361e34..C02-tip` range is holistically clean; current
  behavior remains available through unchanged temporary external Vello use,
  with no new external-Vello ownership or publication/readback claim.
- Handoff: C03 receives a clean backend-neutral transaction/device foundation.

## C03 Internal Vello Raster Engine Cutover

- Outcome: characterize pinned output, internalize/adapt the Vello 0.9 main crate,
  validate selected glyphs before external encoding, establish raster leases and
  transaction-owned encoding/submission, remove external `vello`, and close
  provenance plus its production dependency roles.
- Specification: S04-S06A, S07 raster phase, S10A, raster portions of S13A and
  S16-S17, S25-S26, S28-S29 font/API rows, and C03-applicable S31-S37 evidence.
- Prerequisites: C02.
- Entry state: async device/transaction ownership is clean, but external Vello
  still owns raster execution and surface/device conveniences.
- Exit evidence: production uses private checked WGPU raster modules with no CPU
  mode, unsafe, map/poll, direct engine submit, or silent glyph omission; current
  render behavior and characterization pixels remain supported.
- Handoff: C04 receives one Surgeist-owned raster/device/resource boundary.

## C04 Atomic Frame Publication And Presentation

- Outcome: implement headless draft-versus-published ownership, presented
  setup/configure/acquire/blit/present transactions, cancellation cleanup, exact
  wasm test dependency closure, and the non-readback lifecycle/error matrix.
- Specification: S12-S13A, publication/lifecycle portions of S25-S26, applicable
  S29 and S31-S37 evidence.
- Prerequisites: C03.
- Entry state: all raster work encodes and submits through Surgeist transactions.
- Exit evidence: failed/canceled headless frames preserve publication, presented
  state commits only after clean scopes/signals, wasm source/test targets compile
  with the exact test-only entropy role, and temporary readback remains unchanged.
- Handoff: C05 receives stable surface publication and lifecycle semantics.

## C05 Surface Readback

- Outcome: replace explicit and temporary blocking readback with the native/wasm
  cancellation-safe state machine over C04's published headless texture.
- Specification: S09 `ImageBuffer`, readback S13-S13A, S25-S26, S28-S29 readback
  rows, and applicable S31-S35 evidence.
- Prerequisites: C04.
- Entry state: non-readback operations and publication are async and atomic.
- Exit evidence: one explicit readback path owns copy/map/progress/cleanup; no
  production pass waits, maps, polls, or exposes partial bytes.
- Handoff: C06 may plan against stable inputs, outputs, and surface states.

## C06 Frame Spatial And Filter Planning

- Outcome: establish direct-versus-graph planning, immutable dependencies,
  maximal raster partitioning, signed spatial mapping, text-effect bounds,
  ordered filter bounds, and explicit fan-out lifetimes.
- Specification: S15, S17, S19, planning portions of S20 and S22, S28, and
  applicable S31-S32 evidence.
- Prerequisites: C05.
- Entry state: public operations, surfaces, raster execution, and readback have
  stable phase and lifecycle contracts.
- Exit evidence: every supported scene produces one finite validated plan without
  backend execution, guessed bounds, forward edges, or stale aliases.
- Handoff: C07 lowers only validated resources, pass intents, and extents.

## C07 Working Formats Resources And Pass Infrastructure

- Outcome: extend C03's one per-device resource manager to effect textures,
  masks, kernels, budgets, leases, shader/pipeline caches, safe serialization,
  quality selection, and the finite custom-pass vocabulary.
- Specification: S12, S16, S18, S25, S28, and applicable S31-S35 evidence.
- Prerequisites: C06.
- Entry state: graph plans have complete resource intents and bounded extents.
- Exit evidence: high/reduced selection is typed, reuse/trimming deterministic,
  one manager owns raster and effect allocations, and no final report is exposed.
- Handoff: C08 receives executable resources and pass lowering.

## C08 Direct And Capture Graph Spine

- Outcome: preserve one-pass direct rasterization while executing root clear,
  bounded capture, canonicalization, minimal source-over, output conversion, and
  headless/presented delivery through the GPU graph spine.
- Specification: S15-S19, minimal root/source-over/present S23, S25-S26, S28,
  and applicable S31-S34 evidence.
- Prerequisites: C07.
- Entry state: plans lower to one owned resource manager and a finite pass set.
- Exit evidence: direct and graph-spine pixels agree in both precisions without
  readback, CPU pixels, atlas re-entry, or premature public report changes.
- Handoff: C09 receives a complete source-to-output GPU graph.

## C09 Composition Clip Mask And Blend Passes

- Outcome: implement ordered outer clips, resolved alpha masks at every image
  quality, opacity, isolation, and all currently supported blend modes.
- Specification: S09, C09-applicable S11, S16, S18-S19, S23, S25,
  S27-S30, applicable S31-S34, and per-cycle S36-S37 evidence.
- Prerequisites: C08.
- Entry state: bounded raster content remains a canonical GPU working image.
- Exit evidence: GPU composition matches ordering/sampling oracles while broad
  diagnostics and legacy reports remain truthful.
- Handoff: C10 composes color-filter output through the same graph boundary.

## C10 Ordered Color Filter Execution

- Outcome: execute brightness, contrast, grayscale, hue rotation, invert,
  opacity, saturation, sepia, and legal fusion on the GPU with authored-order
  clamping and finite scalar lowering.
- Specification: S16, S18, S20-S21, S27-S28, and applicable S30-S34 evidence.
- Prerequisites: C09.
- Entry state: graph content can be isolated, composited, and delivered GPU-only.
- Exit evidence: high/reduced GPU results match independent constants/oracles
  without enabling broad layer/reference filters.
- Handoff: C11 composes spatial filters with the ordered color path.

## C11 Gaussian Blur And Filter Drop Shadow

- Outcome: execute Gaussian blur and CSS filter drop shadow through GPU image
  passes with correct edges, continuous offset, SourceAlpha, fan-out, and signed
  bounds while retaining non-solid-paint diagnostics.
- Specification: S16, S18-S22, S25, S27-S30, and applicable S31-S34 evidence.
- Prerequisites: C10.
- Entry state: color filters and composition share one working-pixel contract.
- Exit evidence: pixels, bounds, lifetimes, ordering, and precision match the
  oracle without readback, replay, or silent quality reduction.
- Handoff: C12 receives the complete supported filter pass chain.

## C12 Bounded Backdrop Execution

- Outcome: copy completed parent pixels once, apply the ordered GPU filter chain
  with backdrop edge semantics, and composite unfiltered foreground in authored
  order for the supported bounded subset.
- Specification: S16-S17, S19-S20, S23-S24, S27-S28, and applicable S30-S34.
- Prerequisites: C11.
- Entry state: every filter operation allowed inside a bounded backdrop is GPU
  implemented and composable.
- Exit evidence: base, sibling, clip, foreground, and later-observer behavior is
  correct without replay/readback; broader backdrop policy remains diagnostic.
- Handoff: C13 can replace every selected legacy materialization route.

## C13 GPU Only Cutover And Matrix Reconciliation

- Outcome: expose final remaining routes/capabilities/statistics, remove CPU/materialized
  execution and superseded public phases, isolate the oracle to tests, and
  reconcile every primitive/property/diagnostic inventory row.
- Specification: S01, S03, S11, S14, S20, S25, S27, remaining S29-S30C rows, S35,
  and S38.
- Prerequisites: C12.
- Entry state: every selected replacement primitive is independently GPU-complete.
- Exit evidence: production has no CPU pixel route, graph readback/re-entry,
  stale public phase, external Vello, owned unsafe, or matrix drift.
- Handoff: C14 receives the final production architecture and public surface.

## C14 Platform Evidence Documentation And Final Quality

- Outcome: complete native feature combinations, presented direct/graph smoke,
  wasm compilation, Rust 1.97 evidence, dependency/provenance inspection,
  API/docs/example updates, and full deterministic/real-GPU acceptance.
- Specification: S02-S06B and S31-S38.
- Prerequisites: C13.
- Entry state: final GPU-only behavior and matrix reconciliation are published.
- Exit evidence: every available required feature, target, quality, lifecycle,
  font-preflight, lint, format, unsafe-absence, and compatibility gate is green;
  unavailable required target/host evidence is an exact canonical blocker.
- Handoff: publish and remotely verify the leaf candidate, then report public API
  migration, browser-host follow-up, and root-owned facade/artifact/gitlink work.
