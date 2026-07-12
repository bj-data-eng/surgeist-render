# GPU Render Pipeline Implementation Sequence

## Authority

- Owning repository: `surgeist-render`.
- Desired state: `plans/specs/gpu-render-pipeline.md` at commit `3826a9098e859874a515bbebaf470a47d754d76c`, content SHA-256
  `f01972e19f8a5ddc90936edfc6ea7955feff3d1b1fdf5d181e77e8d10cc1f60a`.
- Specification review disposition: the user explicitly waived the remaining clean-context re-review after the third review's findings were incorporated.
- Ordering rule: a cycle becomes ready only after every prerequisite cycle is
  landed, published, remotely verified, and handed forward without a material specification or sequence change.
- Root integration remains outside this sequence and outside this repository.

## C01 Public Contract Foundation

- Owner: `surgeist-render`.
- Outcome: establish Options, text-bounds, runtime/error foundations, disable
  Vello CPU selection, and remove the unsafe resize hint before any cycle can
  publish; legacy capability/statistics APIs remain unchanged until C11.
- Specification: S07-S08, S10, S12, non-readback S13, safe-resize S26, S29
  Options/non-readback-error/runtime/text rows, and S35.
- Prerequisites: none.
- Entry state: `main` contains the authority specification and this reviewed
  sequence, with no unowned change.
- Exit evidence: available native model/feature evidence is valid, legacy
  capability/stat reports remain truthful, Vello CPU selection is absent, and
  all owned source is `unsafe`-free; C12 retains actual wasm compilation.
- Handoff: C02 consumes the public runtime/error/stat contracts; C12 owns target-specific compilation; no root handoff.

## C02 Async GPU Transactions And Device Terminality

- Owner: `surgeist-render`.
- Outcome: make create/render/resume/present asynchronous, establish identity
  before slot access and draft-versus-published headless atomicity, and provide
  scoped errors, per-device terminality, cancellation cleanup, and presentation;
  C03 exclusively owns readback.
- Specification: S12-S13, non-readback S13A, identity/device/publication S25-S26,
  S29 async create/render/resume, identity, and publication rows, S31, S35.
- Prerequisites: C01.
- Entry state: C01 contracts are the published public front door.
- Exit evidence: identity precedes indexing, canceled/failed headless frames
  preserve published pixels, and scoped non-readback operations follow terminal
  mapping without cross-device damage; blocking readback remains owned by C03.
- Handoff: C03 completes the async front door by replacing all explicit and temporary legacy readback progress; no root handoff.

## C03 Surface Lifecycle And Readback

- Owner: `surgeist-render`.
- Outcome: close remaining lifecycle semantics and replace explicit plus legacy internal readback with cancellation-safe native/wasm async progress over
  C02's published headless target.
- Specification: S09 `ImageBuffer`, readback S13-S13A, S25-S26, S28, S29
  `read_headless`/`ReadbackFailed`/image/surface behavior rows, S31, S33-S35.
- Prerequisites: C02.
- Entry state: non-readback GPU entrypoints and terminal states are async and stable; explicit and legacy materialization readback use the old blocking helper.
- Exit evidence: available native evidence covers every remaining surface state
  and one nonblocking readback state machine with complete cleanup; C12 retains
  wasm compilation evidence.
- Handoff: C04 consumes stable surface/output semantics and C12 consumes the wasm branch; no root handoff.

## C04 Frame, Spatial, And Filter Planning

- Owner: `surgeist-render`.
- Outcome: establish the closed direct-versus-graph plan, immutable resource
  dependency graph, Vello partitioning, signed spatial model, text-effect
  bounds, ordered filter bounds, and explicit fan-out lifetimes.
- Specification: S15, S17, S19, private planning portions of S20/S22, S28, and S31-S32; C09 owns their public constructors/payloads.
- Prerequisites: C03.
- Entry state: renderer operations and surfaces expose stable async/lifecycle
  contracts, while current production pixel routes remain intact.
- Exit evidence: all supported authored scenes normalize to one finite plan,
  graph validation rejects invalid dependencies, and spatial/filter planning
  requires no backend execution or guessed resource bounds.
- Handoff: C05 consumes only validated graph resources, pass intents, and extents; no root handoff.

## C05 Working Formats, Resources, And Pass Infrastructure

- Owner: `surgeist-render`.
- Outcome: establish deterministic effect-format selection, persistent
  per-device resource ownership, generation-aware leases, budgets, shader and
  pipeline caches, safe uniform encoding, and the finite executable pass
  vocabulary.
- Specification: S12, S16, S18, S25, S28, and S31-S35.
- Prerequisites: C04.
- Entry state: every future GPU operation has a validated plan and bounded
  resource description.
- Exit evidence: high/reduced selection and failure are typed, reuse/trimming are deterministic, internal telemetry is coherent, and no public backend
  type, final route statistic, capability flip, or dependency is added.
- Handoff: C06 consumes the resource manager, working-format decision, and pass lowering boundary; no root handoff.

## C06 Direct And Capture Graph Spine

- Owner: `surgeist-render`.
- Outcome: retain one-pass direct Vello rendering while executing root clear,
  bounded capture, canonicalization, minimal root source-over, output conversion,
  and headless/presented delivery through the GPU graph spine.
- Specification: S15-S19, minimal root/source-over/present S23, S25-S26, S28,
  and S31-S34.
- Prerequisites: C05.
- Entry state: graph plans lower to owned resources and a finite pass set, but
  no migrated CSS effect depends on the graph spine.
- Exit evidence: the private production executor preserves capture/source-over/
  output contracts in both precisions without readback or Vello re-entry; public
  route/stat/capability surfaces remain on the legacy contract until C11.
- Handoff: C07 consumes a complete source-to-output GPU graph; no root handoff.

## C07 Composition, Clip, Mask, And Blend Passes

- Owner: `surgeist-render`.
- Outcome: implement advanced ordered composition for outer clips, resolved
  alpha masks at all image qualities, opacity, isolation, and supported blends.
- Specification: S09, S16, S18-S19, S23, S25, S27-S28, S29 resolved-mask API/Eq
  rows, and S30-S34; C11 owns legacy execution-type removal and capability names.
- Prerequisites: C06.
- Entry state: bounded Vello content can become and remain a canonical GPU
  working image through output.
- Exit evidence: private GPU composition uses exact ordering/sampling while
  broad diagnostics and all public route/stat/capability reports retain their
  truthful legacy state until C11.
- Handoff: C08 may compose color-filter results through the same graph boundary; no root handoff.

## C08 Ordered Color-Filter Execution

- Owner: `surgeist-render`.
- Outcome: execute brightness, contrast, grayscale, hue rotation, invert,
  opacity, saturation, sepia, and legal fusion on the GPU with finite scalar
  lowering and a clamp after every authored operation.
- Specification: S16, S18, S20-S21, S27-S28, and S30-S34.
- Prerequisites: C07.
- Entry state: graph content can be isolated, composited, and delivered without
  a CPU pixel edge.
- Exit evidence: private GPU color execution agrees with independent constants
  and precision tolerances without changing public route/capability/stat reports
  or enabling broad layer/reference filters.
- Handoff: C09 composes spatial filters with the ordered color path; no root handoff.

## C09 Gaussian Blur And Filter Drop Shadow

- Owner: `surgeist-render`.
- Outcome: execute ordinary Gaussian blur and CSS filter drop shadow through
  GPU image passes with transparent edges, continuous offset, SourceAlpha,
  source fan-out, expanded signed bounds, and solid-paint diagnostics.
- Specification: S16, S18-S22, S25, S27-S28, applicable S29
  `FilterBlur`/`FilterDropShadow` rows, and S30-S34.
- Prerequisites: C08.
- Entry state: ordered color filters and composition share one canonical
  working-image contract.
- Exit evidence: private GPU pixels, bounds, lifetimes, and mixed order preserve
  required clamps/precision without silent quality reduction, replay, CPU pixels,
  public report changes, or non-solid support expansion.
- Handoff: C10 consumes the complete supported filter pass chain; no root handoff.

## C10 Bounded Backdrop Execution

- Owner: `surgeist-render`.
- Outcome: execute the supported bounded backdrop subset by copying completed
  parent pixels once, applying the ordered GPU filter chain with backdrop edge
  semantics, and compositing unfiltered foreground in authored order.
- Specification: S16-S17, S19-S20, S23-S24, S27-S28, and S30-S34.
- Prerequisites: C09.
- Entry state: every filter operation allowed inside a bounded backdrop has a
  composable GPU implementation.
- Exit evidence: private GPU backdrop behavior preserves base, siblings, clip,
  foreground, and later observers without replay/readback or public report
  changes; root/nested/transformed policies remain diagnostics.
- Handoff: C11 may replace every selected legacy materialization route with the complete graph; no root handoff.

## C11 GPU-Only Cutover And Matrix Reconciliation

- Owner: `surgeist-render`.
- Outcome: atomically expose direct/GPU-graph routes, final S11 capabilities and
  S14 statistics, retire CPU/materialized execution and superseded public phases,
  isolate the oracle to tests, and reconcile every final report.
- Specification: S03, S11, S14, S20, S25, S27, all remaining S29 capability/
  statistics/materialized/CPU/removal/diagnostic rows, S30-S30C, S35, and S38.
- Prerequisites: C10.
- Entry state: composition, color, blur/shadow, and bounded backdrop primitives
  are independently GPU-complete and composable.
- Exit evidence: all 101 primitive rows, 22 property mappings, and 53 typed
  subcases reconcile mechanically; production contains no CPU pixel route,
  Vello CPU selection, graph readback/re-entry, stale public phase, or owned
  `unsafe`.
- Handoff: C12 receives the final production architecture and public surface; no root handoff yet.

## C12 Platform Evidence, Documentation, And Final Quality

- Owner: `surgeist-render`.
- Outcome: complete native feature combinations, presented direct/graph smoke,
  applicable wasm compilation, MSRV evidence, API/docs/example updates, and the
  full deterministic and real-GPU quality/lifecycle acceptance contract.
- Specification: S02, S04-S06, S31-S38.
- Prerequisites: C11.
- Entry state: final GPU-only behavior and matrix reconciliation are published
  on the leaf mainline.
- Exit evidence: every available required feature, target, quality, lifecycle,
  lint, formatting, unsafe-absence, and Rust-1.89 gate is green; unavailable
  toolchain/host evidence is reported only through its exact permission or host
  blocker.
- Handoff: publish the fetchable leaf candidate and report the complete public
  API migration, target evidence, browser-host follow-up, and root-owned facade,
  artifact, and submodule work to the root coordinator.
