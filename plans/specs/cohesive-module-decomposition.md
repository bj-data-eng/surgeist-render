# P02-I02 Cohesive Module Decomposition

## M01 Outcome

The largest `surgeist-render` source files are decomposed into private modules
that each own one coherent rendering responsibility. The decomposition is
mechanical, and one bounded semantic cleanup replaces historical planning
identifiers in non-plan filenames and code with rendering-domain names. Public
API, rendering behavior, resource and lifecycle semantics, feature behavior,
and dependency direction remain unchanged; diagnostic wording changes only to
remove project chronology.

The initiative addresses file-level concentration, not function length. It does
not impose a physical-line lint, maximum file size, item-count target, or rule
that every file must be small. A file may remain large when its contents share
one owner and invariant; a file must be decomposed when the exact responsibility
map in M05 assigns its independent owners to child modules.

The initiative is complete when:

1. each current concentration named in M03 is represented by the private module
   hierarchy in M05;
2. each moved item has one explicit internal owner and imports through that
   owner instead of an `include!`, `#[path]`, generated concatenation, or copied
   definition;
3. crate-root exports and downstream construction paths remain unchanged;
4. behavior is characterized before each move and remains green after it;
5. no compatibility shim, forwarding layer, glob-reexport maze, or duplicated
   model is introduced solely to make the move compile;
6. the final hierarchy makes each named responsibility directly discoverable
   from its module path;
7. no filename or code artifact outside `plans/` retains an identifier whose
   meaning is a completed project, initiative, sequence, cycle, task, or
   specification-section number.

## M02 Ownership And Boundary

The owning repository is `surgeist-render`. This initiative may change private
module files, private visibility, internal imports, and test-module placement in
this leaf only.

In scope:

- `src/pass.rs`, `src/frame.rs`, `src/backend.rs`, `src/shader.rs`,
  `src/style.rs`, `src/renderer.rs`, `src/resource.rs`,
  `src/gpu_transaction.rs`, `src/reference.rs`, `src/readback.rs`, and the
  condition-focused suite resulting from P02-I01;
- conversion of an in-scope `name.rs` into `name/mod.rs` plus the exact private
  children in M05;
- the smallest `pub(super)` or `pub(crate)` visibility adjustment required for
  a real sibling-module relationship;
- internal import and documentation-link repair caused by those moves;
- semantic replacement of historical planning identifiers in every tracked
  filename and Rust or WGSL code artifact outside `plans/`, including symbols,
  diagnostics, labels, comments, and test support.

Out of scope:

- the public hierarchical front-door redesign discussed separately;
- public type, function, trait, error, default, feature, or reexport changes;
- semantic renaming other than removal of historical planning identifiers;
- algorithm, rendering, shader, capability, error, lifecycle, cache, or resource
  behavior changes;
- another function-size or file-size lint, tracked size ledger, generated module
  index, parser, CI rule, or permanent enforcement test;
- decomposition of a file not named in M02 unless a compiler-proven move of an
  in-scope owner requires relocating one inseparable adjacent item;
- root, sibling, adapter, API-artifact, and gitlink work.

P02-I01 is a prerequisite. The durable test cleanup must be published before
the first I02 implementation cycle so I02 characterizes the suite that the
crate intends to retain rather than repeatedly moving temporary closure tests.

## M03 Current Evidence

At baseline `4f7fcb8b81d96f16b426f045b336aaba345c4cfa`, repository-owned Rust
source contains 100,989 physical lines. The relevant
concentrations are:

| File | Physical lines | Independent responsibilities observed in source |
| --- | ---: | --- |
| `src/tests.rs` | 35,684 | model/style tests, graph planning, GPU execution, lifecycle, readback, platform, and internal Vello characterization |
| `src/pass.rs` | 16,710 | runtime graph model, graph closure validation, preparation, encoding, lowering, parameter construction, and test observation support |
| `src/frame.rs` | 7,242 | route selection, semantic bounds, graph building, graph validation, lowering views, spatial/filter planning, and graph test probes |
| `src/backend.rs` | 6,354 | device lifecycle, capabilities/signals, graph execution, offscreen rendering, presented rendering, texture construction, and test injection |
| `src/shader.rs` | 4,245 | parameter serialization, shader key model, device cache, pass validation, WGPU layout/pipeline construction, and test observations |
| `src/style.rs` | 3,279 | authored image/background values, placement/repeat resolution, filters, clips/masks, box decorations, and background normalization |
| `src/renderer.rs` | 3,237 | public renderer/options, route dispatch, publication, capability/error translation, and large fixture-specific test support |
| `src/resource.rs` | 2,639 | Gaussian kernel modeling and resource-manager identity, accounting, leasing, retention, and test observations |
| `src/reference.rs` | 2,307 | color/filter, blur/shadow, mask/composite, and image-conversion CPU reference oracles |
| `src/gpu_transaction.rs` | 2,028 | transaction state, graph and Vello submission commits, readback submission, post-submit control, and test instrumentation |
| `src/readback.rs` | 1,651 | row layout, staging lifecycle, native completion/polling, future ownership, decode, and test state machines |

The numbers explain why the files were inspected; they are not acceptance
thresholds. M05 is based on the independent owners visible in the item groups,
not on the physical counts.

At the published C05 base
`14b0ab5f8d7fbb2d93e2a958587e1075657f0f7b`, the tracked non-`plans/`
`*.rs`/`*.wgsl` inventory contains 2,139 token matches on 2,013 matching lines
across 22 files. The exact PCRE2 content predicate is:

```text
(?<![A-Za-z0-9_])(?:[PISCT][0-9]{2}[A-Za-z0-9_]*|[pisct][0-9]{2}_[A-Za-z0-9_]*)(?![A-Za-z0-9_])
```

Token count means every non-overlapping match, line count means every matching
source line, and file count means every matching selected file. The lowercase
underscore excludes Rust primitives such as `i32`. Apply that predicate to every
tracked non-`plans/` pathname as well as content, and additionally reject the
case-insensitive filename-segment predicate
`(?:^|[/_.-])[pisct][0-9]{2}(?=$|[/_.-])|sequence[0-9]+` so a lowercase path such
as `c08-graph.rs` cannot escape. Both filename predicates are empty at the base.
These are finite planning facts, not a required count or permanent
source-inspection test.

## M04 Decomposition Principles

### M04.1 Preserve The Front Doors

The existing crate root continues to declare `mod backend;`, `mod frame;`,
`mod pass;`, and the other current module names. Converting a file into a module
directory does not change the crate-root path. Existing root `pub use` entries,
public documentation paths, constructors, trait implementations, and error
types remain source-compatible.

Each module's `mod.rs` is a small internal front door. It owns:

- child declarations;
- explicit reexports needed by current sibling modules;
- the smallest coordinating type or operation that genuinely spans its
  children.

It does not copy child implementation, glob-reexport every child, or become a
second monolith.

### M04.2 Move By Owner

Move a type with its intrinsic constructors, validation, conversions, and
private helpers. Move an operation with the state transition or algorithm it
owns. Do not split an `impl` across children merely to balance line counts when
one type still owns the behavior.

When two child modules need the same semantic type, place that type in the
lowest common owning module and import it explicitly. Do not duplicate the type
or create a generic `common`, `util`, `misc`, `types`, or `helpers` grab bag.

### M04.3 Visibility

Preserve the narrowest effective visibility. A move may change private to
`pub(super)` when a named sibling child is a legitimate caller. `pub(crate)` is
allowed only when the item already had crate-wide callers or the original
module front door intentionally exposes it to those callers. No public
visibility is added.

### M04.4 Mechanical Evidence

Before each module move, run the focused tests for every owned behavior being
moved. After the move, the same tests and affected feature checks pass without
oracle changes. Import changes, file moves, and necessary visibility changes
are expected; algorithm rewrites and expectation changes are not.

If source inspection reveals a correctness or modeling defect, report it as a
separate initiative candidate. Do not repair it inside this mechanical range
unless the unchanged code cannot compile after a faithful move and the issue is
solely an import or visibility consequence.

### M04.5 Mixed Production And Test Owners

An intermediate task may move `#[cfg(test)]` fields and collectors with the
production type whose invariant they currently observe when separating them in
the same task would require global state, a leak, indirection, or changed test
semantics. The immediately following test-support task extracts that support to
the named `test_support.rs` owner. The final hierarchy still prohibits a
production child from importing test support. No intermediate bridge may detach
an observation from the value or operation that produced it.

### M04.6 Semantic Planning-Name Retirement

Planning identifiers are chronology such as `P02`, `I01`, `S34`, `C08`, `T03`,
or `sequence2` when they denote completed planning structure. Replace each with
the shortest rendering-domain term that identifies its actual owner or behavior;
do not perform a blind prefix substitution. The established semantic families
are:

| Historical family | Semantic vocabulary |
| --- | --- |
| `C03` | prepared Vello scene, checked encoding, or private raster behavior |
| `C06` | graph planning, bounds, filter planning, or graph validation |
| `C07` | prepared-graph or generation-bound handoff |
| `C08` | base/custom-spine graph, graph submission, or core pass shader |
| `C09` | composition, mask/blend, or layer-composite behavior |
| `C10` | ordered color-filter behavior |
| `C11` | spatial filter, blur, or drop-shadow behavior |
| `C12` | bounded-backdrop behavior |
| `S21` | reference color transform or reference color matrix |
| `S34` | GPU pixel, edge, or centroid tolerance |

Module context supplies information already present in the path, so renamed
items do not repeat module names or add weasel words. Closed graph-classification
variants use `Base`, `Composition`, `ColorFilter`, `SpatialFilter`, and
`Backdrop`; preparable graph types use the corresponding semantic prefix.
Test names continue to state a triggering condition and observable outcome.
Git history and `plans/` remain the only provenance owners.

## M05 Required Private Hierarchy

The names below express responsibilities. Workers may use an equally short Rust
identifier only when an existing production term makes it objectively clearer;
they may not merge or omit a responsibility without revising this specification.

### M05.1 Frame Planning

`src/frame/mod.rs` retains `FrameContext`, `FramePlan`, `DirectVelloPlan`, and
the coordination that selects and returns a plan.

| Child | Owner |
| --- | --- |
| `bounds.rs` | semantic command contributions, logical bounds, finite spatial primitives, and coordinate mapping |
| `filter.rs` | resolved filter intent, kernel support, edge policy, and filter spatial planning |
| `graph.rs` | semantic graph identities, resources, passes, builder, planner, and graph construction |
| `validate.rs` | semantic graph structure, imports, lifetimes, anchors, and lowering precondition validation |
| `lower.rs` | graph-lowering views and conversion from semantic graph facts to lowering facts |
| `test_support.rs` | frame-owned test observations and invalid-graph probes, compiled only for tests |

### M05.2 Runtime Pass Pipeline

`src/pass/mod.rs` retains the narrow orchestration from a lowered graph to a
prepared/encoded graph and explicitly reexports the current crate-visible pass
contract.

| Child | Owner |
| --- | --- |
| `model.rs` | runtime resource, pass, read, result, filter, composite, spatial, and cache-key facts |
| `close.rs` | executable-subset closure and graph accounting validation |
| `lower.rs` | conversion from frame graph-lowering facts into the runtime model |
| `prepare.rs` | allocation/kernel/pass analysis, preflight, realization, and prepared bindings |
| `encode.rs` | prepared graph encoding, capture handoff, pass scheduling, and encoding receipts |
| `parameters.rs` | pass-specific uniform and parameter construction owned above shader byte serialization |
| `test_support.rs` | pass-owned fixtures, fault injection, malformed plans, and behavioral observations, compiled only for tests |

### M05.3 Backend And Renderer Runtime

`src/backend/mod.rs` retains `Backend` and the coordinating calls that require
the complete backend state.

| Backend child | Owner |
| --- | --- |
| `device.rs` | device state/lifecycle, immutable capabilities, terminal signals, and callbacks |
| `execute.rs` | direct and prepared-graph GPU execution and commit results |
| `offscreen.rs` | offscreen target creation, texture leases, and local-scene rendering |
| `present.rs` | surface acquisition, presented targets, configuration, and presentation |
| `texture.rs` | shared backend texture descriptors and safe texture construction |
| `test_support.rs` | backend-owned failure controls and observations, compiled only for tests |

`src/renderer/mod.rs` retains public `Renderer` methods as the crate-facing
orchestrator.

| Renderer child | Owner |
| --- | --- |
| `dispatch.rs` | route selection, pre-execution gating, preparation, and typed unsupported-graph translation |
| `publication.rs` | successful-frame publication and failure-atomic statistics state |
| `options.rs` | `Options`, `EffectQualityPolicy`, `ResourceCacheBudget`, and `Antialiasing` |
| `test_support.rs` | renderer-owned fixture preparation and publication fault controls, compiled only for tests |

`src/gpu_transaction/mod.rs` retains `GpuOperationTransaction` and its shared
operation-stage contract.

| Transaction child | Owner |
| --- | --- |
| `graph.rs` | graph submission payload, resource readiness/accounting, host effect, and output commit |
| `vello.rs` | internal Vello payload, submission, and resource commit proof |
| `readback.rs` | pending and committed readback submission |
| `test_support.rs` | post-submit controls and transaction observations, compiled only for tests |

### M05.4 Shader And Resource Infrastructure

`src/shader/mod.rs` retains the crate-visible shader/cache front door.

| Shader child | Owner |
| --- | --- |
| `parameters.rs` | color, shadow, spatial, blur-edge, composite, and other GPU byte serialization |
| `key.rs` | shader, format, binding, sampler, mask, layout, module, and pipeline key models |
| `validate.rs` | pass-key compatibility and semantic validation |
| `pipeline.rs` | WGSL selection plus WGPU bind-group layout, pipeline layout, and render-pipeline creation |
| `cache.rs` | committed and provisional device pass cache state and realization |
| `test_support.rs` | shader/cache observations and vector facts, compiled only for tests |

`src/resource/mod.rs` retains the resource-management front door.

| Resource child | Owner |
| --- | --- |
| `gaussian.rs` | Gaussian kernel keys, limits, normalized samples, packing, and plans |
| `manager.rs` | manager/frame/resource identities, cache keys, entries, allocation preflight, and manager state |
| `lease.rs` | acquisition scopes, leases, cleanup, retention, and accounting outcomes |
| `test_support.rs` | resource-manager observations and test tokens, compiled only for tests |

### M05.5 Authored Style And Reference Domains

`src/style/mod.rs` explicitly reexports the same current public style types.

| Style child | Owner |
| --- | --- |
| `image.rs` | style resources, image sources/layers, size/position, placement, repeat, attachment, and background areas |
| `filter.rs` | filter lists/operations/amounts, filtered image paint, backdrop input, and drop-shadow payload |
| `clip.rs` | clip input, normalized clip, geometry, transforms, and validation |
| `mask.rs` | mask input, sources, modes, layers, stacks, and composition |
| `decoration.rs` | border, outline, radii, fragments, normalized decoration commands, and style normalization |
| `background.rs` | background layers/stacks/blends and normalized background commands |

`src/reference/mod.rs` remains test-only and owns shared reference pixel types.

| Reference child | Owner |
| --- | --- |
| `color.rs` | straight/premultiplied conversion, color transforms, and compiled color-filter reference behavior |
| `filter.rs` | Gaussian blur, drop shadow, materialized filter pipeline, and extent planning |
| `mask.rs` | mask sampling, extend policy, opacity, source-over, and blend reference behavior |

`src/readback/mod.rs` retains the readback operation front door.

| Readback child | Owner |
| --- | --- |
| `layout.rs` | validated row layout, mapped-range validation, and row decoding |
| `lifecycle.rs` | staging phases, disposition, cleanup action, and owner state |
| `native.rs` | completion callback, native polling, helper ownership, and readback future |
| `test_support.rs` | readback observations and standalone state-machine probes, compiled only for tests |

### M05.6 Focused Test Hierarchy

After P02-I01, replace `src/tests.rs` with `src/tests/mod.rs` and these focused
children:

| Child | Test ownership |
| --- | --- |
| `model.rs` | geometry, scene, paint, image, layer, text, and validation contracts |
| `style.rs` | image placement/repeat, backgrounds, decorations, filters, clips, and masks |
| `frame.rs` | route selection, semantic graph construction/lowering, and pass closure |
| `gpu.rs` | shader, graph execution, pixels, precision, resources, caches, and transactions |
| `surface.rs` | headless/presented surface lifecycle, publication, cancellation, and readback |
| `platform.rs` | feature/target behavior and executable host-facing examples that remain tests |
| `vello.rs` | internal Vello characterization and parity behavior |
| `support.rs` | only fixtures/oracles used by at least two sibling test domains |

A helper used by one child remains in that child. Cross-child support is
`pub(super)` and named for the domain operation, never for convenience. The
hierarchy contains no source parser, plan-closure test, or committed test
inventory.

## M06 Dependency And Cycle Discipline

Within each new module directory, child dependencies follow the phase direction
already present in source:

```text
authored/model -> semantic plan -> lowering -> runtime model
               -> preparation -> encoding -> backend submission/publication
```

Validation may depend on the model it validates. Within one module directory,
models do not depend on preparation, encoding, backend, test-support, or higher
phases. Production children never depend on `test_support`. Reference modules
remain test-only.

This mechanical initiative does not eliminate the following complete set of
existing bidirectional edges among the module directories in M05:

- `backend::execute` consumes transactions and commit payloads owned by
  `gpu_transaction::{mod,graph,vello}`, while those transaction owners consume
  device signals owned by `backend::device`;
- `backend::{mod,execute}` consumes lowered, prepared, and encoded graph
  contracts owned by `pass::{model,lower,prepare,encode}`, while pass validation
  and preparation consume capabilities owned by `backend::device`;
- `backend::{mod,execute,offscreen}` consumes manager, cleanup, format, and
  lease contracts owned by `resource::{mod,manager,lease}`, while resource
  realization consumes capabilities owned by `backend::device`;
- `backend::{mod,device,execute}` consumes public policy and cache-budget types
  owned by `renderer::options`, while `renderer::{mod,dispatch,publication}`
  consumes the backend front door and its execution results;
- `pass::{model,parameters}` supplies runtime facts serialized or keyed by
  `shader::{parameters,key,validate}`, while `pass::prepare` consumes the
  realization/cache contracts owned by `shader::{pipeline,cache}`;
- `frame::mod` consumes public `renderer::options::Antialiasing`, while
  `renderer::dispatch` consumes the frame-planning front door;
- `pass::{model,prepare}` consumes public policy types owned by
  `renderer::options`, while `renderer::{dispatch,test_support}` consumes the
  pass lowering, preparation, and diagnostic front doors;
- `resource::{manager,lease}` consumes public
  `renderer::options::ResourceCacheBudget`, while renderer/backend orchestration
  consumes the resource front door.

Those types remain with the exact owners above because relocating the public
option types or transaction, pass, shader, device, and resource contracts would
be an architectural change rather than a file move. Their imports must name the
owning front door or child explicitly; no compatibility shim is added. These
eight baseline edges are allowed and are not evidence of an incomplete move.
The initiative may not introduce another crate-level mutual edge.

If two proposed children within one module directory would require a cycle, the
worker must move the shared owner downward or retain the inseparable group in
`mod.rs`; it must not introduce indirection, a trait, dynamic dispatch,
duplicated data, or a generic helper module to disguise the cycle. If a cycle
cannot be resolved without changing semantics, the cycle plan must revise this
specification before implementation rather than improvise an architecture.

## M07 Test And Verification Contract

Each implementation range is behavior-preserving. Its worker records:

- the exact pre-move focused test commands for every responsibility moved;
- post-move results for the same tests;
- `cargo check` and warning-denied Clippy for every affected feature
  combination;
- a public-surface comparison showing that crate-root exports are unchanged;
- the complete file-move and visibility-change inventory in the commit diff;
- confirmation that no algorithm or oracle changed.

The planning-name task additionally records a transient exact inventory over
tracked non-plan filenames and Rust/WGSL code. Its acceptance predicate is empty
after semantic renaming. This source inspection is workflow evidence only: it is
not committed as a parser, test, inventory, generated index, lint, or CI rule.

The final command matrix is derived from current `AGENTS.md`, `Cargo.toml`, and
`README.md` and uses already installed artifacts offline:

```sh
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
```

The dependency inspections from the published P02-I01 matrix, absence of a
tracked `Cargo.lock`, and the canonical tracked/untracked owned-Rust unsafe scan
also remain required final evidence. I02 does not install or update missing
tooling. The installed wasm target, Rust 1.97 toolchain, and native presentation
host are verified prerequisites before a cycle starts. If one later becomes
unavailable, the affected cycle is blocked and is neither reviewed as complete
nor published; a native host run is not replaced by a headless or compile-only
check.

## M08 Product Impacts

- Public API: internal-only and source-compatible; root exports unchanged.
- Behavior: unchanged; diagnostic prose and backend labels change only to remove
  planning chronology while preserving error codes, conditions, and context.
- Dependencies/features/targets: unchanged.
- Generated artifacts: none in this leaf; root-owned API artifacts untouched.
- Documentation/examples: public docs and examples unchanged except paths in
  private source links when required by a move.
- MSRV: preserve Rust 1.97 compatibility.
- Migration: none for callers.
- Root integration: pointer evaluation only after the final published leaf
  candidate; no adapter or artifact delta expected.
- Safety: no Surgeist-owned executable `unsafe` or unsafe-enabling allowance.

## M09 Initiative Acceptance

P02-I02 is accepted when every M05 responsibility has its named private owner,
the public front door and observable behavior remain unchanged, focused and full
verification pass, no size enforcement or relocation workaround was introduced,
no tracked filename or Rust/WGSL code outside `plans/` retains a planning
identifier, and the reviewed leaf candidate is published to the authority remote
`main` with a complete root handoff.
