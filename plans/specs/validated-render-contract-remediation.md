# P03-I01 Validated Render Contract Remediation

## R01 Outcome

`surgeist-render` closes four validated product-quality defects in its current
rendering surface:

1. image pixels are never identified for backend upload, retained-resource
   reuse, or renderer upload telemetry by a collision-prone 64-bit fingerprint
   alone;
2. every publicly constructed rectangle has finite derived maximum coordinates;
3. every item exported through the current crate front door has rustdoc that
   explains its applicable phase, units, invariants, defaults, failures, and
   behavior; and
4. the normalized-command statistics helper and its import are compiled only as
   test support, and its dead-code allowance is removed.

The initiative is complete when distinct same-sized image contents remain
distinct even when a test deliberately gives them the same 64-bit fingerprint,
overflowing rectangle maxima fail at construction and canonical validation,
strict missing-docs rustdoc succeeds for the current public surface, and no
production target retains `RenderCommands::stats` or its dead-code allowance.

## R02 Ownership And Boundary

The owning repository is `surgeist-render`. It owns the image values passed to
its backend, its retained GPU-resource identity, intrinsic geometry validation,
its current source API documentation, and its private render-command model.

In scope:

- `src/image.rs` image content identity and Peniko image-data construction;
- the private frame, resource, renderer-publication, statistics, and focused
  test paths that consume image identity;
- `src/geometry.rs` public rectangle construction and
  `src/validation.rs` canonical rectangle validation;
- rustdoc on every currently exported item in the 17 source files identified
  by R03, including public enum variants and public fields;
- test-only isolation of `RenderCommands::stats` and its import, plus removal of
  its lint allowance from `src/command.rs`;
- focused, condition-named behavior tests and one-time strict rustdoc evidence.

Out of scope:

- the deferred authored-style ownership correction between `surgeist-style`,
  root, and this crate;
- a hierarchical public front door, module visibility redesign, public rename,
  or removal of current exports;
- root facade, adapters, generated API artifacts, gitlink promotion, or sibling
  repository changes;
- a dependency, feature, target, shader, rendering algorithm, cache policy,
  resource budget, or lifecycle change unrelated to exact image identity;
- a permanent `missing_docs` lint, documentation parser, inventory test,
  generator, script, CI rule, source-text test, or plan-closure test;
- production-visible API added only for tests;
- any `unsafe` code or lint allowance that permits it.

## R03 Current Evidence

The initiative baseline is published `main` at
`b02fa0c372472c88a511f45cb74b1ec0b356d181`.

### R03.1 Image identity

- `src/image.rs` defines public `ImageId(u64)` and derives the ID produced by
  `Image::from_rgba` with `DefaultHasher` over dimensions and pixel bytes.
- The same hash is passed to `peniko::Blob::from_raw_parts` as backend resource
  identity. The installed `linebender_resource_handle` source states that raw
  identifiers not uniquely associated with data can produce inconsistencies.
- Installed `vello_encoding` indexes `ImageCache` residency solely by
  `image.data.id()` and returns an occupied resident without comparing bytes.
- `ResolvedMaskUploadKey` contains `ImageId`, physical size, quality, and extend
  mode, but no exact pixel witness.
- `ResourceManagerState::acquire_with_payload` reuses an idle entry when its key
  and byte length match, bypassing the texture-creation and upload closure.
- Renderer upload telemetry retains a `HashSet<ImageId>`, so a collision can
  also misclassify first observation as a cache hit.
- Existing tests prove ordinary byte changes usually change `ImageId` and prove
  that mask keys separate the currently modeled metadata. They do not force an
  ID collision or compare exact content after one.

### R03.2 Rectangle maxima

- `Rect::try_new` validates `x`, `y`, `width`, and `height` independently and
  returns a value without validating `x + width` or `y + height`.
- `Rect::max` performs those additions and uses the crate-private unchecked
  `Point::new`, so finite components such as `f64::MAX` and `f64::MAX` expose an
  infinite maximum.
- `validate_rect` repeats only origin and size validation.
- The existing frame-bounds test rejects the equivalent overflowing internal
  fixture later, proving the desired finite-bounds rule exists downstream but
  not at the public or canonical geometry boundary.

### R03.3 Public documentation

At the baseline, this command fails with 933 missing-docs diagnostics:

```sh
CARGO_NET_OFFLINE=true RUSTDOCFLAGS="-D missing_docs" \
  cargo doc -p surgeist-render --no-deps
```

The diagnostics are distributed as follows:

| Source area | Missing items |
| --- | ---: |
| `src/style/image.rs` | 154 |
| `src/error.rs` | 115 |
| `src/capability.rs` | 108 |
| `src/style/decoration.rs` | 101 |
| `src/geometry.rs`, `src/shape.rs` | 116 |
| `src/style/filter.rs` | 51 |
| `src/layer.rs` | 47 |
| `src/style/background.rs` | 45 |
| `src/text.rs` | 42 |
| `src/paint.rs` | 36 |
| `src/style/clip.rs`, `src/style/mask.rs` | 70 |
| `src/image.rs` | 23 |
| `src/scene.rs` | 17 |
| `src/style/mod.rs`, `src/renderer/options.rs` | 8 |

The inventory is planning evidence only. No count, file list, or source parser
becomes an executable repository test or permanent gate.

### R03.4 Dead command path

- `src/command.rs` retains private `RenderCommands::stats` in every library
  build under a method-scoped `#[allow(dead_code)]`.
- The helper is not dead test support: 21 current call sites in
  `src/tests/model.rs`, `src/tests/style.rs`, and `src/tests/vello.rs` inspect
  normalized-command behavior through it when `cfg(test)` is active.
- The production renderer instead starts from route and timing state, clones
  previously observed image identities, calls `collect_render_stats`, and
  publishes the resulting state.
- Routing production through the helper would lose those semantics, while
  deleting it would break focused normalization tests. The defect is that test
  support and its allowance remain compiled in non-test library targets.

## R04 Resolved Design Decisions

### R04.1 Collision-safe content identity

Render-owned content equality is represented by a private cloned identity value
that retains:

- the existing public `ImageId` as a compact deterministic fingerprint;
- the exact logical dimensions needed to distinguish image contents; and
- shared ownership of the exact RGBA8 bytes.

Equality compares the full dimensions and bytes. Hashing may use the compact
fingerprint as an accelerator because Rust hash collections still resolve hash
collisions with exact equality; no correctness decision may use the hash alone.
The private value is an immutable normalized/runtime content identity and is not
exported.

`ImageId` remains the current copyable public newtype and retains `new` and
`get`. Its rustdoc must state that it is an opaque compact fingerprint or
caller-supplied resource handle, not a proof of byte equality and not a valid
sole backend cache key. This preserves the existing public representation while
removing every render-owned collision-sensitive use.

Rejected alternatives:

- A wider probabilistic digest remains collision-prone and does not establish a
  correctness boundary.
- A new Surgeist process-global counter would duplicate identity allocation
  already owned by Peniko and would introduce hidden global coordination.
- Making `ImageId` retain all pixels would unnecessarily make the public copyable
  handle non-copyable and couple authored resource references to backend bytes.

### R04.2 Peniko and Vello identity

`Image::from_rgba` constructs its Peniko blob through `peniko::Blob::new` rather
than `Blob::from_raw_parts`. Peniko therefore owns generation of the unique blob
identity expected by Vello's ID-indexed atlas cache. Surgeist does not derive,
override, or synchronize that backend identity with `ImageId`.

Cloning one `Image` preserves its Peniko blob identity. Independently
constructing two images, including byte-identical images, gives each backend
blob a unique identity. Render-owned exact-content reuse remains independent and
may still recognize identical bytes safely.

### R04.3 Exact retained-mask and telemetry identity

`ResolvedMaskUploadKey` incorporates the private exact content identity in
addition to physical size, quality, and extend mode. It becomes cloneable rather
than copyable where required. `ResourceCacheKey`, allocation preflight, graph
imports, leases, and test observation adapt only as needed to preserve ownership
and exact equality.

An idle resolved-mask texture is reusable only when the complete key, byte
length, and therefore exact content match. Distinct bytes with a deliberately
duplicated `ImageId` allocate or select distinct resources and execute the upload
closure. Identical content with identical sampling facts remains eligible for
reuse.

Renderer first-observation telemetry stores the same collision-safe private
content identity rather than `ImageId`. `Stats::cache_hits`,
`Stats::cache_misses`, and `Stats::uploaded_bytes` retain their documented
meaning for exact image content. Test-only observation adapts privately and
does not expose a new production API.

### R04.4 Rectangle invariant

`Rect::try_new` computes each derived maximum after validating its components
and rejects the rectangle with `ErrorCode::InvalidInput` when either maximum is
non-finite. Zero sizes and finite maxima remain valid. The error diagnostic names
the overflowing derived coordinate rather than blaming an individually valid
component.

`validate_rect` enforces the same rule for crate-private fixtures and internal
construction paths before those values enter planning or rendering. Public
`TryFrom<kurbo::Rect>` inherits the constructor rule. Crate-private `Rect::new`
remains available for focused invalid fixtures; it does not become public and
does not weaken canonical validation.

### R04.5 Documentation contract

Every item exported by the current `src/lib.rs` front door receives rustdoc.
Documentation depth follows behavior:

- types name their rendering/model phase, semantic role, units, and intrinsic
  invariants;
- enum variants and public fields state the distinct observable choice or value;
- fallible constructors state accepted inputs and the applicable error outcome;
- builders, conversions, defaults, and behavior-bearing methods state mutation,
  loss, context, and return semantics;
- identity types state equality, lifetime, and collision expectations;
- capability and error models state when the result is reported and how callers
  distinguish it;
- obvious accessors use concise descriptions and do not repeat long type-level
  rationale.

The docs describe current behavior without declaring permanent ownership of the
deferred authored-style domain. They do not introduce plan identifiers, planning
provenance, future promises, or root-owned API artifacts into source.

Strict rustdoc with `-D missing_docs` is completion evidence for this initiative,
not a committed crate lint. Normal rustdoc and Clippy must also reject broken
links, warnings, or misleading code examples.

### R04.6 Test-only statistics isolation

Compile `RenderCommands::stats` only under `cfg(test)`, remove its
`#[allow(dead_code)]`, and compile its `collect_render_stats` import only under
the same test configuration. The existing 21 test call sites remain unchanged
and continue to observe normalized-command statistics. Non-test library targets
contain neither the helper nor its import.

Preserve the active stateful statistics publication path in
`src/renderer/dispatch.rs` and renderer test support. Do not create a second
helper, route production through the test helper, suppress the lint elsewhere,
or weaken the existing normalization tests.

## R05 Observable Behavior Matrix

| Condition | Required result |
| --- | --- |
| Different RGBA8 bytes, same dimensions, ordinary hashes | Exact content identities differ; backend blob identities differ |
| Different RGBA8 bytes and a deliberately forced equal `ImageId` in test scope | Mask keys differ; retained texture is not reused; upload telemetry records both contents |
| Equal RGBA8 bytes, dimensions, quality, and extend | Exact equality permits existing safe reuse semantics |
| Same content with different quality or extend | Mask keys differ as before |
| Direct Vello images with a forced public fingerprint collision | Peniko blob IDs differ and Vello cannot alias atlas residency by that fingerprint |
| `Rect::try_new(f64::MAX, 0.0, f64::MAX, 1.0)` | `InvalidInput` at construction |
| `Rect::try_new(0.0, f64::MAX, 1.0, f64::MAX)` | `InvalidInput` at construction |
| Finite origin with zero size | Accepted when both derived maxima remain finite |
| Internally constructed rectangle with an infinite derived maximum | `validate_rect` returns `InvalidInput` before planning |
| Current public item without prior rustdoc | It has accurate rustdoc; strict missing-docs rustdoc emits no diagnostic |
| Active renderer statistics publication | Behavior remains stateful; normalized-command stats remain test-only; non-test builds contain no parallel helper or dead-code allowance |

## R06 Code And Test Outline

### R06.1 Image paths

- Add the private exact content-identity model beside `Image` in `src/image.rs`.
- Construct it once from the already validated size and shared bytes.
- Use `peniko::Blob::new` for `ImageData::data`.
- Carry exact content through resolved-mask keys, graph imports, resource
  preflight/acquisition, and renderer upload observations without duplicating
  pixel buffers.
- Add a `#[cfg(test)]`-only constructor or identity seam at the narrowest private
  owner that can deliberately duplicate a public fingerprint for two different
  pixel buffers.
- Add condition-named tests for key inequality, retained upload non-reuse,
  telemetry accounting, and distinct Peniko blob IDs under the forced collision.
- Preserve ordinary identical-content and sampling-key tests.

### R06.2 Geometry paths

- Share one private derived-max validation operation between `Rect::try_new` and
  `validate_rect`, or keep equivalent small checks only if doing so does not
  duplicate error semantics.
- Add focused public-constructor tests for x-maximum and y-maximum overflow and
  finite boundary acceptance.
- Add a canonical-validation test using a crate-private invalid fixture.
- Preserve downstream bounds rejection tests as defense in depth.

### R06.3 Documentation paths

- Document the current exports in `capability.rs`, `error.rs`, `geometry.rs`,
  `image.rs`, `layer.rs`, `paint.rs`, `renderer/options.rs`, `scene.rs`,
  `shape.rs`, `style/*.rs`, and `text.rs`.
- Keep existing accurate docs and improve them only where the new identity or
  rectangle semantics require it.
- Do not change visibility, reexports, defaults, behavior, or diagnostics merely
  to reduce the missing-docs inventory.
- Verify zero missing-docs diagnostics from rustdoc without committing an
  inventory or lint attribute.

### R06.4 Test-only command statistics

- Gate the existing normalized-command stats method and its direct import with
  `cfg(test)` and remove only the dead-code allowance.
- Retain all current normalization call sites and focused statistics tests,
  including those exercising the real renderer publication path; no test should
  assert source absence.

## R07 Compatibility And Impacts

- Public API: behavior-correcting for `Rect::try_new`; documentation-additive
  elsewhere. Previously accepted rectangles with non-finite derived maxima are
  intentionally rejected. Public names, fields, variants, and reexports remain
  unchanged.
- Image identity: `ImageId` representation and ordinary deterministic value
  remain unchanged, but its documented contract no longer promises uniqueness.
  Backend and cache identity changes are private and correctness-preserving.
- Dependencies and features: unchanged. No software acquisition is required.
- MSRV: root-owned Rust 1.97 compatibility remains unchanged.
- Generated artifacts: none in this leaf. Root-owned API artifacts and pointer
  promotion are excluded.
- Docs/examples: source rustdoc changes only; README and example behavior remain
  unchanged unless a broken source link requires the smallest factual repair.
- Performance: exact bytes are shared by `Arc`; equality may compare bytes only
  after compact identity and metadata match. No duplicate production pixel copy
  is introduced beyond existing Peniko ownership.
- Safety: all owned code remains free of `unsafe`.

## R08 Initiative Acceptance

The initiative is accepted only when all of the following are true:

1. a deterministic regression forces equal public fingerprints for different
   same-sized pixels and proves distinct render-owned keys, backend blob IDs,
   retained uploads, and upload telemetry;
2. identical exact content and equal sampling facts retain safe reuse behavior;
3. `Rect::try_new` rejects overflow in both derived axes and `validate_rect`
   rejects the same internally constructed state;
4. finite and zero-area rectangle behavior remains covered;
5. strict missing-docs rustdoc succeeds with no missing item, broken link,
   warning, or failed doctest for the current public surface;
6. non-test builds omit `RenderCommands::stats` and its import, the dead-code
   allowance is absent, and normalized-command plus active renderer statistics
   tests remain green;
7. the configured native and feature verification matrix passes using already
   installed offline tooling;
8. repository-wide owned Rust contains no executable `unsafe` and no allowance
   that permits it;
9. no source parser, inventory gate, plan identifier outside `plans/`, permanent
   missing-docs lint, dependency, feature, generated artifact, or root/sibling
   change is introduced; and
10. every implementation task and the integrated cycle receive clean independent
    review before the candidate is landed and published on leaf `main`.
