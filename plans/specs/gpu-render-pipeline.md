# GPU Render Pipeline Specification

Authority: this specification is the initiative-wide desired-state contract for
the `surgeist-render` GPU render-pipeline migration. The canonical
`$surgeist-agent` workflow governs planning, implementation, review, landing,
and publication. `AGENTS.md`, `Cargo.toml`, and current source remain
authoritative for mutable repository facts. The committed migration overlay is
source evidence; where it and this specification differ, this reviewed
specification owns the selected desired state.

## S01 Outcome

`surgeist-render` renders every currently supported scene through WGPU, either
through one direct pass in its crate-owned Vello-derived raster engine or
through a render-owned GPU pass graph around bounded raster captures.
Production filter, mask, backdrop, and compositing execution never reads pixels
to the CPU, runs a CPU pixel algorithm, or uploads the result again.

The completed renderer has two observable routes:

- `DirectVello` renders a scene with no custom effect requirement in one
  internal Vello raster pass, preserving the high-quality vector fast path.
- `GpuGraph` executes a closed render-owned graph of Vello raster captures, image
  passes, and composite passes, then presents the GPU-resident result.

The CPU implementation remains only a private `cfg(test)` semantic oracle.
There is no production CPU backend or automatic CPU fallback.

This initiative preserves all `Preserve` rows and implements all `Migrate`,
`Correct`, and `Enable` rows in
`plans/2026-07-09-render-pipeline-migration-overlay.md`. It does not turn a
`HoldDiagnostic` or `HoldRoot` row into support merely because the new backend
contains a useful internal primitive.

## S02 Scope And Ownership

`surgeist-render` owns:

- authored renderer-facing inputs already exposed by this crate;
- normalized render commands and effect planning;
- resolved frame and device-pixel mappings;
- the private Vello-derived scene, recording, raster scheduling, shader binding,
  resource-lease, and WGPU lowering modules copied from the pinned Vello 0.9
  main crate and adapted under S06A;
- render-owned GPU textures, uploads, samplers, buffers, shaders, pipelines,
  pass scheduling, submission, and readback of explicit headless outputs;
- semantic and runtime rendering capability reports;
- rendering diagnostics and last-frame statistics;
- private CPU reference algorithms and fixture-based quality tests.

Root `surgeist` owns:

- style/layout/text interpretation and Surgeist-to-Surgeist lowering;
- symbolic color and resource resolution;
- text shaping and the ink bounds supplied with shaped runs;
- root backdrop policy and any future backdrop-root selection;
- image orientation and color-profile conversion;
- facade adaptation, generated API artifacts, integration tests, and the
  `surgeist-render` gitlink.

Sibling crates, including `surgeist-shape`, are read-only context. This crate
does not depend on them and does not duplicate their semantic models.

## S03 Non-Goals

This initiative does not:

- add compatibility aliases, deprecated constructors, or legacy execution
  shims;
- add a CPU production renderer, CPU effect fallback, or CPU-side pixel
  materialization in `Renderer::render`;
- enable filtered image paint, broad layer filters, authored shape masks,
  luminance masks, multi-layer masks, mask-composite modes, backdrop isolation,
  root backdrops, transformed backdrops, repeated top-level backdrops, inset
  shadows, text shadows, non-solid shadow paint, repeating gradients, new blend
  modes, 3D transforms, or another overlay `HoldDiagnostic` row;
- resolve style, layout, font shaping, symbolic resources, color profiles, or
  host/window policy;
- vendor or modify `vello_encoding`, `vello_shaders`, or their WGSL raster
  algorithms; they remain pinned external dependencies and safe backend inputs;
- add a dependency outside the already-present, explicitly authorized S36
  internalization set, or add a feature, generator, script, CI workflow, or
  downloaded toolchain without exact user permission; the known web execution
  blocker in S37 does not itself grant that permission;
- add or retain Surgeist-owned `unsafe`.

The absence of a compatibility requirement permits intentional public API
replacement, but it does not permit deleting a semantic capability without the
replacement and its evidence.

## S04 Current Evidence

At initiative base `d59ad253300b68311f4e81a70e2b6ce73c922a4d`:

- `Renderer::render` normalizes a scene, materializes resolved backdrops,
  materializes resolved layer masks, encodes one Vello scene, and renders it.
- Each materialization creates an effect-local offscreen texture cache, renders
  to `Rgba8Unorm`, calls `read_texture_rgba`, executes filters or masks on CPU
  reference buffers, creates a new `Image`, and re-uploads through Vello.
- `RenderBackdropCapture` clones prior sibling commands rather than reading the
  completed parent image at the backdrop's paint position.
- `Backend` owns a Vello `RenderContext` and renderer slots but no persistent
  render-owned per-device effect resource manager.
- `shader.rs` has descriptors and a clear-fill probe but no source-bound image,
  blur, composite, or present pipeline.
- `Options::use_cpu` selects Vello's diagnostic CPU stages even though Vello
  still requires GPU resources.
- `reference.rs`, public filter execution-plan types, and
  `ResolvedAlphaMaskExecution` compile into production.
- `ImageBuffer` exposes public fields, so invalid byte lengths are constructible.
- `FontData::from_bytes` accepts arbitrary bytes and collection indices, while
  pinned Vello scene lowering unwraps `skrifa::FontRef::from_index`; malformed
  public font input can therefore panic instead of returning a typed error.
- `ResolvedLayerAlphaMask` stores a generic readback buffer with no logical
  placement or reusable image identity.
- `Capabilities::VELLO_0_9` mixes semantic support claims with implementation
  names and CPU/materialization facts.
- `src/backend.rs` contains one macOS HAL `unsafe` block used only to set a
  resize presentation hint.
- Vello 0.9 accepts `Rgba8Unorm` storage targets and assumes registered textures
  are unpremultiplied. Its texture registration copies an `Rgba8Unorm` texture
  into the Vello image atlas each frame, so it is not the custom graph's
  high-precision re-entry path.
- WGPU 29 exposes per-adapter format features and safe render, sample, copy,
  and queue-ordering APIs needed by the graph. `Rgba16Float` support must still
  be probed on the selected adapter rather than inferred from the backend name.

Baseline evidence is clean for default tests, host `render-web` checking, and
`render-window` checking. The installed Rust toolchain does not contain the
`wasm32-unknown-unknown` target; host `render-web` checking is not a substitute
for the final wasm compile gate.

At architecture-revision head
`9aa1d97c30a75d0bd552d6b07b62d8d87a7bd39b`, unpublished C02 work has already
established renderer/device identity, async non-readback entry points,
per-device terminal signals, immutable runtime capability snapshots, and the
first scoped GPU transaction model. The external `vello = 0.9.0` crate still
owns renderer construction, recording execution, an internal `queue.submit`,
resource pooling, and surface convenience types. The current Task 4 review is
not clean because production presented setup remains hidden behind that
external integration boundary.

The pinned `vello 0.9.0` package identified by Cargo checksum
`261359dbef879f8110ef7e1c442246c838d33d3d91cb05e0ea9288d432760c9f`
contains 5,497 physical Rust lines. Its main crate contains one unsafe trusted
shader-module creation site, one internal queue-submission site, a blocking
device-poll helper, optional CPU shader dispatch, and close-to-WGPU buffer,
binding, pipeline, atlas, and command-recording ownership. The companion
`vello_encoding 0.9.0` and `vello_shaders 0.9.0` packages contain the packed
scene model and raster shaders and remain external under the revised contract.

## S05 Normative Sources

Filter behavior follows [Filter Effects Module Level 1](https://www.w3.org/TR/filter-effects-1/):
functions execute in authored order, operate in sRGB, and feed each result to
the next function; `blur()` length is Gaussian standard deviation; and
`drop-shadow()` is SourceAlpha blur, offset, solid-color source-in, then merge
below the original source.

Backdrop behavior follows the bounded subset of
[Filter Effects Module Level 2](https://drafts.csswg.org/filter-effects-2/): the
backdrop is the completed image behind the element at its paint position, the
filter list applies before the element foreground, and the filtered backdrop is
clipped before outer effects. The editor draft does not settle a universal
Backdrop Root contract; root and nested backdrop-root behavior therefore remain
typed diagnostics in this initiative.

Blend and isolation behavior follows
[Compositing and Blending Level 1](https://www.w3.org/TR/compositing-1/).
Clip/mask/filter/opacity order follows
[CSS Masking Level 1](https://www.w3.org/TR/css-masking-1/) together with the
Filter Effects order: render the source, apply filters, apply clip and mask,
apply opacity, then blend/composite into the parent.

The pinned Vello 0.9 main source is import/provenance evidence. The pinned
`vello_encoding`, `vello_shaders`, and WGPU source in the local Cargo registry
is backend API evidence. This specification does not promise a backend behavior
that those pinned external versions cannot expose safely.

## S06 Resolved Design Decisions

| Decision | Selected contract | Rejected material alternative |
| --- | --- | --- |
| Production execution | WGPU only, directly or through the internal Vello-derived raster engine | CPU fallback violates the stated product boundary and hides missing GPU capability. |
| Raster-engine ownership | Copy the pinned Vello 0.9 main-crate implementation into private `surgeist-render` modules, then adapt it while retaining external `vello_encoding` and `vello_shaders` | Keeping external `vello` preserves opaque submission/resource ownership; vendoring all three crates unnecessarily takes ownership of the packed format and raster shader algorithms. |
| GPU submission ownership | The render transaction owns command encoders, queue submission, scope resolution, and resource-lease commit/abort | Allowing the internal raster engine to submit independently prevents frame-atomic cleanup and coherent graph scheduling. |
| Imported execution modes | Preserve only checked WGPU raster execution | Retaining Vello CPU dispatch, blocking helpers, debug downloads, hot reload, or trusted unsafe shader creation contradicts the production and safety boundaries. |
| Mixed-scene ownership | Once a scene needs an effect graph, render owns the working image until presentation | Re-entering Vello through its RGBA8 atlas quantizes and copies every custom result. |
| Working pixels | Premultiplied numeric-sRGB `Rgba16Float`; explicit reduced `Rgba8Unorm` route | Linear-light working pixels would contradict CSS filter-function color space; unpremultiplied working pixels make filtering and composition unstable. |
| Quality fallback | High precision is required by default; reduced precision is opt-in | Silent reduction makes quality device-dependent and unobservable. |
| Mask input | Immutable `Image` plus explicit logical bounds | A physical `ImageBuffer` couples public semantics to one surface scale and has no cache identity. |
| Text bounds | Caller-supplied run-local ink bounds with an explicit unspecified state | Parsing validated font data does not authorize render to invent ink bounds; guessing from advances is not ink geometry. |
| Blur implementation | Separable Gaussian fragment passes with CPU-planned weights only | CPU pixels are forbidden; an initial compute-only design adds complexity without a demonstrated need. |
| Backdrop source | Copy the completed parent GPU image at the paint position | Cloning and replaying prior commands duplicates work and mishandles parent/base/effect state. |
| Resource retention | Per-device manager with a public retained-byte budget | Unbounded retention is not operationally safe; a hidden fixed policy prevents applications from selecting memory policy. |
| Resize optimization | Safe WGPU lifecycle only | The Metal HAL hint requires repository-owned `unsafe` and is not semantic behavior. |
| Public algorithm plans | Private | Algorithm phases are backend-owned and should not become cross-crate contracts. |

The default retained-resource budget is 64 MiB. The budget constrains idle,
reusable render-owned textures, mask uploads, and kernel buffers; it never
rejects resources required by the active frame. Device limits and allocation
failure remain explicit runtime failures. Zero bytes is valid and disables
retention after each frame.

### S06A Internal Vello Engine Contract

The pinned Vello 0.9 main-crate source is internalized into the existing
`surgeist-render` package under private `src/vello_engine/` modules. It is not a
new public crate, workspace member, feature, backend choice, or compatibility
fork. Source files retain their copyright/SPDX headers. A tracked
`NOTICE-VELLO.md` records package name, version, Cargo checksum, source URL,
copied files, each copied file's pre-adaptation SHA-256, omitted files, material
adaptations, and the preserved Apache-2.0/MIT license texts. The package
checksum is the immutable S04 import checksum, not a future lockfile lookup.
The first committed import may perform required module-path and lint adaptation
and must replace the upstream unsafe site; no commit may contain an executable
unsafe construct even transiently.

The final internal engine keeps only behavior needed by the production raster
path:

- scene lowering compatible with `vello_encoding 0.9.0`;
- fallible font-data parsing with no copied upstream unwrap/expect path;
- the fixed coarse/fine Vello recording schedule and shader permutations;
- checked WGPU shader-module, bind-group, pipeline, buffer, image-atlas, and
  command encoding;
- per-device persistent raster caches and transaction-owned transient leases.

It removes the external `vello` dependency, Vello utility device/surface
ownership, CPU shader dispatch and materialized CPU buffers, `use_cpu`, debug
layers/downloads, hot reload, profiler integration, deprecated async rendering,
blocking executors, production mapping/polling, direct queue submission, and
trusted unchecked shader creation. `vello_shaders` is built with default
features disabled and only `wgsl` enabled, so its external CPU shader module is
not part of the production dependency graph.

The private phase boundary is:

```text
TextRun
  -> preflight_selected_glyphs
ValidatedGlyphRun
  -> encode_into
VelloScene
  -> prepare_raster(RasterParameters)
PreparedVelloPass { recording, target_intent, resource_intents }
  -> encode_into(FrameCommandEncoder, VelloEngineState)
EncodedVelloPass { transient_lease, atlas_commit }
  -> submit_with(GpuOperationTransaction)
SubmittedVelloPass
  -> commit resources after clean scope/signal resolution
```

`ValidatedGlyphRun`, `VelloScene`, and `PreparedVelloPass` contain no WGPU
resource. Every glyph run crosses S10A preflight before the internal scene can
append an external `vello_encoding::GlyphRun`; external resolver omission is
therefore not an error channel. The per-device
`VelloEngineState` owns checked shader pipelines, resolver state, and persistent
image-atlas metadata while the one S25 `ResourceManager` owns its live and idle
allocations. `EncodedVelloPass` cannot submit;
the engine encodes into an encoder already owned by the frame transaction and
returns the private `VelloResourceLease` with an encoded-pass token. The
transaction is the only owner that calls `queue.submit`, and it retains every
raster/effect lease until that submission is accepted and all S13A
scopes/signals resolve.

Successful submission commits reusable raster resources in queue order. Error
or cancellation before the clean linearization point drops or quarantines
uncertain transient buffers/images instead of returning them idle. A canceled
atlas update is marked dirty or replaced before reuse; it can never make an
incomplete frame public. No Vello-derived resource identity crosses the private
engine boundary or appears in public capabilities/statistics.

The internal engine remains a vector-raster pass, not the CSS effect graph.
Changing Vello draw tags, packed encoding, fine-raster output semantics, or WGSL
algorithms requires a later reviewed specification revision that explicitly
expands ownership to the relevant external crate. Filters, masks, blends,
backdrops, canonicalization, and presentation remain the crate-owned WGPU passes
defined by S15-S24.

### S06B Planning Supersession And Retained Work

This specification revision atomically changes the status of
`plans/cycles/gpu-render-pipeline-c02-async-gpu-transactions-and-device-terminality.md`
to `superseded`. It is inert now; no worker may receive another task from it.
Its task text remains only historical evidence. A replacement plan must use a
distinct path and the reviewed current specification and sequence revisions.

Published C01 commit `5361e3460278dffb877b9d485a2d12977977c3ef` remains
complete and is the adoption base for the entire unpublished range. Planning
commits and implementation commits after that SHA are not separate legal cycle
acceptance points. Every provisional hunk and forward correction after that
base remains subject to one successor cycle's exact-range task and holistic
review before it can become accepted implementation.

The complete adoption map is:

| Existing artifact/range | Desired-state disposition |
| --- | --- |
| Published C01 through `5361e34` | Retain as complete; do not reopen or rewrite; use as the sole published adoption base |
| Old-plan commits `c13609e`, `2b6d4ad`, and `6621c62` | Historical planning only; contribute no acceptance evidence |
| `64bc5cb` and `7e2a31e` identity changes | Provisional foundation candidate: audit renderer/device/surface identity and validation order |
| `7b638c6` async API changes | Provisional foundation candidate: audit the async non-readback front door |
| `d80c6ec` and `ed3355f` terminal-device/capability changes | Provisional foundation candidate: audit terminal signals, immutable reports, and creation-race closure |
| Generic scope/generation/error/lease work in `03f0db7`, `43187ed`, and `9aa1d97` | Provisionally retain only backend-neutral transaction foundations after complete diff audit |
| External-Vello presented-setup orchestration and its no-op test seam introduced in the same transaction commits | Not accepted; remove by forward correction without rewriting history, then reimplement setup only under S06A |
| Old C02 tasks 5-7 | No clean accepted implementation exists; desired headless publication, presentation, and lifecycle behavior remains unimplemented |
| Old sequence C03 readback and every later item | Preserve the semantic work; dependency-order and renumber it in the revised sequence |

The old requirement to retain external `vello::Renderer::render_to_texture`,
its renderer/device slots, or its surface/setup ownership is invalidated. The
current external renderer may remain only as unchanged temporary production
behavior while the provisional foundation is audited. Internalization and
production raster cutover under S06A must precede new headless-publication,
surface-setup, submission, or presentation implementation. Readback and graph
work follow those lifecycle prerequisites. The revised sequence owns exact
cycle allocation; each just-in-time plan owns executable adoption tasks and
ranges. Canonical workflow transitions are not redefined here.

## S07 Phase Model

The rendering flow has seven distinct phases:

1. **Authored:** `Scene`, `Layer`, `FilterList`, `BackdropFilterInput`, `Image`,
   `ResolvedLayerAlphaMask`, and text runs preserve caller meaning and typed
   unresolved boundaries.
2. **Normalized:** `RenderCommands` validate semantic support, canonicalize
   geometry and paint, preserve authored ordering, and contain no backend
   resource.
3. **Resolved frame:** `FramePlan` uses surface size, scale, base color, effect
   bounds, text bounds, transforms, and layer order. It is either the least
   powerful direct variant or a closed semantic GPU graph.
4. **Runtime capability:** `DeviceCapabilities` records facts from the selected
   safe WGPU adapter/device and output surface. It is private; a stable public
   projection is `RuntimeCapabilities`.
5. **Raster algorithm:** `ValidatedGlyphRun` proves selected-glyph preflight;
   `PreparedVelloPass` is a private recording/resource intent produced from a
   Vello-encodable span. Neither contains a live WGPU object or can submit.
6. **Backend algorithm:** `ExecutableFramePlan` chooses the working format,
   concrete texture descriptors, pipelines, and topological pass sequence. It
   cannot contain an unsupported operation or an unresolved resource.
7. **Runtime resource:** per-device resources have allocation identity,
   generation, lease state, last-used frame, and deterministic retention state.

No type is reused across these phases merely because fields happen to match.
Contextual conversion is named and fallible:

```text
Scene
  -> normalize_against(Capabilities::CURRENT)
RenderCommands
  -> plan_for(FrameContext)
FramePlan
  -> resolve_against(DeviceCapabilities, EffectQualityPolicy)
ExecutableFramePlan
  -> execute_with(DeviceState, SurfaceTarget)
GPU submission
```

`DirectVello` is selected only when all normalized commands can be encoded as
one Vello scene against the output target. The presence of an alpha mask,
bounded backdrop, or another migrated image/composite requirement selects
`GpuGraph`; the planner never emits a graph merely because graph plumbing is
available.

## S08 Public Options

`Options` remains `Clone + Copy + Debug + PartialEq` but has private fields.
The exact public shape is:

```rust
pub struct Options {
    antialiasing: Antialiasing,
    debug: bool,
    effect_quality_policy: EffectQualityPolicy,
    resource_cache_budget: ResourceCacheBudget,
}

pub enum EffectQualityPolicy {
    RequireHighPrecision,
    AllowReducedPrecision,
}

pub struct ResourceCacheBudget(u64);
```

`Options::default()` selects `Antialiasing::Area`, `debug == false`,
`RequireHighPrecision`, and `ResourceCacheBudget::DEFAULT` (64 MiB).
The exact const API is `Options::new()`, `antialiasing()`,
`with_antialiasing(Antialiasing)`, `debug()`, `with_debug(bool)`,
`effect_quality_policy()`,
`with_effect_quality_policy(EffectQualityPolicy)`,
`resource_cache_budget()`, and
`with_resource_cache_budget(ResourceCacheBudget)`. `new()` equals `default()`.
`ResourceCacheBudget::new(u64)`, `bytes()`, `DISABLED`, and `DEFAULT` are total
because every `u64` value has defined retention semantics; `DISABLED` is zero
and `DEFAULT` is `64 * 1024 * 1024` bytes.

`Options`, `EffectQualityPolicy`, and `ResourceCacheBudget` are `Clone + Copy +
Debug + Eq + PartialEq`; the policy implements `Default` as
`RequireHighPrecision` and the budget implements `Default` as `DEFAULT`.

`Options::use_cpu` is removed. The internal raster engine contains no CPU
execution mode or corresponding field. No replacement CPU selector exists.

`EffectQualityPolicy::AllowReducedPrecision` means prefer high precision and
select reduced precision only when high precision is unavailable. It does not
force reduced precision and does not permit resolution downscaling.

## S09 Public Image And Mask Models

`ImageBuffer` is a readback/fixture value, not a production effect input. It has
private fields and these constructors/accessors:

```rust
impl ImageBuffer {
    pub fn try_new(size: PhysicalSize, rgba: Vec<u8>) -> Result<Self>;
    pub const fn size(&self) -> PhysicalSize;
    pub fn rgba(&self) -> &[u8];
    pub fn into_rgba(self) -> Vec<u8>;
}
```

The constructor uses checked `width * height * 4` arithmetic and requires the
exact byte length. A zero-area size is valid only with zero bytes. Invalid or
overflowing lengths return typed `InvalidValue`; they never panic or saturate.
`Renderer::read_headless` returns this validated type.
`ImageBuffer` remains `Clone + Debug + Eq + PartialEq`.

`ResolvedAlphaMaskExecution` is removed from the public API and production
build. CPU mask execution exists only in the private test oracle.

`ResolvedLayerAlphaMask` becomes an authored/resolved render input with private
fields:

```rust
pub struct ResolvedLayerAlphaMask {
    image: Image,
    bounds: Rect,
}

impl ResolvedLayerAlphaMask {
    pub fn try_new(image: Image, bounds: Rect) -> Result<Self>;
    pub const fn image(&self) -> &Image;
    pub const fn bounds(&self) -> Rect;
}
```

`bounds` is a finite, positive-area rectangle in the owning layer's local
coordinate space. The image's alpha channel is mapped across that rectangle;
sampling outside it yields zero alpha. RGB channels are ignored. Sampling
preserves the complete public image-quality contract:

| `ImageQuality` | Mask sampler |
| --- | --- |
| `Low` | nearest texel-center sample |
| `Medium` | bilinear 2x2 sample |
| `High` | bicubic 4x4 Mitchell-Netravali sample with `B = 1/3`, `C = 1/3`, matching pinned Vello 0.9 |

The output sample point is first tested against the semantic `bounds`; a point
outside is transparent regardless of `Image::extend`. For a point inside,
texel taps outside the image domain use the image's exact `Extend` mode:
`Pad` clamps to the nearest edge texel, `Repeat` wraps with Euclidean period,
and `Reflect` mirrors over a two-domain period. Bilinear and bicubic edge taps
apply that rule per tap rather than clamping the final coordinate. High-quality
filtering operates on alpha taps and clamps the weighted alpha to `[0, 1]`
before composition. Boundary-pixel tests cover all three quality modes, all
three extend modes, and sample points immediately inside and outside each side
of the semantic rectangle. There is no High-to-Medium mask downgrade.

`Layer::with_resolved_alpha_mask(ResolvedLayerAlphaMask)` installs this already
valid value and returns `Self` infallibly. `ResolvedLayerAlphaMask` is `Clone +
Debug + PartialEq`; its current `Eq` implementation is intentionally removed
because `Image` and logical `Rect` contain floating-point semantic values. The
old `ImageBuffer` constructor and mode-bearing constructor are removed. Alpha
is the only supported resolved mask mode; luminance and authored shape masks
retain their existing typed diagnostics.

At composition, the resolved mask multiplies source premultiplied RGB and alpha
after the layer's filters and outer clip and before layer opacity and blend.
Nested resolved masks execute from the innermost child outward.

## S10 Public Text Bounds

`TextRunBounds` is an authored rendering fact with a private representation:

```rust
pub struct TextRunBounds {
    value: TextRunBoundsValue,
}

pub enum TextRunBoundsKind {
    Unspecified,
    Empty,
    Ink,
}

enum TextRunBoundsValue {
    Unspecified,
    Empty,
    Ink(Rect),
}

impl TextRunBounds {
    pub const fn unspecified() -> Self;
    pub const fn empty() -> Self;
    pub fn try_ink(rect: Rect) -> Result<Self>;
    pub const fn kind(self) -> TextRunBoundsKind;
    pub const fn ink_rect(self) -> Option<Rect>;
}
```

Construction uses `TextRunBounds::unspecified()`, `empty()`, and
`try_ink(Rect)`. The field and payload enum are private, there is no public
struct literal, and
`TextRunBoundsKind::Ink` cannot itself create a `TextRunBounds`. `try_ink`
requires a finite positive-area rectangle in run-local coordinates before
`TextRun::transform`. `kind()` returns the payload-free public kind and
`ink_rect()` returns `Some(Rect)` only for a validated ink value.
The exact changed constructor appends the authored bounds after the existing
glyph slice:

```rust
impl<'a> TextRun<'a> {
    pub fn try_new(
        font: FontRef<'a>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: &'a [TextGlyph],
        bounds: TextRunBounds,
    ) -> Result<Self>;

    pub const fn bounds(&self) -> TextRunBounds;
}
```

`TextShadowRun` preserves the wrapped run's bounds. `TextRunBounds` is `Clone +
Copy + Debug + PartialEq`; the kind also implements `Eq`.

An unspecified bound is valid for a direct Vello run because direct rendering
does not need CPU-side glyph geometry. If graph planning must bound that run,
it returns `UnresolvedResourceKind::TextRunInkBounds`; it does not estimate from
glyph advances. `Empty` contributes no pixels and is valid for an empty or
non-inking shaped run.

The Ahem fixture remains test-only. It proves stable glyph ink extents and
direct-versus-captured raster alignment; production does not inspect it to
derive font metrics or invent text bounds.

### S10A Validated Font Data

Font bytes and a collection index are authored renderer input, but their
readability is an invariant of `FontData`, not a precondition left to the
raster backend. The infallible constructor is replaced without a compatibility
shim:

```rust
impl FontData {
    pub fn try_from_bytes(bytes: Vec<u8>, index: u32) -> Result<Self>;
}
```

`try_from_bytes` calls `skrifa::FontRef::from_index(bytes.as_slice(), index)`
before constructing the private `peniko::FontData`. Success stores immutable
owned bytes/index exactly once. Failure returns `ErrorCode::InvalidInput` with
this exact `InvalidValue` model:

| Field | Value |
| --- | --- |
| `field` | `"font_data"` |
| `value` | `"len={byte_len}, index={index}"` |
| `invariant` | `"must contain a readable OpenType font at the requested collection index"` |

The diagnostic never formats raw font bytes. Empty/malformed bytes and a valid
font with an out-of-range collection index take the same typed boundary and do
not enter normalization, internal Vello scene encoding, or WGPU work.

OpenType table parsing is lazy, so a readable container may still contain a
malformed selected outline, color, palette, bitmap, or embedded-PNG table.
Every text run therefore passes through private
`preflight_selected_glyphs(&TextRun) -> Result<ValidatedGlyphRun>` before copied
Vello scene code appends a glyph run or calls external `vello_encoding`. The
token borrows the exact immutable font bytes, index, glyph slice, normalized
coordinates, size, hinting, embolden, transform, fill/stroke style, and selected
representation consumed by encoding; it cannot be constructed by callers or
reused for different inputs.

For each selected glyph, preflight uses `skrifa` to execute the same relevant
read as lowering:

1. choose outline, COLR/CPAL, or bitmap representation using the same order as
   the internal scene;
2. for an outline, obtain the glyph and draw it into a validation sink with the
   exact unhinted/hinted size, coordinates, embolden, and style facts that the
   external resolver will receive;
3. for COLR, traverse the selected paint graph and validate every referenced
   palette entry and outline;
4. for a bitmap, validate the selected strike/glyph dimensions and checked byte
   length, and fully decode/validate selected PNG or packed-mask data;
5. reject a missing selected glyph rather than substituting an empty external
   encoding.

Preflight has three exact failure classes:

| Failure | Result |
| --- | --- |
| Malformed font container or selected outline/COLR/CPAL/bitmap/PNG data | The same S10A `font_data` `InvalidValue` using stored byte length/index |
| Missing selected glyph ID | `InvalidValue { field: "text_glyph.id", value: glyph_id, invariant: "must identify a drawable glyph in the selected FontData" }` |
| Valid selected glyph image encoding that the internal engine cannot represent | Owning S13 internal-raster preparation `RenderFailed` with private safe context |

Copied Vello glyph lowering removes both upstream
`FontRef::from_index(...).unwrap()` calls and every other panic-prone
font-derived conversion or parse unwrap. It consumes only
`ValidatedGlyphRun`; no fallback font or silent glyph omission is permitted.
The retained external `vello_encoding` crate is not modified, but its
`GlyphCache::session -> None` and `get_or_insert -> None` omission branches are
unreachable because the same immutable selected glyph and draw settings were
successfully preflighted before the external encoding was built.

Public construction excludes the gross byte/index failure, while the fallible
private edge closes errors from lazy tables and protects future internal model
changes. This validation does not shape text, calculate ink bounds, select a
font, or alter root ownership.

## S11 Semantic Capabilities

`Capabilities` continues to describe renderer semantics independently of the
selected adapter. Its implementation-named constant becomes
`Capabilities::CURRENT`; `VELLO_0_9` is removed rather than retained as a shim.

All unaffected capability groups and accessors preserve their current names and
values. The affected groups have these exact public const queries:

| Group | Query | Final value |
| --- | --- | ---: |
| Filter | `supports_layer_filters()` | false |
| Filter | `supports_ordered_filter_lists()` | true |
| Filter | `supports_gpu_color_filter_execution()` | true |
| Filter | `supports_gpu_blur_filter_execution()` | true |
| Filter | `supports_gpu_drop_shadow_filter_execution()` | true |
| Filter | `supports_filter_region_planning()` | true |
| Mask/clip | `supports_shape_clips()` | true |
| Mask/clip | `supports_clip_reference_execution()` | false |
| Mask/clip | `supports_layer_masks()` | false |
| Mask/clip | `supports_resolved_alpha_mask_execution()` | true |
| Mask/clip | `supports_luminance_mask_mode()` | false |
| Mask/clip | `supports_multi_layer_mask_composition()` | false |
| Mask/clip | `supports_mask_composite_modes()` | false |
| Offscreen | `supports_direct_vello_opacity_isolation()` | true |
| Offscreen | `supports_direct_vello_blend_isolation()` | true |
| Offscreen | `supports_offscreen_layer_rendering()` | false |
| Offscreen | `supports_persistent_effect_resources()` | true |
| Offscreen | `supports_bounded_vello_capture()` | true |
| Offscreen | `supports_image_pass_execution()` | true |
| Offscreen | `supports_composite_pass_execution()` | true |
| Offscreen | `supports_nested_opacity_composition()` | true |
| Offscreen | `supports_mask_execution()` | false |
| Offscreen | `supports_layer_filter_execution()` | false |
| Offscreen | `supports_broad_backdrop_execution()` | false |
| Offscreen | `supports_bounded_backdrop_capture()` | true |
| Offscreen | `supports_bounded_backdrop_filter_execution()` | true |
| Offscreen | `supports_backdrop_isolation_composition()` | false |

The exact affected accessor disposition is:

| Current accessor | Desired disposition |
| --- | --- |
| `supports_color_filter_classification` | Rename to `supports_ordered_filter_lists` |
| `supports_color_filter_pipeline_execution` | Rename to `supports_gpu_color_filter_execution` |
| `supports_materialized_image_filter_classification` | Remove; classification is private algorithm state |
| `supports_materialized_blur_filter_execution` | Rename to `supports_gpu_blur_filter_execution` |
| `supports_materialized_drop_shadow_filter_execution` | Rename to `supports_gpu_drop_shadow_filter_execution` |
| `supports_filter_region_outset_planning` | Rename to `supports_filter_region_planning` |
| `supports_cpu_reference_blur_fallback` | Remove without replacement |
| `supports_materialized_alpha_mask_execution` | Rename to `supports_resolved_alpha_mask_execution` |
| `supports_texture_cache_upload_lifecycle` | Rename to `supports_persistent_effect_resources` |
| `supports_rect_fullscreen_shader_passes` | Replace with `supports_image_pass_execution` and `supports_composite_pass_execution` |
| `supports_cpu_reference_buffers` | Remove without replacement |
| `supports_nested_opacity_planning` | Rename to `supports_nested_opacity_composition` |
| `supports_filter_execution` | Rename to `supports_layer_filter_execution`; remains false |
| `supports_backdrop_execution` | Rename to `supports_broad_backdrop_execution`; remains false |
| `supports_materialized_backdrop_filter_execution` | Rename to `supports_bounded_backdrop_filter_execution` |

The corresponding public `PrimitiveOperation` disposition is exact:

| Current variant | Desired variant/disposition |
| --- | --- |
| `ColorFilterClassification` | `OrderedFilterList` |
| `ColorFilterPipelineExecution` | `GpuColorFilterExecution` |
| `MaterializedImageFilterClassification` | Remove; no public algorithm diagnostic |
| `MaterializedBlurFilterExecution` | `GpuBlurFilterExecution` |
| `MaterializedDropShadowFilterExecution` | `GpuDropShadowFilterExecution` |
| `FilterRegionOutsetPlanning` | `FilterRegionPlanning` |
| `CpuReferenceBlurFallback` | Remove without replacement |
| `ColorFilterBlur`, `ColorFilterDropShadow` | Remove with the public color-only classifier |
| `MaterializedAlphaMaskExecution` | `ResolvedAlphaMaskExecution` |
| `TextureCacheUploadLifecycle` | `PersistentEffectResources` |
| `RectFullscreenShaderPass` | `ImagePassExecution` |
| `CpuReferenceBuffer` | Remove without replacement |
| `NestedOpacityPlanning` | `NestedOpacityComposition` |
| `FilterExecution` | `LayerFilterExecution` |
| `BackdropExecution` | `BroadBackdropExecution` |
| `MaterializedBackdropFilterExecution` | `BoundedBackdropFilterExecution` |
| none | Add `BoundedVelloCapture` and `CompositePassExecution` |

`BoundedBackdropCapture`, `MaskExecution`, `BackdropIsolationComposition`, and
all unaffected diagnostic variants retain their names. Removed variants have no
alias. `Capabilities::ensure_supported` maps every remaining/new public
operation to exactly one query above and returns typed `UnsupportedPrimitive`
for a false query. No semantic capability advertises CPU buffers/fallback,
materialized byte execution, an implementation version, or support inferred
only from internal plumbing.

New/renamed operation labels are respectively `ordered filter list`, `GPU color
filter execution`, `GPU blur filter execution`, `GPU drop-shadow filter
execution`, `filter-region planning`, `resolved alpha-mask execution`,
`persistent effect resources`, `image-pass execution`, `nested opacity
composition`, `layer-filter execution`, `broad backdrop execution`, `bounded
backdrop-filter execution`, `bounded Vello capture`, and `composite-pass
execution`. The affected capability group structs keep private fields and
`Clone + Copy + Debug + Eq + PartialEq`; `Capabilities` and its family accessors
retain the same traits and access pattern.

Capability truth changes only after the corresponding public behavior executes
on a real GPU route. Internal shaders or pass descriptors alone do not change a
claim.

## S12 Runtime Capabilities

Runtime capabilities describe a selected safe WGPU device/surface, not semantic
support and not Cargo features. The public model is:

```rust
pub enum RuntimeCapabilities {
    Unavailable(RuntimeCapabilityUnavailableReason),
    Available(AvailableRuntimeCapabilities),
}

pub struct AvailableRuntimeCapabilities {
    surface_format: Format,
    effect_precisions: EffectPrecisionCapabilities,
    max_effect_texture_dimension_2d: u32,
}

pub struct EffectPrecisionCapabilities {
    high_precision: bool,
    reduced_precision: bool,
}
```

All fields are private and all four public report types are `Clone + Copy +
Debug + Eq + PartialEq`. `RuntimeCapabilities::available()` returns
`Option<AvailableRuntimeCapabilities>` and `unavailable_reason()` returns
`Option<RuntimeCapabilityUnavailableReason>`. The available report exposes
`surface_format()`, `effect_precisions()`, and
`max_effect_texture_dimension_2d()`. The precision report exposes
`supports_high_precision()` and `supports_reduced_precision()`. Both precision
flags being false is valid: direct Vello may still render while the effect
graph is unavailable.

`Renderer::runtime_capabilities(&mut self, surface: &Surface) ->
RuntimeCapabilities` observes any pending device-loss signal and projects the
device selected by that surface. It performs no allocation or submission. A
contract-only surface returns `Unavailable(AdapterUnavailable)`.
An available report is based on adapter format features and device limits, not
backend names, platform `cfg`, or error-message matching.

High-precision effects require `Rgba16Float` as a render attachment and sampled
texture with the filtering used by the pass set. Reduced-precision effects
require the same operations on `Rgba8Unorm`. Storage binding is not a
requirement for custom effect textures because the initial implementation uses
fragment render passes. Internal raster capture capability remains separately
implied by the selected WGPU device and its required `Rgba8Unorm` storage
target.

`max_effect_texture_dimension_2d` is the selected device's 2D texture limit.
Dynamic allocation exhaustion cannot be predicted by a capability report and
is returned as a backend failure without selecting CPU execution.

## S13 Errors And Quality Evidence

The `Error` wrapper adds an optional typed
`RuntimeCapabilityUnavailable` diagnostic and accessor. The exact model is:

```rust
pub struct RuntimeCapabilityUnavailable {
    operation: RuntimeOperation,
    reason: RuntimeCapabilityUnavailableReason,
}

#[non_exhaustive]
pub enum RuntimeOperation {
    AdapterSelection,
    SurfaceRendering,
    SurfaceReadback,
    SurfaceResume,
    EffectRendering,
    EffectTextureAllocation,
    EffectPresentation,
}

#[non_exhaustive]
pub enum RuntimeCapabilityUnavailableReason {
    AdapterUnavailable,
    SurfaceUnavailable { state: RenderSurfaceAvailability },
    DeviceLost { reason: DeviceLossReason },
    DeviceFaulted { kind: GpuFaultKind },
    SurfaceIdentityMismatch { kind: SurfaceIdentityMismatchKind },
    EffectFormatUnavailable { policy: EffectQualityPolicy },
    TextureDimensionExceeded {
        requested: PhysicalSize,
        maximum: u32,
    },
    SurfaceFormatUnavailable { format: Format },
}

pub enum RenderSurfaceAvailability {
    Suspended,
    NonRenderable,
    Uninitialized,
    Occluded,
    Lost,
}

pub enum SurfaceIdentityMismatchKind {
    ForeignRenderer,
    StaleDeviceGeneration,
}

pub enum DeviceLossReason {
    Unknown,
    Destroyed,
}

pub enum GpuFaultKind {
    Validation,
    OutOfMemory,
    Internal,
}
```

These public enums and the diagnostic are `Clone + Copy + Debug + Eq +
PartialEq`; the open-ended runtime operation and reason enums are
`#[non_exhaustive]`. The diagnostic has private fields, no public constructor,
and public `operation()`/`reason()` accessors. Backend construction uses a
crate-private validated constructor. It accepts only the following pairings;
every other pair returns `InvalidValue` in model tests and cannot become an
`Error` payload:

| Operation | Permitted reason forms |
| --- | --- |
| `AdapterSelection` | `AdapterUnavailable`, `DeviceLost`, `DeviceFaulted` |
| `SurfaceRendering` | `AdapterUnavailable`, `SurfaceUnavailable { Suspended, NonRenderable, Occluded, Lost }`, `SurfaceIdentityMismatch`, `DeviceLost`, `DeviceFaulted` |
| `SurfaceReadback` | `AdapterUnavailable`, `SurfaceUnavailable { Suspended, NonRenderable, Uninitialized, Lost }`, `SurfaceIdentityMismatch`, `DeviceLost`, `DeviceFaulted` |
| `SurfaceResume` | `SurfaceIdentityMismatch`, `DeviceLost`, `DeviceFaulted` |
| `EffectRendering` | `EffectFormatUnavailable`, `DeviceLost`, `DeviceFaulted` |
| `EffectTextureAllocation` | `TextureDimensionExceeded`, `DeviceLost`, `DeviceFaulted` |
| `EffectPresentation` | `SurfaceFormatUnavailable`, `DeviceLost`, `DeviceFaulted` |

The private constructor is the sole field-writing path. Public callers can
observe and copy a diagnostic returned by render, but cannot fabricate an
operation/reason combination. `RuntimeCapabilities::Unavailable` remains a
reason-only observation and may report adapter, identity, surface, device, or
effect-format availability without inventing an operation.

`Error` fields become private. Public `code()`, `message()`, typed diagnostic
accessors, `Display`, and `std::error::Error::source` provide observation.
`Error::new` and `with_source` become crate-private backend constructors.
Public semantic constructors remain for `InvalidValue`,
`UnsupportedPrimitive`, `UnresolvedResource`, `DegradedQuality`, and
`RuntimeCapabilityUnavailable`.

The exact public `Error` surface is:

```rust
impl Error {
    pub fn invalid_value(
        field: impl Into<String>,
        value: impl fmt::Display,
        invariant: &'static str,
    ) -> Self;
    pub fn from_invalid_value(value: InvalidValue) -> Self;
    pub fn unsupported_render_primitive(value: UnsupportedPrimitive) -> Self;
    pub fn unresolved_resource(value: UnresolvedResource) -> Self;
    pub fn degraded_quality(value: DegradedQuality) -> Self;
    pub fn runtime_capability_unavailable(
        value: RuntimeCapabilityUnavailable,
    ) -> Self;

    pub const fn code(&self) -> ErrorCode;
    pub fn message(&self) -> &str;
    pub const fn invalid_value_diagnostic(&self) -> Option<&InvalidValue>;
    pub const fn unsupported_primitive(&self) -> Option<UnsupportedPrimitive>;
    pub const fn unresolved_resource_diagnostic(
        &self,
    ) -> Option<&UnresolvedResource>;
    pub const fn degraded_quality_diagnostic(
        &self,
    ) -> Option<&DegradedQuality>;
    pub const fn runtime_capability_unavailable_diagnostic(
        &self,
    ) -> Option<&RuntimeCapabilityUnavailable>;
}
```

`Error` implements `Debug`, `Display`, and `std::error::Error` and is not
`Clone`, `Eq`, or `PartialEq` because it may own an underlying error source.
`ErrorCode` remains `Clone + Copy + Debug + Eq + PartialEq`. Existing typed
diagnostic trait contracts remain unchanged; the new runtime diagnostic and its
component enums use the traits stated above.

Backend source ownership is target-correct and private:

```rust
#[cfg(not(target_arch = "wasm32"))]
type BackendErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

#[cfg(target_arch = "wasm32")]
type BackendErrorSource = Box<dyn std::error::Error + 'static>;
```

The crate-private `with_source` constructor has the corresponding target-bound
signature. `Error::source()` returns `Option<&(dyn Error + 'static)>` on every
target. `Error` is required to be `Send + Sync` on native; neither auto-trait is
promised on wasm because WGPU 29's wasm error source omits those bounds. No
source is stringified merely to force cross-target auto-traits.

The wrapper invariant is bidirectional: `InvalidInput`,
`UnsupportedPrimitive`, `UnresolvedResource`, `DegradedQuality`, and
`RuntimeCapabilityUnavailable` error codes each have exactly their matching
typed payload and no other typed payload; those payloads never appear with a
different code. Internal generic backend construction cannot accept one of
those five codes. `ErrorCode::AdapterUnavailable`, `SurfaceLost`, and
`SurfaceUnavailable` are removed in favor of the runtime diagnostic;
`ErrorCode::UnsupportedPrimitive`,
`ErrorCode::RuntimeCapabilityUnavailable`, and `ErrorCode::ReadbackFailed` are
added. Other current backend codes remain, including timeout, outdated,
out-of-memory, configure, render, and present failures. `UnsupportedBackend`
remains for calling an operation on the wrong backend kind, not for a semantic
primitive.

Runtime conditions map exactly as follows:

| Condition | Operation | Reason/code |
| --- | --- | --- |
| `Renderer::new` finds no adapter | none | construction succeeds; surfaces are contract-only and runtime report is unavailable |
| Presented surface creation with no adapter | `AdapterSelection` | `AdapterUnavailable` runtime diagnostic |
| Render requiring GPU on a contract-only surface | `SurfaceRendering` | `AdapterUnavailable` runtime diagnostic |
| Read requiring GPU on a contract-only surface | `SurfaceReadback` | `AdapterUnavailable` runtime diagnostic |
| Zero-size available headless render | `SurfaceRendering` | `SurfaceUnavailable { NonRenderable }` before adapter lookup |
| Zero-size available headless read | none | validated empty `ImageBuffer`, no GPU operation |
| Suspended surface render | `SurfaceRendering` | `SurfaceUnavailable { Suspended }` |
| Zero-size/non-renderable presented surface | `SurfaceRendering` | `SurfaceUnavailable { NonRenderable }` |
| Occluded surface that cannot acquire | `SurfaceRendering` | `SurfaceUnavailable { Occluded }` |
| WGPU surface reports lost | `SurfaceRendering` | `SurfaceUnavailable { Lost }`; surface lifecycle becomes lost |
| Foreign or stale surface passed to render/read/resume | owning surface operation | `SurfaceIdentityMismatch` with exact kind; no slot indexing or WGPU call |
| Device-loss callback observed before, during, or after an operation | owning operation | `DeviceLost` with mapped WGPU reason; device state becomes terminal lost |
| Uncaptured WGPU error escapes an owned scope | owning operation, then future device operations | owning backend error for active operation; device becomes terminal `DeviceFaulted` |
| Required high/reduced effect features absent | `EffectRendering` | `EffectFormatUnavailable { policy }` |
| Planned effect extent exceeds limit | `EffectTextureAllocation` | `TextureDimensionExceeded` |
| Output format cannot receive graph result | `EffectPresentation` | `SurfaceFormatUnavailable` |
| Dynamic WGPU out of memory | owning operation | `SurfaceOutOfMemory` with safe source; no fallback |
| Surface timeout/outdated | none | existing timeout/outdated code and existing retry/lifecycle policy |
| Generic pipeline/allocation/submission failure | none | existing create/render/present code with safe source |

The owning stage determines every backend code; captured/uncaptured class does
not create a second mapping:

| Failure source/stage | Exact result when no terminal loss supersedes it |
| --- | --- |
| Adapter request yields none | validated runtime diagnostic above |
| Device request/create | `DeviceCreateFailed` |
| Internal raster-engine creation | `RendererCreateFailed` |
| Surface object creation | `SurfaceCreateFailed` |
| Surface configuration or presented resize | `SurfaceConfigureFailed` |
| Surface acquire timeout/outdated/lost/out-of-memory | `SurfaceTimeout` / `SurfaceOutdated` / typed `SurfaceUnavailable { Lost }` / `SurfaceOutOfMemory` |
| Other surface acquire failure | `PresentFailed` |
| Internal raster draw/capture, effect allocation, pipeline creation, draw encoding, or draw submission validation/internal error | `RenderFailed` |
| Output conversion, surface write, or present validation/internal error | `PresentFailed` |
| Headless copy allocation/encoding/submission validation/internal error, `map_async` callback error, checked row-decoding failure, or `PollError::WrongSubmissionIndex` | `ReadbackFailed` |
| Captured or active-generation uncaptured out-of-memory at any stage | `SurfaceOutOfMemory` |
| Active-generation uncaptured validation/internal error | the owning stage's code above, followed by terminal `DeviceFaulted` state |
| No-active-generation uncaptured validation/out-of-memory/internal error | next device operation returns typed `DeviceFaulted` with the recorded class |
| Device-lost signal observed at any stage | owning operation returns typed `DeviceLost`; this terminal classification takes precedence over a generic stage error |

Safe WGPU, internal raster, map, and poll sources are retained under the
target-specific source representation. Tests inject private error
classifications to prove the table; they do not parse source display text.

An unavailable adapter, format, limit, surface, or device never becomes
`UnsupportedPrimitive`: semantic support and runtime availability are distinct.

Existing `InvalidValue`, `UnsupportedPrimitive`, and `UnresolvedResource`
semantics remain. `UnresolvedResourceKind` adds `TextRunInkBounds`. Backend
allocation/submission failures preserve a safe source error when available and
do not parse source prose for control flow.

`DegradedQualityKind::SoftwareFallback` and `FastBlurClamp` are removed because
neither behavior exists. The final variants are
`ReducedIntermediatePrecision` and the preserved
`UnsupportedPaintSpaceConversion`. A successfully selected reduced GPU route
is not an error: it is reported by `Stats::effect_precision ==
Some(EffectPrecision::Reduced)`. `RequireHighPrecision` with no high-precision
route returns the runtime capability error above, not a degraded-quality error.
The public degraded-quality value remains a typed report/error surface, and its
constructor/accessors preserve the existing private-field invariant.

Failure is frame-atomic at the public boundary. A failed render does not replace
the renderer's last successful `Stats` or the surface's last successful
parameters. A headless render never draws into its published texture. It draws
into a draft texture and swaps that texture into the published slot only at the
clean transaction linearization point; a failed or canceled submitted frame
drops/quarantines the draft and preserves the prior readable pixels. A resize
to a different physical extent intentionally invalidates the old publication
before the next render. Presented surfaces retain the explicit attempted-present
exception below because an external presentation cannot be rolled back. Every
acquired frame resource is returned or destroyed through safe cleanup even when
planning, allocation, encoding, submission, mapping, or presentation fails.

### S13A Async GPU Operation And Error Commit Protocol

Every public operation that creates, submits, presents, or maps a WGPU resource
is asynchronous on native and wasm:

```rust
impl Renderer {
    pub async fn create_surface(
        &mut self,
        attachment: Attachment,
        options: SurfaceOptions,
    ) -> Result<Surface>;

    pub async fn create_headless(&mut self, size: Size, scale: f64)
        -> Result<Surface>;

    pub async fn render(
        &mut self,
        surface: &mut Surface,
        scene: &Scene,
        parameters: Parameters,
    ) -> Result<Stats>;

    pub async fn resume_surface(
        &mut self,
        surface: &mut Surface,
        attachment: Attachment,
    ) -> Result<()>;

    pub async fn read_headless(&mut self, surface: &Surface)
        -> Result<ImageBuffer>;
}
```

The returned futures are not promised to be `Send`: WGPU 29 error-scope guards
are deliberately thread-local/non-`Send`, and window/WebGPU use is local to its
host event plane. The exclusive `&mut Renderer` borrow makes concurrent frame
transactions on one renderer unrepresentable. Native tests may drive these
futures with the already-pinned `pollster`; production methods contain no
`pollster::block_on`.

Each device registers both safe callbacks at creation:

- `set_device_lost_callback` writes the first typed loss record;
- `on_uncaptured_error` classifies an unexpected WGPU validation,
  out-of-memory, or internal error and writes it with the currently active
  operation generation into the same synchronized device signal state.

Every render-owned GPU operation uses a private `GpuOperationTransaction` with
a monotonic generation, draft public state, one or more owned command encoders,
and RAII leases. Before any WGPU resource/pipeline creation or submission, it
pushes nested WGPU 29 error scopes in outer-to-inner order `Internal`,
`OutOfMemory`, `Validation`. The internal Vello engine may prepare recordings,
write queue upload data, and encode compute passes only through the active
transaction; it cannot call `queue.submit`. After the transaction submits all
draw command buffers, it pops scopes in reverse order and awaits every returned
future. This is an asynchronous error-attribution boundary; it is not a buffer
map, pixel readback, busy poll, or blocking CPU wait.

An encoded Vello pass carries a private `VelloResourceLease`. Before submission,
drop/cancellation destroys or quarantines its uncertain transient resources.
After submission is accepted, queue ordering permits a successful transaction
to return compatible resources to the per-device idle pool. Scope failure or a
terminal device signal aborts publication and never makes the lease reusable
through an implicit `Drop` path. Persistent atlas writes are either committed
as valid cache content or marked dirty/recreated before the next raster use.

For a presented frame, the transaction owns the acquired surface texture.
Drawing is rendered into the internal Vello/custom intermediate without a
raster-engine surface convenience or submission call. After drawing scopes
resolve cleanly, the transaction encodes/submits the output blit/present pass
under a second set of the same three scopes and calls
`SurfaceTexture::present` while those scopes and the active operation generation
remain installed. It then pops/awaits all three scopes, rechecks
device-loss/uncaptured signals, and commits last-successful stats/parameters
only when clean. A present may therefore have been attempted for a frame
ultimately reported as failed; no rollback of an external surface side effect
is promised. A loss signaled after the final clean linearization point is
observed by the next operation and does not retroactively rewrite the completed
result.

Captured errors map deterministically: out of memory to
`SurfaceOutOfMemory`; validation to the owning create/render/present code;
internal to the owning create/render/present code. Safe WGPU/internal-raster
sources and descriptions are retained for display but never parsed for control
flow. Synchronous raster-planning/encoding errors and `CurrentSurfaceTexture`
variants enter the same transaction before commit.

An uncaptured error for the active generation aborts that operation and moves
the device to terminal `Faulted`; an uncaptured record arriving with no active
generation is consumed at the next device operation and also faults the device.
This fallback must be empty in normal execution because all owned calls are
scoped. It prevents WGPU's default uncaptured-error panic and prevents later
resource reuse after an error that escaped attribution. Public runtime reports
use `DeviceFaulted { kind }`, with `GpuFaultKind::{Validation,
OutOfMemory, Internal}`.

If any transaction stage fails, or if its future is canceled/dropped:

- error-scope guards pop safely on drop;
- an unpresented surface texture is discarded by WGPU ownership;
- frame leases are aborted and uncertain transient contents are not retained;
- no draft `Stats`, last parameters, route, or frame generation becomes public;
- already-submitted GPU work may finish, but it has no later semantic consumer.

Headless readback has a separate private, single-owner state machine:

```text
Allocated
  -> CopySubmitted { submission_index }
  -> MapPending
  -> Mapped
  -> PublishedBytes

Allocated | CopySubmitted | MapPending | Mapped
  -> Failed | Canceled
  -> staging buffer unmapped/dropped, never returned idle while uncertain
```

`read_headless` copies only the current published headless texture into a
row-padded `MAP_READ | COPY_DST` staging buffer, records the returned
`SubmissionIndex`, and registers one `map_async(MapMode::Read)` callback. The
callback owns an `Arc` completion cell, stores exactly one success/error result,
and wakes the latest registered task waker. Mapped bytes are copied while the
mapped view is alive, row padding is stripped with checked arithmetic, the view
is dropped, `unmap` is called, and only then is `ImageBuffer::try_new` invoked.
`get_mapped_range` is called only after callback success with the same validated
nonempty aligned range used for `map_async`, so its documented panic conditions
are excluded by construction rather than caught. `BufferAsyncError` and checked
row-decoding invariant failures map to `ReadbackFailed`; an impossible final
`ImageBuffer` length mismatch is also wrapped as `ReadbackFailed` rather than
exposed as caller-authored `InvalidValue`.

On native WGPU-core backends, a short-lived helper thread owns cloned safe
device/completion handles and drives the exact submission with
`Device::poll(PollType::Wait { submission_index: Some(index), timeout:
Some(50ms) })`. `PollError::Timeout` is a progress slice, not a public timeout;
it rechecks completion/cancellation and waits again. `WrongSubmissionIndex`
terminates as `ReadbackFailed` with its source. The helper never reads or
transforms pixels and never blocks the caller's async executor. On wasm,
`Device::poll` is not used; the browser event loop resolves WGPU's mapping
promise and invokes the same completion callback.

Dropping the readback future marks the completion cell canceled, calls safe
`unmap`, drops the public staging lease, and ensures a late callback discards
its result. The native helper exits at the next bounded poll slice; the callback
retains only cleanup ownership until terminal delivery. A canceled, failed, or
late readback cannot alter the surface's published texture, stats, parameters,
or resource state. Staging buffers with uncertain mapping state are dropped,
not pooled.

The no-wait production invariant means no map/readback, `Device::poll`, busy
loop, or blocking executor wait between passes. Awaiting the owned error-scope
futures after a submission stage and awaiting explicit `read_headless` mapping
are the only completion waits. Error scopes yield to the native/wasm async
host; explicit native readback delegates bounded `Device::poll` waits to its
helper thread, while wasm readback yields to the browser event loop.

## S14 Public Statistics

`Stats` remains a copyable last-frame value and retains the existing scene
counts and timings. It adds:

```rust
pub enum RenderRoute {
    DirectVello,
    GpuGraph,
}

pub enum EffectPrecision {
    High,
    Reduced,
}
```

and these fields:

- `route: Option<RenderRoute>`;
- `effect_precision: Option<EffectPrecision>`;
- `vello_passes: usize`;
- `image_passes: usize`;
- `composite_passes: usize`;
- `copy_operations: usize`;
- `custom_present_passes: usize`;
- `effect_texture_allocations: usize`;
- `effect_texture_reuses: usize`;
- `retained_effect_bytes: u64`.

`RenderRoute` and `EffectPrecision` are `Clone + Copy + Debug + Eq +
PartialEq`; `Stats` remains `Clone + Copy + Debug + Default + PartialEq`.
Before any successful frame, `Stats::default()` and `Renderer::stats()` have
`route == None`, no effect precision, and zero counters. A successful direct
frame has `route == Some(DirectVello)`, one Vello pass, zero custom pass/copy
counters, no effect precision, and no frame-caused effect allocation/reuse. A
successful graph frame has `Some(GpuGraph)`, the selected precision, and actual
events from this mapping:

| Executable operation | Counter increment |
| --- | --- |
| Direct scene render or `VelloCapture` | `vello_passes += 1` |
| `ClearRoot` | `image_passes += 1` |
| `CanonicalizeCapture` | `image_passes += 1` |
| `CopyBackdrop` or destination-parent copy | `copy_operations += 1` |
| `ColorFilter` | `image_passes += 1` |
| `BlurHorizontal`, `BlurVertical` | `image_passes += 1` for each pass |
| `DropShadowColorize` | `image_passes += 1` |
| `Composite` including source/shadow merge | `composite_passes += 1` |
| Custom graph `Present` | `custom_present_passes += 1` |

An elided identity/no-op emits no increment. Allocations and reuses count each
successful resource-manager lease acquisition by its actual source;
`retained_effect_bytes` is the byte-accounted idle total after end-of-frame
trim. Counts use saturating accumulation and cannot affect rendering decisions.

A contract-only, suspended, lost, or otherwise failed render returns `Err` and
does not fabricate a frame or change last-successful `Stats`. Resources acquired
by a failed in-progress frame are cleaned and may affect only private diagnostic
telemetry, never the public last-successful value. Surface creation, capability
queries, explicit readback, and resize operations do not alter render stats.

Existing image/cache statistics remain source-level telemetry; they are not
reinterpreted as authoritative Vello atlas internals. New resource statistics
come from the render-owned resource manager.

## S15 Frame Plan Invariants

`FramePlan` is private and uses the least powerful top-level enum:

```rust
enum FramePlan {
    DirectVello(DirectVelloPlan),
    GpuGraph(GpuRenderGraph),
}
```

`DirectVelloPlan` contains exactly one normalized Vello-encodable command tree,
surface mapping, and base color. It allocates no render-owned effect texture and
uses the current direct surface path.

`GpuRenderGraph` is a closed ordered graph. Resource and pass identities are
private generation-aware newtypes; no raw integer crosses the module boundary.
A builder can reference only a resource created earlier, a live imported image,
or the current parent target. The completed graph validates:

- every resource has one descriptor, one producer, an exact scheduled read
  count, and a lifetime ending after its final scheduled read;
- every read follows its producer in topological order;
- no pass reads and writes the same texture subresource;
- no released lease is referenced;
- every produced result reaches at least one later consumer or is the final
  output; immutable resources may fan out to multiple explicit reads without
  duplication, replay, or re-rendering;
- the graph has one root working image and one present operation;
- every Vello capture uses a transparent base;
- the surface base color is initialized exactly once;
- every semantic layer is composited in authored paint order;
- there is no CPU pixel node, readback node, arbitrary callback, or open plugin
  pass.

Empty command spans and degenerate transformed bounds lower to explicit no-op
results. They do not allocate a 1x1 semantic substitute or produce invalid
texture dimensions.

The builder records every consumer edge before scheduling and decrements a
resource's remaining-read count only after encoding that read. A lease becomes
releasable exactly at zero remaining reads. Drop shadow deliberately uses this
fan-out: the current filtered source is read once to derive/blur SourceAlpha and
again as the unchanged source-over input. The graph probe
`drop_shadow_source_fanout_lives_through_both_consumers` proves that a preceding
effect result supplies both branches and is released only after the merge.

## S16 Closed Pass Set

The graph's private pass set is finite:

```text
ClearRoot
VelloCapture
CanonicalizeCapture
CopyBackdrop
ColorFilter
BlurHorizontal
BlurVertical
DropShadowColorize
Composite
Present
```

`ClearRoot` creates the full-surface premultiplied working parent and clears it
to `Parameters::base_color` once. Nested isolated groups clear to transparent
black.

`VelloCapture` prepares and transaction-encodes one maximal consecutive
Vello-only span or one bounded subtree through the internal raster engine into
an `Rgba8Unorm` storage texture. It uses the selected
`Antialiasing`, transparent base color, and a local-to-capture transform that
preserves signed device origin. It does not apply the owning effect layer's
outer filter, clip, mask, opacity, or parent blend early.

`CanonicalizeCapture` samples Vello's unpremultiplied RGBA8 output and writes
premultiplied numeric-sRGB values to the selected working format. Each capture
is canonicalized exactly once.

`CopyBackdrop` copies the requested signed device rectangle from the completed
current parent into a temporary texture. Pixels outside the surface are
transparent black. It never clones or replays prior commands.

`ColorFilter` applies one or more authored color functions while preserving a
clamp boundary after each source function.

`BlurHorizontal` and `BlurVertical` apply one separable Gaussian operation.
They are distinct nodes with an intermediate texture and explicit edge policy.

`DropShadowColorize` samples the blurred SourceAlpha at a continuous offset and
produces the solid premultiplied shadow image. `Composite` then merges the
original source over that shadow before the next authored filter.

`Composite` combines a source with optional outer clip coverage, optional alpha
mask, opacity, transform/mapping, and one supported `BlendMode` into its parent.

`Present` samples the final working image, clamps, unpremultiplies safely, and
writes `Rgba8` headless/presented output or a supported `Bgra8` presented
output. Alpha zero always emits zero RGB. `Bgra8` headless remains rejected.
Headless output remains the same straight RGBA8 representation returned by
`ImageBuffer`.

WGSL files under `src/shaders/` are implementation source, loaded with
`include_str!`, and compiled by WGPU. Uniform/storage bytes are serialized by
explicit safe little-endian encoders with documented WGSL alignment and padding.
Custom pass serialization uses no pointer cast or derived POD implementation.
The already-present `bytemuck` crate is a direct dependency only for safe
Vello-compatible casts whose source types implement POD in pinned external
crates; the internal engine adds no owned `Pod`/`Zeroable` derive or unsafe impl.

## S17 Mixed-Scene Partitioning

Graph partitioning maximizes direct Vello work without changing semantic group
boundaries:

- maximal consecutive commands that need no result from a custom pass become
  one Vello capture;
- a Vello-only nested group may stay inside that capture when its clip,
  transform, opacity, isolation, and blend are fully local to the captured
  subtree;
- a group whose blend observes an external parent is captured without that
  blend and composited by render;
- an outer effect captures its source before that layer's filter/clip/mask/
  opacity/blend, while already-completed inner effects are included;
- foreground and filtered backdrop are separate sources until they are combined
  in the backdrop layer group;
- graph results never re-enter Vello through `register_texture`,
  `override_image`, or a newly constructed RGBA8 `Image`.

The internal raster engine encodes Vello capture work into command encoders
owned by the same frame transaction as custom passes. The executor may use one
encoder or an ordered finite encoder sequence according to WGPU usage rules,
but only the transaction submits them. WGPU queue ordering is the synchronization
contract; the renderer never maps a buffer, polls for CPU-visible completion,
or waits between production passes. The one async error-scope resolution after
the complete drawing submission stage is the S13A transaction commit boundary,
not an inter-pass synchronization edge.

## S18 Working Pixel Contract

`Format` remains the public output format. Private `WorkingFormat` describes
effect intermediates:

| Working format | Channels | Alpha | Color space | Required use |
| --- | --- | --- | --- | --- |
| `HighPrecision` | `Rgba16Float` | Premultiplied | Numeric sRGB | render attachment, sampled texture, copy source/destination |
| `ReducedPrecision` | `Rgba8Unorm` | Premultiplied | Numeric sRGB | render attachment, sampled texture, copy source/destination |

Working values are finite and clamped to `[0, 1]` at every CSS filter-function
boundary and at explicit CSS compositing boundaries that require clamping.
Intermediate Gaussian accumulation may use wider shader arithmetic but stores a
valid premultiplied result.

Selection is deterministic:

1. Use high precision when its complete required format feature set exists.
2. If high precision is unavailable and policy is `AllowReducedPrecision`, use
   reduced precision when its complete feature set exists.
3. Otherwise return `RuntimeCapabilityUnavailable` for `EffectRendering`.

No format is selected from adapter/backend names. No hidden downscale, CPU
route, or Vello-atlas re-entry is a quality fallback.

The high- and reduced-precision shader paths share one semantic implementation.
Only texture format and quantization differ. A real reduced-format executor test
must exercise the same production pipelines even on a machine where runtime
selection normally chooses high precision.

Reduced precision is compared in the representation it can preserve. For a
straight RGBA8 pixel `q`, define
`premul8(c, a) = (c * a + 127) / 255` with integer division. Reduced-route
quality compares alpha and each reconstructed `premul8` color channel; it does
not require unstable straight RGB at alpha near zero. Alpha zero still requires
RGB zero exactly. This is an explicit quality loss reported by
`EffectPrecision::Reduced`, not a weakening applied to the high-precision route.

## S19 Spatial And Sampling Model

Private spatial types distinguish:

- logical source bounds in the command's local space;
- logical filter execution and clip bounds;
- signed device origin (`i32`, `i32`);
- positive device extent (`u32`, `u32`);
- local-to-effect, effect-to-parent, and parent-to-surface transforms;
- texel-center mappings for every sampled resource.

Outward device conversion floors minima and ceils maxima after scale/transform,
uses checked arithmetic, preserves negative origins, and rejects values outside
the typed integer range. Texture dimensions are the positive extent only; the
signed origin remains in the mapping and is never clamped into an allocation.

Texel `(i, j)` represents the point at
`origin + ((i + 0.5) / raster_scale, (j + 0.5) / raster_scale)` in the mapped
logical space. Every pass uses this convention, preventing half-pixel drift
between Vello, filters, masks, and composition.

For a local-space effect under a 2D affine transform, raster scale is surface
scale multiplied by the transform's largest singular value. This produces an
isotropic local raster with enough resolution for the most magnified axis; the
effect is then transformed at composition. Thus a local circular blur remains
local and becomes appropriately elliptical under non-uniform scale or skew.
A zero singular value yields an explicit empty result. Non-finite or overflowing
derived values return typed invalid/runtime errors.

Backdrop effects execute in surface/device space. Transformed backdrops remain
diagnostic in this initiative, so the planner never guesses inverse-transform or
backdrop-root semantics.

When an effect extent exceeds `max_effect_texture_dimension_2d`, planning returns
`TextureDimensionExceeded` with the requested extent and limit. It does not
crop, tile, or reduce resolution silently. Tiled large effects are a separate
future behavior decision.

## S20 Filter Planning And Bounds

All current public types in `filter.rs` that describe kernels, outsets,
execution regions, device conversion, compiled color steps, or materialized
pipelines become private backend/algorithm types. `lib.rs` no longer reexports:

- `BlurPolicy`, `BlurRadiusInterpretation`, `KernelSupportRadius`,
  `LargeBlurRadiusAction`, `LargeBlurRadiusPolicy`, or
  `TransparentEdgeSamplingPolicy`;
- `FilterSourceBounds`, `FilterInflatedBounds`, `FilterClipBounds`,
  `FilterExecutionRegion`, `FilterRegionPlan`, `FilterOutset`,
  `DevicePixelConversionPolicy`, or `FilterDeviceBounds`;
- `CompiledColorFilterPipeline`, `MaterializedImageFilterPipeline`, or
  `MaterializedImageFilterStep`.

`FilterList::color_filter_pipeline`,
`FilterList::materialized_image_filter_pipeline`, public
`ColorFilterPipeline`, and public `ColorFilterOp` are removed as leaked
algorithm classification. The authored `FilterList`, `FilterOp`, and
`FilterOpKind` remain sufficient to inspect and preserve source meaning.

These values are not caller policy. Authored public `FilterList`, `FilterOp`,
and bounded backdrop inputs remain the semantic front door.

An internal filter planner folds over the list in authored order. For each
operation it records the current source bounds, result bounds, edge policy, and
pixel operation. Color-only and opacity operations preserve bounds. Blur expands
all sides by the selected finite support radius. Drop shadow expands by the
union of the original source and the offset blurred-alpha bounds; it does not
shift or discard the original source.

The CSS `blur()` value is Gaussian standard deviation in logical pixels. The
initial kernel support is the existing 2.5 standard deviations, with inclusive
integer taps through `ceil(2.5 * sigma)`. Zero standard deviation is identity
and emits no blur pass. `FilterBlur::try_new` now requires a finite value in the
closed interval `[0, 256]`; larger values return `InvalidValue` at construction.
They are not clamped because silent blur reduction is a semantic quality loss.

Kernel planning occurs on CPU as scalar metadata, not pixel execution. It:

- computes symmetric Gaussian weights in `f64`;
- normalizes the full discrete kernel to sum to one;
- converts finite weights/offsets to `f32` for upload;
- pairs adjacent taps for linear sampling when the selected format supports
  filterable sampling;
- retains an exact unpaired nearest/full-tap form as a backend-internal route
  only when it preserves the same sampled grid;
- caches immutable kernel buffers by standard-deviation bits, raster scale,
  support policy, and sampling form.

Ordinary blur samples transparent black outside the semantic source bounds,
implemented by shader bounds checks rather than allocation-edge sampler state.
Backdrop blur mirrors coordinates at the semantic backdrop border-box bounds,
not at padded allocation bounds.

## S21 Color Filter Semantics

Color functions operate on unpremultiplied numeric-sRGB channels. For each
source operation, the shader safely unpremultiplies (`alpha == 0` yields zero
RGB), evaluates the exact operation, clamps straight RGB and alpha to `[0, 1]`,
then premultiplies before proceeding. A fused GPU pass may loop over several
operations, but it may not collapse matrices or omit these per-function clamp
boundaries.

The specification, not the legacy CPU oracle, owns the scalar constants. Matrix
notation below is row-major and multiplies a column RGB vector:

| Operation | Exact straight-sRGB result before clamp |
| --- | --- |
| Brightness `a` | `c * a` |
| Contrast `a` | `(c - 0.5) * a + 0.5` |
| Grayscale `a` | `mix(c, vec3(L), a)`, where `L = 0.2126*r + 0.7152*g + 0.0722*b` |
| Hue rotate `theta` | `H(theta) * c`, using the exact matrix below |
| Invert `a` | `(1 - a) * c + a * (1 - c)` |
| Opacity `a` | premultiplied RGB and alpha multiplied by `a` |
| Saturate `a` | `S(a) * c`, using the exact matrix below |
| Sepia `a` | `mix(c, T*c, a)`, where `T` is the exact matrix below |

```text
S(a) =
[ 0.213 + 0.787a,  0.715 - 0.715a,  0.072 - 0.072a ]
[ 0.213 - 0.213a,  0.715 + 0.285a,  0.072 - 0.072a ]
[ 0.213 - 0.213a,  0.715 - 0.715a,  0.072 + 0.928a ]

H(theta), with C = cos(theta), S = sin(theta), is
[ 0.213 + 0.787C - 0.213S,  0.715 - 0.715C - 0.715S,  0.072 - 0.072C + 0.928S ]
[ 0.213 - 0.213C + 0.143S,  0.715 + 0.285C + 0.140S,  0.072 - 0.072C - 0.283S ]
[ 0.213 - 0.213C - 0.787S,  0.715 - 0.715C + 0.715S,  0.072 + 0.928C + 0.072S ]

T =
[ 0.393, 0.769, 0.189 ]
[ 0.349, 0.686, 0.168 ]
[ 0.272, 0.534, 0.131 ]
```

Every authored `f64` has one finite GPU-lowering rule:

| Public value | GPU scalar normalization |
| --- | --- |
| `UnitFilterAmount` | convert the validated `[0, 1]` value to nearest `f32` |
| non-negative `FilterAmount` | encode every finite value as private `{ zero, mantissa: f32, exponent: i32 }`, with positive value approximately `mantissa * 2^exponent` and mantissa in `[0.5, 1)` |
| `FilterAngle` | compute `radians.rem_euclid(2 * PI)` in `f64`, then convert to `f32` and evaluate sine/cosine |
| `FilterBlur` | constructor bounds the logical standard deviation to `[0, 256]`; multiply by raster scale in checked `f64`, reject non-finite/unrepresentable device support, then convert finite taps/weights to `f32` |
| shadow offsets and spatial values | transform/snap with S19 checked `f64` arithmetic; reject values outside the typed device range before finite `f32` uniform conversion |

The amount encoder derives exponent/significand from finite `f64` bits and
rounds only the normalized mantissa to `f32`; the full `f64` exponent range fits
`i32`. A mantissa rounding to `1.0` is renormalized to `0.5` with an incremented
exponent. Brightness, contrast, and saturation are lowered per channel to
`clamp(base + amount * delta, 0, 1)`: brightness uses `(base, delta) = (0, c)`,
contrast uses `(0.5, c - 0.5)`, and saturation uses the `a = 0` luminance row as
`base` with `delta = c - base`. A shared WGSL helper compares the encoded
amount against the positive/negative distance to the nearest clamp boundary by
normalized mantissa/exponent. If the product reaches a boundary it returns
exactly `0` or `1`; otherwise WGSL `ldexp` evaluates the product only after the
comparison proves the result finite and within range. Zero `delta` returns the
clamped base. Thus `f64::MAX`, subnormal amounts, and near-gray saturation never
construct an infinite/NaN `f32` and remain distinguishable wherever the target
working format can preserve the distinction. Every converted mantissa and
other scalar is checked with `is_finite` before upload.

Color operations preserve authored order, including order relative to blur,
opacity, and drop shadow. An identity run may be elided only when its
per-operation clamp cannot change any valid input. The test oracle is updated
to these constants. Independent known-vector tests (not oracle self-comparison)
cover grayscale primary colors, zero/identity saturation, zero/full sepia,
zero/quarter-turn hue rotation, and brightness/contrast saturation. Boundary
tests cover `f32::MAX`, `f64::MAX`, positive/subnormal `FilterAmount` exponent
boundaries, huge positive and negative angles, near-gray saturation, and the
largest accepted blur.

## S22 Drop-Shadow Model And Execution

CSS filter drop shadow does not reuse the broader box-shadow model. The public
semantic model becomes:

```rust
pub struct FilterDropShadow {
    offset: Point,
    blur: FilterBlur,
    color: Color,
}

impl FilterDropShadow {
    pub fn try_new(offset: Point, blur: FilterBlur, color: Color)
        -> Result<Self>;
    pub fn try_from_shadow(shadow: Shadow) -> Result<Self>;
    pub const fn offset(self) -> Point;
    pub const fn blur(self) -> FilterBlur;
    pub const fn color(self) -> Color;
}
```

`FilterDropShadow::try_new` validates finite offset, valid non-negative blur,
and finite solid color. It is `Clone + Copy + Debug + PartialEq` (not `Eq`
because its values contain floats). `FilterOpKind::DropShadow` stores this type;
`FilterOp::drop_shadow` accepts it. The authored diagnostic conversion remains:

```rust
impl FilterOp {
    pub const fn drop_shadow(shadow: FilterDropShadow) -> Self;
    pub fn try_drop_shadow(shadow: Shadow) -> Result<Self>;
}
```

Both paths validate outer kind, zero spread, and solid paint before creating
the executable value. Inset returns the existing `InsetBoxShadow` diagnostic;
nonzero spread returns typed `InvalidValue`; gradient or image paint returns
the existing `NonSolidShadowPaint` diagnostic. Thus unsupported authored input
remains representable and diagnosable while no executable drop-shadow value can
contain inset state, spread, or non-solid paint. Box-shadow APIs and their
existing support/diagnostics are otherwise unchanged.

One drop-shadow filter executes:

1. retain one immutable current-source resource with two explicit consumer
   edges;
2. extract SourceAlpha from the first read of the current full RGBA source;
3. Gaussian-blur that alpha with ordinary transparent-black edges;
4. sample the blurred alpha at the continuous logical offset using linear
   interpolation;
5. multiply it by the solid color alpha and premultiplied color;
6. read the same unchanged source resource through its second edge and
   source-over it above the shadow;
7. release the source after the merge, clamp, and feed that merged result to
   the next authored filter.

The offset is not rounded to an integer device pixel. Result bounds include the
original source and the full offset blur support on all signed axes. A fully
transparent input still produces transparent output.

`SHD-04` is complete only when this route agrees with the semantic oracle and
the prior CPU materialization path is absent. Non-solid filter shadow paint
remains `SHD-06` diagnostic.

## S23 Clip, Mask, Opacity, And Blend Composition

Outer layer operations execute in this exact order:

```text
children and inner groups
-> owning filter result, when any
-> outer clip coverage
-> resolved alpha mask
-> layer opacity
-> layer blend/composite into parent
```

For graph content, outer clip coverage is generated by Vello from the existing
render-owned `RenderClip` geometry into an antialiased RGBA8 coverage texture;
the composite pass samples alpha. This preserves Vello's fill rule and
antialiasing. Direct-only groups continue to use Vello's native layer/clip path.

Normal source-over uses premultiplied blending (`One`,
`OneMinusSrcAlpha`) directly into the parent. All other currently supported
blend modes (`Multiply`, `Screen`, `Overlay`, `Darken`, `Lighten`, and `Plus`)
use a destination-sampling composite:

1. copy the current parent to a distinct same-format target;
2. sample source and old parent;
3. evaluate the Compositing and Blending formula in premultiplied numeric sRGB;
4. clamp each result channel and alpha as required;
5. preserve old parent pixels outside the bounded composite rectangle;
6. swap the current parent identity.

`Plus` uses this path even though fixed-function additive blending exists,
because floating-point render attachments do not guarantee the CSS
plus-lighter clamp to one. A pass never samples from the texture it writes.

For separable blend modes, transparent source and transparent backdrop colors
are treated according to the standard source-over blend formula, not by
unconditionally dividing zero alpha. Group isolation always starts from
transparent black; root base color is never inserted into an isolated child.

## S24 Backdrop Contract

The supported backdrop subset remains bounded, top-level, untransformed, and
non-repeated. For each supported backdrop layer, the graph executes:

1. read the completed current root parent at the layer's paint position;
2. copy the requested signed device capture rectangle, including the root base
   color already present there;
3. apply the authored filter list in order, using mirror edges for backdrop
   blur at the semantic capture/border-box bounds;
4. apply the backdrop clip to the filtered result;
5. render the layer foreground to a transparent source;
6. source-over the foreground above the filtered backdrop into a transparent
   local group;
7. apply the layer's outer clip, resolved mask, opacity, and blend once;
8. composite the group into the root parent.

The renderer does not replay `source_commands`, clear a second base color, or
filter foreground content as part of the backdrop. Later siblings observe the
completed result.

Normalization no longer stores cloned prior commands in
`RenderBackdropCapture`; it stores only filters, semantic capture bounds, and
clip. The graph's current-parent dependency supplies source ordering.

Existing diagnostics remain for root backdrop policy, nested/backdrop-isolation
semantics, transformed backdrops, and repeated top-level backdrop capture.
Those inputs fail before backend execution with their existing typed
`PrimitiveOperation`; they never reach a partial GPU approximation.

## S25 Resource Manager

`Backend` owns a `DeviceState` for every selected WGPU device slot. Each state
contains:

- the private `VelloEngineState` for direct and capture raster work;
- immutable probed `DeviceCapabilities`;
- one persistent `ResourceManager`;
- shader modules, bind-group layouts, samplers, and render-pipeline caches;
- monotonic frame and resource generations.

Each `Renderer` also owns a private `RendererIdentity(Arc<()>)`. Every created
surface stores a clone of that identity and either no device identity
(`ContractOnly`) or a private `DeviceSlotIdentity { slot, generation }`.
Renderer methods first compare renderer identity with `Arc::ptr_eq`, then
validate the slot and generation, and only then index device storage. A failed
identity check performs no WGPU/Vello call and returns the exact S13 typed
`SurfaceIdentityMismatch`. No process-global numeric renderer ID or public raw
slot is introduced.

The private device lifecycle is a closed state machine:

```text
Ready { generation, ready_resources, loss_signal }
  -- observe DeviceLostCallback -->
Lost { generation, first_loss }

Ready { generation, ready_resources, gpu_error_signal }
  -- observe uncaptured WGPU error -->
Faulted { generation, first_fault }
```

Device setup registers WGPU's safe `set_device_lost_callback`. The callback
writes only the mapped `DeviceLossReason` and diagnostic message into an
`Arc<Mutex<Option<DeviceLossRecord>>>`; poison recovery retains the contained
record instead of panicking. It never touches renderer resources or a surface.
`DeviceState` observes that signal at entry to every device-requiring public
operation and again after submission/presentation before committing frame
success. Backend prose is retained for display/source context but is never
parsed to detect loss.

`Ready -> Lost` and `Ready -> Faulted` are terminal for that device generation.
Either transition takes and drops the internal raster engine, resource manager
contents, shader/pipeline caches, and
all idle leases before another operation can obtain them. Any active frame is
failed, its scope-owned leases are dropped, and last-successful surface
parameters/stats are not updated. A duplicate/late signal preserves the first
loss/fault record and performs no second cleanup. A frame that has presented and
passes the post-present signal check is the linearized success; a later loss is
observed by the next operation.

This renderer does not silently create a replacement device. Every later render,
readback, runtime-capability query, or surface resume that names the lost device
returns the same typed `DeviceLost` or `DeviceFaulted` reason without making a
WGPU/internal-raster call.
Recovery is explicit: the caller creates a new `Renderer` and recreates its
surfaces/scene resources. No handle or cache generation crosses renderer/device
instances.

The resource manager owns:

- Vello recording buffers, transient images, and atlas allocations keyed by
  exact internal usage role;
- transient working/capture/coverage textures keyed by exact descriptor and
  usage role;
- retained resolved-mask uploads keyed by `ImageId`, dimensions, and image
  sampling facts;
- Gaussian kernel buffers keyed by exact kernel plan;
- frame leases that make double release and stale reuse unrepresentable.

A resource state is `Leased` or `Idle`; the generation changes when an
allocation identity is replaced. Handles remain private. One frame may reuse an
idle compatible resource only after all prior queue use has been ordered before
the new frame submission. WGPU queue ordering supplies this guarantee without a
CPU wait.

At every frame boundary, successful or failed, all frame leases return through
safe scope-owned cleanup. Idle resources are trimmed deterministically by
`(last_used_frame, resource_id)` until retained byte count is at or below the
configured budget. Zero budget releases all idle byte-accounted resources.
Unknown driver memory for pipeline objects is not fabricated into byte stats;
pipeline caches live for the device state.

The budget counts texture allocation byte size by format and extent, uploaded
mask texture bytes, and kernel buffer bytes with checked `u64` arithmetic.
Active-frame bytes are reported internally but are not rejected by the
retention budget. Device limits are checked before allocation; WGPU allocation
or submission failures remain typed backend failures.

The internal raster atlas remains private to `VelloEngineState`, but every live
allocation and retained byte is registered with the same resource manager.
Render-owned mask uploads and raster-atlas entries use distinct role keys and
cannot alias accidentally.

## S26 Surface And Device Lifecycle

Headless targets add the safe usages needed by the custom present pass while
retaining internal Vello direct storage rendering and explicit readback. Headless
surfaces remain `Format::Rgba8` only; `Format::Bgra8` is rejected before WGPU
work with `SurfaceCreateFailed`, preserving the current contract. Presented
Rgba8/Bgra8 formats are selected only when advertised by the surface and the
graph present shader writes/swizzles to that selected format. Presented graph
frames safely acquire a surface texture, render/present to its supported
`Rgba8` or `Bgra8` view, and map acquire/present failures into the existing
surface lifecycle.

A nonzero headless surface owns zero or one published texture plus transient
draft textures. Both direct Vello and graph renders target a draft distinct
from the publication. After all S13A scopes/signals resolve, commit atomically
swaps the draft into the published slot and makes the previous publication an
idle resource. Failure/cancellation drops or quarantines the draft and leaves
the old publication byte-for-byte readable. Creation and a physical-size
change have no publication until a render succeeds.

Lifecycle rules are:

- resize updates logical/physical surface state and reallocates a target only
  when the physical extent changes;
- a zero-size presented surface remains non-renderable without allocating an
  effect graph;
- suspend prevents render and retains device-scoped caches;
- repeated `suspend` is an idempotent success;
- resume on the same device may reuse compatible device resources;
- compatible resume of an already-available surface is an idempotent success;
- surface loss (`wgpu::SurfaceError::Lost`) marks only that presented surface
  lost; `resume_surface` may recreate it on the same ready device;
- device loss is distinct from surface loss, makes the device state terminal,
  and causes every surface naming that device to report `DeviceLost` on its
  next operation;
- `runtime_capabilities` reports `Unavailable(DeviceLost { .. })` or
  `Unavailable(DeviceFaulted { .. })` after the corresponding signal is
  observed;
- `resume_surface` never revives a terminal lost device and returns that same
  typed diagnostic;
- renderer drop releases every state through safe Rust/WGPU ownership.

Surface identity and state are checked in this order: renderer identity,
operation/backend-kind compatibility, device slot/generation, public
suspension, private lifecycle, then requested GPU capability. Thus a presented
read or headless `resume_surface` remains `UnsupportedBackend` even if that
unrelated device later becomes terminal. The resulting operation matrix is
complete:

| Surface condition | `render` | `read_headless` | `resume_surface` | `runtime_capabilities` |
| --- | --- | --- | --- | --- |
| Foreign renderer | `SurfaceRendering` + `ForeignRenderer` | `SurfaceReadback` + `ForeignRenderer` | `SurfaceResume` + `ForeignRenderer` | `Unavailable(ForeignRenderer)` |
| Stale device generation | owning operation + `StaleDeviceGeneration` | same | same | `Unavailable(StaleDeviceGeneration)` |
| Contract-only, nonzero, available | `SurfaceRendering` + `AdapterUnavailable` | `SurfaceReadback` + `AdapterUnavailable` | `UnsupportedBackend` for renderer resume | `Unavailable(AdapterUnavailable)` |
| Requested headless, zero-size, contract-only | `NonRenderable` before adapter lookup | validated empty image | `UnsupportedBackend` for renderer resume | `Unavailable(AdapterUnavailable)` |
| Headless, zero-size, device-backed | `NonRenderable` | validated empty image | `UnsupportedBackend` for renderer resume | device report if ready; otherwise terminal device report |
| Headless, nonzero, no publication | allocate/render draft, then publish on success | `SurfaceReadback` + `Uninitialized` | `UnsupportedBackend` for renderer resume | device report |
| Headless, nonzero, published | render separate draft, then swap on success | map current publication | `UnsupportedBackend` for renderer resume | device report |
| Any headless public state `Suspended` | `SurfaceRendering` + `Suspended` | `SurfaceReadback` + `Suspended` | `UnsupportedBackend`; use compatible `Surface::resume` | device report unless device terminal |
| Presented `Ready` | acquire/render/present | `UnsupportedBackend` | idempotent success for compatible attachment | available device/surface report |
| Presented `ResizePending` | configure requested extent, then render; failed configure leaves pending | `UnsupportedBackend` | compatible resume applies pending attachment/lifecycle | available report if format/device remain valid |
| Presented `NonRenderable` | `SurfaceRendering` + `NonRenderable` | `UnsupportedBackend` | compatible resume preserves requested zero extent | `Unavailable(NonRenderable)` |
| Presented `Occluded` | `SurfaceRendering` + `Occluded` | `UnsupportedBackend` | compatible resume may reacquire; otherwise remains occluded | `Unavailable(Occluded)` |
| Presented `Lost`, ready device | `SurfaceRendering` + `Lost` | `UnsupportedBackend` | recreates/configures surface and enters requested ready/nonrenderable state | `Unavailable(Lost)` until resume succeeds |
| Any compatible GPU operation naming terminal device | owning operation + same `DeviceLost`/`DeviceFaulted` | same when headless | same when presented | same terminal unavailable reason |

`Surface::suspend` is idempotent for every kind. Compatible `Surface::resume`
is idempotent for headless/contract-only surfaces; presented surfaces continue
to require async `Renderer::resume_surface`. Calling either resume form on an
already-available surface with a compatible attachment is a no-op and retains
the currently installed attachment; replacing an attachment requires a prior
`suspend` or a private `Lost` lifecycle. An incompatible attachment kind
returns `SurfaceCreateFailed` without changing state. A resize while suspended
updates the desired logical/physical size but does not resume. Same-physical-size
headless resize preserves its publication; a changed physical size immediately
drops the publication and enters `Empty` or `PendingAllocation`. A presented
resize records `ResizePending` or `NonRenderable`; it does not configure until
the next async render/resume.

Public queries are exact projections: `Surface::state()` reports only
`Available`/`Suspended`; `physical_size()` reports the requested current extent;
and `resource_state()` reports `ContractOnly`, `Empty` for zero-size headless,
`PendingAllocation` for nonzero headless without a current-size publication,
`Ready` only with a readable publication, or `Presented` for every presented
lifecycle. No query triggers allocation, resume, callback consumption, or WGPU
work.

Zero-size headless behavior is explicit:

| Operation/state | Result |
| --- | --- |
| Create available `Rgba8` headless surface at zero physical width or height | Success with no texture allocation and `SurfaceResourceState::Empty` when a device exists; contract-only remains `ContractOnly` |
| Render zero-size headless surface | `SurfaceUnavailable { NonRenderable }` before adapter/effect planning; last successful state unchanged |
| Read available zero-size headless surface | `ImageBuffer::try_new(physical_size, vec![])`; no adapter, map, or pixel operation required |
| Read suspended zero-size headless surface | `SurfaceUnavailable { Suspended }` |
| Resize nonzero headless to zero | Drop target, enter empty state, preserve no stale pixels |
| Resize zero headless to nonzero | Enter pending allocation; next async render allocates under S13A scopes |
| Read nonzero headless before first successful render or after size-changing resize | `SurfaceUnavailable { Uninitialized }`; no staging allocation |
| Failed/canceled render with prior publication | Preserve prior publication and `Ready` resource state |
| Failed/canceled render without prior publication | Preserve `PendingAllocation`; read remains `Uninitialized` |
| Create any `Bgra8` headless surface, including zero-size | Preserve `SurfaceCreateFailed` format rejection |

`SurfaceResourceState` adds `Empty`; it remains `Clone + Copy + Debug + Eq +
PartialEq`. The private headless resource state adds `Empty` alongside
`Pending` and `Ready`. A zero-area `ImageBuffer` remains valid exactly for this
readback/fixture result.

Device loss is isolated by WGPU device slot, not renderer-wide:

| Operation after one slot is lost/faulted | Result |
| --- | --- |
| Existing surface naming that slot | Same terminal `DeviceLost`/`DeviceFaulted` diagnostic, no backend call |
| `create_headless` when the renderer's default headless slot is terminal | Same terminal diagnostic; no automatic default-device replacement |
| `resume_surface` naming the terminal slot | Same terminal diagnostic |
| Existing surface on another ready slot | Continues using its own resources and capabilities |
| New presented surface whose safe WGPU selection yields another/new ready slot | Register an independent `DeviceState` and succeed |
| New presented surface whose selected slot is terminal | Return that slot's terminal diagnostic |

No operation silently replaces a terminal generation. Explicit full recovery
for a lost default/headless device remains construction of a new `Renderer`;
healthy slots in the old renderer are not unnecessarily disabled.

`apply_presented_resize_state` no longer calls `as_hal`, touches a Metal layer,
or contains `unsafe`. `set_surface_resizing` preserves its public state tracking
and becomes a safe rendering hint with no backend-specific side effect. This
minor optimization is not observable render semantics.

`read_texture_rgba` is replaced by the S13A readback state machine and remains
reachable only from explicit `Renderer::read_headless` and test assertions.
Production `Renderer::render` and every function it calls contain no map,
poll-for-readback, CPU pixel execution, or readback/re-upload edge.

## S27 CPU Reference Oracle

`reference.rs` is declared only under `#[cfg(test)]`. CPU pixel types and
algorithms are crate-private fixture/oracle code. `image.rs`, `filter.rs`, and
production renderer modules have no import from `reference`.

The oracle defines deterministic premultiplied RGBA semantics for:

- straight/premultiplied conversion;
- every supported color function with a clamp after each operation;
- Gaussian blur with the same finite support and edge rule;
- continuous drop-shadow offset and source-over merge;
- resolved alpha-mask nearest/bilinear/Mitchell sampling, all `Extend` modes,
  semantic-boundary transparency, and multiplication;
- the currently supported blend set;
- ordinary source-over composition and backdrop group order.

The oracle is used to compute expected bytes or floats for tests only. A GPU
result may be explicitly read after rendering to assert it. The existence of a
readback assertion does not make readback a production execution edge.

Oracle color constants are copied from S21 into independent test code rather
than imported from shader/uniform implementation constants. S21 known-vector
tests carry literal expected values so an identical oracle/shader coefficient
mistake cannot self-validate.

Production source contains no `cfg(not(test))` dead-code allowance for CPU
executors and no public type whose purpose is to execute materialized pixels.

## S28 Module And Code Outline

The desired implementation remains within the existing crate and uses these
ownership boundaries:

| Module/area | Desired responsibility |
| --- | --- |
| `renderer.rs` | Public orchestration, successful-frame state update, direct/graph dispatch, runtime capability projection |
| `command.rs` | Normalized semantic render commands, bounds contribution, no cloned backdrop source commands |
| `frame.rs` | Private `FrameContext`, `FramePlan`, graph builder, partitioning, logical/spatial planning, graph validation |
| `backend.rs` | WGPU device and surface integration, async scoped GPU-operation transactions, safe acquisition/submission/presentation, per-device loss/fault state lookup |
| `vello_engine/scene.rs` | Private Vello-compatible scene lowering retained from the pinned main crate, with no public reexport |
| `vello_engine/glyph.rs` | Fallible S10A selected-glyph/table/image preflight before any external encoding; no shaping or inferred bounds |
| `vello_engine/recording.rs` | Private resource-intent and compute-dispatch IR; no WGPU submission |
| `vello_engine/raster.rs` | Fixed coarse/fine Vello recording schedule over external `vello_encoding` types |
| `vello_engine/shaders.rs` | Checked WGPU pipeline construction from external `vello_shaders::SHADERS`; no CPU/hot-reload path |
| `vello_engine/encoder.rs` | Encode prepared raster work into transaction-owned command encoders and produce `VelloResourceLease` values |
| `vello_engine/mod.rs` | Per-device `VelloEngineState`, private phase transitions, and adapted error boundary |
| `readback.rs` | Private staging/map state machine, native bounded poll helper, wasm callback completion, cancellation cleanup |
| `resource.rs` | Private generation-aware Vello/effect leases, texture/upload/kernel retention, deterministic budget trimming and stats |
| `pass.rs` | Private executable pass/resource nodes, runtime lowering, scheduler, bind groups, command encoding |
| `shader.rs` | Pipeline keys/cache creation, safe uniform encoders, shared shader constants |
| `src/shaders/*.wgsl` | Canonicalization, color filter, blur, shadow colorize, composite/blend, and present shader source |
| `filter.rs` | Private filter classification, scalar formulas, bounds, kernels, and algorithm plans |
| `reference.rs` | `cfg(test)` CPU oracle only |
| `capability.rs` | Static semantic capabilities plus public runtime capability projection types |
| `error.rs` | Typed semantic and runtime diagnostics |
| `image.rs` | Validated images/readback buffers; no CPU effect executor |
| `layer.rs` | Resolved mask semantic input and preserved layer contracts |
| `text.rs` | Validated immutable font bytes/index plus explicit run-local text bounds; no shaping or inferred ink geometry |
| `stats.rs` | Route/pass/resource telemetry |
| `surface.rs` | Existing safe dynamic lifecycle and output target state |

`frame.rs`, `resource.rs`, `pass.rs`, `readback.rs`, `vello_engine/`, and all
shader implementation modules are private. The public front door remains
reexports from `lib.rs`; no `wgpu`, `vello_encoding`, `vello_shaders`, internal
Vello engine, graph, resource handle, shader key, or working-format type is
exposed.

The existing `texture.rs` and `shader.rs` probes may be absorbed into the new
private owners rather than duplicated. There is one authoritative texture
lifecycle model and one pipeline cache after migration. Superseded
`OffscreenTextureResourceCache`, per-effect caches, clear-only shader probes,
and materialization helpers are removed once their final consumer moves.

## S29 Public Compatibility Classification

The initiative intentionally contains breaking public changes:

| Surface | Classification | Required replacement |
| --- | --- | --- |
| `Options` public fields | Breaking | Private fields, accessors, and `with_*` builders |
| `Options::use_cpu` | Breaking removal | None; production is GPU-only |
| GPU-touching `Renderer` methods | Breaking sync-to-async change | Await exact S13A methods; no production `block_on` shim |
| `Capabilities::VELLO_0_9` | Breaking rename | `Capabilities::CURRENT` |
| Materialized/CPU capability accessors | Breaking removal/rename | Semantic GPU execution accessors |
| Public filter algorithm/compiled-plan types | Breaking removal | None; callers retain authored `FilterList`/`FilterOp` |
| `ImageBuffer` public fields | Breaking | `try_new`, `size`, `rgba`, `into_rgba` |
| `ResolvedAlphaMaskExecution` | Breaking removal | None in production; private test oracle |
| Resolved alpha-mask API | Breaking | `ResolvedLayerAlphaMask::try_new(Image, Rect)`, `image`, `bounds`, and `Layer::with_resolved_alpha_mask`; remove buffer `size`/`mode`/`alpha_mask` accessors and `Layer::try_resolved_alpha_mask` |
| `ResolvedLayerAlphaMask: Eq` | Breaking trait removal | `Clone + Debug + PartialEq` only |
| Text-run construction | Breaking | Append `bounds: TextRunBounds` after `glyphs` in the exact S10 `TextRun::try_new` signature |
| `FontData::from_bytes` | Breaking replacement | `FontData::try_from_bytes` validates bytes and collection index and returns the exact S10A typed `InvalidValue`; no infallible alias |
| Filter drop-shadow payload | Breaking | `FilterDropShadow` instead of broad `Shadow` |
| `FilterBlur::try_new` accepted range | Breaking behavior | Values above 256 logical pixels now return typed `InvalidValue`; no clamp or fallback |
| `Error` public fields/raw constructor | Breaking | Private fields plus `code`, `message`, typed accessors, semantic constructors |
| `Error` backend source auto-traits | Target-specific contract | `Send + Sync` source/Error on native; no such promise on wasm; `source()` preserved on both |
| `ErrorCode::ReadbackFailed` | Additive enum variant | Exact readback copy/map/poll classification in S13/S13A |
| Runtime capability/error types | Additive plus enum changes | Exact S12-S13 report, reason, operation, identity kind, private validated diagnostic construction, and accessors |
| `Stats` fields and route enums | Breaking for struct literals, additive observation | Use `Stats::default`/returned values and new typed fields |
| `SurfaceResourceState` | Additive enum variant, breaking for exhaustive matches | Add `Empty` for device-backed zero-size headless surfaces |
| Contract-only render/read failures | Breaking behavior | Typed `RuntimeCapabilityUnavailable` with distinct `SurfaceRendering`/`SurfaceReadback`, not generic backend codes |
| Zero-size headless render | Breaking behavior | Typed `SurfaceUnavailable { NonRenderable }`; no successful no-op frame |
| Zero-size headless read | Additive behavior | Validated empty `ImageBuffer` without GPU work |
| Nonzero headless read before publication/after size change | Breaking behavior | Typed `SurfaceReadback` + `SurfaceUnavailable { Uninitialized }` |
| Headless failed/canceled frame visibility | Corrected behavior | Draft/published swap preserves last successful readable pixels |
| Foreign/stale surface use | Corrected typed behavior | `SurfaceIdentityMismatchKind::{ForeignRenderer, StaleDeviceGeneration}` before slot access |
| Duplicate suspend/compatible resume | Clarified behavior | Idempotent success; incompatible attachment remains `SurfaceCreateFailed` |
| Headless `Bgra8` | Behavior preserved | Remains a `SurfaceCreateFailed` diagnostic |
| Affected diagnostic enum variants | Breaking rename/removal | Exact S11 and S13 mappings |

No deprecated alias, forwarding constructor, duplicate enum variant, or
compatibility feature is added. Root adaptation and generated API refresh occur
later in the root-owned integration cycle.

Unrelated public fields and models are not privatized merely as cleanup. This
specification changes only surfaces required to encode the GPU-only boundary or
remove leaked algorithm/test phases.

`lib.rs` applies this exact affected reexport inventory:

| Action | Public names |
| --- | --- |
| Remove | `BlurPolicy`, `BlurRadiusInterpretation`, `CompiledColorFilterPipeline`, `DevicePixelConversionPolicy`, `FilterClipBounds`, `FilterDeviceBounds`, `FilterExecutionRegion`, `FilterInflatedBounds`, `FilterOutset`, `FilterRegionPlan`, `FilterSourceBounds`, `KernelSupportRadius`, `LargeBlurRadiusAction`, `LargeBlurRadiusPolicy`, `MaterializedImageFilterPipeline`, `MaterializedImageFilterStep`, `TransparentEdgeSamplingPolicy` |
| Remove | `ColorFilterPipeline`, `ColorFilterOp`, `ResolvedAlphaMaskExecution` |
| Add from capability | `RuntimeCapabilities`, `AvailableRuntimeCapabilities`, `EffectPrecisionCapabilities` |
| Add from renderer | `EffectQualityPolicy`, `ResourceCacheBudget` |
| Add from error | `RuntimeCapabilityUnavailable`, `RuntimeCapabilityUnavailableReason`, `RuntimeOperation`, `RenderSurfaceAvailability`, `SurfaceIdentityMismatchKind`, `DeviceLossReason`, `GpuFaultKind` |
| Add from filter/style | `FilterDropShadow` |
| Add from stats | `RenderRoute`, `EffectPrecision` |
| Add from text | `TextRunBounds`, `TextRunBoundsKind` |

All existing public names not explicitly removed/renamed in S08-S14, S20,
S22, or the table above retain their current reexport and role.

## S30 Overlay Reconciliation

The following overlay rows receive a changed implementation or corrected
contract. Every other row retains the overlay's `Preserve`, `HoldDiagnostic`,
or `HoldRoot` disposition and must be regression-checked at final
reconciliation.

| Overlay IDs | Desired result | Specification sections |
| --- | --- | --- |
| `SHD-04` | Correct SourceAlpha Gaussian drop shadow with continuous offset and source merge on GPU | S20-S22 |
| `FLT-02` | GPU separable Gaussian blur | S16, S18-S20 |
| `FLT-03`, `FLT-04`, `FLT-06`-`FLT-10` | Ordered GPU sRGB color/alpha functions | S18, S21 |
| `FLT-05` | Correct grayscale coefficients/order/clamp on GPU | S21 |
| `FLT-12` | Fusion preserves every authored per-function clamp | S20-S21 |
| `FLT-13` | Signed, outward-snapped, transform-aware source/result/device bounds | S19-S20 |
| `FLT-14` | CPU implementation is a private test oracle only | S03, S27 |
| `BDP-01` | Capture completed prior GPU parent once with explicit bounded mapping | S16-S17, S24 |
| `BDP-02` | Ordered GPU backdrop filter chain and group composition | S20-S24 |
| `MSK-05` | Image-backed resolved alpha-mask composite on GPU | S09, S23 |
| `DIA-02` | Separate semantic and lifecycle-aware runtime capability contracts | S11-S13 |
| `BEP-02` | Persistent per-device effect resource/upload lifecycle | S25-S26 |
| `BEP-03` | Bounded Vello capture in a closed graph | S15-S17 |
| `BEP-04` | Real source-bound image/composite/present shader pipelines | S16, S28 |
| `BEP-05` | High-quality separable GPU blur | S18-S20 |
| `BEP-07` | GPU backdrop capture/filter/composite without replay/readback | S24 |
| `BEP-08` | CPU reference path cannot compile into production | S27 |

`BEP-06` broad mask compositor remains diagnostic. The internal composite pass
supports only the already-resolved alpha-mask contract and does not imply shape,
luminance, multi-layer, or mask-composite support.

The correction register is satisfied as follows:

| Correction | Required outcome | Specification sections |
| --- | --- | --- |
| `COR-01` | Root base once; nested isolated groups transparent; backdrop reads completed parent | S16, S23-S24 |
| `COR-02` | Signed outward-snapped device bounds | S19-S20 |
| `COR-03` | Explicit transform/effect spaces, continuous shadow offset, expanded bounds | S19-S22 |
| `COR-04` | Real caller-supplied glyph ink bounds | S10, S19 |
| `COR-05` | No production readback, per-effect cache, CPU pixels, or re-upload | S17, S25-S27 |
| `COR-06` | Semantic/runtime capability split and typed alternate-GPU quality evidence | S11-S14, S18 |
| `COR-07` | AA, capture-grid, transform, base, and effect parity coverage | S32-S33 |
| `COR-08` | Numeric-sRGB operation order/clamp and distinct ordinary/backdrop edges | S18, S20-S21, S24, S27 |

### S30A Property Cross-Reference

This table directly incorporates the 22 property rows from
`plans/2026-07-10-render-css-matrix-reconciliation.md`; no reader must follow
the overlay to a second planning source. The listed mixed boundaries are part
of each property's final contract.

| CSS/style surface | Required primitive/status mapping after this initiative |
| --- | --- |
| `color`, `background-color`, border/outline/text-decoration color | `PNT-01` supported; `PNT-02` root-owned; `PNT-03` diagnostic |
| `background-image` | `PNT-04`-`PNT-06`, `PNT-08` supported; `PNT-07`, `PNT-09` diagnostic |
| `background-position` | `IMG-02` supported |
| `background-size` | `IMG-03`, `RES-02` supported |
| `background-repeat` | `IMG-04` supported; `IMG-05`, `IMG-06` diagnostic |
| `background-origin` | `IMG-07` supported |
| `background-clip` | `IMG-08`, `MSK-01`, `MSK-02`, `MSK-04` supported |
| `background-attachment` | `IMG-09`, `XFM-05` supported |
| borders, side border style/width | `BOX-02`-`BOX-05` supported; `BOX-06` diagnostic |
| `border-radius` | `GEO-02`, `BOX-07` supported |
| outline and longhands | `BOX-08` solid/dashed/dotted supported; `BOX-09.OutlineDoubleStyle` and `BOX-09.OutlineAutoStyle` diagnostic |
| `box-decoration-break` | `BOX-10` supported |
| `box-shadow` | `SHD-01` rect/rounded/circle and `SHD-03` supported; `SHD-01.EllipsePathShadowShape`, `SHD-02`, and `SHD-06` diagnostic |
| `opacity` | `CMP-01`, `CMP-04` supported |
| `filter` | `FLT-01`-`FLT-10`, `FLT-12`, `FLT-13` supported on the GPU route; `SHD-04` corrected and supported; `FLT-14` oracle-only; `FLT-11.FilterResource`, `FLT-01.LayerFilter`, and `FLT-01.LayerFilterExecution` remain diagnostic |
| `backdrop-filter` | `BDP-01`, `BDP-02` corrected and supported; `BDP-03`, `BDP-04` diagnostic |
| `clip-path` | `MSK-01`, `MSK-02`, `MSK-04` supported; `MSK-03` diagnostic |
| `mask`, `mask-image`, size/position/repeat | `MSK-05` corrected and supported; `MSK-06`-`MSK-08` plus `BEP-06.LayerMask`, `BEP-06.AlphaMaskSourceExecution`, and `BEP-06.MaskExecution` diagnostic; applicable `IMG-02`-`IMG-04` placement primitives remain supported without enabling broad mask-layer execution |
| transform and transform longhands | `XFM-01`-`XFM-03`, `XFM-05` supported; `XFM-04` diagnostic |
| `text-shadow` | shared `TXT-03.TextShadow`/`SHD-05.TextShadow` diagnostic |
| `::selection` | `TXT-04` root-owned; resulting ordinary commands preserve their own primitive statuses |
| `content`, pseudo-elements, list markers | `TXT-05` root-owned; resulting ordinary commands preserve their own primitive statuses |

A property passes final reconciliation only when every ID in its row has its
specified executable support, typed diagnostic, or root boundary. GPU plumbing
cannot turn a mixed row into blanket support. The root handoff must later adapt
renamed public APIs without changing these ownership/status outcomes.

### S30B Complete Final Primitive Inventory

This specification owns the final disposition of all 101 primitive/backend
IDs. `Supported` means executable production behavior, `Diagnostic` means a
preserved typed rejection/capability boundary, `Root` means
`DeferredToRoot`, and `OracleOnly` means test-only CPU evidence with no
production route.

| ID | Primitive/backend row | Final disposition | Final route |
| --- | --- | --- | --- |
| `PNT-01` | Solid RGBA paint | Supported | `Normalize -> VelloDirect`; explicit pass-boundary conversion |
| `PNT-02` | Symbolic color token | Root | Root resolves before render |
| `PNT-03` | Paint-space color conversion | Diagnostic | Typed unsupported/degraded color boundary |
| `PNT-04` | Linear gradient | Supported | `Normalize -> VelloDirect` |
| `PNT-05` | Radial gradient | Supported | `Normalize -> VelloDirect` |
| `PNT-06` | Conic/sweep gradient | Supported | `Normalize -> VelloDirect` |
| `PNT-07` | Repeating gradients | Diagnostic | Typed repeating-gradient boundary |
| `PNT-08` | Image paint | Supported | `Normalize -> VelloDirect`; retained image identity |
| `PNT-09` | Filtered image paint | Diagnostic | Typed filtered-image boundary |
| `GEO-01` | Rect fill/stroke | Supported | `VelloDirect`/bounded Vello capture |
| `GEO-02` | Rounded rect fill/stroke | Supported | `VelloDirect`/bounded Vello capture |
| `GEO-03` | Circle/ellipse fill/stroke | Supported | `VelloDirect`/bounded Vello capture |
| `GEO-04` | Arbitrary path fill | Supported | `VelloDirect`/bounded Vello capture |
| `GEO-05` | Arbitrary path centered stroke | Supported | `VelloDirect`/bounded Vello capture |
| `GEO-06` | Arbitrary path inside/outside stroke | Diagnostic | Typed alignment boundary |
| `GEO-07` | Geometry boolean/offset support | Diagnostic | Typed geometry-operation boundary |
| `GEO-08` | Hit-test geometry | Root | Root owns hit testing |
| `IMG-01` | Image fit | Supported | `Normalize -> VelloDirect` |
| `IMG-02` | Background position | Supported | `Normalize` |
| `IMG-03` | Background size | Supported | `Normalize` |
| `IMG-04` | Repeat no-repeat/repeat | Supported | `Normalize -> VelloDirect` |
| `IMG-05` | Repeat round | Diagnostic | Typed repeat-round boundary |
| `IMG-06` | Repeat space | Diagnostic | Typed repeat-space boundary |
| `IMG-07` | Background origin | Supported | `Normalize` |
| `IMG-08` | Background clip | Supported | `Normalize -> VelloDirect`/graph clip coverage |
| `IMG-09` | Background attachment | Supported | `Normalize -> VelloDirect` with coordinate tags |
| `IMG-10` | Multi-layer image stack | Supported | Authored-order `Normalize -> VelloDirect` |
| `BOX-01` | Background layer stack | Supported | `Normalize -> VelloDirect` |
| `BOX-02` | Border side solid | Supported | `Normalize -> VelloDirect` |
| `BOX-03` | Border style none/hidden | Supported | Normalized suppression |
| `BOX-04` | Border dashed/dotted | Supported | `Normalize -> VelloDirect` |
| `BOX-05` | Border double | Supported | Normalized Vello bands |
| `BOX-06` | Border groove/ridge/inset/outset | Diagnostic | Typed 3D-border boundary |
| `BOX-07` | Border radius clipping | Supported | `Normalize -> VelloDirect`/graph coverage |
| `BOX-08` | Outline solid/dashed/dotted | Supported | `Normalize -> VelloDirect` |
| `BOX-09` | Unsupported outline styles | Diagnostic | Typed `OutlineDoubleStyle`/`OutlineAutoStyle` sub-boundaries |
| `BOX-10` | Box decoration break | Supported | Normalized Vello fragments |
| `SHD-01` | Outer box shadow | Supported | Rect/rounded/circle through Vello; ellipse/path is an explicit diagnostic subcase |
| `SHD-02` | Inset box shadow | Diagnostic | Typed inset-shadow boundary |
| `SHD-03` | Multiple shadows | Supported | Ordered Vello commands |
| `SHD-04` | Drop shadow filter | Supported | Source/image -> GPU blur/colorize/composite |
| `SHD-05` | Text shadow | Diagnostic | Typed text-shadow boundary |
| `SHD-06` | Non-solid shadow paint | Diagnostic | Typed non-solid-paint boundary |
| `FLT-01` | Filter list model | Supported | Ordered normalized pass intents; broad layer execution remains an explicit diagnostic subcase |
| `FLT-02` | Blur | Supported | GPU horizontal/vertical `ImagePass` |
| `FLT-03` | Brightness | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-04` | Contrast | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-05` | Grayscale | Supported | Corrected sRGB GPU `ColorFilter` |
| `FLT-06` | Hue rotate | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-07` | Invert | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-08` | Opacity filter | Supported | Ordered premultiplied GPU `ColorFilter` |
| `FLT-09` | Saturate | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-10` | Sepia | Supported | Ordered sRGB GPU `ColorFilter` |
| `FLT-11` | URL/SVG/reference filter | Diagnostic | Typed unresolved/unsupported filter-graph boundary |
| `FLT-12` | Filter fusion | Supported | Fused GPU pass with per-source-operation clamps |
| `FLT-13` | Filter region/outsets | Supported | Explicit signed source/result/clip/device bounds |
| `FLT-14` | Software/reference fallback | OracleOnly | Private `cfg(test)` CPU oracle; no fallback |
| `BDP-01` | Backdrop capture | Supported | Copy completed current GPU parent in signed bounds |
| `BDP-02` | Backdrop filter chain | Supported | `CopyBackdrop -> ImagePass -> CompositePass` |
| `BDP-03` | Backdrop isolation | Diagnostic | Typed nested/backdrop-root boundary |
| `BDP-04` | Root backdrop policy | Diagnostic | Typed root/host-policy boundary |
| `MSK-01` | Shape clip | Supported | Vello direct or graph coverage dependency |
| `MSK-02` | Path clip | Supported | Vello fill-rule clip or graph coverage dependency |
| `MSK-03` | Clip URL/reference | Diagnostic | Typed unresolved clip boundary |
| `MSK-04` | Basic shape clip | Supported | `Normalize -> VelloDirect`/graph coverage |
| `MSK-05` | Alpha mask | Supported | Image alpha -> premultiplied GPU `CompositePass` |
| `MSK-06` | Luminance mask | Diagnostic | Typed luminance-mode boundary |
| `MSK-07` | Multi-layer mask | Diagnostic | Typed mask-stack boundary |
| `MSK-08` | Mask composite | Diagnostic | Typed mask-composite boundary |
| `CMP-01` | Layer opacity | Supported | Vello direct or ordered GPU composite |
| `CMP-02` | Mix blend mode | Supported | Current set through Vello/custom parity paths; additional modes remain an explicit diagnostic subcase |
| `CMP-03` | Background blend mode | Diagnostic | Typed background-blend boundary |
| `CMP-04` | Isolation group | Supported | Transparent Vello/pass-plan group |
| `CMP-05` | Porter-Duff/composite ops | Diagnostic | Typed composite-operation boundary |
| `XFM-01` | 2D affine transform | Supported | Vello direct or explicit capture mapping |
| `XFM-02` | Transform origin | Supported | `Normalize` |
| `XFM-03` | Skew | Supported | Vello direct or explicit capture mapping |
| `XFM-04` | 3D transform flattening | Diagnostic | Typed 3D-transform boundary |
| `XFM-05` | Coordinate-space tagging | Supported | Explicit local/viewport/surface normalization |
| `TXT-01` | Glyph fill paint | Supported | Vello direct/capture with explicit effect bounds |
| `TXT-02` | Text decoration paint | Supported | Solid normalized bounded Vello stroke geometry; other decoration styles remain an explicit diagnostic subcase |
| `TXT-03` | Text shadow | Diagnostic | Typed text-shadow boundary |
| `TXT-04` | Selection paint bucket | Root | Root materializes ordinary commands |
| `TXT-05` | Generated content paint bucket | Root | Root materializes ordinary commands |
| `RES-01` | Resolved image handle | Supported | Root -> image identity -> Vello/custom resources |
| `RES-02` | Intrinsic image metadata | Supported | Root -> normalization |
| `RES-03` | Image orientation/color profile | Root | Root resolves before render |
| `RES-04` | Animated image frame | Root | Runtime/root selects frame |
| `DIA-01` | Unsupported primitive diagnostics | Supported | Typed normalization/planning diagnostic |
| `DIA-02` | Backend capability matrix | Supported | Semantic plus per-surface runtime reports; native WebCanvas remains a target diagnostic subcase |
| `DIA-03` | Unresolved resource diagnostics | Supported | Typed pre-execution diagnostic |
| `DIA-04` | Invalid value diagnostics | Supported | Typed construction/normalization failure |
| `DIA-05` | Degraded-quality diagnostics | Supported | Typed selected GPU precision/quality evidence |
| `BEP-01` | Vello scene encoder | Supported | One-pass direct and bounded capture spans |
| `BEP-02` | Texture cache/upload | Supported | Vello image cache plus persistent per-device resources |
| `BEP-03` | Offscreen layer renderer | Supported | Bounded Vello capture backend primitive; broad offscreen-layer capability remains a diagnostic subcase |
| `BEP-04` | Fullscreen/rect shader pass | Supported | Bounded source-bound image/composite/present pipelines |
| `BEP-05` | Separable blur pass | Supported | High/reduced precision GPU image passes |
| `BEP-06` | Mask compositor | Diagnostic | Narrow resolved-alpha pass only; broad capability false |
| `BEP-07` | Backdrop compositor | Supported | Completed-parent copy/filter/group composite |
| `BEP-08` | CPU reference path | OracleOnly | Private `cfg(test)` CPU oracle only |

### S30C Typed Diagnostic Subcase Inventory

S30B row totals describe primary primitive disposition; they do not erase a
typed unsupported sub-boundary inside a supported or root-owned row. The stable
subcase keys below are part of final reconciliation. Every listed public
`PrimitiveOperation` is probed with its owning `PrimitiveFamily` and must map to
exactly one key here; resource/degraded diagnostics use their named payload
kind. No prose-only `LayerFilter` or unnamed subset remains.

| Parent row(s) | Stable subcase key | Exact typed result |
| --- | --- | --- |
| `PNT-02` | `PNT-02.UnresolvedSymbolicColor` | `PrimitiveOperation::UnresolvedSymbolicColor` |
| `PNT-03` | `PNT-03.ColorMixFunction` | `PrimitiveOperation::ColorMixFunction` |
| `PNT-03` | `PNT-03.UnsupportedColorSpace` | `PrimitiveOperation::UnsupportedColorSpace` |
| `PNT-03`, `DIA-05` | `PNT-03.UnsupportedPaintSpaceConversion` | `DegradedQualityKind::UnsupportedPaintSpaceConversion` |
| `PNT-07` | `PNT-07.RepeatingGradient` | `PrimitiveOperation::RepeatingGradient` |
| `PNT-09` | `PNT-09.FilteredImagePaint` | `PrimitiveOperation::FilteredImagePaint` |
| `PNT-09` | `PNT-09.ColorFilteredImagePaint` | `PrimitiveOperation::ColorFilteredImagePaint` |
| `GEO-06` | `GEO-06.InsideOutsidePathStrokeAlignment` | `PrimitiveOperation::InsideOutsidePathStrokeAlignment` |
| `GEO-07` | `GEO-07.GeometryBooleanOperation` | `PrimitiveOperation::GeometryBooleanOperation` |
| `GEO-07` | `GEO-07.GeometryOffsetOperation` | `PrimitiveOperation::GeometryOffsetOperation` |
| `IMG-05` | `IMG-05.BackgroundRepeatRound` | `PrimitiveOperation::BackgroundRepeatRound` |
| `IMG-06` | `IMG-06.BackgroundRepeatSpace` | `PrimitiveOperation::BackgroundRepeatSpace` |
| `RES-03` | `RES-03.ImageOrientationConversion` | `PrimitiveOperation::ImageOrientationConversion` |
| `RES-03` | `RES-03.ImageColorProfileConversion` | `PrimitiveOperation::ImageColorProfileConversion` |
| `SHD-01` | `SHD-01.EllipsePathShadowShape` | `PrimitiveOperation::EllipsePathShadowShape` |
| `SHD-02` | `SHD-02.InsetBoxShadow` | `PrimitiveOperation::InsetBoxShadow` |
| `SHD-05`, `TXT-03` | `TXT-03.TextShadow` | `PrimitiveOperation::TextShadow` |
| `SHD-06` | `SHD-06.NonSolidShadowPaint` | `PrimitiveOperation::NonSolidShadowPaint` |
| `FLT-01` | `FLT-01.LayerFilter` | `PrimitiveOperation::LayerFilter` |
| `FLT-01` | `FLT-01.LayerFilterExecution` | `PrimitiveOperation::LayerFilterExecution` |
| `FLT-11`, `DIA-03` | `FLT-11.FilterResource` | `UnresolvedResourceKind::Filter` |
| `BDP-03` | `BDP-03.BroadBackdropExecution` | `PrimitiveOperation::BroadBackdropExecution` |
| `BDP-03` | `BDP-03.BackdropIsolationComposition` | `PrimitiveOperation::BackdropIsolationComposition` |
| `BDP-04` | `BDP-04.RootBackdropPolicy` | `PrimitiveOperation::RootBackdropPolicy` |
| `MSK-03` | `MSK-03.ClipReferenceExecution` | `PrimitiveOperation::ClipReferenceExecution` |
| `MSK-03`, `DIA-03` | `MSK-03.ClipResource` | `UnresolvedResourceKind::Clip` |
| `BEP-06` | `BEP-06.LayerMask` | `PrimitiveOperation::LayerMask` |
| `BEP-06` | `BEP-06.AlphaMaskSourceExecution` | `PrimitiveOperation::AlphaMaskSourceExecution` |
| `BEP-06` | `BEP-06.MaskExecution` | `PrimitiveOperation::MaskExecution` |
| `MSK-06` | `MSK-06.LuminanceMaskMode` | `PrimitiveOperation::LuminanceMaskMode` |
| `MSK-07` | `MSK-07.MultiLayerMaskComposition` | `PrimitiveOperation::MultiLayerMaskComposition` |
| `MSK-08` | `MSK-08.MaskCompositeMode` | `PrimitiveOperation::MaskCompositeMode` |
| `BOX-06` | `BOX-06.BorderGrooveStyle` | `PrimitiveOperation::BorderGrooveStyle` |
| `BOX-06` | `BOX-06.BorderRidgeStyle` | `PrimitiveOperation::BorderRidgeStyle` |
| `BOX-06` | `BOX-06.BorderInsetStyle` | `PrimitiveOperation::BorderInsetStyle` |
| `BOX-06` | `BOX-06.BorderOutsetStyle` | `PrimitiveOperation::BorderOutsetStyle` |
| `BOX-09` | `BOX-09.OutlineDoubleStyle` | `PrimitiveOperation::OutlineDoubleStyle` |
| `BOX-09` | `BOX-09.OutlineAutoStyle` | `PrimitiveOperation::OutlineAutoStyle` |
| `TXT-02` | `TXT-02.TextDecorationStyle` | `PrimitiveOperation::TextDecorationStyle` |
| `CMP-02` | `CMP-02.AdditionalMixBlendMode` | `PrimitiveOperation::AdditionalMixBlendMode` |
| `CMP-03` | `CMP-03.BackgroundBlendMode` | `PrimitiveOperation::BackgroundBlendMode` |
| `CMP-05` | `CMP-05.PorterDuffCompositeMode` | `PrimitiveOperation::PorterDuffCompositeMode` |
| `XFM-04` | `XFM-04.Matrix3dTransform` | `PrimitiveOperation::Matrix3dTransform` |
| `XFM-04` | `XFM-04.PerspectiveTransform` | `PrimitiveOperation::PerspectiveTransform` |
| `XFM-04` | `XFM-04.Rotate3dTransform` | `PrimitiveOperation::Rotate3dTransform` |
| `XFM-04` | `XFM-04.TranslateZTransform` | `PrimitiveOperation::TranslateZTransform` |
| `XFM-04` | `XFM-04.ScaleZTransform` | `PrimitiveOperation::ScaleZTransform` |
| `BEP-03` | `BEP-03.OffscreenLayerRendering` | `PrimitiveOperation::OffscreenLayerRendering` |
| `DIA-02` | `DIA-02.WebCanvasSurface.native` | native `PrimitiveOperation::WebCanvasSurface`; wasm support remains target-specific |
| `DIA-03` | `DIA-03.ImageResource` | `UnresolvedResourceKind::Image` |
| `DIA-03` | `DIA-03.MaskResource` | `UnresolvedResourceKind::Mask` |
| `DIA-03` | `DIA-03.TextRunInkBounds` | `UnresolvedResourceKind::TextRunInkBounds` |
| `DIA-05` | `DIA-05.ReducedIntermediatePrecision` | `DegradedQualityKind::ReducedIntermediatePrecision` observation surface |

The final totals are exactly 69 `Supported`, 24 `Diagnostic`, 6 `Root`, and 2
`OracleOnly`, totaling 101 unique IDs. There is no `FutureRender` row after
`BEP-05` is enabled.

Three test-owned inventories make reconciliation deterministic without making a
planning document a build input:

- `final_primitive_inventory_has_101_unique_capability_consistent_rows` defines
  the same 101 IDs/dispositions in `src/tests.rs`, asserts the exact totals,
  and invokes each row's typed capability/construction probe. Supported probes
  must succeed, Diagnostic probes must return the exact typed operation,
  Root rows must expose no production render operation, and OracleOnly rows
  must be absent from production capabilities/modules.
- `final_property_inventory_maps_22_surfaces_to_known_primitive_ids` defines the
  S30A mapping, asserts exactly 22 unique property surfaces, accepts only an
  exact S30B ID or S30C stable subcase key with a known parent, rejects unknown
  or duplicate references, and proves no property is blanket-supported while
  one of its mapped rows/subcases is Diagnostic or Root.
- `final_diagnostic_subcase_inventory_maps_every_typed_boundary_once` defines
  every S30C stable key, rejects an unknown parent ID or duplicate key, expands
  grouped rows to individual public operations/payload kinds, and proves every
  final false semantic capability operation appears exactly once. It separately
  asserts native `WebCanvasSurface`, all five unresolved-resource kinds, and
  both retained degraded-quality kinds.

These inventory tests are ordinary crate tests using existing dependencies;
they add no parser, generator, script, or machine-readable production artifact.
Row-specific GPU quality tests remain required in addition to the deterministic
inventory probes.

Mechanical reconciliation uses S30B and S30A as the final desired-state
inventory. The legacy overlay/ledger are provenance evidence only. Capability
names may change exactly as specified, but implementation may not silently drop,
add, or reclassify an inventory row.

## S31 Behavior And Failure Matrix

| Input/runtime condition | Route/result |
| --- | --- |
| Empty or Vello-only scene | One transaction-owned internal Vello raster pass with base color and one frame submission stage |
| Internal Vello preparation/encoding failure | Owning render-stage error; raster/effect leases abort and no frame state publishes |
| Canceled encoded Vello pass before submission | Uncertain transient resources drop/quarantine; no idle reuse or publication |
| Any nonzero render on contract-only surface | `RuntimeCapabilityUnavailable(SurfaceRendering, AdapterUnavailable)`; last-successful stats unchanged |
| Any read on nonzero contract-only surface | `RuntimeCapabilityUnavailable(SurfaceReadback, AdapterUnavailable)` |
| Foreign/stale surface passed to renderer | Exact operation plus `SurfaceIdentityMismatch`; no device-slot access |
| Zero-size available headless render/read | Render diagnoses `NonRenderable`; read returns validated empty RGBA buffer without GPU work |
| Nonzero headless read before first publication/after size-changing resize | `SurfaceReadback` + `Uninitialized`; no map or staging allocation |
| Failed/canceled headless render with prior publication | Old readable texture/stats/parameters remain published |
| Failed/canceled headless render without publication | Resource remains `PendingAllocation`; read remains `Uninitialized` |
| `Bgra8` headless creation | Preserved `SurfaceCreateFailed` format rejection |
| Layer with resolved alpha mask | `GpuGraph`, image alpha sampled in layer-local bounds |
| Nested resolved alpha masks | `GpuGraph`, inner-to-outer filter/clip/mask/opacity/blend order |
| Supported bounded backdrop | `GpuGraph`, completed-parent capture then filtered group |
| Color, blur, or drop shadow inside supported backdrop list | Ordered image passes on selected GPU working format |
| High-precision format available | `EffectPrecision::High` |
| High unavailable, reduced available, reduced allowed | Success with `EffectPrecision::Reduced` |
| High unavailable, reduced disallowed | Typed runtime capability error |
| Neither effect format available | Typed runtime capability error |
| Effect extent exceeds device dimension | Typed dimension error before allocation |
| Malformed font bytes | S10A `InvalidInput` at `FontData::try_from_bytes`; no normalization/raster/WGPU work |
| Valid font with out-of-range collection index | Same exact S10A `InvalidInput`; no normalization/raster/WGPU work |
| Readable font container with malformed selected outline/color/palette/bitmap/PNG table | Same exact S10A `InvalidInput` during fallible scene encoding; no raster recording or WGPU work |
| Missing selected glyph ID | S10A `text_glyph.id` `InvalidInput` before external encoding; no empty-glyph substitution |
| Valid but unsupported glyph image encoding | Owning internal-raster preparation `RenderFailed`; no silent glyph omission or fallback font |
| Missing text ink bounds in direct scene | Direct Vello success |
| Missing text ink bounds in bounded graph subtree | Typed unresolved text-bounds error |
| Degenerate transformed effect subtree | Explicit empty/no-op result |
| Suspended/non-renderable/occluded/lost surface | Exact S13 typed runtime lifecycle failure; no frame-state update |
| Device-loss signal | Terminal S25 lost-device transition and exact S13 `DeviceLost` diagnostic |
| Uncaptured device error | Terminal S13A/S25 fault transition and exact `DeviceFaulted` runtime report |
| One lost device among several | Only its surfaces/default headless path fail; ready device slots continue |
| Authored shape/luminance/multi-layer mask | Existing typed unsupported diagnostic |
| Broad layer filter or filtered image paint | Existing typed unsupported diagnostic |
| Root/nested/transformed/repeated backdrop | Existing typed unsupported diagnostic |
| WGPU allocation/submission failure | Backend error, leases cleaned, no CPU retry |
| Explicit native `read_headless` | Callback plus bounded helper-thread `Device::poll(Wait)` progress; mapped staging cleanup on every exit |
| Explicit wasm `read_headless` | Browser-event-loop map callback; no `Device::poll` |
| Duplicate suspend/compatible resume | Idempotent success with resource/publication retention |

## S32 Named Model And Planner Tests

Focused non-GPU tests cover these named contracts:

- `final_primitive_inventory_has_101_unique_capability_consistent_rows`;
- `final_property_inventory_maps_22_surfaces_to_known_primitive_ids`;
- `final_diagnostic_subcase_inventory_maps_every_typed_boundary_once`;
- `options_default_requires_high_precision_and_bounds_retention`;
- `resource_cache_budget_zero_disables_idle_retention`;
- `image_buffer_rejects_short_long_and_overflowing_byte_lengths`;
- `font_data_rejects_malformed_bytes_before_raster_lowering`;
- `font_data_rejects_out_of_range_collection_index_before_raster_lowering`;
- `font_data_constructor_never_panics_for_arbitrary_bytes_and_indices`;
- `font_lowering_rejects_malformed_lazy_tables_without_panic_or_gpu_work`;
- `selected_glyph_preflight_rejects_missing_outline_before_external_encoding`;
- `selected_glyph_preflight_validates_exact_outline_draw_settings`;
- `selected_glyph_preflight_validates_colr_palette_bitmap_and_png_inputs`;
- `selected_glyph_preflight_distinguishes_unsupported_image_from_malformed_data`;
- `external_glyph_resolver_omission_branches_are_blocked_by_preflight`;
- `unsupported_glyph_image_encoding_returns_render_failed_without_omission`;
- `ahem_font_data_validates_at_collection_index_zero`;
- `internal_vello_font_parsing_is_fallible_and_never_unwraps`;
- `resolved_alpha_mask_requires_finite_positive_local_bounds`;
- `text_run_bounds_distinguish_unspecified_empty_and_ink`;
- `capabilities_current_report_semantics_without_backend_or_cpu_names`;
- `affected_capability_queries_map_one_to_one_to_primitive_operations`;
- `runtime_capability_report_keeps_precision_flags_independent`;
- `precision_resolver_covers_both_high_only_reduced_only_and_neither`;
- `runtime_errors_distinguish_semantic_unsupported_from_device_unavailable`;
- `runtime_diagnostic_constructor_rejects_every_unlisted_operation_reason_pair`;
- `typed_error_codes_cannot_exist_without_their_matching_payload`;
- `native_and_wasm_error_source_storage_preserves_source_contract`;
- `gpu_error_classification_table_maps_injected_validation_oom_internal_and_stage`;
- `dropped_gpu_operation_future_aborts_draft_state_and_leases`;
- `internal_vello_provenance_names_exact_package_checksum_source_file_hashes_and_adaptations`;
- `prepared_vello_pass_contains_no_wgpu_resource_or_submission_authority`;
- `encoded_vello_pass_requires_transaction_submission_and_explicit_lease_commit`;
- `canceled_vello_pass_drops_uncertain_resources_and_marks_atlas_dirty`;
- `direct_vello_is_the_least_powerful_plan_for_effect_free_scenes`;
- `gpu_graph_is_selected_only_for_supported_custom_requirements`;
- `graph_builder_rejects_forward_stale_and_read_write_aliases`;
- `drop_shadow_source_fanout_lives_through_both_consumers`;
- `graph_base_color_is_initialized_once_and_isolation_is_transparent`;
- `maximal_vello_spans_preserve_authored_command_order`;
- `backdrop_plan_depends_on_current_parent_not_cloned_commands`;
- `signed_device_bounds_floor_minima_and_ceil_maxima`;
- `negative_and_fractional_origins_preserve_texel_center_mapping`;
- `largest_singular_value_raster_scale_preserves_local_effect_space`;
- `zero_singular_value_produces_an_empty_plan`;
- `filter_bounds_fold_blur_and_signed_drop_shadow_outsets_in_order`;
- `color_filter_fusion_preserves_each_source_clamp`;
- `color_filter_known_vectors_use_spec_constants_not_oracle_constants`;
- `filter_scalar_lowering_handles_f32_f64_exponents_and_huge_angles_finitely`;
- `drop_shadow_model_cannot_express_inset_spread_or_non_solid_paint`;
- `resource_leases_reject_stale_generation_and_double_release_by_model`;
- `resource_trim_order_is_last_used_then_resource_identity`;
- `failed_frame_returns_all_leases_and_preserves_last_successful_stats`;
- `device_loss_is_terminal_idempotent_and_releases_device_resources`;
- `surface_loss_can_resume_but_device_loss_requires_a_new_renderer`;
- `surface_resize_suspend_resume_and_two_surfaces_own_resources`;
- `foreign_and_stale_surfaces_fail_before_device_slot_access`;
- `surface_operation_matrix_covers_every_kind_state_and_duplicate_transition`;
- `headless_draft_publication_preserves_pixels_across_failed_and_canceled_frames`;
- `readback_state_machine_cleans_map_pending_mapped_failed_and_canceled_buffers`;
- `zero_size_headless_render_diagnoses_and_read_returns_empty`;
- `nonzero_headless_read_before_publication_reports_uninitialized_without_map`;
- `headless_bgra8_remains_a_surface_create_diagnostic`;
- `terminal_default_device_rejects_headless_without_disabling_ready_slots`.

Model tests use public construction paths for public invariants. Private graph
tests inspect private phase values without exposing a test-only production API.

## S33 Named GPU Quality Tests

Required real-backend tests render through production pipelines and fail when
the required host adapter is unavailable; they do not turn missing execution
into a passing skip. Contract-only adapter behavior has separate deterministic
tests.

The GPU suite includes:

- `real_gpu_error_scope_captures_deliberate_validation_error`;
- `real_gpu_smoke_emits_no_uncaptured_error`;
- `uncaptured_gpu_error_faults_only_its_device_generation`;
- `internal_vello_checked_shader_creation_reports_validation_without_unsafe`;
- `internal_vello_encoding_shares_the_frame_transaction_submission`;
- `internal_vello_direct_pixels_match_pinned_vello_characterization_cases`;
- `direct_vello_scene_uses_one_pass_and_no_effect_allocation`;
- `capture_canonicalize_present_round_trips_transparent_partial_and_opaque_pixels`;
- `reduced_precision_low_alpha_pixels_use_alpha_and_premul8_tolerances`;
- `high_precision_low_alpha_pixels_preserve_straight_rgb`;
- `solid_shape_direct_and_graph_routes_match_on_interior_and_aa_edges`;
- `ahem_glyph_direct_and_graph_routes_share_ink_extent_and_capture_grid`;
- `direct_graph_parity_covers_every_antialiasing_and_scale_pair`;
- `high_precision_color_functions_match_cpu_oracle_for_boundary_pixels`;
- `reduced_precision_color_functions_match_cpu_oracle_with_declared_tolerance`;
- `filter_function_order_changes_output_and_matches_ordered_oracle`;
- `blur_impulse_is_symmetric_normalized_and_matches_oracle`;
- `ordinary_blur_samples_transparent_black_at_all_edges`;
- `backdrop_blur_mirrors_at_semantic_bounds_not_allocation_padding`;
- `drop_shadow_preserves_source_uses_fractional_offset_and_expands_signed_bounds`;
- `resolved_alpha_mask_preserves_partial_alpha_and_nested_order`;
- `resolved_alpha_mask_low_medium_high_and_extend_modes_match_boundary_oracle`;
- `outer_clip_precedes_mask_and_opacity_but_follows_filter`;
- `all_supported_blends_match_oracle_over_transparent_and_opaque_bases`;
- `plus_blend_clamps_high_precision_results`;
- `backdrop_reads_only_completed_prior_content_and_base_once`;
- `backdrop_foreground_is_not_filtered_and_composites_above_backdrop`;
- `later_siblings_observe_completed_backdrop_group`;
- `nonuniform_scale_and_skew_preserve_local_blur_shape`;
- `negative_bounds_and_subpixel_transforms_do_not_shift_capture`;
- `repeated_frames_reuse_resources_without_growth_or_readback`;
- `budget_zero_releases_idle_resources_without_changing_pixels`;
- `resize_suspend_resume_and_two_surfaces_keep_device_resources_coherent`;
- `destroyed_device_callback_reports_terminal_loss_without_stale_resource_use`;
- `native_readback_callback_progresses_and_cleans_up_with_diagnostic_deadline`;
- `canceled_native_readback_discards_late_callback_without_publication_change`;
- `render_window_smoke_executes_direct_and_graph_presented_frames`;
- `render_path_submits_without_map_or_cpu_wait`.

Before removing the external `vello` dependency, the internalization cycle
records expected pixels for the named characterization cases by rendering the
same authored scenes through pinned Vello 0.9 and the current production
surface route. The cases cover solid fill and stroke edges, gradients, image
sampling, clipping, transforms, and Ahem glyphs at the S33 scale and
antialiasing pairs. Expected values are retained as explicit test constants or
small source-readable tables, not a generated binary fixture. The replacement
engine must satisfy those same cases and S34 tolerances before the external
dependency is removed.

Real-backend validation is intentionally separate from deterministic error
classification. The real suite safely induces only a validation error inside
an owned scope and uses `Device::destroy` for loss. It does not exhaust memory
or attempt to provoke a driver-internal fault. A private `cfg(test)`
classification seam feeds `Validation`, `OutOfMemory`, and `Internal` records
plus each owning operation stage through the same production mapping/commit
logic without allocating or submitting; S32 proves every table cell and that
the seam is absent from production builds.

Direct/graph parity executes the Cartesian product
`{Area, Msaa8, Msaa16} x {1.0, 1.25, 2.0}` for the solid-shape edge fixture and
the Ahem glyph fixture. Each case asserts the requested antialiasing mode in the
private normalized plan, the expected public route/pass stats, the capture
grid, and S34 interior/edge tolerances. Precision
selection uses private immutable capability fixtures for `{high,reduced}` =
`{true,true}`, `{true,false}`, `{false,true}`, and `{false,false}` under both
quality policies. Real high/reduced format tests invoke the same production
executor and shaders with an explicitly test-resolved supported working format;
they do not use a substitute shader or CPU fake.

Every callback/loss/readback test creates a fresh condition cell and waits for
its own state transition, never for elapsed sleep. Native waits use a five-second
test-only diagnostic deadline and print the operation generation, readback
state, submission index, and device signal on expiry. The deadline diagnoses a
stalled test; it is not production timeout semantics. Wasm remains compile-gated
at leaf scope and browser callback execution remains the S36 root-owned host
evidence.

## S34 Quality Tolerances

Expected pixels come from exact authored constants or the private oracle. All
comparisons include alpha and verify premultiplied invariants before final
unpremultiplication.

| Test class | High precision straight RGBA8 | Reduced precision alpha/premul8 |
| --- | --- | --- |
| Canonicalize/present solid and alpha round trip | maximum 2 levels/channel | alpha error at most 1; each premul8 color error at most 1 |
| Color functions and normal composition | maximum 2 levels/channel | alpha error at most 2; each premul8 color error at most 2 |
| Supported blend modes | maximum 3 levels/channel | alpha error at most 3; each premul8 color error at most 3 |
| Gaussian/drop-shadow filtered pixels | maximum 4 levels/channel and total-alpha energy within 1.5% | alpha and premul8 errors at most 4; total-alpha energy within 2.5% |
| Direct-versus-graph Vello interiors | maximum 2 levels/channel | alpha and premul8 errors at most 2 |
| Direct-versus-graph antialiased boundary pixels | maximum 4 levels/channel, identical nonzero support within one pixel | alpha and premul8 errors at most 4, identical nonzero support within one pixel |
| Transform/capture placement | alpha-weighted centroid delta at most 0.25 device pixel per axis | at most 0.35 device pixel per axis |

Tolerance is applied only to corresponding pixels after exact dimensions and
origin are checked. A test cannot pass by shifting, cropping, ignoring
transparent-edge leakage, or comparing only a broad average. Identity and
fully transparent invariants remain byte-exact where quantization cannot affect
them.

The canonicalization suite includes straight input `[128, 0, 0, 1]`, all RGB
extremes at alpha `0`, `1`, `2`, `15`, `16`, `127`, `254`, and `255`, and
partial-alpha filter/composite outputs. It proves the reduced route against the
alpha/premul8 metric and separately proves that the high route retains stable
straight RGB at low nonzero alpha.

## S35 Static Production-Path Guards

Source/diff tests and final inspection prove:

- `Options` and `src/vello_engine/` contain no CPU selector, `use_cpu`, CPU
  shader/materialized-buffer path, debug download, blocking helper, profiler,
  hot reload, or deprecated async renderer;
- `Cargo.toml` has no external `vello` dependency, and production source has no
  `vello::` path; only pinned `vello_encoding` and WGSL-only `vello_shaders`
  remain external;
- `src/vello_engine/` contains no `queue.submit`, buffer map, `Device::poll`,
  surface/device convenience owner, or unchecked/trusted shader creation;
- public/internal font parsing contains no `unwrap`, `expect`, fallback font, or
  silent malformed-font omission, and every constructed `FontData` passed to
  raster lowering satisfies S10A;
- every external `vello_encoding` glyph run is built from a
  `ValidatedGlyphRun`; no direct unpreflighted glyph-run append path exists;
- `NOTICE-VELLO.md` names the exact Vello 0.9 package checksum, imported and
  omitted files, pre-adaptation per-file hashes, adaptations, and preserved
  license texts;
- production modules do not import `reference`;
- `Renderer::render` has no call path to the readback state machine, buffer
  mapping, `Device::poll`, CPU pixel execution, rendered `Image::from_rgba`, or
  internal-raster atlas registration for effect-graph re-entry;
- production `Device::poll` appears only in the native explicit-readback helper,
  and private injected GPU error classification is compiled only under
  `#[cfg(test)]`;
- `RenderBackdropCapture` has no cloned `source_commands`;
- one persistent resource manager exists per device rather than per effect;
- every shader/pipeline source is tracked and included from an owning module;
- all tracked and non-ignored Surgeist-owned Rust source is free of `unsafe`,
  unsafe attributes, extern blocks, and unsafe lint allowances.

These guards supplement behavior tests; string scanning alone is not evidence
that pixels or lifecycle semantics are correct.

## S36 Feature, Dependency, And Artifact Contract

The reviewed normal dependency set is exactly:

- existing `kurbo = 0.13.1`, `peniko = 0.6.1`, optional path/version
  `surgeist-window = 0.1.0`, and `wgpu = 29.0.3`;
- already-present `bytemuck = 1.25.0` with no feature requested by this crate,
  used only for safe casts of pinned external POD types; Cargo feature
  unification still enables `bytemuck/derive` through pinned
  `vello_encoding`, but Surgeist-owned source does not use its derive macros;
- already-present `log = 0.4.33`, `png = 0.18.1`, and `skrifa = 0.42.1` with
  default features disabled and `autohint_shaping,std` enabled, preserving the
  copied scene/glyph behavior;
- `vello_encoding = 0.9.0` and `vello_shaders = 0.9.0`, with
  `vello_shaders` default features disabled and only `wgsl` enabled.

The external `vello` dependency is removed. Its `futures-intrusive`,
`static_assertions`, and `thiserror` direct uses are removed with deprecated
async/debug helpers, upstream-only assertions, and the external error wrapper;
they do not become direct dependencies. The currently unused direct
`glifo = 0.1.1` dependency is also removed. `pollster = 0.4.0` moves from normal
to dev dependencies because only tests and the tracked native smoke example
drive the async public API synchronously. The other dev dependency remains
`proptest = 1.11.0` with default features disabled and `std` enabled. Its role
remains property/model testing; it is never a production pixel engine. Every
retained direct dependency must have a production, test, or example source use
and appear once at its intended Cargo role. No image-filter, compute, wasm
harness, or other dependency is added under the current permission envelope.

The feature/target support matrix is closed:

| Target | Features | Supported surface/evidence contract |
| --- | --- | --- |
| Native | none | Supported: real headless direct and graph execution plus default compile/test/lint |
| Native | `render-web` | Supported: same real headless execution; native `WebCanvas` remains the existing typed platform diagnostic; compile/test/lint with feature |
| Native | `render-window` | Supported: headless tests plus real presented direct and graph smoke through a live `surgeist-window::Handle` |
| Native | `render-window,render-web` | Supported additive combination: same native presented/headless behavior and web-platform diagnostic; compile/test/lint and presented smoke |
| `wasm32-unknown-unknown` | `render-web` | Supported leaf build: WebCanvas adapter/presentation code must compile; real browser execution is root-host integration evidence described below |
| `wasm32-unknown-unknown` | none | Not a supported production combination; no WebGPU feature contract |
| `wasm32-unknown-unknown` | `render-window` or both | Not supported; native-window integration is not a wasm contract |

Native presented execution is supplied by a tracked
`examples/render_window_smoke.rs`. It composes the public
`surgeist-window` lifecycle without moving lifecycle ownership into production
render code, creates a live handle, renders one direct scene and one graph scene,
asserts the reported routes, presents both, and exits deterministically. An
unavailable graphical session is a required-host blocker, not a passing skip.

Browser WebGPU execution needs a browser/canvas event-loop host and is therefore
root-owned integration, not a leaf-owned window/runtime harness. The leaf
handoff must state that wasm compilation passed but browser execution remains
unverified until root runs a real canvas direct frame and graph frame and
observes successful presentation/routes. No leaf support claim is inferred from
host compilation alone. If a future leaf cycle is explicitly expanded to own a
browser harness, the exact target, compatible wasm binding tool, target-specific
dependencies, and harness artifact require user permission and a reviewed
specification revision before acquisition or edits.

The crate has no declared local MSRV, API generator, snapshot generator, or CI
workflow. At root HEAD `19590f6d9fa01c0df197c5ef07fb626c5cf18ced`, the
authoritative facade manifest declares `rust-version = "1.97"`; therefore Rust
1.97 is the root-integration compatibility floor for every leaf API and
implementation choice even though this leaf manifest does not duplicate that
field. Rust 2024 compatibility is also preserved. At specification review time
the installed active/stable toolchain is exactly Rust 1.97.0, alongside
nightly-2025-08-02 and 1.92.0. No API stabilized after 1.97 may enter production
without an equivalent 1.97-compatible form.

Custom WGSL files are tracked implementation artifacts, not generated output.
`vello_shaders` remains an external crate whose existing build script produces
its embedded `SHADERS`; no generated shader Rust is copied or hand-edited.
`NOTICE-VELLO.md` plus
`LICENSES/Vello-0.9.0-APACHE-2.0.txt` and
`LICENSES/Vello-0.9.0-MIT.txt` are tracked provenance/license artifacts for the
internalized main-crate source. Source headers remain intact.
The existing Ahem font and its license/provenance files remain the only fixture
needed by this initiative. No binary fixture is downloaded or modified.

Leaf documentation changes are required. `README.md` describes the direct and
GPU-graph routes, GPU-only/no-fallback policy, effect quality policy, runtime
capability query, async GPU-operation model, and native presented smoke command.
`src/lib.rs` crate docs
and every changed public type/method document phase, units, failure, and default
semantics. The native smoke example is both executable target evidence and a
public composition example. There are no other existing leaf examples to
update.

Root facade updates, root API artifact generation, and the submodule pointer are
explicit later handoff work outside this repository.

## S37 Verification Contract

The complete implementation is exercised with already-present artifacts and
offline Cargo resolution:

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
rustc +stable --version # must report Rust 1.97.x
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets
CARGO_NET_OFFLINE=true cargo +stable check -p surgeist-render --all-targets --features render-window,render-web
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e normal --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e dev --depth 1
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i bytemuck
CARGO_NET_OFFLINE=true cargo tree -p surgeist-render -e features -i vello_shaders
```

Default and feature test commands execute the required real headless GPU suite;
adapter absence fails rather than skips. The two example commands are the real
native presented evidence. The wasm command is mandatory target compilation;
host `render-web` checking is not a substitute. Browser presentation is named
as root follow-up rather than falsely counted as leaf execution.

The current toolchain lacks `wasm32-unknown-unknown`. If it remains absent when
the wasm gate is reached, all independent evidence may proceed, but the gate
returns the canonical missing-tooling blocker requesting exact permission for
`rustup target add wasm32-unknown-unknown`; it does not run that acquisition or
treat the check as passed. A missing graphical session similarly blocks the
native presented command without downloading a display/server substitute.

Before the two `+stable` compatibility commands, `rustc +stable --version` must
report exactly Rust 1.97.x. If stable has advanced beyond the 1.97 line when the
gate is reached and no exact 1.97 toolchain is installed, the acquisition
blocker requests permission for `rustup toolchain install 1.97.0 --profile
minimal`; it does not install the toolchain or treat a newer compiler as MSRV
evidence. If an exact toolchain is already present, the equivalent two checks
under that toolchain are mandatory. Any target component needed on it requires
its own exact permission request rather than being folded into this blocker.

The final unsafe-absence gate builds the owned Rust manifest from tracked and
non-ignored untracked `*.rs` files, excludes only dependency/build-cache roots,
and applies the canonical `$surgeist-agent` unsafe scan. The compiler lint and
repository-wide scan must both be clean.

Final dependency/provenance inspection additionally proves that `Cargo.toml`
has no `vello` package dependency, `cargo tree` contains direct
`vello_encoding` and WGSL-only `vello_shaders`, every S36 direct dependency has
an intended source use, `glifo` is absent, `pollster` is dev-only, and no removed
Vello helper became direct. The feature tree may contain `bytemuck/derive` only
through pinned external crates; the owned-source guard proves no
`Pod`/`Zeroable` derive or unsafe implementation. Static source inspection
confirms `src/vello_engine/` has no CPU dispatch, blocking/poll/map, direct
submission, trusted shader API, or Vello utility surface/device owner. The three
S36 provenance/license artifacts exist, their recorded package version/checksum
match the immutable S04 import record, and the notice retains every imported
file's pre-adaptation SHA-256. The removed `vello` lockfile entry is not final
provenance authority.

## S38 Finite Acceptance Criteria

This initiative is accepted only when all of the following are true:

1. Every overlay `Migrate`, `Correct`, and `Enable` row named in S30 has its
   specified production route and executable evidence.
2. Every overlay `Preserve` row remains supported, and every `HoldDiagnostic`
   and `HoldRoot` row remains correctly typed and owned.
3. Effect-free scenes take one transaction-owned internal `DirectVello` raster
   pass with no effect allocation or engine-owned submission.
4. Supported alpha-mask and bounded-backdrop scenes execute through the closed
   GPU graph with no production readback, CPU pixel algorithm, or Vello-atlas
   graph re-entry.
5. High precision is preferred, reduced precision is explicit and observable,
   and missing GPU capability returns a typed runtime failure without CPU
   fallback.
6. Color functions, Gaussian blur, drop shadow, masks, clips, opacity, supported
   blends, transforms, and backdrop groups satisfy S20-S24 and the S34 quality
   tolerances.
7. The persistent per-device resource manager demonstrates reuse, bounded idle
   retention, deterministic cleanup, coherent renderer/surface/device identity,
   draft-versus-published headless atomicity, and cancellation-safe readback.
8. Public models and phase boundaries match S07-S14 and S29; invalid values and
   unresolved/runtime failures are not represented as strings or backend-name
   guesses.
9. CPU reference code is test-only; the internal Vello engine has no CPU mode,
   blocking helper, direct submission, or unchecked shader path; and all
   Surgeist-owned source contains no `unsafe` or unsafe allowance.
10. The dependency, feature, fixture, artifact, and crate/root boundaries in S02,
    S03, S06A, and S36 remain intact, including exact Vello provenance and the
    absence of the external `vello` dependency.
11. Every named applicable model, GPU, lifecycle, readback, guard, feature,
    Rust-1.97 compatibility, and target check is green; unavailable required tooling is
    reported through the canonical permission blocker rather than treated as a
    pass.

Planning, implementation review, landing, publication, and candidate handoff
follow the canonical `$surgeist-agent` workflow and are not redefined here.
