# Render Pipeline Migration Overlay

Date: 2026-07-09

Crate: `surgeist-render`

Status: sequencing input. This document is not an implementation plan and does
not mark any unsupported primitive as implemented.

## Objective

Overlay the current exhaustive CSS/style primitive inventory with the migration
facts needed to replace the staged CPU/offscreen execution path with a
GPU-resident Vello plus WGPU pass architecture without losing intentional
functionality, diagnostics, root ownership boundaries, or rendering quality.

The overlay gives every reconciled primitive/backend row a stable ID, current
status, migration disposition, target route, and required verification gate.
Future sequence and implementation plans must cite these IDs rather than relying
only on prose feature names.

Backwards compatibility shims are not required. Behavioral preservation is
required for supported contracts, except where this overlay explicitly records
a correctness defect that must be fixed instead of preserved.

## Sources Of Truth

Use these sources together:

1. `plans/2026-07-08-render-css-support-matrix.md` defines the exhaustive
   render-owned primitive inventory and intended realization families.
2. `plans/2026-07-10-render-css-matrix-reconciliation.md` defines the current
   status and authoritative code/test evidence for all 101 rows.
3. `plans/2026-07-10-render-root-handoff.md` defines the crate boundary and
   root responsibilities.
4. `guidance/surgeist-rust-modeling-guide.md` defines the modeling standard.
5. Current code and tests define the implementation being migrated, but they do
   not override a known correctness defect in this overlay.
6. The CSS graphical-effect order follows the
   [CSS Masking compositing model](https://www.w3.org/TR/css-masking-1/#intro),
   [Compositing and Blending order](https://www.w3.org/TR/compositing-1/#compositingandblendingorder),
   and [Filter Effects](https://www.w3.org/TR/filter-effects-1/#FilterProperty).
   Filter implementations also follow the Filter Effects
   [primitive-result contract](https://www.w3.org/TR/filter-effects-1/#FilterPrimitiveOverview)
   and [drop-shadow shorthand](https://www.w3.org/TR/filter-effects-1/#dropShadowEquivalent).
   Backdrop planning must account for the still-evolving
   [Filter Effects Level 2 Backdrop Root](https://drafts.csswg.org/filter-effects-2/#backdrop-root)
   model and must not claim behavior where that model is unresolved.

When current tests encode a known defect, the correction register below is
authoritative. A replacement path must prove the intended semantic result; byte
parity with the defective result is not a completion gate.

## Scope

This overlay covers:

- all 101 primitive/backend rows in the reconciliation ledger;
- all 22 CSS/style property cross-reference rows through their mapped primitive
  rows;
- the direct Vello fast path;
- current materialized image, mask, and backdrop execution;
- future WGPU image, filter, mask, and compositing passes;
- pixel, spatial, ordering, resource, capability, and test contracts;
- preservation of typed diagnostics and root-owned boundaries during migration.

This overlay does not:

- implement code;
- select the exact number of future implementation phases;
- make a current diagnostic row supported without its own reviewed plan and
  completion evidence;
- move CSS parsing, cascade, layout, text shaping, URL loading, hit testing,
  generated content, selection ranges, or animation scheduling into render;
- require obsolete public APIs or compatibility adapters to remain.

## Dispositions

| Disposition | Meaning during migration |
| --- | --- |
| `Preserve` | Keep the current intentional contract and realization family. Refactoring is allowed only behind passing preservation gates. |
| `Migrate` | Keep the current supported contract while replacing its production execution path. The prior path remains only until parity is proven. |
| `Correct` | Replace a currently supported but defective execution behavior with the intended contract recorded here. Do not preserve defective pixels. |
| `HoldDiagnostic` | Preserve the typed rejection, capability value, and operation identity until a separate expansion plan proves support. Internal foundation work must not flip the public claim early. |
| `HoldRoot` | Preserve the current root/style/layout/text/runtime ownership boundary. |
| `Enable` | Render-owned foundation or future work intended to become supported through a later implementation plan. Status changes only after its row gates pass. |

## Target Routes

| Route | Responsibility |
| --- | --- |
| `Normalize` | Validate and lower render-ready values into normalized commands or pass intents. |
| `VelloDirect` | Encode ordinary vector, image, text, clip, opacity, or supported blend work directly into a Vello scene. |
| `VelloCapture` | Render a bounded Vello scene span into an explicitly described offscreen resource. |
| `ImagePass` | Run a WGPU compute/render image operation such as conversion, color filtering, blur, or paint-through-mask. |
| `CompositePass` | Apply masks, opacity, blend/composite operations, and pass results into a parent target. |
| `ResourceManager` | Own device resources, pipeline caches, texture pools, budgets, lifetimes, and reuse. |
| `CpuReference` | Provide deterministic semantic oracles and explicit fallback evidence; it is not the default production effect path. |
| `Diagnostic` | Return the existing typed unsupported/unresolved/degraded result. |
| `Root` | Remain outside this crate and arrive only as resolved render input. |

## Required Architecture Contracts

### Phase Boundary

The target flow is:

```text
Scene
  -> normalized render commands
  -> RenderPlan with typed pass/resource dependencies
  -> per-device PassExecutor
  -> surface/headless target
```

`encode_vello_scene` remains a narrow encoder for `VelloDirect` or
`VelloCapture` spans. It must not become the dispatcher for custom CSS effects.
The first screen path for a scene that needs no custom pass remains one direct
Vello scene render with no offscreen allocation.

The pass model remains a closed render-owned model that distinguishes direct
Vello work, capture, image/filter work, masking/compositing, presentation, and
reference/readback work. Mixed frames must preserve authored paint order and
initialize the modeled parent base exactly once; the direct-only fast path must
remain direct. Exact pass decomposition belongs in later implementation plans.

### Pixel Contract

Surface presentation format and intermediate working format are different
semantic domains and must not share one broad `Format` model.

Every pass resource must explicitly name:

- physical extent and device-pixel origin;
- storage format and precision;
- alpha mode (`Straight`, `Premultiplied`, or `Opaque`);
- color encoding/transfer and operation color space;
- texture usage and sample count;
- sampling and edge-extension policy;
- whether the resource is transient, retained, external, or readback-only.

The pinned Vello capture boundary introduces one straight-alpha RGBA8 boundary.
Custom work must canonicalize that capture into a premultiplied working domain,
avoid repeated quantization, and convert to presentation format only at final
output. Runtime format limitations require a typed fallback or diagnostic.

CSS filter functions operate in sRGB and preserve authored order and
per-function clamp semantics even when fused. Ordinary filter blur and backdrop
blur retain distinct edge policies; backdrop mirroring is tied to the clipped,
transformed border-box edge rather than allocation padding. URL/SVG filter
graphs remain diagnostic under `FLT-11`.

### Spatial Contract

Logical bounds alone are not an offscreen allocation contract. Every capture
must carry:

- logical bounds;
- outward-snapped bounds after applying the complete logical-to-device mapping;
- a signed snapped device origin and positive physical extent;
- the semantic coordinate space in which each pixel-moving effect executes;
- mappings between effect, capture, and parent spaces;
- raster-quality policy, filter outsets, clip bounds, and edge policy.

Fractional origins, noninteger surface scales, and transformed content must not
be rasterized on one grid and then accidentally resampled on another. Text-only
and other bounds-unknown groups need explicit root/render bounds or a render
owned bounds computation; they must not silently disappear or use an unbounded
sentinel.

Foreground filters preserve local filter-space semantics under transforms;
backdrop capture preserves its separate screen/device-space contract. The
implementation may choose its rasterization strategy later, but it may not
substitute device-axis behavior that changes the normalized effect.

### Effect And Compositing Order

Foreground groups preserve the normalized order: isolated paint, ordered
filters, clip, mask, opacity, then blend/composite. Isolated captures start
transparent black rather than the final surface base.

Backdrop work depends on completed prior content, applies its ordered filter and
border-box clip, combines the foreground once, and then applies outer effects to
the combined group before parent composition.

Until broader backdrop-root behavior is modeled and tested, nested,
transformed, repeated, and root backdrop cases retain their existing typed
diagnostics.

### Resource And Submission Contract

The production path must:

- keep intermediate effect data GPU-resident;
- maintain typed per-device pipeline/resources across frames with bounded
  lifetime and deterministic lifecycle behavior;
- submit dependent work in queue order without a CPU wait between passes;
- avoid `MAP_READ`, `PollType::wait_indefinitely`, and full-buffer content
  hashing in ordinary production effect execution;
- release or rebuild resources correctly after resize, suspend/resume, device
  loss, or backend recreation.

Vello may submit its own work internally; the plan only requires a frame
executor that preserves dependency order and does not force an intermediate CPU
round trip.

### Capability Contract

Separate semantic backend capabilities from runtime device capabilities:

- semantic capabilities describe supported render operations and diagnostics;
- runtime capabilities describe the actual adapter formats, storage/filtering
  support, limits, sample support, and fallback routes;
- a public semantic capability may become true only when every required runtime
  route has a supported implementation or a typed declared fallback;
- internal pass plumbing does not by itself make a broad CSS primitive
  supported;
- contract-only planning must remain distinguishable from executable device
  support.

## Correction Register

Ranges are closed over the stable IDs in this document. Later plans must update
a correction mapping when a route gains or loses a consumer.

| ID | Affected rows | Required migration correction |
| --- | --- | --- |
| `COR-01` | `BDP-01`, `BDP-02`, `MSK-05`, `CMP-04`, `BEP-01..BEP-04`, `BEP-06`, `BEP-07` | Distinguish transparent isolated captures from backdrop/prior-content captures and initialize the parent base exactly once. |
| `COR-02` | `SHD-04`, `FLT-02..FLT-10`, `FLT-12`, `FLT-13`, `BDP-01`, `BDP-02`, `MSK-05`, `BEP-03..BEP-07` | Replace size-only rounding with signed, outward-snapped device bounds and stable return mapping. |
| `COR-03` | `SHD-04`, `FLT-02..FLT-10`, `FLT-12..FLT-14`, `BDP-01`, `BDP-02`, `MSK-05`, `XFM-01`, `XFM-03`, `XFM-05`, `BEP-03..BEP-08` | Preserve effect-space semantics and raster quality under transforms; correct drop-shadow order, subpixel offsets, and expanded bounds. |
| `COR-04` | `TXT-01`, `MSK-05`, `BEP-03` | Use real glyph ink bounds for bounds-dependent effects rather than requiring incidental clips or relying on advances. |
| `COR-05` | `SHD-04`, `FLT-02..FLT-10`, `FLT-12`, `BDP-01`, `BDP-02`, `MSK-05`, `BEP-02..BEP-07` | Remove synchronous production readback, per-effect cache recreation, CPU materialization, and image reupload from migrated routes. |
| `COR-06` | `DIA-02`, `DIA-05`, `BEP-02..BEP-07` | Separate semantic support claims from runtime adapter capabilities and typed fallback/degradation. |
| `COR-07` | `SHD-04`, `FLT-02..FLT-10`, `FLT-12..FLT-14`, `BDP-01`, `BDP-02`, `MSK-05`, `BEP-03..BEP-08` | Add missing cross-route quality coverage for AA, capture grids, transforms, base colors, and effect parity. |
| `COR-08` | `SHD-04`, `FLT-01`, `FLT-02..FLT-10`, `FLT-12..FLT-14`, `BDP-02`, `BEP-04`, `BEP-05`, `BEP-07`, `BEP-08` | Model CSS filter sRGB/clamp semantics and distinct ordinary versus backdrop edge behavior; keep byte-reference evidence separate from the semantic oracle. |

## Verification Gates

| Gate | Required evidence |
| --- | --- |
| `G1 Contract` | Public/normalized models preserve invariants, command ordering, diagnostics, root ownership, and capability identity. |
| `G2 Direct` | Direct Vello output and no-offscreen allocation remain unchanged for ordinary scenes across supported AA modes and surface scales. |
| `G3 Effect` | GPU effect output matches the applicable semantic oracle or approved tolerance for order, alpha, color space, clamps, and edge policy. |
| `G4 Spatial` | Bounds, signed origins, scale, transforms, outsets, clips, and return mappings preserve effect-space semantics and quality. |
| `G5 Resource` | Repeated frames prove cache reuse, bounded memory, deterministic release/eviction, no production readback, and correct device/surface lifecycle. |
| `G6 Composition` | Nested filter/clip/mask/opacity/blend/backdrop stacks preserve normative order and isolation, including opaque and transparent surface bases. |
| `G7 Targets` | Required native headless, `render-window`, and applicable `render-web`/wasm routes compile and execute; unavailable executable coverage is not a pass. |
| `G8 Reconcile` | Matrix statuses, capabilities, root handoff, tests, and implementation agree after the migration item; a clean-context holistic review is clean. |

`G1` and `G8` apply globally. Later plans add every other applicable gate and
define their concrete commands, fixtures, and tolerances.

## Primitive Overlay

The IDs below are stable within this migration effort. Current status is copied
one-to-one from the reconciliation ledger.

### Paint Sources

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `PNT-01` | Solid RGBA paint | Supported | Preserve | `Normalize -> VelloDirect`; explicit pixel conversion at pass boundaries | `G1`, `G2`, `G6` |
| `PNT-02` | Symbolic color token | DeferredToRoot | HoldRoot | `Root` | `G1`, `G8` |
| `PNT-03` | Paint-space color conversion | Diagnostic | HoldDiagnostic | `Diagnostic`; later typed color conversion route | `G1`, `G8` |
| `PNT-04` | Linear gradient | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `PNT-05` | Radial gradient | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `PNT-06` | Conic/sweep gradient | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `PNT-07` | Repeating gradients | Diagnostic | HoldDiagnostic | `Diagnostic`; later normalization or `ImagePass` | `G1`, `G8` |
| `PNT-08` | Image paint | Supported | Preserve | `Normalize -> VelloDirect`; `ResourceManager` for retained images | `G1`, `G2`, `G5` |
| `PNT-09` | Filtered image paint | Diagnostic | HoldDiagnostic | `Diagnostic`; later `ImagePass` after resolved source | `G1`, `G8` |

### Geometry Targets

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `GEO-01` | Rect fill/stroke | Supported | Preserve | `VelloDirect` | `G1`, `G2`, `G4` |
| `GEO-02` | Rounded rect fill/stroke | Supported | Preserve | `VelloDirect` | `G1`, `G2`, `G4` |
| `GEO-03` | Circle/ellipse fill/stroke | Supported | Preserve | `VelloDirect` | `G1`, `G2`, `G4` |
| `GEO-04` | Arbitrary path fill | Supported | Preserve | `VelloDirect` | `G1`, `G2`, `G4` |
| `GEO-05` | Arbitrary path centered stroke | Supported | Preserve | `VelloDirect` | `G1`, `G2`, `G4` |
| `GEO-06` | Arbitrary path inside/outside stroke | Diagnostic | HoldDiagnostic | `Diagnostic`; later explicit fill geometry | `G1`, `G8` |
| `GEO-07` | Geometry boolean/offset support | Diagnostic | HoldDiagnostic | `Diagnostic`; later render geometry pipeline only with separate plan | `G1`, `G8` |
| `GEO-08` | Hit-test geometry | DeferredToRoot | HoldRoot | `Root` | `G1`, `G8` |

### Image Sampling And CSS Image Layers

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `IMG-01` | Image fit | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `IMG-02` | Background position | Supported | Preserve | `Normalize` | `G1`, `G4` |
| `IMG-03` | Background size | Supported | Preserve | `Normalize` | `G1`, `G4` |
| `IMG-04` | Repeat no-repeat/repeat | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `IMG-05` | Repeat round | Diagnostic | HoldDiagnostic | `Diagnostic`; later normalization expansion | `G1`, `G8` |
| `IMG-06` | Repeat space | Diagnostic | HoldDiagnostic | `Diagnostic`; later normalization expansion | `G1`, `G8` |
| `IMG-07` | Background origin | Supported | Preserve | `Normalize` | `G1`, `G4` |
| `IMG-08` | Background clip | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G6` |
| `IMG-09` | Background attachment | Supported | Preserve | `Normalize -> VelloDirect` with coordinate tags | `G1`, `G4`, `G6` |
| `IMG-10` | Multi-layer image stack | Supported | Preserve | `Normalize -> VelloDirect` in authored order | `G1`, `G2`, `G6` |

### Box Decoration, Borders, And Outlines

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `BOX-01` | Background layer stack | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G6` |
| `BOX-02` | Border side solid | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `BOX-03` | Border style none/hidden | Supported | Preserve | `Normalize` suppression | `G1` |
| `BOX-04` | Border dashed/dotted | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `BOX-05` | Border double | Supported | Preserve | `Normalize -> VelloDirect` bands | `G1`, `G2`, `G4` |
| `BOX-06` | Border groove/ridge/inset/outset | Diagnostic | HoldDiagnostic | `Diagnostic`; later explicit paint-band plan | `G1`, `G8` |
| `BOX-07` | Border radius clipping | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G6` |
| `BOX-08` | Outline solid/dashed/dotted | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `BOX-09` | Outline auto | Diagnostic | HoldDiagnostic | `Diagnostic`; host/theme policy stays outside current render contract | `G1`, `G8` |
| `BOX-10` | Box decoration break | Supported | Preserve | `Normalize -> VelloDirect` fragments | `G1`, `G2`, `G6` |

### Shadows

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `SHD-01` | Outer box shadow | Supported | Preserve | `Normalize -> VelloDirect` for current supported shapes | `G1`, `G2`, `G4`, `G6` |
| `SHD-02` | Inset box shadow | Diagnostic | HoldDiagnostic | `Diagnostic`; later `VelloCapture -> ImagePass -> CompositePass` | `G1`, `G8` |
| `SHD-03` | Multiple shadows | Supported | Preserve | Ordered `VelloDirect` commands | `G1`, `G2`, `G6` |
| `SHD-04` | Drop shadow filter | Supported | Correct | `VelloCapture/Image source -> ImagePass -> CompositePass`; apply `COR-03` and `COR-08` | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `SHD-05` | Text shadow | Diagnostic | HoldDiagnostic | `Diagnostic`; later glyph-alpha `VelloCapture -> ImagePass` | `G1`, `G8` |
| `SHD-06` | Non-solid shadow paint | Diagnostic | HoldDiagnostic | `Diagnostic`; later blurred alpha plus render paint evaluation | `G1`, `G8` |

### Filters

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `FLT-01` | Filter list model | Supported | Preserve | `Normalize` to ordered pass intents with typed sRGB function and edge policies | `G1`, `G6` |
| `FLT-02` | Blur | Supported | Migrate | `VelloCapture/Image source -> ImagePass`; apply `COR-02`, `COR-03`, and `COR-08` | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-03` | Brightness | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-04` | Contrast | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-05` | Grayscale | Supported | Correct | sRGB color `ImagePass`; apply `COR-08` | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-06` | Hue rotate | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-07` | Invert | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-08` | Opacity filter | Supported | Migrate | ordered sRGB/alpha `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-09` | Saturate | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-10` | Sepia | Supported | Migrate | sRGB color `ImagePass`; semantic oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-11` | URL/SVG/reference filter | Diagnostic | HoldDiagnostic | `Diagnostic`; resolved filter graph requires separate model and plan | `G1`, `G8` |
| `FLT-12` | Filter fusion | Supported | Correct | ordered fused `ImagePass`; apply `COR-08` | `G1`, `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-13` | Filter region/outsets | Supported | Correct | `Normalize` explicit source, execution, clip, and device bounds; apply `COR-02` and `COR-03` | `G1`, `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `FLT-14` | Software/reference fallback | Supported | Correct | `CpuReference` semantic oracle and explicit fallback; apply `COR-08` | `G1`, `G3`, `G4`, `G7`, `G8` |

### Backdrop Filters

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `BDP-01` | Backdrop capture | Supported | Correct | `VelloCapture` from completed prior content with explicit backdrop root, signed device mapping, and bounded temporary | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BDP-02` | Backdrop filter chain | Supported | Correct | `VelloCapture -> ImagePass -> CompositePass`; apply `COR-01`, `COR-03`, and `COR-08` | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BDP-03` | Backdrop isolation | Diagnostic | HoldDiagnostic | `Diagnostic` until backdrop-root and nested completion semantics are modeled | `G1`, `G8` |
| `BDP-04` | Root backdrop policy | Diagnostic | HoldDiagnostic | `Diagnostic`; root/host policy remains explicit | `G1`, `G8` |

### Masks And Clips

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `MSK-01` | Shape clip | Supported | Preserve | `VelloDirect` for direct groups; typed clip dependency for effect groups | `G1`, `G2`, `G6` |
| `MSK-02` | Path clip | Supported | Preserve | `VelloDirect` with fill rule; typed clip dependency for effect groups | `G1`, `G2`, `G4`, `G6` |
| `MSK-03` | Clip URL/reference | Diagnostic | HoldDiagnostic | `Diagnostic`; root resolves references to geometry | `G1`, `G8` |
| `MSK-04` | Basic shape clip | Supported | Preserve | `Normalize -> VelloDirect` | `G1`, `G2`, `G4` |
| `MSK-05` | Alpha mask | Supported | Correct | `VelloCapture/Image source -> premultiplied CompositePass`; `CpuReference` oracle | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `MSK-06` | Luminance mask | Diagnostic | HoldDiagnostic | `Diagnostic`; later color-to-alpha `ImagePass` | `G1`, `G8` |
| `MSK-07` | Multi-layer mask | Diagnostic | HoldDiagnostic | `Diagnostic`; later ordered mask-resource graph | `G1`, `G8` |
| `MSK-08` | Mask composite | Diagnostic | HoldDiagnostic | `Diagnostic`; later Porter-Duff `CompositePass` | `G1`, `G8` |

### Blend, Opacity, And Compositing

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `CMP-01` | Layer opacity | Supported | Preserve | `VelloDirect` where possible; ordered `CompositePass` for effect groups | `G1`, `G2`, `G3`, `G6` |
| `CMP-02` | Mix blend mode | Supported | Preserve | Current `VelloDirect` blend set; `CompositePass` only after parity | `G1`, `G2`, `G3`, `G6` |
| `CMP-03` | Background blend mode | Diagnostic | HoldDiagnostic | `Diagnostic`; later ordered background `CompositePass` list | `G1`, `G8` |
| `CMP-04` | Isolation group | Supported | Preserve | Transparent-black group with `VelloDirect` or pass-plan isolation | `G1`, `G2`, `G6` |
| `CMP-05` | Porter-Duff/composite ops | Diagnostic | HoldDiagnostic | `Diagnostic`; later typed `CompositePass` operation set | `G1`, `G8` |

### Transforms And Coordinate Spaces

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `XFM-01` | 2D affine transform | Supported | Preserve | `Normalize -> VelloDirect`; capture raster mapping when effects require it | `G1`, `G2`, `G4`, `G6` |
| `XFM-02` | Transform origin | Supported | Preserve | `Normalize` | `G1`, `G2`, `G4` |
| `XFM-03` | Skew | Supported | Preserve | `Normalize -> VelloDirect`; capture raster mapping when effects require it | `G1`, `G2`, `G4` |
| `XFM-04` | 3D transform flattening | Diagnostic | HoldDiagnostic | `Diagnostic`; root may flatten to supported 2D | `G1`, `G8` |
| `XFM-05` | Coordinate-space tagging | Supported | Preserve | `Normalize` into explicit local/viewport/surface mappings | `G1`, `G4`, `G6` |

### Text Paint Hooks

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `TXT-01` | Glyph fill paint | Supported | Preserve | `VelloDirect`; provide/derive effect bounds under `COR-04` | `G1`, `G2`, `G4` |
| `TXT-02` | Text decoration paint | Supported | Preserve | `Normalize -> VelloDirect` bounded stroke geometry | `G1`, `G2`, `G4` |
| `TXT-03` | Text shadow | Diagnostic | HoldDiagnostic | `Diagnostic`; later glyph-alpha capture path | `G1`, `G8` |
| `TXT-04` | Selection paint bucket | DeferredToRoot | HoldRoot | `Root` materializes ordinary commands | `G1`, `G8` |
| `TXT-05` | Generated content paint bucket | DeferredToRoot | HoldRoot | `Root` materializes ordinary commands | `G1`, `G8` |

### Resource Handles And Images

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `RES-01` | Resolved image handle | Supported | Preserve | `Root -> ResourceManager -> VelloDirect/ImagePass` | `G1`, `G2`, `G5` |
| `RES-02` | Intrinsic image metadata | Supported | Preserve | `Root -> Normalize` | `G1`, `G4` |
| `RES-03` | Image orientation/color profile | DeferredToRoot | HoldRoot | `Root` | `G1`, `G8` |
| `RES-04` | Animated image frame | DeferredToRoot | HoldRoot | `Root` | `G1`, `G8` |

### Diagnostics And Capability Reporting

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `DIA-01` | Unsupported primitive diagnostics | Supported | Preserve | `Diagnostic` at normalization/planning boundary | `G1`, `G8` |
| `DIA-02` | Backend capability matrix | Supported | Migrate | Preserve semantic capabilities and add lifecycle-aware runtime device capabilities under `COR-06` | `G1`, `G5`, `G7`, `G8` |
| `DIA-03` | Unresolved resource diagnostics | Supported | Preserve | `Diagnostic` before execution | `G1`, `G8` |
| `DIA-04` | Invalid value diagnostics | Supported | Preserve | Typed construction/normalization failure | `G1`, `G8` |
| `DIA-05` | Degraded-quality diagnostics | Supported | Preserve | Typed quality/fallback evidence from planner/executor | `G1`, `G3`, `G5`, `G7`, `G8` |

### Backend Pipeline Requirements

| ID | Reconciled row | Current | Disposition | Target route | Gates |
| --- | --- | --- | --- | --- | --- |
| `BEP-01` | Vello scene encoder | Supported | Preserve | `VelloDirect` and bounded `VelloCapture` spans | `G1`, `G2`, `G4`, `G7` |
| `BEP-02` | Texture cache/upload | Supported | Migrate | Preserve direct-path image-cache behavior and add a persistent per-device `ResourceManager` for pass resources | `G1`, `G2`, `G5`, `G7`, `G8` |
| `BEP-03` | Offscreen layer renderer | Diagnostic | Enable | Bounded `VelloCapture`; capability changes only after complete supported contract | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BEP-04` | Fullscreen/rect shader pass | Diagnostic | Enable | Real bounded `ImagePass` pipelines, source bindings, samplers, mappings, and pipeline cache | `G3`, `G4`, `G5`, `G7`, `G8` |
| `BEP-05` | Separable blur pass | FutureRender | Enable | high-precision `ImagePass`; apply `COR-03` and `COR-08` | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BEP-06` | Mask compositor | Diagnostic | HoldDiagnostic | Internal alpha-mask `CompositePass` may land first; broad capability stays false until all claimed mask inputs execute | `G1`, `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BEP-07` | Backdrop compositor | Supported | Correct | GPU-resident capture/filter/clip/foreground-combine/outer-effect/composite chain | `G3`, `G4`, `G5`, `G6`, `G7`, `G8` |
| `BEP-08` | CPU reference path | Supported | Correct | `CpuReference` semantic oracle and explicit fallback; apply `COR-03` and `COR-08` | `G1`, `G3`, `G4`, `G7`, `G8` |

## Property Cross-Reference Preservation Rule

The 22 property cross-reference rows in the reconciliation ledger remain
authoritative and are not duplicated here. A CSS/style surface is preserved
only when every primitive ID mapped by that row has satisfied its applicable
gates. A diagnostic or root-owned sub-boundary must remain visible even when the
other mapped primitive IDs are supported.

For example, `filter` is not migration-complete merely because `FLT-03` through
`FLT-10` pass; it also depends on `FLT-01`, `FLT-02`, `FLT-11` through `FLT-14`,
the relevant shadow IDs, and backend IDs `BEP-03` through `BEP-08` according to
the supported/diagnostic split. The future sequence must include a final
property-level reconciliation after primitive-level work.

## Dependency Constraints

The later sequence must respect these dependencies without treating them as a
preselected phase list:

- corrections and typed pixel/spatial/capability models precede route changes;
- the render plan, persistent resources, and capture/canonicalization/composite
  path precede feature effects;
- mask, color-filter, blur, and shadow foundations precede any whole-chain
  filter cutover;
- backdrop migration follows stable ordinary capture/filter/composite behavior;
- a semantic row is complete only after all of its supported consumers use one
  coherent route and the superseded production edge can retire;
- diagnostic expansion remains separate and never follows from internal
  plumbing alone;
- final reconciliation covers every primitive, property mapping, target,
  quality, resource, and boundary gate.

## Per-Item Migration Rule

Each future sequence item must:

1. cite every affected overlay ID and correction ID;
2. distinguish foundation work from semantic route cutover;
3. name the applicable gates and preserve unrelated direct behavior,
   diagnostics, and root boundaries;
4. update capabilities only after executable support exists and retire a
   superseded route only after its final consumer moves;
5. reconcile the matrix, root handoff, and correction mappings through the
   `AGENTS.md` workflow.

## Mechanical Completeness Contract

For this baseline overlay, verify exactly 101 unique primitive IDs and copied
status totals of 68 `Supported`, 26 `Diagnostic`, 6 `DeferredToRoot`, and 1
`FutureRender`. Those numbers describe the 2026-07-10 reconciliation ledger;
they are not invariants after an intentional status or inventory change. Every
later overlay revision must derive its row count, row order, names, statuses,
and totals from the then-current reconciliation ledger and match it exactly.

Before this overlay or a later revision is accepted, also verify:

- every reconciled primitive/backend row appears exactly once;
- correction mappings name explicit overlay IDs and remain route-complete;
- every `Migrate`, `Correct`, or `Enable` row names the applicable target route
  and reconciliation gate;
- every `HoldDiagnostic` row preserves a typed operation/capability boundary;
- every `HoldRoot` row remains outside the crate;
- no migration item can mark a property surface complete while one of its
  mapped primitive rows is unverified;
- no public support claim is inferred from internal pass plumbing;
- supported target routes receive executable verification rather than skipped
  coverage;
- no backwards compatibility shim is required.

## Overlay Completion

This migration overlay is complete when:

1. the mechanical completeness contract passes against the current ledger;
2. the correction register agrees with current code evidence;
3. the architecture contracts agree with the modeling guide and crate boundary;
4. the worktree contains only the intended planning change;
5. a separate clean-context reviewer inspects this overlay against
   `AGENTS.md`, the modeling guide, all source matrices/handoffs, current code
   evidence, and the 101-row completeness requirement and returns clean with no
   findings.
