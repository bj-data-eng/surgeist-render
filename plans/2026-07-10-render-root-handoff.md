# Render Root Handoff Contract

Date: 2026-07-10

Crate: `surgeist-render`

Sources of truth:

- Matrix reconciliation ledger:
  `plans/2026-07-10-render-css-matrix-reconciliation.md`
- Active sequence plan:
  `plans/2026-07-09-render-css-implementation-sequence.md`
- Sequence 15 Task 4 plan:
  `plans/2026-07-10-full-matrix-reconciliation-root-handoff-implementation.md`

This note is the root-facing render contract after Sequence 15 reconciliation.
It summarizes what root may hand to render, what render will diagnose, and what
root/text/layout/style/runtime must resolve first.

## Handoff Rule

Root hands render an ordered, render-ready command stream plus concrete
resources, geometry, glyph runs, and capability-aware effect requests. Render
does not parse CSS, cascade style, shape text, calculate layout, construct
pseudo-elements, load URLs, or infer runtime animation/resource policy.

Unsupported behavior remains unsupported even when root could synthesize a
workaround. Such workarounds are root-owned materialization, not new render
support.

## Status Vocabulary

| Status | Root-facing meaning |
| --- | --- |
| `Supported` | Root may pass render-ready values for the named primitive. Render validates and encodes/normalizes/executes the implemented path. |
| `Diagnostic` | The concept is represented at the render boundary, but render rejects the unsupported or unresolved case through typed diagnostics/capabilities. |
| `DeferredToRoot` | Root/style/layout/text/runtime must resolve or materialize the concept before render sees ordinary commands. |
| `FutureRender` | The work remains render-owned but is intentionally not implemented in the current sequence. |

## Supported Render Primitives

These rows are `Supported` in the reconciliation ledger. They are render-owned
only for render-ready inputs.

| Family | Supported handoff surface |
| --- | --- |
| Paint sources | Concrete `Color`/`Paint::Color`; finite sRGB/HSL-derived paint colors; render-space linear, radial, and sweep gradients; in-memory `Image`, `ImageId`, and resolved image resources. |
| Geometry | Rects; rounded rects; circles/ellipses; arbitrary filled paths with fill rules; centered path strokes with width/caps/joins/miter/dashes. |
| Image sampling and backgrounds | `ImageFit`; resolved background position/size; no-repeat/repeat/repeat-x/repeat-y tile planning; border/padding/content origin boxes; rect/shape/path background clips; scroll/fixed/local attachment plans with coordinate tags; ordered multi-layer background stacks. |
| Box decoration | Color/image background stacks; solid/none/hidden/dashed/dotted/double borders; normalized/scaled border radii; solid/dashed/dotted outlines; slice/clone box-decoration fragments. |
| Shadows | Solid outer box shadows for rect/rounded/circle geometry; ordered shadow lists; materialized image drop-shadow filters with supported parameters. |
| Filters | `FilterList`; blur planning and materialized image/backdrop blur; brightness, contrast, grayscale, hue-rotate, invert, opacity filter, saturate, sepia; color-filter fusion; filter region/outset planning; deterministic CPU/reference fallback buffers. |
| Backdrop filters | Bounded materialized backdrop capture; supported backdrop filter chains; foreground composite over filtered backdrop. |
| Clips and masks | Shape clips; path clips; root-lowered basic shape clips; materialized alpha mask buffers and resolved layer alpha masks. |
| Blend, opacity, compositing | Finite layer opacity; current direct Vello `BlendMode` set; isolated direct layer groups for supported opacity/blend/clip paths. |
| Transforms and spaces | Finite 2D affine transforms; transform-origin wrapping; skew helpers; local/viewport/surface coordinate-space tags. |
| Text paint hooks | Prepared glyph runs with concrete/gradient fill paint and font bytes/refs; materialized text decoration line geometry with supported solid paint. |
| Resources | Resolved image handles, in-memory image buffers, cache identity, intrinsic size, and optional density metadata. |
| Diagnostics and capabilities | `Capabilities::VELLO_0_9`; typed `UnsupportedPrimitive`, `UnresolvedResource`, `InvalidValue`, and `DegradedQuality` diagnostics. |
| Backend | Vello scene encoding; image texture cache/upload; bounded backdrop compositor; CPU reference paths for filters, alpha masks, blends, and validation. |

## Diagnostic Boundaries Root Must Handle

Root must read render capabilities before lowering and must handle typed
diagnostics returned by constructors, normalization, or scene validation.

| Boundary | Render diagnostic contract | Root action |
| --- | --- | --- |
| Unsupported CSS color spaces and `color-mix()` | `Diagnostic` via unsupported paint/color-space operations or degraded-quality diagnostics. | Resolve to concrete supported paint values or keep out of render. |
| Repeating gradients | `Diagnostic` via `PrimitiveOperation::RepeatingGradient`. | Expand/materialize to supported imagery/commands or defer. |
| Filtered image paint broad boundary | `Diagnostic` for broad CSS filtered-image paint; narrow materialized image filtering is separate. | Materialize the image/filter result through supported paths or reject upstream. |
| Inside/outside path strokes | `Diagnostic` for unsupported stroke alignment. | Convert to explicit fill geometry or use centered stroke only. |
| Geometry booleans and offsets | `Diagnostic` for unsupported boolean/offset operations. | Precompute geometry before handoff. |
| Hit testing | `DeferredToRoot`; render has no pointer-event hit-test primitive. | Keep hit testing in root/layout/runtime. |
| `background-repeat: round` and `space` | `Diagnostic` via repeat capability flags. | Lower to explicit tiles or avoid. |
| 3D border styles | `Diagnostic` for groove/ridge/inset/outset. | Derive explicit colored bands or avoid. |
| `outline-style: auto` | `Diagnostic` via `OutlineAutoStyle`. | Apply host/theme policy before render. |
| Inset shadows | `Diagnostic` via `InsetBoxShadow`. | Materialize outside render or avoid. |
| Non-solid shadow paint | `Diagnostic` via `NonSolidShadowPaint`. | Resolve to solid paint or avoid. |
| Text shadow | `Diagnostic` via `TextShadow`. | Provide root/text fallback materialized commands or handle rejection. |
| URL/SVG/reference filters | `Diagnostic`/`UnresolvedResourceKind::Filter`. | Resolve/filter externally or pass a future supported materialized result. |
| Backdrop isolation outside bounded supported cases | `Diagnostic` for broad nested/transformed/repeated isolation. | Construct supported stacking order or avoid unsupported backdrop cases. |
| Root backdrop filters | `Diagnostic` via root backdrop policy. | Decide root-element backdrop behavior outside render. |
| Clip URL/reference | `Diagnostic`/`UnresolvedResourceKind::Clip`. | Resolve to concrete geometry before handoff. |
| Luminance masks | `Diagnostic` for non-materialized luminance policy. | Convert to alpha mask or avoid. |
| Multi-layer masks | `Diagnostic` via multi-layer mask composition boundary. | Collapse to materialized alpha mask or defer. |
| Mask composite modes | `Diagnostic` for unsupported mask composite operators. | Precompose before render or avoid. |
| Background blend modes | `Diagnostic` for non-normal background blend lists. | Precompose blended backgrounds or avoid. |
| Porter-Duff/composite operators | `Diagnostic` for unsupported composite modes. | Materialize the composite result before render. |
| 3D transforms | `Diagnostic` for unsupported 3D transform operations. | Flatten to 2D affine values or avoid. |
| Broad offscreen layer rendering | `Diagnostic` by backend capability. | Do not require broad offscreen layer effects at render handoff. |
| Fullscreen/rect shader pass execution | `Diagnostic` by backend capability. | Use supported CPU/materialized paths or defer. |
| Broad mask compositor execution | `Diagnostic` by backend capability. | Convert to materialized alpha buffers or avoid unsupported mask stacks. |

## Values To Resolve Before Render Handoff

| Owner | Must resolve or materialize before render |
| --- | --- |
| root | CSS parsing; DOM/style tree ownership; stacking tree construction; command ordering; scroll/fixed/local coordinate intent; root backdrop policy; hit testing; pseudo-element and generated-content construction. |
| style | Cascade, inheritance, shorthands, logical-to-physical mapping, `currentColor`, system colors, relative colors, color-mix/wide-gamut policy, border/outline/shadow/filter authored values. |
| layout | Final physical rects; border/padding/content boxes; fragment lists; radii after CSS reference-box resolution; transform origins/reference boxes; background/mask positioning areas; capture/filter bounds. |
| text | Font selection; shaping; bidi/script handling; glyph IDs/positions/advances; text run ordering; decoration geometry; text color resolution. |
| runtime | URL/resource fetch; decoded bytes/handles; animated image frame selection; resource lifetime; selection ranges; viewport/scroll state; host policy decisions. |

## Resource Policy

Render consumes handles, bytes, buffers, and resolved metadata. Render does not
load URLs.

| Resource kind | Render consumes | Root/runtime responsibility |
| --- | --- | --- |
| Images | `Image`, `ImageId`, `ResolvedImageResource`, intrinsic size, optional density, cache identity. | Fetch/decode URLs, provide bytes or handles, own lifetimes, report missing resources, select current animated frame. |
| Color-managed images | Already-converted bitmap/color data plus root-resolved metadata policy. | Apply orientation and color-profile conversion before handoff. |
| Masks | Materialized alpha buffers or resolved layer alpha masks. | Resolve mask URLs/layers/modes/composites into supported alpha input. |
| Filters/clips | Resolved supported filter lists and concrete clip geometry, or typed unresolved resource diagnostics. | Resolve URL/SVG/filter/clip references outside render. |
| Fonts | Font bytes/refs attached to prepared text runs. | Locate fonts, load bytes, choose fallback fonts, and shape/layout text. |

## Text Policy

Render consumes prepared glyph runs and font bytes/refs. Render does not shape
or layout text.

| Text surface | Render contract | Root/text responsibility |
| --- | --- | --- |
| Glyph fill | `Supported` prepared `TextRun` with concrete/gradient fill paint and font data. | Shape text, compute glyph positions, resolve text color and font data. |
| Text decoration | `Supported` materialized line geometry with supported solid paint, thickness, transform, and command order. | Compute underline/overline/line-through geometry and style semantics. |
| Text shadow | `Diagnostic`; render preserves the model but rejects execution. | Implement fallback outside render or handle the diagnostic. |
| Selection text/paint | `DeferredToRoot`; render sees ordinary fill/text commands only. | Compute selection ranges, geometry, foreground/background paints, and ordering. |
| Generated text | `DeferredToRoot`; render sees ordinary prepared text/image/fill commands only. | Construct pseudo-elements, generated content, list markers, glyph runs, and resources. |

## Pseudo-Element And Selection Policy

Pseudo-elements, generated content, list markers, and selections are not render
tree features. Root materializes them as ordinary command streams before
handoff.

| Surface | Handoff rule |
| --- | --- |
| `::before`, `::after`, `content`, list markers | Root/style creates the content and emits ordinary render commands in final paint order. |
| `::selection` | Runtime/root computes ranges and emits ordinary background/text commands in final paint order. |
| Text clips or text-shaped masks | Root/text/runtime materializes supported geometry or alpha buffers before render. |

## FutureRender Work Items

| Work item | Current contract | Root integration note |
| --- | --- | --- |
| GPU separable blur pass for broad layer/drop-shadow/backdrop use | `FutureRender`: render-owned but not implemented by Sequence 15. Current support is materialized CPU/reference/narrow image execution plus bounded backdrop paths. | Materialize sources into currently supported image/backdrop/filter paths or avoid broad layer blur until render implements the GPU pass. |

## Root Integration Checklist

- Read `Capabilities::VELLO_0_9` and family accessors before selecting lowering
  paths.
- Treat `Diagnostic`, `DeferredToRoot`, and `FutureRender` rows as hard
  boundaries, not as supported behavior.
- Hand render only final geometry, final ordering, prepared glyph runs, resolved
  resources, and supported effect payloads.
- Preserve typed render diagnostics in root-facing error/reporting paths.
- Keep URL loading, CSS parsing, style cascade, text shaping, layout, hit
  testing, selection range calculation, pseudo-element construction, animation
  sampling, and host policy decisions outside render.
