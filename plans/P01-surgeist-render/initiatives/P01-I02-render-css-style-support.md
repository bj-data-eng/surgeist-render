# Render CSS/Style Primitive Support Matrix

Date: 2026-07-08

## Purpose

This matrix is the render-owned reference for supporting the full CSS/style
paint and effect surface. It is exhaustive by render primitive family rather
than by CSS property so the crate can design from first principles and root can
adapt its lowering to the render contract.

Render is self-contained. It does not parse CSS, cascade style, load external
resources, depend on shape, or depend on style. It receives render-ready or
render-symbolic data from root and owns the primitives, diagnostics, and backend
pipelines needed to realize that data with Vello and additional render-local
pipelines.

Backwards compatibility shims are intentionally out of scope at this phase.

## Matrix Columns

| Column | Meaning |
| --- | --- |
| Primitive | Render-owned operation or data contract. |
| Enables | CSS/style surface that root can lower into the primitive. |
| Input expectation | What render expects root to provide. |
| Realization path | How render should implement the primitive. |
| Current state | Current `surgeist-render` status. |
| Completion contract | Observable behavior or tests required before marking supported. |

## Realization Labels

| Label | Meaning |
| --- | --- |
| `Vello direct` | Encode directly into `vello::Scene` or Peniko/Vello data. |
| `Render normalization` | Resolve into lower-level render commands before backend encoding. |
| `Pre-Vello pipeline` | Run a render-local WGPU/CPU pass before Vello composition. |
| `Post-Vello pipeline` | Render a Vello layer to texture, then run WGPU/CPU processing. |
| `Compositor pipeline` | Requires offscreen render targets, layer isolation, blending, masking, or backdrop capture. |
| `Diagnostic` | Render must reject unsupported or unresolved input with a typed error. |

## Current Baseline

Current render can encode basic fills, strokes, images, text runs, shadows, and
layers. The public API has concrete `Color`, `Paint`, `Gradient`, `Image`,
`Shape`, `Stroke`, `Shadow`, `Layer`, `Filter::try_blur`, `BlendMode`, and
`Transform` types.

Important current gaps:

- `Layer::mask` and `Layer::filter` are modeled but rejected by current
  capabilities.
- `Filter` is blur-only.
- `Paint` supports concrete color, simple linear/radial/sweep gradients, and
  in-memory images, but not CSS image layers or symbolic colors.
- Image drawing supports fit but not CSS repeat/position/clip/origin semantics.
- Shadows are limited to solid-color rect/rounded/circle shadows at encoding.
- Path inside/outside stroke alignment is rejected.
- Blend modes are modeled but only a subset is present.
- Backdrop filters, masks, clip-path URLs, filter graphs, color-space
  realization, and CSS background/border decoration semantics do not exist yet.

## Boundary Rules

- Render consumes final layout geometry, pixel sizes, resolved resource handles,
  and any already-realized paths or masks supplied by root.
- Render may preserve render-symbolic data only when backend realization needs
  render context, such as color-space conversion, filter graph lowering, or
  texture allocation.
- Render does not depend on sibling crates. Any shape-like or text-like input is
  accepted through render-owned public types or backend-neutral value structs.
- Root owns cross-crate lowering, resource policy, pseudo-element
  materialization, generated-content materialization, animation sampling, and
  integration.
- Unsupported or unresolved inputs are render diagnostics, not silent no-ops.

## 1. Paint Sources

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Solid RGBA paint | `color`, `background-color`, border/outline colors, shadow colors, text paint, generated/selection paint colors | Premultiplied policy and color values must be explicit; root may provide concrete RGBA or render may realize symbolic color through a color pipeline | `Vello direct` | Supported for concrete RGBA | Unit tests for finite channel validation, transparent black, alpha preservation, text/fill/stroke/shadow use |
| Symbolic color token | `currentColor`, system colors, `color()`, `color-mix()`, relative colors, wide-gamut CSS color spaces | Root supplies symbolic color payload plus current color/system palette/color management context, or resolves before handoff | `Render normalization` plus optional color conversion pipeline | Missing | Matrix-driven API accepts symbolic color payloads or explicitly documents root-resolved-only policy; tests cover `currentColor`, system color, `color-mix`, unsupported color space diagnostics |
| Paint-space color conversion | Lab/LCH/Oklab/Oklch/HWB/HSL/predefined color spaces to render target space | Source color space, target paint space, alpha handling, interpolation method when relevant | `Render normalization` or precomputed CPU helper | Missing | Deterministic conversion tests with known vectors; invalid/unsupported space diagnostics |
| Linear gradient | `linear-gradient`, background/border/mask images where lowered to gradient | Start/end points or CSS angle lowered to render-space geometry; ordered stops | `Vello direct` | Basic linear supported | Tests for stop validation, repeated use as fill/background layer, transparent stops, nonuniform transforms |
| Radial gradient | `radial-gradient` | Center, radii/shape/extent already resolved or resolvable from target rect | `Vello direct` for simple circular, render normalization for CSS ellipse/extent | Basic circular radial supported | Tests for circle/ellipse lowering, contain/cover extents, position, clipped background layer |
| Conic/sweep gradient | `conic-gradient` | Center, start angle, stops, repeating flag | `Vello direct` for sweep, render normalization for CSS conic semantics | Basic sweep supported without CSS semantics | Tests for start angle, center position, stop interpolation, repeating conic rejection or support |
| Repeating gradients | `repeating-linear-gradient`, `repeating-radial-gradient`, `repeating-conic-gradient` | Resolved repeating stop period and tile behavior | `Render normalization` if representable, otherwise texture shader pipeline | Missing | Tests for repeated bands, clipped layer bounds, degenerate stop diagnostics |
| Image paint | `url(...)`, image backgrounds, mask images, list-style images, generated image content | Resolved image handle or in-memory image; render does not load URLs | `Vello direct` for simple image draw, image sampler pipeline for CSS tiling | In-memory images supported | Tests for resolved handle API, missing resource diagnostics, reuse/caching, alpha preservation |
| Filtered image paint | CSS `filter(image, ...)`, filtered background images | Resolved source image plus filter list | `Pre-Vello pipeline` or `Post-Vello pipeline` depending on source | Missing | Tests for image-filter color functions, blur bounds, clip interaction |

## 2. Geometry Targets

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Rect fill/stroke | Box backgrounds, borders, outlines, decorations | Physical rect | `Vello direct` | Supported | Tests for fill/stroke parity and transforms |
| Rounded rect fill/stroke | Border radius, background clip, rounded masks, rounded backdrop bounds | Physical rect plus normalized corner radii | `Vello direct` for fill/stroke, render normalization for stroke alignment | Supported for basic radii | Tests for elliptical radii, clamping, clip/background interactions |
| Circle/ellipse fill/stroke | Basic shape clips, SVG-like geometry, masks | Center/radii | `Vello direct` | Supported | Tests for fill/stroke/image clip interactions |
| Arbitrary path fill | `clip-path: path(...)` future, polygon lowering, complex masks, custom shapes | Render-owned path elements with fill rule | `Vello direct` | Basic path supported | Tests for winding/even-odd policy, invalid path diagnostics |
| Arbitrary path centered stroke | Outlines, vector shapes, future CSS path semantics | Path, stroke width/join/caps/dashes | `Vello direct` | Supported for centered stroke | Tests for joins/caps/dashes on paths |
| Arbitrary path inside/outside stroke | CSS-like inside/outside border semantics for arbitrary paths | Path and alignment | `Render normalization` to fill geometry or diagnostic | Rejected | Tests for supported fill expansion or precise unsupported diagnostic |
| Geometry boolean/offset support | Complex masks, non-center stroke alignment, clip combination | Render-owned geometry operations or lowered geometry from root | `Render normalization` | Missing | Tests for intersection/union/difference or documented diagnostics |
| Hit-test geometry | Pointer events and interaction | Out of render scope except optional debug metadata | Diagnostic / root-owned | Not owned | Matrix says render does not own hit testing |

## 3. Image Sampling And CSS Image Layers

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Image fit | `object-fit`-like image placement and existing render image command | Rect, image size, fit mode | `Vello direct`/sampler transform | Supported for fill/contain/cover/stretch/none | Tests for all fit modes |
| Background position | `background-position`, `mask-position` | Target painting area, image intrinsic size, resolved position components | `Render normalization` | Missing | Tests for keywords, percentages, lengths, four-component positions |
| Background size | `background-size`, `mask-size` | Painting area, intrinsic dimensions, `auto/cover/contain/explicit` | `Render normalization` | Missing | Tests for cover/contain/auto/explicit and missing intrinsic diagnostics |
| Repeat no-repeat/repeat | `background-repeat`, `mask-repeat` | Tile rect and clip rect | `Pre-Vello pipeline` or Vello image extend where sufficient | Image extend has repeat/pad/reflect, not CSS layer semantics | Tests for repeat-x/repeat-y/no-repeat and clipping |
| Repeat round | `background-repeat: round` | Tile sizing policy resolved against paint area | `Render normalization` plus sampler | Missing | Tests for tile count rounding and non-distortion expectations |
| Repeat space | `background-repeat: space` | Tile spacing resolved against paint area | `Render normalization` plus multiple image draws or shader | Missing | Tests for tile distribution and leftover spacing |
| Background origin | `background-origin` | Border/padding/content boxes supplied by root | `Render normalization` | Missing | Tests for origin box selection |
| Background clip | `background-clip`, text clip if later supported | Clip geometry supplied as render-owned shape/path/text mask | `Compositor pipeline` for non-shape clips, Vello clip for simple | Layer clip supports shape | Tests for border/padding/content clips and rounded radii |
| Background attachment | `background-attachment: scroll/fixed/local` | Root supplies scroll-adjusted paint rect or viewport-space anchor | Root normalization plus render transform | Missing | Tests document root-owned scroll adjustment; render tests fixed anchor behavior if accepted |
| Multi-layer image stack | Layered backgrounds/masks | Ordered layer list with per-layer size/position/repeat/origin/clip | `Render normalization` into draws/layers | Missing | Tests for list length matching, ordering, transparency |

## 4. Box Decoration, Borders, And Outlines

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Background layer stack | `background-*` | Box geometry and layer list | `Render normalization` | Missing | Tests for color behind images, gradient/image ordering |
| Border side solid | `border-*-width/style/color`, `border` | Per-side width/style/color and normalized radii | `Render normalization` into fills/strokes | Partial via stroke, not CSS sides | Tests for four different side widths/colors |
| Border style none/hidden | CSS border suppression | Per-side style | `Render normalization` | Missing | Tests that none/hidden suppress paint and affect joins as specified by root input |
| Border dashed/dotted | CSS dashed/dotted borders | Per-side geometry, dash policy | `Render normalization` and `Vello direct` stroke where compatible | Dash exists for strokes | Tests for side-specific dashes, rounded corners, phase continuity |
| Border double | `border-style: double` | Side widths and colors | `Render normalization` into multiple bands | Missing | Tests for thin/medium/large widths and radii |
| Border groove/ridge/inset/outset | 3D CSS border styles | Resolved colors or lighting policy from root/render | `Render normalization` into multiple colored bands | Missing | Tests for color derivation or explicit diagnostic |
| Border radius clipping | Rounded background/border/mask clips | Normalized corner radii | `Vello direct` clips plus normalization | Basic rounded shapes | Tests for radius clamping and background clipping |
| Outline solid/dashed/dotted | `outline-*` | Outline offset if supported, style/width/color | `Render normalization` | Missing | Tests for outlines outside border box and non-layout effect |
| Outline auto | `outline-style: auto` | Root/host policy or render theme token | Diagnostic unless explicit theme primitive exists | Missing | Tests for unsupported diagnostic or theme-provided style |
| Box decoration break | `box-decoration-break: slice/clone` | Fragment list from root/layout | Root supplies fragments; render paints per fragment | Missing | Tests for clone vs slice on provided fragments |

## 5. Shadows

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Outer box shadow | `box-shadow` without `inset` | Shape/box, offset, blur, spread, color | `Render normalization`; Vello shadow where possible, custom blur pipeline where needed | Partial for solid-color rect/rounded/circle | Tests for spread, blur, negative/positive offsets, rounded rects |
| Inset box shadow | `box-shadow: inset ...` | Box clip and inner shadow parameters | `Compositor pipeline` | Missing | Tests for clipped inner blur and rounded corners |
| Multiple shadows | Shadow lists | Ordered list | `Render normalization` into ordered commands | Partial through repeated commands by caller | Tests for stacking order and transparent overlap |
| Drop shadow filter | `filter: drop-shadow(...)` | Alpha mask of source layer plus shadow params | `Post-Vello pipeline` | Missing | Tests distinguish drop-shadow alpha from box-shadow box geometry |
| Text shadow | `text-shadow` or text-facing lowering | Glyph alpha mask or text run plus shadow params | Text/root supplies runs; render applies shadow layer | Missing | Tests for glyph-shaped shadow and ordering behind text |
| Non-solid shadow paint | Gradient/image shadow paints if render API permits | Paint payload | Diagnostic unless custom pipeline exists | Rejected today | Tests for explicit rejection or implemented rasterization |

## 6. Filters

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Filter list model | `filter`, `backdrop-filter`, image filters | Non-empty ordered list of filter ops or `none` | `Render normalization` | Blur-only layer filter exists | Tests for list order and identity handling |
| Blur | `blur()` | Standard deviation/radius, edge mode, filter region | `Post-Vello pipeline` or Vello blur when available | Modeled but rejected as layer filter | Tests for inflated bounds, large-radius clamp policy, transparent edges |
| Brightness | `brightness()` | Scalar amount | Fused color matrix/component transfer shader | Missing | Pixel tests for amount 0, 1, >1 |
| Contrast | `contrast()` | Scalar amount | Fused color shader | Missing | Pixel tests for amount 0, 1, >1 |
| Grayscale | `grayscale()` | Clamped amount | Fused color matrix shader | Missing | Pixel tests for 0, .5, 1 |
| Hue rotate | `hue-rotate()` | Angle | Fused color matrix shader | Missing | Pixel tests for angle wrap and identity |
| Invert | `invert()` | Clamped amount | Fused component transfer shader | Missing | Pixel tests including transparent pixels |
| Opacity filter | `opacity()` | Clamped amount | Fused color shader or layer opacity if equivalent | Missing | Tests distinguish filter opacity order from layer opacity |
| Saturate | `saturate()` | Scalar amount | Fused color matrix shader | Missing | Pixel tests for 0, 1, >1 |
| Sepia | `sepia()` | Clamped amount | Fused color matrix shader | Missing | Pixel tests for 0, .5, 1 |
| URL/SVG/reference filter | `filter: url(...)` | Resolved filter graph handle or explicit unsupported resource | Diagnostic until graph exists | Missing | Tests for unsupported URL diagnostic |
| Filter fusion | Chains of color-only filters | Ordered color ops | `Pre/Post-Vello pipeline` with shader fusion | Missing | Tests for chain equivalence and order sensitivity |
| Filter region/outsets | Pixel-moving filters | Source bounds plus filter-specific inflation | `Render normalization` | Missing | Tests for dirty-region inflation and clipping |
| Software/reference fallback | Deterministic test oracle, non-GPU fallback | CPU buffers | CPU helper behind tests or backend option | Missing | Golden tests for blur/color ops independent of GPU variance |

## 7. Backdrop Filters

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Backdrop capture | `backdrop-filter` | Layer backdrop region, clip shape, existing scene content behind layer | `Compositor pipeline` | Missing | Tests for rect and rounded-rect backdrop capture |
| Backdrop filter chain | `backdrop-filter: blur(...) saturate(...)` | Captured backdrop texture and filter list | `Post-Vello pipeline` | Missing | Tests for filter order and clipping |
| Backdrop isolation | CSS stacking/compositing interaction | Root supplies stacking order; render isolates texture passes | `Compositor pipeline` | Missing | Tests for sibling ordering and nested backdrops |
| Root backdrop policy | Backdrop on root element | Root policy or explicit diagnostic | Diagnostic / root-owned | Missing | Tests for root backdrop rejection or accepted behavior |

## 8. Masks And Clips

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Shape clip | `overflow`, `background-clip`, `clip-path` basic shapes after lowering | Render-owned shape/path | `Vello direct` clip/layer | Shape clip exists | Tests for nested clips and transform interaction |
| Path clip | `clip-path: polygon(...)`, future path semantics | Render-owned path and fill rule | `Vello direct` clip/layer | Path shape exists | Tests for even-odd/nonzero and bounds |
| Clip URL/reference | `clip-path: url(...)` | Resolved clip geometry or resource diagnostic | Diagnostic until resource graph exists | Missing | Tests for unsupported URL diagnostic |
| Basic shape clip | `inset()`, `circle()`, `ellipse()`, `polygon()` | Root-lowered or render-symbolic basic shape plus reference box | `Render normalization` | Missing as CSS shapes | Tests for all basic shapes |
| Alpha mask | `mask-image`, shape masks, layer masks | Mask texture/shape and mode | `Compositor pipeline` | Modeled but rejected | Tests for alpha mask compositing |
| Luminance mask | CSS mask mode if exposed later | Mask texture and color-to-alpha policy | `Compositor pipeline` | Missing | Tests for luminance conversion or diagnostic |
| Multi-layer mask | `mask` shorthand and longhands | Mask layer list with image/position/size/repeat | `Compositor pipeline` | Missing | Tests for layer order and repetition |
| Mask composite | Future CSS mask-composite | Blend/composite operators | `Compositor pipeline` | Missing | Diagnostic until style/root expose it |

## 9. Blend, Opacity, And Compositing

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Layer opacity | `opacity` | Finite scalar, CSS usually 0..1 after root/style validation | `Compositor pipeline` or Vello layer | Modeled | Tests for nested opacity and filter-order interaction |
| Mix blend mode | `mix-blend-mode` if exposed later, existing blend enum | Blend mode and isolated backdrop | `Compositor pipeline` | Partial enum only | Tests for all supported modes |
| Background blend mode | `background-blend-mode` if exposed later | Per-background-layer blend list | `Compositor pipeline` | Missing | Diagnostic until style/root expose it |
| Isolation group | `isolation`, filter/mask/backdrop groups | Root stacking context intent or render layer policy | `Compositor pipeline` | Internal `LayerIsolation` exists | Tests for isolated vs non-isolated blending |
| Porter-Duff/composite ops | Masks, SVG filters, future canvas-like effects | Composite operator | `Compositor pipeline` | Missing | Tests for source-over, source-in, destination-in as needed |

## 10. Transforms And Coordinate Spaces

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| 2D affine transform | `transform`, `translate`, `rotate`, `scale` after 2D lowering | Matrix or render transform | `Vello direct` layer transform | Supported | Tests for translate/scale/rotate/matrix composition |
| Transform origin | `transform-origin` | Root-resolved origin point or reference box plus symbolic origin | `Render normalization` | Missing as origin primitive | Tests for origin-wrapped transform |
| Skew | `skew`, `skewX`, `skewY` | 2D affine matrix | `Vello direct` once matrix API accepts it | Missing as public helper | Tests for skew matrix |
| 3D transform flattening | `matrix3d`, `rotate3d`, perspective, translateZ, scaleZ | Root-flattened 2D matrix or explicit 3D unsupported payload | Diagnostic unless 3D compositor exists | Missing | Tests for unsupported 3D diagnostics |
| Coordinate-space tagging | background fixed, backdrop capture, masks, transforms | Space IDs or explicit pre-resolved coordinates | `Render normalization` | Missing | Tests for transformed clips/images/backdrops |

## 11. Text Paint Hooks

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Glyph fill paint | `color`, text paint, generated text | Shaped glyphs plus paint | `Vello direct` | Supported for concrete paint | Tests for color/gradient fill if allowed |
| Text decoration paint | underline/overline/line-through color/thickness/style | Text/root supplies decoration geometry or render owns simple line decoration | `Vello direct` / render normalization | Missing as CSS decoration primitive | Tests for decoration colors and thickness |
| Text shadow | `text-shadow` | Glyph alpha or text run plus shadow list | `Post-Vello pipeline` or repeated text draws | Missing | Tests for glyph-shaped shadows |
| Selection paint bucket | `::selection` | Root/runtime supplies selection geometry and paints | `Vello direct`/layered text paint | Missing | Tests once root supplies selection ranges |
| Generated content paint bucket | `content`, `::before`, `::after`, list markers | Root materializes content into normal render commands | Existing primitives | Not a special render feature | Tests verify render consumes already-materialized commands |

## 12. Resource Handles And Images

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Resolved image handle | URL images, mask images, list marker images | Root-owned resource handle or explicit bytes | Render texture cache | In-memory image ID exists | Tests for handle identity and reuse |
| Intrinsic image metadata | background-size auto/contain/cover, image aspect ratio | Width, height, density/orientation policy if relevant | `Render normalization` | Partial through image buffer size | Tests for intrinsic sizing |
| Image orientation/color profile | CSS image rendering if exposed | Root-resolved bitmap or metadata | Diagnostic unless render owns conversion | Missing | Tests define whether root or render owns it |
| Animated image frame | GIF/APNG/video-like resources | Runtime/root selects frame | Render consumes current frame | Not owned | Matrix says scheduling/frame selection out of scope |

## 13. Diagnostics And Capability Reporting

| Primitive | Enables | Input expectation | Realization path | Current state | Completion contract |
| --- | --- | --- | --- | --- | --- |
| Unsupported primitive diagnostics | Root and tests can detect missing render support | Typed error code and affected primitive | `Diagnostic` | Existing unsupported capability enum is narrow | Tests for each unsupported matrix row |
| Backend capability matrix | Root can choose fallback/lowering | Public capability flags by primitive family | Render-owned capability struct | Narrow current flags | Tests for Vello 0.9 baseline capability values |
| Unresolved resource diagnostics | URL filters/images/masks/clips not resolved by root | Resource token or handle state | `Diagnostic` | Missing | Tests for unresolved image/mask/filter |
| Invalid value diagnostics | Non-finite values, impossible geometry, empty lists | Validated constructors | `Diagnostic` | Good for current types | Tests for new primitive invariants |
| Degraded-quality diagnostics | Fast blur clamps, software fallback, unsupported wide gamut | Optional warnings/statistics | Render stats/capabilities | Missing | Tests for stats when quality-degraded path is selected |

## 14. Backend Pipeline Requirements

| Pipeline | Needed for | Required capabilities | Current state | Completion contract |
| --- | --- | --- | --- | --- |
| Vello scene encoder | Fills, strokes, text, simple images, simple clips, gradients | Stable scene encoding and validation | Present | Existing checks stay green |
| Texture cache/upload | Images, masks, offscreen layers | Image identity, format, reuse, lifetime | Partial image upload | Tests for reuse and release |
| Offscreen layer renderer | Filters, masks, opacity, blend, backdrop | Render subtree to texture with bounds | Missing | Tests for layer isolation and nested effects |
| Fullscreen/rect shader pass | Color filters, component transfer, simple composites | WGPU pipelines over texture views | Missing | Pixel tests for color filters |
| Separable blur pass | Blur, drop-shadow, backdrop blur | Horizontal/vertical passes, bounds/outsets, clamp policy | Missing | Pixel/bounds tests |
| Mask compositor | Alpha/luminance masks and clips | Destination-in/source-in style compositing | Missing | Pixel tests for mask edges |
| Backdrop compositor | Backdrop filters and blend modes | Capture prior content, apply filters, composite foreground | Missing | Ordering tests |
| CPU reference path | Deterministic tests and possible fallback | CPU buffer ops for filters | Missing | Reference tests independent of GPU |

## Property Cross-Reference

| CSS/style surface | Primary render primitives |
| --- | --- |
| `color`, `background-color`, `border-*-color`, `outline-color`, text decoration color | Solid/symbolic color, color conversion |
| `background-image` | Image paint, gradients, repeating gradients, filtered image paint |
| `background-position` | Background position |
| `background-size` | Background size, intrinsic image metadata |
| `background-repeat` | Image repeat/no-repeat/round/space |
| `background-origin` | Origin box selection |
| `background-clip` | Shape/path/text clips |
| `background-attachment` | Coordinate-space tagging and root scroll lowering |
| `border`, side borders, `border-style`, `border-width` | Border side paint, stroke/fill normalization |
| `border-radius` and corner radii | Rounded rect geometry and clips |
| `outline` and outline longhands | Outline paint |
| `box-decoration-break` | Fragmented decoration paint |
| `box-shadow` | Outer/inset/multiple shadows |
| `opacity` | Layer opacity and compositing |
| `filter` | Filter list, color filters, blur, drop-shadow, URL diagnostics |
| `backdrop-filter` | Backdrop capture and filter chain |
| `clip-path` | Shape/path/basic-shape/reference clips |
| `mask`, `mask-image`, `mask-size`, `mask-position`, `mask-repeat` | Mask layer stack, image sampling, mask compositing |
| `transform`, `transform-origin`, `translate`, `rotate`, `scale` | 2D transforms, origin normalization, 3D diagnostics |
| `text-shadow` | Text/glyph shadow pipeline |
| `::selection` | Selection paint bucket once root/runtime materialize ranges |
| `content`, pseudo-elements, list markers | Normal render commands after root materialization |

## Initial Implementation Sequence

1. Replace the narrow capability model with this matrix's primitive-family
   capability surface and diagnostics.
2. Add render-owned paint/effect data models for CSS background layers, borders,
   masks, filters, and symbolic colors.
3. Implement render normalization for backgrounds, image sizing/repeat, border
   sides, outlines, and transform origins.
4. Add offscreen layer infrastructure.
5. Implement color-filter shader fusion.
6. Implement blur and drop-shadow pipelines.
7. Implement alpha masks and clip-path/basic-shape lowering.
8. Implement backdrop capture and backdrop filters.
9. Add CPU reference tests/golden checks for filters and compositing.
10. Reconcile remaining diagnostics against root integration needs.

## Review Checklist

- Every CSS/style paint/effect family has at least one render primitive row.
- Every row names a realization path or a diagnostic.
- Render does not depend on style, css, shape, text, runtime, or root.
- External resources are handles or diagnostics, not render-loaded URLs.
- Pseudo-elements and generated content are root-materialized command streams,
  not render-owned tree nodes.
- Backwards compatibility is not used to preserve obsolete APIs.
