# Render Style Paint Support Planning Directive

This directive asks `surgeist-render` to design a crate-local implementation
plan for consuming the new CSS/style paint surface.

Root is moving Surgeist-to-Surgeist lowering into the root facade. Render should
not own CSS parsing, style cascade, selector matching, resource loading policy,
or cross-crate lowering adapters. Render should expose intentional paint-facing
contracts that root can feed after style has resolved authored values into the
appropriate symbolic or concrete render inputs.

## Scope

Write an implementation plan for render-owned paint and effect support. The
plan should cover only render responsibilities:

- background layers: colors, gradients, image layers, repeat, attachment if
  supported, size, position, clip, and origin
- border, outline, radius, and box-decoration paint inputs
- shadows, opacity, blend/compositing inputs, filters, and backdrop filters
- masks and clip paths as render-facing paint/effect data, with shape-owned
  geometry called out separately
- transforms as render-facing transform data, with layout hit-testing and
  runtime animation scheduling left to their owning layers
- color realization inputs: `currentColor`, system colors, color functions,
  color spaces, `color-mix`, and relative colors where render must either
  consume resolved colors or own final paint-space conversion
- generated-content paint inputs only after root/product materialization exists

## CSS/Style Features This Enables

The render plan should explicitly account for the render-side needs of:

- layered backgrounds, including multiple gradients and images
- background repeat, size, position, clip, and origin semantics
- border radii and outlines, including interaction with shape geometry
- box shadows and text shadows where render owns the final paint primitive
- opacity and effect-stack ordering
- filters, backdrop filters, masks, and clip paths
- transforms and transform-origin as paint/compositing inputs
- symbolic colors and color-function outputs from style/root
- `::selection` paint buckets once runtime/root provide selection state
- pseudo-element paint buckets only after root decides materialization policy

## Boundary Rules

Do not add CSS parsing, cascade, selector matching, or style rule resolution to
render. The expected boundary is:

- CSS owns syntax and authored value contracts.
- Style owns style rules, cascade/resolution data, and symbolic style models.
- Root owns Surgeist-to-Surgeist lowering, resource policy, and integration.
- Shape owns reusable shape, path, radius, stroke, and clip geometry contracts.
- Text owns shaping, measurement, and text layout contracts.
- Runtime owns animation/frame scheduling and dynamic state clocks.
- Render owns render-facing draw/effect data and backend-oriented paint
  contracts.

Do not load external images, fonts, masks, or imported resources in render
unless the plan makes a separate reviewed case for a backend-local resource
hook. Root should own resource graph policy; render may consume resolved handles
or explicit unresolved-resource diagnostics.

Do not materialize pseudo-elements as anonymous render or retained product
nodes in this plan. Render may plan how to consume already-materialized paint
inputs once root/product policy exists.

## Planning Requirements

The implementation plan should:

- identify the public render APIs root would call
- distinguish symbolic style data from render-ready paint data
- specify which values render expects root/style to resolve before handoff
- specify which values render should preserve symbolically for backend
  realization
- define explicit unsupported-value diagnostics for paint features render cannot
  consume yet
- identify required shape contracts for radii, borders, outlines, masks, and
  clip paths without duplicating shape algorithms
- identify required runtime contracts for animated paint/effect values without
  owning scheduling
- include tests for layered backgrounds, gradients, color realization,
  box-decoration paint inputs, effect-stack ordering, masks/clips, transforms,
  and unsupported-value rejection
- call out any deferred surfaces such as external resources, generated content,
  pseudo-elements, or backend-specific effects

## Review Gate

Before implementation, have a clean-context reviewer check the render plan
against:

- this directive
- render's crate boundary
- `/Users/codex/Development/surgeist/guidance/surgeist-rust-modeling-guide.md`
- the root inventory at
  `/Users/codex/Development/surgeist/plans/2026-07-04-css-integration-support-inventory.md`

Completion for this directive is a reviewed implementation plan in render's
`plans/` folder, not code changes.
