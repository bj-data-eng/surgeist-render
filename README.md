# surgeist-render

GPU rendering contracts and backend-facing draw data for Surgeist surfaces and
scenes. The crate owns scene validation, GPU execution, surface resources,
explicit headless readback, and render diagnostics. Style, layout, text shaping,
application/window lifecycle, and Surgeist-to-Surgeist lowering remain outside
this crate.

## Execution contract

Every successful frame takes exactly one public route reported by
`Stats::route`:

- `RenderRoute::DirectVello` renders an effect-free scene with one
  transaction-owned internal Vello raster pass.
- `RenderRoute::GpuGraph` renders resolved image alpha masks, supported
  blend/composition, and bounded backdrop filter lists (ordered color, blur,
  and drop-shadow operations) with crate-owned WGPU image/composite passes.

Production is GPU-only: there is no production CPU fallback, CPU effect retry,
implicit readback, or Vello-atlas re-entry for graph results. CPU pixel
algorithms are test-only quality oracles. A missing adapter, format, device
limit, or surface lifecycle capability returns a typed error without publishing
partial pixels or statistics.

Effect graphs prefer high-precision premultiplied numeric-sRGB
`Rgba16Float`. `EffectQualityPolicy::RequireHighPrecision`, the default, rejects
a device that cannot provide it. `EffectQualityPolicy::AllowReducedPrecision`
permits an explicit `Rgba8Unorm` route only when high precision is unavailable;
the selected result is observable through `Stats::effect_precision`. Neither
policy authorizes CPU execution.

## Capabilities, operations, and publication

`Capabilities::CURRENT` reports semantic support for authored rendering
operations. `Renderer::runtime_capabilities` instead reports immutable facts
about the selected device and surface, including effect-format availability and
maximum two-dimensional effect texture size. Cargo features choose compiled
host adapters; they are not capability reports.

Renderer creation, surface creation/resume, rendering, and readback are async
GPU operations. A render uses a transaction-owned command submission and is
failure-atomic: validation, allocation, encoding, submission, device loss,
cancellation, or presentation failure leaves the previous complete publication
and the renderer's last successful `Stats` unchanged. `Renderer::stats` means
the last successful published frame, not the most recent attempt.

Headless pixels are copied to CPU memory only when callers explicitly await
`Renderer::read_headless`. It returns tightly packed straight-alpha RGBA8
physical pixels from the current complete publication. Rendering itself never
uses that readback path.

## Host and repository boundaries

Native headless execution is available without a host window. With
`render-window`, the tracked presented smoke target renders and presents one
direct frame and one GPU-graph frame through the live window lifecycle:

```sh
CARGO_NET_OFFLINE=true cargo run -p surgeist-render --example render_window_smoke --features render-window
```

That target owns only example lifecycle and requires a live graphical session;
host unavailability is a blocker, not a passing skip. The additive
`render-window,render-web` feature combination uses the same native smoke with
both features enabled.

For `wasm32-unknown-unknown`, the leaf contract is compile-only with
`render-web --lib --tests`. Real browser canvas event-loop execution and
presentation require a browser host and remain root integration evidence; a
successful wasm build is not a browser execution claim. Native `WebCanvas`
construction remains a typed unsupported-platform diagnostic.

The root `surgeist` repository owns the public root facade, cross-crate
adapters, browser-host integration, generated API artifacts, and the
`surgeist-render` gitlink. This leaf repository owns its source API and focused
verification; it does not generate or update root artifacts. Rust 1.97 is the
root integration compatibility floor; the leaf manifest intentionally does not
duplicate that root-owned `rust-version` declaration.
