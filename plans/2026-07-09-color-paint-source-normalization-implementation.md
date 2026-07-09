# Color Realization And Paint Source Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize render-owned color and paint-source inputs so later image sampling, backgrounds, borders, masks, and text paint hooks can consume one explicit paint contract.

**Architecture:** This phase preserves the Sequence 2 root-resolved symbolic color policy while making it explicit in capability and diagnostic surfaces. It adds deterministic render-local color conversion for a small concrete subset, typed diagnostics for unsupported color spaces and symbolic CSS color payloads, normalized paint layer models around existing `Paint`, and stronger gradient edge-case coverage without changing backend encoding behavior.

**Tech Stack:** Rust 2024, existing `Color`, `Paint`, `Gradient`, `GradientStop`, `StyleColor`, typed diagnostics, Vello/Peniko-compatible concrete paint values, `cargo test`, `cargo clippy`, `cargo fmt`.

---

## Source Scope

Sequence item:

- `plans/2026-07-09-render-css-implementation-sequence.md`, sequence 5.

Matrix coverage:

- `plans/2026-07-08-render-css-support-matrix.md`, paint source rows.
- Solid RGBA paint.
- Symbolic color token root-resolved policy.
- Paint-space color conversion.
- Linear/radial/conic gradients.
- Repeating gradient diagnostics.

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Execution Protocol

Execute this plan through the crate coordinator workflow in `AGENTS.md`:

1. Assign exactly one scoped task, or one tightly coupled task group, to an implementation worker.
2. Workers do not commit.
3. After the worker reports completion, assign a separate clean-context reviewer for that scoped task.
4. Reconcile reviewer findings before moving to the next task.
5. Run the focused checks listed in the task.
6. Commit the scoped task as a logical coordinator commit only after the worker result, reviewer result, and focused checks are clean.
7. Repeat for the next task.
8. After all tasks are committed, run the final checks and assign the final holistic reviewer.

## Current Baseline

Existing render-owned paint behavior includes:

- `Color::try_rgba` for concrete sRGB RGBA channels.
- `Paint::{color, gradient, image}`.
- `Gradient::{try_linear, try_radial, try_sweep}`.
- `GradientStop::try_new`.
- Vello/Peniko encoding for current gradients.
- Sequence 2 model-only `StyleColor`, with tests documenting root-resolved concrete colors.
- Existing degraded-quality diagnostics can name unsupported paint-space conversion.

Current gaps:

- Paint-source capabilities do not name symbolic color policy, paint-space conversion, or repeating gradient support.
- No public render-owned color-input conversion model.
- No typed unsupported diagnostics for symbolic color payloads, `color-mix`, or repeating gradients.
- No normalized paint layer model that later backgrounds/borders/masks can share.
- Gradient tests do not explicitly cover transparent stops, radial/sweep accessors, or repeating-gradient diagnostics.

## File Map

- Modify `src/paint.rs`
  - Add render-owned color input/conversion and normalized paint layer models.
  - Add intentional gradient accessors needed by tests and later phases.
  - Keep fields private and constructors validated.
- Modify `src/capability.rs`
  - Add paint-source capability facts for symbolic color policy, color conversion, and repeating gradients.
- Modify `src/error.rs`
  - Add paint-source operation labels for unsupported symbolic colors, color mixing, color spaces, and repeating gradients.
- Modify `src/style.rs`
  - Add explicit symbolic color ownership policy if needed.
- Modify `src/tests.rs`
  - Add targeted model, diagnostic, conversion, and gradient coverage.
- Modify `src/lib.rs`
  - Export only intentional public paint/color model additions.

Do not modify `src/backend.rs`, `src/encode.rs`, or backend submission behavior in this phase except through tests that exercise existing paint encoding paths.

## Public Model Contract

The implementation must keep these semantic boundaries:

```rust
pub enum SymbolicColorPolicy {
    RootResolvedOnly,
}

pub enum PaintColorSpace {
    Srgb,
    Hsl,
}

pub struct PaintColor {
    space: PaintColorSpace,
    channels: [f32; 4],
}

pub struct NormalizedPaintLayer {
    paint: Paint,
}
```

Notes:

- `StyleColor` remains concrete and root-resolved. Do not add a public symbolic color payload type in this phase.
- `SymbolicColorPolicy::RootResolvedOnly` documents that `currentColor`, system colors, `color-mix()`, relative colors, and unresolved CSS symbolic colors are root-owned or diagnostic-only at the render boundary.
- `PaintColor` is a render-local conversion input for finite concrete values. It should convert `Srgb` and `Hsl` to `Color`. Do not implement Lab/LCH/Oklab/Oklch/HWB/wide-gamut conversion in this phase.
- Unsupported color spaces and repeating gradients are typed diagnostics, not silent fallback.
- `NormalizedPaintLayer` is model-only in this phase and wraps an already valid `Paint`. Do not wire it into `Scene`, `command`, `encode`, or backend behavior.

## Task 1: Paint Capabilities And Symbolic Color Policy

**Files:**

- Modify: `src/capability.rs`
- Modify: `src/error.rs`
- Modify: `src/style.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing capability and policy tests**

Add tests named:

```rust
#[test]
fn paint_capabilities_name_color_policy_and_conversion_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.paint_sources();

    assert!(capabilities.supports_solid_rgba());
    assert!(capabilities.supports_gradients());
    assert!(capabilities.supports_srgb_color_conversion());
    assert!(capabilities.supports_hsl_color_conversion());
    assert_eq!(
        capabilities.symbolic_color_policy(),
        SymbolicColorPolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_unresolved_symbolic_colors());
    assert!(!capabilities.supports_color_mix());
    assert!(!capabilities.supports_repeating_gradients());
}

#[test]
fn symbolic_color_policy_keeps_style_colors_root_resolved() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let style_color = StyleColor::new(color);

    assert_eq!(style_color.color(), color);
    assert_eq!(StyleColor::symbolic_policy(), SymbolicColorPolicy::RootResolvedOnly);
}
```

- [ ] **Step 2: Add capability and policy model surface**

Add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicColorPolicy {
    RootResolvedOnly,
}
```

Extend `PaintSourceCapabilities` with private fields/accessors:

```rust
srgb_color_conversion: bool,
hsl_color_conversion: bool,
unresolved_symbolic_colors: bool,
color_mix: bool,
repeating_gradients: bool,
symbolic_color_policy: SymbolicColorPolicy,

pub const fn supports_srgb_color_conversion(self) -> bool;
pub const fn supports_hsl_color_conversion(self) -> bool;
pub const fn supports_unresolved_symbolic_colors(self) -> bool;
pub const fn supports_color_mix(self) -> bool;
pub const fn supports_repeating_gradients(self) -> bool;
pub const fn symbolic_color_policy(self) -> SymbolicColorPolicy;
```

Initialize `Capabilities::VELLO_0_9` with:

```rust
srgb_color_conversion: true,
hsl_color_conversion: true,
unresolved_symbolic_colors: false,
color_mix: false,
repeating_gradients: false,
symbolic_color_policy: SymbolicColorPolicy::RootResolvedOnly,
```

Add:

```rust
impl StyleColor {
    pub const fn symbolic_policy() -> SymbolicColorPolicy;
}
```

- [ ] **Step 3: Add typed diagnostic labels**

Extend `PrimitiveOperation` with:

```rust
UnresolvedSymbolicColor
ColorMixFunction
UnsupportedColorSpace
RepeatingGradient
```

Labels:

- `UnresolvedSymbolicColor`: `"unresolved symbolic color"`
- `ColorMixFunction`: `"color-mix function"`
- `UnsupportedColorSpace`: `"unsupported color space"`
- `RepeatingGradient`: `"repeating gradient"`

Wire these through `Capabilities::ensure_supported`:

- `UnresolvedSymbolicColor` uses `supports_unresolved_symbolic_colors()`.
- `ColorMixFunction` uses `supports_color_mix()`.
- `UnsupportedColorSpace` returns `false`.
- `RepeatingGradient` uses `supports_repeating_gradients()`.

- [ ] **Step 4: Add diagnostic tests**

Add:

```rust
#[test]
fn unsupported_symbolic_color_inputs_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::UnresolvedSymbolicColor,
        PrimitiveOperation::ColorMixFunction,
        PrimitiveOperation::UnsupportedColorSpace,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::PaintSources, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("symbolic or unsupported color input is not render-resolved");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}
```

- [ ] **Step 5: Export policy type and run focused checks**

Export `SymbolicColorPolicy` from `src/lib.rs`.

Run:

```sh
cargo test -p surgeist-render paint_capabilities
cargo test -p surgeist-render symbolic_color_policy
cargo test -p surgeist-render unsupported_symbolic_color_inputs
```

Expected: all pass.

## Task 2: Deterministic Concrete Color Conversion

**Files:**

- Modify: `src/paint.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing color conversion tests**

Add tests named:

```rust
#[test]
fn paint_colors_convert_srgb_to_concrete_rgba() {
    let color = PaintColor::try_srgb(0.25, 0.5, 0.75, 0.8)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(color, Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap());
}

#[test]
fn paint_colors_convert_hsl_known_vectors() {
    let red = PaintColor::try_hsl(0.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();
    let cyan = PaintColor::try_hsl(180.0, 1.0, 0.5, 1.0)
        .unwrap()
        .to_color()
        .unwrap();

    assert_eq!(red, Color::try_rgba(1.0, 0.0, 0.0, 1.0).unwrap());
    assert_eq!(cyan, Color::try_rgba(0.0, 1.0, 1.0, 1.0).unwrap());
}
```

- [ ] **Step 2: Add paint color model and conversion**

Add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaintColorSpace {
    Srgb,
    Hsl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintColor {
    space: PaintColorSpace,
    channels: [f32; 4],
}
```

Add constructors/accessors:

```rust
impl PaintColor {
    pub fn try_srgb(r: f32, g: f32, b: f32, a: f32) -> Result<Self>;
    pub fn try_hsl(hue_degrees: f32, saturation: f32, lightness: f32, alpha: f32) -> Result<Self>;
    pub const fn space(self) -> PaintColorSpace;
    pub const fn channels(self) -> [f32; 4];
    pub fn to_color(self) -> Result<Color>;
}
```

Validation expectations:

- `try_srgb` validates through `Color::try_rgba`.
- `try_hsl` requires finite hue, finite `0..=1` saturation/lightness/alpha.
- Hue wraps by `rem_euclid(360.0)` during conversion.
- `to_color` returns concrete `Color`.

Use a deterministic HSL-to-RGBA helper with standard CSS-like HSL math.

- [ ] **Step 3: Add invalid conversion tests**

Add:

```rust
#[test]
fn paint_colors_reject_invalid_conversion_inputs() {
    assert!(PaintColor::try_srgb(f32::NAN, 0.0, 0.0, 1.0).is_err());
    assert!(PaintColor::try_hsl(f32::NAN, 1.0, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.5, 0.5, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, -0.1, 1.0).is_err());
    assert!(PaintColor::try_hsl(0.0, 1.0, 0.5, f32::INFINITY).is_err());
}
```

- [ ] **Step 4: Export color types and run focused checks**

Export `PaintColor` and `PaintColorSpace` from `src/lib.rs`.

Run:

```sh
cargo test -p surgeist-render paint_colors
```

Expected: all pass.

## Task 3: Normalized Paint Layer Model

**Files:**

- Modify: `src/paint.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing normalized paint layer tests**

Add tests named:

```rust
#[test]
fn normalized_paint_layers_preserve_valid_paint_sources() {
    let color = NormalizedPaintLayer::try_new(Paint::from(Color::BLACK)).unwrap();
    let gradient_paint = Paint::from(
        Gradient::try_linear(
            Point::try_new(0.0, 0.0).unwrap(),
            Point::try_new(10.0, 0.0).unwrap(),
            vec![
                GradientStop::try_new(0.0, Color::BLACK).unwrap(),
                GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
            ],
        )
        .unwrap(),
    );
    let gradient = NormalizedPaintLayer::try_new(gradient_paint.clone()).unwrap();

    assert_eq!(color.paint(), &Paint::from(Color::BLACK));
    assert_eq!(gradient.paint(), &gradient_paint);
}

#[test]
fn normalized_paint_layers_reject_invalid_paint_sources() {
    let error = Gradient::try_linear(
        Point::new(f64::NAN, 0.0),
        Point::try_new(1.0, 0.0).unwrap(),
        vec![GradientStop::try_new(0.0, Color::BLACK).unwrap()],
    )
    .expect_err("invalid gradient construction should fail before paint layer");

    assert_eq!(error.code, ErrorCode::InvalidInput);
}
```

The second test intentionally verifies invalid paint sources still fail before they become normalized layers. Do not add invalid constructors that bypass existing validation.

- [ ] **Step 2: Add normalized paint layer model**

Add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedPaintLayer {
    paint: Paint,
}
```

Add:

```rust
impl NormalizedPaintLayer {
    pub fn try_new(paint: Paint) -> Result<Self>;
    pub const fn paint(&self) -> &Paint;
}
```

`try_new` validates through existing `validate_paint(&paint)` and stores the paint unchanged.

- [ ] **Step 3: Run focused checks**

Export `NormalizedPaintLayer` from `src/lib.rs`.

Run:

```sh
cargo test -p surgeist-render normalized_paint_layers
```

Expected: all pass.

## Task 4: Gradient Semantics And Repeating Diagnostics

**Files:**

- Modify: `src/paint.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add gradient accessor tests**

Add tests named:

```rust
#[test]
fn gradients_expose_render_ready_geometry_and_stops() {
    let stops = vec![
        GradientStop::try_new(0.0, Color::BLACK).unwrap(),
        GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
    ];
    let linear = Gradient::try_linear(
        Point::try_new(1.0, 2.0).unwrap(),
        Point::try_new(3.0, 4.0).unwrap(),
        stops.clone(),
    )
    .unwrap();
    let radial = Gradient::try_radial(Point::try_new(5.0, 6.0).unwrap(), 7.0, stops.clone()).unwrap();
    let sweep = Gradient::try_sweep(Point::try_new(8.0, 9.0).unwrap(), stops.clone()).unwrap();

    assert_eq!(linear.stops(), stops.as_slice());
    assert_eq!(linear.linear_points(), Some((Point::try_new(1.0, 2.0).unwrap(), Point::try_new(3.0, 4.0).unwrap())));
    assert_eq!(radial.radial_geometry(), Some((Point::try_new(5.0, 6.0).unwrap(), 7.0)));
    assert_eq!(sweep.sweep_center(), Some(Point::try_new(8.0, 9.0).unwrap()));
}

#[test]
fn gradients_preserve_transparent_stops() {
    let stop = GradientStop::try_new(0.5, Color::TRANSPARENT).unwrap();

    assert_eq!(stop.color(), Color::TRANSPARENT);
}
```

- [ ] **Step 2: Add gradient accessors**

Add to `impl Gradient`:

```rust
pub fn stops(&self) -> &[GradientStop];
pub const fn linear_points(&self) -> Option<(Point, Point)>;
pub const fn radial_geometry(&self) -> Option<(Point, f64)>;
pub const fn sweep_center(&self) -> Option<Point>;
```

Keep `GradientKind` crate-private.

- [ ] **Step 3: Add repeating gradient diagnostic test**

Add:

```rust
#[test]
fn repeating_gradients_report_typed_diagnostics() {
    let unsupported = UnsupportedPrimitive::new(
        PrimitiveFamily::PaintSources,
        PrimitiveOperation::RepeatingGradient,
    );

    let error = Capabilities::VELLO_0_9
        .ensure_supported(unsupported)
        .expect_err("repeating gradients require later normalization");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(error.unsupported_primitive(), Some(unsupported));
}
```

- [ ] **Step 4: Run focused checks**

Run:

```sh
cargo test -p surgeist-render gradients
cargo test -p surgeist-render repeating_gradients
```

Expected: all pass.

## Task 5: Paint Integration Cleanup And Checks

**Files:**

- Modify: `src/tests.rs`
- Modify other files only if a focused review finds an integration bug from Tasks 1-4.

- [ ] **Step 1: Add existing backend paint regression tests**

Add tests named:

```rust
#[test]
fn concrete_color_paint_renders_without_color_realization() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), Color::try_rgba(0.25, 0.5, 0.75, 1.0).unwrap());

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("concrete color paint should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert!(pixel_alpha(&output, 0, 0) > 0);
}

#[test]
fn gradient_paint_renders_with_transparent_stop() {
    let gradient = Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(2.0, 0.0).unwrap(),
        vec![
            GradientStop::try_new(0.0, Color::BLACK).unwrap(),
            GradientStop::try_new(1.0, Color::TRANSPARENT).unwrap(),
        ],
    )
    .unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(2.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.fill(Rect::new(0.0, 0.0, 2.0, 2.0), gradient);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("gradient paint should render");
}
```

- [ ] **Step 2: Verify no backend or dependency drift**

Run:

```sh
git diff -- src/backend.rs src/encode.rs src/renderer.rs
git diff -- Cargo.toml Cargo.lock
git diff -- Cargo.toml | rg 'path = "../' || true
```

Expected:

- No backend/renderer/encoder behavior changes unless an earlier task review explicitly required them.
- No dependency changes in `Cargo.toml` or `Cargo.lock`.
- No new sibling path dependencies. The existing optional `surgeist-window` path dependency is pre-existing and should remain untouched.

- [ ] **Step 3: Run sequence-item focused checks**

Run:

```sh
cargo test -p surgeist-render paint
cargo test -p surgeist-render color
cargo test -p surgeist-render gradient
cargo test -p surgeist-render unsupported
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Expected: all pass.

## Final Review Gate

After all task-scoped worker/reviewer cycles and coordinator commits are complete, assign a final clean-context holistic reviewer to inspect:

- this plan
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `git diff` for this implementation plan
- crate boundary and absence of sibling dependencies
- root-resolved symbolic color policy
- deterministic color conversion
- normalized paint layer model
- gradient accessors and repeating gradient diagnostics
- tests and required checks

Required final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Completion for this sequence item requires a clean holistic review and all required final checks passing.
