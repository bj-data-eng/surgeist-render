# Surgeist Render Rust Modeling Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `surgeist-render` into closer compliance with `guidance/surgeist-rust-modeling-guide.md` by replacing validation-by-convention with typed construction boundaries for render geometry, draw commands, resource handles, capabilities, and surface lifecycle.

**Architecture:** Replace loose public construction paths with crate-owned semantic models and typed front doors. Backwards compatibility shims are not required at this phase; intentional public API changes are acceptable when they make invalid render states harder to express.

**Tech Stack:** Rust 2024, Vello 0.9, wgpu 29, kurbo 0.13, peniko 0.6, proptest 1.11, root-owned `public-api` generator.

---

## Review Findings

The crate already has a clean front door in `src/lib.rs` and mostly private modules, but several important render concepts are still loose enough that invalid states are ordinary values:

- `src/geometry.rs` exposes raw `f64` geometry fields and validates many of them only during scene encoding through `src/validation.rs`.
- `src/shape.rs`, `src/paint.rs`, `src/layer.rs`, `src/text.rs`, and `src/surface.rs` expose public fields whose invariants are currently prose-and-runtime-error contracts.
- `src/scene.rs` stores authored commands directly in `Command`, while `src/encode.rs` performs validation, normalization, capability checks, and Vello lowering in one pass.
- `src/image.rs`, `src/text.rs`, and `src/renderer.rs` use raw `u64`/`usize` IDs for resource identity.
- `src/surface.rs` and `src/backend.rs` model lifecycle with booleans and `Option` fields: `available`, `valid`, `resizing`, `texture`, `view`, `pending_physical_size`.
- Unsupported features such as layer masks, filters, non-solid shadows, and path-aligned strokes are reported from scattered encoder branches rather than a single capability contract.

This plan intentionally avoids sibling crate edits. The optional `surgeist-window` path dependency remains behind `render-window`; workers must not modify `/Users/codex/Development/surgeist-window`.

## File Structure

- Modify `src/error.rs` to add typed validation and capability error payloads.
- Modify `src/geometry.rs` to introduce semantic geometry constructors and normalized physical sizing.
- Modify `src/shape.rs`, `src/paint.rs`, `src/layer.rs`, and `src/text.rs` to add validated construction paths for render-facing values.
- Modify `src/image.rs` to introduce `ImageId`.
- Add `src/capability.rs` for renderer capability reporting and unsupported-operation classification.
- Add `src/command.rs` for normalized render commands and normalized layer operations.
- Modify `src/scene.rs` to retain authored commands but expose normalization into `command::RenderCommands`.
- Modify `src/encode.rs` to lower normalized commands and stop duplicating validation/capability checks.
- Modify `src/surface.rs` and `src/backend.rs` to replace lifecycle booleans/options with typed surface/backend state.
- Modify `src/renderer.rs` to call normalization before encoding and to use typed IDs/state.
- Modify `src/stats.rs` to count normalized command/resource IDs.
- Modify `src/tests.rs` with focused regression tests for each task.
- Regenerate `api/public-api.txt` only when public API changes are intentional, using the root-owned API generator from the root `surgeist` repo:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
```

## Coordinator Execution Protocol

Every implementation task below must be run through this crate's coordinator workflow from `AGENTS.md`:

1. Check `git status --short --branch` before assigning the task.
2. Assign exactly one implementation worker to the scoped task or tightly coupled task group. Tell the worker they are not alone in the codebase, must not revert others' work, and must report changed files, tests run, skipped checks, and final git status.
3. Wait for the worker result.
4. Assign a separate reviewer to inspect only that task's changes against this plan, `guidance/surgeist-rust-modeling-guide.md`, crate boundaries, generated-artifact rules, and focused tests.
5. Reconcile reviewer findings. Critical and Important findings must be fixed and re-reviewed before the task can be committed.
6. Run the task's focused verification command as coordinator.
7. Commit the task as a traceable logical point only after the worker/reviewer cycle is clean.

The "Review gate and commit" step in each task means this entire protocol, not a self-review shortcut.

## Task 1: Typed Modeling Error Details

**Files:**
- Modify: `src/error.rs`
- Modify: `src/validation.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add focused error-detail tests**

Add these tests near the existing invalid-input tests in `src/tests.rs`:

```rust
#[test]
fn invalid_value_errors_name_rejected_value() {
    let error = Error::invalid_value(
        "rectangle width",
        f64::NAN,
        "must be finite and non-negative",
    );

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(
        error.message.contains("rectangle width"),
        "error should name the rejected field: {}",
        error.message
    );
    assert!(
        error.message.contains("NaN"),
        "error should include the rejected value: {}",
        error.message
    );
}

#[test]
fn unsupported_operation_errors_name_capability() {
    let capability = UnsupportedCapability::LayerMask;
    let error = Error::unsupported_capability(capability);

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert!(
        error.message.contains("layer mask"),
        "message should name the unsupported capability: {}",
        error.message
    );
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render invalid_value_errors_name_rejected_value unsupported_operation_errors_name_capability
```

Expected: fail because `Error::invalid_value`, `UnsupportedCapability`, and `Error::unsupported_capability` do not exist yet.

- [ ] **Step 3: Implement typed error helpers**

Add this public enum and helper to `src/error.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedCapability {
    LayerFilter,
    LayerMask,
    NonSolidShadowPaint,
    PathStrokeAlignment,
    WebCanvasSurface,
}

impl UnsupportedCapability {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LayerFilter => "layer filter",
            Self::LayerMask => "layer mask",
            Self::NonSolidShadowPaint => "non-solid shadow paint",
            Self::PathStrokeAlignment => "inside/outside path stroke alignment",
            Self::WebCanvasSurface => "web canvas surface",
        }
    }
}

impl Error {
    #[must_use]
    pub fn invalid_value(name: impl Into<String>, value: impl std::fmt::Display, rule: &'static str) -> Self {
        Self::new(
            ErrorCode::InvalidInput,
            format!("{} value {value} is invalid: {rule}", name.into()),
        )
    }

    #[must_use]
    pub fn unsupported_capability(capability: UnsupportedCapability) -> Self {
        Self::new(
            ErrorCode::UnsupportedBackend,
            format!("renderer capability is unsupported: {}", capability.label()),
        )
    }
}
```

Export `UnsupportedCapability` from `src/lib.rs`:

```rust
pub use error::{Error, ErrorCode, Result, UnsupportedCapability};
```

- [ ] **Step 4: Route validation helpers through typed error helpers**

In `src/validation.rs`, update primitive validation helpers to call `Error::invalid_value`. For example:

```rust
pub(crate) fn validate_positive_f64(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and greater than 0",
        ));
    }
    Ok(())
}
```

Apply the same shape to `validate_non_negative_f64`, `validate_finite_f64`, and color-channel validation.

- [ ] **Step 5: Run focused tests**

Run:

```sh
cargo test -p surgeist-render invalid_value_errors_name_rejected_value unsupported_operation_errors_name_capability rejects_invalid_surface_geometry rejects_malformed_scene_values
```

Expected: all listed tests pass.

- [ ] **Step 6: Refresh public API artifact**

Because this task intentionally exposes `UnsupportedCapability` and new `Error` methods, run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: `api/public-api.txt` includes `UnsupportedCapability` and the new `Error` methods only.

- [ ] **Step 7: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/error.rs src/validation.rs src/lib.rs src/tests.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/error.rs src/validation.rs src/lib.rs src/tests.rs api/public-api.txt
git commit -m "model typed render errors"
```

## Task 2: Validated Geometry Front Doors

**Files:**
- Modify: `src/geometry.rs`
- Modify: `src/validation.rs`
- Modify: `src/backend.rs`
- Modify: `src/encode.rs`
- Modify: `src/image.rs`
- Modify: `src/layer.rs`
- Modify: `src/paint.rs`
- Modify: `src/renderer.rs`
- Modify: `src/scene.rs`
- Modify: `src/shape.rs`
- Modify: `src/stats.rs`
- Modify: `src/surface.rs`
- Modify: `src/text.rs`
- Test: `src/tests.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add construction-boundary tests**

Add these tests to `src/tests.rs`:

```rust
#[test]
fn geometry_try_constructors_reject_invalid_values() {
    assert!(Point::try_new(f64::NAN, 0.0).is_err());
    assert!(Size::try_new(-1.0, 1.0).is_err());
    assert!(Rect::try_new(0.0, 0.0, 1.0, f64::INFINITY).is_err());
    assert!(Radii::try_all(-0.1).is_err());
    assert!(Transform::try_new([1.0, 0.0, 0.0, f64::NAN, 0.0, 0.0]).is_err());
}

#[test]
fn physical_size_try_from_logical_size_rejects_invalid_scale() {
    let error = PhysicalSize::try_from_logical(Size::try_new(10.0, 10.0).unwrap(), 0.0)
        .expect_err("scale zero should be rejected before conversion");
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[test]
fn physical_size_try_from_logical_size_rejects_u32_overflow() {
    let error = PhysicalSize::try_from_logical(
        Size::try_new(f64::from(u32::MAX), 1.0).unwrap(),
        2.0,
    )
        .expect_err("physical device pixels should fit in u32");
    assert_eq!(error.code, ErrorCode::InvalidInput);
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render geometry_try_constructors_reject_invalid_values physical_size_try_from_logical_size_rejects_invalid_scale physical_size_try_from_logical_size_rejects_u32_overflow
```

Expected: fail because these constructors do not exist.

- [ ] **Step 3: Add fallible constructors to geometry types**

In `src/geometry.rs`, import crate errors:

```rust
use super::{Error, Result};
```

Replace loose public fields/constructors with private fields, fallible constructors, and accessors where invalid values are currently ordinary values. For example, `Size` should become:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    width: f64,
    height: f64,
}
```

Apply the same pattern to `Point`, `Rect`, `Radii`, `Transform`, and `PhysicalSize`: keep fields private, add read-only accessors such as `width()`, `height()`, `x()`, `y()`, `origin()`, `size()`, and `as_array()`, and use fallible constructors for public construction. `PhysicalSize` should also have `PhysicalSize::new(width: u32, height: u32) -> Self` because `u32` device-pixel dimensions are already range-safe.

```rust
impl Point {
    pub fn try_new(x: f64, y: f64) -> Result<Self> {
        if !x.is_finite() {
            return Err(Error::invalid_value("point x", x, "must be finite"));
        }
        if !y.is_finite() {
            return Err(Error::invalid_value("point y", y, "must be finite"));
        }
        Ok(Self { x, y })
    }
}

impl Size {
    pub fn try_new(width: f64, height: f64) -> Result<Self> {
        if !width.is_finite() || width < 0.0 {
            return Err(Error::invalid_value("size width", width, "must be finite and non-negative"));
        }
        if !height.is_finite() || height < 0.0 {
            return Err(Error::invalid_value("size height", height, "must be finite and non-negative"));
        }
        Ok(Self { width, height })
    }
}

impl Rect {
    pub fn try_new(x: f64, y: f64, width: f64, height: f64) -> Result<Self> {
        Ok(Self {
            origin: Point::try_new(x, y)?,
            size: Size::try_new(width, height).map_err(|_| {
                if !width.is_finite() || width < 0.0 {
                    Error::invalid_value("rectangle width", width, "must be finite and non-negative")
                } else {
                    Error::invalid_value("rectangle height", height, "must be finite and non-negative")
                }
            })?,
        })
    }
}

impl Radii {
    pub fn try_all(radius: f64) -> Result<Self> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::invalid_value("corner radius", radius, "must be finite and non-negative"));
        }
        Ok(Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        })
    }
}

impl Transform {
    pub fn try_new(values: [f64; 6]) -> Result<Self> {
        for value in values {
            if !value.is_finite() {
                return Err(Error::invalid_value("transform", value, "must contain only finite values"));
            }
        }
        Ok(Self(values))
    }
}
```

- [ ] **Step 4: Add typed physical-size conversion**

In `src/geometry.rs`, add:

```rust
impl PhysicalSize {
    pub fn try_from_logical(size: Size, scale: f64) -> Result<Self> {
        Size::try_new(size.width(), size.height())?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::invalid_value("surface scale", scale, "must be finite and greater than 0"));
        }
        let width = size.width() * scale;
        let height = size.height() * scale;
        if width > f64::from(u32::MAX) {
            return Err(Error::invalid_value("physical width", width, "must fit in u32 device pixels"));
        }
        if height > f64::from(u32::MAX) {
            return Err(Error::invalid_value("physical height", height, "must fit in u32 device pixels"));
        }
        Ok(Self {
            width: width.round() as u32,
            height: height.round() as u32,
        })
    }
}

pub(crate) fn physical_size(size: Size, scale: f64) -> PhysicalSize {
    PhysicalSize::try_from_logical(size, scale)
        .expect("callers validate surface size and scale before physical conversion")
}
```

Update all crate internals to use accessors instead of direct geometry fields, for example `size.width()` instead of `size.width` and `rect.origin()` instead of `rect.origin`.

After Task 2, update later task snippets and existing crate tests mechanically to use the new fallible geometry constructors, for example:

```rust
let size = Size::try_new(10.0, 10.0).unwrap();
let rect = Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap();
let point = Point::try_new(0.0, 0.0).unwrap();
```

- [ ] **Step 5: Run focused tests**

Run:

```sh
cargo test -p surgeist-render geometry_try_constructors_reject_invalid_values physical_size_try_from_logical_size_rejects_invalid_scale physical_size_try_from_logical_size_rejects_u32_overflow rejects_invalid_surface_geometry
```

Expected: pass.

- [ ] **Step 6: Refresh API artifact if public constructors changed**

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: public geometry fields are removed, geometry accessors/fallible constructors are added, and no unrelated public APIs change.

- [ ] **Step 7: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/geometry.rs src/validation.rs src/backend.rs src/encode.rs src/image.rs src/layer.rs src/paint.rs src/renderer.rs src/scene.rs src/shape.rs src/stats.rs src/surface.rs src/text.rs src/tests.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/geometry.rs src/validation.rs src/backend.rs src/encode.rs src/image.rs src/layer.rs src/paint.rs src/renderer.rs src/scene.rs src/shape.rs src/stats.rs src/surface.rs src/text.rs src/tests.rs api/public-api.txt
git commit -m "model validated render geometry"
```

## Task 2A: Validated Draw Value Front Doors

**Files:**
- Modify: `src/shape.rs`
- Modify: `src/paint.rs`
- Modify: `src/layer.rs`
- Modify: `src/text.rs`
- Modify: `src/validation.rs`
- Modify: `src/scene.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add construction-boundary tests**

Add these tests to `src/tests.rs`:

```rust
#[test]
fn draw_value_try_constructors_reject_invalid_values() {
    assert!(Shape::try_circle(Point::try_new(0.0, 0.0).unwrap(), -1.0).is_err());
    assert!(Color::try_rgba(2.0, 0.0, 0.0, 1.0).is_err());
    assert!(Stroke::try_new(0.0).is_err());
    assert!(Dash::try_new(0.0, &[1.0, f64::NAN]).is_err());
    assert!(GradientStop::try_new(1.5, Color::BLACK).is_err());
    assert!(Gradient::try_linear(
        Point::try_new(0.0, 0.0).unwrap(),
        Point::try_new(1.0, 1.0).unwrap(),
        vec![],
    )
    .is_err());
    assert!(Layer::new().try_opacity(f32::NAN).is_err());
    assert!(Shadow::try_new(Point::try_new(0.0, 0.0).unwrap(), -1.0, 0.0, Color::BLACK).is_err());
    assert!(TextGlyph::try_new(1, 0.0, f32::NAN, 1.0).is_err());
    assert!(TextRun::try_new(
        FontRef::new(1),
        -1.0,
        Transform::identity(),
        TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap(),
        &[],
    )
    .is_err());
}

#[test]
fn draw_value_constructors_preserve_valid_values() {
    let stroke = Stroke::try_new(2.0).unwrap().align(StrokeAlign::Inside);
    let stop = GradientStop::try_new(0.5, Color::BLACK).unwrap();
    let layer = Layer::new().try_opacity(0.5).unwrap();
    let text_paint = TextPaint::try_fill(Paint::color(Color::BLACK)).unwrap();
    let glyph = TextGlyph::try_new(7, 1.0, 2.0, 3.0).unwrap();
    let glyphs = [glyph];
    let text_run = TextRun::try_new(
        FontRef::new(1),
        12.0,
        Transform::identity(),
        text_paint.clone(),
        &glyphs,
    )
    .unwrap();

    assert_eq!(stroke.width(), 2.0);
    assert_eq!(stop.offset(), 0.5);
    assert_eq!(layer.opacity(), 0.5);
    assert_eq!(text_paint.fill(), &Paint::color(Color::BLACK));
    assert_eq!(glyph.id(), 7);
    assert_eq!(text_run.size(), 12.0);
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render draw_value_try_constructors_reject_invalid_values draw_value_constructors_preserve_valid_values
```

Expected: fail because the fallible constructors and accessors do not exist yet.

- [ ] **Step 3: Make invalidable draw values private-representation types**

Change `Shape`, `Color`, `Paint`, `Gradient`, `GradientStop`, `Stroke`, `Dash`, `Layer`, `Shadow`, `TextPaint`, `TextGlyph`, and `TextRun` so ordinary callers cannot construct invalid values through public enum variants or public fields. Because backwards compatibility shims are not required, replace public enum variants that carry invalidable data with private representations plus validated constructors and read-only accessors. Route render code and tests through fallible constructors.

Use public structs with private enum storage when callers need a closed value family. For example, replace public `Shape` variants with:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    kind: ShapeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShapeKind {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(Path),
}

impl Shape {
    #[must_use]
    pub fn rect(rect: Rect) -> Self {
        Self {
            kind: ShapeKind::Rect(rect),
        }
    }

    pub(crate) fn kind(&self) -> &ShapeKind {
        &self.kind
    }
}
```

Apply the same shape to `Paint` / `PaintKind` and `Gradient` / `GradientKind`, so callers use `Paint::color`, `Paint::gradient`, `Paint::image`, `Gradient::try_linear`, `Gradient::try_radial`, and `Gradient::try_sweep` instead of public variants. `Color` should have private channels and `Color::try_rgba`; keep constants like `BLACK` and `TRANSPARENT` as validated constants. `TextPaint` should have a private `fill` field and `TextPaint::try_fill`.

In `src/shape.rs`, use this shape for `Stroke`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    width: f64,
    join: LineJoin,
    start_cap: LineCap,
    end_cap: LineCap,
    miter_limit: f64,
    dash: Option<Dash>,
    align: StrokeAlign,
}

impl Stroke {
    pub fn try_new(width: f64) -> Result<Self> {
        validate_positive_f64(width, "stroke width")?;
        Ok(Self {
            width,
            join: LineJoin::Miter,
            start_cap: LineCap::Butt,
            end_cap: LineCap::Butt,
            miter_limit: 4.0,
            dash: None,
            align: StrokeAlign::Center,
        })
    }

    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    #[must_use]
    pub const fn align(mut self, align: StrokeAlign) -> Self {
        self.align = align;
        self
    }

    #[must_use]
    pub const fn join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    #[must_use]
    pub const fn caps(mut self, start: LineCap, end: LineCap) -> Self {
        self.start_cap = start;
        self.end_cap = end;
        self
    }

    pub fn try_miter_limit(mut self, miter_limit: f64) -> Result<Self> {
        validate_positive_f64(miter_limit, "stroke miter limit")?;
        self.miter_limit = miter_limit;
        Ok(self)
    }

    pub fn try_dash(mut self, dash: Dash) -> Result<Self> {
        validate_dash(dash)?;
        self.dash = Some(dash);
        Ok(self)
    }

    #[must_use]
    pub const fn join_kind(self) -> LineJoin {
        self.join
    }

    #[must_use]
    pub const fn start_cap(self) -> LineCap {
        self.start_cap
    }

    #[must_use]
    pub const fn end_cap(self) -> LineCap {
        self.end_cap
    }

    #[must_use]
    pub const fn miter_limit(self) -> f64 {
        self.miter_limit
    }

    #[must_use]
    pub const fn dash(self) -> Option<Dash> {
        self.dash
    }

    #[must_use]
    pub const fn align_kind(self) -> StrokeAlign {
        self.align
    }

    pub(crate) const fn parts(self) -> (f64, LineJoin, LineCap, LineCap, f64, Option<Dash>, StrokeAlign) {
        (
            self.width,
            self.join,
            self.start_cap,
            self.end_cap,
            self.miter_limit,
            self.dash,
            self.align,
        )
    }
}
```

Apply the same private-field pattern to `Dash::try_new`, `GradientStop::try_new`, `Filter::try_blur`, `Layer`, `Shadow::try_new`, `TextPaint::try_fill`, `TextGlyph::try_new`, and `TextRun::try_new`: validate at construction, keep private fields, and expose read-only accessors used by encoding/stats. Add `validate_dash(dash: Dash) -> Result<()>` and `validate_filter(filter: Filter) -> Result<()>` in `src/validation.rs` so stroke/layer builders and aggregate validation share the same invariant checks. Replace public `Filter::Blur { radius }` with a private representation and `Filter::try_blur(radius)`.

For `TextRun`, replace public fields with a validated borrowed run:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct TextRun<'a> {
    font: FontRef<'a>,
    size: f32,
    transform: Transform,
    paint: TextPaint,
    glyphs: &'a [TextGlyph],
}

impl<'a> TextRun<'a> {
    pub fn try_new(
        font: FontRef<'a>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: &'a [TextGlyph],
    ) -> Result<Self> {
        validate_text_run(size, transform, glyphs)?;
        validate_paint(paint.fill())?;
        Ok(Self {
            font,
            size,
            transform,
            paint,
            glyphs,
        })
    }

    pub fn font(&self) -> &FontRef<'a> {
        &self.font
    }

    pub fn size(&self) -> f32 {
        self.size
    }

    pub fn transform(&self) -> Transform {
        self.transform
    }

    pub fn paint(&self) -> &TextPaint {
        &self.paint
    }

    pub fn glyphs(&self) -> &'a [TextGlyph] {
        self.glyphs
    }
}
```

For `Layer`, add validated builders/accessors for every configurable field so callers and `Scene` never need struct literals:

```rust
impl Layer {
    pub fn try_clip(mut self, clip: Shape) -> Result<Self> {
        validate_shape(&clip)?;
        self.clip = Some(clip);
        Ok(self)
    }

    pub fn try_mask(mut self, mask: Shape) -> Result<Self> {
        validate_shape(&mask)?;
        self.mask = Some(mask);
        Ok(self)
    }

    pub fn try_filter(mut self, filter: Filter) -> Result<Self> {
        validate_filter(filter)?;
        self.filter = Some(filter);
        Ok(self)
    }

    pub fn try_transform(mut self, transform: Transform) -> Result<Self> {
        validate_transform(transform, "layer transform")?;
        self.transform = transform;
        Ok(self)
    }

    pub fn blend(mut self, blend: BlendMode) -> Self {
        self.blend = blend;
        self
    }

    pub fn try_opacity(mut self, opacity: f32) -> Result<Self> {
        if !opacity.is_finite() {
            return Err(Error::invalid_value("layer opacity", opacity, "must be finite"));
        }
        self.opacity = opacity;
        Ok(self)
    }

    pub fn clip(&self) -> Option<&Shape> {
        self.clip.as_ref()
    }

    pub fn mask(&self) -> Option<&Shape> {
        self.mask.as_ref()
    }

    pub fn filter(&self) -> Option<Filter> {
        self.filter
    }

    pub fn transform(&self) -> Transform {
        self.transform
    }

    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    pub fn blend(&self) -> BlendMode {
        self.blend
    }
}
```

- [ ] **Step 4: Add shape and gradient constructors**

In `src/shape.rs`, add:

```rust
impl Shape {
    pub fn try_circle(center: Point, radius: f64) -> Result<Self> {
        validate_point(center, "circle center")?;
        validate_non_negative_f64(radius, "circle radius")?;
        Ok(Self {
            kind: ShapeKind::Circle { center, radius },
        })
    }

    pub fn try_ellipse(center: Point, radii: Size) -> Result<Self> {
        validate_point(center, "ellipse center")?;
        validate_size(radii, "ellipse radii")?;
        Ok(Self {
            kind: ShapeKind::Ellipse { center, radii },
        })
    }

    pub fn try_rounded_rect(rect: Rect, radii: Radii) -> Result<Self> {
        validate_rect(rect, "rounded rectangle")?;
        validate_radii(radii, "rounded rectangle radii")?;
        Ok(Self {
            kind: ShapeKind::RoundedRect { rect, radii },
        })
    }
}
```

In `src/paint.rs`, add fallible constructors for `Gradient` variants that validate points, radii, and stops before storing them. Reject empty stop lists because a gradient with no stops has no renderable color contract.

- [ ] **Step 5: Update internal callers to use accessors**

Update `src/scene.rs`, `src/encode.rs`, `src/stats.rs`, `src/validation.rs`, and tests so they no longer access privatized fields or public enum variants directly. For example, replace `stroke.width` with `stroke.width()` or destructure through `stroke.parts()` in encoder-only code, and change `validate_shape` / `validate_paint` to inspect `shape.kind()` / `paint.kind()`. Update `Scene::transform` and `Scene::clip` to use `Layer::new().try_transform(transform)?` and `Layer::new().try_clip(shape)?`; if keeping infallible `Scene::transform` / `Scene::clip` methods is undesirable after removing shims, replace them with `try_transform` / `try_clip` methods that return `Result<&mut Self>`.

- [ ] **Step 6: Run focused tests**

Run:

```sh
cargo test -p surgeist-render draw_value_try_constructors_reject_invalid_values draw_value_constructors_preserve_valid_values rejects_malformed_scene_values layer_default_is_visible text_run_requires_font_data
```

Expected: pass.

- [ ] **Step 7: Refresh API artifact**

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: public field entries for privatized draw values disappear, and new constructors/accessors appear.

- [ ] **Step 8: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/shape.rs src/paint.rs src/layer.rs src/text.rs src/validation.rs src/scene.rs src/tests.rs src/lib.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/shape.rs src/paint.rs src/layer.rs src/text.rs src/validation.rs src/scene.rs src/tests.rs src/lib.rs api/public-api.txt
git commit -m "model validated draw values"
```

## Task 3: Resource Handle Newtypes

**Files:**
- Modify: `src/image.rs`
- Modify: `src/text.rs`
- Modify: `src/stats.rs`
- Modify: `src/renderer.rs`
- Modify: `src/scene.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add ID-type tests**

Add these tests to `src/tests.rs`:

```rust
#[test]
fn image_ids_are_typed_resource_handles() {
    let image = Image::from_rgba(
        Size::try_new(1.0, 1.0).unwrap(),
        Arc::<[u8]>::from([0, 0, 0, 255]),
    )
    .unwrap();
    let id = image.id();

    assert_eq!(id.get(), image.id().get());
}

#[test]
fn font_refs_use_typed_font_ids() {
    let font = FontRef::new(FontId::new(42));

    assert_eq!(font.id(), FontId::new(42));
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render image_ids_are_typed_resource_handles font_refs_use_typed_font_ids
```

Expected: fail because `Image::id()` returns `u64`, `FontId` does not exist, and `FontRef::id()` does not exist.

- [ ] **Step 3: Add `ImageId`**

In `src/image.rs`, add:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ImageId(u64);

impl ImageId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
```

Change `Image` to store `id: ImageId`, wrap `stable_hash(...)` with `ImageId::new(...)`, and update `Image::id`:

```rust
#[must_use]
pub const fn id(&self) -> ImageId {
    self.id
}
```

When constructing `peniko::Blob`, pass `id.get()`.

- [ ] **Step 4: Add `FontId`**

In `src/text.rs`, add:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FontId(u64);

impl FontId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
```

Change `FontRef` to store a private typed ID and keep `name` public for the existing authored font-label flow:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct FontRef<'a> {
    id: FontId,
    pub name: Option<Cow<'a, str>>,
    pub(crate) data: Option<FontData>,
}
```

Update its constructor and accessor:

```rust
impl<'a> FontRef<'a> {
    #[must_use]
    pub fn new(id: impl Into<FontId>) -> Self {
        Self {
            id: id.into(),
            name: None,
            data: None,
        }
    }

    #[must_use]
    pub const fn id(&self) -> FontId {
        self.id
    }

    #[must_use]
    pub fn named(mut self, name: impl Into<Cow<'a, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_data(mut self, data: FontData) -> Self {
        self.data = Some(data);
        self
    }

    pub(crate) fn to_owned_static(&self) -> FontRef<'static> {
        FontRef {
            id: self.id,
            name: self.name.as_ref().map(|name| Cow::Owned(name.clone().into_owned())),
            data: self.data.clone(),
        }
    }
}

impl From<u64> for FontId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}
```

- [ ] **Step 5: Export resource ID types**

In `src/lib.rs`, update exports:

```rust
pub use image::{Extend, Image, ImageBuffer, ImageFit, ImageId, ImageQuality};
pub use text::{FontData, FontId, FontRef, TextGlyph, TextPaint, TextRun};
```

- [ ] **Step 6: Update internal hash sets**

Change `HashSet<u64>` to `HashSet<ImageId>` in `src/renderer.rs` and `src/stats.rs`. Keep `uploaded_bytes` as `u64` because that is a byte count, not an identity.

- [ ] **Step 7: Update scene text command copy**

In `src/scene.rs`, copy text-run font identity through the crate-owned helper instead of constructing `FontRef` by struct literal:

```rust
font: run.font.to_owned_static(),
```

- [ ] **Step 8: Run focused tests**

Run:

```sh
cargo test -p surgeist-render image_ids_are_typed_resource_handles font_refs_use_typed_font_ids warm_image_reuse_reports_cache_hit text_run_requires_font_data
```

Expected: pass.

- [ ] **Step 9: Refresh API artifact**

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: public API shows `ImageId`, `FontId`, and changed `Image::id` / `FontRef` ID APIs.

- [ ] **Step 10: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/image.rs src/text.rs src/stats.rs src/renderer.rs src/scene.rs src/tests.rs src/lib.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/image.rs src/text.rs src/stats.rs src/renderer.rs src/scene.rs src/tests.rs src/lib.rs api/public-api.txt
git commit -m "model render resource identities"
```

## Task 4: Renderer Capability Contract

**Files:**
- Create: `src/capability.rs`
- Modify: `src/lib.rs`
- Modify: `src/encode.rs`
- Modify: `src/renderer.rs`
- Test: `src/tests.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add capability tests**

Add these tests to `src/tests.rs`:

```rust
#[test]
fn renderer_reports_backend_capabilities() {
    let renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let capabilities = renderer.capabilities();

    assert!(!capabilities.supports_layer_masks());
    assert!(!capabilities.supports_layer_filters());
    assert!(!capabilities.supports_inside_outside_path_strokes());
    assert_eq!(
        capabilities.supports_web_canvas_surfaces(),
        cfg!(all(feature = "render-web", target_arch = "wasm32"))
    );
}

#[test]
fn capabilities_map_unsupported_operations_to_typed_errors() {
    let capabilities = Capabilities::VELLO_0_9;

    let error = capabilities
        .ensure(UnsupportedCapability::LayerMask)
        .expect_err("layer masks are not supported in this milestone");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert!(error.message.contains("layer mask"));
}

#[test]
fn layer_masks_report_capability_error() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(4.0, 2.0).unwrap(), 1.0)
        .unwrap();
    let mut scene = Scene::new();

    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("unsupported mask should fail render");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert!(error.message.contains("layer mask"));
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render renderer_reports_backend_capabilities capabilities_map_unsupported_operations_to_typed_errors layer_masks_report_capability_error
```

Expected: fail because `Renderer::capabilities` and `Capabilities` do not exist yet.

- [ ] **Step 3: Add capability model**

Create `src/capability.rs`:

```rust
use super::UnsupportedCapability;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    layer_masks: bool,
    layer_filters: bool,
    inside_outside_path_strokes: bool,
    web_canvas_surfaces: bool,
}

impl Capabilities {
    pub(crate) const VELLO_0_9: Self = Self {
        layer_masks: false,
        layer_filters: false,
        inside_outside_path_strokes: false,
        web_canvas_surfaces: cfg!(all(feature = "render-web", target_arch = "wasm32")),
    };

    #[must_use]
    pub const fn supports_layer_masks(self) -> bool {
        self.layer_masks
    }

    #[must_use]
    pub const fn supports_layer_filters(self) -> bool {
        self.layer_filters
    }

    #[must_use]
    pub const fn supports_inside_outside_path_strokes(self) -> bool {
        self.inside_outside_path_strokes
    }

    #[must_use]
    pub const fn supports_web_canvas_surfaces(self) -> bool {
        self.web_canvas_surfaces
    }

    pub(crate) fn ensure(self, capability: UnsupportedCapability) -> super::Result<()> {
        let supported = match capability {
            UnsupportedCapability::LayerMask => self.layer_masks,
            UnsupportedCapability::LayerFilter => self.layer_filters,
            UnsupportedCapability::PathStrokeAlignment => self.inside_outside_path_strokes,
            UnsupportedCapability::WebCanvasSurface => self.web_canvas_surfaces,
            UnsupportedCapability::NonSolidShadowPaint => false,
        };
        if supported {
            Ok(())
        } else {
            Err(super::Error::unsupported_capability(capability))
        }
    }
}
```

- [ ] **Step 4: Wire capabilities into public API and renderer**

In `src/lib.rs`, add:

```rust
mod capability;
pub use capability::Capabilities;
```

In `src/renderer.rs`, add:

```rust
pub fn capabilities(&self) -> Capabilities {
    Capabilities::VELLO_0_9
}
```

- [ ] **Step 5: Replace scattered unsupported errors**

In `src/encode.rs`, replace hard-coded unsupported messages with `Error::unsupported_capability(...)`. For example:

```rust
if layer.filter().is_some() {
    return Err(Error::unsupported_capability(UnsupportedCapability::LayerFilter));
}
if layer.mask().is_some() {
    return Err(Error::unsupported_capability(UnsupportedCapability::LayerMask));
}
```

For non-solid shadow paint in `solid_color`, return `UnsupportedCapability::NonSolidShadowPaint`.

- [ ] **Step 6: Run focused tests**

Run:

```sh
cargo test -p surgeist-render renderer_reports_backend_capabilities capabilities_map_unsupported_operations_to_typed_errors layer_masks_report_capability_error layer_filters_report_explicit_error
```

Expected: pass.

- [ ] **Step 7: Refresh API artifact**

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: public API includes `Capabilities` and `Renderer::capabilities`.

- [ ] **Step 8: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/capability.rs src/lib.rs src/encode.rs src/renderer.rs src/tests.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/capability.rs src/lib.rs src/encode.rs src/renderer.rs src/tests.rs api/public-api.txt
git commit -m "model renderer capabilities"
```

## Task 5: Normalize Render Commands Before Encoding

**Files:**
- Create: `src/command.rs`
- Modify: `src/lib.rs`
- Modify: `src/scene.rs`
- Modify: `src/encode.rs`
- Modify: `src/stats.rs`
- Modify: `src/renderer.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add normalization tests**

Add these tests to `src/tests.rs`:

```rust
#[test]
fn scene_normalization_rejects_unsupported_commands_before_encoding() {
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_mask(Shape::rect(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap()))
            .unwrap(),
        |scene| {
            scene.fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK);
        },
    );

    let error = scene
        .normalize(Capabilities::VELLO_0_9)
        .expect_err("unsupported masks should fail during normalization");
    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
}

#[test]
fn scene_normalization_preserves_stats() {
    let mut scene = Scene::new();
    scene
        .fill(Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(), Color::BLACK)
        .layer(Layer::new(), |scene| {
            scene.stroke(
                Rect::try_new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Stroke::try_new(1.0).unwrap(),
                Color::BLACK,
            );
        });

    let normalized = scene.normalize(Capabilities::VELLO_0_9).unwrap();
    let stats = normalized.stats();

    assert_eq!(stats.commands, 3);
    assert_eq!(stats.fills, 1);
    assert_eq!(stats.strokes, 1);
    assert_eq!(stats.layers, 1);
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render scene_normalization_rejects_unsupported_commands_before_encoding scene_normalization_preserves_stats
```

Expected: fail because `Scene::normalize` and normalized command stats do not exist.

- [ ] **Step 3: Add normalized command module**

Create `src/command.rs` with:

```rust
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderCommands {
    pub(crate) commands: Vec<RenderCommand>,
}

impl RenderCommands {
    #[must_use]
    pub(crate) fn new(commands: Vec<RenderCommand>) -> Self {
        Self { commands }
    }

    #[must_use]
    pub(crate) fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        let mut uploaded_images = std::collections::HashSet::new();
        crate::stats::collect_render_stats(&self.commands, &mut stats, &mut uploaded_images);
        stats
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderCommand {
    Fill { shape: RenderShape, paint: RenderPaint },
    Stroke { shape: RenderStrokeShape, stroke: RenderStroke, paint: RenderPaint },
    Shadow { shape: ShadowShape, shadow: RenderShadow },
    Image { image: Image, rect: Rect, fit: ImageFit },
    TextRun {
        font: FontRef<'static>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: Vec<TextGlyph>,
    },
    Layer { layer: NormalizedLayer, children: Vec<RenderCommand> },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderShape {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
    Ellipse { center: Point, radii: Size },
    Path(Path),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderStrokeShape {
    Rect(kurbo::Rect),
    RoundedRect(kurbo::RoundedRect),
    Circle(kurbo::Circle),
    Ellipse(kurbo::Ellipse),
    Path(kurbo::BezPath),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderStroke {
    pub(crate) width: f64,
    pub(crate) join: LineJoin,
    pub(crate) start_cap: LineCap,
    pub(crate) end_cap: LineCap,
    pub(crate) miter_limit: f64,
    pub(crate) dash: Option<Dash>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RenderPaint {
    Color(Color),
    Gradient(Gradient),
    Image(Image),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ShadowShape {
    Rect(Rect),
    RoundedRect { rect: Rect, radii: Radii },
    Circle { center: Point, radius: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderShadow {
    pub(crate) offset: Point,
    pub(crate) blur: f64,
    pub(crate) spread: f64,
    pub(crate) color: Color,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NormalizedLayer {
    pub(crate) clip: Option<RenderShape>,
    pub(crate) transform: Transform,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) isolation: LayerIsolation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerIsolation {
    None,
    ClipOnly,
    BackendLayer,
}
```

- [ ] **Step 4: Add `Scene::normalize`**

In `src/lib.rs`, add:

```rust
mod command;
```

In `src/scene.rs`, add a normalization method:

```rust
pub(crate) fn normalize(&self, capabilities: Capabilities) -> Result<RenderCommands> {
    normalize_commands(&self.commands, capabilities).map(RenderCommands::new)
}
```

Add helper logic in `src/command.rs` that validates every command once, converts authored values into render-specific values, expands aligned strokes, and performs capability checks without calling into `src/encode.rs`:

```rust
fn normalize_commands(commands: &[Command], capabilities: Capabilities) -> Result<Vec<RenderCommand>> {
    let mut normalized = Vec::with_capacity(commands.len());
    for command in commands {
        normalized.push(match command {
            Command::Fill { shape, paint } => {
                RenderCommand::Fill {
                    shape: RenderShape::try_from(shape.clone())?,
                    paint: RenderPaint::try_from(paint.clone())?,
                }
            }
            Command::Stroke { shape, stroke, paint } => {
                let shape = RenderStrokeShape::from_authored(shape, *stroke, capabilities)?;
                RenderCommand::Stroke {
                    shape,
                    stroke: RenderStroke::try_from(*stroke)?,
                    paint: RenderPaint::try_from(paint.clone())?,
                }
            }
            Command::Shadow { shape, shadow } => {
                RenderCommand::Shadow {
                    shape: ShadowShape::try_from(shape.clone())?,
                    shadow: RenderShadow::try_from(shadow.clone())?,
                }
            }
            Command::Image { image, rect, fit } => {
                validate_rect(*rect, "image target rectangle")?;
                RenderCommand::Image { image: image.clone(), rect: *rect, fit: *fit }
            }
            Command::TextRun { font, size, transform, paint, glyphs } => {
                validate_text_run(*size, *transform, glyphs)?;
                validate_paint(paint.fill())?;
                RenderCommand::TextRun {
                    font: font.clone(),
                    size: *size,
                    transform: *transform,
                    paint: paint.clone(),
                    glyphs: glyphs.clone(),
                }
            }
            Command::Layer { layer, children } => {
                RenderCommand::Layer {
                    layer: NormalizedLayer::from_authored(layer, capabilities)?,
                    children: normalize_commands(children, capabilities)?,
                }
            }
        });
    }
    Ok(normalized)
}
```

Implement the conversion helpers in `src/command.rs`. `RenderStrokeShape::from_authored` owns the current aligned-stroke expansion from `src/encode.rs`; path strokes with non-center alignment must call `capabilities.ensure(UnsupportedCapability::PathStrokeAlignment)`. `NormalizedLayer::from_authored` owns layer isolation policy:

```rust
impl NormalizedLayer {
    fn from_authored(layer: &Layer, capabilities: Capabilities) -> Result<Self> {
        validate_layer(layer)?;
        if layer.mask().is_some() {
            capabilities.ensure(UnsupportedCapability::LayerMask)?;
        }
        if layer.filter().is_some() {
            capabilities.ensure(UnsupportedCapability::LayerFilter)?;
        }
        let isolation = if layer.clip().is_some()
            && layer.blend() == BlendMode::Normal
            && (layer.opacity() - 1.0).abs() < f32::EPSILON
        {
            LayerIsolation::ClipOnly
        } else if layer.clip().is_some()
            || layer.blend() != BlendMode::Normal
            || (layer.opacity() - 1.0).abs() > f32::EPSILON
        {
            LayerIsolation::BackendLayer
        } else {
            LayerIsolation::None
        };
        Ok(Self {
            clip: layer.clip().cloned().map(RenderShape::try_from).transpose()?,
            transform: layer.transform(),
            opacity: layer.opacity(),
            blend: layer.blend(),
            isolation,
        })
    }
}
```

- [ ] **Step 5: Update renderer to normalize before encoding**

In `src/renderer.rs`, change render flow to:

```rust
let normalized = scene.normalize(self.capabilities())?;
let mut uploaded_images = self.uploaded_images.clone();
collect_render_stats(&normalized.commands, &mut stats, &mut uploaded_images);
let vello_scene = encode_vello_scene(&normalized, surface.scale())?;
```

- [ ] **Step 6: Update encoder to consume normalized commands**

Change `encode_vello_scene` in `src/encode.rs`:

```rust
pub(crate) fn encode_vello_scene(commands: &RenderCommands, scale: f64) -> Result<vello::Scene> {
    let mut encoded = vello::Scene::new();
    encode_vello_commands(&commands.commands, &mut encoded, kurbo::Affine::scale(scale))?;
    Ok(encoded)
}
```

Change `encode_vello_commands` to match `RenderCommand` instead of `Command`. Remove these validation and capability calls from encoder helpers after normalization owns them:

```rust
validate_shape(shape)?;
validate_stroke(stroke)?;
validate_paint(paint)?;
validate_shadow(shadow)?;
validate_layer(layer)?;
validate_text_run(size, run_transform, glyphs)?;
return Err(Error::unsupported_capability(UnsupportedCapability::LayerFilter));
return Err(Error::unsupported_capability(UnsupportedCapability::LayerMask));
return Err(Error::unsupported_capability(UnsupportedCapability::PathStrokeAlignment));
```

Keep backend-only failures that require Vello-specific context, such as missing font data in `encode_text_run`, in `src/encode.rs`.

- [ ] **Step 7: Update stats**

Keep `collect_stats` for authored `Scene::stats`. Add a normalized counterpart in `src/stats.rs`:

```rust
pub(crate) fn collect_render_stats(
    commands: &[RenderCommand],
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    for command in commands {
        stats.commands = stats.commands.saturating_add(1);
        match command {
            RenderCommand::Fill { paint, .. } => {
                stats.fills = stats.fills.saturating_add(1);
                collect_render_paint_stats(paint, stats, uploaded_images);
            }
            RenderCommand::Stroke { paint, .. } => {
                stats.strokes = stats.strokes.saturating_add(1);
                collect_render_paint_stats(paint, stats, uploaded_images);
            }
            RenderCommand::Shadow { .. } => {
                stats.shadows = stats.shadows.saturating_add(1);
            }
            RenderCommand::Image { image, .. } => collect_image_stats(image, stats, uploaded_images),
            RenderCommand::TextRun { glyphs, .. } => {
                stats.glyphs = stats.glyphs.saturating_add(glyphs.len());
            }
            RenderCommand::Layer { children, .. } => {
                stats.layers = stats.layers.saturating_add(1);
                collect_render_stats(children, stats, uploaded_images);
            }
        }
    }
}

fn collect_render_paint_stats(
    paint: &RenderPaint,
    stats: &mut Stats,
    uploaded_images: &mut std::collections::HashSet<ImageId>,
) {
    if let RenderPaint::Image(image) = paint {
        collect_image_stats(image, stats, uploaded_images);
    }
}
```

- [ ] **Step 8: Run focused tests**

Run:

```sh
cargo test -p surgeist-render scene_normalization_rejects_unsupported_commands_before_encoding scene_normalization_preserves_stats aligned_path_strokes_report_explicit_error scene_encoding_is_deterministic failed_render_does_not_warm_image_reuse_stats
```

Expected: pass.

- [ ] **Step 9: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/command.rs src/lib.rs src/scene.rs src/encode.rs src/stats.rs src/renderer.rs src/tests.rs
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/command.rs src/lib.rs src/scene.rs src/encode.rs src/stats.rs src/renderer.rs src/tests.rs
git commit -m "normalize render commands before encoding"
```

## Task 6: Surface Lifecycle Typestates

**Files:**
- Modify: `src/surface.rs`
- Modify: `src/backend.rs`
- Modify: `src/renderer.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`
- Modify: `api/public-api.txt`

- [ ] **Step 1: Add lifecycle tests**

Add these tests to `src/tests.rs`:

```rust
#[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
use super::surface::{PresentedLifecycle, ResizeState};

#[test]
fn surface_state_reports_availability_without_bool_peeking() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(1.0, 1.0).unwrap(), 1.0)
        .unwrap();

    assert_eq!(surface.state(), SurfaceState::Available);
    surface.suspend().unwrap();
    assert_eq!(surface.state(), SurfaceState::Suspended);
}

#[test]
fn headless_backend_resource_state_tracks_readiness() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer
        .create_headless(Size::try_new(2.0, 2.0).unwrap(), 1.0)
        .unwrap();

    assert_eq!(surface.resource_state(), SurfaceResourceState::Ready);
    surface.resize(Size::try_new(3.0, 3.0).unwrap(), 1.0).unwrap();
    assert_eq!(surface.resource_state(), SurfaceResourceState::PendingAllocation);
}

#[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
#[test]
fn presented_surface_lifecycle_state_names_pending_resize() {
    let state = PresentedLifecycle::pending_resize(PhysicalSize::new(20, 10));

    assert_eq!(
        state,
        PresentedLifecycle::ResizePending {
            physical_size: PhysicalSize::new(20, 10),
            resizing: ResizeState::Idle,
        }
    );
}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```sh
cargo test -p surgeist-render surface_state_reports_availability_without_bool_peeking headless_backend_resource_state_tracks_readiness
```

Expected: fail because `SurfaceState`, `SurfaceResourceState`, `PresentedLifecycle`, `ResizeState`, and accessors do not exist.

- [ ] **Step 3: Replace availability boolean with state enum**

In `src/surface.rs`, add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceState {
    Available,
    Suspended,
}
```

Change `Surface` from `available: bool` to `state: SurfaceState`. Add:

```rust
#[must_use]
pub const fn state(&self) -> SurfaceState {
    self.state
}

pub(crate) fn ensure_available(&self) -> Result<()> {
    if self.state == SurfaceState::Suspended {
        return Err(Error::new(ErrorCode::SurfaceUnavailable, "surface is suspended"));
    }
    Ok(())
}
```

Update `suspend`, `resume`, `Renderer::render`, and `Renderer::set_surface_resizing` to use `state` / `ensure_available`.

- [ ] **Step 4: Replace headless texture/view options with resource state**

In `src/surface.rs`, add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceResourceState {
    ContractOnly,
    PendingAllocation,
    Ready,
    Presented,
}

pub(crate) enum HeadlessResources {
    Pending,
    Ready {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
}
```

Change `SurfaceBackend::Headless` to:

```rust
Headless {
    dev_id: usize,
    resources: HeadlessResources,
    physical_size: PhysicalSize,
}
```

- [ ] **Step 5: Replace presented booleans with lifecycle state**

In `src/surface.rs`, add typed presented-surface lifecycle state behind the same cfg as `SurfaceBackend::Presented`:

```rust
#[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResizeState {
    Idle,
    Resizing,
}

#[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentedLifecycle {
    Ready { resizing: ResizeState },
    ResizePending {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    NonRenderable {
        physical_size: PhysicalSize,
        resizing: ResizeState,
    },
    Occluded { resizing: ResizeState },
    Lost,
}

#[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
impl PresentedLifecycle {
    pub(crate) const fn pending_resize(physical_size: PhysicalSize) -> Self {
        Self::ResizePending {
            physical_size,
            resizing: ResizeState::Idle,
        }
    }

    pub(crate) const fn resizing(self) -> bool {
        matches!(
            self,
            Self::Ready { resizing: ResizeState::Resizing }
                | Self::ResizePending { resizing: ResizeState::Resizing, .. }
                | Self::NonRenderable { resizing: ResizeState::Resizing, .. }
                | Self::Occluded { resizing: ResizeState::Resizing }
        )
    }
}
```

Change `SurfaceBackend::Presented` from separate `valid`, `resizing`, and `pending_physical_size` fields to:

```rust
Presented {
    surface: Box<vello::util::RenderSurface<'static>>,
    lifecycle: PresentedLifecycle,
}
```

Update `Surface::resize` so non-zero presented resizes set `PresentedLifecycle::ResizePending { physical_size, resizing }`, zero-size resizes set `PresentedLifecycle::NonRenderable { physical_size, resizing }`, and identical sizes leave the state unchanged. Update `Renderer::set_surface_resizing` to update the `ResizeState` inside the lifecycle enum instead of toggling a standalone boolean.

- [ ] **Step 6: Add state accessors**

Add:

```rust
#[must_use]
pub const fn resource_state(&self) -> SurfaceResourceState {
    match &self.backend {
        SurfaceBackend::ContractOnly { .. } => SurfaceResourceState::ContractOnly,
        SurfaceBackend::Headless { resources, .. } => match resources {
            HeadlessResources::Pending => SurfaceResourceState::PendingAllocation,
            HeadlessResources::Ready { .. } => SurfaceResourceState::Ready,
        },
        #[cfg(any(feature = "render-window", all(feature = "render-web", target_arch = "wasm32")))]
        SurfaceBackend::Presented { .. } => SurfaceResourceState::Presented,
    }
}
```

- [ ] **Step 7: Update backend read/render logic**

In `src/backend.rs`, replace `texture: Option<_>` and `view: Option<_>` handling with `HeadlessResources` matching:

```rust
if matches!(resources, HeadlessResources::Pending) {
    let (texture, view) = create_headless_texture(
        &backend.context.devices[*dev_id].device,
        *physical_size,
        surface.options.format,
    );
    *resources = HeadlessResources::Ready { texture, view };
}
let HeadlessResources::Ready { view, .. } = resources else {
    unreachable!("headless resources should be ready after allocation");
};
```

Update `Renderer::read_headless` to require `HeadlessResources::Ready { texture, .. }`.

For presented surfaces, update `render_vello_surface` to consume `PresentedLifecycle` transitions:

```rust
match lifecycle {
    PresentedLifecycle::ResizePending { physical_size, resizing } => {
        backend.context.resize_surface(native, physical_size.width(), physical_size.height());
        *lifecycle = PresentedLifecycle::Ready { resizing: *resizing };
    }
    PresentedLifecycle::NonRenderable { .. } | PresentedLifecycle::Lost => {
        return Ok(RenderTimings::default());
    }
    PresentedLifecycle::Ready { .. } | PresentedLifecycle::Occluded { .. } => {}
}
```

Map `wgpu::CurrentSurfaceTexture::Lost` to `PresentedLifecycle::Lost`, map `Occluded` to `PresentedLifecycle::Occluded`, and keep `Outdated` as an explicit reconfiguration path that returns `SurfaceOutdated`.

- [ ] **Step 8: Run focused tests**

Run:

```sh
cargo test -p surgeist-render surface_state_reports_availability_without_bool_peeking headless_backend_resource_state_tracks_readiness surface_suspend_and_resume_preserve_attachment_kind headless_resize_keeps_target_when_physical_size_is_unchanged headless_render_can_be_read_back
cargo check -p surgeist-render --tests --features render-window
cargo test -p surgeist-render --features render-window presented_surface_lifecycle_state_names_pending_resize --no-run
cargo check -p surgeist-render --tests --features render-web
```

Expected: pass. If `render-web` is not checkable on the host target, report the exact compiler error and rerun the closest supported wasm check documented by the toolchain instead of silently skipping it.

- [ ] **Step 9: Refresh API artifact if public state accessors are exposed**

Before refreshing the artifact, export the state types from `src/lib.rs`:

```rust
pub use surface::{
    Attachment, Format, Parameters, PresentMode, Surface, SurfaceOptions, SurfaceResourceState,
    SurfaceState, WebCanvas,
};
```

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff -- api/public-api.txt
```

Expected: public API includes `SurfaceState`, `SurfaceResourceState`, `Surface::state`, and `Surface::resource_state`.

- [ ] **Step 10: Review gate and commit**

Run:

```sh
git diff --stat
git diff -- src/surface.rs src/backend.rs src/renderer.rs src/lib.rs src/tests.rs api/public-api.txt
cargo test -p surgeist-render
cargo fmt --check
git status --short --branch
```

Expected: tests and formatting pass. Commit:

```sh
git add src/surface.rs src/backend.rs src/renderer.rs src/lib.rs src/tests.rs api/public-api.txt
git commit -m "model surface lifecycle state"
```

## Task 7: Final Compliance Review And Checks

**Files:**
- Modify only if review finds plan-compliance gaps.

- [ ] **Step 1: Run baseline checks**

Run:

```sh
cargo test -p surgeist-render
cargo check -p surgeist-render --tests --features render-window
cargo test -p surgeist-render --features render-window presented_surface_lifecycle_state_names_pending_resize --no-run
cargo check -p surgeist-render --tests --features render-web
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
git status --short --branch
```

Expected: all checks pass and git status shows only intended committed task history.

- [ ] **Step 2: Inspect public API artifact**

Run:

```sh
(cd ../surgeist && cargo run --manifest-path api/generator/Cargo.toml -- --crate surgeist-render)
git diff --exit-code -- api/public-api.txt
```

Expected: no diff after generation. If there is a diff, inspect it and commit only intentional public API changes.

- [ ] **Step 3: Holistic review prompt**

Dispatch a separate reviewer with this exact scope:

```text
Review the complete surgeist-render implementation against guidance/surgeist-rust-modeling-guide.md and plans/2026-06-28-rust-modeling-compliance.md. Focus on whether the implementation meaningfully improves typed modeling for backend capabilities, draw command normalization, resource handles, scene encoding, and surface lifecycle without crossing crate boundaries or hand-editing generated artifacts. Check tests and public API artifact handling. Report Critical, Important, Minor findings; say "clean" only if no Critical or Important findings remain.
```

- [ ] **Step 4: Reconcile review**

If the reviewer reports Critical or Important findings, create a follow-up task-specific fix, run the relevant focused checks, commit it, and repeat Step 3. Completion requires a clean holistic review.

## Final Verification

Before reporting completion to the root coordinator or user, run:

```sh
cargo test -p surgeist-render
cargo check -p surgeist-render --tests --features render-window
cargo test -p surgeist-render --features render-window presented_surface_lifecycle_state_names_pending_resize --no-run
cargo check -p surgeist-render --tests --features render-web
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
git status --short --branch
```

Do not push unless another repo/thread needs to fetch the commits, root needs a submodule pointer update, or the user explicitly requests publication.
