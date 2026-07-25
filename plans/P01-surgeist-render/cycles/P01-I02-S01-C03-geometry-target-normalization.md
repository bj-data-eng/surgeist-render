# Geometry Target Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize render-owned geometry target behavior against the CSS/style support matrix before higher-level paint, background, border, mask, and filter phases depend on it.

**Architecture:** This phase audits and sharpens the existing `Shape`, `Path`, `Stroke`, `Dash`, and geometry capability surface. Existing Vello-direct geometry remains direct; missing geometry operations become typed unsupported diagnostics rather than implicit behavior. Render remains independent of sibling shape/layout crates.

**Tech Stack:** Rust 2024, existing `surgeist-render` geometry/shape/command normalization, Vello/Kurbo only through current backend-neutral wrappers, `cargo test`, `cargo clippy`, `cargo fmt`.

---

## Source Scope

Sequence item:

- `plans/2026-07-09-render-css-implementation-sequence.md`, sequence 3.

Matrix coverage:

- `plans/2026-07-08-render-css-support-matrix.md`, geometry target rows.
- Rect fill/stroke.
- Rounded rect fill/stroke.
- Circle/ellipse fill/stroke.
- Arbitrary path fill.
- Arbitrary path centered stroke.
- Arbitrary path inside/outside stroke diagnostics.
- Geometry boolean/offset support diagnostics.
- Hit-test geometry out-of-scope handling.

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Current Baseline

Existing render-owned geometry already includes:

- `Shape::{rect, try_rounded_rect, try_circle, try_ellipse, path}`
- `Path` plus `PathElement::{MoveTo, LineTo, QuadTo, CubicTo, Close}`
- `Stroke`, `Dash`, `LineJoin`, `LineCap`, `StrokeAlign`
- direct normalization for rect/rounded-rect/circle/ellipse fills and strokes
- direct normalization for arbitrary path fill
- direct normalization for centered arbitrary path stroke
- typed unsupported diagnostic for inside/outside arbitrary path stroke alignment

This phase should preserve that baseline and make the geometry contract more explicit. Do not add geometry boolean or offset algorithms in this phase.

## File Map

- Modify `src/shape.rs`
  - Add any missing render-owned geometry model/accessor surface needed by tests and downstream phases.
  - Keep fields private and constructor validated.
- Modify `src/validation.rs`
  - Use public geometry inspection helpers when path storage becomes private.
- Modify `src/capability.rs`
  - Add explicit capability facts for unsupported geometry boolean/offset/hit-test ownership if missing.
- Modify `src/error.rs`
  - Add `PrimitiveOperation` labels for unsupported geometry operations only if current operations cannot name them precisely.
- Modify `src/command.rs`
  - Keep normalization behavior explicit and typed.
- Modify `src/tests.rs`
  - Add targeted behavior and diagnostic coverage.
- Modify `src/lib.rs`
  - Export only intentional public geometry model additions.

Do not modify `src/backend.rs`, `src/encode.rs`, or backend submission behavior in this phase except through tests that exercise existing render paths.

## Public Model Contract

The implementation must keep these semantic boundaries:

```rust
pub struct Path {
    elements: Vec<PathElement>,
}
pub enum FillRule { NonZero, EvenOdd }
pub struct FilledPath { path: Path, fill_rule: FillRule }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitTestOwnership { RootOwned }
```

Notes:

- `Path` may expose read-only element access through `elements()` so tests and future phases can inspect without mutation.
- `FilledPath` models fill-rule intent for future path clips/masks without changing current `Shape::path` behavior.
- `Shape::path(path)` continues to mean Vello/Kurbo nonzero fill behavior for the existing scene path.
- `FilledPath` is model-only in this phase. Do not wire it into `Scene`, `command`, `encode`, or backend behavior.
- Unsupported geometry boolean and offset operations are named through `PrimitiveOperation` diagnostics, not a separate public operation enum.
- `HitTestOwnership::RootOwned` documents that render does not own pointer hit testing.

## Task 1: Geometry Capability And Diagnostic Audit

**Files:**

- Modify: `src/capability.rs`
- Modify: `src/error.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs` only if a new public type is introduced

- [ ] **Step 1: Add failing capability/diagnostic tests**

Add tests named:

```rust
#[test]
fn geometry_capabilities_name_boolean_offset_and_hit_test_boundaries() {
    let capabilities = Capabilities::VELLO_0_9;

    assert!(!capabilities.geometry_targets().supports_geometry_booleans());
    assert!(!capabilities.geometry_targets().supports_geometry_offsets());
    assert_eq!(HitTestOwnership::RootOwned, HitTestOwnership::RootOwned);
}

#[test]
fn unsupported_geometry_operations_report_typed_diagnostics() {
    let boolean = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryBooleanOperation,
    );
    let offset = UnsupportedPrimitive::new(
        PrimitiveFamily::GeometryTargets,
        PrimitiveOperation::GeometryOffsetOperation,
    );

    for unsupported in [boolean, offset] {
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("geometry operation should be explicitly unsupported");
        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}
```

- [ ] **Step 2: Add capability and diagnostic model surface**

Expected implementation:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitTestOwnership {
    RootOwned,
}
```

Extend `GeometryTargetCapabilities` with private fields and accessors:

```rust
geometry_booleans: bool,
geometry_offsets: bool,
hit_testing: HitTestOwnership,

pub const fn supports_geometry_booleans(self) -> bool;
pub const fn supports_geometry_offsets(self) -> bool;
pub const fn hit_testing(self) -> HitTestOwnership;
```

Extend `PrimitiveOperation` with:

```rust
GeometryBooleanOperation
GeometryOffsetOperation
```

Map both operations through `Capabilities::ensure_supported`.

- [ ] **Step 3: Run focused tests**

Run:

```sh
cargo test -p surgeist-render geometry_capabilities
cargo test -p surgeist-render unsupported_geometry_operations
```

Expected: both pass.

## Task 2: Path Inspection And Fill Rule Models

**Files:**

- Modify: `src/shape.rs`
- Modify: `src/validation.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing tests for path inspection and fill-rule intent**

Add tests named:

```rust
#[test]
fn paths_expose_elements_without_exposing_mutation() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .close();

    assert_eq!(path.elements().len(), 3);
    assert!(matches!(path.elements()[0], PathElement::MoveTo(_)));
}

#[test]
fn filled_paths_preserve_fill_rule_intent() {
    let mut path = Path::new();
    path.move_to(Point::try_new(0.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 0.0).unwrap())
        .line_to(Point::try_new(4.0, 4.0).unwrap())
        .close();
    let filled = FilledPath::try_new(path.clone(), FillRule::EvenOdd).unwrap();

    assert_eq!(filled.path(), &path);
    assert_eq!(filled.fill_rule(), FillRule::EvenOdd);
}
```

- [ ] **Step 2: Implement model additions**

Add:

```rust
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilledPath {
    path: Path,
    fill_rule: FillRule,
}
```

Constructor and accessors:

```rust
impl Path {
    pub fn elements(&self) -> &[PathElement];
}

impl FilledPath {
    pub fn try_new(path: Path, fill_rule: FillRule) -> Result<Self>;
    pub fn path(&self) -> &Path;
    pub const fn fill_rule(&self) -> FillRule;
}
```

`FilledPath::try_new` validates path elements with existing validation and stores the path unchanged.

Update existing validation code to inspect paths through `Path::elements()` rather than reading storage directly. `Path` storage should be private after this task.

- [ ] **Step 3: Add invalid path diagnostic test**

Add:

```rust
#[test]
fn filled_paths_reject_invalid_path_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = FilledPath::try_new(path, FillRule::NonZero)
        .expect_err("filled paths validate stored path elements");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}
```

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render path
cargo test -p surgeist-render filled_path
```

Expected: all pass.

## Task 3: Direct Geometry Support Regression Coverage

**Files:**

- Modify: `src/tests.rs`

- [ ] **Step 1: Add tests for direct fill/stroke support**

Add tests named:

```rust
#[test]
fn direct_geometry_targets_render_without_unsupported_diagnostics() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(32.0, 32.0), 1.0).unwrap();
    let mut scene = Scene::new();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 24.0).unwrap())
        .line_to(Point::try_new(8.0, 30.0).unwrap())
        .close();

    scene.fill(Shape::rect(Rect::try_new(1.0, 1.0, 4.0, 4.0).unwrap()), Color::BLACK);
    scene.stroke(
        Shape::rect(Rect::try_new(1.0, 7.0, 4.0, 4.0).unwrap()),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 1.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_rounded_rect(
            Rect::try_new(6.0, 7.0, 4.0, 4.0).unwrap(),
            Radii::try_all(1.0).unwrap(),
        )
        .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_circle(Point::try_new(4.0, 14.0).unwrap(), 2.0).unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_circle(Point::try_new(4.0, 20.0).unwrap(), 2.0).unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(
        Shape::try_ellipse(Point::try_new(14.0, 14.0).unwrap(), Size::try_new(3.0, 2.0).unwrap())
            .unwrap(),
        Color::BLACK,
    );
    scene.stroke(
        Shape::try_ellipse(Point::try_new(14.0, 20.0).unwrap(), Size::try_new(3.0, 2.0).unwrap())
            .unwrap(),
        Stroke::try_new(1.0).unwrap(),
        Color::BLACK,
    );
    scene.fill(Shape::path(path), Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("direct geometry targets should render");
}
```

- [ ] **Step 2: Add centered path stroke regression**

Add:

```rust
#[test]
fn centered_path_strokes_support_join_cap_and_dash_inputs() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(24.0, 24.0), 1.0).unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(2.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 2.0).unwrap())
        .line_to(Point::try_new(20.0, 20.0).unwrap());
    let stroke = Stroke::try_new(2.0)
        .unwrap()
        .join(LineJoin::Round)
        .caps(LineCap::Round, LineCap::Square)
        .try_dash(Dash::try_new(0.0, &[2.0, 1.0]).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.stroke(Shape::path(path), stroke, Color::BLACK);

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("centered path strokes should render");
}
```

- [ ] **Step 3: Run focused tests**

Run:

```sh
cargo test -p surgeist-render direct_geometry_targets
cargo test -p surgeist-render centered_path_strokes
```

Expected: both pass.

## Task 4: Unsupported Geometry Boundary Regression

**Files:**

- Modify: `src/tests.rs`
- Modify: `src/command.rs` only if existing diagnostics are not typed enough

- [ ] **Step 1: Preserve inside/outside arbitrary path stroke diagnostic**

Add or strengthen a test named:

```rust
#[test]
fn inside_outside_path_strokes_keep_typed_geometry_diagnostic() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(8.0, 8.0), 1.0).unwrap();
    let mut path = Path::new();
    path.move_to(Point::try_new(1.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 1.0).unwrap())
        .line_to(Point::try_new(6.0, 6.0).unwrap())
        .close();
    let mut scene = Scene::new();
    scene.stroke(
        Shape::path(path),
        Stroke::try_new(1.0).unwrap().align(StrokeAlign::Inside),
        Color::BLACK,
    );

    let error = renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect_err("inside path stroke alignment requires offset lowering");

    assert_eq!(error.code, ErrorCode::UnsupportedBackend);
    assert_eq!(
        error.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::GeometryTargets,
            PrimitiveOperation::InsideOutsidePathStrokeAlignment,
        ))
    );
}
```

- [ ] **Step 2: Document hit testing as root-owned**

Add:

```rust
#[test]
fn hit_test_geometry_is_root_owned_not_render_lowered() {
    assert_eq!(
        Capabilities::VELLO_0_9.geometry_targets().hit_testing(),
        HitTestOwnership::RootOwned
    );
}
```

- [ ] **Step 3: Run focused tests**

Run:

```sh
cargo test -p surgeist-render inside_outside_path_strokes
cargo test -p surgeist-render hit_test_geometry
```

Expected: both pass.

## Task 5: Integration Cleanup And Checks

**Files:**

- Modify: `src/capability.rs`
- Modify: `src/error.rs`
- Modify: `src/shape.rs`
- Modify: `src/validation.rs`
- Modify: `src/command.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Verify no sibling dependencies and no backend behavior drift**

Run:

```sh
git diff -- src/backend.rs src/encode.rs src/renderer.rs
git diff -- Cargo.toml Cargo.lock
git diff -- Cargo.toml | rg 'path = "../' || true
```

Expected:

- No backend/renderer/encoder behavior changes unless earlier task review explicitly required them.
- No dependency changes in `Cargo.toml` or `Cargo.lock`.
- No new sibling path dependencies. The existing optional `surgeist-window` path dependency is pre-existing and should remain untouched.

- [ ] **Step 2: Run sequence-item focused checks**

Run:

```sh
cargo test -p surgeist-render geometry
cargo test -p surgeist-render path
cargo test -p surgeist-render stroke
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
- direct geometry behavior and unsupported diagnostics
- tests and required checks

Required final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Completion for this sequence item requires a clean holistic review and all required final checks passing.
