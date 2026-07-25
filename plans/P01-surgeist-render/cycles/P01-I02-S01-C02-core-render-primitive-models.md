# Core Render Primitive Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add render-owned, constructor-validated public models for CSS/style paint, image layer, filter, mask, clip, border, outline, and background inputs without adding lowering or backend behavior.

**Architecture:** This phase is an API/model foundation only. New values must use private fields, explicit constructors, and typed diagnostics from Sequence 1; render keeps the symbolic-color policy root-resolved-only, so color front doors accept concrete `Color` and do not expose symbolic color payloads. Later phases will normalize these models into current `Scene` commands or new pipelines.

**Tech Stack:** Rust 2024, existing `surgeist-render` value types, Vello/Peniko only through current backend-neutral wrappers, `cargo test`, `cargo clippy`, `cargo fmt`.

---

## Source Scope

Sequence item:

- `plans/2026-07-09-render-css-implementation-sequence.md`, sequence 2.

Matrix coverage:

- `plans/2026-07-08-render-css-support-matrix.md`
- Paint sources
- CSS image layer model
- Filter list model
- Masks and clips model
- Box decoration, border, outline model
- Resource handles and images

Standing guidance:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`

## Policy Decisions

Symbolic color handoff:

- Render is root-resolved-only for this phase.
- Render accepts concrete `Color` values for style-facing color inputs.
- Render does not expose `currentColor`, system color, CSS `color()`, relative color, or color-mix payloads in this phase.
- If root cannot resolve a color before calling render, root must not construct these inputs yet. Future color realization work belongs to Sequence 5.

Resource policy:

- Render does not load URLs.
- Resolved image resources are represented by render-owned handles/metadata or by existing in-memory `Image` values.
- Unresolved references remain explicit in phase-specific types and may produce `UnresolvedResource` diagnostics when a future normalization step needs resolved resources.

Behavior policy:

- Do not wire the new models into `Scene`, `Renderer`, `encode`, or backend submission in this phase.
- Do not add sibling crate dependencies.
- Do not preserve backwards compatibility shims; this crate is still in development.

## File Map

- Modify `src/lib.rs`
  - Export the new public model types.
- Create `src/style.rs`
  - Own style-facing render primitive models that are not yet normalized into `Scene` commands.
  - Keep fields private and expose validated constructors/accessors.
- Modify `src/validation.rs`
  - Add small reusable validation helpers only if the new model constructors need them and existing helpers are insufficient.
- Modify `src/tests.rs`
  - Add construction, accessor, ordering, and invalid-state tests for each model family.

Do not modify `src/backend.rs`, `src/encode.rs`, `src/renderer.rs`, or `src/scene.rs` in this phase.

## Public Model Contract

The implementation must keep these public names and semantic boundaries:

```rust
// src/style.rs
pub struct StyleColor { color: Color }
pub struct StyleResourceRef { identifier: String }
pub struct ResolvedImageResource { id: ImageId, intrinsic_size: Size }
pub struct StyleImageSource { kind: StyleImageSourceKind }
pub enum StyleImageSourceKind { Image(Image), Resolved(ResolvedImageResource), Paint(Paint) }
pub struct StyleImageLayer { source: StyleImageSource, position: BackgroundPosition, size: BackgroundSize, repeat: BackgroundRepeat, origin: BackgroundBox, clip: BackgroundBox, attachment: BackgroundAttachment }
pub struct FilterList { ops: Option<Vec<FilterOp>> }
pub struct FilterOp { kind: FilterOpKind }
pub enum FilterOpKind { Blur(FilterBlur), Brightness(FilterAmount), Contrast(FilterAmount), Grayscale(UnitFilterAmount), HueRotate(FilterAngle), Invert(UnitFilterAmount), Opacity(UnitFilterAmount), Saturate(FilterAmount), Sepia(UnitFilterAmount), DropShadow(Shadow) }
pub struct FilterBlur { radius: f64 }
pub struct FilterAmount { value: f64 }
pub struct UnitFilterAmount { value: f64 }
pub struct FilterAngle { radians: f64 }
pub struct BackgroundSize { kind: BackgroundSizeKind }
pub enum BackgroundSizeKind { Auto, Cover, Contain, Explicit { width: SizeComponent, height: SizeComponent } }
pub struct SizeComponent { kind: SizeComponentKind }
pub enum SizeComponentKind { Auto, Length(f64), Percent(f64) }
pub struct ClipInput { kind: ClipInputKind }
pub enum ClipInputKind { Shape(Shape), Reference(StyleResourceRef) }
pub struct MaskInput { source: MaskSource, mode: MaskMode }
pub struct MaskSource { kind: MaskSourceKind }
pub enum MaskSourceKind { Shape(Shape), ImageLayer(StyleImageLayer), Reference(StyleResourceRef) }
pub struct BorderSide { style: BorderStyle, width: f64, paint: Paint }
pub struct BorderEdges { top: BorderSide, right: BorderSide, bottom: BorderSide, left: BorderSide }
pub struct Outline { style: OutlineStyle, width: f64, paint: Paint, offset: f64 }
pub struct BackgroundLayer { image: StyleImageLayer }
pub struct BackgroundStack { color: Option<Color>, layers: Vec<BackgroundLayer> }
```

Validation rules:

- Finite numbers are required for positions, lengths, offsets, angles, and scalar amounts.
- Widths and sizes must be non-negative unless an existing model already requires positive values.
- `FilterList::try_ops` rejects an empty list; `FilterList::none` represents identity.
- `FilterOp` has private fields and named constructors so callers cannot pair unit-range operations with unbounded amounts.
- `StyleImageSource`, `BackgroundSize`, `SizeComponent`, `ClipInput`, and `MaskSource` use private-field wrappers plus inspection kind enums so constructors validate payloads before public code can hold them.
- `StyleImageLayer::try_new` validates the source plus size/position/repeat values but does not compute placement.
- `BorderSide::try_new` accepts zero width for suppressed/none/hidden style and rejects negative or non-finite widths.
- Paint-bearing types validate existing `Paint` through current validation helpers.
- Resource/reference identifiers must not be empty after trimming.

## Task 1: Style Module Skeleton And Root-Resolved Color Policy

**Files:**

- Create: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing tests for style module exports and color policy**

Add tests named:

```rust
#[test]
fn style_color_inputs_are_root_resolved_concrete_colors() {
    let color = Color::try_rgba(0.25, 0.5, 0.75, 0.8).unwrap();
    let input = StyleColor::new(color);

    assert_eq!(input.color(), color);
}

#[test]
fn style_reference_identifiers_must_not_be_empty() {
    let error = StyleResourceRef::try_new("  ").expect_err("empty identifiers are invalid");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("style resource reference")
    );
}
```

- [ ] **Step 2: Implement minimal module and exports**

Add `src/style.rs` with:

```rust
use super::{Color, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StyleColor {
    color: Color,
}

impl StyleColor {
    #[must_use]
    pub const fn new(color: Color) -> Self {
        Self { color }
    }

    #[must_use]
    pub const fn color(self) -> Color {
        self.color
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StyleResourceRef {
    identifier: String,
}

impl StyleResourceRef {
    pub fn try_new(identifier: impl Into<String>) -> Result<Self> {
        let identifier = identifier.into();
        if identifier.trim().is_empty() {
            return Err(Error::invalid_value(
                "style resource reference",
                identifier,
                "must not be empty",
            ));
        }
        Ok(Self { identifier })
    }

    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}
```

Export `StyleColor` and `StyleResourceRef` from `src/lib.rs`.

- [ ] **Step 3: Run focused test**

Run:

```sh
cargo test -p surgeist-render style_color
cargo test -p surgeist-render style_reference
```

Expected: both tests pass.

## Task 2: Resource And CSS Image Layer Models

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing tests for resolved resources and image layers**

Add tests named:

```rust
#[test]
fn resolved_image_resources_preserve_handle_and_intrinsic_size() {
    let resource = ResolvedImageResource::try_new(ImageId::new(7), Size::new(24.0, 12.0)).unwrap();

    assert_eq!(resource.id(), ImageId::new(7));
    assert_eq!(resource.intrinsic_size(), Size::new(24.0, 12.0));
}

#[test]
fn css_image_layers_preserve_sampling_inputs_without_lowering() {
    let resource = ResolvedImageResource::try_new(ImageId::new(11), Size::new(8.0, 8.0)).unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .position(BackgroundPosition::percent(0.25, 0.75).unwrap())
        .size(BackgroundSize::cover())
        .repeat(BackgroundRepeat::repeat_x())
        .origin(BackgroundBox::Padding)
        .clip(BackgroundBox::Content)
        .attachment(BackgroundAttachment::Fixed);

    assert_eq!(layer.source().kind(), &StyleImageSourceKind::Resolved(resource));
    assert_eq!(layer.position().x().kind(), PositionComponentKind::Percent);
    assert_eq!(layer.position().y().value(), 0.75);
    assert_eq!(layer.size(), BackgroundSize::cover());
    assert_eq!(layer.repeat(), BackgroundRepeat::repeat_x());
    assert_eq!(layer.origin(), BackgroundBox::Padding);
    assert_eq!(layer.clip(), BackgroundBox::Content);
    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
}
```

- [ ] **Step 2: Implement resource and image layer types**

Add types with private fields and accessor methods:

```rust
pub struct ResolvedImageResource { id: ImageId, intrinsic_size: Size }
pub struct StyleImageSource { kind: StyleImageSourceKind }
pub enum StyleImageSourceKind { Image(Image), Resolved(ResolvedImageResource), Paint(Paint) }
pub struct StyleImageLayer { source: StyleImageSource, position: BackgroundPosition, size: BackgroundSize, repeat: BackgroundRepeat, origin: BackgroundBox, clip: BackgroundBox, attachment: BackgroundAttachment }
pub struct BackgroundPosition { x: PositionComponent, y: PositionComponent }
pub struct PositionComponent { kind: PositionComponentKind, value: f64 }
pub enum PositionComponentKind { Length, Percent }
pub struct BackgroundSize { kind: BackgroundSizeKind }
pub enum BackgroundSizeKind { Auto, Cover, Contain, Explicit { width: SizeComponent, height: SizeComponent } }
pub struct SizeComponent { kind: SizeComponentKind }
pub enum SizeComponentKind { Auto, Length(f64), Percent(f64) }
pub struct BackgroundRepeat { x: RepeatMode, y: RepeatMode }
pub enum RepeatMode { Repeat, NoRepeat, Round, Space }
pub enum BackgroundBox { Border, Padding, Content }
pub enum BackgroundAttachment { Scroll, Fixed, Local }
```

Constructor requirements:

- `ResolvedImageResource::try_new` validates `intrinsic_size`.
- `StyleImageSource::image` validates `Image` size.
- `StyleImageSource::paint` validates `Paint`.
- `StyleImageSource::resolved` accepts a prevalidated `ResolvedImageResource`.
- `StyleImageLayer::try_new` accepts a prevalidated `StyleImageSource`.
- position/size components validate finite and non-negative values where applicable.
- defaults are position `0% 0%`, size `Auto`, repeat `Repeat Repeat`, origin `Padding`, clip `Border`, attachment `Scroll`.

- [ ] **Step 3: Add invalid-state tests**

Add tests for:

```rust
#[test]
fn resolved_image_resources_reject_invalid_intrinsic_size() {
    let error = ResolvedImageResource::try_new(ImageId::new(7), Size::new(f64::NAN, 12.0))
        .expect_err("invalid intrinsic size should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("resolved image intrinsic size width")
    );
}

#[test]
fn background_position_rejects_non_finite_percent() {
    let error = BackgroundPosition::percent(f64::NAN, 0.0)
        .expect_err("non-finite percentages should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background position x percent")
    );
}

#[test]
fn background_size_rejects_negative_length() {
    let error = SizeComponent::try_length(-1.0)
        .expect_err("negative explicit background sizes should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background size length")
    );
}
```

Each test must assert `ErrorCode::InvalidInput` and a typed `InvalidValue` diagnostic.

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render resolved_image
cargo test -p surgeist-render css_image_layers
cargo test -p surgeist-render background_position
cargo test -p surgeist-render background_size
```

Expected: all tests pass.

## Task 3: Filter List And Operation Models

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing tests for filter identity, ordering, and validation**

Add tests named:

```rust
#[test]
fn filter_lists_distinguish_none_from_ordered_ops() {
    let list = FilterList::try_ops(vec![
        FilterOp::brightness(FilterAmount::try_new(1.2).unwrap()),
        FilterOp::blur(FilterBlur::try_new(4.0).unwrap()),
    ])
    .unwrap();

    assert!(!list.is_none());
    assert_eq!(list.ops().len(), 2);
    assert!(FilterList::none().is_none());
}

#[test]
fn filter_lists_reject_empty_ordered_ops() {
    let error = FilterList::try_ops(Vec::new()).expect_err("empty op lists must use none");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter operations")
    );
}
```

- [ ] **Step 2: Implement filter value types**

Add:

```rust
pub struct FilterList { ops: Option<Vec<FilterOp>> }
pub struct FilterOp { kind: FilterOpKind }
pub enum FilterOpKind { Blur(FilterBlur), Brightness(FilterAmount), Contrast(FilterAmount), Grayscale(UnitFilterAmount), HueRotate(FilterAngle), Invert(UnitFilterAmount), Opacity(UnitFilterAmount), Saturate(FilterAmount), Sepia(UnitFilterAmount), DropShadow(Shadow) }
pub struct FilterBlur { radius: f64 }
pub struct FilterAmount { value: f64 }
pub struct UnitFilterAmount { value: f64 }
pub struct FilterAngle { radians: f64 }
```

Constructor requirements:

- `FilterList::none()` stores no operations.
- `FilterList::try_ops` rejects empty vectors.
- `FilterBlur::try_new` requires finite non-negative radius.
- `UnitFilterAmount::try_new` requires finite `0.0..=1.0` for grayscale, invert, opacity, and sepia tests.
- `FilterAmount::try_new` requires finite non-negative amount for brightness, contrast, and saturate tests.
- `FilterAngle::try_radians` requires finite radians.
- `FilterOp` constructors are named methods such as `FilterOp::blur(...)`.
- `FilterOp` must not expose public enum variants that let callers pair unit-range operations with unbounded amounts.

- [ ] **Step 3: Add invalid filter tests**

Add tests for:

```rust
#[test]
fn filter_blur_rejects_negative_radius() {
    let error = FilterBlur::try_new(-0.1).expect_err("negative blur radius should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter blur radius")
    );
}

#[test]
fn filter_unit_amount_rejects_out_of_range_value() {
    let error = UnitFilterAmount::try_new(1.5)
        .expect_err("unit filter amounts must be clamped before render");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter unit amount")
    );
}

#[test]
fn filter_angle_rejects_nan() {
    let error = FilterAngle::try_radians(f64::NAN).expect_err("filter angles must be finite");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("filter angle")
    );
}
```

Each test must assert `ErrorCode::InvalidInput` and a typed `InvalidValue` diagnostic.

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render filter
```

Expected: all filter model tests pass.

## Task 4: Mask And Clip Input Models

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing tests for clip and mask phase-specific values**

Add tests named:

```rust
#[test]
fn clip_inputs_preserve_shape_or_unresolved_reference() {
    let shape = Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap());
    let clip = ClipInput::try_shape(shape.clone()).unwrap();
    let reference = ClipInput::reference(StyleResourceRef::try_new("#clip").unwrap());

    assert_eq!(clip.shape(), Some(&shape));
    assert_eq!(reference.reference().map(StyleResourceRef::identifier), Some("#clip"));
}

#[test]
fn mask_inputs_preserve_mode_and_source() {
    let mask = MaskInput::try_shape(
        Shape::rect(Rect::try_new(0.0, 0.0, 10.0, 10.0).unwrap()),
        MaskMode::Luminance,
    )
    .unwrap();

    assert_eq!(mask.mode(), MaskMode::Luminance);
    assert!(matches!(mask.source().kind(), MaskSourceKind::Shape(_)));
}
```

- [ ] **Step 2: Implement mask and clip types**

Add:

```rust
pub struct ClipInput { kind: ClipInputKind }
pub enum ClipInputKind { Shape(Shape), Reference(StyleResourceRef) }
pub struct MaskInput { source: MaskSource, mode: MaskMode }
pub struct MaskSource { kind: MaskSourceKind }
pub enum MaskSourceKind { Shape(Shape), ImageLayer(StyleImageLayer), Reference(StyleResourceRef) }
pub enum MaskMode { Alpha, Luminance }
```

Constructor requirements:

- `ClipInput::try_shape` validates shape.
- `ClipInput::reference` accepts only a prevalidated `StyleResourceRef`.
- `ClipInput::kind()` returns `ClipInputKind` by reference for inspection.
- `MaskInput::try_shape` validates shape.
- `MaskInput::image_layer` accepts a prevalidated `StyleImageLayer`.
- `MaskInput::reference` accepts a prevalidated `StyleResourceRef`.
- `MaskSource::kind()` returns `MaskSourceKind` by reference for inspection.
- Accessors must not expose mutable internals.

- [ ] **Step 3: Add invalid clip/mask tests**

Add tests for:

```rust
#[test]
fn clip_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = ClipInput::try_shape(Shape::path(path)).expect_err("invalid clip paths fail");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}

#[test]
fn mask_inputs_reject_invalid_shape_points() {
    let mut path = Path::new();
    path.move_to(Point::new(f64::NAN, 0.0));

    let error = MaskInput::try_shape(Shape::path(path), MaskMode::Alpha)
        .expect_err("invalid mask paths fail");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("path point x")
    );
}
```

Each test must assert typed invalid value diagnostics.

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render clip_inputs
cargo test -p surgeist-render mask_inputs
```

Expected: all tests pass.

## Task 5: Border, Outline, And Background Stack Models

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing tests for border sides, outlines, and background order**

Add tests named:

```rust
#[test]
fn border_edges_preserve_four_independent_sides() {
    let top = BorderSide::try_new(BorderStyle::Solid, 1.0, Color::BLACK).unwrap();
    let right = BorderSide::try_new(BorderStyle::Dashed, 2.0, Color::BLACK).unwrap();
    let bottom = BorderSide::try_new(BorderStyle::Dotted, 3.0, Color::BLACK).unwrap();
    let left = BorderSide::try_new(BorderStyle::Double, 4.0, Color::BLACK).unwrap();
    let edges = BorderEdges::new(top.clone(), right.clone(), bottom.clone(), left.clone());

    assert_eq!(edges.top(), &top);
    assert_eq!(edges.right(), &right);
    assert_eq!(edges.bottom(), &bottom);
    assert_eq!(edges.left(), &left);
}

#[test]
fn background_stacks_preserve_color_behind_ordered_layers() {
    let layer_a = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let layer_b = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack =
        BackgroundStack::try_new(Some(Color::BLACK), vec![layer_a.clone(), layer_b.clone()])
            .unwrap();

    assert_eq!(stack.color(), Some(Color::BLACK));
    assert_eq!(stack.layers(), &[layer_a, layer_b]);
}
```

- [ ] **Step 2: Implement decoration types**

Add:

```rust
pub struct BorderSide { style: BorderStyle, width: f64, paint: Paint }
pub struct BorderEdges { top: BorderSide, right: BorderSide, bottom: BorderSide, left: BorderSide }
pub enum BorderStyle { None, Hidden, Solid, Dashed, Dotted, Double, Groove, Ridge, Inset, Outset }
pub struct Outline { style: OutlineStyle, width: f64, paint: Paint, offset: f64 }
pub enum OutlineStyle { None, Solid, Dashed, Dotted, Double, Auto }
pub struct BackgroundLayer { image: StyleImageLayer }
pub struct BackgroundStack { color: Option<Color>, layers: Vec<BackgroundLayer> }
```

Constructor requirements:

- `BorderSide::try_new` validates finite non-negative width and paint.
- `Outline::try_new` validates finite non-negative width, finite offset, and paint.
- `BackgroundStack::try_new` preserves layer order and allows an empty layer list when a background color is present.
- Do not lower border styles, outlines, or backgrounds into draw commands in this phase.

- [ ] **Step 3: Add invalid decoration tests**

Add tests for:

```rust
#[test]
fn border_sides_reject_negative_width() {
    let error = BorderSide::try_new(BorderStyle::Solid, -1.0, Color::BLACK)
        .expect_err("negative border widths should be rejected");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("border side width")
    );
}

#[test]
fn outlines_reject_non_finite_offset() {
    let error = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, f64::NAN)
        .expect_err("outline offset must be finite");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("outline offset")
    );
}

#[test]
fn background_stacks_reject_empty_and_colorless_inputs() {
    let error = BackgroundStack::try_new(None, Vec::new())
        .expect_err("empty transparent background stacks should use no value");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background stack")
    );
}
```

Each test must assert `ErrorCode::InvalidInput` and typed invalid value diagnostics. If the implementation chooses to allow an empty transparent stack, document that choice in the test name and plan review must accept it before implementation.

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render border
cargo test -p surgeist-render outline
cargo test -p surgeist-render background_stack
```

Expected: all tests pass.

## Task 6: Integration Exports, Capability Consistency, And Cleanup

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add a public surface smoke test**

Add a test named:

```rust
#[test]
fn core_style_models_compose_without_backend_lowering() {
    let color = StyleColor::new(Color::BLACK);
    let paint = Paint::from(color.color());
    let image_layer = StyleImageLayer::try_new(StyleImageSource::paint(paint).unwrap()).unwrap();
    let background = BackgroundStack::try_new(
        Some(Color::TRANSPARENT),
        vec![BackgroundLayer::new(image_layer.clone())],
    )
    .unwrap();
    let filter =
        FilterList::try_ops(vec![FilterOp::opacity(UnitFilterAmount::try_new(0.5).unwrap())])
            .unwrap();
    let mask = MaskInput::image_layer(image_layer, MaskMode::Alpha);
    let outline = Outline::try_new(OutlineStyle::Solid, 1.0, Color::BLACK, 2.0).unwrap();

    assert_eq!(background.layers().len(), 1);
    assert_eq!(filter.ops().len(), 1);
    assert_eq!(mask.mode(), MaskMode::Alpha);
    assert_eq!(outline.offset(), 2.0);
}
```

- [ ] **Step 2: Verify exports and no backend wiring**

Confirm:

```sh
rg "pub use style" src/lib.rs
rg "StyleImageLayer|FilterList|BorderSide|BackgroundStack" src/backend.rs src/encode.rs src/renderer.rs src/scene.rs
```

Expected:

- `src/lib.rs` exports all public style model types.
- No backend/rendering files reference the new style model types in this phase.

- [ ] **Step 3: Run sequence-item focused checks**

Run:

```sh
cargo test -p surgeist-render style
cargo test -p surgeist-render filter
cargo test -p surgeist-render clip
cargo test -p surgeist-render mask
cargo test -p surgeist-render border
cargo test -p surgeist-render outline
cargo test -p surgeist-render background
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Expected: all pass.

## Final Review Gate

After all task-scoped worker/reviewer cycles and coordinator commits are complete, assign a final clean-context holistic reviewer to inspect:

- the plan
- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `git diff` for this implementation plan
- crate boundary and absence of sibling dependencies
- no backend behavior or lowering changes
- tests and required checks

Required final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Completion for this sequence item requires a clean holistic review and all required final checks passing.
