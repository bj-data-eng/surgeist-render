# Transform And Coordinate Space Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize render-owned transform and coordinate-space primitives so background, image, mask, clip, and later backdrop/effect phases can share one explicit coordinate contract.

**Architecture:** This phase extends the existing `Transform` model with validated 2D helper constructors, explicit composition, skew helpers, and transform-origin wrapping while preserving the current Vello-direct layer transform path. Unsupported 3D transforms become typed capability diagnostics. Coordinate-space tags are model-only data carried by style/clip/mask inputs for later lowering; they do not change backend behavior in this phase.

**Tech Stack:** Rust 2024, existing render-owned `Transform`, `Layer`, `StyleImageLayer`, `ClipInput`, `MaskInput`, typed diagnostics, Vello layer transform encoding, `cargo test`, `cargo clippy`, `cargo fmt`.

---

## Source Scope

Sequence item:

- `plans/2026-07-09-render-css-implementation-sequence.md`, sequence 4.

Matrix coverage:

- `plans/2026-07-08-render-css-support-matrix.md`, section 10, transforms and coordinate spaces.
- 2D affine transform.
- Transform origin.
- Skew.
- 3D transform diagnostics.
- Coordinate-space tagging.
- Transformed clips/images behavior.

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

Existing render-owned transform behavior includes:

- `Transform([f64; 6])` with `identity`, `try_new`, `as_array`, `Default`, and `From<Transform> for kurbo::Affine`.
- `Layer::try_transform` validates a `Transform`.
- `Scene::transform` wraps children in a layer.
- Vello encoding already applies layer transforms.
- Tests cover a simple translation and that pure transforms do not require backend layer isolation.

Current gaps:

- No public helper constructors for translate, scale, rotate, skew, or matrix composition.
- No transform-origin normalization helper.
- No typed 3D transform unsupported diagnostic.
- No coordinate-space tag model for fixed backgrounds, transformed masks/clips, or future backdrop capture.

## File Map

- Modify `src/geometry.rs`
  - Add transform helper constructors, composition, origin wrapping, and coordinate-space tag models.
  - Keep fields private and constructors validated.
- Modify `src/validation.rs`
  - Add validation helpers for coordinate-space tags if needed.
- Modify `src/capability.rs`
  - Add transform/coordinate capability facts.
- Modify `src/error.rs`
  - Add transform/coordinate primitive family and unsupported 3D operation labels.
- Modify `src/style.rs`
  - Carry optional coordinate-space tags on style image, clip, and mask inputs.
- Modify `src/tests.rs`
  - Add targeted model, diagnostic, and render regression tests.
- Modify `src/lib.rs`
  - Export only intentional public transform and coordinate-space additions.

Do not modify `src/backend.rs`, `src/encode.rs`, or backend submission behavior in this phase except through tests that exercise existing layer transform paths.

## Public Model Contract

The implementation must keep these semantic boundaries:

```rust
pub struct Transform([f64; 6]);

pub struct CoordinateSpaceId {
    value: u64,
}

pub enum CoordinateSpaceKind {
    Local,
    Viewport,
    Surface,
    Named(CoordinateSpaceId),
}

pub struct CoordinateSpaceTag {
    kind: CoordinateSpaceKind,
    transform: Transform,
}
```

Notes:

- `Transform` coefficient order stays the existing Kurbo/Vello affine order: `[a, b, c, d, e, f]`, where `x' = a*x + c*y + e` and `y' = b*x + d*y + f`.
- `Transform::then(next)` must mean "apply `self`, then apply `next`." This keeps origin wrapping readable.
- `Transform::around(origin)` must produce `translate(-origin).then(self).then(translate(origin))`.
- Coordinate-space tags are model-only in this phase. They may be carried by style image, clip, and mask inputs, but must not be lowered into backend rendering yet.
- `CoordinateSpaceTag` is the shared model needed by future backdrop capture, but there is no backdrop carrier in this phase because backdrop primitives are sequence 7 work.
- 3D transform support is diagnostic-only in this phase. Do not add perspective flattening or 3D compositor behavior.

## Task 1: 2D Transform Helper Constructors

**Files:**

- Modify: `src/geometry.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs` only if new public types are introduced

- [ ] **Step 1: Add failing transform constructor tests**

Add tests named:

```rust
#[test]
fn transform_helpers_preserve_affine_coefficients() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 4.0).unwrap();
    let rotate = Transform::rotation(std::f64::consts::FRAC_PI_2).unwrap();

    assert_eq!(translate.as_array(), [1.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(scale.as_array(), [2.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
    assert!(rotate.as_array()[0].abs() < 1.0e-12);
    assert!((rotate.as_array()[1] - 1.0).abs() < 1.0e-12);
    assert!((rotate.as_array()[2] + 1.0).abs() < 1.0e-12);
    assert!(rotate.as_array()[3].abs() < 1.0e-12);
}

#[test]
fn transform_skew_helpers_preserve_tangent_coefficients() {
    let skew_x = Transform::skew_x(std::f64::consts::FRAC_PI_4).unwrap();
    let skew_y = Transform::skew_y(std::f64::consts::FRAC_PI_4).unwrap();

    assert!((skew_x.as_array()[2] - 1.0).abs() < 1.0e-12);
    assert!((skew_y.as_array()[1] - 1.0).abs() < 1.0e-12);
}
```

- [ ] **Step 2: Add helper constructors**

Add methods to `impl Transform`:

```rust
pub fn translation(x: f64, y: f64) -> Result<Self>;
pub fn scale(x: f64, y: f64) -> Result<Self>;
pub fn rotation(radians: f64) -> Result<Self>;
pub fn skew_x(radians: f64) -> Result<Self>;
pub fn skew_y(radians: f64) -> Result<Self>;
```

Implementation expectations:

- Every method routes through `Transform::try_new`.
- `rotation` uses `radians.sin_cos()` or equivalent finite trigonometry and rejects non-finite input through `try_new`.
- `skew_x` and `skew_y` use `radians.tan()` and reject non-finite tangent results through `try_new`.

- [ ] **Step 3: Add invalid input test**

Add:

```rust
#[test]
fn transform_helpers_reject_non_finite_inputs() {
    assert!(Transform::translation(f64::NAN, 0.0).is_err());
    assert!(Transform::scale(1.0, f64::INFINITY).is_err());
    assert!(Transform::rotation(f64::NAN).is_err());
    assert!(Transform::skew_x(f64::INFINITY).is_err());
    assert!(Transform::skew_y(f64::NAN).is_err());
}
```

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render transform_helpers
cargo test -p surgeist-render transform_skew
```

Expected: both pass.

## Task 2: Transform Composition And Origin Normalization

**Files:**

- Modify: `src/geometry.rs`
- Modify: `src/tests.rs`

- [ ] **Step 1: Add failing composition and origin tests**

Add tests named:

```rust
#[test]
fn transform_then_composes_in_application_order() {
    let translate = Transform::translation(2.0, 3.0).unwrap();
    let scale = Transform::scale(2.0, 2.0).unwrap();
    let composed = translate.then(scale).unwrap();

    assert_eq!(composed.as_array(), [2.0, 0.0, 0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn transform_around_wraps_transform_origin() {
    let origin = Point::try_new(10.0, 5.0).unwrap();
    let transform = Transform::scale(2.0, 3.0).unwrap().around(origin).unwrap();

    assert_eq!(transform.as_array(), [2.0, 0.0, 0.0, 3.0, -10.0, -10.0]);
}
```

- [ ] **Step 2: Implement composition helpers**

Add methods to `impl Transform`:

```rust
pub fn then(self, next: Self) -> Result<Self>;
pub fn around(self, origin: Point) -> Result<Self>;
```

`then` formula for `self = [a, b, c, d, e, f]` and `next = [na, nb, nc, nd, ne, nf]`:

```rust
[
    na * a + nc * b,
    nb * a + nd * b,
    na * c + nc * d,
    nb * c + nd * d,
    na * e + nc * f + ne,
    nb * e + nd * f + nf,
]
```

`around(origin)` must call:

```rust
Transform::translation(-origin.x(), -origin.y())?
    .then(self)?
    .then(Transform::translation(origin.x(), origin.y())?)
```

- [ ] **Step 3: Add transformed layer regression tests**

Add tests named:

```rust
#[test]
fn composed_layer_transforms_render_in_order() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(6.0, 2.0), 1.0).unwrap();
    let transform = Transform::translation(1.0, 0.0)
        .unwrap()
        .then(Transform::scale(2.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(0.0, 0.0, 1.0, 2.0), Color::BLACK);
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("composed transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn origin_wrapped_layer_transform_renders_about_origin() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 4.0), 1.0).unwrap();
    let transform = Transform::scale(2.0, 2.0)
        .unwrap()
        .around(Point::try_new(1.0, 1.0).unwrap())
        .unwrap();
    let mut scene = Scene::new();
    scene.transform(transform, |scene| {
        scene.fill(Rect::new(1.0, 1.0, 1.0, 1.0), Color::BLACK);
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("origin-wrapped transform should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert!(pixel_alpha(&output, 1, 1) > 0);
    assert!(pixel_alpha(&output, 2, 2) > 0);
}
```

- [ ] **Step 4: Run focused tests**

Run:

```sh
cargo test -p surgeist-render transform_then
cargo test -p surgeist-render transform_around
cargo test -p surgeist-render composed_layer_transforms
cargo test -p surgeist-render origin_wrapped_layer_transform
```

Expected: all pass.

## Task 3: Transform Capabilities And 3D Diagnostics

**Files:**

- Modify: `src/capability.rs`
- Modify: `src/error.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing capability and diagnostic tests**

Add tests named:

```rust
#[test]
fn transform_capabilities_name_2d_origin_skew_and_coordinate_tags() {
    let capabilities = Capabilities::VELLO_0_9.transform_coordinate_spaces();

    assert!(capabilities.supports_affine_2d());
    assert!(capabilities.supports_transform_origin());
    assert!(capabilities.supports_skew());
    assert!(capabilities.supports_coordinate_space_tags());
    assert!(!capabilities.supports_transform_3d());
}

#[test]
fn unsupported_3d_transforms_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::Matrix3dTransform,
        PrimitiveOperation::PerspectiveTransform,
        PrimitiveOperation::Rotate3dTransform,
        PrimitiveOperation::TranslateZTransform,
        PrimitiveOperation::ScaleZTransform,
    ] {
        let unsupported = UnsupportedPrimitive::new(
            PrimitiveFamily::TransformsAndCoordinateSpaces,
            operation,
        );

        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("3D transforms are unsupported in this render phase");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
    }
}
```

- [ ] **Step 2: Add capability model surface**

Add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformCoordinateSpaceCapabilities {
    affine_2d: bool,
    transform_origin: bool,
    skew: bool,
    transform_3d: bool,
    coordinate_space_tags: bool,
}
```

Expose narrow accessors:

```rust
pub const fn supports_affine_2d(self) -> bool;
pub const fn supports_transform_origin(self) -> bool;
pub const fn supports_skew(self) -> bool;
pub const fn supports_transform_3d(self) -> bool;
pub const fn supports_coordinate_space_tags(self) -> bool;
```

Add `transform_coordinate_spaces` to `Capabilities`, initialize `Capabilities::VELLO_0_9` with:

```rust
affine_2d: true,
transform_origin: true,
skew: true,
transform_3d: false,
coordinate_space_tags: true,
```

- [ ] **Step 3: Add typed diagnostic labels**

Extend `PrimitiveFamily` with:

```rust
TransformsAndCoordinateSpaces
```

Label: `"transforms and coordinate spaces"`.

Extend `PrimitiveOperation` with:

```rust
Matrix3dTransform
PerspectiveTransform
Rotate3dTransform
TranslateZTransform
ScaleZTransform
```

Labels:

- `Matrix3dTransform`: `"matrix3d transform"`
- `PerspectiveTransform`: `"perspective transform"`
- `Rotate3dTransform`: `"rotate3d transform"`
- `TranslateZTransform`: `"translateZ transform"`
- `ScaleZTransform`: `"scaleZ transform"`

Wire each 3D transform operation through `Capabilities::ensure_supported` using `supports_transform_3d()`.

- [ ] **Step 4: Export capability type and run focused checks**

Export `TransformCoordinateSpaceCapabilities` from `src/lib.rs`.

Run:

```sh
cargo test -p surgeist-render transform_capabilities
cargo test -p surgeist-render unsupported_3d_transforms
```

Expected: both pass.

## Task 4: Coordinate-Space Tag Models

**Files:**

- Modify: `src/geometry.rs`
- Modify: `src/validation.rs` if needed
- Modify: `src/style.rs`
- Modify: `src/tests.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing coordinate-space model tests**

Add tests named:

```rust
#[test]
fn coordinate_space_tags_preserve_kind_and_transform() {
    let named = CoordinateSpaceId::try_new(7).unwrap();
    let transform = Transform::translation(3.0, 4.0).unwrap();
    let tag = CoordinateSpaceTag::try_new(
        CoordinateSpaceKind::Named(named),
        transform,
    )
    .unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Named(named));
    assert_eq!(tag.transform(), transform);
}

#[test]
fn coordinate_space_ids_reject_reserved_zero() {
    let error = CoordinateSpaceId::try_new(0).expect_err("zero is reserved");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("coordinate space id")
    );
}
```

- [ ] **Step 2: Add coordinate-space model types**

Add in `src/geometry.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateSpaceId {
    value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateSpaceKind {
    Local,
    Viewport,
    Surface,
    Named(CoordinateSpaceId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateSpaceTag {
    kind: CoordinateSpaceKind,
    transform: Transform,
}
```

Add constructors/accessors:

```rust
impl CoordinateSpaceId {
    pub fn try_new(value: u64) -> Result<Self>;
    pub const fn get(self) -> u64;
}

impl CoordinateSpaceTag {
    pub fn try_new(kind: CoordinateSpaceKind, transform: Transform) -> Result<Self>;
    pub fn local() -> Self;
    pub fn viewport(transform: Transform) -> Result<Self>;
    pub fn surface(transform: Transform) -> Result<Self>;
    pub const fn kind(self) -> CoordinateSpaceKind;
    pub const fn transform(self) -> Transform;
}
```

Validation expectations:

- `CoordinateSpaceId::try_new(0)` returns `Error::invalid_value("coordinate space id", 0, "must be non-zero")`.
- `CoordinateSpaceTag::try_new` validates the transform through existing finite transform validation.
- `CoordinateSpaceTag::local()` uses `CoordinateSpaceKind::Local` and `Transform::identity()`.

- [ ] **Step 3: Carry tags on style image, clip, and mask inputs**

Add optional coordinate-space storage and accessors:

```rust
impl StyleImageLayer {
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self;
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag>;
}

impl ClipInput {
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self;
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag>;
}

impl MaskInput {
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self;
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag>;
}
```

Default coordinate-space storage is `None`, meaning the primitive remains in the current local render space.

- [ ] **Step 4: Add style carrier tests**

Add:

```rust
#[test]
fn fixed_background_layers_can_carry_viewport_coordinate_space() {
    let layer = StyleImageLayer::try_new(
        StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap(),
    )
    .unwrap()
    .with_attachment(BackgroundAttachment::Fixed)
    .with_coordinate_space(
        CoordinateSpaceTag::viewport(Transform::translation(10.0, 20.0).unwrap()).unwrap(),
    );

    assert_eq!(layer.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        layer.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}

#[test]
fn masks_and_clips_can_carry_coordinate_space_tags() {
    let tag = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let clip = ClipInput::try_shape(Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)))
        .unwrap()
        .with_coordinate_space(tag);
    let mask = MaskInput::try_shape(
        Shape::rect(Rect::new(0.0, 0.0, 1.0, 1.0)),
        MaskMode::Alpha,
    )
    .unwrap()
    .with_coordinate_space(tag);

    assert_eq!(clip.coordinate_space(), Some(tag));
    assert_eq!(mask.coordinate_space(), Some(tag));
}

#[test]
fn coordinate_space_tags_model_future_backdrop_capture_space() {
    let tag = CoordinateSpaceTag::viewport(Transform::translation(4.0, 6.0).unwrap()).unwrap();

    assert_eq!(tag.kind(), CoordinateSpaceKind::Viewport);
    assert_eq!(tag.transform().as_array(), [1.0, 0.0, 0.0, 1.0, 4.0, 6.0]);
}
```

- [ ] **Step 5: Export public additions and run focused checks**

Export `CoordinateSpaceId`, `CoordinateSpaceKind`, and `CoordinateSpaceTag` from `src/lib.rs`.

Run:

```sh
cargo test -p surgeist-render coordinate_space
cargo test -p surgeist-render fixed_background_layers
cargo test -p surgeist-render masks_and_clips_can_carry_coordinate_space_tags
```

Expected: all pass.

## Task 5: Transform Regression Coverage And Integration Checks

**Files:**

- Modify: `src/tests.rs`
- Modify other files only if a focused review finds an integration bug from Tasks 1-4.

- [ ] **Step 1: Add transformed clip and image tests**

Add tests named:

```rust
#[test]
fn transformed_shape_clips_render_in_layer_space() {
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.layer(
        Layer::new()
            .try_transform(Transform::translation(2.0, 0.0).unwrap())
            .unwrap()
            .try_clip(Shape::rect(Rect::new(0.0, 0.0, 2.0, 2.0)))
            .unwrap(),
        |scene| {
            scene.fill(Rect::new(0.0, 0.0, 4.0, 2.0), Color::BLACK);
        },
    );

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("transformed clip should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
    assert!(pixel_alpha(&output, 3, 0) > 0);
}

#[test]
fn transformed_images_render_in_layer_space() {
    let image =
        Image::from_rgba(Size::new(1.0, 1.0), Arc::<[u8]>::from([0, 0, 0, 255])).unwrap();
    let mut renderer = pollster::block_on(Renderer::new(Options::default())).unwrap();
    let mut surface = renderer.create_headless(Size::new(4.0, 2.0), 1.0).unwrap();
    let mut scene = Scene::new();
    scene.transform(Transform::translation(2.0, 0.0).unwrap(), |scene| {
        scene.image(
            image,
            Rect::new(0.0, 0.0, 2.0, 2.0),
            ImageFit::Stretch,
        );
    });

    renderer
        .render(&mut surface, &scene, Parameters::default())
        .expect("transformed image should render");
    let output = renderer.read_headless(&surface).unwrap();

    assert_eq!(pixel_alpha(&output, 0, 0), 0);
    assert_eq!(pixel_alpha(&output, 1, 0), 0);
    assert!(pixel_alpha(&output, 2, 0) > 0);
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
cargo test -p surgeist-render transform
cargo test -p surgeist-render coordinate_space
cargo test -p surgeist-render clip
cargo test -p surgeist-render image
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
- transform helpers, origin wrapping, skew behavior, 3D diagnostics, and coordinate-space tag models
- tests and required checks

Required final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Completion for this sequence item requires a clean holistic review and all required final checks passing.
