# Image Resource Sampling Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the render-owned image resource and CSS image sampling primitives required by Sequence 6 of the CSS/style support matrix.

**Architecture:** Keep render self-contained by normalizing already-resolved image resources, placement inputs, repeat inputs, attachment coordinate tags, and filtered-image boundaries into render-owned models. Vello continues to handle simple image draws; CSS sampling semantics are represented as validated normalization results and precise unsupported diagnostics where this phase cannot execute the primitive yet.

**Tech Stack:** Rust 2024, surgeist-render public value models, Vello 0.9 capability reporting, existing `Error`/`UnsupportedPrimitive` diagnostics, crate-local unit tests.

---

## Source Guidance

Required references:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-render-css-implementation-sequence.md`

Sequence 6 covers:

- Image paint
- Resolved image handle
- Intrinsic image metadata
- Image fit
- Background position
- Background size
- Repeat/no-repeat/round/space
- Background attachment coordinate input
- Filtered image paint boundary and diagnostics
- Image orientation and color-profile ownership policy

Render boundary decisions for this phase:

- Render does not load URLs.
- Root supplies resolved image handles or in-memory images.
- Root lowers CSS parser keywords into render `BackgroundPosition`, `BackgroundSize`, `BackgroundRepeat`, `BackgroundAttachment`, and `CoordinateSpaceTag` values. Render still owns the final math for positioned image tiles because tile size is known only after background-size normalization.
- Four-component CSS positions such as `right 10px bottom 20px` are represented as render-owned edge-offset components, not as parser tokens.
- `PositionComponent::Percent` and `SizeComponentKind::Percent` are normalized ratios (`0.25` means 25%), matching existing tests.
- Image orientation and color-profile conversion are root-resolved-only in this phase. Render reports typed diagnostics if asked to own conversion.
- Animated image frame selection remains root/runtime-owned and out of render scope.
- Filtered image paint is represented as a resolved image plus `FilterList`, but execution remains unsupported until the filter pipeline phase.
- Backwards compatibility shims are not required.

## File Responsibilities

- `src/capability.rs`: add the image-sampling capability family, Vello 0.9 baseline values, accessors, and unsupported-operation mapping.
- `src/error.rs`: add `PrimitiveFamily::ImageSampling` plus typed `PrimitiveOperation` variants for unsupported CSS image sampling boundaries.
- `src/style.rs`: add render-owned image resource policy/metadata, unresolved image diagnostics, placement normalization, repeat geometry normalization, attachment coordinate policy, and filtered-image paint boundary models.
- `src/lib.rs`: export new public image sampling types.
- `src/tests.rs`: add unit tests for capabilities, diagnostics, intrinsic sizing, placement math, repeat tile/clipping geometry, attachment coordinate behavior, filtered-image boundary, and ownership policy.

Do not modify sibling crates. Do not add dependencies. Do not alter backend encoding unless a test proves an existing image draw regression.

## Task 1: Capability And Diagnostic Surface

**Files:**

- Modify: `src/error.rs`
- Modify: `src/capability.rs`
- Modify: `src/lib.rs` only if new public capability-owned enums are added outside existing exports
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for image sampling capabilities and diagnostics**

Add tests near the existing capability tests in `src/tests.rs`:

```rust
#[test]
fn image_sampling_capabilities_name_css_sampling_boundaries() {
    let capabilities = Capabilities::VELLO_0_9.image_sampling();

    assert!(capabilities.supports_image_fit());
    assert!(capabilities.supports_background_position());
    assert!(capabilities.supports_background_size());
    assert!(capabilities.supports_repeat_xy());
    assert_eq!(
        capabilities.attachment_coordinate_policy(),
        BackgroundAttachmentCoordinatePolicy::RootResolvedOrTagged
    );
    assert_eq!(
        capabilities.image_orientation_policy(),
        ImageOrientationPolicy::RootResolvedOnly
    );
    assert_eq!(
        capabilities.image_color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
    assert!(!capabilities.supports_repeat_round());
    assert!(!capabilities.supports_repeat_space());
    assert!(!capabilities.supports_filtered_image_paint());
    assert!(!capabilities.supports_image_orientation_conversion());
    assert!(!capabilities.supports_image_color_profile_conversion());
}

#[test]
fn unsupported_image_sampling_operations_report_typed_diagnostics() {
    for operation in [
        PrimitiveOperation::BackgroundRepeatRound,
        PrimitiveOperation::BackgroundRepeatSpace,
        PrimitiveOperation::FilteredImagePaint,
        PrimitiveOperation::ImageOrientationConversion,
        PrimitiveOperation::ImageColorProfileConversion,
    ] {
        let unsupported = UnsupportedPrimitive::new(PrimitiveFamily::ImageSampling, operation);
        let error = Capabilities::VELLO_0_9
            .ensure_supported(unsupported)
            .expect_err("Vello baseline should reject this image sampling primitive");

        assert_eq!(error.code, ErrorCode::UnsupportedBackend);
        assert_eq!(error.unsupported_primitive(), Some(unsupported));
        assert!(error.message.contains(unsupported.label()));
    }
}
```

Run:

```sh
cargo test -p surgeist-render image_sampling_capabilities_name_css_sampling_boundaries unsupported_image_sampling_operations_report_typed_diagnostics
```

Expected: fail to compile because the new capability family, policies, and operations do not exist.

- [ ] **Step 2: Add typed diagnostic operations**

In `src/error.rs`, add `ImageSampling` to `PrimitiveFamily`:

```rust
ImageSampling,
```

Update `PrimitiveFamily::label`:

```rust
Self::ImageSampling => "image sampling",
```

Add these `PrimitiveOperation` variants:

```rust
BackgroundRepeatRound,
BackgroundRepeatSpace,
FilteredImagePaint,
ImageOrientationConversion,
ImageColorProfileConversion,
```

Update `PrimitiveOperation::label`:

```rust
Self::BackgroundRepeatRound => "background repeat round",
Self::BackgroundRepeatSpace => "background repeat space",
Self::FilteredImagePaint => "filtered image paint",
Self::ImageOrientationConversion => "image orientation conversion",
Self::ImageColorProfileConversion => "image color profile conversion",
```

- [ ] **Step 3: Add image sampling capabilities**

In `src/capability.rs`, extend `Capabilities`:

```rust
image_sampling: ImageSamplingCapabilities,
```

Add this field to `Capabilities::VELLO_0_9`:

```rust
image_sampling: ImageSamplingCapabilities {
    image_fit: true,
    background_position: true,
    background_size: true,
    repeat_xy: true,
    repeat_round: false,
    repeat_space: false,
    filtered_image_paint: false,
    image_orientation_conversion: false,
    image_color_profile_conversion: false,
    attachment_coordinate_policy: BackgroundAttachmentCoordinatePolicy::RootResolvedOrTagged,
    image_orientation_policy: ImageOrientationPolicy::RootResolvedOnly,
    image_color_profile_policy: ImageColorProfilePolicy::RootResolvedOnly,
},
```

Add an accessor:

```rust
#[must_use]
pub const fn image_sampling(self) -> ImageSamplingCapabilities {
    self.image_sampling
}
```

Extend `Capabilities::supports`:

```rust
(PrimitiveFamily::ImageSampling, PrimitiveOperation::BackgroundRepeatRound) => {
    self.image_sampling.supports_repeat_round()
}
(PrimitiveFamily::ImageSampling, PrimitiveOperation::BackgroundRepeatSpace) => {
    self.image_sampling.supports_repeat_space()
}
(PrimitiveFamily::ImageSampling, PrimitiveOperation::FilteredImagePaint) => {
    self.image_sampling.supports_filtered_image_paint()
}
(PrimitiveFamily::ImageSampling, PrimitiveOperation::ImageOrientationConversion) => {
    self.image_sampling.supports_image_orientation_conversion()
}
(PrimitiveFamily::ImageSampling, PrimitiveOperation::ImageColorProfileConversion) => {
    self.image_sampling.supports_image_color_profile_conversion()
}
```

Add the new types in `src/capability.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundAttachmentCoordinatePolicy {
    RootResolvedOrTagged,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageOrientationPolicy {
    RootResolvedOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageColorProfilePolicy {
    RootResolvedOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageSamplingCapabilities {
    image_fit: bool,
    background_position: bool,
    background_size: bool,
    repeat_xy: bool,
    repeat_round: bool,
    repeat_space: bool,
    filtered_image_paint: bool,
    image_orientation_conversion: bool,
    image_color_profile_conversion: bool,
    attachment_coordinate_policy: BackgroundAttachmentCoordinatePolicy,
    image_orientation_policy: ImageOrientationPolicy,
    image_color_profile_policy: ImageColorProfilePolicy,
}
```

Add const accessors:

```rust
impl ImageSamplingCapabilities {
    #[must_use]
    pub const fn supports_image_fit(self) -> bool { self.image_fit }
    #[must_use]
    pub const fn supports_background_position(self) -> bool { self.background_position }
    #[must_use]
    pub const fn supports_background_size(self) -> bool { self.background_size }
    #[must_use]
    pub const fn supports_repeat_xy(self) -> bool { self.repeat_xy }
    #[must_use]
    pub const fn supports_repeat_round(self) -> bool { self.repeat_round }
    #[must_use]
    pub const fn supports_repeat_space(self) -> bool { self.repeat_space }
    #[must_use]
    pub const fn supports_filtered_image_paint(self) -> bool { self.filtered_image_paint }
    #[must_use]
    pub const fn supports_image_orientation_conversion(self) -> bool {
        self.image_orientation_conversion
    }
    #[must_use]
    pub const fn supports_image_color_profile_conversion(self) -> bool {
        self.image_color_profile_conversion
    }
    #[must_use]
    pub const fn attachment_coordinate_policy(self) -> BackgroundAttachmentCoordinatePolicy {
        self.attachment_coordinate_policy
    }
    #[must_use]
    pub const fn image_orientation_policy(self) -> ImageOrientationPolicy {
        self.image_orientation_policy
    }
    #[must_use]
    pub const fn image_color_profile_policy(self) -> ImageColorProfilePolicy {
        self.image_color_profile_policy
    }
}
```

- [ ] **Step 4: Export new public capability types**

In `src/lib.rs`, add the new types to the existing capability export list:

```rust
BackgroundAttachmentCoordinatePolicy, ImageColorProfilePolicy, ImageOrientationPolicy,
ImageSamplingCapabilities,
```

- [ ] **Step 5: Run focused checks**

Run:

```sh
cargo test -p surgeist-render image_sampling_capabilities_name_css_sampling_boundaries unsupported_image_sampling_operations_report_typed_diagnostics
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 2: Resource Metadata, Missing Resource Diagnostics, And Ownership Policy Models

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for resolved metadata, unresolved image diagnostics, and root-owned conversion policy**

Add tests near the existing `ResolvedImageResource` tests:

```rust
#[test]
fn resolved_image_resources_carry_root_resolved_metadata_policy() {
    let resource = ResolvedImageResource::try_new(ImageId::new(12), Size::new(40.0, 20.0))
        .unwrap()
        .with_density(ImageResourceDensity::try_new(2.0).unwrap());

    assert_eq!(resource.id(), ImageId::new(12));
    assert_eq!(resource.intrinsic_size(), Size::new(40.0, 20.0));
    assert_eq!(resource.density().map(ImageResourceDensity::value), Some(2.0));
    assert_eq!(resource.orientation_policy(), ImageOrientationPolicy::RootResolvedOnly);
    assert_eq!(
        resource.color_profile_policy(),
        ImageColorProfilePolicy::RootResolvedOnly
    );
}

#[test]
fn image_resource_density_rejects_invalid_values() {
    let error = ImageResourceDensity::try_new(0.0)
        .expect_err("image density must be positive when supplied");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image resource density")
    );
}

#[test]
fn unresolved_style_image_sources_report_image_resource_diagnostics() {
    let reference = StyleResourceRef::try_new("hero.png").unwrap();
    let source = StyleImageSource::unresolved(reference.clone());

    assert_eq!(
        source.kind(),
        &StyleImageSourceKind::Unresolved(reference.clone())
    );

    let error = source
        .require_resolved()
        .expect_err("unresolved image source must report an image resource diagnostic");
    assert_eq!(error.code, ErrorCode::UnresolvedResource);
    assert_eq!(
        error.unresolved_resource_diagnostic(),
        Some(&UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            reference.identifier()
        ))
    );
}
```

Run:

```sh
cargo test -p surgeist-render resolved_image_resources_carry_root_resolved_metadata_policy image_resource_density_rejects_invalid_values unresolved_style_image_sources_report_image_resource_diagnostics
```

Expected: fail to compile because `ImageResourceDensity`, metadata accessors, unresolved image sources, and `StyleImageSource::require_resolved` do not exist.

- [ ] **Step 2: Add density and policy accessors**

In `src/style.rs`, import the capability policy enums if needed:

```rust
ImageColorProfilePolicy, ImageOrientationPolicy,
```

Change `ResolvedImageResource` to:

```rust
pub struct ResolvedImageResource {
    id: ImageId,
    intrinsic_size: Size,
    density: Option<ImageResourceDensity>,
}
```

Update `try_new`:

```rust
Ok(Self {
    id,
    intrinsic_size,
    density: None,
})
```

Add:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageResourceDensity {
    value: f64,
}

impl ImageResourceDensity {
    pub fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_value(
                "image resource density",
                value,
                "must be finite and positive",
            ));
        }
        Ok(Self { value })
    }

    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}
```

Add `ResolvedImageResource` methods:

```rust
#[must_use]
pub const fn with_density(mut self, density: ImageResourceDensity) -> Self {
    self.density = Some(density);
    self
}

#[must_use]
pub const fn density(&self) -> Option<ImageResourceDensity> {
    self.density
}

#[must_use]
pub const fn orientation_policy(&self) -> ImageOrientationPolicy {
    ImageOrientationPolicy::RootResolvedOnly
}

#[must_use]
pub const fn color_profile_policy(&self) -> ImageColorProfilePolicy {
    ImageColorProfilePolicy::RootResolvedOnly
}
```

- [ ] **Step 3: Add unresolved image source diagnostics**

In `src/style.rs`, extend `StyleImageSourceKind`:

```rust
Unresolved(StyleResourceRef),
```

Add methods on `StyleImageSource`:

```rust
#[must_use]
pub fn unresolved(reference: StyleResourceRef) -> Self {
    Self {
        kind: StyleImageSourceKind::Unresolved(reference),
    }
}

pub fn require_resolved(&self) -> Result<()> {
    if let StyleImageSourceKind::Unresolved(reference) = &self.kind {
        return Err(Error::unresolved_resource(UnresolvedResource::new(
            UnresolvedResourceKind::Image,
            reference.identifier(),
        )));
    }
    Ok(())
}
```

Add imports for `UnresolvedResource` and `UnresolvedResourceKind`.

- [ ] **Step 4: Export new metadata type**

In `src/lib.rs`, add `ImageResourceDensity` to the style export list.

- [ ] **Step 5: Run focused checks**

Run:

```sh
cargo test -p surgeist-render resolved_image_resources_carry_root_resolved_metadata_policy image_resource_density_rejects_invalid_values unresolved_style_image_sources_report_image_resource_diagnostics resolved_image_resources_preserve_handle_and_intrinsic_size
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 3: Background Image Placement Normalization

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for background size and position normalization**

Add tests near the image layer tests:

```rust
#[test]
fn image_placement_auto_uses_intrinsic_size_and_position_ratio() {
    let input = ImagePlacementInput::try_new(
        Rect::new(10.0, 20.0, 100.0, 50.0),
        Size::new(20.0, 10.0),
        BackgroundPosition::percent(0.5, 1.0).unwrap(),
        BackgroundSize::auto(),
    )
    .unwrap();

    let placement = input.resolve().unwrap();

    assert_eq!(placement.paint_rect(), Rect::new(10.0, 20.0, 100.0, 50.0));
    assert_eq!(placement.tile_rect(), Rect::new(50.0, 60.0, 20.0, 10.0));
}

#[test]
fn image_placement_cover_and_contain_preserve_aspect_ratio() {
    let paint_rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let intrinsic = Size::new(20.0, 20.0);

    let cover = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::cover(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(cover.tile_rect(), Rect::new(0.0, -25.0, 100.0, 100.0));

    let contain = ImagePlacementInput::try_new(
        paint_rect,
        intrinsic,
        BackgroundPosition::percent(0.5, 0.5).unwrap(),
        BackgroundSize::contain(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(contain.tile_rect(), Rect::new(25.0, 0.0, 50.0, 50.0));
}

#[test]
fn image_placement_explicit_size_resolves_lengths_percents_and_auto_axis() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::length(5.0, 10.0).unwrap(),
        BackgroundSize::explicit(
            SizeComponent::try_percent(0.5).unwrap(),
            SizeComponent::auto(),
        ),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(5.0, 10.0, 100.0, 50.0));
}

#[test]
fn image_placement_edge_offsets_represent_four_component_positions() {
    let placement = ImagePlacementInput::try_new(
        Rect::new(-20.0, -10.0, 200.0, 100.0),
        Size::new(40.0, 20.0),
        BackgroundPosition::edge_offsets(
            PositionEdgeOffset::end(15.0).unwrap(),
            PositionEdgeOffset::end(5.0).unwrap(),
        ),
        BackgroundSize::auto(),
    )
    .unwrap()
    .resolve()
    .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(125.0, 65.0, 40.0, 20.0));
}

#[test]
fn image_placement_rejects_invalid_paint_or_intrinsic_size() {
    let error = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 0.0, 100.0),
        Size::new(10.0, 10.0),
        BackgroundPosition::default(),
        BackgroundSize::auto(),
    )
    .expect_err("paint rect must be positive");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("image placement paint rect")
    );
}
```

Run:

```sh
cargo test -p surgeist-render image_placement
```

Expected: fail to compile because the placement and edge-offset position types do not exist.

- [ ] **Step 2: Add edge-offset position components**

In `src/style.rs`, add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionEdge {
    Start,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionEdgeOffset {
    edge: PositionEdge,
    offset: f64,
}
```

Add constructors and accessors:

```rust
impl PositionEdgeOffset {
    pub fn start(offset: f64) -> Result<Self> {
        Self::try_new(PositionEdge::Start, offset)
    }

    pub fn end(offset: f64) -> Result<Self> {
        Self::try_new(PositionEdge::End, offset)
    }

    fn try_new(edge: PositionEdge, offset: f64) -> Result<Self> {
        if !offset.is_finite() {
            return Err(Error::invalid_value(
                "background position edge offset",
                offset,
                "must be finite",
            ));
        }
        Ok(Self { edge, offset })
    }

    #[must_use]
    pub const fn edge(self) -> PositionEdge {
        self.edge
    }

    #[must_use]
    pub const fn offset(self) -> f64 {
        self.offset
    }
}
```

Extend `PositionComponentKind`:

```rust
EdgeOffset(PositionEdge),
```

Add `PositionComponent::edge_offset`:

```rust
#[must_use]
pub const fn edge_offset(offset: PositionEdgeOffset) -> Self {
    Self {
        kind: PositionComponentKind::EdgeOffset(offset.edge()),
        value: offset.offset(),
    }
}
```

Add `BackgroundPosition::edge_offsets`:

```rust
#[must_use]
pub const fn edge_offsets(x: PositionEdgeOffset, y: PositionEdgeOffset) -> Self {
    Self {
        x: PositionComponent::edge_offset(x),
        y: PositionComponent::edge_offset(y),
    }
}
```

- [ ] **Step 3: Add placement input and output models**

In `src/style.rs`, add `Rect` to the existing imports.

Add:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePlacementInput {
    paint_rect: Rect,
    intrinsic_size: Size,
    position: BackgroundPosition,
    size: BackgroundSize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedImagePlacement {
    paint_rect: Rect,
    tile_rect: Rect,
}
```

Add constructors and accessors:

```rust
impl ImagePlacementInput {
    pub fn try_new(
        paint_rect: Rect,
        intrinsic_size: Size,
        position: BackgroundPosition,
        size: BackgroundSize,
    ) -> Result<Self> {
        validate_finite_f64(paint_rect.x(), "image placement paint rect x")?;
        validate_finite_f64(paint_rect.y(), "image placement paint rect y")?;
        if paint_rect.width() <= 0.0
            || paint_rect.height() <= 0.0
            || !paint_rect.width().is_finite()
            || !paint_rect.height().is_finite()
        {
            return Err(Error::invalid_value(
                "image placement paint rect",
                format!("{paint_rect:?}"),
                "must have finite positive width and height",
            ));
        }
        validate_size(intrinsic_size, "image placement intrinsic size")?;
        Ok(Self {
            paint_rect,
            intrinsic_size,
            position,
            size,
        })
    }

    #[must_use]
    pub const fn paint_rect(self) -> Rect { self.paint_rect }

    #[must_use]
    pub const fn intrinsic_size(self) -> Size { self.intrinsic_size }

    #[must_use]
    pub const fn position(self) -> BackgroundPosition { self.position }

    #[must_use]
    pub const fn size(self) -> BackgroundSize { self.size }

    pub fn resolve(self) -> Result<ResolvedImagePlacement> {
        let tile_size = resolve_background_size(self.paint_rect, self.intrinsic_size, self.size)?;
        let tile_x = resolve_position_component(
            self.paint_rect.x(),
            self.paint_rect.width(),
            tile_size.width(),
            self.position.x(),
        );
        let tile_y = resolve_position_component(
            self.paint_rect.y(),
            self.paint_rect.height(),
            tile_size.height(),
            self.position.y(),
        );
        Ok(ResolvedImagePlacement {
            paint_rect: self.paint_rect,
            tile_rect: Rect::new(tile_x, tile_y, tile_size.width(), tile_size.height()),
        })
    }
}

impl ResolvedImagePlacement {
    pub fn from_parts(paint_rect: Rect, tile_rect: Rect) -> Result<Self> {
        if paint_rect.width() <= 0.0
            || paint_rect.height() <= 0.0
            || !paint_rect.x().is_finite()
            || !paint_rect.y().is_finite()
            || !paint_rect.width().is_finite()
            || !paint_rect.height().is_finite()
        {
            return Err(Error::invalid_value(
                "image placement paint rect",
                format!("{paint_rect:?}"),
                "must have finite positive width and height",
            ));
        }
        if tile_rect.width() <= 0.0
            || tile_rect.height() <= 0.0
            || !tile_rect.x().is_finite()
            || !tile_rect.y().is_finite()
            || !tile_rect.width().is_finite()
            || !tile_rect.height().is_finite()
        {
            return Err(Error::invalid_value(
                "image placement tile rect",
                format!("{tile_rect:?}"),
                "must have finite positive width and height",
            ));
        }
        Ok(Self {
            paint_rect,
            tile_rect,
        })
    }

    #[must_use]
    pub const fn paint_rect(self) -> Rect { self.paint_rect }

    #[must_use]
    pub const fn tile_rect(self) -> Rect { self.tile_rect }
}
```

Add helper functions in `src/style.rs`:

```rust
fn resolve_background_size(
    paint_rect: Rect,
    intrinsic_size: Size,
    size: BackgroundSize,
) -> Result<Size> {
    let intrinsic_width = intrinsic_size.width();
    let intrinsic_height = intrinsic_size.height();
    let scale_x = paint_rect.width() / intrinsic_width;
    let scale_y = paint_rect.height() / intrinsic_height;
    match size.kind() {
        BackgroundSizeKind::Auto => Ok(intrinsic_size),
        BackgroundSizeKind::Cover => {
            let scale = scale_x.max(scale_y);
            Ok(Size::new(intrinsic_width * scale, intrinsic_height * scale))
        }
        BackgroundSizeKind::Contain => {
            let scale = scale_x.min(scale_y);
            Ok(Size::new(intrinsic_width * scale, intrinsic_height * scale))
        }
        BackgroundSizeKind::Explicit { width, height } => {
            let width = resolve_size_component(width, paint_rect.width());
            let height = resolve_size_component(height, paint_rect.height());
            match (width, height) {
                (Some(width), Some(height)) => Ok(Size::new(width, height)),
                (Some(width), None) => Ok(Size::new(width, width * intrinsic_height / intrinsic_width)),
                (None, Some(height)) => Ok(Size::new(height * intrinsic_width / intrinsic_height, height)),
                (None, None) => Ok(intrinsic_size),
            }
        }
    }
}

fn resolve_size_component(component: SizeComponent, axis: f64) -> Option<f64> {
    match component.kind() {
        SizeComponentKind::Auto => None,
        SizeComponentKind::Length(value) => Some(value),
        SizeComponentKind::Percent(value) => Some(axis * value),
    }
}

fn resolve_position_component(origin: f64, axis: f64, tile_axis: f64, component: PositionComponent) -> f64 {
    match component.kind() {
        PositionComponentKind::Length => origin + component.value(),
        PositionComponentKind::Percent => origin + (axis - tile_axis) * component.value(),
        PositionComponentKind::EdgeOffset(PositionEdge::Start) => origin + component.value(),
        PositionComponentKind::EdgeOffset(PositionEdge::End) => {
            origin + axis - tile_axis - component.value()
        }
    }
}
```

If `cargo clippy` asks for formatting around long helper signatures, run `cargo fmt`.

- [ ] **Step 4: Export placement and edge-offset types**

In `src/lib.rs`, export:

```rust
ImagePlacementInput, PositionEdge, PositionEdgeOffset, ResolvedImagePlacement,
```

- [ ] **Step 5: Run focused checks**

Run:

```sh
cargo test -p surgeist-render image_placement
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 4: Repeat Geometry Normalization

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for repeat/no-repeat/repeat-x/repeat-y geometry, clipping, and unsupported round/space**

Add tests near the image placement tests:

```rust
#[test]
fn image_repeat_plan_maps_css_repeat_axes() {
    let cases = [
        (BackgroundRepeat::no_repeat(), ImageRepeatMode::NoRepeat),
        (BackgroundRepeat::repeat_x(), ImageRepeatMode::RepeatX),
        (BackgroundRepeat::repeat_y(), ImageRepeatMode::RepeatY),
        (BackgroundRepeat::repeat(), ImageRepeatMode::RepeatBoth),
    ];

    for (repeat, expected) in cases {
        let plan = ImageRepeatPlan::try_new(repeat, Capabilities::VELLO_0_9).unwrap();
        assert_eq!(plan.repeat(), repeat);
        assert_eq!(plan.mode(), expected);
    }
}

#[test]
fn image_repeat_plan_resolves_tile_rects_inside_clip_rect() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 70.0, 40.0),
        Rect::new(0.0, 5.0, 20.0, 10.0),
    )
    .unwrap();

    let repeat_x = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(repeat_x.clip_rect(), Rect::new(0.0, 0.0, 70.0, 40.0));
    assert_eq!(
        repeat_x.tile_rects(),
        &[
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(20.0, 5.0, 20.0, 10.0),
            Rect::new(40.0, 5.0, 20.0, 10.0),
            Rect::new(60.0, 5.0, 20.0, 10.0),
        ]
    );

    let repeat_y = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_y(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    assert_eq!(
        repeat_y.tile_rects(),
        &[
            Rect::new(0.0, 5.0, 20.0, 10.0),
            Rect::new(0.0, 15.0, 20.0, 10.0),
            Rect::new(0.0, 25.0, 20.0, 10.0),
            Rect::new(0.0, 35.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_includes_tiles_before_the_anchor_when_visible() {
    let placement = ResolvedImagePlacement::from_parts(
        Rect::new(0.0, 0.0, 50.0, 20.0),
        Rect::new(15.0, 0.0, 20.0, 10.0),
    )
    .unwrap();

    let repeated = ImageRepeatPlan::try_new(BackgroundRepeat::repeat_x(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();

    assert_eq!(
        repeated.tile_rects(),
        &[
            Rect::new(-5.0, 0.0, 20.0, 10.0),
            Rect::new(15.0, 0.0, 20.0, 10.0),
            Rect::new(35.0, 0.0, 20.0, 10.0),
        ]
    );
}

#[test]
fn image_repeat_plan_rejects_round_and_space_with_typed_diagnostics() {
    let round = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::Round, RepeatMode::Repeat),
        Capabilities::VELLO_0_9,
    )
    .expect_err("round repeat is not supported yet");
    assert_eq!(
        round.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatRound
        ))
    );

    let space = ImageRepeatPlan::try_new(
        BackgroundRepeat::new(RepeatMode::NoRepeat, RepeatMode::Space),
        Capabilities::VELLO_0_9,
    )
    .expect_err("space repeat is not supported yet");
    assert_eq!(
        space.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::BackgroundRepeatSpace
        ))
    );
}
```

Run:

```sh
cargo test -p surgeist-render image_repeat_plan
```

Expected: fail to compile because `ImageRepeatPlan`, `ImageRepeatMode`, and resolved repeat geometry do not exist.

- [ ] **Step 2: Add repeat plan model**

In `src/style.rs`, import:

```rust
Capabilities, PrimitiveFamily, PrimitiveOperation, UnsupportedPrimitive,
```

Add:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageRepeatMode {
    NoRepeat,
    RepeatX,
    RepeatY,
    RepeatBoth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageRepeatPlan {
    repeat: BackgroundRepeat,
    mode: ImageRepeatMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedImageRepeat {
    clip_rect: Rect,
    tile_rects: Vec<Rect>,
}
```

Add:

```rust
impl ImageRepeatPlan {
    pub fn try_new(repeat: BackgroundRepeat, capabilities: Capabilities) -> Result<Self> {
        if matches!(repeat.x(), RepeatMode::Round) || matches!(repeat.y(), RepeatMode::Round) {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::ImageSampling,
                PrimitiveOperation::BackgroundRepeatRound,
            ))?;
        }
        if matches!(repeat.x(), RepeatMode::Space) || matches!(repeat.y(), RepeatMode::Space) {
            capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::ImageSampling,
                PrimitiveOperation::BackgroundRepeatSpace,
            ))?;
        }
        let mode = match (repeat.x(), repeat.y()) {
            (RepeatMode::NoRepeat, RepeatMode::NoRepeat) => ImageRepeatMode::NoRepeat,
            (RepeatMode::Repeat, RepeatMode::NoRepeat) => ImageRepeatMode::RepeatX,
            (RepeatMode::NoRepeat, RepeatMode::Repeat) => ImageRepeatMode::RepeatY,
            (RepeatMode::Repeat, RepeatMode::Repeat) => ImageRepeatMode::RepeatBoth,
            _ => unreachable!("round and space are handled before mode mapping"),
        };
        Ok(Self { repeat, mode })
    }

    #[must_use]
    pub const fn repeat(self) -> BackgroundRepeat { self.repeat }

    #[must_use]
    pub const fn mode(self) -> ImageRepeatMode { self.mode }

    pub fn resolve(self, placement: ResolvedImagePlacement) -> Result<ResolvedImageRepeat> {
        let mut x_positions = repeat_positions(
            placement.paint_rect().x(),
            placement.paint_rect().width(),
            placement.tile_rect().x(),
            placement.tile_rect().width(),
            matches!(self.mode, ImageRepeatMode::RepeatX | ImageRepeatMode::RepeatBoth),
        )?;
        let mut y_positions = repeat_positions(
            placement.paint_rect().y(),
            placement.paint_rect().height(),
            placement.tile_rect().y(),
            placement.tile_rect().height(),
            matches!(self.mode, ImageRepeatMode::RepeatY | ImageRepeatMode::RepeatBoth),
        )?;
        if x_positions.is_empty() {
            x_positions.push(placement.tile_rect().x());
        }
        if y_positions.is_empty() {
            y_positions.push(placement.tile_rect().y());
        }
        let mut tile_rects = Vec::new();
        for y in y_positions {
            for x in &x_positions {
                tile_rects.push(Rect::new(
                    *x,
                    y,
                    placement.tile_rect().width(),
                    placement.tile_rect().height(),
                ));
            }
        }
        Ok(ResolvedImageRepeat {
            clip_rect: placement.paint_rect(),
            tile_rects,
        })
    }
}

impl ResolvedImageRepeat {
    #[must_use]
    pub const fn clip_rect(&self) -> Rect {
        self.clip_rect
    }

    #[must_use]
    pub fn tile_rects(&self) -> &[Rect] {
        &self.tile_rects
    }
}
```

Add helper:

```rust
fn repeat_positions(
    clip_origin: f64,
    clip_axis: f64,
    tile_origin: f64,
    tile_axis: f64,
    repeats: bool,
) -> Result<Vec<f64>> {
    if tile_axis <= 0.0 || !tile_axis.is_finite() {
        return Err(Error::invalid_value(
            "image repeat tile size",
            tile_axis,
            "must be finite and positive",
        ));
    }
    if !repeats {
        return Ok(vec![tile_origin]);
    }
    let clip_end = clip_origin + clip_axis;
    let mut origin = tile_origin;
    while origin > clip_origin {
        origin -= tile_axis;
    }
    let mut positions = Vec::new();
    while origin < clip_end {
        if origin + tile_axis > clip_origin {
            positions.push(origin);
        }
        origin += tile_axis;
    }
    Ok(positions)
}
```

- [ ] **Step 3: Export repeat types**

In `src/lib.rs`, export:

```rust
ImageRepeatMode, ImageRepeatPlan, ResolvedImageRepeat,
```

- [ ] **Step 4: Run focused checks**

Run:

```sh
cargo test -p surgeist-render image_repeat_plan
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 5: Attachment Coordinate And Filtered Image Boundary

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for attachment coordinate resolution**

Add tests near the fixed background coordinate tests:

```rust
#[test]
fn image_attachment_plan_uses_root_resolved_scroll_and_local_coordinates() {
    let scroll = ImageAttachmentPlan::try_new(BackgroundAttachment::Scroll, None).unwrap();
    assert_eq!(scroll.attachment(), BackgroundAttachment::Scroll);
    assert_eq!(scroll.coordinate_space().map(CoordinateSpaceTag::kind), None);

    let local_tag = CoordinateSpaceTag::local();
    let local = ImageAttachmentPlan::try_new(BackgroundAttachment::Local, Some(local_tag)).unwrap();
    assert_eq!(local.attachment(), BackgroundAttachment::Local);
    assert_eq!(
        local.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Local)
    );
}

#[test]
fn fixed_image_attachment_requires_viewport_coordinate_tag() {
    let missing = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, None)
        .expect_err("fixed backgrounds require an explicit viewport tag");
    assert_eq!(missing.code, ErrorCode::InvalidInput);
    assert_eq!(
        missing.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let surface = CoordinateSpaceTag::surface(Transform::identity()).unwrap();
    let wrong = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(surface))
        .expect_err("fixed backgrounds must be tagged in viewport coordinates");
    assert_eq!(
        wrong.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background attachment coordinate space")
    );

    let viewport =
        CoordinateSpaceTag::viewport(Transform::translation(3.0, 4.0).unwrap()).unwrap();
    let fixed = ImageAttachmentPlan::try_new(BackgroundAttachment::Fixed, Some(viewport)).unwrap();
    assert_eq!(fixed.attachment(), BackgroundAttachment::Fixed);
    assert_eq!(
        fixed.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}
```

Run:

```sh
cargo test -p surgeist-render image_attachment_plan fixed_image_attachment_requires_viewport_coordinate_tag
```

Expected: fail to compile because `ImageAttachmentPlan` does not exist.

- [ ] **Step 2: Add attachment plan model**

In `src/style.rs`, add `CoordinateSpaceKind` to imports.

Add:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageAttachmentPlan {
    attachment: BackgroundAttachment,
    coordinate_space: Option<CoordinateSpaceTag>,
}
```

Add:

```rust
impl ImageAttachmentPlan {
    pub fn try_new(
        attachment: BackgroundAttachment,
        coordinate_space: Option<CoordinateSpaceTag>,
    ) -> Result<Self> {
        if matches!(attachment, BackgroundAttachment::Fixed) {
            let Some(tag) = coordinate_space else {
                return Err(Error::invalid_value(
                    "background attachment coordinate space",
                    "none",
                    "fixed backgrounds require a viewport coordinate tag",
                ));
            };
            if tag.kind() != CoordinateSpaceKind::Viewport {
                return Err(Error::invalid_value(
                    "background attachment coordinate space",
                    format!("{:?}", tag.kind()),
                    "fixed backgrounds require a viewport coordinate tag",
                ));
            }
        }
        Ok(Self {
            attachment,
            coordinate_space,
        })
    }

    #[must_use]
    pub const fn attachment(self) -> BackgroundAttachment { self.attachment }

    #[must_use]
    pub const fn coordinate_space(self) -> Option<CoordinateSpaceTag> { self.coordinate_space }
}
```

- [ ] **Step 3: Add failing tests for filtered image paint boundary**

Add tests near `FilterList` tests:

```rust
#[test]
fn filtered_image_paint_preserves_resolved_image_and_filter_list() {
    let resource = ResolvedImageResource::try_new(ImageId::new(30), Size::new(16.0, 16.0))
        .unwrap();
    let filters = FilterList::try_ops(vec![FilterOp::brightness(
        FilterAmount::try_new(1.25).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource.clone(), filters.clone()).unwrap();

    assert_eq!(paint.resource(), &resource);
    assert_eq!(paint.filters(), &filters);
}

#[test]
fn filtered_image_paint_rejects_none_filter_list_and_reports_execution_boundary() {
    let resource = ResolvedImageResource::try_new(ImageId::new(31), Size::new(8.0, 8.0)).unwrap();
    let error = FilteredImagePaint::try_new(resource.clone(), FilterList::none())
        .expect_err("filtered image paint requires a non-empty filter list");
    assert_eq!(error.code, ErrorCode::InvalidInput);

    let filters = FilterList::try_ops(vec![FilterOp::contrast(
        FilterAmount::try_new(0.75).unwrap(),
    )])
    .unwrap();
    let paint = FilteredImagePaint::try_new(resource, filters).unwrap();
    let unsupported = paint
        .ensure_supported(Capabilities::VELLO_0_9)
        .expect_err("filtered image paint execution belongs to filter phases");
    assert_eq!(
        unsupported.unsupported_primitive(),
        Some(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint
        ))
    );
}
```

Run:

```sh
cargo test -p surgeist-render filtered_image_paint
```

Expected: fail to compile because `FilteredImagePaint` does not exist.

- [ ] **Step 4: Add filtered image paint model**

In `src/style.rs`, add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredImagePaint {
    resource: ResolvedImageResource,
    filters: FilterList,
}
```

Add:

```rust
impl FilteredImagePaint {
    pub fn try_new(resource: ResolvedImageResource, filters: FilterList) -> Result<Self> {
        if filters.is_none() {
            return Err(Error::invalid_value(
                "filtered image paint filters",
                "none",
                "must contain at least one filter operation",
            ));
        }
        Ok(Self { resource, filters })
    }

    #[must_use]
    pub const fn resource(&self) -> &ResolvedImageResource { &self.resource }

    #[must_use]
    pub const fn filters(&self) -> &FilterList { &self.filters }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        capabilities.ensure_supported(UnsupportedPrimitive::new(
            PrimitiveFamily::ImageSampling,
            PrimitiveOperation::FilteredImagePaint,
        ))
    }
}
```

- [ ] **Step 5: Export attachment and filtered-image types**

In `src/lib.rs`, export:

```rust
FilteredImagePaint, ImageAttachmentPlan,
```

- [ ] **Step 6: Run focused checks**

Run:

```sh
cargo test -p surgeist-render image_attachment_plan fixed_image_attachment_requires_viewport_coordinate_tag filtered_image_paint
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 6: Integration Coverage And Existing Image Fit Guardrails

**Files:**

- Modify: `src/tests.rs`
- No production edits unless tests reveal a regression

- [ ] **Step 1: Add integration-style tests for combined image sampling normalization**

Add tests near existing image placement tests:

```rust
#[test]
fn css_image_layer_normalizes_placement_repeat_and_attachment_together() {
    let resource = ResolvedImageResource::try_new(ImageId::new(90), Size::new(25.0, 10.0))
        .unwrap();
    let layer = StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
        .unwrap()
        .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
        .with_size(BackgroundSize::explicit(
            SizeComponent::try_length(50.0).unwrap(),
            SizeComponent::auto(),
        ))
        .with_repeat(BackgroundRepeat::repeat_x())
        .with_attachment(BackgroundAttachment::Fixed)
        .with_coordinate_space(
            CoordinateSpaceTag::viewport(Transform::translation(2.0, 3.0).unwrap()).unwrap(),
        );

    let placement = ImagePlacementInput::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        resource.intrinsic_size(),
        layer.position(),
        layer.size(),
    )
    .unwrap()
    .resolve()
    .unwrap();
    let repeat = ImageRepeatPlan::try_new(layer.repeat(), Capabilities::VELLO_0_9)
        .unwrap()
        .resolve(placement)
        .unwrap();
    let attachment = ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space())
        .unwrap();

    assert_eq!(placement.tile_rect(), Rect::new(70.0, 0.0, 50.0, 20.0));
    assert_eq!(repeat.clip_rect(), Rect::new(0.0, 0.0, 120.0, 80.0));
    assert_eq!(
        repeat.tile_rects(),
        &[
            Rect::new(-30.0, 0.0, 50.0, 20.0),
            Rect::new(20.0, 0.0, 50.0, 20.0),
            Rect::new(70.0, 0.0, 50.0, 20.0),
        ]
    );
    assert_eq!(
        attachment.coordinate_space().map(CoordinateSpaceTag::kind),
        Some(CoordinateSpaceKind::Viewport)
    );
}
```

Run:

```sh
cargo test -p surgeist-render css_image_layer_normalizes_placement_repeat_and_attachment_together
```

Expected: pass if Tasks 1-5 are complete.

- [ ] **Step 2: Run existing image fit tests**

Run:

```sh
cargo test -p surgeist-render image_fit
```

Expected: pass. If this command finds no tests because names differ, run:

```sh
cargo test -p surgeist-render image
```

Expected: existing image tests pass, including image transform/fit cases.

- [ ] **Step 3: Run crate checks**

Run:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

Expected: all pass.

## Coordinator Execution Requirements

For this plan:

1. Assign each task above to one implementation worker or one tightly coupled worker group at a time.
2. Tell workers:
   - They are not alone in the codebase.
   - They must not revert unrelated changes.
   - They must not commit.
   - They must report tests run and `git status --short --branch`.
3. Use a separate reviewer after each worker result.
4. Reconcile reviewer findings before moving to the next task.
5. After a scoped task is clean, run its focused checks and commit that logical point.
6. After all tasks are complete, assign a final clean-context holistic reviewer against:
   - this plan
   - `AGENTS.md`
   - `guidance/surgeist-rust-modeling-guide.md`
   - `plans/2026-07-08-render-css-support-matrix.md`
   - `plans/2026-07-09-render-css-implementation-sequence.md`
   - full git diff
   - tests and crate boundary
7. Run final checks:

```sh
cargo test -p surgeist-render
cargo clippy -p surgeist-render --all-targets -- -D warnings
cargo fmt --check
```

8. Commit the final implementation only after the final holistic review and final checks are clean.

## Completion Checklist

- [ ] Image sampling capabilities are public and tested.
- [ ] Unsupported round/space repeat, filtered image paint execution, orientation conversion, and color-profile conversion report typed diagnostics.
- [ ] Resolved image resources carry optional density metadata and root-resolved orientation/color-profile policy.
- [ ] Unresolved image sources report `UnresolvedResourceKind::Image` diagnostics.
- [ ] Image placement normalizes intrinsic size, cover, contain, explicit length, explicit percent, auto axis, position ratio/length inputs, and edge-offset positions.
- [ ] Repeat normalization supports no-repeat, repeat-x, repeat-y, repeat-both, visible tile rectangle generation, and paint-rect clipping.
- [ ] Fixed attachment requires viewport coordinate tagging; scroll/local can use root-resolved or explicit tags.
- [ ] Filtered image paint is modeled as resolved image plus non-empty filter list and rejected at execution boundary.
- [ ] Existing direct image fit behavior remains covered.
- [ ] No sibling crates are edited.
- [ ] No new dependencies are added.
- [ ] Required checks pass.
