# Background Layer Stack Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the render-owned background layer stack normalization required by Sequence 7 of the CSS/style support matrix.

**Architecture:** Root supplies final border/padding/content geometry and any non-box clip geometry; render converts `BackgroundStack` plus `StyleImageLayer` inputs into ordered, validated background render intents. The normalized commands stay backend-neutral in this phase and reuse Sequence 6 placement, repeat, attachment, unresolved-image diagnostics, and paint/source models.

**Tech Stack:** Rust 2024, surgeist-render style value models, existing `Error` diagnostics, Sequence 6 image sampling primitives, crate-local unit tests.

---

## Source Guidance

Required references:

- `AGENTS.md`
- `guidance/surgeist-rust-modeling-guide.md`
- `plans/2026-07-08-render-css-support-matrix.md`
- `plans/2026-07-09-render-css-implementation-sequence.md`
- `plans/2026-07-09-image-resource-sampling-normalization-implementation.md`

Sequence 7 covers:

- Background layer stack
- Background origin
- Background clip
- Multi-layer image stack
- Background color behind images

Render boundary decisions for this phase:

- Render does not infer layout boxes. Root supplies physical border, padding, and content rectangles through a render-owned value type.
- Render does not parse CSS list matching. Root supplies an already-built `BackgroundStack`; this phase adds explicit validation for any optional per-layer clip override list.
- `BackgroundStack` layer order remains CSS-authored order: index 0 is the topmost background layer. `NormalizedBackgroundStack::commands()` is render order: background color first, then layers from back to front.
- Box clips use root-supplied `BackgroundAreas`. Shape/path clips are accepted as render-owned `Shape` values through an explicit per-layer override.
- Background color paints the root-supplied border box in this phase. Any future `background-clip` variations for color-only backgrounds should be added as an explicit style/root input rather than inferred here.
- Paint-backed image layers, such as gradient-like `Paint` sources, retain per-layer origin, clip, size, position, repeat, attachment, and coordinate-space semantics. Because render-owned `Paint` does not carry intrinsic image metadata, paint-backed layers use the selected origin box size as their intrinsic sampling size.
- Text clips, compositor masks, blend modes, and backend/offscreen lowering remain later sequence work.
- Backwards compatibility shims are not required.

## File Responsibilities

- `src/style.rs`: add background area geometry, clip geometry, normalization input, normalized stack/command/layer types, and normalization helpers.
- `src/lib.rs`: export the new public background normalization types.
- `src/tests.rs`: add tests for area selection, command ordering, list matching, origin/clip geometry, image layer sampling, shape/path clip overrides, unresolved resources, and background color behind layers.

Do not edit sibling crates. Do not add dependencies. Do not change backend encoding unless a test proves an existing regression.

## Task 1: Background Areas And Clip Geometry

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for root-supplied background areas and clip geometry**

Add tests near the existing background stack tests in `src/tests.rs`:

```rust
#[test]
fn background_areas_select_origin_and_clip_boxes() {
    let areas = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 120.0, 80.0),
        Rect::new(10.0, 8.0, 100.0, 60.0),
        Rect::new(20.0, 18.0, 80.0, 40.0),
    )
    .unwrap();

    assert_eq!(areas.rect_for(BackgroundBox::Border), Rect::new(0.0, 0.0, 120.0, 80.0));
    assert_eq!(areas.rect_for(BackgroundBox::Padding), Rect::new(10.0, 8.0, 100.0, 60.0));
    assert_eq!(areas.rect_for(BackgroundBox::Content), Rect::new(20.0, 18.0, 80.0, 40.0));
}

#[test]
fn background_areas_reject_invalid_rects() {
    let error = BackgroundAreas::try_new(
        Rect::new(0.0, 0.0, 100.0, 100.0),
        Rect::new(0.0, 0.0, 0.0, 50.0),
        Rect::new(0.0, 0.0, 10.0, 10.0),
    )
    .expect_err("background areas require positive boxes");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background padding box")
    );
}

#[test]
fn background_clip_geometry_preserves_box_or_shape_inputs() {
    let rect_clip = BackgroundClipGeometry::try_rect(Rect::new(0.0, 0.0, 12.0, 8.0)).unwrap();
    assert_eq!(rect_clip.kind(), &BackgroundClipGeometryKind::Rect(Rect::new(0.0, 0.0, 12.0, 8.0)));

    let shape = Shape::rect(Rect::new(1.0, 2.0, 3.0, 4.0));
    let shape_clip = BackgroundClipGeometry::try_shape(shape.clone()).unwrap();
    assert_eq!(shape_clip.shape(), Some(&shape));
}
```

Run:

```sh
cargo test -p surgeist-render background_areas
cargo test -p surgeist-render background_clip_geometry
```

Expected: fail to compile because the new types do not exist.

- [ ] **Step 2: Add `BackgroundAreas`**

In `src/style.rs`, add:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundAreas {
    border_box: Rect,
    padding_box: Rect,
    content_box: Rect,
}

impl BackgroundAreas {
    pub fn try_new(border_box: Rect, padding_box: Rect, content_box: Rect) -> Result<Self> {
        validate_background_rect(border_box, "background border box")?;
        validate_background_rect(padding_box, "background padding box")?;
        validate_background_rect(content_box, "background content box")?;
        Ok(Self {
            border_box,
            padding_box,
            content_box,
        })
    }

    #[must_use]
    pub const fn border_box(self) -> Rect {
        self.border_box
    }

    #[must_use]
    pub const fn padding_box(self) -> Rect {
        self.padding_box
    }

    #[must_use]
    pub const fn content_box(self) -> Rect {
        self.content_box
    }

    #[must_use]
    pub const fn rect_for(self, box_kind: BackgroundBox) -> Rect {
        match box_kind {
            BackgroundBox::Border => self.border_box,
            BackgroundBox::Padding => self.padding_box,
            BackgroundBox::Content => self.content_box,
        }
    }
}
```

Add helper near other style validation helpers:

```rust
fn validate_background_rect(rect: Rect, field: &str) -> Result<()> {
    validate_finite_f64(rect.x(), &format!("{field} x"))?;
    validate_finite_f64(rect.y(), &format!("{field} y"))?;
    if !rect.width().is_finite()
        || !rect.height().is_finite()
        || rect.width() <= 0.0
        || rect.height() <= 0.0
    {
        return Err(Error::invalid_value(
            field,
            format!("{rect:?}"),
            "must have finite positive width and height",
        ));
    }
    Ok(())
}
```

- [ ] **Step 3: Add background clip geometry**

Add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundClipGeometry {
    kind: BackgroundClipGeometryKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BackgroundClipGeometryKind {
    Rect(Rect),
    Shape(Shape),
}

impl BackgroundClipGeometry {
    pub fn try_rect(rect: Rect) -> Result<Self> {
        validate_background_rect(rect, "background clip rect")?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Rect(rect),
        })
    }

    pub fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: BackgroundClipGeometryKind::Shape(shape),
        })
    }

    #[must_use]
    pub fn kind(&self) -> &BackgroundClipGeometryKind {
        &self.kind
    }

    #[must_use]
    pub const fn rect(&self) -> Option<Rect> {
        match self.kind {
            BackgroundClipGeometryKind::Rect(rect) => Some(rect),
            BackgroundClipGeometryKind::Shape(_) => None,
        }
    }

    #[must_use]
    pub fn shape(&self) -> Option<&Shape> {
        match &self.kind {
            BackgroundClipGeometryKind::Rect(_) => None,
            BackgroundClipGeometryKind::Shape(shape) => Some(shape),
        }
    }
}
```

- [ ] **Step 4: Export area and clip types**

In `src/lib.rs`, export:

```rust
BackgroundAreas, BackgroundClipGeometry, BackgroundClipGeometryKind,
```

- [ ] **Step 5: Run focused checks**

Run:

```sh
cargo test -p surgeist-render background_areas
cargo test -p surgeist-render background_clip_geometry
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 2: Normalized Background Command Ordering

**Files:**

- Modify: `src/style.rs`
- Modify: `src/lib.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for render-order normalization**

Add tests near the background stack tests:

```rust
#[test]
fn background_stack_normalization_paints_color_behind_layers() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top, back]).unwrap();
    let input = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap();

    let normalized = input.normalize(Capabilities::VELLO_0_9).unwrap();
    assert_eq!(normalized.commands().len(), 3);
    let NormalizedBackgroundCommandKind::ColorFill { color, .. } =
        normalized.commands()[0].kind()
    else {
        panic!("expected background color command");
    };
    assert_eq!(*color, Color::BLACK);
    assert!(matches!(
        normalized.commands()[1].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
    assert!(matches!(
        normalized.commands()[2].kind(),
        NormalizedBackgroundCommandKind::Layer { .. }
    ));
}

#[test]
fn background_stack_normalization_preserves_top_layer_as_last_render_command() {
    let top = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Content),
    );
    let back = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap()
            .with_clip(BackgroundBox::Padding),
    );
    let stack = BackgroundStack::try_new(None, vec![top, back]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(4.0, 4.0, 92.0, 52.0),
            Rect::new(8.0, 8.0, 84.0, 44.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let last = normalized.commands().last().unwrap();
    assert_eq!(last.clip().rect(), Some(Rect::new(8.0, 8.0, 84.0, 44.0)));
}

#[test]
fn background_stack_normalization_preserves_paint_layer_sampling_semantics() {
    let paint_layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 1.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_percent(0.5).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_y())
            .with_attachment(BackgroundAttachment::Local)
            .with_coordinate_space(CoordinateSpaceTag::local()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![paint_layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 60.0),
            Rect::new(20.0, 20.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind()
    else {
        panic!("expected normalized paint-backed layer");
    };
    assert!(matches!(layer.source(), NormalizedBackgroundLayerSource::Paint(_)));
    assert_eq!(layer.placement().paint_rect(), Rect::new(20.0, 20.0, 80.0, 40.0));
    assert_eq!(layer.placement().tile_rect(), Rect::new(60.0, 40.0, 40.0, 20.0));
    assert_eq!(layer.repeat().clip_rect(), Rect::new(20.0, 20.0, 80.0, 40.0));
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Local);
}
```

Run:

```sh
cargo test -p surgeist-render background_stack_normalization
```

Expected: fail to compile because the normalization types do not exist.

- [ ] **Step 2: Add normalization input and output models**

Add:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundNormalizationInput {
    stack: BackgroundStack,
    areas: BackgroundAreas,
    layer_clip_overrides: Vec<Option<BackgroundClipGeometry>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundStack {
    commands: Vec<NormalizedBackgroundCommand>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundCommand {
    clip: BackgroundClipGeometry,
    kind: NormalizedBackgroundCommandKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedBackgroundLayer {
    source: NormalizedBackgroundLayerSource,
    placement: ResolvedImagePlacement,
    repeat: ResolvedImageRepeat,
    attachment: ImageAttachmentPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBackgroundLayerSource {
    Paint(Paint),
    Image(Image),
    ResolvedImage(ResolvedImageResource),
}

#[derive(Clone, Debug, PartialEq)]
pub enum NormalizedBackgroundCommandKind {
    ColorFill { rect: Rect, color: Color },
    Layer { layer: NormalizedBackgroundLayer },
}
```

Add:

```rust
impl BackgroundNormalizationInput {
    pub fn try_new(stack: BackgroundStack, areas: BackgroundAreas) -> Result<Self> {
        let layer_clip_overrides = vec![None; stack.layers().len()];
        Ok(Self {
            stack,
            areas,
            layer_clip_overrides,
        })
    }

    #[must_use]
    pub fn stack(&self) -> &BackgroundStack {
        &self.stack
    }

    #[must_use]
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    #[must_use]
    pub fn layer_clip_overrides(&self) -> &[Option<BackgroundClipGeometry>] {
        &self.layer_clip_overrides
    }
}

impl NormalizedBackgroundStack {
    #[must_use]
    pub fn commands(&self) -> &[NormalizedBackgroundCommand] {
        &self.commands
    }
}

impl NormalizedBackgroundCommand {
    #[must_use]
    pub fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    #[must_use]
    pub fn kind(&self) -> &NormalizedBackgroundCommandKind {
        &self.kind
    }
}

impl NormalizedBackgroundLayer {
    #[must_use]
    pub fn source(&self) -> &NormalizedBackgroundLayerSource {
        &self.source
    }

    #[must_use]
    pub const fn placement(&self) -> ResolvedImagePlacement {
        self.placement
    }

    #[must_use]
    pub fn repeat(&self) -> &ResolvedImageRepeat {
        &self.repeat
    }

    #[must_use]
    pub const fn attachment(&self) -> ImageAttachmentPlan {
        self.attachment
    }
}
```

- [ ] **Step 3: Implement color and layer normalization**

Add:

```rust
impl BackgroundNormalizationInput {
    pub fn normalize(&self, capabilities: Capabilities) -> Result<NormalizedBackgroundStack> {
        let mut commands = Vec::new();
        if let Some(color) = self.stack.color() {
            let rect = self.areas.border_box();
            commands.push(NormalizedBackgroundCommand {
                clip: BackgroundClipGeometry::try_rect(rect)?,
                kind: NormalizedBackgroundCommandKind::ColorFill { rect, color },
            });
        }

        for (layer_index, layer) in self.stack.layers().iter().enumerate().rev() {
            commands.push(self.normalize_layer(layer_index, layer.image(), capabilities)?);
        }

        Ok(NormalizedBackgroundStack { commands })
    }

    fn normalize_layer(
        &self,
        layer_index: usize,
        layer: &StyleImageLayer,
        capabilities: Capabilities,
    ) -> Result<NormalizedBackgroundCommand> {
        let clip = self.layer_clip_geometry(layer_index, layer)?;
        let origin_rect = self.areas.rect_for(layer.origin());
        let (source, intrinsic_size) = match layer.source().kind() {
            StyleImageSourceKind::Paint(paint) => {
                validate_paint(paint)?;
                (
                    NormalizedBackgroundLayerSource::Paint(paint.clone()),
                    origin_rect.size(),
                )
            }
            StyleImageSourceKind::Image(image) => {
                (NormalizedBackgroundLayerSource::Image(image.clone()), image.size())
            }
            StyleImageSourceKind::Resolved(resource) => (
                NormalizedBackgroundLayerSource::ResolvedImage(resource.clone()),
                resource.intrinsic_size(),
            ),
            StyleImageSourceKind::Unresolved(_) => {
                layer.source().require_resolved()?;
                unreachable!("unresolved image sources return an error")
            }
        };
        let placement = ImagePlacementInput::try_new(
            origin_rect,
            intrinsic_size,
            layer.position(),
            layer.size(),
        )?
        .resolve()?;
        let repeat = ImageRepeatPlan::try_new(layer.repeat(), capabilities)?.resolve(placement)?;
        let attachment = ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space())?;
        Ok(NormalizedBackgroundCommand {
            clip,
            kind: NormalizedBackgroundCommandKind::Layer {
                layer: NormalizedBackgroundLayer {
                    source,
                    placement,
                    repeat,
                    attachment,
                },
            },
        })
    }

    fn layer_clip_geometry(
        &self,
        layer_index: usize,
        layer: &StyleImageLayer,
    ) -> Result<BackgroundClipGeometry> {
        if let Some(override_clip) = &self.layer_clip_overrides[layer_index] {
            return Ok(override_clip.clone());
        }
        BackgroundClipGeometry::try_rect(self.areas.rect_for(layer.clip()))
    }
}
```

- [ ] **Step 4: Export normalization types**

In `src/lib.rs`, export:

```rust
BackgroundNormalizationInput, NormalizedBackgroundCommand,
NormalizedBackgroundCommandKind, NormalizedBackgroundLayer, NormalizedBackgroundLayerSource,
NormalizedBackgroundStack,
```

- [ ] **Step 5: Run focused checks**

Run:

```sh
cargo test -p surgeist-render background_stack_normalization
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 3: Image Source Coverage And Diagnostics

**Files:**

- Test: `src/tests.rs`
- No production edits unless the Task 2 shared normalizer has a bug

- [ ] **Step 1: Add failing tests for image layers using origin, clip, placement, repeat, and attachment**

Add:

```rust
#[test]
fn background_stack_normalizes_image_layers_with_origin_clip_repeat_and_attachment() {
    let image = Image::from_rgba(Size::new(20.0, 10.0), vec![255; 20 * 10 * 4]).unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image.clone()).unwrap())
            .unwrap()
            .with_origin(BackgroundBox::Content)
            .with_clip(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(1.0, 0.0).unwrap())
            .with_size(BackgroundSize::explicit(
                SizeComponent::try_length(40.0).unwrap(),
                SizeComponent::auto(),
            ))
            .with_repeat(BackgroundRepeat::repeat_x())
            .with_attachment(BackgroundAttachment::Fixed)
            .with_coordinate_space(
                CoordinateSpaceTag::viewport(Transform::translation(1.0, 2.0).unwrap()).unwrap(),
            ),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(5.0, 5.0, 90.0, 50.0),
            Rect::new(10.0, 10.0, 80.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let command = normalized.commands().first().unwrap();
    assert_eq!(command.clip().rect(), Some(Rect::new(5.0, 5.0, 90.0, 50.0)));
    let NormalizedBackgroundCommandKind::Layer { layer } = command.kind()
    else {
        panic!("expected normalized image layer");
    };
    assert!(matches!(layer.source(), NormalizedBackgroundLayerSource::Image(_)));
    assert_eq!(layer.placement().paint_rect(), Rect::new(10.0, 10.0, 80.0, 40.0));
    assert_eq!(layer.placement().tile_rect(), Rect::new(50.0, 10.0, 40.0, 20.0));
    assert_eq!(layer.repeat().clip_rect(), Rect::new(10.0, 10.0, 80.0, 40.0));
    assert_eq!(layer.attachment().attachment(), BackgroundAttachment::Fixed);
}

#[test]
fn background_stack_normalizes_resolved_image_layers_with_intrinsic_size() {
    let resource = ResolvedImageResource::try_new(ImageId::new(400), Size::new(24.0, 12.0))
        .unwrap();
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::resolved(resource.clone()))
            .unwrap()
            .with_origin(BackgroundBox::Padding)
            .with_position(BackgroundPosition::percent(0.5, 0.5).unwrap())
            .with_size(BackgroundSize::contain())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let normalized = BackgroundNormalizationInput::try_new(
        BackgroundStack::try_new(None, vec![layer]).unwrap(),
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 120.0, 80.0),
            Rect::new(10.0, 10.0, 100.0, 50.0),
            Rect::new(20.0, 20.0, 80.0, 30.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    let NormalizedBackgroundCommandKind::Layer { layer } = normalized.commands()[0].kind()
    else {
        panic!("expected normalized layer");
    };
    assert!(matches!(
        layer.source(),
        NormalizedBackgroundLayerSource::ResolvedImage(_)
    ));
    assert_eq!(layer.placement().tile_rect(), Rect::new(10.0, 10.0, 100.0, 50.0));
}

#[test]
fn background_stack_reports_unresolved_image_layers() {
    let source = StyleImageSource::unresolved(StyleResourceRef::try_new("hero.png").unwrap());
    let layer = BackgroundLayer::new(StyleImageLayer::try_new(source).unwrap());
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
            Rect::new(0.0, 0.0, 100.0, 60.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .expect_err("unresolved image layer should fail normalization");

    assert_eq!(error.code, ErrorCode::UnresolvedResource);
}
```

Run:

```sh
cargo test -p surgeist-render background_stack_normalizes_image_layers
cargo test -p surgeist-render background_stack_normalizes_resolved_image_layers
cargo test -p surgeist-render background_stack_reports_unresolved_image_layers
```

Expected: pass if Task 2 is complete. If any fail, fix the shared Task 2 normalizer rather than adding a parallel image-only path.

- [ ] **Step 2: Run focused checks**

Run:

```sh
cargo test -p surgeist-render background_stack_normalizes_image_layers
cargo test -p surgeist-render background_stack_normalizes_resolved_image_layers
cargo test -p surgeist-render background_stack_reports_unresolved_image_layers
cargo test -p surgeist-render background_stack_normalization
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 4: Shape Clip Overrides And List Matching

**Files:**

- Modify: `src/style.rs`
- Test: `src/tests.rs`

- [ ] **Step 1: Add failing tests for layer clip override list matching and shape/path clips**

Add:

```rust
#[test]
fn background_normalization_rejects_clip_override_length_mismatch() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let error = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(Vec::new())
    .expect_err("clip override list must match background layer count");

    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(
        error.invalid_value_diagnostic().map(InvalidValue::field),
        Some("background layer clip overrides")
    );
}

#[test]
fn background_normalization_accepts_shape_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let shape = Shape::rect(Rect::new(1.0, 1.0, 8.0, 8.0));
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(BackgroundClipGeometry::try_shape(shape.clone()).unwrap())])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}

#[test]
fn background_normalization_accepts_path_clip_overrides() {
    let layer = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::BLACK)).unwrap())
            .unwrap(),
    );
    let mut path = Path::new();
    path.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0))
        .line_to(Point::new(10.0, 10.0))
        .close();
    let shape = Shape::path(path);
    let stack = BackgroundStack::try_new(None, vec![layer]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )
        .unwrap(),
    )
    .unwrap()
    .with_layer_clip_overrides(vec![Some(BackgroundClipGeometry::try_shape(shape.clone()).unwrap())])
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert_eq!(normalized.commands()[0].clip().shape(), Some(&shape));
}
```

Run:

```sh
cargo test -p surgeist-render background_normalization
```

Expected: fail to compile until `with_layer_clip_overrides` exists.

- [ ] **Step 2: Add clip override validation**

Add to `BackgroundNormalizationInput`:

```rust
pub fn with_layer_clip_overrides(
    mut self,
    layer_clip_overrides: Vec<Option<BackgroundClipGeometry>>,
) -> Result<Self> {
    if layer_clip_overrides.len() != self.stack.layers().len() {
        return Err(Error::invalid_value(
            "background layer clip overrides",
            layer_clip_overrides.len(),
            "must match background layer count",
        ));
    }
    self.layer_clip_overrides = layer_clip_overrides;
    Ok(self)
}
```

- [ ] **Step 3: Run focused checks**

Run:

```sh
cargo test -p surgeist-render background_normalization
cargo fmt --check
cargo clippy -p surgeist-render --all-targets -- -D warnings
```

Expected: all pass.

## Task 5: Integration Guardrails

**Files:**

- Modify: `src/tests.rs`
- No production edits unless tests reveal a bug in Tasks 1-4

- [ ] **Step 1: Add an end-to-end normalization test for mixed color, paint, and image layers**

Add:

```rust
#[test]
fn background_normalization_mixes_color_paint_and_image_layers_in_render_order() {
    let image = Image::from_rgba(Size::new(10.0, 10.0), vec![255; 10 * 10 * 4]).unwrap();
    let top_image = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::image(image).unwrap())
            .unwrap()
            .with_size(BackgroundSize::auto())
            .with_repeat(BackgroundRepeat::no_repeat()),
    );
    let back_paint = BackgroundLayer::new(
        StyleImageLayer::try_new(StyleImageSource::paint(Paint::from(Color::TRANSPARENT)).unwrap())
            .unwrap(),
    );
    let stack = BackgroundStack::try_new(Some(Color::BLACK), vec![top_image, back_paint]).unwrap();
    let normalized = BackgroundNormalizationInput::try_new(
        stack,
        BackgroundAreas::try_new(
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
            Rect::new(0.0, 0.0, 40.0, 40.0),
        )
        .unwrap(),
    )
    .unwrap()
    .normalize(Capabilities::VELLO_0_9)
    .unwrap();

    assert!(matches!(
        normalized.commands()[0].kind(),
        NormalizedBackgroundCommandKind::ColorFill { .. }
    ));
    let NormalizedBackgroundCommandKind::Layer { layer: back_layer } =
        normalized.commands()[1].kind()
    else {
        panic!("expected back layer command");
    };
    assert!(matches!(back_layer.source(), NormalizedBackgroundLayerSource::Paint(_)));

    let NormalizedBackgroundCommandKind::Layer { layer: top_layer } =
        normalized.commands()[2].kind()
    else {
        panic!("expected top layer command");
    };
    assert!(matches!(top_layer.source(), NormalizedBackgroundLayerSource::Image(_)));
}
```

Run:

```sh
cargo test -p surgeist-render background_normalization_mixes_color_paint_and_image_layers_in_render_order
```

Expected: pass if Tasks 1-4 are complete.

- [ ] **Step 2: Run existing background and image sampling tests**

Run:

```sh
cargo test -p surgeist-render background
cargo test -p surgeist-render image_repeat
cargo test -p surgeist-render image_placement
```

Expected: all pass.

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

- [ ] Root-supplied border/padding/content background areas are modeled and validated.
- [ ] Background origin selects the correct root-supplied area for image placement.
- [ ] Background clip selects box geometry by default and can accept explicit shape/path clip overrides.
- [ ] Background color normalizes behind all background layers.
- [ ] Multi-layer stacks normalize from CSS-authored order into back-to-front render order.
- [ ] Paint layers and image layers can coexist in one normalized stack.
- [ ] Image layers reuse Sequence 6 placement, repeat, attachment, and unresolved-resource diagnostics.
- [ ] Clip override list length mismatch reports a typed invalid-value diagnostic.
- [ ] No backend behavior, sibling crates, root pointers, or dependencies are changed.
- [ ] Required checks pass.
