use super::super::{
    Capabilities, Color, Error, Image, Paint, PrimitiveFamily, PrimitiveOperation, Rect, Result,
    UnsupportedPrimitive, validation::validate_paint,
};
use super::image::{
    BackgroundAreas, BackgroundClipGeometry, ImageAttachmentPlan, ImagePlacementInput,
    ImageRepeatPlan, ResolvedImagePlacement, ResolvedImageRepeat, ResolvedImageResource,
    StyleImageLayer, StyleImageSourceKind,
};

#[derive(Clone, Debug, PartialEq)]
/// One authored background layer backed by an image-style layer.
pub struct BackgroundLayer {
    image: StyleImageLayer,
}

impl BackgroundLayer {
    /// Creates a background layer from its authored image-layer values.
    #[must_use]
    pub const fn new(image: StyleImageLayer) -> Self {
        Self { image }
    }

    /// Returns the image-layer description.
    #[must_use]
    pub const fn image(&self) -> &StyleImageLayer {
        &self.image
    }
}

#[derive(Clone, Debug, PartialEq)]
/// An authored background color and ordered image layers.
///
/// At least a color or one layer is present. During normalization, the color is
/// emitted first and layers are emitted in reverse of this stored order.
pub struct BackgroundStack {
    color: Option<Color>,
    layers: Vec<BackgroundLayer>,
}

impl BackgroundStack {
    /// Creates a non-empty background stack.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when both `color` is absent
    /// and `layers` is empty.
    pub fn try_new(color: Option<Color>, layers: Vec<BackgroundLayer>) -> Result<Self> {
        if color.is_none() && layers.is_empty() {
            return Err(Error::invalid_value(
                "background stack",
                "none + []",
                "must include a color or at least one layer",
            ));
        }
        Ok(Self { color, layers })
    }

    /// Returns the optional concrete background color.
    #[must_use]
    pub const fn color(&self) -> Option<Color> {
        self.color
    }

    /// Returns the authored layers in their stored order.
    #[must_use]
    pub fn layers(&self) -> &[BackgroundLayer] {
        &self.layers
    }
}

#[derive(Clone, Debug, PartialEq)]
/// A non-empty authored list of blend modes for background layers.
///
/// The current rendering contract accepts only [`BackgroundBlendMode::Normal`]
/// entries; other modeled choices produce an unsupported-primitive diagnostic.
pub struct BackgroundBlendList {
    modes: Vec<BackgroundBlendMode>,
}

impl BackgroundBlendList {
    /// Validates a background blend-mode list.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] for an empty list and an
    /// unsupported-primitive error when any mode is not `Normal`.
    pub fn try_new(modes: Vec<BackgroundBlendMode>) -> Result<Self> {
        if modes.is_empty() {
            return Err(Error::invalid_value(
                "background blend list",
                "[]",
                "must contain at least one mode",
            ));
        }
        if modes
            .iter()
            .any(|mode| *mode != BackgroundBlendMode::Normal)
        {
            return Err(Error::unsupported_render_primitive(
                UnsupportedPrimitive::new(
                    PrimitiveFamily::Compositing,
                    PrimitiveOperation::BackgroundBlendMode,
                ),
            ));
        }
        Ok(Self { modes })
    }

    /// Returns the validated modes in authored order.
    #[must_use]
    pub fn modes(&self) -> &[BackgroundBlendMode] {
        &self.modes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// An authored blend choice between background layers.
pub enum BackgroundBlendMode {
    /// Standard source-over blending.
    Normal,
    /// Multiply blending; currently reported as unsupported by list validation.
    Multiply,
    /// Screen blending; currently reported as unsupported by list validation.
    Screen,
    /// Overlay blending; currently reported as unsupported by list validation.
    Overlay,
    /// Darken blending; currently reported as unsupported by list validation.
    Darken,
    /// Lighten blending; currently reported as unsupported by list validation.
    Lighten,
    /// Additive blending; currently reported as unsupported by list validation.
    Plus,
}

#[derive(Clone, Debug, PartialEq)]
/// Context supplied to normalize a background stack into render-facing commands.
///
/// The background areas provide the logical border, padding, and content boxes.
/// Per-layer clip overrides may replace the clip derived from each layer.
pub struct BackgroundNormalizationInput {
    stack: BackgroundStack,
    areas: BackgroundAreas,
    layer_clip_overrides: Vec<Option<BackgroundClipGeometry>>,
}

#[derive(Clone, Debug, PartialEq)]
/// An intrinsically validated, ordered sequence of background render commands.
pub struct NormalizedBackgroundStack {
    commands: Vec<NormalizedBackgroundCommand>,
}

#[derive(Clone, Debug, PartialEq)]
/// One normalized background command paired with its clip geometry.
pub struct NormalizedBackgroundCommand {
    clip: BackgroundClipGeometry,
    kind: NormalizedBackgroundCommandKind,
}

#[derive(Clone, Debug, PartialEq)]
/// A normalized background layer with resolved placement, repetition, and attachment.
pub struct NormalizedBackgroundLayer {
    source: NormalizedBackgroundLayerSource,
    placement: ResolvedImagePlacement,
    repeat: ResolvedImageRepeat,
    attachment: ImageAttachmentPlan,
}

#[derive(Clone, Debug, PartialEq)]
/// The render-facing source retained by a normalized background layer.
pub enum NormalizedBackgroundLayerSource {
    /// A validated paint source.
    Paint(Paint),
    /// A validated image carrying pixel content.
    Image(Image),
    /// An image resource already resolved by the surrounding context.
    ResolvedImage(ResolvedImageResource),
}

#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "background commands keep their planned public shape for direct matching"
)]
/// The operation represented by a normalized background command.
pub enum NormalizedBackgroundCommandKind {
    /// Fills a logical rectangle with a concrete color.
    ColorFill {
        /// The logical rectangle to fill.
        rect: Rect,
        /// The concrete fill color.
        color: Color,
    },
    /// Draws one normalized image-style layer.
    Layer {
        /// The normalized layer payload.
        layer: NormalizedBackgroundLayer,
    },
}

impl BackgroundNormalizationInput {
    /// Creates normalization input with no per-layer clip overrides.
    pub fn try_new(stack: BackgroundStack, areas: BackgroundAreas) -> Result<Self> {
        let layer_clip_overrides = vec![None; stack.layers().len()];
        Ok(Self {
            stack,
            areas,
            layer_clip_overrides,
        })
    }

    /// Returns the authored background stack.
    #[must_use]
    pub fn stack(&self) -> &BackgroundStack {
        &self.stack
    }

    /// Returns the logical background areas used during normalization.
    #[must_use]
    pub const fn areas(&self) -> BackgroundAreas {
        self.areas
    }

    /// Returns one optional clip override for each background layer.
    #[must_use]
    pub fn layer_clip_overrides(&self) -> &[Option<BackgroundClipGeometry>] {
        &self.layer_clip_overrides
    }

    /// Replaces the per-layer clip overrides.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] unless the override count
    /// equals the background layer count. `None` keeps that layer's authored
    /// background-box clip.
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

    /// Normalizes the stack using the supplied semantic capability contract.
    ///
    /// The optional color command is emitted first, followed by image layers in
    /// reverse stored order. Symbolic image references return an unresolved-resource
    /// diagnostic; invalid source, placement, repeat, attachment, or clip data and
    /// currently unsupported choices return their corresponding typed errors.
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
            StyleImageSourceKind::Image(image) => (
                NormalizedBackgroundLayerSource::Image(image.clone()),
                image.size(),
            ),
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
        let attachment =
            ImageAttachmentPlan::try_new(layer.attachment(), layer.coordinate_space())?;
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

impl NormalizedBackgroundStack {
    /// Returns the normalized commands in rendering order.
    #[must_use]
    pub fn commands(&self) -> &[NormalizedBackgroundCommand] {
        &self.commands
    }
}

impl NormalizedBackgroundCommand {
    /// Returns the geometry that clips this command.
    #[must_use]
    pub fn clip(&self) -> &BackgroundClipGeometry {
        &self.clip
    }

    /// Returns the normalized operation payload.
    #[must_use]
    pub fn kind(&self) -> &NormalizedBackgroundCommandKind {
        &self.kind
    }
}

impl NormalizedBackgroundLayer {
    /// Returns the validated or context-resolved layer source.
    #[must_use]
    pub fn source(&self) -> &NormalizedBackgroundLayerSource {
        &self.source
    }

    /// Returns the resolved logical paint and first-tile rectangles.
    #[must_use]
    pub const fn placement(&self) -> ResolvedImagePlacement {
        self.placement
    }

    /// Returns the resolved clip rectangle and ordered tile rectangles.
    #[must_use]
    pub fn repeat(&self) -> &ResolvedImageRepeat {
        &self.repeat
    }

    /// Returns the validated attachment and coordinate-space plan.
    #[must_use]
    pub const fn attachment(&self) -> ImageAttachmentPlan {
        self.attachment
    }
}
