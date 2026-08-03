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
pub struct BackgroundLayer {
    image: StyleImageLayer,
}

impl BackgroundLayer {
    #[must_use]
    pub const fn new(image: StyleImageLayer) -> Self {
        Self { image }
    }

    #[must_use]
    pub const fn image(&self) -> &StyleImageLayer {
        &self.image
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundStack {
    color: Option<Color>,
    layers: Vec<BackgroundLayer>,
}

impl BackgroundStack {
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

    #[must_use]
    pub const fn color(&self) -> Option<Color> {
        self.color
    }

    #[must_use]
    pub fn layers(&self) -> &[BackgroundLayer] {
        &self.layers
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackgroundBlendList {
    modes: Vec<BackgroundBlendMode>,
}

impl BackgroundBlendList {
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

    #[must_use]
    pub fn modes(&self) -> &[BackgroundBlendMode] {
        &self.modes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundBlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Plus,
}

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
#[expect(
    clippy::large_enum_variant,
    reason = "background commands keep their planned public shape for direct matching"
)]
pub enum NormalizedBackgroundCommandKind {
    ColorFill { rect: Rect, color: Color },
    Layer { layer: NormalizedBackgroundLayer },
}

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
