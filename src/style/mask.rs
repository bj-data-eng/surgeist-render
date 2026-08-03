use super::super::{
    Capabilities, CoordinateSpaceTag, Error, PrimitiveFamily, PrimitiveOperation, Result, Shape,
    UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive, validation::validate_shape,
};
use super::image::{StyleImageLayer, StyleResourceRef};

#[derive(Clone, Debug, PartialEq)]
pub struct MaskInput {
    source: MaskSource,
    mode: MaskMode,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskSource {
    kind: MaskSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaskSourceKind {
    Shape(Shape),
    ImageLayer(StyleImageLayer),
    Reference(StyleResourceRef),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskMode {
    Alpha,
    Luminance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskLayerStack {
    layers: Vec<MaskLayer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskLayer {
    input: MaskInput,
    composite_mode: MaskCompositeMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskCompositeMode {
    Add,
    Subtract,
    Intersect,
    Exclude,
}

impl MaskInput {
    pub fn try_shape(shape: Shape, mode: MaskMode) -> Result<Self> {
        Ok(Self {
            source: MaskSource::try_shape(shape)?,
            mode,
            coordinate_space: None,
        })
    }

    #[must_use]
    pub const fn image_layer(layer: StyleImageLayer, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::image_layer(layer),
            mode,
            coordinate_space: None,
        }
    }

    #[must_use]
    pub const fn reference(reference: StyleResourceRef, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::reference(reference),
            mode,
            coordinate_space: None,
        }
    }

    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    #[must_use]
    pub const fn source(&self) -> &MaskSource {
        &self.source
    }

    #[must_use]
    pub const fn mode(&self) -> MaskMode {
        self.mode
    }

    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.source.kind() {
            MaskSourceKind::Reference(reference) => {
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::Mask,
                    reference.identifier(),
                )));
            }
            MaskSourceKind::ImageLayer(layer) => {
                layer.source().require_resolved()?;
            }
            MaskSourceKind::Shape(_) => {}
        }

        match self.mode {
            MaskMode::Alpha => Err(Error::unsupported_render_primitive(
                UnsupportedPrimitive::new(
                    PrimitiveFamily::MasksAndClips,
                    PrimitiveOperation::AlphaMaskSourceExecution,
                ),
            )),
            MaskMode::Luminance => capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LuminanceMaskMode,
            )),
        }
    }

    fn ensure_stack_input_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.source.kind() {
            MaskSourceKind::Reference(reference) => {
                return Err(Error::unresolved_resource(UnresolvedResource::new(
                    UnresolvedResourceKind::Mask,
                    reference.identifier(),
                )));
            }
            MaskSourceKind::ImageLayer(layer) => {
                layer.source().require_resolved()?;
            }
            MaskSourceKind::Shape(_) => {}
        }

        match self.mode {
            MaskMode::Alpha => Ok(()),
            MaskMode::Luminance => capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::LuminanceMaskMode,
            )),
        }
    }
}

impl MaskLayerStack {
    pub fn try_new(layers: impl IntoIterator<Item = MaskLayer>) -> Result<Self> {
        let layers = layers.into_iter().collect::<Vec<_>>();
        if layers.is_empty() {
            return Err(Error::invalid_value(
                "mask layer stack",
                0,
                "must contain at least one layer",
            ));
        }
        Ok(Self { layers })
    }

    #[must_use]
    pub fn single(layer: impl Into<MaskLayer>) -> Self {
        Self {
            layers: vec![layer.into()],
        }
    }

    #[must_use]
    pub fn layers(&self) -> &[MaskLayer] {
        &self.layers
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn ensure_supported(&self, capabilities: Capabilities) -> Result<()> {
        for layer in &self.layers {
            layer.input.ensure_stack_input_supported(capabilities)?;
            layer.ensure_composite_supported(capabilities)?;
        }

        if self.layers.len() > 1 {
            return capabilities.ensure_supported(UnsupportedPrimitive::new(
                PrimitiveFamily::MasksAndClips,
                PrimitiveOperation::MultiLayerMaskComposition,
            ));
        }

        self.layers[0].input.ensure_supported(capabilities)
    }
}

impl MaskLayer {
    #[must_use]
    pub const fn new(input: MaskInput) -> Self {
        Self {
            input,
            composite_mode: MaskCompositeMode::Add,
        }
    }

    pub const fn try_new(input: MaskInput, composite_mode: MaskCompositeMode) -> Result<Self> {
        Ok(Self {
            input,
            composite_mode,
        })
    }

    #[must_use]
    pub const fn input(&self) -> &MaskInput {
        &self.input
    }

    #[must_use]
    pub const fn composite_mode(&self) -> MaskCompositeMode {
        self.composite_mode
    }

    fn ensure_composite_supported(&self, capabilities: Capabilities) -> Result<()> {
        match self.composite_mode {
            MaskCompositeMode::Add => Ok(()),
            MaskCompositeMode::Subtract
            | MaskCompositeMode::Intersect
            | MaskCompositeMode::Exclude => {
                capabilities.ensure_supported(UnsupportedPrimitive::new(
                    PrimitiveFamily::MasksAndClips,
                    PrimitiveOperation::MaskCompositeMode,
                ))
            }
        }
    }
}

impl From<MaskInput> for MaskLayer {
    fn from(input: MaskInput) -> Self {
        Self::new(input)
    }
}

impl MaskSource {
    fn try_shape(shape: Shape) -> Result<Self> {
        validate_shape(&shape)?;
        Ok(Self {
            kind: MaskSourceKind::Shape(shape),
        })
    }

    const fn image_layer(layer: StyleImageLayer) -> Self {
        Self {
            kind: MaskSourceKind::ImageLayer(layer),
        }
    }

    const fn reference(reference: StyleResourceRef) -> Self {
        Self {
            kind: MaskSourceKind::Reference(reference),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> &MaskSourceKind {
        &self.kind
    }
}
