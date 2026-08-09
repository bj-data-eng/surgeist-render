use super::super::{
    Capabilities, CoordinateSpaceTag, Error, PrimitiveFamily, PrimitiveOperation, Result, Shape,
    UnresolvedResource, UnresolvedResourceKind, UnsupportedPrimitive, validation::validate_shape,
};
use super::image::{StyleImageLayer, StyleResourceRef};

#[derive(Clone, Debug, PartialEq)]
/// An authored mask source, interpretation mode, and optional coordinate space.
///
/// Symbolic resource references and unresolved image-layer sources remain
/// diagnostic inputs until the surrounding context supplies concrete resources.
pub struct MaskInput {
    source: MaskSource,
    mode: MaskMode,
    coordinate_space: Option<CoordinateSpaceTag>,
}

#[derive(Clone, Debug, PartialEq)]
/// The source wrapper stored by a [`MaskInput`].
pub struct MaskSource {
    kind: MaskSourceKind,
}

#[derive(Clone, Debug, PartialEq)]
/// The authored content used as a mask source.
pub enum MaskSourceKind {
    /// Validated logical shape geometry.
    Shape(Shape),
    /// An authored image layer whose source must be concrete before execution.
    ImageLayer(StyleImageLayer),
    /// A symbolic mask-resource reference.
    Reference(StyleResourceRef),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// How source channels are interpreted as mask coverage.
pub enum MaskMode {
    /// Uses the source alpha channel; direct single-source execution currently
    /// returns an unsupported-primitive diagnostic.
    Alpha,
    /// Derives coverage from luminance when the supplied capabilities support it.
    Luminance,
}

#[derive(Clone, Debug, PartialEq)]
/// A non-empty ordered stack of authored mask layers.
pub struct MaskLayerStack {
    layers: Vec<MaskLayer>,
}

#[derive(Clone, Debug, PartialEq)]
/// One authored mask input and its composition choice.
pub struct MaskLayer {
    input: MaskInput,
    composite_mode: MaskCompositeMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// How a mask layer combines with preceding layers in stack order.
pub enum MaskCompositeMode {
    /// Selects additive coverage composition.
    Add,
    /// Selects subtractive coverage composition; accepted only when capabilities report support.
    Subtract,
    /// Selects coverage intersection; accepted only when capabilities report support.
    Intersect,
    /// Selects exclusion composition; accepted only when capabilities report support.
    Exclude,
}

impl MaskInput {
    /// Creates a mask from validated logical shape geometry.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the shape is invalid.
    pub fn try_shape(shape: Shape, mode: MaskMode) -> Result<Self> {
        Ok(Self {
            source: MaskSource::try_shape(shape)?,
            mode,
            coordinate_space: None,
        })
    }

    /// Creates a mask from an authored image layer.
    #[must_use]
    pub const fn image_layer(layer: StyleImageLayer, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::image_layer(layer),
            mode,
            coordinate_space: None,
        }
    }

    /// Creates a mask from a symbolic resource reference.
    #[must_use]
    pub const fn reference(reference: StyleResourceRef, mode: MaskMode) -> Self {
        Self {
            source: MaskSource::reference(reference),
            mode,
            coordinate_space: None,
        }
    }

    /// Associates the mask with a tagged coordinate space.
    #[must_use]
    pub fn with_coordinate_space(mut self, coordinate_space: CoordinateSpaceTag) -> Self {
        self.coordinate_space = Some(coordinate_space);
        self
    }

    /// Returns the authored mask source.
    #[must_use]
    pub const fn source(&self) -> &MaskSource {
        &self.source
    }

    /// Returns the source-channel interpretation mode.
    #[must_use]
    pub const fn mode(&self) -> MaskMode {
        self.mode
    }

    /// Returns the optional coordinate-space tag.
    #[must_use]
    pub const fn coordinate_space(&self) -> Option<CoordinateSpaceTag> {
        self.coordinate_space
    }

    /// Checks source resolution and the current mask-execution capability.
    ///
    /// Symbolic references and unresolved image sources return unresolved-resource
    /// diagnostics. Alpha mode currently returns an unsupported alpha-source
    /// diagnostic; luminance mode is checked against the supplied capabilities.
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
    /// Collects a non-empty ordered stack of mask layers.
    ///
    /// Returns [`crate::ErrorCode::InvalidInput`] when the iterator is empty.
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

    /// Creates a stack containing exactly one layer.
    #[must_use]
    pub fn single(layer: impl Into<MaskLayer>) -> Self {
        Self {
            layers: vec![layer.into()],
        }
    }

    /// Returns the mask layers in composition order.
    #[must_use]
    pub fn layers(&self) -> &[MaskLayer] {
        &self.layers
    }

    /// Returns the number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether the stack contains no layers.
    ///
    /// Public construction keeps this false.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Checks source resolution, modes, composition choices, and stack size.
    ///
    /// Non-`Add` composition and multiple layers are checked against their
    /// semantic capabilities. A single layer is additionally checked through
    /// [`MaskInput::ensure_supported`].
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
    /// Creates a mask layer using additive composition.
    #[must_use]
    pub const fn new(input: MaskInput) -> Self {
        Self {
            input,
            composite_mode: MaskCompositeMode::Add,
        }
    }

    /// Creates a mask layer with an explicit composition mode.
    ///
    /// The mode is retained as authored; capability checks occur when the layer
    /// stack is checked for support.
    pub const fn try_new(input: MaskInput, composite_mode: MaskCompositeMode) -> Result<Self> {
        Ok(Self {
            input,
            composite_mode,
        })
    }

    /// Returns the mask input.
    #[must_use]
    pub const fn input(&self) -> &MaskInput {
        &self.input
    }

    /// Returns the composition mode.
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
    /// Converts a mask input into an additively composed layer without loss.
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

    /// Returns the source choice.
    #[must_use]
    pub const fn kind(&self) -> &MaskSourceKind {
        &self.kind
    }
}
