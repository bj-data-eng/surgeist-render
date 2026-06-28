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
