use super::{stats::collect_stats, *};
use std::borrow::Cow;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub(crate) commands: Vec<Command>,
}

impl Scene {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.commands.clear();
    }

    pub fn fill(&mut self, shape: impl Into<Shape>, paint: impl Into<Paint>) -> &mut Self {
        self.commands.push(Command::Fill {
            shape: shape.into(),
            paint: paint.into(),
        });
        self
    }

    pub fn stroke(
        &mut self,
        shape: impl Into<Shape>,
        stroke: Stroke,
        paint: impl Into<Paint>,
    ) -> &mut Self {
        self.commands.push(Command::Stroke {
            shape: shape.into(),
            stroke,
            paint: paint.into(),
        });
        self
    }

    pub fn shadow(&mut self, shape: impl Into<Shape>, shadow: Shadow) -> &mut Self {
        self.commands.push(Command::Shadow {
            shape: shape.into(),
            shadow,
        });
        self
    }

    pub fn image(&mut self, image: Image, rect: Rect, fit: ImageFit) -> &mut Self {
        self.commands.push(Command::Image { image, rect, fit });
        self
    }

    pub fn text_run(&mut self, run: TextRun<'_>) -> &mut Self {
        self.commands.push(Command::TextRun {
            font: FontRef {
                id: run.font().id,
                name: run
                    .font()
                    .name
                    .as_ref()
                    .map(|name| Cow::Owned(name.clone().into_owned())),
                data: run.font().data.clone(),
            },
            size: run.size(),
            transform: run.transform(),
            paint: run.paint().clone(),
            glyphs: run.glyphs().to_vec(),
        });
        self
    }

    pub fn layer(&mut self, layer: Layer, children: impl FnOnce(&mut Scene)) -> &mut Self {
        let mut child = Scene::new();
        children(&mut child);
        self.commands.push(Command::Layer {
            layer,
            children: child.commands,
        });
        self
    }

    pub fn transform(
        &mut self,
        transform: Transform,
        children: impl FnOnce(&mut Scene),
    ) -> &mut Self {
        let layer = Layer::new()
            .try_transform(transform)
            .expect("Transform constructors validate layer transform values");
        self.layer(layer, children)
    }

    pub fn clip(
        &mut self,
        shape: impl Into<Shape>,
        children: impl FnOnce(&mut Scene),
    ) -> &mut Self {
        let layer = Layer::new()
            .try_clip(shape.into())
            .expect("Shape constructors validate layer clip values");
        self.layer(layer, children)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    #[must_use]
    pub fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        let mut uploaded_images = std::collections::HashSet::new();
        collect_stats(&self.commands, &mut stats, &mut uploaded_images);
        stats
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Command {
    Fill {
        shape: Shape,
        paint: Paint,
    },
    Stroke {
        shape: Shape,
        stroke: Stroke,
        paint: Paint,
    },
    Shadow {
        shape: Shape,
        shadow: Shadow,
    },
    Image {
        image: Image,
        rect: Rect,
        fit: ImageFit,
    },
    TextRun {
        font: FontRef<'static>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: Vec<TextGlyph>,
    },
    Layer {
        layer: Layer,
        children: Vec<Command>,
    },
}
