use super::{
    command::{RenderCommands, normalize_commands},
    stats::collect_stats,
    *,
};

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

    pub fn shadows(&mut self, shape: impl Into<Shape>, shadows: ShadowList) -> &mut Self {
        let shape = shape.into();
        for shadow in shadows.into_vec() {
            self.commands.push(Command::Shadow {
                shape: shape.clone(),
                shadow,
            });
        }
        self
    }

    pub fn image(&mut self, image: Image, rect: Rect, fit: ImageFit) -> &mut Self {
        self.commands.push(Command::Image { image, rect, fit });
        self
    }

    pub fn text_run(&mut self, run: TextRun<'_>) -> &mut Self {
        self.commands.push(Command::TextRun {
            font: run.font().to_owned_static(),
            size: run.size(),
            transform: run.transform(),
            paint: run.paint().clone(),
            glyphs: run.glyphs().to_vec(),
            bounds: run.bounds(),
        });
        self
    }

    pub fn text_shadow_run(&mut self, run: TextShadowRun<'_>) -> &mut Self {
        let text = run.run();
        self.commands.push(Command::TextShadowRun {
            font: text.font().to_owned_static(),
            size: text.size(),
            transform: text.transform(),
            paint: text.paint().clone(),
            glyphs: text.glyphs().to_vec(),
            bounds: text.bounds(),
            shadows: run.shadows().clone(),
        });
        self
    }

    pub fn text_decoration_line(&mut self, line: TextDecorationLine) -> &mut Self {
        let mut path = Path::new();
        path.move_to(line.start()).line_to(line.end());
        let stroke = Stroke::try_new(line.thickness())
            .expect("TextDecorationLine constructors validate positive thickness");
        if line.transform() == Transform::identity() {
            return self.stroke(Shape::path(path), stroke, line.paint().clone());
        }

        let paint = line.paint().clone();
        self.transform(line.transform(), move |scene| {
            scene.stroke(Shape::path(path), stroke, paint);
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

    pub(crate) fn normalize(&self, capabilities: Capabilities) -> Result<RenderCommands> {
        normalize_commands(&self.commands, capabilities).map(RenderCommands::new)
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
        bounds: TextRunBounds,
    },
    TextShadowRun {
        font: FontRef<'static>,
        size: f32,
        transform: Transform,
        paint: TextPaint,
        glyphs: Vec<TextGlyph>,
        bounds: TextRunBounds,
        shadows: ShadowList,
    },
    Layer {
        layer: Layer,
        children: Vec<Command>,
    },
}
