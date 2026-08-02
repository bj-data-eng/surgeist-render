#![forbid(unsafe_code)]

use surgeist_render::{
    Attachment, Color, EffectQualityPolicy, Image, Layer, Options, Parameters, Rect, RenderRoute,
    RenderSurfaceAvailability, Renderer, ResolvedLayerAlphaMask,
    RuntimeCapabilityUnavailableReason, Scene, Size, Surface, SurfaceOptions,
};
use surgeist_window::{Frame, Handler, Ready, Resize, Result, Scope};

#[derive(Default)]
struct PresentedRouteSmoke {
    renderer: Option<Renderer>,
    surface: Option<Surface>,
    presented_frames: u8,
}

impl Handler for PresentedRouteSmoke {
    fn ready(&mut self, ready: &mut Ready<'_>) -> Result<()> {
        let metrics = ready.metrics();
        let size = render_size(metrics.logical_size())?;
        let mut renderer = pollster::block_on(Renderer::new(
            Options::default()
                .with_effect_quality_policy(EffectQualityPolicy::AllowReducedPrecision),
        ))
        .map_err(|source| render_error("failed to create the smoke renderer", source))?;
        let surface = pollster::block_on(renderer.create_surface(
            Attachment::from_window(ready.handle()?),
            SurfaceOptions {
                size,
                scale: metrics.scale_factor(),
                ..SurfaceOptions::default()
            },
        ))
        .map_err(|source| render_error("failed to create the presented smoke surface", source))?;

        self.renderer = Some(renderer);
        self.surface = Some(surface);
        Ok(())
    }

    fn resize(&mut self, resize: &mut Resize<'_>) -> Result<()> {
        let size = render_size(resize.size())?;
        if let Some(surface) = self.surface.as_mut() {
            surface
                .resize(size, resize.scale())
                .map_err(|source| render_error("failed to resize the smoke surface", source))?;
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame<'_>) -> Result<()> {
        if frame.is_occluded() {
            frame.draw();
            return Ok(());
        }
        match self.presented_frames {
            0 => {
                let size = required_surface(self)?.size();
                let mut scene = Scene::new();
                scene.fill(
                    Rect::try_new(0.0, 0.0, size.width(), size.height())
                        .map_err(|source| render_error("invalid direct-frame bounds", source))?,
                    Color::BLACK,
                );
                let Some(stats) = render_presented(self, frame, &scene)? else {
                    return Ok(());
                };
                assert_eq!(stats.route, Some(RenderRoute::DirectVello));
                self.presented_frames = 1;
                frame.again();
            }
            1 => {
                let size = required_surface(self)?.size();
                let bounds = Rect::try_new(0.0, 0.0, size.width(), size.height())
                    .map_err(|source| render_error("invalid graph-frame bounds", source))?;
                let mask = Image::from_rgba(
                    Size::try_new(1.0, 1.0)
                        .map_err(|source| render_error("invalid mask size", source))?,
                    vec![255, 255, 255, 255],
                )
                .map_err(|source| render_error("invalid graph mask", source))?;
                let layer = Layer::new().with_resolved_alpha_mask(
                    ResolvedLayerAlphaMask::try_new(mask, bounds)
                        .map_err(|source| render_error("invalid graph mask bounds", source))?,
                );
                let color = Color::try_rgba(0.1, 0.45, 0.9, 1.0)
                    .map_err(|source| render_error("invalid graph color", source))?;
                let mut scene = Scene::new();
                scene.layer(layer, |scene| {
                    scene.fill(bounds, color);
                });
                let Some(stats) = render_presented(self, frame, &scene)? else {
                    return Ok(());
                };
                assert_eq!(stats.route, Some(RenderRoute::GpuGraph));
                self.presented_frames = 2;
                drop(self.surface.take());
                drop(self.renderer.take());
                frame.exit();
            }
            _ => {
                frame.exit();
            }
        };
        Ok(())
    }
}

fn render_presented(
    smoke: &mut PresentedRouteSmoke,
    frame: &mut Frame<'_>,
    scene: &Scene,
) -> Result<Option<surgeist_render::Stats>> {
    let PresentedRouteSmoke {
        renderer: Some(renderer),
        surface: Some(surface),
        ..
    } = smoke
    else {
        return Err(surgeist_window::Error::new(
            surgeist_window::ErrorCode::UnknownNativeError,
            "the presented renderer and surface must exist before drawing",
        ));
    };
    match pollster::block_on(renderer.render(surface, scene, Parameters::default())) {
        Ok(stats) => Ok(Some(stats)),
        Err(source)
            if matches!(
                source
                    .runtime_capability_unavailable_diagnostic()
                    .map(|diagnostic| diagnostic.reason()),
                Some(RuntimeCapabilityUnavailableReason::SurfaceUnavailable {
                    state: RenderSurfaceAvailability::Occluded
                })
            ) =>
        {
            frame.draw();
            Ok(None)
        }
        Err(source) => Err(render_error(
            "failed to render and present a smoke frame",
            source,
        )),
    }
}

fn required_surface(smoke: &PresentedRouteSmoke) -> Result<&Surface> {
    smoke.surface.as_ref().ok_or_else(|| {
        surgeist_window::Error::new(
            surgeist_window::ErrorCode::UnknownNativeError,
            "the presented surface must exist before drawing",
        )
    })
}

fn render_size(size: surgeist_window::Size) -> Result<Size> {
    Size::try_new(size.width, size.height)
        .map_err(|source| render_error("the native window reported an invalid size", source))
}

fn render_error(context: &'static str, source: surgeist_render::Error) -> surgeist_window::Error {
    surgeist_window::Error::new(surgeist_window::ErrorCode::UnknownNativeError, context)
        .with_source(source)
}

fn main() -> Result<()> {
    surgeist_window::app(PresentedRouteSmoke::default())
        .open(
            surgeist_window::open("render-window-smoke")
                .title("Surgeist presented route smoke")
                .size(surgeist_window::size(640, 360)),
        )
        .run()
}
