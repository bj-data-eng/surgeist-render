use super::FrameContext;
use crate::{
    command::{RenderClip, RenderCommand, RenderLayerMask, commands_bounds_for_planning},
    error::{Error, Result},
    filter::{DevicePixelConversionPolicy, FilterOutset, FilterRegionPlan, FilterSourceBounds},
    geometry::{PhysicalSize, Point, Rect, Transform},
    text::TextRunBoundsKind,
};
#[cfg(test)]
use crate::{geometry::Size, paint::Color, renderer::Antialiasing};

impl FrameContext {
    #[cfg(test)]
    pub(super) fn try_for_spatial_test(surface_scale: f64) -> Result<Self> {
        Self::try_new(
            Size::new(1.0, 1.0),
            surface_scale,
            Antialiasing::Area,
            Color::TRANSPARENT,
        )
    }

    pub(super) fn output_spatial_plan(self) -> Result<FrameSpatialPlan> {
        self.plan_local_bounds(self.output_bounds, Transform::identity())
    }

    pub(super) fn initial_parent_contribution(self) -> SemanticSourceBounds {
        match self.output_bounds {
            LogicalBounds::NonEmpty(bounds) if self.base_color.a() > 0.0 => {
                SemanticSourceBounds::exact_known(bounds)
            }
            LogicalBounds::Empty(_) | LogicalBounds::NonEmpty(_) => {
                SemanticSourceBounds::exactly_empty()
            }
        }
    }

    pub(super) fn plan_local_bounds(
        self,
        logical_bounds: LogicalBounds,
        transform: Transform,
    ) -> Result<FrameSpatialPlan> {
        let linear_metrics = linear_transform_metrics(transform)?;
        let logical_bounds = match logical_bounds {
            LogicalBounds::Empty(bounds) => {
                return Ok(FrameSpatialPlan::Empty(EmptyFrameSpatialPlan {
                    logical_bounds: LogicalBounds::Empty(bounds),
                }));
            }
            LogicalBounds::NonEmpty(bounds) => bounds,
        };
        if linear_metrics.rank_deficient {
            return Ok(FrameSpatialPlan::Empty(EmptyFrameSpatialPlan {
                logical_bounds: LogicalBounds::NonEmpty(logical_bounds),
            }));
        }

        let raster_scale = RasterScale::try_new(checked_mul(
            self.surface_scale,
            linear_metrics.largest_singular_value,
            "frame local raster scale",
        )?)?;
        let source = FilterSourceBounds::try_new(logical_bounds.rect())?;
        let region = FilterRegionPlan::try_new(source, FilterOutset::zero(), None)?;
        let device = DevicePixelConversionPolicy::outward()
            .convert_region(region.execution_region(), raster_scale.get())?;
        let device_origin = SignedDeviceOrigin::new(device.x(), device.y());
        let device_extent = PositiveDeviceExtent::try_new(device.width(), device.height())?;
        let texel_center_mapping = TexelCenterMapping::try_new(device_origin, raster_scale)?;

        Ok(FrameSpatialPlan::NonEmpty(NonEmptyFrameSpatialPlan {
            logical_bounds,
            device_origin,
            device_extent,
            raster_scale,
            texel_center_mapping,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SemanticSourceBounds {
    known_non_empty_extent: Option<NonEmptyLogicalBounds>,
    contains_unresolved_content: bool,
}

impl SemanticSourceBounds {
    const fn from_parts(
        known_non_empty_extent: Option<NonEmptyLogicalBounds>,
        contains_unresolved_content: bool,
    ) -> Self {
        Self {
            known_non_empty_extent,
            contains_unresolved_content,
        }
    }

    pub(super) const fn exactly_empty() -> Self {
        Self::from_parts(None, false)
    }

    pub(super) const fn exact_known(bounds: NonEmptyLogicalBounds) -> Self {
        Self::from_parts(Some(bounds), false)
    }

    const fn wholly_unresolved() -> Self {
        Self::from_parts(None, true)
    }

    pub(super) const fn is_exactly_empty(self) -> bool {
        self.known_non_empty_extent.is_none() && !self.contains_unresolved_content
    }

    const fn known_non_empty_extent(self) -> Option<NonEmptyLogicalBounds> {
        self.known_non_empty_extent
    }

    const fn contains_unresolved_content(self) -> bool {
        self.contains_unresolved_content
    }

    pub(super) const fn from_logical_bounds(bounds: LogicalBounds) -> Self {
        match bounds {
            LogicalBounds::Empty(_) => Self::exactly_empty(),
            LogicalBounds::NonEmpty(bounds) => Self::exact_known(bounds),
        }
    }

    fn try_for_command(command: &RenderCommand) -> Result<Self> {
        match commands_bounds_for_planning(std::slice::from_ref(command))? {
            Some(bounds) => Self::try_from_rect(bounds.rect()),
            None => Ok(Self::wholly_unresolved()),
        }
    }

    fn try_from_rect(rect: Rect) -> Result<Self> {
        LogicalBounds::try_from_rect(rect, "semantic source bounds").map(Self::from_logical_bounds)
    }

    pub(super) fn try_union(self, other: Self) -> Result<Self> {
        let known_non_empty_extent = match (
            self.known_non_empty_extent(),
            other.known_non_empty_extent(),
        ) {
            (Some(a), Some(b)) => Some(a.try_union(b, "semantic source bounds union")?),
            (Some(bounds), None) | (None, Some(bounds)) => Some(bounds),
            (None, None) => None,
        };
        Ok(Self::from_parts(
            known_non_empty_extent,
            self.contains_unresolved_content() || other.contains_unresolved_content(),
        ))
    }

    pub(super) fn try_intersect(self, other: Self, name: &'static str) -> Result<Self> {
        if self.is_exactly_empty() || other.is_exactly_empty() {
            return Ok(Self::exactly_empty());
        }
        let known_non_empty_extent = match (
            self.known_non_empty_extent(),
            other.known_non_empty_extent(),
        ) {
            (Some(a), Some(b)) => {
                let a = a.rect();
                let b = b.rect();
                let a_max = a.max();
                let b_max = b.max();
                let min_x = a.x().max(b.x());
                let min_y = a.y().max(b.y());
                let max_x = a_max.x().min(b_max.x());
                let max_y = a_max.y().min(b_max.y());
                let width = checked_sub(max_x, min_x, &format!("{name} width"))?.max(0.0);
                let height = checked_sub(max_y, min_y, &format!("{name} height"))?.max(0.0);
                Self::try_from_rect(Rect::new(min_x, min_y, width, height))?
                    .known_non_empty_extent()
            }
            (Some(_), None) | (None, Some(_)) | (None, None) => None,
        };
        Ok(Self::from_parts(
            known_non_empty_extent,
            self.contains_unresolved_content() || other.contains_unresolved_content(),
        ))
    }

    fn try_transform(self, transform: Transform, name: &'static str) -> Result<Self> {
        if linear_transform_is_rank_deficient(transform)? {
            return Ok(Self::exactly_empty());
        }
        let known_non_empty_extent = match self.known_non_empty_extent() {
            Some(bounds) => {
                match LogicalBounds::NonEmpty(bounds).try_transform(transform, name)? {
                    LogicalBounds::Empty(_) => None,
                    LogicalBounds::NonEmpty(bounds) => Some(bounds),
                }
            }
            None => None,
        };
        Ok(Self::from_parts(
            known_non_empty_extent,
            self.contains_unresolved_content(),
        ))
    }

    pub(super) fn try_for_clip(clip: &RenderClip) -> Result<Self> {
        Self::try_from_rect(clip.bounds_for_planning()?.rect())
    }

    pub(super) fn require_non_empty_for_graph(
        self,
        name: &'static str,
    ) -> Result<Option<NonEmptyLogicalBounds>> {
        if self.contains_unresolved_content() {
            return Err(Error::invalid_value(
                name,
                "unspecified",
                "must be explicit before semantic graph source planning",
            ));
        }
        Ok(self.known_non_empty_extent())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SemanticSourceContribution {
    pub(super) commands: Vec<RenderCommand>,
    pub(super) source_bounds: SemanticSourceBounds,
    current_parent: SemanticSourceBounds,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SemanticContributionDomain {
    RootOutputBounded(LogicalBounds),
    LocalUnbounded,
}

impl SemanticSourceContribution {
    pub(super) fn try_from_commands(
        commands: Vec<RenderCommand>,
        initial_parent: SemanticSourceBounds,
        domain: SemanticContributionDomain,
        context: FrameContext,
        local_to_surface: Transform,
    ) -> Result<Self> {
        let mut contributing_commands = Vec::with_capacity(commands.len());
        let mut source_bounds = SemanticSourceBounds::exactly_empty();
        let mut current_parent = initial_parent;
        for command in commands {
            let prior_parent = current_parent;
            let contribution =
                Self::try_from_command(command, prior_parent, context, local_to_surface)?
                    .try_apply_domain(domain, prior_parent)?;
            current_parent = contribution.current_parent;
            if let Some(command) = contribution.command {
                source_bounds = source_bounds.try_union(contribution.source_bounds)?;
                contributing_commands.push(command);
            }
        }
        Ok(Self {
            commands: contributing_commands,
            source_bounds,
            current_parent,
        })
    }

    fn try_from_command(
        command: RenderCommand,
        current_parent: SemanticSourceBounds,
        context: FrameContext,
        local_to_surface: Transform,
    ) -> Result<SemanticCommandContribution> {
        match command {
            command @ RenderCommand::TextRun { .. } => {
                Self::try_from_text_run(command, current_parent)
            }
            command @ RenderCommand::Layer { .. } => {
                Self::try_from_layer(command, current_parent, context, local_to_surface)
            }
            command @ (RenderCommand::Fill { .. }
            | RenderCommand::Stroke { .. }
            | RenderCommand::Shadow { .. }
            | RenderCommand::Image { .. }) => {
                let bounds = SemanticSourceBounds::try_for_command(&command)?;
                SemanticCommandContribution::try_new(command, bounds, current_parent)
            }
        }
    }

    fn try_from_text_run(
        command: RenderCommand,
        current_parent: SemanticSourceBounds,
    ) -> Result<SemanticCommandContribution> {
        let RenderCommand::TextRun {
            transform,
            ref glyphs,
            bounds,
            ..
        } = command
        else {
            unreachable!("text contribution helper received a non-text command");
        };
        let source_bounds = if glyphs.is_empty() || bounds.kind() == TextRunBoundsKind::Empty {
            SemanticSourceBounds::exactly_empty()
                .try_transform(transform, "text source transform")?
        } else if bounds.kind() == TextRunBoundsKind::Unspecified {
            SemanticSourceBounds::wholly_unresolved()
                .try_transform(transform, "text source transform")?
        } else {
            let ink_bounds = bounds.ink_rect().ok_or_else(|| {
                Error::invalid_value(
                    "text source bounds",
                    "missing ink rectangle",
                    "must carry an ink rectangle when the bounds kind is ink",
                )
            })?;
            SemanticSourceBounds::try_from_rect(ink_bounds)?
                .try_transform(transform, "text source transform")?
        };
        SemanticCommandContribution::try_new(command, source_bounds, current_parent)
    }

    fn try_from_layer(
        command: RenderCommand,
        current_parent: SemanticSourceBounds,
        context: FrameContext,
        local_to_surface: Transform,
    ) -> Result<SemanticCommandContribution> {
        let RenderCommand::Layer {
            mut layer,
            children,
        } = command
        else {
            unreachable!("layer contribution helper received a non-layer command");
        };
        let layer_to_surface = layer.transform.then(local_to_surface)?;
        let children = Self::try_from_commands(
            children,
            SemanticSourceBounds::exactly_empty(),
            SemanticContributionDomain::LocalUnbounded,
            context,
            layer_to_surface,
        )?;
        let mut source_bounds = children.current_parent;
        source_bounds = Self::include_backdrop_contribution(
            source_bounds,
            current_parent,
            &mut layer,
            context,
            layer_to_surface,
        )?;
        if let Some(clip) = layer.clip.as_ref() {
            source_bounds = source_bounds.try_intersect(
                SemanticSourceBounds::try_for_clip(clip)?,
                "layer clip intersection",
            )?;
        }
        if layer
            .mask
            .as_ref()
            .is_some_and(RenderLayerMask::annihilates_source)
            || layer.opacity <= 0.0
        {
            source_bounds = SemanticSourceBounds::exactly_empty();
        }
        source_bounds =
            source_bounds.try_transform(layer.transform, "semantic layer source transform")?;
        SemanticCommandContribution::try_new(
            RenderCommand::Layer {
                layer,
                children: children.commands,
            },
            source_bounds,
            current_parent,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SemanticCommandContribution {
    command: Option<RenderCommand>,
    source_bounds: SemanticSourceBounds,
    current_parent: SemanticSourceBounds,
}

impl SemanticCommandContribution {
    fn try_new(
        command: RenderCommand,
        source_bounds: SemanticSourceBounds,
        current_parent: SemanticSourceBounds,
    ) -> Result<Self> {
        if source_bounds.is_exactly_empty() {
            return Ok(Self {
                command: None,
                source_bounds,
                current_parent,
            });
        }
        Ok(Self {
            command: Some(command),
            source_bounds,
            current_parent: current_parent.try_union(source_bounds)?,
        })
    }

    fn try_apply_domain(
        self,
        domain: SemanticContributionDomain,
        prior_parent: SemanticSourceBounds,
    ) -> Result<Self> {
        let SemanticContributionDomain::RootOutputBounded(output_bounds) = domain else {
            return Ok(self);
        };
        let output_bounds = SemanticSourceBounds::from_logical_bounds(output_bounds);
        let source_bounds = self.source_bounds.try_intersect(
            output_bounds,
            "root observable source output-domain intersection",
        )?;
        let current_parent = prior_parent
            .try_intersect(
                output_bounds,
                "root prior-parent output-domain intersection",
            )?
            .try_union(source_bounds)?
            .try_intersect(
                output_bounds,
                "root current-parent output-domain intersection",
            )?;
        let command = if source_bounds.is_exactly_empty() {
            None
        } else {
            self.command
        };
        Ok(Self {
            command,
            source_bounds,
            current_parent,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LogicalBounds {
    Empty(EmptyLogicalBounds),
    NonEmpty(NonEmptyLogicalBounds),
}

impl LogicalBounds {
    pub(crate) fn try_from_rect(rect: Rect, name: &str) -> Result<Self> {
        validate_finite(rect.x(), &format!("{name} x"))?;
        validate_finite(rect.y(), &format!("{name} y"))?;
        validate_non_negative(rect.width(), &format!("{name} width"))?;
        validate_non_negative(rect.height(), &format!("{name} height"))?;
        checked_add(rect.x(), rect.width(), &format!("{name} max x"))?;
        checked_add(rect.y(), rect.height(), &format!("{name} max y"))?;

        if rect.width() == 0.0 || rect.height() == 0.0 {
            Ok(Self::Empty(EmptyLogicalBounds { rect }))
        } else {
            Ok(Self::NonEmpty(NonEmptyLogicalBounds { rect }))
        }
    }

    pub(crate) fn try_transform(self, transform: Transform, name: &str) -> Result<Self> {
        let Self::NonEmpty(bounds) = self else {
            return Ok(self);
        };
        let rect = bounds.rect();
        let max = rect.max();
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (x, y) in [
            (rect.x(), rect.y()),
            (max.x(), rect.y()),
            (max.x(), max.y()),
            (rect.x(), max.y()),
        ] {
            let point = transform_point(transform, x, y, name)?;
            min_x = min_x.min(point.x());
            min_y = min_y.min(point.y());
            max_x = max_x.max(point.x());
            max_y = max_y.max(point.y());
        }
        let width = checked_sub(max_x, min_x, &format!("{name} width"))?;
        let height = checked_sub(max_y, min_y, &format!("{name} height"))?;
        let transformed_rect = Rect::new(min_x, min_y, width, height);
        let transformed = Self::try_from_rect(transformed_rect, name)?;
        if linear_transform_is_rank_deficient(transform)?
            && matches!(transformed, Self::NonEmpty(_))
        {
            return Ok(Self::Empty(EmptyLogicalBounds {
                rect: Rect::new(min_x, min_y, 0.0, 0.0),
            }));
        }
        Ok(transformed)
    }

    pub(crate) fn union(self, other: Self, name: &str) -> Result<Self> {
        match (self, other) {
            (Self::Empty(_), bounds) | (bounds, Self::Empty(_)) => Ok(bounds),
            (Self::NonEmpty(a), Self::NonEmpty(b)) => {
                let a = a.rect();
                let b = b.rect();
                let a_max = a.max();
                let b_max = b.max();
                let min_x = a.x().min(b.x());
                let min_y = a.y().min(b.y());
                let max_x = a_max.x().max(b_max.x());
                let max_y = a_max.y().max(b_max.y());
                let width = checked_sub(max_x, min_x, &format!("{name} width"))?;
                let height = checked_sub(max_y, min_y, &format!("{name} height"))?;
                Self::try_from_rect(Rect::new(min_x, min_y, width, height), name)
            }
        }
    }

    #[must_use]
    pub(crate) const fn rect(self) -> Rect {
        match self {
            Self::Empty(bounds) => bounds.rect(),
            Self::NonEmpty(bounds) => bounds.rect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EmptyLogicalBounds {
    rect: Rect,
}

impl EmptyLogicalBounds {
    #[must_use]
    const fn rect(self) -> Rect {
        self.rect
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NonEmptyLogicalBounds {
    rect: Rect,
}

impl NonEmptyLogicalBounds {
    #[must_use]
    pub(super) const fn rect(self) -> Rect {
        self.rect
    }

    pub(super) fn try_inflate_uniform(self, amount: f64, name: &str) -> Result<Self> {
        validate_non_negative(amount, &format!("{name} outset"))?;
        let rect = self.rect();
        let doubled = checked_mul(amount, 2.0, &format!("{name} total outset"))?;
        non_empty_logical_bounds(
            Rect::new(
                checked_sub(rect.x(), amount, &format!("{name} x"))?,
                checked_sub(rect.y(), amount, &format!("{name} y"))?,
                checked_add(rect.width(), doubled, &format!("{name} width"))?,
                checked_add(rect.height(), doubled, &format!("{name} height"))?,
            ),
            name,
        )
    }

    pub(super) fn try_translate(self, offset: Point, name: &str) -> Result<Self> {
        let rect = self.rect();
        non_empty_logical_bounds(
            Rect::new(
                checked_add(rect.x(), offset.x(), &format!("{name} x"))?,
                checked_add(rect.y(), offset.y(), &format!("{name} y"))?,
                rect.width(),
                rect.height(),
            ),
            name,
        )
    }

    pub(super) fn try_union(self, other: Self, name: &str) -> Result<Self> {
        match LogicalBounds::NonEmpty(self).union(LogicalBounds::NonEmpty(other), name)? {
            LogicalBounds::NonEmpty(bounds) => Ok(bounds),
            LogicalBounds::Empty(_) => Err(Error::invalid_value(
                name,
                "empty",
                "must remain non-empty when two non-empty bounds are united",
            )),
        }
    }
}

fn non_empty_logical_bounds(rect: Rect, name: &str) -> Result<NonEmptyLogicalBounds> {
    match LogicalBounds::try_from_rect(rect, name)? {
        LogicalBounds::NonEmpty(bounds) => Ok(bounds),
        LogicalBounds::Empty(_) => Err(Error::invalid_value(
            name,
            "empty",
            "must have positive width and height",
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SignedDeviceOrigin {
    pub(super) x: i32,
    pub(super) y: i32,
}

impl SignedDeviceOrigin {
    #[must_use]
    pub(super) const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PositiveDeviceExtent {
    pub(super) width: u32,
    pub(super) height: u32,
}

impl PositiveDeviceExtent {
    fn try_new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_value(
                "frame device extent",
                format!("{width}x{height}"),
                "must have positive width and height",
            ));
        }
        Ok(Self { width, height })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RasterScale(f64);

impl RasterScale {
    fn try_new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_value(
                "frame local raster scale",
                value,
                "must be finite and greater than 0",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub(super) const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TexelCenterMapping {
    pub(super) origin: Point,
    pub(super) raster_scale: RasterScale,
}

impl TexelCenterMapping {
    fn try_new(device_origin: SignedDeviceOrigin, raster_scale: RasterScale) -> Result<Self> {
        let origin = Point::try_new(
            checked_div(
                f64::from(device_origin.x),
                raster_scale.get(),
                "frame texel mapping origin x",
            )?,
            checked_div(
                f64::from(device_origin.y),
                raster_scale.get(),
                "frame texel mapping origin y",
            )?,
        )?;
        Ok(Self {
            origin,
            raster_scale,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Texel-center mappings remain available for resolved pass lowering."
        )
    )]
    pub(super) fn point_for(self, i: u32, j: u32) -> Result<Point> {
        let x_offset = checked_div(
            checked_add(f64::from(i), 0.5, "frame texel center i")?,
            self.raster_scale.get(),
            "frame texel center x offset",
        )?;
        let y_offset = checked_div(
            checked_add(f64::from(j), 0.5, "frame texel center j")?,
            self.raster_scale.get(),
            "frame texel center y offset",
        )?;
        Point::try_new(
            checked_add(self.origin.x(), x_offset, "frame texel center x")?,
            checked_add(self.origin.y(), y_offset, "frame texel center y")?,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum FrameSpatialPlan {
    Empty(EmptyFrameSpatialPlan),
    NonEmpty(NonEmptyFrameSpatialPlan),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct EmptyFrameSpatialPlan {
    pub(super) logical_bounds: LogicalBounds,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NonEmptyFrameSpatialPlan {
    pub(super) logical_bounds: NonEmptyLogicalBounds,
    pub(super) device_origin: SignedDeviceOrigin,
    pub(super) device_extent: PositiveDeviceExtent,
    pub(super) raster_scale: RasterScale,
    pub(super) texel_center_mapping: TexelCenterMapping,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DestinationToLayerLocalMapping {
    pub(super) affine: Transform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LinearTransformMetrics {
    largest_singular_value: f64,
    rank_deficient: bool,
}

fn linear_transform_metrics(transform: Transform) -> Result<LinearTransformMetrics> {
    let [a, b, c, d, _, _] = transform.as_array();
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if coefficient_scale == 0.0 {
        return Ok(LinearTransformMetrics {
            largest_singular_value: 0.0,
            rank_deficient: true,
        });
    }

    let a = checked_div(a, coefficient_scale, "frame transform coefficient a")?;
    let b = checked_div(b, coefficient_scale, "frame transform coefficient b")?;
    let c = checked_div(c, coefficient_scale, "frame transform coefficient c")?;
    let d = checked_div(d, coefficient_scale, "frame transform coefficient d")?;
    let frobenius_squared = checked_add(
        checked_add(
            checked_mul(a, a, "frame transform a squared")?,
            checked_mul(b, b, "frame transform b squared")?,
            "frame transform first column norm",
        )?,
        checked_add(
            checked_mul(c, c, "frame transform c squared")?,
            checked_mul(d, d, "frame transform d squared")?,
            "frame transform second column norm",
        )?,
        "frame transform Frobenius norm squared",
    )?;
    let determinant = checked_sub(
        checked_mul(a, d, "frame transform normalized determinant ad")?,
        checked_mul(b, c, "frame transform normalized determinant bc")?,
        "frame transform normalized determinant",
    )?;
    let discriminant = checked_sub(
        checked_mul(
            frobenius_squared,
            frobenius_squared,
            "frame singular-value discriminant norm",
        )?,
        checked_mul(
            4.0,
            checked_mul(
                determinant,
                determinant,
                "frame singular-value determinant squared",
            )?,
            "frame singular-value determinant term",
        )?,
        "frame singular-value discriminant",
    )?
    .max(0.0);
    let largest_eigenvalue = checked_mul(
        0.5,
        checked_add(
            frobenius_squared,
            discriminant.sqrt(),
            "frame largest singular-value eigenvalue",
        )?,
        "frame largest singular-value eigenvalue",
    )?;
    let normalized = largest_eigenvalue.sqrt();
    validate_finite(normalized, "frame normalized largest singular value")?;
    let largest_singular_value = checked_mul(
        coefficient_scale,
        normalized,
        "frame largest singular value",
    )?;
    Ok(LinearTransformMetrics {
        largest_singular_value,
        rank_deficient: determinant == 0.0,
    })
}

fn linear_transform_is_rank_deficient(transform: Transform) -> Result<bool> {
    let [a, b, c, d, _, _] = transform.as_array();
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if coefficient_scale == 0.0 {
        return Ok(true);
    }

    let a = checked_div(a, coefficient_scale, "frame transform coefficient a")?;
    let b = checked_div(b, coefficient_scale, "frame transform coefficient b")?;
    let c = checked_div(c, coefficient_scale, "frame transform coefficient c")?;
    let d = checked_div(d, coefficient_scale, "frame transform coefficient d")?;
    let determinant = checked_sub(
        checked_mul(a, d, "frame transform normalized determinant ad")?,
        checked_mul(b, c, "frame transform normalized determinant bc")?,
        "frame transform normalized determinant",
    )?;
    Ok(determinant == 0.0)
}

pub(super) fn destination_to_layer_local_mapping(
    layer_to_destination: Transform,
) -> Result<Option<DestinationToLayerLocalMapping>> {
    let [a, b, c, d, e, f] = layer_to_destination.as_array();
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if coefficient_scale == 0.0 {
        return Ok(None);
    }
    let normalized_a = checked_div(
        a,
        coefficient_scale,
        "composition normalized affine coefficient a",
    )?;
    let normalized_b = checked_div(
        b,
        coefficient_scale,
        "composition normalized affine coefficient b",
    )?;
    let normalized_c = checked_div(
        c,
        coefficient_scale,
        "composition normalized affine coefficient c",
    )?;
    let normalized_d = checked_div(
        d,
        coefficient_scale,
        "composition normalized affine coefficient d",
    )?;
    let normalized_determinant = checked_sub(
        checked_mul(
            normalized_a,
            normalized_d,
            "composition normalized affine determinant ad",
        )?,
        checked_mul(
            normalized_b,
            normalized_c,
            "composition normalized affine determinant bc",
        )?,
        "composition normalized affine determinant",
    )?;
    if normalized_determinant == 0.0 {
        return Ok(None);
    }
    let inverse_denominator = checked_mul(
        coefficient_scale,
        normalized_determinant,
        "composition affine inverse denominator",
    )?;
    let inverse_a = checked_div(
        normalized_d,
        inverse_denominator,
        "composition inverse affine coefficient a",
    )?;
    let inverse_b = checked_div(
        -normalized_b,
        inverse_denominator,
        "composition inverse affine coefficient b",
    )?;
    let inverse_c = checked_div(
        -normalized_c,
        inverse_denominator,
        "composition inverse affine coefficient c",
    )?;
    let inverse_d = checked_div(
        normalized_a,
        inverse_denominator,
        "composition inverse affine coefficient d",
    )?;
    let inverse_e = checked_sub(
        0.0,
        checked_add(
            checked_mul(inverse_a, e, "composition inverse affine translation ae")?,
            checked_mul(inverse_c, f, "composition inverse affine translation cf")?,
            "composition inverse affine translation x",
        )?,
        "composition inverse affine translation x",
    )?;
    let inverse_f = checked_sub(
        0.0,
        checked_add(
            checked_mul(inverse_b, e, "composition inverse affine translation be")?,
            checked_mul(inverse_d, f, "composition inverse affine translation df")?,
            "composition inverse affine translation y",
        )?,
        "composition inverse affine translation y",
    )?;
    Ok(Some(DestinationToLayerLocalMapping {
        affine: Transform::try_new([
            inverse_a, inverse_b, inverse_c, inverse_d, inverse_e, inverse_f,
        ])?,
    }))
}

pub(super) fn mask_upload_spatial(
    image_dimensions: PhysicalSize,
) -> Result<NonEmptyFrameSpatialPlan> {
    let width = image_dimensions.width();
    let height = image_dimensions.height();
    let logical_bounds = non_empty_logical_bounds(
        Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
        "resolved mask image pixel bounds",
    )?;
    let device_origin = SignedDeviceOrigin::new(0, 0);
    let device_extent = PositiveDeviceExtent::try_new(width, height)?;
    let raster_scale = RasterScale::try_new(1.0)?;
    let texel_center_mapping = TexelCenterMapping::try_new(device_origin, raster_scale)?;
    Ok(NonEmptyFrameSpatialPlan {
        logical_bounds,
        device_origin,
        device_extent,
        raster_scale,
        texel_center_mapping,
    })
}

fn transform_point(transform: Transform, x: f64, y: f64, name: &str) -> Result<Point> {
    let [a, b, c, d, e, f] = transform.as_array();
    let transformed_x = checked_add(
        checked_add(
            checked_mul(a, x, &format!("{name} x product"))?,
            checked_mul(c, y, &format!("{name} x cross product"))?,
            &format!("{name} x linear value"),
        )?,
        e,
        &format!("{name} x"),
    )?;
    let transformed_y = checked_add(
        checked_add(
            checked_mul(b, x, &format!("{name} y cross product"))?,
            checked_mul(d, y, &format!("{name} y product"))?,
            &format!("{name} y linear value"),
        )?,
        f,
        &format!("{name} y"),
    )?;
    Point::try_new(transformed_x, transformed_y)
}

fn validate_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_value(name, value, "must be finite"));
    }
    Ok(())
}

fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::invalid_value(
            name,
            value,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

fn checked_add(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left + right, name, left, "+", right)
}

fn checked_sub(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left - right, name, left, "-", right)
}

pub(super) fn checked_mul(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left * right, name, left, "*", right)
}

pub(super) fn checked_div(left: f64, right: f64, name: &str) -> Result<f64> {
    checked_finite_result(left / right, name, left, "/", right)
}

fn checked_finite_result(
    result: f64,
    name: &str,
    left: f64,
    operation: &'static str,
    right: f64,
) -> Result<f64> {
    if !result.is_finite() {
        return Err(Error::invalid_value(
            name,
            format!("{left} {operation} {right}"),
            "must produce a finite value",
        ));
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) struct SpatialPrimitivesForTest {
    pub(crate) logical_and_device_phases_are_distinct: bool,
    pub(crate) logical_bounds: Option<[f64; 4]>,
    pub(crate) device_origin: Option<(i32, i32)>,
    pub(crate) device_extent: Option<(u32, u32)>,
    pub(crate) raster_scale: f64,
    pub(crate) texel_center: Option<(f64, f64)>,
    pub(crate) is_empty: bool,
}

#[cfg(test)]
pub(crate) fn transformed_logical_bounds_for_test(
    rect: Rect,
    transform: Transform,
) -> Result<[f64; 4]> {
    let transformed = LogicalBounds::try_from_rect(rect, "frame logical bounds")?
        .try_transform(transform, "frame transformed logical bounds")?
        .rect();
    Ok([
        transformed.x(),
        transformed.y(),
        transformed.width(),
        transformed.height(),
    ])
}

#[cfg(test)]
pub(crate) fn spatial_primitives_for_test(
    rect: Rect,
    transform: Transform,
    surface_scale: f64,
    texel: (u32, u32),
) -> Result<SpatialPrimitivesForTest> {
    let logical_bounds = LogicalBounds::try_from_rect(rect, "frame logical bounds")?;
    let plan = FrameContext::try_for_spatial_test(surface_scale)?
        .plan_local_bounds(logical_bounds, transform)?;
    let logical_rect = logical_bounds.rect();
    let logical_bounds = Some([
        logical_rect.x(),
        logical_rect.y(),
        logical_rect.width(),
        logical_rect.height(),
    ]);

    match plan {
        FrameSpatialPlan::Empty(plan) => {
            let _logical_bounds = plan.logical_bounds;
            Ok(SpatialPrimitivesForTest {
                logical_and_device_phases_are_distinct: true,
                logical_bounds,
                device_origin: None,
                device_extent: None,
                raster_scale: 0.0,
                texel_center: None,
                is_empty: true,
            })
        }
        FrameSpatialPlan::NonEmpty(plan) => {
            let _logical_bounds = plan.logical_bounds;
            let texel_center = plan.texel_center_mapping.point_for(texel.0, texel.1)?;
            Ok(SpatialPrimitivesForTest {
                logical_and_device_phases_are_distinct: true,
                logical_bounds,
                device_origin: Some((plan.device_origin.x, plan.device_origin.y)),
                device_extent: Some((plan.device_extent.width, plan.device_extent.height)),
                raster_scale: plan.raster_scale.get(),
                texel_center: Some((texel_center.x(), texel_center.y())),
                is_empty: false,
            })
        }
    }
}
