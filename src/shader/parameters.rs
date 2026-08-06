use super::key::ShaderMaskSamplingKey;
use crate::{
    Color, Error, Point, Rect, Result,
    layer::BlendMode,
    pass::{
        RuntimeColorClampBoundary, RuntimeColorOperation, RuntimeColorOperationKind,
        RuntimeLayerCompositeParameters, RuntimeSpatialDescriptor,
    },
};

#[cfg(all(test, not(target_arch = "wasm32")))]
use crate::image::{Extend, ImageQuality};

const COLOR_FILTER_OPERATION_HEADER_BYTE_LEN: u64 = 16;
const COLOR_FILTER_OPERATION_RECORD_BYTE_LEN: u64 = 32;

/// Exact 32-byte WGSL drop-shadow parameter block.
///
/// Bytes `0..8` retain the continuous logical offset, bytes `8..16` are zero
/// alignment, and bytes `16..32` contain the finite solid premultiplied color.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DropShadowParameterBytes([u8; 32]);

impl DropShadowParameterBytes {
    pub(crate) fn try_new(offset: Point, color: Color) -> Result<Self> {
        let offset_x = narrow_drop_shadow_scalar("drop shadow offset x", offset.x())?;
        let offset_y = narrow_drop_shadow_scalar("drop shadow offset y", offset.y())?;
        let channels = [color.r(), color.g(), color.b(), color.a()];
        if channels
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(drop_shadow_parameter_error(
                "drop shadow solid color",
                "must contain finite unit-interval channels",
            ));
        }
        let premultiplied = [
            color.r() * color.a(),
            color.g() * color.a(),
            color.b() * color.a(),
            color.a(),
        ];
        let mut bytes = [0_u8; 32];
        bytes[0..4].copy_from_slice(&offset_x.to_le_bytes());
        bytes[4..8].copy_from_slice(&offset_y.to_le_bytes());
        for (index, value) in premultiplied.into_iter().enumerate() {
            bytes[16 + index * 4..20 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self(bytes))
    }

    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[cfg(test)]
pub(crate) fn drop_shadow_parameter_bytes_for_test(
    offset: Point,
    color: Color,
) -> Result<[u8; 32]> {
    DropShadowParameterBytes::try_new(offset, color).map(|bytes| bytes.0)
}

fn narrow_drop_shadow_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(drop_shadow_parameter_error(
            field,
            "must remain finite after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn drop_shadow_parameter_error(field: &'static str, invariant: &'static str) -> Error {
    Error::invalid_value(field, "runtime drop shadow", invariant)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBufferLimits {
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
}

impl ColorFilterOperationBufferLimits {
    #[must_use]
    pub(crate) fn from_device_limits(limits: &wgpu::Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn for_test(
        max_buffer_size: u64,
        max_storage_buffer_binding_size: u64,
    ) -> Self {
        Self {
            max_buffer_size,
            max_storage_buffer_binding_size,
        }
    }
}

/// Exact checked storage-buffer bytes for one ordered color-filter run.
///
/// Bytes `0..16` are the operation count followed by three zero `u32` pads.
/// Every following 32-byte record stores tag, zero flag, exponent, one zero
/// alignment word, a four-scalar finite payload, and no other state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ColorFilterOperationBytes {
    bytes: Vec<u8>,
}

impl ColorFilterOperationBytes {
    pub(crate) fn try_from_runtime_operations_with_limits(
        operations: &[RuntimeColorOperation],
        limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        let operation_count = u32::try_from(operations.len())
            .map_err(|_| color_filter_operation_count_error(operations.len()))?;
        let byte_len = checked_color_filter_operation_byte_len(operation_count, limits)?;
        let byte_len = usize::try_from(byte_len).map_err(|_| {
            color_filter_operation_byte_len_error(
                byte_len,
                "must fit the host address space after device-limit validation",
            )
        })?;

        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_len).map_err(|_| {
            color_filter_operation_byte_len_error(
                byte_len,
                "must fit available host serialization memory",
            )
        })?;
        bytes.resize(byte_len, 0);
        bytes[0..4].copy_from_slice(&operation_count.to_le_bytes());

        for (index, operation) in operations.iter().copied().enumerate() {
            let record = serialize_color_filter_operation(operation)?;
            let offset = 16 + index * 32;
            bytes[offset..offset + 32].copy_from_slice(&record);
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[cfg(test)]
    pub(crate) fn try_from_runtime_operations_for_test(
        operations: &[RuntimeColorOperation],
        limits: ColorFilterOperationBufferLimits,
    ) -> Result<Self> {
        Self::try_from_runtime_operations_with_limits(operations, limits)
    }
}

fn checked_color_filter_operation_byte_len(
    operation_count: u32,
    limits: ColorFilterOperationBufferLimits,
) -> Result<u64> {
    let records = COLOR_FILTER_OPERATION_RECORD_BYTE_LEN
        .checked_mul(u64::from(operation_count))
        .ok_or_else(|| {
            color_filter_operation_byte_len_error(
                operation_count,
                "must produce a checked u64 record byte length",
            )
        })?;
    let byte_len = COLOR_FILTER_OPERATION_HEADER_BYTE_LEN
        .checked_add(records)
        .ok_or_else(|| {
            color_filter_operation_byte_len_error(
                operation_count,
                "must produce a checked u64 header-plus-record byte length",
            )
        })?;
    if byte_len > limits.max_buffer_size || byte_len > limits.max_storage_buffer_binding_size {
        return Err(color_filter_operation_byte_len_error(
            byte_len,
            "must not exceed max_buffer_size or max_storage_buffer_binding_size",
        ));
    }
    Ok(byte_len)
}

fn serialize_color_filter_operation(operation: RuntimeColorOperation) -> Result<[u8; 32]> {
    if operation.clamp_boundary()
        != RuntimeColorClampBoundary::ClampStraightRgbaToUnitThenPremultiply
    {
        return Err(Error::invalid_value(
            "color filter operation clamp boundary",
            "unsupported runtime boundary",
            "must clamp straight RGBA and premultiply after every operation",
        ));
    }

    let (tag, zero, exponent, payload) = match operation.operation() {
        RuntimeColorOperationKind::Brightness(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (0_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Contrast(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (1_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Grayscale(amount) => (
            2_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::HueRotate(angle) => {
            let sine = checked_color_filter_payload(angle.sine())?;
            let cosine = checked_color_filter_payload(angle.cosine())?;
            (3_u32, 0, 0, [sine, cosine, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Invert(amount) => (
            4_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::Opacity(amount) => (
            5_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
        RuntimeColorOperationKind::Saturate(amount) => {
            let (zero, exponent, mantissa) = checked_runtime_amount_parts(amount)?;
            (6_u32, zero, exponent, [mantissa, 0.0, 0.0, 0.0])
        }
        RuntimeColorOperationKind::Sepia(amount) => (
            7_u32,
            0,
            0,
            [checked_runtime_unit(amount.value())?, 0.0, 0.0, 0.0],
        ),
    };

    let mut bytes = [0_u8; 32];
    bytes[0..4].copy_from_slice(&tag.to_le_bytes());
    bytes[4..8].copy_from_slice(&zero.to_le_bytes());
    bytes[8..12].copy_from_slice(&exponent.to_le_bytes());
    for (index, value) in payload.into_iter().enumerate() {
        let offset = 16 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn checked_runtime_amount_parts(
    amount: crate::filter::RuntimeFilterAmount,
) -> Result<(u32, i32, f32)> {
    if amount.zero() {
        if amount.mantissa() != 0.0 || amount.exponent() != 0 {
            return Err(color_filter_operation_scalar_error(
                "zero amount must carry zero mantissa and exponent",
            ));
        }
        return Ok((1, 0, 0.0));
    }
    let mantissa = checked_color_filter_payload(amount.mantissa())?;
    if !(0.5..1.0).contains(&mantissa) {
        return Err(color_filter_operation_scalar_error(
            "nonzero amount mantissa must be normalized to [0.5, 1)",
        ));
    }
    Ok((0, amount.exponent(), mantissa))
}

fn checked_runtime_unit(value: f32) -> Result<f32> {
    let value = checked_color_filter_payload(value)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(color_filter_operation_scalar_error(
            "unit amount must remain in the inclusive unit interval",
        ));
    }
    Ok(value)
}

fn checked_color_filter_payload(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(color_filter_operation_scalar_error(
            "every scalar payload must be finite",
        ));
    }
    Ok(value)
}

fn color_filter_operation_count_error(value: impl std::fmt::Display) -> Error {
    Error::invalid_value(
        "color filter operation count",
        value,
        "must fit in u32 before operation-buffer byte calculation",
    )
}

fn color_filter_operation_byte_len_error(
    value: impl std::fmt::Display,
    invariant: &'static str,
) -> Error {
    Error::invalid_value(
        "color filter operation buffer byte length",
        value,
        invariant,
    )
}

fn color_filter_operation_scalar_error(invariant: &'static str) -> Error {
    Error::invalid_value(
        "color filter operation scalar payload",
        "runtime operation",
        invariant,
    )
}

#[cfg(test)]
pub(crate) fn color_filter_operation_byte_len_for_test(
    operation_count: u64,
    limits: ColorFilterOperationBufferLimits,
) -> Result<u64> {
    let operation_count = u32::try_from(operation_count)
        .map_err(|_| color_filter_operation_count_error(operation_count))?;
    checked_color_filter_operation_byte_len(operation_count, limits)
}

/// Exact 48-byte WGSL spatial uniform with explicit little-endian encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PassSpatialUniformBytes([u8; 48]);

impl PassSpatialUniformBytes {
    pub(crate) fn try_from_runtime_spatial_descriptors(
        source: RuntimeSpatialDescriptor,
        destination: RuntimeSpatialDescriptor,
    ) -> Result<Self> {
        let source_origin = source.texel_origin();
        let source_origin_x =
            narrow_spatial_scalar("pass spatial source origin x", source_origin.x())?;
        let source_origin_y =
            narrow_spatial_scalar("pass spatial source origin y", source_origin.y())?;
        let source_raster_scale =
            narrow_raster_scale("pass spatial source raster scale", source.raster_scale())?;

        let destination_origin = destination.texel_origin();
        let destination_origin_x =
            narrow_spatial_scalar("pass spatial destination origin x", destination_origin.x())?;
        let destination_origin_y =
            narrow_spatial_scalar("pass spatial destination origin y", destination_origin.y())?;
        let destination_raster_scale = narrow_raster_scale(
            "pass spatial destination raster scale",
            destination.raster_scale(),
        )?;

        let source_extent = source.device_extent();
        let destination_extent = destination.device_extent();
        let mut bytes = [0_u8; 48];
        bytes[0..4].copy_from_slice(&source_origin_x.to_le_bytes());
        bytes[4..8].copy_from_slice(&source_origin_y.to_le_bytes());
        bytes[8..12].copy_from_slice(&source_raster_scale.to_le_bytes());
        bytes[16..20].copy_from_slice(&destination_origin_x.to_le_bytes());
        bytes[20..24].copy_from_slice(&destination_origin_y.to_le_bytes());
        bytes[24..28].copy_from_slice(&destination_raster_scale.to_le_bytes());
        bytes[32..36].copy_from_slice(&source_extent.width().to_le_bytes());
        bytes[36..40].copy_from_slice(&source_extent.height().to_le_bytes());
        bytes[40..44].copy_from_slice(&destination_extent.width().to_le_bytes());
        bytes[44..48].copy_from_slice(&destination_extent.height().to_le_bytes());
        Ok(Self(bytes))
    }

    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 48] {
        &self.0
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn into_bytes_for_test(self) -> [u8; 48] {
        self.0
    }
}

/// Exact semantic backdrop mirror rectangle in logical coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BlurEdgeParameterBytes([u8; 16]);

impl BlurEdgeParameterBytes {
    pub(crate) fn try_from_semantic_bounds(bounds: Rect) -> Result<Self> {
        let minimum_x = narrow_spatial_scalar("blur semantic mirror minimum x", bounds.x())?;
        let minimum_y = narrow_spatial_scalar("blur semantic mirror minimum y", bounds.y())?;
        let maximum_x = narrow_spatial_scalar(
            "blur semantic mirror maximum x",
            bounds.x() + bounds.width(),
        )?;
        let maximum_y = narrow_spatial_scalar(
            "blur semantic mirror maximum y",
            bounds.y() + bounds.height(),
        )?;
        if maximum_x <= minimum_x || maximum_y <= minimum_y {
            return Err(Error::invalid_value(
                "blur semantic mirror bounds",
                format!("{bounds:?}"),
                "must narrow to a finite positive logical rectangle",
            ));
        }
        let mut bytes = [0_u8; 16];
        for (index, value) in [minimum_x, minimum_y, maximum_x, maximum_y]
            .into_iter()
            .enumerate()
        {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(Self(bytes))
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Exact 112-byte WGSL composite parameter block.
///
/// The byte ranges are fixed as follows: affine linear coefficients `0..16`,
/// affine translation plus zero alignment bytes `16..32`, mask rectangle
/// `32..48`, image dimensions plus zero alignment bytes `48..64`, normalized
/// texel-center facts `64..80`, opacity/blend/quality/extend `80..96`, and
/// exact clip/mask presence plus zero alignment bytes `96..112`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompositeParameterBytes([u8; 112]);

#[derive(Clone, Copy, Debug, PartialEq)]
struct CompositeMaskParameterFacts {
    bounds: [f64; 4],
    dimensions: [u32; 2],
    texel_center_facts: [f64; 4],
    sampling: ShaderMaskSamplingKey,
}

impl CompositeParameterBytes {
    pub(crate) fn try_from_runtime_layer(
        parameters: &RuntimeLayerCompositeParameters,
    ) -> Result<Self> {
        let mask = parameters.alpha_mask().map(|mask| {
            let bounds = mask.bounds();
            let dimensions = mask.image_dimensions();
            let texel_centers = mask.texel_center_facts();
            let [half_x, half_y] = texel_centers.half_texel_normalized();
            let [texel_x, texel_y] = texel_centers.texel_size_normalized();
            CompositeMaskParameterFacts {
                bounds: [bounds.x(), bounds.y(), bounds.width(), bounds.height()],
                dimensions: [dimensions.width(), dimensions.height()],
                texel_center_facts: [half_x, half_y, texel_x, texel_y],
                sampling: mask.sampling(),
            }
        });
        Self::try_from_facts(
            parameters.destination_to_layer_local().affine().as_array(),
            parameters.opacity(),
            parameters.blend(),
            parameters.has_clip(),
            mask,
        )
    }

    fn try_from_facts(
        affine: [f64; 6],
        opacity: f32,
        blend: BlendMode,
        has_clip: bool,
        mask: Option<CompositeMaskParameterFacts>,
    ) -> Result<Self> {
        let [a, b, c, d, e, f] = affine;
        let affine = [
            narrow_composite_scalar("composite affine coefficient a", a)?,
            narrow_composite_scalar("composite affine coefficient b", b)?,
            narrow_composite_scalar("composite affine coefficient c", c)?,
            narrow_composite_scalar("composite affine coefficient d", d)?,
            narrow_composite_scalar("composite affine translation x", e)?,
            narrow_composite_scalar("composite affine translation y", f)?,
        ];
        validate_narrowed_composite_affine(affine)?;

        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(Error::invalid_value(
                "composite opacity",
                opacity,
                "must be finite and clamped to the inclusive unit interval",
            ));
        }

        let mut bytes = [0_u8; 112];
        write_f32(&mut bytes, 0, affine[0]);
        write_f32(&mut bytes, 4, affine[1]);
        write_f32(&mut bytes, 8, affine[2]);
        write_f32(&mut bytes, 12, affine[3]);
        write_f32(&mut bytes, 16, affine[4]);
        write_f32(&mut bytes, 20, affine[5]);

        if let Some(mask) = mask {
            let bounds = [
                narrow_composite_scalar("composite mask bounds x", mask.bounds[0])?,
                narrow_composite_scalar("composite mask bounds y", mask.bounds[1])?,
                narrow_positive_composite_scalar("composite mask bounds width", mask.bounds[2])?,
                narrow_positive_composite_scalar("composite mask bounds height", mask.bounds[3])?,
            ];
            for (index, value) in bounds.into_iter().enumerate() {
                write_f32(&mut bytes, 32 + index * 4, value);
            }

            let [width, height] = mask.dimensions;
            if width == 0 || height == 0 {
                return Err(Error::invalid_value(
                    "composite mask image dimensions",
                    format!("{width}x{height}"),
                    "must be positive before parameter serialization",
                ));
            }
            write_u32(&mut bytes, 48, width);
            write_u32(&mut bytes, 52, height);

            for (index, value) in mask.texel_center_facts.into_iter().enumerate() {
                write_f32(
                    &mut bytes,
                    64 + index * 4,
                    narrow_positive_composite_scalar("composite mask texel fact", value)?,
                );
            }

            let sampling = mask.sampling;
            write_u32(&mut bytes, 88, sampling.quality().parameter_code());
            write_u32(&mut bytes, 92, sampling.extend().parameter_code());
            write_u32(&mut bytes, 100, 1);
        }

        write_f32(&mut bytes, 80, opacity);
        write_u32(&mut bytes, 84, blend_parameter_code(blend));
        write_u32(&mut bytes, 96, u32::from(has_clip));
        Ok(Self(bytes))
    }

    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; 112] {
        &self.0
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn into_bytes_for_test(self) -> [u8; 112] {
        self.0
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) struct CompositeParameterGpuVectorFactsForTest {
    pub(crate) layer_point: [f64; 2],
    pub(crate) mask_bounds: [f64; 4],
    pub(crate) mask_dimensions: [u32; 2],
    pub(crate) quality: ImageQuality,
    pub(crate) extend: Extend,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
    pub(crate) has_clip: bool,
    pub(crate) has_mask: bool,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn composite_parameter_bytes_for_gpu_vector_for_test(
    facts: CompositeParameterGpuVectorFactsForTest,
) -> Result<[u8; 112]> {
    if !facts.opacity.is_finite() {
        return Err(Error::invalid_value(
            "composite opacity",
            facts.opacity,
            "must be finite before clamping",
        ));
    }
    let mask = facts.has_mask.then(|| {
        let [width, height] = facts.mask_dimensions;
        let texel_x = 1.0 / f64::from(width);
        let texel_y = 1.0 / f64::from(height);
        CompositeMaskParameterFacts {
            bounds: facts.mask_bounds,
            dimensions: facts.mask_dimensions,
            texel_center_facts: [texel_x * 0.5, texel_y * 0.5, texel_x, texel_y],
            sampling: ShaderMaskSamplingKey::new(facts.quality, facts.extend),
        }
    });
    CompositeParameterBytes::try_from_facts(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            facts.layer_point[0] - 0.5,
            facts.layer_point[1] - 0.5,
        ],
        facts.opacity.clamp(0.0, 1.0),
        facts.blend,
        facts.has_clip,
        mask,
    )
    .map(CompositeParameterBytes::into_bytes_for_test)
}

fn validate_narrowed_composite_affine(affine: [f32; 6]) -> Result<()> {
    let scale = affine[0]
        .abs()
        .max(affine[1].abs())
        .max(affine[2].abs())
        .max(affine[3].abs());
    if scale == 0.0 {
        return Err(Error::invalid_value(
            "composite affine mapping",
            "zero linear transform",
            "must remain non-singular after f64-to-f32 narrowing",
        ));
    }
    let a = affine[0] / scale;
    let b = affine[1] / scale;
    let c = affine[2] / scale;
    let d = affine[3] / scale;
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(Error::invalid_value(
            "composite affine mapping",
            determinant,
            "must remain finite and non-singular after f64-to-f32 narrowing",
        ));
    }
    Ok(())
}

const fn blend_parameter_code(blend: BlendMode) -> u32 {
    match blend {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Screen => 2,
        BlendMode::Overlay => 3,
        BlendMode::Darken => 4,
        BlendMode::Lighten => 5,
        BlendMode::Plus => 6,
    }
}

fn write_f32(bytes: &mut [u8; 112], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8; 112], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn narrow_composite_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain finite after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn narrow_positive_composite_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = narrow_composite_scalar(field, value)?;
    if narrowed <= 0.0 {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain strictly positive after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn narrow_spatial_scalar(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain finite after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}

fn narrow_raster_scale(field: &'static str, value: f64) -> Result<f32> {
    let narrowed = narrow_spatial_scalar(field, value)?;
    if narrowed <= 0.0 {
        return Err(Error::invalid_value(
            field,
            value,
            "must remain strictly positive after f64-to-f32 narrowing",
        ));
    }
    Ok(narrowed)
}
