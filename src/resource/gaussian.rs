use super::{Error, Result};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum GaussianKernelSamplingForm {
    PairedLinear,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "C08 retains the validated non-filtering kernel route for exact sampling"
        )
    )]
    FullNearest,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GaussianKernelKey {
    standard_deviation_bits: u64,
    raster_scale_bits: u64,
    support_multiple_bits: u64,
    support_radius: u32,
    sampling_form: GaussianKernelSamplingForm,
}

impl GaussianKernelKey {
    pub(crate) const fn from_exact_plan(
        standard_deviation_bits: u64,
        raster_scale_bits: u64,
        support_multiple_bits: u64,
        support_radius: u32,
        sampling_form: GaussianKernelSamplingForm,
    ) -> Self {
        Self {
            standard_deviation_bits,
            raster_scale_bits,
            support_multiple_bits,
            support_radius,
            sampling_form,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GaussianKernelPlan {
    key: GaussianKernelKey,
    upload_bytes: Arc<[u8]>,
    byte_len: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GaussianKernelBufferLimits {
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
}

impl GaussianKernelBufferLimits {
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

impl GaussianKernelPlan {
    pub(crate) fn try_new(
        standard_deviation: f64,
        raster_scale: f64,
        support_multiple: f64,
        sampling_form: GaussianKernelSamplingForm,
    ) -> Result<Self> {
        let (device_standard_deviation, support_radius) =
            validate_gaussian_geometry(standard_deviation, raster_scale, support_multiple)?;
        let positive_weights =
            normalized_positive_gaussian_weights(device_standard_deviation, support_radius)?;
        let sample_count = gaussian_sample_count(support_radius, sampling_form)?;
        let upload_bytes = serialize_gaussian_samples(
            support_radius,
            sampling_form,
            sample_count,
            &positive_weights,
        )?;
        let byte_len = u64::try_from(upload_bytes.len()).map_err(|_| {
            Error::invalid_value(
                "Gaussian kernel byte length",
                upload_bytes.len(),
                "must fit in u64",
            )
        })?;
        Ok(Self {
            key: GaussianKernelKey::from_exact_plan(
                standard_deviation.to_bits(),
                raster_scale.to_bits(),
                support_multiple.to_bits(),
                support_radius,
                sampling_form,
            ),
            upload_bytes: upload_bytes.into(),
            byte_len,
        })
    }

    pub(crate) const fn key(&self) -> GaussianKernelKey {
        self.key
    }

    pub(crate) fn upload_bytes(&self) -> &[u8] {
        &self.upload_bytes
    }

    pub(crate) const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn validate_upload_byte_len(&self, actual_len: usize) -> Result<()> {
        if actual_len != self.upload_bytes.len() {
            return Err(Error::invalid_value(
                "Gaussian kernel upload byte length",
                actual_len,
                "must equal the exact serialized kernel plan length",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_buffer_limits(&self, limits: GaussianKernelBufferLimits) -> Result<()> {
        if self.byte_len == 0
            || self.byte_len > limits.max_buffer_size
            || self.byte_len > limits.max_storage_buffer_binding_size
        {
            return Err(Error::invalid_value(
                "Gaussian kernel buffer byte length",
                self.byte_len,
                "must be positive and no greater than the selected device buffer and storage-binding limits",
            ));
        }
        Ok(())
    }
}

fn validate_gaussian_geometry(
    standard_deviation: f64,
    raster_scale: f64,
    support_multiple: f64,
) -> Result<(f64, u32)> {
    for (field, value) in [
        ("Gaussian standard deviation", standard_deviation),
        ("Gaussian raster scale", raster_scale),
        ("Gaussian support multiple", support_multiple),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::invalid_value(
                field,
                value,
                "must be finite and greater than zero",
            ));
        }
    }
    let device_standard_deviation = standard_deviation * raster_scale;
    if !device_standard_deviation.is_finite() || device_standard_deviation <= 0.0 {
        return Err(Error::invalid_value(
            "Gaussian device standard deviation",
            device_standard_deviation,
            "must be finite and greater than zero",
        ));
    }
    let support = device_standard_deviation * support_multiple;
    if !support.is_finite() || support > f64::from(u32::MAX) {
        return Err(Error::invalid_value(
            "Gaussian support radius",
            support,
            "must be finite and fit in u32 device pixels",
        ));
    }
    Ok((device_standard_deviation, support.ceil() as u32))
}

fn normalized_positive_gaussian_weights(
    device_standard_deviation: f64,
    support_radius: u32,
) -> Result<Vec<f64>> {
    let weight_count = usize::try_from(support_radius)
        .ok()
        .and_then(|radius| radius.checked_add(1))
        .ok_or_else(|| {
            Error::invalid_value(
                "Gaussian kernel weight count",
                support_radius,
                "must fit addressable memory",
            )
        })?;
    let mut positive_weights = Vec::new();
    positive_weights
        .try_reserve_exact(weight_count)
        .map_err(|_| {
            Error::invalid_value(
                "Gaussian kernel weight count",
                weight_count,
                "must fit available addressable memory",
            )
        })?;
    for offset in 0..=support_radius {
        let offset = f64::from(offset);
        let ratio = offset / device_standard_deviation;
        positive_weights.push((-0.5 * ratio * ratio).exp());
    }
    let normalization = positive_weights[0] + 2.0 * positive_weights.iter().skip(1).sum::<f64>();
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(Error::invalid_value(
            "Gaussian kernel normalization",
            normalization,
            "must be finite and greater than zero",
        ));
    }
    for weight in &mut positive_weights {
        *weight /= normalization;
    }
    Ok(positive_weights)
}

fn gaussian_sample_count(
    support_radius: u32,
    sampling_form: GaussianKernelSamplingForm,
) -> Result<u32> {
    match sampling_form {
        GaussianKernelSamplingForm::FullNearest => support_radius
            .checked_mul(2)
            .and_then(|count| count.checked_add(1)),
        GaussianKernelSamplingForm::PairedLinear => support_radius
            .checked_add(1)
            .and_then(|count| count.checked_div(2))
            .and_then(|pairs| pairs.checked_mul(2))
            .and_then(|paired| paired.checked_add(1)),
    }
    .ok_or_else(|| {
        Error::invalid_value(
            "Gaussian kernel sample count",
            support_radius,
            "must fit in u32",
        )
    })
}

fn serialize_gaussian_samples(
    support_radius: u32,
    sampling_form: GaussianKernelSamplingForm,
    sample_count: u32,
    positive_weights: &[f64],
) -> Result<Vec<u8>> {
    let byte_capacity = usize::try_from(sample_count)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| {
            Error::invalid_value(
                "Gaussian kernel byte length",
                sample_count,
                "must fit addressable memory",
            )
        })?;
    let mut upload_bytes = Vec::new();
    upload_bytes.try_reserve_exact(byte_capacity).map_err(|_| {
        Error::invalid_value(
            "Gaussian kernel byte length",
            byte_capacity,
            "must fit available addressable memory",
        )
    })?;
    append_kernel_sample(&mut upload_bytes, 0.0, positive_weights[0])?;
    match sampling_form {
        GaussianKernelSamplingForm::FullNearest => {
            append_full_gaussian_samples(&mut upload_bytes, support_radius, positive_weights)?
        }
        GaussianKernelSamplingForm::PairedLinear => {
            append_paired_gaussian_samples(&mut upload_bytes, support_radius, positive_weights)?
        }
    }
    if upload_bytes.len() != byte_capacity {
        return Err(Error::invalid_value(
            "Gaussian kernel byte length",
            upload_bytes.len(),
            "must match the exact serialized sample count",
        ));
    }
    Ok(upload_bytes)
}

fn append_full_gaussian_samples(
    upload_bytes: &mut Vec<u8>,
    support_radius: u32,
    positive_weights: &[f64],
) -> Result<()> {
    for offset in 1..=support_radius {
        let offset_index =
            usize::try_from(offset).expect("validated u32 Gaussian offsets must fit usize");
        let weight = positive_weights[offset_index];
        append_kernel_sample(upload_bytes, f64::from(offset), weight)?;
        append_kernel_sample(upload_bytes, -f64::from(offset), weight)?;
    }
    Ok(())
}

fn append_paired_gaussian_samples(
    upload_bytes: &mut Vec<u8>,
    support_radius: u32,
    positive_weights: &[f64],
) -> Result<()> {
    let mut first = 1_u32;
    while first <= support_radius {
        let second = first
            .checked_add(1)
            .filter(|value| *value <= support_radius);
        let first_index =
            usize::try_from(first).expect("validated u32 Gaussian offsets must fit usize");
        let first_weight = positive_weights[first_index];
        let (offset, weight) =
            paired_gaussian_sample(first, second, first_weight, positive_weights)?;
        append_kernel_sample(upload_bytes, offset, weight)?;
        append_kernel_sample(upload_bytes, -offset, weight)?;
        let Some(next) = first.checked_add(2) else {
            break;
        };
        first = next;
    }
    Ok(())
}

fn paired_gaussian_sample(
    first: u32,
    second: Option<u32>,
    first_weight: f64,
    positive_weights: &[f64],
) -> Result<(f64, f64)> {
    let Some(second) = second else {
        return Ok((f64::from(first), first_weight));
    };
    let second_index =
        usize::try_from(second).expect("validated u32 Gaussian offsets must fit usize");
    let second_weight = positive_weights[second_index];
    let weight = first_weight + second_weight;
    if weight <= 0.0 {
        return Err(Error::invalid_value(
            "Gaussian paired sample weight",
            weight,
            "must remain greater than zero",
        ));
    }
    let offset = (f64::from(first) * first_weight + f64::from(second) * second_weight) / weight;
    Ok((offset, weight))
}

fn append_kernel_sample(bytes: &mut Vec<u8>, offset: f64, weight: f64) -> Result<()> {
    let offset = offset as f32;
    let weight = weight as f32;
    if !offset.is_finite() || !weight.is_finite() {
        return Err(Error::invalid_value(
            "Gaussian kernel sample",
            format!("offset {offset}, weight {weight}"),
            "must narrow to finite f32 values",
        ));
    }
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&weight.to_le_bytes());
    Ok(())
}
