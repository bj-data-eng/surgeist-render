use std::collections::BTreeMap;

use super::{
    close::preparation_error,
    model::{
        RuntimeBlur, RuntimeComposite, RuntimeCompositeKind, RuntimePass, RuntimePassKind,
        RuntimeReadBinding, RuntimeReadRole, RuntimeResourceId, RuntimeResourceRequest,
        RuntimeResultBinding, RuntimeSamplingEdge, RuntimeSpatialDescriptor,
    },
};
use crate::{
    Result,
    shader::{
        BlurEdgeParameterBytes, ColorFilterOperationBufferLimits, ColorFilterOperationBytes,
        CompositeParameterBytes, DropShadowParameterBytes, PassSpatialUniformBytes,
    },
};

pub(super) fn prepare_blur_edge_parameters(
    pass: &RuntimePass,
) -> Result<Option<BlurEdgeParameterBytes>> {
    let blur = match &pass.kind {
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            blur
        }
        _ => return Ok(None),
    };
    match blur.edge {
        RuntimeSamplingEdge::SemanticBorderMirror(bounds) => {
            BlurEdgeParameterBytes::try_from_semantic_bounds(bounds).map(Some)
        }
        RuntimeSamplingEdge::TransparentBlack => Ok(None),
        RuntimeSamplingEdge::ClampToExtent => Err(preparation_error(
            "a Gaussian blur cannot use clamp-to-extent edge semantics",
        )),
    }
}

pub(super) fn prepare_drop_shadow_parameters(
    pass: &RuntimePass,
) -> Result<Option<DropShadowParameterBytes>> {
    let RuntimePassKind::DropShadowColorize(Some(shadow)) = &pass.kind else {
        return Ok(None);
    };
    let bytes = DropShadowParameterBytes::try_new(shadow.offset, shadow.color)?;
    if bytes.as_bytes().len() != 32 {
        return Err(preparation_error(
            "drop-shadow parameter serialization changed its exact WGSL byte length",
        ));
    }
    Ok(Some(bytes))
}

pub(super) fn prepare_color_filter_operations(
    pass: &RuntimePass,
    limits: ColorFilterOperationBufferLimits,
) -> Result<Option<ColorFilterOperationBytes>> {
    let RuntimePassKind::ColorFilter(Some(filter)) = &pass.kind else {
        return Ok(None);
    };
    let bytes = ColorFilterOperationBytes::try_from_runtime_operations_with_limits(
        filter.operations(),
        limits,
    )?;
    if bytes.as_bytes().is_empty() {
        return Err(preparation_error(
            "prepared color-filter operation bytes are empty",
        ));
    }
    Ok(Some(bytes))
}

pub(super) fn prepared_pass_spatial_uniform(
    pass: &RuntimePass,
    resources: &BTreeMap<RuntimeResourceId, &RuntimeResourceRequest>,
    root_working_image: RuntimeResourceId,
) -> Result<Option<PassSpatialUniformBytes>> {
    if pass.cache_keys.is_none() {
        return Ok(None);
    }
    let result_spatial = || -> Result<RuntimeSpatialDescriptor> {
        let RuntimeResultBinding::Resource(resource) = pass.result else {
            return Err(preparation_error(
                "custom pass has no concrete runtime result resource",
            ));
        };
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass result spatial binding is missing"))
    };
    let read_spatial = |role| -> Result<RuntimeSpatialDescriptor> {
        let resource = pass
            .reads
            .iter()
            .find(|read| read.role == role)
            .map(|read| read.resource)
            .ok_or_else(|| preparation_error("custom pass source spatial binding is missing"))?;
        resources
            .get(&resource)
            .map(|resource| resource.spatial)
            .ok_or_else(|| preparation_error("custom pass source resource is missing"))
    };

    let (source, destination) = match &pass.kind {
        RuntimePassKind::CanonicalizeCapture => (
            read_spatial(RuntimeReadRole::CaptureSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::CopyBackdrop => (
            read_spatial(RuntimeReadRole::CompletedParent)?,
            result_spatial()?,
        ),
        RuntimePassKind::ColorFilter(Some(filter)) => {
            (filter.spatial.source, filter.spatial.result)
        }
        RuntimePassKind::BlurHorizontal(Some(blur)) | RuntimePassKind::BlurVertical(Some(blur)) => {
            (blur.spatial.source, blur.spatial.result)
        }
        RuntimePassKind::DropShadowColorize(Some(shadow)) => {
            (shadow.spatial.source, shadow.spatial.result)
        }
        RuntimePassKind::Composite(Some(_)) => (
            read_spatial(RuntimeReadRole::CompositeSource)?,
            result_spatial()?,
        ),
        RuntimePassKind::Present => {
            let source = read_spatial(RuntimeReadRole::FinalWorkingImage)?;
            let destination = resources
                .get(&root_working_image)
                .map(|resource| resource.spatial)
                .ok_or_else(|| preparation_error("present destination spatial is missing"))?;
            (source, destination)
        }
        RuntimePassKind::ClearRoot { .. }
        | RuntimePassKind::VelloCapture(_)
        | RuntimePassKind::ColorFilter(None)
        | RuntimePassKind::BlurHorizontal(None)
        | RuntimePassKind::BlurVertical(None)
        | RuntimePassKind::DropShadowColorize(None)
        | RuntimePassKind::Composite(None) => {
            return Err(preparation_error(
                "non-executable pass unexpectedly requested spatial serialization",
            ));
        }
    };
    PassSpatialUniformBytes::try_from_runtime_spatial_descriptors(source, destination).map(Some)
}

pub(super) fn prepared_pass_composite_parameters(
    pass: &RuntimePass,
) -> Result<Option<CompositeParameterBytes>> {
    match &pass.kind {
        RuntimePassKind::Composite(Some(RuntimeComposite {
            kind: RuntimeCompositeKind::Layer { parameters, .. },
            ..
        })) => {
            let bytes = CompositeParameterBytes::try_from_runtime_layer(parameters)?;
            if bytes.as_bytes().len() != 112 {
                return Err(preparation_error(
                    "composite parameter serialization changed its exact WGSL byte length",
                ));
            }
            Ok(Some(bytes))
        }
        _ => Ok(None),
    }
}

pub(super) fn c12_blur_edge_uniform_bytes(
    blur: &RuntimeBlur,
    source: &RuntimeReadBinding,
    prepared_parameters: Option<&BlurEdgeParameterBytes>,
) -> Result<Option<[u8; 16]>> {
    let RuntimeSamplingEdge::SemanticBorderMirror(bounds) = blur.edge else {
        if blur.edge != RuntimeSamplingEdge::TransparentBlack
            || source.sampling_edge() != RuntimeSamplingEdge::TransparentBlack
            || prepared_parameters.is_some()
        {
            return Err(preparation_error(
                "the C11 transparent blur changed its checked edge contract",
            ));
        }
        return Ok(None);
    };
    let expected = BlurEdgeParameterBytes::try_from_semantic_bounds(bounds)?;
    if source.sampling_edge() != blur.edge || prepared_parameters != Some(&expected) {
        return Err(preparation_error(
            "the C12 mirrored blur changed its checked semantic edge",
        ));
    }
    let values = [
        bounds.x() as f32,
        bounds.y() as f32,
        (bounds.x() + bounds.width()) as f32,
        (bounds.y() + bounds.height()) as f32,
    ];
    let mut bytes = [0_u8; 16];
    for (index, value) in values.into_iter().enumerate() {
        let offset = index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(Some(bytes))
}
