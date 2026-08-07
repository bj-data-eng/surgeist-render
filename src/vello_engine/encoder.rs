// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use peniko::ImageFormat;

use crate::{BackendErrorCode, Error, PhysicalSize, Result, resource::ResourceManager};

use super::resources::{
    AbortedVelloResources, CleanVelloResourceRetention, ScopeResolvedVelloResourceLease,
    ScopeResolvedVelloResourceLeaseAggregate, VelloResourceLease, VelloResourceLeaseAggregate,
};
use super::{
    BufferRole, DispatchIntent, FineRasterVariant, RasterCommand, RasterKernel, RasterPhase,
    Recording, ResourceBinding, ResourceIntent,
};
use super::super::shaders::{CheckedShaderSet, CheckedWgpuScope};

#[must_use = "active internal Vello encoding scopes must be explicitly resolved"]
pub(crate) struct ActiveVelloEncodingScope<'a> {
    scope: CheckedWgpuScope<'a>,
}

impl<'a> ActiveVelloEncodingScope<'a> {
    pub(crate) fn begin(device: &'a wgpu::Device) -> Self {
        Self {
            scope: CheckedWgpuScope::begin(device),
        }
    }

    pub(super) fn device(&self) -> &wgpu::Device {
        self.scope.device()
    }

    pub(crate) async fn finish(self) -> Result<()> {
        self.scope
            .finish("checked internal Vello resource or command encoding failed")
            .await
    }

    pub(crate) async fn finish_with_lease(
        self,
        lease: VelloResourceLease,
    ) -> std::result::Result<ScopeResolvedVelloResourceLease, VelloEncodingFailure> {
        match self.finish().await {
            Ok(()) => Ok(lease.after_clean_scope()),
            Err(error) => Err(VelloEncodingFailure::after_encoding(
                error,
                lease.abort(),
            )),
        }
    }

    pub(crate) async fn finish_with_leases(
        self,
        leases: VelloResourceLeaseAggregate,
    ) -> std::result::Result<ScopeResolvedVelloResourceLeaseAggregate, VelloEncodingFailure> {
        match self.finish().await {
            Ok(()) => Ok(leases.after_clean_scope()),
            Err(error) => Err(VelloEncodingFailure::after_encoding(error, leases.abort())),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TransactionTargetIntent {
    extent: PhysicalSize,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

impl TransactionTargetIntent {
    pub(crate) const fn new(
        extent: PhysicalSize,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            extent,
            format,
            usage,
        }
    }
}

pub(crate) struct TransactionEncodingState<'state, 'device> {
    scope: &'state mut ActiveVelloEncodingScope<'device>,
    queue: &'state wgpu::Queue,
    command_encoder: &'state mut wgpu::CommandEncoder,
    target_view: &'state wgpu::TextureView,
    target: TransactionTargetIntent,
    clean_resource_retention: CleanVelloResourceRetention,
}

impl<'state, 'device> TransactionEncodingState<'state, 'device> {
    pub(crate) fn new(
        scope: &'state mut ActiveVelloEncodingScope<'device>,
        queue: &'state wgpu::Queue,
        command_encoder: &'state mut wgpu::CommandEncoder,
        target_view: &'state wgpu::TextureView,
        target: TransactionTargetIntent,
    ) -> Self {
        Self {
            scope,
            queue,
            command_encoder,
            target_view,
            target,
            clean_resource_retention: CleanVelloResourceRetention::DirectAtlasOnly,
        }
    }

    pub(crate) fn new_reusable_graph_capture(
        scope: &'state mut ActiveVelloEncodingScope<'device>,
        queue: &'state wgpu::Queue,
        command_encoder: &'state mut wgpu::CommandEncoder,
        target_view: &'state wgpu::TextureView,
        target: TransactionTargetIntent,
    ) -> Self {
        Self {
            scope,
            queue,
            command_encoder,
            target_view,
            target,
            clean_resource_retention: CleanVelloResourceRetention::ReusableGraphFrame,
        }
    }

    pub(crate) const fn target_extent(&self) -> PhysicalSize {
        self.target.extent
    }

    pub(crate) const fn target_format(&self) -> wgpu::TextureFormat {
        self.target.format
    }

    pub(crate) const fn target_usage(&self) -> wgpu::TextureUsages {
        self.target.usage
    }

    #[cfg(test)]
    pub(crate) fn target_view_identity_for_test(&self) -> usize {
        std::ptr::from_ref(self.target_view) as usize
    }

    pub(super) fn device(&self) -> &wgpu::Device {
        self.scope.device()
    }

    pub(super) fn active_scope(&self) -> &ActiveVelloEncodingScope<'device> {
        &*self.scope
    }

    fn preflight_target_limits(&self) -> Result<()> {
        let max_extent = self.device().limits().max_texture_dimension_2d;
        if self.target.extent.width() > max_extent || self.target.extent.height() > max_extent {
            return Err(render_failed(
                "internal Vello target extent exceeds the device 2D texture limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct VelloEncodingFailure {
    error: Error,
    aborted_resources: AbortedVelloResources,
}

impl VelloEncodingFailure {
    pub(crate) fn before_resource_allocation(error: Error) -> Self {
        Self {
            error,
            aborted_resources: AbortedVelloResources::without_resources(),
        }
    }

    pub(crate) fn after_encoding(error: Error, aborted_resources: AbortedVelloResources) -> Self {
        Self {
            error,
            aborted_resources,
        }
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Typed checked-encoding failures remain inspectable by the transaction route."
        )
    )]
    pub(crate) fn error(&self) -> &Error {
        &self.error
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "The typed abort outcome remains consumable by the resource manager."
        )
    )]
    pub(crate) fn into_aborted_resources(self) -> AbortedVelloResources {
        self.aborted_resources
    }

    pub(crate) fn into_error_and_aborted_resources(self) -> (Error, AbortedVelloResources) {
        (self.error, self.aborted_resources)
    }
}

pub(crate) struct VelloEngineState {
    shaders: CheckedShaderSet,
}

impl VelloEngineState {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Checked engine construction remains private to scene lowering."
        )
    )]
    pub(crate) async fn new_checked(device: &wgpu::Device) -> Result<Self> {
        Ok(Self {
            shaders: CheckedShaderSet::create(device).await?,
        })
    }
}

pub(crate) fn encode_recording(
    engine: &VelloEngineState,
    recording: &Recording,
    resource_intents: &[ResourceIntent],
    resources: &ResourceManager,
    state: &mut TransactionEncodingState<'_, '_>,
) -> std::result::Result<VelloResourceLease, VelloEncodingFailure> {
    log::trace!("encoding checked internal Vello raster recording");
    preflight_recording(recording, resource_intents, state)
        .map_err(VelloEncodingFailure::before_resource_allocation)?;
    let mut lease = VelloResourceLease::allocate_with_retention(
        state.active_scope(),
        resources,
        resource_intents,
        state.clean_resource_retention,
    )
    .map_err(VelloEncodingFailure::before_resource_allocation)?;
    let result = recording
        .commands
        .iter()
        .try_for_each(|command| encode_command(engine, command, state, &mut lease));
    match result {
        Ok(()) => Ok(lease),
        Err(error) => Err(VelloEncodingFailure::after_encoding(
            error,
            lease.abort(),
        )),
    }
}

fn preflight_recording(
    recording: &Recording,
    resource_intents: &[ResourceIntent],
    state: &TransactionEncodingState<'_, '_>,
) -> Result<()> {
    state.preflight_target_limits()?;
    let limits = state.device().limits();
    for command in &recording.commands {
        if let RasterCommand::Dispatch(dispatch) = command {
            preflight_dispatch(dispatch, resource_intents, &limits)?;
        }
    }
    Ok(())
}

fn preflight_dispatch(
    dispatch: &DispatchIntent,
    resource_intents: &[ResourceIntent],
    limits: &wgpu::Limits,
) -> Result<()> {
    let binding_count = u32::try_from(dispatch.bindings.len()).map_err(|_| {
        render_failed("internal Vello dispatch binding count does not fit the device limit type")
    })?;
    if binding_count > limits.max_bindings_per_bind_group {
        return Err(render_failed(
            "internal Vello dispatch exceeds the device bind-group binding limit",
        ));
    }

    if let Some(indirect) = &dispatch.indirect {
        preflight_indirect_dispatch(indirect, resource_intents)?;
        return Ok(());
    }

    let (x, y, z) = dispatch.workgroups;
    let max = limits.max_compute_workgroups_per_dimension;
    if x > max || y > max || z > max {
        return Err(render_failed(
            "internal Vello direct dispatch exceeds the device workgroup-dimension limit",
        ));
    }
    Ok(())
}

fn preflight_indirect_dispatch(
    indirect: &super::IndirectDispatch,
    resource_intents: &[ResourceIntent],
) -> Result<()> {
    let alignment = u64::from(std::mem::size_of::<u32>() as u32);
    if !indirect.offset.is_multiple_of(alignment) {
        return Err(render_failed(
            "internal Vello indirect dispatch offset is not aligned",
        ));
    }
    let parameter_bytes = alignment.checked_mul(3).ok_or_else(|| {
        render_failed("internal Vello indirect dispatch parameter size overflows")
    })?;
    let required_end = indirect.offset.checked_add(parameter_bytes).ok_or_else(|| {
        render_failed("internal Vello indirect dispatch offset overflows")
    })?;
    let buffer = resource_intents
        .iter()
        .find_map(|intent| match intent {
            ResourceIntent::Buffer(buffer) if buffer.resource == indirect.buffer => Some(buffer),
            ResourceIntent::Buffer(_) | ResourceIntent::Image(_) => None,
        })
        .ok_or_else(|| {
            render_failed("internal Vello indirect dispatch references an unknown buffer")
        })?;
    if buffer.role == BufferRole::Config {
        return Err(render_failed(
            "internal Vello indirect dispatch requires a storage-capable buffer",
        ));
    }
    if required_end > buffer.byte_len {
        return Err(render_failed(
            "internal Vello indirect dispatch exceeds its prepared allocation",
        ));
    }
    Ok(())
}

fn encode_command(
    engine: &VelloEngineState,
    command: &RasterCommand,
    state: &mut TransactionEncodingState<'_, '_>,
    lease: &mut VelloResourceLease,
) -> Result<()> {
    match command {
        RasterCommand::UploadScene { buffer, packed } => {
            let buffer = lease.buffer_for_upload(*buffer, packed.len())?;
            state.queue.write_buffer(buffer, 0, packed);
        }
        RasterCommand::UploadConfig { buffer, config } => {
            let bytes = bytemuck::bytes_of(config);
            let buffer = lease.buffer_for_upload(*buffer, bytes.len())?;
            state.queue.write_buffer(buffer, 0, bytes);
        }
        RasterCommand::UploadGradientRamps { image, ramps } => {
            let bytes = bytemuck::cast_slice(ramps.as_slice());
            write_rgba8_texture(state.queue, lease, *image, [0, 0], None, bytes)?;
        }
        RasterCommand::UploadMaskLut {
            buffer,
            variant,
            samples,
        } => {
            if matches!(variant, FineRasterVariant::Area) {
                return Err(render_failed(
                    "internal Vello recording supplies a mask lookup table for area rasterization",
                ));
            }
            let buffer = lease.buffer_for_upload(*buffer, samples.len())?;
            state.queue.write_buffer(buffer, 0, samples);
        }
        RasterCommand::WriteImage {
            image,
            origin,
            image_data,
        } => {
            if image_data.format != ImageFormat::Rgba8 {
                return Err(render_failed(
                    "internal Vello recording requires RGBA8 image data",
                ));
            }
            write_rgba8_texture(
                state.queue,
                lease,
                *image,
                *origin,
                Some((image_data.width, image_data.height)),
                image_data.data.data(),
            )?;
        }
        RasterCommand::ClearBuffer(buffer) => {
            state.command_encoder.clear_buffer(lease.buffer(*buffer)?, 0, None);
        }
        RasterCommand::Dispatch(dispatch) => encode_dispatch(engine, dispatch, state, lease)?,
        RasterCommand::Release(reference) => lease.record_release(*reference)?,
    }
    Ok(())
}

fn write_rgba8_texture(
    queue: &wgpu::Queue,
    lease: &VelloResourceLease,
    image: super::ImageHandle,
    origin: [u32; 2],
    requested_extent: Option<(u32, u32)>,
    bytes: &[u8],
) -> Result<()> {
    let target_extent = lease.image_extent(image)?;
    let (width, height) = requested_extent
        .unwrap_or((target_extent.width(), target_extent.height()));
    if width == 0 || height == 0 {
        return Err(render_failed(
            "internal Vello texture upload has an empty extent",
        ));
    }
    let end_x = origin[0].checked_add(width);
    let end_y = origin[1].checked_add(height);
    if end_x.is_none_or(|end| end > target_extent.width())
        || end_y.is_none_or(|end| end > target_extent.height())
    {
        return Err(render_failed(
            "internal Vello texture upload exceeds its prepared image",
        ));
    }
    let row_bytes = width.checked_mul(4).ok_or_else(|| {
        render_failed("internal Vello texture row length overflows")
    })?;
    let expected_len = usize::try_from(row_bytes)
        .ok()
        .and_then(|row_bytes| {
            usize::try_from(height)
                .ok()
                .and_then(|height| row_bytes.checked_mul(height))
        })
        .ok_or_else(|| render_failed("internal Vello texture upload length overflows"))?;
    if bytes.len() != expected_len {
        return Err(render_failed(
            "internal Vello texture upload does not match its prepared extent",
        ));
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: lease.image_texture(image)?,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: origin[0],
                y: origin[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn encode_dispatch(
    engine: &VelloEngineState,
    dispatch: &DispatchIntent,
    state: &mut TransactionEncodingState<'_, '_>,
    lease: &VelloResourceLease,
) -> Result<()> {
    if !phase_matches_kernel(dispatch.phase, dispatch.kernel) {
        return Err(render_failed(
            "internal Vello recording assigns a kernel to the wrong phase",
        ));
    }
    if dispatch.indirect.is_none()
        && (dispatch.workgroups.0 == 0 || dispatch.workgroups.1 == 0 || dispatch.workgroups.2 == 0)
    {
        return Ok(());
    }

    let pipeline = engine.shaders.pipeline(dispatch.kernel);
    if dispatch.bindings.len() != pipeline.binding_indices().len() {
        return Err(render_failed(
            "internal Vello recording does not match its shader binding layout",
        ));
    }
    let entries = dispatch
        .bindings
        .iter()
        .zip(pipeline.binding_indices().iter().copied())
        .map(|(binding, index)| binding_entry(binding, index, lease, state.target_view))
        .collect::<Result<Vec<_>>>()?;
    let bind_group = state.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Surgeist internal Vello dispatch bindings"),
        layout: pipeline.bind_group_layout(),
        entries: &entries,
    });
    let mut compute_pass = state
        .command_encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Surgeist internal Vello compute pass"),
            timestamp_writes: None,
        });
    compute_pass.set_pipeline(pipeline.pipeline());
    compute_pass.set_bind_group(0, &bind_group, &[]);
    if let Some(indirect) = &dispatch.indirect {
        compute_pass.dispatch_workgroups_indirect(
            lease.indirect_buffer(indirect.buffer, indirect.offset)?,
            indirect.offset,
        );
    } else {
        compute_pass.dispatch_workgroups(
            dispatch.workgroups.0,
            dispatch.workgroups.1,
            dispatch.workgroups.2,
        );
    }
    Ok(())
}

fn binding_entry<'a>(
    binding: &ResourceBinding,
    index: u32,
    lease: &'a VelloResourceLease,
    target_view: &'a wgpu::TextureView,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let resource = match binding {
        ResourceBinding::Buffer(buffer) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: lease.buffer(*buffer)?,
            offset: 0,
            size: None,
        }),
        ResourceBinding::Image(image) => wgpu::BindingResource::TextureView(lease.image_view(*image)?),
        ResourceBinding::TargetOutput => wgpu::BindingResource::TextureView(target_view),
    };
    Ok(wgpu::BindGroupEntry {
        binding: index,
        resource,
    })
}

const fn phase_matches_kernel(phase: RasterPhase, kernel: RasterKernel) -> bool {
    matches!(
        (phase, kernel),
        (RasterPhase::Fine, RasterKernel::FineArea)
            | (RasterPhase::Fine, RasterKernel::FineMsaa8)
            | (RasterPhase::Fine, RasterKernel::FineMsaa16)
            | (RasterPhase::Coarse, RasterKernel::PathTagReduce)
            | (RasterPhase::Coarse, RasterKernel::PathTagReduce2)
            | (RasterPhase::Coarse, RasterKernel::PathTagScan1)
            | (RasterPhase::Coarse, RasterKernel::PathTagScan)
            | (RasterPhase::Coarse, RasterKernel::PathTagScanLarge)
            | (RasterPhase::Coarse, RasterKernel::BboxClear)
            | (RasterPhase::Coarse, RasterKernel::Flatten)
            | (RasterPhase::Coarse, RasterKernel::DrawReduce)
            | (RasterPhase::Coarse, RasterKernel::DrawLeaf)
            | (RasterPhase::Coarse, RasterKernel::ClipReduce)
            | (RasterPhase::Coarse, RasterKernel::ClipLeaf)
            | (RasterPhase::Coarse, RasterKernel::Binning)
            | (RasterPhase::Coarse, RasterKernel::TileAlloc)
            | (RasterPhase::Coarse, RasterKernel::PathCountSetup)
            | (RasterPhase::Coarse, RasterKernel::PathCount)
            | (RasterPhase::Coarse, RasterKernel::Backdrop)
            | (RasterPhase::Coarse, RasterKernel::Coarse)
            | (RasterPhase::Coarse, RasterKernel::PathTilingSetup)
            | (RasterPhase::Coarse, RasterKernel::PathTiling)
    )
}

fn render_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}
