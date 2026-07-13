// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashSet;

use vello_shaders::{BindType, ComputeShader, SHADERS};

use crate::{BackendErrorCode, Error, Result};

use super::recording::RasterKernel;

#[must_use = "checked WGPU scopes must be explicitly resolved"]
pub(super) struct CheckedWgpuScope<'a> {
    device: &'a wgpu::Device,
    validation: Option<wgpu::ErrorScopeGuard>,
    out_of_memory: Option<wgpu::ErrorScopeGuard>,
    internal: Option<wgpu::ErrorScopeGuard>,
}

impl<'a> CheckedWgpuScope<'a> {
    pub(super) fn begin(device: &'a wgpu::Device) -> Self {
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            device,
            validation: Some(validation),
            out_of_memory: Some(out_of_memory),
            internal: Some(internal),
        }
    }

    pub(super) fn device(&self) -> &wgpu::Device {
        self.device
    }

    pub(super) async fn finish(mut self, failure_message: &'static str) -> Result<()> {
        let validation = match self.validation.take() {
            Some(scope) => scope,
            None => {
                return Err(render_failed(
                    "internal Vello validation scope was already resolved",
                ));
            }
        };
        let out_of_memory = match self.out_of_memory.take() {
            Some(scope) => scope,
            None => {
                return Err(render_failed(
                    "internal Vello out-of-memory scope was already resolved",
                ));
            }
        };
        let internal = match self.internal.take() {
            Some(scope) => scope,
            None => {
                return Err(render_failed(
                    "internal Vello internal-error scope was already resolved",
                ));
            }
        };

        let validation_error = validation.pop().await;
        let memory_error = out_of_memory.pop().await;
        let internal_error = internal.pop().await;
        if let Some(source) = validation_error.or(memory_error).or(internal_error) {
            return Err(classify_checked_wgpu_error(failure_message, source));
        }
        Ok(())
    }
}

impl Drop for CheckedWgpuScope<'_> {
    fn drop(&mut self) {
        drop(self.validation.take());
        drop(self.out_of_memory.take());
        drop(self.internal.take());
    }
}

pub(super) struct CheckedComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    binding_indices: Vec<u32>,
}

impl CheckedComputePipeline {
    fn create(device: &wgpu::Device, shader: &ComputeShader<'_>) -> Result<Self> {
        if shader.bindings.len() != shader.wgsl.binding_indices.len() {
            return Err(render_failed(
                "internal Vello shader metadata has inconsistent binding indices",
            ));
        }

        let binding_indices = shader
            .wgsl
            .binding_indices
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let mut seen_indices = HashSet::new();
        if !binding_indices
            .iter()
            .copied()
            .all(|index| seen_indices.insert(index))
        {
            return Err(render_failed(
                "internal Vello shader metadata repeats a binding index",
            ));
        }

        let entries = shader
            .bindings
            .iter()
            .copied()
            .zip(binding_indices.iter().copied())
            .map(|(binding, index)| wgpu::BindGroupLayoutEntry {
                binding: index,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: binding_type(binding),
                count: None,
            })
            .collect::<Vec<_>>();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader.name.as_ref()),
            source: wgpu::ShaderSource::Wgsl(shader.wgsl.code.clone()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Surgeist internal Vello bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Surgeist internal Vello compute layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(shader.name.as_ref()),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            binding_indices,
        })
    }

    pub(super) const fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    pub(super) const fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub(super) fn binding_indices(&self) -> &[u32] {
        &self.binding_indices
    }
}

pub(super) struct CheckedShaderSet {
    pathtag_reduce: CheckedComputePipeline,
    pathtag_reduce2: CheckedComputePipeline,
    pathtag_scan1: CheckedComputePipeline,
    pathtag_scan: CheckedComputePipeline,
    pathtag_scan_large: CheckedComputePipeline,
    bbox_clear: CheckedComputePipeline,
    flatten: CheckedComputePipeline,
    draw_reduce: CheckedComputePipeline,
    draw_leaf: CheckedComputePipeline,
    clip_reduce: CheckedComputePipeline,
    clip_leaf: CheckedComputePipeline,
    binning: CheckedComputePipeline,
    tile_alloc: CheckedComputePipeline,
    path_count_setup: CheckedComputePipeline,
    path_count: CheckedComputePipeline,
    backdrop: CheckedComputePipeline,
    coarse: CheckedComputePipeline,
    path_tiling_setup: CheckedComputePipeline,
    path_tiling: CheckedComputePipeline,
    fine_area: CheckedComputePipeline,
    fine_msaa8: CheckedComputePipeline,
    fine_msaa16: CheckedComputePipeline,
}

impl CheckedShaderSet {
    pub(super) async fn create(device: &wgpu::Device) -> Result<Self> {
        checked_wgpu_build(device, || {
            Ok(Self {
                pathtag_reduce: CheckedComputePipeline::create(device, &SHADERS.pathtag_reduce)?,
                pathtag_reduce2: CheckedComputePipeline::create(device, &SHADERS.pathtag_reduce2)?,
                pathtag_scan1: CheckedComputePipeline::create(device, &SHADERS.pathtag_scan1)?,
                pathtag_scan: CheckedComputePipeline::create(device, &SHADERS.pathtag_scan_small)?,
                pathtag_scan_large: CheckedComputePipeline::create(
                    device,
                    &SHADERS.pathtag_scan_large,
                )?,
                bbox_clear: CheckedComputePipeline::create(device, &SHADERS.bbox_clear)?,
                flatten: CheckedComputePipeline::create(device, &SHADERS.flatten)?,
                draw_reduce: CheckedComputePipeline::create(device, &SHADERS.draw_reduce)?,
                draw_leaf: CheckedComputePipeline::create(device, &SHADERS.draw_leaf)?,
                clip_reduce: CheckedComputePipeline::create(device, &SHADERS.clip_reduce)?,
                clip_leaf: CheckedComputePipeline::create(device, &SHADERS.clip_leaf)?,
                binning: CheckedComputePipeline::create(device, &SHADERS.binning)?,
                tile_alloc: CheckedComputePipeline::create(device, &SHADERS.tile_alloc)?,
                path_count_setup: CheckedComputePipeline::create(
                    device,
                    &SHADERS.path_count_setup,
                )?,
                path_count: CheckedComputePipeline::create(device, &SHADERS.path_count)?,
                backdrop: CheckedComputePipeline::create(device, &SHADERS.backdrop_dyn)?,
                coarse: CheckedComputePipeline::create(device, &SHADERS.coarse)?,
                path_tiling_setup: CheckedComputePipeline::create(
                    device,
                    &SHADERS.path_tiling_setup,
                )?,
                path_tiling: CheckedComputePipeline::create(device, &SHADERS.path_tiling)?,
                fine_area: CheckedComputePipeline::create(device, &SHADERS.fine_area)?,
                fine_msaa8: CheckedComputePipeline::create(device, &SHADERS.fine_msaa8)?,
                fine_msaa16: CheckedComputePipeline::create(device, &SHADERS.fine_msaa16)?,
            })
        })
        .await
    }

    pub(super) const fn pipeline(&self, kernel: RasterKernel) -> &CheckedComputePipeline {
        match kernel {
            RasterKernel::PathTagReduce => &self.pathtag_reduce,
            RasterKernel::PathTagReduce2 => &self.pathtag_reduce2,
            RasterKernel::PathTagScan1 => &self.pathtag_scan1,
            RasterKernel::PathTagScan => &self.pathtag_scan,
            RasterKernel::PathTagScanLarge => &self.pathtag_scan_large,
            RasterKernel::BboxClear => &self.bbox_clear,
            RasterKernel::Flatten => &self.flatten,
            RasterKernel::DrawReduce => &self.draw_reduce,
            RasterKernel::DrawLeaf => &self.draw_leaf,
            RasterKernel::ClipReduce => &self.clip_reduce,
            RasterKernel::ClipLeaf => &self.clip_leaf,
            RasterKernel::Binning => &self.binning,
            RasterKernel::TileAlloc => &self.tile_alloc,
            RasterKernel::PathCountSetup => &self.path_count_setup,
            RasterKernel::PathCount => &self.path_count,
            RasterKernel::Backdrop => &self.backdrop,
            RasterKernel::Coarse => &self.coarse,
            RasterKernel::PathTilingSetup => &self.path_tiling_setup,
            RasterKernel::PathTiling => &self.path_tiling,
            RasterKernel::FineArea => &self.fine_area,
            RasterKernel::FineMsaa8 => &self.fine_msaa8,
            RasterKernel::FineMsaa16 => &self.fine_msaa16,
        }
    }
}

async fn checked_wgpu_build<T>(
    device: &wgpu::Device,
    build: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let scope = CheckedWgpuScope::begin(device);
    let result = build();
    scope
        .finish("checked internal Vello shader creation failed")
        .await?;
    result
}

fn binding_type(binding: BindType) -> wgpu::BindingType {
    match binding {
        BindType::Buffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindType::BufReadOnly => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindType::Uniform => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindType::Image => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        BindType::ImageRead => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
    }
}

fn render_failed(message: &'static str) -> Error {
    Error::new(BackendErrorCode::RenderFailed, message)
}

fn classify_checked_wgpu_error(failure_message: &'static str, source: wgpu::Error) -> Error {
    let code = match &source {
        wgpu::Error::OutOfMemory { .. } => BackendErrorCode::SurfaceOutOfMemory,
        wgpu::Error::Validation { .. } | wgpu::Error::Internal { .. } => {
            BackendErrorCode::RenderFailed
        }
    };
    Error::new(code, failure_message).with_source(source)
}

#[cfg(test)]
pub(super) async fn checked_shader_validation_for_test(device: &wgpu::Device) -> Result<()> {
    checked_wgpu_build(device, || {
        let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surgeist deliberate invalid internal Vello WGSL"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main( {".into()),
        });
        Ok(())
    })
    .await
}

#[cfg(test)]
pub(super) fn checked_scope_out_of_memory_for_test() -> Error {
    classify_checked_wgpu_error(
        "checked internal Vello resource or command encoding failed",
        wgpu::Error::OutOfMemory {
            source: Box::new(std::io::Error::other("injected internal Vello OOM")),
        },
    )
}
