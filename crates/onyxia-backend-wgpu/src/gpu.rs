//! Device/queue initialization and the pipeline machinery.
//!
//! The pipeline-layout construction is ported from the old
//! `onyxia-core::dispatch` **including the wgpu-29 immediates fix**: bind
//! group layouts are built by reflecting the shader's group-0 buffer
//! bindings from the `naga::Module`, because an auto layout hardcodes
//! `immediate_size: 0` (breaking `set_immediates`) and auto layouts can't
//! be reused in explicit pipeline layouts. Do not replace this with
//! `layout: None`.

use onyxia_ir::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Reserved immediate (push-constant) space per pipeline, in bytes.
pub const IMMEDIATE_SIZE: u32 = 256;

/// Compute workgroup size used by all generated kernels.
pub const WORKGROUP_SIZE: u32 = 256;

/// Shared GPU device and queue.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    /// Adapter description (name, backend) for display/diagnostics.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize an adapter/device with the features the backend needs.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Runtime(format!("no suitable GPU adapter: {e}")))?;
        let adapter_info = adapter.get_info();
        // Take the adapter's full limits: model weights (embedding tables)
        // exceed the 128/256 MiB downlevel buffer defaults.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("onyxia"),
                required_features: wgpu::Features::IMMEDIATES,
                required_limits: wgpu::Limits {
                    max_immediate_size: IMMEDIATE_SIZE,
                    ..adapter.limits()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Runtime(format!("device request failed: {e}")))?;
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Blocking wrapper for native callers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }
}

/// Compiled-pipeline cache keyed by generated shader source label.
#[derive(Default)]
pub struct PipelineCache {
    cache: HashMap<String, (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>)>,
}

impl PipelineCache {
    /// Get or compile the pipeline for `label`, parsing `wgsl` on first use.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        wgsl: &str,
    ) -> Result<(Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>)> {
        if let Some((p, l)) = self.cache.get(label) {
            return Ok((Arc::clone(p), Arc::clone(l)));
        }

        let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| {
            Error::Runtime(format!(
                "kernel '{label}' failed to parse: {}\n--- source ---\n{wgsl}",
                e.emit_to_string(wgsl)
            ))
        })?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module.clone())),
        });

        // Reflect group-0 buffer bindings (see module docs for why).
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        for (_, var) in module.global_variables.iter() {
            let Some(binding) = &var.binding else {
                continue;
            };
            if binding.group != 0 {
                continue;
            }
            let ty = match &var.space {
                naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                naga::AddressSpace::Storage { access } => wgpu::BufferBindingType::Storage {
                    read_only: !access.contains(naga::StorageAccess::STORE),
                },
                _ => continue,
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.sort_by_key(|e| e.binding);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}_bgl")),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}_layout")),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: IMMEDIATE_SIZE,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let entry = (Arc::new(pipeline), Arc::new(bgl));
        self.cache.insert(label.to_string(), entry.clone());
        Ok(entry)
    }
}

/// Distribute a linear workgroup count over ≤3 dispatch dimensions
/// (wgpu caps each at 65535). Returns `([x, y, z], x_stride)`.
pub fn dispatch_size(linear_workgroups: u32) -> ([u32; 3], u32) {
    const MAX: u32 = 65535;
    if linear_workgroups <= MAX {
        (
            [linear_workgroups.max(1), 1, 1],
            linear_workgroups.max(1) * WORKGROUP_SIZE,
        )
    } else {
        let y = linear_workgroups.div_ceil(MAX);
        ([MAX, y, 1], MAX * WORKGROUP_SIZE)
    }
}

/// Size-bucketed free list of GPU buffers (power-of-two buckets, ≥4 bytes).
#[derive(Default)]
pub struct BufferPool {
    free: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
    /// Fresh allocations made (diagnostics).
    pub allocations: usize,
    /// Buffers served from the pool (diagnostics).
    pub reuses: usize,
}

impl BufferPool {
    fn bucket(size: u64) -> u64 {
        size.max(4).next_power_of_two()
    }

    /// Acquire a storage buffer of at least `size` bytes.
    pub fn acquire(&mut self, device: &wgpu::Device, size: u64) -> Arc<wgpu::Buffer> {
        let bucket = Self::bucket(size);
        if let Some(buf) = self.free.get_mut(&bucket).and_then(Vec::pop) {
            self.reuses += 1;
            return buf;
        }
        self.allocations += 1;
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bucket,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Return a buffer to the pool.
    pub fn release(&mut self, buffer: Arc<wgpu::Buffer>) {
        self.free
            .entry(Self::bucket(buffer.size()))
            .or_default()
            .push(buffer);
    }
}
