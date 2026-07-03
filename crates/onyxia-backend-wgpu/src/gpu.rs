//! Device/queue initialization and the pipeline machinery.
//!
//! Bind group layouts are built by reflecting the shader's group-0 buffer
//! bindings from the `naga::Module`, because (as of wgpu 29) an auto layout
//! hardcodes `immediate_size: 0` (breaking `set_immediates`) and auto
//! layouts can't be reused in explicit pipeline layouts. Do not replace
//! this with `layout: None`.
//!
//! **Immediates are a progressive feature.** Core WebGPU has no push
//! constants, so browsers never offer `Features::IMMEDIATES`. When the
//! adapter lacks it (or `ONYXIA_NO_IMMEDIATES=1` forces the issue for
//! testing), [`PipelineCache`] rewrites each kernel's `var<immediate>`
//! params into a trailing read-only storage-buffer binding — byte layout
//! is identical in both address spaces (4-byte packed scalars and
//! `array<u32,8>`), so the same `Imm` blobs bind unchanged — and the
//! session binds a small pooled buffer per dispatch instead of calling
//! `set_immediates`.

use onyxia_ir::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Whether the device has real push constants (`Features::IMMEDIATES`).
    /// False in browsers (core WebGPU has none) → storage-buffer fallback.
    pub use_immediates: bool,
}

impl GpuContext {
    /// Initialize an adapter/device with the features the backend needs.
    /// Immediates are used when available; `ONYXIA_NO_IMMEDIATES=1` forces
    /// the fallback so native test runs can cover the web path.
    pub async fn new() -> Result<Self> {
        let forced_off = std::env::var("ONYXIA_NO_IMMEDIATES").is_ok_and(|v| v != "0");
        Self::new_with(!forced_off).await
    }

    /// Like [`new`](Self::new), but immediates can be disabled explicitly
    /// (used by tests to exercise the fallback on hardware that has them).
    pub async fn new_with(allow_immediates: bool) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Runtime(format!("no suitable GPU adapter: {e}")))?;
        let adapter_info = adapter.get_info();
        let use_immediates =
            allow_immediates && adapter.features().contains(wgpu::Features::IMMEDIATES);
        let mut required_features = if use_immediates {
            wgpu::Features::IMMEDIATES
        } else {
            wgpu::Features::empty()
        };
        // Timestamp queries power the opt-in per-dispatch profiler
        // (`WgpuSession::enable_profiling`); requesting the feature is
        // free when unused.
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // Take the adapter's full limits: model weights (embedding tables)
        // exceed the 128/256 MiB downlevel buffer defaults.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("onyxia"),
                required_features,
                required_limits: wgpu::Limits {
                    max_immediate_size: if use_immediates { IMMEDIATE_SIZE } else { 0 },
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
            use_immediates,
        })
    }

    /// Blocking wrapper for native callers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }
}

/// Compiled-pipeline cache keyed by generated shader source label.
pub struct PipelineCache {
    cache: HashMap<String, (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>)>,
    /// Immediate space in the pipeline layout; 0 selects the
    /// storage-buffer params fallback (kernels are rewritten on the fly).
    immediate_size: u32,
}

impl PipelineCache {
    /// A cache for devices with immediates (`IMMEDIATE_SIZE`) or without (0).
    pub fn new(immediate_size: u32) -> Self {
        Self {
            cache: HashMap::new(),
            immediate_size,
        }
    }

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

        let source = if self.immediate_size == 0 {
            immediates_to_storage(wgsl)
        } else {
            wgsl.to_string()
        };
        let wgsl = source.as_str();

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
            immediate_size: self.immediate_size,
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

/// Rewrite a kernel's `var<immediate>` params declaration into a read-only
/// storage buffer on the next free group-0 binding index.
///
/// Sound because the params structs use only 4-byte scalars and
/// `array<u32,8>`, whose byte layout is identical in the immediate and
/// storage address spaces (a *uniform* buffer would NOT work: uniform
/// arrays have 16-byte element stride). The session binds the params
/// buffer at index `buffers.len()`, which equals max+1 here because
/// generated kernels number their bindings densely from 0.
fn immediates_to_storage(wgsl: &str) -> String {
    const DECL: &str = "var<immediate>";
    if !wgsl.contains(DECL) {
        return wgsl.to_string();
    }
    debug_assert_eq!(wgsl.matches(DECL).count(), 1, "one params block per kernel");
    let next = wgsl
        .split("@binding(")
        .skip(1)
        .filter_map(|s| s.split(')').next()?.trim().parse::<u32>().ok())
        .max()
        .map_or(0, |max| max + 1);
    wgsl.replace(DECL, &format!("@group(0) @binding({next}) var<storage, read>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediates_rewrite_to_next_free_binding() {
        let src = "struct P { n: u32 }\nvar<immediate> p: P;\n\
                   @group(0) @binding(0) var<storage, read> a: array<f32>;\n\
                   @group(0) @binding(1) var<storage, read_write> o: array<f32>;\n";
        let out = immediates_to_storage(src);
        assert!(
            out.contains("@group(0) @binding(2) var<storage, read> p: P;"),
            "{out}"
        );
        assert!(!out.contains("var<immediate>"));
        // No params block → untouched.
        assert_eq!(immediates_to_storage("fn main() {}"), "fn main() {}");
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

/// Live/peak byte counters for the GPU buffers of one session.
///
/// Every buffer a session creates ([`TrackedBuffer`]) adds its allocated
/// size on creation and subtracts it on drop, so `live` is the session's
/// true resident footprint: weights, pooled intermediates, params
/// buffers, and tensor handles the caller still holds (e.g. a KV cache).
#[derive(Default, Debug)]
pub struct MemCounter {
    live: AtomicU64,
    peak: AtomicU64,
}

impl MemCounter {
    fn add(&self, bytes: u64) {
        let live = self.live.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    /// Bytes currently allocated.
    pub fn live(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// High-water mark of [`Self::live`].
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

/// A `wgpu::Buffer` counted against a [`MemCounter`] for its whole
/// lifetime. Derefs to the underlying buffer.
pub struct TrackedBuffer {
    buffer: wgpu::Buffer,
    mem: Arc<MemCounter>,
}

impl TrackedBuffer {
    pub fn new(buffer: wgpu::Buffer, mem: &Arc<MemCounter>) -> Self {
        mem.add(buffer.size());
        Self {
            buffer,
            mem: Arc::clone(mem),
        }
    }
}

impl std::ops::Deref for TrackedBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        self.mem.live.fetch_sub(self.buffer.size(), Ordering::Relaxed);
    }
}

/// Size-bucketed free list of GPU buffers (power-of-two buckets, ≥4 bytes).
#[derive(Default)]
pub struct BufferPool {
    free: HashMap<u64, Vec<Arc<TrackedBuffer>>>,
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
    pub fn acquire(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        mem: &Arc<MemCounter>,
    ) -> Arc<TrackedBuffer> {
        let bucket = Self::bucket(size);
        if let Some(buf) = self.free.get_mut(&bucket).and_then(Vec::pop) {
            self.reuses += 1;
            return buf;
        }
        self.allocations += 1;
        Arc::new(TrackedBuffer::new(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bucket,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mem,
        ))
    }

    /// Return a buffer to the pool (it stays resident, and counted).
    pub fn release(&mut self, buffer: Arc<TrackedBuffer>) {
        self.free
            .entry(Self::bucket(buffer.size()))
            .or_default()
            .push(buffer);
    }
}
