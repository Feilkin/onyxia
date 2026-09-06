//! The wgpu session: prepare (legalize → order → liveness → upload
//! weights) and run (bind symbols → evaluate shapes → dispatch kernels).
//!
//! Composites with a kernel in the [`crate::fused`] registry survive
//! legalization and execute fused; everything else inlines through its
//! decomposition down to primitives, which run as generated
//! one-thread-per-element kernels (correctness-first). Fused GQA,
//! RotaryEmbedding, and MatMulNBits kernels live in `fused.rs`.

use crate::gpu::{
    BindGroupCache, BufferPool, GpuContext, IMMEDIATE_SIZE, MemCounter, PipelineCache,
    TrackedBuffer, WORKGROUP_SIZE, dispatch_size,
};
use crate::kernels::{self, Imm, MAX_RANK};
use crate::layout::{Caps, Layout, Repr};
use crate::profile::{KernelTiming, Profiler};
use onyxia_ir::graph::{Module, NodeId, NodeKind, Origin, ValueId};
use onyxia_ir::interp::{Tensor, bind_shapes};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, UnaryOp};
use onyxia_ir::{DataType, Error, Result, Session as _};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A device-resident tensor handle. Cheap to clone (buffer is shared).
#[derive(Clone)]
pub struct GpuTensor {
    pub(crate) buffer: Arc<TrackedBuffer>,
    /// Logical dtype (the physical GPU layout is backend-private).
    pub dtype: DataType,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Convert host bytes (logical layout) to device bytes (physical layout).
/// Every layout except the narrowed i64 and Bool stores the host bytes
/// verbatim (packed 8-bit and f16 words are the bytes in memory order).
fn to_phys(t: &Tensor, caps: Caps) -> Result<Vec<u8>> {
    let l = Layout::of(t.dtype(), caps)?;
    let mut data: Vec<u8> = match l.repr {
        Repr::I64Narrow => t
            .bytes()
            .chunks_exact(8)
            .map(|c| {
                let v = i64::from_le_bytes(c.try_into().unwrap());
                i32::try_from(v)
                    .map(|v| v.to_le_bytes().to_vec())
                    .map_err(|_| {
                        Error::Unsupported(format!(
                            "i64 value {v} does not fit the wgpu backend's 32-bit storage \
                             (this adapter lacks SHADER_INT64)"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()
            .map(|v| v.concat())?,
        _ if t.dtype() == DataType::Bool => t
            .bytes()
            .iter()
            .flat_map(|&b| (b as u32).to_le_bytes())
            .collect(),
        _ => t.bytes().to_vec(),
    };
    data.resize(l.buffer_bytes(t.numel()) as usize, 0);
    Ok(data)
}

/// Convert device bytes back to a host tensor of the logical dtype.
fn from_phys(dtype: DataType, shape: &[usize], bytes: &[u8], caps: Caps) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let l = Layout::of(dtype, caps)?;
    let logical: Vec<u8> = match l.repr {
        Repr::I64Narrow => bytes[..numel * 4]
            .chunks_exact(4)
            .flat_map(|c| (i32::from_le_bytes(c.try_into().unwrap()) as i64).to_le_bytes())
            .collect(),
        _ if dtype == DataType::Bool => bytes[..numel * 4]
            .chunks_exact(4)
            .map(|c| (u32::from_le_bytes(c.try_into().unwrap()) != 0) as u8)
            .collect(),
        _ => bytes[..dtype.storage_bytes(numel)].to_vec(),
    };
    Tensor::new(dtype, shape.to_vec(), logical)
}

/// The wgpu backend.
pub struct WgpuBackend {
    ctx: GpuContext,
    decompositions: onyxia_ir::DecompositionRegistry,
    kernels: crate::fused::KernelRegistry,
}

impl WgpuBackend {
    /// Create over an initialized GPU context, with the standard fused
    /// kernels registered.
    pub fn new(ctx: GpuContext) -> Self {
        Self {
            ctx,
            decompositions: onyxia_ir::standard_decompositions(),
            kernels: crate::fused::standard_kernels(),
        }
    }

    /// Same, but executing *only* primitive kernels — every composite runs
    /// through its decomposition. Used by differential tests to compare
    /// fused kernels against their decompositions on the same device.
    pub fn without_fused_kernels(ctx: GpuContext) -> Self {
        Self {
            ctx,
            decompositions: onyxia_ir::standard_decompositions(),
            kernels: crate::fused::KernelRegistry::default(),
        }
    }
}

impl onyxia_ir::Backend for WgpuBackend {
    type Session = WgpuSession;

    fn supports(&self, composite: &str) -> bool {
        self.kernels.contains(composite)
    }

    fn prepare(&self, mut module: Module) -> Result<Self::Session> {
        let kernels = self.kernels.clone();
        // Fuse patterns this backend has kernels for, then legalize.
        onyxia_ir::fuse_composites(&mut module, &|name| kernels.contains(name));
        let mut module = onyxia_ir::inline_composites(module, &self.decompositions, &|name| {
            kernels.contains(name)
        })?;
        // Weight tables larger than one storage binding (mobile GPUs cap it
        // at 128 MiB; the embedding table is 671 MB) become row chunks.
        let max_binding = self.ctx.device.limits().max_storage_buffer_binding_size as usize;
        onyxia_ir::split_large_tables(&mut module, max_binding)?;
        onyxia_ir::validate::validate(&module)?;
        let order = module.topo_order()?;

        // Liveness: the last step index that reads each value. Module
        // outputs (and inputs) are never freed within a run. Inverted
        // into per-step death lists so the run loop touches only the
        // values that actually die at each step.
        let mut last_use: Vec<Option<usize>> = vec![Some(0); module.values.len()];
        for (step, &node_id) in order.iter().enumerate() {
            for &v in &module.node(node_id).inputs {
                last_use[v.index()] = Some(step);
            }
        }
        for (_, id) in module.outputs.iter().chain(module.inputs.iter()) {
            last_use[id.index()] = None;
        }
        let mut deaths: Vec<Vec<u32>> = vec![Vec::new(); order.len()];
        for (vi, lu) in last_use.iter().enumerate() {
            if let Some(step) = lu {
                deaths[*step].push(vi as u32);
            }
        }

        // Upload constants once. `write_buffer` stages each write until
        // the next submit, so flush every so often — otherwise a
        // multi-GiB model holds both the staging and device copies
        // alive at once and exhausts memory during prepare.
        const FLUSH_BYTES: u64 = 256 << 20;
        let mem = Arc::new(MemCounter::default());
        let mut consts: HashMap<ValueId, GpuTensor> = HashMap::new();
        let mut staged: u64 = 0;
        // Constants nothing reads (e.g. a table replaced by its chunks) stay
        // on the host.
        let used: HashSet<ValueId> = module
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().copied())
            .chain(module.outputs.iter().map(|(_, v)| *v))
            .collect();
        for id in module.value_ids() {
            let def = module.value(id);
            let Origin::Const(cid) = def.origin else {
                continue;
            };
            if !used.contains(&id) {
                continue;
            }
            let layout = Layout::of(def.ty.dtype, self.ctx.caps)?; // fail early
            let host = onyxia_ir::interp::const_tensor(&module, cid)?;
            let data = to_phys(&host, self.ctx.caps)?;
            let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: def.name.as_deref(),
                size: layout.buffer_bytes(host.numel()),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&buffer, 0, &data);
            staged += data.len() as u64;
            if staged >= FLUSH_BYTES {
                self.ctx.queue.submit([]);
                self.ctx
                    .device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map_err(|e| Error::Runtime(format!("GPU poll failed: {e:?}")))?;
                staged = 0;
            }
            consts.insert(
                id,
                GpuTensor {
                    buffer: Arc::new(TrackedBuffer::new(buffer, &mem)),
                    dtype: def.ty.dtype,
                    shape: host.shape().to_vec(),
                },
            );
        }

        Ok(WgpuSession {
            device: Arc::clone(&self.ctx.device),
            queue: Arc::clone(&self.ctx.queue),
            module,
            order,
            deaths,
            consts,
            kernels: self.kernels.clone(),
            pipelines: PipelineCache::new(if self.ctx.use_immediates {
                IMMEDIATE_SIZE
            } else {
                0
            }),
            bind_groups: BindGroupCache::default(),
            pool: BufferPool::default(),
            mem,
            encoder: None,
            pass: None,
            use_immediates: self.ctx.use_immediates,
            submit_chunk: self.ctx.submit_chunk,
            matmul_tile: self.ctx.matmul_tile,
            imm_buffers: Vec::new(),
            imm_free: Vec::new(),
            profiler: None,
            cpu: CpuTiming::default(),
            caps: self.ctx.caps,
        })
    }
}

/// Where the CPU spends a step, in nanoseconds, accumulated across calls:
/// `shapes` = symbol binding + shape evaluation + register setup,
/// `encode` = the dispatch loop through `queue.submit`, `wait` = blocking
/// on the GPU in `download` (the GPU's own execution time plus queue
/// latency), `readback` = staging copy setup and the host copies.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuTiming {
    pub shapes_ns: u64,
    pub encode_ns: u64,
    pub wait_ns: u64,
    pub readback_ns: u64,
    /// Dispatches encoded.
    pub dispatches: u64,
    /// Bind-group cache misses (each one is a `create_bind_group`).
    pub bind_misses: u64,
}

/// Dispatches encoded per intermediate `queue.submit` during a run.
/// A prepared wgpu session.
pub struct WgpuSession {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    module: Module,
    order: Vec<NodeId>,
    /// Value indices whose last read is at each step (freed to the pool).
    deaths: Vec<Vec<u32>>,
    consts: HashMap<ValueId, GpuTensor>,
    kernels: crate::fused::KernelRegistry,
    pipelines: PipelineCache,
    bind_groups: BindGroupCache,
    pool: BufferPool,
    /// Live/peak byte accounting for every buffer this session creates.
    mem: Arc<MemCounter>,
    encoder: Option<wgpu::CommandEncoder>,
    /// Shared compute pass for the in-flight batch (non-profiling mode);
    /// ended (dropped) at submit, before the encoder finishes.
    pass: Option<wgpu::ComputePass<'static>>,
    /// False → params bind as a storage buffer instead of `set_immediates`
    /// (the web path; see `gpu.rs` module docs).
    use_immediates: bool,
    /// See [`GpuContext::submit_chunk`].
    submit_chunk: usize,
    /// See [`GpuContext::matmul_tile`].
    matmul_tile: crate::gpu::MatmulTile,
    /// Params buffers for the in-flight batch. Each dispatch gets its own
    /// (all `write_buffer`s execute before the batch), returned to
    /// `imm_free` at submit. MUST NOT come from the tensor pool: a params
    /// `write_buffer` executes before the batch, so sharing a buffer with a
    /// tensor that dies mid-batch lets the tensor write clobber the params.
    imm_buffers: Vec<Arc<TrackedBuffer>>,
    /// Free list of `IMMEDIATE_SIZE` params buffers (fallback mode only).
    imm_free: Vec<Arc<TrackedBuffer>>,
    /// Per-dispatch GPU timing, when enabled (see [`Self::enable_profiling`]).
    profiler: Option<Profiler>,
    /// Accumulated CPU-side phase times since the last [`Self::take_cpu_timing`].
    cpu: CpuTiming,
    /// Shader dtype features → physical layouts.
    caps: Caps,
}

impl WgpuSession {
    /// Buffer-pool statistics `(fresh_allocations, reuses)`.
    pub fn pool_stats(&self) -> (usize, usize) {
        self.pool.stats()
    }

    /// Total bytes of live GPU buffers created by this session: uploaded
    /// weights, pooled intermediates, params buffers, and tensor handles
    /// the caller still holds (e.g. a device-resident KV cache). Grows
    /// with context length as the KV cache does.
    pub fn resident_bytes(&self) -> u64 {
        self.mem.live()
    }

    /// High-water mark of [`Self::resident_bytes`] since `prepare`.
    pub fn peak_resident_bytes(&self) -> u64 {
        self.mem.peak()
    }

    /// Enable per-dispatch GPU timing. Returns `false` (and stays off)
    /// when the device lacks timestamp queries — core WebGPU makes them
    /// optional, so callers must treat profiling as best-effort.
    ///
    /// Drain the accumulated CPU-side phase times (see [`CpuTiming`]).
    /// `bind_misses` is cumulative since the session was created.
    pub fn take_cpu_timing(&mut self) -> CpuTiming {
        let misses = self.cpu.bind_misses;
        let mut out = std::mem::take(&mut self.cpu);
        out.bind_misses = misses;
        self.cpu.bind_misses = misses;
        out
    }

    /// While enabled, every dispatch's GPU execution time is recorded;
    /// drain the measurements with [`Self::take_timings`].
    pub fn enable_profiling(&mut self) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        if self.profiler.is_none() {
            self.profiler = Some(Profiler::new(&self.queue));
        }
        true
    }

    /// Drain per-dispatch GPU timings recorded since the last call
    /// (flushes in-flight work first). Empty when profiling is disabled.
    pub async fn take_timings(&mut self) -> Result<Vec<KernelTiming>> {
        self.submit();
        match &mut self.profiler {
            Some(p) => p.collect(&self.device).await,
            None => Ok(Vec::new()),
        }
    }

    /// Flush pending work and block until the GPU is idle. Benchmarks use
    /// this to time dispatch batches without a readback.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_idle(&mut self) -> Result<()> {
        self.submit();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| Error::Runtime(format!("GPU poll failed: {e:?}")))?;
        Ok(())
    }

    pub(crate) fn dispatch(
        &mut self,
        label: &str,
        wgsl: impl FnOnce() -> String,
        buffers: &[&Arc<TrackedBuffer>],
        imm: &Imm,
        size: usize,
    ) -> Result<()> {
        let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
        let (wg, _x_stride) = dispatch_size(linear);
        self.dispatch_grid(label, wgsl, buffers, imm, wg)
    }

    /// Dispatch a row-reduction kernel: exactly `rows` workgroups.
    pub(crate) fn dispatch_rows(
        &mut self,
        label: &str,
        wgsl: impl FnOnce() -> String,
        buffers: &[&Arc<TrackedBuffer>],
        imm: &Imm,
        rows: usize,
    ) -> Result<()> {
        self.dispatch_grid(label, wgsl, buffers, imm, [rows.max(1) as u32, 1, 1])
    }

    /// Dispatch with an explicit workgroup grid (kernels that don't map
    /// one thread per output element). `wgsl` runs only on a
    /// pipeline-cache miss; bind groups are cached by buffer identity.
    pub(crate) fn dispatch_grid(
        &mut self,
        label: &str,
        wgsl: impl FnOnce() -> String,
        buffers: &[&Arc<TrackedBuffer>],
        imm: &Imm,
        wg: [u32; 3],
    ) -> Result<()> {
        let (pipeline, layout) = self.pipelines.get_or_create(&self.device, label, wgsl)?;
        let imm_buf = self.imm_fallback_buffer(imm);
        let bind_group = {
            let mut all: Vec<&Arc<TrackedBuffer>> = buffers.to_vec();
            all.extend(&imm_buf);
            self.bind_groups
                .get_or_create(&self.device, label, &layout, &all)
        };
        self.encode_pass(label, &pipeline, &bind_group, imm, wg);
        self.cpu.dispatches += 1;
        self.imm_buffers.extend(imm_buf);
        Ok(())
    }

    /// In fallback mode: a storage buffer holding this dispatch's params
    /// blob, bound where `set_immediates` would have put it. Drawn from a
    /// dedicated free list, never the tensor pool (see `imm_buffers`).
    fn imm_fallback_buffer(&mut self, imm: &Imm) -> Option<Arc<TrackedBuffer>> {
        if self.use_immediates {
            return None;
        }
        debug_assert!(imm.bytes().len() <= IMMEDIATE_SIZE as usize);
        let buf = self.imm_free.pop().unwrap_or_else(|| {
            Arc::new(TrackedBuffer::new(
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("params"),
                    size: IMMEDIATE_SIZE as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                &self.mem,
            ))
        });
        self.queue.write_buffer(&buf, 0, imm.bytes());
        Some(buf)
    }

    fn encode_pass(
        &mut self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        imm: &Imm,
        wg: [u32; 3],
    ) {
        let encoder = self.encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("onyxia_batch"),
                })
        });
        // Profiling needs pass-granularity timestamps, so each dispatch
        // gets its own pass. Otherwise every dispatch in the batch shares
        // one pass — per-pass begin/end costs real CPU in wgpu-core and
        // the driver, and WebGPU already orders dispatches within a pass.
        if let Some(p) = self.profiler.as_mut() {
            self.pass = None;
            let (set, base) = p.begin_pass(&self.device, label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: p.query_set(set),
                    beginning_of_pass_write_index: Some(base),
                    end_of_pass_write_index: Some(base + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            if self.use_immediates {
                pass.set_immediates(0, imm.bytes());
            }
            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
            return;
        }
        let pass = self.pass.get_or_insert_with(|| {
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("onyxia_batch"),
                    timestamp_writes: None,
                })
                .forget_lifetime()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if self.use_immediates {
            pass.set_immediates(0, imm.bytes());
        }
        pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
    }

    fn submit(&mut self) {
        // End the shared pass first: it records into the encoder on drop.
        self.pass = None;
        if let Some(mut encoder) = self.encoder.take() {
            if let Some(p) = &mut self.profiler {
                p.resolve(&self.device, &mut encoder);
            }
            self.queue.submit([encoder.finish()]);
        }
        // Safe to recycle now: a later batch's `write_buffer`s are queue-
        // ordered after this submit's execution.
        self.imm_free.append(&mut self.imm_buffers);
    }

    pub(crate) fn alloc_out(&mut self, dtype: DataType, shape: Vec<usize>) -> GpuTensor {
        let bytes = Layout::of(dtype, self.caps)
            .map(|l| l.buffer_bytes(shape.iter().product()))
            .unwrap_or(4);
        let buffer = self.pool.acquire(&self.device, bytes, &self.mem);
        GpuTensor {
            buffer,
            dtype,
            shape,
        }
    }

    /// The physical layout of a dtype on this session's device.
    pub(crate) fn layout(&self, dtype: DataType) -> Result<Layout> {
        Layout::of(dtype, self.caps)
    }

    /// Materialize `x` broadcast to `shape` (numpy rules).
    pub(crate) fn broadcast_to(&mut self, x: &GpuTensor, shape: Vec<usize>) -> Result<GpuTensor> {
        let t = self.layout(x.dtype)?;
        check_rank(&shape, "broadcast")?;
        let out = self.alloc_out(x.dtype, shape);
        let (imm, size) = size_imm_l(&t, out.numel());
        let imm = imm
            .u(out.shape.len() as u32)
            .u(x.shape.len() as u32)
            .arr8(&out.shape)
            .arr8(&x.shape);
        self.dispatch(
            &format!("broadcast_{}", t.tag()),
            || kernels::broadcast(&t),
            &[&x.buffer, &out.buffer],
            &imm,
            size,
        )?;
        Ok(out)
    }

    /// Convert `x` to `dtype` (a Cast dispatch; aliases when the layouts
    /// already agree).
    pub(crate) fn cast_to(&mut self, x: &GpuTensor, dtype: DataType) -> Result<GpuTensor> {
        let (ls, ld) = (self.layout(x.dtype)?, self.layout(dtype)?);
        let expr = cast_expr(&ls, &ld);
        if expr == "v" && ls.store() == ld.store() && ls.lanes() == ld.lanes() {
            return Ok(GpuTensor {
                buffer: Arc::clone(&x.buffer),
                dtype,
                shape: x.shape.clone(),
            });
        }
        let out = self.alloc_out(dtype, x.shape.clone());
        let (imm, size) = size_imm_l(&ld, out.numel());
        self.dispatch(
            &format!("cast_{}_{}_{dtype}", ls.tag(), ld.tag()),
            || kernels::cast(&ls, &ld, &expr),
            &[&x.buffer, &out.buffer],
            &imm,
            size,
        )?;
        Ok(out)
    }
}

/// Immediate prefix + thread count for a kernel that runs one thread per
/// output *word* of layout `l` over `numel` elements.
fn size_imm_l(l: &Layout, numel: usize) -> (Imm, usize) {
    let words = l.words(numel) as usize;
    let linear = (words as u32).div_ceil(WORKGROUP_SIZE);
    let (_wg, x_stride) = dispatch_size(linear);
    (Imm::new().u(numel as u32).u(x_stride), words)
}

/// Common immediate prefix: size + x_stride for the bounds check.
pub(crate) fn size_imm(size: usize) -> (Imm, usize) {
    let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
    let (_wg, x_stride) = dispatch_size(linear);
    (Imm::new().u(size as u32).u(x_stride), size)
}

fn check_rank(shape: &[usize], what: &str) -> Result<()> {
    if shape.len() > MAX_RANK {
        return Err(Error::Unsupported(format!(
            "{what}: rank {} exceeds the kernel maximum of {MAX_RANK}",
            shape.len()
        )));
    }
    Ok(())
}

#[async_trait::async_trait(?Send)]
impl onyxia_ir::Session for WgpuSession {
    type Tensor = GpuTensor;

    fn upload(&mut self, tensor: &Tensor) -> Result<GpuTensor> {
        // Pooled, so per-step inputs (ids, positions, masks) reuse the
        // same buffers — and the bind groups that read them — step to step.
        let data = to_phys(tensor, self.caps)?;
        let buffer = self
            .pool
            .acquire(&self.device, data.len() as u64, &self.mem);
        self.queue.write_buffer(&buffer, 0, &data);
        Ok(GpuTensor {
            buffer,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
    }

    async fn run(&mut self, inputs: &[(&str, GpuTensor)]) -> Result<Vec<(String, GpuTensor)>> {
        let t0 = std::time::Instant::now();
        // 1. Bind symbols from the provided input shapes.
        let described: Vec<(&str, DataType, &[usize])> = inputs
            .iter()
            .map(|(n, t)| (*n, t.dtype, t.shape.as_slice()))
            .collect();
        let bindings = bind_shapes(&self.module, &described)?;

        // 2. Concrete shape for every value.
        let shapes: Vec<Vec<usize>> = self
            .module
            .values
            .iter()
            .map(|def| {
                def.ty.shape.eval(&bindings).map_err(|e| {
                    Error::Binding(format!(
                        "cannot resolve shape {} for '{}': {e} (late-bound dims are \
                         not yet supported on the wgpu backend)",
                        def.ty.shape,
                        def.name.as_deref().unwrap_or("<unnamed>")
                    ))
                })
            })
            .collect::<Result<_>>()?;

        // 3. Register file.
        let mut regs: Vec<Option<GpuTensor>> = vec![None; self.module.values.len()];
        for (id, t) in &self.consts {
            regs[id.index()] = Some(t.clone());
        }
        for (name, id) in &self.module.inputs {
            let (_, t) = inputs
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| Error::Binding(format!("missing input '{name}'")))?;
            regs[id.index()] = Some(t.clone());
        }

        // 4. Dispatch.
        let t1 = std::time::Instant::now();
        self.cpu.shapes_ns += (t1 - t0).as_nanos() as u64;
        for step in 0..self.order.len() {
            let node_id = self.order[step];
            if let Some(p) = &mut self.profiler {
                let node = self.module.node(node_id);
                p.tag = node.loc.name.clone().unwrap_or_default();
            }
            self.run_node(node_id, &regs, &shapes, &bindings)
                .map(|outs| {
                    for (out, &out_id) in outs.into_iter().zip(&self.module.node(node_id).outputs) {
                        regs[out_id.index()] = Some(out);
                    }
                })
                .map_err(|e| {
                    let node = self.module.node(node_id);
                    let name = node.loc.name.as_deref().unwrap_or("<unnamed>");
                    Error::Runtime(format!("{} (node '{name}'): {e}", kind_name(node)))
                })?;

            // Release dead intermediates: dropping the last handle returns
            // the buffer to the pool (see `TrackedBuffer`).
            for &vi in &self.deaths[step] {
                regs[vi as usize] = None;
            }
            // Pipelining: hand the GPU the batch so far while the CPU keeps
            // encoding. Queue order keeps every buffer hand-off correct.
            if self.profiler.is_none()
                && self.submit_chunk > 0
                && (step + 1) % self.submit_chunk == 0
            {
                self.submit();
            }
        }
        self.submit();
        self.cpu.encode_ns += t1.elapsed().as_nanos() as u64;
        self.cpu.bind_misses = self.bind_groups.misses;

        // 5. Collect outputs.
        self.module
            .outputs
            .iter()
            .map(|(name, id)| {
                regs[id.index()]
                    .clone()
                    .map(|t| (name.clone(), t))
                    .ok_or_else(|| Error::Runtime(format!("output '{name}' was never produced")))
            })
            .collect()
    }

    async fn download(&mut self, tensor: &GpuTensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        self.submit();
        let size = self.layout(tensor.dtype)?.buffer_bytes(tensor.numel());
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, size);
        let sub = self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let t1 = std::time::Instant::now();
        self.cpu.readback_ns += (t1 - t0).as_nanos() as u64;
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout: None,
            })
            .map_err(|e| Error::Runtime(format!("GPU poll failed: {e:?}")))?;
        #[cfg(target_arch = "wasm32")]
        let _ = sub;
        rx.await
            .map_err(|e| Error::Runtime(format!("buffer map canceled: {e}")))?
            .map_err(|e| Error::Runtime(format!("buffer map failed: {e}")))?;
        let t2 = std::time::Instant::now();
        self.cpu.wait_ns += (t2 - t1).as_nanos() as u64;
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        let out = from_phys(tensor.dtype, &tensor.shape, &bytes, self.caps);
        self.cpu.readback_ns += t2.elapsed().as_nanos() as u64;
        out
    }
}

fn kind_name(node: &onyxia_ir::Node) -> &str {
    match &node.kind {
        NodeKind::Prim(p) => p.name(),
        NodeKind::Composite(c) => &c.name,
    }
}

impl WgpuSession {
    /// Execute one node, returning its output tensors.
    fn run_node(
        &mut self,
        node_id: NodeId,
        regs: &[Option<GpuTensor>],
        shapes: &[Vec<usize>],
        bindings: &onyxia_ir::Bindings,
    ) -> Result<Vec<GpuTensor>> {
        let node = self.module.node(node_id).clone();
        match &node.kind {
            NodeKind::Prim(_) => self
                .run_prim(&node, regs, shapes, bindings)
                .map(|t| vec![t]),
            NodeKind::Composite(c) => {
                let kernels = self.kernels.clone();
                let kernel = kernels.get(&c.name).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "composite '{}' reached the executor without a registered \
                         kernel (legalization should have inlined it)",
                        c.name
                    ))
                })?;
                let mut inputs: Vec<GpuTensor> = node
                    .inputs
                    .iter()
                    .map(|&v| {
                        regs[v.index()]
                            .clone()
                            .ok_or_else(|| Error::Runtime("input not materialized".into()))
                    })
                    .collect::<Result<_>>()?;
                let mut outs_meta: Vec<(DataType, Vec<usize>)> = node
                    .outputs
                    .iter()
                    .map(|&o| (self.module.value(o).ty.dtype, shapes[o.index()].clone()))
                    .collect();
                // Fused kernels are written for f32; run f16 composites
                // through them with casts at the boundary.
                let f16 = inputs.iter().any(|t| t.dtype == DataType::F16)
                    || outs_meta.iter().any(|(d, _)| *d == DataType::F16);
                let mut f16_outs = Vec::new();
                if f16 {
                    for t in &mut inputs {
                        if t.dtype == DataType::F16 {
                            *t = self.cast_to(t, DataType::F32)?;
                        }
                    }
                }
                if f16 {
                    for (i, (d, _)) in outs_meta.iter_mut().enumerate() {
                        if *d == DataType::F16 {
                            *d = DataType::F32;
                            f16_outs.push(i);
                        }
                    }
                }
                let mut outs = kernel.execute(self, &c.attrs, &inputs, &outs_meta)?;
                for i in f16_outs {
                    outs[i] = self.cast_to(&outs[i], DataType::F16)?;
                }
                Ok(outs)
            }
        }
    }

    /// Workgroup grid for the tiled f32 matmul, or `None` when a dimension
    /// exceeds the 65535-workgroup limit (the generic kernel runs then).
    pub(crate) fn tiled_grid(&self, m: usize, n: usize, batch: usize) -> Option<[u32; 3]> {
        let t = self.tile_size();
        let grid = [n.div_ceil(t), m.div_ceil(t), batch];
        grid.iter()
            .all(|&g| g <= 65535)
            .then(|| grid.map(|g| g as u32))
    }

    fn tile_size(&self) -> usize {
        match self.matmul_tile {
            crate::gpu::MatmulTile::Classic => 16,
            crate::gpu::MatmulTile::Rb | crate::gpu::MatmulTile::Coop => 64,
        }
    }

    /// Plain-f32 tiled matmul into `out`. `mnkb` = `[m, n, k, batch]`,
    /// `strides` = per-batch element strides of `a`/`b` (0 = broadcast).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_tiled(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        mnkb: [usize; 4],
        strides: [u32; 2],
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        let [m, n, k, batch] = mnkb;
        let grid = self.tiled_grid(m, n, batch).ok_or_else(|| {
            Error::Unsupported(format!(
                "tiled matmul grid for m={m} n={n} batch={batch} exceeds 65535 workgroups"
            ))
        })?;
        use crate::gpu::MatmulTile;
        let classic = self.matmul_tile == MatmulTile::Classic;
        let label = format!(
            "matmul_tiled{}_f32_{}{}",
            match self.matmul_tile {
                MatmulTile::Classic => "",
                MatmulTile::Rb => "_rb",
                MatmulTile::Coop => "_coop",
            },
            if trans_a { "t" } else { "n" },
            if trans_b { "t" } else { "n" },
        );
        if classic {
            let imm = Imm::new()
                .u(m as u32)
                .u(n as u32)
                .u(k as u32)
                .u(strides[0])
                .u(strides[1]);
            return self.dispatch_grid(
                &label,
                || kernels::matmul_tiled(trans_a, trans_b),
                &[&a.buffer, &b.buffer, &out.buffer],
                &imm,
                grid,
            );
        }
        // Split K when the tile grid can't fill the device.
        const TARGET_WG: usize = 256;
        const MAX_KS: usize = 32;
        let base_wg = (grid[0] as usize) * (grid[1] as usize) * batch;
        let ks = if base_wg >= TARGET_WG {
            1
        } else {
            TARGET_WG
                .div_ceil(base_wg)
                .min(k.div_ceil(64))
                .clamp(1, MAX_KS)
        };
        let chunk = k.div_ceil(ks).div_ceil(32) * 32;
        let ks = k.div_ceil(chunk.max(1)).max(1);
        let size = batch * m * n;
        let scratch = (ks > 1).then(|| self.acquire_scratch((ks * size * 4) as u64));
        let dst = scratch.as_ref().unwrap_or(&out.buffer);
        let imm = Imm::new()
            .u(m as u32)
            .u(n as u32)
            .u(k as u32)
            .u(strides[0])
            .u(strides[1])
            .u(batch as u32)
            .u(chunk as u32);
        let coop = self.matmul_tile == MatmulTile::Coop;
        self.dispatch_grid(
            &label,
            || {
                if coop {
                    kernels::matmul_coop(trans_a, trans_b)
                } else {
                    kernels::matmul_tiled_rb(trans_a, trans_b)
                }
            },
            &[&a.buffer, &b.buffer, dst],
            &imm,
            [grid[0], grid[1], (ks * batch) as u32],
        )?;
        if let Some(scratch) = scratch {
            let (imm, n_out) = size_imm(size);
            let imm = imm.u(ks as u32);
            self.dispatch(
                "matvec_reduce_f32",
                kernels::matvec_reduce,
                &[&scratch, &out.buffer],
                &imm,
                n_out,
            )?;
            self.release_scratch(scratch);
        }
        Ok(())
    }

    /// Block dequantization (`Prim::Dequantize` and the prefill half of
    /// the fused `MatMulNBits`): `(q - zp) * scale` per element, output in
    /// the scales' dtype.
    pub(crate) fn dequantize(
        &mut self,
        data: &GpuTensor,
        scales: &GpuTensor,
        zp: Option<&GpuTensor>,
        block_size: usize,
        out_dtype: DataType,
        out_shape: Vec<usize>,
    ) -> Result<GpuTensor> {
        let (dl, sl) = (self.layout(data.dtype)?, self.layout(scales.dtype)?);
        if out_dtype != scales.dtype {
            return Err(Error::Unsupported(format!(
                "dequantize output dtype {out_dtype} differs from the scales' {}",
                scales.dtype
            )));
        }
        let zl = zp.map(|z| self.layout(z.dtype)).transpose()?;
        let default_zp: i32 = if data.dtype == DataType::I4 {
            0
        } else {
            1 << (data.dtype.bits() - 1)
        };
        let out = self.alloc_out(out_dtype, out_shape);
        let (imm, size) = size_imm_l(&sl, out.numel());
        let imm = imm.u(block_size as u32).i(default_zp);
        let mut buffers = vec![&data.buffer, &scales.buffer];
        if let Some(z) = zp {
            buffers.push(&z.buffer);
        }
        buffers.push(&out.buffer);
        self.dispatch(
            &format!(
                "dequantize_{}_{}{}",
                dl.tag(),
                sl.tag(),
                zl.map(|z| format!("_zp{}", z.tag())).unwrap_or_default()
            ),
            || kernels::dequantize(&dl, &sl, zl.as_ref()),
            &buffers,
            &imm,
            size,
        )?;
        Ok(out)
    }

    /// A pooled scratch buffer of at least `bytes` (return it with
    /// [`Self::release_scratch`] once its last dispatch is encoded).
    pub(crate) fn acquire_scratch(&mut self, bytes: u64) -> Arc<TrackedBuffer> {
        self.pool.acquire(&self.device, bytes, &self.mem)
    }

    pub(crate) fn release_scratch(&mut self, buffer: Arc<TrackedBuffer>) {
        self.pool.release(buffer);
    }

    /// M=1 matmul via the split-K matvec kernels (see the matvec section
    /// of `kernels.rs`). When K is sliced (`ks > 1`) partial sums land in
    /// a pooled scratch buffer and a second dispatch folds them into the
    /// output; within a batch the queue orders the two dispatches, so the
    /// scratch can return to the pool as soon as both are encoded.
    fn matvec(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: GpuTensor,
        n: usize,
        k: usize,
        trans_b: bool,
    ) -> Result<GpuTensor> {
        /// Workgroups needed to fill a discrete GPU.
        const TARGET_WG: usize = 512;
        const MAX_KS: usize = 64;

        // Vec4 variants when the vectorized axis is 4-aligned (K for the
        // [N,K] layout, N for [K,N]); the scalar kernels remain as the
        // fallback for odd sizes.
        let vec4 = if trans_b { k % 4 == 0 } else { n % 4 == 0 };
        let base_wg = match (trans_b, vec4) {
            (true, true) => n.div_ceil(4), // 4 rows per workgroup
            (true, false) => n,
            // Both [K,N] kernels tile 64 scalar columns per workgroup.
            (false, _) => n.div_ceil(64),
        };
        let per_slice = if trans_b { 256 } else { 64 };
        let ks = if base_wg >= TARGET_WG {
            1
        } else {
            TARGET_WG
                .div_ceil(base_wg)
                .min(k.div_ceil(per_slice))
                .clamp(1, MAX_KS)
        };

        let scratch = (ks > 1).then(|| {
            self.pool
                .acquire(&self.device, (ks * n * 4) as u64, &self.mem)
        });
        let dst: &Arc<TrackedBuffer> = scratch.as_ref().unwrap_or(&out.buffer);
        let buffers = [&a.buffer, &b.buffer, dst];

        if trans_b {
            let total = (base_wg * ks) as u32;
            let x_wgs = total.min(65535);
            let grid = [x_wgs, total.div_ceil(x_wgs), 1];
            if vec4 {
                let k4 = k / 4;
                let imm = Imm::new()
                    .u(n as u32)
                    .u(k4 as u32)
                    .u(ks as u32)
                    .u(k4.div_ceil(ks) as u32)
                    .u(x_wgs);
                self.dispatch_grid(
                    "matvec_transb_v4_f32",
                    kernels::matvec_transb_v4,
                    &buffers,
                    &imm,
                    grid,
                )?;
            } else {
                let imm = Imm::new()
                    .u(n as u32)
                    .u(k as u32)
                    .u(ks as u32)
                    .u(k.div_ceil(ks) as u32)
                    .u(x_wgs);
                self.dispatch_grid(
                    "matvec_transb_f32",
                    kernels::matvec_transb,
                    &buffers,
                    &imm,
                    grid,
                )?;
            }
        } else {
            let grid = [base_wg as u32, ks as u32, 1];
            let imm = Imm::new()
                .u(if vec4 { n / 4 } else { n } as u32)
                .u(k as u32)
                .u(ks as u32)
                .u(k.div_ceil(ks) as u32);
            if vec4 {
                self.dispatch_grid(
                    "matvec_kn_v4_f32",
                    kernels::matvec_kn_v4,
                    &buffers,
                    &imm,
                    grid,
                )?;
            } else {
                self.dispatch_grid("matvec_kn_f32", kernels::matvec_kn, &buffers, &imm, grid)?;
            }
        }

        if let Some(scratch) = scratch {
            let (imm, size) = size_imm(n);
            let imm = imm.u(ks as u32);
            self.dispatch(
                "matvec_reduce_f32",
                kernels::matvec_reduce,
                &[&scratch, &out.buffer],
                &imm,
                size,
            )?;
            self.pool.release(scratch);
        }
        Ok(out)
    }

    /// Execute one primitive node, returning its output tensor.
    fn run_prim(
        &mut self,
        node: &onyxia_ir::Node,
        regs: &[Option<GpuTensor>],
        shapes: &[Vec<usize>],
        bindings: &onyxia_ir::Bindings,
    ) -> Result<GpuTensor> {
        let NodeKind::Prim(prim) = &node.kind else {
            unreachable!("run_prim called on a composite");
        };
        let input = |i: usize| -> Result<&GpuTensor> {
            regs[node.inputs[i].index()]
                .as_ref()
                .ok_or_else(|| Error::Runtime("input not materialized".into()))
        };
        let out_id = node.outputs[0];
        let out_shape = shapes[out_id.index()].clone();
        let out_dtype = self.module.value(out_id).ty.dtype;
        check_rank(&out_shape, prim.name())?;

        match prim {
            // ── zero-copy ────────────────────────────────────────────
            Prim::Reshape { .. } => {
                let x = input(0)?;
                Ok(GpuTensor {
                    buffer: Arc::clone(&x.buffer),
                    dtype: out_dtype,
                    shape: out_shape,
                })
            }

            Prim::Cast { .. } => {
                let x = input(0)?.clone();
                let (ls, ld) = (self.layout(x.dtype)?, self.layout(out_dtype)?);
                let expr = cast_expr(&ls, &ld);
                if expr == "v" && ls.store() == ld.store() && ls.lanes() == ld.lanes() {
                    // Same physical representation: alias.
                    return Ok(GpuTensor {
                        buffer: x.buffer,
                        dtype: out_dtype,
                        shape: out_shape,
                    });
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&ld, out.numel());
                self.dispatch(
                    &format!("cast_{}_{}_{out_dtype}", ls.tag(), ld.tag()),
                    || kernels::cast(&ls, &ld, &expr),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── element-wise ─────────────────────────────────────────
            Prim::Unary(op) => {
                let x = input(0)?.clone();
                let t = self.layout(x.dtype)?;
                let (expr, needs_erf) = unary_expr(*op, &t)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                self.dispatch(
                    &format!("unary_{}_{}", prim.name(), t.tag()),
                    || kernels::unary(&t, &t, expr, needs_erf),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Binary(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = self.layout(a.dtype)?;
                let expr = binary_expr(*op, &t)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(a.shape.len() as u32)
                    .u(b.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&a.shape)
                    .arr8(&b.shape);
                self.dispatch(
                    &format!("binary_{}_{}", prim.name(), t.tag()),
                    || kernels::binary(&t, &t, &t, expr),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Compare(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = self.layout(a.dtype)?;
                let ob = self.layout(DataType::Bool)?;
                let expr = compare_expr(*op, &t);
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&ob, out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(a.shape.len() as u32)
                    .u(b.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&a.shape)
                    .arr8(&b.shape);
                self.dispatch(
                    &format!("compare_{}_{}", prim.name(), t.tag()),
                    || kernels::binary(&t, &t, &ob, expr),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Select => {
                let (c, a, b) = (input(0)?.clone(), input(1)?.clone(), input(2)?.clone());
                let t = self.layout(a.dtype)?;
                let cl = self.layout(c.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(c.shape.len() as u32)
                    .u(a.shape.len() as u32)
                    .u(b.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&c.shape)
                    .arr8(&a.shape)
                    .arr8(&b.shape);
                self.dispatch(
                    &format!("select_{}", t.tag()),
                    || kernels::select3(&cl, &t),
                    &[&c.buffer, &a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── linear algebra ───────────────────────────────────────
            Prim::MatMul { trans_a, trans_b } => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = self.layout(a.dtype)?;
                let (ar, br) = (a.shape.len(), b.shape.len());
                let (m, k) = {
                    let (r, c) = (a.shape[ar - 2], a.shape[ar - 1]);
                    if *trans_a { (c, r) } else { (r, c) }
                };
                let n = if *trans_b {
                    b.shape[br - 2]
                } else {
                    b.shape[br - 1]
                };
                let batch: usize = out_shape[..out_shape.len() - 2].iter().product();
                // Batch dims either match the output, are absent/scalar,
                // or get materialized by a broadcast copy first.
                let out_batch = out_shape[..out_shape.len() - 2].to_vec();
                let expand = |this: &mut Self, x: &GpuTensor| -> Result<GpuTensor> {
                    let r = x.shape.len();
                    let bn: usize = x.shape[..r - 2].iter().product();
                    if bn == batch || bn == 1 {
                        return Ok(x.clone());
                    }
                    let mut full = out_batch.clone();
                    full.extend_from_slice(&x.shape[r - 2..]);
                    this.broadcast_to(x, full)
                };
                let a = expand(self, &a)?;
                let b = expand(self, &b)?;
                let stride_of = |batch_numel: usize, mat: usize| -> u32 {
                    if batch_numel == batch { mat as u32 } else { 0 }
                };
                let a_bs = stride_of(a.shape[..ar - 2].iter().product(), m * k);
                let b_bs = stride_of(b.shape[..br - 2].iter().product(), k * n);

                // Fast paths (plain f32): unbatched matrix × vector for
                // decode-step projections (`trans_a` is irrelevant at
                // m == 1 — `[K,1]` and `[1,K]` share a memory layout),
                // tiled matmul for everything else. Grid dims cap at
                // 65535 workgroups; anything larger falls through to the
                // generic kernel.
                if t.is_plain_f32() && k > 0 && n > 0 {
                    if batch == 1 && m == 1 && n.div_ceil(64) <= 65535 {
                        let out = self.alloc_out(out_dtype, out_shape);
                        return self.matvec(&a, &b, out, n, k, *trans_b);
                    }
                    if self.tiled_grid(m, n, batch).is_some() {
                        let out = self.alloc_out(out_dtype, out_shape);
                        self.matmul_tiled(
                            &a,
                            &b,
                            &out,
                            [m, n, k, batch],
                            [a_bs, b_bs],
                            *trans_a,
                            *trans_b,
                        )?;
                        return Ok(out);
                    }
                }

                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(m as u32)
                    .u(n as u32)
                    .u(k as u32)
                    .u(a_bs)
                    .u(b_bs)
                    .u(*trans_a as u32)
                    .u(*trans_b as u32);
                self.dispatch(
                    &format!("matmul_{}", t.tag()),
                    || kernels::matmul(&t),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Reduce { op, axes, .. } => {
                let x = input(0)?.clone();
                let t = self.layout(x.dtype)?;
                let (init, combine, finalize) = reduce_exprs(*op, &t);
                let mut mask = 0u32;
                let mut count = 1usize;
                for &a in axes {
                    mask |= 1 << a;
                    count *= x.shape[a];
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(x.shape.len() as u32)
                    .u(mask)
                    .u(count as u32)
                    .arr8(&x.shape);
                self.dispatch(
                    &format!("reduce_{}_{}", prim.name(), t.tag()),
                    || kernels::reduce(&t, init, combine, finalize),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── data movement ────────────────────────────────────────
            Prim::Transpose { perm } => {
                let x = input(0)?.clone();
                let t = self.layout(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(x.shape.len() as u32)
                    .arr8(perm)
                    .arr8(&x.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("transpose_{}", t.tag()),
                    || kernels::transpose(&t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Broadcast { .. } => {
                let x = input(0)?.clone();
                let t = self.layout(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(x.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&x.shape);
                self.dispatch(
                    &format!("broadcast_{}", t.tag()),
                    || kernels::broadcast(&t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Concat { axis } => {
                let out = self.alloc_out(out_dtype, out_shape.clone());
                let t = self.layout(out_dtype)?;
                let mut offset = 0usize;
                for i in 0..node.inputs.len() {
                    let x = input(i)?.clone();
                    if t.lanes() == 1 {
                        let (imm, size) = size_imm(x.numel());
                        let imm = imm
                            .u(x.shape.len() as u32)
                            .u(*axis as u32)
                            .u(offset as u32)
                            .arr8(&x.shape)
                            .arr8(&out_shape);
                        self.dispatch(
                            &format!("concat_{}", t.tag()),
                            || kernels::concat_emplace(&t),
                            &[&x.buffer, &out.buffer],
                            &imm,
                            size,
                        )?;
                    } else {
                        let (imm, size) = size_imm_l(&t, out.numel());
                        let imm = imm
                            .u(x.shape.len() as u32)
                            .u(*axis as u32)
                            .u(offset as u32)
                            .u(x.shape[*axis] as u32)
                            .arr8(&x.shape)
                            .arr8(&out_shape);
                        let wgsl = kernels::concat_packed(&t)?;
                        self.dispatch(
                            &format!("concat_packed_{}", t.tag()),
                            || wgsl,
                            &[&x.buffer, &out.buffer],
                            &imm,
                            size,
                        )?;
                    }
                    offset += x.shape[*axis];
                }
                Ok(out)
            }

            Prim::Slice { specs } => {
                let x = input(0)?.clone();
                let t = self.layout(x.dtype)?;
                // Per-axis start/step; unlisted axes are identity. Starts
                // may be symbolic (e.g. slicing an iota at `past_len`) —
                // they resolve under the current bindings.
                let rank = x.shape.len();
                let mut starts = vec![0u64; rank];
                let mut steps = vec![1i64; rank];
                for spec in specs {
                    starts[spec.axis] = spec.start.eval(bindings)?;
                    steps[spec.axis] = spec.step;
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(rank as u32)
                    .arr8(&starts.iter().map(|&s| s as usize).collect::<Vec<_>>())
                    .arr8_i(&steps)
                    .arr8(&x.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("slice_{}", t.tag()),
                    || kernels::slice(&t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Gather { axis } => {
                let (data, indices) = (input(0)?.clone(), input(1)?.clone());
                let t = self.layout(data.dtype)?;
                let il = self.layout(indices.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                let imm = imm
                    .u(*axis as u32)
                    .u(data.shape.len() as u32)
                    .u(indices.shape.len() as u32)
                    .arr8(&data.shape)
                    .arr8(&indices.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("gather_{}_{}", t.tag(), il.tag()),
                    || kernels::gather(&t, &il),
                    &[&data.buffer, &indices.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Scatter { reduction } => {
                use onyxia_ir::ScatterReduce;
                let (data, indices, updates) =
                    (input(0)?.clone(), input(1)?.clone(), input(2)?.clone());
                let t = self.layout(data.dtype)?;
                let il = self.layout(indices.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                // Stage 1: copy data into out.
                let (imm, size) = size_imm_l(&t, data.numel());
                self.dispatch(
                    &format!("copy_{}", t.tag()),
                    || kernels::copy(&t),
                    &[&data.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                // Stage 2: scatter updates (threads over update elements).
                let ir = indices.shape.len();
                let k = indices.shape[ir - 1];
                let slice_len: usize = data.shape[k..].iter().product();
                let (imm, size) = size_imm(updates.numel());
                let imm = imm
                    .u(k as u32)
                    .u(slice_len as u32)
                    .u(data.shape.len() as u32)
                    .arr8(&data.shape);
                if *reduction == ScatterReduce::None && t.lanes() == 1 {
                    self.dispatch(
                        &format!("scatter_{}_{}", t.tag(), il.tag()),
                        || kernels::scatter(&t, &il),
                        &[&indices.buffer, &updates.buffer, &out.buffer],
                        &imm,
                        size,
                    )?;
                } else {
                    let combine = match reduction {
                        ScatterReduce::None => "upd",
                        ScatterReduce::Add => "cur + upd",
                        ScatterReduce::Mul => "cur * upd",
                        ScatterReduce::Max => "max(cur, upd)",
                        ScatterReduce::Min => "min(cur, upd)",
                    };
                    let wgsl = kernels::scatter_atomic(&t, &il, combine)?;
                    self.dispatch(
                        &format!("scatter_{reduction:?}_{}_{}", t.tag(), il.tag()),
                        || wgsl,
                        &[&indices.buffer, &updates.buffer, &out.buffer],
                        &imm,
                        size,
                    )?;
                }
                Ok(out)
            }

            Prim::Iota { dtype, .. } => {
                let t = self.layout(*dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm_l(&t, out.numel());
                self.dispatch(
                    &format!("iota_{}", t.tag()),
                    || kernels::iota(&t),
                    &[&out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::DimValues { exprs } => {
                let vals: Vec<i64> = exprs
                    .iter()
                    .map(|e| e.eval_signed(bindings))
                    .collect::<Result<_>>()?;
                let t = Tensor::from_i64(&vals, &[vals.len()])?;
                self.upload(&t)
            }

            Prim::Dequantize { block_size, .. } => {
                let (data, scales) = (input(0)?.clone(), input(1)?.clone());
                let zp = if node.inputs.len() > 2 {
                    Some(input(2)?.clone())
                } else {
                    None
                };
                self.dequantize(
                    &data,
                    &scales,
                    zp.as_ref(),
                    *block_size,
                    out_dtype,
                    out_shape,
                )
            }
        }
    }
}

// ─────────────────── expression tables ─────────────────────────────────
//
// Expressions are over the layout's *compute* type: f32, i32, u32, or
// (with SHADER_INT64) i64. Packed 8-bit values compute as u32/i32 and are
// truncated on store, which matches the interpreter's wrapping semantics.

fn unary_expr(op: UnaryOp, l: &Layout) -> Result<(&'static str, bool)> {
    use UnaryOp::*;
    let c = l.compute();
    Ok(match op {
        Neg => ("-v", false),
        Abs => ("abs(v)", false),
        Sqrt => ("sqrt(v)", false),
        Rsqrt => ("inverseSqrt(v)", false),
        Exp => ("exp(v)", false),
        Log => ("log(v)", false),
        Sin => ("sin(v)", false),
        Cos => ("cos(v)", false),
        Tanh => ("tanh(v)", false),
        Erf => ("erf(v)", true),
        Floor => ("floor(v)", false),
        Ceil => ("ceil(v)", false),
        Round => ("round(v)", false), // WGSL round: ties to even
        Sign => match c {
            "u32" => ("select(0u, 1u, v != 0u)", false),
            _ => ("sign(v)", false),
        },
        Tan => ("tan(v)", false),
        Asin => ("asin(v)", false),
        Acos => ("acos(v)", false),
        Atan => ("atan(v)", false),
        Sinh => ("sinh(v)", false),
        Cosh => ("cosh(v)", false),
        Asinh => ("asinh(v)", false),
        Acosh => ("acosh(v)", false),
        Atanh => ("atanh(v)", false),
        Not => {
            if l.logical != DataType::Bool {
                return Err(Error::DType("Not on non-bool".into()));
            }
            ("select(1u, 0u, v != 0u)", false)
        }
        BitNot => ("~v", false),
    })
}

fn binary_expr(op: BinaryOp, l: &Layout) -> Result<&'static str> {
    use BinaryOp::*;
    let c = l.compute();
    Ok(match (op, c) {
        (Add, _) => "av + bv",
        (Sub, _) => "av - bv",
        (Mul, _) => "av * bv",
        (Div, _) => "av / bv",
        (Pow, "f32") => "pow(av, bv)",
        (Pow, _) => "ipow(av, bv)",
        (Max, _) => "max(av, bv)",
        (Min, _) => "min(av, bv)",
        (And, _) => "u32((av != 0u) && (bv != 0u))",
        (Or, _) => "u32((av != 0u) || (bv != 0u))",
        (Xor, _) => "u32((av != 0u) != (bv != 0u))",
        (BitAnd, _) => "av & bv",
        (BitOr, _) => "av | bv",
        (BitXor, _) => "av ^ bv",
        (Shl, "u32") => "select(av << bv, 0u, bv >= 32u)",
        (Shl, "i64") => "select(av << u32(bv), i64(0), bv >= i64(64))",
        (Shl, _) => "select(av << u32(bv), 0, bv >= 32)",
        (Shr, "u32") => "select(av >> bv, 0u, bv >= 32u)",
        (Shr, "i64") => {
            "select(av >> u32(bv), select(i64(0), i64(-1), av < i64(0)), bv >= i64(64))"
        }
        (Shr, _) => "select(av >> u32(bv), select(0, -1, av < 0), bv >= 32)",
    })
}

fn compare_expr(op: CmpOp, l: &Layout) -> &'static str {
    use CmpOp::*;
    // Shader compilers may assume no NaNs; test the bit pattern so Eq/Ne
    // keep IEEE semantics (`NaN != NaN`) for floats.
    match (op, l.compute()) {
        (Eq, "f32") => {
            "u32((av == bv) && !(((bitcast<u32>(av) & 0x7fffffffu) > 0x7f800000u) || ((bitcast<u32>(bv) & 0x7fffffffu) > 0x7f800000u)))"
        }
        (Ne, "f32") => {
            "u32((av != bv) || ((bitcast<u32>(av) & 0x7fffffffu) > 0x7f800000u) || ((bitcast<u32>(bv) & 0x7fffffffu) > 0x7f800000u))"
        }
        (Eq, _) => "u32(av == bv)",
        (Ne, _) => "u32(av != bv)",
        (Lt, _) => "u32(av < bv)",
        (Le, _) => "u32(av <= bv)",
        (Gt, _) => "u32(av > bv)",
        (Ge, _) => "u32(av >= bv)",
    }
}

fn reduce_exprs(op: ReduceOp, l: &Layout) -> (&'static str, &'static str, &'static str) {
    use ReduceOp::*;
    let c = l.compute();
    let (zero, one) = match c {
        "f32" => ("0.0", "1.0"),
        "u32" => ("0u", "1u"),
        "i64" => ("i64(0)", "i64(1)"),
        _ => ("0", "1"),
    };
    match op {
        Sum => (zero, "acc + v", "acc"),
        Mean => (
            zero,
            "acc + v",
            match c {
                "f32" => "acc / f32(p.reduce_count)",
                "u32" => "acc / p.reduce_count",
                "i64" => "acc / i64(p.reduce_count)",
                _ => "acc / i32(p.reduce_count)",
            },
        ),
        Prod => (one, "acc * v", "acc"),
        Max => (
            match c {
                "f32" => "bitcast<f32>(0xff800000u)", // -inf
                "u32" => "0u",
                "i64" => "(i64(-1) << 63u)",
                _ => "(-2147483647 - 1)",
            },
            "max(acc, v)",
            "acc",
        ),
        Min => (
            match c {
                "f32" => "bitcast<f32>(0x7f800000u)", // +inf
                "u32" => "4294967295u",
                "i64" => "~(i64(-1) << 63u)",
                _ => "2147483647",
            },
            "min(acc, v)",
            "acc",
        ),
    }
}

/// Conversion expression for Cast between compute types. `"v"` means the
/// value is unchanged (the layouts may still differ — e.g. packed u8 to
/// plain u32 — in which case a copy kernel runs).
fn cast_expr(src: &Layout, dst: &Layout) -> String {
    let (s, d) = (src.compute(), dst.compute());
    if dst.logical == DataType::Bool && src.logical != DataType::Bool {
        let zero = match s {
            "f32" => "0.0",
            "u32" => "0u",
            "i64" => "i64(0)",
            _ => "0",
        };
        return format!("select(0u, 1u, v != {zero})");
    }
    if s == d {
        return "v".to_string();
    }
    format!("{d}(v)")
}
