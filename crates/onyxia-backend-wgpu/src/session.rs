//! The wgpu session: prepare (legalize → order → liveness → upload
//! weights) and run (bind symbols → evaluate shapes → dispatch kernels).
//!
//! Composites with a kernel in the [`crate::fused`] registry survive
//! legalization and execute fused; everything else inlines through its
//! decomposition down to primitives, which run as generated
//! one-thread-per-element kernels (correctness-first). Fused GQA,
//! RotaryEmbedding, and MatMulNBits kernels are planned — see `fused.rs`.

use crate::gpu::{
    BufferPool, GpuContext, IMMEDIATE_SIZE, MemCounter, PipelineCache, TrackedBuffer,
    WORKGROUP_SIZE, dispatch_size,
};
use crate::kernels::{self, Imm, MAX_RANK};
use crate::profile::{KernelTiming, Profiler};
use onyxia_ir::graph::{Module, NodeId, NodeKind, Origin, ValueId};
use onyxia_ir::interp::{Tensor, bind_shapes};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, UnaryOp};
use onyxia_ir::{DataType, Error, Result};
use std::collections::HashMap;
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

/// Physical WGSL scalar for a logical dtype. The wgpu backend stores I64
/// as i32 (range-checked at upload) and Bool as u32; all supported types
/// are 4 bytes on-device.
fn phys(dt: DataType) -> Result<&'static str> {
    Ok(match dt {
        DataType::F32 => "f32",
        DataType::I64 | DataType::I32 => "i32",
        DataType::U32 | DataType::Bool => "u32",
        other => {
            return Err(Error::Unsupported(format!(
                "dtype {other} on the wgpu backend (f16/quantized kernels are future work)"
            )));
        }
    })
}

fn phys_bytes(numel: usize) -> u64 {
    // Multiply in u64: usize is 32-bit on wasm32, and buffer sizes may
    // legitimately exceed what a usize product can hold before the cast.
    numel.max(1) as u64 * 4
}

/// Convert host bytes (logical layout) to device bytes (physical layout).
fn to_phys(t: &Tensor) -> Result<Vec<u8>> {
    match t.dtype() {
        DataType::F32 | DataType::I32 | DataType::U32 => Ok(t.bytes().to_vec()),
        DataType::I64 => t
            .bytes()
            .chunks_exact(8)
            .map(|c| {
                let v = i64::from_le_bytes(c.try_into().unwrap());
                i32::try_from(v)
                    .map(|v| v.to_le_bytes().to_vec())
                    .map_err(|_| {
                        Error::Unsupported(format!(
                            "i64 value {v} does not fit the wgpu backend's 32-bit storage"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()
            .map(|v| v.concat()),
        DataType::Bool => Ok(t
            .bytes()
            .iter()
            .flat_map(|&b| (b as u32).to_le_bytes())
            .collect()),
        other => Err(Error::Unsupported(format!("upload of dtype {other}"))),
    }
}

/// Convert device bytes back to a host tensor of the logical dtype.
fn from_phys(dtype: DataType, shape: &[usize], bytes: &[u8]) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let data = &bytes[..numel * 4];
    let logical: Vec<u8> = match dtype {
        DataType::F32 | DataType::I32 | DataType::U32 => data.to_vec(),
        DataType::I64 => data
            .chunks_exact(4)
            .flat_map(|c| (i32::from_le_bytes(c.try_into().unwrap()) as i64).to_le_bytes())
            .collect(),
        DataType::Bool => data
            .chunks_exact(4)
            .map(|c| (u32::from_le_bytes(c.try_into().unwrap()) != 0) as u8)
            .collect(),
        other => return Err(Error::Unsupported(format!("download of dtype {other}"))),
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

    fn prepare(&self, module: Module) -> Result<Self::Session> {
        let kernels = self.kernels.clone();
        let module = onyxia_ir::inline_composites(module, &self.decompositions, &|name| {
            kernels.contains(name)
        })?;
        onyxia_ir::validate::validate(&module)?;
        let order = module.topo_order()?;

        // Liveness: the last step index that reads each value. Module
        // outputs (and inputs) are never freed within a run.
        let mut last_use: Vec<Option<usize>> = vec![Some(0); module.values.len()];
        for (step, &node_id) in order.iter().enumerate() {
            for &v in &module.node(node_id).inputs {
                last_use[v.index()] = Some(step);
            }
        }
        for (_, id) in module.outputs.iter().chain(module.inputs.iter()) {
            last_use[id.index()] = None;
        }

        // Upload constants once.
        let mem = Arc::new(MemCounter::default());
        let mut consts: HashMap<ValueId, GpuTensor> = HashMap::new();
        for id in module.value_ids() {
            let def = module.value(id);
            let Origin::Const(cid) = def.origin else {
                continue;
            };
            phys(def.ty.dtype)?; // fail early on unsupported dtypes
            let host = onyxia_ir::interp::const_tensor(&module, cid)?;
            let data = to_phys(&host)?;
            let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: def.name.as_deref(),
                size: phys_bytes(host.numel()),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&buffer, 0, &data);
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
            last_use,
            consts,
            kernels: self.kernels.clone(),
            pipelines: PipelineCache::new(if self.ctx.use_immediates {
                IMMEDIATE_SIZE
            } else {
                0
            }),
            pool: BufferPool::default(),
            mem,
            encoder: None,
            use_immediates: self.ctx.use_immediates,
            imm_buffers: Vec::new(),
            imm_free: Vec::new(),
            profiler: None,
        })
    }
}

/// A prepared wgpu session.
pub struct WgpuSession {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    module: Module,
    order: Vec<NodeId>,
    last_use: Vec<Option<usize>>,
    consts: HashMap<ValueId, GpuTensor>,
    kernels: crate::fused::KernelRegistry,
    pipelines: PipelineCache,
    pool: BufferPool,
    /// Live/peak byte accounting for every buffer this session creates.
    mem: Arc<MemCounter>,
    encoder: Option<wgpu::CommandEncoder>,
    /// False → params bind as a storage buffer instead of `set_immediates`
    /// (the web path; see `gpu.rs` module docs).
    use_immediates: bool,
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
}

impl WgpuSession {
    /// Buffer-pool statistics `(fresh_allocations, reuses)`.
    pub fn pool_stats(&self) -> (usize, usize) {
        (self.pool.allocations, self.pool.reuses)
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
        wgsl: &str,
        buffers: &[&wgpu::Buffer],
        imm: &Imm,
        size: usize,
    ) -> Result<()> {
        let (pipeline, layout) = self.pipelines.get_or_create(&self.device, label, wgsl)?;
        let imm_buf = self.imm_fallback_buffer(imm);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        if let Some(buf) = &imm_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: buffers.len() as u32,
                resource: buf.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        });
        let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
        let (wg, _x_stride) = dispatch_size(linear);
        self.encode_pass(label, &pipeline, &bind_group, imm, wg);
        self.imm_buffers.extend(imm_buf);
        Ok(())
    }

    /// Dispatch a row-reduction kernel: exactly `rows` workgroups.
    pub(crate) fn dispatch_rows(
        &mut self,
        label: &str,
        wgsl: &str,
        buffers: &[&wgpu::Buffer],
        imm: &Imm,
        rows: usize,
    ) -> Result<()> {
        let (pipeline, layout) = self.pipelines.get_or_create(&self.device, label, wgsl)?;
        let imm_buf = self.imm_fallback_buffer(imm);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        if let Some(buf) = &imm_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: buffers.len() as u32,
                resource: buf.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        });
        self.encode_pass(
            label,
            &pipeline,
            &bind_group,
            imm,
            [rows.max(1) as u32, 1, 1],
        );
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
        let ts = self
            .profiler
            .as_mut()
            .map(|p| p.begin_pass(&self.device, label));
        let timestamp_writes = ts.map(|(set, base)| wgpu::ComputePassTimestampWrites {
            query_set: self
                .profiler
                .as_ref()
                .expect("profiler present when ts is")
                .query_set(set),
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        });
        let encoder = self.encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("onyxia_batch"),
                })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if self.use_immediates {
            pass.set_immediates(0, imm.bytes());
        }
        pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
    }

    fn submit(&mut self) {
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
        let buffer =
            self.pool
                .acquire(&self.device, phys_bytes(shape.iter().product()), &self.mem);
        GpuTensor {
            buffer,
            dtype,
            shape,
        }
    }
}

/// Common immediate prefix: size + x_stride for the bounds check.
fn size_imm(size: usize) -> (Imm, usize) {
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
        let data = to_phys(tensor)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload"),
            size: phys_bytes(tensor.numel()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &data);
        Ok(GpuTensor {
            buffer: Arc::new(TrackedBuffer::new(buffer, &self.mem)),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
    }

    async fn run(&mut self, inputs: &[(&str, GpuTensor)]) -> Result<Vec<(String, GpuTensor)>> {
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

            // Release dead intermediates to the pool.
            for (vi, lu) in self.last_use.iter().enumerate() {
                if *lu == Some(step) {
                    if let Some(t) = regs[vi].take() {
                        if let Ok(buffer) = Arc::try_unwrap(t.buffer) {
                            self.pool.release(Arc::new(buffer));
                        }
                    }
                }
            }
        }
        self.submit();

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
        self.submit();
        let size = phys_bytes(tensor.numel());
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
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        from_phys(tensor.dtype, &tensor.shape, &bytes)
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
                let inputs: Vec<GpuTensor> = node
                    .inputs
                    .iter()
                    .map(|&v| {
                        regs[v.index()]
                            .clone()
                            .ok_or_else(|| Error::Runtime("input not materialized".into()))
                    })
                    .collect::<Result<_>>()?;
                let outs_meta: Vec<(DataType, Vec<usize>)> = node
                    .outputs
                    .iter()
                    .map(|&o| (self.module.value(o).ty.dtype, shapes[o.index()].clone()))
                    .collect();
                kernel.execute(self, &c.attrs, &inputs, &outs_meta)
            }
        }
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
                let (ts, td) = (phys(x.dtype)?, phys(out_dtype)?);
                let expr = cast_expr(ts, td, out_dtype);
                if expr == "v" {
                    // Same physical representation: alias.
                    return Ok(GpuTensor {
                        buffer: x.buffer,
                        dtype: out_dtype,
                        shape: out_shape,
                    });
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                self.dispatch(
                    &format!("cast_{ts}_{td}_{out_dtype}"),
                    &kernels::cast(ts, td, &expr),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── element-wise ─────────────────────────────────────────
            Prim::Unary(op) => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let (expr, needs_erf) = unary_expr(*op, t)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                self.dispatch(
                    &format!("unary_{}_{t}", prim.name()),
                    &kernels::unary(t, expr, needs_erf),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Binary(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(a.dtype)?;
                let expr = binary_expr(*op, t)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(a.shape.len() as u32)
                    .u(b.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&a.shape)
                    .arr8(&b.shape);
                self.dispatch(
                    &format!("binary_{}_{t}", prim.name()),
                    &kernels::binary(t, t, expr),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Compare(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(a.dtype)?;
                let expr = compare_expr(*op);
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(a.shape.len() as u32)
                    .u(b.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&a.shape)
                    .arr8(&b.shape);
                self.dispatch(
                    &format!("compare_{}_{t}", prim.name()),
                    &kernels::binary(t, "u32", expr),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Select => {
                let (c, a, b) = (input(0)?.clone(), input(1)?.clone(), input(2)?.clone());
                let t = phys(a.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
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
                    &format!("select_{t}"),
                    &kernels::select3(t),
                    &[&c.buffer, &a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── linear algebra ───────────────────────────────────────
            Prim::MatMul { trans_a, trans_b } => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(a.dtype)?;
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
                let stride_of = |batch_numel: usize, mat: usize, what: &str| -> Result<u32> {
                    if batch_numel == batch {
                        Ok(mat as u32)
                    } else if batch_numel == 1 {
                        Ok(0)
                    } else {
                        Err(Error::Unsupported(format!(
                            "matmul {what} batch broadcast pattern \
                             ({batch_numel} vs {batch})"
                        )))
                    }
                };
                let a_bs = stride_of(a.shape[..ar - 2].iter().product(), m * k, "lhs")?;
                let b_bs = stride_of(b.shape[..br - 2].iter().product(), k * n, "rhs")?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(m as u32)
                    .u(n as u32)
                    .u(k as u32)
                    .u(a_bs)
                    .u(b_bs)
                    .u(*trans_a as u32)
                    .u(*trans_b as u32);
                self.dispatch(
                    &format!("matmul_{t}"),
                    &kernels::matmul(t),
                    &[&a.buffer, &b.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Reduce { op, axes, .. } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let (init, combine, finalize) = reduce_exprs(*op, t);
                let mut mask = 0u32;
                let mut count = 1usize;
                for &a in axes {
                    mask |= 1 << a;
                    count *= x.shape[a];
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(x.shape.len() as u32)
                    .u(mask)
                    .u(count as u32)
                    .arr8(&x.shape);
                self.dispatch(
                    &format!("reduce_{}_{t}", prim.name()),
                    &kernels::reduce(t, init, combine, finalize),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            // ── data movement ────────────────────────────────────────
            Prim::Transpose { perm } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(x.shape.len() as u32)
                    .arr8(perm)
                    .arr8(&x.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("transpose_{t}"),
                    &kernels::transpose(t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Broadcast { .. } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(out.shape.len() as u32)
                    .u(x.shape.len() as u32)
                    .arr8(&out.shape)
                    .arr8(&x.shape);
                self.dispatch(
                    &format!("broadcast_{t}"),
                    &kernels::broadcast(t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Concat { axis } => {
                let out = self.alloc_out(out_dtype, out_shape.clone());
                let mut offset = 0usize;
                for i in 0..node.inputs.len() {
                    let x = input(i)?.clone();
                    let t = phys(x.dtype)?;
                    let (imm, size) = size_imm(x.numel());
                    let imm = imm
                        .u(x.shape.len() as u32)
                        .u(*axis as u32)
                        .u(offset as u32)
                        .arr8(&x.shape)
                        .arr8(&out_shape);
                    self.dispatch(
                        &format!("concat_{t}"),
                        &kernels::concat_emplace(t),
                        &[&x.buffer, &out.buffer],
                        &imm,
                        size,
                    )?;
                    offset += x.shape[*axis];
                }
                Ok(out)
            }

            Prim::Slice { specs } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
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
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(rank as u32)
                    .arr8(&starts.iter().map(|&s| s as usize).collect::<Vec<_>>())
                    .arr8_i(&steps)
                    .arr8(&x.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("slice_{t}"),
                    &kernels::slice(t),
                    &[&x.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Gather { axis } => {
                let (data, indices) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(data.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                let imm = imm
                    .u(*axis as u32)
                    .u(data.shape.len() as u32)
                    .u(indices.shape.len() as u32)
                    .arr8(&data.shape)
                    .arr8(&indices.shape)
                    .arr8(&out.shape);
                self.dispatch(
                    &format!("gather_{t}"),
                    &kernels::gather(t),
                    &[&data.buffer, &indices.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Scatter => {
                let (data, indices, updates) =
                    (input(0)?.clone(), input(1)?.clone(), input(2)?.clone());
                let t = phys(data.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                // Stage 1: copy data into out.
                let (imm, size) = size_imm(data.numel());
                self.dispatch(
                    &format!("copy_{t}"),
                    &kernels::copy(t),
                    &[&data.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                // Stage 2: scatter updates.
                let ir = indices.shape.len();
                let k = indices.shape[ir - 1];
                let slice_len: usize = data.shape[k..].iter().product();
                let (imm, size) = size_imm(updates.numel());
                let imm = imm
                    .u(k as u32)
                    .u(slice_len as u32)
                    .u(data.shape.len() as u32)
                    .arr8(&data.shape);
                self.dispatch(
                    &format!("scatter_{t}"),
                    &kernels::scatter(t),
                    &[&indices.buffer, &updates.buffer, &out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Iota { dtype, .. } => {
                let t = phys(*dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let (imm, size) = size_imm(out.numel());
                self.dispatch(
                    &format!("iota_{t}"),
                    &kernels::iota(t),
                    &[&out.buffer],
                    &imm,
                    size,
                )?;
                Ok(out)
            }

            Prim::Dequantize { .. } => Err(Error::Unsupported(
                "Dequantize kernel not yet implemented on the wgpu backend \
                 (needed for quantized models only)"
                    .into(),
            )),
        }
    }
}

// ─────────────────── expression tables ─────────────────────────────────

fn unary_expr(op: UnaryOp, t: &str) -> Result<(&'static str, bool)> {
    use UnaryOp::*;
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
        Not => {
            if t != "u32" {
                return Err(Error::DType("Not on non-bool".into()));
            }
            ("select(1u, 0u, v != 0u)", false)
        }
    })
}

fn binary_expr(op: BinaryOp, t: &str) -> Result<&'static str> {
    use BinaryOp::*;
    Ok(match (op, t) {
        (Add, _) => "av + bv",
        (Sub, _) => "av - bv",
        (Mul, _) => "av * bv",
        (Div, _) => "av / bv",
        (Pow, "f32") => "pow(av, bv)",
        (Pow, _) => {
            return Err(Error::Unsupported("integer pow on the wgpu backend".into()));
        }
        (Max, _) => "max(av, bv)",
        (Min, _) => "min(av, bv)",
        (And, _) => "u32((av != 0u) && (bv != 0u))",
        (Or, _) => "u32((av != 0u) || (bv != 0u))",
        (Xor, _) => "u32((av != 0u) != (bv != 0u))",
    })
}

fn compare_expr(op: CmpOp) -> &'static str {
    use CmpOp::*;
    match op {
        Eq => "u32(av == bv)",
        Ne => "u32(av != bv)",
        Lt => "u32(av < bv)",
        Le => "u32(av <= bv)",
        Gt => "u32(av > bv)",
        Ge => "u32(av >= bv)",
    }
}

fn reduce_exprs(op: ReduceOp, t: &str) -> (&'static str, &'static str, &'static str) {
    use ReduceOp::*;
    let is_f = t == "f32";
    match op {
        Sum => (if is_f { "0.0" } else { "0" }, "acc + v", "acc"),
        Mean => (
            if is_f { "0.0" } else { "0" },
            "acc + v",
            if is_f {
                "acc / f32(p.reduce_count)"
            } else {
                "acc / i32(p.reduce_count)"
            },
        ),
        Prod => (if is_f { "1.0" } else { "1" }, "acc * v", "acc"),
        Max => (
            if is_f { "-3.402823e38" } else { "-2147483647" },
            "max(acc, v)",
            "acc",
        ),
        Min => (
            if is_f { "3.402823e38" } else { "2147483647" },
            "min(acc, v)",
            "acc",
        ),
    }
}

/// Conversion expression for Cast, in physical types. `"v"` means the
/// physical bits are identical (alias, no dispatch).
fn cast_expr(src: &str, dst: &str, dst_logical: DataType) -> String {
    let to_bool = dst_logical == DataType::Bool;
    match (src, dst) {
        (s, d) if s == d && !to_bool => "v".to_string(),
        ("f32", "i32") => "i32(v)".to_string(),
        ("i32", "f32") => "f32(v)".to_string(),
        ("u32", "f32") => "f32(v)".to_string(),
        ("f32", "u32") if to_bool => "select(0u, 1u, v != 0.0)".to_string(),
        ("i32", "u32") if to_bool => "select(0u, 1u, v != 0)".to_string(),
        ("u32", "u32") if to_bool => "select(0u, 1u, v != 0u)".to_string(),
        ("u32", "i32") => "i32(v)".to_string(),
        ("f32", "u32") => "u32(v)".to_string(),
        ("i32", "u32") => "u32(v)".to_string(),
        (s, d) => format!("{d}({s}(v))"), // fallback; naga validates
    }
}
