//! CubeCL backend for Onyxia — the second-backend spike (plan milestone E).
//!
//! Implements **primitives only**: `supports()` returns false for every
//! composite, so legalization inlines all decompositions. This is the proof
//! that the backend contract is the primitive set — a backend with no
//! hand-written composite kernels runs any lowered model, including custom
//! ops it has never heard of.
//!
//! Kernels are written in CubeCL (`#[cube]` Rust, JIT-compiled by the
//! CubeCL runtime) instead of generated WGSL strings; execution runs on
//! `cubecl-wgpu` here, but nothing in this crate names a graphics API —
//! switching to `cubecl-cuda` is a type-parameter change.
//!
//! Spike simplifications (vs the wgpu backend): no fused composites, no
//! buffer liveness/pooling (CubeCL's own allocator handles reuse),
//! blocking readback (native only), and no Scatter/Dequantize/f16.

mod kernels;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use kernels::MAX_RANK;
use onyxia_ir::graph::{Module, NodeId, NodeKind, Origin, ValueId};
use onyxia_ir::interp::{Tensor, bind_shapes};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, UnaryOp};
use onyxia_ir::{DataType, Error, Result};
use std::collections::HashMap;

const CUBE_DIM: u32 = 256;

/// A device-resident tensor handle (CubeCL server handle + logical type).
#[derive(Clone)]
pub struct CubeTensor {
    handle: Handle,
    pub dtype: DataType,
    pub shape: Vec<usize>,
}

impl CubeTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Physical scalar for a logical dtype — same layout policy as the wgpu
/// backend (I64 stored as i32, Bool as u32; everything 4 bytes).
fn phys(dt: DataType) -> Result<&'static str> {
    Ok(match dt {
        DataType::F32 => "f32",
        DataType::I64 | DataType::I32 => "i32",
        DataType::U32 | DataType::Bool => "u32",
        other => {
            return Err(Error::Unsupported(format!(
                "dtype {other} on the cubecl backend"
            )));
        }
    })
}

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
                            "i64 value {v} does not fit the cubecl backend's 32-bit storage"
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

/// Pack shape params: values then 8-slot shape blocks, mirroring the wgpu
/// backend's `Imm` layout (minus x_stride — `ABSOLUTE_POS` linearizes).
#[derive(Default)]
struct P(Vec<u32>);

impl P {
    fn new(size: usize) -> Self {
        Self(vec![size as u32])
    }
    fn u(mut self, v: u32) -> Self {
        self.0.push(v);
        self
    }
    fn arr8(mut self, dims: &[usize]) -> Self {
        for i in 0..MAX_RANK {
            self.0.push(dims.get(i).copied().unwrap_or(0) as u32);
        }
        self
    }
    fn arr8_i(mut self, vals: &[i64]) -> Self {
        for i in 0..MAX_RANK {
            self.0
                .push((vals.get(i).copied().unwrap_or(0) as i32) as u32);
        }
        self
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// The CubeCL backend.
pub struct CubeclBackend {
    device: WgpuDevice,
    decompositions: onyxia_ir::DecompositionRegistry,
}

impl CubeclBackend {
    pub fn new() -> Self {
        Self {
            device: WgpuDevice::DefaultDevice,
            decompositions: onyxia_ir::standard_decompositions(),
        }
    }
}

impl Default for CubeclBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl onyxia_ir::Backend for CubeclBackend {
    type Session = CubeclSession;

    /// Primitives only — every composite legalizes through its
    /// decomposition. That asymmetry with the wgpu backend is the point.
    fn supports(&self, _composite: &str) -> bool {
        false
    }

    fn prepare(&self, module: Module) -> Result<Self::Session> {
        let module = onyxia_ir::inline_composites(module, &self.decompositions, &|_| false)?;
        onyxia_ir::validate::validate(&module)?;
        let order = module.topo_order()?;

        let client = WgpuRuntime::client(&self.device);

        // Upload constants once.
        let mut consts: HashMap<ValueId, CubeTensor> = HashMap::new();
        for id in module.value_ids() {
            let def = module.value(id);
            let Origin::Const(cid) = def.origin else {
                continue;
            };
            phys(def.ty.dtype)?; // fail early on unsupported dtypes
            let host = onyxia_ir::interp::const_tensor(&module, cid)?;
            let data = to_phys(&host)?;
            consts.insert(
                id,
                CubeTensor {
                    handle: client.create_from_slice(&data),
                    dtype: def.ty.dtype,
                    shape: host.shape().to_vec(),
                },
            );
        }

        Ok(CubeclSession {
            client,
            module,
            order,
            consts,
        })
    }
}

/// A prepared CubeCL session.
pub struct CubeclSession {
    client: ComputeClient<WgpuRuntime>,
    module: Module,
    order: Vec<NodeId>,
    consts: HashMap<ValueId, CubeTensor>,
}

#[async_trait::async_trait(?Send)]
impl onyxia_ir::Session for CubeclSession {
    type Tensor = CubeTensor;

    fn upload(&mut self, tensor: &Tensor) -> Result<CubeTensor> {
        let data = to_phys(tensor)?;
        // Zero-element tensors (e.g. an empty KV cache at first prefill)
        // still need a live handle to bind.
        let handle = if data.is_empty() {
            self.client.empty(4)
        } else {
            self.client.create_from_slice(&data)
        };
        Ok(CubeTensor {
            handle,
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
    }

    async fn run(&mut self, inputs: &[(&str, CubeTensor)]) -> Result<Vec<(String, CubeTensor)>> {
        let described: Vec<(&str, DataType, &[usize])> = inputs
            .iter()
            .map(|(n, t)| (*n, t.dtype, t.shape.as_slice()))
            .collect();
        let bindings = bind_shapes(&self.module, &described)?;

        let shapes: Vec<Vec<usize>> = self
            .module
            .values
            .iter()
            .map(|def| {
                def.ty
                    .shape
                    .eval(&bindings)
                    .map_err(|e| Error::Binding(format!("cannot resolve shape: {e}")))
            })
            .collect::<Result<_>>()?;

        let mut regs: Vec<Option<CubeTensor>> = vec![None; self.module.values.len()];
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

        for &node_id in &self.order.clone() {
            let outs = self.run_node(node_id, &regs, &shapes, &bindings)?;
            for (out, &out_id) in outs.into_iter().zip(&self.module.node(node_id).outputs) {
                regs[out_id.index()] = Some(out);
            }
        }

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

    async fn download(&mut self, tensor: &CubeTensor) -> Result<Tensor> {
        let bytes = self
            .client
            .read_one(tensor.handle.clone())
            .map_err(|e| Error::Runtime(format!("cubecl readback failed: {e:?}")))?;
        from_phys(tensor.dtype, &tensor.shape, &bytes)
    }
}

/// Launch geometry for `size` output elements, one thread each.
fn geometry(size: usize) -> (CubeCount, CubeDim) {
    let cubes = (size as u32).div_ceil(CUBE_DIM).max(1);
    let x = cubes.min(65535);
    let y = cubes.div_ceil(65535);
    (CubeCount::Static(x, y, 1), CubeDim::new_1d(CUBE_DIM))
}

/// Launch a data-movement kernel generic over the physical element type.
/// Exactly one arm runs; args are constructed inside the selected arm.
macro_rules! launch_phys {
    ($t:expr, $k:ident, $client:expr, $count:expr, $dim:expr, [$($arg:expr),* $(,)?]) => {
        match $t {
            "f32" => unsafe {
                kernels::$k::launch_unchecked::<f32, WgpuRuntime>(
                    $client, $count, $dim, $($arg),*)
            },
            "i32" => unsafe {
                kernels::$k::launch_unchecked::<i32, WgpuRuntime>(
                    $client, $count, $dim, $($arg),*)
            },
            _ => unsafe {
                kernels::$k::launch_unchecked::<u32, WgpuRuntime>(
                    $client, $count, $dim, $($arg),*)
            },
        }
    };
}

impl CubeclSession {
    fn alloc_out(&self, dtype: DataType, shape: Vec<usize>) -> CubeTensor {
        let numel: usize = shape.iter().product();
        CubeTensor {
            handle: self.client.empty(numel.max(1) * 4),
            dtype,
            shape,
        }
    }

    fn params(&self, p: P) -> (Handle, usize) {
        let len = p.len();
        (self.client.create_from_slice(bytemuck::cast_slice(&p.0)), len)
    }

    fn arg(&self, t: &CubeTensor) -> ArrayArg<WgpuRuntime> {
        unsafe { ArrayArg::from_raw_parts(t.handle.clone(), t.numel().max(1)) }
    }

    fn parg(&self, p: &(Handle, usize)) -> ArrayArg<WgpuRuntime> {
        unsafe { ArrayArg::from_raw_parts(p.0.clone(), p.1) }
    }

    fn run_node(
        &mut self,
        node_id: NodeId,
        regs: &[Option<CubeTensor>],
        shapes: &[Vec<usize>],
        bindings: &onyxia_ir::Bindings,
    ) -> Result<Vec<CubeTensor>> {
        let node = self.module.node(node_id).clone();
        let NodeKind::Prim(prim) = &node.kind else {
            return Err(Error::Unsupported(
                "composite reached the cubecl executor (legalization inlines all)".into(),
            ));
        };
        let input = |i: usize| -> Result<&CubeTensor> {
            regs[node.inputs[i].index()]
                .as_ref()
                .ok_or_else(|| Error::Runtime("input not materialized".into()))
        };
        let out_id = node.outputs[0];
        let out_shape = shapes[out_id.index()].clone();
        let out_dtype = self.module.value(out_id).ty.dtype;
        if out_shape.len() > MAX_RANK {
            return Err(Error::Unsupported(format!(
                "rank {} exceeds the kernel maximum of {MAX_RANK}",
                out_shape.len()
            )));
        }

        let t = match prim {
            // ── zero-copy ────────────────────────────────────────────
            Prim::Reshape { .. } => {
                let x = input(0)?;
                CubeTensor {
                    handle: x.handle.clone(),
                    dtype: out_dtype,
                    shape: out_shape,
                }
            }

            Prim::Cast { .. } => {
                let x = input(0)?.clone();
                let (ts, td) = (phys(x.dtype)?, phys(out_dtype)?);
                if ts == td && out_dtype != DataType::Bool {
                    // Same physical representation: alias.
                    return Ok(vec![CubeTensor {
                        handle: x.handle,
                        dtype: out_dtype,
                        shape: out_shape,
                    }]);
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(P::new(size));
                let (count, dim) = geometry(size);
                match (ts, td, out_dtype) {
                    (_, _, DataType::Bool) => {
                        return Err(Error::Unsupported(
                            "cast to Bool on the cubecl backend (spike)".into(),
                        ));
                    }
                    ("f32", "i32", _) => unsafe {
                        kernels::cast_f32_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    ("i32", "f32", _) => unsafe {
                        kernels::cast_i32_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    ("u32", "f32", _) => unsafe {
                        kernels::cast_u32_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    ("u32", "i32", _) => unsafe {
                        kernels::cast_u32_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    (a, b, _) => {
                        return Err(Error::Unsupported(format!(
                            "cast {a} → {b} on the cubecl backend (spike)"
                        )));
                    }
                }
                out
            }

            // ── element-wise ─────────────────────────────────────────
            Prim::Unary(op) => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(P::new(size));
                let (count, dim) = geometry(size);
                match (t, op) {
                    ("u32", UnaryOp::Not) => unsafe {
                        kernels::not_u32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    ("f32", _) => unsafe {
                        kernels::unary_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                            unary_code(*op)?,
                        )
                    },
                    ("i32", UnaryOp::Neg | UnaryOp::Abs) => unsafe {
                        kernels::unary_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&x),
                            self.arg(&out),
                            self.parg(&p),
                            unary_code(*op)?,
                        )
                    },
                    (t, op) => {
                        return Err(Error::Unsupported(format!("unary {op:?} on {t}")));
                    }
                }
                out
            }

            Prim::Binary(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(a.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(out.shape.len() as u32)
                        .u(a.shape.len() as u32)
                        .u(b.shape.len() as u32)
                        .arr8(&out.shape)
                        .arr8(&a.shape)
                        .arr8(&b.shape),
                );
                let (count, dim) = geometry(size);
                let code = binary_code(*op, t)?;
                match t {
                    "f32" => unsafe {
                        kernels::binary_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&a),
                            self.arg(&b),
                            self.arg(&out),
                            self.parg(&p),
                            code,
                        )
                    },
                    "i32" => unsafe {
                        kernels::binary_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&a),
                            self.arg(&b),
                            self.arg(&out),
                            self.parg(&p),
                            code,
                        )
                    },
                    _ => unsafe {
                        kernels::binary_u32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&a),
                            self.arg(&b),
                            self.arg(&out),
                            self.parg(&p),
                            code,
                        )
                    },
                }
                out
            }

            Prim::Compare(op) => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(a.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(out.shape.len() as u32)
                        .u(a.shape.len() as u32)
                        .u(b.shape.len() as u32)
                        .arr8(&out.shape)
                        .arr8(&a.shape)
                        .arr8(&b.shape),
                );
                let (count, dim) = geometry(size);
                let code = cmp_code(*op);
                match t {
                    "f32" => unsafe {
                        kernels::compare_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&a),
                            self.arg(&b),
                            self.arg(&out),
                            self.parg(&p),
                            code,
                        )
                    },
                    _ => unsafe {
                        kernels::compare_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&a),
                            self.arg(&b),
                            self.arg(&out),
                            self.parg(&p),
                            code,
                        )
                    },
                }
                out
            }

            Prim::Select => {
                let (c, a, b) = (input(0)?.clone(), input(1)?.clone(), input(2)?.clone());
                let t = phys(a.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(out.shape.len() as u32)
                        .u(c.shape.len() as u32)
                        .u(a.shape.len() as u32)
                        .u(b.shape.len() as u32)
                        .arr8(&out.shape)
                        .arr8(&c.shape)
                        .arr8(&a.shape)
                        .arr8(&b.shape),
                );
                let (count, dim) = geometry(size);
                launch_phys!(
                    t,
                    select3,
                    &self.client,
                    count,
                    dim,
                    [
                        self.arg(&c),
                        self.arg(&a),
                        self.arg(&b),
                        self.arg(&out),
                        self.parg(&p)
                    ]
                );
                out
            }

            // ── linear algebra ───────────────────────────────────────
            Prim::MatMul { trans_a, trans_b } => {
                let (a, b) = (input(0)?.clone(), input(1)?.clone());
                if phys(a.dtype)? != "f32" {
                    return Err(Error::Unsupported("non-f32 matmul on cubecl".into()));
                }
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
                let stride_of = |batch_numel: usize, mat: usize| -> Result<u32> {
                    if batch_numel == batch {
                        Ok(mat as u32)
                    } else if batch_numel == 1 {
                        Ok(0)
                    } else {
                        Err(Error::Unsupported(format!(
                            "matmul batch broadcast pattern ({batch_numel} vs {batch})"
                        )))
                    }
                };
                let a_bs = stride_of(a.shape[..ar - 2].iter().product(), m * k)?;
                let b_bs = stride_of(b.shape[..br - 2].iter().product(), k * n)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(m as u32)
                        .u(n as u32)
                        .u(k as u32)
                        .u(a_bs)
                        .u(b_bs)
                        .u(*trans_a as u32)
                        .u(*trans_b as u32),
                );
                let (count, dim) = geometry(size);
                unsafe {
                    kernels::matmul_f32::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        count,
                        dim,
                        self.arg(&a),
                        self.arg(&b),
                        self.arg(&out),
                        self.parg(&p),
                    )
                }
                out
            }

            Prim::Reduce { op, axes, .. } => {
                let x = input(0)?.clone();
                if phys(x.dtype)? != "f32" {
                    return Err(Error::Unsupported("non-f32 reduce on cubecl".into()));
                }
                let mut mask = 0u32;
                let mut count_r = 1usize;
                for &a in axes {
                    mask |= 1 << a;
                    count_r *= x.shape[a];
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(x.shape.len() as u32)
                        .u(mask)
                        .u(count_r as u32)
                        .arr8(&x.shape),
                );
                let (count, dim) = geometry(size);
                unsafe {
                    kernels::reduce_f32::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        count,
                        dim,
                        self.arg(&x),
                        self.arg(&out),
                        self.parg(&p),
                        reduce_code(*op),
                    )
                }
                out
            }

            // ── data movement ────────────────────────────────────────
            Prim::Transpose { perm } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(x.shape.len() as u32)
                        .arr8(perm)
                        .arr8(&x.shape)
                        .arr8(&out.shape),
                );
                let (count, dim) = geometry(size);
                launch_phys!(
                    t,
                    transpose,
                    &self.client,
                    count,
                    dim,
                    [self.arg(&x), self.arg(&out), self.parg(&p)]
                );
                out
            }

            Prim::Broadcast { .. } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(out.shape.len() as u32)
                        .u(x.shape.len() as u32)
                        .arr8(&out.shape)
                        .arr8(&x.shape),
                );
                let (count, dim) = geometry(size);
                launch_phys!(
                    t,
                    broadcast,
                    &self.client,
                    count,
                    dim,
                    [self.arg(&x), self.arg(&out), self.parg(&p)]
                );
                out
            }

            Prim::Concat { axis } => {
                let out = self.alloc_out(out_dtype, out_shape.clone());
                let mut offset = 0usize;
                for i in 0..node.inputs.len() {
                    let x = input(i)?.clone();
                    let t = phys(x.dtype)?;
                    let size = x.numel();
                    if size > 0 {
                        let p = self.params(
                            P::new(size)
                                .u(x.shape.len() as u32)
                                .u(*axis as u32)
                                .u(offset as u32)
                                .arr8(&x.shape)
                                .arr8(&out_shape),
                        );
                        let (count, dim) = geometry(size);
                        launch_phys!(
                            t,
                            concat_emplace,
                            &self.client,
                            count,
                            dim,
                            [self.arg(&x), self.arg(&out), self.parg(&p)]
                        );
                    }
                    offset += x.shape[*axis];
                }
                out
            }

            Prim::Slice { specs } => {
                let x = input(0)?.clone();
                let t = phys(x.dtype)?;
                let rank = x.shape.len();
                let mut starts = vec![0usize; rank];
                let mut steps = vec![1i64; rank];
                for spec in specs {
                    starts[spec.axis] = spec.start.eval(bindings)? as usize;
                    steps[spec.axis] = spec.step;
                }
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(rank as u32)
                        .arr8(&starts)
                        .arr8_i(&steps)
                        .arr8(&x.shape)
                        .arr8(&out.shape),
                );
                let (count, dim) = geometry(size);
                launch_phys!(
                    t,
                    slice_copy,
                    &self.client,
                    count,
                    dim,
                    [self.arg(&x), self.arg(&out), self.parg(&p)]
                );
                out
            }

            Prim::Gather { axis } => {
                let (data, indices) = (input(0)?.clone(), input(1)?.clone());
                let t = phys(data.dtype)?;
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(
                    P::new(size)
                        .u(*axis as u32)
                        .u(data.shape.len() as u32)
                        .u(indices.shape.len() as u32)
                        .arr8(&data.shape)
                        .arr8(&indices.shape)
                        .arr8(&out.shape),
                );
                let (count, dim) = geometry(size);
                launch_phys!(
                    t,
                    gather,
                    &self.client,
                    count,
                    dim,
                    [
                        self.arg(&data),
                        self.arg(&indices),
                        self.arg(&out),
                        self.parg(&p)
                    ]
                );
                out
            }

            Prim::Iota { dtype, .. } => {
                let out = self.alloc_out(out_dtype, out_shape);
                let size = out.numel();
                let p = self.params(P::new(size));
                let (count, dim) = geometry(size);
                match phys(*dtype)? {
                    "f32" => unsafe {
                        kernels::iota_f32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                    _ => unsafe {
                        kernels::iota_i32::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            count,
                            dim,
                            self.arg(&out),
                            self.parg(&p),
                        )
                    },
                }
                out
            }

            other => {
                return Err(Error::Unsupported(format!(
                    "primitive {} on the cubecl backend (spike)",
                    other.name()
                )));
            }
        };
        Ok(vec![t])
    }
}

fn binary_code(op: BinaryOp, t: &str) -> Result<u32> {
    Ok(match (op, t) {
        (BinaryOp::Add, _) => kernels::OP_ADD,
        (BinaryOp::Sub, _) => kernels::OP_SUB,
        (BinaryOp::Mul, _) => kernels::OP_MUL,
        (BinaryOp::Div, _) => kernels::OP_DIV,
        (BinaryOp::Pow, "f32") => kernels::OP_POW,
        (BinaryOp::Max, _) => kernels::OP_MAX,
        (BinaryOp::Min, _) => kernels::OP_MIN,
        (BinaryOp::And, "u32") => kernels::OP_AND,
        (BinaryOp::Or, "u32") => kernels::OP_OR,
        (BinaryOp::Xor, "u32") => kernels::OP_XOR,
        (op, t) => return Err(Error::Unsupported(format!("binary {op:?} on {t}"))),
    })
}

fn cmp_code(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => kernels::CMP_EQ,
        CmpOp::Ne => kernels::CMP_NE,
        CmpOp::Lt => kernels::CMP_LT,
        CmpOp::Le => kernels::CMP_LE,
        CmpOp::Gt => kernels::CMP_GT,
        CmpOp::Ge => kernels::CMP_GE,
    }
}

fn unary_code(op: UnaryOp) -> Result<u32> {
    Ok(match op {
        UnaryOp::Neg => kernels::UN_NEG,
        UnaryOp::Sqrt => kernels::UN_SQRT,
        UnaryOp::Rsqrt => kernels::UN_RSQRT,
        UnaryOp::Exp => kernels::UN_EXP,
        UnaryOp::Log => kernels::UN_LOG,
        UnaryOp::Cos => kernels::UN_COS,
        UnaryOp::Sin => kernels::UN_SIN,
        UnaryOp::Tanh => kernels::UN_TANH,
        UnaryOp::Erf => kernels::UN_ERF,
        UnaryOp::Abs => kernels::UN_ABS,
        UnaryOp::Floor => kernels::UN_FLOOR,
        UnaryOp::Ceil => kernels::UN_CEIL,
        // Not is Bool-only; it dispatches to `not_u32` before reaching here.
        UnaryOp::Not => {
            return Err(Error::Unsupported("unary Not on a non-Bool tensor".into()));
        }
    })
}

fn reduce_code(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => kernels::RED_SUM,
        ReduceOp::Mean => kernels::RED_MEAN,
        ReduceOp::Max => kernels::RED_MAX,
        ReduceOp::Min => kernels::RED_MIN,
        ReduceOp::Prod => kernels::RED_PROD,
    }
}
