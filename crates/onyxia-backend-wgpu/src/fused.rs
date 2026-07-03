//! Fused composite kernels — the backend's half of the two-registry design
//! (see the repository's ARCHITECTURE.md).
//!
//! A composite whose name is registered here survives legalization intact
//! and executes through its [`CompositeKernel`] instead of its
//! decomposition. Every kernel added here must pass a differential test
//! against the decomposition running on the same device
//! (`tests/gpu_test.rs::fused_kernels_match_decompositions`) — the
//! decomposition is the specification.
//!
//! Currently registered: `Softmax` and `SimplifiedLayerNormalization`
//! (one-workgroup-per-row with shared-memory tree reductions — one
//! dispatch instead of ~5), and `Gelu` (single elementwise pass).
//! Planned next, following the same pattern:
//! `com.microsoft.GroupQueryAttention`, `RotaryEmbedding`, `MatMulNBits`
//! (reference WGSL for all three lives in `legacy-shaders/`).

use crate::gpu::WORKGROUP_SIZE;
use crate::kernels::{self, Imm};
use crate::session::{GpuTensor, WgpuSession};
use onyxia_ir::{Attrs, DataType, Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// A hand-written kernel for one composite op.
///
/// `Send + Sync` so the registry can be shared (`Arc`) across threads;
/// kernels are stateless — per-dispatch state lives in the session.
pub trait CompositeKernel: Send + Sync {
    /// Execute the composite. `outs` carries `(dtype, shape)` per declared
    /// output; the kernel allocates and returns matching tensors.
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>>;
}

/// Shared, immutable registry of fused kernels, keyed by composite name.
#[derive(Clone, Default)]
pub struct KernelRegistry {
    map: Arc<HashMap<String, Box<dyn CompositeKernel>>>,
}

impl KernelRegistry {
    /// Build from a list of `(name, kernel)` pairs.
    pub fn from_kernels(kernels: Vec<(&str, Box<dyn CompositeKernel>)>) -> Self {
        Self {
            map: Arc::new(
                kernels
                    .into_iter()
                    .map(|(n, k)| (n.to_string(), k))
                    .collect(),
            ),
        }
    }

    /// Whether a kernel is registered for `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Look up a kernel.
    pub fn get(&self, name: &str) -> Option<&dyn CompositeKernel> {
        self.map.get(name).map(|k| k.as_ref())
    }
}

/// The fused kernels shipped with the wgpu backend.
pub fn standard_kernels() -> KernelRegistry {
    KernelRegistry::from_kernels(vec![
        ("Softmax", Box::new(SoftmaxKernel)),
        ("SimplifiedLayerNormalization", Box::new(RmsNormKernel)),
        ("Gelu", Box::new(GeluKernel)),
    ])
}

/// Rows must fit a 1-D dispatch for the row-reduction kernels.
fn check_rows(rows: usize, what: &str) -> Result<()> {
    if rows > 65535 {
        return Err(Error::Unsupported(format!(
            "{what}: {rows} rows exceed the 1-D dispatch limit (needs 2-D row kernels)"
        )));
    }
    Ok(())
}

/// Fused last-axis softmax: one workgroup per row, tree reductions in
/// shared memory, single dispatch.
struct SoftmaxKernel;

impl CompositeKernel for SoftmaxKernel {
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>> {
        let [x] = inputs else {
            return Err(Error::InvalidGraph("Softmax expects 1 input".into()));
        };
        let Some(&cols) = x.shape.last() else {
            return Err(Error::Unsupported("fused softmax on a scalar".into()));
        };
        let axis = attrs.int("axis")? as usize;
        if axis != x.shape.len() - 1 {
            return Err(Error::Unsupported(
                "fused softmax requires the last axis (lowering normalizes \
                 Gemma's; fall back to the decomposition otherwise)"
                    .into(),
            ));
        }
        if x.dtype != DataType::F32 {
            return Err(Error::Unsupported("fused softmax is f32-only".into()));
        }
        let rows = x.numel() / cols.max(1);
        check_rows(rows, "softmax")?;
        let out = session.alloc_out(outs[0].0, outs[0].1.clone());
        let imm = Imm::new().u(rows as u32).u(cols as u32);
        session.dispatch_rows(
            "fused_softmax_row_f32",
            &softmax_row_wgsl(),
            &[&x.buffer, &out.buffer],
            &imm,
            rows,
        )?;
        Ok(vec![out])
    }
}

/// Fused RMS norm (SimplifiedLayerNormalization): one workgroup per row.
struct RmsNormKernel;

impl CompositeKernel for RmsNormKernel {
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>> {
        let [x, w] = inputs else {
            return Err(Error::InvalidGraph(
                "SimplifiedLayerNormalization expects 2 inputs".into(),
            ));
        };
        if x.dtype != DataType::F32 {
            return Err(Error::Unsupported("fused rms-norm is f32-only".into()));
        }
        let Some(&cols) = x.shape.last() else {
            return Err(Error::Unsupported("fused rms-norm on a scalar".into()));
        };
        let rows = x.numel() / cols.max(1);
        check_rows(rows, "rms-norm")?;
        let eps = attrs.float_or("epsilon", 1e-5)? as f32;
        let out = session.alloc_out(outs[0].0, outs[0].1.clone());
        let imm = Imm::new().u(rows as u32).u(cols as u32).f(eps);
        session.dispatch_rows(
            "fused_rmsnorm_row_f32",
            &rmsnorm_row_wgsl(),
            &[&x.buffer, &w.buffer, &out.buffer],
            &imm,
            rows,
        )?;
        Ok(vec![out])
    }
}

/// Fused GELU: a single elementwise pass through the generated unary
/// kernel machinery (exact and tanh forms).
struct GeluKernel;

impl CompositeKernel for GeluKernel {
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>> {
        let [x] = inputs else {
            return Err(Error::InvalidGraph("Gelu expects 1 input".into()));
        };
        if x.dtype != DataType::F32 {
            return Err(Error::Unsupported("fused gelu is f32-only".into()));
        }
        let tanh_form = attrs.str("approximate").unwrap_or("none") == "tanh";
        let (label, expr) = if tanh_form {
            (
                "fused_gelu_tanh_f32",
                "0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)))",
            )
        } else {
            ("fused_gelu_f32", "0.5 * v * (1.0 + erf(v * 0.7071067812))")
        };
        let out = session.alloc_out(outs[0].0, outs[0].1.clone());
        let size = out.numel();
        let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
        let (_, x_stride) = crate::gpu::dispatch_size(linear);
        let imm = Imm::new().u(size as u32).u(x_stride);
        session.dispatch(
            label,
            &kernels::unary("f32", expr, !tanh_form),
            &[&x.buffer, &out.buffer],
            &imm,
            size,
        )?;
        Ok(vec![out])
    }
}

// ─────────────────────────── WGSL ──────────────────────────────────────

fn softmax_row_wgsl() -> String {
    "
struct P { rows: u32, cols: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let base = row * p.cols;

    // Row max.
    var m = -3.402823e38;
    for (var i = lid.x; i < p.cols; i = i + 256u) {
        m = max(m, x[base + i]);
    }
    scratch[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier();

    // Sum of exponentials.
    var sum = 0.0;
    for (var i = lid.x; i < p.cols; i = i + 256u) {
        sum = sum + exp(x[base + i] - row_max);
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    let row_sum = scratch[0];

    // Normalize.
    for (var i = lid.x; i < p.cols; i = i + 256u) {
        out[base + i] = exp(x[base + i] - row_max) / row_sum;
    }
}
"
    .to_string()
}

fn rmsnorm_row_wgsl() -> String {
    "
struct P { rows: u32, cols: u32, eps: f32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let base = row * p.cols;

    // Mean of squares.
    var sum = 0.0;
    for (var i = lid.x; i < p.cols; i = i + 256u) {
        let v = x[base + i];
        sum = sum + v * v;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    let inv = inverseSqrt(scratch[0] / f32(p.cols) + p.eps);

    for (var i = lid.x; i < p.cols; i = i + 256u) {
        out[base + i] = x[base + i] * inv * w[i];
    }
}
"
    .to_string()
}
