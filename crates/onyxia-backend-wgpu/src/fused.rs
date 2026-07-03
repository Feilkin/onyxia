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
//! dispatch instead of ~5), `Gelu` (single elementwise pass),
//! `com.microsoft.RotaryEmbedding` (single elementwise pass with inline
//! cos/sin gather), and `com.microsoft.GroupQueryAttention` (three
//! dispatches: present-K/V concat + a chunked online-softmax attention
//! pass). Planned next, following the same pattern: `MatMulNBits`
//! (reference WGSL lives in `legacy-shaders/`).

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
        ("com.microsoft.RotaryEmbedding", Box::new(RotaryKernel)),
        ("com.microsoft.GemmaRotaryEmbedding", Box::new(RotaryKernel)),
        (
            "com.microsoft.GroupQueryAttention",
            Box::new(GroupQueryAttentionKernel),
        ),
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

/// Fused non-interleaved rotary embedding: one elementwise dispatch that
/// gathers cos/sin by position inline — replaces the decomposition's
/// gather/slice/mul/concat chain (~10 dispatches per call, twice per
/// layer).
struct RotaryKernel;

impl CompositeKernel for RotaryKernel {
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>> {
        let [x, pos, cos_cache, sin_cache] = inputs else {
            return Err(Error::InvalidGraph("RotaryEmbedding expects 4 inputs".into()));
        };
        if attrs.int_or("interleaved", 0)? != 0 {
            return Err(Error::Unsupported("fused rotary: interleaved=1".into()));
        }
        if x.dtype != DataType::F32 {
            return Err(Error::Unsupported("fused rotary is f32-only".into()));
        }
        let [_bsz, _seq, hidden] = x.shape[..] else {
            return Err(Error::Unsupported(format!(
                "fused rotary expects 3-D input, got rank {}",
                x.shape.len()
            )));
        };
        // Cache rows are [max_seq, cache_w]; the rotated width may be
        // narrower when `rotary_embedding_dim` says so.
        let cache_w = cos_cache.shape[1];
        let half_r = match attrs.int_or("rotary_embedding_dim", 0)? {
            0 => cache_w,
            r => r as usize / 2,
        };
        // num_heads=0 (the common export) means "infer from the cache",
        // mirroring the decomposition.
        let heads = match attrs.int_or("num_heads", 0)? {
            n if n > 0 => n as usize,
            _ if half_r > 0 && hidden % (2 * half_r) == 0 => hidden / (2 * half_r),
            _ => 1,
        };
        if heads == 0 || hidden % heads != 0 {
            return Err(Error::Shape(format!(
                "fused rotary: hidden {hidden} not divisible by num_heads {heads}"
            )));
        }
        let d = hidden / heads;
        if 2 * half_r > d {
            return Err(Error::Shape(format!(
                "fused rotary: rotary width {} exceeds head dim {d}",
                2 * half_r
            )));
        }
        let out = session.alloc_out(outs[0].0, outs[0].1.clone());
        let size = out.numel();
        let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
        let (_, x_stride) = crate::gpu::dispatch_size(linear);
        let imm = Imm::new()
            .u(size as u32)
            .u(x_stride)
            .u(hidden as u32)
            .u(d as u32)
            .u(half_r as u32)
            .u(cache_w as u32);
        session.dispatch(
            "fused_rotary_f32",
            &rotary_wgsl(),
            &[
                &x.buffer,
                &pos.buffer,
                &cos_cache.buffer,
                &sin_cache.buffer,
                &out.buffer,
            ],
            &imm,
            size,
        )?;
        Ok(vec![out])
    }
}

/// Fused GroupQueryAttention: three dispatches (present-K concat,
/// present-V concat, attention) instead of the ~20-dispatch
/// decomposition, and no materialized `[B,H,S,T]` score/mask tensors.
///
/// The attention kernel runs one workgroup per `(batch, head, query)`
/// and streams keys in shared-memory chunks with an online softmax
/// (flash-attention style), so context length is unbounded; only the
/// head dim is capped (≤ 256, one output lane per thread). Causal and
/// sliding-window masking match the decomposition. Like the
/// decomposition, sequences are assumed dense: the optional
/// `seqlens_k`/`total_sequence_length` inputs are ignored.
struct GroupQueryAttentionKernel;

impl CompositeKernel for GroupQueryAttentionKernel {
    fn execute(
        &self,
        session: &mut WgpuSession,
        attrs: &Attrs,
        inputs: &[GpuTensor],
        outs: &[(DataType, Vec<usize>)],
    ) -> Result<Vec<GpuTensor>> {
        let [q, k, v, past_k, past_v, ..] = inputs else {
            return Err(Error::InvalidGraph("GQA expects at least 5 inputs".into()));
        };
        if q.dtype != DataType::F32 {
            return Err(Error::Unsupported("fused GQA is f32-only".into()));
        }
        let heads = attrs.int("num_heads")? as usize;
        let kv_heads = attrs.int("kv_num_heads")? as usize;
        if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
            return Err(Error::Attribute(format!(
                "GQA num_heads={heads} must be a positive multiple of kv_num_heads={kv_heads}"
            )));
        }
        let [bsz, seq, hidden] = q.shape[..] else {
            return Err(Error::Unsupported("fused GQA expects 3-D query".into()));
        };
        if hidden % heads != 0 {
            return Err(Error::Shape(format!(
                "GQA hidden {hidden} not divisible by num_heads {heads}"
            )));
        }
        let d = hidden / heads;
        if d > 256 {
            return Err(Error::Unsupported(format!(
                "fused GQA head dim {d} exceeds 256 (one output lane per thread)"
            )));
        }
        let past = past_k.shape[2];
        let total = past + seq;
        if seq > 65535 || heads > 65535 || bsz > 65535 {
            return Err(Error::Unsupported(
                "fused GQA grid dimension exceeds 65535 workgroups".into(),
            ));
        }
        let window = attrs.int_or("local_window_size", -1)? as i32;
        let scale = match attrs.float_or("scale", 0.0)? {
            0.0 => 1.0 / (d as f64).sqrt(),
            s => s,
        } as f32;

        // Present caches: past ++ new, straight into BNSH.
        let present_k = session.alloc_out(outs[1].0, outs[1].1.clone());
        let present_v = session.alloc_out(outs[2].0, outs[2].1.clone());
        for (old, new, present) in [(past_k, k, &present_k), (past_v, v, &present_v)] {
            let size = present.numel();
            let linear = (size as u32).div_ceil(WORKGROUP_SIZE);
            let (_, x_stride) = crate::gpu::dispatch_size(linear);
            let imm = Imm::new()
                .u(size as u32)
                .u(x_stride)
                .u(kv_heads as u32)
                .u(past as u32)
                .u(total as u32)
                .u(seq as u32)
                .u(d as u32);
            session.dispatch(
                "fused_gqa_concat_f32",
                &gqa_concat_wgsl(),
                &[&old.buffer, &new.buffer, &present.buffer],
                &imm,
                size,
            )?;
        }

        let out = session.alloc_out(outs[0].0, outs[0].1.clone());
        let imm = Imm::new()
            .u(heads as u32)
            .u(kv_heads as u32)
            .u((heads / kv_heads) as u32)
            .u(seq as u32)
            .u(past as u32)
            .u(total as u32)
            .u(d as u32)
            .f(scale)
            .i(window);
        session.dispatch_grid(
            "fused_gqa_attention_f32",
            &gqa_attention_wgsl(),
            &[
                &q.buffer,
                &present_k.buffer,
                &present_v.buffer,
                &out.buffer,
            ],
            &imm,
            [seq as u32, heads as u32, bsz as u32],
        )?;
        Ok(vec![out, present_k, present_v])
    }
}

// ─────────────────────────── WGSL ──────────────────────────────────────

/// Build `present = past ++ new` in BNSH. `past` is `[B,KV,P,D]`, `new`
/// arrives in BSH (`[B,S,KV*D]`); one thread per present element.
fn gqa_concat_wgsl() -> String {
    "
struct P {
    size: u32, x_stride: u32,
    kv: u32, past: u32, total: u32, s: u32, d: u32,
}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> old: array<f32>;
@group(0) @binding(1) var<storage, read> new_kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * p.x_stride;
    if (idx >= p.size) { return; }
    let dd = idx % p.d;
    let t = (idx / p.d) % p.total;
    let kvh = (idx / (p.d * p.total)) % p.kv;
    let b = idx / (p.d * p.total * p.kv);
    if (t < p.past) {
        out[idx] = old[((b * p.kv + kvh) * p.past + t) * p.d + dd];
    } else {
        out[idx] = new_kv[(b * p.s + (t - p.past)) * p.kv * p.d + kvh * p.d + dd];
    }
}
"
    .to_string()
}

/// One workgroup per `(batch, head, query position)`; keys stream
/// through a shared-memory tile with an online (rescaling) softmax, so
/// any context length fits in the fixed tile. Query lives in shared
/// memory; each thread owns one output dim in the value phase.
fn gqa_attention_wgsl() -> String {
    "
struct P {
    heads: u32, kv: u32, group: u32,
    s: u32, past: u32, total: u32, d: u32,
    scale: f32, window: i32,
}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> pk: array<f32>;
@group(0) @binding(2) var<storage, read> pv: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> q_s: array<f32, 256>;
var<workgroup> tile: array<f32, 1024>;
var<workgroup> red: array<f32, 256>;
const NEG: f32 = -3.402823e38;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let sq = wg.x;
    let h = wg.y;
    let b = wg.z;
    let lane = lid.x;
    let kvh = h / p.group;
    let base = (b * p.kv + kvh) * p.total * p.d;
    let row = i32(p.past + sq);

    if (lane < p.d) {
        q_s[lane] = q[(b * p.s + sq) * p.heads * p.d + h * p.d + lane];
    }
    workgroupBarrier();

    // Online softmax state: running max, running sum, running context
    // for this lane's output dim. All lanes agree on m and l (both come
    // off red[0]), so the rescales are uniform.
    var m = NEG;
    var l = 0.0;
    var acc = 0.0;
    for (var c0 = 0u; c0 < p.total; c0 += 1024u) {
        let cn = min(1024u, p.total - c0);

        // Masked, scaled scores for this chunk of keys.
        for (var t = lane; t < cn; t += 256u) {
            let col = i32(c0 + t);
            var sc = 0.0;
            let kb = base + (c0 + t) * p.d;
            for (var dd = 0u; dd < p.d; dd += 1u) {
                sc += q_s[dd] * pk[kb + dd];
            }
            sc *= p.scale;
            let dead = col > row || (p.window > 0 && col <= row - p.window);
            tile[t] = select(sc, NEG, dead);
        }
        workgroupBarrier();

        // Chunk max.
        var lm = NEG;
        for (var t = lane; t < cn; t += 256u) {
            lm = max(lm, tile[t]);
        }
        red[lane] = lm;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lane < st) { red[lane] = max(red[lane], red[lane + st]); }
            workgroupBarrier();
        }
        let m_new = max(m, red[0]);
        workgroupBarrier();

        // Exponentiate in place + chunk sum. Fully-masked entries must
        // become exactly 0 even when the running max is still NEG
        // (exp(NEG - NEG) would be 1).
        var ls = 0.0;
        for (var t = lane; t < cn; t += 256u) {
            let e = select(exp(tile[t] - m_new), 0.0, tile[t] <= -3.0e38);
            tile[t] = e;
            ls += e;
        }
        red[lane] = ls;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lane < st) { red[lane] = red[lane] + red[lane + st]; }
            workgroupBarrier();
        }
        // While m is still NEG nothing has been accumulated; force the
        // rescale to 0 rather than trusting exp() on a -inf argument.
        let rescale = select(exp(m - m_new), 0.0, m <= -3.0e38);
        l = l * rescale + red[0];

        // Fold this chunk into the running context.
        if (lane < p.d) {
            acc = acc * rescale;
            for (var t = 0u; t < cn; t = t + 1u) {
                acc += tile[t] * pv[base + (c0 + t) * p.d + lane];
            }
        }
        m = m_new;
        workgroupBarrier(); // tile and red are rewritten next chunk
    }

    if (lane < p.d) {
        out[(b * p.s + sq) * p.heads * p.d + h * p.d + lane] = acc / l;
    }
}
"
    .to_string()
}

fn rotary_wgsl() -> String {
    "
struct P {
    size: u32, x_stride: u32,
    hidden: u32, d: u32, half_r: u32, cache_w: u32,
}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> pos_ids: array<i32>;
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(3) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * p.x_stride;
    if (idx >= p.size) { return; }
    let dd = (idx % p.hidden) % p.d;
    if (dd >= p.half_r * 2u) {
        out[idx] = x[idx]; // beyond the rotated width: pass through
        return;
    }
    // idx / hidden is b*S + s — exactly the flat position_ids index.
    let pos = u32(pos_ids[idx / p.hidden]);
    if (dd < p.half_r) {
        let c = cos_cache[pos * p.cache_w + dd];
        let s = sin_cache[pos * p.cache_w + dd];
        out[idx] = x[idx] * c - x[idx + p.half_r] * s;
    } else {
        let j = dd - p.half_r;
        let c = cos_cache[pos * p.cache_w + j];
        let s = sin_cache[pos * p.cache_w + j];
        out[idx] = x[idx] * c + x[idx - p.half_r] * s;
    }
}
"
    .to_string()
}

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
