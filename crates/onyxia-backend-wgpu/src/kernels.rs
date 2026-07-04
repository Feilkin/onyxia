//! Generated WGSL kernels for the primitive set.
//!
//! One uniform scheme: each kernel is a template specialized by physical
//! scalar type and operation expression, generated as plain WGSL text and
//! compiled through naga's front-end. All shape information travels in
//! immediates (push constants) as `u32`s, ranks are capped at
//! [`MAX_RANK`], and every kernel runs one thread per output element —
//! correctness-first; fused/tiled kernels come later via the composite
//! kernel registry.
//!
//! **Physical types**: the GPU side knows `f32`, `i32`, `u32` only.
//! Logical `I64` is stored as `i32` (range-checked at upload), `Bool` as
//! `u32`. This mapping is private to this backend.

use crate::gpu::WORKGROUP_SIZE;

/// Maximum tensor rank supported by the generated kernels.
pub const MAX_RANK: usize = 8;

/// Immediates byte builder. Fields are packed in declaration order; all
/// fields are 4-byte scalars or `array<u32, 8>`, so layout is trivially
/// sequential.
#[derive(Default)]
pub struct Imm(Vec<u8>);

impl Imm {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn u(mut self, v: u32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn i(mut self, v: i32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    /// A shape (or similar) as a fixed `array<u32, 8>`, zero-padded.
    pub fn arr8(mut self, dims: &[usize]) -> Self {
        for i in 0..MAX_RANK {
            let v = dims.get(i).copied().unwrap_or(0) as u32;
            self.0.extend(v.to_le_bytes());
        }
        self
    }
    /// Like [`arr8`](Self::arr8) but signed (bitcast on the WGSL side).
    pub fn arr8_i(mut self, vals: &[i64]) -> Self {
        for i in 0..MAX_RANK {
            let v = vals.get(i).copied().unwrap_or(0) as i32;
            self.0.extend(v.to_le_bytes());
        }
        self
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }
}

fn header() -> String {
    format!(
        "const WG_SIZE: u32 = {WORKGROUP_SIZE}u;\n\
         fn linear_idx(gid: vec3<u32>, x_stride: u32) -> u32 {{\n\
             return gid.x + gid.y * x_stride;\n\
         }}\n"
    )
}

/// Right-aligned broadcast source index (shared by several templates).
const SRC_INDEX: &str = "
fn src_index(out_idx: u32, out_shape: array<u32,8>, out_rank: u32,
             in_shape: array<u32,8>, in_rank: u32) -> u32 {
    var rem = out_idx;
    var idx = 0u;
    var stride = 1u;
    for (var k = 0u; k < out_rank; k = k + 1u) {
        let d = out_rank - 1u - k;
        let coord = rem % out_shape[d];
        rem = rem / out_shape[d];
        if (k < in_rank) {
            let di = in_rank - 1u - k;
            let c = select(coord, 0u, in_shape[di] == 1u);
            idx = idx + c * stride;
            stride = stride * in_shape[di];
        }
    }
    return idx;
}
";

const ERF: &str = "
fn erf(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}
";

/// Element-wise binary op with full N-D broadcasting.
/// Bindings: 0=a, 1=b, 2=out. Immediates: size, x_stride, out_rank,
/// a_rank, b_rank, out_shape, a_shape, b_shape.
pub fn binary(t_in: &str, t_out: &str, expr: &str) -> String {
    format!(
        "{h}{src}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, a_rank: u32, b_rank: u32,
    out_shape: array<u32,8>, a_shape: array<u32,8>, b_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<{t_in}>;
@group(0) @binding(1) var<storage, read> b: array<{t_in}>;
@group(0) @binding(2) var<storage, read_write> out: array<{t_out}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let av = a[src_index(idx, p.out_shape, p.out_rank, p.a_shape, p.a_rank)];
    let bv = b[src_index(idx, p.out_shape, p.out_rank, p.b_shape, p.b_rank)];
    out[idx] = {expr};
}}",
        h = header(),
        src = SRC_INDEX,
    )
}

/// Element-wise unary op. Bindings: 0=in, 1=out.
pub fn unary(t: &str, expr: &str, needs_erf: bool) -> String {
    format!(
        "{h}{erf}
struct P {{ size: u32, x_stride: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let v = x[idx];
    out[idx] = {expr};
}}",
        h = header(),
        erf = if needs_erf { ERF } else { "" },
    )
}

/// Dtype conversion. Bindings: 0=in, 1=out.
pub fn cast(t_in: &str, t_out: &str, expr: &str) -> String {
    format!(
        "{h}
struct P {{ size: u32, x_stride: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t_in}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t_out}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let v = x[idx];
    out[idx] = {expr};
}}",
        h = header(),
    )
}

/// Three-way-broadcast select. Bindings: 0=cond(u32), 1=a, 2=b, 3=out.
pub fn select3(t: &str) -> String {
    format!(
        "{h}{src}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, c_rank: u32, a_rank: u32, b_rank: u32,
    out_shape: array<u32,8>, c_shape: array<u32,8>,
    a_shape: array<u32,8>, b_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> c: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<{t}>;
@group(0) @binding(2) var<storage, read> b: array<{t}>;
@group(0) @binding(3) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let cv = c[src_index(idx, p.out_shape, p.out_rank, p.c_shape, p.c_rank)];
    let av = a[src_index(idx, p.out_shape, p.out_rank, p.a_shape, p.a_rank)];
    let bv = b[src_index(idx, p.out_shape, p.out_rank, p.b_shape, p.b_rank)];
    out[idx] = select(bv, av, cv != 0u);
}}",
        h = header(),
        src = SRC_INDEX,
    )
}

/// Batched matmul, one thread per output element. Batch dims must be equal
/// or scalar (checked at plan time). Bindings: 0=a, 1=b, 2=out.
pub fn matmul(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    m: u32, n: u32, k: u32,
    a_batch_stride: u32, b_batch_stride: u32,
    trans_a: u32, trans_b: u32,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
@group(0) @binding(2) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let bi = idx / (p.m * p.n);
    let r = (idx / p.n) % p.m;
    let c = idx % p.n;
    let a_base = bi * p.a_batch_stride;
    let b_base = bi * p.b_batch_stride;
    var acc: {t} = {zero};
    for (var kk = 0u; kk < p.k; kk = kk + 1u) {{
        let ae = a_base + select(r * p.k + kk, kk * p.m + r, p.trans_a == 1u);
        let be = b_base + select(kk * p.n + c, c * p.k + kk, p.trans_b == 1u);
        acc = acc + a[ae] * b[be];
    }}
    out[idx] = acc;
}}",
        h = header(),
        zero = if t == "f32" { "0.0" } else { "0" },
    )
}

/// Reduction over an axes bitmask, one thread per output element.
/// Bindings: 0=in, 1=out.
pub fn reduce(t: &str, init: &str, combine: &str, finalize: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    in_rank: u32, axes_mask: u32, reduce_count: u32,
    in_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    // Base index from the non-reduced coordinates.
    var out_rem = idx;
    var base = 0u;
    var stride = 1u;
    for (var kk = 0u; kk < p.in_rank; kk = kk + 1u) {{
        let d = p.in_rank - 1u - kk;
        if ((p.axes_mask & (1u << d)) == 0u) {{
            let coord = out_rem % p.in_shape[d];
            out_rem = out_rem / p.in_shape[d];
            base = base + coord * stride;
        }}
        stride = stride * p.in_shape[d];
    }}
    var acc: {t} = {init};
    for (var r = 0u; r < p.reduce_count; r = r + 1u) {{
        var r_rem = r;
        var off = 0u;
        var s2 = 1u;
        for (var kk = 0u; kk < p.in_rank; kk = kk + 1u) {{
            let d = p.in_rank - 1u - kk;
            if ((p.axes_mask & (1u << d)) != 0u) {{
                let coord = r_rem % p.in_shape[d];
                r_rem = r_rem / p.in_shape[d];
                off = off + coord * s2;
            }}
            s2 = s2 * p.in_shape[d];
        }}
        let v = x[base + off];
        acc = {combine};
    }}
    out[idx] = {finalize};
}}",
        h = header(),
    )
}

/// ONNX Gather along an axis. Bindings: 0=data, 1=indices(i32), 2=out.
pub fn gather(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    axis: u32, data_rank: u32, indices_rank: u32,
    data_shape: array<u32,8>, indices_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> data: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let out_rank = p.data_rank - 1u + p.indices_rank;
    // Output coordinates.
    var coords: array<u32,8>;
    var rem = idx;
    for (var kk = 0u; kk < out_rank; kk = kk + 1u) {{
        let d = out_rank - 1u - kk;
        coords[d] = rem % p.out_shape[d];
        rem = rem / p.out_shape[d];
    }}
    // Indices linear index from the middle coordinate block.
    var ii = 0u;
    var stride = 1u;
    for (var kk = 0u; kk < p.indices_rank; kk = kk + 1u) {{
        let d = p.indices_rank - 1u - kk;
        ii = ii + coords[p.axis + d] * stride;
        stride = stride * p.indices_shape[d];
    }}
    let dim = i32(p.data_shape[p.axis]);
    var iv = indices[ii];
    if (iv < 0) {{ iv = iv + dim; }}
    iv = clamp(iv, 0, dim - 1);
    // Data linear index.
    var di = 0u;
    stride = 1u;
    for (var kk = 0u; kk < p.data_rank; kk = kk + 1u) {{
        let d = p.data_rank - 1u - kk;
        var coord: u32;
        if (d < p.axis) {{
            coord = coords[d];
        }} else if (d == p.axis) {{
            coord = u32(iv);
        }} else {{
            coord = coords[d + p.indices_rank - 1u];
        }}
        di = di + coord * stride;
        stride = stride * p.data_shape[d];
    }}
    out[idx] = data[di];
}}",
        h = header(),
    )
}

/// ScatterND writes (run after copying data to out).
/// Bindings: 0=indices(i32), 1=updates, 2=out.
pub fn scatter(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    k: u32, slice_len: u32, data_rank: u32,
    data_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read> updates: array<{t}>;
@group(0) @binding(2) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    let u = idx / p.slice_len;
    let off = idx % p.slice_len;
    var base = 0u;
    for (var d = 0u; d < p.k; d = d + 1u) {{
        let dim = i32(p.data_shape[d]);
        var iv = indices[u * p.k + d];
        if (iv < 0) {{ iv = iv + dim; }}
        iv = clamp(iv, 0, dim - 1);
        base = base * p.data_shape[d] + u32(iv);
    }}
    out[base * p.slice_len + off] = updates[idx];
}}",
        h = header(),
    )
}

/// Transpose by permutation. Bindings: 0=in, 1=out.
pub fn transpose(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    perm: array<u32,8>, in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    var ocoords: array<u32,8>;
    var rem = idx;
    for (var kk = 0u; kk < p.rank; kk = kk + 1u) {{
        let d = p.rank - 1u - kk;
        ocoords[d] = rem % p.out_shape[d];
        rem = rem / p.out_shape[d];
    }}
    // in[perm[i]] = out[i]
    var icoords: array<u32,8>;
    for (var i = 0u; i < p.rank; i = i + 1u) {{
        icoords[p.perm[i]] = ocoords[i];
    }}
    var ii = 0u;
    var stride = 1u;
    for (var kk = 0u; kk < p.rank; kk = kk + 1u) {{
        let d = p.rank - 1u - kk;
        ii = ii + icoords[d] * stride;
        stride = stride * p.in_shape[d];
    }}
    out[idx] = x[ii];
}}",
        h = header(),
    )
}

/// Strided slice. `step` entries are i32 bit-packed into u32 slots.
/// Bindings: 0=in, 1=out.
pub fn slice(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    start: array<u32,8>, step: array<u32,8>,
    in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    var rem = idx;
    var ii = 0u;
    var stride = 1u;
    for (var kk = 0u; kk < p.rank; kk = kk + 1u) {{
        let d = p.rank - 1u - kk;
        let oc = rem % p.out_shape[d];
        rem = rem / p.out_shape[d];
        let ic = u32(i32(p.start[d]) + i32(oc) * bitcast<i32>(p.step[d]));
        ii = ii + ic * stride;
        stride = stride * p.in_shape[d];
    }}
    out[idx] = x[ii];
}}",
        h = header(),
    )
}

/// Broadcast (Expand) copy. Bindings: 0=in, 1=out.
pub fn broadcast(t: &str) -> String {
    format!(
        "{h}{src}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, in_rank: u32,
    out_shape: array<u32,8>, in_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    out[idx] = x[src_index(idx, p.out_shape, p.out_rank, p.in_shape, p.in_rank)];
}}",
        h = header(),
        src = SRC_INDEX,
    )
}

/// Copy one concat input into its slot of the output (one dispatch per
/// input, threads over the *input*). Bindings: 0=in, 1=out.
pub fn concat_emplace(t: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    axis: u32, axis_offset: u32,
    in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: array<{t}>;
@group(0) @binding(1) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    var rem = idx;
    var oi = 0u;
    var stride = 1u;
    for (var kk = 0u; kk < p.rank; kk = kk + 1u) {{
        let d = p.rank - 1u - kk;
        var coord = rem % p.in_shape[d];
        rem = rem / p.in_shape[d];
        if (d == p.axis) {{ coord = coord + p.axis_offset; }}
        oi = oi + coord * stride;
        stride = stride * p.out_shape[d];
    }}
    out[oi] = x[idx];
}}",
        h = header(),
    )
}

/// The integer ramp. Bindings: 0=out.
pub fn iota(t: &str) -> String {
    let expr = match t {
        "f32" => "f32(idx)",
        _ => "i32(idx)",
    };
    format!(
        "{h}
struct P {{ size: u32, x_stride: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> out: array<{t}>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    out[idx] = {expr};
}}",
        h = header(),
    )
}

/// Plain element copy (Scatter's first stage). Bindings: 0=in, 1=out.
pub fn copy(t: &str) -> String {
    cast(t, t, "v")
}

// ──────────────────────── tiled matmul (m > 1) ─────────────────────────

/// Shared-memory tiled batched matmul, 16×16 output tile per workgroup.
/// Both operands stage through workgroup tiles with coalesced loads in
/// every layout (the load index puts `tx` on the contiguous axis), so
/// `trans_a`/`trans_b` cost nothing; tiles are padded (stride 17) against
/// bank conflicts. Grid: `x` = ceil(N/16), `y` = ceil(M/16), `z` = batch.
/// Bindings: 0=a, 1=b, 2=out.
pub fn matmul_tiled(trans_a: bool, trans_b: bool) -> String {
    // As/Bs hold the operand tile; layout (and thus the load and product
    // expressions) depends on which axis is contiguous in memory.
    let (a_guard, a_load, a_term) = if trans_a {
        // a is [K,M]: As[k][m].
        (
            "k0 + ty < p.k && m0 + tx < p.m",
            "a[a_base + (k0 + ty) * p.m + (m0 + tx)]",
            "As[kk * 17u + ty]",
        )
    } else {
        // a is [M,K]: As[m][k].
        (
            "m0 + ty < p.m && k0 + tx < p.k",
            "a[a_base + (m0 + ty) * p.k + (k0 + tx)]",
            "As[ty * 17u + kk]",
        )
    };
    let (b_guard, b_load, b_term) = if trans_b {
        // b is [N,K]: Bs[n][k].
        (
            "n0 + ty < p.n && k0 + tx < p.k",
            "b[b_base + (n0 + ty) * p.k + (k0 + tx)]",
            "Bs[tx * 17u + kk]",
        )
    } else {
        // b is [K,N]: Bs[k][n].
        (
            "k0 + ty < p.k && n0 + tx < p.n",
            "b[b_base + (k0 + ty) * p.n + (n0 + tx)]",
            "Bs[kk * 17u + tx]",
        )
    };
    format!(
        "
struct P {{ m: u32, n: u32, k: u32, a_bs: u32, b_bs: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> As: array<f32, 272>;
var<workgroup> Bs: array<f32, 272>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tx = lid.x;
    let ty = lid.y;
    let n0 = wg.x * 16u;
    let m0 = wg.y * 16u;
    let a_base = wg.z * p.a_bs;
    let b_base = wg.z * p.b_bs;
    var acc = 0.0;
    for (var k0 = 0u; k0 < p.k; k0 += 16u) {{
        var av = 0.0;
        if ({a_guard}) {{ av = {a_load}; }}
        As[ty * 17u + tx] = av;
        var bv = 0.0;
        if ({b_guard}) {{ bv = {b_load}; }}
        Bs[ty * 17u + tx] = bv;
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk += 1u) {{
            acc += {a_term} * {b_term};
        }}
        workgroupBarrier();
    }}
    if (m0 + ty < p.m && n0 + tx < p.n) {{
        out[wg.z * p.m * p.n + (m0 + ty) * p.n + (n0 + tx)] = acc;
    }}
}}"
    )
}

// ─────────────────── matrix-vector fast path (m = 1) ───────────────────
//
// The generic one-thread-per-output-element matmul launches only N
// threads for an M=1 matmul — a few workgroups on decode-shaped
// projections, leaving the GPU almost idle. These kernels split the
// contraction (K) across threads and, when N alone can't fill the
// device, across `ks` workgroup slices whose partial sums a second tiny
// dispatch ([`matvec_reduce`]) folds together. Layout drives the
// threading so weight reads are always coalesced:
//
// - `[K,N]` (`trans_b == false`): 64 adjacent columns per workgroup ×
//   4 K-lanes; lanes step rows, columns sit in consecutive addresses.
// - `[N,K]` (`trans_b == true`): one output row per workgroup; 256
//   threads stride K, adjacent threads on adjacent addresses.

/// M=1 matmul over `[K,N]` weights with vec4 column loads (requires
/// `N % 4 == 0`; otherwise the scalar [`matvec_kn`] runs). Same
/// 64-scalar-column tile as the scalar kernel — keeping the workgroup
/// count (occupancy matters more than width on the small projections) —
/// but as 16 vec4 lanes × 16 K-lanes. Grid: `x` = ceil(N/64) tiles,
/// `y` = K slices. Bindings as [`matvec_kn`], with `b`/`dst` viewed as
/// vec4 (same byte layout).
pub fn matvec_kn_v4() -> String {
    "
struct P { n4: u32, k: u32, ks: u32, chunk: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
var<workgroup> scratch: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x % 16u;
    let ty = lid.x / 16u;
    let n4 = wg.x * 16u + tx;
    let k0 = wg.y * p.chunk;
    let k1 = min(k0 + p.chunk, p.k);
    var acc = vec4<f32>(0.0);
    if (n4 < p.n4) {
        for (var k = k0 + ty; k < k1; k += 16u) {
            acc += a[k] * b[k * p.n4 + n4];
        }
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 8u; s > 0u; s = s >> 1u) {
        if (ty < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s * 16u];
        }
        workgroupBarrier();
    }
    if (ty == 0u && n4 < p.n4) {
        dst[wg.y * p.n4 + n4] = scratch[tx];
    }
}
"
    .to_string()
}

/// M=1 matmul over `[N,K]` weights with vec4 K loads (requires
/// `K % 4 == 0`; otherwise the scalar [`matvec_transb`] runs). One row
/// per 2.5 KB is latency-bound at one row per workgroup, so each
/// workgroup covers **4 rows** as 4 × 64 lanes. Grid: linearized
/// `ceil(N/4) × ks` workgroups. Bindings as [`matvec_transb`], with
/// `a`/`b` viewed as vec4.
pub fn matvec_transb_v4() -> String {
    "
struct P { n: u32, k4: u32, ks: u32, chunk4: u32, x_wgs: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x % 64u;
    let row_i = lid.x / 64u;
    let wg_lin = wg.y * p.x_wgs + wg.x;
    let groups = (p.n + 3u) / 4u;
    // Grid rounding can overshoot; dead threads compute on row 0 and
    // skip the write (no early return around the barriers).
    let live_wg = wg_lin < groups * p.ks;
    let g = select(0u, wg_lin / p.ks, live_wg);
    let slice = select(0u, wg_lin % p.ks, live_wg);
    let n = g * 4u + row_i;
    let live = live_wg && n < p.n;
    let k0 = slice * p.chunk4;
    let k1 = min(k0 + p.chunk4, p.k4);
    var acc = 0.0;
    if (live) {
        for (var i = k0 + lane; i < k1; i += 64u) {
            acc += dot(a[i], b[n * p.k4 + i]);
        }
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lane == 0u && live) {
        dst[slice * p.n + n] = scratch[lid.x];
    }
}
"
    .to_string()
}

/// M=1 matmul over `[K,N]` weights. Grid: `x` = ceil(N/64) column
/// tiles, `y` = K slices. Bindings: 0=a (len K), 1=b, 2=dst
/// (`[ks, N]` partials; equals the output when `ks == 1`).
pub fn matvec_kn() -> String {
    "
struct P { n: u32, k: u32, ks: u32, chunk: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x % 64u;
    let ty = lid.x / 64u;
    let n = wg.x * 64u + tx;
    let k0 = wg.y * p.chunk;
    let k1 = min(k0 + p.chunk, p.k);
    var acc = 0.0;
    if (n < p.n) {
        for (var k = k0 + ty; k < k1; k += 4u) {
            acc += a[k] * b[k * p.n + n];
        }
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    if (ty == 0u && n < p.n) {
        dst[wg.y * p.n + n] = scratch[tx] + scratch[64u + tx]
            + scratch[128u + tx] + scratch[192u + tx];
    }
}
"
    .to_string()
}

/// M=1 matmul over `[N,K]` weights (`trans_b`). Grid: `x_wgs × y`
/// workgroups linearized to `n * ks` slices. Bindings as
/// [`matvec_kn`].
pub fn matvec_transb() -> String {
    "
struct P { n: u32, k: u32, ks: u32, chunk: u32, x_wgs: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let wg_lin = wg.y * p.x_wgs + wg.x;
    // Grid rounding can overshoot; dead workgroups compute on row 0 and
    // skip the write (no early return around the barriers).
    let live = wg_lin < p.n * p.ks;
    let n = select(0u, wg_lin / p.ks, live);
    let ks = select(0u, wg_lin % p.ks, live);
    let k0 = ks * p.chunk;
    let k1 = min(k0 + p.chunk, p.k);
    var acc = 0.0;
    for (var k = k0 + lid.x; k < k1; k += 256u) {
        acc += a[k] * b[n * p.k + k];
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u && live) {
        dst[ks * p.n + n] = scratch[0];
    }
}
"
    .to_string()
}

/// Fold `[ks, N]` matvec partials into the `[N]` output. One thread per
/// output element. Bindings: 0=partials, 1=out.
pub fn matvec_reduce() -> String {
    format!(
        "{h}
struct P {{ size: u32, x_stride: u32, ks: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
    var acc = 0.0;
    for (var j = 0u; j < p.ks; j += 1u) {{
        acc += partials[j * p.size + idx];
    }}
    out[idx] = acc;
}}",
        h = header(),
    )
}
