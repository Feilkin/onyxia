//! Generated WGSL kernels for the primitive set.
//!
//! One uniform scheme: each kernel is a template specialized by the
//! physical [`Layout`]s of its operands and an operation expression,
//! generated as plain WGSL text and compiled through naga's front-end.
//! All shape information travels in immediates (push constants) as
//! `u32`s, ranks are capped at [`MAX_RANK`].
//!
//! Every generic kernel has the same shape: per-binding `ld_*` load
//! functions (from [`Layout::load_fn`]), a `compute(e)` function giving
//! output element `e` in the compute type, and a `main` that runs one
//! thread per output *word* ([`Layout::store_block`]) — so packed
//! layouts (four 8-bit values or two f16s per `u32`) are written whole.
//! Correctness-first; fused/tiled kernels come via the composite kernel
//! registry and the matvec/tiled sections below.

use crate::gpu::WORKGROUP_SIZE;
use crate::layout::Layout;

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

/// Common prelude: `enable f16;` when any operand is native f16, the
/// workgroup size, and the dispatch-linearization helper.
fn prelude(layouts: &[&Layout]) -> String {
    let mut s = String::new();
    if layouts.iter().any(|l| l.needs_f16()) {
        s.push_str("enable f16;\n");
    }
    s.push_str(&format!(
        "const WG_SIZE: u32 = {WORKGROUP_SIZE}u;\n\
         fn linear_idx(gid: vec3<u32>, x_stride: u32) -> u32 {{\n\
             return gid.x + gid.y * x_stride;\n\
         }}\n"
    ));
    s
}

/// Prelude for the hand-written f32 kernels below (no f16).
fn header() -> String {
    prelude(&[])
}

/// The standard `main`: one thread per output word of `out`.
fn main_fn(out: &Layout) -> String {
    format!(
        "@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
{}
}}",
        out.store_block("out")
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

pub(crate) const ERF: &str = "
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

/// Element-wise binary op with full N-D broadcasting; `expr` is over
/// `av`/`bv` (compute types) and yields `out`'s compute type. Also used
/// for comparisons (Bool output). Bindings: 0=a, 1=b, 2=out.
pub fn binary(a: &Layout, b: &Layout, out: &Layout, expr: &str) -> String {
    let ipow = match a.compute() {
        "f32" => String::new(),
        c => format!(
            "fn ipow(base: {c}, e: {c}) -> {c} {{
    var r: {c} = {c}(1);
    var b = base;
    var n = e;
    if (n < {c}(0)) {{ return {c}(0); }}
    loop {{
        if (n == {c}(0)) {{ break; }}
        if ((n & {c}(1)) != {c}(0)) {{ r = r * b; }}
        b = b * b;
        n = n >> 1u;
    }}
    return r;
}}
"
        ),
    };
    format!(
        "{h}{src}{ipow}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, a_rank: u32, b_rank: u32,
    out_shape: array<u32,8>, a_shape: array<u32,8>, b_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: {ab};
@group(0) @binding(1) var<storage, read> b: {bb};
@group(0) @binding(2) var<storage, read_write> out: {ob};
{lda}{ldb}
fn compute(idx: u32) -> {oc} {{
    let av = ld_a(src_index(idx, p.out_shape, p.out_rank, p.a_shape, p.a_rank));
    let bv = ld_b(src_index(idx, p.out_shape, p.out_rank, p.b_shape, p.b_rank));
    return {expr};
}}
{main}",
        h = prelude(&[a, b, out]),
        src = SRC_INDEX,
        ab = a.binding(),
        bb = b.binding(),
        ob = out.binding(),
        lda = a.load_fn("ld_a", "a"),
        ldb = b.load_fn("ld_b", "b"),
        oc = out.compute(),
        main = main_fn(out),
    )
}

/// Element-wise unary op; `expr` over `v`. Bindings: 0=in, 1=out.
pub fn unary(x: &Layout, out: &Layout, expr: &str, needs_erf: bool) -> String {
    format!(
        "{h}{erf}
struct P {{ size: u32, x_stride: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {xb};
@group(0) @binding(1) var<storage, read_write> out: {ob};
{ldx}
fn compute(idx: u32) -> {oc} {{
    let v = ld_x(idx);
    return {expr};
}}
{main}",
        h = prelude(&[x, out]),
        erf = if needs_erf { ERF } else { "" },
        xb = x.binding(),
        ob = out.binding(),
        ldx = x.load_fn("ld_x", "x"),
        oc = out.compute(),
        main = main_fn(out),
    )
}

/// Dtype conversion; `expr` over `v` yields `out`'s compute type.
/// Bindings: 0=in, 1=out.
pub fn cast(x: &Layout, out: &Layout, expr: &str) -> String {
    unary(x, out, expr, false)
}

/// Plain element copy (Scatter's first stage). Bindings: 0=in, 1=out.
pub fn copy(l: &Layout) -> String {
    cast(l, l, "v")
}

/// Three-way-broadcast select. Bindings: 0=cond(u32), 1=a, 2=b, 3=out
/// (`a`, `b`, `out` share a layout).
pub fn select3(c: &Layout, t: &Layout) -> String {
    format!(
        "{h}{src}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, c_rank: u32, a_rank: u32, b_rank: u32,
    out_shape: array<u32,8>, c_shape: array<u32,8>,
    a_shape: array<u32,8>, b_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> c: {cb};
@group(0) @binding(1) var<storage, read> a: {tb};
@group(0) @binding(2) var<storage, read> b: {tb};
@group(0) @binding(3) var<storage, read_write> out: {tb};
{ldc}{lda}{ldb}
fn compute(idx: u32) -> {tc} {{
    let cv = ld_c(src_index(idx, p.out_shape, p.out_rank, p.c_shape, p.c_rank));
    let av = ld_a(src_index(idx, p.out_shape, p.out_rank, p.a_shape, p.a_rank));
    let bv = ld_b(src_index(idx, p.out_shape, p.out_rank, p.b_shape, p.b_rank));
    return select(bv, av, cv != 0u);
}}
{main}",
        h = prelude(&[c, t]),
        src = SRC_INDEX,
        cb = c.binding(),
        tb = t.binding(),
        ldc = c.load_fn("ld_c", "c"),
        lda = t.load_fn("ld_a", "a"),
        ldb = t.load_fn("ld_b", "b"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// Batched matmul, one thread per output element. Batch dims must be equal
/// or scalar (checked at plan time). Bindings: 0=a, 1=b, 2=out.
pub fn matmul(t: &Layout) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    m: u32, n: u32, k: u32,
    a_batch_stride: u32, b_batch_stride: u32,
    trans_a: u32, trans_b: u32,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: {tb};
@group(0) @binding(1) var<storage, read> b: {tb};
@group(0) @binding(2) var<storage, read_write> out: {tb};
{lda}{ldb}
fn compute(idx: u32) -> {tc} {{
    let bi = idx / (p.m * p.n);
    let r = (idx / p.n) % p.m;
    let c = idx % p.n;
    let a_base = bi * p.a_batch_stride;
    let b_base = bi * p.b_batch_stride;
    var acc: {tc} = {zero};
    for (var kk = 0u; kk < p.k; kk = kk + 1u) {{
        let ae = a_base + select(r * p.k + kk, kk * p.m + r, p.trans_a == 1u);
        let be = b_base + select(kk * p.n + c, c * p.k + kk, p.trans_b == 1u);
        acc = acc + ld_a(ae) * ld_b(be);
    }}
    return acc;
}}
{main}",
        h = prelude(&[t]),
        tb = t.binding(),
        lda = t.load_fn("ld_a", "a"),
        ldb = t.load_fn("ld_b", "b"),
        tc = t.compute(),
        zero = t.zero(),
        main = main_fn(t),
    )
}

/// Reduction over an axes bitmask, one thread per output element.
/// `combine` is over `acc`/`v`, `finalize` over `acc`. Bindings: 0=in, 1=out.
pub fn reduce(t: &Layout, init: &str, combine: &str, finalize: &str) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    in_rank: u32, axes_mask: u32, reduce_count: u32,
    in_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: {tb};
{ldx}
fn compute(idx: u32) -> {tc} {{
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
    var acc: {tc} = {init};
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
        let v = ld_x(base + off);
        acc = {combine};
    }}
    return {finalize};
}}
{main}",
        h = prelude(&[t]),
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// ONNX Gather along an axis. Bindings: 0=data, 1=indices, 2=out.
pub fn gather(t: &Layout, idx_l: &Layout) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    axis: u32, data_rank: u32, indices_rank: u32,
    data_shape: array<u32,8>, indices_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> data: {tb};
@group(0) @binding(1) var<storage, read> indices: {ib};
@group(0) @binding(2) var<storage, read_write> out: {tb};
{ldd}{ldi}
fn compute(idx: u32) -> {tc} {{
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
    var iv = i32(ld_i(ii));
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
    return ld_d(di);
}}
{main}",
        h = prelude(&[t, idx_l]),
        tb = t.binding(),
        ib = idx_l.binding(),
        ldd = t.load_fn("ld_d", "data"),
        ldi = idx_l.load_fn("ld_i", "indices"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// The flat target element of one ScatterND update (shared by both
/// scatter kernels). Threads run over update elements.
const SCATTER_TARGET: &str = "
    let u = idx / p.slice_len;
    let off = idx % p.slice_len;
    var base = 0u;
    for (var d = 0u; d < p.k; d = d + 1u) {
        let dim = i32(p.data_shape[d]);
        var iv = i32(ld_i(u * p.k + d));
        if (iv < 0) { iv = iv + dim; }
        iv = clamp(iv, 0, dim - 1);
        base = base * p.data_shape[d] + u32(iv);
    }
    let tgt = base * p.slice_len + off;
";

/// ScatterND overwrite for one-element-per-word layouts (run after
/// copying data to out). Bindings: 0=indices, 1=updates, 2=out.
pub fn scatter(t: &Layout, idx_l: &Layout) -> String {
    let store = match t.store() {
        "f16" => "f16(ld_u(idx))".to_string(),
        _ => "ld_u(idx)".to_string(),
    };
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    k: u32, slice_len: u32, data_rank: u32,
    data_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> indices: {ib};
@group(0) @binding(1) var<storage, read> updates: {tb};
@group(0) @binding(2) var<storage, read_write> out: {tb};
{ldi}{ldu}
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
{target}
    out[tgt] = {store};
}}",
        h = prelude(&[t, idx_l]),
        ib = idx_l.binding(),
        tb = t.binding(),
        ldi = idx_l.load_fn("ld_i", "indices"),
        ldu = t.load_fn("ld_u", "updates"),
        target = SCATTER_TARGET,
    )
}

/// ScatterND with a reduction, or into a packed layout: a compare-and-
/// swap loop on the 32-bit output word. `combine` is over `cur`/`upd`
/// (compute type). Bindings: 0=indices, 1=updates, 2=out (atomic u32).
pub fn scatter_atomic(
    t: &Layout,
    idx_l: &Layout,
    combine: &str,
) -> Result<String, onyxia_ir::Error> {
    let extract = t.lane_extract("old", "lane")?;
    let insert = t.lane_insert("old", "lane", "nv")?;
    let lanes = t.lanes();
    Ok(format!(
        "{h}
struct P {{
    size: u32, x_stride: u32,
    k: u32, slice_len: u32, data_rank: u32,
    data_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> indices: {ib};
@group(0) @binding(1) var<storage, read> updates: {tb};
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>>;
{ldi}{ldu}
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = linear_idx(gid, p.x_stride);
    if (idx >= p.size) {{ return; }}
{target}
    let upd = ld_u(idx);
    let wi = tgt / {lanes}u;
    let lane = tgt % {lanes}u;
    loop {{
        let old = atomicLoad(&out[wi]);
        let cur = {extract};
        let nv = {combine};
        let packed = {insert};
        let r = atomicCompareExchangeWeak(&out[wi], old, packed);
        if (r.exchanged) {{ break; }}
    }}
}}",
        h = prelude(&[t, idx_l]),
        ib = idx_l.binding(),
        tb = t.binding(),
        ldi = idx_l.load_fn("ld_i", "indices"),
        ldu = t.load_fn("ld_u", "updates"),
        target = SCATTER_TARGET,
    ))
}

/// Transpose by permutation. Bindings: 0=in, 1=out.
pub fn transpose(t: &Layout) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    perm: array<u32,8>, in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: {tb};
{ldx}
fn compute(idx: u32) -> {tc} {{
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
    return ld_x(ii);
}}
{main}",
        h = prelude(&[t]),
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// Strided slice. `step` entries are i32 bit-packed into u32 slots.
/// Bindings: 0=in, 1=out.
pub fn slice(t: &Layout) -> String {
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    start: array<u32,8>, step: array<u32,8>,
    in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: {tb};
{ldx}
fn compute(idx: u32) -> {tc} {{
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
    return ld_x(ii);
}}
{main}",
        h = prelude(&[t]),
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// Broadcast (Expand) copy. Bindings: 0=in, 1=out.
pub fn broadcast(t: &Layout) -> String {
    format!(
        "{h}{src}
struct P {{
    size: u32, x_stride: u32,
    out_rank: u32, in_rank: u32,
    out_shape: array<u32,8>, in_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: {tb};
{ldx}
fn compute(idx: u32) -> {tc} {{
    return ld_x(src_index(idx, p.out_shape, p.out_rank, p.in_shape, p.in_rank));
}}
{main}",
        h = prelude(&[t]),
        src = SRC_INDEX,
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
        tc = t.compute(),
        main = main_fn(t),
    )
}

/// Copy one concat input into its slot of the output (one dispatch per
/// input, threads over the *input*). One-element-per-word layouts only.
/// Bindings: 0=in, 1=out.
pub fn concat_emplace(t: &Layout) -> String {
    let store = match t.store() {
        "f16" => "f16(ld_x(idx))",
        _ => "ld_x(idx)",
    };
    format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    axis: u32, axis_offset: u32,
    in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: {tb};
{ldx}
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
    out[oi] = {store};
}}",
        h = prelude(&[t]),
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
    )
}

/// Concat into a packed layout: one dispatch per input, threads over the
/// *output words*; each word is read, the lanes that fall in this input's
/// slot are replaced, and the word is written back (queue order between
/// the per-input dispatches makes the read-modify-write safe). `size` is
/// the output element count. Bindings: 0=in, 1=out.
pub fn concat_packed(t: &Layout) -> Result<String, onyxia_ir::Error> {
    let lanes = t.lanes();
    let extract = t.lane_extract("word", "l")?;
    let insert = t.lane_insert("word", "l", "v")?;
    Ok(format!(
        "{h}
struct P {{
    size: u32, x_stride: u32, rank: u32,
    axis: u32, axis_offset: u32, in_len: u32,
    in_shape: array<u32,8>, out_shape: array<u32,8>,
}}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> x: {tb};
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
{ldx}
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let w = linear_idx(gid, p.x_stride);
    if (w >= (p.size + {lanes}u - 1u) / {lanes}u) {{ return; }}
    var word = out[w];
    for (var l = 0u; l < {lanes}u; l = l + 1u) {{
        let e = w * {lanes}u + l;
        if (e >= p.size) {{ continue; }}
        // Output coordinates → input index if inside this input's slot.
        var rem = e;
        var ii = 0u;
        var stride = 1u;
        var inside = true;
        for (var kk = 0u; kk < p.rank; kk = kk + 1u) {{
            let d = p.rank - 1u - kk;
            var coord = rem % p.out_shape[d];
            rem = rem / p.out_shape[d];
            if (d == p.axis) {{
                if (coord < p.axis_offset || coord >= p.axis_offset + p.in_len) {{
                    inside = false;
                }}
                coord = coord - p.axis_offset;
            }}
            ii = ii + coord * stride;
            stride = stride * p.in_shape[d];
        }}
        if (inside) {{
            let v = ld_x(ii);
            let _unused = {extract};
            word = {insert};
        }}
    }}
    out[w] = word;
}}",
        h = prelude(&[t]),
        tb = t.binding(),
        ldx = t.load_fn("ld_x", "x"),
    ))
}

/// The integer ramp. Bindings: 0=out.
pub fn iota(t: &Layout) -> String {
    format!(
        "{h}
struct P {{ size: u32, x_stride: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> out: {tb};
fn compute(idx: u32) -> {tc} {{
    return {tc}(idx);
}}
{main}",
        h = prelude(&[t]),
        tb = t.binding(),
        tc = t.compute(),
        main = main_fn(t),
    )
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

// ───────────────────────── block dequantization ─────────────────────────

/// `Prim::Dequantize`: `out[e] = (q[e] - zp[e / bs]) * scale[e / bs]`,
/// one thread per output word. `data` is a packed 4-/8-bit layout,
/// `scales` a float layout (also the output's), `zp` (when bound) the
/// data's layout. `p.zp_default` is the implicit zero point when no
/// zero-point tensor is bound. Bindings: 0=data, 1=scales, [2=zp], last=out.
pub fn dequantize(data: &Layout, scales: &Layout, zp: Option<&Layout>) -> String {
    let (zp_bind, zp_ld, zp_expr, out_slot) = match zp {
        Some(z) => (
            format!(
                "@group(0) @binding(2) var<storage, read> zp: {};\n",
                z.binding()
            ),
            z.load_fn("ld_z", "zp"),
            "i32(ld_z(blk))".to_string(),
            3,
        ),
        None => (String::new(), String::new(), "p.zp_default".to_string(), 2),
    };
    format!(
        "{h}
struct P {{ size: u32, x_stride: u32, bs: u32, zp_default: i32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> data: {db};
@group(0) @binding(1) var<storage, read> scales: {sb};
{zp_bind}@group(0) @binding({out_slot}) var<storage, read_write> out: {sb};
{ldd}{lds}{zp_ld}
fn compute(e: u32) -> f32 {{
    let blk = e / p.bs;
    let q = i32(ld_d(e));
    let z = {zp_expr};
    return f32(q - z) * ld_s(blk);
}}
{main}",
        h = prelude(&[data, scales]),
        db = data.binding(),
        sb = scales.binding(),
        ldd = data.load_fn("ld_d", "data"),
        lds = scales.load_fn("ld_s", "scales"),
        main = main_fn(scales),
    )
}

/// Fused `MatMulNBits` decode step (M = 1): `y[n] = Σ_k a[k] · (q[n,k] -
/// zp[n,blk]) · s[n,blk]` straight from the packed 4-bit weights — the
/// dequantized matrix never exists. Same threading as
/// [`matvec_transb_v4`]: 4 rows × 64 lanes per workgroup, each lane
/// consuming one `u32` word (8 nibbles, two vec4 activation loads) per
/// step, split-K across `ks` slices into `[ks, N]` partials. `p.rw` is
/// the packed row stride in words (K padded to whole blocks).
/// Bindings: 0=a (vec4 view, K/4), 1=b (packed `[N, rw]` words),
/// 2=scales (`[N, nb]`), [3=zp (packed `[N, nb]` nibbles)], last=dst.
pub fn matmul_nbits_matvec(zp: bool) -> String {
    let (zp_bind, zp_expr, dst_slot) = if zp {
        (
            "@group(0) @binding(3) var<storage, read> zp: array<u32>;\n",
            "f32((zp[(n * p.nb + blk) >> 3u] >> (((n * p.nb + blk) & 7u) * 4u)) & 0xfu)",
            4,
        )
    } else {
        ("", "8.0", 3)
    };
    format!(
        "
struct P {{ n: u32, k8: u32, rw: u32, bs8: u32, nb: u32, ks: u32, chunk8: u32, x_wgs: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
{zp_bind}@group(0) @binding({dst_slot}) var<storage, read_write> dst: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
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
    let k0 = slice * p.chunk8;
    let k1 = min(k0 + p.chunk8, p.k8);
    var acc = 0.0;
    if (live) {{
        // Blocks are whole words (bs % 8 == 0), so a lane's word never
        // straddles two scales; consecutive lanes read consecutive words.
        for (var i = k0 + lane; i < k1; i += 64u) {{
            let w = b[n * p.rw + i];
            let blk = i / p.bs8;
            let s = scales[n * p.nb + blk];
            let z = {zp_expr};
            let a0 = a[i * 2u];
            let a1 = a[i * 2u + 1u];
            let q0 = vec4<f32>(f32(w & 0xfu), f32((w >> 4u) & 0xfu),
                               f32((w >> 8u) & 0xfu), f32((w >> 12u) & 0xfu));
            let q1 = vec4<f32>(f32((w >> 16u) & 0xfu), f32((w >> 20u) & 0xfu),
                               f32((w >> 24u) & 0xfu), f32(w >> 28u));
            let asum = dot(a0, vec4<f32>(1.0)) + dot(a1, vec4<f32>(1.0));
            acc += s * (dot(q0, a0) + dot(q1, a1) - z * asum);
        }}
    }}
    scratch[lid.x] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {{
        if (lane < s) {{
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }}
        workgroupBarrier();
    }}
    if (lane == 0u && live) {{
        dst[slice * p.n + n] = scratch[lid.x];
    }}
}}
"
    )
}

/// Fused `MatMulNBits` prefill (M > 1): the register-blocked tile of
/// [`matmul_tiled_rb`] with the weight tile dequantized on the way into
/// workgroup memory, so the f32 weight matrix never exists. 64×64 output
/// tile, 256 threads × 4×4 outputs, K step 32 (4 packed words per weight
/// row; blocks are whole words). Grid: `x` = ceil(N/64), `y` =
/// ceil(M/64), `z` = `ks` K-slices (partials `[ks, M, N]`, folded by
/// `matvec_reduce`; `p.chunk` in elements, a multiple of 32).
/// Bindings: 0=a (`[M, K]`), 1=b (packed `[N, rw]` words, `rw` = K padded
/// to whole blocks, in words), 2=scales (`[N, nb]`), [3=zp (packed
/// `[N, nb]` nibbles)], last=out.
pub fn matmul_nbits_tiled(zp: bool) -> String {
    let (zp_bind, zp_expr, out_slot) = if zp {
        (
            "@group(0) @binding(3) var<storage, read> zp: array<u32>;\n",
            "f32((zp[(n * p.nb + blk) >> 3u] >> (((n * p.nb + blk) & 7u) * 4u)) & 0xfu)",
            4,
        )
    } else {
        ("", "8.0", 3)
    };
    format!(
        "
struct P {{ m: u32, n: u32, k8: u32, rw: u32, bs8: u32, nb: u32, chunk: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
{zp_bind}@group(0) @binding({out_slot}) var<storage, read_write> out: array<f32>;
// k-major vec4 tiles: As[k][m/4], Bs[k][n/4]; 32 k × 64 columns each.
var<workgroup> As: array<vec4<f32>, 512>;
var<workgroup> Bs: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let n0 = wg.x * 64u;
    let m0 = wg.y * 64u;
    let k = p.k8 * 8u;
    let k_lo = wg.z * p.chunk;
    let k_hi = min(k_lo + p.chunk, k);
    var acc: array<vec4<f32>, 4>;
    for (var k0 = k_lo; k0 < k_hi; k0 += 32u) {{
        // A tile 64 m × 32 k: thread owns (m, 8 consecutive k).
        {{
            let mm = lin / 4u;
            let kq = (lin % 4u) * 8u;
            let m = m0 + mm;
            for (var j = 0u; j < 8u; j += 1u) {{
                let kk = k0 + kq + j;
                var v = 0.0;
                if (m < p.m && kk < k_hi) {{ v = a[m * k + kk]; }}
                As[(kq + j) * 16u + mm / 4u][mm % 4u] = v;
            }}
        }}
        // B tile 32 k × 64 n: one packed word (8 k) per thread, dequantized.
        {{
            let nn = lin / 4u;
            let wi = lin % 4u;
            let n = n0 + nn;
            let word = k0 / 8u + wi;
            var vals = array<f32, 8>();
            if (n < p.n && word * 8u < k_hi) {{
                let w = b[n * p.rw + word];
                let blk = word / p.bs8;
                let s = scales[n * p.nb + blk];
                let z = {zp_expr};
                for (var j = 0u; j < 8u; j += 1u) {{
                    vals[j] = (f32((w >> (j * 4u)) & 0xfu) - z) * s;
                }}
            }}
            for (var j = 0u; j < 8u; j += 1u) {{
                Bs[(wi * 8u + j) * 16u + nn / 4u][nn % 4u] = vals[j];
            }}
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk += 1u) {{
            let av = As[kk * 16u + ty];
            let bv = Bs[kk * 16u + tx];
            acc[0] += av.x * bv;
            acc[1] += av.y * bv;
            acc[2] += av.z * bv;
            acc[3] += av.w * bv;
        }}
        workgroupBarrier();
    }}
    let out_base = wg.z * p.m * p.n;
    for (var i = 0u; i < 4u; i += 1u) {{
        let m = m0 + ty * 4u + i;
        if (m < p.m) {{
            for (var j = 0u; j < 4u; j += 1u) {{
                let n = n0 + tx * 4u + j;
                if (n < p.n) {{ out[out_base + m * p.n + n] = acc[i][j]; }}
            }}
        }}
    }}
}}
"
    )
}

/// Register-blocked tiled matmul: 64×64 output tile per workgroup, 256
/// threads each owning a 4×4 micro-tile, K step 16. Operand tiles are
/// staged in workgroup memory *k-major* as `vec4` columns (`As[k][m/4]`,
/// `Bs[k][n/4]`), so the inner loop is two vec4 shared loads per 16
/// FMAs — versus one shared load per FMA in [`matmul_tiled`]. Global
/// loads put consecutive threads on the contiguous axis of each layout.
/// Bindings as [`matmul_tiled`]; grid `x` = ceil(N/64), `y` = ceil(M/64),
/// `z` = `ks × batch`: when the tile grid alone can't fill the device
/// (small N, long K — the down projections) K is split into `ks` slices
/// of `p.chunk` whose partial sums land in `[ks, batch, M, N]` and are
/// folded by `matvec_reduce`.
pub fn matmul_tiled_rb(trans_a: bool, trans_b: bool) -> String {
    // A tile: 64 m × 16 k. Four elements per thread.
    let a_load = if trans_a {
        // a is [K,M]: contiguous along m → thread owns (k, 4 consecutive m).
        "
        {
            let kk = lin / 16u;
            let mq = (lin % 16u) * 4u;
            var v = vec4<f32>(0.0);
            let k = k0 + kk;
            if (k < k_hi) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let m = m0 + mq + j;
                    if (m < p.m) { v[j] = a[a_base + k * p.m + m]; }
                }
            }
            As[kk * 16u + mq / 4u] = v;
        }"
    } else {
        // a is [M,K]: contiguous along k → thread owns (m, 4 consecutive k).
        "
        {
            let mm = lin / 4u;
            let kq = (lin % 4u) * 4u;
            let m = m0 + mm;
            for (var j = 0u; j < 4u; j += 1u) {
                let k = k0 + kq + j;
                var v = 0.0;
                if (m < p.m && k < k_hi) { v = a[a_base + m * p.k + k]; }
                As[(kq + j) * 16u + mm / 4u][mm % 4u] = v;
            }
        }"
    };
    // B tile: 16 k × 64 n.
    let b_load = if trans_b {
        // b is [N,K]: contiguous along k → thread owns (n, 4 consecutive k).
        "
        {
            let nn = lin / 4u;
            let kq = (lin % 4u) * 4u;
            let n = n0 + nn;
            for (var j = 0u; j < 4u; j += 1u) {
                let k = k0 + kq + j;
                var v = 0.0;
                if (n < p.n && k < k_hi) { v = b[b_base + n * p.k + k]; }
                Bs[(kq + j) * 16u + nn / 4u][nn % 4u] = v;
            }
        }"
    } else {
        // b is [K,N]: contiguous along n → thread owns (k, 4 consecutive n).
        "
        {
            let kk = lin / 16u;
            let nq = (lin % 16u) * 4u;
            var v = vec4<f32>(0.0);
            let k = k0 + kk;
            if (k < k_hi) {
                for (var j = 0u; j < 4u; j += 1u) {
                    let n = n0 + nq + j;
                    if (n < p.n) { v[j] = b[b_base + k * p.n + n]; }
                }
            }
            Bs[kk * 16u + nq / 4u] = v;
        }"
    };
    format!(
        "
struct P {{ m: u32, n: u32, k: u32, a_bs: u32, b_bs: u32, batch: u32, chunk: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> As: array<vec4<f32>, 256>;
var<workgroup> Bs: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let n0 = wg.x * 64u;
    let m0 = wg.y * 64u;
    // z = slice * batch + bz: K slices are the leading dim of the partials.
    let bz = wg.z % p.batch;
    let slice = wg.z / p.batch;
    let a_base = bz * p.a_bs;
    let b_base = bz * p.b_bs;
    let k_lo = slice * p.chunk;
    let k_hi = min(k_lo + p.chunk, p.k);
    var acc: array<vec4<f32>, 4>;
    for (var k0 = k_lo; k0 < k_hi; k0 += 16u) {{
        {a_load}
        {b_load}
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk += 1u) {{
            let av = As[kk * 16u + ty];
            let bv = Bs[kk * 16u + tx];
            acc[0] += av.x * bv;
            acc[1] += av.y * bv;
            acc[2] += av.z * bv;
            acc[3] += av.w * bv;
        }}
        workgroupBarrier();
    }}
    let out_base = wg.z * p.m * p.n;
    for (var i = 0u; i < 4u; i += 1u) {{
        let m = m0 + ty * 4u + i;
        if (m < p.m) {{
            for (var j = 0u; j < 4u; j += 1u) {{
                let n = n0 + tx * 4u + j;
                if (n < p.n) {{ out[out_base + m * p.n + n] = acc[i][j]; }}
            }}
        }}
    }}
}}"
    )
}

/// Cooperative-matrix (tensor core) tiled matmul, native only
/// (`Features::EXPERIMENTAL_COOPERATIVE_MATRIX`, 32-wide subgroups).
///
/// Uses the configuration desktop Vulkan drivers actually expose —
/// **f16 A/B, f32 accumulate, 16×16×16** (NVIDIA lists no f32-input
/// shape at all; the 8×8 f32 form wgpu documents silently produces
/// zeros there). Operands are therefore rounded to f16 on the way into
/// workgroup memory: the same 10-bit mantissa as TF32, the default for
/// fp32 matmuls in cuBLAS/onnxruntime, but with f16's narrower exponent
/// range — hence opt-in, see `MatmulTile`.
///
/// Same 64×64 tile, split-K partials, immediates and grid as
/// [`matmul_tiled_rb`]; K step 32. 128 threads = 4 subgroups, each
/// owning a 32×32 quadrant as 2×2 `coop_mat16x16<f32, C>`. Tiles are
/// staged k-major (`As[k][m]`, `Bs[k][n]`), which makes A column-major
/// (`coopLoad`) and B row-major (`coopLoadT`); the C quadrant goes
/// through workgroup memory for the bounds-guarded store.
pub fn matmul_coop(trans_a: bool, trans_b: bool) -> String {
    // Tile loads: 128 threads; A 64×32 and B 32×64 → 16 elements each.
    let a_load = if trans_a {
        // a [K,M], contiguous along m: thread owns (k, 16 consecutive m).
        "
        {
            let kk = lin / 4u;
            let mq = (lin % 4u) * 16u;
            let k = k0 + kk;
            for (var j = 0u; j < 16u; j += 1u) {
                let m = m0 + mq + j;
                var v = 0.0;
                if (k < k_hi && m < p.m) { v = a[a_base + k * p.m + m]; }
                As[kk * 64u + mq + j] = f16(v);
            }
        }"
    } else {
        // a [M,K], contiguous along k: thread owns (m, 16 consecutive k).
        "
        {
            let mm = lin / 2u;
            let kq = (lin % 2u) * 16u;
            let m = m0 + mm;
            for (var j = 0u; j < 16u; j += 1u) {
                let k = k0 + kq + j;
                var v = 0.0;
                if (m < p.m && k < k_hi) { v = a[a_base + m * p.k + k]; }
                As[(kq + j) * 64u + mm] = f16(v);
            }
        }"
    };
    let b_load = if trans_b {
        // b [N,K], contiguous along k: thread owns (n, 16 consecutive k).
        "
        {
            let nn = lin / 2u;
            let kq = (lin % 2u) * 16u;
            let n = n0 + nn;
            for (var j = 0u; j < 16u; j += 1u) {
                let k = k0 + kq + j;
                var v = 0.0;
                if (n < p.n && k < k_hi) { v = b[b_base + n * p.k + k]; }
                Bs[(kq + j) * 64u + nn] = f16(v);
            }
        }"
    } else {
        // b [K,N], contiguous along n: thread owns (k, 16 consecutive n).
        "
        {
            let kk = lin / 4u;
            let nq = (lin % 4u) * 16u;
            let k = k0 + kk;
            for (var j = 0u; j < 16u; j += 1u) {
                let n = n0 + nq + j;
                var v = 0.0;
                if (k < k_hi && n < p.n) { v = b[b_base + k * p.n + n]; }
                Bs[kk * 64u + nq + j] = f16(v);
            }
        }"
    };
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;
struct P {{ m: u32, n: u32, k: u32, a_bs: u32, b_bs: u32, batch: u32, chunk: u32 }}
var<immediate> p: P;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> As: array<f16, 2048>;
var<workgroup> Bs: array<f16, 2048>;
var<workgroup> Cs: array<f32, 4096>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lin = lid.x;
    // Subgroup index, assuming 32-wide subgroups (gated at device creation).
    let sg = lin / 32u;
    let n0 = wg.x * 64u;
    let m0 = wg.y * 64u;
    let bz = wg.z % p.batch;
    let slice = wg.z / p.batch;
    let a_base = bz * p.a_bs;
    let b_base = bz * p.b_bs;
    let k_lo = slice * p.chunk;
    let k_hi = min(k_lo + p.chunk, p.k);
    let qm = sg / 2u;
    let qn = sg % 2u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();
    for (var k0 = k_lo; k0 < k_hi; k0 += 32u) {{
        {a_load}
        {b_load}
        workgroupBarrier();
        for (var ks = 0u; ks < 32u; ks += 16u) {{
            let a0 = coopLoad<coop_mat16x16<f16, A>>(&As[ks * 64u + qm * 32u], 64u);
            let a1 = coopLoad<coop_mat16x16<f16, A>>(&As[ks * 64u + qm * 32u + 16u], 64u);
            let b0 = coopLoadT<coop_mat16x16<f16, B>>(&Bs[ks * 64u + qn * 32u], 64u);
            let b1 = coopLoadT<coop_mat16x16<f16, B>>(&Bs[ks * 64u + qn * 32u + 16u], 64u);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
        }}
        workgroupBarrier();
    }}
    coopStoreT(c00, &Cs[(qm * 32u) * 64u + qn * 32u], 64u);
    coopStoreT(c01, &Cs[(qm * 32u) * 64u + qn * 32u + 16u], 64u);
    coopStoreT(c10, &Cs[(qm * 32u + 16u) * 64u + qn * 32u], 64u);
    coopStoreT(c11, &Cs[(qm * 32u + 16u) * 64u + qn * 32u + 16u], 64u);
    workgroupBarrier();
    // Guarded store: 4096 outputs over 128 threads, 32 each, coalesced.
    let out_base = wg.z * p.m * p.n;
    for (var e = lin; e < 4096u; e += 128u) {{
        let r = e / 64u;
        let c = e % 64u;
        let m = m0 + r;
        let n = n0 + c;
        if (m < p.m && n < p.n) {{ out[out_base + m * p.n + n] = Cs[e]; }}
    }}
}}"
    )
}
