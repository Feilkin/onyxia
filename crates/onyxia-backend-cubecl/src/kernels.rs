//! CubeCL kernels for the primitive set.
//!
//! Direct ports of the wgpu backend's generated WGSL (`onyxia-backend-wgpu/
//! src/kernels.rs`) — same one-thread-per-output-element strategy, same
//! index math, so the two backends are apples-to-apples for comparison.
//!
//! Conventions:
//! - `p: &Array<u32>` carries all shape parameters; `p[0]` is always the
//!   output element count (bounds check against `ABSOLUTE_POS`). Shapes are
//!   packed as 8 zero-padded slots (`MAX_RANK`).
//! - Op selection is `#[comptime]`: CubeCL JIT-specializes one kernel per
//!   op code (the comptime branch folds away), where the wgpu backend
//!   generates distinct WGSL strings.
//! - Physical dtypes mirror the wgpu backend: f32, i32 (logical I64), u32
//!   (logical Bool).

// `#[cube]` fns are eDSL code, not plain Rust: the macro requires the
// `let mut r = 0; if comptime… { r = … }` shape (branch typing) and does
// not expand compound assignment, so clippy's suggestions here would not
// compile or would fight the macro.
#![allow(clippy::assign_op_pattern)]
#![allow(unused_assignments)]

use cubecl::prelude::*;

pub const MAX_RANK: usize = 8;

// Comptime op codes (binary).
pub const OP_ADD: u32 = 0;
pub const OP_SUB: u32 = 1;
pub const OP_MUL: u32 = 2;
pub const OP_DIV: u32 = 3;
pub const OP_POW: u32 = 4;
pub const OP_MAX: u32 = 5;
pub const OP_MIN: u32 = 6;
pub const OP_AND: u32 = 7;
pub const OP_OR: u32 = 8;
pub const OP_XOR: u32 = 9;

// Comptime op codes (compare).
pub const CMP_EQ: u32 = 0;
pub const CMP_NE: u32 = 1;
pub const CMP_LT: u32 = 2;
pub const CMP_LE: u32 = 3;
pub const CMP_GT: u32 = 4;
pub const CMP_GE: u32 = 5;

// Comptime op codes (unary).
pub const UN_NEG: u32 = 0;
pub const UN_SQRT: u32 = 1;
pub const UN_RSQRT: u32 = 2;
pub const UN_EXP: u32 = 3;
pub const UN_LOG: u32 = 4;
pub const UN_COS: u32 = 5;
pub const UN_SIN: u32 = 6;
pub const UN_TANH: u32 = 7;
pub const UN_ERF: u32 = 8;
pub const UN_ABS: u32 = 9;
pub const UN_FLOOR: u32 = 10;
pub const UN_CEIL: u32 = 11;

// Comptime op codes (reduce).
pub const RED_SUM: u32 = 0;
pub const RED_MEAN: u32 = 1;
pub const RED_MAX: u32 = 2;
pub const RED_MIN: u32 = 3;
pub const RED_PROD: u32 = 4;

/// Map an output linear index to a (broadcast) source linear index.
/// `out_shape`/`in_shape` are 8-slot blocks inside `p` at the given offsets;
/// the input is right-aligned and size-1 dims broadcast.
#[cube]
fn src_index(
    idx: usize,
    p: &Array<u32>,
    out_at: usize,
    out_rank: usize,
    in_at: usize,
    in_rank: usize,
) -> usize {
    let mut rem = idx;
    let mut src = 0usize;
    let mut stride = 1usize;
    for kk in 0..out_rank {
        let d = out_rank - 1 - kk;
        let dim = p[out_at + d] as usize;
        let coord = rem % dim;
        rem = rem / dim;
        if d + in_rank >= out_rank {
            let in_d = d + in_rank - out_rank;
            let in_dim = p[in_at + in_d] as usize;
            if in_dim != 1 {
                src += coord * stride;
            }
            stride *= in_dim;
        }
    }
    src
}

/// Coordinate `d` of linear index `idx` under the 8-slot shape at
/// `shape_at` (rank `rank`).
#[cube]
fn coord_of(idx: usize, p: &Array<u32>, shape_at: usize, rank: usize, d: usize) -> usize {
    let mut stride = 1usize;
    for dd in (d + 1)..rank {
        stride *= p[shape_at + dd] as usize;
    }
    (idx / stride) % (p[shape_at + d] as usize)
}

/// Binary op with N-D broadcasting (float).
/// p = [size, out_rank, a_rank, b_rank, out8@4, a8@12, b8@20]
#[cube(launch_unchecked)]
pub fn binary_f32(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    p: &Array<u32>,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let av = a[src_index(idx, p, 4, p[1] as usize, 12, p[2] as usize)];
        let bv = b[src_index(idx, p, 4, p[1] as usize, 20, p[3] as usize)];
        let r = if comptime![op == OP_ADD] {
            av + bv
        } else if comptime![op == OP_SUB] {
            av - bv
        } else if comptime![op == OP_MUL] {
            av * bv
        } else if comptime![op == OP_DIV] {
            av / bv
        } else if comptime![op == OP_POW] {
            av.powf(bv)
        } else if comptime![op == OP_MAX] {
            av.max(bv)
        } else {
            av.min(bv)
        };
        out[idx] = r;
    }
}

/// Binary op with N-D broadcasting (i32). Same layout as [`binary_f32`].
#[cube(launch_unchecked)]
pub fn binary_i32(
    a: &Array<i32>,
    b: &Array<i32>,
    out: &mut Array<i32>,
    p: &Array<u32>,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let av = a[src_index(idx, p, 4, p[1] as usize, 12, p[2] as usize)];
        let bv = b[src_index(idx, p, 4, p[1] as usize, 20, p[3] as usize)];
        let mut r = 0i32;
        if comptime![op == OP_ADD] {
            r = av + bv;
        } else if comptime![op == OP_SUB] {
            r = av - bv;
        } else if comptime![op == OP_MUL] {
            r = av * bv;
        } else if comptime![op == OP_DIV] {
            r = av / bv;
        } else if comptime![op == OP_MAX] {
            r = bv;
            if av > bv {
                r = av;
            }
        } else {
            r = bv;
            if av < bv {
                r = av;
            }
        }
        out[idx] = r;
    }
}

/// Logical binary on u32 0/1 (Bool physical). Same layout as [`binary_f32`].
#[cube(launch_unchecked)]
pub fn binary_u32(
    a: &Array<u32>,
    b: &Array<u32>,
    out: &mut Array<u32>,
    p: &Array<u32>,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let av = a[src_index(idx, p, 4, p[1] as usize, 12, p[2] as usize)];
        let bv = b[src_index(idx, p, 4, p[1] as usize, 20, p[3] as usize)];
        let r = if comptime![op == OP_AND] {
            av & bv
        } else if comptime![op == OP_OR] {
            av | bv
        } else {
            av ^ bv
        };
        out[idx] = r;
    }
}

/// Comparison with broadcasting (float) → u32 0/1. Layout as [`binary_f32`].
#[cube(launch_unchecked)]
pub fn compare_f32(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<u32>,
    p: &Array<u32>,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let av = a[src_index(idx, p, 4, p[1] as usize, 12, p[2] as usize)];
        let bv = b[src_index(idx, p, 4, p[1] as usize, 20, p[3] as usize)];
        let c = if comptime![op == CMP_EQ] {
            av == bv
        } else if comptime![op == CMP_NE] {
            av != bv
        } else if comptime![op == CMP_LT] {
            av < bv
        } else if comptime![op == CMP_LE] {
            av <= bv
        } else if comptime![op == CMP_GT] {
            av > bv
        } else {
            av >= bv
        };
        let mut r = 0u32;
        if c {
            r = 1u32;
        }
        out[idx] = r;
    }
}

/// Comparison with broadcasting (i32) → u32 0/1. Layout as [`binary_f32`].
#[cube(launch_unchecked)]
pub fn compare_i32(
    a: &Array<i32>,
    b: &Array<i32>,
    out: &mut Array<u32>,
    p: &Array<u32>,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let av = a[src_index(idx, p, 4, p[1] as usize, 12, p[2] as usize)];
        let bv = b[src_index(idx, p, 4, p[1] as usize, 20, p[3] as usize)];
        let c = if comptime![op == CMP_EQ] {
            av == bv
        } else if comptime![op == CMP_NE] {
            av != bv
        } else if comptime![op == CMP_LT] {
            av < bv
        } else if comptime![op == CMP_LE] {
            av <= bv
        } else if comptime![op == CMP_GT] {
            av > bv
        } else {
            av >= bv
        };
        let mut r = 0u32;
        if c {
            r = 1u32;
        }
        out[idx] = r;
    }
}

/// Element-wise unary (float). p = [size]
#[cube(launch_unchecked)]
pub fn unary_f32(x: &Array<f32>, out: &mut Array<f32>, p: &Array<u32>, #[comptime] op: u32) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let v = x[idx];
        let r = if comptime![op == UN_NEG] {
            -v
        } else if comptime![op == UN_SQRT] {
            v.sqrt()
        } else if comptime![op == UN_RSQRT] {
            v.inverse_sqrt()
        } else if comptime![op == UN_EXP] {
            v.exp()
        } else if comptime![op == UN_LOG] {
            v.ln()
        } else if comptime![op == UN_COS] {
            v.cos()
        } else if comptime![op == UN_SIN] {
            v.sin()
        } else if comptime![op == UN_TANH] {
            v.tanh()
        } else if comptime![op == UN_ERF] {
            cubecl::frontend::Erf::erf(v)
        } else if comptime![op == UN_FLOOR] {
            v.floor()
        } else if comptime![op == UN_CEIL] {
            v.ceil()
        } else {
            v.abs()
        };
        out[idx] = r;
    }
}

/// Element-wise unary (i32): Neg/Abs. p = [size]
#[cube(launch_unchecked)]
pub fn unary_i32(x: &Array<i32>, out: &mut Array<i32>, p: &Array<u32>, #[comptime] op: u32) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let v = x[idx];
        let mut r = v;
        if comptime![op == UN_NEG] {
            r = -v;
        } else {
            if v < 0 {
                r = -v;
            }
        }
        out[idx] = r;
    }
}

/// Logical Not on u32 0/1. p = [size]
#[cube(launch_unchecked)]
pub fn not_u32(x: &Array<u32>, out: &mut Array<u32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let mut r = 0u32;
        if x[idx] == 0 {
            r = 1u32;
        }
        out[idx] = r;
    }
}

/// Dtype conversions (one concrete kernel per physical pair). p = [size]
#[cube(launch_unchecked)]
pub fn cast_f32_i32(x: &Array<f32>, out: &mut Array<i32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = i32::cast_from(x[idx]);
    }
}

#[cube(launch_unchecked)]
pub fn cast_i32_f32(x: &Array<i32>, out: &mut Array<f32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = f32::cast_from(x[idx]);
    }
}

#[cube(launch_unchecked)]
pub fn cast_u32_f32(x: &Array<u32>, out: &mut Array<f32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = f32::cast_from(x[idx]);
    }
}

#[cube(launch_unchecked)]
pub fn cast_u32_i32(x: &Array<u32>, out: &mut Array<i32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = i32::cast_from(x[idx]);
    }
}

/// Where(cond, a, b) with broadcasting.
/// p = [size, out_rank, c_rank, a_rank, b_rank, out8@5, c8@13, a8@21, b8@29]
#[cube(launch_unchecked)]
pub fn select3<N: Numeric>(
    c: &Array<u32>,
    a: &Array<N>,
    b: &Array<N>,
    out: &mut Array<N>,
    p: &Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let cv = c[src_index(idx, p, 5, p[1] as usize, 13, p[2] as usize)];
        let av = a[src_index(idx, p, 5, p[1] as usize, 21, p[3] as usize)];
        let bv = b[src_index(idx, p, 5, p[1] as usize, 29, p[4] as usize)];
        let mut r = bv;
        if cv != 0 {
            r = av;
        }
        out[idx] = r;
    }
}

/// Batched matmul, one thread per output element.
/// p = [size, m, n, k, a_batch_stride, b_batch_stride, trans_a, trans_b]
#[cube(launch_unchecked)]
pub fn matmul_f32(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let m = p[1] as usize;
        let n = p[2] as usize;
        let k = p[3] as usize;
        let bi = idx / (m * n);
        let r = (idx / n) % m;
        let c = idx % n;
        let a_base = bi * (p[4] as usize);
        let b_base = bi * (p[5] as usize);
        let trans_a = p[6];
        let trans_b = p[7];
        let mut acc = 0.0f32;
        for kk in 0..k {
            let mut ae = r * k + kk;
            if trans_a == 1 {
                ae = kk * m + r;
            }
            let mut be = kk * n + c;
            if trans_b == 1 {
                be = c * k + kk;
            }
            acc += a[a_base + ae] * b[b_base + be];
        }
        out[idx] = acc;
    }
}

/// Reduction over an axes bitmask, one thread per output element (float).
/// p = [size, in_rank, axes_mask, reduce_count, in8@4]
#[cube(launch_unchecked)]
pub fn reduce_f32(x: &Array<f32>, out: &mut Array<f32>, p: &Array<u32>, #[comptime] op: u32) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let in_rank = p[1] as usize;
        let mask = p[2];
        let count = p[3] as usize;
        // Base linear index from the non-reduced coordinates.
        let mut out_rem = idx;
        let mut base = 0usize;
        let mut stride = 1usize;
        for kk in 0..in_rank {
            let d = in_rank - 1 - kk;
            let dim = p[4 + d] as usize;
            if (mask & (1u32 << (d as u32))) == 0 {
                let coord = out_rem % dim;
                out_rem = out_rem / dim;
                base += coord * stride;
            }
            stride *= dim;
        }
        let init = comptime![if op == RED_MAX {
            f32::MIN
        } else if op == RED_MIN {
            f32::MAX
        } else if op == RED_PROD {
            1.0f32
        } else {
            0.0f32
        }];
        let mut acc = f32::new(init);
        for r in 0..count {
            let mut r_rem = r;
            let mut off = 0usize;
            let mut s2 = 1usize;
            for kk in 0..in_rank {
                let d = in_rank - 1 - kk;
                let dim = p[4 + d] as usize;
                if (mask & (1u32 << (d as u32))) != 0 {
                    let coord = r_rem % dim;
                    r_rem = r_rem / dim;
                    off += coord * s2;
                }
                s2 *= dim;
            }
            let v = x[base + off];
            if comptime![op == RED_MAX] {
                acc = acc.max(v);
            } else if comptime![op == RED_MIN] {
                acc = acc.min(v);
            } else if comptime![op == RED_PROD] {
                acc *= v;
            } else {
                acc += v;
            }
        }
        if comptime![op == RED_MEAN] {
            acc /= f32::cast_from(count);
        }
        out[idx] = acc;
    }
}

/// Transpose by permutation, one thread per output element.
/// p = [size, rank, perm8@2, in8@10, out8@18]
#[cube(launch_unchecked)]
pub fn transpose<N: Numeric>(x: &Array<N>, out: &mut Array<N>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let rank = p[1] as usize;
        // Input index: input dim in_d gets the coordinate of the output dim
        // od with perm[od] == in_d.
        let mut ii = 0usize;
        let mut stride = 1usize;
        for kk in 0..rank {
            let in_d = rank - 1 - kk;
            let mut coord = 0usize;
            for od in 0..rank {
                if p[2 + od] as usize == in_d {
                    coord = coord_of(idx, p, 18, rank, od);
                }
            }
            ii += coord * stride;
            stride *= p[10 + in_d] as usize;
        }
        out[idx] = x[ii];
    }
}

/// Broadcast (Expand) copy. p = [size, out_rank, in_rank, out8@3, in8@11]
#[cube(launch_unchecked)]
pub fn broadcast<N: Numeric>(x: &Array<N>, out: &mut Array<N>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = x[src_index(idx, p, 3, p[1] as usize, 11, p[2] as usize)];
    }
}

/// Copy one concat input into its slot of the output (one launch per
/// input; threads over the *input*).
/// p = [in_size, rank, axis, axis_offset, in8@4, out8@12]
#[cube(launch_unchecked)]
pub fn concat_emplace<N: Numeric>(x: &Array<N>, out: &mut Array<N>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let rank = p[1] as usize;
        let mut rem = idx;
        let mut oi = 0usize;
        let mut stride = 1usize;
        for kk in 0..rank {
            let d = rank - 1 - kk;
            let dim = p[4 + d] as usize;
            let mut coord = rem % dim;
            rem = rem / dim;
            if d == p[2] as usize {
                coord += p[3] as usize;
            }
            oi += coord * stride;
            stride *= p[12 + d] as usize;
        }
        out[oi] = x[idx];
    }
}

/// Strided slice. Steps are bitcast i32 (may be negative).
/// p = [size, rank, start8@2, step8@10, in8@18, out8@26]
#[cube(launch_unchecked)]
pub fn slice_copy<N: Numeric>(x: &Array<N>, out: &mut Array<N>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let rank = p[1] as usize;
        let mut rem = idx;
        let mut ii = 0usize;
        let mut stride = 1usize;
        for kk in 0..rank {
            let d = rank - 1 - kk;
            let odim = p[26 + d] as usize;
            let oc = rem % odim;
            rem = rem / odim;
            let step = i32::cast_from(p[10 + d]);
            let ic = i32::cast_from(p[2 + d]) + i32::cast_from(oc) * step;
            ii += (ic as u32 as usize) * stride;
            stride *= p[18 + d] as usize;
        }
        out[idx] = x[ii];
    }
}

/// ONNX Gather along an axis (indices are i32, negatives wrap).
/// p = [size, axis, data_rank, indices_rank, data8@4, indices8@12, out8@20]
#[cube(launch_unchecked)]
pub fn gather<N: Numeric>(
    data: &Array<N>,
    indices: &Array<i32>,
    out: &mut Array<N>,
    p: &Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        let axis = p[1] as usize;
        let data_rank = p[2] as usize;
        let indices_rank = p[3] as usize;
        let out_rank = data_rank - 1 + indices_rank;
        // Indices linear index from the middle coordinate block.
        let mut ii = 0usize;
        let mut stride = 1usize;
        for kk in 0..indices_rank {
            let d = indices_rank - 1 - kk;
            ii += coord_of(idx, p, 20, out_rank, axis + d) * stride;
            stride *= p[12 + d] as usize;
        }
        let dim = i32::cast_from(p[4 + axis]);
        let mut iv = indices[ii];
        if iv < 0 {
            iv += dim;
        }
        if iv < 0 {
            iv = 0;
        }
        if iv > dim - 1 {
            iv = dim - 1;
        }
        // Data linear index.
        let mut di = 0usize;
        stride = 1usize;
        for kk in 0..data_rank {
            let d = data_rank - 1 - kk;
            let mut coord = iv as u32 as usize;
            if d < axis {
                coord = coord_of(idx, p, 20, out_rank, d);
            } else if d > axis {
                coord = coord_of(idx, p, 20, out_rank, d + indices_rank - 1);
            }
            di += coord * stride;
            stride *= p[4 + d] as usize;
        }
        out[idx] = data[di];
    }
}

/// Integer ramp 0..n (float output). p = [size]
#[cube(launch_unchecked)]
pub fn iota_f32(out: &mut Array<f32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = f32::cast_from(idx);
    }
}

/// Integer ramp 0..n (i32 output). p = [size]
#[cube(launch_unchecked)]
pub fn iota_i32(out: &mut Array<i32>, p: &Array<u32>) {
    let idx = ABSOLUTE_POS;
    if idx < p[0] as usize {
        out[idx] = i32::cast_from(idx);
    }
}
