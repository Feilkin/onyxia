//! Structural ONNX ops: indexing, padding, splitting, tiling, scans, and
//! selection. All of them reduce to reshape/transpose/broadcast/concat/
//! slice plus integer index arithmetic feeding `Gather`/`Scatter`.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry, convert_proto_dtype};
use onyxia_ir::graph::SymbolicContent;
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, ScatterReduce, SliceSpec, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "Size", size);
    r.register("", "Flatten", flatten);
    r.register("", "Tile", tile);
    r.register("", "Split", split);
    r.register("", "Gemm", gemm);
    r.register("", "OneHot", one_hot);
    r.register("", "EyeLike", eye_like);
    r.register("", "ArgMax", |c| arg_rule(c, true));
    r.register("", "ArgMin", |c| arg_rule(c, false));
    r.register("", "GatherElements", gather_elements);
    r.register("", "GatherND", gather_nd);
    r.register("", "ScatterElements", scatter_elements);
    r.register("", "Scatter", scatter_elements);
    r.register("", "Pad", pad);
    r.register("", "CenterCropPad", center_crop_pad);
    r.register("", "DepthToSpace", depth_to_space);
    r.register("", "SpaceToDepth", space_to_depth);
    r.register("", "CumSum", cumsum);
    r.register("", "CumProd", cumprod);
    r.register("", "TopK", topk);
    r.register("", "ReverseSequence", reverse_sequence);
    r.register("", "TensorScatter", tensor_scatter);
    r.register("", "Col2Im", col2im);
    for op in ["Compress", "NonZero", "Unique", "NonMaxSuppression"] {
        r.register("", op, data_dependent);
    }
    r.register("", "Det", |c| {
        Err(Error::Unsupported(format!(
            "node '{}': Det has no primitive decomposition (needs an LU/elimination kernel)",
            c.node_name()
        )))
    });
}

fn data_dependent(ctx: &mut LowerCtx) -> Result<()> {
    Err(Error::Unsupported(format!(
        "node '{}': data-dependent output shape (not representable in the static-shape IR)",
        ctx.node_name()
    )))
}

// ───────────────────────────── basics ──────────────────────────────────

fn size(ctx: &mut LowerCtx) -> Result<()> {
    let numel = match ctx.peek(0)? {
        crate::Lowered::Value(v) => prod(ctx.ty(*v).shape.dims()),
        crate::Lowered::Content(c) => DimExpr::constant(c.shape.iter().product::<usize>() as u64),
    };
    ctx.set_content(0, SymbolicContent::scalar(numel));
    Ok(())
}

fn flatten(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let d = dims(ctx, x);
    let r = d.len();
    let axis_raw = ctx.attr_i("axis").unwrap_or(1);
    let axis = if axis_raw < 0 {
        axis_raw + r as i64
    } else {
        axis_raw
    };
    if axis < 0 || axis as usize > r {
        return Err(Error::Shape(format!(
            "Flatten axis {axis_raw} out of range for rank {r}"
        )));
    }
    let axis = axis as usize;
    let y = reshape(ctx, x, vec![prod(&d[..axis]), prod(&d[axis..])])?;
    out(ctx, y)
}

fn tile(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let reps = require_const_ints(ctx, 1, "Tile repeats")?;
    let mut y = x;
    for (axis, &rep) in reps.iter().enumerate() {
        if rep == 1 {
            continue;
        }
        if rep < 0 {
            return Err(Error::Shape(format!("Tile repeat {rep} is negative")));
        }
        let d = dims(ctx, y);
        let mut with1 = d.clone();
        with1.insert(axis, c(1));
        let v = reshape(ctx, y, with1.clone())?;
        let mut target = with1;
        target[axis] = c(rep as u64);
        let v = broadcast(ctx, v, target)?;
        let mut merged = d;
        merged[axis] = merged[axis].clone() * c(rep as u64);
        y = reshape(ctx, v, merged)?;
    }
    out(ctx, y)
}

fn split(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let d = dims(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(0), d.len())?;
    let n_out = ctx.num_outputs();
    let sizes: Vec<DimExpr> = if ctx.has_input(1) {
        require_const_ints(ctx, 1, "Split sizes")?
            .into_iter()
            .map(|v| c(v as u64))
            .collect()
    } else if let Some(s) = ctx.attr_is("split") {
        s.into_iter().map(|v| c(v as u64)).collect()
    } else {
        let n = ctx
            .attr_i("num_outputs")
            .map(|v| v as usize)
            .unwrap_or(n_out);
        let total = d[axis].clone();
        match total.as_const() {
            Some(t) => {
                // Equal parts; when not divisible, ceil-sized chunks with a
                // smaller last one (opset 18 semantics).
                let chunk = t.div_ceil(n as u64);
                (0..n)
                    .map(|i| c(t.saturating_sub(chunk * i as u64).min(chunk)))
                    .collect()
            }
            None => {
                let part = total.div_exact(&c(n as u64)).ok_or_else(|| {
                    Error::Unsupported("Split of a symbolic dim into unequal parts".into())
                })?;
                vec![part; n]
            }
        }
    };
    if sizes.len() != n_out {
        return Err(Error::Shape(format!(
            "Split declares {n_out} outputs but {} sizes",
            sizes.len()
        )));
    }
    let mut start = c(0);
    for (i, s) in sizes.iter().enumerate() {
        let end = start.clone() + s.clone();
        let part = ctx.emit(
            Prim::Slice {
                specs: vec![SliceSpec {
                    axis,
                    start: start.clone(),
                    end: end.clone(),
                    step: 1,
                }],
            },
            &[x],
        )?;
        ctx.set_value_opt(i, part);
        start = end;
    }
    Ok(())
}

fn gemm(ctx: &mut LowerCtx) -> Result<()> {
    let (a, b) = (val(ctx, 0)?, val(ctx, 1)?);
    let dt = dtype(ctx, a);
    let alpha = ctx.attr_f("alpha").unwrap_or(1.0) as f64;
    let beta = ctx.attr_f("beta").unwrap_or(1.0) as f64;
    let trans_a = ctx.attr_i("transA").unwrap_or(0) != 0;
    let trans_b = ctx.attr_i("transB").unwrap_or(0) != 0;
    let mut y = ctx.emit(Prim::MatMul { trans_a, trans_b }, &[a, b])?;
    if alpha != 1.0 {
        let s = scalar(ctx, dt, alpha)?;
        y = mul(ctx, y, s)?;
    }
    if let Some(cv) = opt_val(ctx, 2)? {
        let mut cv = cv;
        if beta != 1.0 {
            let s = scalar(ctx, dt, beta)?;
            cv = mul(ctx, cv, s)?;
        }
        if beta != 0.0 {
            y = add(ctx, y, cv)?;
        }
    }
    out(ctx, y)
}

fn one_hot(ctx: &mut LowerCtx) -> Result<()> {
    let indices = val(ctx, 0)?;
    let depth = require_const_ints(ctx, 1, "OneHot depth")?
        .first()
        .copied()
        .ok_or_else(|| Error::Shape("OneHot depth is empty".into()))?;
    if depth < 0 {
        return Err(Error::Shape("OneHot depth must be non-negative".into()));
    }
    let values = val(ctx, 2)?;
    let dt = dtype(ctx, values);
    let r = rank(ctx, indices);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r + 1)?;
    let idx = cast(ctx, indices, DataType::I64)?;
    let idx = wrap_negative(ctx, idx, c(depth as u64))?;
    let idx = unsqueeze(ctx, idx, axis)?;
    let ramp = iota_along(ctx, c(depth as u64), r + 1, axis, DataType::I64)?;
    let hit = cmp(ctx, CmpOp::Eq, ramp, idx)?;
    let off = slice_axis(ctx, values, 0, 0, 1)?;
    let off = reshape(ctx, off, vec![])?;
    let on = slice_axis(ctx, values, 0, 1, 2)?;
    let on = reshape(ctx, on, vec![])?;
    let _ = dt;
    let y = select(ctx, hit, on, off)?;
    out(ctx, y)
}

fn eye_like(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let d = dims(ctx, x);
    if d.len() != 2 {
        return Err(Error::Shape("EyeLike requires a rank-2 input".into()));
    }
    let dt = match ctx.attr_i("dtype") {
        Some(v) => convert_proto_dtype(v)?,
        None => dtype(ctx, x),
    };
    let k = ctx.attr_i("k").unwrap_or(0);
    let rows = iota_along(ctx, d[0].clone(), 2, 0, DataType::I64)?;
    let cols = iota_along(ctx, d[1].clone(), 2, 1, DataType::I64)?;
    let kc = scalar(ctx, DataType::I64, k as f64)?;
    let shifted = add(ctx, rows, kc)?;
    let hit = cmp(ctx, CmpOp::Eq, cols, shifted)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let y = select(ctx, hit, one, zero)?;
    out(ctx, y)
}

fn arg_rule(ctx: &mut LowerCtx, is_max: bool) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(0), r)?;
    let keepdims = ctx.attr_i("keepdims").unwrap_or(1) != 0;
    let last = ctx.attr_i("select_last_index").unwrap_or(0) != 0;
    let y = arg_reduce(ctx, x, axis, keepdims, is_max, last)?;
    out(ctx, y)
}

// ─────────────────────── gather / scatter variants ─────────────────────

fn gather_elements(ctx: &mut LowerCtx) -> Result<()> {
    let (data, indices) = (val(ctx, 0)?, val(ctx, 1)?);
    let r = rank(ctx, data);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(0), r)?;
    let idx = cast(ctx, indices, DataType::I64)?;
    let dd = dims(ctx, data);
    let lin = linear_index_with_axis(ctx, &dd, idx, axis)?;
    let y = linear_gather(ctx, data, lin)?;
    out(ctx, y)
}

fn gather_nd(ctx: &mut LowerCtx) -> Result<()> {
    let (data, indices) = (val(ctx, 0)?, val(ctx, 1)?);
    let b = ctx.attr_i("batch_dims").unwrap_or(0) as usize;
    let dd = dims(ctx, data);
    let id = dims(ctx, indices);
    let q = id.len();
    let k = id[q - 1]
        .as_const()
        .ok_or_else(|| Error::Unsupported("GatherND with symbolic index width".into()))?
        as usize;
    if b + k > dd.len() {
        return Err(Error::Shape(
            "GatherND index width exceeds data rank".into(),
        ));
    }
    let idx = cast(ctx, indices, DataType::I64)?;
    let lead = &id[..q - 1]; // output leading dims
    let lr = lead.len();
    // Strides over the indexed block data[..b+k].
    let block = &dd[..b + k];
    let st = strides(block);
    let mut lin: Option<ValueId> = None;
    // Batch coordinates (iota) for dims < b.
    for d in 0..b {
        let coord = iota_along(ctx, lead[d].clone(), lr, d, DataType::I64)?;
        let s = dim_value(ctx, st[d].clone())?;
        let term = mul(ctx, coord, s)?;
        lin = Some(match lin {
            None => term,
            Some(acc) => add(ctx, acc, term)?,
        });
    }
    // Explicit coordinates from the index tuples.
    for j in 0..k {
        let col = slice_axis(ctx, idx, q - 1, j as u64, j as u64 + 1)?;
        let col = reshape(ctx, col, lead.to_vec())?;
        let coord = wrap_negative(ctx, col, dd[b + j].clone())?;
        let term = if st[b + j] == c(1) {
            coord
        } else {
            let s = dim_value(ctx, st[b + j].clone())?;
            mul(ctx, coord, s)?
        };
        lin = Some(match lin {
            None => term,
            Some(acc) => add(ctx, acc, term)?,
        });
    }
    let lin = lin.ok_or_else(|| Error::Shape("GatherND with empty index".into()))?;
    let lin = broadcast(ctx, lin, lead.to_vec())?;
    let tail = &dd[b + k..];
    let p = prod(block);
    let data2 = if tail.is_empty() {
        reshape(ctx, data, vec![p])?
    } else {
        reshape(ctx, data, vec![p, prod(tail)])?
    };
    let g = ctx.emit(Prim::Gather { axis: 0 }, &[data2, lin])?;
    let mut out_dims = lead.to_vec();
    out_dims.extend_from_slice(tail);
    let y = reshape(ctx, g, out_dims)?;
    out(ctx, y)
}

/// Full coordinate tuples `[*shape, rank]` for "iota everywhere except
/// `axis`, where the coordinate is `idx`" — ScatterElements' index form.
fn coordinate_tuples(
    ctx: &mut LowerCtx,
    data_dims: &[DimExpr],
    idx: ValueId,
    axis: usize,
) -> Result<ValueId> {
    let r = data_dims.len();
    let id = dims(ctx, idx);
    let mut cols = Vec::with_capacity(r);
    for d in 0..r {
        let coord = if d == axis {
            wrap_negative(ctx, idx, data_dims[d].clone())?
        } else {
            iota_along(ctx, id[d].clone(), r, d, DataType::I64)?
        };
        let coord = broadcast(ctx, coord, id.clone())?;
        cols.push(unsqueeze(ctx, coord, r)?);
    }
    concat(ctx, &cols, r)
}

fn scatter_elements(ctx: &mut LowerCtx) -> Result<()> {
    let reduction = super::scatter_reduction(ctx)?;
    let (data, indices, updates) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let dd = dims(ctx, data);
    let r = dd.len();
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(0), r)?;
    let idx = cast(ctx, indices, DataType::I64)?;
    let tuples = coordinate_tuples(ctx, &dd, idx, axis)?;
    let n = prod(&dims(ctx, idx));
    let tuples = reshape(ctx, tuples, vec![n.clone(), c(r as u64)])?;
    let upd = reshape(ctx, updates, vec![n])?;
    let y = ctx.emit(Prim::Scatter { reduction }, &[data, tuples, upd])?;
    out(ctx, y)
}

/// TensorScatter (opset 24): write `update` into `past_cache` at
/// `write_indices[b] + t` along `axis` (linear) or modulo the cache
/// length (circular).
fn tensor_scatter(ctx: &mut LowerCtx) -> Result<()> {
    let (cache, update) = (val(ctx, 0)?, val(ctx, 1)?);
    let cd = dims(ctx, cache);
    let ud = dims(ctx, update);
    let r = cd.len();
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-2), r)?;
    let circular = ctx.attr_s("mode").unwrap_or("linear") == "circular";
    let t = iota_along(ctx, ud[axis].clone(), r, axis, DataType::I64)?;
    let pos = if let Some(w) = opt_val(ctx, 2)? {
        let w = cast(ctx, w, DataType::I64)?;
        let mut ws = vec![c(1); r];
        ws[0] = ud[0].clone();
        let w = reshape(ctx, w, ws)?;
        add(ctx, t, w)?
    } else {
        t
    };
    let pos = if circular {
        // pos mod max_seq
        let m = dim_value(ctx, cd[axis].clone())?;
        let q = div(ctx, pos, m)?;
        let qm = mul(ctx, q, m)?;
        sub(ctx, pos, qm)?
    } else {
        pos
    };
    let pos = broadcast(ctx, pos, ud.clone())?;
    let tuples = coordinate_tuples(ctx, &cd, pos, axis)?;
    let n = prod(&ud);
    let tuples = reshape(ctx, tuples, vec![n.clone(), c(r as u64)])?;
    let upd = reshape(ctx, update, vec![n])?;
    let y = ctx.emit(
        Prim::Scatter {
            reduction: ScatterReduce::None,
        },
        &[cache, tuples, upd],
    )?;
    out(ctx, y)
}

// ─────────────────────────────── padding ───────────────────────────────

/// Index into a length-`d` axis for padded position `i` (may be outside
/// `[0, d)`), per mode.
fn pad_index(mode: &str, i: i64, d: i64) -> i64 {
    match mode {
        "edge" => i.clamp(0, d - 1),
        "wrap" => i.rem_euclid(d),
        _ => {
            // reflect (no edge repeat): period 2(d-1)
            if d == 1 {
                return 0;
            }
            let p = 2 * (d - 1);
            let m = i.rem_euclid(p);
            if m < d { m } else { p - m }
        }
    }
}

fn pad(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let d = dims(ctx, x);
    let r = d.len();
    let mode = ctx.attr_s("mode").unwrap_or("constant").to_string();
    let pads: Vec<i64> = if ctx.has_input(1) {
        require_const_ints(ctx, 1, "Pad pads")?
    } else {
        ctx.attr_is("pads")
            .ok_or_else(|| ctx.missing_attr("pads"))?
    };
    let axes: Vec<usize> = if ctx.has_input(3) {
        require_const_ints(ctx, 3, "Pad axes")?
            .iter()
            .map(|&a| ctx.norm_axis(a, r))
            .collect::<Result<_>>()?
    } else {
        (0..r).collect()
    };
    if pads.len() != 2 * axes.len() {
        return Err(Error::Shape(format!(
            "Pad: {} pad values for {} axes",
            pads.len(),
            axes.len()
        )));
    }
    let n = axes.len();
    let value = if mode == "constant" {
        if let Some(v) = opt_val(ctx, 2)? {
            let v = reshape(ctx, v, vec![])?;
            cast(ctx, v, dt)?
        } else {
            scalar(ctx, dt, 0.0)?
        }
    } else {
        scalar(ctx, dt, 0.0)?
    };
    let mut y = x;
    for (i, &axis) in axes.iter().enumerate() {
        let (before, after) = (pads[i], pads[i + n]);
        if before == 0 && after == 0 {
            continue;
        }
        if mode == "constant" {
            y = pad_axis_const(ctx, y, axis, before, after, value)?;
            continue;
        }
        // Non-constant modes: index table over the (constant) axis.
        let dlen = dims(ctx, y)[axis]
            .as_const()
            .ok_or_else(|| Error::Unsupported(format!("Pad mode '{mode}' on a symbolic axis")))?
            as i64;
        // Negative pads crop first.
        let (mut lo, mut hi) = (0i64, dlen);
        if before < 0 {
            lo = -before;
        }
        if after < 0 {
            hi += after;
        }
        if lo != 0 || hi != dlen {
            y = slice_axis(ctx, y, axis, lo as u64, hi.max(lo) as u64)?;
        }
        let dlen = hi - lo;
        let (before, after) = (before.max(0), after.max(0));
        if before == 0 && after == 0 {
            continue;
        }
        let table: Vec<i64> = (-before..dlen + after)
            .map(|i| pad_index(&mode, i, dlen))
            .collect();
        let idx = const_i64(ctx, &table, &[table.len() as u64])?;
        y = ctx.emit(Prim::Gather { axis }, &[y, idx])?;
    }
    out(ctx, y)
}

fn center_crop_pad(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let d = dims(ctx, x);
    let r = d.len();
    let shape = require_const_ints(ctx, 1, "CenterCropPad shape")?;
    let axes: Vec<usize> = match ctx.attr_is("axes") {
        Some(a) => a
            .iter()
            .map(|&v| ctx.norm_axis(v, r))
            .collect::<Result<_>>()?,
        None => (0..r).collect(),
    };
    let zero = scalar(ctx, dt, 0.0)?;
    let mut y = x;
    for (i, &axis) in axes.iter().enumerate() {
        let target = shape[i];
        let cur = dims(ctx, y)[axis]
            .as_const()
            .ok_or_else(|| Error::Unsupported("CenterCropPad on a symbolic axis".into()))?
            as i64;
        if cur > target {
            let start = (cur - target) / 2;
            y = slice_axis(ctx, y, axis, start as u64, (start + target) as u64)?;
        } else if cur < target {
            let total = target - cur;
            let before = total / 2;
            y = pad_axis_const(ctx, y, axis, before, total - before, zero)?;
        }
    }
    out(ctx, y)
}

// ───────────────────────── space/depth shuffles ────────────────────────

fn depth_to_space(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let d = dims(ctx, x);
    if d.len() != 4 {
        return Err(Error::Shape("DepthToSpace requires NCHW input".into()));
    }
    let b = ctx
        .attr_i("blocksize")
        .ok_or_else(|| ctx.missing_attr("blocksize"))? as u64;
    let (n, ch, h, w) = (d[0].clone(), d[1].clone(), d[2].clone(), d[3].clone());
    let cb = ch
        .div_exact(&c(b * b))
        .ok_or_else(|| Error::Shape("DepthToSpace: C not divisible by blocksize²".into()))?;
    let crd = ctx.attr_s("mode").unwrap_or("DCR") == "CRD";
    let y = if crd {
        let v = reshape(
            ctx,
            x,
            vec![n.clone(), cb.clone(), c(b), c(b), h.clone(), w.clone()],
        )?;
        transpose(ctx, v, &[0, 1, 4, 2, 5, 3])?
    } else {
        let v = reshape(
            ctx,
            x,
            vec![n.clone(), c(b), c(b), cb.clone(), h.clone(), w.clone()],
        )?;
        transpose(ctx, v, &[0, 3, 4, 1, 5, 2])?
    };
    let y = reshape(ctx, y, vec![n, cb, h * c(b), w * c(b)])?;
    out(ctx, y)
}

fn space_to_depth(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let d = dims(ctx, x);
    if d.len() != 4 {
        return Err(Error::Shape("SpaceToDepth requires NCHW input".into()));
    }
    let b = ctx
        .attr_i("blocksize")
        .ok_or_else(|| ctx.missing_attr("blocksize"))? as u64;
    let (n, ch, h, w) = (d[0].clone(), d[1].clone(), d[2].clone(), d[3].clone());
    let hb = h
        .div_exact(&c(b))
        .ok_or_else(|| Error::Shape("SpaceToDepth: H not divisible".into()))?;
    let wb = w
        .div_exact(&c(b))
        .ok_or_else(|| Error::Shape("SpaceToDepth: W not divisible".into()))?;
    let v = reshape(
        ctx,
        x,
        vec![n.clone(), ch.clone(), hb.clone(), c(b), wb.clone(), c(b)],
    )?;
    let v = transpose(ctx, v, &[0, 3, 5, 1, 2, 4])?;
    let y = reshape(ctx, v, vec![n, ch * c(b * b), hb, wb])?;
    out(ctx, y)
}

// ───────────────────────────── scans ───────────────────────────────────

/// Triangular `[n, n]` mask as `dt`: entry (i, j) is 1 when `i op j`.
fn tri_mask(ctx: &mut LowerCtx, n: DimExpr, op: CmpOp, dt: DataType) -> Result<ValueId> {
    let i = iota_along(ctx, n.clone(), 2, 0, DataType::I64)?;
    let j = iota_along(ctx, n, 2, 1, DataType::I64)?;
    let keep = cmp(ctx, op, i, j)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    select(ctx, keep, one, zero)
}

/// CumSum as a matmul with a triangular mask: `y[j] = Σ_i x[i]·M[i,j]`.
fn cumsum(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let axis = require_const_ints(ctx, 1, "CumSum axis")?[0];
    let axis = ctx.norm_axis(axis, r)?;
    let exclusive = ctx.attr_i("exclusive").unwrap_or(0) != 0;
    let reverse = ctx.attr_i("reverse").unwrap_or(0) != 0;
    let (xt, inv) = axis_to_last(ctx, x, axis)?;
    let n = dims(ctx, xt)[r - 1].clone();
    let op = match (reverse, exclusive) {
        (false, false) => CmpOp::Le,
        (false, true) => CmpOp::Lt,
        (true, false) => CmpOp::Ge,
        (true, true) => CmpOp::Gt,
    };
    let m = tri_mask(ctx, n.clone(), op, dt)?;
    let xt2 = if r == 1 {
        reshape(ctx, xt, vec![c(1), n.clone()])?
    } else {
        xt
    };
    let y = matmul(ctx, xt2, m)?;
    let y = if r == 1 { reshape(ctx, y, vec![n])? } else { y };
    let y = transpose(ctx, y, &inv)?;
    out(ctx, y)
}

/// CumProd by log-depth doubling (static length): `y ← y · shift(y, d)`.
fn cumprod(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let axis = require_const_ints(ctx, 1, "CumProd axis")?[0];
    let axis = ctx.norm_axis(axis, r)?;
    let exclusive = ctx.attr_i("exclusive").unwrap_or(0) != 0;
    let reverse = ctx.attr_i("reverse").unwrap_or(0) != 0;
    let (mut y, inv) = axis_to_last(ctx, x, axis)?;
    let last = r - 1;
    let n = dims(ctx, y)[last]
        .as_const()
        .ok_or_else(|| Error::Unsupported("CumProd over a symbolic axis".into()))?;
    if reverse {
        y = flip(ctx, y, last)?;
    }
    let one = scalar(ctx, dt, 1.0)?;
    let mut d = 1u64;
    while d < n {
        let mut od = dims(ctx, y);
        od[last] = c(d);
        let ones = broadcast(ctx, one, od)?;
        let head = slice_axis(ctx, y, last, 0, n - d)?;
        let shifted = concat(ctx, &[ones, head], last)?;
        y = mul(ctx, y, shifted)?;
        d *= 2;
    }
    if exclusive && n > 0 {
        let mut od = dims(ctx, y);
        od[last] = c(1);
        let ones = broadcast(ctx, one, od)?;
        let head = slice_axis(ctx, y, last, 0, n - 1)?;
        y = concat(ctx, &[ones, head], last)?;
    }
    if reverse {
        y = flip(ctx, y, last)?;
    }
    let y = transpose(ctx, y, &inv)?;
    out(ctx, y)
}

/// TopK by rank counting: `rank_i = #{j : x_j ≻ x_i or (x_j = x_i, j < i)}`,
/// then a one-hot `[n, K]` selection matmul. Quadratic in `n`.
fn topk(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let largest = ctx.attr_i("largest").unwrap_or(1) != 0;
    let k = require_const_ints(ctx, 1, "TopK k")?[0] as u64;
    let (xt, inv) = axis_to_last(ctx, x, axis)?;
    let last = r - 1;
    let n = dims(ctx, xt)[last].clone();
    let xi = unsqueeze(ctx, xt, r)?; // [..., n, 1]
    let xj = unsqueeze(ctx, xt, last)?; // [..., 1, n]
    let better = cmp(ctx, if largest { CmpOp::Gt } else { CmpOp::Lt }, xj, xi)?;
    let equal = cmp(ctx, CmpOp::Eq, xj, xi)?;
    let i = iota_along(ctx, n.clone(), 2, 0, DataType::I64)?; // [n, 1]
    let j = iota_along(ctx, n.clone(), 2, 1, DataType::I64)?; // [1, n]
    let earlier = cmp(ctx, CmpOp::Lt, j, i)?;
    let tie = binary(ctx, BinaryOp::And, equal, earlier)?;
    let ahead = binary(ctx, BinaryOp::Or, better, tie)?;
    let one_i = scalar(ctx, DataType::I64, 1.0)?;
    let zero_i = scalar(ctx, DataType::I64, 0.0)?;
    let ahead_i = select(ctx, ahead, one_i, zero_i)?;
    let rank_i = reduce(ctx, ReduceOp::Sum, ahead_i, &[r], false)?; // [..., n]
    let kk = iota_along(ctx, c(k), 2, 1, DataType::I64)?; // [1, K]
    let rank_e = unsqueeze(ctx, rank_i, r)?; // [..., n, 1]
    let hit = cmp(ctx, CmpOp::Eq, rank_e, kk)?; // [..., n, K]
    // values = x[..., 1, n] @ onehot[..., n, K]
    let one_f = scalar(ctx, dt, 1.0)?;
    let zero_f = scalar(ctx, dt, 0.0)?;
    let onehot_f = select(ctx, hit, one_f, zero_f)?;
    let xrow = unsqueeze(ctx, xt, last)?; // [..., 1, n]
    let vals = matmul(ctx, xrow, onehot_f)?; // [..., 1, K]
    let vals = squeeze(ctx, vals, last)?;
    let onehot_i = select(ctx, hit, one_i, zero_i)?;
    let irow = iota_along(ctx, n, 2, 1, DataType::I64)?; // [1, n]
    let idx = matmul(ctx, irow, onehot_i)?; // [..., 1, K]
    let idx = squeeze(ctx, idx, last)?;
    let vals = transpose(ctx, vals, &inv)?;
    let idx = transpose(ctx, idx, &inv)?;
    ctx.set_value(0, vals);
    ctx.set_value_opt(1, idx);
    Ok(())
}

fn reverse_sequence(ctx: &mut LowerCtx) -> Result<()> {
    let (x, lens) = (val(ctx, 0)?, val(ctx, 1)?);
    let d = dims(ctx, x);
    let r = d.len();
    let batch_axis = ctx.attr_i("batch_axis").unwrap_or(1) as usize;
    let time_axis = ctx.attr_i("time_axis").unwrap_or(0) as usize;
    let t = iota_along(ctx, d[time_axis].clone(), r, time_axis, DataType::I64)?;
    let lens = cast(ctx, lens, DataType::I64)?;
    let mut ls = vec![c(1); r];
    ls[batch_axis] = d[batch_axis].clone();
    let l = reshape(ctx, lens, ls)?;
    let one = scalar(ctx, DataType::I64, 1.0)?;
    let lm1 = sub(ctx, l, one)?;
    let rev = sub(ctx, lm1, t)?;
    let in_seq = cmp(ctx, CmpOp::Lt, t, l)?;
    let idx = select(ctx, in_seq, rev, t)?;
    let idx = broadcast(ctx, idx, d.clone())?;
    let lin = linear_index_with_axis(ctx, &d, idx, time_axis)?;
    let y = linear_gather(ctx, x, lin)?;
    out(ctx, y)
}

/// Col2Im: the inverse of im2col — a scatter-add of column entries into
/// the image, driven by a constant index table.
fn col2im(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?; // [N, C*prod(k), L]
    let dt = dtype(ctx, x);
    let image_shape = require_const_ints(ctx, 1, "Col2Im image_shape")?;
    let block = require_const_ints(ctx, 2, "Col2Im block_shape")?;
    let nd = image_shape.len();
    let dil = ctx.attr_is("dilations").unwrap_or(vec![1; nd]);
    let pads = ctx.attr_is("pads").unwrap_or(vec![0; 2 * nd]);
    let strides_a = ctx.attr_is("strides").unwrap_or(vec![1; nd]);
    let xd = static_dims(ctx, x, "Col2Im")?;
    let (n, ck, l) = (xd[0], xd[1], xd[2]);
    let kprod: i64 = block.iter().product();
    let ch = ck as i64 / kprod;
    // Output positions per spatial axis.
    let mut out_sp = Vec::with_capacity(nd);
    for i in 0..nd {
        let span = image_shape[i] + pads[i] + pads[i + nd] - (dil[i] * (block[i] - 1) + 1);
        out_sp.push(span / strides_a[i] + 1);
    }
    let lp: i64 = out_sp.iter().product();
    if lp as u64 != l {
        return Err(Error::Shape(format!("Col2Im: expected L={lp}, got {l}")));
    }
    let img: i64 = image_shape.iter().product();
    // For every (kernel offset kk, output position o): the image linear
    // index, or -1 when it falls in padding.
    let mut table = Vec::with_capacity((kprod * lp) as usize);
    for kk in 0..kprod {
        let mut kc = vec![0i64; nd];
        let mut rem = kk;
        for i in (0..nd).rev() {
            kc[i] = rem % block[i];
            rem /= block[i];
        }
        for o in 0..lp {
            let mut oc = vec![0i64; nd];
            let mut rem = o;
            for i in (0..nd).rev() {
                oc[i] = rem % out_sp[i];
                rem /= out_sp[i];
            }
            let mut lin = 0i64;
            let mut valid = true;
            for i in 0..nd {
                let p = oc[i] * strides_a[i] - pads[i] + kc[i] * dil[i];
                if p < 0 || p >= image_shape[i] {
                    valid = false;
                }
                lin = lin * image_shape[i] + p;
            }
            table.push(if valid { lin } else { -1 });
        }
    }
    // Scatter-add into a per-(n,c) image with one extra "trash" slot for
    // padding hits, then drop the slot.
    let planes = n * ch as u64;
    let x3 = reshape(ctx, x, vec![c(n), c(ch as u64), c(kprod as u64), c(l)])?;
    let x3 = reshape(ctx, x3, vec![c(planes), c((kprod * lp) as u64)])?;
    let table: Vec<i64> = table
        .into_iter()
        .map(|v| if v < 0 { img } else { v })
        .collect();
    // Tuple indices [planes*kprod*lp, 2] = (plane, position).
    let mut tuples = Vec::with_capacity(table.len() * planes as usize * 2);
    for p in 0..planes {
        for &t in &table {
            tuples.push(p as i64);
            tuples.push(t);
        }
    }
    let idx = const_i64(ctx, &tuples, &[(table.len() as u64) * planes, 2])?;
    let upd = reshape(ctx, x3, vec![c(planes * table.len() as u64)])?;
    let zero = scalar(ctx, dt, 0.0)?;
    let base = broadcast(ctx, zero, vec![c(planes), c(img as u64 + 1)])?;
    let acc = ctx.emit(
        Prim::Scatter {
            reduction: ScatterReduce::Add,
        },
        &[base, idx, upd],
    )?;
    let acc = slice_axis(ctx, acc, 1, 0, img as u64)?;
    let mut od = vec![c(n), c(ch as u64)];
    od.extend(image_shape.iter().map(|&v| c(v as u64)));
    let y = reshape(ctx, acc, od)?;
    let _ = UnaryOp::Neg;
    out(ctx, y)
}
