//! Reductions beyond the primitive kinds, and the normalization family.
//!
//! `LayerNormalization` and `RMSNormalization` are emitted as composites
//! (backends may fuse them; the portable decompositions live in
//! `onyxia_ir::decomp`). The rest are emitted directly as primitives.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry, attrs};
use onyxia_ir::prim::{CmpOp, Prim, ReduceOp, UnaryOp};
use onyxia_ir::{AttrValue, DataType, DimExpr, Error, Result, SymbolicShape, TensorType, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "ReduceL1", |c| reduce_ext(c, Ext::L1));
    r.register("", "ReduceL2", |c| reduce_ext(c, Ext::L2));
    r.register("", "ReduceLogSum", |c| reduce_ext(c, Ext::LogSum));
    r.register("", "ReduceLogSumExp", |c| reduce_ext(c, Ext::LogSumExp));
    r.register("", "ReduceSumSquare", |c| reduce_ext(c, Ext::SumSquare));
    r.register("", "LayerNormalization", layer_norm);
    r.register("", "RMSNormalization", rms_norm);
    r.register("", "GroupNormalization", group_norm);
    r.register("", "InstanceNormalization", instance_norm);
    r.register("", "BatchNormalization", batch_norm);
    r.register("", "MeanVarianceNormalization", mvn);
    r.register("", "LpNormalization", lp_norm);
    r.register("", "LRN", lrn);
}

#[derive(Clone, Copy)]
enum Ext {
    L1,
    L2,
    LogSum,
    LogSumExp,
    SumSquare,
}

/// Axes of a Reduce* op: input (opset 18) or attribute, defaulting to all.
/// Returns `None` for the noop case (empty axes with `noop_with_empty_axes`).
fn reduce_axes(ctx: &LowerCtx, rank: usize) -> Result<Option<Vec<usize>>> {
    let raw = if ctx.has_input(1) {
        Some(require_const_ints(ctx, 1, "reduce axes")?)
    } else {
        ctx.attr_is("axes")
    };
    let noop = ctx.attr_i("noop_with_empty_axes").unwrap_or(0) != 0;
    match raw {
        Some(a) if !a.is_empty() => {
            let mut v: Vec<usize> = a
                .iter()
                .map(|&x| ctx.norm_axis(x, rank))
                .collect::<Result<_>>()?;
            v.sort_unstable();
            v.dedup();
            Ok(Some(v))
        }
        _ if noop => Ok(None),
        _ => Ok(Some((0..rank).collect())),
    }
}

fn reduce_ext(ctx: &mut LowerCtx, ext: Ext) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let keepdims = ctx.attr_i("keepdims").unwrap_or(1) != 0;
    let Some(axes) = reduce_axes(ctx, r)? else {
        return out(ctx, x);
    };
    let y = match ext {
        Ext::L1 => {
            let a = unary(ctx, UnaryOp::Abs, x)?;
            reduce(ctx, ReduceOp::Sum, a, &axes, keepdims)?
        }
        Ext::L2 => {
            let sq = mul(ctx, x, x)?;
            let s = reduce(ctx, ReduceOp::Sum, sq, &axes, keepdims)?;
            unary(ctx, UnaryOp::Sqrt, s)?
        }
        Ext::SumSquare => {
            let sq = mul(ctx, x, x)?;
            reduce(ctx, ReduceOp::Sum, sq, &axes, keepdims)?
        }
        Ext::LogSum => {
            let s = reduce(ctx, ReduceOp::Sum, x, &axes, keepdims)?;
            unary(ctx, UnaryOp::Log, s)?
        }
        Ext::LogSumExp => {
            // Max-shifted for stability; an all-(-inf) row would give
            // NaN via (-inf) - (-inf), so clamp the shift to finite.
            let dt = dtype(ctx, x);
            let m = reduce(ctx, ReduceOp::Max, x, &axes, true)?;
            let lo = scalar(ctx, dt, lowest_finite(dt))?;
            let hi = scalar(ctx, dt, -lowest_finite(dt))?;
            let mc = max(ctx, m, lo)?;
            let mc = min(ctx, mc, hi)?;
            let sh = sub(ctx, x, mc)?;
            let e = unary(ctx, UnaryOp::Exp, sh)?;
            let s = reduce(ctx, ReduceOp::Sum, e, &axes, keepdims)?;
            let l = unary(ctx, UnaryOp::Log, s)?;
            let mk = if keepdims {
                mc
            } else {
                let mut d = dims(ctx, mc);
                for &a in axes.iter().rev() {
                    d.remove(a);
                }
                reshape(ctx, mc, d)?
            };
            add(ctx, l, mk)?
        }
    };
    out(ctx, y)
}

fn lowest_finite(dt: DataType) -> f64 {
    match dt {
        DataType::F16 => -65504.0,
        _ => f32::MIN as f64,
    }
}

// ───────────────────────── normalizations ──────────────────────────────

/// Shape with the trailing axes (from `axis`) collapsed to 1.
fn stat_shape(d: &[DimExpr], axis: usize) -> Vec<DimExpr> {
    d.iter()
        .enumerate()
        .map(|(i, e)| if i >= axis { c(1) } else { e.clone() })
        .collect()
}

fn layer_norm(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let scale = val(ctx, 1)?;
    let bias = opt_val(ctx, 2)?;
    let d = dims(ctx, x);
    let r = d.len();
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let eps = ctx.attr_f("epsilon").unwrap_or(1e-5) as f64;
    let dt = dtype(ctx, x);
    let stat_ty = TensorType::new(dt, SymbolicShape(stat_shape(&d, axis)));
    let mut inputs = vec![x, scale];
    if let Some(b) = bias {
        inputs.push(b);
    }
    let x_ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "LayerNormalization",
        attrs(vec![
            ("axis", AttrValue::Int(axis as i64)),
            ("epsilon", AttrValue::Float(eps)),
            ("has_bias", AttrValue::Int(bias.is_some() as i64)),
        ]),
        &inputs,
        vec![x_ty, stat_ty.clone(), stat_ty],
    )?;
    ctx.set_value(0, outs[0]);
    ctx.set_value_opt(1, outs[1]);
    ctx.set_value_opt(2, outs[2]);
    Ok(())
}

fn rms_norm(ctx: &mut LowerCtx) -> Result<()> {
    let (x, scale) = (val(ctx, 0)?, val(ctx, 1)?);
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let eps = ctx.attr_f("epsilon").unwrap_or(1e-5) as f64;
    let x_ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "RMSNormalization",
        attrs(vec![
            ("axis", AttrValue::Int(axis as i64)),
            ("epsilon", AttrValue::Float(eps)),
        ]),
        &[x, scale],
        vec![x_ty],
    )?;
    out(ctx, outs[0])
}

/// `(x - mean) / sqrt(var + eps)` over `axes` (population variance).
fn standardize(ctx: &mut LowerCtx, x: ValueId, axes: &[usize], eps: f64) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let mean = reduce(ctx, ReduceOp::Mean, x, axes, true)?;
    let dlt = sub(ctx, x, mean)?;
    let sq = mul(ctx, dlt, dlt)?;
    let var = reduce(ctx, ReduceOp::Mean, sq, axes, true)?;
    let e = scalar(ctx, dt, eps)?;
    let ve = add(ctx, var, e)?;
    let inv = unary(ctx, UnaryOp::Rsqrt, ve)?;
    mul(ctx, dlt, inv)
}

/// Reshape a per-channel `[C]` vector to broadcast against `[N, C, ...]`.
fn per_channel(ctx: &mut LowerCtx, v: ValueId, rank: usize) -> Result<ValueId> {
    let n = dims(ctx, v)[0].clone();
    let mut s = vec![c(1); rank];
    s[1] = n;
    reshape(ctx, v, s)
}

fn group_norm(ctx: &mut LowerCtx) -> Result<()> {
    let (x, scale, bias) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let d = dims(ctx, x);
    let r = d.len();
    let g = ctx
        .attr_i("num_groups")
        .ok_or_else(|| ctx.missing_attr("num_groups"))? as u64;
    let eps = ctx.attr_f("epsilon").unwrap_or(1e-5) as f64;
    let (n, ch) = (d[0].clone(), d[1].clone());
    let cg = ch
        .div_exact(&c(g))
        .ok_or_else(|| Error::Shape("GroupNormalization: C not divisible by num_groups".into()))?;
    let spatial = prod(&d[2..]);
    let xg = reshape(ctx, x, vec![n.clone(), c(g), cg * spatial])?;
    let normed = standardize(ctx, xg, &[2], eps)?;
    let normed = reshape(ctx, normed, d.clone())?;
    let s = per_channel(ctx, scale, r)?;
    let b = per_channel(ctx, bias, r)?;
    let y = mul(ctx, normed, s)?;
    let y = add(ctx, y, b)?;
    out(ctx, y)
}

fn instance_norm(ctx: &mut LowerCtx) -> Result<()> {
    let (x, scale, bias) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let r = rank(ctx, x);
    let eps = ctx.attr_f("epsilon").unwrap_or(1e-5) as f64;
    let axes: Vec<usize> = (2..r).collect();
    let normed = standardize(ctx, x, &axes, eps)?;
    let s = per_channel(ctx, scale, r)?;
    let b = per_channel(ctx, bias, r)?;
    let y = mul(ctx, normed, s)?;
    let y = add(ctx, y, b)?;
    out(ctx, y)
}

fn batch_norm(ctx: &mut LowerCtx) -> Result<()> {
    let training = ctx.attr_i("training_mode").unwrap_or(0) != 0;
    let x = val(ctx, 0)?;
    let (scale, bias, mut mean, mut var) = (val(ctx, 1)?, val(ctx, 2)?, val(ctx, 3)?, val(ctx, 4)?);
    if training {
        // Batch statistics over every axis but the channel; running
        // stats blend them with the inputs by `momentum`.
        let r = rank(ctx, x);
        let axes: Vec<usize> = (0..r).filter(|&a| a != 1).collect();
        let momentum = ctx.attr_f("momentum").unwrap_or(0.9) as f64;
        let dt = dtype(ctx, x);
        let cur_mean = reduce(ctx, ReduceOp::Mean, x, &axes, false)?;
        let m_keep = reduce(ctx, ReduceOp::Mean, x, &axes, true)?;
        let d = sub(ctx, x, m_keep)?;
        let dd = mul(ctx, d, d)?;
        let cur_var = reduce(ctx, ReduceOp::Mean, dd, &axes, false)?;
        let mom = scalar(ctx, dt, momentum)?;
        let inv_mom = scalar(ctx, dt, 1.0 - momentum)?;
        let rm_a = mul(ctx, mean, mom)?;
        let rm_b = mul(ctx, cur_mean, inv_mom)?;
        let running_mean = add(ctx, rm_a, rm_b)?;
        let rv_a = mul(ctx, var, mom)?;
        let rv_b = mul(ctx, cur_var, inv_mom)?;
        let running_var = add(ctx, rv_a, rv_b)?;
        ctx.set_value_opt(1, running_mean);
        ctx.set_value_opt(2, running_var);
        mean = cur_mean;
        var = cur_var;
    }
    let r = rank(ctx, x).max(2);
    let dt = dtype(ctx, x);
    let eps = ctx.attr_f("epsilon").unwrap_or(1e-5) as f64;
    // Rank-1 input: C is assumed 1 — treat as [N, 1].
    let x1 = if rank(ctx, x) == 1 {
        let n = dims(ctx, x)[0].clone();
        reshape(ctx, x, vec![n, c(1)])?
    } else {
        x
    };
    let s = per_channel(ctx, scale, r)?;
    let b = per_channel(ctx, bias, r)?;
    let m = per_channel(ctx, mean, r)?;
    let v = per_channel(ctx, var, r)?;
    let e = scalar(ctx, dt, eps)?;
    let ve = add(ctx, v, e)?;
    let inv = unary(ctx, UnaryOp::Rsqrt, ve)?;
    let xm = sub(ctx, x1, m)?;
    let y = mul(ctx, xm, inv)?;
    let y = mul(ctx, y, s)?;
    let y = add(ctx, y, b)?;
    let y = if rank(ctx, x) == 1 {
        let d = dims(ctx, x);
        reshape(ctx, y, d)?
    } else {
        y
    };
    out(ctx, y)
}

/// MeanVarianceNormalization: `(x - mean) / sqrt(var + 1e-9)` over `axes`
/// (the 1e-9 matches the ONNX function body).
fn mvn(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let axes: Vec<usize> = match ctx.attr_is("axes") {
        Some(a) => a
            .iter()
            .map(|&v| ctx.norm_axis(v, r))
            .collect::<Result<_>>()?,
        None => [0usize, 2, 3].into_iter().filter(|&a| a < r).collect(),
    };
    let dt = dtype(ctx, x);
    // The function body computes var as E[x²] - E[x]², then sqrt(var + eps).
    let mean = reduce(ctx, ReduceOp::Mean, x, &axes, true)?;
    let sq = mul(ctx, x, x)?;
    let msq = reduce(ctx, ReduceOp::Mean, sq, &axes, true)?;
    let mm = mul(ctx, mean, mean)?;
    let var = sub(ctx, msq, mm)?;
    let e = scalar(ctx, dt, 1e-9)?;
    let ve = add(ctx, var, e)?;
    let sd = unary(ctx, UnaryOp::Sqrt, ve)?;
    let xm = sub(ctx, x, mean)?;
    let y = div(ctx, xm, sd)?;
    out(ctx, y)
}

fn lp_norm(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let p = ctx.attr_i("p").unwrap_or(2);
    let dt = dtype(ctx, x);
    let norm = match p {
        1 => {
            let a = unary(ctx, UnaryOp::Abs, x)?;
            reduce(ctx, ReduceOp::Sum, a, &[axis], true)?
        }
        2 => {
            let sq = mul(ctx, x, x)?;
            let s = reduce(ctx, ReduceOp::Sum, sq, &[axis], true)?;
            unary(ctx, UnaryOp::Sqrt, s)?
        }
        other => return Err(Error::Unsupported(format!("LpNormalization p={other}"))),
    };
    // Zero norm → zero output (spec).
    let zero = scalar(ctx, dt, 0.0)?;
    let is_zero = cmp(ctx, CmpOp::Eq, norm, zero)?;
    let one = scalar(ctx, dt, 1.0)?;
    let safe = select(ctx, is_zero, one, norm)?;
    let y = div(ctx, x, safe)?;
    let y = select(ctx, is_zero, zero, y)?;
    out(ctx, y)
}

/// LRN: `x / (bias + alpha/size · Σ_{window over channels} x²)^beta`.
fn lrn(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let size = ctx.attr_i("size").ok_or_else(|| ctx.missing_attr("size"))?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1e-4) as f64;
    let beta = ctx.attr_f("beta").unwrap_or(0.75) as f64;
    let bias = ctx.attr_f("bias").unwrap_or(1.0) as f64;
    let ch = dims(ctx, x)[1]
        .as_const()
        .ok_or_else(|| Error::Unsupported("LRN with a symbolic channel dim".into()))?;
    let before = (size - 1) / 2;
    let after = size - 1 - before;
    let sq = mul(ctx, x, x)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let padded = pad_axis_const(ctx, sq, 1, before, after, zero)?;
    let mut acc: Option<ValueId> = None;
    for k in 0..size as u64 {
        let win = slice_axis(ctx, padded, 1, k, k + ch)?;
        acc = Some(match acc {
            None => win,
            Some(a) => add(ctx, a, win)?,
        });
    }
    let ssum = acc.expect("size >= 1");
    let coef = scalar(ctx, dt, alpha / size as f64)?;
    let b = scalar(ctx, dt, bias)?;
    let be = scalar(ctx, dt, beta)?;
    let t = mul(ctx, ssum, coef)?;
    let t = add(ctx, t, b)?;
    let denom = ctx.emit(Prim::Binary(onyxia_ir::BinaryOp::Pow), &[t, be])?;
    let y = div(ctx, x, denom)?;
    out(ctx, y)
}
