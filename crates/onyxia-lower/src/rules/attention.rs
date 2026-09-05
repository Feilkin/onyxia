//! The standard-domain attention ops (opset 23+): `Attention` as a
//! composite (fusable; decomposition in `onyxia_ir::decomp`), and
//! `RotaryEmbedding` emitted directly.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry, attrs};
use onyxia_ir::prim::{Prim, SliceSpec};
use onyxia_ir::{AttrValue, DataType, Error, Result, SymbolicShape, TensorType, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "Attention", attention);
    r.register("", "RotaryEmbedding", rotary_embedding);
}

/// `[B, S, H*D]` → `[B, H, S, D]`.
fn to_heads(ctx: &mut LowerCtx, x: ValueId, heads: u64) -> Result<ValueId> {
    let d = dims(ctx, x);
    let hd = d[2]
        .clone()
        .div_exact(&c(heads))
        .ok_or_else(|| Error::Shape(format!("hidden {} not divisible by {heads} heads", d[2])))?;
    let v = reshape(ctx, x, vec![d[0].clone(), d[1].clone(), c(heads), hd])?;
    transpose(ctx, v, &[0, 2, 1, 3])
}

/// `[B, H, S, D]` → `[B, S, H*D]`.
fn from_heads(ctx: &mut LowerCtx, x: ValueId) -> Result<ValueId> {
    let v = transpose(ctx, x, &[0, 2, 1, 3])?;
    let d = dims(ctx, v);
    reshape(
        ctx,
        v,
        vec![d[0].clone(), d[1].clone(), d[2].clone() * d[3].clone()],
    )
}

fn attention(ctx: &mut LowerCtx) -> Result<()> {
    let (mut q, mut k, mut v) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let three_d = rank(ctx, q) == 3;
    let q_heads_attr = ctx.attr_i("q_num_heads");
    let kv_heads_attr = ctx.attr_i("kv_num_heads");
    if three_d {
        let qh = q_heads_attr.ok_or_else(|| ctx.missing_attr("q_num_heads"))? as u64;
        let kvh = kv_heads_attr.ok_or_else(|| ctx.missing_attr("kv_num_heads"))? as u64;
        q = to_heads(ctx, q, qh)?;
        k = to_heads(ctx, k, kvh)?;
        v = to_heads(ctx, v, kvh)?;
    }
    let qd = dims(ctx, q);
    let kd = dims(ctx, k);
    let vd = dims(ctx, v);
    let (qh, kvh) = (
        qd[1]
            .as_const()
            .ok_or_else(|| Error::Unsupported("Attention with symbolic head count".into()))?,
        kd[1]
            .as_const()
            .ok_or_else(|| Error::Unsupported("Attention with symbolic head count".into()))?,
    );
    let head = qd[3]
        .as_const()
        .ok_or_else(|| Error::Unsupported("Attention with symbolic head size".into()))?;
    let scale = match ctx.attr_f("scale") {
        Some(s) => s as f64,
        None => 1.0 / (head as f64).sqrt(),
    };
    let dt = dtype(ctx, q);

    let mut inputs = vec![q, k, v];
    let mask = opt_val(ctx, 3)?;
    let mask_bool = mask
        .map(|m| dtype(ctx, m) == DataType::Bool)
        .unwrap_or(false);
    if let Some(m) = mask {
        let m = if mask_bool { m } else { cast(ctx, m, dt)? };
        inputs.push(m);
    }
    let past = match (opt_val(ctx, 4)?, opt_val(ctx, 5)?) {
        (Some(pk), Some(pv)) => {
            inputs.push(pk);
            inputs.push(pv);
            Some((pk, pv))
        }
        (None, None) => None,
        _ => {
            return Err(Error::Unsupported(
                "Attention with only one of past_key/past_value".into(),
            ));
        }
    };
    let nonpad = opt_val(ctx, 6)?;
    if let Some(np) = nonpad {
        let np = cast(ctx, np, DataType::I64)?;
        inputs.push(np);
    }
    let total = match past {
        Some((pk, _)) => dims(ctx, pk)[2].clone() + kd[2].clone(),
        None => kd[2].clone(),
    };
    let y_ty = TensorType::new(
        dt,
        SymbolicShape(vec![
            qd[0].clone(),
            qd[1].clone(),
            qd[2].clone(),
            vd[3].clone(),
        ]),
    );
    let pk_ty = TensorType::new(
        dt,
        SymbolicShape(vec![
            kd[0].clone(),
            kd[1].clone(),
            total.clone(),
            kd[3].clone(),
        ]),
    );
    let pv_ty = TensorType::new(
        dt,
        SymbolicShape(vec![
            vd[0].clone(),
            vd[1].clone(),
            total.clone(),
            vd[3].clone(),
        ]),
    );
    let qk_ty = TensorType::new(
        dt,
        SymbolicShape(vec![qd[0].clone(), qd[1].clone(), qd[2].clone(), total]),
    );
    let a = attrs(vec![
        (
            "is_causal",
            AttrValue::Int(ctx.attr_i("is_causal").unwrap_or(0)),
        ),
        ("q_num_heads", AttrValue::Int(qh as i64)),
        ("kv_num_heads", AttrValue::Int(kvh as i64)),
        ("scale", AttrValue::Float(scale)),
        (
            "softcap",
            AttrValue::Float(ctx.attr_f("softcap").unwrap_or(0.0) as f64),
        ),
        (
            "qk_matmul_output_mode",
            AttrValue::Int(ctx.attr_i("qk_matmul_output_mode").unwrap_or(0)),
        ),
        ("has_mask", AttrValue::Int(mask.is_some() as i64)),
        ("mask_is_bool", AttrValue::Int(mask_bool as i64)),
        ("has_past", AttrValue::Int(past.is_some() as i64)),
        ("has_nonpad", AttrValue::Int(nonpad.is_some() as i64)),
    ]);
    let outs = ctx
        .builder()
        .composite("Attention", a, &inputs, vec![y_ty, pk_ty, pv_ty, qk_ty])?;
    let y = if three_d {
        from_heads(ctx, outs[0])?
    } else {
        outs[0]
    };
    ctx.set_value(0, y);
    ctx.set_value_opt(1, outs[1]);
    ctx.set_value_opt(2, outs[2]);
    ctx.set_value_opt(3, outs[3]);
    Ok(())
}

/// Strided slice of the last axis: `x[..., start::2]` up to `end`.
fn every_other(ctx: &mut LowerCtx, x: ValueId, start: u64, end: u64) -> Result<ValueId> {
    let axis = rank(ctx, x) - 1;
    ctx.emit(
        Prim::Slice {
            specs: vec![SliceSpec {
                axis,
                start: c(start),
                end: c(end),
                step: 2,
            }],
        },
        &[x],
    )
}

/// ONNX RotaryEmbedding (opset 23), following the spec's reference
/// algorithm in `[B, S, H, D]` layout.
fn rotary_embedding(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let (cos_cache, sin_cache) = (val(ctx, 1)?, val(ctx, 2)?);
    let pos = opt_val(ctx, 3)?;
    let interleaved = ctx.attr_i("interleaved").unwrap_or(0) != 0;
    let rot_attr = ctx.attr_i("rotary_embedding_dim").unwrap_or(0) as u64;
    let num_heads = ctx.attr_i("num_heads").unwrap_or(0) as u64;
    let orig_rank = rank(ctx, x);
    // → [B, S, H, D]
    let xs = if orig_rank == 4 {
        transpose(ctx, x, &[0, 2, 1, 3])?
    } else {
        if num_heads == 0 {
            return Err(Error::Attribute(
                "RotaryEmbedding: num_heads required for 3D input".into(),
            ));
        }
        let d = dims(ctx, x);
        let hd = d[2].clone().div_exact(&c(num_heads)).ok_or_else(|| {
            Error::Shape("RotaryEmbedding: hidden not divisible by num_heads".into())
        })?;
        reshape(ctx, x, vec![d[0].clone(), d[1].clone(), c(num_heads), hd])?
    };
    let d = dims(ctx, xs);
    let head = d[3]
        .as_const()
        .ok_or_else(|| Error::Unsupported("RotaryEmbedding with symbolic head size".into()))?;
    let rot = if rot_attr == 0 { head } else { rot_attr };
    let half = rot / 2;
    let x_rot = slice_axis(ctx, xs, 3, 0, rot)?;
    let x_pass = slice_axis(ctx, xs, 3, rot, head)?;
    // cos/sin → [B, S, 1, half]
    let (mut cos, mut sin) = (cos_cache, sin_cache);
    if let Some(p) = pos {
        let p = cast(ctx, p, DataType::I64)?;
        cos = ctx.emit(Prim::Gather { axis: 0 }, &[cos, p])?;
        sin = ctx.emit(Prim::Gather { axis: 0 }, &[sin, p])?;
    }
    let cd = dims(ctx, cos);
    let cs = vec![cd[0].clone(), cd[1].clone(), c(1), cd[2].clone()];
    let cos = reshape(ctx, cos, cs.clone())?;
    let sin = reshape(ctx, sin, cs)?;
    let (x1, x2) = if interleaved {
        (
            every_other(ctx, x_rot, 0, rot)?,
            every_other(ctx, x_rot, 1, rot)?,
        )
    } else {
        (
            slice_axis(ctx, x_rot, 3, 0, half)?,
            slice_axis(ctx, x_rot, 3, half, rot)?,
        )
    };
    let c1 = mul(ctx, cos, x1)?;
    let s2 = mul(ctx, sin, x2)?;
    let real = sub(ctx, c1, s2)?;
    let s1 = mul(ctx, sin, x1)?;
    let c2 = mul(ctx, cos, x2)?;
    let imag = add(ctx, s1, c2)?;
    let rotated = if interleaved {
        // Interleave: stack on a new trailing axis, then flatten.
        let r5 = unsqueeze(ctx, real, 4)?;
        let i5 = unsqueeze(ctx, imag, 4)?;
        let st = concat(ctx, &[r5, i5], 4)?;
        let mut sd = dims(ctx, x_rot);
        sd[3] = c(rot);
        reshape(ctx, st, sd)?
    } else {
        concat(ctx, &[real, imag], 3)?
    };
    let y = if rot < head {
        concat(ctx, &[rotated, x_pass], 3)?
    } else {
        rotated
    };
    let y = if orig_rank == 4 {
        transpose(ctx, y, &[0, 2, 1, 3])?
    } else {
        let od = dims(ctx, x);
        reshape(ctx, y, od)?
    };
    out(ctx, y)
}
