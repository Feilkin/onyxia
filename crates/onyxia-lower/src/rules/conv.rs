//! Convolution and pooling, by im2col: a constant index table gathers
//! every (output position, kernel offset) pair into a `[N, C, O, K]`
//! column tensor, after which Conv is one batched MatMul and pooling is
//! one reduction over `K`. No new primitive is needed; a backend that
//! wants a direct kernel can add one later without changing the IR.
//!
//! Spatial dims must be static (the table is built at lowering); batch
//! and channel dims may be symbolic.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry};
use onyxia_ir::prim::{BinaryOp, Prim, ReduceOp, ScatterReduce, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "Conv", conv);
    r.register("", "ConvTranspose", conv_transpose);
    r.register("", "MaxPool", |c| pool(c, PoolKind::Max));
    r.register("", "AveragePool", |c| pool(c, PoolKind::Avg));
    r.register("", "LpPool", |c| pool(c, PoolKind::Lp));
    r.register("", "GlobalMaxPool", |c| global_pool(c, PoolKind::Max));
    r.register("", "GlobalAveragePool", |c| global_pool(c, PoolKind::Avg));
    r.register("", "GlobalLpPool", |c| global_pool(c, PoolKind::Lp));
    r.register("", "MaxUnpool", max_unpool);
    r.register("", "ConvInteger", conv_integer);
}

#[derive(Clone, Copy, PartialEq)]
enum PoolKind {
    Max,
    Avg,
    Lp,
}

/// Window geometry over the spatial axes.
struct Geo {
    input: Vec<i64>,
    kernel: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
    /// `[begin..., end...]`.
    pads: Vec<i64>,
    out: Vec<i64>,
}

impl Geo {
    fn nd(&self) -> usize {
        self.input.len()
    }
    fn out_count(&self) -> i64 {
        self.out.iter().product()
    }
    fn kernel_count(&self) -> i64 {
        self.kernel.iter().product()
    }
    fn effective_kernel(&self, i: usize) -> i64 {
        (self.kernel[i] - 1) * self.dilations[i] + 1
    }
}

/// Resolve strides/dilations/pads/auto_pad/ceil_mode into a [`Geo`].
fn geometry(ctx: &LowerCtx, input: Vec<i64>, kernel: Vec<i64>, ceil_mode: bool) -> Result<Geo> {
    let nd = input.len();
    if kernel.len() != nd {
        return Err(Error::Shape(format!(
            "node '{}': kernel rank {} vs spatial rank {nd}",
            ctx.node_name(),
            kernel.len()
        )));
    }
    let strides = ctx.attr_is("strides").unwrap_or(vec![1; nd]);
    let dilations = ctx.attr_is("dilations").unwrap_or(vec![1; nd]);
    let auto_pad = ctx.attr_s("auto_pad").unwrap_or("NOTSET").to_string();
    let mut pads = ctx.attr_is("pads").unwrap_or(vec![0; 2 * nd]);
    if pads.len() != 2 * nd {
        return Err(Error::Shape(format!("pads must have {} entries", 2 * nd)));
    }
    let mut geo = Geo {
        input,
        kernel,
        strides,
        dilations,
        pads: vec![0; 2 * nd],
        out: vec![0; nd],
    };
    match auto_pad.as_str() {
        "SAME_UPPER" | "SAME_LOWER" => {
            for i in 0..nd {
                let o = (geo.input[i] + geo.strides[i] - 1) / geo.strides[i];
                let total =
                    ((o - 1) * geo.strides[i] + geo.effective_kernel(i) - geo.input[i]).max(0);
                let half = total / 2;
                let (b, e) = if auto_pad == "SAME_UPPER" {
                    (half, total - half)
                } else {
                    (total - half, half)
                };
                pads[i] = b;
                pads[i + nd] = e;
                geo.out[i] = o;
            }
        }
        "VALID" => {
            pads = vec![0; 2 * nd];
        }
        _ => {}
    }
    geo.pads = pads;
    if auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER" {
        for i in 0..nd {
            let span = geo.input[i] + geo.pads[i] + geo.pads[i + nd] - geo.effective_kernel(i);
            let s = geo.strides[i];
            let mut o = if ceil_mode {
                (span + s - 1).div_euclid(s) + 1
            } else {
                span.div_euclid(s) + 1
            };
            // Ceil mode: the last window must start inside the input or
            // the leading padding, never in the trailing padding.
            if ceil_mode && (o - 1) * s >= geo.input[i] + geo.pads[i] {
                o -= 1;
            }
            geo.out[i] = o.max(0);
        }
    }
    Ok(geo)
}

/// Spatial dims of `x` (axes 2..) as static i64s.
fn spatial(ctx: &LowerCtx, x: ValueId, what: &str) -> Result<Vec<i64>> {
    let d = dims(ctx, x);
    d[2..]
        .iter()
        .map(|e| {
            e.as_const().map(|v| v as i64).ok_or_else(|| {
                Error::Unsupported(format!(
                    "node '{}': {what} requires static spatial dims, got {}",
                    ctx.node_name(),
                    ctx.ty(x).shape
                ))
            })
        })
        .collect()
}

/// Pad the spatial axes of `x` per `geo` (plus whatever extra trailing
/// padding the last windows need) with `value`, then flatten them to one
/// axis. Returns the padded-flat tensor `[N, C, P]` and the padded extents.
fn pad_and_flatten(
    ctx: &mut LowerCtx,
    x: ValueId,
    geo: &Geo,
    value: ValueId,
) -> Result<(ValueId, Vec<i64>)> {
    let nd = geo.nd();
    let mut y = x;
    let mut extents = Vec::with_capacity(nd);
    for i in 0..nd {
        let needed = (geo.out[i] - 1).max(0) * geo.strides[i] + geo.effective_kernel(i);
        let after = (needed - geo.input[i] - geo.pads[i]).max(geo.pads[i + nd]);
        y = pad_axis_const(ctx, y, 2 + i, geo.pads[i], after, value)?;
        extents.push(geo.input[i] + geo.pads[i] + after);
    }
    let d = dims(ctx, y);
    let flat = reshape(ctx, y, vec![d[0].clone(), d[1].clone(), prod(&d[2..])])?;
    Ok((flat, extents))
}

/// Constant `[O, K]` table of positions in the padded-flat spatial axis.
fn window_table(geo: &Geo, extents: &[i64]) -> Vec<i64> {
    let nd = geo.nd();
    let (o_n, k_n) = (geo.out_count(), geo.kernel_count());
    let mut table = Vec::with_capacity((o_n * k_n) as usize);
    let mut oc = vec![0i64; nd];
    for o in 0..o_n {
        let mut rem = o;
        for i in (0..nd).rev() {
            oc[i] = rem % geo.out[i];
            rem /= geo.out[i];
        }
        let mut kc = vec![0i64; nd];
        for k in 0..k_n {
            let mut rem = k;
            for i in (0..nd).rev() {
                kc[i] = rem % geo.kernel[i];
                rem /= geo.kernel[i];
            }
            let mut lin = 0i64;
            for i in 0..nd {
                let p = oc[i] * geo.strides[i] + kc[i] * geo.dilations[i];
                lin = lin * extents[i] + p;
            }
            table.push(lin);
        }
    }
    table
}

/// im2col: `[N, C, spatial...]` → `[N, C, O, K]`.
fn im2col(ctx: &mut LowerCtx, x: ValueId, geo: &Geo, pad_value: ValueId) -> Result<ValueId> {
    let (flat, extents) = pad_and_flatten(ctx, x, geo, pad_value)?;
    let table = window_table(geo, &extents);
    let idx = const_i64(
        ctx,
        &table,
        &[geo.out_count() as u64, geo.kernel_count() as u64],
    )?;
    ctx.emit(Prim::Gather { axis: 2 }, &[flat, idx])
}

/// Output dims `[N, C', out...]`.
fn out_dims(n: DimExpr, ch: DimExpr, geo: &Geo) -> Vec<DimExpr> {
    let mut d = vec![n, ch];
    d.extend(geo.out.iter().map(|&v| c(v as u64)));
    d
}

// ───────────────────────────────── Conv ────────────────────────────────

/// Convolution core shared by Conv / ConvInteger / ConvTranspose:
/// `x [N, C, sp...]`, `w [M, C/g, k...]`, optional bias `[M]`.
fn conv_core(
    ctx: &mut LowerCtx,
    x: ValueId,
    w: ValueId,
    bias: Option<ValueId>,
    geo: &Geo,
    group: u64,
) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let xd = dims(ctx, x);
    let wd = dims(ctx, w);
    let (n, ch) = (xd[0].clone(), xd[1].clone());
    let m = wd[0].clone();
    let cg = ch
        .div_exact(&c(group))
        .ok_or_else(|| Error::Shape("Conv: C not divisible by group".into()))?;
    let mg = m
        .div_exact(&c(group))
        .ok_or_else(|| Error::Shape("Conv: M not divisible by group".into()))?;
    let zero = scalar(ctx, dt, 0.0)?;
    let cols = im2col(ctx, x, geo, zero)?; // [N, C, O, K]
    let o = c(geo.out_count() as u64);
    let k = c(geo.kernel_count() as u64);
    let g = c(group);
    let cols = reshape(
        ctx,
        cols,
        vec![n.clone(), g.clone(), cg.clone(), o.clone(), k.clone()],
    )?;
    let cols = transpose(ctx, cols, &[0, 1, 3, 2, 4])?; // [N, g, O, Cg, K]
    let cols = reshape(
        ctx,
        cols,
        vec![n.clone(), g.clone(), o.clone(), cg.clone() * k.clone()],
    )?;
    let wm = reshape(ctx, w, vec![g.clone(), mg.clone(), cg * k])?;
    let wm = transpose(ctx, wm, &[0, 2, 1])?; // [g, Cg*K, Mg]
    let wm = unsqueeze(ctx, wm, 0)?; // [1, g, Cg*K, Mg]
    let y = matmul(ctx, cols, wm)?; // [N, g, O, Mg]
    let y = transpose(ctx, y, &[0, 1, 3, 2])?; // [N, g, Mg, O]
    let mut y = reshape(ctx, y, out_dims(n, m, geo))?;
    if let Some(b) = bias {
        let r = rank(ctx, y);
        let bn = dims(ctx, b)[0].clone();
        let mut bs = vec![c(1); r];
        bs[1] = bn;
        let b = reshape(ctx, b, bs)?;
        y = add(ctx, y, b)?;
    }
    Ok(y)
}

fn conv(ctx: &mut LowerCtx) -> Result<()> {
    let (x, w) = (val(ctx, 0)?, val(ctx, 1)?);
    let bias = opt_val(ctx, 2)?;
    let input = spatial(ctx, x, "Conv")?;
    let kernel = match ctx.attr_is("kernel_shape") {
        Some(k) => k,
        None => spatial(ctx, w, "Conv")?,
    };
    let geo = geometry(ctx, input, kernel, false)?;
    let group = ctx.attr_i("group").unwrap_or(1) as u64;
    let y = conv_core(ctx, x, w, bias, &geo, group)?;
    out(ctx, y)
}

/// ConvInteger: `(x - x_zp) ⊛ (w - w_zp)` accumulated in int32.
fn conv_integer(ctx: &mut LowerCtx) -> Result<()> {
    let (x, w) = (val(ctx, 0)?, val(ctx, 1)?);
    let x = cast(ctx, x, DataType::I32)?;
    let w = cast(ctx, w, DataType::I32)?;
    let x = match opt_val(ctx, 2)? {
        Some(zp) => {
            let zp = cast(ctx, zp, DataType::I32)?;
            sub(ctx, x, zp)?
        }
        None => x,
    };
    let w = match opt_val(ctx, 3)? {
        Some(zp) => {
            let zp = cast(ctx, zp, DataType::I32)?;
            let r = rank(ctx, w);
            // Per-output-channel zero point broadcasts along M.
            let zp = if rank(ctx, zp) == 1 {
                let mut s = vec![c(1); r];
                s[0] = dims(ctx, zp)[0].clone();
                reshape(ctx, zp, s)?
            } else {
                zp
            };
            sub(ctx, w, zp)?
        }
        None => w,
    };
    let input = spatial(ctx, x, "ConvInteger")?;
    let kernel = match ctx.attr_is("kernel_shape") {
        Some(k) => k,
        None => spatial(ctx, w, "ConvInteger")?,
    };
    let geo = geometry(ctx, input, kernel, false)?;
    let group = ctx.attr_i("group").unwrap_or(1) as u64;
    let y = conv_core(ctx, x, w, None, &geo, group)?;
    out(ctx, y)
}

/// Insert `s-1` zeros between consecutive elements along `axis`.
fn dilate_axis(ctx: &mut LowerCtx, x: ValueId, axis: usize, s: i64) -> Result<ValueId> {
    if s == 1 {
        return Ok(x);
    }
    let dt = dtype(ctx, x);
    let d = dims(ctx, x);
    let l = d[axis]
        .as_const()
        .ok_or_else(|| Error::Unsupported("ConvTranspose on a symbolic spatial dim".into()))?;
    if l == 0 {
        return Ok(x);
    }
    let v = unsqueeze(ctx, x, axis + 1)?; // [.., L, 1, ..]
    let zero = scalar(ctx, dt, 0.0)?;
    let mut zd = dims(ctx, v);
    zd[axis + 1] = c(s as u64 - 1);
    let z = broadcast(ctx, zero, zd)?;
    let v = concat(ctx, &[v, z], axis + 1)?; // [.., L, s, ..]
    let mut md = d.clone();
    md[axis] = c(l * s as u64);
    let v = reshape(ctx, v, md)?;
    slice_axis(ctx, v, axis, 0, (l - 1) * s as u64 + 1)
}

/// ConvTranspose as a direct convolution over the stride-dilated,
/// re-padded input with the flipped, channel-transposed kernel.
fn conv_transpose(ctx: &mut LowerCtx) -> Result<()> {
    let (x, w) = (val(ctx, 0)?, val(ctx, 1)?);
    let bias = opt_val(ctx, 2)?;
    let input = spatial(ctx, x, "ConvTranspose")?;
    let nd = input.len();
    let kernel = match ctx.attr_is("kernel_shape") {
        Some(k) => k,
        None => spatial(ctx, w, "ConvTranspose")?,
    };
    let strides = ctx.attr_is("strides").unwrap_or(vec![1; nd]);
    let dilations = ctx.attr_is("dilations").unwrap_or(vec![1; nd]);
    let output_padding = ctx.attr_is("output_padding").unwrap_or(vec![0; nd]);
    let group = ctx.attr_i("group").unwrap_or(1) as u64;
    let auto_pad = ctx.attr_s("auto_pad").unwrap_or("NOTSET").to_string();
    let eff: Vec<i64> = (0..nd)
        .map(|i| (kernel[i] - 1) * dilations[i] + 1)
        .collect();
    let mut pads = ctx.attr_is("pads").unwrap_or(vec![0; 2 * nd]);
    let output_shape = ctx.attr_is("output_shape");
    let set_total = |i: usize, total: i64, pads: &mut Vec<i64>| {
        let half = total / 2;
        if auto_pad == "SAME_UPPER" {
            pads[i] = half;
            pads[i + nd] = total - half;
        } else {
            pads[i] = total - half;
            pads[i + nd] = half;
        }
    };
    if let Some(os) = &output_shape {
        let os: Vec<i64> = if os.len() == nd + 2 {
            os[2..].to_vec()
        } else {
            os.clone()
        };
        for i in 0..nd {
            let total = strides[i] * (input[i] - 1) + output_padding[i] + eff[i] - os[i];
            set_total(i, total, &mut pads);
        }
    } else if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
        for i in 0..nd {
            let os = input[i] * strides[i];
            let total = strides[i] * (input[i] - 1) + output_padding[i] + eff[i] - os;
            set_total(i, total, &mut pads);
        }
    } else if auto_pad == "VALID" {
        pads = vec![0; 2 * nd];
    }
    // Stride-dilate the input, then pad for the equivalent direct conv.
    let mut xd = x;
    for i in 0..nd {
        xd = dilate_axis(ctx, xd, 2 + i, strides[i])?;
    }
    let dt = dtype(ctx, x);
    let zero = scalar(ctx, dt, 0.0)?;
    for i in 0..nd {
        let before = eff[i] - 1 - pads[i];
        let after = eff[i] - 1 - pads[i + nd] + output_padding[i];
        xd = pad_axis_const(ctx, xd, 2 + i, before, after, zero)?;
    }
    // Kernel: [C, M/g, k...] → [g, Cg, Mg, k] → [g, Mg, Cg, k] → [M, Cg, k],
    // flipped along every spatial axis.
    let wd = dims(ctx, w);
    let ch = wd[0].clone();
    let mg = wd[1].clone();
    let cg = ch
        .div_exact(&c(group))
        .ok_or_else(|| Error::Shape("ConvTranspose: C not divisible by group".into()))?;
    let mut w5 = vec![c(group), cg.clone(), mg.clone()];
    w5.extend_from_slice(&wd[2..]);
    let wt = reshape(ctx, w, w5)?;
    let mut perm: Vec<usize> = vec![0, 2, 1];
    perm.extend(3..3 + nd);
    let wt = transpose(ctx, wt, &perm)?;
    let mut w4 = vec![c(group) * mg, cg];
    w4.extend_from_slice(&wd[2..]);
    let mut wt = reshape(ctx, wt, w4)?;
    for i in 0..nd {
        wt = flip(ctx, wt, 2 + i)?;
    }
    let dil_input = spatial(ctx, xd, "ConvTranspose")?;
    let geo = Geo {
        input: dil_input.clone(),
        kernel: kernel.clone(),
        strides: vec![1; nd],
        dilations: dilations.clone(),
        pads: vec![0; 2 * nd],
        out: (0..nd).map(|i| dil_input[i] - eff[i] + 1).collect(),
    };
    let y = conv_core(ctx, xd, wt, bias, &geo, group)?;
    out(ctx, y)
}

// ──────────────────────────────── pooling ──────────────────────────────

fn pool(ctx: &mut LowerCtx, kind: PoolKind) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let input = spatial(ctx, x, "pooling")?;
    let kernel = ctx
        .attr_is("kernel_shape")
        .ok_or_else(|| ctx.missing_attr("kernel_shape"))?;
    let ceil_mode = ctx.attr_i("ceil_mode").unwrap_or(0) != 0;
    let geo = geometry(ctx, input, kernel, ceil_mode)?;
    let xd = dims(ctx, x);
    let (n, ch) = (xd[0].clone(), xd[1].clone());
    let od = out_dims(n.clone(), ch.clone(), &geo);
    let y = match kind {
        PoolKind::Max => {
            if ctx.attr_i("storage_order").unwrap_or(0) != 0 {
                return Err(Error::Unsupported("MaxPool storage_order=1".into()));
            }
            let pad_v = scalar(ctx, dt, lowest(dt))?;
            let cols = im2col(ctx, x, &geo, pad_v)?; // [N, C, O, K]
            let m = reduce(ctx, ReduceOp::Max, cols, &[3], false)?;
            if ctx.has_output(1) {
                // Indices: argmax over K, then look up the unpadded flat
                // position of that tap and offset by the (n, c) plane.
                let am = arg_reduce(ctx, cols, 3, false, true, false)?; // [N, C, O] i64
                let (_, extents) = pad_and_flatten(ctx, x, &geo, pad_v)?;
                let table = window_table(&geo, &extents);
                let k = geo.kernel_count();
                let nd = geo.nd();
                let orig: Vec<i64> = table
                    .iter()
                    .map(|&lin| {
                        // Padded-flat → unpadded coordinates.
                        let mut rem = lin;
                        let mut coords = vec![0i64; nd];
                        for i in (0..nd).rev() {
                            coords[i] = rem % extents[i] - geo.pads[i];
                            rem /= extents[i];
                        }
                        let mut o = 0i64;
                        for i in 0..nd {
                            o = o * geo.input[i] + coords[i].clamp(0, geo.input[i] - 1);
                        }
                        o
                    })
                    .collect();
                let tab = const_i64(ctx, &orig, &[orig.len() as u64])?;
                let oi = iota_along(ctx, c(geo.out_count() as u64), 3, 2, DataType::I64)?;
                let kk = scalar(ctx, DataType::I64, k as f64)?;
                let base = mul(ctx, oi, kk)?;
                let lin = add(ctx, base, am)?;
                let pos = ctx.emit(Prim::Gather { axis: 0 }, &[tab, lin])?; // [N, C, O]
                let plane: i64 = geo.input.iter().product();
                let ni = iota_along(ctx, n.clone(), 3, 0, DataType::I64)?;
                let ci = iota_along(ctx, ch.clone(), 3, 1, DataType::I64)?;
                let cn = dim_value(ctx, ch.clone())?;
                let nc = mul(ctx, ni, cn)?;
                let nc = add(ctx, nc, ci)?;
                let pl = scalar(ctx, DataType::I64, plane as f64)?;
                let off = mul(ctx, nc, pl)?;
                let idx = add(ctx, pos, off)?;
                let idx = reshape(ctx, idx, od.clone())?;
                ctx.set_value(1, idx);
            }
            reshape(ctx, m, od)?
        }
        PoolKind::Avg => {
            let zero = scalar(ctx, dt, 0.0)?;
            let cols = im2col(ctx, x, &geo, zero)?;
            let s = reduce(ctx, ReduceOp::Sum, cols, &[3], false)?;
            let include_pad = ctx.attr_i("count_include_pad").unwrap_or(0) != 0;
            // Divisor: taps inside the input (or inside the explicitly
            // padded extent), counted by pooling a ones tensor.
            let one = scalar(ctx, dt, 1.0)?;
            let mut od1 = vec![c(1), c(1)];
            let ones_geo;
            let ones = if include_pad {
                od1.extend(
                    geo.input
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| c((v + geo.pads[i] + geo.pads[i + geo.nd()]) as u64)),
                );
                ones_geo = Geo {
                    input: (0..geo.nd())
                        .map(|i| geo.input[i] + geo.pads[i] + geo.pads[i + geo.nd()])
                        .collect(),
                    pads: vec![0; 2 * geo.nd()],
                    kernel: geo.kernel.clone(),
                    strides: geo.strides.clone(),
                    dilations: geo.dilations.clone(),
                    out: geo.out.clone(),
                };
                broadcast(ctx, one, od1)?
            } else {
                od1.extend(geo.input.iter().map(|&v| c(v as u64)));
                ones_geo = Geo {
                    input: geo.input.clone(),
                    pads: geo.pads.clone(),
                    kernel: geo.kernel.clone(),
                    strides: geo.strides.clone(),
                    dilations: geo.dilations.clone(),
                    out: geo.out.clone(),
                };
                broadcast(ctx, one, od1)?
            };
            let cnt = im2col(ctx, ones, &ones_geo, zero)?;
            let cnt = reduce(ctx, ReduceOp::Sum, cnt, &[3], false)?; // [1, 1, O]
            let y = div(ctx, s, cnt)?;
            reshape(ctx, y, od)?
        }
        PoolKind::Lp => {
            let p = ctx.attr_i("p").unwrap_or(2) as f64;
            let zero = scalar(ctx, dt, 0.0)?;
            let a = unary(ctx, UnaryOp::Abs, x)?;
            let pc = scalar(ctx, dt, p)?;
            let ap = ctx.emit(Prim::Binary(BinaryOp::Pow), &[a, pc])?;
            let cols = im2col(ctx, ap, &geo, zero)?;
            let s = reduce(ctx, ReduceOp::Sum, cols, &[3], false)?;
            let ip = scalar(ctx, dt, 1.0 / p)?;
            let y = ctx.emit(Prim::Binary(BinaryOp::Pow), &[s, ip])?;
            reshape(ctx, y, od)?
        }
    };
    out(ctx, y)
}

fn global_pool(ctx: &mut LowerCtx, kind: PoolKind) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let axes: Vec<usize> = (2..r).collect();
    let y = match kind {
        PoolKind::Max => reduce(ctx, ReduceOp::Max, x, &axes, true)?,
        PoolKind::Avg => reduce(ctx, ReduceOp::Mean, x, &axes, true)?,
        PoolKind::Lp => {
            let p = ctx.attr_i("p").unwrap_or(2) as f64;
            let a = unary(ctx, UnaryOp::Abs, x)?;
            let pc = scalar(ctx, dt, p)?;
            let ap = ctx.emit(Prim::Binary(BinaryOp::Pow), &[a, pc])?;
            let s = reduce(ctx, ReduceOp::Sum, ap, &axes, true)?;
            let ip = scalar(ctx, dt, 1.0 / p)?;
            ctx.emit(Prim::Binary(BinaryOp::Pow), &[s, ip])?
        }
    };
    out(ctx, y)
}

/// MaxUnpool: scatter `X` into zeros at the flat indices `I`.
fn max_unpool(ctx: &mut LowerCtx) -> Result<()> {
    let (x, idx) = (val(ctx, 0)?, val(ctx, 1)?);
    let dt = dtype(ctx, x);
    let xd = static_dims(ctx, x, "MaxUnpool")?;
    let nd = xd.len() - 2;
    let out_shape: Vec<u64> = if ctx.has_input(2) {
        require_const_ints(ctx, 2, "MaxUnpool output_shape")?
            .into_iter()
            .map(|v| v as u64)
            .collect()
    } else {
        let kernel = ctx
            .attr_is("kernel_shape")
            .ok_or_else(|| ctx.missing_attr("kernel_shape"))?;
        let strides = ctx.attr_is("strides").unwrap_or(vec![1; nd]);
        let pads = ctx.attr_is("pads").unwrap_or(vec![0; 2 * nd]);
        let mut s = vec![xd[0], xd[1]];
        for i in 0..nd {
            s.push(
                ((xd[2 + i] as i64 - 1) * strides[i] + kernel[i] - pads[i] - pads[i + nd]) as u64,
            );
        }
        s
    };
    let total: u64 = out_shape.iter().product();
    let n = prod(&dims(ctx, x));
    let flat_x = reshape(ctx, x, vec![n.clone()])?;
    let idx = cast(ctx, idx, DataType::I64)?;
    let flat_i = reshape(ctx, idx, vec![n, c(1)])?;
    let zero = scalar(ctx, dt, 0.0)?;
    let base = broadcast(ctx, zero, vec![c(total)])?;
    let y = ctx.emit(
        Prim::Scatter {
            reduction: ScatterReduce::None,
        },
        &[base, flat_i, flat_x],
    )?;
    let y = reshape(ctx, y, out_shape.iter().map(|&v| c(v)).collect())?;
    out(ctx, y)
}
