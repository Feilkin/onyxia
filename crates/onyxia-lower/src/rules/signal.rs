//! Resampling and signal-processing ops. The common trick: anything that
//! is a fixed linear map along one axis (Resize interpolation, the DFT)
//! becomes a constant matrix applied with `MatMul`; windows and mel
//! filterbanks are constants computed at lowering.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry, convert_proto_dtype};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "Resize", resize);
    r.register("", "Upsample", upsample);
    r.register("", "HannWindow", |c| window(c, Window::Hann));
    r.register("", "HammingWindow", |c| window(c, Window::Hamming));
    r.register("", "BlackmanWindow", |c| window(c, Window::Blackman));
    r.register("", "MelWeightMatrix", mel_weight_matrix);
    r.register("", "DFT", dft);
    r.register("", "STFT", stft);
    r.register("", "AffineGrid", affine_grid);
    r.register("", "GridSample", grid_sample);
    r.register("", "Einsum", einsum);
}

// ──────────────────────────────── Resize ───────────────────────────────

struct ResizeCfg {
    mode: String,
    ctm: String,
    nearest_mode: String,
    cubic_a: f64,
    exclude_outside: bool,
    extrapolation: f64,
    antialias: bool,
}

/// Interpolation coefficients for a fractional position `ratio ∈ (0, 1]`
/// (the reference's `get_coeffs`).
fn coeffs(cfg: &ResizeCfg, ratio: f64, scale: f64) -> Vec<f64> {
    match cfg.mode.as_str() {
        "nearest" => {
            if ratio == 1.0 {
                return vec![0.0, 1.0];
            }
            match cfg.nearest_mode.as_str() {
                "round_prefer_ceil" => {
                    vec![(ratio < 0.5) as u8 as f64, (ratio >= 0.5) as u8 as f64]
                }
                "floor" => vec![1.0, 0.0],
                "ceil" => vec![0.0, 1.0],
                _ => vec![(ratio <= 0.5) as u8 as f64, (ratio > 0.5) as u8 as f64],
            }
        }
        "linear" => {
            if cfg.antialias {
                let s = scale.min(1.0);
                let start = (-1.0 / s).floor() as i64 + 1;
                let footprint = 2 - 2 * start;
                let mut v: Vec<f64> = (0..footprint)
                    .map(|i| {
                        let arg = ((start + i) as f64 - ratio) * s;
                        (1.0 - arg.abs()).clamp(0.0, 1.0)
                    })
                    .collect();
                let sum: f64 = v.iter().sum();
                for x in &mut v {
                    *x /= sum;
                }
                v
            } else {
                vec![1.0 - ratio, ratio]
            }
        }
        _ => {
            // cubic
            let a = cfg.cubic_a;
            if cfg.antialias {
                let s = scale.min(1.0);
                let i_start = (-2.0 / s).floor() as i64 + 1;
                let i_end = 2 - i_start;
                let f = |x: f64| -> f64 {
                    let x = x.abs();
                    let (x2, x3) = (x * x, x * x * x);
                    if x <= 1.0 {
                        (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
                    } else if x < 2.0 {
                        a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
                    } else {
                        0.0
                    }
                };
                let mut v: Vec<f64> = (i_start..i_end)
                    .map(|i| f(s * (i as f64 - ratio)))
                    .collect();
                let sum: f64 = v.iter().sum();
                for x in &mut v {
                    *x /= sum;
                }
                v
            } else {
                let r = ratio;
                vec![
                    ((a * (r + 1.0) - 5.0 * a) * (r + 1.0) + 8.0 * a) * (r + 1.0) - 4.0 * a,
                    ((a + 2.0) * r - (a + 3.0)) * r * r + 1.0,
                    ((a + 2.0) * (1.0 - r) - (a + 3.0)) * (1.0 - r) * (1.0 - r) + 1.0,
                    ((a * ((1.0 - r) + 1.0) - 5.0 * a) * ((1.0 - r) + 1.0) + 8.0 * a)
                        * ((1.0 - r) + 1.0)
                        - 4.0 * a,
                ]
            }
        }
    }
}

/// Dense `[out, in]` interpolation matrix for one axis plus a per-row
/// additive term (extrapolation value where `tf_crop_and_resize` falls
/// outside the input).
fn resize_matrix(
    cfg: &ResizeCfg,
    input_w: usize,
    out_w: usize,
    scale: f64,
    roi: Option<(f64, f64)>,
) -> (Vec<f64>, Vec<f64>) {
    let mut m = vec![0.0; out_w * input_w];
    let mut extra = vec![0.0; out_w];
    let output_w = scale * input_w as f64;
    for y in 0..out_w {
        let yf = y as f64;
        let (x_ori, extrapolated) = match cfg.ctm.as_str() {
            "align_corners" => (
                if output_w == 1.0 {
                    0.0
                } else {
                    yf * (input_w as f64 - 1.0) / (output_w - 1.0)
                },
                false,
            ),
            "asymmetric" => (yf / scale, false),
            "tf_crop_and_resize" => {
                let (s, e) = roi.unwrap_or((0.0, 1.0));
                let x = if output_w == 1.0 {
                    (e - s) * (input_w as f64 - 1.0) / 2.0
                } else {
                    yf * (e - s) * (input_w as f64 - 1.0) / (output_w - 1.0)
                } + s * (input_w as f64 - 1.0);
                (x, x < 0.0 || x > input_w as f64 - 1.0)
            }
            "pytorch_half_pixel" => (
                if output_w == 1.0 {
                    -0.5
                } else {
                    (yf + 0.5) / scale - 0.5
                },
                false,
            ),
            "half_pixel_symmetric" => {
                let adjustment = out_w as f64 / output_w;
                let center = input_w as f64 / 2.0;
                let offset = center * (1.0 - adjustment);
                (offset + (yf + 0.5) / scale - 0.5, false)
            }
            _ => ((yf + 0.5) / scale - 0.5, false),
        };
        if extrapolated {
            extra[y] = cfg.extrapolation;
            continue;
        }
        let x_int = x_ori.floor();
        let ratio = if x_ori == x_int { 1.0 } else { x_ori - x_int };
        let mut cf = coeffs(cfg, ratio, scale);
        let n = cf.len() as i64;
        let pad = ((n as f64) / 2.0).ceil() as i64;
        let x_padded = x_ori + pad as f64;
        let p = x_padded.floor();
        let frac = x_padded - p;
        let offset = if n % 2 == 0 {
            if frac == 0.0 { -(n / 2) } else { -(n / 2) + 1 }
        } else {
            let base = -((n - 1) / 2);
            if frac <= 0.5 { base } else { base + 1 }
        };
        let start = p as i64 + offset;
        let idx: Vec<i64> = (0..n).map(|k| start + k - pad).collect();
        if cfg.exclude_outside {
            for (k, &i) in idx.iter().enumerate() {
                if i < 0 || i >= input_w as i64 {
                    cf[k] = 0.0;
                }
            }
            let s: f64 = cf.iter().sum();
            let s = if s == 0.0 { 1.0 } else { s };
            for v in &mut cf {
                *v /= s;
            }
        }
        for (k, &i) in idx.iter().enumerate() {
            let ci = i.clamp(0, input_w as i64 - 1) as usize;
            m[y * input_w + ci] += cf[k];
        }
    }
    (m, extra)
}

/// Apply `[out, in]` matrices along the chosen axes of `x`.
fn apply_axis_matrices(
    ctx: &mut LowerCtx,
    x: ValueId,
    plan: &[(usize, usize, Vec<f64>, Vec<f64>)], // (axis, out_w, matrix, extra)
) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let compute_dt = if dt.is_float() { dt } else { DataType::F32 };
    let mut y = cast(ctx, x, compute_dt)?;
    for (axis, out_w, m, extra) in plan {
        let (moved, inv) = axis_to_last(ctx, y, *axis)?;
        let in_w = dims(ctx, moved).last().unwrap().as_const().unwrap() as usize;
        // matrix as [in, out] so the matmul contracts the last axis.
        let mut mt = vec![0.0; in_w * out_w];
        for o in 0..*out_w {
            for i in 0..in_w {
                mt[i * out_w + o] = m[o * in_w + i];
            }
        }
        let w = const_typed(ctx, compute_dt, &mt, &[in_w as u64, *out_w as u64])?;
        let r = rank(ctx, moved);
        let lhs = if r == 1 {
            unsqueeze(ctx, moved, 0)?
        } else {
            moved
        };
        let mut v = matmul(ctx, lhs, w)?;
        if r == 1 {
            v = squeeze(ctx, v, 0)?;
        }
        if extra.iter().any(|&e| e != 0.0) {
            let e = const_typed(ctx, compute_dt, extra, &[*out_w as u64])?;
            v = add(ctx, v, e)?;
        }
        y = transpose(ctx, v, &inv)?;
    }
    cast(ctx, y, dt)
}

#[allow(clippy::too_many_arguments)]
fn resize_core(
    ctx: &mut LowerCtx,
    x: ValueId,
    cfg: &ResizeCfg,
    scales: Option<Vec<f64>>,
    sizes: Option<Vec<i64>>,
    axes: Vec<usize>,
    roi: Option<Vec<f64>>,
    keep: &str,
) -> Result<ValueId> {
    let xd = static_dims(ctx, x, "Resize")?;
    let r = xd.len();
    let mut scale_all = vec![1.0f64; r];
    let mut size_all: Vec<i64> = xd.iter().map(|&v| v as i64).collect();
    let mut roi_all = vec![0.0f64; 2 * r];
    roi_all[r..].fill(1.0);
    if let Some(roi) = &roi {
        let na = axes.len();
        for (i, &a) in axes.iter().enumerate() {
            roi_all[a] = roi[i];
            roi_all[r + a] = roi[na + i];
        }
    }
    if let Some(sizes) = &sizes {
        for (i, &a) in axes.iter().enumerate() {
            size_all[a] = sizes[i];
        }
        for a in 0..r {
            scale_all[a] = size_all[a] as f64 / xd[a] as f64;
        }
        if keep != "stretch" {
            let sc: Vec<f64> = axes.iter().map(|&a| scale_all[a]).collect();
            let s = if keep == "not_larger" {
                sc.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };
            for a in 0..r {
                if axes.contains(&a) {
                    scale_all[a] = s;
                    size_all[a] = (s * xd[a] as f64 + 0.5) as i64;
                }
            }
        }
    } else if let Some(scales) = &scales {
        for (i, &a) in axes.iter().enumerate() {
            scale_all[a] = scales[i];
        }
        for a in 0..r {
            size_all[a] = (scale_all[a] * xd[a] as f64) as i64;
        }
    }
    let mut plan = Vec::new();
    for &a in &axes {
        let s = scale_all[a];
        let out_w = size_all[a] as usize;
        let roi_a = Some((roi_all[a], roi_all[r + a]));
        let trivial_roi = roi.is_none() || (roi_all[a] == 0.0 && roi_all[r + a] == 1.0);
        if (s - 1.0).abs() < 1e-9 && out_w == xd[a] as usize && trivial_roi {
            continue;
        }
        let (m, extra) = resize_matrix(cfg, xd[a] as usize, out_w, s, roi_a);
        plan.push((a, out_w, m, extra));
    }
    apply_axis_matrices(ctx, x, &plan)
}

fn resize(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let cfg = ResizeCfg {
        mode: ctx.attr_s("mode").unwrap_or("nearest").to_string(),
        ctm: ctx
            .attr_s("coordinate_transformation_mode")
            .unwrap_or("half_pixel")
            .to_string(),
        nearest_mode: ctx
            .attr_s("nearest_mode")
            .unwrap_or("round_prefer_floor")
            .to_string(),
        cubic_a: ctx.attr_f("cubic_coeff_a").unwrap_or(-0.75) as f64,
        exclude_outside: ctx.attr_i("exclude_outside").unwrap_or(0) != 0,
        extrapolation: ctx.attr_f("extrapolation_value").unwrap_or(0.0) as f64,
        antialias: ctx.attr_i("antialias").unwrap_or(0) != 0,
    };
    let axes: Vec<usize> = match ctx.attr_is("axes") {
        Some(a) => a
            .iter()
            .map(|&v| ctx.norm_axis(v, r))
            .collect::<Result<_>>()?,
        None => (0..r).collect(),
    };
    let roi = if ctx.has_input(1) {
        const_floats(ctx, 1).filter(|v| !v.is_empty())
    } else {
        None
    };
    let scales = if ctx.has_input(2) {
        const_floats(ctx, 2).filter(|v| !v.is_empty())
    } else {
        None
    };
    let sizes = if ctx.has_input(3) {
        const_ints(ctx, 3)
    } else {
        None
    };
    if scales.is_none() && sizes.is_none() {
        return Err(Error::Unsupported(format!(
            "node '{}': Resize scales/sizes must be constant",
            ctx.node_name()
        )));
    }
    let keep = ctx
        .attr_s("keep_aspect_ratio_policy")
        .unwrap_or("stretch")
        .to_string();
    let y = resize_core(ctx, x, &cfg, scales, sizes, axes, roi, &keep)?;
    out(ctx, y)
}

/// Deprecated Upsample: asymmetric nearest/linear resize by `scales`.
fn upsample(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let mode = ctx.attr_s("mode").unwrap_or("nearest").to_string();
    let scales = if ctx.has_input(1) {
        const_floats(ctx, 1)
    } else {
        ctx.node_attr_floats("scales")
    }
    .ok_or_else(|| Error::Unsupported("Upsample scales must be constant".into()))?;
    let cfg = ResizeCfg {
        mode,
        ctm: "asymmetric".into(),
        nearest_mode: "floor".into(),
        cubic_a: -0.75,
        exclude_outside: false,
        extrapolation: 0.0,
        antialias: false,
    };
    let y = resize_core(
        ctx,
        x,
        &cfg,
        Some(scales),
        None,
        (0..r).collect(),
        None,
        "stretch",
    )?;
    out(ctx, y)
}

// ─────────────────────────────── windows ───────────────────────────────

#[derive(Clone, Copy)]
enum Window {
    Hann,
    Hamming,
    Blackman,
}

fn window(ctx: &mut LowerCtx, kind: Window) -> Result<()> {
    let size = require_const_ints(ctx, 0, "window size")?[0].max(0) as usize;
    let periodic = ctx.attr_i("periodic").unwrap_or(1) != 0;
    let dt = convert_proto_dtype(ctx.attr_i("output_datatype").unwrap_or(1))?;
    let n1 = if periodic {
        size as f64
    } else {
        (size as f64 - 1.0).max(1.0)
    };
    let pi = std::f64::consts::PI;
    let vals: Vec<f64> = (0..size)
        .map(|i| {
            let n = i as f64;
            match kind {
                Window::Hann => (n * pi / n1).sin().powi(2),
                Window::Hamming => {
                    let alpha = 25.0 / 46.0;
                    alpha - (n * pi * 2.0 / n1).cos() * (1.0 - alpha)
                }
                Window::Blackman => {
                    0.42 - 0.5 * (n * 2.0 * pi / n1).cos() + 0.08 * (n * 4.0 * pi / n1).cos()
                }
            }
        })
        .collect();
    let y = const_typed(ctx, dt, &vals, &[size as u64])?;
    out(ctx, y)
}

fn mel_weight_matrix(ctx: &mut LowerCtx) -> Result<()> {
    let num_mel = require_const_ints(ctx, 0, "num_mel_bins")?[0] as usize;
    let dft_len = require_const_ints(ctx, 1, "dft_length")?[0];
    let sample_rate = require_const_ints(ctx, 2, "sample_rate")?[0];
    let lower = const_floats(ctx, 3)
        .ok_or_else(|| Error::Unsupported("lower_edge_hertz must be constant".into()))?[0];
    let upper = const_floats(ctx, 4)
        .ok_or_else(|| Error::Unsupported("upper_edge_hertz must be constant".into()))?[0];
    let dt = convert_proto_dtype(ctx.attr_i("output_datatype").unwrap_or(1))?;
    let nbins = (dft_len / 2 + 1) as usize;
    let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let lo_mel = mel(lower);
    let hi_mel = mel(upper);
    let count = num_mel + 2;
    let step = (hi_mel - lo_mel) / count as f64;
    let bins: Vec<i64> = (0..count)
        .map(|i| {
            let m = i as f64 * step + lo_mel;
            let hz = 700.0 * (10f64.powf(m / 2595.0) - 1.0);
            (((dft_len + 1) as f64 * hz) / sample_rate as f64).floor() as i64
        })
        .collect();
    let mut m = vec![0.0f64; nbins * num_mel];
    for i in 0..num_mel {
        let (lo, cen, hi) = (bins[i], bins[i + 1], bins[i + 2]);
        let l2c = cen - lo;
        if l2c == 0 {
            if (cen as usize) < nbins {
                m[cen as usize * num_mel + i] = 1.0;
            }
        } else {
            for j in lo..=cen {
                if j >= 0 && (j as usize) < nbins {
                    m[j as usize * num_mel + i] = (j - lo) as f64 / l2c as f64;
                }
            }
        }
        let c2h = hi - cen;
        if c2h > 0 {
            for j in cen..hi {
                if j >= 0 && (j as usize) < nbins {
                    m[j as usize * num_mel + i] = (hi - j) as f64 / c2h as f64;
                }
            }
        }
    }
    let y = const_typed(ctx, dt, &m, &[nbins as u64, num_mel as u64])?;
    out(ctx, y)
}

// ────────────────────────────────── DFT ────────────────────────────────

/// DFT along `axis` of a `[..., 2|1]` complex-last tensor via cos/sin
/// matrices. Returns `[..., K, 2]` (or `[..., L, 1]` for the inverse
/// one-sided case).
fn dft_core(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
    dft_len: usize,
    inverse: bool,
    onesided: bool,
) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let last = r - 1;
    let xd = dims(ctx, x);
    let parts = xd[last].as_const().unwrap_or(1) as usize;
    let n_in = xd[axis]
        .as_const()
        .ok_or_else(|| Error::Unsupported("DFT over a symbolic axis".into()))?
        as usize;
    // Real / imaginary planes [..., n] with `axis` moved last.
    let re = slice_axis(ctx, x, last, 0, 1)?;
    let re = squeeze(ctx, re, last)?;
    let (re, inv) = axis_to_last(ctx, re, axis)?;
    let im = if parts == 2 {
        let im = slice_axis(ctx, x, last, 1, 2)?;
        let im = squeeze(ctx, im, last)?;
        Some(axis_to_last(ctx, im, axis)?.0)
    } else {
        None
    };
    let pi2 = 2.0 * std::f64::consts::PI;
    let (yr, yi, out_real_only) = if inverse && onesided {
        // IRFFT: one-sided spectrum [.., n_in] → real signal [.., L].
        let l = dft_len;
        let mut cw = vec![0.0; n_in * l];
        let mut sw = vec![0.0; n_in * l];
        for k in 0..n_in {
            let w = if k == 0 || (l % 2 == 0 && k == l / 2) {
                1.0
            } else {
                2.0
            };
            for n in 0..l {
                let th = pi2 * (k * n) as f64 / l as f64;
                cw[k * l + n] = w * th.cos() / l as f64;
                sw[k * l + n] = -w * th.sin() / l as f64;
            }
        }
        let cm = const_typed(ctx, dt, &cw, &[n_in as u64, l as u64])?;
        let sm = const_typed(ctx, dt, &sw, &[n_in as u64, l as u64])?;
        let a = mm(ctx, re, cm)?;
        let y = match im {
            Some(im) => {
                let b = mm(ctx, im, sm)?;
                add(ctx, a, b)?
            }
            None => a,
        };
        (y, None, true)
    } else {
        // Truncate / zero-pad the signal to L, then [n, K] matrices.
        let l = dft_len;
        let fit = |ctx: &mut LowerCtx, v: ValueId| -> Result<ValueId> {
            let v = if n_in > l {
                slice_axis(ctx, v, last - 1, 0, l as u64)?
            } else {
                v
            };
            if n_in < l {
                let zero = scalar(ctx, dt, 0.0)?;
                pad_axis_const(ctx, v, last - 1, 0, (l - n_in) as i64, zero)
            } else {
                Ok(v)
            }
        };
        let re = fit(ctx, re)?;
        let im = match im {
            Some(v) => Some(fit(ctx, v)?),
            None => None,
        };
        let k_out = if onesided { l / 2 + 1 } else { l };
        let mut cm = vec![0.0; l * k_out];
        let mut sm = vec![0.0; l * k_out];
        for n in 0..l {
            for k in 0..k_out {
                let th = pi2 * (n * k) as f64 / l as f64;
                cm[n * k_out + k] = th.cos();
                sm[n * k_out + k] = th.sin();
            }
        }
        let norm = if inverse { 1.0 / l as f64 } else { 1.0 };
        if inverse {
            for v in cm.iter_mut().chain(sm.iter_mut()) {
                *v *= norm;
            }
        }
        let cmat = const_typed(ctx, dt, &cm, &[l as u64, k_out as u64])?;
        let smat = const_typed(ctx, dt, &sm, &[l as u64, k_out as u64])?;
        // forward: Yr = re·C + im·S ; Yi = im·C - re·S
        // inverse: Yr = re·C - im·S ; Yi = im·C + re·S   (scaled by 1/L)
        let rc = mm(ctx, re, cmat)?;
        let rs = mm(ctx, re, smat)?;
        let (yr, yi) = match im {
            Some(im) => {
                let ic = mm(ctx, im, cmat)?;
                let is = mm(ctx, im, smat)?;
                if inverse {
                    (sub(ctx, rc, is)?, add(ctx, ic, rs)?)
                } else {
                    (add(ctx, rc, is)?, sub(ctx, ic, rs)?)
                }
            }
            None => {
                if inverse {
                    (rc, rs)
                } else {
                    let nrs = unary(ctx, UnaryOp::Neg, rs)?;
                    (rc, nrs)
                }
            }
        };
        (yr, Some(yi), false)
    };
    let yr = transpose(ctx, yr, &inv)?;
    let yr = unsqueeze(ctx, yr, last)?;
    let y = if out_real_only {
        yr
    } else {
        let yi = transpose(ctx, yi.unwrap(), &inv)?;
        let yi = unsqueeze(ctx, yi, last)?;
        concat(ctx, &[yr, yi], last)?
    };
    Ok(y)
}

/// MatMul of `[..., n]` by `[n, k]`, promoting rank-1 operands.
fn mm(ctx: &mut LowerCtx, a: ValueId, m: ValueId) -> Result<ValueId> {
    let r = rank(ctx, a);
    let lhs = if r == 1 { unsqueeze(ctx, a, 0)? } else { a };
    let y = matmul(ctx, lhs, m)?;
    if r == 1 { squeeze(ctx, y, 0) } else { Ok(y) }
}

fn dft(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let inverse = ctx.attr_i("inverse").unwrap_or(0) != 0;
    let onesided = ctx.attr_i("onesided").unwrap_or(0) != 0;
    // opset 17: axis attr (default 1); opset 20: axis input (default -2).
    let axis_raw = if ctx.has_input(2) {
        require_const_ints(ctx, 2, "DFT axis")?[0]
    } else {
        ctx.attr_i("axis").unwrap_or(-2)
    };
    let axis = ctx.norm_axis(axis_raw, r)?;
    let n_in = dims(ctx, x)[axis]
        .as_const()
        .ok_or_else(|| Error::Unsupported("DFT over a symbolic axis".into()))?
        as usize;
    let dft_len = if ctx.has_input(1) {
        require_const_ints(ctx, 1, "dft_length")?[0] as usize
    } else if inverse && onesided {
        2 * (n_in - 1)
    } else {
        n_in
    };
    let y = dft_core(ctx, x, axis, dft_len, inverse, onesided)?;
    out(ctx, y)
}

fn stft(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?; // [B, L, 1|2]
    let dt = dtype(ctx, x);
    let step = require_const_ints(ctx, 1, "frame_step")?[0] as usize;
    let window = opt_val(ctx, 2)?;
    let xd = static_dims(ctx, x, "STFT")?;
    let (bsz, len, parts) = (xd[0], xd[1] as usize, xd[2]);
    let frame_len = if ctx.has_input(3) {
        require_const_ints(ctx, 3, "frame_length")?[0] as usize
    } else if let Some(w) = window {
        static_dims(ctx, w, "STFT window")?[0] as usize
    } else {
        len
    };
    let onesided = ctx.attr_i("onesided").unwrap_or(1) != 0;
    let n_frames = 1 + (len - frame_len) / step;
    // Frame index table [n_frames, frame_len] → gather on the signal axis.
    let table: Vec<i64> = (0..n_frames)
        .flat_map(|f| (0..frame_len).map(move |i| (f * step + i) as i64))
        .collect();
    let idx = const_i64(ctx, &table, &[n_frames as u64, frame_len as u64])?;
    let frames = ctx.emit(Prim::Gather { axis: 1 }, &[x, idx])?; // [B, F, W, P]
    let frames = if let Some(w) = window {
        let w = reshape(ctx, w, vec![c(1), c(1), c(frame_len as u64), c(1)])?;
        mul(ctx, frames, w)?
    } else {
        frames
    };
    let _ = (bsz, parts, dt);
    let y = dft_core(ctx, frames, 2, frame_len, false, onesided)?;
    out(ctx, y)
}

// ───────────────────────── AffineGrid / GridSample ─────────────────────

fn affine_grid(ctx: &mut LowerCtx) -> Result<()> {
    let theta = val(ctx, 0)?;
    let size = require_const_ints(ctx, 1, "AffineGrid size")?;
    let align = ctx.attr_i("align_corners").unwrap_or(0) != 0;
    let dt = dtype(ctx, theta);
    let sp: Vec<usize> = size[2..].iter().map(|&v| v as usize).collect();
    let nd = sp.len();
    let total: usize = sp.iter().product();
    // Homogeneous base grid [total, nd+1] with (x, y[, z], 1), x along
    // the innermost spatial axis.
    let coord = |dim: usize, i: usize| -> f64 {
        if align {
            if dim == 1 {
                0.0
            } else {
                -1.0 + 2.0 * i as f64 / (dim as f64 - 1.0)
            }
        } else {
            let step = 2.0 / dim as f64;
            -1.0 + step / 2.0 + step * i as f64
        }
    };
    let mut base = vec![0.0; total * (nd + 1)];
    for p in 0..total {
        let mut rem = p;
        let mut coords = vec![0usize; nd];
        for i in (0..nd).rev() {
            coords[i] = rem % sp[i];
            rem /= sp[i];
        }
        for j in 0..nd {
            // column j is the coordinate of spatial axis nd-1-j.
            let ax = nd - 1 - j;
            base[p * (nd + 1) + j] = coord(sp[ax], coords[ax]);
        }
        base[p * (nd + 1) + nd] = 1.0;
    }
    let b = const_typed(ctx, dt, &base, &[total as u64, nd as u64 + 1])?;
    let b = unsqueeze(ctx, b, 0)?; // [1, total, nd+1]
    let g = ctx.emit(
        Prim::MatMul {
            trans_a: false,
            trans_b: true,
        },
        &[b, theta],
    )?; // [N, total, nd]
    let mut od = vec![c(size[0] as u64)];
    od.extend(sp.iter().map(|&v| c(v as u64)));
    od.push(c(nd as u64));
    let y = reshape(ctx, g, od)?;
    out(ctx, y)
}

/// Integer index with GridSample padding: returns (index, valid-mask).
fn gs_pad_index(
    ctx: &mut LowerCtx,
    i: ValueId,
    d: u64,
    mode: &str,
) -> Result<(ValueId, Option<ValueId>)> {
    let zero = scalar(ctx, DataType::I64, 0.0)?;
    let dm1 = scalar(ctx, DataType::I64, d as f64 - 1.0)?;
    match mode {
        "border" => {
            let v = max(ctx, i, zero)?;
            Ok((min(ctx, v, dm1)?, None))
        }
        "reflection" => {
            // Reflect about the pixel edges: period 2d, -1 → 0, d → d-1.
            let period = scalar(ctx, DataType::I64, 2.0 * d as f64)?;
            let q = div(ctx, i, period)?;
            let qp = mul(ctx, q, period)?;
            let m = sub(ctx, i, qp)?; // truncated remainder, may be negative
            let neg = cmp(ctx, CmpOp::Lt, m, zero)?;
            let mp = add(ctx, m, period)?;
            let m = select(ctx, neg, mp, m)?;
            let dd = scalar(ctx, DataType::I64, d as f64)?;
            let over = cmp(ctx, CmpOp::Ge, m, dd)?;
            let pm1 = scalar(ctx, DataType::I64, 2.0 * d as f64 - 1.0)?;
            let refl = sub(ctx, pm1, m)?;
            Ok((select(ctx, over, refl, m)?, None))
        }
        _ => {
            let ge = cmp(ctx, CmpOp::Ge, i, zero)?;
            let le = cmp(ctx, CmpOp::Le, i, dm1)?;
            let valid = binary(ctx, BinaryOp::And, ge, le)?;
            let v = max(ctx, i, zero)?;
            Ok((min(ctx, v, dm1)?, Some(valid)))
        }
    }
}

fn grid_sample(ctx: &mut LowerCtx) -> Result<()> {
    let (x, grid) = (val(ctx, 0)?, val(ctx, 1)?);
    let mode = ctx.attr_s("mode").unwrap_or("linear").to_string();
    let padding = ctx.attr_s("padding_mode").unwrap_or("zeros").to_string();
    let align = ctx.attr_i("align_corners").unwrap_or(0) != 0;
    let xd = static_dims(ctx, x, "GridSample")?;
    let gd = dims(ctx, grid);
    let nd = xd.len() - 2;
    let (n, ch) = (xd[0], xd[1]);
    let sp: Vec<u64> = xd[2..].to_vec();
    let plane: u64 = sp.iter().product();
    let dt = dtype(ctx, x);
    let cdt = if dt.is_float() { dt } else { DataType::F32 };
    let gdt = dtype(ctx, grid);
    let out_sp: Vec<DimExpr> = gd[1..1 + nd].to_vec();
    let out_rank = nd + 2;
    // Per spatial axis j (grid column nd-1-j): denormalized float coord
    // of shape [N, 1, out...].
    let mut coords = Vec::with_capacity(nd);
    for (j, &spj) in sp.iter().enumerate().take(nd) {
        let col = nd - 1 - j;
        let g = slice_axis(ctx, grid, nd + 1, col as u64, col as u64 + 1)?; // [N, out..., 1]
        let mut gs = vec![c(n), c(1)];
        gs.extend(out_sp.iter().cloned());
        let g = reshape(ctx, g, gs)?;
        let g = cast(ctx, g, cdt)?;
        let len = spj as f64;
        let one = scalar(ctx, cdt, 1.0)?;
        let g1 = add(ctx, g, one)?;
        let v = if align {
            let f = scalar(ctx, cdt, (len - 1.0) / 2.0)?;
            mul(ctx, g1, f)?
        } else {
            let f = scalar(ctx, cdt, len / 2.0)?;
            let h = scalar(ctx, cdt, 0.5)?;
            let t = mul(ctx, g1, f)?;
            sub(ctx, t, h)?
        };
        coords.push(v);
    }
    let _ = gdt;
    // Taps per axis: (integer base index, fractional weights).
    let taps: usize = match mode.as_str() {
        "nearest" => 1,
        "cubic" => 4,
        _ => 2,
    };
    // For each axis: list of (index tensor i64, weight tensor) per tap.
    let mut axis_taps: Vec<Vec<(ValueId, ValueId)>> = Vec::with_capacity(nd);
    for (j, &cv) in coords.iter().enumerate() {
        let mut list = Vec::with_capacity(taps);
        match mode.as_str() {
            "nearest" => {
                let r = unary(ctx, UnaryOp::Round, cv)?;
                let i = cast(ctx, r, DataType::I64)?;
                let w = scalar(ctx, cdt, 1.0)?;
                list.push((i, w));
            }
            "cubic" => {
                let fl = unary(ctx, UnaryOp::Floor, cv)?;
                let t = sub(ctx, cv, fl)?; // fraction
                let i0 = cast(ctx, fl, DataType::I64)?;
                let a = -0.75f64;
                // Keys coefficients at distances t+1, t, 1-t, 2-t.
                let one = scalar(ctx, cdt, 1.0)?;
                let two = scalar(ctx, cdt, 2.0)?;
                let d0 = add(ctx, t, one)?;
                let d2 = sub(ctx, one, t)?;
                let d3 = sub(ctx, two, t)?;
                let near = |ctx: &mut LowerCtx, x: ValueId| -> Result<ValueId> {
                    // ((a+2)x - (a+3))x² + 1
                    let ca2 = scalar(ctx, cdt, a + 2.0)?;
                    let ca3 = scalar(ctx, cdt, a + 3.0)?;
                    let one = scalar(ctx, cdt, 1.0)?;
                    let p = mul(ctx, ca2, x)?;
                    let p = sub(ctx, p, ca3)?;
                    let x2 = mul(ctx, x, x)?;
                    let p = mul(ctx, p, x2)?;
                    add(ctx, p, one)
                };
                let far = |ctx: &mut LowerCtx, x: ValueId| -> Result<ValueId> {
                    // ((a x - 5a) x + 8a) x - 4a
                    let ca = scalar(ctx, cdt, a)?;
                    let c5 = scalar(ctx, cdt, 5.0 * a)?;
                    let c8 = scalar(ctx, cdt, 8.0 * a)?;
                    let c4 = scalar(ctx, cdt, 4.0 * a)?;
                    let p = mul(ctx, ca, x)?;
                    let p = sub(ctx, p, c5)?;
                    let p = mul(ctx, p, x)?;
                    let p = add(ctx, p, c8)?;
                    let p = mul(ctx, p, x)?;
                    sub(ctx, p, c4)
                };
                let w0 = far(ctx, d0)?;
                let w1 = near(ctx, t)?;
                let w2 = near(ctx, d2)?;
                let w3 = far(ctx, d3)?;
                for (k, w) in [(-1i64, w0), (0, w1), (1, w2), (2, w3)] {
                    let off = scalar(ctx, DataType::I64, k as f64)?;
                    let i = add(ctx, i0, off)?;
                    list.push((i, w));
                }
            }
            _ => {
                let fl = unary(ctx, UnaryOp::Floor, cv)?;
                let t = sub(ctx, cv, fl)?;
                let i0 = cast(ctx, fl, DataType::I64)?;
                let one_i = scalar(ctx, DataType::I64, 1.0)?;
                let i1 = add(ctx, i0, one_i)?;
                let one = scalar(ctx, cdt, 1.0)?;
                let w0 = sub(ctx, one, t)?;
                list.push((i0, w0));
                list.push((i1, t));
            }
        }
        let _ = j;
        axis_taps.push(list);
    }
    // Enumerate all tap combinations.
    let total_combos = taps.pow(nd as u32);
    let xflat = reshape(ctx, x, vec![c(n * ch * plane)])?;
    let xflat = cast(ctx, xflat, cdt)?;
    let strides_sp: Vec<u64> = {
        let mut s = vec![1u64; nd];
        for i in (0..nd.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * sp[i + 1];
        }
        s
    };
    let ni = iota_along(ctx, c(n), out_rank, 0, DataType::I64)?;
    let ci = iota_along(ctx, c(ch), out_rank, 1, DataType::I64)?;
    let cp = scalar(ctx, DataType::I64, (ch * plane) as f64)?;
    let pl = scalar(ctx, DataType::I64, plane as f64)?;
    let nb = mul(ctx, ni, cp)?;
    let cb = mul(ctx, ci, pl)?;
    let base = add(ctx, nb, cb)?; // [N, C, 1...]
    let mut acc: Option<ValueId> = None;
    for combo in 0..total_combos {
        let mut rem = combo;
        let mut lin: Option<ValueId> = None;
        let mut weight: Option<ValueId> = None;
        let mut valid: Option<ValueId> = None;
        for j in 0..nd {
            let k = rem % taps;
            rem /= taps;
            let (idx, w) = axis_taps[j][k];
            let (pi, v) = gs_pad_index(ctx, idx, sp[j], &padding)?;
            let s = scalar(ctx, DataType::I64, strides_sp[j] as f64)?;
            let term = mul(ctx, pi, s)?;
            lin = Some(match lin {
                None => term,
                Some(l) => add(ctx, l, term)?,
            });
            weight = Some(match weight {
                None => w,
                Some(p) => mul(ctx, p, w)?,
            });
            if let Some(v) = v {
                valid = Some(match valid {
                    None => v,
                    Some(p) => binary(ctx, BinaryOp::And, p, v)?,
                });
            }
        }
        let lin = lin.unwrap();
        let lin = add(ctx, lin, base)?; // [N, C, out...]
        let mut od = vec![c(n), c(ch)];
        od.extend(out_sp.iter().cloned());
        let lin = broadcast(ctx, lin, od)?;
        let vals = linear_gather(ctx, xflat, lin)?;
        let mut w = weight.unwrap();
        if let Some(v) = valid {
            let zero = scalar(ctx, cdt, 0.0)?;
            w = select(ctx, v, w, zero)?;
        }
        let term = mul(ctx, vals, w)?;
        acc = Some(match acc {
            None => term,
            Some(a) => add(ctx, a, term)?,
        });
    }
    let y = cast(ctx, acc.unwrap(), dt)?;
    out(ctx, y)
}

// ─────────────────────────────── Einsum ────────────────────────────────

fn einsum(ctx: &mut LowerCtx) -> Result<()> {
    let eq: String = ctx
        .attr_s("equation")
        .ok_or_else(|| ctx.missing_attr("equation"))?
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let n_in = ctx.num_inputs();
    let inputs: Vec<ValueId> = (0..n_in).map(|i| val(ctx, i)).collect::<Result<_>>()?;
    let (lhs, rhs) = match eq.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (eq.clone(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != n_in {
        return Err(Error::Attribute(format!(
            "Einsum '{eq}' has {} terms for {n_in} inputs",
            terms.len()
        )));
    }
    // Expand ellipses into synthetic labels ('0'..'9' + more, never letters).
    let ell_labels: Vec<char> = "0123456789ABCDEFGHIJ".chars().collect();
    let mut ell_rank = 0usize;
    let mut term_labels: Vec<Vec<char>> = Vec::with_capacity(n_in);
    for (i, t) in terms.iter().enumerate() {
        let r = rank(ctx, inputs[i]);
        let (before, after) = match t.split_once("...") {
            Some((b, a)) => (b, Some(a)),
            None => (*t, None),
        };
        let mut labels: Vec<char> = before.chars().collect();
        if let Some(a) = after {
            let explicit = before.len() + a.len();
            let er = r
                .checked_sub(explicit)
                .ok_or_else(|| Error::Shape(format!("Einsum term '{t}' exceeds rank {r}")))?;
            ell_rank = ell_rank.max(er);
            // Right-aligned ellipsis labels.
            for k in 0..er {
                labels.push(ell_labels[ell_labels.len() - er + k]);
            }
            labels.extend(a.chars());
        }
        if labels.len() != r {
            return Err(Error::Shape(format!(
                "Einsum term '{t}' has {} labels for rank {r}",
                labels.len()
            )));
        }
        term_labels.push(labels);
    }
    let ell_all: Vec<char> = ell_labels[ell_labels.len() - ell_rank..].to_vec();
    let out_labels: Vec<char> = match rhs {
        Some(r) => {
            let (before, after) = match r.split_once("...") {
                Some((b, a)) => (b.to_string(), Some(a.to_string())),
                None => (r.clone(), None),
            };
            let mut v: Vec<char> = before.chars().collect();
            if let Some(a) = after {
                v.extend(ell_all.iter().cloned());
                v.extend(a.chars());
            }
            v
        }
        None => {
            // Implicit: ellipsis dims first, then labels appearing once, sorted.
            let mut counts = std::collections::BTreeMap::new();
            for t in &term_labels {
                for &l in t {
                    if !ell_all.contains(&l) {
                        *counts.entry(l).or_insert(0usize) += 1;
                    }
                }
            }
            let mut v = ell_all.clone();
            v.extend(counts.iter().filter(|(_, n)| **n == 1).map(|(l, _)| *l));
            v
        }
    };
    // Label dims (first occurrence with size ≠ 1 wins, for broadcasting).
    let mut label_dim: std::collections::BTreeMap<char, DimExpr> =
        std::collections::BTreeMap::new();
    for (i, t) in term_labels.iter().enumerate() {
        let d = dims(ctx, inputs[i]);
        for (k, &l) in t.iter().enumerate() {
            let e = label_dim.entry(l).or_insert_with(|| d[k].clone());
            if *e == c(1) {
                *e = d[k].clone();
            }
        }
    }
    // Full label order: output labels first, then the contracted ones.
    let mut all: Vec<char> = out_labels.clone();
    for t in &term_labels {
        for &l in t {
            if !all.contains(&l) {
                all.push(l);
            }
        }
    }
    let full_rank = all.len();
    // Bring every operand to [all...] layout (diagonals via gather).
    let mut prod_v: Option<ValueId> = None;
    for (i, t) in term_labels.iter().enumerate() {
        let x = inputs[i];
        let xd = dims(ctx, x);
        // Unique labels of this term, in first-occurrence order.
        let mut uniq: Vec<char> = Vec::new();
        for &l in t {
            if !uniq.contains(&l) {
                uniq.push(l);
            }
        }
        let v = if uniq.len() != t.len() {
            // Repeated labels: gather the diagonal with a linear index.
            let st = strides(&xd);
            let ur = uniq.len();
            let udims: Vec<DimExpr> = uniq.iter().map(|l| label_dim[l].clone()).collect();
            let mut lin: Option<ValueId> = None;
            for (ui, &l) in uniq.iter().enumerate() {
                let coord = iota_along(ctx, udims[ui].clone(), ur, ui, DataType::I64)?;
                let total_stride = t
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| **m == l)
                    .map(|(k, _)| st[k].clone())
                    .fold(c(0), |a, b| a + b);
                let s = dim_value(ctx, total_stride)?;
                let term = mul(ctx, coord, s)?;
                lin = Some(match lin {
                    None => term,
                    Some(a) => add(ctx, a, term)?,
                });
            }
            let lin = broadcast(ctx, lin.unwrap(), udims)?;
            linear_gather(ctx, x, lin)?
        } else {
            x
        };
        // Transpose uniq → order of `all`, then unsqueeze missing labels.
        let mut present: Vec<(usize, usize)> = all
            .iter()
            .enumerate()
            .filter_map(|(ai, l)| uniq.iter().position(|u| u == l).map(|ui| (ai, ui)))
            .collect();
        present.sort();
        let perm: Vec<usize> = present.iter().map(|&(_, ui)| ui).collect();
        let vt = transpose(ctx, v, &perm)?;
        let vtd = dims(ctx, vt);
        let mut target = vec![c(1); full_rank];
        for (pi, &(ai, _)) in present.iter().enumerate() {
            target[ai] = vtd[pi].clone();
        }
        let vr = reshape(ctx, vt, target)?;
        prod_v = Some(match prod_v {
            None => vr,
            Some(p) => mul(ctx, p, vr)?,
        });
    }
    let mut y = prod_v.ok_or_else(|| Error::Shape("Einsum with no inputs".into()))?;
    // Broadcast to the full label shape before reducing (a 1-sized label
    // in every operand stays 1 — fine).
    let full: Vec<DimExpr> = all.iter().map(|l| label_dim[l].clone()).collect();
    y = broadcast(ctx, y, full)?;
    let contracted: Vec<usize> = (out_labels.len()..full_rank).collect();
    y = reduce(ctx, ReduceOp::Sum, y, &contracted, false)?;
    out(ctx, y)
}
