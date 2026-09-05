//! RNN / GRU / LSTM, unrolled over a static sequence length. Each step is
//! a couple of MatMuls and activations; `sequence_lens` masks steps past
//! each row's length so the hidden state freezes and `Y` reads zero.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry};
use onyxia_ir::prim::{CmpOp, Prim, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "RNN", |c| rnn(c, Cell::Rnn));
    r.register("", "GRU", |c| rnn(c, Cell::Gru));
    r.register("", "LSTM", |c| rnn(c, Cell::Lstm));
}

#[derive(Clone, Copy, PartialEq)]
enum Cell {
    Rnn,
    Gru,
    Lstm,
}

impl Cell {
    fn gates(self) -> u64 {
        match self {
            Cell::Rnn => 1,
            Cell::Gru => 3,
            Cell::Lstm => 4,
        }
    }
    fn default_acts(self) -> Vec<&'static str> {
        match self {
            Cell::Rnn => vec!["Tanh"],
            Cell::Gru => vec!["Sigmoid", "Tanh"],
            Cell::Lstm => vec!["Sigmoid", "Tanh", "Tanh"],
        }
    }
}

/// One activation by ONNX name with optional alpha/beta.
fn activation(
    ctx: &mut LowerCtx,
    name: &str,
    x: ValueId,
    alpha: Option<f64>,
    beta: Option<f64>,
) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    Ok(match name {
        "Tanh" => unary(ctx, UnaryOp::Tanh, x)?,
        "Sigmoid" => sigmoid(ctx, x)?,
        "Relu" => {
            let z = scalar(ctx, dt, 0.0)?;
            max(ctx, x, z)?
        }
        "Affine" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(1.0))?;
            let b = scalar(ctx, dt, beta.unwrap_or(0.0))?;
            let ax = mul(ctx, x, a)?;
            add(ctx, ax, b)?
        }
        "LeakyRelu" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(0.01))?;
            let z = scalar(ctx, dt, 0.0)?;
            let neg = cmp(ctx, CmpOp::Lt, x, z)?;
            let ax = mul(ctx, x, a)?;
            select(ctx, neg, ax, x)?
        }
        "ThresholdedRelu" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(1.0))?;
            let z = scalar(ctx, dt, 0.0)?;
            let keep = cmp(ctx, CmpOp::Ge, x, a)?;
            select(ctx, keep, x, z)?
        }
        "ScaledTanh" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(1.0))?;
            let b = scalar(ctx, dt, beta.unwrap_or(1.0))?;
            let bx = mul(ctx, x, b)?;
            let t = unary(ctx, UnaryOp::Tanh, bx)?;
            mul(ctx, t, a)?
        }
        "HardSigmoid" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(0.2))?;
            let b = scalar(ctx, dt, beta.unwrap_or(0.5))?;
            let one = scalar(ctx, dt, 1.0)?;
            let z = scalar(ctx, dt, 0.0)?;
            let ax = mul(ctx, x, a)?;
            let axb = add(ctx, ax, b)?;
            let m = max(ctx, axb, z)?;
            min(ctx, m, one)?
        }
        "Elu" => {
            let a = scalar(ctx, dt, alpha.unwrap_or(1.0))?;
            let one = scalar(ctx, dt, 1.0)?;
            let z = scalar(ctx, dt, 0.0)?;
            let neg = cmp(ctx, CmpOp::Lt, x, z)?;
            let e = unary(ctx, UnaryOp::Exp, x)?;
            let em1 = sub(ctx, e, one)?;
            let aem1 = mul(ctx, a, em1)?;
            select(ctx, neg, aem1, x)?
        }
        "Softsign" => {
            let one = scalar(ctx, dt, 1.0)?;
            let ax = unary(ctx, UnaryOp::Abs, x)?;
            let d = add(ctx, one, ax)?;
            div(ctx, x, d)?
        }
        "Softplus" => softplus(ctx, x)?,
        other => return Err(Error::Unsupported(format!("RNN activation '{other}'"))),
    })
}

fn rnn(ctx: &mut LowerCtx, cell: Cell) -> Result<()> {
    let (x, w, r) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let bias = opt_val(ctx, 3)?;
    let seq_lens = opt_val(ctx, 4)?;
    let initial_h = opt_val(ctx, 5)?;
    let (initial_c, peep) = if cell == Cell::Lstm {
        (opt_val(ctx, 6)?, opt_val(ctx, 7)?)
    } else {
        (None, None)
    };
    let dt = dtype(ctx, x);
    let layout = ctx.attr_i("layout").unwrap_or(0);
    let direction = ctx.attr_s("direction").unwrap_or("forward").to_string();
    let clip = ctx.attr_f("clip").map(|v| v as f64);
    let linear_before_reset = ctx.attr_i("linear_before_reset").unwrap_or(0) != 0;
    let acts: Vec<String> = ctx
        .node_attr_strings("activations")
        .unwrap_or_else(|| cell.default_acts().iter().map(|s| s.to_string()).collect());
    let alphas = ctx.node_attr_floats("activation_alpha").unwrap_or_default();
    let betas = ctx.node_attr_floats("activation_beta").unwrap_or_default();

    // X → [T, B, I]
    let x = if layout == 1 {
        transpose(ctx, x, &[1, 0, 2])?
    } else {
        x
    };
    let xd = dims(ctx, x);
    let t_len = xd[0]
        .as_const()
        .ok_or_else(|| Error::Unsupported("RNN over a symbolic sequence length".into()))?;
    let bsz = xd[1].clone();
    let wd = dims(ctx, w);
    let n_dir = wd[0].as_const().unwrap_or(1) as usize;
    let g = cell.gates();
    let hidden = wd[1]
        .clone()
        .div_exact(&c(g))
        .ok_or_else(|| Error::Shape("RNN: weight rows not divisible by gate count".into()))?;
    let h_const = hidden
        .as_const()
        .ok_or_else(|| Error::Unsupported("RNN with symbolic hidden size".into()))?;
    let dirs: Vec<bool> = match direction.as_str() {
        "forward" => vec![false],
        "reverse" => vec![true],
        _ => vec![false, true],
    };
    if dirs.len() != n_dir {
        return Err(Error::Shape(format!(
            "RNN: {n_dir} weight directions for '{direction}'"
        )));
    }
    // sequence_lens mask per step: [T, B, 1] bool.
    let mask_t: Option<ValueId> = match seq_lens {
        Some(sl) => {
            let sl = cast(ctx, sl, DataType::I64)?;
            let sl = reshape(ctx, sl, vec![c(1), bsz.clone(), c(1)])?;
            let t = iota_along(ctx, c(t_len), 3, 0, DataType::I64)?;
            Some(cmp(ctx, CmpOp::Lt, t, sl)?)
        }
        None => None,
    };
    let zero = scalar(ctx, dt, 0.0)?;
    let mut ys = Vec::with_capacity(n_dir);
    let mut yhs = Vec::with_capacity(n_dir);
    let mut ycs = Vec::with_capacity(n_dir);
    for (d, &reverse) in dirs.iter().enumerate() {
        let n_act = acts.len() / n_dir;
        let act = |k: usize| -> (String, Option<f64>, Option<f64>) {
            let i = d * n_act + k;
            (
                acts.get(i)
                    .cloned()
                    .unwrap_or_else(|| cell.default_acts()[k].to_string()),
                alphas.get(i).copied(),
                betas.get(i).copied(),
            )
        };
        // Per-direction parameters.
        let wd_ = slice_axis(ctx, w, 0, d as u64, d as u64 + 1)?;
        let wd_ = squeeze(ctx, wd_, 0)?; // [G*H, I]
        let rd_ = slice_axis(ctx, r, 0, d as u64, d as u64 + 1)?;
        let rd_ = squeeze(ctx, rd_, 0)?; // [G*H, H]
        // x·Wᵀ for every step at once: [T, B, G*H] (+ Wb + Rb).
        let xw = ctx.emit(
            Prim::MatMul {
                trans_a: false,
                trans_b: true,
            },
            &[x, wd_],
        )?;
        let (wb, rb) = match bias {
            Some(b) => {
                let bd = slice_axis(ctx, b, 0, d as u64, d as u64 + 1)?;
                let bd = squeeze(ctx, bd, 0)?; // [2*G*H]
                let wb = slice_axis(ctx, bd, 0, 0, g * h_const)?;
                let rb = slice_axis(ctx, bd, 0, g * h_const, 2 * g * h_const)?;
                (Some(wb), Some(rb))
            }
            None => (None, None),
        };
        let mut xw = xw;
        if let Some(wb) = wb {
            xw = add(ctx, xw, wb)?;
        }
        if cell != Cell::Gru {
            if let Some(rb) = rb {
                xw = add(ctx, xw, rb)?;
            }
        }
        let init = |ctx: &mut LowerCtx, v: Option<ValueId>| -> Result<ValueId> {
            match v {
                Some(v) => {
                    let s = slice_axis(ctx, v, 0, d as u64, d as u64 + 1)?;
                    squeeze(ctx, s, 0)
                }
                None => broadcast(ctx, zero, vec![bsz.clone(), hidden.clone()]),
            }
        };
        let mut h = init(ctx, initial_h)?;
        let mut cstate = if cell == Cell::Lstm {
            Some(init(ctx, initial_c)?)
        } else {
            None
        };
        let peeps: Option<[ValueId; 3]> = match (cell, peep) {
            (Cell::Lstm, Some(p)) => {
                let pd = slice_axis(ctx, p, 0, d as u64, d as u64 + 1)?;
                let pd = squeeze(ctx, pd, 0)?; // [3H] = [pi, po, pf]
                Some([
                    slice_axis(ctx, pd, 0, 0, h_const)?,
                    slice_axis(ctx, pd, 0, h_const, 2 * h_const)?,
                    slice_axis(ctx, pd, 0, 2 * h_const, 3 * h_const)?,
                ])
            }
            _ => None,
        };
        let clamp = |ctx: &mut LowerCtx, v: ValueId| -> Result<ValueId> {
            match clip {
                Some(cv) => {
                    let lo = scalar(ctx, dt, -cv)?;
                    let hi = scalar(ctx, dt, cv)?;
                    let v = max(ctx, v, lo)?;
                    min(ctx, v, hi)
                }
                None => Ok(v),
            }
        };
        let mut outs: Vec<Option<ValueId>> = vec![None; t_len as usize];
        let order: Vec<u64> = if reverse {
            (0..t_len).rev().collect()
        } else {
            (0..t_len).collect()
        };
        for &t in &order {
            let xt = slice_axis(ctx, xw, 0, t, t + 1)?;
            let xt = squeeze(ctx, xt, 0)?; // [B, G*H]
            let hr = ctx.emit(
                Prim::MatMul {
                    trans_a: false,
                    trans_b: true,
                },
                &[h, rd_],
            )?; // [B, G*H]
            // Gate `k` of a `[.., G*H]` tensor (last axis).
            let gate = |ctx: &mut LowerCtx, v: ValueId, k: u64| -> Result<ValueId> {
                let axis = rank(ctx, v) - 1;
                slice_axis(ctx, v, axis, k * h_const, (k + 1) * h_const)
            };
            let (h_new, c_new) = match cell {
                Cell::Rnn => {
                    let pre = add(ctx, xt, hr)?;
                    let pre = clamp(ctx, pre)?;
                    let (a, al, be) = act(0);
                    (activation(ctx, &a, pre, al, be)?, None)
                }
                Cell::Gru => {
                    let (fa, fal, fbe) = act(0);
                    let (ga, gal, gbe) = act(1);
                    let rbz = match rb {
                        Some(rb) => Some(gate(ctx, rb, 0)?),
                        None => None,
                    };
                    let rbr = match rb {
                        Some(rb) => Some(gate(ctx, rb, 1)?),
                        None => None,
                    };
                    let rbh = match rb {
                        Some(rb) => Some(gate(ctx, rb, 2)?),
                        None => None,
                    };
                    let xz = gate(ctx, xt, 0)?;
                    let hz = gate(ctx, hr, 0)?;
                    let mut zpre = add(ctx, xz, hz)?;
                    if let Some(b) = rbz {
                        zpre = add(ctx, zpre, b)?;
                    }
                    let xr = gate(ctx, xt, 1)?;
                    let hrr = gate(ctx, hr, 1)?;
                    let mut rpre = add(ctx, xr, hrr)?;
                    if let Some(b) = rbr {
                        rpre = add(ctx, rpre, b)?;
                    }
                    let zpre = clamp(ctx, zpre)?;
                    let rpre = clamp(ctx, rpre)?;
                    let z = activation(ctx, &fa, zpre, fal, fbe)?;
                    let rg = activation(ctx, &fa, rpre, fal, fbe)?;
                    let xh = gate(ctx, xt, 2)?;
                    let hpre = if linear_before_reset {
                        let mut hrh = gate(ctx, hr, 2)?;
                        if let Some(b) = rbh {
                            hrh = add(ctx, hrh, b)?;
                        }
                        let t = mul(ctx, rg, hrh)?;
                        add(ctx, xh, t)?
                    } else {
                        let rh = mul(ctx, rg, h)?;
                        // Rows of R for the h-gate: R is [G*H, H].
                        let rh_rows = slice_axis(ctx, rd_, 0, 2 * h_const, 3 * h_const)?;
                        let rh_r = ctx.emit(
                            Prim::MatMul {
                                trans_a: false,
                                trans_b: true,
                            },
                            &[rh, rh_rows],
                        )?;
                        let mut t = add(ctx, xh, rh_r)?;
                        if let Some(b) = rbh {
                            t = add(ctx, t, b)?;
                        }
                        t
                    };
                    let hpre = clamp(ctx, hpre)?;
                    let hh = activation(ctx, &ga, hpre, gal, gbe)?;
                    let one = scalar(ctx, dt, 1.0)?;
                    let omz = sub(ctx, one, z)?;
                    let a = mul(ctx, omz, hh)?;
                    let b = mul(ctx, z, h)?;
                    (add(ctx, a, b)?, None)
                }
                Cell::Lstm => {
                    let (fa, fal, fbe) = act(0);
                    let (ga, gal, gbe) = act(1);
                    let (ha, hal, hbe) = act(2);
                    let pre = add(ctx, xt, hr)?; // [B, 4H] as i, o, f, c
                    let cprev = cstate.unwrap();
                    let mut ipre = gate(ctx, pre, 0)?;
                    let mut opre = gate(ctx, pre, 1)?;
                    let mut fpre = gate(ctx, pre, 2)?;
                    let cpre = gate(ctx, pre, 3)?;
                    if let Some([pi, _, pf]) = peeps {
                        let t = mul(ctx, pi, cprev)?;
                        ipre = add(ctx, ipre, t)?;
                        let t = mul(ctx, pf, cprev)?;
                        fpre = add(ctx, fpre, t)?;
                    }
                    let ipre = clamp(ctx, ipre)?;
                    let fpre = clamp(ctx, fpre)?;
                    let cpre = clamp(ctx, cpre)?;
                    let ig = activation(ctx, &fa, ipre, fal, fbe)?;
                    let fg = activation(ctx, &fa, fpre, fal, fbe)?;
                    let cg = activation(ctx, &ga, cpre, gal, gbe)?;
                    let fc = mul(ctx, fg, cprev)?;
                    let ic = mul(ctx, ig, cg)?;
                    let cn = add(ctx, fc, ic)?;
                    if let Some([_, po, _]) = peeps {
                        let t = mul(ctx, po, cn)?;
                        opre = add(ctx, opre, t)?;
                    }
                    let opre = clamp(ctx, opre)?;
                    let og = activation(ctx, &fa, opre, fal, fbe)?;
                    let hc = activation(ctx, &ha, cn, hal, hbe)?;
                    (mul(ctx, og, hc)?, Some(cn))
                }
            };
            // Sequence-length masking.
            let (h_next, y_t, c_next) = match mask_t {
                Some(m) => {
                    let mt = slice_axis(ctx, m, 0, t, t + 1)?;
                    let mt = squeeze(ctx, mt, 0)?; // [B, 1]
                    let hn = select(ctx, mt, h_new, h)?;
                    let yt = select(ctx, mt, h_new, zero)?;
                    let cn = match (c_new, cstate) {
                        (Some(cn), Some(cp)) => Some(select(ctx, mt, cn, cp)?),
                        _ => None,
                    };
                    (hn, yt, cn)
                }
                None => (h_new, h_new, c_new),
            };
            h = h_next;
            cstate = c_next.or(cstate);
            outs[t as usize] = Some(y_t);
        }
        let steps: Vec<ValueId> = outs
            .into_iter()
            .map(|o| o.expect("every step visited"))
            .map(|v| unsqueeze(ctx, v, 0))
            .collect::<Result<_>>()?;
        let y_dir = concat(ctx, &steps, 0)?; // [T, B, H]
        ys.push(unsqueeze(ctx, y_dir, 1)?); // [T, 1, B, H]
        yhs.push(unsqueeze(ctx, h, 0)?); // [1, B, H]
        if let Some(cs) = cstate {
            ycs.push(unsqueeze(ctx, cs, 0)?);
        }
    }
    let y = concat(ctx, &ys, 1)?; // [T, D, B, H]
    let y_h = concat(ctx, &yhs, 0)?; // [D, B, H]
    let (y, y_h) = if layout == 1 {
        (
            transpose(ctx, y, &[2, 0, 1, 3])?,
            transpose(ctx, y_h, &[1, 0, 2])?,
        )
    } else {
        (y, y_h)
    };
    ctx.set_value_opt(0, y);
    ctx.set_value_opt(1, y_h);
    if cell == Cell::Lstm && ctx.has_output(2) {
        let y_c = concat(ctx, &ycs, 0)?;
        let y_c = if layout == 1 {
            transpose(ctx, y_c, &[1, 0, 2])?
        } else {
            y_c
        };
        ctx.set_value(2, y_c);
    }
    let _: DimExpr = c(0);
    Ok(())
}
