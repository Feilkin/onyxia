//! Loss functions: NegativeLogLikelihoodLoss and SoftmaxCrossEntropyLoss,
//! both a gather of the target class plus optional weighting/reduction.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry};
use onyxia_ir::prim::{CmpOp, ReduceOp};
use onyxia_ir::{DataType, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "NegativeLogLikelihoodLoss", nll);
    r.register("", "SoftmaxCrossEntropyLoss", sce);
}

/// Shared tail: `logp [N, C, d...]`, `target [N, d...]`.
fn nll_core(
    ctx: &mut LowerCtx,
    logp: ValueId,
    target: ValueId,
    weight: Option<ValueId>,
) -> Result<ValueId> {
    let dt = dtype(ctx, logp);
    let ld = dims(ctx, logp);
    let (n, cdim) = (ld[0].clone(), ld[1].clone());
    let dtail = prod(&ld[2..]);
    let x3 = reshape(ctx, logp, vec![n.clone(), cdim.clone(), dtail.clone()])?;
    let target = cast(ctx, target, DataType::I64)?;
    let td = dims(ctx, target);
    let t2 = reshape(ctx, target, vec![n.clone(), dtail.clone()])?;
    let ignore = ctx.attr_i("ignore_index");
    let reduction = ctx.attr_s("reduction").unwrap_or("mean").to_string();
    // Mask of non-ignored samples.
    let keep = match ignore {
        Some(ig) => {
            let igc = scalar(ctx, DataType::I64, ig as f64)?;
            Some(cmp(ctx, CmpOp::Ne, t2, igc)?)
        }
        None => None,
    };
    // Clamp targets for the gather (ignored ones may be out of range).
    let zero_i = scalar(ctx, DataType::I64, 0.0)?;
    let cm1 = dim_value(ctx, cdim.clone() - c(1))?;
    let tc = max(ctx, t2, zero_i)?;
    let tc = min(ctx, tc, cm1)?;
    // loss[n, d] = -x3[n, tc[n,d], d]
    let t3 = unsqueeze(ctx, tc, 1)?; // [N, 1, D]
    let lin = linear_index_with_axis(ctx, &dims(ctx, x3), t3, 1)?;
    let picked = linear_gather(ctx, x3, lin)?; // [N, 1, D]
    let picked = reshape(ctx, picked, vec![n.clone(), dtail.clone()])?;
    let mut loss = unary(ctx, onyxia_ir::UnaryOp::Neg, picked)?;
    // Per-sample weights: weight[target] (0 where ignored), or 1/0.
    let w: Option<ValueId> = match weight {
        Some(w) => {
            let gw = ctx.emit(onyxia_ir::Prim::Gather { axis: 0 }, &[w, tc])?; // [N, D]
            Some(match keep {
                Some(k) => {
                    let zero = scalar(ctx, dt, 0.0)?;
                    select(ctx, k, gw, zero)?
                }
                None => gw,
            })
        }
        None => match keep {
            Some(k) => {
                let one = scalar(ctx, dt, 1.0)?;
                let zero = scalar(ctx, dt, 0.0)?;
                Some(select(ctx, k, one, zero)?)
            }
            None => None,
        },
    };
    if let Some(w) = w {
        loss = mul(ctx, loss, w)?;
        if reduction == "mean" {
            let all: Vec<usize> = vec![0, 1];
            let ls = reduce(ctx, ReduceOp::Sum, loss, &all, false)?;
            let ws = reduce(ctx, ReduceOp::Sum, w, &all, false)?;
            return div(ctx, ls, ws);
        }
    }
    match reduction.as_str() {
        "none" => reshape(ctx, loss, td),
        "sum" => reduce(ctx, ReduceOp::Sum, loss, &[0, 1], false),
        _ => reduce(ctx, ReduceOp::Mean, loss, &[0, 1], false),
    }
}

fn nll(ctx: &mut LowerCtx) -> Result<()> {
    let (x, target) = (val(ctx, 0)?, val(ctx, 1)?);
    let weight = opt_val(ctx, 2)?;
    let y = nll_core(ctx, x, target, weight)?;
    out(ctx, y)
}

fn sce(ctx: &mut LowerCtx) -> Result<()> {
    let (scores, target) = (val(ctx, 0)?, val(ctx, 1)?);
    let weight = opt_val(ctx, 2)?;
    let logp = log_softmax_axes(ctx, scores, &[1])?;
    let y = nll_core(ctx, logp, target, weight)?;
    ctx.set_value(0, y);
    ctx.set_value_opt(1, logp);
    Ok(())
}
