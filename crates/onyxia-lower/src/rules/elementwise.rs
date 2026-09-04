//! Element-wise ONNX ops that are pure arithmetic over the primitive
//! unary/binary/compare/select kinds: activations, rounding, integer and
//! bitwise ops, clipping, and the variadic reducers (Sum/Mean).

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, UnaryOp};
use onyxia_ir::{DataType, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    // Straight unary kinds.
    for (name, op) in [
        ("Round", UnaryOp::Round),
        ("Sign", UnaryOp::Sign),
        ("Tan", UnaryOp::Tan),
        ("Asin", UnaryOp::Asin),
        ("Acos", UnaryOp::Acos),
        ("Atan", UnaryOp::Atan),
        ("Sinh", UnaryOp::Sinh),
        ("Cosh", UnaryOp::Cosh),
        ("Asinh", UnaryOp::Asinh),
        ("Acosh", UnaryOp::Acosh),
        ("Atanh", UnaryOp::Atanh),
        ("BitwiseNot", UnaryOp::BitNot),
    ] {
        // `Rule` is a plain fn pointer, so each op gets its own fn below.
        let rule: crate::Rule = match op {
            UnaryOp::Round => |c| plain_unary(c, UnaryOp::Round),
            UnaryOp::Sign => |c| plain_unary(c, UnaryOp::Sign),
            UnaryOp::Tan => |c| plain_unary(c, UnaryOp::Tan),
            UnaryOp::Asin => |c| plain_unary(c, UnaryOp::Asin),
            UnaryOp::Acos => |c| plain_unary(c, UnaryOp::Acos),
            UnaryOp::Atan => |c| plain_unary(c, UnaryOp::Atan),
            UnaryOp::Sinh => |c| plain_unary(c, UnaryOp::Sinh),
            UnaryOp::Cosh => |c| plain_unary(c, UnaryOp::Cosh),
            UnaryOp::Asinh => |c| plain_unary(c, UnaryOp::Asinh),
            UnaryOp::Acosh => |c| plain_unary(c, UnaryOp::Acosh),
            UnaryOp::Atanh => |c| plain_unary(c, UnaryOp::Atanh),
            UnaryOp::BitNot => |c| plain_unary(c, UnaryOp::BitNot),
            _ => unreachable!(),
        };
        r.register("", name, rule);
    }
    r.register("", "BitwiseAnd", |c| plain_binary(c, BinaryOp::BitAnd));
    r.register("", "BitwiseOr", |c| plain_binary(c, BinaryOp::BitOr));
    r.register("", "BitwiseXor", |c| plain_binary(c, BinaryOp::BitXor));
    r.register("", "BitShift", bit_shift);

    r.register("", "Reciprocal", reciprocal);
    r.register("", "Sigmoid", sigmoid_rule);
    r.register("", "Relu", relu);
    r.register("", "LeakyRelu", leaky_relu);
    r.register("", "PRelu", prelu);
    r.register("", "Elu", elu);
    r.register("", "Selu", selu);
    r.register("", "Celu", celu);
    r.register("", "ThresholdedRelu", thresholded_relu);
    r.register("", "HardSigmoid", hard_sigmoid);
    r.register("", "HardSwish", hard_swish);
    r.register("", "Softplus", softplus_rule);
    r.register("", "Softsign", softsign);
    r.register("", "Mish", mish);
    r.register("", "Swish", swish);
    r.register("", "Shrink", shrink);
    r.register("", "Clip", clip);
    r.register("", "IsNaN", is_nan);
    r.register("", "IsInf", is_inf);
    r.register("", "Mod", mod_);
    r.register("", "Sum", |c| variadic_fold(c, BinaryOp::Add, false));
    r.register("", "Mean", |c| variadic_fold(c, BinaryOp::Add, true));
    r.register("", "CastLike", cast_like);
    r.register("", "LogSoftmax", log_softmax);
    r.register("", "Hardmax", hardmax);
    r.register("", "Dropout", dropout);
}

fn plain_unary(ctx: &mut LowerCtx, op: UnaryOp) -> Result<()> {
    let x = val(ctx, 0)?;
    let y = unary(ctx, op, x)?;
    out(ctx, y)
}

fn plain_binary(ctx: &mut LowerCtx, op: BinaryOp) -> Result<()> {
    let (a, b) = (val(ctx, 0)?, val(ctx, 1)?);
    let y = binary(ctx, op, a, b)?;
    out(ctx, y)
}

fn bit_shift(ctx: &mut LowerCtx) -> Result<()> {
    let op = match ctx.attr_s("direction") {
        Some("LEFT") => BinaryOp::Shl,
        Some("RIGHT") => BinaryOp::Shr,
        other => {
            return Err(Error::Attribute(format!(
                "BitShift direction must be LEFT or RIGHT, got {other:?}"
            )));
        }
    };
    plain_binary(ctx, op)
}

fn reciprocal(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let one = scalar(ctx, dtype(ctx, x), 1.0)?;
    let y = div(ctx, one, x)?;
    out(ctx, y)
}

fn sigmoid_rule(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let y = sigmoid(ctx, x)?;
    out(ctx, y)
}

fn relu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let zero = scalar(ctx, dtype(ctx, x), 0.0)?;
    let y = max(ctx, x, zero)?;
    out(ctx, y)
}

/// `select(x < 0, alpha * x, x)`.
fn leaky_relu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(0.01) as f64;
    let dt = dtype(ctx, x);
    let a = scalar(ctx, dt, alpha)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let neg = cmp(ctx, CmpOp::Lt, x, zero)?;
    let ax = mul(ctx, x, a)?;
    let y = select(ctx, neg, ax, x)?;
    out(ctx, y)
}

fn prelu(ctx: &mut LowerCtx) -> Result<()> {
    let (x, slope) = (val(ctx, 0)?, val(ctx, 1)?);
    let zero = scalar(ctx, dtype(ctx, x), 0.0)?;
    let neg = cmp(ctx, CmpOp::Lt, x, zero)?;
    let sx = mul(ctx, x, slope)?;
    let y = select(ctx, neg, sx, x)?;
    out(ctx, y)
}

/// `select(x < 0, alpha * (exp(x) - 1), x)`.
fn elu_core(ctx: &mut LowerCtx, x: ValueId, alpha: f64) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let a = scalar(ctx, dt, alpha)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let neg = cmp(ctx, CmpOp::Lt, x, zero)?;
    let e = unary(ctx, UnaryOp::Exp, x)?;
    let em1 = sub(ctx, e, one)?;
    let aem1 = mul(ctx, a, em1)?;
    select(ctx, neg, aem1, x)
}

fn elu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1.0) as f64;
    let y = elu_core(ctx, x, alpha)?;
    out(ctx, y)
}

fn selu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1.673_263_242_354_377_2) as f64;
    let gamma = ctx.attr_f("gamma").unwrap_or(1.050_700_987_355_480_5) as f64;
    let e = elu_core(ctx, x, alpha)?;
    let g = scalar(ctx, dtype(ctx, x), gamma)?;
    let y = mul(ctx, g, e)?;
    out(ctx, y)
}

/// `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`.
fn celu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1.0) as f64;
    let dt = dtype(ctx, x);
    let a = scalar(ctx, dt, alpha)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let xa = div(ctx, x, a)?;
    let e = unary(ctx, UnaryOp::Exp, xa)?;
    let em1 = sub(ctx, e, one)?;
    let aem1 = mul(ctx, a, em1)?;
    let neg = min(ctx, zero, aem1)?;
    let pos = max(ctx, zero, x)?;
    let y = add(ctx, pos, neg)?;
    out(ctx, y)
}

fn thresholded_relu(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1.0) as f64;
    let dt = dtype(ctx, x);
    let a = scalar(ctx, dt, alpha)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let keep = cmp(ctx, CmpOp::Gt, x, a)?;
    let y = select(ctx, keep, x, zero)?;
    out(ctx, y)
}

/// `max(0, min(1, alpha * x + beta))`.
fn hard_sigmoid_core(ctx: &mut LowerCtx, x: ValueId, alpha: f64, beta: f64) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let a = scalar(ctx, dt, alpha)?;
    let b = scalar(ctx, dt, beta)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let ax = mul(ctx, x, a)?;
    let axb = add(ctx, ax, b)?;
    let m = min(ctx, axb, one)?;
    max(ctx, m, zero)
}

fn hard_sigmoid(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(0.2) as f64;
    let beta = ctx.attr_f("beta").unwrap_or(0.5) as f64;
    let y = hard_sigmoid_core(ctx, x, alpha, beta)?;
    out(ctx, y)
}

fn hard_swish(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let hs = hard_sigmoid_core(ctx, x, 1.0 / 6.0, 0.5)?;
    let y = mul(ctx, x, hs)?;
    out(ctx, y)
}

fn softplus_rule(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let y = softplus(ctx, x)?;
    out(ctx, y)
}

fn softsign(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let one = scalar(ctx, dtype(ctx, x), 1.0)?;
    let ax = unary(ctx, UnaryOp::Abs, x)?;
    let d = add(ctx, one, ax)?;
    let y = div(ctx, x, d)?;
    out(ctx, y)
}

fn mish(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let sp = softplus(ctx, x)?;
    let t = unary(ctx, UnaryOp::Tanh, sp)?;
    let y = mul(ctx, x, t)?;
    out(ctx, y)
}

fn swish(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let alpha = ctx.attr_f("alpha").unwrap_or(1.0) as f64;
    let mut ax = x;
    if alpha != 1.0 {
        let a = scalar(ctx, dtype(ctx, x), alpha)?;
        ax = mul(ctx, x, a)?;
    }
    let s = sigmoid(ctx, ax)?;
    let y = mul(ctx, x, s)?;
    out(ctx, y)
}

/// `x < -lambd → x + bias; x > lambd → x - bias; else 0`.
fn shrink(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let bias = ctx.attr_f("bias").unwrap_or(0.0) as f64;
    let lambd = ctx.attr_f("lambd").unwrap_or(0.5) as f64;
    let dt = dtype(ctx, x);
    let b = scalar(ctx, dt, bias)?;
    let l = scalar(ctx, dt, lambd)?;
    let nl = scalar(ctx, dt, -lambd)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let lo = cmp(ctx, CmpOp::Lt, x, nl)?;
    let hi = cmp(ctx, CmpOp::Gt, x, l)?;
    let xpb = add(ctx, x, b)?;
    let xmb = sub(ctx, x, b)?;
    let upper = select(ctx, hi, xmb, zero)?;
    let y = select(ctx, lo, xpb, upper)?;
    out(ctx, y)
}

/// Clip: bounds as optional inputs (opset 11+) or attributes (older).
fn clip(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let mut y = x;
    let lo = if ctx.has_input(1) {
        Some(val(ctx, 1)?)
    } else if let Some(v) = ctx.attr_f("min") {
        Some(scalar(ctx, dt, v as f64)?)
    } else {
        None
    };
    let hi = if ctx.has_input(2) {
        Some(val(ctx, 2)?)
    } else if let Some(v) = ctx.attr_f("max") {
        Some(scalar(ctx, dt, v as f64)?)
    } else {
        None
    };
    if let Some(lo) = lo {
        y = max(ctx, y, lo)?;
    }
    if let Some(hi) = hi {
        y = min(ctx, y, hi)?;
    }
    out(ctx, y)
}

fn is_nan(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let y = cmp(ctx, CmpOp::Ne, x, x)?;
    out(ctx, y)
}

fn is_inf(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let neg = ctx.attr_i("detect_negative").unwrap_or(1) != 0;
    let pos = ctx.attr_i("detect_positive").unwrap_or(1) != 0;
    let mut acc: Option<ValueId> = None;
    if pos {
        let inf = scalar(ctx, dt, f64::INFINITY)?;
        acc = Some(cmp(ctx, CmpOp::Eq, x, inf)?);
    }
    if neg {
        let ninf = scalar(ctx, dt, f64::NEG_INFINITY)?;
        let n = cmp(ctx, CmpOp::Eq, x, ninf)?;
        acc = Some(match acc {
            Some(p) => binary(ctx, BinaryOp::Or, p, n)?,
            None => n,
        });
    }
    let y = match acc {
        Some(v) => v,
        None => {
            // Neither: all false.
            let f = cmp(ctx, CmpOp::Ne, x, x)?;
            let f2 = cmp(ctx, CmpOp::Eq, x, x)?;
            binary(ctx, BinaryOp::And, f, f2)?
        }
    };
    out(ctx, y)
}

/// Mod: `fmod=1` is C fmod (sign of the dividend); `fmod=0` is Python
/// modulo (sign of the divisor), integer-only per the spec.
fn mod_(ctx: &mut LowerCtx) -> Result<()> {
    let (x, y) = (val(ctx, 0)?, val(ctx, 1)?);
    let dt = dtype(ctx, x);
    let fmod = ctx.attr_i("fmod").unwrap_or(0) != 0;
    // r = x - trunc(x / y) * y
    let q = div(ctx, x, y)?;
    let tq = if dt.is_float() {
        // trunc = select(q < 0, ceil(q), floor(q))
        let zero = scalar(ctx, dt, 0.0)?;
        let neg = cmp(ctx, CmpOp::Lt, q, zero)?;
        let cq = unary(ctx, UnaryOp::Ceil, q)?;
        let fq = unary(ctx, UnaryOp::Floor, q)?;
        select(ctx, neg, cq, fq)?
    } else {
        q // integer division truncates
    };
    let tqy = mul(ctx, tq, y)?;
    let r = sub(ctx, x, tqy)?;
    let res = if fmod {
        r
    } else {
        // Python: if r != 0 and sign(r) != sign(y): r + y
        let zero = scalar(ctx, dt, 0.0)?;
        let nz = cmp(ctx, CmpOp::Ne, r, zero)?;
        let r_neg = cmp(ctx, CmpOp::Lt, r, zero)?;
        let y_neg = cmp(ctx, CmpOp::Lt, y, zero)?;
        let diff = binary(ctx, BinaryOp::Xor, r_neg, y_neg)?;
        let fix = binary(ctx, BinaryOp::And, nz, diff)?;
        let ry = add(ctx, r, y)?;
        select(ctx, fix, ry, r)?
    };
    out(ctx, res)
}

/// Sum / Mean over a variadic input list (broadcasting).
fn variadic_fold(ctx: &mut LowerCtx, op: BinaryOp, mean: bool) -> Result<()> {
    let n = ctx.num_inputs();
    let mut acc = val(ctx, 0)?;
    for i in 1..n {
        let v = val(ctx, i)?;
        acc = binary(ctx, op, acc, v)?;
    }
    if mean && n > 1 {
        let d = scalar(ctx, dtype(ctx, acc), n as f64)?;
        acc = div(ctx, acc, d)?;
    }
    out(ctx, acc)
}

fn cast_like(ctx: &mut LowerCtx) -> Result<()> {
    let to = match ctx.peek(1)? {
        crate::Lowered::Value(v) => ctx.ty(*v).dtype,
        crate::Lowered::Content(_) => DataType::I64,
    };
    let prim = Prim::Cast { to };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let x = val(ctx, 0)?;
    let y = cast(ctx, x, to)?;
    out(ctx, y)
}

fn log_softmax(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let y = log_softmax_axes(ctx, x, &[axis])?;
    out(ctx, y)
}

/// Hardmax: one-hot of the first maximum along `axis`.
fn hardmax(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let r = rank(ctx, x);
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), r)?;
    let dt = dtype(ctx, x);
    let am = arg_reduce(ctx, x, axis, true, true, false)?;
    let d = dims(ctx, x)[axis].clone();
    let idx = iota_along(ctx, d, r, axis, DataType::I64)?;
    let hit = cmp(ctx, CmpOp::Eq, idx, am)?;
    let one = scalar(ctx, dt, 1.0)?;
    let zero = scalar(ctx, dt, 0.0)?;
    let y = select(ctx, hit, one, zero)?;
    out(ctx, y)
}

/// Dropout at inference: identity, with an all-true mask if requested.
fn dropout(ctx: &mut LowerCtx) -> Result<()> {
    if ctx.has_input(2) {
        if let Some(v) = const_floats(ctx, 2) {
            if v.first().copied().unwrap_or(0.0) != 0.0 {
                return Err(Error::Unsupported("Dropout in training mode".into()));
            }
        }
    }
    let x = val(ctx, 0)?;
    ctx.set_value(0, x);
    if ctx.num_outputs() > 1 && ctx.has_output(1) {
        let t = cmp(ctx, CmpOp::Eq, x, x)?; // all true (NaN aside)
        let tt = cmp(ctx, CmpOp::Ne, x, x)?;
        let mask = binary(ctx, BinaryOp::Or, t, tt)?;
        ctx.set_value(1, mask);
    }
    Ok(())
}
