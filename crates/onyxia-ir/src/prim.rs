//! The closed primitive set.
//!
//! These sixteen operations are the entire backend contract: a backend that
//! executes them can run any model, because every composite is required to
//! decompose into them (see [`crate::decomp`]). The enum is deliberately
//! closed — passes match on it exhaustively, and adding a variant makes the
//! compiler point at every pass that needs updating.
//!
//! Parameters live inside the variants. Anything that was an ONNX runtime
//! *input* but is semantically a parameter (Reshape targets, Slice bounds)
//! is resolved to [`DimExpr`]s during lowering — by then, shape arithmetic
//! has been evaluated symbolically (see [`crate::fold`]).
//!
//! Conventions shared by all primitives:
//! - **Broadcasting** ([`Binary`](Prim::Binary), [`Compare`](Prim::Compare),
//!   [`Select`](Prim::Select)) follows ONNX/numpy multidirectional rules.
//! - **No implicit dtype promotion.** Inputs must have identical dtypes;
//!   lowering inserts explicit [`Cast`](Prim::Cast)s.
//! - **Axes are normalized**: non-negative, in-range, resolved from any
//!   negative ONNX form during lowering.

use crate::dim::DimExpr;
use crate::types::DataType;

/// Element-wise unary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Arithmetic negation. Integer and float.
    Neg,
    /// Absolute value. Integer and float.
    Abs,
    /// Square root. Float only.
    Sqrt,
    /// Reciprocal square root. Float only.
    Rsqrt,
    /// Natural exponent. Float only.
    Exp,
    /// Natural logarithm. Float only.
    Log,
    /// Sine. Float only.
    Sin,
    /// Cosine. Float only.
    Cos,
    /// Hyperbolic tangent. Float only.
    Tanh,
    /// Gauss error function. Float only.
    Erf,
    /// Round toward negative infinity. Float only.
    Floor,
    /// Round toward positive infinity. Float only.
    Ceil,
    /// Logical not. Bool only.
    Not,
}

/// Element-wise binary operation kind (dtype-preserving).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Addition. Integer and float.
    Add,
    /// Subtraction. Integer and float.
    Sub,
    /// Multiplication. Integer and float.
    Mul,
    /// Division. Integer division truncates toward zero and errors on zero;
    /// float division follows IEEE 754.
    Div,
    /// Exponentiation. Float base; integer base requires non-negative
    /// exponent.
    Pow,
    /// Element-wise maximum. NaN handling follows Rust `f32::max`
    /// (non-NaN operand wins).
    Max,
    /// Element-wise minimum. NaN handling follows Rust `f32::min`.
    Min,
    /// Logical and. Bool only.
    And,
    /// Logical or. Bool only.
    Or,
    /// Logical xor. Bool only.
    Xor,
}

/// Element-wise comparison kind (produces [`DataType::Bool`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    /// Arithmetic mean. A first-class kind (rather than Sum ÷ count) so
    /// backends keep numerical-stability latitude.
    Mean,
    Max,
    Min,
    Prod,
}

/// Per-axis slice specification. Semantics are a half-open range
/// `[start, end)` traversed with `step`.
///
/// Bounds are already normalized by lowering: non-negative, clamped, with
/// ONNX sentinel values (negative indices, `INT64_MAX`) resolved. For a
/// negative `step`, traversal runs `start, start+step, …` while the index
/// stays `> end` (so `end` is exclusive in the direction of travel).
///
/// Symbolic `start`/`end` are supported for `step == ±1`; other steps
/// require constant bounds (checked by shape inference — the output length
/// `ceil((end-start)/step)` is only polynomial when `|step| == 1`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SliceSpec {
    /// Axis to slice (normalized).
    pub axis: usize,
    /// Inclusive start, in the direction of travel.
    pub start: DimExpr,
    /// Exclusive end, in the direction of travel.
    pub end: DimExpr,
    /// Step; never zero.
    pub step: i64,
}

/// A primitive operation. See the module docs for shared conventions.
///
/// Note there is no `Constant` primitive: constants are value *origins*
/// ([`Origin::Const`](crate::graph::Origin)), not nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Prim {
    /// Element-wise unary op. `(x: T) -> T`.
    Unary(UnaryOp),
    /// Element-wise binary op with broadcasting. `(a: T, b: T) -> T`.
    Binary(BinaryOp),
    /// Element-wise comparison with broadcasting. `(a: T, b: T) -> Bool`.
    Compare(CmpOp),
    /// Element-wise selection with three-way broadcasting.
    /// `(cond: Bool, a: T, b: T) -> T`.
    Select,
    /// Element type conversion. `(x) -> to`. Overflowing conversions follow
    /// Rust `as` semantics (float→int saturates).
    Cast {
        /// Target dtype.
        to: DataType,
    },
    /// Batched matrix multiplication.
    /// `(a: T[..., M, K], b: T[..., K, N]) -> T[..., M, N]`, leading batch
    /// dims broadcast. With `trans_a`/`trans_b` the corresponding operand's
    /// trailing two dims are read transposed.
    MatMul { trans_a: bool, trans_b: bool },
    /// Reduction over `axes` (normalized, sorted, non-empty; reduce-over-all
    /// lists every axis explicitly).
    Reduce {
        op: ReduceOp,
        axes: Vec<usize>,
        keepdims: bool,
    },
    /// Reshape to `shape`. Value-semantic (backends may implement as a
    /// view). Element count must be preserved; inferred (`-1`) dims are
    /// already resolved by lowering via exact division.
    Reshape { shape: Vec<DimExpr> },
    /// Permute dimensions. `perm` is a full permutation of the input rank.
    Transpose { perm: Vec<usize> },
    /// Broadcast to `shape`, following ONNX `Expand`: the input shape and
    /// the target broadcast *together* (either side may stretch dims of
    /// size 1).
    Broadcast { shape: Vec<DimExpr> },
    /// Concatenate all inputs (the one variadic primitive) along `axis`.
    /// Inputs share dtype, rank, and every dim except `axis`.
    Concat { axis: usize },
    /// Slice per [`SliceSpec`]; axes not listed pass through whole.
    Slice { specs: Vec<SliceSpec> },
    /// Gather slices of `data` along `axis` by integer `indices`
    /// (ONNX `Gather`). `(data, indices: I64) -> out` where the output rank
    /// is `rank(data) - 1 + rank(indices)`. Negative indices count from the
    /// end.
    Gather { axis: usize },
    /// Scatter updates into a copy of `data` (ONNX `ScatterND`).
    /// `(data, indices: I64[..., k], updates) -> T[data.shape]`.
    Scatter,
    /// The integer ramp `0, 1, …, len-1` as `dtype`. `() -> dtype[len]`.
    /// ONNX `Range(start, limit, delta)` lowers to `Iota` plus elementwise
    /// arithmetic.
    Iota { len: DimExpr, dtype: DataType },
    /// The bound values of shape-domain expressions, materialized as a
    /// 1-D i64 tensor. `() -> I64[exprs.len()]`. This is the escape
    /// hatch for a symbolic shape value consumed by a genuine runtime
    /// tensor operation (e.g. GQA's `seqlens_k` computed via `Shape`
    /// arithmetic): the expressions are evaluated against the run's dim
    /// bindings.
    DimValues { exprs: Vec<DimExpr> },
    /// Block-dequantize packed sub-byte data.
    /// `(data: U4|I4|U8 [..., n_blocks, block_size],
    ///   scales: F [..., n_blocks],
    ///   [zero_points: same dtype as data, [..., n_blocks]])
    ///  -> F[out_shape]`.
    ///
    /// `data`'s shape is in **logical elements** (packing is a storage
    /// detail of the dtype). Each block of `block_size` elements shares one
    /// scale (and zero point). `value = (q - zero_point) * scale`; the
    /// default zero point is `2^(bits-1)` for unsigned dtypes and 0 for
    /// signed. `out_shape` must have the same element count as `data`
    /// (typically `[..., n_blocks*block_size]`). Output dtype is the
    /// scales' dtype.
    Dequantize {
        block_size: usize,
        out_shape: Vec<DimExpr>,
    },
}

impl Prim {
    /// A short lowercase name for labels, errors, and dot output.
    pub fn name(&self) -> &'static str {
        match self {
            Prim::Unary(op) => match op {
                UnaryOp::Neg => "neg",
                UnaryOp::Abs => "abs",
                UnaryOp::Sqrt => "sqrt",
                UnaryOp::Rsqrt => "rsqrt",
                UnaryOp::Exp => "exp",
                UnaryOp::Log => "log",
                UnaryOp::Sin => "sin",
                UnaryOp::Cos => "cos",
                UnaryOp::Tanh => "tanh",
                UnaryOp::Erf => "erf",
                UnaryOp::Floor => "floor",
                UnaryOp::Ceil => "ceil",
                UnaryOp::Not => "not",
            },
            Prim::Binary(op) => match op {
                BinaryOp::Add => "add",
                BinaryOp::Sub => "sub",
                BinaryOp::Mul => "mul",
                BinaryOp::Div => "div",
                BinaryOp::Pow => "pow",
                BinaryOp::Max => "max",
                BinaryOp::Min => "min",
                BinaryOp::And => "and",
                BinaryOp::Or => "or",
                BinaryOp::Xor => "xor",
            },
            Prim::Compare(op) => match op {
                CmpOp::Eq => "eq",
                CmpOp::Ne => "ne",
                CmpOp::Lt => "lt",
                CmpOp::Le => "le",
                CmpOp::Gt => "gt",
                CmpOp::Ge => "ge",
            },
            Prim::Select => "select",
            Prim::Cast { .. } => "cast",
            Prim::MatMul { .. } => "matmul",
            Prim::Reduce { op, .. } => match op {
                ReduceOp::Sum => "reduce_sum",
                ReduceOp::Mean => "reduce_mean",
                ReduceOp::Max => "reduce_max",
                ReduceOp::Min => "reduce_min",
                ReduceOp::Prod => "reduce_prod",
            },
            Prim::Reshape { .. } => "reshape",
            Prim::Transpose { .. } => "transpose",
            Prim::Broadcast { .. } => "broadcast",
            Prim::Concat { .. } => "concat",
            Prim::Slice { .. } => "slice",
            Prim::Gather { .. } => "gather",
            Prim::Scatter => "scatter",
            Prim::Iota { .. } => "iota",
            Prim::DimValues { .. } => "dim_values",
            Prim::Dequantize { .. } => "dequantize",
        }
    }

    /// Expected input arity: `(min, max)`; `max == usize::MAX` for the
    /// variadic [`Concat`](Prim::Concat).
    pub fn arity(&self) -> (usize, usize) {
        match self {
            Prim::Unary(_) | Prim::Cast { .. } | Prim::Reduce { .. } => (1, 1),
            Prim::Reshape { .. }
            | Prim::Transpose { .. }
            | Prim::Broadcast { .. }
            | Prim::Slice { .. } => (1, 1),
            Prim::Binary(_) | Prim::Compare(_) | Prim::MatMul { .. } | Prim::Gather { .. } => {
                (2, 2)
            }
            Prim::Select | Prim::Scatter => (3, 3),
            Prim::Iota { .. } | Prim::DimValues { .. } => (0, 0),
            Prim::Dequantize { .. } => (2, 3),
            Prim::Concat { .. } => (1, usize::MAX),
        }
    }
}
