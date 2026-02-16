//! Core operator registry.
//!
//! Provides a pre-populated registry with all 39 core operators.

use onyxia_core::OperatorRegistry;

use crate::families::{BinaryElementwiseOp, ComparisonOp, ReductionOp, UnaryElementwiseOp};
use crate::operators::{
    CastOp, ConcatOp, ConstantOfShapeOp, ConstantOp, ExpandOp, GatherOp, GeluOp,
    GemmaRotaryEmbeddingOp, GroupQueryAttentionOp, MatMulF32Op, MatMulNBitsOp,
    MicrosoftRotaryEmbeddingOp, RangeOp, ReshapeOp, RmsNormOp, RotaryEmbeddingOp, ScatterNDOp,
    ShapeOp, SliceOp, SoftmaxOp, TransposeOp, TriluOp, UnsqueezeOp, WhereOp,
};

/// Returns an operator registry pre-populated with all 40 core operators.
///
/// The registry includes:
/// - 6 binary elementwise operators (Add, Sub, Mul, Div, Pow, Max)
/// - 5 unary elementwise operators (Cos, Sin, Sqrt, Neg, Tanh)
/// - 2 comparison operators (Equal, Greater)
/// - 2 reduction operators (ReduceSum, ReduceMean)
/// - 23 individual operators (activation, normalization, matrix ops, shape ops, etc.)
/// - 2 custom Microsoft operators (com.microsoft.RotaryEmbedding, com.microsoft.GemmaRotaryEmbedding)
///
/// Custom operators can be added to the returned registry via
/// `registry.register(name, operator)`.
pub fn core_operator_registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();

    // Binary elementwise operators (6)
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Sub", BinaryElementwiseOp::sub());
    registry.register("Mul", BinaryElementwiseOp::mul());
    registry.register("Div", BinaryElementwiseOp::div());
    registry.register("Pow", BinaryElementwiseOp::pow());
    registry.register("Max", BinaryElementwiseOp::max());

    // Unary elementwise operators (5)
    registry.register("Cos", UnaryElementwiseOp::cos());
    registry.register("Sin", UnaryElementwiseOp::sin());
    registry.register("Sqrt", UnaryElementwiseOp::sqrt());
    registry.register("Neg", UnaryElementwiseOp::neg());
    registry.register("Tanh", UnaryElementwiseOp::tanh());

    // Comparison operators (2)
    registry.register("Equal", ComparisonOp::equal());
    registry.register("Greater", ComparisonOp::greater());

    // Reduction operators (2)
    registry.register("ReduceSum", ReductionOp::reduce_sum());
    registry.register("ReduceMean", ReductionOp::reduce_mean());

    // Activation operators (2)
    registry.register("Gelu", GeluOp);
    registry.register("Softmax", SoftmaxOp);

    // Normalization operators (1)
    registry.register("SimplifiedLayerNormalization", RmsNormOp);

    // Matrix multiplication operators (2)
    registry.register("MatMul", MatMulF32Op);
    registry.register("MatMulNBits", MatMulNBitsOp);

    // Metadata operators (3)
    registry.register("Constant", ConstantOp);
    registry.register("ConstantOfShape", ConstantOfShapeOp);
    registry.register("Shape", ShapeOp);

    // Shape manipulation operators (5)
    registry.register("Reshape", ReshapeOp);
    registry.register("Unsqueeze", UnsqueezeOp);
    registry.register("Transpose", TransposeOp);
    registry.register("Concat", ConcatOp);
    registry.register("Expand", ExpandOp);

    // Indexing operators (5)
    registry.register("Gather", GatherOp);
    registry.register("Slice", SliceOp);
    registry.register("ScatterND", ScatterNDOp);
    registry.register("Range", RangeOp);
    registry.register("Trilu", TriluOp);

    // Type conversion operators (1)
    registry.register("Cast", CastOp);

    // Conditional operators (1)
    registry.register("Where", WhereOp);

    // Attention operators (2)
    registry.register("RotaryEmbedding", RotaryEmbeddingOp);
    registry.register("com.microsoft.GroupQueryAttention", GroupQueryAttentionOp);

    // Microsoft custom operators (2)
    registry.register("com.microsoft.RotaryEmbedding", MicrosoftRotaryEmbeddingOp);
    registry.register("com.microsoft.GemmaRotaryEmbedding", GemmaRotaryEmbeddingOp);

    registry
}
