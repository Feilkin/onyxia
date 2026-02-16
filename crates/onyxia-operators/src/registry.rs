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

    // Binary elementwise operators
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Sub", BinaryElementwiseOp::sub());
    registry.register("Mul", BinaryElementwiseOp::mul());
    registry.register("Div", BinaryElementwiseOp::div());
    registry.register("Pow", BinaryElementwiseOp::pow());
    registry.register("Max", BinaryElementwiseOp::max());

    // Unary elementwise operators
    registry.register("Cos", UnaryElementwiseOp::cos());
    registry.register("Sin", UnaryElementwiseOp::sin());
    registry.register("Sqrt", UnaryElementwiseOp::sqrt());
    registry.register("Neg", UnaryElementwiseOp::neg());
    registry.register("Tanh", UnaryElementwiseOp::tanh());

    // Comparison operators
    registry.register("Equal", ComparisonOp::equal());
    registry.register("Greater", ComparisonOp::greater());

    // Reduction operators
    registry.register("ReduceSum", ReductionOp::reduce_sum());
    registry.register("ReduceMean", ReductionOp::reduce_mean());

    // Activation operators
    registry.register("Gelu", GeluOp);
    registry.register("Softmax", SoftmaxOp);

    // Normalization operators
    registry.register("SimplifiedLayerNormalization", RmsNormOp);

    // Matrix multiplication operators
    registry.register("MatMul", MatMulF32Op);

    // Metadata operators
    registry.register("Constant", ConstantOp);
    registry.register("ConstantOfShape", ConstantOfShapeOp);
    registry.register("Shape", ShapeOp);

    // Shape manipulation operators
    registry.register("Reshape", ReshapeOp);
    registry.register("Unsqueeze", UnsqueezeOp);
    registry.register("Transpose", TransposeOp);
    registry.register("Concat", ConcatOp);
    registry.register("Expand", ExpandOp);

    // Indexing operators
    registry.register("Gather", GatherOp);
    registry.register("Slice", SliceOp);
    registry.register("ScatterND", ScatterNDOp);
    registry.register("Range", RangeOp);
    registry.register("Trilu", TriluOp);

    // Type conversion operators
    registry.register("Cast", CastOp);

    // Conditional operators
    registry.register("Where", WhereOp);

    // Attention operators
    registry.register("RotaryEmbedding", RotaryEmbeddingOp);

    // com.microsoft operators
    registry.register("com.microsoft.RotaryEmbedding", MicrosoftRotaryEmbeddingOp);
    registry.register("com.microsoft.GemmaRotaryEmbedding", GemmaRotaryEmbeddingOp);
    registry.register("com.microsoft.GroupQueryAttention", GroupQueryAttentionOp);
    registry.register("com.microsoft.MatMulNBits", MatMulNBitsOp);

    registry
}
