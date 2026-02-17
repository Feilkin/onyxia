//! Core operator registry.
//!
//! Provides a pre-populated registry with core operators.

use onyxia_core::OperatorRegistry;

use crate::families::{BinaryElementwiseOp, ComparisonOp, MaxOp, UnaryMathOp};
use crate::operators::{
    CastOp, ConcatOp, ConstantOfShapeOp, ConstantOp, ExpandOp, GatherOp, GroupQueryAttentionOp,
    MatMulOp, RangeOp, ReduceMeanOp, ReduceSumOp, ReshapeOp, RotaryEmbeddingOp, ScatterNDOp,
    ShapeOp, SimplifiedLayerNormOp, SliceOp, SoftmaxOp, TransposeOp, TriluOp, UnsqueezeOp, WhereOp,
};

/// Returns an operator registry pre-populated with core operators.
///
/// The registry includes:
/// - 5 binary elementwise operators (Add, Mul, Div, Sub, Pow)
/// - 5 comparison operators (Equal, Greater, Less, GreaterOrEqual, LessOrEqual)
/// - 6 unary math operators (Neg, Sqrt, Cos, Sin, Tanh, Gelu)
/// - 8 shape manipulation operators (Reshape, Concat, Expand, Transpose, Unsqueeze, Slice, Shape, ConstantOfShape)
/// - 2 indexing operators (Gather, ScatterND)
/// - 1 matrix operator (MatMul)
/// - 1 element-wise variadic operator (Max)
/// - 2 reduction operators (ReduceMean, ReduceSum)
/// - 1 activation operator (Softmax)
/// - 1 type conversion operator (Cast)
/// - 3 special operators (Range, Trilu, Where)
/// - 3 Microsoft contrib operators (SimplifiedLayerNormalization, RotaryEmbedding, GroupQueryAttention)
///
/// Custom operators can be added to the returned registry via
/// `registry.register(name, operator)`.
pub fn core_operator_registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();

    // Binary elementwise operators
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Mul", BinaryElementwiseOp::mul());
    registry.register("Div", BinaryElementwiseOp::div());
    registry.register("Sub", BinaryElementwiseOp::sub());
    registry.register("Pow", BinaryElementwiseOp::pow());

    // Comparison operators
    registry.register("Equal", ComparisonOp::equal());
    registry.register("Greater", ComparisonOp::greater());
    registry.register("Less", ComparisonOp::less());
    registry.register("GreaterOrEqual", ComparisonOp::greater_or_equal());
    registry.register("LessOrEqual", ComparisonOp::less_or_equal());

    // Unary math operators
    registry.register("Neg", UnaryMathOp::neg());
    registry.register("Sqrt", UnaryMathOp::sqrt());
    registry.register("Cos", UnaryMathOp::cos());
    registry.register("Sin", UnaryMathOp::sin());
    registry.register("Tanh", UnaryMathOp::tanh());
    registry.register("Gelu", UnaryMathOp::gelu());

    // Shape manipulation operators
    registry.register("Reshape", ReshapeOp);
    registry.register("Concat", ConcatOp);
    registry.register("Expand", ExpandOp);
    registry.register("Transpose", TransposeOp);
    registry.register("Unsqueeze", UnsqueezeOp);
    registry.register("Slice", SliceOp);
    registry.register("Shape", ShapeOp);
    registry.register("ConstantOfShape", ConstantOfShapeOp::new(0.0));

    // Constant operator
    registry.register("Constant", ConstantOp);

    // Indexing operators
    registry.register("Gather", GatherOp);
    registry.register("ScatterND", ScatterNDOp);

    // Matrix operators
    registry.register("MatMul", MatMulOp);

    // Element-wise variadic operators
    registry.register("Max", MaxOp);

    // Reduction operators
    registry.register("ReduceMean", ReduceMeanOp);
    registry.register("ReduceSum", ReduceSumOp);

    // Activation operators
    registry.register("Softmax", SoftmaxOp);

    // Type conversion operators
    registry.register("Cast", CastOp);

    // Special operators
    registry.register("Range", RangeOp);
    registry.register("Trilu", TriluOp);
    registry.register("Where", WhereOp);

    // Microsoft contrib operators
    registry.register(
        "com.microsoft::SimplifiedLayerNormalization",
        SimplifiedLayerNormOp,
    );
    registry.register("com.microsoft::RotaryEmbedding", RotaryEmbeddingOp);
    registry.register("com.microsoft::GemmaRotaryEmbedding", RotaryEmbeddingOp); // Alias for Gemma models
    registry.register("com.microsoft::GroupQueryAttention", GroupQueryAttentionOp);

    registry
}
