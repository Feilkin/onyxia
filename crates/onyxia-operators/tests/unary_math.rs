//! Tests for unary math operators (Neg, Sqrt, Cos, Sin, Tanh).
//!
//! These tests require a GPU. Run with:
//! ```sh
//! cargo nextest run -p onyxia-operators unary_math --run-ignored=all
//! ```

mod common;

use common::{CompilerPipeline, Runtime, Tensor, make_unary_graph};
use onyxia_onnx::DataType;
use onyxia_operators::core_operator_registry;
use std::f32::consts::PI;

/// Helper to assert that two f32 values are approximately equal.
fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
    let diff = (a - b).abs();
    assert!(
        diff < epsilon,
        "Values not approximately equal: {} vs {} (diff: {})",
        a,
        b,
        diff
    );
}

/// Helper to assert that two f32 vectors are approximately equal element-wise.
fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Vector lengths differ");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < epsilon,
            "Element {} differs: {} vs {} (diff: {})",
            i,
            a,
            e,
            diff
        );
    }
}

// ================================================================================
// Neg operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_neg_basic() {
    // Test negation: [-1.0, 0.0, 1.0, 2.5] -> [1.0, 0.0, -1.0, -2.5]
    let graph = make_unary_graph("Neg", "neg_op", DataType::F32, &[4], &[4]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    // Execute
    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.5], &[4]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![1.0, 0.0, -1.0, -2.5]);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_neg_multidimensional() {
    // Test negation on 2D tensor [2, 3]
    let graph = make_unary_graph("Neg", "neg_op", DataType::F32, &[2, 3], &[2, 3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, -4.0, -5.0, -6.0], &[2, 3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    assert_eq!(result, vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]);
}

// ================================================================================
// Sqrt operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_sqrt_basic() {
    // Test sqrt: [0.0, 1.0, 4.0, 9.0, 16.0] -> [0.0, 1.0, 2.0, 3.0, 4.0]
    let graph = make_unary_graph("Sqrt", "sqrt_op", DataType::F32, &[5], &[5]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![0.0f32, 1.0, 4.0, 9.0, 16.0], &[5]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_sqrt_negative() {
    // Test sqrt of negative numbers produces NaN
    let graph = make_unary_graph("Sqrt", "sqrt_op", DataType::F32, &[2], &[2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![-1.0f32, -4.0], &[2]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // Both results should be NaN
    assert!(result[0].is_nan(), "sqrt(-1) should be NaN");
    assert!(result[1].is_nan(), "sqrt(-4) should be NaN");
}

// ================================================================================
// Cos operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_cos_basic() {
    // Test cos at common angles: [0, π/4, π/2, π]
    // Expected: [1.0, ~0.707, ~0.0, -1.0]
    let graph = make_unary_graph("Cos", "cos_op", DataType::F32, &[4], &[4]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![0.0f32, PI / 4.0, PI / 2.0, PI], &[4]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![1.0, 0.707106781, 0.0, -1.0];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_cos_full_circle() {
    // Test cos at 0, π/2, π, 3π/2, 2π
    // Expected: [1, 0, -1, 0, 1]
    let graph = make_unary_graph("Cos", "cos_op", DataType::F32, &[5], &[5]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![0.0f32, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI], &[5]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![1.0, 0.0, -1.0, 0.0, 1.0];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

// ================================================================================
// Sin operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_sin_basic() {
    // Test sin at common angles: [0, π/4, π/2, π]
    // Expected: [0.0, ~0.707, 1.0, ~0.0]
    let graph = make_unary_graph("Sin", "sin_op", DataType::F32, &[4], &[4]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![0.0f32, PI / 4.0, PI / 2.0, PI], &[4]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![0.0, 0.707106781, 1.0, 0.0];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_sin_full_circle() {
    // Test sin at 0, π/2, π, 3π/2, 2π
    // Expected: [0, 1, 0, -1, 0]
    let graph = make_unary_graph("Sin", "sin_op", DataType::F32, &[5], &[5]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![0.0f32, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI], &[5]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![0.0, 1.0, 0.0, -1.0, 0.0];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

// ================================================================================
// Tanh operator tests
// ================================================================================

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_tanh_basic() {
    // Test tanh at [-1, 0, 1]
    // tanh(-1) ≈ -0.7616, tanh(0) = 0, tanh(1) ≈ 0.7616
    let graph = make_unary_graph("Tanh", "tanh_op", DataType::F32, &[3], &[3]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], &[3]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![-0.7615941559, 0.0, 0.7615941559];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_tanh_saturation() {
    // Test tanh saturation at extreme values
    // tanh(±∞) → ±1, but for large finite values tanh(±100) ≈ ±1
    let graph = make_unary_graph("Tanh", "tanh_op", DataType::F32, &[5], &[5]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![-100.0f32, -10.0, 0.0, 10.0, 100.0], &[5]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    // At ±100, tanh should saturate very close to ±1
    assert_approx_eq(result[0], -1.0, 1e-6);
    assert_approx_eq(result[1], -0.9999999958776927, 1e-6);
    assert_approx_eq(result[2], 0.0, 1e-6);
    assert_approx_eq(result[3], 0.9999999958776927, 1e-6);
    assert_approx_eq(result[4], 1.0, 1e-6);
}

#[ignore = "requires GPU"]
#[pollster::test]
async fn test_tanh_multidimensional() {
    // Test tanh on 2D tensor [2, 2]
    let graph = make_unary_graph("Tanh", "tanh_op", DataType::F32, &[2, 2], &[2, 2]);

    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let model = pipeline.compile(&graph, &registry).unwrap();

    let runtime = Runtime::new().await.unwrap();
    let mut executor = runtime.load_model(model).await.unwrap();

    let input = Tensor::from_vec(vec![-2.0f32, -1.0, 1.0, 2.0], &[2, 2]);
    let outputs = executor.run(&[("input", input)]).unwrap();
    let result: Vec<f32> = outputs["output"].to_vec().unwrap();

    let expected = vec![-0.9640275801, -0.7615941559, 0.7615941559, 0.9640275801];
    assert_vec_approx_eq(&result, &expected, 1e-6);
}
