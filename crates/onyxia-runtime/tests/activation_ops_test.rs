//! End-to-end tests for activation operations.
//!
//! Tests: Gelu, Cos, Sin, Tanh

mod common;

use common::make_unary_graph;
use onyxia_onnx::DataType;
use onyxia_planner::{KernelRegistry, compile};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: GELU activation on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_gelu_e2e() {
    // Build graph
    let graph = make_unary_graph("Gelu", "gelu_node", DataType::F32, &[6], &[6]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known GELU outputs
    // GELU(0) ≈ 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
    let x = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], &[6]);

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 6);

    // Verify approximate GELU values (tanh approximation)
    assert!(
        (y[2] - 0.0).abs() < 0.01,
        "GELU(0) should be ~0, got {}",
        y[2]
    );
    assert!(
        (y[3] - 0.8413).abs() < 0.01,
        "GELU(1) should be ~0.8413, got {}",
        y[3]
    );
    assert!(
        (y[1] + 0.1587).abs() < 0.01,
        "GELU(-1) should be ~-0.1587, got {}",
        y[1]
    );

    println!("✓ End-to-end GELU test passed!");
    println!("  Input x: [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Cos activation on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_cos_e2e() {
    // Build graph
    let graph = make_unary_graph("Cos", "cos_node", DataType::F32, &[6], &[6]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known cosine outputs
    // cos(0) = 1, cos(π/2) ≈ 0, cos(π) ≈ -1, cos(2π) ≈ 1
    use std::f32::consts::PI;
    let x = Tensor::from_vec(
        vec![0.0f32, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI, -PI / 2.0],
        &[6],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 6);

    // Verify cosine values with appropriate tolerance
    assert!(
        (y[0] - 1.0).abs() < 0.0001,
        "cos(0) should be 1, got {}",
        y[0]
    );
    assert!(y[1].abs() < 0.0001, "cos(π/2) should be ~0, got {}", y[1]);
    assert!(
        (y[2] + 1.0).abs() < 0.0001,
        "cos(π) should be -1, got {}",
        y[2]
    );
    assert!(y[3].abs() < 0.0001, "cos(3π/2) should be ~0, got {}", y[3]);
    assert!(
        (y[4] - 1.0).abs() < 0.0001,
        "cos(2π) should be 1, got {}",
        y[4]
    );
    assert!(y[5].abs() < 0.0001, "cos(-π/2) should be ~0, got {}", y[5]);

    println!("✓ End-to-end Cos test passed!");
    println!("  Input x: [0, π/2, π, 3π/2, 2π, -π/2]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Sin activation on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_sin_e2e() {
    // Build graph
    let graph = make_unary_graph("Sin", "sin_node", DataType::F32, &[6], &[6]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known sine outputs
    // sin(0) = 0, sin(π/2) ≈ 1, sin(π) ≈ 0, sin(3π/2) ≈ -1, sin(2π) ≈ 0, sin(-π/2) ≈ -1
    use std::f32::consts::PI;
    let x = Tensor::from_vec(
        vec![0.0f32, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI, -PI / 2.0],
        &[6],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 6);

    // Verify sine values with appropriate tolerance
    assert!(y[0].abs() < 0.0001, "sin(0) should be 0, got {}", y[0]);
    assert!(
        (y[1] - 1.0).abs() < 0.0001,
        "sin(π/2) should be 1, got {}",
        y[1]
    );
    assert!(y[2].abs() < 0.0001, "sin(π) should be ~0, got {}", y[2]);
    assert!(
        (y[3] + 1.0).abs() < 0.0001,
        "sin(3π/2) should be -1, got {}",
        y[3]
    );
    assert!(y[4].abs() < 0.0001, "sin(2π) should be ~0, got {}", y[4]);
    assert!(
        (y[5] + 1.0).abs() < 0.0001,
        "sin(-π/2) should be -1, got {}",
        y[5]
    );

    println!("✓ End-to-end Sin test passed!");
    println!("  Input x: [0, π/2, π, 3π/2, 2π, -π/2]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Tanh activation on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_tanh_e2e() {
    // Build graph
    let graph = make_unary_graph("Tanh", "tanh_node", DataType::F32, &[8], &[8]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known tanh outputs
    // tanh(0) = 0, tanh(∞) → 1, tanh(-∞) → -1
    // tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
    // tanh(5) ≈ 0.9999 (very close to 1)
    let x = Tensor::from_vec(vec![-5.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0], &[8]);

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 8);

    // Verify tanh values with appropriate tolerance
    assert!(
        (y[0] + 0.9999).abs() < 0.001,
        "tanh(-5) should be ≈ -0.9999, got {}",
        y[0]
    );
    assert!(
        (y[1] + 0.7616).abs() < 0.001,
        "tanh(-1) should be ≈ -0.7616, got {}",
        y[1]
    );
    assert!(
        (y[2] + 0.4621).abs() < 0.001,
        "tanh(-0.5) should be ≈ -0.4621, got {}",
        y[2]
    );
    assert!(y[3].abs() < 0.0001, "tanh(0) should be 0, got {}", y[3]);
    assert!(
        (y[4] - 0.4621).abs() < 0.001,
        "tanh(0.5) should be ≈ 0.4621, got {}",
        y[4]
    );
    assert!(
        (y[5] - 0.7616).abs() < 0.001,
        "tanh(1) should be ≈ 0.7616, got {}",
        y[5]
    );
    assert!(
        (y[6] - 0.9640).abs() < 0.001,
        "tanh(2) should be ≈ 0.9640, got {}",
        y[6]
    );
    assert!(
        (y[7] - 0.9999).abs() < 0.001,
        "tanh(5) should be ≈ 0.9999, got {}",
        y[7]
    );

    // Verify that all outputs are in the range (-1, 1)
    for (i, val) in y.iter().enumerate() {
        assert!(
            val.abs() < 1.0,
            "tanh output at index {} should be in range (-1, 1), got {}",
            i,
            val
        );
    }

    println!("✓ End-to-end Tanh test passed!");
    println!("  Input x: [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Sqrt (square root) on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_sqrt_e2e() {
    // Build graph
    let graph = make_unary_graph("Sqrt", "sqrt_node", DataType::F32, &[10], &[10]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known square root outputs
    // sqrt(0) = 0, sqrt(1) = 1, sqrt(4) = 2, sqrt(9) = 3, sqrt(16) = 4
    // sqrt(25) = 5, sqrt(100) = 10, sqrt(2) ≈ 1.414, sqrt(0.25) = 0.5
    let x = Tensor::from_vec(
        vec![0.0f32, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25, 0.01],
        &[10],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 10);

    // Verify sqrt values with appropriate tolerance
    assert!(y[0].abs() < 0.0001, "sqrt(0) should be 0, got {}", y[0]);
    assert!(
        (y[1] - 1.0).abs() < 0.0001,
        "sqrt(1) should be 1, got {}",
        y[1]
    );
    assert!(
        (y[2] - 1.414213).abs() < 0.001,
        "sqrt(2) should be ≈ 1.414, got {}",
        y[2]
    );
    assert!(
        (y[3] - 2.0).abs() < 0.0001,
        "sqrt(4) should be 2, got {}",
        y[3]
    );
    assert!(
        (y[4] - 3.0).abs() < 0.0001,
        "sqrt(9) should be 3, got {}",
        y[4]
    );
    assert!(
        (y[5] - 4.0).abs() < 0.0001,
        "sqrt(16) should be 4, got {}",
        y[5]
    );
    assert!(
        (y[6] - 5.0).abs() < 0.0001,
        "sqrt(25) should be 5, got {}",
        y[6]
    );
    assert!(
        (y[7] - 10.0).abs() < 0.0001,
        "sqrt(100) should be 10, got {}",
        y[7]
    );
    assert!(
        (y[8] - 0.5).abs() < 0.0001,
        "sqrt(0.25) should be 0.5, got {}",
        y[8]
    );
    assert!(
        (y[9] - 0.1).abs() < 0.0001,
        "sqrt(0.01) should be 0.1, got {}",
        y[9]
    );

    // Verify that all outputs are non-negative (property of sqrt)
    for (i, val) in y.iter().enumerate() {
        assert!(
            *val >= 0.0,
            "sqrt output at index {} should be non-negative, got {}",
            i,
            val
        );
    }

    println!("✓ End-to-end Sqrt test passed!");
    println!("  Input x: [0.0, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25, 0.01]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Sqrt with multidimensional tensors.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_sqrt_multidim_e2e() {
    // Build graph with 3D tensor
    let graph = make_unary_graph("Sqrt", "sqrt_node", DataType::F32, &[2, 2, 3], &[2, 2, 3]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with 3D tensor: 2x2x3 = 12 elements
    let x = Tensor::from_vec(
        vec![
            1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
        ],
        &[2, 2, 3],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 12);

    // Verify square roots
    let expected = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    for (i, (&result, &expected_val)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (result - expected_val).abs() < 0.0001,
            "sqrt at index {} should be {}, got {}",
            i,
            expected_val,
            result
        );
    }

    println!("✓ End-to-end Sqrt multidimensional test passed!");
    println!("  Input shape: [2, 2, 3]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Neg (negation) on GPU and verify correct output.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_neg_e2e() {
    // Build graph
    let graph = make_unary_graph("Neg", "neg_node", DataType::F32, &[10], &[10]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test inputs with known negation outputs
    // -(-5) = 5, -(3) = -3, -(0) = 0, -(-1.5) = 1.5, -(100) = -100
    let x = Tensor::from_vec(
        vec![
            -5.0f32, 3.0, 0.0, -1.5, 100.0, -0.25, 7.5, -42.0, 0.001, -0.001,
        ],
        &[10],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 10);

    // Expected negated values
    let expected = vec![
        5.0f32, -3.0, 0.0, 1.5, -100.0, 0.25, -7.5, 42.0, -0.001, 0.001,
    ];

    // Verify negation with appropriate tolerance
    for (i, (&result, &expected_val)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (result - expected_val).abs() < 1e-6,
            "neg at index {} should be {}, got {}",
            i,
            expected_val,
            result
        );
    }

    println!("✓ End-to-end Neg test passed!");
    println!("  Input x: [-5.0, 3.0, 0.0, -1.5, 100.0, -0.25, 7.5, -42.0, 0.001, -0.001]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Neg with multidimensional tensor.
#[pollster::test]
#[ignore] // Requires GPU
async fn test_neg_multidim_e2e() {
    // Build graph for 2x3x2 tensor
    let graph = make_unary_graph("Neg", "neg_node", DataType::F32, &[2, 3, 2], &[2, 3, 2]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with 3D tensor: 2x3x2 = 12 elements
    let x = Tensor::from_vec(
        vec![
            1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
        ],
        &[2, 3, 2],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 12);

    // Verify negation
    let expected = vec![
        -1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0,
    ];
    for (i, (&result, &expected_val)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (result - expected_val).abs() < 1e-6,
            "neg at index {} should be {}, got {}",
            i,
            expected_val,
            result
        );
    }

    println!("✓ End-to-end Neg multidimensional test passed!");
    println!("  Input shape: [2, 3, 2]");
    println!("  Output y: {:?}", y);
}

/// End-to-end test: Neg with special values (infinity, NaN).
#[pollster::test]
#[ignore] // Requires GPU
async fn test_neg_special_values_e2e() {
    // Build graph
    let graph = make_unary_graph("Neg", "neg_node", DataType::F32, &[5], &[5]);
    graph.validate().expect("Graph validation should succeed");

    // Compile and execute
    let registry = KernelRegistry::with_defaults();
    let plan = compile(&graph, &registry, &HashMap::new()).expect("Compilation should succeed");

    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Test with special float values
    let x = Tensor::from_vec(
        vec![f32::INFINITY, f32::NEG_INFINITY, 0.0f32, -0.0f32, f32::NAN],
        &[5],
    );

    let outputs = executor
        .run(&[("input", x)])
        .expect("Execution should succeed");

    let y = outputs["output"]
        .to_vec::<f32>()
        .expect("Should convert to f32");
    assert_eq!(y.len(), 5);

    // Verify special value negation
    assert!(
        y[0].is_infinite() && y[0].is_sign_negative(),
        "neg(inf) should be -inf, got {}",
        y[0]
    );
    assert!(
        y[1].is_infinite() && y[1].is_sign_positive(),
        "neg(-inf) should be inf, got {}",
        y[1]
    );
    assert_eq!(y[2], 0.0, "neg(0.0) should be 0.0");
    assert_eq!(y[3], 0.0, "neg(-0.0) should be 0.0");
    assert!(y[4].is_nan(), "neg(NaN) should be NaN");

    println!("✓ End-to-end Neg special values test passed!");
    println!("  Input x: [inf, -inf, 0.0, -0.0, NaN]");
    println!(
        "  Output y: [{}, {}, {}, {}, {}]",
        y[0], y[1], y[2], y[3], y[4]
    );
}
