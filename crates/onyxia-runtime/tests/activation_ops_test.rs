//! End-to-end tests for activation operations.
//!
//! Tests: Gelu, Cos

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
    assert!(
        y[1].abs() < 0.0001,
        "cos(π/2) should be ~0, got {}",
        y[1]
    );
    assert!(
        (y[2] + 1.0).abs() < 0.0001,
        "cos(π) should be -1, got {}",
        y[2]
    );
    assert!(
        y[3].abs() < 0.0001,
        "cos(3π/2) should be ~0, got {}",
        y[3]
    );
    assert!(
        (y[4] - 1.0).abs() < 0.0001,
        "cos(2π) should be 1, got {}",
        y[4]
    );
    assert!(
        y[5].abs() < 0.0001,
        "cos(-π/2) should be ~0, got {}",
        y[5]
    );

    println!("✓ End-to-end Cos test passed!");
    println!("  Input x: [0, π/2, π, 3π/2, 2π, -π/2]");
    println!("  Output y: {:?}", y);
}
