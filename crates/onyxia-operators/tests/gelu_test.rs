//! End-to-end test for Gelu activation operator.

mod common;

use common::{CompilerPipeline, Runtime, Tensor, make_unary_graph};
use onyxia_onnx::DataType;
use onyxia_operators::core_operator_registry;

#[pollster::test]
#[ignore] // Requires GPU
async fn test_gelu_e2e() {
    // Create a simple Gelu graph: input[6] -> Gelu -> output[6]
    let graph = make_unary_graph("Gelu", "gelu_op", DataType::F32, &[6], &[6]);

    // Compile
    let registry = core_operator_registry();
    let mut pipeline = CompilerPipeline::new();
    let compiled = pipeline
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Execute
    let runtime = Runtime::new().await.expect("Runtime init should succeed");
    let mut executor = runtime
        .load_model(compiled)
        .await
        .expect("Model loading should succeed");

    // Test inputs: [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    // Expected GELU outputs (tanh approximation):
    //   GELU(-2) ≈ -0.0454
    //   GELU(-1) ≈ -0.1587
    //   GELU(0) = 0
    //   GELU(1) ≈ 0.8413
    //   GELU(2) ≈ 1.9546
    //   GELU(3) ≈ 2.9964
    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let input = Tensor::from_vec(input_data.clone(), &[6]);

    let outputs = executor
        .run(&[("input", input)])
        .expect("Execution should succeed");

    let result: Vec<f32> = outputs["output"].to_vec().expect("Should convert to f32");

    assert_eq!(result.len(), 6);

    // Verify approximate GELU values with tolerance
    assert!(
        (result[0] + 0.0454).abs() < 0.01,
        "GELU(-2) should be ~-0.0454, got {}",
        result[0]
    );
    assert!(
        (result[1] + 0.1587).abs() < 0.01,
        "GELU(-1) should be ~-0.1587, got {}",
        result[1]
    );
    assert!(
        (result[2] - 0.0).abs() < 0.01,
        "GELU(0) should be ~0, got {}",
        result[2]
    );
    assert!(
        (result[3] - 0.8413).abs() < 0.01,
        "GELU(1) should be ~0.8413, got {}",
        result[3]
    );
    assert!(
        (result[4] - 1.9546).abs() < 0.01,
        "GELU(2) should be ~1.9546, got {}",
        result[4]
    );
    assert!(
        (result[5] - 2.9964).abs() < 0.01,
        "GELU(3) should be ~2.9964, got {}",
        result[5]
    );

    println!("✓ Gelu operator test passed!");
    println!("  Input:  {:?}", input_data);
    println!("  Output: {:?}", result);
}
