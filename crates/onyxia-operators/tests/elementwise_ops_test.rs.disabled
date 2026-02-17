//! End-to-end tests for elementwise operations.
//!
//! Tests: Add, Mul

mod common;

use common::make_binary_elementwise_graph;
use onyxia_compiler::CompilerPipeline;
use onyxia_onnx::DataType;
use onyxia_operators::core_operator_registry;
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

/// End-to-end test: Add two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore = "requires GPU"]
async fn test_add_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Add", "add_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Verify plan structure
    assert_eq!(plan.operations.len(), 1);
    assert_eq!(plan.shaders.len(), 1);
    assert_eq!(plan.inputs.len(), 2);
    assert_eq!(plan.outputs.len(), 1);

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    println!(
        "GPU: {} ({:?})",
        runtime.adapter_info().name,
        runtime.adapter_info().backend
    );

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Run with concrete inputs
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Verify output
    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0], "Add result incorrect");

    println!("✓ End-to-end Add test passed!");
    println!("  Input a: [1.0, 2.0, 3.0, 4.0]");
    println!("  Input b: [5.0, 6.0, 7.0, 8.0]");
    println!("  Output c: {:?}", c);
}

/// End-to-end test: Multiply two vectors on GPU and verify correct output.
#[pollster::test]
#[ignore = "requires GPU"]
async fn test_mul_e2e() {
    // Build graph
    let graph = make_binary_elementwise_graph("Mul", "mul_node", DataType::F32, &[4]);
    graph.validate().expect("Graph validation should succeed");

    // Compile to CompiledModel
    let registry = core_operator_registry();
    let plan = CompilerPipeline::new(HashMap::new())
        .compile(&graph, &registry)
        .expect("Compilation should succeed");

    // Verify plan structure
    assert_eq!(plan.operations.len(), 1);
    assert_eq!(plan.shaders.len(), 1);

    // Initialize runtime and load plan
    let runtime = Runtime::new()
        .await
        .expect("Runtime initialization should succeed");

    let mut executor = runtime
        .load_model(plan)
        .await
        .expect("Plan loading should succeed");

    // Run with concrete inputs
    let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], &[4]);
    let b = Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], &[4]);

    let outputs = executor
        .run(&[("a", a), ("b", b)])
        .expect("Execution should succeed");

    // Verify output
    let c = outputs["c"].to_vec::<f32>().expect("Should convert to f32");
    assert_eq!(c, vec![6.0, 12.0, 20.0, 30.0], "Mul result incorrect");

    println!("✓ End-to-end Mul test passed!");
    println!("  Input a: [2.0, 3.0, 4.0, 5.0]");
    println!("  Input b: [3.0, 4.0, 5.0, 6.0]");
    println!("  Output c: {:?}", c);
}
