//! Shape inference using kernel-defined shape rules.
//!
//! Instead of a centralized match block, shape inference is delegated to each
//! kernel implementation via the `OpKernel::infer_output_shapes` method.

use crate::error::Result;
use crate::kernel::KernelRegistry;
use onyxia_onnx::{Graph, TensorShape};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Infer shapes for all tensors in the graph using kernel-defined inference rules.
///
/// This uses the kernel registry to dispatch shape inference to the appropriate
/// kernel for each operation. Dynamic dimensions are resolved using the provided
/// concrete values.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to infer shapes for (will be mutated)
/// * `registry` - Kernel registry for looking up shape inference implementations
/// * `dynamic_dimensions` - Concrete values for dynamic dimensions (e.g., {"batch": 1})
///
/// # Returns
///
/// Returns Ok(()) if shape inference succeeds, or an error if a shape cannot be inferred.
pub fn infer_shapes(
    graph: &mut Graph,
    registry: &KernelRegistry,
    dynamic_dimensions: &HashMap<String, usize>,
) -> Result<()> {
    debug!(
        "Starting shape inference with {} nodes, {} tensors",
        graph.nodes.len(),
        graph.tensor_info.len()
    );

    // Process nodes in topological order
    for node_idx in 0..graph.nodes.len() {
        let node = &graph.nodes[node_idx];

        // Collect input shapes
        let mut input_shapes = Vec::new();
        let mut has_missing_input = false;
        let mut has_unknown_non_optional = false;

        for (input_idx, input_name) in node.inputs.iter().enumerate() {
            // Skip empty string inputs (optional inputs in ONNX)
            if input_name.is_empty() {
                debug!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    input_idx = input_idx,
                    "Optional input (empty string) - using Absent placeholder"
                );
                input_shapes.push(TensorShape::Absent);
                continue;
            }

            if let Some(&tensor_id) = graph.tensors.get(input_name) {
                let tensor_info = &graph.tensor_info[tensor_id];
                let shape = tensor_info.shape.clone();

                // Track if non-optional inputs have unknown shapes
                if matches!(shape, TensorShape::Unknown) {
                    has_unknown_non_optional = true;
                }

                input_shapes.push(shape);
            } else {
                // Non-optional input tensor is missing from graph
                debug!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    input_idx = input_idx,
                    missing_input = input_name.as_str(),
                    "Skipping shape inference: missing input tensor"
                );
                has_missing_input = true;
                break;
            }
        }

        if has_missing_input {
            continue;
        }

        // Only skip if non-optional inputs have unknown shapes
        // Optional inputs (empty strings) with Unknown shapes are OK
        if has_unknown_non_optional {
            // Can't infer output shapes without non-optional input shapes
            debug!(
                node = node.name.as_str(),
                op_type = node.op_type.as_str(),
                "Skipping shape inference: unknown shapes in non-optional inputs"
            );
            continue;
        }

        // Look up kernel for this operation
        let kernel = match registry.get(&node.op_type) {
            Some(k) => k,
            None => {
                // No kernel registered for this op_type
                warn!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    "No kernel registered for operator, cannot infer shape"
                );
                continue;
            }
        };

        // Infer output shapes using the kernel
        let output_shapes =
            match kernel.infer_output_shapes(node, &input_shapes, dynamic_dimensions) {
                Ok(shapes) => shapes,
                Err(e) => {
                    warn!(
                        node = node.name.as_str(),
                        op_type = node.op_type.as_str(),
                        error = %e,
                        "Failed to infer output shapes"
                    );
                    continue;
                }
            };

        // Apply inferred shapes to output tensors
        if output_shapes.len() != node.outputs.len() {
            warn!(
                node = node.name.as_str(),
                op_type = node.op_type.as_str(),
                expected = node.outputs.len(),
                got = output_shapes.len(),
                "Kernel returned wrong number of output shapes"
            );
            continue;
        }

        for (output_name, output_shape) in node.outputs.iter().zip(output_shapes.iter()) {
            if let Some(&tensor_id) = graph.tensors.get(output_name) {
                let tensor_info = &mut graph.tensor_info[tensor_id];

                // Only update if currently unknown
                if matches!(tensor_info.shape, TensorShape::Unknown) {
                    tensor_info.shape = output_shape.clone();
                    debug!(
                        node = node.name.as_str(),
                        op_type = node.op_type.as_str(),
                        output = output_name.as_str(),
                        shape = ?output_shape,
                        "Inferred output shape"
                    );
                }
            }
        }
    }

    // Log summary
    let unknown_count = graph
        .tensor_info
        .iter()
        .filter(|t| matches!(t.shape, TensorShape::Unknown))
        .count();
    let total = graph.tensor_info.len();

    if unknown_count > 0 {
        debug!(
            unknown_shapes = unknown_count,
            total_tensors = total,
            percentage = format!("{:.1}%", (unknown_count as f64 / total as f64) * 100.0),
            "Shape inference completed with unknown shapes"
        );
    } else {
        debug!(
            total_tensors = total,
            "Shape inference completed successfully - all shapes resolved"
        );
    }

    Ok(())
}
