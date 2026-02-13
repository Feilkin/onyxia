//! Three-phase shape inference for the Onyxia planner.
//!
//! ## Phase 1: Dynamic Dimension Substitution
//!
//! Replaces all `Dynamic(Named(...))` dimensions with concrete `Static` values
//! from the user-provided `dynamic_dimensions` map. After this phase, no Named
//! dimensions remain in the graph.
//!
//! ## Phase 2: Forward Shape and Value Inference
//!
//! Runs a single forward pass over the graph in topological order, using
//! kernel-defined `infer_output_shapes` and `try_fold` rules. Propagates both
//! shapes and constant values (for data-dependent shape inference like Reshape
//! reading a computed shape tensor).

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::kernel::KernelRegistry;
use crate::scheduler::Scheduler;
use crate::symbolic_expr::{evaluate_expr, parse_expr};
use onyxia_onnx::{Dimension, Graph, TensorId, TensorShape};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Phase 1: Substitute all named dynamic dimensions with concrete values.
///
/// Converts every `TensorShape::Dynamic` into `TensorShape::Static` by
/// looking up `Dimension::Named` entries in `dynamic_dimensions`.
/// After this phase, no `Named` dimensions remain in the graph.
///
/// # Errors
///
/// Returns `CodegenError::InvalidShape` if a named dimension is not found
/// in `dynamic_dimensions`.
pub fn resolve_dynamic_dimensions(
    graph: &mut Graph,
    dynamic_dimensions: &HashMap<String, usize>,
) -> Result<()> {
    debug!(
        "Phase 1: Resolving dynamic dimensions for {} tensors",
        graph.tensor_info.len()
    );

    for info in graph.tensor_info.iter_mut() {
        if let TensorShape::Dynamic(dims) = &info.shape {
            let mut resolved = Vec::with_capacity(dims.len());
            for dim in dims {
                match dim {
                    Dimension::Static(size) => resolved.push(*size),
                    Dimension::Named(name) => {
                        // Parse and evaluate the dimension expression
                        let expr = parse_expr(name).map_err(|parse_err| {
                            CodegenError::InvalidShape(format!(
                                "Failed to parse dimension '{}': {} (tensor: '{}')",
                                name, parse_err, info.name
                            ))
                        })?;

                        let size = evaluate_expr(&expr, dynamic_dimensions).map_err(|eval_err| {
                            CodegenError::InvalidShape(format!(
                                "Failed to evaluate dimension '{}': {} (tensor: '{}')",
                                name, eval_err, info.name
                            ))
                        })?;

                        resolved.push(size);
                    }
                }
            }
            info.shape = TensorShape::Static(resolved);
        }
    }

    debug!("Phase 1 complete: all dynamic dimensions resolved");
    Ok(())
}

/// Phase 2: Forward shape and value inference using kernel-defined rules.
///
/// Runs a single forward pass over the graph's nodes in topological order,
/// calling each kernel's `infer_output_shapes` and `try_fold` to resolve
/// `Unknown` shapes and propagate constant values.
///
/// Must be called **after** [`resolve_dynamic_dimensions`] — all shapes are
/// guaranteed to be either `Static`, `Unknown`, or `Absent` (no `Named` dims).
pub fn infer_shapes(graph: &mut Graph, registry: &KernelRegistry) -> Result<()> {
    debug!(
        "Phase 2: Starting forward shape and value inference ({} nodes, {} tensors)",
        graph.nodes.len(),
        graph.tensor_info.len()
    );

    // Build topological ordering
    let scheduler = Scheduler::new(graph.clone());
    let order = scheduler.schedule()?;

    debug!("Topological order computed: {} nodes", order.len());

    // Initialize value map with initializers
    let mut value_map: HashMap<TensorId, TensorValue> = HashMap::new();

    for (tensor_id, tensor_info) in graph.tensor_info.iter().enumerate() {
        if let Some(value) = TensorValue::from_initializer(tensor_info)? {
            value_map.insert(tensor_id, value);
            debug!(
                tensor = tensor_info.name.as_str(),
                tensor_id = tensor_id,
                "Initialized constant tensor value"
            );
        }
    }

    // Forward pass: infer shapes and fold values
    for &node_idx in &order {
        let node = &graph.nodes[node_idx];

        // Collect input shapes and values
        let mut input_shapes = Vec::new();
        let mut input_values = Vec::new();
        let mut has_missing_input = false;

        for (input_idx, input_name) in node.inputs.iter().enumerate() {
            if input_name.is_empty() {
                // Absent (optional) input
                input_shapes.push(TensorShape::Absent);
                input_values.push(None);
                continue;
            }

            if let Some(&tensor_id) = graph.tensors.get(input_name) {
                let tensor_info = &graph.tensor_info[tensor_id];
                input_shapes.push(tensor_info.shape.clone());
                input_values.push(value_map.get(&tensor_id).cloned());
            } else {
                debug!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    input_idx = input_idx,
                    missing_input = input_name.as_str(),
                    "Skipping inference: missing input tensor"
                );
                has_missing_input = true;
                break;
            }
        }

        if has_missing_input {
            continue;
        }

        // Look up kernel
        let kernel = match registry.get(&node.op_type) {
            Some(k) => k,
            None => {
                warn!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    "No kernel registered for operator, cannot infer shape"
                );
                continue;
            }
        };

        // Build inference context and perform inference
        let (output_shapes, output_values) = {
            let ctx = InferenceContext::new(node, graph, input_shapes, input_values);

            // Infer output shapes
            let shapes = match kernel.infer_output_shapes(&ctx) {
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

            if shapes.len() != node.outputs.len() {
                warn!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    expected = node.outputs.len(),
                    got = shapes.len(),
                    "Kernel returned wrong number of output shapes"
                );
                continue;
            }

            // Try constant folding
            let values = match kernel.try_fold(&ctx) {
                Ok(values) => values,
                Err(e) => {
                    debug!(
                        node = node.name.as_str(),
                        op_type = node.op_type.as_str(),
                        error = %e,
                        "Constant folding failed (non-fatal)"
                    );
                    vec![None; node.outputs.len()]
                }
            };

            if values.len() != node.outputs.len() {
                warn!(
                    node = node.name.as_str(),
                    op_type = node.op_type.as_str(),
                    expected = node.outputs.len(),
                    got = values.len(),
                    "Kernel returned wrong number of output values from try_fold"
                );
                continue;
            }

            (shapes, values)
        };

        // Apply inferred shapes (ctx is now dropped)
        for (output_name, output_shape) in node.outputs.iter().zip(output_shapes.iter()) {
            if let Some(&tensor_id) = graph.tensors.get(output_name) {
                let tensor_info = &mut graph.tensor_info[tensor_id];
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

        // Store folded values
        for (output_name, output_value) in node.outputs.iter().zip(output_values.iter()) {
            if let Some(value) = output_value {
                if let Some(&tensor_id) = graph.tensors.get(output_name) {
                    value_map.insert(tensor_id, value.clone());
                    debug!(
                        node = node.name.as_str(),
                        op_type = node.op_type.as_str(),
                        output = output_name.as_str(),
                        "Folded output to constant value"
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
            "Shape inference completed with unknown shapes remaining"
        );
    } else {
        debug!(
            total_tensors = total,
            "Shape inference completed — all shapes resolved"
        );
    }

    Ok(())
}
