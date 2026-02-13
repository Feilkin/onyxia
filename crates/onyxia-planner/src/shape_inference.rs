//! Three-phase shape inference for the Onyxia planner.
//!
//! ## Phase 1: Dynamic Dimension Substitution
//!
//! Replaces all `Dynamic(Named(...))` dimensions with concrete `Static` values
//! from the user-provided `dynamic_dimensions` map. After this phase, no Named
//! dimensions remain in the graph.
//!
//! ## Phase 2: Iterative Forward Shape Inference
//!
//! Runs multiple forward passes over the graph, using kernel-defined
//! `infer_output_shapes` rules, until shapes converge (fixed-point).
//! This handles cascading dependencies where one node's output shape
//! depends on another node's not-yet-inferred output.

use crate::error::{CodegenError, Result};
use crate::kernel::KernelRegistry;
use onyxia_onnx::{Dimension, Graph, TensorShape};
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
                        let size = dynamic_dimensions.get(name).ok_or_else(|| {
                            CodegenError::InvalidShape(format!(
                                "Dynamic dimension '{}' not provided in dynamic_dimensions \
                                 (tensor: '{}')",
                                name, info.name
                            ))
                        })?;
                        resolved.push(*size);
                    }
                }
            }
            info.shape = TensorShape::Static(resolved);
        }
    }

    debug!("Phase 1 complete: all dynamic dimensions resolved");
    Ok(())
}

/// Phase 2: Iterative forward shape inference using kernel-defined rules.
///
/// Runs multiple passes over the graph's nodes, calling each kernel's
/// `infer_output_shapes` to resolve `Unknown` shapes from known inputs.
/// Iterates until no new shapes are inferred (fixed-point convergence).
///
/// Must be called **after** [`resolve_dynamic_dimensions`] — all shapes are
/// guaranteed to be either `Static`, `Unknown`, or `Absent` (no `Named` dims).
pub fn infer_shapes(graph: &mut Graph, registry: &KernelRegistry) -> Result<()> {
    const MAX_ITERATIONS: usize = 20;

    debug!(
        "Phase 2: Starting iterative shape inference ({} nodes, {} tensors)",
        graph.nodes.len(),
        graph.tensor_info.len()
    );

    for iteration in 0..MAX_ITERATIONS {
        let mut changed = false;

        for node_idx in 0..graph.nodes.len() {
            let node = &graph.nodes[node_idx];

            // Collect input shapes
            let mut input_shapes = Vec::new();
            let mut has_missing_input = false;
            let mut has_unknown_non_optional = false;

            for (input_idx, input_name) in node.inputs.iter().enumerate() {
                if input_name.is_empty() {
                    input_shapes.push(TensorShape::Absent);
                    continue;
                }

                if let Some(&tensor_id) = graph.tensors.get(input_name) {
                    let tensor_info = &graph.tensor_info[tensor_id];
                    let shape = tensor_info.shape.clone();

                    if matches!(shape, TensorShape::Unknown) {
                        has_unknown_non_optional = true;
                    }

                    input_shapes.push(shape);
                } else {
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

            // Skip nodes whose non-optional inputs are still Unknown
            if has_unknown_non_optional {
                continue;
            }

            // Look up kernel
            let kernel = match registry.get(&node.op_type) {
                Some(k) => k,
                None => {
                    if iteration == 0 {
                        warn!(
                            node = node.name.as_str(),
                            op_type = node.op_type.as_str(),
                            "No kernel registered for operator, cannot infer shape"
                        );
                    }
                    continue;
                }
            };

            // Infer output shapes (no dynamic_dimensions — Phase 1 already resolved them)
            let output_shapes = match kernel.infer_output_shapes(graph, node, &input_shapes) {
                Ok(shapes) => shapes,
                Err(e) => {
                    if iteration == 0 {
                        warn!(
                            node = node.name.as_str(),
                            op_type = node.op_type.as_str(),
                            error = %e,
                            "Failed to infer output shapes"
                        );
                    }
                    continue;
                }
            };

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

            // Apply inferred shapes — only update Unknown → known
            for (output_name, output_shape) in node.outputs.iter().zip(output_shapes.iter()) {
                if let Some(&tensor_id) = graph.tensors.get(output_name) {
                    let tensor_info = &mut graph.tensor_info[tensor_id];

                    if matches!(tensor_info.shape, TensorShape::Unknown)
                        && !matches!(output_shape, TensorShape::Unknown)
                    {
                        tensor_info.shape = output_shape.clone();
                        changed = true;
                        debug!(
                            node = node.name.as_str(),
                            op_type = node.op_type.as_str(),
                            output = output_name.as_str(),
                            shape = ?output_shape,
                            iteration = iteration,
                            "Inferred output shape"
                        );
                    }
                }
            }
        }

        if !changed {
            debug!("Phase 2: converged after {} iteration(s)", iteration + 1);
            break;
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
