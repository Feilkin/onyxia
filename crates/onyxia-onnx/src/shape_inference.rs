//! Shape inference for ONNX graphs.
//!
//! Propagates shape information through the graph using operation-specific inference rules.

use crate::graph::{Dimension, Graph, Node, TensorShape};
use crate::{OnnxError, Result};
use std::collections::HashMap;
use tracing::{debug, trace, warn};

/// Infer shapes for all tensors in the graph.
///
/// This mutates the graph to fill in `TensorShape::Unknown` with inferred shapes
/// based on operation semantics and input shapes.
pub fn infer_shapes(graph: &mut Graph) -> Result<()> {
    let mut context = InferenceContext::new(graph);

    // Iterate through nodes in order and infer output shapes + evaluate constants
    for i in 0..graph.nodes.len() {
        let node = &graph.nodes[i];
        infer_node_output_shapes(&mut context, node, graph)?;
    }

    // Apply inferred shapes back to the graph
    context.apply_to_graph(graph);

    // Log summary of unknown shapes
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

/// Context for shape inference tracking.
struct InferenceContext {
    /// Tensor ID -> inferred shape
    shapes: HashMap<String, TensorShape>,
    /// Tensor ID -> evaluated constant value (for integer constants)
    constants: HashMap<String, Vec<i64>>,
}

impl InferenceContext {
    fn new(graph: &Graph) -> Self {
        let mut shapes = HashMap::new();
        let mut constants = HashMap::new();

        // Initialize with known shapes from graph and extract constants from weight initializers
        for (name, &tensor_id) in &graph.tensors {
            let info = &graph.tensor_info[tensor_id];
            if !matches!(info.shape, TensorShape::Unknown) {
                shapes.insert(name.clone(), info.shape.clone());
            }

            // Extract constants from weight tensors with initializer data
            // This handles both initializers and Constant node outputs
            if let Some(ref initializer) = info.initializer
                && let Some(values) = decode_constant_bytes(initializer, info.dtype)
            {
                constants.insert(name.clone(), values);
            }
        }

        Self { shapes, constants }
    }

    fn get_shape(&self, name: &str) -> Option<&TensorShape> {
        self.shapes.get(name)
    }

    fn set_shape(&mut self, name: String, shape: TensorShape) {
        self.shapes.insert(name, shape);
    }

    fn get_constant(&self, name: &str) -> Option<&Vec<i64>> {
        self.constants.get(name)
    }

    fn set_constant(&mut self, name: String, value: Vec<i64>) {
        self.constants.insert(name, value);
    }

    fn apply_to_graph(self, graph: &mut Graph) {
        for (name, shape) in self.shapes {
            if let Some(&tensor_id) = graph.tensors.get(&name) {
                graph.tensor_info[tensor_id].shape = shape;
            }
        }
    }
}

/// Infer output shapes for a single node.
#[allow(clippy::collapsible_if)]
fn infer_node_output_shapes(
    context: &mut InferenceContext,
    node: &Node,
    graph: &Graph,
) -> Result<()> {
    match node.op_type.as_str() {
        // Shape-preserving unary operations
        "Cast" | "Gelu" | "Relu" | "Sigmoid" | "Tanh" | "Softmax" | "Neg" | "Abs" | "Sqrt"
        | "Exp" | "Log" | "Floor" | "Ceil" => {
            if let Some(input) = node.inputs.first() {
                if let Some(shape) = context.get_shape(input).cloned() {
                    for output in &node.outputs {
                        context.set_shape(output.clone(), shape.clone());
                    }
                }
            }
        }

        // Binary operations with broadcasting
        "Add" | "Sub" | "Mul" | "Div" => {
            if node.inputs.len() >= 2 {
                if let (Some(shape_a), Some(shape_b)) = (
                    context.get_shape(&node.inputs[0]),
                    context.get_shape(&node.inputs[1]),
                ) {
                    let result_shape = broadcast_shapes(shape_a, shape_b)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
        }

        // Normalization operations (preserve shape)
        "SimplifiedLayerNormalization" | "LayerNormalization" | "BatchNormalization" => {
            if let Some(input) = node.inputs.first() {
                if let Some(shape) = context.get_shape(input).cloned() {
                    for output in &node.outputs {
                        context.set_shape(output.clone(), shape.clone());
                    }
                }
            }
        }

        // MatMul and quantized variants
        "MatMul" | "MatMulNBits" | "MatMulInteger" => {
            if node.inputs.len() >= 2 {
                if let (Some(shape_a), Some(shape_b)) = (
                    context.get_shape(&node.inputs[0]),
                    context.get_shape(&node.inputs[1]),
                ) {
                    let result_shape = matmul_shape(shape_a, shape_b)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
        }

        // Shape operation: returns 1D tensor with input's shape
        "Shape" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let (rank, shape_values) = match input_shape {
                        TensorShape::Static(dims) => (
                            dims.len(),
                            Some(dims.iter().map(|&d| d as i64).collect::<Vec<_>>()),
                        ),
                        TensorShape::Dynamic(dims) => {
                            // Only produce constant if all dimensions are static
                            let values: Option<Vec<i64>> = dims
                                .iter()
                                .map(|d| match d {
                                    Dimension::Static(s) => Some(*s as i64),
                                    Dimension::Named(_) => None,
                                })
                                .collect();
                            (dims.len(), values)
                        }
                        TensorShape::Unknown => {
                            return Ok(()); // Can't infer without knowing input shape
                        }
                    };
                    for output in &node.outputs {
                        context.set_shape(output.clone(), TensorShape::Static(vec![rank]));
                        // Store the shape as a constant value
                        if let Some(ref vals) = shape_values {
                            context.set_constant(output.clone(), vals.clone());
                        }
                    }
                }
            }
        }

        // Gather: lookup operation
        "Gather" => {
            // Output shape depends on data shape, indices shape, and axis
            // Simplified: if we have data shape and indices shape, we can infer
            if node.inputs.len() >= 2 {
                if let (Some(data_shape), Some(indices_shape)) = (
                    context.get_shape(&node.inputs[0]),
                    context.get_shape(&node.inputs[1]),
                ) {
                    // For now, handle simple case: gathering along axis 0
                    // Result has indices shape + data shape (minus gathered axis)
                    let result_shape = gather_shape(data_shape, indices_shape)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
            // Propagate constants through Gather
            if node.inputs.len() >= 2 {
                if let (Some(data_const), Some(indices_const)) = (
                    context.get_constant(&node.inputs[0]).cloned(),
                    context.get_constant(&node.inputs[1]).cloned(),
                ) {
                    let axis = node
                        .attributes
                        .get("axis")
                        .and_then(|a| match a {
                            crate::graph::AttributeValue::Int(v) => Some(*v),
                            _ => None,
                        })
                        .unwrap_or(0);

                    // Get data shape for gather
                    if let Some(data_shape) = context.get_shape(&node.inputs[0]) {
                        let data_dims: Vec<usize> = match data_shape {
                            TensorShape::Static(dims) => dims.clone(),
                            TensorShape::Dynamic(dims) => dims
                                .iter()
                                .filter_map(|d| match d {
                                    Dimension::Static(s) => Some(*s),
                                    _ => None,
                                })
                                .collect(),
                            TensorShape::Unknown => vec![],
                        };

                        if !data_dims.is_empty() {
                            if let Some(result) = evaluate_gather_constant(
                                &data_const,
                                &indices_const,
                                axis,
                                &data_dims,
                            ) {
                                for output in &node.outputs {
                                    context.set_constant(output.clone(), result.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reshape: output shape from second input (constant) or attributes
        "Reshape" => {
            if node.inputs.len() >= 2 {
                let input_shape = context.get_shape(&node.inputs[0]).cloned();
                let shape_values = context.get_constant(&node.inputs[1]).cloned();

                match (input_shape, shape_values) {
                    (Some(input_shape), Some(shape_values)) => {
                        // shape_values is the target shape, e.g., [-1, 256]
                        // Resolve -1 using input element count
                        if let Ok(result) = resolve_reshape_shape(&input_shape, &shape_values) {
                            for output in &node.outputs {
                                context.set_shape(output.clone(), result.clone());
                            }
                        }
                    }
                    (None, _) => {
                        trace!(
                            op_type = "Reshape",
                            node_name = %node.name,
                            input = %node.inputs[0],
                            "Cannot infer Reshape output - input shape unknown"
                        );
                    }
                    (_, None) => {
                        trace!(
                            op_type = "Reshape",
                            node_name = %node.name,
                            shape_input = %node.inputs[1],
                            "Cannot infer Reshape output - shape constant not evaluated"
                        );
                    }
                }
            }
        }

        // Transpose: permutes dimensions
        "Transpose" => {
            if let Some(input) = node.inputs.first() {
                if let Some(shape) = context.get_shape(input) {
                    // Default transpose is reverse all dimensions
                    let result_shape = transpose_shape(shape, &node.attributes)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
        }

        // Constant: shape extracted at parse time + evaluate constant values
        "Constant" => {
            // Extract constant values from initializer data for integer tensors
            for output_name in &node.outputs {
                if let Ok(tensor_info) = graph.tensor_by_name(output_name) {
                    if let Some(ref initializer) = tensor_info.initializer {
                        if let Some(values) = decode_constant_bytes(initializer, tensor_info.dtype)
                        {
                            context.set_constant(output_name.clone(), values);
                        }
                    }
                }
            }
        }

        // Unsqueeze: add dimensions at specified axes
        "Unsqueeze" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape =
                        unsqueeze_shape(input_shape, &node.inputs, &node.attributes, context)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                    // Propagate constants through unsqueeze if input is constant
                    if let Some(input_const) = context.get_constant(input).cloned() {
                        for output in &node.outputs {
                            context.set_constant(output.clone(), input_const.clone());
                        }
                    }
                }
            }
        }

        // Squeeze: remove dimensions at specified axes
        "Squeeze" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape =
                        squeeze_shape(input_shape, &node.inputs, &node.attributes, context)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                    // Propagate constants through squeeze if input is constant
                    if let Some(input_const) = context.get_constant(input).cloned() {
                        for output in &node.outputs {
                            context.set_constant(output.clone(), input_const.clone());
                        }
                    }
                }
            }
        }

        // Concat: concatenate along an axis
        "Concat" => {
            let result_shape = concat_shape(&node.inputs, &node.attributes, context)?;
            for output in &node.outputs {
                context.set_shape(output.clone(), result_shape.clone());
            }
            // Propagate constants through Concat if all inputs are constants
            let all_constants: Option<Vec<&Vec<i64>>> = node
                .inputs
                .iter()
                .map(|name| context.get_constant(name))
                .collect();

            if let Some(constants) = all_constants {
                let axis = node
                    .attributes
                    .get("axis")
                    .and_then(|a| match a {
                        crate::graph::AttributeValue::Int(v) => Some(*v),
                        _ => None,
                    })
                    .unwrap_or(0);

                if let Some(result) = evaluate_concat_constant(&constants, axis) {
                    for output in &node.outputs {
                        context.set_constant(output.clone(), result.clone());
                    }
                }
            }
        }

        // ReduceSum, ReduceMean: reduce along specified axes
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape =
                        reduce_shape(input_shape, &node.inputs, &node.attributes, context)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
        }

        // Custom operations - preserve input shape for now
        "RotaryEmbedding" => {
            if let Some(input) = node.inputs.first() {
                if let Some(shape) = context.get_shape(input).cloned() {
                    for output in &node.outputs {
                        context.set_shape(output.clone(), shape.clone());
                    }
                }
            }
        }

        // Attention operations - complex shape transformations
        "GroupQueryAttention" | "Attention" | "MultiHeadAttention" => {
            // These have complex output shapes depending on parameters
            // For now, we'll need to look at attributes
            // Simplified: preserve query shape for first output
            if let Some(query) = node.inputs.first() {
                if let Some(shape) = context.get_shape(query) {
                    if let Some(output) = node.outputs.first() {
                        context.set_shape(output.clone(), shape.clone());
                    }
                }
            }
        }

        // Default: unknown operation, can't infer
        _ => {
            // Leave outputs as Unknown
            if !node.outputs.is_empty() {
                warn!(
                    op_type = %node.op_type,
                    node_name = %node.name,
                    outputs = ?node.outputs,
                    "Unsupported operation - cannot infer output shapes"
                );
            }
        }
    }

    Ok(())
}

/// Broadcast two shapes according to NumPy broadcasting rules.
fn broadcast_shapes(a: &TensorShape, b: &TensorShape) -> Result<TensorShape> {
    let dims_a = match a {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown), // Can't infer, propagate unknown
    };

    let dims_b = match b {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown), // Can't infer, propagate unknown
    };

    // Broadcast: align from the right, take max of each dimension
    let mut result = Vec::new();
    let max_len = dims_a.len().max(dims_b.len());

    for i in 0..max_len {
        let idx_a = dims_a.len().saturating_sub(max_len - i);
        let idx_b = dims_b.len().saturating_sub(max_len - i);

        let dim_a = dims_a.get(idx_a);
        let dim_b = dims_b.get(idx_b);

        let result_dim = match (dim_a, dim_b) {
            (Some(Dimension::Static(1)), Some(d)) | (Some(d), Some(Dimension::Static(1))) => {
                d.clone()
            }
            (Some(a), Some(b)) if a == b => a.clone(),
            (Some(Dimension::Static(a)), Some(Dimension::Static(b)))
                if *a != *b && *a != 1 && *b != 1 =>
            {
                return Err(OnnxError::ShapeInference(format!(
                    "Cannot broadcast dimensions {} and {}",
                    a, b
                )));
            }
            (Some(a), None) | (None, Some(a)) => a.clone(),
            (Some(a), Some(_)) => a.clone(), // Dynamic dimensions, keep first
            (None, None) => unreachable!(),
        };

        result.push(result_dim);
    }

    Ok(TensorShape::Dynamic(result))
}

/// Infer MatMul output shape.
fn matmul_shape(a: &TensorShape, b: &TensorShape) -> Result<TensorShape> {
    // MatMul([..., M, K], [..., K, N]) -> [..., M, N]
    let dims_a = match a {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    let dims_b = match b {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    if dims_a.len() < 2 || dims_b.len() < 2 {
        return Err(OnnxError::ShapeInference(
            "MatMul requires at least 2D tensors".to_string(),
        ));
    }

    // Broadcast batch dimensions
    let batch_a = &dims_a[..dims_a.len() - 2];
    let batch_b = &dims_b[..dims_b.len() - 2];

    let mut result_dims = Vec::new();

    // Broadcast batch dims
    let max_batch_len = batch_a.len().max(batch_b.len());
    for i in 0..max_batch_len {
        let idx_a = batch_a.len().saturating_sub(max_batch_len - i);
        let idx_b = batch_b.len().saturating_sub(max_batch_len - i);

        let dim_a = batch_a.get(idx_a);
        let dim_b = batch_b.get(idx_b);

        let result_dim = match (dim_a, dim_b) {
            (Some(Dimension::Static(1)), Some(d)) | (Some(d), Some(Dimension::Static(1))) => {
                d.clone()
            }
            (Some(a), Some(b)) if a == b => a.clone(),
            (Some(a), None) | (None, Some(a)) => a.clone(),
            (Some(a), Some(_)) => a.clone(), // Take first for dynamic
            (None, None) => unreachable!(),
        };

        result_dims.push(result_dim);
    }

    // Add matrix dimensions: M from A, N from B
    let m = dims_a[dims_a.len() - 2].clone();
    let n = dims_b[dims_b.len() - 1].clone();

    result_dims.push(m);
    result_dims.push(n);

    Ok(TensorShape::Dynamic(result_dims))
}

/// Infer Gather output shape.
fn gather_shape(data: &TensorShape, indices: &TensorShape) -> Result<TensorShape> {
    // Simplified: Gather along axis 0
    // Output is indices.shape + data.shape[1:]
    let data_dims = match data {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    let indices_dims = match indices {
        TensorShape::Static(dims) => dims
            .iter()
            .map(|&d| Dimension::Static(d))
            .collect::<Vec<_>>(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    let mut result = indices_dims;
    if data_dims.len() > 1 {
        result.extend_from_slice(&data_dims[1..]);
    }

    Ok(TensorShape::Dynamic(result))
}

/// Infer Transpose output shape.
fn transpose_shape(
    input: &TensorShape,
    attributes: &HashMap<String, crate::graph::AttributeValue>,
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get perm attribute if present, otherwise reverse all dims
    let perm = if let Some(crate::graph::AttributeValue::Ints(perm)) = attributes.get("perm") {
        perm.clone()
    } else {
        // Default: reverse all dimensions
        (0..dims.len() as i64).rev().collect()
    };

    let mut result = vec![Dimension::Static(0); dims.len()];
    for (i, &p) in perm.iter().enumerate() {
        if p < 0 || p >= dims.len() as i64 {
            return Err(OnnxError::ShapeInference(format!(
                "Invalid transpose permutation index: {}",
                p
            )));
        }
        result[i] = dims[p as usize].clone();
    }

    Ok(TensorShape::Dynamic(result))
}

/// Infer Unsqueeze output shape - adds dimensions at specified axes.
fn unsqueeze_shape(
    input: &TensorShape,
    inputs: &[String],
    attributes: &HashMap<String, crate::graph::AttributeValue>,
    context: &InferenceContext,
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    // In ONNX, Unsqueeze can have axes as an input (newer) or attribute (older)
    let axes = if inputs.len() >= 2 {
        // Try to get axes from constant evaluation
        if let Some(const_axes) = context.get_constant(&inputs[1]) {
            const_axes.clone()
        } else {
            // Can't infer without the constant value
            trace!(
                op_type = "Unsqueeze",
                axes_input = %inputs[1],
                "Cannot infer Unsqueeze output - axes constant not evaluated"
            );
            return Ok(TensorShape::Unknown);
        }
    } else if let Some(crate::graph::AttributeValue::Ints(axes)) = attributes.get("axes") {
        axes.clone()
    } else {
        return Err(OnnxError::ShapeInference(
            "Unsqueeze missing axes".to_string(),
        ));
    };

    // Add new dimensions at specified axes
    let mut result = dims.clone();
    let mut sorted_axes: Vec<i64> = axes.clone();
    sorted_axes.sort();

    for &axis in &sorted_axes {
        let pos = if axis < 0 {
            (result.len() as i64 + axis + 1) as usize
        } else {
            axis as usize
        };
        result.insert(pos, Dimension::Static(1));
    }

    Ok(TensorShape::Dynamic(result))
}

/// Infer Squeeze output shape - removes dimensions at specified axes.
fn squeeze_shape(
    input: &TensorShape,
    inputs: &[String],
    attributes: &HashMap<String, crate::graph::AttributeValue>,
    context: &InferenceContext,
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    let axes = if inputs.len() >= 2 {
        // Try to get axes from constant evaluation
        if let Some(const_axes) = context.get_constant(&inputs[1]) {
            const_axes.clone()
        } else {
            // Can't infer without the constant value
            trace!(
                op_type = "Squeeze",
                axes_input = %inputs[1],
                "Cannot infer Squeeze output - axes constant not evaluated"
            );
            return Ok(TensorShape::Unknown);
        }
    } else if let Some(crate::graph::AttributeValue::Ints(axes)) = attributes.get("axes") {
        axes.clone()
    } else {
        // No axes specified - squeeze all size-1 dimensions
        let mut result = Vec::new();
        for dim in dims {
            if dim != Dimension::Static(1) {
                result.push(dim);
            }
        }
        return Ok(TensorShape::Dynamic(result));
    };

    // Remove dimensions at specified axes (must be size 1)
    let mut result = dims.clone();
    let mut sorted_axes: Vec<i64> = axes.clone();
    sorted_axes.sort();
    sorted_axes.reverse(); // Remove from back to front to maintain indices

    for &axis in &sorted_axes {
        let pos = if axis < 0 {
            (result.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        if pos < result.len() {
            result.remove(pos);
        }
    }

    Ok(TensorShape::Dynamic(result))
}

/// Infer Concat output shape - concatenates along specified axis.
fn concat_shape(
    inputs: &[String],
    attributes: &HashMap<String, crate::graph::AttributeValue>,
    context: &InferenceContext,
) -> Result<TensorShape> {
    if inputs.is_empty() {
        return Err(OnnxError::ShapeInference(
            "Concat requires at least one input".to_string(),
        ));
    }

    // Get axis attribute
    let axis = if let Some(crate::graph::AttributeValue::Int(axis)) = attributes.get("axis") {
        *axis
    } else {
        return Err(OnnxError::ShapeInference(
            "Concat missing axis attribute".to_string(),
        ));
    };

    // Get first input shape as reference
    let first_shape = context
        .get_shape(&inputs[0])
        .ok_or_else(|| OnnxError::ShapeInference("Concat input shape unknown".to_string()))?;

    let mut result_dims = match first_shape {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    let rank = result_dims.len();
    let axis_pos = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    if axis_pos >= rank {
        return Err(OnnxError::ShapeInference(format!(
            "Concat axis {} out of range for rank {}",
            axis, rank
        )));
    }

    // Sum up the axis dimension from all inputs
    let mut axis_sum: Option<usize> = None;

    for input in inputs {
        let shape = context
            .get_shape(input)
            .ok_or_else(|| OnnxError::ShapeInference("Concat input shape unknown".to_string()))?;

        let dims = match shape {
            TensorShape::Static(dims) => dims
                .iter()
                .map(|&d| Dimension::Static(d))
                .collect::<Vec<_>>(),
            TensorShape::Dynamic(dims) => dims.clone(),
            TensorShape::Unknown => return Ok(TensorShape::Unknown),
        };

        // Check rank matches
        if dims.len() != rank {
            return Err(OnnxError::ShapeInference(
                "Concat inputs must have same rank".to_string(),
            ));
        }

        // Add axis dimension
        if let Dimension::Static(d) = dims[axis_pos] {
            axis_sum = Some(axis_sum.unwrap_or(0) + d);
        } else {
            // Dynamic dimension - can't compute exact result
            axis_sum = None;
            break;
        }
    }

    // Set result axis dimension
    if let Some(sum) = axis_sum {
        result_dims[axis_pos] = Dimension::Static(sum);
    } else {
        // Leave as first input's dimension (might be dynamic)
        result_dims[axis_pos] = match first_shape {
            TensorShape::Static(dims) => Dimension::Static(dims[axis_pos]),
            TensorShape::Dynamic(dims) => dims[axis_pos].clone(),
            TensorShape::Unknown => unreachable!(),
        };
    }

    Ok(TensorShape::Dynamic(result_dims))
}

/// Infer Reduce output shape - reduces along specified axes.
fn reduce_shape(
    input: &TensorShape,
    inputs: &[String],
    attributes: &HashMap<String, crate::graph::AttributeValue>,
    context: &InferenceContext,
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    let axes = if inputs.len() >= 2 {
        // Try to get axes from constant evaluation
        if let Some(const_axes) = context.get_constant(&inputs[1]) {
            const_axes.clone()
        } else {
            // Can't infer without the constant value
            trace!(
                op_type = "Reduce",
                axes_input = %inputs[1],
                "Cannot infer Reduce output - axes constant not evaluated"
            );
            return Ok(TensorShape::Unknown);
        }
    } else if let Some(crate::graph::AttributeValue::Ints(axes)) = attributes.get("axes") {
        axes.clone()
    } else {
        // No axes specified - reduce all dimensions
        let keep_dims =
            if let Some(crate::graph::AttributeValue::Int(keep)) = attributes.get("keepdims") {
                *keep != 0
            } else {
                true // Default is true
            };

        if keep_dims {
            return Ok(TensorShape::Dynamic(vec![Dimension::Static(1); dims.len()]));
        } else {
            return Ok(TensorShape::Static(vec![])); // Scalar
        }
    };

    // Check if we keep dimensions
    let keep_dims =
        if let Some(crate::graph::AttributeValue::Int(keep)) = attributes.get("keepdims") {
            *keep != 0
        } else {
            true // Default is true
        };

    // Process axes
    let mut result = dims.clone();
    let mut sorted_axes: Vec<i64> = axes.clone();
    sorted_axes.sort();

    if keep_dims {
        // Replace reduced dimensions with 1
        for &axis in &sorted_axes {
            let pos = if axis < 0 {
                (result.len() as i64 + axis) as usize
            } else {
                axis as usize
            };
            if pos < result.len() {
                result[pos] = Dimension::Static(1);
            }
        }
    } else {
        // Remove reduced dimensions
        sorted_axes.reverse(); // Remove from back to front
        for &axis in &sorted_axes {
            let pos = if axis < 0 {
                (result.len() as i64 + axis) as usize
            } else {
                axis as usize
            };
            if pos < result.len() {
                result.remove(pos);
            }
        }
    }

    Ok(TensorShape::Dynamic(result))
}

/// Decode constant initializer bytes to i64 values based on dtype.
fn decode_constant_bytes(initializer: &[u8], dtype: crate::graph::DataType) -> Option<Vec<i64>> {
    use crate::graph::DataType;

    match dtype {
        DataType::I64 => {
            if !initializer.len().is_multiple_of(8) {
                return None;
            }
            Some(
                initializer
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect(),
            )
        }
        DataType::I32 => {
            if !initializer.len().is_multiple_of(4) {
                return None;
            }
            Some(
                initializer
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        DataType::U32 => {
            if !initializer.len().is_multiple_of(4) {
                return None;
            }
            Some(
                initializer
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        DataType::U8 => Some(initializer.iter().map(|&b| b as i64).collect()),
        _ => None, // Don't decode floating-point types
    }
}

/// Resolve reshape target shape, handling -1 (inferred) dimension.
fn resolve_reshape_shape(input_shape: &TensorShape, target_shape: &[i64]) -> Result<TensorShape> {
    // Calculate total elements from input shape
    let input_elements = match input_shape {
        TensorShape::Static(dims) => {
            let total: usize = dims.iter().product();
            Some(total)
        }
        TensorShape::Dynamic(dims) => {
            // Try to compute if all dims are static
            let mut total: usize = 1;
            for dim in dims {
                match dim {
                    Dimension::Static(d) => total *= d,
                    Dimension::Named(_) => return Ok(TensorShape::Unknown),
                }
            }
            Some(total)
        }
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    let input_elements = input_elements.ok_or_else(|| {
        OnnxError::ShapeInference("Cannot resolve reshape: unknown input shape".to_string())
    })?;

    // Find the -1 dimension and compute product of other dimensions
    let mut neg_one_idx: Option<usize> = None;
    let mut other_product: usize = 1;

    for (i, &dim) in target_shape.iter().enumerate() {
        if dim == -1 {
            if neg_one_idx.is_some() {
                return Err(OnnxError::ShapeInference(
                    "Reshape: multiple -1 dimensions not allowed".to_string(),
                ));
            }
            neg_one_idx = Some(i);
        } else if dim == 0 {
            // 0 means copy from input shape at this position
            if i < match input_shape {
                TensorShape::Static(dims) => dims.len(),
                TensorShape::Dynamic(dims) => dims.len(),
                TensorShape::Unknown => 0,
            } {
                let input_dim = match input_shape {
                    TensorShape::Static(dims) => dims[i],
                    TensorShape::Dynamic(dims) => match &dims[i] {
                        Dimension::Static(d) => *d,
                        Dimension::Named(_) => return Ok(TensorShape::Unknown),
                    },
                    TensorShape::Unknown => return Ok(TensorShape::Unknown),
                };
                other_product *= input_dim;
            }
        } else if dim > 0 {
            other_product *= dim as usize;
        } else {
            return Err(OnnxError::ShapeInference(format!(
                "Reshape: invalid dimension {}",
                dim
            )));
        }
    }

    // Resolve the output shape
    let mut result: Vec<usize> = Vec::with_capacity(target_shape.len());
    for (i, &dim) in target_shape.iter().enumerate() {
        if dim == -1 {
            // Infer this dimension
            if other_product == 0 {
                return Err(OnnxError::ShapeInference(
                    "Reshape: cannot infer dimension with zero product".to_string(),
                ));
            }
            result.push(input_elements / other_product);
        } else if dim == 0 {
            // Copy from input
            if i < match input_shape {
                TensorShape::Static(dims) => dims.len(),
                TensorShape::Dynamic(dims) => dims.len(),
                TensorShape::Unknown => 0,
            } {
                let input_dim = match input_shape {
                    TensorShape::Static(dims) => dims[i],
                    TensorShape::Dynamic(dims) => match &dims[i] {
                        Dimension::Static(d) => *d,
                        Dimension::Named(_) => return Ok(TensorShape::Unknown),
                    },
                    TensorShape::Unknown => return Ok(TensorShape::Unknown),
                };
                result.push(input_dim);
            }
        } else {
            result.push(dim as usize);
        }
    }

    Ok(TensorShape::Static(result))
}

/// Evaluate gather on constant inputs.
fn evaluate_gather_constant(
    data: &[i64],
    indices: &[i64],
    axis: i64,
    data_shape: &[usize],
) -> Option<Vec<i64>> {
    // Handle 1D data case (most common for shape-related constants)
    if data_shape.len() == 1 && axis == 0 {
        let result: Option<Vec<i64>> = indices
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (data_shape[0] as i64 + idx) as usize
                } else {
                    idx as usize
                };
                data.get(actual_idx).copied()
            })
            .collect();
        return result;
    }
    // For more complex shapes, we'd need more sophisticated indexing
    None
}

/// Evaluate concat on constant inputs.
fn evaluate_concat_constant(inputs: &[&Vec<i64>], axis: i64) -> Option<Vec<i64>> {
    // For 1D constants (axis 0), just concatenate
    if axis == 0 {
        let result: Vec<i64> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        Some(result)
    } else {
        None // More complex shapes not supported yet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AttributeValue, DataType, Graph, Node, TensorInfo, TensorKind, TensorShape,
    };

    #[test]
    fn test_constant_shape_propagation() {
        let mut graph = Graph::new();

        // Create input tensor with known shape
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3, 4]),
            kind: TensorKind::Input,
            initializer: None,
        };
        graph.add_tensor(input);

        // Create output tensor for Shape op
        let output = TensorInfo {
            name: "shape_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Shape node
        let mut shape_node = Node::new("Shape");
        shape_node.inputs.push("input".to_string());
        shape_node.outputs.push("shape_out".to_string());
        graph.add_node(shape_node);

        graph.inputs.push("input".to_string());
        graph.outputs.push("shape_out".to_string());

        // Run shape inference
        infer_shapes(&mut graph).unwrap();

        // Check that shape_out has shape [3] (rank of input)
        let shape_out = graph.tensor_by_name("shape_out").unwrap();
        assert_eq!(shape_out.shape, TensorShape::Static(vec![3]));
    }

    #[test]
    fn test_reshape_with_constant_shape() {
        let mut graph = Graph::new();

        // Create input tensor [2, 6]
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 6]),
            kind: TensorKind::Input,
            initializer: None,
        };
        graph.add_tensor(input);

        // Create shape constant [3, 4]
        let shape_data: Vec<u8> = vec![3i64, 4i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let shape_tensor = TensorInfo {
            name: "shape".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(shape_data),
        };
        graph.add_tensor(shape_tensor);

        // Create output tensor
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Constant node for shape
        let mut const_node = Node::new("Constant");
        const_node.outputs.push("shape".to_string());
        graph.add_node(const_node);

        // Create Reshape node
        let mut reshape_node = Node::new("Reshape");
        reshape_node.inputs.push("input".to_string());
        reshape_node.inputs.push("shape".to_string());
        reshape_node.outputs.push("output".to_string());
        graph.add_node(reshape_node);

        graph.inputs.push("input".to_string());
        graph.outputs.push("output".to_string());

        // Run inference
        infer_shapes(&mut graph).unwrap();

        // Check output shape
        let output_info = graph.tensor_by_name("output").unwrap();
        assert_eq!(output_info.shape, TensorShape::Static(vec![3, 4]));
    }

    #[test]
    fn test_reshape_with_negative_one() {
        let mut graph = Graph::new();

        // Create input tensor [2, 3, 4] = 24 elements
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3, 4]),
            kind: TensorKind::Input,
            initializer: None,
        };
        graph.add_tensor(input);

        // Create shape constant [-1, 6] -> should resolve to [4, 6]
        let shape_data: Vec<u8> = vec![-1i64, 6i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let shape_tensor = TensorInfo {
            name: "shape".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(shape_data),
        };
        graph.add_tensor(shape_tensor);

        // Create output tensor
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Constant node for shape
        let mut const_node = Node::new("Constant");
        const_node.outputs.push("shape".to_string());
        graph.add_node(const_node);

        // Create Reshape node
        let mut reshape_node = Node::new("Reshape");
        reshape_node.inputs.push("input".to_string());
        reshape_node.inputs.push("shape".to_string());
        reshape_node.outputs.push("output".to_string());
        graph.add_node(reshape_node);

        graph.inputs.push("input".to_string());
        graph.outputs.push("output".to_string());

        // Run inference
        infer_shapes(&mut graph).unwrap();

        // Check output shape is [4, 6] (24 / 6 = 4)
        let output_info = graph.tensor_by_name("output").unwrap();
        assert_eq!(output_info.shape, TensorShape::Static(vec![4, 6]));
    }

    #[test]
    fn test_unsqueeze_with_constant_axes() {
        let mut graph = Graph::new();

        // Create input tensor [2, 3]
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        };
        graph.add_tensor(input);

        // Create axes constant [0, 2] -> output should be [1, 2, 1, 3]
        let axes_data: Vec<u8> = vec![0i64, 2i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let axes_tensor = TensorInfo {
            name: "axes".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(axes_data),
        };
        graph.add_tensor(axes_tensor);

        // Create output tensor
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Constant node for axes
        let mut const_node = Node::new("Constant");
        const_node.outputs.push("axes".to_string());
        graph.add_node(const_node);

        // Create Unsqueeze node
        let mut unsqueeze_node = Node::new("Unsqueeze");
        unsqueeze_node.inputs.push("input".to_string());
        unsqueeze_node.inputs.push("axes".to_string());
        unsqueeze_node.outputs.push("output".to_string());
        graph.add_node(unsqueeze_node);

        graph.inputs.push("input".to_string());
        graph.outputs.push("output".to_string());

        // Run inference
        infer_shapes(&mut graph).unwrap();

        // Check output shape is [1, 2, 1, 3]
        let output_info = graph.tensor_by_name("output").unwrap();
        match &output_info.shape {
            TensorShape::Dynamic(dims) => {
                let static_dims: Vec<usize> = dims
                    .iter()
                    .map(|d| match d {
                        Dimension::Static(s) => *s,
                        _ => panic!("Expected static dimensions"),
                    })
                    .collect();
                assert_eq!(static_dims, vec![1, 2, 1, 3]);
            }
            other => panic!("Expected Dynamic shape, got {:?}", other),
        }
    }

    #[test]
    fn test_gather_constant_propagation() {
        let mut graph = Graph::new();

        // Create constant data [10, 20, 30, 40]
        let data_bytes: Vec<u8> = vec![10i64, 20, 30, 40]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let data_tensor = TensorInfo {
            name: "data".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Weight,
            initializer: Some(data_bytes),
        };
        graph.add_tensor(data_tensor);

        // Create constant indices [1, 3]
        let indices_bytes: Vec<u8> = vec![1i64, 3i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let indices_tensor = TensorInfo {
            name: "indices".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(indices_bytes),
        };
        graph.add_tensor(indices_tensor);

        // Create output tensor
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Constant nodes
        let mut const_data_node = Node::new("Constant");
        const_data_node.outputs.push("data".to_string());
        graph.add_node(const_data_node);

        let mut const_indices_node = Node::new("Constant");
        const_indices_node.outputs.push("indices".to_string());
        graph.add_node(const_indices_node);

        // Create Gather node
        let mut gather_node = Node::new("Gather");
        gather_node.inputs.push("data".to_string());
        gather_node.inputs.push("indices".to_string());
        gather_node.outputs.push("output".to_string());
        gather_node
            .attributes
            .insert("axis".to_string(), AttributeValue::Int(0));
        graph.add_node(gather_node);

        graph.outputs.push("output".to_string());

        // Run inference using InferenceContext to check constants
        let mut context = InferenceContext::new(&graph);
        for node in &graph.nodes {
            infer_node_output_shapes(&mut context, node, &graph).unwrap();
        }

        // Check that output constant is [20, 40]
        let output_const = context.get_constant("output").unwrap();
        assert_eq!(output_const, &vec![20, 40]);
    }

    #[test]
    fn test_concat_constant_propagation() {
        let mut graph = Graph::new();

        // Create constant a [1, 2]
        let a_bytes: Vec<u8> = vec![1i64, 2i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let a_tensor = TensorInfo {
            name: "a".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(a_bytes),
        };
        graph.add_tensor(a_tensor);

        // Create constant b [3, 4, 5]
        let b_bytes: Vec<u8> = vec![3i64, 4i64, 5i64]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let b_tensor = TensorInfo {
            name: "b".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![3]),
            kind: TensorKind::Weight,
            initializer: Some(b_bytes),
        };
        graph.add_tensor(b_tensor);

        // Create output tensor
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(output);

        // Create Constant nodes
        let mut const_a = Node::new("Constant");
        const_a.outputs.push("a".to_string());
        graph.add_node(const_a);

        let mut const_b = Node::new("Constant");
        const_b.outputs.push("b".to_string());
        graph.add_node(const_b);

        // Create Concat node
        let mut concat_node = Node::new("Concat");
        concat_node.inputs.push("a".to_string());
        concat_node.inputs.push("b".to_string());
        concat_node.outputs.push("output".to_string());
        concat_node
            .attributes
            .insert("axis".to_string(), AttributeValue::Int(0));
        graph.add_node(concat_node);

        graph.outputs.push("output".to_string());

        // Run inference
        let mut context = InferenceContext::new(&graph);
        for node in &graph.nodes {
            infer_node_output_shapes(&mut context, node, &graph).unwrap();
        }

        // Check that output constant is [1, 2, 3, 4, 5]
        let output_const = context.get_constant("output").unwrap();
        assert_eq!(output_const, &vec![1, 2, 3, 4, 5]);

        // Also check output shape is [5]
        let output_shape = context.get_shape("output").unwrap();
        match output_shape {
            TensorShape::Dynamic(dims) => {
                assert_eq!(dims.len(), 1);
                assert_eq!(dims[0], Dimension::Static(5));
            }
            _ => panic!("Expected Dynamic shape"),
        }
    }

    #[test]
    fn test_constant_chain_shape_gather_unsqueeze_concat_reshape() {
        // Test the full chain: Constant → Shape → Gather → Concat → Reshape
        // (Simplified: skip Unsqueeze since Gather already produces [1] shape)
        let mut graph = Graph::new();

        // Input tensor with shape [2, 3, 4]
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3, 4]),
            kind: TensorKind::Input,
            initializer: None,
        };
        graph.add_tensor(input);

        // Shape output
        let shape_out = TensorInfo {
            name: "shape_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(shape_out);

        // Gather indices (constant for index 0) - scalar
        let indices_bytes: Vec<u8> = vec![0i64].iter().flat_map(|&v| v.to_le_bytes()).collect();
        let indices = TensorInfo {
            name: "indices".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(indices_bytes),
        };
        graph.add_tensor(indices);

        // Gather output - shape [1] containing [2]
        let gather_out = TensorInfo {
            name: "gather_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(gather_out);

        // Concat second input constant [-1]
        let neg_one_bytes: Vec<u8> = vec![-1i64].iter().flat_map(|&v| v.to_le_bytes()).collect();
        let neg_one = TensorInfo {
            name: "neg_one".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Static(vec![1]),
            kind: TensorKind::Weight,
            initializer: Some(neg_one_bytes),
        };
        graph.add_tensor(neg_one);

        // Concat output
        let concat_out = TensorInfo {
            name: "concat_out".to_string(),
            dtype: DataType::I64,
            shape: TensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(concat_out);

        // Final reshape output
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Unknown,
            kind: TensorKind::Output,
            initializer: None,
        };
        graph.add_tensor(output);

        // Build the nodes in order
        // 1. Constant for indices
        let mut const_indices = Node::new("Constant");
        const_indices.outputs.push("indices".to_string());
        graph.add_node(const_indices);

        // 2. Constant for [-1]
        let mut const_neg_one = Node::new("Constant");
        const_neg_one.outputs.push("neg_one".to_string());
        graph.add_node(const_neg_one);

        // 3. Shape node
        let mut shape_node = Node::new("Shape");
        shape_node.inputs.push("input".to_string());
        shape_node.outputs.push("shape_out".to_string());
        graph.add_node(shape_node);

        // 4. Gather node
        let mut gather_node = Node::new("Gather");
        gather_node.inputs.push("shape_out".to_string());
        gather_node.inputs.push("indices".to_string());
        gather_node.outputs.push("gather_out".to_string());
        gather_node
            .attributes
            .insert("axis".to_string(), AttributeValue::Int(0));
        graph.add_node(gather_node);

        // 5. Concat node: gather_out ([2]) and neg_one ([-1]) -> [2, -1]
        let mut concat_node = Node::new("Concat");
        concat_node.inputs.push("gather_out".to_string());
        concat_node.inputs.push("neg_one".to_string());
        concat_node.outputs.push("concat_out".to_string());
        concat_node
            .attributes
            .insert("axis".to_string(), AttributeValue::Int(0));
        graph.add_node(concat_node);

        // 6. Reshape node
        let mut reshape_node = Node::new("Reshape");
        reshape_node.inputs.push("input".to_string());
        reshape_node.inputs.push("concat_out".to_string());
        reshape_node.outputs.push("output".to_string());
        graph.add_node(reshape_node);

        graph.inputs.push("input".to_string());
        graph.outputs.push("output".to_string());

        // Run shape inference
        infer_shapes(&mut graph).unwrap();

        // The chain should resolve:
        // Shape([2,3,4]) = [2,3,4]
        // Gather([2,3,4], [0]) = [2]
        // Concat([2], [-1]) = [2, -1]
        // Reshape([2,3,4], [2, -1]) = [2, 12]

        let output_info = graph.tensor_by_name("output").unwrap();
        assert_eq!(output_info.shape, TensorShape::Static(vec![2, 12]));
    }

    #[test]
    fn test_decode_constant_bytes_i64() {
        let values = vec![1i64, -2i64, 3i64];
        let bytes: Vec<u8> = values.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let decoded = decode_constant_bytes(&bytes, DataType::I64).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_constant_bytes_i32() {
        let values = vec![1i32, -2i32, 3i32];
        let bytes: Vec<u8> = values.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let decoded = decode_constant_bytes(&bytes, DataType::I32).unwrap();
        assert_eq!(decoded, vec![1i64, -2i64, 3i64]);
    }

    #[test]
    fn test_resolve_reshape_shape_basic() {
        let input_shape = TensorShape::Static(vec![2, 3, 4]);
        let target = vec![6i64, 4i64];
        let result = resolve_reshape_shape(&input_shape, &target).unwrap();
        assert_eq!(result, TensorShape::Static(vec![6, 4]));
    }

    #[test]
    fn test_resolve_reshape_shape_with_neg_one() {
        let input_shape = TensorShape::Static(vec![2, 3, 4]); // 24 elements
        let target = vec![-1i64, 8i64];
        let result = resolve_reshape_shape(&input_shape, &target).unwrap();
        assert_eq!(result, TensorShape::Static(vec![3, 8])); // 24 / 8 = 3
    }

    #[test]
    fn test_gemma_unknown_shape_reduction() {
        // Skip test if model file is not present
        let model_path = "../../models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx";
        if !std::path::Path::new(model_path).exists() {
            eprintln!(
                "Skipping test: Gemma model file not found at {}",
                model_path
            );
            return;
        }

        // Load the model
        let model = crate::load_model(model_path).expect("Failed to load Gemma model");
        let graph = crate::parser::parse_model(&model).expect("Failed to parse Gemma model");

        // Count unknown shapes
        let unknown_count = graph
            .tensor_info
            .iter()
            .filter(|t| matches!(t.shape, TensorShape::Unknown))
            .count();

        // Before constant evaluation, there were 435 unknown shapes
        // After, we expect fewer (significantly reduced)
        let total_tensors = graph.tensor_info.len();

        eprintln!(
            "Gemma model: {}/{} tensors have unknown shape ({:.1}%)",
            unknown_count,
            total_tensors,
            (unknown_count as f64 / total_tensors as f64) * 100.0
        );

        // We expect the unknown count to be reduced from 435
        // After constant evaluation, many Reshape outputs should have known shapes
        assert!(
            unknown_count < 435,
            "Expected fewer than 435 unknown shapes after constant evaluation, got {}",
            unknown_count
        );
    }
}
