//! Shape inference for ONNX graphs.
//!
//! Propagates shape information through the graph using operation-specific inference rules.

use crate::graph::{Dimension, Graph, Node, TensorShape};
use crate::{OnnxError, Result};
use std::collections::HashMap;

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
        let constants = HashMap::new(); // Reserved for future constant evaluation

        // Initialize with known shapes from graph
        for (name, &tensor_id) in &graph.tensors {
            let info = &graph.tensor_info[tensor_id];
            if !matches!(info.shape, TensorShape::Unknown) {
                shapes.insert(name.clone(), info.shape.clone());
            }

            // If this is a weight tensor with initializer data, it might be a constant we can evaluate
            // For now, we'll handle explicit Constant node outputs when processing nodes
        }

        Self { shapes, constants }
    }

    fn get_shape(&self, name: &str) -> Option<&TensorShape> {
        self.shapes.get(name)
    }

    fn set_shape(&mut self, name: String, shape: TensorShape) {
        self.shapes.insert(name, shape);
    }

    #[allow(dead_code)] // Reserved for future constant evaluation
    fn get_constant(&self, name: &str) -> Option<&Vec<i64>> {
        self.constants.get(name)
    }

    #[allow(dead_code)] // Reserved for future constant evaluation
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
fn infer_node_output_shapes(
    context: &mut InferenceContext,
    node: &Node,
    __graph: &Graph,
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
                    let rank = match input_shape {
                        TensorShape::Static(dims) => dims.len(),
                        TensorShape::Dynamic(dims) => dims.len(),
                        TensorShape::Unknown => {
                            return Ok(()); // Can't infer without knowing input shape
                        }
                    };
                    for output in &node.outputs {
                        context.set_shape(output.clone(), TensorShape::Static(vec![rank]));
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
        }

        // Reshape: output shape from second input (constant) or attributes
        "Reshape" => {
            // Shape is typically provided as the second input (a constant)
            // For now, mark as unknown - would need constant evaluation
            // TODO: Evaluate constant tensors to get actual reshape dimensions
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

        // Constant: shape extracted at parse time
        "Constant" => {
            // Already handled by parser - shape extracted from tensor attribute
        }

        // Unsqueeze: add dimensions at specified axes
        "Unsqueeze" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape =
                        unsqueeze_shape(input_shape, &node.inputs, &node.attributes)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
                    }
                }
            }
        }

        // Squeeze: remove dimensions at specified axes
        "Squeeze" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape = squeeze_shape(input_shape, &node.inputs, &node.attributes)?;
                    for output in &node.outputs {
                        context.set_shape(output.clone(), result_shape.clone());
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
        }

        // ReduceSum, ReduceMean: reduce along specified axes
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceMin" => {
            if let Some(input) = node.inputs.first() {
                if let Some(input_shape) = context.get_shape(input) {
                    let result_shape = reduce_shape(input_shape, &node.inputs, &node.attributes)?;
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
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    // In ONNX, Unsqueeze can have axes as an input (newer) or attribute (older)
    let axes = if inputs.len() >= 2 {
        // Axes as input - would need constant evaluation
        // For now, can't infer without evaluating the constant
        return Ok(TensorShape::Unknown);
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
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    let axes = if inputs.len() >= 2 {
        // Axes as input - would need constant evaluation
        return Ok(TensorShape::Unknown);
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
) -> Result<TensorShape> {
    let dims = match input {
        TensorShape::Static(dims) => dims.iter().map(|&d| Dimension::Static(d)).collect(),
        TensorShape::Dynamic(dims) => dims.clone(),
        TensorShape::Unknown => return Ok(TensorShape::Unknown),
    };

    // Get axes from second input or from attributes
    let axes = if inputs.len() >= 2 {
        // Axes as input - would need constant evaluation
        return Ok(TensorShape::Unknown);
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
