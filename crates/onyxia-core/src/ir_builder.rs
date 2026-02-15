//! Build IR graphs from ONNX graphs.

use crate::ir::{IrGraph, IrNode, IrTensorId, TensorDef};
use crate::types::TensorShape;
use crate::{Error, Result};
use onyxia_onnx::Graph;
use std::collections::HashMap;

impl IrGraph {
    /// Build an IR graph from an ONNX graph.
    ///
    /// This converts the ONNX graph structure into the IR representation suitable
    /// for optimization passes and planning. The IR is freshly built (not cloned)
    /// and uses `petgraph::StableGraph` for efficient mutation during passes.
    ///
    /// # Process
    ///
    /// 1. Convert all ONNX tensors to `TensorDef` with shapes preserved as-is
    /// 2. Convert all ONNX nodes to `IrNode` with attributes copied
    /// 3. Build producer/consumer lookup tables
    /// 4. Preserve input/output lists
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A node references a tensor that doesn't exist
    /// - ONNX shape is `Unknown` (forces explicit error handling)
    /// - The ONNX graph is malformed
    pub fn from_onnx(onnx_graph: &Graph) -> Result<Self> {
        let mut ir_graph = IrGraph::new();

        // Step 1: Convert all tensors
        let mut tensor_map: HashMap<String, IrTensorId> = HashMap::new();

        for (tensor_name, &onnx_tensor_id) in &onnx_graph.tensors {
            let onnx_tensor = onnx_graph.tensor(onnx_tensor_id).map_err(|e| {
                Error::InvalidGraph(format!("Failed to get tensor {}: {}", tensor_name, e))
            })?;

            // Convert ONNX shape to core TensorShape
            // This will error if the shape is Unknown, forcing explicit handling
            let shape = TensorShape::from_onnx(&onnx_tensor.shape)?;

            let mut tensor_def = TensorDef::new(
                onnx_tensor.name.clone(),
                onnx_tensor.dtype,
                shape,
                onnx_tensor.kind,
            );

            // Copy initializer data if present
            if let Some(ref initializer_data) = onnx_tensor.initializer {
                tensor_def.initializer = Some(initializer_data.clone());
            }

            let tensor_id = ir_graph.add_tensor(tensor_def);
            tensor_map.insert(tensor_name.clone(), tensor_id);
        }

        // Step 2: Convert all nodes
        for onnx_node in &onnx_graph.nodes {
            let mut ir_node = IrNode::new(onnx_node.op_type.clone());

            // Convert input tensor names to IDs
            for input_name in &onnx_node.inputs {
                // ONNX uses empty strings for optional absent inputs
                if input_name.is_empty() {
                    continue;
                }

                let tensor_id = tensor_map.get(input_name).ok_or_else(|| {
                    Error::InvalidGraph(format!(
                        "Node {} references unknown input tensor: {}",
                        onnx_node.name, input_name
                    ))
                })?;

                ir_node.add_tensor_input(*tensor_id)?;
            }

            // Convert output tensor names to IDs
            for output_name in &onnx_node.outputs {
                if output_name.is_empty() {
                    continue;
                }

                let tensor_id = tensor_map.get(output_name).ok_or_else(|| {
                    Error::InvalidGraph(format!(
                        "Node {} references unknown output tensor: {}",
                        onnx_node.name, output_name
                    ))
                })?;

                ir_node.add_output(*tensor_id)?;
            }

            // Copy attributes
            for (key, value) in onnx_node.attributes.clone() {
                ir_node.set_attribute(key, value)?;
            }

            // Add node to graph (this also updates producer/consumer tables)
            ir_graph.add_node(ir_node);
        }

        // Step 3: Set input/output lists
        for input_name in &onnx_graph.inputs {
            let tensor_id = tensor_map.get(input_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Graph input not found: {}", input_name))
            })?;
            ir_graph.inputs.push(*tensor_id);
        }

        for output_name in &onnx_graph.outputs {
            let tensor_id = tensor_map.get(output_name).ok_or_else(|| {
                Error::InvalidGraph(format!("Graph output not found: {}", output_name))
            })?;
            ir_graph.outputs.push(*tensor_id);
        }

        Ok(ir_graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape as OnnxTensorShape};

    #[test]
    fn test_from_onnx_simple_graph() {
        // Create a simple ONNX graph: input -> Relu -> output
        let mut onnx_graph = Graph::new();

        // Add tensors
        let input_tensor = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![1, 2, 3]),
            kind: TensorKind::Input,
            initializer: None,
        };

        let output_tensor = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![1, 2, 3]),
            kind: TensorKind::Output,
            initializer: None,
        };

        onnx_graph.add_tensor(input_tensor);
        onnx_graph.add_tensor(output_tensor);

        // Add node
        let mut node = Node::new("Relu");
        node.name = "relu_node".to_string();
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];
        onnx_graph.add_node(node);

        // Set inputs/outputs
        onnx_graph.inputs = vec!["input".to_string()];
        onnx_graph.outputs = vec!["output".to_string()];

        // Convert to IR
        let ir_graph = IrGraph::from_onnx(&onnx_graph).unwrap();

        assert_eq!(ir_graph.tensor_count(), 2);
        assert_eq!(ir_graph.node_count(), 1);
        assert_eq!(ir_graph.inputs.len(), 1);
        assert_eq!(ir_graph.outputs.len(), 1);

        // Check tensor properties
        let input_id = ir_graph.inputs[0];
        let input_tensor = ir_graph.tensor(input_id).unwrap();
        assert_eq!(input_tensor.name, "input");
        assert_eq!(input_tensor.dtype, DataType::F32);
        assert!(input_tensor.shape.is_static());

        // Check node properties
        let nodes: Vec<_> = ir_graph.nodes().collect();
        assert_eq!(nodes.len(), 1);
        let (node_id, node) = nodes[0];
        assert_eq!(node.op_type(), Some("Relu"));
        assert_eq!(node.inputs().len(), 1);
        assert_eq!(node.outputs().len(), 1);

        // Check producer/consumer relationships
        let output_id = ir_graph.outputs[0];
        assert_eq!(ir_graph.tensor_producer(output_id), Some(node_id));
        assert_eq!(ir_graph.tensor_consumers(input_id), vec![node_id]);
    }

    #[test]
    fn test_from_onnx_chain() {
        // Create a chain: input -> A -> intermediate -> B -> output
        let mut onnx_graph = Graph::new();

        onnx_graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![2, 2]),
            kind: TensorKind::Input,
            initializer: None,
        });

        onnx_graph.add_tensor(TensorInfo {
            name: "intermediate".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![2, 2]),
            kind: TensorKind::Intermediate,
            initializer: None,
        });

        onnx_graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![2, 2]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node_a = Node::new("Add");
        node_a.inputs = vec!["input".to_string()];
        node_a.outputs = vec!["intermediate".to_string()];
        onnx_graph.add_node(node_a);

        let mut node_b = Node::new("Mul");
        node_b.inputs = vec!["intermediate".to_string()];
        node_b.outputs = vec!["output".to_string()];
        onnx_graph.add_node(node_b);

        onnx_graph.inputs = vec!["input".to_string()];
        onnx_graph.outputs = vec!["output".to_string()];

        let ir_graph = IrGraph::from_onnx(&onnx_graph).unwrap();

        assert_eq!(ir_graph.tensor_count(), 3);
        assert_eq!(ir_graph.node_count(), 2);

        // Check topological order
        let topo_order = ir_graph.topological_order();
        assert_eq!(topo_order.len(), 2);

        // First node should be Add, second should be Mul
        let node_a = ir_graph.node(topo_order[0]).unwrap();
        let node_b = ir_graph.node(topo_order[1]).unwrap();
        assert_eq!(node_a.op_type(), Some("Add"));
        assert_eq!(node_b.op_type(), Some("Mul"));
    }

    #[test]
    fn test_from_onnx_with_initializer() {
        let mut onnx_graph = Graph::new();

        // Add a constant tensor with initializer data
        let constant_tensor = TensorInfo {
            name: "constant".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Static(vec![2]),
            kind: TensorKind::Weight,
            initializer: Some(vec![0, 0, 128, 63, 0, 0, 0, 64]), // 1.0f32, 2.0f32
        };

        onnx_graph.add_tensor(constant_tensor);

        let ir_graph = IrGraph::from_onnx(&onnx_graph).unwrap();

        let constant_id = ir_graph.tensor_by_name("constant").unwrap();
        let constant_tensor = ir_graph.tensor(constant_id).unwrap();

        assert!(constant_tensor.has_initializer());
        assert_eq!(constant_tensor.initializer.as_ref().unwrap().len(), 8);
    }

    #[test]
    fn test_from_onnx_unknown_shape_error() {
        let mut onnx_graph = Graph::new();

        // Add a tensor with Unknown shape
        let unknown_tensor = TensorInfo {
            name: "unknown".to_string(),
            dtype: DataType::F32,
            shape: OnnxTensorShape::Unknown,
            kind: TensorKind::Intermediate,
            initializer: None,
        };

        onnx_graph.add_tensor(unknown_tensor);

        // Should return an error because Unknown shapes are not allowed
        let result = IrGraph::from_onnx(&onnx_graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_onnx_missing_tensor() {
        let mut onnx_graph = Graph::new();

        // Add a node that references a non-existent tensor
        let mut node = Node::new("Relu");
        node.inputs = vec!["missing_tensor".to_string()];
        node.outputs = vec!["output".to_string()];
        onnx_graph.add_node(node);

        // Should return an error
        let result = IrGraph::from_onnx(&onnx_graph);
        assert!(result.is_err());
    }
}
