//! Shape inference pass.
//!
//! Forward pass in topological order that calls `operator.infer_output_shapes()` for
//! each node to propagate shapes through the graph.

use onyxia_core::{Error, InferenceCtx, IrGraph, IrNode, OperatorRegistry, Pass, Result, Stage};

/// Pass that infers output shapes for all operations in the graph.
///
/// Walks the graph in topological order, calling `Operator::infer_output_shapes()`
/// for each node. Skips nodes that were fully folded in the previous stage, since
/// their output shapes are already determined by the folded values.
///
/// This pass runs after constant folding, which means chains like
/// Shape→Gather→Concat→Reshape can be completely folded before shape inference
/// runs, eliminating unnecessary shape inference calls.
pub struct ShapeInferencePass;

impl ShapeInferencePass {
    /// Create a new shape inference pass.
    pub fn new() -> Self {
        Self
    }

    /// Infer shapes for a single node.
    fn infer_node(
        &self,
        node: &IrNode,
        graph: &mut IrGraph,
        registry: &OperatorRegistry,
    ) -> Result<bool> {
        // Look up operator
        let operator = registry.get(&node.op_type).ok_or_else(|| {
            Error::ShapeInference(format!("No operator registered for type: {}", node.op_type))
        })?;

        // Build inference context
        let ctx = InferenceCtx::new(node, graph);

        // Call operator's shape inference
        let output_shapes = operator.infer_output_shapes(&ctx).map_err(|e| {
            Error::ShapeInference(format!(
                "Failed to infer shapes for node '{}' (op_type: {}): {}",
                node.op_type, node.op_type, e
            ))
        })?;

        // Validate output count
        if output_shapes.len() != node.outputs.len() {
            return Err(Error::ShapeInference(format!(
                "Operator {} returned {} output shapes but node has {} outputs",
                node.op_type,
                output_shapes.len(),
                node.outputs.len()
            )));
        }

        // Update output tensor shapes
        let mut changed = false;
        for (i, new_shape) in output_shapes.into_iter().enumerate() {
            let tensor_id = node.outputs[i];
            let tensor = graph.tensor(tensor_id)?;
            let old_shape = tensor.shape.clone();

            if new_shape != old_shape {
                graph.tensor_mut(tensor_id)?.shape = new_shape;
                changed = true;
            }
        }

        Ok(changed)
    }
}

impl Pass for ShapeInferencePass {
    fn name(&self) -> &str {
        "shape_inference"
    }

    fn stage(&self) -> Stage {
        Stage::Inference
    }

    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool> {
        let mut changed = false;

        // Process nodes in topological order
        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node = graph.node(node_id)?.clone();

            // Skip nodes that are fully folded (all outputs have constant values)
            // Their shapes are already determined by the folded values
            let all_outputs_folded = node.outputs.iter().all(|&tensor_id| {
                graph
                    .tensor(tensor_id)
                    .map(|t| t.has_value())
                    .unwrap_or(false)
            });

            if all_outputs_folded {
                continue;
            }

            let node_changed = self.infer_node(&node, graph, registry)?;
            changed = changed || node_changed;
        }

        Ok(changed)
    }
}

impl Default for ShapeInferencePass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::{DataType, IrGraph, IrNode, Operator, TensorDef, TensorKind, TensorShape};

    // Mock operator that adds dimensions
    struct MockAddOperator;

    impl Operator for MockAddOperator {
        fn name(&self) -> &str {
            "MockAdd"
        }

        fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
            // Just return the shape of the first input
            Ok(vec![ctx.input_shape(0)?.clone()])
        }

        fn plan(&self, _ctx: &mut onyxia_core::PlanCtx) -> Result<Vec<onyxia_core::Step>> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_shape_inference_propagates_shapes() {
        let mut graph = IrGraph::new();

        // Create input tensor with known shape
        let input = TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2, 3]),
            TensorKind::Input,
        );
        let input_id = graph.add_tensor(input);

        // Create output tensor with unknown shape (will be inferred)
        let output = TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![]), // Empty shape to be inferred
            TensorKind::Intermediate,
        );
        let output_id = graph.add_tensor(output);

        // Add node
        let mut node = IrNode::new("MockAdd".to_string());
        node.add_input(input_id);
        node.add_output(output_id);
        graph.add_node(node);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        // Run shape inference pass
        let pass = ShapeInferencePass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);
        assert_eq!(
            graph.tensor(output_id).unwrap().shape,
            TensorShape::Static(vec![1, 2, 3])
        );
    }

    #[test]
    fn test_shape_inference_error_on_missing_operator() {
        let mut graph = IrGraph::new();

        let input_id = graph.add_tensor(TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let output_id = graph.add_tensor(TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![]),
            TensorKind::Intermediate,
        ));

        let mut node = IrNode::new("UnknownOp".to_string());
        node.add_input(input_id);
        node.add_output(output_id);
        graph.add_node(node);

        let registry = OperatorRegistry::new();
        let pass = ShapeInferencePass::new();

        let result = pass.run(&mut graph, &registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No operator"));
    }

    #[test]
    fn test_shape_inference_multi_node_graph() {
        let mut graph = IrGraph::new();

        // Create a chain: input -> node1 -> intermediate -> node2 -> output
        let input_id = graph.add_tensor(TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![4, 8]),
            TensorKind::Input,
        ));

        let intermediate_id = graph.add_tensor(TensorDef::new(
            "intermediate".to_string(),
            DataType::F32,
            TensorShape::Static(vec![]),
            TensorKind::Intermediate,
        ));

        let output_id = graph.add_tensor(TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![]),
            TensorKind::Output,
        ));

        // Node 1: input -> intermediate
        let mut node1 = IrNode::new("MockAdd".to_string());
        node1.add_input(input_id);
        node1.add_output(intermediate_id);
        graph.add_node(node1);

        // Node 2: intermediate -> output
        let mut node2 = IrNode::new("MockAdd".to_string());
        node2.add_input(intermediate_id);
        node2.add_output(output_id);
        graph.add_node(node2);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        // Run shape inference pass
        let pass = ShapeInferencePass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);
        assert_eq!(
            graph.tensor(intermediate_id).unwrap().shape,
            TensorShape::Static(vec![4, 8])
        );
        assert_eq!(
            graph.tensor(output_id).unwrap().shape,
            TensorShape::Static(vec![4, 8])
        );
    }
}
