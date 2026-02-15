//! Constant folding pass.
//!
//! Forward pass in topological order that evaluates operations at compile time when
//! all inputs are known constants.
//!
//! Replaces operators with Value nodes containing the folded results.

use onyxia_core::{Error, FoldCtx, IrGraph, IrNodeId, OperatorRegistry, Pass, Result, Stage};

/// Pass that folds constant operations at compile time.
///
/// Walks the graph in topological order. For each node:
/// 1. Parses initializers on-demand through FoldCtx
/// 2. If all inputs have values, calls `operator.try_fold()` to evaluate at compile time
/// 3. Replaces the operator node with a Value node containing the folded result
///
/// This enables chains like Shape→Gather→Concat→Reshape to be fully resolved at
/// compile time, eliminating unnecessary GPU operations.
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    /// Create a new constant folding pass.
    pub fn new() -> Self {
        Self
    }

    /// Attempt to fold a single node.
    fn fold_node(
        &self,
        node_id: IrNodeId,
        graph: &mut IrGraph,
        registry: &OperatorRegistry,
    ) -> Result<bool> {
        let node = graph.node(node_id)?.clone();

        // Skip Value nodes (already folded)
        let (op_type, _, _, outputs) = match node.as_operator() {
            Some(op) => op,
            None => return Ok(false),
        };

        // Look up operator
        let operator = registry.get(op_type).ok_or_else(|| {
            Error::ConstantFolding(format!("No operator registered for type: {}", op_type))
        })?;

        // Build fold context
        let ctx = FoldCtx::new(&node, graph);

        // Call operator's constant folding
        let folded_outputs = operator.try_fold(&ctx).map_err(|e| {
            Error::ConstantFolding(format!(
                "Failed to fold node '{}' (op_type: {}): {}",
                op_type, op_type, e
            ))
        })?;

        // If operator returned no folded values, skip
        if folded_outputs.is_empty() {
            return Ok(false);
        }

        // Validate output count
        if folded_outputs.len() != outputs.len() {
            return Err(Error::ConstantFolding(format!(
                "Operator {} returned {} folded outputs but node has {} outputs",
                op_type,
                folded_outputs.len(),
                outputs.len()
            )));
        }

        // Replace node with Value node for each successfully folded output
        // For now, we only handle single-output operators (most common case)
        if outputs.len() == 1 {
            if let Some(value) = folded_outputs.into_iter().next().unwrap() {
                graph.replace_single_output_with_value(node_id, value)?;
                return Ok(true);
            }
        } else {
            // Multi-output operators are more complex
            // We'd need to create multiple value nodes or handle differently
            // For now, skip (can be added later if needed)
            // Note: Skipping constant folding for multi-output operator
        }

        Ok(false)
    }
}

impl Pass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn stage(&self) -> Stage {
        Stage::Folding
    }

    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool> {
        let mut changed = false;

        // Process nodes in topological order
        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node_changed = self.fold_node(node_id, graph, registry)?;
            changed = changed || node_changed;
        }

        Ok(changed)
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::ir::IrInput;
    use onyxia_core::{
        DataType, IrGraph, IrNode, Operator, TensorDef, TensorKind, TensorShape, TensorValue,
    };

    // Mock operator that folds addition
    struct MockAddOperator;

    impl Operator for MockAddOperator {
        fn name(&self) -> &str {
            "MockAdd"
        }

        fn infer_output_shapes(&self, ctx: &onyxia_core::InferenceCtx) -> Result<Vec<TensorShape>> {
            Ok(vec![ctx.input_shape(0)?.clone()])
        }

        fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
            // Fold if both inputs are F32
            if !ctx.all_inputs_have_values() {
                return Ok(vec![None]);
            }

            let a = ctx.input_value(0).and_then(|v| v.as_f32());
            let b = ctx.input_value(1).and_then(|v| v.as_f32());

            if let (Some(a), Some(b)) = (a, b) {
                if a.len() == b.len() {
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                    let input_shape = ctx.input_value(0).unwrap().shape.clone();
                    return Ok(vec![Some(TensorValue::new(
                        onyxia_core::TensorData::F32(result),
                        input_shape,
                        onyxia_core::DataType::F32,
                    ))]);
                }
            }

            Ok(vec![None])
        }

        fn plan(&self, _ctx: &mut onyxia_core::PlanCtx) -> Result<Vec<onyxia_core::Step>> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_initialize_constants() {
        let mut graph = IrGraph::new();

        // Create a constant tensor with initializer
        let mut tensor = TensorDef::new(
            "const".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Weight,
        );
        // Two F32 values: 1.0 and 2.0
        tensor.initializer = Some(vec![0, 0, 128, 63, 0, 0, 0, 64]);
        let tensor_id = graph.add_tensor(tensor);

        let registry = OperatorRegistry::new();
        let pass = ConstantFoldingPass::new();

        let changed = pass.run(&mut graph, &registry).unwrap();
        // Should not change (no operators to fold)
        assert!(!changed);

        // Constant tensor still has initializer (not converted to Value node)
        let tensor = graph.tensor(tensor_id).unwrap();
        assert!(tensor.has_initializer());
    }

    #[test]
    fn test_fold_constant_addition() {
        let mut graph = IrGraph::new();

        // Create two constant inputs as Value nodes
        let a_value = TensorValue::new(
            onyxia_core::TensorData::F32(vec![1.0, 2.0]),
            vec![2],
            onyxia_core::DataType::F32,
        );
        let a_value_node = IrNode::new_value(a_value);
        let a_value_node_id = graph.add_node(a_value_node);

        let b_value = TensorValue::new(
            onyxia_core::TensorData::F32(vec![3.0, 4.0]),
            vec![2],
            onyxia_core::DataType::F32,
        );
        let b_value_node = IrNode::new_value(b_value);
        let b_value_node_id = graph.add_node(b_value_node);

        // Create output tensor
        let output = TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        );
        let output_id = graph.add_tensor(output);

        // Add operator node with ValueNode inputs
        let mut node = IrNode::new_operator("MockAdd".to_string());
        node.add_input(IrInput::ValueNode(a_value_node_id)).unwrap();
        node.add_input(IrInput::ValueNode(b_value_node_id)).unwrap();
        node.add_output(output_id).unwrap();
        let add_node_id = graph.add_node(node);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        // Run constant folding pass
        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);

        // Verify Add node was replaced with Value node
        let node = graph.node(add_node_id).unwrap();
        assert!(
            node.is_value(),
            "Add node should be replaced with Value node"
        );

        // Verify value is correct
        let value = node.as_value().unwrap();
        assert_eq!(value.as_f32(), Some(&[4.0, 6.0][..]));
    }

    #[test]
    fn test_no_fold_without_values() {
        let mut graph = IrGraph::new();

        // Create inputs without values (regular tensors)
        let a_id = graph.add_tensor(TensorDef::new(
            "a".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let b_id = graph.add_tensor(TensorDef::new(
            "b".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let output_id = graph.add_tensor(TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));

        let mut node = IrNode::new_operator("MockAdd".to_string());
        node.add_tensor_input(a_id).unwrap();
        node.add_tensor_input(b_id).unwrap();
        node.add_output(output_id).unwrap();
        let add_node_id = graph.add_node(node);

        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        // Should not fold (inputs don't have values)
        assert!(!changed);

        // Node should still be an Operator, not replaced with Value node
        let node = graph.node(add_node_id).unwrap();
        assert!(node.is_operator());
    }
}
