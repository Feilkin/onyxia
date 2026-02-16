//! Constant folding pass.
//!
//! Forward pass in topological order that evaluates operations at compile time when
//! all inputs are known constants.
//!
//! Folded operators are removed from the graph; the constant value is stored on
//! the output edge (`IrEdge::constant_value`).

use onyxia_core::{Error, FoldCtx, IrGraph, IrNodeId, OperatorRegistry, Pass, Result, Stage};

/// Pass that folds constant operations at compile time.
///
/// Walks the graph in topological order. For each node:
/// 1. Checks whether all inputs are constants (via edge `constant_value` or `initializer`)
/// 2. If so, calls `operator.try_fold()` to evaluate at compile time
/// 3. Stores the folded result on the output edge and removes the node
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
        let op_type = node.op_type();
        let outputs = node.outputs();

        // Look up operator
        let operator = registry.get(op_type).ok_or_else(|| {
            Error::ConstantFolding(format!("No operator registered for type: {}", op_type))
        })?;

        // Build fold context
        let ctx = FoldCtx::new(&node, graph);

        if !ctx.all_inputs_have_values() {
            return Ok(false);
        }

        // Call operator's constant folding
        let folded_outputs = operator.try_fold(&ctx).map_err(|e| {
            Error::ConstantFolding(format!("Failed to fold node (op_type: {}): {}", op_type, e))
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

        // Fold single-output operators (most common case)
        if outputs.len() == 1
            && let Some(value) = folded_outputs.into_iter().next().unwrap()
        {
            graph.fold_node_to_constant(node_id, value)?;
            return Ok(true);
        }
        // Multi-output folding not yet implemented

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
    use onyxia_core::ir::IrEdge;
    use onyxia_core::{DataType, IrGraph, IrNode, Operator, TensorShape, TensorValue};

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
        let edge = IrEdge::with_initializer(
            "const".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            vec![0, 0, 128, 63, 0, 0, 0, 64],
        );
        let edge_id = graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = ConstantFoldingPass::new();

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(!changed);

        let edge = graph.edge(edge_id).unwrap();
        assert!(edge.has_initializer());
    }

    #[test]
    fn test_fold_constant_addition() {
        let mut graph = IrGraph::new();

        // Create two constant input edges (with constant_value set)
        let edge_a = IrEdge::with_constant(
            "a".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorValue::new(
                onyxia_core::TensorData::F32(vec![1.0, 2.0]),
                vec![2],
                DataType::F32,
            ),
        );
        let edge_a_id = graph.add_edge(edge_a);

        let edge_b = IrEdge::with_constant(
            "b".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorValue::new(
                onyxia_core::TensorData::F32(vec![3.0, 4.0]),
                vec![2],
                DataType::F32,
            ),
        );
        let edge_b_id = graph.add_edge(edge_b);

        // Create output edge
        let output_edge = IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        );
        let output_id = graph.add_edge(output_edge);

        // Add operator node
        let mut node = IrNode::new("MockAdd".to_string());
        node.add_input(edge_a_id);
        node.add_input(edge_b_id);
        node.add_output(output_id).unwrap();
        let add_node_id = graph.add_node(node);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        // Run constant folding pass
        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);

        // Node should be removed (folded)
        assert!(graph.is_fully_folded(add_node_id).unwrap());

        // Output edge should have the constant value
        let output_edge = graph.edge(output_id).unwrap();
        assert!(output_edge.is_constant());
        let value = output_edge.constant_value().unwrap();
        assert_eq!(value.as_f32(), Some(&[4.0, 6.0][..]));
    }

    #[test]
    fn test_no_fold_without_values() {
        let mut graph = IrGraph::new();

        // Create inputs without values (regular dynamic tensors)
        let a_id = graph.add_edge(IrEdge::new(
            "a".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let b_id = graph.add_edge(IrEdge::new(
            "b".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let output_id = graph.add_edge(IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));

        let mut node = IrNode::new("MockAdd".to_string());
        node.add_input(a_id);
        node.add_input(b_id);
        node.add_output(output_id).unwrap();
        let add_node_id = graph.add_node(node);

        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(!changed);

        // Node should still exist
        assert!(!graph.is_fully_folded(add_node_id).unwrap());
    }
}
