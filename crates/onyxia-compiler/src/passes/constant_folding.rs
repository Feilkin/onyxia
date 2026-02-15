//! Constant folding pass.
//!
//! Forward pass in topological order that evaluates operations at compile time when
//! all inputs are known constants.

use onyxia_core::{
    Error, FoldCtx, IrGraph, IrNode, OperatorRegistry, Pass, Result, Stage, TensorValue,
};

/// Pass that folds constant operations at compile time.
///
/// Walks the graph in topological order. For each node:
/// 1. Initializes `TensorDef.value` from initializers (weights, constants)
/// 2. If all inputs have values, calls `operator.try_fold()` to evaluate at compile time
/// 3. Stores folded results in output `TensorDef.value`
///
/// This enables chains like Shape→Gather→Concat→Reshape to be fully resolved at
/// compile time, eliminating unnecessary GPU operations.
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    /// Create a new constant folding pass.
    pub fn new() -> Self {
        Self
    }

    /// Initialize tensor values from initializers.
    fn initialize_constants(&self, graph: &mut IrGraph) -> Result<bool> {
        let mut changed = false;

        // Iterate over all tensors
        for i in 0..graph.tensor_count() {
            let tensor_id = onyxia_core::IrTensorId::new(i);
            let tensor = graph.tensor(tensor_id)?;

            // Skip if already has a value or no initializer
            if tensor.has_value() || !tensor.has_initializer() {
                continue;
            }

            // Parse initializer bytes into TensorValue
            let initializer = tensor.initializer.as_ref().unwrap();
            let shape = match &tensor.shape {
                onyxia_core::TensorShape::Static(dims) => dims.clone(),
                _ => {
                    // Can't fold tensors with non-static shapes
                    continue;
                }
            };

            let value = TensorValue::from_bytes(initializer, tensor.dtype, &shape)?;

            // Store the value
            graph.tensor_mut(tensor_id)?.value = Some(value);
            changed = true;
        }

        Ok(changed)
    }

    /// Attempt to fold a single node.
    fn fold_node(
        &self,
        node: &IrNode,
        graph: &mut IrGraph,
        registry: &OperatorRegistry,
    ) -> Result<bool> {
        // Look up operator
        let operator = registry.get(&node.op_type).ok_or_else(|| {
            Error::ConstantFolding(format!("No operator registered for type: {}", node.op_type))
        })?;

        // Build fold context
        let ctx = FoldCtx::new(node, graph);

        // Call operator's constant folding
        let folded_outputs = operator.try_fold(&ctx).map_err(|e| {
            Error::ConstantFolding(format!(
                "Failed to fold node '{}' (op_type: {}): {}",
                node.op_type, node.op_type, e
            ))
        })?;

        // If operator returned no folded values, skip
        if folded_outputs.is_empty() {
            return Ok(false);
        }

        // Validate output count
        if folded_outputs.len() != node.outputs.len() {
            return Err(Error::ConstantFolding(format!(
                "Operator {} returned {} folded outputs but node has {} outputs",
                node.op_type,
                folded_outputs.len(),
                node.outputs.len()
            )));
        }

        // Update output tensor values
        let mut changed = false;
        for (i, folded_value) in folded_outputs.into_iter().enumerate() {
            if let Some(value) = folded_value {
                let tensor_id = node.outputs[i];
                graph.tensor_mut(tensor_id)?.value = Some(value);
                changed = true;
            }
        }

        Ok(changed)
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

        // First, initialize constants from initializers
        let init_changed = self.initialize_constants(graph)?;
        changed = changed || init_changed;

        // Process nodes in topological order
        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node = graph.node(node_id)?.clone();
            let node_changed = self.fold_node(&node, graph, registry)?;
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
    use onyxia_core::{DataType, IrGraph, IrNode, Operator, TensorDef, TensorKind, TensorShape};

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
                    return Ok(vec![Some(TensorValue::F32(result))]);
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
        assert!(changed);

        let value = &graph.tensor(tensor_id).unwrap().value;
        assert!(value.is_some());
        assert_eq!(value.as_ref().unwrap().as_f32(), Some(&[1.0, 2.0][..]));
    }

    #[test]
    fn test_fold_constant_addition() {
        let mut graph = IrGraph::new();

        // Create two constant inputs
        let mut a_tensor = TensorDef::new(
            "a".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Weight,
        );
        a_tensor.value = Some(TensorValue::F32(vec![1.0, 2.0]));
        let a_id = graph.add_tensor(a_tensor);

        let mut b_tensor = TensorDef::new(
            "b".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Weight,
        );
        b_tensor.value = Some(TensorValue::F32(vec![3.0, 4.0]));
        let b_id = graph.add_tensor(b_tensor);

        // Create output tensor
        let output = TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        );
        let output_id = graph.add_tensor(output);

        // Add node
        let mut node = IrNode::new("MockAdd".to_string());
        node.add_input(a_id);
        node.add_input(b_id);
        node.add_output(output_id);
        graph.add_node(node);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        // Run constant folding pass
        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);

        let output_value = &graph.tensor(output_id).unwrap().value;
        assert!(output_value.is_some());
        assert_eq!(
            output_value.as_ref().unwrap().as_f32(),
            Some(&[4.0, 6.0][..])
        );
    }

    #[test]
    fn test_no_fold_without_values() {
        let mut graph = IrGraph::new();

        // Create inputs without values
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

        let mut node = IrNode::new("MockAdd".to_string());
        node.add_input(a_id);
        node.add_input(b_id);
        node.add_output(output_id);
        graph.add_node(node);

        let mut registry = OperatorRegistry::new();
        registry.register("MockAdd", MockAddOperator);

        let pass = ConstantFoldingPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        // Should not fold (inputs don't have values)
        assert!(!changed);
        assert!(graph.tensor(output_id).unwrap().value.is_none());
    }
}
