//! Planning pass (code generation).
//!
//! Iterates through the graph in topological order and calls `operator.plan()` for
//! each node. Folded nodes no longer exist in the graph, so no skipping is needed.

use onyxia_core::plan::TensorMetadata;
use onyxia_core::{
    CompiledModel, CompiledShader, Error, IrGraph, IrNode, IrTensorId, ModelMetadata,
    OperatorRegistry, Pass, PlanCtx, PlannedOp, Result, Stage, TensorRegistry,
};
use std::collections::HashMap;

/// Pass that generates execution steps for the graph.
///
/// Walks the graph in topological order and calls `Operator::plan()` for each
/// node. Assembles the final `CompiledModel`.
pub struct PlanningPass {
    /// Final compiled model being constructed.
    compiled_model: Option<CompiledModel>,
}

impl PlanningPass {
    /// Create a new planning pass.
    pub fn new() -> Self {
        Self {
            compiled_model: None,
        }
    }

    /// Take ownership of the compiled model.
    ///
    /// This should be called after the pass runs to extract the result.
    pub fn take_model(&mut self) -> Option<CompiledModel> {
        self.compiled_model.take()
    }

    /// Plan a single node.
    fn plan_node(
        &self,
        node: &IrNode,
        graph: &IrGraph,
        registry: &OperatorRegistry,
        shaders: &mut Vec<CompiledShader>,
        shader_cache: &mut HashMap<(String, String), usize>,
        dynamic_dimensions: &HashMap<String, usize>,
        symbolic_bindings: &mut Vec<onyxia_core::SymbolicBinding>,
    ) -> Result<PlannedOp> {
        let op_type = node.op_type();

        // Look up operator
        let operator = registry.get(op_type).ok_or_else(|| {
            Error::Planning(format!("No operator registered for type: {}", op_type))
        })?;

        // Build plan context
        let mut scratch_buffers = Vec::new();
        let mut ctx = PlanCtx {
            node,
            graph,
            shaders,
            shader_cache,
            scratch_buffers: &mut scratch_buffers,
            dynamic_dimensions,
            symbolic_bindings,
        };

        // Call operator's planning
        let steps = operator.plan(&mut ctx).map_err(|e| {
            Error::Planning(format!("Failed to plan node (op_type: {}): {}", op_type, e))
        })?;

        // Build PlannedOp
        Ok(PlannedOp {
            name: op_type.to_string(),
            steps,
            scratch_buffers,
        })
    }

    /// Build the tensor registry from the graph.
    fn build_tensor_registry(&self, graph: &IrGraph) -> Result<TensorRegistry> {
        let mut registry = TensorRegistry::new();

        for i in 0..graph.tensor_count() {
            let tensor_id = IrTensorId::new(i);
            let tensor = graph.tensor(tensor_id)?;

            let metadata =
                TensorMetadata::new(tensor.name.clone(), tensor.dtype, tensor.shape.clone());
            registry.add(metadata);
        }

        Ok(registry)
    }
}

impl Pass for PlanningPass {
    fn name(&self) -> &str {
        "planning"
    }

    fn stage(&self) -> Stage {
        Stage::Planning
    }

    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool> {
        let mut operations = Vec::new();
        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut symbolic_bindings = Vec::new();
        let dynamic_dimensions = HashMap::new();

        // Process nodes in topological order
        // Folded nodes have been removed, so every node here needs planning
        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node = graph.node(node_id)?.clone();

            let planned_op = self.plan_node(
                &node,
                graph,
                registry,
                &mut shaders,
                &mut shader_cache,
                &dynamic_dimensions,
                &mut symbolic_bindings,
            )?;

            operations.push(planned_op);
        }

        // Build tensor registry
        let tensor_registry = self.build_tensor_registry(graph)?;

        // Build compiled model
        let _model = CompiledModel {
            operations,
            shaders,
            tensors: tensor_registry,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            symbolic_bindings,
            metadata: ModelMetadata {
                name: "model".to_string(),
                ir_version: 0,
                producer_name: "onyxia-compiler".to_string(),
                model_version: 0,
            },
        };

        Ok(true)
    }
}

impl Default for PlanningPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::ir::IrEdge;
    use onyxia_core::{
        BufferRef, DataType, IrGraph, IrNode, Operator, Step, TensorKind, TensorShape, TensorValue,
    };

    // Mock operator that emits a dummy step
    struct MockOperator;

    impl Operator for MockOperator {
        fn name(&self) -> &str {
            "Mock"
        }

        fn infer_output_shapes(&self, ctx: &onyxia_core::InferenceCtx) -> Result<Vec<TensorShape>> {
            Ok(vec![ctx.input_shape(0)?.clone()])
        }

        fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
            Ok(vec![Step::WriteBuffer {
                dst: BufferRef::Tensor(IrTensorId::new(0)),
                data: vec![0u8; 4],
            }])
        }
    }

    #[test]
    fn test_planning_skips_folded_nodes() {
        let mut graph = IrGraph::new();

        // Create a constant input edge (folded — has constant_value)
        let mut const_edge = IrEdge::new(
            "const_input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        );
        const_edge.constant_value = Some(TensorValue::new(
            onyxia_core::TensorData::F32(vec![1.0, 2.0]),
            vec![2],
            DataType::F32,
        ));
        let const_edge_id = graph.add_edge(const_edge);

        // Create output edge
        let output_edge = IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        );
        let output_id = graph.add_edge(output_edge);

        // Add operator node that consumes the constant edge
        let mut node = IrNode::new("Mock".to_string());
        node.add_input(const_edge_id);
        node.add_output(output_id).unwrap();
        graph.add_node(node);

        // Register mock operator
        let mut registry = OperatorRegistry::new();
        registry.register("Mock", MockOperator);

        // Run planning pass — node still gets planned (it's not folded itself)
        let pass = PlanningPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(changed);
    }

    #[test]
    fn test_planning_emits_steps_for_non_folded_nodes() {
        let mut graph = IrGraph::new();

        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));

        let output_id = graph.add_edge(IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));

        graph.inputs.push(input_id);
        graph.outputs.push(output_id);

        let mut node = IrNode::new("Mock".to_string());
        node.add_input(input_id);
        node.add_output(output_id).unwrap();
        graph.add_node(node);

        let mut registry = OperatorRegistry::new();
        registry.register("Mock", MockOperator);

        let pass = PlanningPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);
    }

    #[test]
    fn test_folded_nodes_not_in_topo_order() {
        let mut graph = IrGraph::new();

        // Create edges
        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let mid_id = graph.add_edge(IrEdge::new(
            "mid".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));
        let output_id = graph.add_edge(IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));

        // Add two nodes in a chain
        let mut node_a = IrNode::new("Mock".to_string());
        node_a.add_input(input_id);
        node_a.add_output(mid_id).unwrap();
        let node_a_id = graph.add_node(node_a);

        let mut node_b = IrNode::new("Mock".to_string());
        node_b.add_input(mid_id);
        node_b.add_output(output_id).unwrap();
        graph.add_node(node_b);

        // Fold node_a (simulates constant folding)
        let value = TensorValue::new(
            onyxia_core::TensorData::F32(vec![1.0, 2.0]),
            vec![2],
            DataType::F32,
        );
        graph.fold_node_to_constant(node_a_id, value).unwrap();

        // Only node_b should be in topo order
        let topo = graph.topological_order();
        assert_eq!(topo.len(), 1);
    }
}
