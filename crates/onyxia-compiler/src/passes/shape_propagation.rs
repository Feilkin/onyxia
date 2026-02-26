//! Shape propagation compiler pass.
//!
//! Walks the graph in topological order and calls [`Operator::infer_shapes()`]
//! for each node, updating [`IrEdge::shape`] on output edges in-place.
//!
//! This pass runs at [`Stage::Inference`], after [`InitializeConstantsPass`]
//! has converted weight initializers to constant values.
//!
//! # Shape sources
//!
//! - **Graph input edges** already carry `SymbolicShape` from `IrGraph::from_onnx()`
//!   (named dims such as `"batch_size"` are preserved). This pass does not
//!   touch them.
//! - **Constant / weight edges** already have fully-static `Ranked([Fixed(...)])`
//!   shapes from `InitializeConstantsPass`. This pass does not touch them either.
//! - **Operator output edges** start as `Unranked`. This pass writes the result
//!   of `Operator::infer_shapes()` to them. If an operator is unregistered or
//!   its `infer_shapes()` returns `Unranked`, the edge shape stays `Unranked` —
//!   the pass never fails due to missing shape information.
//!
//! [`InitializeConstantsPass`]: crate::passes::InitializeConstantsPass

use onyxia_core::{
    IrGraph, IrNodeId, OperatorRegistry, Pass, Result, ShapeInferenceCtx, Stage, SymbolicShape,
};

/// Pass that propagates symbolic shapes through operator nodes.
///
/// For each node in topological order, the pass:
/// 1. Collects the `shape` of each input edge.
/// 2. Looks up the operator in the registry.
/// 3. Calls [`Operator::infer_shapes()`] with those shapes.
/// 4. Writes the returned shapes back to the node's output edges.
///
/// Missing or unrecognised operators are silently skipped — their output
/// edges remain `Unranked`.
pub struct ShapePropagationPass;

impl ShapePropagationPass {
    /// Create a new shape propagation pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ShapePropagationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for ShapePropagationPass {
    fn name(&self) -> &str {
        "shape_propagation"
    }

    fn stage(&self) -> Stage {
        Stage::Inference
    }

    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool> {
        let node_ids: Vec<IrNodeId> = graph.topological_order();
        let mut changed = false;

        for node_id in node_ids {
            // Collect everything we need before borrowing graph mutably.
            let (op_type, input_ids, output_ids) = {
                let node = graph.node(node_id)?;
                (
                    node.op_type.clone(),
                    node.inputs().to_vec(),
                    node.outputs().to_vec(),
                )
            };

            // Look up operator; skip nodes without a registered operator.
            let Some(op) = registry.get(&op_type) else {
                continue;
            };

            // Collect input shapes (cloned so we can release the borrow).
            let input_shapes: Vec<SymbolicShape> = input_ids
                .iter()
                .filter_map(|&id| graph.edge(id).ok().map(|e| e.shape.clone()))
                .collect();
            let input_shape_refs: Vec<&SymbolicShape> = input_shapes.iter().collect();

            // Build shape inference context and call the operator.
            let inferred = {
                let node = graph.node(node_id)?;
                let ctx = ShapeInferenceCtx::new(node, graph);
                op.infer_shapes(&input_shape_refs, &ctx)?
            };

            // Write inferred shapes back to output edges.
            for (out_id, inferred_shape) in output_ids.iter().zip(inferred.iter()) {
                // Only update if we have a more useful shape than Unranked.
                if matches!(inferred_shape, SymbolicShape::Unranked) {
                    continue;
                }
                let edge = graph.edge_mut(*out_id)?;
                edge.shape = inferred_shape.clone();
                changed = true;
            }
        }

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::ir::{IrEdge, IrNode};
    use onyxia_core::{DataType, Dim, IrGraph, OperatorRegistry, SymbolicShape};

    fn add_input_edge(
        graph: &mut IrGraph,
        name: &str,
        shape: SymbolicShape,
    ) -> onyxia_core::IrEdgeId {
        let edge = IrEdge::new(name.to_string(), DataType::F32, shape);
        let id = graph.add_edge(edge);
        graph.inputs.push(id);
        id
    }

    fn add_operator_output_edge(graph: &mut IrGraph, name: &str) -> onyxia_core::IrEdgeId {
        let edge = IrEdge::new(name.to_string(), DataType::F32, SymbolicShape::Unranked);
        graph.add_edge(edge)
    }

    fn add_op_node(
        graph: &mut IrGraph,
        op_type: &str,
        inputs: Vec<onyxia_core::IrEdgeId>,
        outputs: Vec<onyxia_core::IrEdgeId>,
    ) -> onyxia_core::IrNodeId {
        let mut node = IrNode::new(op_type.to_string());
        for id in inputs {
            node.add_input(id);
        }
        for id in outputs {
            node.add_output(id).unwrap();
        }
        graph.add_node(node)
    }

    // ── Tests ──────────────────────────────────────────────────────────────

    /// With an empty registry the pass should run without errors.
    #[test]
    fn test_empty_graph_no_error() {
        let mut graph = IrGraph::new();
        let registry = OperatorRegistry::new();
        let pass = ShapePropagationPass::new();

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(!changed);
    }

    /// Operators not in the registry are silently skipped.
    #[test]
    fn test_unknown_operator_skipped() {
        let mut graph = IrGraph::new();
        let a = add_input_edge(&mut graph, "a", SymbolicShape::fixed(&[2, 3]));
        let out = add_operator_output_edge(&mut graph, "out");
        add_op_node(&mut graph, "UnknownOp", vec![a], vec![out]);

        let registry = OperatorRegistry::new(); // nothing registered
        let pass = ShapePropagationPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(!changed, "unknown op should not change any shapes");
        assert_eq!(
            graph.edge(out).unwrap().shape,
            SymbolicShape::Unranked,
            "output should remain Unranked"
        );
    }

    /// Graph input edges must not be overwritten by the pass.
    #[test]
    fn test_graph_input_edges_untouched() {
        let mut graph = IrGraph::new();
        let named_shape =
            SymbolicShape::Ranked(vec![Dim::Named("batch_size".to_string()), Dim::Fixed(512)]);
        let a = add_input_edge(&mut graph, "a", named_shape.clone());

        let registry = OperatorRegistry::new();
        let pass = ShapePropagationPass::new();
        pass.run(&mut graph, &registry).unwrap();

        assert_eq!(
            graph.edge(a).unwrap().shape,
            named_shape,
            "graph input shape must not be modified by the pass"
        );
    }

    /// An operator whose default `infer_shapes` returns `Unranked` should
    /// leave the output edge shape unchanged.
    #[test]
    fn test_unranked_default_leaves_output_unranked() {
        use onyxia_core::{Operator, compile_ctx::CompileCtx, dispatch::OpDispatch};

        struct NopOp;
        impl Operator for NopOp {
            fn name(&self) -> &str {
                "Nop"
            }
            fn create_dispatch(
                &self,
                _ctx: &mut CompileCtx,
            ) -> onyxia_core::Result<Box<dyn OpDispatch>> {
                Err(onyxia_core::Error::Compilation(
                    "not implemented".to_string(),
                ))
            }
            // infer_shapes falls back to the default: returns [Unranked]
        }

        let mut graph = IrGraph::new();
        let a = add_input_edge(&mut graph, "a", SymbolicShape::fixed(&[4]));
        let out = add_operator_output_edge(&mut graph, "out");
        add_op_node(&mut graph, "Nop", vec![a], vec![out]);

        let mut registry = OperatorRegistry::new();
        registry.register("Nop", NopOp);

        let pass = ShapePropagationPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(
            !changed,
            "default infer_shapes should not count as a change"
        );
        assert_eq!(graph.edge(out).unwrap().shape, SymbolicShape::Unranked);
    }

    /// An operator that implements `infer_shapes` should update the output shape.
    #[test]
    fn test_infer_shapes_updates_output_edge() {
        use onyxia_core::{
            Operator, compile_ctx::CompileCtx, dispatch::OpDispatch,
            shape_inference::ShapeInferenceCtx,
        };

        /// Passthrough-shape op: copies the shape of input 0 to output 0.
        struct PassthroughShapeOp;
        impl Operator for PassthroughShapeOp {
            fn name(&self) -> &str {
                "Passthrough"
            }
            fn create_dispatch(
                &self,
                _ctx: &mut CompileCtx,
            ) -> onyxia_core::Result<Box<dyn OpDispatch>> {
                Err(onyxia_core::Error::Compilation(
                    "not implemented".to_string(),
                ))
            }
            fn infer_shapes(
                &self,
                input_shapes: &[&SymbolicShape],
                _ctx: &ShapeInferenceCtx,
            ) -> onyxia_core::Result<Vec<SymbolicShape>> {
                Ok(vec![input_shapes[0].clone()])
            }
        }

        let mut graph = IrGraph::new();
        let a = add_input_edge(&mut graph, "a", SymbolicShape::fixed(&[3, 4]));
        let out = add_operator_output_edge(&mut graph, "out");
        add_op_node(&mut graph, "Passthrough", vec![a], vec![out]);

        let mut registry = OperatorRegistry::new();
        registry.register("Passthrough", PassthroughShapeOp);

        let pass = ShapePropagationPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed, "pass should report a change");
        assert_eq!(
            graph.edge(out).unwrap().shape,
            SymbolicShape::fixed(&[3, 4]),
            "output shape should match inferred shape"
        );
    }
}
