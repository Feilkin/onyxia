//! Shape inference context and helpers.
//!
//! [`ShapeInferenceCtx`] is passed to [`crate::operator::Operator::infer_shapes()`]
//! and provides read-only access to a node's attributes and its constant input
//! values. This allows operators like Reshape to read the target shape from a
//! constant input tensor at compile time.

use crate::ir::{EdgeData, IrEdge, IrGraph, IrNode};
use crate::types::TensorValue;
use crate::{Error, Result};
use onyxia_onnx::AttributeValue;

/// Read-only context for compile-time shape inference.
///
/// Provides access to node attributes and constant edge values for operators
/// like Reshape that read shape from a constant input.
pub struct ShapeInferenceCtx<'a> {
    /// The IR node being analyzed.
    pub node: &'a IrNode,
    /// The full IR graph (for accessing input edge data).
    pub graph: &'a IrGraph,
}

impl<'a> ShapeInferenceCtx<'a> {
    /// Create a new shape inference context.
    pub fn new(node: &'a IrNode, graph: &'a IrGraph) -> Self {
        Self { node, graph }
    }

    /// Get the number of inputs to this node.
    pub fn input_count(&self) -> usize {
        self.node.inputs().len()
    }

    /// Get the input edge (tensor metadata) for the given index.
    ///
    /// Returns the [`IrEdge`] which has name, dtype, shape (from ONNX), and
    /// potentially constant data.
    pub fn input_edge(&self, index: usize) -> Result<&IrEdge> {
        let inputs = self.node.inputs();
        let input_id = inputs.get(index).ok_or_else(|| {
            Error::Compilation(format!(
                "Input index {index} out of range (node has {} inputs)",
                inputs.len()
            ))
        })?;
        self.graph.edge(*input_id)
    }

    /// Get the constant value for an input, if available.
    ///
    /// Returns `Ok(Some(value))` if the input is a constant or initializer
    /// whose value has been fully evaluated at compile time. Returns
    /// `Ok(None)` if the input is a runtime value.
    pub fn input_constant_value(&self, index: usize) -> Result<Option<&TensorValue>> {
        let edge = self.input_edge(index)?;
        match &edge.data {
            EdgeData::Constant(value) => Ok(Some(value)),
            _ => Ok(None),
        }
    }

    /// Get a node attribute by name.
    pub fn attr(&self, name: &str) -> Option<&AttributeValue> {
        self.node.attributes.get(name)
    }

    /// Get a required i64 attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute is missing or not of type `Int`.
    pub fn attr_i64(&self, name: &str) -> Result<i64> {
        match self.attr(name) {
            Some(AttributeValue::Int(v)) => Ok(*v),
            _ => Err(Error::Attribute(format!(
                "Missing required i64 attribute '{name}'"
            ))),
        }
    }

    /// Get a required ints attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute is missing or not of type `Ints`.
    pub fn attr_ints(&self, name: &str) -> Result<&[i64]> {
        match self.attr(name) {
            Some(AttributeValue::Ints(v)) => Ok(v.as_slice()),
            _ => Err(Error::Attribute(format!(
                "Missing required ints attribute '{name}'"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{EdgeData, IrEdge, IrGraph, IrNode};
    use crate::types::{DataType, TensorData, TensorShape, TensorValue};

    fn build_simple_graph() -> (IrGraph, crate::ir::IrNodeId) {
        let mut graph = IrGraph::new();

        let input_edge = IrEdge::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        );
        let input_id = graph.add_edge(input_edge);
        graph.inputs.push(input_id);

        let out_edge = IrEdge::new("y".to_string(), DataType::F32, TensorShape::Unknown);
        let out_id = graph.add_edge(out_edge);
        graph.outputs.push(out_id);

        let mut node = IrNode::new("TestOp".to_string());
        node.add_input(input_id);
        node.add_output(out_id).unwrap();

        let node_id = graph.add_node(node);

        (graph, node_id)
    }

    fn build_constant_graph() -> (IrGraph, crate::ir::IrNodeId) {
        let mut graph = IrGraph::new();

        let const_value =
            TensorValue::new(TensorData::I64(vec![3i64, 3i64]), vec![2], DataType::I64);
        let mut const_edge = IrEdge::new(
            "shape_input".to_string(),
            DataType::I64,
            TensorShape::Static(vec![2]),
        );
        const_edge.data = EdgeData::Constant(const_value);
        let const_id = graph.add_edge(const_edge);

        let out_edge = IrEdge::new("out".to_string(), DataType::F32, TensorShape::Unknown);
        let out_id = graph.add_edge(out_edge);

        let mut node = IrNode::new("Reshape".to_string());
        node.add_input(const_id);
        node.add_output(out_id).unwrap();
        node.set_attribute("axis".to_string(), onyxia_onnx::AttributeValue::Int(1))
            .unwrap();
        node.set_attribute(
            "axes".to_string(),
            onyxia_onnx::AttributeValue::Ints(vec![0, 1]),
        )
        .unwrap();

        let node_id = graph.add_node(node);

        (graph, node_id)
    }

    #[test]
    fn test_input_edge_index_out_of_range() {
        let (graph, node_id) = build_simple_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert!(ctx.input_edge(99).is_err());
    }

    #[test]
    fn test_input_edge_ok() {
        let (graph, node_id) = build_simple_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        let edge = ctx.input_edge(0).unwrap();
        assert_eq!(edge.name, "x");
    }

    #[test]
    fn test_input_constant_value_runtime_returns_none() {
        let (graph, node_id) = build_simple_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert!(ctx.input_constant_value(0).unwrap().is_none());
    }

    #[test]
    fn test_input_constant_value_returns_value() {
        let (graph, node_id) = build_constant_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        let val = ctx.input_constant_value(0).unwrap();
        assert!(val.is_some());
    }

    #[test]
    fn test_attr_i64_ok() {
        let (graph, node_id) = build_constant_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert_eq!(ctx.attr_i64("axis").unwrap(), 1);
    }

    #[test]
    fn test_attr_i64_missing() {
        let (graph, node_id) = build_constant_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert!(ctx.attr_i64("nonexistent").is_err());
    }

    #[test]
    fn test_attr_ints_ok() {
        let (graph, node_id) = build_constant_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert_eq!(ctx.attr_ints("axes").unwrap(), &[0i64, 1]);
    }

    #[test]
    fn test_attr_ints_missing() {
        let (graph, node_id) = build_constant_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert!(ctx.attr_ints("nonexistent").is_err());
    }

    #[test]
    fn test_attr_returns_none_for_missing() {
        let (graph, node_id) = build_simple_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert!(ctx.attr("missing").is_none());
    }

    #[test]
    fn test_input_count() {
        let (graph, node_id) = build_simple_graph();
        let node = graph.node(node_id).unwrap();
        let ctx = ShapeInferenceCtx::new(node, &graph);
        assert_eq!(ctx.input_count(), 1);
    }
}
