//! Intermediate representation for the compiler graph.
//!
//! The IR is a directed graph where:
//! - **Nodes** (`IrNode`) are operators (e.g., Add, MatMul, Reshape)
//! - **Edges** (`IrEdge`) are tensor value flows between operators
//!
//! During constant folding, edges can acquire a `constant_value` and their
//! producing operator is removed from the graph. Downstream consumers
//! see the constant directly on the edge.

use crate::types::{DataType, TensorShape, TensorValue};
use crate::{Error, Result};
use onyxia_onnx::AttributeValue;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::Topo;

use std::collections::HashMap;

/// Type alias for IR node identifiers (backed by petgraph NodeIndex).
pub type IrNodeId = NodeIndex;

/// Unique identifier for an edge (tensor flow) in the IR graph.
///
/// This is an index into `IrGraph::edges`. Unlike node IDs (which use petgraph's
/// stable NodeIndex), edge IDs are simple usize indices that remain valid across
/// graph mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrEdgeId(pub usize);

impl IrEdgeId {
    /// Create a new edge ID.
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Backward-compatibility alias: `IrTensorId` equals `IrEdgeId`.
pub type IrTensorId = IrEdgeId;

// ──────────────────────────────── IrGraph ────────────────────────────────

/// Intermediate representation graph.
///
/// Nodes are operators; edges are tensor value flows stored in a side-table.
/// petgraph edges exist solely for topological ordering.
pub struct IrGraph {
    /// The graph structure (nodes only, no edge data).
    graph: StableGraph<IrNode, ()>,

    /// Edge metadata side-table.
    edges: Vec<IrEdge>,

    /// Lookup table: edge name -> edge ID.
    edge_by_name: HashMap<String, IrEdgeId>,

    /// Lookup table: edge ID -> producing node ID.
    edge_producer: HashMap<IrEdgeId, IrNodeId>,

    /// Lookup table: edge ID -> consuming node IDs.
    edge_consumers: HashMap<IrEdgeId, Vec<IrNodeId>>,

    /// Graph input edge IDs.
    pub inputs: Vec<IrEdgeId>,

    /// Graph output edge IDs.
    pub outputs: Vec<IrEdgeId>,
}

impl IrGraph {
    /// Create a new empty IR graph.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            edges: Vec::new(),
            edge_by_name: HashMap::new(),
            edge_producer: HashMap::new(),
            edge_consumers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    // ── Node access ──

    /// Get an immutable reference to a node.
    pub fn node(&self, id: IrNodeId) -> Result<&IrNode> {
        self.graph
            .node_weight(id)
            .ok_or_else(|| Error::InvalidGraph(format!("Node {:?} not found", id)))
    }

    /// Get a mutable reference to a node.
    pub fn node_mut(&mut self, id: IrNodeId) -> Result<&mut IrNode> {
        self.graph
            .node_weight_mut(id)
            .ok_or_else(|| Error::InvalidGraph(format!("Node {:?} not found", id)))
    }

    /// Get the inputs of a node.
    pub fn node_inputs(&self, id: IrNodeId) -> Result<&[IrEdgeId]> {
        Ok(self.node(id)?.inputs())
    }

    /// Get the outputs of a node.
    pub fn node_outputs(&self, id: IrNodeId) -> Result<&[IrEdgeId]> {
        Ok(self.node(id)?.outputs())
    }

    /// Iterate over all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = (IrNodeId, &IrNode)> {
        self.graph
            .node_indices()
            .filter_map(|id| self.graph.node_weight(id).map(|node| (id, node)))
    }

    // ── Edge (tensor) access ──

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get an immutable reference to an edge.
    pub fn edge(&self, id: IrEdgeId) -> Result<&IrEdge> {
        self.edges
            .get(id.index())
            .ok_or_else(|| Error::InvalidGraph(format!("Edge {:?} not found", id)))
    }

    /// Get a mutable reference to an edge.
    pub fn edge_mut(&mut self, id: IrEdgeId) -> Result<&mut IrEdge> {
        self.edges
            .get_mut(id.index())
            .ok_or_else(|| Error::InvalidGraph(format!("Edge {:?} not found", id)))
    }

    /// Backward-compat alias for `edge()`.
    pub fn tensor(&self, id: IrEdgeId) -> Result<&IrEdge> {
        self.edge(id)
    }

    /// Backward-compat alias for `edge_mut()`.
    pub fn tensor_mut(&mut self, id: IrEdgeId) -> Result<&mut IrEdge> {
        self.edge_mut(id)
    }

    /// Look up an edge by name.
    pub fn tensor_by_name(&self, name: &str) -> Option<IrEdgeId> {
        self.edge_by_name.get(name).copied()
    }

    /// Get the node that produces an edge, if any.
    pub fn tensor_producer(&self, id: IrEdgeId) -> Option<IrNodeId> {
        self.edge_producer.get(&id).copied()
    }

    /// Get the nodes that consume an edge.
    pub fn tensor_consumers(&self, id: IrEdgeId) -> Vec<IrNodeId> {
        self.edge_consumers.get(&id).cloned().unwrap_or_default()
    }

    // ── Graph mutation ──

    /// Add a new node to the graph and return its ID.
    ///
    /// This also updates the producer/consumer lookup tables and
    /// adds petgraph edges for topological ordering.
    pub fn add_node(&mut self, mut node: IrNode) -> IrNodeId {
        let placeholder = IrNode::new(String::new());
        let node_id = self.graph.add_node(placeholder);
        node.node_index = node_id;

        // Register producer/consumer relationships
        for &output_id in &node.outputs {
            self.edge_producer.insert(output_id, node_id);
        }

        for &input_id in &node.inputs {
            self.edge_consumers
                .entry(input_id)
                .or_default()
                .push(node_id);

            // Add petgraph edge for topological ordering
            if let Some(&producer_id) = self.edge_producer.get(&input_id) {
                self.graph.add_edge(producer_id, node_id, ());
            }
        }

        // Replace the placeholder with the real node
        *self.graph.node_weight_mut(node_id).unwrap() = node;

        node_id
    }

    /// Remove a node from the graph.
    ///
    /// This also removes the node from producer/consumer lookup tables. With
    /// `StableGraph`, other node indices remain valid.
    pub fn remove_node(&mut self, id: IrNodeId) -> Result<()> {
        let node = self.node(id)?.clone();

        // Remove from producer lookup
        for &output_id in &node.outputs {
            self.edge_producer.remove(&output_id);
        }

        // Remove from consumer lookup
        for &input_id in &node.inputs {
            if let Some(consumers) = self.edge_consumers.get_mut(&input_id) {
                consumers.retain(|&c| c != id);
            }
        }

        // Remove node from graph (automatically removes petgraph edges)
        self.graph.remove_node(id);

        Ok(())
    }

    /// Replace a node's operation and attributes.
    ///
    /// Preserves the node ID and input/output edges.
    pub fn replace_node(
        &mut self,
        id: IrNodeId,
        new_op_type: String,
        new_attributes: HashMap<String, AttributeValue>,
    ) -> Result<()> {
        let node = self.node_mut(id)?;
        node.op_type = new_op_type;
        node.attributes = new_attributes;
        Ok(())
    }

    /// Add an edge (tensor) to the graph and return its ID.
    pub fn add_edge(&mut self, edge: IrEdge) -> IrEdgeId {
        let id = IrEdgeId::new(self.edges.len());
        self.edge_by_name.insert(edge.name.clone(), id);
        self.edges.push(edge);
        id
    }

    /// Backward-compat alias for `add_edge()`.
    pub fn add_tensor(&mut self, tensor: IrEdge) -> IrEdgeId {
        self.add_edge(tensor)
    }

    // ── Constant folding ──

    /// Fold a single-output operator into a constant on its output edge.
    ///
    /// Removes the operator node from the graph and stores the constant
    /// value on the output edge. Consumers keep referencing the same
    /// `IrEdgeId`; they see the constant via `edge.constant_value`.
    pub fn fold_node_to_constant(&mut self, node_id: IrNodeId, value: TensorValue) -> Result<()> {
        let node = self.node(node_id)?.clone();

        if node.outputs.len() != 1 {
            return Err(Error::InvalidGraph(format!(
                "fold_node_to_constant requires exactly 1 output, got {}",
                node.outputs.len()
            )));
        }

        let output_edge_id = node.outputs[0];
        let edge = self.edge_mut(output_edge_id)?;

        // Update dtype, shape, and data to match the folded value
        edge.dtype = value.dtype;
        edge.shape = TensorShape::Static(value.shape.clone());
        edge.data = EdgeData::Constant(value);

        // Remove the node (cleans up producer/consumer tables + petgraph)
        self.remove_node(node_id)
    }

    /// Backward-compat alias for `fold_node_to_constant`.
    pub fn replace_single_output_with_value(
        &mut self,
        node_id: IrNodeId,
        value: TensorValue,
    ) -> Result<IrNodeId> {
        self.fold_node_to_constant(node_id, value)?;
        Ok(node_id)
    }

    /// Check if an operator node has been folded away.
    ///
    /// Returns `true` if the node no longer exists in the graph
    /// (it was removed during constant folding).
    pub fn is_fully_folded(&self, node_id: IrNodeId) -> Result<bool> {
        Ok(self.graph.node_weight(node_id).is_none())
    }

    // ── Graph queries ──

    /// Get the topological order of nodes in the graph.
    ///
    /// Returns nodes in an order such that all inputs to a node are produced
    /// before the node itself.
    pub fn topological_order(&self) -> Vec<IrNodeId> {
        let mut topo = Topo::new(&self.graph);
        let mut order = Vec::new();

        while let Some(id) = topo.next(&self.graph) {
            if self.graph.node_weight(id).is_some() {
                order.push(id);
            }
        }

        order
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges (tensors) in the graph.
    pub fn tensor_count(&self) -> usize {
        self.edges.len()
    }

    /// Find a node by its name.
    ///
    /// Searches through all nodes in the graph and returns the first node
    /// with a matching name.
    ///
    /// # Errors
    ///
    /// Returns an error if no node with the given name exists.
    pub fn find_node_by_name(&self, name: &str) -> Result<IrNodeId> {
        for (node_id, node) in self.nodes() {
            if node.name == name {
                return Ok(node_id);
            }
        }

        Err(Error::InvalidGraph(format!("Node '{}' not found", name)))
    }

    /// Get a node's name.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not exist.
    pub fn node_name(&self, node_id: IrNodeId) -> Result<&str> {
        Ok(&self.node(node_id)?.name)
    }

    /// Find a tensor by its name.
    ///
    /// Searches through all edges in the graph and returns the first edge
    /// with a matching name.
    ///
    /// # Errors
    ///
    /// Returns an error if no tensor with the given name exists.
    pub fn find_tensor_by_name(&self, name: &str) -> Result<IrEdgeId> {
        for (i, edge) in self.edges.iter().enumerate() {
            if edge.name == name {
                return Ok(IrEdgeId::new(i));
            }
        }

        Err(Error::InvalidGraph(format!("Tensor '{}' not found", name)))
    }
}

impl Default for IrGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────── IrNode ─────────────────────────────────

/// A node in the IR graph — an operator that transforms tensor edges.
///
/// After constant folding, folded operators are *removed* from the graph
/// entirely; the constant value lives on the output `IrEdge`.
#[derive(Debug, Clone)]
pub struct IrNode {
    /// Node name (from ONNX, may be empty).
    pub name: String,

    /// ONNX operator type (e.g., "Add", "MatMul").
    pub op_type: String,

    /// Operator attributes (e.g., axis, epsilon, transpose flags).
    pub attributes: HashMap<String, AttributeValue>,

    /// Input edge IDs.
    pub inputs: Vec<IrEdgeId>,

    /// Output edge IDs.
    pub outputs: Vec<IrEdgeId>,

    /// The graph node index (for efficient graph traversal).
    pub node_index: IrNodeId,
}

impl IrNode {
    /// Create a new operator node.
    pub fn new(op_type: String) -> Self {
        Self {
            name: String::new(),
            op_type,
            attributes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_index: NodeIndex::default(),
        }
    }

    /// Alias for `new`.
    pub fn new_operator(op_type: String) -> Self {
        Self::new(op_type)
    }

    /// Get the operator type.
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// Get input edge IDs.
    pub fn inputs(&self) -> &[IrEdgeId] {
        &self.inputs
    }

    /// Get output edge IDs.
    pub fn outputs(&self) -> &[IrEdgeId] {
        &self.outputs
    }

    /// Get the node index.
    pub fn node_index(&self) -> IrNodeId {
        self.node_index
    }

    /// Set the node index.
    pub fn set_node_index(&mut self, index: IrNodeId) {
        self.node_index = index;
    }

    /// Add an input edge.
    pub fn add_input(&mut self, edge_id: IrEdgeId) {
        self.inputs.push(edge_id);
    }

    /// Backward-compat alias for `add_input`.
    pub fn add_tensor_input(&mut self, tensor_id: IrEdgeId) -> Result<()> {
        self.add_input(tensor_id);
        Ok(())
    }

    /// Add an output edge.
    pub fn add_output(&mut self, edge_id: IrEdgeId) -> Result<()> {
        self.outputs.push(edge_id);
        Ok(())
    }

    /// Set an attribute.
    pub fn set_attribute(&mut self, key: String, value: AttributeValue) -> Result<()> {
        self.attributes.insert(key, value);
        Ok(())
    }

    /// Get an attribute.
    pub fn get_attribute(&self, key: &str) -> Option<&AttributeValue> {
        self.attributes.get(key)
    }
}

// ──────────────────────────────── EdgeData ───────────────────────────────

/// What compile-time data an edge carries.
///
/// Encodes the three mutually exclusive states an edge can be in:
/// - **Runtime**: no compile-time data — value arrives at runtime (graph input
///   or operator output).
/// - **Initializer**: raw weight bytes from the ONNX model file, parsed
///   on-demand during constant folding.
/// - **Constant**: fully evaluated compile-time value (from folding or a
///   parsed initializer).
#[derive(Debug, Clone)]
pub enum EdgeData {
    /// No compile-time data; value arrives at runtime.
    Runtime,

    /// Raw weight bytes from the ONNX model file, parsed on demand.
    Initializer(Vec<u8>),

    /// Fully evaluated compile-time constant.
    Constant(TensorValue),
}

// ──────────────────────────────── IrEdge ─────────────────────────────────

/// An edge (tensor value flow) in the IR graph.
///
/// Edges carry metadata about the tensor that flows along them:
/// - `name` / `dtype` / `shape` describe the tensor.
/// - `data` describes what compile-time data (if any) the edge holds.
///
/// During constant folding, `data` transitions from `Initializer` or
/// `Runtime` to `Constant`, and the producing operator is removed.
#[derive(Debug, Clone)]
pub struct IrEdge {
    /// Tensor name (must be unique within the graph).
    pub name: String,

    /// Data type.
    pub dtype: DataType,

    /// Shape (static, symbolic, or absent).
    pub shape: TensorShape,

    /// Compile-time data carried by this edge.
    pub data: EdgeData,
}

/// Backward-compatibility alias.
pub type TensorDef = IrEdge;

impl IrEdge {
    /// Create a new runtime edge (no compile-time data).
    pub fn new(name: String, dtype: DataType, shape: TensorShape) -> Self {
        Self {
            name,
            dtype,
            shape,
            data: EdgeData::Runtime,
        }
    }

    /// Create a new edge with initializer data (weight).
    pub fn with_initializer(
        name: String,
        dtype: DataType,
        shape: TensorShape,
        initializer: Vec<u8>,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            data: EdgeData::Initializer(initializer),
        }
    }

    /// Create a new edge with a known constant value.
    pub fn with_constant(
        name: String,
        dtype: DataType,
        shape: TensorShape,
        value: TensorValue,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            data: EdgeData::Constant(value),
        }
    }

    /// Check if this edge has initializer data.
    pub fn has_initializer(&self) -> bool {
        matches!(self.data, EdgeData::Initializer(_))
    }

    /// Check if this edge holds a constant value.
    pub fn is_constant(&self) -> bool {
        matches!(self.data, EdgeData::Constant(_))
    }

    /// Get the initializer bytes, if this edge is an initializer.
    pub fn initializer(&self) -> Option<&[u8]> {
        match &self.data {
            EdgeData::Initializer(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Get the constant value, if this edge holds one.
    pub fn constant_value(&self) -> Option<&TensorValue> {
        match &self.data {
            EdgeData::Constant(value) => Some(value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        let graph = IrGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.tensor_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = IrGraph::new();
        let edge = IrEdge::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2, 3]),
        );
        let edge_id = graph.add_edge(edge);

        assert_eq!(graph.tensor_count(), 1);
        assert_eq!(graph.edge(edge_id).unwrap().name, "x");
        assert_eq!(graph.tensor_by_name("x"), Some(edge_id));
    }

    #[test]
    fn test_add_node() {
        let mut graph = IrGraph::new();

        let input = IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2]),
        );
        let output = IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2]),
        );

        let input_id = graph.add_edge(input);
        let output_id = graph.add_edge(output);

        let mut node = IrNode::new("Relu".to_string());
        node.add_input(input_id);
        node.add_output(output_id).unwrap();
        let node_id = graph.add_node(node);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node(node_id).unwrap().op_type(), "Relu");
        assert_eq!(graph.tensor_producer(output_id), Some(node_id));
        assert_eq!(graph.tensor_consumers(input_id), vec![node_id]);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = IrGraph::new();

        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 2]),
        ));
        let output_id = graph.add_edge(IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 2]),
        ));

        let mut node = IrNode::new("Add".to_string());
        node.add_input(input_id);
        node.add_output(output_id).unwrap();
        let node_id = graph.add_node(node);

        graph.remove_node(node_id).unwrap();

        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.tensor_producer(output_id), None);
        assert_eq!(graph.tensor_consumers(input_id), Vec::<IrNodeId>::new());
    }

    #[test]
    fn test_topological_order() {
        let mut graph = IrGraph::new();

        let t0 = graph.add_edge(IrEdge::new(
            "t0".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let t1 = graph.add_edge(IrEdge::new(
            "t1".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let t2 = graph.add_edge(IrEdge::new(
            "t2".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let t3 = graph.add_edge(IrEdge::new(
            "t3".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));

        let mut node_a = IrNode::new("A".to_string());
        node_a.add_input(t0);
        node_a.add_output(t1).unwrap();
        let id_a = graph.add_node(node_a);

        let mut node_b = IrNode::new("B".to_string());
        node_b.add_input(t1);
        node_b.add_output(t2).unwrap();
        let id_b = graph.add_node(node_b);

        let mut node_c = IrNode::new("C".to_string());
        node_c.add_input(t2);
        node_c.add_output(t3).unwrap();
        let id_c = graph.add_node(node_c);

        let order = graph.topological_order();
        assert_eq!(order, vec![id_a, id_b, id_c]);
    }

    #[test]
    fn test_stable_graph_indices() {
        let mut graph = IrGraph::new();

        let t0 = graph.add_edge(IrEdge::new(
            "t0".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let t1 = graph.add_edge(IrEdge::new(
            "t1".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));
        let t2 = graph.add_edge(IrEdge::new(
            "t2".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
        ));

        let mut node_a = IrNode::new("A".to_string());
        node_a.add_input(t0);
        node_a.add_output(t1).unwrap();
        let id_a = graph.add_node(node_a);

        let mut node_b = IrNode::new("B".to_string());
        node_b.add_input(t1);
        node_b.add_output(t2).unwrap();
        let id_b = graph.add_node(node_b);

        let mut node_c = IrNode::new("C".to_string());
        node_c.add_input(t2);
        let id_c = graph.add_node(node_c);

        // Remove middle node
        graph.remove_node(id_b).unwrap();

        // Original node IDs should still be valid
        assert!(graph.node(id_a).is_ok());
        assert!(graph.node(id_c).is_ok());
    }

    #[test]
    fn test_fold_node_to_constant() {
        use crate::types::TensorData;

        let mut graph = IrGraph::new();

        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        ));
        let output_id = graph.add_edge(IrEdge::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        ));

        let mut node = IrNode::new("Relu".to_string());
        node.add_input(input_id);
        node.add_output(output_id).unwrap();
        let node_id = graph.add_node(node);

        // Create a consumer
        let mut consumer = IrNode::new("Add".to_string());
        consumer.add_input(output_id);
        let consumer_id = graph.add_node(consumer);

        // Fold the node
        let value = TensorValue::new(
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            DataType::F32,
        );

        graph.fold_node_to_constant(node_id, value.clone()).unwrap();

        // Node should be removed
        assert!(graph.node(node_id).is_err());
        assert!(graph.is_fully_folded(node_id).unwrap());

        // Output edge should have the constant value
        let output_edge = graph.edge(output_id).unwrap();
        assert!(output_edge.is_constant());
        assert_eq!(output_edge.constant_value().unwrap().shape, vec![2, 3]);

        // Consumer still references the edge (unchanged)
        let consumer_node = graph.node(consumer_id).unwrap();
        assert_eq!(consumer_node.inputs()[0], output_id);
    }
}
