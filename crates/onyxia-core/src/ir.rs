//! Intermediate representation for the compiler graph.
//!
//! The IR is built from an ONNX graph and provides a mutable representation
//! suitable for optimization passes and planning. Uses `petgraph::StableGraph`
//! to ensure node/tensor indices remain valid after removals.

use crate::types::{DataType, TensorKind, TensorShape, TensorValue};
use crate::{Error, Result};
use onyxia_onnx::AttributeValue;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::{EdgeRef, Topo};

use std::collections::HashMap;

/// Type alias for IR node identifiers (backed by petgraph NodeIndex).
pub type IrNodeId = NodeIndex;

/// Unique identifier for a tensor in the IR graph.
///
/// This is an index into `IrGraph::tensors`. Unlike node IDs (which use petgraph's
/// stable NodeIndex), tensor IDs are simple usize indices that remain valid across
/// graph mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrTensorId(pub usize);

impl IrTensorId {
    /// Create a new tensor ID.
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Input to an IR node — either a tensor ID or a reference to a Value node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrInput {
    /// Reference to a tensor in the tensor side-table.
    Tensor(IrTensorId),

    /// Reference to a Value node in the graph.
    /// (Not yet used — will be populated in Task 032 during constant folding)
    ValueNode(IrNodeId),
}

impl IrInput {
    /// Create a tensor input.
    pub fn tensor(id: IrTensorId) -> Self {
        Self::Tensor(id)
    }

    /// Create a value node input.
    pub fn value_node(id: IrNodeId) -> Self {
        Self::ValueNode(id)
    }

    /// Get the tensor ID if this is a Tensor input.
    pub fn as_tensor(&self) -> Option<IrTensorId> {
        match self {
            IrInput::Tensor(id) => Some(*id),
            IrInput::ValueNode(_) => None,
        }
    }

    /// Get the node ID if this is a ValueNode input.
    pub fn as_value_node(&self) -> Option<IrNodeId> {
        match self {
            IrInput::Tensor(_) => None,
            IrInput::ValueNode(id) => Some(*id),
        }
    }
}

/// Intermediate representation graph.
///
/// Uses `petgraph::StableGraph` to ensure node indices remain valid after node
/// removal during optimization passes. Tensors are stored in a side-table rather
/// than as graph edges to provide efficient random access and mutation.
pub struct IrGraph {
    /// The graph structure (nodes only, no edge data).
    graph: StableGraph<IrNode, ()>,

    /// Tensor metadata side-table.
    tensors: Vec<TensorDef>,

    /// Lookup table: tensor name → tensor ID.
    tensor_by_name: HashMap<String, IrTensorId>,

    /// Lookup table: tensor ID → producing node ID.
    tensor_producer: HashMap<IrTensorId, IrNodeId>,

    /// Lookup table: tensor ID → consuming node IDs.
    tensor_consumers: HashMap<IrTensorId, Vec<IrNodeId>>,

    /// Graph input tensor IDs.
    pub inputs: Vec<IrTensorId>,

    /// Graph output tensor IDs.
    pub outputs: Vec<IrTensorId>,
}

impl IrGraph {
    /// Create a new empty IR graph.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            tensors: Vec::new(),
            tensor_by_name: HashMap::new(),
            tensor_producer: HashMap::new(),
            tensor_consumers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    // --- Node access ---

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
    pub fn node_inputs(&self, id: IrNodeId) -> Result<&[IrInput]> {
        Ok(self.node(id)?.inputs())
    }

    /// Get the outputs of a node.
    pub fn node_outputs(&self, id: IrNodeId) -> Result<&[IrTensorId]> {
        Ok(self.node(id)?.outputs())
    }

    /// Iterate over all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = (IrNodeId, &IrNode)> {
        self.graph
            .node_indices()
            .filter_map(|id| self.graph.node_weight(id).map(|node| (id, node)))
    }

    // --- Tensor access ---

    /// Get an immutable reference to a tensor.
    pub fn tensor(&self, id: IrTensorId) -> Result<&TensorDef> {
        self.tensors
            .get(id.index())
            .ok_or_else(|| Error::InvalidGraph(format!("Tensor {:?} not found", id)))
    }

    /// Get a mutable reference to a tensor.
    pub fn tensor_mut(&mut self, id: IrTensorId) -> Result<&mut TensorDef> {
        self.tensors
            .get_mut(id.index())
            .ok_or_else(|| Error::InvalidGraph(format!("Tensor {:?} not found", id)))
    }

    /// Look up a tensor by name.
    pub fn tensor_by_name(&self, name: &str) -> Option<IrTensorId> {
        self.tensor_by_name.get(name).copied()
    }

    /// Get the node that produces a tensor, if any.
    pub fn tensor_producer(&self, tensor_id: IrTensorId) -> Option<IrNodeId> {
        self.tensor_producer.get(&tensor_id).copied()
    }

    /// Get the nodes that consume a tensor.
    pub fn tensor_consumers(&self, tensor_id: IrTensorId) -> Vec<IrNodeId> {
        self.tensor_consumers
            .get(&tensor_id)
            .cloned()
            .unwrap_or_default()
    }

    // --- Graph mutation ---

    /// Add a new node to the graph and return its ID.
    ///
    /// This also updates the producer/consumer lookup tables.
    pub fn add_node(&mut self, mut node: IrNode) -> IrNodeId {
        let node_id = self.graph.add_node(IrNode::new_operator(String::new()));
        node.set_node_index(node_id);

        // Only update producer/consumer for Operator nodes
        if let IrNode::Operator {
            inputs, outputs, ..
        } = &node
        {
            // Update producer/consumer lookup tables
            for &output_id in outputs {
                self.tensor_producer.insert(output_id, node_id);
            }

            for input in inputs {
                match input {
                    IrInput::Tensor(tensor_id) => {
                        self.tensor_consumers
                            .entry(*tensor_id)
                            .or_default()
                            .push(node_id);

                        if let Some(producer_id) = self.tensor_producer(*tensor_id) {
                            self.graph.add_edge(producer_id, node_id, ());
                        }
                    }
                    IrInput::ValueNode(value_node_id) => {
                        self.graph.add_edge(*value_node_id, node_id, ());
                    }
                }
            }
        }

        // Update the node in place
        *self.graph.node_weight_mut(node_id).unwrap() = node;

        node_id
    }

    /// Remove a node from the graph.
    ///
    /// This also removes the node from producer/consumer lookup tables. With
    /// `StableGraph`, other node indices remain valid.
    pub fn remove_node(&mut self, id: IrNodeId) -> Result<()> {
        let node = self.node(id)?;

        if let IrNode::Operator {
            inputs, outputs, ..
        } = node
        {
            let inputs = inputs.clone();
            let outputs = outputs.clone();

            // Remove from producer lookup
            for output_id in outputs {
                self.tensor_producer.remove(&output_id);
            }

            // Remove from consumer lookup
            for input in inputs {
                if let IrInput::Tensor(tensor_id) = input
                    && let Some(consumers) = self.tensor_consumers.get_mut(&tensor_id)
                {
                    consumers.retain(|&consumer_id| consumer_id != id);
                }
            }
        }

        // Remove node from graph
        self.graph.remove_node(id);

        Ok(())
    }

    /// Replace a node with a new operation and attributes.
    ///
    /// This is a convenience method for rewrite passes that preserves the node ID
    /// and input/output tensors but changes the operation.
    pub fn replace_node(
        &mut self,
        id: IrNodeId,
        new_op_type: String,
        new_attributes: HashMap<String, AttributeValue>,
    ) -> Result<()> {
        let node = self.node_mut(id)?;
        match node {
            IrNode::Operator {
                op_type,
                attributes,
                ..
            } => {
                *op_type = new_op_type;
                *attributes = new_attributes;
                Ok(())
            }
            IrNode::Value { .. } => Err(Error::InvalidGraph(
                "Cannot replace Value node with operator".to_string(),
            )),
        }
    }

    /// Add a tensor to the graph and return its ID.
    pub fn add_tensor(&mut self, tensor: TensorDef) -> IrTensorId {
        let id = IrTensorId::new(self.tensors.len());
        self.tensor_by_name.insert(tensor.name.clone(), id);
        self.tensors.push(tensor);
        id
    }

    // --- Graph queries ---

    /// Get the topological order of nodes in the graph.
    ///
    /// Returns nodes in an order such that all inputs to a node are produced
    /// before the node itself. This replaces the standalone `Scheduler`.
    pub fn topological_order(&self) -> Vec<IrNodeId> {
        let mut topo = Topo::new(&self.graph);
        let mut order = Vec::new();

        while let Some(node_id) = topo.next(&self.graph) {
            if self.graph.node_weight(node_id).is_some() {
                order.push(node_id);
            }
        }

        order
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of tensors in the graph.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Replace an operator node with a constant value node.
    ///
    /// This is used during constant folding when an operator's output can be
    /// computed at compile time. The operator node is replaced with a Value node,
    /// and all consumers that referenced the operator's output tensor are updated
    /// to reference the value node directly.
    ///
    /// # Arguments
    /// * `node_id` - The operator node to replace
    /// * `output_index` - Which output of the operator to replace (usually 0)
    /// * `value` - The constant value
    ///
    /// # Returns
    /// The new value node's ID (same as the old node_id due to StableGraph)
    ///
    /// # Errors
    /// Returns error if:
    /// - Node is not an Operator
    /// - Output index out of bounds
    /// - Graph structure is inconsistent
    ///
    /// # Example
    ///
    /// ```rust
    /// # use onyxia_core::{IrGraph, IrNode, TensorDef, TensorValue, TensorData, DataType, TensorShape, TensorKind};
    /// // During constant folding, if we can evaluate "Add" at compile time:
    /// let mut graph = IrGraph::new();
    /// let input = graph.add_tensor(TensorDef::new(
    ///     "input".to_string(),
    ///     DataType::F32,
    ///     TensorShape::Static(vec![2]),
    ///     TensorKind::Input,
    /// ));
    /// let output = graph.add_tensor(TensorDef::new(
    ///     "output".to_string(),
    ///     DataType::F32,
    ///     TensorShape::Static(vec![2]),
    ///     TensorKind::Intermediate,
    /// ));
    /// let mut node = IrNode::new_operator("Add".to_string());
    /// node.add_tensor_input(input).unwrap();
    /// node.add_output(output).unwrap();
    /// let add_node_id = graph.add_node(node);
    ///
    /// let result = TensorValue::new(
    ///     TensorData::F32(vec![3.0, 4.0]),
    ///     vec![2],
    ///     DataType::F32,
    /// );
    /// let value_node_id = graph.replace_single_output_with_value(add_node_id, result).unwrap();
    ///
    /// // Consumers now reference the value node:
    /// // OLD: consumer.inputs = [IrInput::Tensor(add_output_tensor_id)]
    /// // NEW: consumer.inputs = [IrInput::ValueNode(value_node_id)]
    /// ```
    pub fn replace_with_value(
        &mut self,
        node_id: IrNodeId,
        output_index: usize,
        value: TensorValue,
    ) -> Result<IrNodeId> {
        // Get the old operator node
        let old_node = self.node(node_id)?.clone();
        let (_op_type, _, inputs, outputs) = old_node
            .as_operator()
            .ok_or_else(|| Error::InvalidGraph("Can only replace Operator nodes".to_string()))?;

        // Validate output index
        if output_index >= outputs.len() {
            return Err(Error::InvalidGraph(format!(
                "Output index {} out of bounds for node with {} outputs",
                output_index,
                outputs.len()
            )));
        }

        let replaced_tensor_id = outputs[output_index];

        // Find all consumers of this output tensor
        let consumers = self.tensor_consumers(replaced_tensor_id);

        // Update each consumer to reference the value node instead of the tensor
        for consumer_id in consumers {
            let consumer = self.node_mut(consumer_id)?;

            if let IrNode::Operator { inputs, .. } = consumer {
                for input in inputs.iter_mut() {
                    if let IrInput::Tensor(tensor_id) = input
                        && *tensor_id == replaced_tensor_id
                    {
                        *input = IrInput::ValueNode(node_id);
                    }
                }
            }
        }

        // Remove the old node's producer/consumer registrations
        for &output_id in outputs {
            self.tensor_producer.remove(&output_id);
        }

        for &input in inputs {
            if let IrInput::Tensor(tensor_id) = input
                && let Some(consumers) = self.tensor_consumers.get_mut(&tensor_id)
            {
                consumers.retain(|&id| id != node_id);
            }
        }

        // Remove old edges (operator's inputs → operator)
        let old_edges: Vec<_> = self
            .graph
            .edges_directed(node_id, petgraph::Direction::Incoming)
            .map(|e| e.id())
            .collect();

        for edge_id in old_edges {
            self.graph.remove_edge(edge_id);
        }

        // Create new Value node with the same node_id
        let value_node = IrNode::Value {
            value,
            node_index: node_id,
        };

        // Replace in graph (StableGraph preserves node_id)
        *self.graph.node_weight_mut(node_id).ok_or_else(|| {
            Error::InvalidGraph("Node disappeared during replacement".to_string())
        })? = value_node;

        Ok(node_id)
    }

    /// Replace an operator node with a value node when it has a single output.
    ///
    /// Convenience wrapper around `replace_with_value()` for the common case
    /// of single-output operators.
    pub fn replace_single_output_with_value(
        &mut self,
        node_id: IrNodeId,
        value: TensorValue,
    ) -> Result<IrNodeId> {
        self.replace_with_value(node_id, 0, value)
    }

    /// Check if an operator node is fully constant-folded.
    ///
    /// Returns true if the node is a Value node (already folded).
    /// This is used during planning to skip nodes that don't need GPU execution.
    pub fn is_fully_folded(&self, node_id: IrNodeId) -> Result<bool> {
        let node = self.node(node_id)?;
        Ok(node.is_value())
    }
}

impl Default for IrGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the IR graph — either an operator or a constant value.
#[derive(Debug, Clone)]
pub enum IrNode {
    /// An operator node that will be executed on the GPU.
    Operator {
        /// ONNX operator type (e.g., "Add", "MatMul", "RmsNorm").
        op_type: String,

        /// Operator attributes (e.g., axis, epsilon, transpose flags).
        attributes: HashMap<String, AttributeValue>,

        /// Input references (tensor IDs or value node IDs).
        inputs: Vec<IrInput>,

        /// Output tensor IDs.
        outputs: Vec<IrTensorId>,

        /// The graph node index (for efficient graph traversal).
        node_index: IrNodeId,
    },

    /// A constant value node (result of constant folding).
    /// Replaces operators whose outputs are fully determined at compile time.
    Value {
        /// The constant value.
        value: TensorValue,

        /// The graph node index.
        node_index: IrNodeId,
    },
}

impl IrNode {
    /// Create a new operator node.
    pub fn new_operator(op_type: String) -> Self {
        Self::Operator {
            op_type,
            attributes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            node_index: NodeIndex::default(),
        }
    }

    /// Create a new operator node (alias for backward compatibility).
    pub fn new(op_type: String) -> Self {
        Self::new_operator(op_type)
    }

    /// Create a new value node.
    pub fn new_value(value: TensorValue) -> Self {
        Self::Value {
            value,
            node_index: NodeIndex::default(),
        }
    }

    /// Check if this is an operator node.
    pub fn is_operator(&self) -> bool {
        matches!(self, IrNode::Operator { .. })
    }

    /// Check if this is a value node.
    pub fn is_value(&self) -> bool {
        matches!(self, IrNode::Value { .. })
    }

    /// Get the operator fields, if this is an operator.
    pub fn as_operator(
        &self,
    ) -> Option<(
        &String,
        &HashMap<String, AttributeValue>,
        &[IrInput],
        &[IrTensorId],
    )> {
        match self {
            IrNode::Operator {
                op_type,
                attributes,
                inputs,
                outputs,
                ..
            } => Some((op_type, attributes, inputs, outputs)),
            IrNode::Value { .. } => None,
        }
    }

    /// Get mutable operator fields, if this is an operator.
    pub fn as_operator_mut(
        &mut self,
    ) -> Option<(
        &mut String,
        &mut HashMap<String, AttributeValue>,
        &mut Vec<IrInput>,
        &mut Vec<IrTensorId>,
    )> {
        match self {
            IrNode::Operator {
                op_type,
                attributes,
                inputs,
                outputs,
                ..
            } => Some((op_type, attributes, inputs, outputs)),
            IrNode::Value { .. } => None,
        }
    }

    /// Get the value, if this is a value node.
    pub fn as_value(&self) -> Option<&TensorValue> {
        match self {
            IrNode::Value { value, .. } => Some(value),
            IrNode::Operator { .. } => None,
        }
    }

    /// Get the node index (works for both variants).
    pub fn node_index(&self) -> IrNodeId {
        match self {
            IrNode::Operator { node_index, .. } => *node_index,
            IrNode::Value { node_index, .. } => *node_index,
        }
    }

    /// Set the node index (works for both variants).
    pub fn set_node_index(&mut self, index: IrNodeId) {
        match self {
            IrNode::Operator { node_index, .. } => *node_index = index,
            IrNode::Value { node_index, .. } => *node_index = index,
        }
    }

    /// Get the op_type if this is an operator (convenience method).
    pub fn op_type(&self) -> Option<&str> {
        match self {
            IrNode::Operator { op_type, .. } => Some(op_type),
            IrNode::Value { .. } => None,
        }
    }

    /// Get inputs if this is an operator.
    pub fn inputs(&self) -> &[IrInput] {
        match self {
            IrNode::Operator { inputs, .. } => inputs,
            IrNode::Value { .. } => &[],
        }
    }

    /// Get outputs if this is an operator.
    pub fn outputs(&self) -> &[IrTensorId] {
        match self {
            IrNode::Operator { outputs, .. } => outputs,
            IrNode::Value { .. } => &[],
        }
    }

    /// Add an input (only for operators).
    pub fn add_input(&mut self, input: IrInput) -> Result<()> {
        match self {
            IrNode::Operator { inputs, .. } => {
                inputs.push(input);
                Ok(())
            }
            IrNode::Value { .. } => Err(Error::InvalidGraph(
                "Cannot add input to Value node".to_string(),
            )),
        }
    }

    /// Add a tensor input (convenience method).
    pub fn add_tensor_input(&mut self, tensor_id: IrTensorId) -> Result<()> {
        self.add_input(IrInput::Tensor(tensor_id))
    }

    /// Add an output (only for operators).
    pub fn add_output(&mut self, tensor_id: IrTensorId) -> Result<()> {
        match self {
            IrNode::Operator { outputs, .. } => {
                outputs.push(tensor_id);
                Ok(())
            }
            IrNode::Value { .. } => Err(Error::InvalidGraph(
                "Cannot add output to Value node".to_string(),
            )),
        }
    }

    /// Set an attribute (only for operators).
    pub fn set_attribute(&mut self, key: String, value: AttributeValue) -> Result<()> {
        match self {
            IrNode::Operator { attributes, .. } => {
                attributes.insert(key, value);
                Ok(())
            }
            IrNode::Value { .. } => Err(Error::InvalidGraph(
                "Cannot set attribute on Value node".to_string(),
            )),
        }
    }

    /// Get an attribute (only for operators).
    pub fn get_attribute(&self, key: &str) -> Option<&AttributeValue> {
        match self {
            IrNode::Operator { attributes, .. } => attributes.get(key),
            IrNode::Value { .. } => None,
        }
    }
}

/// Tensor metadata in the IR graph.
#[derive(Debug, Clone)]
pub struct TensorDef {
    /// Tensor name (must be unique within the graph).
    pub name: String,

    /// Data type.
    pub dtype: DataType,

    /// Shape (static, symbolic, or absent).
    pub shape: TensorShape,

    /// Tensor kind (input, output, intermediate, or constant).
    pub kind: TensorKind,

    /// Constant-folded value (populated during constant folding pass).
    pub value: Option<TensorValue>,

    /// Initializer data for constants (raw bytes).
    pub initializer: Option<Vec<u8>>,
}

impl TensorDef {
    /// Create a new tensor definition.
    pub fn new(name: String, dtype: DataType, shape: TensorShape, kind: TensorKind) -> Self {
        Self {
            name,
            dtype,
            shape,
            kind,
            value: None,
            initializer: None,
        }
    }

    /// Check if this tensor has a constant value.
    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }

    /// Check if this tensor has initializer data.
    pub fn has_initializer(&self) -> bool {
        self.initializer.is_some()
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
    fn test_add_tensor() {
        let mut graph = IrGraph::new();
        let tensor = TensorDef::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2, 3]),
            TensorKind::Input,
        );
        let tensor_id = graph.add_tensor(tensor);

        assert_eq!(graph.tensor_count(), 1);
        assert_eq!(graph.tensor(tensor_id).unwrap().name, "x");
        assert_eq!(graph.tensor_by_name("x"), Some(tensor_id));
    }

    #[test]
    fn test_add_node() {
        let mut graph = IrGraph::new();

        // Add input and output tensors
        let input = TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2]),
            TensorKind::Input,
        );
        let output = TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2]),
            TensorKind::Intermediate,
        );

        let input_id = graph.add_tensor(input);
        let output_id = graph.add_tensor(output);

        // Add node
        let mut node = IrNode::new("Relu".to_string());
        node.add_tensor_input(input_id);
        node.add_output(output_id);

        let node_id = graph.add_node(node);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node(node_id).unwrap().op_type(), Some("Relu"));
        assert_eq!(graph.tensor_producer(output_id), Some(node_id));
        assert_eq!(graph.tensor_consumers(input_id), vec![node_id]);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = IrGraph::new();

        // Add tensors
        let input_id = graph.add_tensor(TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 2]),
            TensorKind::Input,
        ));
        let output_id = graph.add_tensor(TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 2]),
            TensorKind::Intermediate,
        ));

        // Add node
        let mut node = IrNode::new("Add".to_string());
        node.add_tensor_input(input_id);
        node.add_output(output_id);
        let node_id = graph.add_node(node);

        // Remove node
        graph.remove_node(node_id).unwrap();

        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.tensor_producer(output_id), None);
        assert_eq!(graph.tensor_consumers(input_id), Vec::<IrNodeId>::new());
    }

    #[test]
    fn test_topological_order() {
        let mut graph = IrGraph::new();

        // Create a simple chain: A -> B -> C
        let t0 = graph.add_tensor(TensorDef::new(
            "t0".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let t1 = graph.add_tensor(TensorDef::new(
            "t1".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));
        let t2 = graph.add_tensor(TensorDef::new(
            "t2".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));
        let t3 = graph.add_tensor(TensorDef::new(
            "t3".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Output,
        ));

        let mut node_a = IrNode::new("A".to_string());
        node_a.add_tensor_input(t0);
        node_a.add_output(t1);
        let id_a = graph.add_node(node_a);

        let mut node_b = IrNode::new("B".to_string());
        node_b.add_tensor_input(t1);
        node_b.add_output(t2);
        let id_b = graph.add_node(node_b);

        let mut node_c = IrNode::new("C".to_string());
        node_c.add_tensor_input(t2);
        node_c.add_output(t3);
        let id_c = graph.add_node(node_c);

        let order = graph.topological_order();
        assert_eq!(order, vec![id_a, id_b, id_c]);
    }

    #[test]
    fn test_stable_graph_indices() {
        let mut graph = IrGraph::new();

        // Add three nodes
        let t0 = graph.add_tensor(TensorDef::new(
            "t0".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Input,
        ));
        let t1 = graph.add_tensor(TensorDef::new(
            "t1".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));
        let t2 = graph.add_tensor(TensorDef::new(
            "t2".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorKind::Intermediate,
        ));

        let mut node_a = IrNode::new("A".to_string());
        node_a.add_tensor_input(t0);
        node_a.add_output(t1);
        let id_a = graph.add_node(node_a);

        let mut node_b = IrNode::new("B".to_string());
        node_b.add_tensor_input(t1);
        node_b.add_output(t2);
        let _id_b = graph.add_node(node_b);

        let mut node_c = IrNode::new("C".to_string());
        node_c.add_tensor_input(t2).unwrap();
        let id_c = graph.add_node(node_c);

        // Remove middle node
        graph.remove_node(_id_b).unwrap();

        // Original node IDs should still be valid
        assert!(graph.node(id_a).is_ok());
        assert!(graph.node(id_c).is_ok());
    }

    #[test]
    fn test_replace_with_value() {
        use crate::types::TensorData;

        let mut graph = IrGraph::new();

        // Create input tensor
        let input = TensorDef {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Input,
            value: None,
            initializer: None,
        };
        let input_id = graph.add_tensor(input);

        // Create output tensor
        let output = TensorDef {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![2, 3]),
            kind: TensorKind::Intermediate,
            value: None,
            initializer: None,
        };
        let output_id = graph.add_tensor(output);

        // Create operator node
        let mut node = IrNode::new_operator("Relu".to_string());
        node.add_tensor_input(input_id).unwrap();
        node.add_output(output_id).unwrap();
        let node_id = graph.add_node(node);

        // Create consumer node
        let mut consumer = IrNode::new_operator("Add".to_string());
        consumer.add_tensor_input(output_id).unwrap();
        let consumer_id = graph.add_node(consumer);

        // Replace with value
        let value = TensorValue::new(
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            DataType::F32,
        );

        graph
            .replace_single_output_with_value(node_id, value)
            .unwrap();

        // Verify node is now a Value node
        assert!(graph.node(node_id).unwrap().is_value());

        // Verify consumer now references value node
        let consumer_node = graph.node(consumer_id).unwrap();
        match consumer_node.inputs().get(0).unwrap() {
            IrInput::ValueNode(id) => assert_eq!(*id, node_id),
            _ => panic!("Expected ValueNode input"),
        }
    }

    #[test]
    fn test_is_fully_folded() {
        use crate::types::TensorData;

        let mut graph = IrGraph::new();

        // Create regular operator node
        let op_node = IrNode::new_operator("Add".to_string());
        let op_id = graph.add_node(op_node);
        assert!(!graph.is_fully_folded(op_id).unwrap());

        // Create value node
        let value = TensorValue::new(TensorData::F32(vec![1.0, 2.0]), vec![2], DataType::F32);
        let value_node = IrNode::new_value(value);
        let value_id = graph.add_node(value_node);
        assert!(graph.is_fully_folded(value_id).unwrap());
    }
}
