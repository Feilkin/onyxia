//! Graph representation for ONNX models.
//!
//! This module defines the structured graph representation of ONNX models,
//! providing a stable API independent of the underlying protobuf schema.

use crate::{OnnxError, Result};
use std::collections::HashMap;

/// Unique identifier for a node in the graph.
pub type NodeId = usize;

/// Unique identifier for a tensor in the graph.
pub type TensorId = usize;

/// Internal graph representation of an ONNX model.
#[derive(Debug, Clone)]
pub struct Graph {
    /// All nodes (operations) in the graph.
    pub nodes: Vec<Node>,

    /// All tensors in the graph, indexed by name.
    pub tensors: HashMap<String, TensorId>,

    /// Tensor metadata.
    pub tensor_info: Vec<TensorInfo>,

    /// Names of input tensors.
    pub inputs: Vec<String>,

    /// Names of output tensors.
    pub outputs: Vec<String>,

    /// Graph metadata.
    pub metadata: GraphMetadata,
}

/// Metadata about the graph.
#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    /// Graph name (from ONNX).
    pub name: String,

    /// IR version.
    pub ir_version: i64,

    /// Producer name.
    pub producer_name: String,

    /// Model version.
    pub model_version: i64,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            tensors: HashMap::new(),
            tensor_info: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: GraphMetadata::default(),
        }
    }

    /// Get tensor ID by name.
    pub fn tensor_id(&self, name: &str) -> Result<TensorId> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| OnnxError::MissingTensor(name.to_string()))
    }

    /// Get tensor info by ID.
    pub fn tensor(&self, id: TensorId) -> Result<&TensorInfo> {
        self.tensor_info
            .get(id)
            .ok_or_else(|| OnnxError::InvalidGraph(format!("Invalid tensor ID: {}", id)))
    }

    /// Get tensor info by name.
    pub fn tensor_by_name(&self, name: &str) -> Result<&TensorInfo> {
        let id = self.tensor_id(name)?;
        self.tensor(id)
    }

    /// Add a tensor to the graph.
    pub fn add_tensor(&mut self, info: TensorInfo) -> TensorId {
        let id = self.tensor_info.len();
        let name = info.name.clone();
        self.tensor_info.push(info);
        self.tensors.insert(name, id);
        id
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Validate graph structure.
    pub fn validate(&self) -> Result<()> {
        // Check that all inputs exist
        for input in &self.inputs {
            self.tensor_id(input)?;
        }

        // Check that all outputs exist
        for output in &self.outputs {
            self.tensor_id(output)?;
        }

        // Check that all node inputs/outputs reference valid tensors
        for node in &self.nodes {
            for input in &node.inputs {
                // Skip empty inputs (ONNX uses "" for optional inputs)
                if !input.is_empty() {
                    self.tensor_id(input)?;
                }
            }
            for output in &node.outputs {
                // Skip empty outputs
                if !output.is_empty() {
                    self.tensor_id(output)?;
                }
            }
        }

        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A node (operation) in the graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node name (from ONNX, may be empty).
    pub name: String,

    /// Operation type (e.g., "MatMul", "Add", "Conv").
    pub op_type: String,

    /// Input tensor names.
    pub inputs: Vec<String>,

    /// Output tensor names.
    pub outputs: Vec<String>,

    /// Node attributes.
    pub attributes: HashMap<String, AttributeValue>,

    /// Domain (for custom operators).
    pub domain: String,
}

impl Node {
    /// Create a new node.
    pub fn new(op_type: impl Into<String>) -> Self {
        Self {
            name: String::new(),
            op_type: op_type.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
            domain: String::new(),
        }
    }

    /// Get an attribute value.
    pub fn attr<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<AttributeValue>,
        T::Error: std::fmt::Display,
    {
        let value = self
            .attributes
            .get(name)
            .ok_or_else(|| OnnxError::MissingAttribute(name.to_string()))?;

        T::try_from(value.clone()).map_err(|e| OnnxError::TypeMismatch {
            expected: std::any::type_name::<T>().to_string(),
            actual: format!("{}", e),
        })
    }

    /// Check if an attribute exists.
    pub fn has_attr(&self, name: &str) -> bool {
        self.attributes.contains_key(name)
    }
}

/// Attribute value types.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Vec<u8>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

impl TryFrom<AttributeValue> for f32 {
    type Error = String;

    fn try_from(value: AttributeValue) -> std::result::Result<Self, Self::Error> {
        match value {
            AttributeValue::Float(v) => Ok(v),
            _ => Err("Not a float".to_string()),
        }
    }
}

impl TryFrom<AttributeValue> for i64 {
    type Error = String;

    fn try_from(value: AttributeValue) -> std::result::Result<Self, Self::Error> {
        match value {
            AttributeValue::Int(v) => Ok(v),
            _ => Err("Not an int".to_string()),
        }
    }
}

impl TryFrom<AttributeValue> for String {
    type Error = String;

    fn try_from(value: AttributeValue) -> std::result::Result<Self, Self::Error> {
        match value {
            AttributeValue::String(v) => Ok(v),
            _ => Err("Not a string".to_string()),
        }
    }
}

impl TryFrom<AttributeValue> for Vec<i64> {
    type Error = String;

    fn try_from(value: AttributeValue) -> std::result::Result<Self, Self::Error> {
        match value {
            AttributeValue::Ints(v) => Ok(v),
            _ => Err("Not an int array".to_string()),
        }
    }
}

/// Information about a tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,

    /// Data type.
    pub dtype: DataType,

    /// Tensor shape.
    pub shape: TensorShape,

    /// Tensor kind (input, output, weight, intermediate).
    pub kind: TensorKind,

    /// Initializer data (for weights).
    pub initializer: Option<Vec<u8>>,
}

/// Data types supported by ONNX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    I32,
    I64,
    U8,
    U32,
    Bool,
    /// Quantized 4-bit (custom type for MatMulNBits).
    Q4,
    /// Quantized 8-bit.
    Q8,
}

impl DataType {
    /// Size of this data type in bytes.
    ///
    /// Note: Bool returns 4 bytes to match GPU storage requirements. WGSL does not
    /// support bool in storage buffers, so boolean values are represented as u32
    /// (0 for false, 1 for true) in GPU memory.
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 | DataType::Bool => 4,
            DataType::F16 => 2,
            DataType::I64 => 8,
            DataType::U8 => 1,
            DataType::Q4 => 1, // Packed
            DataType::Q8 => 1,
        }
    }
}

/// Tensor shape representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorShape {
    /// Static shape (all dimensions known).
    Static(Vec<usize>),

    /// Dynamic shape with symbolic dimensions.
    Dynamic(Vec<Dimension>),

    /// Unknown/unspecified shape (not yet inferred).
    Unknown,

    /// Optional input that is absent (ONNX empty string).
    Absent,
}

impl TensorShape {
    /// Check if the shape is fully static.
    pub fn is_static(&self) -> bool {
        matches!(self, TensorShape::Static(_))
    }

    /// Get static dimensions if available.
    pub fn as_static(&self) -> Option<&[usize]> {
        match self {
            TensorShape::Static(dims) => Some(dims),
            _ => None,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> Option<usize> {
        match self {
            TensorShape::Static(dims) => Some(dims.len()),
            TensorShape::Dynamic(dims) => Some(dims.len()),
            TensorShape::Unknown | TensorShape::Absent => None,
        }
    }
}

/// A single dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// Static dimension with known size.
    Static(usize),

    /// Named symbolic dimension (e.g., "batch", "sequence", "N").
    /// The actual value must be provided by the user at runtime.
    Named(String),
}

/// Kind of tensor (determines storage and lifetime).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// Model input (provided by user).
    Input,

    /// Model output (returned to user).
    Output,

    /// Static weight from ONNX (stored in runtime).
    Weight,

    /// Intermediate value computed during execution.
    Intermediate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new();

        let tensor = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 3, 224, 224]),
            kind: TensorKind::Input,
            initializer: None,
        };

        let id = graph.add_tensor(tensor);
        assert_eq!(id, 0);
        assert_eq!(graph.tensor_id("input").unwrap(), 0);
    }

    #[test]
    fn test_node_attributes() {
        let mut node = Node::new("Conv");
        node.attributes
            .insert("kernel_shape".to_string(), AttributeValue::Ints(vec![3, 3]));

        let kernel: Vec<i64> = node.attr("kernel_shape").unwrap();
        assert_eq!(kernel, vec![3, 3]);
    }

    #[test]
    fn test_tensor_shape() {
        let static_shape = TensorShape::Static(vec![1, 2, 3]);
        assert!(static_shape.is_static());
        assert_eq!(static_shape.ndim(), Some(3));

        let dynamic_shape = TensorShape::Dynamic(vec![
            Dimension::Named("batch".to_string()),
            Dimension::Static(512),
        ]);
        assert!(!dynamic_shape.is_static());
        assert_eq!(dynamic_shape.ndim(), Some(2));
    }
}
