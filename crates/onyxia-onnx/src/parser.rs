//! Parser to convert ONNX models to internal graph representation.

use crate::graph::*;
use crate::onnx::{AttributeProto, ModelProto, NodeProto, TensorProto, tensor_proto};
use crate::{OnnxError, Result};
use std::collections::HashMap;

/// Parse an ONNX ModelProto into a Graph.
pub fn parse_model(model: &ModelProto) -> Result<Graph> {
    let graph_proto = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidGraph("Model has no graph".to_string()))?;

    let mut graph = Graph::new();

    // Set metadata
    graph.metadata.name = graph_proto.name.clone();
    graph.metadata.ir_version = model.ir_version;
    graph.metadata.producer_name = model.producer_name.clone();
    graph.metadata.model_version = model.model_version;

    // Parse initializers (weights) first
    let mut initializers: HashMap<String, &TensorProto> = HashMap::new();
    for init in &graph_proto.initializer {
        initializers.insert(init.name.clone(), init);
    }

    // Parse all tensors (inputs, outputs, and intermediates)
    // Start with inputs
    for input in &graph_proto.input {
        // Skip if it's an initializer (it's a weight, not an input)
        if initializers.contains_key(&input.name) {
            continue;
        }

        let tensor_info = parse_value_info(input, TensorKind::Input)?;
        graph.add_tensor(tensor_info);
        graph.inputs.push(input.name.clone());
    }

    // Parse outputs
    for output in &graph_proto.output {
        let tensor_info = parse_value_info(output, TensorKind::Output)?;
        graph.add_tensor(tensor_info);
        graph.outputs.push(output.name.clone());
    }

    // Parse initializers as weight tensors
    for init in initializers.values() {
        let tensor_info = parse_initializer(init, TensorKind::Weight)?;
        graph.add_tensor(tensor_info);
    }

    // Parse nodes and collect intermediate tensors
    for node_proto in &graph_proto.node {
        let node = parse_node(node_proto)?;

        // Register intermediate tensors (outputs that aren't already registered)
        for output in &node.outputs {
            if !graph.tensors.contains_key(output) {
                // Special handling for Constant nodes - extract shape and data from tensor attribute
                let tensor_info = if node.op_type == "Constant" {
                    extract_constant_tensor_info(node_proto, output)?
                } else {
                    TensorInfo {
                        name: output.clone(),
                        dtype: DataType::F32, // Default, may be inferred later
                        shape: TensorShape::Unknown,
                        kind: TensorKind::Intermediate,
                        initializer: None,
                    }
                };
                graph.add_tensor(tensor_info);
            }
        }

        graph.add_node(node);
    }

    // Validate the graph
    graph.validate()?;

    Ok(graph)
}

/// Parse a ValueInfoProto into TensorInfo.
fn parse_value_info(
    value_info: &crate::onnx::ValueInfoProto,
    kind: TensorKind,
) -> Result<TensorInfo> {
    let name = value_info.name.clone();

    let (dtype, shape) = if let Some(type_proto) = &value_info.r#type {
        // TypeProto has a oneof "value" field
        if let Some(value) = &type_proto.value {
            use crate::onnx::type_proto::Value;
            match value {
                Value::TensorType(tensor_type) => {
                    let dtype = parse_data_type(tensor_type.elem_type)?;
                    let shape = parse_shape(&tensor_type.shape)?;
                    (dtype, shape)
                }
                _ => (DataType::F32, TensorShape::Unknown), // Non-tensor types
            }
        } else {
            (DataType::F32, TensorShape::Unknown)
        }
    } else {
        (DataType::F32, TensorShape::Unknown)
    };

    Ok(TensorInfo {
        name,
        dtype,
        shape,
        kind,
        initializer: None,
    })
}

/// Parse a TensorProto (initializer) into TensorInfo.
fn parse_initializer(tensor: &TensorProto, kind: TensorKind) -> Result<TensorInfo> {
    let name = tensor.name.clone();
    let dtype = parse_data_type(tensor.data_type)?;

    let shape = if tensor.dims.is_empty() {
        TensorShape::Static(vec![])
    } else {
        TensorShape::Static(tensor.dims.iter().map(|&d| d as usize).collect())
    };

    // Extract raw data
    let initializer = if !tensor.raw_data.is_empty() {
        Some(tensor.raw_data.clone())
    } else {
        // Handle typed data fields (float_data, int32_data, etc.)
        None // TODO: Convert typed data to raw bytes
    };

    Ok(TensorInfo {
        name,
        dtype,
        shape,
        kind,
        initializer,
    })
}

/// Extract TensorInfo from a Constant node's tensor attribute, including shape and data.
fn extract_constant_tensor_info(node: &NodeProto, name: &str) -> Result<TensorInfo> {
    // Find the "value" attribute which contains the TensorProto
    for attr in &node.attribute {
        if attr.name == "value"
            && let Some(ref tensor) = attr.t
        {
            let shape = if tensor.dims.is_empty() {
                TensorShape::Static(vec![]) // Scalar
            } else {
                TensorShape::Static(tensor.dims.iter().map(|&d| d as usize).collect())
            };

            let dtype = tensor_data_type_to_dtype(tensor.data_type);

            // Extract raw data from tensor
            // Data can be in `raw_data` or in typed arrays like `int64_data`, `int32_data`, etc.
            let initializer = extract_tensor_raw_data(tensor);

            return Ok(TensorInfo {
                name: name.to_string(),
                dtype,
                shape,
                kind: TensorKind::Intermediate,
                initializer,
            });
        }
    }

    // No tensor attribute found - shouldn't happen for valid Constant nodes
    Ok(TensorInfo {
        name: name.to_string(),
        dtype: DataType::F32,
        shape: TensorShape::Unknown,
        kind: TensorKind::Intermediate,
        initializer: None,
    })
}

/// Extract raw data from a TensorProto, handling both raw_data and typed arrays.
fn extract_tensor_raw_data(tensor: &TensorProto) -> Option<Vec<u8>> {
    // If raw_data is present, use it directly
    if !tensor.raw_data.is_empty() {
        return Some(tensor.raw_data.clone());
    }

    // Otherwise, try typed arrays
    // Handle int64_data
    if !tensor.int64_data.is_empty() {
        let bytes: Vec<u8> = tensor
            .int64_data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        return Some(bytes);
    }

    // Handle int32_data
    if !tensor.int32_data.is_empty() {
        let bytes: Vec<u8> = tensor
            .int32_data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        return Some(bytes);
    }

    // Handle float_data
    if !tensor.float_data.is_empty() {
        let bytes: Vec<u8> = tensor
            .float_data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        return Some(bytes);
    }

    // Handle double_data
    if !tensor.double_data.is_empty() {
        let bytes: Vec<u8> = tensor
            .double_data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        return Some(bytes);
    }

    None
}

/// Convert ONNX TensorProto data type to internal DataType.
fn tensor_data_type_to_dtype(data_type: i32) -> DataType {
    use crate::onnx::tensor_proto::DataType as OnnxDataType;
    match OnnxDataType::try_from(data_type) {
        Ok(OnnxDataType::Float) => DataType::F32,
        Ok(OnnxDataType::Float16) => DataType::F16,
        Ok(OnnxDataType::Int32) => DataType::I32,
        Ok(OnnxDataType::Int64) => DataType::I64,
        Ok(OnnxDataType::Uint8) => DataType::U8,
        Ok(OnnxDataType::Uint32) => DataType::U32,
        Ok(OnnxDataType::Bool) => DataType::Bool,
        _ => DataType::F32, // Default
    }
}

/// Extract shape from a Constant node's tensor attribute (for backwards compatibility).
#[allow(dead_code)]
fn extract_constant_shape(node: &NodeProto) -> Result<TensorShape> {
    // Find the "value" attribute which contains the TensorProto
    for attr in &node.attribute {
        if attr.name == "value"
            && let Some(ref tensor) = attr.t
        {
            let shape = if tensor.dims.is_empty() {
                TensorShape::Static(vec![]) // Scalar
            } else {
                TensorShape::Static(tensor.dims.iter().map(|&d| d as usize).collect())
            };
            return Ok(shape);
        }
    }
    // No tensor attribute found - shouldn't happen for valid Constant nodes
    Ok(TensorShape::Unknown)
}

/// Parse a NodeProto into a Node.
fn parse_node(node: &NodeProto) -> Result<Node> {
    let mut parsed_node = Node::new(node.op_type.clone());
    parsed_node.name = node.name.clone();
    parsed_node.domain = node.domain.clone();
    parsed_node.inputs = node.input.clone();
    parsed_node.outputs = node.output.clone();

    // Parse attributes
    for attr in &node.attribute {
        if let Some(value) = parse_attribute(attr)? {
            parsed_node.attributes.insert(attr.name.clone(), value);
        }
    }

    Ok(parsed_node)
}

/// Parse an AttributeProto into an AttributeValue.
fn parse_attribute(attr: &AttributeProto) -> Result<Option<AttributeValue>> {
    use crate::onnx::attribute_proto::AttributeType;

    let attr_type = AttributeType::try_from(attr.r#type)
        .map_err(|_| OnnxError::InvalidGraph(format!("Invalid attribute type: {}", attr.r#type)))?;

    let value = match attr_type {
        AttributeType::Float => AttributeValue::Float(attr.f),
        AttributeType::Int => AttributeValue::Int(attr.i),
        AttributeType::String => {
            AttributeValue::String(String::from_utf8(attr.s.clone()).map_err(|e| {
                OnnxError::InvalidGraph(format!("Invalid UTF-8 in attribute: {}", e))
            })?)
        }
        AttributeType::Floats => AttributeValue::Floats(attr.floats.clone()),
        AttributeType::Ints => AttributeValue::Ints(attr.ints.clone()),
        AttributeType::Strings => {
            let strings: Result<Vec<String>> = attr
                .strings
                .iter()
                .map(|s| {
                    String::from_utf8(s.clone()).map_err(|e| {
                        OnnxError::InvalidGraph(format!("Invalid UTF-8 in attribute: {}", e))
                    })
                })
                .collect();
            AttributeValue::Strings(strings?)
        }
        AttributeType::Tensor => {
            // Tensor attributes are not fully parsed yet
            // For Constant nodes, shape is extracted separately during node parsing
            return Ok(None);
        }
        _ => return Ok(None), // Unsupported attribute types
    };

    Ok(Some(value))
}

/// Parse ONNX data type into our DataType.
fn parse_data_type(onnx_type: i32) -> Result<DataType> {
    use tensor_proto::DataType as OnnxDataType;

    let onnx_dt = OnnxDataType::try_from(onnx_type)
        .map_err(|_| OnnxError::InvalidGraph(format!("Unknown data type: {}", onnx_type)))?;

    match onnx_dt {
        OnnxDataType::Float => Ok(DataType::F32),
        OnnxDataType::Float16 => Ok(DataType::F16),
        OnnxDataType::Int32 => Ok(DataType::I32),
        OnnxDataType::Int64 => Ok(DataType::I64),
        OnnxDataType::Uint8 => Ok(DataType::U8),
        OnnxDataType::Uint32 => Ok(DataType::U32),
        OnnxDataType::Bool => Ok(DataType::Bool),
        _ => Err(OnnxError::UnsupportedDataType(format!(
            "Unsupported data type: {:?}",
            onnx_dt
        ))),
    }
}

/// Parse ONNX shape into our TensorShape.
fn parse_shape(shape_proto: &Option<crate::onnx::TensorShapeProto>) -> Result<TensorShape> {
    let shape_proto = match shape_proto {
        Some(s) => s,
        None => return Ok(TensorShape::Unknown),
    };

    if shape_proto.dim.is_empty() {
        return Ok(TensorShape::Static(vec![]));
    }

    let mut dimensions = Vec::new();
    let mut all_static = true;

    for dim_proto in &shape_proto.dim {
        use crate::onnx::tensor_shape_proto::dimension::Value;

        let dimension = match &dim_proto.value {
            Some(Value::DimValue(v)) => Dimension::Static(*v as usize),
            Some(Value::DimParam(name)) => {
                all_static = false;
                Dimension::Named(name.clone())
            }
            None => {
                all_static = false;
                Dimension::Named("unknown".to_string())
            }
        };

        dimensions.push(dimension);
    }

    if all_static {
        // Convert to static shape
        let static_dims: Vec<usize> = dimensions
            .iter()
            .map(|d| match d {
                Dimension::Static(v) => *v,
                _ => unreachable!(),
            })
            .collect();
        Ok(TensorShape::Static(static_dims))
    } else {
        Ok(TensorShape::Dynamic(dimensions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_model() {
        let model = ModelProto {
            ir_version: 8,
            graph: Some(crate::onnx::GraphProto {
                name: "test_graph".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph = parse_model(&model).unwrap();
        assert_eq!(graph.metadata.name, "test_graph");
        assert_eq!(graph.metadata.ir_version, 8);
    }
}
