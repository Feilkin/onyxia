//! Parser to convert ONNX models to internal graph representation.

use crate::graph::*;
use crate::onnx::{AttributeProto, ModelProto, NodeProto, TensorProto, tensor_proto};
use crate::{OnnxError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Parse an ONNX ModelProto into a Graph.
///
/// # Arguments
///
/// * `model` - The ONNX ModelProto to parse
/// * `base_dir` - Optional base directory for resolving external data files (relative to model file location)
pub fn parse_model(model: &ModelProto, base_dir: Option<&Path>) -> Result<Graph> {
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
        let tensor_info = parse_initializer(init, TensorKind::Weight, base_dir)?;
        graph.add_tensor(tensor_info);
    }

    // Parse value_info for intermediate tensors (optional type/shape annotations)
    // This provides type and shape information for intermediate values in the graph.
    // ONNX spec: "It is optional for a value to appear in value_info list"
    for value_info_proto in &graph_proto.value_info {
        if !graph.tensors.contains_key(&value_info_proto.name) {
            let tensor_info = parse_value_info(value_info_proto, TensorKind::Intermediate)?;
            graph.add_tensor(tensor_info);
        }
    }

    // Parse nodes and collect intermediate tensors
    for node_proto in &graph_proto.node {
        let node = parse_node(node_proto)?;

        // Register intermediate tensors (outputs that aren't already registered)
        for output in &node.outputs {
            if !graph.tensors.contains_key(output) {
                // Tensor not yet registered - create new TensorInfo
                let tensor_info = if node.op_type == "Constant" {
                    extract_constant_tensor_info(node_proto, output, base_dir)?
                } else {
                    // Default fallback for tensors without value_info entries
                    // Type and shape will be inferred during compilation
                    TensorInfo {
                        name: output.clone(),
                        dtype: DataType::F32, // Default fallback
                        shape: TensorShape::Unknown,
                        kind: TensorKind::Intermediate,
                        initializer: None,
                    }
                };
                graph.add_tensor(tensor_info);
            } else if node.op_type == "Constant" {
                //Tensor already exists (e.g., from value_info), but Constant nodes have the actualdata
                // Extract and update the tensor's initializer
                if let Some(&tensor_id) = graph.tensors.get(output) {
                    let constant_info = extract_constant_tensor_info(node_proto, output, base_dir)?;
                    if let Some(existing_info) = graph.tensor_info.get_mut(tensor_id) {
                        // Update the existing tensor with initializer data and correct dtype/shape
                        existing_info.initializer = constant_info.initializer;
                        existing_info.dtype = constant_info.dtype;
                        existing_info.shape = constant_info.shape;
                    }
                }
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
                other => {
                    return Err(OnnxError::UnsupportedDataType(format!(
                        "value '{name}' has a non-tensor type ({other:?}); \
                         sequences, maps, optionals, and sparse tensors are not supported"
                    )));
                }
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

/// Convert a TensorProto's dims to usizes, rejecting negative values (a
/// corrupt model's `-1` would otherwise wrap to a huge allocation).
fn tensor_dims(tensor: &TensorProto) -> Result<Vec<usize>> {
    tensor
        .dims
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                OnnxError::InvalidModel(format!(
                    "tensor '{}' has a negative dimension ({d})",
                    tensor.name
                ))
            })
        })
        .collect()
}

/// Parse a TensorProto (initializer) into TensorInfo.
fn parse_initializer(
    tensor: &TensorProto,
    kind: TensorKind,
    base_dir: Option<&Path>,
) -> Result<TensorInfo> {
    let name = tensor.name.clone();
    let dtype = parse_data_type(tensor.data_type)?;

    let shape = TensorShape::Static(tensor_dims(tensor)?);

    // Extract raw data - check if it's external or embedded
    let initializer = if tensor.data_location == 1 {
        // EXTERNAL
        // Load from external file
        load_external_data(tensor, base_dir)?
    } else if !tensor.raw_data.is_empty() {
        // Embedded raw data
        Some(tensor.raw_data.clone())
    } else if let Some(bytes) = typed_data_to_raw(tensor, dtype)? {
        // Typed data fields (float_data, int32_data, …) — the older
        // encoding some exporters still emit instead of raw_data.
        Some(bytes)
    } else {
        let numel: usize = tensor_dims(tensor)?.iter().product();
        if numel == 0 {
            Some(Vec::new()) // zero-element tensor: empty data is correct
        } else {
            return Err(OnnxError::InvalidModel(format!(
                "initializer '{}' ({} elements) carries no data: raw_data, \
                 external data, and typed data fields are all empty",
                tensor.name, numel
            )));
        }
    };

    Ok(TensorInfo {
        name,
        dtype,
        shape,
        kind,
        initializer,
    })
}

/// Convert a TensorProto's typed data fields (`float_data`, `int32_data`,
/// `int64_data`, `uint64_data`) into raw little-endian bytes — the same
/// layout `raw_data` uses. Returns `Ok(None)` when no typed field is
/// populated, and an error when a populated field doesn't match the
/// declared dtype or holds a dtype we don't support.
///
/// Per the ONNX spec, `int32_data` carries every sub-32-bit integer type
/// plus bool and (bit-cast) float16; each entry holds one element for
/// dtypes of 8 bits or wider.
fn typed_data_to_raw(tensor: &TensorProto, dtype: DataType) -> Result<Option<Vec<u8>>> {
    let type_mismatch = |field: &str| {
        OnnxError::InvalidModel(format!(
            "initializer '{}': {field} is populated but the declared data type is {dtype:?}",
            tensor.name
        ))
    };

    if !tensor.float_data.is_empty() {
        if dtype != DataType::F32 {
            return Err(type_mismatch("float_data"));
        }
        return Ok(Some(
            tensor.float_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    if !tensor.int64_data.is_empty() {
        if dtype != DataType::I64 {
            return Err(type_mismatch("int64_data"));
        }
        return Ok(Some(
            tensor.int64_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    if !tensor.int32_data.is_empty() {
        let data = &tensor.int32_data;
        let bytes = match dtype {
            DataType::I32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DataType::U8 => data.iter().map(|&v| v as u8).collect(),
            DataType::Bool => data.iter().map(|&v| (v != 0) as u8).collect(),
            // float16 is stored bit-cast in the low 16 bits.
            DataType::F16 => data
                .iter()
                .flat_map(|&v| (v as u16).to_le_bytes())
                .collect(),
            other => {
                return Err(OnnxError::UnsupportedDataType(format!(
                    "initializer '{}': int32_data with data type {other:?}",
                    tensor.name
                )));
            }
        };
        return Ok(Some(bytes));
    }
    if !tensor.uint64_data.is_empty() {
        if dtype != DataType::U32 {
            return Err(type_mismatch("uint64_data"));
        }
        return Ok(Some(
            tensor
                .uint64_data
                .iter()
                .flat_map(|&v| (v as u32).to_le_bytes())
                .collect(),
        ));
    }
    if !tensor.double_data.is_empty() || !tensor.string_data.is_empty() {
        return Err(OnnxError::UnsupportedDataType(format!(
            "initializer '{}': double/string typed data",
            tensor.name
        )));
    }
    Ok(None)
}

/// Inline external tensor data from in-memory buffers into the model.
///
/// Rewrites every external-data tensor (initializers and `Constant` node
/// attributes) so its bytes are embedded as `raw_data` and `data_location`
/// becomes DEFAULT. This lets the rest of the parser run unchanged with no
/// filesystem access — used on the web, where external data is fetched over
/// HTTP into memory rather than read from disk.
///
/// `external` maps each external-data `location` (e.g. `"model.onnx_data"`) to
/// its full byte buffer. Callers should drop those buffers afterwards to free
/// memory, since the bytes are now copied into the model.
pub(crate) fn inline_external_data(
    model: &mut ModelProto,
    external: &HashMap<String, Vec<u8>>,
) -> Result<()> {
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| OnnxError::InvalidGraph("Model has no graph".to_string()))?;

    for tensor in &mut graph.initializer {
        inline_tensor(tensor, external)?;
    }
    for node in &mut graph.node {
        for attr in &mut node.attribute {
            if let Some(tensor) = attr.t.as_mut() {
                inline_tensor(tensor, external)?;
            }
        }
    }
    Ok(())
}

/// Inline a single tensor's external data from `external` into its `raw_data`.
fn inline_tensor(tensor: &mut TensorProto, external: &HashMap<String, Vec<u8>>) -> Result<()> {
    if tensor.data_location != 1 {
        return Ok(()); // not external
    }

    let mut location: Option<String> = None;
    let mut offset: usize = 0;
    let mut length: Option<usize> = None;
    for entry in &tensor.external_data {
        match entry.key.as_str() {
            "location" => location = Some(entry.value.clone()),
            "offset" => {
                offset = entry.value.parse().map_err(|e| {
                    OnnxError::InvalidModel(format!("Invalid offset in external_data: {}", e))
                })?
            }
            "length" => {
                length = Some(entry.value.parse().map_err(|e| {
                    OnnxError::InvalidModel(format!("Invalid length in external_data: {}", e))
                })?)
            }
            _ => {}
        }
    }

    let location = location.ok_or_else(|| {
        OnnxError::InvalidModel("External data missing 'location' key".to_string())
    })?;
    let buffer = external.get(&location).ok_or_else(|| {
        OnnxError::InvalidModel(format!("External data buffer '{}' not provided", location))
    })?;

    let end = match length {
        Some(len) => offset + len,
        None => buffer.len(),
    };
    if end > buffer.len() {
        return Err(OnnxError::InvalidModel(format!(
            "External data range {}..{} out of bounds for '{}' (len {})",
            offset,
            end,
            location,
            buffer.len()
        )));
    }

    tensor.raw_data = buffer[offset..end].to_vec();
    tensor.data_location = 0; // DEFAULT (embedded)
    tensor.external_data.clear();
    Ok(())
}

/// Load external tensor data from a file.
fn load_external_data(tensor: &TensorProto, base_dir: Option<&Path>) -> Result<Option<Vec<u8>>> {
    // Parse external_data key-value pairs
    let mut location: Option<String> = None;
    let mut offset: u64 = 0;
    let mut length: Option<usize> = None;

    for entry in &tensor.external_data {
        match entry.key.as_str() {
            "location" => location = Some(entry.value.clone()),
            "offset" => {
                offset = entry.value.parse().map_err(|e| {
                    OnnxError::InvalidModel(format!("Invalid offset in external_data: {}", e))
                })?
            }
            "length" => {
                length = Some(entry.value.parse().map_err(|e| {
                    OnnxError::InvalidModel(format!("Invalid length in external_data: {}", e))
                })?)
            }
            _ => {} // Ignore unknown keys (e.g., checksum)
        }
    }

    let location = location.ok_or_else(|| {
        OnnxError::InvalidModel("External data missing 'location' key".to_string())
    })?;

    // Resolve path relative to base_dir
    let external_path = if let Some(base) = base_dir {
        base.join(&location)
    } else {
        PathBuf::from(&location)
    };

    // Open file and read data
    let mut file = File::open(&external_path).map_err(|e| {
        OnnxError::InvalidModel(format!(
            "Failed to open external data file '{}': {}",
            external_path.display(),
            e
        ))
    })?;

    // Seek to offset if specified
    if offset > 0 {
        file.seek(SeekFrom::Start(offset))?;
    }

    // Read data
    let data = if let Some(len) = length {
        // Read exactly 'length' bytes
        let mut buffer = vec![0u8; len];
        file.read_exact(&mut buffer)?;
        buffer
    } else {
        // Read to end of file
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        buffer
    };

    Ok(Some(data))
}

/// Extract TensorInfo from a Constant node's tensor attribute, including shape and data.
fn extract_constant_tensor_info(
    node: &NodeProto,
    name: &str,
    base_dir: Option<&Path>,
) -> Result<TensorInfo> {
    // Find the "value" attribute which contains the TensorProto
    for attr in &node.attribute {
        if attr.name == "value"
            && let Some(ref tensor) = attr.t
        {
            let shape = TensorShape::Static(tensor_dims(tensor)?);
            let dtype = parse_data_type(tensor.data_type)?;

            // Extract raw data - check if it's external or embedded
            let initializer = if tensor.data_location == 1 {
                // EXTERNAL
                // Load from external file
                load_external_data(tensor, base_dir)?
            } else if !tensor.raw_data.is_empty() {
                Some(tensor.raw_data.clone())
            } else {
                typed_data_to_raw(tensor, dtype)?
            };

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
            let tensor = attr.t.as_ref().ok_or_else(|| {
                OnnxError::InvalidGraph(format!(
                    "tensor attribute '{}' carries no tensor",
                    attr.name
                ))
            })?;
            if tensor.data_location == 1 {
                // External data in an attribute tensor needs a base dir we
                // don't have here; `inline_external_data` (the web path)
                // rewrites these to embedded before parsing.
                return Err(OnnxError::InvalidModel(format!(
                    "tensor attribute '{}' uses external data, which is only \
                     supported after inlining",
                    attr.name
                )));
            }
            let dtype = parse_data_type(tensor.data_type)?;
            let data = if !tensor.raw_data.is_empty() {
                tensor.raw_data.clone()
            } else {
                typed_data_to_raw(tensor, dtype)?.unwrap_or_default()
            };
            AttributeValue::Tensor(crate::graph::AttrTensor {
                dtype,
                dims: tensor_dims(tensor)?,
                data,
            })
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

        let graph = parse_model(&model, None).unwrap();
        assert_eq!(graph.metadata.name, "test_graph");
        assert_eq!(graph.metadata.ir_version, 8);
    }

    /// Initializers using the typed data fields (`float_data`, …) instead
    /// of `raw_data` must convert to the same little-endian byte layout.
    #[test]
    fn typed_data_initializers_convert_to_raw_bytes() {
        use crate::onnx::tensor_proto::DataType as DT;

        let cases: Vec<(TensorProto, Vec<u8>)> = vec![
            (
                TensorProto {
                    data_type: DT::Float as i32,
                    dims: vec![2],
                    float_data: vec![1.5, -2.0],
                    ..Default::default()
                },
                [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat(),
            ),
            (
                TensorProto {
                    data_type: DT::Int64 as i32,
                    dims: vec![2],
                    int64_data: vec![-1, 7],
                    ..Default::default()
                },
                [(-1i64).to_le_bytes(), 7i64.to_le_bytes()].concat(),
            ),
            (
                TensorProto {
                    data_type: DT::Uint8 as i32,
                    dims: vec![3],
                    int32_data: vec![0, 128, 255],
                    ..Default::default()
                },
                vec![0, 128, 255],
            ),
            (
                TensorProto {
                    data_type: DT::Bool as i32,
                    dims: vec![2],
                    int32_data: vec![1, 0],
                    ..Default::default()
                },
                vec![1, 0],
            ),
        ];
        for (tensor, expect) in cases {
            let info = parse_initializer(&tensor, TensorKind::Weight, None).unwrap();
            assert_eq!(info.initializer.as_deref(), Some(expect.as_slice()));
        }
    }

    /// An initializer with a populated typed field of the *wrong* type, or
    /// no data at all, must error instead of silently dropping the weight.
    #[test]
    fn initializer_without_data_errors() {
        use crate::onnx::tensor_proto::DataType as DT;

        let empty = TensorProto {
            name: "w".into(),
            data_type: DT::Float as i32,
            dims: vec![4],
            ..Default::default()
        };
        assert!(parse_initializer(&empty, TensorKind::Weight, None).is_err());

        let mismatched = TensorProto {
            name: "w".into(),
            data_type: DT::Int64 as i32,
            dims: vec![1],
            float_data: vec![1.0],
            ..Default::default()
        };
        assert!(parse_initializer(&mismatched, TensorKind::Weight, None).is_err());
    }

    /// Tensor attributes must keep their dtype and shape (regression: they
    /// were dropped entirely, so ConstantOfShape lost its fill value).
    #[test]
    fn tensor_attribute_preserves_dtype_and_data() {
        use crate::onnx::attribute_proto::AttributeType;
        use crate::onnx::tensor_proto::DataType as DT;

        let attr = AttributeProto {
            name: "value".into(),
            r#type: AttributeType::Tensor as i32,
            t: Some(TensorProto {
                data_type: DT::Int64 as i32,
                dims: vec![1],
                int64_data: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        };
        let parsed = parse_attribute(&attr).unwrap().expect("attribute kept");
        let AttributeValue::Tensor(t) = parsed else {
            panic!("expected tensor attribute, got {parsed:?}");
        };
        assert_eq!(t.dtype, DataType::I64);
        assert_eq!(t.dims, vec![1]);
        assert_eq!(t.data, 1i64.to_le_bytes());
    }
}
