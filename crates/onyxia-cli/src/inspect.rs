//! Node inspection utilities.

use anyhow::{Context, Result};
use onyxia_compiler::CompilerPipeline;
use onyxia_core::{DataType, EdgeData, IrGraph, IrNodeId, TensorShape, TensorValue};
use onyxia_onnx::Graph;
use std::collections::HashMap;

/// Display detailed information about one or more nodes.
pub fn inspect_nodes(
    model: &Graph,
    node_names: &[String],
    dynamic_dims: HashMap<String, usize>,
    full_values: bool,
) -> Result<()> {
    // Convert to IR
    let ir_graph = IrGraph::from_onnx(model)?;

    // Run the compiler pipeline up to inference stage to resolve shapes
    let registry = onyxia_operators::core_operator_registry();
    let mut pipeline = CompilerPipeline::new(dynamic_dims);

    // Full compilation includes all passes including shape inference
    pipeline
        .compile(model, &registry)
        .context("Failed to compile model for inspection")?;

    // Find and display each requested node
    for node_name in node_names {
        let node_id = ir_graph
            .find_node_by_name(node_name)
            .with_context(|| format!("Node '{}' not found", node_name))?;

        display_node(&ir_graph, node_id, full_values)?;
        println!(); // Blank line between nodes
    }

    Ok(())
}

fn display_node(graph: &IrGraph, node_id: IrNodeId, full_values: bool) -> Result<()> {
    let node = graph.node(node_id)?;

    println!("Node: {} ({})", node.name, node.op_type);

    // Show attributes
    if node.attributes.is_empty() {
        println!("  Attributes: (none)");
    } else {
        println!("  Attributes:");
        for (key, value) in &node.attributes {
            println!("    {}: {:?}", key, value);
        }
    }

    // Show inputs
    println!("  Inputs:");
    for (i, &edge_id) in node.inputs.iter().enumerate() {
        display_edge(graph, edge_id, i, full_values)?;
    }

    // Show outputs
    println!("  Outputs:");
    for (i, &edge_id) in node.outputs.iter().enumerate() {
        let edge = graph.edge(edge_id)?;
        println!("    {}: {}", i, edge.name);
        println!("       Shape: {:?}", edge.shape);
        println!("       Type: {:?}", edge.dtype);
    }

    Ok(())
}

fn display_edge(
    graph: &IrGraph,
    edge_id: onyxia_core::IrEdgeId,
    index: usize,
    full_values: bool,
) -> Result<()> {
    let edge = graph.edge(edge_id)?;

    println!("    {}: {}", index, edge.name);
    println!("       Shape: {:?}", edge.shape);
    println!("       Type: {:?}", edge.dtype);

    // Check what kind of data this edge carries
    match &edge.data {
        EdgeData::Runtime => {
            println!("       Source: Runtime tensor");
        }
        EdgeData::Initializer(raw_data) => {
            println!("       Source: Constant initializer");

            // Try to parse and display the value
            if let Ok(value) = parse_initializer(edge.dtype, &edge.shape, raw_data) {
                display_tensor_value(&value, full_values);
            }
        }
        EdgeData::Constant(value) => {
            println!("       Source: Folded constant");
            display_tensor_value(value, full_values);
        }
    }

    Ok(())
}

fn display_tensor_value(value: &TensorValue, full: bool) {
    const MAX_ELEMENTS: usize = 20;

    match &value.data {
        onyxia_core::TensorData::I64(v) => {
            if full || v.len() <= MAX_ELEMENTS {
                println!("       Value: {:?}", v);
            } else {
                println!(
                    "       Value: {:?} ... ({} elements total, use --full-values to see all)",
                    &v[..MAX_ELEMENTS],
                    v.len()
                );
            }
        }
        onyxia_core::TensorData::I32(v) => {
            if full || v.len() <= MAX_ELEMENTS {
                println!("       Value: {:?}", v);
            } else {
                println!(
                    "       Value: {:?} ... ({} elements total, use --full-values to see all)",
                    &v[..MAX_ELEMENTS],
                    v.len()
                );
            }
        }
        onyxia_core::TensorData::F32(v) => {
            if full || v.len() <= MAX_ELEMENTS {
                println!("       Value: {:?}", v);
            } else {
                println!(
                    "       Value: {:?} ... ({} elements total, use --full-values to see all)",
                    &v[..MAX_ELEMENTS],
                    v.len()
                );
            }
        }
        onyxia_core::TensorData::Bool(v) => {
            if full || v.len() <= MAX_ELEMENTS {
                println!("       Value: {:?}", v);
            } else {
                println!(
                    "       Value: {:?} ... ({} elements total, use --full-values to see all)",
                    &v[..MAX_ELEMENTS],
                    v.len()
                );
            }
        }
        onyxia_core::TensorData::U8(v) => {
            if full || v.len() <= MAX_ELEMENTS {
                println!("       Value: {:?}", v);
            } else {
                println!(
                    "       Value: {:?} ... ({} elements total, use --full-values to see all)",
                    &v[..MAX_ELEMENTS],
                    v.len()
                );
            }
        }
    }
}

/// Parse raw initializer data into a TensorValue.
fn parse_initializer(dtype: DataType, shape: &TensorShape, data: &[u8]) -> Result<TensorValue> {
    use std::mem::size_of;

    // Get the static dimensions for parsing
    let dims = match shape {
        TensorShape::Static(dims) => dims.clone(),
        _ => {
            anyhow::bail!(
                "Cannot parse initializer with non-static shape: {:?}",
                shape
            );
        }
    };

    // Calculate expected element count
    let element_count: usize = dims.iter().product();

    // Parse based on dtype
    let tensor_data = match dtype {
        DataType::I64 => {
            let expected_bytes = element_count * size_of::<i64>();
            if data.len() != expected_bytes {
                anyhow::bail!(
                    "Initializer size mismatch: expected {} bytes, got {}",
                    expected_bytes,
                    data.len()
                );
            }

            let mut values = Vec::with_capacity(element_count);
            for chunk in data.chunks_exact(size_of::<i64>()) {
                let bytes: [u8; 8] = chunk.try_into()?;
                values.push(i64::from_le_bytes(bytes));
            }
            onyxia_core::TensorData::I64(values)
        }
        DataType::I32 => {
            let expected_bytes = element_count * size_of::<i32>();
            if data.len() != expected_bytes {
                anyhow::bail!(
                    "Initializer size mismatch: expected {} bytes, got {}",
                    expected_bytes,
                    data.len()
                );
            }

            let mut values = Vec::with_capacity(element_count);
            for chunk in data.chunks_exact(size_of::<i32>()) {
                let bytes: [u8; 4] = chunk.try_into()?;
                values.push(i32::from_le_bytes(bytes));
            }
            onyxia_core::TensorData::I32(values)
        }
        DataType::F32 => {
            let expected_bytes = element_count * size_of::<f32>();
            if data.len() != expected_bytes {
                anyhow::bail!(
                    "Initializer size mismatch: expected {} bytes, got {}",
                    expected_bytes,
                    data.len()
                );
            }

            let mut values = Vec::with_capacity(element_count);
            for chunk in data.chunks_exact(size_of::<f32>()) {
                let bytes: [u8; 4] = chunk.try_into()?;
                values.push(f32::from_le_bytes(bytes));
            }
            onyxia_core::TensorData::F32(values)
        }
        DataType::Bool => {
            if data.len() != element_count {
                anyhow::bail!(
                    "Initializer size mismatch: expected {} bytes, got {}",
                    element_count,
                    data.len()
                );
            }

            let values = data.iter().map(|&b| b != 0).collect();
            onyxia_core::TensorData::Bool(values)
        }
        DataType::U8 => {
            if data.len() != element_count {
                anyhow::bail!(
                    "Initializer size mismatch: expected {} bytes, got {}",
                    element_count,
                    data.len()
                );
            }

            onyxia_core::TensorData::U8(data.to_vec())
        }
        _ => {
            anyhow::bail!("Unsupported data type for initializer parsing: {:?}", dtype);
        }
    };

    Ok(TensorValue {
        data: tensor_data,
        shape: dims,
        dtype,
    })
}
