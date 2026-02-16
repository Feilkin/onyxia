//! Node inspection utilities.

use anyhow::{Context, Result};
use onyxia_compiler::CompilerPipeline;
use onyxia_core::{DataType, EdgeData, IrGraph, IrNodeId, TensorShape, TensorValue};
use onyxia_onnx::Graph;
use regex::Regex;
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

/// List nodes in the model with optional filtering.
pub fn list_nodes(
    model: &Graph,
    op_types: &[String],
    name_pattern: Option<&str>,
    show_shapes: bool,
    summary: bool,
    dynamic_dims: HashMap<String, usize>,
) -> Result<()> {
    // Convert to IR
    let mut ir_graph = IrGraph::from_onnx(model)?;

    if show_shapes {
        // Run resolution to get actual shapes
        let registry = onyxia_operators::core_operator_registry();
        let mut pipeline = CompilerPipeline::new(dynamic_dims);
        pipeline.run_until_stage(&mut ir_graph, &registry, onyxia_core::Stage::Resolution)?;
    }

    if summary {
        display_summary(&ir_graph)?;
    } else {
        display_filtered_nodes(&ir_graph, op_types, name_pattern, show_shapes)?;
    }

    Ok(())
}

fn display_summary(graph: &IrGraph) -> Result<()> {
    use std::collections::BTreeMap;

    let mut op_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut total = 0;

    for (_, node) in graph.nodes() {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        total += 1;
    }

    println!("Model Summary:");
    println!("  Total nodes: {}", total);
    println!();

    // Sort by count descending
    let mut sorted: Vec<_> = op_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("  Top operators:");
    let top = sorted.iter().take(15);
    for (op_type, count) in top {
        println!("    {:<15} {}", format!("{}:", op_type), count);
    }

    if sorted.len() > 15 {
        let remaining: usize = sorted.iter().skip(15).map(|(_, c)| **c).sum();
        let remaining_types = sorted.len() - 15;
        println!();
        println!(
            "  Other operators: {} types ({} total nodes)",
            remaining_types, remaining
        );
    }

    Ok(())
}

fn display_filtered_nodes(
    graph: &IrGraph,
    op_types: &[String],
    name_pattern: Option<&str>,
    show_shapes: bool,
) -> Result<()> {
    let pattern = name_pattern
        .map(|p| Regex::new(p))
        .transpose()
        .context("Invalid regex pattern")?;

    let mut matching_nodes = Vec::new();

    for (node_id, node) in graph.nodes() {
        // Filter by op_type
        if !op_types.is_empty() && !op_types.contains(&node.op_type) {
            continue;
        }

        // Filter by name pattern
        if let Some(ref regex) = pattern
            && !regex.is_match(&node.name)
        {
            continue;
        }

        matching_nodes.push(node_id);
    }

    if op_types.is_empty() && name_pattern.is_none() {
        println!("Found {} nodes:", matching_nodes.len());
    } else {
        println!("Found {} matching nodes:", matching_nodes.len());
    }

    if show_shapes {
        println!();
        for node_id in matching_nodes {
            display_node_with_shapes(graph, node_id)?;
            println!();
        }
    } else {
        for node_id in matching_nodes {
            let node = graph.node(node_id)?;
            println!("  {}", node.name);
        }
    }

    Ok(())
}

fn display_node_with_shapes(graph: &IrGraph, node_id: IrNodeId) -> Result<()> {
    let node = graph.node(node_id)?;

    println!("{}", node.name);

    for (i, &edge_id) in node.inputs.iter().enumerate() {
        let edge = graph.edge(edge_id)?;
        let shape_desc = format!("{:?}", edge.shape);

        // Check if it's a constant
        let constant_note = if edge.is_constant() {
            " (constant)"
        } else if edge.has_initializer() {
            " (initializer)"
        } else {
            ""
        };

        println!("  Input {}: {}{}", i, shape_desc, constant_note);
    }

    for (i, &edge_id) in node.outputs.iter().enumerate() {
        let edge = graph.edge(edge_id)?;
        println!("  Output {}: {:?}", i, edge.shape);
    }

    Ok(())
}

/// Inspect tensor(s) by name.
pub fn inspect_tensor(
    model: &Graph,
    names: &[String],
    list_constants: bool,
    full_values: bool,
) -> Result<()> {
    let ir_graph = IrGraph::from_onnx(model)?;

    if list_constants {
        list_constant_tensors(&ir_graph)?;
    } else {
        for name in names {
            display_tensor(&ir_graph, name, full_values)?;
            println!();
        }
    }

    Ok(())
}

fn display_tensor(graph: &IrGraph, name: &str, full_values: bool) -> Result<()> {
    let tensor_id = graph
        .find_tensor_by_name(name)
        .with_context(|| format!("Tensor '{}' not found", name))?;

    let tensor = graph.tensor(tensor_id)?;

    println!("Tensor: {}", tensor.name);
    println!("  Type: {:?}", tensor.dtype);
    println!("  Shape: {:?}", tensor.shape);

    // Determine tensor kind
    let kind = match &tensor.data {
        onyxia_core::EdgeData::Runtime => "Runtime tensor",
        onyxia_core::EdgeData::Initializer(_) => "Constant initializer",
        onyxia_core::EdgeData::Constant(_) => "Folded constant",
    };
    println!("  Kind: {}", kind);

    // Show value if it's a constant or initializer
    match &tensor.data {
        onyxia_core::EdgeData::Initializer(init_data) => {
            let total_bytes = init_data.len();
            let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

            if total_mb > 1.0 {
                println!("  Size: {:.2} MB", total_mb);
            }

            println!();

            // Parse and display value
            if let Ok(value) = parse_initializer(tensor.dtype, &tensor.shape, init_data) {
                display_tensor_value(&value, full_values);
            } else {
                println!("  Value: (unable to parse)");
            }
        }
        onyxia_core::EdgeData::Constant(value) => {
            println!();
            display_tensor_value(value, full_values);
        }
        onyxia_core::EdgeData::Runtime => {
            // No value to show for runtime tensors
        }
    }

    // Show consumers
    let consumers = graph.tensor_consumers(tensor_id);
    if !consumers.is_empty() {
        println!();
        println!("  Consumers:");
        for consumer_id in consumers {
            let consumer_name = graph.node_name(consumer_id)?;
            let node = graph.node(consumer_id)?;

            // Find which input index this tensor is
            for (i, &input_id) in node.inputs.iter().enumerate() {
                if input_id == tensor_id {
                    println!("    {} (input {})", consumer_name, i);
                    break;
                }
            }
        }
    }

    Ok(())
}

fn list_constant_tensors(graph: &IrGraph) -> Result<()> {
    let mut constants = Vec::new();

    // Iterate through all edges
    for i in 0..graph.edge_count() {
        let tensor_id = onyxia_core::IrEdgeId::new(i);
        let tensor = graph.tensor(tensor_id)?;

        // Check if it's an initializer or constant
        match &tensor.data {
            onyxia_core::EdgeData::Initializer(_) | onyxia_core::EdgeData::Constant(_) => {
                constants.push((tensor.name.clone(), tensor.dtype, tensor.shape.clone()));
            }
            onyxia_core::EdgeData::Runtime => {
                // Skip runtime tensors
            }
        }
    }

    println!("Found {} constant tensors:", constants.len());
    println!();

    for (name, dtype, shape) in constants {
        println!("  {} : {:?} {:?}", name, dtype, shape);
    }

    Ok(())
}
