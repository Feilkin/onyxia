//! Node inspection utilities.

use crate::{TraceDirection, TraceFormat};
use anyhow::{Context, Result};
use onyxia_compiler::CompilerPipeline;
use onyxia_core::{DataType, EdgeData, IrGraph, IrNodeId, TensorShape, TensorValue};
use onyxia_onnx::Graph;
use regex::Regex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

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
    let mut pipeline = CompilerPipeline::new();

    // Note: Dynamic dimensions are no longer passed at compile time in the simplified pipeline
    // TODO: Update inspection workflow when operator dispatch is fully implemented
    let _ = dynamic_dims; // Suppress unused warning

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
    let ir_graph = IrGraph::from_onnx(model)?;

    if show_shapes {
        // Run resolution to get actual shapes
        let registry = onyxia_operators::core_operator_registry();
        let pipeline = CompilerPipeline::new();
        // Note: Dynamic dimensions no longer passed at compile time
        let _ = dynamic_dims; // Suppress unused warning
        // Note: Partial compilation (run_until_stage) not supported in simplified pipeline
        // Would need full compile here, but that's expensive for just listing nodes
        let _ = (pipeline, registry); // Suppress unused warnings
        // TODO: Re-evaluate shape display when dispatch model is complete
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

/// Trace data flow around a specific node.
pub fn trace_node(
    model: &Graph,
    node_name: &str,
    depth: usize,
    direction: TraceDirection,
    format: TraceFormat,
    output: Option<&Path>,
) -> Result<()> {
    let ir_graph = IrGraph::from_onnx(model)?;

    let node_id = ir_graph
        .find_node_by_name(node_name)
        .with_context(|| format!("Node '{}' not found", node_name))?;

    // Collect subgraph
    let subgraph = collect_subgraph(&ir_graph, node_id, depth, direction)?;

    // Display or write
    match format {
        TraceFormat::Text => {
            display_trace_text(&ir_graph, node_id, &subgraph, depth)?;
        }
        TraceFormat::Dot => {
            let dot = generate_trace_dot(&ir_graph, node_id, &subgraph)?;
            if let Some(path) = output {
                std::fs::write(path, &dot)?;
                println!("Wrote trace to {}", path.display());
            } else {
                println!("{}", dot);
            }
        }
    }

    Ok(())
}

struct Subgraph {
    upstream: Vec<Vec<IrNodeId>>, // upstream[depth] = nodes at that distance
    downstream: Vec<Vec<IrNodeId>>, // downstream[depth] = nodes at that distance
}

fn collect_subgraph(
    graph: &IrGraph,
    start: IrNodeId,
    max_depth: usize,
    direction: TraceDirection,
) -> Result<Subgraph> {
    let mut upstream = vec![Vec::new(); max_depth + 1];
    let mut downstream = vec![Vec::new(); max_depth + 1];

    match direction {
        TraceDirection::Both => {
            collect_upstream(graph, start, max_depth, &mut upstream)?;
            collect_downstream(graph, start, max_depth, &mut downstream)?;
        }
        TraceDirection::Upstream => {
            collect_upstream(graph, start, max_depth, &mut upstream)?;
        }
        TraceDirection::Downstream => {
            collect_downstream(graph, start, max_depth, &mut downstream)?;
        }
    }

    Ok(Subgraph {
        upstream,
        downstream,
    })
}

fn collect_upstream(
    graph: &IrGraph,
    start: IrNodeId,
    max_depth: usize,
    upstream: &mut [Vec<IrNodeId>],
) -> Result<()> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0));

    while let Some((node_id, depth)) = queue.pop_front() {
        if depth > max_depth || visited.contains(&node_id) {
            continue;
        }
        visited.insert(node_id);

        if depth > 0 {
            upstream[depth].push(node_id);
        }

        // Get input producers
        let node = graph.node(node_id)?;
        for &input_id in &node.inputs {
            if let Some(producer_id) = graph.tensor_producer(input_id) {
                queue.push_back((producer_id, depth + 1));
            }
        }
    }

    Ok(())
}

fn collect_downstream(
    graph: &IrGraph,
    start: IrNodeId,
    max_depth: usize,
    downstream: &mut [Vec<IrNodeId>],
) -> Result<()> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0));

    while let Some((node_id, depth)) = queue.pop_front() {
        if depth > max_depth || visited.contains(&node_id) {
            continue;
        }
        visited.insert(node_id);

        if depth > 0 {
            downstream[depth].push(node_id);
        }

        // Get output consumers
        let node = graph.node(node_id)?;
        for &tensor_id in &node.outputs {
            let consumers = graph.tensor_consumers(tensor_id);
            for consumer_id in consumers {
                queue.push_back((consumer_id, depth + 1));
            }
        }
    }

    Ok(())
}

fn display_trace_text(
    graph: &IrGraph,
    center: IrNodeId,
    subgraph: &Subgraph,
    max_depth: usize,
) -> Result<()> {
    let center_name = graph.node_name(center)?;
    let center_node = graph.node(center)?;
    let op_type = &center_node.op_type;

    println!("Tracing node: {} ({})", center_name, op_type);
    println!();

    // Display upstream
    if !subgraph.upstream.iter().all(|v| v.is_empty()) {
        println!("Upstream (depth {}):", max_depth);
        for depth in (1..=max_depth).rev() {
            if !subgraph.upstream[depth].is_empty() {
                println!("  Level {}:", depth);
                for &node_id in &subgraph.upstream[depth] {
                    display_node_summary(graph, node_id, "    ")?;
                }
            }
        }
        println!();
    }

    // Display center
    println!("Current:");
    display_node_detail(graph, center)?;
    println!();

    // Display downstream
    if !subgraph.downstream.iter().all(|v| v.is_empty()) {
        println!("Downstream (depth {}):", max_depth);
        for depth in 1..=max_depth {
            if !subgraph.downstream[depth].is_empty() {
                println!("  Level {}:", depth);
                for &node_id in &subgraph.downstream[depth] {
                    display_node_summary(graph, node_id, "    ")?;
                }
            }
        }
    }

    Ok(())
}

fn display_node_summary(graph: &IrGraph, node_id: IrNodeId, indent: &str) -> Result<()> {
    let name = graph.node_name(node_id)?;
    let node = graph.node(node_id)?;
    println!("{}{} ({})", indent, name, node.op_type);
    Ok(())
}

fn display_node_detail(graph: &IrGraph, node_id: IrNodeId) -> Result<()> {
    let name = graph.node_name(node_id)?;
    let node = graph.node(node_id)?;

    println!("  {} ({})", name, node.op_type);

    for (i, &input_id) in node.inputs.iter().enumerate() {
        let tensor = graph.tensor(input_id)?;
        println!("    Input {}: {} {:?}", i, tensor.name, tensor.shape);
    }

    for (i, &tensor_id) in node.outputs.iter().enumerate() {
        let tensor = graph.tensor(tensor_id)?;
        println!("    Output {}: {} {:?}", i, tensor.name, tensor.shape);
    }

    Ok(())
}

fn generate_trace_dot(graph: &IrGraph, center: IrNodeId, subgraph: &Subgraph) -> Result<String> {
    let mut dot = String::from("digraph trace {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    // Collect all nodes in the subgraph
    let mut all_nodes = HashSet::new();
    all_nodes.insert(center);

    for level in &subgraph.upstream {
        for &node_id in level {
            all_nodes.insert(node_id);
        }
    }

    for level in &subgraph.downstream {
        for &node_id in level {
            all_nodes.insert(node_id);
        }
    }

    // Track which tensors are used
    let mut used_tensors = HashSet::new();

    // Add nodes
    for &node_id in &all_nodes {
        let node = graph.node(node_id)?;
        let name = graph.node_name(node_id)?;
        let node_dot_id = format!("n{}", node_id.index());

        // Highlight center node
        let style = if node_id == center {
            ", style=\"filled\", fillcolor=\"lightblue\""
        } else {
            ""
        };

        dot.push_str(&format!(
            "  {} [label=\"{}\\n({})\"{}];\n",
            node_dot_id,
            escape_dot_string(name),
            escape_dot_string(&node.op_type),
            style
        ));

        // Track tensors
        for &inp in &node.inputs {
            used_tensors.insert(inp);
        }
        for &out in &node.outputs {
            used_tensors.insert(out);
        }
    }

    dot.push('\n');

    // Add tensor nodes
    for &tensor_id in &used_tensors {
        let tensor = graph.tensor(tensor_id)?;
        let tensor_dot_id = format!("t{}", tensor_id.index());

        let shape_str = format!("{:?}", tensor.shape);

        dot.push_str(&format!(
            "  {} [label=\"{}\\n{}\", shape=ellipse, style=filled, fillcolor=lightyellow];\n",
            tensor_dot_id,
            escape_dot_string(&tensor.name),
            escape_dot_string(&shape_str)
        ));
    }

    dot.push('\n');

    // Add edges
    for &node_id in &all_nodes {
        let node = graph.node(node_id)?;
        let node_dot_id = format!("n{}", node_id.index());

        for &inp in &node.inputs {
            let tensor_dot_id = format!("t{}", inp.index());
            dot.push_str(&format!("  {} -> {};\n", tensor_dot_id, node_dot_id));
        }

        for &out in &node.outputs {
            let tensor_dot_id = format!("t{}", out.index());
            dot.push_str(&format!("  {} -> {};\n", node_dot_id, tensor_dot_id));
        }
    }

    dot.push_str("}\n");
    Ok(dot)
}

/// Escape special characters for DOT format.
fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\"', "\\\"")
        .replace('\n', "\\n")
}
