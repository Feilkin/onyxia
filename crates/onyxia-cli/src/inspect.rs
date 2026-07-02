//! Node/tensor inspection utilities.
//!
//! These operate directly on the parsed ONNX graph — no lowering, no GPU.
//! They exist to answer "what is in this model file", which is a question
//! about the ONNX graph, not about how onyxia executes it (`onyxia ir-dot`
//! shows the lowered view).

use crate::{TraceDirection, TraceFormat};
use anyhow::{Context, Result};
use onyxia_onnx::{DataType, Graph, Node, NodeId, TensorId, TensorInfo, TensorShape};
use regex::Regex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

/// Producer/consumer maps derived from node inputs/outputs, by tensor name.
struct Topology<'g> {
    producer: HashMap<&'g str, NodeId>,
    consumers: HashMap<&'g str, Vec<NodeId>>,
}

impl<'g> Topology<'g> {
    fn of(graph: &'g Graph) -> Self {
        let mut producer = HashMap::new();
        let mut consumers: HashMap<&str, Vec<NodeId>> = HashMap::new();
        for (id, node) in graph.nodes.iter().enumerate() {
            for out in &node.outputs {
                producer.insert(out.as_str(), id);
            }
            for inp in &node.inputs {
                consumers.entry(inp.as_str()).or_default().push(id);
            }
        }
        Self {
            producer,
            consumers,
        }
    }
}

fn find_node<'g>(graph: &'g Graph, name: &str) -> Result<(NodeId, &'g Node)> {
    graph
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| n.name == name)
        .with_context(|| format!("Node '{name}' not found"))
}

fn tensor<'g>(graph: &'g Graph, name: &str) -> Option<&'g TensorInfo> {
    graph.tensors.get(name).map(|&id| &graph.tensor_info[id])
}

/// Display detailed information about one or more nodes.
pub fn inspect_nodes(graph: &Graph, node_names: &[String], full_values: bool) -> Result<()> {
    for node_name in node_names {
        let (_, node) = find_node(graph, node_name)?;
        display_node(graph, node, full_values);
        println!(); // Blank line between nodes
    }
    Ok(())
}

fn display_node(graph: &Graph, node: &Node, full_values: bool) {
    let op = if node.domain.is_empty() {
        node.op_type.clone()
    } else {
        format!("{}::{}", node.domain, node.op_type)
    };
    println!("Node: {} ({op})", node.name);

    if node.attributes.is_empty() {
        println!("  Attributes: (none)");
    } else {
        println!("  Attributes:");
        for (key, value) in &node.attributes {
            println!("    {}: {:?}", key, value);
        }
    }

    println!("  Inputs:");
    for (i, name) in node.inputs.iter().enumerate() {
        display_tensor_line(graph, name, i, full_values);
    }

    println!("  Outputs:");
    for (i, name) in node.outputs.iter().enumerate() {
        match tensor(graph, name) {
            Some(info) => {
                println!("    {}: {}", i, info.name);
                println!("       Shape: {:?}", info.shape);
                println!("       Type: {:?}", info.dtype);
            }
            None => println!("    {}: {} (no tensor info)", i, name),
        }
    }
}

fn display_tensor_line(graph: &Graph, name: &str, index: usize, full_values: bool) {
    let Some(info) = tensor(graph, name) else {
        println!("    {}: {} (no tensor info)", index, name);
        return;
    };
    println!("    {}: {}", index, info.name);
    println!("       Shape: {:?}", info.shape);
    println!("       Type: {:?}", info.dtype);
    match &info.initializer {
        Some(bytes) => {
            println!("       Source: Constant initializer");
            display_initializer(info.dtype, &info.shape, bytes, full_values);
        }
        None => println!("       Source: Runtime tensor"),
    }
}

/// Pretty-print initializer bytes (truncated unless `full`).
fn display_initializer(dtype: DataType, shape: &TensorShape, data: &[u8], full: bool) {
    const MAX_ELEMENTS: usize = 20;

    // Element count from shape when static, else from the byte length.
    let count = shape
        .as_static()
        .map(|dims| dims.iter().product::<usize>())
        .unwrap_or(data.len() / dtype.size().max(1));
    let take = if full { count } else { count.min(MAX_ELEMENTS) };

    let rendered = match dtype {
        DataType::I64 => Some(format!(
            "{:?}",
            data.chunks_exact(8)
                .take(take)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        )),
        DataType::I32 => Some(format!(
            "{:?}",
            data.chunks_exact(4)
                .take(take)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        )),
        DataType::F32 => Some(format!(
            "{:?}",
            data.chunks_exact(4)
                .take(take)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        )),
        DataType::Bool | DataType::U8 | DataType::Q4 | DataType::Q8 => Some(format!(
            "{:?}",
            data.iter().take(take).copied().collect::<Vec<_>>()
        )),
        _ => None,
    };
    match rendered {
        Some(values) if take < count => println!(
            "       Value: {values} ... ({count} elements total, use --full-values to see all)"
        ),
        Some(values) => println!("       Value: {values}"),
        None => println!("       Value: ({count} elements, dtype {dtype:?} not displayed)"),
    }
}

/// List nodes in the model with optional filtering.
pub fn list_nodes(
    graph: &Graph,
    op_types: &[String],
    name_pattern: Option<&str>,
    show_shapes: bool,
    summary: bool,
) -> Result<()> {
    if summary {
        display_summary(graph);
        return Ok(());
    }

    let pattern = name_pattern
        .map(Regex::new)
        .transpose()
        .context("Invalid regex pattern")?;

    let matching: Vec<&Node> = graph
        .nodes
        .iter()
        .filter(|node| op_types.is_empty() || op_types.contains(&node.op_type))
        .filter(|node| {
            pattern
                .as_ref()
                .is_none_or(|regex| regex.is_match(&node.name))
        })
        .collect();

    if op_types.is_empty() && name_pattern.is_none() {
        println!("Found {} nodes:", matching.len());
    } else {
        println!("Found {} matching nodes:", matching.len());
    }

    for node in matching {
        if show_shapes {
            println!();
            println!("{}", node.name);
            for (i, name) in node.inputs.iter().enumerate() {
                let (shape, constant) = match tensor(graph, name) {
                    Some(info) => (
                        format!("{:?}", info.shape),
                        if info.initializer.is_some() {
                            " (initializer)"
                        } else {
                            ""
                        },
                    ),
                    None => ("?".to_string(), ""),
                };
                println!("  Input {}: {}{}", i, shape, constant);
            }
            for (i, name) in node.outputs.iter().enumerate() {
                let shape = tensor(graph, name)
                    .map(|info| format!("{:?}", info.shape))
                    .unwrap_or_else(|| "?".to_string());
                println!("  Output {}: {}", i, shape);
            }
        } else {
            println!("  {}", node.name);
        }
    }

    Ok(())
}

fn display_summary(graph: &Graph) {
    use std::collections::BTreeMap;

    let mut op_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for node in &graph.nodes {
        *op_counts.entry(&node.op_type).or_insert(0) += 1;
    }

    println!("Model Summary:");
    println!("  Total nodes: {}", graph.nodes.len());
    println!();

    let mut sorted: Vec<_> = op_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("  Top operators:");
    for (op_type, count) in sorted.iter().take(15) {
        println!("    {:<15} {}", format!("{}:", op_type), count);
    }

    if sorted.len() > 15 {
        let remaining: usize = sorted.iter().skip(15).map(|(_, c)| **c).sum();
        println!();
        println!(
            "  Other operators: {} types ({} total nodes)",
            sorted.len() - 15,
            remaining
        );
    }
}

/// Inspect tensor(s) by name.
pub fn inspect_tensor(
    graph: &Graph,
    names: &[String],
    list_constants: bool,
    full_values: bool,
) -> Result<()> {
    if list_constants {
        let mut constants: Vec<&TensorInfo> = graph
            .tensor_info
            .iter()
            .filter(|info| info.initializer.is_some())
            .collect();
        constants.sort_by(|a, b| a.name.cmp(&b.name));
        println!("Found {} constant tensors:", constants.len());
        println!();
        for info in constants {
            println!("  {} : {:?} {:?}", info.name, info.dtype, info.shape);
        }
        return Ok(());
    }

    let topo = Topology::of(graph);
    for name in names {
        display_tensor_detail(graph, &topo, name, full_values)?;
        println!();
    }
    Ok(())
}

fn display_tensor_detail(
    graph: &Graph,
    topo: &Topology,
    name: &str,
    full_values: bool,
) -> Result<()> {
    let info = tensor(graph, name).with_context(|| format!("Tensor '{name}' not found"))?;

    println!("Tensor: {}", info.name);
    println!("  Type: {:?}", info.dtype);
    println!("  Shape: {:?}", info.shape);

    match &info.initializer {
        Some(bytes) => {
            println!("  Kind: Constant initializer");
            let total_mb = bytes.len() as f64 / (1024.0 * 1024.0);
            if total_mb > 1.0 {
                println!("  Size: {:.2} MB", total_mb);
            }
            println!();
            display_initializer(info.dtype, &info.shape, bytes, full_values);
        }
        None => println!("  Kind: Runtime tensor"),
    }

    if let Some(consumers) = topo.consumers.get(name) {
        println!();
        println!("  Consumers:");
        for &consumer_id in consumers {
            let node = &graph.nodes[consumer_id];
            for (i, input) in node.inputs.iter().enumerate() {
                if input == name {
                    println!("    {} (input {})", node.name, i);
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Trace data flow around a specific node.
pub fn trace_node(
    graph: &Graph,
    node_name: &str,
    depth: usize,
    direction: TraceDirection,
    format: TraceFormat,
    output: Option<&Path>,
) -> Result<()> {
    let (node_id, _) = find_node(graph, node_name)?;
    let topo = Topology::of(graph);
    let subgraph = collect_subgraph(graph, &topo, node_id, depth, direction);

    match format {
        TraceFormat::Text => display_trace_text(graph, node_id, &subgraph, depth),
        TraceFormat::Dot => {
            let dot = generate_trace_dot(graph, node_id, &subgraph)?;
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
    upstream: Vec<Vec<NodeId>>,   // upstream[depth] = nodes at that distance
    downstream: Vec<Vec<NodeId>>, // downstream[depth] = nodes at that distance
}

fn collect_subgraph(
    graph: &Graph,
    topo: &Topology,
    start: NodeId,
    max_depth: usize,
    direction: TraceDirection,
) -> Subgraph {
    let mut upstream = vec![Vec::new(); max_depth + 1];
    let mut downstream = vec![Vec::new(); max_depth + 1];

    if matches!(direction, TraceDirection::Both | TraceDirection::Upstream) {
        walk(graph, start, max_depth, &mut upstream, |node| {
            node.inputs
                .iter()
                .filter_map(|inp| topo.producer.get(inp.as_str()).copied())
                .collect()
        });
    }
    if matches!(direction, TraceDirection::Both | TraceDirection::Downstream) {
        walk(graph, start, max_depth, &mut downstream, |node| {
            node.outputs
                .iter()
                .flat_map(|out| {
                    topo.consumers
                        .get(out.as_str())
                        .into_iter()
                        .flatten()
                        .copied()
                })
                .collect()
        });
    }

    Subgraph {
        upstream,
        downstream,
    }
}

fn walk(
    graph: &Graph,
    start: NodeId,
    max_depth: usize,
    levels: &mut [Vec<NodeId>],
    neighbors: impl Fn(&Node) -> Vec<NodeId>,
) {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start, 0));

    while let Some((node_id, depth)) = queue.pop_front() {
        if depth > max_depth || !visited.insert(node_id) {
            continue;
        }
        if depth > 0 {
            levels[depth].push(node_id);
        }
        for next in neighbors(&graph.nodes[node_id]) {
            queue.push_back((next, depth + 1));
        }
    }
}

fn display_trace_text(graph: &Graph, center: NodeId, subgraph: &Subgraph, max_depth: usize) {
    let center_node = &graph.nodes[center];
    println!(
        "Tracing node: {} ({})",
        center_node.name, center_node.op_type
    );
    println!();

    if !subgraph.upstream.iter().all(|v| v.is_empty()) {
        println!("Upstream (depth {}):", max_depth);
        for depth in (1..=max_depth).rev() {
            if !subgraph.upstream[depth].is_empty() {
                println!("  Level {}:", depth);
                for &node_id in &subgraph.upstream[depth] {
                    let node = &graph.nodes[node_id];
                    println!("    {} ({})", node.name, node.op_type);
                }
            }
        }
        println!();
    }

    println!("Current:");
    println!("  {} ({})", center_node.name, center_node.op_type);
    for (i, name) in center_node.inputs.iter().enumerate() {
        let shape = tensor(graph, name)
            .map(|info| format!("{:?}", info.shape))
            .unwrap_or_else(|| "?".to_string());
        println!("    Input {}: {} {}", i, name, shape);
    }
    for (i, name) in center_node.outputs.iter().enumerate() {
        let shape = tensor(graph, name)
            .map(|info| format!("{:?}", info.shape))
            .unwrap_or_else(|| "?".to_string());
        println!("    Output {}: {} {}", i, name, shape);
    }
    println!();

    if !subgraph.downstream.iter().all(|v| v.is_empty()) {
        println!("Downstream (depth {}):", max_depth);
        for depth in 1..=max_depth {
            if !subgraph.downstream[depth].is_empty() {
                println!("  Level {}:", depth);
                for &node_id in &subgraph.downstream[depth] {
                    let node = &graph.nodes[node_id];
                    println!("    {} ({})", node.name, node.op_type);
                }
            }
        }
    }
}

fn generate_trace_dot(graph: &Graph, center: NodeId, subgraph: &Subgraph) -> Result<String> {
    let mut dot = String::from("digraph trace {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    let mut all_nodes: HashSet<NodeId> = HashSet::new();
    all_nodes.insert(center);
    for level in subgraph.upstream.iter().chain(&subgraph.downstream) {
        all_nodes.extend(level.iter().copied());
    }

    let mut used_tensors: HashSet<&str> = HashSet::new();

    for &node_id in &all_nodes {
        let node = &graph.nodes[node_id];
        let style = if node_id == center {
            ", style=\"filled\", fillcolor=\"lightblue\""
        } else {
            ""
        };
        dot.push_str(&format!(
            "  n{} [label=\"{}\\n({})\"{}];\n",
            node_id,
            escape_dot_string(&node.name),
            escape_dot_string(&node.op_type),
            style
        ));
        used_tensors.extend(node.inputs.iter().map(String::as_str));
        used_tensors.extend(node.outputs.iter().map(String::as_str));
    }

    dot.push('\n');

    let tensor_dot_id = |name: &str| -> String {
        let id: TensorId = graph.tensors.get(name).copied().unwrap_or(usize::MAX);
        if id == usize::MAX {
            format!("t_{}", name.replace(['/', '.', '[', ']', ':'], "_"))
        } else {
            format!("t{}", id)
        }
    };

    for &name in &used_tensors {
        let shape = tensor(graph, name)
            .map(|info| format!("{:?}", info.shape))
            .unwrap_or_else(|| "?".to_string());
        dot.push_str(&format!(
            "  {} [label=\"{}\\n{}\", shape=ellipse, style=filled, fillcolor=lightyellow];\n",
            tensor_dot_id(name),
            escape_dot_string(name),
            escape_dot_string(&shape)
        ));
    }

    dot.push('\n');

    for &node_id in &all_nodes {
        let node = &graph.nodes[node_id];
        for inp in &node.inputs {
            dot.push_str(&format!("  {} -> n{};\n", tensor_dot_id(inp), node_id));
        }
        for out in &node.outputs {
            dot.push_str(&format!("  n{} -> {};\n", node_id, tensor_dot_id(out)));
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
