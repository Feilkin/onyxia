//! ONNX model parser for Onyxia.
//!
//! This crate parses ONNX protobuf models and provides a structured graph
//! representation independent of the underlying protobuf schema.
//!
//! # Example
//!
//! ```no_run
//! use onyxia_onnx::{load_model, parse_model};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load raw protobuf
//! let model = load_model("model.onnx")?;
//!
//! // Parse into structured graph
//! let graph = parse_model(&model)?;
//!
//! println!("Model: {}", graph.metadata.name);
//! println!("Nodes: {}", graph.nodes.len());
//! println!("Tensors: {}", graph.tensor_info.len());
//! # Ok(())
//! # }
//! ```

use prost::Message;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Generated protobuf types from ONNX schema.
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod graph;
pub mod parser;

pub use graph::{
    AttributeValue, DataType, Dimension, Graph, GraphMetadata, Node, NodeId, TensorId, TensorInfo,
    TensorKind, TensorShape,
};
pub use onnx::ModelProto;
pub use parser::parse_model;

/// Simplification level for DOT graph generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotSimplification {
    /// Show all nodes and edges (slowest, most detailed)
    Full,
    /// Group operations by layer, show layer-level flow
    Layers,
    /// Only show high-level summary (fastest)
    Summary,
}

/// Errors that can occur when loading or processing ONNX models.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("Failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse ONNX protobuf: {0}")]
    DecodeError(#[from] prost::DecodeError),

    #[error("Invalid ONNX model: {0}")]
    InvalidModel(String),

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Missing attribute: {0}")]
    MissingAttribute(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),
}

/// Result type for ONNX operations.
pub type Result<T> = std::result::Result<T, OnnxError>;

/// Load an ONNX model from a file.
///
/// # Arguments
///
/// * `path` - Path to the ONNX model file
///
/// # Returns
///
/// Returns the parsed `ModelProto` or an error if loading/parsing fails.
///
/// # Example
///
/// ```no_run
/// use onyxia_onnx::load_model;
///
/// let model = load_model("path/to/model.onnx")?;
/// # Ok::<(), onyxia_onnx::OnnxError>(())
/// ```
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ModelProto> {
    let bytes = fs::read(path)?;
    let model = ModelProto::decode(&bytes[..])?;
    Ok(model)
}

/// Convert an ONNX model to Graphviz DOT format.
///
/// Generates a directed graph representing the computation graph structure
/// of the ONNX model, showing operators as nodes and tensors as edges.
///
/// # Arguments
///
/// * `model` - The ONNX model to convert
///
/// # Returns
///
/// Returns a String containing the DOT format representation.
pub fn to_dot(model: &ModelProto) -> String {
    to_dot_with_options(model, DotSimplification::Full)
}

/// Convert an ONNX model to Graphviz DOT format with simplification options.
///
/// # Arguments
///
/// * `model` - The ONNX model to convert
/// * `simplification` - Level of simplification to apply
///
/// # Returns
///
/// Returns a String containing the DOT format representation.
pub fn to_dot_with_options(model: &ModelProto, simplification: DotSimplification) -> String {
    match simplification {
        DotSimplification::Full => to_dot_full(model),
        DotSimplification::Layers => to_dot_layers(model),
        DotSimplification::Summary => to_dot_summary(model),
    }
}

/// Generate full detailed DOT graph.
fn to_dot_full(model: &ModelProto) -> String {
    let mut dot = String::from("digraph onnx_model {\n");
    dot.push_str("    rankdir=TB;\n");
    dot.push_str("    node [shape=box, style=rounded];\n");
    // Performance optimizations for large graphs
    dot.push_str("    concentrate=true;\n");
    dot.push_str("    nodesep=0.3;\n");
    dot.push_str("    ranksep=0.5;\n\n");

    // Get the graph if it exists
    if let Some(graph) = &model.graph {
        // Add graph name as a label
        if !graph.name.is_empty() {
            dot.push_str(&format!(
                "    label=\"{}\";\n",
                escape_dot_string(&graph.name)
            ));
        }
        dot.push_str("    labelloc=\"t\";\n\n");

        // Add input nodes
        for input in &graph.input {
            let input_name = escape_dot_string(&input.name);
            dot.push_str(&format!(
                "    \"{}\" [shape=ellipse, style=filled, fillcolor=lightblue, label=\"{}\"];\n",
                input_name, input_name
            ));
        }

        // Add output nodes
        for output in &graph.output {
            let output_name = escape_dot_string(&output.name);
            dot.push_str(&format!(
                "    \"{}\" [shape=ellipse, style=filled, fillcolor=lightgreen, label=\"{}\"];\n",
                output_name, output_name
            ));
        }

        dot.push('\n');

        // Add operator nodes and edges
        for (idx, node) in graph.node.iter().enumerate() {
            let node_id = if !node.name.is_empty() {
                escape_dot_string(&node.name)
            } else {
                format!("node_{}", idx)
            };

            let op_type = escape_dot_string(&node.op_type);
            let label = if !node.name.is_empty() {
                format!("{}\\n[{}]", escape_dot_string(&node.name), op_type)
            } else {
                op_type.clone()
            };

            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\", style=filled, fillcolor=lightyellow];\n",
                node_id, label
            ));

            // Add edges from inputs to this node
            for input in &node.input {
                if !input.is_empty() {
                    let input_escaped = escape_dot_string(input);
                    dot.push_str(&format!("    \"{}\" -> \"{}\";\n", input_escaped, node_id));
                }
            }

            // Add edges from this node to outputs
            for output in &node.output {
                if !output.is_empty() {
                    let output_escaped = escape_dot_string(output);
                    dot.push_str(&format!("    \"{}\" -> \"{}\";\n", node_id, output_escaped));
                }
            }
        }
    }

    dot.push_str("}\n");
    dot
}

/// Generate layer-grouped DOT graph (much faster for large models).
fn to_dot_layers(model: &ModelProto) -> String {
    use std::collections::{HashMap, HashSet};

    let mut dot = String::from("digraph onnx_model {\n");
    // Use left-to-right layout so layers flow vertically (not horizontally)
    dot.push_str("    rankdir=LR;\n");
    dot.push_str("    node [shape=box, style=rounded];\n");
    dot.push_str("    compound=true;\n");
    dot.push_str("    concentrate=true;\n");
    dot.push_str("    nodesep=0.5;\n");
    dot.push_str("    ranksep=1.0;\n\n");

    if let Some(graph) = &model.graph {
        if !graph.name.is_empty() {
            dot.push_str(&format!(
                "    label=\"{}\";\n",
                escape_dot_string(&graph.name)
            ));
        }
        dot.push_str("    labelloc=\"t\";\n\n");

        // Group nodes by layer (detect patterns like "layers.N")
        let mut layer_nodes: HashMap<String, Vec<(usize, &onnx::NodeProto)>> = HashMap::new();
        let mut ungrouped: Vec<(usize, &onnx::NodeProto)> = Vec::new();

        for (idx, node) in graph.node.iter().enumerate() {
            let name = if !node.name.is_empty() {
                &node.name
            } else {
                &node.op_type
            };

            // Try to extract layer number
            if let Some(layer_id) = extract_layer_id(name) {
                layer_nodes.entry(layer_id).or_default().push((idx, node));
            } else {
                ungrouped.push((idx, node));
            }
        }

        // Identify shared inputs that connect to multiple layers (to avoid clutter)
        let all_outputs: HashSet<String> = graph
            .node
            .iter()
            .flat_map(|n| n.output.iter().cloned())
            .collect();

        let mut input_layer_count: HashMap<String, HashSet<String>> = HashMap::new();
        for (layer_id, nodes) in &layer_nodes {
            for (_, node) in nodes {
                for input in &node.input {
                    if !input.is_empty() && !all_outputs.contains(input) {
                        input_layer_count
                            .entry(input.clone())
                            .or_default()
                            .insert(layer_id.clone());
                    }
                }
            }
        }

        // Inputs used by 4+ layers are considered "shared" and will be duplicated per layer
        let shared_inputs: HashSet<String> = input_layer_count
            .iter()
            .filter(|(_, layers)| layers.len() >= 4)
            .map(|(input, _)| input.clone())
            .collect();

        // Add ungrouped nodes first (embeddings, norms, etc.), but skip shared inputs
        for (idx, node) in &ungrouped {
            let node_id = if !node.name.is_empty() {
                escape_dot_string(&node.name)
            } else {
                format!("node_{}", idx)
            };
            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\\n[{}]\", style=filled, fillcolor=lightyellow];\n",
                node_id,
                if !node.name.is_empty() {
                    &node.name
                } else {
                    "?"
                },
                node.op_type
            ));
        }

        // Add shared inputs (that aren't operator outputs) as separate nodes to reference later
        let shared_non_ops: Vec<&String> = shared_inputs
            .iter()
            .filter(|input| !all_outputs.contains(*input))
            .collect();

        // Add layer subgraphs with nodes inside
        let mut layer_keys: Vec<_> = layer_nodes.keys().collect();
        layer_keys.sort();

        for layer_key in layer_keys {
            let nodes = &layer_nodes[layer_key];
            dot.push_str(&format!(
                "\n    subgraph cluster_{} {{\n",
                layer_key.replace('.', "_")
            ));
            dot.push_str(&format!("        label=\"Layer {}\";\n", layer_key));
            dot.push_str("        style=filled;\n");
            dot.push_str("        fillcolor=lightgrey;\n\n");

            // Add layer-local duplicates of shared inputs
            for shared_input in &shared_non_ops {
                let local_id = format!(
                    "{}_{}",
                    layer_key.replace('.', "_"),
                    escape_dot_string(shared_input)
                );
                dot.push_str(&format!(
                    "        \"{}\" [label=\"{}\", shape=ellipse, style=filled, fillcolor=lightblue];\n",
                    local_id,
                    escape_dot_string(shared_input)
                ));
            }

            // Add all nodes in this layer to the subgraph
            for (idx, node) in nodes {
                let node_id = if !node.name.is_empty() {
                    escape_dot_string(&node.name)
                } else {
                    format!("node_{}", idx)
                };
                dot.push_str(&format!(
                    "        \"{}\" [label=\"{}\\n[{}]\", style=filled, fillcolor=lightyellow];\n",
                    node_id,
                    if !node.name.is_empty() {
                        &node.name
                    } else {
                        "?"
                    },
                    node.op_type
                ));
            }

            dot.push_str("    }\n");
        }

        // Add edges (use layer-local duplicates for shared inputs)
        for (idx, node) in graph.node.iter().enumerate() {
            let node_id = if !node.name.is_empty() {
                escape_dot_string(&node.name)
            } else {
                format!("node_{}", idx)
            };

            let node_layer = extract_layer_id(if !node.name.is_empty() {
                &node.name
            } else {
                &node.op_type
            });

            for input in &node.input {
                if !input.is_empty() && !all_outputs.contains(input) {
                    // Check if this is a shared input
                    if shared_inputs.contains(input) && node_layer.is_some() {
                        // Use layer-local duplicate
                        let layer_key = node_layer.as_ref().unwrap();
                        let local_id = format!(
                            "{}_{}",
                            layer_key.replace('.', "_"),
                            escape_dot_string(input)
                        );
                        dot.push_str(&format!("    \"{}\" -> \"{}\";\n", local_id, node_id));
                    } else {
                        // Regular edge
                        let input_escaped = escape_dot_string(input);
                        dot.push_str(&format!("    \"{}\" -> \"{}\";\n", input_escaped, node_id));
                    }
                }
            }
        }
    }

    dot.push_str("}\n");
    dot
}

/// Generate summary DOT graph (fastest, only high-level structure).
fn to_dot_summary(model: &ModelProto) -> String {
    use std::collections::HashMap;

    let mut dot = String::from("digraph onnx_model {\n");
    dot.push_str("    rankdir=TB;\n");
    dot.push_str("    node [shape=box, style=\"rounded,filled\"];\n\n");

    if let Some(graph) = &model.graph {
        if !graph.name.is_empty() {
            dot.push_str(&format!(
                "    label=\"{}\";\n",
                escape_dot_string(&graph.name)
            ));
        }
        dot.push_str("    labelloc=\"t\";\n\n");

        // Count nodes by layer
        let mut layer_counts: HashMap<String, usize> = HashMap::new();
        let mut op_type_counts: HashMap<String, usize> = HashMap::new();

        for node in &graph.node {
            let name = if !node.name.is_empty() {
                &node.name
            } else {
                &node.op_type
            };
            if let Some(layer_id) = extract_layer_id(name) {
                *layer_counts.entry(layer_id).or_insert(0) += 1;
            }
            *op_type_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        dot.push_str("    \"inputs\" [label=\"Inputs\", fillcolor=lightblue];\n");

        if !layer_counts.is_empty() {
            let mut layers: Vec<_> = layer_counts.iter().collect();
            layers.sort_by_key(|(k, _)| k.as_str());

            dot.push_str(&format!(
                "    \"layers\" [label=\"{} Transformer Layers\\n({} ops each)\", fillcolor=lightyellow];\n",
                layers.len(),
                layers[0].1
            ));
            dot.push_str("    \"inputs\" -> \"layers\";\n");
        }

        dot.push_str("    \"outputs\" [label=\"Outputs\", fillcolor=lightgreen];\n");
        if !layer_counts.is_empty() {
            dot.push_str("    \"layers\" -> \"outputs\";\n");
        } else {
            dot.push_str("    \"inputs\" -> \"outputs\";\n");
        }

        // Add op type summary as a note
        dot.push_str("\n    \"summary\" [shape=note, fillcolor=lightyellow, label=\"Op Types:\\l");
        let mut ops: Vec<_> = op_type_counts.iter().collect();
        ops.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (op_type, count) in ops.iter().take(10) {
            dot.push_str(&format!("  {} x{}\\l", escape_dot_string(op_type), count));
        }
        if ops.len() > 10 {
            dot.push_str(&format!("  ... and {} more\\l", ops.len() - 10));
        }
        dot.push_str("\"];\n");
    }

    dot.push_str("}\n");
    dot
}

/// Extract layer ID from node name (e.g., "layers.23" from "/model/layers.23/attn/...").
fn extract_layer_id(name: &str) -> Option<String> {
    if let Some(start) = name.find("layers.") {
        let rest = &name[start + 7..];
        if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
            return Some(format!("layers.{}", &rest[..end]));
        } else if rest.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("layers.{}", rest));
        }
    }
    None
}

/// Escape special characters in DOT strings.
fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_proto_exists() {
        // Verify generated types are accessible
        let _model = ModelProto::default();
    }

    #[test]
    fn test_escape_dot_string() {
        assert_eq!(escape_dot_string("hello"), "hello");
        assert_eq!(escape_dot_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_dot_string("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_to_dot_empty_model() {
        let model = ModelProto::default();
        let dot = to_dot(&model);
        assert!(dot.contains("digraph onnx_model"));
        assert!(dot.contains("rankdir=TB"));
    }

    #[test]
    fn test_extract_layer_id() {
        assert_eq!(
            extract_layer_id("/model/layers.23/attn/q_proj"),
            Some("layers.23".to_string())
        );
        assert_eq!(extract_layer_id("layers.5"), Some("layers.5".to_string()));
        assert_eq!(extract_layer_id("no_layer_here"), None);
        assert_eq!(extract_layer_id("/model/embed"), None);
    }
}
