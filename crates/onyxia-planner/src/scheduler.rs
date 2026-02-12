//! Scheduler for ordering operations in the execution graph.
//!
//! This module implements topological sorting and scheduling algorithms to
//! determine the optimal execution order for operations in the graph.

use crate::error::{CodegenError, Result};
use onyxia_onnx::{Graph, NodeId};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use std::collections::HashMap;

/// Schedule operations in a graph for execution.
pub struct Scheduler {
    /// The graph to schedule.
    graph: Graph,
}

impl Scheduler {
    /// Create a new scheduler for a graph.
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// Schedule the graph operations in topological order.
    pub fn schedule(&self) -> Result<Vec<NodeId>> {
        // Build a dependency graph using petgraph
        let dep_graph = self.build_dependency_graph()?;

        // Check for cycles
        if petgraph::algo::is_cyclic_directed(&dep_graph) {
            return Err(CodegenError::SchedulingError(
                "Graph contains cycles".to_string(),
            ));
        }

        // Perform topological sort
        let sorted = self.topological_sort(&dep_graph)?;

        Ok(sorted)
    }

    /// Build a dependency graph where nodes are operations and edges represent dependencies.
    fn build_dependency_graph(&self) -> Result<DiGraph<NodeId, ()>> {
        let mut dep_graph = DiGraph::new();
        let mut node_indices: HashMap<NodeId, NodeIndex> = HashMap::new();

        // Add all nodes to the graph
        for (node_id, _) in self.graph.nodes.iter().enumerate() {
            let idx = dep_graph.add_node(node_id);
            node_indices.insert(node_id, idx);
        }

        // Build a map from tensor name to producing node
        let mut tensor_producers: HashMap<String, NodeId> = HashMap::new();
        for (node_id, node) in self.graph.nodes.iter().enumerate() {
            for output in &node.outputs {
                tensor_producers.insert(output.clone(), node_id);
            }
        }

        // Add edges based on data dependencies
        for (consumer_id, node) in self.graph.nodes.iter().enumerate() {
            for input in &node.inputs {
                // Find the producer of this input
                if let Some(&producer_id) = tensor_producers.get(input) {
                    // Add edge: producer -> consumer
                    let producer_idx = node_indices[&producer_id];
                    let consumer_idx = node_indices[&consumer_id];
                    dep_graph.add_edge(producer_idx, consumer_idx, ());
                }
            }
        }

        Ok(dep_graph)
    }

    /// Perform topological sort on the dependency graph.
    fn topological_sort(&self, dep_graph: &DiGraph<NodeId, ()>) -> Result<Vec<NodeId>> {
        let mut topo = Topo::new(&dep_graph);
        let mut sorted = Vec::new();

        while let Some(node_idx) = topo.next(&dep_graph) {
            let node_id = dep_graph[node_idx];
            sorted.push(node_id);
        }

        // Verify we got all nodes
        if sorted.len() != self.graph.nodes.len() {
            return Err(CodegenError::SchedulingError(
                "Topological sort did not visit all nodes".to_string(),
            ));
        }

        Ok(sorted)
    }

    /// Get the graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

/// Compute pass grouping (for future optimization).
///
/// Groups independent operations that can be executed together.
#[derive(Debug, Clone)]
pub struct ComputePass {
    /// Node IDs in this pass.
    pub nodes: Vec<NodeId>,
}

impl ComputePass {
    /// Create a new compute pass.
    pub fn new(nodes: Vec<NodeId>) -> Self {
        Self { nodes }
    }
}

/// Group nodes into compute passes.
pub fn group_into_passes(_graph: &Graph, ordered_nodes: &[NodeId]) -> Result<Vec<ComputePass>> {
    // For now, create one pass per node (no batching)
    // TODO: Implement intelligent batching based on dependencies
    let passes = ordered_nodes
        .iter()
        .map(|&node_id| ComputePass::new(vec![node_id]))
        .collect();

    Ok(passes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

    #[test]
    fn test_simple_schedule() {
        // Build a simple graph: A -> B -> C
        let mut graph = Graph::new();

        // Add tensors
        let t0 = TensorInfo {
            name: "t0".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t0);

        let t1 = TensorInfo {
            name: "t1".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t1);

        let t2 = TensorInfo {
            name: "t2".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t2);

        // Add nodes: A produces t0, B consumes t0 and produces t1, C consumes t1 and produces t2
        let mut node_a = Node::new("Add");
        node_a.outputs = vec!["t0".to_string()];
        graph.add_node(node_a);

        let mut node_b = Node::new("Mul");
        node_b.inputs = vec!["t0".to_string()];
        node_b.outputs = vec!["t1".to_string()];
        graph.add_node(node_b);

        let mut node_c = Node::new("Gelu");
        node_c.inputs = vec!["t1".to_string()];
        node_c.outputs = vec!["t2".to_string()];
        graph.add_node(node_c);

        // Schedule
        let scheduler = Scheduler::new(graph);
        let order = scheduler.schedule().unwrap();

        // Verify order: A (0) must come before B (1), B must come before C (2)
        assert_eq!(order.len(), 3);

        let pos_a = order.iter().position(|&id| id == 0).unwrap();
        let pos_b = order.iter().position(|&id| id == 1).unwrap();
        let pos_c = order.iter().position(|&id| id == 2).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_parallel_schedule() {
        // Build a graph: A -> B and A -> C (B and C are independent)
        let mut graph = Graph::new();

        // Add tensors
        let t0 = TensorInfo {
            name: "t0".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t0);

        let t1 = TensorInfo {
            name: "t1".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t1);

        let t2 = TensorInfo {
            name: "t2".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Intermediate,
            initializer: None,
        };
        graph.add_tensor(t2);

        // Node A produces t0
        let mut node_a = Node::new("Add");
        node_a.outputs = vec!["t0".to_string()];
        graph.add_node(node_a);

        // Node B consumes t0, produces t1
        let mut node_b = Node::new("Mul");
        node_b.inputs = vec!["t0".to_string()];
        node_b.outputs = vec!["t1".to_string()];
        graph.add_node(node_b);

        // Node C consumes t0, produces t2 (independent of B)
        let mut node_c = Node::new("Gelu");
        node_c.inputs = vec!["t0".to_string()];
        node_c.outputs = vec!["t2".to_string()];
        graph.add_node(node_c);

        // Schedule
        let scheduler = Scheduler::new(graph);
        let order = scheduler.schedule().unwrap();

        // Verify: A must come first, B and C can be in any order after A
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], 0); // A must be first

        // B and C can be in any order
        assert!(order.contains(&1));
        assert!(order.contains(&2));
    }
}
