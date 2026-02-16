//! Initialize constants pass.
//!
//! Converts edges with `EdgeData::Initializer` (raw ONNX bytes) into
//! `EdgeData::Constant` (parsed `TensorValue`) when the edge has a known
//! static shape.
//!
//! Runs in the `Resolution` stage so that parsed constants are available
//! for both shape inference and constant folding.

use onyxia_core::ir::EdgeData;
use onyxia_core::{IrGraph, OperatorRegistry, Pass, Result, Stage, TensorShape, TensorValue};

/// Pass that parses initializer bytes into typed constant values.
///
/// For each edge with `EdgeData::Initializer`:
/// 1. Checks if the shape is fully static
/// 2. Parses the raw bytes via `TensorValue::from_bytes`
/// 3. Replaces `EdgeData::Initializer` with `EdgeData::Constant`
///
/// Edges with non-static shapes are left as initializers (they can't be
/// parsed without knowing the dimensions).
pub struct InitializeConstantsPass;

impl InitializeConstantsPass {
    /// Create a new initialize constants pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for InitializeConstantsPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for InitializeConstantsPass {
    fn name(&self) -> &str {
        "initialize_constants"
    }

    fn stage(&self) -> Stage {
        Stage::Resolution
    }

    fn run(&self, graph: &mut IrGraph, _registry: &OperatorRegistry) -> Result<bool> {
        let mut changed = false;

        for idx in 0..graph.tensor_count() {
            let edge_id = onyxia_core::IrEdgeId::new(idx);
            let edge = graph.edge(edge_id)?;

            // Only process initializers with static shapes
            let (bytes, dtype, shape) = match (&edge.data, &edge.shape) {
                (EdgeData::Initializer(bytes), TensorShape::Static(dims)) => {
                    (bytes.clone(), edge.dtype, dims.clone())
                }
                _ => continue,
            };

            let value = TensorValue::from_bytes(&bytes, dtype, &shape)?;
            graph.edge_mut(edge_id)?.data = EdgeData::Constant(value);
            changed = true;
        }

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::ir::IrEdge;
    use onyxia_core::{DataType, IrGraph, TensorShape};

    #[test]
    fn test_parses_f32_initializer() {
        let mut graph = IrGraph::new();

        // 1.0f32, 2.0f32 as little-endian bytes
        let edge = IrEdge::with_initializer(
            "w".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            vec![0, 0, 128, 63, 0, 0, 0, 64],
        );
        let id = graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);
        let edge = graph.edge(id).unwrap();
        assert!(edge.is_constant());
        let val = edge.constant_value().unwrap();
        assert_eq!(val.as_f32(), Some(&[1.0f32, 2.0][..]));
    }

    #[test]
    fn test_parses_i64_initializer() {
        let mut graph = IrGraph::new();

        // 42i64 as little-endian bytes
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&42i64.to_le_bytes());
        let edge = IrEdge::with_initializer(
            "idx".to_string(),
            DataType::I64,
            TensorShape::Static(vec![1]),
            bytes,
        );
        let id = graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(changed);
        let val = graph.edge(id).unwrap().constant_value().unwrap();
        assert_eq!(val.as_i64(), Some(&[42i64][..]));
    }

    #[test]
    fn test_skips_runtime_edges() {
        let mut graph = IrGraph::new();

        let edge = IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        );
        graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(!changed);
    }

    #[test]
    fn test_skips_non_static_initializer() {
        use onyxia_core::SymbolicDim;

        let mut graph = IrGraph::new();

        let edge = IrEdge::with_initializer(
            "w".to_string(),
            DataType::F32,
            TensorShape::Symbolic(vec![SymbolicDim::Fixed(2), SymbolicDim::Fixed(3)]),
            vec![0u8; 24], // 6 f32s
        );
        graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(!changed);
    }

    #[test]
    fn test_already_constant_unchanged() {
        let mut graph = IrGraph::new();

        let val = TensorValue::new(
            onyxia_core::TensorData::F32(vec![1.0]),
            vec![1],
            DataType::F32,
        );
        let edge = IrEdge::with_constant(
            "c".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1]),
            val,
        );
        graph.add_edge(edge);

        let registry = OperatorRegistry::new();
        let pass = InitializeConstantsPass::new();
        let changed = pass.run(&mut graph, &registry).unwrap();

        assert!(!changed);
    }
}
