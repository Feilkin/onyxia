//! Symbolic dimension resolution pass.
//!
//! Resolves symbolic dimensions in tensor shapes using the provided dynamic_dimensions map.
//! Fully resolved shapes become Static, partially resolved shapes stay Symbolic.

use onyxia_core::{
    IrGraph, IrTensorId, OperatorRegistry, Pass, Result, Stage, SymbolicDim, TensorShape,
};
use std::collections::HashMap;

/// Pass that resolves symbolic dimensions to concrete values.
///
/// Walks all tensor definitions in the IR and evaluates symbolic dimension expressions
/// against the provided dimension map. Fully resolved shapes become Static, partially
/// resolved shapes stay Symbolic with evaluated sub-expressions.
pub struct SymbolicResolutionPass {
    dynamic_dimensions: HashMap<String, usize>,
}

impl SymbolicResolutionPass {
    /// Create a new symbolic resolution pass.
    pub fn new(dynamic_dimensions: HashMap<String, usize>) -> Self {
        Self { dynamic_dimensions }
    }

    /// Resolve a symbolic dimension expression.
    fn resolve_dim(&self, dim: &SymbolicDim) -> SymbolicDim {
        match dim {
            SymbolicDim::Fixed(n) => SymbolicDim::Fixed(*n),
            SymbolicDim::Expr(expr) => {
                match onyxia_core::symbolic_expr::evaluate_expr(expr, &self.dynamic_dimensions) {
                    Ok(value) => SymbolicDim::Fixed(value),
                    Err(_) => SymbolicDim::Expr(expr.clone()), // Keep as symbolic if can't resolve
                }
            }
        }
    }

    /// Resolve a tensor shape.
    fn resolve_shape(&self, shape: &TensorShape) -> TensorShape {
        match shape {
            TensorShape::Static(_) | TensorShape::Absent => shape.clone(),
            TensorShape::Symbolic(dims) => {
                let resolved_dims: Vec<_> = dims.iter().map(|d| self.resolve_dim(d)).collect();

                // If all dimensions are now fixed, convert to Static
                if resolved_dims.iter().all(|d| d.is_fixed()) {
                    let static_dims: Vec<usize> = resolved_dims
                        .iter()
                        .filter_map(|d| match d {
                            SymbolicDim::Fixed(n) => Some(*n),
                            _ => None,
                        })
                        .collect();
                    TensorShape::Static(static_dims)
                } else {
                    TensorShape::Symbolic(resolved_dims)
                }
            }
        }
    }
}

impl Pass for SymbolicResolutionPass {
    fn name(&self) -> &str {
        "symbolic_resolution"
    }

    fn stage(&self) -> Stage {
        Stage::Resolution
    }

    fn run(&self, graph: &mut IrGraph, _registry: &OperatorRegistry) -> Result<bool> {
        let mut changed = false;

        // Resolve all tensor shapes
        let tensor_ids: Vec<IrTensorId> = (0..graph.tensor_count()).map(IrTensorId::new).collect();

        for tensor_id in tensor_ids {
            let tensor = graph.tensor(tensor_id)?;
            let old_shape = tensor.shape.clone();
            let new_shape = self.resolve_shape(&old_shape);

            if new_shape != old_shape {
                graph.tensor_mut(tensor_id)?.shape = new_shape;
                changed = true;
            }
        }

        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::{DataType, IrGraph, SymbolicExpr, TensorDef, TensorKind};

    #[test]
    fn test_resolve_static_dims() {
        let mut graph = IrGraph::new();
        let tensor = TensorDef::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2, 3]),
            TensorKind::Input,
        );
        graph.add_tensor(tensor);

        let registry = OperatorRegistry::new();
        let pass = SymbolicResolutionPass::new(HashMap::new());

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(!changed); // Already static, no change
    }

    #[test]
    fn test_resolve_symbolic_to_static() {
        let mut graph = IrGraph::new();
        let tensor = TensorDef::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Symbolic(vec![
                SymbolicDim::Expr(SymbolicExpr::Variable("batch".to_string())),
                SymbolicDim::Fixed(128),
            ]),
            TensorKind::Input,
        );
        let tensor_id = graph.add_tensor(tensor);

        let dims = HashMap::from([("batch".to_string(), 4)]);
        let registry = OperatorRegistry::new();
        let pass = SymbolicResolutionPass::new(dims);

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(changed);

        let resolved = &graph.tensor(tensor_id).unwrap().shape;
        assert_eq!(resolved, &TensorShape::Static(vec![4, 128]));
    }

    #[test]
    fn test_partial_resolution() {
        let mut graph = IrGraph::new();
        let tensor = TensorDef::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Symbolic(vec![
                SymbolicDim::Expr(SymbolicExpr::Variable("batch".to_string())),
                SymbolicDim::Expr(SymbolicExpr::Variable("seq".to_string())),
            ]),
            TensorKind::Input,
        );
        let tensor_id = graph.add_tensor(tensor);

        // Only provide "batch", not "seq"
        let dims = HashMap::from([("batch".to_string(), 4)]);
        let registry = OperatorRegistry::new();
        let pass = SymbolicResolutionPass::new(dims);

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(changed);

        let resolved = &graph.tensor(tensor_id).unwrap().shape;
        match resolved {
            TensorShape::Symbolic(dims) => {
                assert_eq!(dims[0], SymbolicDim::Fixed(4));
                assert!(matches!(dims[1], SymbolicDim::Expr(_)));
            }
            _ => panic!("Expected Symbolic shape after partial resolution"),
        }
    }

    #[test]
    fn test_resolve_expression() {
        use onyxia_core::{BinOpKind, SymbolicExpr};

        let mut graph = IrGraph::new();
        let expr = SymbolicExpr::BinOp(
            Box::new(SymbolicExpr::Variable("seq".to_string())),
            BinOpKind::Mul,
            Box::new(SymbolicExpr::Literal(8)),
        );
        let tensor = TensorDef::new(
            "x".to_string(),
            DataType::F32,
            TensorShape::Symbolic(vec![SymbolicDim::Fixed(1), SymbolicDim::Expr(expr)]),
            TensorKind::Input,
        );
        let tensor_id = graph.add_tensor(tensor);

        let dims = HashMap::from([("seq".to_string(), 64)]);
        let registry = OperatorRegistry::new();
        let pass = SymbolicResolutionPass::new(dims);

        let changed = pass.run(&mut graph, &registry).unwrap();
        assert!(changed);

        let resolved = &graph.tensor(tensor_id).unwrap().shape;
        assert_eq!(resolved, &TensorShape::Static(vec![1, 512]));
    }
}
