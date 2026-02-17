//! Operator registry for dynamic dispatch.

use crate::operator::Operator;
use std::collections::HashMap;

/// Registry for operator implementations.
///
/// Maps ONNX operation type names (e.g., "Add", "MatMul") to their corresponding
/// `Operator` implementations. This enables dynamic dispatch during compilation.
///
/// # Example
///
/// ```ignore
/// let mut registry = OperatorRegistry::new();
/// registry.register("Add", AddOperator::new());
/// registry.register("MatMul", MatMulOperator::new());
///
/// let op = registry.get("Add").unwrap();
/// // Use op to create dispatch objects during compilation
/// ```
pub struct OperatorRegistry {
    /// Map from op_type string to operator implementation.
    operators: HashMap<String, Box<dyn Operator>>,
}

impl OperatorRegistry {
    /// Create a new empty operator registry.
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    /// Register an operator.
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = OperatorRegistry::new()
    ///     .register("Add", AddOperator)
    ///     .register("Mul", MulOperator);
    /// ```
    pub fn register<O>(&mut self, name: &str, operator: O) -> &mut Self
    where
        O: Operator + 'static,
    {
        self.operators.insert(name.to_string(), Box::new(operator));
        self
    }

    /// Look up an operator by name.
    ///
    /// Returns `None` if no operator is registered with the given name.
    pub fn get(&self, name: &str) -> Option<&dyn Operator> {
        self.operators.get(name).map(|op| op.as_ref())
    }

    /// Check if an operator is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.operators.contains_key(name)
    }

    /// Get the number of registered operators.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Iterate over all registered operator names.
    pub fn operator_names(&self) -> impl Iterator<Item = &str> {
        self.operators.keys().map(|s| s.as_str())
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompileCtx, OpDispatch, Result};

    // Mock operators for testing
    struct AddOp;
    impl Operator for AddOp {
        fn name(&self) -> &str {
            "Add"
        }
        fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
            Err(crate::Error::Compilation(
                "mock operator for testing".to_string(),
            ))
        }
    }

    struct MulOp;
    impl Operator for MulOp {
        fn name(&self) -> &str {
            "Mul"
        }
        fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
            Err(crate::Error::Compilation(
                "mock operator for testing".to_string(),
            ))
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = OperatorRegistry::new();
        registry.register("Add", AddOp);
        registry.register("Mul", MulOp);

        assert_eq!(registry.len(), 2);
        assert!(registry.contains("Add"));
        assert!(registry.contains("Mul"));
        assert!(!registry.contains("Sub"));

        let add_op = registry.get("Add").unwrap();
        assert_eq!(add_op.name(), "Add");

        let mul_op = registry.get("Mul").unwrap();
        assert_eq!(mul_op.name(), "Mul");

        assert!(registry.get("Sub").is_none());
    }

    #[test]
    fn test_method_chaining() {
        let mut registry = OperatorRegistry::new();
        registry.register("Add", AddOp).register("Mul", MulOp);

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_operator_names() {
        let mut registry = OperatorRegistry::new();
        registry.register("Add", AddOp);
        registry.register("Mul", MulOp);

        let mut names: Vec<_> = registry.operator_names().collect();
        names.sort();

        assert_eq!(names, vec!["Add", "Mul"]);
    }

    #[test]
    fn test_empty_registry() {
        let registry = OperatorRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }
}
