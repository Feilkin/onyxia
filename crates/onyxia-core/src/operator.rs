//! Operator trait for extensible operation system.

use crate::Result;

/// Trait for implementing ONNX operators.
///
/// Operators create dispatch objects at compile time which execute the
/// actual GPU work at runtime. The dispatch model eliminates the need for
/// compile-time shape inference and constant folding — operators compute
/// everything they need from concrete input tensors at runtime.
///
/// # Example
///
/// ```ignore
/// struct AddOperator;
///
/// impl Operator for AddOperator {
///     fn name(&self) -> &str {
///         "Add"
///     }
///
///     fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
///         // Compile shader with naga_oil
///         let module = ctx.compile_shader("add", SHADER_SOURCE, &HashMap::new())?;
///         
///         Ok(Box::new(AddDispatch { module }))
///     }
/// }
/// ```
pub trait Operator: Send + Sync {
    /// Get the operator name (e.g., "Add", "MatMul", "RmsNorm").
    ///
    /// This should match the ONNX operation type for standard operators,
    /// or use a custom name for non-standard operators.
    fn name(&self) -> &str;

    /// Create a dispatch object for this operation.
    ///
    /// Called by the compiler when walking the ONNX graph. The returned
    /// `OpDispatch` implementation captures pre-compiled shaders and
    /// attributes, then performs the actual GPU work at runtime.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Compile context providing access to node attributes, weight
    ///   values, and shader compilation services.
    ///
    /// # Returns
    ///
    /// A boxed dispatch object that will execute this operation on the GPU.
    fn create_dispatch(
        &self,
        ctx: &mut crate::compile_ctx::CompileCtx,
    ) -> Result<Box<dyn crate::dispatch::OpDispatch>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock operator for testing
    struct MockOp;

    impl Operator for MockOp {
        fn name(&self) -> &str {
            "Mock"
        }

        fn create_dispatch(
            &self,
            _ctx: &mut crate::compile_ctx::CompileCtx,
        ) -> Result<Box<dyn crate::dispatch::OpDispatch>> {
            Err(crate::Error::Compilation("not implemented".to_string()))
        }
    }

    #[test]
    fn test_operator_trait_object() {
        let op: Box<dyn Operator> = Box::new(MockOp);
        assert_eq!(op.name(), "Mock");
    }
}
