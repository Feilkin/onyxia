//! Operator trait for extensible operation system.

use crate::Result;
use crate::context::{FoldCtx, InferenceCtx, PlanCtx};
use crate::plan::Step;
use crate::types::{TensorShape, TensorValue};

/// Trait for implementing ONNX operators.
///
/// Operators are responsible for three key tasks during compilation:
///
/// 1. **Shape inference** via `infer_output_shapes()` — determine output shapes
///    from input shapes and attributes. Must return `Err` if shapes cannot be
///    determined; the `TensorShape::Unknown` variant has been removed to force
///    explicit error handling.
///
/// 2. **Constant folding** via `try_fold()` — attempt to evaluate the operation
///    at compile time if all inputs are known constants. Returns `None` for
///    outputs that cannot be folded. Default implementation performs no folding.
///
/// 3. **Planning** via `plan()` — emit GPU execution steps (shader dispatches,
///    buffer copies, etc.) for the operation. Called only for nodes that were
///    not fully constant-folded.
///
/// All three methods are called with context objects that provide access to
/// node inputs, outputs, attributes, and graph state.
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
///     fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
///         let a_shape = ctx.input_shape(0)?;
///         let b_shape = ctx.input_shape(1)?;
///         // Broadcast shapes...
///         Ok(vec![result_shape])
///     }
///
///     fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
///         // Compile shader, emit dispatch step...
///         Ok(vec![step])
///     }
/// }
/// ```
pub trait Operator: Send + Sync {
    /// Get the operator name (e.g., "Add", "MatMul", "RmsNorm").
    ///
    /// This should match the ONNX operation type for standard operators,
    /// or use a custom name for non-standard operators.
    fn name(&self) -> &str;

    /// Infer output shapes from input shapes and attributes.
    ///
    /// This method is called during the shape inference pass to determine
    /// the shapes of all output tensors produced by this operator.
    ///
    /// # Requirements
    ///
    /// - Must return `Err` if output shapes cannot be determined. The
    ///   `TensorShape::Unknown` variant has been removed, so operators
    ///   cannot silently propagate unknown shapes.
    ///
    /// - Should support symbolic shapes when possible (e.g., broadcast
    ///   operations can propagate symbolic dimensions).
    ///
    /// - For data-dependent shape inference (e.g., Reshape), check if
    ///   required inputs have constant values via `ctx.input_value()`.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Context providing access to input shapes, attributes, and
    ///   constant-folded values.
    ///
    /// # Returns
    ///
    /// A vector of output shapes, one per output tensor. The length must
    /// match the number of outputs declared in the node.
    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>>;

    /// Attempt constant folding (compile-time evaluation).
    ///
    /// This method is called during the constant folding pass to try to
    /// evaluate the operation at compile time. If all inputs are known
    /// constants and the operation can be performed on the CPU, this
    /// should return the computed output values.
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns an empty vector, indicating
    /// that no outputs can be folded. Most operators should use this
    /// default unless they implement CPU fallback logic.
    ///
    /// # Returns
    ///
    /// A vector of optional values, one per output tensor. `Some(value)`
    /// indicates the output was successfully folded to a constant; `None`
    /// indicates it could not be folded. An empty vector indicates no
    /// folding was attempted (equivalent to all `None`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
    ///     if !ctx.all_inputs_have_values() {
    ///         return Ok(vec![]);
    ///     }
    ///     // Perform constant folding...
    ///     Ok(vec![Some(result)])
    /// }
    /// ```
    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let _ = ctx; // Suppress unused parameter warning
        Ok(vec![])
    }

    /// Emit execution steps for this operation.
    ///
    /// This method is called during the planning phase to generate GPU
    /// execution steps. It should compile any required WGSL shaders,
    /// allocate scratch buffers, and emit dispatch/copy/write steps.
    ///
    /// This is only called for nodes that were not fully constant-folded.
    /// If all outputs were folded to constants, the node is removed from
    /// the execution graph and this method is not called.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Context providing access to input/output buffer references,
    ///   tensor shapes, shader compilation, and scratch allocation.
    ///
    /// # Returns
    ///
    /// A vector of execution steps to perform. Common step types:
    /// - `Step::Dispatch` — run a compute shader
    /// - `Step::CopyBuffer` — copy data between buffers
    /// - `Step::WriteBuffer` — write immediate data to a buffer
    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>>;

    /// Create a dispatch object for this operation.
    ///
    /// Called by the compiler when walking the ONNX graph. The dispatch
    /// object captures pre-compiled shaders and attributes, then executes
    /// the actual GPU work at runtime via `OpDispatch::dispatch()`.
    ///
    /// # Default Implementation
    ///
    /// Returns an error — operators must implement this method to work
    /// with the new dispatch-based architecture.
    fn create_dispatch(
        &self,
        ctx: &mut crate::compile_ctx::CompileCtx,
    ) -> Result<Box<dyn crate::dispatch::OpDispatch>> {
        let _ = ctx;
        Err(crate::Error::Compilation(format!(
            "Operator '{}' does not implement create_dispatch()",
            self.name()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock operator for testing
    struct IdentityOp;

    impl Operator for IdentityOp {
        fn name(&self) -> &str {
            "Identity"
        }

        fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
            let input_shape = ctx.input_shape(0)?;
            Ok(vec![input_shape.clone()])
        }

        fn plan(&self, _ctx: &mut PlanCtx) -> Result<Vec<Step>> {
            // Would normally emit dispatch steps
            Ok(vec![])
        }
    }

    #[test]
    fn test_operator_trait_object() {
        let op: Box<dyn Operator> = Box::new(IdentityOp);
        assert_eq!(op.name(), "Identity");
    }

    #[test]
    fn test_default_try_fold() {
        // The default try_fold implementation should return an empty vector
        // This test will fail to compile until we implement FoldCtx, but
        // it demonstrates the expected behavior
    }
}
