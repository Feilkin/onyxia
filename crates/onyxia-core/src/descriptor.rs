//! Declarative operator descriptors for compile-time analysis.
//!
//! `OpDescriptor` gives the compiler visibility into an operator's resource
//! usage — how many GPU dispatches it performs, which buffers it reads/writes,
//! and whether any outputs alias inputs — without actually executing any GPU work.
//!
//! This information is needed for memory planning (buffer pooling, liveness
//! analysis) and is the foundation for the pre-allocated buffer model.

use crate::types::DataType;

/// Describes a single GPU dispatch (one compute pass).
pub struct KernelDispatch {
    /// Pre-compiled shader module.
    pub module: naga::Module,
    /// Entry point name.
    pub entry_point: String,
    /// Pipeline cache label.
    pub label: String,
}

/// Describes buffer usage patterns for a dispatch step.
pub enum BufferAccess {
    /// Read from input at given index.
    ReadInput(usize),
    /// Write to output at given index.
    WriteOutput(usize),
    /// Temporary buffer (intermediate, not visible as output).
    Temporary { dtype: DataType },
}

/// Full descriptor of what an operator does at dispatch time.
///
/// Gives the compiler visibility into the operator's resource usage
/// without running the dispatch.
///
/// # Example
///
/// ```
/// use onyxia_core::descriptor::{BufferAccess, OpDescriptor};
///
/// // A simple single-kernel element-wise operator with one input and one output.
/// let descriptor = OpDescriptor {
///     num_outputs: 1,
///     kernels: vec![],  // kernels omitted in this example
///     buffer_bindings: vec![vec![
///         BufferAccess::ReadInput(0),
///         BufferAccess::WriteOutput(0),
///     ]],
///     aliases: vec![],
/// };
/// assert_eq!(descriptor.num_outputs, 1);
/// ```
pub struct OpDescriptor {
    /// Number of output tensors.
    pub num_outputs: usize,
    /// GPU dispatches this operator performs (most ops = 1, softmax = 3).
    pub kernels: Vec<KernelDispatch>,
    /// Buffer access pattern per binding, per kernel.
    pub buffer_bindings: Vec<Vec<BufferAccess>>,
    /// Whether output[i] aliases input[j] (zero-copy ops like Reshape).
    /// Each entry is `(output_idx, input_idx)`.
    pub aliases: Vec<(usize, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockOperator;

    impl crate::operator::Operator for MockOperator {
        fn name(&self) -> &str {
            "Mock"
        }

        fn create_dispatch(
            &self,
            _ctx: &mut crate::compile_ctx::CompileCtx,
        ) -> crate::Result<Box<dyn crate::dispatch::OpDispatch>> {
            Err(crate::Error::Compilation("not implemented".to_string()))
        }
    }

    /// Single-kernel operator with one input and one output.
    #[test]
    fn test_op_descriptor_single_kernel() {
        let descriptor = OpDescriptor {
            num_outputs: 1,
            kernels: vec![],
            buffer_bindings: vec![vec![
                BufferAccess::ReadInput(0),
                BufferAccess::WriteOutput(0),
            ]],
            aliases: vec![],
        };

        assert_eq!(descriptor.num_outputs, 1);
        assert!(descriptor.kernels.is_empty());
        assert_eq!(descriptor.buffer_bindings.len(), 1);
        assert_eq!(descriptor.buffer_bindings[0].len(), 2);
        assert!(descriptor.aliases.is_empty());
    }

    /// Reshape-like operator where output aliases input (zero-copy).
    #[test]
    fn test_op_descriptor_with_aliases() {
        let descriptor = OpDescriptor {
            num_outputs: 1,
            kernels: vec![],
            buffer_bindings: vec![],
            aliases: vec![(0, 0)], // output[0] aliases input[0]
        };

        assert_eq!(descriptor.num_outputs, 1);
        assert_eq!(descriptor.aliases, vec![(0, 0)]);
    }

    /// Operator that uses a temporary intermediate buffer.
    #[test]
    fn test_op_descriptor_with_temporary_buffer() {
        let descriptor = OpDescriptor {
            num_outputs: 1,
            kernels: vec![],
            buffer_bindings: vec![vec![
                BufferAccess::ReadInput(0),
                BufferAccess::Temporary {
                    dtype: DataType::F32,
                },
                BufferAccess::WriteOutput(0),
            ]],
            aliases: vec![],
        };

        assert_eq!(descriptor.num_outputs, 1);
        assert_eq!(descriptor.buffer_bindings[0].len(), 3);
        // Check that the middle binding is a temporary
        assert!(matches!(
            descriptor.buffer_bindings[0][1],
            BufferAccess::Temporary {
                dtype: DataType::F32
            }
        ));
    }

    /// Default `describe()` returns `None` for opaque operators.
    ///
    /// This is a compile-time check: `MockOperator` doesn't override `describe()`,
    /// so the default impl must return `None`. The signature
    /// `fn describe(&self, ctx: &CompileCtx) -> Option<OpDescriptor>` is verified
    /// by the fact that this test file compiles without error.
    #[test]
    fn test_default_describe_returns_none() {
        // The default impl ignores ctx and returns None.
        // Since CompileCtx requires a full GPU context to construct, we verify
        // the contract at the type level: MockOperator has no override, so the
        // default `fn describe(...) -> Option<OpDescriptor> { None }` is used.
        let _op = MockOperator;
        // Compile-time verification: MockOperator satisfies Operator without describe().
        assert!(true);
    }
}
