//! Optimization pass trait and stage definitions.

use crate::Result;
use crate::ir::IrGraph;
use crate::registry::OperatorRegistry;

/// Compilation stage for organizing passes.
///
/// Passes are grouped into stages and run in a fixed order. Within each stage,
/// passes run in the order they were registered.
///
/// Note: In the current dispatch-based architecture, only the Resolution stage
/// is actively used (for InitializeConstants pass). Other stages are preserved
/// for future extensibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Resolution stage (earliest).
    ///
    /// Currently used for the InitializeConstants pass which parses
    /// ONNX weight data.
    Resolution,

    /// Folding stage (reserved for future use).
    ///
    /// Could be used for compile-time constant evaluation if re-introduced.
    Folding,

    /// Inference stage (reserved for future use).
    ///
    /// Could be used for compile-time shape inference if re-introduced.
    Inference,

    /// Optimization stage (reserved for future use).
    ///
    /// Could be used for graph rewriting passes (dead code elimination,
    /// operator fusion, etc.).
    Optimization,

    /// Planning stage (reserved for future use).
    ///
    /// Could be used for advanced code generation passes.
    Planning,
}

/// Trait for implementing compiler passes.
///
/// A pass is a graph transformation that runs during a specific compilation
/// stage. Passes are standalone objects (not owned by operators) to allow
/// optimizations that span multiple operator types.
///
/// # Return Value
///
/// The `run()` method returns `Ok(true)` if the pass made changes to the
/// graph, or `Ok(false)` if no changes were made. This allows the compiler
/// to detect when the graph reaches a fixed point and skip unnecessary work.
///
/// # Example
///
/// ```ignore
/// struct DeadCodeEliminationPass;
///
/// impl Pass for DeadCodeEliminationPass {
///     fn name(&self) -> &str {
///         "dead_code_elimination"
///     }
///
///     fn stage(&self) -> Stage {
///         Stage::Optimization
///     }
///
///     fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool> {
///         let mut changed = false;
///         // Remove nodes that produce no outputs...
///         Ok(changed)
///     }
/// }
/// ```
pub trait Pass: Send + Sync {
    /// Get the pass name (used for logging and debugging).
    fn name(&self) -> &str;

    /// Get the compilation stage this pass belongs to.
    fn stage(&self) -> Stage;

    /// Run the pass on the given graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The IR graph to transform (mutable).
    /// * `registry` - The operator registry (for looking up operator implementations).
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the pass made changes to the graph.
    /// * `Ok(false)` if no changes were made.
    /// * `Err(_)` if the pass encountered an error.
    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock pass for testing
    struct NoOpPass;

    impl Pass for NoOpPass {
        fn name(&self) -> &str {
            "noop"
        }

        fn stage(&self) -> Stage {
            Stage::Optimization
        }

        fn run(&self, _graph: &mut IrGraph, _registry: &OperatorRegistry) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn test_pass_trait() {
        let pass: Box<dyn Pass> = Box::new(NoOpPass);
        assert_eq!(pass.name(), "noop");
        assert_eq!(pass.stage(), Stage::Optimization);
    }

    #[test]
    fn test_stage_ordering() {
        assert!(Stage::Resolution < Stage::Folding);
        assert!(Stage::Folding < Stage::Inference);
        assert!(Stage::Inference < Stage::Optimization);
        assert!(Stage::Optimization < Stage::Planning);
    }
}
