//! Binary elementwise operator family.
//!
//! Covers: Add, Mul

use onyxia_core::{CompileCtx, OpDispatch, Operator, Result};

/// Binary elementwise operator family.
///
/// All binary elementwise operations share the same structure:
/// - NumPy-style broadcasting for shape inference
/// - Element-by-element computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (which WGSL function to call)
/// - Fold functions (which CPU operations to perform)
pub struct BinaryElementwiseOp {
    name: &'static str,
}

impl BinaryElementwiseOp {
    /// Create an Add operator.
    pub fn add() -> Self {
        Self { name: "Add" }
    }

    /// Create a Mul operator.
    pub fn mul() -> Self {
        Self { name: "Mul" }
    }
}

impl Operator for BinaryElementwiseOp {
    fn name(&self) -> &str {
        self.name
    }

    fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        Err(onyxia_core::Error::Compilation(format!(
            "{} not yet ported to dispatch model (will be done in task 043)",
            self.name
        )))
    }
}
