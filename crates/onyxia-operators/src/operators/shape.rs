//! Shape manipulation operators.

use onyxia_core::{CompileCtx, DataType, OpDispatch, Operator, Result};

/// Reshape operator - changes tensor shape without copying data.
///
/// Since GPU buffers are flat arrays, no data movement is needed in theory.
/// However, our runtime allocates separate buffers per tensor, so we emit
/// a CopyBuffer step. Future optimization: buffer aliasing to avoid copies.
pub struct ReshapeOp;

impl Operator for ReshapeOp {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn create_dispatch(&self, _ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        Err(onyxia_core::Error::Compilation(
            "Reshape not yet ported to dispatch model (will be done in task 043)".to_string(),
        ))
    }
}
