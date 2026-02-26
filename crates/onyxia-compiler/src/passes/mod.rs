//! Compiler passes for graph transformation and code generation.

mod constant_folding;
mod initialize_constants;
mod shape_propagation;

pub use constant_folding::ConstantFoldingPass;
pub use initialize_constants::InitializeConstantsPass;
pub use shape_propagation::ShapePropagationPass;
