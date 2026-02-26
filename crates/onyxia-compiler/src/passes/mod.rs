//! Compiler passes for graph transformation and code generation.

mod initialize_constants;
mod shape_propagation;

pub use initialize_constants::InitializeConstantsPass;
pub use shape_propagation::ShapePropagationPass;
