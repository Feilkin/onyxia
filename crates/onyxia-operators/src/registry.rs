//! Core operator registry.
//!
//! Provides a pre-populated registry with 3 minimal core operators.

use onyxia_core::OperatorRegistry;

use crate::families::BinaryElementwiseOp;
use crate::operators::ReshapeOp;

/// Returns an operator registry pre-populated with 3 core operators.
///
/// The registry includes:
/// - 2 binary elementwise operators (Add, Mul)
/// - 1 shape manipulation operator (Reshape)
///
/// Custom operators can be added to the returned registry via
/// `registry.register(name, operator)`.
pub fn core_operator_registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();

    // Binary elementwise operators
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Mul", BinaryElementwiseOp::mul());

    // Shape manipulation operators
    registry.register("Reshape", ReshapeOp);

    registry
}
