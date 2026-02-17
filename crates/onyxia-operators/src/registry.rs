//! Core operator registry.
//!
//! Provides a pre-populated registry with core operators.

use onyxia_core::OperatorRegistry;

use crate::families::BinaryElementwiseOp;
use crate::operators::ReshapeOp;

/// Returns an operator registry pre-populated with core operators.
///
/// The registry includes:
/// - 5 binary elementwise operators (Add, Mul, Div, Sub, Pow)
/// - 1 shape manipulation operator (Reshape)
///
/// Custom operators can be added to the returned registry via
/// `registry.register(name, operator)`.
pub fn core_operator_registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();

    // Binary elementwise operators
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Mul", BinaryElementwiseOp::mul());
    registry.register("Div", BinaryElementwiseOp::div());
    registry.register("Sub", BinaryElementwiseOp::sub());
    registry.register("Pow", BinaryElementwiseOp::pow());

    // Shape manipulation operators
    registry.register("Reshape", ReshapeOp);

    registry
}
