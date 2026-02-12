//! Built-in kernel implementations for ONNX operators.

pub mod add;
pub mod constant;
pub mod gelu;
pub mod matmul_f32;
pub mod mul;
pub mod rmsnorm;
pub mod shape;

pub use add::AddKernel;
pub use constant::ConstantKernel;
pub use gelu::GeluKernel;
pub use matmul_f32::MatMulF32Kernel;
pub use mul::MulKernel;
pub use rmsnorm::RmsNormKernel;
pub use shape::ShapeKernel;
