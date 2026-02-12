//! Built-in kernel implementations for ONNX operators.

pub mod add;
pub mod cast;
pub mod constant;
pub mod gelu;
pub mod matmul_f32;
pub mod mul;
pub mod reshape;
pub mod rmsnorm;
pub mod shape;
pub mod sub;
pub mod unsqueeze;

pub use add::AddKernel;
pub use cast::CastKernel;
pub use constant::ConstantKernel;
pub use gelu::GeluKernel;
pub use matmul_f32::MatMulF32Kernel;
pub use mul::MulKernel;
pub use reshape::ReshapeKernel;
pub use rmsnorm::RmsNormKernel;
pub use shape::ShapeKernel;
pub use sub::SubKernel;
pub use unsqueeze::UnsqueezeKernel;
