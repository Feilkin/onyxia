//! Error types for codegen operations.

use thiserror::Error;

/// Result type for codegen operations.
pub type Result<T> = std::result::Result<T, CodegenError>;

/// Errors that can occur during code generation.
#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("Unsupported operator: {0}")]
    UnsupportedOp(String),

    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Shader generation failed: {0}")]
    ShaderError(String),

    #[error("ONNX error: {0}")]
    OnnxError(#[from] onyxia_onnx::OnnxError),

    #[error("Scheduling error: {0}")]
    SchedulingError(String),
}
