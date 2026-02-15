//! Error types for the runtime crate.

use thiserror::Error;

/// Runtime execution errors.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// GPU initialization failed.
    #[error("GPU initialization failed: {0}")]
    InitError(String),

    /// Buffer allocation failed.
    #[error("Buffer allocation failed: {0}")]
    AllocationError(String),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),

    /// Execution failed.
    #[error("Execution failed: {0}")]
    ExecutionError(String),

    /// Invalid tensor.
    #[error("Invalid tensor: {0}")]
    TensorError(String),

    /// Tensor not found.
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// Dimension error.
    #[error("Dimension error: {0}")]
    DimensionError(String),

    /// Invalid input/output.
    #[error("Invalid input or output: {0}")]
    InvalidInputOutput(String),

    /// wgpu error.
    #[error("wgpu error: {0}")]
    WgpuError(String),

    /// Buffer async error.
    #[error("Buffer async error: {0}")]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),
}

/// Specialized Result type for runtime operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;
