//! Model metadata types.
//!
//! Contains metadata about ONNX models (name, version, producer, etc.).
//! The old execution plan types (PlannedOp, Step, CompiledShader) have been
//! removed in favor of the dispatch-based execution model in dispatch.rs.

/// Model metadata (name, version, producer, etc.).
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Model name.
    pub name: String,

    /// IR version.
    pub ir_version: i64,

    /// Producer name.
    pub producer_name: String,

    /// Model version.
    pub model_version: i64,
}
