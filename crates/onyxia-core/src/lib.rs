//! Core intermediate representation, operator traits, and dispatch types for Onyxia.
//!
//! This crate provides the foundational abstractions that all other Onyxia crates depend on:
//! - Graph-based IR (`IrGraph`, `IrNode`, `IrEdge`)
//! - Operator and Pass traits for extensibility
//! - Compile context for dispatch creation (`CompileCtx`)
//! - Dispatch types for execution (`DispatchModel`, `OpDispatch`)
//! - Operator registry for dynamic dispatch

pub mod broadcast;
pub mod compile_ctx;
pub mod dispatch;
pub mod ir;
pub mod ir_builder;
pub mod operator;
pub mod pass;
pub mod plan;
pub mod registry;
pub mod types;

// Re-export commonly used types
pub use broadcast::broadcast_shape;
pub use compile_ctx::CompileCtx;
pub use dispatch::{
    CompiledModel as DispatchModel, DispatchCtx, DispatchEntry, OpDispatch, RuntimeTensor,
    WeightRegister,
};
pub use ir::{EdgeData, IrEdge, IrEdgeId, IrGraph, IrNode, IrNodeId, IrTensorId, TensorDef};
pub use operator::Operator;
pub use pass::{Pass, Stage};
pub use plan::ModelMetadata;
pub use registry::OperatorRegistry;
pub use types::{DataType, TensorData, TensorShape, TensorValue};

/// Result type using the crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Core error type for onyxia-core operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Attribute error: {0}")]
    Attribute(String),

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Shape error: {0}")]
    Shape(String),

    #[error("Runtime error: {0}")]
    Runtime(String),
}
