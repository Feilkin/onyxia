//! Core intermediate representation, operator traits, and plan types for Onyxia.
//!
//! This crate provides the foundational abstractions that all other Onyxia crates depend on:
//! - Graph-based IR (`IrGraph`, `IrNode`, `IrEdge`)
//! - Operator and Pass traits for extensibility
//! - Context types for shape inference, constant folding, and planning
//! - Plan types for execution (`CompiledModel`, `PlannedOp`, `Step`)
//! - Operator registry for dynamic dispatch

pub mod context;
pub mod ir;
pub mod ir_builder;
pub mod operator;
pub mod pass;
pub mod plan;
pub mod registry;
pub mod symbolic_expr;
pub mod types;

// Re-export commonly used types
pub use context::{FoldCtx, InferenceCtx, PlanCtx};
pub use ir::{IrEdge, IrEdgeId, IrGraph, IrNode, IrNodeId, IrTensorId, TensorDef};
pub use operator::Operator;
pub use pass::{Pass, Stage};
pub use plan::{
    BindingDesc, BufferRef, CompiledModel, CompiledShader, ModelMetadata, PlannedOp,
    ScratchBufferDesc, ShaderIndex, Step, SymbolicBinding, TensorMetadata, TensorRegistry,
};
pub use registry::OperatorRegistry;
pub use symbolic_expr::{BinOpKind, SymbolicExpr};
pub use types::{DataType, SymbolicDim, TensorData, TensorKind, TensorShape, TensorValue};

/// Result type using the crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Core error type for onyxia-core operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Shape inference error: {0}")]
    ShapeInference(String),

    #[error("Constant folding error: {0}")]
    ConstantFolding(String),

    #[error("Planning error: {0}")]
    Planning(String),

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Attribute error: {0}")]
    Attribute(String),

    #[error("Symbolic expression error: {0}")]
    SymbolicExpr(String),

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}
