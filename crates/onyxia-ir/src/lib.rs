//! Backend-neutral intermediate representation for Onyxia.
//!
//! This crate is the foundation described in `doc/ir-design.md`. It contains:
//!
//! - **Types and symbolic shapes** ([`DataType`], [`DimExpr`], [`SymbolicShape`],
//!   [`TensorType`]) — shapes are polynomials over named dimension symbols,
//!   resolved to concrete sizes when a session binds real inputs.
//! - **The graph** ([`Module`], [`Node`], [`Prim`], [`Composite`]) — an SSA
//!   value graph. Primitives are a *closed* enum with fully specified
//!   semantics; composites are named, attribute-carrying nodes whose
//!   decompositions live in a registry (see milestone B).
//! - **Shape inference** ([`infer`]) — one total function over the primitive
//!   set. Adding a primitive without a shape rule is a compile error.
//! - **The reference interpreter** ([`interp`]) — the executable
//!   specification of the primitives. Every backend kernel must match it
//!   within tolerance. Written for clarity, never for speed.
//! - **Constant folding** ([`fold`]) — evaluates constant subgraphs via the
//!   interpreter and propagates compile-time *shape values* (small integer
//!   tensors of [`DimExpr`]s) so ONNX shape-arithmetic chains disappear at
//!   lowering.
//!
//! This crate has no GPU dependencies and compiles on every target,
//! including `wasm32-unknown-unknown`.

pub mod attrs;
pub mod backend;
pub mod builder;
pub mod decomp;
pub mod dim;
pub mod dot;
pub mod fold;
pub mod graph;
pub mod infer;
pub mod interp;
pub mod prim;
pub mod types;
pub mod validate;

pub use attrs::{AttrValue, Attrs};
pub use backend::{Backend, Session};
pub use builder::GraphBuilder;
pub use decomp::{DecompositionRegistry, inline_composites, standard_decompositions};
pub use dim::{Bindings, DimExpr, SymId, SymbolTable, SymbolicShape};
pub use graph::{
    Composite, ConstId, ConstPool, Module, Node, NodeId, NodeKind, Origin, SourceInfo, ValueDef,
    ValueId,
};
pub use prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
pub use types::{DataType, TensorType};

/// Result type using the crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for IR construction, inference, validation, and interpretation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A shape could not be inferred or is inconsistent.
    #[error("shape error: {0}")]
    Shape(String),

    /// The graph is structurally invalid (dangling ids, arity, cycles, …).
    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    /// A data type rule was violated (mismatched or unsupported dtypes).
    #[error("dtype error: {0}")]
    DType(String),

    /// An attribute is missing or has the wrong type.
    #[error("attribute error: {0}")]
    Attribute(String),

    /// A symbol binding is missing, inconsistent, or evaluates negative.
    #[error("binding error: {0}")]
    Binding(String),

    /// The reference interpreter hit an execution error.
    #[error("interpreter error: {0}")]
    Interp(String),

    /// A feature or operation is not (yet) supported.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A backend failed at execution time (device loss, kernel compile,
    /// resource limits, …).
    #[error("runtime error: {0}")]
    Runtime(String),
}
