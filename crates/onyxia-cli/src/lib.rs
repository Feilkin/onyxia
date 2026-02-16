//! Onyxia CLI library - shared functionality for testing and binary.

pub mod generate;
pub mod inspect;
pub mod llm;
pub mod sampling;
pub mod tokenizer;

/// Direction to trace around a node.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TraceDirection {
    Both,
    Upstream,
    Downstream,
}

/// Output format for trace.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TraceFormat {
    Text,
    Dot,
}
