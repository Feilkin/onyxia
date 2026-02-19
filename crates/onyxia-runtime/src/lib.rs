//! GPU executor for compiled Onyxia graphs.
//!
//! This crate executes the compiled graphs from `onyxia-compiler` on the GPU
//! using `wgpu` as the hardware abstraction layer.
//!
//! # Architecture
//!
//! The runtime manages three main responsibilities:
//! 1. **GPU initialization** - Set up wgpu device and queue
//! 2. **Buffer management** - Allocate/upload/download GPU buffers
//! 3. **Execution** - Dispatch compute operations through register-based routing
//!
//! # Example
//!
//! ```no_run
//! use onyxia_runtime::{Runtime, Tensor};
//! use onyxia_compiler::compile;
//! use onyxia_onnx::load_model;
//!
//! #[pollster::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize runtime
//!     let runtime = Runtime::new().await?;
//!     
//!     // Load and compile model
//!     let onnx_model = load_model("model.onnx")?;
//!     let compiled_model = compile(&onnx_model)?;
//!     
//!     // Create executor
//!     let mut executor = runtime.load_model(compiled_model).await?;
//!     
//!     // Prepare input
//!     let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
//!     
//!     // Execute model
//!     let outputs = executor.run(&[("input", input)])?;
//!     
//!     // Extract output
//!     let result = outputs["output"].to_vec::<f32>()?;
//!     println!("Result: {:?}", result);
//!     
//!     Ok(())
//! }
//! ```

mod dispatch_executor;
mod error;
mod runtime;
mod tensor;

// Public exports
pub use dispatch_executor::DispatchExecutor;
pub use error::{Result, RuntimeError};
pub use runtime::Runtime;
pub use tensor::Tensor;
