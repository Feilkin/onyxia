//! Runtime initialization and GPU device management.

use crate::dispatch_executor::DispatchExecutor;
use crate::error::{Result, RuntimeError};
use onyxia_core::GpuContext;
use onyxia_core::dispatch::CompiledModel;
use tracing::instrument;

/// Main entry point for GPU runtime.
///
/// Manages GPU device initialization and provides methods to load and execute models.
/// Internally holds a [`GpuContext`] that can be shared with the compiler for
/// constant folding and feature-dependent optimizations.
///
/// # Example
/// ```no_run
/// # use onyxia_runtime::Runtime;
/// #[pollster::main]
/// async fn main() -> anyhow::Result<()> {
///     let runtime = Runtime::new().await?;
///     // Load and execute models...
///     Ok(())
/// }
/// ```
pub struct Runtime {
    gpu: GpuContext,
}

impl Runtime {
    /// Initialize the runtime, creating a shared GPU context.
    ///
    /// This selects the best available GPU and creates a device with
    /// `IMMEDIATES` feature support.
    ///
    /// # Errors
    /// Returns an error if no suitable GPU is found or device creation fails.
    #[instrument(name = "Runtime::new")]
    pub async fn new() -> Result<Self> {
        let gpu = GpuContext::new()
            .await
            .map_err(|e| RuntimeError::InitError(format!("{e}")))?;
        Ok(Self { gpu })
    }

    /// Get the shared GPU context.
    ///
    /// The returned context can be passed to the compiler for compile-time GPU
    /// operations such as constant folding, avoiding a second device creation.
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// Load a compiled model for execution.
    ///
    /// Reuses the shared GPU device/queue from this runtime — no additional
    /// device is created.
    ///
    /// # Arguments
    /// * `model` - Compiled model with dispatch entries and routing
    ///
    /// # Errors
    /// Returns an error if resource initialization fails.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_runtime::Runtime;
    /// # use onyxia_core::dispatch::CompiledModel;
    /// # #[pollster::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let runtime = Runtime::new().await?;
    /// # let model: CompiledModel = todo!();
    /// let executor = runtime.load_model(model)?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "Runtime::load_model", skip(self, model))]
    pub fn load_model(&self, model: CompiledModel) -> Result<DispatchExecutor> {
        DispatchExecutor::new(self.gpu.device.clone(), self.gpu.queue.clone(), model)
    }

    /// Get information about the GPU adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.gpu.adapter_info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[pollster::test]
    async fn test_runtime_init() {
        let runtime = Runtime::new().await;
        assert!(runtime.is_ok(), "Failed to initialize runtime");

        if let Ok(runtime) = runtime {
            let info = runtime.adapter_info();
            println!("GPU: {} ({:?})", info.name, info.backend);
        }
    }
}
