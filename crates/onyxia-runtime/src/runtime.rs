//! Runtime initialization and GPU device management.

use crate::dispatch_executor::DispatchExecutor;
use crate::error::{Result, RuntimeError};
use onyxia_core::dispatch::CompiledModel;
use std::sync::Arc;

/// Main entry point for GPU runtime.
///
/// Manages GPU device initialization and provides methods to load and execute models.
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
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    adapter_info: wgpu::AdapterInfo,
}

impl Runtime {
    /// Initialize the runtime with the default GPU adapter.
    ///
    /// This will automatically select the best available GPU.
    /// Device creation is deferred until `load_model()` when buffer requirements are known.
    ///
    /// # Errors
    /// Returns an error if no suitable GPU is found.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                RuntimeError::InitError(format!("Failed to find suitable GPU adapter: {e}"))
            })?;

        let adapter_info = adapter.get_info();

        Ok(Self {
            _instance: instance,
            adapter,
            adapter_info,
        })
    }

    /// Load a compiled model for execution.
    ///
    /// Creates a GPU device and initializes the dispatch executor with weight data.
    ///
    /// # Arguments
    /// * `model` - Compiled model with dispatch entries and routing
    ///
    /// # Errors
    /// Returns an error if device creation or resource initialization fails.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_runtime::Runtime;
    /// # use onyxia_core::dispatch::CompiledModel;
    /// # #[pollster::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let runtime = Runtime::new().await?;
    /// # let model: CompiledModel = todo!();
    /// let executor = runtime.load_model(model).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model(&self, model: CompiledModel) -> Result<DispatchExecutor> {
        // Request device with appropriate limits
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = 4 * 1024 * 1024 * 1024; // 4GB max
        limits.max_storage_buffer_binding_size = 2 * 1024 * 1024 * 1024; // 2GB max
        limits.max_immediate_size = 128;

        // Create device with calculated limits
        let (device, queue) = self
            .adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Onyxia Device"),
                    required_features: wgpu::Features::IMMEDIATES,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| {
                RuntimeError::InitError(format!("Failed to create device: {}", e))
            })?;

        DispatchExecutor::new(Arc::new(device), Arc::new(queue), model)
    }

    /// Get information about the GPU adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
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
