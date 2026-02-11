//! Runtime initialization and GPU device management.

use crate::error::{Result, RuntimeError};
use crate::executor::ModelExecutor;
use onyxia_codegen::CompiledModel;
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
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
}

impl Runtime {
    /// Initialize the runtime with the default GPU adapter.
    ///
    /// This will automatically select the best available GPU.
    ///
    /// # Errors
    /// Returns an error if no suitable GPU is found or initialization fails.
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
            .await;
        
        let adapter = match adapter {
            Ok(a) => a,
            Err(e) => return Err(RuntimeError::InitError(format!("Failed to find suitable GPU adapter: {}", e))),
        };
        
        Self::with_adapter(&adapter).await
    }
    
    /// Initialize the runtime with a specific GPU adapter.
    ///
    /// Useful for selecting a specific GPU in multi-GPU systems.
    ///
    /// # Errors
    /// Returns an error if device initialization fails.
    pub async fn with_adapter(adapter: &wgpu::Adapter) -> Result<Self> {
        let adapter_info = adapter.get_info();
        
        let device_desc = wgpu::DeviceDescriptor::default();
        
        let (device, queue) = adapter
            .request_device(&device_desc)
            .await
            .map_err(|e| RuntimeError::InitError(format!("Failed to create device: {}", e)))?;
        
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
    
    /// Load a compiled model into an executor.
    ///
    /// # Errors
    /// Returns an error if shader compilation or buffer allocation fails.
    pub fn load_model(&self, model: CompiledModel) -> Result<ModelExecutor> {
        ModelExecutor::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            model,
        )
    }
    
    /// Get information about the GPU adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
    
    /// Get a reference to the device.
    pub(crate) fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    
    /// Get a reference to the queue.
    pub(crate) fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
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
