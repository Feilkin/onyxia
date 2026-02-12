//! Runtime initialization and GPU device management.

use crate::error::{Result, RuntimeError};
use crate::executor::ModelExecutor;
use onyxia_codegen::CompiledModel;
use onyxia_onnx::{Dimension, TensorShape};
use std::collections::HashMap;
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
    instance: wgpu::Instance,
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
            .await;

        let adapter = match adapter {
            Ok(a) => a,
            Err(e) => {
                return Err(RuntimeError::InitError(format!(
                    "Failed to find suitable GPU adapter: {}",
                    e
                )));
            }
        };

        let adapter_info = adapter.get_info();

        Ok(Self {
            instance,
            adapter,
            adapter_info,
        })
    }

    /// Calculate the maximum buffer size required for a model.
    fn calculate_max_buffer_size(
        model: &CompiledModel,
        dynamic_dimensions: &HashMap<String, usize>,
    ) -> Result<u64> {
        let mut max_size = 0u64;

        for tensor_info in model.tensors.all() {
            let size = match &tensor_info.shape {
                TensorShape::Static(dims) => {
                    let element_count: usize = dims.iter().product();
                    let element_size = tensor_info.dtype.size();
                    (element_count * element_size) as u64
                }
                TensorShape::Dynamic(dims) => {
                    let mut element_count = 1usize;
                    for dim in dims {
                        match dim {
                            Dimension::Static(val) => element_count *= val,
                            Dimension::Named(name) => {
                                let val = dynamic_dimensions.get(name).ok_or_else(|| {
                                    RuntimeError::TensorError(format!(
                                        "Dynamic dimension '{}' not provided in dynamic_dimensions",
                                        name
                                    ))
                                })?;
                                element_count *= val;
                            }
                        }
                    }
                    let element_size = tensor_info.dtype.size();
                    (element_count * element_size) as u64
                }
                TensorShape::Unknown => {
                    return Err(RuntimeError::TensorError(format!(
                        "Cannot calculate buffer size for tensor '{}' with unknown shape",
                        tensor_info.name
                    )));
                }
            };

            max_size = max_size.max(size);
        }

        Ok(max_size)
    }

    /// Load a compiled model into an executor.
    ///
    /// This creates a GPU device with buffer limits calculated from the model's requirements.
    ///
    /// # Arguments
    /// * `model` - The compiled model to execute
    /// * `dynamic_dimensions` - Concrete values for symbolic dimensions (e.g., {"batch_size": 1, "sequence_length": 512})
    ///
    /// # Errors
    /// Returns an error if device creation, shader compilation, or buffer allocation fails.
    pub async fn load_model(
        &self,
        model: CompiledModel,
        dynamic_dimensions: HashMap<String, usize>,
    ) -> Result<ModelExecutor> {
        // Calculate maximum buffer size needed
        let max_buffer_size = Self::calculate_max_buffer_size(&model, &dynamic_dimensions)?;

        // Add 10% headroom for intermediate buffers
        let required_buffer_size = (max_buffer_size as f64 * 1.1) as u64;

        // Get default limits and override max_storage_buffer_binding_size
        let mut limits = wgpu::Limits::default();
        // Clamp to u32::MAX since wgpu uses u32 for buffer sizes
        limits.max_storage_buffer_binding_size = required_buffer_size.min(u32::MAX as u64) as u32;

        // Create device with calculated limits
        let (device, queue) = self
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Onyxia Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| {
                RuntimeError::InitError(format!(
                    "Failed to create device with max buffer size {}: {}",
                    required_buffer_size, e
                ))
            })?;

        ModelExecutor::new(
            Arc::new(device),
            Arc::new(queue),
            model,
            dynamic_dimensions,
        )
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
