//! Runtime initialization and GPU device management.

use crate::error::{Result, RuntimeError};
use crate::plan_executor::PlanExecutor;
use onyxia_onnx::TensorShape;
use onyxia_compiler::ExecutionPlan;
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

    /// Calculate maximum buffer size from an execution plan.
    ///
    /// All shapes are static in an execution plan, so this is straightforward.
    fn calculate_max_buffer_size(plan: &ExecutionPlan) -> Result<u64> {
        let mut max_size = 0u64;

        for tensor_info in plan.tensors.all() {
            let size = match &tensor_info.shape {
                TensorShape::Static(dims) => {
                    let element_count: usize = dims.iter().product();
                    let element_size = tensor_info.dtype.size();
                    (element_count * element_size) as u64
                }
                TensorShape::Dynamic(_) | TensorShape::Unknown | TensorShape::Absent => {
                    return Err(RuntimeError::TensorError(format!(
                        "Tensor '{}' in execution plan has non-static shape. \
                         All shapes should be resolved at plan time.",
                        tensor_info.name
                    )));
                }
            };

            max_size = max_size.max(size);
        }

        // Also account for scratch buffers
        for operation in &plan.operations {
            for scratch_desc in &operation.scratch_buffers {
                max_size = max_size.max(scratch_desc.size);
            }
        }

        Ok(max_size)
    }

    /// Load an execution plan into a model executor.
    ///
    /// This creates a GPU device and materializes pre-compiled shaders into pipelines.
    /// Dynamic dimensions are already resolved at plan-time, so all shapes are static.
    ///
    /// # Arguments
    /// * `plan` - Pre-compiled execution plan with resolved shapes and naga modules
    ///
    /// # Errors
    /// Returns an error if device creation or resource materialization fails.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_runtime::Runtime;
    /// # use onyxia_compiler::ExecutionPlan;
    /// # #[pollster::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let runtime = Runtime::new().await?;
    /// # let plan: ExecutionPlan = todo!();
    /// let executor = runtime.load_model(plan).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model(&self, plan: ExecutionPlan) -> Result<PlanExecutor> {
        // Calculate maximum buffer size needed
        let max_buffer_size = Self::calculate_max_buffer_size(&plan)?;

        // Add 10% headroom for intermediate buffers
        let required_buffer_size = max_buffer_size;

        // Check if adapter can support the required buffer size
        let adapter_limits = self.adapter.limits();
        if required_buffer_size > adapter_limits.max_buffer_size {
            return Err(RuntimeError::InitError(format!(
                "Model requires buffer size of {} bytes ({:.2} MB), but adapter only supports {} bytes ({:.2} MB)",
                required_buffer_size,
                required_buffer_size as f64 / (1024.0 * 1024.0),
                adapter_limits.max_buffer_size,
                adapter_limits.max_buffer_size as f64 / (1024.0 * 1024.0)
            )));
        }

        // Get default limits and override buffer size limits
        let mut limits = wgpu::Limits::default();
        let required_size_u32 = required_buffer_size.min(u32::MAX as u64) as u32;

        // Set both buffer size and binding size limits
        limits.max_buffer_size = required_buffer_size;
        limits.max_storage_buffer_binding_size = required_size_u32;

        // Request immediate data support (push constants)
        limits.max_immediate_size = 128;

        // Create device with calculated limits
        let (device, queue) = self
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Onyxia Device"),
                required_features: wgpu::Features::IMMEDIATES,
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

        PlanExecutor::new(Arc::new(device), Arc::new(queue), plan)
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
