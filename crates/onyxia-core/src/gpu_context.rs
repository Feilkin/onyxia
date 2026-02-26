//! Shared GPU context for both compilation and runtime.
//!
//! `GpuContext` is created once and shared by both the compiler (for constant
//! folding and feature queries) and the runtime (for model execution).

use crate::Result;
use std::sync::Arc;

/// Shared GPU context for both compilation and runtime.
///
/// Created once, shared by both the compiler (for constant folding and
/// feature queries) and the runtime (for model execution).
///
/// # Example
///
/// ```no_run
/// # #[pollster::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use onyxia_core::GpuContext;
///
/// let gpu = GpuContext::new().await?;
/// println!("GPU: {} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
/// # Ok(())
/// # }
/// ```
pub struct GpuContext {
    /// GPU device for resource creation and pipeline setup.
    pub device: Arc<wgpu::Device>,

    /// Command queue for GPU submissions.
    pub queue: Arc<wgpu::Queue>,

    /// Information about the selected GPU adapter.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize with the default high-performance GPU adapter.
    ///
    /// Selects the best available GPU, creates a device with
    /// `IMMEDIATES` feature support, and returns the shared context.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable GPU is found or device creation fails.
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
                crate::Error::Runtime(format!("Failed to find suitable GPU adapter: {e}"))
            })?;

        let adapter_info = adapter.get_info();

        // Use adapter's limits, only bumping what we need
        let mut limits = adapter.limits();
        limits.max_immediate_size = limits.max_immediate_size.max(128);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Onyxia Device"),
                required_features: wgpu::Features::IMMEDIATES,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| crate::Error::Runtime(format!("Failed to create GPU device: {e}")))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
}
