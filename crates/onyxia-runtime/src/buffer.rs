//! GPU buffer allocation and management.

use crate::error::{Result, RuntimeError};
use onyxia_onnx::{TensorId, TensorInfo};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU buffer wrapper with metadata.
#[derive(Debug)]
pub(crate) struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    /// Create a new GPU buffer.
    pub fn new(buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffer,
            size,
            usage,
        }
    }
}

/// Manages GPU buffer allocation and data transfer.
pub(crate) struct BufferManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: HashMap<TensorId, GpuBuffer>,
}

impl BufferManager {
    /// Create a new buffer manager.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            buffers: HashMap::new(),
        }
    }
    
    /// Allocate a buffer for a tensor.
    pub fn allocate(&mut self, tensor_id: TensorId, info: &TensorInfo) -> Result<()> {
        let size = self.calculate_buffer_size(info)?;
        
        // Determine usage based on tensor kind
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Tensor: {}", info.name)),
            size,
            usage,
            mapped_at_creation: false,
        });
        
        self.buffers.insert(
            tensor_id,
            GpuBuffer::new(buffer, size, usage),
        );
        
        Ok(())
    }
    
    /// Upload data from CPU to GPU buffer.
    pub fn upload(&self, tensor_id: TensorId, data: &[u8]) -> Result<()> {
        let gpu_buffer = self.buffers.get(&tensor_id).ok_or_else(|| {
            RuntimeError::TensorNotFound(format!("Tensor ID {} not found", tensor_id))
        })?;
        
        if data.len() as u64 > gpu_buffer.size {
            return Err(RuntimeError::AllocationError(format!(
                "Data size {} exceeds buffer size {}",
                data.len(),
                gpu_buffer.size
            )));
        }
        
        self.queue.write_buffer(&gpu_buffer.buffer, 0, data);
        Ok(())
    }
    
    /// Download data from GPU buffer to CPU.
    pub async fn download(&self, tensor_id: TensorId) -> Result<Vec<u8>> {
        let gpu_buffer = self.buffers.get(&tensor_id).ok_or_else(|| {
            RuntimeError::TensorNotFound(format!("Tensor ID {} not found", tensor_id))
        })?;
        
        // Create a staging buffer for reading back data
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: gpu_buffer.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from GPU buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download encoder"),
            });
        
        encoder.copy_buffer_to_buffer(
            &gpu_buffer.buffer,
            0,
            &staging_buffer,
            0,
            gpu_buffer.size,
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // Map the staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        
        // Wait for the GPU to finish
        // TODO: Fix polling API for wgpu 28
        // self.device.poll(wgpu::MaintainResult::Wait {});
        
        receiver
            .await
            .map_err(|_| RuntimeError::ExecutionError("Failed to receive map result".to_string()))?
            .map_err(|e| RuntimeError::BufferAsyncError(e))?;
        
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
    
    /// Get a reference to a GPU buffer.
    pub fn get_buffer(&self, tensor_id: TensorId) -> Option<&wgpu::Buffer> {
        self.buffers.get(&tensor_id).map(|b| &b.buffer)
    }
    
    /// Free a buffer (for intermediate tensors).
    pub fn free(&mut self, tensor_id: TensorId) {
        self.buffers.remove(&tensor_id);
    }
    
    /// Calculate the size of a buffer for a tensor.
    fn calculate_buffer_size(&self, info: &TensorInfo) -> Result<u64> {
        use onyxia_onnx::{DataType, TensorShape};
        
        let element_count = match &info.shape {
            TensorShape::Static(dims) => dims.iter().map(|&d| d as usize).product::<usize>(),
            TensorShape::Dynamic(_) | TensorShape::Unknown => {
                return Err(RuntimeError::TensorError(
                    "Dynamic/unknown shapes not yet supported".to_string(),
                ))
            }
        };
        
        let element_size = match info.dtype {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::I32 => 4,
            DataType::I64 => 8,
            DataType::U8 => 1,
            DataType::U32 => 4,
            DataType::Bool => 1,
            DataType::Q4 => 1, // Packed, actual size depends on packing
            DataType::Q8 => 1,
        };
        
        // Ensure alignment (WGPU requires buffer sizes to be multiples of 4)
        let size = (element_count * element_size) as u64;
        let aligned_size = (size + 3) & !3; // Round up to multiple of 4
        
        Ok(aligned_size)
    }
    
    /// Get the total number of allocated buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}
