//! Execution of pre-planned models via ExecutionPlan.
//!
//! This module materializes an ExecutionPlan into GPU resources without
//! runtime shader compilation or dimension resolution.

use crate::error::{Result, RuntimeError};
use crate::tensor::Tensor;
use onyxia_onnx::{TensorId, TensorShape};
use onyxia_planner::plan::{BindingDesc, BufferRef, ExecutionPlan, PlannedOp, Step};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Materializes an execution plan into GPU resources and executes operations.
///
/// Key difference from ModelExecutor: Shaders arrive as pre-compiled naga modules,
/// dimensions are already resolved — no runtime preprocessing needed.
#[derive(Debug)]
pub struct PlanExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    plan: ExecutionPlan,

    /// One pipeline per entry in `plan.shaders` — indexed by ShaderIndex.
    pipelines: Vec<wgpu::ComputePipeline>,
    /// One bind group layout per pipeline (same indexing as `pipelines`).
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    /// GPU buffers for model tensors, keyed by TensorId.
    tensor_buffers: HashMap<TensorId, wgpu::Buffer>,
    /// Per-operation scratch buffers. `scratch_pools[op_index][scratch_index]`.
    scratch_pools: Vec<Vec<wgpu::Buffer>>,
}

impl PlanExecutor {
    /// Create a new plan executor with materialized GPU resources.
    ///
    /// # Arguments
    /// * `device` - GPU device
    /// * `queue` - Command queue
    /// * `plan` - Pre-compiled execution plan with resolved shapes
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        plan: ExecutionPlan,
    ) -> Result<Self> {
        let mut executor = Self {
            device: Arc::clone(&device),
            queue: Arc::clone(&queue),
            pipelines: Vec::with_capacity(plan.shaders.len()),
            bind_group_layouts: Vec::with_capacity(plan.shaders.len()),
            tensor_buffers: HashMap::new(),
            scratch_pools: Vec::with_capacity(plan.operations.len()),
            plan,
        };

        // Materialize GPU resources
        executor.create_pipelines()?;
        executor.allocate_tensor_buffers()?;
        executor.allocate_scratch_buffers()?;

        Ok(executor)
    }

    /// Create compute pipelines from pre-compiled naga modules.
    fn create_pipelines(&mut self) -> Result<()> {
        for (index, compiled_shader) in self.plan.shaders.iter().enumerate() {
            // Create shader module from pre-compiled naga module
            // This is the key difference: no WGSL parsing, no naga_oil preprocessing
            // We need to clone the module because ShaderSource::Naga requires owned data
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&compiled_shader.label),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(compiled_shader.module.clone())),
                });

            // Derive bind group layout from shader entry point
            let bind_group_layout =
                self.create_bind_group_layout_for_shader(index, compiled_shader)?;

            // Calculate immediate size from shader module
            let immediate_size = self.calculate_immediate_size(&compiled_shader.module)?;

            // Create pipeline layout
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("Pipeline Layout: {}", compiled_shader.label)),
                        bind_group_layouts: &[&bind_group_layout],
                        immediate_size,
                    });

            // Create compute pipeline
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Pipeline: {}", compiled_shader.label)),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(&compiled_shader.entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

            self.pipelines.push(pipeline);
            self.bind_group_layouts.push(bind_group_layout);
        }

        Ok(())
    }

    /// Create bind group layout for a shader by inspecting its bindings.
    fn create_bind_group_layout_for_shader(
        &self,
        _shader_index: usize,
        compiled_shader: &onyxia_planner::plan::CompiledShader,
    ) -> Result<wgpu::BindGroupLayout> {
        // Extract bindings from the naga module entry point
        let entry_point = compiled_shader
            .module
            .entry_points
            .iter()
            .find(|ep| ep.name == compiled_shader.entry_point)
            .ok_or_else(|| {
                RuntimeError::ShaderError(format!(
                    "Entry point '{}' not found in shader '{}'",
                    compiled_shader.entry_point, compiled_shader.label
                ))
            })?;

        // Build bind group layout entries from the shader's global variables
        let mut entries = Vec::new();
        for (handle, var) in compiled_shader.module.global_variables.iter() {
            if let Some(binding) = var.binding.as_ref() {
                // Check if this variable is used by the entry point
                let is_used = entry_point.function.expressions.iter().any(
                    |(_, expr)| matches!(expr, naga::Expression::GlobalVariable(h) if *h == handle),
                );

                if is_used || true {
                    // Determine read-only vs read-write from the storage access flags
                    let read_only = match var.space {
                        naga::AddressSpace::Storage { access } => {
                            // If STORE flag is not present, it's read-only
                            !access.contains(naga::StorageAccess::STORE)
                        }
                        _ => false, // Non-storage buffers default to read-write
                    };

                    entries.push(wgpu::BindGroupLayoutEntry {
                        binding: binding.binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
            }
        }

        // Sort by binding number
        entries.sort_by_key(|e| e.binding);

        Ok(self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Bind Group Layout: {}", compiled_shader.label)),
                entries: &entries,
            }))
    }

    /// Calculate the immediate data size required by a shader.
    ///
    /// Inspects the naga module for var<immediate> declarations and calculates
    /// the total size of immediate data required by the shader.
    fn calculate_immediate_size(&self, module: &naga::Module) -> Result<u32> {
        let mut max_size = 0u32;

        // Iterate through global variables looking for immediate address space
        for (_handle, global_var) in module.global_variables.iter() {
            if matches!(global_var.space, naga::AddressSpace::Immediate) {
                // Get the type info for this variable using handle indexing
                let type_info = &module.types[global_var.ty];
                // Calculate the size in bytes
                let size = self.calculate_type_size(&module.types, &type_info.inner)?;
                max_size = max_size.max(size);
            }
        }

        Ok(max_size)
    }

    /// Calculate the size of a naga type in bytes.
    fn calculate_type_size(
        &self,
        types: &naga::UniqueArena<naga::Type>,
        type_inner: &naga::TypeInner,
    ) -> Result<u32> {
        use naga::TypeInner;

        match type_inner {
            TypeInner::Scalar(scalar) => Ok(scalar.width as u32),
            TypeInner::Vector { scalar, size } => {
                let element_size = scalar.width as u32;
                let count = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                Ok(element_size * count)
            }
            TypeInner::Matrix {
                scalar,
                columns,
                rows,
            } => {
                let element_size = scalar.width as u32;
                let col_count = match columns {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                let row_count = match rows {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                Ok(element_size * col_count * row_count)
            }
            TypeInner::Struct { members, .. } => {
                // For structs, we need to account for alignment and padding
                // Use the offset + size of the last member
                if let Some(last_member) = members.last() {
                    let last_type = &types[last_member.ty];
                    let last_size = self.calculate_type_size(types, &last_type.inner)?;
                    Ok(last_member.offset + last_size)
                } else {
                    Ok(0)
                }
            }
            TypeInner::Array {
                base: _,
                size,
                stride,
            } => {
                let count = match size {
                    naga::ArraySize::Constant(c) => c.get(),
                    naga::ArraySize::Dynamic => {
                        return Err(RuntimeError::ShaderError(
                            "Dynamic arrays not supported in immediate data".to_string(),
                        ));
                    }
                    naga::ArraySize::Pending(_) => {
                        return Err(RuntimeError::ShaderError(
                            "Pending array size not supported in immediate data".to_string(),
                        ));
                    }
                };
                Ok(stride * count)
            }
            _ => Err(RuntimeError::ShaderError(format!(
                "Unsupported type in immediate data: {:?}",
                type_inner
            ))),
        }
    }

    /// Allocate GPU buffers for all model tensors.
    fn allocate_tensor_buffers(&mut self) -> Result<()> {
        for (id, info) in self.plan.tensors.all().iter().enumerate() {
            // All shapes are static at this point (resolved at plan time)
            let size = match &info.shape {
                TensorShape::Static(dims) => {
                    let element_count: usize = dims.iter().product();
                    let element_size = info.dtype.size();
                    (element_count * element_size) as u64
                }
                TensorShape::Dynamic(_) | TensorShape::Unknown | TensorShape::Absent => {
                    return Err(RuntimeError::TensorError(format!(
                        "Tensor '{}' has non-static shape in execution plan. \
                         All shapes should be resolved at plan time.",
                        info.name
                    )));
                }
            };

            // Ensure size is non-zero and properly aligned
            let size = size.max(4); // Minimum 4 bytes
            let size = (size + 3) & !3; // Align to 4 bytes

            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Tensor: {}", info.name)),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.tensor_buffers.insert(id, buffer);

            // Upload initializer data if present
            if let Some(ref initializer) = info.initializer {
                let buffer = self.tensor_buffers.get(&id).ok_or_else(|| {
                    RuntimeError::AllocationError(format!("Buffer {} not found", id))
                })?;

                // Ensure data is properly aligned
                let mut aligned_data = initializer.clone();
                while aligned_data.len() < size as usize {
                    aligned_data.push(0);
                }

                self.queue.write_buffer(buffer, 0, &aligned_data);
            }
        }

        Ok(())
    }

    /// Allocate scratch buffers for each operation.
    fn allocate_scratch_buffers(&mut self) -> Result<()> {
        for operation in &self.plan.operations {
            let mut scratch_buffers = Vec::new();

            for scratch_desc in &operation.scratch_buffers {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&scratch_desc.label),
                    size: scratch_desc.size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                scratch_buffers.push(buffer);
            }

            self.scratch_pools.push(scratch_buffers);
        }

        Ok(())
    }

    /// Execute the plan with provided inputs.
    ///
    /// # Arguments
    /// * `inputs` - Named input tensors
    ///
    /// # Returns
    /// Named output tensors
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Upload inputs to GPU
        for (name, tensor) in inputs {
            let (tensor_id, _) = self.plan.tensors.find_by_name(name).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Input '{}' not found", name))
            })?;

            let buffer = self.tensor_buffers.get(&tensor_id).ok_or_else(|| {
                RuntimeError::AllocationError(format!("Buffer for input '{}' not found", name))
            })?;

            let data = tensor.raw_data()?;
            self.queue.write_buffer(buffer, 0, data);
        }

        // Execute all operations
        let operations = self.plan.operations.clone();
        for (op_index, operation) in operations.iter().enumerate() {
            self.execute_operation(op_index, operation)?;
        }

        // Ensure all GPU operations complete before downloading results
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::ExecutionError(format!("GPU poll failed: {:?}", e)))?;

        // Download outputs from GPU
        let mut outputs = HashMap::new();
        for output_id in &self.plan.outputs {
            let info = self.plan.tensors.get(*output_id).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Output {} not found", output_id))
            })?;

            let data = self.download_tensor(*output_id)?;

            // Extract shape
            let shape: Vec<usize> = match &info.shape {
                TensorShape::Static(dims) => dims.iter().map(|&d| d as usize).collect(),
                _ => {
                    return Err(RuntimeError::TensorError(
                        "Output tensor has non-static shape".to_string(),
                    ));
                }
            };

            outputs.insert(
                info.name.clone(),
                Tensor::from_raw(data, &shape, info.dtype),
            );
        }

        Ok(outputs)
    }

    /// Execute a single operation by dispatching its steps.
    fn execute_operation(&mut self, op_index: usize, operation: &PlannedOp) -> Result<()> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Operation: {}", operation.name)),
            });

        for step in &operation.steps {
            match step {
                Step::Dispatch {
                    shader_index,
                    bindings,
                    workgroups,
                    immediates,
                } => {
                    self.execute_dispatch(
                        &mut encoder,
                        op_index,
                        *shader_index,
                        bindings,
                        *workgroups,
                        immediates.as_deref(),
                    )?;
                }
                Step::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    self.execute_copy(
                        &mut encoder,
                        op_index,
                        src,
                        *src_offset,
                        dst,
                        *dst_offset,
                        *size,
                    )?;
                }
                Step::WriteBuffer { dst, data } => {
                    self.execute_write(&mut encoder, op_index, dst, data)?;
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Execute a dispatch step.
    fn execute_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        op_index: usize,
        shader_index: usize,
        bindings: &[BindingDesc],
        workgroups: [u32; 3],
        immediates: Option<&[u8]>,
    ) -> Result<()> {
        let pipeline = self.pipelines.get(shader_index).ok_or_else(|| {
            RuntimeError::ShaderError(format!("Pipeline {} not found", shader_index))
        })?;

        let bind_group_layout = self.bind_group_layouts.get(shader_index).ok_or_else(|| {
            RuntimeError::ShaderError(format!("Bind group layout {} not found", shader_index))
        })?;

        // Create bind group entries from bindings
        let mut entries = Vec::new();
        for (binding_index, binding_desc) in bindings.iter().enumerate() {
            let buffer = self.resolve_buffer_ref(op_index, &binding_desc.buffer)?;

            entries.push(wgpu::BindGroupEntry {
                binding: binding_index as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dispatch Bind Group"),
            layout: bind_group_layout,
            entries: &entries,
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Set immediate data if provided
        if let Some(data) = immediates {
            compute_pass.set_immediates(0, data);
        }

        compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);

        Ok(())
    }

    /// Execute a buffer copy step.
    fn execute_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        op_index: usize,
        src: &BufferRef,
        src_offset: u64,
        dst: &BufferRef,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let src_buffer = self.resolve_buffer_ref(op_index, src)?;
        let dst_buffer = self.resolve_buffer_ref(op_index, dst)?;

        encoder.copy_buffer_to_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size);

        Ok(())
    }

    /// Execute a buffer write step.
    fn execute_write(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        op_index: usize,
        dst: &BufferRef,
        data: &[u8],
    ) -> Result<()> {
        let dst_buffer = self.resolve_buffer_ref(op_index, dst)?;
        self.queue.write_buffer(dst_buffer, 0, data);

        Ok(())
    }

    /// Resolve a buffer reference to an actual GPU buffer.
    fn resolve_buffer_ref(&self, op_index: usize, buffer_ref: &BufferRef) -> Result<&wgpu::Buffer> {
        match buffer_ref {
            BufferRef::Tensor(tensor_id) => self.tensor_buffers.get(tensor_id).ok_or_else(|| {
                RuntimeError::AllocationError(format!("Tensor buffer {} not found", tensor_id))
            }),
            BufferRef::Scratch(scratch_index) => {
                let scratch_pool = self.scratch_pools.get(op_index).ok_or_else(|| {
                    RuntimeError::AllocationError(format!(
                        "Scratch pool for operation {} not found",
                        op_index
                    ))
                })?;

                scratch_pool.get(*scratch_index).ok_or_else(|| {
                    RuntimeError::AllocationError(format!(
                        "Scratch buffer {} for operation {} not found",
                        scratch_index, op_index
                    ))
                })
            }
        }
    }

    /// Download a tensor from GPU to CPU.
    fn download_tensor(&self, tensor_id: TensorId) -> Result<Vec<u8>> {
        let buffer = self.tensor_buffers.get(&tensor_id).ok_or_else(|| {
            RuntimeError::AllocationError(format!("Buffer {} not found", tensor_id))
        })?;

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer.size());

        self.queue.submit(Some(encoder.finish()));

        // Poll the device to ensure the copy completes before mapping
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                RuntimeError::ExecutionError(format!("GPU poll failed during download: {:?}", e))
            })?;

        // Map staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Poll the device to trigger the map callback
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                RuntimeError::ExecutionError(format!("GPU poll failed during mapping: {:?}", e))
            })?;

        // Wait for the GPU to finish (receiver.await will handle this)
        pollster::block_on(rx)
            .map_err(|_| RuntimeError::ExecutionError("Failed to receive map result".to_string()))?
            .map_err(|e| RuntimeError::ExecutionError(format!("Map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_planner::plan::ExecutionPlan;
    use onyxia_planner::{ModelMetadata, TensorRegistry};

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_plan_executor_empty_plan() {
        let runtime = crate::Runtime::new().await.unwrap();

        // Create an empty execution plan
        let plan = ExecutionPlan {
            operations: Vec::new(),
            shaders: Vec::new(),
            tensors: TensorRegistry::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: ModelMetadata {
                name: "test_empty".to_string(),
                version: 1,
                ir_version: 9,
                producer: "test".to_string(),
            },
        };

        // Should be able to load an empty plan without crashing
        let result = runtime.load_model(plan).await;
        assert!(result.is_ok(), "Failed to load empty plan: {:?}", result);
    }
}
