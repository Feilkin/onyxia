//! Execution of pre-planned models via CompiledModel.
//!
//! This module materializes a CompiledModel into GPU resources without
//! runtime shader compilation or dimension resolution.

use crate::error::{Result, RuntimeError};
use crate::tensor::Tensor;
use onyxia_core::{BindingDesc, BufferRef, CompiledModel, CompiledShader, PlannedOp, Step};
use onyxia_onnx::TensorId;
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
    plan: CompiledModel,

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
        plan: CompiledModel,
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
        compiled_shader: &CompiledShader,
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

                if is_used {
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
    ///
    /// If a tensor has `initial_data` (weight / constant known at compile
    /// time), the data is uploaded to the GPU buffer immediately after
    /// allocation via `queue.write_buffer()`.
    fn allocate_tensor_buffers(&mut self) -> Result<()> {
        for (id, info) in self.plan.tensors.all().iter().enumerate() {
            // All shapes are static at this point (resolved at plan time)
            let size = match &info.shape {
                onyxia_core::TensorShape::Static(dims) => {
                    let element_count: usize = dims.iter().product();
                    let element_size = info.dtype.size();
                    (element_count * element_size) as u64
                }
                onyxia_core::TensorShape::Symbolic(_)
                | onyxia_core::TensorShape::Absent
                | onyxia_core::TensorShape::Unknown => {
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

            // Upload initial data (weights / constants) to the GPU buffer
            if let Some(data) = &info.initial_data {
                self.queue.write_buffer(&buffer, 0, data);
            }

            self.tensor_buffers.insert(id, buffer);
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

    /// Copy tensor buffers from source to destination.
    ///
    /// This copies the GPU buffer contents from source tensors to destination tensors.
    /// Both tensors must exist in the execution plan. If the destination buffer is
    /// smaller than the source buffer, it will be automatically reallocated.
    ///
    /// This is useful for patterns where outputs from one execution need to become
    /// inputs to the next execution (e.g., recurrent models, stateful operations).
    ///
    /// # Arguments
    /// * `copies` - Pairs of (source_tensor_name, destination_tensor_name) to copy
    ///
    /// # Example
    /// ```ignore
    /// let copies = vec![
    ///     ("output_state".to_string(), "input_state".to_string()),
    ///     ("output_hidden".to_string(), "input_hidden".to_string()),
    /// ];
    /// executor.copy_tensors(&copies)?;
    /// ```
    pub fn copy_tensors(&mut self, copies: &[(String, String)]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tensor Copy"),
            });

        for (source_name, dest_name) in copies {
            let (source_id, source_info) = self
                .plan
                .tensors
                .find_by_name(source_name)
                .ok_or_else(|| RuntimeError::TensorNotFound(source_name.to_string()))?;
            let (dest_id, _) = self
                .plan
                .tensors
                .find_by_name(dest_name)
                .ok_or_else(|| RuntimeError::TensorNotFound(dest_name.to_string()))?;

            // Calculate buffer size from source tensor shape
            let size = match &source_info.shape {
                onyxia_core::TensorShape::Static(dims) => {
                    let element_count: usize = dims.iter().product();
                    let element_size = source_info.dtype.size();
                    let raw_size = (element_count * element_size) as u64;
                    // Align to 4 bytes and ensure minimum size
                    (raw_size.max(4) + 3) & !3
                }
                _ => {
                    return Err(RuntimeError::TensorError(format!(
                        "Tensor '{}' has non-static shape",
                        source_name
                    )));
                }
            };

            // Check if destination buffer needs reallocation
            let needs_reallocation = {
                let dest_buffer = self.tensor_buffers.get(&dest_id.0).ok_or_else(|| {
                    RuntimeError::AllocationError(format!("Buffer for '{}' not found", dest_name))
                })?;
                dest_buffer.size() < size
            };

            // If destination buffer is too small, reallocate it to match the source buffer size
            if needs_reallocation {
                let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Tensor: {} (reallocated)", dest_name)),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                self.tensor_buffers.insert(dest_id.0, new_buffer);
            }

            // Get the buffers for copying
            let source_buffer = self.tensor_buffers.get(&source_id.0).ok_or_else(|| {
                RuntimeError::AllocationError(format!("Buffer for '{}' not found", source_name))
            })?;
            let dest_buffer = self.tensor_buffers.get(&dest_id.0).unwrap();

            encoder.copy_buffer_to_buffer(source_buffer, 0, dest_buffer, 0, size);
        }

        // Submit the copy commands
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Get a reference to the execution plan.
    ///
    /// This provides access to the plan's tensor registry and other metadata.
    pub fn plan(&self) -> &CompiledModel {
        &self.plan
    }

    /// Update symbolic dimensions at runtime.
    ///
    /// Evaluates each `SymbolicBinding` expression with the new dimension values
    /// and patches the immediates in-place. Returns `true` if any shaders need
    /// recompilation (because they used dimensions as shader defs rather than
    /// immediates).
    ///
    /// For symbolic dims (encoded as immediates): patches immediately, no recompilation.
    /// For static dims (encoded as shader defs): flags affected shaders for recompilation.
    ///
    /// # Arguments
    ///
    /// * `dims` - New dimension values to apply
    ///
    /// # Returns
    ///
    /// Returns `true` if any dimensions that were compiled as shader defs have
    /// changed, indicating that shader recompilation is needed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A symbolic expression references an undefined variable
    /// - Expression evaluation fails (division by zero, overflow, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut dims = HashMap::new();
    /// dims.insert("sequence_length".to_string(), 256);
    /// dims.insert("batch_size".to_string(), 2);
    ///
    /// let needs_recompilation = executor.update_dimensions(&dims)?;
    /// if needs_recompilation {
    ///     // Need to recompile shaders that used dimensions as shader defs
    ///     // For now, this returns true but doesn't actually recompile
    ///     // Full recompilation would require rebuilding the plan
    /// }
    /// ```
    pub fn update_dimensions(&mut self, dims: &HashMap<String, usize>) -> Result<bool> {
        use onyxia_core::symbolic_expr::evaluate_expr;

        // Track if any shader-def dimensions changed (not yet implemented)
        let needs_recompilation = false;

        // Process all symbolic bindings
        for binding in &self.plan.symbolic_bindings {
            // Evaluate the expression with new dimension values
            let new_value = evaluate_expr(&binding.expr, dims).map_err(|e| {
                RuntimeError::DimensionError(format!(
                    "Failed to evaluate dimension expression: {}",
                    e
                ))
            })?;

            // Find the operation that uses this shader
            let operation = self
                .plan
                .operations
                .iter_mut()
                .find(|op| {
                    op.steps.iter().any(|step| match step {
                        Step::Dispatch { shader_index, .. } => {
                            *shader_index == binding.shader_index
                        }
                        _ => false,
                    })
                })
                .ok_or_else(|| {
                    RuntimeError::ExecutionError(format!(
                        "No operation found for shader index {}",
                        binding.shader_index
                    ))
                })?;

            // Patch the immediates in the dispatch step
            for step in &mut operation.steps {
                if let Step::Dispatch {
                    shader_index,
                    immediates,
                    ..
                } = step
                    && *shader_index == binding.shader_index
                {
                    // Ensure immediates buffer exists and is large enough
                    let immediates_buf = immediates.get_or_insert_with(Vec::new);

                    // Ensure buffer is large enough
                    let required_size = binding.immediate_offset + 4; // u32 = 4 bytes
                    if immediates_buf.len() < required_size {
                        immediates_buf.resize(required_size, 0);
                    }

                    // Patch the value at the recorded offset
                    let bytes = (new_value as u32).to_le_bytes();
                    immediates_buf[binding.immediate_offset..binding.immediate_offset + 4]
                        .copy_from_slice(&bytes);
                }
            }
        }

        Ok(needs_recompilation)
    }

    /// Execute the plan but only download the specified outputs to CPU.
    ///
    /// All outputs are still computed on GPU, but only the named ones are
    /// transferred to CPU and returned. This is useful for operations like
    /// KV caching where most outputs (e.g., `present.*` tensors) should stay
    /// on GPU, and only specific outputs (e.g., `logits`) need to be downloaded.
    ///
    /// # Arguments
    /// * `inputs` - Named input tensors to upload
    /// * `output_names` - Names of outputs to download
    ///
    /// # Returns
    /// Named output tensors (only the requested ones)
    pub fn run_with_outputs(
        &mut self,
        inputs: &[(&str, Tensor)],
        output_names: &[&str],
    ) -> Result<HashMap<String, Tensor>> {
        // Upload inputs to GPU
        for (name, tensor) in inputs {
            let (tensor_id, _) = self.plan.tensors.find_by_name(name).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Input '{}' not found", name))
            })?;

            let buffer = self.tensor_buffers.get(&tensor_id.0).ok_or_else(|| {
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

        // Download only the requested outputs from GPU
        let mut outputs = HashMap::new();
        for output_name in output_names {
            let (output_id, info) =
                self.plan.tensors.find_by_name(output_name).ok_or_else(|| {
                    RuntimeError::TensorNotFound(format!("Output '{}' not found", output_name))
                })?;

            let data = self.download_tensor(output_id.0)?;

            // Extract shape
            let shape: Vec<usize> = match &info.shape {
                onyxia_core::TensorShape::Static(dims) => dims.iter().copied().collect(),
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

    /// Execute the plan and download all outputs.
    ///
    /// This is a convenience wrapper around `run_with_outputs()` that downloads
    /// all outputs defined in the execution plan.
    ///
    /// # Arguments
    /// * `inputs` - Named input tensors to upload
    ///
    /// # Returns
    /// Named output tensors
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Collect all output names (separate from the mutable borrow)
        let output_ids = self.plan.outputs.clone();
        let output_names: Vec<String> = output_ids
            .iter()
            .filter_map(|id| self.plan.tensors.get(*id).map(|info| info.name.clone()))
            .collect();
        let output_name_refs: Vec<&str> = output_names.iter().map(|s| s.as_str()).collect();

        // Delegate to run_with_outputs
        self.run_with_outputs(inputs, &output_name_refs)
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
            BufferRef::Tensor(tensor_id) => {
                self.tensor_buffers.get(&tensor_id.0).ok_or_else(|| {
                    RuntimeError::AllocationError(format!(
                        "Tensor buffer {:?} not found",
                        tensor_id
                    ))
                })
            }
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

    /// Download a tensor from GPU to CPU for testing purposes.
    ///
    /// This method is public to support integration tests but is not
    /// part of the stable API.
    #[doc(hidden)]
    pub fn download_tensor(&self, tensor_id: TensorId) -> Result<Vec<u8>> {
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
    /* Tests temporarily disabled - need updating for new onyxia-core types
        use super::*;
        use onyxia_compiler::plan::CompiledModel;
        use onyxia_compiler::{ModelMetadata, TensorRegistry};
        use onyxia_onnx::{DataType, TensorInfo, TensorKind};

        #[pollster::test]
        #[ignore] // Requires GPU
        async fn test_plan_executor_empty_plan() {
            let runtime = crate::Runtime::new().await.unwrap();

            // Create an empty execution plan
            let plan = CompiledModel {
                operations: Vec::new(),
                shaders: Vec::new(),
                tensors: TensorRegistry::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                symbolic_bindings: Vec::new(),
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

        #[pollster::test]
        #[ignore] // Requires GPU
        async fn test_copy_tensors() {
            let runtime = crate::Runtime::new().await.unwrap();

            // Create a plan with source and destination tensors
            let mut tensors = TensorRegistry::new();
            let source_id = tensors.add(TensorInfo {
                name: "source_tensor".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Output,
                initializer: None,
            });
            let dest_id = tensors.add(TensorInfo {
                name: "dest_tensor".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Input,
                initializer: None,
            });

            let plan = CompiledModel {
                operations: Vec::new(),
                shaders: Vec::new(),
                tensors,
                inputs: vec![dest_id],
                outputs: vec![source_id],
                symbolic_bindings: Vec::new(),
                metadata: ModelMetadata {
                    name: "test_copy_tensors".to_string(),
                    version: 1,
                    ir_version: 9,
                    producer: "test".to_string(),
                },
            };

            let mut executor = runtime.load_model(plan).await.unwrap();

            // Upload initial data to destination (simulating first run)
            let zero_data = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], &[4]);

            executor
                .run(&[("dest_tensor", zero_data)])
                .expect("First run failed");

            // Manually upload to source buffer (simulating it being written by a compute shader)
            executor.queue.write_buffer(
                executor.tensor_buffers.get(&source_id.0).unwrap(),
                0,
                &[1.0f32, 2.0, 3.0, 4.0]
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );

            // Copy source -> destination
            let copies = vec![("source_tensor".to_string(), "dest_tensor".to_string())];
            executor.copy_tensors(&copies).expect("Tensor copy failed");

            // Wait for copy to complete
            executor
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();

            // Verify that destination now has the data from source
            let dest_result = executor.download_tensor(dest_id).unwrap();
            let dest_f32: Vec<f32> = dest_result
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            assert_eq!(
                dest_f32,
                vec![1.0, 2.0, 3.0, 4.0],
                "Destination buffer should contain copied data from source"
            );

            // Verify that source still has its original data (copy, not move)
            let source_result = executor.download_tensor(source_id).unwrap();
            let source_f32: Vec<f32> = source_result
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            assert_eq!(
                source_f32,
                vec![1.0, 2.0, 3.0, 4.0],
                "Source buffer should still contain original data"
            );
        }

        #[pollster::test]
        #[ignore] // Requires GPU
        async fn test_update_dimensions() {
            use onyxia_core::symbolic_expr::{SymbolicExpr, parse_expr};

            let runtime = crate::Runtime::new().await.unwrap();

            // Create a plan with symbolic bindings
            let mut tensors = TensorRegistry::new();
            let input_id = tensors.add(TensorInfo {
                name: "input".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Input,
                initializer: None,
            });

            // Create a simple shader
            let shader = onyxia_compiler::plan::CompiledShader {
                label: "test_shader".to_string(),
                module: naga::Module::default(),
                entry_point: "main".to_string(),
            };

            // Create a step with immediates
            let step = onyxia_compiler::plan::Step::Dispatch {
                shader_index: 0,
                bindings: vec![onyxia_compiler::plan::BindingDesc {
                    buffer: onyxia_compiler::plan::BufferRef::Tensor(input_id),
                    read_only: false,
                }],
                workgroups: [1, 1, 1],
                immediates: Some(vec![0, 0, 0, 0]), // Initial value for seq_len
            };

            // Create operation
            let operation = onyxia_compiler::plan::PlannedOp {
                name: "test_op".to_string(),
                op_type: "Test".to_string(),
                inputs: vec![input_id],
                outputs: vec![],
                steps: vec![step],
                scratch_buffers: Vec::new(),
            };

            // Create a symbolic binding
            let expr = parse_expr("seq_len").unwrap();
            let binding = onyxia_core::SymbolicBinding {
                shader_index: 0,
                immediate_offset: 0,
                expr,
            };

            let plan = CompiledModel {
                operations: vec![operation],
                shaders: vec![shader],
                tensors,
                inputs: vec![input_id],
                outputs: vec![],
                symbolic_bindings: vec![binding],
                metadata: ModelMetadata {
                    name: "test_update_dims".to_string(),
                    version: 1,
                    ir_version: 9,
                    producer: "test".to_string(),
                },
            };

            let mut executor = runtime.load_model(plan).await.unwrap();

            // Update dimensions
            let mut dims = HashMap::new();
            dims.insert("seq_len".to_string(), 256);

            let needs_recompilation = executor.update_dimensions(&dims).unwrap();
            assert!(!needs_recompilation); // No shader-def dimensions changed

            // Verify the immediate was patched
            match &executor.plan.operations[0].steps[0] {
                onyxia_compiler::plan::Step::Dispatch { immediates, .. } => {
                    let immediates_buf = immediates.as_ref().unwrap();
                    let value = u32::from_le_bytes([
                        immediates_buf[0],
                        immediates_buf[1],
                        immediates_buf[2],
                        immediates_buf[3],
                    ]);
                    assert_eq!(value, 256, "Immediate should be updated to 256");
                }
                _ => panic!("Expected Dispatch step"),
            }

            // Update dimensions again with different value
            dims.insert("seq_len".to_string(), 512);
            executor.update_dimensions(&dims).unwrap();

            // Verify the immediate was updated again
            match &executor.plan.operations[0].steps[0] {
                onyxia_compiler::plan::Step::Dispatch { immediates, .. } => {
                    let immediates_buf = immediates.as_ref().unwrap();
                    let value = u32::from_le_bytes([
                        immediates_buf[0],
                        immediates_buf[1],
                        immediates_buf[2],
                        immediates_buf[3],
                    ]);
                    assert_eq!(value, 512, "Immediate should be updated to 512");
                }
                _ => panic!("Expected Dispatch step"),
            }
        }

        #[pollster::test]
        #[ignore] // Requires GPU
        async fn test_update_dimensions_expression() {
            use onyxia_core::symbolic_expr::parse_expr;

            let runtime = crate::Runtime::new().await.unwrap();

            // Create a plan with a symbolic binding for an expression
            let mut tensors = TensorRegistry::new();
            let input_id = tensors.add(TensorInfo {
                name: "input".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Input,
                initializer: None,
            });

            let shader = onyxia_compiler::plan::CompiledShader {
                label: "test_shader".to_string(),
                module: naga::Module::default(),
                entry_point: "main".to_string(),
            };

            let step = onyxia_compiler::plan::Step::Dispatch {
                shader_index: 0,
                bindings: vec![onyxia_compiler::plan::BindingDesc {
                    buffer: onyxia_compiler::plan::BufferRef::Tensor(input_id),
                    read_only: false,
                }],
                workgroups: [1, 1, 1],
                immediates: Some(vec![0, 0, 0, 0]),
            };

            let operation = onyxia_compiler::plan::PlannedOp {
                name: "test_op".to_string(),
                op_type: "Test".to_string(),
                inputs: vec![input_id],
                outputs: vec![],
                steps: vec![step],
                scratch_buffers: Vec::new(),
            };

            // Create a symbolic binding with an expression
            let expr = parse_expr("seq_len * num_heads").unwrap();
            let binding = onyxia_core::SymbolicBinding {
                shader_index: 0,
                immediate_offset: 0,
                expr,
            };

            let plan = CompiledModel {
                operations: vec![operation],
                shaders: vec![shader],
                tensors,
                inputs: vec![input_id],
                outputs: vec![],
                symbolic_bindings: vec![binding],
                metadata: ModelMetadata {
                    name: "test_update_dims_expr".to_string(),
                    version: 1,
                    ir_version: 9,
                    producer: "test".to_string(),
                },
            };

            let mut executor = runtime.load_model(plan).await.unwrap();

            // Update dimensions
            let mut dims = HashMap::new();
            dims.insert("seq_len".to_string(), 64);
            dims.insert("num_heads".to_string(), 8);

            executor.update_dimensions(&dims).unwrap();

            // Verify the expression was evaluated correctly (64 * 8 = 512)
            match &executor.plan.operations[0].steps[0] {
                onyxia_compiler::plan::Step::Dispatch { immediates, .. } => {
                    let immediates_buf = immediates.as_ref().unwrap();
                    let value = u32::from_le_bytes([
                        immediates_buf[0],
                        immediates_buf[1],
                        immediates_buf[2],
                        immediates_buf[3],
                    ]);
                    assert_eq!(value, 512, "Expression should evaluate to 64 * 8 = 512");
                }
                _ => panic!("Expected Dispatch step"),
            }
        }

        #[pollster::test]
        #[ignore] // Requires GPU
        async fn test_run_with_outputs_subset() {
            let runtime = crate::Runtime::new().await.unwrap();

            // Create a plan with multiple outputs
            let mut tensors = TensorRegistry::new();
            let input_id = tensors.add(TensorInfo {
                name: "input".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Input,
                initializer: None,
            });
            let output1_id = tensors.add(TensorInfo {
                name: "output1".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Output,
                initializer: None,
            });
            let output2_id = tensors.add(TensorInfo {
                name: "output2".to_string(),
                shape: TensorShape::Static(vec![4]),
                dtype: DataType::F32,
                kind: TensorKind::Output,
                initializer: None,
            });

            let plan = CompiledModel {
                operations: Vec::new(),
                shaders: Vec::new(),
                tensors,
                inputs: vec![input_id],
                outputs: vec![output1_id, output2_id],
                symbolic_bindings: Vec::new(),
                metadata: ModelMetadata {
                    name: "test_selective_output".to_string(),
                    version: 1,
                    ir_version: 9,
                    producer: "test".to_string(),
                },
            };

            let mut executor = runtime.load_model(plan).await.unwrap();

            // Run with input
            let input_data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]);

            // Request only output1, not output2
            let outputs = executor
                .run_with_outputs(&[("input", input_data.clone())], &["output1"])
                .expect("run_with_outputs failed");

            // Verify we got only the requested output
            assert_eq!(outputs.len(), 1, "Should only have one output");
            assert!(outputs.contains_key("output1"), "Missing output1");
            assert!(!outputs.contains_key("output2"), "Should not have output2");
        }
    } // mod tests
    */
}
