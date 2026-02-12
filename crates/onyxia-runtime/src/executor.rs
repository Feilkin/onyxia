//! Model execution orchestration.

use crate::buffer::BufferManager;
use crate::error::{Result, RuntimeError};
use crate::tensor::Tensor;
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};
use onyxia_codegen::{CompiledModel, OpType, Operation, ShaderHandle};
use std::collections::HashMap;
use std::sync::Arc;

/// Executor for a specific model instance.
///
/// Manages GPU resources and executes the compiled model's operations.
pub struct ModelExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    model: CompiledModel,
    dynamic_dimensions: HashMap<String, usize>,
    buffer_manager: BufferManager,
    compute_pipelines: HashMap<ShaderHandle, wgpu::ComputePipeline>,
    bind_group_layouts: HashMap<ShaderHandle, wgpu::BindGroupLayout>,
}

impl ModelExecutor {
    /// Create a new model executor.
    ///
    /// This will allocate GPU buffers and compile shaders.
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: CompiledModel,
        dynamic_dimensions: HashMap<String, usize>,
    ) -> Result<Self> {
        let mut executor = Self {
            device: Arc::clone(&device),
            queue: Arc::clone(&queue),
            buffer_manager: BufferManager::new(device, queue),
            model,
            dynamic_dimensions,
            compute_pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        };
        
        // Allocate buffers for all tensors
        executor.allocate_buffers()?;
        
        // Compile shaders for all operations
        executor.compile_shaders()?;
        
        Ok(executor)
    }
    
    /// Execute the model with given inputs.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_runtime::{Runtime, Tensor};
    /// # #[pollster::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// #     let runtime = Runtime::new().await?;
    /// #     let model = todo!(); // Load compiled model
    /// #     let mut executor = runtime.load_model(model)?;
    /// let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    /// let outputs = executor.run(&[("input", input)])?;
    /// let result = &outputs["output"];
    /// #     Ok(())
    /// # }
    /// ```
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Upload inputs to GPU
        for (name, tensor) in inputs {
            let (tensor_id, _) = self
                .model
                .tensors
                .find_by_name(name)
                .ok_or_else(|| RuntimeError::TensorNotFound(format!("Input '{}' not found", name)))?;
            
            self.buffer_manager.upload(tensor_id, tensor.raw_data()?)?;
        }
        
        // Execute operations
        self.execute_operations()?;
        
        // Download outputs from GPU
        let mut outputs = HashMap::new();
        for output_id in &self.model.outputs {
            let info = self
                .model
                .tensors
                .get(*output_id)
                .ok_or_else(|| RuntimeError::TensorNotFound(format!("Output {} not found", output_id)))?;
            
            // Use pollster to block on async download
            let data = pollster::block_on(self.buffer_manager.download(*output_id))?;
            
            // Extract shape from TensorInfo
            let shape: Vec<usize> = match &info.shape {
                onyxia_onnx::TensorShape::Static(dims) => {
                    dims.iter().map(|&d| d as usize).collect()
                }
                onyxia_onnx::TensorShape::Dynamic(_) | onyxia_onnx::TensorShape::Unknown => {
                    return Err(RuntimeError::TensorError(
                        "Dynamic/unknown shapes not yet supported".to_string(),
                    ))
                }
            };
            
            outputs.insert(
                info.name.clone(),
                Tensor::from_raw(data, &shape, info.dtype),
            );
        }
        
        Ok(outputs)
    }
    
    /// Allocate GPU buffers for all tensors.
    fn allocate_buffers(&mut self) -> Result<()> {
        for (id, info) in self.model.tensors.all().iter().enumerate() {
            self.buffer_manager.allocate_with_dimensions(id, info, &self.dynamic_dimensions)?;
            
            // Upload initializer data if present
            if let Some(ref initializer) = info.initializer {
                self.buffer_manager.upload(id, initializer)?;
            }
        }
        Ok(())
    }
    
    /// Compile WGSL shaders to compute pipelines.
    fn compile_shaders(&mut self) -> Result<()> {
        let mut compiled_shaders = HashMap::new();
        
        for operation in &self.model.operations {
            if compiled_shaders.contains_key(&operation.shader) {
                continue; // Already compiled
            }
            
            let (pipeline, layout) = self.compile_operation_shader(operation)?;
            self.compute_pipelines.insert(operation.shader.clone(), pipeline);
            self.bind_group_layouts.insert(operation.shader.clone(), layout);
            compiled_shaders.insert(operation.shader.clone(), ());
        }
        
        Ok(())
    }
    
    /// Compile a single operation's shader.
    fn compile_operation_shader(
        &self,
        operation: &Operation,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        // Get shader source based on operation type
        let shader_name = self.get_shader_name(&operation.op_type);
        let source = onyxia_codegen::shaders::get_shader_source(&shader_name)
            .ok_or_else(|| RuntimeError::ShaderError(format!("Shader '{}' not found", shader_name)))?;
        
        // Build shader defs for this operation
        let shader_defs = self.build_shader_defs(operation)?;
        
        // Compile with naga_oil
        let mut composer = Composer::default();
        let module = composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: &format!("{}.wgsl", shader_name),
                shader_defs,
                ..Default::default()
            })
            .map_err(|e| RuntimeError::ShaderError(format!("Shader compilation failed: {}", e)))?;
        
        // Create shader module from naga Module - wgpu 28 only accepts WGSL or SPIR-V
        // So we need to convert back to WGSL
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| RuntimeError::ShaderError(format!("Module validation failed: {}", e)))?;
        
        let wgsl = naga::back::wgsl::write_string(
            &module,
            &info,
            naga::back::wgsl::WriterFlags::empty(),
        )
        .map_err(|e| RuntimeError::ShaderError(format!("WGSL generation failed: {}", e)))?;
        
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Shader: {:?}", operation.op_type)),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        
        // Create bind group layout
        // For now, use a simple layout with storage buffers for all tensors
        let bind_group_layout = self.create_bind_group_layout(operation)?;
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Pipeline Layout: {:?}", operation.op_type)),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        
        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Pipeline: {:?}", operation.op_type)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        Ok((pipeline, bind_group_layout))
    }
    
    /// Get shader name for an operation type.
    fn get_shader_name(&self, op_type: &OpType) -> String {
        match op_type {
            OpType::Add => "add".to_string(),
            OpType::Mul => "mul".to_string(),
            OpType::Gelu => "gelu".to_string(),
            OpType::SimplifiedLayerNormalization => "rmsnorm".to_string(),
            OpType::MatMulNBits => "matmul_f32".to_string(), // TODO: Replace with quantized version
            _ => format!("generic_{:?}", op_type).to_lowercase(),
        }
    }
    
    /// Build shader definitions for an operation.
    fn build_shader_defs(&self, operation: &Operation) -> Result<HashMap<String, ShaderDefValue>> {
        let mut defs = HashMap::new();
        
        // Add dynamic dimensions as shader defs
        for (dim_name, dim_value) in &self.dynamic_dimensions {
            // Convert to uppercase for shader constants (e.g., BATCH, SEQUENCE)
            defs.insert(
                dim_name.to_uppercase(),
                ShaderDefValue::UInt(*dim_value as u32),
            );
        }
        
        // Add workgroup size
        defs.insert("WORKGROUP_SIZE".to_string(), ShaderDefValue::UInt(256));
        
        // Add operation-specific defs
        match &operation.op_type {
            OpType::MatMulNBits => {
                defs.insert("TILE_M".to_string(), ShaderDefValue::UInt(16));
                defs.insert("TILE_N".to_string(), ShaderDefValue::UInt(16));
                defs.insert("TILE_K".to_string(), ShaderDefValue::UInt(16));
            }
            _ => {}
        }
        
        Ok(defs)
    }
    
    /// Create bind group layout for an operation.
    fn create_bind_group_layout(&self, operation: &Operation) -> Result<wgpu::BindGroupLayout> {
        // Create entries for all input and output buffers
        let mut entries = Vec::new();
        
        // Add inputs
        for (i, _) in operation.inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        
        // Add outputs
        let output_binding_start = operation.inputs.len();
        for (i, _) in operation.outputs.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (output_binding_start + i) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        
        Ok(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Bind Group Layout: {:?}", operation.op_type)),
            entries: &entries,
        }))
    }
    
    /// Execute all operations in the model.
    fn execute_operations(&mut self) -> Result<()> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Model execution"),
        });
        
        for operation in &self.model.operations {
            self.dispatch_operation(&mut encoder, operation)?;
        }
        
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
        
        Ok(())
    }
    
    /// Dispatch a single operation.
    fn dispatch_operation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        operation: &Operation,
    ) -> Result<()> {
        let pipeline = self.compute_pipelines.get(&operation.shader).ok_or_else(|| {
            RuntimeError::ExecutionError(format!("Pipeline not found for {:?}", operation.op_type))
        })?;
        
        let bind_group_layout = self.bind_group_layouts.get(&operation.shader).ok_or_else(|| {
            RuntimeError::ExecutionError(format!("Bind group layout not found for {:?}", operation.op_type))
        })?;
        
        // Create bind group with actual buffers
        let bind_group = self.create_bind_group(operation, bind_group_layout)?;
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Compute: {:?}", operation.op_type)),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        let [x, y, z] = operation.workgroup_dims;
        compute_pass.dispatch_workgroups(x, y, z);
        
        Ok(())
    }
    
    /// Create a bind group for an operation.
    fn create_bind_group(
        &self,
        operation: &Operation,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::BindGroup> {
        let mut entries = Vec::new();
        
        // Add input buffers
        for (i, &tensor_id) in operation.inputs.iter().enumerate() {
            let buffer = self.buffer_manager.get_buffer(tensor_id).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Buffer for tensor {} not found", tensor_id))
            })?;
            
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        
        // Add output buffers
        let output_binding_start = operation.inputs.len();
        for (i, &tensor_id) in operation.outputs.iter().enumerate() {
            let buffer = self.buffer_manager.get_buffer(tensor_id).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Buffer for tensor {} not found", tensor_id))
            })?;
            
            entries.push(wgpu::BindGroupEntry {
                binding: (output_binding_start + i) as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group: {:?}", operation.op_type)),
            layout,
            entries: &entries,
        }))
    }
    
    /// Get the compiled model.
    pub fn model(&self) -> &CompiledModel {
        &self.model
    }
}
