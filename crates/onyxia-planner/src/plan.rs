//! Execution plan types for runtime execution.
//!
//! This module defines the new type hierarchy that represents an execution plan —
//! the output of the planner crate. These types describe *what* the runtime should
//! do without referencing any `wgpu` types.

use naga;
use onyxia_onnx::{TensorId, TensorInfo};

/// Metadata about the compiled model.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Original model name.
    pub name: String,
    /// Model version.
    pub version: i64,
    /// IR version.
    pub ir_version: i64,
    /// Producer information.
    pub producer: String,
}

/// Registry of all tensors in the model.
#[derive(Debug, Clone)]
pub struct TensorRegistry {
    tensors: Vec<TensorInfo>,
}

impl TensorRegistry {
    /// Create a new tensor registry.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }
    
    /// Add a tensor to the registry.
    pub fn add(&mut self, info: TensorInfo) -> TensorId {
        let id = self.tensors.len();
        self.tensors.push(info);
        id
    }
    
    /// Get tensor info by ID.
    pub fn get(&self, id: TensorId) -> Option<&TensorInfo> {
        self.tensors.get(id)
    }
    
    /// Get all tensors.
    pub fn all(&self) -> &[TensorInfo] {
        &self.tensors
    }
    
    /// Find tensor by name.
    pub fn find_by_name(&self, name: &str) -> Option<(TensorId, &TensorInfo)> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_, info)| info.name == name)
    }
}

impl Default for TensorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Index into `ExecutionPlan.shaders`.
pub type ShaderIndex = usize;

/// The fundamental unit of GPU work. One ONNX operation can emit multiple steps.
#[derive(Debug, Clone)]
pub enum Step {
    /// Dispatch a compute shader.
    Dispatch {
        /// Index into `ExecutionPlan.shaders` — the runtime creates one
        /// pipeline per entry and uses this index to look it up.
        shader_index: ShaderIndex,
        /// Bindings for this dispatch.
        bindings: Vec<BindingDesc>,
        /// Workgroup dimensions [x, y, z].
        workgroups: [u32; 3],
        /// Optional immediate data to pass to the shader (for var<immediate>).
        immediates: Option<Vec<u8>>,
    },
    /// Copy data between buffers.
    CopyBuffer {
        /// Source buffer reference.
        src: BufferRef,
        /// Destination buffer reference.
        dst: BufferRef,
        /// Size in bytes to copy.
        size: u64,
    },
    /// Write CPU data into a buffer.
    WriteBuffer {
        /// Destination buffer reference.
        dst: BufferRef,
        /// Data to write.
        data: Vec<u8>,
    },
}

/// A fully preprocessed shader ready for the runtime to create a pipeline.
///
/// The planner runs naga_oil (WGSL + shader defs → `naga::Module`) at plan time,
/// so the runtime never deals with WGSL text or shader defs.
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// Human-readable label (e.g., "add", "matmul_dequant_pass1").
    pub label: String,
    /// Pre-compiled naga IR module (from naga_oil preprocessing).
    pub module: naga::Module,
    /// Entry point name (default: "main").
    pub entry_point: String,
}

/// References a buffer — either a model tensor or a scratch buffer local to an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferRef {
    /// References a tensor in the model's TensorRegistry.
    Tensor(TensorId),
    /// References a scratch buffer allocated for this operation.
    Scratch(usize),
}

/// Describes one binding in a dispatch step.
#[derive(Debug, Clone)]
pub struct BindingDesc {
    /// Which buffer this binding points to.
    pub buffer: BufferRef,
    /// Whether the buffer is read-only in this dispatch.
    pub read_only: bool,
}

/// Describes a temporary buffer needed within an operation.
#[derive(Debug, Clone)]
pub struct ScratchBufferDesc {
    /// Size in bytes.
    pub size: u64,
    /// Human-readable label.
    pub label: String,
}

/// A planned operation — one (or fused group of) ONNX node(s) mapped to GPU commands.
#[derive(Debug, Clone)]
pub struct PlannedOp {
    /// Human-readable name (e.g., the ONNX node name).
    pub name: String,
    /// ONNX op_type (e.g., "Add", "MatMulNBits").
    pub op_type: String,
    /// Input tensor IDs (from the model's TensorRegistry).
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs.
    pub outputs: Vec<TensorId>,
    /// GPU steps to execute for this operation.
    pub steps: Vec<Step>,
    /// Scratch buffers needed by this operation's steps.
    pub scratch_buffers: Vec<ScratchBufferDesc>,
}

/// The top-level output of the planner.
///
/// **Invariant:** All tensor shapes in `tensors` are `TensorShape::Static` —
/// dynamic dimensions have been resolved using `dynamic_dimensions` at plan time.
/// The runtime can calculate buffer sizes without any dimension lookups.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Ordered list of operations to execute.
    pub operations: Vec<PlannedOp>,
    /// Deduplicated list of compiled shaders.
    /// Steps reference these by `ShaderIndex` (index into this vec).
    /// The runtime creates one `ComputePipeline` per entry.
    pub shaders: Vec<CompiledShader>,
    /// Tensor registry with all tensors in the model.
    /// **All shapes are `TensorShape::Static`** — dynamic dims resolved at plan time.
    pub tensors: TensorRegistry,
    /// Input tensor IDs (in order).
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs (in order).
    pub outputs: Vec<TensorId>,
    /// Model metadata.
    pub metadata: ModelMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{DataType, TensorInfo, TensorKind, TensorShape};

    #[test]
    fn test_construct_execution_plan() {
        // Create a simple execution plan with 1 shader, 1 op, 1 step
        let mut registry = TensorRegistry::new();

        // Add input tensor
        let input_id = registry.add(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        let output_id = registry.add(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        // Create a simple shader (empty module for test)
        let shader = CompiledShader {
            label: "test_add".to_string(),
            module: naga::Module::default(),
            entry_point: "main".to_string(),
        };

        // Create a step referencing shader_index 0
        let step = Step::Dispatch {
            shader_index: 0,
            bindings: vec![
                BindingDesc {
                    buffer: BufferRef::Tensor(input_id),
                    read_only: true,
                },
                BindingDesc {
                    buffer: BufferRef::Tensor(output_id),
                    read_only: false,
                },
            ],
            workgroups: [1, 1, 1],
            immediates: None,
        };

        // Create operation
        let op = PlannedOp {
            name: "test_op".to_string(),
            op_type: "Add".to_string(),
            inputs: vec![input_id],
            outputs: vec![output_id],
            steps: vec![step],
            scratch_buffers: Vec::new(),
        };

        // Create execution plan
        let plan = ExecutionPlan {
            operations: vec![op],
            shaders: vec![shader],
            tensors: registry,
            inputs: vec![input_id],
            outputs: vec![output_id],
            metadata: ModelMetadata {
                name: "test_model".to_string(),
                version: 1,
                ir_version: 9,
                producer: "test".to_string(),
            },
        };

        // Verify structure
        assert_eq!(plan.operations.len(), 1);
        assert_eq!(plan.shaders.len(), 1);
        assert_eq!(plan.inputs.len(), 1);
        assert_eq!(plan.outputs.len(), 1);
        assert_eq!(plan.operations[0].name, "test_op");
        assert_eq!(plan.operations[0].op_type, "Add");
        assert_eq!(plan.operations[0].steps.len(), 1);
        assert_eq!(plan.shaders[0].label, "test_add");
        assert_eq!(plan.shaders[0].entry_point, "main");

        // Verify step references shader_index 0
        match &plan.operations[0].steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                ..
            } => {
                assert_eq!(*shader_index, 0);
                assert_eq!(bindings.len(), 2);
                assert_eq!(*workgroups, [1, 1, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }

    #[test]
    fn test_buffer_ref_types() {
        let tensor_ref = BufferRef::Tensor(42);
        let scratch_ref = BufferRef::Scratch(1);

        assert_eq!(tensor_ref, BufferRef::Tensor(42));
        assert_eq!(scratch_ref, BufferRef::Scratch(1));
        assert_ne!(tensor_ref, scratch_ref);
    }

    #[test]
    fn test_step_variants() {
        // Test Dispatch step
        let dispatch = Step::Dispatch {
            shader_index: 0,
            bindings: vec![],
            workgroups: [64, 1, 1],
            immediates: None,
        };
        match dispatch {
            Step::Dispatch { shader_index, .. } => assert_eq!(shader_index, 0),
            _ => panic!("Expected Dispatch"),
        }

        // Test CopyBuffer step
        let copy = Step::CopyBuffer {
            src: BufferRef::Tensor(0),
            dst: BufferRef::Scratch(0),
            size: 1024,
        };
        match copy {
            Step::CopyBuffer { size, .. } => assert_eq!(size, 1024),
            _ => panic!("Expected CopyBuffer"),
        }

        // Test WriteBuffer step
        let write = Step::WriteBuffer {
            dst: BufferRef::Tensor(1),
            data: vec![1, 2, 3, 4],
        };
        match write {
            Step::WriteBuffer { data, .. } => assert_eq!(data.len(), 4),
            _ => panic!("Expected WriteBuffer"),
        }
    }

    #[test]
    fn test_scratch_buffer_desc() {
        let scratch = ScratchBufferDesc {
            size: 2048,
            label: "temp_buffer".to_string(),
        };

        assert_eq!(scratch.size, 2048);
        assert_eq!(scratch.label, "temp_buffer");
    }
}
