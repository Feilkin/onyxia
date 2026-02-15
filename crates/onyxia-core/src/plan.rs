//! Execution plan types for GPU execution.
//!
//! Defines the output of the compilation pipeline: a compiled model ready
//! to execute on the GPU via the runtime.

use crate::ir::IrTensorId;
use crate::symbolic_expr::SymbolicExpr;
use crate::types::{DataType, TensorShape};

/// Index into the shader array.
pub type ShaderIndex = usize;

/// A compiled ONNX model ready for GPU execution.
///
/// This is the final output of the compiler pipeline, containing all the
/// information needed by the runtime to execute the model on the GPU.
#[derive(Debug)]
pub struct CompiledModel {
    /// Planned operations in execution order.
    pub operations: Vec<PlannedOp>,

    /// Compiled shaders (naga modules).
    pub shaders: Vec<CompiledShader>,

    /// Tensor registry (shapes, types, names).
    pub tensors: TensorRegistry,

    /// Input tensor IDs.
    pub inputs: Vec<IrTensorId>,

    /// Output tensor IDs.
    pub outputs: Vec<IrTensorId>,

    /// Symbolic dimension bindings (for runtime resolution).
    ///
    /// Maps symbolic expressions to their locations in shader immediates.
    /// The runtime evaluates these expressions with user-provided dimension
    /// values and patches the immediates before execution.
    pub symbolic_bindings: Vec<SymbolicBinding>,

    /// Model metadata (name, version, etc.).
    pub metadata: ModelMetadata,
}

/// A single planned operation.
///
/// This is the result of calling `Operator::plan()` for a node. It contains
/// the execution steps and any scratch buffers needed for this operation.
#[derive(Debug, Clone)]
pub struct PlannedOp {
    /// Operation name (for debugging/profiling).
    pub name: String,

    /// Execution steps for this operation.
    pub steps: Vec<Step>,

    /// Scratch buffers allocated for this operation.
    pub scratch_buffers: Vec<ScratchBufferDesc>,
}

impl PlannedOp {
    /// Create a new planned operation.
    pub fn new(name: String) -> Self {
        Self {
            name,
            steps: Vec::new(),
            scratch_buffers: Vec::new(),
        }
    }
}

/// An execution step (shader dispatch, buffer copy, or write).
#[derive(Debug, Clone)]
pub enum Step {
    /// Dispatch a compute shader.
    Dispatch {
        /// Index of the shader in the shader array.
        shader_index: ShaderIndex,

        /// Buffer bindings for the shader.
        bindings: Vec<BindingDesc>,

        /// Workgroup dimensions (x, y, z).
        workgroups: [u32; 3],

        /// Immediate data to patch into shader uniforms (optional).
        ///
        /// For dynamic dimensions, the runtime evaluates symbolic expressions
        /// and updates these bytes before dispatch.
        immediates: Option<Vec<u8>>,
    },

    /// Copy data between buffers.
    CopyBuffer {
        /// Source buffer.
        src: BufferRef,

        /// Source offset in bytes.
        src_offset: u64,

        /// Destination buffer.
        dst: BufferRef,

        /// Destination offset in bytes.
        dst_offset: u64,

        /// Number of bytes to copy.
        size: u64,
    },

    /// Write immediate data to a buffer.
    WriteBuffer {
        /// Destination buffer.
        dst: BufferRef,

        /// Data to write.
        data: Vec<u8>,
    },
}

/// A compiled shader (naga module + metadata).
#[derive(Debug)]
pub struct CompiledShader {
    /// Shader label (e.g., "matmul_f32", "rms_norm").
    pub label: String,

    /// Compiled naga module.
    pub module: naga::Module,

    /// Entry point name (usually "main").
    pub entry_point: String,
}

/// Reference to a buffer (either a tensor or a scratch buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferRef {
    /// Reference to a tensor buffer.
    Tensor(IrTensorId),

    /// Reference to a scratch buffer (index into scratch_buffers vec).
    Scratch(usize),
}

/// Shader buffer binding descriptor.
#[derive(Debug, Clone)]
pub struct BindingDesc {
    /// The buffer to bind.
    pub buffer: BufferRef,

    /// Whether this is a read-only binding.
    pub read_only: bool,
}

/// Scratch buffer descriptor.
///
/// Scratch buffers are temporary buffers allocated for intermediate results
/// that don't correspond to any ONNX tensor (e.g., transpose workspaces).
#[derive(Debug, Clone)]
pub struct ScratchBufferDesc {
    /// Size in bytes.
    pub size: u64,

    /// Label for debugging.
    pub label: String,
}

/// Symbolic dimension binding (for runtime dimension resolution).
#[derive(Debug, Clone)]
pub struct SymbolicBinding {
    /// The shader that contains this binding.
    pub shader_index: ShaderIndex,

    /// Offset into the shader's immediates buffer.
    pub immediate_offset: usize,

    /// The symbolic expression to evaluate.
    pub expr: SymbolicExpr,
}

/// Tensor registry (all tensors in the model).
#[derive(Debug, Clone, Default)]
pub struct TensorRegistry {
    /// Tensor metadata (indexed by IrTensorId).
    pub tensors: Vec<TensorMetadata>,
}

impl TensorRegistry {
    /// Create a new empty tensor registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tensor to the registry.
    pub fn add(&mut self, metadata: TensorMetadata) -> IrTensorId {
        let id = IrTensorId::new(self.tensors.len());
        self.tensors.push(metadata);
        id
    }

    /// Get tensor metadata by ID.
    pub fn get(&self, id: IrTensorId) -> Option<&TensorMetadata> {
        self.tensors.get(id.index())
    }

    /// Get all tensor metadata.
    pub fn all(&self) -> &[TensorMetadata] {
        &self.tensors
    }
}

/// Metadata for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name.
    pub name: String,

    /// Data type.
    pub dtype: DataType,

    /// Shape (must be static at planning time).
    pub shape: TensorShape,

    /// Size in bytes.
    pub size_bytes: usize,
}

impl TensorMetadata {
    /// Create new tensor metadata.
    pub fn new(name: String, dtype: DataType, shape: TensorShape) -> Self {
        let size_bytes = if let Some(dims) = shape.as_static() {
            let numel: usize = dims.iter().product();
            numel * dtype.size()
        } else {
            0 // Should not happen at planning time
        };

        Self {
            name,
            dtype,
            shape,
            size_bytes,
        }
    }
}

/// Model metadata (name, version, producer, etc.).
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Model name.
    pub name: String,

    /// IR version.
    pub ir_version: i64,

    /// Producer name.
    pub producer_name: String,

    /// Model version.
    pub model_version: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IrTensorId;

    #[test]
    fn test_buffer_ref_equality() {
        let tensor_ref = BufferRef::Tensor(IrTensorId::new(0));
        let scratch_ref = BufferRef::Scratch(0);

        assert_ne!(tensor_ref, scratch_ref);
        assert_eq!(tensor_ref, BufferRef::Tensor(IrTensorId::new(0)));
    }

    #[test]
    fn test_planned_op_new() {
        let op = PlannedOp::new("Add".to_string());
        assert_eq!(op.name, "Add");
        assert!(op.steps.is_empty());
        assert!(op.scratch_buffers.is_empty());
    }

    #[test]
    fn test_tensor_registry() {
        let mut registry = TensorRegistry::new();

        let metadata = TensorMetadata::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        );

        let id = registry.add(metadata);
        assert_eq!(id.index(), 0);

        let retrieved = registry.get(id).unwrap();
        assert_eq!(retrieved.name, "input");
        assert_eq!(retrieved.dtype, DataType::F32);
        assert_eq!(retrieved.size_bytes, 6 * 4); // 6 elements * 4 bytes
    }

    #[test]
    fn test_tensor_metadata_size_calculation() {
        let metadata = TensorMetadata::new(
            "test".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3, 4]),
        );

        assert_eq!(metadata.size_bytes, 24 * 4); // 24 elements * 4 bytes
    }

    #[test]
    fn test_step_dispatch() {
        let step = Step::Dispatch {
            shader_index: 0,
            bindings: vec![BindingDesc {
                buffer: BufferRef::Tensor(IrTensorId::new(0)),
                read_only: true,
            }],
            workgroups: [8, 8, 1],
            immediates: None,
        };

        match step {
            Step::Dispatch { workgroups, .. } => {
                assert_eq!(workgroups, [8, 8, 1]);
            }
            _ => panic!("Expected Dispatch step"),
        }
    }
}
