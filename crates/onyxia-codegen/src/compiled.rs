//! Compiled model representation.
//!
//! This module defines the output of the compilation process: a `CompiledModel`
//! that contains WGSL shaders, execution metadata, and tensor layouts ready for
//! execution by the runtime.

use onyxia_onnx::{TensorId, TensorInfo};

/// A compiled model ready for execution.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// Ordered list of operations to execute.
    pub operations: Vec<Operation>,
    
    /// Tensor registry with all tensors in the model.
    pub tensors: TensorRegistry,
    
    /// Input tensor IDs (in order).
    pub inputs: Vec<TensorId>,
    
    /// Output tensor IDs (in order).
    pub outputs: Vec<TensorId>,
    
    /// Model metadata.
    pub metadata: ModelMetadata,
}

impl CompiledModel {
    /// Get the shader code for an operation.
    pub fn get_shader_code(&self, _handle: &ShaderHandle) -> Option<&ShaderCode> {
        // TODO: Store shader code in the model
        // For now, return None - shaders will be generated on demand
        None
    }
}

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

/// A single operation in the execution plan.
#[derive(Debug, Clone)]
pub struct Operation {
    /// Operation type.
    pub op_type: OpType,
    
    /// Handle to the shader for this operation.
    pub shader: ShaderHandle,
    
    /// Input tensor IDs.
    pub inputs: Vec<TensorId>,
    
    /// Output tensor IDs.
    pub outputs: Vec<TensorId>,
    
    /// Operation parameters.
    pub params: OpParams,
    
    /// Workgroup dispatch dimensions.
    pub workgroup_dims: [u32; 3],
}

/// Operation type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Matrix multiplication (quantized).
    MatMulNBits,
    
    /// Simplified layer normalization (RMSNorm).
    SimplifiedLayerNormalization,
    
    /// Group query attention.
    GroupQueryAttention,
    
    /// Rotary positional embedding.
    RotaryEmbedding,
    
    /// Element-wise addition.
    Add,
    
    /// Element-wise multiplication.
    Mul,
    
    /// GELU activation.
    Gelu,
    
    /// Reshape operation.
    Reshape,
    
    /// Gather operation.
    Gather,
    
    /// Constant (no-op, data already in weights).
    Constant,
    
    /// Generic operation (for unsupported ops).
    Generic(String),
}

/// Handle to a compiled shader.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderHandle {
    /// Unique identifier for the shader.
    pub id: usize,
}

/// Operation-specific parameters.
#[derive(Debug, Clone)]
pub enum OpParams {
    /// Parameters for MatMulNBits.
    MatMulNBits {
        /// Number of quantization bits (typically 4).
        quant_bits: u32,
        
        /// Block size for quantization.
        block_size: usize,
    },
    
    /// Parameters for LayerNorm.
    LayerNorm {
        /// Epsilon for numerical stability.
        epsilon: f32,
        
        /// Axis to normalize over.
        axis: i64,
    },
    
    /// Parameters for Reshape.
    Reshape {
        /// Target shape.
        shape: Vec<i64>,
    },
    
    /// Parameters for Gather.
    Gather {
        /// Axis to gather along.
        axis: i64,
    },
    
    /// No parameters.
    None,
}

/// WGSL shader code.
#[derive(Debug, Clone)]
pub struct ShaderCode {
    /// WGSL source code.
    pub wgsl: String,
    
    /// Entry point function name.
    pub entry_point: String,
    
    /// Bind group layout information.
    pub bind_group_layout: BindGroupLayout,
}

/// Bind group layout for a shader.
#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    /// Bindings in order.
    pub bindings: Vec<Binding>,
}

/// A single binding in a bind group.
#[derive(Debug, Clone)]
pub struct Binding {
    /// Binding index.
    pub index: u32,
    
    /// Binding type.
    pub binding_type: BindingType,
    
    /// Visibility (compute shader).
    pub visibility: ShaderStage,
}

/// Type of a binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingType {
    /// Storage buffer (read-only).
    StorageRead,
    
    /// Storage buffer (read-write).
    StorageReadWrite,
    
    /// Uniform buffer.
    Uniform,
}

/// Shader stage visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_onnx::{DataType, TensorKind, TensorShape};
    
    #[test]
    fn test_tensor_registry() {
        let mut registry = TensorRegistry::new();
        
        let tensor = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 224, 224, 3]),
            kind: TensorKind::Input,
            initializer: None,
        };
        
        let id = registry.add(tensor);
        assert_eq!(id, 0);
        assert!(registry.get(0).is_some());
        assert_eq!(registry.get(0).unwrap().name, "input");
    }
    
    #[test]
    fn test_compiled_model() {
        let mut registry = TensorRegistry::new();
        
        let input = TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Input,
            initializer: None,
        };
        let input_id = registry.add(input);
        
        let output = TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 10]),
            kind: TensorKind::Output,
            initializer: None,
        };
        let output_id = registry.add(output);
        
        let model = CompiledModel {
            operations: vec![],
            tensors: registry,
            inputs: vec![input_id],
            outputs: vec![output_id],
            metadata: ModelMetadata::default(),
        };
        
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
    }
}
