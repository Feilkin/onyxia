//! Kernel trait and registry for extensible operation mapping.
//!
//! This module defines the core extensibility point: an `OpKernel` trait that
//! maps ONNX nodes to GPU execution steps, and a `KernelRegistry` that maps
//! op_type strings to kernel implementations.

use crate::error::{CodegenError, Result};
use crate::plan::{BufferRef, CompiledShader, ScratchBufferDesc, ShaderIndex, Step};
use crate::shaders::ShaderDefs;
use naga_oil::compose::{Composer, NagaModuleDescriptor};
use onyxia_onnx::{Dimension, Graph, Node, TensorId, TensorInfo, TensorShape};
use std::collections::HashMap;

/// Trait for mapping an ONNX node to GPU execution steps.
///
/// This is the core extensibility point for adding new operations. Each
/// implementation handles one or more ONNX op_types and generates the
/// appropriate GPU commands (shaders, buffer bindings, etc.).
pub trait OpKernel: Send + Sync {
    /// Human-readable kernel name.
    fn name(&self) -> &str;

    /// Plan the GPU steps needed to execute this operation.
    ///
    /// Called once at plan time (not per-inference). All shader defs are
    /// resolvable because dynamic_dimensions are known.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Planning context with access to tensors, dimensions, and shader compilation
    ///
    /// # Returns
    ///
    /// A list of GPU steps to execute for this operation.
    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>>;
}

/// Context provided to OpKernel::plan() during operation planning.
///
/// Gives kernels access to:
/// - Node and graph metadata
/// - Resolved tensor IDs and shapes
/// - Dynamic dimension values (user-provided at plan time)
/// - Shader compilation (via naga_oil)
/// - Scratch buffer allocation
pub struct PlanContext<'a> {
    /// The ONNX node being planned.
    pub node: &'a Node,

    /// The full graph (for looking up tensor info).
    pub graph: &'a Graph,

    /// Resolved input tensor IDs.
    pub input_ids: &'a [TensorId],

    /// Resolved output tensor IDs.
    pub output_ids: &'a [TensorId],

    /// User-provided concrete values for dynamic dimensions.
    /// Maps dimension names like "batch", "sequence" to actual sizes.
    pub dynamic_dimensions: &'a HashMap<String, usize>,

    /// Accumulated scratch buffers for this operation.
    pub scratch_buffers: Vec<ScratchBufferDesc>,

    /// Shared deduplicated shader list (builds up during planning).
    pub shaders: &'a mut Vec<CompiledShader>,

    /// naga_oil Composer for shader compilation.
    pub composer: Composer,
}

impl<'a> PlanContext<'a> {
    /// Create a new PlanContext.
    ///
    /// This is the production constructor used by compile_to_plan().
    pub fn new(
        node: &'a Node,
        graph: &'a Graph,
        input_ids: &'a [TensorId],
        output_ids: &'a [TensorId],
        dynamic_dimensions: &'a HashMap<String, usize>,
        shaders: &'a mut Vec<CompiledShader>,
    ) -> Self {
        Self {
            node,
            graph,
            input_ids,
            output_ids,
            dynamic_dimensions,
            scratch_buffers: Vec::new(),
            shaders,
            composer: Composer::default().with_capabilities(naga::valid::Capabilities::all()),
        }
    }

    /// Get a BufferRef for the nth input tensor.
    pub fn input(&self, n: usize) -> BufferRef {
        BufferRef::Tensor(self.input_ids[n])
    }

    /// Get a BufferRef for the nth output tensor.
    pub fn output(&self, n: usize) -> BufferRef {
        BufferRef::Tensor(self.output_ids[n])
    }

    /// Get TensorInfo for the nth input tensor.
    pub fn input_info(&self, n: usize) -> Result<&TensorInfo> {
        let id = self.input_ids[n];
        self.graph
            .tensor(id)
            .map_err(|e| CodegenError::OnnxError(e))
    }

    /// Get TensorInfo for the nth output tensor.
    pub fn output_info(&self, n: usize) -> Result<&TensorInfo> {
        let id = self.output_ids[n];
        self.graph
            .tensor(id)
            .map_err(|e| CodegenError::OnnxError(e))
    }

    /// Resolve a tensor shape to concrete dimensions.
    ///
    /// Named dimensions are looked up in dynamic_dimensions.
    /// Static dimensions are returned as-is.
    ///
    /// # Arguments
    ///
    /// * `shape` - The tensor shape to resolve
    ///
    /// # Returns
    ///
    /// A vector of concrete dimension sizes, or an error if a named dimension
    /// is not found in dynamic_dimensions.
    pub fn resolve_shape(&self, shape: &TensorShape) -> Result<Vec<usize>> {
        match shape {
            TensorShape::Static(dims) => Ok(dims.clone()),
            TensorShape::Dynamic(dims) => {
                let mut resolved = Vec::with_capacity(dims.len());
                for dim in dims {
                    match dim {
                        Dimension::Static(size) => resolved.push(*size),
                        Dimension::Named(name) => {
                            let size = self.dynamic_dimensions.get(name).ok_or_else(|| {
                                CodegenError::InvalidShape(format!(
                                    "Dynamic dimension '{}' not provided",
                                    name
                                ))
                            })?;
                            resolved.push(*size);
                        }
                    }
                }
                Ok(resolved)
            }
            TensorShape::Unknown => Err(CodegenError::InvalidShape(
                "Cannot resolve unknown shape".to_string(),
            )),
        }
    }

    /// Compile a WGSL source with shader defs into a CompiledShader.
    ///
    /// Returns a `ShaderIndex` into the plan's deduplicated shaders list.
    /// If the same label+defs combo was already compiled, returns the existing index.
    ///
    /// # Arguments
    ///
    /// * `label` - Human-readable shader label (e.g., "add", "matmul_q4")
    /// * `wgsl` - WGSL source code
    /// * `defs` - Shader definitions for runtime specialization
    ///
    /// # Returns
    ///
    /// Index into the shaders list where this compiled shader is stored.
    pub fn compile_shader(
        &mut self,
        label: &str,
        wgsl: &str,
        defs: ShaderDefs,
    ) -> Result<ShaderIndex> {
        // Check if we already have this shader (simple label-based deduplication)
        // In a real implementation, we'd compare label+defs for proper deduplication
        for (idx, shader) in self.shaders.iter().enumerate() {
            if shader.label == label {
                return Ok(idx);
            }
        }

        // Compile the shader using naga_oil
        let module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source: wgsl,
                file_path: label,
                shader_defs: defs,
                ..Default::default()
            })
            .map_err(|e| CodegenError::ShaderError(format!("Failed to compile shader: {}", e)))?;

        // Add to shaders list
        let idx = self.shaders.len();
        self.shaders.push(CompiledShader {
            label: label.to_string(),
            module,
            entry_point: "main".to_string(),
        });

        Ok(idx)
    }

    /// Allocate a scratch buffer and return a BufferRef to it.
    ///
    /// Scratch buffers are temporary storage used during operation execution.
    /// They are local to this operation and not visible to other operations.
    ///
    /// # Arguments
    ///
    /// * `desc` - Description of the scratch buffer to allocate
    ///
    /// # Returns
    ///
    /// A BufferRef::Scratch pointing to the allocated buffer.
    pub fn alloc_scratch(&mut self, desc: ScratchBufferDesc) -> BufferRef {
        let idx = self.scratch_buffers.len();
        self.scratch_buffers.push(desc);
        BufferRef::Scratch(idx)
    }
}

#[cfg(test)]
impl<'a> PlanContext<'a> {
    /// Build a PlanContext for unit testing.
    ///
    /// Takes a Node, a Graph (which must contain the referenced tensors),
    /// resolved input/output IDs, and dynamic dimensions.
    /// Creates an empty Composer and shader list internally.
    pub fn for_test(
        node: &'a Node,
        graph: &'a Graph,
        input_ids: &'a [TensorId],
        output_ids: &'a [TensorId],
        dynamic_dimensions: &'a HashMap<String, usize>,
        shaders: &'a mut Vec<CompiledShader>,
    ) -> Self {
        Self::new(
            node,
            graph,
            input_ids,
            output_ids,
            dynamic_dimensions,
            shaders,
        )
    }
}

/// Registry of OpKernel implementations.
///
/// Maps ONNX op_type strings (e.g., "Add", "MatMul") to kernel implementations
/// that know how to generate GPU code for those operations.
pub struct KernelRegistry {
    kernels: HashMap<String, Box<dyn OpKernel>>,
}

impl KernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Create a registry pre-loaded with all built-in kernels.
    ///
    /// Registers all implemented kernels (Add, etc.)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register("Add", Box::new(crate::kernels::AddKernel));
        registry
    }

    /// Register a kernel for an op_type string.
    ///
    /// # Arguments
    ///
    /// * `op_type` - ONNX operation type (e.g., "Add", "Mul", "MatMul")
    /// * `kernel` - Boxed kernel implementation
    pub fn register(&mut self, op_type: impl Into<String>, kernel: Box<dyn OpKernel>) {
        self.kernels.insert(op_type.into(), kernel);
    }

    /// Look up a kernel by ONNX op_type.
    ///
    /// Returns None if no kernel is registered for this op_type.
    pub fn get(&self, op_type: &str) -> Option<&dyn OpKernel> {
        self.kernels.get(op_type).map(|k| k.as_ref())
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal WGSL shader for testing.
///
/// This is a trivial pass-through shader that copies input to output.
/// Used in unit tests to verify kernel infrastructure without complex shader logic.
#[cfg(test)]
pub(crate) const TRIVIAL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output[id.x] = input[id.x];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::BindingDesc;
    use onyxia_onnx::{DataType, TensorKind};

    /// Dummy kernel for testing that uses TRIVIAL_WGSL.
    struct DummyKernel;

    impl OpKernel for DummyKernel {
        fn name(&self) -> &str {
            "dummy"
        }

        fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
            // Compile the trivial shader
            let shader_index = ctx.compile_shader("trivial", TRIVIAL_WGSL, ShaderDefs::new())?;

            // Create a simple dispatch step
            let step = Step::Dispatch {
                shader_index,
                bindings: vec![
                    BindingDesc {
                        buffer: ctx.input(0),
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: ctx.output(0),
                        read_only: false,
                    },
                ],
                workgroups: [1, 1, 1],
            };

            Ok(vec![step])
        }
    }

    /// Another dummy kernel for multi-kernel tests.
    struct AnotherKernel;

    impl OpKernel for AnotherKernel {
        fn name(&self) -> &str {
            "another"
        }

        fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
            let shader_index = ctx.compile_shader("another", TRIVIAL_WGSL, ShaderDefs::new())?;

            let step = Step::Dispatch {
                shader_index,
                bindings: vec![
                    BindingDesc {
                        buffer: ctx.input(0),
                        read_only: true,
                    },
                    BindingDesc {
                        buffer: ctx.output(0),
                        read_only: false,
                    },
                ],
                workgroups: [2, 1, 1],
            };

            Ok(vec![step])
        }
    }

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add input tensor
        graph.add_tensor(TensorInfo {
            name: "input".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        // Add output tensor
        graph.add_tensor(TensorInfo {
            name: "output".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![1, 4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        graph
    }

    #[test]
    fn test_register_and_plan_dummy_kernel() {
        let mut registry = KernelRegistry::new();
        registry.register("Dummy", Box::new(DummyKernel));

        // Verify kernel was registered
        let kernel = registry.get("Dummy").expect("Kernel should be registered");
        assert_eq!(kernel.name(), "dummy");

        // Create test graph and node
        let graph = create_test_graph();
        let mut node = Node::new("Dummy");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        // Plan the operation
        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let steps = kernel.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got a dispatch step
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
            } => {
                assert_eq!(*shader_index, 0);
                assert_eq!(bindings.len(), 2);
                assert_eq!(*workgroups, [1, 1, 1]);

                // Verify bindings point to correct tensors
                assert_eq!(bindings[0].buffer, BufferRef::Tensor(0));
                assert!(bindings[0].read_only);
                assert_eq!(bindings[1].buffer, BufferRef::Tensor(1));
                assert!(!bindings[1].read_only);
            }
            _ => panic!("Expected Dispatch step"),
        }

        // Verify shader was compiled
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].label, "trivial");
        assert_eq!(shaders[0].entry_point, "main");
    }

    #[test]
    fn test_register_multiple_kernels() {
        let mut registry = KernelRegistry::new();

        registry.register("Dummy", Box::new(DummyKernel));
        registry.register("Another", Box::new(AnotherKernel));

        // Look up both kernels
        let dummy = registry.get("Dummy").expect("Dummy kernel should exist");
        let another = registry
            .get("Another")
            .expect("Another kernel should exist");

        assert_eq!(dummy.name(), "dummy");
        assert_eq!(another.name(), "another");

        // Create test graph
        let graph = create_test_graph();
        let mut node = Node::new("Dummy");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        // Plan with dummy kernel
        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );
        let steps = dummy.plan(&mut ctx).expect("Dummy planning should succeed");
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Dispatch { workgroups, .. } => assert_eq!(*workgroups, [1, 1, 1]),
            _ => panic!("Expected Dispatch"),
        }

        // Plan with another kernel
        let mut node2 = Node::new("Another");
        node2.inputs = vec!["input".to_string()];
        node2.outputs = vec!["output".to_string()];

        let mut ctx2 = PlanContext::for_test(
            &node2,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );
        let steps2 = another
            .plan(&mut ctx2)
            .expect("Another planning should succeed");
        assert_eq!(steps2.len(), 1);
        match &steps2[0] {
            Step::Dispatch { workgroups, .. } => assert_eq!(*workgroups, [2, 1, 1]),
            _ => panic!("Expected Dispatch"),
        }
    }

    #[test]
    fn test_with_defaults() {
        let registry = KernelRegistry::with_defaults();
        // Verify AddKernel is registered
        assert!(registry.get("Add").is_some());
        let add_kernel = registry.get("Add").unwrap();
        assert_eq!(add_kernel.name(), "Add");
    }

    #[test]
    fn test_plan_context_helpers() {
        let graph = create_test_graph();
        let node = Node::new("Test");

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        // Test input/output accessors
        assert_eq!(ctx.input(0), BufferRef::Tensor(0));
        assert_eq!(ctx.output(0), BufferRef::Tensor(1));

        // Test input_info/output_info
        let input_info = ctx.input_info(0).expect("Should get input info");
        assert_eq!(input_info.name, "input");
        assert_eq!(input_info.dtype, DataType::F32);

        let output_info = ctx.output_info(0).expect("Should get output info");
        assert_eq!(output_info.name, "output");
        assert_eq!(output_info.dtype, DataType::F32);
    }

    #[test]
    fn test_resolve_shape_static() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let shape = TensorShape::Static(vec![1, 2, 3]);
        let resolved = ctx
            .resolve_shape(&shape)
            .expect("Should resolve static shape");
        assert_eq!(resolved, vec![1, 2, 3]);
    }

    #[test]
    fn test_resolve_shape_dynamic() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];

        let mut dynamic_dimensions = HashMap::new();
        dynamic_dimensions.insert("batch".to_string(), 4);
        dynamic_dimensions.insert("seq".to_string(), 128);

        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let shape = TensorShape::Dynamic(vec![
            Dimension::Named("batch".to_string()),
            Dimension::Static(16),
            Dimension::Named("seq".to_string()),
        ]);

        let resolved = ctx
            .resolve_shape(&shape)
            .expect("Should resolve dynamic shape");
        assert_eq!(resolved, vec![4, 16, 128]);
    }

    #[test]
    fn test_resolve_shape_missing_dynamic() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new(); // Empty - missing "batch"
        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let shape = TensorShape::Dynamic(vec![Dimension::Named("batch".to_string())]);

        let result = ctx.resolve_shape(&shape);
        assert!(result.is_err());
        match result {
            Err(CodegenError::InvalidShape(msg)) => {
                assert!(msg.contains("Dynamic dimension 'batch' not provided"));
            }
            _ => panic!("Expected InvalidShape error"),
        }
    }

    #[test]
    fn test_resolve_shape_unknown() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let shape = TensorShape::Unknown;
        let result = ctx.resolve_shape(&shape);
        assert!(result.is_err());
        match result {
            Err(CodegenError::InvalidShape(msg)) => {
                assert!(msg.contains("Cannot resolve unknown shape"));
            }
            _ => panic!("Expected InvalidShape error"),
        }
    }

    #[test]
    fn test_alloc_scratch() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        // Allocate first scratch buffer
        let scratch1 = ctx.alloc_scratch(ScratchBufferDesc {
            size: 1024,
            label: "temp1".to_string(),
        });
        assert_eq!(scratch1, BufferRef::Scratch(0));

        // Allocate second scratch buffer
        let scratch2 = ctx.alloc_scratch(ScratchBufferDesc {
            size: 2048,
            label: "temp2".to_string(),
        });
        assert_eq!(scratch2, BufferRef::Scratch(1));

        // Verify buffers were added
        assert_eq!(ctx.scratch_buffers.len(), 2);
        assert_eq!(ctx.scratch_buffers[0].size, 1024);
        assert_eq!(ctx.scratch_buffers[0].label, "temp1");
        assert_eq!(ctx.scratch_buffers[1].size, 2048);
        assert_eq!(ctx.scratch_buffers[1].label, "temp2");
    }

    #[test]
    fn test_compile_shader_deduplication() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions = HashMap::new();
        let mut shaders = Vec::new();

        let mut ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        // Compile same shader twice
        let idx1 = ctx
            .compile_shader("test_shader", TRIVIAL_WGSL, ShaderDefs::new())
            .expect("First compile should succeed");
        let idx2 = ctx
            .compile_shader("test_shader", TRIVIAL_WGSL, ShaderDefs::new())
            .expect("Second compile should succeed");

        // Should return same index (deduplicated)
        assert_eq!(idx1, idx2);
        assert_eq!(shaders.len(), 1);
    }
}
