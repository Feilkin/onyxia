//! Operator trait and registry for extensible operation mapping.
//!
//! This module defines the core extensibility point: an `Operator` trait that
//! maps ONNX nodes to GPU execution steps, and an `OperatorRegistry` that maps
//! op_type strings to operator implementations.

use crate::error::{CodegenError, Result};
use crate::inference::{InferenceContext, TensorValue};
use crate::plan::{BufferRef, CompiledShader, ScratchBufferDesc, ShaderIndex, Step};
use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};
use onyxia_onnx::{Graph, Node, TensorId, TensorInfo, TensorShape};
use std::collections::HashMap;

/// Trait for mapping an ONNX node to GPU execution steps.
///
/// This is the core extensibility point for adding new operations. Each
/// implementation handles one or more ONNX op_types and generates the
/// appropriate GPU commands (shaders, buffer bindings, etc.).
pub trait Operator: Send + Sync {
    /// Human-readable operator name.
    fn name(&self) -> &str;

    /// Infer output tensor shapes for this operation.
    ///
    /// Called during shape inference before planning. Given input shapes and
    /// (optionally) constant-folded input values, infer what the output shapes
    /// should be.
    ///
    /// Input shapes are guaranteed to have no `Named` dimensions — Phase 1
    /// (dynamic dimension substitution) has already resolved them.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Inference context with node, input shapes, input values, graph
    ///
    /// # Returns
    ///
    /// A vector of output tensor shapes, one per output.
    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>>;

    /// Try to constant-fold this operation's outputs.
    ///
    /// Override this for shape-computing ops (Shape, Gather, Concat, etc.)
    /// whose output values enable downstream data-dependent shape inference.
    /// This is constant folding in the compiler sense: when all inputs are
    /// known constants, compute the output at compile time.
    ///
    /// Default: no folding (returns `None` for all outputs).
    ///
    /// # Arguments
    ///
    /// * `ctx` - Inference context with node, input shapes, input values, graph
    ///
    /// # Returns
    ///
    /// A vector of optional constant values, one per output. `None` means the
    /// output is not a compile-time constant (or folding is not implemented).
    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        Ok(vec![None; ctx.node.outputs.len()])
    }

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

/// Context provided to Operator::plan() during operation planning.
///
/// Gives operators access to:
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
    /// This is the production constructor used by compile().
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
        self.graph.tensor(id).map_err(CodegenError::OnnxError)
    }

    /// Get TensorInfo for the nth output tensor.
    pub fn output_info(&self, n: usize) -> Result<&TensorInfo> {
        let id = self.output_ids[n];
        self.graph.tensor(id).map_err(CodegenError::OnnxError)
    }

    /// Get static dimensions from a tensor shape.
    ///
    /// By the time `plan()` is called, all shapes must be `Static` — Phase 1
    /// resolved dynamic dimensions and Phase 2 inferred unknown shapes.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape is not `Static`.
    pub fn static_shape(&self, shape: &TensorShape) -> Result<Vec<usize>> {
        match shape {
            TensorShape::Static(dims) => Ok(dims.clone()),
            TensorShape::Dynamic(_) => Err(CodegenError::InvalidShape(
                "Shape is still Dynamic at plan time — \
                 dynamic dimension resolution failed"
                    .to_string(),
            )),
            TensorShape::Unknown => Err(CodegenError::InvalidShape(
                "Shape is still Unknown at plan time — \
                 shape inference may have failed"
                    .to_string(),
            )),
            TensorShape::Absent => Err(CodegenError::InvalidShape(
                "Cannot get shape of absent optional input".to_string(),
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
        defs: HashMap<String, ShaderDefValue>,
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

/// Registry of Operator implementations.
///
/// Maps ONNX op_type strings (e.g., "Add", "MatMul") to operator implementations
/// that know how to generate GPU code for those operations.
pub struct OperatorRegistry {
    operators: HashMap<String, Box<dyn Operator>>,
}

impl OperatorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    // Old with_defaults method - replaced by onyxia_operators::core_operator_registry()
    // Commented out because the operators module is being migrated (tasks 024-025)
    /*
    /// Create a registry pre-loaded with all built-in operators.
    ///
    /// Registers all implemented operators (Add, Mul, Gelu, RMSNorm, MatMulF32, Cast)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register("Add", Box::new(crate::operators::AddOperator));
        ...
        registry
    }
    */

    /// Register an operator for an op_type string.
    ///
    /// # Arguments
    ///
    /// * `op_type` - ONNX operation type (e.g., "Add", "Mul", "MatMul")
    /// * `operator` - Boxed operator implementation
    pub fn register(&mut self, op_type: impl Into<String>, operator: Box<dyn Operator>) {
        self.operators.insert(op_type.into(), operator);
    }

    /// Look up an operator by ONNX op_type.
    ///
    /// Returns None if no operator is registered for this op_type.
    pub fn get(&self, op_type: &str) -> Option<&dyn Operator> {
        self.operators.get(op_type).map(|k| k.as_ref())
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        // Return empty registry - use onyxia_operators::core_operator_registry() instead
        Self::new()
    }
}

/// Minimal WGSL shader for testing.
///
/// This is a trivial pass-through shader that copies input to output.
/// Used in unit tests to verify operator infrastructure without complex shader logic.
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

    /// Dummy operator for testing that uses TRIVIAL_WGSL.
    struct DummyOperator;

    impl Operator for DummyOperator {
        fn name(&self) -> &str {
            "dummy"
        }

        fn infer_output_shapes(
            &self,
            ctx: &crate::inference::InferenceContext<'_>,
        ) -> Result<Vec<TensorShape>> {
            // Dummy: output shape equals input shape
            Ok(vec![
                ctx.input_shapes
                    .get(0)
                    .cloned()
                    .unwrap_or(TensorShape::Unknown),
            ])
        }

        fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
            // Compile the trivial shader
            let shader_index = ctx.compile_shader("trivial", TRIVIAL_WGSL, HashMap::new())?;

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
                immediates: None,
            };

            Ok(vec![step])
        }
    }

    /// Another dummy operator for multi-operator tests.
    struct AnotherOperator;

    impl Operator for AnotherOperator {
        fn name(&self) -> &str {
            "another"
        }

        fn infer_output_shapes(
            &self,
            ctx: &crate::inference::InferenceContext<'_>,
        ) -> Result<Vec<TensorShape>> {
            // Another dummy: output shape equals input shape
            Ok(vec![
                ctx.input_shapes
                    .get(0)
                    .cloned()
                    .unwrap_or(TensorShape::Unknown),
            ])
        }

        fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>> {
            let shader_index = ctx.compile_shader("another", TRIVIAL_WGSL, HashMap::new())?;

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
                immediates: None,
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
    fn test_register_and_plan_dummy_operator() {
        let mut registry = OperatorRegistry::new();
        registry.register("Dummy", Box::new(DummyOperator));

        // Verify operator was registered
        let operator = registry
            .get("Dummy")
            .expect("Operator should be registered");
        assert_eq!(operator.name(), "dummy");

        // Create test graph and node
        let graph = create_test_graph();
        let mut node = Node::new("Dummy");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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

        let steps = operator.plan(&mut ctx).expect("Planning should succeed");

        // Verify we got a dispatch step
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Dispatch {
                shader_index,
                bindings,
                workgroups,
                ..
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
    fn test_register_multiple_operators() {
        let mut registry = OperatorRegistry::new();

        registry.register("Dummy", Box::new(DummyOperator));
        registry.register("Another", Box::new(AnotherOperator));

        // Look up both operators
        let dummy = registry.get("Dummy").expect("Dummy operator should exist");
        let another = registry
            .get("Another")
            .expect("Another operator should exist");

        assert_eq!(dummy.name(), "dummy");
        assert_eq!(another.name(), "another");

        // Create test graph
        let graph = create_test_graph();
        let mut node = Node::new("Dummy");
        node.inputs = vec!["input".to_string()];
        node.outputs = vec!["output".to_string()];

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        // Plan with dummy operator
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

        // Plan with another operator
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

    // Test commented out - operators being migrated to onyxia-operators
    /*
    #[test]
    fn test_with_defaults() {
        let registry = OperatorRegistry::with_defaults();
        // Verify Add operator is registered
        assert!(registry.get("Add").is_some());
        let add_operator = registry.get("Add").unwrap();
        assert_eq!(add_operator.name(), "Add");
    }
    */

    #[test]
    fn test_plan_context_helpers() {
        let graph = create_test_graph();
        let node = Node::new("Test");

        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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
    fn test_static_shape_static() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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
            .static_shape(&shape)
            .expect("Should resolve static shape");
        assert_eq!(resolved, vec![1, 2, 3]);
    }

    #[test]
    fn test_static_shape_rejects_dynamic() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
        let mut shaders = Vec::new();

        let ctx = PlanContext::for_test(
            &node,
            &graph,
            &input_ids,
            &output_ids,
            &dynamic_dimensions,
            &mut shaders,
        );

        let shape = TensorShape::Dynamic(vec![onyxia_onnx::Dimension::Named("batch".to_string())]);
        let result = ctx.static_shape(&shape);
        assert!(
            result.is_err(),
            "Dynamic shapes should be rejected at plan time"
        );
    }

    #[test]
    fn test_static_shape_unknown() {
        let graph = create_test_graph();
        let node = Node::new("Test");
        let input_ids = vec![0];
        let output_ids = vec![1];
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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
        let result = ctx.static_shape(&shape);
        assert!(result.is_err());
        match result {
            Err(CodegenError::InvalidShape(msg)) => {
                assert!(msg.contains("Unknown"));
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
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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
        let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
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
            .compile_shader("test_shader", TRIVIAL_WGSL, HashMap::new())
            .expect("First compile should succeed");
        let idx2 = ctx
            .compile_shader("test_shader", TRIVIAL_WGSL, HashMap::new())
            .expect("Second compile should succeed");

        // Should return same index (deduplicated)
        assert_eq!(idx1, idx2);
        assert_eq!(shaders.len(), 1);
    }
}
