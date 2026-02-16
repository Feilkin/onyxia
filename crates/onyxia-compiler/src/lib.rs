//! Execution plan compiler for Onyxia.
//!
//! This crate takes ONNX graphs and compiles them into execution plans with
//! pre-compiled WGSL shaders that `onyxia-runtime` can execute on the GPU.
//!
//! The compiler is organized as a pipeline of passes that run in stages:
//! 1. **Resolution** - Resolve symbolic dimensions to concrete values
//! 2. **Folding** - Evaluate constant operations at compile time
//! 3. **Inference** - Propagate tensor shapes through the graph (skips folded nodes)
//! 4. **Optimization** - Apply graph transformations (custom passes)
//! 5. **Planning** - Generate GPU execution steps
//!
//! By running constant folding before shape inference, we avoid unnecessary shape
//! inference for operations that can be completely evaluated at compile time.
//!
//! # Example
//!
//! ```no_run
//! use onyxia_compiler::{compile, CompilerPipeline};
//! use onyxia_core::OperatorRegistry;
//! use onyxia_onnx::Graph;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse ONNX model to graph
//! # let graph = onyxia_onnx::Graph::new();
//!
//! // Compile to execution plan
//! let registry = OperatorRegistry::new();
//! let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
//! let plan = compile(&graph, &registry, &dynamic_dimensions)?;
//!
//! println!("Compiled {} operations", plan.operations.len());
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod passes;
pub mod scheduler;
pub mod symbolic_expr;

pub use error::{CodegenError, Result};
pub use passes::{
    ConstantFoldingPass, InitializeConstantsPass, PlanningPass, ShapeInferencePass,
    SymbolicResolutionPass,
};

// Re-export commonly used types from onyxia-core
pub use onyxia_core::{CompiledModel, IrGraph, Pass, Stage};

use onyxia_onnx::Graph;
use std::collections::HashMap;

/// Compiler pipeline with pluggable passes.
///
/// The pipeline runs in fixed stages: Resolution → Folding → Inference →
/// Optimization → Planning. Built-in passes are registered in their respective
/// stages, and custom passes can be added via `add_pass()`.
///
/// By running constant folding before shape inference, nodes with constant inputs
/// are folded first, and their output shapes are inferred from the folded values,
/// avoiding unnecessary shape inference calls.
pub struct CompilerPipeline {
    /// All passes to run, ordered by (stage, registration order).
    passes: Vec<Box<dyn Pass>>,

    /// Dynamic dimension values for resolution pass.
    dynamic_dimensions: HashMap<String, usize>,
}

impl CompilerPipeline {
    /// Create a pipeline with built-in passes.
    ///
    /// The built-in passes are:
    /// - `SymbolicResolutionPass` (Resolution stage)
    /// - `InitializeConstantsPass` (Resolution stage)
    /// - `ConstantFoldingPass` (Folding stage)
    /// - `ShapeInferencePass` (Inference stage)
    /// - `PlanningPass` (Planning stage)
    ///
    /// Note: Constant folding runs before shape inference so that fully-folded
    /// nodes don't need shape inference at all.
    pub fn new(dynamic_dimensions: HashMap<String, usize>) -> Self {
        let mut pipeline = Self {
            passes: Vec::new(),
            dynamic_dimensions: dynamic_dimensions.clone(),
        };

        // Register built-in passes
        pipeline.add_pass(SymbolicResolutionPass::new(dynamic_dimensions));
        pipeline.add_pass(InitializeConstantsPass::new());
        pipeline.add_pass(ConstantFoldingPass::new());
        pipeline.add_pass(ShapeInferencePass::new());
        pipeline.add_pass(PlanningPass::new());

        pipeline
    }

    /// Add a custom pass to the pipeline.
    ///
    /// The pass will be inserted into the appropriate stage (determined by
    /// `pass.stage()`). Within a stage, passes run in the order they were
    /// registered.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining.
    pub fn add_pass(&mut self, pass: impl Pass + 'static) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Run passes up to and including the specified stage.
    ///
    /// This is useful for inspection tools that need intermediate results
    /// without running the full compilation pipeline.
    ///
    /// # Arguments
    ///
    /// * `graph` - The IR graph to process
    /// * `registry` - The operator registry for looking up operator implementations
    /// * `target_stage` - The stage to run up to (inclusive)
    ///
    /// # Errors
    ///
    /// Returns an error if any pass fails.
    pub fn run_until_stage(
        &mut self,
        graph: &mut IrGraph,
        registry: &onyxia_core::OperatorRegistry,
        target_stage: onyxia_core::Stage,
    ) -> onyxia_core::Result<()> {
        self.passes.sort_by_key(|p| p.stage());

        for pass in &self.passes {
            if pass.stage() > target_stage {
                break;
            }

            let _span =
                tracing::debug_span!("pass", name = pass.name(), stage = ?pass.stage()).entered();
            pass.run(graph, registry)?;
        }

        Ok(())
    }

    /// Run the full pipeline: IrGraph::from_onnx() → stages → CompiledModel.
    ///
    /// # Process
    ///
    /// 1. Convert ONNX graph to IR via `IrGraph::from_onnx()`
    /// 2. Run all passes in stage order (Resolution → Folding → Inference → Optimization → Planning)
    /// 3. Extract the compiled model from the planning pass
    ///
    /// # Arguments
    ///
    /// * `graph` - The ONNX graph to compile
    /// * `registry` - The operator registry for looking up operator implementations
    ///
    /// # Returns
    ///
    /// Returns a `CompiledModel` ready for GPU execution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - ONNX graph conversion fails
    /// - Any pass fails
    /// - Planning pass doesn't produce a model
    #[tracing::instrument(skip_all, fields(num_nodes = graph.nodes.len(), num_tensors = graph.tensor_info.len()))]
    pub fn compile(
        &mut self,
        graph: &Graph,
        registry: &onyxia_core::OperatorRegistry,
    ) -> onyxia_core::Result<CompiledModel> {
        // Step 1: Convert ONNX graph to IR
        let mut ir_graph = IrGraph::from_onnx(graph)?;

        // Step 2: Sort passes by stage
        self.passes.sort_by_key(|p| p.stage());

        // Step 3: Run all passes in order
        for pass in &self.passes {
            let _span =
                tracing::debug_span!("pass", name = pass.name(), stage = ?pass.stage()).entered();
            pass.run(&mut ir_graph, registry)?;
        }

        // Step 4: Extract the compiled model from the planning pass
        // This is a bit awkward because Pass::run() can't return arbitrary data
        // We'll need to build the model directly here instead
        self.build_compiled_model(&mut ir_graph, registry)
    }

    /// Build the final compiled model from the IR graph.
    ///
    /// This is called after all passes have run. It extracts the planned
    /// operations, shaders, and tensor registry from the graph.
    fn build_compiled_model(
        &self,
        graph: &mut IrGraph,
        registry: &onyxia_core::OperatorRegistry,
    ) -> onyxia_core::Result<CompiledModel> {
        use onyxia_core::plan::TensorMetadata;
        use onyxia_core::{ModelMetadata, TensorRegistry};

        let mut operations = Vec::new();
        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut symbolic_bindings = Vec::new();

        // Process nodes in topological order
        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node = graph.node(node_id)?.clone();

            let op_type = node.op_type();

            // Look up operator
            let operator = registry.get(op_type).ok_or_else(|| {
                onyxia_core::Error::Planning(format!(
                    "No operator registered for type: {}",
                    op_type
                ))
            })?;

            // Build plan context
            let mut scratch_buffers = Vec::new();
            let mut ctx = onyxia_core::PlanCtx {
                node: &node,
                graph,
                shaders: &mut shaders,
                shader_cache: &mut shader_cache,
                scratch_buffers: &mut scratch_buffers,
                dynamic_dimensions: &self.dynamic_dimensions,
                symbolic_bindings: &mut symbolic_bindings,
            };

            // Call operator's planning
            let steps = operator.plan(&mut ctx).map_err(|e| {
                onyxia_core::Error::Planning(format!(
                    "Failed to plan node '{}' (op_type: {}): {}",
                    op_type, op_type, e
                ))
            })?;

            // Build PlannedOp
            operations.push(onyxia_core::PlannedOp {
                name: op_type.to_string(),
                steps,
                scratch_buffers,
            });
        }

        // Build tensor registry, preserving weight data for GPU upload
        let mut tensor_registry = TensorRegistry::new();
        let input_set: std::collections::HashSet<_> = graph.inputs.iter().copied().collect();

        for i in 0..graph.tensor_count() {
            let tensor_id = onyxia_core::IrTensorId::new(i);
            let tensor = graph.tensor(tensor_id)?;

            // Carry weight/constant data through to the runtime so it can
            // be uploaded to the GPU buffer during allocation. Skip graph
            // inputs — those arrive at inference time from the user.
            let initial_data = if input_set.contains(&tensor_id) {
                None
            } else {
                match &tensor.data {
                    onyxia_core::EdgeData::Constant(value) => Some(value.to_bytes()),
                    onyxia_core::EdgeData::Initializer(bytes) => Some(bytes.clone()),
                    onyxia_core::EdgeData::Runtime => None,
                }
            };

            let metadata = if let Some(data) = initial_data {
                TensorMetadata::with_initial_data(
                    tensor.name.clone(),
                    tensor.dtype,
                    tensor.shape.clone(),
                    data,
                )
            } else {
                TensorMetadata::new(tensor.name.clone(), tensor.dtype, tensor.shape.clone())
            };
            tensor_registry.add(metadata);
        }

        // Build model
        Ok(CompiledModel {
            operations,
            shaders,
            tensors: tensor_registry,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            symbolic_bindings,
            metadata: ModelMetadata::default(),
        })
    }
}

/// Convenience function: creates a default pipeline and runs it.
///
/// This is the main entry point for the compiler. It creates a pipeline with
/// built-in passes (Resolution, Inference, Folding, Planning) and runs it on
/// the provided ONNX graph.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to compile
/// * `registry` - Operator registry mapping op_types to implementations
/// * `dynamic_dimensions` - Concrete values for symbolic dimensions
///
/// # Returns
///
/// Returns a `CompiledModel` ready for GPU execution.
///
/// # Errors
///
/// Returns an error if:
/// - ONNX graph conversion fails
/// - Any pass fails
/// - Planning pass doesn't produce a model
///
/// # Example
///
/// ```no_run
/// use onyxia_compiler::compile;
/// use onyxia_core::OperatorRegistry;
/// use onyxia_onnx::Graph;
/// use std::collections::HashMap;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let graph = onyxia_onnx::Graph::new();
/// let registry = OperatorRegistry::new();
/// let dynamic_dimensions: HashMap<String, usize> = HashMap::new();
/// let model = compile(&graph, &registry, &dynamic_dimensions)?;
///
/// println!("Compiled {} operations", model.operations.len());
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(skip_all)]
pub fn compile(
    graph: &Graph,
    registry: &onyxia_core::OperatorRegistry,
    dynamic_dimensions: &HashMap<String, usize>,
) -> onyxia_core::Result<CompiledModel> {
    let mut pipeline = CompilerPipeline::new(dynamic_dimensions.clone());
    pipeline.compile(graph, registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::{DataType, IrGraph, IrNode, Operator, TensorDef, TensorShape};

    // Mock operator for testing
    struct MockOperator;

    impl Operator for MockOperator {
        fn name(&self) -> &str {
            "Mock"
        }

        fn infer_output_shapes(
            &self,
            ctx: &onyxia_core::InferenceCtx,
        ) -> onyxia_core::Result<Vec<TensorShape>> {
            Ok(vec![ctx.input_shape(0)?.clone()])
        }

        fn plan(
            &self,
            _ctx: &mut onyxia_core::PlanCtx,
        ) -> onyxia_core::Result<Vec<onyxia_core::Step>> {
            Ok(vec![onyxia_core::Step::WriteBuffer {
                dst: onyxia_core::BufferRef::Tensor(onyxia_core::IrTensorId::new(0)),
                data: vec![0u8; 4],
            }])
        }
    }

    #[test]
    fn test_compiler_pipeline_runs_all_stages() {
        // Create a simple graph
        let mut graph = IrGraph::new();

        let input = TensorDef::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        );
        let input_id = graph.add_tensor(input);

        let output = TensorDef::new(
            "output".to_string(),
            DataType::F32,
            TensorShape::Static(vec![]),
        );
        let output_id = graph.add_tensor(output);

        let mut node = IrNode::new("Mock".to_string());
        node.add_tensor_input(input_id);
        node.add_output(output_id);
        graph.add_node(node);

        graph.inputs.push(input_id);
        graph.outputs.push(output_id);

        // Register mock operator
        let mut registry = onyxia_core::OperatorRegistry::new();
        registry.register("Mock", MockOperator);

        // Create ONNX graph (we'll use IR directly for simplicity in this test)
        // In practice, we'd convert from ONNX
        // For now, just test that the pipeline structure works

        let dynamic_dimensions = HashMap::new();
        let pipeline = CompilerPipeline::new(dynamic_dimensions);

        // Verify passes were registered (5 built-in passes now: SymbolicResolution, InitializeConstants, ConstantFolding, ShapeInference, Planning)
        assert_eq!(pipeline.passes.len(), 5);
    }

    #[test]
    fn test_pipeline_add_custom_pass() {
        use onyxia_core::{IrGraph, OperatorRegistry, Pass, Stage};

        // Custom pass that does nothing
        struct NoOpPass;

        impl Pass for NoOpPass {
            fn name(&self) -> &str {
                "noop"
            }

            fn stage(&self) -> Stage {
                Stage::Optimization
            }

            fn run(
                &self,
                _graph: &mut IrGraph,
                _registry: &OperatorRegistry,
            ) -> onyxia_core::Result<bool> {
                Ok(false)
            }
        }

        let _registry = onyxia_core::OperatorRegistry::new();
        let dynamic_dimensions = HashMap::new();
        let mut pipeline = CompilerPipeline::new(dynamic_dimensions);

        // Add custom pass
        pipeline.add_pass(NoOpPass);

        // Should now have 6 passes (5 built-in + 1 custom)
        assert_eq!(pipeline.passes.len(), 6);
    }

    #[test]
    fn test_convenience_compile_function() {
        // Test that the convenience function works
        // We'll use a minimal ONNX graph

        let graph = onyxia_onnx::Graph::new();
        let registry = onyxia_core::OperatorRegistry::new();
        let dynamic_dimensions = HashMap::new();

        // This should not panic (though it may return an error for empty graph)
        let _result = compile(&graph, &registry, &dynamic_dimensions);
        // We don't assert Ok because an empty graph may fail
    }
}
