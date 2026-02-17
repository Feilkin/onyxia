//! Execution plan compiler for Onyxia.
//!
//! This crate takes ONNX graphs and compiles them into dispatch models with
//! pre-compiled WGSL shaders that `onyxia-runtime` can execute on the GPU.
//!
//! The compiler runs a single pass (InitializeConstants) followed by dispatch
//! model construction. Shape inference and constant folding have been removed
//! in favor of runtime shape computation from actual input tensors.
//!
//! # Example
//!
//! ```no_run
//! use onyxia_compiler::CompilerPipeline;
//! use onyxia_core::OperatorRegistry;
//! use onyxia_onnx::Graph;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse ONNX model to graph
//! # let graph = onyxia_onnx::Graph::new();
//!
//! // Compile to dispatch model
//! let registry = OperatorRegistry::new();
//! let mut pipeline = CompilerPipeline::new();
//! let model = pipeline.compile(&graph, &registry)?;
//!
//! println!("Compiled {} operations", model.entries.len());
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod passes;
pub mod scheduler;

pub use error::{CodegenError, Result};
pub use passes::InitializeConstantsPass;

// Re-export commonly used types from onyxia-core
pub use onyxia_core::{DispatchModel as CompiledModel, IrGraph, Pass, Stage};

use onyxia_core::{CompileCtx, DispatchEntry, DispatchModel, EdgeData, IrTensorId, WeightRegister};
use onyxia_core::{Error, ModelMetadata};
use onyxia_onnx::Graph;
use std::collections::HashMap;

/// Compiler pipeline with pluggable passes.
///
/// The new simplified pipeline: InitializeConstants → build dispatch model.
/// Shape inference and constant folding have been removed.
pub struct CompilerPipeline {
    /// All passes to run, ordered by (stage, registration order).
    passes: Vec<Box<dyn Pass>>,
}

impl CompilerPipeline {
    /// Create a pipeline with built-in passes.
    ///
    /// The built-in pass is:
    /// - `InitializeConstantsPass` (Resolution stage)
    pub fn new() -> Self {
        let mut pipeline = Self { passes: Vec::new() };

        // Register built-in pass
        pipeline.add_pass(InitializeConstantsPass::new());

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

    /// Run the full pipeline: IrGraph::from_onnx() → passes → DispatchModel.
    ///
    /// # Process
    ///
    /// 1. Convert ONNX graph to IR via `IrGraph::from_onnx()`
    /// 2. Run all passes in stage order (just InitializeConstants by default)
    /// 3. Build dispatch model from IR graph
    ///
    /// # Arguments
    ///
    /// * `graph` - The ONNX graph to compile
    /// * `registry` - The operator registry for looking up operator implementations
    ///
    /// # Returns
    ///
    /// Returns a `DispatchModel` ready for GPU execution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - ONNX graph conversion fails
    /// - Any pass fails
    /// - Dispatch model construction fails
    #[tracing::instrument(skip_all, fields(num_nodes = graph.nodes.len(), num_tensors = graph.tensor_info.len()))]
    pub fn compile(
        &mut self,
        graph: &Graph,
        registry: &onyxia_core::OperatorRegistry,
    ) -> onyxia_core::Result<DispatchModel> {
        // Step 1: Convert ONNX graph to IR
        let mut ir_graph = IrGraph::from_onnx(graph)?;

        // Step 2: Sort passes by stage and run them
        self.passes.sort_by_key(|p| p.stage());
        for pass in &self.passes {
            let _span =
                tracing::debug_span!("pass", name = pass.name(), stage = ?pass.stage()).entered();
            pass.run(&mut ir_graph, registry)?;
        }

        // Step 3: Build dispatch model
        self.build_dispatch_model(&ir_graph, registry)
    }

    /// Build the final dispatch model from the IR graph.
    ///
    /// Walks the graph in topological order, calls `operator.create_dispatch()`
    /// for each node, and assigns register routing.
    fn build_dispatch_model(
        &self,
        graph: &IrGraph,
        registry: &onyxia_core::OperatorRegistry,
    ) -> onyxia_core::Result<DispatchModel> {
        let mut entries = Vec::new();
        let mut shader_cache = HashMap::new();

        let topo_order = graph.topological_order();

        for node_id in topo_order {
            let node = graph.node(node_id)?;
            let op_type = node.op_type();

            let operator = registry.get(op_type).ok_or_else(|| {
                Error::Compilation(format!("No operator registered for type: {op_type}"))
            })?;

            let mut ctx = CompileCtx::new(node_id, node, graph, &mut shader_cache);
            let dispatch = operator.create_dispatch(&mut ctx)?;

            // Register routing: input/output tensor IDs map directly to register indices
            let input_regs: Vec<usize> = node.inputs().iter().map(|id| id.index()).collect();

            let output_regs: Vec<usize> = node.outputs().iter().map(|id| id.index()).collect();

            entries.push(DispatchEntry {
                op: dispatch,
                input_regs,
                output_regs,
                name: op_type.to_string(),
            });
        }

        // Build weight registers from initializer/constant data
        let mut weight_registers = Vec::new();
        let input_set: std::collections::HashSet<_> = graph.inputs.iter().copied().collect();

        for i in 0..graph.tensor_count() {
            let tensor_id = IrTensorId::new(i);
            if input_set.contains(&tensor_id) {
                continue; // Skip model inputs — they arrive at runtime
            }

            let tensor = graph.tensor(tensor_id)?;
            let data = match &tensor.data {
                EdgeData::Constant(value) => Some(value.to_bytes()),
                EdgeData::Initializer(bytes) => Some(bytes.clone()),
                EdgeData::Runtime => None,
            };

            if let Some(data) = data {
                let shape = tensor
                    .shape
                    .as_static()
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                weight_registers.push(WeightRegister {
                    register: i,
                    data,
                    shape,
                    dtype: tensor.dtype,
                });
            }
        }

        // Build input/output register mappings
        let input_registers = graph
            .inputs
            .iter()
            .map(|id| {
                let name = graph
                    .tensor(*id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                (name, id.index())
            })
            .collect();

        let output_registers = graph
            .outputs
            .iter()
            .map(|id| {
                let name = graph
                    .tensor(*id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                (name, id.index())
            })
            .collect();

        Ok(DispatchModel {
            entries,
            num_registers: graph.tensor_count(),
            input_registers,
            output_registers,
            weight_registers,
            metadata: ModelMetadata::default(),
        })
    }
}

impl Default for CompilerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: creates a default pipeline and runs it.
///
/// This is the main entry point for the compiler. It creates a pipeline with
/// the built-in pass (InitializeConstants) and runs it on the provided ONNX graph.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to compile
/// * `registry` - Operator registry mapping op_types to implementations
///
/// # Returns
///
/// Returns a `DispatchModel` ready for GPU execution.
///
/// # Errors
///
/// Returns an error if:
/// - ONNX graph conversion fails
/// - Any pass fails
/// - Dispatch model construction fails
///
/// # Example
///
/// ```no_run
/// use onyxia_compiler::compile;
/// use onyxia_core::OperatorRegistry;
/// use onyxia_onnx::Graph;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let graph = onyxia_onnx::Graph::new();
/// let registry = OperatorRegistry::new();
/// let model = compile(&graph, &registry)?;
///
/// println!("Compiled {} operations", model.entries.len());
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(skip_all)]
pub fn compile(
    graph: &Graph,
    registry: &onyxia_core::OperatorRegistry,
) -> onyxia_core::Result<DispatchModel> {
    let mut pipeline = CompilerPipeline::new();
    pipeline.compile(graph, registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_pipeline_runs() {
        // Create pipeline
        let pipeline = CompilerPipeline::new();

        // Verify pass was registered (1 built-in pass: InitializeConstants)
        assert_eq!(pipeline.passes.len(), 1);
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

        let mut pipeline = CompilerPipeline::new();

        // Add custom pass
        pipeline.add_pass(NoOpPass);

        // Should now have 2 passes (1 built-in + 1 custom)
        assert_eq!(pipeline.passes.len(), 2);
    }

    #[test]
    fn test_convenience_compile_function() {
        // Test that the convenience function works
        // We'll use a minimal ONNX graph

        let graph = onyxia_onnx::Graph::new();
        let registry = onyxia_core::OperatorRegistry::new();

        // This should not panic (though it may return an error for empty graph)
        let _result = compile(&graph, &registry);
        // We don't assert Ok because an empty graph may fail
    }

    #[test]
    fn test_pipeline_produces_dispatch_model() {
        use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

        // Create a simple Add graph
        let mut graph = onyxia_onnx::Graph::new();

        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Add");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];
        graph.add_node(node);

        graph.inputs = vec!["a".to_string(), "b".to_string()];
        graph.outputs = vec!["c".to_string()];

        // Compile with core registry (includes Add operator)
        let registry = onyxia_operators::core_operator_registry();

        let mut pipeline = CompilerPipeline::new();
        let model = pipeline.compile(&graph, &registry).unwrap();

        // Verify structure
        assert!(!model.entries.is_empty(), "Model should have operations");
        assert!(model.num_registers > 0, "Model should have registers");
        assert!(
            !model.input_registers.is_empty(),
            "Model should have inputs"
        );
        assert!(
            !model.output_registers.is_empty(),
            "Model should have outputs"
        );
    }

    #[test]
    fn test_register_routing_matches_graph() {
        use onyxia_onnx::{DataType, Node, TensorInfo, TensorKind, TensorShape};

        // Create a simple Add graph
        let mut graph = onyxia_onnx::Graph::new();

        graph.add_tensor(TensorInfo {
            name: "a".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "b".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Input,
            initializer: None,
        });

        graph.add_tensor(TensorInfo {
            name: "c".to_string(),
            dtype: DataType::F32,
            shape: TensorShape::Static(vec![4]),
            kind: TensorKind::Output,
            initializer: None,
        });

        let mut node = Node::new("Add");
        node.inputs = vec!["a".to_string(), "b".to_string()];
        node.outputs = vec!["c".to_string()];
        graph.add_node(node);

        graph.inputs = vec!["a".to_string(), "b".to_string()];
        graph.outputs = vec!["c".to_string()];

        let registry = onyxia_operators::core_operator_registry();
        let mut pipeline = CompilerPipeline::new();
        let model = pipeline.compile(&graph, &registry).unwrap();

        // The Add entry should read from input registers and write to output register
        assert_eq!(model.entries.len(), 1, "Should have one operation");

        let add_entry = &model.entries[0];
        assert_eq!(
            add_entry.input_regs.len(),
            2,
            "Add should have 2 input registers"
        );
        assert_eq!(
            add_entry.output_regs.len(),
            1,
            "Add should have 1 output register"
        );

        // Verify input/output mappings exist
        assert_eq!(model.input_registers.len(), 2);
        assert_eq!(model.output_registers.len(), 1);
    }
}
