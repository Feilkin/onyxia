//! Compile-time context for creating dispatch objects.
//!
//! `CompileCtx` provides the static information available when the compiler
//! walks the ONNX graph to build dispatch objects: node attributes, input
//! metadata, and shader compilation.

use crate::Result;
use crate::ir::{EdgeData, IrEdge, IrGraph, IrNode, IrNodeId};
use crate::types::TensorValue;
use onyxia_onnx::AttributeValue;
use std::collections::HashMap;

/// Compile-time context passed to `Operator::create_dispatch()`.
///
/// Provides access to node attributes, weight/initializer values, and
/// shader compilation. Does NOT provide tensor shapes (those are only
/// known at runtime).
pub struct CompileCtx<'a> {
    /// The IR node being compiled.
    pub node: &'a IrNode,

    /// The node ID in the graph.
    pub node_id: IrNodeId,

    /// The full IR graph (for accessing input edges and their data).
    pub graph: &'a IrGraph,

    /// Shader compilation cache (WGSL source → naga module).
    shader_cache: &'a mut HashMap<String, naga::Module>,
}

impl<'a> CompileCtx<'a> {
    /// Create a new compile context.
    pub fn new(
        node_id: IrNodeId,
        node: &'a IrNode,
        graph: &'a IrGraph,
        shader_cache: &'a mut HashMap<String, naga::Module>,
    ) -> Self {
        Self {
            node,
            node_id,
            graph,
            shader_cache,
        }
    }

    /// Get the number of inputs to this node.
    pub fn input_count(&self) -> usize {
        self.node.inputs().len()
    }

    /// Get the number of outputs from this node.
    pub fn output_count(&self) -> usize {
        self.node.outputs().len()
    }

    /// Get the input edge (tensor metadata) for the given index.
    ///
    /// Returns the IrEdge which has name, dtype, shape (from ONNX), and
    /// potentially weight data.
    pub fn input_edge(&self, index: usize) -> Result<&IrEdge> {
        let inputs = self.node.inputs();
        let input_id = inputs.get(index).ok_or_else(|| {
            crate::Error::Compilation(format!(
                "Input index {index} out of range (node has {} inputs)",
                inputs.len()
            ))
        })?;
        self.graph.edge(*input_id)
    }

    /// Get the constant/initializer value for an input, if available.
    ///
    /// Returns `Some(TensorValue)` if the input is a weight or constant
    /// (parsed by `InitializeConstantsPass`). Returns `None` if the input
    /// is a runtime value.
    pub fn input_value(&self, index: usize) -> Result<Option<&TensorValue>> {
        let edge = self.input_edge(index)?;
        match &edge.data {
            EdgeData::Constant(value) => Ok(Some(value)),
            _ => Ok(None),
        }
    }

    /// Get a node attribute by name.
    pub fn attr(&self, name: &str) -> Option<&AttributeValue> {
        self.node.attributes.get(name)
    }

    /// Get a required i64 attribute.
    pub fn attr_i64(&self, name: &str) -> Result<i64> {
        match self.attr(name) {
            Some(AttributeValue::Int(v)) => Ok(*v),
            _ => Err(crate::Error::Compilation(format!(
                "Missing required i64 attribute '{name}'"
            ))),
        }
    }

    /// Get a required string attribute.
    pub fn attr_string(&self, name: &str) -> Result<&str> {
        match self.attr(name) {
            Some(AttributeValue::String(v)) => Ok(v.as_str()),
            _ => Err(crate::Error::Compilation(format!(
                "Missing required string attribute '{name}'"
            ))),
        }
    }

    /// Get a required ints attribute.
    pub fn attr_ints(&self, name: &str) -> Result<&[i64]> {
        match self.attr(name) {
            Some(AttributeValue::Ints(v)) => Ok(v.as_slice()),
            _ => Err(crate::Error::Compilation(format!(
                "Missing required ints attribute '{name}'"
            ))),
        }
    }

    /// Compile a WGSL shader source into a naga module.
    ///
    /// Results are cached by label — the same shader source is only
    /// compiled once even if used by multiple operator instances.
    pub fn compile_shader(
        &mut self,
        label: &str,
        source: &str,
        defines: &HashMap<String, String>,
    ) -> Result<naga::Module> {
        // Check cache first
        if let Some(module) = self.shader_cache.get(label) {
            return Ok(module.clone());
        }

        // Use naga_oil to preprocess and compile WGSL
        let module = compile_wgsl_to_naga(label, source, defines)?;

        self.shader_cache.insert(label.to_string(), module.clone());
        Ok(module)
    }
}

/// Compile WGSL source to a naga module using naga_oil preprocessing.
fn compile_wgsl_to_naga(
    label: &str,
    source: &str,
    defines: &HashMap<String, String>,
) -> Result<naga::Module> {
    use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};

    let mut composer = Composer::default();

    let shader_defs: HashMap<String, ShaderDefValue> = defines
        .iter()
        .map(|(k, v)| {
            if let Ok(int_val) = v.parse::<i32>() {
                (k.clone(), ShaderDefValue::Int(int_val))
            } else {
                (k.clone(), ShaderDefValue::Bool(v == "true"))
            }
        })
        .collect();

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: label,
            shader_defs,
            ..Default::default()
        })
        .map_err(|e| {
            crate::Error::Compilation(format!("Shader compilation failed for '{label}': {e}"))
        })?;

    Ok(module)
}
