//! Context types for operator methods.
//!
//! Provides structured access to node state, graph data, and compilation
//! services during shape inference, constant folding, and planning.
//!
//! All context types now work with `IrEdgeId` directly — no more
//! `IrInput::Tensor` vs `IrInput::ValueNode` branching.

use crate::ir::{IrEdge, IrGraph, IrNode};
use crate::plan::{BufferRef, ShaderIndex};
use crate::types::{SymbolicDim, TensorData, TensorShape, TensorValue};
use crate::{Error, Result};
use onyxia_onnx::AttributeValue;
use std::collections::HashMap;

/// Context for shape inference operations.
///
/// Provides read-only access to:
/// - Input tensor shapes
/// - Constant-folded input values (for data-dependent shape inference)
/// - Node attributes
/// - Graph structure
pub struct InferenceCtx<'a> {
    /// The node being processed.
    pub node: &'a IrNode,

    /// The graph containing the node.
    pub graph: &'a IrGraph,
}

impl<'a> InferenceCtx<'a> {
    /// Create a new inference context.
    pub fn new(node: &'a IrNode, graph: &'a IrGraph) -> Self {
        Self { node, graph }
    }

    /// Get the shape of an input tensor.
    ///
    /// Returns an error if the input index is out of bounds or the input
    /// tensor is absent (ONNX optional input not provided).
    pub fn input_shape(&self, index: usize) -> Result<TensorShape> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::ShapeInference(format!("Input {} not found", index)))?;

        let edge = self.graph.edge(*edge_id)?;

        // If the edge has a constant value (from folding), use that shape
        if let Some(value) = edge.constant_value() {
            return Ok(TensorShape::Static(value.shape.clone()));
        }

        if edge.shape.is_absent() {
            return Err(Error::ShapeInference(format!(
                "Input {} is absent (optional input not provided)",
                index
            )));
        }

        Ok(edge.shape.clone())
    }

    /// Get static dimensions from an input tensor.
    ///
    /// Returns an error if the input is not fully static or is absent.
    pub fn require_static(&self, index: usize) -> Result<Vec<usize>> {
        let shape = self.input_shape(index)?;
        shape
            .as_static()
            .map(|dims| dims.to_vec())
            .ok_or_else(|| Error::ShapeInference(format!("Input {} must have static shape", index)))
    }

    /// Get the constant-folded value of an input tensor, if available.
    ///
    /// Returns `None` if the input has not been constant-folded, or if the
    /// input index is out of bounds.
    pub fn input_value(&self, index: usize) -> Option<&TensorValue> {
        let edge_id = self.node.inputs().get(index)?;
        let edge = self.graph.edge(*edge_id).ok()?;
        edge.constant_value()
    }

    /// Get the data type of an input tensor.
    pub fn input_dtype(&self, index: usize) -> Result<crate::types::DataType> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::ShapeInference(format!("Input {} not found", index)))?;

        let edge = self.graph.edge(*edge_id)?;

        // Prefer constant value dtype if available
        if let Some(value) = edge.constant_value() {
            return Ok(value.dtype);
        }

        Ok(edge.dtype)
    }

    /// Get the number of inputs.
    pub fn input_count(&self) -> usize {
        self.node.inputs().len()
    }

    /// Get the number of outputs.
    pub fn output_count(&self) -> usize {
        self.node.outputs().len()
    }

    // --- Attribute accessors ---

    /// Get an i64 attribute.
    pub fn attr_i64(&self, key: &str) -> Result<i64> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Int(v) => Ok(*v),
            _ => Err(Error::Attribute(format!("Attribute {} is not an i64", key))),
        }
    }

    /// Get an f32 attribute.
    pub fn attr_f32(&self, key: &str) -> Result<f32> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Float(v) => Ok(*v),
            _ => Err(Error::Attribute(format!("Attribute {} is not an f32", key))),
        }
    }

    /// Get a string attribute.
    pub fn attr_string(&self, key: &str) -> Result<&str> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::String(v) => Ok(v.as_str()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not a string",
                key
            ))),
        }
    }

    /// Get an i64 array attribute.
    pub fn attr_ints(&self, key: &str) -> Result<&[i64]> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Ints(v) => Ok(v.as_slice()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not an i64 array",
                key
            ))),
        }
    }

    /// Get an f32 array attribute.
    pub fn attr_floats(&self, key: &str) -> Result<&[f32]> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Floats(v) => Ok(v.as_slice()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not an f32 array",
                key
            ))),
        }
    }

    /// Get an optional i64 attribute with a default value.
    pub fn attr_i64_or(&self, key: &str, default: i64) -> i64 {
        self.attr_i64(key).unwrap_or(default)
    }

    /// Get an optional f32 attribute with a default value.
    pub fn attr_f32_or(&self, key: &str, default: f32) -> f32 {
        self.attr_f32(key).unwrap_or(default)
    }

    /// Check if an attribute exists.
    pub fn has_attr(&self, key: &str) -> bool {
        self.node.get_attribute(key).is_some()
    }

    // --- Enhanced error reporting ---

    /// Create a detailed shape inference error with full context.
    ///
    /// This includes the node name, operator type, and detailed information
    /// about all input tensors (shapes, types, and constant values if available).
    pub fn shape_error(&self, message: impl Into<String>) -> Error {
        let mut error_msg = format!(
            "Shape inference failed for node '{}' ({})\n\n",
            self.node_name(),
            self.op_type()
        );

        // Add input details
        error_msg.push_str("  Inputs:\n");
        for i in 0..self.input_count() {
            match self.input_details(i) {
                Ok(details) => {
                    error_msg.push_str(&format!("    {}\n", details));
                }
                Err(_) => {
                    error_msg.push_str(&format!("    {}: <error reading input>\n", i));
                }
            }
        }

        error_msg.push_str("\n  Error: ");
        error_msg.push_str(&message.into());

        Error::ShapeInference(error_msg)
    }

    /// Get detailed string description of an input.
    ///
    /// Returns a multi-line formatted string with information about:
    /// - Input index and tensor name
    /// - Shape and data type
    /// - Whether it's a constant or runtime tensor
    /// - The constant value (for small tensors)
    fn input_details(&self, index: usize) -> Result<String> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::ShapeInference(format!("Input {} not found", index)))?;

        let edge = self.graph.edge(*edge_id)?;

        let mut desc = format!("{}: '{}'", index, edge.name);
        desc.push_str(&format!("\n      Shape: {:?}", edge.shape));
        desc.push_str(&format!("\n      Type: {:?}", edge.dtype));

        // Check if it's a constant value (from folding)
        if let Some(value) = edge.constant_value() {
            desc.push_str("\n      Source: Constant");
            if let Some(formatted) = format_small_value(value, 20) {
                desc.push_str(&format!(" with value {}", formatted));
            }
        } else if edge.has_initializer() {
            desc.push_str("\n      Source: Constant (initializer)");
            // Try to parse and show value for small tensors
            if let TensorShape::Static(shape) = &edge.shape {
                let element_count: usize = shape.iter().product();
                if element_count <= 20
                    && let Some(initializer) = edge.initializer()
                    && let Ok(value) = TensorValue::from_bytes(initializer, edge.dtype, shape)
                    && let Some(formatted) = format_small_value(&value, 20)
                {
                    desc.push_str(&format!(" with value {}", formatted));
                }
            }
        } else {
            desc.push_str("\n      Source: Runtime tensor");
        }

        Ok(desc)
    }

    /// Get the node name.
    fn node_name(&self) -> &str {
        if self.node.name.is_empty() {
            "<unnamed>"
        } else {
            &self.node.name
        }
    }

    /// Get the operator type.
    fn op_type(&self) -> &str {
        &self.node.op_type
    }
}

/// Context for constant folding operations.
///
/// Extends `InferenceCtx` with additional helpers for performing compile-time
/// evaluation of operations.
pub struct FoldCtx<'a> {
    /// The underlying inference context.
    pub ctx: InferenceCtx<'a>,
}

impl<'a> FoldCtx<'a> {
    /// Create a new fold context.
    pub fn new(node: &'a IrNode, graph: &'a IrGraph) -> Self {
        Self {
            ctx: InferenceCtx::new(node, graph),
        }
    }

    /// Get the constant value for an input, if available.
    ///
    /// Checks (in order):
    /// 1. `EdgeData::Constant` (set by a prior folding pass)
    /// 2. `EdgeData::Initializer` (raw ONNX bytes, parsed on-demand)
    fn get_input_value(&self, index: usize) -> Result<Option<TensorValue>> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::ConstantFolding("Input not found".to_string()))?;

        let edge = self.graph.edge(*edge_id)?;

        // Already folded?
        if let Some(value) = edge.constant_value() {
            return Ok(Some(value.clone()));
        }

        // Has an initializer we can parse?
        if let Some(initializer) = edge.initializer() {
            let shape = match &edge.shape {
                TensorShape::Static(dims) => dims.clone(),
                _ => return Ok(None), // Can't fold non-static shapes
            };

            let value = TensorValue::from_bytes(initializer, edge.dtype, &shape)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Check if all inputs have constant-folded values.
    pub fn all_inputs_have_values(&self) -> bool {
        (0..self.ctx.input_count()).all(|i| self.get_input_value(i).ok().and_then(|v| v).is_some())
    }

    /// Apply a binary operation to two f32 inputs and return the result.
    ///
    /// This is a convenience method for elementwise binary ops like Add, Mul.
    /// Returns `Ok(vec![None])` if inputs are not f32 (gracefully skips folding).
    pub fn binary_fold_f32<F>(&self, op: F) -> Result<Vec<Option<TensorValue>>>
    where
        F: Fn(f32, f32) -> f32,
    {
        let val_a = match self.get_input_value(0)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let val_b = match self.get_input_value(1)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let Some(a) = val_a.as_f32() else {
            return Ok(vec![None]); // Not f32 — skip folding
        };

        let Some(b) = val_b.as_f32() else {
            return Ok(vec![None]); // Not f32 — skip folding
        };

        // Simple element-wise operation (no broadcasting for now)
        if a.len() != b.len() {
            return Ok(vec![None]); // Skip folding if shapes don't match
        }

        let result_data: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();

        let result = TensorValue::new(
            crate::types::TensorData::F32(result_data),
            val_a.shape.clone(),
            crate::types::DataType::F32,
        );

        Ok(vec![Some(result)])
    }

    /// Apply a binary operation to two i64 inputs and return the result.
    ///
    /// Returns `Ok(vec![None])` if inputs are not i64 (gracefully skips folding).
    pub fn binary_fold_i64<F>(&self, op: F) -> Result<Vec<Option<TensorValue>>>
    where
        F: Fn(i64, i64) -> i64,
    {
        let val_a = match self.get_input_value(0)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let val_b = match self.get_input_value(1)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let Some(a) = val_a.as_i64() else {
            return Ok(vec![None]);
        };

        let Some(b) = val_b.as_i64() else {
            return Ok(vec![None]);
        };

        if a.len() != b.len() {
            return Ok(vec![None]);
        }

        let result_data: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();

        let result = TensorValue::new(
            crate::types::TensorData::I64(result_data),
            val_a.shape.clone(),
            crate::types::DataType::I64,
        );

        Ok(vec![Some(result)])
    }

    /// Apply a binary operation to two i32 inputs and return the result.
    ///
    /// Returns `Ok(vec![None])` if inputs are not i32 (gracefully skips folding).
    pub fn binary_fold_i32<F>(&self, op: F) -> Result<Vec<Option<TensorValue>>>
    where
        F: Fn(i32, i32) -> i32,
    {
        let val_a = match self.get_input_value(0)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let val_b = match self.get_input_value(1)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let Some(a) = val_a.as_i32() else {
            return Ok(vec![None]);
        };

        let Some(b) = val_b.as_i32() else {
            return Ok(vec![None]);
        };

        if a.len() != b.len() {
            return Ok(vec![None]);
        }

        let result_data: Vec<i32> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();

        let result = TensorValue::new(
            crate::types::TensorData::I32(result_data),
            val_a.shape.clone(),
            crate::types::DataType::I32,
        );

        Ok(vec![Some(result)])
    }

    /// Apply a unary operation to an f32 input and return the result.
    ///
    /// Returns `Ok(vec![None])` if input is not f32 (gracefully skips folding).
    pub fn unary_fold_f32<F>(&self, op: F) -> Result<Vec<Option<TensorValue>>>
    where
        F: Fn(f32) -> f32,
    {
        let val = match self.get_input_value(0)? {
            Some(v) => v,
            None => return Ok(vec![None]),
        };

        let Some(input) = val.as_f32() else {
            return Ok(vec![None]); // Not f32 — skip folding
        };

        let result_data: Vec<f32> = input.iter().map(|&x| op(x)).collect();

        let result = TensorValue::new(
            crate::types::TensorData::F32(result_data),
            val.shape.clone(),
            crate::types::DataType::F32,
        );

        Ok(vec![Some(result)])
    }
}

impl<'a> std::ops::Deref for FoldCtx<'a> {
    type Target = InferenceCtx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

/// Context for planning (code generation) operations.
///
/// Provides access to:
/// - Buffer references for inputs/outputs
/// - Tensor metadata (shapes, dtypes)
/// - Shader compilation services
/// - Scratch buffer allocation
/// - Dimension encoding (shader defs vs immediates)
pub struct PlanCtx<'a> {
    /// The node being planned.
    pub node: &'a IrNode,

    /// The graph containing the node.
    pub graph: &'a IrGraph,

    /// Compiled shaders (shared across all operators).
    pub shaders: &'a mut Vec<crate::plan::CompiledShader>,

    /// Shader deduplication cache: (label, defines) -> ShaderIndex.
    pub shader_cache: &'a mut HashMap<(String, String), ShaderIndex>,

    /// Scratch buffers allocated so far.
    pub scratch_buffers: &'a mut Vec<crate::plan::ScratchBufferDesc>,

    /// Dynamic dimension values (for resolving symbolic dimensions at runtime).
    pub dynamic_dimensions: &'a HashMap<String, usize>,

    /// Symbolic dimension bindings (for runtime update support).
    pub symbolic_bindings: &'a mut Vec<crate::plan::SymbolicBinding>,
}

impl<'a> PlanCtx<'a> {
    /// Get the buffer reference for an input.
    ///
    /// Every input is an edge ID, so we always return `BufferRef::Tensor`.
    pub fn input(&self, index: usize) -> Result<BufferRef> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::Planning(format!("Input {} not found", index)))?;

        Ok(BufferRef::Tensor(*edge_id))
    }

    /// Get the buffer reference for an output tensor.
    pub fn output(&self, index: usize) -> Result<BufferRef> {
        let edge_id = self
            .node
            .outputs()
            .get(index)
            .ok_or_else(|| Error::Planning(format!("Output {} not found", index)))?;

        Ok(BufferRef::Tensor(*edge_id))
    }

    /// Get the edge (tensor) metadata for an input.
    pub fn input_tensor(&self, index: usize) -> Result<&IrEdge> {
        let edge_id = self
            .node
            .inputs()
            .get(index)
            .ok_or_else(|| Error::Planning(format!("Input {} not found", index)))?;

        self.graph.edge(*edge_id)
    }

    /// Get the edge (tensor) metadata for an output.
    pub fn output_tensor(&self, index: usize) -> Result<&IrEdge> {
        let edge_id = self
            .node
            .outputs()
            .get(index)
            .ok_or_else(|| Error::Planning(format!("Output {} not found", index)))?;

        self.graph.edge(*edge_id)
    }

    /// Get static dimensions from a tensor shape.
    ///
    /// Returns an error if the shape is not fully static. All shapes should
    /// be resolved to static by the time planning happens.
    pub fn static_dims(&self, shape: &TensorShape) -> Result<Vec<usize>> {
        shape.as_static().map(|dims| dims.to_vec()).ok_or_else(|| {
            Error::Planning(
                "Cannot plan with non-static shape; run dimension resolution pass first"
                    .to_string(),
            )
        })
    }

    /// Get symbolic dimensions from a tensor shape.
    ///
    /// Returns the symbolic dimensions even if some are fixed.
    pub fn symbolic_dims(&self, shape: &TensorShape) -> Result<Vec<SymbolicDim>> {
        match shape {
            TensorShape::Symbolic(dims) => Ok(dims.clone()),
            TensorShape::Static(dims) => Ok(dims.iter().map(|&d| SymbolicDim::Fixed(d)).collect()),
            _ => Err(Error::Planning(
                "Cannot get symbolic dimensions from absent shape".to_string(),
            )),
        }
    }

    /// Compile a WGSL shader and return its index.
    ///
    /// Deduplicates shaders based on (label, defines) to avoid compiling the
    /// same shader multiple times.
    pub fn compile_shader(
        &mut self,
        label: &str,
        source: &str,
        defines: &HashMap<String, String>,
    ) -> Result<ShaderIndex> {
        // Create a cache key from label and defines
        let defines_str = {
            let mut pairs: Vec<_> = defines.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            pairs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(";")
        };

        let cache_key = (label.to_string(), defines_str);

        // Check cache
        if let Some(&index) = self.shader_cache.get(&cache_key) {
            return Ok(index);
        }

        // Compile shader using naga_oil
        let mut composer = naga_oil::compose::Composer::default()
            .with_capabilities(naga::valid::Capabilities::all());

        // Convert string defines to ShaderDefValue (parse as UInt)
        let shader_defs: HashMap<String, naga_oil::compose::ShaderDefValue> = defines
            .iter()
            .map(|(k, v)| {
                let value = v.parse::<u32>().unwrap_or_else(|_| {
                    panic!("Shader define '{}' has non-numeric value '{}'", k, v)
                });
                (k.clone(), naga_oil::compose::ShaderDefValue::UInt(value))
            })
            .collect();

        let naga_module = composer
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                source,
                file_path: &format!("{}.wgsl", label),
                shader_defs,
                ..Default::default()
            })
            .map_err(|e| Error::ShaderCompilation(format!("Shader compilation failed: {:?}", e)))?;

        let shader = crate::plan::CompiledShader {
            label: label.to_string(),
            module: naga_module,
            entry_point: "main".to_string(),
        };

        let index = self.shaders.len();
        self.shaders.push(shader);
        self.shader_cache.insert(cache_key, index);

        Ok(index)
    }

    /// Allocate a scratch buffer and return its reference.
    pub fn alloc_scratch(&mut self, size: u64, label: String) -> BufferRef {
        let index = self.scratch_buffers.len();
        self.scratch_buffers
            .push(crate::plan::ScratchBufferDesc { size, label });
        BufferRef::Scratch(index)
    }

    /// Get an attribute value (pass-through to node attributes).
    pub fn attr_i64(&self, key: &str) -> Result<i64> {
        let ctx = InferenceCtx::new(self.node, self.graph);
        ctx.attr_i64(key)
    }

    /// Get an f32 attribute (pass-through).
    pub fn attr_f32(&self, key: &str) -> Result<f32> {
        let ctx = InferenceCtx::new(self.node, self.graph);
        ctx.attr_f32(key)
    }

    /// Get a string attribute (pass-through).
    pub fn attr_string(&self, key: &str) -> Result<String> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::String(v) => Ok(v.clone()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not a string",
                key
            ))),
        }
    }

    /// Get an i64 array attribute (pass-through).
    pub fn attr_ints(&self, key: &str) -> Result<Vec<i64>> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Ints(v) => Ok(v.clone()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not an i64 array",
                key
            ))),
        }
    }

    pub fn attr_tensor(&self, key: &str) -> Result<Vec<u8>> {
        let attr = self
            .node
            .get_attribute(key)
            .ok_or_else(|| Error::Attribute(format!("Missing attribute: {}", key)))?;

        match attr {
            AttributeValue::Tensor(bytes) => Ok(bytes.clone()),
            _ => Err(Error::Attribute(format!(
                "Attribute {} is not a tensor",
                key
            ))),
        }
    }

    /// Encode a symbolic dimension as an immediate value.
    ///
    /// For dimensions that remain symbolic after resolution, this encodes them
    /// as runtime-patchable immediate values and records a binding so they can
    /// be updated by `PlanExecutor::update_dimensions()`.
    pub fn encode_dim_as_immediate(
        &mut self,
        shader_index: ShaderIndex,
        dim: &SymbolicDim,
        immediates: &mut Vec<u8>,
    ) -> Result<usize> {
        use crate::symbolic_expr::evaluate_expr;

        let offset = immediates.len();

        match dim {
            SymbolicDim::Fixed(value) => {
                immediates.extend_from_slice(&(*value as u32).to_le_bytes());
            }
            SymbolicDim::Expr(expr) => {
                let value = evaluate_expr(expr, self.dynamic_dimensions).map_err(|e| {
                    Error::Planning(format!("Failed to evaluate dimension expression: {}", e))
                })?;

                immediates.extend_from_slice(&(value as u32).to_le_bytes());

                self.symbolic_bindings.push(crate::plan::SymbolicBinding {
                    shader_index,
                    immediate_offset: offset,
                    expr: expr.clone(),
                });
            }
        }

        Ok(offset)
    }

    /// Evaluate a symbolic dimension to a concrete value.
    pub fn evaluate_dim(&self, dim: &SymbolicDim) -> Result<usize> {
        use crate::symbolic_expr::evaluate_expr;

        match dim {
            SymbolicDim::Fixed(value) => Ok(*value),
            SymbolicDim::Expr(expr) => evaluate_expr(expr, self.dynamic_dimensions).map_err(|e| {
                Error::Planning(format!("Failed to evaluate dimension expression: {}", e))
            }),
        }
    }

    /// Get the number of inputs to this node.
    pub fn input_count(&self) -> usize {
        self.node.inputs().len()
    }

    /// Get the constant value of an input tensor, if available.
    ///
    /// Returns `None` if the input doesn't exist or isn't a constant.
    pub fn input_value(&self, index: usize) -> Option<&TensorValue> {
        let edge_id = self.node.inputs().get(index)?;
        let edge = self.graph.edge(*edge_id).ok()?;
        edge.constant_value()
    }
}

// ──────────────────────────────── Helpers ────────────────────────────────

/// Format a small tensor value for display in error messages.
///
/// Returns `None` if the tensor is too large (more than `max_elements`).
/// Arrays are formatted as `[1, 2, 3, ...]` with `...` if truncated.
fn format_small_value(value: &TensorValue, max_elements: usize) -> Option<String> {
    let total_elements = value.total_elements();
    if total_elements == 0 {
        return Some("[]".to_string());
    }

    if total_elements > max_elements {
        return None;
    }

    // Format based on data type
    match &value.data {
        TensorData::I64(vec) => Some(format_vec(vec, max_elements)),
        TensorData::I32(vec) => Some(format_vec(vec, max_elements)),
        TensorData::F32(vec) => Some(format_vec(vec, max_elements)),
        TensorData::Bool(vec) => Some(format_vec(vec, max_elements)),
        TensorData::U8(vec) => Some(format_vec(vec, max_elements)),
    }
}

/// Format a slice of values, truncating if necessary.
fn format_vec<T: std::fmt::Debug>(vec: &[T], max_elements: usize) -> String {
    if vec.is_empty() {
        return "[]".to_string();
    }

    if vec.len() <= max_elements {
        format!("{:?}", vec)
    } else {
        let preview: Vec<_> = vec.iter().take(max_elements).collect();
        format!("{:?}...", preview)
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::ir::{IrEdge, IrGraph, IrNode};
    use crate::types::{DataType, TensorData, TensorShape, TensorValue};

    #[test]
    fn test_inference_ctx_input_shape() {
        let mut graph = IrGraph::new();

        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![1, 2, 3]),
        ));

        let mut node = IrNode::new("Relu".to_string());
        node.add_input(input_id);

        let ctx = InferenceCtx::new(&node, &graph);

        let shape = ctx.input_shape(0).unwrap();
        assert_eq!(shape.as_static(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_inference_ctx_input_shape_from_constant() {
        let mut graph = IrGraph::new();

        // Create an edge with a constant value (as if folded)
        let edge = IrEdge::with_constant(
            "folded".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
            TensorValue::new(TensorData::F32(vec![1.0; 6]), vec![2, 3], DataType::F32),
        );
        let edge_id = graph.add_edge(edge);

        let mut node = IrNode::new("Consumer".to_string());
        node.add_input(edge_id);

        let ctx = InferenceCtx::new(&node, &graph);

        let shape = ctx.input_shape(0).unwrap();
        assert_eq!(shape.as_static(), Some(&[2, 3][..]));
    }

    #[test]
    fn test_inference_ctx_require_static() {
        let mut graph = IrGraph::new();

        let input_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        ));

        let mut node = IrNode::new("Test".to_string());
        node.add_input(input_id);

        let ctx = InferenceCtx::new(&node, &graph);

        let dims = ctx.require_static(0).unwrap();
        assert_eq!(dims, &[2, 3]);
    }

    #[test]
    fn test_inference_ctx_attributes() {
        let mut node = IrNode::new("Test".to_string());
        node.set_attribute("axis".to_string(), AttributeValue::Int(1))
            .unwrap();
        node.set_attribute("epsilon".to_string(), AttributeValue::Float(1e-5))
            .unwrap();
        node.set_attribute("perm".to_string(), AttributeValue::Ints(vec![0, 2, 1]))
            .unwrap();

        let graph = IrGraph::new();
        let ctx = InferenceCtx::new(&node, &graph);

        assert_eq!(ctx.attr_i64("axis").unwrap(), 1);
        assert!((ctx.attr_f32("epsilon").unwrap() - 1e-5).abs() < 1e-9);
        assert_eq!(ctx.attr_ints("perm").unwrap(), &[0, 2, 1]);
    }

    #[test]
    fn test_fold_ctx_binary_fold() {
        let mut graph = IrGraph::new();

        // Create edges with constant values (as if previously folded)
        let edge_a = IrEdge::with_constant(
            "a".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorValue::new(TensorData::F32(vec![1.0, 2.0]), vec![2], DataType::F32),
        );
        let edge_a_id = graph.add_edge(edge_a);

        let edge_b = IrEdge::with_constant(
            "b".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            TensorValue::new(TensorData::F32(vec![3.0, 4.0]), vec![2], DataType::F32),
        );
        let edge_b_id = graph.add_edge(edge_b);

        let mut node = IrNode::new("Add".to_string());
        node.add_input(edge_a_id);
        node.add_input(edge_b_id);

        let ctx = FoldCtx::new(&node, &graph);

        let result = ctx.binary_fold_f32(|a, b| a + b).unwrap();
        assert_eq!(result.len(), 1);

        let value = result[0].as_ref().unwrap();
        assert_eq!(value.as_f32(), Some(&[4.0, 6.0][..]));
    }

    #[test]
    fn test_fold_ctx_from_initializer() {
        let mut graph = IrGraph::new();

        // Create an edge with an initializer (raw bytes for [1.0f32, 2.0f32])
        let edge = IrEdge::with_initializer(
            "weight".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2]),
            vec![0, 0, 128, 63, 0, 0, 0, 64], // 1.0f32, 2.0f32
        );
        let edge_id = graph.add_edge(edge);

        let mut node = IrNode::new("Consumer".to_string());
        node.add_input(edge_id);

        let ctx = FoldCtx::new(&node, &graph);

        assert!(ctx.all_inputs_have_values());
    }

    #[test]
    fn test_plan_ctx_input_always_works() {
        let mut graph = IrGraph::new();

        let edge_id = graph.add_edge(IrEdge::new(
            "input".to_string(),
            DataType::F32,
            TensorShape::Static(vec![2, 3]),
        ));

        let mut node = IrNode::new("Relu".to_string());
        node.add_input(edge_id);

        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut scratch_buffers = Vec::new();
        let dynamic_dimensions = HashMap::new();
        let mut symbolic_bindings = Vec::new();

        let ctx = PlanCtx {
            node: &node,
            graph: &graph,
            shaders: &mut shaders,
            shader_cache: &mut shader_cache,
            scratch_buffers: &mut scratch_buffers,
            dynamic_dimensions: &dynamic_dimensions,
            symbolic_bindings: &mut symbolic_bindings,
        };

        // This now always works — no more ValueNode errors
        let buf = ctx.input(0).unwrap();
        assert_eq!(buf, BufferRef::Tensor(edge_id));

        let tensor = ctx.input_tensor(0).unwrap();
        assert_eq!(tensor.name, "input");
    }

    #[test]
    fn test_plan_ctx_encode_dim_as_immediate_fixed() {
        let graph = IrGraph::new();
        let node = IrNode::new("Test".to_string());

        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut scratch_buffers = Vec::new();
        let dynamic_dimensions = HashMap::new();
        let mut symbolic_bindings = Vec::new();

        let mut ctx = PlanCtx {
            node: &node,
            graph: &graph,
            shaders: &mut shaders,
            shader_cache: &mut shader_cache,
            scratch_buffers: &mut scratch_buffers,
            dynamic_dimensions: &dynamic_dimensions,
            symbolic_bindings: &mut symbolic_bindings,
        };

        let dim = SymbolicDim::Fixed(128);
        let mut immediates = Vec::new();
        let offset = ctx
            .encode_dim_as_immediate(0, &dim, &mut immediates)
            .unwrap();

        assert_eq!(offset, 0);
        assert_eq!(immediates.len(), 4);
        assert_eq!(
            u32::from_le_bytes([immediates[0], immediates[1], immediates[2], immediates[3]]),
            128
        );
        assert_eq!(symbolic_bindings.len(), 0);
    }

    #[test]
    fn test_plan_ctx_encode_dim_as_immediate_expr() {
        use crate::symbolic_expr::SymbolicExpr;

        let graph = IrGraph::new();
        let node = IrNode::new("Test".to_string());

        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut scratch_buffers = Vec::new();
        let mut dynamic_dimensions = HashMap::new();
        dynamic_dimensions.insert("seq_len".to_string(), 256);
        let mut symbolic_bindings = Vec::new();

        let mut ctx = PlanCtx {
            node: &node,
            graph: &graph,
            shaders: &mut shaders,
            shader_cache: &mut shader_cache,
            scratch_buffers: &mut scratch_buffers,
            dynamic_dimensions: &dynamic_dimensions,
            symbolic_bindings: &mut symbolic_bindings,
        };

        let expr = SymbolicExpr::Variable("seq_len".to_string());
        let dim = SymbolicDim::Expr(expr.clone());
        let mut immediates = Vec::new();
        let offset = ctx
            .encode_dim_as_immediate(0, &dim, &mut immediates)
            .unwrap();

        assert_eq!(offset, 0);
        assert_eq!(immediates.len(), 4);
        assert_eq!(
            u32::from_le_bytes([immediates[0], immediates[1], immediates[2], immediates[3]]),
            256
        );

        assert_eq!(symbolic_bindings.len(), 1);
        assert_eq!(symbolic_bindings[0].shader_index, 0);
        assert_eq!(symbolic_bindings[0].immediate_offset, 0);
        assert_eq!(symbolic_bindings[0].expr, expr);
    }

    #[test]
    fn test_plan_ctx_evaluate_dim() {
        let graph = IrGraph::new();
        let node = IrNode::new("Test".to_string());

        let mut shaders = Vec::new();
        let mut shader_cache = HashMap::new();
        let mut scratch_buffers = Vec::new();
        let mut dynamic_dimensions = HashMap::new();
        dynamic_dimensions.insert("batch_size".to_string(), 4);
        let mut symbolic_bindings = Vec::new();

        let ctx = PlanCtx {
            node: &node,
            graph: &graph,
            shaders: &mut shaders,
            shader_cache: &mut shader_cache,
            scratch_buffers: &mut scratch_buffers,
            dynamic_dimensions: &dynamic_dimensions,
            symbolic_bindings: &mut symbolic_bindings,
        };

        let fixed_dim = SymbolicDim::Fixed(64);
        assert_eq!(ctx.evaluate_dim(&fixed_dim).unwrap(), 64);

        use crate::symbolic_expr::SymbolicExpr;
        let expr_dim = SymbolicDim::Expr(SymbolicExpr::Variable("batch_size".to_string()));
        assert_eq!(ctx.evaluate_dim(&expr_dim).unwrap(), 4);
    }
}
