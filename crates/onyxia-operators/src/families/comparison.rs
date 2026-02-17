//! Comparison operator family.
//!
//! Covers: Equal, Greater, Less, GreaterOrEqual, LessOrEqual

use onyxia_core::{
    CompileCtx, DataType, DispatchCtx, OpDispatch, Operator, Result, RuntimeTensor, broadcast_shape,
};
use std::collections::HashMap;

/// Shader source for the Equal operator.
const EQUAL_SHADER: &str = include_str!("../../shaders/comparison/equal.wgsl");

/// Shader source for the Greater operator.
const GREATER_SHADER: &str = include_str!("../../shaders/comparison/greater.wgsl");

/// Shader source for the Less operator.
const LESS_SHADER: &str = include_str!("../../shaders/comparison/less.wgsl");

/// Shader source for the GreaterOrEqual operator.
const GREATER_OR_EQUAL_SHADER: &str =
    include_str!("../../shaders/comparison/greater_or_equal.wgsl");

/// Shader source for the LessOrEqual operator.
const LESS_OR_EQUAL_SHADER: &str = include_str!("../../shaders/comparison/less_or_equal.wgsl");

/// Comparison operator family.
///
/// All comparison operations share the same structure:
/// - NumPy-style broadcasting for shape inference
/// - Element-by-element comparison
/// - WGSL shader dispatch for GPU execution
/// - Boolean output (stored as u32: 0 = false, 1 = true)
///
/// The only difference is the comparison operation (==, >, <, >=, <=).
pub struct ComparisonOp {
    name: &'static str,
    shader_source: &'static str,
}

impl ComparisonOp {
    /// Create an Equal operator (A == B).
    pub fn equal() -> Self {
        Self {
            name: "Equal",
            shader_source: EQUAL_SHADER,
        }
    }

    /// Create a Greater operator (A > B).
    pub fn greater() -> Self {
        Self {
            name: "Greater",
            shader_source: GREATER_SHADER,
        }
    }

    /// Create a Less operator (A < B).
    pub fn less() -> Self {
        Self {
            name: "Less",
            shader_source: LESS_SHADER,
        }
    }

    /// Create a GreaterOrEqual operator (A >= B).
    pub fn greater_or_equal() -> Self {
        Self {
            name: "GreaterOrEqual",
            shader_source: GREATER_OR_EQUAL_SHADER,
        }
    }

    /// Create a LessOrEqual operator (A <= B).
    pub fn less_or_equal() -> Self {
        Self {
            name: "LessOrEqual",
            shader_source: LESS_OR_EQUAL_SHADER,
        }
    }
}

/// Runtime dispatch for comparison operations.
struct ComparisonDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl OpDispatch for ComparisonDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let a = &inputs[0];
        let b = &inputs[1];

        // Compute broadcast output shape at runtime
        let output_shape = broadcast_shape(&a.shape, &b.shape)?;
        let num_elements: usize = output_shape.iter().product();
        let a_size: usize = a.shape.iter().product();
        let b_size: usize = b.shape.iter().product();

        // Allocate output buffer with Bool dtype
        let output = ctx.create_output_tensor(&output_shape, DataType::Bool)?;

        // Compute workgroups with 2D dispatch support for large tensors
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);
        let (dispatch_size, x_stride) = DispatchCtx::compute_dispatch_size(num_workgroups, workgroup_size);

        // Encode immediates (must match ImmediateConstants struct in shader)
        let mut immediates = Vec::with_capacity(16);
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates.extend_from_slice(&(a_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(b_size as u32).to_le_bytes());
        immediates.extend_from_slice(&x_stride.to_le_bytes());

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group with input/output buffers
        let entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("comparison_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader with immediates
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            dispatch_size,
            Some(&immediates),
        )?;;

        Ok(vec![output])
    }
}

impl Operator for ComparisonOp {
    fn name(&self) -> &str {
        self.name
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        Ok(Box::new(ComparisonDispatch {
            module,
            label: self.name.to_string(),
        }))
    }
}
