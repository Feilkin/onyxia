//! Binary elementwise operator family.
//!
//! Covers: Add, Mul

use onyxia_core::{
    CompileCtx, DispatchCtx, OpDispatch, Operator, Result, RuntimeTensor, broadcast_shape,
};
use std::collections::HashMap;

/// Shader source for the Add operator.
const ADD_SHADER: &str = include_str!("../../shaders/elementwise/add.wgsl");

/// Shader source for the Mul operator.
const MUL_SHADER: &str = include_str!("../../shaders/elementwise/mul.wgsl");

/// Binary elementwise operator family.
///
/// All binary elementwise operations share the same structure:
/// - NumPy-style broadcasting for shape inference
/// - Element-by-element computation for constant folding
/// - WGSL shader dispatch for GPU execution
///
/// The only differences are:
/// - Shader source code (which WGSL function to call)
/// - Fold functions (which CPU operations to perform)
pub struct BinaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
}

impl BinaryElementwiseOp {
    /// Create an Add operator.
    pub fn add() -> Self {
        Self {
            name: "Add",
            shader_source: ADD_SHADER,
        }
    }

    /// Create a Mul operator.
    pub fn mul() -> Self {
        Self {
            name: "Mul",
            shader_source: MUL_SHADER,
        }
    }
}

/// Runtime dispatch for binary elementwise operations (Add, Mul).
struct BinaryElementwiseDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl OpDispatch for BinaryElementwiseDispatch {
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

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, a.dtype)?;

        // Encode immediates (must match ImmediateConstants struct in shader)
        let mut immediates = Vec::with_capacity(12);
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates.extend_from_slice(&(a_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(b_size as u32).to_le_bytes());

        // Compute workgroups
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group with input/output buffers (using the layout reference)
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
            label: Some("binary_elementwise_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(&pipeline, &bind_group, [num_workgroups, 1, 1])?;

        Ok(vec![output])
    }
}

impl Operator for BinaryElementwiseOp {
    fn name(&self) -> &str {
        self.name
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        Ok(Box::new(BinaryElementwiseDispatch {
            module,
            label: self.name.to_string(),
        }))
    }
}
