//! Max operator family (element-wise maximum).
//!
//! Computes element-wise maximum of N tensors with broadcasting support.

use onyxia_core::{
    CompileCtx, DispatchCtx, Error, OpDispatch, Operator, Result, RuntimeTensor, broadcast_shape,
};
use std::collections::HashMap;

/// Shader source for the Max operator (binary version).
const MAX_SHADER: &str = include_str!("../../shaders/elementwise/max.wgsl");

/// Max operator (element-wise maximum).
///
/// Computes the element-wise maximum of N input tensors with NumPy-style broadcasting.
///
/// **ONNX Specification:**
/// - Opset: 13
/// - Inputs:
///   - data_0, data_1, ..., data_N (T) - List of tensors (variadic, N ≥ 1)
/// - Outputs:
///   - max (T) - Element-wise maximum
/// - Type constraints: T = float16, float, double, uint32, uint64, int32, int64, bfloat16
///
/// **Behavior:**
/// - Max(A) = A (identity)
/// - Max(A, B) = element-wise maximum with broadcasting
/// - Max(A, B, C) = Max(Max(A, B), C) (associative reduction)
///
/// **Implementation:**
/// - For N=1: Identity operation (return input unchanged)
/// - For N=2: Binary element-wise max with broadcasting
/// - For N>2: Iterative binary max operations (Max(Max(..., B), C))
pub struct MaxOp;

impl Operator for MaxOp {
    fn name(&self) -> &str {
        "Max"
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader("Max", MAX_SHADER, &shader_defs)?;

        Ok(Box::new(MaxDispatch {
            module,
            label: "Max".to_string(),
        }))
    }
}

/// Runtime dispatch for Max operation.
struct MaxDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl MaxDispatch {
    /// Perform binary max operation on two tensors.
    fn binary_max(
        &self,
        a: &RuntimeTensor,
        b: &RuntimeTensor,
        ctx: &mut DispatchCtx,
    ) -> Result<RuntimeTensor> {
        // Compute broadcast output shape
        let output_shape = broadcast_shape(&a.shape, &b.shape)?;
        let num_elements: usize = output_shape.iter().product();
        let a_size: usize = a.shape.iter().product();
        let b_size: usize = b.shape.iter().product();

        // Allocate output buffer
        let output = ctx.create_output_tensor(&output_shape, a.dtype)?;

        // Compute workgroups with 2D dispatch support for large tensors
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);
        let (dispatch_size, x_stride) =
            DispatchCtx::compute_dispatch_size(num_workgroups, workgroup_size);

        // Encode immediates
        let mut immediates = Vec::with_capacity(16);
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());
        immediates.extend_from_slice(&(a_size as u32).to_le_bytes());
        immediates.extend_from_slice(&(b_size as u32).to_le_bytes());
        immediates.extend_from_slice(&x_stride.to_le_bytes());

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group
        let entries = vec![
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
            label: Some("max_bind_group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Dispatch compute shader
        ctx.dispatch_compute(&pipeline, &bind_group, dispatch_size, Some(&immediates))?;

        Ok(output)
    }
}

impl OpDispatch for MaxDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        if inputs.is_empty() {
            return Err(Error::Runtime(
                "Max operator requires at least 1 input".into(),
            ));
        }

        // Handle single input (identity)
        if inputs.len() == 1 {
            return Ok(vec![inputs.into_iter().next().unwrap()]);
        }

        // For N inputs, iteratively compute max
        // result = Max(Max(Max(A, B), C), D)...
        let mut result = inputs[0].clone();

        for input in &inputs[1..] {
            result = self.binary_max(&result, input, ctx)?;
        }

        Ok(vec![result])
    }
}
