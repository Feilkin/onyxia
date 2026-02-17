//! Unary math operator family.
//!
//! Covers: Neg, Sqrt, Cos, Sin, Tanh

use onyxia_core::{CompileCtx, DispatchCtx, OpDispatch, Operator, Result, RuntimeTensor};
use std::collections::HashMap;

/// Shader source for the Neg operator.
const NEG_SHADER: &str = include_str!("../../shaders/unary_math/neg.wgsl");

/// Shader source for the Sqrt operator.
const SQRT_SHADER: &str = include_str!("../../shaders/unary_math/sqrt.wgsl");

/// Shader source for the Cos operator.
const COS_SHADER: &str = include_str!("../../shaders/unary_math/cos.wgsl");

/// Shader source for the Sin operator.
const SIN_SHADER: &str = include_str!("../../shaders/unary_math/sin.wgsl");

/// Shader source for the Tanh operator.
const TANH_SHADER: &str = include_str!("../../shaders/unary_math/tanh.wgsl");

/// Unary math operator family.
///
/// All unary math operations share the same structure:
/// - Single input tensor
/// - Output shape = input shape (no broadcasting)
/// - Element-by-element computation
/// - WGSL shader dispatch for GPU execution
///
/// The only difference is the shader source code (which WGSL math function to call).
pub struct UnaryMathOp {
    name: &'static str,
    shader_source: &'static str,
}

impl UnaryMathOp {
    /// Create a Neg operator (negation: Y = -X).
    pub fn neg() -> Self {
        Self {
            name: "Neg",
            shader_source: NEG_SHADER,
        }
    }

    /// Create a Sqrt operator (square root: Y = sqrt(X)).
    pub fn sqrt() -> Self {
        Self {
            name: "Sqrt",
            shader_source: SQRT_SHADER,
        }
    }

    /// Create a Cos operator (cosine: Y = cos(X)).
    pub fn cos() -> Self {
        Self {
            name: "Cos",
            shader_source: COS_SHADER,
        }
    }

    /// Create a Sin operator (sine: Y = sin(X)).
    pub fn sin() -> Self {
        Self {
            name: "Sin",
            shader_source: SIN_SHADER,
        }
    }

    /// Create a Tanh operator (hyperbolic tangent: Y = tanh(X)).
    pub fn tanh() -> Self {
        Self {
            name: "Tanh",
            shader_source: TANH_SHADER,
        }
    }
}

/// Runtime dispatch for unary math operations.
struct UnaryMathDispatch {
    /// Pre-compiled naga module for the shader.
    module: naga::Module,

    /// Label for pipeline caching.
    label: String,
}

impl OpDispatch for UnaryMathDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
        let input = &inputs[0];
        let num_elements: usize = input.shape.iter().product();

        // Output has same shape and dtype as input
        let output = ctx.create_output_tensor(&input.shape, input.dtype)?;

        // Encode immediates (must match ImmediateConstants struct in shader)
        let mut immediates = Vec::with_capacity(4);
        immediates.extend_from_slice(&(num_elements as u32).to_le_bytes());

        // Compute workgroups
        let workgroup_size: u32 = 256;
        let num_workgroups = (num_elements as u32).div_ceil(workgroup_size);

        // Get or create pipeline
        let (pipeline, bind_group_layout) =
            ctx.get_or_create_pipeline(&self.label, &self.module, "main")?;

        // Create bind group with input/output buffers
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary_math_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader with immediates
        ctx.dispatch_compute(
            &pipeline,
            &bind_group,
            [num_workgroups, 1, 1],
            Some(&immediates),
        )?;

        Ok(vec![output])
    }
}

impl Operator for UnaryMathOp {
    fn name(&self) -> &str {
        self.name
    }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        let mut shader_defs = HashMap::new();
        shader_defs.insert("WORKGROUP_SIZE".to_string(), "256".to_string());

        let module = ctx.compile_shader(self.name, self.shader_source, &shader_defs)?;

        Ok(Box::new(UnaryMathDispatch {
            module,
            label: self.name.to_string(),
        }))
    }
}
