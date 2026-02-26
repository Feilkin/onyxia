//! Dispatch-based model executor.
//!
//! Executes a compiled model by dispatching operations through the
//! register-based tensor routing system.

use crate::error::{Result, RuntimeError};
use crate::tensor::Tensor;
use onyxia_core::dispatch::{CompiledModel, DispatchCtx, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{Level, instrument, span, trace_span};

/// Executes a compiled model using dispatch-based operation routing.
///
/// The executor maintains a register file of `RuntimeTensor` slots.
/// Weight tensors are uploaded at load time, user inputs are written
/// before each execution, and operations read/write registers as they
/// dispatch.
pub struct DispatchExecutor {
    /// GPU dispatch context (device, queue, pipeline cache).
    ctx: DispatchCtx,

    /// The compiled model with dispatch entries and routing info.
    model: CompiledModel,

    /// Register file: tensor slots indexed by register number.
    /// `None` means the register hasn't been written yet.
    registers: Vec<Option<RuntimeTensor>>,
}

impl DispatchExecutor {
    /// Create a new executor and upload weight data to GPU.
    #[instrument(name = "DispatchExecutor::new", skip(device, queue, model))]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: CompiledModel,
    ) -> Result<Self> {
        let ctx = DispatchCtx::new(device, queue);
        let mut registers = vec![None; model.num_registers];

        // Upload weights to GPU registers
        for weight in &model.weight_registers {
            if weight.register >= registers.len() {
                return Err(RuntimeError::ExecutionError(format!(
                    "Weight register {} is out of bounds (num_registers={})",
                    weight.register,
                    registers.len()
                )));
            }

            let tensor = ctx
                .upload_tensor(&weight.data, &weight.shape, weight.dtype)
                .map_err(|e| {
                    RuntimeError::ExecutionError(format!(
                        "Failed to upload weight to register {}: {e}",
                        weight.register
                    ))
                })?;
            registers[weight.register] = Some(tensor);
        }

        Ok(Self {
            ctx,
            model,
            registers,
        })
    }

    /// Execute the model with named inputs, returning named outputs.
    #[instrument(name = "DispatchExecutor::run", skip(self, inputs))]
    pub fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        // Upload user inputs to registers
        for (name, tensor) in inputs {
            let register = self
                .model
                .input_registers
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, reg)| *reg)
                .ok_or_else(|| {
                    RuntimeError::TensorNotFound(format!("Input '{name}' not found in model"))
                })?;

            let runtime_tensor = self
                .ctx
                .upload_tensor(tensor.as_bytes()?, tensor.shape(), tensor.dtype())
                .map_err(|e| {
                    RuntimeError::ExecutionError(format!("Failed to upload input '{name}': {e}"))
                })?;

            self.registers[register] = Some(runtime_tensor);
        }

        // Dispatch all operations in order
        for entry in &self.model.entries {
            let _span = span!(Level::TRACE, "dispatch_op", op = %entry.name).entered();

            // Gather inputs from registers
            let _input_span = trace_span!("gather inputs").entered();
            let op_inputs: Vec<RuntimeTensor> = entry
                .input_regs
                .iter()
                .map(|&reg| {
                    self.registers[reg].clone().ok_or_else(|| {
                        RuntimeError::ExecutionError(format!(
                            "Register {reg} is empty when dispatching '{}' — \
                             an upstream operation may have failed or the graph \
                             has incorrect routing",
                            entry.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            drop(_input_span);

            // Dispatch the operation — try new pre-allocated path first.
            let input_refs: Vec<&RuntimeTensor> = op_inputs.iter().collect();
            let resolved = entry.op.resolve_shapes(&input_refs).map_err(|e| {
                RuntimeError::ExecutionError(format!(
                    "Operation '{}' resolve_shapes failed: {e}",
                    entry.name
                ))
            })?;

            let _output_span = trace_span!("store outputs").entered();
            if let Some(shapes) = resolved {
                // New path: runtime pre-allocates output buffers.
                if shapes.output_shapes.len() != entry.output_regs.len() {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Operation '{}' resolve_shapes returned {} shapes but expected {}",
                        entry.name,
                        shapes.output_shapes.len(),
                        entry.output_regs.len()
                    )));
                }
                let outputs: Vec<RuntimeTensor> = shapes
                    .output_shapes
                    .iter()
                    .zip(&shapes.output_dtypes)
                    .map(|(shape, dtype)| self.ctx.create_output_tensor(shape, *dtype))
                    .collect::<onyxia_core::Result<_>>()
                    .map_err(|e| {
                        RuntimeError::ExecutionError(format!(
                            "Operation '{}' output allocation failed: {e}",
                            entry.name
                        ))
                    })?;
                entry
                    .op
                    .dispatch_with_outputs(op_inputs, outputs.clone(), &mut self.ctx)
                    .map_err(|e| {
                        RuntimeError::ExecutionError(format!(
                            "Operation '{}' dispatch_with_outputs failed: {e}",
                            entry.name
                        ))
                    })?;
                for (output, &reg) in outputs.into_iter().zip(&entry.output_regs) {
                    self.registers[reg] = Some(output);
                }
            } else {
                // Legacy path: operator allocates its own output buffers.
                let outputs = entry.op.dispatch(op_inputs, &mut self.ctx).map_err(|e| {
                    RuntimeError::ExecutionError(format!("Operation '{}' failed: {e}", entry.name))
                })?;
                if outputs.len() != entry.output_regs.len() {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Operation '{}' returned {} outputs but expected {}",
                        entry.name,
                        outputs.len(),
                        entry.output_regs.len()
                    )));
                }
                for (output, &reg) in outputs.into_iter().zip(&entry.output_regs) {
                    self.registers[reg] = Some(output);
                }
            }
            drop(_output_span);
        }

        // Submit all pending GPU commands and wait for completion
        let _flush_span = span!(Level::TRACE, "submit_commands").entered();
        self.ctx.submit_commands().map_err(|e| {
            RuntimeError::ExecutionError(format!("Failed to submit GPU commands: {e}"))
        })?;
        drop(_flush_span);

        // Download outputs
        let mut results = HashMap::new();
        for (name, reg) in &self.model.output_registers {
            let runtime_tensor = self.registers[*reg].as_ref().ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Output register {reg} for '{name}' is empty"))
            })?;

            let data = self.ctx.download_tensor(runtime_tensor).map_err(|e| {
                RuntimeError::ExecutionError(format!("Failed to download tensor '{name}': {e}"))
            })?;
            let tensor = Tensor::from_raw(data, &runtime_tensor.shape, runtime_tensor.dtype);
            results.insert(name.clone(), tensor);
        }

        Ok(results)
    }

    /// Get the model's input register names.
    pub fn input_names(&self) -> Vec<&str> {
        self.model
            .input_registers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get the model's output register names.
    pub fn output_names(&self) -> Vec<&str> {
        self.model
            .output_registers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_core::{
        DataType, DispatchCtx, ModelMetadata, OpDispatch, ResolvedShapes, RuntimeTensor,
        dispatch::{CompiledModel, DispatchEntry},
    };

    // ─── Mock operators ──────────────────────────────────────────────────────

    /// Legacy op: only implements `dispatch()`.
    /// Copies the first input buffer to a fresh output tensor.
    struct LegacyPassthroughOp;

    impl OpDispatch for LegacyPassthroughOp {
        fn dispatch(
            &self,
            inputs: Vec<RuntimeTensor>,
            ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
            let input = &inputs[0];
            let output = ctx.create_output_tensor(&input.shape, input.dtype)?;
            ctx.copy_buffer(&input.buffer, 0, &output.buffer, 0, input.size_bytes as u64)?;
            Ok(vec![output])
        }
    }

    /// New-path op: implements `resolve_shapes()` + `dispatch_with_outputs()`.
    /// Copies the first input into the pre-allocated output buffer.
    struct PreAllocPassthroughOp;

    impl OpDispatch for PreAllocPassthroughOp {
        fn resolve_shapes(
            &self,
            inputs: &[&RuntimeTensor],
        ) -> onyxia_core::Result<Option<ResolvedShapes>> {
            let input = inputs[0];
            Ok(Some(ResolvedShapes {
                output_shapes: vec![input.shape.clone()],
                output_dtypes: vec![input.dtype],
            }))
        }

        fn dispatch_with_outputs(
            &self,
            inputs: Vec<RuntimeTensor>,
            outputs: Vec<RuntimeTensor>,
            ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<()> {
            ctx.copy_buffer(
                &inputs[0].buffer,
                0,
                &outputs[0].buffer,
                0,
                inputs[0].size_bytes as u64,
            )
        }

        fn dispatch(
            &self,
            _inputs: Vec<RuntimeTensor>,
            _ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
            panic!("Legacy dispatch() must not be called on PreAllocPassthroughOp");
        }
    }

    /// Op whose `resolve_shapes()` returns a wrong number of shapes (more than
    /// output registers), which should trigger a shape mismatch error.
    struct BadShapeMockOp;

    impl OpDispatch for BadShapeMockOp {
        fn resolve_shapes(
            &self,
            inputs: &[&RuntimeTensor],
        ) -> onyxia_core::Result<Option<ResolvedShapes>> {
            let input = inputs[0];
            // Return 2 shapes but there is only 1 output register in the test model.
            Ok(Some(ResolvedShapes {
                output_shapes: vec![input.shape.clone(), vec![1]],
                output_dtypes: vec![input.dtype, input.dtype],
            }))
        }

        fn dispatch_with_outputs(
            &self,
            _inputs: Vec<RuntimeTensor>,
            _outputs: Vec<RuntimeTensor>,
            _ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<()> {
            Ok(())
        }

        fn dispatch(
            &self,
            _inputs: Vec<RuntimeTensor>,
            _ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<Vec<RuntimeTensor>> {
            unreachable!()
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /// Build a trivial `CompiledModel` with a single passthrough op.
    /// Register layout: 0 = input, 1 = output.
    fn make_passthrough_model(op: Box<dyn OpDispatch>) -> CompiledModel {
        CompiledModel {
            entries: vec![DispatchEntry {
                op,
                input_regs: vec![0],
                output_regs: vec![1],
                name: "passthrough".to_string(),
                node_name: String::new(),
            }],
            num_registers: 2,
            input_registers: vec![("input".to_string(), 0)],
            output_registers: vec![("output".to_string(), 1)],
            weight_registers: vec![],
            metadata: ModelMetadata::default(),
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────────────

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_legacy_dispatch_path() {
        let gpu = onyxia_core::GpuContext::new()
            .await
            .expect("GPU required for this test");

        let model = make_passthrough_model(Box::new(LegacyPassthroughOp));
        let mut executor =
            DispatchExecutor::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), model)
                .expect("Executor creation should succeed");

        // Input: 4 × f32 values [1.0, 2.0, 3.0, 4.0]
        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = crate::tensor::Tensor::from_raw(input_bytes, &[4], DataType::F32);

        let outputs = executor
            .run(&[("input", tensor)])
            .expect("Execution should succeed");

        let output = outputs.get("output").expect("Output should be present");
        let result: Vec<f32> = output
            .as_bytes()
            .expect("bytes")
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(result, input_data, "Legacy path should copy input → output");
    }

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_pre_alloc_dispatch_path() {
        let gpu = onyxia_core::GpuContext::new()
            .await
            .expect("GPU required for this test");

        let model = make_passthrough_model(Box::new(PreAllocPassthroughOp));
        let mut executor =
            DispatchExecutor::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), model)
                .expect("Executor creation should succeed");

        // Input: 4 × f32 values [5.0, 6.0, 7.0, 8.0]
        let input_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = crate::tensor::Tensor::from_raw(input_bytes, &[4], DataType::F32);

        let outputs = executor
            .run(&[("input", tensor)])
            .expect("Execution should succeed");

        let output = outputs.get("output").expect("Output should be present");
        let result: Vec<f32> = output
            .as_bytes()
            .expect("bytes")
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            result, input_data,
            "Pre-alloc path should copy input → output"
        );
    }

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_resolve_shapes_none_falls_back_to_legacy() {
        // A mock with the default resolve_shapes (returns None) should use the
        // legacy dispatch() path, same as LegacyPassthroughOp.
        let gpu = onyxia_core::GpuContext::new()
            .await
            .expect("GPU required for this test");

        let model = make_passthrough_model(Box::new(LegacyPassthroughOp));
        let mut executor =
            DispatchExecutor::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), model)
                .expect("Executor creation should succeed");

        let input_data: Vec<f32> = vec![9.0, 10.0];
        let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = crate::tensor::Tensor::from_raw(input_bytes, &[2], DataType::F32);

        let outputs = executor
            .run(&[("input", tensor)])
            .expect("Execution should succeed");

        let output = outputs.get("output").expect("Output should be present");
        let result: Vec<f32> = output
            .as_bytes()
            .expect("bytes")
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(result, input_data, "Fallback to legacy path should work");
    }

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_shape_mismatch_error() {
        let gpu = onyxia_core::GpuContext::new()
            .await
            .expect("GPU required for this test");

        let model = make_passthrough_model(Box::new(BadShapeMockOp));
        let mut executor =
            DispatchExecutor::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), model)
                .expect("Executor creation should succeed");

        let input_bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let tensor = crate::tensor::Tensor::from_raw(input_bytes, &[1], DataType::F32);

        let result = executor.run(&[("input", tensor)]);
        assert!(
            result.is_err(),
            "Should fail due to shape count mismatch from resolve_shapes"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("resolve_shapes returned"),
            "Error should mention resolve_shapes mismatch, got: {err_msg}"
        );
    }
}
