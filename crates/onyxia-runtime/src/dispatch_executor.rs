//! Dispatch-based model executor.
//!
//! Executes a compiled model by dispatching operations through the
//! register-based tensor routing system.

use crate::buffer_pool::BufferPool;
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
///
/// A `BufferPool` tracks freed intermediate tensors and reuses their GPU
/// buffers for subsequent allocations, reducing GPU memory pressure for
/// models with many operations.
pub struct DispatchExecutor {
    /// GPU dispatch context (device, queue, pipeline cache).
    ctx: DispatchCtx,

    /// The compiled model with dispatch entries and routing info.
    model: CompiledModel,

    /// Register file: tensor slots indexed by register number.
    /// `None` means the register hasn't been written yet.
    registers: Vec<Option<RuntimeTensor>>,

    /// Buffer pool for GPU memory reuse.
    pool: BufferPool,
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
            pool: BufferPool::new(),
        })
    }

    /// Execute the model with named inputs, returning named outputs.
    #[instrument(name = "DispatchExecutor::run", skip(self, inputs))]
    pub async fn run(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
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

        // Dispatch all operations in order.
        //
        // We use an index-based loop (rather than iterator) so that scoped
        // borrows of `self.model.entries[i]` can be dropped before we call
        // `free_dead_registers`, which needs `&mut self`.
        let num_entries = self.model.entries.len();
        for entry_idx in 0..num_entries {
            let _span = {
                let name = &self.model.entries[entry_idx].name;
                span!(Level::TRACE, "dispatch_op", op = %name).entered()
            };

            // Gather inputs from registers.  The `entry` borrow is scoped so
            // it ends before any mutable operation on the pool/liveness.
            let op_inputs: Vec<RuntimeTensor> = {
                let _input_span = trace_span!("gather inputs").entered();
                let entry = &self.model.entries[entry_idx];
                entry
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
                    .collect::<Result<Vec<_>>>()?
                // `entry` dropped here.
            };

            // Try new pre-allocated path: resolve output shapes.
            let resolved = {
                let entry = &self.model.entries[entry_idx];
                let input_refs: Vec<&RuntimeTensor> = op_inputs.iter().collect();
                entry.op.resolve_shapes(&input_refs).map_err(|e| {
                    RuntimeError::ExecutionError(format!(
                        "Operation '{}' resolve_shapes failed: {e}",
                        entry.name
                    ))
                })?
                // `entry` dropped here.
            };

            let _output_span = trace_span!("store outputs").entered();
            if let Some(shapes) = resolved {
                // Validate shape/register count (short-lived entry borrow).
                let (expected_count, output_regs) = {
                    let entry = &self.model.entries[entry_idx];
                    (entry.output_regs.len(), entry.output_regs.clone())
                    // `entry` dropped here.
                };
                if shapes.output_shapes.len() != expected_count {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Operation '{}' resolve_shapes returned {} shapes but expected {}",
                        self.model.entries[entry_idx].name,
                        shapes.output_shapes.len(),
                        expected_count
                    )));
                }

                // Allocate output tensors — prefer pooled buffers.
                // No `entry` borrow is active here, so we can access `self.pool`.
                let mut outputs = Vec::with_capacity(shapes.output_shapes.len());
                for (shape, dtype) in shapes.output_shapes.iter().zip(&shapes.output_dtypes) {
                    let num_elements: usize = shape.iter().product();
                    let size_bytes = num_elements * dtype.size();
                    let buffer = self.pool.acquire(size_bytes, &self.ctx.device);
                    outputs.push(RuntimeTensor {
                        buffer,
                        shape: shape.to_vec(),
                        dtype: *dtype,
                        size_bytes,
                    });
                }

                // Dispatch with pre-allocated outputs.
                {
                    let entry = &self.model.entries[entry_idx];
                    entry
                        .op
                        .dispatch_with_outputs(op_inputs, outputs.clone(), &mut self.ctx)
                        .await
                        .map_err(|e| {
                            RuntimeError::ExecutionError(format!(
                                "Operation '{}' dispatch_with_outputs failed: {e}",
                                entry.name
                            ))
                        })?;
                    // `entry` dropped here.
                }
                for (output, &reg) in outputs.into_iter().zip(&output_regs) {
                    self.registers[reg] = Some(output);
                }
            } else {
                // Legacy path: operator allocates its own output buffers.
                let output_regs = self.model.entries[entry_idx].output_regs.clone();
                let outputs = {
                    let entry = &self.model.entries[entry_idx];
                    entry.op.dispatch(op_inputs, &mut self.ctx).await.map_err(|e| {
                        RuntimeError::ExecutionError(format!(
                            "Operation '{}' failed: {e}",
                            entry.name
                        ))
                    })?
                    // `entry` dropped here.
                };
                if outputs.len() != output_regs.len() {
                    return Err(RuntimeError::ExecutionError(format!(
                        "Operation '{}' returned {} outputs but expected {}",
                        self.model.entries[entry_idx].name,
                        outputs.len(),
                        output_regs.len()
                    )));
                }
                for (output, &reg) in outputs.into_iter().zip(&output_regs) {
                    self.registers[reg] = Some(output);
                }
            }
            drop(_output_span);

            // After dispatch: release dead registers to the buffer pool.
            // All borrows of self.model.entries[entry_idx] are out of scope here.
            self.free_dead_registers(entry_idx);
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

            let data = self.ctx.download_tensor(runtime_tensor).await.map_err(|e| {
                RuntimeError::ExecutionError(format!("Failed to download tensor '{name}': {e}"))
            })?;
            let tensor = Tensor::from_raw(data, &runtime_tensor.shape, runtime_tensor.dtype);
            results.insert(name.clone(), tensor);
        }

        Ok(results)
    }

    /// Blocking wrapper around [`run`](Self::run) for native callers.
    ///
    /// Drives the async run to completion on the current thread. Only available
    /// off the web — on wasm there is no blocking executor, so callers must
    /// `.await` [`run`](Self::run) directly.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_blocking(&mut self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        pollster::block_on(self.run(inputs))
    }

    /// Release registers that became dead after `entry_idx` to the buffer pool.
    ///
    /// A register is only released if no other `Arc` clone holds its buffer
    /// (strong count == 1), ensuring we don't steal buffers still in use by
    /// an `OpDispatch` implementation.
    fn free_dead_registers(&mut self, entry_idx: usize) {
        let Some(ref liveness) = self.model.liveness else {
            return;
        };
        // Collect the registers to free to avoid borrowing `self` mutably twice.
        let to_free: Vec<usize> = liveness
            .freed_after
            .get(entry_idx)
            .map(|v| v.clone())
            .unwrap_or_default();
        for reg in to_free {
            if let Some(tensor) = self.registers[reg].take() {
                if Arc::strong_count(&tensor.buffer) == 1 {
                    self.pool.release(tensor.buffer);
                } else {
                    // Other clones still exist — put the tensor back.
                    self.registers[reg] = Some(tensor);
                }
            }
        }
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

    /// Return buffer pool statistics `(allocations, reuses, pool_bytes)`.
    ///
    /// Useful for testing and performance diagnostics. `reuses > 0` after a
    /// second inference pass indicates the pool is working correctly.
    /// `pool_bytes` is the total GPU memory currently held by the pool.
    pub fn pool_stats(&self) -> (usize, usize, usize) {
        (
            self.pool.allocations,
            self.pool.reuses,
            self.pool.pool_bytes,
        )
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

    #[async_trait::async_trait(?Send)]
    impl OpDispatch for LegacyPassthroughOp {
        async fn dispatch(
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

    #[async_trait::async_trait(?Send)]
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

        async fn dispatch_with_outputs(
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

        async fn dispatch(
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

    #[async_trait::async_trait(?Send)]
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

        async fn dispatch_with_outputs(
            &self,
            _inputs: Vec<RuntimeTensor>,
            _outputs: Vec<RuntimeTensor>,
            _ctx: &mut DispatchCtx,
        ) -> onyxia_core::Result<()> {
            Ok(())
        }

        async fn dispatch(
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
            liveness: None,
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
            .run_blocking(&[("input", tensor)])
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
            .run_blocking(&[("input", tensor)])
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
            .run_blocking(&[("input", tensor)])
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

        let result = executor.run_blocking(&[("input", tensor)]);
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

    #[pollster::test]
    #[ignore] // Requires GPU
    async fn test_buffer_pool_reuse_on_second_run() {
        // Build a 2-op chain: reg0 (input) → op0 → reg1 (intermediate) → op1 → reg2 (output).
        // Liveness: reg1 is freed after op1 (its last reader).
        // On the second run, allocating reg1's output should hit the pool.
        use onyxia_core::memory::LivenessInfo;

        let gpu = onyxia_core::GpuContext::new()
            .await
            .expect("GPU required for this test");

        let liveness = LivenessInfo {
            // Entry 0 (op0) produces reg1 — reg0 is last read by op0 → freed after entry 0.
            // Entry 1 (op1) produces reg2 — reg1 is last read by op1 → freed after entry 1.
            freed_after: vec![
                vec![0], // reg0 freed after entry 0 (it's not a model output)
                vec![1], // reg1 freed after entry 1
            ],
        };

        let model = CompiledModel {
            entries: vec![
                DispatchEntry {
                    op: Box::new(PreAllocPassthroughOp),
                    input_regs: vec![0],
                    output_regs: vec![1],
                    name: "op0".to_string(),
                    node_name: String::new(),
                },
                DispatchEntry {
                    op: Box::new(PreAllocPassthroughOp),
                    input_regs: vec![1],
                    output_regs: vec![2],
                    name: "op1".to_string(),
                    node_name: String::new(),
                },
            ],
            num_registers: 3,
            input_registers: vec![("input".to_string(), 0)],
            output_registers: vec![("output".to_string(), 2)],
            weight_registers: vec![],
            metadata: ModelMetadata::default(),
            liveness: Some(liveness),
        };

        let mut executor =
            DispatchExecutor::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue), model)
                .expect("Executor creation should succeed");

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // First run — pool may already reuse within the same run (e.g. reg0
        // freed after op0 can be reused for reg2's allocation in op1).
        let tensor = crate::tensor::Tensor::from_raw(input_bytes.clone(), &[4], DataType::F32);
        executor.run_blocking(&[("input", tensor)]).expect("first run");
        let (allocs_after_first, _reuses_after_first, _) = executor.pool_stats();
        assert!(allocs_after_first > 0, "Should have made allocations");

        // Second run — more buffers should be reused from the pool.
        let tensor2 = crate::tensor::Tensor::from_raw(input_bytes.clone(), &[4], DataType::F32);
        let outputs = executor.run_blocking(&[("input", tensor2)]).expect("second run");

        let (_, reuses_after_second, _) = executor.pool_stats();
        assert!(
            reuses_after_second > 0,
            "Should have reused at least one buffer across both runs"
        );

        // Verify correctness is preserved with pooled buffers.
        let output = outputs.get("output").expect("Output should be present");
        let result: Vec<f32> = output
            .as_bytes()
            .expect("bytes")
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(result, input_data, "Pooled buffers must not corrupt output");
    }
}
