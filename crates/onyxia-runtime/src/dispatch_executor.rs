//! Dispatch-based model executor.
//!
//! Executes a compiled model by dispatching operations through the
//! register-based tensor routing system.

use crate::error::{Result, RuntimeError};
use crate::tensor::Tensor;
use onyxia_core::dispatch::{CompiledModel, DispatchCtx, RuntimeTensor};
use std::collections::HashMap;
use std::sync::Arc;

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
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: CompiledModel,
    ) -> Result<Self> {
        let ctx = DispatchCtx::new(device, queue);
        let mut registers = vec![None; model.num_registers];

        // Upload weights to GPU registers
        for weight in &model.weight_registers {
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
            // Gather inputs from registers
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

            // Dispatch the operation
            let outputs = entry.op.dispatch(op_inputs, &mut self.ctx).map_err(|e| {
                RuntimeError::ExecutionError(format!("Operation '{}' failed: {e}", entry.name))
            })?;

            // Store outputs in registers
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

        // Ensure GPU work completes
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::ExecutionError(format!("GPU poll failed: {e:?}")))?;

        // Download outputs
        let mut results = HashMap::new();
        for (name, reg) in &self.model.output_registers {
            let runtime_tensor = self.registers[*reg].as_ref().ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("Output register {reg} for '{name}' is empty"))
            })?;

            let data = self.download_tensor(runtime_tensor)?;
            let tensor = Tensor::from_raw(data, &runtime_tensor.shape, runtime_tensor.dtype);
            results.insert(name.clone(), tensor);
        }

        Ok(results)
    }

    /// Download tensor data from GPU to CPU.
    fn download_tensor(&self, tensor: &RuntimeTensor) -> Result<Vec<u8>> {
        let buffer_size = tensor.size_bytes as u64;

        // Create a staging buffer for readback
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from tensor buffer to staging
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download_copy"),
            });
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, buffer_size);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| {
                RuntimeError::ExecutionError(format!("GPU poll failed during download: {e:?}"))
            })?;

        receiver
            .recv()
            .map_err(|e| RuntimeError::ExecutionError(format!("Map recv failed: {e}")))?
            .map_err(|e| RuntimeError::ExecutionError(format!("Map failed: {e}")))?;

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();

        Ok(data)
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
