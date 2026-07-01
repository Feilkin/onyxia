//! wgpu backend for Onyxia.
//!
//! Executes IR modules as WGSL compute shaders on WebGPU/Vulkan/Metal/DX12.
//! v1 executes the primitive set with generated one-thread-per-element
//! kernels (correctness-first, differential-tested against the reference
//! interpreter); fused composite kernels are the designated follow-up via
//! `Backend::supports` + a kernel registry.
//!
//! Backend-private layout decisions: logical `I64` is stored as `i32` on
//! device (range-checked at upload), `Bool` as `u32`. Immediates (push
//! constants) carry all shape parameters; the pipeline layout is built by
//! reflecting shader bindings from the naga module (the wgpu-29 immediates
//! fix — see `gpu.rs`).

pub mod gpu;
pub mod kernels;
pub mod session;

pub use gpu::GpuContext;
pub use session::{GpuTensor, WgpuBackend, WgpuSession};
