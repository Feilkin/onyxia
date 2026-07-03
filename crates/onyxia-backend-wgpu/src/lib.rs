//! wgpu backend for Onyxia.
//!
//! Executes IR modules as WGSL compute shaders on WebGPU/Vulkan/Metal/DX12.
//! Primitives run as generated one-thread-per-element kernels
//! (correctness-first, differential-tested against the reference
//! interpreter); composites with a hand-written kernel in the [`fused`]
//! registry execute fused, and the rest inline through their
//! decompositions at prepare time.
//!
//! Backend-private layout decisions: logical `I64` is stored as `i32` on
//! device (range-checked at upload), `Bool` as `u32`. Immediates (push
//! constants) carry all shape parameters; on adapters without push
//! constants (all browsers), parameters bind as a storage buffer instead —
//! see `gpu.rs`.

pub mod fused;
pub mod gpu;
pub mod kernels;
pub mod session;

pub use fused::{CompositeKernel, KernelRegistry};
pub use gpu::GpuContext;
pub use session::{GpuTensor, WgpuBackend, WgpuSession};
