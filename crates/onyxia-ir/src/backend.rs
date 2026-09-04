//! The backend contract.
//!
//! A backend consumes an IR [`Module`] and produces a [`Session`] that
//! executes it. Preparation legalizes the module
//! ([`crate::decomp::inline_composites`] with the backend's kernel-registry
//! membership as the `supports` predicate), validates it, fixes a
//! topological order, derives liveness, and uploads constants.
//!
//! **Execution model: an interpreter over the IR, with caches.** There is
//! no separately compiled artifact. Each `run` binds the module's symbols
//! from the input shapes, evaluates every value's shape once, then walks
//! the nodes in order: for each one it selects a kernel for the now
//! concrete shapes, packs the parameters, and dispatches. Pipelines, bind
//! groups, parameter buffers, and device buffers are caches filled on
//! first use, so a steady-state run (a decode step) mostly hits them.
//! Kernel *choice* depends on bound shapes (matvec vs tiled matmul on
//! `M`, split-K factor on `N`/`K`, vectorization on alignment), which is
//! why selection happens per run rather than at `prepare`; a plan cached
//! per shape signature is a possible future optimization, not the
//! current design.
//!
//! Sessions speak **device-resident tensors**: `run` consumes and returns
//! device handles, and moving data across the host boundary is explicit
//! (`upload`/`download`). This is the general mechanism that lets callers
//! keep iterative state (KV caches, diffusion latents) on-device without
//! onyxia knowing anything about the use case.
//!
//! `run`/`download` are async because WebGPU readback cannot block the
//! browser event loop. Native callers can wrap with a blocking executor
//! such as `pollster`.

use crate::Result;
use crate::graph::Module;
use crate::interp::Tensor;

/// A backend: turns modules into executable sessions.
pub trait Backend {
    /// The session type this backend produces.
    type Session: Session;

    /// Whether this backend has a hand-written kernel for the named
    /// composite. Drives legalization: composites without kernels are
    /// inlined through their decompositions.
    fn supports(&self, composite: &str) -> bool;

    /// Legalize, order, and derive liveness for `module`; upload its
    /// constants; return a session that interprets it (see module docs).
    fn prepare(&self, module: Module) -> Result<Self::Session>;
}

/// A prepared, runnable model instance.
#[async_trait::async_trait(?Send)]
pub trait Session {
    /// Device-resident tensor handle. Cheap to clone; an output handle from
    /// one `run` may be passed as an input to a later `run`.
    type Tensor: Clone;

    /// Move a host tensor onto the device.
    fn upload(&mut self, tensor: &Tensor) -> Result<Self::Tensor>;

    /// Execute the model. Inputs are named per the module signature;
    /// returns all module outputs, on-device, in signature order.
    async fn run(&mut self, inputs: &[(&str, Self::Tensor)])
    -> Result<Vec<(String, Self::Tensor)>>;

    /// Move a device tensor back to the host.
    async fn download(&mut self, tensor: &Self::Tensor) -> Result<Tensor>;
}
