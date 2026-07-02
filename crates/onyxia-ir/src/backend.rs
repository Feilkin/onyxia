//! The backend contract (`doc/ir-design.md` §4).
//!
//! A backend consumes an IR [`Module`] and produces a [`Session`] that
//! executes it. Preparation typically legalizes the module first
//! ([`crate::decomp::inline_composites`] with the backend's kernel-registry
//! membership as the `supports` predicate), then selects kernels, plans
//! memory, and compiles pipelines.
//!
//! Sessions speak **device-resident tensors**: `run` consumes and returns
//! device handles, and moving data across the host boundary is explicit
//! (`upload`/`download`). This is the general mechanism that lets callers
//! keep iterative state (KV caches, diffusion latents) on-device without
//! onyxia knowing anything about the use case.
//!
//! `run`/`download` are async for the same reason the interpreter's
//! ancestors were (see the web-async design notes): WebGPU readback cannot
//! block the browser event loop. Native callers can wrap with a blocking
//! executor.

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

    /// Legalize, plan, and compile `module` into a runnable session.
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
