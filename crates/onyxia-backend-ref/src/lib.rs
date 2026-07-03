//! Reference CPU backend: the [`onyxia_ir::interp`] interpreter behind the
//! [`Backend`]/[`Session`] traits.
//!
//! This is the correctness oracle every other backend differential-tests
//! against. It supports no fused kernels —
//! `supports` is always false, so preparation inlines every composite down
//! to primitives — and it is deliberately never optimized.
//!
//! "Device" tensors are host tensors behind an `Rc`; upload and download
//! are (cheap) copies for API symmetry.

use onyxia_ir::interp::{Tensor, eval};
use onyxia_ir::{
    Backend, DecompositionRegistry, Error, Module, Result, Session, inline_composites,
    standard_decompositions, validate::validate,
};
use std::rc::Rc;

/// The reference backend. Holds the decomposition registry used for
/// legalization (defaults to [`standard_decompositions`]).
pub struct RefBackend {
    decompositions: DecompositionRegistry,
}

impl RefBackend {
    /// Reference backend with the standard decompositions.
    pub fn new() -> Self {
        Self {
            decompositions: standard_decompositions(),
        }
    }

    /// Reference backend with a custom decomposition registry (e.g. with
    /// user composites registered).
    pub fn with_decompositions(decompositions: DecompositionRegistry) -> Self {
        Self { decompositions }
    }
}

impl Default for RefBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for RefBackend {
    type Session = RefSession;

    fn supports(&self, _composite: &str) -> bool {
        false // no kernels: everything runs through decompositions
    }

    fn prepare(&self, module: Module) -> Result<Self::Session> {
        let module = inline_composites(module, &self.decompositions, &|_| false)?;
        validate(&module)?;
        Ok(RefSession { module })
    }
}

/// A prepared reference-backend session.
pub struct RefSession {
    module: Module,
}

#[async_trait::async_trait(?Send)]
impl Session for RefSession {
    type Tensor = Rc<Tensor>;

    fn upload(&mut self, tensor: &Tensor) -> Result<Self::Tensor> {
        Ok(Rc::new(tensor.clone()))
    }

    async fn run(
        &mut self,
        inputs: &[(&str, Self::Tensor)],
    ) -> Result<Vec<(String, Self::Tensor)>> {
        let host: Vec<(&str, Tensor)> = inputs.iter().map(|(n, t)| (*n, (**t).clone())).collect();
        let outputs = eval(&self.module, &host)?;
        Ok(outputs.into_iter().map(|(n, t)| (n, Rc::new(t))).collect())
    }

    async fn download(&mut self, tensor: &Self::Tensor) -> Result<Tensor> {
        Ok((**tensor).clone())
    }
}

impl RefSession {
    /// The legalized (pure-primitive) module — useful for inspecting what
    /// preparation produced.
    pub fn module(&self) -> &Module {
        &self.module
    }
}

/// Convenience: prepare + run + download in one call, for tests and tools.
pub fn run_once(module: Module, inputs: &[(&str, Tensor)]) -> Result<Vec<(String, Tensor)>> {
    let mut session = RefBackend::new().prepare(module)?;
    let device_inputs: Vec<(&str, Rc<Tensor>)> = inputs
        .iter()
        .map(|(n, t)| Ok((*n, session.upload(t)?)))
        .collect::<Result<_>>()?;
    let outputs = pollster_block(session.run(&device_inputs))?;
    outputs
        .into_iter()
        .map(|(n, t)| Ok((n, (*t).clone())))
        .collect()
}

/// Minimal single-future executor so this crate doesn't need a runtime dep;
/// the interpreter's futures are always immediately ready.
fn pollster_block<F: std::future::Future>(fut: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(out) => out,
        Poll::Pending => unreachable!("interpreter futures are always ready"),
    }
}

// Silence the unused-import lint for Error, used only in downstream error
// mapping today.
#[allow(unused)]
fn _error_type_is_reexported(e: Error) -> Error {
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use onyxia_ir::{AttrValue, Attrs, DataType, GraphBuilder, TensorType};

    #[test]
    fn prepares_and_runs_composites_via_decomposition() {
        // A Softmax composite: prepare must inline it, run must match a
        // hand-computed softmax.
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[1, 3]));
        let outs = b
            .composite(
                "Softmax",
                Attrs::new().with("axis", AttrValue::Int(1)),
                &[x],
                vec![TensorType::of(DataType::F32, &[1, 3])],
            )
            .unwrap();
        b.output("y", outs[0]);
        let module = b.finish().unwrap();

        let x = Tensor::from_f32(&[0.0, 1.0, 2.0], &[1, 3]).unwrap();
        let outputs = run_once(module, &[("x", x)]).unwrap();
        let got = outputs[0].1.to_f32().unwrap();
        let e: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|v| v.exp()).collect();
        let s: f32 = e.iter().sum();
        for (a, expect) in got.iter().zip(e.iter().map(|v| v / s)) {
            assert!((a - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn device_handles_chain_between_runs() {
        // y = x + 1; feed y back as x — the device-resident round trip that
        // KV-cache reuse relies on.
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[2]));
        let one = b.const_f32(&[1.0, 1.0], &[2]).unwrap();
        let y = b.add(x, one).unwrap();
        b.output("y", y);
        let module = b.finish().unwrap();

        let mut session = RefBackend::new().prepare(module).unwrap();
        let x0 = session
            .upload(&Tensor::from_f32(&[10.0, 20.0], &[2]).unwrap())
            .unwrap();
        let out1 = pollster::block_on(session.run(&[("x", x0)])).unwrap();
        let out2 = pollster::block_on(session.run(&[("x", out1[0].1.clone())])).unwrap();
        let final_ = pollster::block_on(session.download(&out2[0].1)).unwrap();
        assert_eq!(final_.to_f32().unwrap(), vec![12.0, 22.0]);
    }
}
