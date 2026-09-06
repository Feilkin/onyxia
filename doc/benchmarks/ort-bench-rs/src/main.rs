//! onnxruntime benchmark through the C API (via the `ort` crate), mirroring
//! `onyxia bench` and `ort_bench.py` without Python per-step overhead:
//! warmup prefill(P) + 2 decode steps, then a measured prefill(P) and D
//! measured single-token decode steps. Logits are copied to host every
//! step. Modes: `cuda-host` (KV round-trips host, like `run()` callers do)
//! and `cuda-iobinding` (KV stays on device; only logits come back).
//!
//! Models without a `logits` output (embedding models) get the forward
//! protocol instead: D timed stateless passes over P tokens, the last
//! output (the pooled embedding) copied to host every pass.
//!
//! Usage: ort-bench-rs <libonnxruntime.so> <model.onnx> <mode> <P> <D>
use ndarray::{Array2, Array4};
use ort::ep;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, ValueType};
use std::time::Instant;

const DUMMY: i64 = 42;

struct Kv {
    inputs: Vec<String>,  // past_key_values.*
    outputs: Vec<String>, // present.*
    n_kv_heads: usize,
    head_dim: usize,
}

fn kv_layout(session: &Session) -> Kv {
    let mut inputs = Vec::new();
    let (mut h, mut d) = (0usize, 0usize);
    for i in session.inputs() {
        if i.name().starts_with("past_key_values.") {
            inputs.push(i.name().to_string());
            if let ValueType::Tensor { shape, .. } = i.dtype() {
                h = shape[1] as usize;
                d = shape[3] as usize;
            }
        }
    }
    let outputs = session
        .outputs()
        .iter()
        .filter(|o| o.name().starts_with("present."))
        .map(|o| o.name().to_string())
        .collect();
    Kv {
        inputs,
        outputs,
        n_kv_heads: h,
        head_dim: d,
    }
}

fn step_inputs(pos: usize, ids: &[i64], names: &[String]) -> Vec<(String, DynValue)> {
    let s = ids.len();
    let mut v = vec![
        (
            "input_ids".to_string(),
            Tensor::from_array(Array2::from_shape_vec((1, s), ids.to_vec()).unwrap())
                .unwrap()
                .into_dyn(),
        ),
        (
            "position_ids".to_string(),
            Tensor::from_array(Array2::from_shape_vec(
                (1, s),
                (pos as i64..(pos + s) as i64).collect(),
            )
            .unwrap())
            .unwrap()
            .into_dyn(),
        ),
    ];
    if names.iter().any(|n| n == "attention_mask") {
        v.push((
            "attention_mask".to_string(),
            Tensor::from_array(Array2::<i64>::ones((1, pos + s)))
                .unwrap()
                .into_dyn(),
        ));
    }
    v.retain(|(n, _)| names.contains(n));
    v
}

/// Runs a step; returns the last position's logits (on host).
trait Runner {
    fn step(&mut self, ids: &[i64]) -> Vec<f32>;
    fn reset(&mut self);
}

/// Plain `run()`: every output (KV included) lands on the host and is fed
/// back next step — what a straightforward API user does.
struct HostKv {
    session: Session,
    kv: Kv,
    names: Vec<String>,
    pos: usize,
    past: Vec<DynValue>,
}

impl HostKv {
    fn empty_kv(&self) -> Vec<DynValue> {
        self.kv
            .inputs
            .iter()
            .map(|_| {
                Tensor::from_array(Array4::<f32>::zeros((1, self.kv.n_kv_heads, 0, self.kv.head_dim)))
                    .unwrap()
                    .into_dyn()
            })
            .collect()
    }
}

impl Runner for HostKv {
    fn step(&mut self, ids: &[i64]) -> Vec<f32> {
        let s = ids.len();
        let fresh = step_inputs(self.pos, ids, &self.names);
        let mut inputs: Vec<(String, SessionInputValue<'_>)> = fresh
            .iter()
            .map(|(n, v)| (n.clone(), SessionInputValue::from(v)))
            .collect();
        for (name, v) in self.kv.inputs.iter().zip(&self.past) {
            inputs.push((name.clone(), SessionInputValue::from(v)));
        }
        let mut outputs = self.session.run(inputs).unwrap();
        let logits = outputs.remove("logits").unwrap();
        let arr = logits.try_extract_array::<f32>().unwrap();
        let vocab = arr.shape()[2];
        let last: Vec<f32> = arr.as_slice().unwrap()[(s - 1) * vocab..s * vocab].to_vec();
        let present: Vec<DynValue> = self
            .kv
            .outputs
            .iter()
            .map(|n| outputs.remove(n).unwrap())
            .collect();
        drop(outputs);
        self.past = present;
        self.pos += s;
        last
    }
    fn reset(&mut self) {
        self.pos = 0;
        self.past = self.empty_kv();
    }
}

/// IoBinding: present.* outputs are allocated on the CUDA device and bound
/// straight back as next step's past.*; only logits are bound to host.
struct DeviceKv {
    session: Session,
    kv: Kv,
    names: Vec<String>,
    pos: usize,
    past: Vec<DynValue>,
    cuda: MemoryInfo<'static>,
    cpu: MemoryInfo<'static>,
}

impl Runner for DeviceKv {
    fn step(&mut self, ids: &[i64]) -> Vec<f32> {
        let s = ids.len();
        let fresh = step_inputs(self.pos, ids, &self.names);
        let mut binding = self.session.create_binding().unwrap();
        for (n, v) in &fresh {
            binding.bind_input(n.as_str(), v).unwrap();
        }
        for (name, v) in self.kv.inputs.iter().zip(&self.past) {
            binding.bind_input(name.as_str(), v).unwrap();
        }
        for name in &self.kv.outputs {
            binding.bind_output_to_device(name.as_str(), &self.cuda).unwrap();
        }
        binding.bind_output_to_device("logits", &self.cpu).unwrap();
        let mut outputs = self.session.run_binding(&binding).unwrap();
        let logits = outputs.remove("logits").unwrap();
        let arr = logits.try_extract_array::<f32>().unwrap();
        let vocab = arr.shape()[2];
        let last: Vec<f32> = arr.as_slice().unwrap()[(s - 1) * vocab..s * vocab].to_vec();
        let present: Vec<DynValue> = self
            .kv
            .outputs
            .iter()
            .map(|n| outputs.remove(n).unwrap())
            .collect();
        drop(outputs);
        self.past = present;
        self.pos += s;
        last
    }
    fn reset(&mut self) {
        self.pos = 0;
        self.past = self
            .kv
            .inputs
            .iter()
            .map(|_| {
                Tensor::from_array(Array4::<f32>::zeros((1, self.kv.n_kv_heads, 0, self.kv.head_dim)))
                    .unwrap()
                    .into_dyn()
            })
            .collect();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (lib, model, mode) = (&args[1], &args[2], args[3].as_str());
    let p: usize = args[4].parse().unwrap();
    let d: usize = args[5].parse().unwrap();

    assert!(ort::init_from(lib).unwrap().commit());
    let session = Session::builder()
        .unwrap()
        .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
        .unwrap()
        .commit_from_file(model)
        .unwrap();
    if !session.outputs().iter().any(|o| o.name() == "logits") {
        return forward_bench(session, mode, p, d);
    }
    let kv = kv_layout(&session);
    let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    let mut runner: Box<dyn Runner> = match mode {
        "cuda-host" => Box::new(HostKv {
            session,
            kv,
            names: names.clone(),
            pos: 0,
            past: Vec::new(),
        }),
        "cuda-iobinding" => Box::new(DeviceKv {
            session,
            kv,
            names: names.clone(),
            pos: 0,
            past: Vec::new(),
            cuda: MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).unwrap(),
            cpu: MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput).unwrap(),
        }),
        other => panic!("unknown mode {other}"),
    };
    runner.reset();

    let prompt = vec![DUMMY; p];
    // Warmup.
    runner.step(&prompt);
    runner.step(&[DUMMY]);
    runner.step(&[DUMMY]);
    runner.reset();

    let t0 = Instant::now();
    let logits = runner.step(&prompt);
    let prefill_s = t0.elapsed().as_secs_f64();
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    let mut steps = Vec::with_capacity(d);
    for _ in 0..d {
        let t = Instant::now();
        runner.step(&[DUMMY]);
        steps.push(t.elapsed().as_secs_f64());
    }
    let mean = steps.iter().sum::<f64>() / d as f64;
    let min = steps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = steps.iter().cloned().fold(0.0, f64::max);
    let var = steps.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / d as f64;
    println!("onnxruntime (C API via ort crate) mode={mode} argmax@prefill={argmax}");
    println!(
        "prefill: {p} tokens in {:.1} ms ({:.1} tok/s)",
        prefill_s * 1e3,
        p as f64 / prefill_s
    );
    println!(
        "decode:  {d} tokens, {:.2} ms/tok mean (min {:.2}, max {:.2}, σ {:.2}) → {:.2} tok/s",
        mean * 1e3,
        min * 1e3,
        max * 1e3,
        var.sqrt() * 1e3,
        1.0 / mean
    );
}

/// Forward protocol: warmup, then `d` timed passes over `p` tokens.
fn forward_bench(mut session: Session, mode: &str, p: usize, d: usize) {
    let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    let out = session.outputs().last().unwrap().name().to_string();
    let mut inputs: Vec<(String, DynValue)> = vec![(
        "input_ids".to_string(),
        Tensor::from_array(Array2::from_elem((1, p), DUMMY)).unwrap().into_dyn(),
    )];
    if names.iter().any(|n| n == "attention_mask") {
        inputs.push((
            "attention_mask".to_string(),
            Tensor::from_array(Array2::<i64>::ones((1, p))).unwrap().into_dyn(),
        ));
    }
    let run = |session: &mut Session| -> Vec<f32> {
        let feed: Vec<(String, SessionInputValue<'_>)> = inputs
            .iter()
            .map(|(n, v)| (n.clone(), SessionInputValue::from(v)))
            .collect();
        let outputs = session.run(feed).unwrap();
        let arr = outputs[out.as_str()].try_extract_array::<f32>().unwrap();
        arr.iter().copied().collect()
    };
    run(&mut session);
    let mut passes = Vec::with_capacity(d);
    let mut emb = Vec::new();
    for _ in 0..d {
        let t = Instant::now();
        emb = run(&mut session);
        passes.push(t.elapsed().as_secs_f64());
    }
    let mean = passes.iter().sum::<f64>() / d as f64;
    let min = passes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = passes.iter().cloned().fold(0.0, f64::max);
    let var = passes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / d as f64;
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("onnxruntime (C API via ort crate) mode={mode}");
    println!(
        "forward: {p} tokens, {:.2} ms/pass mean (min {:.2}, max {:.2}, σ {:.2}) → {:.1} tok/s, {:.1} passes/s",
        mean * 1e3,
        min * 1e3,
        max * 1e3,
        var.sqrt() * 1e3,
        p as f64 / mean,
        1.0 / mean
    );
    println!(
        "output '{out}' len {}, L2 norm {norm:.4}, head {:?}",
        emb.len(),
        &emb[..emb.len().min(4)]
    );
}
