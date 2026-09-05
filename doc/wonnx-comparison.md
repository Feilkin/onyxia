# Onyxia vs wonnx — derivation audit and architecture comparison

Written 2026-08-23 for the RustConf talk. Question asked: since most of
Onyxia's code was AI-written, and wonnx (webonnx/wonnx, 2021–2024) has
been in training data for years, did any of Onyxia's architecture or code
come from wonnx?

Method: cloned wonnx (`c62f5d3`, 2024-05-18); grepped ~75 distinctive
wonnx identifiers into Onyxia and ~70 Onyxia identifiers into wonnx
(current tree, the pre-rewrite February generation at `58a2de6`, and
docs); compared WGSL index math (broadcast, transpose, softmax, gather,
matmul), erf constants, comments/doc phrases, dependencies, and pipeline
structure.

## Verdict: no meaningful overlap

- One identifier hit in either direction: `GpuTensor` (generic; the two
  structs differ in fields and visibility). No shared doc/comment
  phrases. wonnx is never mentioned in Onyxia source, docs, commits, or
  `Cargo.lock`.
- erf: wonnx uses a tanh-based approximation
  (`tanh(2/√π · (x + 0.08943x³))`, `pi = 3.1415`); Onyxia uses
  Abramowitz–Stegun 7.1.26. No constant in common.
- Broadcast index math: wonnx unrolls per-dimension at Tera template time
  with baked-in literal strides; Onyxia's `src_index()` is a runtime loop
  over `array<u32,8>` shapes. Transpose, softmax (one thread per row vs
  256-thread workgroup with tree reductions), gather (axis 0 only vs any
  axis), matmul (mat4x4 blocks vs naive/tiled/split-K) — all different.
- Only shared constants: workgroup size 256 and the 65535 dispatch cap —
  WebGPU limits, not design choices.
- Dependencies: wonnx uses `rust-protobuf` 2.x with a checked-in generated
  `onnx.rs` and Tera; Onyxia uses `prost` + `protox` and `format!`.

## Architecture comparison

| Axis | wonnx | Onyxia |
|---|---|---|
| Graph | `Arc<Node>` DAG wrapping raw `NodeProto`s, built backwards from outputs | SSA `Module` with arena ids, `NodeKind::{Prim, Composite}`, `ConstPool` |
| Op model | one `match op_type` arm → one of 24 Tera WGSL templates; Conv+Relu fusion in the optimizer | 17 closed primitives with generated kernels + open composites with registry decompositions; optional fused kernels; legalization |
| WGSL generation | Tera templates, shapes baked in as literals, one pipeline per node | `format!` templates parameterized by dtype/op only; shapes via immediates; pipeline cache; layouts reflected via naga |
| Shapes | fully static; `DimensionsMissing` → "run onnx-simplifier"; `nnx prepare --set batch_size=1` | symbolic `DimExpr`, bound at every `run`, no recompilation |
| Memory | compile-time lease/release `BufferManager` | liveness-derived refcounted pool; caller-owned device handles |
| Const folding | on the GPU via a nested `GpuModel` | on the CPU via the interpreter + symbolic shape folding at lowering |
| Tests | per-op tests vs ndarray (2 ULPs), SqueezeNet/MNIST/BERT | interpreter as spec; GPU differentials in both immediates modes; fused vs decomposition; greedy-identical to ORT |
| Web | `wonnx-wasm` + npm package, Chrome-Canary-flag era instructions | `demos/gemma-chat` on wasm32/WebGPU; storage-buffer params fallback |
| Op coverage | ~80 CNN-era ops (Conv, pooling, BatchNorm, Resize); Gather axis 0 only; no i64/f16/attention | 55 lowering rules, transformer-era (GQA, RoPE, RMS-norm, MatMulNBits); no Conv |
| LLM support | none (no KV cache, no contrib domain) | device-resident KV, fused GQA with sliding window, generation loop in CLI/demo |
| Size | ~5.4k Rust + 1.4k WGSL templates + 2k preprocessing; 6 crates | ~19k Rust (kernels inline) + 2k tests; 7 crates + demo |

## wonnx status (2026-08-23)

- GitHub repository **archived 2025-05-07**; last commit 2024-05-18, last
  push 2024-07-21. 1.75k stars.
- crates.io `wonnx` 0.5.1 published 2023-09-30; master pins `wgpu 0.19.3`
  (Jan 2024). Onyxia is on wgpu 29.
- It genuinely ran in browsers in its day (WebGPU shipped in stable
  Chrome 113, May 2023), but the README still describes the Canary +
  `Unsafe WebGPU` flag workflow and it has not tracked the spec or wgpu
  since early 2024.

## Fair framing for the talk

wonnx proved the premise — ONNX on wgpu, pure Rust, in the browser, with
80 ops and a CLI. It targeted the CNN era with static shapes and one
templated shader per op. Onyxia targets the transformer era: symbolic
shapes bound per run, a closed primitive set plus decomposed composites,
an interpreter as the spec, and device-resident tensors for KV caches.
They overlap on elementwise/reduce/gather basics and nowhere else.
