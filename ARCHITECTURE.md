# Onyxia Architecture Overview

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust.
ONNX graphs are lowered into a small backend-neutral IR; backends consume the
IR and execute it — today via WGSL compute shaders on wgpu.

This document describes the crates and how data flows through them.

## The stack

```
onyxia-onnx          protobuf → Graph. No IR knowledge.
        │  lower(graph, &LoweringRegistry) — opset/attribute normalization,
        │  shape-subgraph folding happen HERE, at the boundary
        ▼
onyxia-ir            Primitive enum, composites, symbolic shapes, SSA values.
        │            Passes: shape inference, constant folding (CPU ref
        │            impls), legalization. Backend/Session traits.
        │            No wgpu, no naga — compiles anywhere, tests without a GPU.
        ▼  Backend::prepare(Module) → Session
onyxia-backend-wgpu  Generated primitive kernels, fused composite kernels,
onyxia-backend-cubecl  pipeline cache, buffer pool, symbol binding,
onyxia-backend-ref   device-resident tensors. (cubecl = primitives only,
        │            ref = interpreter adapter.)
        ▼
onyxia-cli, demos/   Generation loop, KV plumbing, tokenizer, UI.
```

## Core ideas

### Primitives and composites

`onyxia_ir::Prim` is a **closed enum** of ~16 tensor operations (elementwise
unary/binary, compare, select, matmul, reduce, reshape, transpose, broadcast,
concat, slice, gather, scatter, cast, iota, dequantize) with fully specified
semantics. This is the entire backend contract: a backend that implements the
primitives can run any model, including custom ops it has never heard of.
Because the set is closed, shape inference (`infer.rs`) is one total function
with no default arm — adding a primitive makes the compiler point at every
pass that must handle it.

Everything else is a **composite** (`NodeKind::Composite`): a domain-qualified
name plus normalized attributes. Its *decomposition* — a function expanding
it into primitives — lives in a registry (`decomp.rs`), not in the graph.
Softmax, Gelu, SimplifiedLayerNormalization, RotaryEmbedding,
GroupQueryAttention, MatMulNBits are composites.

### Two registries

1. **Lowering registry** (`onyxia-lower`, backend-neutral):
   `(domain, op_type) → rule`. Every ONNX op enters the IR through it —
   built-ins emit primitives directly; contrib/custom ops emit composites.
   Lowering also evaluates shape-computation subgraphs symbolically
   (`Shape → Gather → Concat → Reshape` chains fold to nothing — they never
   reach the GPU).
2. **Kernel registry** (per backend, `fused.rs` in the wgpu backend):
   `composite name → hand-written fused kernel`. Optional fast path.
   **Legalization** (`decomp::inline_composites`) inlines the decomposition
   of any composite the backend lacks a kernel for, recursively, at
   `prepare` time. A composite with neither is a compile-time error.

Every fused kernel differential-tests against its own decomposition
on-device (`fused_kernels_match_decompositions`).

### The reference interpreter is the spec

`onyxia_ir::interp` evaluates modules on the CPU (naive loops, f64
accumulation internally). When a kernel and the interpreter disagree, the
kernel is wrong until proven otherwise. `onyxia-backend-ref` wraps it behind
the `Backend` trait so backend tests are backend-shaped.

One caveat: interpreter-vs-GPU differentials share any *lowering* bug (both
execute the same IR). Semantics changes should also be checked against an
independent implementation such as onnxruntime.

### Symbolic shapes, bound at run time

Dims are affine expressions over symbols (`DimExpr`: const/sym/add/mul, plus
exact division for `Reshape(-1)`), declared from ONNX `dim_param`
(`batch_size`, `sequence_length`, …). At each `Session::run`, symbols are
inferred from the actual input shapes (`bind_shapes`), every value's shape is
evaluated once, and kernels receive concrete sizes via immediates — kernels
never compute shapes, and there is no per-shape recompilation.

### SSA values, device-resident tensors

The IR is an SSA value graph — no aliasing, no buffer assignments. Liveness
is derived, and the backend's planner reuses buffers (refcounted pool;
handles held by the caller are never recycled). `Session::run` consumes and
returns **device tensor handles**; `upload`/`download` are explicit. An
output handle fed back as an input is how the demos keep KV caches on-device
— onyxia contains zero LLM-specific behavior.

`run`/`download` are async because WebGPU readback cannot block the browser
event loop; native callers wrap with `pollster`.

## Crate map

| Crate | Contents |
|-------|----------|
| `onyxia-onnx` | `Graph`/`Node`/`TensorInfo` (stable API over protobuf), external-data loading, ONNX-level DOT export |
| `onyxia-ir` | `graph.rs` Module/values/nodes/ConstPool · `prim.rs` the primitive enum · `dim.rs` DimExpr/SymbolTable/Bindings · `types.rs` dtypes incl. Q4/Q8 layout · `builder.rs` GraphBuilder · `infer.rs` shape inference · `fold.rs` constant folding + symbolic shape values · `decomp.rs` standard decompositions + legalization · `interp.rs` reference interpreter · `backend.rs` Backend/Session traits · `validate.rs`, `dot.rs`, `attrs.rs` |
| `onyxia-lower` | `LoweringRegistry`, `lower()` driver (symbols from dim_param, initializers moved — not copied — into the ConstPool, inference + folding at the end), `rules.rs` for the standard op set |
| `onyxia-backend-wgpu` | `session.rs` prepare/run/upload/download, register file, liveness-driven pooling, live/peak VRAM accounting (`resident_bytes`) · `kernels.rs` generated one-thread-per-element WGSL for primitives, plus split-K matvec and tiled matmul fast paths · `fused.rs` CompositeKernel trait + registry (Softmax, RMS-norm, Gelu, RotaryEmbedding, GroupQueryAttention with chunked online-softmax) · `profile.rs` opt-in per-dispatch GPU timing via timestamp queries (`enable_profiling`/`take_timings`) · `gpu.rs` device/queue, pipeline cache (bind group layouts built by reflecting shader bindings via naga; where the adapter lacks `IMMEDIATES` — all browsers, core WebGPU has no push constants — kernels are rewritten to take params as a storage buffer; `ONYXIA_NO_IMMEDIATES=1` forces this for native testing, and every GPU differential test runs in both modes), buffer pool · `benches/kernels.rs` criterion microbenchmarks at LLM shapes · `legacy-shaders/` hand-written WGSL kept as reference for fused kernels not yet written |
| `onyxia-backend-cubecl` | `Backend`/`Session` over [CubeCL](https://github.com/tracel-ai/cubecl) (`#[cube]` Rust kernels, JIT-compiled; runs on `cubecl-wgpu`). Primitives only — every composite legalizes through its decomposition, which is the demonstration that the primitive set is the whole backend contract |
| `onyxia-backend-ref` | `run_once(module, inputs)` + `Backend` impl over the interpreter |
| `onyxia-cli` | `run-model`/`chat` generation (`llm.rs` device-resident KV session, `generate.rs`, `sampling.rs`, `tokenizer.rs`), `bench` (prefill/decode throughput + per-kernel GPU-time breakdown, `bench.rs`; see `doc/perf-baseline-2026-07.md`), `validate` (parse + lower, no GPU), ONNX inspection (`inspect.rs`), `dot`/`ir-dot` |
| `demos/gemma-chat` | egui chat UI, native + wasm32 (trunk); vendors its own async LLM session, sampling, tokenizer — application-layer by design |

Backend-private layout decisions live in the backend: logical `I64` is stored
as `i32` on device (range-checked at upload), `Bool` as `u32`.

## Testing strategy

- `onyxia-ir`: unit tests per pass; golden + property tests for the
  interpreter; decomposition-vs-hand-computed tests per composite.
- `onyxia-backend-wgpu` and `onyxia-backend-cubecl` (GPU, `#[ignore]`d,
  `just test-all`): every generated kernel differential-vs-interpreter at
  atol=1e-4/rtol=1e-3 (f32); fused kernels vs their decompositions; GQA
  with symbolic dims, past-KV, and sliding window.
- Whole-model gates (model files required, see the README):
  `cargo run -p onyxia-cli --example debug-prefill` compares per-position
  prefill argmax GPU-vs-reference on a real chat prompt.

## Known gaps

- No fused MatMulNBits kernel yet (its decomposition executes instead —
  and the Dequantize primitive below blocks it on GPU anyway). GQA and
  RotaryEmbedding are fused; MatMul has split-K matvec kernels for M=1
  and a shared-memory tiled kernel for M>1. Decode-speed history in
  `doc/perf-baseline-2026-07.md`.
- No Dequantize GPU kernel (q4 models run only on the reference backend).
- f16, late-bound dims on GPU (data-dependent shapes), >65535-row fused
  reductions.
- No CPU-side per-op tracing spans (Tracy) yet; GPU-side per-dispatch
  timing exists (`profile.rs`).
- ONNX `If`/`Loop`/`Scan` (regions) intentionally not designed yet.
