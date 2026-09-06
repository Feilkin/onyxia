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

`onyxia_ir::Prim` is a **closed enum** of 16 tensor operations (elementwise
unary/binary, compare, select, matmul, reduce, reshape, transpose, broadcast,
concat, slice, gather, scatter, cast, iota, dim-values, dequantize) with
fully specified semantics. The elementwise variants carry an operation
*kind* (25 unary kinds, 15 binary kinds, 6 comparisons) — a backend
implements one kernel template per primitive and a table of expressions per
kind. Lowering rules for 163 ONNX operators are written over this set with
no additions (see `doc/onnx-coverage.md` for what that took). This is the entire backend contract: a backend that implements the
primitives can run any model, including custom ops it has never heard of.
Because the set is closed, shape inference (`infer.rs`) is one total function
with no default arm — adding a primitive makes the compiler point at every
pass that must handle it.

Everything else is a **composite** (`NodeKind::Composite`): a domain-qualified
name plus normalized attributes. Its *decomposition* — a function expanding
it into primitives — lives in a registry (`decomp.rs`), not in the graph.
Softmax, Gelu, Trilu, LayerNormalization, RMSNormalization,
SimplifiedLayerNormalization, Attention (the opset-23 standard op),
RotaryEmbedding, GroupQueryAttention, and MatMulNBits are composites
(GatherBlockQuantized lowers straight to primitives). Ops
without a plausible fused kernel (Conv via im2col, Resize via an
interpolation matrix, TopK via rank counting, the RNN family unrolled, …)
lower straight to primitives in `onyxia-lower/src/rules/`.

### Two registries

1. **Lowering registry** (`onyxia-lower`, backend-neutral):
   `(domain, op_type) → rule`. Every ONNX op enters the IR through it —
   built-ins emit primitives directly; contrib/custom ops emit composites.
   Lowering also evaluates shape-computation subgraphs symbolically
   (`Shape → Gather → Concat → Reshape` chains fold to nothing — they never
   reach the GPU).
2. **Kernel registry** (per backend, `fused.rs` in the wgpu backend):
   `composite name → hand-written fused kernel`. Optional fast path.
   **Fusion** (`fuse::fuse_composites`) first rewrites patterns the backend
   has a kernel for — a residual `Add` feeding `SimplifiedLayerNormalization`
   becomes `onyxia.AddRmsNorm`, `Gelu` feeding its gate `Mul` becomes
   `onyxia.GeluMul` — each with a decomposition back to the unfused nodes,
   so other backends and the interpreter see the same semantics. Then
   **Legalization** (`decomp::inline_composites`) inlines the decomposition
   of any composite the backend lacks a kernel for, recursively, at
   `prepare` time. A composite with neither is a compile-time error.
   **Table splitting** (`split::split_large_tables`) is the other
   backend-driven rewrite: a `Gather` table or transposed-`MatMul` weight
   larger than the device's storage-binding limit (128 MiB on mobile Vulkan,
   versus a 671 MB embedding table) becomes row chunks — chunked gathers
   merged by range selects, chunked matmuls concatenated — built from
   existing primitives, so the interpreter checks the rewrite.

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
is derived at `prepare` (last use per value, inverted into per-step death
lists), and the backend reuses buffers through a refcounted pool; handles
held by the caller are never recycled. `Session::run` consumes and
returns **device tensor handles**; `upload`/`download` are explicit. An
output handle fed back as an input is how the demos keep KV caches on-device
— onyxia contains zero LLM-specific behavior.

`run`/`download` are async because WebGPU readback cannot block the browser
event loop; native callers wrap with `pollster`.

### Execution model: an interpreter over the IR, with caches

There is no compiled execution plan. `Backend::prepare` legalizes,
validates, fixes a topological order, derives liveness, and uploads
constants. `Session::run` binds symbols, evaluates every shape, fills a
register file (constants + inputs), and then walks the nodes in order —
`match` on the primitive, choose a kernel for the concrete shapes, pack
parameters, dispatch, free what died at this step. Everything expensive is
memoized rather than planned: pipelines (by kernel label), bind groups (by
buffer identity), parameter buffers (by content, CubeCL backend), device
buffers (pool / per-value reuse). The walk itself is cheap; measured CPU
cost per run was dominated by what those caches now absorb, and decode on
the wgpu backend is ~85 % GPU-bound.

Why not a plan: kernel selection depends on shapes that are only known at
`run` (M=1 matvec vs tiled matmul, split-K factor, vec4 alignment), and
every decode step binds a different `past_sequence_length`, so a recorded
command stream could not be replayed without patching. A plan cached per
shape signature is a plausible later optimization for repeated identical
prefills; it is not the current design. (The February pipeline had an
explicit `ExecutionPlan` and replaced it with this model within a week.)

## Crate map

| Crate | Contents |
|-------|----------|
| `onyxia-onnx` | `Graph`/`Node`/`TensorInfo` (stable API over protobuf), external-data loading, ONNX-level DOT export |
| `onyxia-ir` | `graph.rs` Module/values/nodes/ConstPool · `prim.rs` the primitive enum · `dim.rs` DimExpr/SymbolTable/Bindings · `types.rs` dtypes incl. Q4/Q8 layout · `builder.rs` GraphBuilder · `infer.rs` shape inference · `fold.rs` constant folding + symbolic shape values · `decomp.rs` standard decompositions + legalization · `fuse.rs` backend-driven pattern fusion · `interp.rs` reference interpreter · `backend.rs` Backend/Session traits · `validate.rs`, `dot.rs`, `attrs.rs` |
| `onyxia-lower` | `LoweringRegistry`, `lower()` driver (symbols from dim_param, initializers moved — not copied — into the ConstPool, inference + folding at the end), `rules.rs` for the standard op set |
| `onyxia-backend-wgpu` | `session.rs` prepare/run/upload/download, register file, liveness-driven pooling, chunked submits so the GPU runs while the CPU encodes (`submit_chunk`), host-side step timing (`take_cpu_timing`), live/peak VRAM accounting (`resident_bytes`) · `kernels.rs` generated one-thread-per-element WGSL for primitives, plus split-K matvec and tiled matmul fast paths · `fused.rs` CompositeKernel trait + registry (Softmax, RMS-norm, Gelu, RotaryEmbedding, GroupQueryAttention with chunked online-softmax, MatMulNBits matvec + dequantizing tiled matmul) · `profile.rs` opt-in per-dispatch GPU timing via timestamp queries (`enable_profiling`/`take_timings`) · `gpu.rs` device/queue, pipeline cache (bind group layouts built by reflecting shader bindings via naga; where the adapter lacks `IMMEDIATES` — all browsers, core WebGPU has no push constants — kernels are rewritten to take params as a storage buffer; `ONYXIA_NO_IMMEDIATES=1` forces this for native testing, and every GPU differential test runs in both modes), buffer pool · `benches/kernels.rs` criterion microbenchmarks at LLM shapes · `legacy-shaders/` hand-written WGSL kept as reference for fused kernels not yet written |
| `onyxia-backend-cubecl` | `Backend`/`Session` over [CubeCL](https://github.com/tracel-ai/cubecl) (`#[cube]` Rust kernels, JIT-compiled; runs on `cubecl-wgpu`). Primitives only — every composite legalizes through its decomposition, which is the demonstration that the primitive set is the whole backend contract |
| `onyxia-backend-ref` | `run_once(module, inputs)` + `Backend` impl over the interpreter |
| `onyxia-conformance` | onnx node-test harness: discovery, `TensorProto` loading, per-backend runners, per-operator matrix, expected-pass regression lists |
| `onyxia-cli` | `run-model`/`chat` generation (`llm.rs` device-resident KV session, `generate.rs`, `sampling.rs`, `tokenizer.rs`), `bench` (prefill/decode throughput + per-kernel GPU-time breakdown, `bench.rs`; see `doc/perf-baseline-2026-07.md`), `validate` (parse + lower, no GPU), ONNX inspection (`inspect.rs`), `dot`/`ir-dot` |
| `demos/gemma-chat` | egui chat UI, native + wasm32 (trunk); vendors its own async LLM session, sampling, tokenizer — application-layer by design |

Backend-private layout decisions live in the backend (`layout.rs` in the wgpu
backend): `f16` and `i64` are native when the adapter has `SHADER_F16` /
`SHADER_INT64` and otherwise packed two-per-word (f16) or narrowed to `i32`
(range-checked at upload); `u8`/`i8` are always packed four per `u32` word
and `u4`/`i4` eight per word (WebGPU has no sub-32-bit storage), `Bool` is
a `u32`. Kernels run one thread per output word, so packed lanes are never
written by two threads.

## Testing strategy

- `onyxia-ir`: unit tests per pass; golden + property tests for the
  interpreter; decomposition-vs-hand-computed tests per composite.
- `onyxia-backend-wgpu` and `onyxia-backend-cubecl` (GPU, `#[ignore]`d,
  `just test-all`): every generated kernel differential-vs-interpreter at
  atol=1e-4/rtol=1e-3 (f32); fused kernels vs their decompositions; GQA
  with symbolic dims, past-KV, and sliding window.
- `onyxia-conformance`: the official onnx node tests (`onnx/backend/test/
  data/node`, Apache-2.0, fetched with `just fetch-onnx-tests`) run through
  lowering and a backend. `cargo test` gates the reference backend against
  `expected-pass-ref.txt`; `just conformance` prints the per-operator
  matrix; `--backend wgpu` (feature `wgpu`) runs the same on the GPU.
  Report: `doc/onnx-coverage.md`.
- Whole-model gates (model files required, see the README):
  `cargo run -p onyxia-cli --example debug-prefill` compares per-position
  prefill argmax GPU-vs-reference on a real chat prompt.

## Known gaps

- Fused kernels: GQA (one dispatch for both present-cache concats),
  RotaryEmbedding, Softmax, RMS norm, Gelu, AddRmsNorm, GeluMul, and
  MatMulNBits (4-bit only: decode reads the packed nibbles directly;
  prefill dequantizes weight tiles into shared memory inside the tiled
  matmul). `GatherBlockQuantized` (the q4 exports' embedding table)
  lowers to Gather + Dequantize primitives — no fused kernel, none
  needed. MatMul has split-K matvec kernels for M=1 and a register-blocked
  64×64 tile (4×4 outputs per thread, split-K when the grid is small) for
  M>1; a cooperative-matrix (tensor core, f16-in/f32-acc) tile exists
  behind `ONYXIA_MATMUL_TILE=coop` but is not faster at prefill sizes.
  Speed history in `doc/perf-baseline-2026-07.md`.
- 8-bit `MatMulNBits` and the CubeCL backend's Dequantize/Scatter kernels
  are not written (q4 models run on the reference and wgpu backends).
- On the wgpu backend: late-bound dims (data-dependent shapes),
  >65535-row fused reductions, fused kernels are f32-only (f16 composites
  cast at the boundary), Scatter reductions serialize on contended words.
- Not lowered: data-dependent output shapes (NonZero, Unique, Compress,
  NonMaxSuppression), Det, DeformConv, RoiAlign, MaxRoiPool, QLinearConv;
  sequences/optionals, strings, random sampling, image decoding.
- No CPU-side per-op tracing spans (Tracy) yet; GPU-side per-dispatch
  timing exists (`profile.rs`).
- ONNX `If`/`Loop`/`Scan` (regions) intentionally not designed yet.
