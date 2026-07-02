# Onyxia IR — Implementation Plan

Companion to `ir-design.md` (the *why*; read it first). This document is the
*what, in what order* — written so any capable agent (Opus included) can pick
up any step cold. Steps are sized S (≤half day), M (1–2 days), L (3–5 days).

## Ground rules for whoever implements

- **Do not relitigate decisions.** Design questions are settled in
  `ir-design.md` and in "Pinned decisions" below. If a step seems to require
  a new foundational decision, stop and surface it — don't improvise.
- **Every step lands green.** `cargo nextest run` (and, where marked, the GPU
  suite via `just test-all`) passes at every step boundary. No step depends
  on a future step to compile.
- **The old pipeline keeps working until C5 deletes it.** Milestones A and B
  are purely additive; do not modify `onyxia-core`/`onyxia-compiler`/
  `onyxia-operators`/`onyxia-runtime` before C1.
- **The reference interpreter is the spec.** When a kernel and the
  interpreter disagree, the kernel is wrong until proven otherwise (if the
  interpreter is wrong, fixing it is a spec change — flag it, fix it, note
  it in the doc).

## Pinned decisions (settled — implement as stated)

1. **Broadcasting & dtypes.** `Binary`/`Select` use ONNX/numpy
   multidirectional broadcasting. **No implicit dtype promotion anywhere**:
   primitives require matching input dtypes; lowering inserts explicit
   `Cast` nodes. Comparison ops produce `Bool`.
2. **Quantized tensors.** `TensorType` carries the *logical* shape; the
   dtype (`Q4 { block_size }`, `Q8 { … }`) defines packed storage layout and
   byte size. `Dequantize` is the only primitive that interprets the
   packing; all other code treats quantized buffers as opaque bytes.
3. **Backend traits (`Backend`, `Session`, `DeviceTensor`) live in
   `onyxia-ir`.** They pull no GPU deps. The CPU interpreter lives in
   `onyxia-ir::interp` as a library; `onyxia-backend-ref` is a thin adapter
   implementing the traits over it (written for clarity, never optimized).
4. **Tolerances.** Backend-vs-interpreter differential tests use
   atol=1e-4, rtol=1e-3 (f32); f16 paths may relax to atol=1e-2. The
   interpreter may compute f16 internally in f32.
5. **Attrs.** A small typed value enum (`Int, Ints, Float, String, Bool,
   Tensor(ConstId)`) in an ordered map, plus typed accessor helpers with
   good error messages. Per-composite structs are optional sugar, later.
6. **SSA, no aliasing, no buffer info in the IR.** Buffer reuse is a
   backend-planner concern derived from liveness. `Reshape` in the IR is
   value-semantic; backends may implement it as a view.
7. **Weights in `ConstPool`** keyed by `ConstId`. Storage type starts as
   `Vec<u8>` behind an accessor; the zero-copy retrofit later changes only
   the pool's storage, nothing else.
8. **Symbolic dims are affine only**: `Const | Sym | Add | Mul`, with
   simplification. No division/modulo until a real model forces it.
9. **Primitive set** as listed in `ir-design.md` §3 (16 primitives).
   `Reduce` includes `Mean` as an op-kind. Growing the set requires meeting
   the discipline rule in `ir-design.md` §7.
10. **Crate names**: `onyxia-ir`, `onyxia-backend-wgpu`,
    `onyxia-backend-ref`. The types: `ir::Module`, `ir::ValueId`,
    `ir::NodeId`, `ir::Prim`, `ir::DimExpr`.

---

## Milestone A — `onyxia-ir` foundation (pure CPU crate) — **[L total]**
## **STATUS: DONE (2026-07-02, Fable).** 62 unit tests + doc tests green;
## compiles on wasm32-unknown-unknown. Deviations from the letter of this
## plan are recorded in `ir-design.md` ("Implementation notes"): Constant is
## an `Origin`, not a `Prim`; `Compare` split from `Binary`; `Dequantize`
## signature provisional pending B4.

New crate; no wgpu/naga anywhere in its dependency tree.

### A1. Dim algebra + type core **[S]**
`DataType` (port from core, add `Q4`/`Q8` with layout methods:
`storage_bytes(logical_shape) -> usize`), `SymId`/`SymbolTable`,
`DimExpr` with `Add`/`Mul` simplification (constant folding, term
collection), `SymbolicShape`, `TensorType`.
*Accept:* unit tests for algebra identities (`(S*1)+0 = S`,
`eval(bindings)` round-trips); `Q4` storage-size math matches the existing
`matmul_nbits.rs` expectations.

### A2. Graph types + builder + validation **[M]**
`Module`, `ValueDef`, `Node`, `NodeKind`, `Prim` (all 16, params in
variants), `ConstPool`, `Attrs`. `GraphBuilder` (the API from
`ir-design.md` §2 — this is user-facing; keep it pleasant). `validate()`:
SSA well-formedness, no dangling ValueIds, arity per primitive, dtype rules
per pinned decision 1. Dot export (reuse the style of the existing
`dispatch_dot.rs`).
*Accept:* builder round-trip tests; validation rejects each malformation
class with a precise error.

### A3. Shape inference **[M]**
One total function `infer(prim, &[TensorType], &Attrs) -> Result<Vec<TensorType>>`
— exhaustive match over `Prim` (no default arm, ever). Module-level pass:
walk topologically, fill `ValueDef::ty`.
*Accept:* per-primitive shape tests incl. symbolic dims (broadcast of
`[1, S, 256]` with `[256]`, reduce with keepdims over symbolic axes, etc.).

### A4. Reference interpreter **[M]**
`interp::eval(module, inputs) -> outputs` over concrete tensors. Supports
f32/f16/i64/i32/u8/bool + `Dequantize` for Q4/Q8. Clarity over speed —
naive loops, no unsafe.
*Accept:* golden tests per primitive (hand-computed vectors); property
tests (e.g. transpose∘transpose = id, reduce-sum matches manual sum).

### A5. Compile-time value domain + constant folding **[M]**
`SymbolicContent` (small i64 tensors of `DimExpr`) on values; folding pass:
evaluate nodes whose inputs are all `Const`/known-content via the
interpreter (size-thresholded, port the threshold idea from the old
`constant_folding.rs`); symbolic evaluation for shape-value arithmetic
(Gather/Slice/Concat/Add/Mul over `SymbolicContent`).
*Accept:* a hand-built `Shape → Gather → Concat → Reshape` chain folds to a
single `Reshape` with symbolic target; folding of an all-const subgraph
replaces it with one `Const`.

## Milestone B — Lowering (ONNX → IR) — **[L total]**
## **STATUS: DONE incl. B5 gate (2026-07-02, Fable + Ada).** Gemma 3 270m
## fp32: 455 ONNX nodes → 438 IR nodes (181 composites + 257 primitives) in
## 1.4 ms; histogram reconciles exactly against the architecture (127
## matmul, 109 RMS-norm, 73 reshape, 36 rotary/add, 19 mul, 18 GQA/Gelu, 1
## gather, 1 transpose); zero shape plumbing survives; 1.09 GiB weights
## moved (not copied) into the pool; 4 dim symbols.
## Lives in the new `crates/onyxia-lower` (decompositions are pure IR and
## live in `onyxia-ir::decomp`). Deviations/limitations, all erroring
## cleanly when hit: GQA decomposition assumes dense batch-1 sequences
## (`seqlens_k` ignored; padded batches need the fused kernel); GQA
## `do_rotary=1`/`softcap` and Rotary `interleaved=1` unsupported;
## MatMulNBits zero_points with odd n_blocks unsupported (per-row padding);
## ConstantOfShape scalar dtype is guessed from byte length because
## `onyxia-onnx` drops attribute-tensor dtypes (parser fix, later); opset
## keying replaced by structural detection (axes-as-input vs attr) since
## `onyxia-onnx` doesn't expose opset imports.
## B5 gate: `cargo run -p onyxia-lower --example lower-stats -- \
##   models/gemma-3-270m-it-ONNX/onnx/model.onnx` (in the main worktree).

### B1. Registry + driver **[M]**
`LoweringRegistry` keyed by `(domain, op_type, opset range)`; attribute
normalization helpers; `lower(&onnx::Graph, &registry) -> Result<Module>`:
create symbols from `dim_param`, walk nodes topologically, apply rules,
move initializers into `ConstPool` (no cloning beyond the one move),
run shape inference + folding at the end.
*Accept:* lowering a hand-built one-node Add graph produces a validated
2-input/1-output module.

### B2. Rules for primitive-mapped ops **[M]**
Add Mul Div Sub Pow Max, comparisons, Neg Sqrt Cos Sin Tanh, Cast, Where,
Reshape Expand Transpose Unsqueeze Concat Slice Gather ScatterND, MatMul,
ReduceSum ReduceMean, Range, Constant, ConstantOfShape, Shape (→ shape
value, see A5).
*Accept:* per-op: build tiny ONNX graph programmatically (the existing
tests do this — copy the pattern), lower, run interpreter, compare against
hand-computed output. Opset-version handling tested at least for one op
with known differences.

### B3. Shape-subgraph folding, end-to-end **[S]**
Wire A5 into the lowering driver so shape chains disappear during lowering.
*Accept:* fixture reproducing the Gemma reshape chain (extract the exact
subgraph from the 1B model with the CLI's `trace-node`) lowers to zero
runtime shape nodes.

### B4. Composites + decompositions **[L]**
Softmax, Gelu, Trilu, SimplifiedLayerNormalization, RotaryEmbedding (+
GemmaRotaryEmbedding), MatMulNBits (→ Dequantize + MatMul), and
GroupQueryAttention (incl. KV concat and sliding-window mask — port the
semantics from `group_query_attention.rs`, which is tested and correct).
Derived shape inference through decomposition bodies (ir-design §3).
*Accept:* per composite: interpreter output of the decomposition matches
hand-computed / existing-test vectors, including GQA with past-KV and
sliding window (port the cases from `group_query_attention_test.rs`).

### B5. Whole-model lowering gate **[S]**
*Accept:* Gemma 3 270m lowers with zero unresolved nodes; every value has
an inferred (possibly symbolic) shape; stats printed (nodes in/out, consts
pooled, shape nodes folded). Runs in the main worktree where the model
exists; keep it `#[ignore]`d + a CLI subcommand (`onyxia lower-stats`).

## Milestone C — wgpu backend rebind — **[L total]** *(GPU required)*
## **STATUS: DONE (C5 cutover 2026-07-02, Fable).** CLI + gemma-chat demo
## run on onyxia-lower + onyxia-backend-wgpu; `onyxia-core`,
## `onyxia-compiler`, `onyxia-operators`, `onyxia-runtime`, and
## `tests.disabled` deleted; old WGSL kept as porting reference in
## `onyxia-backend-wgpu/legacy-shaders/`. The ported `LlmSession` (CLI
## `llm.rs`, demo `inference.rs`) holds the KV cache **on-device** — most of
## milestone D landed with the cutover. `dispatch-dot` became `ir-dot`;
## `validate` now parses + lowers with no GPU; inspection commands read the
## ONNX graph directly. README/ARCHITECTURE rewritten. Green: workspace +
## examples build, 122 tests incl. GPU suite, wasm32 check (ir, lower,
## gemma-chat), trunk build; run-model 9.4 tok/s, multi-turn chat recalls
## context across turns. Manual trunk-serve check in Chrome pending (Ada).
## *(C4 gate record below)*
## C4 gate (`cargo run --release -p onyxia-cli --example parity-gate`):
## Gemma 3 270m fp32, greedy, chat-templated prompt — token-identical for
## 64 tokens; decode 9.25 tok/s new vs 7.16 old (ratio 1.29, well within
## 10%); prepare 2.0 s vs 3.9 s. `--device-kv` (milestone-D preview: feed
## `present.*` handles back as `past_key_values.*` without host round-trip)
## also token-identical at 9.48 tok/s — the Session handle-feedback design
## works as intended. The gate initially FAILED on tokens: the
## RotaryEmbedding decomposition defaulted `num_heads` to 1 when the attr
## is 0 (as optimum exports it), rotating only head 0 of Q; the old kernel
## infers heads from the cache width (hidden / (2·cache_half)). Fixed in
## `onyxia-ir::decomp` + regression test
## (`rotary_infers_heads_from_cache_width`). Lesson recorded: C3's
## forward-check compared new-GPU against new-ref only — both shared the
## lowering bug; only the old-vs-new gate caught it.
## `examples/debug-prefill.rs` (onyxia-cli) does per-position old/gpu/ref
## argmax triage and stays until C5 deletes the old pipeline.
## *(pre-C4 status below: C1–C3, 2026-07-02)*
## `Backend`/`Session` traits live in `onyxia-ir::backend` (device-resident
## tensors from day one — milestone D's API); `onyxia-backend-ref` wraps the
## interpreter; `onyxia-backend-wgpu` executes the *primitive set* with
## generated one-thread-per-element WGSL kernels (immediates + reflection
## machinery ported intact; I64 stored as i32 on-device, Bool as u32 —
## backend-private layout). Gate: Gemma 3 270m fp32 prefill on GPU matches
## the reference interpreter (max |Δlogit| 4.6e-5, argmax identical);
## 7 GPU differential tests incl. GQA with symbolic dims + sliding window.
## **Fused-kernel registry landed** (`fused.rs`: CompositeKernel trait +
## KernelRegistry consulted by supports/legalization) with Softmax and
## RMS-norm as one-workgroup-per-row reduction kernels and Gelu as a single
## pass — full-model prefill went 482 ms → 141 ms warm; every fused kernel
## differential-tests against its decomposition on-device
## (`fused_kernels_match_decompositions`). **Remaining (post-C4 these are
## optional perf work, not gate blockers — the gate passed without them):** fused
## GQA / RotaryEmbedding / MatMulNBits (follow the SoftmaxKernel pattern;
## port math from the old gqa_*.wgsl), a tiled/vectorized MatMul primitive,
## token-identical parity + tokens/sec gate vs the old pipeline, then
## cutover + purge. Also not yet done: Dequantize kernel (q4 models), f16,
## late-bound reshape dims on GPU, >65535-row fused reductions.

### C1. Traits + legalization **[M]**
`Backend`/`Session`/`DeviceTensor` traits in `onyxia-ir` (pinned decision
3). Shared legalization pass: given `supports(name) -> bool`, inline
composite decompositions recursively until all nodes are primitives or
supported composites.
*Accept:* legalizing against "supports nothing" yields pure primitives;
against "supports Softmax" keeps Softmax nodes intact.

### C2. `onyxia-backend-ref` **[S]**
Thin adapter over the interpreter.
*Accept:* runs the B2/B4 test modules through the `Backend` trait; this is
also the harness every other backend's tests reuse.

### C3. wgpu backend **[L]**
Port from `onyxia-runtime`/`onyxia-operators`: kernel registry (existing
WGSL keyed by composite name / primitive kind — register kernels for
*everything* that has one today so decompositions stay cold), plan builder,
memory planner v1 (liveness-driven; the existing `BufferPool` becomes
usable because the planner — not operators — now allocates), symbol
binding, dispatch (keep the immediates + reflection machinery from
`dispatch.rs` verbatim — see memory note on wgpu 29; do not touch it).
*Accept:* every kernel passes differential-vs-interpreter at pinned
tolerances (convert the existing GPU test corpus — mechanical); pool stats
show reuse > 0 on a second run.

### C4. Model parity gate **[M]**
*Accept (main worktree, GPU):* Gemma 3 270m fp32, greedy sampling, fixed
prompt: token-identical output between old pipeline and new for ≥64 tokens;
tokens/sec within 10% of old. This is the go/no-go for C5.

### C5. Cutover + purge **[M]**
CLI + demos onto the new pipeline. Delete `onyxia-core`,
`onyxia-compiler`, `onyxia-operators` (shaders move to the backend crate),
`onyxia-runtime`, `tests.disabled`, and the stale README/ARCHITECTURE
sections — rewrite both docs against the new reality.
*Accept:* workspace + wasm32 build green; full test suite green; web demo
serves via `trunk serve` (manual check, Chrome 149+).

## Milestone D — Device-resident I/O — **[M total]**
## **STATUS: essentially DONE (2026-07-02).** D1 landed with C1 (traits were
## device-resident from day one); D2 landed with the C5 cutover (CLI +
## demo `LlmSession`s feed `present.*` handles back as `past_key_values.*`;
## only logits are downloaded). Remaining nibble: logits come down as the
## full [1, S, V] tensor — a `download` slice (last position only) would
## save a ~1 MB/token copy during prefill-heavy turns. Perf record: C4 gate
## measured 9.17 tok/s (host round-trip KV) → 9.48 tok/s (device KV).

D1 **[M]**: `Session::run` takes/returns `DeviceTensor`s; explicit
`download`; binding validation (shared-symbol consistency across inputs,
bounds check). *Accept:* differential test: device-resident chained run ==
CPU-round-trip run.
D2 **[S]**: demo `LlmSession` holds `present.*` handles, feeds back as
`past_key_values.*`; only logits downloaded (and only the last position —
add `Session` support for downloading a slice, or accept full download for
now and note it). *Accept:* tokens/sec measured before/after and recorded
in the demo README.

## Milestone E — Second backend spike — **[timeboxed L]**

CubeCL (preferred; rust-gpu acceptable) implementing *primitives only*, no
composite kernels. Target: a small model (MNIST-class CNN or a tiny MLP —
pick something with an official ONNX export) end-to-end, differential vs
`onyxia-backend-ref`. **The deliverable is the experience report for the
talk as much as the code** — record what the backend contract got right and
wrong. Hard timebox: if primitives-complete isn't reachable in the box,
elementwise+matmul+reduce running a two-layer MLP is a sufficient result.

---

## Sequencing & handoff notes

- A and B are GPU-free and safe for any environment; C3+ needs the GPU and
  C4/B5 need the model → run those in the **main worktree**.
- Within milestones: A1→A2→A3→A4→A5 and B1→B2→B3→B4→B5 are strictly
  ordered. C2 can start after C1; D after C5; E after C5 (needs only
  legalization + ref backend, not the wgpu port, so it can also run in
  parallel with C3+ if capacity allows).
- Fable owns (or reviews) milestone A if at all possible — semantic
  subtleties concentrate there and every later step inherits them. B onward
  is spec-following work; hand any step to Opus with: this file,
  `ir-design.md`, and the step's accept criteria as the definition of done.
- Independent of all milestones: the Android demo, benchmarks, and vendoring
  a small test model are separate tracks (per Ada, Opus-suitable, not
  blocked by the IR work — though benchmarks get easier after C4).
