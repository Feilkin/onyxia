# Onyxia IR Design

Status: **draft for discussion** — written 2026-07-02 as the outcome of the
whole-project review. Companion to `ada-notes.md` (the four-layer target
architecture) and the July 2026 review findings.

## 1. The problem

Two original goals are in apparent conflict:

- **Extensibility** — users add custom operators without editing onyxia.
- **Multiple backends** — wgpu today; rustgpu, cubecl, ash tomorrow.

If "adding an operator" means "providing an implementation the runtime can
execute", then N backends seem to force custom-op authors to write N
implementations. That is the tension, and it is real *in the current design* —
but it is not inherent. It is an artifact of one specific decision:

> **The unit of extensibility (an ONNX op) is also the unit of execution
> (a GPU dispatch).**

`Operator::create_dispatch()` conflates three distinct things:

1. *What the op means* (its semantics — output shapes, math).
2. *How to compute it* (a kernel — WGSL, workgroups, bindings).
3. *When/where to run it* (dispatch — pipelines, buffers, encoders).

Because all three live inside one opaque trait object keyed by an op-type
string, the graph knows nothing about its own nodes. Every consequence found
in the review follows:

- Passes are blind → every analysis (`infer_shapes`, `describe`,
  `resolve_shapes`) had to be bolted onto the `Operator` trait, requiring all
  ~27 operator impls to opt in — and none ever did. The three stalled
  migrations weren't a discipline failure; the design made every analysis
  cost O(all operators).
- Backends are welded in → the trait *is* the wgpu interface, so
  `onyxia-core` depends on `wgpu`/`naga` and `compile()` needs a live GPU.
- The IR "conveys everything we need for optimized execution" (ada-notes goal
  #2) in theory but in practice conveys only topology, because semantics are
  hidden in Rust code.
- Shape-manipulation subgraphs (`Shape → Gather → Unsqueeze → Concat →
  Reshape`) execute as real GPU dispatches, because nothing can evaluate them
  at compile time.

The instinct in ada-notes — *"we are mirroring the ONNX operator graph too
closely"* — is correct, with one refinement: the graph *structure* (edges as
tensor values, petgraph) was fine. What mirrors ONNX too closely is the
**node contents**: op-type strings with raw attribute bags, spanning wildly
different granularities (`Add` is one instruction; `GroupQueryAttention` is a
subroutine hiding nine kernels).

## 2. The resolution: primitives + composites, two registries

The move that dissolves the tension (used, in variations, by XLA/StableHLO,
TVM, tinygrad, torch.export): **split the op universe into a small closed set
of primitives and an open set of composites.**

### Primitives — closed, owned by the IR

A fixed enum of ~15 tensor operations with fully specified semantics. This is
the *entire* backend contract: a backend that implements the primitives can
run any model, including models containing custom ops it has never heard of.

Because the set is closed and semantics are known, everything the February
sprint tried to bolt on becomes *library code written once in the IR crate*:

| Stalled migration | Under a closed primitive set |
|---|---|
| shape inference per operator | one `fn infer(prim, in_shapes) -> out_shapes`, total by construction — the compiler's exhaustiveness check *forces* coverage when a primitive is added |
| `describe()`/`OpDescriptor` | unnecessary: primitives are self-describing |
| constant folding on GPU | CPU reference impls of primitives → folding needs no GPU; `compile()` stops requiring a device |
| memory planning | shapes known at bind time → liveness + pre-allocation actually work |

### Composites — open, the extensibility point

A composite node carries a name, its (preserved) attributes, and — via a
registry, not inline — a **decomposition**: a function that expands it into a
subgraph of primitives (and/or other composites). Crucially, a decomposition
is *backend-agnostic*: written once, runs on every backend, current and
future.

Extensibility becomes **two registries** instead of one trait:

1. **Lowering registry** (IR-level, backend-neutral):
   `(domain, op_type, opset range) → lowering rule`. A rule either emits
   primitives directly or emits a composite node. This is how *all* ONNX ops
   enter the IR — built-ins and custom ops go through the same door. ONNX
   contrib ops (`com.microsoft.GroupQueryAttention` etc.) are just lowering
   rules shipped in the box, not engine special cases.
2. **Kernel registry** (per backend, optional fast path):
   `composite name (+ predicate on shapes/dtypes) → hand-written kernel`.
   The wgpu backend registers its existing WGSL kernels here.

Execution rule (**legalization**): the backend walks the graph; for each
composite it either has a kernel (use it) or asks the IR to inline the
decomposition, recursively, until only nodes it can execute remain. A
composite with neither kernel nor decomposition on some backend is a
compile-time error with a precise message — never a runtime surprise.

### The custom-op story after the split

```rust
// Portable: one decomposition, works on wgpu + cubecl + rustgpu + CPU ref.
registry.lower("MyRmsVariant", |node, b: &mut GraphBuilder| {
    let x   = b.arg(0);
    let ms  = b.reduce(Reduce::Mean, b.mul(x, x), &[-1], /*keepdims*/ true);
    let inv = b.rsqrt(b.add(ms, b.const_f32(node.attr_f32("epsilon")?)));
    b.ret(0, b.mul(x, inv))
});

// Optional per-backend fast path — an optimization, never a requirement.
wgpu_backend.register_kernel("MyRmsVariant", my_fused_wgsl);
```

The user is **never obligated** to touch every backend. Hand-written kernels
become a performance opt-in, exactly where hand-tuning belongs. And the two
paths differential-test each other for free: `assert(kernel(x) ≈
inlined_decomposition(x))` — the existing GPU dispatch-test corpus converts
almost directly into this form.

## 3. The primitive set (proposal)

Chosen to cover the current 41 registered ops with room to spare. Parameters
live *in* the variant (an elementwise node is `Elementwise(BinaryOp::Add)`,
not a distinct node kind per op).

| # | Primitive | Notes / covers today |
|---|---|---|
| 1 | ~~`Constant`~~ | *(became a value origin, not a node — see note below)* |
| 2 | `Iota` | Range; index generation for decompositions |
| 3 | `Cast` | Cast |
| 4 | `Unary(op)` | Neg, Sqrt, Rsqrt, Exp, Log, Cos, Sin, Tanh, Erf… |
| 5 | `Binary(op)` | Add, Mul, Div, Sub, Pow, Max, Min, And/Or/Xor |
| 6 | `Compare(op)` | Eq/Ne/Lt/Le/Gt/Ge → Bool (split from Binary: different output-dtype rule) |
| 7 | `Select` | Where |
| 8 | `MatMul` | batched, with transpose flags |
| 9 | `Reduce(op, axes, keepdims)` | ReduceSum/Max/Min/Prod; Mean first-class (pinned decision 9) |
| 10 | `Reshape` | pure metadata when layout allows; explicit copy otherwise |
| 11 | `Transpose(perm)` | |
| 12 | `Broadcast(shape)` | Expand; also implicit via binary broadcasting rules |
| 13 | `Concat(axis)` | the one variadic primitive |
| 14 | `Slice` | per-axis start/end/step; symbolic bounds for step ±1 |
| 15 | `Gather(axis)` | Gather; embedding lookup |
| 16 | `Scatter` | ScatterND |
| 17 | `Dequantize` | q4/q8 blocks → float; keeps quantized models portable |

*Implementation notes (milestone A, 2026-07-02).* Three refinements landed
with the code, all in `onyxia-ir`: **(a)** `Constant` is not a node — values
have an `Origin` (`Input | Const(ConstId) | Node`), so constants are value
origins and there is one less primitive to implement; **(b)** comparisons
split out of `Binary` into `Compare` because their output dtype rule
differs; **(c)** `DimExpr` is a canonical *polynomial* with signed
coefficients — subtraction works (`total_len - past_len`) and exact division
by a monomial resolves `Reshape(-1)` targets symbolically. Names are plain
`String` (not `SmolStr`) until profiling says otherwise. The reference
interpreter evaluates floats in f64 internally — the spec a kernel meets is
"within tolerance of the high-precision result", not bit-equality with one
f32 evaluation order.

Everything else in today's registry becomes a **composite**:

- `Softmax` → max-reduce, sub, exp, sum-reduce, div (existing 3-kernel WGSL
  stays as the wgpu fast path).
- `SimplifiedLayerNormalization` → mul/mean/rsqrt/mul chain (2 kernels stay).
- `RotaryEmbedding`, `GemmaRotaryEmbedding` → slice/mul/add over precomputed
  cos/sin (kernel stays).
- `GroupQueryAttention` → reshape/transpose/matmul/softmax/concat chain (the
  nine `gqa_*.wgsl` kernels stay as the fused fast path; the decomposition is
  the correctness reference we've never had).
- `MatMulNBits` → `Dequantize` + `MatMul` as the portable fallback; the fused
  dequant-matmul kernel remains the fast path (the fallback materializes fp32
  weights — correct, slow, and honest about it).
- `Gelu`, `Trilu`, `ReduceMean`, `ConstantOfShape` → trivial decompositions.
- `Shape`, `Unsqueeze`, and friends **disappear at lowering**: with symbolic
  shapes in the IR, shape-computation subgraphs evaluate at compile time
  instead of dispatching to the GPU. Transformer graphs are full of these;
  today we run them on-device.

**Structural form.** SSA value graph:

```rust
struct Module {
    values:  Vec<ValueDef>,     // ValueId = index
    nodes:   Vec<Node>,
    consts:  ConstPool,         // weights out-of-line, referenced by ConstId
    symbols: SymbolTable,       // dim symbols: "sequence_length", …
    inputs:  Vec<(SmolStr, ValueId)>,
    outputs: Vec<(SmolStr, ValueId)>,
}

struct ValueDef {
    name: Option<SmolStr>,      // ONNX tensor name (I/O, debugging)
    ty: TensorType,             // { dtype, shape: SymbolicShape }
    origin: Origin,             // Input | Const(ConstId) | Output(NodeId, usize)
}

struct Node {
    kind: NodeKind,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    loc: SourceInfo,            // ONNX node name/op_type — errors, tracing, dot
}

enum NodeKind {
    Prim(Prim),                 // closed enum, params inside the variant
    Composite { name: SmolStr, attrs: Attrs }, // domain-qualified name;
                                // decomposition lives in the registry
}
```

The closed enum is *the* load-bearing change: passes pattern-match
exhaustively, and adding a primitive makes the compiler point at every pass
that must handle it — the "friendly compiler" talk thesis, embodied.

Deliberately **not** in a composite node:

- *The decomposition* — registry-held, keyed by name; the graph stays pure
  data (serializable, one definition for all instances). Unknown composite
  with no registry entry = lowering error, never a runtime surprise.
- *A shape function* — derived by running shape inference symbolically over
  the decomposition body off to the side (input types in, walk the
  primitives, output types out — no inlining into the main graph). Optional
  explicit override exists but is an optimization, not an obligation.

Lowering's one real duty per op is **attribute normalization**: resolve ONNX
defaults/opset quirks into canonical typed attrs so kernels and
decompositions see the same clean view.

**Values are SSA — no aliasing, no mutation, no buffer info in the IR.**
Buffer reuse is *derived*, not represented: liveness over SSA is trivial
(last use per value), and the backend's memory planner — the only party that
knows alignment, size buckets, and whether Reshape can be a view on its
hardware — assigns values to buffers. This inverts the February design:
reuse required every operator to implement `resolve_shapes`; here the
planner gets shapes + liveness from the IR unconditionally and operators
don't participate at all. In-place tricks (KV append, donated inputs) are
planner/session concerns; the graph never contains an alias edge.

**Weights live in the `ConstPool`**, not inline in edges. Today initializer
bytes are cloned protobuf → IR → WeightRegister (~3× model size on host —
what brushes the wasm 4 GB ceiling). With the pool as single owner, lowering
copies nothing, and pool storage can later become `Arc<[u8]>` slices into
the fetched buffer or an mmap — the zero-copy retrofit (goal #1) becomes a
change to one type instead of a plumbing rewrite.

**Dynamic shapes.** Three pieces:

1. *Symbolic dims*: `DimExpr = Const(u64) | Sym(SymId) | Add | Mul` (affine
   only — resist division until forced). Graph inputs declare them from ONNX
   `dim_param` (`input_ids: [1, S]`); anonymous dynamic dims get fresh
   symbols.
2. *Shape computations evaluate at lowering as `DimExpr` arithmetic, not as
   tensor ops.* `Shape(x)` produces a value whose **content** is
   symbolically known (a small i64 vector of `DimExpr`s); lowering rules for
   Gather/Slice/Concat/arith evaluate such values symbolically instead of
   emitting nodes. The Gemma 1B token-count chain
   `Shape → Gather[1] → Unsqueeze → Concat → Reshape` folds to
   `Reshape(hidden, [1, S, 4, 256])` — zero runtime nodes, and the target
   shape becomes visible to inference and the planner. (Same approach as
   onnxruntime's symbolic shape inference; it was intractable here only
   because the old IR had no compile-time value domain to evaluate into.)
3. *Graceful degradation*: a chain that escapes symbolic evaluation (truly
   data-dependent shapes — `NonZero` etc.) yields a **late-bound symbol**
   whose value is learned when the producing tensor materializes. Correct
   always; merely unplannable for that one buffer. Gemma-class transformers
   never hit this path.

At run time, **binding** replaces per-operator shape logic: infer symbol
values from actual input shapes (`S := input_ids.shape[1]`), validate
consistency, evaluate every value's `DimExpr`s once, dispatch with fully
concrete shapes — kernels never compute shapes. Bound values feed workgroup
counts and immediates (immediates stay; the wgpu-29 reflection fix carries
over). No per-shape recompilation: one artifact, many bindings. For reuse
across runs with varying `S`, the planner sizes arenas against
user-declared bounds (a `prepare`-time resource declaration, morally the
same as today's `max_seq_len` or wgpu limits); exceeding a bound is a clean
error.

## 4. The layers, restated

```
onyxia-onnx          protobuf → Graph. Dumb, eager today; zero-copy later.
                     No IR knowledge. (Goal #1 in ada-notes — orthogonal,
                     deferrable.)
        │  lower(graph, &LoweringRegistry) — opset resolution, attribute
        │  evaluation, shape-subgraph folding happen HERE, at the boundary
        ▼
onyxia-ir            Primitive enum, composites, symbolic shapes, values.
                     Passes: shape inference, constant folding (CPU ref
                     impls), liveness, legalization, (later) fusion.
                     **No wgpu, no naga. Compiles anywhere, tests without
                     a GPU.**
        │  Backend::prepare(&Module) → Session
        ▼
onyxia-backend-wgpu  Kernel registry (existing WGSL, rebound), pipeline
(-cubecl, -rustgpu…) cache, memory planner, executor, DeviceTensor.
        │
        ▼
onyxia-cli, demos    Unchanged in spirit. LLM plumbing stays vendored in
                     demos — see §5.
```

`onyxia-core` and `onyxia-compiler` as they exist today dissolve into
`onyxia-ir` (types + passes + lowering) and the wgpu backend (everything that
touches a device). `onyxia-operators` splits: lowering rules → `onyxia-ir`
(or a thin `onyxia-onnx-ops` crate); WGSL + dispatch code → the wgpu backend.

**Backend trait (sketch, deliberately small):**

```rust
trait Backend {
    type Session: Session;
    /// Legalize (inline composites it lacks kernels for), plan memory,
    /// compile pipelines.
    fn prepare(&self, module: &ir::Module) -> Result<Self::Session>;
    fn supports(&self, composite: &str) -> bool; // drives legalization
}

trait Session {
    fn upload(&self, t: &Tensor) -> Result<DeviceTensor>;
    async fn run(&mut self, inputs: &[(&str, DeviceTensor)])
        -> Result<Vec<(String, DeviceTensor)>>;
    async fn download(&self, t: &DeviceTensor) -> Result<Tensor>;
}
```

**Backend-local fusion.** The neutral IR never contains backend-specific
nodes. After legalization the backend *consumes* the IR and emits its own
executable plan; rewrites like folding `MatMul → Gelu` into a fused
GEMM-with-epilogue kernel are peephole patterns run during `prepare`,
producing entries in the backend's plan (an NPU backend would instead keep
pure MMA and place the Gelu elsewhere — placement is its business). Pattern-
matching *utilities* can be shared library code in `onyxia-ir`; patterns and
results are backend-private. Note the scale rule: two-node peephole fusion is
robust; re-fusing a fifteen-node decomposed attention subgraph is not — which
is exactly why big fusions ride on composite *names* (lazy inlining) and only
small epilogue fusions ride on *patterns*.

## 5. ONNX purity and the KV-cache question

Guardrail (from ada-notes and reiterated since): onyxia implements the ONNX
spec, not use cases. The test for any feature: *"does the design prevent the
user from doing this themselves?"*

Applied to KV-cache reuse — today the answer is **yes, the design prevents
it**: `run()` accepts CPU tensors, returns CPU tensors, and the register file
is private, so `present.*` must round-trip through host memory every token.
The demo's CPU-side cache is a workaround for a missing *general* capability,
not a missing LLM feature.

The general fix is the `Session` API above: **device-resident tensors as
first-class values** (the same shape as onnxruntime's IOBinding, which exists
for exactly this reason). Outputs stay on-device; downloads are explicit; an
output handle can be passed as a subsequent input. KV reuse then lives
entirely in user code — `demos/gemma-chat` holds the `present.*` handles and
feeds them back as `past_key_values.*` — and onyxia contains zero
LLM-specific behavior. The same mechanism serves diffusion loops, beam
search, or any iterative model.

(Future, still spec-neutral: buffer donation — caller declares an input dead
so an output may reuse its allocation. Useful for in-place KV append;
strictly an optimization; not in scope for the first cut.)

## 6. Migration path

Ordered so the Gemma demos never break and every step is testable:

1. **`onyxia-ir` skeleton** — primitive enum, symbolic shapes, builder,
   shape-inference table, CPU reference interpreter. Pure crate, property
   tests, no GPU. The reference interpreter is small (~15 ops) and
   immediately becomes the differential-test oracle we've wanted since
   February (see `onnx-model-quant-quality` notes).
2. **Lowering for the Gemma op set** — every current op becomes either a
   direct primitive emission or a composite with decomposition. Shape
   subgraphs start folding away here. Validate: lowered-graph CPU-ref output
   ≈ current runtime output on small fixtures.
3. **Rebind the wgpu backend** — new `Backend`/`Session` implemented largely
   from existing `dispatch_executor` + WGSL kernels, registered against
   composites/primitives. Initially register kernels for *everything* that
   has one today → performance parity from day one; decompositions are the
   fallback, not the hot path.
4. **Device-resident I/O** — `DeviceTensor` in/out, explicit download. Port
   the demo's `LlmSession` to hold device handles; the per-token CPU
   round-trip disappears. (Biggest user-visible perf win in the plan.)
5. **Second backend spike** — timeboxed. CubeCL or rust-gpu implementing
   *only the primitives*, running a small model end-to-end via pure
   decomposition. This is the proof that the tension is resolved — and the
   RustConf lesson-#3 material, whatever its performance shows.

Steps 1–2 don't touch the existing pipeline at all; the old and new paths can
coexist until step 3 lands.

**What this deletes** (subsumes most of the trust-restoring purge):
`Operator`/`OpDispatch` as extension points, `infer_shapes`/`describe`/
`resolve_shapes` and their inert passes, `scheduler.rs`, `descriptor.rs`,
the never-reusing buffer-pool wiring. **What survives:** the ONNX parser, all
WGSL shaders, the executor's batching/immediates machinery, the GPU test
corpus (converted to kernel-vs-decomposition differential tests), demos, CLI.

## 7. Open questions

- **Regions / control flow.** ONNX `If`/`Loop`/`Scan` need subgraph-carrying
  nodes eventually. Proposal: design nothing now, but make composite bodies
  proper single-block regions so multi-block control flow is an extension,
  not a rewrite.
- **Layout.** Is `Reshape` metadata-only (views/strides in the IR) or always
  a copy? Proposal: IR stays value-semantic (no aliasing in the graph);
  backends recover zero-copy via buffer planning. Simpler passes, and
  aliasing bugs stay confined to one backend's planner.
- **Fusion.** Deliberately out of scope for the first cut — composites with
  hand-written kernels *are* the fusion story for now. Note that a primitive
  IR is what makes cubecl-style JIT fusion of elementwise chains possible
  later; today's IR can't express the question.
- **Primitive-set discipline.** The set will want to grow (Pad, ArgMax, TopK,
  resize…). Rule of thumb: a new primitive must be (a) not expressible as a
  reasonable decomposition, or (b) so universal that every backend wants it
  native. Everything else is a composite. Growing the enum is cheap by
  design; *shrinking* it never happens — bias small.
- **Should `Reduce` include `Mean`?** Leaning yes (numerical-stability
  latitude for backends) but it's a two-line decomposition either way. Low
  stakes, decide in step 1.
