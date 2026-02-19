## Plan: Restructure onyxia-compiler into IR-based pass pipeline

### Summary

Restructure the compilation architecture around a new `onyxia-core` crate that owns a graph-based intermediate representation (IR), the `Operator`/`Pass` traits, and execution plan types. `onyxia-compiler` becomes a thin pipeline orchestrator that runs passes through named stages. The 38 operator implementations move to a new `onyxia-operators` crate with collapsed families (one `impl Operator` for all binary elementwise ops, one for all unary, etc.) to eliminate ~460+ lines of duplication. Optimization passes are standalone objects registered on the compiler pipeline — not owned by operators — enabling users to define and insert passes from anywhere. Symbolic dimensions flow through the pipeline, using `naga_oil` shader defs for statically-resolved dims (enabling downstream WGSL compiler loop unrolling) and immediates for dynamic ones. Constant-folded nodes are skipped during planning. Rust-side constant folding is kept for now but scoped for future replacement with GPU-based folding to eliminate the current duplication of logic between Rust `try_fold()` and WGSL shaders.

### Problems this plan addresses

| Problem | Current state | How the new architecture fixes it |
|---------|--------------|----------------------------------|
| **Shape inference is complex and hard to debug** | Three-phase resolution mutates a cloned ONNX Graph in-place. Operators inconsistently return `TensorShape::Unknown` vs `Err`. Failures are silent — `Unknown` propagates downstream until `plan()` crashes. | Proper IR with a `TensorShape` enum that has no `Unknown` variant. Operators must return `Err` when shapes can't be determined. Context helpers like `require_static(n)` eliminate repeated 8-line match boilerplate. |
| **Operator implementations are messy and repetitive** | Binary elementwise ops (Add, Sub, Mul, Div, Pow, Max) are 6 nearly identical files — `plan()` is 57 lines of verbatim-copied code differing only in shader label and path. Unary ops (Cos, Sin, Sqrt, Neg, Tanh) repeat the same pattern. ~460+ redundant lines total. | Collapsed operator families: `BinaryElementwiseOp` and `UnaryElementwiseOp` structs parameterized by name, shader source, and fold function. One `impl Operator` per family. |
| **Operator definitions are imperative, not declarative** | Every operator manually constructs workgroups, encodes immediates, sets up bindings, compiles shaders. A new elementwise op requires copy-pasting an entire file. | Operator families are declarative: `BinaryElementwiseOp::new("Add", include_str!("add.wgsl"), \|a,b\| a+b)`. Complex ops keep custom `impl Operator`. |
| **Optimizations are hard/impossible to implement** | No pass infrastructure. Scheduler is a simple topological sort. `group_into_passes()` is a stub. No operator fusion, no buffer aliasing, no dead code elimination. | Pass pipeline with named stages (Resolution → Inference → Folding → Optimization → Planning). Visitor-based `Pass` trait. Graph IR with `StableGraph` supports node removal/replacement for rewrite passes. |
| **Constant folding is verbose, logic hard to follow** | Per-operator `try_fold()` repeats the same boilerplate: extract inputs, match on `TensorValue::F32`, apply function, wrap result. Each binary op duplicates ~23 lines of identical extraction code. Additionally, the fold logic in Rust duplicates what the WGSL shader does. | `FoldCtx` provides helpers like `binary_fold_f32(fn)` that handle all extraction/broadcasting/wrapping. Future: GPU-based folding would run the actual shader to fold, eliminating the Rust/WGSL duplication entirely. |
| **Constant-folded nodes still generate GPU operations** | `compile()` calls `operator.plan()` on every node regardless of whether it was folded. Shape→Gather→Concat→Reshape chains in Gemma generate unnecessary `WriteBuffer`/`CopyBuffer` GPU steps. | Planning pass skips nodes where all outputs have `TensorDef.value` set (i.e., were successfully folded). |
| **Inconsistent shape inference** | Some operators return `TensorShape::Unknown` from `infer_output_shapes` when they should return `Err`. Cos/Sin/Sqrt clone input shape without checking for Unknown/Dynamic. Softmax has no guard at all. | `TensorShape::Unknown` is removed from the enum entirely. Operators must return `Err` for undeterminable shapes. `InferenceCtx` helpers make correct handling the path of least resistance. |
| **No way to update dynamic dimensions after build** | Once `compile()` runs with specific dimension values, the `ExecutionPlan` is fixed. Changing sequence length requires full recompilation. LLM prefill→decode transitions are expensive. | Symbolic dimensions persist in `CompiledModel` via `SymbolicBinding` tuples. `PlanExecutor::update_dimensions()` patches immediates in-place without recompilation. Dimensions compiled as shader defs require selective shader recompilation only. |
| **Tests split between compiler and runtime** | Operator shape inference/fold tests are unit tests in each compiler operator file. GPU correctness tests are in onyxia-runtime/tests/. Test boilerplate duplicated everywhere. | All operator tests (unit + GPU) consolidated in `onyxia-operators`. Shared test helpers. Compiler tests focus on pass pipeline behavior. |

### Current architecture (before)

```
┌─────────────────┐     ┌──────────────────┐      ┌─────────────────┐
│  onyxia-onnx    │────▶│ onyxia-compiler  │────▶│ onyxia-runtime  │
│  (ONNX parser)  │     │ (38 operators,   │      │ (GPU executor)  │
│                 │     │  shape inference, │      │                 │
│                 │     │  scheduling,      │      │                 │
│                 │     │  shaders, plan)   │      │                 │
└─────────────────┘     └──────────────────┘      └─────────────────┘
```

**Current compilation flow:** `Graph` → clone → mutate in-place (resolve dimensions) → mutate in-place (infer shapes + fold constants) → clone again for scheduler → topological sort → call `operator.plan()` on ALL nodes (including folded ones) → assemble `ExecutionPlan`.

**Current `Operator` trait:**
```rust
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn infer_output_shapes(&self, ctx: &InferenceContext<'_>) -> Result<Vec<TensorShape>>;
    fn try_fold(&self, ctx: &InferenceContext<'_>) -> Result<Vec<Option<TensorValue>>> {
        Ok(vec![None; ctx.node.outputs.len()])
    }
    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>>;
}
```

**Current `ExecutionPlan`:**
```rust
pub struct ExecutionPlan {
    pub operations: Vec<PlannedOp>,
    pub shaders: Vec<CompiledShader>,
    pub tensors: TensorRegistry,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub metadata: ModelMetadata,
}
pub struct PlannedOp {
    pub name: String,
    pub op_type: String,       // dead data — never read at runtime
    pub inputs: Vec<TensorId>, // dead data — never read at runtime
    pub outputs: Vec<TensorId>,// dead data — never read at runtime
    pub steps: Vec<Step>,
    pub scratch_buffers: Vec<ScratchBufferDesc>,
}
```

**Current duplication example** — the `plan()` method in Add, Sub, Mul, Div is 57 lines of identical code differing in exactly two tokens:
```rust
let shader_index = ctx.compile_shader(
    "add",                                               // ← "add"/"sub"/"mul"/"div"
    include_str!("../../shaders/elementwise/add.wgsl"),  // ← add/sub/mul/div.wgsl
    shader_defs,
)?;
```
Everything else — output shape computation, workgroup sizing, immediates encoding, binding layout, dispatch — is character-for-character identical.

### New architecture (after)

```
┌─────────────────┐
│  onyxia-core    │  IR, Operator/Pass traits, plan types, TensorShape, OperatorRegistry
│                 │  (the "framework" — defines the abstractions)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│onyxia- │  │  onyxia-     │  Pipeline orchestrator, built-in passes
│onnx    │  │  compiler    │  (SymbolicResolution, ShapeInference, ConstantFolding, Planning)
│(parser)│  │              │
└────┬───┘  └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
    ┌───────────────┐
    │ onyxia-       │  38 core operators (collapsed families), shaders,
    │ operators     │  optional optimization passes (e.g., ElementwiseFusion)
    │ (optional)    │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ onyxia-       │  GPU executor, PlanExecutor, update_dimensions()
    │ runtime       │
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ onyxia-cli    │  CLI interface, LLM inference
    └───────────────┘
```

**New compilation flow:**
```
ONNX Graph
  → IrGraph::from_onnx()          // Build graph-based IR (no mutation of original)
  → Resolution stage               // Resolve symbolic dimensions where possible
  → Inference stage                 // Forward-pass shape inference (Err, not Unknown)
  → Folding stage                   // Constant fold, store values in TensorDef.value
  → Optimization stage              // User-registered passes (fusion, aliasing, etc.)
  → Planning stage                  // Emit PlannedOps, SKIP folded nodes
  → CompiledModel                   // Output with SymbolicBindings for runtime patching
```

---

### Steps

### Phase 1: Create onyxia-core crate

`onyxia-core` is the foundational crate. It defines the abstractions that all other crates depend on: the graph IR, the `Operator` and `Pass` traits, context types, plan types, and the `OperatorRegistry`. It depends on `onyxia-onnx` for the `Graph` type (used in `IrGraph::from_onnx()`), `DataType`, `AttributeValue`, and `TensorKind`. External dependencies: `petgraph`, `naga`, `naga_oil`, `thiserror`, `tracing`, `half`.

#### 1. Graph IR (`ir.rs`, `ir_builder.rs`)

The graph IR is the central data structure that all compilation passes operate on. Unlike the current approach of mutating a cloned ONNX `Graph` in-place, the IR is purpose-built for compiler transformations.

- **`IrGraph`** — wraps `petgraph::StableGraph<IrNode, ()>` with a side-table `Vec<TensorDef>` for tensor metadata. `StableGraph` is essential — unlike `Graph` from petgraph, stable indices survive node removal during optimization passes, so cached `NodeIndex`/`IrTensorId` values remain valid after graph mutations.
- **`IrNode`** — `op_type: String`, `attributes: HashMap<String, AttributeValue>`, `inputs: Vec<IrTensorId>`, `outputs: Vec<IrTensorId>`, `node_index: NodeIndex`.
- **`TensorDef`** — replaces the current `TensorInfo`. Fields: `name: String`, `dtype: DataType`, `shape: TensorShape`, `kind: TensorKind`, `value: Option<TensorValue>` (populated during constant folding), `initializer: Option<Vec<u8>>`.
- **`IrTensorId`** — newtype index into the tensor side-table.
- **Graph query helpers** on `IrGraph`:
  - `node(id) -> &IrNode`, `node_mut(id) -> &mut IrNode`
  - `tensor(id) -> &TensorDef`, `tensor_mut(id) -> &mut TensorDef`
  - `node_inputs(id) -> Vec<IrTensorId>`, `node_outputs(id) -> Vec<IrTensorId>`
  - `tensor_producer(tensor_id) -> Option<IrNodeId>`, `tensor_consumers(tensor_id) -> Vec<IrNodeId>`
  - `topological_order() -> Vec<IrNodeId>` — replaces the standalone `Scheduler` + `petgraph::visit::Topo` approach. The current scheduler clones the graph just to sort it; the IR does this in-place.
  - `remove_node(id)` — disconnects and removes (safe with `StableGraph`)
  - `replace_node(old, new_op, new_attrs)` — for rewrite passes
- **`IrGraph::from_onnx(graph: &Graph) -> IrGraph`** in `ir_builder.rs` — converts ONNX graph to IR. Preserves topological structure. Copies initializer data into `TensorDef.initializer`. Shapes remain as-is from the ONNX parser (may contain symbolic dimension names). No cloning of the input graph; the IR is built fresh.

#### 2. Types (`types.rs`, `symbolic_expr.rs`)

**`TensorShape`** — extends beyond the current `onyxia-onnx` definition:

```rust
pub enum TensorShape {
    Static(Vec<usize>),           // All dims known
    Symbolic(Vec<SymbolicDim>),   // Mix of static and symbolic dims
    Absent,                       // Optional input not provided
    // NO Unknown variant — operators must return Err for undeterminable shapes
}

pub enum SymbolicDim {
    Fixed(usize),
    Expr(SymbolicExpr),
}
```

This replaces the current `Dynamic(Vec<Dimension>)` with expression-carrying variants that can flow through the pipeline. The Gemma model exports expressions like `sequence_length * num_attention_heads` — the current system eagerly resolves these in Phase 1 and fails if any dimension is missing. The new `Symbolic` variant allows partial resolution: resolved dims become `Fixed`, unresolved ones stay as `Expr` and become immediates at dispatch time.

**`TensorShape::Unknown` is removed entirely.** Currently, operators inconsistently return `Unknown` vs `Err`:
- Cos/Sin/Sqrt clone `ctx.input_shapes[0]` without checking if it's `Unknown` or `Dynamic`
- Softmax has no Unknown/Dynamic/Absent guard at all
- Gather and MatMulF32 return `Unknown` when inputs are unknown (should be `Err` since they can't infer)
- Reshape returns `Unknown` when the shape tensor value isn't known at compile time

With no `Unknown` variant, all operators are forced to return `Err` with a descriptive message, making shape inference failures explicit and immediately debuggable.

**`TensorValue`** — same variants as current: `I64(Vec<i64>)`, `I32(Vec<i32>)`, `F32(Vec<f32>)`, `Bool(Vec<bool>)`, `U8(Vec<u8>)`. With utility methods: `cast()`, `as_i64()`, `as_f32()`, `len()`, etc.

**`symbolic_expr.rs`** — moves from `onyxia-compiler` into `onyxia-core`. The recursive-descent parser/evaluator for dimension expressions. Supports `+`, `-`, `*`, `/`, `%`, parentheses. Types: `SymbolicExpr { Literal(i64), Variable(String), BinOp(..) }`, `BinOpKind { Add, Sub, Mul, Div, Mod }`.

#### 3. Operator trait (`operator.rs`)

```rust
pub trait Operator: Send + Sync {
    /// Human-readable name (e.g., "Add", "MatMulF32", "GroupQueryAttention").
    fn name(&self) -> &str;

    /// Infer output shapes given input tensor defs.
    /// Must return Err if shapes cannot be determined — there is no Unknown variant.
    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>>;

    /// Attempt constant folding. Returns one Option per output.
    /// Default: no folding (all None).
    ///
    /// Note: this currently duplicates logic that also exists in WGSL shaders.
    /// A future improvement would run the actual shader on the GPU to fold,
    /// eliminating this Rust-side duplication. For now, Rust-side folding is
    /// kept because compile() is a pure function that doesn't require GPU access,
    /// and the most critical folding (shape computation chains like
    /// Shape→Gather→Concat→Reshape) operates on tiny integer tensors better
    /// suited to CPU evaluation.
    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        let _ = ctx;
        Ok(vec![])
    }

    /// Emit execution steps (GPU dispatches, buffer copies).
    /// Called only for non-folded nodes.
    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>>;
}
```

#### 4. Context types (`context.rs`)

These replace the current `InferenceContext` and `PlanContext`, with helpers that eliminate the pervasive boilerplate.

**`InferenceCtx`** — provided to `infer_output_shapes()`:
- `input_shape(n) -> Result<&TensorShape>` — returns `Err` on missing/absent input. This eliminates the repeated 8-line match block currently copy-pasted across Gather (2x), MatMulF32 (2x), GQA, and many other operators:
  ```rust
  // This block appears in nearly every operator — InferenceCtx eliminates it:
  let dims = match &ctx.input_shapes[0] {
      TensorShape::Static(dims) => dims,
      TensorShape::Unknown | TensorShape::Absent => return Ok(vec![TensorShape::Unknown]),
      TensorShape::Dynamic(_) => return Err(CodegenError::InvalidShape("...".to_string())),
  };
  ```
- `require_static(n) -> Result<&[usize]>` — extracts static dimensions or returns `Err` with context about which input and which operator failed.
- `input_value(n) -> Option<&TensorValue>` — for constant folding chains.
- `attr_i64(key) -> Result<i64>`, `attr_f32(key) -> Result<f32>`, `attr_ints(key) -> Result<&[i64]>`, etc.

**`FoldCtx`** — provided to `try_fold()`. Extends `InferenceCtx` with:
- `all_inputs_have_values() -> bool`
- `binary_fold_f32(f: fn(f32,f32) -> f32) -> Result<Vec<Option<TensorValue>>>` — handles the full boilerplate: extract both F32 inputs, broadcast shapes, apply the function element-wise, wrap in `Ok(vec![Some(TensorValue::F32(...))])`. This replaces the ~23 identical lines currently duplicated across Add, Sub, Mul, Div, Pow, Max.
- `unary_fold_f32(f: fn(f32) -> f32) -> Result<Vec<Option<TensorValue>>>` — same for unary ops.

**`PlanCtx`** — provided to `plan()`:
- `input(n) -> BufferRef`, `output(n) -> BufferRef`
- `input_tensor(n) -> &TensorDef`, `output_tensor(n) -> &TensorDef`
- `static_dims(shape) -> Result<Vec<usize>>` — extracts static dimensions, errors on symbolic.
- `symbolic_dims(shape) -> Vec<SymbolicDim>` — for dimensions that remain symbolic.
- `compile_shader(label, source, defs) -> Result<ShaderIndex>` — compiles WGSL via `naga_oil::Composer` with `ShaderDefValue` specialization. Deduplicates by label+defs (fixing the current label-only dedup bug).
- `alloc_scratch(desc) -> BufferRef::Scratch(idx)`
- Shader compilation classifies each dimension as **static** (→ `ShaderDefValue::UInt`, enabling downstream WGSL compiler loop unrolling) or **symbolic** (→ immediate value set at dispatch time).

#### 5. Pass trait (`pass.rs`)

```rust
pub enum Stage {
    Resolution,    // Resolve symbolic dimension names to values
    Inference,     // Forward-pass shape inference
    Folding,       // Constant folding
    Optimization,  // User-defined optimization passes (fusion, aliasing, etc.)
    Planning,      // Emit execution plan
}

pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn stage(&self) -> Stage;
    fn run(&self, graph: &mut IrGraph, registry: &OperatorRegistry) -> Result<bool>;
    // Returns true if the graph was modified.
}
```

Passes are standalone objects — they are not owned by operators. This avoids the problem of two related operators (e.g., Mul and Add) having to decide which one "owns" a MulAdd fusion pass. Instead, passes are registered on the compiler pipeline from anywhere: from the `onyxia-operators` crate, from user code, or from third-party crates. Within a stage, passes run in registration order.

#### 6. Plan types (`plan.rs`)

```rust
pub struct CompiledModel {
    pub operations: Vec<PlannedOp>,
    pub shaders: Vec<CompiledShader>,
    pub tensors: TensorRegistry,
    pub inputs: Vec<IrTensorId>,
    pub outputs: Vec<IrTensorId>,
    pub symbolic_bindings: Vec<SymbolicBinding>,
    pub metadata: ModelMetadata,
}

pub struct PlannedOp {
    pub name: String,
    pub steps: Vec<Step>,
    pub scratch_buffers: Vec<ScratchBufferDesc>,
    // Note: the current PlannedOp also has op_type, inputs, outputs fields,
    // but analysis shows these are never read at runtime — the runtime relies
    // entirely on `steps` (with BufferRef::Tensor(id) / BufferRef::Scratch(idx))
    // and `scratch_buffers`.
}

pub enum Step {
    Dispatch {
        shader_index: ShaderIndex,
        bindings: Vec<BindingDesc>,
        workgroups: [u32; 3],
        immediates: Option<Vec<u8>>,
    },
    CopyBuffer {
        src: BufferRef, src_offset: u64,
        dst: BufferRef, dst_offset: u64,
        size: u64,
    },
    WriteBuffer { dst: BufferRef, data: Vec<u8> },
}

pub struct CompiledShader {
    pub label: String,
    pub module: naga::Module,
    pub entry_point: String,
}

pub enum BufferRef { Tensor(IrTensorId), Scratch(usize) }
pub struct BindingDesc { pub buffer: BufferRef, pub read_only: bool }
pub struct ScratchBufferDesc { pub size: u64, pub label: String }
pub type ShaderIndex = usize;

/// Records which immediates correspond to symbolic dimensions,
/// enabling runtime patching when dimensions change.
pub struct SymbolicBinding {
    pub shader_index: ShaderIndex,
    pub immediate_offset: usize,
    pub expr: SymbolicExpr,
}
```

#### 7. Operator registry (`registry.rs`)

`OperatorRegistry` — wraps `HashMap<String, Box<dyn Operator>>`. Methods:
- `new() -> Self` — empty registry.
- `register(name: &str, op: impl Operator) -> &mut Self` — register a single operator. Returns `&mut Self` for chaining.
- `get(name: &str) -> Option<&dyn Operator>` — look up by ONNX op_type name.

No `with_defaults()` here — `onyxia-core` doesn't know about any specific operators. The core operator set is provided by `onyxia-operators` (see Phase 3).

---

### Phase 2: Implement onyxia-compiler as pipeline orchestrator

`onyxia-compiler` becomes a thin crate that depends on `onyxia-core` and `onyxia-onnx`. Its sole responsibility is running the pass pipeline. It provides the four built-in passes and the `CompilerPipeline` orchestrator.

#### 8. `CompilerPipeline`

```rust
pub struct CompilerPipeline {
    registry: OperatorRegistry,
    passes: Vec<Box<dyn Pass>>,  // ordered by (stage, registration order)
}

impl CompilerPipeline {
    /// Create a pipeline with built-in passes (Resolution, Inference, Folding, Planning).
    /// The Optimization stage is empty by default — add passes with `add_pass()`.
    pub fn new(registry: OperatorRegistry) -> Self;

    /// Insert a pass into the appropriate stage (determined by pass.stage()).
    /// Within a stage, passes run in registration order.
    pub fn add_pass(&mut self, pass: impl Pass + 'static) -> &mut Self;

    /// Run the full pipeline.
    pub fn compile(
        &self,
        graph: &Graph,
        dynamic_dimensions: &HashMap<String, usize>,
    ) -> Result<CompiledModel>;
}

/// Convenience function for simple usage without pipeline customization.
pub fn compile(
    graph: &Graph,
    registry: &OperatorRegistry,
    dynamic_dimensions: &HashMap<String, usize>,
) -> Result<CompiledModel>;
```

#### 9. Built-in passes (`passes/` module)

**`SymbolicResolutionPass`** (stage: `Resolution`) — walks all `TensorDef`s in the IR. For each `Symbolic` shape, evaluates `SymbolicDim::Expr` against the provided dimension map. Fully resolved shapes become `Static`. Partially resolved shapes stay `Symbolic` (they'll become immediates at dispatch time). Replaces the current `resolve_dynamic_dimensions()` function which mutates the ONNX graph in-place and requires all-or-nothing resolution.

**`ShapeInferencePass`** (stage: `Inference`) — forward pass in topological order. For each node, builds an `InferenceCtx` from the node's input `TensorDef`s and calls `operator.infer_output_shapes(ctx)`. Updates output `TensorDef.shape`. Since `TensorShape::Unknown` doesn't exist, any shape inference failure is an immediate, descriptive `Err` — no more silent propagation of unresolved shapes that only crash later at planning time.

**`ConstantFoldingPass`** (stage: `Folding`) — forward pass in topological order. Initializes `TensorDef.value` from initializers (weights, constants). For each node, if input values are available, calls `operator.try_fold(ctx)`. Stores results in output `TensorDef.value`. This value propagation enables chains like Shape→Gather→Concat→Reshape to be fully resolved at compile time.

**`PlanningPass`** (stage: `Planning`) — iterates topological order. **Skips** any node where ALL output tensors have `TensorDef.value` set (i.e., the node was successfully constant-folded and its results are already known). For remaining nodes, builds a `PlanCtx` and calls `operator.plan()`. Classifies dimensions as shader defs (static → `ShaderDefValue::UInt`) or immediates (symbolic → runtime-patched). Assembles the final `CompiledModel`. This fixes the current behavior where even fully-folded nodes like `Shape` and `Constant` generate unnecessary `WriteBuffer` GPU steps.

---

### Phase 3: Create onyxia-operators crate

`onyxia-operators` is a collection of the core 38 ONNX operators. It depends on `onyxia-core` (for the `Operator` trait, `Pass` trait, IR types, and plan types). This crate is **optional** — users can use `onyxia-core` + `onyxia-compiler` with their own operator implementations if they don't need the standard set.

#### 10. Move shaders

Move the 37 `.wgsl` shader files from `crates/onyxia-compiler/shaders/` to `crates/onyxia-operators/shaders/`, keeping the same directory structure:

```
shaders/
├── activation/       gelu.wgsl, tanh.wgsl, softmax.wgsl
├── attention/        gqa_concat_kv.wgsl, gqa_output.wgsl, gqa_scores.wgsl,
│                     gqa_softmax.wgsl, gqa_update_kv.wgsl, rotary_embedding.wgsl
├── elementwise/      add.wgsl, sub.wgsl, mul.wgsl, div.wgsl, pow.wgsl, max.wgsl,
│                     neg.wgsl, sqrt.wgsl, cos.wgsl, sin.wgsl, equal.wgsl,
│                     greater.wgsl, where.wgsl, cast.wgsl
├── indexing/         gather.wgsl, transpose.wgsl, concat.wgsl, slice.wgsl,
│                     expand.wgsl, trilu.wgsl, scatternd.wgsl, constantofshape.wgsl,
│                     range.wgsl
├── matmul/           matmul_f32.wgsl, matmul_q4.wgsl
├── normalization/    rmsnorm.wgsl
└── reduction/        reducemean.wgsl, reducesum.wgsl
```

Each operator uses `include_str!()` to embed its shader source at compile time. Shaders are co-located with operators to keep operator definitions self-contained — this also simplifies the "custom operator" story since users can define operators with their own shaders in their own crates.

#### 11. Collapsed operator families

The biggest deduplication win. Currently, the 6 binary elementwise operators and 5 unary operators are near-identical files with ~460+ redundant lines.

**`BinaryElementwiseOp`** — single struct parameterized by name, shader source, and fold function:

```rust
pub struct BinaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(f32, f32) -> f32,
}

impl BinaryElementwiseOp {
    pub fn add() -> Self {
        Self { name: "Add", shader_source: include_str!("../shaders/elementwise/add.wgsl"), fold_fn: |a, b| a + b }
    }
    pub fn sub() -> Self { /* ... */ }
    pub fn mul() -> Self { /* ... */ }
    pub fn div() -> Self { /* ... */ }
    pub fn pow() -> Self { /* ... */ }
    pub fn max() -> Self { /* ... */ }
}

impl Operator for BinaryElementwiseOp {
    fn name(&self) -> &str { self.name }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>> {
        broadcast_shapes(ctx) // shared NumPy-style broadcasting
    }

    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        ctx.binary_fold_f32(self.fold_fn)
    }

    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>> {
        // Shared: output shape, workgroup sizing, immediates encoding
        // (num_elements, a_size, b_size), binding layout (2 read-only + 1 output),
        // shader compilation, dispatch. ~55 lines, written once.
    }
}
```

**`UnaryElementwiseOp`** — same pattern for Cos, Sin, Sqrt, Neg, Tanh:

```rust
pub struct UnaryElementwiseOp {
    name: &'static str,
    shader_source: &'static str,
    fold_fn: fn(f32) -> f32,
}
```

**`ComparisonOp`** — for Equal, Greater. Shared planning, different shader and fold function.

**`ReductionOp`** — for ReduceSum, ReduceMean. Shared axis-based shape inference and planning, different shader.

**Individual implementations** for complex ops that don't fit a family pattern:

| Category | Operators |
|----------|----------|
| **Activation** | Gelu, Softmax |
| **Normalization** | RmsNorm (SimplifiedLayerNormalization) |
| **Matrix** | MatMulF32, MatMulNBits |
| **Metadata/constants** | Constant, ConstantOfShape, Shape |
| **Shape manipulation** | Reshape, Unsqueeze, Transpose, Concat, Expand |
| **Indexing** | Gather, Slice, ScatterND, Range, Trilu |
| **Type conversion** | Cast |
| **Conditional** | Where |
| **Attention** | RotaryEmbedding, GroupQueryAttention |

#### 12. Core operator registry

```rust
/// Returns an OperatorRegistry pre-populated with all 38 core operators.
/// Users can register additional custom operators on top.
pub fn core_operator_registry() -> OperatorRegistry {
    let mut registry = OperatorRegistry::new();
    registry.register("Add", BinaryElementwiseOp::add());
    registry.register("Sub", BinaryElementwiseOp::sub());
    // ... all 38 operators
    registry
}
```

#### 13. Optional optimization passes

`onyxia-operators` can ship optimization passes that understand the core operator set. These are not built into the compiler — users add them explicitly:

```rust
let mut pipeline = CompilerPipeline::new(core_operator_registry());
pipeline.add_pass(onyxia_operators::passes::ElementwiseFusionPass);
let model = pipeline.compile(&graph, &dims)?;
```

Example future passes:
- `ElementwiseFusionPass` — fuse chains of elementwise ops into a single shader dispatch
- `BufferAliasingPass` — Reshape/Unsqueeze/Transpose currently emit `CopyBuffer` but could alias buffers when shapes are compatible
- `ComputePassBatchingPass` — batch independent dispatches into single compute passes (the current `group_into_passes()` stub)

---

### Phase 4: Support dynamic dimension updates

#### 14. Symbolic bindings in `CompiledModel`

`CompiledModel.symbolic_bindings` records which immediates in which shaders correspond to symbolic dimensions. Format: `SymbolicBinding { shader_index, immediate_offset, expr: SymbolicExpr }`.

During the Planning pass, when a dimension is `SymbolicDim::Expr(expr)`, the planner:
1. Evaluates the expression with current dimension values → encodes result as an immediate.
2. Records a `SymbolicBinding` so the runtime can re-evaluate and patch the immediate when dimensions change.
3. Dimensions that were statically resolved become `ShaderDefValue::UInt` constants → baked into the shader. These enable WGSL compiler loop unrolling but require shader recompilation if the dimension changes.

#### 15. Runtime dimension patching

**`PlanExecutor::update_dimensions(&mut self, dims: &HashMap<String, usize>)`** in `onyxia-runtime`:
- Evaluates each `SymbolicBinding.expr` with the new dimension values.
- Patches immediates in-place — no recompilation needed for symbolic dimensions.
- For dimensions that were compiled as shader defs (statically resolved), flags affected shaders for selective recompilation.
- This enables `LlmSession` to change sequence lengths between prefill and decode without full recompilation. Currently, the GQA operator hardcodes `past_seq_len = 0` with a TODO noting "This will be wrong for decode!" — symbolic dimensions fix this.

---

### Phase 5: Update downstream crates

#### 16. Update onyxia-onnx

`TensorShape` and `SymbolicExpr` move to `onyxia-core`. `onyxia-onnx` depends on `onyxia-core` for shared types (`DataType`, `TensorKind`, `AttributeValue`, `TensorShape`) and re-exports them. The ONNX parser produces `Graph` using core types. The conversion from ONNX dimension annotations to `TensorShape::Symbolic(vec![...])` happens in the parser, replacing the current `Dynamic(Vec<Dimension>)` representation.

#### 17. Update onyxia-runtime

Depend on `onyxia-core` (for `CompiledModel`, `Step`, `BufferRef`, etc.) instead of `onyxia-compiler`. `Runtime::load_model()` accepts `CompiledModel`. `PlanExecutor` gains `update_dimensions()`. At runtime, `PlanExecutor::create_pipelines()` creates `wgpu::ShaderModule` from `ShaderSource::Naga(Cow::Owned(module.clone()))` — no WGSL parsing at runtime, same as current.

#### 18. Update onyxia-cli

Depend on `onyxia-core`, `onyxia-compiler`, `onyxia-operators`, `onyxia-runtime`. Usage becomes:

```rust
let graph = onyxia_onnx::load_and_parse_model(path)?;
let registry = onyxia_operators::core_operator_registry();
let model = CompilerPipeline::new(registry).compile(&graph, &dims)?;
let runtime = Runtime::new().await?;
let executor = runtime.load_model(model).await?;
```

`LlmSession` can use `executor.update_dimensions()` for sequence length changes between prefill/decode instead of full recompilation.

---

### Phase 6: Migrate tests

#### 19. Operator unit tests → `onyxia-operators`

Shape inference, constant folding, and plan-generation tests for each operator move to `crates/onyxia-operators/tests/`. These test operator behavior, not compiler passes. The current per-operator test boilerplate (each operator file has its own `create_*_test_graph()` helper, ~30 lines of near-identical graph construction) is replaced with shared test utilities similar to the existing `make_binary_elementwise_graph()` and `make_unary_graph()` helpers in `onyxia-runtime/tests/common/mod.rs`.

#### 20. GPU integration tests → `onyxia-operators`

The GPU correctness tests currently in `crates/onyxia-runtime/tests/` (`activation_ops_test.rs`, `elementwise_ops_test.rs`, `matmul_ops_test.rs`, etc.) move to `crates/onyxia-operators/tests/` as integration tests that depend on `onyxia-runtime`. This consolidates the currently split test story — currently, operator behavior is tested in one crate (compiler) and operator correctness in another (runtime). The `common/mod.rs` test helpers move here too.

#### 21. Compiler pass tests stay in `onyxia-compiler`

Tests that validate the pass pipeline stays in `crates/onyxia-compiler/tests/`: symbolic resolution correctly resolves expressions, shape inference propagates through a multi-node graph, constant folding marks the right values, planning skips folded nodes.

#### 22. Runtime tests stay in `onyxia-runtime`

Tests for `PlanExecutor`, `Tensor` management, `update_dimensions()`, buffer allocation stay in `crates/onyxia-runtime/tests/`.

---

### Phase 7: Initial migration (representative subset)

Design and implement the full architecture (Phases 1–6) but migrate only a representative subset of ~12 operators first to validate the design. The remaining 26 operators are migrated in a follow-up task.

#### 23. Migrate these operators first

| Operator | Why it validates the architecture |
|----------|----------------------------------|
| Add, Sub, Mul, Div | Validates `BinaryElementwiseOp` collapsed family, broadcasting |
| Cos, Sqrt, Neg | Validates `UnaryElementwiseOp` collapsed family |
| Reshape | Validates fold-skip (Shape→Gather→Concat→Reshape chains), shape manipulation |
| Gather | Validates multi-input shape inference, axis-based indexing |
| MatMulF32 | Validates complex planning with workgroup computation, multi-dimensional dispatch |
| Softmax | Validates axis-based ops with reduction semantics |
| GQA | Validates multi-step operators (5 dispatches), complex attention patterns |

#### 24. Migrate remaining 26 operators

Follow-up task after validating the architecture with the subset above: Pow, Max, Sin, Tanh, Equal, Greater, Where, Gelu, RmsNorm, MatMulNBits, Constant, ConstantOfShape, Shape, Unsqueeze, Transpose, Concat, Expand, Slice, ScatterND, Range, Trilu, ReduceSum, ReduceMean, Cast, RotaryEmbedding.

---

### New dependency graph

```
onyxia-core          (IR, Operator/Pass traits, plan types, TensorShape, OperatorRegistry)
  ↑         ↑
onyxia-onnx  onyxia-compiler     (ONNX parser)    (pipeline orchestrator, built-in passes)
  ↑              ↑
  └──────┬───────┘
         ↑
    onyxia-operators   (38 core ops, shaders, optional optimization passes)
         ↑
    onyxia-runtime     (GPU executor, PlanExecutor, update_dimensions)
         ↑
    onyxia-cli         (CLI interface, LLM inference)
```

### Verification

- `cargo nextest run -p onyxia-core` — IR construction, graph queries, `TensorShape` utilities, symbolic expression parsing/evaluation
- `cargo nextest run -p onyxia-compiler` — pass pipeline tests: resolution resolves expressions, inference propagates shapes, folding marks values, planning skips folded nodes
- `cargo nextest run -p onyxia-operators` — operator unit tests: shape inference, fold, plan per family and per complex operator
- `cargo nextest run -p onyxia-operators --run-ignored=all` — GPU integration tests (require GPU, migrated from onyxia-runtime)
- `cargo nextest run -p onyxia-runtime` — plan executor, tensor management, `update_dimensions()` patching
- `cargo clippy --workspace` — no warnings across all crates
- End-to-end: load Gemma 3 270m model → compile → run inference → verify output matches current behavior
- Verify that constant-folded Shape→Gather→Concat→Reshape chains generate **zero** GPU operations in the `CompiledModel`

### Design rationale

- **Graph-based IR over linear SSA**: better for pattern-matching, subgraph replacement, and node removal. Optimization passes need to inspect neighborhoods and replace subgraphs — a flat instruction list makes this awkward.
- **`petgraph::StableGraph`**: unlike `petgraph::Graph`, node indices remain valid after removal. Critical for pass correctness — a pass can iterate nodes while another part of the code holds indices.
- **Passes are standalone, not operator-owned**: avoids the problem of two related operators (e.g., Mul and Add) needing to decide which one "owns" a fused MulAdd pass. Passes are registered on the pipeline from anywhere.
- **Named stages for pass ordering**: Resolution → Inference → Folding → Optimization → Planning. Passes declare their stage; within a stage they run in registration order. Simple mental model — no need for complex dependency graphs between passes.
- **Visitor-based passes over pattern-rewrite rules**: more flexible. Complex passes like GQA attention restructuring can walk the graph freely. Pattern-based matching can be built as a library on top of the visitor pattern if needed later.
- **`onyxia-core` as the framework crate**: all trait definitions, IR types, and plan types live here so that operators and passes can be defined in any crate. `onyxia-compiler` is just the driver. `onyxia-operators` is an optional collection.
- **Shaders co-located with operators**: keeps operator definitions self-contained. Custom operators can ship their own shaders.
- **Collapsed operator families via shared impls**: `BinaryElementwiseOp`, `UnaryElementwiseOp`, `ComparisonOp`, `ReductionOp` — one `impl Operator` per family, parameterized by name/shader/fold-function. Complex ops keep individual `impl Operator`.
- **Constant folding via Rust-side `try_fold()`**: kept for now because `compile()` is a pure function (no GPU access needed). The duplication of logic between Rust and WGSL is a known trade-off — future work could introduce GPU-based folding by making compilation GPU-aware, which would eliminate the Rust-side fold entirely for compute ops while keeping it for shape/metadata ops (which are tiny integer computations better suited to CPU).
- **Skip folded nodes in planning** (rather than full DCE pass): simpler implementation, addresses the immediate problem. A full dead-code elimination pass that removes nodes and reclaims tensors can be added later as an optimization pass if needed.
- **Symbolic dimensions throughout**: dimensions carry expressions through the pipeline. Static dims → shader defs (loop unrolling). Symbolic dims → immediates (runtime-patchable). This enables `update_dimensions()` without recompilation.
- **Design all 38 operators, migrate 12 first**: validates the architecture with a representative subset spanning all complexity levels (simple elementwise → complex multi-dispatch attention) before committing to migrating everything.
