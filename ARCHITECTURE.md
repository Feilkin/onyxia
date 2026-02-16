# Onyxia Architecture Overview

## System Design

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust. It compiles ONNX operator graphs into WGSL compute shaders and executes them on the GPU via `wgpu`.

The system is organized into six crates with clear responsibility boundaries:

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│ onyxia-onnx │────▶│ onyxia-core  │     │onyxia-operators│     │onyxia-runtime │
│             │     │              │     │                │     │               │
│ Parse ONNX  │     │ IR, traits,  │     │ 40 operator    │     │ Execute on    │
│ protobuf    │     │ plan types   │     │ impls + WGSL   │     │ GPU via wgpu  │
└─────────────┘     └──────┬───────┘     └────────────────┘     └───────┬───────┘
                           │                                           │
                    ┌──────┴───────┐                            ┌──────┴───────┐
                    │   onyxia-    │                            │  onyxia-cli  │
                    │   compiler   │                            │              │
                    │              │                            │ CLI tools,   │
                    │ Pass pipeline│───────────────────────────▶│ LLM runner   │
                    │ + codegen    │                            │              │
                    └──────────────┘                            └──────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf models into a structured `Graph` API |
| `onyxia-core` | IR graph, operator/pass traits, plan types, tensor types, operator registry |
| `onyxia-operators` | 40 built-in ONNX operator implementations with WGSL shaders |
| `onyxia-compiler` | Pass-based compilation pipeline (resolution → folding → inference → planning) |
| `onyxia-runtime` | GPU execution engine via `wgpu` |
| `onyxia-cli` | CLI for model inspection, DOT export, validation, and text generation |

## Crate Details

### onyxia-onnx

Parses ONNX protobuf model files into a stable `Graph` representation independent of the raw protobuf schema.

**Responsibilities:**
- Parse ONNX `.onnx` files via `prost` (protobuf code generated at build time from `proto/onnx.proto`)
- Produce a structured `Graph` with typed nodes, tensors, and metadata
- DOT graph export for visualization (full, layers, summary views)

**Key Types:**
- `Graph` — Nodes, tensor metadata, input/output mappings
- `Node` — Operator type, inputs, outputs, attributes, domain
- `TensorInfo` — Name, shape, data type, kind, optional initializer data
- `TensorShape` — `Static`, `Dynamic` (named dims), `Unknown`, or `Absent`
- `DataType` — F32, F16, I32, I64, U8, U32, Bool, Q4, Q8

### onyxia-core

The foundational crate that all other Onyxia crates depend on. Defines the intermediate representation, extension traits, and output plan types.

**Responsibilities:**
- **IR graph** (`IrGraph`) — directed graph where nodes are operators and edges are tensor value flows, backed by `petgraph::StableGraph` for safe mutation during passes
- **Operator trait** — three-method interface for shape inference, constant folding, and GPU planning
- **Pass trait** — graph transformation interface with staged execution
- **Plan types** — `CompiledModel`, `PlannedOp`, `Step`, `CompiledShader` (the output of compilation)
- **Operator registry** — dynamic dispatch from op_type strings to `Operator` implementations
- **Tensor types** — `TensorShape` (with symbolic dimension support), `TensorValue`, `DataType`
- **Symbolic expressions** — parser and evaluator for arithmetic dimension expressions (e.g., `seq_len * num_heads`)
- **Context types** — `InferenceCtx`, `FoldCtx`, `PlanCtx` passed to operators during each phase

**Key Types:**

| Type | Purpose |
|------|---------|
| `IrGraph` | Directed graph of operators (nodes) and tensor flows (edges) |
| `IrNode` | Operator with op_type, attributes, input/output edge IDs |
| `IrEdge` | Tensor metadata: name, dtype, shape, compile-time data (`EdgeData`) |
| `EdgeData` | `Runtime` (no data), `Initializer` (raw bytes), or `Constant` (folded value) |
| `Operator` | Trait: `infer_output_shapes()`, `try_fold()`, `plan()` |
| `Pass` | Trait: `stage()`, `run(graph, registry)` |
| `Stage` | Enum: `Resolution`, `Folding`, `Inference`, `Optimization`, `Planning` |
| `CompiledModel` | Final output: operations, shaders, tensor registry, I/O mappings |
| `Step` | `Dispatch` (compute shader), `CopyBuffer`, `WriteBuffer` |
| `OperatorRegistry` | Maps op_type strings to `Box<dyn Operator>` |
| `TensorShape` | `Static(Vec<usize>)`, `Symbolic(Vec<SymbolicDim>)`, `Absent`, `Unknown` |
| `TensorValue` | Compile-time constant tensor (data + shape + dtype) |
| `InferenceCtx` | Read-only context for shape inference (input shapes, values, attributes) |
| `FoldCtx` | Context for constant folding (wraps `InferenceCtx`, adds initializer parsing) |
| `PlanCtx` | Mutable context for planning (buffer refs, shader compilation, scratch allocation) |

### onyxia-operators

Implements 40 ONNX operators organized into operator families (for code reuse) and individual implementations.

**Responsibilities:**
- Implement `Operator` trait for each supported ONNX operation
- Provide WGSL compute shaders (in `shaders/` directory)
- Export `core_operator_registry()` — a pre-populated `OperatorRegistry`

**Operator Families** (shared logic, parameterized by operation):

| Family | Operators | Shared Logic |
|--------|-----------|-------------|
| Binary elementwise | Add, Sub, Mul, Div, Pow, Max | Broadcasting, shape inference, f32 folding |
| Unary elementwise | Cos, Sin, Sqrt, Neg, Tanh | Pass-through shape, element-wise dispatch |
| Comparison | Equal, Greater | Broadcasting, bool output |
| Reduction | ReduceSum, ReduceMean | Axis handling, keepdims |

**Individual Operators** (unique logic):

| Category | Operators |
|----------|-----------|
| Activation | Gelu, Softmax |
| Normalization | RmsNorm (SimplifiedLayerNormalization) |
| Matrix multiplication | MatMulF32, MatMulNBits (4-bit quantized) |
| Metadata | Constant, ConstantOfShape, Shape |
| Shape manipulation | Reshape, Unsqueeze, Transpose, Concat, Expand |
| Indexing | Gather, Slice, ScatterND, Range, Trilu |
| Type conversion | Cast |
| Conditional | Where |
| Attention | RotaryEmbedding, GemmaRotaryEmbedding, MicrosoftRotaryEmbedding, GroupQueryAttention |

**Shader Organization:**
```
shaders/
├── activation/        # gelu.wgsl, softmax.wgsl
├── attention/         # rotary_embedding.wgsl, group_query_attention.wgsl
├── elementwise/       # add.wgsl, sub.wgsl, mul.wgsl, div.wgsl, ...
├── indexing/          # gather.wgsl, slice.wgsl, ...
├── matmul/            # matmul_f32.wgsl, matmul_nbits.wgsl
├── normalization/     # rms_norm.wgsl
└── reduction/         # reduce_sum.wgsl, reduce_mean.wgsl
```

### onyxia-compiler

Orchestrates the compilation pipeline that transforms an ONNX `Graph` into a `CompiledModel` ready for GPU execution.

**Responsibilities:**
- Convert ONNX `Graph` → `IrGraph` (via `IrGraph::from_onnx()`)
- Run a staged pass pipeline over the IR
- Build the final `CompiledModel` with planned operations, compiled shaders, and tensor registry

**Pass Pipeline:**

The compiler runs five stages in order. Each stage contains one or more passes:

```
┌─────────────────────────────────┐
│ 1. Resolution                   │  Substitute symbolic dimensions with user-provided values
│    ├ SymbolicResolutionPass      │  Evaluate symbolic expressions → static dims
│    └ InitializeConstantsPass     │  Parse ONNX initializer bytes into TensorValues
├─────────────────────────────────┤
│ 2. Folding                      │  Evaluate constant operations at compile time
│    └ ConstantFoldingPass         │  Call Operator::try_fold() for constant-input nodes
├─────────────────────────────────┤
│ 3. Inference                    │  Propagate shapes through the graph
│    └ ShapeInferencePass          │  Call Operator::infer_output_shapes() (skips folded nodes)
├─────────────────────────────────┤
│ 4. Optimization                 │  (Custom user passes can be inserted here)
│    └ (user-defined passes)       │  Dead code elimination, operator fusion, etc.
├─────────────────────────────────┤
│ 5. Planning                     │  Generate GPU execution steps
│    └ PlanningPass                │  Call Operator::plan() for each remaining node
└─────────────────────────────────┘
```

Running constant folding *before* shape inference is deliberate — nodes whose inputs are all constants get folded away entirely, so shape inference never needs to run on them.

**Custom Passes:**

Users can insert passes into any stage:

```rust
let mut pipeline = CompilerPipeline::new(dynamic_dimensions);
pipeline.add_pass(MyOptimizationPass);  // runs in Stage::Optimization
let model = pipeline.compile(&graph, &registry)?;
```

**Partial Compilation:**

The CLI tools use `run_until_stage()` to stop the pipeline early for inspection:

```rust
pipeline.run_until_stage(&mut ir_graph, &registry, Stage::Inference)?;
// Shapes are resolved, but no GPU planning has happened
```

### onyxia-runtime

Executes a `CompiledModel` on the GPU using `wgpu` as the hardware abstraction layer.

**Responsibilities:**
- Initialize GPU device and queue via `wgpu` (cross-platform: Vulkan, DX12, Metal)
- Create compute pipelines from `naga::Module` shaders (via `ShaderSource::Naga`)
- Allocate GPU buffers for all tensors and scratch space
- Upload initial data (model weights, constants) to GPU
- Execute compute passes (`Dispatch`, `CopyBuffer`, `WriteBuffer`)
- Download output tensors from GPU → CPU

**Key Types:**
- `Runtime` — GPU device management, creates `PlanExecutor` instances
- `PlanExecutor` — Materializes a `CompiledModel` into GPU pipelines/buffers, exposes `run(inputs) -> outputs`
- `Tensor` — CPU-side tensor for input/output data interchange (typed access via `from_vec()`, `to_vec()`)

**Execution flow:**
```rust
let runtime = Runtime::new().await?;
let mut executor = runtime.load_model(compiled_model).await?;
let outputs = executor.run(&[("input_ids", input_tensor)])?;
```

### onyxia-cli

Command-line interface for model inspection, debugging, validation, and text generation.

**Subcommands:**

| Command | Purpose |
|---------|---------|
| `dot` | Generate Graphviz DOT files with filtering and depth control |
| `inspect` | Show model structure: layers, shapes, dynamic dimensions |
| `inspect-node` | Detailed view of specific nodes (attributes, I/O shapes, values) |
| `list-nodes` | List/filter nodes by op type or name pattern, show summary stats |
| `inspect-tensor` | Inspect tensor metadata and values, list constants |
| `trace-node` | Trace data flow around a node (upstream/downstream, configurable depth) |
| `validate` | Validate model through partial compilation (resolution → folding → inference) |
| `run-model` | End-to-end text generation with tokenizer, sampling, and streaming output |

## Key Architectural Decisions

### 1. Extensible Operator System

Each ONNX operation is implemented as a struct implementing the `Operator` trait:

```rust
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;

    /// Determine output shapes from input shapes and attributes.
    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> Result<Vec<TensorShape>>;

    /// Evaluate at compile time if all inputs are constants (default: no folding).
    fn try_fold(&self, ctx: &FoldCtx) -> Result<Vec<Option<TensorValue>>> {
        Ok(vec![])
    }

    /// Emit GPU execution steps (shader dispatches, buffer copies).
    fn plan(&self, ctx: &mut PlanCtx) -> Result<Vec<Step>>;
}
```

Operators are registered by name in an `OperatorRegistry`. The built-in registry is provided by `onyxia_operators::core_operator_registry()`, and users can add custom operators:

```rust
let mut registry = core_operator_registry();
registry.register("MyCustomOp", MyCustomOperator);
let model = compile(&graph, &registry, &dynamic_dimensions)?;
```

**Why this design:**
- Operators own their shape inference, folding, and planning logic — no centralized match blocks
- One ONNX node can emit multiple GPU steps (e.g., multi-pass algorithms with scratch buffers)
- New operators are added without modifying existing code

### 2. Graph-Based IR with Edge-Centric Data Model

The IR graph (`IrGraph`) uses edges to represent tensor value flows between operators. Edges carry all tensor metadata:

- **Name, dtype, shape** — tensor identity
- **`EdgeData`** — one of three states:
  - `Runtime` — value arrives at inference time (graph inputs, operator outputs)
  - `Initializer(Vec<u8>)` — raw weight bytes from the ONNX file, parsed on demand
  - `Constant(TensorValue)` — fully evaluated at compile time (from folding)

During constant folding, an edge transitions from `Initializer` → `Constant` (or `Runtime` → `Constant` if all producer inputs were constant), and the producing operator node is removed from the graph. Downstream consumers reference the same edge and see the constant value directly.

This design avoids a separate "value node" vs "operator node" distinction — all data flows through edges.

### 3. Staged Compilation Pipeline

The compiler pipeline runs passes in five fixed stages. The key design choice is running **constant folding before shape inference**:

1. **Resolution** — Symbolic dims → static values
2. **Folding** — Evaluate constant subgraphs at compile time
3. **Inference** — Propagate shapes (skipping already-folded nodes)
4. **Optimization** — Custom graph transformations
5. **Planning** — Emit GPU execution steps

This ordering means operations like `Shape → Gather → Concat → Reshape` (common in transformer models) get folded away entirely at compile time, and shape inference never runs on them.

### 4. Compiler Compiles Shaders, Runtime Executes Them

Shader compilation happens at compile time via `naga_oil`. The runtime receives pre-compiled `naga::Module` objects and only needs to create GPU pipelines:

```rust
// Compiler: WGSL text + shader defs → naga::Module (via naga_oil Composer)
let module = composer.make_naga_module(NagaModuleDescriptor {
    source: include_str!("../../shaders/elementwise/add.wgsl"),
    shader_defs: HashMap::from([("WORKGROUP_SIZE", ShaderDefValue::UInt(256))]),
    ..Default::default()
})?;

// Runtime: naga::Module → wgpu pipeline (no WGSL parsing)
let shader = device.create_shader_module(ShaderModuleDescriptor {
    source: ShaderSource::Naga(Cow::Owned(module)),
    ..
});
```

**Boundary:** `naga_oil` is a compiler/operators dependency only. The runtime never touches WGSL text or shader defs.

### 5. WGSL Compute Shaders via naga_oil

WGSL is the shader language, preprocessed by `naga_oil`:

- **Shader defs** for specialization: `#ifdef`, `#if`, `#{VALUE}`
- **Shader composition**: `#import`, `#define_import_path`
- WGSL source files are embedded via `include_str!()` at Rust compile time and preprocessed at plan time

```wgsl
#if QUANT_BITS == 4
    const PACK_FACTOR: u32 = 8u;
#endif

var<workgroup> tile: array<f32, #{TILE_SIZE} * #{BLOCK_SIZE}>;
```

### 6. Symbolic Dimension Resolution

ONNX models use symbolic dimensions (e.g., `[batch_size, sequence_length, 768]`). Onyxia supports full arithmetic expressions in dimension names (as used by PyTorch-exported models):

```
sequence_length * num_attention_heads
(batch_size + 1) * 2
```

The `SymbolicResolutionPass` evaluates these expressions using user-provided dimension values and produces fully static shapes before any other processing.

```rust
let dynamic_dimensions = HashMap::from([
    ("batch_size".to_string(), 1),
    ("sequence_length".to_string(), 512),
]);
let model = compile(&graph, &registry, &dynamic_dimensions)?;
// All shapes in CompiledModel are TensorShape::Static
```

### 7. Async GPU, Sync CLI

The runtime uses `async` for GPU operations (wgpu APIs are async):

```rust
let runtime = Runtime::new().await?;
let executor = runtime.load_model(model).await?;
```

The CLI uses `pollster::block_on` for synchronous entry points.

### 8. Tokenization and Sampling in CLI Only

ONNX models don't include tokenization or sampling. The core library (`onyxia-onnx`, `onyxia-core`, `onyxia-compiler`, `onyxia-runtime`) processes tensors. Tokenization, chat templates, and sampling are implemented in `onyxia-cli` for the `run-model` command, using the `tokenizers` and `minijinja` crates.

## Data Flow

```
User Code                  Compiler                  Runtime                  GPU
────────────────────────────────────────────────────────────────────────────────────
                        compile(graph,
graph = load_model()    registry,
dims = {...}        ──▶   dims)
                          │
                          ├ from_onnx()        ──▶ IrGraph
                          ├ SymbolicResolution  ──▶ Static shapes
                          ├ InitializeConstants ──▶ Parse weights
                          ├ ConstantFolding     ──▶ Fold subgraphs
                          ├ ShapeInference      ──▶ All shapes known
                          ├ Planning            ──▶ GPU steps
                          └─────────────────────── CompiledModel
                                                       │
                                                load_model(model)
                                                       │
                                                Create pipelines  ──▶ GPU pipelines
                                                Allocate buffers   ──▶ GPU memory
                                                Upload weights     ──▶ GPU buffers
                                                       │
inputs = [("x", t)] ──────────────────────▶ executor.run(inputs)
                                                       │
                                                 Upload inputs     ──▶ GPU buffers
                                                 Execute steps     ◀─▶ Compute shaders
                                                 Download outputs  ◀── GPU buffers
                                                       │
outputs ◀──────────────────────────────── HashMap<String, Tensor>
```

## Dependency Graph

```
onyxia-onnx  (no internal deps)
     │
onyxia-core  (depends on: onyxia-onnx)
     │
     ├── onyxia-operators  (depends on: onyxia-core, onyxia-onnx)
     ├── onyxia-compiler   (depends on: onyxia-core, onyxia-onnx)
     └── onyxia-runtime    (depends on: onyxia-core, onyxia-onnx)
                │
          onyxia-cli  (depends on: all crates)
```

## Technology Choices

| Technology | Purpose |
|-----------|---------|
| `wgpu` | GPU hardware abstraction (Vulkan, DX12, Metal) |
| `naga` / `naga_oil` | WGSL shader compilation and preprocessing |
| `petgraph` | Graph data structure for IR (topological sort, stable mutation) |
| `prost` / `prost-build` | Protobuf parsing for ONNX model files |
| `clap` | CLI argument parsing |
| `pollster` | Blocking on async for CLI entry points |
| `tokenizers` | HuggingFace tokenizer support (CLI only) |
| `minijinja` | Chat template rendering (CLI only) |
| `thiserror` | Error type derivation |
| `tracing` | Structured logging / instrumentation |

## Testing Strategy

### Unit Tests (148 passing)

Tests are run with [nextest](https://nexte.st/):

```bash
cargo nextest run                                   # Non-GPU tests
cargo nextest run --run-ignored=all --no-fail-fast   # All tests (requires GPU)
```

- Operators are tested with programmatic `IrGraph` construction — no model files needed
- Shape inference tests verify output shapes for each operator
- Constant folding tests verify compile-time evaluation
- Symbolic expression parser/evaluator tests

### Integration Tests

- End-to-end: `IrGraph` → compile → GPU execute → compare outputs
- Gemma 3 270m model parsing and plan compilation
- GPU tests are marked `#[ignore]` for CI without GPU hardware

### Future Testing

- Numerical validation against ONNX Runtime (atol=1e-4, rtol=1e-3)
- Performance benchmarks (tokens/second, latency, memory)

## Future Work

- **Quantization** — MatMulNBits shader with Q4 dequantization, scale/zero-point handling
- **Attention & KV Cache** — GroupQueryAttention/RotaryEmbedding shaders, persistent KV cache buffers, prefill/decode API
- **Optimizations** — Operator fusion, memory pooling, Flash Attention, workgroup tuning
- **Broader coverage** — More ONNX operators, additional model architectures

## References

- **ONNX Spec:** https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **wgpu:** https://docs.rs/wgpu/latest/wgpu/
- **naga_oil:** https://github.com/bevyengine/naga_oil
- **Flash Attention:** https://arxiv.org/abs/2205.14135
- **Gemma Models:** https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX
