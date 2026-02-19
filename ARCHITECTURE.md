# Onyxia Architecture Overview

## System Design

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust. It uses a dispatch-based execution model where each operator compiles its shaders at compile time and executes GPU work at runtime with full knowledge of input shapes.

The system is organized into six crates with clear responsibility boundaries:

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│ onyxia-onnx │────▶│ onyxia-core  │     │onyxia-operators│     │onyxia-runtime │
│             │     │              │     │                │     │               │
│ Parse ONNX  │     │ IR, traits,  │     │ 3 core         │     │ Execute on    │
│ protobuf    │     │ dispatch     │     │ operators      │     │ GPU via wgpu  │
└─────────────┘     └──────┬───────┘     └────────────────┘     └───────┬───────┘
                           │                                           │
                    ┌──────┴───────┐                            ┌──────┴───────┐
                    │   onyxia-    │                            │  onyxia-cli  │
                    │   compiler   │                            │              │
                    │              │                            │ CLI tools,   │
                    │ Build        │───────────────────────────▶│ model        │
                    │ dispatch     │                            │ inspection   │
                    └──────────────┘                            └──────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf models into a structured `Graph` API |
| `onyxia-core` | IR graph, operator/dispatch traits, compiled model types, tensor types, operator registry |
| `onyxia-operators` | 3 core ONNX operator implementations (Add, Mul, Reshape) |
| `onyxia-compiler` | Simplified pipeline: initialize constants → build dispatch model |
| `onyxia-runtime` | GPU execution engine via `wgpu` with register-based tensor routing |
| `onyxia-cli` | CLI for model inspection, DOT export, and validation |

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

The foundational crate that all other Onyxia crates depend on. Defines the intermediate representation, operator/dispatch traits, and compiled model types.

**Responsibilities:**
- **IR graph** (`IrGraph`) — directed graph where nodes are operators and edges are tensor value flows, backed by `petgraph::StableGraph` for safe mutation during passes
- **Operator trait** — two-method interface: `name()` and `create_dispatch()`
- **Dispatch trait** — runtime execution interface: `dispatch(inputs, ctx) -> outputs`
- **Pass trait** — graph transformation interface with staged execution
- **Compiled model types** — `CompiledModel`, `DispatchEntry`, `WeightRegister`, `RuntimeTensor`
- **Operator registry** — dynamic dispatch from op_type strings to `Operator` implementations
- **Tensor types** — `TensorShape`, `DataType` (no compile-time values or symbolic dimensions)
- **Context types** — `CompileCtx` for operators, `DispatchCtx` for runtime

**Key Types:**

| Type | Purpose |
|------|---------|
| `IrGraph` | Directed graph of operators (nodes) and tensor flows (edges) |
| `IrNode` | Operator with op_type, attributes, input/output edge IDs |
| `IrEdge` | Tensor metadata: name, dtype, shape, compile-time data (`EdgeData`) |
| `EdgeData` | `Runtime` (no data) or `Initializer` (raw weight bytes) |
| `Operator` | Trait: `name()`, `create_dispatch()` |
| `OpDispatch` | Trait: `dispatch(inputs, ctx) -> outputs` |
| `Pass` | Trait: `stage()`, `run(graph, registry)` |
| `Stage` | Enum: `Resolution`, `Folding`, `Inference`, `Optimization`, `Planning` |
| `CompiledModel` | Final output: dispatch entries, registers, I/O mappings, weights |
| `DispatchEntry` | Single operation: dispatch object, input/output register indices, name |
| `WeightRegister` | Weight data to upload: register index, data, shape, dtype |
| `RuntimeTensor` | GPU tensor at runtime: buffer, shape, dtype, size_bytes |
| `OperatorRegistry` | Maps op_type strings to `Box<dyn Operator>` |
| `CompileCtx` | Compile-time context: node metadata, shader compilation |
| `DispatchCtx` | Runtime context: GPU device/queue, pipeline cache |

### onyxia-operators

Implements 3 core ONNX operators for proof-of-concept dispatch-based execution.

**Responsibilities:**
- Implement `Operator` trait for supported ONNX operations
- Provide WGSL compute shaders (in `shaders/` directory)
- Export `core_operator_registry()` — a pre-populated `OperatorRegistry`

**Operator Families** (shared logic, parameterized by operation):

| Family | Operators | Shared Logic |
|--------|-----------|-------------|
| Binary elementwise | Add, Mul | Broadcasting, dynamic shape computation |

**Individual Operators** (unique logic):

| Category | Operators |
|----------|-----------|
| Shape manipulation | Reshape |

**Shader Organization:**
```
shaders/
└── elementwise/    # add.wgsl, mul.wgsl
```

### onyxia-compiler

Orchestrates the compilation pipeline that transforms an ONNX `Graph` into a `CompiledModel` ready for GPU execution.

**Responsibilities:**
- Convert ONNX `Graph` → `IrGraph` (via `IrGraph::from_onnx()`)
- Run the InitializeConstants pass to parse weight data
- Build the final `CompiledModel` with dispatch entries and register routing

**Compilation Pipeline:**

The compiler runs a single built-in pass followed by dispatch model construction:

```
┌─────────────────────────────────┐
│ 1. Resolution Stage             │  Parse weight data from ONNX initializers
│    └ InitializeConstantsPass     │  Convert raw bytes → EdgeData::Initializer
├─────────────────────────────────┤
│ 2. Build Dispatch Model         │  Walk graph, create dispatch objects
│    ├ Topological sort            │  Determine execution order
│    ├ create_dispatch() per node  │  Call Operator::create_dispatch()
│    └ Assign register routing     │  Map tensors to register indices
└─────────────────────────────────┘
```

**Removed Features:**

These features from the previous architecture have been removed:
- Symbolic dimension resolution (no symbolic expressions)
- Constant folding pass (no compile-time evaluation)
- Shape inference pass (shapes computed at runtime)
- Optimization passes (not needed for minimal set)
- Planning pass (replaced by dispatch creation)

**Custom Passes:**

The pass system still exists for extensibility, but no custom passes are currently used:

```rust
let mut pipeline = CompilerPipeline::new();
pipeline.add_pass(MyCustomPass);  // runs in appropriate stage
let model = pipeline.compile(&graph, &registry)?;
```

### onyxia-runtime

Executes a `CompiledModel` on the GPU using a register-based execution model via `wgpu`.

**Responsibilities:**
- Initialize GPU device and queue via `wgpu` (cross-platform: Vulkan, DX12, Metal)
- Create compute pipelines from `naga::Module` shaders (via `ShaderSource::Naga`)
- Manage register file (array of GPU buffers) for tensor routing
- Upload model weights to registers at load time
- Execute dispatch sequence: gather inputs from registers, call `OpDispatch::dispatch()`, store outputs
- Download output tensors from GPU → CPU

**Key Types:**
- `Runtime` — GPU device management, creates `Executor` instances
- `Executor` — Materializes a `CompiledModel`, maintains register file, exposes `run(inputs) -> outputs`
- `Tensor` — CPU-side tensor for input/output data interchange (typed access via `from_vec()`, `to_vec()`)

**Register Machine:**

The runtime maintains a register file (vector of `Option<RuntimeTensor>`) where each ONNX tensor value corresponds to a register index. Operations read inputs from registers and write outputs to registers:

```rust
// For each dispatch entry:
let inputs: Vec<RuntimeTensor> = entry.input_regs
    .iter()
    .map(|&reg| registers[reg].clone())
    .collect();

let outputs = entry.op.dispatch(inputs, &mut dispatch_ctx)?;

for (&reg, tensor) in entry.output_regs.iter().zip(outputs) {
    registers[reg] = Some(tensor);
}
```

**Execution flow:**
```rust
let runtime = Runtime::new().await?;
let mut executor = runtime.load_model(compiled_model).await?;
let outputs = executor.run(&[("input_ids", input_tensor)])?;
```

### onyxia-cli

Command-line interface for model inspection, debugging, and validation.

**Subcommands:**

| Command | Purpose |
|---------|---------|
| `dot` | Generate Graphviz DOT files with filtering and depth control |
| `inspect` | Show model structure: layers, shapes, dynamic dimensions |
| `inspect-node` | Detailed view of specific nodes (attributes, I/O shapes) |
| `list-nodes` | List/filter nodes by op type or name pattern, show summary stats |
| `inspect-tensor` | Inspect tensor metadata and initializer values |
| `trace-node` | Trace data flow around a node (upstream/downstream, configurable depth) |
| `validate` | Validate model through compilation |

## Key Architectural Decisions

### 1. Dispatch-Based Execution Model

Onyxia uses a **dispatch-based** execution model. Each ONNX operator is compiled into a self-contained `OpDispatch` object that executes itself on the GPU given concrete input tensors.

#### Compilation

The compiler walks the ONNX graph in topological order and calls `Operator::create_dispatch()` for each node. The dispatch object captures pre-compiled shaders and attributes. No shape inference happens at compile time — shapes are fully determined at runtime from actual input tensors.

```rust
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;

    /// Create a dispatch object for this operation.
    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>>;
}
```

#### Tensor Routing (Register Machine)

Tensors flow between operations via a register file. Each tensor in the IR graph maps to a register index. Operations read inputs from registers and write outputs to registers:

```rust
// Executor maintains: Vec<Option<RuntimeTensor>>
for entry in &model.entries {
    let inputs = entry.input_regs.iter().map(|&r| registers[r].clone()).collect();
    let outputs = entry.op.dispatch(inputs, &mut ctx)?;
    for (&reg, tensor) in entry.output_regs.iter().zip(outputs) {
        registers[reg] = Some(tensor);
    }
}
```

#### Runtime Dispatch

For each operation, the runtime:
1. Gathers input `RuntimeTensor`s from registers
2. Calls `OpDispatch::dispatch(inputs, ctx)`
3. The operator computes output shapes from input shapes
4. Allocates output buffers, dispatches compute shaders
5. Returns output `RuntimeTensor`s
6. Runtime stores outputs in their designated registers

```rust
pub trait OpDispatch: Send + Sync {
    /// Execute this operation on the GPU.
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>>;
}
```

**Why this design:**
- Operators own their complete execution logic — no centralized orchestration
- Runtime shapes computed from actual input tensors — no compile-time inference needed
- Compilation is simple and fast — no multi-pass analysis
- Register-based routing is cache-friendly and minimizes data movement

### 2. Extensible Operator System

Each ONNX operation is implemented as a struct implementing the `Operator` trait. Operators are registered by name in an `OperatorRegistry`:

```rust
let mut registry = core_operator_registry();
registry.register("MyCustomOp", MyCustomOperator);
let model = compile(&graph, &registry)?;
```

**Why this design:**
- New operators are added without modifying existing code
- One ONNX node creates one dispatch object with all its execution logic
- Clear separation: compilation logic in operators, execution logic in dispatches

### 3. Graph-Based IR with Edge-Centric Data Model

The IR graph (`IrGraph`) uses edges to represent tensor value flows between operators. Edges carry all tensor metadata:

- **Name, dtype, shape** — tensor identity
- **`EdgeData`** — one of two states:
  - `Runtime` — value arrives at inference time (graph inputs, operator outputs)
  - `Initializer(Vec<u8>)` — raw weight bytes from the ONNX file, uploaded to GPU at load time

During compilation, edges maintain their state. At runtime, initializers are uploaded to registers and runtime values flow through the register file.

This design avoids a separate "value node" vs "operator node" distinction — all data flows through edges.

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
#if WORKGROUP_SIZE == 256
    const WG_SIZE: u32 = 256u;
#endif

var<workgroup> tile: array<f32, #{TILE_SIZE}>;
```

### 6. No Symbolic Dimensions

Unlike the previous architecture, this implementation does not support symbolic dimensions. All shapes must be fully static in the ONNX model. This simplifies the implementation significantly.

If symbolic dimensions are needed in the future, they would be handled by a resolution pass that substitutes user-provided values before dispatch model construction.

### 7. Async GPU, Sync CLI

The runtime uses `async` for GPU operations (wgpu APIs are async):

```rust
let runtime = Runtime::new().await?;
let executor = runtime.load_model(model).await?;
```

The CLI uses `pollster::block_on` for synchronous entry points.

## Data Flow

```
User Code                  Compiler                  Runtime                  GPU
────────────────────────────────────────────────────────────────────────────────────
                        compile(graph,
graph = load_model()    registry)
                      ──▶
                          │
                          ├ from_onnx()         ──▶ IrGraph
                          ├ InitializeConstants ──▶ Parse weights
                          ├ Build dispatch model
                          │  ├ Topological sort
                          │  ├ create_dispatch() ──▶ Compile shaders
                          │  └ Assign registers
                          └─────────────────────── CompiledModel
                                                       │
                                                load_model(model)
                                                       │
                                                Allocate registers ──▶ GPU buffers
                                                Create pipelines   ──▶ GPU pipelines
                                                Upload weights     ──▶ GPU buffers
                                                       │
inputs = [("x", t)] ──────────────────────▶ executor.run(inputs)
                                                       │
                                                 Upload inputs     ──▶ GPU buffers (registers)
                                                 For each dispatch:
                                                   Gather inputs from registers
                                                   dispatch(inputs, ctx)
                                                     Compute shapes
                                                     Allocate outputs ──▶ GPU buffers
                                                     Execute shader   ◀─▶ Compute shader
                                                   Store outputs in registers
                                                 Download outputs  ◀── GPU buffers (registers)
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

### Unit Tests

Tests are run with [nextest](https://nexte.st/):

```bash
cargo nextest run                                   # Non-GPU tests
cargo nextest run --run-ignored=all --no-fail-fast  # All tests (requires GPU)
```

- Operators are tested with programmatic `IrGraph` construction — no model files needed
- End-to-end dispatch tests verify GPU execution for simple graphs
- GPU tests are marked `#[ignore]` for CI without GPU hardware

### Future Testing

- Numerical validation against ONNX Runtime (atol=1e-4, rtol=1e-3)
- Performance benchmarks (ops/second, latency, memory)

## Future Work

- **More operators** — Implement remaining ONNX operators (matmul, attention, RmsNorm, etc.)
- **Quantization** — MatMulNBits shader with Q4 dequantization, scale/zero-point handling
- **Attention & KV Cache** — GroupQueryAttention/RotaryEmbedding shaders, persistent KV cache buffers
- **Optimizations** — Operator fusion, memory pooling, Flash Attention, workgroup tuning
- **Symbolic dimensions** — Re-introduce support for dynamic/symbolic shapes if needed
- **Constant folding** — Re-introduce compile-time evaluation for shape operations if needed
- **Broader coverage** — Additional model architectures

## References

- **ONNX Spec:** https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **wgpu:** https://docs.rs/wgpu/latest/wgpu/
- **naga_oil:** https://github.com/bevyengine/naga_oil
- **Flash Attention:** https://arxiv.org/abs/2205.14135
- **Gemma Models:** https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX
