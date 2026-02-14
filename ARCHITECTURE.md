# Onyxia Architecture Overview

## System Design

Onyxia is a **GPU compute shader runtime for ONNX models**, built in 3 main stages:

```
┌─────────────────┐     ┌─────────────────┐      ┌─────────────────┐
│  onyxia-onnx    │───▶│ onyxia-planner  │────▶│ onyxia-runtime  │
│                 │     │                 │      │                 │
│  Parse ONNX     │     │  Compile to     │      │  Execute on GPU │
│  protobuf into  │     │  execution plan │      │  via wgpu (HAL) │
│  graph          │     │  with naga IR   │      │                 │
└─────────────────┘     └─────────────────┘      └─────────────────┘
        │                       │                        │
        │                       │                        │
    ModelProto            ExecutionPlan           Model outputs
```

## Responsibility Boundaries

### onyxia-onnx

**Responsibilities:**
- Parse ONNX protobuf into `Graph` — a stable API independent of the protobuf schema
- Validate graph integrity
- DOT graph export for visualization


**Key Types:**
- `Graph`: Nodes, tensor metadata, input/output mappings
- `Node`: Operator type, inputs, outputs, attributes
- `TensorInfo`: Name, shape, data type, kind
- `TensorShape`: Static, Dynamic, Unknown (not yet inferred), or Absent (optional input not provided)
- `DataType`: F32, F16, I32, I64, U8, U32, Bool, Q4, Q8


### onyxia-planner

**Responsibilities:**
- **Kernel-based shape inference** — each `OpKernel` implements `infer_output_shapes()` for its operation, called once in topological order with value propagation for data-dependent shape inference
- Schedule operations (topological sort with `petgraph`)
- Resolve all dynamic dimensions to static shapes at plan time
- Compile WGSL shaders to `naga::Module` using `naga_oil` (the only crate that touches WGSL)
- Map ONNX operations to GPU steps via `OpKernel` trait + `KernelRegistry`
- Deduplicate compiled shaders across operations
- Produce `ExecutionPlan` with all shapes resolved and shaders pre-compiled

**Key Types:**
- `ExecutionPlan`: Top-level output — operations, shaders, tensor registry, I/O
- `PlannedOp`: One ONNX node → name, op_type, inputs, outputs, steps, scratch buffers
- `Step`: Dispatch (shader + bindings + workgroups), CopyBuffer, WriteBuffer
- `CompiledShader`: label + `naga::Module` + entry point name
- `OpKernel` trait: `infer_output_shapes(&self, ctx: &InferenceContext) -> Result<Vec<TensorShape>>`, optional `try_fold(&self, ctx: &InferenceContext) -> Result<Vec<Option<TensorValue>>>`, and `plan(&self, ctx: &mut PlanContext) -> Result<Vec<Step>>`
- `InferenceContext`: Provides node, graph, input shapes, and constant-folded input values to kernels during shape inference
- `TensorValue`: Represents compile-time constant values for data-dependent shape inference
- `KernelRegistry`: Maps op_type strings to `Box<dyn OpKernel>`
- `PlanContext`: Gives kernels access to node info, tensor shapes, shader compilation, scratch allocation

**Built-in Kernels (19):**
- Elementwise: `AddKernel`, `SubKernel`, `MulKernel`
- Activation: `GeluKernel`
- Normalization: `RmsNormKernel`
- Matrix math: `MatMulF32Kernel`, `MatMulNBitsKernel`
- Metadata: `ConstantKernel`, `ShapeKernel`, `CastKernel`
- Shape manipulation: `ReshapeKernel`, `UnsqueezeKernel`, `TransposeKernel`, `ConcatKernel`
- Indexing/reduction: `GatherKernel`, `ReduceSumKernel`
- Attention: `RotaryEmbeddingKernel`, `GroupQueryAttentionKernel`

### onyxia-runtime

**Responsibilities:**
- Initialize GPU via `wgpu` (hardware abstraction over DX12/Vulkan/Metal)
- Materialize `naga::Module`s into compute pipelines via `ShaderSource::Naga`
- Allocate GPU buffers for tensors and scratch space
- Execute compute passes (Dispatch, CopyBuffer, WriteBuffer)
- Transfer data between CPU and GPU

**Key Types:**
- `Runtime`: GPU device management, `load_model(plan) -> PlanExecutor`
- `PlanExecutor`: Materializes plan into pipelines/buffers, `run(inputs) -> outputs`
- `Tensor`: User-facing CPU tensor for input/output data interchange

### onyxia-cli

**Responsibilities:**
- Model inspection (tensor shapes, operators, metadata)
- DOT graph generation (full, layers, summary views)
- Performance benchmarking (future)

## Key Architectural Decisions

### 1. Extensible Operation System

Operations are added by implementing `OpKernel`:

```rust
pub trait OpKernel: Send + Sync {
    fn name(&self) -> &str;
    
    // Shape inference: given input shapes and values, return output shapes
    fn infer_output_shapes(
        &self,
        ctx: &InferenceContext<'_>,
    ) -> Result<Vec<TensorShape>>;
    
    // Constant folding: compute outputs from constant inputs at compile time
    fn try_fold(
        &self,
        ctx: &InferenceContext<'_>,
    ) -> Result<Vec<Option<TensorValue>>> {
        Ok(vec![None; ctx.node.outputs.len()])
    }
    
    // Planning: generate GPU execution steps
    fn plan(&self, ctx: &mut PlanContext<'_>) -> Result<Vec<Step>>;
}
```

Users register custom kernels via `KernelRegistry`:

```rust
let mut registry = KernelRegistry::with_defaults();
registry.register("MyCustomOp", Box::new(MyCustomKernel));
let plan = compile(&graph, &registry, &dynamic_dimensions)?;
```

**Benefits:**
- ✅ Users add operations without modifying library code
- ✅ Shape inference co-located with kernel implementation
- ✅ One ONNX node can emit multiple GPU steps (no 1:1 assumption)
- ✅ Kernels can allocate scratch buffers for multi-pass algorithms
- ✅ No centralized match block — kernels define their own shape logic

### 2. Planner Compiles Shaders, Runtime Executes Them

**Shader compilation happens entirely at plan time:**

```rust
// Planner: WGSL + shader defs → naga::Module (via naga_oil)
let module = composer.make_naga_module(NagaModuleDescriptor {
    source: include_str!("../../shaders/elementwise/add.wgsl"),
    shader_defs: HashMap::from([("WORKGROUP_SIZE", ShaderDefValue::UInt(256))]),
    ..Default::default()
})?;

// Runtime: naga::Module → pipeline (via wgpu ShaderSource::Naga)
let shader_module = device.create_shader_module(ShaderModuleDescriptor {
    source: ShaderSource::Naga(Cow::Owned(module)),
    ..
});
```

**Boundary:** `naga_oil` is a planner-only dependency. The runtime never touches WGSL text or shader defs.

### 3. Clear Separation of Concerns

Each crate has a **single, well-defined responsibility**:

| Concern | Owner |
|---------|-------|
| ONNX parsing | onyxia-onnx |
| Kernel-based shape inference | onyxia-planner |
| Value propagation and constant folding | onyxia-planner |
| WGSL preprocessing (naga_oil) | onyxia-planner |
| Shader def resolution | onyxia-planner |
| Dynamic dimension resolution | onyxia-planner |
| Three-phase shape inference | onyxia-planner |
| Broadcasting utility | onyxia-planner |
| Pipeline/buffer materialization | onyxia-runtime |
| GPU dispatch & data transfer | onyxia-runtime |

### 4. Tokenization and Sampling are Out of Scope

**Why:** ONNX models don't include tokenization. Sampling strategies are application-specific.

**User's responsibility:**
```rust
// User handles tokenization
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let tokens = tokenizer.encode(prompt, false)?.get_ids();

// Onyxia processes tokens → logits
let outputs = executor.run(&[("input_ids", tokens)])?;

// User handles sampling
let next_token = sample_top_k(outputs["logits"], k=50);
```

### 5. WGSL via naga_oil → naga::Module

**Why WGSL:**
- ✅ Native to wgpu (no cross-compilation needed)
- ✅ High-level, readable shader code
- ✅ Cross-platform (DX12, Vulkan, Metal via wgpu HAL)

**How naga_oil is used (in planner only):**
- ✅ WGSL preprocessing with `Composer` → `naga::Module`
- ✅ Shader defs for specialization (`#ifdef`, `#if`, `#{VALUE}`)
- ✅ Shader composition (`#import`, `#define_import_path`)
- ✅ WGSL files are `include_str!`'d at compile time, preprocessed at plan time

**In WGSL** — shader defs for conditional compilation and value substitution:
```wgsl
#if QUANT_BITS == 4
    const PACK_FACTOR: u32 = 8u;
#endif

var<workgroup> tile: array<f32, #{TILE_SIZE} * #{BLOCK_SIZE}>;
```

### 6. Three-Phase Shape Resolution at Plan Time

**ONNX models use symbolic dimensions** (e.g., `[batch, sequence, 768]`). Onyxia resolves them through a three-phase process at plan time:

**Phase 1 — Dynamic Dimension Substitution:**
Replace all `Dynamic(Named(...))` dimensions with concrete `Static` values from the user-provided `dynamic_dimensions` map. After this phase, no `Named` dimensions remain.

**Phase 2 — Forward Shape and Value Inference:**
Run a single forward pass over the graph in topological order, calling each kernel's `infer_output_shapes()` and `try_fold()` to resolve `Unknown` shapes and propagate constant values. This enables data-dependent shape inference where operations like Reshape read their target shape from computed tensors like `Shape → Gather → Concat`.

**Phase 3 — Planning (Static Only):**
All shapes must be `Static` before planning. Kernels call `ctx.static_shape()` which only accepts `TensorShape::Static` — any remaining `Dynamic` or `Unknown` shapes are errors.

```rust
let dynamic_dimensions = HashMap::from([
    ("batch".to_string(), 1),
    ("sequence".to_string(), 8192),
]);

// Phase 1: Named dims → Static, Phase 2: forward inference + value propagation, Phase 3: plan
let plan = compile(&graph, &registry, &dynamic_dimensions)?;

// Runtime receives only static shapes — no dimension resolution needed.
let executor = runtime.load_model(plan).await?;
```

**Invariant:** Every tensor in `ExecutionPlan.tensors` has `TensorShape::Static`. The runtime asserts this and never performs dimension lookups.

### 7. Async for GPU Operations, Sync API Available

**Runtime uses async internally:**
```rust
impl Runtime {
    pub async fn new() -> Result<Self>;
    pub async fn load_model(&self, plan: ExecutionPlan) -> Result<PlanExecutor>;
}
```

**CLI uses `pollster` for sync entry points:**
```rust
#[pollster::main]
fn main() -> Result<()> {
    let runtime = Runtime::new().await?;
}
```

### 8. Quantization in Shaders (Future)

**ONNX model uses `MatMulNBits`:**
- Weights stored as 4-bit integers (packed), scales/zero-points as f16
- Shader dequantizes on-the-fly (4x memory savings, minimal compute overhead)

```wgsl
@compute
fn matmul_q4(/* ... */) {
    let packed = weights[weight_idx / 8];
    let shift = (weight_idx % 8) * 4;
    let quantized = (packed >> shift) & 0xF;
    let weight = (f32(quantized) - zero_point) * scale;
    acc += input * weight;
}
```

## Data Flow

```
User Code                   Onyxia Planner              Onyxia Runtime              GPU
──────────────────────────────────────────────────────────────────────────────────────────
                         compile(graph,
graph = load_model()     registry,
dynamic_dims = {...}  ──→  dynamic_dims) ──→ ExecutionPlan
                           │                    │
                           ├ resolve shapes     │
                           ├ schedule ops        │
                           ├ compile WGSL→naga   │
                           └ build plan          │
                                                 │
                                          load_model(plan) ──→ Create pipelines
                                                 │              Allocate buffers
                                                 ↓
inputs = [("a", tensor)] ──────────────→ executor.run(inputs)
                                                 │
                                           Upload to GPU ─────→ GPU buffers
                                           Execute steps ←───→ Run shaders
                                           Download outputs ←── GPU buffers
                                                 │
outputs ←────────────────────────────── return HashMap<String, Tensor>
```

## Development Phases

### Phase 1: Graph and Parser Foundation ✅ COMPLETED
- [x] Graph data structures: Graph, Node, TensorInfo, TensorShape, DataType
- [x] Parse ONNX ModelProto → Graph
- [x] Graph validation
- [x] DOT graph visualization (full, layers, summary)
- [x] Integration test with Gemma 3 270m model

### Phase 2: Planner and Kernel System ✅ COMPLETED
- [x] ExecutionPlan types: Step, PlannedOp, BufferRef, CompiledShader, TensorRegistry
- [x] Topological scheduling with petgraph
- [x] Three-phase shape inference: dynamic dim substitution → forward inference with value propagation → static-only planning
- [x] `OpKernel` trait with `InferenceContext` and optional `try_fold` for constant folding
- [x] `TensorValue` type for compile-time constant propagation
- [x] `KernelRegistry` for extensible operation mapping
- [x] `PlanContext` with shader compilation, `static_shape()`, scratch allocation
- [x] `InferenceContext` with input shapes and values for data-dependent shape inference
- [x] `compile()` entry point with integrated shape inference
- [x] Dynamic dimension resolution at plan time
- [x] Shader deduplication
- [x] 19 built-in kernels covering all Gemma 3 270m ops
- [x] Broadcasting utility for ONNX-compliant multidirectional broadcasting
- [x] Error handling and unit tests (101 tests)

### Phase 3: Runtime Execution ✅ COMPLETED
- [x] wgpu device setup with deferred creation
- [x] PlanExecutor: materializes ExecutionPlan into GPU pipelines and buffers
- [x] Pipeline creation from `naga::Module` via `ShaderSource::Naga`
- [x] Bind group layout derivation from naga module introspection
- [x] Immediate data (push constants) support
- [x] Buffer allocation for tensors and scratch buffers
- [x] Compute dispatch, buffer copy, buffer write
- [x] GPU → CPU download with staging buffers
- [x] `Tensor` type for CPU/GPU data interchange
- [x] End-to-end test: Graph → compile → load → run → verify output
- [ ] Validate numerical accuracy against ONNX Runtime

### Phase 4: Quantization Support
- [ ] Parse quantized weights from ONNX initializers
- [ ] Generate MatMulNBits shader with Q4 dequantization
- [ ] Handle scale/zero-point tensors
- [ ] Test on quantized models (Gemma 3 270m q4)
- [ ] Validate numerical accuracy vs fp32

### Phase 5: Attention and KV Cache
- [ ] Generate GroupQueryAttention shader (with GQA optimization)
- [ ] Generate RotaryEmbedding shader (RoPE)
- [ ] Runtime KV cache management (persistent GPU buffers)
- [ ] Prefill API (full sequence processing)
- [ ] Decode API (single token + cache update)
- [ ] Test autoregressive generation

### Phase 6: Optimizations
- [ ] Operator fusion (MatMul + Add + Gelu)
- [ ] Memory pooling and buffer reuse
- [ ] Flash Attention (tiled, memory-efficient)
- [ ] Workgroup size tuning
- [ ] Performance benchmarking (tokens/sec, latency)

### Phase 7: Polish and Advanced Features
- [ ] Better error messages with source location
- [ ] CLI improvements (benchmark, profile, run)
- [ ] Documentation and examples
- [ ] Multi-GPU support (future)

## Testing Strategy

### Unit Tests
- 101 passing across all crates
- Programmatic graph construction, no model files needed

### Integration Tests
- End-to-end: Graph → compile → GPU execute → compare outputs
- Gemma 3 270m model parsing and plan compilation
- GPU tests marked `#[ignore]` for CI without GPU

### Validation Tests (Future)
- Compare with ONNX Runtime on same inputs
- Numerical accuracy (atol=1e-4, rtol=1e-3)

### Performance Tests (Future)
- Tokens/second throughput
- Latency (prefill vs decode)
- Memory usage (peak, average)

## References

- **ONNX Spec:** https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **wgpu:** https://docs.rs/wgpu/latest/wgpu/
- **naga_oil:** https://github.com/bevyengine/naga_oil
- **Flash Attention:** https://arxiv.org/abs/2205.14135
- **Gemma Models:** https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX
