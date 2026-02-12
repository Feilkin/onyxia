# Onyxia Architecture Overview

## System Design

Onyxia is a **GPU compute shader runtime for ONNX models**, built in 3 main stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  onyxia-onnx    │────▶│ onyxia-planner  │────▶│ onyxia-runtime  │
│                 │     │                 │     │                 │
│  Parse ONNX     │     │  Compile to     │     │  Execute on GPU │
│  protobuf into  │     │  execution plan │     │  via wgpu (HAL) │
│  graph          │     │  with WGSL      │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        │
    ModelProto            ExecutionPlan           Model outputs
```

## Responsibility Boundaries

### onyxia-onnx

**Input:** `.onnx` file (protobuf)  
**Output:** `Graph` (stable API)

**Responsibilities:**
- Parse ONNX protobuf using `prost` → `ModelProto`
- Convert `ModelProto` → `Graph` (stable API independent of protobuf schema)
- Expose structured graph (nodes, tensors, operators, attributes)
- Validate graph integrity (missing tensors, type mismatches)
- Provide visualization (DOT graph generation)
- ✅ **Pure data parsing** — no computation or optimization

**Key Types:**
- `Graph`: Node list, tensor metadata, input/output mappings
- `Node`: Operator type, inputs, outputs, attributes
- `TensorInfo`: Name, shape, data type, kind (input/weight/intermediate/output)
- `TensorShape`: Static or dynamic dimensions
- `DataType`: F32, F16, I32, I64, U8, U32, Bool, Q4, Q8

**Does NOT:**
- Compile shaders
- Execute operations
- Optimize graphs

### onyxia-planner

**Input:** `Graph` from onyxia-onnx  
**Output:** `ExecutionPlan` (pre-compiled shaders + metadata)

**Responsibilities:**
- Schedule operations (topological sort with `petgraph`, memory-aware ordering)
- Resolve all dynamic dimensions at plan time
- Compile WGSL compute shaders to naga IR modules using `naga_oil`
- Apply shader definitions (tensor dimensions, workgroup sizes)
- Generate bind group layouts and buffer references
- Produce complete execution plan with all shapes resolved
- ✅ **Pure compilation** — no GPU interaction

**Does NOT:**
- Initialize GPU devices
- Allocate buffers
- Execute shaders
- Handle tokenization or sampling
- Parse ONNX protobuf (delegates to onyxia-onnx)

### onyxia-runtime

**Input:** `ExecutionPlan` from onyxia-planner + user tensors  
**Output:** Computed tensor outputs

**Responsibilities:**
- Initialize GPU via `wgpu` (hardware abstraction over DX12/Vulkan/Metal)
- Allocate and manage GPU buffers
- Create compute pipelines from pre-compiled naga modules
- Execute compute passes
- Transfer data between CPU and GPU
- Manage KV cache for autoregressive generation
- ✅ **Pure execution** — no shader compilation or graph modification

**Does NOT:**
- Compile shaders (receives pre-compiled naga modules)
- Tokenization/detokenization (user's job)
- Sampling logic (user applies to logits)
- Model optimization

### onyxia-cli

**Input:** Command-line arguments  
**Output:** CLI actions (DOT export, benchmarks, etc.)

**Responsibilities:**
- Test interface for model inspection
- DOT graph generation
- Performance benchmarking
- Debugging utilities

## Key Architectural Decisions

### 1. Clear Separation of Concerns

Each crate has a **single, well-defined responsibility**. This enables:
- ✅ Independent testing and development
- ✅ Reusability (e.g., use codegen without runtime)
- ✅ Parallel work on different components

### 2. Tokenization is OUT of Scope

**Why:** ONNX models don't include tokenization. It's a pre/post-processing step.

**User's responsibility:**
```rust
// User handles tokenization
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let tokens = tokenizer.encode(prompt, false)?.get_ids();

// Onyxia only processes tokens → logits
let outputs = executor.run(&[("input_ids", tokens)])?;

// User handles sampling
let next_token = sample_top_k(outputs["logits"], k=50);
```

**Benefits:**
- Flexibility: users choose their tokenizer library
- Simplicity: Onyxia focuses on GPU compute
- Compatibility: works with any tokenization scheme

### 3. Sampling is Out of Scope

**Why:** Sampling strategies (temperature, top-k, top-p, nucleus) are application-specific.

**User's responsibility:**
```rust
fn sample_top_k(logits: &[f32], k: usize, temp: f32) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
    
    // Sort and take top-k
    let mut indexed: Vec<_> = scaled.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    indexed.truncate(k);
    
    // Sample from top-k
    sample_categorical(&indexed)
}
```

**Benefits:**
- Users can implement custom sampling strategies
- No performance overhead from unused sampling methods
- Clear interface: model produces logits, user chooses action

### 4. KV Cache Management is Shared

**Codegen defines the interface:**
```rust
// Compiled model specifies KV cache inputs/outputs
inputs: ["input_ids", "past_key_values.0.key", "past_key_values.0.value", ...]
outputs: ["logits", "present.0.key", "present.0.value", ...]
```

**Runtime manages buffers:**
```rust
// Runtime allocates persistent cache buffers
pub struct KvCache {
    layers: Vec<LayerCache>,  // GPU buffers
}

impl ModelExecutor {
    pub fn decode(&mut self, token: u32, cache: &mut KvCache) -> Result<()>;
}
```

**User orchestrates generation:**
```rust
// User calls prefill + decode loop
let mut cache = executor.prefill(prompt_tokens)?;
for _ in 0..max_tokens {
    let output = executor.decode(next_token, &mut cache)?;
    next_token = sample(output.logits);
}
```

### 5. WGSL via naga_oil (Not rust-gpu or SPIR-V)

**Why WGSL:**
- ✅ Native to wgpu (no cross-compilation needed)
- ✅ High-level, readable shader code
- ✅ Cross-platform (DX12, Vulkan, Metal via wgpu HAL)
- ✅ Good balance of expressiveness and control

**How naga_oil is used:**
- ✅ Runtime shader compilation with `Composer`
- ✅ Enables shader composition (`#import`, `#define_import_path`)
- ✅ Supports shader defs for runtime specialization (`#ifdef`, `#if`, `#{VALUE}`)
- ✅ Direct integration - no wrapper layers
- ⚠️ **Write WGSL in separate files** - embedded at compile time, compiled at runtime

**Pattern:**
```rust
// Codegen: Embed WGSL file
let source = include_str!("../shaders/add.wgsl");

// Runtime: Compile with dynamic shader defs
let mut composer = Composer::default();
let module = composer.make_naga_module(NagaModuleDescriptor {
    source,
    shader_defs: runtime_defs,  // Based on actual tensor shapes
    ..Default::default()
})?;
```

**Why NOT rust-gpu:**
- ❌ Immature ecosystem (experimental)
- ❌ Compilation complexity (SPIR-V → WGSL translation)
- ❌ Harder to debug generated shaders

**Why NOT raw SPIR-V:**
- ❌ Low-level, verbose
- ❌ Requires cross-compilation to native backends
- ❌ Harder to inspect and validate

### 6. Shader Defs via naga_oil

**naga_oil** provides preprocessor-like shader definitions:

**In Rust** - define constants at shader compile time:
```rust
use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;

let mut shader_defs = HashMap::new();
shader_defs.insert("TILE_SIZE".to_string(), ShaderDefValue::UInt(16));
shader_defs.insert("BLOCK_SIZE".to_string(), ShaderDefValue::UInt(64));
shader_defs.insert("QUANT_BITS".to_string(), ShaderDefValue::UInt(4));
```

**In WGSL** - use defs for conditional compilation or value substitution:
```wgsl
// Conditional compilation
#if QUANT_BITS == 4
    const PACK_FACTOR: u32 = 8u;
#endif

// Value substitution
var<workgroup> tile: array<f32, #{TILE_SIZE} * #{BLOCK_SIZE}>;
```

**Benefits:**
- ✅ No runtime overhead (constants compiled into shader)
- ✅ Better than uniform buffers for static config
- ✅ Enables specialized shader variants per tensor shape
- ✅ Cleaner code (no magic numbers)
- ✅ Single `.wgsl` file can generate multiple specialized shaders

**Use cases:**
- Tensor dimensions (when known at compile time)
- Workgroup sizes
- Quantization parameters
- Kernel configuration (tile size, block size)

### 7. Async for GPU Operations, Sync API Available

**Runtime uses async internally:**
```rust
impl Runtime {
    pub async fn new() -> Result<Self>;  // GPU init is async
}
```

**But provides sync wrapper for CLI:**
```rust
#[pollster::main]
fn main() -> Result<()> {
    let runtime = Runtime::new().await?;  // pollster blocks
}
```

**Benefits:**
- GUI/server apps can use async naturally
- CLI apps can use `pollster::block_on` for simplicity
- Flexibility for different use cases

### 8. Dynamic Shape Handling

**ONNX models use symbolic dimensions:**
- Static dimensions: `[1, 512, 768]` - all concrete numbers
- Dynamic dimensions: `[batch, sequence, 768]` - symbolic names

**Onyxia's approach: Max dimensions at load time**

```rust
// User specifies maximum dimensions when loading model
let max_dims = HashMap::from([
    ("batch".to_string(), 1),
    ("sequence".to_string(), 8192),  // max context length
]);

let executor = runtime.load_model(compiled, max_dims)?;
// ↑ Pre-compiles all shaders and allocates buffers for max sizes
```

**At runtime, actual inputs can be smaller:**

```rust
// Provide actual tensor (batch=1, seq=3)
let input_ids = Tensor::from_vec(vec![1, 2, 3], &[1, 3]);
let outputs = executor.run(&[("input_ids", input_ids)])?;
// ↑ Runtime validates: 3 ≤ 8192 ✅
```

**Benefits:**
- ✅ No runtime shader compilation (predictable performance)
- ✅ Generic design (no LLM-specific assumptions)
- ✅ User controls memory/compile cost upfront
- ✅ Flexible (actual inputs can vary up to max)

**Dimension name handling:**
- ONNX uses arbitrary strings: `"batch"`, `"sequence"`, `"N"`, `"dynamic_axis_0"`, etc.
- Onyxia preserves these as `Dimension::Named(string)` without interpretation
- User provides concrete values via `HashMap<String, usize>` at load time

### 9. Quantization in Shaders

**ONNX model uses `MatMulNBits`:**
- Weights stored as 4-bit integers (packed)
- Scales and zero-points as f16

**Shader dequantizes on-the-fly:**
```wgsl
@compute
fn matmul_q4(/* ... */) {
    // Load packed 4-bit weights
    let packed = weights[weight_idx / 8];
    let shift = (weight_idx % 8) * 4;
    let quantized = (packed >> shift) & 0xF;
    
    // Dequantize: weight = (quantized - zp) * scale
    let weight = (f32(quantized) - zero_point) * scale;
    
    // Use dequantized weight in computation
    acc += input * weight;
}
```

**Benefits:**
- ✅ 4x memory savings (vs f32 weights)
- ✅ Faster memory bandwidth
- ✅ Compute overhead minimal (ALU is fast)

## Data Flow Example: LLM Generation

```
User Code (CLI)              Onyxia Runtime              GPU
────────────────────────────────────────────────────────────────
// Initialization (once)
dynamic_dims = {                
  "batch": 1,
  "sequence": 8192      ──→ runtime.load_model() ───→ Create pipelines
}                                   ↓                   Allocate buffers
                                    ↓                   (for these sizes)
                                                        
tokenizer.encode()          
      |
   [tokens]             ──→ executor.run(inputs) 
                                    ↓                   
                              Upload to GPU ────────→ GPU buffers
                                    ↓                   
                              Execute compute ←────→ Run shaders
                                    ↓                   
                              Download logits ←───── GPU buffers
                                    ↓                   
   logits ←─────────────────  return outputs
      |
sampling logic
      |
  next_token
      |
      └─→ executor.run([token]) ──→ (repeat)
                                    
Note: Prefill/decode distinction, KV cache management, 
and generation loops are CLI-layer concerns.
```

## Development Phases

### Phase 1: Graph and Planner Foundation ✅ COMPLETED
- [x] Define Graph data structures (Graph, Node, TensorInfo, TensorShape, DataType)
- [x] Parse ONNX ModelProto → Graph (moved to onyxia-onnx for stable API)
- [x] Implement topological sort for scheduling (petgraph-based in scheduler.rs)
- [x] ExecutionPlan structure (Step, PlannedOp, BufferRef, CompiledShader, TensorRegistry)
- [x] Error handling (OnnxError, PlannerError with proper separation)
- [x] Integration test with Gemma 3 270m model (896 tensors, 39 inputs, 37 outputs)
- [x] **Shape inference system** (18+ operations, ~51% coverage, handles broadcast/matmul/reduce/etc)
- [x] Shape inference integrated into parser (runs automatically after graph validation)

### Phase 2: Core Operator Shaders ⚠️ PARTIALLY COMPLETED
- [x] Create `shaders/` directory structure (elementwise, activation, normalization, matmul)
- [x] Write WGSL shaders: add.wgsl, mul.wgsl, gelu.wgsl, rmsnorm.wgsl, matmul_f32.wgsl
- [x] OpKernel trait for shader compilation with naga_oil (`naga_oil::compose::Composer`)
- [x] Kernel implementations: AddKernel, MulKernel, MatMulF32Kernel, GeluKernel, RmsNormKernel
- [x] Test kernel compilation (9 tests passing)
- [ ] **BLOCKED: Plan generation is incomplete** - `compile()` returns empty plan
- [ ] **CRITICAL GAP**: No connection between Graph nodes and kernel selection
- [ ] Missing: Code to translate ONNX nodes → PlannedOp instances with compiled shaders

### Phase 3: Runtime Execution — **BLOCKED on Phase 2**
- [x] Runtime initialization (wgpu 28 device setup)
- [x] **Deferred device creation** - Runtime stores instance+adapter, creates device in load_model()
- [x] **Buffer size calculation** - calculate_max_buffer_size() resolves dynamic dimensions
- [x] Buffer allocation and management infrastructure
- [x] Pipeline creation from pre-compiled naga modules
- [x] CPU ↔ GPU data transfer (upload/download with alignment)
- [x] Integration test scaffolding (tests pass but model loading fails on unknown shapes)
- [x] **Implemented dynamic_dimensions parameter in load_model()**
- [x] PlanExecutor with run() method (uploads inputs, executes planned steps, downloads outputs)
- [ ] **BLOCKED: Cannot execute operations - plan has no steps from planner**
- [ ] **BLOCKED: executor.execute_steps() has nothing to execute**
- [ ] Execute simple operations end-to-end (waiting for planner step generation)
- [ ] Validate against ONNX Runtime

**Current approach:**
- User specifies dynamic_dimensions HashMap at load_model() time (concrete values for symbolic dims)
- All shaders compiled and buffers allocated for these dimension sizes
- At run(), actual input shapes must match specified dimensions
- No runtime shader recompilation

**🚨 CRITICAL BLOCKER: Planner \u2192 Runtime Gap**

The pipeline is **90% complete but broken at the planner stage**:

```
✅ ONNX Parser works → Graph with nodes
❌ Planner returns empty steps list → ExecutionPlan has no steps
✅ Runtime loads model → Creates PlanExecutor
❌ PlanExecutor.run() has nothing to execute → Returns empty results
```

**What's missing in `crates/onyxia-planner/src/lib.rs:87`:**

```rust
// Current:
let steps = Vec::new(); // TODO: Generate planned operations

// Needs to become:
let steps = generate_steps(graph, &ordered_nodes, &registry)?;

fn generate_steps(
    graph: &Graph, 
    ordered_nodes: &[usize],
    registry: &TensorRegistry,
) -> Result<Vec<Step>> {
    let mut steps = Vec::new();
    
    for &node_id in ordered_nodes {
        let node = &graph.nodes[node_id];
        
        // Select kernel for this operation
        let kernel: Box<dyn OpKernel> = match node.op_type.as_str() {
            "Add" => Box::new(AddKernel),
            "Mul" => Box::new(MulKernel),
            "Gelu" => Box::new(GeluKernel),
            "RmsNorm" => Box::new(RmsNormKernel),
            "MatMul" => Box::new(MatMulF32Kernel),
            op => return Err(PlannerError::UnsupportedOperation(op.to_string())),
        };
        
        // Compile kernel to naga module with shader defs
        let compiled_shader = kernel.compile(node, registry)?;
        
        // Create planned operation
        let op = PlannedOp {
            kernel_name: node.op_type.clone(),
            inputs: map_tensor_refs(&node.inputs, registry)?,
            outputs: map_tensor_refs(&node.outputs, registry)?,
            workgroups: calculate_workgroups(node, registry)?,
        };
        
        steps.push(Step {
            op,
            shader: compiled_shader,
        });
    }
    
    Ok(steps)
}
```

**To unblock Phase 3:**
1. Implement `generate_steps()` in planner
2. Add kernel selection logic for all implemented kernels (Add, Mul, Gelu, RMSNorm, MatMul)
3. Map ONNX node.op_type strings → OpKernel implementations
4. Extract operation parameters from node attributes
5. Calculate workgroup sizes based on tensor shapes
6. Compile kernels with runtime-specific shader defs (tensor dimensions, etc.)

**Estimated effort:** 2-3 days of focused work to unblock end-to-end execution

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
- [ ] Dynamic shape support (batch/sequence length variation)
- [ ] Better error messages with source location
- [ ] CLI improvements (benchmark, profile, inspect)
- [ ] Documentation and examples
- [ ] Multi-GPU support (future)

## Testing Strategy

### Unit Tests
- Per-crate tests for isolated functionality
- Mock/stub external dependencies

### Integration Tests
- End-to-end: ONNX → compile → execute → compare outputs
- Known models with reference outputs

### Validation Tests
- Compare with ONNX Runtime on same inputs
- Numerical accuracy (atol=1e-4, rtol=1e-3)

### Performance Tests
- Tokens/second throughput
- Latency (prefill vs decode)
- Memory usage (peak, average)

## References

- **ONNX Spec:** https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **wgpu:** https://docs.rs/wgpu/latest/wgpu/ (Hardware abstraction layer)
- **naga_oil:** https://github.com/bevyengine/naga_oil (Shader preprocessing and composition)
- **Flash Attention:** https://arxiv.org/abs/2205.14135
- **Gemma Models:** https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX
