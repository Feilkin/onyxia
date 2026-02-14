# Onyxia

**GPU compute shader runtime for ONNX models.** Compiles ONNX operator graphs into GPU compute shaders and executes them via `wgpu`.

## Architecture

```
ONNX Model → onyxia-onnx → onyxia-planner → onyxia-runtime → GPU Execution
  (.onnx)    (parse → Graph)  (naga::Module    (wgpu pipelines   (results)
                               shaders)         + dispatch)
```

- **onyxia-onnx**: Parse ONNX protobuf into a stable Graph API
- **onyxia-planner**: Kernel-based shape inference and compilation into execution plans with pre-compiled shaders
- **onyxia-runtime**: Execute plans on GPU hardware via wgpu
- **onyxia-cli**: Command-line tools for testing and debugging

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Current Status

**The end-to-end pipeline is working** — from ONNX parsing through GPU execution:

```
✅ ONNX Model → Parser → Graph
✅ Graph → Planner → ExecutionPlan
✅ ExecutionPlan → Runtime → GPU execution
✅ GPU outputs → CPU tensors
```

### What Works

- ✅ **ONNX parsing** with stable Graph API
- ✅ **Kernel-based shape inference** — three-phase: dynamic dim substitution → forward inference with value propagation → static-only planning
- ✅ **DOT graph visualization** (full, layers, summary views)
- ✅ **Extensible kernel system** — users add operations via `OpKernel` trait
- ✅ **Shader compilation** — WGSL → `naga::Module` via naga_oil at plan time
- ✅ **Dynamic dimension resolution** at plan time
- ✅ **GPU execution** with buffer management and compute dispatch
- ✅ **End-to-end pipeline** verified
- ✅ **101 tests passing**, 22 GPU tests skipped in CI

### Built-in Kernels

| Kernel | ONNX Op | Category |
|--------|---------|----------|
| `AddKernel` | Add | Elementwise |
| `SubKernel` | Sub | Elementwise |
| `MulKernel` | Mul | Elementwise |
| `GeluKernel` | Gelu | Activation |
| `RmsNormKernel` | SimplifiedLayerNormalization | Normalization |
| `MatMulF32Kernel` | MatMul | Matrix multiplication |
| `MatMulNBitsKernel` | MatMulNBits | Quantized matmul |
| `CastKernel` | Cast | Type conversion |
| `ConstantKernel` | Constant | Metadata |
| `ShapeKernel` | Shape | Metadata |
| `ReshapeKernel` | Reshape | Shape manipulation |
| `UnsqueezeKernel` | Unsqueeze | Shape manipulation |
| `TransposeKernel` | Transpose | Shape manipulation |
| `ConcatKernel` | Concat | Shape manipulation |
| `GatherKernel` | Gather | Indexing |
| `ReduceMeanKernel` | ReduceMean | Reduction |
| `ReduceSumKernel` | ReduceSum | Reduction |
| `RotaryEmbeddingKernel` | RotaryEmbedding | Attention |
| `GroupQueryAttentionKernel` | GroupQueryAttention | Attention |

### What's Next

- 🔜 More kernels for broader ONNX operation coverage
- 🔜 Quantized model support — 4-bit, 8-bit via `MatMulNBits`
- 🔜 KV cache management for efficient LLM generation
- 🔜 Performance optimizations (fusion, tiling, memory pooling)
- 🔜 Numerical validation against ONNX Runtime

## Usage

### Adding Custom Operations

```rust
use onyxia_planner::{OpKernel, InferenceContext, TensorValue, PlanContext, Step, KernelRegistry, compile};

struct MyCustomKernel;

impl OpKernel for MyCustomKernel {
    fn name(&self) -> &str { "MyCustomOp" }
    
    fn infer_output_shapes(
        &self,
        ctx: &InferenceContext<'_>,
    ) -> onyxia_planner::Result<Vec<TensorShape>> {
        // Define shape inference logic for this operation
        Ok(vec![ctx.input_shapes[0].clone()])
    }
    
    fn try_fold(
        &self,
        ctx: &InferenceContext<'_>,
    ) -> onyxia_planner::Result<Vec<Option<TensorValue>>> {
        // Optional: implement constant folding for compile-time evaluation
        Ok(vec![None])
    }
    
    fn plan(&self, ctx: &mut PlanContext<'_>) -> onyxia_planner::Result<Vec<Step>> {
        // Compile shader, set up bindings, return steps
        todo!()
    }
}

// Register and compile
let mut registry = KernelRegistry::with_defaults();
registry.register("MyCustomOp", Box::new(MyCustomKernel));
let plan = compile(&graph, &registry, &dynamic_dimensions)?;
```

### Running a Model

```rust
use onyxia_onnx::load_model;
use onyxia_planner::{compile, KernelRegistry};
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // Parse ONNX model
    let graph = load_model("model.onnx")?;

    // Compile to execution plan
    let registry = KernelRegistry::with_defaults();
    let dynamic_dimensions = HashMap::from([
        ("batch".to_string(), 1),
        ("sequence".to_string(), 512),
    ]);
    let plan = compile(&graph, &registry, &dynamic_dimensions)?;

    // Execute on GPU
    let runtime = Runtime::new().await?;
    let mut executor = runtime.load_model(plan).await?;

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
    let outputs = executor.run(&[("input", input)])?;

    println!("Output: {:?}", outputs["output"].to_vec::<f32>()?);
    Ok(())
}
```

### Inspecting Models (CLI)

```bash
# Parse and analyze model structure
cargo run --bin onyxia -- inspect models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx

# Generate DOT visualization
cargo run --bin onyxia -- dot models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx \
  -o model.dot -s summary

# Convert to PNG (requires Graphviz)
dot -Tpng model.dot -o model.png
```

## Prerequisites

### Protocol Buffers Compiler (`protoc`)

Required for building the ONNX parser. Install via your package manager:

- **Windows (winget)**: `winget install protobuf`
- **Windows (Chocolatey)**: `choco install protoc`
- **macOS**: `brew install protobuf`
- **Linux (apt)**: `apt install protobuf-compiler`
- **Linux (dnf)**: `dnf install protobuf-compiler`

See [protobuf installation guide](https://protobuf.dev/installation/#package-manager) for more options.

## Building

```bash
cargo build
```

## Testing

We use [nextest](https://nexte.st/) as our test runner:

```bash
cargo nextest run
```

GPU-dependent tests are marked `#[ignore]` and can be run with:

```bash
cargo nextest run --run-ignored all
```

## Crates

| Crate | Description |
|-------|-------------|
| `onyxia-onnx` | ONNX protobuf parser, Graph API |
| `onyxia-planner` | Kernel-based shape inference and execution plan compiler |
| `onyxia-runtime` | GPU executor via wgpu |
| `onyxia-cli` | CLI tools for model inspection and DOT export |

## Example Models

The `models/` directory contains sample ONNX models for testing:

- **Gemma 3 270m** (quantized LLM): `models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx`
  - 18 transformer layers, 4 attention heads, vocab size 262K
  - Uses `MatMulNBits` (4-bit quantized weights), `GroupQueryAttention`, `RotaryEmbedding`

## Development Roadmap

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full development plan:

- ✅ Phase 1: Graph and Parser Foundation
- ✅ Phase 2: Planner and Kernel System
- ✅ Phase 3: Runtime Execution
- 🔜 Phase 4: Quantization Support
- 🔜 Phase 5: Attention and KV Cache
- 🔜 Phase 6: Optimizations
- 🔜 Phase 7: Polish and Advanced Features

## License

MIT OR Apache-2.0
