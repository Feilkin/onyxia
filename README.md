# Onyxia

**GPU compute shader runtime for ONNX models.** Compiles ONNX operator graphs into WGSL compute shaders and executes them on the GPU via `wgpu`.

## Architecture

```
ONNX Model (.onnx)
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│ onyxia-onnx │────▶│ onyxia-core  │◀────│  onyxia-ops   │     │onyxia-runtime │
│ Parse ONNX  │     │ IR, traits,  │     │ 40 operator   │     │ GPU execution │
│ protobuf    │     │ plan types   │     │ impls + WGSL  │     │ via wgpu      │
└─────────────┘     └──────┬───────┘     └───────────────┘     └───────────────┘
                           │
                    ┌──────┴───────┐
                    │ onyxia-      │
                    │ compiler     │──────▶ CompiledModel ──────▶ GPU Execution
                    │ Pass pipeline│
                    └──────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf into a structured `Graph` API |
| `onyxia-core` | IR graph, operator/pass traits, plan types, operator registry |
| `onyxia-operators` | 40 built-in ONNX operator implementations with WGSL shaders |
| `onyxia-compiler` | Pass-based compilation pipeline (resolution → folding → inference → planning) |
| `onyxia-runtime` | GPU execution engine via `wgpu` |
| `onyxia-cli` | CLI for model inspection, validation, DOT export, and text generation |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Features

- **ONNX parsing** — stable `Graph` API independent of protobuf schema
- **Pass-based compiler** — staged pipeline: symbolic resolution → constant folding → shape inference → planning
- **40 built-in operators** — elementwise, matmul, attention, normalization, shape manipulation, and more
- **Extensible operator system** — add custom operators via the `Operator` trait
- **Constant folding** — evaluates constant subgraphs at compile time (e.g., `Shape → Gather → Concat → Reshape` chains)
- **Symbolic dimension support** — arithmetic expressions in dimension names (e.g., `seq_len * num_heads`)
- **Shader compilation** — WGSL → `naga::Module` via `naga_oil` at compile time
- **GPU execution** — buffer management, compute dispatch, and CPU↔GPU data transfer
- **CLI tools** — model inspection, node tracing, DOT visualization, validation, text generation
- **148 tests passing**

### Built-in Operators (40)

| Category | Operators |
|----------|-----------|
| Binary elementwise | Add, Sub, Mul, Div, Pow, Max |
| Unary elementwise | Cos, Sin, Sqrt, Neg, Tanh |
| Comparison | Equal, Greater |
| Activation | Gelu, Softmax |
| Normalization | SimplifiedLayerNormalization (RmsNorm) |
| Matrix multiplication | MatMul, MatMulNBits (4-bit quantized) |
| Metadata | Constant, ConstantOfShape, Shape |
| Shape manipulation | Reshape, Unsqueeze, Transpose, Concat, Expand |
| Indexing | Gather, Slice, ScatterND, Range, Trilu |
| Reduction | ReduceSum, ReduceMean |
| Type conversion | Cast |
| Conditional | Where |
| Attention | RotaryEmbedding, GemmaRotaryEmbedding, MicrosoftRotaryEmbedding, GroupQueryAttention |

## Usage

### Running a Model

```rust
use onyxia_onnx::load_model;
use onyxia_compiler::compile;
use onyxia_operators::core_operator_registry;
use onyxia_runtime::{Runtime, Tensor};
use std::collections::HashMap;

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // 1. Parse ONNX model
    let model = load_model("model.onnx")?;
    let graph = onyxia_onnx::parse_model(&model)?;

    // 2. Compile to execution plan
    let registry = core_operator_registry();
    let dynamic_dimensions = HashMap::from([
        ("batch_size".to_string(), 1),
        ("sequence_length".to_string(), 512),
    ]);
    let compiled = compile(&graph, &registry, &dynamic_dimensions)?;

    // 3. Execute on GPU
    let runtime = Runtime::new().await?;
    let mut executor = runtime.load_model(compiled).await?;

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
    let outputs = executor.run(&[("input", input)])?;

    println!("Output: {:?}", outputs["output"].to_vec::<f32>()?);
    Ok(())
}
```

### Adding Custom Operators

```rust
use onyxia_core::{Operator, InferenceCtx, FoldCtx, PlanCtx, Step, TensorShape, TensorValue};

struct MyCustomOperator;

impl Operator for MyCustomOperator {
    fn name(&self) -> &str { "MyCustomOp" }

    fn infer_output_shapes(&self, ctx: &InferenceCtx) -> onyxia_core::Result<Vec<TensorShape>> {
        // Output has same shape as first input
        Ok(vec![ctx.input_shape(0)?])
    }

    fn try_fold(&self, _ctx: &FoldCtx) -> onyxia_core::Result<Vec<Option<TensorValue>>> {
        // Optional: evaluate at compile time if inputs are constant
        Ok(vec![None])
    }

    fn plan(&self, ctx: &mut PlanCtx) -> onyxia_core::Result<Vec<Step>> {
        // Compile WGSL shader, set up buffer bindings, emit dispatch
        todo!()
    }
}

// Register alongside built-in operators
let mut registry = onyxia_operators::core_operator_registry();
registry.register("MyCustomOp", MyCustomOperator);
let compiled = onyxia_compiler::compile(&graph, &registry, &dynamic_dimensions)?;
```

### CLI

```bash
# Inspect model structure
cargo run --bin onyxia -- inspect model.onnx

# Inspect specific nodes
cargo run --bin onyxia -- inspect-node model.onnx --name "/layer0/attention/query"

# List nodes filtered by op type
cargo run --bin onyxia -- list-nodes model.onnx --op-type MatMul --show-shapes

# Trace data flow around a node
cargo run --bin onyxia -- trace-node model.onnx --name "/layer0/ffn/add" --depth 2

# Validate model compilation (without GPU)
cargo run --bin onyxia -- validate model.onnx -d batch_size=1 -d sequence_length=512

# Generate DOT visualization
cargo run --bin onyxia -- dot model.onnx -o model.dot -s summary
dot -Tpng model.dot -o model.png   # requires Graphviz

# Run text generation (requires tokenizer + GPU)
cargo run --bin onyxia -- run-model model.onnx \
  --tokenizer ./tokenizer_dir --prompt "Hello, world" --max-tokens 50
```

## Prerequisites

### Protocol Buffers Compiler (`protoc`)

Required for building the ONNX parser (`onyxia-onnx` uses `prost-build`). Install via your package manager:

- **macOS**: `brew install protobuf`
- **Linux (apt)**: `apt install protobuf-compiler`
- **Linux (dnf)**: `dnf install protobuf-compiler`
- **Windows (winget)**: `winget install protobuf`
- **Windows (Chocolatey)**: `choco install protoc`

See [protobuf installation guide](https://protobuf.dev/installation/#package-manager) for more options.

## Building

```bash
cargo build
```

## Testing

Tests are run with [nextest](https://nexte.st/):

```bash
cargo nextest run                                   # Non-GPU tests (148 passing)
cargo nextest run --run-ignored=all --no-fail-fast   # All tests including GPU
```

GPU-dependent tests are marked `#[ignore]` and require a GPU.

## Example Models

The `models/` directory contains sample ONNX models for testing:

- **Gemma 3 270m** (quantized LLM): `models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx`
  — 18 transformer layers, 4 attention heads, vocab size 262K.
  Uses `MatMulNBits`, `GroupQueryAttention`, `RotaryEmbedding`.

- **Gemma 3 1B** (larger model): `models/gemma-3-1b-it-ONNX/onnx/`

## License

MIT OR Apache-2.0
