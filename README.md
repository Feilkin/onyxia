# Onyxia

**GPU compute shader runtime for ONNX models.** Uses a dispatch-based execution model where operators compile their shaders at compile time and compute shapes at runtime from actual input tensors.

## Architecture

```
ONNX Model (.onnx)
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│ onyxia-onnx │────▶│ onyxia-core  │◀────│  onyxia-ops   │     │onyxia-runtime │
│ Parse ONNX  │     │ IR, dispatch │     │ 3 core        │     │ Register-based│
│ protobuf    │     │ traits       │     │ operators     │     │ GPU execution │
└─────────────┘     └──────┬───────┘     └───────────────┘     └───────────────┘
                           │
                    ┌──────┴───────┐
                    │ onyxia-      │
                    │ compiler     │──────▶ CompiledModel ──────▶ GPU Execution
                    │ Build dispatch│
                    └──────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf into a structured `Graph` API |
| `onyxia-core` | IR graph, operator/dispatch traits, compiled model types, operator registry |
| `onyxia-operators` | 3 core ONNX operator implementations (Add, Mul, Reshape) |
| `onyxia-compiler` | Simplified pipeline: initialize constants → build dispatch model |
| `onyxia-runtime` | Register-based GPU execution engine via `wgpu` |
| `onyxia-cli` | CLI for model inspection, validation, and DOT visualization |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Features

- **ONNX parsing** — stable `Graph` API independent of protobuf schema
- **Dispatch-based execution** — operators compute shapes at runtime from actual input tensors
- **3 core operators** — Add, Mul, Reshape (minimal proof-of-concept set)
- **Extensible operator system** — add custom operators via the `Operator` trait
- **Shader compilation** — WGSL → `naga::Module` via `naga_oil` at compile time
- **Register-based GPU execution** — efficient tensor routing via indexed register file
- **CLI tools** — model inspection, node tracing, DOT visualization, validation

### Built-in Operators (3)

| Category | Operators |
|----------|-----------|
| Binary elementwise | Add, Mul |
| Shape manipulation | Reshape |

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

    // 2. Compile to dispatch model
    let registry = core_operator_registry();
    let mut pipeline = onyxia_compiler::CompilerPipeline::new();
    let compiled = pipeline.compile(&graph, &registry)?;

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
use onyxia_core::{Operator, CompileCtx, OpDispatch, DispatchCtx, RuntimeTensor, Result};
use std::collections::HashMap;

struct MyCustomOperator;

impl Operator for MyCustomOperator {
    fn name(&self) -> &str { "MyCustomOp" }

    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>> {
        // Compile WGSL shader and create dispatch object
        let module = ctx.compile_shader(
            "my_custom_op",
            include_str!("shader.wgsl"),
            &HashMap::new(),
        )?;
        
        Ok(Box::new(MyCustomDispatch { module }))
    }
}

struct MyCustomDispatch {
    module: naga::Module,
}

impl OpDispatch for MyCustomDispatch {
    fn dispatch(
        &self,
        inputs: Vec<RuntimeTensor>,
        ctx: &mut DispatchCtx,
    ) -> Result<Vec<RuntimeTensor>> {
        // Compute output shape from input shapes
        let output_shape = inputs[0].shape.clone();
        
        // Allocate output buffer and dispatch GPU work
        // ... implementation ...
        
        todo!()
    }
}

// Register alongside built-in operators
let mut registry = onyxia_operators::core_operator_registry();
registry.register("MyCustomOp", MyCustomOperator);
let mut pipeline = onyxia_compiler::CompilerPipeline::new();
let compiled = pipeline.compile(&graph, &registry)?;
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

# Validate model compilation
cargo run --bin onyxia -- validate model.onnx

# Generate DOT visualization
cargo run --bin onyxia -- dot model.onnx -o model.dot -s summary
dot -Tpng model.dot -o model.png   # requires Graphviz
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
cargo nextest run                                   # Non-GPU tests
cargo nextest run --run-ignored=all --no-fail-fast  # All tests including GPU
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

Logo color palette: https://lospec.com/palette-list/technogarten