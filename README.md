# Onyxia

**GPU compute shader runtime for ONNX models.** Compiles ONNX operator graphs into WGSL compute shaders and executes them on GPU via `wgpu`.

## Architecture

Onyxia is built in three stages:

```
ONNX Model → onyxia-onnx → onyxia-codegen → onyxia-runtime → GPU Execution
  (.onnx)    (parse → Graph)  (WGSL shaders)   (wgpu exec)     (results)
```

- **onyxia-onnx**: Parse ONNX protobuf and provide stable Graph API
- **onyxia-codegen**: Generate WGSL compute shaders and execution plans
- **onyxia-runtime**: Execute compiled models on GPU hardware
- **onyxia-cli**: Command-line tools for testing and debugging

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Features

- ✅ **ONNX parsing** with stable Graph API (onyxia-onnx)
- ✅ **Shape inference** for 18+ ONNX operations (~51% coverage on real models)
- ✅ **DOT graph visualization** (full, layers, summary views)
- ✅ **Codegen foundation** (scheduling, compiled model structure, shader files)
- ✅ **Runtime infrastructure** (device init, buffer management, deferred device creation)
- ✅ **WGSL shaders** written (Add, Mul, Gelu, RMSNorm, MatMul-f32)
- 🚨 **CRITICAL GAP**: Codegen doesn't generate operations - pipeline broken at codegen stage
- ⏸️  **Blocked**: GPU execution waiting for codegen operation generation
- 🔜 Quantized model support (4-bit, 8-bit via `MatMulNBits`)
- 🔜 KV cache management for efficient LLM generation
- 🔜 Performance optimizations (fusion, tiling, memory pooling)

## Current Status

**The pipeline is ~90% complete but has a critical gap at the codegen stage:**

```
✅ ONNX Model → Parser → Graph (works perfectly)
✅ Graph → Scheduler → Ordered nodes (works)
❌ Graph → Codegen → Operations (returns empty list - BLOCKER)
✅ Operations → Runtime → GPU execution (infrastructure ready, nothing to execute)
```

See [ARCHITECTURE.md](ARCHITECTURE.md#-critical-blocker-codegen--runtime-gap) for details on the blocker and what's needed to unblock end-to-end execution.

## What You Can Do Today

**Working Features:**
- ✅ **Inspect ONNX models**: Parse any ONNX file and examine structure
- ✅ **Visualize graphs**: Generate DOT graphs (full/layers/summary views)
- ✅ **Check shapes**: Run shape inference to see tensor shapes (~51% coverage)
- ✅ **Test parsing**: Validate ONNX model compatibility

**Not Yet Working:**
- ❌ **Run models**: Cannot execute on GPU yet (operations list empty from codegen)
- ❌ **Inference**: No forward pass implementation
- ❌ **Benchmarking**: Can't measure performance

**Example - Inspect a model:**
```bash
# Parse and analyze model structure
cargo run --bin onyxia -- inspect models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx

# Generate visualization
cargo run --bin onyxia -- dot models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx \
  -o model.dot -s summary
dot -Tpng model.dot -o model.png
```

## For Contributors

**High-Priority Work Needed:**

1. **Unblock codegen** (2-3 days) - Implement operation generation in `crates/onyxia-codegen/src/lib.rs`
   - Map ONNX nodes → Operation instances
   - Connect op_type strings to ShaderHandle  
   - Extract parameters from attributes
   
2. **Test end-to-end** (1 day) - Once operations generate, validate pipeline works
3. **Add more shaders** (ongoing) - Cover more ONNX operations
4. **Shape inference improvements** (1-2 days) - Implement constant evaluation for remaining 49%

See [ARCHITECTURE.md Development Phases](ARCHITECTURE.md#development-phases) for detailed roadmap.

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

## CLI Usage

### Generate DOT Graphs

Visualize ONNX model structure:

```bash
# Full graph (all nodes and edges)
cargo run -p onyxia-cli -- dot model.onnx -o model.dot

# Layer-grouped view (faster for large models)
cargo run -p onyxia-cli -- dot model.onnx -o model.dot -s layers

# High-level summary
cargo run -p onyxia-cli -- dot model.onnx -o model.dot -s summary

# Convert to PNG (requires Graphviz)
dot -Tpng model.dot -o model.png
```

## Crates

| Crate | Description | Documentation |
|-------|-------------|---------------|
| `onyxia-onnx` | ONNX protobuf parser | [crates/onyxia-onnx](crates/onyxia-onnx) |
| `onyxia-codegen` | WGSL shader compiler | [crates/onyxia-codegen/DESIGN.md](crates/onyxia-codegen/DESIGN.md) |
| `onyxia-runtime` | GPU executor via wgpu | [crates/onyxia-runtime/DESIGN.md](crates/onyxia-runtime/DESIGN.md) |
| `onyxia-cli` | CLI testing tools | [crates/onyxia-cli](crates/onyxia-cli) |

## Example Models

The `models/` directory contains sample ONNX models for testing:

- **Gemma 3 270m** (quantized LLM): `models/gemma-3-270m-it-ONNX/onnx/model_q4.onnx`
  - 18 transformer layers, 4 attention heads, vocab size 262K
  - Uses `MatMulNBits` (4-bit quantized weights), `GroupQueryAttention`, `RotaryEmbedding`

## Development Roadmap

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full development plan. Current status:

- ✅ Phase 1: Graph and Codegen Foundation (COMPLETED)
  - Graph data structures, ONNX parser, scheduler, compiled model
- 🚧 Phase 2: Core Operator Shaders (CURRENT)
  - WGSL shader generation for elementwise ops, LayerNorm, MatMul
- 🔜 Phase 3: Runtime Execution
- 🔜 Phase 4: Quantization Support
- 🔜 Phase 5: Attention and KV Cache
- 🔜 Phase 6: Optimizations
- 🔜 Phase 7: Polish and Advanced Features

## License

MIT OR Apache-2.0
