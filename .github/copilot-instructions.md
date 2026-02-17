# Copilot Instructions for Onyxia

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust 2024 edition. It uses a dispatch-based execution model where operators compile their shaders at compile time and compute shapes at runtime from actual input tensors.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐      ┌──────────────────┐
│  onyxia-onnx    │────▶│ onyxia-compiler  │────▶│ onyxia-runtime   │
│  (ONNX parser)  │     │ (dispatch model) │      │ (GPU executor)   │
└─────────────────┘     └──────────────────┘      └──────────────────┘
                                ▲                           ▲
                                │                           │
                    ┌───────────┴───────────┐      ┌───────┴──────────┐
                    │  onyxia-operators     │      │   onyxia-cli     │
                    │  (3 core operators)   │      │  (CLI interface) │
                    └───────────────────────┘      └──────────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf models into internal IR |
| `onyxia-core` | IR graph, operator/dispatch traits, compiled model types, operator registry |
| `onyxia-operators` | 3 core operators (Add, Mul, Reshape) with WGSL shaders |
| `onyxia-compiler` | Build dispatch models with pre-compiled WGSL shaders |
| `onyxia-runtime` | Execute dispatch models on GPU via `wgpu` with register-based routing |
| `onyxia-cli` | CLI for model inspection, DOT graphs, validation |

Read [ARCHITECTURE.md](../ARCHITECTURE.md) for more details.

### Key Technology Choices

- **`wgpu`** — GPU hardware abstraction layer (cross-platform)
- **`naga_oil`** — WGSL shader preprocessing and composition
- **WGSL** — Shader language (middle ground between SPIR-V and rust-gpu)
- **`prost`/`prost-build`** — Protobuf parsing for ONNX models (generate Rust types from .proto)
- **`clap`** — CLI argument parsing
- **`pollster`** — Blocking on async for CLI entry points
- **Async** — Use async for GPU operations (wgpu APIs are async)

## Build & Development

**Prerequisite**: `protoc` must be installed. See [README.md](../README.md) for installation instructions.

```bash
cargo build                                         # Build all crates
cargo build -p onyxia-cli                           # Build specific crate
cargo nextest run                                   # Run tests (use nextest, not cargo test)
cargo nextest run -p onyxia-onnx                    # Test specific crate
cargo nextest run --run-ignored=all --no-fail-fast  # Run all tests (requirest a GPU).
cargo clippy --workspace                            # Lint all crates
cargo fmt --all                                     # Format all crates
```

### ONNX Protobuf Generation

The ONNX proto schema is vendored in `crates/onyxia-onnx/proto/onnx.proto`. The `build.rs` script uses `prost-build` to generate Rust types at compile time. Generated code is written to `OUT_DIR` and included via:

```rust
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
```

## Code Conventions

- **Edition**: Rust 2024 — use modern idioms (e.g., `gen` blocks if stabilized)
- **Error handling**: `Result<T, E>` with `?` operator; define crate-specific error types
- **Async**: Use `async`/`await` for GPU operations; prefer `pollster::block_on` for CLI entry points
- **Formatting**: `rustfmt` defaults; run `cargo fmt --all` before committing

## Guidelines for AI Agents

### Dependency Management
- **Always use `cargo add <crate>` to add dependencies** — never edit `Cargo.toml` by hand
- **Verify all new dependencies with the user before adding** — do not add crates autonomously
- Prefer workspace dependencies in root `Cargo.toml` for shared crates

### Code Patterns
- Write idiomatic Rust: iterators, pattern matching, ownership
- Include doc comments (`///`) for all public APIs
- Add unit tests in `#[cfg(test)]` modules; use `nextest` to run them
- For GPU code, write WGSL in separate `.wgsl` files and use `naga_oil` for runtime compilation

### Cross-Crate Communication
- `onyxia-onnx` exports a `Graph` that `onyxia-compiler` consumes
- `onyxia-core` defines the `Operator` trait (with `create_dispatch()`) and `OpDispatch` trait
- `onyxia-operators` implements operators and exports `core_operator_registry()`
- `onyxia-compiler` produces `CompiledModel` (dispatch entries + register routing) for `onyxia-runtime`
- `onyxia-runtime` executes dispatch entries using a register-based execution model
- Keep crate boundaries clean; avoid circular dependencies

### Key Concepts

**Dispatch-based execution:**
- Each operator implements `Operator::create_dispatch()` to produce an `OpDispatch` object
- At runtime, `OpDispatch::dispatch(inputs, ctx) -> outputs` computes shapes from actual inputs and executes GPU work
- No compile-time shape inference or constant folding — shapes determined at runtime

**Register machine:**
- Runtime maintains a vector of `Option<RuntimeTensor>` (the register file)
- Each tensor in the IR graph maps to a register index
- Operations read inputs from registers and write outputs to registers

**Operator trait:**
```rust
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn create_dispatch(&self, ctx: &mut CompileCtx) -> Result<Box<dyn OpDispatch>>;
}
```

**OpDispatch trait:**
```rust
pub trait OpDispatch: Send + Sync {
    fn dispatch(&self, inputs: Vec<RuntimeTensor>, ctx: &mut DispatchCtx) -> Result<Vec<RuntimeTensor>>;
}
```
