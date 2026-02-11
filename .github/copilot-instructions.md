# Copilot Instructions for Onyxia

## Project Overview

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust 2024 edition. It compiles ONNX operator graphs into WGSL compute shaders executed via `wgpu`.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  onyxia-onnx    │────▶│ onyxia-codegen  │────▶│ onyxia-runtime  │
│  (ONNX parser)  │     │ (WGSL compiler) │     │ (GPU executor)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        ▲
                                                        │
                                               ┌─────────────────┐
                                               │   onyxia-cli    │
                                               │ (CLI interface) │
                                               └─────────────────┘
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf models into internal IR |
| `onyxia-codegen` | Generate WGSL shaders and compile execution graphs |
| `onyxia-runtime` | Execute compiled graphs on GPU via `wgpu` |
| `onyxia-cli` | CLI for testing models, generating dot graphs, benchmarking |

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
cargo build                      # Build all crates
cargo build -p onyxia-cli        # Build specific crate
cargo nextest run                # Run tests (use nextest, not cargo test)
cargo nextest run -p onyxia-onnx # Test specific crate
cargo clippy --workspace         # Lint all crates
cargo fmt --all                  # Format all crates
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
- `onyxia-onnx` exports an IR that `onyxia-codegen` consumes
- `onyxia-codegen` produces executable graphs for `onyxia-runtime`
- Keep crate boundaries clean; avoid circular dependencies
