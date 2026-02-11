# Onyxia

GPU compute shader runtime for ONNX models. Compiles ONNX operator graphs into WGSL compute shaders executed via `wgpu`.

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

```bash
cargo nextest run
```

## Crates

| Crate | Description |
|-------|-------------|
| `onyxia-onnx` | ONNX model parser |
| `onyxia-codegen` | WGSL shader compiler |
| `onyxia-runtime` | GPU executor |
| `onyxia-cli` | CLI tool |

## License

MIT OR Apache-2.0
