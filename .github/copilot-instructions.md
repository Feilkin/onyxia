# Copilot Instructions for Onyxia

Onyxia is a **GPU compute shader runtime for ONNX models**, built in Rust
2024 edition. ONNX graphs are lowered to a small backend-neutral IR
(primitives + composites, symbolic shapes); backends execute the IR — today
via WGSL compute shaders on wgpu, and via CubeCL.

**Read these before making changes** — they are the source of truth and are
kept current:

- [README.md](../README.md) — crate map, usage, CLI, build & test commands
- [ARCHITECTURE.md](../ARCHITECTURE.md) — the stack, core ideas, testing strategy

## Architecture in one paragraph

`onyxia-onnx` parses protobuf into a `Graph`. `onyxia-lower` lowers it
through a rule registry into the `onyxia-ir` `Module`: a closed enum of ~16
primitives (the entire backend contract) plus open-set composites whose
decompositions live in a registry. The CPU reference interpreter in
`onyxia-ir::interp` is the executable spec. Backends (`onyxia-backend-wgpu`,
`onyxia-backend-cubecl`, `onyxia-backend-ref`) implement `Backend`/`Session`
over device-resident tensors. `onyxia-cli` and `demos/gemma-chat` are
application-layer code.

## Build & test

```bash
cargo build
cargo nextest run                                   # CPU tests (use nextest, not cargo test)
cargo test --doc --workspace                        # doctests (NOT covered by nextest)
just test-all                                       # + GPU differential tests (needs a GPU)
cargo clippy --workspace --all-targets              # must stay warning-free (CI denies warnings)
cargo fmt --all
```

No `protoc` needed: `onyxia-onnx`'s build script compiles the vendored
`proto/onnx.proto` with `protox` (pure Rust).

## Rules for AI agents

- **The reference interpreter is the spec.** When a kernel and the
  interpreter disagree, the kernel is wrong until proven otherwise.
- Every GPU kernel must have a differential test against the interpreter;
  every fused composite kernel differential-tests against its decomposition.
- Growing the primitive enum is a design decision — surface it, don't
  improvise. The closed set is the point of the design.
- Library crates must not panic on user input (model files, tensors);
  return the crate's typed errors. `unreachable!`/`expect` only for local,
  structurally-guaranteed invariants.
- Verify new dependencies with the user before adding them.
- Include doc comments (`///`) for public APIs; add unit tests in
  `#[cfg(test)]` modules.
