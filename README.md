# Onyxia

**GPU compute shader runtime for ONNX models, in Rust.** ONNX graphs are
lowered to a small backend-neutral IR (primitives + composites, symbolic
shapes); backends execute the IR — today via WGSL compute shaders on wgpu
(Vulkan/Metal/DX12/WebGPU), on desktop, mobile, and the web.

## Architecture

```
ONNX Model (.onnx)
     │  onyxia-onnx        protobuf → Graph
     ▼
onyxia-lower              lowering registry: ONNX ops → IR primitives or
     │                    composites; shape subgraphs fold away here
     ▼
onyxia-ir                 Module: ~16 primitives (closed set), composites
     │                    (open set), symbolic dims, const pool. Passes:
     │                    shape inference, constant folding, legalization.
     │                    CPU reference interpreter = the spec. No GPU deps.
     ▼  Backend::prepare(Module) → Session
onyxia-backend-wgpu       generated WGSL primitive kernels + fused composite
(-cubecl, -ref)           kernels; interprets the IR per run with pipeline /
                          bind-group / buffer caches; device-resident tensors
     │
     ▼
onyxia-cli, demos/        generation loop, KV-cache plumbing, tokenizer —
                          application-layer code, not runtime features
```

| Crate | Purpose |
|-------|---------|
| `onyxia-onnx` | Parse ONNX protobuf into a structured `Graph` API |
| `onyxia-ir` | Backend-neutral IR: primitives, composites, symbolic shapes, passes, CPU reference interpreter, `Backend`/`Session` traits |
| `onyxia-lower` | ONNX → IR lowering registry (built-in + contrib ops enter through the same door) |
| `onyxia-backend-wgpu` | wgpu backend: generated primitive kernels, fused composite kernels, symbol binding, device-resident tensors |
| `onyxia-backend-cubecl` | [CubeCL](https://github.com/tracel-ai/cubecl) backend: primitives only — every composite runs via its decomposition, demonstrating that the primitive set is the whole backend contract |
| `onyxia-backend-ref` | Reference backend over the interpreter — the differential-testing oracle |
| `onyxia-cli` | Text generation, model inspection, validation, DOT export |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## The design in one paragraph

The op universe is split in two. **Primitives** (~16 tensor ops: elementwise,
matmul, reduce, reshape/transpose/concat/slice/gather/scatter, cast, select,
iota, dequantize) are a closed enum with fully specified semantics — they are
the *entire* backend contract. **Composites** (Softmax, Gelu, RMS-norm,
GroupQueryAttention, …) are an open set, each with a backend-agnostic
*decomposition* into primitives held in a registry. A backend executes a
composite with a hand-written fused kernel if it has one, or inlines the
decomposition if it doesn't — so custom ops are written once and run on every
backend, and fused kernels are a performance opt-in that differential-tests
against its own decomposition for free.

## Usage

### Running a model

```rust
use onyxia_ir::{Backend, Session};

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    // 1. Parse and lower to IR (no GPU needed up to here)
    let graph = onyxia_onnx::load_and_parse_model("model.onnx")?;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())?;

    // 2. Prepare on a backend
    let ctx = onyxia_backend_wgpu::GpuContext::new().await?;
    let backend = onyxia_backend_wgpu::WgpuBackend::new(ctx);
    let mut session = backend.prepare(module)?;

    // 3. Execute — tensors are device-resident; upload/download are explicit.
    //    Symbolic dims (e.g. sequence_length) bind from actual input shapes.
    let input = onyxia_ir::interp::Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])?;
    let inputs = vec![("input", session.upload(&input)?)];
    let outputs = session.run(&inputs).await?;
    let result = session.download(&outputs[0].1).await?;

    println!("{:?}", result.to_f32()?);
    Ok(())
}
```

Output handles can be fed back as inputs to a later `run` — that is the whole
KV-cache story for LLMs (see `demos/gemma-chat/src/inference.rs`), and it
works for any iterative model without onyxia knowing anything about LLMs.

### Adding a custom operator

One backend-agnostic lowering rule; no per-backend work required. This is
the real RMS-norm decomposition from `onyxia-ir::decomp`:

```rust
use onyxia_ir::{GraphBuilder, ReduceOp, UnaryOp};

fn my_rms_variant(c: &Composite, inputs: &[ValueId], b: &mut GraphBuilder)
    -> Result<Vec<ValueId>>
{
    let (x, w) = (inputs[0], inputs[1]);
    let eps = c.attrs.float_or("epsilon", 1e-5)?;
    let sq = b.mul(x, x)?;
    let ms = b.reduce(ReduceOp::Mean, sq, &[b.ty(x).shape.rank() - 1], true)?;
    let eps_c = scalar(b, b.ty(x).dtype, eps)?;
    let inv = b.unary(UnaryOp::Rsqrt, b.add(ms, eps_c)?)?;
    let normed = b.mul(x, inv)?;
    Ok(vec![b.mul(normed, w)?])
}
```

Registered via `LoweringRegistry::register(domain, op_type, rule)`; ONNX
built-ins, Microsoft contrib ops, and your custom ops all enter through the
same door. A hand-tuned fused kernel can be added per backend later
(`onyxia-backend-wgpu/src/fused.rs` has the pattern); the decomposition
remains the correctness reference it is differential-tested against.

### CLI

```bash
# Generate text (Gemma-style chat models; see justfile for a shortcut)
cargo run --release -p onyxia-cli -- run-model model.onnx --tokenizer <dir> --prompt "Hi" --temperature 0

# Scripted multi-turn chat (tests KV/multi-turn decode)
cargo run --release -p onyxia-cli -- chat model.onnx --tokenizer <dir> -m "first turn" -m "second turn"

# Validate: parse + lower + shape inference, no GPU
cargo run -p onyxia-cli -- validate model.onnx -v

# Inspect the ONNX graph
cargo run -p onyxia-cli -- inspect model.onnx
cargo run -p onyxia-cli -- inspect-node model.onnx --name "/model/layers.0/attn/q_rotary/RotaryEmbedding"
cargo run -p onyxia-cli -- list-nodes model.onnx --op-type MatMul --show-shapes
cargo run -p onyxia-cli -- trace-node model.onnx --name "/model/layers.0/ffn/add" --depth 2

# DOT visualizations (ONNX-level and lowered IR)
cargo run -p onyxia-cli -- dot model.onnx -o model.dot -s summary
cargo run -p onyxia-cli -- ir-dot model.onnx -o module.dot
```

## Demos

`demos/gemma-chat` — egui chat UI running Gemma 3 270m fp32, native and in
the browser (WebGPU, Chrome 149+):

```bash
cargo run --release -p gemma-chat -- models/gemma-3-270m-it-ONNX   # native
cd demos/gemma-chat && trunk serve --release                        # web
```

## Building and testing

```bash
cargo build
cargo nextest run                 # CPU tests (IR, lowering, interpreter, node tests)
just test-all                     # + GPU tests (kernel-vs-interpreter differentials)
just fetch-onnx-tests             # onnx package (Apache-2.0) → its node test data
just conformance                  # ONNX operator conformance matrix (reference backend)
```

The reference interpreter is the spec: every GPU kernel differential-tests
against it, and every fused composite kernel differential-tests against its
own decomposition on-device.

## Operator coverage

Lowering rules exist for 163 of the 172 tensor operators in the ONNX opset
26 default domain (everything except sequences/optionals, strings, control
flow, random sampling, and image decoding), all expressed over the same 16
primitives. On the official onnx node tests, the reference backend passes
every test that is in scope (1275 of 1765; the rest are unsupported dtypes
such as float64/int16/float8, or ops outside the tensor model) and the wgpu
backend passes 1273, with 4-bit, 8-bit and f16 tensors stored packed (or as
native f16 / i64 where the adapter has the shader features). Per-operator
results, the primitive-count analysis, and the remaining gaps are in
[doc/onnx-coverage.md](doc/onnx-coverage.md).

## Profiling

The CLI has a `tracy` feature that installs a
[Tracy](https://github.com/wolfpld/tracy) tracing subscriber
(`just trace-prompt "..."`). Per-op GPU spans are not instrumented yet, so
traces currently show whole-run timing only.

## Example models

Tests and demos look for models under `models/` (not tracked in git — fetch
them from Hugging Face):

```bash
# Gemma 3 270m (git clone pulls every quantization variant; ~2 GB)
git clone https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX models/gemma-3-270m-it-ONNX

# Gemma 3 1b — use the -GQA export, fp32 only (~4 GB instead of ~15 GB)
hf download onnx-community/gemma-3-1b-it-ONNX-GQA \
    --include "onnx/model.onnx*" --include "*.json" --include "chat_template.jinja" \
    --local-dir models/gemma-3-1b-it-ONNX-GQA
```

For the 1B, only [`gemma-3-1b-it-ONNX-GQA`](https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX-GQA)
works: the older `gemma-3-1b-it-ONNX` repo is a raw PyTorch export
(decomposed attention, in-graph mask construction) that its own model card
supersedes, and Onyxia does not support it.

Both the fp32 `onnx/model.onnx` and the 4-bit `onnx/model_q4.onnx`
(`MatMulNBits`, block size 32) run. The q4 weights stay packed on the
device: decode multiplies straight from the nibbles through a fused
matvec kernel and prefill dequantizes per layer into a pooled scratch
matrix, so the 270m drops from 1.07 to 0.76 GiB resident. Prefer fp32 for
quality — the community q4 quantization noticeably degrades these small
models — and note that at 270m the decode step is launch-bound, so q4 is
only a few percent faster. The fp32 models generate token-for-token
identically to onnxruntime under greedy decoding (including past the 1B's
512-token sliding window). On an RTX 3060 Ti: 270m ≈ 128 tok/s decode,
1b ≈ 60 tok/s decode at 3.9 GiB peak VRAM.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

Logo color palette: https://lospec.com/palette-list/technogarten
