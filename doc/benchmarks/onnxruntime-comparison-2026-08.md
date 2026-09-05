# Onyxia vs onnxruntime — 2026-08-23

Same protocol on both sides (`onyxia bench` and `ort_bench.py` in this
directory): tokenizer-free, warmup prefill(64) + 2 decode steps, then a
measured prefill of 64 tokens and 32 measured single-token decode steps.
Logits are copied to host every step on both sides. KV cache is
device-resident in Onyxia; for onnxruntime both the naive numpy path (KV
round-trips host each step — what a typical Python caller does) and
IOBinding (KV stays on device) were measured and the better one is shown.

Hardware: **RTX 5090** (32 GB, Vulkan for Onyxia, CUDA 13.3 for ORT),
Ryzen 9 9950X3D. Note the July perf docs were measured on an RTX 3060 Ti;
Onyxia numbers here are re-measured on the 5090.

Software: onyxia @ `0e6b4fe` (wgpu 29, Vulkan); onnxruntime-gpu **1.29.0**
(CUDA EP; 73 MemCpy nodes inserted — ORT runs parts of these graphs on CPU);
`onnxruntime-ep-webgpu` **0.2.1** — ORT's WebGPU EP as a runtime plugin,
running natively over Dawn, i.e. the same API layer Onyxia uses via wgpu.
The WebGPU EP rows use host-side KV (the plugin rejects IOBinding to its
device from Python: "Failed to find allocator for device"); on the CUDA
rows host-side KV cost ≤ 0.7 ms/step, so this does not explain the gap.

## Gemma 3 270m fp32 (`onnx-community/gemma-3-270m-it-ONNX`, 1.08 GiB weights)

| runtime | decode | prefill (64 tok) | VRAM |
|---|---|---|---|
| **Onyxia** (wgpu/Vulkan) | **232 tok/s** (4.31 ms/tok) | **2,460 tok/s** (26.0 ms) | 1.07 GiB resident / 1.13 peak |
| onnxruntime CUDA EP, numpy KV | 316 tok/s (3.16 ms/tok) | 8,540 tok/s (7.5 ms) | — |
| onnxruntime CUDA EP, IOBinding KV | 262 tok/s (3.82 ms/tok) | 8,340 tok/s (7.7 ms) | — |
| **onnxruntime WebGPU EP** (plugin 0.2.1, Dawn) | **76 tok/s** (13.1 ms/tok) | 2,950 tok/s (21.7 ms) | — |
| onnxruntime CPU EP (16 cores) | 45 tok/s (22.3 ms/tok) | 1,820 tok/s (35 ms) | — |

Onyxia decode = **73 % of ORT-CUDA**, **3.0× ORT-WebGPU**; prefill = 29 % of CUDA, 0.83× WebGPU.

## Gemma 3 1B fp32 (`onnx-community/gemma-3-1b-it-ONNX-GQA`, ~4 GiB weights)

| runtime | decode | prefill (64 tok) | VRAM |
|---|---|---|---|
| **Onyxia** (wgpu/Vulkan) | **130 tok/s** (7.70 ms/tok) | **1,270 tok/s** (50.3 ms) | 3.80 GiB resident / 3.86 peak |
| onnxruntime CUDA EP, numpy KV | 177 tok/s (5.66 ms/tok) | 5,430 tok/s (11.8 ms) | — |
| onnxruntime CUDA EP, IOBinding KV | 158 tok/s (6.35 ms/tok) | 4,590 tok/s (14.0 ms) | — |
| **onnxruntime WebGPU EP** (plugin 0.2.1, Dawn) | **24 tok/s** (41.3 ms/tok) | 1,200 tok/s (53.5 ms) | — |

Onyxia decode = **73 % of ORT-CUDA**, **5.4× ORT-WebGPU**; prefill = 23 % of CUDA, 1.06× WebGPU.

## Onyxia's CubeCL backend on the same GPU (270m, 2026-08-23)

Re-run after the July perf pass, which only touched the wgpu backend. The
CubeCL backend needed one fix to run at all: the 1B work made the
`seqlens_k = ReduceSum(attention_mask)` subgraph live, and the backend only
had an f32 reduce — `reduce_i32` added (+ a differential test).

| | wgpu backend | CubeCL backend (primitives only) |
|---|---|---|
| prefill warm, S=15 (`forward-check`) | 13.9 ms | 67.1 ms |
| prefill cold incl. JIT | 16.8 ms | 747 ms |
| greedy decode (`run-model`, 36 tokens) | 211 tok/s | 15.4 tok/s |
| max \|Δlogit\| vs wgpu | — | 2.5e-5, argmax identical, text identical |

**Same day, after the M=1 matvec port + launch-overhead fixes**
(`matvec_kn_v4` / `matvec_transb_v4` / `matvec_reduce` ported to
`#[cube]`; content-keyed cache of kernel-parameter handles; reuse of
intermediate output buffers across runs):

| | wgpu backend | CubeCL backend |
|---|---|---|
| prefill warm, S=15 | 11.3 ms | **35 ms** (was 67) |
| greedy decode | 210 tok/s | **48 tok/s** (was 15.4) |
| max \|Δlogit\| vs wgpu | — | 2.5e-5, argmax + text identical |

Where the CubeCL decode step went (per-step trace, 2,632 launches):
creating a ~100-byte parameter buffer per launch was **35 ms** of a 63 ms
loop (13 µs each, more than the launch itself) → 0.25 ms with the cache;
output allocation 9.5 → 2.5 ms; the launches themselves ~15 ms (≈5.7 µs
each, CubeCL's floor). GPU time per step: ~19 ms with the matvec kernels,
~25 ms without. Kernel microbench on the 5090 (M=1, fp32): lm_head
`[262144,640]ᵀ` 0.635 → 0.469 ms (1.33 TB/s, ~75 % of peak); down_proj
`[2048,640]` 65 → 16 µs; q_proj 27 → 12 µs; gate/up 28 → 21 µs.

The remaining 4.4× decode gap to the wgpu backend is launch count: the
fully decomposed graph is 2,632 launches/step vs ~500 with fused
GQA/RoPE/Softmax/RMS-norm, and both the CPU (5.7 µs/launch) and the GPU
(~5 µs/launch) are bound by it. Fusion — a composite-kernel registry for
this backend — is the only lever left, and it is the same ~2-day item as
on the wgpu side.

In July (3060 Ti, pre-perf-pass) CubeCL prefill was *faster* (156 vs
182 ms) because both backends were naive. The wgpu backend since gained
tiled/split-K matmul, fused GQA/RoPE, batched submission, and bind-group
caching; the CubeCL backend is still the line-for-line port of the naive
kernels with kernel-by-kernel launches. Before the matvec port the gap was ~5× on prefill and ~14× on decode —
all of it untuned-vs-tuned, none of it CubeCL-vs-wgpu.

## Reading the numbers

- **Decode** is launch-overhead-bound on both runtimes at these model
  sizes: the 5090's bandwidth floor for the 270m is ~0.6 ms/tok
  (1.08 GiB at ~1.8 TB/s), and neither runtime is within 5× of it. Onyxia
  does ~520 dispatches per token through wgpu/Vulkan; ORT has cuBLAS +
  fused CUDA attention and still pays ~3 ms. The gap is 1.3–1.4×.
- **Prefill** is where cuBLAS shows: Onyxia's 16×16 shared-memory tiled
  matmul is ~3.5–4.4× behind. This is the obvious next kernel to work on
  (or the place to use CubeCL's matmul components — see
  `cubecl-experience.md`).
- **Like-for-like API layer:** ORT's WebGPU EP is the fair comparison —
  same WebGPU abstraction, no cuBLAS. There Onyxia decodes 3× (270m) to
  5× (1B) faster with equal prefill. Possible reasons on the ORT side:
  an early plugin (0.2.1), CPU fallbacks for some contrib ops on this
  export, per-step Python overhead; it is the number a user of the
  plugin gets today, not a ceiling for ORT-on-WebGPU.
- Onyxia is the same source on Vulkan, Metal, DX12, and WebGPU; the
  ORT-CUDA number is a CUDA-only execution provider.
- Correctness: both models are greedy-token-identical to ORT
  (`README.md`), so the comparison is like-for-like.
- Naive-numpy ORT beating IOBinding on decode is Python binding overhead:
  the KV at 64–96 tokens is a few MB, cheaper to copy than to rebind 36
  OrtValues per step.

## Reproduce

```sh
# Onyxia
cargo run --release -p onyxia-cli -- bench models/gemma-3-270m-it-ONNX/onnx/model.onnx --prefill-len 64 --decode-tokens 32

# onnxruntime (needs CUDA; python 3.12 venv)
uv venv -p 3.12 ortenv && uv pip install -p ortenv/bin/python onnxruntime-gpu numpy
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-270m-it-ONNX/onnx/model.onnx cuda-numpy 64 32
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-270m-it-ONNX/onnx/model.onnx cuda-iobinding 64 32
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-270m-it-ONNX/onnx/model.onnx cpu 64 32
uv pip install -p ortenv/bin/python onnxruntime-ep-webgpu
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-270m-it-ONNX/onnx/model.onnx webgpu 64 32
```

Not measured: `onnxruntime-web` in an actual browser (the in-browser
comparison against `demos/gemma-chat`); and ORT with
`onnxruntime-genai`'s generation loop, which would remove the Python
per-step overhead from the ORT side.
