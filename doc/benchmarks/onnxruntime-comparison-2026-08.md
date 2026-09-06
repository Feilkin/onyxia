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

## Update 2026-09-05: q4 models, chunked submits — same protocol, D = 128

Onyxia @ `6205dca` (fused MatMulNBits + GatherBlockQuantized, submits every
64 dispatches); onnxruntime-gpu 1.29.0 CUDA EP and `onnxruntime-ep-webgpu`
0.2.1, freshly installed the same day. 64-token prefill, 128 measured
decode steps; ORT rows show the better of numpy / IOBinding KV. Onyxia
resident VRAM in parentheses.

| model | Onyxia (wgpu/Vulkan) | ORT CUDA EP | ORT WebGPU EP | Onyxia / CUDA | Onyxia / WebGPU |
|---|---|---|---|---|---|
| 270m fp32 decode | **307 tok/s** (3.26 ms) (1.07 GiB) | 287 tok/s (3.48 ms) | 84 tok/s (11.9 ms) | **1.07×** | 3.6× |
| 270m q4 decode | **360 tok/s** (2.78 ms) (0.76 GiB) | 315 tok/s (3.17 ms) | 98 tok/s (10.2 ms) | **1.14×** | 3.7× |
| 1B fp32 decode | **163 tok/s** (6.14 ms) (3.81 GiB) | 164 tok/s (6.09 ms) | 24 tok/s (41.2 ms) | **0.99×** | 6.7× |
| 1B q4 decode | **262 tok/s** (3.82 ms) (0.82 GiB) | 237 tok/s (4.23 ms) | 71 tok/s (14.1 ms) | **1.11×** | 3.7× |
| 270m fp32 prefill | 25.9 ms | 7.4 ms | 20.6 ms | 0.29× | 0.80× |
| 270m q4 prefill | 26.5 ms | 8.1 ms | 19.6 ms | 0.31× | 0.74× |
| 1B fp32 prefill | 45.7 ms | 11.7 ms | 56.8 ms | 0.26× | 1.24× |
| 1B q4 prefill | 50.3 ms | 19.3 ms | 35.4 ms | 0.38× | 0.70× |

Reading it:

- **Decode is now at parity with ORT-CUDA** on every model (0.99–1.14×),
  up from 73 % in August; the whole gain is the chunked submission
  (`doc/perf-baseline-2026-07.md`, 2026-09-05 section). Both runtimes
  are launch-bound here — ORT's q4 1B step is 4.2 ms for 0.86 GB of
  weights that the 5090 could stream in ~0.5 ms — so this is a statement
  about per-step overhead, not kernels.
- **Prefill is still cuBLAS's**: 2.6–3.9× behind. Onyxia's 16×16
  shared-memory tile is the limit; the q4 variant, which dequantizes the
  weight tile on load, is ~10 % slower than the fp32 tile at 1B and is
  the one place ORT's WebGPU EP is ahead (35 vs 50 ms).
- ORT-CUDA on q4 gains less than Onyxia does (237 vs 164 tok/s on the
  1B, +45 %; Onyxia +61 %), and its IOBinding mode is *slower* than
  numpy KV on every row — Python-side binding cost again. The ORT numbers
  carry that per-step Python overhead; `onnxruntime-genai` would remove
  it and was not measured.
- The 1B fp32 ORT-WebGPU row (24 tok/s) is unchanged from August, so the
  plugin is the same build; its q4 decode (71 tok/s) is ~3× its fp32.

### Same day, onnxruntime through the C API (no Python)

`ort-bench-rs/` in this directory drives the same `libonnxruntime.so`
1.29.0 CUDA EP from Rust through the C API (via the `ort` crate 2.0
rc.13, `load-dynamic`), same protocol. This removes the Python per-step
cost the earlier rows carried:

| model | Onyxia | ORT CUDA, C API, host KV | ORT CUDA, C API, IoBinding | ORT CUDA via Python (above) | Onyxia / ORT C API |
|---|---|---|---|---|---|
| 270m fp32 decode | 307 tok/s (3.26 ms) | **298 tok/s** (3.35 ms) | 220 tok/s (4.54 ms) | 287 tok/s | 1.03× |
| 270m q4 decode | 360 tok/s (2.78 ms) | **374 tok/s** (2.67 ms) | 290 tok/s (3.45 ms) | 315 tok/s | 0.96× |
| 1B fp32 decode | 163 tok/s (6.14 ms) | **169 tok/s** (5.91 ms) | 149 tok/s (6.71 ms) | 164 tok/s | 0.96× |
| 1B q4 decode | 262 tok/s (3.82 ms) | **241 tok/s** (4.15 ms) | 181 tok/s (5.53 ms) | 237 tok/s | 1.09× |
| 270m fp32 prefill | 25.9 ms | 7.4 ms | 7.7 ms | 7.4 ms | 0.29× |
| 270m q4 prefill | 26.5 ms | 8.8 ms | 8.3 ms | 8.1 ms | 0.31× |
| 1B fp32 prefill | 45.7 ms | 11.9 ms | 12.2 ms | 11.7 ms | 0.26× |
| 1B q4 prefill | 50.3 ms | 19.2 ms | 19.2 ms | 19.3 ms | 0.38× |

- Python was worth 0.1–0.5 ms per step to ORT; the C API rows are the
  fair ceiling for onnxruntime-CUDA on this export. Onyxia is within
  ±10 % on decode either way (0.96–1.09×) and unchanged on prefill.
- IoBinding is slower than plain host-side KV **from Rust too**, so that
  is ORT's binding path (a binding object per step, device-side output
  allocation, `GetBoundOutputValues`), not the Python wrapper. At these
  KV sizes (a few MB at ≤200 tokens) the host round-trip is cheaper.
- ORT-CUDA on the 270m q4 is now the one row where it leads (374 vs 360
  tok/s) — 2.7 ms per step for 126 MatMulNBits + attention is close to
  what its launch count allows, and Onyxia's per-dispatch CPU cost
  (bind-group misses, `run_prim` bodies) is what stands between them.

```sh
cd doc/benchmarks/ort-bench-rs && cargo build --release
./target/release/ort-bench-rs ../../../ortenv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0 \
    ../../../models/gemma-3-1b-it-ONNX-GQA/onnx/model_q4.onnx cuda-host 64 128
```

### Same day, after the register-blocked prefill tile (`matmul_tiled_rb`)

Prefill only changed; decode within noise. Onyxia 64-token prefill vs
the ORT C-API rows above: 270m fp32 **20.0 ms** (0.37× cuBLAS, 1.03×
ORT-WebGPU), 270m q4 **19.7 ms** (0.45× / 0.99×), 1B fp32 **26.4 ms**
(0.45× / 2.15×), 1B q4 **24.7 ms** (0.78× / 1.43×). Details and the
tensor-core experiment in `doc/perf-baseline-2026-07.md`.

Reproduce (`ortenv` as below, then):

```sh
cargo run --release -p onyxia-cli -- bench models/gemma-3-1b-it-ONNX-GQA/onnx/model_q4.onnx --prefill-len 64 --decode-tokens 128
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-1b-it-ONNX-GQA/onnx/model_q4.onnx cuda-numpy 64 128
ortenv/bin/python doc/benchmarks/ort_bench.py models/gemma-3-1b-it-ONNX-GQA/onnx/model_q4.onnx webgpu 64 128
```

## 2026-09-06: the day-before-RustConf table (all backends, plus an embedding model)

Same machine (RTX 5090, Ryzen 9 9950X3D), onnxruntime-gpu 1.29.0 (CUDA
EP through the C API via `ort-bench-rs`; CPU and WebGPU EP rows through
`ort_bench.py`, where Python costs ≤ 0.5 ms/step), `onnxruntime-ep-webgpu`
0.2.1. Onyxia at this commit, Vulkan. Protocol as above: 64-token
prefill, 128 measured decode steps; the Onyxia prefill is now the median
of 5 (`onyxia bench`; a single 1B fp32 prefill swung 25–52 ms run to run,
the decode numbers did not move). CPU EP uses all 16 cores.

New today: `onnx-community/embeddinggemma-300m-ONNX` (fp32, 1.15 GiB),
an encoder with no KV cache — `onyxia bench` and both ORT scripts got a
forward protocol for models without a `logits` output: 32 timed
stateless passes over 64 tokens, the pooled `sentence_embedding` copied
to host every pass. Running it needed two lowering additions
(`Where(shape == -1, …)` folding against symbolic dims; a
`com.microsoft.MultiHeadAttention` rule onto the `Attention` composite)
and, for CubeCL, integer matmuls through f32 (the CumSum decomposition).
The embedding matches ORT-CPU to 2e-7 max abs (cosine 1 − 2e-12).
`onyxia bench --backend cubecl` is also new; the 1B needed the GQA
decomposition to restate the attention-bias's `total_sequence_length`
dim as `past + seq` (a symbolic reshape) before it would build.

**Decode**, tok/s (ms/tok); **prefill** of 64 tokens, ms. Onyxia
resident VRAM in the first column.

| model | Onyxia wgpu | Onyxia CubeCL | ORT CUDA EP | ORT CPU EP | ORT WebGPU EP |
|---|---|---|---|---|---|
| 270m fp32 decode | **331** (3.03) · 1.16 GiB | 47 (21.1) | 274 (3.65) | 43 (23.4) | 82 (12.1) |
| 270m q4 decode | **423** (2.37) · 0.84 GiB | — (no 4-bit tensors) | 355 (2.82) | 68 (14.7) | 95 (10.5) |
| 1B fp32 decode | **175** (5.72) · 3.91 GiB | 24 (42.4) | 159 (6.29) | 12 (82.8) | 24 (41.2) |
| 1B q4 decode | **306** (3.26) · 0.92 GiB | — | 232 (4.30) | 65 (15.5) | 71 (14.1) |
| 270m fp32 prefill | 19.4 | 79.9 | **7.5** | 31.0 | 20.8 |
| 270m q4 prefill | 18.8 | — | **8.1** | 29.6 | 19.6 |
| 1B fp32 prefill | 25.0 | 172.7 | **12.5** | 101.3 | 55.5 |
| 1B q4 prefill | 24.4 | — | **20.1** | 64.0 | 33.9 |

Embedding model, one 64-token forward pass (mean of 32), ms; passes/s
in parentheses:

| model | Onyxia wgpu | Onyxia CubeCL | ORT CUDA EP | ORT CPU EP | ORT WebGPU EP |
|---|---|---|---|---|---|
| embeddinggemma-300m fp32 | 6.37 (157/s) · 1.16 GiB | 40.6 (25/s) | **2.04** (490/s) | 19.6 (51/s) | 10.6 (95/s) |

Ratios, Onyxia wgpu vs the others:

| | vs ORT CUDA | vs ORT CPU | vs ORT WebGPU |
|---|---|---|---|
| 270m fp32 decode | 1.21× | 7.7× | 4.0× |
| 270m q4 decode | 1.19× | 6.2× | 4.5× |
| 1B fp32 decode | 1.10× | 14.5× | 7.2× |
| 1B q4 decode | 1.32× | 4.7× | 4.3× |
| 270m fp32 prefill | 0.39× | 1.6× | 1.07× |
| 270m q4 prefill | 0.43× | 1.6× | 1.04× |
| 1B fp32 prefill | 0.50× | 4.1× | 2.2× |
| 1B q4 prefill | 0.82× | 2.6× | 1.4× |
| embedding forward | 0.32× | 3.1× | 1.7× |

Reading it:

- **Decode is ahead of ORT-CUDA on every model** (1.10–1.32×), from
  parity a day earlier; the fusion pass (`onyxia_ir::fuse`) and the
  single-dispatch GQA concat are the difference. Both runtimes are still
  launch-bound here.
- **Prefill and the encoder are cuBLAS's**: 0.32–0.82× of ORT-CUDA. The
  embedding model is a pure prefill (64 tokens through 24 layers plus
  pooling), so it lands where the prefill rows do — 3× behind cuBLAS,
  1.7× ahead of the WebGPU EP.
- **CubeCL backend**: 7× behind the wgpu backend on 270m decode, 7× on
  the 1B, 6× on the encoder — unchanged in character since August: the
  fully decomposed graph (no fused GQA / RoPE / norm kernels, no tiled
  matmul) is launch-count-bound. The q4 exports do not run there (no
  packed 4-bit layout in that backend).
- ORT-CPU on q4 is 1.6× (270m) / 5× (1B) its own fp32: MatMulNBits has a
  good AVX-512 kernel; Onyxia wgpu is still 4.7–7.7× ahead of it.

```sh
cargo run --release -p onyxia-cli -- bench models/embeddinggemma-300m-ONNX/onnx/model.onnx --prefill-len 64 --repeats 32
cargo run --release -p onyxia-cli -- bench models/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx --backend cubecl --prefill-len 64 --decode-tokens 128
doc/benchmarks/ort-bench-rs/target/release/ort-bench-rs ortenv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0 models/embeddinggemma-300m-ONNX/onnx/model.onnx cuda-host 64 32
ortenv/bin/python doc/benchmarks/ort_bench.py models/embeddinggemma-300m-ONNX/onnx/model.onnx cpu 64 32
# last-row logits variants for ORT (needs the onnx package: `just fetch-onnx-tests`)
.venv/bin/python doc/benchmarks/last_row.py models/gemma-3-1b-it-ONNX-GQA/onnx/model.onnx
```

### Same day, later: prefill was mostly the logits download

`onyxia bench` grew a host-clock split for the prefill (as it had for
decode), which put 11–12 ms of the 20–25 ms prefill in *readback*: the
session downloaded the whole `[1, 64, 262144]` logits tensor (67 MB
through a fresh staging buffer) to keep one row. `Session::download_range`
(new trait method, default = download-and-slice; the wgpu backend copies
the byte range, the CubeCL backend reads a sub-handle) fetches the last
row only, from `LlmSession` and the gemma-chat demo alike. Two smaller
GPU-side changes on top: the register-blocked prefill tile stages its
operands with `vec4` loads when the rows are 4-aligned (`_v4` variants,
nn and nt; the 1B's projection matmuls 6.7 → 5.9 ms GPU), and the
split-K target went 256 → 512 workgroups (1B fp32 prefill 9.3 → 8.8 ms;
flat elsewhere). Decode is untouched: it never ran these paths.

To keep onnxruntime on the same footing, `last_row.py` writes
`<model>_last.onnx` next to each export: the same graph with a `Slice`
so `logits` is `[1, 1, V]` and `run()` copies one row, exactly what
Onyxia now does. Both ORT scripts also take the prefill as the median of
5. ORT rows below are those variants; the full-logits ORT prefill is in
parentheses for reference.

**Prefill, 64 tokens, ms** (decode unchanged from the morning table):

| model | Onyxia wgpu (was) | ORT CUDA EP, last row (full logits) | ORT CPU EP, last row | ORT WebGPU EP, last row |
|---|---|---|---|---|
| 270m fp32 | **4.1** (19.4) | 4.0 (7.6) | 32.6 | 11.6 |
| 270m q4 | **4.0** (18.8) | 4.9 (8.3) | 31.9 | 11.5 |
| 1B fp32 | **8.9** (25.0) | 8.8 (11.9) | 96.0 | 43.7 |
| 1B q4 | **9.1** (24.4) | 16.1 (19.3) | 56.3 | 26.3 |

Onyxia / ORT-CUDA prefill: 0.98×, 1.23×, 0.99×, 1.77× — parity with
cuBLAS at 64 tokens on the fp32 exports, ahead on q4. The remaining 1B
fp32 prefill is 8.9 ms wall for ~9.5 ms of GPU time (profiled: the
projection matmuls 5.9 ms, lm_head 1.5, split-K reduces 1.1, attention
0.9, norms 1.1) with ~1.1 ms of encode overlapped; the next lever is the
matmul tile itself (a 64×64 tile with 4×4 micro-tiles is at ~25 TFLOPS
against the 5090's ~100), then folding the 182 split-K reduces.

The ORT CUDA *decode* on the `_last` graphs is noisier than on the
originals (σ up to 1.7 ms/step) and its q4 1B row came out slower
(213 vs 231 tok/s); the morning table's decode column stands.

CubeCL backend after the same change: 270m fp32 prefill 79.9 → 70.1 ms,
1B fp32 172.7 → 160.4 ms (decode 49 / 24 tok/s, unchanged) — the same
~10 ms of download gone, the rest is its unfused kernels.

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
