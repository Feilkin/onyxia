# Decode performance baseline — 2026-07-03

Where the ~9 tok/s decode speed of the fp32 Gemma 3 270m (the "1 GB model")
comes from, measured on an RTX 3060 Ti (Vulkan) with the new benchmark
tooling. Numbers below are the baseline to beat; re-run after every kernel
change.

> **Status update (same day):** fixes 1–3 below are implemented
> (`fold_transpose_into_matmul` in the IR, split-K matvec kernels for
> M=1, shared-memory tiled matmul for M>1). Measured after:
> **decode 37 ms/tok (27 tok/s)** wall with GPU busy down to 11.4 ms/tok,
> prefill 345 ms, VRAM 1.07 GiB resident (transpose intermediate gone).
> Decode is now CPU-overhead-bound (GPU is 31 % of wall), exactly as
> predicted — fix 4 (fused GQA/rotary, dispatch-count reduction) is the
> next lever. Kernel microbenches after: trans_b lm_head 26.9 → 3.6 ms
> (176 GiB/s), down_proj 720 → 92 µs (≈15 µs kernel + fixed submit
> overhead), prefill lm_head 100 → 31 ms.

## How to reproduce

```sh
# End-to-end prefill/decode benchmark with per-kernel GPU-time breakdown
# (timestamp queries; tokenizer-free, deterministic dispatch stream):
cargo run --release -p onyxia-cli -- bench \
    models/gemma-3-270m-it-ONNX/onnx/model.onnx \
    --prefill-len 64 --decode-tokens 32 --profile --json baseline.json

# Kernel microbenchmarks at Gemma-shaped sizes (criterion; GB/s throughput):
cargo bench -p onyxia-backend-wgpu
```

`--profile` uses `WgpuSession::enable_profiling()` / `take_timings()`
(per-compute-pass begin/end timestamps, tagged with pipeline label and IR
node name). It degrades gracefully on devices without
`TIMESTAMP_QUERY` — i.e. most browsers.

## Baseline (270m fp32, prefill 64, decode 32)

| metric | value |
|---|---|
| decode wall | 102 ms/tok (9.8 tok/s) |
| decode GPU busy | 79 ms/tok (77 % of wall) |
| prefill | 64 tok in 484 ms (132 tok/s) |
| VRAM | 2.07 GiB resident, 2.13 GiB peak |
| dispatches per decode step | ~385 |

Bandwidth bound for reference: one token must read every weight once,
1.08 GiB at the 3060 Ti's ~448 GB/s ⇒ **~2.5 ms/tok (≈400 tok/s)** is the
physical ceiling. We are ~40× off; the gap decomposes almost entirely
into the four items below.

## Where the time goes (decode, GPU side)

| cost | ms/tok | cause |
|---|---|---|
| `/lm_head/Transpose` | 35.1 | the 671 MB tied-embedding weight is re-transposed **every token** |
| `matmul_f32` (162 excl. lm_head) | 37.9 | naive kernel is severely underutilized at small N (see below) |
| `matmul_f32` (lm_head proper) | 1.6 | already at bandwidth — fine |
| everything else (~220 dispatches) | 4.5 | ~13 µs/dispatch launch overhead × many tiny kernels |
| CPU encode + submit + readback | 23 (wall−GPU) | 385 bind groups/passes per token, blocking logits download |

Context length is a non-factor at these sizes (decode at past=512 ≈
past=64), so attention/GQA decomposition overhead is *not* on the
critical path yet — weights-side matmul is.

### Microbenchmark evidence (`cargo bench -p onyxia-backend-wgpu`)

| case | time | effective GB/s |
|---|---|---|
| matmul 1×640×262144 (lm_head, `[K,N]` layout) | 1.68 ms | **371 GiB/s** ✅ |
| matmul 1×640×262144 with `trans_b` (`[N,K]`) | 26.9 ms | 23 GiB/s |
| matmul 1×640×2048 (gate/up) | 273 µs | 18 GiB/s |
| matmul 1×2048×640 (down) | 720 µs | 6.8 GiB/s |
| matmul 64×640×262144 (prefill lm_head) | 100 ms | 6.2 GiB/s |
| transpose 262144×640 | 73 ms | 8.4 GiB/s |
| dispatch chain ×400 (tiny adds) | 5.2 ms | ≈13 µs/dispatch |

Interpretation: the one-thread-per-output-element matmul is *only* fast
when N is huge and B is `[K,N]` (adjacent threads read adjacent B
columns — coalesced, and there are enough threads to fill the GPU). It
collapses when:

- **N is small** (N threads = a handful of workgroups; a 3060 Ti wants
  ~100 k threads): all the per-layer projections.
- **B is `[N,K]`** (`trans_b`): adjacent threads stride by K —
  uncoalesced.
- **M > 1** (prefill): each thread re-walks a full column of B with no
  tiling/reuse.

## Prioritized fixes

1. **Stop re-transposing lm_head (35 ms/tok, the single biggest item).**
   The ONNX graph computes `MatMul(h, Transpose(embed_tokens))`; lowering
   currently materializes the transpose per run. `Prim::MatMul` already
   carries `trans_a`/`trans_b` — add a fold: `MatMul(a, Transpose(b))
   → MatMul(a, b, trans_b)` when the transpose swaps the last two dims.
   **Requires fix 2 to pay off** (the naive trans_b path is 26.9 ms —
   barely better than transpose+matmul at 36.7 ms).
2. **Matvec kernel for decode-shaped matmuls (M=1), both layouts.**
   Workgroup-per-output-tile with threads striding K and a shared-memory
   reduction (split-K). This is the standard matvec pattern and should
   put every M=1 case near bandwidth. Fixes 1+2 together take decode GPU
   time from ~79 ms to an estimated ~8 ms.
3. **Tiled matmul for prefill (M>1).** Prefill lm_head runs at 6 GiB/s;
   classic shared-memory tiling. This is the known-gaps "no tiled MatMul
   yet" item, now with a number attached.
4. **Dispatch-count reduction (helps both the 4.5 ms GPU tail and the
   23 ms CPU gap).** Fused GQA/RotaryEmbedding kernels (already planned
   in `fused.rs`), plus possibly bind-group caching. Worth re-measuring
   after 1–3; the CPU side becomes the bottleneck once GPU busy drops
   below ~25 ms.

Post fix 1–3, decode should land in the 25–40 ms/tok range
(25–40 tok/s), CPU-overhead-bound; fix 4 is what unlocks the rest.

## Known blockers found along the way

- `models/gemma-3-1b-it-ONNX` (fp32) does not lower yet:
  `node '/Equal': a symbolic shape value is consumed by a runtime tensor
  operation — cannot materialize`. The 1B baseline is blocked on this.
- The lm_head transpose also costs VRAM: its 671 MB intermediate lands in
  a 1 GiB buffer-pool bucket, nearly doubling resident memory
  (2.07 GiB for a 1.08 GiB model). Fix 1 removes that too.
