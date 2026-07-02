# CubeCL backend: experience report

Written 2026-07-02, immediately after landing `onyxia-backend-cubecl`
(commit `ae11c72`). Raw material for the RustConf talk — especially §3
("How Rust Made It Possible") and the multi-backend lesson. Companion to
the condensed status block in `ir-implementation-plan.md` milestone E.

## TL;DR

One working day from empty crate to **full Gemma 3 270m prefill matching
the wgpu backend to |Δlogit| ≤ 3.1e-5**, implementing *only the ~16 IR
primitives* — zero composite kernels, everything (GQA included) legalized
through decompositions. Warm prefill came out **faster than our wgpu
backend** (156 ms vs 182 ms) despite the wgpu backend having hand-fused
Softmax/RMS-norm/Gelu kernels and the CubeCL backend having no fusion, no
memory planning, and index math ported line-for-line from our naive
one-thread-per-element WGSL. Neither backend is tuned; the point is not
"CubeCL is fast" but "an untuned port on a good compute stack beats an
untuned hand-rolled stack" — and that the backend contract made the port
this small.

## What was built

`crates/onyxia-backend-cubecl` — `Backend`/`Session` over `cubecl-wgpu`
0.10, primitives only (`supports()` ≡ `false`):

- `kernels.rs` (639 lines): one `#[cube]` fn per kernel family — binary,
  compare, unary, cast, select, matmul, reduce, transpose, broadcast,
  concat, slice, gather, iota — direct ports of the wgpu backend's
  generated WGSL index math, so the two backends are apples-to-apples.
- `lib.rs` (958 lines): legalize → toposort → upload consts → per-node
  launch. Same structure as the wgpu session, minus everything CubeCL
  owns (see below).
- 6 GPU differential tests vs `onyxia-backend-ref` (the interpreter),
  including the full GQA decomposition with symbolic dims, past-KV, and
  sliding window; plus `examples/forward-check.rs`, the whole-model gate
  against the wgpu backend.

Explicit non-goals (clean `Unsupported` errors): Scatter, Dequantize
(q4), f16, cast→Bool, memory planning, non-blocking readback.

## Numbers

RTX 3060 Ti, Gemma 3 270m fp32, S=15 prefill, both backends in one
process (`cargo run --release -p onyxia-backend-cubecl --example
forward-check`):

| | prepare | prefill cold | prefill warm |
|---|---|---|---|
| onyxia-backend-wgpu (fused Softmax/RMS-norm/Gelu) | 2.07 s | 243 ms | 182 ms |
| onyxia-backend-cubecl (primitives only) | 3.22 s | 631 ms | **156 ms** |

- Cold includes JIT: CubeCL compiles each kernel specialization on first
  use (~2.6× our cold pass, where pipelines also compile lazily but from
  pre-generated WGSL).
- **Decode reverses the result** (greedy, 64 tokens, via
  `onyxia run-model --backend {wgpu,cubecl}`): wgpu **10.1 tok/s**,
  cubecl **6.9 tok/s**. Prefill is compute-bound and CubeCL's codegen
  wins; decode at S=1 is launch-overhead-bound, and there the wgpu
  backend's advantages are structural: fused Softmax/RMS-norm/Gelu means
  *fewer launches* (each decomposed softmax is ~5), and it batches every
  dispatch of a forward pass into a single command-buffer submission,
  where the CubeCL session launches kernel-by-kernel through the client.
  Nice talk shape: *which* backend is faster depends on whether the
  workload amortizes launch overhead — neither stack is simply better.
- max |Δlogit| at the last position: **3.1e-5**, argmax identical.
- Code size: wgpu backend 2 242 lines (kernels 541 + session 1090 +
  gpu 297 + fused 292 + lib 22); cubecl backend **1 597 lines** (kernels
  639 + lib 958) — ~70%, with no fused-kernel layer at all.
- Dependency note: `cubecl-wgpu` 0.10 pins **wgpu 29.0.3 — the exact
  version we already use**, so no duplicate wgpu in the tree. The cubecl
  crates themselves add ~40 s to a cold build.

## What the onyxia backend contract got right (the design claim, tested)

This spike existed to answer: *is the primitive set really the whole
backend contract?* Yes, empirically:

1. **Legalization needed zero work.** `inline_composites(module, …,
   |_| false)` + the standard decomposition registry, and GQA / Softmax /
   RMS-norm / Rotary / Gelu never reached the executor. The one hand-fused
   path the wgpu backend has is genuinely optional.
2. **The differential-test harness transferred wholesale.** `diff_test`
   is character-identical to the wgpu suite except the two backend
   constructor lines. The reference interpreter as "the spec" means a new
   backend starts with a complete conformance suite for free.
3. **Kernels ported 1:1 because the semantics were already pinned.** The
   index math (broadcast `src_index`, axes-bitmask reduce, emplace
   concat) came straight from the WGSL, which the interpreter had already
   differential-tested. No semantic decisions were made in the port —
   only syntax.
4. **The session halved.** Everything backend-*infrastructure* — pipeline
   cache, bind-group layout reflection (the entire wgpu-29 immediates
   saga!), buffer pool, immediates-vs-storage fallback — is CubeCL's
   problem, not ours. What remains is exactly the part the IR dictates:
   bind symbols, evaluate shapes, walk nodes, pack params, launch.
5. **No graphics API named anywhere.** `WgpuRuntime` appears only as a
   type parameter; `cubecl-cuda`/`cubecl-hip` are the same code. (CUDA
   run pending a toolkit install.)

## What CubeCL got right

- **Kernels are Rust.** Same language, same file, same tooling end to
  end. rustfmt formats them; rust-analyzer completes them; the borrow
  checker doesn't apply inside `#[cube]` but ordinary typos die at
  `cargo check` instead of at naga-parse-time with a source dump, which
  is how our WGSL string templates fail.
- **`comptime` replaces string codegen.** Our wgpu backend generates WGSL
  per op via `format!` templates. Here, one kernel fn takes
  `#[comptime] op: u32` and the branch folds at JIT time:

  ```rust
  #[cube(launch_unchecked)]
  pub fn binary_f32(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>,
                    p: &Array<u32>, #[comptime] op: u32) {
      // …broadcast index math…
      let r = if comptime![op == OP_ADD] { av + bv }
         else if comptime![op == OP_POW] { av.powf(bv) }
         else { av.min(bv) };
      out[idx] = r;
  }
  ```

  One source, N specialized kernels, no strings. This is the
  metaprogramming story WGSL templates fake badly.
- **The compute-client model is pleasant.** `client.create_from_slice`,
  `client.empty`, `client.read_one`, refcounted handles, internal
  batching and memory pooling. Our `GpuTensor { Arc<Buffer>, … }` +
  BufferPool + encoder plumbing is ~500 lines of session code that here
  simply doesn't exist.
- **Performance floor is high.** The 156 ms warm number is with our naive
  per-element kernels. CubeCL's own matmul/reduce components (which we
  deliberately didn't use, to keep the comparison honest) would lower it
  further.

## Friction, honestly

Talk-worthy because it's the real texture of using a fast-moving
ecosystem crate:

- **Docs lag the release; the source doesn't.** The repo's `main`-branch
  examples use unreleased API (`Vector<F, N>`, `BufferArg`); the 0.10
  examples use `Array`/`ArrayArg`. The reliable documentation was the
  **vendored crate source in `~/.cargo/registry`** — especially
  `cubecl-core/src/runtime_tests/`, which is a de-facto cookbook. (This
  is itself a Rust-ecosystem point: docs rot, but `cargo` hands you the
  exact source of the version you compile against.)
- **The eDSL boundary leaks in type errors.** Inside `#[cube]`,
  `ABSOLUTE_POS` is `usize`, arrays index by `usize`, and mixing in `u32`
  params produces errors like:

  ```
  error[E0277]: the trait bound `NativeExpand<usize>:
      From<NativeExpand<u32>>` is not satisfied
  ```

  Perfectly diagnosable — but you must know you're programming a macro
  expansion, not plain Rust. First contact costs an hour.
- **Math functions are traits with non-std names.** `ln` (not `log`),
  `inverse_sqrt` (not `rsqrt`), and `.erf()` collides with nightly std's
  unstable `f32::erf` → an `unstable_name_collisions` warning fixed by
  the fully-qualified `cubecl::frontend::Erf::erf(v)`.
- **Generic-over-dtype kernels fight trait bounds.** `O::cast_from(x)`
  doesn't come with `Numeric` alone; the pragmatic answer was concrete
  per-dtype kernels (`cast_f32_i32`, …) — mirroring what our WGSL
  templates do anyway. Data-movement kernels (`transpose<N: Numeric>`)
  generic'd fine.
- **Branch typing inside kernels is fussy.** `if cond { f32::new(x) }
  else { 1.0f32 }` fails to unify (expand type vs literal); the fix is
  computing the constant in a `comptime!` block first. Similarly,
  if-else-assign sometimes wants the `let mut r = …; if … { r = … }`
  shape.
- **The launch API is `unsafe`.** `ArrayArg::from_raw_parts(handle,
  len)` with element counts you must get right, `launch_unchecked` — the
  safety story delegates bounds checking back to your kernel's
  `if idx < size` guard, same as raw WGSL.

None of these were blockers; the whole friction budget was maybe two
hours of the day, and rustc's messages carried the port — I wrote the
kernels against a half-remembered API and let the compiler negotiate the
real one. That workflow ("the compiler as API documentation") only works
in a language where GPU kernels *are* host-language code.

## What a production CubeCL backend would still need

- Scatter, Dequantize (q4 models), f16, cast→Bool — mechanical.
- Async readback (`read_one` blocks; wasm needs the async path — the
  Session trait is already async on our side).
- Memory behavior review: registers hold every intermediate for the whole
  run (no liveness); CubeCL's pool reuses across runs but peak-per-run is
  unplanned. Fine at 270m; needs a look for 1B+.
- Real perf work would *use* CubeCL rather than port WGSL: line
  (vectorized) types, its matmul components, plane operations for
  reductions — i.e., the places where CubeCL's abstractions pay, which
  this spike deliberately avoided to keep the comparison clean.

## Reproduce

```sh
cargo nextest run -p onyxia-backend-cubecl --run-ignored=all   # differentials
cargo run --release -p onyxia-backend-cubecl --example forward-check
```
