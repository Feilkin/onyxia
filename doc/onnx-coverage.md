# ONNX operator coverage

*Generated 2026-09-05 against the vendored spec (`doc/onnx-spec/Operators.md`,
opset 26) and the node tests shipped in the `onnx` 1.22.0 Python package
(`onnx/backend/test/data/node`, Apache-2.0).*

## Summary

| | |
|---|---|
| Operators in the `ai.onnx` default domain (opset 26, incl. 2 deprecated aliases) | 199 |
| Outside the tensor model (sequences/optionals, strings, control flow, random sampling, image decoding) | 27 |
| **Core tensor operators** | **172** |
| Core operators with a lowering rule | **163** |
| Core operators not lowered | 9 (listed below) |
| Primitives in the IR | **16** (unchanged) |
| Node tests in the suite | 1765 |
| Passing on the reference backend | **1275** (0 failures; 490 out of scope) |
| Passing on the wgpu backend | **1273** (native or packed f16 / i64 modes alike) |

"Core" here means every operator that maps tensors to tensors with a shape
that is a function of the input shapes. The 27 excluded operators either
work on non-tensor types (sequences, optionals, strings), take subgraphs
(`If`, `Loop`, `Scan`), draw random numbers, or decode images. None of them
is on the path of an exported inference model in the transformer or vision
families; all of them would need a change to the IR's type or execution
model rather than a new primitive.

## The primitive count

Every one of the 163 lowered operators is expressed over the same 16
primitives that ran Gemma before this work:

```
Unary(kind)  Binary(kind)  Compare(kind)  Select  Cast
MatMul  Reduce(kind)  Reshape  Transpose  Broadcast  Concat  Slice
Gather  Scatter(reduction)  Iota  DimValues  Dequantize
```

What did have to grow is the *kind* tables inside the elementwise
primitives — the part of the contract that is one expression per entry in a
kernel template, not a new kernel:

| | before | after | added |
|---|---|---|---|
| `UnaryOp` kinds | 13 | 25 | Round, Sign, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, BitNot |
| `BinaryOp` kinds | 10 | 15 | BitAnd, BitOr, BitXor, Shl, Shr |
| `Scatter` | overwrite only | `reduction` ∈ {none, add, mul, max, min} | needed by ScatterND/ScatterElements reductions and Col2Im |

A backend that already had the 16 kernel templates gained the new operators
by adding table entries (about 40 lines in the wgpu backend). The CubeCL
backend, which has not been touched, rejects the new kinds with a clear
error and is otherwise unaffected.

### Where the decompositions come from

The interesting operators do not need anything new because the IR already
has the two ingredients most of them reduce to: an integer ramp (`Iota`) to
compute indices with, and `Gather`/`MatMul` to apply them.

| Operator family | Decomposition |
|---|---|
| Conv, ConvTranspose, ConvInteger, MaxPool, AveragePool, LpPool, MaxUnpool | im2col: a constant `[out, kernel]` index table built at lowering, one `Gather`, then one batched `MatMul` (conv) or one `Reduce` over the window axis (pool). ConvTranspose is the direct convolution of the stride-dilated input with the flipped kernel. Spatial dims must be static. |
| Resize, Upsample | a dense `[out, in]` interpolation matrix per resized axis (nearest / linear / cubic / antialias / every coordinate-transformation mode, computed at lowering), applied with `MatMul`. |
| DFT, STFT | cos/sin matrices and `MatMul`; STFT frames the signal with a `Gather` table first. |
| GridSample, AffineGrid | runtime index arithmetic on the grid (`Floor`, `Cast`, clamp/reflect with `Select`), then linear `Gather`s per tap and a weighted sum. |
| GatherElements, GatherND, ScatterElements, TensorScatter, ReverseSequence | a linear index computed from `Iota` coordinates and the given indices, then one `Gather` / `Scatter` on the flattened data. |
| Pad (reflect/edge/wrap) | a constant index vector per axis and `Gather`; constant mode is `Concat` with `Broadcast` blocks. |
| CumSum | `MatMul` with a triangular 0/1 mask (works for symbolic lengths). |
| CumProd | log-depth doubling with `Slice`/`Concat`/`Mul` (static length). |
| TopK | rank counting: `rank_i = #{j: x_j ≻ x_i}` via a broadcast `Compare` + `Reduce`, then a one-hot `MatMul` to select values and indices. Quadratic in the axis length. |
| ArgMax/ArgMin, Hardmax | `Reduce` max, `Compare`, `Select` an `Iota`, `Reduce` min/max. |
| LayerNormalization, RMSNormalization, GroupNormalization, InstanceNormalization, BatchNormalization, MVN, LpNormalization, LRN | `Reduce` mean/variance and elementwise arithmetic; LRN sums shifted slices of a padded channel axis. |
| Attention (opset 23/24), RotaryEmbedding | `MatMul` / softmax / `Select` masks; matches the onnx reference implementation (softcap before the bias, causal mask aligned top-left, all four `qk_matmul_output_mode`s). |
| RNN, GRU, LSTM | unrolled over a static sequence length; `sequence_lens` masks with `Select`. |
| Einsum | operands broadcast to the union of labels, multiplied, then `Reduce`; repeated labels become a diagonal `Gather`. |
| Quantize/DequantizeLinear (8-bit), DynamicQuantizeLinear, MatMulInteger, QLinearMatMul | f32 / i32 arithmetic with explicit `Cast`s; per-axis and blocked scales via `Reshape` + `Broadcast`. |
| NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss | log-softmax, a linear `Gather` of the target class, weight `Gather`, `Reduce`. |
| Hann/Hamming/Blackman windows, MelWeightMatrix, EyeLike, OneHot, Range, Tile, Split, Gemm, … | constants computed at lowering, or reshape/broadcast/slice plumbing. |

`LayerNormalization`, `RMSNormalization`, and `Attention` are emitted as
composites (their decompositions live in `onyxia-ir/src/decomp.rs`) so a
backend can fuse them; everything else lowers directly to primitives in
`onyxia-lower/src/rules/`.

### What a 17th primitive would buy

Nothing in the core set *requires* one. Three places where one would be an
honest performance decision rather than a correctness need:

- **Sort / TopK.** The rank-counting decomposition is O(n²) in the sorted
  axis. Top-k over a 262 144-entry vocabulary is not something you want to
  run that way; a `Sort` primitive (or a `TopK` composite with a fused
  kernel) is the first addition a sampling-heavy backend would make.
- **Convolution.** The im2col path materializes a `kernel_size ×` larger
  tensor. It is the right *specification*; a fused direct-conv kernel is a
  composite kernel, not a new primitive.
- **Scatter with reductions on the GPU.** Implemented as a compare-and-
  swap loop on the 32-bit output word (float atomics are not in core
  WebGPU), which also serves packed 8-bit and f16 outputs. Correct, but
  contended updates serialize.

Two families would need changes beyond primitives: **data-dependent output
shapes** (NonZero, Unique, Compress, NonMaxSuppression — the IR's shapes are
functions of input *shapes*, evaluated once per run) and **control flow**
(`If`/`Loop`/`Scan` carry subgraphs). **Random sampling** would need a
stateful RNG primitive; it is the one candidate for a genuinely new
primitive in the default domain.

## Method

`crates/onyxia-conformance` discovers every `test_*` directory of the node
test suite, loads `model.onnx` and the `input_*.pb` / `output_*.pb`
`TensorProto`s, lowers the model with the standard registry, runs it on a
backend, and compares outputs at rtol 1e-3 / atol 1e-5 (f32) or 1e-2 / 1e-3
(f16), exact for integers and bools.

Two details worth knowing when reading the numbers:

- **Bound parameters.** The node tests feed operator *parameters* (axes,
  shapes, slice bounds, pad amounts, `k`) as runtime input tensors. Exported
  models carry them as initializers, and Onyxia resolves them at lowering.
  When lowering fails for that reason the harness retries with every small
  (≤ 64 elements, rank ≤ 1) input bound as a constant; such passes are
  counted separately (268 of the 1275 on the reference backend) and marked
  `PASS*`.
- **`_expanded` tests** run the ONNX *function body* of an operator
  (LayerNormalization, Attention, RMSNormalization, CenterCropPad, the
  windows, the Reduce* family …) instead of the operator. Those bodies do
  shape arithmetic with `Shape`/`Size`/`Range`/`Where`/`Mod`/… and all of it
  now folds in the lowering's shape domain, so both forms pass.

Skips (490) are tests the harness never attempts or cannot represent:
float64, bfloat16, int16/uint16, uint64, int2/int4/uint2/uint4, float8 and
float4 dtypes (about 270 tests); the 27 out-of-scope operators; the 4
data-dependent-shape operators; opset-27 operators newer than the vendored
spec (`LinearAttention`, `CausalConvWithState`, `FlexAttention`, `BitCast`);
training-mode tests.

Run it:

```bash
just fetch-onnx-tests                                # once: onnx==1.22.0 into .venv
just conformance                                     # matrix on the reference backend
cargo run --release -p onyxia-conformance --features wgpu -- --backend wgpu --quiet --ops
cargo run --release -p onyxia-conformance -- --failures gru   # details for a subset
cargo test -p onyxia-conformance                     # regression gate (expected-pass-ref.txt)
```

## wgpu backend

The wgpu backend passes 1273 of the 1275 in-scope tests. Its physical
layouts (`onyxia-backend-wgpu/src/layout.rs`) are chosen per adapter:

| logical dtype | with the feature | without |
|---|---|---|
| f16 | native `f16` storage (`SHADER_F16`), computed in f32 | two per `u32` word, `pack2x16float` / `unpack2x16float` |
| int64 | native `i64` (`SHADER_INT64`) | narrowed to `i32`, range-checked at upload |
| uint8 / int8 | four per `u32` word, memory order (WebGPU has no 8-bit storage) | same |
| bool | one `u32` per element | same |

Every generated kernel runs one thread per output *word*, so packed lanes
are never written by two threads; the two fallback modes are exercised by
`ONYXIA_NO_F16=1` / `ONYXIA_NO_INT64=1` and give the same results. Device
bytes equal host bytes for every layout except the narrowed i64 and
bool, so 8-bit weights cost one byte each on the device.

The two remaining failures are `DynamicQuantizeLinear`, where one element
sits on a rounding tie that the GPU's `x / scale` (not correctly rounded
under Vulkan's relaxed float division) resolves differently from numpy.

## Per-operator results

`passed/attempted (+skipped)`; a skipped test is one the harness did not
attempt (unsupported dtype, out of scope). Operators outside the tensor
model are omitted.

| Operator | reference | wgpu | note |
|---|---|---|---|
| Abs | 1/1 | 1/1 |  |
| Acos | 2/2 | 2/2 |  |
| Acosh | 2/2 | 2/2 |  |
| Add | 5/5 (+3 skip) | 3/5 (+3 skip) |  |
| And | 8/8 | 8/8 |  |
| ArgMax | 16/16 | 16/16 |  |
| ArgMin | 16/16 | 16/16 |  |
| Asin | 2/2 | 2/2 |  |
| Asinh | 2/2 | 2/2 |  |
| Atan | 2/2 | 2/2 |  |
| Atanh | 2/2 | 2/2 |  |
| AveragePool | 20/20 | 20/20 |  |
| BatchNormalization | 4/4 | 4/4 |  |
| BitShift | 4/4 (+4 skip) | 2/4 (+4 skip) |  |
| BitwiseAnd | 2/2 (+2 skip) | 1/2 (+2 skip) |  |
| BitwiseNot | 2/2 (+1 skip) | 1/2 (+1 skip) |  |
| BitwiseOr | 3/3 (+1 skip) | 1/3 (+1 skip) |  |
| BitwiseXor | 2/2 (+2 skip) | 1/2 (+2 skip) |  |
| Cast | 2/2 (+58 skip) | 0/2 (+58 skip) |  |
| Ceil | 2/2 | 2/2 |  |
| Col2Im | 5/5 | 0/5 |  |
| Compress | 0 (skip 4) | 0 (skip 4) | data-dependent output shape |
| Concat | 12/12 | 12/12 |  |
| Constant | 1/1 | 1/1 |  |
| ConstantOfShape | 3/3 | 3/3 |  |
| Conv | 6/6 | 6/6 |  |
| ConvInteger | 2/2 | 0/2 |  |
| ConvTranspose | 11/11 | 10/11 |  |
| Cos | 2/2 | 2/2 |  |
| Cosh | 2/2 | 2/2 |  |
| CumProd | 2/2 (+7 skip) | 2/2 (+7 skip) |  |
| CumSum | 2/2 (+7 skip) | 2/2 (+7 skip) |  |
| DFT | 10/10 | 10/10 |  |
| DeformConv | 0 (skip 4) | 0 (skip 4) | not written (gather-based, like GridSample) |
| DepthToSpace | 2/2 | 2/2 |  |
| DequantizeLinear | 3/3 (+11 skip) | 0/3 (+11 skip) |  |
| Det | 0 (skip 2) | 0 (skip 2) | needs an LU / elimination kernel |
| Div | 7/7 (+3 skip) | 5/7 (+3 skip) |  |
| Dropout | 6/6 | 6/6 |  |
| Einsum | 0 (skip 6) | 0 (skip 6) |  |
| Equal | 5/5 (+5 skip) | 3/5 (+5 skip) |  |
| Erf | 1/1 | 1/1 |  |
| Exp | 2/2 | 2/2 |  |
| Expand | 2/2 | 2/2 |  |
| EyeLike | 2/2 (+1 skip) | 2/2 (+1 skip) |  |
| Flatten | 9/9 | 9/9 |  |
| Floor | 2/2 | 2/2 |  |
| GRU | 4/4 | 4/4 |  |
| Gather | 4/4 | 4/4 |  |
| GatherElements | 3/3 | 3/3 |  |
| GatherND | 3/3 | 3/3 |  |
| Gemm | 11/11 | 11/11 |  |
| GlobalAveragePool | 2/2 | 2/2 |  |
| GlobalLpPool | — | — | no node tests in the suite |
| GlobalMaxPool | 2/2 | 2/2 |  |
| Greater | 15/15 (+9 skip) | 9/15 (+9 skip) |  |
| GridSample | 18/18 | 18/18 |  |
| Hardmax | 7/7 | 7/7 |  |
| Identity | 1/1 (+2 skip) | 1/1 (+2 skip) |  |
| InstanceNormalization | 2/2 | 2/2 |  |
| IsInf | 4/4 | 3/4 |  |
| IsNaN | 2/2 | 1/2 |  |
| LRN | 2/2 | 2/2 |  |
| LSTM | 4/4 | 4/4 |  |
| Less | 15/15 (+9 skip) | 9/15 (+9 skip) |  |
| Log | 2/2 | 2/2 |  |
| LpNormalization | 6/6 | 6/6 |  |
| LpPool | 8/8 | 8/8 |  |
| MatMul | 7/7 | 6/7 |  |
| MatMulInteger | 1/1 | 0/1 |  |
| Max | 10/10 (+4 skip) | 7/10 (+4 skip) |  |
| MaxPool | 19/19 | 18/19 |  |
| MaxRoiPool | — | — | not written; the suite has no test for it |
| MaxUnpool | 2/2 | 2/2 |  |
| Mean | 3/3 | 3/3 |  |
| MelWeightMatrix | 1/1 | 1/1 |  |
| Min | 10/10 (+4 skip) | 7/10 (+4 skip) |  |
| Mod | 9/9 (+4 skip) | 6/9 (+4 skip) |  |
| Mul | 6/6 (+3 skip) | 4/6 (+3 skip) |  |
| Neg | 2/2 | 2/2 |  |
| NonMaxSuppression | 0 (skip 10) | 0 (skip 10) | data-dependent output shape |
| NonZero | 0 (skip 1) | 0 (skip 1) | data-dependent output shape |
| Not | 3/3 | 3/3 |  |
| OneHot | 4/4 | 4/4 |  |
| Or | 8/8 | 8/8 |  |
| Pad | 6/6 | 6/6 |  |
| Pow | 11/11 (+1 skip) | 7/11 (+1 skip) |  |
| QLinearConv | 0 (skip 1) | 0 (skip 1) | not written (dequantize → Conv → quantize) |
| QLinearMatMul | 8/8 | 0/8 |  |
| QuantizeLinear | 3/3 (+10 skip) | 0/3 (+10 skip) |  |
| RNN | 1/1 | 1/1 |  |
| Reciprocal | 2/2 | 2/2 |  |
| ReduceMax | 10/10 | 10/10 |  |
| ReduceMean | 8/8 | 8/8 |  |
| ReduceMin | 10/10 | 10/10 |  |
| ReduceProd | 9/9 | 9/9 |  |
| ReduceSum | 12/12 | 12/12 |  |
| Reshape | 10/10 | 10/10 |  |
| Resize | 39/39 | 39/39 |  |
| ReverseSequence | 2/2 | 2/2 |  |
| RoiAlign | 0 (skip 3) | 0 (skip 3) | not written (gather-based, like GridSample) |
| Round | 1/1 | 1/1 |  |
| STFT | 2/2 | 2/2 |  |
| Scatter | 2/2 | 2/2 | deprecated alias |
| ScatterElements | 7/7 | 3/7 |  |
| ScatterND | 5/5 | 1/5 |  |
| Shape | 11/11 | 11/11 |  |
| Sigmoid | 2/2 | 2/2 |  |
| Sign | 1/1 | 1/1 |  |
| Sin | 2/2 | 2/2 |  |
| Sinh | 2/2 | 2/2 |  |
| Size | 2/2 | 2/2 |  |
| Slice | 8/8 | 8/8 |  |
| SpaceToDepth | 2/2 | 2/2 |  |
| Split | 16/16 | 16/16 |  |
| Sqrt | 2/2 | 2/2 |  |
| Squeeze | 2/2 | 2/2 |  |
| Sub | 6/6 (+3 skip) | 4/6 (+3 skip) |  |
| Sum | 3/3 | 3/3 |  |
| Tan | 2/2 | 2/2 |  |
| Tanh | 2/2 | 2/2 |  |
| TensorScatter | 3/3 | 3/3 |  |
| Tile | 2/2 | 2/2 |  |
| TopK | 6/6 (+1 skip) | 6/6 (+1 skip) |  |
| Transpose | 7/7 | 7/7 |  |
| Trilu | 18/18 | 18/18 |  |
| Unique | 0 (skip 6) | 0 (skip 6) | data-dependent output shape |
| Unsqueeze | 7/7 | 7/7 |  |
| Upsample | 1/1 | 1/1 | deprecated alias |
| Where | 2/2 | 2/2 |  |
| Xor | 8/8 | 8/8 |  |
| AffineGrid | 4/4 (+4 skip) | 4/4 (+4 skip) |  |
| Attention | 151/151 (+25 skip) | 134/151 (+25 skip) |  |
| BlackmanWindow | 4/4 | 4/4 |  |
| CastLike | 4/4 (+108 skip) | 0/4 (+108 skip) |  |
| Celu | 2/2 | 2/2 |  |
| CenterCropPad | 12/12 | 12/12 |  |
| Clip | 24/24 | 18/24 |  |
| DynamicQuantizeLinear | 6/6 | 0/6 |  |
| Elu | 6/6 | 6/6 |  |
| Gelu | 8/8 | 8/8 |  |
| GreaterOrEqual | — | — | no node tests in the suite |
| GroupNormalization | 4/4 | 4/4 |  |
| HammingWindow | 4/4 | 4/4 |  |
| HannWindow | 4/4 | 4/4 |  |
| HardSigmoid | 6/6 | 6/6 |  |
| HardSwish | 2/2 | 2/2 |  |
| LayerNormalization | 57/57 | 57/57 |  |
| LeakyRelu | 6/6 | 6/6 |  |
| LessOrEqual | — | — | no node tests in the suite |
| LogSoftmax | 21/21 | 21/21 |  |
| MeanVarianceNormalization | 3/3 | 3/3 |  |
| Mish | 2/2 | 2/2 |  |
| NegativeLogLikelihoodLoss | 36/36 | 36/36 |  |
| PRelu | 4/4 | 4/4 |  |
| RMSNormalization | 38/38 | 38/38 |  |
| Range | 3/3 (+5 skip) | 2/3 (+5 skip) |  |
| ReduceL1 | 18/18 | 18/18 |  |
| ReduceL2 | 18/18 | 18/18 |  |
| ReduceLogSum | 10/10 | 10/10 |  |
| ReduceLogSumExp | 1/1 (+17 skip) | 1/1 (+17 skip) |  |
| ReduceSumSquare | 18/18 | 18/18 |  |
| Relu | 2/2 | 2/2 |  |
| RotaryEmbedding | 16/16 | 16/16 |  |
| Selu | 6/6 | 6/6 |  |
| Shrink | 4/4 | 4/4 |  |
| Softmax | 21/21 | 21/21 |  |
| SoftmaxCrossEntropyLoss | 68/68 | 68/68 |  |
| Softplus | 4/4 | 4/4 |  |
| Softsign | 4/4 | 4/4 |  |
| Swish | 2/2 | 2/2 |  |
| ThresholdedRelu | 6/6 | 6/6 |  |
