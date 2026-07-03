# Legacy WGSL shaders (reference only)

Hand-written kernels from an earlier version of this backend. **Nothing
compiles these** — they use different binding/immediates conventions and
won't run against the current session as-is.

They are kept because they encode tested, correct math for the fused
composite kernels still to be written in `src/fused.rs`:

- `gqa_*.wgsl` — the nine-kernel fused GroupQueryAttention path
- `rotary_embedding.wgsl` — fused RoPE
- `matmul_nbits.wgsl` — fused q4 dequant-matmul
- `matmul.wgsl` — a starting point for a tiled MatMul primitive kernel

Port a kernel → delete its file. Delete the whole directory once the fused
set is done.
