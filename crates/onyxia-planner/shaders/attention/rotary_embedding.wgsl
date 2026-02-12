// Rotary Position Embedding (RoPE) for transformer attention
//
// Applies rotation to pairs of embedding dimensions based on position.
// This is the Microsoft ONNX Runtime contrib op "RotaryEmbedding".
//
// For each (batch, seq, head) position, rotates pairs of elements:
//   x_rotated[2i]   = x[2i]   * cos(θ_i) - x[2i+1] * sin(θ_i)
//   x_rotated[2i+1] = x[2i+1] * cos(θ_i) + x[2i]   * sin(θ_i)
//
// The cos and sin values are precomputed and provided as cache tensors.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> position_ids: array<u32>;  // I64 as u32 pairs
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(3) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Immediate constants for tensor dimensions
struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    interleaved: u32,  // 0 = split-half (default in Gemma), 1 = interleaved pairs
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_pairs = params.batch_size * params.seq_len * params.num_heads * (params.head_dim / 2u);
    let thread_id = global_id.x;
    
    if (thread_id >= total_pairs) {
        return;
    }
    
    // Decode thread index to (batch, seq, head, pair_idx)
    let half_head_dim = params.head_dim / 2u;
    let pair_idx = thread_id % half_head_dim;
    let head = (thread_id / half_head_dim) % params.num_heads;
    let seq = (thread_id / (half_head_dim * params.num_heads)) % params.seq_len;
    let batch = thread_id / (half_head_dim * params.num_heads * params.seq_len);
    
    // Fetch position from position_ids (I64, but we only need low u32)
    let batch_seq_idx = batch * params.seq_len + seq;
    let pos_low = position_ids[batch_seq_idx * 2u];  // Low u32 of I64 pair
    let pos = pos_low;  // Safe for positions < 2^31
    
    // Fetch cos and sin values from cache
    // cos_cache and sin_cache are [max_seq, head_dim/2]
    let cache_idx = pos * half_head_dim + pair_idx;
    let cos_val = cos_cache[cache_idx];
    let sin_val = sin_cache[cache_idx];
    
    // Calculate input indices for the pair
    // Input shape: [batch, seq, num_heads, head_dim]
    let base_idx = batch * params.seq_len * params.num_heads * params.head_dim +
                   seq * params.num_heads * params.head_dim +
                   head * params.head_dim;
    
    var x0_idx: u32;
    var x1_idx: u32;
    
    if (params.interleaved == 1u) {
        // Interleaved: [x0, y0, x1, y1, ...]
        x0_idx = base_idx + pair_idx * 2u;
        x1_idx = base_idx + pair_idx * 2u + 1u;
    } else {
        // Split-half: [x0, x1, ..., y0, y1, ...]
        x0_idx = base_idx + pair_idx;
        x1_idx = base_idx + half_head_dim + pair_idx;
    }
    
    // Load input values
    let x0 = input[x0_idx];
    let x1 = input[x1_idx];
    
    // Apply rotation
    // x_rotated[0] = x0 * cos - x1 * sin
    // x_rotated[1] = x1 * cos + x0 * sin
    output[x0_idx] = x0 * cos_val - x1 * sin_val;
    output[x1_idx] = x1 * cos_val + x0 * sin_val;
}
