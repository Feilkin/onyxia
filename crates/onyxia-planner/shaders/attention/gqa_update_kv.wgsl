// Shader to update KV cache with new key/value data for GroupQueryAttention
//
// In buffer-sharing mode with pre-allocated max_seq_len buffers:
// 1. Copy existing past_kv data to the output buffer (present_kv)
// 2. Write new K/V data at offset past_seq_len
//
// Inputs:
// - past_kv: [batch, kv_num_heads, max_seq_len, head_dim] - existing cache
// - current_kv: [batch, seq_len, kv_num_heads * head_dim] - new K or V data
//
// Output:
// - present_kv: [batch, kv_num_heads, max_seq_len, head_dim] - updated cache
//
// Note: This shader processes all elements of present_kv (full max_seq_len buffer)

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    past_seq_len: u32,
    max_seq_len: u32,
    kv_num_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> past_kv: array<f32>;
@group(0) @binding(1) var<storage, read> current_kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> present_kv: array<f32>;
var<immediate> constants: ImmediateConstants;

@compute @workgroup_size(#{WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_elements = constants.batch_size * constants.kv_num_heads * constants.max_seq_len * constants.head_dim;
    if idx >= total_elements { return; }
    
    // Decode 4D index for present_kv: [batch, kv_head, seq_pos, dim]
    // present_kv layout: [batch, kv_num_heads, max_seq_len, head_dim]
    let dim = idx % constants.head_dim;
    let seq_pos = (idx / constants.head_dim) % constants.max_seq_len;
    let kv_head = (idx / (constants.head_dim * constants.max_seq_len)) % constants.kv_num_heads;
    let batch = idx / (constants.head_dim * constants.max_seq_len * constants.kv_num_heads);
    
    if seq_pos < constants.past_seq_len {
        // Copy from past_kv: [batch, kv_num_heads, max_seq_len, head_dim]
        // (past_seq_len positions already contain valid data)
        present_kv[idx] = past_kv[idx];
    } else if seq_pos < constants.past_seq_len + constants.seq_len {
        // Write new data from current_kv: [batch, seq_len, kv_num_heads * head_dim]
        let current_seq_pos = seq_pos - constants.past_seq_len;
        let current_idx = batch * (constants.seq_len * constants.kv_num_heads * constants.head_dim)
                        + current_seq_pos * (constants.kv_num_heads * constants.head_dim)
                        + kv_head * constants.head_dim
                        + dim;
        present_kv[idx] = current_kv[current_idx];
    } else {
        // Beyond current valid sequence length - leave as-is or zero
        // (These positions will be masked out in attention anyway)
        present_kv[idx] = 0.0;
    }
}
