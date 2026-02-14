// Shader to concatenate past KV cache with current K/V for GroupQueryAttention
//
// This shader builds the full key or value tensor by:
// 1. Copying past_kv [batch, kv_heads, past_seq, head_dim] → present_kv[:, :, :past_seq, :]  
// 2. Reshaping and copying current_kv [batch, seq, kv_heads*head_dim] → present_kv[:, :, past_seq:, :]
//
// Output: present_kv [batch, kv_heads, total_seq, head_dim]

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    past_seq_len: u32,
    total_seq_len: u32,
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
    let total_elements = constants.batch_size * constants.kv_num_heads * constants.total_seq_len * constants.head_dim;
    if idx >= total_elements { return; }
    
    // Decode 4D index: [batch, kv_head, seq_pos, dim]
    let dim = idx % constants.head_dim;
    let seq_pos = (idx / constants.head_dim) % constants.total_seq_len;
    let kv_head = (idx / (constants.head_dim * constants.total_seq_len)) % constants.kv_num_heads;
    let batch = idx / (constants.head_dim * constants.total_seq_len * constants.kv_num_heads);
    
    if seq_pos < constants.past_seq_len {
        // Copy from past_kv [batch, kv_heads, past_seq, head_dim]
        let past_idx = batch * (constants.kv_num_heads * constants.past_seq_len * constants.head_dim)
                     + kv_head * (constants.past_seq_len * constants.head_dim)
                     + seq_pos * constants.head_dim
                     + dim;
        present_kv[idx] = past_kv[past_idx];
    } else {
        // Copy from current_kv [batch, seq, kv_heads * head_dim]
        // seq_pos in present_kv maps to (seq_pos - past_seq_len) in current_kv
        let current_seq_pos = seq_pos - constants.past_seq_len;
        let current_idx = batch * (constants.seq_len * constants.kv_num_heads * constants.head_dim)
                        + current_seq_pos * (constants.kv_num_heads * constants.head_dim)
                        + kv_head * constants.head_dim
                        + dim;
        present_kv[idx] = current_kv[current_idx];
    }
}
