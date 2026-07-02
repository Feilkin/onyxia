// KV Cache Concatenation for GroupQueryAttention
//
// Concatenates past KV cache with new KV along the sequence dimension.
// Both tensors are in BNSH format: [batch_size, num_heads, sequence_length, head_size]
//
// Inputs:
//   - past_kv: [batch, heads, past_seq, head_size]
//   - new_kv: [batch, heads, new_seq, head_size]
// Output:
//   - present_kv: [batch, heads, past_seq + new_seq, head_size]

@group(0) @binding(0) var<storage, read> past_kv: array<f32>;
@group(0) @binding(1) var<storage, read> new_kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> present_kv: array<f32>;

struct Params {
    batch_size: u32,
    num_heads: u32,
    past_seq_len: u32,
    new_seq_len: u32,
    head_size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.batch_size * params.num_heads * 
                     (params.past_seq_len + params.new_seq_len) * params.head_size;
    
    if idx >= total_size {
        return;
    }
    
    // Decode BNSH indices from flat index
    let head_size = params.head_size;
    let seq_len_total = params.past_seq_len + params.new_seq_len;
    
    let h_idx = idx % head_size;
    let s_idx = (idx / head_size) % seq_len_total;
    let n_idx = (idx / (head_size * seq_len_total)) % params.num_heads;
    let b_idx = idx / (head_size * seq_len_total * params.num_heads);
    
    // Copy from past or new depending on sequence position
    if s_idx < params.past_seq_len {
        // Copy from past cache
        let past_idx = b_idx * params.num_heads * params.past_seq_len * head_size +
                       n_idx * params.past_seq_len * head_size +
                       s_idx * head_size +
                       h_idx;
        present_kv[idx] = past_kv[past_idx];
    } else {
        // Copy from new cache
        let new_s_idx = s_idx - params.past_seq_len;
        let new_idx = b_idx * params.num_heads * params.new_seq_len * head_size +
                      n_idx * params.new_seq_len * head_size +
                      new_s_idx * head_size +
                      h_idx;
        present_kv[idx] = new_kv[new_idx];
    }
}
