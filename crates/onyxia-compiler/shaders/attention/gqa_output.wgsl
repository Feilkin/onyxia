// Shader to compute attention output for GroupQueryAttention
//
// Computes: output[b, h, q_pos, d] = sum_k(attn_weights[b,h,q_pos,k] * V[b,kv_h,k,d])
// Where kv_h = h / group_size (multiple query heads share same KV head)
//
// Inputs:
// - attn_weights: [batch, num_heads, seq_len, total_seq_len] (after softmax)
// - present_value: [batch, kv_num_heads, total_seq_len, head_dim]
//
// Output:
// - output: [batch, seq_len, num_heads * head_dim]

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    total_seq_len: u32,
    num_heads: u32,
    kv_num_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> attn_weights: array<f32>;
@group(0) @binding(1) var<storage, read> present_value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<immediate> constants: ImmediateConstants;

@compute @workgroup_size(#{WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_elements = constants.batch_size * constants.num_heads * constants.seq_len * constants.head_dim;
    if idx >= total_elements { return; }
    
    // Decode 4D index: [batch, head, q_pos, dim]
    let dim = idx % constants.head_dim;
    let q_pos = (idx / constants.head_dim) % constants.seq_len;
    let head = (idx / (constants.head_dim * constants.seq_len)) % constants.num_heads;
    let batch = idx / (constants.head_dim * constants.seq_len * constants.num_heads);
    
    // Map query head to corresponding KV head (GQA grouping)
    let group_size = constants.num_heads / constants.kv_num_heads;
    let kv_head = head / group_size;
    
    // Compute weighted sum: sum_k(attn_weights[batch,head,q_pos,k] * V[batch,kv_head,k,dim])
    var weighted_sum = 0.0;
    for (var k_pos = 0u; k_pos < constants.total_seq_len; k_pos = k_pos + 1u) {
        // Attention weight index: [batch, num_heads, seq_len, total_seq_len]
        let attn_idx = batch * (constants.num_heads * constants.seq_len * constants.total_seq_len)
                     + head * (constants.seq_len * constants.total_seq_len)
                     + q_pos * constants.total_seq_len
                     + k_pos;
        
        // Value index: [batch, kv_num_heads, total_seq_len, head_dim]
        let v_idx = batch * (constants.kv_num_heads * constants.total_seq_len * constants.head_dim)
                  + kv_head * (constants.total_seq_len * constants.head_dim)
                  + k_pos * constants.head_dim
                  + dim;
        
        weighted_sum += attn_weights[attn_idx] * present_value[v_idx];
    }
    
    // Write to output: [batch, seq_len, num_heads * head_dim]
    let out_idx = batch * (constants.seq_len * constants.num_heads * constants.head_dim)
                + q_pos * (constants.num_heads * constants.head_dim)
                + head * constants.head_dim
                + dim;
    output[out_idx] = weighted_sum;
}
