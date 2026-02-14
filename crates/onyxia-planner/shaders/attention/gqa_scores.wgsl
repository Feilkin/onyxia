// Shader to compute attention scores for GroupQueryAttention
//
// Computes: scores[b, h, q_pos, k_pos] = dot(Q[b,h,q_pos,:], K[b,kv_h,k_pos,:]) / scale
// Where kv_h = h / group_size (multiple query heads share same KV head)
//
// Inputs:
// - query: [batch, seq_len, num_heads * head_dim] → reshaped to [batch, num_heads, seq_len, head_dim]
// - present_key: [batch, kv_num_heads, total_seq_len, head_dim]
//
// Output:
// - scores: [batch, num_heads, seq_len, total_seq_len]

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    total_seq_len: u32,
    num_heads: u32,
    kv_num_heads: u32,
    head_dim: u32,
    // Scale factor (typically 1 / sqrt(head_dim))
    scale: f32,
    // Softcap value (0.0 = disabled)
    softcap: f32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> present_key: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
var<immediate> constants: ImmediateConstants;

@compute @workgroup_size(#{WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_elements = constants.batch_size * constants.num_heads * constants.seq_len * constants.total_seq_len;
    if idx >= total_elements { return; }
    
    // Decode 4D index: [batch, head, q_pos, k_pos]
    let k_pos = idx % constants.total_seq_len;
    let q_pos = (idx / constants.total_seq_len) % constants.seq_len;
    let head = (idx / (constants.total_seq_len * constants.seq_len)) % constants.num_heads;
    let batch = idx / (constants.total_seq_len * constants.seq_len * constants.num_heads);
    
    // Map query head to corresponding KV head (GQA grouping)
    let group_size = constants.num_heads / constants.kv_num_heads;
    let kv_head = head / group_size;
    
    // Compute dot product: Q[batch, head, q_pos, :] · K[batch, kv_head, k_pos, :]
    var dot_product = 0.0;
    for (var d = 0u; d < constants.head_dim; d = d + 1u) {
        // Query index: [batch, seq_len, num_heads * head_dim] → [batch, q_pos, head * head_dim + d]
        let q_idx = batch * (constants.seq_len * constants.num_heads * constants.head_dim)
                  + q_pos * (constants.num_heads * constants.head_dim)
                  + head * constants.head_dim
                  + d;
        
        // Key index: [batch, kv_num_heads, total_seq_len, head_dim]
        let k_idx = batch * (constants.kv_num_heads * constants.total_seq_len * constants.head_dim)
                  + kv_head * (constants.total_seq_len * constants.head_dim)
                  + k_pos * constants.head_dim
                  + d;
        
        dot_product += query[q_idx] * present_key[k_idx];
    }
    
    // Apply scaling
    var score = dot_product * constants.scale;
    
    // Apply softcap if enabled (softcap != 0.0)
    // Formula: score = softcap * tanh(score / softcap)
    if constants.softcap != 0.0 {
        score = constants.softcap * tanh(score / constants.softcap);
    }
    
    scores[idx] = score;
}
