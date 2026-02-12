// Shader to apply causal mask and softmax to attention scores
//
// For each query position q_pos:
// - Applies causal mask: scores[*, *, q_pos, k_pos] = -inf if k_pos > past_seq + q_pos
// - Computes softmax over the k_pos dimension
//
// Input/Output:
// - scores: [batch, num_heads, seq_len, total_seq_len]
//
// Note: This implementation processes one (batch, head, q_pos) row at a time

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    past_seq_len: u32,
    total_seq_len: u32,
    num_heads: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
var<immediate> constants: ImmediateConstants;

// WGSL doesn't have f32(-inf), use a large negative number
const NEG_INF: f32 = -3.402823e+38;

@compute @workgroup_size(#{WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_idx = id.x;
    let total_rows = constants.batch_size * constants.num_heads * constants.seq_len;
    if row_idx >= total_rows { return; }
    
    // Decode 3D index: [batch, head, q_pos]
    let q_pos = row_idx % constants.seq_len;
    let head = (row_idx / constants.seq_len) % constants.num_heads;
    let batch = row_idx / (constants.seq_len * constants.num_heads);
    
    // Calculate the causal mask boundary for this query position
    // A query at position q_pos can attend to keys at positions 0..=(past_seq_len + q_pos)
    let valid_k_pos_max = constants.past_seq_len + q_pos;
    
    // Base offset for this row in the scores array
    let base_offset = batch * (constants.num_heads * constants.seq_len * constants.total_seq_len)
                    + head * (constants.seq_len * constants.total_seq_len)
                    + q_pos * constants.total_seq_len;
    
    // Step 1: Apply causal mask and find max value (for numerical stability)
    var max_val = NEG_INF;
    for (var k_pos = 0u; k_pos < constants.total_seq_len; k_pos = k_pos + 1u) {
        let idx = base_offset + k_pos;
        
        // Apply causal mask: prevent attending to future tokens
        if k_pos > valid_k_pos_max {
            scores[idx] = NEG_INF;
        } else {
            max_val = max(max_val, scores[idx]);
        }
    }
    
    // Step 2: Compute exp(x - max) and sum
    var sum_exp = 0.0;
    for (var k_pos = 0u; k_pos < constants.total_seq_len; k_pos = k_pos + 1u) {
        let idx = base_offset + k_pos;
        if scores[idx] > NEG_INF * 0.5 {  // Check if not masked out
            let exp_val = exp(scores[idx] - max_val);
            scores[idx] = exp_val;
            sum_exp += exp_val;
        } else {
            scores[idx] = 0.0;  // Masked positions become 0 after softmax
        }
    }
    
    // Step 3: Normalize by sum
    for (var k_pos = 0u; k_pos < constants.total_seq_len; k_pos = k_pos + 1u) {
        let idx = base_offset + k_pos;
        scores[idx] = scores[idx] / sum_exp;
    }
}
