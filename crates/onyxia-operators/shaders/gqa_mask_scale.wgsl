// Apply causal mask and scale to attention scores

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    past_seq_len: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.heads * params.seq_q * params.seq_k;
    if idx >= total {
        return;
    }
    
    // Decompose: [batch][heads][seq_q][seq_k]
    let batch = idx / (params.heads * params.seq_q * params.seq_k);
    let remainder = idx % (params.heads * params.seq_q * params.seq_k);
    let head = remainder / (params.seq_q * params.seq_k);
    let remainder2 = remainder % (params.seq_q * params.seq_k);
    let q_pos = remainder2 / params.seq_k;
    let k_pos = remainder2 % params.seq_k;
    
    var val = scores[idx] * params.scale;
    
    // Causal mask with KV-cache offset:
    // effective_query_pos = past_seq_len + q_pos
    // query can only attend to key positions <= effective_query_pos
    let effective_q_pos = params.past_seq_len + q_pos;
    if k_pos > effective_q_pos {
        val = -1e10;  // Large negative value → softmax ≈ 0
    }
    
    scores[idx] = val;
}
