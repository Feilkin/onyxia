// Batched matrix multiplication for attention: Q @ K^T
// Input: Q [batch, heads, seq_q, head_size]
// Input: K [batch, heads, seq_k, head_size]
// Output: scores [batch, heads, seq_q, seq_k]

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_size: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_head = gid.z;
    let q_idx = gid.y;
    let k_idx = gid.x;
    
    if q_idx >= params.seq_q || k_idx >= params.seq_k {
        return;
    }
    
    let batch = batch_head / params.heads;
    let head = batch_head % params.heads;
    
    // Compute dot product of Q[batch,head,q_idx,:] with K[batch,head,k_idx,:]
    var sum = 0.0;
    for (var i = 0u; i < params.head_size; i++) {
        let q_elem_idx = batch * params.heads * params.seq_q * params.head_size
                       + head * params.seq_q * params.head_size
                       + q_idx * params.head_size
                       + i;
        let k_elem_idx = batch * params.heads * params.seq_k * params.head_size
                       + head * params.seq_k * params.head_size
                       + k_idx * params.head_size
                       + i;
        sum += q[q_elem_idx] * k[k_elem_idx];
    }
    
    // Write to output[batch, head, q_idx, k_idx]
    let out_idx = batch * params.heads * params.seq_q * params.seq_k
                + head * params.seq_q * params.seq_k
                + q_idx * params.seq_k
                + k_idx;
    output[out_idx] = sum;
}
