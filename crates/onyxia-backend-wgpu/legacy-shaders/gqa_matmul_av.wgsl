// Batched matrix multiplication: attention @ V
// Input: attn [batch, heads, seq_q, seq_k]
// Input: V [batch, heads, seq_k, head_size]
// Output: [batch, heads, seq_q, head_size]

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_size: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_head = gid.z;
    let q_idx = gid.y;
    let h_idx = gid.x;
    
    if q_idx >= params.seq_q || h_idx >= params.head_size {
        return;
    }
    
    let batch = batch_head / params.heads;
    let head = batch_head % params.heads;
    
    // Compute weighted sum: sum over k of attn[batch,head,q_idx,k] * V[batch,head,k,h_idx]
    var sum = 0.0;
    for (var k = 0u; k < params.seq_k; k++) {
        let attn_elem_idx = batch * params.heads * params.seq_q * params.seq_k
                          + head * params.seq_q * params.seq_k
                          + q_idx * params.seq_k
                          + k;
        let v_elem_idx = batch * params.heads * params.seq_k * params.head_size
                       + head * params.seq_k * params.head_size
                       + k * params.head_size
                       + h_idx;
        sum += attn[attn_elem_idx] * v[v_elem_idx];
    }
    
    // Write to output[batch, head, q_idx, h_idx]
    let out_idx = batch * params.heads * params.seq_q * params.head_size
                + head * params.seq_q * params.head_size
                + q_idx * params.head_size
                + h_idx;
    output[out_idx] = sum;
}
