// Two-pass softmax for attention scores
// Pass 1: Find max value along last dimension for numerical stability

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> max_vals: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_rows = params.batch * params.heads * params.seq_q;
    
    if idx >= total_rows {
        return;
    }
    
    // Find max along seq_k dimension
    var max_val = -1e10;
    let base_idx = idx * params.seq_k;
    
    for (var k = 0u; k < params.seq_k; k++) {
        let val = input[base_idx + k];
        max_val = max(max_val, val);
    }
    
    max_vals[idx] = max_val;
}
