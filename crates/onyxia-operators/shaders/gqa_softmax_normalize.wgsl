// Softmax pass 2 and 3: Compute exp and normalize
// Computes: exp(x - max) / sum(exp(x - max))

struct Params {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@group(0) @binding(2) var<storage, read> max_vals: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_idx = gid.x;
    let total_rows = params.batch * params.heads * params.seq_q;
    
    if row_idx >= total_rows {
        return;
    }
    
    let max_val = max_vals[row_idx];
    let base_idx = row_idx * params.seq_k;
    
    // Compute exp(x - max) and sum
    var sum = 0.0;
    for (var k = 0u; k < params.seq_k; k++) {
        let idx = base_idx + k;
        let exp_val = exp(values[idx] - max_val);
        values[idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize by sum
    for (var k = 0u; k < params.seq_k; k++) {
        let idx = base_idx + k;
        values[idx] /= sum;
    }
}
