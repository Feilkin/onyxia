// Reshape Q/K/V: [batch, seq, heads*size] → [batch, heads, seq, size]

struct Params {
    batch: u32,
    seq: u32,
    heads: u32,
    head_size: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.seq * params.heads * params.head_size;
    if idx >= total {
        return;
    }
    
    // Input: [batch][seq][heads * head_size]
    // Output: [batch][heads][seq][head_size]
    
    let batch = idx / (params.heads * params.seq * params.head_size);
    let remainder = idx % (params.heads * params.seq * params.head_size);
    let head = remainder / (params.seq * params.head_size);
    let remainder2 = remainder % (params.seq * params.head_size);
    let seq = remainder2 / params.head_size;
    let pos = remainder2 % params.head_size;
    
    // Calculate input index: [batch][seq][head * head_size + pos]
    let in_idx = batch * params.seq * (params.heads * params.head_size)
               + seq * (params.heads * params.head_size)
               + head * params.head_size
               + pos;
    
    output[idx] = input[in_idx];
}
