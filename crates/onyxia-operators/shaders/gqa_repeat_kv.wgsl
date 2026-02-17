// Repeat KV heads to match Q heads for grouped attention
// Input: [batch, kv_heads, seq, size]
// Output: [batch, q_heads, seq, size] where q_heads = kv_heads * group_size

struct Params {
    batch: u32,
    kv_heads: u32,
    q_heads: u32,
    seq: u32,
    head_size: u32,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.q_heads * params.seq * params.head_size;
    if idx >= total {
        return;
    }
    
    // Output index: [batch][q_head][seq][pos]
    let batch = idx / (params.q_heads * params.seq * params.head_size);
    let remainder = idx % (params.q_heads * params.seq * params.head_size);
    let q_head = remainder / (params.seq * params.head_size);
    let remainder2 = remainder % (params.seq * params.head_size);
    let seq = remainder2 / params.head_size;
    let pos = remainder2 % params.head_size;
    
    // Map q_head to corresponding kv_head (each kv_head used by group_size q_heads)
    let group_size = params.q_heads / params.kv_heads;
    let kv_head = q_head / group_size;
    
    // Read from input
    let in_idx = batch * params.kv_heads * params.seq * params.head_size
               + kv_head * params.seq * params.head_size
               + seq * params.head_size
               + pos;
    
    output[idx] = input[in_idx];
}
