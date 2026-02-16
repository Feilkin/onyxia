// Gemma Rotary Position Embedding (Microsoft contrib op "GemmaRotaryEmbedding")
//
// Specialized RoPE implementation for Google Gemma models.
// Computes sin/cos from embedding input and applies rotation to q and k tensors.
//
// Formula:
//   sin_val = Sin(emb)
//   cos_val = Cos(emb)
//   q_embed = (q * cos_val) + (q_rot * sin_val)
//   k_embed = (k * cos_val) + (k_rot * sin_val)

@group(0) @binding(0) var<storage, read> emb: array<f32>;         // [batch, seq, dim] F32
@group(0) @binding(1) var<storage, read> q: array<f32>;           // [batch, heads, seq, dim] F16 (stored as f32)
@group(0) @binding(2) var<storage, read> q_rot: array<f32>;       // [batch, heads, seq, dim] F16 (stored as f32)
@group(0) @binding(3) var<storage, read> k: array<f32>;           // [batch, heads, seq, dim] F16 (stored as f32)
@group(0) @binding(4) var<storage, read> k_rot: array<f32>;       // [batch, heads, seq, dim] F16 (stored as f32)
@group(0) @binding(5) var<storage, read_write> q_embed: array<f32>; // [batch, heads, seq, dim] F16 (stored as f32)
@group(0) @binding(6) var<storage, read_write> k_embed: array<f32>; // [batch, heads, seq, dim] F16 (stored as f32)

// Immediate constants for tensor dimensions
struct ImmediateConstants {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    dim: u32,
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Total elements in q or k tensor
    let total_elements = params.batch_size * params.num_heads * params.seq_len * params.dim;
    let thread_id = global_id.x;
    
    if (thread_id >= total_elements) {
        return;
    }
    
    // Decode thread index to (batch, head, seq, dim_idx)
    let dim_idx = thread_id % params.dim;
    let seq = (thread_id / params.dim) % params.seq_len;
    let head = (thread_id / (params.dim * params.seq_len)) % params.num_heads;
    let batch = thread_id / (params.dim * params.seq_len * params.num_heads);
    
    // Index into emb tensor: [batch, seq, dim]
    let emb_idx = batch * params.seq_len * params.dim + seq * params.dim + dim_idx;
    
    // Compute sin and cos from embedding
    let emb_val = emb[emb_idx];
    let sin_val = sin(emb_val);
    let cos_val = cos(emb_val);
    
    // Index into q/k tensors: [batch, num_heads, seq, dim]
    let qk_idx = batch * params.num_heads * params.seq_len * params.dim +
                 head * params.seq_len * params.dim +
                 seq * params.dim +
                 dim_idx;
    
    // Load q, q_rot, k, k_rot values
    let q_val = q[qk_idx];
    let q_rot_val = q_rot[qk_idx];
    let k_val = k[qk_idx];
    let k_rot_val = k_rot[qk_idx];
    
    // Apply rotation: output = (input * cos) + (input_rot * sin)
    q_embed[qk_idx] = (q_val * cos_val) + (q_rot_val * sin_val);
    k_embed[qk_idx] = (k_val * cos_val) + (k_rot_val * sin_val);
}
