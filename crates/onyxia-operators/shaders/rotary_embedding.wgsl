// Rotary Position Embeddings (RoPE)
//
// Applies rotation to pairs of dimensions based on position.
// Non-interleaved mode: First half rotates with second half
// Interleaved mode: Adjacent pairs (0,1), (2,3), ...

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    hidden_size: u32,
    rotary_dim: u32,
    interleaved: u32,
}

@group(0) @binding(0) var<storage, read> immediates: ImmediateConstants;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> position_ids: array<u32>;
@group(0) @binding(3) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(4) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = immediates.batch_size * immediates.seq_len * immediates.hidden_size;
    
    if idx >= total {
        return;
    }
    
    // Decompose index: [batch, seq, hidden]
    let batch = idx / (immediates.seq_len * immediates.hidden_size);
    let remainder = idx % (immediates.seq_len * immediates.hidden_size);
    let seq = remainder / immediates.hidden_size;
    let hidden = remainder % immediates.hidden_size;
    
    // Get position for this element
    let pos_idx = batch * immediates.seq_len + seq;
    let position = position_ids[pos_idx];
    
    // If outside rotary range, copy input
    if hidden >= immediates.rotary_dim {
        output[idx] = input[idx];
        return;
    }
    
    if immediates.interleaved != 0u {
        // Interleaved mode: pairs (0,1), (2,3), (4,5), ...
        let pair_idx = hidden / 2u;
        let is_first = (hidden % 2u) == 0u;
        
        let cos_val = cos_cache[position * (immediates.rotary_dim / 2u) + pair_idx];
        let sin_val = sin_cache[position * (immediates.rotary_dim / 2u) + pair_idx];
        
        let partner_offset = select(-1i, 1i, is_first);
        let partner_idx = i32(idx) + partner_offset;
        
        let x_i = input[idx];
        let x_j = input[partner_idx];
        
        output[idx] = select(
            x_i * sin_val + x_j * cos_val,  // Second element of pair
            x_i * cos_val - x_j * sin_val,  // First element of pair
            is_first
        );
    } else {
        // Non-interleaved mode: first half with second half
        let half_dim = immediates.rotary_dim / 2u;
        
        if hidden < half_dim {
            // First half
            let cos_val = cos_cache[position * half_dim + hidden];
            let sin_val = sin_cache[position * half_dim + hidden];
            
            let partner_idx = idx + half_dim;
            let x_i = input[idx];
            let x_j = input[partner_idx];
            
            output[idx] = x_i * cos_val - x_j * sin_val;
        } else if hidden < immediates.rotary_dim {
            // Second half
            let rot_idx = hidden - half_dim;
            let cos_val = cos_cache[position * half_dim + rot_idx];
            let sin_val = sin_cache[position * half_dim + rot_idx];
            
            let partner_idx = idx - half_dim;
            let x_i = input[partner_idx];
            let x_j = input[idx];
            
            output[idx] = x_i * sin_val + x_j * cos_val;
        }
    }
}
