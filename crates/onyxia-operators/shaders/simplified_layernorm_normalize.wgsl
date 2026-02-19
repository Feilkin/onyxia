// SimplifiedLayerNormalization Pass 2: Apply normalization
//
// output[i] = input[i] / rms[batch] * scale[i % norm_size]

struct ImmediateConstants {
    batch_size: u32,
    norm_size: u32,
}

var<immediate> immediates: ImmediateConstants;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> rms: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn apply_normalization(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = immediates.batch_size * immediates.norm_size;
    
    if idx >= total_elements {
        return;
    }
    
    let batch_idx = idx / immediates.norm_size;
    let elem_idx = idx % immediates.norm_size;
    
    // Normalize and scale
    let rms_val = rms[batch_idx];
    let scale_val = scale[elem_idx];
    
    output[idx] = (input[idx] / rms_val) * scale_val;
}
