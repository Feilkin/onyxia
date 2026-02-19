// SimplifiedLayerNormalization Pass 1: Compute RMS per batch element
//
// RMS = sqrt(mean(x^2) + epsilon)

struct ImmediateConstants {
    batch_size: u32,
    norm_size: u32,
    epsilon: f32,
}

var<immediate> immediates: ImmediateConstants;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn compute_rms(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if batch_idx >= immediates.batch_size {
        return;
    }
    
    // Compute sum of squares for this batch element
    var sum_sq = 0.0;
    let base_offset = batch_idx * immediates.norm_size;
    
    for (var i = 0u; i < immediates.norm_size; i++) {
        let val = input[base_offset + i];
        sum_sq += val * val;
    }
    
    // Compute RMS
    let mean_sq = sum_sq / f32(immediates.norm_size);
    let rms = sqrt(mean_sq + immediates.epsilon);
    
    rms_output[batch_idx] = rms;
}
