// RMS Normalization (Root Mean Square Layer Normalization)
//
// RMSNorm(x) = x / RMS(x) * weight
// where RMS(x) = sqrt(mean(x²) + epsilon)
//
// This is used in modern LLMs like Llama, Gemma instead of LayerNorm.
// It's more efficient as it doesn't require computing mean and variance separately.
//
// Input shape: [batch, seq_len, hidden_dim]
// Weight shape: [hidden_dim]
// Output shape: [batch, seq_len, hidden_dim]
//
// Each workgroup processes one sequence position (normalizes across hidden_dim).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct ImmediateConstants {
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
    epsilon: f32,
}

var<immediate> params: ImmediateConstants;

// Workgroup size for reduction - 256 threads per workgroup
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Shared memory for reduction
var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let batch_seq_idx = workgroup_id.x;  // Each workgroup handles one [batch, seq] position
    let hidden_dim = params.hidden_dim;
    
    // Check if this workgroup is within bounds
    let total_positions = params.batch_size * params.seq_len;
    if (batch_seq_idx >= total_positions) {
        return;
    }
    
    // Base offset for this sequence position
    let base_idx = batch_seq_idx * hidden_dim;
    
    // Phase 1: Compute sum of squares (parallel reduction)
    var local_sum: f32 = 0.0;
    
    // Each thread processes multiple elements if hidden_dim > WG_SIZE
    var i = tid;
    loop {
        if (i >= hidden_dim) {
            break;
        }
        let val = input[base_idx + i];
        local_sum += val * val;
        i += WG_SIZE;
    }
    
    // Store to shared memory
    shared_sum[tid] = local_sum;
    workgroupBarrier();
    
    // Phase 2: Tree reduction to compute total sum
    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    
    // Phase 3: Compute RMS and normalize
    // Thread 0 has the total sum in shared_sum[0]
    let mean_square = shared_sum[0] / f32(hidden_dim);
    let rms = sqrt(mean_square + params.epsilon);
    let scale = 1.0 / rms;
    
    // Phase 4: Apply normalization and weight
    // Each thread processes multiple elements
    i = tid;
    loop {
        if (i >= hidden_dim) {
            break;
        }
        let val = input[base_idx + i];
        output[base_idx + i] = (val * scale) * weight[i];
        i += WG_SIZE;
    }
}
