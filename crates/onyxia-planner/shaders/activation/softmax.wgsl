// Softmax activation: numerically stable softmax along a specified axis
//
// Implements: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
// where max is computed over the softmax axis for numerical stability

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Push constants for dimensions
struct ImmediateConstants {
    num_elements: u32,    // Total elements in tensor
    outer_size: u32,      // Number of independent softmax operations
    axis_dim: u32,        // Size of the softmax dimension
    axis: u32,            // Axis index (for reference)
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
    let outer_idx = global_id.x;
    
    if (outer_idx >= params.outer_size) {
        return;
    }
    
    // Each thread processes one softmax operation over axis_dim elements
    let base_idx = outer_idx * params.axis_dim;
    
    // Step 1: Find max value for numerical stability
    var max_val = input[base_idx];
    for (var i = 1u; i < params.axis_dim; i++) {
        max_val = max(max_val, input[base_idx + i]);
    }
    
    // Step 2: Compute exp(x - max) and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < params.axis_dim; i++) {
        let exp_val = exp(input[base_idx + i] - max_val);
        output[base_idx + i] = exp_val;  // Store temporarily
        sum_exp += exp_val;
    }
    
    // Step 3: Normalize by dividing by sum
    for (var i = 0u; i < params.axis_dim; i++) {
        output[base_idx + i] = output[base_idx + i] / sum_exp;
    }
}
