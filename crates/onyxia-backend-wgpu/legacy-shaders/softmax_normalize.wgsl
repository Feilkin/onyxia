// Softmax Pass 3: Normalize by dividing by exp sum
//
// Algorithm:
// - Each thread handles one or more output elements
// - output[i] = exp(input[i] - max) / exp_sum

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> max_values: array<f32>;
@group(0) @binding(2) var<storage, read> exp_sums: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    outer_size: u32,     // Product of dimensions before axis
    axis_size: u32,      // Size of the axis dimension
    inner_size: u32,     // Product of dimensions after axis
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
    let idx = global_id.x;
    let total_size = params.outer_size * params.axis_size * params.inner_size;
    
    if (idx >= total_size) {
        return;
    }
    
    // Decompose flat index into (outer, axis, inner) coordinates
    let outer = idx / (params.axis_size * params.inner_size);
    let inner = idx % params.inner_size;
    
    let max_val = max_values[outer * params.inner_size + inner];
    let exp_sum = exp_sums[outer * params.inner_size + inner];
    
    output[idx] = exp(input[idx] - max_val) / exp_sum;
}
