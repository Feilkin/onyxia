// Unary square root: Y = sqrt(X)
//
// Element-wise square root of input tensor.
// Note: sqrt of negative values produces NaN.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    num_elements: u32,   // Total number of elements
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
    
    // Bounds check
    if (idx >= params.num_elements) {
        return;
    }
    
    // Perform square root
    output[idx] = sqrt(input[idx]);
}
