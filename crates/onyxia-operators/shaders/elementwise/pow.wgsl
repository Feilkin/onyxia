// Elementwise power operation: Z = X ^ Y
//
// Supports broadcasting for common patterns:
// - Scalar ^ Tensor
// - Vector ^ Vector
// - Matrix ^ Vector (broadcasted)
// - Tensor ^ Scalar

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    size: u32,           // Total number of elements in output
    x_size: u32,         // Size of input X (1 if scalar broadcast)
    y_size: u32,         // Size of input Y (1 if scalar broadcast)
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
    if (idx >= params.size) {
        return;
    }
    
    // Handle broadcasting:
    // - If x_size == 1, broadcast X to all elements
    // - If y_size == 1, broadcast Y to all elements
    // - Otherwise, use regular indexing
    let x_idx = select(idx, 0u, params.x_size == 1u);
    let y_idx = select(idx, 0u, params.y_size == 1u);
    
    let x = input_x[x_idx];
    let y = input_y[y_idx];
    
    // Handle special case: x^0 = 1 (except 0^0 which is mathematically undefined)
    // WGSL pow() is defined as exp2(y * log2(x)), so log2(x) for x<=0 is undefined
    // Use select() to avoid branching: returns 1.0 when y==0, otherwise pow(x,y)
    output[idx] = select(pow(x, y), 1.0, y == 0.0);
}
