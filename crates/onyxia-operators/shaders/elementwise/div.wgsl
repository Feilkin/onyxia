// Elementwise division: C = A / B
//
// Supports broadcasting for common patterns:
// - Scalar / Tensor
// - Vector / Vector
// - Matrix / Vector (broadcasted)
//
// Division by zero yields Inf or NaN according to IEEE 754.

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    size: u32,           // Total number of elements in output
    a_size: u32,         // Size of input A (1 if scalar broadcast)
    b_size: u32,         // Size of input B (1 if scalar broadcast)
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
    // - If a_size == 1, broadcast A to all elements
    // - If b_size == 1, broadcast B to all elements
    // - Otherwise, use regular indexing
    let a_idx = select(idx, 0u, params.a_size == 1u);
    let b_idx = select(idx, 0u, params.b_size == 1u);
    
    // Perform division
    output[idx] = input_a[a_idx] / input_b[b_idx];
}
