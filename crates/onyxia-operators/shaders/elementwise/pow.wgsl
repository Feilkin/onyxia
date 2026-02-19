// Elementwise power: C = A ^ B
//
// Supports broadcasting for common patterns:
// - Scalar ^ Tensor
// - Vector ^ Vector
// - Matrix ^ Vector (broadcasted)
//
// Follows WGSL pow() semantics which match IEEE 754:
// - pow(0, 0) = 1
// - pow(x, 0) = 1 for any x
// - pow(0, y) = 0 for y > 0
// - pow(negative, non-integer) = NaN

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    size: u32,           // Total number of elements in output
    a_size: u32,         // Size of input A (1 if scalar broadcast)
    b_size: u32,         // Size of input B (1 if scalar broadcast)
    x_stride: u32,       // X dimension stride for 2D dispatch support
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Helper: compute linear index from 2D/3D dispatch
fn compute_linear_index(global_id: vec3<u32>, x_stride: u32) -> u32 {
    return global_id.x + global_id.y * x_stride + global_id.z * x_stride * 65535u;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = compute_linear_index(global_id, params.x_stride);
    
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
    
    let base = input_a[a_idx];
    let exponent = input_b[b_idx];
    
    // Special case: 0^0 = 1 (common convention in ML, different from IEEE 754)
    if (base == 0.0 && exponent == 0.0) {
        output[idx] = 1.0;
    } else {
        // Perform power operation
        output[idx] = pow(base, exponent);
    }
}
