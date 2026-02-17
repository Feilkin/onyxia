// Elementwise maximum: C = max(A, B)
//
// Supports broadcasting for common patterns:
// - Scalar + Tensor
// - Vector + Vector
// - Matrix + Vector (broadcasted)

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Push constants for shape information
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
    // - If size == 1, broadcast to all elements (scalar)
    // - If size == output size, use direct indexing
    // - Otherwise, use modulo for cyclic broadcasting (handles vector broadcasting)
    let a_idx = select(idx % params.a_size, 0u, params.a_size == 1u);
    let b_idx = select(idx % params.b_size, 0u, params.b_size == 1u);
    
    // Perform element-wise maximum
    output[idx] = max(input_a[a_idx], input_b[b_idx]);
}
