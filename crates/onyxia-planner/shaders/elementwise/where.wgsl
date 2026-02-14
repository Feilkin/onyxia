// Elementwise Where: output = condition ? X : Y
//
// Supports broadcasting for common patterns:
// - Scalar condition + Tensors
// - Vector + Vector + Vector
// - Matrix + Vector (broadcasted)
//
// The condition input is expected to be represented as i32 or u32
// where 0 = false and non-zero = true.

@group(0) @binding(0) var<storage, read> condition: array<i32>;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> input_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    size: u32,              // Total number of elements in output
    condition_size: u32,    // Size of condition input (1 if scalar broadcast)
    x_size: u32,            // Size of input X (1 if scalar broadcast)
    y_size: u32,            // Size of input Y (1 if scalar broadcast)
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
    
    // Handle broadcasting for all three inputs:
    // - If size == 1, broadcast to all elements
    // - Otherwise, use regular indexing
    let cond_idx = select(idx, 0u, params.condition_size == 1u);
    let x_idx = select(idx, 0u, params.x_size == 1u);
    let y_idx = select(idx, 0u, params.y_size == 1u);
    
    // Load values
    let cond_val = condition[cond_idx];
    let x_val = input_x[x_idx];
    let y_val = input_y[y_idx];
    
    // WGSL select: select(false_val, true_val, condition)
    // If condition is non-zero (true), select x_val; otherwise select y_val
    output[idx] = select(y_val, x_val, cond_val != 0);
}
