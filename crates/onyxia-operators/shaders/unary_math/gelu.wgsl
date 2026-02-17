// GELU (Gaussian Error Linear Unit) activation function
//
// GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
// Using tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

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

// Constants for GELU tanh approximation
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;  // sqrt(2/pi)
const GELU_COEFF: f32 = 0.044715;

// GELU activation using tanh approximation
fn gelu(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    let tanh_inner = tanh(inner);
    return 0.5 * x * (1.0 + tanh_inner);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.num_elements) {
        return;
    }
    
    // Apply GELU activation
    output[idx] = gelu(input[idx]);
}
