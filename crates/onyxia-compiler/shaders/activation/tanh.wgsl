// Hyperbolic Tangent (Tanh) activation function
//
// Tanh(x) = tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
//
// Computes the element-wise hyperbolic tangent of the input tensor.
// This matches the ONNX specification for Tanh operator.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ImmediateConstants {
    size: u32,  // Total number of elements
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
    
    // Apply tanh function (WGSL has built-in tanh)
    output[idx] = tanh(input[idx]);
}
