// Range operation: generates a sequence of numbers
//
// Implements ONNX Range semantics:
//   Given start, delta, and output size, creates a 1D tensor:
//   output[i] = start + i * delta  (for i = 0 to size - 1)
//
// Example:
//   start=2, delta=2, size=4  =>  output=[2, 4, 6, 8]
//   start=0, delta=1, size=5  =>  output=[0, 1, 2, 3, 4]
//
// Algorithm:
//   Each thread computes one output element:
//   1. Check if thread ID is within bounds
//   2. Calculate: output[tid] = start + tid * delta

@group(0) @binding(0) var<storage, read_write> output: array<f32>;

// Immediate constants for range generation
struct ImmediateConstants {
    start: f32,    // Starting value
    delta: f32,    // Step/increment value
    size: u32,     // Number of elements to generate
    _pad: u32,     // Padding for 16-byte alignment
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
    let index = global_id.x;
    
    // Bounds check
    if (index >= params.size) {
        return;
    }
    
    // Generate range value: output[i] = start + i * delta
    output[index] = params.start + f32(index) * params.delta;
}
