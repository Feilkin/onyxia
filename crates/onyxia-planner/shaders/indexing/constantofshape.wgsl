// ConstantOfShape operation: creates tensor filled with constant value
//
// Implements ONNX ConstantOfShape semantics:
//   Given a shape tensor, creates an output tensor of that shape filled
//   with a constant value specified by the 'value' attribute.
//
// Example:
//   shape = [2, 3], value = 1.0
//   output = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//
// Algorithm:
//   Each thread writes one output element. Simple parallel fill operation:
//   1. Check if thread ID is within bounds
//   2. Write the constant value to output[tid]

@group(0) @binding(0) var<storage, read_write> output: array<f32>;

// Immediate constants for fill operation
struct ImmediateConstants {
    output_size: u32,  // Total number of elements to fill
    fill_value: f32,   // Constant value to fill with
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
    if (index >= params.output_size) {
        return;
    }
    
    // Fill output with constant value
    output[index] = params.fill_value;
}
