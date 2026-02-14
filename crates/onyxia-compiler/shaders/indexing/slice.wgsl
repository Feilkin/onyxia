// Slice operation: Extract sub-tensor from input tensor
//
// Implements ONNX Slice semantics:
//   output[i0][i1]...[iN] = input[starts[0] + i0*steps[0]][starts[1] + i1*steps[1]]...[starts[N] + iN*steps[N]]
//
// Supports:
// - Multi-axis slicing
// - Negative indices (resolved at plan time)
// - Strided slicing (steps > 1)
// - Reverse slicing (negative steps)
//
// Example:
//   input: shape [1, 2, 4], data = [[[1,2,3,4], [5,6,7,8]]]
//   starts: [0, 0, 1]
//   ends: [1, 2, 4]
//   steps: [1, 1, 2]
//   => output: shape [1, 2, 2], data = [[[2, 4], [6, 8]]]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Push constants for slice parameters
struct ImmediateConstants {
    rank: u32,                      // Number of dimensions
    num_output_elements: u32,       // Total elements in output tensor
    _padding: vec2<u32>,            // Align to 16 bytes
    
    // Per-dimension parameters (support up to 7 dimensions, max 128 bytes)
    input_strides: array<u32, 7>,   // Strides for input indexing
    output_strides: array<u32, 7>,  // Strides for output indexing
    starts: array<i32, 7>,          // Starting index for each dimension
    steps: array<i32, 7>,           // Step size for each dimension
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
    let output_idx = global_id.x;
    
    // Bounds check
    if (output_idx >= params.num_output_elements) {
        return;
    }
    
    // Convert flat output index to multi-dimensional coordinates
    // Then map each coordinate to input space using starts and steps
    var input_idx = 0u;
    var remaining = output_idx;
    
    for (var i = 0u; i < params.rank; i++) {
        // Get output coordinate for this dimension
        let out_coord = remaining / params.output_strides[i];
        remaining = remaining % params.output_strides[i];
        
        // Map output coordinate to input coordinate: in_coord = start + out_coord * step
        let in_coord = params.starts[i] + i32(out_coord) * params.steps[i];
        
        // Accumulate flattened input index
        input_idx += u32(in_coord) * params.input_strides[i];
    }
    
    // Copy element from input to output
    output[output_idx] = input[input_idx];
}
