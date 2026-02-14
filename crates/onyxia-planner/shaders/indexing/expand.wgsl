// Expand operation: broadcasts tensor to target shape
//
// Implements ONNX Expand semantics:
//   For a tensor with shape [d0, d1, ..., dn] and target shape [t0, t1, ..., tm],
//   output has shape [t0, t1, ..., tm] where:
//   - Dimensions of size 1 are broadcast to match target
//   - New leading dimensions are added as needed
//   - Each dimension must satisfy: input_dim == 1 OR input_dim == target_dim
//
// Example 1:
//   input: [3, 1] with target=[3, 5]
//   output: [3, 5] where output[i][j] = input[i][0] for all j
//
// Example 2:
//   input: [3, 4] with target=[2, 3, 4]
//   output: [2, 3, 4] where output[k][i][j] = input[i][j] for all k
//
// Algorithm:
//   Each thread processes one output element. It:
//   1. Decomposes its thread ID into multi-dimensional output indices
//   2. Maps each output index to corresponding input index (0 if input dim is 1)
//   3. Computes the flat input index using input strides
//   4. Copies the element: output[tid] = input[input_idx]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants for shape and stride information
struct ImmediateConstants {
    output_rank: u32,           // Number of dimensions in output
    output_size: u32,           // Total elements in output tensor
    input_strides: array<u32, 6>,   // Strides for input tensor (row-major)
    output_strides: array<u32, 6>,  // Strides for output tensor (row-major)
    input_shape: array<u32, 6>,     // Input dimensions (padded with 1s)
    output_shape: array<u32, 6>,    // Output dimensions (padded with 1s)
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
    if (output_idx >= params.output_size) {
        return;
    }
    
    // Decompose flat output index into multi-dimensional indices
    // and compute corresponding input index
    var remaining = output_idx;
    var input_idx = 0u;
    
    for (var d = 0u; d < params.output_rank; d = d + 1u) {
        // Get coordinate in output dimension d
        let out_coord = remaining / params.output_strides[d];
        remaining = remaining % params.output_strides[d];
        
        // Map to input coordinate:
        // - If input dimension is 1, use coordinate 0 (broadcast)
        // - Otherwise, use the output coordinate
        let in_coord = select(out_coord, 0u, params.input_shape[d] == 1u);
        
        // Add contribution to input flat index
        input_idx = input_idx + in_coord * params.input_strides[d];
    }
    
    // Copy element from input to output (with broadcasting)
    output[output_idx] = input[input_idx];
}
