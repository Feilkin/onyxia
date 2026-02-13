// Transpose operation: permutes tensor dimensions
//
// Implements ONNX Transpose semantics:
//   For a tensor with shape [d0, d1, ..., dn] and permutation perm,
//   output has shape [d_perm[0], d_perm[1], ..., d_perm[n]]
//
// Example:
//   input: [M, N] with perm=[1, 0]
//   output: [N, M] where output[j][i] = input[i][j]
//
// Algorithm:
//   Each thread processes one output element. It:
//   1. Decomposes its thread ID into multi-dimensional output indices
//   2. Computes the corresponding input flat index using input strides and permutation
//   3. Copies the element: output[tid] = input[input_idx]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Push constants for shape and permutation information
struct ImmediateConstants {
    rank: u32,              // Number of dimensions
    num_elements: u32,      // Total elements in tensor
    dispatch_size_x: u32,   // Number of workgroups in X dimension (for multi-dimensional dispatch)
    input_strides: array<u32, 6>,   // Strides for input tensor (row-major)
    output_strides: array<u32, 6>,  // Strides for output tensor (row-major)
    perm: array<u32, 6>,            // Permutation array (perm[i] = which input dim maps to output dim i)
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
    // Calculate thread ID for multi-dimensional dispatch
    // global_id.x: 0 to (dispatch_size_x * WG_SIZE - 1)
    // global_id.y: 0 to (dispatch_size_y - 1)
    // global_id.z: 0 to (dispatch_size_z - 1)
    let threads_per_row = params.dispatch_size_x * WG_SIZE;
    let tid = global_id.x + global_id.y * threads_per_row + global_id.z * threads_per_row * 65535u;
    
    // Bounds check
    if (tid >= params.num_elements) {
        return;
    }
    
    // Decompose tid into multi-dimensional output index
    // and compute corresponding input index
    var remaining = tid;
    var input_idx = 0u;
    
    for (var d = 0u; d < params.rank; d = d + 1u) {
        // Get coordinate in output dimension d
        let coord = remaining / params.output_strides[d];
        remaining = remaining % params.output_strides[d];
        
        // This coord corresponds to dimension perm[d] in the input
        // Add contribution to input flat index
        input_idx = input_idx + coord * params.input_strides[params.perm[d]];
    }
    
    // Copy element from input to output
    output[tid] = input[input_idx];
}
