// Trilu operation: extract upper or lower triangular part of matrices
//
// Implements ONNX Trilu semantics:
//   Returns the upper or lower triangular part of a 2D matrix (or batch of matrices).
//   For tensors with rank > 2, the operation is applied to the last two dimensions.
//
// Parameters:
//   - upper: 1 for upper triangle, 0 for lower triangle
//   - k: diagonal offset (0 = main diagonal, >0 = above, <0 = below)
//
// Behavior:
//   - Upper triangle (upper=1): Keep elements where row <= col + k
//   - Lower triangle (upper=0): Keep elements where row >= col + k
//
// Example (upper=1, k=0):
//   Input:              Output:
//   [[1, 2, 3],        [[1, 2, 3],
//    [4, 5, 6],   =>    [0, 5, 6],
//    [7, 8, 9]]         [0, 0, 9]]
//
// Example (upper=0, k=0):
//   Input:              Output:
//   [[1, 2, 3],        [[1, 0, 0],
//    [4, 5, 6],   =>    [4, 5, 0],
//    [7, 8, 9]]         [7, 8, 9]]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants for shape and operation parameters
struct ImmediateConstants {
    total_size: u32,    // Total number of elements in the tensor
    m: u32,             // Number of rows (second-to-last dimension)
    n: u32,             // Number of columns (last dimension)
    k: i32,             // Diagonal offset
    upper: u32,         // 1 for upper triangle, 0 for lower triangle
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
    if (idx >= params.total_size) {
        return;
    }
    
    // Calculate position within the matrix
    // For a tensor with shape [..., M, N], we need to:
    // 1. Determine which matrix in the batch (if batched)
    // 2. Determine row and column within that matrix
    
    let matrix_size = params.m * params.n;
    let matrix_idx = idx % matrix_size;
    let row = matrix_idx / params.n;
    let col = matrix_idx % params.n;
    
    // Determine if this element should be kept or zeroed
    var keep = false;
    if (params.upper == 1u) {
        // Upper triangle: row <= col + k
        keep = i32(row) <= (i32(col) + params.k);
    } else {
        // Lower triangle: row >= col + k
        keep = i32(row) >= (i32(col) + params.k);
    }
    
    // Write output (either original value or zero)
    output[idx] = select(0.0, input[idx], keep);
}
