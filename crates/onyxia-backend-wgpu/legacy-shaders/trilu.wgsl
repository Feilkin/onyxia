// Trilu operator shader.
//
// Extracts upper or lower triangular part of matrices.
// Elements outside the triangle are set to zero.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants (push constants)
struct ImmediateConstants {
    num_elements: u32,
    rows: u32,
    cols: u32,
    k: i32,        // Diagonal offset
    upper: u32,    // 1 for upper, 0 for lower
}
var<immediate> immediates: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= immediates.num_elements {
        return;
    }
    
    // Compute row and column indices within the matrix
    let matrix_size = immediates.rows * immediates.cols;
    let local_idx = idx % matrix_size;
    let row = local_idx / immediates.cols;
    let col = local_idx % immediates.cols;
    
    // Determine if element is in triangular region
    var keep: bool;
    if immediates.upper == 1u {
        // Upper triangular: keep if col >= row + k
        keep = i32(col) >= i32(row) + immediates.k;
    } else {
        // Lower triangular: keep if col <= row + k
        keep = i32(col) <= i32(row) + immediates.k;
    }
    
    output[idx] = select(0.0, input[idx], keep);
}
