// Matrix multiplication - naive algorithm for debugging
//
// Computes C = A × B where:
// - A: [..., M, K]
// - B: [..., K, N]  
// - C: [..., M, N]

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    M: u32,           // Rows of A and output
    N: u32,           // Cols of B and output
    K: u32,           // Cols of A, rows of B
    batch_size: u32,  // Total batch dimensions (product of all batch dims)
}

var<immediate> params: ImmediateConstants;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let row = global_id.y;
    let col = global_id.x;
    
    // Bounds check
    if batch >= params.batch_size || row >= params.M || col >= params.N {
        return;
    }
    
    var sum = 0.0;
    
    // Compute dot product
    for (var k = 0u; k < params.K; k++) {
        let a_idx = batch * params.M * params.K + row * params.K + k;
        let b_idx = batch * params.K * params.N + k * params.N + col;
        sum += input_a[a_idx] * input_b[b_idx];
    }
    
    // Write result
    let output_idx = batch * params.M * params.N + row * params.N + col;
    output[output_idx] = sum;
}
