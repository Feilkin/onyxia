// Matrix Multiplication with Q4 Quantization: C = A × B_quantized
//
// Input A: [M, K] — activations (f32)
// Input B: [N, n_blocks_per_col, blob_size] — quantized weights (u8, packed 4-bit values)
// Input scales: [N, n_blocks_per_col] — dequantization scales (f32)
// Input zero_points: [N, n_blocks_per_col_padded] — optional zero points (u8)
// Output C: [M, N] — standard matmul result (f32)
//
// Q4 quantization stores weights as 4-bit integers (0-15), packed 2 per byte.
// Each block of weights shares a scale and zero_point.
// Dequantization formula: weight_f32 = (q4_value - zero_point) * scale
//
// This implementation uses a straightforward per-element approach where each
// thread computes one output element by iterating over K dimension.

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
#ifdef HAS_ZERO_POINTS
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> matrix_c: array<f32>;
#else
@group(0) @binding(3) var<storage, read_write> matrix_c: array<f32>;
#endif

struct ImmediateConstants {
    M: u32,                  // Rows of A and C
    N: u32,                  // Columns of C (output dimension)
    K: u32,                  // Columns of A (inner dimension)
    block_size: u32,         // Number of elements per quantization block
    n_blocks_per_col: u32,   // Number of blocks along K dimension
}

var<immediate> params: ImmediateConstants;

// Workgroup size can be overridden via shader defs
#ifdef WORKGROUP_X
    const WORKGROUP_X_SIZE: u32 = #{WORKGROUP_X};
#else
    const WORKGROUP_X_SIZE: u32 = 16u;
#endif

#ifdef WORKGROUP_Y
    const WORKGROUP_Y_SIZE: u32 = #{WORKGROUP_Y};
#else
    const WORKGROUP_Y_SIZE: u32 = 16u;
#endif

// Unpack a 4-bit value from packed u32 storage
// packed_weights stores 8 values per u32 (4 bits each)
fn unpack_q4(packed_data: u32, element_idx: u32) -> u32 {
    let shift = (element_idx & 7u) * 4u;
    return (packed_data >> shift) & 0xFu;
}

// Get zero point for a specific block (returns 8 if not provided)
fn get_zero_point(n: u32, block_idx: u32) -> f32 {
    #ifdef HAS_ZERO_POINTS
        // Zero points are packed: 8 per u32
        let zero_points_per_row = params.n_blocks_per_col;
        let flat_idx = n * zero_points_per_row + block_idx;
        let packed_idx = flat_idx / 8u;
        let element_idx = flat_idx % 8u;
        let packed = zero_points[packed_idx];
        return f32(unpack_q4(packed, element_idx));
    #else
        return 8.0; // Default zero point for 4-bit unsigned (midpoint of 0-15)
    #endif
}

@compute @workgroup_size(WORKGROUP_X_SIZE, WORKGROUP_Y_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let row = global_id.y;  // Row index in C (0..M)
    let col = global_id.x;  // Column index in C (0..N)
    
    let M = params.M;
    let N = params.N;
    let K = params.K;
    let block_size = params.block_size;
    let n_blocks_per_col = params.n_blocks_per_col;
    
    // Check bounds
    if (row >= M || col >= N) {
        return;
    }
    
    // Accumulator for this output element
    var acc: f32 = 0.0;
    
    // Iterate over K dimension in blocks
    for (var block_idx = 0u; block_idx < n_blocks_per_col; block_idx++) {
        // Get scale and zero_point for this block
        let scale_idx = col * n_blocks_per_col + block_idx;
        let scale = scales[scale_idx];
        let zero_point = get_zero_point(col, block_idx);
        
        // Starting K index for this block
        let k_start = block_idx * block_size;
        let k_end = min(k_start + block_size, K);
        
        // Process each element in the block
        for (var k = k_start; k < k_end; k++) {
            // Load activation value
            let a_idx = row * K + k;
            let a_val = matrix_a[a_idx];
            
            // Unpack quantized weight
            // Layout: packed_weights[col][block_idx][element_within_block / 8]
            // blob_size is in bytes, but packed_weights is array<u32>
            // So we need: blob_size_u32 = blob_size / 4
            let k_in_block = k - k_start;
            let blob_size_u32 = block_size / 8u; // 8 Q4 values per u32
            let packed_idx_base = col * n_blocks_per_col * blob_size_u32 + block_idx * blob_size_u32;
            let packed_idx = packed_idx_base + k_in_block / 8u;
            let element_idx = k_in_block % 8u;
            
            let packed_val = packed_weights[packed_idx];
            let q4_val = f32(unpack_q4(packed_val, element_idx));
            
            // Dequantize: weight_f32 = (q4_value - zero_point) * scale
            let weight = (q4_val - zero_point) * scale;
            
            // Accumulate
            acc += a_val * weight;
        }
    }
    
    // Write result to global memory
    matrix_c[row * N + col] = acc;
}
