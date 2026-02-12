// Matrix Multiplication: C = A × B (f32 precision)
//
// Input A: [M, K]
// Input B: [K, N]
// Output C: [M, N]
//
// Uses tiled algorithm with shared memory for efficiency.
// Each workgroup computes a TILE_M × TILE_N tile of the output.

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;

struct ImmediateConstants {
    M: u32,  // Rows of A and C
    N: u32,  // Columns of B and C
    K: u32,  // Columns of A and rows of B
}

var<immediate> params: ImmediateConstants;

// Tile sizes - can be overridden via shader defs
#ifdef TILE_M
    const TILE_M_SIZE: u32 = #{TILE_M};
#else
    const TILE_M_SIZE: u32 = 16u;
#endif

#ifdef TILE_N
    const TILE_N_SIZE: u32 = #{TILE_N};
#else
    const TILE_N_SIZE: u32 = 16u;
#endif

#ifdef TILE_K
    const TILE_K_SIZE: u32 = #{TILE_K};
#else
    const TILE_K_SIZE: u32 = 16u;
#endif

// Shared memory for tiling
// Each workgroup loads tiles of A and B into shared memory
var<workgroup> tile_a: array<f32, TILE_M_SIZE * TILE_K_SIZE>;
var<workgroup> tile_b: array<f32, TILE_K_SIZE * TILE_N_SIZE>;

@compute @workgroup_size(TILE_M_SIZE, TILE_N_SIZE, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = global_id.y;  // Global row index in C
    let col = global_id.x;  // Global column index in C
    
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    let M = params.M;
    let N = params.N;
    let K = params.K;
    
    // Accumulator for this output element
    var acc: f32 = 0.0;
    
    // Number of tiles along K dimension
    let num_tiles = (K + TILE_K_SIZE - 1u) / TILE_K_SIZE;
    
    // Loop over tiles of K
    for (var tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {
        let k_start = tile_idx * TILE_K_SIZE;
        
        // Load tile of A into shared memory
        // Each thread loads one element
        let a_row = workgroup_id.y * TILE_M_SIZE + local_row;
        let a_col = k_start + local_col;
        if (a_row < M && a_col < K) {
            tile_a[local_row * TILE_K_SIZE + local_col] = matrix_a[a_row * K + a_col];
        } else {
            tile_a[local_row * TILE_K_SIZE + local_col] = 0.0;
        }
        
        // Load tile of B into shared memory
        let b_row = k_start + local_row;
        let b_col = workgroup_id.x * TILE_N_SIZE + local_col;
        if (b_row < K && b_col < N) {
            tile_b[local_row * TILE_N_SIZE + local_col] = matrix_b[b_row * N + b_col];
        } else {
            tile_b[local_row * TILE_N_SIZE + local_col] = 0.0;
        }
        
        // Wait for all threads to finish loading
        workgroupBarrier();
        
        // Compute partial dot product for this tile
        for (var k = 0u; k < TILE_K_SIZE; k++) {
            acc += tile_a[local_row * TILE_K_SIZE + k] * tile_b[k * TILE_N_SIZE + local_col];
        }
        
        // Wait before loading next tile
        workgroupBarrier();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        matrix_c[row * N + col] = acc;
    }
}
