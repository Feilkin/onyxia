// Transpose (permute dimensions) of a tensor
//
// For each output element, compute its coordinates, apply inverse permutation,
// and read from the corresponding input position.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    rank: u32,                  // Tensor rank
    total_elements: u32,        // Total elements
    input_shape: array<u32, 8>, // Input shape (up to 8 dims)
    perm: array<u32, 8>,        // Permutation indices (up to 8 dims)
}

var<immediate> params: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Convert linear index to multi-dimensional coordinates
fn index_to_coords(idx: u32, shape: array<u32, 8>, rank: u32) -> array<u32, 8> {
    var coords: array<u32, 8>;
    var remaining = idx;
    
    for (var i: i32 = i32(rank) - 1; i >= 0; i = i - 1) {
        var stride = 1u;
        for (var j: u32 = u32(i) + 1u; j < rank; j = j + 1u) {
            stride = stride * shape[j];
        }
        coords[i] = remaining / stride;
        remaining = remaining % stride;
    }
    
    return coords;
}

// Convert multi-dimensional coordinates to linear index
fn coords_to_index(coords: array<u32, 8>, shape: array<u32, 8>, rank: u32) -> u32 {
    var idx = 0u;
    var stride = 1u;
    
    for (var i: i32 = i32(rank) - 1; i >= 0; i = i - 1) {
        idx = idx + coords[i] * stride;
        stride = stride * shape[i];
    }
    
    return idx;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.total_elements) {
        return;
    }
    
    // Compute output shape from input shape and permutation
    var output_shape: array<u32, 8>;
    for (var i = 0u; i < params.rank; i = i + 1u) {
        output_shape[i] = params.input_shape[params.perm[i]];
    }
    
    // Convert output linear index to coordinates
    let output_coords = index_to_coords(idx, output_shape, params.rank);
    
    // Apply inverse permutation to get input coordinates
    // If output_coords[i] corresponds to output dimension i, which came from
    // input dimension perm[i], then input_coords[perm[i]] = output_coords[i]
    var input_coords: array<u32, 8>;
    for (var i = 0u; i < params.rank; i = i + 1u) {
        input_coords[params.perm[i]] = output_coords[i];
    }
    
    // Convert to linear input index
    let input_idx = coords_to_index(input_coords, params.input_shape, params.rank);
    
    // Copy from input to output
    output[idx] = input[input_idx];
}
