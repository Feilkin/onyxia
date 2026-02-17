// Expand (broadcast) a tensor to a new shape
//
// Broadcasting rules:
// - Input dimensions are aligned to the right
// - Dimensions of size 1 are broadcast to match target size
// - Other dimensions must match exactly

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    output_rank: u32,              // Output tensor rank
    input_rank: u32,               // Input tensor rank  
    total_elements: u32,           // Total output elements
    padding: u32,                  // Alignment padding
    output_shape: array<u32, 8>,   // Output shape (up to 8 dims)
    input_shape: array<u32, 8>,    // Input shape (up to 8 dims, right-aligned with leading 1s)
}

var<immediate> params: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Convert linear output index to multi-dimensional coordinates
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
    
    // Convert output linear index to coordinates
    let output_coords = index_to_coords(idx, params.output_shape, params.output_rank);
    
    // Map output coordinates to input coordinates using broadcasting rules
    // For each dimension: if input_dim == 1, use 0; otherwise use output coordinate
    var input_coords: array<u32, 8>;
    for (var i = 0u; i < params.output_rank; i = i + 1u) {
        if (params.input_shape[i] == 1u) {
            input_coords[i] = 0u;
        } else {
            input_coords[i] = output_coords[i];
        }
    }
    
    // Convert to linear input index
    let input_idx = coords_to_index(input_coords, params.input_shape, params.output_rank);
    
    // Copy from input to output
    output[idx] = input[input_idx];
}
