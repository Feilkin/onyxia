// Slice operator - extract sub-tensor along multiple axes
//
// The shader maps each output element to its corresponding input element
// by applying start/step transformations per axis.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants for shape and slice parameters
struct ImmediateConstants {
    num_elements: u32,              // Total output elements
    rank: u32,                      // Tensor rank
    input_shape: array<u32, 6>,     // Input tensor shape
    output_shape: array<u32, 6>,    // Output tensor shape
    starts: array<i32, 6>,           // Start indices for each axis
    steps: array<i32, 6>,            // Step sizes for each axis
}

var<immediate> params: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Convert linear index to multi-dimensional coordinates
fn index_to_coords(idx: u32, shape: array<u32, 6>, rank: u32) -> array<u32, 6> {
    var coords: array<u32, 6>;
    var remaining = idx;
    
    for (var i = 0u; i < rank; i = i + 1u) {
        var stride = 1u;
        for (var j = i + 1u; j < rank; j = j + 1u) {
            stride = stride * shape[j];
        }
        coords[i] = remaining / stride;
        remaining = remaining % stride;
    }
    
    return coords;
}

// Convert multi-dimensional coordinates to linear index
fn coords_to_index(coords: array<u32, 6>, shape: array<u32, 6>, rank: u32) -> u32 {
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
    if (idx >= params.num_elements) {
        return;
    }
    
    // Convert output linear index to coordinates
    let output_coords = index_to_coords(idx, params.output_shape, params.rank);
    
    // Map output coordinates to input coordinates using start and step
    var input_coords: array<u32, 6>;
    for (var i = 0u; i < params.rank; i = i + 1u) {
        // input[i] = start[i] + output[i] * step[i]
        let start = params.starts[i];
        let step = params.steps[i];
        let coord = i32(output_coords[i]);
        input_coords[i] = u32(start + coord * step);
    }
    
    // Convert input coordinates to linear index
    let input_idx = coords_to_index(input_coords, params.input_shape, params.rank);
    
    // Copy element from input to output
    output[idx] = input[input_idx];
}
