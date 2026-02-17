// Concatenate up to 4 tensors along a specified axis
//
// The shader reads from multiple input buffers and writes to a single output buffer.
// Each element in the output is copied from the appropriate input based on its position
// along the concatenation axis.

@group(0) @binding(0) var<storage, read> input0: array<f32>;
@group(0) @binding(1) var<storage, read> input1: array<f32>;
@group(0) @binding(2) var<storage, read> input2: array<f32>;
@group(0) @binding(3) var<storage, read> input3: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    rank: u32,              // Tensor rank
    axis: u32,              // Concatenation axis
    num_inputs: u32,        // Number of input tensors
    total_elements: u32,    // Total output elements
    output_shape: array<u32, 8>,    // Output shape (up to 8 dims)
    input_axis_sizes: array<u32, 4>, // Size of each input along axis (up to 4 inputs)
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

// Read from the appropriate input tensor
fn read_input(input_idx: u32, offset: u32) -> f32 {
    switch input_idx {
        case 0u: { return input0[offset]; }
        case 1u: { return input1[offset]; }
        case 2u: { return input2[offset]; }
        case 3u: { return input3[offset]; }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.total_elements) {
        return;
    }
    
    // Convert output linear index to coordinates
    let coords = index_to_coords(idx, params.output_shape, params.rank);
    
    // Determine which input tensor this element comes from
    // based on the coordinate along the concatenation axis
    let axis_coord = coords[params.axis];
    
    var cumulative_size = 0u;
    var input_idx = 0u;
    var axis_offset = 0u;
    
    for (var i = 0u; i < params.num_inputs; i = i + 1u) {
        let input_axis_size = params.input_axis_sizes[i];
        if (axis_coord < cumulative_size + input_axis_size) {
            input_idx = i;
            axis_offset = axis_coord - cumulative_size;
            break;
        }
        cumulative_size = cumulative_size + input_axis_size;
    }
    
    // Compute input coordinates (same as output except along axis)
    var input_coords = coords;
    input_coords[params.axis] = axis_offset;
    
    // Compute input shape (same as output except along axis)
    var input_shape = params.output_shape;
    input_shape[params.axis] = params.input_axis_sizes[input_idx];
    
    // Convert to linear input index
    let input_linear_idx = coords_to_index(input_coords, input_shape, params.rank);
    
    // Read from appropriate input and write to output
    output[idx] = read_input(input_idx, input_linear_idx);
}
