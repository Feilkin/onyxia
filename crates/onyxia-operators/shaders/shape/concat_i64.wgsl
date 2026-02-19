// Concatenate up to 4 I64 tensors along a specified axis
//
// I64 values are stored as pairs of u32 (low,high) in GPU buffers.
// Each logical i64 element occupies 2 u32 storage slots.

@group(0) @binding(0) var<storage, read> input0: array<u32>;
@group(0) @binding(1) var<storage, read> input1: array<u32>;
@group(0) @binding(2) var<storage, read> input2: array<u32>;
@group(0) @binding(3) var<storage, read> input3: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;

// Immediate constants for shape information
struct ImmediateConstants {
    rank: u32,              // Tensor rank
    axis: u32,              // Concatenation axis
    num_inputs: u32,        // Number of input tensors
    total_elements: u32,    // Total output elements (logical i64 count)
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
    
    // Iterate from first dimension to last for row-major layout
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
fn coords_to_index(coords: array<u32, 8>, shape: array<u32, 8>, rank: u32) -> u32 {
    var idx = 0u;
    var stride = 1u;
    
    for (var i: i32 = i32(rank) - 1; i >= 0; i = i - 1) {
        idx = idx + coords[i] * stride;
        stride = stride * shape[i];
    }
    
    return idx;
}

// Read I64 value (2 u32s) from the appropriate input tensor
// Returns (low_u32, high_u32)
fn read_input(input_idx: u32, offset: u32) -> vec2<u32> {
    let u32_offset = offset * 2u;  // Each i64 = 2 u32s
    var low: u32;
    var high: u32;
    
    switch input_idx {
        case 0u: {
            low = input0[u32_offset];
            high = input0[u32_offset + 1u];
        }
        case 1u: {
            low = input1[u32_offset];
            high = input1[u32_offset + 1u];
        }
        case 2u: {
            low = input2[u32_offset];
            high = input2[u32_offset + 1u];
        }
        case 3u: {
            low = input3[u32_offset];
            high = input3[u32_offset + 1u];
        }
        default: {
            low = 0u;
            high = 0u;
        }
    }
    
    return vec2<u32>(low, high);
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
    
    // Read i64 value (2 u32s) from appropriate input and write to output
    let value = read_input(input_idx, input_linear_idx);
    let u32_output_idx = idx * 2u;
    output[u32_output_idx] = value.x;      // low u32
    output[u32_output_idx + 1u] = value.y;  // high u32
}
