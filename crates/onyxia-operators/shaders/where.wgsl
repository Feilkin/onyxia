// Where operator shader.

// Conditional element selection: output[i] = condition[i] ? x[i] : y[i]
// Implements NumPy-style broadcasting by converting the linear output index
// into multi-dimensional coordinates and mapping them into each input's
// (potentially broadcasted) coordinates.

@group(0) @binding(0) var<storage, read> condition: array<u32>;  // bool stored as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    output_rank: u32,               // number of dimensions
    total_elements: u32,            // total output elements
    padding: u32,                   // alignment padding
    output_shape: array<u32, 8>,    // up to 8 dims
    cond_shape: array<u32, 8>,
    x_shape: array<u32, 8>,
    y_shape: array<u32, 8>,
}

var<immediate> params: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Convert linear output index to multi-dimensional coordinates (row-major)
fn index_to_coords(idx: u32, shape: array<u32,8>, rank: u32) -> array<u32,8> {
    var coords: array<u32,8>;
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
fn coords_to_index(coords: array<u32,8>, shape: array<u32,8>, rank: u32) -> u32 {
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
    if (idx >= params.total_elements) {
        return;
    }

    let rank = params.output_rank;
    let out_coords = index_to_coords(idx, params.output_shape, rank);

    // Map to condition coords
    var cond_coords: array<u32,8>;
    for (var i = 0u; i < rank; i = i + 1u) {
        if (params.cond_shape[i] == 1u) {
            cond_coords[i] = 0u;
        } else {
            cond_coords[i] = out_coords[i];
        }
    }
    let cond_idx = coords_to_index(cond_coords, params.cond_shape, rank);

    // Map to x coords
    var x_coords: array<u32,8>;
    for (var i = 0u; i < rank; i = i + 1u) {
        if (params.x_shape[i] == 1u) {
            x_coords[i] = 0u;
        } else {
            x_coords[i] = out_coords[i];
        }
    }
    let x_idx = coords_to_index(x_coords, params.x_shape, rank);

    // Map to y coords
    var y_coords: array<u32,8>;
    for (var i = 0u; i < rank; i = i + 1u) {
        if (params.y_shape[i] == 1u) {
            y_coords[i] = 0u;
        } else {
            y_coords[i] = out_coords[i];
        }
    }
    let y_idx = coords_to_index(y_coords, params.y_shape, rank);

    let cond = condition[cond_idx] != 0u;
    output[idx] = select(y[y_idx], x[x_idx], cond);
}
