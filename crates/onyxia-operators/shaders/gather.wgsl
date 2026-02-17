// Gather operator - gather elements from input tensor along a specified axis
//
// Given:
// - data.shape = [d0, d1, ..., d_{axis-1}, d_axis, d_{axis+1}, ..., d_n]
// - indices.shape = [i0, i1, ..., i_k]
// Output:
// - output.shape = [d0, ..., d_{axis-1}, i0, ..., i_k, d_{axis+1}, ..., d_n]
//
// For each output element, we:
// 1. Convert output linear index to multi-dimensional coordinates
// 2. Extract the indices coordinates (middle section)
// 3. Look up the index value from indices tensor
// 4. Build data coordinates by replacing axis dimension with the looked-up index
// 5. Read from data tensor and write to output

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    output_size: u32,           // Total output elements
    axis: u32,                  // Axis to gather along
    data_rank: u32,             // Rank of data tensor
    indices_rank: u32,          // Rank of indices tensor
    output_rank: u32,           // Rank of output tensor
    data_shape: array<u32, 8>,     // Data shape (up to 8 dims)
    indices_shape: array<u32, 8>,  // Indices shape (up to 8 dims)
    output_shape: array<u32, 8>,   // Output shape (up to 8 dims)
    x_stride: u32,              // X dimension stride for 2D dispatch support
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
    
    for (var i = 0u; i < rank; i = i + 1u) {
        let dim = rank - 1u - i;
        idx = idx + coords[dim] * stride;
        stride = stride * shape[dim];
    }
    
    return idx;
}

// Helper: compute linear index from 2D/3D dispatch
fn compute_linear_index(global_id: vec3<u32>, x_stride: u32) -> u32 {
    return global_id.x + global_id.y * x_stride + global_id.z * x_stride * 65535u;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = compute_linear_index(global_id, params.x_stride);
    if output_idx >= params.output_size {
        return;
    }
    
    // Convert output linear index to multi-dimensional coordinates
    let output_coords = index_to_coords(output_idx, params.output_shape, params.output_rank);
    
    // Split output coordinates into three parts:
    // 1. Prefix: [0..axis) from data
    // 2. Middle: [axis..axis+indices_rank) from indices
    // 3. Suffix: [axis+indices_rank..output_rank) from data
    
    // Extract indices coordinates (middle section)
    var indices_coords: array<u32, 8>;
    for (var i = 0u; i < params.indices_rank; i = i + 1u) {
        indices_coords[i] = output_coords[params.axis + i];
    }
    
    // Convert indices coordinates to linear index
    let indices_idx = coords_to_index(indices_coords, params.indices_shape, params.indices_rank);
    
    // Look up index value from indices tensor
    let index_value = indices[indices_idx];
    
    // Handle negative indices (wrap around)
    let axis_size = i32(params.data_shape[params.axis]);
    var actual_index = index_value;
    if actual_index < 0 {
        actual_index = actual_index + axis_size;
    }
    
    // Build data coordinates:
    // - Copy prefix from output coords
    // - Set axis dimension to actual_index
    // - Copy suffix from output coords (shifted by indices_rank - 1)
    var data_coords: array<u32, 8>;
    
    // Copy prefix (before axis)
    for (var i = 0u; i < params.axis; i = i + 1u) {
        data_coords[i] = output_coords[i];
    }
    
    // Set axis dimension to looked-up index
    data_coords[params.axis] = u32(actual_index);
    
    // Copy suffix (after axis)
    for (var i = params.axis + 1u; i < params.data_rank; i = i + 1u) {
        let output_offset = i - params.axis - 1u + params.indices_rank;
        data_coords[i] = output_coords[params.axis + output_offset];
    }
    
    // Convert data coordinates to linear index
    let data_idx = coords_to_index(data_coords, params.data_shape, params.data_rank);
    
    // Read from data and write to output
    output[output_idx] = data[data_idx];
}
