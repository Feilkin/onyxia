// ScatterND scatter phase - scatter updates into output at specified indices
//
// Given:
// - data.shape = [d0, d1, ..., d_n]
// - indices.shape = [i0, i1, ..., i_m, k] where k <= n+1
// - updates.shape = [i0, i1, ..., i_m, d_k, d_{k+1}, ..., d_n]
//
// For each update element at coordinates [u0, u1, ..., u_p]:
// 1. Split coordinates into: [batch_coords, slice_coords]
//    - batch_coords: [u0, ..., u_{m-1}] (matches indices prefix)
//    - slice_coords: [u_m, ..., u_p] (remaining dimensions)
// 2. Look up index tuple from indices[batch_coords]
// 3. Build output coordinates: [index_tuple, slice_coords]
// 4. Apply update to output at that location (with reduction mode)

@group(0) @binding(0) var<storage, read> indices: array<i32>;
@group(0) @binding(1) var<storage, read> updates: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Immediate constants for shape information
struct ImmediateConstants {
    updates_size: u32,          // Total updates elements
    k: u32,                     // Length of index tuples (indices last dim)
    data_rank: u32,             // Rank of data/output tensor
    indices_rank: u32,          // Rank of indices tensor
    updates_rank: u32,          // Rank of updates tensor
    reduction_mode: u32,        // 0=none, 1=add, 2=mul, 3=max, 4=min
    data_shape: array<u32, 8>,     // Data/output shape (up to 8 dims)
    indices_shape: array<u32, 8>,  // Indices shape (up to 8 dims)
    updates_shape: array<u32, 8>,  // Updates shape (up to 8 dims)
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

// Apply reduction operation
fn apply_reduction(current: f32, update: f32, mode: u32) -> f32 {
    switch mode {
        case 0u: {  // none - just replace
            return update;
        }
        case 1u: {  // add
            return current + update;
        }
        case 2u: {  // mul
            return current * update;
        }
        case 3u: {  // max
            return max(current, update);
        }
        case 4u: {  // min
            return min(current, update);
        }
        default: {
            return update;
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let update_idx = global_id.x;
    if update_idx >= params.updates_size {
        return;
    }
    
    // Convert update linear index to multi-dimensional coordinates
    let update_coords = index_to_coords(update_idx, params.updates_shape, params.updates_rank);
    
    // Split update coordinates into:
    // - batch_coords: [0..indices_rank-1] - indexes into the indices tensor
    // - slice_coords: [indices_rank-1..updates_rank] - remaining dimensions
    
    // Build indices coordinates (batch coords + last dim = 0 initially)
    var indices_coords: array<u32, 8>;
    for (var i = 0u; i < params.indices_rank - 1u; i = i + 1u) {
        indices_coords[i] = update_coords[i];
    }
    
    // Build output coordinates by looking up index tuple from indices
    var output_coords: array<u32, 8>;
    
    // For each dimension in the index tuple (k dimensions)
    for (var i = 0u; i < params.k; i = i + 1u) {
        // Set the last dimension of indices_coords to i
        indices_coords[params.indices_rank - 1u] = i;
        
        // Look up index value
        let indices_linear = coords_to_index(indices_coords, params.indices_shape, params.indices_rank);
        let index_value = indices[indices_linear];
        
        // Handle negative indices
        let dim_size = i32(params.data_shape[i]);
        var actual_index = index_value;
        if actual_index < 0 {
            actual_index = actual_index + dim_size;
        }
        
        output_coords[i] = u32(actual_index);
    }
    
    // Copy the slice coordinates (remaining dimensions after k)
    for (var i = params.k; i < params.data_rank; i = i + 1u) {
        let update_offset = i - params.k + params.indices_rank - 1u;
        output_coords[i] = update_coords[update_offset];
    }
    
    // Convert output coordinates to linear index
    let output_linear = coords_to_index(output_coords, params.data_shape, params.data_rank);
    
    // Read update value
    let update_value = updates[update_idx];
    
    // Apply reduction (note: "none" mode has potential race conditions if indices overlap)
    if params.reduction_mode == 0u {
        // Simple write (no atomics needed for "none" mode assuming no overlapping indices)
        output[output_linear] = update_value;
    } else {
        // For add/mul/max/min, we should use atomics, but WGSL doesn't support atomic float ops
        // For now, we just do a simple read-modify-write (may have race conditions)
        let current_value = output[output_linear];
        output[output_linear] = apply_reduction(current_value, update_value, params.reduction_mode);
    }
}
