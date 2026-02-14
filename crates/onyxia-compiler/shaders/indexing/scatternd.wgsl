// ScatterND operation: scatter updates into a tensor at specified indices
//
// Implements ONNX ScatterND semantics with two-pass approach:
//   Pass 1 (mode=0): Copy data to output
//   Pass 2 (mode=1): Scatter updates at indices with reduction
//
// Inputs:
// - data: tensor to copy (any shape)
// - indices: I64 tensor of indices where last dim = rank of indices
// - updates: values to scatter
// - output: result tensor (same shape as data)
//
// Attributes via immediates:
// - mode: 0=copy, 1=scatter
// - size: for copy=data_size, for scatter=num_updates
// - indices_last_dim: rank of indices (scatter only)
// - reduction: 0=none (replace), 1=add, 2=mul, 3=max, 4=min (scatter only)
// - output_strides: strides for index calculation (scatter only)

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices_u32: array<u32>;  // I64 as u32 pairs
@group(0) @binding(2) var<storage, read> updates: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Unified immediate constants structure
// For copy mode: only size is used
// For scatter mode: all fields are used
struct ImmediateConstants {
    size: u32,              // data_size (copy) or num_updates (scatter)
    indices_last_dim: u32,  // rank of indices (scatter only)
    reduction: u32,         // reduction mode (scatter only)
    pad0: u32,              // padding for alignment
    output_strides: array<u32, 8>,  // strides for index calculation
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Distinguish mode by checking if indices_last_dim is 0 (copy) or non-zero (scatter)
@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Check bounds
    if (idx >= params.size) {
        return;
    }
    
    // Determine mode: if indices_last_dim is 0, it's copy mode
    if (params.indices_last_dim == 0u) {
        // Copy mode: copy data[idx] to output[idx]
        output[idx] = data[idx];
    } else {
        // Scatter mode: scatter updates at indices
        let update_idx = idx;
        
        // Calculate output index from indices tensor
        // indices shape is [..., indices_last_dim]
        // For each update, we have indices_last_dim index values
        var output_idx = 0u;
        let indices_offset = update_idx * params.indices_last_dim;
        
        for (var i = 0u; i < params.indices_last_dim; i++) {
            // Read I64 index as pair of u32 (little-endian)
            // For valid indices < 2^31, we can safely use just the low 32 bits
            let low = indices_u32[(indices_offset + i) * 2u];
            let high = indices_u32[(indices_offset + i) * 2u + 1u];  // Usually 0 for positive indices
            
            // Accumulate to output index using strides
            output_idx += low * params.output_strides[i];
        }
        
        // Get update value
        let update_val = updates[update_idx];
        
        // Apply reduction
        if (params.reduction == 0u) {
            // none: replace
            output[output_idx] = update_val;
        } else if (params.reduction == 1u) {
            // add
            // Note: Potential race condition if multiple updates target same location
            // For now, we accept this limitation (ONNX spec leaves this undefined)
            output[output_idx] += update_val;
        } else if (params.reduction == 2u) {
            // mul
            output[output_idx] *= update_val;
        } else if (params.reduction == 3u) {
            // max
            output[output_idx] = max(output[output_idx], update_val);
        } else if (params.reduction == 4u) {
            // min
            output[output_idx] = min(output[output_idx], update_val);
        }
    }
}
