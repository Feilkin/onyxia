// Gather operation: embedding lookup and tensor indexing
//
// Implements ONNX Gather semantics:
//   output[i][j][k] = data[indices[i][j]][k]  (for axis=0)
//
// General formula for axis `a`:
//   Output shape: data.shape[:a] + indices.shape + data.shape[a+1:]
//
// For Gemma's token embedding:
//   data: [262144, 640] (vocab_size × hidden_size)
//   indices: [batch, seq] of I64 token IDs
//   output: [batch, seq, 640]

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices_u32: array<u32>;  // I64 as u32 pairs
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    num_output_elements: u32,  // Total elements in output tensor
    inner_dim: u32,             // Elements per gathered slice (product of data.shape[axis+1:])
    axis: u32,                  // Gather axis (currently only axis=0 is tested)
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    
    // Bounds check
    if (tid >= params.num_output_elements) {
        return;
    }
    
    // For axis=0:
    // - idx_pos: which index to look up (tid / inner_dim)
    // - inner_pos: position within the gathered row (tid % inner_dim)
    let idx_pos = tid / params.inner_dim;
    let inner_pos = tid % params.inner_dim;
    
    // Read I64 index as pair of u32 (little-endian)
    // WGSL has no native i64, so we store I64 as two u32 values
    // For valid indices < 2^31, we can safely use just the low 32 bits
    let low = indices_u32[idx_pos * 2u];
    let high = indices_u32[idx_pos * 2u + 1u];  // Usually 0 for positive indices
    
    // Use low 32 bits as the index (safe for vocab sizes < 2^31)
    let data_idx = low;
    
    // Gather: read from data at [data_idx, inner_pos]
    // In flattened form: data[data_idx * inner_dim + inner_pos]
    output[tid] = data[data_idx * params.inner_dim + inner_pos];
}
