// Softmax Pass 1: Find maximum along axis
//
// Algorithm:
// - Each workgroup handles one (outer, inner) coordinate
// - Threads cooperatively find max along the axis dimension
// - Uses parallel reduction tree for efficiency

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> max_values: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    outer_size: u32,     // Product of dimensions before axis
    axis_size: u32,      // Size of the axis dimension
    inner_size: u32,     // Product of dimensions after axis
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Shared memory for parallel reduction within workgroup
var<workgroup> partial_max: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let outer = workgroup_id.x / params.inner_size;
    let inner = workgroup_id.x % params.inner_size;
    let thread_idx = local_id.x;
    
    // Bounds check
    if (outer >= params.outer_size || inner >= params.inner_size) {
        return;
    }
    
    // Find max along axis using parallel reduction
    var local_max = -3.402823e38;  // -FLT_MAX
    
    for (var i = thread_idx; i < params.axis_size; i += WG_SIZE) {
        let idx = outer * params.axis_size * params.inner_size + 
                  i * params.inner_size + inner;
        local_max = max(local_max, input[idx]);
    }
    
    partial_max[thread_idx] = local_max;
    workgroupBarrier();
    
    // Tree reduction in workgroup memory
    var active_threads = WG_SIZE / 2u;
    while (active_threads > 0u) {
        if (thread_idx < active_threads) {
            partial_max[thread_idx] = max(partial_max[thread_idx], 
                                          partial_max[thread_idx + active_threads]);
        }
        workgroupBarrier();
        active_threads /= 2u;
    }
    
    if (thread_idx == 0u) {
        max_values[outer * params.inner_size + inner] = partial_max[0];
    }
}
