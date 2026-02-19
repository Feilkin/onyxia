// Softmax Pass 2: Compute exp and sum
//
// Algorithm:
// - Each workgroup handles one (outer, inner) coordinate
// - Threads cooperatively compute sum of exp(x - max)
// - Uses parallel reduction tree for summing

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> max_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_sums: array<f32>;

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
var<workgroup> partial_sum: array<f32, WG_SIZE>;

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
    
    let max_val = max_values[outer * params.inner_size + inner];
    
    // Compute sum of exp(x - max)
    var local_sum = 0.0;
    
    for (var i = thread_idx; i < params.axis_size; i += WG_SIZE) {
        let idx = outer * params.axis_size * params.inner_size + 
                  i * params.inner_size + inner;
        local_sum += exp(input[idx] - max_val);
    }
    
    partial_sum[thread_idx] = local_sum;
    workgroupBarrier();
    
    // Tree reduction for sum
    var active_threads = WG_SIZE / 2u;
    while (active_threads > 0u) {
        if (thread_idx < active_threads) {
            partial_sum[thread_idx] += partial_sum[thread_idx + active_threads];
        }
        workgroupBarrier();
        active_threads /= 2u;
    }
    
    if (thread_idx == 0u) {
        exp_sums[outer * params.inner_size + inner] = partial_sum[0];
    }
}
