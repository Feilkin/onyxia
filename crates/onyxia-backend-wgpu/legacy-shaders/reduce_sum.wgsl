// ReduceSum: Compute sum along specified axes
//
// Algorithm:
// - Each workgroup handles one output element
// - Threads within the workgroup cooperatively sum elements
// - Uses strided access to handle arbitrary axis reduction
// - Tree reduction in workgroup shared memory

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Push constants for shape information
struct ImmediateConstants {
    input_size: u32,       // Total number of elements in input
    output_size: u32,      // Total number of elements in output  
    reduction_size: u32,   // Number of elements to reduce per output element
    stride: u32,           // Stride between elements in reduction (for proper indexing)
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Shared memory for parallel reduction within workgroup
var<workgroup> partial_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_idx = workgroup_id.x;
    let thread_idx = local_id.x;
    
    // Bounds check
    if (output_idx >= params.output_size) {
        return;
    }
    
    // Each thread accumulates its portion of the reduction
    var thread_sum: f32 = 0.0;
    
    // For strided reduction:
    // - If stride == 1: contiguous elements (last axis reduction)
    // - Otherwise: elements are stride apart
    let base_offset = output_idx % params.stride;
    let block_idx = output_idx / params.stride;
    
    // Stride through the reduction range
    for (var i = thread_idx; i < params.reduction_size; i += WG_SIZE) {
        let input_idx = block_idx * params.stride * params.reduction_size + i * params.stride + base_offset;
        if (input_idx < params.input_size) {
            thread_sum += input[input_idx];
        }
    }
    
    // Store partial sum in workgroup memory
    partial_sums[thread_idx] = thread_sum;
    workgroupBarrier();
    
    // Parallel reduction tree within workgroup
    var active_threads = WG_SIZE / 2u;
    while (active_threads > 0u) {
        if (thread_idx < active_threads) {
            partial_sums[thread_idx] += partial_sums[thread_idx + active_threads];
        }
        workgroupBarrier();
        active_threads /= 2u;
    }
    
    // Thread 0 writes the final sum (no division, unlike ReduceMean)
    if (thread_idx == 0u) {
        output[output_idx] = partial_sums[0];
    }
}
