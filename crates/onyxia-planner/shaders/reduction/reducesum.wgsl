// ReduceSum: Sum-reduce a tensor along specified axes
//
// For simplicity, this implementation handles single-axis reduction.
// Each workgroup processes one output element - summing all values along the reduction axis.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate data for shape information
struct ImmediateConstants {
    input_size: u32,        // Total number of elements in input
    output_size: u32,       // Total number of elements in output
    reduce_size: u32,       // Size of the dimension being reduced
    outer_size: u32,        // Product of dimensions before reduction axis
    inner_size: u32,        // Product of dimensions after reduction axis
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// Shared memory for partial sums within a workgroup
var<workgroup> partial_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_idx = global_id.x;
    
    // Bounds check - each workgroup handles one output element
    if (workgroup_id.x >= params.output_size) {
        return;
    }
    
    // Calculate which slice of input this output position corresponds to
    // For reduction along axis N with shape [a, b, c, d]:
    // - outer_size = product of dims before axis N
    // - reduce_size = size of axis N
    // - inner_size = product of dims after axis N
    //
    // Example: reduce [2, 8, 3] along axis 1:
    //   outer_size = 2, reduce_size = 8, inner_size = 3
    //   Each output element sums 8 values from input
    
    let outer_idx = workgroup_id.x / params.inner_size;
    let inner_idx = workgroup_id.x % params.inner_size;
    
    // Each thread in the workgroup handles a portion of the reduction
    let local_thread_idx = local_id.x;
    var sum: f32 = 0.0;
    
    // Sum values along the reduction axis
    // Stride through the reduce dimension with workgroup size
    for (var i: u32 = local_thread_idx; i < params.reduce_size; i += WG_SIZE) {
        let input_idx = outer_idx * params.reduce_size * params.inner_size + 
                        i * params.inner_size + 
                        inner_idx;
        sum += input[input_idx];
    }
    
    // Store partial sum in shared memory
    partial_sums[local_thread_idx] = sum;
    workgroupBarrier();
    
    // Parallel reduction in shared memory (tree reduction)
    // Only the first thread in the workgroup performs the reduction
    if (local_thread_idx == 0u) {
        var total: f32 = 0.0;
        for (var i: u32 = 0u; i < WG_SIZE; i++) {
            total += partial_sums[i];
        }
        output[workgroup_id.x] = total;
    }
}
