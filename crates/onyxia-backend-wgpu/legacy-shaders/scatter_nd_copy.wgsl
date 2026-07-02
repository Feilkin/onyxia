// ScatterND copy phase - simple element-wise copy from data to output
//
// This is the first phase of ScatterND: copy the base data tensor to output.
// The second phase will then scatter updates over this copied data.

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Immediate constants
struct ImmediateConstants {
    size: u32,  // Total number of elements
}

var<immediate> params: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }
    
    output[idx] = data[idx];
}
