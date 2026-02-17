// Cast shader: F32 → I32
// Truncate towards zero (not round)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

struct ImmediateConstants {
    num_elements: u32,
}
var<immediate> immediates: ImmediateConstants;

#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= immediates.num_elements {
        return;
    }
    
    // Truncate towards zero (WGSL i32() truncates)
    output[idx] = i32(input[idx]);
}
