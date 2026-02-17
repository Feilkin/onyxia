// Range operator shader.
//
// Generates a sequence of numbers: [start, start+delta, start+2*delta, ...]

@group(0) @binding(0) var<storage, read_write> output: array<f32>;

// Immediate constants (push constants)
struct ImmediateConstants {
    num_elements: u32,
    start: f32,
    delta: f32,
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
    
    output[idx] = immediates.start + f32(idx) * immediates.delta;
}
