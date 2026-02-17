// Cast shader: I32 → Bool
// Non-zero → true (stored as u32: 1), zero → false (stored as u32: 0)
//
// Note: WGSL bool can't be in storage buffers, so we use u32

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // bool stored as u32

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
    
    // Non-zero → 1 (true), zero → 0 (false)
    output[idx] = select(0u, 1u, input[idx] != 0);
}
