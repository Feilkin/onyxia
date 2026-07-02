// Cast shader: I64 → I32
// Truncate (take low 32 bits)
//
// Note: WGSL doesn't support i64 in storage buffers without extensions.
// We read i64 as pairs of u32 (low, high).

@group(0) @binding(0) var<storage, read> input: array<u32>; // i64 stored as u32 pairs
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
    
    // Take low 32 bits from i64 (stored as u32 pairs)
    let low_bits = input[idx * 2u];
    output[idx] = i32(low_bits);
}
