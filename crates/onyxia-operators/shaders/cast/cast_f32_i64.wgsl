// Cast shader: F32 → I64
// Truncate towards zero
//
// Note: WGSL doesn't support i64 in storage buffers without extensions.
// We store i64 as pairs of u32 (low, high).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // i64 stored as u32 pairs

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
    
    let val = input[idx];
    
    // Clamp to i64 range (approximate)
    let clamped = clamp(val, -9223372036854775808.0, 9223372036854775807.0);
    
    // Convert to i64 (stored as u32 pairs)
    // This is a simplified conversion that works for values in i32 range
    let i64_val = i32(clamped);
    output[idx * 2u] = u32(i64_val);
    output[idx * 2u + 1u] = select(0u, 0xFFFFFFFFu, i64_val < 0);
}
