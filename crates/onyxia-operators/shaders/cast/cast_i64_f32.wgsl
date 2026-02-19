// Cast shader: I64 → F32
// May lose precision for large values
//
// Note: WGSL doesn't support i64 in storage buffers without extensions.
// We read i64 as pairs of u32 (low, high).

@group(0) @binding(0) var<storage, read> input: array<u32>; // i64 stored as u32 pairs
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

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
    
    // Reconstruct i64 from u32 pairs (low, high)
    let low = input[idx * 2u];
    let high = input[idx * 2u + 1u];
    
    // Reconstruct i64 value
    // This is approximate for large values due to f32 precision
    let low_f = f32(low);
    let high_f = f32(i32(high)) * 4294967296.0; // 2^32
    
    output[idx] = high_f + low_f;
}
