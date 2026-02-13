// Type casting shader: Convert between data types
//
// Supports conversions:
// - I64 → F32 (for attention mask processing in Gemma)
// - F32 → F16 (precision reduction)
// - and more as needed

// Push constants for metadata
struct ImmediateConstants {
    num_elements: u32,
}

var<immediate> params: ImmediateConstants;

// Workgroup size - can be overridden via shader defs
#ifdef WORKGROUP_SIZE
    const WG_SIZE: u32 = #{WORKGROUP_SIZE};
#else
    const WG_SIZE: u32 = 256u;
#endif

// I64 → F32 conversion
// WGSL doesn't support i64 natively, so we read as pairs of u32
#ifdef CAST_I64_TO_F32
@group(0) @binding(0) var<storage, read> input_u32: array<u32>;  // I64 stored as u32 pairs
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_elements) {
        return;
    }
    
    // Read low 32 bits of i64 (stored at idx*2)
    // For Gemma attention masks, values fit in i32 range (0, 1, sequence positions)
    let low_bits = input_u32[idx * 2u];
    
    // Convert to f32 via i32 (handles sign correctly for small values)
    output[idx] = f32(i32(low_bits));
}
#endif

// I64 → I32 conversion
// WGSL doesn't support i64 natively, so we read as pairs of u32
#ifdef CAST_I64_TO_I32
@group(0) @binding(0) var<storage, read> input_u32: array<u32>;  // I64 stored as u32 pairs
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_elements) {
        return;
    }
    
    // Read low 32 bits of i64 (stored at idx*2)
    // Assuming values fit in i32 range
    let low_bits = input_u32[idx * 2u];
    
    // Convert to i32 (truncates to lower 32 bits)
    output[idx] = i32(low_bits);
}
#endif

// F32 → F16 conversion
#ifdef CAST_F32_TO_F16
// Note: WGSL f16 requires enable directive and native support
// For now we just store as u16 representation
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;  // Packed f16 as u16

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_elements) {
        return;
    }
    
    // TODO: Implement f32 to f16 conversion
    // For now, just truncate (incorrect but placeholder)
    output[idx] = u32(input[idx]);
}
#endif

// I32 → F32 conversion (simple case)
#ifdef CAST_I32_TO_F32
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_elements) {
        return;
    }
    
    output[idx] = f32(input[idx]);
}
#endif

// F32 → I32 conversion
#ifdef CAST_F32_TO_I32
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_elements) {
        return;
    }
    
    output[idx] = i32(input[idx]);
}
#endif
