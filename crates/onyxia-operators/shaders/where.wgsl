// Where operator shader.
//
// Conditional element selection: output[i] = condition[i] ? x[i] : y[i]
// Supports broadcasting for all three inputs.

@group(0) @binding(0) var<storage, read> condition: array<u32>;  // bool stored as u32
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Immediate constants (push constants)
struct ImmediateConstants {
    num_elements: u32,
    cond_size: u32,
    x_size: u32,
    y_size: u32,
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
    
    // Compute broadcast indices (simple modulo broadcasting)
    let cond_idx = idx % immediates.cond_size;
    let x_idx = idx % immediates.x_size;
    let y_idx = idx % immediates.y_size;
    
    let cond = condition[cond_idx] != 0u;
    output[idx] = select(y[y_idx], x[x_idx], cond);
}
