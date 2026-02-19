//! ConstantOfShape shader - fill a tensor with a constant value.
//!
//! This shader fills every element of the output tensor with a constant value.
//! Used by the ConstantOfShape operator.
//!
//! Bindings:
//! - @group(0) @binding(0): output buffer (read_write)
//!
//! Immediate constants:
//! - num_elements: u32 - total number of elements in output
//! - fill_value: f32 - constant value to fill

@group(0) @binding(0) var<storage, read_write> output: array<f32>;

struct ImmediateConstants {
    num_elements: u32,
    fill_value: f32,
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
    if idx >= params.num_elements {
        return;
    }

    output[idx] = params.fill_value;
}
