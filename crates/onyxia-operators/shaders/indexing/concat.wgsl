// Concat: Concatenate tensors along an arbitrary axis
//
// This shader copies one input tensor to the appropriate region of the output tensor.
// Multiple dispatches (one per input) are used to build the full concatenated output.
//
// For concatenation along axis `k`:
// - outer_size: product of dimensions before axis k
// - inner_size: product of dimensions after axis k
// - input_axis_size: size of this input along axis k
// - output_axis_offset: where this input starts in the output along axis k

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ConcatParams {
    outer_size: u32,           // Product of dims before concat axis
    inner_size: u32,           // Product of dims after concat axis
    input_axis_size: u32,      // Size of this input along concat axis
    output_axis_size: u32,     // Size of output along concat axis
    output_axis_offset: u32,   // Offset where this input starts in output
}

var<immediate> params: ConcatParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let input_size = params.outer_size * params.input_axis_size * params.inner_size;
    
    if (idx >= input_size) {
        return;
    }
    
    // Decompose linear index into (outer, axis, inner) coordinates
    // For input with shape effectively [outer_size, input_axis_size, inner_size]
    let inner_idx = idx % params.inner_size;
    let temp = idx / params.inner_size;
    let axis_idx = temp % params.input_axis_size;
    let outer_idx = temp / params.input_axis_size;
    
    // Map to output coordinates
    // Output has shape [outer_size, output_axis_size, inner_size]
    let output_axis_idx = axis_idx + params.output_axis_offset;
    let output_idx = outer_idx * params.output_axis_size * params.inner_size + 
                     output_axis_idx * params.inner_size + 
                     inner_idx;
    
    output[output_idx] = input[idx];
}
