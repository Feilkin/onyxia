// MatMulNBits - Quantized matrix multiplication with N-bit weights
//
// Computes Y = A × dequantize(B) + bias (optional), where B is quantized to N bits (2-8)
// and stored in a block-wise format with per-block scales and zero-points.
//
// Inputs:
// - A: [..., M, K] - Input tensor (float)
// - B: [N, k_blocks, blob_size] - Packed quantized weights (uint8)
// - scales: [N, k_blocks] - Per-block scaling factors (float)
// - zero_points: [N, k_blocks] or [N, packed_size] - Per-block zero points (optional)
// - bias: [N] - Bias vector (optional)
//
// Output:
// - Y: [..., M, N] - Output tensor (float)

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<u32>; // Packed weights (uint8 as u32)
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>; // Can be f32 or u32 depending on packed flag
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Immediate constants for configuration
struct ImmediateConstants {
    M: u32,                  // Rows of A and output
    N: u32,                  // Cols of output (N in spec)
    K: u32,                  // Cols of A (K in spec)
    batch_size: u32,         // Total batch dimensions
    bits: u32,               // Bit-width (2-8)
    block_size: u32,         // Block size for quantization
    k_blocks: u32,           // Number of blocks: ceil(K / block_size)
    blob_size: u32,          // Bytes per block: block_size * bits / 8
    has_zero_points: u32,    // 1 if zero_points provided, 0 otherwise
    zero_points_packed: u32, // 1 if zero_points are packed (uint8), 0 if unpacked (float)
    has_bias: u32,           // 1 if bias provided, 0 otherwise
    _padding: u32,           // Alignment padding
}

var<immediate> params: ImmediateConstants;

// Extract a quantized value from packed weights
// n: output feature index
// k: input feature index
fn extract_quantized_weight(n: u32, k: u32) -> u32 {
    let block_idx = k / params.block_size;
    let pos_in_block = k % params.block_size;
    
    // Calculate bit position within the block's blob
    let bit_pos = pos_in_block * params.bits;
    let byte_idx = bit_pos / 8u;
    let bit_offset = bit_pos % 8u;
    
    // Get base index into packed array
    // B is shaped [N, k_blocks, blob_size] where each element is a byte
    let base_byte_idx = (n * params.k_blocks + block_idx) * params.blob_size + byte_idx;
    
    // Convert byte index to u32 index (since WGSL reads 4 bytes at a time)
    let u32_idx = base_byte_idx / 4u;
    let byte_offset_in_u32 = (base_byte_idx % 4u) * 8u;
    
    // Read the u32 containing our byte(s)
    let u32_val = input_b[u32_idx];
    let byte_val = (u32_val >> byte_offset_in_u32) & 0xFFu;
    
    // Extract bits from the byte
    let mask = (1u << params.bits) - 1u;
    let value = (byte_val >> bit_offset) & mask;
    
    // Handle case where value spans two bytes
    if bit_offset + params.bits > 8u {
        // Need to read from next byte too
        let next_byte_idx = base_byte_idx + 1u;
        let next_u32_idx = next_byte_idx / 4u;
        let next_byte_offset_in_u32 = (next_byte_idx % 4u) * 8u;
        let next_u32_val = input_b[next_u32_idx];
        let next_byte = (next_u32_val >> next_byte_offset_in_u32) & 0xFFu;
        
        let bits_from_first = 8u - bit_offset;
        let bits_from_second = params.bits - bits_from_first;
        let value_first = (byte_val >> bit_offset) & ((1u << bits_from_first) - 1u);
        let value_second = next_byte & ((1u << bits_from_second) - 1u);
        return value_first | (value_second << bits_from_first);
    }
    
    return value;
}

// Get zero point for a given block
// n: output feature index
// block_idx: block index along K dimension
fn get_zero_point(n: u32, block_idx: u32) -> f32 {
    if params.has_zero_points == 0u {
        // Default zero point: 2^(bits-1)
        let default_zp = 1u << (params.bits - 1u);
        return f32(default_zp);
    }
    
    if params.zero_points_packed == 0u {
        // Unpacked format: [N, k_blocks], stored as f32
        let idx = n * params.k_blocks + block_idx;
        return bitcast<f32>(zero_points[idx]);
    } else {
        // Packed format: [N, ceil(k_blocks * bits / 8)], stored as uint8
        // Extract similar to weight extraction
        let bit_pos = block_idx * params.bits;
        let byte_idx = bit_pos / 8u;
        let bit_offset = bit_pos % 8u;
        
        let packed_size = (params.k_blocks * params.bits + 7u) / 8u;
        let base_byte_idx = n * packed_size + byte_idx;
        
        // Convert byte index to u32 index
        let u32_idx = base_byte_idx / 4u;
        let byte_offset_in_u32 = (base_byte_idx % 4u) * 8u;
        
        let u32_val = zero_points[u32_idx];
        let byte_val = (u32_val >> byte_offset_in_u32) & 0xFFu;
        
        let mask = (1u << params.bits) - 1u;
        let value = (byte_val >> bit_offset) & mask;
        
        // Handle spanning two bytes
        if bit_offset + params.bits > 8u {
            let next_byte_idx = base_byte_idx + 1u;
            let next_u32_idx = next_byte_idx / 4u;
            let next_byte_offset_in_u32 = (next_byte_idx % 4u) * 8u;
            let next_u32_val = zero_points[next_u32_idx];
            let next_byte = (next_u32_val >> next_byte_offset_in_u32) & 0xFFu;
            
            let bits_from_first = 8u - bit_offset;
            let bits_from_second = params.bits - bits_from_first;
            let value_first = (byte_val >> bit_offset) & ((1u << bits_from_first) - 1u);
            let value_second = next_byte & ((1u << bits_from_second) - 1u);
            return f32(value_first | (value_second << bits_from_first));
        }
        
        return f32(value);
    }
}

// Dequantize a weight value
// n: output feature index
// k: input feature index
fn dequantize_weight(n: u32, k: u32) -> f32 {
    let block_idx = k / params.block_size;
    
    // Get quantized value
    let quantized = extract_quantized_weight(n, k);
    
    // Get scale: scales[n, block_idx]
    let scale_idx = n * params.k_blocks + block_idx;
    let scale = scales[scale_idx];
    
    // Get zero point
    let zero_point = get_zero_point(n, block_idx);
    
    // Dequantize: (quantized - zero_point) * scale
    return (f32(quantized) - zero_point) * scale;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let row = global_id.y;  // M dimension
    let col = global_id.x;  // N dimension
    
    // Bounds check
    if batch >= params.batch_size || row >= params.M || col >= params.N {
        return;
    }
    
    var sum = 0.0;
    
    // Compute dot product: A[batch, row, :] · dequantized_B[col, :]
    // Note: B is stored as [N, k_blocks, blob_size], but conceptually it's [N, K] after dequantization
    for (var k = 0u; k < params.K; k++) {
        // A is stored as [..., M, K]
        let a_idx = batch * params.M * params.K + row * params.K + k;
        let a_val = input_a[a_idx];
        
        // Dequantize B[col, k]
        let b_val = dequantize_weight(col, k);
        
        sum += a_val * b_val;
    }
    
    // Add bias if present
    if params.has_bias != 0u {
        sum += bias[col];
    }
    
    // Write result: Y[batch, row, col]
    let output_idx = batch * params.M * params.N + row * params.N + col;
    output[output_idx] = sum;
}
