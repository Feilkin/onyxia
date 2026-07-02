// Common shader utilities for Onyxia operators
//
// This file provides helper functions used across multiple operator shaders.

/// Compute linear thread index from 2D/3D global invocation ID.
///
/// When workgroup counts exceed 65535, dispatches are distributed across
/// Y and Z dimensions. This function reconstructs the linear index.
///
/// # Arguments
///
/// * `global_id` — The @builtin(global_invocation_id) vec3<u32>
/// * `x_stride` — Number of threads in X dimension (num_workgroups_x * workgroup_size_x)
///
/// # Returns
///
/// Linear thread index for accessing 1D buffers
fn compute_linear_index(global_id: vec3<u32>, x_stride: u32) -> u32 {
    return global_id.x + global_id.y * x_stride + global_id.z * x_stride * 65535u;
}

/// Simplified version when x_stride is known to be below the 65535 limit (1D dispatch)
fn compute_linear_index_1d(global_id: vec3<u32>) -> u32 {
    return global_id.x;
}
