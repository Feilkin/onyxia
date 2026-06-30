//! GPU buffer pool for memory reuse.
//!
//! The `BufferPool` maintains a collection of free GPU buffers organized by
//! power-of-2 size buckets. When a new output tensor is needed, the pool
//! is checked first. When an intermediate tensor is no longer live, its
//! buffer is returned to the pool for reuse.
//!
//! # Size bucketing
//!
//! Buffers are grouped by the *next power of two* ≥ their actual size. This
//! means a 1000-byte request may be served by a pooled 1024-byte buffer.
//! At most a 2× overhead is incurred per buffer, which is acceptable for
//! typical neural network activations.
//!
//! # Pool capacity
//!
//! Each size bucket holds at most [`MAX_PER_BUCKET`] buffers. When the bucket
//! is full, a returned buffer is dropped immediately rather than stored.
//! This prevents unbounded GPU memory growth during autoregressive generation
//! where sequence lengths change each token and old-size buffers would
//! otherwise accumulate until OOM.

use std::collections::HashMap;
use std::sync::Arc;

/// Maximum number of free buffers retained per size bucket.
///
/// A small value keeps GPU memory bounded while still providing meaningful
/// reuse for the most recently freed intermediate tensors.
const MAX_PER_BUCKET: usize = 8;

/// Maximum total GPU memory (in bytes) held by the pool across all buckets.
///
/// When this limit is reached, newly released buffers are dropped immediately.
/// 256 MiB is large enough to pool a full transformer layer's intermediates
/// while preventing unbounded growth during autoregressive KV-cache expansion.
const MAX_POOL_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Pool of reusable GPU buffers, keyed by power-of-2 size bucket.
pub struct BufferPool {
    /// Free buffers organized by size bucket.
    ///
    /// Key: bucket size in bytes (next power of 2 ≥ actual size).
    /// Value: stack of free buffers in that bucket.
    free_buffers: HashMap<usize, Vec<Arc<wgpu::Buffer>>>,

    /// Total GPU memory currently held by the pool (bytes).
    pub pool_bytes: usize,

    /// Total number of buffer allocations made (new buffers created).
    pub allocations: usize,

    /// Total number of buffer reuses from the pool.
    pub reuses: usize,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// Create an empty buffer pool.
    pub fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            pool_bytes: 0,
            allocations: 0,
            reuses: 0,
        }
    }

    /// Acquire a buffer of at least `size` bytes.
    ///
    /// Returns a pooled buffer from the appropriate size bucket if one is
    /// available. Otherwise, allocates a fresh buffer with the bucket size.
    ///
    /// The returned buffer has `STORAGE | COPY_SRC | COPY_DST` usage flags
    /// and is suitable for use as a compute shader output tensor.
    pub fn acquire(&mut self, size: usize, device: &wgpu::Device) -> Arc<wgpu::Buffer> {
        let bucket = Self::bucket_size(size);

        if let Some(stack) = self.free_buffers.get_mut(&bucket) {
            if let Some(buf) = stack.pop() {
                self.reuses += 1;
                self.pool_bytes -= bucket;
                return buf;
            }
        }

        // No pooled buffer available — allocate a new one.
        self.allocations += 1;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_tensor"),
            size: bucket as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The buffer is dropped (GPU memory freed) immediately if either:
    /// - Its size bucket already holds [`MAX_PER_BUCKET`] entries, or
    /// - Accepting it would exceed [`MAX_POOL_BYTES`] total.
    ///
    /// The second condition bounds growth caused by monotonically-growing
    /// tensors (e.g. KV-cache concat outputs during autoregressive generation)
    /// that land in ever-larger unique buckets and would never be reused.
    pub fn release(&mut self, buffer: Arc<wgpu::Buffer>) {
        let bucket = Self::bucket_size(buffer.size() as usize);
        let stack = self.free_buffers.entry(bucket).or_default();
        if stack.len() < MAX_PER_BUCKET && self.pool_bytes + bucket <= MAX_POOL_BYTES {
            self.pool_bytes += bucket;
            stack.push(buffer);
        }
        // else: buffer is dropped here, GPU memory is reclaimed.
    }

    /// Round `size` up to the nearest power-of-2 bucket (minimum 4 bytes).
    ///
    /// This ensures pooled buffers can serve requests for the same logical
    /// size class without fragmentation.
    pub fn bucket_size(size: usize) -> usize {
        // wgpu requires buffers to be at least 4 bytes.
        let size = size.max(4);
        size.next_power_of_two()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size_powers_of_two() {
        assert_eq!(BufferPool::bucket_size(1), 4);
        assert_eq!(BufferPool::bucket_size(4), 4);
        assert_eq!(BufferPool::bucket_size(5), 8);
        assert_eq!(BufferPool::bucket_size(8), 8);
        assert_eq!(BufferPool::bucket_size(9), 16);
        assert_eq!(BufferPool::bucket_size(1000), 1024);
        assert_eq!(BufferPool::bucket_size(1024), 1024);
        assert_eq!(BufferPool::bucket_size(1025), 2048);
    }

    #[test]
    fn test_bucket_size_minimum_4() {
        assert_eq!(BufferPool::bucket_size(0), 4);
        assert_eq!(BufferPool::bucket_size(1), 4);
        assert_eq!(BufferPool::bucket_size(2), 4);
        assert_eq!(BufferPool::bucket_size(3), 4);
    }

    #[test]
    fn test_pool_stats_start_at_zero() {
        let pool = BufferPool::new();
        assert_eq!(pool.allocations, 0);
        assert_eq!(pool.reuses, 0);
    }
}
