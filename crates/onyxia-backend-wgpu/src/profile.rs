//! Opt-in per-dispatch GPU timing via timestamp queries.
//!
//! When profiling is enabled on a [`crate::WgpuSession`], every compute
//! pass writes begin/end timestamps ([`wgpu::Features::TIMESTAMP_QUERY`],
//! requested at device creation when the adapter offers it). Timestamps
//! resolve at each submit and are read back asynchronously by
//! [`crate::WgpuSession::take_timings`], tagged with the pipeline label
//! and the IR node that issued the dispatch.
//!
//! This measures **GPU execution time per dispatch** — the time between a
//! pass starting and finishing on the device — which is the number that
//! kernel optimization moves. Wall-clock decode time additionally includes
//! CPU encoding, submission, and readback latency; the gap between the two
//! is itself a useful signal (dispatch overhead vs. kernel cost).

use onyxia_ir::{Error, Result};

/// GPU time attributed to one dispatch.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    /// Pipeline label, e.g. `matmul_f32` or `fused_softmax_row_f32`.
    pub kernel: String,
    /// Name of the IR node that issued the dispatch (empty when the
    /// dispatch was issued outside of node execution).
    pub node: String,
    /// GPU execution time of the pass, in nanoseconds.
    pub time_ns: u64,
}

/// Timestamp pairs per query set (a set holds 2× this many queries;
/// `wgpu::QUERY_SET_MAX_QUERIES` is 4096).
const PAIRS_PER_SET: u32 = 2048;

/// Per-session profiling state. Query sets are recycled across batches;
/// resolved timestamps wait in `pending` until [`Self::collect`].
pub(crate) struct Profiler {
    /// Nanoseconds per timestamp tick (`Queue::get_timestamp_period`).
    period: f32,
    /// Query sets for the in-flight batch, `PAIRS_PER_SET` pairs each.
    sets: Vec<wgpu::QuerySet>,
    /// Pairs handed out in the in-flight batch.
    pairs: u32,
    /// `(kernel, node)` per pair, in allocation order.
    meta: Vec<(String, String)>,
    /// Node tag applied to subsequent dispatches (set by the session).
    pub tag: String,
    /// Submitted batches awaiting readback: staging buffer + its metadata.
    pending: Vec<(wgpu::Buffer, Vec<(String, String)>)>,
}

impl Profiler {
    pub fn new(queue: &wgpu::Queue) -> Self {
        Self {
            period: queue.get_timestamp_period(),
            sets: Vec::new(),
            pairs: 0,
            meta: Vec::new(),
            tag: String::new(),
            pending: Vec::new(),
        }
    }

    /// Allocate a timestamp pair for the next pass. Returns the query set
    /// and the begin index within it (end is begin + 1).
    pub fn begin_pass(&mut self, device: &wgpu::Device, kernel: &str) -> (usize, u32) {
        let set_idx = (self.pairs / PAIRS_PER_SET) as usize;
        if set_idx == self.sets.len() {
            self.sets.push(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("onyxia_profile"),
                ty: wgpu::QueryType::Timestamp,
                count: PAIRS_PER_SET * 2,
            }));
        }
        let base = (self.pairs % PAIRS_PER_SET) * 2;
        self.pairs += 1;
        self.meta.push((kernel.to_string(), self.tag.clone()));
        (set_idx, base)
    }

    pub fn query_set(&self, idx: usize) -> &wgpu::QuerySet {
        &self.sets[idx]
    }

    /// Resolve the batch's timestamps into a staging buffer. Must be
    /// called on the batch's encoder before it is finished; the staging
    /// buffer is queued for readback in [`Self::collect`].
    pub fn resolve(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.pairs == 0 {
            return;
        }
        let size = u64::from(self.pairs) * 2 * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile_resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profile_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut remaining = self.pairs * 2;
        for (i, set) in self.sets.iter().enumerate() {
            let used = remaining.min(PAIRS_PER_SET * 2);
            let offset = u64::from(i as u32 * PAIRS_PER_SET) * 2 * 8;
            encoder.resolve_query_set(set, 0..used, &resolve, offset);
            remaining -= used;
        }
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
        self.pending.push((staging, std::mem::take(&mut self.meta)));
        self.pairs = 0;
        // Query sets are reused by the next batch (queue ordering makes
        // the resolve above read the old values before they rewrite).
    }

    /// Read back all resolved batches. The caller must have submitted the
    /// work and be prepared to await buffer maps (native callers poll the
    /// device; on the web the browser drives progress).
    pub async fn collect(&mut self, device: &wgpu::Device) -> Result<Vec<KernelTiming>> {
        let mut out = Vec::new();
        for (staging, meta) in self.pending.drain(..) {
            let slice = staging.slice(..);
            let (tx, rx) = futures_channel::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            #[cfg(not(target_arch = "wasm32"))]
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|e| Error::Runtime(format!("GPU poll failed: {e:?}")))?;
            #[cfg(target_arch = "wasm32")]
            let _ = device;
            rx.await
                .map_err(|e| Error::Runtime(format!("buffer map canceled: {e}")))?
                .map_err(|e| Error::Runtime(format!("buffer map failed: {e}")))?;
            let bytes = slice.get_mapped_range();
            for (i, (kernel, node)) in meta.into_iter().enumerate() {
                let at = |q: usize| {
                    let o = q * 8;
                    u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap())
                };
                let (begin, end) = (at(i * 2), at(i * 2 + 1));
                out.push(KernelTiming {
                    kernel,
                    node,
                    time_ns: (end.saturating_sub(begin) as f64 * f64::from(self.period)) as u64,
                });
            }
            drop(bytes);
            staging.unmap();
        }
        Ok(out)
    }
}
