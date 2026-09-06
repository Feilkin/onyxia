//! `onyxia bench`: repeatable prefill/decode throughput measurement, with
//! an optional per-kernel GPU-time breakdown from timestamp queries.
//!
//! The benchmark is tokenizer-free: it feeds a fixed dummy token id, which
//! exercises exactly the same dispatch stream as real generation (kernel
//! cost does not depend on token values, only on sequence positions). One
//! warmup prefill + a few decode steps compile every pipeline before
//! anything is measured.
//!
//! Models without a `logits` output (encoders, embedding models) get the
//! forward protocol instead: one warmup pass, then `repeats` timed
//! stateless passes over `prefill_len` tokens.
//!
//! Generic over the backend session; the per-kernel profile, VRAM and
//! host-clock split are wgpu-only and come through [`BenchProbe`].

use crate::llm::LlmSession;
use anyhow::{Context, Result};
use onyxia_backend_wgpu::session::CpuTiming;
use onyxia_backend_wgpu::{KernelTiming, WgpuSession};
use onyxia_ir::Session;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// Any valid token id works; kernel cost is token-value-independent.
const DUMMY_TOKEN: i64 = 42;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Measured prefills per run; the report takes the median.
const PREFILL_SAMPLES: usize = 5;

fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub struct BenchConfig {
    /// Prompt length for the measured prefill (or the forward pass).
    pub prefill_len: usize,
    /// Number of measured single-token decode steps.
    pub decode_tokens: usize,
    /// Number of measured forward passes (forward protocol only).
    pub repeats: usize,
    /// Record per-kernel GPU time (needs timestamp queries).
    pub profile: bool,
    /// Write the full report as JSON here.
    pub json: Option<PathBuf>,
    /// Adapter description, for the report header.
    pub adapter: String,
}

/// Backend-specific instrumentation the report can use when available.
/// Every method has a "not available" default, so a backend without
/// timestamp queries or a memory tracker still benchmarks.
pub trait BenchProbe {
    fn enable_profiling(&mut self) -> bool {
        false
    }
    fn take_timings(&mut self) -> Result<Vec<KernelTiming>> {
        Ok(Vec::new())
    }
    fn take_cpu_timing(&mut self) -> Option<CpuTiming> {
        None
    }
    /// `(resident, peak)` device bytes.
    fn memory(&self) -> Option<(u64, u64)> {
        None
    }
}

impl BenchProbe for WgpuSession {
    fn enable_profiling(&mut self) -> bool {
        WgpuSession::enable_profiling(self)
    }
    fn take_timings(&mut self) -> Result<Vec<KernelTiming>> {
        Ok(pollster::block_on(WgpuSession::take_timings(self))?)
    }
    fn take_cpu_timing(&mut self) -> Option<CpuTiming> {
        Some(WgpuSession::take_cpu_timing(self))
    }
    fn memory(&self) -> Option<(u64, u64)> {
        Some((self.resident_bytes(), self.peak_resident_bytes()))
    }
}

impl BenchProbe for onyxia_backend_cubecl::CubeclSession {}

/// Wall-clock statistics over a set of decode steps, in seconds.
struct StepStats {
    mean: f64,
    min: f64,
    max: f64,
    stddev: f64,
}

impl StepStats {
    fn of(samples: &[f64]) -> Self {
        let n = samples.len().max(1) as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n;
        Self {
            mean,
            min: samples.iter().copied().fold(f64::INFINITY, f64::min),
            max: samples.iter().copied().fold(0.0, f64::max),
            stddev: var.sqrt(),
        }
    }
}

/// Per-kernel aggregate of a timing stream.
struct KernelRow {
    kernel: String,
    count: usize,
    total_ns: u64,
}

fn aggregate<'a>(
    timings: &'a [KernelTiming],
    key: impl Fn(&'a KernelTiming) -> &'a str,
) -> Vec<KernelRow> {
    let mut map: HashMap<&str, (usize, u64)> = HashMap::new();
    for t in timings {
        let e = map.entry(key(t)).or_default();
        e.0 += 1;
        e.1 += t.time_ns;
    }
    let mut rows: Vec<KernelRow> = map
        .into_iter()
        .map(|(k, (count, total_ns))| KernelRow {
            kernel: k.to_string(),
            count,
            total_ns,
        })
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.total_ns));
    rows
}

fn print_table(title: &str, timings: &[KernelTiming], steps: usize, top: usize, by_node: bool) {
    if timings.is_empty() {
        return;
    }
    let gpu_total: u64 = timings.iter().map(|t| t.time_ns).sum();
    let rows = aggregate(timings, |t| if by_node { &t.node } else { &t.kernel });
    let steps = steps.max(1) as f64;
    println!("\n{title} (GPU busy {:.3} ms/step)", ms(gpu_total) / steps);
    println!(
        "  {:<44} {:>10} {:>12} {:>7}",
        if by_node { "node" } else { "kernel" },
        "disp/step",
        "ms/step",
        "% GPU"
    );
    for row in rows.iter().take(top) {
        println!(
            "  {:<44} {:>10.1} {:>12.4} {:>6.1}%",
            truncate(&row.kernel, 44),
            row.count as f64 / steps,
            ms(row.total_ns) / steps,
            100.0 * row.total_ns as f64 / gpu_total.max(1) as f64,
        );
    }
    if rows.len() > top {
        let rest: u64 = rows[top..].iter().map(|r| r.total_ns).sum();
        println!(
            "  {:<44} {:>10} {:>12.4} {:>6.1}%",
            format!("… {} more", rows.len() - top),
            "",
            ms(rest) / steps,
            100.0 * rest as f64 / gpu_total.max(1) as f64,
        );
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max - 1).collect();
        format!("{head}…")
    }
}

/// Run the benchmark on a prepared session and print (and optionally
/// serialize) the report. Picks the forward protocol for models without
/// a `logits` output.
pub fn run<S: Session + BenchProbe>(session: &mut LlmSession<S>, cfg: &BenchConfig) -> Result<()> {
    if !session.output_names().iter().any(|n| n == "logits") {
        return run_forward(session, cfg);
    }
    let ids = vec![DUMMY_TOKEN; cfg.prefill_len];

    // Warmup: compiles every pipeline and populates the buffer pool so the
    // measured phase sees steady-state behavior.
    println!("Warmup (prefill {} + 2 decode steps)...", cfg.prefill_len);
    session.prefill(&ids).context("warmup prefill failed")?;
    for _ in 0..2 {
        session
            .decode(DUMMY_TOKEN)
            .context("warmup decode failed")?;
    }
    session.reset_full();

    let profiling = if cfg.profile {
        let on = session.backend_mut().enable_profiling();
        if !on {
            eprintln!("warning: device lacks TIMESTAMP_QUERY; --profile ignored");
        }
        on
    } else {
        false
    };

    // Measured prefill: the median of several, each from a fresh state
    // (a single prefill on the larger models swings 2× run to run).
    let mut prefills: Vec<f64> = Vec::with_capacity(PREFILL_SAMPLES);
    for i in 0..PREFILL_SAMPLES {
        if i > 0 {
            session.reset_full();
        }
        let start = Instant::now();
        session.prefill(&ids).context("prefill failed")?;
        prefills.push(start.elapsed().as_secs_f64());
    }
    let prefill_timings = session.backend_mut().take_timings()?;
    let prefill_s = median(&prefills);
    let prefill_min = prefills.iter().copied().fold(f64::INFINITY, f64::min);

    // Measured decode.
    let misses_before = session
        .backend_mut()
        .take_cpu_timing()
        .map_or(0, |c| c.bind_misses);
    let mut step_s: Vec<f64> = Vec::with_capacity(cfg.decode_tokens);
    for _ in 0..cfg.decode_tokens {
        let start = Instant::now();
        session.decode(DUMMY_TOKEN).context("decode failed")?;
        step_s.push(start.elapsed().as_secs_f64());
    }
    let decode_timings = session.backend_mut().take_timings()?;
    let cpu = session.backend_mut().take_cpu_timing();

    // Report.
    let stats = StepStats::of(&step_s);
    let decode_gpu_ns: u64 = decode_timings.iter().map(|t| t.time_ns).sum();
    let decode_gpu_per_step = ms(decode_gpu_ns) / cfg.decode_tokens.max(1) as f64;

    let memory = session.backend_mut().memory();

    println!("\nAdapter: {}", cfg.adapter);
    print_memory(memory);
    println!(
        "prefill: {} tokens in {:.1} ms median of {} (min {:.1}) ({:.1} tok/s)",
        cfg.prefill_len,
        prefill_s * 1e3,
        PREFILL_SAMPLES,
        prefill_min * 1e3,
        cfg.prefill_len as f64 / prefill_s,
    );
    println!(
        "decode:  {} tokens, {:.2} ms/tok mean (min {:.2}, max {:.2}, σ {:.2}) → {:.2} tok/s",
        cfg.decode_tokens,
        stats.mean * 1e3,
        stats.min * 1e3,
        stats.max * 1e3,
        stats.stddev * 1e3,
        1.0 / stats.mean,
    );
    if let Some(cpu) = &cpu {
        let n = cfg.decode_tokens.max(1) as f64;
        let per = |ns: u64| ms(ns) / n;
        let accounted = per(cpu.shapes_ns + cpu.encode_ns + cpu.wait_ns + cpu.readback_ns);
        println!(
            "decode step breakdown (host clock, ms/tok): shapes {:.2}, encode {:.2}, \
             gpu wait {:.2}, readback {:.2}, other (inputs, sampling) {:.2}",
            per(cpu.shapes_ns),
            per(cpu.encode_ns),
            per(cpu.wait_ns),
            per(cpu.readback_ns),
            (stats.mean * 1e3 - accounted).max(0.0),
        );
        println!(
            "  encode: {} dispatches/step, {:.0} bind-group cache misses/step",
            cpu.dispatches / cfg.decode_tokens.max(1) as u64,
            (cpu.bind_misses - misses_before) as f64 / n,
        );
    }
    if profiling {
        println!(
            "decode GPU busy: {:.2} ms/tok ({:.0}% of wall; the rest is CPU encode + submit + readback)",
            decode_gpu_per_step,
            100.0 * decode_gpu_per_step / (stats.mean * 1e3),
        );
        println!(
            "note: profiling encodes one pass per dispatch (timestamps); wall times above are \
             inflated — take throughput numbers from an unprofiled run"
        );
        print_table(
            "prefill by kernel",
            &prefill_timings,
            PREFILL_SAMPLES,
            15,
            false,
        );
        print_table(
            "decode by kernel",
            &decode_timings,
            cfg.decode_tokens,
            15,
            false,
        );
        print_table(
            "decode by node",
            &decode_timings,
            cfg.decode_tokens,
            20,
            true,
        );
    }

    if let Some(path) = &cfg.json {
        let mut report = json_report(cfg, prefill_s, &step_s, &prefill_timings, &decode_timings);
        report["vram"] = memory_json(memory);
        write_json(path, &report)?;
    }
    Ok(())
}

fn print_memory(memory: Option<(u64, u64)>) {
    if let Some((resident, peak)) = memory {
        println!(
            "VRAM: {:.2} GiB resident, {:.2} GiB peak",
            resident as f64 / GIB,
            peak as f64 / GIB,
        );
    }
}

fn memory_json(memory: Option<(u64, u64)>) -> serde_json::Value {
    match memory {
        Some((resident, peak)) => serde_json::json!({
            "resident_bytes": resident,
            "peak_bytes": peak,
        }),
        None => serde_json::Value::Null,
    }
}

fn write_json(path: &PathBuf, report: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Forward protocol for stateless models: `repeats` timed passes over
/// `prefill_len` tokens, each from a fresh session state. The last
/// module output (the pooled embedding for the embedding exports) is
/// downloaded every pass, as logits are in the decode protocol, and its
/// values go into the JSON report so two runtimes can be compared.
fn run_forward<S: Session + BenchProbe>(
    session: &mut LlmSession<S>,
    cfg: &BenchConfig,
) -> Result<()> {
    let output = session
        .output_names()
        .last()
        .context("model has no outputs")?
        .clone();
    let ids = vec![DUMMY_TOKEN; cfg.prefill_len];
    println!(
        "No 'logits' output: forward protocol, {} tokens × {} passes, output '{output}'",
        cfg.prefill_len, cfg.repeats
    );

    println!("Warmup (1 forward pass)...");
    session
        .forward(&ids, &output)
        .context("warmup forward failed")?;

    let profiling = cfg.profile && session.backend_mut().enable_profiling();
    if cfg.profile && !profiling {
        eprintln!("warning: device lacks TIMESTAMP_QUERY; --profile ignored");
    }
    let _ = session.backend_mut().take_timings()?;

    let mut pass_s: Vec<f64> = Vec::with_capacity(cfg.repeats);
    let mut last = None;
    for _ in 0..cfg.repeats {
        let start = Instant::now();
        last = Some(session.forward(&ids, &output).context("forward failed")?);
        pass_s.push(start.elapsed().as_secs_f64());
    }
    let timings = session.backend_mut().take_timings()?;
    let stats = StepStats::of(&pass_s);
    let memory = session.backend_mut().memory();
    let last = last.context("no passes run")?;
    let values = last.to_f32()?;

    println!("\nAdapter: {}", cfg.adapter);
    print_memory(memory);
    println!(
        "forward: {} tokens, {:.2} ms/pass mean (min {:.2}, max {:.2}, σ {:.2}) → {:.1} tok/s, {:.1} passes/s",
        cfg.prefill_len,
        stats.mean * 1e3,
        stats.min * 1e3,
        stats.max * 1e3,
        stats.stddev * 1e3,
        cfg.prefill_len as f64 / stats.mean,
        1.0 / stats.mean,
    );
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!(
        "output '{output}' shape {:?}, L2 norm {norm:.4}, head {:?}",
        last.shape(),
        &values[..values.len().min(4)]
    );
    if profiling {
        print_table("forward by kernel", &timings, cfg.repeats, 15, false);
        print_table("forward by node", &timings, cfg.repeats, 20, true);
    }

    if let Some(path) = &cfg.json {
        let kernels: serde_json::Value = aggregate(&timings, |t| &t.kernel)
            .iter()
            .map(|r| {
                serde_json::json!({
                    "kernel": r.kernel,
                    "dispatches": r.count,
                    "total_ms": ms(r.total_ns),
                })
            })
            .collect();
        let report = serde_json::json!({
            "adapter": cfg.adapter,
            "forward": {
                "tokens": cfg.prefill_len,
                "passes": cfg.repeats,
                "mean_ms_per_pass": stats.mean * 1e3,
                "min_ms_per_pass": stats.min * 1e3,
                "max_ms_per_pass": stats.max * 1e3,
                "stddev_ms": stats.stddev * 1e3,
                "tokens_per_sec": cfg.prefill_len as f64 / stats.mean,
                "passes_per_sec": 1.0 / stats.mean,
                "kernels": kernels,
                "output": output,
                "output_shape": last.shape(),
                "output_values": values,
            },
            "vram": memory_json(memory),
        });
        write_json(path, &report)?;
    }
    Ok(())
}

fn json_report(
    cfg: &BenchConfig,
    prefill_s: f64,
    step_s: &[f64],
    prefill_timings: &[KernelTiming],
    decode_timings: &[KernelTiming],
) -> serde_json::Value {
    let stats = StepStats::of(step_s);
    let kernels = |timings: &[KernelTiming]| -> serde_json::Value {
        aggregate(timings, |t| &t.kernel)
            .iter()
            .map(|r| {
                serde_json::json!({
                    "kernel": r.kernel,
                    "dispatches": r.count,
                    "total_ms": ms(r.total_ns),
                })
            })
            .collect()
    };
    serde_json::json!({
        "adapter": cfg.adapter,
        "prefill": {
            "tokens": cfg.prefill_len,
            "seconds": prefill_s,
            "tokens_per_sec": cfg.prefill_len as f64 / prefill_s,
            "samples": PREFILL_SAMPLES,
            "kernels": kernels(prefill_timings),
        },
        "decode": {
            "tokens": cfg.decode_tokens,
            "mean_ms_per_token": stats.mean * 1e3,
            "min_ms_per_token": stats.min * 1e3,
            "max_ms_per_token": stats.max * 1e3,
            "stddev_ms": stats.stddev * 1e3,
            "tokens_per_sec": 1.0 / stats.mean,
            "gpu_busy_ms_per_token":
                ms(decode_timings.iter().map(|t| t.time_ns).sum::<u64>())
                    / cfg.decode_tokens.max(1) as f64,
            "kernels": kernels(decode_timings),
        },
    })
}
