//! Canned inference driver for previewing the UI without a model.
//!
//! Run with `cargo run -p gemma-chat -- --demo` (native only).
//! It speaks the exact same `InferenceEvent` channel the real runtime uses, so
//! the UI code is identical in both modes — this just replaces the source of
//! events with a scripted sequence of load stages and one canned answer.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use eframe::egui;
use futures::StreamExt;
use futures::channel::mpsc as async_mpsc;

use crate::{InferenceEvent, InferenceRequest};

/// How long each load stage is held before advancing.
const STEP: Duration = Duration::from_millis(620);
/// How long each word is held while streaming the answer.
const TOKEN: Duration = Duration::from_millis(52);

/// The canned answer, streamed one word at a time.
const ANSWER: &str = "Onyxia lowers your ONNX graph into a compact intermediate representation, \
then compiles every operator to a WebGPU compute shader ahead of time. The quantized weights are \
uploaded to the GPU once and stay resident, so each token is generated entirely on-device — no \
round-trips back to JavaScript, and no network calls after the model has loaded.";

pub async fn run(
    mut request_rx: async_mpsc::UnboundedReceiver<InferenceRequest>,
    event_tx: mpsc::Sender<InferenceEvent>,
    egui_ctx: egui::Context,
    stop_flag: Arc<AtomicBool>,
) {
    let send = |event: InferenceEvent| {
        let _ = event_tx.send(event);
        egui_ctx.request_repaint();
    };

    // Walk the eight load stages. Stage 0/1 (WASM boot / start) are instant in
    // the real app too, but we show them so the segmented bar fills from empty.
    for stage in 0..crate::theme::STAGE_LABELS.len() {
        send(InferenceEvent::Progress {
            stage,
            label: crate::theme::STAGE_LABELS[stage].to_string(),
        });
        std::thread::sleep(STEP);
    }
    send(InferenceEvent::Ready {
        gpu_name: "Demo GPU · WebGPU".to_string(),
        vram_bytes: 1_181_116_006, // ~1.1 GiB, a plausible placeholder
    });

    // Answer every prompt with the canned response, streamed word by word.
    while let Some(request) = request_rx.next().await {
        match request {
            InferenceRequest::Reset => {}
            InferenceRequest::Generate(_) => {
                stop_flag.store(false, Ordering::Relaxed);
                let words: Vec<&str> = ANSWER.split(' ').collect();
                let mut emitted = 0;
                for (i, word) in words.iter().enumerate() {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let text = if i == 0 {
                        word.to_string()
                    } else {
                        format!(" {word}")
                    };
                    send(InferenceEvent::Token(text));
                    emitted += 1;
                    std::thread::sleep(TOKEN);
                }
                // Fabricate plausible metrics for the preview.
                let tps = 40.0 + (emitted as f64 % 16.0);
                let ttft_ms = 150.0 + (emitted as f64 % 110.0);
                send(InferenceEvent::Done {
                    tokens_per_sec: tps,
                    ttft_ms,
                    vram_bytes: 1_181_116_006, // matches the Ready placeholder
                });
            }
        }
    }
}
