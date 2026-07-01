//! Gemma 3 270M chat demo built on Onyxia.
//!
//! This is an example application showing how to drive the Onyxia ONNX runtime
//! for autoregressive text generation. All Gemma-specific logic (KV cache
//! management, chat template formatting, token sampling) lives here in the
//! application layer — the runtime itself knows nothing about LLMs.
//!
//! Usage:
//!   gemma-chat <model-dir>
//!
//! Where <model-dir> contains:
//!   onnx/model.onnx          — full-precision ONNX model (+ .onnx_data)
//!   tokenizer.json           — HuggingFace tokenizer
//!   chat_template.jinja      — Jinja2 chat template (optional)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{Context, Result};
use eframe::egui;
use futures::StreamExt;
use futures::channel::mpsc as async_mpsc;
use rand::SeedableRng;
use rand::rngs::StdRng;
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

mod inference;
mod sampling;
mod tokenizer;

use inference::{LlmConfig, LlmSession};
use sampling::{SamplingConfig, sample};
use tokenizer::{ChatMessage, Tokenizer};

/// Where to load the model from: a directory on native, a base URL on web.
#[cfg(not(target_arch = "wasm32"))]
type ModelSource = PathBuf;
#[cfg(target_arch = "wasm32")]
type ModelSource = String;

const MAX_TOKENS: usize = 512;
const MAX_SEQ_LEN: usize = 2048;
const NUM_LAYERS: usize = 26;

// ── inter-thread communication ──────────────────────────────────────────────

enum InferenceEvent {
    /// A load-stage update shown while the model is loading.
    Progress(String),
    Ready { gpu_name: String },
    Token(String),
    Done { tokens_per_sec: f64 },
    Error(String),
}

enum InferenceRequest {
    /// Raw user message text; the inference thread formats the full prompt.
    Generate(String),
    /// Clear conversation history and KV cache.
    Reset,
}

// ── app state ───────────────────────────────────────────────────────────────

enum AppStatus {
    Loading,
    Ready,
    Generating,
    Error(String),
}

struct ChatApp {
    input: String,
    /// Display history: (role, content). Role is "user" or "assistant".
    history: Vec<(String, String)>,
    /// Response currently streaming in from the inference thread.
    current_response: String,
    status: AppStatus,
    /// Current load stage (fetching / parsing / compiling / …), shown while loading.
    loading_message: String,
    gpu_name: String,
    last_tokens_per_sec: Option<f64>,

    request_tx: async_mpsc::UnboundedSender<InferenceRequest>,
    event_rx: mpsc::Receiver<InferenceEvent>,
    /// Set by the UI to ask the inference thread to stop generating early.
    /// Shared with the inference thread, which polls it between decode steps.
    stop_flag: Arc<AtomicBool>,
}

impl ChatApp {
    fn new(cc: &eframe::CreationContext, source: ModelSource) -> Self {
        let (request_tx, request_rx) = async_mpsc::unbounded::<InferenceRequest>();
        let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();
        let egui_ctx = cc.egui_ctx.clone();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let task_stop_flag = Arc::clone(&stop_flag);

        // Drive the async inference loop. On native it runs on a dedicated
        // thread (blocking executor); on web it runs as a spawned task on the
        // single browser thread, yielding to the UI at each `.await`.
        //
        // The runtime's futures are `?Send` (they hold wgpu/tracing state across
        // awaits), so the native future is built *inside* the thread closure —
        // only the (Send) arguments cross the thread boundary, not the future.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            pollster::block_on(run_inference(
                source,
                request_rx,
                event_tx,
                egui_ctx,
                task_stop_flag,
            ));
        });
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(run_inference(
            source,
            request_rx,
            event_tx,
            egui_ctx,
            task_stop_flag,
        ));

        Self {
            input: String::new(),
            history: Vec::new(),
            current_response: String::new(),
            status: AppStatus::Loading,
            loading_message: "Starting…".to_string(),
            gpu_name: String::new(),
            last_tokens_per_sec: None,
            request_tx,
            event_rx,
            stop_flag,
        }
    }

    fn submit(&mut self) {
        let msg = std::mem::take(&mut self.input);
        if msg.is_empty() {
            return;
        }
        self.history.push(("user".to_string(), msg.clone()));
        self.status = AppStatus::Generating;
        let _ = self.request_tx.unbounded_send(InferenceRequest::Generate(msg));
    }

    fn new_conversation(&mut self) {
        self.history.clear();
        self.current_response.clear();
        self.last_tokens_per_sec = None;
        let _ = self.request_tx.unbounded_send(InferenceRequest::Reset);
    }
}

impl eframe::App for ChatApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Once egui is painting, remove the HTML placeholder overlay (it sits
        // above the canvas and would otherwise cover the UI and swallow clicks).
        #[cfg(target_arch = "wasm32")]
        if let Some(el) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("loading"))
        {
            el.remove();
        }

        // Drain all pending events from the inference thread.
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                InferenceEvent::Progress(msg) => {
                    self.loading_message = msg;
                }
                InferenceEvent::Ready { gpu_name } => {
                    self.gpu_name = gpu_name;
                    self.status = AppStatus::Ready;
                }
                InferenceEvent::Token(text) => {
                    self.current_response.push_str(&text);
                }
                InferenceEvent::Done { tokens_per_sec } => {
                    self.history.push((
                        "assistant".to_string(),
                        std::mem::take(&mut self.current_response),
                    ));
                    self.last_tokens_per_sec = Some(tokens_per_sec);
                    self.status = AppStatus::Ready;
                }
                InferenceEvent::Error(e) => {
                    self.status = AppStatus::Error(e);
                }
            }
        }

        let is_ready = matches!(self.status, AppStatus::Ready);

        // ── header ──────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("header").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Gemma 3 270M · Onyxia");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    match &self.status {
                        AppStatus::Loading => {
                            ui.label(&self.loading_message);
                            ui.spinner();
                        }
                        AppStatus::Ready => {
                            let label = match self.last_tokens_per_sec {
                                Some(tps) => format!("{} · {:.1} tok/s", self.gpu_name, tps),
                                None => self.gpu_name.clone(),
                            };
                            ui.label(label);
                        }
                        AppStatus::Generating => {
                            ui.label("Generating…");
                            ui.spinner();
                        }
                        AppStatus::Error(e) => {
                            ui.colored_label(egui::Color32::RED, e.as_str());
                        }
                    }
                });
            });
        });

        // ── input panel ─────────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("input_panel").show_inside(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let input_resp = ui.add_enabled(
                    is_ready,
                    egui::TextEdit::singleline(&mut self.input)
                        .hint_text("Message Gemma…")
                        .desired_width(ui.available_width() - 110.0),
                );

                let send_clicked = ui
                    .add_enabled(
                        is_ready && !self.input.is_empty(),
                        egui::Button::new("Send"),
                    )
                    .clicked();

                let enter_pressed = input_resp.lost_focus()
                    && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));

                if (send_clicked || enter_pressed) && is_ready && !self.input.is_empty() {
                    self.submit();
                    // Re-focus the input field for the next message.
                    input_resp.request_focus();
                }

                let is_generating = matches!(self.status, AppStatus::Generating);
                if is_generating {
                    if ui.button("Stop").clicked() {
                        self.stop_flag.store(true, Ordering::Relaxed);
                    }
                } else if ui
                    .add_enabled(is_ready && !self.history.is_empty(), egui::Button::new("New"))
                    .clicked()
                {
                    self.new_conversation();
                }
            });
            ui.add_space(6.0);
        });

        // ── chat history ────────────────────────────────────────────────────
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let is_empty = self.history.is_empty() && self.current_response.is_empty();

            if matches!(self.status, AppStatus::Loading) && is_empty {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Loading Gemma 3 270M…\n{}", self.loading_message))
                            .weak(),
                    );
                });
                return;
            }

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add_space(8.0);

                    for (role, content) in &self.history {
                        let speaker = if role == "user" { "You" } else { "Gemma" };
                        ui.label(egui::RichText::new(format!("{speaker}:")).strong());
                        ui.label(content);
                        ui.add_space(12.0);
                    }

                    if !self.current_response.is_empty() {
                        ui.label(egui::RichText::new("Gemma:").strong());
                        ui.label(&self.current_response);
                    }
                });
        });

        // Keep repainting while the inference thread is active.
        if matches!(self.status, AppStatus::Generating | AppStatus::Loading) {
            ui.ctx().request_repaint();
        }
    }
}

// ── inference thread ────────────────────────────────────────────────────────

async fn run_inference(
    source: ModelSource,
    mut request_rx: async_mpsc::UnboundedReceiver<InferenceRequest>,
    event_tx: mpsc::Sender<InferenceEvent>,
    egui_ctx: egui::Context,
    stop_flag: Arc<AtomicBool>,
) {
    let send = |event: InferenceEvent| {
        let _ = event_tx.send(event);
        egui_ctx.request_repaint();
    };
    let progress = |msg: &str| {
        let _ = event_tx.send(InferenceEvent::Progress(msg.to_string()));
        egui_ctx.request_repaint();
    };

    let (mut session, tokenizer, gpu_name) = match load(source, &progress).await {
        Ok(t) => t,
        Err(e) => {
            send(InferenceEvent::Error(format!("Failed to load model: {e:#}")));
            return;
        }
    };

    send(InferenceEvent::Ready { gpu_name });

    let sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.0,
        seed: None,
    };

    // Conversation history, maintained across turns for multi-turn chat.
    let mut conversation: Vec<ChatMessage> = Vec::new();

    while let Some(request) = request_rx.next().await {
        match request {
            InferenceRequest::Reset => {
                conversation.clear();
                session.reset_full();
            }

            InferenceRequest::Generate(user_msg) => {
                // Clear any stop request left over from a previous generation.
                stop_flag.store(false, Ordering::Relaxed);

                conversation.push(ChatMessage {
                    role: "user".to_string(),
                    content: user_msg,
                });

                // Apply chat template to the full conversation history.
                let prompt = match tokenizer.apply_chat_template(&conversation, true) {
                    Ok(p) => p,
                    Err(e) => {
                        send(InferenceEvent::Error(format!("Chat template error: {e}")));
                        conversation.pop();
                        continue;
                    }
                };

                // Each turn re-prefills from the full formatted conversation
                // so we need a clean KV cache.
                session.reset_full();

                let input_ids = match tokenizer.encode(&prompt, false) {
                    Ok(ids) => ids,
                    Err(e) => {
                        send(InferenceEvent::Error(format!("Tokenize error: {e}")));
                        conversation.pop();
                        continue;
                    }
                };

                let logits = match session.prefill(&input_ids).await {
                    Ok(l) => l,
                    Err(e) => {
                        send(InferenceEvent::Error(format!("Prefill failed: {e}")));
                        conversation.pop();
                        continue;
                    }
                };

                let mut rng = StdRng::from_seed(rand::random());
                let mut token = sample(&logits, &sampling, &mut rng);
                let mut response_tokens: Vec<i64> = Vec::new();

                let decode_start = Instant::now();
                let mut errored = false;

                for _ in 0..MAX_TOKENS {
                    // Stop *before* emitting a terminator (EOS or Gemma's
                    // <end_of_turn>) so it is never shown or stored.
                    if tokenizer.is_eos(token as i64) {
                        break;
                    }

                    // Honor a stop request from the UI, finalizing the partial
                    // response generated so far.
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    response_tokens.push(token as i64);
                    if let Ok(text) = tokenizer.decode(&[token as i64], false) {
                        send(InferenceEvent::Token(text));
                    }

                    // Generate the next token conditioned on the one just emitted.
                    let logits = match session.decode(token as i64).await {
                        Ok(l) => l,
                        Err(e) => {
                            send(InferenceEvent::Error(format!("Decode failed: {e}")));
                            errored = true;
                            break;
                        }
                    };
                    token = sample(&logits, &sampling, &mut rng);
                }

                if errored {
                    conversation.pop();
                    continue;
                }

                let decode_time = decode_start.elapsed().as_secs_f64();
                let tokens_per_sec = response_tokens.len() as f64 / decode_time.max(1e-9);

                // Store the assistant turn in the conversation history.
                let response_text = tokenizer
                    .decode(&response_tokens, true)
                    .unwrap_or_default();
                conversation.push(ChatMessage {
                    role: "model".to_string(),
                    content: response_text,
                });

                send(InferenceEvent::Done { tokens_per_sec });
            }
        }
    }
}

// ── model loading ────────────────────────────────────────────────────────────

// The demo uses the full-precision `model.onnx`. The community `model_q4.onnx`
// 4-bit quantization badly degrades this small model (verified against
// onnxruntime: fp32 recalls long-context facts, q4 collapses into garbage).

/// Yield to the browser event loop so the UI repaints between blocking load
/// stages. No-op on native, where the inference loop has its own thread.
#[cfg(target_arch = "wasm32")]
async fn yield_to_browser() {
    gloo_timers::future::TimeoutFuture::new(0).await;
}
#[cfg(not(target_arch = "wasm32"))]
async fn yield_to_browser() {}

/// Build a session from an already-parsed graph + tokenizer (shared by both
/// platforms). Compilation and GPU init are async so this works on the web.
async fn build_session(
    graph: onyxia_onnx::Graph,
    tokenizer: Tokenizer,
    progress: &dyn Fn(&str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress("Initializing GPU…");
    yield_to_browser().await;
    let runtime = onyxia_runtime::Runtime::new()
        .await
        .context("Failed to initialise GPU runtime")?;
    let gpu_name = runtime.adapter_info().name.clone();

    progress("Compiling model…");
    yield_to_browser().await;
    let registry = onyxia_operators::core_operator_registry();
    let compiled = onyxia_compiler::compile_async(&graph, &registry, runtime.gpu())
        .await
        .context("Compilation failed")?;

    progress("Uploading weights to GPU…");
    yield_to_browser().await;
    let executor = runtime.load_model(compiled).context("Failed to load model")?;
    let session = LlmSession::new(
        executor,
        &LlmConfig {
            max_seq_len: MAX_SEQ_LEN,
            num_layers: NUM_LAYERS,
        },
    );
    Ok((session, tokenizer, gpu_name))
}

/// Native: read the model + tokenizer from a directory on disk.
#[cfg(not(target_arch = "wasm32"))]
async fn load(
    model_dir: PathBuf,
    progress: &dyn Fn(&str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress("Reading model…");
    let onnx_path = model_dir.join("onnx/model.onnx");
    let graph = onyxia_onnx::load_and_parse_model(&onnx_path)
        .with_context(|| format!("Failed to parse ONNX from {}", onnx_path.display()))?;

    let tokenizer_file = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file).with_context(|| {
        format!("Failed to load tokenizer from {}", tokenizer_file.display())
    })?;
    let template_file = model_dir.join("chat_template.jinja");
    if template_file.exists() {
        tokenizer = tokenizer
            .with_chat_template_file(&template_file)
            .context("Failed to load chat template")?;
    }

    build_session(graph, tokenizer, progress).await
}

/// Web: fetch the model + tokenizer over HTTP relative to `base_url`.
#[cfg(target_arch = "wasm32")]
async fn load(
    base_url: String,
    progress: &dyn Fn(&str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress("Fetching model… (~1.1 GB, first load is slow)");
    yield_to_browser().await;
    let model_bytes = fetch_bytes(&format!("{base_url}/onnx/model.onnx")).await?;
    let data_bytes = fetch_bytes(&format!("{base_url}/onnx/model.onnx_data")).await?;

    progress("Parsing model…");
    yield_to_browser().await;
    let mut external = std::collections::HashMap::new();
    external.insert("model.onnx_data".to_string(), data_bytes);
    let graph = onyxia_onnx::parse_model_from_bytes(&model_bytes, external)
        .context("Failed to parse ONNX model")?;

    progress("Fetching tokenizer…");
    yield_to_browser().await;
    let tok_bytes = fetch_bytes(&format!("{base_url}/tokenizer.json")).await?;
    let mut tokenizer =
        Tokenizer::from_bytes(&tok_bytes).context("Failed to load tokenizer")?;
    if let Ok(template) = fetch_string(&format!("{base_url}/chat_template.jinja")).await {
        tokenizer = tokenizer.with_chat_template(template);
    }

    build_session(graph, tokenizer, progress).await
}

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("fetch {url} failed: {e}"))?;
    if !resp.ok() {
        anyhow::bail!("fetch {url} failed: HTTP {}", resp.status());
    }
    resp.binary()
        .await
        .map_err(|e| anyhow::anyhow!("read {url} failed: {e}"))
}

#[cfg(target_arch = "wasm32")]
async fn fetch_string(url: &str) -> Result<String> {
    let resp = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("fetch {url} failed: {e}"))?;
    if !resp.ok() {
        anyhow::bail!("fetch {url} failed: HTTP {}", resp.status());
    }
    resp.text()
        .await
        .map_err(|e| anyhow::anyhow!("read {url} failed: {e}"))
}

// ── entry point ──────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init();

    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-3-270m-it-ONNX"));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Gemma 3 270M · Onyxia")
            .with_inner_size([720.0, 540.0]),
        ..Default::default()
    };

    eframe::run_native(
        "gemma-chat",
        options,
        Box::new(move |cc| Ok(Box::new(ChatApp::new(cc, model_dir)))),
    )
}

/// Web entry point. Trunk calls `main`; we start eframe on the `#canvas`
/// element and fetch the model relative to the served page (`.`).
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document")
            .get_element_by_id("canvas")
            .expect("no element with id 'canvas'")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("'canvas' element is not a <canvas>");

        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(ChatApp::new(cc, ".".to_string())))),
            )
            .await
            .expect("failed to start eframe");
    });
}
