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
//!   onnx/model_q4.onnx      — quantised ONNX model
//!   tokenizer.json           — HuggingFace tokenizer
//!   chat_template.jinja      — Jinja2 chat template (optional)

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result};
use eframe::egui;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod inference;
mod sampling;
mod tokenizer;

use inference::{LlmConfig, LlmSession};
use sampling::{SamplingConfig, sample};
use tokenizer::{ChatMessage, Tokenizer};

const MAX_TOKENS: usize = 512;
const MAX_SEQ_LEN: usize = 2048;
const NUM_LAYERS: usize = 26;

// ── inter-thread communication ──────────────────────────────────────────────

enum InferenceEvent {
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
    gpu_name: String,
    last_tokens_per_sec: Option<f64>,

    request_tx: mpsc::Sender<InferenceRequest>,
    event_rx: mpsc::Receiver<InferenceEvent>,
}

impl ChatApp {
    fn new(cc: &eframe::CreationContext, model_dir: PathBuf) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>();
        let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();
        let egui_ctx = cc.egui_ctx.clone();

        std::thread::spawn(move || {
            inference_thread(model_dir, request_rx, event_tx, egui_ctx);
        });

        Self {
            input: String::new(),
            history: Vec::new(),
            current_response: String::new(),
            status: AppStatus::Loading,
            gpu_name: String::new(),
            last_tokens_per_sec: None,
            request_tx,
            event_rx,
        }
    }

    fn submit(&mut self) {
        let msg = std::mem::take(&mut self.input);
        if msg.is_empty() {
            return;
        }
        self.history.push(("user".to_string(), msg.clone()));
        self.status = AppStatus::Generating;
        let _ = self.request_tx.send(InferenceRequest::Generate(msg));
    }

    fn new_conversation(&mut self) {
        self.history.clear();
        self.current_response.clear();
        self.last_tokens_per_sec = None;
        let _ = self.request_tx.send(InferenceRequest::Reset);
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain all pending events from the inference thread.
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
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
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Gemma 3 270M · Onyxia");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    match &self.status {
                        AppStatus::Loading => {
                            ui.label("Loading model…");
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
        egui::TopBottomPanel::bottom("input_panel").show(ctx, |ui| {
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
                    && ctx.input(|i| i.key_pressed(egui::Key::Enter));

                if (send_clicked || enter_pressed) && is_ready && !self.input.is_empty() {
                    self.submit();
                    // Re-focus the input field for the next message.
                    input_resp.request_focus();
                }

                if ui
                    .add_enabled(is_ready && !self.history.is_empty(), egui::Button::new("New"))
                    .clicked()
                {
                    self.new_conversation();
                }
            });
            ui.add_space(6.0);
        });

        // ── chat history ────────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let is_empty = self.history.is_empty() && self.current_response.is_empty();

            if matches!(self.status, AppStatus::Loading) && is_empty {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Loading Gemma 3 270M…").weak());
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
            ctx.request_repaint();
        }
    }
}

// ── inference thread ────────────────────────────────────────────────────────

fn inference_thread(
    model_dir: PathBuf,
    request_rx: mpsc::Receiver<InferenceRequest>,
    event_tx: mpsc::Sender<InferenceEvent>,
    egui_ctx: egui::Context,
) {
    let send = |event: InferenceEvent| {
        let _ = event_tx.send(event);
        egui_ctx.request_repaint();
    };

    let result = load(model_dir);
    let (mut session, tokenizer, gpu_name) = match result {
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

    for request in &request_rx {
        match request {
            InferenceRequest::Reset => {
                conversation.clear();
                session.reset_full();
            }

            InferenceRequest::Generate(user_msg) => {
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

                let logits = match pollster::block_on(session.prefill(&input_ids)) {
                    Ok(l) => l,
                    Err(e) => {
                        send(InferenceEvent::Error(format!("Prefill failed: {e}")));
                        conversation.pop();
                        continue;
                    }
                };

                let mut rng = StdRng::from_seed(rand::random());
                let mut token = sample(&logits, &sampling, &mut rng);
                let mut response_tokens: Vec<i64> = vec![token as i64];

                if let Ok(text) = tokenizer.decode(&[token as i64], false) {
                    send(InferenceEvent::Token(text));
                }

                let decode_start = Instant::now();
                let mut errored = false;

                for _ in 1..MAX_TOKENS {
                    if tokenizer.is_eos(token as i64) {
                        break;
                    }

                    let logits = match pollster::block_on(session.decode(token as i64)) {
                        Ok(l) => l,
                        Err(e) => {
                            send(InferenceEvent::Error(format!("Decode failed: {e}")));
                            errored = true;
                            break;
                        }
                    };

                    token = sample(&logits, &sampling, &mut rng);
                    response_tokens.push(token as i64);

                    if let Ok(text) = tokenizer.decode(&[token as i64], false) {
                        send(InferenceEvent::Token(text));
                    }

                    if tokenizer.is_eos(token as i64) {
                        break;
                    }
                }

                if errored {
                    conversation.pop();
                    continue;
                }

                let decode_time = decode_start.elapsed().as_secs_f64();
                let decode_tokens = response_tokens.len().saturating_sub(1);
                let tokens_per_sec = decode_tokens as f64 / decode_time.max(1e-9);

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

fn load(model_dir: PathBuf) -> Result<(LlmSession, Tokenizer, String)> {
    let onnx_path = model_dir.join("onnx/model_q4.onnx");

    let graph = onyxia_onnx::load_and_parse_model(&onnx_path)
        .with_context(|| format!("Failed to parse ONNX from {}", onnx_path.display()))?;

    let runtime = pollster::block_on(onyxia_runtime::Runtime::new())
        .context("Failed to initialise GPU runtime")?;

    let gpu_name = runtime.adapter_info().name.clone();

    let registry = onyxia_operators::core_operator_registry();
    let compiled = onyxia_compiler::compile(&graph, &registry, runtime.gpu())
        .context("Compilation failed")?;

    let executor = runtime.load_model(compiled).context("Failed to load model")?;

    let config = LlmConfig {
        max_seq_len: MAX_SEQ_LEN,
        num_layers: NUM_LAYERS,
    };
    let session = LlmSession::new(executor, &config);

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

    Ok((session, tokenizer, gpu_name))
}

// ── entry point ──────────────────────────────────────────────────────────────

fn main() -> eframe::Result {
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
