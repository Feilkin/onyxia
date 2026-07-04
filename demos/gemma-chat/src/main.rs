//! Gemma 3 270M chat demo built on Onyxia.
//!
//! This is an example application showing how to drive the Onyxia ONNX runtime
//! for autoregressive text generation. All Gemma-specific logic (KV cache
//! management, chat template formatting, token sampling) lives here in the
//! application layer — the runtime itself knows nothing about LLMs.
//!
//! Usage:
//!
//! ```text
//! gemma-chat <model-dir>
//! ```
//!
//! Where `<model-dir>` contains:
//!
//! ```text
//! onnx/model.onnx          — full-precision ONNX model (+ .onnx_data)
//! tokenizer.json           — HuggingFace tokenizer
//! chat_template.jinja      — Jinja2 chat template (optional)
//! ```

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
mod theme;
mod tokenizer;

#[cfg(not(target_arch = "wasm32"))]
mod demo;
#[cfg(not(target_arch = "wasm32"))]
mod screenshot;

use inference::LlmSession;
use sampling::{SamplingConfig, sample};
use theme::{JB_SEMIBOLD, SG_BOLD, SG_MEDIUM, Theme, family, mono};
use tokenizer::{ChatMessage, Tokenizer};

use egui::{Align, CornerRadius, FontId, Layout, Margin, Rect, RichText, Sense, Stroke, vec2};

/// Where to load the model from: a directory on native, a base URL on web.
#[cfg(not(target_arch = "wasm32"))]
type ModelSource = PathBuf;
#[cfg(target_arch = "wasm32")]
type ModelSource = String;

const MAX_TOKENS: usize = 512;
const MAX_SEQ_LEN: usize = 2048;

// ── inter-thread communication ──────────────────────────────────────────────

enum InferenceEvent {
    /// A load-stage update shown while the model is loading. `stage` indexes
    /// into [`theme::STAGE_LABELS`] so the segmented progress bar can fill.
    Progress { stage: usize, label: String },
    Ready { gpu_name: String, vram_bytes: u64 },
    Token(String),
    Done { tokens_per_sec: f64, ttft_ms: f64, vram_bytes: u64 },
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
    /// Index into [`theme::STAGE_LABELS`] for the segmented progress bar.
    loading_stage: usize,
    gpu_name: String,
    last_tokens_per_sec: Option<f64>,
    /// Time-to-first-token for the most recent answer.
    last_ttft_ms: Option<f64>,
    /// Resident GPU bytes reported by the session (weights + KV cache +
    /// buffer pool; 0 until ready, refreshed after every answer).
    vram_bytes: u64,
    /// Active visual theme (night / day), toggled from the header.
    theme: Theme,

    request_tx: async_mpsc::UnboundedSender<InferenceRequest>,
    event_rx: mpsc::Receiver<InferenceEvent>,
    /// Set by the UI to ask the inference thread to stop generating early.
    /// Shared with the inference thread, which polls it between decode steps.
    stop_flag: Arc<AtomicBool>,

    /// When set (via `--shots`), drives a scripted screenshot sequence.
    #[cfg(not(target_arch = "wasm32"))]
    shots: Option<screenshot::Shots>,
}

impl ChatApp {
    fn new(cc: &eframe::CreationContext, source: ModelSource, demo: bool) -> Self {
        let (request_tx, request_rx) = async_mpsc::unbounded::<InferenceRequest>();
        let (event_tx, event_rx) = mpsc::channel::<InferenceEvent>();
        let egui_ctx = cc.egui_ctx.clone();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let task_stop_flag = Arc::clone(&stop_flag);

        // Install the custom fonts and night-theme visuals up front.
        let theme = Theme::Night;
        theme::install_fonts(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(theme::visuals(&theme.palette(), theme));

        // Drive the async inference loop. On native it runs on a dedicated
        // thread (blocking executor); on web it runs as a spawned task on the
        // single browser thread, yielding to the UI at each `.await`.
        //
        // The runtime's futures are `?Send` (they hold wgpu/tracing state across
        // awaits), so the native future is built *inside* the thread closure —
        // only the (Send) arguments cross the thread boundary, not the future.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            pollster::block_on(async move {
                if demo {
                    // Scripted stages + canned answer, no model required.
                    demo::run(request_rx, event_tx, egui_ctx, task_stop_flag).await;
                } else {
                    run_inference(source, request_rx, event_tx, egui_ctx, task_stop_flag).await;
                }
            });
        });
        #[cfg(target_arch = "wasm32")]
        {
            let _ = demo; // demo mode is native-only
            wasm_bindgen_futures::spawn_local(run_inference(
                source,
                request_rx,
                event_tx,
                egui_ctx,
                task_stop_flag,
            ));
        }

        Self {
            input: String::new(),
            history: Vec::new(),
            current_response: String::new(),
            status: AppStatus::Loading,
            loading_message: theme::STAGE_LABELS[1].to_string(),
            loading_stage: 1,
            gpu_name: String::new(),
            last_tokens_per_sec: None,
            last_ttft_ms: None,
            vram_bytes: 0,
            theme,
            request_tx,
            event_rx,
            stop_flag,
            #[cfg(not(target_arch = "wasm32"))]
            shots: None,
        }
    }

    /// Model line shown in the header, e.g. `Gemma 3 270M · fp32 · <gpu>`.
    fn model_info(&self) -> String {
        let device = if self.gpu_name.is_empty() {
            "WebGPU".to_string()
        } else {
            self.gpu_name.clone()
        };
        format!("Gemma 3 270M · fp32 · {device}")
    }

    fn set_theme(&mut self, ctx: &egui::Context, theme: Theme) {
        if self.theme != theme {
            self.theme = theme;
            ctx.set_visuals(theme::visuals(&theme.palette(), theme));
        }
    }

    /// Configure the app state for screenshot `idx` (see [`screenshot`]).
    #[cfg(not(target_arch = "wasm32"))]
    fn setup_shot(&mut self, ctx: &egui::Context, idx: usize) {
        const CANNED_Q: &str = "How does Onyxia run models in the browser?";
        const CANNED_A: &str = "Onyxia lowers your ONNX graph into a compact intermediate \
representation, then compiles every operator to a WebGPU compute shader ahead of time. The \
quantized weights are uploaded to the GPU once and stay resident, so each token is generated \
entirely on-device — no round-trips back to JavaScript, and no network calls after the model \
has loaded.";

        self.set_theme(ctx, if idx <= 3 { Theme::Night } else { Theme::Day });
        let user = |t: &str| ("user".to_string(), t.to_string());
        let bot = |t: &str| ("assistant".to_string(), t.to_string());
        match screenshot::SHOTS.get(idx).copied() {
            Some("night_loading") => {
                self.status = AppStatus::Loading;
                self.loading_stage = 4;
                self.loading_message = theme::STAGE_LABELS[4].to_string();
                self.history.clear();
                self.current_response.clear();
            }
            Some("night_empty") | Some("day_empty") => {
                self.status = AppStatus::Ready;
                self.history.clear();
                self.current_response.clear();
                self.last_tokens_per_sec = None;
            }
            Some("night_chat") | Some("day_chat") => {
                self.status = AppStatus::Ready;
                self.history = vec![user(CANNED_Q), bot(CANNED_A)];
                self.current_response.clear();
                self.last_tokens_per_sec = Some(47.0);
                self.last_ttft_ms = Some(214.0);
                self.vram_bytes = 1_181_116_006; // ~1.1 GiB
                self.gpu_name = "Demo GPU · WebGPU".to_string();
            }
            Some("night_streaming") => {
                self.status = AppStatus::Generating;
                self.history = vec![user(CANNED_Q)];
                self.current_response =
                    CANNED_A.split(' ').take(18).collect::<Vec<_>>().join(" ");
                self.last_tokens_per_sec = None;
            }
            _ => {}
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
                InferenceEvent::Progress { stage, label } => {
                    self.loading_stage = stage;
                    self.loading_message = label;
                }
                InferenceEvent::Ready { gpu_name, vram_bytes } => {
                    self.gpu_name = gpu_name;
                    self.vram_bytes = vram_bytes;
                    self.status = AppStatus::Ready;
                }
                InferenceEvent::Token(text) => {
                    self.current_response.push_str(&text);
                }
                InferenceEvent::Done { tokens_per_sec, ttft_ms, vram_bytes } => {
                    self.history.push((
                        "assistant".to_string(),
                        std::mem::take(&mut self.current_response),
                    ));
                    self.last_tokens_per_sec = Some(tokens_per_sec);
                    self.last_ttft_ms = Some(ttft_ms);
                    self.vram_bytes = vram_bytes;
                    self.status = AppStatus::Ready;
                }
                InferenceEvent::Error(e) => {
                    self.status = AppStatus::Error(e);
                }
            }
        }

        // Screenshot harness: override state for the current shot, then request
        // a capture of this frame. Taken out and put back so `setup_shot` can
        // mutate `self` freely.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(mut shots) = self.shots.take() {
            let ctx = ui.ctx().clone();
            if shots.index >= screenshot::SHOTS.len() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.setup_shot(&ctx, shots.index);
                if !shots.requested {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                        egui::UserData::default(),
                    ));
                    shots.requested = true;
                }
            }
            self.shots = Some(shots);
        }

        let pal = self.theme.palette();
        let ctx = ui.ctx().clone();

        // ── header ──────────────────────────────────────────────────────────
        egui::Panel::top("header")
            .frame(egui::Frame::new().fill(pal.bg).inner_margin(Margin::symmetric(28, 16)))
            .show_inside(ui, |ui| self.header(ui, &ctx, &pal));

        // ── composer ────────────────────────────────────────────────────────
        egui::Panel::bottom("composer")
            .frame(egui::Frame::new().fill(pal.bg).inner_margin(Margin::symmetric(28, 16)))
            .show_inside(ui, |ui| self.composer(ui, &pal));

        // ── main region: loading panel or chat ──────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(pal.bg).inner_margin(Margin::symmetric(40, 0)))
            .show_inside(ui, |ui| match &self.status {
                AppStatus::Loading => self.loading_panel(ui, &pal),
                AppStatus::Error(e) => {
                    let msg = e.clone();
                    error_panel(ui, &pal, &msg);
                }
                _ => self.chat_panel(ui, &pal),
            });

        // Screenshot harness: save the captured frame and advance to the next.
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(mut shots) = self.shots.take() {
            let mut captured: Option<std::sync::Arc<egui::ColorImage>> = None;
            ui.input(|i| {
                for event in &i.raw.events {
                    if let egui::Event::Screenshot { image, .. } = event {
                        captured = Some(image.clone());
                    }
                }
            });
            if let Some(image) = captured {
                screenshot::save(&shots.dir, shots.index, screenshot::SHOTS[shots.index], &image);
                shots.index += 1;
                shots.requested = false;
            }
            ui.ctx().request_repaint();
            self.shots = Some(shots);
        }

        // Keep animating while loading (segment sweep) or streaming (caret).
        if matches!(self.status, AppStatus::Generating | AppStatus::Loading) {
            ui.ctx().request_repaint();
        }
    }
}

// ── UI: regions ───────────────────────────────────────────────────────────────

impl ChatApp {
    /// Logo + wordmark + model line on the left; reload + theme toggle on the right.
    fn header(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, pal: &theme::Palette) {
        ui.horizontal(|ui| {
            ui.set_height(30.0);
            let (logo_rect, _) = ui.allocate_exact_size(vec2(26.0, 26.0), Sense::hover());
            theme::logo(ui.painter(), logo_rect, pal.logo_a, pal.logo_b);
            ui.add_space(12.0);
            ui.label(
                RichText::new("Onyxia")
                    .family(family(SG_BOLD))
                    .size(28.0)
                    .color(pal.text),
            );
            ui.add_space(12.0);
            ui.label(
                RichText::new(self.model_info())
                    .family(mono())
                    .size(15.0)
                    .color(pal.muted),
            );

            // right_to_left places the first-added widget rightmost.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                // Theme toggle: a segmented control of two icon buttons.
                egui::Frame::new()
                    .fill(pal.seg_dim)
                    .stroke(Stroke::new(1.0, pal.border))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::same(3))
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        if icon_button(ui, 34.0, self.theme == Theme::Day, pal, |p, r, fg, _| {
                            theme::icon_sun(p, r.shrink(8.0), fg)
                        })
                        .clicked()
                        {
                            self.set_theme(ctx, Theme::Day);
                        }
                        if icon_button(ui, 34.0, self.theme == Theme::Night, pal, |p, r, fg, carve| {
                            theme::icon_moon(p, r.shrink(8.0), fg, carve)
                        })
                        .clicked()
                        {
                            self.set_theme(ctx, Theme::Night);
                        }
                    });
                ui.add_space(10.0);
                if reload_button(ui, pal) {
                    self.new_conversation();
                }
            });
        });
    }

    /// The input row: text field + send, with a stop button while generating.
    fn composer(&mut self, ui: &mut egui::Ui, pal: &theme::Palette) {
        let is_ready = matches!(self.status, AppStatus::Ready);
        let is_generating = matches!(self.status, AppStatus::Generating);

        centered_column(ui, 960.0, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 12.0;
                let btn = 52.0;
                let stop_w = if is_generating { 88.0 } else { 0.0 };
                let input_w = (ui.available_width() - btn - stop_w - 12.0).max(80.0);

                let hint = if is_ready || is_generating {
                    "Ask Onyxia anything"
                } else {
                    "Model is loading…"
                };

                let mut submit = false;
                egui::Frame::new()
                    .fill(pal.input_bg)
                    .stroke(Stroke::new(1.0, pal.input_border))
                    .corner_radius(CornerRadius::same(14))
                    .inner_margin(Margin::symmetric(18, 0))
                    .show(ui, |ui| {
                        // Force the field to the 52px button height and center the
                        // single line of text within it.
                        ui.set_width(input_w);
                        ui.set_height(52.0);
                        ui.centered_and_justified(|ui| {
                            let resp = ui.add_enabled(
                                is_ready,
                                egui::TextEdit::singleline(&mut self.input)
                                    .frame(egui::Frame::NONE)
                                    // hint_text keeps its own atoms' size, so set 19 here too.
                                    .hint_text(RichText::new(hint).size(19.0))
                                    .font(FontId::new(19.0, egui::FontFamily::Proportional))
                                    .text_color(pal.text)
                                    .vertical_align(Align::Center),
                            );
                            if resp.lost_focus()
                                && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter))
                            {
                                submit = true;
                            }
                        });
                    });

                if is_generating && stop_button(ui, pal) {
                    self.stop_flag.store(true, Ordering::Relaxed);
                }

                let enabled = is_ready && !self.input.trim().is_empty();
                if send_button(ui, pal, enabled) {
                    submit = true;
                }
                if submit && enabled {
                    self.submit();
                }
            });
        });
    }

    /// Full-panel staged loader: step label, title, segmented bar, meta line.
    fn loading_panel(&mut self, ui: &mut egui::Ui, pal: &theme::Palette) {
        // Roughly vertically center the ~220px loader block.
        let push_down = ((ui.available_height() - 220.0) * 0.42).max(12.0);
        ui.add_space(push_down);
        centered_column(ui, 1100.0, |ui| {
            let step = (self.loading_stage + 1).min(theme::STAGE_LABELS.len());
            // egui has no letter-spacing / text-transform, so we uppercase by hand.
            ui.label(
                RichText::new(format!(
                    "LOADING MODEL — STEP {step} OF {}",
                    theme::STAGE_LABELS.len()
                ))
                .family(mono())
                .size(15.0)
                .color(pal.accent2),
            );
            ui.add_space(14.0);

            let label = theme::STAGE_LABELS[self.loading_stage.min(theme::STAGE_LABELS.len() - 1)];
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                let title = |t: &str, c| RichText::new(t).family(family(SG_BOLD)).size(52.0).color(c);
                ui.label(title(label, pal.text));
                ui.label(title("…", pal.accent));
            });
            ui.add_space(26.0);

            segbar(ui, pal, self.loading_stage);
            ui.add_space(18.0);

            let done = self.loading_stage.min(theme::STAGE_LABELS.len());
            let pct = (done as f32 / theme::STAGE_LABELS.len() as f32 * 100.0).round() as i32;
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{done} / {} stages complete", theme::STAGE_LABELS.len()))
                        .family(mono())
                        .size(15.0)
                        .color(pal.muted),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{pct}%"))
                            .family(family(JB_SEMIBOLD))
                            .size(15.0)
                            .color(pal.accent2),
                    );
                });
            });
        });
    }

    /// The chat scroll: empty state, or user/bot bubbles with a streaming tail.
    fn chat_panel(&mut self, ui: &mut egui::Ui, pal: &theme::Palette) {
        let empty = self.history.is_empty() && self.current_response.is_empty();
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.add_space(22.0);
                if empty {
                    empty_state(ui, pal);
                    return;
                }
                centered_column(ui, 960.0, |ui| {
                    ui.spacing_mut().item_spacing.y = 22.0;
                    let last = self.history.len().saturating_sub(1);
                    // The most recent answer carries a metrics line.
                    let metrics = self.last_tokens_per_sec.map(|tps| Metrics {
                        tps,
                        ttft_ms: self.last_ttft_ms.unwrap_or(0.0),
                        vram_bytes: self.vram_bytes,
                    });
                    for (i, (role, content)) in self.history.iter().enumerate() {
                        if role == "user" {
                            user_bubble(ui, pal, content);
                        } else {
                            let m = if i == last { metrics } else { None };
                            bot_bubble(ui, pal, content, false, m);
                        }
                    }
                    if !self.current_response.is_empty()
                        || matches!(self.status, AppStatus::Generating)
                    {
                        bot_bubble(ui, pal, &self.current_response, true, None);
                    }
                });
                ui.add_space(22.0);
            });
    }
}

// ── UI: widgets ───────────────────────────────────────────────────────────────

/// A fixed-width column (`width`), left-padded so it sits centered in `ui`.
fn centered_column(ui: &mut egui::Ui, width: f32, add: impl FnOnce(&mut egui::Ui)) {
    let avail = ui.available_width();
    let w = avail.min(width);
    let pad = ((avail - w) * 0.5).max(0.0);
    ui.horizontal_top(|ui| {
        ui.add_space(pad);
        ui.allocate_ui_with_layout(
            vec2(w, ui.available_height()),
            Layout::top_down(Align::Min),
            |ui| {
                ui.set_width(w);
                add(ui);
            },
        );
    });
}

/// A square icon button. `draw` receives (painter, rect, foreground, fill) —
/// `fill` lets the moon glyph carve with the button's own background.
fn icon_button(
    ui: &mut egui::Ui,
    size: f32,
    active: bool,
    pal: &theme::Palette,
    draw: impl Fn(&egui::Painter, Rect, egui::Color32, egui::Color32),
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click());
    let fill = if active {
        pal.accent
    } else if resp.hovered() {
        pal.border
    } else {
        egui::Color32::TRANSPARENT
    };
    if fill != egui::Color32::TRANSPARENT {
        ui.painter().rect_filled(rect.shrink(1.0), CornerRadius::same(9), fill);
    }
    let fg = if active { pal.user_text } else { pal.muted };
    let carve = if active { pal.accent } else { pal.bg };
    draw(ui.painter(), rect, fg, carve);
    resp
}

/// The bordered "↻ reload" pill. Returns true when clicked.
fn reload_button(ui: &mut egui::Ui, pal: &theme::Palette) -> bool {
    let inner = egui::Frame::new()
        .stroke(Stroke::new(1.0, pal.border))
        .corner_radius(CornerRadius::same(11))
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            // The header lays out right-to-left (to right-align this cluster),
            // which `ui.horizontal` inherits and would reverse icon/label. Pin an
            // explicit left-to-right sub-layout, sized to its content (14px icon +
            // 7px gap + ~44px label), so the icon stays first.
            ui.allocate_ui_with_layout(
                vec2(65.0, 14.0),
                Layout::left_to_right(Align::Center),
                |ui| {
                    ui.spacing_mut().item_spacing.x = 7.0;
                    let (ir, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                    theme::icon_reload(ui.painter(), ir, pal.muted);
                    ui.label(
                        RichText::new("reload")
                            .family(mono())
                            .size(13.0)
                            .color(pal.muted),
                    );
                },
            );
        });
    inner.response.interact(Sense::click()).clicked()
}

/// The square send button: an up-arrow on a diagonal gradient fill, dimmed
/// while disabled.
fn send_button(ui: &mut egui::Ui, pal: &theme::Palette, enabled: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(52.0, 52.0), Sense::click());
    let (mut a, mut b) = (pal.send_a, pal.send_b);
    let mut fg = pal.send_text;
    if !enabled {
        a = a.gamma_multiply(0.4);
        b = b.gamma_multiply(0.4);
        fg = fg.gamma_multiply(0.6);
    }
    theme::fill_grad_rrect(ui.painter(), rect, (14.0, 14.0, 14.0, 14.0), a, b);
    theme::icon_send(ui.painter(), rect.shrink(15.0), fg);
    enabled && resp.clicked()
}

/// The outlined "Stop" pill shown while generating.
fn stop_button(ui: &mut egui::Ui, pal: &theme::Palette) -> bool {
    let inner = egui::Frame::new()
        .stroke(Stroke::new(1.0, pal.stop_border))
        .corner_radius(CornerRadius::same(14))
        .inner_margin(Margin::symmetric(18, 0))
        .show(ui, |ui| {
            ui.set_height(52.0);
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Stop")
                        .family(family(SG_BOLD))
                        .size(17.0)
                        .color(pal.stop),
                );
            });
        });
    inner.response.interact(Sense::click()).clicked()
}

/// The eight-segment progress bar. Completed segments fill solid; the active
/// segment carries a highlight that sweeps left-to-right to signal activity.
fn segbar(ui: &mut egui::Ui, pal: &theme::Palette, stage: usize) {
    let n = theme::STAGE_LABELS.len();
    let gap = 7.0;
    let total = ui.available_width();
    let seg_w = ((total - gap * (n as f32 - 1.0)) / n as f32).max(1.0);
    let h = 14.0;
    let (rect, _) = ui.allocate_exact_size(vec2(total, h), Sense::hover());
    let p = ui.painter();
    let time = ui.input(|i| i.time) as f32;
    for i in 0..n {
        let x = rect.left() + i as f32 * (seg_w + gap);
        let seg = Rect::from_min_size(egui::pos2(x, rect.top()), vec2(seg_w, h));
        p.rect_filled(seg, CornerRadius::same(7), pal.seg_dim);
        if i < stage {
            p.rect_filled(seg, CornerRadius::same(7), pal.done);
        } else if i == stage {
            let t = (time * 0.8).fract();
            let hw = seg_w * 0.5;
            let cx = seg.left() - hw + (seg_w + hw * 2.0) * t;
            let hl = Rect::from_center_size(egui::pos2(cx, seg.center().y), vec2(hw, h)).intersect(seg);
            if hl.width() > 0.5 {
                p.rect_filled(hl, CornerRadius::same(7), pal.accent);
            }
        }
    }
}

/// A centered error message (model load / inference failure).
fn error_panel(ui: &mut egui::Ui, pal: &theme::Palette, msg: &str) {
    ui.add_space((ui.available_height() * 0.2).min(120.0));
    centered_column(ui, 720.0, |ui| {
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("Something went wrong")
                    .family(family(SG_BOLD))
                    .size(26.0)
                    .color(pal.stop),
            );
            ui.add_space(10.0);
            ui.label(
                RichText::new(msg)
                    .family(mono())
                    .size(14.0)
                    .color(pal.muted),
            );
        });
    });
}

/// The pre-conversation hero: logo + heading + subtitle, centered.
fn empty_state(ui: &mut egui::Ui, pal: &theme::Palette) {
    ui.add_space(60.0);
    ui.vertical_centered(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(52.0, 52.0), Sense::hover());
        theme::logo(ui.painter(), r, pal.logo_a, pal.logo_b);
        ui.add_space(16.0);
        ui.label(
            RichText::new("Ask Onyxia anything")
                .family(family(SG_BOLD))
                .size(34.0)
                .color(pal.text),
        );
        ui.add_space(8.0);
        ui.label(
            // Subtitle: regular (400) weight, lighter than the heading above.
            RichText::new("Running locally on your GPU via WebGPU. Nothing leaves this device.")
                .size(19.0)
                .color(pal.muted),
        );
    });
}

/// A right-aligned user message bubble (500-weight text, gradient fill).
fn user_bubble(ui: &mut egui::Ui, pal: &theme::Palette, text: &str) {
    ui.with_layout(Layout::top_down(Align::Max), |ui| {
        let pad = vec2(24.0, 15.0);
        let maxw = ui.available_width() * 0.74;
        // Measure the wrapped text, size the bubble to it, paint gradient + text.
        let font = FontId::new(21.0, family(SG_MEDIUM));
        let galley =
            ui.painter()
                .layout(text.to_string(), font, pal.user_text, maxw - pad.x * 2.0);
        let (rect, _) = ui.allocate_exact_size(galley.rect.size() + pad * 2.0, Sense::hover());
        theme::fill_grad_rrect(
            ui.painter(),
            rect,
            (20.0, 20.0, 5.0, 20.0),
            pal.user_a,
            pal.user_b,
        );
        ui.painter().galley(rect.min + pad, galley, pal.user_text);
    });
}

/// Per-answer metrics shown under a completed bot bubble.
#[derive(Clone, Copy)]
struct Metrics {
    tps: f64,
    ttft_ms: f64,
    vram_bytes: u64,
}

/// A left-aligned bot bubble, with an optional streaming caret and metrics line.
fn bot_bubble(
    ui: &mut egui::Ui,
    pal: &theme::Palette,
    text: &str,
    streaming: bool,
    metrics: Option<Metrics>,
) {
    ui.with_layout(Layout::top_down(Align::Min), |ui| {
        let maxw = ui.available_width() * 0.82;
        let mut frame = egui::Frame::new()
            .fill(pal.bot_bg)
            .stroke(Stroke::new(1.0, pal.border))
            .corner_radius(CornerRadius { nw: 20, ne: 20, se: 20, sw: 5 })
            .inner_margin(Margin::symmetric(26, 18));
        if pal.bot_shadow {
            frame = frame.shadow(egui::epaint::Shadow {
                offset: [0, 6],
                blur: 18,
                spread: 0,
                color: egui::Color32::from_black_alpha(30),
            });
        }
        frame.show(ui, |ui| {
            ui.set_max_width(maxw);
            // Lay out the text ourselves so we can paint a caret glyph-free at the
            // end of the last line (a block char would tofu in Space Grotesk).
            // Body text is regular (400) weight (headings/user text are heavier).
            let wrap_w = ui.available_width();
            let font = FontId::new(21.0, egui::FontFamily::Proportional);
            let galley = ui.painter().layout(text.to_string(), font, pal.text, wrap_w);
            let (rect, _) = ui.allocate_exact_size(galley.rect.size(), Sense::hover());
            ui.painter().galley(rect.min, galley.clone(), pal.text);
            if streaming {
                // Blink a thin accent caret ~twice a second.
                if ui.input(|i| i.time).fract() < 0.5
                    && let Some(row) = galley.rows.last()
                {
                    let rr = row.rect();
                    let x = rect.min.x + rr.right() + 2.0;
                    let top = rect.min.y + rr.top() + rr.height() * 0.12;
                    let caret =
                        Rect::from_min_size(egui::pos2(x, top), vec2(3.0, rr.height() * 0.9));
                    ui.painter().rect_filled(caret, CornerRadius::same(1), pal.accent2);
                }
                ui.ctx().request_repaint();
            }
        });

        if let Some(m) = metrics {
            ui.add_space(9.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 16.0;
                ui.label(
                    RichText::new(format!("{:.0} tok/s", m.tps))
                        .family(family(JB_SEMIBOLD))
                        .size(14.0)
                        .color(pal.metric),
                );
                ui.label(
                    RichText::new(format!("TTFT {:.0} ms", m.ttft_ms))
                        .family(mono())
                        .size(14.0)
                        .color(pal.metric_muted),
                );
                if m.vram_bytes > 0 {
                    let gib = m.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    ui.label(
                        RichText::new(format!("{gib:.1} GB VRAM"))
                            .family(mono())
                            .size(14.0)
                            .color(pal.metric_muted),
                    );
                }
            });
        }
    });
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
    let progress = |stage: usize, label: &str| {
        let _ = event_tx.send(InferenceEvent::Progress {
            stage,
            label: label.to_string(),
        });
        egui_ctx.request_repaint();
    };

    let (mut session, tokenizer, gpu_name) = match load(source, &progress).await {
        Ok(t) => t,
        Err(e) => {
            send(InferenceEvent::Error(format!("Failed to load model: {e:#}")));
            return;
        }
    };

    let vram_bytes = session.vram_bytes();
    send(InferenceEvent::Ready { gpu_name, vram_bytes });

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

                // Time-to-first-token spans prefill + the first sample/decode.
                let gen_start = Instant::now();
                let logits = match session.prefill(&input_ids).await {
                    Ok(l) => l,
                    Err(e) => {
                        send(InferenceEvent::Error(format!("Prefill failed: {e}")));
                        conversation.pop();
                        continue;
                    }
                };

                let mut rng = match sampling.seed {
                    Some(seed) => rand::SeedableRng::seed_from_u64(seed),
                    None => StdRng::from_seed(rand::random()),
                };
                let mut token = sample(&logits, &sampling, &mut rng);
                let mut response_tokens: Vec<i64> = Vec::new();

                let decode_start = Instant::now();
                let mut ttft_ms = 0.0;
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

                    if response_tokens.is_empty() {
                        ttft_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
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

                send(InferenceEvent::Done {
                    tokens_per_sec,
                    ttft_ms,
                    vram_bytes: session.vram_bytes(),
                });
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
/// platforms). Lowering and GPU init are async so this works on the web.
async fn build_session(
    graph: onyxia_onnx::Graph,
    tokenizer: Tokenizer,
    progress: &dyn Fn(usize, &str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress(5, "Initializing GPU");
    yield_to_browser().await;
    let ctx = onyxia_backend_wgpu::GpuContext::new()
        .await
        .context("Failed to initialise GPU context")?;
    let gpu_name = ctx.adapter_info.name.clone();

    progress(6, "Lowering model to IR");
    yield_to_browser().await;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())
        .context("Lowering failed")?;

    progress(7, "Uploading weights to GPU");
    yield_to_browser().await;
    let backend = onyxia_backend_wgpu::WgpuBackend::new(ctx);
    let session = LlmSession::new(&backend, module, MAX_SEQ_LEN)
        .context("Failed to prepare session")?;
    Ok((session, tokenizer, gpu_name))
}

/// Native: read the model + tokenizer from a directory on disk.
#[cfg(not(target_arch = "wasm32"))]
async fn load(
    model_dir: PathBuf,
    progress: &dyn Fn(usize, &str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress(3, "Parsing model");
    let onnx_path = model_dir.join("onnx/model.onnx");
    let graph = onyxia_onnx::load_and_parse_model(&onnx_path)
        .with_context(|| format!("Failed to parse ONNX from {}", onnx_path.display()))?;

    progress(4, "Fetching tokenizer");
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
    progress: &dyn Fn(usize, &str),
) -> Result<(LlmSession, Tokenizer, String)> {
    progress(2, "Fetching model");
    yield_to_browser().await;
    let model_bytes = fetch_bytes(&format!("{base_url}/onnx/model.onnx")).await?;
    let data_bytes = fetch_bytes(&format!("{base_url}/onnx/model.onnx_data")).await?;

    progress(3, "Parsing model");
    yield_to_browser().await;
    let mut external = std::collections::HashMap::new();
    external.insert("model.onnx_data".to_string(), data_bytes);
    let graph = onyxia_onnx::parse_model_from_bytes(&model_bytes, external)
        .context("Failed to parse ONNX model")?;

    progress(4, "Fetching tokenizer");
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

    // `--demo` drives the UI with scripted stages + a canned answer, so the
    // full interface (loader, chat, metrics, both themes) can be previewed
    // without a 1.1 GB model on disk.
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--shots <dir>` writes a fixed set of state screenshots, then exits.
    let shots_dir = args
        .iter()
        .position(|a| a == "--shots")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    // Screenshots and `--demo` both drive the UI without a real model.
    let demo = shots_dir.is_some() || args.iter().any(|a| a == "--demo");
    let model_dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-3-270m-it-ONNX"));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Onyxia — WebGPU Chatbot")
            .with_inner_size([1040.0, 720.0]),
        ..Default::default()
    };

    eframe::run_native(
        "gemma-chat",
        options,
        Box::new(move |cc| {
            let mut app = ChatApp::new(cc, model_dir, demo);
            if let Some(dir) = shots_dir {
                app.shots = Some(screenshot::Shots::new(dir));
            }
            Ok(Box::new(app))
        }),
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
                Box::new(|cc| Ok(Box::new(ChatApp::new(cc, ".".to_string(), false)))),
            )
            .await
            .expect("failed to start eframe");
    });
}
