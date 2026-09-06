//! Thin binary wrapper around the `gemma_chat` library (see `lib.rs`).
//! Desktop parses the command line; web hands off to the canvas runner.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    use std::path::PathBuf;

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
    // Every non-flag argument is a model directory; the first loads, the
    // rest are offered by the header picker.
    let mut skip_next = false;
    let mut model_dirs: Vec<PathBuf> = Vec::new();
    for a in &args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--shots" {
            skip_next = true;
        } else if !a.starts_with("--") {
            model_dirs.push(PathBuf::from(a));
        }
    }
    if model_dirs.is_empty() {
        model_dirs.push(PathBuf::from("models/gemma-3-270m-it-ONNX"));
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Onyxia — WebGPU Chatbot")
            .with_inner_size([1040.0, 720.0]),
        ..Default::default()
    };
    gemma_chat::run_native(model_dirs, demo, shots_dir, options)
}

#[cfg(target_arch = "wasm32")]
fn main() {
    gemma_chat::run_web();
}
