//! Deterministic screenshot harness for capturing the UI in each of its states.
//!
//! Enabled with `--shots <dir>` (native only). It steps the app through a fixed
//! list of states (loading / empty / chat / streaming, in both themes), asks
//! egui to capture each frame via `ViewportCommand::Screenshot`, and writes the
//! raw RGBA to `<dir>/NN_<name>.<w>x<h>.rgba`. Convert to PNG with, e.g.:
//!
//! ```text
//! ffmpeg -f rawvideo -pixel_format rgba -video_size 1040x720 -i 01_night_empty.1040x720.rgba out.png
//! ```
//!
//! This module is a dev tool, not part of the shipped demo.

use std::path::PathBuf;

use eframe::egui::ColorImage;

/// The scripted shots, in order. Each name maps to a state configured by
/// `ChatApp::setup_shot`.
pub const SHOTS: [&str; 6] = [
    "night_loading",
    "night_empty",
    "night_chat",
    "night_streaming",
    "day_empty",
    "day_chat",
];

pub struct Shots {
    pub dir: PathBuf,
    /// Which shot we're on.
    pub index: usize,
    /// Whether the capture for the current shot has been requested this cycle.
    pub requested: bool,
}

impl Shots {
    pub fn new(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self {
            dir,
            index: 0,
            requested: false,
        }
    }
}

/// Dump one screenshot as raw RGBA, dimensions encoded in the filename.
pub fn save(dir: &std::path::Path, index: usize, name: &str, image: &ColorImage) {
    let (w, h) = (image.width(), image.height());
    let path = dir.join(format!("{index:02}_{name}.{w}x{h}.rgba"));
    if let Err(e) = std::fs::write(&path, image.as_raw()) {
        eprintln!("screenshot: failed to write {}: {e}", path.display());
    } else {
        eprintln!("screenshot: wrote {}", path.display());
    }
}
