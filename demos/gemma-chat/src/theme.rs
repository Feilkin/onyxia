//! Colors, fonts, and drawing helpers for the chat UI.
//!
//! egui has no stylesheet, so this module is the one place that defines how the
//! UI looks. It provides:
//!
//! - [`Palette`] — every color the UI uses, in a [`Theme::Night`] and a
//!   [`Theme::Day`] variant. [`visuals`] converts a palette into the
//!   [`egui::Visuals`] egui reads for built-in widgets; the rest of the UI
//!   reads palette fields directly.
//! - [`install_fonts`] — loads Space Grotesk (proportional) and JetBrains Mono
//!   (monospace). egui does not synthesize bold, so each weight is registered
//!   as its own named family (see [`SG_BOLD`], [`JB_SEMIBOLD`], …).
//! - Drawing helpers for shapes egui has no primitive for: the [`logo`] mark,
//!   the icon glyphs, and [`fill_grad_rrect`] (a gradient-filled rounded
//!   rectangle, tessellated into a colored mesh because `rect_filled` takes a
//!   single flat color).

use std::sync::Arc;

use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, Mesh, Pos2, Rect, Shape, Stroke, pos2,
    vec2,
};

// ── typography ────────────────────────────────────────────────────────────────
//
// egui does not synthesize bold; each weight is a distinct font file registered
// as its own family. Space Grotesk (proportional) is used for UI text and
// JetBrains Mono for metrics/labels. Space Grotesk ships no static SemiBold, so
// headings that want a 600 weight use Bold(700), which reads fine at that size.

/// Space Grotesk Medium (500) — used for chat bubble text.
pub const SG_MEDIUM: &str = "sg-medium";
/// Space Grotesk Bold (700) — used for the wordmark and headings.
pub const SG_BOLD: &str = "sg-bold";
/// JetBrains Mono SemiBold (600) — used for the highlighted tok/s metric and %.
pub const JB_SEMIBOLD: &str = "jb-semibold";

/// A named font family, for `RichText::new(..).family(theme::family(theme::SG_BOLD))`.
pub fn family(name: &str) -> FontFamily {
    FontFamily::Name(name.into())
}

/// The default monospace family (JetBrains Mono Regular).
pub fn mono() -> FontFamily {
    FontFamily::Monospace
}

/// Register the vendored fonts. Call once, before the first frame.
///
/// Starts from egui's default fonts (kept as fallbacks so emoji and any glyph
/// missing from these two families still render) and prepends our faces.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    let mut add = |name: &str, bytes: &'static [u8]| {
        fonts
            .font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    };
    add(
        "sg-regular",
        include_bytes!("../assets/SpaceGrotesk-2.0.0/ttf/static/SpaceGrotesk-Regular.ttf"),
    );
    add(
        SG_MEDIUM,
        include_bytes!("../assets/SpaceGrotesk-2.0.0/ttf/static/SpaceGrotesk-Medium.ttf"),
    );
    add(
        SG_BOLD,
        include_bytes!("../assets/SpaceGrotesk-2.0.0/ttf/static/SpaceGrotesk-Bold.ttf"),
    );
    add(
        "jb-regular",
        include_bytes!("../assets/JetBrainsMono/fonts/ttf/JetBrainsMono-Regular.ttf"),
    );
    add(
        JB_SEMIBOLD,
        include_bytes!("../assets/JetBrainsMono/fonts/ttf/JetBrainsMono-SemiBold.ttf"),
    );

    // Default proportional/monospace = our regular faces, with egui's defaults
    // retained behind them as fallbacks.
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "sg-regular".to_owned());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "jb-regular".to_owned());

    // Each weighted family is its face followed by the matching fallback chain.
    let prop_fallback = fonts.families[&FontFamily::Proportional].clone();
    let mono_fallback = fonts.families[&FontFamily::Monospace].clone();
    let chain = |head: &str, tail: &[String]| {
        let mut v = Vec::with_capacity(tail.len() + 1);
        v.push(head.to_owned());
        v.extend_from_slice(tail);
        v
    };
    fonts
        .families
        .insert(family(SG_MEDIUM), chain(SG_MEDIUM, &prop_fallback));
    fonts
        .families
        .insert(family(SG_BOLD), chain(SG_BOLD, &prop_fallback));
    fonts
        .families
        .insert(family(JB_SEMIBOLD), chain(JB_SEMIBOLD, &mono_fallback));

    ctx.set_fonts(fonts);
}

// ── palette ───────────────────────────────────────────────────────────────────

/// Which of the two design themes is active.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Night,
    Day,
}

impl Theme {
    pub fn palette(self) -> Palette {
        match self {
            Theme::Night => night(),
            Theme::Day => day(),
        }
    }
}

/// The full set of colors for one theme. Some fields are semi-transparent
/// (alpha < 255) and are meant to composite over [`Palette::bg`].
#[derive(Clone, Copy)]
pub struct Palette {
    pub bg: Color32,
    pub border: Color32,
    pub border_soft: Color32,
    pub seg_dim: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub accent2: Color32,
    pub done: Color32,
    /// User bubble gradient endpoints (equal on night, a true gradient on day).
    pub user_a: Color32,
    pub user_b: Color32,
    pub user_text: Color32,
    pub bot_bg: Color32,
    pub metric: Color32,
    pub metric_muted: Color32,
    pub stop: Color32,
    pub stop_border: Color32,
    pub logo_a: Color32,
    pub logo_b: Color32,
    pub send_a: Color32,
    pub send_b: Color32,
    pub send_text: Color32,
    pub input_bg: Color32,
    pub input_border: Color32,
    /// Whether bot bubbles carry a drop shadow (day theme only).
    pub bot_shadow: bool,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}
/// `a` is 0..=255 (CSS alpha × 255). Composited over the panel fill by egui.
fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

fn night() -> Palette {
    Palette {
        bg: rgb(0x0e, 0x0a, 0x2e),
        border: rgba(154, 96, 255, 51),      // rgba(154,96,255,.20)
        border_soft: rgba(255, 255, 255, 18), // rgba(255,255,255,.07)
        seg_dim: rgba(255, 255, 255, 20),     // rgba(255,255,255,.08)
        text: rgb(0xff, 0xfd, 0xeb),
        muted: rgb(0xb2, 0x95, 0xff),
        accent: rgb(0x9a, 0x60, 0xff),
        accent2: rgb(0x2b, 0xff, 0xd2),
        done: rgb(0x60, 0xff, 0x64),
        // Night user bubble is a flat fill.
        user_a: rgb(0x9a, 0x60, 0xff),
        user_b: rgb(0x9a, 0x60, 0xff),
        user_text: rgb(0x0e, 0x0a, 0x2e),
        bot_bg: rgb(0x19, 0x13, 0x52),
        metric: rgb(0x2b, 0xff, 0xd2),
        metric_muted: rgba(255, 253, 235, 107), // rgba(255,253,235,.42)
        stop: rgb(0xfd, 0x75, 0xca),
        stop_border: rgba(253, 117, 202, 128), // rgba(253,117,202,.5)
        logo_a: rgb(0x9a, 0x60, 0xff),
        logo_b: rgb(0x2b, 0xff, 0xd2),
        send_a: rgb(0x9a, 0x60, 0xff),
        send_b: rgb(0xfd, 0x75, 0xca),
        send_text: rgb(0x0e, 0x0a, 0x2e),
        input_bg: rgba(255, 255, 255, 15), // rgba(255,255,255,.06)
        input_border: rgba(154, 96, 255, 77), // rgba(154,96,255,.3)
        bot_shadow: false,
    }
}

fn day() -> Palette {
    Palette {
        bg: rgb(0xff, 0xfd, 0xeb),
        border: rgba(48, 40, 109, 33),       // rgba(48,40,109,.13)
        border_soft: rgba(48, 40, 109, 26),  // rgba(48,40,109,.1)
        seg_dim: rgba(48, 40, 109, 31),      // rgba(48,40,109,.12)
        text: rgb(0x0e, 0x0a, 0x2e),
        muted: rgb(0x92, 0x4f, 0xdc),
        accent: rgb(0x92, 0x4f, 0xdc),
        accent2: rgb(0xf8, 0x1c, 0xa6),
        done: rgb(0x22, 0xc2, 0x78),
        // Day user bubble is a 135° gradient: #924fdc → #9a60ff.
        user_a: rgb(0x92, 0x4f, 0xdc),
        user_b: rgb(0x9a, 0x60, 0xff),
        user_text: rgb(0xff, 0xfd, 0xeb),
        bot_bg: rgb(0xff, 0xff, 0xff),
        metric: rgb(0x22, 0xc2, 0x78),
        metric_muted: rgba(48, 40, 109, 128), // rgba(48,40,109,.5)
        stop: rgb(0xd5, 0x3c, 0x6a),
        stop_border: rgba(213, 60, 106, 128), // rgba(213,60,106,.5)
        logo_a: rgb(0x92, 0x4f, 0xdc),
        logo_b: rgb(0xf8, 0x1c, 0xa6),
        send_a: rgb(0x92, 0x4f, 0xdc),
        send_b: rgb(0xf8, 0x1c, 0xa6),
        send_text: rgb(0xff, 0xfd, 0xeb),
        input_bg: rgb(0xff, 0xff, 0xff),
        input_border: rgba(146, 79, 220, 102), // rgba(146,79,220,.4)
        bot_shadow: true,
    }
}

/// Build egui visuals for a palette. We paint most surfaces with explicit
/// `Frame`s, so this mainly sets the panel background, the default text color,
/// and the widgets the built-in `TextEdit` reads from.
pub fn visuals(p: &Palette, theme: Theme) -> egui::Visuals {
    let mut v = match theme {
        Theme::Night => egui::Visuals::dark(),
        Theme::Day => egui::Visuals::light(),
    };
    v.panel_fill = p.bg;
    v.window_fill = p.bg;
    v.faint_bg_color = p.bg;
    v.extreme_bg_color = p.input_bg; // TextEdit background
    v.override_text_color = Some(p.text);
    v.selection.bg_fill = p.accent.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, p.accent);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, p.border_soft);
    v
}

// ── paint helpers ─────────────────────────────────────────────────────────────

/// Linear blend between two colors (t = 0 → a, t = 1 → b).
pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 * (1.0 - t) + y as f32 * t).round() as u8;
    Color32::from_rgba_premultiplied(
        l(a.r(), b.r()),
        l(a.g(), b.g()),
        l(a.b(), b.b()),
        l(a.a(), b.a()),
    )
}

/// The Onyxia logo mark: a diamond (a square rotated 45°) filled with a
/// diagonal `a`→`b` gradient.
pub fn logo(painter: &egui::Painter, rect: Rect, a: Color32, b: Color32) {
    let c = rect.center();
    let h = rect.width().min(rect.height()) * 0.5;
    let top = pos2(c.x, c.y - h);
    let right = pos2(c.x + h, c.y);
    let bottom = pos2(c.x, c.y + h);
    let left = pos2(c.x - h, c.y);
    let mid = mix(a, b, 0.5);

    let mut mesh = Mesh::default();
    mesh.colored_vertex(c, mid); // 0
    mesh.colored_vertex(top, a); // 1
    mesh.colored_vertex(right, b); // 2
    mesh.colored_vertex(bottom, b); // 3
    mesh.colored_vertex(left, a); // 4
    for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 1)] {
        mesh.add_triangle(0, i, j);
    }
    painter.add(Shape::mesh(mesh));
}

/// Fill a rounded rectangle with a 135° `a`→`b` gradient.
///
/// egui's `rect_filled` only takes a single color and a colored mesh cannot be
/// clipped to a rounded outline, so we tessellate the rounded rectangle here:
/// walk the four corner arcs into a perimeter polygon and fan-triangulate it
/// from the center, coloring every vertex by its position along the diagonal.
/// `corners` is (nw, ne, se, sw) in the same order as [`egui::CornerRadius`].
pub fn fill_grad_rrect(
    painter: &egui::Painter,
    rect: Rect,
    corners: (f32, f32, f32, f32),
    a: Color32,
    b: Color32,
) {
    use std::f32::consts::{FRAC_PI_2, PI};
    let (nw, ne, se, sw) = corners;
    let seg = 6; // arc subdivisions per corner
    let mut pts: Vec<Pos2> = Vec::with_capacity(4 * (seg + 1));
    let mut arc = |cx: f32, cy: f32, r: f32, start: f32, end: f32| {
        for i in 0..=seg {
            let t = start + (end - start) * i as f32 / seg as f32;
            pts.push(pos2(cx + r * t.cos(), cy + r * t.sin()));
        }
    };
    arc(rect.left() + nw, rect.top() + nw, nw, PI, PI + FRAC_PI_2);
    arc(rect.right() - ne, rect.top() + ne, ne, PI + FRAC_PI_2, 2.0 * PI);
    arc(rect.right() - se, rect.bottom() - se, se, 0.0, FRAC_PI_2);
    arc(rect.left() + sw, rect.bottom() - sw, sw, FRAC_PI_2, PI);

    let color_at = |p: Pos2| {
        let t = ((p.x - rect.left()) / rect.width().max(1.0)
            + (p.y - rect.top()) / rect.height().max(1.0))
            * 0.5;
        mix(a, b, t.clamp(0.0, 1.0))
    };
    let c = rect.center();
    let mut mesh = Mesh::default();
    mesh.colored_vertex(c, color_at(c));
    for p in &pts {
        mesh.colored_vertex(*p, color_at(*p));
    }
    let n = pts.len() as u32;
    for i in 0..n {
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % n);
    }
    painter.add(Shape::mesh(mesh));
}

fn stroke_poly(painter: &egui::Painter, pts: Vec<Pos2>, stroke: Stroke) {
    painter.add(Shape::line(pts, stroke));
}

/// The send glyph: an upward arrow (`M12 19V5 M5 12l7-7 7 7`), scaled into `rect`.
pub fn icon_send(painter: &egui::Painter, rect: Rect, color: Color32) {
    let s = Stroke::new((rect.width() * 0.09).max(1.6), color);
    let p = |x: f32, y: f32| rect.lerp_inside(vec2(x, y));
    stroke_poly(painter, vec![p(0.5, 0.82), p(0.5, 0.2)], s); // shaft
    stroke_poly(painter, vec![p(0.24, 0.46), p(0.5, 0.2), p(0.76, 0.46)], s); // head
}

/// The day (sun) glyph: a ringed circle with eight rays.
pub fn icon_sun(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    let r = rect.width() * 0.22;
    let s = Stroke::new((rect.width() * 0.09).max(1.4), color);
    painter.circle_stroke(c, r, s);
    let ray = rect.width() * 0.12;
    let gap = rect.width() * 0.30;
    for k in 0..8 {
        let ang = std::f32::consts::TAU * (k as f32) / 8.0;
        let dir = vec2(ang.cos(), ang.sin());
        stroke_poly(painter, vec![c + dir * gap, c + dir * (gap + ray)], s);
    }
}

/// The night (moon) glyph: a filled disc with an offset disc carved out.
pub fn icon_moon(painter: &egui::Painter, rect: Rect, color: Color32, carve: Color32) {
    let c = rect.center();
    let r = rect.width() * 0.30;
    painter.circle_filled(c, r, color);
    painter.circle_filled(c + vec2(r * 0.55, -r * 0.45), r * 0.85, carve);
}

/// A reload glyph: a ~300° arc with a small arrowhead (a circular-arrow `↻`).
pub fn icon_reload(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    let r = rect.width() * 0.30;
    let s = Stroke::new((rect.width() * 0.11).max(1.4), color);
    let start = -0.35 * std::f32::consts::PI;
    let sweep = 1.6 * std::f32::consts::PI;
    let steps = 24;
    let arc: Vec<Pos2> = (0..=steps)
        .map(|i| {
            let a = start + sweep * (i as f32 / steps as f32);
            c + vec2(a.cos(), a.sin()) * r
        })
        .collect();
    let tip = *arc.last().unwrap();
    stroke_poly(painter, arc, s);
    // Arrowhead at the arc's end, pointing tangentially.
    let end_ang = start + sweep;
    let tang = vec2(-end_ang.sin(), end_ang.cos());
    let norm = vec2(end_ang.cos(), end_ang.sin());
    let hl = rect.width() * 0.16;
    stroke_poly(painter, vec![tip, tip - tang * hl + norm * hl], s);
    stroke_poly(painter, vec![tip, tip - tang * hl - norm * hl], s);
}

// ── loading stages ────────────────────────────────────────────────────────────

/// The eight load stages shown by the loader, in order. The loader reports its
/// current stage as an index into this list, which drives the segmented bar and
/// the title. (The first two complete before egui paints — native binary start
/// / WASM boot — so they appear already done on the first frame.)
pub const STAGE_LABELS: [&str; 8] = [
    "Loading WASM runtime",
    "Starting",
    "Fetching model",
    "Parsing model",
    "Fetching tokenizer",
    "Initializing GPU",
    "Lowering model to IR",
    "Uploading weights to GPU",
];
