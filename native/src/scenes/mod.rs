//! Scene registry. Each scene lives in its own file and implements `Scene`.
use crate::core::*;

pub mod cosmos;
pub mod daylight;
pub mod embers;
pub mod glitch;
pub mod ink;
pub mod lava;
pub mod tide;
pub mod warp;

/// A scene renders one frame per call. `dt` is the frame time in seconds; the scene keeps its own clock.
/// `img` is a canvas of the screen size; the scene must overwrite every pixel (it is not cleared for you).
pub trait Scene {
    fn draw(&mut self, dt: f32, img: &mut Canvas);
}

/// (name, title) in the default rotation order.
pub const NAMES: &[(&str, &str)] = &[
    ("embers", "EMBERS"),
    ("daylight", "DAYLIGHT"),
    ("cosmos", "DEEP FIELD"),
    ("ink", "SUMI-E"),
    ("tide", "BIOLUME"),
    ("lava", "MOLTEN"),
    ("warp", "HYPERSPACE"),
    ("glitch", "SIGNAL CORRUPT"),
];

pub fn make(name: &str, w: usize, h: usize, logo: &Logo, seed: u64) -> Option<Box<dyn Scene>> {
    Some(match name {
        "embers" => Box::new(embers::Embers::new(w, h, logo, seed)),
        "daylight" => Box::new(daylight::Daylight::new(w, h, logo, seed)),
        "cosmos" => Box::new(cosmos::Cosmos::new(w, h, logo, seed)),
        "ink" => Box::new(ink::Ink::new(w, h, logo, seed)),
        "tide" => Box::new(tide::Tide::new(w, h, logo, seed)),
        "lava" => Box::new(lava::Lava::new(w, h, logo, seed)),
        "warp" => Box::new(warp::Warp::new(w, h, logo, seed)),
        "glitch" => Box::new(glitch::Glitch::new(w, h, logo, seed)),
        _ => return None,
    })
}

/// Shared set-up of pixelsaver.py's `Quiet` base class: logo big and centred, letter ids and edges.
pub struct QuietBase {
    pub w: usize,
    pub h: usize,
    pub p: Placed,
    /// logo pixel scale (`self.cs`)
    pub cs: usize,
    /// letter id per column (1..nl, 0 = gap) and count
    pub ids: Vec<usize>,
    pub nl: usize,
    /// letter id per pixel (0 outside the logo)
    pub lid: Vec<usize>,
    pub top: Mask,
    pub bot: Mask,
    /// one-pixel ring just outside the letters
    pub out: Mask,
    /// vertical position inside the logo, 0 at the top row .. 1 at the bottom, per pixel row (clamped)
    pub rel: Vec<f32>,
}

impl QuietBase {
    pub fn new(w: usize, h: usize, logo: &Logo) -> Self {
        let p = place(logo, w, h, 0.84, 0.46, 0.5);
        let cs = p.s.max(1);
        let l = &p.l;
        let (ids, nl) = letter_ids(l);
        let lid = (0..w * h).map(|i| if l.d[i] { ids[i % w] } else { 0 }).collect();
        let top = l.edge(0, -1);
        let bot = l.edge(0, 1);
        let out = l.dilate(1, true).and_not(l);
        let rel = (0..h).map(|y| ((y as f32 - p.ly as f32) / (p.lh - 1).max(1) as f32).clamp(0.0, 1.0)).collect();
        QuietBase { w, h, cs, ids, nl, lid, top, bot, out, rel, p }
    }

    #[allow(dead_code)]
    /// `Quiet.halo`: a two-radius quantised glow of `lum`, added into `img` tinted `col`.
    pub fn halo(&self, img: &mut Canvas, lum: &Grid, col: Rgb, r1: usize, r2: usize, k: f32) {
        let cs = self.cs;
        let g = box_blur(lum, r1 * cs);
        let a = box_blur(&g, r1 * cs);
        let b = box_blur(&g, r2 * cs);
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let v = qd(((a.d[i] * 1.5 + b.d[i] * 0.8) * k).clamp(0.0, 1.0), 6.0, bay(x, y), 0.5);
                if v > 0.0 {
                    let p = &mut img.px[i];
                    p[0] += v * col[0];
                    p[1] += v * col[1];
                    p[2] += v * col[2];
                }
            }
        }
    }
}
