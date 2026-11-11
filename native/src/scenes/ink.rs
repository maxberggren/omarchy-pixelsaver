//! SUMI-E: a brush paints each letter in pale ink on black, a red seal is pressed, then the ink washes away.
//! Port of pixelsaver.py `Ink`. Everything that depends only on position is precomputed: each pixel knows
//! the moment the brush reaches it (`tp`) and when its wet sheen dries (`tw`), so a frame is one cheap pass.
use super::{QuietBase, Scene};
use crate::core::*;

const CYCLE: f32 = 24.0;
const SEAL: [&str; 5] = ["##.##", "#...#", ".###.", "#...#", "##.##"];
const NEVER: f32 = 1e9;

pub struct Ink {
    q: QuietBase,
    t: f32,
    bg: Vec<Rgb>,
    /// time (within the cycle) at which the brush front passes each letter pixel, NEVER elsewhere
    tp: Vec<f32>,
    /// time after which a painted pixel is no longer "wet"
    tw: Vec<f32>,
    /// time at which a bleed-halo pixel appears (its nearest letter pixel painted), NEVER if not bleed
    tb: Vec<f32>,
    dry: Vec<bool>,
    wash: Vec<f32>,
    seal: (i32, i32, usize),
    seal_glyph: Vec<bool>,
    ts: f32,
}

impl Ink {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let q = QuietBase::new(w, h, logo);
        let mut r = Rng::new(seed);
        let (cs, l) = (q.cs, &q.p.l);
        let bgc = [10.0, 10.0, 14.0];
        let mut bg = vec![bgc; w * h];
        for y in 0..h {
            for x in 0..w {
                let fx = (x as f32 - w as f32 / 2.0) / w as f32;
                let fy = (y as f32 - h as f32 / 2.0) / h as f32;
                bg[y * w + x] = scale(bgc, 1.0 - 0.3 * (fx * fx * 2.0 + fy * fy * 2.0));
            }
        }
        let nl = q.nl;
        let (mut x0, mut x1) = (vec![0.0f32; nl + 1], vec![1.0f32; nl + 1]);
        for k in 1..=nl {
            let cols: Vec<usize> = (0..w).filter(|&x| q.ids[x] == k).collect();
            x0[k] = cols[0] as f32;
            x1[k] = (*cols.last().unwrap() + 1) as f32;
        }
        let wob: Vec<f32> = (0..h).map(|y| ((y as f32 * 0.7).sin() + (y as f32 * 0.29).sin() * 1.5) * cs as f32).collect();
        // brush timing: painted when x < x0 + (ph - st)/0.55 * span + wob  <=>  ph > st + (x - wob - x0) * 0.55 / span
        let mut tp = vec![NEVER; w * h];
        let mut tw = vec![NEVER; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = q.lid[i];
                if k == 0 {
                    continue;
                }
                let st = 1.0 + (k as f32 - 1.0) * 0.7;
                let span = x1[k] - x0[k] + 2.0 * cs as f32;
                tp[i] = st + (x as f32 - wob[y] - x0[k]) * 0.55 / span;
                tw[i] = tp[i] + 3.0 * cs as f32 * 0.55 / span;
            }
        }
        // dry-brush streaks: horizontal noise, more likely toward the end of each letter
        let mut noise = Grid::new(w, h);
        noise.d.iter_mut().for_each(|v| *v = r.f());
        let streak = hblur(&noise, 3 * cs);
        let n = streak.d.len() as f32;
        let mean = streak.d.iter().sum::<f32>() / n;
        let sd = (streak.d.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt() + 1e-6;
        let mut dry = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let k = q.lid[i];
                if k == 0 {
                    continue;
                }
                let frac = ((x as f32 - x0[k]) / (x1[k] - x0[k])).clamp(0.0, 1.0);
                dry[i] = (streak.d[i] - mean) / sd > 2.2 - frac.powf(1.5) * 1.6;
            }
        }
        // bleed halo: ring of radius cs around the letters, thinned by noise; appears with its first painted neighbour
        noise.d.iter_mut().for_each(|v| *v = r.f());
        let bn = box_blur(&noise, 1);
        let rad = cs.max(1) as i32;
        let mut tb = vec![NEVER; w * h];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let i = y as usize * w + x as usize;
                if l.d[i] || bn.d[i] >= 0.5 {
                    continue;
                }
                let mut best = NEVER;
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        let (nx, ny) = (x + dx, y + dy);
                        if l.at(nx, ny) {
                            best = best.min(tp[ny as usize * w + nx as usize]);
                        }
                    }
                }
                tb[i] = best;
            }
        }
        // wash-out order: smooth noise normalised to 0..1
        noise.d.iter_mut().for_each(|v| *v = r.f());
        let wn = box_blur(&noise, 2 * cs + 1);
        let (lo, hi) = wn.d.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let wash = wn.d.iter().map(|v| (v - lo) / (hi - lo + 1e-6)).collect();
        // seal geometry and carved glyph
        let s = 7 * (cs / 2 + 1).max(1);
        let sx = (q.p.lx + q.p.lw - s as i32).min(w as i32 - s as i32 - 2);
        let sy = (q.p.ly + q.p.lh + 3 * cs as i32).min(h as i32 - s as i32 - 2);
        let mut g = vec![false; s * s];
        for i in 1..s - 1 {
            g[s + i] = true;
            g[(s - 2) * s + i] = true;
            g[i * s + 1] = true;
            g[i * s + s - 2] = true;
        }
        let gk = ((s as i32 - 4) / 5).max(1) as usize;
        let o = (s - 5 * gk) / 2;
        for (gy, row) in SEAL.iter().enumerate() {
            for (gx, ch) in row.chars().enumerate() {
                if ch == '#' {
                    for a in 0..gk {
                        for b in 0..gk {
                            let (yy, xx) = (o + gy * gk + a, o + gx * gk + b);
                            if yy < s && xx < s {
                                g[yy * s + xx] = true;
                            }
                        }
                    }
                }
            }
        }
        let ts = 1.0 + (nl as f32 - 1.0) * 0.7 + 1.4;
        Ink { q, t: 0.0, bg, tp, tw, tb, dry, wash, seal: (sx, sy, s), seal_glyph: g, ts }
    }
}

impl Scene for Ink {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let (w, h) = (self.q.w, self.q.h);
        let ph = self.t % CYCLE;
        let washed = smoothstep(ph, 18.0, 22.5);
        let ink = [222.0, 214.0, 198.0];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let mut c = self.bg[i];
                let tp = self.tp[i];
                let tb = self.tb[i];
                if tp < NEVER || tb < NEVER {
                    let alive = qd(((self.wash[i] - washed) * 5.0).clamp(0.0, 1.0), 3.0, bay(x, y), 0.6);
                    if tb < ph && alive > 0.0 {
                        c = add(scale(c, 0.55), [70.0 * 0.45, 68.0 * 0.45, 66.0 * 0.45]);
                    }
                    if ph > tp && alive > 0.0 {
                        let col = if ph < self.tw[i] {
                            [255.0, 252.0, 244.0]
                        } else if self.dry[i] {
                            [110.0, 106.0, 100.0]
                        } else {
                            ink
                        };
                        c = mix(c, col, alive);
                    }
                }
                img.px[i] = c;
            }
        }
        let (sx, sy, s) = self.seal;
        let press = smoothstep(ph, self.ts, self.ts + 0.3);
        if press > 0.0 && sx > 0 && sy > 0 {
            for j in 0..s {
                for i in 0..s {
                    let (x, y) = (sx as usize + i, sy as usize + j);
                    if x >= w || y >= h {
                        continue;
                    }
                    let alive = qd(((self.wash[y * w + x] - washed) * 5.0).clamp(0.0, 1.0), 3.0, bay(x, y), 0.6);
                    if bay(x, y) < press * 1.15 && alive > 0.0 {
                        img.px[y * w + x] = if self.seal_glyph[j * s + i] { [236.0, 210.0, 196.0] } else { [186.0, 40.0, 36.0] };
                    }
                }
            }
        }
    }
}
