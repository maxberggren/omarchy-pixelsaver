//! EMBERS: the logo as a bed of coal lumps that slowly rekindles, glows, and settles back to ash.
//! Port of pixelsaver.py `Embers`. All geometry (lumps, cracks, depth, ash skin) is precomputed; a frame is
//! a handful of per-lump numbers, one pass over the letter pixels, and a separable halo blur restricted to
//! the logo's bounding box plus the glow margin. The rest of the screen is flat background, written once.
use super::{QuietBase, Scene};
use crate::core::*;

const CYCLE: f32 = 32.0;
const BG: Rgb = [4.0, 2.0, 2.0];
const ASH: [Rgb; 4] = [[38.0, 28.0, 26.0], [52.0, 41.0, 38.0], [68.0, 56.0, 51.0], [96.0, 82.0, 74.0]];
const HALO: Rgb = [150.0, 44.0, 10.0];
const R1: usize = 3;
const R2: usize = 8;

/// Breath of the fire over one cycle: cool ash -> rekindle -> glow -> settle back.
#[inline]
fn emb_heat(ph: f32) -> f32 {
    let up = smoothstep(ph, 4.0, 12.0);
    let down = smoothstep(ph, 20.0, 28.0);
    0.08 + 0.92 * up * (1.0 - down)
}

const F_CRACK: u8 = 1;
const F_BOT: u8 = 2;
const F_TOP: u8 = 4;

/// One letter pixel with everything about it that never changes.
struct Px {
    /// index into the screen
    i: u32,
    /// index into the transposed letter band of the halo region (x-major)
    ti: u32,
    /// lump id
    o: u16,
    flags: u8,
    /// depth into the lump (0 at the crack, 1 at the heart)
    u: f32,
    /// edge dither offset (bay - 0.5) * 0.05
    bj: f32,
    /// heat-quantisation dither offset 0.5 + (bay - 0.5) * 0.3
    bq: f32,
    /// resting ash colour (dithered crust, pale lip on top)
    ash: Rgb,
}

/// Vertical box blur of a `w`-wide buffer with `h` rows, computing output rows [ya, yb); input rows clamped
/// to [0, h). `cs` is a scratch row of width w.
fn vpass(src: &[f32], dst: &mut [f32], w: usize, h: usize, ya: usize, yb: usize, r: usize, cs: &mut [f32]) {
    if ya >= yb || w == 0 {
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    let hi = h as isize - 1;
    let row = |y: isize| {
        let y = y.clamp(0, hi) as usize;
        &src[y * w..y * w + w]
    };
    cs.iter_mut().for_each(|v| *v = 0.0);
    for j in ya as isize - r as isize..=ya as isize + r as isize {
        for (c, v) in cs.iter_mut().zip(row(j)) {
            *c += *v;
        }
    }
    for y in ya..yb {
        let d = &mut dst[y * w..y * w + w];
        for (o, c) in d.iter_mut().zip(cs.iter()) {
            *o = *c * inv;
        }
        let (add, sub) = (row(y as isize + r as isize + 1), row(y as isize - r as isize));
        for ((c, a), s) in cs.iter_mut().zip(add).zip(sub) {
            *c += *a - *s;
        }
    }
}

pub struct Embers {
    t: f32,
    pal5: [Rgb; 5],
    /// per letter pixel heat this frame
    heat: Vec<f32>,
    w: usize,
    h: usize,
    cs: usize,
    lx: f32,
    lw: f32,
    px: Vec<Px>,
    // per lump
    sx: Vec<f32>,
    hot: Vec<f32>,
    lag: Vec<f32>,
    ph0: Vec<f32>,
    // per-lump per-frame scratch
    lg: Vec<f32>,
    front: Vec<f32>,
    ka: Vec<f32>,
    kb: Vec<f32>,
    kinv: Vec<f32>,
    ct: Vec<f32>,
    // halo region (logo bbox + glow margin), local buffers of rw x rh
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
    /// letter rows inside the region
    ly0: usize,
    ly1: usize,
    /// letter band transposed (rw lines of ly1-ly0), so horizontal blurs are vectorised line passes
    lum_t: Vec<f32>,
    g_t: Vec<f32>,
    o_t: Vec<f32>,
    ha: Vec<f32>,
    hb: Vec<f32>,
    tmp: Vec<f32>,
    va: Vec<f32>,
    vb: Vec<f32>,
    /// halo quantiser dither offsets per (row & 7, region column)
    dq: Vec<f32>,
    colsum: Vec<f32>,
    /// per region pixel: 0.35 on the ring just outside the letters, 1 elsewhere
    om: Vec<f32>,
    /// spark tops (x, y-1)
    tops: Vec<(f32, f32)>,
    sparks: Particles,
    /// screen pixels outside the region touched by sparks last frame
    dirty: Vec<u32>,
    rng: Rng,
    primed: bool,
}

impl Embers {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let q = QuietBase::new(w, h, logo);
        let mut r = Rng::new(seed);
        let cs = q.cs;
        let l = &q.p.l;
        // letter pixels in row-major order, grouped per letter
        let mut ys = vec![];
        let mut xs = vec![];
        let mut lid = vec![];
        for y in 0..h {
            for x in 0..w {
                if l.d[y * w + x] {
                    ys.push(y as f32);
                    xs.push(x as f32);
                    lid.push(q.lid[y * w + x]);
                }
            }
        }
        let n = ys.len();
        let dmin = (0.42 * q.p.lh as f32).max(3.0);
        let mut d1 = vec![99.0f32; n];
        let mut d2 = vec![99.0f32; n];
        let mut own = vec![0usize; n];
        let (mut sx, mut sy) = (vec![], vec![]);
        for li in 1..=q.nl {
            let mut idx: Vec<usize> = (0..n).filter(|&j| lid[j] == li).collect();
            if idx.is_empty() {
                continue;
            }
            let base = sx.len();
            r.shuffle(&mut idx);
            let mut seeds: Vec<usize> = vec![];
            for &j in &idx {
                if seeds.iter().all(|&s| {
                    let (dx, dy) = (xs[j] - xs[s], (ys[j] - ys[s]) * 1.2);
                    dx * dx + dy * dy >= dmin * dmin
                }) {
                    seeds.push(j);
                    if seeds.len() >= 6 {
                        break;
                    }
                }
            }
            for &s in &seeds {
                sx.push(xs[s]);
                sy.push(ys[s]);
            }
            for &j in &idx {
                let (mut a, mut b, mut k) = (f32::MAX, f32::MAX, 0);
                for (si, &s) in seeds.iter().enumerate() {
                    let (dx, dy) = (xs[j] - xs[s], (ys[j] - ys[s]) * 1.2);
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < a {
                        b = a;
                        a = d;
                        k = si;
                    } else if d < b {
                        b = d;
                    }
                }
                if seeds.len() > 1 {
                    d1[j] = a;
                    d2[j] = b;
                }
                own[j] = base + k;
            }
        }
        let m = sx.len();
        let mut mx = vec![0.0f32; m];
        for j in 0..n {
            let gap = d2[j] - d1[j];
            mx[own[j]] = mx[own[j]].max(gap);
        }
        let hot: Vec<f32> = (0..m).map(|_| r.range(0.7, 1.0)).collect();
        let lag: Vec<f32> =
            (0..m).map(|k| (sx[k] - q.p.lx as f32) / q.p.lw.max(1) as f32 * 3.0 + r.range(0.0, 1.2)).collect();
        let ph0: Vec<f32> = (0..m).map(|_| r.range(0.0, 6.28)).collect();
        let _ = sy;

        // halo region: letter bbox grown by the full blur reach, clipped to the screen
        let margin = (R1 + R2) * cs + 1;
        let (bx0, by0, bx1, by1) = l.bbox().unwrap_or((0, 0, 0, 0));
        let (rx0, ry0) = (bx0.saturating_sub(margin), by0.saturating_sub(margin));
        let (rx1, ry1) = ((bx1 + margin).min(w), (by1 + margin).min(h));
        let (rw, rh) = if bx1 > bx0 { (rx1 - rx0, ry1 - ry0) } else { (0, 0) };
        let (ly0, ly1) = if rw > 0 { (by0 - ry0, by1 - ry0) } else { (0, 0) };

        let mut px = Vec::with_capacity(n);
        for j in 0..n {
            let (x, y) = (xs[j] as usize, ys[j] as usize);
            let i = y * w + x;
            let gap = d2[j] - d1[j];
            let u = (gap / mx[own[j]].max(1e-3)).clamp(0.0, 1.0);
            let b = bay(x, y);
            let mut flags = 0;
            if gap < 1.25 + 0.3 * (cs as f32 - 1.0) {
                flags |= F_CRACK;
            }
            if q.bot.d[i] {
                flags |= F_BOT;
            }
            if q.top.d[i] {
                flags |= F_TOP;
            }
            let ai = if q.top.d[i] { 3 } else { dith(0.2 + 0.55 * u, 3, b, 0.25) };
            px.push(Px {
                i: i as u32,
                ti: ((x - rx0) * (ly1 - ly0) + y - ry0 - ly0) as u32,
                o: own[j] as u16,
                flags,
                u,
                bj: (b - 0.5) * 0.05,
                bq: 0.5 + (b - 0.5) * 0.3,
                ash: ASH[ai],
            });
        }
        let mut om = vec![1.0f32; rw * rh];
        for y in 0..rh {
            for x in 0..rw {
                if q.out.d[(y + ry0) * w + x + rx0] {
                    om[y * rw + x] = 0.35;
                }
            }
        }
        let mut tops = vec![];
        for y in 0..h {
            for x in 0..w {
                if q.top.d[y * w + x] {
                    tops.push((x as f32, y as f32 - 1.0));
                }
            }
        }
        // EMBERPAL sampled at the five heat levels qd(.., 5) can produce
        let pal = grad(
            &[
                (0.0, [70.0, 10.0, 6.0]),
                (0.3, [150.0, 26.0, 8.0]),
                (0.55, [226.0, 78.0, 16.0]),
                (0.78, [255.0, 150.0, 40.0]),
                (0.92, [255.0, 214.0, 120.0]),
                (1.0, [255.0, 246.0, 214.0]),
            ],
            256,
        );
        let pal5 = [pal[51], pal[102], pal[153], pal[204], pal[255]];
        let z = vec![0.0f32; rw * rh];
        Embers {
            pal5,
            heat: vec![0.0; n],
            t: 0.0,
            w,
            h,
            cs,
            lx: q.p.lx as f32,
            lw: q.p.lw as f32,
            px,
            sx,
            hot,
            lag,
            ph0,
            lg: vec![0.0; m],
            front: vec![0.0; m],
            ka: vec![0.0; m],
            kb: vec![0.0; m],
            kinv: vec![0.0; m],
            ct: vec![0.0; m],
            rx0,
            ry0,
            rw,
            rh,
            ly0,
            ly1,
            lum_t: vec![0.0; rw * (ly1 - ly0)],
            g_t: vec![0.0; rw * (ly1 - ly0)],
            o_t: vec![0.0; rw * (ly1 - ly0)],
            ha: z.clone(),
            hb: z.clone(),
            tmp: z.clone(),
            va: z.clone(),
            vb: z,
            colsum: vec![0.0; rw.max(ly1 - ly0)],
            dq: (0..8 * rw).map(|i| 0.5 + (B8[i / rw.max(1)][(i % rw.max(1) + rx0) & 7] - 0.5) * 0.5).collect(),
            om,
            tops,
            sparks: Particles::default(),
            dirty: vec![],
            rng: r,
            primed: false,
        }
    }

    /// Halo: blur the heat (two radii), quantise, add to the background over the region.
    fn halo(&mut self, img: &mut Canvas) {
        let (rw, rh, cs) = (self.rw, self.rh, self.cs);
        if rw == 0 {
            return;
        }
        let (r1, r2) = (R1 * cs, R2 * cs);
        // horizontal passes only on letter rows (the heat is zero elsewhere): g = H1 lum, ha = H1 g, hb = H2 g
        let nr = self.ly1 - self.ly0;
        if nr == 0 {
            return;
        }
        vpass(&self.lum_t, &mut self.g_t, nr, rw, 0, rw, r1, &mut self.colsum[..nr]);
        for (r, hb) in [(r1, &mut self.ha), (r2, &mut self.hb)] {
            vpass(&self.g_t, &mut self.o_t, nr, rw, 0, rw, r, &mut self.colsum[..nr]);
            for (x, col) in self.o_t.chunks_exact(nr).enumerate() {
                for (j, &v) in col.iter().enumerate() {
                    hb[(self.ly0 + j) * rw + x] = v;
                }
            }
        }
        let colsum = &mut self.colsum[..rw];
        // vertical passes: a = V1 V1 ha, b = V2 V1 hb (each only over the rows it can reach)
        let ext = |r: usize| (self.ly0.saturating_sub(r), (self.ly1 + r).min(rh));
        let (t0, t1) = ext(r1);
        let (a0, a1) = ext(2 * r1);
        let (b0, b1) = ext(r1 + r2);
        vpass(&self.ha, &mut self.tmp, rw, rh, t0, t1, r1, colsum);
        vpass(&self.tmp, &mut self.va, rw, rh, a0, a1, r1, colsum);
        vpass(&self.hb, &mut self.tmp, rw, rh, t0, t1, r1, colsum);
        vpass(&self.tmp, &mut self.vb, rw, rh, b0, b1, r2, colsum);
        let w = self.w;
        let k = 0.65f32;
        for y in 0..rh {
            let sy = y + self.ry0;
            let o = sy * w + self.rx0;
            let dst = &mut img.px[o..o + rw];
            let (ra, rb, rm) = (&self.va[y * rw..y * rw + rw], &self.vb[y * rw..y * rw + rw], &self.om[y * rw..y * rw + rw]);
            if y < b0 || y >= b1 {
                for (x, p) in dst.iter_mut().enumerate() {
                    *p = scale(BG, rm[x]);
                }
                continue;
            }
            let dq = &self.dq[(sy & 7) * rw..(sy & 7) * rw + rw];
            for ((((p, &a), &b), &m), &d) in dst.iter_mut().zip(ra).zip(rb).zip(rm).zip(dq) {
                let v = ((a * 1.5 + b * 0.8) * k).clamp(0.0, 1.0);
                let lv = ((v * 6.0 + d) as i32).min(6) as f32 * (1.0 / 6.0);
                *p = [(BG[0] + lv * HALO[0]) * m, (BG[1] + lv * HALO[1]) * m, (BG[2] + lv * HALO[2]) * m];
            }
        }
    }
}

impl Scene for Embers {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h) = (self.w, self.h);
        // the background outside the halo region never changes: write it once, then only repair spark trails
        if !self.primed || img.w != w || img.h != h || img.px.len() != w * h {
            img.fill(BG);
            self.primed = true;
            self.dirty.clear();
        }
        for &i in &self.dirty {
            img.px[i as usize] = BG;
        }
        self.dirty.clear();

        let ph = (t + 2.0) % CYCLE;
        let gx = self.lx - 0.3 * self.lw + ((ph - 13.0) / 7.0) * 1.6 * self.lw;
        let gw = 1.0 / (0.22 * self.lw).max(1e-3);
        for k in 0..self.sx.len() {
            let g = emb_heat(ph - self.lag[k]);
            let breath = 0.88 + 0.12 * (t * 0.55 + self.ph0[k]).sin();
            let mut tt = g * self.hot[k] * breath;
            if g > 0.6 {
                let d = (self.sx[k] - gx) * gw;
                tt *= 1.0 + 0.28 * (-d * d).exp();
            }
            let fo = 0.14 + 0.96 * smoothstep(g, 0.1, 0.85);
            self.lg[k] = g;
            self.front[k] = fo;
            self.ka[k] = (0.18 + 0.82 * tt) * (0.35 + 0.65 * (tt * 3.0).min(1.0));
            self.kb[k] = 0.62 - 0.3 * tt;
            self.kinv[k] = 1.0 / fo.max(0.2);
            self.ct[k] = 0.24
                + 0.1 * (t * 0.45 + self.ph0[k] * 2.0).sin() * (1.0 - g)
                + 0.7 * g * (0.88 + 0.12 * (t * 0.9 + self.ph0[k]).sin());
        }
        let gg = emb_heat(ph - 1.5);
        let gs = 0.25 + 0.75 * gg;
        let pal5 = self.pal5;
        // letter pixels: heat (for the halo) and final colour (stored in-place after the halo pass)
        for (p, hp) in self.px.iter().zip(self.heat.iter_mut()) {
            let o = p.o as usize;
            let edge = p.u - self.front[o] + p.bj;
            let glow = edge < 0.0;
            let mut heat = if glow { (self.ka[o] * (1.0 - self.kb[o] * (p.u * self.kinv[o]).min(1.0))).min(0.78) } else { 0.0 };
            if p.flags & F_CRACK != 0 {
                heat = heat.max(self.ct[o]);
            }
            *hp = heat;
            self.lum_t[p.ti as usize] = heat * gs;
        }
        self.halo(img);
        for (p, &heat) in self.px.iter().zip(self.heat.iter()) {
            let o = p.o as usize;
            let edge = p.u - self.front[o] + p.bj;
            let qi = ((heat.clamp(0.0, 1.0) * 5.0 + p.bq) as i32).min(5);
            let hot = (edge < 0.0 || p.flags & F_CRACK != 0) && qi > 0;
            let mut c = if hot {
                pal5[qi as usize - 1]
            } else if p.flags & F_TOP == 0 && (0.0..0.09).contains(&edge) {
                ASH[3]
            } else {
                p.ash
            };
            if !hot && p.flags & F_BOT != 0 {
                c = scale(c, 0.7);
            }
            img.px[p.i as usize] = c;
        }

        // sparks lift off the hottest tops
        let r = &mut self.rng;
        if !self.tops.is_empty() && r.f() < dt * (0.2 + 3.5 * gg * gg) * w as f32 / 160.0 {
            let (x, y) = self.tops[r.below(self.tops.len())];
            let vx = r.range(-1.5, 1.5);
            let vy = -r.range(4.0, 9.0) * h as f32 / 90.0;
            let c = if r.f() < 0.6 { [255.0, 190.0, 90.0] } else { [255.0, 130.0, 40.0] };
            let life = r.range(1.8, 3.4);
            self.sparks.add(x, y, vx, vy, c, life);
        }
        for s in self.sparks.p.iter_mut() {
            s.vx += (t * 1.3 + s.y * 0.2).sin() * dt * 3.0;
        }
        self.sparks.update(dt, 0.0, 0.25);
        let (rx0, ry0, rx1, ry1) = (self.rx0, self.ry0, self.rx0 + self.rw, self.ry0 + self.rh);
        for s in &self.sparks.p {
            let (ix, iy) = (s.x.round() as i32, s.y.round() as i32);
            if ix < 0 || iy < 0 || ix as usize >= w || iy as usize >= h {
                continue;
            }
            let (ux, uy) = (ix as usize, iy as usize);
            let i = uy * w + ux;
            let a = (s.life / s.max).clamp(0.0, 1.0);
            img.px[i] = mix(img.px[i], s.c, a);
            if ux < rx0 || ux >= rx1 || uy < ry0 || uy >= ry1 {
                self.dirty.push(i as u32);
            }
        }
    }
}
