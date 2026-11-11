//! BIOLUME: a dark sea of plankton; a big glowing swell sweeps left to right, a weaker one returns.
//! Port of pixelsaver.py `Tide`.
//!
//! Speed notes:
//! * The wave field is a sum of three sines of (x, y, t); each is split with sin(a+b) = sin a cos b + cos a sin b
//!   into static per-column factors and per-row/per-frame factors, so a pixel costs six multiply-adds.
//! * The swell's exp() profile is generated per row by recurrence (one multiply per pixel).
//! * crt (scanlines + vignette) is linear, so it is folded into every write instead of being a separate pass.
//! * The letter glow (four box blurs) is limited to the logo box plus the blur reach and is only recomputed
//!   when the quantised brightness of some letter pixel changes.
use super::Scene;
use crate::core::*;

const CYCLE: f32 = 12.0;
const PAL: [Rgb; 5] = [[0., 84., 96.], [0., 128., 132.], [0., 182., 160.], [70., 238., 204.], [220., 255., 242.]];

/// Separable box blur (radius r, edges clamped) of a w*h field: src -> dst, using tmp and a column accumulator.
fn box2(src: &[f32], dst: &mut [f32], tmp: &mut [f32], acc: &mut [f32], w: usize, h: usize, r: usize) {
    if w == 0 || h == 0 {
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    let hi = h as isize - 1;
    let row = |y: isize| y.clamp(0, hi) as usize * w;
    acc[..w].iter_mut().for_each(|a| *a = 0.0);
    for j in -(r as isize)..=(r as isize) {
        let o = row(j);
        for x in 0..w {
            acc[x] += src[o + x];
        }
    }
    for y in 0..h {
        let (o, ia, ib) = (y * w, row(y as isize + r as isize + 1), row(y as isize - r as isize));
        for x in 0..w {
            tmp[o + x] = acc[x] * inv;
            acc[x] += src[ia + x] - src[ib + x];
        }
    }
    let wi = w as isize - 1;
    for y in 0..h {
        let s = &tmp[y * w..(y + 1) * w];
        let d = &mut dst[y * w..(y + 1) * w];
        let at = |i: isize| s[i.clamp(0, wi) as usize];
        let mut sum = 0.0f32;
        for j in -(r as isize)..=(r as isize) {
            sum += at(j);
        }
        for x in 0..w as isize {
            d[x as usize] = sum * inv;
            sum += at(x + r as isize + 1) - at(x - r as isize);
        }
    }
}

/// Bilinear taps for upsampling an axis of n pixels from a grid pooled by f (cell i centred at f*i + (f-1)/2).
fn taps(n: usize, f: usize) -> Vec<(u32, u32, f32)> {
    let m = n.div_ceil(f).max(1);
    (0..n)
        .map(|x| {
            let u = ((x as f32 - (f as f32 - 1.0) / 2.0) / f as f32).clamp(0.0, (m - 1) as f32);
            let i = u as usize;
            (i as u32, (i + 1).min(m - 1) as u32, u - i as f32)
        })
        .collect()
}

pub struct Tide {
    t: f32,
    w: usize,
    h: usize,
    cs: usize,
    // static per-column wave factors
    sa: Vec<f32>,
    ca: Vec<f32>,
    sb: Vec<f32>,
    cb: Vec<f32>,
    sc: Vec<f32>,
    cc: Vec<f32>,
    yr: Vec<f32>,
    // crt: k(x, y) = ka[y] - kb[y] * fx2[x]
    ka: Vec<f32>,
    kb: Vec<f32>,
    fx2: Vec<f32>,
    wake: Vec<f32>,
    v: Vec<f32>,
    grow: Vec<f32>,
    // plankton
    pi: Vec<u32>,
    psn: Vec<f32>,
    pcs: Vec<f32>,
    pnear: Vec<f32>,
    // letters: screen index, region index, top edge
    li: Vec<u32>,
    /// pooled glow cell of each letter pixel and its weight (1 / cell area)
    lcell: Vec<u32>,
    lwt: Vec<f32>,
    ltop: Vec<bool>,
    lidx: Vec<u8>,
    // glow region
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
    /// 0 plain, 1 shadow, 2 outline, 3 letter
    kind: Vec<u8>,
    rbay: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    g3: Vec<f32>,
    tmp: Vec<f32>,
    acc: Vec<f32>,
    glow_valid: bool,
    /// bilinear taps from the cs-pooled glow grid, per region column / row
    hx: Vec<(u32, u32, f32)>,
    hy: Vec<(u32, u32, f32)>,
    hot: Vec<bool>,
}

impl Tide {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let p = place(logo, w, h, 0.8, 0.5, 0.5);
        let cs = p.s.max(1);
        let l = &p.l;
        let d = l.dilate(1, true);
        let out = d.and_not(l);
        let shadow = d.shift(2 * cs as i32, 2 * cs as i32).and_not(&d);
        let top = l.edge(0, -1);
        let n = w * h / 40;
        let (mut pi, mut psn, mut pcs) = (vec![], vec![], vec![]);
        let mut pxy = vec![];
        for _ in 0..n {
            let x = r.int(0, w as i64) as usize;
            let y = r.int(0, h as i64) as usize;
            let ph = r.range(0.0, std::f32::consts::TAU);
            pi.push((y * w + x) as u32);
            pxy.push(y * w + x);
            psn.push(ph.sin());
            pcs.push(ph.cos());
        }
        let near = box_blur(&box_blur(&l.to_grid(), 6 * cs), 6 * cs);
        let pnear = pxy.iter().map(|&i| (near.d[i] * 2.5).clamp(0.0, 1.0)).collect();
        let xr: Vec<f32> = (0..w).map(|x| x as f32 * 160.0 / w as f32).collect();
        let yr = (0..h).map(|y| y as f32 * 90.0 / h as f32).collect();
        let col = |k: f32| -> (Vec<f32>, Vec<f32>) { (xr.iter().map(|x| (x * k).sin()).collect(), xr.iter().map(|x| (x * k).cos()).collect()) };
        let (sa, ca) = col(0.07);
        let (sb, cb) = col(0.031);
        let (sc, cc) = col(0.02);
        let (scan, vig) = (0.88f32, 0.45f32);
        let mut ka = vec![0.0; h];
        let mut kb = vec![0.0; h];
        for y in 0..h {
            let fy = (y as f32 - h as f32 / 2.0) / h as f32;
            let s = if y % 2 == 1 { scan } else { 1.0 };
            ka[y] = s * (1.0 - vig * fy * fy);
            kb[y] = s * vig;
        }
        let fx2 = (0..w)
            .map(|x| {
                let f = (x as f32 - w as f32 / 2.0) / w as f32;
                f * f
            })
            .collect();
        // glow region: logo box plus the reach of the two double blurs (20 cs)
        // (21 cs keeps the pooling grid aligned with the logo's cs x cs blocks)
        let m = if cs == 1 { 21 } else { 21 * cs };
        let (rx0, ry0, rw, rh) = match l.bbox() {
            Some((x0, y0, x1, y1)) => {
                let (a, b) = (x0.saturating_sub(m), y0.saturating_sub(m));
                (a, b, (x1 + m).min(w) - a, (y1 + m).min(h) - b)
            }
            None => (0, 0, 0, 0),
        };
        let rn = rw * rh;
        let mut kind = vec![0u8; rn];
        let mut rbay = vec![0.0; rn];
        let (mut li, mut ltop, mut lcell, mut lwt) = (vec![], vec![], vec![], vec![]);
        for y in 0..rh {
            for x in 0..rw {
                let (sx, sy) = (x + rx0, y + ry0);
                let i = sy * w + sx;
                let ri = y * rw + x;
                rbay[ri] = bay(sx, sy);
                if shadow.d[i] {
                    kind[ri] = 1;
                } else if out.d[i] {
                    kind[ri] = 2;
                }
                if l.d[i] {
                    kind[ri] = 3;
                    let (cx, cy) = (x / cs, y / cs);
                    lcell.push((cy * rw.div_ceil(cs) + cx) as u32);
                    let area = ((cx * cs + cs).min(rw) - cx * cs) * ((cy * cs + cs).min(rh) - cy * cs);
                    lwt.push(1.0 / area as f32);
                    li.push(i as u32);
                    ltop.push(top.d[i]);
                }
            }
        }
        let nl = li.len();
        Tide {
            t: 0.0,
            w,
            h,
            cs,
            sa,
            ca,
            sb,
            cb,
            sc,
            cc,
            yr,
            ka,
            kb,
            fx2,
            wake: vec![0.0; w * h],
            v: vec![0.0; w * h],
            grow: vec![0.0; w],
            pi,
            psn,
            pcs,
            pnear,
            li,
            lcell,
            lwt,
            ltop,
            lidx: vec![255; nl],
            rx0,
            ry0,
            rw,
            rh,
            kind,
            rbay,
            g1: vec![0.0; rn],
            g2: vec![0.0; rn],
            g3: vec![0.0; rn],
            tmp: vec![0.0; rn],
            acc: vec![0.0; rw],
            glow_valid: false,
            hx: taps(rw, cs),
            hy: taps(rh, cs),
            hot: vec![false; rn],
        }
    }

    /// One row of the swell profile into `self.grow` (exp falloffs by recurrence, starting at the front).
    fn swell_row(&mut self, fx: f32, sign: f32, strength: f32) {
        let w = self.w;
        let cs = self.cs as f32;
        self.grow.iter_mut().for_each(|g| *g = 0.0);
        let (ka, kb) = (1.0 / (3.0 * cs), 1.0 / (28.0 * cs));
        // dx = (x - fx) * sign; ahead (dx > 0): exp(-dx * ka); behind (dx <= 0): exp(dx * kb)
        // walk away from the front in both directions
        let xf = fx.floor();
        // pixels with x <= fx
        let (k_left, k_right) = if sign > 0.0 { (kb, ka) } else { (ka, kb) };
        if xf >= 0.0 {
            let start = (xf as usize).min(w - 1);
            let mut g = (-(fx - start as f32) * k_left).exp() * strength;
            let r = (-k_left).exp();
            let mut x = start as isize;
            while x >= 0 && g > 1e-5 {
                // the rule for dx == 0 (behind) only matters at x == fx exactly
                self.grow[x as usize] = g;
                g *= r;
                x -= 1;
            }
        }
        // pixels with x > fx
        let s = (xf + 1.0).max(0.0);
        if (s as usize) < w {
            let start = s as usize;
            let mut g = (-(start as f32 - fx) * k_right).exp() * strength;
            let r = (-k_right).exp();
            let mut x = start;
            while x < w && g > 1e-5 {
                self.grow[x] = g;
                g *= r;
                x += 1;
            }
        }
    }
}

impl Scene for Tide {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h, cs) = (self.w, self.h, self.cs as f32);
        let ph = t.rem_euclid(CYCLE);
        let decay = (-dt * 0.4).exp();
        let wf = w as f32;
        // swell state for this frame
        let swell = if ph < 5.5 {
            Some((-0.15 * wf + ph / 5.0 * 1.3 * wf, 1.0f32, 1.0f32))
        } else if 6.5 < ph && ph < 11.5 {
            Some((1.15 * wf - (ph - 6.5) / 5.0 * 1.3 * wf, -1.0, 0.55))
        } else {
            None
        };
        let crest_col = swell.map(|(_, _, st)| {
            let k = (st * 2.0 - 1.0).clamp(0.0, 1.0);
            [120.0 + 110.0 * k, 236.0 + 19.0 * k, 214.0 + 32.0 * k]
        });
        for y in 0..h {
            let yr = self.yr[y];
            // per-row phases of the three sines, scaled into v = (w + 2.1) / 4.2
            let p1 = -0.8 * t + (yr * 0.09 + t * 0.25).sin() * 1.6;
            let p2 = yr * 0.11 + t * 0.45;
            let p3 = yr * 0.21 - t * 0.6;
            let q = 1.0 / 4.2;
            let (a1, b1) = (p1.cos() * q, p1.sin() * q);
            let (a2, b2) = (p2.cos() * 0.7 * q, p2.sin() * 0.7 * q);
            let (a3, b3) = (p3.cos() * 0.4 * q, p3.sin() * 0.4 * q);
            let mut cr = (1, 0); // crest x range (empty)
            let active = if let Some((f0, sign, st)) = swell {
                let wob = (yr * 0.12 + t * 0.7).sin() * 5.0 * cs + (yr * 0.05).sin() * 9.0 * cs;
                let fx = f0 + sign * wob;
                self.swell_row(fx, sign, st);
                let lo = (fx - 0.8 * cs).floor() + 1.0;
                let hi = (fx + 0.8 * cs).ceil() - 1.0;
                if hi >= 0.0 && lo < wf {
                    cr = (lo.max(0.0) as usize, (hi.min(wf - 1.0)) as usize);
                }
                true
            } else {
                false
            };
            let (ka, kb) = (self.ka[y], self.kb[y]);
            let brow = &B8[y & 7];
            let o = y * w;
            let vr = &mut self.v[o..o + w];
            let wr = &mut self.wake[o..o + w];
            let (sa, ca, sb, cb, sc, cc) = (&self.sa[..w], &self.ca[..w], &self.sb[..w], &self.cb[..w], &self.sc[..w], &self.cc[..w]);
            for x in 0..w {
                vr[x] = a1 * sa[x] + b1 * ca[x] + a2 * sb[x] + b2 * cb[x] + a3 * sc[x] + b3 * cc[x] + 0.5;
            }
            if active {
                let g = &self.grow[..w];
                for x in 0..w {
                    wr[x] = (wr[x] * decay).max(g[x]);
                }
            } else {
                wr.iter_mut().for_each(|v| *v *= decay);
            }
            let fx2 = &self.fx2[..w];
            let out = &mut img.px[o..o + w];
            // per Bayer column: the dither offsets of the two quantisers
            let mut ov = [0.0f32; 8];
            let mut ow = [0.0f32; 8];
            for k in 0..8 {
                ov[k] = 0.5 + (brow[k] - 0.5) * 0.6;
                ow[k] = 0.5 + (brow[k] - 0.5) * 0.7;
            }
            for x in 0..w {
                let (v, wk) = (vr[x], wr[x]);
                let j = x & 7;
                // qd() with truncation instead of floor (the arguments are >= 0)
                let qv = ((v * v * 3.0 + ov[j]) as i32).min(3) as f32;
                let qw = ((wk * 3.2 + ow[j]) as i32).min(4) as f32;
                let mut c = [1.0, 6.0 + qv * 4.0 + qw * 14.0, 12.0 + qv * 6.0 + qw * 13.0];
                let bh = wk > 0.4 && brow[j] < (wk - 0.4) * 1.4;
                c[0] = if bh { 60.0 } else { c[0] };
                c[1] = if bh { c[1].max(200.0) } else { c[1] };
                c[2] = if bh { c[2].max(180.0) } else { c[2] };
                let k = ka - kb * fx2[x];
                out[x] = [c[0] * k, c[1] * k, c[2] * k];
            }
            if let Some(cc) = crest_col {
                for x in cr.0..=cr.1.min(w - 1) {
                    if cr.0 > cr.1 {
                        break;
                    }
                    let k = ka - kb * fx2[x];
                    let p = &mut out[x];
                    *p = [p[0].max(cc[0] * k), p[1].max(cc[1] * k), p[2].max(cc[2] * k)];
                }
            }
        }
        // plankton flare under crests, in the wake, and where the letters disturb the water
        let (s17, c17) = ((1.7 * t).sin(), (1.7 * t).cos());
        let (s3, c3) = ((3.0 * t).sin(), (3.0 * t).cos());
        for j in 0..self.pi.len() {
            let i = self.pi[j] as usize;
            let (sp, cp) = (self.psn[j], self.pcs[j]);
            let stir = self.pnear[j] * (0.35 + 0.25 * (s17 * cp + c17 * sp));
            let pv = self.v[i];
            let pb = ((pv - 0.66) * 4.0).clamp(0.0, 1.0) * (0.6 + 0.4 * (s3 * cp + c3 * sp)) + self.wake[i] * 1.1 + stir;
            let pb = pb.clamp(0.0, 1.0);
            if pb > 0.08 {
                let (x, y) = (i % w, i / w);
                let k = pb.sqrt() * (self.ka[y] - self.kb[y] * self.fx2[x]);
                let c = [(10.0 + 170.0 * pb) * k, (120.0 + 135.0 * pb) * k, (110.0 + 122.0 * pb) * k];
                let p = &mut img.px[i];
                *p = [p[0].max(c[0]), p[1].max(c[1]), p[2].max(c[2])];
            }
        }
        // the letters: brightness from the water and the wake, plus their glow
        let mut changed = !self.glow_valid;
        for j in 0..self.li.len() {
            let i = self.li[j] as usize;
            let f = self.v[i] * 0.8 + 0.2 + self.wake[i] * 0.75;
            let idx = (f > 0.42) as u8 + (f > 0.58) as u8 + (f > 0.76) as u8 + (f > 1.05) as u8;
            if idx != self.lidx[j] {
                self.lidx[j] = idx;
                changed = true;
            }
        }
        let (rw, rh, c) = (self.rw, self.rh, self.cs);
        let (hw, hh) = (rw.div_ceil(c), rh.div_ceil(c));
        if changed && rw > 0 {
            // the glow is a very smooth field: blur on a grid pooled by cs (the logo's own pixel size, so the radii
            // 4cs and 10cs become 4 and 10 cells) and upsample bilinearly
            self.glow_valid = true;
            self.g3[..hw * hh].iter_mut().for_each(|v| *v = 0.0);
            for j in 0..self.li.len() {
                self.g3[self.lcell[j] as usize] += (0.4 + self.lidx[j] as f32 * 0.15) * self.lwt[j];
            }
            box2(&self.g3, &mut self.g1, &mut self.tmp, &mut self.acc, hw, hh, 4);
            box2(&self.g1, &mut self.g2, &mut self.tmp, &mut self.acc, hw, hh, 4);
            box2(&self.g3, &mut self.g1, &mut self.tmp, &mut self.acc, hw, hh, 10);
            box2(&self.g1, &mut self.g3, &mut self.tmp, &mut self.acc, hw, hh, 10);
            for i in 0..hw * hh {
                self.g1[i] = self.g2[i] * 2.0 + self.g3[i] * 1.4;
            }
            // a cell is "cold" if it and its right/lower neighbours are all below the first quantisation step
            // (g * 6 + 0.75 < 1), so every pixel interpolated from it is 0
            const COLD: f32 = 1.0 / 24.0;
            for j in 0..hh {
                let j1 = (j + 1).min(hh - 1);
                for i in 0..hw {
                    let i1 = (i + 1).min(hw - 1);
                    let m = self.g1[j * hw + i].max(self.g1[j * hw + i1]).max(self.g1[j1 * hw + i]).max(self.g1[j1 * hw + i1]);
                    self.hot[j * hw + i] = m >= COLD;
                }
            }
        }
        // glow (quantised, upsampled on the fly), the drop shadow and the outline
        for y in 0..rh {
            let sy = y + self.ry0;
            let (ka, kb) = (self.ka[sy], self.kb[sy]);
            let row = &mut img.px[sy * w + self.rx0..sy * w + self.rx0 + rw];
            let (kd, fx2, rb) = (&self.kind[y * rw..(y + 1) * rw], &self.fx2[self.rx0..self.rx0 + rw], &self.rbay[y * rw..(y + 1) * rw]);
            let (j0, j1, fv) = self.hy[y];
            let (r0, r1) = (&self.g1[j0 as usize * hw..(j0 as usize + 1) * hw], &self.g1[j1 as usize * hw..(j1 as usize + 1) * hw]);
            let hot = &self.hot[j0 as usize * hw..(j0 as usize + 1) * hw];
            for x in 0..rw {
                let kind = kd[x];
                if kind == 3 {
                    continue; // letter: overwritten below
                }
                let k = ka - kb * fx2[x];
                let (i0, i1, fu) = self.hx[x];
                let (i0, i1) = (i0 as usize, i1 as usize);
                if hot[i0] {
                    let a = r0[i0] + (r0[i1] - r0[i0]) * fu;
                    let b = r1[i0] + (r1[i1] - r1[i0]) * fu;
                    let g = (a + (b - a) * fv).clamp(0.0, 1.0);
                    let q = ((g * 6.0 + 0.5 + (rb[x] - 0.5) * 0.5) as i32).min(6) as f32 * 0.1 * k;
                    let p = &mut row[x];
                    p[1] += q * 170.0;
                    p[2] += q * 150.0;
                }
                match kind {
                    1 => {
                        let p = &mut row[x];
                        *p = [p[0] * 0.3, p[1] * 0.3, p[2] * 0.3];
                    }
                    2 => row[x] = [0.0, 4.0 * k, 8.0 * k],
                    _ => {}
                }
            }
        }
        for j in 0..self.li.len() {
            let i = self.li[j] as usize;
            let (x, y) = (i % w, i / w);
            let k = self.ka[y] - self.kb[y] * self.fx2[x];
            let mut col = PAL[self.lidx[j] as usize];
            if self.ltop[j] {
                col = [(col[0] + 40.0).min(255.0), (col[1] + 50.0).min(255.0), (col[2] + 44.0).min(255.0)];
            }
            img.px[i] = [col[0] * k, col[1] * k, col[2] * k];
        }
    }
}

