//! DEEP FIELD: each cycle a new star ignites inside the word and lights up the drifting nebula around it.
//! Port of pixelsaver.py `Cosmos`.
//!
//! Speed notes:
//! * The nebula noise is only generated for the logo rows (plus the blur reach), and stored cropped.
//! * Per-cycle constants of every letter pixel (distance to the star and its exp() falloffs) are computed once
//!   per cycle, so the nebula pass is a handful of multiply-adds per letter pixel.
//! * The nebula + glow ("base") only changes when the integer drift offsets or the phase envelopes change;
//!   it is cached and only recomputed then (during the hold that is a few times per second).
//! * The halo blur runs only on the logo bounding box plus the blur reach, with scratch buffers kept in the
//!   struct and cache-friendly running-sum blurs.
use super::{QuietBase, Scene};
use crate::core::*;

const CYCLE: f32 = 32.0;
const LEVELS: usize = 9;

type Stops = [(f32, Rgb); 6];
const PALETTES: [(Stops, Stops, Rgb, Rgb); 3] = [
    (
        [(0.0, [14., 10., 40.]), (0.25, [60., 18., 96.]), (0.5, [170., 36., 130.]), (0.72, [255., 104., 120.]), (0.88, [255., 190., 160.]), (1.0, [255., 244., 226.])],
        [(0.0, [8., 14., 44.]), (0.25, [18., 48., 120.]), (0.5, [30., 120., 190.]), (0.72, [70., 200., 230.]), (0.88, [170., 240., 250.]), (1.0, [240., 255., 255.])],
        [255., 236., 220.],
        [150., 40., 120.],
    ),
    (
        [(0.0, [22., 14., 36.]), (0.25, [92., 46., 34.]), (0.5, [170., 96., 30.]), (0.72, [240., 170., 60.]), (0.88, [255., 222., 140.]), (1.0, [255., 250., 220.])],
        [(0.0, [8., 24., 44.]), (0.25, [14., 72., 90.]), (0.5, [20., 130., 136.]), (0.72, [70., 200., 186.]), (0.88, [170., 245., 225.]), (1.0, [240., 255., 248.])],
        [255., 246., 214.],
        [120., 80., 30.],
    ),
    (
        [(0.0, [10., 12., 44.]), (0.25, [30., 36., 130.]), (0.5, [70., 90., 220.]), (0.72, [130., 160., 255.]), (0.88, [200., 214., 255.]), (1.0, [250., 252., 255.])],
        [(0.0, [18., 10., 44.]), (0.25, [64., 30., 130.]), (0.5, [130., 70., 210.]), (0.72, [200., 140., 255.]), (0.88, [236., 206., 255.]), (1.0, [255., 248., 255.])],
        [226., 236., 255.],
        [60., 70., 170.],
    ),
];
const SCOL: [Rgb; 4] = [[255., 255., 255.], [190., 214., 255.], [255., 226., 190.], [255., 200., 200.]];

/// Separable box blur (radius r, edges clamped) of a w*h field: src -> dst, using tmp and a column accumulator.
fn box2(src: &[f32], dst: &mut [f32], tmp: &mut [f32], acc: &mut [f32], w: usize, h: usize, r: usize) {
    if w == 0 || h == 0 {
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    let hi = h as isize - 1;
    let row = |y: isize| y.clamp(0, hi) as usize * w;
    // vertical: running column sums, row by row
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
    // horizontal
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

/// `COS_fbm`, evaluated only for screen rows y0..y1. Each octave (white noise box-blurred twice with radius
/// `rad`, wrapping horizontally) is generated with enough rows around y0..y1 that the clamped vertical blur
/// matches the full-screen result. Large-radius octaves are built on a grid `f` times coarser (radius rad/f)
/// and upsampled bilinearly: the double box blur removes everything that grid could not represent, and init
/// gets ~f^2 cheaper. Returns (y1-y0) * tw values.
fn fbm(r: &mut Rng, h: usize, tw: usize, y0: usize, y1: usize, base: f32) -> Vec<f32> {
    let n_out = y1 - y0;
    let mut out = vec![0.0f32; n_out * tw];
    let (mut amp, mut tot) = (1.0f32, 0.0f32);
    for o in 0..3 {
        let rad = ((base / (1 << o) as f32) as usize).max(1);
        let f = (rad / 4).max(1);
        let crad = ((rad as f32 / f as f32).round() as usize).max(1);
        let ga = y0.saturating_sub(2 * rad);
        let gb = (y1 + 2 * rad).min(h);
        let gh = gb - ga;
        let (cw, ch) = if f == 1 { (tw, gh) } else { (tw.div_ceil(f), gh.div_ceil(f) + 1) };
        let n = ch * cw;
        let mut a: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let mut b = vec![0.0f32; n];
        let mut acc = vec![0.0f32; cw];
        let mut pad = vec![0.0f32; cw + 2 * crad + 2];
        let inv = 1.0 / (2 * crad + 1) as f32;
        for _ in 0..2 {
            // horizontal, wrapping: pad[i] = row[(i - crad - 1) mod cw], then a running sum
            for y in 0..ch {
                let s = &a[y * cw..(y + 1) * cw];
                let mut src = (cw * (crad / cw + 2) - crad - 1) % cw;
                let mut i = 0;
                while i < pad.len() {
                    let m = (cw - src).min(pad.len() - i);
                    pad[i..i + m].copy_from_slice(&s[src..src + m]);
                    i += m;
                    src = 0;
                }
                let d = &mut b[y * cw..(y + 1) * cw];
                let mut sum: f32 = pad[1..2 * crad + 2].iter().sum();
                for x in 0..cw {
                    d[x] = sum * inv;
                    sum += pad[x + 2 * crad + 2] - pad[x + 1];
                }
            }
            // vertical, edges clamped
            let hi = ch as isize - 1;
            let row = |y: isize| y.clamp(0, hi) as usize * cw;
            acc.iter_mut().for_each(|v| *v = 0.0);
            for j in -(crad as isize)..=(crad as isize) {
                let o = row(j);
                for x in 0..cw {
                    acc[x] += b[o + x];
                }
            }
            for y in 0..ch {
                let (o, ia, ib) = (y * cw, row(y as isize + crad as isize + 1), row(y as isize - crad as isize));
                for x in 0..cw {
                    a[o + x] = acc[x] * inv;
                    acc[x] += b[ia + x] - b[ib + x];
                }
            }
        }
        let mean = a.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = a.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let (mean, k) = (mean as f32, amp / (var.sqrt() as f32 + 1e-9));
        if f == 1 {
            for y in y0..y1 {
                let (so, d_o) = ((y - ga) * tw, (y - y0) * tw);
                for x in 0..tw {
                    out[d_o + x] += (a[so + x] - mean) * k;
                }
            }
        } else {
            // bilinear upsample (wrapping in x)
            let sx = cw as f32 / tw as f32;
            let cols: Vec<(usize, usize, f32)> = (0..tw)
                .map(|x| {
                    let u = x as f32 * sx;
                    let i = u as usize % cw;
                    (i, (i + 1) % cw, u - u.floor())
                })
                .collect();
            for y in y0..y1 {
                let v = (y - ga) as f32 / f as f32;
                let j = (v as usize).min(ch - 1);
                let (j1, fv) = ((j + 1).min(ch - 1), v - v.floor());
                let (r0, r1) = (&a[j * cw..(j + 1) * cw], &a[j1 * cw..(j1 + 1) * cw]);
                let d_o = (y - y0) * tw;
                for (x, &(i0, i1, fu)) in cols.iter().enumerate() {
                    let top = r0[i0] + (r0[i1] - r0[i0]) * fu;
                    let bot = r1[i0] + (r1[i1] - r1[i0]) * fu;
                    out[d_o + x] += (top + (bot - top) * fv - mean) * k;
                }
            }
        }
        tot += amp;
        amp *= 0.45;
    }
    for v in out.iter_mut() {
        *v = (0.5 + *v / tot * 0.32).clamp(0.0, 1.0);
    }
    out
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

pub struct Cosmos {
    q: QuietBase,
    t: f32,
    r: Rng,
    // letter pixels (screen index, x, y, Bayer threshold)
    li: Vec<u32>,
    lx: Vec<u16>,
    ly: Vec<u16>,
    lb: Vec<f32>,
    /// region index of each letter pixel
    lr: Vec<u32>,
    // nebula fields for rows fy0.., width tw
    fy0: usize,
    tw: usize,
    neb_a: Vec<f32>,
    neb_b: Vec<f32>,
    hue: Vec<f32>,
    dust: Vec<f32>,
    /// per palette: warm and cool colour of each of the 9 levels, star tint, halo tint
    ramps: Vec<([Rgb; LEVELS], [Rgb; LEVELS], Rgb, Rgb)>,
    // stars
    sx: Vec<f32>,
    sy: Vec<i32>,
    sl: Vec<u8>,
    sph: Vec<f32>,
    scol: Vec<Rgb>,
    sxi: Vec<i32>,
    cand: Vec<(u16, u16)>,
    // cycle state
    cyc: i64,
    star: (f32, f32),
    pal: usize,
    meteor_t: f32,
    shoot: Option<(f32, f32, f32)>,
    // per-cycle per-letter-pixel constants
    c_rr: Vec<f32>,
    c_near: Vec<f32>,
    c_core: Vec<f32>,
    c_hm: Vec<f32>,
    // cached base (nebula + glow) over the region rx0..rx0+rw, ry0..ry0+rh
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
    glowok: Vec<bool>,
    rbay: Vec<f32>,
    base: Vec<Rgb>,
    /// pooled halo cell of each letter pixel and its weight (1 / cell area)
    lcell: Vec<u32>,
    lwt: Vec<f32>,
    hx: Vec<(u32, u32, f32)>,
    hy: Vec<(u32, u32, f32)>,
    hot: Vec<bool>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    g3: Vec<f32>,
    tmp: Vec<f32>,
    acc: Vec<f32>,
    key: Option<(i64, usize, usize, [u32; 4])>,
    dw: f32,
    lmask: Vec<bool>,
}

impl Cosmos {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let q = QuietBase::new(w, h, logo);
        let mut r = Rng::new(seed);
        let cs = q.cs;
        let l = q.p.l.clone();
        let (mut li, mut lx, mut ly, mut lb) = (vec![], vec![], vec![], vec![]);
        for y in 0..h {
            for x in 0..w {
                if l.d[y * w + x] {
                    li.push((y * w + x) as u32);
                    lx.push(x as u16);
                    ly.push(y as u16);
                    lb.push(bay(x, y));
                }
            }
        }
        let bb = l.bbox();
        // nebula, only for the rows that hold letters
        let tw = 2 * w;
        let (fy0, fy1) = bb.map(|b| (b.1, b.3)).unwrap_or((0, 0));
        let base = (q.p.lh as f32 * 0.5).max(4.0);
        let neb_a = fbm(&mut r, h, tw, fy0, fy1, base);
        let neb_b = fbm(&mut r, h, tw, fy0, fy1, base * 0.7);
        let hue = fbm(&mut r, h, tw, fy0, fy1, base * 1.4);
        let mut dust = fbm(&mut r, h, tw, fy0, fy1, base * 0.6);
        for d in dust.iter_mut() {
            *d = smoothstep(1.0 - (2.0 * *d - 1.0).abs(), 0.8, 0.97);
        }
        let ramps = PALETTES
            .iter()
            .map(|(a, b, s, hh)| {
                let (ga, gb) = (grad(a, 256), grad(b, 256));
                let mut wa = [[0.0; 3]; LEVELS];
                let mut wb = [[0.0; 3]; LEVELS];
                for k in 0..LEVELS {
                    let vi = (k as f32 / (LEVELS - 1) as f32 * 255.0) as usize;
                    wa[k] = ga[vi];
                    wb[k] = gb[vi];
                }
                (wa, wb, *s, *hh)
            })
            .collect();
        // stars at three depths
        let n = 120.max(li.len() / (5 * cs));
        let (mut sx, mut sy, mut sl, mut sph, mut scol) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            sx.push(r.range(0.0, w as f32));
            sy.push(r.int(0, h as i64) as i32);
            let u = r.f();
            sl.push(if u < 0.6 { 0 } else if u < 0.9 { 1 } else { 2 });
            sph.push(r.range(0.0, std::f32::consts::TAU));
            scol.push(SCOL[r.below(4)]);
        }
        // ignition candidates: thick parts of letters, away from the far ends
        let rad = cs.max(1) as i32;
        let mut cand = vec![];
        for i in 0..li.len() {
            let (x, y) = (lx[i] as i32, ly[i] as i32);
            let xf = x as f32;
            if xf <= q.p.lx as f32 + q.p.lw as f32 * 0.15 || xf >= q.p.lx as f32 + q.p.lw as f32 * 0.85 {
                continue;
            }
            let mut thin = false;
            'n: for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && !l.d[ny as usize * w + nx as usize] {
                        thin = true;
                        break 'n;
                    }
                }
            }
            if !thin {
                cand.push((x as u16, y as u16));
            }
        }
        if cand.is_empty() {
            cand = lx.iter().zip(&ly).map(|(&x, &y)| (x, y)).collect();
        }
        // region: logo bbox plus the halo's reach (2cs + 7cs), clipped to the screen
        // (10 cs >= 9 cs + 1 keeps the pooling grid aligned with the logo's cs x cs blocks)
        let m = 10 * cs;
        let (rx0, ry0, rw, rh) = match bb {
            Some((x0, y0, x1, y1)) => {
                let (a, b) = (x0.saturating_sub(m), y0.saturating_sub(m));
                (a, b, (x1 + m).min(w) - a, (y1 + m).min(h) - b)
            }
            None => (0, 0, 0, 0),
        };
        // the space open to the screen edge (hops of up to 4 steps, so thin walls can be crossed, as in Python)
        let mut ext = vec![false; w * h];
        let mut queue: Vec<u32> = vec![];
        if let Some((x0, y0, x1, y1)) = bb {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if l.d[i] {
                        continue;
                    }
                    let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
                    let onb = x == 0 || y == 0 || x == w - 1 || y == h - 1;
                    if !inside {
                        ext[i] = true;
                        if x + 4 >= x0 && x < x1 + 4 && y + 4 >= y0 && y < y1 + 4 {
                            queue.push(i as u32);
                        }
                    } else if onb {
                        ext[i] = true;
                        queue.push(i as u32);
                    }
                }
            }
            let mut offs = vec![];
            for dy in -4i32..=4 {
                for dx in -4i32..=4 {
                    if dx.abs() + dy.abs() <= 4 && (dx, dy) != (0, 0) {
                        offs.push((dx, dy));
                    }
                }
            }
            while let Some(p) = queue.pop() {
                let (px, py) = ((p as usize % w) as i32, (p as usize / w) as i32);
                for &(dx, dy) in &offs {
                    let (nx, ny) = (px + dx, py + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let i = ny as usize * w + nx as usize;
                    if !ext[i] && !l.d[i] {
                        ext[i] = true;
                        queue.push(i as u32);
                    }
                }
            }
        }
        let rn = rw * rh;
        let mut glowok = vec![false; rn];
        let mut rbay = vec![0.0; rn];
        for y in 0..rh {
            for x in 0..rw {
                let i = (y + ry0) * w + x + rx0;
                glowok[y * rw + x] = ext[i] && !q.out.d[i];
                rbay[y * rw + x] = bay(x + rx0, y + ry0);
            }
        }
        let lr = lx.iter().zip(&ly).map(|(&x, &y)| ((y as usize - ry0) * rw + x as usize - rx0) as u32).collect();
        let nlp = li.len();
        let (mut lcell, mut lwt) = (vec![], vec![]);
        for j in 0..nlp {
            let (x, y) = (lx[j] as usize - rx0, ly[j] as usize - ry0);
            let (cx, cy) = (x / cs, y / cs);
            lcell.push((cy * rw.div_ceil(cs) + cx) as u32);
            let area = ((cx * cs + cs).min(rw) - cx * cs) * ((cy * cs + cs).min(rh) - cy * cs);
            lwt.push(1.0 / area as f32);
        }
        let dw = if cs == 1 { 0.3 } else { 0.4 };
        let lmask = l.d.clone();
        Cosmos {
            t: 0.0,
            r,
            li,
            lx,
            ly,
            lb,
            lr,
            fy0,
            tw,
            neb_a,
            neb_b,
            hue,
            dust,
            ramps,
            sxi: vec![0; sx.len()],
            sx,
            sy,
            sl,
            sph,
            scol,
            cand,
            cyc: -1,
            star: (0.0, 0.0),
            pal: 0,
            meteor_t: 15.0,
            shoot: None,
            c_rr: vec![0.0; nlp],
            c_near: vec![0.0; nlp],
            c_core: vec![0.0; nlp],
            c_hm: vec![0.0; nlp],
            rx0,
            ry0,
            rw,
            rh,
            glowok,
            rbay,
            base: vec![[0.0; 3]; rn],
            lcell,
            lwt,
            hx: taps(rw, cs),
            hy: taps(rh, cs),
            hot: vec![false; rn],
            g1: vec![0.0; rn],
            g2: vec![0.0; rn],
            g3: vec![0.0; rn],
            tmp: vec![0.0; rn],
            acc: vec![0.0; rw],
            key: None,
            dw,
            lmask,
            q,
        }
    }

    fn new_cycle(&mut self, k: i64) {
        let i = self.r.below(self.cand.len());
        self.star = match self.cand.get(i) {
            Some(&(x, y)) => (x as f32, y as f32),
            None => (self.q.p.lcx.floor(), self.q.p.lcy.floor()),
        };
        self.pal = k.rem_euclid(PALETTES.len() as i64) as usize;
        self.meteor_t = self.r.range(14.0, 19.0);
        self.shoot = None;
        let (sx, sy) = self.star;
        let (cs, lw) = (self.q.cs as f32, self.q.p.lw as f32);
        for j in 0..self.li.len() {
            let (dx, dy) = (self.lx[j] as f32 - sx, (self.ly[j] as f32 - sy) * 1.3);
            let rr = (dx * dx + dy * dy).sqrt();
            self.c_rr[j] = rr;
            self.c_near[j] = (-rr / (0.35 * lw)).exp();
            let c = rr / (7.0 * cs);
            self.c_core[j] = (-c * c).exp();
            self.c_hm[j] = rr / (0.45 * lw);
        }
    }

    /// Nebula colours of the letters plus the glow around them, into `self.base` (region-sized).
    #[allow(clippy::too_many_arguments)]
    fn render_base(&mut self, o1: usize, o2: usize, ign: f32, fade: f32, rr_front: f32, front_on: bool, wv: f32) {
        let cs = self.q.cs as f32;
        let tw = self.tw;
        let (warm, cool, _, htint) = self.ramps[self.pal];
        let (w2, c2, _, _) = self.ramps[(self.pal + 1) % self.ramps.len()];
        let fw = 6.0 * cs;
        let fk = 1.0 / (2.5 * cs);
        let ie = ign * (1.0 - fade);
        let mw = if cs > 1.0 { 0.55 } else { 0.15 };
        let core_k = ign * (1.0 - fade * 0.7) * 0.4;
        let (rw, rh, c) = (self.rw, self.rh, self.q.cs);
        let (hw, hh) = (rw.div_ceil(c), rh.div_ceil(c));
        self.g3[..hw * hh].iter_mut().for_each(|v| *v = 0.0);
        self.base.iter_mut().for_each(|v| *v = [0.0; 3]);
        for j in 0..self.li.len() {
            let (x, y) = (self.lx[j] as usize, self.ly[j] as usize);
            let row = (y - self.fy0) * tw;
            let mut xa = x + o1;
            if xa >= tw {
                xa -= tw;
            }
            let mut xb = x + o2;
            if xb >= tw {
                xb -= tw;
            }
            let rr = self.c_rr[j];
            let lit = ((rr_front - rr) / fw).clamp(0.0, 1.0) * (1.0 - fade);
            let mut front = 0.0;
            if front_on {
                let d = (rr - rr_front) * fk;
                if d.abs() < 4.5 {
                    front = (-d * d).exp() * ie;
                }
            }
            let dens = 0.62 * self.neb_a[row + xa] + 0.38 * self.neb_b[row + xb];
            let dens = ((dens - 0.28) / 0.5).clamp(0.0, 1.0) * (1.0 - 0.85 * self.dust[row + xb]);
            let ii = 0.34 + lit * (0.22 + 0.5 * self.c_near[j]);
            let v = (dens * ii + front * 0.32 * (0.3 + dens) + core_k * self.c_core[j]).clamp(0.0, 1.0);
            let b = self.lb[j];
            // truncation == floor here (the argument is > 0) and avoids a libm call
            let vi = ((v * (LEVELS - 1) as f32 + 0.5 + (b - 0.5) * self.dw) as usize).min(LEVELS - 1);
            let hm = ie * self.c_hm[j] + (1.0 - ie) * 0.75 + (self.hue[row + xa] - 0.5) * 1.6;
            let m = (hm - 0.3).clamp(0.0, 1.0) + 0.5 + (b - 0.5) * mw >= 1.0;
            let col = if wv > 0.0 && b < wv {
                if m { c2[vi] } else { w2[vi] }
            } else if m {
                cool[vi]
            } else {
                warm[vi]
            };
            let ri = self.lr[j] as usize;
            self.g3[self.lcell[j] as usize] += v * v * self.lwt[j];
            self.base[ri] = col;
        }
        // soft glow spilling out of the letters (Quiet.halo with r1=2, r2=7, k=0.5), only on open space
        // the halo is a very smooth field: blur on a grid pooled by cs (the logo's pixel size, so the radii 2cs and
        // 7cs become 2 and 7 cells) and interpolate bilinearly; only cells near some glow are expanded
        let k = 0.35 + 0.65 * (1.0 - fade) * ign;
        let gc = [htint[0] * k, htint[1] * k, htint[2] * k];
        if rw == 0 {
            return;
        }
        box2(&self.g3, &mut self.g1, &mut self.tmp, &mut self.acc, hw, hh, 2);
        box2(&self.g1, &mut self.g2, &mut self.tmp, &mut self.acc, hw, hh, 2);
        box2(&self.g1, &mut self.g3, &mut self.tmp, &mut self.acc, hw, hh, 7);
        for i in 0..hw * hh {
            self.g1[i] = (self.g2[i] * 1.5 + self.g3[i] * 0.8) * 0.5;
        }
        // cold cell: it and its right/lower neighbours are below the first quantisation step (g*6 + 0.75 < 1)
        const COLD: f32 = 1.0 / 24.0;
        for j in 0..hh {
            let j1 = (j + 1).min(hh - 1);
            for i in 0..hw {
                let i1 = (i + 1).min(hw - 1);
                let m = self.g1[j * hw + i].max(self.g1[j * hw + i1]).max(self.g1[j1 * hw + i]).max(self.g1[j1 * hw + i1]);
                self.hot[j * hw + i] = m >= COLD;
            }
        }
        for y in 0..rh {
            let (j0, j1, fv) = self.hy[y];
            let (r0, r1) = (&self.g1[j0 as usize * hw..(j0 as usize + 1) * hw], &self.g1[j1 as usize * hw..(j1 as usize + 1) * hw]);
            let hot = &self.hot[j0 as usize * hw..(j0 as usize + 1) * hw];
            for x in 0..rw {
                let (i0, i1, fu) = self.hx[x];
                let (i0, i1) = (i0 as usize, i1 as usize);
                let i = y * rw + x;
                if !hot[i0] || !self.glowok[i] {
                    continue;
                }
                let a = r0[i0] + (r0[i1] - r0[i0]) * fu;
                let b = r1[i0] + (r1[i1] - r1[i0]) * fu;
                let g = (a + (b - a) * fv).clamp(0.0, 1.0);
                let v = ((g * 6.0 + 0.5 + (self.rbay[i] - 0.5) * 0.5) as i32).min(6) as f32 / 6.0;
                if v > 0.0 {
                    self.base[i] = [gc[0] * v, gc[1] * v, gc[2] * v];
                }
            }
        }
    }
}

impl Scene for Cosmos {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h) = (img.w, img.h);
        let cs = self.q.cs;
        let csf = cs as f32;
        let k = (t / CYCLE).floor() as i64;
        let ph = t.rem_euclid(CYCLE);
        if k != self.cyc {
            self.cyc = k;
            self.new_cycle(k);
        }
        let (_, _, stint, _) = self.ramps[self.pal];
        let (sx, sy) = self.star;
        // the arc: night -> ignition -> light front sweeps out -> hold -> fades back to night
        let ign = smoothstep(ph, 2.5, 4.0);
        let fade = smoothstep(ph, 23.0, 29.0);
        let (lw, lh) = (self.q.p.lw as f32, self.q.p.lh as f32);
        let rmax = (lw * lw + lh * lh).sqrt();
        let rf = rmax * smoothstep(ph, 3.5, 12.0).powf(0.8);
        let front_on = rf < rmax * 0.98 && ign * (1.0 - fade) > 0.0;
        let tw = self.tw.max(1);
        let o1 = ((t * 0.8 * csf) as usize) % tw;
        let o2 = ((t * 1.6 * csf) as usize) % tw;
        let wv = smoothstep(ph, 28.0, 31.8);
        let key = (k, o1, o2, [ign.to_bits(), fade.to_bits(), rf.to_bits(), wv.to_bits()]);
        if self.key != Some(key) {
            self.key = Some(key);
            self.render_base(o1, o2, ign, fade, rf, front_on, wv);
        }
        // copy the cached base into the frame, black elsewhere
        let (rx0, ry0, rw, rh) = (self.rx0, self.ry0, self.rw, self.rh);
        for y in 0..h {
            let row = &mut img.px[y * w..(y + 1) * w];
            if y < ry0 || y >= ry0 + rh {
                row.fill([0.0; 3]);
            } else {
                row[..rx0].fill([0.0; 3]);
                row[rx0..rx0 + rw].copy_from_slice(&self.base[(y - ry0) * rw..(y - ry0 + 1) * rw]);
                row[rx0 + rw..].fill([0.0; 3]);
            }
        }
        let lm = &self.lmask;
        let on_l = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && lm[y as usize * w + x as usize];
        // stars at three depths, only seen through the letters
        for i in 0..self.sx.len() {
            let sl = self.sl[i] as f32;
            let x = (self.sx[i] - t * (0.5 + sl * 1.2) * csf).rem_euclid(w as f32);
            let xi = (x as i32).min(w as i32 - 1);
            self.sxi[i] = -1;
            if !on_l(xi, self.sy[i]) {
                continue;
            }
            let tw_ = 0.55 + 0.45 * (t * (0.7 + 0.5 * sl) + self.sph[i]).sin();
            let b = (0.4 + sl * 0.3) * tw_;
            let p = &mut img.px[self.sy[i] as usize * w + xi as usize];
            let c = self.scol[i];
            for ch in 0..3 {
                p[ch] = (p[ch] * 0.5 + c[ch] * b).max(p[ch]);
            }
            if self.sl[i] == 2 && tw_ > 0.8 {
                self.sxi[i] = xi;
            }
        }
        if cs > 1 {
            for i in 0..self.sx.len() {
                let xi = self.sxi[i];
                if xi < 0 {
                    continue;
                }
                let c = scale(self.scol[i], 0.5);
                for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let (qx, qy) = ((xi + dx).clamp(0, w as i32 - 1), (self.sy[i] + dy).clamp(0, h as i32 - 1));
                    if on_l(qx, qy) {
                        img.max(qx, qy, c);
                    }
                }
            }
        }
        // the newborn star: core + diffraction spikes, clipped to the letters
        let s = ign * (1.0 - fade * 0.8) * (0.9 + 0.1 * (t * 1.3).sin());
        if s > 0.02 {
            let (ix, iy) = (sx as i32, sy as i32);
            let ln = ((4.0 + 6.0 * s) * csf) as i32;
            for j in -ln..=ln {
                let a = s * (1.0 - j.abs() as f32 / (ln + 1) as f32).powf(1.2);
                let a = (a * 4.0 + 0.5).floor() / 4.0;
                for (qy, qx) in [(iy, ix + j), (iy + j, ix)] {
                    if on_l(qx, qy) {
                        let p = &mut img.px[qy as usize * w + qx as usize];
                        for ch in 0..3 {
                            p[ch] = (p[ch] * (1.0 - a * 0.5) + stint[ch] * a).min(255.0);
                        }
                    }
                }
            }
            if cs > 1 {
                for (dy, dx) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    for j in 1..=cs as i32 {
                        let (qy, qx) = (iy + dy * j, ix + dx * j);
                        if on_l(qx, qy) {
                            let p = &mut img.px[qy as usize * w + qx as usize];
                            for ch in 0..3 {
                                p[ch] = (p[ch] + stint[ch] * s * 0.35).min(255.0);
                            }
                        }
                    }
                }
            }
            disc(img, sx, sy, ((0.4 + 0.5 * s) * csf).max(0.5), scale(stint, (s * 1.2).min(1.0)));
            if s > 0.5 {
                img.set(ix, iy, [255.0; 3]);
            }
        }
        // one meteor per cycle, during the hold
        if self.shoot.is_none() && self.meteor_t < ph && ph < self.meteor_t + 0.1 {
            let x0 = self.r.range(self.q.p.lx as f32 + lw * 0.4, self.q.p.lx as f32 + lw);
            self.shoot = Some((x0, self.q.p.ly as f32 - 2.0 * csf, t));
        }
        if let Some((x0, y0, t0)) = self.shoot {
            let a = t - t0;
            if a > 2.4 {
                self.shoot = None;
            } else {
                let n = 24 * cs;
                let (ux, uy) = (-0.94f32, 0.34f32);
                for j in 0..n {
                    let d = a * 48.0 * csf - j as f32;
                    if d <= 0.0 {
                        continue;
                    }
                    let (px, py) = ((x0 + ux * d) as i32, (y0 + uy * d) as i32);
                    if !on_l(px, py) {
                        continue;
                    }
                    let f = ((1.0 - j as f32 / n as f32).powf(1.4) * 4.0 + 0.5).floor() / 4.0;
                    let k = (f * 2.5).min(1.0);
                    let c = [
                        (255.0 * f + stint[0] * 0.3 * (1.0 - f)) * k,
                        (252.0 * f + stint[1] * 0.3 * (1.0 - f)) * k,
                        (240.0 * f + stint[2] * 0.3 * (1.0 - f)) * k,
                    ];
                    img.max(px, py, c);
                }
            }
        }
    }
}
