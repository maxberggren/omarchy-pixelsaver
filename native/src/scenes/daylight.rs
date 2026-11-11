//! DAYLIGHT: one whole day passes through the letters (night, dawn, sun arc, pixel cumulus, sunset, moon,
//! stars, Milky Way). Port of pixelsaver.py `Daylight`.
//!
//! Speed notes: the sky is only ever visible inside the letters, so per frame it is evaluated on a
//! precomputed list of letter pixels only (band index, dither threshold, rel^1.6 etc. stored per pixel).
//! Outside the letters only the halos show: the base halo is a static blurred shape (stored as dither
//! levels 0..6) re-tinted per frame; the sun / moon halos are blurred fields that are recomputed only
//! when the body moves to another whole pixel (or the moon's fade level changes), then re-tinted.
//! Exponentials go through a small lookup table. No allocations per frame.
use super::{QuietBase, Scene};
use crate::core::*;

const CYCLE: f64 = 34.0;
const P0: f64 = 0.03;
const NB: usize = 9;

type Key = (f32, [Rgb; 5], Rgb, Rgb, f32, f32, f32);

#[rustfmt::skip]
const DAY_KEYS: [Key; 12] = [
    (0.00, [[12., 14., 46.], [18., 24., 64.], [28., 36., 86.], [40., 50., 110.], [58., 66., 136.]], [92., 98., 150.], [40., 44., 86.], 0.22, 1.0, 0.0),
    (0.08, [[8., 10., 34.], [16., 18., 54.], [28., 30., 78.], [46., 42., 102.], [70., 56., 122.]], [110., 96., 150.], [44., 40., 86.], 0.24, 0.9, 0.1),
    (0.15, [[20., 22., 66.], [52., 46., 116.], [120., 72., 140.], [214., 104., 120.], [255., 168., 104.]], [220., 130., 150.], [70., 46., 104.], 0.32, 0.35, 0.8),
    (0.21, [[36., 62., 150.], [92., 104., 184.], [196., 130., 156.], [252., 170., 120.], [255., 218., 140.]], [255., 210., 190.], [150., 104., 150.], 0.36, 0.0, 1.0),
    (0.28, [[44., 104., 204.], [76., 140., 226.], [124., 176., 238.], [196., 210., 228.], [250., 228., 180.]], [255., 250., 240.], [168., 178., 214.], 0.42, 0.0, 0.35),
    (0.36, [[30., 86., 204.], [44., 112., 226.], [70., 146., 240.], [108., 180., 248.], [156., 214., 252.]], [255., 255., 255.], [170., 192., 228.], 0.48, 0.0, 0.0),
    (0.50, [[30., 86., 204.], [44., 112., 226.], [70., 146., 240.], [108., 180., 248.], [156., 214., 252.]], [255., 255., 255.], [170., 192., 228.], 0.48, 0.0, 0.0),
    (0.58, [[40., 90., 196.], [66., 122., 216.], [112., 158., 228.], [196., 196., 206.], [252., 214., 156.]], [255., 242., 214.], [180., 170., 196.], 0.44, 0.0, 0.4),
    (0.65, [[44., 34., 110.], [112., 52., 140.], [214., 76., 112.], [252., 126., 70.], [255., 196., 92.]], [255., 196., 120.], [140., 52., 98.], 0.40, 0.0, 1.0),
    (0.71, [[18., 18., 60.], [50., 32., 96.], [110., 44., 112.], [186., 66., 92.], [236., 112., 70.]], [236., 104., 96.], [66., 28., 78.], 0.34, 0.25, 0.7),
    (0.79, [[12., 14., 46.], [18., 24., 64.], [28., 36., 86.], [40., 50., 110.], [58., 66., 136.]], [92., 98., 150.], [40., 44., 86.], 0.22, 1.0, 0.05),
    (1.00, [[12., 14., 46.], [18., 24., 64.], [28., 36., 86.], [40., 50., 110.], [58., 66., 136.]], [92., 98., 150.], [40., 44., 86.], 0.22, 1.0, 0.0),
];

/// (sky stops, cloud lit, cloud shade, star visibility, horizon warmth) at day phase p.
fn day_interp(p: f32) -> ([Rgb; 5], Rgb, Rgb, f32, f32) {
    for k in DAY_KEYS.windows(2) {
        let (k0, k1) = (&k[0], &k[1]);
        if k0.0 <= p && p <= k1.0 {
            let a = (p - k0.0) / (k1.0 - k0.0);
            let a = a * a * (3.0 - 2.0 * a);
            let mut st = [[0.0; 3]; 5];
            for i in 0..5 {
                st[i] = mix(k0.1[i], k1.1[i], a);
            }
            return (st, mix(k0.2, k1.2, a), mix(k0.3, k1.3, a), k0.5 + (k1.5 - k0.5) * a, k0.6 + (k1.6 - k0.6) * a);
        }
    }
    let k = &DAY_KEYS[0];
    (k.1, k.2, k.3, k.5, k.6)
}

/// e^-x for x >= 0 via a linear-interpolated table (x beyond the table -> 0).
struct ExpLut(Vec<f32>);
const EXP_RES: f32 = 64.0;
const EXP_MAX: f32 = 24.0;

impl ExpLut {
    fn new() -> Self {
        let n = (EXP_MAX * EXP_RES) as usize + 2;
        ExpLut((0..n).map(|i| (-(i as f32) / EXP_RES).exp()).collect())
    }
    #[inline]
    fn e(&self, x: f32) -> f32 {
        let x = x.max(0.0) * EXP_RES;
        if x >= EXP_MAX * EXP_RES {
            return 0.0;
        }
        let i = x as usize;
        let f = x - i as f32;
        self.0[i] + (self.0[i + 1] - self.0[i]) * f
    }
}

/// Vertical box blur (edge-clamped) of a w x h field, row-major, with a running row sum.
fn vblur(src: &[f32], dst: &mut [f32], sum: &mut [f32], w: usize, h: usize, r: usize) {
    let k = 1.0 / (2 * r + 1) as f32;
    let row = |y: isize| -> &[f32] {
        let o = y.clamp(0, h as isize - 1) as usize * w;
        &src[o..o + w]
    };
    sum.fill(0.0);
    for j in -(r as isize)..=(r as isize) {
        for (s, v) in sum.iter_mut().zip(row(j)) {
            *s += v;
        }
    }
    for (y, d) in dst.chunks_exact_mut(w).enumerate().take(h) {
        let (a, b) = (row(y as isize + r as isize + 1), row(y as isize - r as isize));
        for (((dv, s), av), bv) in d.iter_mut().zip(sum.iter_mut()).zip(a).zip(b) {
            *dv = *s * k;
            *s += av - bv;
        }
    }
}

/// dst (w x h, column-major view) = transpose of src (h rows of w), cache-blocked.
fn transpose(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    const B: usize = 16;
    for yb in (0..h).step_by(B) {
        for xb in (0..w).step_by(B) {
            for y in yb..(yb + B).min(h) {
                for x in xb..(xb + B).min(w) {
                    dst[x * h + y] = src[y * w + x];
                }
            }
        }
    }
}

/// Box blur on a coarse grid (one cell = f x f pixels) that matches a pixel-space box of width 2*r*f+1:
/// 2r-1 whole cells plus the two end cells at weight (1+1/f)/2. `step` is the element stride along the
/// line, `lstride` the stride between lines.
fn cblur(src: &[f32], dst: &mut [f32], lines: usize, n: usize, step: usize, lstride: usize, r: usize, e: f32) {
    let norm = 1.0 / ((2 * r - 1) as f32 + 2.0 * e);
    let ni = n as isize;
    for l in 0..lines {
        let base = l * lstride;
        let at = |i: isize| src[base + i.clamp(0, ni - 1) as usize * step];
        let ri = r as isize;
        let mut acc = 0.0f32;
        for j in -(ri - 1)..ri {
            acc += at(j);
        }
        for i in 0..ni {
            dst[base + i as usize * step] = (acc + e * (at(i - ri) + at(i + ri))) * norm;
            acc += at(i + ri) - at(i - ri + 1);
        }
    }
}

struct Star {
    y: usize,
    x: usize,
    ph: f32,
    fq: f32,
    big: bool,
    val: f32,
}

/// A letter pixel with its static per-pixel data.
#[derive(Clone, Copy)]
struct Lp {
    /// crop coordinates
    cx: u16,
    cy: u16,
    band: u8,
    top: bool,
    bay: f32,
    rel16: f32,
}

pub struct Daylight {
    t: f64,
    w: usize,
    h: usize,
    cs: usize,
    lx: f32,
    ly: f32,
    lw: f32,
    lh: f32,
    // crop
    x0: usize,
    y0: usize,
    cw: usize,
    ch: usize,
    lc: Vec<bool>,
    outm: Vec<bool>,
    lps: Vec<Lp>,
    /// crop index -> letter pixel index (or u32::MAX)
    lidx: Vec<u32>,
    // halos: dither levels 0..6 over the crop
    qb: Vec<u8>,
    qs: Vec<u8>,
    qm: Vec<u8>,
    sun_key: Option<(i32, i32)>,
    moon_key: Option<(i32, i32, i32)>,
    // night sky
    stw: usize,
    stars: Vec<Star>,
    mway: Vec<f32>,
    dust: Vec<(usize, usize)>,
    // clouds: two label strips (ch x 2cw)
    layers: [Vec<u8>; 2],
    exp: ExpLut,
    // scratch
    sky: Vec<Rgb>,
    colx: Vec<f32>,
    s0: Vec<f32>,
    s1: Vec<f32>,
    s2: Vec<f32>,
    s3: Vec<f32>,
    sum: Vec<f32>,
    // coarse grid for the moving sun / moon halos: cell size f (= cs), origin (gox, goy) in crop coords
    f: usize,
    gw: usize,
    gh: usize,
    gox: i32,
    goy: i32,
    /// letter coverage of each cell (0..1)
    cov: Vec<f32>,
    g0: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    /// coarse rows upsampled to full crop width (gh x cw)
    gup: Vec<f32>,
    /// per crop column / row: the two cells to interpolate between and the weight of the second
    ux: Vec<(u32, u32, f32)>,
    uy: Vec<(u32, u32, f32)>,
}

impl Daylight {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let q = QuietBase::new(w, h, logo);
        let mut r = Rng::new(seed);
        let cs = q.cs;
        let (lx, ly, lw, lh) = (q.p.lx, q.p.ly, q.p.lw, q.p.lh);
        let m = 15 * cs as i32;
        let y0 = (ly - m).max(0) as usize;
        let y1 = ((ly + lh + m).min(h as i32)).max(y0 as i32) as usize;
        let x0 = (lx - m).max(0) as usize;
        let x1 = ((lx + lw + m).min(w as i32)).max(x0 as i32) as usize;
        let (cw, ch) = (x1 - x0, y1 - y0);
        let n = cw * ch;
        let l = &q.p.l;
        let mut lc = vec![false; n];
        let mut outm = vec![false; n];
        let mut lps = vec![];
        let mut lidx = vec![u32::MAX; n];
        for cy in 0..ch {
            for cx in 0..cw {
                let (x, y) = (cx + x0, cy + y0);
                let si = y * w + x;
                let ci = cy * cw + cx;
                outm[ci] = q.out.d[si];
                if l.d[si] {
                    lc[ci] = true;
                    lidx[ci] = lps.len() as u32;
                    let b = bay(x, y);
                    let rel = q.rel[y];
                    lps.push(Lp {
                        cx: cx as u16,
                        cy: cy as u16,
                        band: dith(rel, NB, b, 0.5) as u8,
                        top: q.top.d[si],
                        bay: b,
                        rel16: rel.powf(1.6),
                    });
                }
            }
        }
        let exp = ExpLut::new();
        let mut d = Daylight {
            t: 0.0,
            w,
            h,
            cs,
            lx: lx as f32,
            ly: ly as f32,
            lw: lw as f32,
            lh: lh as f32,
            x0,
            y0,
            cw,
            ch,
            lc,
            outm,
            sky: vec![[0.0; 3]; lps.len()],
            lps,
            lidx,
            qb: vec![0; n],
            qs: vec![0; n],
            qm: vec![0; n],
            sun_key: None,
            moon_key: None,
            stw: 2 * cw,
            stars: vec![],
            mway: vec![],
            dust: vec![],
            layers: [vec![], vec![]],
            exp,
            colx: vec![0.0; cw],
            s0: vec![0.0; n],
            s1: vec![0.0; n],
            s2: vec![0.0; n],
            s3: vec![0.0; n],
            sum: vec![0.0; cw.max(ch)],
            f: 1,
            gw: 0,
            gh: 0,
            gox: 0,
            goy: 0,
            cov: vec![],
            g0: vec![],
            g1: vec![],
            g2: vec![],
            gup: vec![],
            ux: vec![],
            uy: vec![],
        };
        if n == 0 {
            return d;
        }
        // coarse grid aligned to the logo's own pixel grid, so each cell is one logo pixel
        let f = cs as i32;
        let org = |l: i32, c0: usize| {
            let o = (l - c0 as i32).rem_euclid(f);
            if o > 0 { o - f } else { 0 }
        };
        let (gox, goy) = (org(lx, x0), org(ly, y0));
        let gw = ((cw as i32 - gox + f - 1) / f) as usize;
        let gh = ((ch as i32 - goy + f - 1) / f) as usize;
        let mut cov = vec![0.0f32; gw * gh];
        for (ci, &b) in d.lc.iter().enumerate() {
            if b {
                let (cx, cy) = ((ci % cw) as i32, (ci / cw) as i32);
                cov[((cy - goy) / f) as usize * gw + ((cx - gox) / f) as usize] += 1.0 / (f * f) as f32;
            }
        }
        let up = |c: usize, o: i32, n: usize| -> (u32, u32, f32) {
            let u = ((c as i32 - o) as f32 - (f - 1) as f32 * 0.5) / f as f32;
            let i0 = (u.floor().max(0.0) as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            ((i0) as u32, i1 as u32, (u - i0 as f32).clamp(0.0, 1.0))
        };
        d.ux = (0..cw).map(|c| up(c, gox, gw)).collect();
        d.uy = (0..ch).map(|c| up(c, goy, gh)).collect();
        d.f = cs;
        d.gw = gw;
        d.gh = gh;
        d.gox = gox;
        d.goy = goy;
        d.cov = cov;
        d.g0 = vec![0.0; gw * gh];
        d.g1 = vec![0.0; gw * gh];
        d.g2 = vec![0.0; gw * gh];
        d.gup = vec![0.0; gh * cw];
        // static base halo of the whole logo
        for (s, &b) in d.s0.iter_mut().zip(&d.lc) {
            *s = if b { 1.0 } else { 0.0 };
        }
        let mut qb = std::mem::take(&mut d.qb);
        d.halo_q(&mut qb, 3, 7, 0.75);
        d.qb = qb;

        // night sky: a scrolling strip (2w wide) of stars and a faint diagonal milky way
        let tw = d.stw;
        let nst = 40.max(l.count() * 2 / 21);
        d.stars = (0..nst)
            .map(|_| Star {
                y: r.below(ch),
                x: r.below(tw),
                ph: r.range(0.0, 6.28),
                fq: r.range(0.4, 1.1),
                big: r.f() < 0.12,
                val: r.range(0.45, 1.0),
            })
            .collect();
        // noise tiled twice horizontally, blurred, normalised
        let mut nz = Grid::new(tw, ch);
        for y in 0..ch {
            for x in 0..cw {
                let v = r.f();
                nz.d[y * tw + x] = v;
                nz.d[y * tw + x + cw] = v;
            }
        }
        let nz = box_blur(&nz, 2 * cs);
        let nn = nz.d.len() as f32;
        let mean = nz.d.iter().sum::<f32>() / nn;
        let sd = (nz.d.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / nn).sqrt() + 1e-6;
        let hf = ch as f32;
        let mut mw = vec![0.0f32; ch * tw];
        for y in 0..ch {
            for x in 0..tw {
                let band = (x as f32 / tw as f32 * std::f32::consts::TAU).sin() * hf * 0.22 + hf * 0.5 - y as f32;
                let i = y * tw + x;
                let z = band / (hf * 0.16);
                mw[i] = ((-z * z).exp() * (0.75 + 0.35 * (nz.d[i] - mean) / sd)).clamp(0.0, 1.0);
            }
        }
        // extra faint dust stars concentrated in the band
        for _ in 0..nst * 3 {
            let (dy, dx) = (r.below(ch), r.below(tw));
            if r.f() < mw[dy * tw + dx] * 0.8 {
                d.dust.push((dy, dx));
            }
        }
        d.mway = mw;
        d.layers = [d.make_clouds(&mut r, 0.55, 3, 0.0), d.make_clouds(&mut r, 0.78, 3, 1.0)];
        d
    }

    /// A tileable strip (2w wide): 0 sky, 1 body, 2 lit rim, 3 shade, 4 lobe line.
    fn make_clouds(&self, r: &mut Rng, size: f32, n: usize, yb: f32) -> Vec<u8> {
        let (cs, lh, h) = (self.cs as i32, self.lh, self.ch);
        let tw = 2 * self.cw;
        let twf = tw as f32;
        let mut lab = vec![0u8; h * tw];
        let mut m = vec![false; h * tw];
        let mut lobe = vec![false; h * tw];
        for i in 0..n {
            let cx = (i as f32 + r.range(0.2, 0.8)) * twf / n as f32;
            let base = self.ly - self.y0 as f32 + lh * (r.range(0.45, 0.8) * yb + 0.32 * (1.0 - yb));
            let k = r.int(3, 6) as usize;
            let width = lh * size * (0.45 + 0.32 * k as f32) * r.range(0.9, 1.1);
            let mut puffs: Vec<(f32, f32, f32)> = (0..k)
                .map(|j| {
                    let f = (j as f32 + 0.5) / k as f32;
                    let pr = lh * size * (0.15 + 0.22 * (std::f32::consts::PI * f).sin().powf(0.7)) * r.range(0.85, 1.15);
                    (pr, cx + (f - 0.5) * width, base - pr * r.range(0.6, 0.9))
                })
                .collect();
            puffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            m.iter_mut().for_each(|v| *v = false);
            lobe.iter_mut().for_each(|v| *v = false);
            let mut bx0 = tw as i32;
            let mut bx1 = 0i32;
            for &(pr, px, py) in &puffs {
                for o in [-twf, 0.0, twf] {
                    let pcx = px + o;
                    let xa = ((pcx - pr).floor() as i32 - 1).max(0);
                    let xb = ((pcx + pr).ceil() as i32 + 1).min(tw as i32 - 1);
                    let ya = ((py - pr / 1.1).floor() as i32 - 1).max(0);
                    let yb2 = ((py + pr / 1.1).ceil() as i32 + 1).min(h as i32 - 1);
                    if xa > xb || ya > yb2 {
                        continue;
                    }
                    bx0 = bx0.min(xa);
                    bx1 = bx1.max(xb);
                    for y in ya..=yb2 {
                        let yf = y as f32;
                        if yf > base {
                            continue;
                        }
                        let dy = (yf - py) * 1.1;
                        for x in xa..=xb {
                            let dx = x as f32 - pcx;
                            let dd = (dx * dx + dy * dy).sqrt();
                            if dd < pr {
                                let ii = y as usize * tw + x as usize;
                                let rim = dd > pr - 0.8 * cs as f32 && yf < py - pr * 0.2;
                                lobe[ii] = rim && m[ii];
                                m[ii] = true;
                            }
                        }
                    }
                }
            }
            let top = puffs.iter().map(|p| p.2 - p.0).fold(f32::MAX, f32::min);
            let span = (base - top).max(1.0);
            for y in 0..h {
                let hgt = ((y as f32 - top) / span).clamp(0.0, 1.0);
                for x in bx0.max(0) as usize..=(bx1.max(0) as usize).min(tw - 1) {
                    let ii = y * tw + x;
                    if !m[ii] {
                        continue;
                    }
                    let mut v = 1u8;
                    if lobe[ii] {
                        v = 4;
                    }
                    if hgt + (bay(x, y) - 0.5) * 0.16 > 0.72 {
                        v = 3;
                    }
                    let up = y as i32 - cs;
                    if up < 0 || !m[up as usize * tw + x] {
                        v = 2;
                    }
                    let lf = x as i32 - cs;
                    if (lf < 0 || !m[y * tw + lf as usize]) && hgt < 0.6 {
                        v = 2;
                    }
                    lab[ii] = v;
                }
            }
        }
        lab
    }

    /// `halo_c` blur of the field in s0 (consumed), quantised to dither levels 0..6 into `out`.
    fn halo_q(&mut self, out: &mut [u8], r1: usize, r2: usize, k: f32) {
        let (w, h, cs) = (self.cw, self.ch, self.cs);
        // Separable edge-clamped box blurs. Horizontal passes run as vertical passes on the transposed
        // field (a running sum per row vectorises; a running sum along a row is one long serial chain).
        let sum = &mut self.sum;
        vblur(&self.s0, &mut self.s1, sum, w, h, r1 * cs);
        transpose(&self.s1, &mut self.s2, w, h);
        vblur(&self.s2, &mut self.s1, sum, h, w, r1 * cs); // s1 = g, transposed
        vblur(&self.s1, &mut self.s2, sum, h, w, r1 * cs);
        transpose(&self.s2, &mut self.s0, h, w);
        vblur(&self.s0, &mut self.s3, sum, w, h, r1 * cs); // s3 = a
        vblur(&self.s1, &mut self.s2, sum, h, w, r2 * cs);
        transpose(&self.s2, &mut self.s1, h, w);
        vblur(&self.s1, &mut self.s0, sum, w, h, r2 * cs); // s0 = b
        let (ka, kb) = (1.5 * k * 6.0, 0.8 * k * 6.0);
        for (y, ((o, a), b)) in out.chunks_exact_mut(w).zip(self.s3.chunks_exact(w)).zip(self.s0.chunks_exact(w)).enumerate() {
            let br = &B8[(y + self.y0) & 7];
            let mut th = [0.0f32; 8];
            for (j, t) in th.iter_mut().enumerate() {
                *t = 0.5 + (br[(j + self.x0) & 7] - 0.5) * 0.5;
            }
            for (x, ((ov, av), bv)) in o.iter_mut().zip(a).zip(b).enumerate() {
                // clamp(v,0,1)*6 then dither; values are >= 0 so the low clamp is a no-op
                let v = (av * ka + bv * kb).min(6.0) + th[x & 7];
                *ov = (v as u8).min(6);
            }
        }
    }

    /// Halo of letters * exp(-dist/sc) * vis around (bx, by), computed on the coarse grid (the field is
    /// smooth at the cell scale), upsampled bilinearly and quantised to dither levels 0..6.
    fn body_halo(&mut self, sun: bool, bx: f32, by: f32, sc: f32, vis: f32, r2: usize) {
        let (gw, gh, f) = (self.gw, self.gh, self.f);
        let ys = if sun { 1.1 } else { 1.0 };
        let inv = 1.0 / sc;
        let half = (f as f32 - 1.0) * 0.5;
        for gy in 0..gh {
            let y = (self.y0 as i32 + self.goy + (gy * f) as i32) as f32 + half;
            let dy = (y - by) * ys;
            for gx in 0..gw {
                let i = gy * gw + gx;
                let c = self.cov[i];
                self.g0[i] = if c > 0.0 {
                    let x = (self.x0 as i32 + self.gox + (gx * f) as i32) as f32 + half;
                    let dx = x - bx;
                    c * self.exp.e((dx * dx + dy * dy).sqrt() * inv) * vis
                } else {
                    0.0
                };
            }
        }
        let e = (1.0 + 1.0 / f as f32) * 0.5;
        let (r1, r2) = (3, r2);
        // g = blur(lum, r1): vertical then horizontal
        cblur(&self.g0, &mut self.g1, gw, gh, gw, 1, r1, e);
        cblur(&self.g1, &mut self.g0, gh, gw, 1, gw, r1, e); // g0 = g
        cblur(&self.g0, &mut self.g1, gw, gh, gw, 1, r1, e);
        cblur(&self.g1, &mut self.g2, gh, gw, 1, gw, r1, e); // g2 = a
        cblur(&self.g0, &mut self.g1, gw, gh, gw, 1, r2, e);
        cblur(&self.g1, &mut self.g0, gh, gw, 1, gw, r2, e); // g0 = b
        let k = 1.2 * 6.0;
        for (a, b) in self.g2.iter_mut().zip(&self.g0) {
            *a = ((*a * 1.5 + b * 0.8) * k).min(6.0);
        }
        // upsample rows to full width
        let cw = self.cw;
        for gy in 0..gh {
            let src = &self.g2[gy * gw..gy * gw + gw];
            for (o, &(i0, i1, t)) in self.gup[gy * cw..gy * cw + cw].iter_mut().zip(&self.ux) {
                let (a, b) = (src[i0 as usize], src[i1 as usize]);
                *o = a + (b - a) * t;
            }
        }
        let out = if sun { &mut self.qs } else { &mut self.qm };
        for (y, o) in out.chunks_exact_mut(cw).enumerate() {
            let (i0, i1, t) = self.uy[y];
            let ra = &self.gup[i0 as usize * cw..i0 as usize * cw + cw];
            let rb = &self.gup[i1 as usize * cw..i1 as usize * cw + cw];
            let br = &B8[(y + self.y0) & 7];
            let mut th = [0.0f32; 8];
            for (j, tv) in th.iter_mut().enumerate() {
                *tv = 0.5 + (br[(j + self.x0) & 7] - 0.5) * 0.5;
            }
            for (x, ((ov, a), b)) in o.iter_mut().zip(ra).zip(rb).enumerate() {
                *ov = ((a + (b - a) * t) + th[x & 7]) as u8;
            }
        }
    }
}

impl Scene for Daylight {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt as f64;
        let (w, h) = (self.w, self.h);
        let (x0, y0, cw, ch) = (self.x0, self.y0, self.cw, self.ch);
        if cw == 0 || ch == 0 {
            img.fill([0.0; 3]);
            return;
        }
        let t = self.t;
        let tf = t as f32;
        let cs = self.cs;
        let csf = cs as f32;
        let (lx, ly, lw, lh) = (self.lx, self.ly, self.lw, self.lh);
        let p = (t / CYCLE + P0).rem_euclid(1.0) as f32;
        let (stops, clit, cshade, stars, warm) = day_interp(p);

        let mut bands = [[0.0f32; 3]; NB];
        for (i, b) in bands.iter_mut().enumerate() {
            let f = i as f32 * 4.0 / (NB - 1) as f32;
            let lo = (f as usize).min(3);
            let hi = (f as usize + 1).min(4);
            let fr = f - lo as f32;
            *b = add(scale(stops[lo], 1.0 - fr), scale(stops[hi], fr));
        }

        // --- sun parameters
        let u = (p - 0.135) / 0.545;
        let sun_on = -0.06 < u && u < 1.06;
        let (mut ssx, mut ssy, mut sr, mut low) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut glowc, mut core_c, mut rimc, mut suncol) = ([0.0f32; 3], [0.0f32; 3], [0.0f32; 3], [0.0f32; 3]);
        if sun_on {
            let uc = u.clamp(0.0, 1.0);
            let alt = (std::f32::consts::PI * uc).sin();
            ssx = lx + lw * (0.04 + 0.92 * u);
            ssy = ly + lh * (1.12 - alt);
            low = (1.0 - alt).powf(1.5);
            sr = (lh * (0.17 + 0.05 * low)).max(2.6);
            glowc = mix([255.0, 214.0, 140.0], [255.0, 120.0, 50.0], low);
            core_c = mix([255.0, 252.0, 236.0], [255.0, 190.0, 96.0], low);
            rimc = mix([255.0, 232.0, 150.0], [255.0, 128.0, 56.0], low);
            suncol = scale(glowc, (0.3 + 0.35 * low) * (((0.56 - (u - 0.5).abs()) * 6.0).min(1.0)));
            // horizon glow column factor
            let inv = 1.0 / (lw * 0.35);
            for (cx, c) in self.colx.iter_mut().enumerate() {
                let z = ((cx + x0) as f32 - ssx) * inv;
                *c = self.exp.e(z * z) * warm * 0.55;
            }
            // the halo field is smooth (scale ~lh): re-blur only when the sun crosses a cs/2-pixel step
            let st = (cs / 2).max(1) as f32;
            let key = ((ssx / st).round() as i32, (ssy / st).round() as i32);
            if self.sun_key != Some(key) {
                self.sun_key = Some(key);
                self.body_halo(true, key.0 as f32 * st, key.1 as f32 * st, lh * 0.7, 1.0, 9);
            }
        }
        let rring = sr + 1.2 * csf;
        let ginv = 1.0 / (lh * 0.5);
        let gk = 0.55 + 0.25 * low;

        // --- pass 1: bands, sun glow and disc, milky way
        let stw = self.stw;
        let soff = (t * 0.5 * cs as f64) as usize % stw;
        let star_on = stars > 0.02;
        let mwc = [46.0 * 0.85, 40.0 * 0.85, 84.0 * 0.85];
        for (lp, s) in self.lps.iter().zip(self.sky.iter_mut()) {
            let mut c = bands[lp.band as usize];
            let b = lp.bay;
            if sun_on {
                let dx = (lp.cx as usize + x0) as f32 - ssx;
                let dy = ((lp.cy as usize + y0) as f32 - ssy) * 1.1;
                let d = (dx * dx + dy * dy).sqrt();
                if d < sr {
                    c = core_c;
                } else if d < rring {
                    c = rimc;
                } else {
                    let g = self.exp.e((d - sr) * ginv) * gk + self.colx[lp.cx as usize] * lp.rel16;
                    let q = qd(g.clamp(0.0, 1.0), 6.0, b, 0.6);
                    if q > 0.0 {
                        for ch in 0..3 {
                            c[ch] += q * (glowc[ch] - c[ch] * 0.55);
                        }
                    }
                }
            }
            if star_on {
                let mv = self.mway[lp.cy as usize * stw + (lp.cx as usize + stw - soff) % stw];
                let q = qd(mv * stars, 4.0, b, 0.7);
                if q > 0.0 {
                    c = add(c, scale(mwc, q));
                }
            }
            *s = c;
        }
        // --- stars and dust (only those landing on a letter matter)
        if star_on {
            let dc = scale([60.0, 56.0, 90.0], stars);
            for &(dy, dx) in &self.dust {
                let x = (dx + soff) % stw;
                if x < cw {
                    let k = self.lidx[dy * cw + x];
                    if k != u32::MAX {
                        let s = &mut self.sky[k as usize];
                        *s = add(*s, dc);
                    }
                }
            }
            let sc = [240.0, 240.0, 255.0];
            for st in &self.stars {
                let x = (st.x + soff) % stw;
                if x >= cw {
                    continue;
                }
                let k = self.lidx[st.y * cw + x];
                let big = cs > 1 && st.big;
                if k == u32::MAX && !big {
                    continue;
                }
                let a = stars * st.val * (0.55 + 0.45 * (tf * st.fq + st.ph).sin());
                if k != u32::MAX {
                    let s = &mut self.sky[k as usize];
                    *s = mix(*s, sc, a);
                }
                if big {
                    for (oy, ox) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                        let (yy, xx) = (st.y as i32 + oy, x as i32 + ox);
                        if yy >= 0 && xx >= 0 && (yy as usize) < ch && (xx as usize) < cw {
                            let k = self.lidx[yy as usize * cw + xx as usize];
                            if k != u32::MAX {
                                let s = &mut self.sky[k as usize];
                                *s = mix(*s, sc, a * 0.5);
                            }
                        }
                    }
                }
            }
        }

        // --- moon
        let v = (p - 0.70).rem_euclid(1.0) / 0.42;
        let moon_on = 0.0 < v && v < 1.0;
        if moon_on {
            let alt = (std::f32::consts::PI * v).sin();
            let mx = lx + lw * (0.06 + 0.88 * v);
            let my = ly + lh * (1.0 - 0.8 * alt);
            let vis = smoothstep(v, 0.0, 0.1) * (1.0 - smoothstep(v, 0.9, 1.0));
            let mr = (lh * 0.17).max(2.4);
            let minv = 1.0 / (lh * 0.35);
            let mgc = [140.0, 150.0, 200.0];
            let lit = [240.0 * vis, 238.0 * vis, 214.0 * vis];
            let drk = [40.0 * 0.5 * vis, 44.0 * 0.5 * vis, 78.0 * 0.5 * vis];
            let (dcx, dcy, dr2) = (mx + mr * 0.55, my - mr * 0.3, (mr * 0.9) * (mr * 0.9));
            for (lp, s) in self.lps.iter().zip(self.sky.iter_mut()) {
                let (x, y) = ((lp.cx as usize + x0) as f32, (lp.cy as usize + y0) as f32);
                let (dx, dy) = (x - mx, y - my);
                let d = (dx * dx + dy * dy).sqrt();
                let mg = self.exp.e((d - mr) * minv) * 0.35 * vis;
                let q = qd(mg, 5.0, lp.bay, 0.6);
                let mut c = if q > 0.0 { add(*s, scale(mgc, q)) } else { *s };
                if d < mr {
                    let (ex, ey) = (x - dcx, y - dcy);
                    c = if ex * ex + ey * ey < dr2 { add(scale(c, 1.0 - 0.5 * vis), drk) } else { add(scale(c, 1.0 - vis), lit) };
                }
                *s = c;
            }
            let st = (cs / 2).max(1) as f32;
            let key = ((mx / st).round() as i32, (my / st).round() as i32, (vis * 40.0).round() as i32);
            if self.moon_key != Some(key) {
                self.moon_key = Some(key);
                self.body_halo(false, key.0 as f32 * st, key.1 as f32 * st, lh * 0.6, key.2 as f32 / 40.0, 9);
            }
        }

        // --- clouds, top highlight, write letters
        let body = add(scale(clit, 0.6), scale(cshade, 0.4));
        let pal = [[0.0f32; 3], body, clit, cshade, add(scale(cshade, 0.6), scale(body, 0.4))];
        let ctw = 2 * cw;
        let offs = [
            (t * 0.9 * cs as f64 + 37.0 * 0.9) as usize % ctw,
            (t * 1.6 * cs as f64 + 37.0 * 1.6) as usize % ctw,
        ];
        let sinv = 1.0 / (lh * 0.7);
        for (lp, s) in self.lps.iter().zip(self.sky.iter()) {
            let mut c = *s;
            let row = lp.cy as usize * ctw;
            let xi = lp.cx as usize + ctw;
            for li in 0..2 {
                let lab = self.layers[li][row + (xi - offs[li]) % ctw];
                if lab > 0 {
                    c = pal[lab as usize];
                    if sun_on && lab != 3 {
                        let dx = (lp.cx as usize + x0) as f32 - ssx;
                        let dy = ((lp.cy as usize + y0) as f32 - ssy) * 1.1;
                        let q = qd(self.exp.e((dx * dx + dy * dy).sqrt() * sinv), 4.0, lp.bay, 0.5) * 0.7;
                        if q > 0.0 {
                            for ch in 0..3 {
                                c[ch] += q * (glowc[ch] - c[ch]);
                            }
                        }
                    }
                }
            }
            if lp.top {
                c = [c[0] * 0.8 + 51.0, c[1] * 0.8 + 51.0, c[2] * 0.8 + 51.0];
            }
            img.px[(lp.cy as usize + y0) * w + lp.cx as usize + x0] = c;
        }

        // --- halos around the letters (crop), black elsewhere
        let mut glow = add(scale(stops[1], 0.4), scale(stops[2], 0.6));
        let gm = (glow[0] + glow[1] + glow[2]) / 3.0;
        for g in glow.iter_mut() {
            *g = (gm + (*g - gm) * 1.7).clamp(0.0, 255.0);
        }
        let gb = scale(glow, 0.36 / 6.0);
        let gs = if sun_on { scale(suncol, 1.0 / 6.0) } else { [0.0; 3] };
        let gmn = if moon_on { [70.0 / 6.0, 80.0 / 6.0, 120.0 / 6.0] } else { [0.0; 3] };
        // per-level colour tables (7 levels each)
        let mut tb = [[0.0f32; 3]; 7];
        let mut ts = [[0.0f32; 3]; 7];
        let mut tm = [[0.0f32; 3]; 7];
        for i in 0..7 {
            tb[i] = scale(gb, i as f32);
            ts[i] = scale(gs, i as f32);
            tm[i] = scale(gmn, i as f32);
        }
        let black = [0.0f32; 3];
        for y in 0..h {
            let row = &mut img.px[y * w..(y + 1) * w];
            if y < y0 || y >= y0 + ch {
                row.fill(black);
                continue;
            }
            let cy = y - y0;
            row[..x0].fill(black);
            row[x0 + cw..].fill(black);
            let cr = cy * cw;
            let (lc, om) = (&self.lc[cr..cr + cw], &self.outm[cr..cr + cw]);
            let (qb, qs, qm) = (&self.qb[cr..cr + cw], &self.qs[cr..cr + cw], &self.qm[cr..cr + cw]);
            let dst = &mut row[x0..x0 + cw];
            for cx in 0..cw {
                if lc[cx] {
                    continue;
                }
                let (a, b, m) = (tb[qb[cx] as usize], if sun_on { ts[qs[cx] as usize] } else { black }, if moon_on { tm[qm[cx] as usize] } else { black });
                let mut c = [a[0] + b[0] + m[0], a[1] + b[1] + m[1], a[2] + b[2] + m[2]];
                if om[cx] {
                    c = scale(c, 0.25);
                }
                dst[cx] = c;
            }
        }
    }
}
