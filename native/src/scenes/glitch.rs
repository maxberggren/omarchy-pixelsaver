//! SIGNAL CORRUPT: dithered plasma behind a chromatic-aberrated logo, random glitch bursts with error
//! messages, then a breakdown: datamosh, hex dump, NO SIGNAL test card and a re-sync roll. Port of
//! pixelsaver.py `Glitch`.
//!
//! Speed notes: the plasma is sampled on 2x2 blocks (as in Python), and its four sine terms are split into
//! per-column / per-row tables (angle-addition identities) plus a static per-block sin/cos of the radius, so a
//! block costs a few multiply-adds and no trig. Pixel classes (background / black outline+shadow / logo /
//! aberration zone) are precomputed. On quiet frames base, aberration, scanlines, vignette and the moving
//! band are one fused pass; the previous frame (datamosh source) is only kept when the next frame may need it.
use super::Scene;
use crate::core::*;

const CYCLE: f32 = 8.0;
pub const BARS: [Rgb; 7] = [
    [192.0, 192.0, 192.0],
    [192.0, 192.0, 0.0],
    [0.0, 192.0, 192.0],
    [0.0, 192.0, 0.0],
    [192.0, 0.0, 192.0],
    [192.0, 0.0, 0.0],
    [0.0, 0.0, 192.0],
];
const MSGS: [&str; 8] =
    ["ERR 0XC0FFEE", "SEGFAULT", "NO CARRIER", "0XDEADBEEF", "CORE DUMPED", "STACK SMASHED", "EOF", "PARITY ERR"];
const ASC: &[u8] = b"OMARCHY.#*/";
const HEX: &[u8] = b"0123456789ABCDEF";

const C_BG: u8 = 0;
const C_BLACK: u8 = 1;
const C_LOGO: u8 = 2;
const ZONE: u8 = 4;

pub struct Glitch {
    w: usize,
    h: usize,
    t: f32,
    r: Rng,
    s: usize,
    cls: Vec<u8>,
    /// block grid (2x2 pixels) size
    bw: usize,
    bh: usize,
    /// per block column: u = x2 * 0.09; per block row: sin/cos(0.7 v), 1.3 v
    cu: Vec<f32>,
    rs7: Vec<f32>,
    rc7: Vec<f32>,
    rv13: Vec<f32>,
    /// per block: sin/cos(dist2 * 0.14), bayer threshold
    bsd: Vec<f32>,
    bcd: Vec<f32>,
    bb: Vec<f32>,
    /// per-frame scratch: column / row sine terms, plasma value and background colour per block
    ca: Vec<f32>,
    cs3: Vec<f32>,
    cc3: Vec<f32>,
    rb: Vec<f32>,
    pblk: Vec<f32>,
    cblk: Vec<Rgb>,
    row: Vec<Rgb>,
    chan: Vec<f32>,
    prev: Vec<Rgb>,
    has_prev: bool,
    bg: [Rgb; 5],
    fg: Vec<Rgb>,
    /// vignette: per-row (1 - 0.45 fy^2) and scanline factor, per-column 0.45 fx^2
    ay: Vec<f32>,
    scan: Vec<f32>,
    ax: Vec<f32>,
    next_g: f32,
    g_end: f32,
    msg: (usize, i32, i32),
    msgs: Vec<(Mask, Mask)>,
    nosig: Mask,
    s2: usize,
    glyphs: [u16; 128],
}

fn put_text(img: &mut Canvas, glyphs: &[u16; 128], s: &[u8], x: i32, y: i32, col: Rgb) {
    for (i, &ch) in s.iter().enumerate() {
        let g = glyphs[(ch & 127) as usize];
        for gy in 0..5 {
            for gx in 0..3 {
                if g & (1 << (14 - (gy * 3 + gx))) != 0 {
                    img.set(x + i as i32 * 4 + gx, y + gy, col);
                }
            }
        }
    }
}

impl Glitch {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let pl = place(logo, w, h, 0.8, 0.5, 0.5);
        let s = pl.s;
        let ts = (h / 64).max(1);
        let l = &pl.l;
        let d = l.dilate(1, true);
        let out = d.and_not(l);
        let shadow = d.shift(2 * s as i32, 2 * s as i32).and_not(&d);
        let zone = l.dilate(3, true);
        let cls = (0..w * h)
            .map(|i| {
                let c = if l.d[i] {
                    C_LOGO
                } else if out.d[i] || shadow.d[i] {
                    C_BLACK
                } else {
                    C_BG
                };
                c | if zone.d[i] { ZONE } else { 0 }
            })
            .collect();
        let (bw, bh) = (w.div_ceil(2), h.div_ceil(2));
        let (hw, hh) = (w as f32 / 2.0, h as f32 / 2.0);
        let cu: Vec<f32> = (0..bw).map(|b| (b * 2) as f32 * 0.09).collect();
        let rv: Vec<f32> = (0..bh).map(|b| (b * 2) as f32 * 0.09).collect();
        let mut bsd = vec![0.0; bw * bh];
        let mut bcd = vec![0.0; bw * bh];
        let mut bb = vec![0.0; bw * bh];
        for by in 0..bh {
            for bx in 0..bw {
                let dd = ((bx * 2) as f32 - hw).hypot((by * 2) as f32 - hh) * 0.14;
                let (sn, cs) = dd.sin_cos();
                bsd[by * bw + bx] = sn;
                bcd[by * bw + bx] = cs;
                bb[by * bw + bx] = bay4x2(bx * 2, by * 2);
            }
        }
        let fg = grad(
            &[
                (0.0, [255.0, 0.0, 140.0]),
                (0.25, [255.0, 190.0, 0.0]),
                (0.5, [0.0, 255.0, 190.0]),
                (0.75, [100.0, 40.0, 255.0]),
                (1.0, [255.0, 0.0, 140.0]),
            ],
            256,
        );
        let msgs = MSGS
            .iter()
            .map(|m| {
                let tm = text_mask(m, ts);
                let mut pad = Mask::new(tm.w + 2, tm.h + 2);
                for y in 0..tm.h {
                    for x in 0..tm.w {
                        pad.d[(y + 1) * pad.w + x + 1] = tm.d[y * tm.w + x];
                    }
                }
                (tm, pad.dilate(1, true))
            })
            .collect();
        let s2 = if w > 120 { ts * 2 } else { ts };
        let mut glyphs = [0u16; 128];
        for (c, g) in glyphs.iter_mut().enumerate() {
            *g = glyph(c as u8 as char);
        }
        Glitch {
            w,
            h,
            t: 0.0,
            r: Rng::new(seed),
            s,
            cls,
            bw,
            bh,
            rs7: rv.iter().map(|v| (v * 0.7).sin()).collect(),
            rc7: rv.iter().map(|v| (v * 0.7).cos()).collect(),
            rv13: rv.iter().map(|v| v * 1.3).collect(),
            cu,
            bsd,
            bcd,
            bb,
            ca: vec![0.0; bw],
            cs3: vec![0.0; bw],
            cc3: vec![0.0; bw],
            rb: vec![0.0; bh],
            pblk: vec![0.0; bw * bh],
            cblk: vec![[0.0; 3]; bw * bh],
            row: vec![[0.0; 3]; w],
            chan: vec![0.0; w],
            prev: vec![[0.0; 3]; w * h],
            has_prev: false,
            bg: [[5.0, 3.0, 16.0], [20.0, 8.0, 48.0], [42.0, 20.0, 96.0], [70.0, 50.0, 150.0], [110.0, 100.0, 200.0]],
            fg,
            ay: (0..h).map(|y| 1.0 - 0.45 * ((y as f32 - hh) / h as f32).powi(2)).collect(),
            scan: (0..h).map(|y| if y % 2 == 1 { 0.84 } else { 1.0 }).collect(),
            ax: (0..w).map(|x| 0.45 * ((x as f32 - hw) / w as f32).powi(2)).collect(),
            next_g: 1.0,
            g_end: 0.0,
            msg: (0, 0, 0),
            msgs,
            nosig: text_mask("NO SIGNAL", s2),
            s2,
            glyphs,
        }
    }

    /// Plasma per 2x2 block: value and dithered background colour.
    fn plasma(&mut self, t: f32) {
        let (bw, bh) = (self.bw, self.bh);
        for bx in 0..bw {
            let u = self.cu[bx];
            self.ca[bx] = (u + t).sin();
            let (sn, cs) = (u * 0.7 + t * 0.6).sin_cos();
            self.cs3[bx] = sn;
            self.cc3[bx] = cs;
        }
        for by in 0..bh {
            self.rb[by] = (self.rv13[by] + t * 1.1).sin();
        }
        let (s18, c18) = (t * 1.8).sin_cos();
        for by in 0..bh {
            let (rb, rs, rc) = (self.rb[by], self.rs7[by], self.rc7[by]);
            let o = by * bw;
            for bx in 0..bw {
                let i = o + bx;
                let p = self.ca[bx] + rb + self.cs3[bx] * rc + self.cc3[bx] * rs + self.bsd[i] * c18 - self.bcd[i] * s18;
                let p = (p + 4.0) / 8.0;
                self.pblk[i] = p;
                let idx = (p * 4.0 + self.bb[i] - 0.5).floor().clamp(0.0, 4.0) as usize;
                self.cblk[i] = self.bg[idx];
            }
        }
    }

    /// Base image (plasma, outline/shadow, logo colours, chromatic aberration in the zone) into img.
    /// With `post`, the band, scanlines and vignette are applied in the same pass.
    fn base(&mut self, img: &mut Canvas, t: f32, post: Option<(f32, f32)>) {
        let (w, h, bw) = (self.w, self.h, self.bw);
        let s = self.s % w.max(1);
        let tf = t * 0.12;
        for y in 0..h {
            let o = y * w;
            let bo = (y / 2) * bw;
            let mut anyz = false;
            for x in 0..w {
                let c = self.cls[o + x];
                anyz |= c & ZONE != 0;
                self.row[x] = match c & 3 {
                    C_BG => self.cblk[bo + x / 2],
                    C_BLACK => [0.0; 3],
                    _ => {
                        let v = self.pblk[bo + x / 2] * 1.6 + tf + x as f32 * 0.004;
                        self.fg[((v.rem_euclid(1.0) * 255.0) as usize).min(255)]
                    }
                };
            }
            let (add, k) = match post {
                Some((b0, b1)) => {
                    let yf = y as f32;
                    (if yf >= b0 && yf < b1 { 16.0 } else { 0.0 }, self.scan[y])
                }
                None => (0.0, 1.0),
            };
            let ay = self.ay[y];
            let dst = &mut img.px[o..o + w];
            for x in 0..w {
                let mut c = self.row[x];
                if anyz && self.cls[o + x] & ZONE != 0 {
                    c[0] = self.row[(x + w - s) % w][0];
                    c[2] = self.row[(x + s) % w][2];
                }
                if post.is_some() {
                    let kk = k * (ay - self.ax[x]);
                    c = [(c[0] + add) * kk, (c[1] + add) * kk, (c[2] + add) * kk];
                }
                dst[x] = c;
            }
        }
    }

    fn micro(&mut self, img: &mut Canvas, k: f32) {
        let (w, h) = (self.w, self.h);
        let wi = w as i64;
        let hi = h as i64;
        let n = self.r.int(2, 4 + (10.0 * k) as i64);
        for _ in 0..n {
            let y0 = self.r.int(0, hi) as usize;
            let hh = self.r.int(1, (hi / 10).max(2)) as usize;
            let sg = if self.r.u64() & 1 == 0 { -1 } else { 1 };
            let sh = (sg * self.r.int(2, 4 + (w as f32 * 0.25 * k) as i64)).rem_euclid(wi) as usize;
            for y in y0..(y0 + hh).min(h) {
                img.px[y * w..(y + 1) * w].rotate_right(sh);
            }
        }
        let sp = ((2.0 + 6.0 * k) as usize) % w.max(1);
        for y in 0..h {
            let rowp = &mut img.px[y * w..(y + 1) * w];
            for x in 0..w {
                self.chan[x] = rowp[x][0];
            }
            for x in 0..w {
                rowp[x][0] = self.chan[(x + w - sp) % w];
            }
            for x in 0..w {
                self.chan[x] = rowp[x][2];
            }
            for x in 0..w {
                rowp[x][2] = self.chan[(x + sp) % w];
            }
        }
        if self.has_prev {
            let n = self.r.int(1, 3 + (6.0 * k) as i64);
            for _ in 0..n {
                let bwid = self.r.int(4, (wi / 4).max(5));
                let bhei = self.r.int(2, (hi / 5).max(3));
                let bx = self.r.int(0, wi - bwid);
                let by = self.r.int(0, hi - bhei);
                let dy = self.r.int(-4, 5);
                let dx = self.r.int(-8, 9);
                for y in by.max(0)..(by + bhei).min(hi) {
                    let sy = (y - dy).rem_euclid(hi) as usize;
                    for x in bx.max(0)..(bx + bwid).min(wi) {
                        let sx = (x - dx).rem_euclid(wi) as usize;
                        img.px[y as usize * w + x as usize] = self.prev[sy * w + sx];
                    }
                }
            }
        }
    }

    fn post(&self, img: &mut Canvas, b0: f32, b1: f32, gain: f32) {
        let w = self.w;
        for y in 0..self.h {
            let yf = y as f32;
            let add = if yf >= b0 && yf < b1 { 16.0 } else { 0.0 };
            let k = self.scan[y] * gain;
            let ay = self.ay[y];
            for (x, c) in img.px[y * w..(y + 1) * w].iter_mut().enumerate() {
                let kk = k * (ay - self.ax[x]);
                *c = [(c[0] + add) * kk, (c[1] + add) * kk, (c[2] + add) * kk];
            }
        }
    }

    fn hexdump(&mut self, img: &mut Canvas, t: f32) {
        let (w, h) = (self.w, self.h);
        let nch = w / 4;
        let mut line = [0u8; 48];
        for i in 0..=h / 6 {
            let addr = (((t * 40.0) as usize + i) * 16) & 0xFFFFFF;
            let mut n = 0;
            for sh in (0..6).rev() {
                line[n] = HEX[(addr >> (sh * 4)) & 15];
                n += 1;
            }
            line[n] = b' ';
            n += 1;
            for j in 0..8 {
                let b = self.r.below(256);
                if j > 0 {
                    line[n] = b' ';
                    n += 1;
                }
                line[n] = HEX[b >> 4];
                line[n + 1] = HEX[b & 15];
                n += 2;
            }
            line[n] = b' ';
            n += 1;
            for _ in 0..8 {
                line[n] = ASC[self.r.below(ASC.len())];
                n += 1;
            }
            let n = n.min(nch);
            let y = (i * 6) as i32;
            put_text(img, &self.glyphs, &line[..n.min(7)], 0, y, [255.0, 40.0, 160.0]);
            if n > 7 {
                let col = if i % 5 != 0 { [80.0, 255.0, 150.0] } else { [255.0; 3] };
                put_text(img, &self.glyphs, &line[7..n], 28, y, col);
            }
        }
    }

    fn nosignal(&mut self, img: &mut Canvas, t: f32) {
        let (w, h) = (self.w, self.h);
        let bwf = w as f32 / 7.0;
        let top = ((h as f32 * 0.66) as usize).min(h);
        let h12 = h / 12;
        let col_of = |x: usize| -> usize { (0..7).rev().find(|&i| (i as f32 * bwf) as usize <= x).unwrap_or(0) };
        let (c18, c36) = ((w as f32 * 0.18) as usize, (w as f32 * 0.36) as usize);
        for x in 0..w {
            let i = col_of(x);
            let (a, b) = (BARS[i], scale(BARS[6 - i], (i % 2) as f32));
            let c = if x < c18 {
                [0.0, 30.0, 80.0]
            } else if x < c36 {
                [240.0; 3]
            } else {
                [20.0; 3]
            };
            for y in 0..h {
                img.px[y * w + x] = if y < top {
                    a
                } else if y < top + h12 {
                    b
                } else {
                    c
                };
            }
        }
        for y in 0..h {
            if self.r.f() < 0.12 {
                for p in &mut img.px[y * w..(y + 1) * w] {
                    let v = self.r.below(255) as f32;
                    *p = [v; 3];
                }
            }
        }
        let s2 = self.s2 as i32;
        let (mw, mh) = (self.nosig.w as i32, self.nosig.h as i32);
        let bx = (w as i32 - mw).div_euclid(2);
        let by = (h as i32 - mh).div_euclid(2);
        for y in (by - 3 * s2).max(0)..(by + mh + 3 * s2).min(h as i32) {
            for x in (bx - 4 * s2).max(0)..(bx + mw + 4 * s2).min(w as i32) {
                img.px[y as usize * w + x as usize] = [0.0; 3];
            }
        }
        if ((t * 4.0) as i64) % 2 == 0 {
            blit(img, &self.nosig, bx, by, [255.0; 3]);
        }
    }
}

impl Scene for Glitch {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h) = (self.w, self.h);
        let hf = h as f32;
        let ph = t.rem_euclid(CYCLE);
        let band = (t * 0.12).rem_euclid(1.4) * hf - 0.2 * hf;
        let (b0, b1) = (band, band + (h / 20).max(2) as f32);

        // schedule the glitch bursts (before deciding whether this frame is quiet)
        if ph < 5.2 && t > self.next_g {
            self.g_end = t + self.r.range(0.15, 0.45);
            self.next_g = self.g_end + self.r.range(0.5, 1.4);
            let m = self.r.below(MSGS.len());
            let mx = self.r.int(2, (w as i64 - 60).max(3)) as i32;
            let my = self.r.int(2, (h as i64 - 8).max(3)) as i32;
            self.msg = (m, mx, my);
        }
        let bt = ph - 5.2;
        let quiet = ph < 5.2 && t >= self.g_end;
        if quiet {
            self.plasma(t);
            self.base(img, t, Some((b0, b1)));
        } else {
            let mut gain = 1.0;
            if bt >= 0.9 && bt < 1.6 {
                // hex dump over the dimmed datamosh
                self.plasma(t);
                self.base(img, t, None);
                self.micro(img, 0.6);
                for p in img.px.iter_mut() {
                    *p = scale(*p, 0.22);
                }
                self.hexdump(img, t);
            } else if bt >= 1.6 && bt < 2.3 {
                self.nosignal(img, t);
            } else {
                self.plasma(t);
                self.base(img, t, None);
                if ph < 5.2 {
                    self.micro(img, 0.25);
                    let (m, mx, my) = self.msg;
                    let (tm, dm) = &self.msgs[m];
                    blit(img, dm, mx - 1, my - 1, [0.0; 3]);
                    blit(img, tm, mx, my, [255.0; 3]);
                } else if bt < 0.9 {
                    self.micro(img, bt / 0.9);
                    if self.r.f() < bt {
                        let y0 = self.r.int(0, h as i64 - 4).max(0) as usize;
                        for p in &mut img.px[(y0 * w).min(w * h)..((y0 + h / 6).min(h) * w)] {
                            *p = [255.0 - p[0], 255.0 - p[1], 255.0 - p[2]];
                        }
                    }
                } else {
                    // re-sync roll
                    let k = ((bt - 2.3) / 0.6).min(1.0);
                    let off = (hf * (1.0 - k) * (1.0 - k)) as usize;
                    if off < h {
                        img.px.rotate_right(off * w);
                    }
                    if off > 0 && off < h {
                        img.px[off.saturating_sub(2) * w..off * w].fill([0.0; 3]);
                    }
                    gain = 1.0 + 0.8 * (1.0 - k);
                }
            }
            self.post(img, b0, b1, gain);
        }

        // keep the frame for the datamosh only if the next frame might use it
        let calm_next = ph + 0.25 < 5.2 && t >= self.g_end && t + 0.25 < self.next_g;
        if !calm_next {
            for (d, s) in self.prev.iter_mut().zip(&img.px) {
                *d = [s[0].clamp(0.0, 255.0), s[1].clamp(0.0, 255.0), s[2].clamp(0.0, 255.0)];
            }
            self.has_prev = true;
        }
    }
}
