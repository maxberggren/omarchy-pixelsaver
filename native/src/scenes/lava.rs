//! MOLTEN: a lava lamp inside the letters, metaball blobs drifting up and down with a dim ghost of them in
//! the background and a warm glow around the word. Port of pixelsaver.py `Lava`.
//! Per frame: one vectorised metaball pass over the screen (the ghost needs it everywhere), a letter-pixel
//! pass for the fill levels, and a separable glow blur whose horizontal passes only touch the letter rows.
//! Everything static (vignette, scanlines, shadow, rim, dither offsets, palette+top colours) is precomputed.
use super::Scene;
use crate::core::*;

const BASE: Rgb = [8.0, 3.0, 10.0];
const GHOST: Rgb = [34.0, 6.0, 16.0];
const GLOW: Rgb = [255.0 * 0.62, 84.0 * 0.62, 24.0 * 0.62];
const PAL: [Rgb; 5] = [[122.0, 16.0, 44.0], [176.0, 28.0, 44.0], [228.0, 62.0, 36.0], [255.0, 152.0, 40.0], [255.0, 236.0, 150.0]];
const NB: usize = 8;

/// Vertical box blur of a `w`-wide, `h`-row buffer: output rows [ya, yb), input rows clamped to [0, h).
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

struct Px {
    /// screen index
    i: u32,
    /// index into the transposed letter band of the glow region (x-major)
    ti: u32,
    /// row ramp + dither jitter (static part of the fill field)
    f0: f32,
    top: bool,
}

pub struct Lava {
    t: f32,
    w: usize,
    h: usize,
    cs: f32,
    lx: f32,
    lw: f32,
    lcy: f32,
    lh: f32,
    bx: [f32; NB],
    br: [f32; NB],
    bp1: [f32; NB],
    bp2: [f32; NB],
    bsp: [f32; NB],
    xf: Vec<f32>,
    /// metaball field (blob only) for the whole screen
    fb: Vec<f32>,
    /// per pixel: crt vignette*scanline, times 0.25 in the drop shadow
    kbg: Vec<f32>,
    /// per pixel: crt factor alone (letters, rim)
    kcrt: Vec<f32>,
    /// ghost vignette
    vg: Vec<f32>,
    px: Vec<Px>,
    /// fill level per screen pixel (only letter pixels meaningful)
    idx: Vec<u8>,
    is_l: Vec<bool>,
    rim: Vec<u32>,
    bubble_src: Vec<(f32, f32)>,
    bub: Particles,
    // glow region (letter bbox + 20cs), local buffers
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
    ly0: usize,
    ly1: usize,
    /// letter band, transposed (rw lines of ly1-ly0): horizontal blurs become vectorised column passes
    lum_t: Vec<f32>,
    ta_t: Vec<f32>,
    tb_t: Vec<f32>,
    h4: Vec<f32>,
    h10: Vec<f32>,
    tmp: Vec<f32>,
    g4: Vec<f32>,
    g10: Vec<f32>,
    dg: Vec<f32>,
    dl: Vec<f32>,
    colsum: Vec<f32>,
    glrow: Vec<f32>,
    rng: Rng,
}

impl Lava {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let p = place(logo, w, h, 0.8, 0.5, 0.5);
        let mut r = Rng::new(seed);
        let csu = p.s.max(1);
        let cs = csu as f32;
        let l = &p.l;
        let (lx, lw, lh) = (p.lx as f32, p.lw as f32, p.lh as f32);
        let mut bx = [0.0; NB];
        let mut br = [0.0; NB];
        let mut bp1 = [0.0; NB];
        let mut bp2 = [0.0; NB];
        let mut bsp = [0.0; NB];
        for i in 0..NB {
            bx[i] = lx + (i as f32 + 0.5) / NB as f32 * lw + r.range(-3.0, 3.0) * cs;
        }
        for v in br.iter_mut() {
            *v = r.range(5.0, 8.0) * cs;
        }
        for i in 0..NB {
            bp1[i] = r.range(0.0, 6.28);
            bp2[i] = r.range(0.0, 6.28);
        }
        for v in bsp.iter_mut() {
            *v = r.range(0.16, 0.32);
        }
        let d = l.dilate(1, true);
        let out = d.and_not(l);
        let shadow = d.shift(2 * csu as i32, 2 * csu as i32).and_not(&d);
        let top = l.edge(0, -1);
        let (wf, hf) = (w as f32, h as f32);
        let mut kbg = vec![0.0f32; w * h];
        let mut kcrt = vec![0.0f32; w * h];
        let mut vg = vec![0.0f32; w * h];
        for y in 0..h {
            let fy = (y as f32 - hf / 2.0) / hf;
            let sc = if y % 2 == 1 { 0.88 } else { 1.0 };
            for x in 0..w {
                let fx = (x as f32 - wf / 2.0) / wf;
                let i = y * w + x;
                kcrt[i] = sc * (1.0 - 0.4 * (fx * fx + fy * fy));
                kbg[i] = kcrt[i] * if shadow.d[i] { 0.25 } else { 1.0 };
                vg[i] = 1.0 - 0.6 * (fx * fx + fy * fy);
            }
        }
        // glow region: letter bbox + the reach of the 10cs tent blur
        let margin = 20 * csu + 1;
        let bb = l.bbox();
        let (bx0, by0, bx1, by1) = bb.unwrap_or((0, 0, 0, 0));
        let (rx0, ry0) = (bx0.saturating_sub(margin), by0.saturating_sub(margin));
        let (rw, rh) = if bb.is_some() { ((bx1 + margin).min(w) - rx0, (by1 + margin).min(h) - ry0) } else { (0, 0) };
        let (ly0, ly1) = if bb.is_some() { (by0 - ry0, by1 - ry0) } else { (0, 0) };
        let jit = if csu > 1 { 0.07 } else { 0.0 };
        let (ly, lhf) = (p.ly as f32, lh);
        let mut px = vec![];
        let mut bubble_src = vec![];
        for y in 0..h {
            let yf = y as f32;
            let ramp = ((yf - (ly + lhf - 3.5 * cs)) / (3.0 * cs)).clamp(0.0, 1.2) * 0.9
                + (((ly + 1.5 * cs) - yf) / (3.0 * cs)).clamp(0.0, 1.0) * 0.35;
            for x in 0..w {
                let i = y * w + x;
                if l.d[i] {
                    px.push(Px {
                        i: i as u32,
                        ti: ((x - rx0) * (ly1 - ly0) + y - ry0 - ly0) as u32,
                        f0: ramp + (bay(x, y) - 0.5) * jit,
                        top: top.d[i],
                    });
                    if yf > ly + lhf * 0.6 {
                        bubble_src.push((x as f32, yf));
                    }
                }
            }
        }
        let rim = (0..w * h).filter(|&i| out.d[i]).map(|i| i as u32).collect();
        let z = vec![0.0f32; rw * rh];
        Lava {
            t: 0.0,
            w,
            h,
            cs,
            lx,
            lw,
            lcy: p.lcy,
            lh,
            bx,
            br,
            bp1,
            bp2,
            bsp,
            xf: (0..w).map(|x| x as f32).collect(),
            fb: vec![0.0; w * h],
            kbg,
            kcrt,
            vg,
            px,
            idx: vec![0; w * h],
            is_l: l.d.clone(),
            rim,
            bubble_src,
            bub: Particles::default(),
            rx0,
            ry0,
            rw,
            rh,
            ly0,
            ly1,
            lum_t: vec![0.0; rw * (ly1 - ly0)],
            ta_t: vec![0.0; rw * (ly1 - ly0)],
            tb_t: vec![0.0; rw * (ly1 - ly0)],
            h4: z.clone(),
            h10: z.clone(),
            tmp: z.clone(),
            g4: z.clone(),
            g10: z,
            colsum: vec![0.0; rw.max(ly1 - ly0)],
            // per-row dither offsets for the ghost (w 0.6) and glow (w 0.5) quantisers
            dg: (0..8 * w).map(|i| 0.5 + (B8[i / w][i % w & 7] - 0.5) * 0.6).collect(),
            dl: (0..8 * w).map(|i| 0.5 + (B8[i / w][i % w & 7] - 0.5) * 0.5).collect(),
            glrow: vec![0.0; w],
            rng: r,
        }
    }
}

impl Scene for Lava {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h, cs) = (self.w, self.h, self.cs);
        if w == 0 || h == 0 {
            return;
        }
        // metaballs: (x, y, y-scale, strength)
        let mut balls = [(0.0f32, 0.0f32, 1.0f32, 0.0f32); NB + 1];
        let mut nb = NB;
        for i in 0..NB {
            let x = self.bx[i] + (t * 0.21 + self.bp1[i]).sin() * 5.0 * cs;
            let y = self.lcy + (t * self.bsp[i] + self.bp2[i]).sin() * self.lh * 0.7;
            let rr = self.br[i] * (1.0 + 0.1 * (t * 0.7 + self.bp1[i] * 3.0).sin());
            balls[i] = (x, y, 1.0, rr * rr * 0.62);
        }
        // every 20 s one big slow blob of heat crosses the whole word
        let sp = (t % 20.0) / 7.0;
        if sp < 1.25 {
            let rr = 11.0 * cs;
            balls[NB] = (self.lx - 12.0 * cs + sp * (self.lw + 24.0 * cs), self.lcy, 0.8, rr * rr * 0.8);
            nb += 1;
        }
        for y in 0..h {
            let row = &mut self.fb[y * w..y * w + w];
            row.iter_mut().for_each(|v| *v = 0.0);
            for &(bx, by, ys, c) in &balls[..nb] {
                let dy = (y as f32 - by) * ys;
                let e = dy * dy + 2.0;
                for (v, &x) in row.iter_mut().zip(&self.xf) {
                    let dx = x - bx;
                    *v += c / (dx * dx + e);
                }
            }
        }
        // letter fill levels -> glow luminance
        for p in &self.px {
            let f = self.fb[p.i as usize] + p.f0;
            let k = (f > 0.42) as u8 + (f > 0.68) as u8 + (f > 1.05) as u8 + (f > 1.7) as u8;
            self.idx[p.i as usize] = k;
            self.lum_t[p.ti as usize] = k as f32 * 0.25;
        }
        // glow = clip(tent(lum, 4cs) * 2.2 + tent(lum, 10cs) * 1.6)
        let (rw, rh) = (self.rw, self.rh);
        let (r4, r10) = (4 * cs as usize, 10 * cs as usize);
        let nr = self.ly1 - self.ly0;
        for (r, hb) in [(r4, &mut self.h4), (r10, &mut self.h10)] {
            vpass(&self.lum_t, &mut self.ta_t, nr, rw, 0, rw, r, &mut self.colsum[..nr]);
            vpass(&self.ta_t, &mut self.tb_t, nr, rw, 0, rw, r, &mut self.colsum[..nr]);
            for (x, col) in self.tb_t.chunks_exact(nr.max(1)).enumerate() {
                for (j, &v) in col.iter().enumerate() {
                    hb[(self.ly0 + j) * rw + x] = v;
                }
            }
        }
        let colsum = &mut self.colsum[..rw];
        let ext = |r: usize| (self.ly0.saturating_sub(r), (self.ly1 + r).min(rh));
        let (a0, a1) = ext(r4);
        let (b0, b1) = ext(2 * r4);
        vpass(&self.h4, &mut self.tmp, rw, rh, a0, a1, r4, colsum);
        vpass(&self.tmp, &mut self.g4, rw, rh, b0, b1, r4, colsum);
        let (a0, a1) = ext(r10);
        let (c0, c1) = ext(2 * r10);
        vpass(&self.h10, &mut self.tmp, rw, rh, a0, a1, r10, colsum);
        vpass(&self.tmp, &mut self.g10, rw, rh, c0, c1, r10, colsum);

        // background: base + ghost of the blobs + quantised glow, shadowed, vignetted, scanlined
        for y in 0..h {
            let (dg, dl) = (&self.dg[(y & 7) * w..(y & 7) * w + w], &self.dl[(y & 7) * w..(y & 7) * w + w]);
            let gl = &mut self.glrow;
            gl.iter_mut().for_each(|v| *v = 0.0);
            if y >= self.ry0 && y < self.ry0 + rh && rw > 0 {
                let ly = y - self.ry0;
                let (ga, gb) = (&self.g4[ly * rw..ly * rw + rw], &self.g10[ly * rw..ly * rw + rw]);
                let (gl, dl) = (&mut gl[self.rx0..self.rx0 + rw], &dl[self.rx0..self.rx0 + rw]);
                for (((o, &a), &b), &d) in gl.iter_mut().zip(ga).zip(gb).zip(dl) {
                    let v = (a * 2.2 + b * 1.6).clamp(0.0, 1.0);
                    *o = ((v * 6.0 + d) as i32).min(6) as f32 * (1.0 / 6.0);
                }
            }
            let o = y * w;
            let (fr, vr, kr) = (&self.fb[o..o + w], &self.vg[o..o + w], &self.kbg[o..o + w]);
            let dst = &mut img.px[o..o + w];
            for (((((p, &f), &v), &k), &d), &l) in dst.iter_mut().zip(fr).zip(vr).zip(kr).zip(dg).zip(gl.iter()) {
                let v = (f * (1.0 / 2.2)).min(1.0) * v;
                let g = ((v * 4.0 + d) as i32).min(4) as f32 * 0.25;
                *p = [
                    (BASE[0] + g * GHOST[0] + l * GLOW[0]) * k,
                    (BASE[1] + g * GHOST[1] + l * GLOW[1]) * k,
                    (BASE[2] + g * GHOST[2] + l * GLOW[2]) * k,
                ];
            }
        }
        for &i in &self.rim {
            img.px[i as usize] = scale([4.0, 1.0, 6.0], self.kcrt[i as usize]);
        }
        let raw = |idx: u8, top: bool| {
            let c = PAL[idx as usize];
            if top {
                [(c[0] + 44.0).min(255.0), (c[1] + 30.0).min(255.0), (c[2] + 24.0).min(255.0)]
            } else {
                c
            }
        };
        for p in &self.px {
            let i = p.i as usize;
            img.px[i] = scale(raw(self.idx[i], p.top), self.kcrt[i]);
        }

        // bubbles rise through the bottom of the letters
        let r = &mut self.rng;
        if r.f() < dt * 2.0 && !self.bubble_src.is_empty() {
            let (x, y) = self.bubble_src[r.below(self.bubble_src.len())];
            let vy = -r.range(3.0, 7.0) * cs;
            let life = r.range(1.2, 2.6);
            self.bub.add(x, y, 0.0, vy, [255.0, 244.0, 200.0], life);
        }
        self.bub.update(dt, 0.0, 0.0);
        for b in &self.bub.p {
            let bx = (b.x.round() as i32).clamp(0, w as i32 - 1) as usize;
            let by = (b.y.round() as i32).clamp(0, h as i32 - 1) as usize;
            let i = by * w + bx;
            if self.is_l[i] {
                let c = raw(self.idx[i], self.top_at(i));
                img.px[i] = scale([(c[0] + 60.0).min(255.0), (c[1] + 60.0).min(255.0), (c[2] + 60.0).min(255.0)], self.kcrt[i]);
            }
        }
    }
}

impl Lava {
    fn top_at(&self, i: usize) -> bool {
        let (w, y) = (self.w, i / self.w);
        y == 0 || !self.is_l[i - w]
    }
}
