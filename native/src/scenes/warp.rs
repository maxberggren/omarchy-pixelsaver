//! HYPERSPACE: stars stream past at lightspeed, the logo assembles from flying particles, sways and shimmers,
//! then explodes in a chromatic shockwave and is sucked back into the tunnel. Port of pixelsaver.py `Warp`.
//!
//! Speed notes: everything that only depends on position is built once in `new()`: the pixels sorted by
//! their distance from the centre (so every ring - tunnel or shockwave - only visits its own annulus,
//! found by binary search), the core-glow colour per pixel and the flash-line falloff per column. The
//! particle glow (`P + blur(P,2)*0.8 + blur(P,5)*0.45`) is one fused running-sum pass over the particles'
//! bounding box with two row accumulators, so there are no full-size temporaries and no per-frame allocation.
use super::Scene;
use crate::core::*;
use std::f32::consts::PI;

const CYCLE: f32 = 12.0;
const NSTARS: usize = 280;

/// One screen pixel in the distance-sorted list: distance from the centre, pixel index, its Bayer threshold.
#[derive(Clone, Copy)]
struct RingPx {
    d: f32,
    i: u32,
    b: f32,
}

pub struct Warp {
    w: usize,
    h: usize,
    t: f32,
    r: Rng,
    s: f32,
    lx: f32,
    ly: f32,
    lw: f32,
    home: Vec<[f32; 2]>,
    /// palette coordinate of each particle (home_x*0.012 + home_y*0.006)
    hx: Vec<f32>,
    start: Vec<[f32; 2]>,
    delay: Vec<f32>,
    pos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,
    col: Vec<Rgb>,
    sx: Vec<f32>,
    sy: Vec<f32>,
    sz: Vec<f32>,
    pal: Vec<Rgb>,
    cyc: i64,
    boom: bool,
    ring: Vec<RingPx>,
    core: Vec<Rgb>,
    flash: Vec<f32>,
    /// per-column vertical sway offset (scratch, recomputed per frame)
    wave: Vec<f32>,
    /// particle layer (zero outside `bb`) and the two running vertical-sum rows of the glow pass
    p: Vec<Rgb>,
    bb: (usize, usize, usize, usize),
    s2: Vec<Rgb>,
    s5: Vec<Rgb>,
}

impl Warp {
    pub fn new(w: usize, h: usize, logo: &Logo, seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let pl = place(logo, w, h, 0.8, 0.5, 0.5);
        let mut home = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if pl.l.d[y * w + x] {
                    home.push([x as f32, y as f32]);
                }
            }
        }
        let n = home.len();
        let hx = home.iter().map(|p| p[0] * 0.012 + p[1] * 0.006).collect();
        let sx = (0..NSTARS).map(|_| r.range(-1.0, 1.0)).collect();
        let sy = (0..NSTARS).map(|_| r.range(-1.0, 1.0)).collect();
        let sz = (0..NSTARS).map(|_| r.range(0.05, 1.0)).collect();
        let pal = grad(
            &[
                (0.0, [90.0, 0.0, 255.0]),
                (0.3, [0.0, 170.0, 255.0]),
                (0.5, [255.0, 255.0, 255.0]),
                (0.7, [0.0, 255.0, 200.0]),
                (1.0, [90.0, 0.0, 255.0]),
            ],
            256,
        );
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mut ring = Vec::with_capacity(w * h);
        let mut core = vec![[0.0; 3]; w * h];
        let (k1, k2) = (1.0 / (w as f32 * 0.05), 1.0 / (w as f32 * 0.22));
        for y in 0..h {
            for x in 0..w {
                let d = (x as f32 - cx).hypot(y as f32 - cy);
                let i = y * w + x;
                ring.push(RingPx { d, i: i as u32, b: bay(x, y) });
                let (e1, e2) = ((-d * k1).exp() * 255.0, (-d * k2).exp());
                core[i] = [e1 + e2 * 70.0, e1 + e2 * 140.0, e1 + e2 * 255.0];
            }
        }
        ring.sort_unstable_by(|a, b| a.d.total_cmp(&b.d));
        let flash = (0..w).map(|x| (-(x as f32 - cx).abs() / (w as f32 * 0.3)).exp()).collect();
        Warp {
            w,
            h,
            t: 0.0,
            r,
            s: pl.s as f32,
            lx: pl.lx as f32,
            ly: pl.ly as f32,
            lw: pl.lw as f32,
            home,
            hx,
            start: vec![[0.0; 2]; n],
            delay: vec![0.0; n],
            pos: vec![[0.0; 2]; n],
            vel: vec![[0.0; 2]; n],
            col: vec![[0.0; 3]; n],
            sx,
            sy,
            sz,
            pal,
            cyc: -1,
            boom: false,
            ring,
            core,
            flash,
            wave: vec![0.0; w],
            p: vec![[0.0; 3]; w * h],
            bb: (0, 0, 0, 0),
            s2: vec![[0.0; 3]; w],
            s5: vec![[0.0; 3]; w],
        }
    }

    fn reset(&mut self) {
        let (w, h) = (self.w as f32, self.h as f32);
        let mx = w.max(h);
        for i in 0..self.home.len() {
            let ang = self.r.range(0.0, 2.0 * PI);
            let rad = self.r.range(0.7, 1.3) * mx;
            let (s, c) = ang.sin_cos();
            self.start[i] = [w / 2.0 + c * rad, h / 2.0 + s * rad];
            self.delay[i] = 0.6 + (self.home[i][0] - self.lx) / self.lw.max(1.0) * 1.2 + self.r.range(0.0, 0.3);
            self.pos[i] = self.start[i];
            self.vel[i] = [0.0; 2];
        }
        self.boom = false;
    }

    /// Dithered ring: pixels with `(1 - |d - rad| / wid) * amp > bayer` take max(px, col).
    fn ring(&self, img: &mut Canvas, rad: f32, wid: f32, amp: f32, col: Rgb) {
        if amp <= 0.0 || wid <= 0.0 {
            return;
        }
        let lo = rad - wid;
        let k = amp / wid;
        let a = self.ring.partition_point(|p| p.d <= lo);
        for p in &self.ring[a..] {
            let dd = (p.d - rad).abs();
            if p.d >= rad + wid {
                break;
            }
            if (wid - dd) * k > p.b {
                let q = &mut img.px[p.i as usize];
                q[0] = q[0].max(col[0]);
                q[1] = q[1].max(col[1]);
                q[2] = q[2].max(col[2]);
            }
        }
    }

    /// img += P + box_blur(P, 2) * 0.8 + box_blur(P, 5) * 0.45 over the particles' bounding box, then clear P.
    fn glow(&mut self, img: &mut Canvas) {
        let (x0, y0, x1, y1) = self.bb;
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let (w, h) = (self.w, self.h);
        let (ox0, ox1, oy0, oy1) = (x0.saturating_sub(5), (x1 + 5).min(w), y0.saturating_sub(5), (y1 + 5).min(h));
        let p = &self.p;
        let (s2, s5) = (&mut self.s2, &mut self.s5);
        let row = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
        for x in x0..x1 {
            s2[x] = [0.0; 3];
            s5[x] = [0.0; 3];
        }
        for j in -5isize..=5 {
            let o = row(oy0 as isize + j);
            for x in x0..x1 {
                let v = p[o + x];
                for c in 0..3 {
                    s5[x][c] += v[c];
                    if j.abs() <= 2 {
                        s2[x][c] += v[c];
                    }
                }
            }
        }
        let (k2, k5) = (0.8 / 25.0, 0.45 / 121.0);
        let clampx = |x: isize| x.clamp(0, w as isize - 1) as usize;
        for y in oy0..oy1 {
            if y > oy0 {
                let yi = y as isize;
                let (a2, b2, a5, b5) = (row(yi + 2), row(yi - 3), row(yi + 5), row(yi - 6));
                for x in x0..x1 {
                    for c in 0..3 {
                        s2[x][c] += p[a2 + x][c] - p[b2 + x][c];
                        s5[x][c] += p[a5 + x][c] - p[b5 + x][c];
                    }
                }
            }
            // horizontal running sums; s2/s5 are zero outside [x0, x1)
            let (mut h2, mut h5) = ([0.0f32; 3], [0.0f32; 3]);
            for j in -5isize..=5 {
                let xx = clampx(ox0 as isize + j);
                for c in 0..3 {
                    h5[c] += s5[xx][c];
                    if j.abs() <= 2 {
                        h2[c] += s2[xx][c];
                    }
                }
            }
            let o = y * w;
            for x in ox0..ox1 {
                if x > ox0 {
                    let xi = x as isize;
                    let (a2, b2, a5, b5) = (clampx(xi + 2), clampx(xi - 3), clampx(xi + 5), clampx(xi - 6));
                    for c in 0..3 {
                        h2[c] += s2[a2][c] - s2[b2][c];
                        h5[c] += s5[a5][c] - s5[b5][c];
                    }
                }
                let v = p[o + x];
                let q = &mut img.px[o + x];
                for c in 0..3 {
                    q[c] += v[c] + h2[c] * k2 + h5[c] * k5;
                }
            }
        }
        for y in y0..y1 {
            self.p[y * w + x0..y * w + x1].fill([0.0; 3]);
        }
        for x in x0..x1 {
            self.s2[x] = [0.0; 3];
            self.s5[x] = [0.0; 3];
        }
        self.bb = (0, 0, 0, 0);
    }

    /// Approximate standard normal (Irwin-Hall of 4 uniforms): cheap and visually identical for jitter.
    #[inline]
    fn gauss(r: &mut Rng) -> f32 {
        (r.f() + r.f() + r.f() + r.f() - 2.0) * 1.732_050_8
    }
}

impl Scene for Warp {
    fn draw(&mut self, dt: f32, img: &mut Canvas) {
        self.t += dt;
        let t = self.t;
        let (w, h) = (self.w, self.h);
        let (wf, hf) = (w as f32, h as f32);
        let cyc = (t / CYCLE).floor() as i64;
        let ph = t.rem_euclid(CYCLE);
        if cyc != self.cyc {
            self.cyc = cyc;
            self.reset();
        }
        let (cx, cy) = (wf / 2.0, hf / 2.0);
        let mx = wf.max(hf);
        let speed = 0.2
            + 3.6 * (1.0 - smoothstep(ph, 0.0, 1.8))
            + 0.6 * (smoothstep(ph, 8.0, 9.0) - smoothstep(ph, 9.0, 10.0))
            + 3.6 * smoothstep(ph, 9.8, 11.2);
        img.fill([2.0, 0.0, 10.0]);

        // tunnel rings at lightspeed
        let tk = (1.0 - smoothstep(ph, 0.0, 1.4)).max(smoothstep(ph, 9.6, 10.6));
        if tk > 0.01 {
            for j in 0..7 {
                let f = (t * 1.6 + j as f32 / 7.0).fract();
                let rad = f * f * mx * 0.9;
                let col = hsv2rgb(0.62 + 0.15 * (j as f32 + t).sin(), 0.8, 0.55);
                self.ring(img, rad, 0.8 + rad * 0.05, tk * (0.3 + 0.7 * f), col);
            }
        }

        // stars
        let fl = wf * 0.32;
        let warm = ((speed - 0.2) / 3.0).min(1.0);
        let head = mix([200.0, 220.0, 255.0], [180.0, 255.0, 255.0], warm);
        let tl = mix([120.0, 140.0, 255.0], [255.0, 60.0, 220.0], warm);
        for i in 0..NSTARS {
            self.sz[i] -= speed * dt * 0.6;
            if self.sz[i] < 0.03 {
                self.sx[i] = self.r.range(-1.0, 1.0);
                self.sy[i] = self.r.range(-1.0, 1.0);
                self.sz[i] = 1.0;
            }
            let (sx, sy, sz) = (self.sx[i] * fl, self.sy[i] * fl, self.sz[i]);
            let tail = (sz + speed * 0.07).min(1.4);
            let b0 = (1.15 - sz).clamp(0.0, 1.0);
            if b0 <= 0.0 {
                continue;
            }
            for kk in 0..10 {
                let k = kk as f32 / 9.0;
                let z = sz + (tail - sz) * k;
                let px = (cx + sx / z) as i32;
                let py = (cy + sy / z) as i32;
                img.max(px, py, scale(mix(head, tl, k), b0 * (1.0 - k * 0.85)));
            }
        }

        // logo particles
        let n = self.home.len();
        let s = self.s;
        let tq = t * 0.25;
        for i in 0..n {
            let u = (self.hx[i] - tq).rem_euclid(1.0);
            self.col[i] = self.pal[((u * 255.0) as usize).min(255)];
        }
        let mut bright = 1.0;
        if ph < 9.0 {
            let mid = ph > 3.4 && ph < 8.0;
            let (mut sweep, mut sxs, mut shear, mut sys) = (0.0, 1.0, 0.0, 1.0);
            if mid {
                for x in 0..w {
                    self.wave[x] = ((t * 3.0 - x as f32 * 0.12 / s).sin() * 0.75 * s).round();
                }
                sweep = self.lx + ((ph - 3.4) / 2.3).rem_euclid(1.0) * (self.lw + 30.0) - 15.0;
                let hold = smoothstep(ph, 3.4, 4.2) * (1.0 - smoothstep(ph, 7.4, 8.0));
                sxs = 1.0 + 0.06 * (t * 1.3).sin() * hold;
                shear = (t * 0.9).sin() * 0.35 * hold;
                sys = 1.0 + 0.05 * (t * 1.1).cos() * hold;
            }
            let q = ph - 8.0;
            let jit = q * 0.9 * s;
            for i in 0..n {
                let (hm, st) = (self.home[i], self.start[i]);
                let k = ((ph - self.delay[i]) / 1.5).clamp(0.0, 1.0);
                let e = 1.0 - (1.0 - k) * (1.0 - k) * (1.0 - k);
                let d = [hm[0] - st[0], hm[1] - st[1]];
                let mut pp = [st[0] + d[0] * e, st[1] + d[1] * e];
                if k > 0.0 && k < 1.0 {
                    let sk = (k * PI).sin() * 0.35;
                    pp[0] -= d[1] * sk;
                    pp[1] += d[0] * sk;
                }
                if mid {
                    pp[1] += self.wave[hm[0] as usize];
                    if (hm[0] - (hm[1] - self.ly) * 0.5 - sweep).abs() < 2.5 * s {
                        let c = &mut self.col[i];
                        *c = [c[0] * 0.2 + 204.0, c[1] * 0.2 + 204.0, c[2] * 0.2 + 204.0];
                    }
                    if self.r.f() < 0.015 {
                        self.col[i] = [255.0; 3];
                    }
                    let (rx, ry) = (pp[0] - cx, pp[1] - cy);
                    pp = [cx + rx * sxs + shear * ry, cy + ry * sys];
                }
                if q > 0.0 {
                    pp[0] += Self::gauss(&mut self.r) * jit;
                    pp[1] += Self::gauss(&mut self.r) * jit;
                    let c = &mut self.col[i];
                    *c = [c[0] + (255.0 - c[0]) * q, c[1] + (255.0 - c[1]) * q, c[2] + (255.0 - c[2]) * q];
                }
                self.pos[i] = pp;
            }
        } else {
            if !self.boom {
                self.boom = true;
                let kw = wf / 120.0;
                for i in 0..n {
                    let dv = [self.home[i][0] - cx, self.home[i][1] - cy];
                    let nrm = dv[0].hypot(dv[1]) + 1e-6;
                    let sp = self.r.range(25.0, 130.0) * kw / nrm;
                    self.vel[i] = [dv[0] * sp + self.r.normal(0.0, 10.0), dv[1] * sp + self.r.normal(0.0, 10.0)];
                }
            }
            let damp = (-dt * 0.9).exp();
            let m = ((ph - 9.0) * 2.0).min(1.0);
            for i in 0..n {
                let v = &mut self.vel[i];
                v[0] *= damp;
                v[1] *= damp;
                self.pos[i][0] += v[0] * dt;
                self.pos[i][1] += v[1] * dt;
                let c = &mut self.col[i];
                *c = [c[0] * (1.0 - m) + 255.0 * m, c[1] * (1.0 - m) + 120.0 * m, c[2] * (1.0 - m) + 220.0 * m];
            }
            bright = (1.0 - (ph - 9.0) / 1.6).max(0.0);
            if ph > 9.6 {
                // debris is sucked into the tunnel
                let pull = smoothstep(ph, 9.6, 10.6);
                let (kv, kd) = (1.0 - pull * 0.15, pull * 4.0 * dt * 30.0);
                for i in 0..n {
                    let (p, v) = (self.pos[i], &mut self.vel[i]);
                    v[0] = v[0] * kv + (cx - p[0]) * kd;
                    v[1] = v[1] * kv + (cy - p[1]) * kd;
                }
            }
        }

        // splat particles (with debris streaks after the boom) into P, tracking the bounding box
        if bright > 0.0 && n > 0 {
            let ntr = if ph >= 9.0 { 6 } else { 1 };
            let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, 0, 0);
            for tr in 0..ntr {
                let k = if ntr > 1 { tr as f32 / 5.0 } else { 0.0 };
                let a = (1.0 - k) * bright;
                if a <= 0.0 {
                    continue;
                }
                let kv = 0.06 * k;
                for i in 0..n {
                    let ix = (self.pos[i][0] - self.vel[i][0] * kv).round();
                    let iy = (self.pos[i][1] - self.vel[i][1] * kv).round();
                    if ix < 0.0 || iy < 0.0 || ix >= wf || iy >= hf {
                        continue;
                    }
                    let (ix, iy) = (ix as usize, iy as usize);
                    let c = self.col[i];
                    let q = &mut self.p[iy * w + ix];
                    q[0] = q[0].max(c[0] * a);
                    q[1] = q[1].max(c[1] * a);
                    q[2] = q[2].max(c[2] * a);
                    bx0 = bx0.min(ix);
                    bx1 = bx1.max(ix + 1);
                    by0 = by0.min(iy);
                    by1 = by1.max(iy + 1);
                }
            }
            self.bb = (bx0, by0, bx1, by1);
            self.glow(img);
        }

        // chromatic shockwave
        let q = ph - 9.0;
        if (0.0..1.8).contains(&q) {
            let kc = (-q * 4.0).exp() * 1.4;
            if kc > 0.002 {
                for (o, c) in img.px.iter_mut().zip(&self.core) {
                    o[0] += c[0] * kc;
                    o[1] += c[1] * kc;
                    o[2] += c[2] * kc;
                }
            }
            let rr = q * mx * 0.7;
            let th = 1.2 + q * 3.0;
            let fade = (1.0 - q / 1.8).max(0.0);
            let o3 = 3.0 * s;
            for (off, col) in [(-o3, [255.0, 40.0, 170.0]), (0.0, [255.0, 255.0, 255.0]), (o3, [40.0, 230.0, 255.0])] {
                self.ring(img, rr + off, th, fade, col);
            }
            if q < 1.1 {
                let kf = (-q * 3.0).exp();
                let row = cy as usize;
                for (dy, col) in [(0i32, [260.0, 299.0, 331.5]), (-1, [40.0, 60.0, 127.5]), (1, [40.0, 60.0, 127.5])] {
                    let y = row as i32 + dy;
                    if y < 0 || y >= h as i32 {
                        continue;
                    }
                    let o = y as usize * w;
                    for x in 0..w {
                        let a = self.flash[x] * kf;
                        let p = &mut img.px[o + x];
                        p[0] += col[0] * a;
                        p[1] += col[1] * a;
                        p[2] += col[2] * a;
                    }
                }
            }
        }
    }
}
