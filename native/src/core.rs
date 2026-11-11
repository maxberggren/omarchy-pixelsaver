//! Shared building blocks: pixel buffers, masks, blurs, dithering, palettes, logo, RNG, particles.
//! Mirrors the helpers in pixelsaver.py (the Python reference implementation).
#![allow(dead_code)]

use crate::font::FONT;

pub type Rgb = [f32; 3];

// --------------------------------------------------------------------------- buffers

/// Float RGB image (values nominally 0..255, may over/undershoot; clamped on output).
#[derive(Clone)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Rgb>,
}

impl Canvas {
    pub fn new(w: usize, h: usize, c: Rgb) -> Self {
        Canvas { w, h, px: vec![c; w * h] }
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }
    pub fn fill(&mut self, c: Rgb) {
        self.px.iter_mut().for_each(|p| *p = c);
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32) -> Option<Rgb> {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            Some(self.px[y as usize * self.w + x as usize])
        } else {
            None
        }
    }
    #[inline]
    pub fn set(&mut self, x: i32, y: i32, c: Rgb) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = y as usize * self.w + x as usize;
            self.px[i] = c;
        }
    }
    #[inline]
    pub fn add(&mut self, x: i32, y: i32, c: Rgb) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let p = &mut self.px[y as usize * self.w + x as usize];
            p[0] += c[0];
            p[1] += c[1];
            p[2] += c[2];
        }
    }
    #[inline]
    pub fn max(&mut self, x: i32, y: i32, c: Rgb) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let p = &mut self.px[y as usize * self.w + x as usize];
            p[0] = p[0].max(c[0]);
            p[1] = p[1].max(c[1]);
            p[2] = p[2].max(c[2]);
        }
    }
    /// Clamp to 0..255 and write 8-bit RGB.
    pub fn to_u8(&self, out: &mut [[u8; 3]]) {
        for (o, p) in out.iter_mut().zip(self.px.iter()) {
            *o = [clamp8(p[0]), clamp8(p[1]), clamp8(p[2])];
        }
    }
}

#[inline]
pub fn clamp8(v: f32) -> u8 {
    // numpy's astype(uint8) after clip truncates toward zero
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v as u8
    }
}

/// Boolean mask.
#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub d: Vec<bool>,
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Self {
        Mask { w, h, d: vec![false; w * h] }
    }
    #[inline]
    pub fn at(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.d[y as usize * self.w + x as usize]
    }
    #[inline]
    pub fn put(&mut self, x: i32, y: i32, v: bool) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            self.d[y as usize * self.w + x as usize] = v;
        }
    }
    pub fn count(&self) -> usize {
        self.d.iter().filter(|&&b| b).count()
    }
    pub fn and(&self, o: &Mask) -> Mask {
        Mask { w: self.w, h: self.h, d: self.d.iter().zip(&o.d).map(|(a, b)| *a && *b).collect() }
    }
    pub fn or(&self, o: &Mask) -> Mask {
        Mask { w: self.w, h: self.h, d: self.d.iter().zip(&o.d).map(|(a, b)| *a || *b).collect() }
    }
    pub fn and_not(&self, o: &Mask) -> Mask {
        Mask { w: self.w, h: self.h, d: self.d.iter().zip(&o.d).map(|(a, b)| *a && !*b).collect() }
    }
    pub fn not(&self) -> Mask {
        Mask { w: self.w, h: self.h, d: self.d.iter().map(|a| !*a).collect() }
    }
    /// numpy `shift(a, dy, dx)`: out[y+dy, x+dx] = a[y, x], zero fill.
    pub fn shift(&self, dy: i32, dx: i32) -> Mask {
        let mut o = Mask::new(self.w, self.h);
        let (w, h) = (self.w as i32, self.h as i32);
        for y in 0..h {
            let sy = y - dy;
            if sy < 0 || sy >= h {
                continue;
            }
            for x in 0..w {
                let sx = x - dx;
                if sx >= 0 && sx < w {
                    o.d[(y * w + x) as usize] = self.d[(sy * w + sx) as usize];
                }
            }
        }
        o
    }
    /// `dilate(m, r, diag)`: r passes of a 3x3 (or plus-shaped) max filter.
    pub fn dilate(&self, r: usize, diag: bool) -> Mask {
        let mut cur = self.clone();
        let (w, h) = (self.w, self.h);
        for _ in 0..r {
            let src = cur.clone();
            for y in 0..h {
                for x in 0..w {
                    if src.d[y * w + x] {
                        continue;
                    }
                    let mut v = false;
                    'n: for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if (dy != 0 || dx != 0) && (diag || dy == 0 || dx == 0) && src.at(x as i32 + dx, y as i32 + dy) {
                                v = true;
                                break 'n;
                            }
                        }
                    }
                    cur.d[y * w + x] = v;
                }
            }
        }
        cur
    }
    /// Pixels of the mask whose neighbour at (dx,dy) is not set: `L & ~shift(L, -dy, -dx)`.
    /// edge(0,-1) = top edge (no pixel above), edge(0,1) = bottom, edge(-1,0) = left, edge(1,0) = right.
    pub fn edge(&self, dx: i32, dy: i32) -> Mask {
        let mut o = Mask::new(self.w, self.h);
        for y in 0..self.h as i32 {
            for x in 0..self.w as i32 {
                if self.at(x, y) && !self.at(x + dx, y + dy) {
                    o.d[y as usize * self.w + x as usize] = true;
                }
            }
        }
        o
    }
    /// Bounding box (x0, y0, x1, y1) exclusive, or None if empty.
    pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for y in 0..self.h {
            for x in 0..self.w {
                if self.d[y * self.w + x] {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x1 == 0 {
            None
        } else {
            Some((x0, y0, x1, y1))
        }
    }
    pub fn to_grid(&self) -> Grid {
        Grid { w: self.w, h: self.h, d: self.d.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect() }
    }
}

/// Scalar float field.
#[derive(Clone)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Grid {
    pub fn new(w: usize, h: usize) -> Self {
        Grid { w, h, d: vec![0.0; w * h] }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }
}

// --------------------------------------------------------------------------- blurs

/// numpy `box_blur(a, r)` on one axis of a strided buffer: window [i-r, i+r], edges clamped.
fn blur_line(src: &[f32], dst: &mut [f32], n: usize, stride: usize, r: usize) {
    let k = (2 * r + 1) as f32;
    let at = |i: isize| src[(i.clamp(0, n as isize - 1) as usize) * stride];
    let mut s = 0.0f32;
    for j in -(r as isize)..=(r as isize) {
        s += at(j);
    }
    for i in 0..n as isize {
        dst[i as usize * stride] = s / k;
        s += at(i + r as isize + 1) - at(i - r as isize);
    }
}

/// Separable box blur (radius r) of a scalar field, edges clamped, like pixelsaver.box_blur.
pub fn box_blur(g: &Grid, r: usize) -> Grid {
    if r == 0 {
        return g.clone();
    }
    let (w, h) = (g.w, g.h);
    let mut tmp = vec![0.0; w * h];
    let mut out = vec![0.0; w * h];
    // axis 0 (vertical) first, then horizontal, as in the Python version
    for x in 0..w {
        blur_line(&g.d[x..], &mut tmp[x..], h, w, r);
    }
    for y in 0..h {
        blur_line(&tmp[y * w..], &mut out[y * w..], w, 1, r);
    }
    Grid { w, h, d: out }
}

/// Horizontal-only box blur (pixelsaver.hblur).
pub fn hblur(g: &Grid, r: usize) -> Grid {
    if r == 0 {
        return g.clone();
    }
    let (w, h) = (g.w, g.h);
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        blur_line(&g.d[y * w..], &mut out[y * w..], w, 1, r);
    }
    Grid { w, h, d: out }
}

/// Box blur of an RGB canvas.
pub fn box_blur_rgb(c: &Canvas, r: usize) -> Canvas {
    let mut out = c.clone();
    for ch in 0..3 {
        let g = Grid { w: c.w, h: c.h, d: c.px.iter().map(|p| p[ch]).collect() };
        let b = box_blur(&g, r);
        for (o, v) in out.px.iter_mut().zip(b.d) {
            o[ch] = v;
        }
    }
    out
}

// --------------------------------------------------------------------------- maths

#[inline]
pub fn smoothstep(x: f32, a: f32, b: f32) -> f32 {
    let x = ((x - a) / (b - a)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[inline]
pub fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

#[inline]
pub fn scale(a: Rgb, k: f32) -> Rgb {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
pub fn add(a: Rgb, b: Rgb) -> Rgb {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub const B4: [[f32; 4]; 4] = [[0., 8., 2., 10.], [12., 4., 14., 6.], [3., 11., 1., 9.], [15., 7., 13., 5.]];

const fn make_b8() -> [[f32; 8]; 8] {
    // np.block([[4b+0, 4b+2], [4b+3, 4b+1]]) / 64
    let mut m = [[0.0f32; 8]; 8];
    let mut y = 0;
    while y < 8 {
        let mut x = 0;
        while x < 8 {
            let q = match (y >= 4, x >= 4) {
                (false, false) => 0.0,
                (false, true) => 2.0,
                (true, false) => 3.0,
                (true, true) => 1.0,
            };
            m[y][x] = (4.0 * B4[y % 4][x % 4] + q) / 64.0;
            x += 1;
        }
        y += 1;
    }
    m
}

/// 8x8 ordered-dither threshold, values in [0, 1).
pub const B8: [[f32; 8]; 8] = make_b8();

#[inline]
pub fn bay(x: usize, y: usize) -> f32 {
    B8[y & 7][x & 7]
}

/// 4x4 Bayer scaled 2x (glitch's `tile(kron(B4/16, ones(2,2)))`).
#[inline]
pub fn bay4x2(x: usize, y: usize) -> f32 {
    B4[(y >> 1) & 3][(x >> 1) & 3] / 16.0
}

/// Quantise v (0..1) to n steps with ordered dither; returns 0..1 (pixelsaver.qd).
#[inline]
pub fn qd(v: f32, n: f32, b: f32, w: f32) -> f32 {
    ((v * n + 0.5 + (b - 0.5) * w).floor().clamp(0.0, n)) / n
}

/// Quantise to an index 0..n-1 (pixelsaver.dith).
#[inline]
pub fn dith(v: f32, n: usize, b: f32, w: f32) -> usize {
    (v * (n as f32 - 1.0) + 0.5 + (b - 0.5) * w).floor().clamp(0.0, n as f32 - 1.0) as usize
}

/// Piecewise-linear palette of n entries (pixelsaver.grad).
pub fn grad(stops: &[(f32, Rgb)], n: usize) -> Vec<Rgb> {
    (0..n)
        .map(|i| {
            let x = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            if x <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                let (a, b) = (w[0], w[1]);
                if x <= b.0 {
                    let t = if b.0 > a.0 { (x - a.0) / (b.0 - a.0) } else { 0.0 };
                    return mix(a.1, b.1, t);
                }
            }
            stops[stops.len() - 1].1
        })
        .collect()
}

/// h, s, v in 0..1 -> RGB 0..255.
pub fn hsv2rgb(h: f32, s: f32, v: f32) -> Rgb {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let i = h6.floor() as i32 % 6;
    let f = h6 - h6.floor();
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r * 255.0, g * 255.0, b * 255.0]
}

/// Filled disc, clipped (pixelsaver.disc).
pub fn disc(img: &mut Canvas, cx: f32, cy: f32, r: f32, col: Rgb) {
    let rr = ((r * 4.0).round() / 4.0).max(0.5);
    let ri = rr.ceil() as i32;
    let (x0, y0) = (cx.round() as i32, cy.round() as i32);
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if ((dx * dx + dy * dy) as f32) <= rr * rr + 0.3 {
                img.set(x0 + dx, y0 + dy, col);
            }
        }
    }
}

// --------------------------------------------------------------------------- RNG

/// Small fast PRNG (xoshiro128+-ish / splitmix64 seeded).
#[derive(Clone)]
pub struct Rng(u64, u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng(next() | 1, next() | 1)
    }
    #[inline]
    pub fn u64(&mut self) -> u64 {
        // xorshift128+
        let mut s1 = self.0;
        let s0 = self.1;
        self.0 = s0;
        s1 ^= s1 << 23;
        self.1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        self.1.wrapping_add(s0)
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn f(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    /// Integer in [a, b).
    #[inline]
    pub fn int(&mut self, a: i64, b: i64) -> i64 {
        if b <= a {
            return a;
        }
        a + (self.u64() % (b - a) as u64) as i64
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.u64() % n as u64) as usize
        }
    }
    pub fn normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let u1 = self.f().max(1e-7);
        let u2 = self.f();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    pub fn sign(&mut self) -> f32 {
        if self.u64() & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// --------------------------------------------------------------------------- logo

/// The logo bitmap (cropped to content).
#[derive(Clone)]
pub struct Logo {
    pub w: usize,
    pub h: usize,
    pub d: Vec<bool>,
}

pub fn load_logo(path: &str) -> Logo {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|_| "PIXEL\nSAVER".to_string());
    let lines: Vec<Vec<char>> = txt.split('\n').map(|l| l.chars().collect()).collect();
    let w = lines.iter().map(|l| l.len()).max().unwrap_or(1).max(1);
    let h = lines.len() * 2;
    let mut m = vec![false; w * h];
    for (r, line) in lines.iter().enumerate() {
        for (c, &ch) in line.iter().enumerate() {
            if ch.is_whitespace() {
                continue;
            }
            let (top, bot) = match ch {
                '▀' | '▘' | '▝' => (true, false),
                '▄' | '▖' | '▗' => (false, true),
                _ => (true, true),
            };
            m[2 * r * w + c] = top;
            m[(2 * r + 1) * w + c] = bot;
        }
    }
    let full = Mask { w, h, d: m };
    match full.bbox() {
        None => {
            let t = text_mask("PIXELSAVER", 2);
            Logo { w: t.w, h: t.h, d: t.d }
        }
        Some((x0, y0, x1, y1)) => {
            let (lw, lh) = (x1 - x0, y1 - y0);
            let mut d = vec![false; lw * lh];
            for y in 0..lh {
                for x in 0..lw {
                    d[y * lw + x] = full.d[(y + y0) * w + x + x0];
                }
            }
            Logo { w: lw, h: lh, d }
        }
    }
}

/// Where the (scaled) logo sits on screen. Mirrors Scene.place().
#[derive(Clone)]
pub struct Placed {
    pub s: usize,
    pub l: Mask,
    pub lx: i32,
    pub ly: i32,
    pub lw: i32,
    pub lh: i32,
    pub lcx: f32,
    pub lcy: f32,
}

pub fn place(logo: &Logo, w: usize, h: usize, maxw: f32, maxh: f32, cy: f32) -> Placed {
    let s = ((w as f32 * maxw / logo.w as f32).min(h as f32 * maxh / logo.h as f32) as usize).max(1);
    let (mw, mh) = (logo.w * s, logo.h * s);
    let x = (w as i32 - mw as i32).div_euclid(2);
    let y = (h as f32 * cy - mh as f32 / 2.0) as i32; // int() truncates toward zero
    let mut l = Mask::new(w, h);
    for j in 0..mh {
        for i in 0..mw {
            if logo.d[(j / s) * logo.w + i / s] {
                l.put(x + i as i32, y + j as i32, true);
            }
        }
    }
    Placed { s, l, lx: x, ly: y, lw: mw as i32, lh: mh as i32, lcx: x as f32 + mw as f32 / 2.0, lcy: y as f32 + mh as f32 / 2.0 }
}

/// Label logo columns by letter (1..n), 0 for gaps (pixelsaver.letter_ids). Returns (ids per column, n).
pub fn letter_ids(l: &Mask) -> (Vec<usize>, usize) {
    let mut ids = vec![0usize; l.w];
    let (mut k, mut prev) = (0, false);
    for x in 0..l.w {
        let c = (0..l.h).any(|y| l.d[y * l.w + x]);
        if c && !prev {
            k += 1;
        }
        ids[x] = if c { k } else { 0 };
        prev = c;
    }
    (ids, k)
}

// --------------------------------------------------------------------------- 3x5 font

pub fn glyph(ch: char) -> u16 {
    let c = ch.to_ascii_uppercase();
    FONT.iter().find(|(k, _)| *k == c).or_else(|| FONT.iter().find(|(k, _)| *k == '?')).map(|g| g.1).unwrap_or(0)
}

pub fn text_mask(s: &str, scale: usize) -> Mask {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let w0 = if n == 0 { 0 } else { n * 4 - 1 };
    let mut m = Mask::new(w0 * scale, 5 * scale);
    for (i, &ch) in chars.iter().enumerate() {
        let g = glyph(ch);
        for gy in 0..5 {
            for gx in 0..3 {
                if g & (1 << (14 - (gy * 3 + gx))) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            m.d[(gy * scale + sy) * m.w + (i * 4 + gx) * scale + sx] = true;
                        }
                    }
                }
            }
        }
    }
    m
}

pub fn blit(img: &mut Canvas, m: &Mask, x: i32, y: i32, col: Rgb) {
    for j in 0..m.h {
        for i in 0..m.w {
            if m.d[j * m.w + i] {
                img.set(x + i as i32, y + j as i32, col);
            }
        }
    }
}

/// Returns the text width in pixels.
pub fn draw_text(img: &mut Canvas, s: &str, x: i32, y: i32, col: Rgb, scale: usize, shadow: Option<Rgb>) -> usize {
    let m = text_mask(s, scale);
    if let Some(sc) = shadow {
        blit(img, &m, x + scale as i32, y + scale as i32, sc);
    }
    blit(img, &m, x, y, col);
    m.w
}

// --------------------------------------------------------------------------- particles

pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub c: Rgb,
    pub life: f32,
    pub max: f32,
}

#[derive(Default)]
pub struct Particles {
    pub p: Vec<Particle>,
}

impl Particles {
    pub fn add(&mut self, x: f32, y: f32, vx: f32, vy: f32, c: Rgb, life: f32) {
        self.p.push(Particle { x, y, vx, vy, c, life, max: life });
    }
    pub fn update(&mut self, dt: f32, g: f32, drag: f32) {
        let k = if drag != 0.0 { (-drag * dt).exp() } else { 1.0 };
        for q in self.p.iter_mut() {
            q.vy += g * dt;
            q.vx *= k;
            q.vy *= k;
            q.x += q.vx * dt;
            q.y += q.vy * dt;
            q.life -= dt;
        }
        self.p.retain(|q| q.life > 0.0);
    }
    pub fn draw(&self, img: &mut Canvas, additive: bool) {
        for q in &self.p {
            let (ix, iy) = (q.x.round() as i32, q.y.round() as i32);
            if ix < 0 || iy < 0 || ix as usize >= img.w || iy as usize >= img.h {
                continue;
            }
            let a = (q.life / q.max).clamp(0.0, 1.0);
            let i = iy as usize * img.w + ix as usize;
            let p = &mut img.px[i];
            for ch in 0..3 {
                p[ch] = if additive { p[ch] + q.c[ch] * a } else { p[ch] * (1.0 - a) + q.c[ch] * a };
            }
        }
    }
}

// --------------------------------------------------------------------------- post

/// pixelsaver.crt: darken odd rows by `scan`, then a radial vignette.
pub fn crt(img: &mut Canvas, scan: f32, vig: f32) {
    let (w, h) = (img.w as f32, img.h as f32);
    for y in 0..img.h {
        let fy = (y as f32 - h / 2.0) / h;
        let sc = if y % 2 == 1 { scan } else { 1.0 };
        for x in 0..img.w {
            let fx = (x as f32 - w / 2.0) / w;
            let k = sc * (1.0 - vig * (fx * fx + fy * fy));
            let p = &mut img.px[y * img.w + x];
            p[0] *= k;
            p[1] *= k;
            p[2] *= k;
        }
    }
}
