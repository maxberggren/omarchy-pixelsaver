//! Cycles scenes with transitions (ordered-dither dissolve, scanline wipe, pixelate), like pixelsaver.Director.
use crate::core::*;
use crate::scenes::{self, Scene};

pub struct Director {
    pub w: usize,
    pub h: usize,
    logo: Logo,
    pub names: Vec<String>,
    pub i: usize,
    k: u64,
    seed: u64,
    duration: f32,
    trans: f32,
    tt: f32,
    cur: Box<dyn Scene>,
    next: Option<Box<dyn Scene>>,
    ca: Canvas,
    cb: Canvas,
    fa: Vec<[u8; 3]>,
    fb: Vec<[u8; 3]>,
    pub frame: Vec<[u8; 3]>,
}

impl Director {
    pub fn new(w: usize, h: usize, logo: &Logo, names: Vec<String>, duration: f32, seed: u64, start: usize) -> Self {
        let i = start % names.len();
        let cur = scenes::make(&names[i], w, h, logo, seed + 1).expect("unknown scene");
        Director {
            w,
            h,
            logo: logo.clone(),
            names,
            i,
            k: 1,
            seed,
            duration,
            trans: 1.1,
            tt: 0.0,
            cur,
            next: None,
            ca: Canvas::new(w, h, [0.0; 3]),
            cb: Canvas::new(w, h, [0.0; 3]),
            fa: vec![[0; 3]; w * h],
            fb: vec![[0; 3]; w * h],
            frame: vec![[0; 3]; w * h],
        }
    }

    fn make(&mut self, i: usize) -> Box<dyn Scene> {
        self.k += 1;
        scenes::make(&self.names[i % self.names.len()], self.w, self.h, &self.logo, self.seed + self.k).unwrap()
    }

    pub fn skip(&mut self, d: i32) {
        let n = self.names.len() as i32;
        self.i = ((self.i as i32 + d).rem_euclid(n)) as usize;
        self.cur = self.make(self.i);
        self.next = None;
        self.tt = 0.0;
    }

    pub fn step(&mut self, dt: f32) {
        self.tt += dt;
        self.cur.draw(dt, &mut self.ca);
        self.ca.to_u8(&mut self.fa);
        if self.names.len() > 1 && self.next.is_none() && self.tt > self.duration {
            self.next = Some(self.make(self.i + 1));
        }
        let Some(nx) = self.next.as_mut() else {
            self.frame.copy_from_slice(&self.fa);
            return;
        };
        nx.draw(dt, &mut self.cb);
        self.cb.to_u8(&mut self.fb);
        let p = (self.tt - self.duration) / self.trans;
        if p >= 1.0 {
            self.i = (self.i + 1) % self.names.len();
            self.cur = self.next.take().unwrap();
            std::mem::swap(&mut self.ca, &mut self.cb);
            self.tt = 0.0;
            self.frame.copy_from_slice(&self.fb);
            return;
        }
        let (w, h) = (self.w, self.h);
        match self.i % 3 {
            0 => {
                for y in 0..h {
                    for x in 0..w {
                        let k = y * w + x;
                        self.frame[k] = if bay(x, y) < p { self.fb[k] } else { self.fa[k] };
                    }
                }
            }
            1 => {
                let edge = p * (h as f32 + 6.0);
                for y in 0..h {
                    let yf = y as f32;
                    for x in 0..w {
                        let k = y * w + x;
                        self.frame[k] = if (yf - edge).abs() < 1.5 {
                            [255; 3]
                        } else if yf < edge {
                            self.fb[k]
                        } else {
                            self.fa[k]
                        };
                    }
                }
            }
            _ => {
                let q = 1.0 - (2.0 * p - 1.0).abs();
                let bs = 1 + (q * 12.0) as usize;
                let src = if p < 0.5 { &self.fa } else { &self.fb };
                for y in 0..h {
                    for x in 0..w {
                        self.frame[y * w + x] = src[(y / bs * bs) * w + x / bs * bs];
                    }
                }
            }
        }
    }
}
